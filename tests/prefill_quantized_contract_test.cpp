#include "q3x/runtime/prefill_quantized_contract.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace checkpoint = q3x::model::checkpoint;
namespace mw = q3x::model::weights;
namespace runtime = q3x::runtime;
namespace st = q3x::io::safetensors;

class TestContext {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

struct SyntheticSource {
  mw::WeightManifest manifest;
  std::vector<runtime::ShardIdentity> shards;
  std::size_t active_shard = 0U;
  std::uint64_t next_offset = 4'096U;
};

[[nodiscard]] std::uint64_t tensor_bytes(
    const st::DType dtype, const std::vector<std::uint64_t>& shape) {
  std::uint64_t elements = 1U;
  for (const std::uint64_t dimension : shape) {
    elements *= dimension;
  }
  return elements * st::bit_width(dtype) / 8U;
}

void add_tensor(SyntheticSource& source, std::string name,
                const st::DType dtype,
                std::vector<std::uint64_t> shape) {
  const std::uint64_t bytes = tensor_bytes(dtype, shape);
  source.next_offset = (source.next_offset + 255U) & ~255ULL;
  while (source.active_shard < source.shards.size() &&
         (source.next_offset > source.shards[source.active_shard].file_size ||
          bytes > source.shards[source.active_shard].file_size -
                      source.next_offset)) {
    ++source.active_shard;
    source.next_offset = 4'096U;
  }
  if (source.active_shard >= source.shards.size()) {
    std::cerr << "synthetic authenticated shards exhausted\n";
    return;
  }
  const runtime::ShardIdentity& shard = source.shards[source.active_shard];
  mw::TensorLocator locator;
  locator.category = mw::TensorCategory::kText;
  locator.shard = shard.filename;
  locator.file = shard.filename;
  locator.file_begin = source.next_offset;
  locator.file_end = source.next_offset + bytes;
  locator.byte_size = bytes;
  locator.dtype = dtype;
  locator.shape = std::move(shape);
  source.manifest.tensors.emplace(std::move(name), std::move(locator));
  source.next_offset += bytes;
}

void add_nvfp4_projection(SyntheticSource& source, const std::string& module,
                          const std::uint64_t output_size,
                          const std::uint64_t input_size) {
  add_tensor(source, module + ".weight", st::DType::kU8,
             {output_size, input_size / 2U});
  add_tensor(source, module + ".weight_scale", st::DType::kF8E4M3,
             {output_size, input_size / 16U});
  add_tensor(source, module + ".weight_scale_2", st::DType::kF32, {});
  add_tensor(source, module + ".input_scale", st::DType::kF32, {});
}

void add_fp8_projection(SyntheticSource& source, const std::string& module,
                        const std::uint64_t output_size,
                        const std::uint64_t input_size) {
  add_tensor(source, module + ".weight", st::DType::kF8E4M3,
             {output_size, input_size});
  add_tensor(source, module + ".weight_scale", st::DType::kF32, {});
  add_tensor(source, module + ".input_scale", st::DType::kF32, {});
}

[[nodiscard]] SyntheticSource make_source() {
  SyntheticSource source;
  for (const checkpoint::KnownCheckpointDescriptor& descriptor :
       checkpoint::known_checkpoint_catalog()) {
    if (descriptor.model == q3x::model::KnownModel::kQwen36_27B) {
      source.manifest.checkpoint = descriptor;
      break;
    }
  }
  source.shards = runtime::pinned_qwen36_27b_shards();

  for (std::uint32_t layer = 0U; layer < 64U; ++layer) {
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer) + ".";
    add_nvfp4_projection(source, prefix + "mlp.gate_proj", 17'408U,
                         5'120U);
    add_nvfp4_projection(source, prefix + "mlp.up_proj", 17'408U,
                         5'120U);
    add_nvfp4_projection(source, prefix + "mlp.down_proj", 5'120U,
                         17'408U);
    if (((layer + 1U) % 4U) != 0U) {
      add_fp8_projection(source, prefix + "linear_attn.in_proj_qkv",
                         10'240U, 5'120U);
      add_fp8_projection(source, prefix + "linear_attn.in_proj_z", 6'144U,
                         5'120U);
      add_fp8_projection(source, prefix + "linear_attn.out_proj", 5'120U,
                         6'144U);
    } else {
      add_fp8_projection(source, prefix + "self_attn.q_proj", 12'288U,
                         5'120U);
      add_fp8_projection(source, prefix + "self_attn.k_proj", 1'024U,
                         5'120U);
      add_fp8_projection(source, prefix + "self_attn.v_proj", 1'024U,
                         5'120U);
      add_fp8_projection(source, prefix + "self_attn.o_proj", 5'120U,
                         6'144U);
    }
  }
  return source;
}

[[nodiscard]] runtime::PrefillSidecarManifestResult build_manifest(
    const SyntheticSource& source, const runtime::PrefillSidecarKind kind) {
  runtime::PrefillSidecarManifestOptions options;
  options.kind = kind;
  return runtime::build_qwen36_27b_prefill_sidecar_manifest(
      source.manifest, source.shards, options);
}

void test_manifest_inventory_and_budgets(TestContext& test,
                                         const SyntheticSource& source) {
  struct Expected {
    runtime::PrefillSidecarKind kind;
    runtime::PrefillSidecarResidencyClass residency;
    std::uint64_t payload;
    std::uint64_t arena;
  };
  const std::array<Expected, 5U> expected = {{
      {runtime::PrefillSidecarKind::kExact,
       runtime::PrefillSidecarResidencyClass::kExact,
       runtime::kPrefillExactSidecarPayloadBytes, 16'840'232'960ULL},
      {runtime::PrefillSidecarKind::kA8Safe,
       runtime::PrefillSidecarResidencyClass::kA8,
       runtime::kPrefillA8SafeSidecarPayloadBytes,
       runtime::kPrefillA8SafeSidecarPayloadBytes},
      {runtime::PrefillSidecarKind::kA8Compact,
       runtime::PrefillSidecarResidencyClass::kA8,
       runtime::kPrefillA8CompactSidecarPayloadBytes, 16'952'901'632ULL},
      {runtime::PrefillSidecarKind::kA4K64,
       runtime::PrefillSidecarResidencyClass::kA4,
       runtime::kPrefillA4K64SidecarPayloadBytes,
       runtime::kPrefillA4K64SidecarPayloadBytes},
      {runtime::PrefillSidecarKind::kA4K128,
       runtime::PrefillSidecarResidencyClass::kA4,
       runtime::kPrefillA4K128SidecarPayloadBytes,
       runtime::kPrefillA4K128SidecarPayloadBytes},
  }};

  for (const Expected& item : expected) {
    const runtime::PrefillSidecarManifestResult built =
        build_manifest(source, item.kind);
    test.expect(built.ok(), "all five sidecar manifests build");
    if (!built) {
      std::cerr << "manifest error: "
                << runtime::to_string(built.diagnostic.code) << " "
                << built.diagnostic.context << " "
                << built.diagnostic.message << '\n';
      continue;
    }
    const runtime::PrefillSidecarManifest& manifest = *built.value;
    test.expect(
        manifest.version_major == 1U && manifest.version_minor == 0U &&
            manifest.kind == item.kind &&
            manifest.residency_class == item.residency &&
            manifest.projections.size() ==
                runtime::kQwen36PrefillProjectionCount &&
            manifest.summary.projection_count == 400U &&
            manifest.summary.mlp_projection_count == 192U &&
            manifest.summary.attention_projection_count == 208U &&
            manifest.summary.logical_weight_elements == 24'326'963'200ULL &&
            manifest.summary.payload_bytes == item.payload &&
            manifest.summary.arena_bytes == item.arena &&
            manifest.manifest_sha256.size() == 64U,
        "manifest version, inventory, digest, and fixed byte budget match");
    const auto& counts = manifest.summary.family_counts;
    test.expect(
        counts[static_cast<std::size_t>(
            runtime::PrefillProjectionFamily::kMlpGate)] == 64U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kMlpUp)] == 64U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kMlpDown)] == 64U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kLinearQkv)] == 48U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kLinearZ)] == 48U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kLinearO)] == 48U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kFullQ)] == 16U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kFullK)] == 16U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kFullV)] == 16U &&
            counts[static_cast<std::size_t>(
                runtime::PrefillProjectionFamily::kFullO)] == 16U,
        "family inventory covers all linear/full attention and MLP shapes");

    bool offsets_valid = true;
    std::uint64_t prior_end = 0U;
    for (const runtime::PrefillProjectionSidecarEntry& entry :
         manifest.projections) {
      offsets_valid = offsets_valid && entry.source_sha256.size() == 64U &&
                      (entry.sidecar_offset %
                       runtime::kPrefillSidecarAlignment) == 0U &&
                      entry.sidecar_offset >= prior_end &&
                      entry.sidecar_byte_size ==
                          entry.weight_bytes + entry.scale_bytes +
                              entry.metadata_bytes;
      prior_end = entry.sidecar_offset + entry.sidecar_byte_size;
    }
    test.expect(offsets_valid && prior_end <= manifest.summary.arena_bytes,
                "all 400 offsets are aligned, disjoint, and source-bound");
    test.expect(runtime::validate_prefill_sidecar_manifest(manifest).ok(),
                "fresh sidecar manifest revalidates");
  }
}

void test_manifest_fail_closed(TestContext& test,
                               const SyntheticSource& source) {
  const auto built =
      build_manifest(source, runtime::PrefillSidecarKind::kA4K64);
  test.expect(built.ok(), "A4 manifest exists for mutation tests");
  if (!built) {
    return;
  }

  runtime::PrefillSidecarManifest mutated = *built.value;
  mutated.projections[1].sidecar_offset += 256U;
  auto diagnostic = runtime::validate_prefill_sidecar_manifest(mutated);
  test.expect(!diagnostic &&
                  diagnostic.code ==
                      runtime::PrefillContractErrorCode::kOffsetMismatch,
              "manifest offset mutation fails closed");

  mutated = *built.value;
  mutated.projections[0].source_sha256.assign(64U, '0');
  diagnostic = runtime::validate_prefill_sidecar_manifest(mutated);
  test.expect(!diagnostic &&
                  diagnostic.code ==
                      runtime::PrefillContractErrorCode::kDigestMismatch,
              "manifest source binding mutation invalidates body digest");

  SyntheticSource missing = source;
  missing.manifest.tensors.erase(
      "model.language_model.layers.0.mlp.gate_proj.weight_scale_2");
  auto rejected =
      build_manifest(missing, runtime::PrefillSidecarKind::kA8Compact);
  test.expect(!rejected &&
                  rejected.diagnostic.code ==
                      runtime::PrefillContractErrorCode::kMissingSourceTensor,
              "missing authenticated source component rejects conversion");

  SyntheticSource wrong_shard = source;
  wrong_shard.shards[0].sha256.assign(64U, 'a');
  rejected = build_manifest(wrong_shard,
                            runtime::PrefillSidecarKind::kA8Safe);
  test.expect(!rejected &&
                  rejected.diagnostic.code == runtime::
                                                  PrefillContractErrorCode::
                                                      kInvalidSourceIdentity,
              "non-pinned shard digest rejects conversion");
}

[[nodiscard]] bool regions_are_disjoint_and_aligned(
    const runtime::PrefillPromptArenaPlan& plan) {
  std::array<runtime::PrefillPromptArenaRegion, 9U> regions = {
      plan.hidden_bf16[0],
      plan.hidden_bf16[1],
      plan.projection_primary_bf16,
      plan.projection_secondary_bf16,
      plan.hidden_quantized,
      plan.hidden_scales_bf16,
      plan.intermediate_quantized,
      plan.intermediate_scales_bf16,
      plan.row_sum_squares_fp32,
  };
  std::uint64_t end = 0U;
  for (const auto& region : regions) {
    if ((region.arena_offset % plan.arena_alignment) != 0U ||
        region.arena_offset < end || region.byte_size == 0U) {
      return false;
    }
    end = region.arena_offset + region.byte_size;
  }
  return end <= plan.arena_bytes;
}

void test_prompt_arena_sizes(TestContext& test) {
  runtime::PrefillPromptArenaOptions options;
  options.prompt_token_count = 512U;
  options.activation = runtime::PrefillPromptActivation::kA8;
  options.activation_scale_group_size = 128U;
  auto built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(built && built.value->arena_bytes == 41'076'736U &&
                  built.value->hidden_bf16[0].byte_size == 5'242'880U &&
                  built.value->hidden_bf16[1].byte_size == 5'242'880U &&
                  built.value->projection_primary_bf16.byte_size ==
                      12'582'912U &&
                  built.value->projection_secondary_bf16.byte_size ==
                      6'291'456U &&
                  built.value->hidden_quantized.byte_size == 2'621'440U &&
                  built.value->hidden_scales_bf16.byte_size == 40'960U &&
                  built.value->attention_output_input_quantized.byte_size ==
                      3'145'728U &&
                  built.value->attention_output_input_scales_bf16.byte_size ==
                      49'152U &&
                  built.value->attention_output_input_quantized.arena_offset ==
                      built.value->intermediate_quantized.arena_offset &&
                  built.value->intermediate_quantized.byte_size ==
                      8'912'896U &&
                  built.value->intermediate_scales_bf16.byte_size ==
                      139'264U &&
                  built.value->row_sum_squares_fp32.byte_size == 2'048U &&
                  built.value->whole_prompt_staging,
              "P512 A8 arena has exact double-slab and staging bytes");
  if (built) {
    test.expect(regions_are_disjoint_and_aligned(*built.value),
                "P512 A8 regions are aligned and disjoint");
  }

  options.prompt_token_count = 4'096U;
  options.activation = runtime::PrefillPromptActivation::kA4;
  options.activation_scale_group_size = 64U;
  built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(built && built.value->arena_bytes == 283'918'336U &&
                  built.value->hidden_bf16[0].byte_size == 41'943'040U &&
                  built.value->projection_primary_bf16.byte_size ==
                      100'663'296U &&
                  built.value->projection_secondary_bf16.byte_size ==
                      50'331'648U &&
                  built.value->hidden_quantized.byte_size == 10'485'760U &&
                  built.value->hidden_scales_bf16.byte_size == 655'360U &&
                  built.value->attention_output_input_quantized.byte_size ==
                      12'582'912U &&
                  built.value->attention_output_input_scales_bf16.byte_size ==
                      786'432U &&
                  built.value->intermediate_quantized.byte_size ==
                      35'651'584U &&
                  built.value->intermediate_scales_bf16.byte_size ==
                      2'228'224U,
              "P4K A4-K64 arena byte budget is exact");
  if (built) {
    test.expect(regions_are_disjoint_and_aligned(*built.value),
                "P4K A4 regions are aligned and disjoint");
  }

  options.prompt_token_count = 40'000U;
  options.activation = runtime::PrefillPromptActivation::kA8;
  options.activation_scale_group_size = 128U;
  options.max_arena_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(built && built.value->arena_bytes == 3'209'120'000ULL &&
                  built.value->hidden_bf16[0].byte_size == 409'600'000U &&
                  built.value->hidden_bf16[1].byte_size == 409'600'000U &&
                  built.value->projection_primary_bf16.byte_size ==
                      983'040'000U &&
                  built.value->projection_secondary_bf16.byte_size ==
                      491'520'000U &&
                  built.value->hidden_quantized.byte_size == 204'800'000U &&
                  built.value->hidden_scales_bf16.byte_size == 3'200'000U &&
                  built.value->attention_output_input_quantized.byte_size ==
                      245'760'000U &&
                  built.value->attention_output_input_scales_bf16.byte_size ==
                      3'840'000U &&
                  built.value->intermediate_quantized.byte_size ==
                      696'320'000U &&
                  built.value->intermediate_scales_bf16.byte_size ==
                      10'880'000U,
              "P40K A8 arena includes 819.2MB BF16 ping-pong plus staging");

  options.activation = runtime::PrefillPromptActivation::kA4;
  options.activation_scale_group_size = 128U;
  built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(built && built.value->arena_bytes == 2'758'560'000ULL &&
                  built.value->hidden_quantized.byte_size == 102'400'000U &&
                  built.value->attention_output_input_quantized.byte_size ==
                      122'880'000U &&
                  built.value->attention_output_input_scales_bf16.byte_size ==
                      3'840'000U &&
                  built.value->intermediate_quantized.byte_size ==
                      348'160'000U &&
                  built.value->hidden_scales_bf16.byte_size == 3'200'000U &&
                  built.value->intermediate_scales_bf16.byte_size ==
                      10'880'000U,
              "P40K A4-K128 packed staging and scale budget is exact");

  options.activation = runtime::PrefillPromptActivation::kA8;
  options.activation_scale_group_size = 128U;
  options.staging_token_capacity = 4'096U;
  built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(built && !built.value->whole_prompt_staging &&
                  built.value->staging_token_capacity == 4'096U &&
                  built.value->hidden_bf16[0].byte_size == 409'600'000U &&
                  built.value->projection_primary_bf16.byte_size ==
                      100'663'296U &&
                  built.value->projection_secondary_bf16.byte_size ==
                      50'331'648U &&
                  built.value->arena_bytes == 1'064'071'424U,
              "P40K S4096 arena keeps full slabs and bounded projections");
}

void test_prompt_arena_fail_closed(TestContext& test) {
  runtime::PrefillPromptArenaOptions options;
  options.prompt_token_count = 40'000U;
  options.activation = runtime::PrefillPromptActivation::kA8;
  options.activation_scale_group_size = 128U;
  options.max_arena_bytes = 3'209'119'999ULL;
  auto built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(!built &&
                  built.diagnostic.code == runtime::
                                               PrefillContractErrorCode::
                                                   kArenaLimitExceeded,
              "P40K A8 arena fails closed one byte below its budget");

  options.max_arena_bytes = std::numeric_limits<std::uint64_t>::max();
  options.prompt_token_count = std::numeric_limits<std::uint64_t>::max();
  built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(!built, "extreme token count fails before arithmetic can wrap");

  options.prompt_token_count = 40'000U;
  options.staging_token_capacity =
      std::numeric_limits<std::uint64_t>::max();
  built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(!built, "extreme staging capacity fails closed");

  options.staging_token_capacity = 0U;
  options.activation_scale_group_size = 64U;
  built = runtime::build_prefill_prompt_arena_plan(options);
  test.expect(!built &&
                  built.diagnostic.code ==
                      runtime::PrefillContractErrorCode::kInvalidOption,
              "A8 with a non-K128 scale ABI is rejected");
}

void test_mutually_exclusive_residency(TestContext& test,
                                       const SyntheticSource& source) {
  const auto exact =
      build_manifest(source, runtime::PrefillSidecarKind::kExact);
  const auto a8 =
      build_manifest(source, runtime::PrefillSidecarKind::kA8Compact);
  const auto a4 =
      build_manifest(source, runtime::PrefillSidecarKind::kA4K128);
  runtime::PrefillPromptArenaOptions arena_options;
  arena_options.prompt_token_count = 40'000U;
  arena_options.activation = runtime::PrefillPromptActivation::kA8;
  arena_options.activation_scale_group_size = 128U;
  arena_options.max_arena_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  const auto a8_arena =
      runtime::build_prefill_prompt_arena_plan(arena_options);
  arena_options.activation = runtime::PrefillPromptActivation::kA4;
  const auto a4_arena =
      runtime::build_prefill_prompt_arena_plan(arena_options);
  test.expect(exact && a8 && a4 && a8_arena && a4_arena,
              "residency fixtures build");
  if (!exact || !a8 || !a4 || !a8_arena || !a4_arena) {
    return;
  }

  runtime::PrefillSidecarResidencyRequest request;
  request.a8 = &*a8.value;
  request.prompt_arena = &*a8_arena.value;
  request.max_total_resident_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
  auto planned = runtime::build_prefill_sidecar_residency_plan(request);
  const std::uint64_t expected_a8_peak =
      a8.value->summary.arena_bytes + a8_arena.value->arena_bytes;
  test.expect(planned && planned.value->sidecar_kind ==
                                 runtime::PrefillSidecarKind::kA8Compact &&
                  planned.value->peak_resident_bytes == expected_a8_peak,
              "one A8 sidecar plus matching P40K arena preflights");

  request.max_total_resident_bytes = expected_a8_peak - 1U;
  planned = runtime::build_prefill_sidecar_residency_plan(request);
  test.expect(!planned &&
                  planned.diagnostic.code == runtime::
                                                  PrefillContractErrorCode::
                                                      kArenaLimitExceeded,
              "combined sidecar/prompt peak fails closed by one byte");

  request = {};
  request.a4 = &*a4.value;
  request.prompt_arena = &*a4_arena.value;
  planned = runtime::build_prefill_sidecar_residency_plan(request);
  test.expect(planned && planned.value->residency_class ==
                                 runtime::PrefillSidecarResidencyClass::kA4,
              "one A4 sidecar plus matching arena preflights");

  request = {};
  request.exact = &*exact.value;
  planned = runtime::build_prefill_sidecar_residency_plan(request);
  test.expect(planned && planned.value->prompt_arena_bytes == 0U,
              "Exact-only residency remains a separate baseline class");

  request = {};
  request.exact = &*exact.value;
  request.a8 = &*a8.value;
  request.prompt_arena = &*a8_arena.value;
  planned = runtime::build_prefill_sidecar_residency_plan(request);
  test.expect(!planned &&
                  planned.diagnostic.code == runtime::
                                                  PrefillContractErrorCode::
                                                      kResidencyConflict,
              "Exact and A8 cannot coexist");

  request = {};
  request.a8 = &*a8.value;
  request.a4 = &*a4.value;
  request.prompt_arena = &*a8_arena.value;
  planned = runtime::build_prefill_sidecar_residency_plan(request);
  test.expect(!planned &&
                  planned.diagnostic.code == runtime::
                                                  PrefillContractErrorCode::
                                                      kResidencyConflict,
              "A8 and A4 cannot coexist");

  request = {};
  request.a8 = &*a8.value;
  request.prompt_arena = &*a4_arena.value;
  planned = runtime::build_prefill_sidecar_residency_plan(request);
  test.expect(!planned &&
                  planned.diagnostic.code == runtime::
                                                  PrefillContractErrorCode::
                                                      kActivationMismatch,
              "sidecar and prompt activation families must match");
}

}  // namespace

int main() {
  TestContext test;
  const SyntheticSource source = make_source();
  test.expect(source.manifest.tensors.size() == 1'392U,
              "synthetic host fixture contains all source components");
  test_manifest_inventory_and_budgets(test, source);
  test_manifest_fail_closed(test, source);
  test_prompt_arena_sizes(test);
  test_prompt_arena_fail_closed(test);
  test_mutually_exclusive_residency(test, source);

  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " prefill quantized contract checks failed\n";
    return 1;
  }
  std::cout << "prefill quantized sidecar and prompt arena contract passed\n";
  return 0;
}
