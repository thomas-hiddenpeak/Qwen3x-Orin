#include "q3x/kernels/sm87_target_aot_projection_fp8_cuda.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace kernels = q3x::kernels;

namespace {

using Role = kernels::Sm87TargetAotProjectionRole;

constexpr std::array<Role, 3U> kFp8Roles{{
    Role::kFp8GdnQkvZ,
    Role::kFp8FullQkv,
    Role::kFp8AttentionOutput,
}};
constexpr std::array<std::uint32_t, 3U> kRawScaleBits{{
    0x0380'0000U,
    0x0381'0000U,
    0x0382'0000U,
}};
constexpr std::array<std::uint16_t, 3U> kCompensatedScaleBits{{
    0x3f80U,
    0x3f81U,
    0x3f82U,
}};
constexpr std::uintptr_t kAllocationBegin = 0x0000'0050'0000'0000ULL;
constexpr std::uintptr_t kPayloadBegin = kAllocationBegin + 4'096U;

static_assert(kernels::kSm87TargetAotFp8CudaMaximumTensorScales == 3U);
static_assert(kernels::sm87_target_aot_fp8_cuda_role(
    Role::kFp8GdnQkvZ));
static_assert(kernels::sm87_target_aot_fp8_cuda_role(
    Role::kFp8FullQkv));
static_assert(kernels::sm87_target_aot_fp8_cuda_role(
    Role::kFp8AttentionOutput));
static_assert(!kernels::sm87_target_aot_fp8_cuda_role(
    Role::kNvFp4GateUp));
static_assert(!kernels::sm87_target_aot_fp8_cuda_role(Role::kInvalid));
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        kRawScaleBits[0U]) == kCompensatedScaleBits[0U]);
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        kRawScaleBits[1U]) == kCompensatedScaleBits[1U]);
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        kRawScaleBits[2U]) == kCompensatedScaleBits[2U]);
// BF16 RNE ties-to-even before the exact 2^120 exponent compensation.
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        0x0380'8000U) == 0x3f80U);
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        0x0381'8000U) == 0x3f82U);
// A BF16 subnormal normalizes exactly after the power-of-two compensation.
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        0x0001'0000U) == 0x3900U);
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(0U) ==
    0U);
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        0x8000'0000U) == 0U);
static_assert(
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        0x7f80'0000U) == 0U);
static_assert(std::is_same_v<
              decltype(kernels::Sm87TargetAotFp8CudaArguments{}.output),
              std::uint16_t*>);

struct TestContext final {
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] kernels::Sm87TargetAotProjectionSha256Digest make_digest(
    const std::uint64_t seed) noexcept {
  kernels::Sm87TargetAotProjectionSha256Digest digest;
  std::uint64_t state = seed | 1U;
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    state ^= state << 13U;
    state ^= state >> 7U;
    state ^= state << 17U;
    digest.bytes[index] = static_cast<std::uint8_t>(state >> 24U);
  }
  // These terminal FP8 codes are admitted weight bytes under the frozen
  // Marlin interpretation. A digest authenticating such a payload is not a
  // block-scale domain and must not be rejected by this schema.
  digest.bytes[0U] = 0x7fU;
  digest.bytes[1U] = 0xffU;
  return digest;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedSourceInventory
make_inventory(const kernels::Sm87TargetAotProjectionPackedLayout& layout,
               const std::uint64_t seed) noexcept {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory;
  inventory.identity = 0x5100'0000'0000'0000ULL + seed;
  inventory.role = layout.role;
  inventory.source_count = layout.partition_count;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    inventory.sources[index] =
        kernels::sm87_target_aot_projection_packed_source_binding(
            layout, index,
            0x6100'0000'0000'0000ULL + seed * 4U + index,
            make_digest(seed * 17U + index * 2U + 1U),
            make_digest(seed * 17U + index * 2U + 2U),
            kRawScaleBits[index]);
  }
  return inventory;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionPackedTransformReceipt
make_transform_receipt(
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const kernels::Sm87TargetAotProjectionPackedSourceInventory& inventory,
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  kernels::Sm87TargetAotProjectionPackedTransformReceipt receipt;
  receipt.artifact_identity = manifest.artifact_identity;
  receipt.source_inventory_identity = inventory.identity;
  receipt.role = layout.role;
  receipt.plan_identity = layout.plan_identity;
  receipt.layout_identity = layout.layout_identity;
  receipt.encoding = layout.encoding;
  receipt.transform_identity = kernels::
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.partition_count = layout.partition_count;
  receipt.payload = {manifest.artifact_identity,
                     manifest.payload_offset,
                     manifest.payload_bytes,
                     manifest.payload_digest,
                     true};
  receipt.deterministic_transform = true;
  receipt.no_arithmetic_conversion = true;
  receipt.no_request_time_repacking = true;
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    const auto& partition = layout.partitions[index];
    const auto& source = inventory.sources[index];
    auto& observed = receipt.partitions[index];
    const std::uint64_t values =
        static_cast<std::uint64_t>(partition.output_features) *
        partition.input_features;
    observed.logical_role = partition.logical_role;
    observed.partition_index = static_cast<std::uint32_t>(index);
    observed.tensor_identity = source.tensor_identity;
    observed.observed_source_weight_digest = source.weight_digest;
    observed.observed_source_scale_digest = source.scale_digest;
    observed.source_weight_bytes_hashed = values;
    observed.source_scale_bytes_hashed = sizeof(std::uint32_t);
    observed.repacked_weight_values = values;
    observed.payload_offset = partition.payload_offset;
    observed.payload_bytes = partition.payload_bytes;
    observed.source_digests_computed_from_tensor_bytes = true;
    observed.canonical_address_bijection_applied = true;
    observed.bit_exact_weight_permutation = true;
    observed.tensor_scale_kept_external = true;
  }
  return receipt;
}

[[nodiscard]] kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt
make_upload_receipt(
    const kernels::Sm87TargetAotProjectionPackedManifest& manifest) noexcept {
  kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt receipt;
  receipt.artifact_identity = manifest.artifact_identity;
  receipt.source_inventory_identity = manifest.source_inventory_identity;
  receipt.role = manifest.role;
  receipt.plan_identity = manifest.plan_identity;
  receipt.layout_identity = manifest.layout_identity;
  receipt.transform_identity = kernels::
      Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.host_payload_offset = manifest.payload_offset;
  receipt.host_payload_bytes = manifest.payload_bytes;
  receipt.host_payload_digest = manifest.payload_digest;
  receipt.host_manifest_seal = manifest.seal;
  receipt.tensor_scale_count = manifest.source_count;
  for (std::size_t index = 0U; index < manifest.source_count; ++index) {
    receipt.tensor_scale_bits[index] =
        manifest.sources[index].tensor_scale_bits;
    receipt.compensated_tensor_scale_bf16_bits[index] =
        kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
            receipt.tensor_scale_bits[index]);
  }
  receipt.device_allocation_identity = 0x7200'0000'0000'0001ULL;
  receipt.device_allocation_owner_identity = 0x7300'0000'0000'0001ULL;
  receipt.device_ordinal = 0;
  receipt.device_allocation_begin = kAllocationBegin;
  receipt.device_allocation_bytes = manifest.payload_bytes + 8'192U;
  receipt.device_allocation_end =
      receipt.device_allocation_begin +
      static_cast<std::uintptr_t>(receipt.device_allocation_bytes);
  receipt.device_payload_begin = kPayloadBegin;
  receipt.device_payload_bytes = manifest.payload_bytes;
  receipt.device_payload_end =
      receipt.device_payload_begin +
      static_cast<std::uintptr_t>(receipt.device_payload_bytes);
  receipt.upload_stream_owner_identity =
      receipt.device_allocation_owner_identity;
  receipt.upload_stream_identity = 0x7400'0000'0000'0001ULL;
  receipt.upload_completion_event_identity = 0x7500'0000'0000'0001ULL;
  receipt.verification_stream_owner_identity =
      receipt.device_allocation_owner_identity;
  receipt.verification_stream_identity = 0x7600'0000'0000'0001ULL;
  receipt.verification_completion_event_identity =
      0x7700'0000'0000'0001ULL;
  receipt.verification_readback_bytes = manifest.payload_bytes;
  receipt.verification_readback_digest = manifest.payload_digest;
  receipt.host_payload_digest_verified_before_copy = true;
  receipt.host_payload_immutable_until_completion = true;
  receipt.copy_enqueued_to_exact_payload_range = true;
  receipt.completion_event_recorded_after_copy = true;
  receipt.completion_event_observed = true;
  receipt.upload_completed = true;
  receipt.verification_copy_enqueued_from_exact_payload_range = true;
  receipt.verification_event_recorded_after_copy = true;
  receipt.verification_event_observed = true;
  receipt.verification_completed = true;
  receipt.device_payload_matches_host_payload = true;
  receipt.allocation_retained_for_asset_lifetime = true;
  receipt.receipt_identity =
      kernels::sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
          receipt);
  return receipt;
}

struct Fixture final {
  kernels::Sm87TargetAotProjectionPackedSourceInventory inventory{};
  kernels::Sm87TargetAotProjectionPackedManifest manifest{};
  kernels::Sm87TargetAotProjectionPackedTransformReceipt transform{};
  kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt upload{};
  kernels::Sm87TargetAotFp8CudaAssetView asset{};
};

[[nodiscard]] Fixture make_fixture(const Role role,
                                   const std::uint64_t seed) noexcept {
  Fixture fixture;
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  fixture.inventory = make_inventory(layout, seed);
  fixture.manifest =
      kernels::sm87_target_aot_projection_make_packed_manifest(
          role, 0x7100'0000'0000'0000ULL + seed, fixture.inventory,
          make_digest(0x8000U + seed));
  fixture.transform = make_transform_receipt(
      layout, fixture.inventory, fixture.manifest);
  fixture.upload = make_upload_receipt(fixture.manifest);
  fixture.asset = kernels::sm87_target_aot_bind_fp8_cuda_asset(
      fixture.manifest, fixture.inventory, fixture.transform,
      fixture.upload);
  return fixture;
}

void test_role_fixture(TestContext& test, const Role role,
                       const std::size_t token_count,
                       const std::uint64_t seed) {
  const Fixture fixture = make_fixture(role, seed);
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  test.expect(fixture.inventory.valid(layout),
              "FP8 source inventory is structurally valid");
  test.expect(
      kernels::sm87_target_aot_projection_validate_transform_receipt(
          fixture.manifest, fixture.inventory, fixture.transform),
      "FP8 transform receipt authenticates the exact source inventory");
  test.expect(
      kernels::
          sm87_target_aot_fp8_cuda_device_upload_receipt_structurally_valid(
              fixture.manifest, fixture.upload),
      "FP8 device upload receipt closes owner/payload/readback intervals");
  test.expect(kernels::sm87_target_aot_fp8_cuda_asset_valid(fixture.asset),
              "FP8 asset view binds an authenticated FP8-only payload");
  test.expect(fixture.asset.tensor_scale_count == layout.partition_count &&
                  fixture.asset.no_request_time_scale_conversion,
              "FP8 asset retains all precomputed compensated scales");
  for (std::size_t index = 0U; index < layout.partition_count; ++index) {
    test.expect(fixture.asset.compensated_tensor_scale_bf16_bits[index] ==
                    kCompensatedScaleBits[index],
                "FP8 compensated scale exactly matches bit-level helper");
  }

  kernels::Sm87TargetAotFp8CudaArguments arguments;
  arguments.role = role;
  arguments.input = reinterpret_cast<const std::uint16_t*>(
      0x0000'0010'0000'0000ULL);
  arguments.asset = fixture.asset;
  arguments.token_count = token_count;
  arguments.output =
      reinterpret_cast<std::uint16_t*>(0x0000'0030'0000'0000ULL);
  test.expect(kernels::sm87_target_aot_fp8_cuda_arguments_valid(arguments),
              "pure FP8 projection arguments admit disjoint BF16 ranges");

  auto invalid_arguments = arguments;
  invalid_arguments.token_count = 192U;
  test.expect(
      !kernels::sm87_target_aot_fp8_cuda_arguments_valid(invalid_arguments),
      "FP8 arguments reject a non-witness token count");
  invalid_arguments = arguments;
  invalid_arguments.output = const_cast<std::uint16_t*>(arguments.input);
  test.expect(
      !kernels::sm87_target_aot_fp8_cuda_arguments_valid(invalid_arguments),
      "FP8 arguments reject input/output aliasing");
  invalid_arguments = arguments;
  invalid_arguments.input = reinterpret_cast<const std::uint16_t*>(
      fixture.asset.payload.begin);
  test.expect(
      !kernels::sm87_target_aot_fp8_cuda_arguments_valid(invalid_arguments),
      "FP8 arguments reject input/persisted-payload aliasing");
  invalid_arguments = arguments;
  invalid_arguments.output = reinterpret_cast<std::uint16_t*>(
      reinterpret_cast<std::uintptr_t>(arguments.output) + 2U);
  test.expect(
      !kernels::sm87_target_aot_fp8_cuda_arguments_valid(invalid_arguments),
      "FP8 arguments reject misaligned BF16 publication");

  kernels::Sm87TargetAotFp8CudaResources uncompiled;
  uncompiled.role = role;
  uncompiled.token_count = token_count;
  test.expect(
      kernels::sm87_target_aot_fp8_cuda_resources_structurally_valid(
          uncompiled),
      "default-off FP8 resource evidence is explicit and unqualified");
  kernels::Sm87TargetAotFp8CudaResources compiled;
  compiled.role = role;
  compiled.token_count = token_count;
  compiled.binary_version = 87;
  compiled.registers_per_thread = 64;
  compiled.static_shared_bytes = 256U;
  compiled.dynamic_shared_bytes =
      kernels::sm87_target_aot_projection_plan(role, token_count)
          .dynamic_shared_bytes;
  compiled.maximum_threads_per_block = 256;
  compiled.active_blocks_per_sm = 1;
  compiled.kernel_compiled = true;
  test.expect(
      kernels::sm87_target_aot_fp8_cuda_resources_structurally_valid(
          compiled),
      "compiled FP8 resource evidence still carries no qualification");
  compiled.production_dispatch_eligible = true;
  test.expect(
      !kernels::sm87_target_aot_fp8_cuda_resources_structurally_valid(
          compiled),
      "resource schema cannot self-issue production eligibility");
}

void test_fail_closed_mutations(TestContext& test) {
  const Fixture fixture = make_fixture(Role::kFp8FullQkv, 2U);
  const auto reseal = [](auto receipt) {
    receipt.receipt_identity =
        kernels::sm87_target_aot_fp8_cuda_compute_upload_receipt_identity(
            receipt);
    return receipt;
  };
  const auto valid = [&fixture](const auto& receipt) {
    return kernels::
        sm87_target_aot_fp8_cuda_device_upload_receipt_structurally_valid(
            fixture.manifest, receipt);
  };

  auto changed = fixture.upload;
  ++changed.device_allocation_owner_identity;
  test.expect(!valid(changed), "receipt identity seals allocation owner");
  changed = fixture.upload;
  --changed.device_allocation_end;
  test.expect(!valid(reseal(changed)),
              "receipt rejects a non-closed allocation interval");
  changed = fixture.upload;
  changed.device_payload_begin = changed.device_allocation_begin - 256U;
  changed.device_payload_end =
      changed.device_payload_begin + changed.device_payload_bytes;
  test.expect(!valid(reseal(changed)),
              "receipt rejects a payload outside its owner allocation");
  changed = fixture.upload;
  changed.host_payload_digest.bytes[3U] ^= 1U;
  test.expect(!valid(reseal(changed)),
              "receipt rejects a substituted host payload digest");
  changed = fixture.upload;
  changed.verification_readback_digest.bytes[4U] ^= 1U;
  test.expect(!valid(reseal(changed)),
              "receipt rejects a substituted readback digest");
  changed = fixture.upload;
  changed.tensor_scale_bits[1U] = kRawScaleBits[0U];
  changed.compensated_tensor_scale_bf16_bits[1U] =
      kCompensatedScaleBits[0U];
  test.expect(!valid(reseal(changed)),
              "receipt rejects raw scale substitution against manifest");
  changed = fixture.upload;
  ++changed.compensated_tensor_scale_bf16_bits[1U];
  test.expect(!valid(reseal(changed)),
              "receipt rejects a forged compensated launch scale");
  changed = fixture.upload;
  changed.upload_completion_event_identity =
      changed.verification_completion_event_identity;
  test.expect(!valid(reseal(changed)),
              "receipt requires distinct upload and readback events");
  changed = fixture.upload;
  changed.verification_completed = false;
  test.expect(!valid(reseal(changed)),
              "receipt rejects incomplete device readback verification");

  auto asset = fixture.asset;
  ++asset.compensated_tensor_scale_bf16_bits[0U];
  test.expect(!kernels::sm87_target_aot_fp8_cuda_asset_valid(asset),
              "asset rejects mutated precomputed launch scale");
  asset = fixture.asset;
  asset.no_request_time_scale_conversion = false;
  test.expect(!kernels::sm87_target_aot_fp8_cuda_asset_valid(asset),
              "asset rejects request-time scale conversion");

  const Fixture nvfp4 = make_fixture(Role::kNvFp4GateUp, 9U);
  test.expect(!nvfp4.asset.valid,
              "FP8 binder rejects an otherwise typed NVFP4 manifest");
}

}  // namespace

int main() {
  TestContext test;
  test_role_fixture(test, Role::kFp8GdnQkvZ, 40'000U, 1U);
  test_role_fixture(test, Role::kFp8FullQkv, 60'000U, 2U);
  test_role_fixture(test, Role::kFp8AttentionOutput, 130'000U, 3U);
  test_fail_closed_mutations(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 target-AOT FP8 CUDA host contract checks passed\n";
  return 0;
}
