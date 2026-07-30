#include "q3x/runtime/prefill_quantized_contract.h"

#include "q3x/core/sha256.h"
#include "q3x/io/safetensors.h"
#include "q3x/model/checkpoint_metadata.h"
#include "q3x/model/model_config.h"

#include <algorithm>
#include <array>
#include <cctype>
#include <limits>
#include <map>
#include <new>
#include <string>
#include <utility>

namespace q3x::runtime {
namespace {

namespace mw = q3x::model::weights;
namespace st = q3x::io::safetensors;

struct ProjectionDescriptor {
  std::uint32_t layer_index = 0U;
  PrefillProjectionFamily family = PrefillProjectionFamily::kMlpGate;
  std::string module;
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
  bool nvfp4_source = false;
};

struct SidecarEncoding {
  PrefillWeightQuantization quantization =
      PrefillWeightQuantization::kExactNvfp4E2m1;
  std::uint32_t scale_group_size = 0U;
  PrefillSidecarLayout layout =
      PrefillSidecarLayout::kExactNvfp4MarlinConsumer;
  std::uint64_t weight_bytes = 0U;
  std::uint64_t scale_bytes = 0U;
  std::uint64_t metadata_bytes = 0U;
};

struct SourceComponent {
  std::string name;
  st::DType dtype = st::DType::kBool;
  std::vector<std::uint64_t> shape;
};

[[nodiscard]] PrefillContractDiagnostic make_diagnostic(
    const PrefillContractErrorCode code, std::string context,
    std::string message, std::string expected = {},
    std::string actual = {}) {
  PrefillContractDiagnostic diagnostic;
  diagnostic.code = code;
  diagnostic.context = std::move(context);
  diagnostic.message = std::move(message);
  diagnostic.expected = std::move(expected);
  diagnostic.actual = std::move(actual);
  return diagnostic;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  output = left + right;
  return true;
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left,
                                    const std::uint64_t right,
                                    std::uint64_t& output) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  output = left * right;
  return true;
}

[[nodiscard]] bool is_power_of_two(const std::uint64_t value) noexcept {
  return value != 0U && (value & (value - 1U)) == 0U;
}

[[nodiscard]] bool checked_align_up(const std::uint64_t value,
                                    const std::uint64_t alignment,
                                    std::uint64_t& output) noexcept {
  if (!is_power_of_two(alignment)) {
    return false;
  }
  const std::uint64_t mask = alignment - 1U;
  if (value > std::numeric_limits<std::uint64_t>::max() - mask) {
    return false;
  }
  output = (value + mask) & ~mask;
  return true;
}

[[nodiscard]] bool lowercase_sha256(const std::string_view digest) noexcept {
  if (digest.size() != 64U) {
    return false;
  }
  return std::all_of(digest.begin(), digest.end(), [](const char character) {
    return (character >= '0' && character <= '9') ||
           (character >= 'a' && character <= 'f');
  });
}

[[nodiscard]] bool valid_sidecar_kind(const PrefillSidecarKind kind) noexcept {
  switch (kind) {
    case PrefillSidecarKind::kExact:
    case PrefillSidecarKind::kA8Safe:
    case PrefillSidecarKind::kA8Compact:
    case PrefillSidecarKind::kA4K64:
    case PrefillSidecarKind::kA4K128:
      return true;
  }
  return false;
}

[[nodiscard]] bool is_full_attention_layer(
    const std::uint32_t layer_index) noexcept {
  return ((layer_index + 1U) % 4U) == 0U;
}

[[nodiscard]] std::vector<ProjectionDescriptor> projection_inventory() {
  std::vector<ProjectionDescriptor> inventory;
  inventory.reserve(kQwen36PrefillProjectionCount);
  for (std::uint32_t layer = 0U; layer < 64U; ++layer) {
    const std::string prefix =
        "model.language_model.layers." + std::to_string(layer) + ".";
    inventory.push_back({layer, PrefillProjectionFamily::kMlpGate,
                         prefix + "mlp.gate_proj", 17'408U, 5'120U, true});
    inventory.push_back({layer, PrefillProjectionFamily::kMlpUp,
                         prefix + "mlp.up_proj", 17'408U, 5'120U, true});
    inventory.push_back({layer, PrefillProjectionFamily::kMlpDown,
                         prefix + "mlp.down_proj", 5'120U, 17'408U, true});
    if (!is_full_attention_layer(layer)) {
      inventory.push_back(
          {layer, PrefillProjectionFamily::kLinearQkv,
           prefix + "linear_attn.in_proj_qkv", 10'240U, 5'120U, false});
      inventory.push_back(
          {layer, PrefillProjectionFamily::kLinearZ,
           prefix + "linear_attn.in_proj_z", 6'144U, 5'120U, false});
      inventory.push_back(
          {layer, PrefillProjectionFamily::kLinearO,
           prefix + "linear_attn.out_proj", 5'120U, 6'144U, false});
    } else {
      inventory.push_back({layer, PrefillProjectionFamily::kFullQ,
                           prefix + "self_attn.q_proj", 12'288U, 5'120U,
                           false});
      inventory.push_back({layer, PrefillProjectionFamily::kFullK,
                           prefix + "self_attn.k_proj", 1'024U, 5'120U,
                           false});
      inventory.push_back({layer, PrefillProjectionFamily::kFullV,
                           prefix + "self_attn.v_proj", 1'024U, 5'120U,
                           false});
      inventory.push_back({layer, PrefillProjectionFamily::kFullO,
                           prefix + "self_attn.o_proj", 5'120U, 6'144U,
                           false});
    }
  }
  return inventory;
}

[[nodiscard]] bool expected_sidecar_encoding(
    const PrefillSidecarKind kind, const ProjectionDescriptor& projection,
    SidecarEncoding& output) noexcept {
  std::uint64_t elements = 0U;
  if (!checked_multiply(projection.output_size, projection.input_size,
                        elements)) {
    return false;
  }

  SidecarEncoding encoding;
  switch (kind) {
    case PrefillSidecarKind::kExact:
      if (projection.nvfp4_source) {
        encoding.quantization =
            PrefillWeightQuantization::kExactNvfp4E2m1;
        encoding.scale_group_size = 16U;
        encoding.layout =
            PrefillSidecarLayout::kExactNvfp4MarlinConsumer;
        encoding.weight_bytes = elements / 2U;
        encoding.scale_bytes = elements / 16U;
        encoding.metadata_bytes = sizeof(float);
      } else {
        encoding.quantization = PrefillWeightQuantization::kExactFp8E4m3;
        encoding.scale_group_size = 0U;
        encoding.layout =
            PrefillSidecarLayout::kExactFp8SupermatrixConsumer;
        encoding.weight_bytes = elements;
        encoding.metadata_bytes = sizeof(float);
      }
      break;
    case PrefillSidecarKind::kA8Safe:
      encoding.quantization = PrefillWeightQuantization::kSymmetricW8;
      encoding.scale_group_size = 128U;
      encoding.layout = PrefillSidecarLayout::kSm87S8K128Consumer;
      encoding.weight_bytes = elements;
      encoding.scale_bytes = (elements / 128U) * sizeof(std::uint16_t);
      break;
    case PrefillSidecarKind::kA8Compact:
      if (projection.nvfp4_source) {
        encoding.quantization = PrefillWeightQuantization::kSymmetricW4;
        encoding.scale_group_size = 32U;
        encoding.layout = PrefillSidecarLayout::kSm87U4B8K32Consumer;
        encoding.weight_bytes = elements / 2U;
        encoding.scale_bytes = (elements / 32U) * sizeof(std::uint16_t);
        encoding.metadata_bytes = sizeof(float);
      } else {
        encoding.quantization = PrefillWeightQuantization::kSymmetricW8;
        encoding.scale_group_size = 128U;
        encoding.layout = PrefillSidecarLayout::kSm87S8K128Consumer;
        encoding.weight_bytes = elements;
        encoding.scale_bytes = (elements / 128U) * sizeof(std::uint16_t);
      }
      break;
    case PrefillSidecarKind::kA4K64:
      encoding.quantization = PrefillWeightQuantization::kSymmetricW4;
      encoding.scale_group_size = 64U;
      encoding.layout = PrefillSidecarLayout::kSm87S4K64Consumer;
      encoding.weight_bytes = elements / 2U;
      encoding.scale_bytes = (elements / 64U) * sizeof(std::uint16_t);
      break;
    case PrefillSidecarKind::kA4K128:
      encoding.quantization = PrefillWeightQuantization::kSymmetricW4;
      encoding.scale_group_size = 128U;
      encoding.layout = PrefillSidecarLayout::kSm87S4K128Consumer;
      encoding.weight_bytes = elements / 2U;
      encoding.scale_bytes = (elements / 128U) * sizeof(std::uint16_t);
      break;
  }
  std::uint64_t bytes = 0U;
  if (projection.input_size != 0U &&
      encoding.scale_group_size != 0U &&
      (projection.input_size % encoding.scale_group_size) != 0U) {
    return false;
  }
  if (!checked_add(encoding.weight_bytes, encoding.scale_bytes, bytes) ||
      !checked_add(bytes, encoding.metadata_bytes, bytes)) {
    return false;
  }
  output = encoding;
  return true;
}

[[nodiscard]] std::uint64_t expected_payload_bytes(
    const PrefillSidecarKind kind) noexcept {
  switch (kind) {
    case PrefillSidecarKind::kExact:
      return kPrefillExactSidecarPayloadBytes;
    case PrefillSidecarKind::kA8Safe:
      return kPrefillA8SafeSidecarPayloadBytes;
    case PrefillSidecarKind::kA8Compact:
      return kPrefillA8CompactSidecarPayloadBytes;
    case PrefillSidecarKind::kA4K64:
      return kPrefillA4K64SidecarPayloadBytes;
    case PrefillSidecarKind::kA4K128:
      return kPrefillA4K128SidecarPayloadBytes;
  }
  return 0U;
}

[[nodiscard]] std::vector<SourceComponent> source_components(
    const ProjectionDescriptor& projection) {
  std::vector<SourceComponent> components;
  components.reserve(projection.nvfp4_source ? 4U : 3U);
  if (projection.nvfp4_source) {
    components.push_back({projection.module + ".weight", st::DType::kU8,
                          {projection.output_size,
                           projection.input_size / 2U}});
    components.push_back(
        {projection.module + ".weight_scale", st::DType::kF8E4M3,
         {projection.output_size, projection.input_size / 16U}});
    components.push_back({projection.module + ".weight_scale_2",
                          st::DType::kF32, {}});
    components.push_back(
        {projection.module + ".input_scale", st::DType::kF32, {}});
  } else {
    components.push_back({projection.module + ".weight",
                          st::DType::kF8E4M3,
                          {projection.output_size, projection.input_size}});
    components.push_back(
        {projection.module + ".weight_scale", st::DType::kF32, {}});
    components.push_back(
        {projection.module + ".input_scale", st::DType::kF32, {}});
  }
  return components;
}

[[nodiscard]] bool expected_tensor_bytes(
    const st::DType dtype, const std::vector<std::uint64_t>& shape,
    std::uint64_t& output) noexcept {
  std::uint64_t elements = 1U;
  for (const std::uint64_t dimension : shape) {
    if (!checked_multiply(elements, dimension, elements)) {
      return false;
    }
  }
  std::uint64_t bits = 0U;
  if (!checked_multiply(elements,
                        static_cast<std::uint64_t>(st::bit_width(dtype)),
                        bits) ||
      (bits % 8U) != 0U) {
    return false;
  }
  output = bits / 8U;
  return true;
}

[[nodiscard]] const model::checkpoint::KnownCheckpointDescriptor*
pinned_dense_checkpoint() noexcept {
  for (const auto& checkpoint : model::checkpoint::known_checkpoint_catalog()) {
    if (checkpoint.model == model::KnownModel::kQwen36_27B) {
      return &checkpoint;
    }
  }
  return nullptr;
}

[[nodiscard]] PrefillContractDiagnostic validate_checkpoint_identity(
    const mw::WeightManifest& manifest,
    const std::vector<ShardIdentity>& authenticated_shards,
    std::map<std::string, const ShardIdentity*, std::less<>>& identities) {
  const auto* const pinned = pinned_dense_checkpoint();
  if (pinned == nullptr || manifest.checkpoint.model != pinned->model ||
      manifest.checkpoint.id != pinned->id ||
      manifest.checkpoint.config_sha256 != pinned->config_sha256 ||
      manifest.checkpoint.quant_config_sha256 !=
          pinned->quant_config_sha256 ||
      manifest.checkpoint.index_sha256 != pinned->index_sha256) {
    return make_diagnostic(
        PrefillContractErrorCode::kUnsupportedCheckpoint, "checkpoint",
        "sidecar conversion requires the pinned Qwen3.6-27B checkpoint");
  }

  const auto& expected = pinned_qwen36_27b_shards();
  if (authenticated_shards.size() != expected.size()) {
    return make_diagnostic(PrefillContractErrorCode::kInvalidSourceIdentity,
                           "authenticated_shards",
                           "authenticated shard count is not pinned",
                           std::to_string(expected.size()),
                           std::to_string(authenticated_shards.size()));
  }
  for (const ShardIdentity& identity : authenticated_shards) {
    if (identity.filename.empty() || identity.file_size == 0U ||
        !lowercase_sha256(identity.sha256) ||
        !identities.emplace(identity.filename, &identity).second) {
      return make_diagnostic(
          PrefillContractErrorCode::kInvalidSourceIdentity,
          identity.filename, "invalid or duplicate authenticated shard");
    }
  }
  for (const ShardIdentity& required : expected) {
    const auto found = identities.find(required.filename);
    if (found == identities.end() ||
        found->second->file_size != required.file_size ||
        found->second->sha256 != required.sha256) {
      return make_diagnostic(
          PrefillContractErrorCode::kInvalidSourceIdentity,
          required.filename,
          "authenticated shard identity differs from the pinned artifact",
          required.sha256,
          found == identities.end() ? "missing" : found->second->sha256);
    }
  }
  return {};
}

[[nodiscard]] bool hash_field(core::Sha256& hash,
                              const std::string_view value) {
  const std::string size = std::to_string(value.size());
  constexpr char kColon = ':';
  constexpr char kNewline = '\n';
  return hash.update(size.data(), size.size()) &&
         hash.update(&kColon, sizeof(kColon)) &&
         hash.update(value.data(), value.size()) &&
         hash.update(&kNewline, sizeof(kNewline));
}

[[nodiscard]] bool hash_u64(core::Sha256& hash,
                            const std::uint64_t value) {
  const std::string text = std::to_string(value);
  return hash_field(hash, text);
}

[[nodiscard]] PrefillContractDiagnostic digest_projection_source(
    const mw::WeightManifest& manifest,
    const std::map<std::string, const ShardIdentity*, std::less<>>& identities,
    const ProjectionDescriptor& projection, std::string& digest) {
  core::Sha256 hash;
  if (!hash_field(hash, projection.module)) {
    return make_diagnostic(PrefillContractErrorCode::kArithmeticOverflow,
                           projection.module,
                           "source digest input length overflowed");
  }
  for (const SourceComponent& component : source_components(projection)) {
    const mw::TensorLocator* const locator = manifest.find(component.name);
    if (locator == nullptr) {
      return make_diagnostic(PrefillContractErrorCode::kMissingSourceTensor,
                             component.name,
                             "projection source component is missing");
    }
    const auto identity = identities.find(locator->shard);
    std::uint64_t expected_bytes = 0U;
    if (identity == identities.end() ||
        locator->category != mw::TensorCategory::kText ||
        locator->dtype != component.dtype ||
        locator->shape != component.shape || locator->file_begin == 0U ||
        locator->file_end <= locator->file_begin ||
        locator->file_end >
            (identity == identities.end() ? 0U
                                          : identity->second->file_size) ||
        locator->byte_size != locator->file_end - locator->file_begin ||
        !expected_tensor_bytes(component.dtype, component.shape,
                               expected_bytes) ||
        locator->byte_size != expected_bytes) {
      return make_diagnostic(
          PrefillContractErrorCode::kSourceTensorMismatch, component.name,
          "projection source component does not match the authenticated ABI");
    }
    const ShardIdentity& shard = *identity->second;
    bool hashed = hash_field(hash, component.name) &&
                  hash_field(hash, shard.filename) &&
                  hash_field(hash, shard.sha256) &&
                  hash_u64(hash, locator->file_begin) &&
                  hash_u64(hash, locator->file_end) &&
                  hash_u64(hash, static_cast<std::uint64_t>(locator->dtype)) &&
                  hash_u64(hash, locator->shape.size());
    for (const std::uint64_t dimension : locator->shape) {
      hashed = hashed && hash_u64(hash, dimension);
    }
    if (!hashed) {
      return make_diagnostic(PrefillContractErrorCode::kArithmeticOverflow,
                             component.name,
                             "source digest input length overflowed");
    }
  }
  digest = hash.finalize().hex();
  return {};
}

[[nodiscard]] bool digest_manifest_body(
    const PrefillSidecarManifest& manifest, std::string& digest) {
  core::Sha256 hash;
  bool hashed =
      hash_u64(hash, manifest.version_major) &&
      hash_u64(hash, manifest.version_minor) &&
      hash_u64(hash, static_cast<std::uint64_t>(manifest.kind)) &&
      hash_u64(hash, static_cast<std::uint64_t>(manifest.residency_class)) &&
      hash_u64(hash, manifest.arena_alignment) &&
      hash_field(hash, manifest.source_checkpoint_id) &&
      hash_field(hash, manifest.source_config_sha256) &&
      hash_field(hash, manifest.source_index_sha256) &&
      hash_u64(hash, manifest.summary.projection_count) &&
      hash_u64(hash, manifest.summary.mlp_projection_count) &&
      hash_u64(hash, manifest.summary.attention_projection_count) &&
      hash_u64(hash, manifest.summary.logical_weight_elements) &&
      hash_u64(hash, manifest.summary.payload_bytes) &&
      hash_u64(hash, manifest.summary.arena_bytes);
  for (const std::size_t count : manifest.summary.family_counts) {
    hashed = hashed && hash_u64(hash, count);
  }
  for (const PrefillProjectionSidecarEntry& entry : manifest.projections) {
    hashed = hashed && hash_u64(hash, entry.ordinal) &&
             hash_u64(hash, entry.layer_index) &&
             hash_u64(hash, static_cast<std::uint64_t>(entry.family)) &&
             hash_field(hash, entry.source_module) &&
             hash_field(hash, entry.source_sha256) &&
             hash_u64(hash, entry.output_size) &&
             hash_u64(hash, entry.input_size) &&
             hash_u64(hash, static_cast<std::uint64_t>(entry.quantization)) &&
             hash_u64(hash, entry.scale_group_size) &&
             hash_u64(hash, static_cast<std::uint64_t>(entry.layout)) &&
             hash_u64(hash, entry.weight_bytes) &&
             hash_u64(hash, entry.scale_bytes) &&
             hash_u64(hash, entry.metadata_bytes) &&
             hash_u64(hash, entry.sidecar_offset) &&
             hash_u64(hash, entry.sidecar_byte_size);
  }
  if (!hashed) {
    return false;
  }
  digest = hash.finalize().hex();
  return true;
}

[[nodiscard]] bool same_region(const PrefillPromptArenaRegion& left,
                               const PrefillPromptArenaRegion& right) noexcept {
  return left.arena_offset == right.arena_offset &&
         left.byte_size == right.byte_size &&
         left.logical_element_capacity == right.logical_element_capacity &&
         left.element_bits == right.element_bits;
}

[[nodiscard]] bool same_prompt_plan(const PrefillPromptArenaPlan& left,
                                    const PrefillPromptArenaPlan& right) noexcept {
  return left.prompt_token_count == right.prompt_token_count &&
         left.activation == right.activation &&
         left.activation_scale_group_size ==
             right.activation_scale_group_size &&
         left.staging_token_capacity == right.staging_token_capacity &&
         left.arena_alignment == right.arena_alignment &&
         left.whole_prompt_staging == right.whole_prompt_staging &&
         same_region(left.hidden_bf16[0], right.hidden_bf16[0]) &&
         same_region(left.hidden_bf16[1], right.hidden_bf16[1]) &&
         same_region(left.hidden_quantized, right.hidden_quantized) &&
         same_region(left.hidden_scales_bf16, right.hidden_scales_bf16) &&
         same_region(left.attention_output_input_quantized,
                     right.attention_output_input_quantized) &&
         same_region(left.attention_output_input_scales_bf16,
                     right.attention_output_input_scales_bf16) &&
         same_region(left.intermediate_quantized,
                     right.intermediate_quantized) &&
         same_region(left.intermediate_scales_bf16,
                     right.intermediate_scales_bf16) &&
         same_region(left.row_sum_squares_fp32,
                     right.row_sum_squares_fp32) &&
         left.arena_bytes == right.arena_bytes;
}

}  // namespace

PrefillSidecarResidencyClass residency_class_for(
    const PrefillSidecarKind kind) noexcept {
  switch (kind) {
    case PrefillSidecarKind::kExact:
      return PrefillSidecarResidencyClass::kExact;
    case PrefillSidecarKind::kA8Safe:
    case PrefillSidecarKind::kA8Compact:
      return PrefillSidecarResidencyClass::kA8;
    case PrefillSidecarKind::kA4K64:
    case PrefillSidecarKind::kA4K128:
      return PrefillSidecarResidencyClass::kA4;
  }
  return PrefillSidecarResidencyClass::kExact;
}

PrefillSidecarManifestResult build_qwen36_27b_prefill_sidecar_manifest(
    const mw::WeightManifest& source_manifest,
    const std::vector<ShardIdentity>& authenticated_shards,
    const PrefillSidecarManifestOptions& options) {
  PrefillSidecarManifestResult result;
  try {
    if (!valid_sidecar_kind(options.kind) ||
        !is_power_of_two(options.arena_alignment) ||
        options.arena_alignment < 16U ||
        options.arena_alignment > (1ULL << 30U)) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kInvalidOption, "manifest_options",
          "sidecar kind or arena alignment is invalid");
      return result;
    }

    std::map<std::string, const ShardIdentity*, std::less<>> identities;
    result.diagnostic = validate_checkpoint_identity(
        source_manifest, authenticated_shards, identities);
    if (!result.diagnostic) {
      return result;
    }

    const std::vector<ProjectionDescriptor> inventory = projection_inventory();
    if (inventory.size() != kQwen36PrefillProjectionCount) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kCountMismatch, "projection_inventory",
          "compiled projection inventory count is invalid");
      return result;
    }

    PrefillSidecarManifest manifest;
    manifest.kind = options.kind;
    manifest.residency_class = residency_class_for(options.kind);
    manifest.arena_alignment = options.arena_alignment;
    manifest.source_checkpoint_id = source_manifest.checkpoint.id;
    manifest.source_config_sha256 =
        source_manifest.checkpoint.config_sha256;
    manifest.source_index_sha256 = source_manifest.checkpoint.index_sha256;
    manifest.projections.reserve(inventory.size());

    std::uint64_t cursor = 0U;
    for (std::size_t index = 0U; index < inventory.size(); ++index) {
      const ProjectionDescriptor& projection = inventory[index];
      SidecarEncoding encoding;
      if (!expected_sidecar_encoding(options.kind, projection, encoding)) {
        result.diagnostic = make_diagnostic(
            PrefillContractErrorCode::kArithmeticOverflow,
            projection.module, "sidecar byte calculation overflowed");
        return result;
      }

      std::string source_sha256;
      result.diagnostic = digest_projection_source(
          source_manifest, identities, projection, source_sha256);
      if (!result.diagnostic) {
        return result;
      }

      std::uint64_t offset = 0U;
      std::uint64_t bytes = 0U;
      std::uint64_t logical_elements = 0U;
      if (!checked_align_up(cursor, options.arena_alignment, offset) ||
          !checked_add(encoding.weight_bytes, encoding.scale_bytes, bytes) ||
          !checked_add(bytes, encoding.metadata_bytes, bytes) ||
          !checked_add(offset, bytes, cursor) ||
          !checked_multiply(projection.output_size, projection.input_size,
                            logical_elements) ||
          !checked_add(manifest.summary.logical_weight_elements,
                       logical_elements,
                       manifest.summary.logical_weight_elements) ||
          !checked_add(manifest.summary.payload_bytes, bytes,
                       manifest.summary.payload_bytes)) {
        result.diagnostic = make_diagnostic(
            PrefillContractErrorCode::kArithmeticOverflow,
            projection.module, "sidecar arena calculation overflowed");
        return result;
      }

      PrefillProjectionSidecarEntry entry;
      entry.ordinal = static_cast<std::uint32_t>(index);
      entry.layer_index = projection.layer_index;
      entry.family = projection.family;
      entry.source_module = projection.module;
      entry.source_sha256 = std::move(source_sha256);
      entry.output_size = projection.output_size;
      entry.input_size = projection.input_size;
      entry.quantization = encoding.quantization;
      entry.scale_group_size = encoding.scale_group_size;
      entry.layout = encoding.layout;
      entry.weight_bytes = encoding.weight_bytes;
      entry.scale_bytes = encoding.scale_bytes;
      entry.metadata_bytes = encoding.metadata_bytes;
      entry.sidecar_offset = offset;
      entry.sidecar_byte_size = bytes;
      manifest.projections.emplace_back(std::move(entry));

      ++manifest.summary.projection_count;
      const std::size_t family_index =
          static_cast<std::size_t>(projection.family);
      ++manifest.summary.family_counts[family_index];
      if (projection.nvfp4_source) {
        ++manifest.summary.mlp_projection_count;
      } else {
        ++manifest.summary.attention_projection_count;
      }
    }
    if (!checked_align_up(cursor, options.arena_alignment,
                          manifest.summary.arena_bytes)) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kArithmeticOverflow, "sidecar_arena",
          "final sidecar arena alignment overflowed");
      return result;
    }
    if (manifest.summary.payload_bytes != expected_payload_bytes(options.kind)) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kSizeMismatch, "sidecar_payload",
          "sidecar payload differs from the fixed model budget",
          std::to_string(expected_payload_bytes(options.kind)),
          std::to_string(manifest.summary.payload_bytes));
      return result;
    }
    if (!digest_manifest_body(manifest, manifest.manifest_sha256)) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kArithmeticOverflow, "manifest_digest",
          "manifest digest input length overflowed");
      return result;
    }
    result.diagnostic = validate_prefill_sidecar_manifest(manifest);
    if (!result.diagnostic) {
      return result;
    }
    result.value.emplace(std::move(manifest));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kAllocationFailure, "sidecar_manifest",
        "allocation failed while building the sidecar manifest");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kInvalidManifest, "sidecar_manifest",
        "unexpected failure while building the sidecar manifest");
    return result;
  }
}

PrefillContractDiagnostic validate_prefill_sidecar_manifest(
    const PrefillSidecarManifest& manifest) {
  try {
    if (manifest.version_major != kPrefillSidecarManifestVersionMajor ||
        manifest.version_minor != kPrefillSidecarManifestVersionMinor) {
      return make_diagnostic(
          PrefillContractErrorCode::kInvalidManifest, "manifest.version",
          "unsupported sidecar manifest version", "1.0",
          std::to_string(manifest.version_major) + "." +
              std::to_string(manifest.version_minor));
    }
    if (!valid_sidecar_kind(manifest.kind) ||
        manifest.residency_class != residency_class_for(manifest.kind) ||
        !is_power_of_two(manifest.arena_alignment) ||
        manifest.arena_alignment < 16U ||
        manifest.arena_alignment > (1ULL << 30U) ||
        !lowercase_sha256(manifest.source_config_sha256) ||
        !lowercase_sha256(manifest.source_index_sha256) ||
        !lowercase_sha256(manifest.manifest_sha256)) {
      return make_diagnostic(PrefillContractErrorCode::kInvalidManifest,
                             "manifest.header",
                             "sidecar manifest header is invalid");
    }
    const auto* const pinned = pinned_dense_checkpoint();
    if (pinned == nullptr || manifest.source_checkpoint_id != pinned->id ||
        manifest.source_config_sha256 != pinned->config_sha256 ||
        manifest.source_index_sha256 != pinned->index_sha256) {
      return make_diagnostic(
          PrefillContractErrorCode::kUnsupportedCheckpoint,
          "manifest.checkpoint",
          "sidecar manifest is not bound to the pinned checkpoint");
    }

    const std::vector<ProjectionDescriptor> inventory = projection_inventory();
    if (manifest.projections.size() != inventory.size() ||
        manifest.summary.projection_count != inventory.size() ||
        manifest.summary.mlp_projection_count !=
            kQwen36PrefillMlpProjectionCount ||
        manifest.summary.attention_projection_count !=
            kQwen36PrefillAttentionProjectionCount) {
      return make_diagnostic(
          PrefillContractErrorCode::kCountMismatch, "manifest.inventory",
          "manifest does not cover all 400 projection matrices");
    }

    PrefillSidecarManifestSummary expected_summary;
    std::uint64_t cursor = 0U;
    for (std::size_t index = 0U; index < inventory.size(); ++index) {
      const ProjectionDescriptor& expected = inventory[index];
      const PrefillProjectionSidecarEntry& actual =
          manifest.projections[index];
      if (actual.ordinal != index ||
          actual.layer_index != expected.layer_index ||
          actual.family != expected.family ||
          actual.source_module != expected.module ||
          actual.output_size != expected.output_size ||
          actual.input_size != expected.input_size ||
          !lowercase_sha256(actual.source_sha256)) {
        return make_diagnostic(
            PrefillContractErrorCode::kInvalidManifest,
            "manifest.projections[" + std::to_string(index) + "]",
            "projection identity, family, shape, or source digest is invalid");
      }
      SidecarEncoding encoding;
      if (!expected_sidecar_encoding(manifest.kind, expected, encoding)) {
        return make_diagnostic(
            PrefillContractErrorCode::kArithmeticOverflow, expected.module,
            "expected sidecar byte calculation overflowed");
      }
      std::uint64_t expected_bytes = 0U;
      std::uint64_t expected_offset = 0U;
      std::uint64_t elements = 0U;
      if (!checked_add(encoding.weight_bytes, encoding.scale_bytes,
                       expected_bytes) ||
          !checked_add(expected_bytes, encoding.metadata_bytes,
                       expected_bytes) ||
          !checked_align_up(cursor, manifest.arena_alignment,
                            expected_offset) ||
          !checked_add(expected_offset, expected_bytes, cursor) ||
          !checked_multiply(expected.output_size, expected.input_size,
                            elements) ||
          !checked_add(expected_summary.logical_weight_elements, elements,
                       expected_summary.logical_weight_elements) ||
          !checked_add(expected_summary.payload_bytes, expected_bytes,
                       expected_summary.payload_bytes)) {
        return make_diagnostic(
            PrefillContractErrorCode::kArithmeticOverflow, expected.module,
            "manifest validation arithmetic overflowed");
      }
      if (actual.quantization != encoding.quantization ||
          actual.scale_group_size != encoding.scale_group_size ||
          actual.layout != encoding.layout ||
          actual.weight_bytes != encoding.weight_bytes ||
          actual.scale_bytes != encoding.scale_bytes ||
          actual.metadata_bytes != encoding.metadata_bytes ||
          actual.sidecar_byte_size != expected_bytes) {
        return make_diagnostic(
            PrefillContractErrorCode::kSizeMismatch, expected.module,
            "projection encoding or byte size differs from its sidecar kind");
      }
      if (actual.sidecar_offset != expected_offset ||
          (actual.sidecar_offset % manifest.arena_alignment) != 0U) {
        return make_diagnostic(
            PrefillContractErrorCode::kOffsetMismatch, expected.module,
            "projection sidecar offset is not the deterministic aligned offset",
            std::to_string(expected_offset),
            std::to_string(actual.sidecar_offset));
      }
      ++expected_summary.projection_count;
      ++expected_summary.family_counts[static_cast<std::size_t>(
          expected.family)];
      if (expected.nvfp4_source) {
        ++expected_summary.mlp_projection_count;
      } else {
        ++expected_summary.attention_projection_count;
      }
    }
    if (!checked_align_up(cursor, manifest.arena_alignment,
                          expected_summary.arena_bytes)) {
      return make_diagnostic(
          PrefillContractErrorCode::kArithmeticOverflow, "manifest.arena",
          "manifest arena alignment overflowed");
    }
    if (expected_summary.projection_count !=
            manifest.summary.projection_count ||
        expected_summary.mlp_projection_count !=
            manifest.summary.mlp_projection_count ||
        expected_summary.attention_projection_count !=
            manifest.summary.attention_projection_count ||
        expected_summary.family_counts != manifest.summary.family_counts ||
        expected_summary.logical_weight_elements !=
            manifest.summary.logical_weight_elements ||
        expected_summary.payload_bytes != manifest.summary.payload_bytes ||
        expected_summary.arena_bytes != manifest.summary.arena_bytes ||
        expected_summary.payload_bytes != expected_payload_bytes(manifest.kind)) {
      return make_diagnostic(PrefillContractErrorCode::kSizeMismatch,
                             "manifest.summary",
                             "manifest summary differs from its entries");
    }
    std::string digest;
    if (!digest_manifest_body(manifest, digest)) {
      return make_diagnostic(
          PrefillContractErrorCode::kArithmeticOverflow, "manifest.digest",
          "manifest digest input length overflowed");
    }
    if (digest != manifest.manifest_sha256) {
      return make_diagnostic(PrefillContractErrorCode::kDigestMismatch,
                             "manifest.digest",
                             "manifest body digest does not match",
                             digest, manifest.manifest_sha256);
    }
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(PrefillContractErrorCode::kAllocationFailure,
                           "manifest.validation",
                           "allocation failed while validating the manifest");
  } catch (...) {
    return make_diagnostic(PrefillContractErrorCode::kInvalidManifest,
                           "manifest.validation",
                           "unexpected manifest validation failure");
  }
}

PrefillPromptArenaPlanResult build_prefill_prompt_arena_plan(
    const PrefillPromptArenaOptions& options) {
  PrefillPromptArenaPlanResult result;
  try {
    if (options.prompt_token_count == 0U ||
        options.prompt_token_count > kPrefillPromptArenaMaximumTokens ||
        !is_power_of_two(options.arena_alignment) ||
        options.arena_alignment < 16U ||
        options.arena_alignment > (1ULL << 30U) ||
        options.max_arena_bytes == 0U) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kInvalidOption, "prompt_arena_options",
          "prompt token count, alignment, or arena limit is invalid",
          "1..40000 tokens and power-of-two alignment >= 16");
      return result;
    }
    std::uint32_t activation_bits = 0U;
    switch (options.activation) {
      case PrefillPromptActivation::kA8:
        if (options.activation_scale_group_size != 128U) {
          result.diagnostic = make_diagnostic(
              PrefillContractErrorCode::kInvalidOption,
              "activation_scale_group_size", "A8 staging requires K128");
          return result;
        }
        activation_bits = 8U;
        break;
      case PrefillPromptActivation::kA4:
        if (options.activation_scale_group_size != 64U &&
            options.activation_scale_group_size != 128U) {
          result.diagnostic = make_diagnostic(
              PrefillContractErrorCode::kInvalidOption,
              "activation_scale_group_size",
              "A4 staging requires K64 or K128");
          return result;
        }
        activation_bits = 4U;
        break;
      default:
        result.diagnostic = make_diagnostic(
            PrefillContractErrorCode::kInvalidOption, "activation",
            "unsupported prompt activation format");
        return result;
    }
    const std::uint64_t staging_tokens =
        options.staging_token_capacity == 0U
            ? options.prompt_token_count
            : options.staging_token_capacity;
    if (staging_tokens == 0U ||
        staging_tokens > options.prompt_token_count) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kInvalidOption,
          "staging_token_capacity",
          "staging capacity must be within the full prompt");
      return result;
    }

    PrefillPromptArenaPlan plan;
    plan.prompt_token_count = options.prompt_token_count;
    plan.activation = options.activation;
    plan.activation_scale_group_size =
        options.activation_scale_group_size;
    plan.staging_token_capacity = staging_tokens;
    plan.arena_alignment = options.arena_alignment;
    plan.whole_prompt_staging = staging_tokens == options.prompt_token_count;
    std::uint64_t cursor = 0U;

    const auto allocate_region =
        [&](const std::uint64_t logical_elements,
            const std::uint32_t element_bits,
            PrefillPromptArenaRegion& region) -> bool {
      std::uint64_t total_bits = 0U;
      std::uint64_t offset = 0U;
      std::uint64_t bytes = 0U;
      if (element_bits == 0U ||
          !checked_multiply(logical_elements, element_bits, total_bits) ||
          (total_bits % 8U) != 0U ||
          !checked_align_up(cursor, options.arena_alignment, offset)) {
        return false;
      }
      bytes = total_bits / 8U;
      if (!checked_add(offset, bytes, cursor)) {
        return false;
      }
      region.arena_offset = offset;
      region.byte_size = bytes;
      region.logical_element_capacity = logical_elements;
      region.element_bits = element_bits;
      return true;
    };

    std::uint64_t full_hidden_elements = 0U;
    std::uint64_t staged_hidden_elements = 0U;
    std::uint64_t staged_intermediate_elements = 0U;
    std::uint64_t staged_attention_output_elements = 0U;
    std::uint64_t hidden_scale_elements = 0U;
    std::uint64_t attention_output_scale_elements = 0U;
    std::uint64_t intermediate_scale_elements = 0U;
    const std::uint64_t group = options.activation_scale_group_size;
    if (!checked_multiply(options.prompt_token_count,
                          kPrefillPromptHiddenWidth,
                          full_hidden_elements) ||
        !checked_multiply(staging_tokens, kPrefillPromptHiddenWidth,
                          staged_hidden_elements) ||
        !checked_multiply(staging_tokens, kPrefillPromptIntermediateWidth,
                          staged_intermediate_elements) ||
        !checked_multiply(staging_tokens,
                          kPrefillPromptAttentionOutputWidth,
                          staged_attention_output_elements) ||
        !checked_multiply(staging_tokens,
                          kPrefillPromptHiddenWidth / group,
                          hidden_scale_elements) ||
        !checked_multiply(staging_tokens,
                          kPrefillPromptAttentionOutputWidth / group,
                          attention_output_scale_elements) ||
        !checked_multiply(staging_tokens,
                          kPrefillPromptIntermediateWidth / group,
                          intermediate_scale_elements) ||
        !allocate_region(full_hidden_elements, 16U, plan.hidden_bf16[0]) ||
        !allocate_region(full_hidden_elements, 16U, plan.hidden_bf16[1]) ||
        !allocate_region(staged_hidden_elements, activation_bits,
                         plan.hidden_quantized) ||
        !allocate_region(hidden_scale_elements, 16U,
                         plan.hidden_scales_bf16) ||
        !allocate_region(staged_intermediate_elements, activation_bits,
                         plan.intermediate_quantized) ||
        !allocate_region(intermediate_scale_elements, 16U,
                         plan.intermediate_scales_bf16) ||
        !allocate_region(options.prompt_token_count, 32U,
                         plan.row_sum_squares_fp32) ||
        !checked_align_up(cursor, options.arena_alignment, plan.arena_bytes)) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kArithmeticOverflow, "prompt_arena",
          "prompt slab or staging byte calculation overflowed");
      return result;
    }
    std::uint64_t attention_output_bits = 0U;
    std::uint64_t attention_output_scale_bits = 0U;
    if (!checked_multiply(staged_attention_output_elements, activation_bits,
                          attention_output_bits) ||
        !checked_multiply(attention_output_scale_elements, 16U,
                          attention_output_scale_bits) ||
        (attention_output_bits % 8U) != 0U ||
        (attention_output_scale_bits % 8U) != 0U) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kArithmeticOverflow,
          "attention_output_staging",
          "O-projection alias byte calculation overflowed");
      return result;
    }
    plan.attention_output_input_quantized = plan.intermediate_quantized;
    plan.attention_output_input_quantized.logical_element_capacity =
        staged_attention_output_elements;
    plan.attention_output_input_quantized.byte_size =
        attention_output_bits / 8U;
    plan.attention_output_input_scales_bf16 =
        plan.intermediate_scales_bf16;
    plan.attention_output_input_scales_bf16.logical_element_capacity =
        attention_output_scale_elements;
    plan.attention_output_input_scales_bf16.byte_size =
        attention_output_scale_bits / 8U;
    if (plan.attention_output_input_quantized.byte_size >
            plan.intermediate_quantized.byte_size ||
        plan.attention_output_input_scales_bf16.byte_size >
            plan.intermediate_scales_bf16.byte_size) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kSizeMismatch,
          "attention_output_staging",
          "O-projection alias exceeds intermediate staging");
      return result;
    }
    if (plan.arena_bytes > options.max_arena_bytes) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kArenaLimitExceeded, "prompt_arena",
          "prompt arena exceeds the configured limit",
          std::to_string(options.max_arena_bytes),
          std::to_string(plan.arena_bytes));
      return result;
    }
    result.value.emplace(plan);
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kAllocationFailure, "prompt_arena",
        "allocation failed while returning the prompt arena plan");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kInvalidOption, "prompt_arena",
        "unexpected prompt arena planning failure");
    return result;
  }
}

PrefillSidecarResidencyPlanResult build_prefill_sidecar_residency_plan(
    const PrefillSidecarResidencyRequest& request) {
  PrefillSidecarResidencyPlanResult result;
  const std::size_t selected_count =
      static_cast<std::size_t>(request.exact != nullptr) +
      static_cast<std::size_t>(request.a8 != nullptr) +
      static_cast<std::size_t>(request.a4 != nullptr);
  if (selected_count != 1U) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kResidencyConflict,
        "sidecar_residency",
        "exactly one of Exact, A8, or A4 may be resident");
    return result;
  }
  if (request.max_total_resident_bytes == 0U) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kInvalidOption, "max_total_resident_bytes",
        "resident byte limit must be nonzero");
    return result;
  }

  const PrefillSidecarManifest* manifest = nullptr;
  PrefillSidecarResidencyClass expected_class =
      PrefillSidecarResidencyClass::kExact;
  if (request.exact != nullptr) {
    manifest = request.exact;
  } else if (request.a8 != nullptr) {
    manifest = request.a8;
    expected_class = PrefillSidecarResidencyClass::kA8;
  } else {
    manifest = request.a4;
    expected_class = PrefillSidecarResidencyClass::kA4;
  }
  result.diagnostic = validate_prefill_sidecar_manifest(*manifest);
  if (!result.diagnostic) {
    return result;
  }
  if (manifest->residency_class != expected_class) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kResidencyConflict,
        "sidecar_residency",
        "manifest kind does not match its requested residency slot");
    return result;
  }

  std::uint64_t prompt_arena_bytes = 0U;
  if (expected_class == PrefillSidecarResidencyClass::kExact) {
    if (request.prompt_arena != nullptr) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kActivationMismatch, "prompt_arena",
          "quantized prompt staging cannot accompany the Exact sidecar");
      return result;
    }
  } else {
    if (request.prompt_arena == nullptr) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kActivationMismatch, "prompt_arena",
          "A8/A4 residency requires its prompt arena contract");
      return result;
    }
    PrefillPromptArenaOptions options;
    options.prompt_token_count = request.prompt_arena->prompt_token_count;
    options.activation = request.prompt_arena->activation;
    options.activation_scale_group_size =
        request.prompt_arena->activation_scale_group_size;
    options.staging_token_capacity =
        request.prompt_arena->staging_token_capacity;
    options.arena_alignment = request.prompt_arena->arena_alignment;
    options.max_arena_bytes = request.prompt_arena->arena_bytes;
    const PrefillPromptArenaPlanResult rebuilt =
        build_prefill_prompt_arena_plan(options);
    if (!rebuilt ||
        !same_prompt_plan(*rebuilt.value, *request.prompt_arena)) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kInvalidManifest, "prompt_arena",
          "prompt arena plan does not reproduce from its contract fields");
      return result;
    }
    const bool activation_matches =
        (expected_class == PrefillSidecarResidencyClass::kA8 &&
         request.prompt_arena->activation == PrefillPromptActivation::kA8) ||
        (expected_class == PrefillSidecarResidencyClass::kA4 &&
         request.prompt_arena->activation == PrefillPromptActivation::kA4);
    if (!activation_matches) {
      result.diagnostic = make_diagnostic(
          PrefillContractErrorCode::kActivationMismatch, "prompt_arena",
          "prompt activation does not match sidecar residency class");
      return result;
    }
    prompt_arena_bytes = request.prompt_arena->arena_bytes;
  }

  std::uint64_t peak = 0U;
  if (!checked_add(manifest->summary.arena_bytes, prompt_arena_bytes, peak)) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kArithmeticOverflow,
        "peak_resident_bytes",
        "sidecar plus prompt arena byte count overflowed");
    return result;
  }
  if (peak > request.max_total_resident_bytes) {
    result.diagnostic = make_diagnostic(
        PrefillContractErrorCode::kArenaLimitExceeded,
        "peak_resident_bytes",
        "sidecar plus prompt arena exceeds the resident budget",
        std::to_string(request.max_total_resident_bytes),
        std::to_string(peak));
    return result;
  }
  PrefillSidecarResidencyPlan plan;
  plan.sidecar_kind = manifest->kind;
  plan.residency_class = manifest->residency_class;
  plan.sidecar_bytes = manifest->summary.arena_bytes;
  plan.prompt_arena_bytes = prompt_arena_bytes;
  plan.peak_resident_bytes = peak;
  result.value.emplace(plan);
  return result;
}

std::string_view to_string(const PrefillProjectionFamily family) noexcept {
  switch (family) {
    case PrefillProjectionFamily::kMlpGate:
      return "mlp_gate";
    case PrefillProjectionFamily::kMlpUp:
      return "mlp_up";
    case PrefillProjectionFamily::kMlpDown:
      return "mlp_down";
    case PrefillProjectionFamily::kLinearQkv:
      return "linear_qkv";
    case PrefillProjectionFamily::kLinearZ:
      return "linear_z";
    case PrefillProjectionFamily::kLinearO:
      return "linear_o";
    case PrefillProjectionFamily::kFullQ:
      return "full_q";
    case PrefillProjectionFamily::kFullK:
      return "full_k";
    case PrefillProjectionFamily::kFullV:
      return "full_v";
    case PrefillProjectionFamily::kFullO:
      return "full_o";
    case PrefillProjectionFamily::kCount:
      break;
  }
  return "invalid";
}

std::string_view to_string(const PrefillSidecarKind kind) noexcept {
  switch (kind) {
    case PrefillSidecarKind::kExact:
      return "exact";
    case PrefillSidecarKind::kA8Safe:
      return "a8_safe";
    case PrefillSidecarKind::kA8Compact:
      return "a8_compact";
    case PrefillSidecarKind::kA4K64:
      return "a4_k64";
    case PrefillSidecarKind::kA4K128:
      return "a4_k128";
  }
  return "invalid";
}

std::string_view to_string(
    const PrefillSidecarResidencyClass residency) noexcept {
  switch (residency) {
    case PrefillSidecarResidencyClass::kExact:
      return "exact";
    case PrefillSidecarResidencyClass::kA8:
      return "a8";
    case PrefillSidecarResidencyClass::kA4:
      return "a4";
  }
  return "invalid";
}

std::string_view to_string(
    const PrefillWeightQuantization quantization) noexcept {
  switch (quantization) {
    case PrefillWeightQuantization::kExactNvfp4E2m1:
      return "exact_nvfp4_e2m1";
    case PrefillWeightQuantization::kExactFp8E4m3:
      return "exact_fp8_e4m3";
    case PrefillWeightQuantization::kSymmetricW8:
      return "symmetric_w8";
    case PrefillWeightQuantization::kSymmetricW4:
      return "symmetric_w4";
  }
  return "invalid";
}

std::string_view to_string(const PrefillSidecarLayout layout) noexcept {
  switch (layout) {
    case PrefillSidecarLayout::kExactNvfp4MarlinConsumer:
      return "exact_nvfp4_marlin_consumer";
    case PrefillSidecarLayout::kExactFp8SupermatrixConsumer:
      return "exact_fp8_supermatrix_consumer";
    case PrefillSidecarLayout::kSm87S8K128Consumer:
      return "sm87_s8_k128_consumer";
    case PrefillSidecarLayout::kSm87U4B8K32Consumer:
      return "sm87_u4b8_k32_consumer";
    case PrefillSidecarLayout::kSm87S4K64Consumer:
      return "sm87_s4_k64_consumer";
    case PrefillSidecarLayout::kSm87S4K128Consumer:
      return "sm87_s4_k128_consumer";
  }
  return "invalid";
}

std::string_view to_string(const PrefillContractErrorCode code) noexcept {
  switch (code) {
    case PrefillContractErrorCode::kNone:
      return "none";
    case PrefillContractErrorCode::kInvalidOption:
      return "invalid_option";
    case PrefillContractErrorCode::kUnsupportedCheckpoint:
      return "unsupported_checkpoint";
    case PrefillContractErrorCode::kInvalidSourceIdentity:
      return "invalid_source_identity";
    case PrefillContractErrorCode::kMissingSourceTensor:
      return "missing_source_tensor";
    case PrefillContractErrorCode::kSourceTensorMismatch:
      return "source_tensor_mismatch";
    case PrefillContractErrorCode::kInvalidManifest:
      return "invalid_manifest";
    case PrefillContractErrorCode::kCountMismatch:
      return "count_mismatch";
    case PrefillContractErrorCode::kSizeMismatch:
      return "size_mismatch";
    case PrefillContractErrorCode::kOffsetMismatch:
      return "offset_mismatch";
    case PrefillContractErrorCode::kDigestMismatch:
      return "digest_mismatch";
    case PrefillContractErrorCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case PrefillContractErrorCode::kArenaLimitExceeded:
      return "arena_limit_exceeded";
    case PrefillContractErrorCode::kResidencyConflict:
      return "residency_conflict";
    case PrefillContractErrorCode::kActivationMismatch:
      return "activation_mismatch";
    case PrefillContractErrorCode::kAllocationFailure:
      return "allocation_failure";
  }
  return "invalid";
}

}  // namespace q3x::runtime
