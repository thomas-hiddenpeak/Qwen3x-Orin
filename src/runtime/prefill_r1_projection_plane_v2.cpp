#include "q3x/runtime/prefill_r1_projection_plane_v2.h"

#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/runtime/prefill_attention_factorized_lane_converter.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_converter.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fcntl.h>
#include <iomanip>
#include <limits>
#include <locale>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <sys/file.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

namespace json = q3x::io::json;
namespace fs = std::filesystem;
namespace mw = q3x::model::weights;

constexpr std::string_view kManifestSchema =
    "q3x.prefill.r1-projection-plane-v2.manifest";
constexpr std::string_view kPolicySchema =
    "q3x.prefill.r1-projection-plane-v2.policy";
constexpr std::string_view kReceiptSchema =
    "q3x.prefill.r1-projection-plane-v2.receipt";
constexpr std::string_view kConverterAbi =
    "q3x_r1_projection_plane_v4_to_v2_byte_permutation_v1";
constexpr std::string_view kMode =
    "performance_upper_bound_r1_projection_plane_v2";

struct GateUpLayout final {
  bool valid = false;
  std::uint64_t packed_weight_offset = 0U;
  std::uint64_t packed_weight_bytes = 0U;
  std::uint64_t weight_scale_offset = 0U;
  std::uint64_t weight_scale_bytes = 0U;
  std::uint64_t gate_metadata_offset = 0U;
  std::uint64_t gate_metadata_bytes = 0U;
  std::uint64_t up_metadata_offset = 0U;
  std::uint64_t up_metadata_bytes = 0U;
  std::uint64_t projection_bytes = 0U;
};

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic make_diagnostic(
    const PrefillR1ProjectionPlaneV2ErrorCode code, std::string context,
    std::string message, std::string expected = {},
    std::string actual = {}, const int system_error = 0) {
  PrefillR1ProjectionPlaneV2Diagnostic result;
  result.code = code;
  result.context = std::move(context);
  result.message = std::move(message);
  result.expected = std::move(expected);
  result.actual = std::move(actual);
  result.system_error = system_error;
  return result;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& output) noexcept {
  return prefill_a4_factorized_lane_contract_detail::checked_add(
      left, right, output);
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left,
                                    const std::uint64_t right,
                                    std::uint64_t& output) noexcept {
  return prefill_a4_factorized_lane_contract_detail::checked_multiply(
      left, right, output);
}

[[nodiscard]] bool checked_align(const std::uint64_t value,
                                 std::uint64_t& output) noexcept {
  return prefill_a4_factorized_lane_contract_detail::checked_align_up(
      value, kPrefillR1ProjectionPlaneV2Alignment, output);
}

[[nodiscard]] bool lower_sha256(const std::string_view value) noexcept {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

[[nodiscard]] std::string sha256_text(const std::string_view value) {
  core::Sha256 hash;
  if (!hash.update(value.data(), value.size())) {
    return {};
  }
  return hash.finalize().hex();
}

void write_quoted(std::ostream& output, const std::string_view value) {
  output << '"';
  for (const unsigned char character : value) {
    switch (character) {
      case '"': output << "\\\""; break;
      case '\\': output << "\\\\"; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (character < 0x20U) {
          output << "\\u00" << std::hex << std::setw(2)
                 << std::setfill('0')
                 << static_cast<unsigned int>(character) << std::dec
                 << std::setfill(' ');
        } else {
          output << static_cast<char>(character);
        }
        break;
    }
  }
  output << '"';
}

[[nodiscard]] std::string_view family_name(
    const PrefillR1ProjectionPlaneV2PhysicalFamily family) noexcept {
  switch (family) {
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearQkv:
      return "linear_qkv";
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearZ:
      return "linear_z";
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearO:
      return "linear_o";
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kFullQ:
      return "full_q";
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kFullK:
      return "full_k";
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kFullV:
      return "full_v";
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kFullO:
      return "full_o";
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpGateUp:
      return "mlp_gate_up";
    case PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpDown:
      return "mlp_down";
  }
  return "invalid";
}

[[nodiscard]] std::string_view logical_family_name(
    const PrefillR1ProjectionPlaneV2LogicalFamily family) noexcept {
  switch (family) {
    case PrefillR1ProjectionPlaneV2LogicalFamily::kLinearQkv:
      return "linear_qkv";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kLinearZ:
      return "linear_z";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kLinearO:
      return "linear_o";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kFullQ:
      return "full_q";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kFullK:
      return "full_k";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kFullV:
      return "full_v";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kFullO:
      return "full_o";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kMlpGate:
      return "mlp_gate";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kMlpUp:
      return "mlp_up";
    case PrefillR1ProjectionPlaneV2LogicalFamily::kMlpDown:
      return "mlp_down";
  }
  return "invalid";
}

[[nodiscard]] GateUpLayout gate_up_layout(const std::uint64_t output_size,
                                          const std::uint64_t input_size) {
  GateUpLayout result;
  const auto source = prefill_a4_factorized_lane_projection_layout_plan(
      output_size, input_size, 1U);
  if (!source) {
    return result;
  }
  result.packed_weight_offset = 0U;
  if (!checked_multiply(source.packed_weight_bytes, 2U,
                        result.packed_weight_bytes) ||
      !checked_align(result.packed_weight_bytes,
                     result.weight_scale_offset) ||
      !checked_multiply(source.weight_scale_bytes, 2U,
                        result.weight_scale_bytes)) {
    return result;
  }
  std::uint64_t cursor = 0U;
  if (!checked_add(result.weight_scale_offset, result.weight_scale_bytes,
                   cursor) ||
      !checked_align(cursor, result.gate_metadata_offset)) {
    return result;
  }
  result.gate_metadata_bytes = source.metadata_bytes;
  if (!checked_add(result.gate_metadata_offset,
                   result.gate_metadata_bytes, cursor) ||
      !checked_align(cursor, result.up_metadata_offset)) {
    return result;
  }
  result.up_metadata_bytes = source.metadata_bytes;
  std::uint64_t used_bytes = 0U;
  std::uint64_t equal_byte_capacity = 0U;
  if (!checked_add(result.up_metadata_offset, result.up_metadata_bytes,
                   cursor) ||
      !checked_align(cursor, used_bytes) ||
      !checked_multiply(source.projection_bytes, 2U,
                        equal_byte_capacity) ||
      used_bytes > equal_byte_capacity) {
    return result;
  }
  // The physical record is intentionally equal-byte with Gate+Up v4 even
  // when a tiny correctness fixture leaves additional tail padding after the
  // two metadata blocks. Production shapes close this gap exactly; fixtures
  // must still exercise the same ABI without being rejected by alignment.
  result.projection_bytes = equal_byte_capacity;
  result.valid = true;
  return result;
}

[[nodiscard]] bool same_base(
    const PrefillMLPFactorizedLaneBaseK256Binding& mlp,
    const PrefillAttentionFactorizedLaneBaseK256Binding& attention) noexcept {
  return mlp.physical_layout == attention.physical_layout &&
         mlp.packed_k_group_size == attention.packed_k_group_size &&
         mlp.scale_group_size == attention.scale_group_size &&
         mlp.manifest_sha256 == attention.manifest_sha256 &&
         mlp.policy_sha256 == attention.policy_sha256 &&
         mlp.payload_sha256 == attention.payload_sha256 &&
         mlp.receipt_sha256 == attention.receipt_sha256;
}

[[nodiscard]] PrefillR1ProjectionPlaneV2BaseK256Binding base_binding(
    const PrefillMLPFactorizedLaneBaseK256Binding& source) {
  return {source.physical_layout,
          source.packed_k_group_size,
          source.scale_group_size,
          source.manifest_sha256,
          source.policy_sha256,
          source.payload_sha256,
          source.receipt_sha256};
}

[[nodiscard]] bool same_base(
    const PrefillR1ProjectionPlaneV2BaseK256Binding& left,
    const PrefillR1ProjectionPlaneV2BaseK256Binding& right) noexcept {
  return left.physical_layout == right.physical_layout &&
         left.packed_k_group_size == right.packed_k_group_size &&
         left.scale_group_size == right.scale_group_size &&
         left.manifest_sha256 == right.manifest_sha256 &&
         left.policy_sha256 == right.policy_sha256 &&
         left.payload_sha256 == right.payload_sha256 &&
         left.receipt_sha256 == right.receipt_sha256;
}

[[nodiscard]] bool valid_base(
    const PrefillR1ProjectionPlaneV2BaseK256Binding& base) noexcept {
  return base.physical_layout ==
             kPrefillMLPFactorizedLaneRequiredBaseK256Layout &&
         base.packed_k_group_size ==
             kPrefillMLPFactorizedLaneRequiredBasePackedK &&
         base.scale_group_size ==
             kPrefillMLPFactorizedLaneRequiredBaseScaleK &&
         lower_sha256(base.manifest_sha256) &&
         lower_sha256(base.policy_sha256) &&
         lower_sha256(base.payload_sha256) &&
         lower_sha256(base.receipt_sha256);
}

[[nodiscard]] bool same_base(
    const PrefillR1ProjectionPlaneV2BaseK256Binding& left,
    const PrefillMLPFactorizedLaneBaseK256Binding& right) noexcept {
  return left.physical_layout == right.physical_layout &&
         left.packed_k_group_size == right.packed_k_group_size &&
         left.scale_group_size == right.scale_group_size &&
         left.manifest_sha256 == right.manifest_sha256 &&
         left.policy_sha256 == right.policy_sha256 &&
         left.payload_sha256 == right.payload_sha256 &&
         left.receipt_sha256 == right.receipt_sha256;
}

[[nodiscard]] bool same_base(
    const PrefillR1ProjectionPlaneV2BaseK256Binding& left,
    const PrefillAttentionFactorizedLaneBaseK256Binding& right) noexcept {
  return left.physical_layout == right.physical_layout &&
         left.packed_k_group_size == right.packed_k_group_size &&
         left.scale_group_size == right.scale_group_size &&
         left.manifest_sha256 == right.manifest_sha256 &&
         left.policy_sha256 == right.policy_sha256 &&
         left.payload_sha256 == right.payload_sha256 &&
         left.receipt_sha256 == right.receipt_sha256;
}

[[nodiscard]] bool same_source_binding(
    const PrefillR1ProjectionPlaneV2SourceBinding& left,
    const PrefillR1ProjectionPlaneV2SourceBinding& right) noexcept {
  return left.version_major == right.version_major &&
         left.version_minor == right.version_minor &&
         left.physical_layout == right.physical_layout &&
         left.manifest_sha256 == right.manifest_sha256 &&
         left.policy_sha256 == right.policy_sha256 &&
         left.payload_sha256 == right.payload_sha256 &&
         left.receipt_sha256 == right.receipt_sha256 &&
         left.payload_bytes == right.payload_bytes;
}

[[nodiscard]] PrefillR1ProjectionPlaneV2SourceBinding source_binding(
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView& view) {
  return {view.version_major, view.version_minor, view.physical_layout,
          view.manifest_sha256, view.policy_sha256, view.payload_sha256,
          view.receipt_sha256, view.bytes};
}

[[nodiscard]] bool valid_source_binding(
    const PrefillR1ProjectionPlaneV2SourceBinding& source,
    const std::uint32_t expected_major, const std::uint32_t expected_minor,
    const std::string_view expected_layout,
    const std::uint64_t expected_bytes) noexcept {
  return source.version_major == expected_major &&
         source.version_minor == expected_minor &&
         source.physical_layout == expected_layout &&
         source.payload_bytes == expected_bytes &&
         lower_sha256(source.manifest_sha256) &&
         lower_sha256(source.policy_sha256) &&
         lower_sha256(source.payload_sha256) &&
         lower_sha256(source.receipt_sha256);
}

[[nodiscard]] bool valid_source_view(
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView& view,
    const std::uint32_t expected_major, const std::uint32_t expected_minor,
    const std::string_view expected_layout,
    const std::uint64_t expected_bytes,
    const std::string_view expected_manifest_sha256) noexcept {
  return view.authenticated && view.version_major == expected_major &&
         view.version_minor == expected_minor &&
         view.physical_layout == expected_layout &&
         view.bytes == expected_bytes &&
         view.manifest_sha256 == expected_manifest_sha256 &&
         lower_sha256(view.manifest_sha256) &&
         lower_sha256(view.policy_sha256) &&
         lower_sha256(view.payload_sha256) &&
         lower_sha256(view.receipt_sha256);
}

[[nodiscard]] PrefillR1ProjectionPlaneV2PhysicalFamily attention_family(
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      return PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearQkv;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      return PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearZ;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      return PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearO;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      return PrefillR1ProjectionPlaneV2PhysicalFamily::kFullQ;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      return PrefillR1ProjectionPlaneV2PhysicalFamily::kFullK;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      return PrefillR1ProjectionPlaneV2PhysicalFamily::kFullV;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      return PrefillR1ProjectionPlaneV2PhysicalFamily::kFullO;
  }
  return static_cast<PrefillR1ProjectionPlaneV2PhysicalFamily>(0xffU);
}

[[nodiscard]] PrefillR1ProjectionPlaneV2LogicalFamily
attention_logical_family(
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  return static_cast<PrefillR1ProjectionPlaneV2LogicalFamily>(
      static_cast<std::uint8_t>(family));
}

[[nodiscard]] std::string expected_attention_module(
    const std::uint32_t layer,
    const PrefillAttentionFactorizedLaneProjectionFamily family) {
  std::string suffix;
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      suffix = "linear_attn.in_proj_qkv";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      suffix = "linear_attn.in_proj_z";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      suffix = "linear_attn.out_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      suffix = "self_attn.q_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      suffix = "self_attn.k_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      suffix = "self_attn.v_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      suffix = "self_attn.o_proj";
      break;
  }
  return "model.language_model.layers." + std::to_string(layer) + "." +
         suffix;
}

[[nodiscard]] std::string expected_mlp_module(
    const std::uint32_t layer,
    const PrefillR1ProjectionPlaneV2LogicalFamily family) {
  std::string suffix;
  switch (family) {
    case PrefillR1ProjectionPlaneV2LogicalFamily::kMlpGate:
      suffix = "gate_proj";
      break;
    case PrefillR1ProjectionPlaneV2LogicalFamily::kMlpUp:
      suffix = "up_proj";
      break;
    case PrefillR1ProjectionPlaneV2LogicalFamily::kMlpDown:
      suffix = "down_proj";
      break;
    default:
      return {};
  }
  return "model.language_model.layers." + std::to_string(layer) +
         ".mlp." + suffix;
}

[[nodiscard]] const PrefillA4FactorizedLaneProjectionLayoutPlan*
attention_layout(
    const PrefillAttentionFactorizedLaneOverlayLayoutPlan& plan,
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      return &plan.linear_qkv;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      return &plan.linear_z;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      return &plan.linear_o;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      return &plan.full_q;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      return &plan.full_k;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      return &plan.full_v;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      return &plan.full_o;
  }
  return nullptr;
}

void append_source_json(
    std::ostream& output,
    const PrefillR1ProjectionPlaneV2SourceBinding& source) {
  output << "{\"manifest_sha256\":";
  write_quoted(output, source.manifest_sha256);
  output << ",\"payload_bytes\":" << source.payload_bytes
         << ",\"payload_sha256\":";
  write_quoted(output, source.payload_sha256);
  output << ",\"physical_layout\":";
  write_quoted(output, source.physical_layout);
  output << ",\"policy_sha256\":";
  write_quoted(output, source.policy_sha256);
  output << ",\"receipt_sha256\":";
  write_quoted(output, source.receipt_sha256);
  output << ",\"version\":[" << source.version_major << ','
         << source.version_minor << "]}";
}

void append_base_json(
    std::ostream& output,
    const PrefillR1ProjectionPlaneV2BaseK256Binding& base) {
  output << "{\"manifest_sha256\":";
  write_quoted(output, base.manifest_sha256);
  output << ",\"packed_k_group_size\":" << base.packed_k_group_size
         << ",\"payload_sha256\":";
  write_quoted(output, base.payload_sha256);
  output << ",\"physical_layout\":";
  write_quoted(output, base.physical_layout);
  output << ",\"policy_sha256\":";
  write_quoted(output, base.policy_sha256);
  output << ",\"receipt_sha256\":";
  write_quoted(output, base.receipt_sha256);
  output << ",\"scale_group_size\":" << base.scale_group_size << '}';
}

[[nodiscard]] std::string manifest_document(
    const PrefillR1ProjectionPlaneV2Manifest& manifest) {
  std::ostringstream output;
  output << "{\"attention_v4\":";
  append_source_json(output, manifest.attention_v4);
  output << ",\"lane_count\":" << manifest.lane_count
         << ",\"logical_projection_count\":"
         << manifest.logical_projection_count
         << ",\"logical_projections\":[";
  for (std::size_t index = 0U;
       index < manifest.logical_projections.size(); ++index) {
    if (index != 0U) output << ',';
    const auto& entry = manifest.logical_projections[index];
    output << "{\"family\":";
    write_quoted(output, logical_family_name(entry.family));
    output << ",\"input_size\":" << entry.input_size
           << ",\"layer\":" << entry.layer_index
           << ",\"logical_ordinal\":" << entry.logical_ordinal
           << ",\"output_size\":" << entry.output_size
           << ",\"physical_ordinal\":" << entry.physical_ordinal
           << ",\"source_module\":";
    write_quoted(output, entry.source_module);
    output << ",\"source_ordinal\":" << entry.source_ordinal
           << ",\"source_sha256\":";
    write_quoted(output, entry.source_sha256);
    output << '}';
  }
  output << "],\"mlp_v4\":";
  append_source_json(output, manifest.mlp_v4);
  output << ",\"payload_bytes\":" << manifest.payload_bytes
         << ",\"physical_layout\":";
  write_quoted(output, manifest.physical_layout);
  output << ",\"physical_projection_count\":"
         << manifest.physical_projection_count << ",\"projections\":[";
  for (std::size_t index = 0U; index < manifest.projections.size(); ++index) {
    if (index != 0U) output << ',';
    const auto& entry = manifest.projections[index];
    output << "{\"family\":";
    write_quoted(output, family_name(entry.family));
    output << ",\"input_size\":" << entry.input_size
           << ",\"layer\":" << entry.layer_index
           << ",\"logical_count\":" << entry.logical_projection_count
           << ",\"metadata\":[" << entry.primary_metadata_offset << ','
           << entry.primary_metadata_bytes << ','
           << entry.secondary_metadata_offset << ','
           << entry.secondary_metadata_bytes << ']'
           << ",\"ordinal\":" << entry.physical_ordinal
           << ",\"output_size\":" << entry.output_size
           << ",\"packed\":[" << entry.packed_weight_offset << ','
           << entry.packed_weight_bytes << ']'
           << ",\"payload\":[" << entry.payload_offset << ','
           << entry.payload_bytes << ']'
           << ",\"scale\":[" << entry.weight_scale_offset << ','
           << entry.weight_scale_bytes << ']'
           << ",\"source_ordinals\":[" << entry.source_primary_ordinal
           << ',' << entry.source_secondary_ordinal << ']'
           << ",\"source_offsets\":[" << entry.source_primary_offset << ','
           << entry.source_secondary_offset << "]}";
  }
  output << "],\"required_base_k256\":";
  append_base_json(output, manifest.required_base_k256);
  output << ",\"schema\":";
  write_quoted(output, kManifestSchema);
  output << ",\"source_checkpoint_id\":";
  write_quoted(output, manifest.source_checkpoint_id);
  output << ",\"source_config_sha256\":";
  write_quoted(output, manifest.source_config_sha256);
  output << ",\"source_index_sha256\":";
  write_quoted(output, manifest.source_index_sha256);
  output << ",\"version\":[" << manifest.version_major << ','
         << manifest.version_minor << "]}";
  return output.str();
}

[[nodiscard]] std::string policy_document(
    const PrefillR1ProjectionPlaneV2Policy& policy) {
  std::ostringstream output;
  output.imbue(std::locale::classic());
  output << "{\"attention_source_policy_sha256\":";
  write_quoted(output, policy.attention_source_policy_sha256);
  output << ",\"converter_abi\":";
  write_quoted(output, policy.converter_abi);
  output << ",\"manifest_sha256\":";
  write_quoted(output, policy.manifest_sha256);
  output << ",\"mlp_source_policy_sha256\":";
  write_quoted(output, policy.mlp_source_policy_sha256);
  output << ",\"mode\":";
  write_quoted(output, policy.mode);
  output << ",\"performance_upper_bound_only\":"
         << (policy.performance_upper_bound_only ? "true" : "false")
         << ",\"physical_layout\":";
  write_quoted(output, policy.physical_layout);
  output << ",\"production_residency_eligible\":"
         << (policy.production_residency_eligible ? "true" : "false")
         << ",\"projections\":[";
  for (std::size_t index = 0U; index < policy.projections.size(); ++index) {
    if (index != 0U) output << ',';
    const auto& entry = policy.projections[index];
    output << "{\"activation_clip_ratio\":" << std::setprecision(17)
           << entry.activation_clip_ratio << ",\"factor\":{"
           << "\"element_count\":" << entry.factor_element_count
           << ",\"path\":";
    write_quoted(output, entry.factor_path);
    output << ",\"scheme\":";
    write_quoted(output, entry.factor_scheme);
    output << ",\"sha256\":";
    write_quoted(output, entry.factor_sha256);
    output << "},\"logical_ordinal\":" << entry.logical_ordinal
           << ",\"physical_ordinal\":" << entry.physical_ordinal
           << ",\"source_module\":";
    write_quoted(output, entry.source_module);
    output << ",\"source_ordinal\":" << entry.source_ordinal
           << ",\"source_sha256\":";
    write_quoted(output, entry.source_sha256);
    output << ",\"weight_clip_ratio\":" << std::setprecision(17)
           << entry.weight_clip_ratio << '}';
  }
  output << ']'
         << ",\"quality_production_eligible\":"
         << (policy.quality_production_eligible ? "true" : "false")
         << ",\"schema\":";
  write_quoted(output, kPolicySchema);
  output << ",\"version\":[" << policy.version_major << ','
         << policy.version_minor << "]}";
  return output.str();
}

[[nodiscard]] std::string receipt_document(
    const PrefillR1ProjectionPlaneV2Receipt& receipt) {
  std::ostringstream output;
  output << "{\"atomic_installation_required\":"
         << (receipt.atomic_installation_required ? "true" : "false")
         << ",\"attention_v4\":";
  append_source_json(output, receipt.attention_v4);
  output << ",\"legacy_r1_co_residency_allowed\":"
         << (receipt.legacy_r1_co_residency_allowed ? "true" : "false")
         << ",\"logical_projection_count\":"
         << receipt.logical_projection_count << ",\"manifest_sha256\":";
  write_quoted(output, receipt.manifest_sha256);
  output << ",\"mlp_v4\":";
  append_source_json(output, receipt.mlp_v4);
  output << ",\"payload_bytes\":" << receipt.payload_bytes
         << ",\"payload_sha256\":";
  write_quoted(output, receipt.payload_sha256);
  output << ",\"performance_upper_bound_only\":"
         << (receipt.performance_upper_bound_only ? "true" : "false")
         << ",\"physical_layout\":";
  write_quoted(output, receipt.physical_layout);
  output << ",\"physical_projection_count\":"
         << receipt.physical_projection_count
         << ",\"policy_bytes\":" << receipt.policy_bytes
         << ",\"policy_sha256\":";
  write_quoted(output, receipt.policy_sha256);
  output << ",\"required_base_k256\":";
  append_base_json(output, receipt.required_base_k256);
  output << ",\"production_residency_eligible\":"
         << (receipt.production_residency_eligible ? "true" : "false")
         << ",\"quality_production_eligible\":"
         << (receipt.quality_production_eligible ? "true" : "false")
         << ",\"schema\":";
  write_quoted(output, kReceiptSchema);
  output << ",\"version\":[" << receipt.version_major << ','
         << receipt.version_minor << "]}";
  return output.str();
}

[[nodiscard]] std::uint64_t canonical_packed_offset(
    const std::uint64_t output_coordinate, const std::uint64_t k64_group,
    const std::uint64_t byte_in_k64,
    const std::uint64_t input_size) noexcept {
  const std::uint64_t k64_groups = input_size / 64U;
  return (((output_coordinate / 64U) * k64_groups + k64_group) * 64U +
          output_coordinate % 64U) *
             32U +
         byte_in_k64;
}

[[nodiscard]] bool ranges_overlap(const void* first,
                                  const std::size_t first_bytes,
                                  const void* second,
                                  const std::size_t second_bytes) noexcept {
  const auto first_begin = reinterpret_cast<std::uintptr_t>(first);
  const auto second_begin = reinterpret_cast<std::uintptr_t>(second);
  if (first_bytes > std::numeric_limits<std::uintptr_t>::max() - first_begin ||
      second_bytes >
          std::numeric_limits<std::uintptr_t>::max() - second_begin) {
    return true;
  }
  const auto first_end = first_begin + first_bytes;
  const auto second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

[[nodiscard]] bool exact_keys(
    const json::Value::Object& object,
    const std::initializer_list<std::string_view> keys) {
  if (object.size() != keys.size()) {
    return false;
  }
  return std::all_of(keys.begin(), keys.end(), [&object](const auto key) {
    return object.find(key) != object.end();
  });
}

[[nodiscard]] bool json_string(const json::Value::Object& object,
                               const std::string_view key,
                               std::string& output) {
  const auto found = object.find(key);
  const std::string* const value =
      found == object.end() ? nullptr : found->second.as_string();
  if (value == nullptr) return false;
  output = *value;
  return true;
}

[[nodiscard]] bool json_uint(const json::Value::Object& object,
                             const std::string_view key,
                             std::uint64_t& output) {
  const auto found = object.find(key);
  const json::Number* const value =
      found == object.end() ? nullptr : found->second.as_number();
  return value != nullptr && value->to_uint64(output);
}

[[nodiscard]] bool json_double(const json::Value::Object& object,
                               const std::string_view key,
                               double& output) {
  const auto found = object.find(key);
  const json::Number* const value =
      found == object.end() ? nullptr : found->second.as_number();
  return value != nullptr && value->to_double(output) &&
         std::isfinite(output);
}

[[nodiscard]] bool json_bool(const json::Value::Object& object,
                             const std::string_view key, bool& output) {
  const auto found = object.find(key);
  const bool* const value =
      found == object.end() ? nullptr : found->second.as_bool();
  if (value == nullptr) return false;
  output = *value;
  return true;
}

[[nodiscard]] bool parse_version(const json::Value& value,
                                 std::uint32_t& major,
                                 std::uint32_t& minor) {
  const auto* const array = value.as_array();
  std::uint64_t parsed_major = 0U;
  std::uint64_t parsed_minor = 0U;
  if (array == nullptr || array->size() != 2U ||
      (*array)[0U].as_number() == nullptr ||
      !(*array)[0U].as_number()->to_uint64(parsed_major) ||
      (*array)[1U].as_number() == nullptr ||
      !(*array)[1U].as_number()->to_uint64(parsed_minor) ||
      parsed_major > std::numeric_limits<std::uint32_t>::max() ||
      parsed_minor > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  major = static_cast<std::uint32_t>(parsed_major);
  minor = static_cast<std::uint32_t>(parsed_minor);
  return true;
}

[[nodiscard]] bool parse_u64_array(const json::Value& value,
                                   const std::size_t count,
                                   std::uint64_t* const output) {
  const auto* const array = value.as_array();
  if (array == nullptr || array->size() != count || output == nullptr) {
    return false;
  }
  for (std::size_t index = 0U; index < count; ++index) {
    const auto* const number = (*array)[index].as_number();
    if (number == nullptr || !number->to_uint64(output[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool parse_source(
    const json::Value& value,
    PrefillR1ProjectionPlaneV2SourceBinding& source) {
  const auto* const object = value.as_object();
  std::uint64_t bytes = 0U;
  if (object == nullptr ||
      !exact_keys(*object,
                  {"manifest_sha256", "payload_bytes", "payload_sha256",
                   "physical_layout", "policy_sha256", "receipt_sha256",
                   "version"}) ||
      !json_string(*object, "manifest_sha256", source.manifest_sha256) ||
      !json_uint(*object, "payload_bytes", bytes) ||
      !json_string(*object, "payload_sha256", source.payload_sha256) ||
      !json_string(*object, "physical_layout", source.physical_layout) ||
      !json_string(*object, "policy_sha256", source.policy_sha256) ||
      !json_string(*object, "receipt_sha256", source.receipt_sha256) ||
      !parse_version(object->at("version"), source.version_major,
                     source.version_minor)) {
    return false;
  }
  source.payload_bytes = bytes;
  return true;
}

[[nodiscard]] bool parse_base(
    const json::Value& value,
    PrefillR1ProjectionPlaneV2BaseK256Binding& base) {
  const auto* const object = value.as_object();
  std::uint64_t packed_k = 0U;
  std::uint64_t scale_k = 0U;
  if (object == nullptr ||
      !exact_keys(*object,
                  {"manifest_sha256", "packed_k_group_size",
                   "payload_sha256", "physical_layout", "policy_sha256",
                   "receipt_sha256", "scale_group_size"}) ||
      !json_string(*object, "manifest_sha256", base.manifest_sha256) ||
      !json_uint(*object, "packed_k_group_size", packed_k) ||
      packed_k > std::numeric_limits<std::uint32_t>::max() ||
      !json_string(*object, "payload_sha256", base.payload_sha256) ||
      !json_string(*object, "physical_layout", base.physical_layout) ||
      !json_string(*object, "policy_sha256", base.policy_sha256) ||
      !json_string(*object, "receipt_sha256", base.receipt_sha256) ||
      !json_uint(*object, "scale_group_size", scale_k) ||
      scale_k > std::numeric_limits<std::uint32_t>::max()) {
    return false;
  }
  base.packed_k_group_size = static_cast<std::uint32_t>(packed_k);
  base.scale_group_size = static_cast<std::uint32_t>(scale_k);
  return true;
}

[[nodiscard]] std::optional<PrefillR1ProjectionPlaneV2PhysicalFamily>
parse_physical_family(const std::string_view name) {
  for (const auto family :
       {PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearQkv,
        PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearZ,
        PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearO,
        PrefillR1ProjectionPlaneV2PhysicalFamily::kFullQ,
        PrefillR1ProjectionPlaneV2PhysicalFamily::kFullK,
        PrefillR1ProjectionPlaneV2PhysicalFamily::kFullV,
        PrefillR1ProjectionPlaneV2PhysicalFamily::kFullO,
        PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpGateUp,
        PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpDown}) {
    if (family_name(family) == name) return family;
  }
  return std::nullopt;
}

[[nodiscard]] std::optional<PrefillR1ProjectionPlaneV2LogicalFamily>
parse_logical_family(const std::string_view name) {
  for (const auto family :
       {PrefillR1ProjectionPlaneV2LogicalFamily::kLinearQkv,
        PrefillR1ProjectionPlaneV2LogicalFamily::kLinearZ,
        PrefillR1ProjectionPlaneV2LogicalFamily::kLinearO,
        PrefillR1ProjectionPlaneV2LogicalFamily::kFullQ,
        PrefillR1ProjectionPlaneV2LogicalFamily::kFullK,
        PrefillR1ProjectionPlaneV2LogicalFamily::kFullV,
        PrefillR1ProjectionPlaneV2LogicalFamily::kFullO,
        PrefillR1ProjectionPlaneV2LogicalFamily::kMlpGate,
        PrefillR1ProjectionPlaneV2LogicalFamily::kMlpUp,
        PrefillR1ProjectionPlaneV2LogicalFamily::kMlpDown}) {
    if (logical_family_name(family) == name) return family;
  }
  return std::nullopt;
}

[[nodiscard]] bool assign_u32(const std::uint64_t value,
                              std::uint32_t& output) {
  if (value > std::numeric_limits<std::uint32_t>::max()) return false;
  output = static_cast<std::uint32_t>(value);
  return true;
}

class UniqueFd final {
 public:
  UniqueFd() noexcept = default;
  explicit UniqueFd(const int fd) noexcept : fd_(fd) {}
  ~UniqueFd() {
    if (fd_ >= 0) (void)::close(fd_);
  }
  UniqueFd(const UniqueFd&) = delete;
  UniqueFd& operator=(const UniqueFd&) = delete;
  UniqueFd(UniqueFd&& other) noexcept : fd_(other.fd_) { other.fd_ = -1; }
  UniqueFd& operator=(UniqueFd&& other) noexcept {
    if (this != &other) {
      if (fd_ >= 0) (void)::close(fd_);
      fd_ = other.fd_;
      other.fd_ = -1;
    }
    return *this;
  }
  [[nodiscard]] int get() const noexcept { return fd_; }
  [[nodiscard]] explicit operator bool() const noexcept { return fd_ >= 0; }

 private:
  int fd_ = -1;
};

class MappedRegion final {
 public:
  MappedRegion() noexcept = default;
  ~MappedRegion() { unmap(); }
  MappedRegion(const MappedRegion&) = delete;
  MappedRegion& operator=(const MappedRegion&) = delete;
  [[nodiscard]] bool map(const int fd, const std::uint64_t bytes,
                         const int protection) noexcept {
    if (bytes == 0U || bytes > std::numeric_limits<std::size_t>::max()) {
      errno = EOVERFLOW;
      return false;
    }
    bytes_ = static_cast<std::size_t>(bytes);
    data_ = ::mmap(nullptr, bytes_, protection, MAP_SHARED, fd, 0);
    return data_ != MAP_FAILED;
  }
  [[nodiscard]] std::uint8_t* data() noexcept {
    return static_cast<std::uint8_t*>(data_);
  }
  [[nodiscard]] const std::uint8_t* data() const noexcept {
    return static_cast<const std::uint8_t*>(data_);
  }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] explicit operator bool() const noexcept {
    return data_ != MAP_FAILED;
  }
  void unmap() noexcept {
    if (data_ != MAP_FAILED) {
      (void)::munmap(data_, bytes_);
      data_ = MAP_FAILED;
      bytes_ = 0U;
    }
  }

 private:
  void* data_ = MAP_FAILED;
  std::size_t bytes_ = 0U;
};

struct FileSnapshot final {
  std::uint64_t device = 0U;
  std::uint64_t inode = 0U;
  std::uint64_t size = 0U;
  std::uint64_t links = 0U;
  std::int64_t mtime_s = 0;
  std::int64_t mtime_ns = 0;
  std::int64_t ctime_s = 0;
  std::int64_t ctime_ns = 0;
};

[[nodiscard]] bool capture_snapshot(const int fd, FileSnapshot& output,
                                    int& error) noexcept {
  struct stat status {};
  if (::fstat(fd, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_size < 0) {
    error = errno != 0 ? errno : EINVAL;
    return false;
  }
  output.device = static_cast<std::uint64_t>(status.st_dev);
  output.inode = static_cast<std::uint64_t>(status.st_ino);
  output.size = static_cast<std::uint64_t>(status.st_size);
  output.links = static_cast<std::uint64_t>(status.st_nlink);
  output.mtime_s = status.st_mtim.tv_sec;
  output.mtime_ns = status.st_mtim.tv_nsec;
  output.ctime_s = status.st_ctim.tv_sec;
  output.ctime_ns = status.st_ctim.tv_nsec;
  return true;
}

[[nodiscard]] bool same_snapshot(const FileSnapshot& left,
                                 const FileSnapshot& right) noexcept {
  return left.device == right.device && left.inode == right.inode &&
         left.size == right.size && left.links == right.links &&
         left.mtime_s == right.mtime_s && left.mtime_ns == right.mtime_ns &&
         left.ctime_s == right.ctime_s && left.ctime_ns == right.ctime_ns;
}

[[nodiscard]] bool same_inode(const FileSnapshot& left,
                              const FileSnapshot& right) noexcept {
  return left.device == right.device && left.inode == right.inode;
}

[[nodiscard]] bool pread_exact(const int fd, void* const output,
                               const std::size_t bytes,
                               const std::uint64_t offset,
                               int& error) noexcept {
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max())) {
    error = EOVERFLOW;
    return false;
  }
  auto* destination = static_cast<std::uint8_t*>(output);
  std::size_t completed = 0U;
  while (completed < bytes) {
    if (offset > std::numeric_limits<std::uint64_t>::max() - completed ||
        offset + completed > static_cast<std::uint64_t>(
                                 std::numeric_limits<off_t>::max())) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count = ::pread(
        fd, destination + completed, bytes - completed,
        static_cast<off_t>(offset + completed));
    if (count < 0) {
      if (errno == EINTR) continue;
      error = errno;
      return false;
    }
    if (count == 0) {
      error = EIO;
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  return true;
}

[[nodiscard]] bool pwrite_exact(const int fd, const void* const input,
                                const std::size_t bytes,
                                const std::uint64_t offset,
                                int& error) noexcept {
  if (offset > static_cast<std::uint64_t>(
                   std::numeric_limits<off_t>::max())) {
    error = EOVERFLOW;
    return false;
  }
  const auto* source = static_cast<const std::uint8_t*>(input);
  std::size_t completed = 0U;
  while (completed < bytes) {
    if (offset > std::numeric_limits<std::uint64_t>::max() - completed ||
        offset + completed > static_cast<std::uint64_t>(
                                 std::numeric_limits<off_t>::max())) {
      error = EOVERFLOW;
      return false;
    }
    const ssize_t count = ::pwrite(
        fd, source + completed, bytes - completed,
        static_cast<off_t>(offset + completed));
    if (count < 0) {
      if (errno == EINTR) continue;
      error = errno;
      return false;
    }
    if (count == 0) {
      error = EIO;
      return false;
    }
    completed += static_cast<std::size_t>(count);
  }
  return true;
}

struct LockedFile final {
  UniqueFd fd;
  FileSnapshot snapshot;
  fs::path path;
  std::string sha256;
};

struct LockedDocument final {
  LockedFile file;
  std::string document;
};

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic open_locked_file(
    const fs::path& path, const std::optional<std::uint64_t> exact_bytes,
    LockedFile& output) {
  output.path = path;
  output.fd = UniqueFd(::open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW));
  if (!output.fd) {
    return make_diagnostic(PrefillR1ProjectionPlaneV2ErrorCode::kOpenFailed,
                           path.string(),
                           "failed to open non-symlink source file", {}, {},
                           errno);
  }
  if (::flock(output.fd.get(), LOCK_SH | LOCK_NB) != 0) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication,
        path.string(), "source file is concurrently locked for mutation",
        {}, {}, errno);
  }
  int error = 0;
  struct stat status {};
  if (::fstat(output.fd.get(), &status) != 0 ||
      !capture_snapshot(output.fd.get(), output.snapshot, error) ||
      (status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0 ||
      status.st_uid != ::geteuid() || status.st_nlink != 1 ||
      (exact_bytes.has_value() && output.snapshot.size != *exact_bytes)) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication,
        path.string(),
        "source must be owner-held, read-only, singly linked, and exact-size",
        exact_bytes.has_value() ? std::to_string(*exact_bytes) : std::string{},
        std::to_string(output.snapshot.size), error != 0 ? error : errno);
  }
  return {};
}

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic hash_locked_file(
    LockedFile& file, std::string& digest) {
  try {
    constexpr std::size_t kChunkBytes = 8U * 1024U * 1024U;
    std::vector<std::uint8_t> buffer(static_cast<std::size_t>(
        std::min<std::uint64_t>(file.snapshot.size, kChunkBytes)));
    core::Sha256 hasher;
    std::uint64_t offset = 0U;
    while (offset < file.snapshot.size) {
      const std::size_t count = static_cast<std::size_t>(
          std::min<std::uint64_t>(buffer.size(), file.snapshot.size - offset));
      int error = 0;
      if (!pread_exact(file.fd.get(), buffer.data(), count, offset, error)) {
        return make_diagnostic(PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
                               file.path.string(),
                               "failed while hashing locked source", {}, {},
                               error);
      }
      if (!hasher.update(buffer.data(), count)) {
        return make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kArithmeticOverflow,
            file.path.string(), "SHA-256 input length overflowed");
      }
      offset += count;
    }
    digest = hasher.finalize().hex();
    return {};
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        file.path.string(), "hash buffer allocation failed");
  }
}

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic read_locked_document(
    const fs::path& path, const std::uint64_t maximum_bytes,
    LockedDocument& output) {
  auto diagnostic = open_locked_file(path, std::nullopt, output.file);
  if (!diagnostic) return diagnostic;
  if (output.file.snapshot.size == 0U ||
      output.file.snapshot.size > maximum_bytes ||
      output.file.snapshot.size > std::numeric_limits<std::size_t>::max()) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication,
        path.string(), "source document exceeds the bounded input limit");
  }
  try {
    output.document.resize(
        static_cast<std::size_t>(output.file.snapshot.size));
  } catch (const std::bad_alloc&) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        path.string(), "document allocation failed");
  }
  int error = 0;
  if (!pread_exact(output.file.fd.get(), output.document.data(),
                   output.document.size(), 0U, error)) {
    return make_diagnostic(PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
                           path.string(),
                           "failed to read complete locked document", {}, {},
                           error);
  }
  output.file.sha256 = sha256_text(output.document);
  if (!lower_sha256(output.file.sha256)) {
    return make_diagnostic(PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
                           path.string(),
                           "failed to hash exact document bytes");
  }
  return {};
}

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic rehash_unchanged(
    LockedFile& file, const std::string_view expected_sha256) {
  FileSnapshot current;
  int error = 0;
  if (!capture_snapshot(file.fd.get(), current, error) ||
      !same_snapshot(file.snapshot, current)) {
    return make_diagnostic(PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
                           file.path.string(),
                           "locked source changed during composition", {}, {},
                           error);
  }
  std::string digest;
  auto diagnostic = hash_locked_file(file, digest);
  if (!diagnostic) return diagnostic;
  if (digest != expected_sha256) {
    return make_diagnostic(PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
                           file.path.string(),
                           "locked source digest changed during composition",
                           std::string(expected_sha256), digest);
  }
  return {};
}

[[nodiscard]] fs::path normalized_existing_path(const fs::path& path,
                                                std::error_code& error) {
  return fs::canonical(path, error).lexically_normal();
}

[[nodiscard]] fs::path normalized_target_path(const fs::path& path,
                                              std::error_code& error) {
  const fs::path parent = path.parent_path().empty() ? fs::path(".")
                                                     : path.parent_path();
  const fs::path canonical_parent = fs::canonical(parent, error);
  if (error) return {};
  return (canonical_parent / path.filename()).lexically_normal();
}

[[nodiscard]] bool target_absent(
    const fs::path& path,
    PrefillR1ProjectionPlaneV2Diagnostic& diagnostic) {
  struct stat status {};
  if (::lstat(path.c_str(), &status) == 0) {
    diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kPublicationConflict,
        path.string(), "publication target already exists");
    return false;
  }
  if (errno != ENOENT) {
    diagnostic = make_diagnostic(PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
                                 path.string(),
                                 "failed to inspect publication target", {},
                                 {}, errno);
    return false;
  }
  return true;
}

void remove_if_present(const fs::path& path) noexcept {
  if (!path.empty()) (void)::unlink(path.c_str());
}

[[nodiscard]] UniqueFd create_temporary_file_near(
    const fs::path& target, const std::string_view tag, fs::path& path,
    PrefillR1ProjectionPlaneV2Diagnostic& diagnostic) {
  try {
    std::string pattern = target.string() + ".tmp." + std::string(tag) +
                          ".XXXXXX";
    std::vector<char> mutable_pattern(pattern.begin(), pattern.end());
    mutable_pattern.push_back('\0');
    const int fd = ::mkstemp(mutable_pattern.data());
    if (fd < 0) {
      diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kOpenFailed, target.string(),
          "failed to create output-adjacent temporary file", {}, {}, errno);
      return {};
    }
    UniqueFd result(fd);
    path = fs::path(mutable_pattern.data());
    const int flags = ::fcntl(fd, F_GETFD);
    if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) != 0 ||
        ::fchmod(fd, S_IRUSR | S_IWUSR) != 0 ||
        ::flock(fd, LOCK_EX | LOCK_NB) != 0) {
      diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure, path.string(),
          "failed to secure temporary output", {}, {}, errno);
      return {};
    }
    return result;
  } catch (const std::bad_alloc&) {
    diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        target.string(), "temporary pathname allocation failed");
    return {};
  }
}

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic seal_document(
    const int fd, const fs::path& path, const std::string_view document) {
  int error = 0;
  if (!pwrite_exact(fd, document.data(), document.size(), 0U, error) ||
      document.size() > static_cast<std::size_t>(
                            std::numeric_limits<off_t>::max()) ||
      ::ftruncate(fd, static_cast<off_t>(document.size())) != 0 ||
      ::fsync(fd) != 0 || ::fchmod(fd, S_IRUSR) != 0 || ::fsync(fd) != 0) {
    return make_diagnostic(PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
                           path.string(), "failed to seal output document",
                           {}, {}, error != 0 ? error : errno);
  }
  return {};
}

[[nodiscard]] bool all_distinct_paths(
    const std::vector<fs::path>& inputs,
    const std::array<fs::path, 4U>& outputs,
    PrefillR1ProjectionPlaneV2Diagnostic& diagnostic) {
  std::vector<fs::path> normalized;
  normalized.reserve(inputs.size() + outputs.size());
  for (const fs::path& input : inputs) {
    std::error_code error;
    fs::path path = normalized_existing_path(input, error);
    if (error) {
      diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath, input.string(),
          "failed to canonicalize input path", {}, error.message());
      return false;
    }
    normalized.emplace_back(std::move(path));
  }
  for (const fs::path& output : outputs) {
    std::error_code error;
    fs::path path = normalized_target_path(output, error);
    if (error) {
      diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath, output.string(),
          "failed to canonicalize output parent", {}, error.message());
      return false;
    }
    normalized.emplace_back(std::move(path));
  }
  for (std::size_t left = 0U; left < normalized.size(); ++left) {
    for (std::size_t right = left + 1U; right < normalized.size(); ++right) {
      if (normalized[left] == normalized[right]) {
        diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath,
            normalized[left].string(),
            "all publication inputs and outputs must be path-distinct");
        return false;
      }
    }
  }
  return true;
}

}  // namespace

std::string prefill_r1_projection_plane_v2_manifest_sha256(
    const PrefillR1ProjectionPlaneV2Manifest& manifest) {
  return sha256_text(manifest_document(manifest));
}

PrefillR1ProjectionPlaneV2ManifestResult
build_prefill_r1_projection_plane_v2_manifest(
    const PrefillMLPFactorizedLaneOverlayManifestBinding& mlp_v4,
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView& mlp_payload,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& attention_v4,
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView&
        attention_payload) {
  PrefillR1ProjectionPlaneV2ManifestResult result;
  try {
    const auto mlp_diagnostic =
        validate_prefill_mlp_factorized_lane_r1_manifest(mlp_v4);
    const auto attention_diagnostic =
        validate_prefill_attention_factorized_lane_r1_manifest(attention_v4);
    if (!mlp_diagnostic.ok() || !attention_diagnostic.ok() ||
        mlp_v4.lane_count != 1U || attention_v4.lane_count != 1U) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          "projection_plane_v2.source_manifest",
          "both inputs must be complete authenticated factorized R1 v4 "
          "manifests");
      return result;
    }
    if (mlp_v4.source_checkpoint_id !=
            attention_v4.source_checkpoint_id ||
        mlp_v4.source_config_sha256 !=
            attention_v4.source_config_sha256 ||
        mlp_v4.source_index_sha256 != attention_v4.source_index_sha256 ||
        !same_base(mlp_v4.required_base_k256,
                   attention_v4.required_base_k256)) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kSourceBindingMismatch,
          "projection_plane_v2.source_binding",
          "MLP and Attention v4 inputs do not bind the same checkpoint and "
          "K256 source publication");
      return result;
    }
    if (!valid_source_view(
            mlp_payload, kPrefillMLPFactorizedLaneOverlayVersionMajor,
            kPrefillMLPFactorizedLaneOverlayVersionMinor,
            kPrefillMLPFactorizedLaneOverlayLayout,
            kPrefillMLPFactorizedLaneR1PayloadBytes,
            mlp_v4.manifest_sha256) ||
        !valid_source_view(
            attention_payload,
            kPrefillAttentionFactorizedLaneOverlayVersionMajor,
            kPrefillAttentionFactorizedLaneOverlayVersionMinor,
            kPrefillAttentionFactorizedLaneOverlayLayout,
            kPrefillAttentionFactorizedLaneR1LayoutPlan.payload_bytes,
            attention_v4.manifest_sha256)) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication,
          "projection_plane_v2.source_payload",
          "source payload view is unauthenticated, incomplete, or does not "
          "match its v4 manifest");
      return result;
    }

    const auto mlp_plan =
        prefill_mlp_factorized_lane_overlay_layout_plan(1U);
    const auto attention_plan =
        prefill_attention_factorized_lane_overlay_layout_plan(1U);
    const auto paired_gate_up = gate_up_layout(
        kPrefillMLPFactorizedLaneGateUpOutputSize,
        kPrefillMLPFactorizedLaneGateUpInputSize);
    if (!mlp_plan || !attention_plan || !paired_gate_up.valid) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
          "projection_plane_v2.layout",
          "fixed R1 v2 layout plans are not representable");
      return result;
    }

    PrefillR1ProjectionPlaneV2Manifest manifest;
    manifest.physical_layout = std::string(kPrefillR1ProjectionPlaneV2Layout);
    manifest.source_checkpoint_id = mlp_v4.source_checkpoint_id;
    manifest.source_config_sha256 = mlp_v4.source_config_sha256;
    manifest.source_index_sha256 = mlp_v4.source_index_sha256;
    manifest.required_base_k256 =
        base_binding(mlp_v4.required_base_k256);
    manifest.mlp_v4 = source_binding(mlp_payload);
    manifest.attention_v4 = source_binding(attention_payload);
    manifest.projections.reserve(
        kPrefillR1ProjectionPlaneV2PhysicalProjectionCount);
    manifest.logical_projections.reserve(
        kPrefillR1ProjectionPlaneV2LogicalProjectionCount);

    std::uint64_t cursor = 0U;
    std::size_t attention_ordinal = 0U;
    for (std::uint32_t layer = 0U;
         layer < kPrefillMLPFactorizedLaneLayerCount; ++layer) {
      while (attention_ordinal < attention_v4.projections.size() &&
             attention_v4.projections[attention_ordinal].layer_index ==
                 layer) {
        const auto& source = attention_v4.projections[attention_ordinal];
        const auto* layout = attention_layout(attention_plan, source.family);
        if (layout == nullptr) {
          result.diagnostic = make_diagnostic(
              PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
              "projection_plane_v2.attention_layout",
              "Attention projection has no v2 adjacent-N8 plan");
          return result;
        }
        PrefillR1ProjectionPlaneV2PhysicalProjection entry;
        entry.physical_ordinal =
            static_cast<std::uint32_t>(manifest.projections.size());
        entry.layer_index = layer;
        entry.family = attention_family(source.family);
        entry.logical_projection_count = 1U;
        entry.source_primary_ordinal = source.ordinal;
        entry.source_primary_offset = source.payload_offset;
        entry.output_size = source.output_size;
        entry.input_size = source.input_size;
        entry.payload_offset = cursor;
        entry.payload_bytes = layout->projection_bytes;
        entry.packed_weight_offset = layout->packed_weight_offset;
        entry.packed_weight_bytes = layout->packed_weight_bytes;
        entry.weight_scale_offset = layout->weight_scale_offset;
        entry.weight_scale_bytes = layout->weight_scale_bytes;
        entry.primary_metadata_offset = layout->metadata_offset;
        entry.primary_metadata_bytes = layout->metadata_bytes;
        manifest.projections.push_back(entry);
        PrefillR1ProjectionPlaneV2LogicalProjection logical;
        logical.logical_ordinal =
            static_cast<std::uint32_t>(
                manifest.logical_projections.size());
        logical.layer_index = layer;
        logical.family = attention_logical_family(source.family);
        logical.physical_ordinal = entry.physical_ordinal;
        logical.source_ordinal = source.ordinal;
        logical.source_module = source.source_module;
        logical.source_sha256 = source.source_sha256;
        logical.output_size = source.output_size;
        logical.input_size = source.input_size;
        manifest.logical_projections.push_back(std::move(logical));
        if (!checked_add(cursor, entry.payload_bytes, cursor)) {
          result.diagnostic = make_diagnostic(
              PrefillR1ProjectionPlaneV2ErrorCode::kArithmeticOverflow,
              "projection_plane_v2.payload", "payload offset overflow");
          return result;
        }
        ++attention_ordinal;
      }

      const auto& gate = mlp_v4.projections[layer * 3U];
      const auto& up = mlp_v4.projections[layer * 3U + 1U];
      PrefillR1ProjectionPlaneV2PhysicalProjection gate_up;
      gate_up.physical_ordinal =
          static_cast<std::uint32_t>(manifest.projections.size());
      gate_up.layer_index = layer;
      gate_up.family =
          PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpGateUp;
      gate_up.logical_projection_count = 2U;
      gate_up.source_primary_ordinal = gate.ordinal;
      gate_up.source_secondary_ordinal = up.ordinal;
      gate_up.source_primary_offset = gate.payload_offset;
      gate_up.source_secondary_offset = up.payload_offset;
      gate_up.output_size = gate.output_size;
      gate_up.input_size = gate.input_size;
      gate_up.payload_offset = cursor;
      gate_up.payload_bytes = paired_gate_up.projection_bytes;
      gate_up.packed_weight_offset = paired_gate_up.packed_weight_offset;
      gate_up.packed_weight_bytes = paired_gate_up.packed_weight_bytes;
      gate_up.weight_scale_offset = paired_gate_up.weight_scale_offset;
      gate_up.weight_scale_bytes = paired_gate_up.weight_scale_bytes;
      gate_up.primary_metadata_offset = paired_gate_up.gate_metadata_offset;
      gate_up.primary_metadata_bytes = paired_gate_up.gate_metadata_bytes;
      gate_up.secondary_metadata_offset = paired_gate_up.up_metadata_offset;
      gate_up.secondary_metadata_bytes = paired_gate_up.up_metadata_bytes;
      manifest.projections.push_back(gate_up);
      for (std::uint32_t branch = 0U; branch < 2U; ++branch) {
        const auto& source = branch == 0U ? gate : up;
        PrefillR1ProjectionPlaneV2LogicalProjection logical;
        logical.logical_ordinal =
            static_cast<std::uint32_t>(
                manifest.logical_projections.size());
        logical.layer_index = layer;
        logical.family =
            branch == 0U
                ? PrefillR1ProjectionPlaneV2LogicalFamily::kMlpGate
                : PrefillR1ProjectionPlaneV2LogicalFamily::kMlpUp;
        logical.physical_ordinal = gate_up.physical_ordinal;
        logical.source_ordinal = source.ordinal;
        logical.source_module = source.source_module;
        logical.source_sha256 = source.source_sha256;
        logical.output_size = source.output_size;
        logical.input_size = source.input_size;
        manifest.logical_projections.push_back(std::move(logical));
      }
      if (!checked_add(cursor, gate_up.payload_bytes, cursor)) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kArithmeticOverflow,
            "projection_plane_v2.payload", "payload offset overflow");
        return result;
      }

      const auto& down = mlp_v4.projections[layer * 3U + 2U];
      PrefillR1ProjectionPlaneV2PhysicalProjection down_entry;
      down_entry.physical_ordinal =
          static_cast<std::uint32_t>(manifest.projections.size());
      down_entry.layer_index = layer;
      down_entry.family =
          PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpDown;
      down_entry.logical_projection_count = 1U;
      down_entry.source_primary_ordinal = down.ordinal;
      down_entry.source_primary_offset = down.payload_offset;
      down_entry.output_size = down.output_size;
      down_entry.input_size = down.input_size;
      down_entry.payload_offset = cursor;
      down_entry.payload_bytes = mlp_plan.down.projection_bytes;
      down_entry.packed_weight_offset = mlp_plan.down.packed_weight_offset;
      down_entry.packed_weight_bytes = mlp_plan.down.packed_weight_bytes;
      down_entry.weight_scale_offset = mlp_plan.down.weight_scale_offset;
      down_entry.weight_scale_bytes = mlp_plan.down.weight_scale_bytes;
      down_entry.primary_metadata_offset = mlp_plan.down.metadata_offset;
      down_entry.primary_metadata_bytes = mlp_plan.down.metadata_bytes;
      manifest.projections.push_back(down_entry);
      PrefillR1ProjectionPlaneV2LogicalProjection down_logical;
      down_logical.logical_ordinal =
          static_cast<std::uint32_t>(
              manifest.logical_projections.size());
      down_logical.layer_index = layer;
      down_logical.family =
          PrefillR1ProjectionPlaneV2LogicalFamily::kMlpDown;
      down_logical.physical_ordinal = down_entry.physical_ordinal;
      down_logical.source_ordinal = down.ordinal;
      down_logical.source_module = down.source_module;
      down_logical.source_sha256 = down.source_sha256;
      down_logical.output_size = down.output_size;
      down_logical.input_size = down.input_size;
      manifest.logical_projections.push_back(std::move(down_logical));
      if (!checked_add(cursor, down_entry.payload_bytes, cursor)) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kArithmeticOverflow,
            "projection_plane_v2.payload", "payload offset overflow");
        return result;
      }
    }
    manifest.logical_projection_count =
        kPrefillR1ProjectionPlaneV2LogicalProjectionCount;
    manifest.physical_projection_count =
        static_cast<std::uint32_t>(manifest.projections.size());
    manifest.payload_bytes = cursor;
    manifest.manifest_sha256 =
        prefill_r1_projection_plane_v2_manifest_sha256(manifest);

    result.diagnostic =
        validate_prefill_r1_projection_plane_v2_manifest(manifest);
    if (!result.diagnostic) {
      return result;
    }
    result.canonical_document = manifest_document(manifest);
    result.value = std::move(manifest);
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.manifest", "manifest allocation failed");
    return result;
  }
}

PrefillR1ProjectionPlaneV2ManifestResult
parse_prefill_r1_projection_plane_v2_manifest(
    const std::string_view document) {
  PrefillR1ProjectionPlaneV2ManifestResult result;
  try {
    json::ParseOptions options;
    options.max_input_bytes = 4U * 1024U * 1024U;
    options.max_nesting_depth = 12U;
    // The canonical inventory contains 400 logical entries plus 336 physical
    // entries.  Each physical entry owns six bounded coordinate arrays, so
    // the parser's aggregate value/item accounting legitimately exceeds
    // 12k even though the document remains well below the 4 MiB byte gate.
    options.max_values = 32'000U;
    options.max_container_items = 32'000U;
    const json::ParseResult parsed = json::parse(document, options);
    const auto* const root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(
            *root,
            {"attention_v4", "lane_count", "logical_projection_count",
             "logical_projections", "mlp_v4", "payload_bytes",
             "physical_layout", "physical_projection_count", "projections",
             "required_base_k256", "schema", "source_checkpoint_id",
             "source_config_sha256", "source_index_sha256", "version"})) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          "projection_plane_v2.manifest",
          "strict manifest JSON schema mismatch");
      return result;
    }

    PrefillR1ProjectionPlaneV2Manifest manifest;
    std::string schema;
    std::uint64_t lane_count = 0U;
    std::uint64_t logical_count = 0U;
    std::uint64_t physical_count = 0U;
    if (!parse_source(root->at("attention_v4"), manifest.attention_v4) ||
        !json_uint(*root, "lane_count", lane_count) ||
        !assign_u32(lane_count, manifest.lane_count) ||
        !json_uint(*root, "logical_projection_count", logical_count) ||
        !assign_u32(logical_count, manifest.logical_projection_count) ||
        !parse_source(root->at("mlp_v4"), manifest.mlp_v4) ||
        !json_uint(*root, "payload_bytes", manifest.payload_bytes) ||
        !json_string(*root, "physical_layout", manifest.physical_layout) ||
        !json_uint(*root, "physical_projection_count", physical_count) ||
        !assign_u32(physical_count, manifest.physical_projection_count) ||
        !parse_base(root->at("required_base_k256"),
                    manifest.required_base_k256) ||
        !json_string(*root, "schema", schema) || schema != kManifestSchema ||
        !json_string(*root, "source_checkpoint_id",
                     manifest.source_checkpoint_id) ||
        !json_string(*root, "source_config_sha256",
                     manifest.source_config_sha256) ||
        !json_string(*root, "source_index_sha256",
                     manifest.source_index_sha256) ||
        !parse_version(root->at("version"), manifest.version_major,
                       manifest.version_minor)) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          "projection_plane_v2.manifest.identity",
          "manifest identity fields are invalid");
      return result;
    }

    const auto* const logical = root->at("logical_projections").as_array();
    if (logical == nullptr || logical->size() != logical_count) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          "projection_plane_v2.manifest.logical_projections",
          "logical projection inventory size mismatch");
      return result;
    }
    manifest.logical_projections.reserve(logical->size());
    for (std::size_t index = 0U; index < logical->size(); ++index) {
      const auto* const object = (*logical)[index].as_object();
      PrefillR1ProjectionPlaneV2LogicalProjection entry;
      std::string family;
      std::uint64_t layer = 0U;
      std::uint64_t ordinal = 0U;
      std::uint64_t physical_ordinal = 0U;
      std::uint64_t source_ordinal = 0U;
      if (object == nullptr ||
          !exact_keys(*object,
                      {"family", "input_size", "layer", "logical_ordinal",
                       "output_size", "physical_ordinal", "source_module",
                       "source_ordinal", "source_sha256"}) ||
          !json_string(*object, "family", family) ||
          !json_uint(*object, "input_size", entry.input_size) ||
          !json_uint(*object, "layer", layer) ||
          !assign_u32(layer, entry.layer_index) ||
          !json_uint(*object, "logical_ordinal", ordinal) ||
          !assign_u32(ordinal, entry.logical_ordinal) ||
          !json_uint(*object, "output_size", entry.output_size) ||
          !json_uint(*object, "physical_ordinal", physical_ordinal) ||
          !assign_u32(physical_ordinal, entry.physical_ordinal) ||
          !json_string(*object, "source_module", entry.source_module) ||
          !json_uint(*object, "source_ordinal", source_ordinal) ||
          !assign_u32(source_ordinal, entry.source_ordinal) ||
          !json_string(*object, "source_sha256", entry.source_sha256)) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
            "projection_plane_v2.manifest.logical_projections[" +
                std::to_string(index) + "]",
            "strict logical projection schema mismatch");
        return result;
      }
      const auto parsed_family = parse_logical_family(family);
      if (!parsed_family.has_value()) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
            "projection_plane_v2.manifest.logical_projections[" +
                std::to_string(index) + "].family",
            "unknown logical projection family");
        return result;
      }
      entry.family = *parsed_family;
      manifest.logical_projections.emplace_back(std::move(entry));
    }

    const auto* const physical = root->at("projections").as_array();
    if (physical == nullptr || physical->size() != physical_count) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          "projection_plane_v2.manifest.projections",
          "physical projection inventory size mismatch");
      return result;
    }
    manifest.projections.reserve(physical->size());
    for (std::size_t index = 0U; index < physical->size(); ++index) {
      const auto* const object = (*physical)[index].as_object();
      PrefillR1ProjectionPlaneV2PhysicalProjection entry;
      std::string family;
      std::uint64_t layer = 0U;
      std::uint64_t logical_projection_count = 0U;
      std::uint64_t ordinal = 0U;
      std::uint64_t metadata[4]{};
      std::uint64_t packed[2]{};
      std::uint64_t payload[2]{};
      std::uint64_t scale[2]{};
      std::uint64_t source_ordinals[2]{};
      std::uint64_t source_offsets[2]{};
      if (object == nullptr ||
          !exact_keys(*object,
                      {"family", "input_size", "layer", "logical_count",
                       "metadata", "ordinal", "output_size", "packed",
                       "payload", "scale", "source_ordinals",
                       "source_offsets"}) ||
          !json_string(*object, "family", family) ||
          !json_uint(*object, "input_size", entry.input_size) ||
          !json_uint(*object, "layer", layer) ||
          !assign_u32(layer, entry.layer_index) ||
          !json_uint(*object, "logical_count", logical_projection_count) ||
          !assign_u32(logical_projection_count,
                      entry.logical_projection_count) ||
          !parse_u64_array(object->at("metadata"), 4U, metadata) ||
          !json_uint(*object, "ordinal", ordinal) ||
          !assign_u32(ordinal, entry.physical_ordinal) ||
          !json_uint(*object, "output_size", entry.output_size) ||
          !parse_u64_array(object->at("packed"), 2U, packed) ||
          !parse_u64_array(object->at("payload"), 2U, payload) ||
          !parse_u64_array(object->at("scale"), 2U, scale) ||
          !parse_u64_array(object->at("source_ordinals"), 2U,
                           source_ordinals) ||
          !assign_u32(source_ordinals[0U], entry.source_primary_ordinal) ||
          !assign_u32(source_ordinals[1U], entry.source_secondary_ordinal) ||
          !parse_u64_array(object->at("source_offsets"), 2U,
                           source_offsets)) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
            "projection_plane_v2.manifest.projections[" +
                std::to_string(index) + "]",
            "strict physical projection schema mismatch");
        return result;
      }
      const auto parsed_family = parse_physical_family(family);
      if (!parsed_family.has_value()) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
            "projection_plane_v2.manifest.projections[" +
                std::to_string(index) + "].family",
            "unknown physical projection family");
        return result;
      }
      entry.family = *parsed_family;
      entry.primary_metadata_offset = metadata[0U];
      entry.primary_metadata_bytes = metadata[1U];
      entry.secondary_metadata_offset = metadata[2U];
      entry.secondary_metadata_bytes = metadata[3U];
      entry.packed_weight_offset = packed[0U];
      entry.packed_weight_bytes = packed[1U];
      entry.payload_offset = payload[0U];
      entry.payload_bytes = payload[1U];
      entry.weight_scale_offset = scale[0U];
      entry.weight_scale_bytes = scale[1U];
      entry.source_primary_offset = source_offsets[0U];
      entry.source_secondary_offset = source_offsets[1U];
      manifest.projections.emplace_back(std::move(entry));
    }

    manifest.manifest_sha256 = sha256_text(document);
    if (manifest.manifest_sha256.empty()) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
          "projection_plane_v2.manifest",
          "failed to hash strict manifest document");
      return result;
    }
    result.diagnostic =
        validate_prefill_r1_projection_plane_v2_manifest(manifest);
    if (!result.diagnostic) return result;
    result.canonical_document = manifest_document(manifest);
    if (result.canonical_document != document) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          "projection_plane_v2.manifest.canonical",
          "manifest must use the canonical field order and spelling");
      return result;
    }
    result.value.emplace(std::move(manifest));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.manifest", "manifest parser allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
        "projection_plane_v2.manifest",
        "unexpected strict manifest parse failure");
    return result;
  }
}

PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_manifest(
    const PrefillR1ProjectionPlaneV2Manifest& manifest) {
  const auto mlp_plan =
      prefill_mlp_factorized_lane_overlay_layout_plan(1U);
  const auto attention_plan =
      prefill_attention_factorized_lane_overlay_layout_plan(1U);
  const auto paired_gate_up = gate_up_layout(
      kPrefillMLPFactorizedLaneGateUpOutputSize,
      kPrefillMLPFactorizedLaneGateUpInputSize);
  if (manifest.version_major !=
          kPrefillR1ProjectionPlaneV2VersionMajor ||
      manifest.version_minor !=
          kPrefillR1ProjectionPlaneV2VersionMinor ||
      manifest.physical_layout != kPrefillR1ProjectionPlaneV2Layout ||
      manifest.lane_count != 1U || !lower_sha256(manifest.source_config_sha256) ||
      !lower_sha256(manifest.source_index_sha256) ||
      manifest.source_checkpoint_id.empty() || !mlp_plan ||
      !attention_plan || !paired_gate_up.valid) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
        "projection_plane_v2.manifest.header",
        "manifest header or fixed layout plan is invalid");
  }
  if (!valid_base(manifest.required_base_k256)) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kSourceBindingMismatch,
        "projection_plane_v2.manifest.required_base_k256",
        "unified v2 manifest does not bind one authenticated K256 base");
  }
  if (!valid_source_binding(
          manifest.mlp_v4,
          kPrefillMLPFactorizedLaneOverlayVersionMajor,
          kPrefillMLPFactorizedLaneOverlayVersionMinor,
          kPrefillMLPFactorizedLaneOverlayLayout,
          kPrefillMLPFactorizedLaneR1PayloadBytes) ||
      !valid_source_binding(
          manifest.attention_v4,
          kPrefillAttentionFactorizedLaneOverlayVersionMajor,
          kPrefillAttentionFactorizedLaneOverlayVersionMinor,
          kPrefillAttentionFactorizedLaneOverlayLayout,
          kPrefillAttentionFactorizedLaneR1LayoutPlan.payload_bytes)) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication,
        "projection_plane_v2.manifest.sources",
        "manifest source bindings are not exact R1 v4 publications");
  }
  if (manifest.projections.size() !=
          kPrefillR1ProjectionPlaneV2PhysicalProjectionCount ||
      manifest.logical_projections.size() !=
          kPrefillR1ProjectionPlaneV2LogicalProjectionCount ||
      manifest.logical_projection_count !=
          kPrefillR1ProjectionPlaneV2LogicalProjectionCount ||
      manifest.physical_projection_count !=
          kPrefillR1ProjectionPlaneV2PhysicalProjectionCount ||
      manifest.payload_bytes != kPrefillR1ProjectionPlaneV2PayloadBytes) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
        "projection_plane_v2.manifest.inventory",
        "manifest is not the complete 400-logical/336-physical plane");
  }

  std::size_t physical = 0U;
  std::size_t logical = 0U;
  std::uint32_t attention_ordinal = 0U;
  std::uint64_t cursor = 0U;
  std::uint32_t logical_count = 0U;
  for (std::uint32_t layer = 0U;
       layer < kPrefillMLPFactorizedLaneLayerCount; ++layer) {
    const bool full =
        prefill_attention_factorized_lane_is_full_layer(layer);
    const std::array<PrefillAttentionFactorizedLaneProjectionFamily, 4U>
        full_families = {
            PrefillAttentionFactorizedLaneProjectionFamily::kFullQ,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullK,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullV,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullO,
        };
    const std::array<PrefillAttentionFactorizedLaneProjectionFamily, 3U>
        linear_families = {
            PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv,
            PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ,
            PrefillAttentionFactorizedLaneProjectionFamily::kLinearO,
        };
    const std::size_t attention_count = full ? 4U : 3U;
    for (std::size_t position = 0U; position < attention_count;
         ++position, ++physical, ++attention_ordinal) {
      const auto source_family =
          full ? full_families[position] : linear_families[position];
      const auto* layout = attention_layout(attention_plan, source_family);
      const auto& entry = manifest.projections[physical];
      const std::uint64_t expected_source_offset =
          prefill_attention_factorized_lane_projection_absolute_offset(
              attention_plan, layer, source_family);
      if (layout == nullptr || entry.physical_ordinal != physical ||
          entry.layer_index != layer ||
          entry.family != attention_family(source_family) ||
          entry.logical_projection_count != 1U ||
          entry.source_primary_ordinal != attention_ordinal ||
          entry.source_secondary_ordinal !=
              std::numeric_limits<std::uint32_t>::max() ||
          entry.source_primary_offset != expected_source_offset ||
          entry.source_secondary_offset != 0U ||
          entry.output_size != layout->output_size ||
          entry.input_size != layout->input_size ||
          entry.payload_offset != cursor ||
          entry.payload_bytes != layout->projection_bytes ||
          entry.packed_weight_offset != layout->packed_weight_offset ||
          entry.packed_weight_bytes != layout->packed_weight_bytes ||
          entry.weight_scale_offset != layout->weight_scale_offset ||
          entry.weight_scale_bytes != layout->weight_scale_bytes ||
          entry.primary_metadata_offset != layout->metadata_offset ||
          entry.primary_metadata_bytes != layout->metadata_bytes ||
          entry.secondary_metadata_offset != 0U ||
          entry.secondary_metadata_bytes != 0U) {
        return make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
            "projection_plane_v2.manifest.projections[" +
                std::to_string(physical) + "]",
            "Attention physical entry differs from the fixed adjacent-N8 "
            "layout");
      }
      const auto& logical_entry = manifest.logical_projections[logical];
      if (logical_entry.logical_ordinal != logical ||
          logical_entry.layer_index != layer ||
          logical_entry.family != attention_logical_family(source_family) ||
          logical_entry.physical_ordinal != physical ||
          logical_entry.source_ordinal != attention_ordinal ||
          logical_entry.source_module !=
              expected_attention_module(layer, source_family) ||
          !lower_sha256(logical_entry.source_sha256) ||
          logical_entry.output_size != layout->output_size ||
          logical_entry.input_size != layout->input_size) {
        return make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
            "projection_plane_v2.manifest.logical_projections[" +
                std::to_string(logical) + "]",
            "Attention logical identity differs from the fixed model "
            "inventory");
      }
      cursor += entry.payload_bytes;
      logical_count += 1U;
      ++logical;
    }

    const auto& gate_up = manifest.projections[physical++];
    const std::uint64_t gate_offset =
        prefill_mlp_factorized_lane_projection_absolute_offset(
            mlp_plan, layer,
            PrefillMLPFactorizedLaneProjectionFamily::kGate);
    const std::uint64_t up_offset =
        prefill_mlp_factorized_lane_projection_absolute_offset(
            mlp_plan, layer,
            PrefillMLPFactorizedLaneProjectionFamily::kUp);
    if (gate_up.physical_ordinal != physical - 1U ||
        gate_up.layer_index != layer ||
        gate_up.family !=
            PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpGateUp ||
        gate_up.logical_projection_count != 2U ||
        gate_up.source_primary_ordinal != layer * 3U ||
        gate_up.source_secondary_ordinal != layer * 3U + 1U ||
        gate_up.source_primary_offset != gate_offset ||
        gate_up.source_secondary_offset != up_offset ||
        gate_up.output_size !=
            kPrefillMLPFactorizedLaneGateUpOutputSize ||
        gate_up.input_size != kPrefillMLPFactorizedLaneGateUpInputSize ||
        gate_up.payload_offset != cursor ||
        gate_up.payload_bytes != paired_gate_up.projection_bytes ||
        gate_up.packed_weight_offset !=
            paired_gate_up.packed_weight_offset ||
        gate_up.packed_weight_bytes != paired_gate_up.packed_weight_bytes ||
        gate_up.weight_scale_offset != paired_gate_up.weight_scale_offset ||
        gate_up.weight_scale_bytes != paired_gate_up.weight_scale_bytes ||
        gate_up.primary_metadata_offset !=
            paired_gate_up.gate_metadata_offset ||
        gate_up.primary_metadata_bytes !=
            paired_gate_up.gate_metadata_bytes ||
        gate_up.secondary_metadata_offset !=
            paired_gate_up.up_metadata_offset ||
        gate_up.secondary_metadata_bytes !=
            paired_gate_up.up_metadata_bytes) {
      return make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
          "projection_plane_v2.manifest.gate_up[" +
              std::to_string(layer) + "]",
          "Gate+Up physical entry differs from the fixed fragment-native "
          "pair layout");
    }
    for (std::uint32_t branch = 0U; branch < 2U; ++branch) {
      const auto expected_family =
          branch == 0U
              ? PrefillR1ProjectionPlaneV2LogicalFamily::kMlpGate
              : PrefillR1ProjectionPlaneV2LogicalFamily::kMlpUp;
      const auto& logical_entry = manifest.logical_projections[logical];
      if (logical_entry.logical_ordinal != logical ||
          logical_entry.layer_index != layer ||
          logical_entry.family != expected_family ||
          logical_entry.physical_ordinal != physical - 1U ||
          logical_entry.source_ordinal != layer * 3U + branch ||
          logical_entry.source_module !=
              expected_mlp_module(layer, expected_family) ||
          !lower_sha256(logical_entry.source_sha256) ||
          logical_entry.output_size !=
              kPrefillMLPFactorizedLaneGateUpOutputSize ||
          logical_entry.input_size !=
              kPrefillMLPFactorizedLaneGateUpInputSize) {
        return make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
            "projection_plane_v2.manifest.logical_projections[" +
                std::to_string(logical) + "]",
            "Gate or Up logical identity differs from the fixed model "
            "inventory");
      }
      ++logical;
    }
    cursor += gate_up.payload_bytes;
    logical_count += 2U;

    const auto& down = manifest.projections[physical++];
    const std::uint64_t down_offset =
        prefill_mlp_factorized_lane_projection_absolute_offset(
            mlp_plan, layer,
            PrefillMLPFactorizedLaneProjectionFamily::kDown);
    if (down.physical_ordinal != physical - 1U ||
        down.layer_index != layer ||
        down.family !=
            PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpDown ||
        down.logical_projection_count != 1U ||
        down.source_primary_ordinal != layer * 3U + 2U ||
        down.source_secondary_ordinal !=
            std::numeric_limits<std::uint32_t>::max() ||
        down.source_primary_offset != down_offset ||
        down.source_secondary_offset != 0U ||
        down.output_size != mlp_plan.down.output_size ||
        down.input_size != mlp_plan.down.input_size ||
        down.payload_offset != cursor ||
        down.payload_bytes != mlp_plan.down.projection_bytes ||
        down.packed_weight_offset != mlp_plan.down.packed_weight_offset ||
        down.packed_weight_bytes != mlp_plan.down.packed_weight_bytes ||
        down.weight_scale_offset != mlp_plan.down.weight_scale_offset ||
        down.weight_scale_bytes != mlp_plan.down.weight_scale_bytes ||
        down.primary_metadata_offset != mlp_plan.down.metadata_offset ||
        down.primary_metadata_bytes != mlp_plan.down.metadata_bytes ||
        down.secondary_metadata_offset != 0U ||
        down.secondary_metadata_bytes != 0U) {
      return make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
          "projection_plane_v2.manifest.down[" + std::to_string(layer) +
              "]",
          "Down physical entry differs from the fixed adjacent-N8 layout");
    }
    const auto& down_logical = manifest.logical_projections[logical];
    if (down_logical.logical_ordinal != logical ||
        down_logical.layer_index != layer ||
        down_logical.family !=
            PrefillR1ProjectionPlaneV2LogicalFamily::kMlpDown ||
        down_logical.physical_ordinal != physical - 1U ||
        down_logical.source_ordinal != layer * 3U + 2U ||
        down_logical.source_module != expected_mlp_module(
                                          layer,
                                          PrefillR1ProjectionPlaneV2LogicalFamily::
                                              kMlpDown) ||
        !lower_sha256(down_logical.source_sha256) ||
        down_logical.output_size != mlp_plan.down.output_size ||
        down_logical.input_size != mlp_plan.down.input_size) {
      return make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
          "projection_plane_v2.manifest.logical_projections[" +
              std::to_string(logical) + "]",
          "Down logical identity differs from the fixed model inventory");
    }
    ++logical;
    cursor += down.payload_bytes;
    logical_count += 1U;
  }
  if (physical != manifest.projections.size() ||
      logical != manifest.logical_projections.size() ||
      attention_ordinal !=
          kPrefillAttentionFactorizedLaneProjectionCount ||
      logical_count != kPrefillR1ProjectionPlaneV2LogicalProjectionCount ||
      cursor != kPrefillR1ProjectionPlaneV2PayloadBytes) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
        "projection_plane_v2.manifest.coverage",
        "physical inventory does not close the complete projection plane");
  }
  const std::string digest =
      prefill_r1_projection_plane_v2_manifest_sha256(manifest);
  if (digest.empty() || digest != manifest.manifest_sha256) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
        "projection_plane_v2.manifest.manifest_sha256",
        "manifest body digest mismatch", digest, manifest.manifest_sha256);
  }
  return {};
}

PrefillR1ProjectionPlaneV2PolicyResult
build_prefill_r1_projection_plane_v2_policy(
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillMLPFactorizedLaneR1Policy& mlp_v4_policy,
    const PrefillAttentionFactorizedLaneR1Policy& attention_v4_policy) {
  PrefillR1ProjectionPlaneV2PolicyResult result;
  result.diagnostic =
      validate_prefill_r1_projection_plane_v2_manifest(manifest);
  if (!result.diagnostic) {
    return result;
  }
  try {
    PrefillR1ProjectionPlaneV2Policy policy;
    policy.physical_layout = manifest.physical_layout;
    policy.manifest_sha256 = manifest.manifest_sha256;
    policy.mlp_source_policy_sha256 = manifest.mlp_v4.policy_sha256;
    policy.attention_source_policy_sha256 =
        manifest.attention_v4.policy_sha256;
    policy.converter_abi = std::string(kConverterAbi);
    policy.mode = std::string(kMode);
    policy.projections.reserve(
        kPrefillR1ProjectionPlaneV2LogicalProjectionCount);
    for (const auto& logical : manifest.logical_projections) {
      const bool mlp =
          logical.family >=
          PrefillR1ProjectionPlaneV2LogicalFamily::kMlpGate;
      PrefillR1ProjectionPlaneV2Policy::Calibration entry;
      entry.logical_ordinal = logical.logical_ordinal;
      entry.physical_ordinal = logical.physical_ordinal;
      entry.source_ordinal = logical.source_ordinal;
      entry.source_module = logical.source_module;
      entry.source_sha256 = logical.source_sha256;
      if (mlp) {
        if (logical.source_ordinal >=
            mlp_v4_policy.binding.projections.size()) {
          result.diagnostic = make_diagnostic(
              PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
              "projection_plane_v2.policy.mlp_source",
              "MLP source policy inventory is incomplete");
          return result;
        }
        const auto& source =
            mlp_v4_policy.binding.projections[logical.source_ordinal];
        entry.weight_clip_ratio = source.weight_clip_ratio;
        entry.activation_clip_ratio = source.activation_clip_ratio;
        entry.factor_scheme = source.factor_source.scheme;
        entry.factor_path = source.factor_source.path;
        entry.factor_sha256 = source.factor_source.sha256;
        entry.factor_element_count = source.factor_source.element_count;
      } else {
        if (logical.source_ordinal >=
            attention_v4_policy.binding.projections.size()) {
          result.diagnostic = make_diagnostic(
              PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
              "projection_plane_v2.policy.attention_source",
              "Attention source policy inventory is incomplete");
          return result;
        }
        const auto& source =
            attention_v4_policy.binding.projections[logical.source_ordinal];
        entry.weight_clip_ratio = source.weight_clip_ratio;
        entry.activation_clip_ratio = source.activation_clip_ratio;
        entry.factor_scheme = source.factor_source.scheme;
        entry.factor_path = source.factor_source.path;
        entry.factor_sha256 = source.factor_source.sha256;
        entry.factor_element_count = source.factor_source.element_count;
      }
      policy.projections.push_back(std::move(entry));
    }
    result.canonical_document = policy_document(policy);
    policy.policy_sha256 = sha256_text(result.canonical_document);
    policy.policy_bytes = result.canonical_document.size();
    result.diagnostic =
        validate_prefill_r1_projection_plane_v2_policy(policy, manifest);
    if (result.diagnostic) {
      result.diagnostic =
          validate_prefill_r1_projection_plane_v2_policy_sources(
              policy, manifest, mlp_v4_policy, attention_v4_policy);
    }
    if (result.diagnostic) {
      result.value = std::move(policy);
    }
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.policy", "policy allocation failed");
    return result;
  }
}

PrefillR1ProjectionPlaneV2PolicyResult
parse_prefill_r1_projection_plane_v2_policy(
    const std::string_view document,
    const PrefillR1ProjectionPlaneV2Manifest& manifest) {
  PrefillR1ProjectionPlaneV2PolicyResult result;
  result.diagnostic =
      validate_prefill_r1_projection_plane_v2_manifest(manifest);
  if (!result.diagnostic) return result;
  try {
    json::ParseOptions options;
    options.max_input_bytes = 4U * 1024U * 1024U;
    options.max_nesting_depth = 12U;
    options.max_values = 8'000U;
    options.max_container_items = 8'000U;
    const json::ParseResult parsed = json::parse(document, options);
    const auto* const root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(
            *root,
            {"attention_source_policy_sha256", "converter_abi",
             "manifest_sha256", "mlp_source_policy_sha256", "mode",
             "performance_upper_bound_only", "physical_layout",
             "production_residency_eligible", "projections",
             "quality_production_eligible", "schema", "version"})) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
          "projection_plane_v2.policy", "strict policy JSON schema mismatch");
      return result;
    }

    PrefillR1ProjectionPlaneV2Policy policy;
    std::string schema;
    if (!json_string(*root, "attention_source_policy_sha256",
                     policy.attention_source_policy_sha256) ||
        !json_string(*root, "converter_abi", policy.converter_abi) ||
        !json_string(*root, "manifest_sha256", policy.manifest_sha256) ||
        !json_string(*root, "mlp_source_policy_sha256",
                     policy.mlp_source_policy_sha256) ||
        !json_string(*root, "mode", policy.mode) ||
        !json_bool(*root, "performance_upper_bound_only",
                   policy.performance_upper_bound_only) ||
        !json_string(*root, "physical_layout", policy.physical_layout) ||
        !json_bool(*root, "production_residency_eligible",
                   policy.production_residency_eligible) ||
        !json_bool(*root, "quality_production_eligible",
                   policy.quality_production_eligible) ||
        !json_string(*root, "schema", schema) || schema != kPolicySchema ||
        !parse_version(root->at("version"), policy.version_major,
                       policy.version_minor)) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
          "projection_plane_v2.policy.identity",
          "policy identity fields are invalid");
      return result;
    }

    const auto* const projections = root->at("projections").as_array();
    if (projections == nullptr ||
        projections->size() != manifest.logical_projections.size()) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
          "projection_plane_v2.policy.projections",
          "policy must cover every logical projection");
      return result;
    }
    policy.projections.reserve(projections->size());
    for (std::size_t index = 0U; index < projections->size(); ++index) {
      const auto* const object = (*projections)[index].as_object();
      PrefillR1ProjectionPlaneV2Policy::Calibration entry;
      std::uint64_t logical_ordinal = 0U;
      std::uint64_t physical_ordinal = 0U;
      std::uint64_t source_ordinal = 0U;
      if (object == nullptr ||
          !exact_keys(*object,
                      {"activation_clip_ratio", "factor",
                       "logical_ordinal", "physical_ordinal",
                       "source_module", "source_ordinal", "source_sha256",
                       "weight_clip_ratio"}) ||
          !json_double(*object, "activation_clip_ratio",
                       entry.activation_clip_ratio) ||
          !json_uint(*object, "logical_ordinal", logical_ordinal) ||
          !assign_u32(logical_ordinal, entry.logical_ordinal) ||
          !json_uint(*object, "physical_ordinal", physical_ordinal) ||
          !assign_u32(physical_ordinal, entry.physical_ordinal) ||
          !json_string(*object, "source_module", entry.source_module) ||
          !json_uint(*object, "source_ordinal", source_ordinal) ||
          !assign_u32(source_ordinal, entry.source_ordinal) ||
          !json_string(*object, "source_sha256", entry.source_sha256) ||
          !json_double(*object, "weight_clip_ratio",
                       entry.weight_clip_ratio)) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
            "projection_plane_v2.policy.projections[" +
                std::to_string(index) + "]",
            "strict logical calibration schema mismatch");
        return result;
      }
      const auto* const factor = object->at("factor").as_object();
      if (factor == nullptr ||
          !exact_keys(*factor, {"element_count", "path", "scheme", "sha256"}) ||
          !json_uint(*factor, "element_count",
                     entry.factor_element_count) ||
          !json_string(*factor, "path", entry.factor_path) ||
          !json_string(*factor, "scheme", entry.factor_scheme) ||
          !json_string(*factor, "sha256", entry.factor_sha256)) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
            "projection_plane_v2.policy.projections[" +
                std::to_string(index) + "].factor",
            "strict factor schema mismatch");
        return result;
      }
      policy.projections.emplace_back(std::move(entry));
    }

    policy.policy_sha256 = sha256_text(document);
    policy.policy_bytes = document.size();
    if (policy.policy_sha256.empty()) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
          "projection_plane_v2.policy", "failed to hash strict policy");
      return result;
    }
    result.diagnostic =
        validate_prefill_r1_projection_plane_v2_policy(policy, manifest);
    if (!result.diagnostic) return result;
    result.canonical_document = policy_document(policy);
    if (result.canonical_document != document) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
          "projection_plane_v2.policy.canonical",
          "policy must use the canonical field order and number spelling");
      return result;
    }
    result.value.emplace(std::move(policy));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.policy", "policy parser allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
        "projection_plane_v2.policy",
        "unexpected strict policy parse failure");
    return result;
  }
}

PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_policy(
    const PrefillR1ProjectionPlaneV2Policy& policy,
    const PrefillR1ProjectionPlaneV2Manifest& manifest) {
  const auto manifest_diagnostic =
      validate_prefill_r1_projection_plane_v2_manifest(manifest);
  if (!manifest_diagnostic) {
    return manifest_diagnostic;
  }
  if (policy.version_major !=
          kPrefillR1ProjectionPlaneV2VersionMajor ||
      policy.version_minor !=
          kPrefillR1ProjectionPlaneV2VersionMinor ||
      policy.physical_layout != manifest.physical_layout ||
      policy.manifest_sha256 != manifest.manifest_sha256 ||
      policy.mlp_source_policy_sha256 != manifest.mlp_v4.policy_sha256 ||
      policy.attention_source_policy_sha256 !=
          manifest.attention_v4.policy_sha256 ||
      policy.converter_abi != kConverterAbi || policy.mode != kMode ||
      !policy.performance_upper_bound_only ||
      policy.quality_production_eligible ||
      policy.production_residency_eligible ||
      policy.projections.size() !=
          kPrefillR1ProjectionPlaneV2LogicalProjectionCount) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
        "projection_plane_v2.policy.binding",
        "policy does not bind the complete R1 v2 performance-only plane");
  }
  for (std::size_t index = 0U; index < policy.projections.size(); ++index) {
    const auto& calibration = policy.projections[index];
    const auto& logical = manifest.logical_projections[index];
    const bool valid_weight_clip =
        std::isfinite(calibration.weight_clip_ratio) &&
        calibration.weight_clip_ratio >= kPrefillA4MinimumClipRatio &&
        calibration.weight_clip_ratio <= 1.0;
    const bool valid_activation_clip =
        std::isfinite(calibration.activation_clip_ratio) &&
        calibration.activation_clip_ratio >= kPrefillA4MinimumClipRatio &&
        calibration.activation_clip_ratio <= 1.0;
    if (calibration.logical_ordinal != logical.logical_ordinal ||
        calibration.physical_ordinal != logical.physical_ordinal ||
        calibration.source_ordinal != logical.source_ordinal ||
        calibration.source_module != logical.source_module ||
        calibration.source_sha256 != logical.source_sha256 ||
        !valid_weight_clip || !valid_activation_clip ||
        calibration.factor_scheme !=
            kPrefillMLPFactorizedLaneR1FactorScheme ||
        !calibration.factor_path.empty() ||
        !lower_sha256(calibration.factor_sha256) ||
        calibration.factor_element_count != logical.input_size) {
      return make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
          "projection_plane_v2.policy.projections[" +
              std::to_string(index) + "]",
          "logical calibration does not preserve the authenticated R1 "
          "identity, clip, and factor contract");
    }
  }
  const std::string document = policy_document(policy);
  const std::string digest = sha256_text(document);
  if (digest.empty() || digest != policy.policy_sha256 ||
      policy.policy_bytes != document.size()) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
        "projection_plane_v2.policy.identity",
        "canonical policy digest or byte count mismatch");
  }
  return {};
}

PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_policy_sources(
    const PrefillR1ProjectionPlaneV2Policy& policy,
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillMLPFactorizedLaneR1Policy& mlp_v4_policy,
    const PrefillAttentionFactorizedLaneR1Policy& attention_v4_policy) {
  const auto structural =
      validate_prefill_r1_projection_plane_v2_policy(policy, manifest);
  if (!structural) {
    return structural;
  }
  const auto valid_common = [&manifest](const auto& source_policy,
                                        const auto& source_binding,
                                        const std::string_view abi,
                                        const std::string_view mode,
                                        const std::size_t count) {
    const auto& binding = source_policy.binding;
    return source_policy.converter_abi == abi &&
           source_policy.mode == mode &&
           source_policy.performance_upper_bound_only &&
           !source_policy.quality_production_eligible &&
           binding.version_major ==
               kPrefillMLPFactorizedLaneOverlayVersionMajor &&
           binding.version_minor ==
               kPrefillMLPFactorizedLaneOverlayVersionMinor &&
           binding.source_checkpoint_id == manifest.source_checkpoint_id &&
           binding.source_config_sha256 == manifest.source_config_sha256 &&
           binding.source_index_sha256 == manifest.source_index_sha256 &&
           binding.manifest_sha256 == source_binding.manifest_sha256 &&
           binding.policy_sha256 == source_binding.policy_sha256 &&
           binding.lane_count == 1U &&
           binding.projections.size() == count;
  };
  if (!valid_common(
          mlp_v4_policy, manifest.mlp_v4,
          kPrefillMLPFactorizedLaneR1ConverterAbi,
          kPrefillMLPFactorizedLaneR1Mode,
          kPrefillMLPFactorizedLaneProjectionCount) ||
      !valid_common(
          attention_v4_policy, manifest.attention_v4,
          kPrefillAttentionFactorizedLaneR1ConverterAbi,
          kPrefillAttentionFactorizedLaneR1Mode,
          kPrefillAttentionFactorizedLaneProjectionCount) ||
      mlp_v4_policy.binding.physical_layout !=
          kPrefillMLPFactorizedLaneOverlayLayout ||
      attention_v4_policy.binding.physical_layout !=
          kPrefillAttentionFactorizedLaneOverlayLayout ||
      !same_base(manifest.required_base_k256,
                 mlp_v4_policy.binding.required_base_k256) ||
      !same_base(manifest.required_base_k256,
                 attention_v4_policy.binding.required_base_k256)) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
        "projection_plane_v2.policy.sources",
        "source policies do not match the authenticated v4 source "
        "publications");
  }

  for (std::size_t index = 0U; index < policy.projections.size(); ++index) {
    const auto& expected = policy.projections[index];
    const auto& logical = manifest.logical_projections[index];
    const bool mlp =
        logical.family >=
        PrefillR1ProjectionPlaneV2LogicalFamily::kMlpGate;
    const auto matches = [&expected](const auto& source) {
      return source.ordinal == expected.source_ordinal &&
             source.source_module == expected.source_module &&
             source.source_sha256 == expected.source_sha256 &&
             source.weight_clip_ratio == expected.weight_clip_ratio &&
             source.activation_clip_ratio ==
                 expected.activation_clip_ratio &&
             source.factor_source.scheme == expected.factor_scheme &&
             source.factor_source.path == expected.factor_path &&
             source.factor_source.sha256 == expected.factor_sha256 &&
             source.factor_source.element_count ==
                 expected.factor_element_count;
    };
    const bool match =
        mlp ? matches(mlp_v4_policy.binding
                          .projections[logical.source_ordinal])
            : matches(attention_v4_policy.binding
                          .projections[logical.source_ordinal]);
    if (!match) {
      return make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
          "projection_plane_v2.policy.sources[" +
              std::to_string(index) + "]",
          "v2 logical calibration differs from its authenticated v4 policy "
          "entry");
    }
  }
  return {};
}

PrefillR1ProjectionPlaneV2ReceiptResult
build_prefill_r1_projection_plane_v2_receipt(
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillR1ProjectionPlaneV2Policy& policy,
    const std::string_view payload_sha256) {
  PrefillR1ProjectionPlaneV2ReceiptResult result;
  result.diagnostic =
      validate_prefill_r1_projection_plane_v2_policy(policy, manifest);
  if (!result.diagnostic) {
    return result;
  }
  if (!lower_sha256(payload_sha256)) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
        "projection_plane_v2.receipt.payload_sha256",
        "output payload SHA-256 must be lowercase hexadecimal");
    return result;
  }
  try {
    PrefillR1ProjectionPlaneV2Receipt receipt;
    receipt.physical_layout = manifest.physical_layout;
    receipt.manifest_sha256 = manifest.manifest_sha256;
    receipt.policy_sha256 = policy.policy_sha256;
    receipt.policy_bytes = policy.policy_bytes;
    receipt.payload_sha256 = std::string(payload_sha256);
    receipt.payload_bytes = manifest.payload_bytes;
    receipt.required_base_k256 = manifest.required_base_k256;
    receipt.mlp_v4 = manifest.mlp_v4;
    receipt.attention_v4 = manifest.attention_v4;
    receipt.logical_projection_count = manifest.logical_projection_count;
    receipt.physical_projection_count = manifest.physical_projection_count;
    result.diagnostic = validate_prefill_r1_projection_plane_v2_receipt(
        receipt, manifest, policy);
    if (!result.diagnostic) {
      return result;
    }
    result.canonical_document = receipt_document(receipt);
    result.canonical_sha256 = sha256_text(result.canonical_document);
    if (result.canonical_sha256.empty()) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
          "projection_plane_v2.receipt",
          "failed to hash canonical package receipt");
      return result;
    }
    result.value = std::move(receipt);
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.receipt", "receipt allocation failed");
    return result;
  }
}

PrefillR1ProjectionPlaneV2ReceiptResult
parse_prefill_r1_projection_plane_v2_receipt(
    const std::string_view document,
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillR1ProjectionPlaneV2Policy& policy) {
  PrefillR1ProjectionPlaneV2ReceiptResult result;
  result.diagnostic =
      validate_prefill_r1_projection_plane_v2_policy(policy, manifest);
  if (!result.diagnostic) return result;
  try {
    json::ParseOptions options;
    options.max_input_bytes = 128U * 1024U;
    options.max_nesting_depth = 8U;
    options.max_values = 256U;
    options.max_container_items = 256U;
    const json::ParseResult parsed = json::parse(document, options);
    const auto* const root = parsed ? parsed.value->as_object() : nullptr;
    if (root == nullptr ||
        !exact_keys(
            *root,
            {"atomic_installation_required", "attention_v4",
             "legacy_r1_co_residency_allowed", "logical_projection_count",
             "manifest_sha256", "mlp_v4", "payload_bytes",
             "payload_sha256", "performance_upper_bound_only",
             "physical_layout", "physical_projection_count", "policy_bytes",
             "policy_sha256", "required_base_k256",
             "production_residency_eligible", "quality_production_eligible",
             "schema", "version"})) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidReceipt,
          "projection_plane_v2.receipt",
          "strict receipt JSON schema mismatch");
      return result;
    }

    PrefillR1ProjectionPlaneV2Receipt receipt;
    std::string schema;
    std::uint64_t logical_count = 0U;
    std::uint64_t physical_count = 0U;
    if (!json_bool(*root, "atomic_installation_required",
                   receipt.atomic_installation_required) ||
        !parse_source(root->at("attention_v4"), receipt.attention_v4) ||
        !json_bool(*root, "legacy_r1_co_residency_allowed",
                   receipt.legacy_r1_co_residency_allowed) ||
        !json_uint(*root, "logical_projection_count", logical_count) ||
        !assign_u32(logical_count, receipt.logical_projection_count) ||
        !json_string(*root, "manifest_sha256", receipt.manifest_sha256) ||
        !parse_source(root->at("mlp_v4"), receipt.mlp_v4) ||
        !json_uint(*root, "payload_bytes", receipt.payload_bytes) ||
        !json_string(*root, "payload_sha256", receipt.payload_sha256) ||
        !json_bool(*root, "performance_upper_bound_only",
                   receipt.performance_upper_bound_only) ||
        !json_string(*root, "physical_layout", receipt.physical_layout) ||
        !json_uint(*root, "physical_projection_count", physical_count) ||
        !assign_u32(physical_count, receipt.physical_projection_count) ||
        !json_uint(*root, "policy_bytes", receipt.policy_bytes) ||
        !json_string(*root, "policy_sha256", receipt.policy_sha256) ||
        !parse_base(root->at("required_base_k256"),
                    receipt.required_base_k256) ||
        !json_bool(*root, "production_residency_eligible",
                   receipt.production_residency_eligible) ||
        !json_bool(*root, "quality_production_eligible",
                   receipt.quality_production_eligible) ||
        !json_string(*root, "schema", schema) || schema != kReceiptSchema ||
        !parse_version(root->at("version"), receipt.version_major,
                       receipt.version_minor)) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidReceipt,
          "projection_plane_v2.receipt.identity",
          "receipt identity fields are invalid");
      return result;
    }
    result.diagnostic = validate_prefill_r1_projection_plane_v2_receipt(
        receipt, manifest, policy);
    if (!result.diagnostic) return result;
    result.canonical_document = receipt_document(receipt);
    if (result.canonical_document != document) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidReceipt,
          "projection_plane_v2.receipt.canonical",
          "receipt must use the canonical field order and spelling");
      return result;
    }
    result.canonical_sha256 = sha256_text(document);
    if (result.canonical_sha256.empty()) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
          "projection_plane_v2.receipt", "failed to hash strict receipt");
      return result;
    }
    result.value.emplace(std::move(receipt));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.receipt", "receipt parser allocation failed");
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidReceipt,
        "projection_plane_v2.receipt",
        "unexpected strict receipt parse failure");
    return result;
  }
}

PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_receipt(
    const PrefillR1ProjectionPlaneV2Receipt& receipt,
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillR1ProjectionPlaneV2Policy& policy) {
  const auto policy_diagnostic =
      validate_prefill_r1_projection_plane_v2_policy(policy, manifest);
  if (!policy_diagnostic) {
    return policy_diagnostic;
  }
  if (receipt.version_major !=
          kPrefillR1ProjectionPlaneV2VersionMajor ||
      receipt.version_minor !=
          kPrefillR1ProjectionPlaneV2VersionMinor ||
      receipt.physical_layout != manifest.physical_layout ||
      receipt.manifest_sha256 != manifest.manifest_sha256 ||
      receipt.policy_sha256 != policy.policy_sha256 ||
      receipt.policy_bytes != policy.policy_bytes ||
      !lower_sha256(receipt.payload_sha256) ||
      receipt.payload_bytes != kPrefillR1ProjectionPlaneV2PayloadBytes ||
      !same_base(receipt.required_base_k256,
                 manifest.required_base_k256) ||
      !same_source_binding(receipt.mlp_v4, manifest.mlp_v4) ||
      !same_source_binding(receipt.attention_v4, manifest.attention_v4) ||
      receipt.logical_projection_count !=
          kPrefillR1ProjectionPlaneV2LogicalProjectionCount ||
      receipt.physical_projection_count !=
          kPrefillR1ProjectionPlaneV2PhysicalProjectionCount ||
      !receipt.atomic_installation_required ||
      receipt.legacy_r1_co_residency_allowed ||
      !receipt.performance_upper_bound_only ||
      receipt.quality_production_eligible ||
      receipt.production_residency_eligible) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidReceipt,
        "projection_plane_v2.receipt.binding",
        "receipt does not bind one indivisible 400-logical/336-physical "
        "performance-only package and both v4 sources");
  }
  return {};
}

PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_atomic_installation(
    const PrefillR1ProjectionPlaneV2Installation& installation) {
  if (installation.manifest == nullptr || installation.policy == nullptr ||
      installation.receipt == nullptr || installation.payload == nullptr) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kPartialInstallation,
        "projection_plane_v2.installation",
        "manifest, policy, package receipt, and payload must be installed as "
        "one atomic set");
  }
  if (installation.legacy_mlp_r1_installed ||
      installation.legacy_attention_r1_installed) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kLegacyCoResidency,
        "projection_plane_v2.installation",
        "split R1 v1 and unified projection-plane v2 may not coexist");
  }
  const auto receipt_diagnostic =
      validate_prefill_r1_projection_plane_v2_receipt(
          *installation.receipt, *installation.manifest,
          *installation.policy);
  if (!receipt_diagnostic) {
    return receipt_diagnostic;
  }
  const auto& payload = *installation.payload;
  const std::string expected_receipt_sha256 =
      sha256_text(receipt_document(*installation.receipt));
  if (!payload.authenticated ||
      payload.data == nullptr ||
      payload.version_major != kPrefillR1ProjectionPlaneV2VersionMajor ||
      payload.version_minor != kPrefillR1ProjectionPlaneV2VersionMinor ||
      payload.physical_layout != kPrefillR1ProjectionPlaneV2Layout ||
      payload.bytes != kPrefillR1ProjectionPlaneV2PayloadBytes ||
      payload.manifest_sha256 != installation.manifest->manifest_sha256 ||
      payload.policy_sha256 != installation.policy->policy_sha256 ||
      payload.payload_sha256 != installation.receipt->payload_sha256 ||
      payload.receipt_sha256 != expected_receipt_sha256) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
        "projection_plane_v2.installation.payload",
        "authenticated v2 payload identity differs from the installed "
        "manifest/policy/receipt");
  }
  return {};
}

PrefillR1ProjectionPlaneV2Diagnostic
permute_prefill_r1_gate_up_projection_pair_v2(
    const std::uint8_t* const gate_v4, const std::size_t gate_v4_bytes,
    const std::uint8_t* const up_v4, const std::size_t up_v4_bytes,
    const std::uint64_t output_size, const std::uint64_t input_size,
    std::uint8_t* const gate_up_v2,
    const std::size_t gate_up_v2_bytes) {
  const auto source = prefill_a4_factorized_lane_projection_layout_plan(
      output_size, input_size, 1U);
  const auto destination = gate_up_layout(output_size, input_size);
  if (!source || !destination.valid || output_size % 64U != 0U ||
      input_size % 64U != 0U ||
      source.projection_bytes >
          std::numeric_limits<std::size_t>::max() ||
      destination.projection_bytes >
          std::numeric_limits<std::size_t>::max() ||
      gate_v4_bytes != source.projection_bytes ||
      up_v4_bytes != source.projection_bytes ||
      gate_up_v2_bytes != destination.projection_bytes ||
      gate_v4 == nullptr || up_v4 == nullptr || gate_up_v2 == nullptr) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
        "projection_plane_v2.permute_gate_up",
        "Gate, Up, or destination extent does not match the lane-one "
        "factorized layout");
  }
  if (ranges_overlap(gate_v4, gate_v4_bytes, gate_up_v2,
                     gate_up_v2_bytes) ||
      ranges_overlap(up_v4, up_v4_bytes, gate_up_v2,
                     gate_up_v2_bytes)) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
        "projection_plane_v2.permute_gate_up",
        "v2 publication must not overlap either immutable v4 source");
  }

  std::fill(gate_up_v2, gate_up_v2 + gate_up_v2_bytes, 0U);
  const std::uint64_t k64_groups = input_size / 64U;
  for (std::uint64_t block = 0U; block < output_size / 64U; ++block) {
    for (std::uint64_t k64 = 0U; k64 < k64_groups; ++k64) {
      for (std::uint64_t n8 = 0U; n8 < 8U; ++n8) {
        for (std::uint64_t lane = 0U; lane < 32U; ++lane) {
          const std::uint64_t destination_offset =
              prefill_r1_projection_plane_v2_gate_up_code_slot_offset(
                  block * 64U + n8 * 8U, k64, n8, lane, output_size,
                  input_size);
          if (destination_offset ==
              kPrefillR1ProjectionPlaneV2InvalidOffset) {
            return make_diagnostic(
                PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
                "projection_plane_v2.permute_gate_up.code",
                "fragment-native Gate+Up coordinate is invalid");
          }
          const std::uint64_t row =
              block * 64U + n8 * 8U + lane / 4U;
          for (std::uint64_t word = 0U; word < 2U; ++word) {
            const std::uint64_t source_offset = canonical_packed_offset(
                row, k64, word * 16U + 4U * (lane % 4U), input_size);
            std::memcpy(gate_up_v2 + destination_offset + word * 4U,
                        gate_v4 + source.packed_weight_offset +
                            source_offset,
                        4U);
            std::memcpy(gate_up_v2 + destination_offset + 8U + word * 4U,
                        up_v4 + source.packed_weight_offset + source_offset,
                        4U);
          }
        }
      }
    }
  }
  for (std::uint64_t row = 0U; row < output_size; ++row) {
    const std::uint64_t source_scale =
        source.weight_scale_offset + row * 2U;
    const std::uint64_t destination_scale =
        destination.weight_scale_offset + row * 4U;
    std::memcpy(gate_up_v2 + destination_scale, gate_v4 + source_scale, 2U);
    std::memcpy(gate_up_v2 + destination_scale + 2U,
                up_v4 + source_scale, 2U);
  }
  std::memcpy(gate_up_v2 + destination.gate_metadata_offset,
              gate_v4 + source.metadata_offset,
              static_cast<std::size_t>(source.metadata_bytes));
  std::memcpy(gate_up_v2 + destination.up_metadata_offset,
              up_v4 + source.metadata_offset,
              static_cast<std::size_t>(source.metadata_bytes));
  return {};
}

PrefillR1ProjectionPlaneV2Diagnostic
permute_prefill_r1_adjacent_n8_projection_v2(
    const std::uint8_t* const source_v4,
    const std::size_t source_v4_bytes, const std::uint64_t output_size,
    const std::uint64_t input_size, std::uint8_t* const destination_v2,
    const std::size_t destination_v2_bytes) {
  const auto plan = prefill_a4_factorized_lane_projection_layout_plan(
      output_size, input_size, 1U);
  if (!plan || output_size % 128U != 0U || input_size % 64U != 0U ||
      plan.projection_bytes > std::numeric_limits<std::size_t>::max() ||
      source_v4_bytes != plan.projection_bytes ||
      destination_v2_bytes != plan.projection_bytes ||
      source_v4 == nullptr || destination_v2 == nullptr) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
        "projection_plane_v2.permute_adjacent_n8",
        "source or destination extent does not match one adjacent-N8 R1 "
        "projection");
  }
  if (ranges_overlap(source_v4, source_v4_bytes, destination_v2,
                     destination_v2_bytes)) {
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
        "projection_plane_v2.permute_adjacent_n8",
        "v2 publication must not overlap its immutable v4 source");
  }

  std::fill(destination_v2, destination_v2 + destination_v2_bytes, 0U);
  const std::uint64_t k64_groups = input_size / 64U;
  for (std::uint64_t panel = 0U; panel < output_size / 128U; ++panel) {
    for (std::uint64_t k64 = 0U; k64 < k64_groups; ++k64) {
      for (std::uint64_t n16 = 0U; n16 < 8U; ++n16) {
        for (std::uint64_t lane = 0U; lane < 32U; ++lane) {
          const std::uint64_t destination_offset =
              prefill_r1_projection_plane_v2_adjacent_n8_code_slot_offset(
                  panel, k64, n16, lane, output_size, input_size);
          if (destination_offset ==
              kPrefillR1ProjectionPlaneV2InvalidOffset) {
            return make_diagnostic(
                PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout,
                "projection_plane_v2.permute_adjacent_n8.code",
                "adjacent-N8 coordinate is invalid");
          }
          for (std::uint64_t word = 0U; word < 4U; ++word) {
            const std::uint64_t row =
                panel * 128U + n16 * 16U + (word / 2U) * 8U +
                lane / 4U;
            const std::uint64_t byte_in_k64 =
                (word % 2U) * 16U + 4U * (lane % 4U);
            const std::uint64_t source_offset = canonical_packed_offset(
                row, k64, byte_in_k64, input_size);
            std::memcpy(destination_v2 + destination_offset + word * 4U,
                        source_v4 + plan.packed_weight_offset +
                            source_offset,
                        4U);
          }
        }
      }
    }
  }
  std::memcpy(destination_v2 + plan.weight_scale_offset,
              source_v4 + plan.weight_scale_offset,
              static_cast<std::size_t>(plan.weight_scale_bytes));
  std::memcpy(destination_v2 + plan.metadata_offset,
              source_v4 + plan.metadata_offset,
              static_cast<std::size_t>(plan.metadata_bytes));
  return {};
}

PrefillR1ProjectionPlaneV2ConversionResult
convert_authenticated_prefill_r1_projection_plane_v4_to_v2(
    const PrefillMLPFactorizedLaneOverlayManifestBinding& mlp_v4,
    const PrefillMLPFactorizedLaneR1Policy& mlp_v4_policy,
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView& mlp_payload,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& attention_v4,
    const PrefillAttentionFactorizedLaneR1Policy& attention_v4_policy,
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView&
        attention_payload,
    const PrefillR1ProjectionPlaneV2MutablePayloadView& output) {
  PrefillR1ProjectionPlaneV2ConversionResult result;
  const auto manifest_result = build_prefill_r1_projection_plane_v2_manifest(
      mlp_v4, mlp_payload, attention_v4, attention_payload);
  if (!manifest_result) {
    result.diagnostic = manifest_result.diagnostic;
    return result;
  }
  const auto policy_result =
      build_prefill_r1_projection_plane_v2_policy(
          *manifest_result.value, mlp_v4_policy, attention_v4_policy);
  if (!policy_result) {
    result.diagnostic = policy_result.diagnostic;
    return result;
  }
  if (sizeof(std::size_t) < sizeof(std::uint64_t) ||
      output.data == nullptr ||
      output.bytes != kPrefillR1ProjectionPlaneV2PayloadBytes ||
      output.bytes > std::numeric_limits<std::size_t>::max() ||
      mlp_payload.data == nullptr || attention_payload.data == nullptr ||
      ranges_overlap(output.data, static_cast<std::size_t>(output.bytes),
                     mlp_payload.data,
                     static_cast<std::size_t>(mlp_payload.bytes)) ||
      ranges_overlap(output.data, static_cast<std::size_t>(output.bytes),
                     attention_payload.data,
                     static_cast<std::size_t>(attention_payload.bytes))) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
        "projection_plane_v2.output",
        "output must be one disjoint writable 12,182,982,656-byte view");
    return result;
  }

  try {
    const auto& manifest = *manifest_result.value;
    const auto mlp_plan =
        prefill_mlp_factorized_lane_overlay_layout_plan(1U);
    for (const auto& entry : manifest.projections) {
      if (entry.payload_offset >
              std::numeric_limits<std::size_t>::max() ||
          entry.payload_bytes >
              std::numeric_limits<std::size_t>::max() ||
          entry.payload_offset + entry.payload_bytes > output.bytes) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
            "projection_plane_v2.output",
            "physical projection output extent is invalid");
        return result;
      }
      std::uint8_t* const destination =
          output.data + static_cast<std::size_t>(entry.payload_offset);
      PrefillR1ProjectionPlaneV2Diagnostic diagnostic;
      if (entry.family ==
          PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpGateUp) {
        const std::size_t source_bytes =
            static_cast<std::size_t>(mlp_plan.gate.projection_bytes);
        diagnostic = permute_prefill_r1_gate_up_projection_pair_v2(
            mlp_payload.data +
                static_cast<std::size_t>(entry.source_primary_offset),
            source_bytes,
            mlp_payload.data +
                static_cast<std::size_t>(entry.source_secondary_offset),
            source_bytes, entry.output_size, entry.input_size, destination,
            static_cast<std::size_t>(entry.payload_bytes));
      } else {
        const bool mlp_down =
            entry.family ==
            PrefillR1ProjectionPlaneV2PhysicalFamily::kMlpDown;
        const auto& source_view =
            mlp_down ? mlp_payload : attention_payload;
        diagnostic = permute_prefill_r1_adjacent_n8_projection_v2(
            source_view.data +
                static_cast<std::size_t>(entry.source_primary_offset),
            static_cast<std::size_t>(entry.payload_bytes),
            entry.output_size, entry.input_size, destination,
            static_cast<std::size_t>(entry.payload_bytes));
      }
      if (!diagnostic) {
        result.diagnostic = std::move(diagnostic);
        return result;
      }
      result.bytes_written += entry.payload_bytes;
      result.logical_projections_written += entry.logical_projection_count;
      ++result.physical_projections_written;
    }
    if (result.bytes_written != kPrefillR1ProjectionPlaneV2PayloadBytes ||
        result.logical_projections_written !=
            kPrefillR1ProjectionPlaneV2LogicalProjectionCount ||
        result.physical_projections_written !=
            kPrefillR1ProjectionPlaneV2PhysicalProjectionCount) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload,
          "projection_plane_v2.output.coverage",
          "conversion did not close the complete projection plane");
      return result;
    }

    core::Sha256 output_hash;
    if (!output_hash.update(output.data,
                            static_cast<std::size_t>(output.bytes))) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
          "projection_plane_v2.output.payload_sha256",
          "failed to hash the complete output payload");
      return result;
    }
    result.payload_sha256 = output_hash.finalize().hex();
    const auto receipt_result =
        build_prefill_r1_projection_plane_v2_receipt(
            manifest, *policy_result.value, result.payload_sha256);
    if (!receipt_result) {
      result.diagnostic = receipt_result.diagnostic;
      return result;
    }
    result.manifest = manifest;
    result.policy = *policy_result.value;
    result.receipt = *receipt_result.value;
    result.manifest_document = manifest_result.canonical_document;
    result.policy_document = policy_result.canonical_document;
    result.receipt_document = receipt_result.canonical_document;
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.convert", "conversion allocation failed");
    return result;
  }
}

PrefillR1ProjectionPlaneV2Diagnostic
publish_prefill_r1_projection_plane_v2_file_set_no_replace(
    const std::array<fs::path, 4U>& temporary_paths,
    const std::array<fs::path, 4U>& target_paths) {
  std::array<bool, 4U> installed{};
  const auto rollback = [&]() noexcept {
    for (std::size_t index = installed.size(); index-- > 0U;) {
      if (installed[index]) (void)::unlink(target_paths[index].c_str());
    }
  };
  try {
    std::error_code error;
    const fs::path parent = fs::canonical(
        target_paths[0U].parent_path().empty()
            ? fs::path(".")
            : target_paths[0U].parent_path(),
        error);
    if (error) {
      return make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath,
          target_paths[0U].string(),
          "failed to canonicalize publication directory", {},
          error.message());
    }
    for (std::size_t index = 0U; index < target_paths.size(); ++index) {
      std::error_code target_error;
      std::error_code temporary_error;
      const fs::path target_parent = fs::canonical(
          target_paths[index].parent_path().empty()
              ? fs::path(".")
              : target_paths[index].parent_path(),
          target_error);
      const fs::path temporary_parent = fs::canonical(
          temporary_paths[index].parent_path().empty()
              ? fs::path(".")
              : temporary_paths[index].parent_path(),
          temporary_error);
      struct stat status {};
      if (target_error || temporary_error || target_parent != parent ||
          temporary_parent != parent ||
          ::lstat(temporary_paths[index].c_str(), &status) != 0 ||
          !S_ISREG(status.st_mode) || status.st_nlink != 1 ||
          (status.st_mode & (S_IWUSR | S_IWGRP | S_IWOTH)) != 0) {
        return make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath,
            temporary_paths[index].string(),
            "sealed temporary and target must share one real directory",
            {}, {}, errno);
      }
    }

    for (std::size_t index = 0U; index < target_paths.size(); ++index) {
      if (::link(temporary_paths[index].c_str(),
                 target_paths[index].c_str()) != 0) {
        const int link_error = errno;
        rollback();
        const UniqueFd directory(
            ::open(parent.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY));
        if (directory) (void)::fsync(directory.get());
        return make_diagnostic(
            link_error == EEXIST
                ? PrefillR1ProjectionPlaneV2ErrorCode::kPublicationConflict
                : PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
            target_paths[index].string(),
            "four-file no-replace publication failed and was rolled back",
            {}, {}, link_error);
      }
      installed[index] = true;
    }
    for (const fs::path& temporary : temporary_paths) {
      if (::unlink(temporary.c_str()) != 0) {
        const int unlink_error = errno;
        rollback();
        const UniqueFd directory(
            ::open(parent.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY));
        if (directory) (void)::fsync(directory.get());
        return make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
            temporary.string(),
            "failed to remove temporary publication link; targets rolled back",
            {}, {}, unlink_error);
      }
    }
    const UniqueFd directory(
        ::open(parent.c_str(), O_RDONLY | O_CLOEXEC | O_DIRECTORY));
    if (!directory || ::fsync(directory.get()) != 0) {
      const int sync_error = errno;
      rollback();
      if (directory) (void)::fsync(directory.get());
      return make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure, parent.string(),
          "failed to durably publish four-file set; targets rolled back", {},
          {}, sync_error);
    }
    return {};
  } catch (const std::bad_alloc&) {
    rollback();
    return make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.publication",
        "publication path allocation failed");
  }
}

PrefillR1ProjectionPlaneV2FileConversionResult
convert_authenticated_prefill_r1_projection_plane_v2_files(
    const PrefillR1ProjectionPlaneV2FileConversionOptions& options) {
  PrefillR1ProjectionPlaneV2FileConversionResult result;
  fs::path payload_temp;
  fs::path manifest_temp;
  fs::path policy_temp;
  fs::path receipt_temp;
  struct TemporaryCleanup final {
    std::array<fs::path*, 4U> paths;
    ~TemporaryCleanup() {
      for (const fs::path* path : paths) remove_if_present(*path);
    }
  } cleanup{{&payload_temp, &manifest_temp, &policy_temp, &receipt_temp}};

  try {
    const std::array<fs::path, 4U> targets = {
        options.output_path,
        fs::path(options.output_path.string() + ".manifest.json"),
        fs::path(options.output_path.string() + ".policy.json"),
        fs::path(options.output_path.string() + ".receipt.json")};
    const std::vector<fs::path> inputs = {
        options.base_k256_payload_path,
        options.base_k256_policy_path,
        options.base_k256_receipt_path,
        options.mlp_r1_payload_path,
        options.mlp_r1_policy_path,
        options.mlp_r1_receipt_path,
        options.attention_r1_payload_path,
        options.attention_r1_policy_path,
        options.attention_r1_receipt_path};
    if (options.model_directory.empty() || options.output_path.empty() ||
        options.output_path.filename().empty() ||
        options.output_path.filename() == "." ||
        options.output_path.filename() == ".." ||
        options.max_document_bytes == 0U ||
        options.max_document_bytes > 16ULL * 1024ULL * 1024ULL ||
        std::any_of(inputs.begin(), inputs.end(),
                    [](const fs::path& path) { return path.empty(); })) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidOption,
          "projection_plane_v2.file_options",
          "MODEL_DIR, nine source files, OUTPUT, and bounded documents are required");
      return result;
    }
    const fs::path output_parent =
        options.output_path.parent_path().empty()
            ? fs::path(".")
            : options.output_path.parent_path();
    std::error_code status_error;
    const fs::file_status parent_status =
        fs::symlink_status(output_parent, status_error);
    if (status_error || !fs::is_directory(parent_status) ||
        fs::is_symlink(parent_status) ||
        !all_distinct_paths(inputs, targets, result.diagnostic)) {
      if (result.diagnostic.ok()) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath,
            output_parent.string(),
            "OUTPUT parent must be an existing non-symlink directory", {},
            status_error.message());
      }
      return result;
    }
    for (const fs::path& target : targets) {
      if (!target_absent(target, result.diagnostic)) return result;
    }

    LockedDocument base_policy_document;
    LockedDocument base_receipt_document;
    LockedDocument mlp_policy_document;
    LockedDocument mlp_receipt_document;
    LockedDocument attention_policy_document;
    LockedDocument attention_receipt_document;
    for (auto pair : {
             std::pair<const fs::path*, LockedDocument*>(
                 &options.base_k256_policy_path, &base_policy_document),
             {&options.base_k256_receipt_path, &base_receipt_document},
             {&options.mlp_r1_policy_path, &mlp_policy_document},
             {&options.mlp_r1_receipt_path, &mlp_receipt_document},
             {&options.attention_r1_policy_path,
              &attention_policy_document},
             {&options.attention_r1_receipt_path,
              &attention_receipt_document}}) {
      result.diagnostic = read_locked_document(
          *pair.first, options.max_document_bytes, *pair.second);
      if (!result.diagnostic) return result;
      result.stats.source_bytes_hashed += pair.second->file.snapshot.size;
    }

    const mw::ManifestResult source_manifest =
        mw::build_qwen36_27b_text_manifest(options.model_directory);
    if (!source_manifest) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          options.model_directory.string(),
          "pinned Qwen3.6-27B checkpoint manifest validation failed");
      return result;
    }
    PrefillSidecarManifestOptions base_options;
    base_options.kind = PrefillSidecarKind::kA4K256;
    const auto base_manifest_result =
        build_qwen36_27b_prefill_sidecar_manifest(
            *source_manifest.value, pinned_qwen36_27b_shards(), base_options);
    if (!base_manifest_result) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          base_manifest_result.diagnostic.context,
          base_manifest_result.diagnostic.message);
      return result;
    }
    const PrefillSidecarManifest& base_manifest =
        *base_manifest_result.value;
    PrefillA4ConverterDiagnostic base_receipt_diagnostic;
    const auto base_receipt = parse_prefill_a4_publication_receipt(
        base_receipt_document.document, base_receipt_diagnostic);
    if (!base_receipt.has_value() || !base_receipt_diagnostic ||
        base_receipt->sidecar_kind != PrefillSidecarKind::kA4K256 ||
        base_policy_document.file.sha256 != base_receipt->policy_sha256) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication,
          options.base_k256_receipt_path.string(),
          "strict K256 receipt/policy authentication failed");
      return result;
    }
    auto base_authentication = authenticate_prefill_a4_publication_for_residency(
        base_manifest, *base_receipt, options.base_k256_payload_path,
        options.base_k256_policy_path);
    if (!base_authentication) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication,
          base_authentication.diagnostic.context,
          base_authentication.diagnostic.message,
          base_authentication.diagnostic.expected,
          base_authentication.diagnostic.actual,
          base_authentication.diagnostic.system_error);
      return result;
    }
    PrefillA4AuthenticatedPublication base_publication =
        std::move(*base_authentication.value);
    LockedFile base_payload_file;
    base_payload_file.fd = UniqueFd(::dup(base_publication.payload_fd()));
    base_payload_file.path = options.base_k256_payload_path;
    int snapshot_error = 0;
    if (!base_payload_file.fd ||
        !capture_snapshot(base_payload_file.fd.get(),
                          base_payload_file.snapshot, snapshot_error) ||
        base_payload_file.snapshot.size != base_receipt->payload_bytes) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication,
          options.base_k256_payload_path.string(),
          "failed to retain authenticated K256 payload descriptor", {}, {},
          snapshot_error != 0 ? snapshot_error : errno);
      return result;
    }
    result.diagnostic =
        hash_locked_file(base_payload_file, base_payload_file.sha256);
    if (!result.diagnostic ||
        base_payload_file.sha256 != base_receipt->payload_sha256) {
      if (result.diagnostic.ok()) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
            options.base_k256_payload_path.string(),
            "K256 payload digest differs from its receipt",
            base_receipt->payload_sha256, base_payload_file.sha256);
      }
      return result;
    }
    result.stats.source_bytes_hashed += base_payload_file.snapshot.size;

    const auto mlp_manifest_result =
        build_prefill_mlp_factorized_lane_r1_manifest(
            base_manifest, *base_receipt, base_receipt_document.file.sha256);
    const auto attention_manifest_result =
        build_prefill_attention_factorized_lane_r1_manifest(
            base_manifest, *base_receipt, base_receipt_document.file.sha256);
    if (!mlp_manifest_result || !attention_manifest_result) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest,
          "projection_plane_v2.source_v4_manifest",
          "failed to rebuild both source v4 manifests from common K256 base");
      return result;
    }
    const auto& mlp_manifest = *mlp_manifest_result.value;
    const auto& attention_manifest = *attention_manifest_result.value;
    const auto mlp_policy = parse_prefill_mlp_factorized_lane_r1_policy(
        mlp_policy_document.document, mlp_manifest);
    const auto attention_policy =
        parse_prefill_attention_factorized_lane_r1_policy(
            attention_policy_document.document, attention_manifest);
    if (!mlp_policy || !attention_policy) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy,
          "projection_plane_v2.source_v4_policy",
          "strict source v4 policy parsing failed");
      return result;
    }
    const auto mlp_receipt = parse_prefill_mlp_factorized_lane_r1_receipt(
        mlp_receipt_document.document, mlp_manifest, *mlp_policy.value);
    const auto attention_receipt =
        parse_prefill_attention_factorized_lane_r1_receipt(
            attention_receipt_document.document, attention_manifest,
            *attention_policy.value);
    if (!mlp_receipt || !attention_receipt ||
        mlp_receipt.value->binding.policy_sha256 !=
            mlp_policy_document.file.sha256 ||
        attention_receipt.value->binding.policy_sha256 !=
            attention_policy_document.file.sha256) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kInvalidReceipt,
          "projection_plane_v2.source_v4_receipt",
          "strict source v4 receipt authentication failed");
      return result;
    }

    LockedFile mlp_payload_file;
    LockedFile attention_payload_file;
    result.diagnostic = open_locked_file(
        options.mlp_r1_payload_path,
        mlp_receipt.value->binding.payload.bytes,
        mlp_payload_file);
    if (!result.diagnostic) return result;
    result.diagnostic = open_locked_file(
        options.attention_r1_payload_path,
        attention_receipt.value->binding.payload.bytes,
        attention_payload_file);
    if (!result.diagnostic) return result;
    std::array<const FileSnapshot*, 9U> snapshots = {
        &base_policy_document.file.snapshot,
        &base_receipt_document.file.snapshot,
        &mlp_policy_document.file.snapshot,
        &mlp_receipt_document.file.snapshot,
        &attention_policy_document.file.snapshot,
        &attention_receipt_document.file.snapshot,
        &base_payload_file.snapshot,
        &mlp_payload_file.snapshot,
        &attention_payload_file.snapshot};
    for (std::size_t left = 0U; left < snapshots.size(); ++left) {
      for (std::size_t right = left + 1U; right < snapshots.size(); ++right) {
        if (same_inode(*snapshots[left], *snapshots[right])) {
          result.diagnostic = make_diagnostic(
              PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath,
              "projection_plane_v2.source_files",
              "hard-linked aliases among source files are forbidden");
          return result;
        }
      }
    }
    result.diagnostic =
        hash_locked_file(mlp_payload_file, mlp_payload_file.sha256);
    if (!result.diagnostic) return result;
    result.diagnostic = hash_locked_file(attention_payload_file,
                                         attention_payload_file.sha256);
    if (!result.diagnostic) return result;
    if (mlp_payload_file.sha256 !=
            mlp_receipt.value->binding.payload.sha256 ||
        attention_payload_file.sha256 !=
            attention_receipt.value->binding.payload.sha256) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
          "projection_plane_v2.source_v4_payload",
          "source v4 payload digest differs from its authenticated receipt");
      return result;
    }
    result.stats.source_bytes_hashed += mlp_payload_file.snapshot.size +
                                        attention_payload_file.snapshot.size;

    PrefillR1ProjectionPlaneV2AuthenticatedPayloadView mlp_view;
    mlp_view.bytes = mlp_payload_file.snapshot.size;
    mlp_view.version_major = mlp_manifest.version_major;
    mlp_view.version_minor = mlp_manifest.version_minor;
    mlp_view.physical_layout = mlp_manifest.physical_layout;
    mlp_view.manifest_sha256 = mlp_manifest.manifest_sha256;
    mlp_view.policy_sha256 = mlp_policy.value->binding.policy_sha256;
    mlp_view.payload_sha256 = mlp_payload_file.sha256;
    mlp_view.receipt_sha256 = mlp_receipt_document.file.sha256;
    mlp_view.authenticated = true;
    PrefillR1ProjectionPlaneV2AuthenticatedPayloadView attention_view;
    attention_view.bytes = attention_payload_file.snapshot.size;
    attention_view.version_major = attention_manifest.version_major;
    attention_view.version_minor = attention_manifest.version_minor;
    attention_view.physical_layout = attention_manifest.physical_layout;
    attention_view.manifest_sha256 = attention_manifest.manifest_sha256;
    attention_view.policy_sha256 =
        attention_policy.value->binding.policy_sha256;
    attention_view.payload_sha256 = attention_payload_file.sha256;
    attention_view.receipt_sha256 = attention_receipt_document.file.sha256;
    attention_view.authenticated = true;

    UniqueFd output = create_temporary_file_near(
        targets[0U], "payload", payload_temp, result.diagnostic);
    if (!output) return result;
    if (kPrefillR1ProjectionPlaneV2PayloadBytes >
            static_cast<std::uint64_t>(std::numeric_limits<off_t>::max()) ||
        ::ftruncate(output.get(), static_cast<off_t>(
                                    kPrefillR1ProjectionPlaneV2PayloadBytes)) !=
            0) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
          payload_temp.string(), "failed to size v2 payload temporary", {},
          {}, errno);
      return result;
    }
    if (options.preallocate_output) {
      const int allocation_error = ::posix_fallocate(
          output.get(), 0,
          static_cast<off_t>(kPrefillR1ProjectionPlaneV2PayloadBytes));
      if (allocation_error != 0) {
        result.diagnostic = make_diagnostic(
            PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
            payload_temp.string(),
            "failed to preallocate complete v2 payload", {}, {},
            allocation_error);
        return result;
      }
    }
    MappedRegion mlp_mapping;
    MappedRegion attention_mapping;
    MappedRegion output_mapping;
    if (!mlp_mapping.map(mlp_payload_file.fd.get(), mlp_view.bytes,
                         PROT_READ) ||
        !attention_mapping.map(attention_payload_file.fd.get(),
                               attention_view.bytes, PROT_READ) ||
        !output_mapping.map(output.get(),
                            kPrefillR1ProjectionPlaneV2PayloadBytes,
                            PROT_READ | PROT_WRITE)) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
          "projection_plane_v2.mmap",
          "failed to map source/output files for bounded virtual-memory composition",
          {}, {}, errno);
      return result;
    }
    mlp_view.data = mlp_mapping.data();
    attention_view.data = attention_mapping.data();
    const PrefillR1ProjectionPlaneV2MutablePayloadView output_view{
        output_mapping.data(), kPrefillR1ProjectionPlaneV2PayloadBytes};
    auto conversion = convert_authenticated_prefill_r1_projection_plane_v4_to_v2(
        mlp_manifest, *mlp_policy.value, mlp_view, attention_manifest,
        *attention_policy.value, attention_view, output_view);
    if (!conversion) {
      result.diagnostic = conversion.diagnostic;
      return result;
    }
    if (::msync(output_mapping.data(), output_mapping.bytes(), MS_SYNC) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
          payload_temp.string(), "failed to flush mapped v2 payload", {}, {},
          errno);
      return result;
    }
    output_mapping.unmap();
    mlp_mapping.unmap();
    attention_mapping.unmap();
    if (::fsync(output.get()) != 0 || ::fchmod(output.get(), S_IRUSR) != 0 ||
        ::fsync(output.get()) != 0) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
          payload_temp.string(), "failed to seal v2 payload", {}, {}, errno);
      return result;
    }

    for (LockedDocument* document :
         {&base_policy_document, &base_receipt_document,
          &mlp_policy_document, &mlp_receipt_document,
          &attention_policy_document, &attention_receipt_document}) {
      result.diagnostic = rehash_unchanged(document->file,
                                           document->file.sha256);
      if (!result.diagnostic) return result;
      result.stats.source_bytes_hashed += document->file.snapshot.size;
    }
    result.diagnostic = rehash_unchanged(base_payload_file,
                                         base_payload_file.sha256);
    if (!result.diagnostic) return result;
    result.diagnostic = rehash_unchanged(mlp_payload_file,
                                         mlp_payload_file.sha256);
    if (!result.diagnostic) return result;
    result.diagnostic = rehash_unchanged(attention_payload_file,
                                         attention_payload_file.sha256);
    if (!result.diagnostic) return result;
    result.stats.source_bytes_hashed += base_payload_file.snapshot.size +
                                        mlp_payload_file.snapshot.size +
                                        attention_payload_file.snapshot.size;
    const auto base_revalidation =
        base_publication.revalidate_unchanged_after_consumption();
    if (!base_revalidation) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
          base_revalidation.context, base_revalidation.message,
          base_revalidation.expected, base_revalidation.actual,
          base_revalidation.system_error);
      return result;
    }

    const auto parsed_manifest = parse_prefill_r1_projection_plane_v2_manifest(
        conversion.manifest_document);
    const auto parsed_policy = parsed_manifest
                                   ? parse_prefill_r1_projection_plane_v2_policy(
                                         conversion.policy_document,
                                         *parsed_manifest.value)
                                   : PrefillR1ProjectionPlaneV2PolicyResult{};
    const auto parsed_receipt =
        parsed_manifest && parsed_policy
            ? parse_prefill_r1_projection_plane_v2_receipt(
                  conversion.receipt_document, *parsed_manifest.value,
                  *parsed_policy.value)
            : PrefillR1ProjectionPlaneV2ReceiptResult{};
    if (!parsed_manifest || !parsed_policy || !parsed_receipt ||
        parsed_receipt.value->payload_sha256 != conversion.payload_sha256) {
      result.diagnostic = make_diagnostic(
          PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure,
          "projection_plane_v2.output_documents",
          "generated v2 documents failed strict exact reparse");
      return result;
    }

    UniqueFd manifest_file = create_temporary_file_near(
        targets[1U], "manifest", manifest_temp, result.diagnostic);
    if (!manifest_file) return result;
    UniqueFd policy_file = create_temporary_file_near(
        targets[2U], "policy", policy_temp, result.diagnostic);
    if (!policy_file) return result;
    UniqueFd receipt_file = create_temporary_file_near(
        targets[3U], "receipt", receipt_temp, result.diagnostic);
    if (!receipt_file) return result;
    result.diagnostic = seal_document(
        manifest_file.get(), manifest_temp, conversion.manifest_document);
    if (!result.diagnostic) return result;
    result.diagnostic = seal_document(policy_file.get(), policy_temp,
                                      conversion.policy_document);
    if (!result.diagnostic) return result;
    result.diagnostic = seal_document(receipt_file.get(), receipt_temp,
                                      conversion.receipt_document);
    if (!result.diagnostic) return result;
    const std::array<fs::path, 4U> temporary_paths = {
        payload_temp, manifest_temp, policy_temp, receipt_temp};
    result.diagnostic =
        publish_prefill_r1_projection_plane_v2_file_set_no_replace(
            temporary_paths, targets);
    if (!result.diagnostic) return result;

    result.manifest = std::move(conversion.manifest);
    result.policy = std::move(conversion.policy);
    result.receipt = std::move(conversion.receipt);
    result.manifest_document = std::move(conversion.manifest_document);
    result.policy_document = std::move(conversion.policy_document);
    result.receipt_document = std::move(conversion.receipt_document);
    result.manifest_sha256 = result.manifest->manifest_sha256;
    result.policy_sha256 = result.policy->policy_sha256;
    result.payload_sha256 = conversion.payload_sha256;
    result.receipt_sha256 = sha256_text(result.receipt_document);
    result.mlp_source_payload_sha256 = mlp_payload_file.sha256;
    result.mlp_source_receipt_sha256 = mlp_receipt_document.file.sha256;
    result.attention_source_payload_sha256 = attention_payload_file.sha256;
    result.attention_source_receipt_sha256 =
        attention_receipt_document.file.sha256;
    result.stats.output_bytes_written = conversion.bytes_written;
    result.stats.logical_projections_written =
        conversion.logical_projections_written;
    result.stats.physical_projections_written =
        conversion.physical_projections_written;
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure,
        "projection_plane_v2.file_convert", "file conversion allocation failed");
    return result;
  } catch (const std::exception& error) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
        "projection_plane_v2.file_convert",
        "unexpected filesystem conversion failure", {}, error.what());
    return result;
  } catch (...) {
    result.diagnostic = make_diagnostic(
        PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure,
        "projection_plane_v2.file_convert",
        "unexpected file conversion failure");
    return result;
  }
}

std::string_view to_string(
    const PrefillR1ProjectionPlaneV2ErrorCode code) noexcept {
  switch (code) {
    case PrefillR1ProjectionPlaneV2ErrorCode::kNone:
      return "none";
    case PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourceManifest:
      return "invalid_source_manifest";
    case PrefillR1ProjectionPlaneV2ErrorCode::kInvalidSourcePublication:
      return "invalid_source_publication";
    case PrefillR1ProjectionPlaneV2ErrorCode::kSourceBindingMismatch:
      return "source_binding_mismatch";
    case PrefillR1ProjectionPlaneV2ErrorCode::kInvalidLayout:
      return "invalid_layout";
    case PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPayload:
      return "invalid_payload";
    case PrefillR1ProjectionPlaneV2ErrorCode::kInvalidPolicy:
      return "invalid_policy";
    case PrefillR1ProjectionPlaneV2ErrorCode::kInvalidReceipt:
      return "invalid_receipt";
    case PrefillR1ProjectionPlaneV2ErrorCode::kPartialInstallation:
      return "partial_installation";
    case PrefillR1ProjectionPlaneV2ErrorCode::kLegacyCoResidency:
      return "legacy_co_residency";
    case PrefillR1ProjectionPlaneV2ErrorCode::kArithmeticOverflow:
      return "arithmetic_overflow";
    case PrefillR1ProjectionPlaneV2ErrorCode::kAllocationFailure:
      return "allocation_failure";
    case PrefillR1ProjectionPlaneV2ErrorCode::kDigestFailure:
      return "digest_failure";
    case PrefillR1ProjectionPlaneV2ErrorCode::kInvalidOption:
      return "invalid_option";
    case PrefillR1ProjectionPlaneV2ErrorCode::kUnsafePath:
      return "unsafe_path";
    case PrefillR1ProjectionPlaneV2ErrorCode::kOpenFailed:
      return "open_failed";
    case PrefillR1ProjectionPlaneV2ErrorCode::kIoFailure:
      return "io_failure";
    case PrefillR1ProjectionPlaneV2ErrorCode::kPublicationConflict:
      return "publication_conflict";
  }
  return "unknown";
}

}  // namespace q3x::runtime
