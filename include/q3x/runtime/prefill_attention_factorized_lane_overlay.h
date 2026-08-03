#pragma once

#include "q3x/runtime/prefill_a4_factorized_lane_contract.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

// Host-publication foundation for the Qwen3.6-27B Attention-only factorized-lane
// experiment.  Nothing in this header grants runtime selection or production
// residency.  The offline converter authenticates these bindings; a future
// loader must independently authenticate them before exposing a payload.
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneOverlayVersionMajor = 4U;
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneOverlayVersionMinor = 0U;
inline constexpr std::uint32_t kPrefillAttentionFactorizedLaneLayerCount = 64U;
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneLinearLayerCount = 48U;
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneFullLayerCount = 16U;
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneLinearProjectionsPerLayer = 3U;
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneFullProjectionsPerLayer = 4U;
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneProjectionCount = 208U;
inline constexpr std::uint64_t
    kPrefillAttentionFactorizedLaneHiddenSize = 5'120U;
inline constexpr std::uint64_t
    kPrefillAttentionFactorizedLaneAttentionOutputSize = 6'144U;
inline constexpr std::uint64_t
    kPrefillAttentionFactorizedLaneLinearQkvOutputSize = 10'240U;
inline constexpr std::uint64_t
    kPrefillAttentionFactorizedLaneLinearZOutputSize = 6'144U;
inline constexpr std::uint64_t
    kPrefillAttentionFactorizedLaneFullQOutputSize = 12'288U;
inline constexpr std::uint64_t
    kPrefillAttentionFactorizedLaneFullKvOutputSize = 1'024U;
inline constexpr std::string_view kPrefillAttentionFactorizedLaneOverlayLayout =
    "sm87_s4_n64_packed_k64_factorized_lane_attention_v4";
inline constexpr std::string_view
    kPrefillAttentionFactorizedLaneRequiredBaseK256Layout =
        "sm87_s4_n64_packed_k64_scale_k256_consumer_v3";
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneRequiredBasePackedK = 64U;
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneRequiredBaseScaleK = 256U;
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneInvalidOrdinal =
        std::numeric_limits<std::uint32_t>::max();

static_assert(kPrefillAttentionFactorizedLaneProjectionCount ==
              kPrefillAttentionFactorizedLaneLinearLayerCount *
                      kPrefillAttentionFactorizedLaneLinearProjectionsPerLayer +
                  kPrefillAttentionFactorizedLaneFullLayerCount *
                      kPrefillAttentionFactorizedLaneFullProjectionsPerLayer);

enum class PrefillAttentionFactorizedLaneProjectionFamily : std::uint8_t {
  kLinearQkv = 0,
  kLinearZ,
  kLinearO,
  kFullQ,
  kFullK,
  kFullV,
  kFullO,
};

[[nodiscard]] constexpr bool
prefill_attention_factorized_lane_is_full_layer(
    const std::uint32_t layer_index) noexcept {
  return layer_index < kPrefillAttentionFactorizedLaneLayerCount &&
         ((layer_index + 1U) % 4U) == 0U;
}

// R1 is the performance upper-bound vehicle; R4 is the accuracy candidate.
// R2 remains structurally representable so publications can be inspected and
// compared, but it has no default promotion role.
enum class PrefillAttentionFactorizedLaneQualificationRole : std::uint8_t {
  kUnsupported = 0,
  kPerformanceUpperBound,
  kStructuralOnly,
  kQualityCandidate,
};

[[nodiscard]] constexpr PrefillAttentionFactorizedLaneQualificationRole
prefill_attention_factorized_lane_qualification_role(
    const std::uint32_t lane_count) noexcept {
  return lane_count == 1U
             ? PrefillAttentionFactorizedLaneQualificationRole::
                   kPerformanceUpperBound
         : lane_count == 2U
             ? PrefillAttentionFactorizedLaneQualificationRole::kStructuralOnly
         : lane_count == 4U
             ? PrefillAttentionFactorizedLaneQualificationRole::kQualityCandidate
             : PrefillAttentionFactorizedLaneQualificationRole::kUnsupported;
}

enum class PrefillAttentionFactorizedLaneOverlayPlanError : std::uint8_t {
  kNone = 0,
  kInvalidAlignment,
  kUnsupportedLaneCount,
  kProjectionPlanInvalid,
  kArithmeticOverflow,
};

// The payload is layer-major and preserves the authenticated K256 Attention
// inventory. Linear layers publish QKV/Z/O; every fourth layer publishes
// full Q/K/V/O. Every projection begins on `alignment`, and the mixed layer
// shapes deliberately do not use one fixed stride.
struct PrefillAttentionFactorizedLaneOverlayLayoutPlan final {
  PrefillAttentionFactorizedLaneOverlayPlanError error =
      PrefillAttentionFactorizedLaneOverlayPlanError::kProjectionPlanInvalid;
  std::uint32_t lane_count = 0U;
  std::uint64_t alignment = 0U;
  PrefillA4FactorizedLaneProjectionLayoutPlan linear_qkv;
  PrefillA4FactorizedLaneProjectionLayoutPlan linear_z;
  PrefillA4FactorizedLaneProjectionLayoutPlan linear_o;
  PrefillA4FactorizedLaneProjectionLayoutPlan full_q;
  PrefillA4FactorizedLaneProjectionLayoutPlan full_k;
  PrefillA4FactorizedLaneProjectionLayoutPlan full_v;
  PrefillA4FactorizedLaneProjectionLayoutPlan full_o;
  std::uint64_t linear_qkv_offset_in_layer = 0U;
  std::uint64_t linear_z_offset_in_layer = 0U;
  std::uint64_t linear_o_offset_in_layer = 0U;
  std::uint64_t full_q_offset_in_layer = 0U;
  std::uint64_t full_k_offset_in_layer = 0U;
  std::uint64_t full_v_offset_in_layer = 0U;
  std::uint64_t full_o_offset_in_layer = 0U;
  std::uint64_t linear_layer_bytes = 0U;
  std::uint64_t full_layer_bytes = 0U;
  std::uint64_t projection_count = 0U;
  std::uint64_t payload_bytes = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return error == PrefillAttentionFactorizedLaneOverlayPlanError::kNone;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
};

namespace prefill_attention_factorized_lane_overlay_detail {

[[nodiscard]] constexpr PrefillAttentionFactorizedLaneOverlayPlanError
map_projection_error(const PrefillA4FactorizedLanePlanError error) noexcept {
  switch (error) {
    case PrefillA4FactorizedLanePlanError::kNone:
      return PrefillAttentionFactorizedLaneOverlayPlanError::kNone;
    case PrefillA4FactorizedLanePlanError::kInvalidAlignment:
      return PrefillAttentionFactorizedLaneOverlayPlanError::kInvalidAlignment;
    case PrefillA4FactorizedLanePlanError::kUnsupportedLaneCount:
      return PrefillAttentionFactorizedLaneOverlayPlanError::kUnsupportedLaneCount;
    case PrefillA4FactorizedLanePlanError::kArithmeticOverflow:
      return PrefillAttentionFactorizedLaneOverlayPlanError::kArithmeticOverflow;
    case PrefillA4FactorizedLanePlanError::kZeroShape:
    case PrefillA4FactorizedLanePlanError::kShapeNotConsumerAligned:
    case PrefillA4FactorizedLanePlanError::kLaneNotConsumerAligned:
      return PrefillAttentionFactorizedLaneOverlayPlanError::kProjectionPlanInvalid;
  }
  return PrefillAttentionFactorizedLaneOverlayPlanError::kProjectionPlanInvalid;
}

}  // namespace prefill_attention_factorized_lane_overlay_detail

[[nodiscard]] constexpr PrefillAttentionFactorizedLaneOverlayLayoutPlan
prefill_attention_factorized_lane_overlay_layout_plan(
    const std::uint32_t lane_count,
    const std::uint64_t alignment =
        kPrefillA4FactorizedLaneMinimumAlignment) noexcept {
  PrefillAttentionFactorizedLaneOverlayLayoutPlan plan;
  plan.lane_count = lane_count;
  plan.alignment = alignment;
  plan.projection_count = kPrefillAttentionFactorizedLaneProjectionCount;
  plan.linear_qkv = prefill_a4_factorized_lane_projection_layout_plan(
      kPrefillAttentionFactorizedLaneLinearQkvOutputSize,
      kPrefillAttentionFactorizedLaneHiddenSize, lane_count, alignment);
  plan.linear_z = prefill_a4_factorized_lane_projection_layout_plan(
      kPrefillAttentionFactorizedLaneLinearZOutputSize,
      kPrefillAttentionFactorizedLaneHiddenSize, lane_count, alignment);
  plan.linear_o = prefill_a4_factorized_lane_projection_layout_plan(
      kPrefillAttentionFactorizedLaneHiddenSize,
      kPrefillAttentionFactorizedLaneAttentionOutputSize, lane_count,
      alignment);
  plan.full_q = prefill_a4_factorized_lane_projection_layout_plan(
      kPrefillAttentionFactorizedLaneFullQOutputSize,
      kPrefillAttentionFactorizedLaneHiddenSize, lane_count, alignment);
  plan.full_k = prefill_a4_factorized_lane_projection_layout_plan(
      kPrefillAttentionFactorizedLaneFullKvOutputSize,
      kPrefillAttentionFactorizedLaneHiddenSize, lane_count, alignment);
  plan.full_v = plan.full_k;
  plan.full_o = plan.linear_o;
  for (const auto error : {plan.linear_qkv.error, plan.linear_z.error,
                           plan.linear_o.error, plan.full_q.error,
                           plan.full_k.error, plan.full_v.error,
                           plan.full_o.error}) {
    if (error != PrefillA4FactorizedLanePlanError::kNone) {
      plan.error = prefill_attention_factorized_lane_overlay_detail::
          map_projection_error(error);
      return plan;
    }
  }

  using namespace prefill_a4_factorized_lane_contract_detail;
  std::uint64_t cursor = 0U;
  plan.linear_qkv_offset_in_layer = 0U;
  if (!checked_add(plan.linear_qkv_offset_in_layer,
                   plan.linear_qkv.projection_bytes, cursor) ||
      !checked_align_up(cursor, alignment, plan.linear_z_offset_in_layer) ||
      !checked_add(plan.linear_z_offset_in_layer,
                   plan.linear_z.projection_bytes, cursor) ||
      !checked_align_up(cursor, alignment, plan.linear_o_offset_in_layer) ||
      !checked_add(plan.linear_o_offset_in_layer,
                   plan.linear_o.projection_bytes, cursor) ||
      !checked_align_up(cursor, alignment, plan.linear_layer_bytes)) {
    plan.error = PrefillAttentionFactorizedLaneOverlayPlanError::kArithmeticOverflow;
    return plan;
  }
  cursor = 0U;
  plan.full_q_offset_in_layer = 0U;
  if (!checked_add(plan.full_q_offset_in_layer, plan.full_q.projection_bytes,
                   cursor) ||
      !checked_align_up(cursor, alignment, plan.full_k_offset_in_layer) ||
      !checked_add(plan.full_k_offset_in_layer, plan.full_k.projection_bytes,
                   cursor) ||
      !checked_align_up(cursor, alignment, plan.full_v_offset_in_layer) ||
      !checked_add(plan.full_v_offset_in_layer, plan.full_v.projection_bytes,
                   cursor) ||
      !checked_align_up(cursor, alignment, plan.full_o_offset_in_layer) ||
      !checked_add(plan.full_o_offset_in_layer, plan.full_o.projection_bytes,
                   cursor) ||
      !checked_align_up(cursor, alignment, plan.full_layer_bytes)) {
    plan.error = PrefillAttentionFactorizedLaneOverlayPlanError::kArithmeticOverflow;
    return plan;
  }
  std::uint64_t linear_bytes = 0U;
  std::uint64_t full_bytes = 0U;
  if (!checked_multiply(plan.linear_layer_bytes,
                        kPrefillAttentionFactorizedLaneLinearLayerCount,
                        linear_bytes) ||
      !checked_multiply(plan.full_layer_bytes,
                        kPrefillAttentionFactorizedLaneFullLayerCount,
                        full_bytes) ||
      !checked_add(linear_bytes, full_bytes, plan.payload_bytes)) {
    plan.error = PrefillAttentionFactorizedLaneOverlayPlanError::kArithmeticOverflow;
    return plan;
  }
  plan.error = PrefillAttentionFactorizedLaneOverlayPlanError::kNone;
  return plan;
}

[[nodiscard]] constexpr std::uint64_t
prefill_attention_factorized_lane_projection_absolute_offset(
    const PrefillAttentionFactorizedLaneOverlayLayoutPlan& plan,
    const std::uint32_t layer_index,
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  if (!plan.valid() || layer_index >= kPrefillAttentionFactorizedLaneLayerCount) {
    return kPrefillA4FactorizedLaneInvalidOffset;
  }
  const std::uint64_t full_before = layer_index / 4U;
  const std::uint64_t linear_before = layer_index - full_before;
  std::uint64_t linear_bytes = 0U;
  std::uint64_t full_bytes = 0U;
  std::uint64_t layer_offset = 0U;
  if (!prefill_a4_factorized_lane_contract_detail::checked_multiply(
          linear_before, plan.linear_layer_bytes, linear_bytes) ||
      !prefill_a4_factorized_lane_contract_detail::checked_multiply(
          full_before, plan.full_layer_bytes, full_bytes) ||
      !prefill_a4_factorized_lane_contract_detail::checked_add(
          linear_bytes, full_bytes, layer_offset)) {
    return kPrefillA4FactorizedLaneInvalidOffset;
  }
  std::uint64_t relative = 0U;
  const bool full_layer =
      prefill_attention_factorized_lane_is_full_layer(layer_index);
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      if (full_layer) return kPrefillA4FactorizedLaneInvalidOffset;
      relative = plan.linear_qkv_offset_in_layer;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      if (full_layer) return kPrefillA4FactorizedLaneInvalidOffset;
      relative = plan.linear_z_offset_in_layer;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      if (full_layer) return kPrefillA4FactorizedLaneInvalidOffset;
      relative = plan.linear_o_offset_in_layer;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      if (!full_layer) return kPrefillA4FactorizedLaneInvalidOffset;
      relative = plan.full_q_offset_in_layer;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      if (!full_layer) return kPrefillA4FactorizedLaneInvalidOffset;
      relative = plan.full_k_offset_in_layer;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      if (!full_layer) return kPrefillA4FactorizedLaneInvalidOffset;
      relative = plan.full_v_offset_in_layer;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      if (!full_layer) return kPrefillA4FactorizedLaneInvalidOffset;
      relative = plan.full_o_offset_in_layer;
      break;
    default:
      return kPrefillA4FactorizedLaneInvalidOffset;
  }
  std::uint64_t absolute = 0U;
  return prefill_a4_factorized_lane_contract_detail::checked_add(
             layer_offset, relative, absolute)
             ? absolute
             : kPrefillA4FactorizedLaneInvalidOffset;
}

[[nodiscard]] constexpr std::uint32_t
prefill_attention_factorized_lane_projection_ordinal(
    const std::uint32_t layer_index,
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  if (layer_index >= kPrefillAttentionFactorizedLaneLayerCount) {
    return kPrefillAttentionFactorizedLaneInvalidOrdinal;
  }
  const bool full_layer =
      prefill_attention_factorized_lane_is_full_layer(layer_index);
  std::uint32_t position = kPrefillAttentionFactorizedLaneInvalidOrdinal;
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      if (!full_layer) position = 0U;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      if (!full_layer) position = 1U;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      if (!full_layer) position = 2U;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      if (full_layer) position = 0U;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      if (full_layer) position = 1U;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      if (full_layer) position = 2U;
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      if (full_layer) position = 3U;
      break;
  }
  if (position == kPrefillAttentionFactorizedLaneInvalidOrdinal) {
    return kPrefillAttentionFactorizedLaneInvalidOrdinal;
  }
  return layer_index *
             kPrefillAttentionFactorizedLaneLinearProjectionsPerLayer +
         layer_index / 4U + position;
}

enum class PrefillAttentionFactorizedLaneMetadataError : std::uint8_t {
  kNone = 0,
  kNullInput,
  kUnsupportedLaneCount,
  kInvalidInputSize,
  kInvalidInverseAlpha,
  kArithmeticOverflow,
  kAllocationFailure,
  kInvalidByteLength,
  kInvalidMagic,
  kUnsupportedVersion,
  kUnsupportedEncoding,
  kExpectedShapeMismatch,
  kDigestMismatch,
};

struct PrefillAttentionFactorizedLaneMetadataSerializationResult final {
  PrefillAttentionFactorizedLaneMetadataError error =
      PrefillAttentionFactorizedLaneMetadataError::kInvalidInputSize;
  std::array<std::uint8_t,
             kPrefillA4FactorizedLaneMetadataDigestBytes>
      inverse_alpha_sha256{};
  std::vector<std::uint8_t> bytes;

  [[nodiscard]] bool valid() const noexcept {
    return error == PrefillAttentionFactorizedLaneMetadataError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
};

struct PrefillAttentionFactorizedLaneMetadataParseResult final {
  PrefillAttentionFactorizedLaneMetadataError error =
      PrefillAttentionFactorizedLaneMetadataError::kInvalidByteLength;
  std::uint32_t lane_count = 0U;
  std::uint64_t input_size = 0U;
  std::array<std::uint8_t,
             kPrefillA4FactorizedLaneMetadataDigestBytes>
      inverse_alpha_sha256{};
  std::vector<float> inverse_alpha;

  [[nodiscard]] bool valid() const noexcept {
    return error == PrefillAttentionFactorizedLaneMetadataError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
};

// Serialization emits exactly the metadata bytes described by
// prefill_a4_factorized_lane_contract.h, excluding projection-end padding.
[[nodiscard]] PrefillAttentionFactorizedLaneMetadataSerializationResult
serialize_prefill_attention_factorized_lane_metadata(
    std::uint32_t lane_count, const float* inverse_alpha,
    std::size_t inverse_alpha_count);

// Parsing requires the independently authenticated expected lane/K.  This is
// intentional: the metadata SHA covers the exact inverse-alpha byte payload,
// while the manifest/policy/receipt bind the header shape. Requiring both
// here prevents a mutated but otherwise structurally valid lane/K header from
// being accepted in isolation.
[[nodiscard]] PrefillAttentionFactorizedLaneMetadataParseResult
parse_prefill_attention_factorized_lane_metadata(
    const std::uint8_t* bytes, std::size_t byte_count,
    std::uint32_t expected_lane_count, std::uint64_t expected_input_size);

// Publication-binding vocabulary consumed by the offline converter. These
// structs remain data-only at runtime: no loader or selector consumes them.
struct PrefillAttentionFactorizedLaneBaseK256Binding final {
  std::string physical_layout;
  std::uint32_t packed_k_group_size =
      kPrefillAttentionFactorizedLaneRequiredBasePackedK;
  std::uint32_t scale_group_size =
      kPrefillAttentionFactorizedLaneRequiredBaseScaleK;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::string payload_sha256;
  // SHA-256 of the exact strict K256 receipt JSON bytes.  The ordinary base
  // identities above bind what the receipt says; this field additionally
  // binds the exact authenticated hand-off document used by the derivative
  // converter.
  std::string receipt_sha256;
};

struct PrefillAttentionFactorizedLaneFactorSourceBinding final {
  // R1 uses "identity_alpha_f32_v1" with an empty path because the converter
  // deterministically materializes FP32 one[K].  Future calibrated R4
  // publications must provide an authenticated external source path.
  std::string scheme;
  std::string path;
  std::string sha256;
  std::uint64_t element_count = 0U;
};

struct PrefillAttentionFactorizedLanePayloadIdentity final {
  std::string path;
  std::string sha256;
  std::uint64_t bytes = 0U;
};

struct PrefillAttentionFactorizedLaneManifestProjection final {
  std::uint32_t ordinal = 0U;
  std::uint32_t layer_index = 0U;
  PrefillAttentionFactorizedLaneProjectionFamily family =
      PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv;
  std::string source_module;
  std::string source_sha256;
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
};

struct PrefillAttentionFactorizedLaneOverlayManifestBinding final {
  std::uint32_t version_major =
      kPrefillAttentionFactorizedLaneOverlayVersionMajor;
  std::uint32_t version_minor =
      kPrefillAttentionFactorizedLaneOverlayVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  PrefillAttentionFactorizedLaneBaseK256Binding required_base_k256;
  std::uint32_t lane_count = 0U;
  std::vector<PrefillAttentionFactorizedLaneManifestProjection> projections;
  std::uint64_t payload_bytes = 0U;
  std::string manifest_sha256;
};

struct PrefillAttentionFactorizedLaneProjectionCalibrationBinding final {
  std::uint32_t ordinal = 0U;
  std::string source_module;
  std::string source_sha256;
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  PrefillAttentionFactorizedLaneFactorSourceBinding factor_source;
};

struct PrefillAttentionFactorizedLaneOverlayPolicyBinding final {
  std::uint32_t version_major =
      kPrefillAttentionFactorizedLaneOverlayVersionMajor;
  std::uint32_t version_minor =
      kPrefillAttentionFactorizedLaneOverlayVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string manifest_sha256;
  PrefillAttentionFactorizedLaneBaseK256Binding required_base_k256;
  std::uint32_t lane_count = 0U;
  std::vector<PrefillAttentionFactorizedLaneProjectionCalibrationBinding>
      projections;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
};

struct PrefillAttentionFactorizedLaneOverlayReceiptBinding final {
  std::uint32_t version_major =
      kPrefillAttentionFactorizedLaneOverlayVersionMajor;
  std::uint32_t version_minor =
      kPrefillAttentionFactorizedLaneOverlayVersionMinor;
  // Must remain false until a real-checkpoint converter, authenticated loader,
  // accuracy gate, and API/EvalScope performance gate are implemented.
  bool production_residency_eligible = false;
  // R1 is an execution upper bound only.  This bit is independently bound so
  // residency authentication can never be mistaken for a quality verdict.
  bool quality_production_eligible = false;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  PrefillAttentionFactorizedLaneBaseK256Binding required_base_k256;
  std::uint32_t lane_count = 0U;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
  PrefillAttentionFactorizedLanePayloadIdentity payload;
  std::uint64_t projection_count = 0U;
};

static_assert(prefill_attention_factorized_lane_overlay_layout_plan(1U).valid());
static_assert(prefill_attention_factorized_lane_overlay_layout_plan(2U).valid());
static_assert(prefill_attention_factorized_lane_overlay_layout_plan(4U).valid());
inline constexpr auto kPrefillAttentionFactorizedLaneR1LayoutPlan =
    prefill_attention_factorized_lane_overlay_layout_plan(1U);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.linear_qkv
                  .projection_bytes == 26'255'616ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.linear_z
                  .projection_bytes == 15'761'664ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.linear_o
                  .projection_bytes == 15'763'712ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.full_q
                  .projection_bytes == 31'502'592ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.full_k
                  .projection_bytes == 2'644'224ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.full_v
                  .projection_bytes == 2'644'224ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.full_o
                  .projection_bytes == 15'763'712ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.linear_layer_bytes ==
              57'780'992ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.full_layer_bytes ==
              52'554'752ULL);
static_assert(kPrefillAttentionFactorizedLaneR1LayoutPlan.payload_bytes ==
              3'614'363'648ULL);
static_assert(prefill_attention_factorized_lane_projection_ordinal(
                  0U,
                  PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv) ==
              0U);
static_assert(prefill_attention_factorized_lane_projection_ordinal(
                  3U, PrefillAttentionFactorizedLaneProjectionFamily::kFullQ) ==
              9U);
static_assert(prefill_attention_factorized_lane_projection_ordinal(
                  63U,
                  PrefillAttentionFactorizedLaneProjectionFamily::kFullO) ==
              207U);

}  // namespace q3x::runtime
