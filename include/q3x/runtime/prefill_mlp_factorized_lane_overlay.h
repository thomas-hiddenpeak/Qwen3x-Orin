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

// Host-publication foundation for the Qwen3.6-27B MLP-only factorized-lane
// experiment.  Nothing in this header grants runtime selection or production
// residency.  A future converter/loader must authenticate all bindings below
// before this payload can be exposed to a kernel.
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneOverlayVersionMajor = 4U;
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneOverlayVersionMinor = 0U;
inline constexpr std::uint32_t kPrefillMLPFactorizedLaneLayerCount = 64U;
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneProjectionsPerLayer = 3U;
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneProjectionCount = 192U;
inline constexpr std::uint64_t
    kPrefillMLPFactorizedLaneGateUpOutputSize = 17'408U;
inline constexpr std::uint64_t
    kPrefillMLPFactorizedLaneGateUpInputSize = 5'120U;
inline constexpr std::uint64_t
    kPrefillMLPFactorizedLaneDownOutputSize = 5'120U;
inline constexpr std::uint64_t
    kPrefillMLPFactorizedLaneDownInputSize = 17'408U;
inline constexpr std::string_view kPrefillMLPFactorizedLaneOverlayLayout =
    "sm87_s4_n64_packed_k64_factorized_lane_mlp_v4";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneRequiredBaseK256Layout =
        "sm87_s4_n64_packed_k64_scale_k256_consumer_v3";
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneRequiredBasePackedK = 64U;
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneRequiredBaseScaleK = 256U;
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneInvalidOrdinal =
        std::numeric_limits<std::uint32_t>::max();

static_assert(kPrefillMLPFactorizedLaneProjectionCount ==
              kPrefillMLPFactorizedLaneLayerCount *
                  kPrefillMLPFactorizedLaneProjectionsPerLayer);

enum class PrefillMLPFactorizedLaneProjectionFamily : std::uint8_t {
  kGate = 0,
  kUp = 1,
  kDown = 2,
};

// R1 is the performance upper-bound vehicle; R4 is the accuracy candidate.
// R2 remains structurally representable so publications can be inspected and
// compared, but it has no default promotion role.
enum class PrefillMLPFactorizedLaneQualificationRole : std::uint8_t {
  kUnsupported = 0,
  kPerformanceUpperBound,
  kStructuralOnly,
  kQualityCandidate,
};

[[nodiscard]] constexpr PrefillMLPFactorizedLaneQualificationRole
prefill_mlp_factorized_lane_qualification_role(
    const std::uint32_t lane_count) noexcept {
  return lane_count == 1U
             ? PrefillMLPFactorizedLaneQualificationRole::
                   kPerformanceUpperBound
         : lane_count == 2U
             ? PrefillMLPFactorizedLaneQualificationRole::kStructuralOnly
         : lane_count == 4U
             ? PrefillMLPFactorizedLaneQualificationRole::kQualityCandidate
             : PrefillMLPFactorizedLaneQualificationRole::kUnsupported;
}

enum class PrefillMLPFactorizedLaneOverlayPlanError : std::uint8_t {
  kNone = 0,
  kInvalidAlignment,
  kUnsupportedLaneCount,
  kProjectionPlanInvalid,
  kArithmeticOverflow,
};

// The payload is layer-major, and every projection begins on `alignment`:
//   layer 0 Gate, Up, Down; layer 1 Gate, Up, Down; ...; layer 63.
// Gate and Up share a shape, while Down has a different scale/metadata size,
// so projections deliberately do not use one fixed stride.
struct PrefillMLPFactorizedLaneOverlayLayoutPlan final {
  PrefillMLPFactorizedLaneOverlayPlanError error =
      PrefillMLPFactorizedLaneOverlayPlanError::kProjectionPlanInvalid;
  std::uint32_t lane_count = 0U;
  std::uint64_t alignment = 0U;
  PrefillA4FactorizedLaneProjectionLayoutPlan gate;
  PrefillA4FactorizedLaneProjectionLayoutPlan up;
  PrefillA4FactorizedLaneProjectionLayoutPlan down;
  std::uint64_t gate_offset_in_layer = 0U;
  std::uint64_t up_offset_in_layer = 0U;
  std::uint64_t down_offset_in_layer = 0U;
  std::uint64_t layer_bytes = 0U;
  std::uint64_t projection_count = 0U;
  std::uint64_t payload_bytes = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return error == PrefillMLPFactorizedLaneOverlayPlanError::kNone;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
};

namespace prefill_mlp_factorized_lane_overlay_detail {

[[nodiscard]] constexpr PrefillMLPFactorizedLaneOverlayPlanError
map_projection_error(const PrefillA4FactorizedLanePlanError error) noexcept {
  switch (error) {
    case PrefillA4FactorizedLanePlanError::kNone:
      return PrefillMLPFactorizedLaneOverlayPlanError::kNone;
    case PrefillA4FactorizedLanePlanError::kInvalidAlignment:
      return PrefillMLPFactorizedLaneOverlayPlanError::kInvalidAlignment;
    case PrefillA4FactorizedLanePlanError::kUnsupportedLaneCount:
      return PrefillMLPFactorizedLaneOverlayPlanError::kUnsupportedLaneCount;
    case PrefillA4FactorizedLanePlanError::kArithmeticOverflow:
      return PrefillMLPFactorizedLaneOverlayPlanError::kArithmeticOverflow;
    case PrefillA4FactorizedLanePlanError::kZeroShape:
    case PrefillA4FactorizedLanePlanError::kShapeNotConsumerAligned:
    case PrefillA4FactorizedLanePlanError::kLaneNotConsumerAligned:
      return PrefillMLPFactorizedLaneOverlayPlanError::kProjectionPlanInvalid;
  }
  return PrefillMLPFactorizedLaneOverlayPlanError::kProjectionPlanInvalid;
}

}  // namespace prefill_mlp_factorized_lane_overlay_detail

[[nodiscard]] constexpr PrefillMLPFactorizedLaneOverlayLayoutPlan
prefill_mlp_factorized_lane_overlay_layout_plan(
    const std::uint32_t lane_count,
    const std::uint64_t alignment =
        kPrefillA4FactorizedLaneMinimumAlignment) noexcept {
  PrefillMLPFactorizedLaneOverlayLayoutPlan plan;
  plan.lane_count = lane_count;
  plan.alignment = alignment;
  plan.projection_count = kPrefillMLPFactorizedLaneProjectionCount;
  plan.gate = prefill_a4_factorized_lane_projection_layout_plan(
      kPrefillMLPFactorizedLaneGateUpOutputSize,
      kPrefillMLPFactorizedLaneGateUpInputSize, lane_count, alignment);
  if (!plan.gate) {
    plan.error =
        prefill_mlp_factorized_lane_overlay_detail::map_projection_error(
            plan.gate.error);
    return plan;
  }
  plan.up = plan.gate;
  plan.down = prefill_a4_factorized_lane_projection_layout_plan(
      kPrefillMLPFactorizedLaneDownOutputSize,
      kPrefillMLPFactorizedLaneDownInputSize, lane_count, alignment);
  if (!plan.down) {
    plan.error =
        prefill_mlp_factorized_lane_overlay_detail::map_projection_error(
            plan.down.error);
    return plan;
  }

  using namespace prefill_a4_factorized_lane_contract_detail;
  plan.gate_offset_in_layer = 0U;
  std::uint64_t cursor = 0U;
  if (!checked_add(plan.gate_offset_in_layer, plan.gate.projection_bytes,
                   cursor) ||
      !checked_align_up(cursor, alignment, plan.up_offset_in_layer) ||
      !checked_add(plan.up_offset_in_layer, plan.up.projection_bytes,
                   cursor) ||
      !checked_align_up(cursor, alignment, plan.down_offset_in_layer) ||
      !checked_add(plan.down_offset_in_layer, plan.down.projection_bytes,
                   cursor) ||
      !checked_align_up(cursor, alignment, plan.layer_bytes) ||
      !checked_multiply(plan.layer_bytes,
                        kPrefillMLPFactorizedLaneLayerCount,
                        plan.payload_bytes)) {
    plan.error = PrefillMLPFactorizedLaneOverlayPlanError::kArithmeticOverflow;
    return plan;
  }
  plan.error = PrefillMLPFactorizedLaneOverlayPlanError::kNone;
  return plan;
}

[[nodiscard]] constexpr std::uint64_t
prefill_mlp_factorized_lane_projection_absolute_offset(
    const PrefillMLPFactorizedLaneOverlayLayoutPlan& plan,
    const std::uint32_t layer_index,
    const PrefillMLPFactorizedLaneProjectionFamily family) noexcept {
  if (!plan.valid() || layer_index >= kPrefillMLPFactorizedLaneLayerCount) {
    return kPrefillA4FactorizedLaneInvalidOffset;
  }
  std::uint64_t layer_offset = 0U;
  if (!prefill_a4_factorized_lane_contract_detail::checked_multiply(
          layer_index, plan.layer_bytes, layer_offset)) {
    return kPrefillA4FactorizedLaneInvalidOffset;
  }
  std::uint64_t relative = 0U;
  switch (family) {
    case PrefillMLPFactorizedLaneProjectionFamily::kGate:
      relative = plan.gate_offset_in_layer;
      break;
    case PrefillMLPFactorizedLaneProjectionFamily::kUp:
      relative = plan.up_offset_in_layer;
      break;
    case PrefillMLPFactorizedLaneProjectionFamily::kDown:
      relative = plan.down_offset_in_layer;
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
prefill_mlp_factorized_lane_projection_ordinal(
    const std::uint32_t layer_index,
    const PrefillMLPFactorizedLaneProjectionFamily family) noexcept {
  if (layer_index >= kPrefillMLPFactorizedLaneLayerCount) {
    return kPrefillMLPFactorizedLaneInvalidOrdinal;
  }
  const auto position = static_cast<std::uint32_t>(family);
  if (position >= kPrefillMLPFactorizedLaneProjectionsPerLayer) {
    return kPrefillMLPFactorizedLaneInvalidOrdinal;
  }
  return layer_index * kPrefillMLPFactorizedLaneProjectionsPerLayer +
         position;
}

enum class PrefillMLPFactorizedLaneMetadataError : std::uint8_t {
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

struct PrefillMLPFactorizedLaneMetadataSerializationResult final {
  PrefillMLPFactorizedLaneMetadataError error =
      PrefillMLPFactorizedLaneMetadataError::kInvalidInputSize;
  std::array<std::uint8_t,
             kPrefillA4FactorizedLaneMetadataDigestBytes>
      inverse_alpha_sha256{};
  std::vector<std::uint8_t> bytes;

  [[nodiscard]] bool valid() const noexcept {
    return error == PrefillMLPFactorizedLaneMetadataError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
};

struct PrefillMLPFactorizedLaneMetadataParseResult final {
  PrefillMLPFactorizedLaneMetadataError error =
      PrefillMLPFactorizedLaneMetadataError::kInvalidByteLength;
  std::uint32_t lane_count = 0U;
  std::uint64_t input_size = 0U;
  std::array<std::uint8_t,
             kPrefillA4FactorizedLaneMetadataDigestBytes>
      inverse_alpha_sha256{};
  std::vector<float> inverse_alpha;

  [[nodiscard]] bool valid() const noexcept {
    return error == PrefillMLPFactorizedLaneMetadataError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return valid(); }
};

// Serialization emits exactly the metadata bytes described by
// prefill_a4_factorized_lane_contract.h, excluding projection-end padding.
[[nodiscard]] PrefillMLPFactorizedLaneMetadataSerializationResult
serialize_prefill_mlp_factorized_lane_metadata(
    std::uint32_t lane_count, const float* inverse_alpha,
    std::size_t inverse_alpha_count);

// Parsing requires the independently authenticated expected lane/K.  This is
// intentional: the metadata SHA covers the exact inverse-alpha byte payload,
// while the future manifest/policy/receipt bind the header shape.  Requiring
// both here prevents a mutated but otherwise structurally valid lane/K header
// from being accepted in isolation.
[[nodiscard]] PrefillMLPFactorizedLaneMetadataParseResult
parse_prefill_mlp_factorized_lane_metadata(
    const std::uint8_t* bytes, std::size_t byte_count,
    std::uint32_t expected_lane_count, std::uint64_t expected_input_size);

// Future publication-binding vocabulary.  These structs are deliberately
// data-only in v4: no converter, loader, or runtime route consumes them yet.
struct PrefillMLPFactorizedLaneBaseK256Binding final {
  std::string physical_layout;
  std::uint32_t packed_k_group_size =
      kPrefillMLPFactorizedLaneRequiredBasePackedK;
  std::uint32_t scale_group_size =
      kPrefillMLPFactorizedLaneRequiredBaseScaleK;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::string payload_sha256;
  // SHA-256 of the exact strict K256 receipt JSON bytes.  The ordinary base
  // identities above bind what the receipt says; this field additionally
  // binds the exact authenticated hand-off document used by the derivative
  // converter.
  std::string receipt_sha256;
};

struct PrefillMLPFactorizedLaneFactorSourceBinding final {
  // R1 uses "identity_alpha_f32_v1" with an empty path because the converter
  // deterministically materializes FP32 one[K].  Future calibrated R4
  // publications must provide an authenticated external source path.
  std::string scheme;
  std::string path;
  std::string sha256;
  std::uint64_t element_count = 0U;
};

struct PrefillMLPFactorizedLanePayloadIdentity final {
  std::string path;
  std::string sha256;
  std::uint64_t bytes = 0U;
};

struct PrefillMLPFactorizedLaneManifestProjection final {
  std::uint32_t ordinal = 0U;
  std::uint32_t layer_index = 0U;
  PrefillMLPFactorizedLaneProjectionFamily family =
      PrefillMLPFactorizedLaneProjectionFamily::kGate;
  std::string source_module;
  std::string source_sha256;
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
};

struct PrefillMLPFactorizedLaneOverlayManifestBinding final {
  std::uint32_t version_major =
      kPrefillMLPFactorizedLaneOverlayVersionMajor;
  std::uint32_t version_minor =
      kPrefillMLPFactorizedLaneOverlayVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  PrefillMLPFactorizedLaneBaseK256Binding required_base_k256;
  std::uint32_t lane_count = 0U;
  std::vector<PrefillMLPFactorizedLaneManifestProjection> projections;
  std::uint64_t payload_bytes = 0U;
  std::string manifest_sha256;
};

struct PrefillMLPFactorizedLaneProjectionCalibrationBinding final {
  std::uint32_t ordinal = 0U;
  std::string source_module;
  std::string source_sha256;
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  PrefillMLPFactorizedLaneFactorSourceBinding factor_source;
};

struct PrefillMLPFactorizedLaneOverlayPolicyBinding final {
  std::uint32_t version_major =
      kPrefillMLPFactorizedLaneOverlayVersionMajor;
  std::uint32_t version_minor =
      kPrefillMLPFactorizedLaneOverlayVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string manifest_sha256;
  PrefillMLPFactorizedLaneBaseK256Binding required_base_k256;
  std::uint32_t lane_count = 0U;
  std::vector<PrefillMLPFactorizedLaneProjectionCalibrationBinding>
      projections;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
};

struct PrefillMLPFactorizedLaneOverlayReceiptBinding final {
  std::uint32_t version_major =
      kPrefillMLPFactorizedLaneOverlayVersionMajor;
  std::uint32_t version_minor =
      kPrefillMLPFactorizedLaneOverlayVersionMinor;
  // Must remain false until a real-checkpoint converter, authenticated loader,
  // accuracy gate, and API/EvalScope performance gate are implemented.
  bool production_residency_eligible = false;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  PrefillMLPFactorizedLaneBaseK256Binding required_base_k256;
  std::uint32_t lane_count = 0U;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
  PrefillMLPFactorizedLanePayloadIdentity payload;
  std::uint64_t projection_count = 0U;
};

static_assert(prefill_mlp_factorized_lane_overlay_layout_plan(1U).valid());
static_assert(prefill_mlp_factorized_lane_overlay_layout_plan(2U).valid());
static_assert(prefill_mlp_factorized_lane_overlay_layout_plan(4U).valid());

}  // namespace q3x::runtime
