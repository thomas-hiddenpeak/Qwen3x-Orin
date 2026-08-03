#pragma once

#include "q3x/runtime/prefill_attention_factorized_lane_converter.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_converter.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

// Host-only publication boundary for the complete R1 projection plane.  It
// changes only the physical order of already-authenticated signed-S4 bytes.
// It never decodes, clips, rescales, or requantizes a weight.
inline constexpr std::uint32_t kPrefillR1ProjectionPlaneV2VersionMajor = 2U;
inline constexpr std::uint32_t kPrefillR1ProjectionPlaneV2VersionMinor = 0U;
inline constexpr std::string_view kPrefillR1ProjectionPlaneV2Layout =
    "sm87_s4_r1_projection_plane_fragment_pair_v2";
inline constexpr std::uint32_t
    kPrefillR1ProjectionPlaneV2LogicalProjectionCount = 400U;
inline constexpr std::uint32_t
    kPrefillR1ProjectionPlaneV2PhysicalProjectionCount = 336U;
inline constexpr std::uint64_t kPrefillR1ProjectionPlaneV2PayloadBytes =
    12'182'982'656ULL;
inline constexpr std::uint64_t kPrefillR1ProjectionPlaneV2Alignment = 256U;
inline constexpr std::uint64_t kPrefillR1ProjectionPlaneV2InvalidOffset =
    std::numeric_limits<std::uint64_t>::max();

static_assert(kPrefillR1ProjectionPlaneV2LogicalProjectionCount ==
              kPrefillMLPFactorizedLaneProjectionCount +
                  kPrefillAttentionFactorizedLaneProjectionCount);
static_assert(kPrefillR1ProjectionPlaneV2PhysicalProjectionCount ==
              kPrefillMLPFactorizedLaneLayerCount * 2U +
                  kPrefillAttentionFactorizedLaneProjectionCount);
static_assert(kPrefillR1ProjectionPlaneV2PayloadBytes ==
              prefill_mlp_factorized_lane_overlay_layout_plan(1U)
                      .payload_bytes +
                  kPrefillAttentionFactorizedLaneR1LayoutPlan.payload_bytes);

enum class PrefillR1ProjectionPlaneV2ErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidSourceManifest,
  kInvalidSourcePublication,
  kSourceBindingMismatch,
  kInvalidLayout,
  kInvalidPayload,
  kInvalidPolicy,
  kInvalidReceipt,
  kPartialInstallation,
  kLegacyCoResidency,
  kArithmeticOverflow,
  kAllocationFailure,
  kDigestFailure,
  kInvalidOption,
  kUnsafePath,
  kOpenFailed,
  kIoFailure,
  kPublicationConflict,
};

struct PrefillR1ProjectionPlaneV2Diagnostic final {
  PrefillR1ProjectionPlaneV2ErrorCode code =
      PrefillR1ProjectionPlaneV2ErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;
  int system_error = 0;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillR1ProjectionPlaneV2ErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// The caller has already authenticated the exact source triplet and held its
// payload stable.  The converter rechecks every identity and byte extent, but
// deliberately does not hash either 3.6/8.6 GiB source a second time.  The
// same identity carrier represents the completed resident v2 arena during
// atomic installation; its data pointer may then be a device address and is
// never dereferenced by the host validator.
struct PrefillR1ProjectionPlaneV2AuthenticatedPayloadView final {
  const std::uint8_t* data = nullptr;
  std::uint64_t bytes = 0U;
  std::uint32_t version_major = 0U;
  std::uint32_t version_minor = 0U;
  std::string physical_layout;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::string payload_sha256;
  std::string receipt_sha256;
  bool authenticated = false;
};

struct PrefillR1ProjectionPlaneV2MutablePayloadView final {
  std::uint8_t* data = nullptr;
  std::uint64_t bytes = 0U;
};

enum class PrefillR1ProjectionPlaneV2PhysicalFamily : std::uint8_t {
  kLinearQkv = 0,
  kLinearZ,
  kLinearO,
  kFullQ,
  kFullK,
  kFullV,
  kFullO,
  kMlpGateUp,
  kMlpDown,
};

enum class PrefillR1ProjectionPlaneV2LogicalFamily : std::uint8_t {
  kLinearQkv = 0,
  kLinearZ,
  kLinearO,
  kFullQ,
  kFullK,
  kFullV,
  kFullO,
  kMlpGate,
  kMlpUp,
  kMlpDown,
};

struct PrefillR1ProjectionPlaneV2SourceBinding final {
  std::uint32_t version_major = 0U;
  std::uint32_t version_minor = 0U;
  std::string physical_layout;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::string payload_sha256;
  std::string receipt_sha256;
  std::uint64_t payload_bytes = 0U;
};

struct PrefillR1ProjectionPlaneV2BaseK256Binding final {
  std::string physical_layout;
  std::uint32_t packed_k_group_size = 0U;
  std::uint32_t scale_group_size = 0U;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::string payload_sha256;
  std::string receipt_sha256;
};

// One physical entry is one launch-time B publication.  Gate+Up owns two
// logical projections; every other entry owns one.  source_secondary_offset
// is valid only for Gate+Up.
struct PrefillR1ProjectionPlaneV2PhysicalProjection final {
  std::uint32_t physical_ordinal = 0U;
  std::uint32_t layer_index = 0U;
  PrefillR1ProjectionPlaneV2PhysicalFamily family =
      PrefillR1ProjectionPlaneV2PhysicalFamily::kLinearQkv;
  std::uint32_t logical_projection_count = 0U;
  std::uint32_t source_primary_ordinal = 0U;
  std::uint32_t source_secondary_ordinal =
      std::numeric_limits<std::uint32_t>::max();
  std::uint64_t source_primary_offset = 0U;
  std::uint64_t source_secondary_offset = 0U;
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
  std::uint64_t payload_offset = 0U;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t packed_weight_offset = 0U;
  std::uint64_t packed_weight_bytes = 0U;
  std::uint64_t weight_scale_offset = 0U;
  std::uint64_t weight_scale_bytes = 0U;
  std::uint64_t primary_metadata_offset = 0U;
  std::uint64_t primary_metadata_bytes = 0U;
  std::uint64_t secondary_metadata_offset = 0U;
  std::uint64_t secondary_metadata_bytes = 0U;
};

struct PrefillR1ProjectionPlaneV2LogicalProjection final {
  std::uint32_t logical_ordinal = 0U;
  std::uint32_t layer_index = 0U;
  PrefillR1ProjectionPlaneV2LogicalFamily family =
      PrefillR1ProjectionPlaneV2LogicalFamily::kLinearQkv;
  std::uint32_t physical_ordinal = 0U;
  std::uint32_t source_ordinal = 0U;
  std::string source_module;
  std::string source_sha256;
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
};

struct PrefillR1ProjectionPlaneV2Manifest final {
  std::uint32_t version_major =
      kPrefillR1ProjectionPlaneV2VersionMajor;
  std::uint32_t version_minor =
      kPrefillR1ProjectionPlaneV2VersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::uint32_t lane_count = 1U;
  PrefillR1ProjectionPlaneV2BaseK256Binding required_base_k256;
  PrefillR1ProjectionPlaneV2SourceBinding mlp_v4;
  PrefillR1ProjectionPlaneV2SourceBinding attention_v4;
  std::vector<PrefillR1ProjectionPlaneV2PhysicalProjection> projections;
  std::vector<PrefillR1ProjectionPlaneV2LogicalProjection>
      logical_projections;
  std::uint32_t logical_projection_count = 0U;
  std::uint32_t physical_projection_count = 0U;
  std::uint64_t payload_bytes = 0U;
  std::string manifest_sha256;
};

struct PrefillR1ProjectionPlaneV2ManifestResult final {
  std::optional<PrefillR1ProjectionPlaneV2Manifest> value;
  std::string canonical_document;
  PrefillR1ProjectionPlaneV2Diagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillR1ProjectionPlaneV2ManifestResult
build_prefill_r1_projection_plane_v2_manifest(
    const PrefillMLPFactorizedLaneOverlayManifestBinding& mlp_v4,
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView& mlp_payload,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& attention_v4,
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView&
        attention_payload);

[[nodiscard]] PrefillR1ProjectionPlaneV2ManifestResult
parse_prefill_r1_projection_plane_v2_manifest(std::string_view document);

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_manifest(
    const PrefillR1ProjectionPlaneV2Manifest& manifest);

[[nodiscard]] std::string prefill_r1_projection_plane_v2_manifest_sha256(
    const PrefillR1ProjectionPlaneV2Manifest& manifest);

struct PrefillR1ProjectionPlaneV2Policy final {
  std::uint32_t version_major =
      kPrefillR1ProjectionPlaneV2VersionMajor;
  std::uint32_t version_minor =
      kPrefillR1ProjectionPlaneV2VersionMinor;
  std::string physical_layout;
  std::string manifest_sha256;
  std::string mlp_source_policy_sha256;
  std::string attention_source_policy_sha256;
  std::string converter_abi;
  std::string mode;
  bool performance_upper_bound_only = true;
  bool quality_production_eligible = false;
  bool production_residency_eligible = false;
  struct Calibration final {
    std::uint32_t logical_ordinal = 0U;
    std::uint32_t physical_ordinal = 0U;
    std::uint32_t source_ordinal = 0U;
    std::string source_module;
    std::string source_sha256;
    double weight_clip_ratio = 0.0;
    double activation_clip_ratio = 0.0;
    std::string factor_scheme;
    std::string factor_path;
    std::string factor_sha256;
    std::uint64_t factor_element_count = 0U;
  };
  std::vector<Calibration> projections;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
};

struct PrefillR1ProjectionPlaneV2PolicyResult final {
  std::optional<PrefillR1ProjectionPlaneV2Policy> value;
  std::string canonical_document;
  PrefillR1ProjectionPlaneV2Diagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillR1ProjectionPlaneV2PolicyResult
build_prefill_r1_projection_plane_v2_policy(
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillMLPFactorizedLaneR1Policy& mlp_v4_policy,
    const PrefillAttentionFactorizedLaneR1Policy& attention_v4_policy);

[[nodiscard]] PrefillR1ProjectionPlaneV2PolicyResult
parse_prefill_r1_projection_plane_v2_policy(
    std::string_view document,
    const PrefillR1ProjectionPlaneV2Manifest& manifest);

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_policy(
    const PrefillR1ProjectionPlaneV2Policy& policy,
    const PrefillR1ProjectionPlaneV2Manifest& manifest);

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_policy_sources(
    const PrefillR1ProjectionPlaneV2Policy& policy,
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillMLPFactorizedLaneR1Policy& mlp_v4_policy,
    const PrefillAttentionFactorizedLaneR1Policy& attention_v4_policy);

struct PrefillR1ProjectionPlaneV2Receipt final {
  std::uint32_t version_major =
      kPrefillR1ProjectionPlaneV2VersionMajor;
  std::uint32_t version_minor =
      kPrefillR1ProjectionPlaneV2VersionMinor;
  std::string physical_layout;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
  std::string payload_sha256;
  std::uint64_t payload_bytes = 0U;
  PrefillR1ProjectionPlaneV2BaseK256Binding required_base_k256;
  PrefillR1ProjectionPlaneV2SourceBinding mlp_v4;
  PrefillR1ProjectionPlaneV2SourceBinding attention_v4;
  std::uint32_t logical_projection_count = 0U;
  std::uint32_t physical_projection_count = 0U;
  bool atomic_installation_required = true;
  bool legacy_r1_co_residency_allowed = false;
  bool performance_upper_bound_only = true;
  bool quality_production_eligible = false;
  bool production_residency_eligible = false;
};

struct PrefillR1ProjectionPlaneV2ReceiptResult final {
  std::optional<PrefillR1ProjectionPlaneV2Receipt> value;
  std::string canonical_document;
  std::string canonical_sha256;
  PrefillR1ProjectionPlaneV2Diagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillR1ProjectionPlaneV2ReceiptResult
build_prefill_r1_projection_plane_v2_receipt(
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillR1ProjectionPlaneV2Policy& policy,
    std::string_view payload_sha256);

[[nodiscard]] PrefillR1ProjectionPlaneV2ReceiptResult
parse_prefill_r1_projection_plane_v2_receipt(
    std::string_view document,
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillR1ProjectionPlaneV2Policy& policy);

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_receipt(
    const PrefillR1ProjectionPlaneV2Receipt& receipt,
    const PrefillR1ProjectionPlaneV2Manifest& manifest,
    const PrefillR1ProjectionPlaneV2Policy& policy);

// The triplet is indivisible at residency time.  A v2 payload may not be
// mounted with either legacy split R1 triplet, even if every individual hash
// is otherwise valid.
struct PrefillR1ProjectionPlaneV2Installation final {
  const PrefillR1ProjectionPlaneV2Manifest* manifest = nullptr;
  const PrefillR1ProjectionPlaneV2Policy* policy = nullptr;
  const PrefillR1ProjectionPlaneV2Receipt* receipt = nullptr;
  const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView* payload = nullptr;
  bool legacy_mlp_r1_installed = false;
  bool legacy_attention_r1_installed = false;
};

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic
validate_prefill_r1_projection_plane_v2_atomic_installation(
    const PrefillR1ProjectionPlaneV2Installation& installation);

// Fragment-native code offsets shared by the publication tests.  Kernel
// headers intentionally own their execution-side mirrors until the complete
// v2 package is wired, preventing an experimental runtime dependency.
[[nodiscard]] constexpr std::uint64_t
prefill_r1_projection_plane_v2_gate_up_code_slot_offset(
    const std::uint64_t output_coordinate, const std::uint64_t k64_group,
    const std::uint64_t n8_fragment, const std::uint64_t lane,
    const std::uint64_t output_size,
    const std::uint64_t input_size) noexcept {
  if (output_size == 0U || output_size % 64U != 0U || input_size == 0U ||
      input_size % 64U != 0U || output_coordinate >= output_size ||
      k64_group >= input_size / 64U || n8_fragment >= 8U || lane >= 32U ||
      n8_fragment != (output_coordinate % 64U) / 8U) {
    return kPrefillR1ProjectionPlaneV2InvalidOffset;
  }
  return (((output_coordinate / 64U) * (input_size / 64U) + k64_group) *
                  8U * 32U +
              n8_fragment * 32U + lane) *
         16U;
}

[[nodiscard]] constexpr std::uint64_t
prefill_r1_projection_plane_v2_adjacent_n8_code_slot_offset(
    const std::uint64_t n128_panel, const std::uint64_t k64_group,
    const std::uint64_t n16_warp, const std::uint64_t lane,
    const std::uint64_t output_size,
    const std::uint64_t input_size) noexcept {
  if (output_size == 0U || output_size % 128U != 0U || input_size == 0U ||
      input_size % 64U != 0U || n128_panel >= output_size / 128U ||
      k64_group >= input_size / 64U || n16_warp >= 8U || lane >= 32U) {
    return kPrefillR1ProjectionPlaneV2InvalidOffset;
  }
  return (((n128_panel * (input_size / 64U) + k64_group) * 8U * 32U +
           n16_warp * 32U + lane) *
          16U);
}

// Pure-host, equal-byte transforms used by the complete converter and small
// correctness fixtures.  Source projection buffers use the v4 factorized R1
// layout.  Gate/Up output uses one physical paired record; adjacent-N8 output
// keeps the ordinary one-projection extent.  All scale and metadata values
// are copied bit-for-bit.
[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic
permute_prefill_r1_gate_up_projection_pair_v2(
    const std::uint8_t* gate_v4, std::size_t gate_v4_bytes,
    const std::uint8_t* up_v4, std::size_t up_v4_bytes,
    std::uint64_t output_size, std::uint64_t input_size,
    std::uint8_t* gate_up_v2, std::size_t gate_up_v2_bytes);

[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic
permute_prefill_r1_adjacent_n8_projection_v2(
    const std::uint8_t* source_v4, std::size_t source_v4_bytes,
    std::uint64_t output_size, std::uint64_t input_size,
    std::uint8_t* destination_v2, std::size_t destination_v2_bytes);

struct PrefillR1ProjectionPlaneV2ConversionResult final {
  std::optional<PrefillR1ProjectionPlaneV2Manifest> manifest;
  std::optional<PrefillR1ProjectionPlaneV2Policy> policy;
  std::optional<PrefillR1ProjectionPlaneV2Receipt> receipt;
  std::string manifest_document;
  std::string policy_document;
  std::string receipt_document;
  std::string payload_sha256;
  std::uint64_t bytes_written = 0U;
  std::uint32_t logical_projections_written = 0U;
  std::uint32_t physical_projections_written = 0U;
  PrefillR1ProjectionPlaneV2Diagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return manifest.has_value() && policy.has_value() &&
           receipt.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillR1ProjectionPlaneV2ConversionResult
convert_authenticated_prefill_r1_projection_plane_v4_to_v2(
    const PrefillMLPFactorizedLaneOverlayManifestBinding& mlp_v4,
    const PrefillMLPFactorizedLaneR1Policy& mlp_v4_policy,
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView& mlp_payload,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& attention_v4,
    const PrefillAttentionFactorizedLaneR1Policy& attention_v4_policy,
    const PrefillR1ProjectionPlaneV2AuthenticatedPayloadView&
        attention_payload,
    const PrefillR1ProjectionPlaneV2MutablePayloadView& output);

// Offline file composition entry point. Every input is explicit and is
// opened without following symlinks, locked, authenticated, and rehashed
// after the byte permutation. Temporary files are created beside OUTPUT;
// nothing is staged in /tmp. The four final targets are installed as one
// rollback-on-error, no-replace set.
struct PrefillR1ProjectionPlaneV2FileConversionOptions final {
  std::filesystem::path model_directory;
  std::filesystem::path base_k256_payload_path;
  std::filesystem::path base_k256_policy_path;
  std::filesystem::path base_k256_receipt_path;
  std::filesystem::path mlp_r1_payload_path;
  std::filesystem::path mlp_r1_policy_path;
  std::filesystem::path mlp_r1_receipt_path;
  std::filesystem::path attention_r1_payload_path;
  std::filesystem::path attention_r1_policy_path;
  std::filesystem::path attention_r1_receipt_path;
  std::filesystem::path output_path;
  std::uint64_t max_document_bytes = 4ULL * 1024ULL * 1024ULL;
  bool preallocate_output = true;
};

struct PrefillR1ProjectionPlaneV2FileConversionStats final {
  std::uint64_t source_bytes_hashed = 0U;
  std::uint64_t output_bytes_written = 0U;
  std::uint32_t logical_projections_written = 0U;
  std::uint32_t physical_projections_written = 0U;
};

struct PrefillR1ProjectionPlaneV2FileConversionResult final {
  std::optional<PrefillR1ProjectionPlaneV2Manifest> manifest;
  std::optional<PrefillR1ProjectionPlaneV2Policy> policy;
  std::optional<PrefillR1ProjectionPlaneV2Receipt> receipt;
  std::string manifest_document;
  std::string policy_document;
  std::string receipt_document;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::string payload_sha256;
  std::string receipt_sha256;
  std::string mlp_source_payload_sha256;
  std::string mlp_source_receipt_sha256;
  std::string attention_source_payload_sha256;
  std::string attention_source_receipt_sha256;
  PrefillR1ProjectionPlaneV2FileConversionStats stats;
  PrefillR1ProjectionPlaneV2Diagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return manifest.has_value() && policy.has_value() &&
           receipt.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillR1ProjectionPlaneV2FileConversionResult
convert_authenticated_prefill_r1_projection_plane_v2_files(
    const PrefillR1ProjectionPlaneV2FileConversionOptions& options);

// Transactional primitive shared by the converter and its small host-only
// rollback test. Each temporary file must be a sealed regular file in the
// same directory as its target. Existing targets are never replaced. On any
// link failure all targets installed by this call are removed.
[[nodiscard]] PrefillR1ProjectionPlaneV2Diagnostic
publish_prefill_r1_projection_plane_v2_file_set_no_replace(
    const std::array<std::filesystem::path, 4U>& temporary_paths,
    const std::array<std::filesystem::path, 4U>& target_paths);

[[nodiscard]] std::string_view to_string(
    PrefillR1ProjectionPlaneV2ErrorCode code) noexcept;

}  // namespace q3x::runtime
