#pragma once

#include "q3x/runtime/prefill_mlp_factorized_lane_overlay.h"
#include "q3x/runtime/prefill_quantized_contract.h"

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

// This is an independent direct-checkpoint publication ABI.  In particular,
// it is not an R1/K256 publication with an empty base field.  The only source
// admitted by this contract is a validated Exact PrefillSidecarManifest,
// whose identity is copied verbatim into DirectSourceManifestBinding.
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneR4PublicationVersionMajor = 1U;
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneR4PublicationVersionMinor = 0U;
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneR4PublicationLaneCount = 4U;
inline constexpr std::uint64_t
    kPrefillMLPFactorizedLaneR4PublicationPayloadBytes =
        8'583'954'432ULL;
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4DirectSourceKind = "exact";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4PublicationFactorScheme =
        "calibrated_alpha_f32_v1";
inline constexpr double
    kPrefillMLPFactorizedLaneR4PublicationMinimumClipRatio = 1.0 / 256.0;
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4PublicationAbi =
        "q3x_checkpoint_to_mlp_factorized_r4_publication_v1";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4PublicationMode =
        "direct_checkpoint_performance_candidate_r4";

static_assert(
    prefill_mlp_factorized_lane_overlay_layout_plan(
        kPrefillMLPFactorizedLaneR4PublicationLaneCount)
            .payload_bytes ==
    kPrefillMLPFactorizedLaneR4PublicationPayloadBytes);

enum class PrefillMLPFactorizedLaneR4PublicationErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidSourceManifest,
  kInvalidManifest,
  kInvalidPolicy,
  kInvalidReceipt,
  kSourceBindingMismatch,
  kDigestMismatch,
  kAllocationFailure,
};

struct PrefillMLPFactorizedLaneR4PublicationDiagnostic final {
  PrefillMLPFactorizedLaneR4PublicationErrorCode code =
      PrefillMLPFactorizedLaneR4PublicationErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillMLPFactorizedLaneR4PublicationErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct PrefillMLPFactorizedLaneR4DirectSourceManifestBinding final {
  // This field is fixed to "exact"; it makes the source epoch explicit
  // without importing a derivative K256 publication vocabulary.
  std::string source_manifest_kind;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string source_manifest_sha256;
};

struct PrefillMLPFactorizedLaneR4Manifest final {
  std::uint32_t version_major =
      kPrefillMLPFactorizedLaneR4PublicationVersionMajor;
  std::uint32_t version_minor =
      kPrefillMLPFactorizedLaneR4PublicationVersionMinor;
  std::string physical_layout;
  PrefillMLPFactorizedLaneR4DirectSourceManifestBinding direct_source;
  std::uint32_t lane_count = 0U;
  std::vector<PrefillMLPFactorizedLaneManifestProjection> projections;
  std::uint64_t payload_bytes = 0U;
  // Digest over the canonical manifest body before the two publication
  // identity fields below.  manifest_bytes is the complete canonical JSON.
  std::string manifest_sha256;
  std::uint64_t manifest_bytes = 0U;
};

struct PrefillMLPFactorizedLaneR4ManifestResult final {
  std::optional<PrefillMLPFactorizedLaneR4Manifest> value;
  std::string canonical_document;
  PrefillMLPFactorizedLaneR4PublicationDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPFactorizedLaneR4ManifestResult
build_prefill_mlp_factorized_lane_r4_direct_manifest(
    const PrefillSidecarManifest& exact_source_manifest);

// Parsing is deliberately tied to the caller's already pinned Exact source
// manifest.  A valid-looking document from another checkpoint fails closed.
[[nodiscard]] PrefillMLPFactorizedLaneR4ManifestResult
parse_prefill_mlp_factorized_lane_r4_direct_manifest(
    std::string_view document,
    const PrefillSidecarManifest& exact_source_manifest);

[[nodiscard]] PrefillMLPFactorizedLaneR4PublicationDiagnostic
validate_prefill_mlp_factorized_lane_r4_direct_manifest(
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const PrefillSidecarManifest& exact_source_manifest);

[[nodiscard]] std::string
prefill_mlp_factorized_lane_r4_manifest_sha256(
    const PrefillMLPFactorizedLaneR4Manifest& manifest);

struct PrefillMLPFactorizedLaneR4CalibrationSpec final {
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  std::string alpha_path;
  std::string alpha_sha256;
  std::uint64_t alpha_element_count = 0U;
};

struct PrefillMLPFactorizedLaneR4ProjectionPolicyBinding final {
  std::uint32_t ordinal = 0U;
  std::string source_module;
  std::string source_sha256;
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  std::string factor_scheme;
  std::string factor_path;
  std::string factor_sha256;
  std::uint64_t factor_element_count = 0U;
};

struct PrefillMLPFactorizedLaneR4Policy final {
  std::uint32_t version_major =
      kPrefillMLPFactorizedLaneR4PublicationVersionMajor;
  std::uint32_t version_minor =
      kPrefillMLPFactorizedLaneR4PublicationVersionMinor;
  std::string mode;
  std::string converter_abi;
  bool performance_candidate_only = true;
  bool production_residency_eligible = false;
  bool quality_production_eligible = false;
  std::string physical_layout;
  PrefillMLPFactorizedLaneR4DirectSourceManifestBinding direct_source;
  std::string manifest_sha256;
  std::uint64_t manifest_bytes = 0U;
  std::uint32_t lane_count = 0U;
  std::vector<PrefillMLPFactorizedLaneR4ProjectionPolicyBinding> projections;
  // These identify the complete canonical policy JSON and are not serialized
  // into that JSON, avoiding a self-referential digest.
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
};

struct PrefillMLPFactorizedLaneR4PolicyResult final {
  std::optional<PrefillMLPFactorizedLaneR4Policy> value;
  std::string canonical_document;
  PrefillMLPFactorizedLaneR4PublicationDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

// calibration must contain exactly 192 layer-major Gate/Up/Down entries.
// Gate and Up in one layer must carry byte-identical alpha identity and an
// identical activation clip ratio.
[[nodiscard]] PrefillMLPFactorizedLaneR4PolicyResult
build_prefill_mlp_factorized_lane_r4_policy(
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const std::vector<PrefillMLPFactorizedLaneR4CalibrationSpec>&
        calibration);

[[nodiscard]] PrefillMLPFactorizedLaneR4PolicyResult
parse_prefill_mlp_factorized_lane_r4_policy(
    std::string_view document,
    const PrefillMLPFactorizedLaneR4Manifest& manifest);

// Revalidates a strict-parser output without needing the original 400-entry
// source manifest.  The direct source identity remains bound by the R4
// manifest digest; loaders that still own the Exact source must additionally
// use validate_prefill_mlp_factorized_lane_r4_direct_manifest().
[[nodiscard]] PrefillMLPFactorizedLaneR4PublicationDiagnostic
validate_prefill_mlp_factorized_lane_r4_policy_binding(
    const PrefillMLPFactorizedLaneR4Policy& policy,
    const PrefillMLPFactorizedLaneR4Manifest& manifest);

struct PrefillMLPFactorizedLaneR4Receipt final {
  std::uint32_t version_major =
      kPrefillMLPFactorizedLaneR4PublicationVersionMajor;
  std::uint32_t version_minor =
      kPrefillMLPFactorizedLaneR4PublicationVersionMinor;
  std::string mode;
  std::string converter_abi;
  bool performance_candidate_only = true;
  bool production_residency_eligible = false;
  bool quality_production_eligible = false;
  std::string physical_layout;
  PrefillMLPFactorizedLaneR4DirectSourceManifestBinding direct_source;
  std::uint32_t lane_count = 0U;
  std::string manifest_sha256;
  std::uint64_t manifest_bytes = 0U;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
  std::string payload_sha256;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t projection_count = 0U;
};

struct PrefillMLPFactorizedLaneR4ReceiptResult final {
  std::optional<PrefillMLPFactorizedLaneR4Receipt> value;
  std::string canonical_document;
  PrefillMLPFactorizedLaneR4PublicationDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPFactorizedLaneR4ReceiptResult
build_prefill_mlp_factorized_lane_r4_receipt(
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const PrefillMLPFactorizedLaneR4Policy& policy,
    std::string_view payload_sha256);

[[nodiscard]] PrefillMLPFactorizedLaneR4ReceiptResult
parse_prefill_mlp_factorized_lane_r4_receipt(
    std::string_view document,
    const PrefillMLPFactorizedLaneR4Manifest& manifest,
    const PrefillMLPFactorizedLaneR4Policy& policy);

[[nodiscard]] std::string_view to_string(
    PrefillMLPFactorizedLaneR4PublicationErrorCode code) noexcept;

}  // namespace q3x::runtime
