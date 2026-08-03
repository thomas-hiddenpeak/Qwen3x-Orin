#pragma once

#include "q3x/runtime/prefill_a4_sidecar_converter.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_overlay.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace q3x::runtime {

inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR1ConverterAbi =
        "q3x_k256_to_mlp_factorized_r1_v1";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR1FactorScheme =
        "identity_alpha_f32_v1";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR1Mode =
        "performance_upper_bound_r1";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR1EligibilityScope =
        "authenticated_abi_only";
inline constexpr std::uint32_t kPrefillMLPFactorizedLaneR1LaneCount = 1U;
inline constexpr std::uint64_t
    kPrefillMLPFactorizedLaneR1PayloadBytes = 8'568'619'008ULL;

enum class PrefillMLPFactorizedLaneConverterErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidOption,
  kInvalidManifest,
  kInvalidBaseReceipt,
  kBaseAuthenticationFailed,
  kSourceBindingMismatch,
  kInvalidPolicy,
  kInvalidReceipt,
  kUnsafePath,
  kOpenFailed,
  kIoFailure,
  kArithmeticOverflow,
  kQuantizationFailure,
  kDigestMismatch,
  kPublicationConflict,
  kAllocationFailure,
};

struct PrefillMLPFactorizedLaneConverterDiagnostic final {
  PrefillMLPFactorizedLaneConverterErrorCode code =
      PrefillMLPFactorizedLaneConverterErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;
  int system_error = 0;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillMLPFactorizedLaneConverterErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct PrefillMLPFactorizedLaneManifestResult final {
  std::optional<PrefillMLPFactorizedLaneOverlayManifestBinding> value;
  PrefillMLPFactorizedLaneConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

// Builds the fixed 192-entry derivative manifest from a validated, exact K256
// base publication.  base_receipt_sha256 is over the exact receipt bytes read
// by the converter, not a reconstructed JSON document.
[[nodiscard]] PrefillMLPFactorizedLaneManifestResult
build_prefill_mlp_factorized_lane_r1_manifest(
    const PrefillSidecarManifest& base_k256_manifest,
    const PrefillA4PublicationReceipt& base_k256_receipt,
    std::string_view base_receipt_sha256);

[[nodiscard]] PrefillMLPFactorizedLaneConverterDiagnostic
validate_prefill_mlp_factorized_lane_r1_manifest(
    const PrefillMLPFactorizedLaneOverlayManifestBinding& manifest);

// Exposed for deterministic tooling and tests.  The hash excludes the
// manifest_sha256 field itself and includes every ordered projection entry.
[[nodiscard]] std::string
prefill_mlp_factorized_lane_r1_manifest_sha256(
    const PrefillMLPFactorizedLaneOverlayManifestBinding& manifest);

struct PrefillMLPFactorizedLaneR1Policy final {
  PrefillMLPFactorizedLaneOverlayPolicyBinding binding;
  std::string converter_abi;
  std::string mode;
  bool performance_upper_bound_only = true;
  bool quality_production_eligible = false;
};

struct PrefillMLPFactorizedLaneR1PolicyResult final {
  std::optional<PrefillMLPFactorizedLaneR1Policy> value;
  // Populated by the canonical builder; parsers leave this empty.
  std::string canonical_document;
  PrefillMLPFactorizedLaneConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPFactorizedLaneR1PolicyResult
build_prefill_mlp_factorized_lane_r1_policy(
    const PrefillMLPFactorizedLaneOverlayManifestBinding& manifest,
    double weight_clip_ratio, double activation_clip_ratio);

// Strict schema: unknown, duplicate, missing, reordered-projection identity,
// non-identity factor, or any base/manifest mutation fails closed.
[[nodiscard]] PrefillMLPFactorizedLaneR1PolicyResult
parse_prefill_mlp_factorized_lane_r1_policy(
    std::string_view document,
    const PrefillMLPFactorizedLaneOverlayManifestBinding& manifest);

struct PrefillMLPFactorizedLaneR1Receipt final {
  PrefillMLPFactorizedLaneOverlayReceiptBinding binding;
  std::string converter_abi;
  std::string mode;
  std::string residency_eligibility_scope;
  bool performance_upper_bound_only = true;
  bool quality_production_eligible = false;
};

struct PrefillMLPFactorizedLaneR1ReceiptResult final {
  std::optional<PrefillMLPFactorizedLaneR1Receipt> value;
  // Populated by the canonical builder; parsers leave this empty.
  std::string canonical_document;
  PrefillMLPFactorizedLaneConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPFactorizedLaneR1ReceiptResult
build_prefill_mlp_factorized_lane_r1_receipt(
    const PrefillMLPFactorizedLaneOverlayManifestBinding& manifest,
    const PrefillMLPFactorizedLaneR1Policy& policy,
    std::string_view payload_sha256);

[[nodiscard]] PrefillMLPFactorizedLaneR1ReceiptResult
parse_prefill_mlp_factorized_lane_r1_receipt(
    std::string_view document,
    const PrefillMLPFactorizedLaneOverlayManifestBinding& manifest,
    const PrefillMLPFactorizedLaneR1Policy& policy);

// Pure-host transform for one or more complete N64 blocks.  Source bytes are
// the authenticated K256 consumer layout:
//   packed [N/64][K/64][64][32]
//   scales [N/64][K/256][64] BF16 LE
// Output is the factorized R1 layout:
//   packed [N/64][K/64][64][32]
//   scales [N/64][1][64] BF16 LE
// Codes are decoded to FP32 using the stored K256 BF16 scales and requantized
// with one full-K scale per row, nearest-even, and an explicit clip ratio.
[[nodiscard]] PrefillMLPFactorizedLaneConverterDiagnostic
transform_prefill_mlp_k256_to_factorized_r1_consumer_blocks(
    const std::uint8_t* base_packed_signed_w4,
    std::size_t base_packed_signed_w4_bytes,
    const std::uint8_t* base_bf16_scales_little_endian,
    std::size_t base_bf16_scale_bytes, std::size_t row_count,
    std::size_t input_size, double weight_clip_ratio,
    std::uint8_t* r1_packed_signed_w4,
    std::size_t r1_packed_signed_w4_bytes,
    std::uint8_t* r1_bf16_scales_little_endian,
    std::size_t r1_bf16_scale_bytes);

struct PrefillMLPFactorizedLaneR1ConversionOptions final {
  std::filesystem::path model_directory;
  std::filesystem::path base_k256_payload_path;
  std::filesystem::path base_k256_policy_path;
  std::filesystem::path base_k256_receipt_path;
  std::filesystem::path output_path;
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  std::uint64_t max_base_receipt_bytes = 64ULL * 1024ULL;
  bool preallocate_output = true;
};

struct PrefillMLPFactorizedLaneR1ConversionStats final {
  std::uint64_t base_bytes_read = 0U;
  std::uint64_t output_bytes_written = 0U;
  std::uint64_t peak_working_bytes = 0U;
  std::uint64_t projections_converted = 0U;
  std::uint64_t n64_blocks_converted = 0U;
};

struct PrefillMLPFactorizedLaneR1ConversionResult final {
  std::optional<PrefillMLPFactorizedLaneOverlayManifestBinding> manifest;
  std::optional<PrefillMLPFactorizedLaneR1Policy> policy;
  std::optional<PrefillMLPFactorizedLaneR1Receipt> receipt;
  PrefillMLPFactorizedLaneR1ConversionStats stats;
  PrefillMLPFactorizedLaneConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return manifest.has_value() && policy.has_value() &&
           receipt.has_value() && diagnostic.ok();
  }
};

// Publishes output_path, output_path + ".policy.json", and output_path +
// ".receipt.json" as one rollback-on-error, hard-link/no-replace set.  The
// same authenticated and locked K256 payload fd returned by the base
// residency gate is consumed and finally revalidated unchanged.
[[nodiscard]] PrefillMLPFactorizedLaneR1ConversionResult
convert_authenticated_k256_to_prefill_mlp_factorized_lane_r1(
    const PrefillMLPFactorizedLaneR1ConversionOptions& options);

[[nodiscard]] std::string_view to_string(
    PrefillMLPFactorizedLaneConverterErrorCode code) noexcept;

}  // namespace q3x::runtime
