#pragma once

#include "q3x/runtime/prefill_a4_sidecar_converter.h"
#include "q3x/runtime/prefill_attention_factorized_lane_overlay.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace q3x::runtime {

inline constexpr std::string_view
    kPrefillAttentionFactorizedLaneR1ConverterAbi =
        "q3x_k256_to_attention_factorized_r1_v1";
inline constexpr std::string_view
    kPrefillAttentionFactorizedLaneR1FactorScheme =
        "identity_alpha_f32_v1";
inline constexpr std::string_view kPrefillAttentionFactorizedLaneR1Mode =
    "performance_upper_bound_r1";
inline constexpr std::string_view
    kPrefillAttentionFactorizedLaneR1EligibilityScope =
        "authenticated_abi_only";
inline constexpr std::uint32_t
    kPrefillAttentionFactorizedLaneR1LaneCount = 1U;
inline constexpr std::uint64_t
    kPrefillAttentionFactorizedLaneR1PayloadBytes =
        kPrefillAttentionFactorizedLaneR1LayoutPlan.payload_bytes;
static_assert(kPrefillAttentionFactorizedLaneR1PayloadBytes ==
              3'614'363'648ULL);

enum class PrefillAttentionFactorizedLaneConverterErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidOption,
  kInvalidManifest,
  kInvalidBaseReceipt,
  kBaseAuthenticationFailed,
  kSourceBindingMismatch,
  kInvalidPolicy,
  kInvalidReceipt,
  kArithmeticOverflow,
  kQuantizationFailure,
  kDigestMismatch,
  kUnsafePath,
  kOpenFailed,
  kIoFailure,
  kPublicationConflict,
  kAllocationFailure,
};

struct PrefillAttentionFactorizedLaneConverterDiagnostic final {
  PrefillAttentionFactorizedLaneConverterErrorCode code =
      PrefillAttentionFactorizedLaneConverterErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillAttentionFactorizedLaneConverterErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct PrefillAttentionFactorizedLaneManifestResult final {
  std::optional<PrefillAttentionFactorizedLaneOverlayManifestBinding> value;
  PrefillAttentionFactorizedLaneConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

// Selects the 208 Attention entries from the authenticated 400-projection
// K256 inventory without reordering them. The exact base receipt bytes are
// bound by base_receipt_sha256 in addition to the manifest/policy/payload
// identities carried by the parsed receipt.
[[nodiscard]] PrefillAttentionFactorizedLaneManifestResult
build_prefill_attention_factorized_lane_r1_manifest(
    const PrefillSidecarManifest& base_k256_manifest,
    const PrefillA4PublicationReceipt& base_k256_receipt,
    std::string_view base_receipt_sha256);

[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
validate_prefill_attention_factorized_lane_r1_manifest(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest);

[[nodiscard]] std::string
prefill_attention_factorized_lane_r1_manifest_sha256(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest);

struct PrefillAttentionFactorizedLaneR1Policy final {
  PrefillAttentionFactorizedLaneOverlayPolicyBinding binding;
  std::string converter_abi;
  std::string mode;
  bool performance_upper_bound_only = true;
  bool quality_production_eligible = false;
};

struct PrefillAttentionFactorizedLaneR1PolicyResult final {
  std::optional<PrefillAttentionFactorizedLaneR1Policy> value;
  std::string canonical_document;
  PrefillAttentionFactorizedLaneConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillAttentionFactorizedLaneR1PolicyResult
build_prefill_attention_factorized_lane_r1_policy(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest,
    double weight_clip_ratio, double activation_clip_ratio);

[[nodiscard]] PrefillAttentionFactorizedLaneR1PolicyResult
parse_prefill_attention_factorized_lane_r1_policy(
    std::string_view document,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest);

struct PrefillAttentionFactorizedLaneR1Receipt final {
  PrefillAttentionFactorizedLaneOverlayReceiptBinding binding;
  std::string converter_abi;
  std::string mode;
  std::string residency_eligibility_scope;
  bool performance_upper_bound_only = true;
  bool quality_production_eligible = false;
};

struct PrefillAttentionFactorizedLaneR1ReceiptResult final {
  std::optional<PrefillAttentionFactorizedLaneR1Receipt> value;
  std::string canonical_document;
  PrefillAttentionFactorizedLaneConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillAttentionFactorizedLaneR1ReceiptResult
build_prefill_attention_factorized_lane_r1_receipt(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest,
    const PrefillAttentionFactorizedLaneR1Policy& policy,
    std::string_view payload_sha256);

[[nodiscard]] PrefillAttentionFactorizedLaneR1ReceiptResult
parse_prefill_attention_factorized_lane_r1_receipt(
    std::string_view document,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest,
    const PrefillAttentionFactorizedLaneR1Policy& policy);

// Thin type-safe Attention wrapper around the already-audited common N64/K256
// to R1 numerical transform used by MLP. No second quantization algorithm is
// maintained here.
[[nodiscard]] PrefillAttentionFactorizedLaneConverterDiagnostic
transform_prefill_attention_k256_to_factorized_r1_consumer_blocks(
    const std::uint8_t* base_packed_signed_w4,
    std::size_t base_packed_signed_w4_bytes,
    const std::uint8_t* base_bf16_scales_little_endian,
    std::size_t base_bf16_scale_bytes, std::size_t row_count,
    std::size_t input_size, double weight_clip_ratio,
    std::uint8_t* r1_packed_signed_w4,
    std::size_t r1_packed_signed_w4_bytes,
    std::uint8_t* r1_bf16_scales_little_endian,
    std::size_t r1_bf16_scale_bytes);

struct PrefillAttentionFactorizedLaneR1ConversionOptions final {
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

struct PrefillAttentionFactorizedLaneR1ConversionStats final {
  std::uint64_t base_bytes_read = 0U;
  std::uint64_t output_bytes_written = 0U;
  std::uint64_t peak_working_bytes = 0U;
  std::uint64_t projections_converted = 0U;
  std::uint64_t n64_blocks_converted = 0U;
};

struct PrefillAttentionFactorizedLaneR1ConversionResult final {
  std::optional<PrefillAttentionFactorizedLaneOverlayManifestBinding> manifest;
  std::optional<PrefillAttentionFactorizedLaneR1Policy> policy;
  std::optional<PrefillAttentionFactorizedLaneR1Receipt> receipt;
  PrefillAttentionFactorizedLaneR1ConversionStats stats;
  PrefillAttentionFactorizedLaneConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return manifest.has_value() && policy.has_value() &&
           receipt.has_value() && diagnostic.ok();
  }
};

// Authenticates and locks the exact K256 payload/policy/receipt set, converts
// only its 208 Attention projections, then publishes payload, policy, and
// receipt as one rollback-on-error no-replace set. No loader/runtime route is
// installed by this offline API.
[[nodiscard]] PrefillAttentionFactorizedLaneR1ConversionResult
convert_authenticated_k256_to_prefill_attention_factorized_lane_r1(
    const PrefillAttentionFactorizedLaneR1ConversionOptions& options);

[[nodiscard]] std::string_view to_string(
    PrefillAttentionFactorizedLaneConverterErrorCode code) noexcept;

}  // namespace q3x::runtime
