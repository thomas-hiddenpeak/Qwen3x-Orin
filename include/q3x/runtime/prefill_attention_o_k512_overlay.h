#pragma once

#include "q3x/model/weight_manifest.h"
#include "q3x/runtime/prefill_a4_sidecar_converter.h"
#include "q3x/runtime/resident_weights.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

inline constexpr std::uint32_t kPrefillAttentionOK512OverlayVersionMajor = 1U;
inline constexpr std::uint32_t kPrefillAttentionOK512OverlayVersionMinor = 0U;
inline constexpr std::size_t kPrefillAttentionOK512OverlayProjectionCount =
    64U;
inline constexpr std::uint64_t kPrefillAttentionOK512OverlayOutputSize =
    5'120U;
inline constexpr std::uint64_t kPrefillAttentionOK512OverlayInputSize =
    6'144U;
inline constexpr std::uint32_t kPrefillAttentionOK512OverlayPackedK = 64U;
inline constexpr std::uint32_t kPrefillAttentionOK512OverlayScaleK = 512U;
inline constexpr std::uint64_t
    kPrefillAttentionOK512OverlayProjectionWeightBytes = 15'728'640ULL;
inline constexpr std::uint64_t
    kPrefillAttentionOK512OverlayProjectionScaleBytes = 122'880ULL;
inline constexpr std::uint64_t
    kPrefillAttentionOK512OverlayProjectionBytes = 15'851'520ULL;
inline constexpr std::uint64_t kPrefillAttentionOK512OverlayPayloadBytes =
    1'014'497'280ULL;
inline constexpr std::string_view kPrefillAttentionOK512OverlayLayout =
    "sm87_s4_n64_packed_k64_scale_k512_attention_o_v1";

enum class PrefillAttentionOK512OverlayErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidOption,
  kInvalidManifest,
  kInvalidPolicy,
  kInvalidReceipt,
  kSourceAuthenticationFailed,
  kSourceTensorMismatch,
  kIoFailure,
  kDigestMismatch,
  kArithmeticOverflow,
  kQuantizationFailure,
  kPublicationConflict,
  kAllocationFailure,
};

struct PrefillAttentionOK512OverlayDiagnostic {
  PrefillAttentionOK512OverlayErrorCode code =
      PrefillAttentionOK512OverlayErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;
  int system_error = 0;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillAttentionOK512OverlayErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// The overlay is deliberately tied to the exact authenticated K128
// publication used for every non-O projection.  This prevents a benchmark
// from silently pairing the K512 O plane with a different calibration bundle.
struct PrefillAttentionOK512BaseBinding {
  std::string physical_layout;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::string payload_sha256;
};

struct PrefillAttentionOK512OverlayEntry {
  std::uint32_t ordinal = 0U;
  std::uint32_t layer_index = 0U;
  std::string family;  // "linear_o" or "full_o"
  std::string source_module;
  std::string source_sha256;
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
  std::uint64_t sidecar_offset = 0U;
  std::uint64_t weight_bytes = 0U;
  std::uint64_t scale_bytes = 0U;
};

struct PrefillAttentionOK512OverlayManifest {
  std::uint32_t version_major = kPrefillAttentionOK512OverlayVersionMajor;
  std::uint32_t version_minor = kPrefillAttentionOK512OverlayVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  PrefillAttentionOK512BaseBinding required_base;
  std::vector<PrefillAttentionOK512OverlayEntry> projections;
  std::uint64_t payload_bytes = 0U;
  std::string manifest_sha256;
};

struct PrefillAttentionOK512OverlayManifestResult {
  std::optional<PrefillAttentionOK512OverlayManifest> value;
  PrefillAttentionOK512OverlayDiagnostic diagnostic;
  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillAttentionOK512OverlayManifestResult
build_qwen36_27b_prefill_attention_o_k512_overlay_manifest(
    const model::weights::WeightManifest& source_manifest,
    const std::vector<ShardIdentity>& authenticated_shards,
    const PrefillAttentionOK512BaseBinding& required_base);

[[nodiscard]] PrefillAttentionOK512OverlayDiagnostic
validate_prefill_attention_o_k512_overlay_manifest(
    const PrefillAttentionOK512OverlayManifest& manifest);

struct PrefillAttentionOK512OverlayCalibration {
  std::uint32_t ordinal = 0U;
  std::string source_module;
  std::string source_sha256;
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  std::uint32_t activation_scale_group_size = 0U;
};

struct PrefillAttentionOK512OverlayPolicy {
  std::uint32_t version_major = kPrefillAttentionOK512OverlayVersionMajor;
  std::uint32_t version_minor = kPrefillAttentionOK512OverlayVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string manifest_sha256;
  PrefillAttentionOK512BaseBinding required_base;
  std::vector<PrefillAttentionOK512OverlayCalibration> projections;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
};

struct PrefillAttentionOK512OverlayPolicyResult {
  std::optional<PrefillAttentionOK512OverlayPolicy> value;
  PrefillAttentionOK512OverlayDiagnostic diagnostic;
  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillAttentionOK512OverlayPolicyResult
parse_prefill_attention_o_k512_overlay_policy(
    std::string_view json,
    const PrefillAttentionOK512OverlayManifest& manifest);

struct PrefillAttentionOK512OverlayPolicyTemplateOptions {
  std::filesystem::path model_directory;
  std::filesystem::path base_k128_receipt_path;
  std::filesystem::path output_path;
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
};

[[nodiscard]] PrefillAttentionOK512OverlayPolicyResult
write_prefill_attention_o_k512_overlay_policy_template(
    const PrefillAttentionOK512OverlayManifest& manifest,
    const std::filesystem::path& output_path, double weight_clip_ratio,
    double activation_clip_ratio);

[[nodiscard]] PrefillAttentionOK512OverlayPolicyResult
write_qwen36_27b_prefill_attention_o_k512_overlay_policy_template(
    const PrefillAttentionOK512OverlayPolicyTemplateOptions& options);

// Host quantizer for one or more complete N64 blocks.  Input is the decoded
// original checkpoint matrix, never K128 codes/scales.  Packed codes use
// [N/64][K/64][64][32], while BF16 scales use [N/64][K/512][64].
[[nodiscard]] PrefillAttentionOK512OverlayDiagnostic
quantize_prefill_attention_o_k512_consumer_blocks(
    const float* source_rows, std::size_t row_count, std::size_t input_size,
    double weight_clip_ratio, std::uint8_t* packed_signed_w4,
    std::size_t packed_signed_w4_bytes,
    std::uint8_t* bf16_scales_little_endian,
    std::size_t bf16_scale_bytes);

struct PrefillAttentionOK512OverlayReceipt {
  std::uint32_t version_major = kPrefillAttentionOK512OverlayVersionMajor;
  std::uint32_t version_minor = kPrefillAttentionOK512OverlayVersionMinor;
  bool production_residency_eligible = false;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
  PrefillAttentionOK512BaseBinding required_base;
  std::string payload_sha256;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t projection_count = 0U;
};

[[nodiscard]] std::optional<PrefillAttentionOK512OverlayReceipt>
parse_prefill_attention_o_k512_overlay_receipt(
    std::string_view json,
    PrefillAttentionOK512OverlayDiagnostic& diagnostic);

struct PrefillAttentionOK512OverlayConversionOptions {
  std::filesystem::path model_directory;
  std::filesystem::path calibration_policy_path;
  std::filesystem::path output_path;
  std::size_t row_chunk_size = 64U;
  std::uint64_t max_policy_bytes = 4ULL * 1024ULL * 1024ULL;
  bool preallocate_output = true;
};

struct PrefillAttentionOK512OverlayConversionStats {
  std::uint64_t source_bytes_read = 0U;
  std::uint64_t output_bytes_written = 0U;
  std::uint64_t peak_working_bytes = 0U;
  std::uint64_t projections_converted = 0U;
  std::uint64_t rows_converted = 0U;
};

struct PrefillAttentionOK512OverlayConversionResult {
  std::optional<PrefillAttentionOK512OverlayReceipt> receipt;
  PrefillAttentionOK512OverlayConversionStats stats;
  PrefillAttentionOK512OverlayDiagnostic diagnostic;
  [[nodiscard]] explicit operator bool() const noexcept {
    return receipt.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillAttentionOK512OverlayConversionResult
convert_pinned_qwen36_27b_prefill_attention_o_k512_overlay(
    const PrefillAttentionOK512OverlayConversionOptions& options);

[[nodiscard]] std::string_view to_string(
    PrefillAttentionOK512OverlayErrorCode code) noexcept;

}  // namespace q3x::runtime
