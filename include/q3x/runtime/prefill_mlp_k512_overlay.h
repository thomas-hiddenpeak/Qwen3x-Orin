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

inline constexpr std::uint32_t kPrefillMLPK512OverlayVersionMajor = 1U;
inline constexpr std::uint32_t kPrefillMLPK512OverlayVersionMinor = 0U;
inline constexpr std::size_t kPrefillMLPK512OverlayProjectionCount =
    192U;
inline constexpr std::uint64_t kPrefillMLPK512OverlayGateUpOutputSize =
    17'408U;
inline constexpr std::uint64_t kPrefillMLPK512OverlayGateUpInputSize =
    5'120U;
inline constexpr std::uint64_t kPrefillMLPK512OverlayDownOutputSize =
    5'120U;
inline constexpr std::uint64_t kPrefillMLPK512OverlayDownInputSize =
    17'408U;
inline constexpr std::uint32_t kPrefillMLPK512OverlayPackedK = 64U;
inline constexpr std::uint32_t kPrefillMLPK512OverlayScaleK = 512U;
inline constexpr std::uint64_t
    kPrefillMLPK512OverlayProjectionWeightBytes = 44'564'480ULL;
inline constexpr std::uint64_t
    kPrefillMLPK512OverlayProjectionScaleBytes = 348'160ULL;
inline constexpr std::uint64_t
    kPrefillMLPK512OverlayProjectionBytes = 44'912'640ULL;
inline constexpr std::uint64_t kPrefillMLPK512OverlayPayloadBytes =
    8'623'226'880ULL;
inline constexpr std::string_view kPrefillMLPK512OverlayLayout =
    "sm87_s4_n64_packed_k64_scale_k512_mlp_v1";

static_assert(kPrefillMLPK512OverlayGateUpOutputSize *
                      kPrefillMLPK512OverlayGateUpInputSize ==
                  kPrefillMLPK512OverlayDownOutputSize *
                      kPrefillMLPK512OverlayDownInputSize,
              "all fixed MLP projections must retain equal weight bytes");
static_assert(kPrefillMLPK512OverlayProjectionWeightBytes ==
              kPrefillMLPK512OverlayGateUpOutputSize *
                  kPrefillMLPK512OverlayGateUpInputSize / 2U);
static_assert(kPrefillMLPK512OverlayProjectionScaleBytes ==
              kPrefillMLPK512OverlayGateUpOutputSize *
                  kPrefillMLPK512OverlayGateUpInputSize /
                  kPrefillMLPK512OverlayScaleK * 2U);
static_assert(kPrefillMLPK512OverlayPayloadBytes ==
              kPrefillMLPK512OverlayProjectionCount *
                  kPrefillMLPK512OverlayProjectionBytes);

enum class PrefillMLPK512OverlayErrorCode : std::uint8_t {
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

struct PrefillMLPK512OverlayDiagnostic {
  PrefillMLPK512OverlayErrorCode code =
      PrefillMLPK512OverlayErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;
  int system_error = 0;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillMLPK512OverlayErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// The overlay is deliberately tied to the exact authenticated K128 or K256
// publication used for every non-MLP projection. This prevents a benchmark
// from silently pairing the K512 MLP plane with a different calibration
// bundle.
struct PrefillMLPK512BaseBinding {
  std::string physical_layout;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::string payload_sha256;
};

[[nodiscard]] constexpr bool prefill_mlp_k512_base_layout_matches_contract(
    const std::string_view physical_layout,
    const PrefillSidecarKind sidecar_kind,
    const std::uint32_t packed_k_group_size,
    const std::uint32_t scale_group_size) noexcept {
  return packed_k_group_size == kPrefillMLPK512OverlayPackedK &&
         ((physical_layout == kPrefillA4K128PhysicalLayout &&
           sidecar_kind == PrefillSidecarKind::kA4K128 &&
           scale_group_size == 128U) ||
          (physical_layout == kPrefillA4K256PhysicalLayout &&
           sidecar_kind == PrefillSidecarKind::kA4K256 &&
           scale_group_size == 256U));
}

struct PrefillMLPK512OverlayEntry {
  std::uint32_t ordinal = 0U;
  std::uint32_t layer_index = 0U;
  std::string family;  // "gate", "up", or "down"
  std::string source_module;
  std::string source_sha256;
  std::uint64_t output_size = 0U;
  std::uint64_t input_size = 0U;
  std::uint64_t sidecar_offset = 0U;
  std::uint64_t weight_bytes = 0U;
  std::uint64_t scale_bytes = 0U;
};

struct PrefillMLPK512OverlayManifest {
  std::uint32_t version_major = kPrefillMLPK512OverlayVersionMajor;
  std::uint32_t version_minor = kPrefillMLPK512OverlayVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  PrefillMLPK512BaseBinding required_base;
  std::vector<PrefillMLPK512OverlayEntry> projections;
  std::uint64_t payload_bytes = 0U;
  std::string manifest_sha256;
};

struct PrefillMLPK512OverlayManifestResult {
  std::optional<PrefillMLPK512OverlayManifest> value;
  PrefillMLPK512OverlayDiagnostic diagnostic;
  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPK512OverlayManifestResult
build_qwen36_27b_prefill_mlp_k512_overlay_manifest(
    const model::weights::WeightManifest& source_manifest,
    const std::vector<ShardIdentity>& authenticated_shards,
    const PrefillMLPK512BaseBinding& required_base);

[[nodiscard]] PrefillMLPK512OverlayDiagnostic
validate_prefill_mlp_k512_overlay_manifest(
    const PrefillMLPK512OverlayManifest& manifest);

struct PrefillMLPK512OverlayCalibration {
  std::uint32_t ordinal = 0U;
  std::string source_module;
  std::string source_sha256;
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  std::uint32_t activation_scale_group_size = 0U;
};

struct PrefillMLPK512OverlayPolicy {
  std::uint32_t version_major = kPrefillMLPK512OverlayVersionMajor;
  std::uint32_t version_minor = kPrefillMLPK512OverlayVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string manifest_sha256;
  PrefillMLPK512BaseBinding required_base;
  std::vector<PrefillMLPK512OverlayCalibration> projections;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
};

struct PrefillMLPK512OverlayPolicyResult {
  std::optional<PrefillMLPK512OverlayPolicy> value;
  PrefillMLPK512OverlayDiagnostic diagnostic;
  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPK512OverlayPolicyResult
parse_prefill_mlp_k512_overlay_policy(
    std::string_view json,
    const PrefillMLPK512OverlayManifest& manifest);

struct PrefillMLPK512OverlayPolicyTemplateOptions {
  std::filesystem::path model_directory;
  std::filesystem::path base_k128_receipt_path;
  std::filesystem::path output_path;
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
};

[[nodiscard]] PrefillMLPK512OverlayPolicyResult
write_prefill_mlp_k512_overlay_policy_template(
    const PrefillMLPK512OverlayManifest& manifest,
    const std::filesystem::path& output_path, double weight_clip_ratio,
    double activation_clip_ratio);

[[nodiscard]] PrefillMLPK512OverlayPolicyResult
write_qwen36_27b_prefill_mlp_k512_overlay_policy_template(
    const PrefillMLPK512OverlayPolicyTemplateOptions& options);

// Host quantizer for one or more complete N64 blocks.  Input is the decoded
// original checkpoint matrix, never base-sidecar codes/scales.  Packed codes use
// [N/64][K/64][64][32], while BF16 scales use [N/64][K/512][64].
[[nodiscard]] PrefillMLPK512OverlayDiagnostic
quantize_prefill_mlp_k512_consumer_blocks(
    const float* source_rows, std::size_t row_count, std::size_t input_size,
    double weight_clip_ratio, std::uint8_t* packed_signed_w4,
    std::size_t packed_signed_w4_bytes,
    std::uint8_t* bf16_scales_little_endian,
    std::size_t bf16_scale_bytes);

struct PrefillMLPK512OverlayReceipt {
  std::uint32_t version_major = kPrefillMLPK512OverlayVersionMajor;
  std::uint32_t version_minor = kPrefillMLPK512OverlayVersionMinor;
  bool production_residency_eligible = false;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
  PrefillMLPK512BaseBinding required_base;
  std::string payload_sha256;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t projection_count = 0U;
};

[[nodiscard]] std::optional<PrefillMLPK512OverlayReceipt>
parse_prefill_mlp_k512_overlay_receipt(
    std::string_view json,
    PrefillMLPK512OverlayDiagnostic& diagnostic);

struct PrefillMLPK512OverlayConversionOptions {
  std::filesystem::path model_directory;
  std::filesystem::path calibration_policy_path;
  std::filesystem::path output_path;
  std::size_t row_chunk_size = 64U;
  std::uint64_t max_policy_bytes = 4ULL * 1024ULL * 1024ULL;
  bool preallocate_output = true;
};

struct PrefillMLPK512OverlayConversionStats {
  std::uint64_t source_bytes_read = 0U;
  std::uint64_t output_bytes_written = 0U;
  std::uint64_t peak_working_bytes = 0U;
  std::uint64_t projections_converted = 0U;
  std::uint64_t rows_converted = 0U;
};

struct PrefillMLPK512OverlayConversionResult {
  std::optional<PrefillMLPK512OverlayReceipt> receipt;
  PrefillMLPK512OverlayConversionStats stats;
  PrefillMLPK512OverlayDiagnostic diagnostic;
  [[nodiscard]] explicit operator bool() const noexcept {
    return receipt.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPK512OverlayConversionResult
convert_pinned_qwen36_27b_prefill_mlp_k512_overlay(
    const PrefillMLPK512OverlayConversionOptions& options);

[[nodiscard]] std::string_view to_string(
    PrefillMLPK512OverlayErrorCode code) noexcept;

}  // namespace q3x::runtime
