#pragma once

#include "q3x/runtime/prefill_quantized_contract.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace q3x::runtime {

inline constexpr std::uint32_t kPrefillA4CalibrationPolicyVersionMajor = 1U;
inline constexpr std::uint32_t kPrefillA4CalibrationPolicyVersionMinor = 0U;
inline constexpr std::uint32_t kPrefillA4PublicationVersionMajor = 1U;
inline constexpr std::uint32_t kPrefillA4PublicationVersionMinor = 0U;
inline constexpr std::uint32_t kPrefillA4WeightGroupSize = 64U;
inline constexpr double kPrefillA4MinimumClipRatio = 1.0 / 256.0;
inline constexpr std::string_view kPrefillA4PhysicalLayout =
    "sm87_s4_n64_k64_consumer_v1";

enum class PrefillA4ConversionMode : std::uint8_t {
  // The only mode eligible for a production residency admission. Every one
  // of the 400 projections must have an explicit, source-bound calibration
  // policy.
  kProductionCalibrated = 0,
  // A deliberately non-admissible nearest-even helper for small synthetic
  // correctness and I/O smoke tests. It must never produce a production
  // publication receipt.
  kExperimentalNearestEvenSmoke,
};

enum class PrefillA4Rounding : std::uint8_t {
  kNearestEvenV1 = 0,
};

enum class PrefillA4ConverterErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidOption,
  kInvalidManifest,
  kInvalidPolicy,
  kPolicyCoverageMismatch,
  kSourceBindingMismatch,
  kUnsupportedCalibration,
  kUnsafePath,
  kOpenFailed,
  kIoFailure,
  kSourceAuthenticationFailed,
  kSourceTensorMismatch,
  kArithmeticOverflow,
  kQuantizationFailure,
  kPublicationConflict,
  kPublicationRejected,
  kDigestMismatch,
  kAllocationFailure,
};

struct PrefillA4ConverterDiagnostic {
  PrefillA4ConverterErrorCode code = PrefillA4ConverterErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;
  int system_error = 0;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillA4ConverterErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Optional SmoothQuant/AWQ-style channel equalization contract. The factors
// file is little-endian FP32 [K]. The converter applies
//
//   equalized_weight[n,k] = source_weight[n,k] * factor[k]
//
// A consumer must apply the inverse factor to the corresponding activation.
// The v1 parser retains this explicit metadata, but the production converter
// and residency gate reject it until the runner can retain authenticated
// inverse factors. No production artifact is emitted in that state.
struct PrefillA4ChannelEqualization {
  std::string scheme;  // "input_channel_multiply_f32_v1"
  std::filesystem::path factors_path;
  std::string factors_sha256;
  std::uint64_t factor_count = 0U;
};

struct PrefillA4ProjectionCalibration {
  std::uint32_t ordinal = 0U;
  std::string source_module;
  std::string source_sha256;
  // Per-K64 symmetric weight threshold is
  // max(abs(weight_group))*weight_clip_ratio. Runtime activation quantization
  // independently uses max(abs(activation_group))*activation_clip_ratio.
  // Neither value may be inferred or defaulted on a production path. The v1
  // numerical domain is [1/256,1] so double-to-float/BF16 underflow cannot
  // silently publish an all-zero route.
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  std::uint32_t activation_scale_group_size = kPrefillA4WeightGroupSize;
  PrefillA4Rounding rounding = PrefillA4Rounding::kNearestEvenV1;
  std::optional<PrefillA4ChannelEqualization> channel_equalization;
};

struct PrefillA4CalibrationPolicy {
  std::uint32_t version_major = kPrefillA4CalibrationPolicyVersionMajor;
  std::uint32_t version_minor = kPrefillA4CalibrationPolicyVersionMinor;
  PrefillA4ConversionMode mode = PrefillA4ConversionMode::kProductionCalibrated;
  PrefillSidecarKind sidecar_kind = PrefillSidecarKind::kA4K64;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string manifest_sha256;
  std::vector<PrefillA4ProjectionCalibration> projections;
  // SHA-256 of the exact bounded JSON bytes supplied to the parser.
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
};

struct PrefillA4CalibrationPolicyResult {
  std::optional<PrefillA4CalibrationPolicy> value;
  PrefillA4ConverterDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Strict JSON schema parser. Unknown/duplicate/missing fields fail closed.
// A production policy must cover the ordered 400-projection A4-K64 manifest
// exactly and bind every projection source digest.
[[nodiscard]] PrefillA4CalibrationPolicyResult
parse_prefill_a4_calibration_policy(
    std::string_view json,
    const PrefillSidecarManifest& manifest);

[[nodiscard]] PrefillA4ConverterDiagnostic
validate_prefill_a4_calibration_policy(
    const PrefillA4CalibrationPolicy& policy,
    const PrefillSidecarManifest& manifest);

// Quantizes one or more N64 blocks from row-major FP32 input. Output is the
// fixed SM87 consumer layout, not canonical row-major:
//
//   weights [N/64][K/64][64][32] bytes
//   scales  [N/64][K/64][64] BF16 little-endian
//
// Within each pair, even K is the low nibble and odd K the high nibble. Signed
// codes are four-bit two's-complement in [-7,7]. N and K must be multiples of
// 64 and both output byte lengths are checked exactly.
//
// This bounded primitive is also the only experimental nearest-round smoke
// entry point. Production callers must pass kProductionCalibrated and a
// validated explicit calibration record.
[[nodiscard]] PrefillA4ConverterDiagnostic
quantize_prefill_a4_k64_consumer_blocks(
    const float* source_rows,
    std::size_t row_count,
    std::size_t input_size,
    const PrefillA4ProjectionCalibration& calibration,
    PrefillA4ConversionMode mode,
    std::uint8_t* packed_signed_w4,
    std::size_t packed_signed_w4_bytes,
    std::uint8_t* bf16_scales_little_endian,
    std::size_t bf16_scale_bytes);

struct PrefillA4SidecarConversionOptions {
  std::filesystem::path model_directory;
  std::filesystem::path calibration_policy_path;
  std::filesystem::path output_path;
  // Must be a multiple of the physical N64 consumer block.
  std::size_t row_chunk_size = 64U;
  // Bounded policy files prevent untrusted JSON allocation amplification.
  std::uint64_t max_policy_bytes = 16ULL * 1024ULL * 1024ULL;
  bool preallocate_output = true;
};

struct PrefillA4PublicationReceipt {
  std::uint32_t version_major = kPrefillA4PublicationVersionMajor;
  std::uint32_t version_minor = kPrefillA4PublicationVersionMinor;
  PrefillA4ConversionMode mode = PrefillA4ConversionMode::kProductionCalibrated;
  // Means authenticated/ABI-admissible only. It is not an accuracy or
  // capability verdict; those gates act on the bound policy SHA separately.
  bool production_residency_eligible = false;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
  std::string payload_sha256;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t projection_count = 0U;
};

struct PrefillA4SidecarConversionStats {
  std::uint64_t source_bytes_read = 0U;
  std::uint64_t output_bytes_written = 0U;
  std::uint64_t peak_working_bytes = 0U;
  std::uint64_t projections_converted = 0U;
  std::uint64_t rows_converted = 0U;
};

struct PrefillA4SidecarConversionResult {
  std::optional<PrefillA4PublicationReceipt> receipt;
  PrefillA4SidecarConversionStats stats;
  PrefillA4ConverterDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return receipt.has_value() && diagnostic.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Full production conversion. It builds and reuses the pinned 400-projection
// manifest, authenticates all three already-open checkpoint shards by exact
// size/SHA-256, streams bounded row chunks, verifies the complete temporary
// payload, then publishes the payload and `<output>.receipt.json` without
// replacing an existing artifact. No CUDA operation is performed.
[[nodiscard]] PrefillA4SidecarConversionResult
convert_pinned_qwen36_27b_prefill_a4_k64_sidecar(
    const PrefillA4SidecarConversionOptions& options);

// Strictly parses a converter publication receipt.
[[nodiscard]] std::optional<PrefillA4PublicationReceipt>
parse_prefill_a4_publication_receipt(
    std::string_view json,
    PrefillA4ConverterDiagnostic& diagnostic);

struct PrefillA4PublicationAuthenticationResult;

class PrefillA4AuthenticatedPublication {
 public:
  PrefillA4AuthenticatedPublication() noexcept = default;
  ~PrefillA4AuthenticatedPublication();
  PrefillA4AuthenticatedPublication(
      const PrefillA4AuthenticatedPublication&) = delete;
  PrefillA4AuthenticatedPublication& operator=(
      const PrefillA4AuthenticatedPublication&) = delete;
  PrefillA4AuthenticatedPublication(
      PrefillA4AuthenticatedPublication&& other) noexcept;
  PrefillA4AuthenticatedPublication& operator=(
      PrefillA4AuthenticatedPublication&& other) noexcept;

  // The runner must consume this already authenticated descriptor rather than
  // reopening payload_path. A shared advisory lock is held for this object's
  // lifetime. The descriptor remains owned by this object.
  [[nodiscard]] int payload_fd() const noexcept { return payload_fd_; }
  [[nodiscard]] const PrefillA4PublicationReceipt& receipt() const noexcept {
    return receipt_;
  }
  [[nodiscard]] const PrefillA4CalibrationPolicy& policy() const noexcept {
    return policy_;
  }
  [[nodiscard]] explicit operator bool() const noexcept {
    return payload_fd_ >= 0;
  }

  // Must be called after the runner finishes reading/copying the payload and
  // before residency becomes visible. It catches in-place mutation of the
  // held inode during consumption without reopening a path.
  [[nodiscard]] PrefillA4ConverterDiagnostic
  revalidate_unchanged_after_consumption() const;

 private:
  friend struct PrefillA4PublicationAuthenticationResult;
  friend PrefillA4PublicationAuthenticationResult
  authenticate_prefill_a4_publication_for_residency(
      const PrefillSidecarManifest&,
      const PrefillA4PublicationReceipt&,
      const std::filesystem::path&,
      const std::filesystem::path&);

  int payload_fd_ = -1;
  PrefillA4PublicationReceipt receipt_;
  PrefillA4CalibrationPolicy policy_;
  std::uint64_t device_id_ = 0U;
  std::uint64_t inode_ = 0U;
  std::uint64_t file_size_ = 0U;
  std::int64_t modification_seconds_ = 0;
  std::int64_t modification_nanoseconds_ = 0;
  std::int64_t change_seconds_ = 0;
  std::int64_t change_nanoseconds_ = 0;
};

struct PrefillA4PublicationAuthenticationResult {
  std::optional<PrefillA4AuthenticatedPublication> value;
  PrefillA4ConverterDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Production residency admission. There is no digest-bypass option.
// Experimental receipts are rejected, the complete policy and payload are
// authenticated, and the returned object retains the exact locked payload fd
// that the runner must consume. Channel equalization publications currently
// fail closed until the runner has an authenticated inverse-factor handle.
[[nodiscard]] PrefillA4PublicationAuthenticationResult
authenticate_prefill_a4_publication_for_residency(
    const PrefillSidecarManifest& manifest,
    const PrefillA4PublicationReceipt& receipt,
    const std::filesystem::path& payload_path,
    const std::filesystem::path& calibration_policy_path);

[[nodiscard]] std::string_view to_string(
    PrefillA4ConversionMode mode) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefillA4Rounding rounding) noexcept;
[[nodiscard]] std::string_view to_string(
    PrefillA4ConverterErrorCode code) noexcept;

}  // namespace q3x::runtime
