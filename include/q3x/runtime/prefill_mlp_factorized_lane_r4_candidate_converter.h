#pragma once

#include "q3x/runtime/prefill_mlp_factorized_lane_r4_publication.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace q3x::runtime {

// This converter exists only to answer the first real-weight performance
// question for the direct-checkpoint R4 layout.  Alpha is exactly FP32 one[K]
// and is reproducibly bound below; it is not advertised as calibrated.
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha5120 =
        "builtin/q3x/identity-alpha-f32-v1/k5120";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha17408 =
        "builtin/q3x/identity-alpha-f32-v1/k17408";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha5120Sha256 =
        "42010c1c68b632e2ab15c82bca6edef2cac2026c889dd0202d609602b756f568";
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha17408Sha256 =
        "15cd4df15b3bcb53816bb119e9d52efa3c0bbee237fa17c5a7c351dc9bfdcbcd";

enum class PrefillMLPFactorizedLaneR4CandidateConverterErrorCode
    : std::uint8_t {
  kNone = 0,
  kInvalidOption,
  kInvalidManifest,
  kSourceTensorMismatch,
  kUnsafePath,
  kOpenFailed,
  kIoFailure,
  kArithmeticOverflow,
  kQuantizationFailure,
  kMetadataFailure,
  kPublicationFailure,
  kDigestMismatch,
  kPublicationConflict,
  kAllocationFailure,
};

struct PrefillMLPFactorizedLaneR4CandidateConverterDiagnostic final {
  PrefillMLPFactorizedLaneR4CandidateConverterErrorCode code =
      PrefillMLPFactorizedLaneR4CandidateConverterErrorCode::kNone;
  std::string context;
  std::string message;
  std::string expected;
  std::string actual;
  int system_error = 0;

  [[nodiscard]] bool ok() const noexcept {
    return code ==
           PrefillMLPFactorizedLaneR4CandidateConverterErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct PrefillMLPFactorizedLaneR4IdentityCandidateConversionOptions final {
  std::filesystem::path model_directory;
  std::filesystem::path output_path;
  // Both clips are deliberately invalid by default.  Callers and the CLI
  // must make the two independent direction-gate choices explicitly.
  double weight_clip_ratio = 0.0;
  double activation_clip_ratio = 0.0;
  std::size_t row_chunk_size = 256U;
  bool preallocate_output = true;
};

struct PrefillMLPFactorizedLaneR4IdentityCandidateConversionStats final {
  std::uint64_t source_shard_bytes_hashed = 0U;
  std::uint64_t source_shards_authenticated = 0U;
  std::uint64_t source_bytes_read = 0U;
  std::uint64_t output_bytes_written = 0U;
  std::uint64_t peak_working_bytes = 0U;
  std::uint64_t projections_converted = 0U;
  std::uint64_t n64_blocks_converted = 0U;
};

struct PrefillMLPFactorizedLaneR4IdentityCandidateConversionResult final {
  std::optional<PrefillMLPFactorizedLaneR4Manifest> manifest;
  std::optional<PrefillMLPFactorizedLaneR4Policy> policy;
  std::optional<PrefillMLPFactorizedLaneR4Receipt> receipt;
  PrefillMLPFactorizedLaneR4IdentityCandidateConversionStats stats;
  PrefillMLPFactorizedLaneR4CandidateConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return manifest.has_value() && policy.has_value() &&
           receipt.has_value() && diagnostic.ok();
  }
};

// Streams original pinned ModelOpt NVFP4 Gate/Up/Down tensors directly into
// the fixed 8,583,954,432-byte R4 payload.  The source is never a K256 or R1
// derivative.  OUTPUT.manifest.json, OUTPUT.policy.json and
// OUTPUT.receipt.json are canonical strict-parser documents.  The four-file
// set is published with no-replace links and rollback on publication error.
//
// This bounded direction-gate publisher verifies every complete pinned source
// shard SHA-256 while its stable snapshot and shared lock are held, then the
// complete logical manifest, every locator and exact tensor read, R4 metadata
// reparsing, and the complete output SHA-256.  The resulting receipt remains
// performance_candidate_only and grants neither production residency nor
// model-quality eligibility.
[[nodiscard]]
PrefillMLPFactorizedLaneR4IdentityCandidateConversionResult
convert_pinned_qwen36_27b_prefill_mlp_factorized_lane_r4_identity_candidate(
    const PrefillMLPFactorizedLaneR4IdentityCandidateConversionOptions&
        options);

[[nodiscard]] std::string_view to_string(
    PrefillMLPFactorizedLaneR4CandidateConverterErrorCode code) noexcept;

}  // namespace q3x::runtime
