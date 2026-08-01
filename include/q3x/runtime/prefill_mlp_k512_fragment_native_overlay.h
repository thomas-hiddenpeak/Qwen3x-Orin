#pragma once

#include "q3x/runtime/prefill_mlp_k512_overlay.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>

namespace q3x::runtime {

// Lossless, offline-only permutation of the authenticated MLP K512 v1
// publication.  Quantized nibbles and BF16 scale bits are preserved exactly;
// only their byte positions change to match the SM87 fragment consumers.
inline constexpr std::uint32_t
    kPrefillMLPK512FragmentNativeVersionMajor = 1U;
inline constexpr std::uint32_t
    kPrefillMLPK512FragmentNativeVersionMinor = 0U;
inline constexpr std::size_t kPrefillMLPK512FragmentNativeLayerCount = 64U;
inline constexpr std::uint64_t
    kPrefillMLPK512FragmentNativeGateUpCodeBytes = 89'128'960ULL;
inline constexpr std::uint64_t
    kPrefillMLPK512FragmentNativeGateUpScaleBytes = 696'320ULL;
inline constexpr std::uint64_t
    kPrefillMLPK512FragmentNativeDownCodeBytes = 44'564'480ULL;
inline constexpr std::uint64_t
    kPrefillMLPK512FragmentNativeDownScaleBytes = 348'160ULL;
inline constexpr std::uint64_t
    kPrefillMLPK512FragmentNativeGateUpRecordBytes =
        kPrefillMLPK512FragmentNativeGateUpCodeBytes +
        kPrefillMLPK512FragmentNativeGateUpScaleBytes;
inline constexpr std::uint64_t
    kPrefillMLPK512FragmentNativeLayerBytes =
        kPrefillMLPK512FragmentNativeGateUpRecordBytes +
        kPrefillMLPK512FragmentNativeDownCodeBytes +
        kPrefillMLPK512FragmentNativeDownScaleBytes;
inline constexpr std::uint64_t
    kPrefillMLPK512FragmentNativePayloadBytes =
        kPrefillMLPK512FragmentNativeLayerCount *
        kPrefillMLPK512FragmentNativeLayerBytes;
inline constexpr std::string_view kPrefillMLPK512FragmentNativeLayout =
    "sm87_s4_gateup_n64_paired_down_n128_fragment_native_scale_k512_mlp_v2";

static_assert(kPrefillMLPK512FragmentNativeGateUpCodeBytes ==
              2U * kPrefillMLPK512OverlayProjectionWeightBytes);
static_assert(kPrefillMLPK512FragmentNativeGateUpScaleBytes ==
              2U * kPrefillMLPK512OverlayProjectionScaleBytes);
static_assert(kPrefillMLPK512FragmentNativeDownCodeBytes ==
              kPrefillMLPK512OverlayProjectionWeightBytes);
static_assert(kPrefillMLPK512FragmentNativeDownScaleBytes ==
              kPrefillMLPK512OverlayProjectionScaleBytes);
static_assert(kPrefillMLPK512FragmentNativeLayerBytes ==
              3U * kPrefillMLPK512OverlayProjectionBytes);
static_assert(kPrefillMLPK512FragmentNativePayloadBytes ==
              kPrefillMLPK512OverlayPayloadBytes);

// One layer is one independently attachable composite record:
//
//   paired Gate+Up codes | paired Gate+Up scales |
//   fragment-native Down codes | canonical K512 Down scales
//
// Gate and Up are never duplicated as projection records in v2.
struct PrefillMLPK512FragmentNativeLayerView final {
  std::uint32_t layer_index = 0U;
  std::uint64_t layer_offset = 0U;
  std::uint64_t gateup_code_offset = 0U;
  std::uint64_t gateup_code_bytes = 0U;
  std::uint64_t gateup_scale_offset = 0U;
  std::uint64_t gateup_scale_bytes = 0U;
  std::uint64_t down_code_offset = 0U;
  std::uint64_t down_code_bytes = 0U;
  std::uint64_t down_scale_offset = 0U;
  std::uint64_t down_scale_bytes = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr PrefillMLPK512FragmentNativeLayerView
prefill_mlp_k512_fragment_native_layer_view(
    const std::size_t layer_index) noexcept {
  if (layer_index >= kPrefillMLPK512FragmentNativeLayerCount) {
    return {};
  }
  const std::uint64_t layer_offset =
      layer_index * kPrefillMLPK512FragmentNativeLayerBytes;
  return {static_cast<std::uint32_t>(layer_index),
          layer_offset,
          layer_offset,
          kPrefillMLPK512FragmentNativeGateUpCodeBytes,
          layer_offset + kPrefillMLPK512FragmentNativeGateUpCodeBytes,
          kPrefillMLPK512FragmentNativeGateUpScaleBytes,
          layer_offset + kPrefillMLPK512FragmentNativeGateUpRecordBytes,
          kPrefillMLPK512FragmentNativeDownCodeBytes,
          layer_offset + kPrefillMLPK512FragmentNativeGateUpRecordBytes +
              kPrefillMLPK512FragmentNativeDownCodeBytes,
          kPrefillMLPK512FragmentNativeDownScaleBytes,
          true};
}

struct PrefillMLPK512FragmentNativeSourceBinding final {
  std::string physical_layout;
  std::string receipt_sha256;
  std::string manifest_sha256;
  std::string policy_sha256;
  std::uint64_t policy_bytes = 0U;
  std::string payload_sha256;
  std::uint64_t payload_bytes = 0U;
};

struct PrefillMLPK512FragmentNativeManifest final {
  std::uint32_t version_major =
      kPrefillMLPK512FragmentNativeVersionMajor;
  std::uint32_t version_minor =
      kPrefillMLPK512FragmentNativeVersionMinor;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  PrefillMLPK512BaseBinding required_base;
  PrefillMLPK512FragmentNativeSourceBinding source_v1;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t layer_count = 0U;
  std::string manifest_sha256;
};

struct PrefillMLPK512FragmentNativeManifestResult final {
  std::optional<PrefillMLPK512FragmentNativeManifest> value;
  PrefillMLPK512OverlayDiagnostic diagnostic;
  [[nodiscard]] explicit operator bool() const noexcept {
    return value.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPK512FragmentNativeManifestResult
build_prefill_mlp_k512_fragment_native_manifest(
    const PrefillMLPK512OverlayReceipt& source_v1_receipt,
    std::string_view source_v1_receipt_sha256);

[[nodiscard]] PrefillMLPK512OverlayDiagnostic
validate_prefill_mlp_k512_fragment_native_manifest(
    const PrefillMLPK512FragmentNativeManifest& manifest);

// Generic correctness/publication transforms.  Shapes must be complete N64
// (Gate+Up) or N128 (Down) blocks and K512 groups.  Source and destination
// ranges may not overlap.  Capacity arguments are exact, not lower bounds.
[[nodiscard]] PrefillMLPK512OverlayDiagnostic
permute_prefill_mlp_k512_gateup_fragment_native(
    const std::uint8_t* gate_codes, std::size_t gate_code_bytes,
    const std::uint8_t* gate_scales, std::size_t gate_scale_bytes,
    const std::uint8_t* up_codes, std::size_t up_code_bytes,
    const std::uint8_t* up_scales, std::size_t up_scale_bytes,
    std::size_t output_size, std::size_t input_size,
    std::uint8_t* paired_codes, std::size_t paired_code_bytes,
    std::uint8_t* paired_scales, std::size_t paired_scale_bytes);

[[nodiscard]] PrefillMLPK512OverlayDiagnostic
unpermute_prefill_mlp_k512_gateup_fragment_native(
    const std::uint8_t* paired_codes, std::size_t paired_code_bytes,
    const std::uint8_t* paired_scales, std::size_t paired_scale_bytes,
    std::size_t output_size, std::size_t input_size,
    std::uint8_t* gate_codes, std::size_t gate_code_bytes,
    std::uint8_t* gate_scales, std::size_t gate_scale_bytes,
    std::uint8_t* up_codes, std::size_t up_code_bytes,
    std::uint8_t* up_scales, std::size_t up_scale_bytes);

[[nodiscard]] PrefillMLPK512OverlayDiagnostic
permute_prefill_mlp_k512_down_fragment_native(
    const std::uint8_t* canonical_codes, std::size_t canonical_code_bytes,
    const std::uint8_t* canonical_scales, std::size_t canonical_scale_bytes,
    std::size_t output_size, std::size_t input_size,
    std::uint8_t* fragment_codes, std::size_t fragment_code_bytes,
    std::uint8_t* fragment_scales, std::size_t fragment_scale_bytes);

[[nodiscard]] PrefillMLPK512OverlayDiagnostic
unpermute_prefill_mlp_k512_down_fragment_native(
    const std::uint8_t* fragment_codes, std::size_t fragment_code_bytes,
    const std::uint8_t* fragment_scales, std::size_t fragment_scale_bytes,
    std::size_t output_size, std::size_t input_size,
    std::uint8_t* canonical_codes, std::size_t canonical_code_bytes,
    std::uint8_t* canonical_scales, std::size_t canonical_scale_bytes);

struct PrefillMLPK512FragmentNativeReceipt final {
  std::uint32_t version_major =
      kPrefillMLPK512FragmentNativeVersionMajor;
  std::uint32_t version_minor =
      kPrefillMLPK512FragmentNativeVersionMinor;
  bool production_residency_eligible = false;
  std::string physical_layout;
  std::string source_checkpoint_id;
  std::string source_config_sha256;
  std::string source_index_sha256;
  PrefillMLPK512BaseBinding required_base;
  PrefillMLPK512FragmentNativeSourceBinding source_v1;
  std::string manifest_sha256;
  std::string payload_sha256;
  std::uint64_t payload_bytes = 0U;
  std::uint64_t layer_count = 0U;
  // Digest of the exact parsed receipt document.  It is intentionally not a
  // JSON field (which would create a self-reference); downstream attach code
  // pins this value out of band.
  std::string receipt_sha256;
};

[[nodiscard]] std::optional<PrefillMLPK512FragmentNativeReceipt>
parse_prefill_mlp_k512_fragment_native_receipt(
    std::string_view json, PrefillMLPK512OverlayDiagnostic& diagnostic);

[[nodiscard]] PrefillMLPK512OverlayDiagnostic
write_prefill_mlp_k512_fragment_native_receipt_no_replace(
    const PrefillMLPK512FragmentNativeReceipt& receipt,
    const std::filesystem::path& output_path);

struct PrefillMLPK512FragmentNativeConversionOptions final {
  std::filesystem::path source_v1_payload_path;
  std::filesystem::path source_v1_receipt_path;
  std::filesystem::path source_v1_policy_path;
  std::string expected_source_v1_receipt_sha256;
  std::filesystem::path output_path;
  std::size_t outer_chunk_rows = 512U;
  std::uint64_t max_receipt_bytes = 1ULL * 1024ULL * 1024ULL;
  std::uint64_t max_policy_bytes = 4ULL * 1024ULL * 1024ULL;
  bool preallocate_output = true;
};

struct PrefillMLPK512FragmentNativeConversionStats final {
  std::uint64_t source_bytes_read = 0U;
  std::uint64_t output_bytes_written = 0U;
  std::uint64_t peak_working_bytes = 0U;
  std::uint64_t layers_permuted = 0U;
};

struct PrefillMLPK512FragmentNativeConversionResult final {
  std::optional<PrefillMLPK512FragmentNativeReceipt> receipt;
  PrefillMLPK512FragmentNativeConversionStats stats;
  PrefillMLPK512OverlayDiagnostic diagnostic;
  [[nodiscard]] explicit operator bool() const noexcept {
    return receipt.has_value() && diagnostic.ok();
  }
};

[[nodiscard]] PrefillMLPK512FragmentNativeConversionResult
convert_authenticated_prefill_mlp_k512_to_fragment_native(
    const PrefillMLPK512FragmentNativeConversionOptions& options);

}  // namespace q3x::runtime
