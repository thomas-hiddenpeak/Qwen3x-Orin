#pragma once

#include "q3x/runtime/prefill_mlp_factorized_lane_overlay.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace q3x::runtime {

// R4 is the calibrated accuracy candidate.  Unlike the R1 upper-bound
// converter, this primitive reads the original checkpoint NVFP4 tensors
// directly; a K256/R1 derivative is never an input to this ABI.
inline constexpr std::uint32_t
    kPrefillMLPFactorizedLaneR4LaneCount = 4U;
inline constexpr std::string_view
    kPrefillMLPFactorizedLaneR4FactorScheme =
        "calibrated_alpha_f32_v1";
inline constexpr double
    kPrefillMLPFactorizedLaneR4MinimumClipRatio = 1.0 / 256.0;

enum class PrefillMLPFactorizedLaneR4ConverterErrorCode : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kInvalidShape,
  kBufferSizeMismatch,
  kArithmeticOverflow,
  kInvalidSourceValue,
  kInvalidAlpha,
  kQuantizationFailure,
  kMetadataFailure,
  kAllocationFailure,
};

struct PrefillMLPFactorizedLaneR4ConverterDiagnostic final {
  PrefillMLPFactorizedLaneR4ConverterErrorCode code =
      PrefillMLPFactorizedLaneR4ConverterErrorCode::kNone;
  std::string context;
  std::string message;
  std::size_t index = 0U;

  [[nodiscard]] bool ok() const noexcept {
    return code == PrefillMLPFactorizedLaneR4ConverterErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

struct PrefillMLPFactorizedLaneR4MetadataResult final {
  PrefillMLPFactorizedLaneMetadataSerializationResult metadata;
  PrefillMLPFactorizedLaneR4ConverterDiagnostic diagnostic;

  [[nodiscard]] explicit operator bool() const noexcept {
    return metadata.valid() && diagnostic.ok();
  }
};

// Derives the projection-local authenticated activation metadata from the
// exact alpha used to transform the weights:
//
//   transformed_weight[n,k] = original_weight[n,k] * alpha[k]
//   transformed_input[m,k]  = original_input[m,k] / alpha[k]
//
// The returned v4 metadata therefore contains FP32 inverse_alpha[k] =
// 1.0F / alpha[k], its exact little-endian digest, lane_count=4, and K.
// Alpha and its FP32 reciprocal must both be finite and strictly positive.
[[nodiscard]] PrefillMLPFactorizedLaneR4MetadataResult
build_prefill_mlp_factorized_lane_r4_metadata(
    const float* alpha, std::size_t alpha_count);

// Converts one or more complete N64 blocks directly from the original
// ModelOpt NVFP4 checkpoint representation:
//
//   source packed  row-major [N][K/2]
//   source scales  E4M3FN   [N][K/16]
//   source value   E2M1(code) * E4M3FN(scale) * weight_scale_2
//
// Alpha is projection-local FP32 [K].  K is split into four contiguous,
// equal K64-aligned lanes.  Each (N row, lane) receives one independently
// rounded positive BF16 scale after applying the explicit clip ratio.  Codes
// are signed A4 [-7,7], nearest-even, and retain the consumer transport order:
//
//   output packed  [N/64][K/64][64][32]
//   output scales  [N/64][4][64] BF16 little-endian
//
// All byte counts are exact, and the two source spans, alpha span, and two
// output spans must be pairwise disjoint.  This is a bounded numerical
// primitive only; it does not authenticate a checkpoint or grant
// runtime/production admission.
[[nodiscard]] PrefillMLPFactorizedLaneR4ConverterDiagnostic
transform_prefill_mlp_nvfp4_to_factorized_r4_consumer_blocks(
    const std::uint8_t* source_packed_nvfp4,
    std::size_t source_packed_nvfp4_bytes,
    const std::uint8_t* source_e4m3_scales,
    std::size_t source_e4m3_scale_bytes,
    float source_weight_scale_2, std::size_t row_count,
    std::size_t input_size, const float* alpha,
    std::size_t alpha_count, double weight_clip_ratio,
    std::uint8_t* r4_packed_signed_w4,
    std::size_t r4_packed_signed_w4_bytes,
    std::uint8_t* r4_bf16_scales_little_endian,
    std::size_t r4_bf16_scale_bytes);

[[nodiscard]] std::string_view to_string(
    PrefillMLPFactorizedLaneR4ConverterErrorCode code) noexcept;

static_assert(kPrefillMLPFactorizedLaneR4LaneCount == 4U);
static_assert(
    prefill_a4_factorized_lane_projection_layout_plan(17'408U, 5'120U,
                                                       4U)
        .valid());
static_assert(
    prefill_a4_factorized_lane_projection_layout_plan(5'120U, 17'408U,
                                                       4U)
        .valid());

}  // namespace q3x::runtime
