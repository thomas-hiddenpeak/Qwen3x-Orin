#pragma once

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Isolated activation producer for the experimental factorized-lane A4W4
// contract.  This surface has no runner selector and does not authenticate a
// sidecar by itself.  `authenticated_inverse_alpha_fp32` must point at the
// exact resident FP32 payload whose digest and containing projection receipt
// were verified before this launcher is reached.
//
// For X[M,K] and L factorization lanes, the numerical contract is
//
//   X'[m,k] = bf16_input[m,k] * inverse_alpha[k]
//   scale[m,l] = bf16(max(abs(X'[m,lane(l)])) * clip_ratio / 7)
//   code[m,k] = round_to_nearest_even(clamp(X'[m,k]) / scale[m,lane(k)])
//
// with signed codes clamped to [-7,7].  A zero lane publishes BF16 one and
// zero codes.  Physical codes retain the established consumer order
// [M/64][K/64][64][32].  Scales use the corresponding lane order
// [M/64][lane][64].  Every padded launch row is explicitly zero/one and never
// reads either BF16 input plane.
inline constexpr std::size_t kSm87A4W4FactorizedLaneQuantizeGateInput =
    5'120U;
inline constexpr std::size_t
    kSm87A4W4FactorizedLaneQuantizeAttentionOInput = 6'144U;
inline constexpr std::size_t kSm87A4W4FactorizedLaneQuantizeDownInput =
    17'408U;
inline constexpr std::size_t
    kSm87A4W4FactorizedLaneQuantizeDownPrimaryInput = 12'288U;
inline constexpr std::size_t
    kSm87A4W4FactorizedLaneQuantizeDownSecondaryInput = 5'120U;
inline constexpr std::size_t kSm87A4W4FactorizedLaneQuantizeThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4FactorizedLaneQuantizeMinimumActiveBlocksPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4FactorizedLaneQuantizeMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4FactorizedLaneQuantizeMaximumLaunchTokens = 4'096U;

[[nodiscard]] constexpr bool
sm87_a4w4_factorized_lane_quantize_launch_tokens_supported(
    const std::size_t launch_token_count) noexcept {
  return launch_token_count != 0U &&
         launch_token_count % kSm87A4W4ConsumerOuterBlock == 0U &&
         launch_token_count <=
             kSm87A4W4FactorizedLaneQuantizeMaximumLaunchTokens;
}

[[nodiscard]] constexpr bool
sm87_a4w4_factorized_lane_quantize_input_supported(
    const std::size_t input_size) noexcept {
  return input_size == kSm87A4W4FactorizedLaneQuantizeGateInput ||
         input_size ==
             kSm87A4W4FactorizedLaneQuantizeAttentionOInput ||
         input_size == kSm87A4W4FactorizedLaneQuantizeDownInput;
}

struct Sm87A4W4FactorizedLaneQuantizePlan final {
  std::size_t logical_token_count{};
  std::size_t launch_token_count{};
  std::size_t input_size{};
  std::size_t lane_count{};
  std::size_t lane_input_size{};
  std::size_t physical_k64_groups{};
  std::size_t launch_ctas{};
  std::size_t packed_capacity_bytes{};
  std::size_t scale_capacity_elements{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return launch_ctas != 0U;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
};

[[nodiscard]] constexpr Sm87A4W4FactorizedLaneQuantizePlan
sm87_a4w4_factorized_lane_quantize_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t input_size,
    const std::size_t lane_count) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count > launch_token_count ||
      !sm87_a4w4_factorized_lane_quantize_launch_tokens_supported(
          launch_token_count) ||
      !sm87_a4w4_factorized_lane_quantize_input_supported(input_size) ||
      (lane_count != 1U && lane_count != 4U) ||
      input_size % lane_count != 0U ||
      (input_size / lane_count) % kSm87A4W4ConsumerKBlock != 0U) {
    return {};
  }
  const std::size_t packed_capacity =
      sm87_a4w4_consumer_packed_capacity_bytes(launch_token_count,
                                               input_size);
  const std::size_t scale_capacity = launch_token_count * lane_count;
  const std::size_t launch_ctas = launch_token_count * lane_count;
  if (packed_capacity == 0U || scale_capacity == 0U ||
      launch_ctas == 0U) {
    return {};
  }
  return {logical_token_count,
          launch_token_count,
          input_size,
          lane_count,
          input_size / lane_count,
          input_size / kSm87A4W4ConsumerKBlock,
          launch_ctas,
          packed_capacity,
          scale_capacity};
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_factorized_lane_scale_capacity_elements(
    const std::size_t launch_token_count,
    const std::size_t lane_count) noexcept {
  return sm87_a4w4_factorized_lane_quantize_launch_tokens_supported(
             launch_token_count) &&
                 (lane_count == 1U || lane_count == 4U)
             ? launch_token_count * lane_count
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_factorized_lane_scale_offset(
    const std::size_t row, const std::size_t lane,
    const std::size_t lane_count) noexcept {
  return ((row / kSm87A4W4ConsumerOuterBlock) * lane_count + lane) *
             kSm87A4W4ConsumerOuterBlock +
         row % kSm87A4W4ConsumerOuterBlock;
}

struct Sm87A4W4FactorizedLaneQuantizeResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
  int multiprocessor_count{};
};

[[nodiscard]] int query_sm87_a4w4_factorized_lane_quantize_resources_cuda(
    Sm87A4W4FactorizedLaneQuantizeResources* resources) noexcept;

// Contiguous BF16 source.  Capacity is expressed in elements and must cover
// exactly the addressed strided span `(logical_M-1)*stride + K`; padding rows
// require no input allocation.
[[nodiscard]] int launch_sm87_a4w4_factorized_lane_quantize_bf16_cuda(
    const std::uint16_t* input_bf16,
    std::size_t input_row_stride_elements,
    std::size_t input_capacity_elements,
    const float* authenticated_inverse_alpha_fp32,
    std::size_t inverse_alpha_capacity_elements,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t input_size, std::size_t lane_count, float clip_ratio,
    std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Pinned Down-product source split as [M,12288] + [M,5120].  Lane boundaries
// are independent of the plane boundary (R4 lane two crosses it), so the
// result is bit-identical to row-wise concatenation followed by the
// contiguous entry above.
[[nodiscard]] int launch_sm87_a4w4_factorized_lane_quantize_bf16_split_cuda(
    const std::uint16_t* primary_bf16,
    std::size_t primary_row_stride_elements,
    std::size_t primary_capacity_elements, std::size_t primary_size,
    const std::uint16_t* secondary_bf16,
    std::size_t secondary_row_stride_elements,
    std::size_t secondary_capacity_elements, std::size_t secondary_size,
    const float* authenticated_inverse_alpha_fp32,
    std::size_t inverse_alpha_capacity_elements,
    std::size_t logical_token_count, std::size_t launch_token_count,
    std::size_t lane_count, float clip_ratio, std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(sm87_a4w4_factorized_lane_quantize_plan(
                  63U, 64U, 5'120U, 4U)
                  .lane_input_size == 1'280U);
static_assert(sm87_a4w4_factorized_lane_quantize_plan(
                  193U, 256U, 17'408U, 4U)
                  .lane_input_size == 4'352U);
static_assert(sm87_a4w4_factorized_lane_quantize_plan(
                  1'853U, 1'920U, 5'120U, 1U)
                  .packed_capacity_bytes == 4'915'200U,
              "the real P1853 Gate R1 launch must remain addressable");
static_assert(sm87_a4w4_factorized_lane_quantize_plan(
                  1'853U, 1'920U, 6'144U, 1U)
                  .packed_capacity_bytes == 5'898'240U,
              "the real P1853 Attention-O R1 launch must remain addressable");
static_assert(sm87_a4w4_factorized_lane_quantize_plan(
                  1'853U, 1'920U, 17'408U, 4U)
                  .launch_ctas == 7'680U,
              "the future M192 Down R4 launch must remain addressable");
static_assert(sm87_a4w4_factorized_lane_quantize_plan(
                  1'853U, 2'048U, 17'408U, 1U)
                  .packed_capacity_bytes == 17'825'792U,
              "the padded M2048 Down R1 launch must remain addressable");
static_assert(!sm87_a4w4_factorized_lane_quantize_plan(
                   64U, 64U, 5'120U, 2U)
                   .valid(),
              "R2 remains ABI-reserved and is not executable");
static_assert(sm87_a4w4_factorized_lane_scale_offset(64U, 3U, 4U) ==
              448U);

}  // namespace q3x::kernels
