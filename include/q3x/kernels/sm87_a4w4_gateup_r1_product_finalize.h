#pragma once

#include "q3x/kernels/sm87_a4w4_factorized_lane_quantize.h"
#include "q3x/kernels/sm87_a4w4_down_factorized_lane.h"
#include "q3x/kernels/sm87_a4w4_gateup_factorized_lane.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Candidate-only consumer for the GateUp N128 maxima sidecar.  One CTA owns
// a complete padded row: it reduces the 136 producer maxima, publishes the
// incumbent lane-count-one BF16 scale, then reads every split BF16 GateUp
// product exactly once while emitting the established consumer-order A4
// payload for Down.  `launch_token_count` is the Down M256 launch extent;
// the maxima sidecar remains sized to the independent GateUp M128 launch
// extent recorded in `gate_launch_token_count`.  No global atomic or
// cooperative-grid dependency exists.
inline constexpr std::size_t kSm87A4W4GateUpR1ProductFinalizeThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpR1ProductFinalizeMinimumBlocksPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpR1ProductFinalizeMaximumRegisters = 128U;

struct Sm87A4W4GateUpR1ProductFinalizePlan final {
  std::size_t logical_token_count{};
  std::size_t gate_launch_token_count{};
  std::size_t launch_token_count{};
  std::size_t input_size{};
  std::size_t partial_tiles{};
  std::size_t launch_ctas{};
  std::size_t partial_capacity_elements{};
  std::size_t packed_capacity_bytes{};
  std::size_t scale_capacity_elements{};

  [[nodiscard]] constexpr bool valid() const noexcept {
    return launch_ctas != 0U;
  }
  [[nodiscard]] constexpr explicit operator bool() const noexcept {
    return valid();
  }
};

[[nodiscard]] constexpr Sm87A4W4GateUpR1ProductFinalizePlan
sm87_a4w4_gateup_r1_product_finalize_plan(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count) noexcept {
  const std::size_t gate_launch_token_count =
      sm87_a4w4_gateup_factorized_launch_token_count(logical_token_count);
  if (gate_launch_token_count == 0U ||
      launch_token_count !=
          sm87_a4w4_down_factorized_launch_token_count(
              logical_token_count) ||
      !sm87_a4w4_factorized_lane_quantize_launch_tokens_supported(
          launch_token_count)) {
    return {};
  }
  const std::size_t partial_capacity =
      sm87_a4w4_gateup_factorized_r1_product_partial_capacity_elements(
          gate_launch_token_count);
  const std::size_t packed_capacity =
      sm87_a4w4_consumer_packed_capacity_bytes(
          launch_token_count,
          kSm87A4W4GateUpFactorizedModelIntermediate);
  const std::size_t scale_capacity =
      sm87_a4w4_factorized_lane_scale_capacity_elements(
          launch_token_count, 1U);
  return partial_capacity == 0U || packed_capacity == 0U ||
                 scale_capacity == 0U
             ? Sm87A4W4GateUpR1ProductFinalizePlan{}
             : Sm87A4W4GateUpR1ProductFinalizePlan{
                   logical_token_count,
                   gate_launch_token_count,
                   launch_token_count,
                   kSm87A4W4GateUpFactorizedModelIntermediate,
                   kSm87A4W4GateUpFactorizedR1ProductPartialTiles,
                   launch_token_count,
                   partial_capacity,
                   packed_capacity,
                   scale_capacity};
}

struct Sm87A4W4GateUpR1ProductFinalizeResources final {
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

[[nodiscard]] int
query_sm87_a4w4_gateup_r1_product_finalize_resources_cuda(
    Sm87A4W4GateUpR1ProductFinalizeResources* resources) noexcept;

[[nodiscard]] int launch_sm87_a4w4_gateup_r1_product_finalize_cuda(
    const std::uint16_t* primary_product_bf16,
    std::size_t primary_product_row_stride_elements,
    std::size_t primary_product_capacity_elements,
    const std::uint16_t* secondary_product_bf16,
    std::size_t secondary_product_row_stride_elements,
    std::size_t secondary_product_capacity_elements,
    const float* authenticated_down_inverse_alpha_fp32,
    std::size_t down_inverse_alpha_capacity_elements,
    const float* r1_product_tile_maxima_fp32,
    std::size_t r1_product_tile_maxima_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    float clip_ratio,
    std::uint8_t* packed_down_a,
    std::size_t packed_down_a_capacity_bytes,
    std::uint16_t* down_a_lane_scales_bf16,
    std::size_t down_a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

static_assert(sm87_a4w4_gateup_r1_product_finalize_plan(1'853U, 2'048U)
                      .partial_capacity_elements == 261'120U);
static_assert(sm87_a4w4_gateup_r1_product_finalize_plan(1'853U, 2'048U)
                      .packed_capacity_bytes == 17'825'792U);
static_assert(sm87_a4w4_gateup_r1_product_finalize_plan(1'853U, 2'048U)
                      .gate_launch_token_count == 1'920U);
static_assert(sm87_a4w4_gateup_r1_product_finalize_plan(1'853U, 2'048U)
                      .scale_capacity_elements == 2'048U);

}  // namespace q3x::kernels
