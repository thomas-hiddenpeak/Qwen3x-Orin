#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Isolated SM87 Gate+Up -> Down-input edge experiment.  One 512-thread CTA
// owns M64N512.  Warps 0..7 compute Gate and warps 8..15 compute Up;
// projection warp p owns M64N16.  A uses a two-slot K512 cp.async ring while
// projection-major-v3 B fragments are loaded directly into registers.
// Within a K512 group, a non-unrolled plane loop loads both N8 B fragments,
// then one A fragment per M16 panel feeds both MMAs.  This keeps one
// projection's two-fragment partial state live without unrolling the complete
// plane/fragment/panel cross product.
//
// After every N128 cell, the drained A ring aliases an M64N128 FP32 Gate
// exchange.  Up writes the exact BF16_RNE SiLU(Gate)*Up seam into the retained
// 64-KiB M64N512 edge plane.  Four cells are then quantized directly to the
// production signed-A4/K512-scale Down-input ABI.  No global BF16 product is
// emitted.  Production dispatch remains default-off behind an authenticated
// projection-major publication and explicit master/leaf admission selectors.
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineComputeTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineEdgeN = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineProjectionWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineFragmentsPerWarp = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelinePanelsPerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineK64PerGroup = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineAStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineAStageBytes = 16'384U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineScaleSlotBytes = 640U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineSlotBytes = 17'024U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelinePipelineBytes = 34'048U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineGateExchangeBytes = 32'768U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineEdgeBytes = 65'536U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineDynamicSharedBytes = 99'584U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelineCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N128RegisterPipelinePersistentCtas = 16U;

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m64n128_register_pipeline_a_scale_capacity(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  return sm87_a4w4_gateup_down_edge_scale_capacity_elements(
      outer_count, logical_k);
}

struct Sm87A4W4GateUpK512M64N128RegisterPipelineResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_k512_m64n128_register_pipeline_resources_cuda(
    Sm87A4W4GateUpK512M64N128RegisterPipelineResources* resources)
    noexcept;

#define Q3X_SM87_A4W4_GATEUP_M64N128_REGISTER_PIPELINE_ARGUMENTS          \
  const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,      \
  const std::uint16_t* a_k512_scales_bf16,                               \
  std::size_t a_scale_capacity_elements,                                 \
  const std::uint8_t* projection_major_v3_b_codes,                       \
  std::size_t projection_major_v3_b_code_capacity_bytes,                 \
  const std::uint16_t* paired_b_scales_bf16,                             \
  std::size_t paired_b_scale_capacity_elements,                          \
  std::size_t logical_token_count, std::size_t launch_token_count,       \
  std::size_t intermediate_size, std::size_t input_size,                 \
  float output_clip_ratio, std::uint8_t* packed_output,                  \
  std::size_t packed_output_capacity_bytes,                              \
  std::uint16_t* output_k512_scales_bf16,                                \
  std::size_t output_scale_capacity_elements

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m64n128_register_pipeline_cuda(
    Q3X_SM87_A4W4_GATEUP_M64N128_REGISTER_PIPELINE_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m64n128_register_pipeline_test_cuda(
    Q3X_SM87_A4W4_GATEUP_M64N128_REGISTER_PIPELINE_ARGUMENTS,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_M64N128_REGISTER_PIPELINE_ARGUMENTS

static_assert(kSm87A4W4GateUpK512M64N128RegisterPipelineDynamicSharedBytes ==
              99'584U);
static_assert(kSm87A4W4GateUpK512M64N128RegisterPipelineEdgeN ==
              kSm87A4W4GateUpDownEdgeScaleK);

}  // namespace q3x::kernels
