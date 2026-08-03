#pragma once

#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Isolated SM87 Gate+Up -> Down-input edge experiment.  One 512-thread CTA
// owns M64N512 and visits four M64N128 cells.  Every one of the 16 warps owns
// one M64N8 slice and computes both Gate and Up.  For each K64 plane one
// ldmatrix.x4 A fragment is reused by consecutive Gate/Up IMMA instructions;
// paired-v2 B uses a one-plane register lookahead loaded with .cg.
// A and K512 scales use a two-slot cp.async ring.
//
// Gate and Up remain warp-local through exact BF16_RNE SiLU(Gate)*Up, so this
// candidate has no Gate shared exchange and no associated CTA barrier.  Each
// warp writes its BF16 pairs into the retained 64-KiB M64N512 edge plane;
// after four cells it is quantized directly to the production signed-A4/K512
// Down-input ABI.  No global BF16 product is emitted.  A build-time and
// runtime default-off route exists only to reproduce the rejected real-model
// experiment; it must not be selected by a production mode.
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineComputeTileN = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineEdgeN = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineThreads = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineWarps = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineWarpN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineProjections = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelinePanelsPerWarp = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineK64PerGroup = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineAStages = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineAStageBytes = 16'384U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineScaleSlotBytes = 640U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineSlotBytes = 17'024U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelinePipelineBytes = 34'048U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineEdgeBytes = 65'536U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineDynamicSharedBytes = 99'584U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelinePersistentCtas = 16U;

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_a_scale_capacity(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  return sm87_a4w4_gateup_down_edge_scale_capacity_elements(
      outer_count, logical_k);
}

struct Sm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineResources final {
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
query_sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_resources_cuda(
    Sm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineResources* resources)
    noexcept;

#define Q3X_SM87_A4W4_GATEUP_M64N8_PAIRED_WARP_PIPELINE_ARGUMENTS        \
  const std::uint8_t* packed_a, std::size_t packed_a_capacity_bytes,      \
  const std::uint16_t* a_k512_scales_bf16,                               \
  std::size_t a_scale_capacity_elements,                                 \
  const std::uint8_t* paired_b_codes,                                    \
  std::size_t paired_b_code_capacity_bytes,                              \
  const std::uint16_t* paired_b_scales_bf16,                             \
  std::size_t paired_b_scale_capacity_elements,                          \
  std::size_t logical_token_count, std::size_t launch_token_count,       \
  std::size_t intermediate_size, std::size_t input_size,                 \
  float output_clip_ratio, std::uint8_t* packed_output,                  \
  std::size_t packed_output_capacity_bytes,                              \
  std::uint16_t* output_k512_scales_bf16,                                \
  std::size_t output_scale_capacity_elements

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_cuda(
    Q3X_SM87_A4W4_GATEUP_M64N8_PAIRED_WARP_PIPELINE_ARGUMENTS,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_m64n8_paired_warp_register_pipeline_test_cuda(
    Q3X_SM87_A4W4_GATEUP_M64N8_PAIRED_WARP_PIPELINE_ARGUMENTS,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

#undef Q3X_SM87_A4W4_GATEUP_M64N8_PAIRED_WARP_PIPELINE_ARGUMENTS

static_assert(kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineDynamicSharedBytes ==
              99'584U);
static_assert(kSm87A4W4GateUpK512M64N8PairedWarpRegisterPipelineEdgeN ==
              kSm87A4W4GateUpDownEdgeScaleK);

}  // namespace q3x::kernels
