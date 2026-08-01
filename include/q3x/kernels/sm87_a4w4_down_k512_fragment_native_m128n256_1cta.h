#pragma once

#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Standalone, default-off Down experiment over the authenticated v2 K512
// fragment payload.  One 256-thread CTA owns a complete M128N256 cell.  Warp
// w owns M128N32 at N=32*w and consumes that stripe as two ordered N16
// microphases.  Each 128-bit B record is loaded once per K64 and is reused by
// eight M16 IMMA panels; B never enters shared memory.
//
// A uses two M128K512 cp.async stages.  Two accompanying K512 scale slots
// keep the next group's A128 and B256 scales in the same pipeline epoch.  The
// resulting 67,072-byte shared allocation and launch_bounds(256, 1) make the
// experiment explicitly one-CTA/SM.  It reuses the existing v2 payload byte
// ABI without repacking or reinterpretation.
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaTileN = 256U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaWarpM = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaWarpN = 32U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp = 8U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaN16Phases = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaN8PerPhase = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaK64PerScale = 8U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaAStages = 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaAStageBytes =
        kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM *
        kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleK / 2U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleSlotBytes =
        (kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM +
         kSm87A4W4DownK512FragmentNativeM128N2561CtaTileN) *
        sizeof(std::uint16_t);
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes =
        kSm87A4W4DownK512FragmentNativeM128N2561CtaAStages *
        (kSm87A4W4DownK512FragmentNativeM128N2561CtaAStageBytes +
         kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleSlotBytes);
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaPersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaCtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaMaximumRegisters = 255U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaMinimumTokens = 128U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaMaximumTokens = 4'096U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaModelOutput = 5'120U;
inline constexpr std::size_t
    kSm87A4W4DownK512FragmentNativeM128N2561CtaModelInput = 17'408U;

static_assert(
    kSm87A4W4DownK512FragmentNativeM128N2561CtaAStageBytes ==
    32'768U);
static_assert(
    kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleSlotBytes ==
    768U);
static_assert(
    kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes ==
    67'072U);

struct Sm87A4W4DownK512FragmentNativeM128N2561CtaPlan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_stripes{};
  std::size_t k512_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t m_owner_ctas{};
  std::size_t n_wave_groups{};
  std::size_t launch_ctas{};
  std::size_t maximum_m_tiles_per_owner{};
  std::size_t maximum_n_stripes_per_group{};
};

[[nodiscard]] constexpr
Sm87A4W4DownK512FragmentNativeM128N2561CtaPlan
sm87_a4w4_down_k512_fragment_native_m128n256_1cta_test_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count %
              kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM !=
          0U ||
      output_size == 0U ||
      output_size %
              kSm87A4W4DownK512FragmentNativeM128N2561CtaTileN !=
          0U ||
      input_size == 0U ||
      input_size %
              kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleK !=
          0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count /
      kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM;
  const std::size_t n_stripes =
      output_size /
      kSm87A4W4DownK512FragmentNativeM128N2561CtaTileN;
  if (!sm87_a4w4_down_k512_product_fits(m_tiles, n_stripes)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * n_stripes;
  const std::size_t m_owner_ctas =
      m_tiles <
              kSm87A4W4DownK512FragmentNativeM128N2561CtaPersistentCtas
          ? m_tiles
          : kSm87A4W4DownK512FragmentNativeM128N2561CtaPersistentCtas;
  const std::size_t available_n_groups =
      kSm87A4W4DownK512FragmentNativeM128N2561CtaPersistentCtas /
      m_owner_ctas;
  const std::size_t n_wave_groups =
      n_stripes < available_n_groups ? n_stripes : available_n_groups;
  const std::size_t launch_ctas = m_owner_ctas * n_wave_groups;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_stripes,
          input_size /
              kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleK,
          input_size / kSm87A4W4DownK512PhysicalK64,
          work_cells,
          m_owner_ctas,
          n_wave_groups,
          launch_ctas,
          (m_tiles + m_owner_ctas - 1U) / m_owner_ctas,
          (n_stripes + n_wave_groups - 1U) / n_wave_groups};
}

[[nodiscard]] constexpr
Sm87A4W4DownK512FragmentNativeM128N2561CtaPlan
sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return token_count >=
                 kSm87A4W4DownK512FragmentNativeM128N2561CtaMinimumTokens &&
                 token_count <=
                     kSm87A4W4DownK512FragmentNativeM128N2561CtaMaximumTokens &&
                 output_size ==
                     kSm87A4W4DownK512FragmentNativeM128N2561CtaModelOutput &&
                 input_size ==
                     kSm87A4W4DownK512FragmentNativeM128N2561CtaModelInput
             ? sm87_a4w4_down_k512_fragment_native_m128n256_1cta_test_plan(
                   token_count, output_size, input_size)
             : Sm87A4W4DownK512FragmentNativeM128N2561CtaPlan{};
}

struct Sm87A4W4DownK512FragmentNativeM128N2561CtaTile final {
  std::size_t m_tile{};
  std::size_t n_stripe{};
  bool valid{};
};

// Mirrors the kernel's nested N-wave/M-owner traversal.  This is exposed so
// the contract test can prove that both the P1920 15-CTA wave and the P512
// four-wave short-M schedule visit every complete cell exactly once.
[[nodiscard]] constexpr
Sm87A4W4DownK512FragmentNativeM128N2561CtaTile
sm87_a4w4_down_k512_fragment_native_m128n256_1cta_tile(
    const Sm87A4W4DownK512FragmentNativeM128N2561CtaPlan& plan,
    const std::size_t cta, const std::size_t n_iteration,
    const std::size_t m_iteration) noexcept {
  if (plan.launch_ctas == 0U || cta >= plan.launch_ctas) {
    return {};
  }
  const std::size_t n_group = cta / plan.m_owner_ctas;
  const std::size_t m_owner = cta % plan.m_owner_ctas;
  const std::size_t n_stripe =
      n_group + n_iteration * plan.n_wave_groups;
  const std::size_t m_tile =
      m_owner + m_iteration * plan.m_owner_ctas;
  return n_stripe < plan.n_stripes && m_tile < plan.m_tiles
             ? Sm87A4W4DownK512FragmentNativeM128N2561CtaTile{
                   m_tile, n_stripe, true}
             : Sm87A4W4DownK512FragmentNativeM128N2561CtaTile{};
}

struct Sm87A4W4DownK512FragmentNativeM128N2561CtaResources final {
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
query_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_resources_cuda(
    Sm87A4W4DownK512FragmentNativeM128N2561CtaResources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* fragment_native_b,
    std::size_t fragment_native_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* fragment_native_b,
    std::size_t fragment_native_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
