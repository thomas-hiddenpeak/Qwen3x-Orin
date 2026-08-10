#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Default-off, BUILD_TESTING-only surface for a phase-local decoded-B
// experiment.  Every admitted role is one exact P40000 dense BF16 GEMM:
//
//   C[M,N] = alpha * A[M,K] * transpose(W[N,K])
//
// A, W, and C are row-major BF16 bit patterns.  The CUDA body uses an
// ordinary two-dimensional multi-wave grid.  It is deliberately independent
// of the production projection selector and has no cuBLAS/cuBLASLt fallback.
enum class Sm87P40PhaseBf16ProjectionRole : std::uint8_t {
  // One independent Gate or Up projection from the rejected experiment.
  // This primitive is retained only for correctness evidence and is not a
  // design promise for the packed-operand production successor.
  kGateOrUpK5120N17408 = 0U,
  kDownK17408N5120 = 1U,
  kFp8K5120N1024 = 2U,
  kFp8K5120N5120 = 3U,
  kFp8K5120N6144 = 4U,
  kFp8K5120N10240 = 5U,
  kFp8K5120N12288 = 6U,
  kFp8K6144N5120 = 7U,
};

inline constexpr std::size_t kSm87P40PhaseBf16Tokens = 40'000U;
inline constexpr std::size_t kSm87P40PhaseBf16TileM = 128U;
inline constexpr std::size_t kSm87P40PhaseBf16TileN = 128U;
inline constexpr std::size_t kSm87P40PhaseBf16TileK = 64U;
inline constexpr std::size_t kSm87P40PhaseBf16Threads = 256U;
inline constexpr std::size_t kSm87P40PhaseBf16Warps = 8U;
inline constexpr std::size_t kSm87P40PhaseBf16PipelineStages = 2U;
inline constexpr std::size_t kSm87P40PhaseBf16SharedLeadingDimension = 72U;
inline constexpr std::size_t kSm87P40PhaseBf16SmCount = 16U;
inline constexpr int kSm87P40PhaseBf16ExpectedRegistersPerThread = 158;
inline constexpr int kSm87P40PhaseBf16ExpectedActiveBlocksPerSm = 1;
inline constexpr std::size_t kSm87P40PhaseBf16SharedABytesPerStage =
    kSm87P40PhaseBf16TileM * kSm87P40PhaseBf16SharedLeadingDimension *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87P40PhaseBf16SharedBBytesPerStage =
    kSm87P40PhaseBf16TileN * kSm87P40PhaseBf16SharedLeadingDimension *
    sizeof(std::uint16_t);
inline constexpr std::size_t kSm87P40PhaseBf16DynamicSharedBytes =
    kSm87P40PhaseBf16PipelineStages *
    (kSm87P40PhaseBf16SharedABytesPerStage +
     kSm87P40PhaseBf16SharedBBytesPerStage);

static_assert(kSm87P40PhaseBf16Threads ==
              32U * kSm87P40PhaseBf16Warps);
static_assert(kSm87P40PhaseBf16Tokens ==
              312U * kSm87P40PhaseBf16TileM + 64U);
static_assert(kSm87P40PhaseBf16SharedABytesPerStage == 18'432U);
static_assert(kSm87P40PhaseBf16SharedBBytesPerStage == 18'432U);
static_assert(kSm87P40PhaseBf16DynamicSharedBytes == 73'728U);

struct Sm87P40PhaseBf16ProjectionPlan final {
  Sm87P40PhaseBf16ProjectionRole role =
      Sm87P40PhaseBf16ProjectionRole::kGateOrUpK5120N17408;
  std::size_t token_count = 0U;
  std::size_t input_features = 0U;
  std::size_t output_features = 0U;
  std::size_t grid_m = 0U;
  std::size_t grid_n = 0U;
  std::size_t k_stages = 0U;
  std::size_t logical_ctas = 0U;
  std::size_t minimum_sm_waves = 0U;
  std::size_t threads = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  bool ordinary_two_dimensional_grid = false;
  bool admitted = false;

  [[nodiscard]] constexpr bool valid() const noexcept;
};

[[nodiscard]] constexpr bool sm87_p40_phase_bf16_role_shape(
    const Sm87P40PhaseBf16ProjectionRole role, std::size_t* const k,
    std::size_t* const n) noexcept {
  if (k == nullptr || n == nullptr) {
    return false;
  }
  switch (role) {
    case Sm87P40PhaseBf16ProjectionRole::kGateOrUpK5120N17408:
      *k = 5'120U;
      *n = 17'408U;
      return true;
    case Sm87P40PhaseBf16ProjectionRole::kDownK17408N5120:
      *k = 17'408U;
      *n = 5'120U;
      return true;
    case Sm87P40PhaseBf16ProjectionRole::kFp8K5120N1024:
      *k = 5'120U;
      *n = 1'024U;
      return true;
    case Sm87P40PhaseBf16ProjectionRole::kFp8K5120N5120:
      *k = 5'120U;
      *n = 5'120U;
      return true;
    case Sm87P40PhaseBf16ProjectionRole::kFp8K5120N6144:
      *k = 5'120U;
      *n = 6'144U;
      return true;
    case Sm87P40PhaseBf16ProjectionRole::kFp8K5120N10240:
      *k = 5'120U;
      *n = 10'240U;
      return true;
    case Sm87P40PhaseBf16ProjectionRole::kFp8K5120N12288:
      *k = 5'120U;
      *n = 12'288U;
      return true;
    case Sm87P40PhaseBf16ProjectionRole::kFp8K6144N5120:
      *k = 6'144U;
      *n = 5'120U;
      return true;
  }
  return false;
}

[[nodiscard]] constexpr Sm87P40PhaseBf16ProjectionPlan
make_sm87_p40_phase_bf16_projection_plan(
    const Sm87P40PhaseBf16ProjectionRole role,
    const std::size_t token_count) noexcept {
  std::size_t k = 0U;
  std::size_t n = 0U;
  if (token_count != kSm87P40PhaseBf16Tokens ||
      !sm87_p40_phase_bf16_role_shape(role, &k, &n) ||
      (k % kSm87P40PhaseBf16TileK) != 0U ||
      (n % kSm87P40PhaseBf16TileN) != 0U) {
    return {role, token_count};
  }
  const std::size_t grid_m =
      (token_count + kSm87P40PhaseBf16TileM - 1U) /
      kSm87P40PhaseBf16TileM;
  const std::size_t grid_n = n / kSm87P40PhaseBf16TileN;
  const std::size_t logical_ctas = grid_m * grid_n;
  return {
      role,
      token_count,
      k,
      n,
      grid_m,
      grid_n,
      k / kSm87P40PhaseBf16TileK,
      logical_ctas,
      (logical_ctas + kSm87P40PhaseBf16SmCount - 1U) /
          kSm87P40PhaseBf16SmCount,
      kSm87P40PhaseBf16Threads,
      kSm87P40PhaseBf16DynamicSharedBytes,
      true,
      true,
  };
}

[[nodiscard]] constexpr bool
Sm87P40PhaseBf16ProjectionPlan::valid() const noexcept {
  std::size_t expected_k = 0U;
  std::size_t expected_n = 0U;
  if (!admitted || token_count != kSm87P40PhaseBf16Tokens ||
      !sm87_p40_phase_bf16_role_shape(role, &expected_k, &expected_n)) {
    return false;
  }
  return input_features == expected_k && output_features == expected_n &&
         grid_m ==
             (token_count + kSm87P40PhaseBf16TileM - 1U) /
                 kSm87P40PhaseBf16TileM &&
         grid_n == output_features / kSm87P40PhaseBf16TileN &&
         k_stages == input_features / kSm87P40PhaseBf16TileK &&
         logical_ctas == grid_m * grid_n &&
         minimum_sm_waves ==
             (logical_ctas + kSm87P40PhaseBf16SmCount - 1U) /
                 kSm87P40PhaseBf16SmCount &&
         threads == kSm87P40PhaseBf16Threads &&
         dynamic_shared_bytes == kSm87P40PhaseBf16DynamicSharedBytes &&
         ordinary_two_dimensional_grid;
}

struct Sm87P40PhaseBf16ProjectionTile final {
  std::size_t block_x = 0U;
  std::size_t block_y = 0U;
  std::size_t first_token = 0U;
  std::size_t token_count = 0U;
  std::size_t first_output_feature = 0U;
  std::size_t output_feature_count = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87P40PhaseBf16ProjectionTile
sm87_p40_phase_bf16_projection_tile(
    const Sm87P40PhaseBf16ProjectionPlan& plan,
    const std::size_t block_x, const std::size_t block_y) noexcept {
  if (!plan.valid() || block_x >= plan.grid_n || block_y >= plan.grid_m) {
    return {};
  }
  const std::size_t first_token = block_y * kSm87P40PhaseBf16TileM;
  const std::size_t remaining = plan.token_count - first_token;
  return {
      block_x,
      block_y,
      first_token,
      remaining < kSm87P40PhaseBf16TileM
          ? remaining
          : kSm87P40PhaseBf16TileM,
      block_x * kSm87P40PhaseBf16TileN,
      kSm87P40PhaseBf16TileN,
      true,
  };
}

struct Sm87P40PhaseBf16ProjectionResources final {
  int compute_major = 0;
  int compute_minor = 0;
  int sm_count = 0;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int active_blocks_per_sm = 0;
  bool admitted = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return admitted && compute_major == 8 && compute_minor == 7 &&
           sm_count == static_cast<int>(kSm87P40PhaseBf16SmCount) &&
           binary_version == 87 &&
           registers_per_thread ==
               kSm87P40PhaseBf16ExpectedRegistersPerThread &&
           static_shared_bytes == 0U &&
           dynamic_shared_bytes == kSm87P40PhaseBf16DynamicSharedBytes &&
           local_bytes == 0U &&
           active_blocks_per_sm ==
               kSm87P40PhaseBf16ExpectedActiveBlocksPerSm;
  }
};

// Seals the exact 16-SM SM87 cubin/resource inventory for one role.  It never
// queries or selects an external GEMM implementation.
[[nodiscard]] int query_sm87_p40_phase_bf16_projection_resources_cuda(
    Sm87P40PhaseBf16ProjectionRole role, std::size_t token_count,
    Sm87P40PhaseBf16ProjectionResources* resources) noexcept;

// Launches dim3(plan.grid_n, plan.grid_m, 1) ordinary CTAs.  A is [M,K], W is
// [N,K], and C is [M,N], all row-major BF16.  alpha is applied in FP32 before
// the single BF16 RNE publication.  The surface rejects every M other than
// 40000, every device other than the exact 16-SM SM87 target, misalignment,
// and output aliasing.  It has no fused activation/residual epilogue and no
// fallback path.
[[nodiscard]] int launch_sm87_p40_phase_bf16_projection_cuda(
    Sm87P40PhaseBf16ProjectionRole role,
    const std::uint16_t* activations, const std::uint16_t* weights,
    std::size_t token_count, float alpha, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
