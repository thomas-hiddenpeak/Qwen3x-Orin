#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_down_whole_p40.h"

#include "sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_oracle_internal.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace q3x::kernels {
namespace {

namespace cg = cooperative_groups;

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kPersistentCtas = 32U;
constexpr unsigned int kTileM = 64U;
constexpr unsigned int kTileN = 256U;
constexpr unsigned int kTileK = 64U;
constexpr unsigned int kIntermediate = 17'408U;
constexpr unsigned int kHidden = 5'120U;
constexpr unsigned int kKTiles = 272U;
constexpr unsigned int kN8Panels = 8U;
constexpr unsigned int kK16Panels = 4U;
constexpr unsigned int kStages = 3U;
constexpr unsigned int kStageVectors = 1'088U;
constexpr unsigned int kActivationVectors = 512U;
constexpr unsigned int kWeightVectors = 512U;
constexpr unsigned int kScaleVectors = 64U;
constexpr unsigned int kPackedCellWeightBytes = 8'192U;
constexpr unsigned int kPackedCellScaleBytes = 1'024U;
constexpr unsigned int kPackedCellBytes = 9'216U;

static_assert(kThreads == kSm87BulkV2NvFp4DownWholeP40Threads);
static_assert(kPersistentCtas ==
              kSm87BulkV2NvFp4DownWholeP40PersistentCtas);
static_assert(kTileM == kSm87BulkV2NvFp4DownWholeP40TileM);
static_assert(kTileN == kSm87BulkV2NvFp4DownWholeP40TileN);
static_assert(kTileK == kSm87BulkV2NvFp4DownWholeP40TileK);
static_assert(kKTiles == kSm87BulkV2NvFp4DownWholeP40KTiles);
static_assert(kStages ==
              kSm87BulkV2NvFp4DownWholeP40PipelineStages);
static_assert(kStageVectors * sizeof(uint4) ==
              kSm87BulkV2NvFp4DownWholeP40BytesPerStage);
static_assert(kPackedCellWeightBytes + kPackedCellScaleBytes ==
              kPackedCellBytes);

struct alignas(16) DownPipeline final {
  uint4 stage[kStages][kStageVectors];
};

static_assert(sizeof(DownPipeline) ==
              kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes);

struct M16K16Activation final {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct K16N8Weight final {
  std::uint32_t x0;
  std::uint32_t x1;
};

struct M16N8Accumulator final {
  float x0;
  float x1;
  float x2;
  float x3;
};

using DownWarpAccumulator = M16N8Accumulator[2U][kN8Panels];

template <bool kPredicate>
__device__ __forceinline__ void cp_async_cg_16(
    void* const shared_destination, const void* const global_source,
    const bool valid = true) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(
          __cvta_generic_to_shared(shared_destination));
  if constexpr (kPredicate) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(shared_address), "l"(global_source),
                   "r"(valid ? 16U : 0U)
                 : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :
                 : "r"(shared_address), "l"(global_source)
                 : "memory");
  }
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      valid ? *reinterpret_cast<const uint4*>(global_source)
            : make_uint4(0U, 0U, 0U, 0U);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <unsigned int kOutstanding>
__device__ __forceinline__ void cp_async_wait_group() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(kOutstanding)
               : "memory");
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ float decode_e2m1(
    const std::uint8_t code) noexcept {
  const std::uint8_t magnitude = code & 0x07U;
  float value = 0.0F;
  if (magnitude == 1U) {
    value = 0.5F;
  } else if (magnitude == 2U) {
    value = 1.0F;
  } else if (magnitude == 3U) {
    value = 1.5F;
  } else if (magnitude == 4U) {
    value = 2.0F;
  } else if (magnitude == 5U) {
    value = 3.0F;
  } else if (magnitude == 6U) {
    value = 4.0F;
  } else if (magnitude == 7U) {
    value = 6.0F;
  }
  return (code & 0x08U) != 0U ? -value : value;
}

[[nodiscard]] __device__ __forceinline__ float decode_e4m3fn_scale(
    const std::uint8_t code) noexcept {
  const unsigned int magnitude = code & 0x7fU;
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  const float value =
      exponent == 0U
          ? ldexpf(static_cast<float>(mantissa), -9)
          : ldexpf(static_cast<float>(8U + mantissa),
                   static_cast<int>(exponent) - 10);
  return (code & 0x80U) != 0U ? -value : value;
}

[[nodiscard]] __device__ __forceinline__ K16N8Weight
decode_weight_fragment(const std::uint8_t* const shared_weight,
                       const std::uint8_t* const shared_scale,
                       const unsigned int fragment,
                       const unsigned int lane) noexcept {
  const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(
      shared_weight + fragment * 64U + lane * 2U);
  const float scale =
      decode_e4m3fn_scale(shared_scale[fragment * 8U + lane / 4U]);
  const auto component = [packed, scale](const unsigned int persisted) {
    const auto code = static_cast<std::uint8_t>(
        (packed >> (4U * persisted)) & 0x0fU);
    return encode_bf16_rne(decode_e2m1(code) * scale);
  };
  // Persisted [K0,K8,K1,K9] becomes the col-major MMA pair
  // x0=[K0,K1], x1=[K8,K9]. Every decoded value first rounds to BF16.
  const std::uint16_t k0 = component(0U);
  const std::uint16_t k8 = component(1U);
  const std::uint16_t k1 = component(2U);
  const std::uint16_t k9 = component(3U);
  return {static_cast<std::uint32_t>(k0) |
              (static_cast<std::uint32_t>(k1) << 16U),
          static_cast<std::uint32_t>(k8) |
              (static_cast<std::uint32_t>(k9) << 16U)};
}

__device__ __forceinline__ void load_activation_fragment(
    M16K16Activation& fragment,
    const std::uint16_t* const shared_activation,
    const unsigned int m16, const unsigned int k16,
    const unsigned int lane) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      m16 * 16U + lane % 8U + (quadrant & 1U) * 8U;
  const unsigned int column =
      k16 * 16U + (quadrant >> 1U) * 8U;
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(
          shared_activation + row * kTileK + column));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1), "=r"(fragment.x2),
        "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_activation;
  (void)m16;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void mma_bf16(
    M16N8Accumulator& accumulator,
    const M16K16Activation& activation,
    const K16N8Weight& weight) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(activation.x0), "r"(activation.x1),
        "r"(activation.x2), "r"(activation.x3),
        "r"(weight.x0), "r"(weight.x1));
#else
  (void)accumulator;
  (void)activation;
  (void)weight;
#endif
}

__device__ __forceinline__ void clear_accumulator(
    DownWarpAccumulator& accumulator) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U; m16 < 2U; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      accumulator[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

__device__ __forceinline__ void issue_stage(
    DownPipeline* const pipeline, const unsigned int slot,
    const std::uint16_t* const h,
    const std::uint8_t* const down_payload,
    const unsigned int m_tile, const unsigned int n_tile,
    const unsigned int k_tile) noexcept {
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / 8U;
    const unsigned int vector = vector_index % 8U;
    const auto* const source = reinterpret_cast<const uint4*>(
                                  h +
                                  (static_cast<std::size_t>(m_tile) *
                                       kTileM +
                                   row) *
                                      kIntermediate +
                                  k_tile * kTileK) +
                              vector;
    cp_async_cg_16<false>(
        &pipeline->stage[slot][vector_index], source);
  }

  const std::uint64_t cell =
      (static_cast<std::uint64_t>(n_tile) * kKTiles + k_tile) *
      kPackedCellBytes;
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector = threadIdx.x + pass * kThreads;
    const auto* const source = reinterpret_cast<const uint4*>(
        down_payload + cell) + vector;
    cp_async_cg_16<false>(
        &pipeline->stage[slot][kActivationVectors + vector], source);
  }
  if (threadIdx.x < kScaleVectors) {
    const auto* const source = reinterpret_cast<const uint4*>(
        down_payload + cell + kPackedCellWeightBytes) + threadIdx.x;
    cp_async_cg_16<false>(
        &pipeline->stage[slot]
                        [kActivationVectors + kWeightVectors +
                         threadIdx.x],
        source);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void consume_stage(
    const DownPipeline* const pipeline, const unsigned int slot,
    DownWarpAccumulator& accumulator) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const auto* const shared_activation =
      reinterpret_cast<const std::uint16_t*>(
          &pipeline->stage[slot][0U]);
  const auto* const shared_weight =
      reinterpret_cast<const std::uint8_t*>(
          &pipeline->stage[slot][kActivationVectors]);
  const auto* const shared_scale =
      reinterpret_cast<const std::uint8_t*>(
          &pipeline->stage[slot]
                          [kActivationVectors + kWeightVectors]);

#pragma unroll
  for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
    M16K16Activation activation[2U];
#pragma unroll
    for (unsigned int local_m16 = 0U; local_m16 < 2U; ++local_m16) {
      load_activation_fragment(activation[local_m16], shared_activation,
                               warp_m * 2U + local_m16, k16, lane);
    }

    // Two-stage shared-to-register feed without retaining an entire N64 B
    // panel. The next decoded N8 fragment is prepared in the alternate pair
    // while the current pair feeds the two M16 accumulators. This preserves
    // the 64-float full-K C tile and avoids a register-resident decoded B
    // matrix.
    K16N8Weight decoded[2U];
    decoded[0U] = decode_weight_fragment(
        shared_weight, shared_scale,
        (k16 * 4U + warp_n) * kN8Panels, lane);
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      const unsigned int current = n8 & 1U;
      const unsigned int next = current ^ 1U;
      if (n8 + 1U < kN8Panels) {
        decoded[next] = decode_weight_fragment(
            shared_weight, shared_scale,
            (k16 * 4U + warp_n) * kN8Panels + n8 + 1U, lane);
      }
      mma_bf16(accumulator[0U][n8], activation[0U], decoded[current]);
      mma_bf16(accumulator[1U][n8], activation[1U], decoded[current]);
    }
  }
}

__device__ __forceinline__ void run_full_k(
    DownPipeline* const pipeline, const std::uint16_t* const h,
    const std::uint8_t* const down_payload,
    const unsigned int m_tile, const unsigned int n_tile,
    DownWarpAccumulator& accumulator) noexcept {
  clear_accumulator(accumulator);
  issue_stage(pipeline, 0U, h, down_payload, m_tile, n_tile, 0U);
  issue_stage(pipeline, 1U, h, down_payload, m_tile, n_tile, 1U);
  issue_stage(pipeline, 2U, h, down_payload, m_tile, n_tile, 2U);

#pragma unroll 1
  for (unsigned int k_tile = 0U; k_tile < kKTiles; ++k_tile) {
    if (k_tile + 2U < kKTiles) {
      cp_async_wait_group<2U>();
    } else if (k_tile + 1U < kKTiles) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();
    const unsigned int slot = k_tile % kStages;
    consume_stage(pipeline, slot, accumulator);
    // Every shared read from this K64 stage is complete. Refill the dead slot
    // without changing the ascending per-accumulator K order.
    __syncthreads();
    if (k_tile + kStages < kKTiles) {
      issue_stage(pipeline, slot, h, down_payload, m_tile, n_tile,
                  k_tile + kStages);
    }
  }
  cp_async_wait_group<0U>();
  __syncthreads();
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
add_scaled_residual_pair(const float low, const float high,
                         const float tensor_scale,
                         const std::uint32_t residual_bits) noexcept {
  const float branch0 = decode_bf16(encode_bf16_rne(low * tensor_scale));
  const float branch1 = decode_bf16(encode_bf16_rne(high * tensor_scale));
  const float residual0 =
      decode_bf16(static_cast<std::uint16_t>(residual_bits));
  const float residual1 =
      decode_bf16(static_cast<std::uint16_t>(residual_bits >> 16U));
  return static_cast<std::uint32_t>(
             encode_bf16_rne(branch0 + residual0)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(branch1 + residual1))
          << 16U);
}

__device__ __forceinline__ void publish_residual(
    const DownWarpAccumulator& accumulator,
    const unsigned int m_tile, const unsigned int n_tile,
    const float tensor_scale, std::uint16_t* const residual) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int local_m16 = 0U; local_m16 < 2U; ++local_m16) {
    const unsigned int local_row0 =
        warp_m * 32U + local_m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = m_tile * kTileM + local_row0;
    const unsigned int global_row1 = m_tile * kTileM + local_row1;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      const unsigned int column =
          n_tile * kTileN + warp_n * 64U + n8 * 8U +
          lane_in_group * 2U;
      const auto& value = accumulator[local_m16][n8];
      auto* const destination0 = reinterpret_cast<std::uint32_t*>(
          residual + static_cast<std::size_t>(global_row0) * kHidden +
          column);
      auto* const destination1 = reinterpret_cast<std::uint32_t*>(
          residual + static_cast<std::size_t>(global_row1) * kHidden +
          column);
      *destination0 = add_scaled_residual_pair(
          value.x0, value.x1, tensor_scale, *destination0);
      *destination1 = add_scaled_residual_pair(
          value.x2, value.x3, tensor_scale, *destination1);
    }
  }
}

[[nodiscard]] __device__ __forceinline__ bool cancellation_requested(
    const std::uint32_t* const signal) noexcept {
  return signal != nullptr &&
         *reinterpret_cast<const volatile std::uint32_t*>(signal) != 0U;
}

[[nodiscard]] __device__ __forceinline__ bool cta_cancellation_requested(
    const std::uint32_t* const signal,
    Sm87BulkV2NvFp4DownWholeP40DeviceControl* const control) noexcept {
  bool requested = false;
  if (threadIdx.x == 0U) {
    requested =
        cancellation_requested(signal) ||
        atomicAdd(&control->cancellation_observed, 0U) != 0U;
    if (requested) {
      atomicExch(&control->cancellation_observed, 1U);
    }
  }
  // Entry to both run_full_k() and publish_residual() must be CTA-uniform:
  // each contains CTA barriers or collective warp ownership.  A mapped host
  // cancellation word can change while the CTA is reading it, so only lane
  // zero observes the edge and one CTA collective broadcasts the decision.
  // The collective return value is per-call and cannot be overwritten by a
  // fast warp starting the next super-wave.
  return __syncthreads_or(requested ? 1 : 0) != 0;
}

__global__ __launch_bounds__(256, 2)
void nvfp4_down_whole_p40_kernel(
    const std::uint16_t* const h,
    const std::uint8_t* const down_payload,
    const float tensor_scale, std::uint16_t* const residual,
    Sm87BulkV2NvFp4DownWholeP40DeviceControl* const control,
    const std::uint32_t* const cancellation_signal,
    const unsigned long long transaction_epoch,
    const unsigned int m_tiles, const unsigned int n_tiles) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* const pipeline = reinterpret_cast<DownPipeline*>(shared_bytes);
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    control->transaction_epoch = transaction_epoch;
    control->requested_m_tiles = m_tiles;
    control->requested_n_tiles = n_tiles;
    control->completed_super_waves = 0U;
    control->completed_output_tiles = 0U;
    control->cancellation_observed = 0U;
    control->wave_cancelled = 0U;
    control->first_unfinished_super_wave =
        ((m_tiles + kSm87BulkV2NvFp4DownWholeP40CohortM - 1U) /
         kSm87BulkV2NvFp4DownWholeP40CohortM) *
        ((n_tiles + kSm87BulkV2NvFp4DownWholeP40CohortN - 1U) /
         kSm87BulkV2NvFp4DownWholeP40CohortN);
    control->error_code = 0U;
    control->policy = kSm87BulkV2NvFp4DownWholeP40RequiredPolicy;
    control->reserved[0U] = 0U;
    control->reserved[1U] = 0U;
  }
  if (threadIdx.x == 0U) {
    control->cta_completed_super_waves[blockIdx.x] = 0U;
  }
  // One launch-initialization barrier; there is deliberately no barrier per
  // super-wave. Fixed equal full-K work keeps each complete cohort near phase
  // without replacing useful work with 395 cooperative barriers.
  cg::this_grid().sync();

  const unsigned int m_lane =
      blockIdx.x / kSm87BulkV2NvFp4DownWholeP40CohortN;
  const unsigned int n_lane =
      blockIdx.x % kSm87BulkV2NvFp4DownWholeP40CohortN;
  const unsigned int m_cohorts =
      (m_tiles + kSm87BulkV2NvFp4DownWholeP40CohortM - 1U) /
      kSm87BulkV2NvFp4DownWholeP40CohortM;
  const unsigned int n_cohorts =
      (n_tiles + kSm87BulkV2NvFp4DownWholeP40CohortN - 1U) /
      kSm87BulkV2NvFp4DownWholeP40CohortN;
  const unsigned int super_waves = m_cohorts * n_cohorts;
  unsigned int completed_waves = 0U;

  for (unsigned int super_wave = 0U; super_wave < super_waves;
       ++super_wave) {
    if (cta_cancellation_requested(cancellation_signal, control)) {
      if (threadIdx.x == 0U) {
        atomicExch(&control->wave_cancelled, 1U);
        atomicMin(&control->first_unfinished_super_wave, super_wave);
      }
      break;
    }
    const unsigned int m_cohort = super_wave / n_cohorts;
    const unsigned int n_cohort = super_wave % n_cohorts;
    const unsigned int m_tile =
        m_cohort * kSm87BulkV2NvFp4DownWholeP40CohortM + m_lane;
    const unsigned int n_tile =
        n_cohort * kSm87BulkV2NvFp4DownWholeP40CohortN + n_lane;
    const bool output_owner = m_tile < m_tiles && n_tile < n_tiles;
    if (output_owner) {
      DownWarpAccumulator accumulator;
      run_full_k(pipeline, h, down_payload, m_tile, n_tile, accumulator);
      // Cancellation can arrive while the complete ascending-K accumulator
      // is private.  Re-observe it before any residual store.  The shared
      // receipt also causes later-finishing CTAs to converge on cancellation
      // even if their direct mapped-word observation lags.
      if (cta_cancellation_requested(cancellation_signal, control)) {
        if (threadIdx.x == 0U) {
          atomicExch(&control->wave_cancelled, 1U);
          atomicMin(&control->first_unfinished_super_wave, super_wave);
        }
        break;
      }
      publish_residual(accumulator, m_tile, n_tile, tensor_scale, residual);
      __syncthreads();
      if (threadIdx.x == 0U) {
        atomicAdd(&control->completed_output_tiles, 1U);
      }
    }
    ++completed_waves;
    if (threadIdx.x == 0U) {
      control->cta_completed_super_waves[blockIdx.x] = completed_waves;
    }
  }

  // One terminal barrier makes every per-CTA progress slot visible before
  // block zero publishes the request-level completion summary. Masked tail
  // lanes wait here but issue no padded load, MMA, or store.
  cg::this_grid().sync();
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    unsigned int minimum = super_waves;
    for (unsigned int cta = 0U; cta < kPersistentCtas; ++cta) {
      const unsigned int progress = control->cta_completed_super_waves[cta];
      minimum = progress < minimum ? progress : minimum;
    }
    control->completed_super_waves = minimum;
    control->first_unfinished_super_wave =
        minimum == super_waves ? super_waves : minimum;
  }
}

[[nodiscard]] bool finite_positive(const float value) noexcept {
  return std::isfinite(value) && value > 0.0F;
}

[[nodiscard]] bool pointer_aligned(const void* const pointer,
                                   const std::uintptr_t alignment) noexcept {
  return pointer != nullptr && alignment != 0U &&
         (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
}

[[nodiscard]] cudaError_t configure_dynamic_shared() noexcept {
  return cudaFuncSetAttribute(
      nvfp4_down_whole_p40_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes));
}

[[nodiscard]] cudaError_t launch_raw_unchecked(
    const sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail::RawArguments&
        arguments) noexcept {
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const std::uint16_t* h_argument = arguments.h;
  const std::uint8_t* payload_argument = arguments.down_payload;
  float scale_argument = arguments.tensor_scale;
  std::uint16_t* residual_argument = arguments.residual;
  Sm87BulkV2NvFp4DownWholeP40DeviceControl* control_argument =
      arguments.device_control;
  const std::uint32_t* cancellation_argument =
      arguments.cancellation_signal;
  unsigned long long epoch_argument = arguments.transaction_epoch;
  unsigned int m_tiles_argument = arguments.m_tiles;
  unsigned int n_tiles_argument = arguments.n_tiles;
  void* kernel_arguments[] = {
      &h_argument,          &payload_argument, &scale_argument,
      &residual_argument,   &control_argument, &cancellation_argument,
      &epoch_argument,      &m_tiles_argument, &n_tiles_argument,
  };
  return cudaLaunchCooperativeKernel(
      nvfp4_down_whole_p40_kernel, dim3{kPersistentCtas},
      dim3{kThreads}, kernel_arguments,
      kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes, stream);
}

[[nodiscard]] bool raw_arguments_valid(
    const sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail::RawArguments&
        arguments) noexcept {
  return arguments.transaction_epoch != 0U && arguments.h != nullptr &&
         arguments.down_payload != nullptr && arguments.residual != nullptr &&
         arguments.device_control != nullptr &&
         arguments.cuda_stream != nullptr &&
         arguments.m_tiles > 0U &&
         arguments.m_tiles <= kSm87BulkV2NvFp4DownWholeP40MTiles &&
         arguments.n_tiles > 0U &&
         arguments.n_tiles <= kSm87BulkV2NvFp4DownWholeP40NTiles &&
         finite_positive(arguments.tensor_scale);
}

}  // namespace

int query_sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_resources_cuda(
    const Sm87BulkV2NvFp4DownWholeP40CodeEvidence* const code_evidence,
    Sm87BulkV2NvFp4DownWholeP40Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = cudaGetDevice(&device);
  if (status == cudaSuccess) {
    status = cudaGetDeviceProperties(&properties, device);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kSm87BulkV2NvFp4DownWholeP40SmCount) ||
      properties.cooperativeLaunch == 0 ||
      properties.sharedMemPerBlockOptin <
          kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  status = configure_dynamic_shared();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, nvfp4_down_whole_p40_kernel);
  int active_blocks = 0;
  if (status == cudaSuccess) {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, nvfp4_down_whole_p40_kernel, kThreads,
        kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->kernel_symbol_identity =
      kSm87BulkV2NvFp4DownWholeP40KernelSymbolIdentity;
  resources->device_ordinal = device;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87BulkV2NvFp4DownWholeP40DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->cooperative_grid_capacity =
      active_blocks * properties.multiProcessorCount;
  if (code_evidence != nullptr) {
    resources->code = *code_evidence;
  }
  resources->kernel_compiled = true;
  resources->cooperative_launch_supported = true;
  resources->dynamic_shared_attribute_configured = true;
  // Resources and CodeEvidence are public value records, not admission
  // capabilities.  Even structurally valid caller input cannot promote this
  // default-off candidate; retained hash authentication belongs to a future
  // private startup owner.
  resources->resource_gate_passed = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_cuda(
    const Sm87BulkV2NvFp4DownWholeP40Arguments& arguments) noexcept {
  if (arguments.transaction_epoch == 0U || arguments.h == nullptr ||
      arguments.residual == nullptr || arguments.device_control == nullptr ||
      arguments.cuda_stream == nullptr ||
      !pointer_aligned(arguments.h, alignof(uint4)) ||
      !pointer_aligned(arguments.residual, alignof(std::uint32_t)) ||
      !pointer_aligned(arguments.device_control,
                       alignof(Sm87BulkV2NvFp4DownWholeP40DeviceControl)) ||
      (arguments.cancellation_signal != nullptr &&
       !pointer_aligned(arguments.cancellation_signal,
                        alignof(std::uint32_t))) ||
      arguments.down_asset.payload.role !=
          Sm87TargetAotProjectionRole::kNvFp4Down ||
      !sm87_target_aot_nvfp4_cuda_asset_valid(arguments.down_asset)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto h_range = sm87_target_aot_nvfp4_cuda_byte_range(
      arguments.h, kSm87BulkV2NvFp4DownWholeP40HBytes);
  const auto residual_range = sm87_target_aot_nvfp4_cuda_byte_range(
      arguments.residual,
      kSm87BulkV2NvFp4DownWholeP40ResidualBytes);
  const Sm87TargetAotNvFp4CudaByteRange payload_range{
      arguments.down_asset.payload.begin,
      arguments.down_asset.payload.end,
      arguments.down_asset.payload.valid};
  const auto control_range = sm87_target_aot_nvfp4_cuda_byte_range(
      arguments.device_control,
      sizeof(Sm87BulkV2NvFp4DownWholeP40DeviceControl));
  const auto cancellation_range =
      arguments.cancellation_signal == nullptr
          ? Sm87TargetAotNvFp4CudaByteRange{}
          : sm87_target_aot_nvfp4_cuda_byte_range(
                arguments.cancellation_signal, sizeof(std::uint32_t));
  if (!h_range.valid || !residual_range.valid || !payload_range.valid ||
      !control_range.valid ||
      (arguments.cancellation_signal != nullptr &&
       !cancellation_range.valid) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(h_range, residual_range) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(h_range, payload_range) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(h_range, control_range) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(residual_range,
                                                payload_range) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(residual_range,
                                                control_range) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(payload_range,
                                                control_range) ||
      (arguments.cancellation_signal != nullptr &&
       (sm87_target_aot_nvfp4_cuda_ranges_overlap(h_range,
                                                  cancellation_range) ||
        sm87_target_aot_nvfp4_cuda_ranges_overlap(residual_range,
                                                  cancellation_range) ||
        sm87_target_aot_nvfp4_cuda_ranges_overlap(payload_range,
                                                  cancellation_range) ||
        sm87_target_aot_nvfp4_cuda_ranges_overlap(control_range,
                                                  cancellation_range)))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  float tensor_scale = 0.0F;
  std::memcpy(&tensor_scale, &arguments.down_asset.tensor_scale_bits[0U],
              sizeof(tensor_scale));
  if (!finite_positive(tensor_scale)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail::RawArguments raw;
  raw.transaction_epoch = arguments.transaction_epoch;
  raw.h = arguments.h;
  raw.down_payload = reinterpret_cast<const std::uint8_t*>(
      arguments.down_asset.payload.begin);
  raw.tensor_scale = tensor_scale;
  raw.residual = arguments.residual;
  raw.device_control = arguments.device_control;
  raw.cancellation_signal = arguments.cancellation_signal;
  raw.m_tiles = kSm87BulkV2NvFp4DownWholeP40MTiles;
  raw.n_tiles = kSm87BulkV2NvFp4DownWholeP40NTiles;
  raw.cuda_stream = arguments.cuda_stream;
  return static_cast<int>(launch_raw_unchecked(raw));
}

namespace sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail {

int launch_raw(const RawArguments& arguments) noexcept {
  if (!raw_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(launch_raw_unchecked(arguments));
}

}  // namespace sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail

}  // namespace q3x::kernels
