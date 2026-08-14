#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_whole_p40.h"

#include "sm87_bulk_dataflow_v2_fp8_whole_p40_oracle_internal.h"

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace q3x::kernels {
namespace {

namespace cg = cooperative_groups;

constexpr unsigned int kThreads = kSm87BulkV2Fp8WholeP40Threads;
constexpr unsigned int kPersistentCtas =
    kSm87BulkV2Fp8WholeP40PersistentCtas;
constexpr unsigned int kTileM = kSm87BulkV2Fp8WholeP40TileM;
constexpr unsigned int kTileN = kSm87BulkV2Fp8WholeP40TileN;
constexpr unsigned int kTileK = kSm87BulkV2Fp8WholeP40TileK;
constexpr unsigned int kStages = kSm87BulkV2Fp8WholeP40PipelineStages;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kM16PanelsPerWarp = 2U;
constexpr unsigned int kN8PanelsPerWarp = 4U;
constexpr unsigned int kK16Panels = 4U;
constexpr unsigned int kActivationVectors = kTileM * kTileK / 8U;
constexpr unsigned int kWeightVectors = kTileN * kTileK / 16U;
constexpr unsigned int kAuthenticatedN256CellBytes = 16'384U;
constexpr unsigned int kAuthenticatedK16Bytes = 4'096U;
constexpr unsigned int kN128K16Bytes = 2'048U;

static_assert(kActivationVectors == 512U && kWeightVectors == 512U);
static_assert(kSm87BulkV2Fp8WholeP40ABytesPerStage == 8'192U);
static_assert(kSm87BulkV2Fp8WholeP40BBytesPerStage == 8'192U);

struct alignas(32) Fp8WholePipeline final {
  uint4 activations[kStages][kActivationVectors];
  uint4 weights[kStages][kWeightVectors];
};

static_assert(sizeof(Fp8WholePipeline) ==
              kSm87BulkV2Fp8WholeP40DynamicSharedBytes);

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

using WarpAccumulator =
    M16N8Accumulator[kM16PanelsPerWarp][kN8PanelsPerWarp];
using DecodedWeightStage = K16N8Weight[kN8PanelsPerWarp];

__device__ __forceinline__ void cp_async_cg_16(
    void* const shared_destination,
    const void* const global_source) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(global_source)
               : "memory");
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      *reinterpret_cast<const uint4*>(global_source);
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

[[nodiscard]] __device__ __forceinline__ unsigned int
activation_swizzled_vector(const unsigned int row,
                            const unsigned int vector) noexcept {
  return row * (kTileK / 8U) + (vector ^ (row & 7U));
}

__device__ __forceinline__ void load_activation_fragment(
    M16K16Activation& fragment,
    const std::uint16_t* const shared_activations,
    const unsigned int m16, const unsigned int k16,
    const unsigned int lane) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      m16 * 16U + lane % 8U + (quadrant & 1U) * 8U;
  const unsigned int column =
      k16 * 16U + (quadrant >> 1U) * 8U;
  const auto* const source =
      shared_activations +
      activation_swizzled_vector(row, column / 8U) * 8U;
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1), "=r"(fragment.x2),
        "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_activations;
  (void)m16;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
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
      : "r"(activation.x0), "r"(activation.x1), "r"(activation.x2),
        "r"(activation.x3), "r"(weight.x0), "r"(weight.x1));
#else
  (void)accumulator;
  (void)activation;
  (void)weight;
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ K16N8Weight
decode_weight_fragment(const std::uint8_t* const shared_weight,
                       const unsigned int k16,
                       const unsigned int local_n8_panel,
                       const unsigned int lane) noexcept {
  // A N128 half contains sixteen N8 fragments per K16 panel.  Payload byte
  // order within each fragment remains the authenticated
  // [lane][K0,K8,K1,K9] order.
  const unsigned int fragment = k16 * 16U + local_n8_panel;
  const std::uint32_t packed = *reinterpret_cast<const std::uint32_t*>(
      shared_weight + fragment * 128U + lane * 4U);
  const std::uint16_t component0 =
      sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
          static_cast<std::uint8_t>(packed));
  const std::uint16_t component1 =
      sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
          static_cast<std::uint8_t>(packed >> 16U));
  const std::uint16_t component2 =
      sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
          static_cast<std::uint8_t>(packed >> 8U));
  const std::uint16_t component3 =
      sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
          static_cast<std::uint8_t>(packed >> 24U));
  return {static_cast<std::uint32_t>(component0) |
              (static_cast<std::uint32_t>(component1) << 16U),
          static_cast<std::uint32_t>(component2) |
              (static_cast<std::uint32_t>(component3) << 16U)};
}

__device__ __forceinline__ void clear_accumulators(
    WarpAccumulator& accumulators) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16PanelsPerWarp; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      accumulators[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kKTiles>
__device__ __forceinline__ void issue_pipeline_stage(
    Fp8WholePipeline* const pipeline, const unsigned int slot,
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int first_m, const std::uint64_t partition_offset,
    const unsigned int partition_n128_tile,
    const unsigned int k_tile) noexcept {
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / (kTileK / 8U);
    const unsigned int vector = vector_index % (kTileK / 8U);
    const auto* const source = reinterpret_cast<const uint4*>(
                                   input +
                                   static_cast<std::size_t>(first_m + row) *
                                       kInputFeatures +
                                   k_tile * kTileK) +
                               vector;
    cp_async_cg_16(
        &pipeline->activations[slot]
                              [activation_swizzled_vector(row, vector)],
        source);
  }
  const unsigned int parent_n256_tile = partition_n128_tile / 2U;
  const unsigned int n128_half = partition_n128_tile & 1U;
  const std::uint64_t cell =
      partition_offset +
      (static_cast<std::uint64_t>(parent_n256_tile) * kKTiles + k_tile) *
          kAuthenticatedN256CellBytes;
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    // An authenticated N256 cell is K16-major.  Each 4-KiB K16 slab holds
    // the two 2-KiB N128 halves; a whole-cell +8-KiB cut would incorrectly
    // transpose the K16 and N-half dimensions.
    const unsigned int k16 = vector_index / 128U;
    const unsigned int local_vector = vector_index % 128U;
    const auto* const source = reinterpret_cast<const uint4*>(
        payload + cell + k16 * kAuthenticatedK16Bytes +
        n128_half * kN128K16Bytes + local_vector * sizeof(uint4));
    cp_async_cg_16(&pipeline->weights[slot][vector_index], source);
  }
  cp_async_commit_group();
}

template <unsigned int kInputFeatures, unsigned int kKTiles>
__device__ __forceinline__ void run_full_k(
    Fp8WholePipeline* const pipeline,
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int first_m, const std::uint64_t partition_offset,
    const unsigned int partition_n128_tile,
    WarpAccumulator& accumulators) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  clear_accumulators(accumulators);

#pragma unroll
  for (unsigned int stage = 0U; stage < kStages; ++stage) {
    issue_pipeline_stage<kInputFeatures, kKTiles>(
        pipeline, stage, input, payload, first_m, partition_offset,
        partition_n128_tile, stage);
  }

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
    const auto* const shared_activations =
        reinterpret_cast<const std::uint16_t*>(
            pipeline->activations[slot]);
    const auto* const shared_weight =
        reinterpret_cast<const std::uint8_t*>(pipeline->weights[slot]);

    DecodedWeightStage decoded[2U];
    const unsigned int first_n8 = warp_n * kN8PanelsPerWarp;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      decoded[0U][n8] = decode_weight_fragment(
          shared_weight, 0U, first_n8 + n8, lane);
    }

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      const unsigned int current = k16 & 1U;
      const unsigned int next = current ^ 1U;
      if (k16 + 1U < kK16Panels) {
#pragma unroll
        for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
          decoded[next][n8] = decode_weight_fragment(
              shared_weight, k16 + 1U, first_n8 + n8, lane);
        }
      }
      M16K16Activation activation[kM16PanelsPerWarp];
#pragma unroll
      for (unsigned int m16 = 0U; m16 < kM16PanelsPerWarp; ++m16) {
        load_activation_fragment(activation[m16], shared_activations,
                                 warp_m * kM16PanelsPerWarp + m16,
                                 k16, lane);
      }
      if (k16 + 1U == kK16Panels) {
        __syncthreads();
        if (k_tile + kStages < kKTiles) {
          issue_pipeline_stage<kInputFeatures, kKTiles>(
              pipeline, slot, input, payload, first_m, partition_offset,
              partition_n128_tile, k_tile + kStages);
        }
      }
#pragma unroll
      for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
#pragma unroll
        for (unsigned int m16 = 0U; m16 < kM16PanelsPerWarp; ++m16) {
          mma_m16n8k16_bf16(accumulators[m16][n8],
                            activation[m16], decoded[current][n8]);
        }
      }
    }
  }
  cp_async_wait_group<0U>();
  __syncthreads();
}

__device__ __forceinline__ void publish_output(
    const WarpAccumulator& accumulators,
    const unsigned int output_features, const unsigned int first_m,
    const unsigned int first_n, const float compensated_scale,
    std::uint16_t* const output) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16PanelsPerWarp; ++m16) {
    const unsigned int local_row0 =
        warp_m * 32U + m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      const unsigned int column =
          first_n + warp_n * 32U + n8 * 8U + lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      const std::uint32_t packed0 =
          static_cast<std::uint32_t>(encode_bf16_rne(
              __fmul_rn(value.x0, compensated_scale))) |
          (static_cast<std::uint32_t>(encode_bf16_rne(
               __fmul_rn(value.x1, compensated_scale)))
           << 16U);
      const std::uint32_t packed1 =
          static_cast<std::uint32_t>(encode_bf16_rne(
              __fmul_rn(value.x2, compensated_scale))) |
          (static_cast<std::uint32_t>(encode_bf16_rne(
               __fmul_rn(value.x3, compensated_scale)))
           << 16U);
      *reinterpret_cast<std::uint32_t*>(
          output + static_cast<std::size_t>(first_m + local_row0) *
                       output_features +
          column) = packed0;
      *reinterpret_cast<std::uint32_t*>(
          output + static_cast<std::size_t>(first_m + local_row1) *
                       output_features +
          column) = packed1;
    }
  }
}

[[nodiscard]] __device__ __forceinline__ bool cta_cancellation_requested(
    const std::uint32_t* const cancellation_signal,
    Sm87BulkV2Fp8WholeP40DeviceControl* const control) noexcept {
  int requested = 0;
  if (threadIdx.x == 0U) {
    requested =
        (cancellation_signal != nullptr &&
         *reinterpret_cast<const volatile std::uint32_t*>(
             cancellation_signal) != 0U) ||
                atomicAdd(&control->cancellation_observed, 0U) != 0U
            ? 1
            : 0;
  }
  // The CTA-wide predicate is produced by one observer and returned
  // uniformly.  No warp can overwrite a shared flag for the next cohort.
  return __syncthreads_or(requested) != 0;
}

__device__ __forceinline__ void initialize_control_single_writer(
    Sm87BulkV2Fp8WholeP40DeviceControl* const control,
    const std::uint64_t transaction_epoch,
    const Sm87TargetAotProjectionRole role,
    const unsigned int expected_cells) noexcept {
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    control->transaction_epoch = transaction_epoch;
    control->expected_cells = expected_cells;
    control->started_cells = 0U;
    control->completed_cells = 0U;
    control->completed_ctas = 0U;
    control->cancellation_observed = 0U;
    control->launch_completed = 0U;
    control->first_incomplete_cohort = 0xffff'ffffU;
    control->role = static_cast<std::uint32_t>(role);
    control->policy = kSm87BulkV2Fp8WholeP40RequiredPolicy;
#pragma unroll
    for (unsigned int index = 0U; index < 4U; ++index) {
      control->reserved[index] = 0U;
    }
  }
}

template <Sm87TargetAotProjectionRole kRole,
          unsigned int kInputFeatures, unsigned int kOutputFeatures,
          unsigned int kKTiles, unsigned int kPartitionCount>
__global__ __launch_bounds__(256, 2)
void sm87_bulk_v2_fp8_whole_p40_kernel(
    const std::uint16_t* const input,
    const std::uint8_t* const payload,
    const float scale0, const float scale1, const float scale2,
    std::uint16_t* const primary_output,
    std::uint16_t* const secondary_output,
    std::uint16_t* const tertiary_output,
    Sm87BulkV2Fp8WholeP40DeviceControl* const control,
    const std::uint32_t* const cancellation_signal,
    const std::uint64_t transaction_epoch,
    const unsigned int m_tiles, const unsigned int n_tiles) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const pipeline =
      reinterpret_cast<Fp8WholePipeline*>(dynamic_storage);
  const unsigned int m_groups =
      (m_tiles + kSm87BulkV2Fp8WholeP40CohortM - 1U) /
      kSm87BulkV2Fp8WholeP40CohortM;
  const unsigned int n_groups =
      (n_tiles + kSm87BulkV2Fp8WholeP40CohortN - 1U) /
      kSm87BulkV2Fp8WholeP40CohortN;
  const unsigned int cohort_count = m_groups * n_groups;
  initialize_control_single_writer(control, transaction_epoch, kRole,
                                   m_tiles * n_tiles);
  cg::this_grid().sync();

  const unsigned int m_lane =
      blockIdx.x / kSm87BulkV2Fp8WholeP40CohortN;
  const unsigned int n_lane =
      blockIdx.x % kSm87BulkV2Fp8WholeP40CohortN;
  unsigned int local_completed = 0U;
  bool cancelled = false;

  for (unsigned int cohort = 0U; cohort < cohort_count; ++cohort) {
    if (cta_cancellation_requested(cancellation_signal, control)) {
      if (threadIdx.x == 0U) {
        atomicExch(&control->cancellation_observed, 1U);
        atomicMin(&control->first_incomplete_cohort, cohort);
      }
      cancelled = true;
      break;
    }
    const unsigned int m_group = cohort / n_groups;
    const unsigned int n_epoch = cohort % n_groups;
    const unsigned int n_group =
        (m_group & 1U) == 0U ? n_epoch : n_groups - 1U - n_epoch;
    const unsigned int m_tile =
        m_group * kSm87BulkV2Fp8WholeP40CohortM + m_lane;
    const unsigned int n_tile =
        n_group * kSm87BulkV2Fp8WholeP40CohortN + n_lane;
    // P40's final M cohort has only one useful lane.  Invalid M or N lanes
    // perform no global load, MMA, or store.
    if (m_tile >= m_tiles || n_tile >= n_tiles) {
      continue;
    }

    unsigned int partition = 0U;
    unsigned int partition_first_n128 = 0U;
    std::uint64_t partition_offset = 0U;
    float compensated_scale = scale0;
    if constexpr (kRole == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
      if (n_tile >= 80U) {
        partition = 1U;
        partition_first_n128 = 80U;
        partition_offset = 52'428'800U;
        compensated_scale = scale1;
      }
    } else if constexpr (kRole ==
                         Sm87TargetAotProjectionRole::kFp8FullQkv) {
      if (n_tile >= 104U) {
        partition = 2U;
        partition_first_n128 = 104U;
        partition_offset = 68'157'440U;
        compensated_scale = scale2;
      } else if (n_tile >= 96U) {
        partition = 1U;
        partition_first_n128 = 96U;
        partition_offset = 62'914'560U;
        compensated_scale = scale1;
      }
    }
    if (partition >= kPartitionCount) {
      continue;
    }

    if (threadIdx.x == 0U) {
      atomicAdd(&control->started_cells, 1U);
    }
    WarpAccumulator accumulators;
    run_full_k<kInputFeatures, kKTiles>(
        pipeline, input, payload, m_tile * kTileM, partition_offset,
        n_tile - partition_first_n128, accumulators);

    // Cancellation after a complete full-K tile suppresses every output and
    // receipt publication for that tile.  The single-thread observation plus
    // __syncthreads_or keeps the CTA uniform before the publication branch.
    if (cta_cancellation_requested(cancellation_signal, control)) {
      if (threadIdx.x == 0U) {
        atomicExch(&control->cancellation_observed, 1U);
        atomicMin(&control->first_incomplete_cohort, cohort);
      }
      cancelled = true;
      break;
    }

    std::uint16_t* destination = primary_output;
    unsigned int destination_features = kOutputFeatures;
    unsigned int destination_first_n = n_tile * kTileN;
    if constexpr (kRole == Sm87TargetAotProjectionRole::kFp8FullQkv) {
      destination = partition == 0U
                        ? primary_output
                        : (partition == 1U ? secondary_output
                                           : tertiary_output);
      destination_features = partition == 0U ? 12'288U : 1'024U;
      destination_first_n =
          (n_tile - partition_first_n128) * kTileN;
    }
    publish_output(accumulators, destination_features, m_tile * kTileM,
                   destination_first_n, compensated_scale, destination);
    ++local_completed;
  }

  if (threadIdx.x == 0U) {
    if (cancellation_signal != nullptr &&
        *reinterpret_cast<const volatile std::uint32_t*>(
            cancellation_signal) != 0U) {
      atomicExch(&control->cancellation_observed, 1U);
      cancelled = true;
    }
    atomicAdd(&control->completed_cells, local_completed);
    atomicAdd(&control->completed_ctas, 1U);
    if (cancelled) {
      atomicExch(&control->cancellation_observed, 1U);
    }
  }
  cg::this_grid().sync();
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    control->launch_completed =
        control->transaction_epoch == transaction_epoch &&
                control->role == static_cast<std::uint32_t>(kRole) &&
                control->policy == kSm87BulkV2Fp8WholeP40RequiredPolicy &&
                control->cancellation_observed == 0U &&
                control->completed_ctas == kPersistentCtas &&
                control->started_cells == control->expected_cells &&
                control->completed_cells == control->expected_cells
            ? 1U
            : 0U;
  }
}

template <typename Kernel>
[[nodiscard]] cudaError_t set_dynamic_shared(Kernel kernel) noexcept {
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87BulkV2Fp8WholeP40DynamicSharedBytes));
}

[[nodiscard]] cudaError_t validate_sm87_device(
    cudaDeviceProp* const properties) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(properties, device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     static_cast<int>(kSm87BulkV2Fp8WholeP40SmCount) &&
                 properties->cooperativeLaunch != 0
             ? cudaSuccess
             : cudaErrorNotSupported;
}

template <typename Kernel>
[[nodiscard]] cudaError_t query_kernel_resources(
    Kernel kernel, const Sm87TargetAotProjectionRole role,
    const Sm87BulkV2Fp8WholeP40CodeEvidence& code,
    const cudaDeviceProp& properties,
    Sm87BulkV2Fp8WholeP40KernelResources* const resources) noexcept {
  cudaError_t status = set_dynamic_shared(kernel);
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, kThreads,
      kSm87BulkV2Fp8WholeP40DynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  resources->role = role;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87BulkV2Fp8WholeP40DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->cooperative_grid_capacity =
      active_blocks * properties.multiProcessorCount;
  resources->code = code;
  resources->cooperative_launch_supported = true;
  resources->runtime_envelope_observed =
      resources->binary_version == 87 &&
      resources->registers_per_thread > 0 &&
      resources->registers_per_thread <=
          static_cast<int>(kSm87BulkV2Fp8WholeP40MaximumRegisters) &&
      resources->static_shared_bytes == 0U &&
      resources->local_bytes == 0U &&
      resources->maximum_threads_per_block >= static_cast<int>(kThreads) &&
      resources->active_blocks_per_sm >= 2 &&
      resources->cooperative_grid_capacity >= 32;
  resources->external_static_record_consistent =
      sm87_bulk_v2_fp8_whole_p40_code_evidence_valid(resources->code);
  // The caller-provided record is observation input, never a launch
  // capability.  Only a future private owner may bind hashes to the loaded
  // ELF and issue such authority.
  resources->admission_capability_issued = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return cudaSuccess;
}

[[nodiscard]] bool output_shape_valid(
    const Sm87TargetAotProjectionRole role,
    const std::uint16_t* const primary,
    const std::uint16_t* const secondary,
    const std::uint16_t* const tertiary) noexcept {
  return sm87_bulk_v2_fp8_whole_p40_output_shape_valid(
      role, primary, secondary, tertiary);
}

[[nodiscard]] float bf16_to_float(const std::uint16_t bits) noexcept {
  const std::uint32_t bits32 = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits32, sizeof(result));
  return result;
}

[[nodiscard]] bool scale_valid(const float scale) noexcept {
  return std::isfinite(scale) && scale > 0.0F;
}

template <typename Kernel>
[[nodiscard]] cudaError_t launch_cooperative(
    Kernel kernel,
    const sm87_bulk_v2_fp8_whole_p40_oracle_detail::RawArguments& arguments,
    const std::array<float, 3U>& scales) noexcept {
  const std::uint16_t* input_argument = arguments.input;
  const std::uint8_t* payload_argument = arguments.payload;
  float scale0_argument = scales[0U];
  float scale1_argument = scales[1U];
  float scale2_argument = scales[2U];
  std::uint16_t* primary_argument = arguments.primary_output;
  std::uint16_t* secondary_argument = arguments.secondary_output;
  std::uint16_t* tertiary_argument = arguments.tertiary_output;
  Sm87BulkV2Fp8WholeP40DeviceControl* control_argument =
      arguments.device_control;
  const std::uint32_t* cancellation_argument =
      arguments.cancellation_signal;
  std::uint64_t epoch_argument = arguments.transaction_epoch;
  unsigned int m_tiles_argument = arguments.m_tiles;
  unsigned int n_tiles_argument = arguments.n_tiles;
  void* kernel_arguments[] = {
      &input_argument,       &payload_argument,
      &scale0_argument,      &scale1_argument,
      &scale2_argument,      &primary_argument,
      &secondary_argument,   &tertiary_argument,
      &control_argument,     &cancellation_argument,
      &epoch_argument,       &m_tiles_argument,
      &n_tiles_argument,
  };
  return cudaLaunchCooperativeKernel(
      kernel, dim3{kPersistentCtas}, dim3{kThreads}, kernel_arguments,
      kSm87BulkV2Fp8WholeP40DynamicSharedBytes,
      reinterpret_cast<cudaStream_t>(arguments.cuda_stream));
}

[[nodiscard]] cudaError_t launch_raw_unchecked(
    const sm87_bulk_v2_fp8_whole_p40_oracle_detail::RawArguments& arguments,
    const std::array<float, 3U>& scales) noexcept {
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    return launch_cooperative(
        sm87_bulk_v2_fp8_whole_p40_kernel<
            Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
            5'120U, 16'384U, 80U, 2U>,
        arguments, scales);
  }
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8FullQkv) {
    return launch_cooperative(
        sm87_bulk_v2_fp8_whole_p40_kernel<
            Sm87TargetAotProjectionRole::kFp8FullQkv,
            5'120U, 14'336U, 80U, 3U>,
        arguments, scales);
  }
  return launch_cooperative(
      sm87_bulk_v2_fp8_whole_p40_kernel<
          Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          6'144U, 5'120U, 96U, 1U>,
      arguments, scales);
}

}  // namespace

int query_sm87_bulk_dataflow_v2_fp8_whole_p40_resources_cuda(
    const std::array<Sm87BulkV2Fp8WholeP40CodeEvidence, 3U>* const
        code_evidence,
    Sm87BulkV2Fp8WholeP40FamilyResources* const resources) noexcept {
  if (code_evidence == nullptr || resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  cudaDeviceProp properties{};
  cudaError_t status = validate_sm87_device(&properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = query_kernel_resources(
      sm87_bulk_v2_fp8_whole_p40_kernel<
          Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
          5'120U, 16'384U, 80U, 2U>,
      Sm87TargetAotProjectionRole::kFp8GdnQkvZ, (*code_evidence)[0U],
      properties, &resources->roles[0U]);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = query_kernel_resources(
      sm87_bulk_v2_fp8_whole_p40_kernel<
          Sm87TargetAotProjectionRole::kFp8FullQkv,
          5'120U, 14'336U, 80U, 3U>,
      Sm87TargetAotProjectionRole::kFp8FullQkv, (*code_evidence)[1U],
      properties, &resources->roles[1U]);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = query_kernel_resources(
      sm87_bulk_v2_fp8_whole_p40_kernel<
          Sm87TargetAotProjectionRole::kFp8AttentionOutput,
          6'144U, 5'120U, 96U, 1U>,
      Sm87TargetAotProjectionRole::kFp8AttentionOutput,
      (*code_evidence)[2U], properties, &resources->roles[2U]);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->all_runtime_envelopes_observed =
      resources->roles[0U].runtime_envelope_observed &&
      resources->roles[1U].runtime_envelope_observed &&
      resources->roles[2U].runtime_envelope_observed;
  resources->all_external_static_records_consistent =
      resources->roles[0U].external_static_record_consistent &&
      resources->roles[1U].external_static_record_consistent &&
      resources->roles[2U].external_static_record_consistent;
  resources->admission_capability_issued = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_bulk_dataflow_v2_fp8_whole_p40_cuda(
    const Sm87BulkV2Fp8WholeP40Arguments& arguments) noexcept {
  const auto plan =
      sm87_bulk_v2_fp8_whole_p40_role_plan(arguments.role);
  if (!plan.valid || arguments.transaction_epoch == 0U ||
      arguments.input == nullptr || arguments.device_control == nullptr ||
      arguments.cuda_stream == nullptr ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(arguments.primary_output) % 16U !=
          0U ||
      (arguments.secondary_output != nullptr &&
       reinterpret_cast<std::uintptr_t>(arguments.secondary_output) % 16U !=
           0U) ||
      (arguments.tertiary_output != nullptr &&
       reinterpret_cast<std::uintptr_t>(arguments.tertiary_output) % 16U !=
           0U) ||
      reinterpret_cast<std::uintptr_t>(arguments.device_control) % 64U !=
          0U ||
      (arguments.cancellation_signal != nullptr &&
       reinterpret_cast<std::uintptr_t>(arguments.cancellation_signal) % 4U !=
           0U) ||
      !output_shape_valid(arguments.role, arguments.primary_output,
                          arguments.secondary_output,
                          arguments.tertiary_output) ||
      !sm87_target_aot_fp8_cuda_asset_valid(
          arguments.authenticated_asset) ||
      arguments.authenticated_asset.payload.role != arguments.role ||
      !sm87_bulk_v2_fp8_whole_p40_ranges_valid(
          arguments.role, arguments.input,
          arguments.authenticated_asset.payload.begin,
          arguments.authenticated_asset.payload.end,
          arguments.primary_output, arguments.secondary_output,
          arguments.tertiary_output, arguments.device_control,
          arguments.cancellation_signal)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  std::array<float, 3U> scales{};
  for (std::size_t partition = 0U; partition < plan.partition_count;
       ++partition) {
    scales[partition] = bf16_to_float(
        arguments.authenticated_asset.compensated_tensor_scale_bf16_bits[
            partition]);
    if (!scale_valid(scales[partition])) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  sm87_bulk_v2_fp8_whole_p40_oracle_detail::RawArguments raw;
  raw.transaction_epoch = arguments.transaction_epoch;
  raw.role = arguments.role;
  raw.input = arguments.input;
  raw.payload = reinterpret_cast<const std::uint8_t*>(
      arguments.authenticated_asset.payload.begin);
  raw.compensated_scale_bf16_bits =
      arguments.authenticated_asset.compensated_tensor_scale_bf16_bits;
  raw.primary_output = arguments.primary_output;
  raw.secondary_output = arguments.secondary_output;
  raw.tertiary_output = arguments.tertiary_output;
  raw.device_control = arguments.device_control;
  raw.cancellation_signal = arguments.cancellation_signal;
  raw.m_tiles = kSm87BulkV2Fp8WholeP40MTiles;
  raw.n_tiles = plan.n_tiles;
  raw.cuda_stream = arguments.cuda_stream;
  // This prevalidated seam performs exactly one launch and no CUDA query.
  return static_cast<int>(launch_raw_unchecked(raw, scales));
}

namespace sm87_bulk_v2_fp8_whole_p40_oracle_detail {

int launch_raw(const RawArguments& arguments) noexcept {
  const auto plan = sm87_bulk_v2_fp8_whole_p40_role_plan(arguments.role);
  if (!plan.valid || arguments.transaction_epoch == 0U ||
      arguments.input == nullptr || arguments.payload == nullptr ||
      arguments.primary_output == nullptr ||
      arguments.device_control == nullptr || arguments.cuda_stream == nullptr ||
      arguments.m_tiles == 0U ||
      arguments.m_tiles > kSm87BulkV2Fp8WholeP40MTiles ||
      arguments.n_tiles == 0U || arguments.n_tiles > plan.n_tiles ||
      !output_shape_valid(arguments.role, arguments.primary_output,
                          arguments.secondary_output,
                          arguments.tertiary_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  std::array<float, 3U> scales{};
  for (std::size_t partition = 0U; partition < plan.partition_count;
       ++partition) {
    scales[partition] =
        bf16_to_float(arguments.compensated_scale_bf16_bits[partition]);
    if (!scale_valid(scales[partition])) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  cudaDeviceProp properties{};
  cudaError_t status = validate_sm87_device(&properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (arguments.role == Sm87TargetAotProjectionRole::kFp8GdnQkvZ) {
    status = set_dynamic_shared(
        sm87_bulk_v2_fp8_whole_p40_kernel<
            Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
            5'120U, 16'384U, 80U, 2U>);
  } else if (arguments.role ==
             Sm87TargetAotProjectionRole::kFp8FullQkv) {
    status = set_dynamic_shared(
        sm87_bulk_v2_fp8_whole_p40_kernel<
            Sm87TargetAotProjectionRole::kFp8FullQkv,
            5'120U, 14'336U, 80U, 3U>);
  } else {
    status = set_dynamic_shared(
        sm87_bulk_v2_fp8_whole_p40_kernel<
            Sm87TargetAotProjectionRole::kFp8AttentionOutput,
            6'144U, 5'120U, 96U, 1U>);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  (void)cudaGetLastError();
  return static_cast<int>(launch_raw_unchecked(arguments, scales));
}

}  // namespace sm87_bulk_v2_fp8_whole_p40_oracle_detail

}  // namespace q3x::kernels
