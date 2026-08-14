#include "q3x/kernels/sm87_macrofeed_v4_nvfp4_gate_up.h"

#include "sm87_macrofeed_v3_nvfp4_decode.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87MacroFeedV4NvFp4GateUpThreads);
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87MacroFeedV4NvFp4GateUpBlockM);
constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87MacroFeedV4NvFp4GateUpBlockN);
constexpr unsigned int kTileK =
    static_cast<unsigned int>(kSm87MacroFeedV4NvFp4GateUpBlockK);
constexpr unsigned int kStages =
    static_cast<unsigned int>(kSm87MacroFeedV4NvFp4GateUpPipelineStages);
constexpr unsigned int kPersistentCtas =
    static_cast<unsigned int>(kSm87MacroFeedV4NvFp4GateUpPersistentCtas);
constexpr unsigned int kM16Panels = 4U;
constexpr unsigned int kN8PanelsPerWarp = 4U;
constexpr unsigned int kN8PanelsPerN64 = 8U;
constexpr unsigned int kK16Panels = 4U;
constexpr unsigned int kActivationVectors = kTileM * kTileK / 8U;
constexpr unsigned int kBranchWeightVectors =
    kSm87MacroFeedV4NvFp4GateUpBranchWeightBytes / sizeof(uint4);
constexpr unsigned int kBranchScaleVectors =
    kSm87MacroFeedV4NvFp4GateUpBranchScaleBytes / sizeof(uint4);

static_assert(kThreads == 8U * kWarpSize);
static_assert(kTileM == 64U && kTileN == 128U && kTileK == 64U);
static_assert(kStages == 2U && kPersistentCtas == 32U);
static_assert(kActivationVectors == 512U);
static_assert(kBranchWeightVectors == 256U);
static_assert(kBranchScaleVectors == 32U);

struct alignas(32) PipelineStorage final {
  uint4 activations[kStages][kTileM][kTileK / 8U];
  uint4 weights[kStages][2U][kBranchWeightVectors];
  uint4 scales[kStages][2U][kBranchScaleVectors];
};

static_assert(sizeof(PipelineStorage) ==
              kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes);
static_assert(2U * kTileM * kTileN * sizeof(std::uint16_t) <=
              sizeof(PipelineStorage));

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
    M16N8Accumulator[kM16Panels][kN8PanelsPerWarp];
using DecodedWeightStage = K16N8Weight[kN8PanelsPerWarp];

template <bool kPredicate>
__device__ __forceinline__ void cp_async_cg_16(
    void* const shared_destination, const void* const global_source,
    const bool valid = true) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared_destination));
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

__device__ __forceinline__ void load_activation_fragment(
    M16K16Activation& fragment,
    const std::uint16_t* const shared_activations,
    const unsigned int m16, const unsigned int k16,
    const unsigned int lane) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      m16 * 16U + lane % 8U + (quadrant & 1U) * 8U;
  const unsigned int column = k16 * 16U + (quadrant >> 1U) * 8U;
  const auto* const source = shared_activations + row * kTileK + column;
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

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ K16N8Weight
decode_weight_fragment(const std::uint8_t* const shared_weight,
                       const std::uint8_t* const shared_scale,
                       const unsigned int k16,
                       const unsigned int local_n_warp,
                       const unsigned int n8_panel,
                       const unsigned int lane) noexcept {
  const unsigned int fragment =
      (k16 * 2U + local_n_warp) * kN8PanelsPerN64 + n8_panel;
  const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(
      shared_weight + fragment * 64U + lane * 2U);
  const std::uint8_t scale_code =
      shared_scale[fragment * 8U + lane / 4U];
  const uint2 decoded =
      macrofeed_v3_nvfp4_detail::decode_nvfp4x4_to_bf16x4(packed,
                                                            scale_code);
  return {decoded.x, decoded.y};
}

__device__ __forceinline__ void clear_accumulators(
    WarpAccumulator& accumulators) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      accumulators[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kKTiles,
          std::uint64_t kPartitionBytes>
__device__ __forceinline__ void issue_pipeline_stage(
    PipelineStorage* const storage, const unsigned int slot,
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int rows, const unsigned int first_m,
    const unsigned int n128_tile, const unsigned int k_tile) noexcept {
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / (kTileK / 8U);
    const unsigned int vector = vector_index % (kTileK / 8U);
    const bool row_valid = first_m + row < rows;
    const unsigned int source_row = row_valid ? first_m + row : 0U;
    const auto* const source = reinterpret_cast<const uint4*>(
                                   input +
                                   static_cast<std::size_t>(source_row) *
                                       kInputFeatures +
                                   k_tile * kTileK) +
                               vector;
    cp_async_cg_16<true>(&storage->activations[slot][row][vector], source,
                         row_valid);
  }

  const unsigned int canonical_n_tile = n128_tile / 2U;
  const unsigned int n_half = n128_tile % 2U;
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int combined = threadIdx.x + pass * kThreads;
    const unsigned int branch = combined / kBranchWeightVectors;
    const unsigned int local_vector = combined % kBranchWeightVectors;
    const unsigned int vector_in_fragment = local_vector & 3U;
    const unsigned int fragment = local_vector >> 2U;
    const unsigned int n8_panel = fragment % kN8PanelsPerN64;
    const unsigned int outer = fragment / kN8PanelsPerN64;
    const unsigned int local_n_warp = outer & 1U;
    const unsigned int k16 = outer >> 1U;
    const unsigned int global_n_warp = n_half * 2U + local_n_warp;
    const std::uint64_t cell =
        static_cast<std::uint64_t>(branch) * kPartitionBytes +
        (static_cast<std::uint64_t>(canonical_n_tile) * kKTiles + k_tile) *
            kSm87MacroFeedV3NvFp4GateUpCellBytes;
    const std::uint64_t global_fragment =
        (static_cast<std::uint64_t>(k16) * 4U + global_n_warp) *
            kN8PanelsPerN64 +
        n8_panel;
    const auto* const source = reinterpret_cast<const uint4*>(
        payload + cell + global_fragment * 64U) + vector_in_fragment;
    cp_async_cg_16<false>(
        &storage->weights[slot][branch][local_vector], source);
  }

  if (threadIdx.x < 2U * kBranchScaleVectors) {
    const unsigned int combined = threadIdx.x;
    const unsigned int branch = combined / kBranchScaleVectors;
    const unsigned int local_vector = combined % kBranchScaleVectors;
    const unsigned int n8_pair = local_vector & 3U;
    const unsigned int outer = local_vector >> 2U;
    const unsigned int local_n_warp = outer & 1U;
    const unsigned int k16 = outer >> 1U;
    const unsigned int global_n_warp = n_half * 2U + local_n_warp;
    const std::uint64_t cell =
        static_cast<std::uint64_t>(branch) * kPartitionBytes +
        (static_cast<std::uint64_t>(canonical_n_tile) * kKTiles + k_tile) *
            kSm87MacroFeedV3NvFp4GateUpCellBytes;
    const std::uint64_t scale_byte =
        (static_cast<std::uint64_t>(k16) * 4U + global_n_warp) * 64U +
        n8_pair * 16U;
    const auto* const source = reinterpret_cast<const uint4*>(
        payload + cell +
        kSm87MacroFeedV3NvFp4GateUpWeightBytesPerCell + scale_byte);
    cp_async_cg_16<false>(
        &storage->scales[slot][branch][local_vector], source);
  }
  cp_async_commit_group();
}

template <unsigned int kInputFeatures, unsigned int kKTiles,
          std::uint64_t kPartitionBytes>
__device__ __forceinline__ void run_full_k(
    PipelineStorage* const storage, const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int rows,
    const unsigned int first_m, const unsigned int n128_tile,
    WarpAccumulator& accumulators) noexcept {
  static_assert(kKTiles >= kStages);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int branch = warp / 4U;
  const unsigned int branch_quarter = warp % 4U;
  const unsigned int local_n_warp = branch_quarter / 2U;
  const unsigned int first_n8 =
      (branch_quarter % 2U) * kN8PanelsPerWarp;

  clear_accumulators(accumulators);
  issue_pipeline_stage<kInputFeatures, kKTiles, kPartitionBytes>(
      storage, 0U, input, payload, rows, first_m, n128_tile, 0U);
  issue_pipeline_stage<kInputFeatures, kKTiles, kPartitionBytes>(
      storage, 1U, input, payload, rows, first_m, n128_tile, 1U);

#pragma unroll 1
  for (unsigned int k_tile = 0U; k_tile < kKTiles; ++k_tile) {
    if (k_tile + 1U < kKTiles) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();

    const unsigned int slot = k_tile % kStages;
    const auto* const shared_activations =
        reinterpret_cast<const std::uint16_t*>(storage->activations[slot]);
    const auto* const shared_weight = reinterpret_cast<const std::uint8_t*>(
        storage->weights[slot][branch]);
    const auto* const shared_scale = reinterpret_cast<const std::uint8_t*>(
        storage->scales[slot][branch]);

    DecodedWeightStage decoded[2U];
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      decoded[0U][n8] = decode_weight_fragment(
          shared_weight, shared_scale, 0U, local_n_warp,
          first_n8 + n8, lane);
    }

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      const unsigned int current = k16 & 1U;
      const unsigned int next = current ^ 1U;
      if (k16 + 1U < kK16Panels) {
#pragma unroll
        for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
          decoded[next][n8] = decode_weight_fragment(
              shared_weight, shared_scale, k16 + 1U, local_n_warp,
              first_n8 + n8, lane);
        }
      }

      M16K16Activation activation[kM16Panels];
#pragma unroll
      for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
        load_activation_fragment(activation[m16], shared_activations, m16,
                                 k16, lane);
      }

      if (k16 + 1U == kK16Panels) {
        __syncthreads();
        if (k_tile + kStages < kKTiles) {
          issue_pipeline_stage<kInputFeatures, kKTiles, kPartitionBytes>(
              storage, slot, input, payload, rows, first_m, n128_tile,
              k_tile + kStages);
        }
      }

#pragma unroll
      for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
#pragma unroll
        for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
          mma_m16n8k16_bf16(accumulators[m16][n8], activation[m16],
                            decoded[current][n8]);
        }
      }
    }
  }
  cp_async_wait_group<0U>();
  __syncthreads();
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
pack_scaled_bf16_pair(const float low, const float high,
                      const float tensor_scale) noexcept {
  return static_cast<std::uint32_t>(encode_bf16_rne(low * tensor_scale)) |
         (static_cast<std::uint32_t>(encode_bf16_rne(high * tensor_scale))
          << 16U);
}

__device__ __forceinline__ void publish_branch_to_shared(
    std::uint16_t* const temporary, const WarpAccumulator& accumulators,
    const unsigned int branch, const unsigned int branch_quarter,
    const unsigned int lane, const float tensor_scale) noexcept {
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
    const unsigned int row0 = m16 * 16U + lane_group;
    const unsigned int row1 = row0 + 8U;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      const unsigned int column =
          branch_quarter * 32U + n8 * 8U + lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      *reinterpret_cast<std::uint32_t*>(
          temporary + branch * kTileM * kTileN + row0 * kTileN + column) =
          pack_scaled_bf16_pair(value.x0, value.x1, tensor_scale);
      *reinterpret_cast<std::uint32_t*>(
          temporary + branch * kTileM * kTileN + row1 * kTileN + column) =
          pack_scaled_bf16_pair(value.x2, value.x3, tensor_scale);
    }
  }
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t silu_times_up(
    const std::uint16_t gate_bits, const std::uint16_t up_bits) noexcept {
  const float gate = decode_bf16(gate_bits);
  const float up = decode_bf16(up_bits);
  return encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
}

template <unsigned int kOutputFeatures>
__device__ __forceinline__ void consume_gate_up_shared(
    const std::uint16_t* const temporary, const unsigned int rows,
    const unsigned int first_m, const unsigned int first_n,
    const unsigned int warp, const unsigned int lane,
    std::uint16_t* const output) noexcept {
  if (warp >= 4U) {
    return;
  }
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
    const unsigned int local_row0 = m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = first_m + local_row0;
    const unsigned int global_row1 = first_m + local_row1;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8PanelsPerWarp; ++n8) {
      const unsigned int local_column =
          warp * 32U + n8 * 8U + lane_in_group * 2U;
      const unsigned int global_column = first_n + local_column;
      const std::uint32_t gate0 = *reinterpret_cast<const std::uint32_t*>(
          temporary + local_row0 * kTileN + local_column);
      const std::uint32_t up0 = *reinterpret_cast<const std::uint32_t*>(
          temporary + kTileM * kTileN + local_row0 * kTileN +
          local_column);
      if (global_row0 < rows) {
        const std::uint32_t result =
            static_cast<std::uint32_t>(silu_times_up(
                static_cast<std::uint16_t>(gate0),
                static_cast<std::uint16_t>(up0))) |
            (static_cast<std::uint32_t>(silu_times_up(
                 static_cast<std::uint16_t>(gate0 >> 16U),
                 static_cast<std::uint16_t>(up0 >> 16U)))
             << 16U);
        *reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row0) *
                         kOutputFeatures +
            global_column) = result;
      }
      const std::uint32_t gate1 = *reinterpret_cast<const std::uint32_t*>(
          temporary + local_row1 * kTileN + local_column);
      const std::uint32_t up1 = *reinterpret_cast<const std::uint32_t*>(
          temporary + kTileM * kTileN + local_row1 * kTileN +
          local_column);
      if (global_row1 < rows) {
        const std::uint32_t result =
            static_cast<std::uint32_t>(silu_times_up(
                static_cast<std::uint16_t>(gate1),
                static_cast<std::uint16_t>(up1))) |
            (static_cast<std::uint32_t>(silu_times_up(
                 static_cast<std::uint16_t>(gate1 >> 16U),
                 static_cast<std::uint16_t>(up1 >> 16U)))
             << 16U);
        *reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row1) *
                         kOutputFeatures +
            global_column) = result;
      }
    }
  }
}

__device__ __forceinline__ void projection_task(
    const unsigned int linear_task, unsigned int& m_tile,
    unsigned int& n_tile) noexcept {
  // N-major bounded scheduling keeps a canonical packed B/scale cell in the
  // same 32-CTA cohort while the cohort advances over independent M64 rows.
  m_tile = linear_task % kSm87MacroFeedV4NvFp4GateUpGridM;
  n_tile = linear_task / kSm87MacroFeedV4NvFp4GateUpGridM;
}

__global__ __launch_bounds__(kThreads, 2)
void sm87_macrofeed_v4_nvfp4_gate_up_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const float gate_tensor_scale,
    const float up_tensor_scale, std::uint16_t* __restrict__ output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage = reinterpret_cast<PipelineStorage*>(dynamic_storage);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int branch = warp / 4U;
  const unsigned int branch_quarter = warp % 4U;

  for (unsigned int linear_task = blockIdx.x;
       linear_task < kSm87MacroFeedV4NvFp4GateUpLogicalTasks;
       linear_task += kPersistentCtas) {
    unsigned int m_tile = 0U;
    unsigned int n_tile = 0U;
    projection_task(linear_task, m_tile, n_tile);
    WarpAccumulator accumulators;
    run_full_k<kSm87MacroFeedV4NvFp4GateUpInputFeatures,
               kSm87MacroFeedV4NvFp4GateUpKTiles,
               kSm87MacroFeedV4NvFp4GateUpPartitionBytes>(
        storage, input, payload, kSm87MacroFeedV4NvFp4GateUpTokens,
        m_tile * kTileM, n_tile, accumulators);

    auto* const temporary =
        reinterpret_cast<std::uint16_t*>(dynamic_storage);
    publish_branch_to_shared(
        temporary, accumulators, branch, branch_quarter, lane,
        branch == 0U ? gate_tensor_scale : up_tensor_scale);
    __syncthreads();
    consume_gate_up_shared<kSm87MacroFeedV4NvFp4GateUpOutputFeatures>(
        temporary, kSm87MacroFeedV4NvFp4GateUpTokens, m_tile * kTileM,
        n_tile * kTileN, warp, lane, output);
    __syncthreads();
  }
}

__global__ __launch_bounds__(kThreads, 2)
void sm87_macrofeed_v4_nvfp4_gate_up_tile_test_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const unsigned int valid_rows,
    const unsigned int canonical_n_half,
    const float gate_tensor_scale, const float up_tensor_scale,
    std::uint16_t* __restrict__ output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage = reinterpret_cast<PipelineStorage*>(dynamic_storage);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int branch = warp / 4U;
  const unsigned int branch_quarter = warp % 4U;
  WarpAccumulator accumulators;
  run_full_k<kSm87MacroFeedV4NvFp4GateUpTestInputFeatures,
             kSm87MacroFeedV4NvFp4GateUpTestKTiles,
             kSm87MacroFeedV4NvFp4GateUpTestPartitionBytes>(
      storage, input, payload, valid_rows, 0U, canonical_n_half,
      accumulators);
  auto* const temporary =
      reinterpret_cast<std::uint16_t*>(dynamic_storage);
  publish_branch_to_shared(
      temporary, accumulators, branch, branch_quarter, lane,
      branch == 0U ? gate_tensor_scale : up_tensor_scale);
  __syncthreads();
  consume_gate_up_shared<kTileN>(temporary, valid_rows, 0U, 0U, warp, lane,
                                 output);
}

[[nodiscard]] cudaError_t validate_device(int* const device_ordinal,
                                          cudaDeviceProp* const properties)
    noexcept {
  if (device_ordinal == nullptr || properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaGetDevice(device_ordinal);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(properties, *device_ordinal);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     static_cast<int>(kSm87MacroFeedV4NvFp4GateUpSmCount)
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool device_pointer(const void* const pointer,
                                  const int device_ordinal) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device_ordinal;
}

struct PointerRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] PointerRange pointer_range(const void* const pointer,
                                         const std::size_t bytes) noexcept {
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin == 0U || bytes == 0U ||
      begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return {};
  }
  return {begin, begin + bytes, true};
}

[[nodiscard]] bool disjoint(const PointerRange& left,
                            const PointerRange& right) noexcept {
  return left.valid && right.valid &&
         (left.end <= right.begin || right.end <= left.begin);
}

[[nodiscard]] cudaError_t query_resources_body(
    Sm87MacroFeedV4NvFp4GateUpCudaResources* const resources) noexcept {
  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = validate_device(&device, &properties);
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes,
                                 sm87_macrofeed_v4_nvfp4_gate_up_kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, sm87_macrofeed_v4_nvfp4_gate_up_kernel, kThreads,
      kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  resources->identity = kSm87MacroFeedV4NvFp4GateUpIdentity;
  resources->device_ordinal = device;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->sm_count = properties.multiProcessorCount;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->kernel_compiled = true;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  resources->static_resource_gate_passed =
      sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(*resources);
  return cudaSuccess;
}

}  // namespace

bool sm87_macrofeed_v4_nvfp4_gate_up_arguments_valid(
    const Sm87MacroFeedV4NvFp4GateUpArguments& arguments) noexcept {
  const auto plan =
      sm87_macrofeed_v4_nvfp4_gate_up_plan(arguments.token_count);
  if (!plan.valid() || arguments.input == nullptr ||
      arguments.payload == nullptr || arguments.output == nullptr ||
      arguments.payload_bytes != kSm87MacroFeedV4NvFp4GateUpPayloadBytes ||
      !std::isfinite(arguments.gate_tensor_scale) ||
      arguments.gate_tensor_scale <= 0.0F ||
      !std::isfinite(arguments.up_tensor_scale) ||
      arguments.up_tensor_scale <= 0.0F ||
      reinterpret_cast<std::uintptr_t>(arguments.input) % alignof(uint4) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(arguments.payload) %
              kSm87MacroFeedV4NvFp4GateUpPayloadAlignment !=
          0U ||
      reinterpret_cast<std::uintptr_t>(arguments.output) %
              alignof(std::uint32_t) !=
          0U ||
      !sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
          arguments.canonical_v3_payload_receipt) ||
      arguments.canonical_v3_payload_receipt.payload_begin !=
          reinterpret_cast<std::uintptr_t>(arguments.payload) ||
      arguments.canonical_v3_payload_receipt.payload_bytes !=
          arguments.payload_bytes) {
    return false;
  }

  constexpr std::size_t kInputBytes =
      kSm87MacroFeedV4NvFp4GateUpTokens *
      kSm87MacroFeedV4NvFp4GateUpInputFeatures * sizeof(std::uint16_t);
  constexpr std::size_t kOutputBytes =
      kSm87MacroFeedV4NvFp4GateUpTokens *
      kSm87MacroFeedV4NvFp4GateUpOutputFeatures * sizeof(std::uint16_t);
  const auto input = pointer_range(arguments.input, kInputBytes);
  const auto payload =
      pointer_range(arguments.payload, arguments.payload_bytes);
  const auto output = pointer_range(arguments.output, kOutputBytes);
  return disjoint(input, payload) && disjoint(input, output) &&
         disjoint(payload, output);
}

int query_sm87_macrofeed_v4_nvfp4_gate_up_cuda_resources(
    Sm87MacroFeedV4NvFp4GateUpCudaResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  return static_cast<int>(query_resources_body(resources));
}

int launch_sm87_macrofeed_v4_nvfp4_gate_up_cuda(
    const Sm87MacroFeedV4NvFp4GateUpArguments& arguments,
    Sm87MacroFeedV4NvFp4GateUpLaunchReceipt* const receipt) noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  if (!sm87_macrofeed_v4_nvfp4_gate_up_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Sm87MacroFeedV4NvFp4GateUpCudaResources resources{};
  const cudaError_t resource_status = query_resources_body(&resources);
  if (resource_status != cudaSuccess) {
    return static_cast<int>(resource_status);
  }
  if (!resources.static_resource_gate_passed ||
      !device_pointer(arguments.input, resources.device_ordinal) ||
      !device_pointer(arguments.payload, resources.device_ordinal) ||
      !device_pointer(arguments.output, resources.device_ordinal) ||
      resources.device_ordinal !=
          arguments.canonical_v3_payload_receipt.device_ordinal) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  sm87_macrofeed_v4_nvfp4_gate_up_kernel
      <<<kPersistentCtas, kThreads,
         kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes, stream>>>(
          arguments.input, arguments.payload, arguments.gate_tensor_scale,
          arguments.up_tensor_scale, arguments.output);
  const cudaError_t status = cudaPeekAtLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *receipt = {kSm87MacroFeedV4NvFp4GateUpIdentity,
              arguments.canonical_v3_payload_receipt.payload_identity,
              arguments.token_count,
              kSm87MacroFeedV4NvFp4GateUpLogicalTasks,
              1U,
              true,
              true,
              true,
              true,
              false};
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v4_nvfp4_gate_up_tile_test_cuda(
    const std::uint16_t* const input_m64_k256,
    const std::uint8_t* const canonical_gate_then_up_payload,
    const float gate_tensor_scale, const float up_tensor_scale,
    const std::size_t valid_rows, const std::size_t canonical_n_half,
    std::uint16_t* const output_m64_n128,
    void* const cuda_stream) noexcept {
  if (input_m64_k256 == nullptr ||
      canonical_gate_then_up_payload == nullptr ||
      output_m64_n128 == nullptr || valid_rows == 0U ||
      valid_rows > kSm87MacroFeedV4NvFp4GateUpBlockM ||
      canonical_n_half >= 2U ||
      !std::isfinite(gate_tensor_scale) || gate_tensor_scale <= 0.0F ||
      !std::isfinite(up_tensor_scale) || up_tensor_scale <= 0.0F ||
      reinterpret_cast<std::uintptr_t>(input_m64_k256) % alignof(uint4) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(canonical_gate_then_up_payload) %
              alignof(uint4) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(output_m64_n128) %
              alignof(std::uint32_t) !=
          0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  Sm87MacroFeedV4NvFp4GateUpCudaResources resources{};
  const cudaError_t resource_status = query_resources_body(&resources);
  if (resource_status != cudaSuccess) {
    return static_cast<int>(resource_status);
  }
  if (!resources.static_resource_gate_passed ||
      !device_pointer(input_m64_k256, resources.device_ordinal) ||
      !device_pointer(canonical_gate_then_up_payload,
                      resources.device_ordinal) ||
      !device_pointer(output_m64_n128, resources.device_ordinal)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  sm87_macrofeed_v4_nvfp4_gate_up_tile_test_kernel
      <<<1U, kThreads,
         kSm87MacroFeedV4NvFp4GateUpDynamicSharedBytes, stream>>>(
          input_m64_k256, canonical_gate_then_up_payload,
          static_cast<unsigned int>(valid_rows),
          static_cast<unsigned int>(canonical_n_half), gate_tensor_scale,
          up_tensor_scale, output_m64_n128);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
