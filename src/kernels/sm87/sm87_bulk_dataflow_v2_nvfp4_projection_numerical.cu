#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_projection.h"

#include "sm87_bulk_dataflow_v2_nvfp4_projection_oracle_internal.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <array>
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
constexpr unsigned int kDownOwners = 20U;
constexpr unsigned int kTileRows = 64U;
constexpr unsigned int kRowsPerGroup = 4U;
constexpr unsigned int kGroupTokens = kTileRows * kRowsPerGroup;
constexpr unsigned int kMacroTokens = 1'024U;
constexpr unsigned int kHidden = 5'120U;
constexpr unsigned int kIntermediate = 17'408U;
constexpr unsigned int kTileK = 64U;
constexpr unsigned int kK16Panels = 4U;
constexpr unsigned int kN8Panels = 8U;
constexpr unsigned int kGateKTiles = 80U;
constexpr unsigned int kGateTasks = 272U;
constexpr unsigned int kDownNTiles = 20U;
constexpr unsigned int kStages = 3U;
constexpr unsigned int kStageVectors = 1'088U;
constexpr unsigned int kActivationVectors = 512U;
constexpr unsigned int kGateWeightVectorsPerBranch = 128U;
constexpr unsigned int kGateScaleVectorsPerBranch = 16U;
constexpr unsigned int kGateStageVectors =
    kActivationVectors + kGateWeightVectorsPerBranch +
    kGateScaleVectorsPerBranch;
constexpr unsigned int kDownWeightVectors = 512U;
constexpr unsigned int kDownScaleVectors = 64U;
constexpr unsigned int kPackedCellWeightBytes = 8'192U;
constexpr unsigned int kPackedCellScaleBytes = 1'024U;
constexpr unsigned int kPackedCellBytes = 9'216U;
constexpr std::uint64_t kGatePartitionBytes = 50'135'040ULL;
constexpr unsigned int kSharedStateReady = 0xffff'fffeU;
constexpr unsigned int kSharedStateCancelled = 0xffff'fffdU;
constexpr unsigned int kSharedStateNoClaim = 0xffff'fffcU;
constexpr unsigned int kSharedStateComplete = 0xffff'fffbU;
constexpr unsigned int kInvalidClaim = 0xffff'ffffU;

static_assert(kSm87BulkV2NvFp4DynamicSharedBytes ==
              kStages * kStageVectors * sizeof(uint4));
static_assert(kGateKTiles == kSm87BulkV2NvFp4GateKTiles);
static_assert(kGateTasks == kSm87BulkV2NvFp4GateNTiles);
static_assert(kDownNTiles == kSm87BulkV2NvFp4DownNTiles);
static_assert(kPackedCellBytes ==
              kPackedCellWeightBytes + kPackedCellScaleBytes);

struct alignas(16) NumericalPipeline final {
  uint4 stage[kStages][kStageVectors];
};

static_assert(sizeof(NumericalPipeline) ==
              kSm87BulkV2NvFp4DynamicSharedBytes);

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

using GateWarpAccumulator = M16N8Accumulator[kN8Panels / 2U];
using DownWarpAccumulator = M16N8Accumulator[2U][kN8Panels];

// Gate and Up run as two exact full-K branch passes. Eight warps divide one
// M64xJ64 branch into M16xN32 fragments, halving per-thread accumulator
// residency while the Down owner retains its FP32 tile during a steal. A.ca
// makes the second activation pass cache-reusable; B is still read once.
struct alignas(16) GateBranchPipeline final {
  uint4 stage[kStages][kGateStageVectors];
  std::uint16_t gate_temporary[kTileRows * 64U];
  uint4 decoded_workspace[128U];
};

static_assert(sizeof(GateBranchPipeline) == 41'728U);
static_assert(sizeof(GateBranchPipeline) <=
              kSm87BulkV2NvFp4DynamicSharedBytes);

template <bool kCacheAll>
__device__ __forceinline__ void cp_async_16(
    void* const shared_destination,
    const void* const global_source) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(
          shared_destination));
  if constexpr (kCacheAll) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
                 :
                 : "r"(shared_address), "l"(global_source)
                 : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
                 :
                 : "r"(shared_address), "l"(global_source)
                 : "memory");
  }
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
  // Persisted [K0,K8,K1,K9] -> MMA b.x0=[K0,K1], b.x1=[K8,K9].
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
  const unsigned int column = k16 * 16U + (quadrant >> 1U) * 8U;
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
      : "r"(activation.x0), "r"(activation.x1), "r"(activation.x2),
        "r"(activation.x3), "r"(weight.x0), "r"(weight.x1));
#else
  (void)accumulator;
  (void)activation;
  (void)weight;
#endif
}

__device__ __forceinline__ void clear_gate_accumulator(
    GateWarpAccumulator& accumulator) noexcept {
#pragma unroll
  for (unsigned int n8 = 0U; n8 < kN8Panels / 2U; ++n8) {
    accumulator[n8] = {0.0F, 0.0F, 0.0F, 0.0F};
  }
}

__device__ __forceinline__ void clear_down_accumulator(
    DownWarpAccumulator& accumulator) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U; m16 < 2U; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      accumulator[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

__device__ __forceinline__ void issue_gate_stage(
    GateBranchPipeline* const pipeline, const unsigned int slot,
    const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int branch,
    const unsigned int q, const unsigned int k_tile) noexcept {
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / 8U;
    const unsigned int vector = vector_index % 8U;
    const auto* const source = reinterpret_cast<const uint4*>(
                                  input +
                                  static_cast<std::size_t>(row) * kHidden +
                                  k_tile * kTileK) +
                              vector;
    cp_async_16<true>(&pipeline->stage[slot][vector_index], source);
  }

  const unsigned int source_n_tile = q / 4U;
  const unsigned int source_n_warp = q % 4U;
  const std::uint64_t partition_offset =
      branch == 0U ? 0U : kGatePartitionBytes;
  constexpr unsigned int kGateWeightBase = kActivationVectors;
  if (threadIdx.x < kGateWeightVectorsPerBranch) {
    const unsigned int local_vector = threadIdx.x;
    const unsigned int fragment = local_vector / 4U;
    const unsigned int vector_in_fragment = local_vector % 4U;
    const unsigned int k16 = fragment / kN8Panels;
    const unsigned int n8 = fragment % kN8Panels;
    const std::uint64_t cell =
        partition_offset +
        (static_cast<std::uint64_t>(source_n_tile) * kGateKTiles +
         k_tile) *
            kPackedCellBytes;
    const std::uint64_t source_fragment =
        (static_cast<std::uint64_t>(k16) * 4U + source_n_warp) *
            kN8Panels +
        n8;
    const auto* const weight_source = reinterpret_cast<const uint4*>(
        payload + cell + source_fragment * 64U) + vector_in_fragment;
    cp_async_16<false>(
        &pipeline->stage[slot][kGateWeightBase + local_vector],
        weight_source);
  }

  if (threadIdx.x < kGateScaleVectorsPerBranch) {
    const unsigned int scale_vector = threadIdx.x;
    const unsigned int scale_k16 = scale_vector / 4U;
    const unsigned int n8_pair = scale_vector % 4U;
    const std::uint64_t scale_cell =
        partition_offset +
        (static_cast<std::uint64_t>(source_n_tile) * kGateKTiles +
         k_tile) *
            kPackedCellBytes;
    const std::uint64_t first_scale_fragment =
        (static_cast<std::uint64_t>(scale_k16) * 4U + source_n_warp) *
            kN8Panels +
        n8_pair * 2U;
    const auto* const scale_source = reinterpret_cast<const uint4*>(
        payload + scale_cell + kPackedCellWeightBytes +
        first_scale_fragment * 8U);
    constexpr unsigned int kGateScaleBase =
        kGateWeightBase + kGateWeightVectorsPerBranch;
    cp_async_16<false>(
        &pipeline->stage[slot][kGateScaleBase + scale_vector],
        scale_source);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void run_gate_branch_full_k(
    GateBranchPipeline* const pipeline,
    const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int branch,
    const unsigned int q,
    GateWarpAccumulator& accumulator) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int m16 = warp / 2U;
  const unsigned int n32 = warp % 2U;
  clear_gate_accumulator(accumulator);

  issue_gate_stage(pipeline, 0U, input, payload, branch, q, 0U);
  issue_gate_stage(pipeline, 1U, input, payload, branch, q, 1U);
  issue_gate_stage(pipeline, 2U, input, payload, branch, q, 2U);
#pragma unroll 1
  for (unsigned int k_tile = 0U; k_tile < kGateKTiles; ++k_tile) {
    if (k_tile + 2U < kGateKTiles) {
      cp_async_wait_group<2U>();
    } else if (k_tile + 1U < kGateKTiles) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();
    const unsigned int slot = k_tile % kStages;
    const auto* const shared_activation =
        reinterpret_cast<const std::uint16_t*>(
            &pipeline->stage[slot][0U]);
    constexpr unsigned int kGateWeightBase = kActivationVectors;
    constexpr unsigned int kGateScaleBase =
        kGateWeightBase + kGateWeightVectorsPerBranch;
#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      M16K16Activation activation;
      load_activation_fragment(activation, shared_activation, m16, k16,
                               lane);
      __syncthreads();

      const auto* const branch_weight =
          reinterpret_cast<const std::uint8_t*>(
              &pipeline->stage[slot][kGateWeightBase]);
      const auto* const branch_scale =
          reinterpret_cast<const std::uint8_t*>(
              &pipeline->stage[slot][kGateScaleBase]);
      auto* const decoded = reinterpret_cast<K16N8Weight*>(
          pipeline->decoded_workspace);
      const unsigned int decode_n8 = threadIdx.x / kWarpSize;
      const unsigned int decode_lane = threadIdx.x % kWarpSize;
      decoded[decode_n8 * kWarpSize + decode_lane] =
          decode_weight_fragment(
              branch_weight, branch_scale,
              k16 * kN8Panels + decode_n8, decode_lane);
      __syncthreads();
#pragma unroll
      for (unsigned int local_n8 = 0U; local_n8 < kN8Panels / 2U;
           ++local_n8) {
        const unsigned int n8 = n32 * (kN8Panels / 2U) + local_n8;
        mma_bf16(accumulator[local_n8], activation,
                 decoded[n8 * kWarpSize + lane]);
      }
      __syncthreads();
      if (k16 + 1U == kK16Panels && k_tile + kStages < kGateKTiles) {
        issue_gate_stage(pipeline, slot, input, payload, branch, q,
                         k_tile + kStages);
      }
    }
  }
  cp_async_wait_group<0U>();
  __syncthreads();
}

__device__ __forceinline__ void publish_gate_temporary(
    std::uint16_t* const temporary,
    const GateWarpAccumulator& accumulator, const unsigned int warp,
    const unsigned int lane,
    const float tensor_scale) noexcept {
  const unsigned int m16 = warp / 2U;
  const unsigned int n32 = warp % 2U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  const unsigned int row0 = m16 * 16U + lane_group;
  const unsigned int row1 = row0 + 8U;
#pragma unroll
  for (unsigned int local_n8 = 0U; local_n8 < kN8Panels / 2U;
       ++local_n8) {
    const unsigned int column =
        (n32 * (kN8Panels / 2U) + local_n8) * 8U +
        lane_in_group * 2U;
    const auto& value = accumulator[local_n8];
    *reinterpret_cast<std::uint32_t*>(
        temporary + row0 * 64U + column) =
        static_cast<std::uint32_t>(
            encode_bf16_rne(value.x0 * tensor_scale)) |
        (static_cast<std::uint32_t>(
             encode_bf16_rne(value.x1 * tensor_scale))
         << 16U);
    *reinterpret_cast<std::uint32_t*>(
        temporary + row1 * 64U + column) =
        static_cast<std::uint32_t>(
            encode_bf16_rne(value.x2 * tensor_scale)) |
        (static_cast<std::uint32_t>(
             encode_bf16_rne(value.x3 * tensor_scale))
         << 16U);
  }
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t silu_times_up(
    const std::uint16_t gate_bits,
    const std::uint16_t up_bits) noexcept {
  const float gate = decode_bf16(gate_bits);
  return encode_bf16_rne(
      gate / (1.0F + expf(-gate)) * decode_bf16(up_bits));
}

__device__ __forceinline__ void publish_up_to_h(
    const std::uint16_t* const gate_temporary,
    const GateWarpAccumulator& up_accumulator,
    const float up_tensor_scale, const unsigned int q,
    std::uint16_t* const h) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int m16 = warp / 2U;
  const unsigned int n32 = warp % 2U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  const unsigned int row0 = m16 * 16U + lane_group;
  const unsigned int row1 = row0 + 8U;
#pragma unroll
  for (unsigned int local_n8 = 0U; local_n8 < kN8Panels / 2U;
       ++local_n8) {
    const unsigned int column =
        (n32 * (kN8Panels / 2U) + local_n8) * 8U +
        lane_in_group * 2U;
    const std::uint32_t gate0 = *reinterpret_cast<const std::uint32_t*>(
        gate_temporary + row0 * 64U + column);
    const std::uint32_t gate1 = *reinterpret_cast<const std::uint32_t*>(
        gate_temporary + row1 * 64U + column);
    const auto& up = up_accumulator[local_n8];
    const std::uint32_t up0 =
        static_cast<std::uint32_t>(
            encode_bf16_rne(up.x0 * up_tensor_scale)) |
        (static_cast<std::uint32_t>(
             encode_bf16_rne(up.x1 * up_tensor_scale))
         << 16U);
    const std::uint32_t up1 =
        static_cast<std::uint32_t>(
            encode_bf16_rne(up.x2 * up_tensor_scale)) |
        (static_cast<std::uint32_t>(
             encode_bf16_rne(up.x3 * up_tensor_scale))
         << 16U);
    *reinterpret_cast<std::uint32_t*>(
        h + static_cast<std::size_t>(row0) * kIntermediate + q * 64U +
        column) =
        static_cast<std::uint32_t>(silu_times_up(
            static_cast<std::uint16_t>(gate0),
            static_cast<std::uint16_t>(up0))) |
        (static_cast<std::uint32_t>(silu_times_up(
             static_cast<std::uint16_t>(gate0 >> 16U),
             static_cast<std::uint16_t>(up0 >> 16U)))
         << 16U);
    *reinterpret_cast<std::uint32_t*>(
        h + static_cast<std::size_t>(row1) * kIntermediate + q * 64U +
        column) =
        static_cast<std::uint32_t>(silu_times_up(
            static_cast<std::uint16_t>(gate1),
            static_cast<std::uint16_t>(up1))) |
        (static_cast<std::uint32_t>(silu_times_up(
             static_cast<std::uint16_t>(gate1 >> 16U),
             static_cast<std::uint16_t>(up1 >> 16U)))
         << 16U);
  }
}

[[nodiscard]] __device__ __forceinline__ bool cancellation_requested(
    const std::uint32_t* const signal) noexcept {
  return signal != nullptr &&
         *reinterpret_cast<const volatile std::uint32_t*>(signal) != 0U;
}

[[nodiscard]] __device__ __forceinline__ std::uintptr_t
load_numerical_continuation(
    const std::uintptr_t* const address) noexcept {
  return *reinterpret_cast<const volatile std::uintptr_t*>(address);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
load_numerical_scale_bits(
    const std::uint32_t* const address) noexcept {
  return *reinterpret_cast<const volatile std::uint32_t*>(address);
}

[[nodiscard]] __device__ __forceinline__ const std::uint32_t*
load_cancellation_continuation(
    const Sm87BulkV2NvFp4DeviceControl* const control) noexcept {
  return reinterpret_cast<const std::uint32_t*>(
      load_numerical_continuation(
          &control->numerical_cancellation_signal));
}

[[nodiscard]] __device__ __forceinline__ unsigned int claim_gate_task(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int row) noexcept {
  unsigned int next = atomicAdd(&control->gate_next_q[row], 0U);
  const unsigned int retired = atomicAdd(&control->retired_q[row], 0U);
  while (next < kGateTasks &&
         next < retired + kSm87BulkV2NvFp4ReadinessSlots) {
    const unsigned int observed =
        atomicCAS(&control->gate_next_q[row], next, next + 1U);
    if (observed == next) {
      atomicAdd(&control->claimed_gate_tasks, 1U);
      return row * kGateTasks + next;
    }
    next = observed;
  }
  return kInvalidClaim;
}

[[nodiscard]] __device__ __forceinline__ unsigned int scan_gate_task(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int active_rows,
    const unsigned int scan_seed) noexcept {
  for (unsigned int offset = 0U; offset < active_rows; ++offset) {
    const unsigned int row = (scan_seed + offset) % active_rows;
    const unsigned int claim = claim_gate_task(control, row);
    if (claim != kInvalidClaim) {
      return claim;
    }
  }
  return kInvalidClaim;
}

[[nodiscard]] __device__ __forceinline__ bool execute_gate_claim(
    NumericalPipeline* const pipeline,
    const std::uint16_t* const input,
    const std::uint8_t* const gate_payload,
    const float gate_scale, const float up_scale,
    std::uint16_t* const h,
    Sm87BulkV2NvFp4DeviceControl* const control,
    const std::uint32_t* const cancellation_signal,
    const unsigned int encoded_claim) noexcept {
  const unsigned int row = encoded_claim / kGateTasks;
  const unsigned int q = encoded_claim % kGateTasks;
  const auto* const row_input =
      input + static_cast<std::size_t>(row) * kTileRows * kHidden;
  auto* const row_h =
      h + static_cast<std::size_t>(row) * kTileRows * kIntermediate;
  auto* const gate_pipeline =
      reinterpret_cast<GateBranchPipeline*>(pipeline);
  GateWarpAccumulator accumulator;
  run_gate_branch_full_k(
      gate_pipeline, row_input, gate_payload, 0U, q, accumulator);
  const unsigned int warp = threadIdx.x / kWarpSize;
  publish_gate_temporary(
      gate_pipeline->gate_temporary, accumulator, warp,
      threadIdx.x % kWarpSize, gate_scale);
  __syncthreads();
  if (cancellation_requested(cancellation_signal)) {
    if (threadIdx.x == 0U) {
      atomicExch(&control->cancellation_observed, 1U);
    }
    __syncthreads();
    return false;
  }
  run_gate_branch_full_k(
      gate_pipeline, row_input, gate_payload, 1U, q, accumulator);
  if (cancellation_requested(cancellation_signal)) {
    if (threadIdx.x == 0U) {
      atomicExch(&control->cancellation_observed, 1U);
    }
    __syncthreads();
    return false;
  }
  publish_up_to_h(
      gate_pipeline->gate_temporary, accumulator, up_scale, q, row_h);
  __syncthreads();
  if (threadIdx.x == 0U) {
    __threadfence();
    const unsigned int slot =
        q % kSm87BulkV2NvFp4ReadinessSlots;
    atomicExch(&control->readers_remaining[row][slot], kDownOwners);
    __threadfence();
    atomicExch(
        &control->readiness_generation[row][slot],
        q / kSm87BulkV2NvFp4ReadinessSlots + 1U);
    atomicAdd(&control->completed_gate_tasks, 1U);
  }
  __syncthreads();
  return true;
}

// Down owners spend almost all of their time retaining a 64-float FP32 Down
// fragment.  The Gate inputs are needed only on a steal decision.  Volatile
// continuation loads at that branch keep those values out of the live range
// of the Down accumulator while preserving the exact same Gate implementation.
[[nodiscard]] __device__ __forceinline__ bool
execute_gate_claim_from_control(
    NumericalPipeline* const pipeline,
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int encoded_claim) noexcept {
  return execute_gate_claim(
      pipeline,
      reinterpret_cast<const std::uint16_t*>(
          load_numerical_continuation(&control->numerical_input)),
      reinterpret_cast<const std::uint8_t*>(
          load_numerical_continuation(&control->numerical_gate_payload)),
      __uint_as_float(
          load_numerical_scale_bits(&control->gate_tensor_scale_bits)),
      __uint_as_float(
          load_numerical_scale_bits(&control->up_tensor_scale_bits)),
      reinterpret_cast<std::uint16_t*>(
          load_numerical_continuation(&control->numerical_h)),
      control, load_cancellation_continuation(control), encoded_claim);
}

[[nodiscard]] __device__ __forceinline__ unsigned int shared_decision(
    NumericalPipeline* const pipeline) noexcept {
  return *reinterpret_cast<volatile unsigned int*>(pipeline);
}

__device__ __forceinline__ void set_shared_decision(
    NumericalPipeline* const pipeline,
    const unsigned int value) noexcept {
  *reinterpret_cast<volatile unsigned int*>(pipeline) = value;
}

__device__ __forceinline__ void issue_down_stage(
    NumericalPipeline* const pipeline, const unsigned int slot,
    const std::uint16_t* const h,
    const std::uint8_t* const down_payload,
    const unsigned int n_tile, const unsigned int q) noexcept {
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / 8U;
    const unsigned int vector = vector_index % 8U;
    const auto* const source = reinterpret_cast<const uint4*>(
                                  h +
                                  static_cast<std::size_t>(row) *
                                      kIntermediate +
                                  q * 64U) +
                              vector;
    cp_async_16<true>(&pipeline->stage[slot][vector_index], source);
  }
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector = threadIdx.x + pass * kThreads;
    const std::uint64_t cell =
        (static_cast<std::uint64_t>(n_tile) * kGateTasks + q) *
        kPackedCellBytes;
    const auto* const source = reinterpret_cast<const uint4*>(
        down_payload + cell) + vector;
    cp_async_16<false>(
        &pipeline->stage[slot][kActivationVectors + vector], source);
  }
  if (threadIdx.x < kDownScaleVectors) {
    const std::uint64_t cell =
        (static_cast<std::uint64_t>(n_tile) * kGateTasks + q) *
        kPackedCellBytes;
    const auto* const source = reinterpret_cast<const uint4*>(
        down_payload + cell + kPackedCellWeightBytes) + threadIdx.x;
    cp_async_16<false>(
        &pipeline->stage[slot]
                        [kActivationVectors + kDownWeightVectors +
                         threadIdx.x],
        source);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void consume_down_stage(
    NumericalPipeline* const pipeline, const unsigned int slot,
    DownWarpAccumulator& accumulator) noexcept {
  // Read %tid.x at the consumption branch.  A plain threadIdx.x expression is
  // loop invariant, so ptxas hoists the derived shared-address terms above the
  // work-steal loop and spills them while the 64-float Down accumulator is
  // live.  The volatile special-register read deliberately trades a handful
  // of integer instructions for zero local traffic on this branch.
  unsigned int consumer_thread = 0U;
  asm volatile("mov.u32 %0, %%tid.x;"
               : "=r"(consumer_thread));
  const unsigned int warp = consumer_thread / kWarpSize;
  const unsigned int lane = consumer_thread % kWarpSize;
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
                          [kActivationVectors + kDownWeightVectors]);
#pragma unroll
  for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
    M16K16Activation activation[2U];
#pragma unroll
    for (unsigned int local_m16 = 0U; local_m16 < 2U; ++local_m16) {
      load_activation_fragment(activation[local_m16], shared_activation,
                               warp_m * 2U + local_m16, k16, lane);
    }
    __syncthreads();

    // All eight warps retain their A fragment before this K16 slice is
    // reclaimed as a 2 KiB decoded-B workspace. Decode one N64 warp at a
    // time, shared by its two M32 consumers, without materializing B in
    // global memory or changing K order.
#pragma unroll 1
    for (unsigned int decode_n_warp = 0U; decode_n_warp < 4U;
         ++decode_n_warp) {
      const unsigned int decode_n8 = consumer_thread / kWarpSize;
      const unsigned int decode_lane = consumer_thread % kWarpSize;
      const unsigned int decode_ordinal =
          decode_n8 * kWarpSize + decode_lane;
      const unsigned int workspace_row = decode_ordinal / 4U;
      const unsigned int workspace_quarter = decode_ordinal % 4U;
      auto* const decoded = reinterpret_cast<K16N8Weight*>(
          const_cast<std::uint16_t*>(shared_activation) +
          workspace_row * kTileK + k16 * 16U +
          workspace_quarter * 4U);
      *decoded = decode_weight_fragment(
          shared_weight, shared_scale,
          (k16 * 4U + decode_n_warp) * kN8Panels + decode_n8,
          decode_lane);
      __syncthreads();
      if (warp_n == decode_n_warp) {
#pragma unroll
        for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
          const unsigned int ordinal = n8 * kWarpSize + lane;
          const unsigned int row = ordinal / 4U;
          const unsigned int quarter = ordinal % 4U;
          const auto* const shared_decoded =
              reinterpret_cast<const K16N8Weight*>(
                  shared_activation + row * kTileK + k16 * 16U +
                  quarter * 4U);
#pragma unroll
          for (unsigned int local_m16 = 0U; local_m16 < 2U;
               ++local_m16) {
            mma_bf16(accumulator[local_m16][n8],
                     activation[local_m16], *shared_decoded);
          }
        }
      }
      __syncthreads();
    }
  }
}

__device__ __forceinline__ void retire_down_read(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int row, const unsigned int q) noexcept {
  const unsigned int slot = q % kSm87BulkV2NvFp4ReadinessSlots;
  unsigned int readers =
      atomicAdd(&control->readers_remaining[row][slot], 0U);
  while (readers != 0U) {
    const unsigned int observed = atomicCAS(
        &control->readers_remaining[row][slot], readers, readers - 1U);
    if (observed == readers) {
      if (readers == 1U) {
        __threadfence();
        atomicExch(&control->retired_q[row], q + 1U);
      }
      return;
    }
    readers = observed;
  }
}

[[nodiscard]] __device__ __forceinline__ unsigned int ready_batch_count(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int row, const unsigned int first_q) noexcept {
  unsigned int count = 0U;
  while (count < kStages && first_q + count < kGateTasks) {
    const unsigned int q = first_q + count;
    const unsigned int slot = q % kSm87BulkV2NvFp4ReadinessSlots;
    const unsigned int generation =
        q / kSm87BulkV2NvFp4ReadinessSlots + 1U;
    if (atomicAdd(&control->readiness_generation[row][slot], 0U) !=
        generation) {
      break;
    }
    ++count;
  }
  return count;
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t add_residual_pair(
    const float low, const float high, const float tensor_scale,
    const std::uint32_t residual_bits) noexcept {
  const float residual0 = decode_bf16(
      static_cast<std::uint16_t>(residual_bits));
  const float residual1 = decode_bf16(
      static_cast<std::uint16_t>(residual_bits >> 16U));
  const float branch0 = decode_bf16(
      encode_bf16_rne(low * tensor_scale));
  const float branch1 = decode_bf16(
      encode_bf16_rne(high * tensor_scale));
  return static_cast<std::uint32_t>(
             encode_bf16_rne(branch0 + residual0)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(branch1 + residual1))
          << 16U);
}

__device__ __forceinline__ void publish_down_residual(
    const DownWarpAccumulator& accumulator,
    const unsigned int n_tile, const float tensor_scale,
    std::uint16_t* const residual) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int local_m16 = 0U; local_m16 < 2U; ++local_m16) {
    const unsigned int row0 =
        warp_m * 32U + local_m16 * 16U + lane_group;
    const unsigned int row1 = row0 + 8U;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      const unsigned int column =
          n_tile * 256U + warp_n * 64U + n8 * 8U +
          lane_in_group * 2U;
      const auto& value = accumulator[local_m16][n8];
      auto* const destination0 = reinterpret_cast<std::uint32_t*>(
          residual + static_cast<std::size_t>(row0) * kHidden + column);
      auto* const destination1 = reinterpret_cast<std::uint32_t*>(
          residual + static_cast<std::size_t>(row1) * kHidden + column);
      *destination0 = add_residual_pair(
          value.x0, value.x1, tensor_scale, *destination0);
      *destination1 = add_residual_pair(
          value.x2, value.x3, tensor_scale, *destination1);
    }
  }
}

__device__ __forceinline__ void execute_dedicated_producer(
    NumericalPipeline* const pipeline,
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int active_rows) noexcept {
  const unsigned int total_gate_tasks = active_rows * kGateTasks;
  for (;;) {
    if (threadIdx.x == 0U) {
      unsigned int decision = kSharedStateNoClaim;
      if (cancellation_requested(load_cancellation_continuation(control)) ||
          atomicAdd(&control->cancellation_observed, 0U) != 0U) {
        atomicExch(&control->cancellation_observed, 1U);
        decision = kSharedStateCancelled;
      } else if (atomicAdd(&control->completed_gate_tasks, 0U) >=
                 total_gate_tasks) {
        decision = kSharedStateComplete;
      } else {
        decision = scan_gate_task(
            control, active_rows,
            (blockIdx.x - kDownOwners) % active_rows);
      }
      set_shared_decision(pipeline, decision);
    }
    __syncthreads();
    const unsigned int decision = shared_decision(pipeline);
    __syncthreads();
    if (decision == kSharedStateCancelled ||
        decision == kSharedStateComplete) {
      break;
    }
    if (decision != kSharedStateNoClaim && decision != kInvalidClaim) {
      if (!execute_gate_claim_from_control(
              pipeline, control, decision)) {
        break;
      }
    } else if (threadIdx.x == 0U) {
      __nanosleep(64U);
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void execute_down_owner(
    NumericalPipeline* const pipeline,
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int active_rows) noexcept {
  const unsigned int n_tile = blockIdx.x;
  for (unsigned int row = 0U; row < active_rows; ++row) {
    DownWarpAccumulator accumulator;
    clear_down_accumulator(accumulator);
    unsigned int q = 0U;
    bool cancelled = false;
    while (q < kGateTasks) {
      if (threadIdx.x == 0U) {
        unsigned int decision = kSharedStateNoClaim;
        if (cancellation_requested(
                load_cancellation_continuation(control)) ||
            atomicAdd(&control->cancellation_observed, 0U) != 0U) {
          // Record cancellation even when it is already asserted at group
          // entry. Mid-claim cancellation is handled by execute_gate_claim.
          atomicExch(&control->cancellation_observed, 1U);
          decision = kSharedStateCancelled;
        } else {
          const unsigned int slot =
              q % kSm87BulkV2NvFp4ReadinessSlots;
          const unsigned int generation =
              q / kSm87BulkV2NvFp4ReadinessSlots + 1U;
          decision =
              atomicAdd(&control->readiness_generation[row][slot], 0U) ==
                      generation
                  ? kSharedStateReady
                  : claim_gate_task(control, row);
        }
        set_shared_decision(pipeline, decision);
      }
      __syncthreads();
      const unsigned int decision = shared_decision(pipeline);
      __syncthreads();
      if (decision == kSharedStateCancelled) {
        cancelled = true;
        break;
      }
      if (decision == kSharedStateReady) {
        if (threadIdx.x == 0U) {
          set_shared_decision(
              pipeline, ready_batch_count(control, row, q));
        }
        __syncthreads();
        const unsigned int batch = shared_decision(pipeline);
        __syncthreads();
        const auto* const active_h =
            reinterpret_cast<const std::uint16_t*>(
                load_numerical_continuation(&control->numerical_h));
        const auto* const active_down_payload =
            reinterpret_cast<const std::uint8_t*>(
                load_numerical_continuation(
                    &control->numerical_down_payload));
        for (unsigned int item = 0U; item < batch; ++item) {
          issue_down_stage(
              pipeline, item,
              active_h + static_cast<std::size_t>(row) * kTileRows *
                             kIntermediate,
              active_down_payload,
              n_tile, q + item);
        }
        for (unsigned int item = 0U; item < batch; ++item) {
          const unsigned int outstanding = batch - item - 1U;
          if (outstanding == 2U) {
            cp_async_wait_group<2U>();
          } else if (outstanding == 1U) {
            cp_async_wait_group<1U>();
          } else {
            cp_async_wait_group<0U>();
          }
          __syncthreads();
          if (threadIdx.x == 0U) {
            retire_down_read(control, row, q + item);
          }
          consume_down_stage(pipeline, item, accumulator);
          __syncthreads();
        }
        q += batch;
        continue;
      }
      if (decision != kSharedStateNoClaim &&
          decision != kInvalidClaim) {
        if (!execute_gate_claim_from_control(
                pipeline, control, decision)) {
          cancelled = true;
          break;
        }
      } else if (threadIdx.x == 0U) {
        __nanosleep(64U);
      }
      __syncthreads();
    }
    if (cancelled) {
      break;
    }
    publish_down_residual(
        accumulator, n_tile,
        __uint_as_float(control->down_tensor_scale_bits),
        reinterpret_cast<std::uint16_t*>(control->numerical_residual) +
            static_cast<std::size_t>(row) * kTileRows * kHidden);
    __syncthreads();
    if (threadIdx.x == 0U) {
      __threadfence();
      atomicAdd(&control->completed_down_tasks, 1U);
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void execute_work_conserving_role(
    NumericalPipeline* const pipeline,
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int active_rows) noexcept {
  if (blockIdx.x < kDownOwners) {
    execute_down_owner(pipeline, control, active_rows);
  } else {
    execute_dedicated_producer(pipeline, control, active_rows);
  }
}

__device__ __forceinline__ void initialize_group_control(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int active_rows) noexcept {
  const unsigned int global_thread =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread < kRowsPerGroup) {
    control->gate_next_q[global_thread] = 0U;
    control->retired_q[global_thread] = 0U;
  }
  if (global_thread <
      kRowsPerGroup * kSm87BulkV2NvFp4ReadinessSlots) {
    const unsigned int row =
        global_thread / kSm87BulkV2NvFp4ReadinessSlots;
    const unsigned int slot =
        global_thread % kSm87BulkV2NvFp4ReadinessSlots;
    control->readiness_generation[row][slot] = 0U;
    control->readers_remaining[row][slot] = 0U;
  }
  if (global_thread == 0U) {
    control->claimed_gate_tasks = 0U;
    control->completed_gate_tasks = 0U;
    control->completed_down_tasks = 0U;
    control->cancellation_observed = 0U;
    control->active_rows = static_cast<std::uint16_t>(active_rows);
  }
}

__device__ __forceinline__ void initialize_macro_control(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int group_epoch,
    const std::uint16_t* const input,
    const std::uint8_t* const gate_payload,
    const float gate_scale, const float up_scale,
    const std::uint8_t* const down_payload,
    const float down_scale, std::uint16_t* const residual,
    std::uint16_t* const h,
    const std::uint32_t* const cancellation_signal) noexcept {
  const unsigned int global_thread =
      blockIdx.x * blockDim.x + threadIdx.x;
  auto* const control_words = reinterpret_cast<std::uint32_t*>(control);
  constexpr unsigned int kControlWords =
      sizeof(Sm87BulkV2NvFp4DeviceControl) / sizeof(std::uint32_t);
  for (unsigned int word = global_thread; word < kControlWords;
       word += kPersistentCtas * kThreads) {
    control_words[word] = 0U;
  }
  if (global_thread == 0U) {
    control->group_epoch = group_epoch;
    control->active_group = 0U;
    control->active_rows = 0U;
    control->down_tensor_scale_bits = __float_as_uint(down_scale);
    control->numerical_down_payload =
        reinterpret_cast<std::uintptr_t>(down_payload);
    control->numerical_residual =
        reinterpret_cast<std::uintptr_t>(residual);
    control->numerical_input = reinterpret_cast<std::uintptr_t>(input);
    control->numerical_gate_payload =
        reinterpret_cast<std::uintptr_t>(gate_payload);
    control->numerical_h = reinterpret_cast<std::uintptr_t>(h);
    control->gate_tensor_scale_bits = __float_as_uint(gate_scale);
    control->up_tensor_scale_bits = __float_as_uint(up_scale);
    control->numerical_cancellation_signal =
        reinterpret_cast<std::uintptr_t>(cancellation_signal);
  }
}

__global__ __launch_bounds__(256, 2)
void work_conserving_nvfp4_tail_numerical_kernel(
    const std::uint16_t* const input,
    const std::uint8_t* const gate_payload,
    const float gate_scale, const float up_scale,
    const std::uint8_t* const down_payload,
    const float down_scale, std::uint16_t* const residual,
    std::uint16_t* const h,
    Sm87BulkV2NvFp4DeviceControl* const control,
    const std::uint32_t* const cancellation_signal,
    const unsigned int group_epoch) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* const pipeline =
      reinterpret_cast<NumericalPipeline*>(shared_bytes);

  initialize_macro_control(
      control, group_epoch, input, gate_payload, gate_scale, up_scale,
      down_payload, down_scale, residual, h, cancellation_signal);
  cg::this_grid().sync();
  initialize_group_control(control, 1U);
  cg::this_grid().sync();
  execute_work_conserving_role(
      pipeline, control, 1U);
  cg::this_grid().sync();
  if (blockIdx.x == 0U && threadIdx.x == 0U &&
      control->cancellation_observed == 0U) {
    control->macro_completed_groups = 1U;
    control->macro_claimed_gate_tasks = control->claimed_gate_tasks;
    control->macro_completed_gate_tasks = control->completed_gate_tasks;
    control->macro_completed_down_tasks = control->completed_down_tasks;
  }
}

__global__ __launch_bounds__(256, 2)
void work_conserving_nvfp4_m1024_numerical_kernel(
    const std::uint16_t* const input,
    const std::uint8_t* const gate_payload,
    const float gate_scale, const float up_scale,
    const std::uint8_t* const down_payload,
    const float down_scale, std::uint16_t* const residual,
    std::uint16_t* const h,
    Sm87BulkV2NvFp4DeviceControl* const control,
    const std::uint32_t* const cancellation_signal,
    const unsigned int group_epoch) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* const pipeline = reinterpret_cast<NumericalPipeline*>(shared_bytes);
  initialize_macro_control(
      control, group_epoch, input, gate_payload, gate_scale, up_scale,
      down_payload, down_scale, residual, h, cancellation_signal);
  cg::this_grid().sync();

  while (true) {
    initialize_group_control(control, kRowsPerGroup);
    cg::this_grid().sync();
    execute_work_conserving_role(
        pipeline, control, kRowsPerGroup);
    cg::this_grid().sync();
    if (blockIdx.x == 0U && threadIdx.x == 0U &&
        control->cancellation_observed == 0U) {
      ++control->macro_completed_groups;
      control->macro_claimed_gate_tasks += control->claimed_gate_tasks;
      control->macro_completed_gate_tasks +=
          control->completed_gate_tasks;
      control->macro_completed_down_tasks +=
          control->completed_down_tasks;
      if (control->macro_completed_groups < kRowsPerGroup) {
        control->numerical_input +=
            static_cast<std::uintptr_t>(kGroupTokens) * kHidden *
            sizeof(std::uint16_t);
        control->numerical_residual +=
            static_cast<std::uintptr_t>(kGroupTokens) * kHidden *
            sizeof(std::uint16_t);
        control->active_group = static_cast<std::uint16_t>(
            control->macro_completed_groups);
      }
    }
    cg::this_grid().sync();
    if (control->cancellation_observed != 0U ||
        control->macro_completed_groups >= kRowsPerGroup) {
      break;
    }
  }
}

[[nodiscard]] cudaError_t validate_device(
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
                 properties->multiProcessorCount == 16 &&
                 properties->cooperativeLaunch != 0 &&
                 properties->sharedMemPerBlockOptin >=
                     kSm87BulkV2NvFp4DynamicSharedBytes
             ? cudaSuccess
             : cudaErrorNotSupported;
}

template <typename Kernel>
[[nodiscard]] cudaError_t set_numerical_dynamic_shared(
    Kernel kernel) noexcept {
  return cudaFuncSetAttribute(
      kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87BulkV2NvFp4DynamicSharedBytes));
}

[[nodiscard]] bool finite_positive(const float value) noexcept {
  return std::isfinite(value) && value > 0.0F;
}

struct ExecutionRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
};

[[nodiscard]] bool make_execution_range(
    const void* const pointer, const std::size_t bytes,
    ExecutionRange* const range) noexcept {
  if (pointer == nullptr || range == nullptr || bytes == 0U) {
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  *range = {begin, begin + bytes};
  return true;
}

[[nodiscard]] bool execution_ranges_disjoint(
    const ExecutionRange& left, const ExecutionRange& right) noexcept {
  return left.end <= right.begin || right.end <= left.begin;
}

[[nodiscard]] bool execution_device_pointer(
    const void* const pointer, const int device_ordinal) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device_ordinal;
}

[[nodiscard]] bool exact_execution_ranges_valid(
    const Sm87BulkV2NvFp4MacroArguments& arguments) noexcept {
  const auto segment =
      sm87_bulk_v2_nvfp4_segment_plan(arguments.segment);
  if (!segment.valid) {
    return false;
  }
  const std::size_t input_bytes =
      static_cast<std::size_t>(segment.token_count) * kHidden *
      sizeof(std::uint16_t);
  constexpr std::size_t kHBytes =
      kGroupTokens * kIntermediate * sizeof(std::uint16_t);
  std::array<ExecutionRange, 7U> ranges{};
  if (!make_execution_range(arguments.normalized_input, input_bytes,
                            &ranges[0U]) ||
      !make_execution_range(arguments.residual, input_bytes, &ranges[1U]) ||
      !make_execution_range(arguments.group_h_scratch, kHBytes,
                            &ranges[2U]) ||
      !make_execution_range(arguments.device_control,
                            sizeof(Sm87BulkV2NvFp4DeviceControl),
                            &ranges[3U]) ||
      !make_execution_range(
          reinterpret_cast<const void*>(arguments.assets.gate_up.payload.begin),
          static_cast<std::size_t>(arguments.assets.gate_up.payload.bytes),
          &ranges[4U]) ||
      !make_execution_range(
          reinterpret_cast<const void*>(arguments.assets.down.payload.begin),
          static_cast<std::size_t>(arguments.assets.down.payload.bytes),
          &ranges[5U])) {
    return false;
  }
  std::size_t range_count = 6U;
  if (arguments.cancellation_signal != nullptr) {
    if (!make_execution_range(arguments.cancellation_signal,
                              sizeof(std::uint32_t), &ranges[6U])) {
      return false;
    }
    range_count = 7U;
  }
  for (std::size_t left = 0U; left < range_count; ++left) {
    for (std::size_t right = left + 1U; right < range_count; ++right) {
      if (!execution_ranges_disjoint(ranges[left], ranges[right])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] cudaError_t launch_raw_macro_unchecked(
    const sm87_bulk_v2_nvfp4_oracle_detail::RawTailArguments&
        arguments) noexcept {
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const std::uint16_t* input_argument = arguments.normalized_input;
  const std::uint8_t* gate_payload_argument = arguments.gate_up_payload;
  float gate_scale_argument = arguments.gate_tensor_scale;
  float up_scale_argument = arguments.up_tensor_scale;
  const std::uint8_t* down_payload_argument = arguments.down_payload;
  float down_scale_argument = arguments.down_tensor_scale;
  std::uint16_t* residual_argument = arguments.residual;
  std::uint16_t* h_argument = arguments.group_h_scratch;
  Sm87BulkV2NvFp4DeviceControl* control_argument =
      arguments.device_control;
  const std::uint32_t* cancellation_argument =
      arguments.cancellation_signal;
  unsigned int epoch_argument = arguments.group_epoch;
  void* kernel_arguments[] = {
      &input_argument,       &gate_payload_argument,
      &gate_scale_argument,  &up_scale_argument,
      &down_payload_argument, &down_scale_argument,
      &residual_argument,    &h_argument,
      &control_argument,     &cancellation_argument,
      &epoch_argument,
  };
  const auto launch = [&](auto kernel) {
    cudaError_t status = set_numerical_dynamic_shared(kernel);
    return status == cudaSuccess
               ? cudaLaunchCooperativeKernel(
                     kernel, dim3{kPersistentCtas}, dim3{kThreads},
                     kernel_arguments, kSm87BulkV2NvFp4DynamicSharedBytes,
                     stream)
               : status;
  };
  if (arguments.token_count != kMacroTokens) {
    return launch(work_conserving_nvfp4_tail_numerical_kernel);
  }
  return launch(work_conserving_nvfp4_m1024_numerical_kernel);
}

}  // namespace

int query_sm87_bulk_dataflow_v2_nvfp4_tail_numerical_resources_cuda(
    const Sm87BulkV2NvFp4CodeEvidence* const code_evidence,
    Sm87BulkV2NvFp4TailNumericalResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  cudaDeviceProp properties{};
  cudaError_t status = validate_device(&properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = set_numerical_dynamic_shared(
      work_conserving_nvfp4_tail_numerical_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes, work_conserving_nvfp4_tail_numerical_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, work_conserving_nvfp4_tail_numerical_kernel,
      kThreads, kSm87BulkV2NvFp4DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  auto& kernel = resources->kernel;
  kernel.binary_version = attributes.binaryVersion;
  kernel.registers_per_thread = attributes.numRegs;
  kernel.static_shared_bytes = attributes.sharedSizeBytes;
  kernel.dynamic_shared_bytes = kSm87BulkV2NvFp4DynamicSharedBytes;
  kernel.local_bytes = attributes.localSizeBytes;
  kernel.maximum_threads_per_block = attributes.maxThreadsPerBlock;
  kernel.active_blocks_per_sm = active_blocks;
  kernel.cooperative_grid_capacity =
      active_blocks * properties.multiProcessorCount;
  if (code_evidence != nullptr) {
    kernel.code = *code_evidence;
  }
  kernel.kernel_compiled = true;
  kernel.cooperative_launch_supported = true;
  kernel.numerical_contract_qualified = false;
  kernel.production_dispatch_eligible = false;
  kernel.resource_and_code_gate_passed =
      kernel.binary_version == 87 &&
      kernel.registers_per_thread > 0 &&
      kernel.registers_per_thread <=
          static_cast<int>(kSm87BulkV2NvFp4MaximumRegisters) &&
      kernel.static_shared_bytes == 0U && kernel.local_bytes == 0U &&
      kernel.maximum_threads_per_block >= static_cast<int>(kThreads) &&
      kernel.active_blocks_per_sm >= 2 &&
      kernel.cooperative_grid_capacity >= 32 &&
      sm87_bulk_v2_nvfp4_code_evidence_valid(kernel.code);
  resources->exact_m64_tail_geometry = true;
  resources->authenticated_raw_payload_path = true;
  resources->ascending_full_k_without_split_k = true;
  resources->numerical_body_compiled = true;
  resources->exact_control_stepping_stone = true;
  resources->cross_group_weight_residency_qualified = false;
  resources->p40_hot_path_qualified = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_bulk_dataflow_v2_nvfp4_macro_numerical_cuda(
    const Sm87BulkV2NvFp4MacroArguments& arguments) noexcept {
  const auto segment =
      sm87_bulk_v2_nvfp4_segment_plan(arguments.segment);
  if (arguments.transaction_epoch == 0U ||
      arguments.transaction_epoch >
          std::numeric_limits<std::uint32_t>::max() ||
      arguments.layer >= kSm87BulkV2NvFp4LayerCount ||
      !segment.valid ||
      arguments.normalized_input == nullptr || arguments.residual == nullptr ||
      arguments.group_h_scratch == nullptr ||
      arguments.device_control == nullptr || arguments.cuda_stream == nullptr ||
      !sm87_bulk_v2_nvfp4_assets_valid(arguments.assets)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  float gate_scale = 0.0F;
  float up_scale = 0.0F;
  float down_scale = 0.0F;
  std::memcpy(&gate_scale, &arguments.assets.gate_up.tensor_scale_bits[0U],
              sizeof(gate_scale));
  std::memcpy(&up_scale, &arguments.assets.gate_up.tensor_scale_bits[1U],
              sizeof(up_scale));
  std::memcpy(&down_scale, &arguments.assets.down.tensor_scale_bits[0U],
              sizeof(down_scale));
  if (!finite_positive(gate_scale) || !finite_positive(up_scale) ||
      !finite_positive(down_scale)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaDeviceProp properties{};
  const cudaError_t device_status = validate_device(&properties);
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  int device_ordinal = -1;
  const cudaError_t current_device_status = cudaGetDevice(&device_ordinal);
  if (current_device_status != cudaSuccess) {
    return static_cast<int>(current_device_status);
  }
  const auto& gate_upload =
      arguments.assets.gate_up.device_upload_receipt;
  const auto& down_upload = arguments.assets.down.device_upload_receipt;
  if (gate_upload.device_ordinal != device_ordinal ||
      down_upload.device_ordinal != device_ordinal ||
      !execution_device_pointer(arguments.normalized_input, device_ordinal) ||
      !execution_device_pointer(arguments.residual, device_ordinal) ||
      !execution_device_pointer(arguments.group_h_scratch, device_ordinal) ||
      !execution_device_pointer(arguments.device_control, device_ordinal) ||
      (arguments.cancellation_signal != nullptr &&
       !execution_device_pointer(arguments.cancellation_signal,
                                 device_ordinal)) ||
      !execution_device_pointer(
          reinterpret_cast<const void*>(arguments.assets.gate_up.payload.begin),
          device_ordinal) ||
      !execution_device_pointer(
          reinterpret_cast<const void*>(arguments.assets.down.payload.begin),
          device_ordinal) ||
      !exact_execution_ranges_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  sm87_bulk_v2_nvfp4_oracle_detail::RawTailArguments raw;
  raw.normalized_input = arguments.normalized_input;
  raw.gate_up_payload = reinterpret_cast<const std::uint8_t*>(
      arguments.assets.gate_up.payload.begin);
  raw.gate_tensor_scale = gate_scale;
  raw.up_tensor_scale = up_scale;
  raw.down_payload = reinterpret_cast<const std::uint8_t*>(
      arguments.assets.down.payload.begin);
  raw.down_tensor_scale = down_scale;
  raw.residual = arguments.residual;
  raw.group_h_scratch = arguments.group_h_scratch;
  raw.device_control = arguments.device_control;
  raw.cancellation_signal = arguments.cancellation_signal;
  raw.group_epoch = static_cast<std::uint32_t>(arguments.transaction_epoch);
  raw.token_count = segment.token_count;
  raw.cuda_stream = arguments.cuda_stream;
  (void)cudaGetLastError();
  return static_cast<int>(launch_raw_macro_unchecked(raw));
}

int launch_sm87_bulk_dataflow_v2_nvfp4_tail_numerical_cuda(
    const Sm87BulkV2NvFp4MacroArguments& arguments) noexcept {
  if (arguments.segment != kSm87BulkV2NvFp4FullMacroSegments) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return launch_sm87_bulk_dataflow_v2_nvfp4_macro_numerical_cuda(arguments);
}

int launch_sm87_bulk_dataflow_v2_nvfp4_p40_layer_cuda(
    const Sm87BulkV2NvFp4P40LayerArguments& arguments) noexcept {
  if (arguments.transaction_epoch == 0U ||
      arguments.transaction_epoch >
          std::numeric_limits<std::uint32_t>::max() -
              (kSm87BulkV2NvFp4SegmentsPerLayer - 1U) ||
      arguments.layer >= kSm87BulkV2NvFp4LayerCount ||
      arguments.normalized_input == nullptr || arguments.residual == nullptr ||
      arguments.group_h_scratch == nullptr ||
      arguments.device_control == nullptr || arguments.cuda_stream == nullptr ||
      !sm87_bulk_v2_nvfp4_assets_valid(arguments.assets)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  for (std::uint32_t segment = 0U;
       segment < kSm87BulkV2NvFp4SegmentsPerLayer; ++segment) {
    const auto plan = sm87_bulk_v2_nvfp4_segment_plan(segment);
    Sm87BulkV2NvFp4MacroArguments macro;
    macro.transaction_epoch = arguments.transaction_epoch + segment;
    macro.layer = arguments.layer;
    macro.segment = segment;
    macro.normalized_input =
        arguments.normalized_input +
        static_cast<std::size_t>(plan.first_token) * kHidden;
    macro.residual =
        arguments.residual +
        static_cast<std::size_t>(plan.first_token) * kHidden;
    macro.group_h_scratch = arguments.group_h_scratch;
    macro.device_control = arguments.device_control;
    macro.cancellation_signal = arguments.cancellation_signal;
    macro.assets = arguments.assets;
    macro.cuda_stream = arguments.cuda_stream;
    const int status =
        launch_sm87_bulk_dataflow_v2_nvfp4_macro_numerical_cuda(macro);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return static_cast<int>(cudaSuccess);
}

namespace sm87_bulk_v2_nvfp4_oracle_detail {

int launch_raw_tail(const RawTailArguments& arguments) noexcept {
  if (arguments.token_count != kSm87BulkV2NvFp4TailTokens) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return launch_raw_macro(arguments);
}

int launch_raw_macro(const RawTailArguments& arguments) noexcept {
  if (arguments.normalized_input == nullptr ||
      arguments.gate_up_payload == nullptr ||
      arguments.down_payload == nullptr || arguments.residual == nullptr ||
      arguments.group_h_scratch == nullptr ||
      arguments.device_control == nullptr || arguments.cuda_stream == nullptr ||
      arguments.group_epoch == 0U ||
      (arguments.token_count != kSm87BulkV2NvFp4TailTokens &&
       arguments.token_count != kSm87BulkV2NvFp4MacroTokens) ||
      !finite_positive(arguments.gate_tensor_scale) ||
      !finite_positive(arguments.up_tensor_scale) ||
      !finite_positive(arguments.down_tensor_scale)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaDeviceProp properties{};
  const cudaError_t device_status = validate_device(&properties);
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  (void)cudaGetLastError();
  return static_cast<int>(launch_raw_macro_unchecked(arguments));
}

}  // namespace sm87_bulk_v2_nvfp4_oracle_detail

}  // namespace q3x::kernels
