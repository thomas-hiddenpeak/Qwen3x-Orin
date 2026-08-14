#include "q3x/kernels/sm87_target_aot_projection_cuda.h"

#include "sm87_target_aot_projection_launch_internal.h"
#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_NVFP4_PROJECTION_ADMISSION)
#include "sm87_target_aot_projection_nvfp4_oracle_internal.h"
#endif
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
#include "../../runtime/sm87_target_aot_layer0_m192_oracle_internal.h"
#endif

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cmath>
#include <cstring>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87TargetAotProjectionThreads);
constexpr unsigned int kWarps =
    static_cast<unsigned int>(kSm87TargetAotProjectionWarps);
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87TargetAotProjectionBlockM);
constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87TargetAotProjectionBlockN);
constexpr unsigned int kTileK =
    static_cast<unsigned int>(kSm87TargetAotProjectionBlockK);
constexpr unsigned int kStages =
    static_cast<unsigned int>(kSm87TargetAotProjectionPipelineStages);
constexpr unsigned int kPersistentCtas =
    static_cast<unsigned int>(kSm87TargetAotProjectionPersistentCtas);
constexpr unsigned int kWarpM = 64U;
constexpr unsigned int kWarpN = 64U;
constexpr unsigned int kM16Panels = kWarpM / 16U;
constexpr unsigned int kN8Panels = kWarpN / 8U;
constexpr unsigned int kK16Panels = kTileK / 16U;
constexpr unsigned int kActivationVectors = kTileM * kTileK / 8U;
constexpr unsigned int kBranchWeightBytes = kTileN * kTileK / 4U;
constexpr unsigned int kBranchWeightVectors =
    kBranchWeightBytes / sizeof(uint4);
constexpr unsigned int kBranchScaleBytes = kTileN * (kTileK / 16U) / 2U;
constexpr unsigned int kBranchScaleVectors =
    kBranchScaleBytes / sizeof(uint4);
constexpr unsigned int kPackedCellWeightBytes =
    static_cast<unsigned int>(kSm87TargetAotNvFp4BBytesPerStage);
constexpr unsigned int kPackedCellScaleBytes =
    static_cast<unsigned int>(kSm87TargetAotNvFp4ScaleBytesPerStage);
constexpr unsigned int kPackedCellBytes =
    kPackedCellWeightBytes + kPackedCellScaleBytes;
constexpr unsigned int kGateUpKTiles =
    static_cast<unsigned int>(kSm87TargetAotProjectionHidden / kTileK);
constexpr unsigned int kGateUpNTiles =
    static_cast<unsigned int>(kSm87TargetAotProjectionIntermediate / kTileN);
constexpr unsigned int kDownKTiles =
    static_cast<unsigned int>(kSm87TargetAotProjectionIntermediate / kTileK);
constexpr unsigned int kDownNTiles =
    static_cast<unsigned int>(kSm87TargetAotProjectionHidden / kTileN);
constexpr std::uint64_t kGatePartitionBytes =
    static_cast<std::uint64_t>(kGateUpNTiles) * kGateUpKTiles *
    kPackedCellBytes;

static_assert(kThreads == kWarps * kWarpSize);
static_assert(kTileM == 128U && kTileN == 256U && kTileK == 64U);
static_assert(kStages == 3U && kPersistentCtas == 16U);
static_assert(kM16Panels == 4U && kN8Panels == 8U && kK16Panels == 4U);
static_assert(kActivationVectors == 1'024U);
static_assert(kBranchWeightBytes == 4'096U);
static_assert(kBranchWeightVectors == 256U);
static_assert(kBranchScaleBytes == 512U);
static_assert(kBranchScaleVectors == 32U);
static_assert(kGateUpKTiles == 80U && kGateUpNTiles == 68U);
static_assert(kDownKTiles == 272U && kDownNTiles == 20U);

// GateUp assigns one N128 half from each independent partition to the two
// branch regions. Down assigns the low/high N128 halves of its sole partition
// to the same regions. This is exactly A16KiB + 2*(B4KiB + scale512B) per
// stage, repeated three times.
struct alignas(32) NvFp4PipelineStorage final {
  uint4 activations[kStages][kTileM][kTileK / 8U];
  uint4 weights[kStages][2U][kBranchWeightVectors];
  uint4 scales[kStages][2U][kBranchScaleVectors];
};

static_assert(sizeof(NvFp4PipelineStorage) ==
              kSm87TargetAotNvFp4SharedBytes);

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

using WarpAccumulator = M16N8Accumulator[kM16Panels][kN8Panels];
using DecodedWeightStage = K16N8Weight[kN8Panels];

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
    const unsigned int warp_m, const unsigned int m16,
    const unsigned int k16, const unsigned int lane) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row = warp_m * kWarpM + m16 * 16U +
                           lane % 8U + (quadrant & 1U) * 8U;
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
  (void)warp_m;
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

[[nodiscard]] __device__ __forceinline__ float decode_e2m1(
    const std::uint8_t code) noexcept {
  const std::uint8_t magnitude = code & 0x07U;
  float value = 0.0F;
  switch (magnitude) {
    case 1U:
      value = 0.5F;
      break;
    case 2U:
      value = 1.0F;
      break;
    case 3U:
      value = 1.5F;
      break;
    case 4U:
      value = 2.0F;
      break;
    case 5U:
      value = 3.0F;
      break;
    case 6U:
      value = 4.0F;
      break;
    case 7U:
      value = 6.0F;
      break;
    case 0U:
    default:
      break;
  }
  return (code & 0x08U) != 0U ? -value : value;
}

[[nodiscard]] __device__ __forceinline__ float decode_e4m3fn_scale(
    const std::uint8_t code) noexcept {
  const unsigned int magnitude = code & 0x7fU;
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  float value = 0.0F;
  if (exponent == 0U) {
    value = ldexpf(static_cast<float>(mantissa), -9);
  } else {
    value = ldexpf(static_cast<float>(8U + mantissa),
                   static_cast<int>(exponent) - 10);
  }
  return (code & 0x80U) != 0U ? -value : value;
}

// The frozen persisted order is [K0,K8,K1,K9]: components [0,2] become the
// two BF16 halves of MMA register x0 and [1,3] become the halves of x1. Each
// product is rounded to BF16 before entering MMA; no decoded matrix is
// materialized in shared or global memory.
[[nodiscard]] __device__ __forceinline__ K16N8Weight decode_weight_fragment(
    const std::uint8_t* const shared_weight,
    const std::uint8_t* const shared_scale, const unsigned int k16,
    const unsigned int local_n_warp, const unsigned int n8_panel,
    const unsigned int lane) noexcept {
  const unsigned int fragment =
      (k16 * 2U + local_n_warp) * kN8Panels + n8_panel;
  const std::uint16_t packed = *reinterpret_cast<const std::uint16_t*>(
      shared_weight + fragment * 64U + lane * 2U);
  const std::uint8_t scale_code =
      shared_scale[fragment * 8U + lane / 4U];
  const float scale = decode_e4m3fn_scale(scale_code);
  const auto packed_component = [packed](const unsigned int component) {
    return static_cast<std::uint8_t>((packed >> (4U * component)) & 0x0fU);
  };
  const std::uint16_t component0 =
      encode_bf16_rne(decode_e2m1(packed_component(
                          sm87_target_aot_projection_mma_b_register_component(
                              0U))) *
                      scale);
  const std::uint16_t component1 =
      encode_bf16_rne(decode_e2m1(packed_component(
                          sm87_target_aot_projection_mma_b_register_component(
                              1U))) *
                      scale);
  const std::uint16_t component2 =
      encode_bf16_rne(decode_e2m1(packed_component(
                          sm87_target_aot_projection_mma_b_register_component(
                              2U))) *
                      scale);
  const std::uint16_t component3 =
      encode_bf16_rne(decode_e2m1(packed_component(
                          sm87_target_aot_projection_mma_b_register_component(
                              3U))) *
                      scale);
  // The persisted lane-component order is [K0, K8, K1, K9].  PTX
  // m16n8k16.col expects b.x0=[K0,K1] and b.x1=[K8,K9] for this lane, so the
  // helper above deliberately crosses the middle two persisted components.
  return {static_cast<std::uint32_t>(component0) |
              (static_cast<std::uint32_t>(component1) << 16U),
          static_cast<std::uint32_t>(component2) |
              (static_cast<std::uint32_t>(component3) << 16U)};
}

__device__ __forceinline__ void clear_accumulators(
    WarpAccumulator& accumulators) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      accumulators[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

template <bool kGateUp, unsigned int kKTiles>
__device__ __forceinline__ void issue_pipeline_stage(
    NvFp4PipelineStorage* const storage, const unsigned int slot,
    const std::uint16_t* const input, const std::uint8_t* const payload,
    const unsigned int rows, const unsigned int first_m,
    const unsigned int n_tile, const unsigned int n_half,
    const unsigned int k_tile) noexcept {
  constexpr unsigned int kInputFeatures =
      kGateUp ? static_cast<unsigned int>(kSm87TargetAotProjectionHidden)
              : static_cast<unsigned int>(
                    kSm87TargetAotProjectionIntermediate);

#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
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

#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int combined = threadIdx.x + pass * kThreads;
    const unsigned int branch = combined / kBranchWeightVectors;
    const unsigned int local_vector = combined % kBranchWeightVectors;
    const unsigned int vector_in_fragment = local_vector & 3U;
    const unsigned int fragment = local_vector >> 2U;
    const unsigned int n8_panel = fragment % kN8Panels;
    const unsigned int outer = fragment / kN8Panels;
    const unsigned int local_n_warp = outer & 1U;
    const unsigned int k16 = outer >> 1U;
    const unsigned int global_n_warp =
        kGateUp ? n_half * 2U + local_n_warp
                : branch * 2U + local_n_warp;
    const std::uint64_t partition_offset =
        kGateUp && branch == 1U ? kGatePartitionBytes : 0U;
    const std::uint64_t cell =
        partition_offset +
        (static_cast<std::uint64_t>(n_tile) * kKTiles + k_tile) *
            kPackedCellBytes;
    const std::uint64_t global_fragment =
        (static_cast<std::uint64_t>(k16) * 4U + global_n_warp) *
            kN8Panels +
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
    const unsigned int global_n_warp =
        kGateUp ? n_half * 2U + local_n_warp
                : branch * 2U + local_n_warp;
    const std::uint64_t partition_offset =
        kGateUp && branch == 1U ? kGatePartitionBytes : 0U;
    const std::uint64_t cell =
        partition_offset +
        (static_cast<std::uint64_t>(n_tile) * kKTiles + k_tile) *
            kPackedCellBytes;
    const std::uint64_t scale_byte =
        (static_cast<std::uint64_t>(k16) * 4U + global_n_warp) * 64U +
        n8_pair * 16U;
    const auto* const source = reinterpret_cast<const uint4*>(
        payload + cell + kPackedCellWeightBytes + scale_byte);
    cp_async_cg_16<false>(
        &storage->scales[slot][branch][local_vector], source);
  }
  cp_async_commit_group();
}

template <bool kGateUp, unsigned int kKTiles>
__device__ __forceinline__ void run_full_k(
    NvFp4PipelineStorage* const storage, const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int rows,
    const unsigned int first_m, const unsigned int n_tile,
    const unsigned int n_half, WarpAccumulator& accumulators) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_n = warp % 4U;
  const unsigned int branch = kGateUp ? warp / 4U : warp_n / 2U;
  const unsigned int branch_warp = warp % 4U;
  const unsigned int warp_m = kGateUp ? branch_warp / 2U : warp / 4U;
  const unsigned int local_n_warp =
      kGateUp ? branch_warp % 2U : warp_n % 2U;

  clear_accumulators(accumulators);
  issue_pipeline_stage<kGateUp, kKTiles>(
      storage, 0U, input, payload, rows, first_m, n_tile, n_half, 0U);
  issue_pipeline_stage<kGateUp, kKTiles>(
      storage, 1U, input, payload, rows, first_m, n_tile, n_half, 1U);
  issue_pipeline_stage<kGateUp, kKTiles>(
      storage, 2U, input, payload, rows, first_m, n_tile, n_half, 2U);

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
            storage->activations[slot]);
    const auto* const shared_weight = reinterpret_cast<const std::uint8_t*>(
        storage->weights[slot][branch]);
    const auto* const shared_scale = reinterpret_cast<const std::uint8_t*>(
        storage->scales[slot][branch]);

    DecodedWeightStage decoded[2U];
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      decoded[0U][n8] = decode_weight_fragment(
          shared_weight, shared_scale, 0U, local_n_warp, n8, lane);
    }

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      const unsigned int current = k16 & 1U;
      const unsigned int next = current ^ 1U;
      if (k16 + 1U < kK16Panels) {
#pragma unroll
        for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
          decoded[next][n8] = decode_weight_fragment(
              shared_weight, shared_scale, k16 + 1U, local_n_warp, n8,
              lane);
        }
      }

      M16K16Activation activation[kM16Panels];
#pragma unroll
      for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
        load_activation_fragment(activation[m16], shared_activations,
                                 warp_m, m16, k16, lane);
      }

      // K16=3 is the last use of this shared slot.  Once every warp has
      // captured its final A and decoded-B fragments in registers, the slot
      // is dead even though the final MMA has not retired.  Refill it here so
      // global-to-shared movement overlaps that MMA instead of starting only
      // after the complete stage drains.  The next iteration's wait+barrier
      // protects both the arriving stage and the retiring warp-local MMA.
      if (k16 + 1U == kK16Panels) {
        __syncthreads();
        if (k_tile + kStages < kKTiles) {
          issue_pipeline_stage<kGateUp, kKTiles>(
              storage, slot, input, payload, rows, first_m, n_tile, n_half,
              k_tile + kStages);
        }
      }
#pragma unroll
      for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
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

__device__ __forceinline__ std::uint32_t pack_scaled_bf16_pair(
    const float low, const float high, const float tensor_scale) noexcept {
  return static_cast<std::uint32_t>(
             encode_bf16_rne(low * tensor_scale)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(high * tensor_scale))
          << 16U);
}

__device__ __forceinline__ void publish_branch_to_shared(
    std::uint16_t* const temporary, const WarpAccumulator& accumulators,
    const unsigned int branch, const unsigned int warp_m,
    const unsigned int local_n_warp, const unsigned int lane,
    const float tensor_scale) noexcept {
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
    const unsigned int row0 = warp_m * kWarpM + m16 * 16U + lane_group;
    const unsigned int row1 = row0 + 8U;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      const unsigned int column =
          local_n_warp * kWarpN + n8 * 8U + lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      *reinterpret_cast<std::uint32_t*>(
          temporary + branch * kTileM * (kTileN / 2U) +
          row0 * (kTileN / 2U) + column) =
          pack_scaled_bf16_pair(value.x0, value.x1, tensor_scale);
      *reinterpret_cast<std::uint32_t*>(
          temporary + branch * kTileM * (kTileN / 2U) +
          row1 * (kTileN / 2U) + column) =
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

__device__ __forceinline__ void consume_gate_up_shared(
    const std::uint16_t* const temporary, const unsigned int rows,
    const unsigned int first_m, const unsigned int first_n,
    const unsigned int warp, const unsigned int lane,
    std::uint16_t* const output) noexcept {
  if (warp >= 4U) {
    return;
  }
  const unsigned int warp_m = warp / 2U;
  const unsigned int local_n_warp = warp % 2U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
    const unsigned int local_row0 =
        warp_m * kWarpM + m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = first_m + local_row0;
    const unsigned int global_row1 = first_m + local_row1;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      const unsigned int local_column =
          local_n_warp * kWarpN + n8 * 8U + lane_in_group * 2U;
      const unsigned int global_column = first_n + local_column;
      const std::uint32_t gate0 = *reinterpret_cast<const std::uint32_t*>(
          temporary + local_row0 * (kTileN / 2U) + local_column);
      const std::uint32_t up0 = *reinterpret_cast<const std::uint32_t*>(
          temporary + kTileM * (kTileN / 2U) +
          local_row0 * (kTileN / 2U) + local_column);
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
                         kSm87TargetAotProjectionIntermediate +
            global_column) = result;
      }
      const std::uint32_t gate1 = *reinterpret_cast<const std::uint32_t*>(
          temporary + local_row1 * (kTileN / 2U) + local_column);
      const std::uint32_t up1 = *reinterpret_cast<const std::uint32_t*>(
          temporary + kTileM * (kTileN / 2U) +
          local_row1 * (kTileN / 2U) + local_column);
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
                         kSm87TargetAotProjectionIntermediate +
            global_column) = result;
      }
    }
  }
}

__device__ __forceinline__ std::uint32_t add_residual_pair(
    const std::uint32_t branch_bits,
    const std::uint32_t residual_bits) noexcept {
  const float branch0 =
      decode_bf16(static_cast<std::uint16_t>(branch_bits));
  const float branch1 =
      decode_bf16(static_cast<std::uint16_t>(branch_bits >> 16U));
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

__device__ __forceinline__ void publish_down_residual(
    const WarpAccumulator& accumulators, const unsigned int rows,
    const unsigned int first_m, const unsigned int first_n,
    const unsigned int warp, const unsigned int lane,
    const float tensor_scale, std::uint16_t* const residual) noexcept {
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
    const unsigned int local_row0 =
        warp_m * kWarpM + m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = first_m + local_row0;
    const unsigned int global_row1 = first_m + local_row1;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      const unsigned int column =
          first_n + warp_n * kWarpN + n8 * 8U + lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      if (global_row0 < rows) {
        auto* const destination = reinterpret_cast<std::uint32_t*>(
            residual + static_cast<std::size_t>(global_row0) *
                           kSm87TargetAotProjectionHidden +
            column);
        *destination = add_residual_pair(
            pack_scaled_bf16_pair(value.x0, value.x1, tensor_scale),
            *destination);
      }
      if (global_row1 < rows) {
        auto* const destination = reinterpret_cast<std::uint32_t*>(
            residual + static_cast<std::size_t>(global_row1) *
                           kSm87TargetAotProjectionHidden +
            column);
        *destination = add_residual_pair(
            pack_scaled_bf16_pair(value.x2, value.x3, tensor_scale),
            *destination);
      }
    }
  }
}

__device__ __forceinline__ void projection_task(
    const unsigned int linear_task, const unsigned int grid_m,
    const unsigned int grid_n, const unsigned int raster_group_m,
    const bool n_stationary,
    unsigned int& m_tile, unsigned int& n_tile) noexcept {
  if (n_stationary) {
    m_tile = linear_task % grid_m;
    n_tile = linear_task / grid_m;
    return;
  }
  const unsigned int group_span = raster_group_m * grid_n;
  const unsigned int group = linear_task / group_span;
  const unsigned int first_m = group * raster_group_m;
  const unsigned int remaining_m = grid_m - first_m;
  const unsigned int active_m =
      remaining_m < raster_group_m ? remaining_m : raster_group_m;
  const unsigned int group_offset = linear_task % group_span;
  n_tile = group_offset / active_m;
  m_tile = first_m + group_offset % active_m;
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_target_aot_nvfp4_gate_up_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const unsigned int rows,
    const float gate_tensor_scale, const float up_tensor_scale,
    std::uint16_t* __restrict__ output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<NvFp4PipelineStorage*>(dynamic_storage);
  const unsigned int grid_m = (rows + kTileM - 1U) / kTileM;
  constexpr unsigned int kGridN = kGateUpNTiles;
  const unsigned int logical_tasks = grid_m * kGridN;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int branch = warp / 4U;
  const unsigned int branch_warp = warp % 4U;
  const unsigned int warp_m = branch_warp / 2U;
  const unsigned int local_n_warp = branch_warp % 2U;

  for (unsigned int linear_task = blockIdx.x;
       linear_task < logical_tasks; linear_task += kPersistentCtas) {
    unsigned int m_tile = 0U;
    unsigned int n_tile = 0U;
    projection_task(linear_task, grid_m, kGridN, 2U, false, m_tile,
                    n_tile);
    const unsigned int first_m = m_tile * kTileM;

#pragma unroll
    for (unsigned int n_half = 0U; n_half < 2U; ++n_half) {
      WarpAccumulator accumulators;
      run_full_k<true, kGateUpKTiles>(
          storage, input, payload, rows, first_m, n_tile, n_half,
          accumulators);

      // All asynchronous stages are drained. Reuse the first 64KiB of the
      // dead 75KiB pipeline as two CTA-private M128xN128 BF16 publications.
      auto* const temporary =
          reinterpret_cast<std::uint16_t*>(dynamic_storage);
      publish_branch_to_shared(
          temporary, accumulators, branch, warp_m, local_n_warp, lane,
          branch == 0U ? gate_tensor_scale : up_tensor_scale);
      __syncthreads();
      consume_gate_up_shared(
          temporary, rows, first_m, n_tile * kTileN + n_half * (kTileN / 2U),
          warp, lane, output);
      // Gate/Up temporaries are not reclaimed until every consuming warp has
      // completed SiLU*Up; the next half may then recreate the pipeline.
      __syncthreads();
    }
  }
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_target_aot_nvfp4_down_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const unsigned int rows,
    const float tensor_scale, std::uint16_t* __restrict__ residual) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<NvFp4PipelineStorage*>(dynamic_storage);
  const unsigned int grid_m = (rows + kTileM - 1U) / kTileM;
  constexpr unsigned int kGridN = kDownNTiles;
  const unsigned int logical_tasks = grid_m * kGridN;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;

  for (unsigned int linear_task = blockIdx.x;
       linear_task < logical_tasks; linear_task += kPersistentCtas) {
    unsigned int m_tile = 0U;
    unsigned int n_tile = 0U;
    projection_task(linear_task, grid_m, kGridN, 1U, true, m_tile, n_tile);
    const unsigned int first_m = m_tile * kTileM;
    WarpAccumulator accumulators;
    run_full_k<false, kDownKTiles>(storage, input, payload, rows, first_m,
                                  n_tile, 0U, accumulators);
    publish_down_residual(accumulators, rows, first_m, n_tile * kTileN,
                          warp, lane, tensor_scale, residual);
    __syncthreads();
  }
}

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
__device__ __forceinline__ void publish_down_residual_to_oracle_output(
    const WarpAccumulator& accumulators, const unsigned int rows,
    const unsigned int first_m, const unsigned int first_n,
    const unsigned int warp, const unsigned int lane,
    const float tensor_scale, const std::uint16_t* const residual,
    std::uint16_t* const output) noexcept {
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kM16Panels; ++m16) {
    const unsigned int local_row0 =
        warp_m * kWarpM + m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = first_m + local_row0;
    const unsigned int global_row1 = first_m + local_row1;
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kN8Panels; ++n8) {
      const unsigned int column =
          first_n + warp_n * kWarpN + n8 * 8U + lane_in_group * 2U;
      const auto& value = accumulators[m16][n8];
      if (global_row0 < rows) {
        const auto* const source = reinterpret_cast<const std::uint32_t*>(
            residual + static_cast<std::size_t>(global_row0) *
                           kSm87TargetAotProjectionHidden + column);
        auto* const destination = reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row0) *
                         kSm87TargetAotProjectionHidden + column);
        *destination = add_residual_pair(
            pack_scaled_bf16_pair(value.x0, value.x1, tensor_scale), *source);
      }
      if (global_row1 < rows) {
        const auto* const source = reinterpret_cast<const std::uint32_t*>(
            residual + static_cast<std::size_t>(global_row1) *
                           kSm87TargetAotProjectionHidden + column);
        auto* const destination = reinterpret_cast<std::uint32_t*>(
            output + static_cast<std::size_t>(global_row1) *
                         kSm87TargetAotProjectionHidden + column);
        *destination = add_residual_pair(
            pack_scaled_bf16_pair(value.x2, value.x3, tensor_scale), *source);
      }
    }
  }
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_target_aot_nvfp4_down_m192_oracle_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const unsigned int rows,
    const float tensor_scale,
    const std::uint16_t* __restrict__ residual,
    std::uint16_t* __restrict__ output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<NvFp4PipelineStorage*>(dynamic_storage);
  const unsigned int grid_m = (rows + kTileM - 1U) / kTileM;
  constexpr unsigned int kGridN = kDownNTiles;
  const unsigned int logical_tasks = grid_m * kGridN;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;

  for (unsigned int linear_task = blockIdx.x;
       linear_task < logical_tasks; linear_task += kPersistentCtas) {
    unsigned int m_tile = 0U;
    unsigned int n_tile = 0U;
    projection_task(linear_task, grid_m, kGridN, 1U, true, m_tile,
                    n_tile);
    const unsigned int first_m = m_tile * kTileM;
    WarpAccumulator accumulators;
    run_full_k<false, kDownKTiles>(storage, input, payload, rows, first_m,
                                  n_tile, 0U, accumulators);
    publish_down_residual_to_oracle_output(
        accumulators, rows, first_m, n_tile * kTileN, warp, lane,
        tensor_scale, residual, output);
    __syncthreads();
  }
}
#endif

[[nodiscard]] cudaError_t query_kernel_resources(
    const Sm87TargetAotProjectionRole role,
    Sm87TargetAotNvFp4CudaResources* const resources) noexcept {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaErrorInvalidValue;
  if (role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    status = cudaFuncGetAttributes(
        &attributes, sm87_target_aot_nvfp4_gate_up_kernel);
  } else if (role == Sm87TargetAotProjectionRole::kNvFp4Down) {
    status = cudaFuncGetAttributes(&attributes,
                                   sm87_target_aot_nvfp4_down_kernel);
  }
  if (status != cudaSuccess) {
    return status;
  }
  resources->role = role;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kSm87TargetAotNvFp4SharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->kernel_compiled = true;
  // Observation is not admission. These remain false until the normative
  // plan/layout are updated after numerical and static-resource evidence.
  resources->static_resources_qualified = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return cudaSuccess;
}

[[nodiscard]] cudaError_t set_execution_dynamic_shared_attribute(
    const Sm87TargetAotProjectionRole role) noexcept {
  if (role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    return cudaFuncSetAttribute(
        sm87_target_aot_nvfp4_gate_up_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSm87TargetAotNvFp4SharedBytes));
  }
  if (role == Sm87TargetAotProjectionRole::kNvFp4Down) {
    return cudaFuncSetAttribute(
        sm87_target_aot_nvfp4_down_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSm87TargetAotNvFp4SharedBytes));
  }
  return cudaErrorInvalidValue;
}

[[nodiscard]] cudaError_t validate_execution_device() noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties.major == 8 && properties.minor == 7 &&
                 properties.multiProcessorCount ==
                     static_cast<int>(kSm87TargetAotProjectionSmCount) &&
                 properties.sharedMemPerBlockOptin >=
                     kSm87TargetAotNvFp4SharedBytes
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool execution_device_pointer(
    const void* const pointer, const int device_ordinal) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device_ordinal;
}

[[nodiscard]] int launch_authenticated_nvfp4_body(
    const Sm87TargetAotNvFp4CudaArguments& arguments) noexcept {
  if (!sm87_target_aot_nvfp4_cuda_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t device_status = validate_execution_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  const auto& upload = arguments.asset.device_upload_receipt;
  int current_device = -1;
  const cudaError_t current_status = cudaGetDevice(&current_device);
  if (current_status != cudaSuccess) {
    return static_cast<int>(current_status);
  }
  if (current_device != upload.device_ordinal ||
      !execution_device_pointer(
          reinterpret_cast<const void*>(arguments.asset.payload.begin),
          upload.device_ordinal) ||
      !execution_device_pointer(arguments.input, upload.device_ordinal) ||
      !execution_device_pointer(arguments.output_or_residual,
                                upload.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  float scales[2U]{};
  for (std::size_t index = 0U; index < arguments.asset.tensor_scale_count;
       ++index) {
    std::memcpy(&scales[index], &arguments.asset.tensor_scale_bits[index],
                sizeof(float));
    if (!std::isfinite(scales[index]) || scales[index] <= 0.0F) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  const cudaError_t attribute_status =
      set_execution_dynamic_shared_attribute(arguments.role);
  if (attribute_status != cudaSuccess) {
    return static_cast<int>(attribute_status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const auto* const payload = reinterpret_cast<const std::uint8_t*>(
      arguments.asset.payload.begin);
  (void)cudaGetLastError();
  if (arguments.role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    sm87_target_aot_nvfp4_gate_up_kernel
        <<<kPersistentCtas, kThreads, kSm87TargetAotNvFp4SharedBytes,
           stream>>>(arguments.input, payload,
                     static_cast<unsigned int>(arguments.token_count),
                     scales[0U], scales[1U], arguments.output_or_residual);
  } else {
    sm87_target_aot_nvfp4_down_kernel
        <<<kPersistentCtas, kThreads, kSm87TargetAotNvFp4SharedBytes,
           stream>>>(arguments.input, payload,
                     static_cast<unsigned int>(arguments.token_count),
                     scales[0U], arguments.output_or_residual);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_target_aot_nvfp4_cuda_resources(
    const Sm87TargetAotProjectionRole role, const std::size_t token_count,
    Sm87TargetAotNvFp4CudaResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  const auto plan = sm87_target_aot_projection_plan(role, token_count);
  if (!plan.valid() || !sm87_target_aot_nvfp4_cuda_role(role)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(query_kernel_resources(role, resources));
}

int launch_sm87_target_aot_nvfp4_cuda(
    const Sm87TargetAotNvFp4CudaArguments& arguments) noexcept {
  if (!sm87_target_aot_nvfp4_cuda_arguments_valid(arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  // This v1 symbol is an immutable compile/fail-closed sentinel. The host
  // receipt schema does not query pointer residency or authenticate device
  // bytes. A future executable admission must use a separately reviewed
  // launcher with the real uploader, device oracle, guard/tail, numerical,
  // and resource evidence; changing plan/layout bits never opens this symbol.
  return static_cast<int>(cudaErrorNotSupported);
}

namespace sm87_target_aot_projection_execution_detail {

int launch_authenticated_nvfp4(
    const Sm87TargetAotNvFp4CudaArguments& arguments) noexcept {
  return launch_authenticated_nvfp4_body(arguments);
}

}  // namespace sm87_target_aot_projection_execution_detail

#if defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_NVFP4_PROJECTION_ADMISSION)
namespace sm87_target_aot_nvfp4_oracle_detail {

int launch_raw_v1(const RawV1Arguments& arguments) noexcept {
  const bool gate_up = arguments.role ==
                       Sm87TargetAotProjectionRole::kNvFp4GateUp;
  const bool down =
      arguments.role == Sm87TargetAotProjectionRole::kNvFp4Down;
  if ((!gate_up && !down) || arguments.input == nullptr ||
      arguments.payload == nullptr ||
      (arguments.token_count != 64U && arguments.token_count != 1'024U) ||
      !std::isfinite(arguments.tensor_scale0) ||
      arguments.tensor_scale0 <= 0.0F ||
      (gate_up && (!std::isfinite(arguments.tensor_scale1) ||
                   arguments.tensor_scale1 <= 0.0F)) ||
      arguments.output_or_residual == nullptr ||
      arguments.cuda_stream == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t device_status = validate_execution_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  int device_ordinal = -1;
  cudaError_t status = cudaGetDevice(&device_ordinal);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (!execution_device_pointer(arguments.input, device_ordinal) ||
      !execution_device_pointer(arguments.payload, device_ordinal) ||
      !execution_device_pointer(arguments.output_or_residual,
                                device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  status = set_execution_dynamic_shared_attribute(arguments.role);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  (void)cudaGetLastError();
  if (gate_up) {
    sm87_target_aot_nvfp4_gate_up_kernel
        <<<kPersistentCtas, kThreads, kSm87TargetAotNvFp4SharedBytes,
           stream>>>(arguments.input, arguments.payload,
                     static_cast<unsigned int>(arguments.token_count),
                     arguments.tensor_scale0, arguments.tensor_scale1,
                     arguments.output_or_residual);
  } else {
    sm87_target_aot_nvfp4_down_kernel
        <<<kPersistentCtas, kThreads, kSm87TargetAotNvFp4SharedBytes,
           stream>>>(arguments.input, arguments.payload,
                     static_cast<unsigned int>(arguments.token_count),
                     arguments.tensor_scale0,
                     arguments.output_or_residual);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace sm87_target_aot_nvfp4_oracle_detail
#endif

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_LAYER0_M192_ORACLE_ADMISSION)
namespace sm87_target_aot_layer0_m192_oracle_detail {

namespace {

[[nodiscard]] cudaError_t set_oracle_dynamic_shared_attribute(
    const Sm87TargetAotProjectionRole role) noexcept {
  if (role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    return cudaFuncSetAttribute(
        sm87_target_aot_nvfp4_gate_up_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSm87TargetAotNvFp4SharedBytes));
  }
  if (role == Sm87TargetAotProjectionRole::kNvFp4Down) {
    return cudaFuncSetAttribute(
        sm87_target_aot_nvfp4_down_m192_oracle_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kSm87TargetAotNvFp4SharedBytes));
  }
  return cudaErrorInvalidValue;
}

[[nodiscard]] bool exact_device_pointer(const void* const pointer,
                                        const int device_ordinal) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device_ordinal;
}

[[nodiscard]] bool fixed_m192_arguments_valid(
    const Sm87TargetAotProjectionRole role,
    const std::uint16_t* const input,
    const Sm87TargetAotNvFp4CudaAssetView& asset,
    const std::uint16_t* const residual,
    std::uint16_t* const output) noexcept {
  if (!sm87_target_aot_nvfp4_cuda_role(role) ||
      !sm87_target_aot_nvfp4_cuda_asset_valid(asset) ||
      asset.payload.role != role || input == nullptr ||
      output == nullptr ||
      (role == Sm87TargetAotProjectionRole::kNvFp4GateUp &&
       residual != nullptr) ||
      (role == Sm87TargetAotProjectionRole::kNvFp4Down &&
       residual == nullptr) ||
      reinterpret_cast<std::uintptr_t>(input) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(output) % 16U != 0U ||
      (residual != nullptr &&
       reinterpret_cast<std::uintptr_t>(residual) % 16U != 0U)) {
    return false;
  }
  const std::uint64_t input_features =
      role == Sm87TargetAotProjectionRole::kNvFp4GateUp
          ? kSm87TargetAotProjectionHidden
          : kSm87TargetAotProjectionIntermediate;
  const std::uint64_t output_features =
      role == Sm87TargetAotProjectionRole::kNvFp4GateUp
          ? kSm87TargetAotProjectionIntermediate
          : kSm87TargetAotProjectionHidden;
  const auto input_range = sm87_target_aot_nvfp4_cuda_byte_range(
      input, kTokenCount * input_features * sizeof(std::uint16_t));
  const auto output_range = sm87_target_aot_nvfp4_cuda_byte_range(
      output,
      kTokenCount * output_features * sizeof(std::uint16_t));
  const auto residual_range =
      residual == nullptr
          ? Sm87TargetAotNvFp4CudaByteRange{}
          : sm87_target_aot_nvfp4_cuda_byte_range(
                residual, kTokenCount * kSm87TargetAotProjectionHidden *
                              sizeof(std::uint16_t));
  const auto& upload = asset.device_upload_receipt;
  const Sm87TargetAotNvFp4CudaByteRange allocation_range{
      upload.device_allocation_begin, upload.device_allocation_end,
      upload.device_allocation_begin != 0U &&
          upload.device_allocation_end > upload.device_allocation_begin};
  if (!input_range.valid || !output_range.valid || !allocation_range.valid ||
      (residual != nullptr && !residual_range.valid) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(input_range, output_range) ||
      (residual != nullptr &&
       (sm87_target_aot_nvfp4_cuda_ranges_overlap(input_range,
                                                  residual_range) ||
        sm87_target_aot_nvfp4_cuda_ranges_overlap(output_range,
                                                  residual_range))) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(input_range,
                                                allocation_range) ||
      sm87_target_aot_nvfp4_cuda_ranges_overlap(output_range,
                                                allocation_range) ||
      (residual != nullptr &&
       sm87_target_aot_nvfp4_cuda_ranges_overlap(residual_range,
                                                allocation_range))) {
    return false;
  }

  int active_device = -1;
  if (cudaGetDevice(&active_device) != cudaSuccess ||
      active_device != asset.device_upload_receipt.device_ordinal ||
      !exact_device_pointer(input, active_device) ||
      !exact_device_pointer(output, active_device) ||
      (residual != nullptr &&
       !exact_device_pointer(residual, active_device)) ||
      !exact_device_pointer(
          reinterpret_cast<const void*>(asset.payload.begin),
          active_device)) {
    return false;
  }
  for (std::size_t index = 0U; index < asset.tensor_scale_count; ++index) {
    float scale = 0.0F;
    std::memcpy(&scale, &asset.tensor_scale_bits[index], sizeof(scale));
    if (!std::isfinite(scale) || scale <= 0.0F) {
      return false;
    }
  }
  return true;
}

}  // namespace

int query_resources(const Sm87TargetAotProjectionRole role,
                    KernelResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  if (!sm87_target_aot_nvfp4_cuda_role(role)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = set_oracle_dynamic_shared_attribute(role);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  if (role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    status = cudaFuncGetAttributes(
        &attributes, sm87_target_aot_nvfp4_gate_up_kernel);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, sm87_target_aot_nvfp4_gate_up_kernel, kThreads,
          kSm87TargetAotNvFp4SharedBytes);
    }
  } else {
    status = cudaFuncGetAttributes(
        &attributes, sm87_target_aot_nvfp4_down_m192_oracle_kernel);
    if (status == cudaSuccess) {
      status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
          &active_blocks, sm87_target_aot_nvfp4_down_m192_oracle_kernel,
          kThreads,
          kSm87TargetAotNvFp4SharedBytes);
    }
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  int device = -1;
  cudaDeviceProp properties{};
  status = cudaGetDevice(&device);
  if (status == cudaSuccess) {
    status = cudaGetDeviceProperties(&properties, device);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->role = role;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kSm87TargetAotNvFp4SharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->physical_ctas = kPersistentCtas;
  resources->device_ordinal = device;
  resources->device_major = properties.major;
  resources->device_minor = properties.minor;
  resources->device_sm_count = properties.multiProcessorCount;
  return static_cast<int>(cudaSuccess);
}

int launch(const Sm87TargetAotProjectionRole role,
           const std::uint16_t* const input,
           const Sm87TargetAotNvFp4CudaAssetView& authenticated_asset,
           const std::uint16_t* const residual,
           std::uint16_t* const output,
           void* const cuda_stream) noexcept {
  if (!fixed_m192_arguments_valid(role, input, authenticated_asset, residual,
                                  output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = set_oracle_dynamic_shared_attribute(role);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  float scale0 = 0.0F;
  float scale1 = 0.0F;
  std::memcpy(&scale0, &authenticated_asset.tensor_scale_bits[0U],
              sizeof(scale0));
  if (authenticated_asset.tensor_scale_count == 2U) {
    std::memcpy(&scale1, &authenticated_asset.tensor_scale_bits[1U],
                sizeof(scale1));
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  const auto* const payload = reinterpret_cast<const std::uint8_t*>(
      authenticated_asset.payload.begin);
  (void)cudaGetLastError();
  if (role == Sm87TargetAotProjectionRole::kNvFp4GateUp) {
    sm87_target_aot_nvfp4_gate_up_kernel
        <<<kPersistentCtas, kThreads, kSm87TargetAotNvFp4SharedBytes,
           stream>>>(input, payload, static_cast<unsigned int>(kTokenCount),
                     scale0, scale1, output);
  } else {
    sm87_target_aot_nvfp4_down_m192_oracle_kernel
        <<<kPersistentCtas, kThreads, kSm87TargetAotNvFp4SharedBytes,
           stream>>>(input, payload, static_cast<unsigned int>(kTokenCount),
                     scale0, residual, output);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // namespace sm87_target_aot_layer0_m192_oracle_detail
#endif

}  // namespace q3x::kernels
