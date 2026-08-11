#include "q3x/kernels/sm87_p40_packed_nvfp4_v2.h"

#include "p40_packed_projection_common.cuh"
#include "third_party/vllm_marlin/marlin_template.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

using namespace p40_packed_detail;

#if defined(Q3X_ENABLE_P40_PACKED_NVFP4_V2_ADMISSION)
inline constexpr bool kV2Admitted = true;
#else
inline constexpr bool kV2Admitted = false;
#endif

using Bf16MarlinType = marlin::MarlinScalarType<vllm::kBFloat16.id()>;
using FragA = typename Bf16MarlinType::FragA;
using FragB = typename Bf16MarlinType::FragB;
using FragC = typename Bf16MarlinType::FragC;

inline constexpr unsigned int kThreads =
    kSm87P40PackedNvFp4V2Threads;
inline constexpr unsigned int kWarps = kSm87P40PackedNvFp4V2Warps;
inline constexpr unsigned int kTileM = kSm87P40PackedNvFp4V2TileM;
inline constexpr unsigned int kKBlock = kSm87P40PackedNvFp4V2TileK;
inline constexpr unsigned int kK64PerBlock = kKBlock / 64U;
inline constexpr unsigned int kStages =
    kSm87P40PackedNvFp4V2PipelineStages;
inline constexpr unsigned int kM16Panels = kTileM / 16U;
inline constexpr unsigned int kN8PanelsPerWarp = 4U;
inline constexpr unsigned int kGateBranchWarps = 4U;
inline constexpr unsigned int kGateTileN = 128U;
inline constexpr unsigned int kGateGridN =
    kSm87P40PackedNvFp4V2GateGridN;
inline constexpr unsigned int kGateGroupM = 2U;
inline constexpr unsigned int kGateK64Tiles =
    kSm87P40PackedProjectionHidden / 64U;
inline constexpr unsigned int kDownTileN = 256U;
inline constexpr unsigned int kDownGridN =
    kSm87P40PackedNvFp4V2DownGridN;
inline constexpr unsigned int kDownK64Tiles =
    kSm87P40PackedProjectionIntermediate / 64U;
inline constexpr unsigned int kGridM =
    kSm87P40PackedNvFp4V2GridM;

inline constexpr unsigned int kFragmentWeightBytes = 256U;
inline constexpr unsigned int kFragmentScaleBytes = 32U;
inline constexpr unsigned int kFragmentBytes =
    kFragmentWeightBytes + kFragmentScaleBytes;
inline constexpr unsigned int kGatePhysicalWarps = 8U;
inline constexpr unsigned int kDownPhysicalWarps = 4U;
inline constexpr unsigned int kGateCellBytes =
    4U * kGatePhysicalWarps * kFragmentBytes;
inline constexpr unsigned int kDownCellBytes =
    4U * kDownPhysicalWarps * kFragmentBytes;
inline constexpr unsigned int kGateCellVectors = kGateCellBytes / 16U;
inline constexpr unsigned int kDownCellVectors = kDownCellBytes / 16U;
inline constexpr unsigned int kCombinedCellVectors =
    2U * kDownCellVectors;
inline constexpr unsigned int kA64Vectors = kTileM * 64U / 8U;
inline constexpr unsigned int kAStageVectors =
    kK64PerBlock * kA64Vectors;
inline constexpr unsigned int kBStageVectors =
    kK64PerBlock * kGateCellVectors;
inline constexpr unsigned int kMaximumRegistersPerThread = 224U;
inline constexpr unsigned int kMinimumBlocksPerSm = 1U;
inline constexpr std::size_t kSharedLimitBytes = 160U * 1'024U;
inline constexpr unsigned int kGateResourceReady = 1U << 0U;
inline constexpr unsigned int kDownResourceReady = 1U << 1U;

// Set only by the binding-time resource query after function configuration,
// occupancy, spill, register, and shared-memory admission all pass.  The
// production request path consumes this sealed fact rather than rediscovering
// kernel resources once per model layer.
std::atomic<unsigned int> g_resource_ready_mask{0U};

struct alignas(32) SharedStorage {
  uint4 activations[kStages][kK64PerBlock][kA64Vectors];
  // Gate stores one Gate128+Up128 cell per K64. Down stores two adjacent
  // N128 cells per K64. Both are exactly 9,216 bytes per K64.
  uint4 packed_operand[kStages][kK64PerBlock][kGateCellVectors];
};

inline constexpr std::size_t kDynamicSharedBytes = sizeof(SharedStorage);

static_assert(kThreads == 256U);
static_assert(kWarps == 8U);
static_assert(kTileM == 128U);
static_assert(kKBlock == 128U);
static_assert(kK64PerBlock == 2U);
static_assert(kStages == 3U);
static_assert(kM16Panels == 8U);
static_assert(kGateK64Tiles == 80U);
static_assert(kDownK64Tiles == 272U);
static_assert(kGateCellBytes == 9'216U);
static_assert(kDownCellBytes == 4'608U);
static_assert(kGateCellVectors == kCombinedCellVectors);
static_assert(kAStageVectors == 2'048U);
static_assert(kBStageVectors == 1'152U);
static_assert(kDynamicSharedBytes ==
              kSm87P40PackedNvFp4V2DynamicSharedBytes);
static_assert(kDynamicSharedBytes <= kSharedLimitBytes);
static_assert(kGridM * kTileM ==
              kSm87P40PackedProjectionTokens + 64U);

struct DecodedBStage {
  FragB fragments[kN8PanelsPerWarp];
};

[[nodiscard]] __device__ __forceinline__ unsigned int
gate_m_tile_from_task(const unsigned int task) noexcept {
  constexpr unsigned int kGroupSpan = kGateGroupM * kGateGridN;
  const unsigned int group = task / kGroupSpan;
  const unsigned int first_m = group * kGateGroupM;
  const unsigned int active_m =
      kGridM - first_m < kGateGroupM ? kGridM - first_m
                                    : kGateGroupM;
  return first_m + (task % kGroupSpan) % active_m;
}

[[nodiscard]] __device__ __forceinline__ unsigned int
gate_n_tile_from_task(const unsigned int task) noexcept {
  constexpr unsigned int kGroupSpan = kGateGroupM * kGateGridN;
  const unsigned int group = task / kGroupSpan;
  const unsigned int first_m = group * kGateGroupM;
  const unsigned int active_m =
      kGridM - first_m < kGateGroupM ? kGridM - first_m
                                    : kGateGroupM;
  return (task % kGroupSpan) / active_m;
}

template <bool kGateUp, unsigned int kInputFeatures,
          unsigned int kK64Tiles>
__device__ __forceinline__ void issue_stage(
    SharedStorage* const storage, const unsigned int shared_slot,
    const std::uint16_t* const input, const unsigned int first_token,
    const unsigned int token_limit, const unsigned int first_k64,
    const unsigned int first_n_tile,
    const std::uint8_t* const payload) noexcept {
  static_assert(kAStageVectors % kThreads == 0U);
#pragma unroll
  for (unsigned int pass = 0U; pass < kAStageVectors / kThreads;
       ++pass) {
    const unsigned int logical = threadIdx.x + pass * kThreads;
    const unsigned int k64 = logical / kA64Vectors;
    const unsigned int cell = logical % kA64Vectors;
    const unsigned int row = cell / 8U;
    const unsigned int chunk = cell % 8U;
    const unsigned int token = first_token + row;
    const unsigned int source_token = token < token_limit ? token : 0U;
    const auto* const source = reinterpret_cast<const uint4*>(
        input + static_cast<std::size_t>(source_token) * kInputFeatures +
        (first_k64 + k64) * 64U);
    // Both A and packed B bypass L1.  The v1 A .ca stream displaced B/scale
    // locality; the v2 fragment pipeline keeps their cache policy coupled.
    cp_async_16<false, true>(
        &storage->activations[shared_slot][k64]
                             [transform_a_cell(cell)],
        source + chunk, token < token_limit);
  }

#pragma unroll 1
  for (unsigned int vector = threadIdx.x; vector < kBStageVectors;
       vector += kThreads) {
    const unsigned int k64 = vector / kGateCellVectors;
    const unsigned int within_k64 = vector % kGateCellVectors;
    const std::uint8_t* source = nullptr;
    if constexpr (kGateUp) {
      source = payload +
               (static_cast<std::size_t>(first_n_tile) * kK64Tiles +
                first_k64 + k64) *
                   kGateCellBytes +
               within_k64 * sizeof(uint4);
    } else {
      const unsigned int n_half = within_k64 / kDownCellVectors;
      const unsigned int cell_vector = within_k64 % kDownCellVectors;
      source = payload +
               (static_cast<std::size_t>(first_n_tile + n_half) *
                    kK64Tiles +
                first_k64 + k64) *
                   kDownCellBytes +
               cell_vector * sizeof(uint4);
    }
    cp_async_16<false>(
        &storage->packed_operand[shared_slot][k64][within_k64],
        reinterpret_cast<const uint4*>(source));
  }
  cp_async_commit_group();
}

template <bool kGateUp>
__device__ __forceinline__ void load_decoded_b_stage(
    DecodedBStage* const decoded, const SharedStorage* const storage,
    const unsigned int shared_slot, const unsigned int k64,
    const unsigned int local_k16, const unsigned int warp,
    const unsigned int lane) noexcept {
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  constexpr unsigned int kPhysicalWarps =
      kGateUp ? kGatePhysicalWarps : kDownPhysicalWarps;
  const unsigned int cell_index = kGateUp ? 0U : warp / 4U;
  const unsigned int physical_warp = kGateUp ? warp : warp % 4U;
  const auto* const cell = reinterpret_cast<const std::uint8_t*>(
      storage->packed_operand[shared_slot][k64]) +
      cell_index * kDownCellBytes;
  const auto* const fragment =
      cell + (local_k16 * kPhysicalWarps + physical_warp) *
                 kFragmentBytes;
#pragma unroll
  for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
       ++n_panel) {
    const unsigned int row = n_panel * 8U + lane_group;
    const auto* const packed_row =
        fragment + row * (kFragmentWeightBytes / 32U);
    const std::uint16_t packed =
        static_cast<std::uint16_t>(packed_row[lane_in_group]) |
        static_cast<std::uint16_t>(
            static_cast<unsigned int>(packed_row[lane_in_group + 4U])
            << 8U);
    const std::uint8_t encoded_scale =
        fragment[kFragmentWeightBytes + row];
    const uint2 bits =
        decode_nvfp4x4_to_bf16x4(packed, encoded_scale);
    auto* const words =
        reinterpret_cast<std::uint32_t*>(&decoded->fragments[n_panel]);
    words[0] = bits.x;
    words[1] = bits.y;
  }
}

__device__ __forceinline__ void consume_decoded_b_stage(
    FragC (&accumulators)[kM16Panels][kN8PanelsPerWarp],
    const DecodedBStage& decoded, const SharedStorage* const storage,
    const unsigned int shared_slot, const unsigned int k64,
    const unsigned int local_k16, const unsigned int lane) noexcept {
  unsigned int transformed_cell = transform_a_cell(
      2U * local_k16 + 8U * (lane % 16U) + lane / 16U);
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
    FragA activation;
    marlin::ldsm<4, vllm::kBFloat16.id()>(
        activation,
        &storage->activations[shared_slot][k64][transformed_cell]);
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
         ++n_panel) {
      marlin::mma<vllm::kBFloat16.id(), false>(
          activation, decoded.fragments[n_panel],
          accumulators[m_panel][n_panel]);
    }
    transformed_cell += 128U;
    asm volatile("" : "+r"(transformed_cell));
  }
}

template <bool kGateUp, unsigned int kInputFeatures,
          unsigned int kK64Tiles>
__device__ __forceinline__ void run_full_k_mainloop(
    FragC (&accumulators)[kM16Panels][kN8PanelsPerWarp],
    SharedStorage* const storage, const std::uint16_t* const input,
    const unsigned int first_token, const unsigned int token_limit,
    const unsigned int first_n_tile,
    const std::uint8_t* const payload) noexcept {
  static_assert(kK64Tiles % kK64PerBlock == 0U);
  constexpr unsigned int kKblocks = kK64Tiles / kK64PerBlock;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;

#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
         ++n_panel) {
#pragma unroll
      for (unsigned int value = 0U; value < 4U; ++value) {
        accumulators[m_panel][n_panel].elems[value] = 0.0F;
      }
    }
  }

  constexpr unsigned int kPrime = kKblocks < kStages ? kKblocks : kStages;
#pragma unroll
  for (unsigned int stage = 0U; stage < kPrime; ++stage) {
    issue_stage<kGateUp, kInputFeatures, kK64Tiles>(
        storage, stage, input, first_token, token_limit,
        stage * kK64PerBlock, first_n_tile, payload);
  }

#pragma unroll 1
  for (unsigned int block = 0U; block < kKblocks; ++block) {
    if (block + 2U < kKblocks) {
      cp_async_wait_group<2U>();
    } else if (block + 1U < kKblocks) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();

    const unsigned int shared_slot = block % kStages;
#pragma unroll
    for (unsigned int k64 = 0U; k64 < kK64PerBlock; ++k64) {
      DecodedBStage decoded[2];
      load_decoded_b_stage<kGateUp>(&decoded[0], storage, shared_slot,
                                    k64, 0U, warp, lane);
#pragma unroll
      for (unsigned int local_k16 = 0U; local_k16 < 4U; ++local_k16) {
        const unsigned int current = local_k16 & 1U;
        if (local_k16 + 1U < 4U) {
          load_decoded_b_stage<kGateUp>(
              &decoded[current ^ 1U], storage, shared_slot, k64,
              local_k16 + 1U, warp, lane);
        }
        consume_decoded_b_stage(accumulators, decoded[current], storage,
                                shared_slot, k64, local_k16, lane);
      }
    }

    __syncthreads();
    if (block + kStages < kKblocks) {
      const unsigned int future = block + kStages;
      issue_stage<kGateUp, kInputFeatures, kK64Tiles>(
          storage, shared_slot, input, first_token, token_limit,
          future * kK64PerBlock, first_n_tile, payload);
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          unsigned int kK64Tiles>
__device__ __forceinline__ void execute_gate_task(
    SharedStorage* const storage, const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int first_token,
    const unsigned int token_limit, const unsigned int first_n_tile,
    const unsigned int first_output_column, const float gate_scale,
    const float up_scale, std::uint16_t* const output) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int branch = warp / kGateBranchWarps;
  const unsigned int branch_warp = warp % kGateBranchWarps;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  FragC accumulators[kM16Panels][kN8PanelsPerWarp];
  run_full_k_mainloop<true, kInputFeatures, kK64Tiles>(
      accumulators, storage, input, first_token, token_limit,
      first_n_tile, payload);

  // The FP32 accumulator lifetime ends at this exact BF16 Gate/Up boundary.
  // The now-dead pipeline storage is reused for both branches.
  auto* const staged = reinterpret_cast<std::uint32_t*>(storage);
  constexpr unsigned int kPairsPerRow = kGateTileN / 2U;
  constexpr unsigned int kPairsPerBranch = kTileM * kPairsPerRow;
  const float branch_scale = branch == 0U ? gate_scale : up_scale;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
    const unsigned int row0 = m_panel * 16U + lane_group;
    const unsigned int row1 = row0 + 8U;
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
         ++n_panel) {
      const unsigned int pair_column =
          branch_warp * 16U + n_panel * 4U + lane_in_group;
      const FragC& value = accumulators[m_panel][n_panel];
      staged[branch * kPairsPerBranch + row0 * kPairsPerRow +
             pair_column] =
          pack_scaled_bf16_pair(value.elems[0], value.elems[1],
                                branch_scale);
      staged[branch * kPairsPerBranch + row1 * kPairsPerRow +
             pair_column] =
          pack_scaled_bf16_pair(value.elems[2], value.elems[3],
                                branch_scale);
    }
  }
  __syncthreads();

  // Unlike v1, every warp owns an equal slice of the fused SiLU*Up
  // epilogue. Gate and Up producer warps are no longer half-idle here.
  const auto* const staged_bf16 =
      reinterpret_cast<const std::uint16_t*>(staged);
  constexpr unsigned int kBf16PerBranch = 2U * kPairsPerBranch;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
    const unsigned int row0 = m_panel * 16U + lane_group;
    const unsigned int row1 = row0 + 8U;
    const unsigned int token0 = first_token + row0;
    const unsigned int token1 = first_token + row1;
#pragma unroll
    for (unsigned int epilogue_panel = 0U; epilogue_panel < 2U;
         ++epilogue_panel) {
      const unsigned int pair_column =
          warp * 8U + epilogue_panel * 4U + lane_in_group;
      const unsigned int output_column =
          first_output_column + 2U * pair_column;
      const unsigned int pair0 = row0 * kPairsPerRow + pair_column;
      const unsigned int pair1 = row1 * kPairsPerRow + pair_column;
#pragma unroll
      for (unsigned int element = 0U; element < 2U; ++element) {
        const unsigned int index0 = 2U * pair0 + element;
        const unsigned int index1 = 2U * pair1 + element;
        if (token0 < token_limit) {
          output[static_cast<std::size_t>(token0) * kOutputFeatures +
                 output_column + element] =
              gate_up_silu_mul_bf16(
                  staged_bf16[index0],
                  staged_bf16[kBf16PerBranch + index0]);
        }
        if (token1 < token_limit) {
          output[static_cast<std::size_t>(token1) * kOutputFeatures +
                 output_column + element] =
              gate_up_silu_mul_bf16(
                  staged_bf16[index1],
                  staged_bf16[kBf16PerBranch + index1]);
        }
      }
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          unsigned int kK64Tiles>
__device__ __forceinline__ void execute_down_task(
    SharedStorage* const storage, const std::uint16_t* const input,
    const std::uint8_t* const payload, const unsigned int first_token,
    const unsigned int token_limit, const unsigned int first_n_tile,
    const unsigned int first_output_column, const float global_scale,
    std::uint16_t* const residual_in_out) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  FragC accumulators[kM16Panels][kN8PanelsPerWarp];
  run_full_k_mainloop<false, kInputFeatures, kK64Tiles>(
      accumulators, storage, input, first_token, token_limit,
      first_n_tile, payload);

#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
    const unsigned int row0 = m_panel * 16U + lane_group;
    const unsigned int row1 = row0 + 8U;
    const unsigned int token0 = first_token + row0;
    const unsigned int token1 = first_token + row1;
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
         ++n_panel) {
      const unsigned int output_column =
          first_output_column + warp * 32U + n_panel * 8U +
          2U * lane_in_group;
      const FragC& value = accumulators[m_panel][n_panel];
      if (token0 < token_limit) {
        const std::size_t output0 =
            static_cast<std::size_t>(token0) * kOutputFeatures +
            output_column;
        const std::uint32_t residual0 =
            *reinterpret_cast<const std::uint32_t*>(residual_in_out +
                                                    output0);
        *reinterpret_cast<std::uint32_t*>(residual_in_out + output0) =
            add_residual_bf16_pair(
                pack_scaled_bf16_pair(value.elems[0], value.elems[1],
                                      global_scale),
                residual0);
      }
      if (token1 < token_limit) {
        const std::size_t output1 =
            static_cast<std::size_t>(token1) * kOutputFeatures +
            output_column;
        const std::uint32_t residual1 =
            *reinterpret_cast<const std::uint32_t*>(residual_in_out +
                                                    output1);
        *reinterpret_cast<std::uint32_t*>(residual_in_out + output1) =
            add_residual_bf16_pair(
                pack_scaled_bf16_pair(value.elems[2], value.elems[3],
                                      global_scale),
                residual1);
      }
    }
  }
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_p40_packed_nvfp4_v2_gate_up_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const float gate_scale,
    const float up_scale, std::uint16_t* __restrict__ output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<SharedStorage*>(dynamic_storage);
  const unsigned int task = blockIdx.x;
  const unsigned int m_tile = gate_m_tile_from_task(task);
  const unsigned int n_tile = gate_n_tile_from_task(task);
  execute_gate_task<kSm87P40PackedProjectionHidden,
                    kSm87P40PackedProjectionIntermediate,
                    kGateK64Tiles>(
      storage, input, payload, m_tile * kTileM,
      kSm87P40PackedProjectionTokens, n_tile, n_tile * kGateTileN,
      gate_scale, up_scale, output);
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_p40_packed_nvfp4_v2_down_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const float global_scale,
    std::uint16_t* __restrict__ residual_in_out) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<SharedStorage*>(dynamic_storage);
  const unsigned int task = blockIdx.x;
  const unsigned int m_tile = task / kDownGridN;
  const unsigned int n_block = task % kDownGridN;
  execute_down_task<kSm87P40PackedProjectionIntermediate,
                    kSm87P40PackedProjectionHidden,
                    kDownK64Tiles>(
      storage, input, payload, m_tile * kTileM,
      kSm87P40PackedProjectionTokens, 2U * n_block,
      n_block * kDownTileN, global_scale, residual_in_out);
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_p40_packed_nvfp4_v2_gate_up_tile_test_kernel(
    const std::uint16_t* input, const std::uint8_t* compact_payload,
    const float gate_scale, const float up_scale, std::uint16_t* output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<SharedStorage*>(dynamic_storage);
  execute_gate_task<128U, 128U, 2U>(
      storage, input, compact_payload, 0U, 128U, 0U, 0U,
      gate_scale, up_scale, output);
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_p40_packed_nvfp4_v2_down_tile_test_kernel(
    const std::uint16_t* input, const std::uint8_t* compact_payload,
    const float global_scale, std::uint16_t* residual_in_out) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<SharedStorage*>(dynamic_storage);
  execute_down_task<128U, 256U, 2U>(
      storage, input, compact_payload, 0U, 128U, 0U, 0U,
      global_scale, residual_in_out);
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_p40_packed_nvfp4_v2_gate_up_full_k_test_kernel(
    const std::uint16_t* input, const std::uint8_t* compact_payload,
    const unsigned int token_limit, const float gate_scale,
    const float up_scale, std::uint16_t* output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<SharedStorage*>(dynamic_storage);
  execute_gate_task<kSm87P40PackedProjectionHidden, 128U,
                    kGateK64Tiles>(
      storage, input, compact_payload, 0U, token_limit, 0U, 0U,
      gate_scale, up_scale, output);
}

__global__ __launch_bounds__(kThreads, 1)
void sm87_p40_packed_nvfp4_v2_down_full_k_test_kernel(
    const std::uint16_t* input, const std::uint8_t* compact_payload,
    const unsigned int token_limit, const float global_scale,
    std::uint16_t* residual_in_out) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<SharedStorage*>(dynamic_storage);
  execute_down_task<kSm87P40PackedProjectionIntermediate, 256U,
                    kDownK64Tiles>(
      storage, input, compact_payload, 0U, token_limit, 0U, 0U,
      global_scale, residual_in_out);
}

[[nodiscard]] cudaError_t exact_sm87_device(
    int* const device_out = nullptr) noexcept {
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
  if (!kV2Admitted || properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kSm87P40PackedProjectionSmCount)) {
    return cudaErrorNotSupported;
  }
  if (device_out != nullptr) {
    *device_out = device;
  }
  return cudaSuccess;
}

template <bool kGateUp>
[[nodiscard]] cudaError_t configure_kernel() noexcept {
  if constexpr (kGateUp) {
    return cudaFuncSetAttribute(
        sm87_p40_packed_nvfp4_v2_gate_up_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynamicSharedBytes));
  }
  return cudaFuncSetAttribute(
      sm87_p40_packed_nvfp4_v2_down_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
}

template <bool kGateUp>
[[nodiscard]] cudaError_t query_resources(
    Sm87P40PackedProjectionResources* const resources) noexcept {
  cudaError_t status = configure_kernel<kGateUp>();
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  if constexpr (kGateUp) {
    status = cudaFuncGetAttributes(
        &attributes, sm87_p40_packed_nvfp4_v2_gate_up_kernel);
  } else {
    status = cudaFuncGetAttributes(
        &attributes, sm87_p40_packed_nvfp4_v2_down_kernel);
  }
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  if constexpr (kGateUp) {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, sm87_p40_packed_nvfp4_v2_gate_up_kernel,
        static_cast<int>(kThreads), kDynamicSharedBytes);
  } else {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, sm87_p40_packed_nvfp4_v2_down_kernel,
        static_cast<int>(kThreads), kDynamicSharedBytes);
  }
  if (status != cudaSuccess) {
    return status;
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  const std::size_t total_shared =
      resources->static_shared_bytes + resources->dynamic_shared_bytes;
  const bool admitted =
      resources->registers_per_thread <=
          static_cast<int>(kMaximumRegistersPerThread) &&
      resources->local_bytes == 0U &&
      resources->maximum_threads_per_block >=
          static_cast<int>(kThreads) &&
      resources->active_blocks_per_sm >=
          static_cast<int>(kMinimumBlocksPerSm) &&
      total_shared <= kSharedLimitBytes;
  return admitted ? cudaSuccess : cudaErrorNotSupported;
}

[[nodiscard]] bool valid_artifact_view(
    const Sm87P40PackedProjectionDeviceView& artifact,
    const Sm87P40PackedProjectionRole role) noexcept {
  const auto plan = sm87_p40_packed_projection_plan(role);
  if (!plan.valid() || artifact.payload == nullptr ||
      artifact.payload_bytes != plan.payload_bytes ||
      artifact.artifact_identity == 0U || artifact.role != role ||
      artifact.tactic != plan.tactic ||
      artifact.source_count != plan.source_count ||
      !aligned(artifact.payload,
               kSm87P40PackedProjectionPayloadAlignment)) {
    return false;
  }
  for (std::size_t index = 0U; index < plan.source_count; ++index) {
    if (!std::isfinite(artifact.scalar_scales[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] cudaError_t validate_launch(
    const Sm87P40PackedProjectionRole role,
    const std::uint16_t* const input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    const std::size_t token_count,
    const std::uint16_t* const output) noexcept {
  if (!valid_artifact_view(artifact, role) ||
      token_count != kSm87P40PackedProjectionTokens ||
      !aligned(input, 16U) || !aligned(output, 16U)) {
    return cudaErrorInvalidValue;
  }
  const auto plan = sm87_p40_packed_projection_plan(role);
  const std::size_t input_features =
      role == Sm87P40PackedProjectionRole::kNvFp4GateUp
          ? kSm87P40PackedProjectionHidden
          : kSm87P40PackedProjectionIntermediate;
  const std::size_t output_features =
      role == Sm87P40PackedProjectionRole::kNvFp4GateUp
          ? kSm87P40PackedProjectionIntermediate
          : kSm87P40PackedProjectionHidden;
  if (multiply_would_overflow(token_count, input_features) ||
      multiply_would_overflow(token_count, output_features)) {
    return cudaErrorInvalidValue;
  }
  std::array<ByteRange, 3U> ranges{};
  if (!make_range(input,
                  token_count * input_features * sizeof(std::uint16_t),
                  &ranges[0U]) ||
      !make_range(artifact.payload, plan.payload_bytes, &ranges[1U]) ||
      !make_range(output,
                  token_count * output_features * sizeof(std::uint16_t),
                  &ranges[2U])) {
    return cudaErrorInvalidValue;
  }
  for (std::size_t left = 0U; left < ranges.size(); ++left) {
    for (std::size_t right = left + 1U; right < ranges.size(); ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return cudaErrorInvalidValue;
      }
    }
  }
  int device = -1;
  cudaError_t status = exact_sm87_device(&device);
  if (status != cudaSuccess) {
    return status;
  }
  for (const void* const pointer :
       std::array<const void*, 3U>{input, artifact.payload, output}) {
    status = validate_device_pointer(pointer, device);
    if (status != cudaSuccess) {
      return status;
    }
  }
  return cudaSuccess;
}

[[nodiscard]] cudaError_t validate_tile_test(
    const std::uint16_t* const input,
    const std::uint8_t* const compact_payload,
    const std::uint16_t* const output, const float scale0,
    const float scale1) noexcept {
  if (!aligned(input, 16U) || !aligned(compact_payload, 16U) ||
      !aligned(output, 16U) || !std::isfinite(scale0) ||
      !std::isfinite(scale1)) {
    return cudaErrorInvalidValue;
  }
  int device = -1;
  cudaError_t status = exact_sm87_device(&device);
  if (status != cudaSuccess) {
    return status;
  }
  for (const void* const pointer :
       std::array<const void*, 3U>{input, compact_payload, output}) {
    status = validate_device_pointer(pointer, device);
    if (status != cudaSuccess) {
      return status;
    }
  }
  return cudaSuccess;
}

template <bool kGateUp>
[[nodiscard]] cudaError_t validate_full_k_test(
    const std::uint16_t* const input,
    const std::uint8_t* const compact_payload,
    const std::uint16_t* const output, const std::size_t token_limit,
    const float scale0, const float scale1) noexcept {
  if ((token_limit != 64U && token_limit != kTileM) ||
      !std::isfinite(scale0) || !std::isfinite(scale1) ||
      !aligned(input, 16U) || !aligned(compact_payload, 16U) ||
      !aligned(output, 16U)) {
    return cudaErrorInvalidValue;
  }
  constexpr std::size_t kInputFeatures =
      kGateUp ? kSm87P40PackedProjectionHidden
              : kSm87P40PackedProjectionIntermediate;
  constexpr std::size_t kOutputFeatures = kGateUp ? 128U : 256U;
  constexpr std::size_t kPayloadBytes =
      kGateUp ? static_cast<std::size_t>(kGateK64Tiles) * kGateCellBytes
              : 2U * static_cast<std::size_t>(kDownK64Tiles) *
                    kDownCellBytes;
  std::array<ByteRange, 3U> ranges{};
  if (!make_range(input, kTileM * kInputFeatures * sizeof(std::uint16_t),
                  &ranges[0U]) ||
      !make_range(compact_payload, kPayloadBytes, &ranges[1U]) ||
      !make_range(output, kTileM * kOutputFeatures * sizeof(std::uint16_t),
                  &ranges[2U])) {
    return cudaErrorInvalidValue;
  }
  for (std::size_t left = 0U; left < ranges.size(); ++left) {
    for (std::size_t right = left + 1U; right < ranges.size(); ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return cudaErrorInvalidValue;
      }
    }
  }
  int device = -1;
  cudaError_t status = exact_sm87_device(&device);
  if (status != cudaSuccess) {
    return status;
  }
  for (const void* const pointer :
       std::array<const void*, 3U>{input, compact_payload, output}) {
    status = validate_device_pointer(pointer, device);
    if (status != cudaSuccess) {
      return status;
    }
  }
  return cudaSuccess;
}

template <bool kGateUp>
[[nodiscard]] bool resource_was_bound() noexcept {
  constexpr unsigned int kBit =
      kGateUp ? kGateResourceReady : kDownResourceReady;
  return (g_resource_ready_mask.load(std::memory_order_acquire) & kBit) != 0U;
}

}  // namespace

int query_sm87_p40_packed_nvfp4_v2_resources_cuda(
    const Sm87P40PackedProjectionRole role,
    Sm87P40PackedProjectionResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  const cudaError_t device_status = exact_sm87_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  if (role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
    const cudaError_t status = query_resources<true>(resources);
    if (status == cudaSuccess) {
      g_resource_ready_mask.fetch_or(kGateResourceReady,
                                     std::memory_order_release);
    }
    return static_cast<int>(status);
  }
  if (role == Sm87P40PackedProjectionRole::kNvFp4Down) {
    const cudaError_t status = query_resources<false>(resources);
    if (status == cudaSuccess) {
      g_resource_ready_mask.fetch_or(kDownResourceReady,
                                     std::memory_order_release);
    }
    return static_cast<int>(status);
  }
  return static_cast<int>(cudaErrorInvalidValue);
}

int launch_sm87_p40_packed_nvfp4_v2_gate_up_cuda(
    const std::uint16_t* const input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    const std::size_t token_count,
    std::uint16_t* const activated_output,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_launch(
      Sm87P40PackedProjectionRole::kNvFp4GateUp, input, artifact,
      token_count, activated_output);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  if (!resource_was_bound<true>()) {
    return static_cast<int>(cudaErrorNotReady);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  constexpr unsigned int kTasks = kGridM * kGateGridN;
  sm87_p40_packed_nvfp4_v2_gate_up_kernel
      <<<kTasks, kThreads, kDynamicSharedBytes, stream>>>(
          input, artifact.payload, artifact.scalar_scales[0U],
          artifact.scalar_scales[1U], activated_output);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_packed_nvfp4_v2_down_cuda(
    const std::uint16_t* const input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    const std::size_t token_count,
    std::uint16_t* const residual_in_out,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_launch(
      Sm87P40PackedProjectionRole::kNvFp4Down, input, artifact,
      token_count, residual_in_out);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  if (!resource_was_bound<false>()) {
    return static_cast<int>(cudaErrorNotReady);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  constexpr unsigned int kTasks = kGridM * kDownGridN;
  sm87_p40_packed_nvfp4_v2_down_kernel
      <<<kTasks, kThreads, kDynamicSharedBytes, stream>>>(
          input, artifact.payload, artifact.scalar_scales[0U],
          residual_in_out);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_packed_nvfp4_v2_gate_up_tile_test_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const compact_payload, const float gate_scale,
    const float up_scale, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_tile_test(
      input, compact_payload, output, gate_scale, up_scale);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  const cudaError_t configure = cudaFuncSetAttribute(
      sm87_p40_packed_nvfp4_v2_gate_up_tile_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
  if (configure != cudaSuccess) {
    return static_cast<int>(configure);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_p40_packed_nvfp4_v2_gate_up_tile_test_kernel
      <<<1U, kThreads, kDynamicSharedBytes, stream>>>(
          input, compact_payload, gate_scale, up_scale, output);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_packed_nvfp4_v2_down_tile_test_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const compact_payload, const float global_scale,
    std::uint16_t* const residual_in_out,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_tile_test(
      input, compact_payload, residual_in_out, global_scale, 1.0F);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  const cudaError_t configure = cudaFuncSetAttribute(
      sm87_p40_packed_nvfp4_v2_down_tile_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
  if (configure != cudaSuccess) {
    return static_cast<int>(configure);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_p40_packed_nvfp4_v2_down_tile_test_kernel
      <<<1U, kThreads, kDynamicSharedBytes, stream>>>(
          input, compact_payload, global_scale, residual_in_out);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_packed_nvfp4_v2_gate_up_full_k_test_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const compact_payload,
    const std::size_t token_limit, const float gate_scale,
    const float up_scale, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_full_k_test<true>(
      input, compact_payload, output, token_limit, gate_scale, up_scale);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  const cudaError_t configure = cudaFuncSetAttribute(
      sm87_p40_packed_nvfp4_v2_gate_up_full_k_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
  if (configure != cudaSuccess) {
    return static_cast<int>(configure);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_p40_packed_nvfp4_v2_gate_up_full_k_test_kernel
      <<<1U, kThreads, kDynamicSharedBytes, stream>>>(
          input, compact_payload, static_cast<unsigned int>(token_limit),
          gate_scale, up_scale, output);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_packed_nvfp4_v2_down_full_k_test_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const compact_payload,
    const std::size_t token_limit, const float global_scale,
    std::uint16_t* const residual_in_out,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_full_k_test<false>(
      input, compact_payload, residual_in_out, token_limit, global_scale,
      1.0F);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  const cudaError_t configure = cudaFuncSetAttribute(
      sm87_p40_packed_nvfp4_v2_down_full_k_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
  if (configure != cudaSuccess) {
    return static_cast<int>(configure);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_p40_packed_nvfp4_v2_down_full_k_test_kernel
      <<<1U, kThreads, kDynamicSharedBytes, stream>>>(
          input, compact_payload, static_cast<unsigned int>(token_limit),
          global_scale, residual_in_out);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
