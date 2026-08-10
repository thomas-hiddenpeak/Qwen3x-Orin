#include "q3x/kernels/sm87_p40_packed_projection.h"

#include "p40_packed_projection_common.cuh"
#include "third_party/vllm_marlin/marlin_template.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Implemented by p40_packed_fp8.cu. The public dispatch remains in this
// translation unit so prepare/resource ownership is unambiguous.
int prepare_sm87_p40_packed_fp8_projection_cuda(
    Sm87P40PackedProjectionRole role,
    const Sm87P40PackedCanonicalSource* sources,
    std::size_t source_count, std::uint8_t* destination,
    std::size_t destination_bytes, void* cuda_stream) noexcept;

int query_sm87_p40_packed_fp8_resources_cuda(
    Sm87P40PackedProjectionRole role,
    Sm87P40PackedProjectionResources* resources) noexcept;

namespace {

using namespace p40_packed_detail;

#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
inline constexpr bool kPackedProjectionAdmitted = true;
#else
inline constexpr bool kPackedProjectionAdmitted = false;
#endif

using Bf16MarlinType = marlin::MarlinScalarType<vllm::kBFloat16.id()>;
using FragA = typename Bf16MarlinType::FragA;
using FragB = typename Bf16MarlinType::FragB;
using FragC = typename Bf16MarlinType::FragC;

inline constexpr unsigned int kGateThreads = 256U;
inline constexpr unsigned int kGateWarps = 8U;
inline constexpr unsigned int kGateBranchWarps = 4U;
inline constexpr unsigned int kGateLogicalTileN = 128U;
inline constexpr unsigned int kGateGridN = 136U;
inline constexpr unsigned int kGateGroupM = 4U;
inline constexpr unsigned int kGateKtiles =
    kSm87P40PackedProjectionHidden / kPackedProjectionTileK;
inline constexpr unsigned int kGateFragmentWeightBytes = 256U;
inline constexpr unsigned int kGateFragmentScaleBytes = 32U;
inline constexpr unsigned int kGateFragmentBytes =
    kGateFragmentWeightBytes + kGateFragmentScaleBytes;
inline constexpr unsigned int kGateCellBytes =
    4U * kGateWarps * kGateFragmentBytes;

inline constexpr unsigned int kDownThreads = 128U;
inline constexpr unsigned int kDownWarps = 4U;
inline constexpr unsigned int kDownTileN = 128U;
inline constexpr unsigned int kDownGridN = 40U;
inline constexpr unsigned int kDownKtiles =
    kSm87P40PackedProjectionIntermediate / kPackedProjectionTileK;
inline constexpr unsigned int kDownFragmentWeightBytes = 256U;
inline constexpr unsigned int kDownFragmentScaleBytes = 32U;
inline constexpr unsigned int kDownFragmentBytes =
    kDownFragmentWeightBytes + kDownFragmentScaleBytes;
inline constexpr unsigned int kDownCellBytes =
    4U * kDownWarps * kDownFragmentBytes;

inline constexpr unsigned int kTileM =
    kSm87P40PackedProjectionTileM;
inline constexpr unsigned int kM16Panels = kTileM / 16U;
inline constexpr unsigned int kN8PanelsPerWarp = 4U;
inline constexpr unsigned int kAStageVectors =
    kTileM * kPackedProjectionTileK / 8U;
inline constexpr unsigned int kGateCellVectors = kGateCellBytes / 16U;
inline constexpr unsigned int kDownCellVectors = kDownCellBytes / 16U;
inline constexpr unsigned int kMaximumRegistersPerThread = 128U;
inline constexpr unsigned int kMinimumBlocksPerSm = 2U;
inline constexpr std::size_t kSharedLimitBytes = 72U * 1'024U;

struct alignas(32) GateSharedStorage {
  uint4 activations[kPackedProjectionPipelineStages][kAStageVectors];
  uint4 packed_operand[kPackedProjectionPipelineStages][kGateCellVectors];
};

struct alignas(32) DownSharedStorage {
  uint4 activations[kPackedProjectionPipelineStages][kAStageVectors];
  uint4 packed_operand[kPackedProjectionPipelineStages][kDownCellVectors];
};

inline constexpr std::size_t kGateDynamicSharedBytes =
    sizeof(GateSharedStorage);
inline constexpr std::size_t kDownDynamicSharedBytes =
    sizeof(DownSharedStorage);

static_assert(kGateKtiles == 80U);
static_assert(kDownKtiles == 272U);
static_assert(kGateCellBytes == 9'216U);
static_assert(kDownCellBytes == 4'608U);
static_assert(kGateDynamicSharedBytes == 69'632U);
static_assert(kDownDynamicSharedBytes == 51'200U);
static_assert(kGateDynamicSharedBytes <= kSharedLimitBytes);
static_assert(kDownDynamicSharedBytes <= kSharedLimitBytes);
static_assert(kGateThreads == kGateWarps * kWarpSize);
static_assert(kDownThreads == kDownWarps * kWarpSize);
static_assert(kPackedProjectionPersistentCtas ==
              kSm87P40PackedProjectionPersistentCtas);
static_assert(kPackedProjectionSmCount ==
              kSm87P40PackedProjectionSmCount);
static_assert(kPackedProjectionPipelineStages == 4U);
static_assert(kPackedProjectionTileK == 64U);
static_assert(sm87_p40_packed_projection_plan(
                  Sm87P40PackedProjectionRole::kNvFp4GateUp)
                  .partitions[0U]
                  .cell_bytes *
              2U == kGateCellBytes);
static_assert(sm87_p40_packed_projection_plan(
                  Sm87P40PackedProjectionRole::kNvFp4Down)
                  .partitions[0U]
                  .cell_bytes == kDownCellBytes);

struct DecodedBStage {
  FragB fragments[kN8PanelsPerWarp];
};

[[nodiscard]] __device__ __forceinline__ unsigned int
gate_m_tile_from_task(const unsigned int task) noexcept {
  constexpr unsigned int kGroupSpan = kGateGroupM * kGateGridN;
  const unsigned int group = task / kGroupSpan;
  const unsigned int first_m = group * kGateGroupM;
  const unsigned int active_m =
      kSm87P40PackedProjectionGridM - first_m < kGateGroupM
          ? kSm87P40PackedProjectionGridM - first_m
          : kGateGroupM;
  return first_m + (task % kGroupSpan) % active_m;
}

[[nodiscard]] __device__ __forceinline__ unsigned int
gate_n_tile_from_task(const unsigned int task) noexcept {
  constexpr unsigned int kGroupSpan = kGateGroupM * kGateGridN;
  const unsigned int group = task / kGroupSpan;
  const unsigned int first_m = group * kGateGroupM;
  const unsigned int active_m =
      kSm87P40PackedProjectionGridM - first_m < kGateGroupM
          ? kSm87P40PackedProjectionGridM - first_m
          : kGateGroupM;
  return (task % kGroupSpan) / active_m;
}

template <unsigned int kThreads, unsigned int kOperandVectors,
          typename Storage>
__device__ __forceinline__ void issue_stage(
    Storage* const storage, const unsigned int shared_slot,
    const std::uint16_t* const input, const unsigned int input_features,
    const unsigned int first_token, const unsigned int first_k,
  const std::uint8_t* const packed_cell) noexcept {
  static_assert(kAStageVectors % kThreads == 0U);
#pragma unroll
  for (unsigned int pass = 0U; pass < kAStageVectors / kThreads; ++pass) {
    const unsigned int logical_cell = threadIdx.x + pass * kThreads;
    const unsigned int row = logical_cell / 8U;
    const unsigned int chunk = logical_cell % 8U;
    const auto* const source = reinterpret_cast<const uint4*>(
        input + static_cast<std::size_t>(first_token + row) *
                    input_features +
        first_k);
    cp_async_16<true>(
        &storage->activations[shared_slot][transform_a_cell(logical_cell)],
        source + chunk);
  }
#pragma unroll 1
  for (unsigned int vector = threadIdx.x; vector < kOperandVectors;
       vector += kThreads) {
    cp_async_16<false>(&storage->packed_operand[shared_slot][vector],
                       reinterpret_cast<const uint4*>(packed_cell) + vector);
  }
  cp_async_commit_group();
}

template <unsigned int kPhysicalWarps, typename Storage>
__device__ __forceinline__ void load_decoded_b_stage(
    DecodedBStage* const decoded, const Storage* const storage,
    const unsigned int shared_slot, const unsigned int local_k16,
    const unsigned int physical_warp, const unsigned int lane) noexcept {
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  const auto* const cell = reinterpret_cast<const std::uint8_t*>(
      storage->packed_operand[shared_slot]);
  const auto* const fragment =
      cell + (local_k16 * kPhysicalWarps + physical_warp) *
                 kGateFragmentBytes;
#pragma unroll
  for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp; ++n_panel) {
    const unsigned int row = n_panel * 8U + lane_group;
    const auto* const packed_row =
        fragment + row * (kGateFragmentWeightBytes / 32U);
    const std::uint16_t packed =
        static_cast<std::uint16_t>(packed_row[lane_in_group]) |
        static_cast<std::uint16_t>(
            static_cast<unsigned int>(packed_row[lane_in_group + 4U])
            << 8U);
    const std::uint8_t encoded_scale =
        fragment[kGateFragmentWeightBytes + row];
    const uint2 bits =
        decode_nvfp4x4_to_bf16x4(packed, encoded_scale);
    auto* const words =
        reinterpret_cast<std::uint32_t*>(&decoded->fragments[n_panel]);
    words[0] = bits.x;
    words[1] = bits.y;
  }
}

template <typename Storage>
__device__ __forceinline__ void consume_decoded_b_stage(
    FragC (&accumulators)[kM16Panels][kN8PanelsPerWarp],
    const DecodedBStage& decoded, const Storage* const storage,
    const unsigned int shared_slot, const unsigned int local_k16,
    const unsigned int lane) noexcept {
  unsigned int transformed_cell = transform_a_cell(
      2U * local_k16 + 8U * (lane % 16U) + lane / 16U);
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
    FragA activation;
    marlin::ldsm<4, vllm::kBFloat16.id()>(
        activation, &storage->activations[shared_slot][transformed_cell]);
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp; ++n_panel) {
      marlin::mma<vllm::kBFloat16.id(), false>(
          activation, decoded.fragments[n_panel],
          accumulators[m_panel][n_panel]);
    }
    transformed_cell += 128U;
    asm volatile("" : "+r"(transformed_cell));
  }
}

template <unsigned int kThreads, unsigned int kOperandVectors,
          unsigned int kPhysicalWarps, unsigned int kInputFeatures,
          unsigned int kKtiles, typename Storage>
__device__ __forceinline__ void run_full_k_mainloop(
    FragC (&accumulators)[kM16Panels][kN8PanelsPerWarp],
    Storage* const storage, const std::uint16_t* const input,
    const unsigned int first_token,
    const std::uint8_t* const packed_cells) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp; ++n_panel) {
#pragma unroll
      for (unsigned int value = 0U; value < 4U; ++value) {
        accumulators[m_panel][n_panel].elems[value] = 0.0F;
      }
    }
  }

  constexpr unsigned int kPrimeStages =
      kKtiles < kPackedProjectionPipelineStages
          ? kKtiles
          : kPackedProjectionPipelineStages;
#pragma unroll
  for (unsigned int stage = 0U; stage < kPrimeStages; ++stage) {
    issue_stage<kThreads, kOperandVectors>(
        storage, stage, input, kInputFeatures, first_token,
        stage * kPackedProjectionTileK,
        packed_cells + static_cast<std::size_t>(stage) *
                           kOperandVectors * sizeof(uint4));
  }

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kKtiles; ++stage) {
    if (stage + 3U < kKtiles) {
      cp_async_wait_group<3U>();
    } else if (stage + 2U < kKtiles) {
      cp_async_wait_group<2U>();
    } else if (stage + 1U < kKtiles) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();

    const unsigned int shared_slot =
        stage % kPackedProjectionPipelineStages;
    DecodedBStage decoded[2];
    load_decoded_b_stage<kPhysicalWarps>(&decoded[0], storage, shared_slot,
                                         0U, warp, lane);
#pragma unroll
    for (unsigned int local_k16 = 0U; local_k16 < 4U; ++local_k16) {
      const unsigned int current = local_k16 & 1U;
      if (local_k16 + 1U < 4U) {
        load_decoded_b_stage<kPhysicalWarps>(
            &decoded[current ^ 1U], storage, shared_slot,
            local_k16 + 1U, warp, lane);
      }
      consume_decoded_b_stage(accumulators, decoded[current], storage,
                              shared_slot, local_k16, lane);
    }

    __syncthreads();
    if (stage + kPackedProjectionPipelineStages < kKtiles) {
      const unsigned int future =
          stage + kPackedProjectionPipelineStages;
      issue_stage<kThreads, kOperandVectors>(
          storage, shared_slot, input, kInputFeatures, first_token,
          future * kPackedProjectionTileK,
          packed_cells + static_cast<std::size_t>(future) *
                             kOperandVectors * sizeof(uint4));
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          unsigned int kKtiles>
__device__ __forceinline__ void execute_gate_task(
    GateSharedStorage* const storage, const std::uint16_t* const input,
    const std::uint8_t* const packed_cells, const unsigned int first_token,
    const unsigned int first_output_column, const float gate_scale,
    const float up_scale, std::uint16_t* const output) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int branch = warp / kGateBranchWarps;
  const unsigned int branch_warp = warp % kGateBranchWarps;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  FragC accumulators[kM16Panels][kN8PanelsPerWarp];
  run_full_k_mainloop<kGateThreads, kGateCellVectors, kGateWarps,
                      kInputFeatures, kKtiles>(
      accumulators, storage, input, first_token, packed_cells);

  // All A/B pipeline storage is dead. Reuse its prefix for the exact BF16
  // Gate and Up publication boundaries, ending the FP32 accumulator lifetime
  // before expf extends the epilogue live range.
  auto* const staged = reinterpret_cast<std::uint32_t*>(storage);
  constexpr unsigned int kPairsPerRow = kGateLogicalTileN / 2U;
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

  if (branch == 0U) {
    const auto* const staged_bf16 =
        reinterpret_cast<const std::uint16_t*>(staged);
    constexpr unsigned int kBf16PerBranch = 2U * kPairsPerBranch;
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
      const unsigned int row0 = m_panel * 16U + lane_group;
      const unsigned int row1 = row0 + 8U;
#pragma unroll
      for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
           ++n_panel) {
        const unsigned int pair_column =
            branch_warp * 16U + n_panel * 4U + lane_in_group;
        const unsigned int output_column =
            first_output_column + 2U * pair_column;
        const unsigned int pair0 = row0 * kPairsPerRow + pair_column;
        const unsigned int pair1 = row1 * kPairsPerRow + pair_column;
#pragma unroll
        for (unsigned int element = 0U; element < 2U; ++element) {
          const unsigned int index0 = 2U * pair0 + element;
          const unsigned int index1 = 2U * pair1 + element;
          output[static_cast<std::size_t>(first_token + row0) *
                     kOutputFeatures +
                 output_column + element] =
              gate_up_silu_mul_bf16(
                  staged_bf16[index0],
                  staged_bf16[kBf16PerBranch + index0]);
          output[static_cast<std::size_t>(first_token + row1) *
                     kOutputFeatures +
                 output_column + element] =
              gate_up_silu_mul_bf16(
                  staged_bf16[index1],
                  staged_bf16[kBf16PerBranch + index1]);
        }
      }
    }
  }
  __syncthreads();
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          unsigned int kKtiles>
__device__ __forceinline__ void execute_down_task(
    DownSharedStorage* const storage, const std::uint16_t* const input,
    const std::uint8_t* const packed_cells, const unsigned int first_token,
    const unsigned int first_output_column, const float global_scale,
    std::uint16_t* const residual_in_out) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  FragC accumulators[kM16Panels][kN8PanelsPerWarp];
  run_full_k_mainloop<kDownThreads, kDownCellVectors, kDownWarps,
                      kInputFeatures, kKtiles>(
      accumulators, storage, input, first_token, packed_cells);

#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
    const unsigned int row0 = m_panel * 16U + lane_group;
    const unsigned int row1 = row0 + 8U;
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
         ++n_panel) {
      const unsigned int output_column =
          first_output_column + warp * 32U + n_panel * 8U +
          2U * lane_in_group;
      const FragC& value = accumulators[m_panel][n_panel];
      const std::size_t output0 =
          static_cast<std::size_t>(first_token + row0) * kOutputFeatures +
          output_column;
      const std::size_t output1 =
          static_cast<std::size_t>(first_token + row1) * kOutputFeatures +
          output_column;
      const std::uint32_t residual0 =
          *reinterpret_cast<const std::uint32_t*>(residual_in_out +
                                                  output0);
      const std::uint32_t residual1 =
          *reinterpret_cast<const std::uint32_t*>(residual_in_out +
                                                  output1);
      *reinterpret_cast<std::uint32_t*>(residual_in_out + output0) =
          add_residual_bf16_pair(
              pack_scaled_bf16_pair(value.elems[0], value.elems[1],
                                    global_scale),
              residual0);
      *reinterpret_cast<std::uint32_t*>(residual_in_out + output1) =
          add_residual_bf16_pair(
              pack_scaled_bf16_pair(value.elems[2], value.elems[3],
                                    global_scale),
              residual1);
    }
  }
  __syncthreads();
}

__global__ __launch_bounds__(kGateThreads, 2)
void sm87_p40_packed_nvfp4_gate_up_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const float gate_scale,
    const float up_scale, std::uint16_t* __restrict__ output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<GateSharedStorage*>(dynamic_storage);
  constexpr unsigned int kLogicalTasks =
      kSm87P40PackedProjectionGridM * kGateGridN;
  for (unsigned int task = blockIdx.x; task < kLogicalTasks;
       task += gridDim.x) {
    const unsigned int m_tile = gate_m_tile_from_task(task);
    const unsigned int n_tile = gate_n_tile_from_task(task);
    const auto* const cells =
        payload + static_cast<std::size_t>(n_tile) * kGateKtiles *
                      kGateCellBytes;
    execute_gate_task<kSm87P40PackedProjectionHidden,
                      kSm87P40PackedProjectionIntermediate, kGateKtiles>(
        storage, input, cells, m_tile * kTileM,
        n_tile * kGateLogicalTileN, gate_scale, up_scale, output);
  }
}

// Four compiler-side resident blocks cap register allocation at 128/thread.
// Dynamic shared memory remains the hardware occupancy limiter (three CTAs/SM
// on the admitted 16-SM Orin), so the production requirement of two is kept.
__global__ __launch_bounds__(kDownThreads, 4)
void sm87_p40_packed_nvfp4_down_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ payload, const float global_scale,
    std::uint16_t* __restrict__ residual_in_out) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<DownSharedStorage*>(dynamic_storage);
  constexpr unsigned int kLogicalTasks =
      kSm87P40PackedProjectionGridM * kDownGridN;
  for (unsigned int task = blockIdx.x; task < kLogicalTasks;
       task += gridDim.x) {
    const unsigned int m_tile = task / kDownGridN;
    const unsigned int n_tile = task % kDownGridN;
    const auto* const cells =
        payload + static_cast<std::size_t>(n_tile) * kDownKtiles *
                      kDownCellBytes;
    execute_down_task<kSm87P40PackedProjectionIntermediate,
                      kSm87P40PackedProjectionHidden, kDownKtiles>(
        storage, input, cells, m_tile * kTileM, n_tile * kDownTileN,
        global_scale, residual_in_out);
  }
}

// T1 cell kernels reuse the production mainloop and exact epilogues with one
// K64 cell. They exercise every raw code without allocating a P40 activation
// matrix and carry no performance authority.
__global__ __launch_bounds__(kGateThreads, 2)
void sm87_p40_packed_nvfp4_gate_up_cell_test_kernel(
    const std::uint16_t* input, const std::uint8_t* packed_cell,
    const float gate_scale, const float up_scale, std::uint16_t* output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<GateSharedStorage*>(dynamic_storage);
  execute_gate_task<64U, 128U, 1U>(storage, input, packed_cell, 0U, 0U,
                                   gate_scale, up_scale, output);
}

__global__ __launch_bounds__(kDownThreads, 2)
void sm87_p40_packed_nvfp4_down_cell_test_kernel(
    const std::uint16_t* input, const std::uint8_t* packed_cell,
    const float global_scale, std::uint16_t* residual_in_out) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<DownSharedStorage*>(dynamic_storage);
  execute_down_task<64U, 128U, 1U>(storage, input, packed_cell, 0U, 0U,
                                   global_scale, residual_in_out);
}

template <bool kGateUp>
__global__ __launch_bounds__(256, 2) void pack_nvfp4_consumer_layout_kernel(
    const std::uint8_t* source0_weight,
    const std::uint8_t* source0_scale,
    const std::uint8_t* source1_weight,
    const std::uint8_t* source1_scale,
    std::uint8_t* destination) {
  constexpr unsigned int kInputFeatures =
      kGateUp ? kSm87P40PackedProjectionHidden
              : kSm87P40PackedProjectionIntermediate;
  constexpr unsigned int kGridN = kGateUp ? kGateGridN : kDownGridN;
  constexpr unsigned int kKtiles = kInputFeatures / 64U;
  constexpr unsigned int kPhysicalWarps = kGateUp ? 8U : 4U;
  constexpr unsigned int kCellBytes =
      kGateUp ? kGateCellBytes : kDownCellBytes;
  constexpr std::uint64_t kRowFragments =
      static_cast<std::uint64_t>(kGridN) * kKtiles * 4U *
      kPhysicalWarps * 32U;
  const std::uint64_t stride =
      static_cast<std::uint64_t>(blockDim.x) * gridDim.x;
  for (std::uint64_t unit =
           static_cast<std::uint64_t>(blockIdx.x) * blockDim.x +
           threadIdx.x;
       unit < kRowFragments; unit += stride) {
    std::uint64_t remaining = unit;
    const unsigned int row = static_cast<unsigned int>(remaining % 32U);
    remaining /= 32U;
    const unsigned int physical_warp =
        static_cast<unsigned int>(remaining % kPhysicalWarps);
    remaining /= kPhysicalWarps;
    const unsigned int local_k16 =
        static_cast<unsigned int>(remaining % 4U);
    remaining /= 4U;
    const unsigned int k_tile =
        static_cast<unsigned int>(remaining % kKtiles);
    const unsigned int n_tile =
        static_cast<unsigned int>(remaining / kKtiles);
    const unsigned int branch = kGateUp ? physical_warp / 4U : 0U;
    const unsigned int local_warp =
        kGateUp ? physical_warp % 4U : physical_warp;
    const unsigned int output_column =
        n_tile * 128U + local_warp * 32U + row;
    const unsigned int global_k16 = k_tile * 4U + local_k16;
    const auto* const weight =
        branch == 0U ? source0_weight : source1_weight;
    const auto* const scale =
        branch == 0U ? source0_scale : source1_scale;
    const auto* const source_weight =
        weight + static_cast<std::size_t>(output_column) *
                     (kInputFeatures / 2U) +
        global_k16 * 8U;
    const std::uint8_t source_scale =
        scale[static_cast<std::size_t>(output_column) *
                  (kInputFeatures / 16U) +
              global_k16];
    const std::size_t cell_offset =
        (static_cast<std::size_t>(n_tile) * kKtiles + k_tile) *
        kCellBytes;
    const std::size_t fragment_offset =
        (local_k16 * kPhysicalWarps + physical_warp) *
        kGateFragmentBytes;
    auto* const destination_weight =
        destination + cell_offset + fragment_offset + row * 8U;
    *reinterpret_cast<uint2*>(destination_weight) =
        *reinterpret_cast<const uint2*>(source_weight);
    destination[cell_offset + fragment_offset +
                kGateFragmentWeightBytes + row] = source_scale;
  }
}

[[nodiscard]] cudaError_t exact_sm87_device(
    int* const device_out = nullptr,
    cudaDeviceProp* const properties_out = nullptr) noexcept {
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
  if (!kPackedProjectionAdmitted || properties.major != 8 ||
      properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kPackedProjectionSmCount)) {
    return cudaErrorNotSupported;
  }
  if (device_out != nullptr) {
    *device_out = device;
  }
  if (properties_out != nullptr) {
    *properties_out = properties;
  }
  return cudaSuccess;
}

template <bool kGateUp>
[[nodiscard]] cudaError_t configure_nvfp4_kernel() noexcept {
  if constexpr (kGateUp) {
    return cudaFuncSetAttribute(
        sm87_p40_packed_nvfp4_gate_up_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kGateDynamicSharedBytes));
  }
  return cudaFuncSetAttribute(
      sm87_p40_packed_nvfp4_down_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDownDynamicSharedBytes));
}

template <bool kGateUp>
[[nodiscard]] cudaError_t query_nvfp4_resources(
    Sm87P40PackedProjectionResources* const resources) noexcept {
  constexpr unsigned int kThreads = kGateUp ? kGateThreads : kDownThreads;
  constexpr std::size_t kDynamicShared =
      kGateUp ? kGateDynamicSharedBytes : kDownDynamicSharedBytes;
  cudaError_t status = configure_nvfp4_kernel<kGateUp>();
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  if constexpr (kGateUp) {
    status = cudaFuncGetAttributes(
        &attributes, sm87_p40_packed_nvfp4_gate_up_kernel);
  } else {
    status = cudaFuncGetAttributes(
        &attributes, sm87_p40_packed_nvfp4_down_kernel);
  }
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  if constexpr (kGateUp) {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, sm87_p40_packed_nvfp4_gate_up_kernel,
        static_cast<int>(kThreads), kDynamicShared);
  } else {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, sm87_p40_packed_nvfp4_down_kernel,
        static_cast<int>(kThreads), kDynamicShared);
  }
  if (status != cudaSuccess) {
    return status;
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kDynamicShared;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  const std::size_t total_shared =
      resources->static_shared_bytes + resources->dynamic_shared_bytes;
  const bool admitted =
      resources->registers_per_thread <=
          static_cast<int>(kMaximumRegistersPerThread) &&
      resources->local_bytes == 0U &&
      resources->maximum_threads_per_block >= static_cast<int>(kThreads) &&
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
      !aligned(artifact.payload, kSm87P40PackedProjectionPayloadAlignment)) {
    return false;
  }
  for (std::size_t index = 0U; index < plan.source_count; ++index) {
    if (!std::isfinite(artifact.scalar_scales[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] cudaError_t validate_nvfp4_launch(
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

[[nodiscard]] cudaError_t validate_prepare_sources(
    const Sm87P40PackedProjectionRole role,
    const Sm87P40PackedCanonicalSource* const sources,
    const std::size_t source_count, std::uint8_t* const destination,
    const std::size_t destination_bytes) noexcept {
  const auto plan = sm87_p40_packed_projection_plan(role);
  if (!plan.valid() || sources == nullptr || destination == nullptr ||
      source_count != plan.source_count ||
      destination_bytes != plan.payload_bytes ||
      !aligned(destination, kSm87P40PackedProjectionPayloadAlignment)) {
    return cudaErrorInvalidValue;
  }
  const std::array<Sm87P40PackedLogicalRole, 2U> roles =
      role == Sm87P40PackedProjectionRole::kNvFp4GateUp
          ? std::array<Sm87P40PackedLogicalRole, 2U>{
                Sm87P40PackedLogicalRole::kNvFp4Gate,
                Sm87P40PackedLogicalRole::kNvFp4Up}
          : std::array<Sm87P40PackedLogicalRole, 2U>{
                Sm87P40PackedLogicalRole::kNvFp4Down,
                Sm87P40PackedLogicalRole::kInvalid};
  const std::size_t input_features =
      role == Sm87P40PackedProjectionRole::kNvFp4GateUp
          ? kSm87P40PackedProjectionHidden
          : kSm87P40PackedProjectionIntermediate;
  const std::size_t output_features =
      role == Sm87P40PackedProjectionRole::kNvFp4GateUp
          ? kSm87P40PackedProjectionIntermediate
          : kSm87P40PackedProjectionHidden;
  std::array<ByteRange, 7U> ranges{};
  if (!make_range(destination, destination_bytes, &ranges[0U])) {
    return cudaErrorInvalidValue;
  }
  int device = -1;
  cudaError_t status = exact_sm87_device(&device);
  if (status != cudaSuccess) {
    return status;
  }
  for (std::size_t index = 0U; index < source_count; ++index) {
    const auto& source = sources[index];
    if (source.role != roles[index] || source.weight == nullptr ||
        source.block_scale == nullptr ||
        source.global_scale_device == nullptr ||
        source.output_features != output_features ||
        source.input_features != input_features ||
        !aligned(source.weight, 16U) || !aligned(source.block_scale, 16U) ||
        !aligned(source.global_scale_device, alignof(float))) {
      return cudaErrorInvalidValue;
    }
    const std::size_t weight_bytes =
        output_features * input_features / 2U;
    const std::size_t scale_bytes =
        output_features * input_features / 16U;
    if (!make_range(source.weight, weight_bytes,
                    &ranges[1U + index * 3U]) ||
        !make_range(source.block_scale, scale_bytes,
                    &ranges[2U + index * 3U]) ||
        !make_range(source.global_scale_device, sizeof(float),
                    &ranges[3U + index * 3U])) {
      return cudaErrorInvalidValue;
    }
    for (const void* const pointer :
         std::array<const void*, 3U>{source.weight, source.block_scale,
                                    source.global_scale_device}) {
      status = validate_device_pointer(pointer, device);
      if (status != cudaSuccess) {
        return status;
      }
    }
  }
  status = validate_device_pointer(destination, device);
  if (status != cudaSuccess) {
    return status;
  }
  const std::size_t used_ranges = 1U + source_count * 3U;
  for (std::size_t left = 0U; left < used_ranges; ++left) {
    for (std::size_t right = left + 1U; right < used_ranges; ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return cudaErrorInvalidValue;
      }
    }
  }
  return cudaSuccess;
}

}  // namespace

int prepare_sm87_p40_packed_projection_cuda(
    const Sm87P40PackedProjectionRole role,
    const Sm87P40PackedCanonicalSource* const sources,
    const std::size_t source_count, std::uint8_t* const destination,
    const std::size_t destination_bytes, void* const cuda_stream) noexcept {
  if (role != Sm87P40PackedProjectionRole::kNvFp4GateUp &&
      role != Sm87P40PackedProjectionRole::kNvFp4Down) {
    return prepare_sm87_p40_packed_fp8_projection_cuda(
        role, sources, source_count, destination, destination_bytes,
        cuda_stream);
  }
  const cudaError_t validation = validate_prepare_sources(
      role, sources, source_count, destination, destination_bytes);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  constexpr unsigned int kPackBlocks = 4'096U;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if (role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
    pack_nvfp4_consumer_layout_kernel<true>
        <<<kPackBlocks, 256U, 0U, stream>>>(
            sources[0U].weight, sources[0U].block_scale,
            sources[1U].weight, sources[1U].block_scale, destination);
  } else {
    pack_nvfp4_consumer_layout_kernel<false>
        <<<kPackBlocks, 256U, 0U, stream>>>(
            sources[0U].weight, sources[0U].block_scale,
            sources[0U].weight, sources[0U].block_scale, destination);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

int query_sm87_p40_packed_projection_resources_cuda(
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
    return static_cast<int>(query_nvfp4_resources<true>(resources));
  }
  if (role == Sm87P40PackedProjectionRole::kNvFp4Down) {
    return static_cast<int>(query_nvfp4_resources<false>(resources));
  }
  return query_sm87_p40_packed_fp8_resources_cuda(role, resources);
}

int launch_sm87_p40_packed_nvfp4_gate_up_cuda(
    const std::uint16_t* const input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    const std::size_t token_count,
    std::uint16_t* const activated_output,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_nvfp4_launch(
      Sm87P40PackedProjectionRole::kNvFp4GateUp, input, artifact,
      token_count, activated_output);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  Sm87P40PackedProjectionResources resources{};
  const int resource_status = query_sm87_p40_packed_projection_resources_cuda(
      Sm87P40PackedProjectionRole::kNvFp4GateUp, &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  const cudaError_t configure = configure_nvfp4_kernel<true>();
  if (configure != cudaSuccess) {
    return static_cast<int>(configure);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_p40_packed_nvfp4_gate_up_kernel
      <<<kPackedProjectionPersistentCtas, kGateThreads,
         kGateDynamicSharedBytes, stream>>>(
          input, artifact.payload, artifact.scalar_scales[0U],
          artifact.scalar_scales[1U], activated_output);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_packed_nvfp4_down_cuda(
    const std::uint16_t* const input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    const std::size_t token_count,
    std::uint16_t* const residual_in_out,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_nvfp4_launch(
      Sm87P40PackedProjectionRole::kNvFp4Down, input, artifact,
      token_count, residual_in_out);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  Sm87P40PackedProjectionResources resources{};
  const int resource_status = query_sm87_p40_packed_projection_resources_cuda(
      Sm87P40PackedProjectionRole::kNvFp4Down, &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  const cudaError_t configure = configure_nvfp4_kernel<false>();
  if (configure != cudaSuccess) {
    return static_cast<int>(configure);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_p40_packed_nvfp4_down_kernel
      <<<kPackedProjectionPersistentCtas, kDownThreads,
         kDownDynamicSharedBytes, stream>>>(
          input, artifact.payload, artifact.scalar_scales[0U],
          residual_in_out);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_packed_nvfp4_gate_up_cell_test_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const packed_cell, const float gate_scale,
    const float up_scale, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!aligned(input, 16U) || !aligned(packed_cell, 16U) ||
      !aligned(output, 16U) || !std::isfinite(gate_scale) ||
      !std::isfinite(up_scale)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t device_status = exact_sm87_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  const cudaError_t configure = cudaFuncSetAttribute(
      sm87_p40_packed_nvfp4_gate_up_cell_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kGateDynamicSharedBytes));
  if (configure != cudaSuccess) {
    return static_cast<int>(configure);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_p40_packed_nvfp4_gate_up_cell_test_kernel
      <<<1U, kGateThreads, kGateDynamicSharedBytes, stream>>>(
          input, packed_cell, gate_scale, up_scale, output);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_packed_nvfp4_down_cell_test_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const packed_cell, const float global_scale,
    std::uint16_t* const residual_in_out,
    void* const cuda_stream) noexcept {
  if (!aligned(input, 16U) || !aligned(packed_cell, 16U) ||
      !aligned(residual_in_out, 16U) || !std::isfinite(global_scale)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t device_status = exact_sm87_device();
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  const cudaError_t configure = cudaFuncSetAttribute(
      sm87_p40_packed_nvfp4_down_cell_test_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDownDynamicSharedBytes));
  if (configure != cudaSuccess) {
    return static_cast<int>(configure);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_p40_packed_nvfp4_down_cell_test_kernel
      <<<1U, kDownThreads, kDownDynamicSharedBytes, stream>>>(
          input, packed_cell, global_scale, residual_in_out);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
