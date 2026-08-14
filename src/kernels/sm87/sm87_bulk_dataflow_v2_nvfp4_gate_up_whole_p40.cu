#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40.h"

#include "sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_oracle_internal.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>

namespace q3x::kernels {
namespace {

namespace cg = cooperative_groups;

constexpr unsigned int kThreads =
    kSm87BulkV2NvFp4GateUpWholeP40Threads;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kPersistentCtas =
    kSm87BulkV2NvFp4GateUpWholeP40PersistentCtas;
constexpr unsigned int kTileM =
    kSm87BulkV2NvFp4GateUpWholeP40TileM;
constexpr unsigned int kTileN =
    kSm87BulkV2NvFp4GateUpWholeP40TileN;
constexpr unsigned int kTileK =
    kSm87BulkV2NvFp4GateUpWholeP40TileK;
constexpr unsigned int kStages =
    kSm87BulkV2NvFp4GateUpWholeP40PipelineStages;
constexpr unsigned int kK16Panels = kTileK / 16U;
constexpr unsigned int kLocalM16Panels = 2U;
constexpr unsigned int kLocalN8Panels = 4U;
constexpr unsigned int kActivationVectors = kTileM * kTileK / 8U;
constexpr unsigned int kBranchWeightVectors =
    kTileN * kTileK / 2U / sizeof(uint4);
constexpr unsigned int kBranchScaleVectors =
    kTileN * (kTileK / 16U) / sizeof(uint4);
constexpr unsigned int kPackedCellWeightBytes = 8'192U;
constexpr unsigned int kPackedCellScaleBytes = 1'024U;
constexpr unsigned int kPackedCellBytes =
    kPackedCellWeightBytes + kPackedCellScaleBytes;
constexpr std::uint64_t kGatePartitionBytes =
    kSm87BulkV2NvFp4GatePartitionPayloadBytes;

static_assert(kThreads == 256U && kPersistentCtas == 32U);
static_assert(kTileM == 64U && kTileN == 64U && kTileK == 64U);
static_assert(kStages == 3U && kK16Panels == 4U);
static_assert(kActivationVectors == 512U);
static_assert(kBranchWeightVectors == 128U);
static_assert(kBranchScaleVectors == 16U);

struct alignas(16) GateUpStage final {
  uint4 activation[kActivationVectors];
  uint4 weight[2U][kBranchWeightVectors];
  uint4 scale[2U][kBranchScaleVectors];
};

struct alignas(16) GateUpPipeline final {
  GateUpStage stage[kStages];
};

static_assert(sizeof(GateUpStage) ==
              kSm87BulkV2NvFp4GateUpWholeP40BytesPerStage);
static_assert(sizeof(GateUpPipeline) ==
              kSm87BulkV2NvFp4GateUpWholeP40PipelineSharedBytes);

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

struct WarpRegisterStage final {
  M16K16Activation activation[kLocalM16Panels];
  K16N8Weight weight[kLocalN8Panels];
};

using BranchWarpAccumulator =
    M16N8Accumulator[kLocalM16Panels][kLocalN8Panels];

__device__ __forceinline__ void cp_async_cg_16(
    void* const shared_destination,
    const void* const global_source) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(
          __cvta_generic_to_shared(shared_destination));
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
  // Authenticated persisted order [K0,K8,K1,K9] is crossed into the PTX
  // m16n8k16.col register order b.x0=[K0,K1], b.x1=[K8,K9].
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

__device__ __forceinline__ void clear_accumulator(
    BranchWarpAccumulator& accumulator) noexcept {
#pragma unroll
  for (unsigned int m16 = 0U; m16 < kLocalM16Panels; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < kLocalN8Panels; ++n8) {
      accumulator[m16][n8] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }
}

__device__ __forceinline__ void issue_stage(
    GateUpPipeline* const pipeline, const unsigned int slot,
    const std::uint16_t* const input,
    const std::uint8_t* const payload,
    const unsigned int m_tile, const unsigned int n_tile,
    const unsigned int k_tile) noexcept {
  auto& stage = pipeline->stage[slot];
  const unsigned int first_m = m_tile * kTileM;

  // One shared A copy serves both independent branch-warp groups.  The eight
  // CTA lanes with the same M lane request the same bytes through .cg; the
  // M-group-major schedule exposes the fixed 2.5-MiB four-row footprint as a
  // reuse opportunity while N cohorts advance.  Cross-cohort L2 residency is
  // deliberately unclaimed until a retained NCU capture proves it.
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int vector_index = threadIdx.x + pass * kThreads;
    const unsigned int row = vector_index / (kTileK / 8U);
    const unsigned int vector = vector_index % (kTileK / 8U);
    const auto* const source = reinterpret_cast<const uint4*>(
                                  input +
                                  static_cast<std::size_t>(first_m + row) *
                                      kSm87BulkV2NvFp4Hidden +
                                  k_tile * kTileK) +
                              vector;
    cp_async_cg_16(&stage.activation[vector_index], source);
  }

  const unsigned int source_n_tile = n_tile / 4U;
  const unsigned int source_n_warp = n_tile % 4U;
  if (threadIdx.x < kBranchWeightVectors) {
    const unsigned int local_vector = threadIdx.x;
    const unsigned int fragment = local_vector / 4U;
    const unsigned int vector_in_fragment = local_vector % 4U;
    const unsigned int k16 = fragment / 8U;
    const unsigned int n8 = fragment % 8U;
    const std::uint64_t source_fragment =
        (static_cast<std::uint64_t>(k16) * 4U + source_n_warp) * 8U +
        n8;
#pragma unroll
    for (unsigned int branch = 0U; branch < 2U; ++branch) {
      const std::uint64_t partition_offset =
          branch == 0U ? 0U : kGatePartitionBytes;
      const std::uint64_t cell =
          partition_offset +
          (static_cast<std::uint64_t>(source_n_tile) *
               kSm87BulkV2NvFp4GateUpWholeP40KTiles +
           k_tile) *
              kPackedCellBytes;
      const auto* const source = reinterpret_cast<const uint4*>(
                                     payload + cell +
                                     source_fragment * 64U) +
                                 vector_in_fragment;
      cp_async_cg_16(&stage.weight[branch][local_vector], source);
    }
  }

  if (threadIdx.x < kBranchScaleVectors) {
    const unsigned int local_vector = threadIdx.x;
    const unsigned int k16 = local_vector / 4U;
    const unsigned int n8_pair = local_vector % 4U;
    const std::uint64_t source_fragment =
        (static_cast<std::uint64_t>(k16) * 4U + source_n_warp) * 8U +
        n8_pair * 2U;
#pragma unroll
    for (unsigned int branch = 0U; branch < 2U; ++branch) {
      const std::uint64_t partition_offset =
          branch == 0U ? 0U : kGatePartitionBytes;
      const std::uint64_t cell =
          partition_offset +
          (static_cast<std::uint64_t>(source_n_tile) *
               kSm87BulkV2NvFp4GateUpWholeP40KTiles +
           k_tile) *
              kPackedCellBytes;
      const auto* const source = reinterpret_cast<const uint4*>(
          payload + cell + kPackedCellWeightBytes +
          source_fragment * 8U);
      cp_async_cg_16(&stage.scale[branch][local_vector], source);
    }
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void load_register_stage(
    WarpRegisterStage& registers, const GateUpStage& stage,
    const unsigned int branch, const unsigned int branch_warp,
    const unsigned int k16, const unsigned int lane) noexcept {
  const unsigned int warp_m = branch_warp / 2U;
  const unsigned int warp_n = branch_warp % 2U;
  const auto* const shared_activation =
      reinterpret_cast<const std::uint16_t*>(stage.activation);
#pragma unroll
  for (unsigned int local_m16 = 0U;
       local_m16 < kLocalM16Panels; ++local_m16) {
    load_activation_fragment(registers.activation[local_m16],
                             shared_activation,
                             warp_m * kLocalM16Panels + local_m16,
                             k16, lane);
  }
  const auto* const shared_weight =
      reinterpret_cast<const std::uint8_t*>(stage.weight[branch]);
  const auto* const shared_scale =
      reinterpret_cast<const std::uint8_t*>(stage.scale[branch]);
#pragma unroll
  for (unsigned int local_n8 = 0U;
       local_n8 < kLocalN8Panels; ++local_n8) {
    registers.weight[local_n8] = decode_weight_fragment(
        shared_weight, shared_scale,
        k16 * 8U + warp_n * kLocalN8Panels + local_n8, lane);
  }
}

__device__ __forceinline__ void run_full_k(
    GateUpPipeline* const pipeline,
    const std::uint16_t* const input,
    const std::uint8_t* const payload,
    const unsigned int m_tile, const unsigned int n_tile,
    BranchWarpAccumulator& accumulator) noexcept {
  unsigned int tid = 0U;
  asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
  const unsigned int warp = tid / kWarpSize;
  const unsigned int lane = tid % kWarpSize;
  const unsigned int branch = warp / 4U;
  const unsigned int branch_warp = warp % 4U;
  WarpRegisterStage registers[2U];
  clear_accumulator(accumulator);

  issue_stage(pipeline, 0U, input, payload, m_tile, n_tile, 0U);
  issue_stage(pipeline, 1U, input, payload, m_tile, n_tile, 1U);
  issue_stage(pipeline, 2U, input, payload, m_tile, n_tile, 2U);

#pragma unroll 1
  for (unsigned int k_tile = 0U;
       k_tile < kSm87BulkV2NvFp4GateUpWholeP40KTiles; ++k_tile) {
    if (k_tile + 2U <
        kSm87BulkV2NvFp4GateUpWholeP40KTiles) {
      cp_async_wait_group<2U>();
    } else if (k_tile + 1U <
               kSm87BulkV2NvFp4GateUpWholeP40KTiles) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();
    const unsigned int slot = k_tile % kStages;
    load_register_stage(registers[0U], pipeline->stage[slot], branch,
                        branch_warp, 0U, lane);

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16Panels; ++k16) {
      const unsigned int current = k16 & 1U;
      const unsigned int next = current ^ 1U;
      if (k16 + 1U < kK16Panels) {
        load_register_stage(registers[next], pipeline->stage[slot], branch,
                            branch_warp, k16 + 1U, lane);
      }

      // Once both S2R register slots own the last K16 operands, this shared
      // slot is dead. Refill it while the final MMA retires.  There is no
      // grid-wide synchronization in this or any other cohort.
      if (k16 + 1U == kK16Panels) {
        __syncthreads();
        if (k_tile + kStages <
            kSm87BulkV2NvFp4GateUpWholeP40KTiles) {
          issue_stage(pipeline, slot, input, payload, m_tile, n_tile,
                      k_tile + kStages);
        }
      }
#pragma unroll
      for (unsigned int local_m16 = 0U;
           local_m16 < kLocalM16Panels; ++local_m16) {
#pragma unroll
        for (unsigned int local_n8 = 0U;
             local_n8 < kLocalN8Panels; ++local_n8) {
          mma_bf16(accumulator[local_m16][local_n8],
                   registers[current].activation[local_m16],
                   registers[current].weight[local_n8]);
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
  return static_cast<std::uint32_t>(
             encode_bf16_rne(low * tensor_scale)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(high * tensor_scale))
          << 16U);
}

__device__ __forceinline__ void publish_branch(
    std::uint16_t* const temporary,
    const BranchWarpAccumulator& accumulator,
    const unsigned int branch, const unsigned int branch_warp,
    const unsigned int lane, const float tensor_scale) noexcept {
  const unsigned int warp_m = branch_warp / 2U;
  const unsigned int warp_n = branch_warp % 2U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int local_m16 = 0U;
       local_m16 < kLocalM16Panels; ++local_m16) {
    const unsigned int row0 =
        warp_m * 32U + local_m16 * 16U + lane_group;
    const unsigned int row1 = row0 + 8U;
#pragma unroll
    for (unsigned int local_n8 = 0U;
         local_n8 < kLocalN8Panels; ++local_n8) {
      const unsigned int column =
          warp_n * 32U + local_n8 * 8U + lane_in_group * 2U;
      const auto& value = accumulator[local_m16][local_n8];
      *reinterpret_cast<std::uint32_t*>(
          temporary + branch * kTileM * kTileN + row0 * kTileN +
          column) =
          pack_scaled_bf16_pair(value.x0, value.x1, tensor_scale);
      *reinterpret_cast<std::uint32_t*>(
          temporary + branch * kTileM * kTileN + row1 * kTileN +
          column) =
          pack_scaled_bf16_pair(value.x2, value.x3, tensor_scale);
    }
  }
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t silu_times_up(
    const std::uint16_t gate_bits,
    const std::uint16_t up_bits) noexcept {
  const float gate = decode_bf16(gate_bits);
  return encode_bf16_rne(
      gate / (1.0F + expf(-gate)) * decode_bf16(up_bits));
}

__device__ __forceinline__ void publish_h(
    const std::uint16_t* const temporary, const unsigned int m_tile,
    const unsigned int n_tile, std::uint16_t* const h) noexcept {
  const unsigned int warp = threadIdx.x / kWarpSize;
  if (warp >= 4U) {
    return;
  }
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 2U;
  const unsigned int warp_n = warp % 2U;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int local_m16 = 0U;
       local_m16 < kLocalM16Panels; ++local_m16) {
    const unsigned int local_row0 =
        warp_m * 32U + local_m16 * 16U + lane_group;
    const unsigned int local_row1 = local_row0 + 8U;
    const unsigned int global_row0 = m_tile * kTileM + local_row0;
    const unsigned int global_row1 = m_tile * kTileM + local_row1;
#pragma unroll
    for (unsigned int local_n8 = 0U;
         local_n8 < kLocalN8Panels; ++local_n8) {
      const unsigned int local_column =
          warp_n * 32U + local_n8 * 8U + lane_in_group * 2U;
      const unsigned int global_column = n_tile * kTileN + local_column;
      const std::uint32_t gate0 =
          *reinterpret_cast<const std::uint32_t*>(
              temporary + local_row0 * kTileN + local_column);
      const std::uint32_t gate1 =
          *reinterpret_cast<const std::uint32_t*>(
              temporary + local_row1 * kTileN + local_column);
      const std::uint32_t up0 =
          *reinterpret_cast<const std::uint32_t*>(
              temporary + kTileM * kTileN +
              local_row0 * kTileN + local_column);
      const std::uint32_t up1 =
          *reinterpret_cast<const std::uint32_t*>(
              temporary + kTileM * kTileN +
              local_row1 * kTileN + local_column);
      *reinterpret_cast<std::uint32_t*>(
          h + static_cast<std::size_t>(global_row0) *
                  kSm87BulkV2NvFp4Intermediate +
          global_column) =
          static_cast<std::uint32_t>(silu_times_up(
              static_cast<std::uint16_t>(gate0),
              static_cast<std::uint16_t>(up0))) |
          (static_cast<std::uint32_t>(silu_times_up(
               static_cast<std::uint16_t>(gate0 >> 16U),
               static_cast<std::uint16_t>(up0 >> 16U)))
           << 16U);
      *reinterpret_cast<std::uint32_t*>(
          h + static_cast<std::size_t>(global_row1) *
                  kSm87BulkV2NvFp4Intermediate +
          global_column) =
          static_cast<std::uint32_t>(silu_times_up(
              static_cast<std::uint16_t>(gate1),
              static_cast<std::uint16_t>(up1))) |
          (static_cast<std::uint32_t>(silu_times_up(
               static_cast<std::uint16_t>(gate1 >> 16U),
               static_cast<std::uint16_t>(up1 >> 16U)))
           << 16U);
    }
  }
}

[[nodiscard]] __device__ __forceinline__ bool cancellation_requested(
    const std::uint32_t* const signal) noexcept {
  return signal != nullptr &&
         *reinterpret_cast<const volatile std::uint32_t*>(signal) != 0U;
}

__device__ __forceinline__ void initialize_control(
    Sm87BulkV2NvFp4GateUpWholeP40DeviceControl* const control,
    const std::uint64_t transaction_epoch,
    const unsigned int expected_cells) noexcept {
  const unsigned int global_thread =
      blockIdx.x * blockDim.x + threadIdx.x;
  if (global_thread != 0U) {
    return;
  }
  auto* const words = reinterpret_cast<std::uint32_t*>(control);
  constexpr unsigned int kWords =
      sizeof(Sm87BulkV2NvFp4GateUpWholeP40DeviceControl) /
      sizeof(std::uint32_t);
  // One thread owns both zeroing and publication.  Distributing these sixteen
  // stores across the grid would let a late zero race the epoch/receipt writes;
  // the immediately following entry grid sync publishes this ordered result.
#pragma unroll
  for (unsigned int word = 0U; word < kWords; ++word) {
    words[word] = 0U;
  }
  control->transaction_epoch = transaction_epoch;
  control->expected_cells = expected_cells;
  control->first_incomplete_cohort = 0xffff'ffffU;
}

[[nodiscard]] __device__ __forceinline__ bool cta_cancellation_requested(
    const std::uint32_t* const signal,
    Sm87BulkV2NvFp4GateUpWholeP40DeviceControl* const control) noexcept {
  bool requested = false;
  if (threadIdx.x == 0U) {
    requested =
        cancellation_requested(signal) ||
        atomicAdd(&control->cancellation_observed, 0U) != 0U;
    if (requested) {
      atomicExch(&control->cancellation_observed, 1U);
    }
  }
  // Cancellation controls entry to code containing CTA barriers, so the
  // collective's returned predicate must be uniform even when device
  // observers see the signal or cross-CTA receipt at different instants.
  return __syncthreads_or(requested ? 1 : 0) != 0;
}

__global__ __launch_bounds__(256, 2)
void sm87_bulk_v2_nvfp4_gate_up_whole_p40_kernel(
    const std::uint16_t* const input,
    const std::uint8_t* const payload,
    const float gate_tensor_scale, const float up_tensor_scale,
    std::uint16_t* const h,
    Sm87BulkV2NvFp4GateUpWholeP40DeviceControl* const control,
    const std::uint32_t* const cancellation_signal,
    const std::uint64_t transaction_epoch,
    const unsigned int m_tiles, const unsigned int n_tiles) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* const pipeline =
      reinterpret_cast<GateUpPipeline*>(shared_bytes);
  const unsigned int n_groups =
      (n_tiles + kSm87BulkV2NvFp4GateUpWholeP40RasterN - 1U) /
      kSm87BulkV2NvFp4GateUpWholeP40RasterN;
  const unsigned int m_groups =
      (m_tiles + kSm87BulkV2NvFp4GateUpWholeP40RasterM - 1U) /
      kSm87BulkV2NvFp4GateUpWholeP40RasterM;
  const unsigned int cohort_count = m_groups * n_groups;
  initialize_control(control, transaction_epoch, m_tiles * n_tiles);
  cg::this_grid().sync();

  const unsigned int m_lane =
      blockIdx.x / kSm87BulkV2NvFp4GateUpWholeP40RasterN;
  const unsigned int n_lane =
      blockIdx.x % kSm87BulkV2NvFp4GateUpWholeP40RasterN;
  unsigned int local_completed = 0U;
  bool cancelled = false;

  // Every CTA advances the same fixed cohort counter.  N groups snake at each
  // M-group boundary, but no cohort contains a grid barrier.  Full active
  // cohorts execute identical full-K work, keeping the 4M x 8N phase relation
  // deterministic without 5,338 global synchronization points.
  for (unsigned int cohort = 0U; cohort < cohort_count; ++cohort) {
    if (cta_cancellation_requested(cancellation_signal, control)) {
      if (threadIdx.x == 0U) {
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
        m_group * kSm87BulkV2NvFp4GateUpWholeP40RasterM + m_lane;
    const unsigned int n_tile =
        n_group * kSm87BulkV2NvFp4GateUpWholeP40RasterN + n_lane;
    if (m_tile >= m_tiles || n_tile >= n_tiles) {
      continue;
    }
    if (threadIdx.x == 0U) {
      atomicAdd(&control->started_cells, 1U);
    }
    BranchWarpAccumulator accumulator;
    run_full_k(pipeline, input, payload, m_tile, n_tile, accumulator);
    if (cta_cancellation_requested(cancellation_signal, control)) {
      if (threadIdx.x == 0U) {
        atomicMin(&control->first_incomplete_cohort, cohort);
      }
      cancelled = true;
      break;
    }
    const unsigned int warp = threadIdx.x / kWarpSize;
    const unsigned int branch = warp / 4U;
    const unsigned int branch_warp = warp % 4U;
    auto* const temporary =
        reinterpret_cast<std::uint16_t*>(shared_bytes);
    publish_branch(temporary, accumulator, branch, branch_warp,
                   threadIdx.x % kWarpSize,
                   branch == 0U ? gate_tensor_scale : up_tensor_scale);
    __syncthreads();
    publish_h(temporary, m_tile, n_tile, h);
    __syncthreads();
    ++local_completed;
  }

  if (threadIdx.x == 0U) {
    atomicAdd(&control->completed_cells, local_completed);
    atomicAdd(&control->completed_ctas, 1U);
    if (cancelled) {
      atomicExch(&control->cancellation_observed, 1U);
    }
  }
  cg::this_grid().sync();
  if (blockIdx.x == 0U && threadIdx.x == 0U) {
    control->launch_completed =
        control->cancellation_observed == 0U &&
                control->completed_ctas == kPersistentCtas &&
                control->started_cells == control->expected_cells &&
                control->completed_cells == control->expected_cells
            ? 1U
            : 0U;
  }
}

[[nodiscard]] bool finite_positive(const float value) noexcept {
  return std::isfinite(value) && value > 0.0F;
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
                 properties->multiProcessorCount == 16 &&
                 properties->cooperativeLaunch != 0
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] cudaError_t launch_raw_unchecked(
    const sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail::
        RawArguments& arguments) noexcept {
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  const std::uint16_t* input_argument = arguments.normalized_input;
  const std::uint8_t* payload_argument = arguments.gate_up_payload;
  float gate_scale_argument = arguments.gate_tensor_scale;
  float up_scale_argument = arguments.up_tensor_scale;
  std::uint16_t* h_argument = arguments.h;
  Sm87BulkV2NvFp4GateUpWholeP40DeviceControl* control_argument =
      arguments.device_control;
  const std::uint32_t* cancellation_argument =
      arguments.cancellation_signal;
  std::uint64_t epoch_argument = arguments.transaction_epoch;
  unsigned int m_tiles_argument = arguments.m_tiles;
  unsigned int n_tiles_argument = arguments.n_tiles;
  void* kernel_arguments[] = {
      &input_argument,       &payload_argument,
      &gate_scale_argument,  &up_scale_argument,
      &h_argument,           &control_argument,
      &cancellation_argument, &epoch_argument,
      &m_tiles_argument,     &n_tiles_argument,
  };
  return cudaLaunchCooperativeKernel(
      sm87_bulk_v2_nvfp4_gate_up_whole_p40_kernel,
      dim3{kPersistentCtas}, dim3{kThreads}, kernel_arguments,
      kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes, stream);
}

}  // namespace

int query_sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_resources_cuda(
    const Sm87BulkV2NvFp4GateUpWholeP40CodeEvidence* const code_evidence,
    Sm87BulkV2NvFp4GateUpWholeP40Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  cudaDeviceProp properties{};
  cudaError_t status = validate_sm87_device(&properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes, sm87_bulk_v2_nvfp4_gate_up_whole_p40_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, sm87_bulk_v2_nvfp4_gate_up_whole_p40_kernel,
      kThreads, kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87BulkV2NvFp4GateUpWholeP40DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->cooperative_grid_capacity =
      active_blocks * properties.multiProcessorCount;
  if (code_evidence != nullptr) {
    resources->code = *code_evidence;
  }
  resources->cooperative_launch_supported = true;
  resources->exact_oracle_attached =
      resources->code.same_elf_exact_oracle;
  // This plain struct is not an admission capability.  Even a structurally
  // valid caller-supplied record cannot promote itself; the retained
  // hash-bound audit and runtime receipt are combined outside this query.
  resources->resource_gate_passed = false;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_cuda(
    const Sm87BulkV2NvFp4GateUpWholeP40Arguments& arguments) noexcept {
  if (arguments.transaction_epoch == 0U ||
      arguments.normalized_input == nullptr || arguments.h == nullptr ||
      arguments.device_control == nullptr || arguments.cuda_stream == nullptr ||
      !sm87_target_aot_nvfp4_cuda_asset_valid(arguments.gate_up_asset) ||
      arguments.gate_up_asset.payload.role !=
          Sm87TargetAotProjectionRole::kNvFp4GateUp ||
      !sm87_bulk_v2_nvfp4_gate_up_whole_p40_public_arguments_ranges_valid(
          arguments)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  float gate_scale = 0.0F;
  float up_scale = 0.0F;
  std::memcpy(&gate_scale,
              &arguments.gate_up_asset.tensor_scale_bits[0U],
              sizeof(gate_scale));
  std::memcpy(&up_scale,
              &arguments.gate_up_asset.tensor_scale_bits[1U],
              sizeof(up_scale));
  if (!finite_positive(gate_scale) || !finite_positive(up_scale)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail::RawArguments raw;
  raw.transaction_epoch = arguments.transaction_epoch;
  raw.normalized_input = arguments.normalized_input;
  raw.gate_up_payload = reinterpret_cast<const std::uint8_t*>(
      arguments.gate_up_asset.payload.begin);
  raw.gate_tensor_scale = gate_scale;
  raw.up_tensor_scale = up_scale;
  raw.h = arguments.h;
  raw.device_control = arguments.device_control;
  raw.cancellation_signal = arguments.cancellation_signal;
  raw.m_tiles = kSm87BulkV2NvFp4GateUpWholeP40MTiles;
  raw.n_tiles = kSm87BulkV2NvFp4GateUpWholeP40NTiles;
  raw.cuda_stream = arguments.cuda_stream;
  return static_cast<int>(launch_raw_unchecked(raw));
}

namespace sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail {

int launch_raw(const RawArguments& arguments) noexcept {
  if (arguments.transaction_epoch == 0U ||
      arguments.normalized_input == nullptr ||
      arguments.gate_up_payload == nullptr || arguments.h == nullptr ||
      arguments.device_control == nullptr || arguments.cuda_stream == nullptr ||
      arguments.m_tiles == 0U ||
      arguments.m_tiles > kSm87BulkV2NvFp4GateUpWholeP40MTiles ||
      arguments.n_tiles == 0U ||
      arguments.n_tiles > kSm87BulkV2NvFp4GateUpWholeP40NTiles ||
      !finite_positive(arguments.gate_tensor_scale) ||
      !finite_positive(arguments.up_tensor_scale)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaDeviceProp properties{};
  const cudaError_t status = validate_sm87_device(&properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  (void)cudaGetLastError();
  return static_cast<int>(launch_raw_unchecked(arguments));
}

}  // namespace sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail

}  // namespace q3x::kernels
