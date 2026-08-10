#include "q3x/kernels/sm87_p40_packed_projection.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
inline constexpr bool kPackedProjectionAdmitted = true;
#else
inline constexpr bool kPackedProjectionAdmitted = false;
#endif

inline constexpr unsigned int kWarpSize = 32U;
inline constexpr unsigned int kThreads = 128U;
inline constexpr unsigned int kPersistentCtas = 32U;
inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kTileM = 64U;

// Wide input and attention-output projections share the same physical tile,
// but not the same raster.  A complete K64 stage is retained in registers
// before its shared slot is recycled, so the next packed-B load overlaps the
// current decode/MMA chain rather than waiting behind it.
inline constexpr unsigned int kWideTileN = 128U;
inline constexpr unsigned int kWideTileK = 64U;
inline constexpr unsigned int kWideStages = 4U;
inline constexpr unsigned int kWideAStageVectors =
    kTileM * kWideTileK * sizeof(std::uint16_t) / sizeof(uint4);
inline constexpr unsigned int kWideBStageVectors =
    kWideTileN * kWideTileK / sizeof(uint4);
inline constexpr std::size_t kWideDynamicSharedBytes =
    kWideStages * (kWideAStageVectors + kWideBStageVectors) *
    sizeof(uint4);

// Full-Attention K/V own a small-N tactic.  K128 halves the number of stage
// turns while keeping two complete CTAs resident on SM87.  It is deliberately
// not expressed as a masked wide tile: doing so would waste half the N work
// and erase the shape distinction that the deployment plan authenticates.
inline constexpr unsigned int kSmallTileN = 64U;
inline constexpr unsigned int kSmallTileK = 128U;
inline constexpr unsigned int kSmallStages = 3U;
inline constexpr unsigned int kSmallAStageVectors =
    kTileM * kSmallTileK * sizeof(std::uint16_t) / sizeof(uint4);
inline constexpr unsigned int kSmallBStageVectors =
    kSmallTileN * kSmallTileK / sizeof(uint4);
inline constexpr std::size_t kSmallDynamicSharedBytes =
    kSmallStages * (kSmallAStageVectors + kSmallBStageVectors) *
    sizeof(uint4);

inline constexpr unsigned int kWideK16PerStage = kWideTileK / 16U;
inline constexpr unsigned int kSmallK16PerStage = kSmallTileK / 16U;
inline constexpr unsigned int kWideM16PerWarp = 4U;
inline constexpr unsigned int kWideN8PerWarp = 4U;
inline constexpr unsigned int kSmallM16PerWarp = 4U;
inline constexpr unsigned int kSmallN8PerWarp = 2U;

static_assert(kWideAStageVectors == 512U);
static_assert(kWideBStageVectors == 512U);
static_assert(kWideDynamicSharedBytes == 65'536U);
static_assert(kSmallAStageVectors == 1'024U);
static_assert(kSmallBStageVectors == 512U);
static_assert(kSmallDynamicSharedBytes == 73'728U);
static_assert(2U * kSmallDynamicSharedBytes < 164U * 1'024U);

enum class DeviceTactic : unsigned int {
  kWideK5120 = 0U,
  kSmallKvK5120 = 1U,
  kOutputK6144 = 2U,
};

enum class GroupTopology : unsigned int {
  kLinearInput = 0U,
  kFullInput = 1U,
  kAttentionOutput = 2U,
};

struct DevicePartition {
  const uint4* sidecar = nullptr;
  std::uint16_t* output = nullptr;
  float weight_scale = 0.0F;
  unsigned int rows = 0U;
  unsigned int nblocks = 0U;
  unsigned int first_n_tile = 0U;
  DeviceTactic tactic = DeviceTactic::kWideK5120;
};

struct ByteRange {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

struct Accumulator {
  float x0;
  float x1;
  float x2;
  float x3;
};

struct ActivationFragment {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct alignas(32) WidePipelineStorage {
  uint4 activations[kWideStages][kWideAStageVectors];
  uint4 weights[kWideStages][kWideBStageVectors];
};

struct alignas(32) SmallPipelineStorage {
  uint4 activations[kSmallStages][kSmallAStageVectors];
  uint4 weights[kSmallStages][kSmallBStageVectors];
};

static_assert(sizeof(WidePipelineStorage) == kWideDynamicSharedBytes);
static_assert(sizeof(SmallPipelineStorage) == kSmallDynamicSharedBytes);

struct WideRegisterStage {
  ActivationFragment activations[kWideK16PerStage][kWideM16PerWarp];
  std::uint32_t packed_weights[kWideK16PerStage][kWideN8PerWarp];
};

struct SmallRegisterStage {
  ActivationFragment activations[kSmallK16PerStage][kSmallM16PerWarp];
  std::uint32_t packed_weights[kSmallK16PerStage][kSmallN8PerWarp];
};

[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0U &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] ByteRange make_range(const void* const pointer,
                                   const std::size_t bytes) noexcept {
  if (pointer == nullptr || bytes == 0U) {
    return {};
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return {};
  }
  return {begin, begin + bytes, true};
}

[[nodiscard]] bool overlaps(const ByteRange& first,
                            const ByteRange& second) noexcept {
  return !first.valid || !second.valid ||
         (first.begin < second.end && second.begin < first.end);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
decode_e4m3fn_to_bf16_bits(const std::uint8_t code) {
  const std::uint16_t sign =
      static_cast<std::uint16_t>(code & 0x80U) << 8U;
  const std::uint16_t magnitude =
      static_cast<std::uint16_t>(code & 0x7fU);
  if (magnitude == 0x7fU) {
    return static_cast<std::uint16_t>(sign | 0x7fc0U);
  }
  const std::uint16_t exponent = magnitude >> 3U;
  const std::uint16_t mantissa = magnitude & 0x07U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return sign;
    }
    const std::uint16_t leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const std::uint16_t bf16_exponent = 118U + leading;
    const std::uint16_t bf16_mantissa =
        (mantissa - (1U << leading)) << (7U - leading);
    return static_cast<std::uint16_t>(
        sign | (bf16_exponent << 7U) | bf16_mantissa);
  }
  return static_cast<std::uint16_t>(
      sign | ((120U + exponent) << 7U) | (mantissa << 4U));
}

[[nodiscard]] __device__ __forceinline__ uint2 decode_fp8x4_to_bf16x4(
    const std::uint32_t packed,
    const std::uint16_t* const decoded_weights) {
  // Sidecars store [v0,v2,v1,v3], which is the fragment-native permutation
  // used by the existing exact supermatrix route.  The per-CTA codebook
  // keeps E4M3 conversion out of the inner arithmetic chain.
  const std::uint16_t value0 =
      decoded_weights[static_cast<std::uint8_t>(packed)];
  const std::uint16_t value2 =
      decoded_weights[static_cast<std::uint8_t>(packed >> 8U)];
  const std::uint16_t value1 =
      decoded_weights[static_cast<std::uint8_t>(packed >> 16U)];
  const std::uint16_t value3 =
      decoded_weights[static_cast<std::uint8_t>(packed >> 24U)];
  return {static_cast<std::uint32_t>(value0) |
              (static_cast<std::uint32_t>(value1) << 16U),
          static_cast<std::uint32_t>(value2) |
              (static_cast<std::uint32_t>(value3) << 16U)};
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) {
  std::uint32_t bits = __float_as_uint(value);
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
pack_scaled_bf16_pair(const float low, const float high,
                      const float scale) {
  return static_cast<std::uint32_t>(encode_bf16_rne(low * scale)) |
         (static_cast<std::uint32_t>(encode_bf16_rne(high * scale))
          << 16U);
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    Accumulator* const accumulator,
    const ActivationFragment& activation, const uint2 weights) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator->x0), "+f"(accumulator->x1),
        "+f"(accumulator->x2), "+f"(accumulator->x3)
      : "r"(activation.x0), "r"(activation.x1),
        "r"(activation.x2), "r"(activation.x3), "r"(weights.x),
        "r"(weights.y));
#endif
}

template <bool kPredicate>
__device__ __forceinline__ void cp_async_ca_shared_global_16(
    void* const shared_destination, const void* const global_source,
    const bool valid = true) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const auto shared_address = static_cast<std::uint32_t>(
      __cvta_generic_to_shared(shared_destination));
  if constexpr (kPredicate) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(shared_address), "l"(global_source),
                   "r"(valid ? 16U : 0U)
                 : "memory");
  } else {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
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

__device__ __forceinline__ void cp_async_cg_shared_global_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const auto shared_address = static_cast<std::uint32_t>(
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

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <unsigned int kGroups>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" : : "n"(kGroups) : "memory");
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t load_shared_u32(
    const std::uint32_t* const source) {
  std::uint32_t value = 0U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const auto shared_address = static_cast<std::uint32_t>(
      __cvta_generic_to_shared(source));
  asm volatile("ld.shared.u32 %0, [%1];"
               : "=r"(value)
               : "r"(shared_address)
               : "memory");
#endif
  return value;
}

template <unsigned int kLeadingDimension>
__device__ __forceinline__ void load_activation_fragment(
    ActivationFragment* const fragment,
    const std::uint16_t* const shared_activations,
    const unsigned int logical_m_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row_in_panel =
      lane % 8U + (quadrant & 1U) * 8U;
  const unsigned int logical_row =
      logical_m_panel * 16U + row_in_panel;
  const unsigned int logical_chunk =
      k16 * 2U + (quadrant >> 1U);
  const unsigned int physical_chunk =
      logical_chunk ^ (logical_row & 7U);
  const auto* const source =
      shared_activations + logical_row * kLeadingDimension +
      physical_chunk * 8U;
  const auto shared_address = static_cast<std::uint32_t>(
      __cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment->x0), "=r"(fragment->x1),
        "=r"(fragment->x2), "=r"(fragment->x3)
      : "r"(shared_address)
      : "memory");
#endif
}

template <unsigned int kColumns>
__device__ __forceinline__ void issue_wide_stage(
    WidePipelineStorage* const storage, const unsigned int slot,
    const uint4* const sidecar_stage,
    const std::uint16_t* const activations, const unsigned int token_count,
    const unsigned int m_tile, const unsigned int first_k) {
  static_assert(kColumns == 5'120U || kColumns == 6'144U);
  const unsigned int first_token = m_tile * kTileM;
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    const unsigned int row = index / 8U;
    const unsigned int chunk = index % 8U;
    const bool valid = first_token + row < token_count;
    const unsigned int source_token = valid ? first_token + row : first_token;
    const auto* const source = reinterpret_cast<const uint4*>(
        activations + static_cast<std::size_t>(source_token) * kColumns +
        first_k);
    const unsigned int physical_chunk = chunk ^ (row & 7U);
    cp_async_ca_shared_global_16<true>(
        storage->activations[slot] + row * 8U + physical_chunk,
        source + chunk, valid);
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    cp_async_cg_shared_global_16(storage->weights[slot] + index,
                                 sidecar_stage + index);
  }
  cp_async_commit_group();
}

template <unsigned int kColumns>
__device__ __forceinline__ void issue_small_stage(
    SmallPipelineStorage* const storage, const unsigned int slot,
    const uint4* const sidecar_stage,
    const std::uint16_t* const activations, const unsigned int token_count,
    const unsigned int m_tile, const unsigned int first_k) {
  static_assert(kColumns == 5'120U);
  const unsigned int first_token = m_tile * kTileM;
#pragma unroll
  for (unsigned int pass = 0U; pass < 8U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    const unsigned int row = index / 16U;
    const unsigned int chunk = index % 16U;
    const bool valid = first_token + row < token_count;
    const unsigned int source_token = valid ? first_token + row : first_token;
    const auto* const source = reinterpret_cast<const uint4*>(
        activations + static_cast<std::size_t>(source_token) * kColumns +
        first_k);
    const unsigned int physical_chunk = chunk ^ (row & 7U);
    cp_async_ca_shared_global_16<true>(
        storage->activations[slot] + row * 16U + physical_chunk,
        source + chunk, valid);
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    cp_async_cg_shared_global_16(storage->weights[slot] + index,
                                 sidecar_stage + index);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void load_wide_register_stage(
    WideRegisterStage* const registers,
    const WidePipelineStorage* const storage, const unsigned int slot,
    const unsigned int warp, const unsigned int lane) {
  const auto* const shared_a = reinterpret_cast<const std::uint16_t*>(
      storage->activations[slot]);
  const auto* const shared_b = reinterpret_cast<const std::uint32_t*>(
      storage->weights[slot]);
#pragma unroll
  for (unsigned int k16 = 0U; k16 < kWideK16PerStage; ++k16) {
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < kWideM16PerWarp;
         ++m_panel) {
      load_activation_fragment<kWideTileK>(
          &registers->activations[k16][m_panel], shared_a, m_panel, k16,
          lane);
    }
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kWideN8PerWarp;
         ++n_panel) {
      registers->packed_weights[k16][n_panel] = load_shared_u32(
          shared_b +
          ((k16 * 4U + warp) * kWideN8PerWarp + n_panel) *
              kWarpSize +
          lane);
    }
  }
}

__device__ __forceinline__ void load_small_register_stage(
    SmallRegisterStage* const registers,
    const SmallPipelineStorage* const storage, const unsigned int slot,
    const unsigned int warp, const unsigned int lane) {
  const auto* const shared_a = reinterpret_cast<const std::uint16_t*>(
      storage->activations[slot]);
  const auto* const shared_b = reinterpret_cast<const std::uint32_t*>(
      storage->weights[slot]);
#pragma unroll
  for (unsigned int k16 = 0U; k16 < kSmallK16PerStage; ++k16) {
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < kSmallM16PerWarp;
         ++m_panel) {
      load_activation_fragment<kSmallTileK>(
          &registers->activations[k16][m_panel], shared_a,
          m_panel, k16, lane);
    }
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kSmallN8PerWarp;
         ++n_panel) {
      registers->packed_weights[k16][n_panel] = load_shared_u32(
          shared_b +
          ((k16 * 4U + warp) * kSmallN8PerWarp + n_panel) *
              kWarpSize +
          lane);
    }
  }
}

__device__ __forceinline__ void consume_wide_register_stage(
    const WideRegisterStage& registers,
    const std::uint16_t* const decoded_weights,
    Accumulator (&accumulators)[kWideM16PerWarp][kWideN8PerWarp]) {
#pragma unroll
  for (unsigned int k16 = 0U; k16 < kWideK16PerStage; ++k16) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kWideN8PerWarp;
         ++n_panel) {
      const uint2 decoded = decode_fp8x4_to_bf16x4(
          registers.packed_weights[k16][n_panel], decoded_weights);
#pragma unroll
      for (unsigned int m_panel = 0U; m_panel < kWideM16PerWarp;
           ++m_panel) {
        mma_m16n8k16_bf16(&accumulators[m_panel][n_panel],
                          registers.activations[k16][m_panel], decoded);
      }
    }
  }
}

__device__ __forceinline__ void consume_small_register_stage(
    const SmallRegisterStage& registers,
    const std::uint16_t* const decoded_weights,
    Accumulator (&accumulators)[kSmallM16PerWarp][kSmallN8PerWarp]) {
#pragma unroll
  for (unsigned int k16 = 0U; k16 < kSmallK16PerStage; ++k16) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kSmallN8PerWarp;
         ++n_panel) {
      const uint2 decoded = decode_fp8x4_to_bf16x4(
          registers.packed_weights[k16][n_panel], decoded_weights);
#pragma unroll
      for (unsigned int m_panel = 0U; m_panel < kSmallM16PerWarp;
           ++m_panel) {
        mma_m16n8k16_bf16(&accumulators[m_panel][n_panel],
                          registers.activations[k16][m_panel], decoded);
      }
    }
  }
}

template <unsigned int kColumns>
__device__ __forceinline__ void execute_wide_task(
    WidePipelineStorage* const storage,
    const DevicePartition& partition,
    const std::uint16_t* const activations,
    const std::uint16_t* const decoded_weights,
    const unsigned int token_count, const unsigned int m_tile,
    const unsigned int nblock) {
  constexpr unsigned int kKStages = kColumns / kWideTileK;
  static_assert(kKStages == 80U || kKStages == 96U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const uint4* const sidecar_tile =
      partition.sidecar + static_cast<std::size_t>(nblock) * kKStages *
                                  kWideBStageVectors;

  Accumulator accumulators[kWideM16PerWarp][kWideN8PerWarp];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kWideM16PerWarp;
       ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kWideN8PerWarp;
         ++n_panel) {
      accumulators[m_panel][n_panel] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

#pragma unroll
  for (unsigned int stage = 0U; stage < kWideStages; ++stage) {
    issue_wide_stage<kColumns>(
        storage, stage, sidecar_tile + stage * kWideBStageVectors,
        activations, token_count, m_tile, stage * kWideTileK);
  }

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kKStages; ++stage) {
    if (stage + 3U < kKStages) {
      cp_async_wait_group<3U>();
    } else if (stage + 2U < kKStages) {
      cp_async_wait_group<2U>();
    } else if (stage + 1U < kKStages) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();
    const unsigned int slot = stage % kWideStages;
    WideRegisterStage registers{};
    load_wide_register_stage(&registers, storage, slot, warp, lane);

    // The whole shared stage is now register-resident.  This barrier closes
    // every S2R read before cp.async starts recycling the slot.
    __syncthreads();
    if (stage + kWideStages < kKStages) {
      const unsigned int future = stage + kWideStages;
      issue_wide_stage<kColumns>(
          storage, slot,
          sidecar_tile +
              static_cast<std::size_t>(future) * kWideBStageVectors,
          activations, token_count, m_tile, future * kWideTileK);
    }
    consume_wide_register_stage(registers, decoded_weights, accumulators);
  }
  cp_async_wait_group<0U>();
  __syncthreads();

  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  const unsigned int first_token = m_tile * kTileM;
  const unsigned int first_column = nblock * kWideTileN;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kWideM16PerWarp;
       ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kWideN8PerWarp;
         ++n_panel) {
      const unsigned int token0 = m_panel * 16U + lane_group;
      const unsigned int token1 = token0 + 8U;
      const unsigned int column =
          warp * 32U + n_panel * 8U + 2U * lane_in_group;
      if (first_token + token0 < token_count) {
        *reinterpret_cast<std::uint32_t*>(
            partition.output +
            static_cast<std::size_t>(first_token + token0) *
                partition.rows +
            first_column + column) = pack_scaled_bf16_pair(
            accumulators[m_panel][n_panel].x0,
            accumulators[m_panel][n_panel].x1, partition.weight_scale);
      }
      if (first_token + token1 < token_count) {
        *reinterpret_cast<std::uint32_t*>(
            partition.output +
            static_cast<std::size_t>(first_token + token1) *
                partition.rows +
            first_column + column) = pack_scaled_bf16_pair(
            accumulators[m_panel][n_panel].x2,
            accumulators[m_panel][n_panel].x3, partition.weight_scale);
      }
    }
  }
  __syncthreads();
}

template <unsigned int kColumns>
__device__ __forceinline__ void execute_small_task(
    SmallPipelineStorage* const storage,
    const DevicePartition& partition,
    const std::uint16_t* const activations,
    const std::uint16_t* const decoded_weights,
    const unsigned int token_count, const unsigned int m_tile,
    const unsigned int nblock) {
  constexpr unsigned int kKStages = kColumns / kSmallTileK;
  static_assert(kKStages == 40U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const uint4* const sidecar_tile =
      partition.sidecar + static_cast<std::size_t>(nblock) * kKStages *
                                  kSmallBStageVectors;

  Accumulator accumulators[kSmallM16PerWarp][kSmallN8PerWarp];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kSmallM16PerWarp;
       ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kSmallN8PerWarp;
         ++n_panel) {
      accumulators[m_panel][n_panel] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

#pragma unroll
  for (unsigned int stage = 0U; stage < kSmallStages; ++stage) {
    issue_small_stage<kColumns>(
        storage, stage, sidecar_tile + stage * kSmallBStageVectors,
        activations, token_count, m_tile, stage * kSmallTileK);
  }

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kKStages; ++stage) {
    if (stage + 2U < kKStages) {
      cp_async_wait_group<2U>();
    } else if (stage + 1U < kKStages) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();
    const unsigned int slot = stage % kSmallStages;
    SmallRegisterStage registers{};
    load_small_register_stage(&registers, storage, slot, warp, lane);
    __syncthreads();
    if (stage + kSmallStages < kKStages) {
      const unsigned int future = stage + kSmallStages;
      issue_small_stage<kColumns>(
          storage, slot,
          sidecar_tile +
              static_cast<std::size_t>(future) * kSmallBStageVectors,
          activations, token_count, m_tile, future * kSmallTileK);
    }
    consume_small_register_stage(registers, decoded_weights, accumulators);
  }
  cp_async_wait_group<0U>();
  __syncthreads();

  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  const unsigned int first_token = m_tile * kTileM;
  const unsigned int first_column = nblock * kSmallTileN;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kSmallM16PerWarp;
       ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kSmallN8PerWarp;
         ++n_panel) {
      const unsigned int token0 =
          m_panel * 16U + lane_group;
      const unsigned int token1 = token0 + 8U;
      const unsigned int column =
          warp * 16U + n_panel * 8U + 2U * lane_in_group;
      if (first_token + token0 < token_count) {
        *reinterpret_cast<std::uint32_t*>(
            partition.output +
            static_cast<std::size_t>(first_token + token0) *
                partition.rows +
            first_column + column) = pack_scaled_bf16_pair(
            accumulators[m_panel][n_panel].x0,
            accumulators[m_panel][n_panel].x1, partition.weight_scale);
      }
      if (first_token + token1 < token_count) {
        *reinterpret_cast<std::uint32_t*>(
            partition.output +
            static_cast<std::size_t>(first_token + token1) *
                partition.rows +
            first_column + column) = pack_scaled_bf16_pair(
            accumulators[m_panel][n_panel].x2,
            accumulators[m_panel][n_panel].x3, partition.weight_scale);
      }
    }
  }
  __syncthreads();
}

[[nodiscard]] __device__ __forceinline__ DevicePartition select_partition(
    const DevicePartition& partition0,
    const DevicePartition& partition1,
    const DevicePartition& partition2, const unsigned int n_tile,
    unsigned int* const local_nblock) {
  DevicePartition selected = partition0;
  if (partition1.nblocks != 0U && n_tile >= partition1.first_n_tile &&
      n_tile - partition1.first_n_tile < partition1.nblocks) {
    selected = partition1;
  } else if (partition2.nblocks != 0U &&
             n_tile >= partition2.first_n_tile &&
             n_tile - partition2.first_n_tile < partition2.nblocks) {
    selected = partition2;
  }
  *local_nblock = n_tile - selected.first_n_tile;
  return selected;
}

__device__ __forceinline__ void map_grouped_task(
    const unsigned int local_task, const unsigned int grid_m,
    const unsigned int nblocks, const unsigned int group_m,
    unsigned int* const m_tile, unsigned int* const nblock) {
  const unsigned int group_span = group_m * nblocks;
  const unsigned int group = local_task / group_span;
  const unsigned int first_m = group * group_m;
  const unsigned int remaining = grid_m - first_m;
  const unsigned int active_m = remaining < group_m ? remaining : group_m;
  const unsigned int offset = local_task % group_span;
  *m_tile = first_m + offset % active_m;
  *nblock = offset / active_m;
}

template <GroupTopology kTopology>
__global__ __launch_bounds__(kThreads, 2) void p40_packed_fp8_kernel(
    const DevicePartition partition0,
    const DevicePartition partition1,
    const DevicePartition partition2,
    const std::uint16_t* __restrict__ activations,
    const unsigned int token_count, const unsigned int task_count) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  __shared__ std::uint16_t decoded_weights[256U];
  decoded_weights[threadIdx.x] = decode_e4m3fn_to_bf16_bits(
      static_cast<std::uint8_t>(threadIdx.x));
  decoded_weights[threadIdx.x + kThreads] = decode_e4m3fn_to_bf16_bits(
      static_cast<std::uint8_t>(threadIdx.x + kThreads));
  __syncthreads();

  const unsigned int grid_m = (token_count + kTileM - 1U) / kTileM;
  constexpr unsigned int kGridN =
      kTopology == GroupTopology::kAttentionOutput ? 40U : 128U;
  constexpr unsigned int kGroupM =
      kTopology == GroupTopology::kAttentionOutput ? 3U : 4U;
  for (unsigned int task = blockIdx.x; task < task_count;
       task += gridDim.x) {
    unsigned int m_tile = 0U;
    unsigned int global_n_tile = 0U;
    map_grouped_task(task, grid_m, kGridN, kGroupM, &m_tile,
                     &global_n_tile);
    unsigned int nblock = 0U;
    const DevicePartition partition = select_partition(
        partition0, partition1, partition2, global_n_tile, &nblock);

    if constexpr (kTopology == GroupTopology::kFullInput) {
      if (partition.tactic == DeviceTactic::kSmallKvK5120) {
        execute_small_task<5'120U>(
            reinterpret_cast<SmallPipelineStorage*>(dynamic_storage),
            partition, activations, decoded_weights, token_count, m_tile,
            nblock);
      } else {
        execute_wide_task<5'120U>(
            reinterpret_cast<WidePipelineStorage*>(dynamic_storage),
            partition, activations, decoded_weights, token_count, m_tile,
            nblock);
      }
    } else if constexpr (kTopology == GroupTopology::kAttentionOutput) {
      execute_wide_task<6'144U>(
          reinterpret_cast<WidePipelineStorage*>(dynamic_storage),
          partition, activations, decoded_weights, token_count, m_tile,
          nblock);
    } else {
      execute_wide_task<5'120U>(
          reinterpret_cast<WidePipelineStorage*>(dynamic_storage),
          partition, activations, decoded_weights, token_count, m_tile,
          nblock);
    }
  }
}

__global__ __launch_bounds__(kThreads) void pack_wide_fp8_kernel(
    const std::uint8_t* __restrict__ canonical,
    std::uint32_t* __restrict__ sidecar, const unsigned int columns,
    const unsigned int kstage_count) {
  const unsigned int tile = blockIdx.x;
  const unsigned int nblock = tile / kstage_count;
  const unsigned int kstage = tile % kstage_count;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  std::uint32_t* const sidecar_stage =
      sidecar + static_cast<std::size_t>(tile) *
                    (kWideTileN * kWideTileK / sizeof(std::uint32_t));

#pragma unroll
  for (unsigned int k16 = 0U; k16 < kWideK16PerStage; ++k16) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kWideN8PerWarp;
         ++n_panel) {
      const unsigned int row = nblock * kWideTileN + warp * 32U +
                               n_panel * 8U + lane_group;
      const unsigned int column = kstage * kWideTileK + k16 * 16U +
                                  2U * lane_in_group;
      const std::uint8_t* const source =
          canonical + static_cast<std::size_t>(row) * columns + column;
      sidecar_stage
          [((k16 * 4U + warp) * kWideN8PerWarp + n_panel) *
               kWarpSize +
           lane] =
          static_cast<std::uint32_t>(source[0]) |
          (static_cast<std::uint32_t>(source[8]) << 8U) |
          (static_cast<std::uint32_t>(source[1]) << 16U) |
          (static_cast<std::uint32_t>(source[9]) << 24U);
    }
  }
}

__global__ __launch_bounds__(kThreads) void pack_small_fp8_kernel(
    const std::uint8_t* __restrict__ canonical,
    std::uint32_t* __restrict__ sidecar, const unsigned int columns,
    const unsigned int kstage_count) {
  const unsigned int tile = blockIdx.x;
  const unsigned int nblock = tile / kstage_count;
  const unsigned int kstage = tile % kstage_count;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  std::uint32_t* const sidecar_stage =
      sidecar + static_cast<std::size_t>(tile) *
                    (kSmallTileN * kSmallTileK / sizeof(std::uint32_t));

#pragma unroll
  for (unsigned int k16 = 0U; k16 < kSmallK16PerStage; ++k16) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kSmallN8PerWarp;
         ++n_panel) {
      const unsigned int row = nblock * kSmallTileN + warp * 16U +
                               n_panel * 8U + lane_group;
      const unsigned int column = kstage * kSmallTileK + k16 * 16U +
                                  2U * lane_in_group;
      const std::uint8_t* const source =
          canonical + static_cast<std::size_t>(row) * columns + column;
      sidecar_stage
          [((k16 * 4U + warp) * kSmallN8PerWarp + n_panel) *
               kWarpSize +
           lane] =
          static_cast<std::uint32_t>(source[0]) |
          (static_cast<std::uint32_t>(source[8]) << 8U) |
          (static_cast<std::uint32_t>(source[1]) << 16U) |
          (static_cast<std::uint32_t>(source[9]) << 24U);
    }
  }
}

[[nodiscard]] constexpr bool is_fp8_role(
    const Sm87P40PackedProjectionRole role) noexcept {
  return role == Sm87P40PackedProjectionRole::kFp8LinearQkvZ ||
         role == Sm87P40PackedProjectionRole::kFp8FullQkv ||
         role == Sm87P40PackedProjectionRole::kFp8AttentionOutput;
}

[[nodiscard]] constexpr DeviceTactic device_tactic(
    const Sm87P40PackedTactic tactic) noexcept {
  if (tactic == Sm87P40PackedTactic::kFp8SmallKvM64N64K128Gm4) {
    return DeviceTactic::kSmallKvK5120;
  }
  if (tactic == Sm87P40PackedTactic::kFp8OutputM64N128K64Gm3) {
    return DeviceTactic::kOutputK6144;
  }
  return DeviceTactic::kWideK5120;
}

[[nodiscard]] cudaError_t validate_device_pointer(
    const void* const pointer, const int expected_device) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    return status;
  }
#if CUDART_VERSION >= 10000
  return attributes.type == cudaMemoryTypeDevice &&
                 attributes.device == expected_device
             ? cudaSuccess
             : cudaErrorInvalidValue;
#else
  return attributes.memoryType == cudaMemoryTypeDevice &&
                 attributes.device == expected_device
             ? cudaSuccess
             : cudaErrorInvalidValue;
#endif
}

[[nodiscard]] cudaError_t current_sm87_device(
    int* const device, cudaDeviceProp* const properties) noexcept {
  if (device == nullptr || properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  cudaError_t status = cudaGetDevice(device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(properties, *device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     static_cast<int>(kRequiredSmCount)
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] cudaError_t validate_fp8_prepare(
    const Sm87P40PackedProjectionRole role,
    const Sm87P40PackedCanonicalSource* const sources,
    const std::size_t source_count, std::uint8_t* const destination,
    const std::size_t destination_bytes,
    Sm87P40PackedProjectionPlan* const plan) noexcept {
  if (!kPackedProjectionAdmitted || !is_fp8_role(role)) {
    return cudaErrorNotSupported;
  }
  if (sources == nullptr || destination == nullptr || plan == nullptr) {
    return cudaErrorInvalidValue;
  }
  *plan = sm87_p40_packed_projection_plan(role);
  if (!plan->valid() || plan->source_count != source_count ||
      plan->payload_bytes != destination_bytes ||
      !aligned(destination, kSm87P40PackedProjectionPayloadAlignment)) {
    return cudaErrorInvalidValue;
  }

  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = current_sm87_device(&device, &properties);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_device_pointer(destination, device);
  if (status != cudaSuccess) {
    return status;
  }
  const ByteRange destination_range =
      make_range(destination, destination_bytes);
  if (!destination_range.valid) {
    return cudaErrorInvalidValue;
  }

  ByteRange weight_ranges[kSm87P40PackedProjectionMaximumSources]{};
  ByteRange scale_ranges[kSm87P40PackedProjectionMaximumSources]{};
  for (std::size_t index = 0U; index < source_count; ++index) {
    const auto& source = sources[index];
    const auto& partition = plan->partitions[index];
    if (source.role != partition.role || source.weight == nullptr ||
        source.block_scale != nullptr ||
        source.global_scale_device == nullptr ||
        source.output_features != partition.output_features ||
        source.input_features != partition.input_features ||
        !aligned(source.weight, alignof(uint4)) ||
        !aligned(source.global_scale_device, alignof(float)) ||
        partition.payload_bytes !=
            source.output_features * source.input_features) {
      return cudaErrorInvalidValue;
    }
    status = validate_device_pointer(source.weight, device);
    if (status != cudaSuccess) {
      return status;
    }
    status = validate_device_pointer(source.global_scale_device, device);
    if (status != cudaSuccess) {
      return status;
    }
    weight_ranges[index] = make_range(
        source.weight, source.output_features * source.input_features);
    scale_ranges[index] =
        make_range(source.global_scale_device, sizeof(float));
    if (!weight_ranges[index].valid || !scale_ranges[index].valid ||
        overlaps(weight_ranges[index], destination_range) ||
        overlaps(scale_ranges[index], destination_range) ||
        overlaps(weight_ranges[index], scale_ranges[index])) {
      return cudaErrorInvalidValue;
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (overlaps(weight_ranges[index], weight_ranges[prior]) ||
          overlaps(weight_ranges[index], scale_ranges[prior]) ||
          overlaps(scale_ranges[index], weight_ranges[prior]) ||
          overlaps(scale_ranges[index], scale_ranges[prior])) {
        return cudaErrorInvalidValue;
      }
    }
  }
  return cudaSuccess;
}

template <GroupTopology kTopology>
[[nodiscard]] constexpr std::size_t dynamic_shared_bytes() noexcept {
  return kTopology == GroupTopology::kFullInput
             ? kSmallDynamicSharedBytes
             : kWideDynamicSharedBytes;
}

template <GroupTopology kTopology>
[[nodiscard]] cudaError_t configure_fp8_kernel() noexcept {
  return cudaFuncSetAttribute(
      p40_packed_fp8_kernel<kTopology>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(dynamic_shared_bytes<kTopology>()));
}

template <GroupTopology kTopology>
[[nodiscard]] cudaError_t read_fp8_resources(
    const cudaDeviceProp& properties,
    Sm87P40PackedProjectionResources* const resources) noexcept {
  cudaError_t status = configure_fp8_kernel<kTopology>();
  if (status != cudaSuccess) {
    return status;
  }
  const auto kernel = p40_packed_fp8_kernel<kTopology>;
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, static_cast<int>(kThreads),
      dynamic_shared_bytes<kTopology>());
  if (status != cudaSuccess) {
    return status;
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = dynamic_shared_bytes<kTopology>();
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;

  const std::size_t shared_per_cta =
      resources->static_shared_bytes + resources->dynamic_shared_bytes;
  const bool two_ctas_fit_shared =
      shared_per_cta <=
      static_cast<std::size_t>(properties.sharedMemPerMultiprocessor) / 2U;
  return resources->local_bytes == 0U &&
                 resources->maximum_threads_per_block >=
                     static_cast<int>(kThreads) &&
                 resources->active_blocks_per_sm >= 2 &&
                 two_ctas_fit_shared
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] cudaError_t validate_fp8_artifact(
    const std::uint16_t* const input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    const std::size_t token_count,
    const std::array<std::uint16_t*,
                     kSm87P40PackedProjectionMaximumSources>& outputs,
    Sm87P40PackedProjectionPlan* const plan,
    DevicePartition (&partitions)[kSm87P40PackedProjectionMaximumSources])
    noexcept {
  if (!kPackedProjectionAdmitted || !is_fp8_role(artifact.role)) {
    return cudaErrorNotSupported;
  }
  if (input == nullptr || artifact.payload == nullptr || plan == nullptr ||
      artifact.artifact_identity == 0U ||
      token_count != kSm87P40PackedProjectionTokens) {
    return cudaErrorInvalidValue;
  }
  *plan = sm87_p40_packed_projection_plan(artifact.role);
  if (!plan->valid() || artifact.tactic != plan->tactic ||
      artifact.source_count != plan->source_count ||
      artifact.payload_bytes != plan->payload_bytes ||
      !aligned(input, alignof(uint4)) ||
      !aligned(artifact.payload,
               kSm87P40PackedProjectionPayloadAlignment)) {
    return cudaErrorInvalidValue;
  }

  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status = current_sm87_device(&device, &properties);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_device_pointer(input, device);
  if (status != cudaSuccess) {
    return status;
  }
  status = validate_device_pointer(artifact.payload, device);
  if (status != cudaSuccess) {
    return status;
  }
  const std::size_t input_features =
      artifact.role == Sm87P40PackedProjectionRole::kFp8AttentionOutput
          ? 6'144U
          : 5'120U;
  const ByteRange input_range = make_range(
      input, token_count * input_features * sizeof(std::uint16_t));
  const ByteRange payload_range =
      make_range(artifact.payload, artifact.payload_bytes);
  if (!input_range.valid || !payload_range.valid ||
      overlaps(input_range, payload_range)) {
    return cudaErrorInvalidValue;
  }

  ByteRange output_ranges[kSm87P40PackedProjectionMaximumSources]{};
  for (std::size_t index = 0U; index < plan->partitions.size(); ++index) {
    if (index >= plan->source_count) {
      if (outputs[index] != nullptr) {
        return cudaErrorInvalidValue;
      }
      continue;
    }
    const auto& partition = plan->partitions[index];
    if (outputs[index] == nullptr ||
        !aligned(outputs[index], alignof(std::uint32_t)) ||
        !std::isfinite(artifact.scalar_scales[index]) ||
        artifact.scalar_scales[index] < 0.0F) {
      return cudaErrorInvalidValue;
    }
    status = validate_device_pointer(outputs[index], device);
    if (status != cudaSuccess) {
      return status;
    }
    output_ranges[index] = make_range(
        outputs[index], token_count * partition.output_features *
                            sizeof(std::uint16_t));
    if (!output_ranges[index].valid ||
        overlaps(output_ranges[index], input_range) ||
        overlaps(output_ranges[index], payload_range)) {
      return cudaErrorInvalidValue;
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (overlaps(output_ranges[index], output_ranges[prior])) {
        return cudaErrorInvalidValue;
      }
    }
    partitions[index] = {
        reinterpret_cast<const uint4*>(
            artifact.payload + partition.payload_offset),
        outputs[index], artifact.scalar_scales[index],
        partition.output_features, partition.task_n_tiles,
        partition.first_task_n_tile, device_tactic(partition.tactic)};
  }
  return cudaSuccess;
}

template <GroupTopology kTopology>
[[nodiscard]] int launch_fp8_kernel(
    const DevicePartition (&partitions)
        [kSm87P40PackedProjectionMaximumSources],
    const std::uint16_t* const input,
    const Sm87P40PackedProjectionPlan& plan,
    void* const cuda_stream) noexcept {
  cudaError_t status = configure_fp8_kernel<kTopology>();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  p40_packed_fp8_kernel<kTopology>
      <<<kPersistentCtas, kThreads, dynamic_shared_bytes<kTopology>(),
         reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
          partitions[0U], partitions[1U], partitions[2U], input,
          plan.token_count, static_cast<unsigned int>(plan.logical_tasks));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int prepare_sm87_p40_packed_fp8_projection_cuda(
    const Sm87P40PackedProjectionRole role,
    const Sm87P40PackedCanonicalSource* const sources,
    const std::size_t source_count, std::uint8_t* const destination,
    const std::size_t destination_bytes,
    void* const cuda_stream) noexcept {
  Sm87P40PackedProjectionPlan plan{};
  const cudaError_t validation = validate_fp8_prepare(
      role, sources, source_count, destination, destination_bytes, &plan);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  (void)cudaGetLastError();
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  for (std::size_t index = 0U; index < source_count; ++index) {
    const auto& source = sources[index];
    const auto& partition = plan.partitions[index];
    const unsigned int nblocks =
        partition.output_features / partition.tile_n;
    const unsigned int kstages =
        partition.input_features / partition.tile_k;
    const unsigned int blocks = nblocks * kstages;
    auto* const sidecar = reinterpret_cast<std::uint32_t*>(
        destination + partition.payload_offset);
    if (partition.tactic ==
        Sm87P40PackedTactic::kFp8SmallKvM64N64K128Gm4) {
      pack_small_fp8_kernel<<<blocks, kThreads, 0U, stream>>>(
          source.weight, sidecar, partition.input_features, kstages);
    } else {
      pack_wide_fp8_kernel<<<blocks, kThreads, 0U, stream>>>(
          source.weight, sidecar, partition.input_features, kstages);
    }
    const cudaError_t status = cudaPeekAtLastError();
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
  }
  return static_cast<int>(cudaSuccess);
}

int query_sm87_p40_packed_fp8_resources_cuda(
    const Sm87P40PackedProjectionRole role,
    Sm87P40PackedProjectionResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  if (!kPackedProjectionAdmitted || !is_fp8_role(role) ||
      !sm87_p40_packed_projection_plan(role).valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  int device = -1;
  cudaDeviceProp properties{};
  const cudaError_t status = current_sm87_device(&device, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (role == Sm87P40PackedProjectionRole::kFp8LinearQkvZ) {
    return static_cast<int>(read_fp8_resources<GroupTopology::kLinearInput>(
        properties, resources));
  }
  if (role == Sm87P40PackedProjectionRole::kFp8FullQkv) {
    return static_cast<int>(read_fp8_resources<GroupTopology::kFullInput>(
        properties, resources));
  }
  return static_cast<int>(
      read_fp8_resources<GroupTopology::kAttentionOutput>(properties,
                                                          resources));
}

int launch_sm87_p40_packed_fp8_cuda(
    const std::uint16_t* const input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    const std::size_t token_count,
    const std::array<std::uint16_t*,
                     kSm87P40PackedProjectionMaximumSources>& outputs,
    void* const cuda_stream) noexcept {
  Sm87P40PackedProjectionPlan plan{};
  DevicePartition partitions[kSm87P40PackedProjectionMaximumSources]{};
  cudaError_t status = validate_fp8_artifact(
      input, artifact, token_count, outputs, &plan, partitions);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  Sm87P40PackedProjectionResources resources{};
  status = static_cast<cudaError_t>(
      query_sm87_p40_packed_fp8_resources_cuda(artifact.role, &resources));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  (void)cudaGetLastError();
  if (artifact.role == Sm87P40PackedProjectionRole::kFp8LinearQkvZ) {
    return launch_fp8_kernel<GroupTopology::kLinearInput>(
        partitions, input, plan, cuda_stream);
  }
  if (artifact.role == Sm87P40PackedProjectionRole::kFp8FullQkv) {
    return launch_fp8_kernel<GroupTopology::kFullInput>(
        partitions, input, plan, cuda_stream);
  }
  return launch_fp8_kernel<GroupTopology::kAttentionOutput>(
      partitions, input, plan, cuda_stream);
}

}  // namespace q3x::kernels
