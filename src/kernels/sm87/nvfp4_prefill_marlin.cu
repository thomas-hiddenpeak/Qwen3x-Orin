#include "q3x/kernels/sm87_nvfp4_prefill_marlin.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = 8U;
constexpr unsigned int kThreads = kWarpSize * kWarpsPerBlock;
constexpr unsigned int kRows = 17'408U;
constexpr unsigned int kColumns = 5'120U;
constexpr unsigned int kTokenCount = 512U;
constexpr unsigned int kResidentTokens = 64U;
constexpr unsigned int kOutputRowsPerBlock = 256U;
constexpr unsigned int kColumnsPerStage = 64U;
constexpr unsigned int kGlobalPipelineStages = 4U;
constexpr unsigned int kRegisterPipelineSlots = 2U;
constexpr unsigned int kKStages = kColumns / kColumnsPerStage;
constexpr unsigned int kOutputRowBlocks = kRows / kOutputRowsPerBlock;
constexpr unsigned int kTokenTiles = kTokenCount / kResidentTokens;
constexpr unsigned int kPersistentPhases = 4U;
constexpr unsigned int kPersistentBlocks =
    kTokenTiles * kPersistentPhases;
constexpr unsigned int kLogicalBranches = 2U;
constexpr unsigned int kLogicalOutputBlocks =
    kLogicalBranches * kOutputRowBlocks;
constexpr unsigned int kPackedColumns = kColumns / 2U;
constexpr unsigned int kScaleColumns = kColumns / 16U;
constexpr unsigned int kSharedActivationLeadingDimension = 72U;

// One N256/K64 cell.  Payload words are already permuted into the lane order
// consumed by the eight warps.  E4M3 scales deliberately remain one byte each;
// a CTA-local exact code table converts each possible byte once, and register
// slots select the corresponding BF16 scale on demand.
constexpr unsigned int kPackedWordsPerSidecarStage =
    4U * 4U * kThreads;
constexpr unsigned int kPackedBytesPerSidecarStage =
    kPackedWordsPerSidecarStage * sizeof(std::uint16_t);
constexpr unsigned int kScaleBytesPerSidecarStage =
    4U * kOutputRowsPerBlock;
constexpr unsigned int kSidecarBytesPerStage =
    kPackedBytesPerSidecarStage + kScaleBytesPerSidecarStage;
constexpr std::size_t kSidecarBytesPerProjection =
    static_cast<std::size_t>(kOutputRowBlocks) * kKStages *
    kSidecarBytesPerStage;
constexpr std::size_t kCanonicalPackedBytes =
    static_cast<std::size_t>(kRows) * kPackedColumns;
constexpr std::size_t kCanonicalScaleBytes =
    static_cast<std::size_t>(kRows) * kScaleColumns;
constexpr std::size_t kActivationBytes =
    static_cast<std::size_t>(kTokenCount) * kColumns *
    sizeof(std::uint16_t);
constexpr std::size_t kOutputBytes =
    static_cast<std::size_t>(kTokenCount) * kRows *
    sizeof(std::uint16_t);

static_assert(kKStages == 80U);
static_assert(kOutputRowBlocks == 68U);
static_assert(kTokenTiles == 8U);
static_assert(kPersistentBlocks == 32U);
static_assert(kLogicalOutputBlocks == 136U);
static_assert(kPackedWordsPerSidecarStage == 4'096U);
static_assert(kPackedBytesPerSidecarStage == 8'192U);
static_assert(kScaleBytesPerSidecarStage == 1'024U);
static_assert(kSidecarBytesPerStage == 9'216U);
static_assert(kSidecarBytesPerProjection == 50'135'040U);

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ float decode_e4m3fn(
    const std::uint8_t bits) {
  const unsigned int sign =
      static_cast<unsigned int>(bits & 0x80U) << 24U;
  const unsigned int magnitude = static_cast<unsigned int>(bits & 0x7fU);
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  if (magnitude == 0x7fU) {
    return __uint_as_float(sign | 0x7fc0'0000U);
  }
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
    const unsigned int leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const unsigned int fp32_exponent = 118U + leading;
    const unsigned int fp32_mantissa =
        (mantissa - (1U << leading)) << (23U - leading);
    return __uint_as_float(sign | (fp32_exponent << 23U) |
                           fp32_mantissa);
  }
  return __uint_as_float(sign | ((120U + exponent) << 23U) |
                         (mantissa << 20U));
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t multiply_bf16x2_bits(
    const std::uint32_t value_bits, const std::uint16_t scale_bits) {
  const __nv_bfloat162_raw value_raw{
      static_cast<std::uint16_t>(value_bits),
      static_cast<std::uint16_t>(value_bits >> 16U)};
  const __nv_bfloat162_raw scale_raw{scale_bits, scale_bits};
  const __nv_bfloat162_raw result_raw = static_cast<__nv_bfloat162_raw>(
      __hmul2_rn(__nv_bfloat162(value_raw), __nv_bfloat162(scale_raw)));
  return static_cast<std::uint32_t>(result_raw.x) |
         (static_cast<std::uint32_t>(result_raw.y) << 16U);
}

// Exact lower-half specialization of the production E2M1 decoder.  PRMT
// constructs four BF16 values and the raw E4M3 byte has already been decoded
// to one BF16 register value by the register-slot loader.
[[nodiscard]] __device__ __forceinline__ uint2 decode_nvfp4x4(
    const std::uint16_t packed, const std::uint16_t decoded_scale) {
  constexpr std::uint32_t kLowBytes0To3 = 0xc080'0000U;
  constexpr std::uint32_t kLowBytes4To7 = 0xc080'4000U;
  constexpr std::uint32_t kHighBytes0To3 = 0x3f3f'3f00U;
  constexpr std::uint32_t kHighBytes4To7 = 0x4040'4040U;
  constexpr std::uint32_t kMagnitudeSelectorMask = 0x7777U;
  constexpr std::uint32_t kSignByteMask = 0x8080'8080U;
  constexpr std::uint32_t kFirstFourSignSelector = 0xd9c8U;
  constexpr std::uint32_t kFirstPairInterleave = 0x5140U;
  constexpr std::uint32_t kSecondPairInterleave = 0x7362U;

  const std::uint32_t packed32 = static_cast<std::uint32_t>(packed);
  const std::uint32_t selector = packed32 & kMagnitudeSelectorMask;
  const std::uint32_t low =
      __byte_perm(kLowBytes0To3, kLowBytes4To7, selector);
  const std::uint32_t signs =
      __byte_perm(packed32 << 4U, packed32, kFirstFourSignSelector) &
      kSignByteMask;
  const std::uint32_t high =
      __byte_perm(kHighBytes0To3, kHighBytes4To7, selector) | signs;

  uint2 result{};
  result.x = multiply_bf16x2_bits(
      __byte_perm(low, high, kFirstPairInterleave), decoded_scale);
  result.y = multiply_bf16x2_bits(
      __byte_perm(low, high, kSecondPairInterleave), decoded_scale);
  return result;
}

__device__ __forceinline__ void cp_async_ca_shared_global_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(global_source)
               : "memory");
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      *reinterpret_cast<const uint4*>(global_source);
#endif
}

__device__ __forceinline__ void cp_async_cg_shared_global_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
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

template <unsigned int PendingGroups>
__device__ __forceinline__ void cp_async_wait_group();

template <>
__device__ __forceinline__ void cp_async_wait_group<0U>() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

template <>
__device__ __forceinline__ void cp_async_wait_group<1U>() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

template <>
__device__ __forceinline__ void cp_async_wait_group<2U>() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 2;" ::: "memory");
#endif
}

template <>
__device__ __forceinline__ void cp_async_wait_group<3U>() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 3;" ::: "memory");
#endif
}

struct InlineM16N8Accumulator {
  float x0;
  float x1;
  float x2;
  float x3;
};

__device__ __forceinline__ void mma_m16n8k16_bf16(
    InlineM16N8Accumulator& accumulator, const std::uint32_t a0,
    const std::uint32_t a1, const std::uint32_t a2,
    const std::uint32_t a3, const std::uint32_t b0,
    const std::uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
pack_bf16_fragment_pair(const __nv_bfloat16 low,
                        const __nv_bfloat16 high) {
  return static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
         (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
pack_scaled_bf16_output_pair(const float low, const float high,
                             const float scale) {
  return static_cast<std::uint32_t>(encode_bf16_rne(low * scale)) |
         (static_cast<std::uint32_t>(encode_bf16_rne(high * scale)) << 16U);
}

struct alignas(32) MarlinSharedStage {
  uint4 activations[576];  // M64 * LD72 BF16 = 9,216 B.
  uint4 sidecar[576];      // Compact fragment-native N256/K64 = 9,216 B.
};

struct alignas(32) MarlinSharedStorage {
  MarlinSharedStage pipeline[kGlobalPipelineStages];
};

constexpr unsigned int kDynamicSharedBytes = sizeof(MarlinSharedStorage);
static_assert(sizeof(MarlinSharedStage) == 18'432U);
static_assert(kDynamicSharedBytes == 73'728U);

// Two slots keep the decoded B fragments one K16 ahead of MMA.  A fragments
// are loaded just in time per M16 panel; retaining four complete A panels in
// both slots would consume 32 extra registers and collapse the exact cell to
// one resident CTA.
struct WeightRegisterSlot {
  uint2 weights[4];
};

__global__ void nvfp4_prefill_marlin_pack_kernel(
    const std::uint8_t* const canonical_packed_weights,
    const std::uint8_t* const canonical_block_scales,
    std::uint8_t* const sidecar) {
  const unsigned int cell = blockIdx.x;
  const unsigned int output_block = cell / kKStages;
  const unsigned int k_stage = cell % kKStages;
  const unsigned int local_thread = threadIdx.x;
  const unsigned int warp = local_thread / kWarpSize;
  const unsigned int lane = local_thread % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  auto* const stage_bytes =
      sidecar + static_cast<std::size_t>(cell) * kSidecarBytesPerStage;
  auto* const stage_words = reinterpret_cast<std::uint16_t*>(stage_bytes);

#pragma unroll
  for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
      const unsigned int local_row =
          warp * 32U + n_panel * 8U + lane_group;
      const unsigned int output_row =
          output_block * kOutputRowsPerBlock + local_row;
      const auto* const canonical_row =
          canonical_packed_weights +
          static_cast<std::size_t>(output_row) * kPackedColumns +
          k_stage * 32U + k16 * 8U;
      const std::uint16_t packed =
          static_cast<std::uint16_t>(canonical_row[lane_in_group]) |
          static_cast<std::uint16_t>(
              static_cast<unsigned int>(canonical_row[lane_in_group + 4U])
              << 8U);
      stage_words[(k16 * 4U + n_panel) * kThreads + local_thread] =
          packed;
    }
  }

  auto* const stage_scales = stage_bytes + kPackedBytesPerSidecarStage;
  const unsigned int output_row =
      output_block * kOutputRowsPerBlock + local_thread;
  const auto* const canonical_scale_row =
      canonical_block_scales +
      static_cast<std::size_t>(output_row) * kScaleColumns + k_stage * 4U;
#pragma unroll
  for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
    stage_scales[k16 * kOutputRowsPerBlock + local_thread] =
        canonical_scale_row[k16];
  }
}

__device__ __forceinline__ void issue_pipeline_stage(
    MarlinSharedStage* const shared,
    const std::uint8_t* const sidecar_cell,
    const std::uint16_t* const tile_activations,
    const unsigned int k_stage) {
  constexpr unsigned int kActivationChunksPerToken = 8U;
  constexpr unsigned int kSharedChunksPerToken = 9U;
  constexpr unsigned int kActivationChunks =
      kResidentTokens * kActivationChunksPerToken;
  static_assert(kActivationChunks == 2U * kThreads);

#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    const unsigned int token = index / kActivationChunksPerToken;
    const unsigned int chunk = index % kActivationChunksPerToken;
    cp_async_ca_shared_global_16(
        shared->activations + token * kSharedChunksPerToken + chunk,
        reinterpret_cast<const uint4*>(
            tile_activations + static_cast<std::size_t>(token) * kColumns +
            k_stage * kColumnsPerStage) +
            chunk);
  }

  constexpr unsigned int kSidecarChunks =
      kSidecarBytesPerStage / sizeof(uint4);
  static_assert(kSidecarChunks == 576U);
  for (unsigned int index = threadIdx.x; index < kSidecarChunks;
       index += kThreads) {
    cp_async_cg_shared_global_16(
        shared->sidecar + index,
        reinterpret_cast<const uint4*>(sidecar_cell) + index);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void load_weight_register_slot(
    WeightRegisterSlot& slot, const MarlinSharedStage* const shared,
    const std::uint16_t* const decoded_scale_values,
    const unsigned int k16) {
  const auto* const sidecar_words =
      reinterpret_cast<const std::uint16_t*>(shared->sidecar);
  const auto* const sidecar_scales =
      reinterpret_cast<const std::uint8_t*>(shared->sidecar) +
      kPackedBytesPerSidecarStage;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int lane_group = lane / 4U;
#pragma unroll
  for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
    const unsigned int local_row =
        warp * 32U + n_panel * 8U + lane_group;
    const std::uint16_t packed =
        sidecar_words[(k16 * 4U + n_panel) * kThreads + threadIdx.x];
    const std::uint16_t decoded_scale = decoded_scale_values[
        sidecar_scales[k16 * kOutputRowsPerBlock + local_row]];
    slot.weights[n_panel] = decode_nvfp4x4(packed, decoded_scale);
  }
}

__device__ __forceinline__ void store_branch_output(
    const InlineM16N8Accumulator (&accumulators)[4][4],
    const float weight_scale_2, const unsigned int first_token,
    const unsigned int first_output_row, std::uint16_t* const output) {
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const bool even_group = (lane & 4U) == 0U;
  const unsigned int octet = lane >> 3U;

#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 4U; ++m_panel) {
#pragma unroll
    for (unsigned int n_pair = 0U; n_pair < 2U; ++n_pair) {
      const InlineM16N8Accumulator accumulator_a =
          accumulators[m_panel][2U * n_pair];
      const InlineM16N8Accumulator accumulator_b =
          accumulators[m_panel][2U * n_pair + 1U];
      const unsigned int local_column =
          warp * 32U + n_pair * 16U + 2U * (lane & 7U);

      const std::uint32_t low_a = pack_scaled_bf16_output_pair(
          accumulator_a.x0, accumulator_a.x1, weight_scale_2);
      const std::uint32_t low_b = pack_scaled_bf16_output_pair(
          accumulator_b.x0, accumulator_b.x1, weight_scale_2);
      const std::uint32_t low_peer = __shfl_xor_sync(
          0xffff'ffffU, even_group ? low_b : low_a, 4, 8);
      const std::uint32_t low_even = even_group ? low_a : low_peer;
      const std::uint32_t low_odd = even_group ? low_peer : low_b;
      const unsigned int low_even_token = m_panel * 16U + 2U * octet;
      *reinterpret_cast<std::uint32_t*>(
          output + static_cast<std::size_t>(first_token + low_even_token) *
                       kRows +
          first_output_row + local_column) = low_even;
      *reinterpret_cast<std::uint32_t*>(
          output + static_cast<std::size_t>(first_token + low_even_token + 1U) *
                       kRows +
          first_output_row + local_column) = low_odd;

      const std::uint32_t high_a = pack_scaled_bf16_output_pair(
          accumulator_a.x2, accumulator_a.x3, weight_scale_2);
      const std::uint32_t high_b = pack_scaled_bf16_output_pair(
          accumulator_b.x2, accumulator_b.x3, weight_scale_2);
      const std::uint32_t high_peer = __shfl_xor_sync(
          0xffff'ffffU, even_group ? high_b : high_a, 4, 8);
      const std::uint32_t high_even = even_group ? high_a : high_peer;
      const std::uint32_t high_odd = even_group ? high_peer : high_b;
      const unsigned int high_even_token = low_even_token + 8U;
      *reinterpret_cast<std::uint32_t*>(
          output + static_cast<std::size_t>(first_token + high_even_token) *
                       kRows +
          first_output_row + local_column) = high_even;
      *reinterpret_cast<std::uint32_t*>(
          output + static_cast<std::size_t>(first_token + high_even_token + 1U) *
                       kRows +
          first_output_row + local_column) = high_odd;
    }
  }
}

__global__ __launch_bounds__(kThreads, 2) void
nvfp4_prefill_marlin_pair_kernel(
    const std::uint8_t* const gate_sidecar,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_sidecar,
    const float up_weight_scale_2,
    const std::uint16_t* const activations,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const shared =
      reinterpret_cast<MarlinSharedStorage*>(dynamic_storage);
  __shared__ std::uint16_t decoded_scale_values[256];
  decoded_scale_values[threadIdx.x] =
      encode_bf16_rne(decode_e4m3fn(static_cast<std::uint8_t>(threadIdx.x)));
  __syncthreads();
  const unsigned int token_tile = blockIdx.x % kTokenTiles;
  const unsigned int work_phase = blockIdx.x / kTokenTiles;
  const unsigned int first_token = token_tile * kResidentTokens;
  const auto* const tile_activations =
      activations + static_cast<std::size_t>(first_token) * kColumns;

  // The flattened work queue is [Gate N tiles, Up N tiles].  A CTA completes
  // one N256 item before choosing the next branch, so shared/register B state
  // never contains both projections at once.
  for (unsigned int logical_output_block = work_phase;
       logical_output_block < kLogicalOutputBlocks;
       logical_output_block += kPersistentPhases) {
    const bool up_branch = logical_output_block >= kOutputRowBlocks;
    const unsigned int output_block =
        up_branch ? logical_output_block - kOutputRowBlocks
                  : logical_output_block;
    const auto* const branch_sidecar =
        up_branch ? up_sidecar : gate_sidecar;
    const float branch_scale =
        up_branch ? up_weight_scale_2 : gate_weight_scale_2;
    auto* const branch_output = up_branch ? up_output : gate_output;

    InlineM16N8Accumulator accumulators[4][4];
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 4U; ++m_panel) {
#pragma unroll
      for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
        accumulators[m_panel][n_panel] =
            InlineM16N8Accumulator{0.0F, 0.0F, 0.0F, 0.0F};
      }
    }

    const std::size_t first_sidecar_cell =
        static_cast<std::size_t>(output_block) * kKStages;
#pragma unroll
    for (unsigned int stage = 0U; stage < kGlobalPipelineStages; ++stage) {
      issue_pipeline_stage(
          &shared->pipeline[stage],
          branch_sidecar +
              (first_sidecar_cell + stage) * kSidecarBytesPerStage,
          tile_activations, stage);
    }

#pragma unroll 1
    for (unsigned int stage = 0U; stage < kKStages; ++stage) {
      const unsigned int stages_after = kKStages - stage - 1U;
      if (stages_after >= 3U) {
        cp_async_wait_group<3U>();
      } else if (stages_after == 2U) {
        cp_async_wait_group<2U>();
      } else if (stages_after == 1U) {
        cp_async_wait_group<1U>();
      } else {
        cp_async_wait_group<0U>();
      }
      __syncthreads();

      const unsigned int shared_slot = stage % kGlobalPipelineStages;
      WeightRegisterSlot registers[kRegisterPipelineSlots];
      load_weight_register_slot(registers[0],
                                &shared->pipeline[shared_slot],
                                decoded_scale_values, 0U);
#pragma unroll
      for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
        const unsigned int register_slot =
            k16 % kRegisterPipelineSlots;
        if (k16 + 1U < 4U) {
          load_weight_register_slot(
              registers[(k16 + 1U) % kRegisterPipelineSlots],
              &shared->pipeline[shared_slot], decoded_scale_values,
              k16 + 1U);
        }
        namespace wmma = nvcuda::wmma;
        const auto* const shared_a =
            reinterpret_cast<const __nv_bfloat16*>(
                shared->pipeline[shared_slot].activations);
#pragma unroll
        for (unsigned int m_panel = 0U; m_panel < 4U; ++m_panel) {
          wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                         wmma::row_major>
              activation_fragment;
          wmma::load_matrix_sync(
              activation_fragment,
              shared_a +
                  m_panel * 16U * kSharedActivationLeadingDimension +
                  k16 * 16U,
              kSharedActivationLeadingDimension);
          const std::uint32_t a0 = pack_bf16_fragment_pair(
              activation_fragment.x[0], activation_fragment.x[1]);
          const std::uint32_t a1 = pack_bf16_fragment_pair(
              activation_fragment.x[2], activation_fragment.x[3]);
          const std::uint32_t a2 = pack_bf16_fragment_pair(
              activation_fragment.x[4], activation_fragment.x[5]);
          const std::uint32_t a3 = pack_bf16_fragment_pair(
              activation_fragment.x[6], activation_fragment.x[7]);
#pragma unroll
          for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
            mma_m16n8k16_bf16(
                accumulators[m_panel][n_panel],
                a0, a1, a2, a3,
                registers[register_slot].weights[n_panel].x,
                registers[register_slot].weights[n_panel].y);
          }
        }
      }
      __syncthreads();

      if (stage + kGlobalPipelineStages < kKStages) {
        const std::size_t future_cell =
            first_sidecar_cell + stage + kGlobalPipelineStages;
        issue_pipeline_stage(
            &shared->pipeline[shared_slot],
            branch_sidecar + future_cell * kSidecarBytesPerStage,
            tile_activations, stage + kGlobalPipelineStages);
      }
    }

    store_branch_output(accumulators, branch_scale, first_token,
                        output_block * kOutputRowsPerBlock,
                        branch_output);
    __syncthreads();
  }
}

[[nodiscard]] bool pointer_is_aligned_16(const void* const pointer) noexcept {
  return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0U;
}

[[nodiscard]] bool valid_pointer_range(const void* const pointer,
                                       const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return begin != 0U &&
         begin <= std::numeric_limits<std::uintptr_t>::max() - bytes;
}

[[nodiscard]] bool pointer_ranges_overlap(
    const void* const lhs, const std::size_t lhs_bytes,
    const void* const rhs, const std::size_t rhs_bytes) noexcept {
  const std::uintptr_t lhs_begin = reinterpret_cast<std::uintptr_t>(lhs);
  const std::uintptr_t rhs_begin = reinterpret_cast<std::uintptr_t>(rhs);
  return lhs_begin < rhs_begin + rhs_bytes &&
         rhs_begin < lhs_begin + lhs_bytes;
}

[[nodiscard]] cudaError_t install_pair_dynamic_shared_attribute() noexcept {
  return cudaFuncSetAttribute(
      nvfp4_prefill_marlin_pair_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
}

}  // namespace

std::size_t sm87_nvfp4_prefill_marlin_sidecar_bytes_per_projection()
    noexcept {
  return kSidecarBytesPerProjection;
}

int launch_sm87_nvfp4_prefill_marlin_pack_cuda(
    const std::uint8_t* const canonical_packed_weights,
    const std::uint8_t* const canonical_block_scales,
    std::uint8_t* const sidecar, const std::size_t rows,
    const std::size_t columns, void* const cuda_stream) noexcept {
  if (canonical_packed_weights == nullptr ||
      canonical_block_scales == nullptr || sidecar == nullptr ||
      rows != kRows || columns != kColumns ||
      !pointer_is_aligned_16(canonical_packed_weights) ||
      !pointer_is_aligned_16(canonical_block_scales) ||
      !pointer_is_aligned_16(sidecar) ||
      !valid_pointer_range(canonical_packed_weights, kCanonicalPackedBytes) ||
      !valid_pointer_range(canonical_block_scales, kCanonicalScaleBytes) ||
      !valid_pointer_range(sidecar, kSidecarBytesPerProjection) ||
      pointer_ranges_overlap(canonical_packed_weights, kCanonicalPackedBytes,
                             canonical_block_scales,
                             kCanonicalScaleBytes) ||
      pointer_ranges_overlap(canonical_packed_weights, kCanonicalPackedBytes,
                             sidecar, kSidecarBytesPerProjection) ||
      pointer_ranges_overlap(canonical_block_scales, kCanonicalScaleBytes,
                             sidecar, kSidecarBytesPerProjection)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  nvfp4_prefill_marlin_pack_kernel<<<kOutputRowBlocks * kKStages,
                                      kThreads, 0U, stream>>>(
      canonical_packed_weights, canonical_block_scales, sidecar);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_prefill_marlin_pair_cuda(
    const std::uint8_t* const gate_sidecar,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_sidecar,
    const float up_weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const gate_output,
    std::uint16_t* const up_output, void* const cuda_stream) noexcept {
  if (gate_sidecar == nullptr || up_sidecar == nullptr ||
      activations == nullptr || gate_output == nullptr ||
      up_output == nullptr || token_count != kTokenCount || rows != kRows ||
      columns != kColumns || !std::isfinite(gate_weight_scale_2) ||
      !std::isfinite(up_weight_scale_2) || gate_weight_scale_2 < 0.0F ||
      up_weight_scale_2 < 0.0F || !pointer_is_aligned_16(gate_sidecar) ||
      !pointer_is_aligned_16(up_sidecar) ||
      !pointer_is_aligned_16(activations) ||
      !pointer_is_aligned_16(gate_output) ||
      !pointer_is_aligned_16(up_output) ||
      !valid_pointer_range(gate_sidecar, kSidecarBytesPerProjection) ||
      !valid_pointer_range(up_sidecar, kSidecarBytesPerProjection) ||
      !valid_pointer_range(activations, kActivationBytes) ||
      !valid_pointer_range(gate_output, kOutputBytes) ||
      !valid_pointer_range(up_output, kOutputBytes) ||
      pointer_ranges_overlap(gate_sidecar, kSidecarBytesPerProjection,
                             up_sidecar, kSidecarBytesPerProjection) ||
      pointer_ranges_overlap(gate_sidecar, kSidecarBytesPerProjection,
                             activations, kActivationBytes) ||
      pointer_ranges_overlap(up_sidecar, kSidecarBytesPerProjection,
                             activations, kActivationBytes) ||
      pointer_ranges_overlap(gate_sidecar, kSidecarBytesPerProjection,
                             gate_output, kOutputBytes) ||
      pointer_ranges_overlap(gate_sidecar, kSidecarBytesPerProjection,
                             up_output, kOutputBytes) ||
      pointer_ranges_overlap(up_sidecar, kSidecarBytesPerProjection,
                             gate_output, kOutputBytes) ||
      pointer_ranges_overlap(up_sidecar, kSidecarBytesPerProjection,
                             up_output, kOutputBytes) ||
      pointer_ranges_overlap(activations, kActivationBytes, gate_output,
                             kOutputBytes) ||
      pointer_ranges_overlap(activations, kActivationBytes, up_output,
                             kOutputBytes) ||
      pointer_ranges_overlap(gate_output, kOutputBytes, up_output,
                             kOutputBytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const cudaError_t attribute_status =
      install_pair_dynamic_shared_attribute();
  if (attribute_status != cudaSuccess) {
    return static_cast<int>(attribute_status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  nvfp4_prefill_marlin_pair_kernel
      <<<kPersistentBlocks, kThreads, kDynamicSharedBytes, stream>>>(
          gate_sidecar, gate_weight_scale_2, up_sidecar,
          up_weight_scale_2, activations, gate_output, up_output);
  return static_cast<int>(cudaGetLastError());
}

int query_sm87_nvfp4_prefill_marlin_pair_resources(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const dynamic_shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads_per_block,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      dynamic_shared_bytes == nullptr || local_bytes == nullptr ||
      maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = install_pair_dynamic_shared_attribute();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status =
      cudaFuncGetAttributes(&attributes, nvfp4_prefill_marlin_pair_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, nvfp4_prefill_marlin_pair_kernel,
      static_cast<int>(kThreads), kDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *dynamic_shared_bytes = kDynamicSharedBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
