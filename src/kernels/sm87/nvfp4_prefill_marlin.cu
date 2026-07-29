#include "q3x/kernels/sm87_nvfp4_prefill_marlin.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kWarpsPerBlock = 8U;
constexpr unsigned int kThreads = kWarpSize * kWarpsPerBlock;
constexpr unsigned int kRows = 17'408U;
constexpr unsigned int kColumns = 5'120U;
constexpr unsigned int kTokenCount = 512U;
constexpr unsigned int kResidentTokens = 64U;
constexpr unsigned int kOutputRowsPerBlock = 128U;
constexpr unsigned int kColumnsPerStage = 64U;
constexpr unsigned int kKStages = kColumns / kColumnsPerStage;
constexpr unsigned int kOutputRowBlocks =
    kRows / kOutputRowsPerBlock;
constexpr unsigned int kTokenTiles = kTokenCount / kResidentTokens;
constexpr unsigned int kPersistentPhases = 4U;
constexpr unsigned int kPersistentBlocks =
    kTokenTiles * kPersistentPhases;
constexpr unsigned int kPackedColumns = kColumns / 2U;
constexpr unsigned int kScaleColumns = kColumns / 16U;
constexpr unsigned int kSharedActivationLeadingDimension = 72U;

// One projection and one N128/K64 cell.  The first region is already in the
// exact lane order consumed by four branch warps.  The second region expands
// one E4M3FN scale per N row/K16 group into an exact BF16 value.
constexpr unsigned int kPackedWordsPerSidecarStage =
    4U * 4U * (kThreads / 2U);
constexpr unsigned int kScaleWordsPerSidecarStage =
    4U * kOutputRowsPerBlock;
constexpr unsigned int kSidecarWordsPerStage =
    kPackedWordsPerSidecarStage + kScaleWordsPerSidecarStage;
constexpr unsigned int kSidecarBytesPerStage =
    kSidecarWordsPerStage * sizeof(std::uint16_t);
constexpr std::size_t kSidecarBytesPerProjection =
    static_cast<std::size_t>(kOutputRowBlocks) * kKStages *
    kSidecarBytesPerStage;

static_assert(kKStages == 80U);
static_assert(kOutputRowBlocks == 136U);
static_assert(kTokenTiles == 8U);
static_assert(kPersistentBlocks == 32U);
static_assert(kPackedWordsPerSidecarStage == 2'048U);
static_assert(kScaleWordsPerSidecarStage == 512U);
static_assert(kSidecarBytesPerStage == 5'120U);
static_assert(kSidecarBytesPerProjection == 55'705'600U);

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

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

// Exact lower-half specialization of the retained production NVFP4 decoder.
// PRMT constructs the four E2M1 BF16 values; one packed BF16 multiply applies
// the predecoded E4M3FN scale without introducing an FP32 boundary.
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

__device__ __forceinline__ void cp_async_wait_group_0() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_1() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;" ::: "memory");
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
  uint4 gate_sidecar[320]; // Fragment-native N128/K64 = 5,120 B.
  uint4 up_sidecar[320];
};

struct alignas(32) MarlinSharedStorage {
  MarlinSharedStage pipeline[2];
  std::uint16_t branch_output[2][kResidentTokens * kOutputRowsPerBlock];
};

static_assert(sizeof(MarlinSharedStage) == 19'456U);
static_assert(sizeof(MarlinSharedStorage) == 71'680U);
static_assert(2U * sizeof(MarlinSharedStorage) < 160U * 1'024U);

struct RegisterSlot {
  std::uint32_t activation[4][4];
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
  if (local_thread >= kThreads / 2U) {
    return;
  }

  const unsigned int warp = local_thread / kWarpSize;
  const unsigned int lane = local_thread % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  auto* const stage_words = reinterpret_cast<std::uint16_t*>(
      sidecar + static_cast<std::size_t>(cell) * kSidecarBytesPerStage);

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
      stage_words[(k16 * 4U + n_panel) * (kThreads / 2U) +
                  local_thread] = packed;
    }
  }

  const unsigned int output_row =
      output_block * kOutputRowsPerBlock + local_thread;
  const auto* const canonical_scale_row =
      canonical_block_scales +
      static_cast<std::size_t>(output_row) * kScaleColumns +
      k_stage * 4U;
#pragma unroll
  for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
    stage_words[kPackedWordsPerSidecarStage +
                k16 * kOutputRowsPerBlock + local_thread] =
        encode_bf16_rne(decode_e4m3fn(canonical_scale_row[k16]));
  }
}

__device__ __forceinline__ void issue_pipeline_stage(
    MarlinSharedStage* const shared,
    const std::uint8_t* const gate_sidecar,
    const std::uint8_t* const up_sidecar,
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
  static_assert(kSidecarChunks == 320U);
  for (unsigned int index = threadIdx.x; index < kSidecarChunks;
       index += kThreads) {
    cp_async_cg_shared_global_16(
        shared->gate_sidecar + index,
        reinterpret_cast<const uint4*>(gate_sidecar) + index);
    cp_async_cg_shared_global_16(
        shared->up_sidecar + index,
        reinterpret_cast<const uint4*>(up_sidecar) + index);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void load_register_slot(
    RegisterSlot& slot, const MarlinSharedStage* const shared,
    const unsigned int k16, const bool up_branch) {
  namespace wmma = nvcuda::wmma;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < 4U; ++m_panel) {
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                   wmma::row_major>
        activation_fragment;
    const auto* const shared_a =
        reinterpret_cast<const __nv_bfloat16*>(shared->activations);
    wmma::load_matrix_sync(
        activation_fragment,
        shared_a + m_panel * 16U * kSharedActivationLeadingDimension +
            k16 * 16U,
        kSharedActivationLeadingDimension);
    slot.activation[m_panel][0] = pack_bf16_fragment_pair(
        activation_fragment.x[0], activation_fragment.x[1]);
    slot.activation[m_panel][1] = pack_bf16_fragment_pair(
        activation_fragment.x[2], activation_fragment.x[3]);
    slot.activation[m_panel][2] = pack_bf16_fragment_pair(
        activation_fragment.x[4], activation_fragment.x[5]);
    slot.activation[m_panel][3] = pack_bf16_fragment_pair(
        activation_fragment.x[6], activation_fragment.x[7]);
  }

  const auto* const sidecar = reinterpret_cast<const std::uint16_t*>(
      up_branch ? shared->up_sidecar : shared->gate_sidecar);
  const unsigned int branch_thread = threadIdx.x % (kThreads / 2U);
  const unsigned int branch_warp = branch_thread / kWarpSize;
  const unsigned int lane = branch_thread % kWarpSize;
  const unsigned int lane_group = lane / 4U;
#pragma unroll
  for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
    const unsigned int local_row =
        branch_warp * 32U + n_panel * 8U + lane_group;
    const std::uint16_t packed =
        sidecar[(k16 * 4U + n_panel) * (kThreads / 2U) +
                branch_thread];
    const std::uint16_t scale =
        sidecar[kPackedWordsPerSidecarStage +
                k16 * kOutputRowsPerBlock + local_row];
    slot.weights[n_panel] = decode_nvfp4x4(packed, scale);
  }
}

__device__ __forceinline__ void store_branch_output(
    const InlineM16N8Accumulator (&accumulators)[4][4],
    const float weight_scale_2, std::uint16_t* const output) {
  const unsigned int branch_thread = threadIdx.x % (kThreads / 2U);
  const unsigned int branch_warp = branch_thread / kWarpSize;
  const unsigned int lane = branch_thread % kWarpSize;
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
          branch_warp * 32U + n_pair * 16U + 2U * (lane & 7U);

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
          output + low_even_token * kOutputRowsPerBlock + local_column) =
          low_even;
      *reinterpret_cast<std::uint32_t*>(
          output + (low_even_token + 1U) * kOutputRowsPerBlock +
          local_column) = low_odd;

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
          output + high_even_token * kOutputRowsPerBlock + local_column) =
          high_even;
      *reinterpret_cast<std::uint32_t*>(
          output + (high_even_token + 1U) * kOutputRowsPerBlock +
          local_column) = high_odd;
    }
  }
}

__global__ __launch_bounds__(kThreads, 2) void
nvfp4_prefill_marlin_gate_up_silu_kernel(
    const std::uint8_t* const gate_sidecar,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_sidecar,
    const float up_weight_scale_2,
    const std::uint16_t* const activations,
    std::uint16_t* const output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const shared =
      reinterpret_cast<MarlinSharedStorage*>(dynamic_storage);
  const unsigned int token_tile = blockIdx.x % kTokenTiles;
  const unsigned int output_phase = blockIdx.x / kTokenTiles;
  const std::size_t first_token =
      static_cast<std::size_t>(token_tile) * kResidentTokens;
  const auto* const tile_activations =
      activations + first_token * kColumns;
  const bool up_branch = threadIdx.x >= kThreads / 2U;
  const unsigned int branch = up_branch ? 1U : 0U;
  const float branch_scale =
      up_branch ? up_weight_scale_2 : gate_weight_scale_2;

  for (unsigned int output_block = output_phase;
       output_block < kOutputRowBlocks;
       output_block += kPersistentPhases) {
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
    issue_pipeline_stage(
        &shared->pipeline[0],
        gate_sidecar + first_sidecar_cell * kSidecarBytesPerStage,
        up_sidecar + first_sidecar_cell * kSidecarBytesPerStage,
        tile_activations, 0U);
    issue_pipeline_stage(
        &shared->pipeline[1],
        gate_sidecar +
            (first_sidecar_cell + 1U) * kSidecarBytesPerStage,
        up_sidecar +
            (first_sidecar_cell + 1U) * kSidecarBytesPerStage,
        tile_activations, 1U);

#pragma unroll 1
    for (unsigned int stage = 0U; stage < kKStages; ++stage) {
      if (stage + 1U < kKStages) {
        cp_async_wait_group_1();
      } else {
        cp_async_wait_group_0();
      }
      __syncthreads();

      const unsigned int shared_slot = stage % 2U;
      RegisterSlot registers[2];
      load_register_slot(registers[0], &shared->pipeline[shared_slot], 0U,
                         up_branch);
#pragma unroll
      for (unsigned int k16 = 0U; k16 < 4U; ++k16) {
        const unsigned int register_slot = k16 % 2U;
        if (k16 + 1U < 4U) {
          load_register_slot(registers[(k16 + 1U) % 2U],
                             &shared->pipeline[shared_slot], k16 + 1U,
                             up_branch);
        }
#pragma unroll
        for (unsigned int m_panel = 0U; m_panel < 4U; ++m_panel) {
#pragma unroll
          for (unsigned int n_panel = 0U; n_panel < 4U; ++n_panel) {
            mma_m16n8k16_bf16(
                accumulators[m_panel][n_panel],
                registers[register_slot].activation[m_panel][0],
                registers[register_slot].activation[m_panel][1],
                registers[register_slot].activation[m_panel][2],
                registers[register_slot].activation[m_panel][3],
                registers[register_slot].weights[n_panel].x,
                registers[register_slot].weights[n_panel].y);
          }
        }
      }
      __syncthreads();

      if (stage + 2U < kKStages) {
        const std::size_t future_cell =
            first_sidecar_cell + stage + 2U;
        issue_pipeline_stage(
            &shared->pipeline[shared_slot],
            gate_sidecar + future_cell * kSidecarBytesPerStage,
            up_sidecar + future_cell * kSidecarBytesPerStage,
            tile_activations, stage + 2U);
      }
    }

    store_branch_output(accumulators, branch_scale,
                        shared->branch_output[branch]);
    __syncthreads();

    const unsigned int first_output_row =
        output_block * kOutputRowsPerBlock;
    constexpr unsigned int kBranchElements =
        kResidentTokens * kOutputRowsPerBlock;
    for (unsigned int index = threadIdx.x; index < kBranchElements;
         index += kThreads) {
      const float gate = decode_bf16(shared->branch_output[0][index]);
      const float up = decode_bf16(shared->branch_output[1][index]);
      const float silu = gate / (1.0F + expf(-gate));
      const unsigned int token = index / kOutputRowsPerBlock;
      const unsigned int local_row = index % kOutputRowsPerBlock;
      output[(first_token + token) * kRows + first_output_row + local_row] =
          encode_bf16_rne(silu * up);
    }
    __syncthreads();
  }
}

[[nodiscard]] bool pointer_is_aligned_16(const void* const pointer) noexcept {
  return (reinterpret_cast<std::uintptr_t>(pointer) & 15U) == 0U;
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
      !pointer_is_aligned_16(sidecar)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  nvfp4_prefill_marlin_pack_kernel<<<kOutputRowBlocks * kKStages,
                                      kThreads / 2U, 0U, stream>>>(
      canonical_packed_weights, canonical_block_scales, sidecar);
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_nvfp4_prefill_marlin_gate_up_silu_cuda(
    const std::uint8_t* const gate_sidecar,
    const float gate_weight_scale_2,
    const std::uint8_t* const up_sidecar,
    const float up_weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (gate_sidecar == nullptr || up_sidecar == nullptr ||
      activations == nullptr || output == nullptr ||
      token_count != kTokenCount || rows != kRows || columns != kColumns ||
      !std::isfinite(gate_weight_scale_2) ||
      !std::isfinite(up_weight_scale_2) || gate_weight_scale_2 < 0.0F ||
      up_weight_scale_2 < 0.0F || !pointer_is_aligned_16(gate_sidecar) ||
      !pointer_is_aligned_16(up_sidecar) ||
      !pointer_is_aligned_16(activations) ||
      !pointer_is_aligned_16(output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const cudaError_t attribute_status = cudaFuncSetAttribute(
      nvfp4_prefill_marlin_gate_up_silu_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(sizeof(MarlinSharedStorage)));
  if (attribute_status != cudaSuccess) {
    return static_cast<int>(attribute_status);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  nvfp4_prefill_marlin_gate_up_silu_kernel
      <<<kPersistentBlocks, kThreads, sizeof(MarlinSharedStorage), stream>>>(
          gate_sidecar, gate_weight_scale_2, up_sidecar,
          up_weight_scale_2, activations, output);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::kernels
