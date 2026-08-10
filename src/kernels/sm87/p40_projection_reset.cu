#include "q3x/kernels/sm87_p40_projection_reset.h"

#include "third_party/vllm_marlin/marlin_template.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

#if defined(Q3X_ENABLE_P40_PROJECTION_RESET_ADMISSION)
inline constexpr bool kProjectionResetAdmitted = true;
#else
inline constexpr bool kProjectionResetAdmitted = false;
#endif

inline constexpr unsigned int kWarpSize = 32U;
inline constexpr unsigned int kM16Panels = 4U;
inline constexpr unsigned int kN8Panels = 4U;
inline constexpr unsigned int kK16PerStage = 2U;
inline constexpr unsigned int kAChunksPerRow = 4U;

using Bf16MarlinType = marlin::MarlinScalarType<vllm::kBFloat16.id()>;
using FragA = typename Bf16MarlinType::FragA;
using FragB = typename Bf16MarlinType::FragB;
using FragC = typename Bf16MarlinType::FragC;
using FragS = typename Bf16MarlinType::FragS;

inline constexpr unsigned int kFp8M16PanelsPerWarp = 4U;
inline constexpr unsigned int kFp8N8PanelsPerWarp = 8U;
inline constexpr unsigned int kFp8K16PerStage = 4U;
inline constexpr unsigned int kFp8ActivationChunksPerRow = 8U;
inline constexpr unsigned int kFp8VectorsPerOperandStage = 1'024U;

// CUDA only guarantees the declared alignment of the dynamic shared symbol.
// Every vector access below is 16-byte wide, so matching the symbol's 32-byte
// alignment is both sufficient and avoids an invalid over-aligned cast.
struct alignas(32) ProjectionResetStorage {
  uint4 activations[kSm87P40ProjectionResetPipelineStages]
                   [kSm87P40ProjectionResetTileM * kAChunksPerRow];
  int4 weights[kSm87P40ProjectionResetPipelineStages]
              [kK16PerStage * 2U * kWarpSize];
  int4 scales[kSm87P40ProjectionResetPipelineStages]
             [kK16PerStage * 2U * 4U];
  std::uint8_t alignment_padding
      [kSm87P40ProjectionResetAlignedMainloopSharedBytes -
       kSm87P40ProjectionResetMainloopSharedBytes];
};

static_assert(sizeof(ProjectionResetStorage) ==
              kSm87P40ProjectionResetAlignedMainloopSharedBytes);

struct RegisterStage {
  FragA activations[kM16Panels];
  int2 packed_weights{};
  int encoded_scales = 0;
  FragB decoded_weights[kN8Panels];
};

struct alignas(32) Fp8ProjectionResetStorage {
  uint4 activations[kSm87P40ProjectionResetFp8PipelineStages]
                   [kFp8VectorsPerOperandStage];
  uint4 weights[kSm87P40ProjectionResetFp8PipelineStages]
               [kFp8VectorsPerOperandStage];
};

static_assert(sizeof(Fp8ProjectionResetStorage) ==
              kSm87P40ProjectionResetFp8DynamicSharedBytes);

struct Fp8Accumulator {
  float x0;
  float x1;
  float x2;
  float x3;
};

struct Fp8ActivationFragment {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct Fp8RegisterStage {
  Fp8ActivationFragment activations[kFp8M16PanelsPerWarp];
  std::uint32_t packed_weights[kFp8N8PanelsPerWarp];
};

struct Fp8DevicePartition {
  const uint4* sidecar = nullptr;
  std::uint16_t* output = nullptr;
  float weight_scale = 0.0F;
  unsigned int rows = 0U;
  unsigned int nblocks = 0U;
};

struct ByteRange {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] __host__ __device__ constexpr unsigned int transform_a_cell(
    const unsigned int logical_cell) noexcept {
  const unsigned int row = logical_cell / kAChunksPerRow;
  const unsigned int chunk = logical_cell % kAChunksPerRow;
  const unsigned int paired_row = row / 2U;
  const unsigned int paired_column = (row % 2U) * kAChunksPerRow + chunk;
  // Humming's 64-byte A swizzle packs two logical rows into one physical
  // eight-vector row before applying the four-column XOR.  This gives the
  // K32 ldmatrix lanes distinct banks; the old row-local XOR produced a
  // deterministic two-way conflict on SM87.
  return paired_row * (2U * kAChunksPerRow) +
         (paired_column ^ (paired_row % kAChunksPerRow));
}

template <bool kPredicate>
__device__ __forceinline__ void cp_async_cg_zfill_16(
    void* const shared_destination, const void* const global_source,
    const bool valid = true) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
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

[[nodiscard]] __device__ __forceinline__ nv_bfloat162 broadcast_bf16_bits(
    const std::uint16_t bits) {
  const std::uint32_t pair = static_cast<std::uint32_t>(bits) |
                             (static_cast<std::uint32_t>(bits) << 16U);
  return *reinterpret_cast<const nv_bfloat162*>(&pair);
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

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) {
  return __uint_as_float(static_cast<std::uint32_t>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_scaled_bf16_pair(
    const float low, const float high, const float global_scale) {
  return static_cast<std::uint32_t>(encode_bf16_rne(low * global_scale)) |
         (static_cast<std::uint32_t>(
              encode_bf16_rne(high * global_scale))
          << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t
gate_up_silu_mul_bf16(const std::uint16_t gate_bits,
                      const std::uint16_t up_bits) {
  const float gate = decode_bf16(gate_bits);
  const float up = decode_bf16(up_bits);
  return encode_bf16_rne(gate / (1.0F + expf(-gate)) * up);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
add_residual_bf16_pair(const std::uint32_t branch_bits,
                       const std::uint32_t residual_bits) {
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

[[nodiscard]] __device__ __forceinline__ uint2
decode_fp8x4_to_bf16x4(const std::uint32_t packed) {
  // The authenticated sidecar stores [v0,v2,v1,v3].  Restore the exact
  // m16n8 register order [v0,v1] / [v2,v3] without a shared-memory LUT.
  const std::uint16_t value0 = decode_e4m3fn_to_bf16_bits(
      static_cast<std::uint8_t>(packed));
  const std::uint16_t value2 = decode_e4m3fn_to_bf16_bits(
      static_cast<std::uint8_t>(packed >> 8U));
  const std::uint16_t value1 = decode_e4m3fn_to_bf16_bits(
      static_cast<std::uint8_t>(packed >> 16U));
  const std::uint16_t value3 = decode_e4m3fn_to_bf16_bits(
      static_cast<std::uint8_t>(packed >> 24U));
  return {static_cast<std::uint32_t>(value0) |
              (static_cast<std::uint32_t>(value1) << 16U),
          static_cast<std::uint32_t>(value2) |
              (static_cast<std::uint32_t>(value3) << 16U)};
}

__device__ __forceinline__ void fp8_mma_m16n8k16_bf16(
    Fp8Accumulator* const accumulator,
    const Fp8ActivationFragment& activation, const uint2 weights) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator->x0), "+f"(accumulator->x1),
        "+f"(accumulator->x2), "+f"(accumulator->x3)
      : "r"(activation.x0), "r"(activation.x1), "r"(activation.x2),
        "r"(activation.x3), "r"(weights.x), "r"(weights.y));
#endif
}

__device__ __forceinline__ void load_fp8_activation_fragment(
    Fp8ActivationFragment* const fragment,
    const std::uint16_t* const shared_activations,
    const unsigned int warp_m, const unsigned int m_panel,
    const unsigned int k16, const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row_in_panel =
      (lane % 8U) + (quadrant & 1U) * 8U;
  const unsigned int logical_row =
      warp_m * 64U + m_panel * 16U + row_in_panel;
  const unsigned int logical_chunk =
      k16 * 2U + (quadrant >> 1U);
  const unsigned int physical_chunk =
      logical_chunk ^ (logical_row & 7U);
  const std::uint16_t* const source =
      shared_activations +
      logical_row * kSm87P40ProjectionResetFp8TileK +
      physical_chunk * 8U;
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment->x0), "=r"(fragment->x1),
        "=r"(fragment->x2), "=r"(fragment->x3)
      : "r"(shared_address)
      : "memory");
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t
load_shared_u32(const std::uint32_t* const source) {
  std::uint32_t value = 0U;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile("ld.shared.u32 %0, [%1];"
               : "=r"(value)
               : "r"(shared_address)
               : "memory");
#endif
  return value;
}

__device__ __forceinline__ void load_fp8_register_stage(
    Fp8RegisterStage* const registers,
    const Fp8ProjectionResetStorage* const storage,
    const unsigned int shared_slot, const unsigned int k16,
    const unsigned int warp_m, const unsigned int warp_n,
    const unsigned int lane) {
  const auto* const shared_a =
      reinterpret_cast<const std::uint16_t*>(
          storage->activations[shared_slot]);
  const auto* const shared_b =
      reinterpret_cast<const std::uint32_t*>(
          storage->weights[shared_slot]);
#pragma unroll
  for (unsigned int m_panel = 0U;
       m_panel < kFp8M16PanelsPerWarp; ++m_panel) {
    load_fp8_activation_fragment(&registers->activations[m_panel], shared_a,
                                 warp_m, m_panel, k16, lane);
  }
#pragma unroll
  for (unsigned int n_panel = 0U;
       n_panel < kFp8N8PanelsPerWarp; ++n_panel) {
    const unsigned int old_half = n_panel / 4U;
    const unsigned int old_n_panel = n_panel % 4U;
    const unsigned int old_thread =
        (2U * warp_n + old_half) * kWarpSize + lane;
    registers->packed_weights[n_panel] = load_shared_u32(
        shared_b +
        (k16 * 4U + old_n_panel) *
            kSm87P40ProjectionResetFp8Threads +
        old_thread);
  }
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures>
__device__ __forceinline__ void issue_pipeline_stage(
    ProjectionResetStorage* const storage, const unsigned int shared_slot,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const unsigned int m_tile, const unsigned int n_tile,
    const unsigned int first_k) {
  static_assert(kInputFeatures % kSm87P40ProjectionResetTileK == 0U);
  static_assert(kOutputFeatures % kSm87P40ProjectionResetTileN == 0U);
  const std::size_t first_token =
      static_cast<std::size_t>(m_tile) * kSm87P40ProjectionResetTileM;

#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int logical_cell =
        threadIdx.x + pass * kSm87P40ProjectionResetThreads;
    const unsigned int row = logical_cell / kAChunksPerRow;
    const unsigned int chunk = logical_cell % kAChunksPerRow;
    const auto* const source = reinterpret_cast<const uint4*>(
        input + (first_token + row) * kInputFeatures + first_k);
    cp_async_cg_zfill_16<false>(
        &storage->activations[shared_slot]
                             [transform_a_cell(logical_cell)],
        source + chunk);
  }

  const unsigned int k16_plane = threadIdx.x / 64U;
  const unsigned int plane_offset = threadIdx.x % 64U;
  const auto* const packed_weight =
      reinterpret_cast<const int4*>(marlin_weight);
  const std::size_t weight_index =
      static_cast<std::size_t>(first_k / 16U + k16_plane) *
          (kOutputFeatures / 2U) +
      static_cast<std::size_t>(n_tile) * 64U + plane_offset;
  cp_async_cg_zfill_16<false>(
      &storage->weights[shared_slot][threadIdx.x],
      &packed_weight[weight_index]);

  if (threadIdx.x < 16U) {
    const unsigned int scale_k16 = threadIdx.x / 8U;
    const unsigned int scale_offset = threadIdx.x % 8U;
    const auto* const encoded_scales =
        reinterpret_cast<const int4*>(marlin_scales);
    const std::size_t scale_index =
        static_cast<std::size_t>(first_k / 16U + scale_k16) *
            (kOutputFeatures / 16U) +
        static_cast<std::size_t>(n_tile) * 8U + scale_offset;
    cp_async_cg_zfill_16<false>(
        &storage->scales[shared_slot][threadIdx.x],
        &encoded_scales[scale_index]);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void load_register_stage(
    const ProjectionResetStorage* const storage,
    const unsigned int shared_slot, const unsigned int k16_plane,
    const unsigned int warp, const unsigned int lane,
    RegisterStage* const registers) {
  const unsigned int n64 = warp / 2U;
  const unsigned int n32 = warp % 2U;
  const auto* const quantized_halves =
      reinterpret_cast<const int2*>(storage->weights[shared_slot]);
  const std::size_t quantized_int4_index =
      static_cast<std::size_t>(k16_plane) * 64U +
      static_cast<std::size_t>(n64) * 32U + lane;
  registers->packed_weights =
      quantized_halves[2U * quantized_int4_index + n32];

  const unsigned int lane_group = lane / 4U;
  const auto* const encoded_scales =
      reinterpret_cast<const int*>(storage->scales[shared_slot]);
  registers->encoded_scales = encoded_scales
      [k16_plane * 32U + n64 * 16U + lane_group * 2U + n32];

  unsigned int transformed = transform_a_cell(
      2U * k16_plane + kAChunksPerRow * (lane % 16U) + lane / 16U);
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
    marlin::ldsm<4, vllm::kBFloat16.id()>(
        registers->activations[m_panel],
        &storage->activations[shared_slot][transformed]);
    transformed += 16U * kAChunksPerRow;
  }
}

__device__ __forceinline__ void decode_register_stage(
    RegisterStage* const registers) {
  FragS scale_fragments[2];
  marlin::dequant_fp8_scales<nv_bfloat162, vllm::kFE4M3fn.id()>(
      registers->encoded_scales,
      reinterpret_cast<nv_bfloat162*>(&scale_fragments[0]));
  const int quantized[2] = {registers->packed_weights.x,
                            registers->packed_weights.y};
#pragma unroll
  for (unsigned int pair = 0U; pair < 2U; ++pair) {
    FragB& first = registers->decoded_weights[2U * pair];
    FragB& second = registers->decoded_weights[2U * pair + 1U];
    marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
        quantized[pair] << 8, reinterpret_cast<nv_bfloat162*>(&first));
    marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
        quantized[pair], reinterpret_cast<nv_bfloat162*>(&second));
    const auto* const scale_values =
        reinterpret_cast<const std::uint16_t*>(&scale_fragments[pair]);
    const nv_bfloat162 first_scale = broadcast_bf16_bits(scale_values[0]);
    const nv_bfloat162 second_scale = broadcast_bf16_bits(scale_values[1]);
    first[0] = __hmul2(first[0], first_scale);
    first[1] = __hmul2(first[1], first_scale);
    second[0] = __hmul2(second[0], second_scale);
    second[1] = __hmul2(second[1], second_scale);
  }
}

__device__ __forceinline__ void consume_register_stage(
    const RegisterStage& registers,
    FragC (&accumulators)[kM16Panels][kN8Panels]) {
#pragma unroll
  for (unsigned int n_panel = 0U; n_panel < kN8Panels; ++n_panel) {
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
      marlin::mma<vllm::kBFloat16.id(), false>(
          registers.activations[m_panel],
          registers.decoded_weights[n_panel],
          accumulators[m_panel][n_panel]);
    }
  }
}

__device__ __forceinline__ void stage_accumulators(
    const unsigned int warp, const unsigned int lane,
    const float global_scale,
    const FragC (&accumulators)[kM16Panels][kN8Panels],
    std::uint16_t* const staged) {
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8Panels; ++n_panel) {
      const FragC& accumulator = accumulators[m_panel][n_panel];
      const unsigned int local_column =
          warp * 32U + n_panel * 8U + 2U * lane_in_group;
      const unsigned int low_token =
          m_panel * 16U + lane_group;
      *reinterpret_cast<std::uint32_t*>(
          staged + low_token * kSm87P40ProjectionResetTileN +
          local_column) = pack_scaled_bf16_pair(
              accumulator.elems[0], accumulator.elems[1], global_scale);

      const unsigned int high_token = low_token + 8U;
      *reinterpret_cast<std::uint32_t*>(
          staged + high_token * kSm87P40ProjectionResetTileN +
          local_column) = pack_scaled_bf16_pair(
              accumulator.elems[2], accumulator.elems[3], global_scale);
    }
  }
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          bool kGateUp>
__device__ __forceinline__ void execute_task(
    ProjectionResetStorage* const storage,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float global_scale, const unsigned int m_tile,
    const unsigned int n_tile, std::uint16_t* const output) {
  constexpr unsigned int kK32Stages =
      kInputFeatures / kSm87P40ProjectionResetTileK;
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;

  FragC accumulators[kM16Panels][kN8Panels];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8Panels; ++n_panel) {
#pragma unroll
      for (unsigned int value = 0U; value < 4U; ++value) {
        accumulators[m_panel][n_panel].elems[value] = 0.0F;
      }
    }
  }

  issue_pipeline_stage<kInputFeatures, kOutputFeatures>(
      storage, 0U, input, marlin_weight, marlin_scales, m_tile, n_tile, 0U);
  issue_pipeline_stage<kInputFeatures, kOutputFeatures>(
      storage, 1U, input, marlin_weight, marlin_scales, m_tile, n_tile,
      kSm87P40ProjectionResetTileK);

  RegisterStage registers[2];
#pragma unroll 1
  for (unsigned int stage = 0U; stage < kK32Stages; ++stage) {
    if (stage + 1U < kK32Stages) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();

    const unsigned int shared_slot =
        stage % kSm87P40ProjectionResetPipelineStages;
    load_register_stage(storage, shared_slot, 0U, warp, lane,
                        &registers[0]);
    decode_register_stage(&registers[0]);
    load_register_stage(storage, shared_slot, 1U, warp, lane,
                        &registers[1]);

    if (stage + 2U < kK32Stages) {
      const unsigned int future_stage = stage + 2U;
      issue_pipeline_stage<kInputFeatures, kOutputFeatures>(
          storage,
          future_stage % kSm87P40ProjectionResetPipelineStages,
          input, marlin_weight, marlin_scales, m_tile, n_tile,
          future_stage * kSm87P40ProjectionResetTileK);
    }

    consume_register_stage(registers[0], accumulators);
    decode_register_stage(&registers[1]);
    consume_register_stage(registers[1], accumulators);
  }

  // All warps must leave the last shared stage before the Gate epilogue
  // reuses the allocation, or before the next persistent task overwrites it.
  __syncthreads();
  const std::size_t first_token =
      static_cast<std::size_t>(m_tile) * kSm87P40ProjectionResetTileM;
  if constexpr (kGateUp) {
    auto* const staged = reinterpret_cast<std::uint16_t*>(storage);
    stage_accumulators(warp, lane, global_scale, accumulators, staged);
    __syncthreads();
    constexpr unsigned int kActivatedColumnsPerTile =
        kSm87P40ProjectionResetTileN / 2U;
    constexpr unsigned int kPairs =
        kSm87P40ProjectionResetTileM * kActivatedColumnsPerTile /
        kSm87P40ProjectionResetThreads;
#pragma unroll
    for (unsigned int pass = 0U; pass < kPairs; ++pass) {
      const unsigned int logical =
          threadIdx.x + pass * kSm87P40ProjectionResetThreads;
      const unsigned int row = logical / kActivatedColumnsPerTile;
      const unsigned int column = logical % kActivatedColumnsPerTile;
      const std::uint16_t gate = staged
          [row * kSm87P40ProjectionResetTileN + 2U * column];
      const std::uint16_t up = staged
          [row * kSm87P40ProjectionResetTileN + 2U * column + 1U];
      output[(first_token + row) *
                 kSm87P40ProjectionResetIntermediate +
             n_tile * kActivatedColumnsPerTile + column] =
          gate_up_silu_mul_bf16(gate, up);
    }
    __syncthreads();
  } else {
    auto* const staged = reinterpret_cast<std::uint16_t*>(storage);
    stage_accumulators(warp, lane, global_scale, accumulators, staged);
    __syncthreads();
    constexpr unsigned int kPairs =
        kSm87P40ProjectionResetTileM *
        (kSm87P40ProjectionResetTileN / 2U) /
        kSm87P40ProjectionResetThreads;
#pragma unroll
    for (unsigned int pass = 0U; pass < kPairs; ++pass) {
      const unsigned int pair =
          threadIdx.x + pass * kSm87P40ProjectionResetThreads;
      const unsigned int row =
          pair / (kSm87P40ProjectionResetTileN / 2U);
      const unsigned int pair_column =
          pair % (kSm87P40ProjectionResetTileN / 2U);
      const unsigned int column = 2U * pair_column;
      const std::size_t output_index =
          (first_token + row) * kSm87P40ProjectionResetHidden +
          n_tile * kSm87P40ProjectionResetTileN + column;
      const std::uint32_t branch =
          *reinterpret_cast<const std::uint32_t*>(
              staged + row * kSm87P40ProjectionResetTileN + column);
      const std::uint32_t residual =
          *reinterpret_cast<const std::uint32_t*>(output + output_index);
      *reinterpret_cast<std::uint32_t*>(output + output_index) =
          add_residual_bf16_pair(branch, residual);
    }
    __syncthreads();
  }
}

template <unsigned int kColumns>
__device__ __forceinline__ void issue_fp8_pipeline_stage(
    Fp8ProjectionResetStorage* const storage,
    const unsigned int shared_slot, const uint4* const sidecar_stage,
    const std::uint16_t* const activations, const unsigned int m_tile,
    const unsigned int first_k) {
  static_assert(kColumns == 5'120U || kColumns == 6'144U);
  const unsigned int first_token =
      m_tile * kSm87P40ProjectionResetFp8TileM;
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index =
        threadIdx.x + pass * kSm87P40ProjectionResetFp8Threads;
    const unsigned int row = index / kFp8ActivationChunksPerRow;
    const unsigned int chunk = index % kFp8ActivationChunksPerRow;
    const bool valid = first_token + row <
                       kSm87P40ProjectionResetTokens;
    const unsigned int source_token = valid ? first_token + row : first_token;
    const auto* const source = reinterpret_cast<const uint4*>(
        activations + static_cast<std::size_t>(source_token) * kColumns +
        first_k);
    const unsigned int physical_chunk = chunk ^ (row & 7U);
    cp_async_cg_zfill_16<true>(
        storage->activations[shared_slot] +
            row * kFp8ActivationChunksPerRow + physical_chunk,
        source + chunk, valid);
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index =
        threadIdx.x + pass * kSm87P40ProjectionResetFp8Threads;
    cp_async_cg_zfill_16<false>(storage->weights[shared_slot] + index,
                                sidecar_stage + index);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void consume_fp8_k64_stage(
    const Fp8ProjectionResetStorage* const storage,
    const unsigned int shared_slot, const unsigned int warp_m,
    const unsigned int warp_n, const unsigned int lane,
    Fp8Accumulator (&accumulators)[kFp8M16PanelsPerWarp]
                                  [kFp8N8PanelsPerWarp]) {
  Fp8RegisterStage register_stages[2];
  load_fp8_register_stage(&register_stages[0], storage, shared_slot, 0U,
                          warp_m, warp_n, lane);
#pragma unroll
  for (unsigned int k16 = 0U; k16 < kFp8K16PerStage; ++k16) {
    const unsigned int current_slot = k16 & 1U;
    if (k16 + 1U < kFp8K16PerStage) {
      load_fp8_register_stage(&register_stages[current_slot ^ 1U], storage,
                              shared_slot, k16 + 1U, warp_m, warp_n, lane);
    }
#pragma unroll
    for (unsigned int n_panel = 0U;
         n_panel < kFp8N8PanelsPerWarp; ++n_panel) {
      const uint2 decoded = decode_fp8x4_to_bf16x4(
          register_stages[current_slot].packed_weights[n_panel]);
#pragma unroll
      for (unsigned int m_panel = 0U;
           m_panel < kFp8M16PanelsPerWarp; ++m_panel) {
        fp8_mma_m16n8k16_bf16(
            &accumulators[m_panel][n_panel],
            register_stages[current_slot].activations[m_panel], decoded);
      }
    }
  }
}

template <unsigned int kColumns>
__device__ __forceinline__ void execute_fp8_task(
    Fp8ProjectionResetStorage* const storage,
    const Fp8DevicePartition& partition,
    const std::uint16_t* const activations,
    const unsigned int m_tile, const unsigned int local_nblock) {
  constexpr unsigned int kKStages =
      kColumns / kSm87P40ProjectionResetFp8TileK;
  static_assert(kKStages == 80U || kKStages == 96U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp / 4U;
  const unsigned int warp_n = warp % 4U;
  const uint4* const sidecar_tile =
      partition.sidecar +
      static_cast<std::size_t>(local_nblock) * kKStages *
          kFp8VectorsPerOperandStage;

  Fp8Accumulator accumulators[kFp8M16PanelsPerWarp]
                             [kFp8N8PanelsPerWarp];
#pragma unroll
  for (unsigned int m_panel = 0U;
       m_panel < kFp8M16PanelsPerWarp; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U;
         n_panel < kFp8N8PanelsPerWarp; ++n_panel) {
      accumulators[m_panel][n_panel] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  issue_fp8_pipeline_stage<kColumns>(storage, 0U, sidecar_tile,
                                     activations, m_tile, 0U);
  issue_fp8_pipeline_stage<kColumns>(
      storage, 1U, sidecar_tile + kFp8VectorsPerOperandStage,
      activations, m_tile, kSm87P40ProjectionResetFp8TileK);
  issue_fp8_pipeline_stage<kColumns>(
      storage, 2U, sidecar_tile + 2U * kFp8VectorsPerOperandStage,
      activations, m_tile, 2U * kSm87P40ProjectionResetFp8TileK);

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
    const unsigned int shared_slot =
        stage % kSm87P40ProjectionResetFp8PipelineStages;
    consume_fp8_k64_stage(storage, shared_slot, warp_m, warp_n, lane,
                          accumulators);
    // Every warp must finish S2R before the stage is recycled by cp.async.
    __syncthreads();
    if (stage + kSm87P40ProjectionResetFp8PipelineStages < kKStages) {
      const unsigned int future_stage =
          stage + kSm87P40ProjectionResetFp8PipelineStages;
      issue_fp8_pipeline_stage<kColumns>(
          storage, shared_slot,
          sidecar_tile +
              static_cast<std::size_t>(future_stage) *
                  kFp8VectorsPerOperandStage,
          activations, m_tile,
          future_stage * kSm87P40ProjectionResetFp8TileK);
    }
  }
  cp_async_wait_group<0U>();
  __syncthreads();

  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  const unsigned int first_token =
      m_tile * kSm87P40ProjectionResetFp8TileM;
  const unsigned int first_output_column =
      local_nblock * kSm87P40ProjectionResetFp8TileN;
#pragma unroll
  for (unsigned int m_panel = 0U;
       m_panel < kFp8M16PanelsPerWarp; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U;
         n_panel < kFp8N8PanelsPerWarp; ++n_panel) {
      const unsigned int token0 =
          warp_m * 64U + m_panel * 16U + lane_group;
      const unsigned int token1 = token0 + 8U;
      const unsigned int local_column =
          warp_n * 64U + n_panel * 8U + 2U * lane_in_group;
      if (first_token + token0 < kSm87P40ProjectionResetTokens) {
        *reinterpret_cast<std::uint32_t*>(
            partition.output +
            static_cast<std::size_t>(first_token + token0) *
                partition.rows +
            first_output_column + local_column) =
            pack_scaled_bf16_pair(accumulators[m_panel][n_panel].x0,
                                  accumulators[m_panel][n_panel].x1,
                                  partition.weight_scale);
      }
      if (first_token + token1 < kSm87P40ProjectionResetTokens) {
        *reinterpret_cast<std::uint32_t*>(
            partition.output +
            static_cast<std::size_t>(first_token + token1) *
                partition.rows +
            first_output_column + local_column) =
            pack_scaled_bf16_pair(accumulators[m_panel][n_panel].x2,
                                  accumulators[m_panel][n_panel].x3,
                                  partition.weight_scale);
      }
    }
  }
  __syncthreads();
}

template <unsigned int kColumns, unsigned int kPartitionCount,
          unsigned int kGroupM>
__global__ __launch_bounds__(kSm87P40ProjectionResetFp8Threads, 1) void
sm87_p40_projection_reset_fp8_supermatrix_kernel(
    const Fp8DevicePartition partition0,
    const Fp8DevicePartition partition1,
    const Fp8DevicePartition partition2,
    const std::uint16_t* __restrict__ activations,
    const unsigned int total_nblocks) {
  static_assert(kColumns == 5'120U || kColumns == 6'144U);
  static_assert(kPartitionCount >= 1U && kPartitionCount <= 3U);
  static_assert(kGroupM == 1U || kGroupM == 2U);
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<Fp8ProjectionResetStorage*>(dynamic_storage);
  constexpr unsigned int kGridM = 313U;
  const unsigned int task_count = kGridM * total_nblocks;
  for (unsigned int task = blockIdx.x; task < task_count;
       task += gridDim.x) {
    const unsigned int group_span = kGroupM * total_nblocks;
    const unsigned int group = task / group_span;
    const unsigned int first_m = group * kGroupM;
    const unsigned int active_m =
        kGridM - first_m < kGroupM ? kGridM - first_m : kGroupM;
    const unsigned int group_offset = task % group_span;
    const unsigned int m_tile = first_m + group_offset % active_m;
    unsigned int local_nblock = group_offset / active_m;

    Fp8DevicePartition selected = partition0;
    if constexpr (kPartitionCount >= 2U) {
      if (local_nblock >= partition0.nblocks) {
        local_nblock -= partition0.nblocks;
        selected = partition1;
        if constexpr (kPartitionCount == 3U) {
          if (local_nblock >= partition1.nblocks) {
            local_nblock -= partition1.nblocks;
            selected = partition2;
          }
        }
      }
    }
    execute_fp8_task<kColumns>(storage, selected, activations, m_tile,
                               local_nblock);
  }
}

__global__ __launch_bounds__(kSm87P40ProjectionResetThreads, 2) void
sm87_p40_projection_reset_gate_up_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ interleaved_marlin_weight,
    const std::uint8_t* __restrict__ interleaved_marlin_scales,
    const float* __restrict__ marlin_global_scale,
    std::uint16_t* __restrict__ activated_output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<ProjectionResetStorage*>(dynamic_storage);
  constexpr unsigned int kGridM = 625U;
  constexpr unsigned int kGridN = 272U;
  constexpr unsigned int kTasks = kGridM * kGridN;
  const float global_scale = marlin_global_scale[0];
  // Four adjacent M64 owners consume each interleaved N128 slab before the
  // raster advances N.  Four A slabs plus one B/scale slab fit below Orin's
  // 4 MiB L2 budget; the final group contains the one remaining M tile.
  for (unsigned int task = blockIdx.x; task < kTasks;
       task += gridDim.x) {
    constexpr unsigned int kGroupM = 4U;
    constexpr unsigned int kGroupSpan = kGroupM * kGridN;
    const unsigned int group = task / kGroupSpan;
    const unsigned int first_m = group * kGroupM;
    const unsigned int active_m =
        kGridM - first_m < kGroupM ? kGridM - first_m : kGroupM;
    const unsigned int group_offset = task % kGroupSpan;
    const unsigned int m_tile = first_m + group_offset % active_m;
    const unsigned int n_tile = group_offset / active_m;
    execute_task<kSm87P40ProjectionResetHidden,
                 kSm87P40ProjectionResetMergedGateUp, true>(
        storage, input, interleaved_marlin_weight,
        interleaved_marlin_scales, global_scale, m_tile, n_tile,
        activated_output);
  }
}

__global__ __launch_bounds__(kSm87P40ProjectionResetThreads, 2) void
sm87_p40_projection_reset_down_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ marlin_weight,
    const std::uint8_t* __restrict__ marlin_scales,
    const float* __restrict__ marlin_global_scale,
    std::uint16_t* __restrict__ residual_in_out) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<ProjectionResetStorage*>(dynamic_storage);
  constexpr unsigned int kGridM = 625U;
  constexpr unsigned int kGridN = 40U;
  constexpr unsigned int kTasks = kGridM * kGridN;
  const float global_scale = marlin_global_scale[0];
  // One Down M64 A slab plus one N128 B/scale slab fits in L2, while two A
  // slabs do not.  Keep A resident by sweeping all N tiles before advancing M.
  for (unsigned int task = blockIdx.x; task < kTasks;
       task += gridDim.x) {
    const unsigned int m_tile = task / kGridN;
    const unsigned int n_tile = task % kGridN;
    execute_task<kSm87P40ProjectionResetIntermediate,
                 kSm87P40ProjectionResetHidden, false>(
        storage, input, marlin_weight, marlin_scales, global_scale,
        m_tile, n_tile, residual_in_out);
  }
}

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

[[nodiscard]] bool overlaps(const ByteRange& left,
                            const ByteRange& right) noexcept {
  return left.valid && right.valid && left.begin < right.end &&
         right.begin < left.end;
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

[[nodiscard]] cudaError_t validate_launch(
    const Sm87P40ProjectionResetNvFp4Role role,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count, std::uint16_t* const output) noexcept {
  const auto plan = sm87_p40_projection_reset_nvfp4_plan(role, token_count);
  if (!plan.valid()) {
    return cudaErrorNotSupported;
  }
  if (!aligned(input, 16U) || !aligned(marlin_weight, 16U) ||
      !aligned(marlin_scales, 16U) ||
      !aligned(marlin_global_scale, alignof(float)) ||
      !aligned(output, 16U)) {
    return cudaErrorInvalidValue;
  }
  const ByteRange input_range = make_range(
      input, token_count * plan.input_features * sizeof(std::uint16_t));
  const ByteRange weight_range = make_range(
      marlin_weight,
      plan.input_features * plan.packed_output_features / 2U);
  const ByteRange scale_range = make_range(
      marlin_scales,
      plan.input_features * plan.packed_output_features / 16U);
  const ByteRange global_scale_range =
      make_range(marlin_global_scale, sizeof(float));
  const ByteRange output_range = make_range(
      output,
      token_count * plan.published_output_features * sizeof(std::uint16_t));
  const std::array<ByteRange, 5U> ranges{
      input_range, weight_range, scale_range, global_scale_range,
      output_range};
  for (const ByteRange& range : ranges) {
    if (!range.valid) {
      return cudaErrorInvalidValue;
    }
  }
  for (std::size_t left = 0U; left < ranges.size(); ++left) {
    for (std::size_t right = left + 1U; right < ranges.size(); ++right) {
      if (overlaps(ranges[left], ranges[right])) {
        return cudaErrorInvalidValue;
      }
    }
  }
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  for (const void* const pointer :
       std::array<const void*, 5U>{input, marlin_weight, marlin_scales,
                                  marlin_global_scale, output}) {
    status = validate_device_pointer(pointer, device);
    if (status != cudaSuccess) {
      return status;
    }
  }
  return cudaSuccess;
}

template <bool kGateUp>
[[nodiscard]] cudaError_t configure_kernel() noexcept {
  const auto kernel = kGateUp ? sm87_p40_projection_reset_gate_up_kernel
                              : sm87_p40_projection_reset_down_kernel;
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87P40ProjectionResetDynamicSharedBytes));
}

template <bool kGateUp>
[[nodiscard]] cudaError_t read_resources(
    Sm87P40ProjectionResetResources* const resources) noexcept {
  cudaError_t status = configure_kernel<kGateUp>();
  if (status != cudaSuccess) {
    return status;
  }
  const auto kernel = kGateUp ? sm87_p40_projection_reset_gate_up_kernel
                              : sm87_p40_projection_reset_down_kernel;
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel,
      static_cast<int>(kSm87P40ProjectionResetThreads),
      kSm87P40ProjectionResetDynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87P40ProjectionResetDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  return attributes.localSizeBytes == 0U && active_blocks >= 2
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool infer_fp8_role(
    const Sm87Fp8PrefillSupermatrixPartition* const partitions,
    const std::size_t partition_count, const std::size_t columns,
    Sm87P40ProjectionResetFp8Role* const role) noexcept {
  if (partitions == nullptr || role == nullptr) {
    return false;
  }
  if (columns == 5'120U && partition_count == 2U &&
      partitions[0].rows == 10'240U &&
      partitions[1].rows == 6'144U) {
    *role = Sm87P40ProjectionResetFp8Role::kLinearQkvZInput;
    return true;
  }
  if (columns == 5'120U && partition_count == 3U &&
      partitions[0].rows == 12'288U &&
      partitions[1].rows == 1'024U &&
      partitions[2].rows == 1'024U) {
    *role = Sm87P40ProjectionResetFp8Role::kFullQkvInput;
    return true;
  }
  if (columns == 6'144U && partition_count == 1U &&
      partitions[0].rows == 5'120U) {
    *role = Sm87P40ProjectionResetFp8Role::kAttentionOutput;
    return true;
  }
  return false;
}

[[nodiscard]] cudaError_t validate_fp8_launch(
    const Sm87Fp8PrefillSupermatrixPartition* const partitions,
    const std::size_t partition_count,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t columns,
    Sm87P40ProjectionResetFp8Role* const role,
    Fp8DevicePartition (&device_partitions)[3],
    unsigned int* const total_nblocks) noexcept {
  if (!kProjectionResetAdmitted || role == nullptr ||
      total_nblocks == nullptr ||
      !infer_fp8_role(partitions, partition_count, columns, role)) {
    return cudaErrorNotSupported;
  }
  const auto plan = sm87_p40_projection_reset_fp8_plan(*role, token_count);
  if (!plan.valid() || plan.partition_count != partition_count ||
      plan.input_features != columns || !aligned(activations, 16U)) {
    return cudaErrorInvalidValue;
  }
  const ByteRange activation_range = make_range(
      activations, token_count * columns * sizeof(std::uint16_t));
  if (!activation_range.valid) {
    return cudaErrorInvalidValue;
  }
  std::array<ByteRange, 3U> sidecar_ranges{};
  std::array<ByteRange, 3U> output_ranges{};
  *total_nblocks = 0U;
  for (std::size_t index = 0U; index < partition_count; ++index) {
    const auto& partition = partitions[index];
    if (!aligned(partition.register_feed_sidecar, 16U) ||
        !aligned(partition.output, alignof(std::uint32_t)) ||
        !std::isfinite(partition.weight_scale) ||
        partition.weight_scale < 0.0F ||
        (partition.rows % kSm87P40ProjectionResetFp8TileN) != 0U) {
      return cudaErrorInvalidValue;
    }
    sidecar_ranges[index] = make_range(
        partition.register_feed_sidecar, partition.rows * columns);
    output_ranges[index] = make_range(
        partition.output,
        token_count * partition.rows * sizeof(std::uint16_t));
    if (!sidecar_ranges[index].valid || !output_ranges[index].valid ||
        overlaps(activation_range, sidecar_ranges[index]) ||
        overlaps(activation_range, output_ranges[index]) ||
        overlaps(sidecar_ranges[index], output_ranges[index])) {
      return cudaErrorInvalidValue;
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (overlaps(sidecar_ranges[index], sidecar_ranges[prior]) ||
          overlaps(sidecar_ranges[index], output_ranges[prior]) ||
          overlaps(output_ranges[index], sidecar_ranges[prior]) ||
          overlaps(output_ranges[index], output_ranges[prior])) {
        return cudaErrorInvalidValue;
      }
    }
    const std::size_t nblocks =
        partition.rows / kSm87P40ProjectionResetFp8TileN;
    if (nblocks >
        std::numeric_limits<unsigned int>::max() - *total_nblocks) {
      return cudaErrorInvalidValue;
    }
    device_partitions[index] = {
        reinterpret_cast<const uint4*>(
            partition.register_feed_sidecar),
        partition.output, partition.weight_scale,
        static_cast<unsigned int>(partition.rows),
        static_cast<unsigned int>(nblocks)};
    *total_nblocks += static_cast<unsigned int>(nblocks);
  }
  return *total_nblocks == plan.grid_n ? cudaSuccess
                                       : cudaErrorInvalidValue;
}

template <unsigned int kColumns, unsigned int kPartitionCount,
          unsigned int kGroupM>
[[nodiscard]] cudaError_t configure_fp8_kernel() noexcept {
  return cudaFuncSetAttribute(
      sm87_p40_projection_reset_fp8_supermatrix_kernel<
          kColumns, kPartitionCount, kGroupM>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87P40ProjectionResetFp8DynamicSharedBytes));
}

template <unsigned int kColumns, unsigned int kPartitionCount,
          unsigned int kGroupM>
[[nodiscard]] cudaError_t read_fp8_resources(
    Sm87P40ProjectionResetResources* const resources) noexcept {
  cudaError_t status =
      configure_fp8_kernel<kColumns, kPartitionCount, kGroupM>();
  if (status != cudaSuccess) {
    return status;
  }
  const auto kernel =
      sm87_p40_projection_reset_fp8_supermatrix_kernel<
          kColumns, kPartitionCount, kGroupM>;
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel,
      static_cast<int>(kSm87P40ProjectionResetFp8Threads),
      kSm87P40ProjectionResetFp8DynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87P40ProjectionResetFp8DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  return attributes.localSizeBytes == 0U && active_blocks >= 1
             ? cudaSuccess
             : cudaErrorNotSupported;
}

template <unsigned int kColumns, unsigned int kPartitionCount,
          unsigned int kGroupM>
[[nodiscard]] int launch_fp8_kernel(
    const Fp8DevicePartition (&partitions)[3],
    const std::uint16_t* const activations,
    const unsigned int total_nblocks,
    void* const cuda_stream) noexcept {
  cudaError_t status =
      configure_fp8_kernel<kColumns, kPartitionCount, kGroupM>();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  sm87_p40_projection_reset_fp8_supermatrix_kernel<
      kColumns, kPartitionCount, kGroupM>
      <<<static_cast<unsigned int>(
             kSm87P40ProjectionResetFp8PersistentCtas),
         static_cast<unsigned int>(kSm87P40ProjectionResetFp8Threads),
         kSm87P40ProjectionResetFp8DynamicSharedBytes,
         reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
          partitions[0], partitions[1], partitions[2], activations,
          total_nblocks);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_p40_projection_reset_nvfp4_capability_cuda(
    const Sm87P40ProjectionResetNvFp4Role role,
    const std::size_t token_count,
    Sm87P40ProjectionResetCapability* const capability) noexcept {
  if (capability == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *capability = {};
  capability->plan =
      sm87_p40_projection_reset_nvfp4_plan(role, token_count);
  if (!kProjectionResetAdmitted || !capability->plan.valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  capability->device = device;
  capability->compute_major = properties.major;
  capability->compute_minor = properties.minor;
  capability->sm_count = properties.multiProcessorCount;
  capability->shared_bytes_per_sm =
      static_cast<std::size_t>(properties.sharedMemPerMultiprocessor);
  capability->supported =
      properties.major == 8 && properties.minor == 7 &&
      properties.multiProcessorCount ==
          static_cast<int>(kSm87P40ProjectionResetSmCount) &&
      capability->shared_bytes_per_sm >=
          2U * kSm87P40ProjectionResetDynamicSharedBytes;
  return static_cast<int>(capability->supported ? cudaSuccess
                                                : cudaErrorNotSupported);
}

int query_sm87_p40_projection_reset_nvfp4_resources_cuda(
    const Sm87P40ProjectionResetNvFp4Role role,
    const std::size_t token_count,
    Sm87P40ProjectionResetResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  Sm87P40ProjectionResetCapability capability{};
  const auto capability_status = static_cast<cudaError_t>(
      query_sm87_p40_projection_reset_nvfp4_capability_cuda(
          role, token_count, &capability));
  if (capability_status != cudaSuccess || !capability.supported) {
    return static_cast<int>(capability_status == cudaSuccess
                                ? cudaErrorNotSupported
                                : capability_status);
  }
  if (role == Sm87P40ProjectionResetNvFp4Role::kInterleavedGateUpSilu) {
    return static_cast<int>(read_resources<true>(resources));
  }
  if (role == Sm87P40ProjectionResetNvFp4Role::kDownResidual) {
    return static_cast<int>(read_resources<false>(resources));
  }
  return static_cast<int>(cudaErrorNotSupported);
}

int query_sm87_p40_projection_reset_fp8_capability_cuda(
    const Sm87P40ProjectionResetFp8Role role,
    const std::size_t token_count,
    Sm87P40ProjectionResetFp8Capability* const capability) noexcept {
  if (capability == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *capability = {};
  capability->plan = sm87_p40_projection_reset_fp8_plan(role, token_count);
  if (!kProjectionResetAdmitted || !capability->plan.valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  capability->device = device;
  capability->compute_major = properties.major;
  capability->compute_minor = properties.minor;
  capability->sm_count = properties.multiProcessorCount;
  capability->shared_bytes_per_sm =
      static_cast<std::size_t>(properties.sharedMemPerMultiprocessor);
  capability->opt_in_shared_bytes_per_block =
      static_cast<std::size_t>(properties.sharedMemPerBlockOptin);
  capability->supported =
      properties.major == 8 && properties.minor == 7 &&
      properties.multiProcessorCount ==
          static_cast<int>(kSm87P40ProjectionResetSmCount) &&
      capability->shared_bytes_per_sm >=
          kSm87P40ProjectionResetFp8DynamicSharedBytes &&
      capability->opt_in_shared_bytes_per_block >=
          kSm87P40ProjectionResetFp8DynamicSharedBytes;
  return static_cast<int>(capability->supported ? cudaSuccess
                                                : cudaErrorNotSupported);
}

int query_sm87_p40_projection_reset_fp8_resources_cuda(
    const Sm87P40ProjectionResetFp8Role role,
    const std::size_t token_count,
    Sm87P40ProjectionResetResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  Sm87P40ProjectionResetFp8Capability capability{};
  const auto capability_status = static_cast<cudaError_t>(
      query_sm87_p40_projection_reset_fp8_capability_cuda(
          role, token_count, &capability));
  if (capability_status != cudaSuccess || !capability.supported) {
    return static_cast<int>(capability_status == cudaSuccess
                                ? cudaErrorNotSupported
                                : capability_status);
  }
  if (role == Sm87P40ProjectionResetFp8Role::kLinearQkvZInput) {
    return static_cast<int>(read_fp8_resources<5'120U, 2U, 2U>(resources));
  }
  if (role == Sm87P40ProjectionResetFp8Role::kFullQkvInput) {
    return static_cast<int>(read_fp8_resources<5'120U, 3U, 2U>(resources));
  }
  if (role == Sm87P40ProjectionResetFp8Role::kAttentionOutput) {
    return static_cast<int>(read_fp8_resources<6'144U, 1U, 1U>(resources));
  }
  return static_cast<int>(cudaErrorNotSupported);
}

int launch_sm87_p40_projection_reset_nvfp4_gate_up_silu_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const interleaved_marlin_weight,
    const std::uint8_t* const interleaved_marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const activated_output,
    void* const cuda_stream) noexcept {
  const auto role =
      Sm87P40ProjectionResetNvFp4Role::kInterleavedGateUpSilu;
  cudaError_t status = validate_launch(
      role, input, interleaved_marlin_weight, interleaved_marlin_scales,
      marlin_global_scale, token_count, activated_output);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  Sm87P40ProjectionResetResources resources{};
  status = static_cast<cudaError_t>(
      query_sm87_p40_projection_reset_nvfp4_resources_cuda(
          role, token_count, &resources));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = configure_kernel<true>();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  sm87_p40_projection_reset_gate_up_kernel
      <<<static_cast<unsigned int>(kSm87P40ProjectionResetPersistentCtas),
         static_cast<unsigned int>(kSm87P40ProjectionResetThreads),
         kSm87P40ProjectionResetDynamicSharedBytes,
         reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
          input, interleaved_marlin_weight, interleaved_marlin_scales,
          marlin_global_scale, activated_output);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_projection_reset_nvfp4_down_residual_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const residual_in_out,
    void* const cuda_stream) noexcept {
  const auto role = Sm87P40ProjectionResetNvFp4Role::kDownResidual;
  cudaError_t status = validate_launch(
      role, input, marlin_weight, marlin_scales, marlin_global_scale,
      token_count, residual_in_out);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  Sm87P40ProjectionResetResources resources{};
  status = static_cast<cudaError_t>(
      query_sm87_p40_projection_reset_nvfp4_resources_cuda(
          role, token_count, &resources));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = configure_kernel<false>();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  sm87_p40_projection_reset_down_kernel
      <<<static_cast<unsigned int>(kSm87P40ProjectionResetPersistentCtas),
         static_cast<unsigned int>(kSm87P40ProjectionResetThreads),
         kSm87P40ProjectionResetDynamicSharedBytes,
         reinterpret_cast<cudaStream_t>(cuda_stream)>>>(
          input, marlin_weight, marlin_scales, marlin_global_scale,
          residual_in_out);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_p40_projection_reset_fp8_supermatrix_cuda(
    const Sm87Fp8PrefillSupermatrixPartition* const partitions,
    const std::size_t partition_count,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t columns,
    void* const cuda_stream) noexcept {
  Sm87P40ProjectionResetFp8Role role =
      Sm87P40ProjectionResetFp8Role::kLinearQkvZInput;
  Fp8DevicePartition device_partitions[3]{};
  unsigned int total_nblocks = 0U;
  const cudaError_t status = validate_fp8_launch(
      partitions, partition_count, activations, token_count, columns,
      &role, device_partitions, &total_nblocks);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (role == Sm87P40ProjectionResetFp8Role::kLinearQkvZInput) {
    return launch_fp8_kernel<5'120U, 2U, 2U>(
        device_partitions, activations, total_nblocks, cuda_stream);
  }
  if (role == Sm87P40ProjectionResetFp8Role::kFullQkvInput) {
    return launch_fp8_kernel<5'120U, 3U, 2U>(
        device_partitions, activations, total_nblocks, cuda_stream);
  }
  return launch_fp8_kernel<6'144U, 1U, 1U>(
      device_partitions, activations, total_nblocks, cuda_stream);
}

}  // namespace q3x::kernels
