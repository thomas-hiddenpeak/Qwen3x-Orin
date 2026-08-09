#include "q3x/kernels/sm87_nvfp4_prefill_shape_wide.h"

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

// Compiling this translation unit does not make the route selectable.  The
// isolated admission switch is deliberately private to q3x_kernels and no
// production runner consumes these entry points.
#if defined(Q3X_ENABLE_NVFP4_SHAPE_WIDE_PREFILL_ADMISSION)
inline constexpr bool kShapeWideKernelBodyAdmitted = true;
#else
inline constexpr bool kShapeWideKernelBodyAdmitted = false;
#endif

inline constexpr unsigned int kWarpSize = 32U;
inline constexpr unsigned int kWarps = 8U;
inline constexpr unsigned int kK16PerStage = 4U;
inline constexpr unsigned int kGateM16Panels = 8U;
inline constexpr unsigned int kDownM16Panels = 8U;
inline constexpr unsigned int kGateN64Tiles =
    kSm87NvFp4PrefillShapeWideIntermediate /
    kSm87NvFp4PrefillShapeWideGateBranchTileN;

using Bf16MarlinType = marlin::MarlinScalarType<vllm::kBFloat16.id()>;
using FragA = typename Bf16MarlinType::FragA;
using FragB = typename Bf16MarlinType::FragB;
using FragC = typename Bf16MarlinType::FragC;

static_assert(kSm87NvFp4PrefillShapeWideThreads == kWarps * kWarpSize);
static_assert(kGateN64Tiles == 272U);

struct alignas(32) ShapeWideGateStorage {
  uint4 activations[kSm87NvFp4PrefillShapeWideGatePipelineStages]
                   [kSm87NvFp4PrefillShapeWideGateTileM *
                    kSm87NvFp4PrefillShapeWideTileK / 8U];
  int4 weights[kSm87NvFp4PrefillShapeWideGatePipelineStages]
              [kK16PerStage * 2U * kWarpSize];
  int4 scales[kSm87NvFp4PrefillShapeWideGatePipelineStages]
             [kK16PerStage * 2U * 4U];
};

struct alignas(32) ShapeWideDownStorage {
  uint4 activations[kSm87NvFp4PrefillShapeWideDownPipelineStages]
                   [kSm87NvFp4PrefillShapeWideDownTileM *
                    kSm87NvFp4PrefillShapeWideTileK / 8U];
  int4 weights[kSm87NvFp4PrefillShapeWideDownPipelineStages]
              [kK16PerStage * 2U * kWarpSize];
  int4 scales[kSm87NvFp4PrefillShapeWideDownPipelineStages]
             [kK16PerStage * 2U * 4U];
};

static_assert(sizeof(ShapeWideGateStorage) ==
              kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes);
static_assert(sizeof(ShapeWideDownStorage) ==
              kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes);

struct DecodedBPair {
  FragB gate;
  FragB up;
};

struct DecodedBDownPair {
  FragB first;
  FragB second;
};

struct ByteRange {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
};

[[nodiscard]] __host__ __device__ constexpr unsigned int transform_a_cell(
    const unsigned int logical_cell) noexcept {
  const unsigned int row = logical_cell / 8U;
  return row * 8U + ((logical_cell % 8U) ^ (row % 8U));
}

template <bool kCacheAll>
__device__ __forceinline__ void cp_async_zfill_16(
    void* const shared_destination, const void* const global_source,
    const bool valid) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  if constexpr (kCacheAll) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(shared_address), "l"(global_source),
                   "r"(valid ? 16U : 0U)
                 : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(shared_address), "l"(global_source),
                   "r"(valid ? 16U : 0U)
                 : "memory");
  }
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      valid ? *reinterpret_cast<const uint4*>(global_source)
            : make_uint4(0U, 0U, 0U, 0U);
#endif
}

__device__ __forceinline__ void cp_async_cg_16(
    void* const shared_destination, const void* const global_source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const std::uint32_t shared_address =
      static_cast<std::uint32_t>(__cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(global_source)
               : "memory");
#else
  *reinterpret_cast<int4*>(shared_destination) =
      *reinterpret_cast<const int4*>(global_source);
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

[[nodiscard]] __device__ __forceinline__ std::uint32_t
decode_selected_scale_pair_bits(const int encoded_scale_word,
                                const unsigned int j) {
  constexpr std::uint32_t kSignMask = 0x8000'8000U;
  constexpr std::uint32_t kMagnitudeMask = 0x7f00'7f00U;
  std::uint32_t encoded = static_cast<std::uint32_t>(encoded_scale_word);
  if ((j & 1U) == 0U) {
    encoded <<= 8U;
  }
  return ((encoded & kSignMask) >> 1U) |
         ((encoded & kMagnitudeMask) >> 4U);
}

[[nodiscard]] __device__ __forceinline__ nv_bfloat162 broadcast_bf16_bits(
    const std::uint16_t bits) {
  const std::uint32_t pair = static_cast<std::uint32_t>(bits) |
                             (static_cast<std::uint32_t>(bits) << 16U);
  return *reinterpret_cast<const nv_bfloat162*>(&pair);
}

__device__ __forceinline__ void decode_selected_n8(
    const int quantized, const int encoded_scale_word, const unsigned int j,
    const unsigned int half, FragB* const decoded) {
  marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
      half == 0U ? quantized << 8 : quantized,
      reinterpret_cast<nv_bfloat162*>(decoded));
  const std::uint32_t scale_pair_bits =
      decode_selected_scale_pair_bits(encoded_scale_word, j);
  const std::uint16_t selected_scale = static_cast<std::uint16_t>(
      half == 0U ? scale_pair_bits : scale_pair_bits >> 16U);
  const nv_bfloat162 scale = broadcast_bf16_bits(selected_scale);
  (*decoded)[0] = __hmul2((*decoded)[0], scale);
  (*decoded)[1] = __hmul2((*decoded)[1], scale);
}

__device__ __forceinline__ void decode_n16(
    const int quantized, const int encoded_scale_word,
    const unsigned int j, DecodedBDownPair* const decoded) {
  marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
      quantized << 8,
      reinterpret_cast<nv_bfloat162*>(&decoded->first));
  marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
      quantized,
      reinterpret_cast<nv_bfloat162*>(&decoded->second));
  const std::uint32_t scale_pair_bits =
      decode_selected_scale_pair_bits(encoded_scale_word, j);
  const nv_bfloat162 first_scale = broadcast_bf16_bits(
      static_cast<std::uint16_t>(scale_pair_bits));
  const nv_bfloat162 second_scale = broadcast_bf16_bits(
      static_cast<std::uint16_t>(scale_pair_bits >> 16U));
  decoded->first[0] = __hmul2(decoded->first[0], first_scale);
  decoded->first[1] = __hmul2(decoded->first[1], first_scale);
  decoded->second[0] = __hmul2(decoded->second[0], second_scale);
  decoded->second[1] = __hmul2(decoded->second[1], second_scale);
}

template <unsigned int kK16Plane>
__device__ __forceinline__ void load_gate_b_pair(
    const ShapeWideGateStorage* const storage,
    const unsigned int shared_slot, const unsigned int warp,
    const unsigned int lane, DecodedBPair* const decoded) {
  static_assert(kK16Plane < kK16PerStage);
  const unsigned int lane_group = lane / 4U;
  const unsigned int j = warp / 2U;
  const unsigned int half = warp % 2U;
  const auto* const encoded_scale_words =
      reinterpret_cast<const int*>(storage->scales[shared_slot]);
  const auto* const gate_quant_words = reinterpret_cast<const int*>(
      &storage->weights[shared_slot][kK16Plane * 64U + lane]);
  const auto* const up_quant_words = reinterpret_cast<const int*>(
      &storage->weights[shared_slot][kK16Plane * 64U + 32U + lane]);
  const int gate_scale = encoded_scale_words
      [kK16Plane * 32U + lane_group * 2U + j / 2U];
  const int up_scale = encoded_scale_words
      [kK16Plane * 32U + 16U + lane_group * 2U + j / 2U];
  decode_selected_n8(gate_quant_words[j], gate_scale, j, half,
                     &decoded->gate);
  decode_selected_n8(up_quant_words[j], up_scale, j, half, &decoded->up);
}

template <unsigned int kK16Plane>
__device__ __forceinline__ void load_down_b(
    const ShapeWideDownStorage* const storage,
    const unsigned int shared_slot, const unsigned int warp,
    const unsigned int lane, DecodedBDownPair* const decoded) {
  static_assert(kK16Plane < kK16PerStage);
  const unsigned int lane_group = lane / 4U;
  const unsigned int n64_tile = warp / 4U;
  const unsigned int j = warp % 4U;
  const auto* const quantized_words = reinterpret_cast<const int*>(
      &storage->weights[shared_slot]
                       [kK16Plane * 64U + n64_tile * 32U + lane]);
  const auto* const encoded_scale_words =
      reinterpret_cast<const int*>(storage->scales[shared_slot]);
  const int scale = encoded_scale_words
      [kK16Plane * 32U + n64_tile * 16U + lane_group * 2U + j / 2U];
  decode_n16(quantized_words[j], scale, j, decoded);
}

template <unsigned int kK16Plane>
__device__ __forceinline__ void consume_gate_b_pair(
    const ShapeWideGateStorage* const storage,
    const unsigned int shared_slot, const unsigned int lane,
    const DecodedBPair& decoded,
    FragC (&accumulators)[kGateM16Panels][2]) {
  static_assert(kK16Plane < kK16PerStage);
  unsigned int transformed_a_cell = transform_a_cell(
      2U * kK16Plane + 8U * (lane % 16U) + lane / 16U);
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kGateM16Panels; ++m_panel) {
    FragA activation;
    marlin::ldsm<4, vllm::kBFloat16.id()>(
        activation,
        &storage->activations[shared_slot][transformed_a_cell]);
    marlin::mma<vllm::kBFloat16.id(), false>(
        activation, decoded.gate, accumulators[m_panel][0]);
    marlin::mma<vllm::kBFloat16.id(), false>(
        activation, decoded.up, accumulators[m_panel][1]);
    transformed_a_cell += 128U;
    asm volatile("" : "+r"(transformed_a_cell));
  }
}

template <unsigned int kK16Plane>
__device__ __forceinline__ void consume_down_b(
    const ShapeWideDownStorage* const storage,
    const unsigned int shared_slot, const unsigned int lane,
    const DecodedBDownPair& decoded,
    FragC (&accumulators)[kDownM16Panels][2]) {
  static_assert(kK16Plane < kK16PerStage);
  unsigned int transformed_a_cell = transform_a_cell(
      2U * kK16Plane + 8U * (lane % 16U) + lane / 16U);
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kDownM16Panels; ++m_panel) {
    FragA activation;
    marlin::ldsm<4, vllm::kBFloat16.id()>(
        activation,
        &storage->activations[shared_slot][transformed_a_cell]);
    marlin::mma<vllm::kBFloat16.id(), false>(
        activation, decoded.first, accumulators[m_panel][0]);
    marlin::mma<vllm::kBFloat16.id(), false>(
        activation, decoded.second, accumulators[m_panel][1]);
    transformed_a_cell += 128U;
    asm volatile("" : "+r"(transformed_a_cell));
  }
}

__device__ __forceinline__ void issue_gate_stage(
    ShapeWideGateStorage* const storage, const unsigned int shared_slot,
    const std::uint16_t* const input,
    const std::uint8_t* const merged_marlin_weight,
    const std::uint8_t* const merged_marlin_scales,
    const std::size_t first_token, const unsigned int valid_rows,
    const unsigned int output_tile, const unsigned int first_k) {
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int logical_cell =
        threadIdx.x + pass * kSm87NvFp4PrefillShapeWideThreads;
    const unsigned int row = logical_cell / 8U;
    const unsigned int chunk = logical_cell % 8U;
    const bool valid = row < valid_rows;
    const auto* const safe_source = reinterpret_cast<const uint4*>(
        input +
        (first_token + (valid ? row : 0U)) *
            kSm87NvFp4PrefillShapeWideHidden +
        first_k);
    cp_async_zfill_16<true>(
        &storage->activations[shared_slot][transform_a_cell(logical_cell)],
        safe_source + chunk, valid);
  }

  const unsigned int k16_plane = threadIdx.x / 64U;
  const unsigned int source = (threadIdx.x % 64U) / 32U;
  const unsigned int lane = threadIdx.x % 32U;
  const unsigned int n64_tile = output_tile + source * kGateN64Tiles;
  const auto* const packed_weight =
      reinterpret_cast<const int4*>(merged_marlin_weight);
  const std::size_t weight_index =
      static_cast<std::size_t>(first_k / 16U + k16_plane) *
          (kSm87NvFp4PrefillShapeWideMergedGateUp / 2U) +
      static_cast<std::size_t>(n64_tile) * 32U + lane;
  cp_async_cg_16(
      &storage->weights[shared_slot]
                       [k16_plane * 64U + source * 32U + lane],
      &packed_weight[weight_index]);

  if (threadIdx.x < 32U) {
    const unsigned int scale_k16 = threadIdx.x / 8U;
    const unsigned int scale_source = (threadIdx.x % 8U) / 4U;
    const unsigned int scale_vector = threadIdx.x % 4U;
    const unsigned int scale_n64_tile =
        output_tile + scale_source * kGateN64Tiles;
    const auto* const encoded_scales =
        reinterpret_cast<const int4*>(merged_marlin_scales);
    const std::size_t scale_index =
        static_cast<std::size_t>(first_k / 16U + scale_k16) *
            (kSm87NvFp4PrefillShapeWideMergedGateUp / 16U) +
        static_cast<std::size_t>(scale_n64_tile) * 4U + scale_vector;
    cp_async_cg_16(
        &storage->scales[shared_slot]
                       [scale_k16 * 8U + scale_source * 4U + scale_vector],
        &encoded_scales[scale_index]);
  }
  cp_async_commit_group();
}

__device__ __forceinline__ void issue_down_stage(
    ShapeWideDownStorage* const storage, const unsigned int shared_slot,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const std::size_t first_token, const unsigned int valid_rows,
    const unsigned int output_tile, const unsigned int first_k) {
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int logical_cell =
        threadIdx.x + pass * kSm87NvFp4PrefillShapeWideThreads;
    const unsigned int row = logical_cell / 8U;
    const unsigned int chunk = logical_cell % 8U;
    const bool valid = row < valid_rows;
    const auto* const safe_source = reinterpret_cast<const uint4*>(
        input +
        (first_token + (valid ? row : 0U)) *
            kSm87NvFp4PrefillShapeWideIntermediate +
        first_k);
    cp_async_zfill_16<false>(
        &storage->activations[shared_slot][transform_a_cell(logical_cell)],
        safe_source + chunk, valid);
  }

  {
    const unsigned int k16_plane = threadIdx.x / 64U;
    const unsigned int source = (threadIdx.x % 64U) / 32U;
    const unsigned int lane = threadIdx.x % 32U;
    const unsigned int n64_tile = output_tile * 2U + source;
    const auto* const packed_weight =
        reinterpret_cast<const int4*>(marlin_weight);
    const std::size_t weight_index =
        static_cast<std::size_t>(first_k / 16U + k16_plane) *
            (kSm87NvFp4PrefillShapeWideHidden / 2U) +
        static_cast<std::size_t>(n64_tile) * 32U + lane;
    cp_async_cg_16(
        &storage->weights[shared_slot]
                         [k16_plane * 64U + source * 32U + lane],
        &packed_weight[weight_index]);
  }

  if (threadIdx.x < 32U) {
    const unsigned int scale_k16 = threadIdx.x / 8U;
    const unsigned int scale_source = (threadIdx.x % 8U) / 4U;
    const unsigned int scale_vector = threadIdx.x % 4U;
    const unsigned int scale_n64_tile = output_tile * 2U + scale_source;
    const auto* const encoded_scales =
        reinterpret_cast<const int4*>(marlin_scales);
    const std::size_t scale_index =
        static_cast<std::size_t>(first_k / 16U + scale_k16) *
            (kSm87NvFp4PrefillShapeWideHidden / 16U) +
        static_cast<std::size_t>(scale_n64_tile) * 4U + scale_vector;
    cp_async_cg_16(
        &storage->scales[shared_slot]
                        [scale_k16 * 8U + scale_source * 4U +
                         scale_vector],
        &encoded_scales[scale_index]);
  }
  cp_async_commit_group();
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
gate_up_silu_mul_bf16_scalar(const std::uint16_t gate_bits,
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

__global__ __launch_bounds__(kSm87NvFp4PrefillShapeWideThreads, 2) void
sm87_nvfp4_prefill_shape_wide_gate_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ merged_marlin_weight,
    const std::uint8_t* __restrict__ merged_marlin_scales,
    const float* __restrict__ marlin_global_scale,
    std::uint16_t* __restrict__ activated) {
  constexpr unsigned int kGridN = 272U;
  constexpr unsigned int kGridM = 313U;
  constexpr unsigned int kGroupM = 2U;
  constexpr unsigned int kK64Stages =
      kSm87NvFp4PrefillShapeWideHidden /
      kSm87NvFp4PrefillShapeWideTileK;
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<ShapeWideGateStorage*>(dynamic_storage);

  // Humming/Triton-style grouped ordering: two adjacent M owners consume the
  // same Gate/Up N64 pair before the raster advances N. The final odd M tile
  // is a one-row group, so every one of the 313x272 tiles remains bijective.
  const unsigned int group_blocks = kGroupM * kGridN;
  const unsigned int group = blockIdx.x / group_blocks;
  const unsigned int first_group_m = group * kGroupM;
  const unsigned int remaining_m = kGridM - first_group_m;
  const unsigned int active_group_m =
      remaining_m < kGroupM ? remaining_m : kGroupM;
  const unsigned int group_offset = blockIdx.x % group_blocks;
  const unsigned int m_tile =
      first_group_m + group_offset % active_group_m;
  const unsigned int output_tile = group_offset / active_group_m;
  const std::size_t first_token =
      static_cast<std::size_t>(m_tile) *
      kSm87NvFp4PrefillShapeWideGateTileM;
  const unsigned int valid_rows = static_cast<unsigned int>(
      kSm87NvFp4PrefillShapeWideTokens - first_token <
              kSm87NvFp4PrefillShapeWideGateTileM
          ? kSm87NvFp4PrefillShapeWideTokens - first_token
          : kSm87NvFp4PrefillShapeWideGateTileM);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;

  FragC accumulators[kGateM16Panels][2];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kGateM16Panels; ++m_panel) {
#pragma unroll
    for (unsigned int branch = 0U; branch < 2U; ++branch) {
#pragma unroll
      for (unsigned int value = 0U; value < 4U; ++value) {
        accumulators[m_panel][branch].elems[value] = 0.0F;
      }
    }
  }

  issue_gate_stage(storage, 0U, input, merged_marlin_weight,
                   merged_marlin_scales, first_token, valid_rows,
                   output_tile, 0U);
  issue_gate_stage(storage, 1U, input, merged_marlin_weight,
                   merged_marlin_scales, first_token, valid_rows,
                   output_tile, kSm87NvFp4PrefillShapeWideTileK);
  issue_gate_stage(storage, 2U, input, merged_marlin_weight,
                   merged_marlin_scales, first_token, valid_rows,
                   output_tile, 2U * kSm87NvFp4PrefillShapeWideTileK);

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kK64Stages; ++stage) {
    if (stage + 2U < kK64Stages) {
      cp_async_wait_group<2U>();
    } else if (stage + 1U < kK64Stages) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();

    const unsigned int shared_slot = stage % 3U;
    DecodedBPair decoded;
    load_gate_b_pair<0U>(storage, shared_slot, warp, lane, &decoded);
    consume_gate_b_pair<0U>(storage, shared_slot, lane, decoded,
                            accumulators);
    load_gate_b_pair<1U>(storage, shared_slot, warp, lane, &decoded);
    consume_gate_b_pair<1U>(storage, shared_slot, lane, decoded,
                            accumulators);
    load_gate_b_pair<2U>(storage, shared_slot, warp, lane, &decoded);
    consume_gate_b_pair<2U>(storage, shared_slot, lane, decoded,
                            accumulators);
    load_gate_b_pair<3U>(storage, shared_slot, warp, lane, &decoded);
    consume_gate_b_pair<3U>(storage, shared_slot, lane, decoded,
                            accumulators);

    __syncthreads();
    if (stage + 3U < kK64Stages) {
      issue_gate_stage(
          storage, shared_slot, input, merged_marlin_weight,
          merged_marlin_scales, first_token, valid_rows, output_tile,
          (stage + 3U) * kSm87NvFp4PrefillShapeWideTileK);
    }
  }

  // End the accumulator lifetime before exact expf.  The dead pipeline
  // storage is large enough for [Gate/Up][M128][N64] BF16.
  const float global_scale = marlin_global_scale[0];
  auto* const staged = reinterpret_cast<std::uint32_t*>(storage);
  constexpr unsigned int kPairsPerRow = 32U;
  constexpr unsigned int kPairsPerBranch =
      kSm87NvFp4PrefillShapeWideGateTileM * kPairsPerRow;
  const unsigned int pair_column = warp * 4U + lane_in_group;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kGateM16Panels; ++m_panel) {
    const unsigned int local_token0 = m_panel * 16U + lane_group;
    const unsigned int local_token1 = local_token0 + 8U;
    const FragC& gate = accumulators[m_panel][0];
    const FragC& up = accumulators[m_panel][1];
    if (local_token0 < valid_rows) {
      const unsigned int index =
          local_token0 * kPairsPerRow + pair_column;
      staged[index] = pack_scaled_bf16_pair(
          gate.elems[0], gate.elems[1], global_scale);
      staged[kPairsPerBranch + index] = pack_scaled_bf16_pair(
          up.elems[0], up.elems[1], global_scale);
    }
    if (local_token1 < valid_rows) {
      const unsigned int index =
          local_token1 * kPairsPerRow + pair_column;
      staged[index] = pack_scaled_bf16_pair(
          gate.elems[2], gate.elems[3], global_scale);
      staged[kPairsPerBranch + index] = pack_scaled_bf16_pair(
          up.elems[2], up.elems[3], global_scale);
    }
  }
  __syncthreads();

  const auto* const staged_bf16 =
      reinterpret_cast<const std::uint16_t*>(staged);
  constexpr unsigned int kBf16PerBranch = 2U * kPairsPerBranch;
#pragma unroll 1
  for (unsigned int m_panel = 0U; m_panel < kGateM16Panels; ++m_panel) {
    const unsigned int local_token0 = m_panel * 16U + lane_group;
    const unsigned int local_token1 = local_token0 + 8U;
    const unsigned int output_column =
        output_tile * 64U + warp * 8U + 2U * lane_in_group;
    if (local_token0 < valid_rows) {
      const unsigned int pair_index =
          local_token0 * kPairsPerRow + pair_column;
#pragma unroll
      for (unsigned int element = 0U; element < 2U; ++element) {
        const unsigned int index = 2U * pair_index + element;
        activated[(first_token + local_token0) *
                      kSm87NvFp4PrefillShapeWideIntermediate +
                  output_column + element] =
            gate_up_silu_mul_bf16_scalar(
                staged_bf16[index],
                staged_bf16[kBf16PerBranch + index]);
      }
    }
    if (local_token1 < valid_rows) {
      const unsigned int pair_index =
          local_token1 * kPairsPerRow + pair_column;
#pragma unroll
      for (unsigned int element = 0U; element < 2U; ++element) {
        const unsigned int index = 2U * pair_index + element;
        activated[(first_token + local_token1) *
                      kSm87NvFp4PrefillShapeWideIntermediate +
                  output_column + element] =
            gate_up_silu_mul_bf16_scalar(
                staged_bf16[index],
                staged_bf16[kBf16PerBranch + index]);
      }
    }
  }
}

__global__ __launch_bounds__(kSm87NvFp4PrefillShapeWideThreads, 2) void
sm87_nvfp4_prefill_shape_wide_down_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ marlin_weight,
    const std::uint8_t* __restrict__ marlin_scales,
    const float* __restrict__ marlin_global_scale,
    const std::uint16_t* residual, std::uint16_t* output) {
  constexpr unsigned int kGridN = 40U;
  constexpr unsigned int kK64Stages =
      kSm87NvFp4PrefillShapeWideIntermediate /
      kSm87NvFp4PrefillShapeWideTileK;
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<ShapeWideDownStorage*>(dynamic_storage);

  // Group-M=1 is the strict A-major raster: all N128 owners of one M128
  // activation tile are adjacent before advancing M.
  const unsigned int m_tile = blockIdx.x / kGridN;
  const unsigned int output_tile = blockIdx.x % kGridN;
  const std::size_t first_token =
      static_cast<std::size_t>(m_tile) *
      kSm87NvFp4PrefillShapeWideDownTileM;
  const unsigned int valid_rows = static_cast<unsigned int>(
      kSm87NvFp4PrefillShapeWideTokens - first_token <
              kSm87NvFp4PrefillShapeWideDownTileM
          ? kSm87NvFp4PrefillShapeWideTokens - first_token
          : kSm87NvFp4PrefillShapeWideDownTileM);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;

  FragC accumulators[kDownM16Panels][2];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kDownM16Panels; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < 2U; ++n_panel) {
#pragma unroll
      for (unsigned int value = 0U; value < 4U; ++value) {
        accumulators[m_panel][n_panel].elems[value] = 0.0F;
      }
    }
  }

  issue_down_stage(storage, 0U, input, marlin_weight, marlin_scales,
                   first_token, valid_rows, output_tile, 0U);
  issue_down_stage(storage, 1U, input, marlin_weight, marlin_scales,
                   first_token, valid_rows, output_tile,
                   kSm87NvFp4PrefillShapeWideTileK);
  issue_down_stage(storage, 2U, input, marlin_weight, marlin_scales,
                   first_token, valid_rows, output_tile,
                   2U * kSm87NvFp4PrefillShapeWideTileK);

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kK64Stages; ++stage) {
    if (stage + 2U < kK64Stages) {
      cp_async_wait_group<2U>();
    } else if (stage + 1U < kK64Stages) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();

    const unsigned int shared_slot = stage % 3U;
    DecodedBDownPair decoded;
    load_down_b<0U>(storage, shared_slot, warp, lane, &decoded);
    consume_down_b<0U>(storage, shared_slot, lane, decoded, accumulators);
    load_down_b<1U>(storage, shared_slot, warp, lane, &decoded);
    consume_down_b<1U>(storage, shared_slot, lane, decoded, accumulators);
    load_down_b<2U>(storage, shared_slot, warp, lane, &decoded);
    consume_down_b<2U>(storage, shared_slot, lane, decoded, accumulators);
    load_down_b<3U>(storage, shared_slot, warp, lane, &decoded);
    consume_down_b<3U>(storage, shared_slot, lane, decoded, accumulators);

    __syncthreads();
    if (stage + 3U < kK64Stages) {
      issue_down_stage(
          storage, shared_slot, input, marlin_weight, marlin_scales,
          first_token, valid_rows, output_tile,
          (stage + 3U) * kSm87NvFp4PrefillShapeWideTileK);
    }
  }

  const float global_scale = marlin_global_scale[0];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kDownM16Panels; ++m_panel) {
    const unsigned int local_token0 = m_panel * 16U + lane_group;
    const unsigned int local_token1 = local_token0 + 8U;
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < 2U; ++n_panel) {
      const unsigned int output_column =
          output_tile * 128U + warp * 16U + n_panel * 8U +
          2U * lane_in_group;
      const FragC& branch = accumulators[m_panel][n_panel];
      if (local_token0 < valid_rows) {
        const std::size_t output_index =
            (first_token + local_token0) *
                kSm87NvFp4PrefillShapeWideHidden +
            output_column;
        const std::uint32_t residual_bits =
            *reinterpret_cast<const std::uint32_t*>(residual +
                                                     output_index);
        const std::uint32_t result = add_residual_bf16_pair(
            pack_scaled_bf16_pair(branch.elems[0], branch.elems[1],
                                  global_scale),
            residual_bits);
        *reinterpret_cast<std::uint32_t*>(output + output_index) = result;
      }
      if (local_token1 < valid_rows) {
        const std::size_t output_index =
            (first_token + local_token1) *
                kSm87NvFp4PrefillShapeWideHidden +
            output_column;
        const std::uint32_t residual_bits =
            *reinterpret_cast<const std::uint32_t*>(residual +
                                                     output_index);
        const std::uint32_t result = add_residual_bf16_pair(
            pack_scaled_bf16_pair(branch.elems[2], branch.elems[3],
                                  global_scale),
            residual_bits);
        *reinterpret_cast<std::uint32_t*>(output + output_index) = result;
      }
    }
  }
}

[[nodiscard]] constexpr bool multiply_would_overflow(
    const std::size_t left, const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] bool make_range(const void* const pointer,
                              const std::size_t bytes,
                              ByteRange* const range) noexcept {
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

[[nodiscard]] constexpr bool overlaps(const ByteRange& left,
                                      const ByteRange& right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] cudaError_t validate_device_pointer(
    const void* const pointer, const int expected_device) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    return status;
  }
#if CUDART_VERSION >= 10000
  if (attributes.type != cudaMemoryTypeDevice ||
      attributes.device != expected_device) {
    return cudaErrorInvalidValue;
  }
#else
  if (attributes.memoryType != cudaMemoryTypeDevice ||
      attributes.device != expected_device) {
    return cudaErrorInvalidValue;
  }
#endif
  return cudaSuccess;
}

[[nodiscard]] cudaError_t query_device_contract(
    const Sm87NvFp4PrefillShapeWidePlan& plan,
    Sm87NvFp4PrefillShapeWideCapability* const capability) noexcept {
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
  capability->plan = plan;
  capability->device = device;
  capability->compute_major = properties.major;
  capability->compute_minor = properties.minor;
  capability->sm_count = properties.multiProcessorCount;
  capability->optin_shared_bytes_per_block =
      static_cast<std::size_t>(properties.sharedMemPerBlockOptin);
  capability->shared_bytes_per_sm =
      static_cast<std::size_t>(properties.sharedMemPerMultiprocessor);
  capability->supported =
      kShapeWideKernelBodyAdmitted && properties.major == 8 &&
      properties.minor == 7 &&
      properties.multiProcessorCount ==
          static_cast<int>(kSm87NvFp4PrefillShapeWideSmCount) &&
      capability->optin_shared_bytes_per_block >=
          plan.dynamic_shared_bytes &&
      capability->shared_bytes_per_sm >=
          kSm87NvFp4PrefillShapeWideMinimumBlocksPerSm *
              plan.dynamic_shared_bytes;
  return capability->supported ? cudaSuccess : cudaErrorNotSupported;
}

template <Sm87NvFp4PrefillShapeWideRole kRole>
[[nodiscard]] cudaError_t configure_kernel() noexcept {
  if constexpr (kRole == Sm87NvFp4PrefillShapeWideRole::kGateUp) {
    return cudaFuncSetAttribute(
        sm87_nvfp4_prefill_shape_wide_gate_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(
            kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes));
  } else {
    return cudaFuncSetAttribute(
        sm87_nvfp4_prefill_shape_wide_down_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(
            kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes));
  }
}

template <Sm87NvFp4PrefillShapeWideRole kRole>
[[nodiscard]] cudaError_t query_kernel_resources(
    Sm87NvFp4PrefillShapeWideResources* const resources) noexcept {
  constexpr std::size_t kDynamicSharedBytes =
      kRole == Sm87NvFp4PrefillShapeWideRole::kGateUp
          ? kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes
          : kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes;
  cudaError_t status = configure_kernel<kRole>();
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  if constexpr (kRole == Sm87NvFp4PrefillShapeWideRole::kGateUp) {
    status = cudaFuncGetAttributes(
        &attributes, sm87_nvfp4_prefill_shape_wide_gate_kernel);
  } else {
    status = cudaFuncGetAttributes(
        &attributes, sm87_nvfp4_prefill_shape_wide_down_kernel);
  }
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  if constexpr (kRole == Sm87NvFp4PrefillShapeWideRole::kGateUp) {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, sm87_nvfp4_prefill_shape_wide_gate_kernel,
        static_cast<int>(kSm87NvFp4PrefillShapeWideThreads),
        kDynamicSharedBytes);
  } else {
    status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &active_blocks, sm87_nvfp4_prefill_shape_wide_down_kernel,
        static_cast<int>(kSm87NvFp4PrefillShapeWideThreads),
        kDynamicSharedBytes);
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
          static_cast<int>(
              kSm87NvFp4PrefillShapeWideMaximumRegisters) &&
      resources->local_bytes == 0U &&
      resources->maximum_threads_per_block >=
          static_cast<int>(kSm87NvFp4PrefillShapeWideThreads) &&
      resources->active_blocks_per_sm >=
          static_cast<int>(
              kSm87NvFp4PrefillShapeWideMinimumBlocksPerSm) &&
      total_shared <= kSm87NvFp4PrefillShapeWideSharedLimitBytes;
  return admitted ? cudaSuccess : cudaErrorNotSupported;
}

[[nodiscard]] cudaError_t validate_launch_arguments(
    const Sm87NvFp4PrefillShapeWideRole role,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::uint16_t* const residual, const std::size_t token_count,
    std::uint16_t* const output) noexcept {
  const auto plan = sm87_nvfp4_prefill_shape_wide_plan(role, token_count);
  if (!plan.valid()) {
    return cudaErrorNotSupported;
  }
  if (!aligned(input, 16U) || !aligned(marlin_weight, 16U) ||
      !aligned(marlin_scales, 16U) ||
      !aligned(marlin_global_scale, alignof(float)) ||
      !aligned(output, 16U)) {
    return cudaErrorInvalidValue;
  }
  if (role == Sm87NvFp4PrefillShapeWideRole::kGateUp) {
    if (residual != nullptr) {
      return cudaErrorInvalidValue;
    }
  } else if (!aligned(residual, 16U) ||
             residual != static_cast<const std::uint16_t*>(output)) {
    return cudaErrorInvalidValue;
  }

  if (multiply_would_overflow(token_count, plan.input_features) ||
      multiply_would_overflow(token_count,
                              plan.published_output_features) ||
      multiply_would_overflow(plan.input_features,
                              plan.weight_output_features)) {
    return cudaErrorInvalidValue;
  }
  const std::size_t input_bytes =
      token_count * plan.input_features * sizeof(std::uint16_t);
  const std::size_t weight_bytes =
      plan.input_features * plan.weight_output_features / 2U;
  const std::size_t scale_bytes =
      plan.input_features * plan.weight_output_features / 16U;
  const std::size_t output_bytes =
      token_count * plan.published_output_features * sizeof(std::uint16_t);

  std::array<ByteRange, 5U> ranges{};
  if (!make_range(input, input_bytes, &ranges[0]) ||
      !make_range(marlin_weight, weight_bytes, &ranges[1]) ||
      !make_range(marlin_scales, scale_bytes, &ranges[2]) ||
      !make_range(marlin_global_scale, sizeof(float), &ranges[3]) ||
      !make_range(output, output_bytes, &ranges[4])) {
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

template <Sm87NvFp4PrefillShapeWideRole kRole>
[[nodiscard]] cudaError_t launch_kernel(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::uint16_t* const residual, const std::size_t token_count,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  cudaError_t status = configure_kernel<kRole>();
  if (status != cudaSuccess) {
    return status;
  }
  const auto plan = sm87_nvfp4_prefill_shape_wide_plan(kRole, token_count);
  const std::size_t blocks = plan.grid_m * plan.grid_n;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  if constexpr (kRole == Sm87NvFp4PrefillShapeWideRole::kGateUp) {
    sm87_nvfp4_prefill_shape_wide_gate_kernel
        <<<static_cast<unsigned int>(blocks),
           static_cast<unsigned int>(kSm87NvFp4PrefillShapeWideThreads),
           kSm87NvFp4PrefillShapeWideGateDynamicSharedBytes, stream>>>(
            input, marlin_weight, marlin_scales, marlin_global_scale,
            output);
  } else {
    sm87_nvfp4_prefill_shape_wide_down_kernel
        <<<static_cast<unsigned int>(blocks),
           static_cast<unsigned int>(kSm87NvFp4PrefillShapeWideThreads),
           kSm87NvFp4PrefillShapeWideDownDynamicSharedBytes, stream>>>(
            input, marlin_weight, marlin_scales, marlin_global_scale,
            residual, output);
  }
  return cudaPeekAtLastError();
}

[[nodiscard]] int launch_checked(
    const Sm87NvFp4PrefillShapeWideRole role,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::uint16_t* const residual, const std::size_t token_count,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_launch_arguments(
      role, input, marlin_weight, marlin_scales, marlin_global_scale,
      residual, token_count, output);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  Sm87NvFp4PrefillShapeWideCapability capability{};
  const auto capability_status = static_cast<cudaError_t>(
      query_sm87_nvfp4_prefill_shape_wide_capability_cuda(
          role, token_count, &capability));
  if (capability_status != cudaSuccess || !capability.supported) {
    return static_cast<int>(capability_status == cudaSuccess
                                ? cudaErrorNotSupported
                                : capability_status);
  }
  Sm87NvFp4PrefillShapeWideResources resources{};
  const auto resource_status = static_cast<cudaError_t>(
      query_sm87_nvfp4_prefill_shape_wide_resources_cuda(
          role, token_count, &resources));
  if (resource_status != cudaSuccess) {
    return static_cast<int>(resource_status);
  }

  if (role == Sm87NvFp4PrefillShapeWideRole::kGateUp) {
    return static_cast<int>(
        launch_kernel<Sm87NvFp4PrefillShapeWideRole::kGateUp>(
            input, marlin_weight, marlin_scales, marlin_global_scale,
            nullptr, token_count, output, cuda_stream));
  }
  if (role == Sm87NvFp4PrefillShapeWideRole::kDown) {
    return static_cast<int>(
        launch_kernel<Sm87NvFp4PrefillShapeWideRole::kDown>(
            input, marlin_weight, marlin_scales, marlin_global_scale,
            residual, token_count, output, cuda_stream));
  }
  return static_cast<int>(cudaErrorNotSupported);
}

}  // namespace

int query_sm87_nvfp4_prefill_shape_wide_capability_cuda(
    const Sm87NvFp4PrefillShapeWideRole role,
    const std::size_t token_count,
    Sm87NvFp4PrefillShapeWideCapability* const capability) noexcept {
  if (capability == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *capability = {};
  const auto plan = sm87_nvfp4_prefill_shape_wide_plan(role, token_count);
  if (!plan.valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(query_device_contract(plan, capability));
}

int query_sm87_nvfp4_prefill_shape_wide_resources_cuda(
    const Sm87NvFp4PrefillShapeWideRole role,
    const std::size_t token_count,
    Sm87NvFp4PrefillShapeWideResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  if (!sm87_nvfp4_prefill_shape_wide_plan(role, token_count).valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  Sm87NvFp4PrefillShapeWideCapability capability{};
  const auto capability_status = static_cast<cudaError_t>(
      query_sm87_nvfp4_prefill_shape_wide_capability_cuda(
          role, token_count, &capability));
  if (capability_status != cudaSuccess || !capability.supported) {
    return static_cast<int>(capability_status == cudaSuccess
                                ? cudaErrorNotSupported
                                : capability_status);
  }
  if (role == Sm87NvFp4PrefillShapeWideRole::kGateUp) {
    return static_cast<int>(
        query_kernel_resources<Sm87NvFp4PrefillShapeWideRole::kGateUp>(
            resources));
  }
  if (role == Sm87NvFp4PrefillShapeWideRole::kDown) {
    return static_cast<int>(
        query_kernel_resources<Sm87NvFp4PrefillShapeWideRole::kDown>(
            resources));
  }
  return static_cast<int>(cudaErrorNotSupported);
}

int launch_sm87_nvfp4_prefill_shape_wide_gate_up_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const merged_marlin_weight,
    const std::uint8_t* const merged_marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count, std::uint16_t* const activated,
    void* const cuda_stream) noexcept {
  return launch_checked(Sm87NvFp4PrefillShapeWideRole::kGateUp, input,
                        merged_marlin_weight, merged_marlin_scales,
                        marlin_global_scale, nullptr, token_count, activated,
                        cuda_stream);
}

int launch_sm87_nvfp4_prefill_shape_wide_down_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::uint16_t* const residual, const std::size_t token_count,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  return launch_checked(Sm87NvFp4PrefillShapeWideRole::kDown, input,
                        marlin_weight, marlin_scales, marlin_global_scale,
                        residual, token_count, output, cuda_stream);
}

}  // namespace q3x::kernels
