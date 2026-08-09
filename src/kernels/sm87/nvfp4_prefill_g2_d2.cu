#include "q3x/kernels/sm87_nvfp4_prefill_g2_d2.h"

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

// Building the kernel never selects it. Only the independent G2/D2 admission
// definition makes this API observable as supported; there is no C1 or Marlin
// fallback behind these entry points.
#if defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
inline constexpr bool kG2D2KernelBodyAdmitted = true;
#else
inline constexpr bool kG2D2KernelBodyAdmitted = false;
#endif

inline constexpr unsigned int kWarpSize = 32U;
inline constexpr unsigned int kWarps = 8U;
inline constexpr unsigned int kK16PerStage = 4U;
inline constexpr unsigned int kM16Panels = 8U;
inline constexpr unsigned int kN64TilesPerPhysicalTile = 2U;
inline constexpr unsigned int kGateUpN64Tiles =
    kSm87NvFp4PrefillG2D2Intermediate /
    kSm87NvFp4PrefillG2D2GateUpBranchTileN;

using Bf16MarlinType =
    marlin::MarlinScalarType<vllm::kBFloat16.id()>;
using FragA = typename Bf16MarlinType::FragA;
using FragB = typename Bf16MarlinType::FragB;
using FragC = typename Bf16MarlinType::FragC;

static_assert(kSm87NvFp4PrefillG2D2Threads == kWarps * kWarpSize);
static_assert(kGateUpN64Tiles == 272U);

struct alignas(32) NvFp4PrefillG2D2Storage {
  // A is stored as 16-byte cells in Marlin's bank-conflict-free XOR layout.
  uint4 activations[kSm87NvFp4PrefillG2D2PipelineStages]
                   [kSm87NvFp4PrefillG2D2TileM *
                    kSm87NvFp4PrefillG2D2TileK / 8U];
  // Each stage contains four K16 planes and two physical N64 fragments.
  int4 weights[kSm87NvFp4PrefillG2D2PipelineStages]
              [kK16PerStage * kN64TilesPerPhysicalTile * kWarpSize];
  // One N64 scale fragment is four int4 values per K16 plane.
  int4 scales[kSm87NvFp4PrefillG2D2PipelineStages]
             [kK16PerStage * kN64TilesPerPhysicalTile * 4U];
};

static_assert(sizeof(NvFp4PrefillG2D2Storage) ==
              kSm87NvFp4PrefillG2D2DynamicSharedBytes);

struct DecodedBPair {
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
  // This is the selected-word specialization of Marlin's
  // dequant_fp8_scales<BF16,E4M3>. Even j selects Out2 (source bytes 0/2),
  // odd j selects Out1 (source bytes 1/3); no temporary FragS[4] is formed.
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
    const unsigned int j,
    DecodedBPair* const decoded) {
  marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
      quantized << 8, reinterpret_cast<nv_bfloat162*>(&decoded->first));
  marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
      quantized, reinterpret_cast<nv_bfloat162*>(&decoded->second));
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

template <Sm87NvFp4PrefillG2D2Role kRole, unsigned int kK16Plane>
__device__ __forceinline__ void load_decoded_b_pair(
    const NvFp4PrefillG2D2Storage* const storage,
    const unsigned int shared_slot, const unsigned int warp,
    const unsigned int lane,
    DecodedBPair* const decoded) {
  static_assert(kK16Plane < kK16PerStage);
  const unsigned int lane_group = lane / 4U;
  const auto* const encoded_scale_words =
      reinterpret_cast<const int*>(storage->scales[shared_slot]);
  if constexpr (kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2) {
    const unsigned int j = warp / 2U;
    const unsigned int half = warp % 2U;
    const auto* const gate_quant_words = reinterpret_cast<const int*>(
        &storage->weights[shared_slot][kK16Plane * 64U + lane]);
    const auto* const up_quant_words = reinterpret_cast<const int*>(
        &storage->weights[shared_slot][kK16Plane * 64U + 32U + lane]);
    const int gate_scale = encoded_scale_words
        [kK16Plane * 32U + lane_group * 2U + j / 2U];
    const int up_scale = encoded_scale_words
        [kK16Plane * 32U + 16U + lane_group * 2U + j / 2U];
    decode_selected_n8(gate_quant_words[j], gate_scale, j, half,
                       &decoded->first);
    decode_selected_n8(up_quant_words[j], up_scale, j, half,
                       &decoded->second);
  } else {
    const unsigned int n64_tile = warp / 4U;
    const unsigned int j = warp % 4U;
    const auto* const quant_words = reinterpret_cast<const int*>(
        &storage->weights[shared_slot]
                         [kK16Plane * 64U + n64_tile * 32U + lane]);
    const int scale = encoded_scale_words
        [kK16Plane * 32U + n64_tile * 16U + lane_group * 2U + j / 2U];
    decode_n16(quant_words[j], scale, j, decoded);
  }
}

template <Sm87NvFp4PrefillG2D2Role kRole, unsigned int kK16Plane>
__device__ __forceinline__ void consume_decoded_b_pair(
    const NvFp4PrefillG2D2Storage* const storage,
    const unsigned int shared_slot, const unsigned int lane,
    const DecodedBPair& decoded,
    FragC (&accumulators)[kM16Panels][2]) {
  static_assert(kK16Plane < kK16PerStage);
  unsigned int transformed_a_cell = transform_a_cell(
      2U * kK16Plane + 8U * (lane % 16U) + lane / 16U);
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
    FragA activation;
    marlin::ldsm<4, vllm::kBFloat16.id()>(
        activation,
        &storage->activations[shared_slot]
                             [transformed_a_cell]);
    marlin::mma<vllm::kBFloat16.id(), false>(
        activation, decoded.first, accumulators[m_panel][0]);
    marlin::mma<vllm::kBFloat16.id(), false>(
        activation, decoded.second, accumulators[m_panel][1]);
    // Adding sixteen complete XOR-layout rows leaves the intra-row bank
    // permutation unchanged. The dependency fence makes ptxas retain one
    // rolling shared index instead of hoisting eight addresses across the
    // 64-register accumulator lifetime.
    transformed_a_cell += 128U;
    asm volatile("" : "+r"(transformed_a_cell));
  }
}

template <Sm87NvFp4PrefillG2D2Role kRole>
__device__ __forceinline__ void issue_pipeline_stage(
    NvFp4PrefillG2D2Storage* const storage,
    const unsigned int shared_slot, const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const std::size_t first_token, const unsigned int valid_rows,
    const unsigned int output_tile, const unsigned int first_k) {
  constexpr unsigned int kInputFeatures =
      kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2
          ? kSm87NvFp4PrefillG2D2Hidden
          : kSm87NvFp4PrefillG2D2Intermediate;
  constexpr unsigned int kWeightOutputFeatures =
      kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2
          ? kSm87NvFp4PrefillG2D2MergedGateUp
          : kSm87NvFp4PrefillG2D2Hidden;

  // Four 16-byte cells per thread cover M128xK64. A false cp.async predicate
  // zero-fills each invalid M7712 tail row without forming an invalid source.
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int logical_cell =
        threadIdx.x + pass * kSm87NvFp4PrefillG2D2Threads;
    const unsigned int row = logical_cell / 8U;
    const unsigned int chunk = logical_cell % 8U;
    const bool valid = row < valid_rows;
    const auto* const safe_source = reinterpret_cast<const uint4*>(
        input + (first_token + (valid ? row : 0U)) * kInputFeatures +
        first_k);
    cp_async_zfill_16<
        kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2>(
        &storage->activations[shared_slot]
                             [transform_a_cell(logical_cell)],
        safe_source + chunk, valid);
  }

  // Marlin sidecars are K16-major/N64-major. G2 reads Gate p and Up p+272
  // from the canonical GateThenUp artifact; D2 reads adjacent N64 fragments.
  const unsigned int k16_plane = threadIdx.x / 64U;
  const unsigned int source = (threadIdx.x % 64U) / 32U;
  const unsigned int lane = threadIdx.x % 32U;
  const unsigned int n64_tile =
      kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2
          ? output_tile + source * kGateUpN64Tiles
          : output_tile * 2U + source;
  const auto* const packed_weight =
      reinterpret_cast<const int4*>(marlin_weight);
  const std::size_t weight_index =
      static_cast<std::size_t>(first_k / 16U + k16_plane) *
          (kWeightOutputFeatures / 2U) +
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
        kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2
            ? output_tile + scale_source * kGateUpN64Tiles
            : output_tile * 2U + scale_source;
    const auto* const encoded_scales =
        reinterpret_cast<const int4*>(marlin_scales);
    const std::size_t scale_index =
        static_cast<std::size_t>(first_k / 16U + scale_k16) *
            (kWeightOutputFeatures / 16U) +
        static_cast<std::size_t>(scale_n64_tile) * 4U + scale_vector;
    cp_async_cg_16(
        &storage->scales[shared_slot]
                       [scale_k16 * 8U + scale_source * 4U + scale_vector],
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
  const std::uint16_t output0 = encode_bf16_rne(branch0 + residual0);
  const std::uint16_t output1 = encode_bf16_rne(branch1 + residual1);
  return static_cast<std::uint32_t>(output0) |
         (static_cast<std::uint32_t>(output1) << 16U);
}

template <Sm87NvFp4PrefillG2D2Role kRole>
__global__ __launch_bounds__(kSm87NvFp4PrefillG2D2Threads, 2) void
sm87_nvfp4_prefill_g2_d2_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ marlin_weight,
    const std::uint8_t* __restrict__ marlin_scales,
    const float* __restrict__ marlin_global_scale,
    const unsigned int token_count, std::uint16_t* output) {
  constexpr unsigned int kInputFeatures =
      kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2
          ? kSm87NvFp4PrefillG2D2Hidden
          : kSm87NvFp4PrefillG2D2Intermediate;
  constexpr unsigned int kPublishedOutputFeatures =
      kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2
          ? kSm87NvFp4PrefillG2D2Intermediate
          : kSm87NvFp4PrefillG2D2Hidden;
  constexpr unsigned int kGridN =
      kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2 ? 272U : 40U;
  constexpr unsigned int kK64Stages =
      kInputFeatures / kSm87NvFp4PrefillG2D2TileK;

  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<NvFp4PrefillG2D2Storage*>(dynamic_storage);
  const unsigned int m_tiles = static_cast<unsigned int>(
      (token_count + kSm87NvFp4PrefillG2D2TileM - 1U) /
      kSm87NvFp4PrefillG2D2TileM);
  unsigned int m_tile = 0U;
  unsigned int output_tile = 0U;
  if constexpr (kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2) {
    m_tile = blockIdx.x / kGridN;
    output_tile = blockIdx.x % kGridN;
  } else {
    output_tile = blockIdx.x / m_tiles;
    m_tile = blockIdx.x % m_tiles;
  }
  const std::size_t first_token =
      static_cast<std::size_t>(m_tile) * kSm87NvFp4PrefillG2D2TileM;
  const unsigned int valid_rows = static_cast<unsigned int>(
      token_count - first_token < kSm87NvFp4PrefillG2D2TileM
          ? token_count - first_token
          : kSm87NvFp4PrefillG2D2TileM);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;

  FragC accumulators[kM16Panels][2];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
#pragma unroll
    for (unsigned int branch = 0U; branch < 2U; ++branch) {
#pragma unroll
      for (unsigned int value = 0U; value < 4U; ++value) {
        accumulators[m_panel][branch].elems[value] = 0.0F;
      }
    }
  }

  issue_pipeline_stage<kRole>(
      storage, 0U, input, marlin_weight, marlin_scales, first_token,
      valid_rows, output_tile, 0U);
  issue_pipeline_stage<kRole>(
      storage, 1U, input, marlin_weight, marlin_scales, first_token,
      valid_rows, output_tile, kSm87NvFp4PrefillG2D2TileK);

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kK64Stages; ++stage) {
    if (stage + 1U < kK64Stages) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();

    const unsigned int shared_slot = stage % 2U;
    // Keep one decoded B pair live. This preserves the defining eight-panel B
    // fanout while staying inside the hard two-CTA register envelope.
    DecodedBPair decoded;
    load_decoded_b_pair<kRole, 0U>(storage, shared_slot, warp, lane,
                                   &decoded);
    consume_decoded_b_pair<kRole, 0U>(storage, shared_slot, lane,
                                      decoded, accumulators);
    load_decoded_b_pair<kRole, 1U>(storage, shared_slot, warp, lane,
                                   &decoded);
    consume_decoded_b_pair<kRole, 1U>(storage, shared_slot, lane,
                                      decoded, accumulators);
    load_decoded_b_pair<kRole, 2U>(storage, shared_slot, warp, lane,
                                   &decoded);
    consume_decoded_b_pair<kRole, 2U>(storage, shared_slot, lane,
                                      decoded, accumulators);
    load_decoded_b_pair<kRole, 3U>(storage, shared_slot, warp, lane,
                                   &decoded);
    consume_decoded_b_pair<kRole, 3U>(storage, shared_slot, lane,
                                      decoded, accumulators);

    // Every consumer leaves this slot before its stage+2 producer overwrites
    // it. Keeping the overwrite barrier explicit also protects the M7712
    // zero-filled tail from a later async group.
    __syncthreads();
    if (stage + 2U < kK64Stages) {
      issue_pipeline_stage<kRole>(
          storage, shared_slot, input, marlin_weight, marlin_scales,
          first_token, valid_rows, output_tile,
          (stage + 2U) * kSm87NvFp4PrefillG2D2TileK);
    }
  }

  const float global_scale = marlin_global_scale[0];
  if constexpr (kRole == Sm87NvFp4PrefillG2D2Role::kGateUpG2) {
    // The K pipeline is dead here. Reuse its shared allocation to terminate
    // the 64-register accumulator lifetime before exact expf. Layout is
    // [Gate/Up][M128][N64/2 packed BF16 pairs] = 32 KiB.
    auto* const staged = reinterpret_cast<std::uint32_t*>(storage);
    constexpr unsigned int kPairsPerRow = 32U;
    constexpr unsigned int kPairsPerBranch =
        kSm87NvFp4PrefillG2D2TileM * kPairsPerRow;
    const unsigned int pair_column = warp * 4U + lane_in_group;
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
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
    for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
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
          output[(first_token + local_token0) * kPublishedOutputFeatures +
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
          output[(first_token + local_token1) * kPublishedOutputFeatures +
                 output_column + element] =
              gate_up_silu_mul_bf16_scalar(
                  staged_bf16[index],
                  staged_bf16[kBf16PerBranch + index]);
        }
      }
    }
  } else {
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < kM16Panels; ++m_panel) {
      const unsigned int local_token0 = m_panel * 16U + lane_group;
      const unsigned int local_token1 = local_token0 + 8U;
      // A D2 warp owns two adjacent N8 panels. Each panel has its own complete
      // low/high M8 result; preserve branch BF16 RNE before reading and adding
      // the BF16 residual. Invalid M7712 tail rows never read residual memory.
#pragma unroll
      for (unsigned int n_panel = 0U; n_panel < 2U; ++n_panel) {
        const FragC& branch = accumulators[m_panel][n_panel];
        const unsigned int output_column =
            output_tile * 128U + warp * 16U + n_panel * 8U +
            2U * lane_in_group;
        if (local_token0 < valid_rows) {
          const std::size_t output_index =
              (first_token + local_token0) * kPublishedOutputFeatures +
              output_column;
          const std::uint32_t residual_bits =
              *reinterpret_cast<const std::uint32_t*>(output +
                                                       output_index);
          const std::uint32_t result = add_residual_bf16_pair(
              pack_scaled_bf16_pair(branch.elems[0], branch.elems[1],
                                    global_scale),
              residual_bits);
          *reinterpret_cast<std::uint32_t*>(output + output_index) = result;
        }
        if (local_token1 < valid_rows) {
          const std::size_t output_index =
              (first_token + local_token1) * kPublishedOutputFeatures +
              output_column;
          const std::uint32_t residual_bits =
              *reinterpret_cast<const std::uint32_t*>(output +
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
    const Sm87NvFp4PrefillG2D2Plan& plan,
    Sm87NvFp4PrefillG2D2Capability* const capability) noexcept {
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
  capability->supported =
      kG2D2KernelBodyAdmitted && properties.major == 8 &&
      properties.minor == 7 &&
      properties.multiProcessorCount ==
          static_cast<int>(kSm87NvFp4PrefillG2D2SmCount) &&
      properties.sharedMemPerBlockOptin >=
          kSm87NvFp4PrefillG2D2DynamicSharedBytes;
  return capability->supported ? cudaSuccess : cudaErrorNotSupported;
}

template <Sm87NvFp4PrefillG2D2Role kRole>
[[nodiscard]] cudaError_t configure_kernel() noexcept {
  const auto kernel = sm87_nvfp4_prefill_g2_d2_kernel<kRole>;
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4PrefillG2D2DynamicSharedBytes));
}

template <Sm87NvFp4PrefillG2D2Role kRole>
[[nodiscard]] cudaError_t query_kernel_resources(
    Sm87NvFp4PrefillG2D2Resources* const resources) noexcept {
  const auto kernel = sm87_nvfp4_prefill_g2_d2_kernel<kRole>;
  cudaError_t status = configure_kernel<kRole>();
  if (status != cudaSuccess) {
    return status;
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel,
      static_cast<int>(kSm87NvFp4PrefillG2D2Threads),
      kSm87NvFp4PrefillG2D2DynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87NvFp4PrefillG2D2DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;

  const std::size_t total_shared =
      resources->static_shared_bytes + resources->dynamic_shared_bytes;
  const bool admitted =
      resources->registers_per_thread <=
          static_cast<int>(kSm87NvFp4PrefillG2D2MaximumRegisters) &&
      resources->local_bytes == 0U &&
      resources->maximum_threads_per_block >=
          static_cast<int>(kSm87NvFp4PrefillG2D2Threads) &&
      resources->active_blocks_per_sm >=
          static_cast<int>(kSm87NvFp4PrefillG2D2MinimumBlocksPerSm) &&
      total_shared <= kSm87NvFp4PrefillG2D2SharedLimitBytes;
  return admitted ? cudaSuccess : cudaErrorNotSupported;
}

[[nodiscard]] cudaError_t validate_launch_arguments(
    const Sm87NvFp4PrefillG2D2Role role,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::uint16_t* const residual, const std::size_t token_count,
    std::uint16_t* const output) noexcept {
  const auto plan = sm87_nvfp4_prefill_g2_d2_plan(role, token_count);
  if (!plan.valid()) {
    return cudaErrorNotSupported;
  }
  if (!aligned(input, 16U) || !aligned(marlin_weight, 16U) ||
      !aligned(marlin_scales, 16U) ||
      !aligned(marlin_global_scale, alignof(float)) ||
      !aligned(output, 16U)) {
    return cudaErrorInvalidValue;
  }
  if (role == Sm87NvFp4PrefillG2D2Role::kGateUpG2) {
    if (residual != nullptr) {
      return cudaErrorInvalidValue;
    }
  } else if (!aligned(residual, 16U) ||
             residual != static_cast<const std::uint16_t*>(output)) {
    // D2 is deliberately in-place. Accepting an out-of-place residual would
    // add another full tensor traffic path to this supposedly fixed dataflow.
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

template <Sm87NvFp4PrefillG2D2Role kRole>
[[nodiscard]] cudaError_t launch_kernel(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  cudaError_t status = configure_kernel<kRole>();
  if (status != cudaSuccess) {
    return status;
  }
  const auto plan = sm87_nvfp4_prefill_g2_d2_plan(kRole, token_count);
  const std::size_t blocks = plan.grid_m * plan.grid_n;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_nvfp4_prefill_g2_d2_kernel<kRole>
      <<<static_cast<unsigned int>(blocks),
         static_cast<unsigned int>(kSm87NvFp4PrefillG2D2Threads),
         kSm87NvFp4PrefillG2D2DynamicSharedBytes, stream>>>(
          input, marlin_weight, marlin_scales, marlin_global_scale,
          static_cast<unsigned int>(token_count), output);
  return cudaPeekAtLastError();
}

[[nodiscard]] int launch_checked(
    const Sm87NvFp4PrefillG2D2Role role,
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
  Sm87NvFp4PrefillG2D2Capability capability{};
  const auto capability_status = static_cast<cudaError_t>(
      query_sm87_nvfp4_prefill_g2_d2_capability_cuda(
          role, token_count, &capability));
  if (capability_status != cudaSuccess || !capability.supported) {
    return static_cast<int>(capability_status == cudaSuccess
                                ? cudaErrorNotSupported
                                : capability_status);
  }
  Sm87NvFp4PrefillG2D2Resources resources{};
  const auto resource_status = static_cast<cudaError_t>(
      query_sm87_nvfp4_prefill_g2_d2_resources_cuda(
          role, token_count, &resources));
  if (resource_status != cudaSuccess) {
    return static_cast<int>(resource_status);
  }

  if (role == Sm87NvFp4PrefillG2D2Role::kGateUpG2) {
    return static_cast<int>(
        launch_kernel<Sm87NvFp4PrefillG2D2Role::kGateUpG2>(
            input, marlin_weight, marlin_scales, marlin_global_scale,
            token_count, output, cuda_stream));
  }
  if (role == Sm87NvFp4PrefillG2D2Role::kDownD2) {
    return static_cast<int>(
        launch_kernel<Sm87NvFp4PrefillG2D2Role::kDownD2>(
            input, marlin_weight, marlin_scales, marlin_global_scale,
            token_count, output, cuda_stream));
  }
  return static_cast<int>(cudaErrorNotSupported);
}

}  // namespace

int query_sm87_nvfp4_prefill_g2_d2_capability_cuda(
    const Sm87NvFp4PrefillG2D2Role role,
    const std::size_t token_count,
    Sm87NvFp4PrefillG2D2Capability* const capability) noexcept {
  if (capability == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *capability = {};
  const auto plan = sm87_nvfp4_prefill_g2_d2_plan(role, token_count);
  if (!plan.valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(query_device_contract(plan, capability));
}

int query_sm87_nvfp4_prefill_g2_d2_resources_cuda(
    const Sm87NvFp4PrefillG2D2Role role,
    const std::size_t token_count,
    Sm87NvFp4PrefillG2D2Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  if (!sm87_nvfp4_prefill_g2_d2_plan(role, token_count).valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  Sm87NvFp4PrefillG2D2Capability capability{};
  const auto capability_status = static_cast<cudaError_t>(
      query_sm87_nvfp4_prefill_g2_d2_capability_cuda(
          role, token_count, &capability));
  if (capability_status != cudaSuccess || !capability.supported) {
    return static_cast<int>(capability_status == cudaSuccess
                                ? cudaErrorNotSupported
                                : capability_status);
  }
  if (role == Sm87NvFp4PrefillG2D2Role::kGateUpG2) {
    return static_cast<int>(
        query_kernel_resources<Sm87NvFp4PrefillG2D2Role::kGateUpG2>(
            resources));
  }
  if (role == Sm87NvFp4PrefillG2D2Role::kDownD2) {
    return static_cast<int>(
        query_kernel_resources<Sm87NvFp4PrefillG2D2Role::kDownD2>(
            resources));
  }
  return static_cast<int>(cudaErrorNotSupported);
}

int launch_sm87_nvfp4_prefill_gate_up_g2_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const merged_marlin_weight,
    const std::uint8_t* const merged_marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count, std::uint16_t* const activated,
    void* const cuda_stream) noexcept {
  return launch_checked(Sm87NvFp4PrefillG2D2Role::kGateUpG2, input,
                        merged_marlin_weight, merged_marlin_scales,
                        marlin_global_scale, nullptr, token_count, activated,
                        cuda_stream);
}

int launch_sm87_nvfp4_prefill_down_d2_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::uint16_t* const residual, const std::size_t token_count,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  return launch_checked(Sm87NvFp4PrefillG2D2Role::kDownD2, input,
                        marlin_weight, marlin_scales, marlin_global_scale,
                        residual, token_count, output, cuda_stream);
}

}  // namespace q3x::kernels
