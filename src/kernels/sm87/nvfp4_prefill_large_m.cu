#include "q3x/kernels/sm87_nvfp4_prefill_large_m.h"

#include "third_party/vllm_marlin/dequant.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

// Compiling the complete cell does not select it.  The test-only admission
// switch is the sole route that can make the capability visible; the normal
// build remains fail closed and there is no Marlin fallback behind this API.
#if defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
inline constexpr bool kLargeMKernelBodyAdmitted = true;
#else
inline constexpr bool kLargeMKernelBodyAdmitted = false;
#endif

inline constexpr unsigned int kWarpSize = 32U;
inline constexpr unsigned int kWarps = 8U;
inline constexpr unsigned int kNWarps = 4U;
inline constexpr unsigned int kM16PanelsPerWarp = 4U;
inline constexpr unsigned int kN8PanelsPerWarp = 8U;
inline constexpr unsigned int kSharedALeadingDimension = 72U;

static_assert(kSm87NvFp4PrefillLargeMThreads == kWarps * kWarpSize);
static_assert(kWarps == 2U * kNWarps);

struct alignas(32) NvFp4PrefillLargeMStorage {
  // Each A row has eight live uint4 cells (K64 BF16) and one padding cell.
  uint4 activations[kSm87NvFp4PrefillLargeMPipelineStages]
                   [kSm87NvFp4PrefillLargeMTileM * 9U];
  // One stage contains four Marlin K16 planes. Each plane contains four N64
  // fragments, one int4 per consumer lane.
  int4 weights[kSm87NvFp4PrefillLargeMPipelineStages][4U * 4U * 32U];
  // `process_modelopt_scales_kernel` writes one Marlin-encoded byte per
  // K16/N column. These are not canonical block_scale E4M3 bytes. The
  // vendored Marlin reconstruction below and the epilogue global scale are a
  // coupled numerical contract.
  int4 scales[kSm87NvFp4PrefillLargeMPipelineStages][4U * 16U];
};

static_assert(sizeof(NvFp4PrefillLargeMStorage) ==
              kSm87NvFp4PrefillLargeMDynamicSharedBytes);

struct InlineM16N8Accumulator {
  float x0;
  float x1;
  float x2;
  float x3;
};

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_bf16_pair(
    const __nv_bfloat16 low, const __nv_bfloat16 high) {
  return static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
         (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
}

[[nodiscard]] __device__ __forceinline__ nv_bfloat162 broadcast_bf16(
    const __nv_bfloat16 value) {
  const std::uint32_t bits =
      static_cast<std::uint32_t>(__bfloat16_as_ushort(value));
  const std::uint32_t pair = bits | (bits << 16U);
  return *reinterpret_cast<const nv_bfloat162*>(&pair);
}

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

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          bool kActivationCacheAll>
__device__ __forceinline__ void issue_large_m_pipeline_stage(
    NvFp4PrefillLargeMStorage* const storage,
    const unsigned int shared_slot,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const std::size_t first_token,
    const unsigned int valid_rows,
    const unsigned int output_tile,
    const unsigned int first_k) {
  static_assert(kInputFeatures % kSm87NvFp4PrefillLargeMTileK == 0U);
  static_assert(kOutputFeatures % kSm87NvFp4PrefillLargeMTileN == 0U);

  // A: four 16-byte cells per thread cover M128xK64. Invalid rows in the
  // native M32 tail use cp.async zero fill, never an out-of-range pointer.
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index = threadIdx.x + pass *
                                               kSm87NvFp4PrefillLargeMThreads;
    const unsigned int row = index / 8U;
    const unsigned int chunk = index % 8U;
    const bool valid = row < valid_rows;
    const auto* const safe_source = reinterpret_cast<const uint4*>(
        input + (first_token + (valid ? row : 0U)) * kInputFeatures +
        first_k);
    cp_async_zfill_16<kActivationCacheAll>(
        &storage->activations[shared_slot][row * 9U + chunk],
        safe_source + chunk, valid);
  }

  // B: gptq_marlin_repack emits K16-major/N64-major fragments. A K64/N256
  // stage is exactly four K16 planes x four N64 fragments x 32 lane int4s.
  const auto* const packed_weight =
      reinterpret_cast<const int4*>(marlin_weight);
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int local =
        threadIdx.x + pass * kSm87NvFp4PrefillLargeMThreads;
    const unsigned int k16_plane = local / 128U;
    const unsigned int plane_offset = local % 128U;
    const std::size_t global =
        static_cast<std::size_t>(first_k / 16U + k16_plane) *
            (kOutputFeatures / 2U) +
        static_cast<std::size_t>(output_tile) * 128U + plane_offset;
    cp_async_cg_16(&storage->weights[shared_slot][local],
                   &packed_weight[global]);
  }

  // Scale: group-major Marlin sidecar, 16 int4s per K16 plane for N256.
  if (threadIdx.x < 64U) {
    const unsigned int k16_plane = threadIdx.x / 16U;
    const unsigned int plane_offset = threadIdx.x % 16U;
    const auto* const encoded_scales =
        reinterpret_cast<const int4*>(marlin_scales);
    const std::size_t global =
        static_cast<std::size_t>(first_k / 16U + k16_plane) *
            (kOutputFeatures / 16U) +
        static_cast<std::size_t>(output_tile) * 16U + plane_offset;
    cp_async_cg_16(&storage->scales[shared_slot][threadIdx.x],
                   &encoded_scales[global]);
  }
  cp_async_commit_group();
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          bool kBStationary, bool kActivationCacheAll>
__global__ __launch_bounds__(kSm87NvFp4PrefillLargeMThreads, 1) void
sm87_nvfp4_prefill_large_m_kernel(
    const std::uint16_t* __restrict__ input,
    const std::uint8_t* __restrict__ marlin_weight,
    const std::uint8_t* __restrict__ marlin_scales,
    const float* __restrict__ marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* __restrict__ output) {
  static_assert((kInputFeatures == kSm87NvFp4PrefillLargeMHidden &&
                 kOutputFeatures ==
                     kSm87NvFp4PrefillLargeMGateUpOutput) ||
                (kInputFeatures == kSm87NvFp4PrefillLargeMIntermediate &&
                 kOutputFeatures == kSm87NvFp4PrefillLargeMHidden));
  constexpr unsigned int kK64Stages =
      kInputFeatures / kSm87NvFp4PrefillLargeMTileK;
  constexpr unsigned int kOutputTiles =
      kOutputFeatures / kSm87NvFp4PrefillLargeMTileN;

  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage =
      reinterpret_cast<NvFp4PrefillLargeMStorage*>(dynamic_storage);
  const unsigned int m_tiles = static_cast<unsigned int>(
      (token_count + kSm87NvFp4PrefillLargeMTileM - 1U) /
      kSm87NvFp4PrefillLargeMTileM);
  unsigned int m_tile = 0U;
  unsigned int output_tile = 0U;
  if constexpr (kBStationary) {
    output_tile = blockIdx.x / m_tiles;
    m_tile = blockIdx.x % m_tiles;
  } else {
    m_tile = blockIdx.x / kOutputTiles;
    output_tile = blockIdx.x % kOutputTiles;
  }
  const std::size_t first_token =
      static_cast<std::size_t>(m_tile) *
      kSm87NvFp4PrefillLargeMTileM;
  const unsigned int valid_rows = static_cast<unsigned int>(
      token_count - first_token < kSm87NvFp4PrefillLargeMTileM
          ? token_count - first_token
          : kSm87NvFp4PrefillLargeMTileM);

  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int m_warp = warp / kNWarps;
  const unsigned int n_warp = warp % kNWarps;
  const unsigned int lane_group = lane / 4U;

  InlineM16N8Accumulator
      accumulators[kM16PanelsPerWarp][kN8PanelsPerWarp];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16PanelsPerWarp; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp; ++n_panel) {
      accumulators[m_panel][n_panel] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  issue_large_m_pipeline_stage<kInputFeatures, kOutputFeatures,
                               kActivationCacheAll>(
      storage, 0U, input, marlin_weight, marlin_scales, first_token,
      valid_rows, output_tile, 0U);
  issue_large_m_pipeline_stage<kInputFeatures, kOutputFeatures,
                               kActivationCacheAll>(
      storage, 1U, input, marlin_weight, marlin_scales, first_token,
      valid_rows, output_tile, 64U);
  issue_large_m_pipeline_stage<kInputFeatures, kOutputFeatures,
                               kActivationCacheAll>(
      storage, 2U, input, marlin_weight, marlin_scales, first_token,
      valid_rows, output_tile, 128U);

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kK64Stages; ++stage) {
    const unsigned int remaining = kK64Stages - stage;
    if (remaining >= 3U) {
      cp_async_wait_group<2U>();
    } else if (remaining == 2U) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();

    const unsigned int shared_slot = stage % 3U;
    const auto* const shared_a = reinterpret_cast<const __nv_bfloat16*>(
        storage->activations[shared_slot]);

#pragma unroll
    for (unsigned int k16_plane = 0U; k16_plane < 4U; ++k16_plane) {
      const int4 quant = storage->weights[shared_slot]
                                         [k16_plane * 128U + n_warp * 32U +
                                          lane];

      // Mirror marlin_template's grouped-scale lane mapping. Each lane group
      // loads eight Marlin-encoded bytes; dequant_fp8_scales reconstructs
      // eight BF16 values into four FragS objects. This is intentionally not
      // canonical E4M3 block-scale decoding.
      const auto* const encoded_scale_pairs =
          reinterpret_cast<const int2*>(storage->scales[shared_slot]);
      const int2 encoded = encoded_scale_pairs
          [k16_plane * 32U + n_warp * 8U + lane_group];
      using FragB =
          typename marlin::MarlinScalarType<vllm::kBFloat16.id()>::FragB;
      using FragS =
          typename marlin::MarlinScalarType<vllm::kBFloat16.id()>::FragS;
      FragS scale_fragments[4];
      marlin::dequant_fp8_scales<nv_bfloat162, vllm::kFE4M3fn.id()>(
          encoded.x,
          reinterpret_cast<nv_bfloat162*>(&scale_fragments[0]));
      marlin::dequant_fp8_scales<nv_bfloat162, vllm::kFE4M3fn.id()>(
          encoded.y,
          reinterpret_cast<nv_bfloat162*>(&scale_fragments[2]));

      std::uint32_t decoded_b[kN8PanelsPerWarp][2U];
#pragma unroll
      for (unsigned int j = 0U; j < 4U; ++j) {
        const int quant_1 = reinterpret_cast<const int*>(&quant)[j];
        const int quant_0 = quant_1 << 8;
        FragB fragment_b0;
        FragB fragment_b1;
        marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
            quant_0, reinterpret_cast<nv_bfloat162*>(&fragment_b0));
        marlin::dequant<nv_bfloat162, vllm::kFE2M1f.id(), true>(
            quant_1, reinterpret_cast<nv_bfloat162*>(&fragment_b1));
        const auto* const scales = reinterpret_cast<const __nv_bfloat16*>(
            &scale_fragments[j]);
        const nv_bfloat162 scale_0 = broadcast_bf16(scales[0]);
        const nv_bfloat162 scale_1 = broadcast_bf16(scales[1]);
        fragment_b0[0] = __hmul2(fragment_b0[0], scale_0);
        fragment_b0[1] = __hmul2(fragment_b0[1], scale_0);
        fragment_b1[0] = __hmul2(fragment_b1[0], scale_1);
        fragment_b1[1] = __hmul2(fragment_b1[1], scale_1);
        const auto* const b0 =
            reinterpret_cast<const std::uint32_t*>(&fragment_b0);
        const auto* const b1 =
            reinterpret_cast<const std::uint32_t*>(&fragment_b1);
        decoded_b[2U * j][0] = b0[0];
        decoded_b[2U * j][1] = b0[1];
        decoded_b[2U * j + 1U][0] = b1[0];
        decoded_b[2U * j + 1U][1] = b1[1];
      }

      // One A fragment feeds every N8 fragment owned by this warp. Keep the
      // shared load/pack outside the N loop; repeating it for j=0..3 would
      // inflate the A/shared instruction stream by four without adding reuse.
#pragma unroll
      for (unsigned int m_panel = 0U; m_panel < kM16PanelsPerWarp;
           ++m_panel) {
        const unsigned int local_m = m_warp * 64U + m_panel * 16U;
        if (local_m < valid_rows) {
          nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                                 __nv_bfloat16,
                                 nvcuda::wmma::row_major>
              fragment_a;
          nvcuda::wmma::load_matrix_sync(
              fragment_a,
              shared_a + local_m * kSharedALeadingDimension +
                  k16_plane * 16U,
              kSharedALeadingDimension);
          const std::uint32_t a0 =
              pack_bf16_pair(fragment_a.x[0], fragment_a.x[1]);
          const std::uint32_t a1 =
              pack_bf16_pair(fragment_a.x[2], fragment_a.x[3]);
          const std::uint32_t a2 =
              pack_bf16_pair(fragment_a.x[4], fragment_a.x[5]);
          const std::uint32_t a3 =
              pack_bf16_pair(fragment_a.x[6], fragment_a.x[7]);
#pragma unroll
          for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
               ++n_panel) {
            mma_m16n8k16_bf16(
                accumulators[m_panel][n_panel], a0, a1, a2, a3,
                decoded_b[n_panel][0], decoded_b[n_panel][1]);
          }
        }
      }
    }

    // All consumers leave the current slot before stage+3 overwrites it.
    __syncthreads();
    if (stage + 3U < kK64Stages) {
      const unsigned int future_stage = stage + 3U;
      issue_large_m_pipeline_stage<kInputFeatures, kOutputFeatures,
                                   kActivationCacheAll>(
          storage, shared_slot, input, marlin_weight, marlin_scales,
          first_token, valid_rows, output_tile, future_stage * 64U);
    }
  }

  const float global_scale = marlin_global_scale[0];
  const unsigned int lane_octet = lane >> 3U;
  const bool even_group = (lane & 4U) == 0U;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16PanelsPerWarp; ++m_panel) {
#pragma unroll
    for (unsigned int n_pair = 0U; n_pair < 4U; ++n_pair) {
      const InlineM16N8Accumulator accumulator_a =
          accumulators[m_panel][2U * n_pair];
      const InlineM16N8Accumulator accumulator_b =
          accumulators[m_panel][2U * n_pair + 1U];
      const std::uint32_t low_a = pack_bf16_pair(
          __float2bfloat16_rn(accumulator_a.x0 * global_scale),
          __float2bfloat16_rn(accumulator_a.x1 * global_scale));
      const std::uint32_t low_b = pack_bf16_pair(
          __float2bfloat16_rn(accumulator_b.x0 * global_scale),
          __float2bfloat16_rn(accumulator_b.x1 * global_scale));
      const std::uint32_t low_peer = __shfl_xor_sync(
          0xffff'ffffU, even_group ? low_b : low_a, 4, 8);
      const std::uint32_t low_even = even_group ? low_a : low_peer;
      const std::uint32_t low_odd = even_group ? low_peer : low_b;

      const unsigned int local_column =
          n_warp * 64U + n_pair * 16U + 2U * (lane & 7U);
      const unsigned int output_column =
          output_tile * kSm87NvFp4PrefillLargeMTileN + local_column;
      const unsigned int local_low_even_token =
          m_warp * 64U + m_panel * 16U + 2U * lane_octet;
      const std::size_t low_even_token =
          first_token + local_low_even_token;
      if (local_low_even_token < valid_rows) {
        *reinterpret_cast<std::uint32_t*>(
            output + low_even_token * kOutputFeatures + output_column) =
            low_even;
      }
      if (local_low_even_token + 1U < valid_rows) {
        *reinterpret_cast<std::uint32_t*>(
            output + (low_even_token + 1U) * kOutputFeatures +
            output_column) = low_odd;
      }

      const std::uint32_t high_a = pack_bf16_pair(
          __float2bfloat16_rn(accumulator_a.x2 * global_scale),
          __float2bfloat16_rn(accumulator_a.x3 * global_scale));
      const std::uint32_t high_b = pack_bf16_pair(
          __float2bfloat16_rn(accumulator_b.x2 * global_scale),
          __float2bfloat16_rn(accumulator_b.x3 * global_scale));
      const std::uint32_t high_peer = __shfl_xor_sync(
          0xffff'ffffU, even_group ? high_b : high_a, 4, 8);
      const std::uint32_t high_even = even_group ? high_a : high_peer;
      const std::uint32_t high_odd = even_group ? high_peer : high_b;
      const unsigned int local_high_even_token =
          local_low_even_token + 8U;
      const std::size_t high_even_token =
          first_token + local_high_even_token;
      if (local_high_even_token < valid_rows) {
        *reinterpret_cast<std::uint32_t*>(
            output + high_even_token * kOutputFeatures + output_column) =
            high_even;
      }
      if (local_high_even_token + 1U < valid_rows) {
        *reinterpret_cast<std::uint32_t*>(
            output + (high_even_token + 1U) * kOutputFeatures +
            output_column) = high_odd;
      }
    }
  }
}

struct ByteRange {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
};

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

[[nodiscard]] cudaError_t query_device_contract(
    const Sm87NvFp4PrefillLargeMPlan& plan,
    Sm87NvFp4PrefillLargeMCapability* const capability) noexcept {
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
      kLargeMKernelBodyAdmitted && properties.major == 8 &&
      properties.minor == 7 &&
      properties.multiProcessorCount ==
          static_cast<int>(kSm87NvFp4PrefillLargeMSmCount) &&
      properties.sharedMemPerBlockOptin >=
          kSm87NvFp4PrefillLargeMDynamicSharedBytes;
  return capability->supported ? cudaSuccess : cudaErrorNotSupported;
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

[[nodiscard]] cudaError_t validate_launch_arguments(
    const Sm87NvFp4PrefillLargeMRole role,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const output) noexcept {
  const auto plan = sm87_nvfp4_prefill_large_m_plan(role, token_count);
  if (!plan.valid()) {
    return cudaErrorNotSupported;
  }
  if (!aligned(input, 16U) || !aligned(marlin_weight, 16U) ||
      !aligned(marlin_scales, 16U) ||
      !aligned(marlin_global_scale, alignof(float)) ||
      !aligned(output, 16U)) {
    return cudaErrorInvalidValue;
  }

  if (multiply_would_overflow(token_count, plan.input_features) ||
      multiply_would_overflow(token_count, plan.output_features) ||
      multiply_would_overflow(plan.input_features, plan.output_features)) {
    return cudaErrorInvalidValue;
  }
  const std::size_t input_bytes =
      token_count * plan.input_features * sizeof(std::uint16_t);
  const std::size_t weight_bytes =
      plan.input_features * plan.output_features / 2U;
  const std::size_t scale_bytes =
      plan.input_features * plan.output_features / 16U;
  const std::size_t output_bytes =
      token_count * plan.output_features * sizeof(std::uint16_t);

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

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          bool kBStationary, bool kActivationCacheAll>
[[nodiscard]] cudaError_t configure_large_m_kernel() noexcept {
  const auto kernel =
      sm87_nvfp4_prefill_large_m_kernel<kInputFeatures, kOutputFeatures,
                                        kBStationary, kActivationCacheAll>;
  return cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4PrefillLargeMDynamicSharedBytes));
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          bool kBStationary, bool kActivationCacheAll>
[[nodiscard]] cudaError_t query_large_m_kernel_resources(
    Sm87NvFp4PrefillLargeMResources* const resources) noexcept {
  const auto kernel =
      sm87_nvfp4_prefill_large_m_kernel<kInputFeatures, kOutputFeatures,
                                        kBStationary, kActivationCacheAll>;
  cudaError_t status = configure_large_m_kernel<
      kInputFeatures, kOutputFeatures, kBStationary, kActivationCacheAll>();
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
      static_cast<int>(kSm87NvFp4PrefillLargeMThreads),
      kSm87NvFp4PrefillLargeMDynamicSharedBytes);
  if (status != cudaSuccess) {
    return status;
  }
  if (active_blocks < 1) {
    return cudaErrorNotSupported;
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87NvFp4PrefillLargeMDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  return cudaSuccess;
}

template <unsigned int kInputFeatures, unsigned int kOutputFeatures,
          bool kBStationary, bool kActivationCacheAll>
[[nodiscard]] cudaError_t launch_large_m_kernel(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  cudaError_t status = configure_large_m_kernel<
      kInputFeatures, kOutputFeatures, kBStationary, kActivationCacheAll>();
  if (status != cudaSuccess) {
    return status;
  }
  const auto plan = sm87_nvfp4_prefill_large_m_plan(
      kBStationary ? Sm87NvFp4PrefillLargeMRole::kDown
                   : Sm87NvFp4PrefillLargeMRole::kGateUp,
      token_count);
  const std::size_t blocks = plan.grid_m * plan.grid_n;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  sm87_nvfp4_prefill_large_m_kernel<
      kInputFeatures, kOutputFeatures, kBStationary, kActivationCacheAll>
      <<<static_cast<unsigned int>(blocks),
         static_cast<unsigned int>(kSm87NvFp4PrefillLargeMThreads),
         kSm87NvFp4PrefillLargeMDynamicSharedBytes, stream>>>(
          input, marlin_weight, marlin_scales, marlin_global_scale,
          token_count, output);
  return cudaPeekAtLastError();
}

[[nodiscard]] int launch_checked_large_m_cell(
    const Sm87NvFp4PrefillLargeMRole role,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const cudaError_t validation = validate_launch_arguments(
      role, input, marlin_weight, marlin_scales, marlin_global_scale,
      token_count, output);
  if (validation != cudaSuccess) {
    return static_cast<int>(validation);
  }
  Sm87NvFp4PrefillLargeMCapability capability{};
  const auto capability_status = static_cast<cudaError_t>(
      query_sm87_nvfp4_prefill_large_m_capability_cuda(
          role, token_count, &capability));
  if (capability_status != cudaSuccess || !capability.supported) {
    return static_cast<int>(capability_status == cudaSuccess
                                ? cudaErrorNotSupported
                                : capability_status);
  }
  cudaError_t status = cudaErrorNotSupported;
  if (role == Sm87NvFp4PrefillLargeMRole::kGateUp) {
    status = launch_large_m_kernel<
        kSm87NvFp4PrefillLargeMHidden,
        kSm87NvFp4PrefillLargeMGateUpOutput, false, true>(
        input, marlin_weight, marlin_scales, marlin_global_scale,
        token_count, output, cuda_stream);
  } else if (role == Sm87NvFp4PrefillLargeMRole::kDown) {
    status = launch_large_m_kernel<
        kSm87NvFp4PrefillLargeMIntermediate,
        kSm87NvFp4PrefillLargeMHidden, true, false>(
        input, marlin_weight, marlin_scales, marlin_global_scale,
        token_count, output, cuda_stream);
  }
  return static_cast<int>(status);
}

}  // namespace

int query_sm87_nvfp4_prefill_large_m_capability_cuda(
    const Sm87NvFp4PrefillLargeMRole role,
    const std::size_t token_count,
    Sm87NvFp4PrefillLargeMCapability* const capability) noexcept {
  if (capability == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *capability = {};
  const auto plan = sm87_nvfp4_prefill_large_m_plan(role, token_count);
  if (!plan.valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(query_device_contract(plan, capability));
}

int query_sm87_nvfp4_prefill_large_m_resources_cuda(
    const Sm87NvFp4PrefillLargeMRole role,
    const std::size_t token_count,
    Sm87NvFp4PrefillLargeMResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  if (!sm87_nvfp4_prefill_large_m_plan(role, token_count).valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  Sm87NvFp4PrefillLargeMCapability capability{};
  const auto capability_status = static_cast<cudaError_t>(
      query_sm87_nvfp4_prefill_large_m_capability_cuda(
          role, token_count, &capability));
  if (capability_status != cudaSuccess || !capability.supported) {
    return static_cast<int>(capability_status == cudaSuccess
                                ? cudaErrorNotSupported
                                : capability_status);
  }
  cudaError_t status = cudaErrorNotSupported;
  if (role == Sm87NvFp4PrefillLargeMRole::kGateUp) {
    status = query_large_m_kernel_resources<
        kSm87NvFp4PrefillLargeMHidden,
        kSm87NvFp4PrefillLargeMGateUpOutput, false, true>(resources);
  } else if (role == Sm87NvFp4PrefillLargeMRole::kDown) {
    status = query_large_m_kernel_resources<
        kSm87NvFp4PrefillLargeMIntermediate,
        kSm87NvFp4PrefillLargeMHidden, true, false>(resources);
  }
  return static_cast<int>(status);
}

int launch_sm87_nvfp4_prefill_large_m_gate_up_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  return launch_checked_large_m_cell(
      Sm87NvFp4PrefillLargeMRole::kGateUp, input, marlin_weight,
      marlin_scales, marlin_global_scale, token_count, output, cuda_stream);
}

int launch_sm87_nvfp4_prefill_large_m_down_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  return launch_checked_large_m_cell(
      Sm87NvFp4PrefillLargeMRole::kDown, input, marlin_weight,
      marlin_scales, marlin_global_scale, token_count, output, cuda_stream);
}

}  // namespace q3x::kernels
