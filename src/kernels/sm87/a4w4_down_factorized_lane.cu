#include "q3x/kernels/sm87_a4w4_down_factorized_lane.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87A4W4DownFactorizedTileM);
inline constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87A4W4DownFactorizedTileN);
inline constexpr unsigned int kK64PerStage =
    static_cast<unsigned int>(kSm87A4W4DownFactorizedK64PerStage);
inline constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87A4W4DownFactorizedThreads);
inline constexpr unsigned int kPackedK64Bytes = 32U;

template <unsigned int Rows>
struct alignas(16) PackedK256Stage final {
  std::uint8_t plane[kK64PerStage][Rows * kPackedK64Bytes];
};

struct alignas(16) FactorizedScales final {
  std::uint16_t a[kTileM];
  std::uint16_t b[kTileN];
};

struct alignas(16) SharedStorage final {
  PackedK256Stage<kTileM>
      a[kSm87A4W4DownFactorizedAStages];
  PackedK256Stage<kTileN>
      b[kSm87A4W4DownFactorizedBStages];
  FactorizedScales scales;
};

// One warp owns M64N32.  Keeping all 16 fragments as named fields prevents
// an address-taken runtime array from silently turning the whole-K S32 state
// into local memory.
struct alignas(16) Accumulators final {
  Sm87A4W4Accumulator m0n0{};
  Sm87A4W4Accumulator m0n1{};
  Sm87A4W4Accumulator m0n2{};
  Sm87A4W4Accumulator m0n3{};
  Sm87A4W4Accumulator m1n0{};
  Sm87A4W4Accumulator m1n1{};
  Sm87A4W4Accumulator m1n2{};
  Sm87A4W4Accumulator m1n3{};
  Sm87A4W4Accumulator m2n0{};
  Sm87A4W4Accumulator m2n1{};
  Sm87A4W4Accumulator m2n2{};
  Sm87A4W4Accumulator m2n3{};
  Sm87A4W4Accumulator m3n0{};
  Sm87A4W4Accumulator m3n1{};
  Sm87A4W4Accumulator m3n2{};
  Sm87A4W4Accumulator m3n3{};
};

static_assert(sizeof(PackedK256Stage<kTileM>) ==
              kSm87A4W4DownFactorizedStageABytes);
static_assert(sizeof(PackedK256Stage<kTileN>) ==
              kSm87A4W4DownFactorizedStageBBytes);
static_assert(sizeof(FactorizedScales) ==
              kSm87A4W4DownFactorizedScaleBytes);
static_assert(sizeof(SharedStorage) ==
              kSm87A4W4DownFactorizedDynamicSharedBytes);
static_assert(sizeof(Accumulators) == 64U * sizeof(std::int32_t));

[[nodiscard]] constexpr bool aligned(const void* const pointer,
                                     const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] bool byte_ranges_overlap(
    const void* const first, const std::size_t first_bytes,
    const void* const second, const std::size_t second_bytes) noexcept {
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  constexpr std::uintptr_t maximum =
      std::numeric_limits<std::uintptr_t>::max();
  if (first_bytes > maximum - first_begin ||
      second_bytes > maximum - second_begin) {
    return true;
  }
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_begin < second_end && second_begin < first_end;
}

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

__device__ __forceinline__ void cp_async_cg_16(
    void* const destination, const void* const source) noexcept {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(source)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_ca_16(
    void* const destination, const void* const source) noexcept {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(source)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_commit() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.commit_group;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("cp.async.wait_group 0;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

template <unsigned int Rows>
__device__ __forceinline__ void issue_packed_stage(
    PackedK256Stage<Rows>& destination,
    const std::uint8_t* const packed,
    const unsigned int outer_start, const unsigned int k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      Rows * kPackedK64Bytes / 16U;
  static_assert(kVectorsPerPlane <= kThreads);
#pragma unroll
  for (unsigned int plane = 0U; plane < kK64PerStage; ++plane) {
    if (threadIdx.x < kVectorsPerPlane) {
      const unsigned int row = threadIdx.x >> 1U;
      const unsigned int row_vector = threadIdx.x & 1U;
      const unsigned int physical_k64_group =
          kK64PerStage * k256_stage + plane;
      const std::uint8_t* const source =
          packed + sm87_a4w4_down_factorized_packed_offset(
                       outer_start + row, physical_k64_group,
                       16U * row_vector, physical_k64_group_count);
      std::uint8_t* const target =
          destination.plane[plane] +
          sm87_a4w4_swizzled_k64_byte_offset(
              row, 16U * row_vector);
      cp_async_cg_16(target, source);
    }
  }
  cp_async_commit();
}

__device__ __forceinline__ void issue_scales(
    FactorizedScales& destination,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::uint16_t* const b_lane_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start) noexcept {
  if (threadIdx.x < 48U) {
    const bool is_a = threadIdx.x < 32U;
    const unsigned int vector =
        is_a ? threadIdx.x : threadIdx.x - 32U;
    const unsigned int row = 8U * vector;
    const unsigned int outer_start =
        is_a ? m_tile_start : n_tile_start;
    const std::uint16_t* const source =
        (is_a ? a_lane_scales_bf16 : b_lane_scales_bf16) +
        sm87_a4w4_down_factorized_scale_offset(outer_start + row);
    std::uint16_t* const target =
        (is_a ? destination.a : destination.b) + row;
    cp_async_ca_16(target, source);
  }
  cp_async_commit();
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4AFragment
load_a_ldmatrix_x4(const std::uint8_t* const shared_a,
                   const unsigned int lane) noexcept {
  const unsigned int matrix = lane >> 3U;
  const unsigned int logical_row =
      (lane & 7U) + ((matrix & 1U) << 3U);
  const unsigned int logical_byte = (matrix >> 1U) * 16U;
  const auto* const source =
      shared_a +
      sm87_a4w4_swizzled_k64_byte_offset(logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4AFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1), "=r"(fragment.x2),
        "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return fragment;
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4BFragment
load_b_ldmatrix_x2(const std::uint8_t* const shared_b,
                   const unsigned int lane) noexcept {
  const unsigned int provider = lane & 15U;
  const unsigned int logical_row = provider & 7U;
  const unsigned int logical_byte = (provider >> 3U) * 16U;
  const auto* const source =
      shared_b +
      sm87_a4w4_swizzled_k64_byte_offset(logical_row, logical_byte);
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  Sm87A4W4BFragment fragment{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(shared_address)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return fragment;
}

__device__ __forceinline__ void accumulate_k256_stage(
    const PackedK256Stage<kTileM>& a,
    const PackedK256Stage<kTileN>& b,
    const unsigned int warp_m64, const unsigned int warp_n32,
    Accumulators& accumulators) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int local_m = warp_m64 * 64U;
  const unsigned int local_n = warp_n32 * 32U;

#pragma unroll
  for (unsigned int plane = 0U; plane < kK64PerStage; ++plane) {
    const Sm87A4W4BFragment b0 = load_b_ldmatrix_x2(
        b.plane[plane] + (local_n + 0U) * kPackedK64Bytes, lane);
    const Sm87A4W4BFragment b1 = load_b_ldmatrix_x2(
        b.plane[plane] + (local_n + 8U) * kPackedK64Bytes, lane);
    const Sm87A4W4BFragment b2 = load_b_ldmatrix_x2(
        b.plane[plane] + (local_n + 16U) * kPackedK64Bytes, lane);
    const Sm87A4W4BFragment b3 = load_b_ldmatrix_x2(
        b.plane[plane] + (local_n + 24U) * kPackedK64Bytes, lane);

    const Sm87A4W4AFragment a0 = load_a_ldmatrix_x4(
        a.plane[plane] + (local_m + 0U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.m0n0, a0, b0);
    sm87_a4w4_mma_m16n8k64(accumulators.m0n1, a0, b1);
    sm87_a4w4_mma_m16n8k64(accumulators.m0n2, a0, b2);
    sm87_a4w4_mma_m16n8k64(accumulators.m0n3, a0, b3);

    const Sm87A4W4AFragment a1 = load_a_ldmatrix_x4(
        a.plane[plane] + (local_m + 16U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.m1n0, a1, b0);
    sm87_a4w4_mma_m16n8k64(accumulators.m1n1, a1, b1);
    sm87_a4w4_mma_m16n8k64(accumulators.m1n2, a1, b2);
    sm87_a4w4_mma_m16n8k64(accumulators.m1n3, a1, b3);

    const Sm87A4W4AFragment a2 = load_a_ldmatrix_x4(
        a.plane[plane] + (local_m + 32U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.m2n0, a2, b0);
    sm87_a4w4_mma_m16n8k64(accumulators.m2n1, a2, b1);
    sm87_a4w4_mma_m16n8k64(accumulators.m2n2, a2, b2);
    sm87_a4w4_mma_m16n8k64(accumulators.m2n3, a2, b3);

    const Sm87A4W4AFragment a3 = load_a_ldmatrix_x4(
        a.plane[plane] + (local_m + 48U) * kPackedK64Bytes, lane);
    sm87_a4w4_mma_m16n8k64(accumulators.m3n0, a3, b0);
    sm87_a4w4_mma_m16n8k64(accumulators.m3n1, a3, b1);
    sm87_a4w4_mma_m16n8k64(accumulators.m3n2, a3, b2);
    sm87_a4w4_mma_m16n8k64(accumulators.m3n3, a3, b3);
  }
}

template <unsigned int MFragment, unsigned int NFragment>
__device__ __forceinline__ void store_fragment(
    const Sm87A4W4Accumulator& accumulator,
    const FactorizedScales& scales,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_m64,
    const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  static_assert(MFragment < 4U);
  static_assert(NFragment < 4U);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int column1 = column0 + 1U;
  const unsigned int local_m =
      warp_m64 * 64U + MFragment * 16U;
  const unsigned int local_n =
      warp_n32 * 32U + NFragment * 8U;
  const unsigned int base_m = m_tile_start + local_m;
  const unsigned int base_n = n_tile_start + local_n;
  const float a0 = decode_bf16(scales.a[local_m + row0]);
  const float a1 = decode_bf16(scales.a[local_m + row1]);
  const float b0 = decode_bf16(scales.b[local_n + column0]);
  const float b1 = decode_bf16(scales.b[local_n + column1]);

  if (base_m + row0 < logical_token_count) {
    output_bf16[static_cast<std::size_t>(base_m + row0) *
                        output_row_stride_elements +
                    base_n + column0] =
        encode_bf16(__fmul_rn(
            static_cast<float>(accumulator.x0), __fmul_rn(a0, b0)));
    output_bf16[static_cast<std::size_t>(base_m + row0) *
                        output_row_stride_elements +
                    base_n + column1] =
        encode_bf16(__fmul_rn(
            static_cast<float>(accumulator.x1), __fmul_rn(a0, b1)));
  }
  if (base_m + row1 < logical_token_count) {
    output_bf16[static_cast<std::size_t>(base_m + row1) *
                        output_row_stride_elements +
                    base_n + column0] =
        encode_bf16(__fmul_rn(
            static_cast<float>(accumulator.x2), __fmul_rn(a1, b0)));
    output_bf16[static_cast<std::size_t>(base_m + row1) *
                        output_row_stride_elements +
                    base_n + column1] =
        encode_bf16(__fmul_rn(
            static_cast<float>(accumulator.x3), __fmul_rn(a1, b1)));
  }
}

__device__ __forceinline__ void store_accumulators(
    const Accumulators& accumulators,
    const FactorizedScales& scales,
    const unsigned int logical_token_count,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_m64,
    const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
#define Q3X_STORE(M, N)                                                    \
  store_fragment<M##U, N##U>(                                             \
      accumulators.m##M##n##N, scales, logical_token_count, m_tile_start, \
      n_tile_start, warp_m64, warp_n32, output_bf16,                      \
      output_row_stride_elements)
  Q3X_STORE(0, 0);
  Q3X_STORE(0, 1);
  Q3X_STORE(0, 2);
  Q3X_STORE(0, 3);
  Q3X_STORE(1, 0);
  Q3X_STORE(1, 1);
  Q3X_STORE(1, 2);
  Q3X_STORE(1, 3);
  Q3X_STORE(2, 0);
  Q3X_STORE(2, 1);
  Q3X_STORE(2, 2);
  Q3X_STORE(2, 3);
  Q3X_STORE(3, 0);
  Q3X_STORE(3, 1);
  Q3X_STORE(3, 2);
  Q3X_STORE(3, 3);
#undef Q3X_STORE
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownFactorizedThreads,
                      kSm87A4W4DownFactorizedCtasPerSm)
void q3x_sm87_a4w4_down_factorized_lane_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_lane_scales_bf16,
    const unsigned int logical_token_count,
    const unsigned int physical_k64_group_count,
    const unsigned int k256_stage_count,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared = *reinterpret_cast<SharedStorage*>(dynamic_storage);
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int warp_m64 = warp & 3U;
  const unsigned int warp_n32 = warp >> 2U;

  for (unsigned int work = blockIdx.x; work < work_tile_count;
       work += gridDim.x) {
    const unsigned int n_tile = work / m_tile_count;
    const unsigned int m_tile = work - n_tile * m_tile_count;
    const unsigned int m_tile_start = m_tile * kTileM;
    const unsigned int n_tile_start = n_tile * kTileN;
    Accumulators accumulators{};

    issue_scales(shared.scales, a_lane_scales_bf16,
                 b_lane_scales_bf16, m_tile_start, n_tile_start);
    issue_packed_stage(shared.a[0], packed_a, m_tile_start, 0U,
                       physical_k64_group_count);
    issue_packed_stage(shared.b[0], packed_b, n_tile_start, 0U,
                       physical_k64_group_count);
    cp_async_wait_all();
    __syncthreads();

    for (unsigned int stage = 0U; stage < k256_stage_count; ++stage) {
      const unsigned int current_slot = stage & 1U;
      const unsigned int next_stage = stage + 1U;
      if (next_stage < k256_stage_count) {
        const unsigned int next_slot = next_stage & 1U;
        issue_packed_stage(shared.a[next_slot], packed_a, m_tile_start,
                           next_stage, physical_k64_group_count);
        issue_packed_stage(shared.b[next_slot], packed_b, n_tile_start,
                           next_stage, physical_k64_group_count);
      }

      accumulate_k256_stage(shared.a[current_slot],
                            shared.b[current_slot], warp_m64, warp_n32,
                            accumulators);
      if (next_stage < k256_stage_count) {
        cp_async_wait_all();
        __syncthreads();
      }
    }

    store_accumulators(accumulators, shared.scales,
                       logical_token_count, m_tile_start, n_tile_start,
                       warp_m64, warp_n32, output_bf16,
                       output_row_stride_elements);
    __syncthreads();
  }
}

namespace {

[[nodiscard]] int validate_target(
    cudaDeviceProp* const output_properties = nullptr) noexcept {
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
  if (properties.major != kSm87A4W4RequiredComputeMajor ||
      properties.minor != kSm87A4W4RequiredComputeMinor ||
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount) ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4DownFactorizedDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_factorized_lane_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4DownFactorizedDynamicSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_factorized_lane_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

struct ResourceCache final {
  Sm87A4W4DownFactorizedLaneResources resources{};
  int status{static_cast<int>(cudaErrorUnknown)};
};

[[nodiscard]] ResourceCache build_resource_cache() noexcept {
  ResourceCache cache{};
  cudaDeviceProp properties{};
  cache.status = validate_target(&properties);
  if (cache.status != static_cast<int>(cudaSuccess)) {
    return cache;
  }
  const cudaError_t configure_status = configure_kernel();
  if (configure_status != cudaSuccess) {
    cache.status = static_cast<int>(configure_status);
    return cache;
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_down_factorized_lane_kernel);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_down_factorized_lane_kernel,
      static_cast<int>(kThreads),
      kSm87A4W4DownFactorizedDynamicSharedBytes);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }

  cache.resources.registers_per_thread = attributes.numRegs;
  cache.resources.static_shared_bytes = attributes.sharedSizeBytes;
  cache.resources.dynamic_shared_bytes =
      kSm87A4W4DownFactorizedDynamicSharedBytes;
  cache.resources.configured_dynamic_shared_limit_bytes =
      attributes.maxDynamicSharedSizeBytes;
  cache.resources.device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  cache.resources.local_bytes = attributes.localSizeBytes;
  cache.resources.maximum_threads_per_block =
      attributes.maxThreadsPerBlock;
  cache.resources.active_blocks_per_sm = active_blocks;
  cache.resources.compute_major = properties.major;
  cache.resources.compute_minor = properties.minor;

  cache.status =
      cache.resources.registers_per_thread <= 0 ||
              cache.resources.registers_per_thread >
                  static_cast<int>(
                      kSm87A4W4DownFactorizedMaximumRegisters) ||
              cache.resources.static_shared_bytes != 0U ||
              cache.resources.dynamic_shared_bytes !=
                  kSm87A4W4DownFactorizedDynamicSharedBytes ||
              cache.resources.configured_dynamic_shared_limit_bytes <
                  kSm87A4W4DownFactorizedDynamicSharedBytes ||
              cache.resources.device_optin_shared_limit_bytes <
                  kSm87A4W4DownFactorizedDynamicSharedBytes ||
              cache.resources.local_bytes != 0U ||
              cache.resources.maximum_threads_per_block <
                  static_cast<int>(kThreads) ||
              cache.resources.active_blocks_per_sm <
                  static_cast<int>(
                      kSm87A4W4DownFactorizedCtasPerSm)
          ? static_cast<int>(cudaErrorLaunchOutOfResources)
          : static_cast<int>(cudaSuccess);
  return cache;
}

[[nodiscard]] const ResourceCache& resource_cache() noexcept {
  static const ResourceCache cache = build_resource_cache();
  return cache;
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_lane_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool production_shape,
    void* const cuda_stream) noexcept {
  const Sm87A4W4DownFactorizedLanePlan plan =
      production_shape
          ? sm87_a4w4_down_factorized_lane_plan(
                logical_token_count, launch_token_count, output_size,
                input_size)
          : sm87_a4w4_down_factorized_lane_test_plan(
                logical_token_count, launch_token_count, output_size,
                input_size);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_lane_scales_bf16, 16U) ||
      !aligned(packed_b, 16U) ||
      !aligned(b_lane_scales_bf16, 16U) ||
      !aligned(output_bf16, alignof(std::uint32_t)) ||
      output_row_stride_elements < output_size ||
      output_row_stride_elements % 2U != 0U ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      plan.logical_token_count >
          std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.k256_stages > std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_down_factorized_product_fits(
          logical_token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_down_factorized_packed_capacity_bytes(
          launch_token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_down_factorized_packed_capacity_bytes(output_size,
                                                       input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_down_factorized_scale_capacity_elements(
          launch_token_count);
  const std::size_t required_b_scales =
      sm87_a4w4_down_factorized_scale_capacity_elements(output_size);
  const std::size_t required_output_elements =
      logical_token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_down_factorized_product_fits(
          required_a_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_factorized_product_fits(
          required_b_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_factorized_product_fits(
          required_output_elements, sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_b_scale_bytes =
      required_b_scales * sizeof(std::uint16_t);
  const std::size_t required_output_bytes =
      required_output_elements * sizeof(std::uint16_t);
  if (byte_ranges_overlap(output_bf16, required_output_bytes, packed_a,
                          required_a_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          a_lane_scales_bf16,
                          required_a_scale_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes, packed_b,
                          required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          b_lane_scales_bf16,
                          required_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const ResourceCache& cache = resource_cache();
  if (cache.status != static_cast<int>(cudaSuccess)) {
    return cache.status;
  }

  const unsigned int planned_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned_ctas < maximum_launch_ctas ? planned_ctas
                                         : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_factorized_lane_kernel
      <<<launch_ctas, kThreads,
         kSm87A4W4DownFactorizedDynamicSharedBytes, stream>>>(
          packed_a, a_lane_scales_bf16, packed_b,
          b_lane_scales_bf16,
          static_cast<unsigned int>(plan.logical_token_count),
          static_cast<unsigned int>(plan.physical_k64_groups),
          static_cast<unsigned int>(plan.k256_stages),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.work_tiles), output_bf16,
          static_cast<unsigned int>(output_row_stride_elements));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_down_factorized_lane_resources_cuda(
    Sm87A4W4DownFactorizedLaneResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ResourceCache& cache = resource_cache();
  *resources = cache.resources;
  return cache.status;
}

int launch_sm87_a4w4_down_factorized_lane_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_lane_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_lane_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_lane_scales_bf16, b_scale_capacity_elements,
      logical_token_count, launch_token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4DownFactorizedPersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_down_factorized_lane_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_lane_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_lane_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_lane_scales_bf16, b_scale_capacity_elements,
      logical_token_count, launch_token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

}  // namespace q3x::kernels
