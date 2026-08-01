#include "q3x/kernels/sm87_a4w4_down_k512_bstationary.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(
        kSm87A4W4DownK512BStationaryPackedRowK64Bytes);

struct alignas(16) Sm87A4W4DownK512BStationaryStage final {
  std::uint8_t a[kSm87A4W4DownK512BStationaryK64PerStage]
                [kSm87A4W4DownK512BStationaryTileM *
                 kSm87A4W4DownK512BStationaryPackedRowK64Bytes];
  std::uint8_t b[kSm87A4W4DownK512BStationaryK64PerStage]
                [kSm87A4W4DownK512BStationaryTileN *
                 kSm87A4W4DownK512BStationaryPackedRowK64Bytes];
};

struct alignas(16) Sm87A4W4DownK512BStationaryShared final {
  Sm87A4W4DownK512BStationaryStage
      stage[kSm87A4W4DownK512BStationaryStages];
};

struct alignas(16) Sm87A4W4DownK512BStationaryFloatAccumulator final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(Sm87A4W4DownK512BStationaryStage) ==
              kSm87A4W4DownK512BStationaryStageBytes);
static_assert(sizeof(Sm87A4W4DownK512BStationaryShared) ==
              kSm87A4W4DownK512BStationaryDynamicSharedBytes);
static_assert(sizeof(Sm87A4W4DownK512BStationaryFloatAccumulator) == 16U);

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

__device__ __forceinline__ void cp_async_16(
    void* const destination, const void* const source) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;"
               :
               : "r"(shared_address), "l"(source)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void cp_async_commit() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" : : : "memory");
#else
  asm volatile("trap;");
#endif
}

template <unsigned int Remaining>
__device__ __forceinline__ void cp_async_wait() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(Remaining)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

// One K256 stage has 256 A vectors and 256 B vectors per K64 plane.  Every
// thread issues one aligned 16-byte cp.async for each of four K64 planes,
// keeping the canonical outer64 layout intact across M128 and N128.
__device__ __forceinline__ void issue_k256_stage(
    Sm87A4W4DownK512BStationaryStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4DownK512BStationaryTileM * kPackedK64Bytes / 16U);
  constexpr unsigned int kBVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4DownK512BStationaryTileN * kPackedK64Bytes / 16U);
  constexpr unsigned int kVectorsPerPlane =
      kAVectorsPerPlane + kBVectorsPerPlane;
  static_assert(kAVectorsPerPlane == 256U);
  static_assert(kBVectorsPerPlane == 256U);

#pragma unroll 1
  for (unsigned int plane = 0U;
       plane < kSm87A4W4DownK512BStationaryK64PerStage; ++plane) {
    const unsigned int physical_k64_group =
        physical_k256_stage *
            static_cast<unsigned int>(
                kSm87A4W4DownK512BStationaryK64PerStage) +
        plane;
#pragma unroll
    for (unsigned int vector = threadIdx.x; vector < kVectorsPerPlane;
         vector += kSm87A4W4DownK512BStationaryThreads) {
      const bool is_a = vector < kAVectorsPerPlane;
      const unsigned int local_vector =
          is_a ? vector : vector - kAVectorsPerPlane;
      const unsigned int row = local_vector / 2U;
      const unsigned int byte = 16U * (local_vector % 2U);
      const std::size_t shared_offset =
          sm87_a4w4_swizzled_k64_byte_offset(row, byte);
      const std::size_t outer =
          static_cast<std::size_t>(is_a ? m_tile_start : n_tile_start) +
          row;
      const std::uint8_t* const source =
          (is_a ? packed_a : packed_b) +
          sm87_a4w4_down_k512_packed_offset(
              outer, physical_k64_group, byte,
              physical_k64_group_count);
      std::uint8_t* const destination =
          (is_a ? stage.a[plane] : stage.b[plane]) + shared_offset;
      cp_async_16(destination, source);
    }
  }
  cp_async_commit();
}

__device__ __forceinline__ void accumulate_k512_group(
    const Sm87A4W4DownK512BStationaryStage& first,
    const Sm87A4W4DownK512BStationaryStage& second,
    Sm87A4W4DownK512BStationaryFloatAccumulator
        (&accumulators)[kSm87A4W4DownK512BStationaryWarpTileN / 8U],
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp % kSm87A4W4DownK512BStationaryWarpRows;
  const unsigned int warp_n =
      warp / kSm87A4W4DownK512BStationaryWarpRows;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4DownK512BStationaryWarpTileM;
  const unsigned int local_n_start =
      warp_n * kSm87A4W4DownK512BStationaryWarpTileN;
  const unsigned int row0 = lane / 4U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane % 4U);
  const unsigned int column1 = column0 + 1U;

  const float a0 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
          static_cast<std::size_t>(m_tile_start + local_m_start +
                                   row0),
          k512_group, k512_group_count)]);
  const float a1 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
          static_cast<std::size_t>(m_tile_start + local_m_start +
                                   row1),
          k512_group, k512_group_count)]);

#pragma unroll
  for (unsigned int fragment_n = 0U;
       fragment_n < kSm87A4W4DownK512BStationaryWarpTileN / 8U;
       ++fragment_n) {
    const unsigned int fragment_n_start =
        local_n_start + fragment_n * 8U;
    Sm87A4W4Accumulator partial{};
#pragma unroll 1
    for (unsigned int half = 0U;
         half < kSm87A4W4DownK512BStationaryStagesPerScale; ++half) {
      const Sm87A4W4DownK512BStationaryStage& stage =
          half == 0U ? first : second;
#pragma unroll 1
      for (unsigned int plane = 0U;
           plane < kSm87A4W4DownK512BStationaryK64PerStage; ++plane) {
        const Sm87A4W4AFragment a_fragment =
            sm87_a4w4_load_a_fragment_swizzled_shared(
                stage.a[plane] + local_m_start * kPackedK64Bytes, lane);
        const Sm87A4W4BFragment b_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.b[plane] + fragment_n_start * kPackedK64Bytes,
                lane);
        sm87_a4w4_mma_m16n8k64(partial, a_fragment, b_fragment);
      }
    }
    const float b0 = decode_bf16(
        b_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
            static_cast<std::size_t>(n_tile_start + fragment_n_start +
                                     column0),
            k512_group, k512_group_count)]);
    const float b1 = decode_bf16(
        b_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
            static_cast<std::size_t>(n_tile_start + fragment_n_start +
                                     column1),
            k512_group, k512_group_count)]);
    Sm87A4W4DownK512BStationaryFloatAccumulator& output =
        accumulators[fragment_n];
    output.x0 = __fmaf_rn(static_cast<float>(partial.x0),
                          __fmul_rn(a0, b0), output.x0);
    output.x1 = __fmaf_rn(static_cast<float>(partial.x1),
                          __fmul_rn(a0, b1), output.x1);
    output.x2 = __fmaf_rn(static_cast<float>(partial.x2),
                          __fmul_rn(a1, b0), output.x2);
    output.x3 = __fmaf_rn(static_cast<float>(partial.x3),
                          __fmul_rn(a1, b1), output.x3);
  }
}

__device__ __forceinline__ void store_output(
    const Sm87A4W4DownK512BStationaryFloatAccumulator
        (&accumulators)[kSm87A4W4DownK512BStationaryWarpTileN / 8U],
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp % kSm87A4W4DownK512BStationaryWarpRows;
  const unsigned int warp_n =
      warp / kSm87A4W4DownK512BStationaryWarpRows;
  const unsigned int row0 = lane / 4U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane % 4U);
  const unsigned int column1 = column0 + 1U;

#pragma unroll
  for (unsigned int fragment_n = 0U;
       fragment_n < kSm87A4W4DownK512BStationaryWarpTileN / 8U;
       ++fragment_n) {
    const Sm87A4W4DownK512BStationaryFloatAccumulator& values =
        accumulators[fragment_n];
    const unsigned int base_m =
        m_tile_start +
        warp_m * kSm87A4W4DownK512BStationaryWarpTileM;
    const unsigned int base_n =
        n_tile_start +
        warp_n * kSm87A4W4DownK512BStationaryWarpTileN +
        fragment_n * 8U;
    output_bf16[static_cast<std::size_t>(base_m + row0) *
                    output_row_stride_elements +
                base_n + column0] = encode_bf16(values.x0);
    output_bf16[static_cast<std::size_t>(base_m + row0) *
                    output_row_stride_elements +
                base_n + column1] = encode_bf16(values.x1);
    output_bf16[static_cast<std::size_t>(base_m + row1) *
                    output_row_stride_elements +
                base_n + column0] = encode_bf16(values.x2);
    output_bf16[static_cast<std::size_t>(base_m + row1) *
                    output_row_stride_elements +
                base_n + column1] = encode_bf16(values.x3);
  }
}

__device__ __forceinline__ void compute_tile(
    Sm87A4W4DownK512BStationaryShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  Sm87A4W4DownK512BStationaryFloatAccumulator
      accumulators[kSm87A4W4DownK512BStationaryWarpTileN / 8U]{};
  const unsigned int physical_stage_count =
      static_cast<unsigned int>(
          kSm87A4W4DownK512BStationaryStagesPerScale) *
      k512_group_count;
  const unsigned int initial_stage_count =
      physical_stage_count < kSm87A4W4DownK512BStationaryStages
          ? physical_stage_count
          : static_cast<unsigned int>(
                kSm87A4W4DownK512BStationaryStages);
  for (unsigned int stage = 0U; stage < initial_stage_count; ++stage) {
    issue_k256_stage(shared.stage[stage], packed_a, packed_b,
                     m_tile_start, n_tile_start, stage,
                     physical_k64_group_count);
  }

  // At the head of group g, slots (2g)%3 and (2g+1)%3 are its complete K512
  // pair; slot (2g+2)%3 is the next group's first half.  wait_group 1 makes
  // the pair safe, then the consumed slots are replaced by phases 2g+3 and
  // 2g+4.  This preserves a three-stage rolling window with two CTA barriers
  // per K512 group instead of synchronizing on every K128 quarter.
  for (unsigned int group = 0U; group < k512_group_count; ++group) {
    if (group + 1U < k512_group_count) {
      cp_async_wait<1U>();
    } else {
      cp_async_wait<0U>();
    }
    __syncthreads();
    const unsigned int first_phase =
        group * kSm87A4W4DownK512BStationaryStagesPerScale;
    accumulate_k512_group(
        shared.stage[first_phase %
                     kSm87A4W4DownK512BStationaryStages],
        shared.stage[(first_phase + 1U) %
                     kSm87A4W4DownK512BStationaryStages],
        accumulators, a_k512_scales_bf16, b_k512_scales_bf16,
        m_tile_start, n_tile_start, group, k512_group_count);
    __syncthreads();
    const unsigned int future0 =
        first_phase + kSm87A4W4DownK512BStationaryStages;
    const unsigned int future1 = future0 + 1U;
    if (future0 < physical_stage_count) {
      issue_k256_stage(
          shared.stage[future0 %
                       kSm87A4W4DownK512BStationaryStages],
          packed_a, packed_b, m_tile_start, n_tile_start, future0,
          physical_k64_group_count);
    }
    if (future1 < physical_stage_count) {
      issue_k256_stage(
          shared.stage[future1 %
                       kSm87A4W4DownK512BStationaryStages],
          packed_a, packed_b, m_tile_start, n_tile_start, future1,
          physical_k64_group_count);
    }
  }

  store_output(accumulators, m_tile_start, n_tile_start, output_bf16,
               output_row_stride_elements);
  __syncthreads();
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK512BStationaryThreads,
                      kSm87A4W4DownK512BStationaryCtasPerSm)
void q3x_sm87_a4w4_down_k512_bstationary_m128n128k512_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int n_tile_count) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared =
      *reinterpret_cast<Sm87A4W4DownK512BStationaryShared*>(
          dynamic_storage);

  // Exact owner groups preserve a lockstep B wave when every CTA owns the
  // same number of M tiles (all natural M<=2048 spans and M=4096).  A naive
  // owner loop is pathological for shapes such as M=2176: CTA 0 would run a
  // second complete forty-N sweep after the other fifteen CTAs retire.  For
  // an uneven owner count, retain the incumbent's balanced N-major work
  // distribution.  It may mix two adjacent B panels at a wave boundary, but
  // never turns a one-tile M remainder into a full-N critical-path tail.
  const bool exact_owner_groups =
      m_tile_count <= gridDim.x || m_tile_count % gridDim.x == 0U;
  if (exact_owner_groups) {
    for (unsigned int owner_base = 0U; owner_base < m_tile_count;
         owner_base += gridDim.x) {
      const unsigned int m_tile = owner_base + blockIdx.x;
      if (m_tile >= m_tile_count) {
        continue;
      }
      const unsigned int m_tile_start =
          m_tile * kSm87A4W4DownK512BStationaryTileM;
      for (unsigned int n_tile = 0U; n_tile < n_tile_count; ++n_tile) {
        const unsigned int n_tile_start =
            n_tile * kSm87A4W4DownK512BStationaryTileN;
        compute_tile(shared, packed_a, a_k512_scales_bf16, packed_b,
                     b_k512_scales_bf16, k512_group_count,
                     physical_k64_group_count, m_tile_start,
                     n_tile_start, output_bf16,
                     output_row_stride_elements);
      }
    }
  } else {
    const unsigned int work_tile_count = m_tile_count * n_tile_count;
    for (unsigned int work_tile = blockIdx.x;
         work_tile < work_tile_count; work_tile += gridDim.x) {
      const unsigned int n_tile = work_tile / m_tile_count;
      const unsigned int m_tile = work_tile - n_tile * m_tile_count;
      compute_tile(
          shared, packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16, k512_group_count,
          physical_k64_group_count,
          m_tile * kSm87A4W4DownK512BStationaryTileM,
          n_tile * kSm87A4W4DownK512BStationaryTileN, output_bf16,
          output_row_stride_elements);
    }
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
      properties.multiProcessorCount != kRequiredSmCount ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4DownK512BStationaryDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int admit_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_bstationary_m128n128k512_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownK512BStationaryDynamicSharedBytes)));
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool production_shape,
    void* const cuda_stream) noexcept {
  const Sm87A4W4DownK512BStationaryPlan plan =
      production_shape
          ? sm87_a4w4_down_k512_bstationary_plan(
                token_count, output_size, input_size)
          : sm87_a4w4_down_k512_bstationary_test_plan(
                token_count, output_size, input_size);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_b, 16U) ||
      !aligned(b_k512_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(output_bf16, alignof(std::uint32_t)) ||
      output_row_stride_elements < output_size ||
      output_row_stride_elements % 2U != 0U ||
      output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      plan.k512_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.n_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_down_k512_product_fits(
          token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(output_size, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(token_count, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(output_size, input_size);
  const std::size_t required_output_elements =
      token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_down_k512_product_fits(
          required_a_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(
          required_b_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(
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
                          a_k512_scales_bf16,
                          required_a_scale_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes, packed_b,
                          required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          b_k512_scales_bf16,
                          required_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87A4W4DownK512BStationaryResources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_bstationary_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const unsigned int planned_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      planned_ctas < maximum_launch_ctas ? planned_ctas
                                         : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_k512_bstationary_m128n128k512_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(
             kSm87A4W4DownK512BStationaryThreads),
         kSm87A4W4DownK512BStationaryDynamicSharedBytes, stream>>>(
          packed_a, a_k512_scales_bf16, packed_b,
          b_k512_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          output_bf16,
          static_cast<unsigned int>(output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.n_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_down_k512_bstationary_resources_cuda(
    Sm87A4W4DownK512BStationaryResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK512BStationaryResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const int admission_status = admit_dynamic_shared();
  if (admission_status != static_cast<int>(cudaSuccess)) {
    return admission_status;
  }

  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_down_k512_bstationary_m128n128k512_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_bstationary_m128n128k512_kernel,
      static_cast<int>(kSm87A4W4DownK512BStationaryThreads),
      kSm87A4W4DownK512BStationaryDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512BStationaryDynamicSharedBytes;
  resources->configured_dynamic_shared_limit_bytes =
      attributes.maxDynamicSharedSizeBytes;
  resources->device_optin_shared_limit_bytes =
      properties.sharedMemPerBlockOptin;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4DownK512BStationaryMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512BStationaryDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512BStationaryDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK512BStationaryDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4DownK512BStationaryThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4DownK512BStationaryCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k512_bstationary_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements, token_count,
      output_size, input_size, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4DownK512BStationaryPersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_down_k512_bstationary_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_b,
    const std::size_t packed_b_capacity_bytes,
    const std::uint16_t* const b_k512_scales_bf16,
    const std::size_t b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t output_size,
    const std::size_t input_size,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements, token_count,
      output_size, input_size, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

}  // namespace q3x::kernels
