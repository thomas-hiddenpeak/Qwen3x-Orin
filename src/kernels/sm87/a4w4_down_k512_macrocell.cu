#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(kSm87A4W4DownK512PackedRowK64Bytes);

struct alignas(16) Sm87A4W4DownK512Stage final {
  std::uint8_t a[kSm87A4W4DownK512K64PerStage]
                [kSm87A4W4DownK512TileM *
                 kSm87A4W4DownK512PackedRowK64Bytes];
  std::uint8_t b[kSm87A4W4DownK512K64PerStage]
                [kSm87A4W4DownK512TileN *
                 kSm87A4W4DownK512PackedRowK64Bytes];
};

struct alignas(16) Sm87A4W4DownK512Shared final {
  Sm87A4W4DownK512Stage stage[kSm87A4W4DownK512Stages];
};

struct alignas(16) Sm87A4W4DownK512FloatAccumulator final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(Sm87A4W4DownK512Stage) ==
              kSm87A4W4DownK512StageBytes);
static_assert(sizeof(Sm87A4W4DownK512Shared) ==
              kSm87A4W4DownK512DynamicSharedBytes);
static_assert(sizeof(Sm87A4W4DownK512FloatAccumulator) == 16U);

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

// One K256 stage is exactly 32 KiB.  Each thread issues four 16-byte A and
// four 16-byte B copies, all using .cg so the 128-KiB ring does not evict the
// synchronized B wave from L1 merely to retain data already resident in
// shared memory.
__device__ __forceinline__ void issue_k256_stage(
    Sm87A4W4DownK512Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4DownK512TileM * kPackedK64Bytes / 16U);
  static_assert(kSm87A4W4DownK512TileM ==
                kSm87A4W4DownK512TileN);
  static_assert(kVectorsPerPlane == kSm87A4W4DownK512Threads);

#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4DownK512K64PerStage; ++plane) {
    const unsigned int row = threadIdx.x / 2U;
    const unsigned int row_vector = threadIdx.x % 2U;
    const unsigned int physical_k64_group =
        physical_k256_stage *
            static_cast<unsigned int>(kSm87A4W4DownK512K64PerStage) +
        plane;
    const std::size_t shared_offset =
        sm87_a4w4_swizzled_k64_byte_offset(row, 16U * row_vector);
    cp_async_16(
        stage.a[plane] + shared_offset,
        packed_a + sm87_a4w4_down_k512_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64_group, 16U * row_vector,
                       physical_k64_group_count));
    cp_async_16(
        stage.b[plane] + shared_offset,
        packed_b + sm87_a4w4_down_k512_packed_offset(
                       static_cast<std::size_t>(n_tile_start) + row,
                       physical_k64_group, 16U * row_vector,
                       physical_k64_group_count));
  }
  cp_async_commit();
}

__device__ __forceinline__ void accumulate_k512_group(
    const Sm87A4W4DownK512Stage& first,
    const Sm87A4W4DownK512Stage& second,
    Sm87A4W4DownK512FloatAccumulator
        (&accumulators)[kSm87A4W4DownK512WarpTileN / 8U]
                       [kSm87A4W4DownK512WarpTileM / 16U],
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m = warp % kSm87A4W4DownK512WarpRows;
  const unsigned int warp_n = warp / kSm87A4W4DownK512WarpRows;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4DownK512WarpTileM;
  const unsigned int local_n_start =
      warp_n * kSm87A4W4DownK512WarpTileN;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);

  const float a00 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
          static_cast<std::size_t>(m_tile_start + local_m_start +
                                   coordinate0.m),
          k512_group, k512_group_count)]);
  const float a01 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
          static_cast<std::size_t>(m_tile_start + local_m_start +
                                   coordinate2.m),
          k512_group, k512_group_count)]);
  const float a10 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
          static_cast<std::size_t>(m_tile_start + local_m_start + 16U +
                                   coordinate0.m),
          k512_group, k512_group_count)]);
  const float a11 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
          static_cast<std::size_t>(m_tile_start + local_m_start + 16U +
                                   coordinate2.m),
          k512_group, k512_group_count)]);

#pragma unroll
  for (unsigned int fragment_n = 0U;
       fragment_n < kSm87A4W4DownK512WarpTileN / 8U; ++fragment_n) {
    const unsigned int fragment_n_start =
        local_n_start + fragment_n * 8U;
    Sm87A4W4Accumulator partial0{};
    Sm87A4W4Accumulator partial1{};

#pragma unroll
    for (unsigned int half = 0U;
         half < kSm87A4W4DownK512StagesPerScale; ++half) {
      const Sm87A4W4DownK512Stage& stage = half == 0U ? first : second;
#pragma unroll
      for (unsigned int plane = 0U;
           plane < kSm87A4W4DownK512K64PerStage; ++plane) {
        const Sm87A4W4AFragment a_fragment0 =
            sm87_a4w4_load_a_fragment_swizzled_shared(
                stage.a[plane] + local_m_start * kPackedK64Bytes, lane);
        const Sm87A4W4AFragment a_fragment1 =
            sm87_a4w4_load_a_fragment_swizzled_shared(
                stage.a[plane] +
                    (local_m_start + 16U) * kPackedK64Bytes,
                lane);
        const Sm87A4W4BFragment b_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.b[plane] + fragment_n_start * kPackedK64Bytes,
                lane);
        sm87_a4w4_mma_m16n8k64(partial0, a_fragment0, b_fragment);
        sm87_a4w4_mma_m16n8k64(partial1, a_fragment1, b_fragment);
      }
    }

    const unsigned int global_n0 =
        n_tile_start + fragment_n_start + coordinate0.n;
    const unsigned int global_n1 =
        n_tile_start + fragment_n_start + coordinate1.n;
    const float b0 = decode_bf16(
        b_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
            global_n0, k512_group, k512_group_count)]);
    const float b1 = decode_bf16(
        b_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
            global_n1, k512_group, k512_group_count)]);

    Sm87A4W4DownK512FloatAccumulator& output0 =
        accumulators[fragment_n][0U];
    Sm87A4W4DownK512FloatAccumulator& output1 =
        accumulators[fragment_n][1U];
    output0.x0 = __fmaf_rn(static_cast<float>(partial0.x0),
                            __fmul_rn(a00, b0), output0.x0);
    output0.x1 = __fmaf_rn(static_cast<float>(partial0.x1),
                            __fmul_rn(a00, b1), output0.x1);
    output0.x2 = __fmaf_rn(static_cast<float>(partial0.x2),
                            __fmul_rn(a01, b0), output0.x2);
    output0.x3 = __fmaf_rn(static_cast<float>(partial0.x3),
                            __fmul_rn(a01, b1), output0.x3);
    output1.x0 = __fmaf_rn(static_cast<float>(partial1.x0),
                            __fmul_rn(a10, b0), output1.x0);
    output1.x1 = __fmaf_rn(static_cast<float>(partial1.x1),
                            __fmul_rn(a10, b1), output1.x1);
    output1.x2 = __fmaf_rn(static_cast<float>(partial1.x2),
                            __fmul_rn(a11, b0), output1.x2);
    output1.x3 = __fmaf_rn(static_cast<float>(partial1.x3),
                            __fmul_rn(a11, b1), output1.x3);
  }
}

__device__ __forceinline__ void store_output(
    const Sm87A4W4DownK512FloatAccumulator
        (&accumulators)[kSm87A4W4DownK512WarpTileN / 8U]
                       [kSm87A4W4DownK512WarpTileM / 16U],
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m = warp % kSm87A4W4DownK512WarpRows;
  const unsigned int warp_n = warp / kSm87A4W4DownK512WarpRows;

#pragma unroll
  for (unsigned int fragment_n = 0U;
       fragment_n < kSm87A4W4DownK512WarpTileN / 8U; ++fragment_n) {
#pragma unroll
    for (unsigned int m_partial = 0U; m_partial < 2U; ++m_partial) {
      const Sm87A4W4DownK512FloatAccumulator& values =
          accumulators[fragment_n][m_partial];
      const Sm87A4W4AccumulatorCoordinate coordinate0 =
          sm87_a4w4_accumulator_coordinate(lane, 0U);
      const Sm87A4W4AccumulatorCoordinate coordinate1 =
          sm87_a4w4_accumulator_coordinate(lane, 1U);
      const Sm87A4W4AccumulatorCoordinate coordinate2 =
          sm87_a4w4_accumulator_coordinate(lane, 2U);
      const Sm87A4W4AccumulatorCoordinate coordinate3 =
          sm87_a4w4_accumulator_coordinate(lane, 3U);
      const unsigned int base_m =
          m_tile_start + warp_m * kSm87A4W4DownK512WarpTileM +
          m_partial * 16U;
      const unsigned int base_n =
          n_tile_start + warp_n * kSm87A4W4DownK512WarpTileN +
          fragment_n * 8U;
      output_bf16[static_cast<std::size_t>(base_m + coordinate0.m) *
                      output_row_stride_elements +
                  base_n + coordinate0.n] = encode_bf16(values.x0);
      output_bf16[static_cast<std::size_t>(base_m + coordinate1.m) *
                      output_row_stride_elements +
                  base_n + coordinate1.n] = encode_bf16(values.x1);
      output_bf16[static_cast<std::size_t>(base_m + coordinate2.m) *
                      output_row_stride_elements +
                  base_n + coordinate2.n] = encode_bf16(values.x2);
      output_bf16[static_cast<std::size_t>(base_m + coordinate3.m) *
                      output_row_stride_elements +
                  base_n + coordinate3.n] = encode_bf16(values.x3);
    }
  }
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK512Threads,
                      kSm87A4W4DownK512CtasPerSm)
void q3x_sm87_a4w4_down_k512_macrocell_m128n128k512_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int work_tile_count) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared =
      *reinterpret_cast<Sm87A4W4DownK512Shared*>(dynamic_storage);

  // N-major indexing plus a 16-CTA stride is the B-wave schedule for P2048:
  // block b retains M tile b while all blocks advance through the same N tile.
  for (unsigned int work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const unsigned int n_tile = work_tile / m_tile_count;
    const unsigned int m_tile = work_tile - n_tile * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4DownK512TileM;
    const unsigned int n_tile_start =
        n_tile * kSm87A4W4DownK512TileN;
    Sm87A4W4DownK512FloatAccumulator
        accumulators[kSm87A4W4DownK512WarpTileN / 8U]
                    [kSm87A4W4DownK512WarpTileM / 16U]{};

    const unsigned int physical_stage_count = 2U * k512_group_count;
    const unsigned int initial_stages =
        physical_stage_count < kSm87A4W4DownK512Stages
            ? physical_stage_count
            : static_cast<unsigned int>(kSm87A4W4DownK512Stages);
    for (unsigned int phase = 0U; phase < initial_stages; ++phase) {
      issue_k256_stage(
          shared.stage[phase], packed_a, packed_b, m_tile_start,
          n_tile_start, phase, physical_k64_group_count);
    }

    for (unsigned int group = 0U; group < k512_group_count; ++group) {
      // Four groups are normally outstanding.  Waiting for <=2 leaves the
      // next logical K512 pair in flight while making the current pair safe.
      if (group + 1U < k512_group_count) {
        cp_async_wait<2U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();

      const unsigned int first_phase = 2U * group;
      accumulate_k512_group(
          shared.stage[first_phase % kSm87A4W4DownK512Stages],
          shared.stage[(first_phase + 1U) % kSm87A4W4DownK512Stages],
          accumulators, a_k512_scales_bf16, b_k512_scales_bf16,
          m_tile_start, n_tile_start, group, k512_group_count);

      // All warps release both consumed stages before either ring slot is
      // overwritten by the group+2 lookahead pair.
      __syncthreads();
      const unsigned int future0 = first_phase +
                                   kSm87A4W4DownK512Stages;
      const unsigned int future1 = future0 + 1U;
      if (future0 < physical_stage_count) {
        issue_k256_stage(
            shared.stage[future0 % kSm87A4W4DownK512Stages], packed_a,
            packed_b, m_tile_start, n_tile_start, future0,
            physical_k64_group_count);
      }
      if (future1 < physical_stage_count) {
        issue_k256_stage(
            shared.stage[future1 % kSm87A4W4DownK512Stages], packed_a,
            packed_b, m_tile_start, n_tile_start, future1,
            physical_k64_group_count);
      }
    }

    store_output(accumulators, m_tile_start, n_tile_start, output_bf16,
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
      properties.multiProcessorCount != kRequiredSmCount ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4DownK512DynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int admit_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_macrocell_m128n128k512_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4DownK512DynamicSharedBytes)));
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
  const Sm87A4W4DownK512Plan plan =
      production_shape
          ? sm87_a4w4_down_k512_plan(token_count, output_size, input_size)
          : sm87_a4w4_down_k512_test_plan(token_count, output_size,
                                           input_size);
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
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_down_k512_product_fits(
          token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_down_k512_packed_capacity_bytes(output_size, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(token_count,
                                                   input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(output_size,
                                                   input_size);
  const std::size_t required_output_elements =
      token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements ||
      !sm87_a4w4_down_k512_product_fits(required_a_scales,
                                         sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_b_scales,
                                         sizeof(std::uint16_t)) ||
      !sm87_a4w4_down_k512_product_fits(required_output_elements,
                                         sizeof(std::uint16_t))) {
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

  Sm87A4W4DownK512Resources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_macrocell_resources_cuda(&resources);
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
  q3x_sm87_a4w4_down_k512_macrocell_m128n128k512_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(kSm87A4W4DownK512Threads),
         kSm87A4W4DownK512DynamicSharedBytes, stream>>>(
          packed_a, a_k512_scales_bf16, packed_b, b_k512_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          output_bf16,
          static_cast<unsigned int>(output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.work_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_down_k512_macrocell_resources_cuda(
    Sm87A4W4DownK512Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK512Resources{};
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
      q3x_sm87_a4w4_down_k512_macrocell_m128n128k512_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_macrocell_m128n128k512_kernel,
      static_cast<int>(kSm87A4W4DownK512Threads),
      kSm87A4W4DownK512DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512DynamicSharedBytes;
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
          static_cast<int>(kSm87A4W4DownK512MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512DynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512DynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK512DynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4DownK512Threads) ||
      resources->active_blocks_per_sm !=
          static_cast<int>(kSm87A4W4DownK512CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k512_macrocell_bf16_cuda(
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
      static_cast<unsigned int>(kSm87A4W4DownK512PersistentCtas), true,
      cuda_stream);
}

int launch_sm87_a4w4_down_k512_macrocell_test_bf16_cuda(
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
