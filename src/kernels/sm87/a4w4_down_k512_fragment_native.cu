#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native.h"

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
        kSm87A4W4DownK512FragmentPackedRowK64Bytes);

struct alignas(16) Sm87A4W4DownK512FragmentAStage final {
  std::uint8_t a[kSm87A4W4DownK512FragmentK64PerStage]
                [kSm87A4W4DownK512FragmentTileM *
                 kSm87A4W4DownK512FragmentPackedRowK64Bytes];
};

struct alignas(16) Sm87A4W4DownK512FragmentScaleSlot final {
  std::uint16_t a[kSm87A4W4DownK512FragmentTileM];
  std::uint16_t b[kSm87A4W4DownK512FragmentTileN];
};

struct alignas(16) Sm87A4W4DownK512FragmentShared final {
  Sm87A4W4DownK512FragmentAStage
      stage[kSm87A4W4DownK512FragmentStages];
  Sm87A4W4DownK512FragmentScaleSlot scale;
};

struct alignas(16) Sm87A4W4DownK512FragmentBVector final {
  std::uint32_t n0_x0{};
  std::uint32_t n0_x1{};
  std::uint32_t n1_x0{};
  std::uint32_t n1_x1{};
};

struct alignas(16) Sm87A4W4DownK512FragmentFloatAccumulator final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(Sm87A4W4DownK512FragmentAStage) ==
              kSm87A4W4DownK512FragmentStageABytes);
static_assert(sizeof(Sm87A4W4DownK512FragmentScaleSlot) ==
              kSm87A4W4DownK512FragmentScaleSlotBytes);
static_assert(sizeof(Sm87A4W4DownK512FragmentShared) ==
              kSm87A4W4DownK512FragmentDynamicSharedBytes);
static_assert(sizeof(Sm87A4W4DownK512FragmentBVector) == 16U);
static_assert(sizeof(Sm87A4W4DownK512FragmentFloatAccumulator) == 16U);

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

// One K256 A stage is 8 KiB. Every thread issues two aligned 16-byte copies;
// A retains the authenticated outer64 payload and is swizzled only in shared.
__device__ __forceinline__ void issue_a_k256_stage(
    Sm87A4W4DownK512FragmentAStage& stage,
    const std::uint8_t* const packed_a,
    const unsigned int m_tile_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4DownK512FragmentTileM * kPackedK64Bytes / 16U);
  constexpr unsigned int kVectorsPerStage =
      kVectorsPerPlane *
      static_cast<unsigned int>(kSm87A4W4DownK512FragmentK64PerStage);
  static_assert(kVectorsPerPlane == 128U);
  static_assert(kVectorsPerStage ==
                2U * kSm87A4W4DownK512FragmentThreads);

#pragma unroll
  for (unsigned int vector = threadIdx.x; vector < kVectorsPerStage;
       vector += kSm87A4W4DownK512FragmentThreads) {
    const unsigned int plane = vector / kVectorsPerPlane;
    const unsigned int vector_in_plane = vector - plane * kVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int byte = 16U * (vector_in_plane % 2U);
    const unsigned int physical_k64_group =
        physical_k256_stage *
            static_cast<unsigned int>(
                kSm87A4W4DownK512FragmentK64PerStage) +
        plane;
    cp_async_16(
        stage.a[plane] +
            sm87_a4w4_swizzled_k64_byte_offset(row, byte),
        packed_a + sm87_a4w4_down_k512_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64_group, byte,
                       physical_k64_group_count));
  }
  cp_async_commit();
}

__device__ __forceinline__ void issue_k512_scales(
    Sm87A4W4DownK512FragmentScaleSlot& slot,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n128_panel,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  if (threadIdx.x < kSm87A4W4DownK512FragmentTileM) {
    slot.a[threadIdx.x] =
        a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
            static_cast<std::size_t>(m_tile_start + threadIdx.x),
            k512_group, k512_group_count)];
  } else if (threadIdx.x <
             kSm87A4W4DownK512FragmentTileM +
                 kSm87A4W4DownK512FragmentTileN) {
    const unsigned int n =
        threadIdx.x - kSm87A4W4DownK512FragmentTileM;
    slot.b[n] =
        b_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
            static_cast<std::size_t>(
                n128_panel * kSm87A4W4DownK512FragmentTileN +
                n),
            k512_group, k512_group_count)];
  }
}

// The offline layout makes all four registers consecutive and 16-byte
// aligned. Explicit vector PTX prevents scalarization and documents that B
// goes global/L1/L2 -> registers, never through shared memory.
[[nodiscard]] __device__ __forceinline__
    Sm87A4W4DownK512FragmentBVector
load_fragment_native_b(
    const std::uint8_t* const fragment_native_b,
    const unsigned int n128_panel,
    const unsigned int k512_group,
    const unsigned int k64_in_group,
    const unsigned int n16_warp,
    const unsigned int lane,
    const unsigned int k512_group_count) noexcept {
  const std::uint8_t* const pointer =
      fragment_native_b + sm87_a4w4_down_k512_fragment_b_vector_offset(
                              n128_panel, k512_group, k64_in_group,
                              n16_warp, lane, k512_group_count);
  Sm87A4W4DownK512FragmentBVector value{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.n0_x0), "=r"(value.n0_x1),
                 "=r"(value.n1_x0), "=r"(value.n1_x1)
               : "l"(pointer)
               : "memory");
#else
  asm volatile("trap;");
#endif
  return value;
}

__device__ __forceinline__ void accumulate_k512_group(
    const Sm87A4W4DownK512FragmentAStage& first,
    const Sm87A4W4DownK512FragmentAStage& second,
    const Sm87A4W4DownK512FragmentScaleSlot& scale,
    Sm87A4W4DownK512FragmentFloatAccumulator
        (&accumulators)[kSm87A4W4DownK512FragmentM16PerWarp]
                       [kSm87A4W4DownK512FragmentN8PerWarp],
    const std::uint8_t* const fragment_native_b,
    const unsigned int n128_panel,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  Sm87A4W4Accumulator
      partials[kSm87A4W4DownK512FragmentM16PerWarp]
              [kSm87A4W4DownK512FragmentN8PerWarp]{};

  // Each 128-bit B load feeds eight MMAs: two adjacent N8 fragments times
  // four M16 fragments. All eight physical K64 partials remain S32 until the
  // single authenticated K512 scale boundary below.
#pragma unroll
  for (unsigned int half = 0U;
       half < kSm87A4W4DownK512FragmentStagesPerScale; ++half) {
    const Sm87A4W4DownK512FragmentAStage& stage =
        half == 0U ? first : second;
#pragma unroll
    for (unsigned int plane = 0U;
         plane < kSm87A4W4DownK512FragmentK64PerStage; ++plane) {
      const unsigned int k64 =
          half * kSm87A4W4DownK512FragmentK64PerStage + plane;
      const Sm87A4W4DownK512FragmentBVector b =
          load_fragment_native_b(fragment_native_b, n128_panel,
                                 k512_group, k64, warp, lane,
                                 k512_group_count);
      const Sm87A4W4BFragment b0{b.n0_x0, b.n0_x1};
      const Sm87A4W4BFragment b1{b.n1_x0, b.n1_x1};
#pragma unroll
      for (unsigned int m16 = 0U;
           m16 < kSm87A4W4DownK512FragmentM16PerWarp; ++m16) {
        const Sm87A4W4AFragment a =
            sm87_a4w4_load_a_fragment_swizzled_shared(
                stage.a[plane] + m16 * kSm87A4W4MmaM * kPackedK64Bytes,
                lane);
        sm87_a4w4_mma_m16n8k64(partials[m16][0U], a, b0);
        sm87_a4w4_mma_m16n8k64(partials[m16][1U], a, b1);
      }
    }
  }

  const unsigned int row0 = lane / 4U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane % 4U);
  const unsigned int column1 = column0 + 1U;
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4DownK512FragmentM16PerWarp; ++m16) {
    const unsigned int local_m = m16 * kSm87A4W4MmaM;
    const float a0 = decode_bf16(scale.a[local_m + row0]);
    const float a1 = decode_bf16(scale.a[local_m + row1]);
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4DownK512FragmentN8PerWarp; ++n8) {
      const unsigned int local_n =
          warp * kSm87A4W4DownK512FragmentWarpTileN +
          n8 * kSm87A4W4MmaN;
      const float b0 = decode_bf16(scale.b[local_n + column0]);
      const float b1 = decode_bf16(scale.b[local_n + column1]);
      const Sm87A4W4Accumulator& partial = partials[m16][n8];
      Sm87A4W4DownK512FragmentFloatAccumulator& output =
          accumulators[m16][n8];
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
}

__device__ __forceinline__ void store_output(
    const Sm87A4W4DownK512FragmentFloatAccumulator
        (&accumulators)[kSm87A4W4DownK512FragmentM16PerWarp]
                       [kSm87A4W4DownK512FragmentN8PerWarp],
    const unsigned int m_tile_start,
    const unsigned int n128_panel,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int row0 = lane / 4U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane % 4U);
  const unsigned int column1 = column0 + 1U;
  const unsigned int n_tile_start =
      n128_panel * kSm87A4W4DownK512FragmentTileN;

#pragma unroll
  for (unsigned int m16 = 0U;
       m16 < kSm87A4W4DownK512FragmentM16PerWarp; ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 < kSm87A4W4DownK512FragmentN8PerWarp; ++n8) {
      const Sm87A4W4DownK512FragmentFloatAccumulator& values =
          accumulators[m16][n8];
      const unsigned int base_m =
          m_tile_start + m16 * kSm87A4W4MmaM;
      const unsigned int base_n =
          n_tile_start + warp * kSm87A4W4DownK512FragmentWarpTileN +
          n8 * kSm87A4W4MmaN;
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
}

__device__ __forceinline__ void compute_tile(
    Sm87A4W4DownK512FragmentShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const fragment_native_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int m_tile_start,
    const unsigned int n128_panel,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  Sm87A4W4DownK512FragmentFloatAccumulator
      accumulators[kSm87A4W4DownK512FragmentM16PerWarp]
                  [kSm87A4W4DownK512FragmentN8PerWarp]{};
  const unsigned int physical_stage_count =
      static_cast<unsigned int>(
          kSm87A4W4DownK512FragmentStagesPerScale) *
      k512_group_count;
  const unsigned int initial_stage_count =
      physical_stage_count < kSm87A4W4DownK512FragmentStages
          ? physical_stage_count
          : static_cast<unsigned int>(
                kSm87A4W4DownK512FragmentStages);
  for (unsigned int stage = 0U; stage < initial_stage_count; ++stage) {
    issue_a_k256_stage(shared.stage[stage], packed_a, m_tile_start, stage,
                       physical_k64_group_count);
  }

  for (unsigned int group = 0U; group < k512_group_count; ++group) {
    issue_k512_scales(shared.scale, a_k512_scales_bf16,
                      b_k512_scales_bf16, m_tile_start, n128_panel,
                      group, k512_group_count);
    if (group + 1U < k512_group_count) {
      cp_async_wait<1U>();
    } else {
      cp_async_wait<0U>();
    }
    __syncthreads();
    const unsigned int first_phase =
        group * kSm87A4W4DownK512FragmentStagesPerScale;
    accumulate_k512_group(
        shared.stage[first_phase % kSm87A4W4DownK512FragmentStages],
        shared.stage[(first_phase + 1U) %
                     kSm87A4W4DownK512FragmentStages],
        shared.scale, accumulators, fragment_native_b, n128_panel, group,
        k512_group_count);
    __syncthreads();
    const unsigned int future0 =
        first_phase + kSm87A4W4DownK512FragmentStages;
    const unsigned int future1 = future0 + 1U;
    if (future0 < physical_stage_count) {
      issue_a_k256_stage(
          shared.stage[future0 % kSm87A4W4DownK512FragmentStages],
          packed_a, m_tile_start, future0, physical_k64_group_count);
    }
    if (future1 < physical_stage_count) {
      issue_a_k256_stage(
          shared.stage[future1 % kSm87A4W4DownK512FragmentStages],
          packed_a, m_tile_start, future1, physical_k64_group_count);
    }
  }

  store_output(accumulators, m_tile_start, n128_panel, output_bf16,
               output_row_stride_elements);
  __syncthreads();
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4DownK512FragmentThreads,
                      kSm87A4W4DownK512FragmentCtasPerSm)
void q3x_sm87_a4w4_down_k512_fragment_native_m64n128k512_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const fragment_native_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) {
  extern __shared__ __align__(16) unsigned char dynamic_storage[];
  auto& shared =
      *reinterpret_cast<Sm87A4W4DownK512FragmentShared*>(
          dynamic_storage);

  // A two-dimensional grid is N-major by construction: for each blockIdx.y
  // panel, adjacent blockIdx.x values are paired M64 owners of the same B.
  // The cell has no persistent M/N loop state across its register-heavy body.
  const unsigned int m_tile_start =
      blockIdx.x * kSm87A4W4DownK512FragmentTileM;
  const unsigned int n128_panel = blockIdx.y;
  compute_tile(shared, packed_a, a_k512_scales_bf16,
               fragment_native_b, b_k512_scales_bf16,
               k512_group_count, physical_k64_group_count,
               m_tile_start, n128_panel, output_bf16,
               output_row_stride_elements);
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
          kSm87A4W4DownK512FragmentDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int admit_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_fragment_native_m64n128k512_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87A4W4DownK512FragmentDynamicSharedBytes)));
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const fragment_native_b,
    const std::size_t fragment_native_b_capacity_bytes,
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
  const Sm87A4W4DownK512FragmentPlan plan =
      production_shape
          ? sm87_a4w4_down_k512_fragment_native_plan(
                token_count, output_size, input_size)
          : sm87_a4w4_down_k512_fragment_native_test_plan(
                token_count, output_size, input_size);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(fragment_native_b, 16U) ||
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
      sm87_a4w4_down_k512_fragment_b_capacity_bytes(output_size,
                                                     input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(token_count, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_down_k512_scale_capacity_elements(output_size, input_size);
  const std::size_t required_output_elements =
      token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      fragment_native_b_capacity_bytes < required_b_bytes ||
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
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          fragment_native_b, required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          b_k512_scales_bf16,
                          required_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87A4W4DownK512FragmentResources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_fragment_native_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  if (plan.launch_ctas > maximum_launch_ctas) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_k512_fragment_native_m64n128k512_kernel
      <<<dim3(static_cast<unsigned int>(plan.m_tiles),
              static_cast<unsigned int>(plan.n_tiles), 1U),
         static_cast<unsigned int>(kSm87A4W4DownK512FragmentThreads),
         kSm87A4W4DownK512FragmentDynamicSharedBytes, stream>>>(
          packed_a, a_k512_scales_bf16, fragment_native_b,
          b_k512_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups), output_bf16,
          static_cast<unsigned int>(output_row_stride_elements));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_down_k512_fragment_native_resources_cuda(
    Sm87A4W4DownK512FragmentResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK512FragmentResources{};
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
      q3x_sm87_a4w4_down_k512_fragment_native_m64n128k512_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_fragment_native_m64n128k512_kernel,
      static_cast<int>(kSm87A4W4DownK512FragmentThreads),
      kSm87A4W4DownK512FragmentDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512FragmentDynamicSharedBytes;
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
              kSm87A4W4DownK512FragmentMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512FragmentDynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512FragmentDynamicSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK512FragmentDynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4DownK512FragmentThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4DownK512FragmentCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k512_fragment_native_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const fragment_native_b,
    const std::size_t fragment_native_b_capacity_bytes,
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
      a_scale_capacity_elements, fragment_native_b,
      fragment_native_b_capacity_bytes, b_k512_scales_bf16,
      b_scale_capacity_elements, token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4DownK512FragmentMaximumLaunchCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_down_k512_fragment_native_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const fragment_native_b,
    const std::size_t fragment_native_b_capacity_bytes,
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
      a_scale_capacity_elements, fragment_native_b,
      fragment_native_b_capacity_bytes, b_k512_scales_bf16,
      b_scale_capacity_elements, token_count, output_size, input_size,
      output_bf16, output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

}  // namespace q3x::kernels
