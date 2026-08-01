#include "q3x/kernels/sm87_a4w4_attention_o_k512_cell.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes =
    static_cast<unsigned int>(
        kSm87A4W4AttentionOK512PackedRowK64Bytes);

struct alignas(16) Sm87A4W4AttentionOK512Stage final {
  std::uint8_t a[kSm87A4W4AttentionOK512K64PerCopy]
                [kSm87A4W4AttentionOK512TileM *
                 kSm87A4W4AttentionOK512PackedRowK64Bytes];
  std::uint8_t b[kSm87A4W4AttentionOK512K64PerCopy]
                [kSm87A4W4AttentionOK512TileN *
                 kSm87A4W4AttentionOK512PackedRowK64Bytes];
};

struct alignas(16) Sm87A4W4AttentionOK512Shared final {
  Sm87A4W4AttentionOK512Stage
      stage[kSm87A4W4AttentionOK512Stages];
};

struct alignas(16) Sm87A4W4FloatAccumulator final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(Sm87A4W4AttentionOK512Stage) ==
              kSm87A4W4AttentionOK512StageBytes);
static_assert(sizeof(Sm87A4W4AttentionOK512Shared) ==
              kSm87A4W4AttentionOK512SharedBytes);
static_assert(sizeof(Sm87A4W4FloatAccumulator) == 16U);

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

// Copies one physical K256 phase: four independently swizzled K64 planes for
// A (16,384 bytes) and B (8,192 bytes).  Every thread issues exactly six
// aligned 16-byte cp.async.cg operations.
__device__ __forceinline__ void issue_k256_stage(
    Sm87A4W4AttentionOK512Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int physical_k256_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kAVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4AttentionOK512TileM * kPackedK64Bytes / 16U);
  constexpr unsigned int kBVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4AttentionOK512TileN * kPackedK64Bytes / 16U);
  constexpr unsigned int kAVectors =
      static_cast<unsigned int>(kSm87A4W4AttentionOK512K64PerCopy) *
      kAVectorsPerPlane;
  constexpr unsigned int kBVectors =
      static_cast<unsigned int>(kSm87A4W4AttentionOK512K64PerCopy) *
      kBVectorsPerPlane;
  static_assert(kAVectors == 4U * kSm87A4W4AttentionOK512Threads);
  static_assert(kBVectors == 2U * kSm87A4W4AttentionOK512Threads);

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 4U; ++iteration) {
    const unsigned int vector = threadIdx.x +
                                iteration *
                                    kSm87A4W4AttentionOK512Threads;
    const unsigned int plane = vector / kAVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kAVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int row_vector = vector_in_plane % 2U;
    const unsigned int physical_k64_group =
        physical_k256_group *
            static_cast<unsigned int>(
                kSm87A4W4AttentionOK512K64PerCopy) +
        plane;
    cp_async_16(
        stage.a[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                             row, 16U * row_vector),
        packed_a + sm87_a4w4_attention_o_k512_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_k64_group, 16U * row_vector,
                       physical_k64_group_count));
  }

#pragma unroll
  for (unsigned int iteration = 0U; iteration < 2U; ++iteration) {
    const unsigned int vector = threadIdx.x +
                                iteration *
                                    kSm87A4W4AttentionOK512Threads;
    const unsigned int plane = vector / kBVectorsPerPlane;
    const unsigned int vector_in_plane =
        vector - plane * kBVectorsPerPlane;
    const unsigned int row = vector_in_plane / 2U;
    const unsigned int row_vector = vector_in_plane % 2U;
    const unsigned int physical_k64_group =
        physical_k256_group *
            static_cast<unsigned int>(
                kSm87A4W4AttentionOK512K64PerCopy) +
        plane;
    cp_async_16(
        stage.b[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                             row, 16U * row_vector),
        packed_b + sm87_a4w4_attention_o_k512_packed_offset(
                       static_cast<std::size_t>(n_tile_start) + row,
                       physical_k64_group, 16U * row_vector,
                       physical_k64_group_count));
  }
  cp_async_commit();
}

// Adds four K64 MMA operations into all 32 resident S32 partial registers.
// The same partial tuple is passed through both physical K256 stages, so no
// floating-point operation occurs before all eight K64 terms are present.
__device__ __forceinline__ void accumulate_k256_stage(
    const Sm87A4W4AttentionOK512Stage& stage,
    Sm87A4W4Accumulator (&partials)[8U]) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;

#pragma unroll
  for (unsigned int plane = 0U;
       plane < kSm87A4W4AttentionOK512K64PerCopy; ++plane) {
    const Sm87A4W4AFragment a_fragment =
        sm87_a4w4_load_a_fragment_swizzled_shared(
            stage.a[plane] + warp * 16U * kPackedK64Bytes, lane);
#pragma unroll
    for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
      const Sm87A4W4BFragment b_fragment =
          sm87_a4w4_load_b_fragment_swizzled_shared(
              stage.b[plane] + fragment_n * 8U * kPackedK64Bytes,
              lane);
      sm87_a4w4_mma_m16n8k64(partials[fragment_n], a_fragment,
                             b_fragment);
    }
  }
}

__device__ __forceinline__ void apply_k512_group(
    Sm87A4W4FloatAccumulator (&accumulators)[8U],
    const Sm87A4W4Accumulator (&partials)[8U],
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const unsigned int global_m0 =
      m_tile_start + warp * 16U + coordinate0.m;
  const unsigned int global_m1 =
      m_tile_start + warp * 16U + coordinate2.m;
  const float a_scale0 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_attention_o_k512_scale_offset(
          global_m0, k512_group, k512_group_count)]);
  const float a_scale1 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_attention_o_k512_scale_offset(
          global_m1, k512_group, k512_group_count)]);

#pragma unroll
  for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
    const Sm87A4W4AccumulatorCoordinate coordinate1 =
        sm87_a4w4_accumulator_coordinate(lane, 1U);
    const unsigned int global_n0 =
        n_tile_start + fragment_n * 8U + coordinate0.n;
    const unsigned int global_n1 =
        n_tile_start + fragment_n * 8U + coordinate1.n;
    const float b_scale0 = decode_bf16(
        b_k512_scales_bf16[sm87_a4w4_attention_o_k512_scale_offset(
            global_n0, k512_group, k512_group_count)]);
    const float b_scale1 = decode_bf16(
        b_k512_scales_bf16[sm87_a4w4_attention_o_k512_scale_offset(
            global_n1, k512_group, k512_group_count)]);
    const float scale00 = __fmul_rn(a_scale0, b_scale0);
    const float scale01 = __fmul_rn(a_scale0, b_scale1);
    const float scale10 = __fmul_rn(a_scale1, b_scale0);
    const float scale11 = __fmul_rn(a_scale1, b_scale1);
    accumulators[fragment_n].x0 = __fmaf_rn(
        static_cast<float>(partials[fragment_n].x0), scale00,
        accumulators[fragment_n].x0);
    accumulators[fragment_n].x1 = __fmaf_rn(
        static_cast<float>(partials[fragment_n].x1), scale01,
        accumulators[fragment_n].x1);
    accumulators[fragment_n].x2 = __fmaf_rn(
        static_cast<float>(partials[fragment_n].x2), scale10,
        accumulators[fragment_n].x2);
    accumulators[fragment_n].x3 = __fmaf_rn(
        static_cast<float>(partials[fragment_n].x3), scale11,
        accumulators[fragment_n].x3);
  }
}

__device__ __forceinline__ void store_output(
    const Sm87A4W4FloatAccumulator (&accumulators)[8U],
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
#pragma unroll
  for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
#pragma unroll
    for (unsigned int output = 0U; output < 4U; ++output) {
      const Sm87A4W4AccumulatorCoordinate coordinate =
          sm87_a4w4_accumulator_coordinate(lane, output);
      const unsigned int global_m =
          m_tile_start + warp * 16U + coordinate.m;
      const unsigned int global_n =
          n_tile_start + fragment_n * 8U + coordinate.n;
      const float values[4U] = {
          accumulators[fragment_n].x0,
          accumulators[fragment_n].x1,
          accumulators[fragment_n].x2,
          accumulators[fragment_n].x3};
      output_bf16[static_cast<std::size_t>(global_m) *
                      output_row_stride_elements +
                  global_n] = encode_bf16(values[output]);
    }
  }
}

}  // namespace

extern "C" __global__
    __launch_bounds__(kSm87A4W4AttentionOK512Threads,
                      kSm87A4W4AttentionOK512CtasPerSm)
void q3x_sm87_a4w4_attention_o_k512_m128n64k512_kernel(
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
  __shared__ Sm87A4W4AttentionOK512Shared shared;

  for (unsigned int work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const unsigned int n_tile = work_tile / m_tile_count;
    const unsigned int m_tile = work_tile - n_tile * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4AttentionOK512TileM;
    const unsigned int n_tile_start =
        n_tile * kSm87A4W4AttentionOK512TileN;
    Sm87A4W4FloatAccumulator accumulators[8U]{};

    issue_k256_stage(shared.stage[0U], packed_a, packed_b,
                     m_tile_start, n_tile_start, 0U,
                     physical_k64_group_count);
    issue_k256_stage(shared.stage[1U], packed_a, packed_b,
                     m_tile_start, n_tile_start, 1U,
                     physical_k64_group_count);

    for (unsigned int group = 0U; group < k512_group_count; ++group) {
      Sm87A4W4Accumulator partials[8U]{};

      // Two committed groups are outstanding here.  Retire only the older
      // physical K256 stage, leaving its partner eligible to overlap.
      cp_async_wait<1U>();
      __syncthreads();
      accumulate_k256_stage(shared.stage[0U], partials);
      __syncthreads();

      const unsigned int next_even_phase = 2U * group + 2U;
      if (next_even_phase < 2U * k512_group_count) {
        issue_k256_stage(shared.stage[0U], packed_a, packed_b,
                         m_tile_start, n_tile_start, next_even_phase,
                         physical_k64_group_count);
        // Current odd plus next even are outstanding.  Retire current odd.
        cp_async_wait<1U>();
      } else {
        // The final odd stage is the only outstanding group.
        cp_async_wait<0U>();
      }
      __syncthreads();
      accumulate_k256_stage(shared.stage[1U], partials);

      // This is the sole floating-point boundary for the logical K512 group.
      apply_k512_group(accumulators, partials, a_k512_scales_bf16,
                       b_k512_scales_bf16, m_tile_start, n_tile_start,
                       group, k512_group_count);
      __syncthreads();

      const unsigned int next_odd_phase = 2U * group + 3U;
      if (next_odd_phase < 2U * k512_group_count) {
        issue_k256_stage(shared.stage[1U], packed_a, packed_b,
                         m_tile_start, n_tile_start, next_odd_phase,
                         physical_k64_group_count);
      }
    }

    store_output(accumulators, m_tile_start, n_tile_start, output_bf16,
                 output_row_stride_elements);
    __syncthreads();
  }
}

namespace {

[[nodiscard]] int validate_sm87(
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
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_attention_o_k512_resources_cuda(
    Sm87A4W4AttentionOK512Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4AttentionOK512Resources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_attention_o_k512_m128n64k512_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_attention_o_k512_m128n64k512_kernel,
      static_cast<int>(kSm87A4W4AttentionOK512Threads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = attributes.maxDynamicSharedSizeBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  if (resources->registers_per_thread >
          static_cast<int>(kSm87A4W4AttentionOK512MaximumRegisters) ||
      resources->local_bytes != 0U ||
      resources->static_shared_bytes !=
          kSm87A4W4AttentionOK512SharedBytes ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4AttentionOK512Threads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4AttentionOK512CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

namespace {

[[nodiscard]] int launch_sm87_a4w4_attention_o_k512_impl(
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
  const Sm87A4W4AttentionOK512Plan plan =
      sm87_a4w4_attention_o_k512_plan(token_count, output_size,
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
      token_count > std::numeric_limits<unsigned int>::max() ||
      output_size > std::numeric_limits<unsigned int>::max() ||
      input_size > std::numeric_limits<unsigned int>::max() ||
      plan.k512_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.work_tiles > std::numeric_limits<unsigned int>::max() ||
      !sm87_a4w4_attention_o_k512_product_fits(
          token_count, output_row_stride_elements)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_attention_o_k512_packed_capacity_bytes(token_count,
                                                        input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_attention_o_k512_packed_capacity_bytes(output_size,
                                                        input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_attention_o_k512_scale_capacity_elements(token_count,
                                                          input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_attention_o_k512_scale_capacity_elements(output_size,
                                                          input_size);
  const std::size_t required_output_elements =
      token_count * output_row_stride_elements;
  if (required_a_bytes == 0U || required_b_bytes == 0U ||
      required_a_scales == 0U || required_b_scales == 0U ||
      !sm87_a4w4_attention_o_k512_product_fits(
          required_a_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_attention_o_k512_product_fits(
          required_b_scales, sizeof(std::uint16_t)) ||
      !sm87_a4w4_attention_o_k512_product_fits(
          required_output_elements, sizeof(std::uint16_t)) ||
      packed_a_capacity_bytes < required_a_bytes ||
      packed_b_capacity_bytes < required_b_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      b_scale_capacity_elements < required_b_scales ||
      output_capacity_elements < required_output_elements) {
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

  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const unsigned int persistent_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  const unsigned int launch_ctas =
      persistent_ctas < maximum_launch_ctas
          ? persistent_ctas
          : maximum_launch_ctas;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_attention_o_k512_m128n64k512_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(kSm87A4W4AttentionOK512Threads), 0U,
         stream>>>(packed_a, a_k512_scales_bf16, packed_b,
                   b_k512_scales_bf16,
                   static_cast<unsigned int>(plan.k512_groups),
                   static_cast<unsigned int>(plan.physical_k64_groups),
                   output_bf16,
                   static_cast<unsigned int>(
                       output_row_stride_elements),
                   static_cast<unsigned int>(plan.m_tiles),
                   static_cast<unsigned int>(plan.work_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int launch_sm87_a4w4_attention_o_k512_bf16_cuda(
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
  return launch_sm87_a4w4_attention_o_k512_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements, token_count,
      output_size, input_size, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4AttentionOK512PersistentCtas),
      cuda_stream);
}

int launch_sm87_a4w4_attention_o_k512_test_bf16_cuda(
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
  return launch_sm87_a4w4_attention_o_k512_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity_elements, packed_b, packed_b_capacity_bytes,
      b_k512_scales_bf16, b_scale_capacity_elements, token_count,
      output_size, input_size, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, cuda_stream);
}

}  // namespace q3x::kernels
