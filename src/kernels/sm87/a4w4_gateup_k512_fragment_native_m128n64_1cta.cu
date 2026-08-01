#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) FragmentNativeM128N641CtaAStage final {
  std::uint8_t plane
      [kSm87A4W4GateUpK512FragmentNativeM128N641CtaK64PerScale]
      [kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM *
       kPackedK64Bytes];
};

struct alignas(16) FragmentNativeM128N641CtaScaleSlot final {
  std::uint16_t
      a[kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM];
  std::uint16_t
      paired_b[kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileN][2U];
};

struct alignas(16) FragmentNativeM128N641CtaShared final {
  FragmentNativeM128N641CtaAStage
      stage[kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStages];
  FragmentNativeM128N641CtaScaleSlot
      scale[kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStages];
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

struct alignas(16) PairedBFragment final {
  Sm87A4W4BFragment gate;
  Sm87A4W4BFragment up;
};

static_assert(sizeof(FragmentNativeM128N641CtaAStage) ==
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaAStageBytes);
static_assert(sizeof(FragmentNativeM128N641CtaScaleSlot) ==
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaScaleSlotBytes);
static_assert(sizeof(FragmentNativeM128N641CtaShared) ==
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes);
static_assert(sizeof(Float4) == 16U);
static_assert(sizeof(PairedBFragment) ==
              kSm87A4W4GateUpK512FragmentNativePairSlotBytes);

[[nodiscard]] constexpr std::size_t a_scale_capacity_elements(
    const std::size_t outer_count,
    const std::size_t logical_k) noexcept {
  const std::size_t groups =
      sm87_a4w4_gateup_k512_fragment_native_scale_groups(logical_k);
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  if (groups == 0U || blocks == 0U ||
      !sm87_a4w4_gateup_k512_fragment_native_product_fits(
          blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_gateup_k512_fragment_native_product_fits(
             block_groups, 64U)
             ? block_groups * 64U
             : 0U;
}

[[nodiscard]] __host__ __device__ constexpr std::size_t a_scale_offset(
    const std::size_t outer_coordinate,
    const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / 64U * k512_group_count + k512_group) *
              64U +
          outer_coordinate % 64U);
}

[[nodiscard]] constexpr bool aligned(
    const void* const pointer, const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
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

[[nodiscard]] __device__ __forceinline__ float silu_product(
    const float gate, const float up) noexcept {
  if (gate >= 0.0F) {
    return (gate / (1.0F + expf(-gate))) * up;
  }
  const float exponential = expf(gate);
  return (gate * exponential / (1.0F + exponential)) * up;
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

// One K64 plane has 256 aligned vectors, exactly one per CTA thread.
__device__ __forceinline__ void issue_a_plane(
    std::uint8_t* const destination,
    const std::uint8_t* const packed_a,
    const unsigned int m_tile_start,
    const unsigned int physical_k64,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM *
      kPackedK64Bytes / 16U;
  static_assert(kVectorsPerPlane ==
                kSm87A4W4GateUpK512FragmentNativeM128N641CtaThreads);
  const unsigned int vector = threadIdx.x;
  const unsigned int row = vector / 2U;
  const unsigned int row_vector = vector & 1U;
  cp_async_16(
      destination + sm87_a4w4_swizzled_k64_byte_offset(
                        row, 16U * row_vector),
      packed_a + sm87_a4w4_consumer_packed_offset(
                     static_cast<std::size_t>(m_tile_start) + row,
                     physical_k64, 16U * row_vector,
                     physical_k64_group_count));
}

__device__ __forceinline__ void issue_scales(
    FragmentNativeM128N641CtaScaleSlot& destination,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        destination.a + first_row,
        a_k512_scales_bf16 +
            a_scale_offset(
                static_cast<std::size_t>(m_tile_start) + first_row,
                k512_group, k512_group_count));
  }
  if (threadIdx.x >= 16U && threadIdx.x < 32U) {
    const unsigned int first_row = 4U * (threadIdx.x - 16U);
    cp_async_16(
        destination.paired_b[first_row],
        paired_b_scales_bf16 +
            sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
                static_cast<std::size_t>(absolute_n_tile_start) +
                    first_row,
                k512_group, k512_group_count));
  }
}

__device__ __forceinline__ void issue_k512_group(
    FragmentNativeM128N641CtaAStage& destination,
    FragmentNativeM128N641CtaScaleSlot& scale,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count) noexcept {
#pragma unroll
  for (unsigned int plane = 0U;
       plane <
       kSm87A4W4GateUpK512FragmentNativeM128N641CtaK64PerScale;
       ++plane) {
    issue_a_plane(
        destination.plane[plane], packed_a, m_tile_start,
        k512_group *
                kSm87A4W4GateUpK512FragmentNativeM128N641CtaK64PerScale +
            plane,
        physical_k64_group_count);
    if (plane == 0U) {
      issue_scales(scale, a_k512_scales_bf16,
                   paired_b_scales_bf16, m_tile_start,
                   absolute_n_tile_start, k512_group,
                   k512_group_count);
    }
    cp_async_commit();
  }
}

[[nodiscard]] __device__ __forceinline__ PairedBFragment
load_paired_b_fragment_ca(const std::uint8_t* const pointer) noexcept {
  PairedBFragment result{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(result.gate.x0), "=r"(result.gate.x1),
        "=r"(result.up.x0), "=r"(result.up.x1)
      : "l"(pointer)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return result;
}

__device__ __forceinline__ void clear_partials(
    Sm87A4W4Accumulator (&gate_partial)[8U],
    Sm87A4W4Accumulator (&up_partial)[8U]) noexcept {
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    gate_partial[panel] = Sm87A4W4Accumulator{};
    up_partial[panel] = Sm87A4W4Accumulator{};
  }
}

__device__ __forceinline__ void accumulate_plane(
    const FragmentNativeM128N641CtaAStage& stage,
    const unsigned int plane,
    const PairedBFragment& b,
    Sm87A4W4Accumulator (&gate_partial)[8U],
    Sm87A4W4Accumulator (&up_partial)[8U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const Sm87A4W4AFragment a =
        sm87_a4w4_load_a_fragment_swizzled_shared(
            stage.plane[plane] +
                panel * 16U * kPackedK64Bytes,
            lane);
    sm87_a4w4_mma_m16n8k64(gate_partial[panel], a, b.gate);
    sm87_a4w4_mma_m16n8k64(up_partial[panel], a, b.up);
  }
}

__device__ __forceinline__ void apply_k512_group(
    Float4 (&gate_accumulator)[8U],
    Float4 (&up_accumulator)[8U],
    const Sm87A4W4Accumulator (&gate_partial)[8U],
    const Sm87A4W4Accumulator (&up_partial)[8U],
    const FragmentNativeM128N641CtaScaleSlot& scale,
    const unsigned int warp_n8) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int lane_column = lane & 3U;
  const unsigned int first_n =
      warp_n8 * 8U + lane_column * 2U;
  const float gate_scale0 =
      decode_bf16(scale.paired_b[first_n][0U]);
  const float up_scale0 =
      decode_bf16(scale.paired_b[first_n][1U]);
  const float gate_scale1 =
      decode_bf16(scale.paired_b[first_n + 1U][0U]);
  const float up_scale1 =
      decode_bf16(scale.paired_b[first_n + 1U][1U]);
  const unsigned int row_in_fragment = lane >> 2U;

#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const unsigned int first_m =
        panel * 16U + row_in_fragment;
    const float a_scale0 = decode_bf16(scale.a[first_m]);
    const float a_scale1 = decode_bf16(scale.a[first_m + 8U]);
    gate_accumulator[panel].x0 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x0),
        __fmul_rn(a_scale0, gate_scale0),
        gate_accumulator[panel].x0);
    gate_accumulator[panel].x1 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x1),
        __fmul_rn(a_scale0, gate_scale1),
        gate_accumulator[panel].x1);
    gate_accumulator[panel].x2 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x2),
        __fmul_rn(a_scale1, gate_scale0),
        gate_accumulator[panel].x2);
    gate_accumulator[panel].x3 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x3),
        __fmul_rn(a_scale1, gate_scale1),
        gate_accumulator[panel].x3);
    up_accumulator[panel].x0 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x0),
        __fmul_rn(a_scale0, up_scale0), up_accumulator[panel].x0);
    up_accumulator[panel].x1 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x1),
        __fmul_rn(a_scale0, up_scale1), up_accumulator[panel].x1);
    up_accumulator[panel].x2 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x2),
        __fmul_rn(a_scale1, up_scale0), up_accumulator[panel].x2);
    up_accumulator[panel].x3 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x3),
        __fmul_rn(a_scale1, up_scale1), up_accumulator[panel].x3);
  }
}

__device__ __forceinline__ void store_product(
    const Float4 (&gate_accumulator)[8U],
    const Float4 (&up_accumulator)[8U],
    const unsigned int m_tile_start,
    const unsigned int output_n_tile_start,
    const unsigned int warp_n8,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const float gate_values[4U] = {
        gate_accumulator[panel].x0, gate_accumulator[panel].x1,
        gate_accumulator[panel].x2, gate_accumulator[panel].x3};
    const float up_values[4U] = {
        up_accumulator[panel].x0, up_accumulator[panel].x1,
        up_accumulator[panel].x2, up_accumulator[panel].x3};
#pragma unroll
    for (unsigned int output = 0U; output < 4U; ++output) {
      const Sm87A4W4AccumulatorCoordinate coordinate =
          sm87_a4w4_accumulator_coordinate(lane, output);
      const unsigned int output_m =
          m_tile_start + panel * 16U + coordinate.m;
      const unsigned int output_n =
          output_n_tile_start + warp_n8 * 8U + coordinate.n;
      output_bf16[
          static_cast<std::size_t>(output_m) *
              output_row_stride_elements +
          output_n] =
          encode_bf16(silu_product(gate_values[output],
                                   up_values[output]));
    }
  }
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4GateUpK512FragmentNativeM128N641CtaThreads,
        kSm87A4W4GateUpK512FragmentNativeM128N641CtaCtasPerSm)
void q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const paired_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int absolute_n_start,
    const unsigned int n_tile_count,
    const unsigned int m_tile_count,
    const unsigned int work_cell_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<FragmentNativeM128N641CtaShared*>(
          dynamic_shared);
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp_n8 = warp;

  for (unsigned int cell = blockIdx.x; cell < work_cell_count;
       cell += gridDim.x) {
    const unsigned int n_tile = cell / m_tile_count;
    const unsigned int m_tile = cell - n_tile * m_tile_count;
    if (n_tile >= n_tile_count) {
      return;
    }
    const unsigned int m_tile_start =
        m_tile *
        kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileM;
    const unsigned int absolute_n_tile_start =
        absolute_n_start +
        n_tile *
            kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileN;
    const unsigned int output_n_tile_start =
        n_tile *
        kSm87A4W4GateUpK512FragmentNativeM128N641CtaTileN;

    Float4 gate_accumulator[8U]{};
    Float4 up_accumulator[8U]{};
    Sm87A4W4Accumulator gate_partial[8U]{};
    Sm87A4W4Accumulator up_partial[8U]{};

    issue_k512_group(
        shared.stage[0U], shared.scale[0U], packed_a,
        a_k512_scales_bf16, paired_b_scales_bf16, m_tile_start,
        absolute_n_tile_start, 0U, k512_group_count,
        physical_k64_group_count);
    cp_async_wait<0U>();
    __syncthreads();

    for (unsigned int group = 0U; group < k512_group_count; ++group) {
      const unsigned int current = group & 1U;
      const unsigned int next = current ^ 1U;
      const bool has_next = group + 1U < k512_group_count;

      clear_partials(gate_partial, up_partial);
#pragma unroll
      for (unsigned int plane = 0U;
           plane <
           kSm87A4W4GateUpK512FragmentNativeM128N641CtaK64PerScale;
           ++plane) {
        const std::size_t slot_offset =
            sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                absolute_n_tile_start + warp_n8 * 8U, group,
                plane, lane, k512_group_count);
        const PairedBFragment b = load_paired_b_fragment_ca(
            paired_b_codes + slot_offset);
        accumulate_plane(shared.stage[current], plane, b, gate_partial,
                         up_partial);
        if (has_next) {
          issue_a_plane(
              shared.stage[next].plane[plane], packed_a,
              m_tile_start,
              (group + 1U) *
                      kSm87A4W4GateUpK512FragmentNativeM128N641CtaK64PerScale +
                  plane,
              physical_k64_group_count);
          if (plane == 0U) {
            issue_scales(shared.scale[next],
                         a_k512_scales_bf16,
                         paired_b_scales_bf16, m_tile_start,
                         absolute_n_tile_start, group + 1U,
                         k512_group_count);
          }
          cp_async_commit();
        }
      }
      apply_k512_group(
          gate_accumulator, up_accumulator, gate_partial, up_partial,
          shared.scale[current], warp_n8);

      if (has_next) {
        cp_async_wait<0U>();
        __syncthreads();
      }
    }

    store_product(gate_accumulator, up_accumulator, m_tile_start,
                  output_n_tile_start, warp_n8, output_bf16,
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

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
}

[[nodiscard]] int launch_impl(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity,
    const std::uint8_t* const paired_b_codes,
    const std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    const bool model_only,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpK512FragmentNativeM128N641CtaPlan plan =
      sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
          token_count, intermediate_size, input_size, n_start,
          n_count);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      (model_only &&
       !sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_is_model_plan(
           plan))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(paired_b_codes, 16U) ||
      !aligned(paired_b_scales_bf16, 16U) ||
      !aligned(output_bf16, 2U) ||
      output_row_stride_elements < n_count) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_a_codes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count,
                                                input_size);
  const std::size_t required_a_scales =
      a_scale_capacity_elements(token_count, input_size);
  const std::size_t required_b_codes =
      sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
          intermediate_size, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
          intermediate_size, input_size);
  if (required_a_codes == 0U || required_a_scales == 0U ||
      required_b_codes == 0U || required_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_codes ||
      a_scale_capacity < required_a_scales ||
      paired_b_code_capacity_bytes < required_b_codes ||
      paired_b_scale_capacity < required_b_scales ||
      token_count >
          std::numeric_limits<std::size_t>::max() /
              output_row_stride_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_output =
      (token_count - 1U) * output_row_stride_elements + n_count;
  if (output_capacity_elements < required_output ||
      plan.k512_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.work_cells > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaError_t configure_status = configure_kernel();
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }
  unsigned int launch_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  if (launch_ctas > maximum_launch_ctas) {
    launch_ctas = maximum_launch_ctas;
  }
  const cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_kernel
      <<<launch_ctas,
         kSm87A4W4GateUpK512FragmentNativeM128N641CtaThreads,
         kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes,
         stream>>>(
          packed_a, a_k512_scales_bf16, paired_b_codes,
          paired_b_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          static_cast<unsigned int>(plan.n_start),
          static_cast<unsigned int>(plan.n_tiles),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.work_cells), output_bf16,
          static_cast<unsigned int>(output_row_stride_elements));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_resources_cuda(
    Sm87A4W4GateUpK512FragmentNativeM128N641CtaResources* const
        resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources =
      Sm87A4W4GateUpK512FragmentNativeM128N641CtaResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_sm87(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaError_t configure_status = configure_kernel();
  if (configure_status != cudaSuccess) {
    return static_cast<int>(configure_status);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_kernel,
      static_cast<int>(
          kSm87A4W4GateUpK512FragmentNativeM128N641CtaThreads),
      kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      attributes.maxDynamicSharedSizeBytes <
          static_cast<int>(
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaSharedBytes) ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(
              kSm87A4W4GateUpK512FragmentNativeM128N641CtaCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity,
    const std::uint8_t* const paired_b_codes,
    const std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity, paired_b_codes,
      paired_b_code_capacity_bytes, paired_b_scales_bf16,
      paired_b_scale_capacity, token_count, intermediate_size,
      input_size, n_start, n_count, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      static_cast<unsigned int>(
          kSm87A4W4GateUpK512FragmentNativeM128N641CtaPersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::size_t a_scale_capacity,
    const std::uint8_t* const paired_b_codes,
    const std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* const paired_b_scales_bf16,
    const std::size_t paired_b_scale_capacity,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements,
    const unsigned int maximum_launch_ctas,
    void* const cuda_stream) noexcept {
  return launch_impl(
      packed_a, packed_a_capacity_bytes, a_k512_scales_bf16,
      a_scale_capacity, paired_b_codes,
      paired_b_code_capacity_bytes, paired_b_scales_bf16,
      paired_b_scale_capacity, token_count, intermediate_size,
      input_size, n_start, n_count, output_bf16,
      output_row_stride_elements, output_capacity_elements,
      maximum_launch_ctas, false, cuda_stream);
}

}  // namespace q3x::kernels
