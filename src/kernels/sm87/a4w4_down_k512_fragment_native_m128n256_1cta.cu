#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native_m128n256_1cta.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) M128N256AStage final {
  std::uint8_t plane
      [kSm87A4W4DownK512FragmentNativeM128N2561CtaK64PerScale]
      [kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM *
       kPackedK64Bytes];
};

struct alignas(16) M128N256ScaleSlot final {
  std::uint16_t
      a[kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM];
  std::uint16_t
      b[kSm87A4W4DownK512FragmentNativeM128N2561CtaTileN];
};

struct alignas(16) M128N256Shared final {
  M128N256AStage
      stage[kSm87A4W4DownK512FragmentNativeM128N2561CtaAStages];
  M128N256ScaleSlot
      scale[kSm87A4W4DownK512FragmentNativeM128N2561CtaAStages];
};

struct alignas(16) BRecord final {
  Sm87A4W4BFragment n0;
  Sm87A4W4BFragment n1;
};

struct alignas(16) Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

static_assert(sizeof(M128N256AStage) ==
              kSm87A4W4DownK512FragmentNativeM128N2561CtaAStageBytes);
static_assert(sizeof(M128N256ScaleSlot) ==
              kSm87A4W4DownK512FragmentNativeM128N2561CtaScaleSlotBytes);
static_assert(sizeof(M128N256Shared) ==
              kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes);
static_assert(sizeof(BRecord) == 16U);
static_assert(sizeof(Float4) == 16U);

[[nodiscard]] constexpr bool aligned(
    const void* const pointer, const std::size_t alignment) noexcept {
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

// One M128K64 plane is exactly 256 aligned vectors, one per CTA thread.
__device__ __forceinline__ void issue_a_plane(
    std::uint8_t* const destination,
    const std::uint8_t* const packed_a,
    const unsigned int m_tile_start,
    const unsigned int physical_k64,
    const unsigned int physical_k64_group_count) noexcept {
  const unsigned int row = threadIdx.x >> 1U;
  const unsigned int row_vector = threadIdx.x & 1U;
  cp_async_16(
      destination + sm87_a4w4_swizzled_k64_byte_offset(
                        row, 16U * row_vector),
      packed_a + sm87_a4w4_down_k512_packed_offset(
                     static_cast<std::size_t>(m_tile_start) + row,
                     physical_k64, 16U * row_vector,
                     physical_k64_group_count));
}

__device__ __forceinline__ void issue_scales(
    M128N256ScaleSlot& destination,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  if (threadIdx.x < 16U) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        destination.a + first_row,
        a_k512_scales_bf16 + sm87_a4w4_down_k512_scale_offset(
                                    static_cast<std::size_t>(m_tile_start) +
                                        first_row,
                                    k512_group, k512_group_count));
  } else if (threadIdx.x < 48U) {
    const unsigned int first_row = 8U * (threadIdx.x - 16U);
    cp_async_16(
        destination.b + first_row,
        b_k512_scales_bf16 + sm87_a4w4_down_k512_scale_offset(
                                    static_cast<std::size_t>(n_tile_start) +
                                        first_row,
                                    k512_group, k512_group_count));
  }
}

__device__ __forceinline__ void issue_k512_group(
    M128N256AStage& destination,
    M128N256ScaleSlot& scale,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count) noexcept {
#pragma unroll
  for (unsigned int plane = 0U;
       plane <
       kSm87A4W4DownK512FragmentNativeM128N2561CtaK64PerScale;
       ++plane) {
    issue_a_plane(
        destination.plane[plane], packed_a, m_tile_start,
        k512_group *
                kSm87A4W4DownK512FragmentNativeM128N2561CtaK64PerScale +
            plane,
        physical_k64_group_count);
    if (plane == 0U) {
      issue_scales(scale, a_k512_scales_bf16,
                   b_k512_scales_bf16, m_tile_start,
                   n_tile_start, k512_group, k512_group_count);
    }
    cp_async_commit();
  }
}

[[nodiscard]] __device__ __forceinline__ BRecord load_b_record_ca(
    const std::uint8_t* const fragment_native_b,
    const unsigned int n256_stripe,
    const unsigned int warp_n32,
    const unsigned int n16_phase,
    const unsigned int k512_group,
    const unsigned int k64_in_group,
    const unsigned int lane,
    const unsigned int k512_group_count) noexcept {
  const unsigned int absolute_n16 = 2U * warp_n32 + n16_phase;
  const unsigned int n128_panel =
      2U * n256_stripe + absolute_n16 / 8U;
  const unsigned int n16_in_panel = absolute_n16 & 7U;
  const std::uint8_t* const pointer =
      fragment_native_b +
      sm87_a4w4_down_k512_fragment_b_vector_offset(
          n128_panel, k512_group, k64_in_group, n16_in_panel,
          lane, k512_group_count);
  BRecord result{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile("ld.global.ca.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(result.n0.x0), "=r"(result.n0.x1),
                 "=r"(result.n1.x0), "=r"(result.n1.x1)
               : "l"(pointer)
               : "memory");
#else
  asm volatile("trap;");
#endif
  return result;
}

__device__ __forceinline__ void clear_partials(
    Sm87A4W4Accumulator
        (&partial)[kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp]
                   [kSm87A4W4DownK512FragmentNativeM128N2561CtaN8PerPhase])
    noexcept {
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 <
       kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp;
       ++m16) {
    partial[m16][0U] = Sm87A4W4Accumulator{};
    partial[m16][1U] = Sm87A4W4Accumulator{};
  }
}

template <unsigned int Phase>
__device__ __forceinline__ void accumulate_phase(
    const M128N256AStage& stage,
    const std::uint8_t* const fragment_native_b,
    const unsigned int n256_stripe,
    const unsigned int warp_n32,
    const unsigned int k512_group,
    const unsigned int k512_group_count,
    Sm87A4W4Accumulator
        (&partial)[kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp]
                   [kSm87A4W4DownK512FragmentNativeM128N2561CtaN8PerPhase])
    noexcept {
  static_assert(
      Phase <
      kSm87A4W4DownK512FragmentNativeM128N2561CtaN16Phases);
  unsigned int lane = threadIdx.x & 31U;
  if constexpr (Phase == 1U) {
    // Split phase-1 address CSE from phase 0 without adding another special
    // register read.  Otherwise ptxas holds one lane/swizzle term across the
    // whole first N16 body and spills it at the 255-register boundary.
    asm volatile("" : "+r"(lane));
  }
#pragma unroll
  for (unsigned int plane = 0U;
       plane <
       kSm87A4W4DownK512FragmentNativeM128N2561CtaK64PerScale;
       ++plane) {
    const BRecord b = load_b_record_ca(
        fragment_native_b, n256_stripe, warp_n32, Phase,
        k512_group, plane, lane, k512_group_count);
#pragma unroll
    for (unsigned int m16 = 0U;
         m16 <
         kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp;
         ++m16) {
      const Sm87A4W4AFragment a =
          sm87_a4w4_load_a_fragment_swizzled_shared(
              stage.plane[plane] + m16 * kSm87A4W4MmaM *
                                       kPackedK64Bytes,
              lane);
      // One B record feeds all eight ordered M16 panels before its registers
      // die.  No M-half reload or second B feed is permitted.
      sm87_a4w4_mma_m16n8k64(partial[m16][0U], a, b.n0);
      sm87_a4w4_mma_m16n8k64(partial[m16][1U], a, b.n1);
    }
  }
}

template <unsigned int Phase>
__device__ __forceinline__ void apply_phase(
    Float4
        (&accumulator)[kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp]
                       [4U],
    const Sm87A4W4Accumulator
        (&partial)[kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp]
                   [kSm87A4W4DownK512FragmentNativeM128N2561CtaN8PerPhase],
    const M128N256ScaleSlot& scale,
    const unsigned int warp_n32) noexcept {
  static_assert(
      Phase <
      kSm87A4W4DownK512FragmentNativeM128N2561CtaN16Phases);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int column1 = column0 + 1U;
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 <
       kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp;
       ++m16) {
    const float a0 = decode_bf16(scale.a[m16 * 16U + row0]);
    const float a1 = decode_bf16(scale.a[m16 * 16U + row1]);
#pragma unroll
    for (unsigned int n8 = 0U;
         n8 <
         kSm87A4W4DownK512FragmentNativeM128N2561CtaN8PerPhase;
         ++n8) {
      constexpr unsigned int kN8PhaseOffset = 2U * Phase;
      const unsigned int local_n =
          warp_n32 * 32U + Phase * 16U + n8 * 8U;
      const float b0 = decode_bf16(scale.b[local_n + column0]);
      const float b1 = decode_bf16(scale.b[local_n + column1]);
      const Sm87A4W4Accumulator& source = partial[m16][n8];
      Float4& destination = accumulator[m16][kN8PhaseOffset + n8];
      destination.x0 = __fmaf_rn(
          static_cast<float>(source.x0), __fmul_rn(a0, b0),
          destination.x0);
      destination.x1 = __fmaf_rn(
          static_cast<float>(source.x1), __fmul_rn(a0, b1),
          destination.x1);
      destination.x2 = __fmaf_rn(
          static_cast<float>(source.x2), __fmul_rn(a1, b0),
          destination.x2);
      destination.x3 = __fmaf_rn(
          static_cast<float>(source.x3), __fmul_rn(a1, b1),
          destination.x3);
    }
  }
}

__device__ __forceinline__ void store_output(
    const Float4
        (&accumulator)[kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp]
                       [4U],
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int warp_n32,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int row0 = lane >> 2U;
  const unsigned int row1 = row0 + 8U;
  const unsigned int column0 = 2U * (lane & 3U);
  const unsigned int column1 = column0 + 1U;
#pragma unroll
  for (unsigned int m16 = 0U;
       m16 <
       kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp;
       ++m16) {
#pragma unroll
    for (unsigned int n8 = 0U; n8 < 4U; ++n8) {
      const Float4& value = accumulator[m16][n8];
      const unsigned int base_m = m_tile_start + m16 * 16U;
      const unsigned int base_n =
          n_tile_start + warp_n32 * 32U + n8 * 8U;
      output_bf16[static_cast<std::size_t>(base_m + row0) *
                          output_row_stride_elements +
                      base_n + column0] = encode_bf16(value.x0);
      output_bf16[static_cast<std::size_t>(base_m + row0) *
                          output_row_stride_elements +
                      base_n + column1] = encode_bf16(value.x1);
      output_bf16[static_cast<std::size_t>(base_m + row1) *
                          output_row_stride_elements +
                      base_n + column0] = encode_bf16(value.x2);
      output_bf16[static_cast<std::size_t>(base_m + row1) *
                          output_row_stride_elements +
                      base_n + column1] = encode_bf16(value.x3);
    }
  }
}

}  // namespace

extern "C" __global__
    __launch_bounds__(
        kSm87A4W4DownK512FragmentNativeM128N2561CtaThreads,
        kSm87A4W4DownK512FragmentNativeM128N2561CtaCtasPerSm)
void q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const fragment_native_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int m_tile_count,
    const unsigned int n_stripe_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared = *reinterpret_cast<M128N256Shared*>(dynamic_shared);
  const unsigned int warp_n32 = threadIdx.x >> 5U;

  // For P1920, m_owner_ctas=15 and n_wave_groups=1: all 15 CTAs walk one
  // N256 stripe together.  For P512, four M owners are replicated over four
  // N waves, filling all 16 SMs without changing complete-cell ownership.
  for (unsigned int n_stripe = blockIdx.y; n_stripe < n_stripe_count;
       n_stripe += gridDim.y) {
    const unsigned int n_tile_start =
        n_stripe *
        kSm87A4W4DownK512FragmentNativeM128N2561CtaTileN;
    for (unsigned int m_tile = blockIdx.x; m_tile < m_tile_count;
         m_tile += gridDim.x) {
      const unsigned int m_tile_start =
          m_tile *
          kSm87A4W4DownK512FragmentNativeM128N2561CtaTileM;
      Float4 accumulator
          [kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp]
          [4U]{};
      Sm87A4W4Accumulator partial
          [kSm87A4W4DownK512FragmentNativeM128N2561CtaM16PerWarp]
          [kSm87A4W4DownK512FragmentNativeM128N2561CtaN8PerPhase]{};

      issue_k512_group(
          shared.stage[0U], shared.scale[0U], packed_a,
          a_k512_scales_bf16, b_k512_scales_bf16, m_tile_start,
          n_tile_start, 0U, k512_group_count,
          physical_k64_group_count);
      cp_async_wait<0U>();
      __syncthreads();

      for (unsigned int group = 0U; group < k512_group_count; ++group) {
        const unsigned int current = group & 1U;
        const unsigned int next = current ^ 1U;
        const bool has_next = group + 1U < k512_group_count;

        clear_partials(partial);
        accumulate_phase<0U>(
            shared.stage[current], fragment_native_b, n_stripe,
            warp_n32, group, k512_group_count, partial);
        if (has_next) {
          issue_k512_group(
              shared.stage[next], shared.scale[next], packed_a,
              a_k512_scales_bf16, b_k512_scales_bf16,
              m_tile_start, n_tile_start, group + 1U,
              k512_group_count, physical_k64_group_count);
        }
        apply_phase<0U>(accumulator, partial, shared.scale[current],
                        warp_n32);

        clear_partials(partial);
        accumulate_phase<1U>(
            shared.stage[current], fragment_native_b, n_stripe,
            warp_n32, group, k512_group_count, partial);
        apply_phase<1U>(accumulator, partial, shared.scale[current],
                        warp_n32);

        if (has_next) {
          cp_async_wait<0U>();
          __syncthreads();
        }
      }

      store_output(accumulator, m_tile_start, n_tile_start, warp_n32,
                   output_bf16, output_row_stride_elements);
      __syncthreads();
    }
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
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount) ||
      properties.sharedMemPerBlockOptin <
          kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] cudaError_t configure_kernel() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes));
  if (status != cudaSuccess) {
    return status;
  }
  return cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_kernel,
      cudaFuncAttributePreferredSharedMemoryCarveout, 100);
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
  const auto plan = production_shape
                        ? sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
                              token_count, output_size, input_size)
                        : sm87_a4w4_down_k512_fragment_native_m128n256_1cta_test_plan(
                              token_count, output_size, input_size);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k512_scales_bf16, 16U) ||
      !aligned(fragment_native_b, 16U) ||
      !aligned(b_k512_scales_bf16, 16U) ||
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
      plan.n_stripes > std::numeric_limits<unsigned int>::max() ||
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
      fragment_native_b_capacity_bytes < required_b_bytes ||
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

  const std::size_t required_output_bytes =
      required_output_elements * sizeof(std::uint16_t);
  if (byte_ranges_overlap(output_bf16, required_output_bytes, packed_a,
                          required_a_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          a_k512_scales_bf16,
                          required_a_scales * sizeof(std::uint16_t)) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          fragment_native_b, required_b_bytes) ||
      byte_ranges_overlap(output_bf16, required_output_bytes,
                          b_k512_scales_bf16,
                          required_b_scales * sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87A4W4DownK512FragmentNativeM128N2561CtaResources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }
  unsigned int launch_ctas = static_cast<unsigned int>(plan.launch_ctas);
  if (launch_ctas > maximum_launch_ctas) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_kernel
      <<<dim3(static_cast<unsigned int>(plan.m_owner_ctas),
              static_cast<unsigned int>(plan.n_wave_groups), 1U),
         kSm87A4W4DownK512FragmentNativeM128N2561CtaThreads,
         kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes,
         stream>>>(
          packed_a, a_k512_scales_bf16, fragment_native_b,
          b_k512_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.n_stripes), output_bf16,
          static_cast<unsigned int>(output_row_stride_elements));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_resources_cuda(
    Sm87A4W4DownK512FragmentNativeM128N2561CtaResources* const
        resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources =
      Sm87A4W4DownK512FragmentNativeM128N2561CtaResources{};
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
      q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_kernel,
      static_cast<int>(
          kSm87A4W4DownK512FragmentNativeM128N2561CtaThreads),
      kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes;
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
              kSm87A4W4DownK512FragmentNativeM128N2561CtaMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes ||
      resources->device_optin_shared_limit_bytes <
          kSm87A4W4DownK512FragmentNativeM128N2561CtaSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4DownK512FragmentNativeM128N2561CtaThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(
              kSm87A4W4DownK512FragmentNativeM128N2561CtaCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_bf16_cuda(
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
          kSm87A4W4DownK512FragmentNativeM128N2561CtaPersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_test_bf16_cuda(
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
