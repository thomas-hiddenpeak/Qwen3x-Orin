#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes = 32U;

struct alignas(16) FragmentNativeAStage final {
  std::uint8_t plane[2U]
                    [kSm87A4W4GateUpK512FragmentNativeTileM *
                     kPackedK64Bytes];
};

struct alignas(16) FragmentNativeShared final {
  FragmentNativeAStage
      stage[kSm87A4W4GateUpK512FragmentNativeAStages];
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

static_assert(sizeof(FragmentNativeAStage) ==
              kSm87A4W4GateUpK512FragmentNativeAStageBytes);
static_assert(sizeof(FragmentNativeShared) ==
              kSm87A4W4GateUpK512FragmentNativeSharedBytes);
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
             block_groups,
             kSm87A4W4GateUpK512FragmentNativeTileM)
             ? block_groups *
                   kSm87A4W4GateUpK512FragmentNativeTileM
             : 0U;
}

[[nodiscard]] __host__ __device__ constexpr std::size_t a_scale_offset(
    const std::size_t outer_coordinate,
    const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate /
               kSm87A4W4GateUpK512FragmentNativeTileM *
               k512_group_count +
           k512_group) *
              kSm87A4W4GateUpK512FragmentNativeTileM +
          outer_coordinate %
              kSm87A4W4GateUpK512FragmentNativeTileM);
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

__device__ __forceinline__ void issue_a_k128(
    FragmentNativeAStage& stage,
    const std::uint8_t* const packed_a,
    const unsigned int m_tile_start,
    const unsigned int physical_k128_group,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      kSm87A4W4GateUpK512FragmentNativeTileM *
      kPackedK64Bytes / 16U;
  static_assert(2U * kVectorsPerPlane ==
                kSm87A4W4GateUpK512FragmentNativeThreads);
  const unsigned int vector = threadIdx.x;
  const unsigned int plane = vector / kVectorsPerPlane;
  const unsigned int vector_in_plane =
      vector - plane * kVectorsPerPlane;
  const unsigned int row = vector_in_plane / 2U;
  const unsigned int row_vector = vector_in_plane & 1U;
  const unsigned int physical_k64 =
      2U * physical_k128_group + plane;
  cp_async_16(
      stage.plane[plane] + sm87_a4w4_swizzled_k64_byte_offset(
                                  row, 16U * row_vector),
      packed_a + sm87_a4w4_consumer_packed_offset(
                     static_cast<std::size_t>(m_tile_start) + row,
                     physical_k64, 16U * row_vector,
                     physical_k64_group_count));
}

[[nodiscard]] __device__ __forceinline__ PairedBFragment
load_paired_b_fragment_cg(const std::uint8_t* const pointer) noexcept {
  PairedBFragment result{};
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "ld.global.cg.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(result.gate.x0), "=r"(result.gate.x1),
        "=r"(result.up.x0), "=r"(result.up.x1)
      : "l"(pointer)
      : "memory");
#else
  asm volatile("trap;");
#endif
  return result;
}

[[nodiscard]] __device__ __forceinline__ unsigned long long load_u64_cg(
    const std::uint16_t* const pointer) noexcept {
  unsigned long long result = 0ULL;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("ld.global.cg.u64 %0, [%1];"
               : "=l"(result)
               : "l"(pointer)
               : "memory");
#else
  asm volatile("trap;");
#endif
  return result;
}

__device__ __forceinline__ void accumulate_k64(
    const FragmentNativeAStage& stage,
    const unsigned int plane,
    const PairedBFragment& b,
    Sm87A4W4Accumulator (&gate_partial)[4U],
    Sm87A4W4Accumulator (&up_partial)[4U]) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    const Sm87A4W4AFragment a =
        sm87_a4w4_load_a_fragment_swizzled_shared(
            stage.plane[plane] + panel * 16U * kPackedK64Bytes,
            lane);
    sm87_a4w4_mma_m16n8k64(gate_partial[panel], a, b.gate);
    sm87_a4w4_mma_m16n8k64(up_partial[panel], a, b.up);
  }
}

__device__ __forceinline__ void clear_partials(
    Sm87A4W4Accumulator (&gate_partial)[4U],
    Sm87A4W4Accumulator (&up_partial)[4U]) noexcept {
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    gate_partial[panel] = Sm87A4W4Accumulator{};
    up_partial[panel] = Sm87A4W4Accumulator{};
  }
}

__device__ __forceinline__ void apply_k512_group(
    Float4 (&gate_accumulator)[4U],
    Float4 (&up_accumulator)[4U],
    const Sm87A4W4Accumulator (&gate_partial)[4U],
    const Sm87A4W4Accumulator (&up_partial)[4U],
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int absolute_n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane_column = lane & 3U;

  // Only lanes 0..3 fetch the two adjacent N scales assigned to their
  // column pair.  The 64-bit record is [G(n0),U(n0),G(n1),U(n1)].
  unsigned long long paired_scale = 0ULL;
  if (lane < 4U) {
    const unsigned int first_n =
        absolute_n_tile_start + warp * 8U + lane * 2U;
    paired_scale = load_u64_cg(
        paired_b_scales_bf16 +
        sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
            first_n, k512_group, k512_group_count));
  }
  unsigned int paired_scale_low = __shfl_sync(
      0xffff'ffffU, static_cast<unsigned int>(paired_scale),
      lane_column);
  unsigned int paired_scale_high = __shfl_sync(
      0xffff'ffffU,
      static_cast<unsigned int>(paired_scale >> 32U), lane_column);
  const float gate_scale0 =
      decode_bf16(static_cast<std::uint16_t>(paired_scale_low));
  const float up_scale0 = decode_bf16(
      static_cast<std::uint16_t>(paired_scale_low >> 16U));
  const float gate_scale1 =
      decode_bf16(static_cast<std::uint16_t>(paired_scale_high));
  const float up_scale1 = decode_bf16(
      static_cast<std::uint16_t>(paired_scale_high >> 16U));

  const unsigned int row_in_fragment = lane >> 2U;
  const unsigned int scale_source_lane = lane & ~3U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
    unsigned int packed_a_scale = 0U;
    if (lane_column == 0U) {
      const unsigned int first_m =
          m_tile_start + panel * 16U + row_in_fragment;
      const std::uint16_t scale0 =
          a_k512_scales_bf16[a_scale_offset(
              first_m, k512_group, k512_group_count)];
      const std::uint16_t scale1 =
          a_k512_scales_bf16[a_scale_offset(
              first_m + 8U, k512_group, k512_group_count)];
      packed_a_scale = static_cast<unsigned int>(scale0) |
                       (static_cast<unsigned int>(scale1) << 16U);
    }
    packed_a_scale = __shfl_sync(
        0xffff'ffffU, packed_a_scale, scale_source_lane);
    const float a_scale0 =
        decode_bf16(static_cast<std::uint16_t>(packed_a_scale));
    const float a_scale1 = decode_bf16(
        static_cast<std::uint16_t>(packed_a_scale >> 16U));

    const float gate00 = __fmul_rn(a_scale0, gate_scale0);
    const float gate01 = __fmul_rn(a_scale0, gate_scale1);
    const float gate10 = __fmul_rn(a_scale1, gate_scale0);
    const float gate11 = __fmul_rn(a_scale1, gate_scale1);
    const float up00 = __fmul_rn(a_scale0, up_scale0);
    const float up01 = __fmul_rn(a_scale0, up_scale1);
    const float up10 = __fmul_rn(a_scale1, up_scale0);
    const float up11 = __fmul_rn(a_scale1, up_scale1);
    gate_accumulator[panel].x0 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x0), gate00,
        gate_accumulator[panel].x0);
    gate_accumulator[panel].x1 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x1), gate01,
        gate_accumulator[panel].x1);
    gate_accumulator[panel].x2 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x2), gate10,
        gate_accumulator[panel].x2);
    gate_accumulator[panel].x3 = __fmaf_rn(
        static_cast<float>(gate_partial[panel].x3), gate11,
        gate_accumulator[panel].x3);
    up_accumulator[panel].x0 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x0), up00,
        up_accumulator[panel].x0);
    up_accumulator[panel].x1 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x1), up01,
        up_accumulator[panel].x1);
    up_accumulator[panel].x2 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x2), up10,
        up_accumulator[panel].x2);
    up_accumulator[panel].x3 = __fmaf_rn(
        static_cast<float>(up_partial[panel].x3), up11,
        up_accumulator[panel].x3);
  }
}

__device__ __forceinline__ void store_product(
    const Float4 (&gate_accumulator)[4U],
    const Float4 (&up_accumulator)[4U],
    const unsigned int m_tile_start,
    const unsigned int output_n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 4U; ++panel) {
#pragma unroll
    for (unsigned int output = 0U; output < 4U; ++output) {
      const Sm87A4W4AccumulatorCoordinate coordinate =
          sm87_a4w4_accumulator_coordinate(lane, output);
      const float gate_values[4U] = {
          gate_accumulator[panel].x0, gate_accumulator[panel].x1,
          gate_accumulator[panel].x2, gate_accumulator[panel].x3};
      const float up_values[4U] = {
          up_accumulator[panel].x0, up_accumulator[panel].x1,
          up_accumulator[panel].x2, up_accumulator[panel].x3};
      const unsigned int output_m =
          m_tile_start + panel * 16U + coordinate.m;
      const unsigned int output_n =
          output_n_tile_start + warp * 8U + coordinate.n;
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
    __launch_bounds__(kSm87A4W4GateUpK512FragmentNativeThreads,
                      kSm87A4W4GateUpK512FragmentNativeCtasPerSm)
void q3x_sm87_a4w4_gateup_k512_fragment_native_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const paired_b_codes,
    const std::uint16_t* const paired_b_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    const unsigned int absolute_n_start,
    const unsigned int n_tile_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile_count) {
  extern __shared__ __align__(16) unsigned char dynamic_shared[];
  auto& shared =
      *reinterpret_cast<FragmentNativeShared*>(dynamic_shared);
  const unsigned int warp = threadIdx.x >> 5U;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int total_k128_groups = 4U * k512_group_count;

  // N is outermost for every resident CTA.  Around the P2K operating point,
  // one CTA owns each M64 tile, so all CTAs request the same N64 weight tile
  // concurrently and the register-native B stream can be served from L2.
  for (unsigned int n_tile = 0U; n_tile < n_tile_count; ++n_tile) {
    const unsigned int absolute_n_tile_start =
        absolute_n_start +
        n_tile * kSm87A4W4GateUpK512FragmentNativeTileN;
    const unsigned int output_n_tile_start =
        n_tile * kSm87A4W4GateUpK512FragmentNativeTileN;
    for (unsigned int m_tile = blockIdx.x; m_tile < m_tile_count;
         m_tile += gridDim.x) {
      const unsigned int m_tile_start =
          m_tile * kSm87A4W4GateUpK512FragmentNativeTileM;
      Float4 gate_accumulator[4U]{};
      Float4 up_accumulator[4U]{};
      Sm87A4W4Accumulator gate_partial[4U]{};
      Sm87A4W4Accumulator up_partial[4U]{};

      const unsigned int initial_stages =
          total_k128_groups <
                  kSm87A4W4GateUpK512FragmentNativeAStages
              ? total_k128_groups
              : kSm87A4W4GateUpK512FragmentNativeAStages;
      for (unsigned int stage = 0U; stage < initial_stages; ++stage) {
        issue_a_k128(shared.stage[stage], packed_a, m_tile_start,
                     stage, physical_k64_group_count);
        cp_async_commit();
      }
      if (initial_stages == 3U) {
        cp_async_wait<2U>();
      } else if (initial_stages == 2U) {
        cp_async_wait<1U>();
      } else {
        cp_async_wait<0U>();
      }
      __syncthreads();

      for (unsigned int physical_k128 = 0U;
           physical_k128 < total_k128_groups; ++physical_k128) {
        const unsigned int ring =
            physical_k128 %
            kSm87A4W4GateUpK512FragmentNativeAStages;
        const unsigned int group = physical_k128 / 4U;
        const unsigned int first_k64_phase =
            2U * (physical_k128 & 3U);
#pragma unroll
        for (unsigned int plane = 0U; plane < 2U; ++plane) {
          const std::size_t slot_offset =
              sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
                  absolute_n_tile_start + warp * 8U, group,
                  first_k64_phase + plane, lane,
                  k512_group_count);
          const PairedBFragment b = load_paired_b_fragment_cg(
              paired_b_codes + slot_offset);
          accumulate_k64(shared.stage[ring], plane, b,
                         gate_partial, up_partial);
        }

        if ((physical_k128 & 3U) == 3U) {
          apply_k512_group(
              gate_accumulator, up_accumulator, gate_partial,
              up_partial, a_k512_scales_bf16,
              paired_b_scales_bf16, m_tile_start,
              absolute_n_tile_start, group, k512_group_count);
          if (group + 1U < k512_group_count) {
            clear_partials(gate_partial, up_partial);
          }
        }

        // Every warp has finished this A slot before it is recycled.  With
        // three slots, the next A copy remains two groups ahead of compute.
        __syncthreads();
        const unsigned int next_k128 =
            physical_k128 +
            kSm87A4W4GateUpK512FragmentNativeAStages;
        if (next_k128 < total_k128_groups) {
          issue_a_k128(shared.stage[ring], packed_a, m_tile_start,
                       next_k128, physical_k64_group_count);
          cp_async_commit();
        }
        const unsigned int future =
            total_k128_groups - physical_k128 - 1U;
        if (future >= 3U) {
          cp_async_wait<2U>();
        } else if (future == 2U) {
          cp_async_wait<1U>();
        } else if (future == 1U) {
          cp_async_wait<0U>();
        }
        if (future != 0U) {
          __syncthreads();
        }
      }

      store_product(gate_accumulator, up_accumulator, m_tile_start,
                    output_n_tile_start, output_bf16,
                    output_row_stride_elements);
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
      properties.multiProcessorCount != static_cast<int>(kRequiredSmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
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
  const Sm87A4W4GateUpK512FragmentNativePlan plan =
      sm87_a4w4_gateup_k512_fragment_native_plan(
          token_count, intermediate_size, input_size, n_start,
          n_count);
  if (plan.launch_ctas == 0U || maximum_launch_ctas == 0U ||
      (model_only &&
       !sm87_a4w4_gateup_k512_fragment_native_is_model_plan(plan))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (!aligned(packed_a, 16U) || !aligned(a_k512_scales_bf16, 2U) ||
      !aligned(paired_b_codes, 16U) ||
      !aligned(paired_b_scales_bf16, 8U) ||
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
      paired_b_scale_capacity < required_b_scales) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (token_count >
      std::numeric_limits<std::size_t>::max() /
          output_row_stride_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_output =
      (token_count - 1U) * output_row_stride_elements + n_count;
  if (output_capacity_elements < required_output) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  unsigned int launch_ctas =
      static_cast<unsigned int>(plan.launch_ctas);
  if (launch_ctas > maximum_launch_ctas) {
    launch_ctas = maximum_launch_ctas;
  }
  const cudaStream_t stream =
      reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_gateup_k512_fragment_native_kernel
      <<<launch_ctas,
         kSm87A4W4GateUpK512FragmentNativeThreads,
         kSm87A4W4GateUpK512FragmentNativeSharedBytes,
         stream>>>(
          packed_a, a_k512_scales_bf16, paired_b_codes,
          paired_b_scales_bf16,
          static_cast<unsigned int>(plan.k512_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          static_cast<unsigned int>(plan.n_start),
          static_cast<unsigned int>(plan.n_tiles), output_bf16,
          static_cast<unsigned int>(output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_gateup_k512_fragment_native_resources_cuda(
    Sm87A4W4GateUpK512FragmentNativeResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpK512FragmentNativeResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_sm87(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_gateup_k512_fragment_native_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_k512_fragment_native_kernel,
      static_cast<int>(kSm87A4W4GateUpK512FragmentNativeThreads),
      kSm87A4W4GateUpK512FragmentNativeSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4GateUpK512FragmentNativeSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4GateUpK512FragmentNativeMaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      attributes.maxDynamicSharedSizeBytes <
          static_cast<int>(
              kSm87A4W4GateUpK512FragmentNativeSharedBytes) ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(
              kSm87A4W4GateUpK512FragmentNativeThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(
              kSm87A4W4GateUpK512FragmentNativeCtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_k512_fragment_native_bf16_cuda(
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
          kSm87A4W4GateUpK512FragmentNativePersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_gateup_k512_fragment_native_test_bf16_cuda(
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
