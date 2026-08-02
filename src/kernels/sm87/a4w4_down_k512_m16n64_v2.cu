#include "q3x/kernels/sm87_a4w4_down_k512_m16n64_v2.h"

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

struct alignas(16) Sm87A4W4DownK512M16N64V2Stage final {
  std::uint8_t a[kSm87A4W4DownK512M16N64V2K64PerStage]
                [kSm87A4W4DownK512M16N64V2TileM *
                 kSm87A4W4DownK512PackedRowK64Bytes];
  std::uint8_t b[kSm87A4W4DownK512M16N64V2K64PerStage]
                [kSm87A4W4DownK512M16N64V2TileN *
                 kSm87A4W4DownK512PackedRowK64Bytes];
};

struct alignas(16) Sm87A4W4DownK512M16N64V2Shared final {
  Sm87A4W4DownK512M16N64V2Stage
      stage[kSm87A4W4DownK512M16N64V2Stages];
};

struct alignas(16) Sm87A4W4DownK512M16N64V2Float4 final {
  float x0{};
  float x1{};
  float x2{};
  float x3{};
};

// Keep every output fragment as a separately named object.  A runtime-indexed
// Float4[8] is lowered to exactly 128 bytes of thread-local stack by nvcc, even
// though the arithmetic itself only needs 32 long-lived FP32 registers.
struct Sm87A4W4DownK512M16N64V2Accumulators final {
  Sm87A4W4DownK512M16N64V2Float4 n0{};
  Sm87A4W4DownK512M16N64V2Float4 n1{};
  Sm87A4W4DownK512M16N64V2Float4 n2{};
  Sm87A4W4DownK512M16N64V2Float4 n3{};
  Sm87A4W4DownK512M16N64V2Float4 n4{};
  Sm87A4W4DownK512M16N64V2Float4 n5{};
  Sm87A4W4DownK512M16N64V2Float4 n6{};
  Sm87A4W4DownK512M16N64V2Float4 n7{};
};

static_assert(sizeof(Sm87A4W4DownK512M16N64V2Stage) ==
              kSm87A4W4DownK512M16N64V2StageBytes);
static_assert(sizeof(Sm87A4W4DownK512M16N64V2Shared) ==
              kSm87A4W4DownK512M16N64V2DynamicSharedBytes);
static_assert(sizeof(Sm87A4W4DownK512M16N64V2Float4) == 16U);
static_assert(sizeof(Sm87A4W4DownK512M16N64V2Accumulators) == 128U);

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

// One K256 stage is 32 KiB.  The linear copy domain contains the four A
// planes followed by the four B planes.  With 512 threads every lane issues
// exactly four aligned 16-byte copies instead of the predecessor's eight.
__device__ __forceinline__ void issue_k256_stage(
    Sm87A4W4DownK512M16N64V2Stage& stage,
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int physical_k256_stage,
    const unsigned int physical_k64_group_count) noexcept {
  constexpr unsigned int kVectorsPerPlane =
      static_cast<unsigned int>(
          kSm87A4W4DownK512M16N64V2TileM * kPackedK64Bytes / 16U);
  constexpr unsigned int kVectorsPerOperand =
      kSm87A4W4DownK512M16N64V2K64PerStage * kVectorsPerPlane;
  constexpr unsigned int kTotalVectors = 2U * kVectorsPerOperand;
  static_assert(kVectorsPerPlane == 256U);
  static_assert(kTotalVectors ==
                4U * kSm87A4W4DownK512M16N64V2Threads);

#pragma unroll
  for (unsigned int linear = threadIdx.x; linear < kTotalVectors;
       linear += kSm87A4W4DownK512M16N64V2Threads) {
    const unsigned int operand = linear / kVectorsPerOperand;
    const unsigned int operand_vector =
        linear - operand * kVectorsPerOperand;
    const unsigned int plane = operand_vector / kVectorsPerPlane;
    const unsigned int plane_vector =
        operand_vector - plane * kVectorsPerPlane;
    const unsigned int row = plane_vector / 2U;
    const unsigned int row_vector = plane_vector & 1U;
    const unsigned int physical_k64_group =
        physical_k256_stage *
            static_cast<unsigned int>(
                kSm87A4W4DownK512M16N64V2K64PerStage) +
        plane;
    const std::size_t shared_offset =
        sm87_a4w4_swizzled_k64_byte_offset(row, 16U * row_vector);
    const unsigned int outer_start =
        operand == 0U ? m_tile_start : n_tile_start;
    std::uint8_t* const destination =
        operand == 0U ? stage.a[plane] : stage.b[plane];
    const std::uint8_t* const source =
        operand == 0U ? packed_a : packed_b;
    cp_async_16(
        destination + shared_offset,
        source + sm87_a4w4_down_k512_packed_offset(
                     static_cast<std::size_t>(outer_start) + row,
                     physical_k64_group, 16U * row_vector,
                     physical_k64_group_count));
  }
  cp_async_commit();
}

// One warp owns M16N64.  Every N8 fragment reduces all eight physical K64
// partials into one S32 tuple before the K512 scale product is applied.  The
// fragment-at-a-time partial lifetime is what makes the 128-register gate
// possible while preserving the predecessor's exact arithmetic order.
__device__ __forceinline__ void update_accumulator(
    Sm87A4W4DownK512M16N64V2Float4& output,
    const Sm87A4W4Accumulator& partial,
    const float scale00,
    const float scale01,
    const float scale10,
    const float scale11) noexcept {
  output.x0 =
      __fmaf_rn(static_cast<float>(partial.x0), scale00, output.x0);
  output.x1 =
      __fmaf_rn(static_cast<float>(partial.x1), scale01, output.x1);
  output.x2 =
      __fmaf_rn(static_cast<float>(partial.x2), scale10, output.x2);
  output.x3 =
      __fmaf_rn(static_cast<float>(partial.x3), scale11, output.x3);
}

__device__ __forceinline__ void accumulate_k512_group(
    const Sm87A4W4DownK512M16N64V2Stage& first,
    const Sm87A4W4DownK512M16N64V2Stage& second,
    Sm87A4W4DownK512M16N64V2Accumulators& accumulators,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k512_group,
    const unsigned int k512_group_count) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp % kSm87A4W4DownK512M16N64V2WarpRows;
  const unsigned int warp_n =
      warp / kSm87A4W4DownK512M16N64V2WarpRows;
  const unsigned int local_m_start =
      warp_m * kSm87A4W4DownK512M16N64V2WarpTileM;
  const unsigned int local_n_start =
      warp_n * kSm87A4W4DownK512M16N64V2WarpTileN;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);

  const float a0 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
          static_cast<std::size_t>(m_tile_start + local_m_start +
                                   coordinate0.m),
          k512_group, k512_group_count)]);
  const float a1 = decode_bf16(
      a_k512_scales_bf16[sm87_a4w4_down_k512_scale_offset(
          static_cast<std::size_t>(m_tile_start + local_m_start +
                                   coordinate2.m),
          k512_group, k512_group_count)]);

#pragma unroll 1
  for (unsigned int fragment_n = 0U;
       fragment_n < kSm87A4W4DownK512M16N64V2WarpTileN / 8U;
       ++fragment_n) {
    const unsigned int fragment_n_start =
        local_n_start + fragment_n * 8U;
    Sm87A4W4Accumulator partial{};
#pragma unroll
    for (unsigned int half = 0U;
         half < kSm87A4W4DownK512M16N64V2StagesPerScale; ++half) {
      const Sm87A4W4DownK512M16N64V2Stage& stage =
          half == 0U ? first : second;
#pragma unroll
      for (unsigned int plane = 0U;
           plane < kSm87A4W4DownK512M16N64V2K64PerStage; ++plane) {
        const Sm87A4W4AFragment a_fragment =
            sm87_a4w4_load_a_fragment_swizzled_shared(
                stage.a[plane] + local_m_start * kPackedK64Bytes,
                lane);
        const Sm87A4W4BFragment b_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                stage.b[plane] + fragment_n_start * kPackedK64Bytes,
                lane);
        sm87_a4w4_mma_m16n8k64(partial, a_fragment, b_fragment);
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
    const float scale00 = __fmul_rn(a0, b0);
    const float scale01 = __fmul_rn(a0, b1);
    const float scale10 = __fmul_rn(a1, b0);
    const float scale11 = __fmul_rn(a1, b1);
    // The induction variable deliberately remains rolled so only one S32
    // partial is live.  The named switch prevents its destination from
    // acquiring a runtime address and becoming local memory.
    switch (fragment_n) {
      case 0U:
        update_accumulator(accumulators.n0, partial, scale00, scale01,
                           scale10, scale11);
        break;
      case 1U:
        update_accumulator(accumulators.n1, partial, scale00, scale01,
                           scale10, scale11);
        break;
      case 2U:
        update_accumulator(accumulators.n2, partial, scale00, scale01,
                           scale10, scale11);
        break;
      case 3U:
        update_accumulator(accumulators.n3, partial, scale00, scale01,
                           scale10, scale11);
        break;
      case 4U:
        update_accumulator(accumulators.n4, partial, scale00, scale01,
                           scale10, scale11);
        break;
      case 5U:
        update_accumulator(accumulators.n5, partial, scale00, scale01,
                           scale10, scale11);
        break;
      case 6U:
        update_accumulator(accumulators.n6, partial, scale00, scale01,
                           scale10, scale11);
        break;
      default:
        update_accumulator(accumulators.n7, partial, scale00, scale01,
                           scale10, scale11);
        break;
    }
  }
}

template <unsigned int FragmentN>
__device__ __forceinline__ void store_output_fragment(
    const Sm87A4W4DownK512M16N64V2Float4& values,
    const unsigned int base_m,
    const unsigned int warp_n_start,
    const Sm87A4W4AccumulatorCoordinate coordinate0,
    const Sm87A4W4AccumulatorCoordinate coordinate1,
    const Sm87A4W4AccumulatorCoordinate coordinate2,
    const Sm87A4W4AccumulatorCoordinate coordinate3,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  static_assert(FragmentN < 8U);
  constexpr unsigned int kFragmentNOffset = FragmentN * 8U;
  output_bf16[static_cast<std::size_t>(base_m + coordinate0.m) *
                      output_row_stride_elements +
                  warp_n_start + kFragmentNOffset + coordinate0.n] =
      encode_bf16(values.x0);
  output_bf16[static_cast<std::size_t>(base_m + coordinate1.m) *
                      output_row_stride_elements +
                  warp_n_start + kFragmentNOffset + coordinate1.n] =
      encode_bf16(values.x1);
  output_bf16[static_cast<std::size_t>(base_m + coordinate2.m) *
                      output_row_stride_elements +
                  warp_n_start + kFragmentNOffset + coordinate2.n] =
      encode_bf16(values.x2);
  output_bf16[static_cast<std::size_t>(base_m + coordinate3.m) *
                      output_row_stride_elements +
                  warp_n_start + kFragmentNOffset + coordinate3.n] =
      encode_bf16(values.x3);
}

__device__ __forceinline__ void store_output(
    const Sm87A4W4DownK512M16N64V2Accumulators& accumulators,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m =
      warp % kSm87A4W4DownK512M16N64V2WarpRows;
  const unsigned int warp_n =
      warp / kSm87A4W4DownK512M16N64V2WarpRows;
  const unsigned int base_m =
      m_tile_start + warp_m * kSm87A4W4DownK512M16N64V2WarpTileM;
  const unsigned int warp_n_start =
      n_tile_start + warp_n * kSm87A4W4DownK512M16N64V2WarpTileN;
  const Sm87A4W4AccumulatorCoordinate coordinate0 =
      sm87_a4w4_accumulator_coordinate(lane, 0U);
  const Sm87A4W4AccumulatorCoordinate coordinate1 =
      sm87_a4w4_accumulator_coordinate(lane, 1U);
  const Sm87A4W4AccumulatorCoordinate coordinate2 =
      sm87_a4w4_accumulator_coordinate(lane, 2U);
  const Sm87A4W4AccumulatorCoordinate coordinate3 =
      sm87_a4w4_accumulator_coordinate(lane, 3U);

  store_output_fragment<0U>(accumulators.n0, base_m, warp_n_start,
                            coordinate0, coordinate1, coordinate2,
                            coordinate3, output_bf16,
                            output_row_stride_elements);
  store_output_fragment<1U>(accumulators.n1, base_m, warp_n_start,
                            coordinate0, coordinate1, coordinate2,
                            coordinate3, output_bf16,
                            output_row_stride_elements);
  store_output_fragment<2U>(accumulators.n2, base_m, warp_n_start,
                            coordinate0, coordinate1, coordinate2,
                            coordinate3, output_bf16,
                            output_row_stride_elements);
  store_output_fragment<3U>(accumulators.n3, base_m, warp_n_start,
                            coordinate0, coordinate1, coordinate2,
                            coordinate3, output_bf16,
                            output_row_stride_elements);
  store_output_fragment<4U>(accumulators.n4, base_m, warp_n_start,
                            coordinate0, coordinate1, coordinate2,
                            coordinate3, output_bf16,
                            output_row_stride_elements);
  store_output_fragment<5U>(accumulators.n5, base_m, warp_n_start,
                            coordinate0, coordinate1, coordinate2,
                            coordinate3, output_bf16,
                            output_row_stride_elements);
  store_output_fragment<6U>(accumulators.n6, base_m, warp_n_start,
                            coordinate0, coordinate1, coordinate2,
                            coordinate3, output_bf16,
                            output_row_stride_elements);
  store_output_fragment<7U>(accumulators.n7, base_m, warp_n_start,
                            coordinate0, coordinate1, coordinate2,
                            coordinate3, output_bf16,
                            output_row_stride_elements);
}

__device__ __forceinline__ void compute_tile(
    Sm87A4W4DownK512M16N64V2Shared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k512_scales_bf16,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k512_scales_bf16,
    const unsigned int k512_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const output_bf16,
    const unsigned int output_row_stride_elements,
    const unsigned int m_tile,
    const unsigned int n_tile) noexcept {
  const unsigned int m_tile_start =
      m_tile * kSm87A4W4DownK512M16N64V2TileM;
  const unsigned int n_tile_start =
      n_tile * kSm87A4W4DownK512M16N64V2TileN;
  Sm87A4W4DownK512M16N64V2Accumulators accumulators{};

  const unsigned int physical_stage_count = 2U * k512_group_count;
  const unsigned int initial_stages =
      physical_stage_count < kSm87A4W4DownK512M16N64V2Stages
          ? physical_stage_count
          : static_cast<unsigned int>(
                kSm87A4W4DownK512M16N64V2Stages);
  for (unsigned int phase = 0U; phase < initial_stages; ++phase) {
    issue_k256_stage(shared.stage[phase], packed_a, packed_b,
                     m_tile_start, n_tile_start, phase,
                     physical_k64_group_count);
  }

  for (unsigned int group = 0U; group < k512_group_count; ++group) {
    if (group + 1U < k512_group_count) {
      cp_async_wait<2U>();
    } else {
      cp_async_wait<0U>();
    }
    __syncthreads();

    const unsigned int first_phase = 2U * group;
    accumulate_k512_group(
        shared.stage[first_phase % kSm87A4W4DownK512M16N64V2Stages],
        shared.stage[(first_phase + 1U) %
                     kSm87A4W4DownK512M16N64V2Stages],
        accumulators, a_k512_scales_bf16, b_k512_scales_bf16,
        m_tile_start, n_tile_start, group, k512_group_count);

    __syncthreads();
    const unsigned int future0 =
        first_phase + kSm87A4W4DownK512M16N64V2Stages;
    const unsigned int future1 = future0 + 1U;
    if (future0 < physical_stage_count) {
      issue_k256_stage(
          shared.stage[future0 % kSm87A4W4DownK512M16N64V2Stages],
          packed_a, packed_b, m_tile_start, n_tile_start, future0,
          physical_k64_group_count);
    }
    if (future1 < physical_stage_count) {
      issue_k256_stage(
          shared.stage[future1 % kSm87A4W4DownK512M16N64V2Stages],
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
    __launch_bounds__(kSm87A4W4DownK512M16N64V2Threads,
                      kSm87A4W4DownK512M16N64V2CtasPerSm)
void q3x_sm87_a4w4_down_k512_m16n64_v2_m128n128k512_kernel(
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
      *reinterpret_cast<Sm87A4W4DownK512M16N64V2Shared*>(
          dynamic_storage);

  const unsigned int launch_ctas = gridDim.x;
  const unsigned int base_waves = m_tile_count / launch_ctas;
  const unsigned int base_m_tiles = base_waves * launch_ctas;
  const unsigned int residual_m_tiles = m_tile_count - base_m_tiles;
  const unsigned int base_iterations = base_waves * n_tile_count;
  const unsigned int residual_work_tiles =
      residual_m_tiles * n_tile_count;

  for (unsigned int iteration = 0U;; ++iteration) {
    unsigned int m_tile = 0U;
    unsigned int n_tile = 0U;
    if (iteration < base_iterations) {
      const unsigned int wave = iteration / n_tile_count;
      n_tile = iteration - wave * n_tile_count;
      m_tile = wave * launch_ctas + blockIdx.x;
    } else {
      if (residual_m_tiles == 0U) {
        break;
      }
      const unsigned int residual_iteration =
          iteration - base_iterations;
      const unsigned int ordinal =
          blockIdx.x + residual_iteration * launch_ctas;
      if (ordinal >= residual_work_tiles) {
        break;
      }
      n_tile = ordinal / residual_m_tiles;
      m_tile = base_m_tiles + ordinal - n_tile * residual_m_tiles;
    }
    compute_tile(shared, packed_a, a_k512_scales_bf16, packed_b,
                 b_k512_scales_bf16, k512_group_count,
                 physical_k64_group_count, output_bf16,
                 output_row_stride_elements, m_tile, n_tile);
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
          kSm87A4W4DownK512M16N64V2DynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output_properties != nullptr) {
    *output_properties = properties;
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int admit_dynamic_shared() noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      q3x_sm87_a4w4_down_k512_m16n64_v2_m128n128k512_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(
          kSm87A4W4DownK512M16N64V2DynamicSharedBytes)));
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
    const bool require_model_shape,
    void* const cuda_stream) noexcept {
  const Sm87A4W4DownK512M16N64V2Plan plan =
      require_model_shape
          ? sm87_a4w4_down_k512_m16n64_v2_plan(
                token_count, output_size, input_size)
          : sm87_a4w4_down_k512_m16n64_v2_test_plan(
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

  Sm87A4W4DownK512M16N64V2Resources resources{};
  const int resource_status =
      query_sm87_a4w4_down_k512_m16n64_v2_resources_cuda(&resources);
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
  q3x_sm87_a4w4_down_k512_m16n64_v2_m128n128k512_kernel
      <<<launch_ctas,
         static_cast<unsigned int>(
             kSm87A4W4DownK512M16N64V2Threads),
         kSm87A4W4DownK512M16N64V2DynamicSharedBytes, stream>>>(
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

int query_sm87_a4w4_down_k512_m16n64_v2_resources_cuda(
    Sm87A4W4DownK512M16N64V2Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4DownK512M16N64V2Resources{};
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
      q3x_sm87_a4w4_down_k512_m16n64_v2_m128n128k512_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_down_k512_m16n64_v2_m128n128k512_kernel,
      static_cast<int>(kSm87A4W4DownK512M16N64V2Threads),
      kSm87A4W4DownK512M16N64V2DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87A4W4DownK512M16N64V2DynamicSharedBytes;
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
              kSm87A4W4DownK512M16N64V2MaximumRegisters) ||
      resources->static_shared_bytes != 0U ||
      resources->dynamic_shared_bytes !=
          kSm87A4W4DownK512M16N64V2DynamicSharedBytes ||
      resources->configured_dynamic_shared_limit_bytes <
          kSm87A4W4DownK512M16N64V2DynamicSharedBytes ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4DownK512M16N64V2Threads) ||
      resources->active_blocks_per_sm != 1) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_down_k512_m16n64_v2_bf16_cuda(
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
          kSm87A4W4DownK512M16N64V2PersistentCtas),
      true, cuda_stream);
}

int launch_sm87_a4w4_down_k512_m16n64_v2_test_bf16_cuda(
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
