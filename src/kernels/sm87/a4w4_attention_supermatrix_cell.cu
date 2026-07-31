#include "q3x/kernels/sm87_a4w4_attention_supermatrix_cell.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kRequiredSmCount = 16U;
inline constexpr unsigned int kPackedK64Bytes = 32U;
inline constexpr unsigned int kPhysicalK64PerK128 = 2U;
inline constexpr unsigned int kAPlaneBytes =
    kSm87A4W4AttentionCellTileM * kPackedK64Bytes;
inline constexpr unsigned int kBPanelPlaneBytes =
    kSm87A4W4AttentionCellPanelN * kPackedK64Bytes;

struct alignas(16) Sm87A4W4AttentionAStage final {
  std::uint8_t code[kPhysicalK64PerK128][kAPlaneBytes];
  std::uint16_t scale[kSm87A4W4AttentionCellTileM];
};

struct alignas(16) Sm87A4W4AttentionBPairStage final {
  std::uint8_t code[kSm87A4W4AttentionCellPanelsPerCell]
                   [kPhysicalK64PerK128][kBPanelPlaneBytes];
  std::uint16_t scale[kSm87A4W4AttentionCellPanelsPerCell]
                     [kSm87A4W4AttentionCellPanelN];
};

struct alignas(16) Sm87A4W4AttentionShared final {
  Sm87A4W4AttentionAStage a[kSm87A4W4AttentionCellAStages];
  Sm87A4W4AttentionBPairStage
      b[kSm87A4W4AttentionCellBPairStages];
};

static_assert(sizeof(Sm87A4W4AttentionAStage) ==
              kSm87A4W4AttentionCellStageBytes);
static_assert(sizeof(Sm87A4W4AttentionBPairStage) ==
              kSm87A4W4AttentionCellStageBytes);
static_assert(sizeof(Sm87A4W4AttentionShared) ==
              kSm87A4W4AttentionCellSharedBytes);

[[nodiscard]] constexpr bool aligned(const void* const pointer,
                                     const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] constexpr bool product_fits(const std::size_t first,
                                          const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
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

[[nodiscard]] constexpr bool consumer_capacity_fits(
    const std::size_t outer_count,
    const std::size_t physical_k64_group_count) noexcept {
  const std::size_t outer_blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return outer_blocks != 0U && physical_k64_group_count != 0U &&
         product_fits(outer_blocks, physical_k64_group_count) &&
         product_fits(outer_blocks * physical_k64_group_count,
                      kSm87A4W4ConsumerOuterBlock) &&
         product_fits(outer_blocks * physical_k64_group_count *
                          kSm87A4W4ConsumerOuterBlock,
                      kSm87A4W4ConsumerPackedKBlockBytes);
}

[[nodiscard]] constexpr bool shared_scale_capacity_fits(
    const std::size_t outer_count,
    const std::size_t k128_group_count) noexcept {
  const std::size_t outer_blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return outer_blocks != 0U && k128_group_count != 0U &&
         product_fits(outer_blocks, k128_group_count) &&
         product_fits(outer_blocks * k128_group_count,
                      kSm87A4W4ConsumerOuterBlock);
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

__device__ __forceinline__ void issue_a_stage(
    Sm87A4W4AttentionAStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr unsigned int kVectorsPerPhysicalK64 = kAPlaneBytes / 16U;
  constexpr unsigned int kCodeVectors =
      kPhysicalK64PerK128 * kVectorsPerPhysicalK64;
  constexpr unsigned int kScaleVectors =
      kSm87A4W4AttentionCellTileM * sizeof(std::uint16_t) / 16U;
  static_assert(kCodeVectors == 2U * kSm87A4W4AttentionCellThreads);
  static_assert(kScaleVectors == 16U);

  for (unsigned int vector = threadIdx.x; vector < kCodeVectors;
       vector += blockDim.x) {
    const unsigned int half = vector / kVectorsPerPhysicalK64;
    const unsigned int half_vector =
        vector - half * kVectorsPerPhysicalK64;
    const unsigned int row = half_vector / 2U;
    const unsigned int row_vector = half_vector % 2U;
    const unsigned int physical_group = 2U * k128_group + half;
    cp_async_16(
        stage.code[half] + sm87_a4w4_swizzled_k64_byte_offset(
                               row, 16U * row_vector),
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_group, 16U * row_vector,
                       physical_k64_group_count));
  }
  if (threadIdx.x < kScaleVectors) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.scale + first_row,
        a_k128_scales_bf16 + sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + first_row,
            k128_group, k128_group_count));
  }
}

__device__ __forceinline__ void issue_b_panel(
    std::uint8_t (&stage_code)[kPhysicalK64PerK128][kBPanelPlaneBytes],
    std::uint16_t (&stage_scale)[kSm87A4W4AttentionCellPanelN],
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int panel,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr unsigned int kVectorsPerPhysicalK64 =
      kBPanelPlaneBytes / 16U;
  constexpr unsigned int kCodeVectors =
      kPhysicalK64PerK128 * kVectorsPerPhysicalK64;
  constexpr unsigned int kScaleVectors =
      kSm87A4W4AttentionCellPanelN * sizeof(std::uint16_t) / 16U;
  static_assert(kCodeVectors == kSm87A4W4AttentionCellThreads);
  static_assert(kScaleVectors == 8U);

  const unsigned int vector = threadIdx.x;
  const unsigned int half = vector / kVectorsPerPhysicalK64;
  const unsigned int half_vector =
      vector - half * kVectorsPerPhysicalK64;
  const unsigned int row = half_vector / 2U;
  const unsigned int row_vector = half_vector % 2U;
  const unsigned int physical_group = 2U * k128_group + half;
  const unsigned int n_start =
      panel * kSm87A4W4AttentionCellPanelN;
  cp_async_16(
      stage_code[half] + sm87_a4w4_swizzled_k64_byte_offset(
                               row, 16U * row_vector),
      packed_b + sm87_a4w4_consumer_packed_offset(
                     static_cast<std::size_t>(n_start) + row,
                     physical_group, 16U * row_vector,
                     physical_k64_group_count));
  if (threadIdx.x < kScaleVectors) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage_scale + first_row,
        b_k128_scales_bf16 + sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(n_start) + first_row,
            k128_group, k128_group_count));
  }
}

__device__ __forceinline__ void issue_b_pair_stage(
    Sm87A4W4AttentionBPairStage& stage,
    const std::uint8_t* const packed_b0,
    const std::uint16_t* const b0_k128_scales_bf16,
    const unsigned int panel0,
    const std::uint8_t* const packed_b1,
    const std::uint16_t* const b1_k128_scales_bf16,
    const unsigned int panel1,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  issue_b_panel(stage.code[0U], stage.scale[0U], packed_b0,
                b0_k128_scales_bf16, panel0, k128_group,
                physical_k64_group_count, k128_group_count);
  issue_b_panel(stage.code[1U], stage.scale[1U], packed_b1,
                b1_k128_scales_bf16, panel1, k128_group,
                physical_k64_group_count, k128_group_count);
}

__device__ __forceinline__ void issue_group(
    Sm87A4W4AttentionShared& shared,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_b0,
    const std::uint16_t* const b0_k128_scales_bf16,
    const unsigned int panel0,
    const std::uint8_t* const packed_b1,
    const std::uint16_t* const b1_k128_scales_bf16,
    const unsigned int panel1,
    const unsigned int m_tile_start,
    const unsigned int group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  issue_a_stage(shared.a[group % kSm87A4W4AttentionCellAStages],
                packed_a, a_k128_scales_bf16, m_tile_start, group,
                physical_k64_group_count, k128_group_count);
  issue_b_pair_stage(
      shared.b[group % kSm87A4W4AttentionCellBPairStages],
      packed_b0, b0_k128_scales_bf16, panel0, packed_b1,
      b1_k128_scales_bf16, panel1, group,
      physical_k64_group_count, k128_group_count);
  cp_async_commit();
}

__device__ __forceinline__ void accumulate_group(
    const Sm87A4W4AttentionAStage& a_stage,
    const Sm87A4W4AttentionBPairStage& b_stage,
    float (&accumulators)[2U][8U][4U]) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int fragment_m_start = warp * 16U;
  const Sm87A4W4AFragment a_fragments[kPhysicalK64PerK128] = {
      sm87_a4w4_load_a_fragment_swizzled_shared(
          a_stage.code[0U] + fragment_m_start * kPackedK64Bytes, lane),
      sm87_a4w4_load_a_fragment_swizzled_shared(
          a_stage.code[1U] + fragment_m_start * kPackedK64Bytes, lane)};
  const unsigned int local_m0 = fragment_m_start + lane / 4U;
  const unsigned int local_m1 = local_m0 + 8U;
  const float a_scale0 = decode_bf16(a_stage.scale[local_m0]);
  const float a_scale1 = decode_bf16(a_stage.scale[local_m1]);

#pragma unroll
  for (unsigned int panel = 0U;
       panel < kSm87A4W4AttentionCellPanelsPerCell; ++panel) {
#pragma unroll
    for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
      const unsigned int fragment_n_start = fragment_n * 8U;
      Sm87A4W4Accumulator partial{};
#pragma unroll
      for (unsigned int half = 0U; half < kPhysicalK64PerK128; ++half) {
        const Sm87A4W4BFragment b_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                b_stage.code[panel][half] +
                    fragment_n_start * kPackedK64Bytes,
                lane);
        sm87_a4w4_mma_m16n8k64(partial, a_fragments[half], b_fragment);
      }
      const unsigned int local_n0 =
          fragment_n_start + 2U * (lane % 4U);
      const unsigned int local_n1 = local_n0 + 1U;
      const float b_scale0 =
          decode_bf16(b_stage.scale[panel][local_n0]);
      const float b_scale1 =
          decode_bf16(b_stage.scale[panel][local_n1]);
      accumulators[panel][fragment_n][0U] +=
          static_cast<float>(partial.x0) * (a_scale0 * b_scale0);
      accumulators[panel][fragment_n][1U] +=
          static_cast<float>(partial.x1) * (a_scale0 * b_scale1);
      accumulators[panel][fragment_n][2U] +=
          static_cast<float>(partial.x2) * (a_scale1 * b_scale0);
      accumulators[panel][fragment_n][3U] +=
          static_cast<float>(partial.x3) * (a_scale1 * b_scale1);
    }
  }
}

__device__ __forceinline__ void wait_for_group(
    const unsigned int group,
    const unsigned int group_count) noexcept {
  if (group == 0U) {
    if (group_count >= 3U) {
      cp_async_wait<2U>();
    } else if (group_count == 2U) {
      cp_async_wait<1U>();
    } else {
      cp_async_wait<0U>();
    }
  } else if (group == 1U) {
    if (group + 2U < group_count) {
      cp_async_wait<3U>();
    } else if (group + 1U < group_count) {
      cp_async_wait<2U>();
    } else {
      cp_async_wait<0U>();
    }
  } else if (group + 2U < group_count) {
    cp_async_wait<3U>();
  } else if (group + 1U < group_count) {
    cp_async_wait<2U>();
  } else {
    cp_async_wait<0U>();
  }
}

template <unsigned int FirstPanels, unsigned int SecondPanels>
__device__ __forceinline__ void select_pair(
    const unsigned int pair_cell,
    const std::uint8_t* const packed_first_b,
    const std::uint16_t* const first_b_k128_scales_bf16,
    const std::uint8_t* const packed_second_b,
    const std::uint16_t* const second_b_k128_scales_bf16,
    const std::uint8_t*& packed_b0,
    const std::uint16_t*& b0_k128_scales_bf16,
    unsigned int& panel0,
    const std::uint8_t*& packed_b1,
    const std::uint16_t*& b1_k128_scales_bf16,
    unsigned int& panel1) noexcept {
  constexpr unsigned int paired =
      FirstPanels < SecondPanels ? FirstPanels : SecondPanels;
  if (pair_cell < paired) {
    packed_b0 = packed_first_b;
    b0_k128_scales_bf16 = first_b_k128_scales_bf16;
    panel0 = pair_cell;
    packed_b1 = packed_second_b;
    b1_k128_scales_bf16 = second_b_k128_scales_bf16;
    panel1 = pair_cell;
    return;
  }
  const unsigned int remaining_cell = pair_cell - paired;
  if constexpr (FirstPanels > SecondPanels) {
    packed_b0 = packed_first_b;
    b0_k128_scales_bf16 = first_b_k128_scales_bf16;
    panel0 = paired + 2U * remaining_cell;
    packed_b1 = packed_first_b;
    b1_k128_scales_bf16 = first_b_k128_scales_bf16;
    panel1 = panel0 + 1U;
  } else {
    packed_b0 = packed_second_b;
    b0_k128_scales_bf16 = second_b_k128_scales_bf16;
    panel0 = paired + 2U * remaining_cell;
    packed_b1 = packed_second_b;
    b1_k128_scales_bf16 = second_b_k128_scales_bf16;
    panel1 = panel0 + 1U;
  }
}

template <unsigned int FirstPanels, unsigned int SecondPanels>
__device__ __forceinline__ void store_panel(
    const unsigned int pair_cell, const unsigned int slot,
    const unsigned int m_tile_start,
    const float (&accumulators)[2U][8U][4U],
    std::uint16_t* const first_output_bf16,
    const unsigned int first_output_row_stride_elements,
    std::uint16_t* const second_output_bf16,
    const unsigned int second_output_row_stride_elements) noexcept {
  constexpr unsigned int paired =
      FirstPanels < SecondPanels ? FirstPanels : SecondPanels;
  bool second_projection = false;
  unsigned int panel = 0U;
  if (pair_cell < paired) {
    second_projection = slot == 1U;
    panel = pair_cell;
  } else {
    const unsigned int remaining_cell = pair_cell - paired;
    second_projection = FirstPanels <= SecondPanels;
    panel = paired + 2U * remaining_cell + slot;
  }
  std::uint16_t* const output =
      second_projection ? second_output_bf16 : first_output_bf16;
  const unsigned int stride = second_projection
                                  ? second_output_row_stride_elements
                                  : first_output_row_stride_elements;
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
#pragma unroll
  for (unsigned int fragment_n = 0U; fragment_n < 8U; ++fragment_n) {
#pragma unroll
    for (unsigned int row_pair = 0U; row_pair < 2U; ++row_pair) {
      const unsigned int value = 2U * row_pair;
      const Sm87A4W4AccumulatorCoordinate coordinate =
          sm87_a4w4_accumulator_coordinate(lane, value);
      const unsigned int global_m =
          m_tile_start + warp * 16U + coordinate.m;
      const unsigned int global_n =
          panel * kSm87A4W4AttentionCellPanelN +
          fragment_n * 8U + coordinate.n;
      const std::uint32_t output_pair =
          static_cast<std::uint32_t>(
              encode_bf16(accumulators[slot][fragment_n][value])) |
          (static_cast<std::uint32_t>(
               encode_bf16(accumulators[slot][fragment_n][value + 1U]))
           << 16U);
      *reinterpret_cast<std::uint32_t*>(
          output + static_cast<std::size_t>(global_m) * stride + global_n) =
          output_pair;
    }
  }
}

template <unsigned int FirstPanels, unsigned int SecondPanels>
__global__ __launch_bounds__(kSm87A4W4AttentionCellThreads,
                             kSm87A4W4AttentionCellCtasPerSm)
void q3x_sm87_a4w4_attention_pair_supermatrix_m128n64x2k128_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_first_b,
    const std::uint16_t* const first_b_k128_scales_bf16,
    const std::uint8_t* const packed_second_b,
    const std::uint16_t* const second_b_k128_scales_bf16,
    const unsigned int k128_group_count,
    const unsigned int physical_k64_group_count,
    std::uint16_t* const first_output_bf16,
    const unsigned int first_output_row_stride_elements,
    std::uint16_t* const second_output_bf16,
    const unsigned int second_output_row_stride_elements,
    const unsigned int m_tile_count,
    const unsigned int work_cell_count) {
  __shared__ Sm87A4W4AttentionShared shared;

  for (unsigned int work_cell = blockIdx.x; work_cell < work_cell_count;
       work_cell += gridDim.x) {
    const unsigned int pair_cell = work_cell / m_tile_count;
    const unsigned int m_tile = work_cell - pair_cell * m_tile_count;
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4AttentionCellTileM;
    const std::uint8_t* packed_b0 = nullptr;
    const std::uint16_t* b0_scales = nullptr;
    const std::uint8_t* packed_b1 = nullptr;
    const std::uint16_t* b1_scales = nullptr;
    unsigned int panel0 = 0U;
    unsigned int panel1 = 0U;
    select_pair<FirstPanels, SecondPanels>(
        pair_cell, packed_first_b, first_b_k128_scales_bf16,
        packed_second_b, second_b_k128_scales_bf16,
        packed_b0, b0_scales, panel0, packed_b1, b1_scales, panel1);

    float accumulators[2U][8U][4U]{};
    issue_group(shared, packed_a, a_k128_scales_bf16,
                packed_b0, b0_scales, panel0, packed_b1, b1_scales,
                panel1, m_tile_start, 0U,
                physical_k64_group_count, k128_group_count);
    if (k128_group_count > 1U) {
      issue_group(shared, packed_a, a_k128_scales_bf16,
                  packed_b0, b0_scales, panel0, packed_b1, b1_scales,
                  panel1, m_tile_start, 1U,
                  physical_k64_group_count, k128_group_count);
    }
    if (k128_group_count > 2U) {
      issue_b_pair_stage(shared.b[2U], packed_b0, b0_scales, panel0,
                         packed_b1, b1_scales, panel1, 2U,
                         physical_k64_group_count, k128_group_count);
      cp_async_commit();
    }

    for (unsigned int group = 0U; group < k128_group_count; ++group) {
      wait_for_group(group, k128_group_count);
      __syncthreads();
      accumulate_group(shared.a[group % kSm87A4W4AttentionCellAStages],
                       shared.b[group %
                                kSm87A4W4AttentionCellBPairStages],
                       accumulators);
      __syncthreads();

      if (group + kSm87A4W4AttentionCellAStages < k128_group_count) {
        const unsigned int future_a =
            group + kSm87A4W4AttentionCellAStages;
        issue_a_stage(
            shared.a[future_a % kSm87A4W4AttentionCellAStages],
            packed_a, a_k128_scales_bf16, m_tile_start, future_a,
            physical_k64_group_count, k128_group_count);
        cp_async_commit();
      }
      if (group + kSm87A4W4AttentionCellBPairStages <
          k128_group_count) {
        const unsigned int future_b =
            group + kSm87A4W4AttentionCellBPairStages;
        issue_b_pair_stage(
            shared.b[future_b % kSm87A4W4AttentionCellBPairStages],
            packed_b0, b0_scales, panel0, packed_b1, b1_scales,
            panel1, future_b, physical_k64_group_count,
            k128_group_count);
        cp_async_commit();
      }
    }

    store_panel<FirstPanels, SecondPanels>(
        pair_cell, 0U, m_tile_start, accumulators,
        first_output_bf16, first_output_row_stride_elements,
        second_output_bf16, second_output_row_stride_elements);
    store_panel<FirstPanels, SecondPanels>(
        pair_cell, 1U, m_tile_start, accumulators,
        first_output_bf16, first_output_row_stride_elements,
        second_output_bf16, second_output_row_stride_elements);
    __syncthreads();
  }
}

[[nodiscard]] int validate_sm87(
    cudaDeviceProp* const properties = nullptr) noexcept {
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaDeviceProp local{};
  status = cudaGetDeviceProperties(&local, device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (local.major != kSm87A4W4RequiredComputeMajor ||
      local.minor != kSm87A4W4RequiredComputeMinor ||
      local.multiProcessorCount != static_cast<int>(kRequiredSmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

template <unsigned int FirstPanels, unsigned int SecondPanels>
[[nodiscard]] int launch_pair(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_first_b,
    const std::size_t packed_first_b_capacity_bytes,
    const std::uint16_t* const first_b_k128_scales_bf16,
    const std::size_t first_b_scale_capacity_elements,
    const std::uint8_t* const packed_second_b,
    const std::size_t packed_second_b_capacity_bytes,
    const std::uint16_t* const second_b_k128_scales_bf16,
    const std::size_t second_b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t input_size,
    std::uint16_t* const first_output_bf16,
    const std::size_t first_output_row_stride_elements,
    const std::size_t first_output_capacity_elements,
    std::uint16_t* const second_output_bf16,
    const std::size_t second_output_row_stride_elements,
    const std::size_t second_output_capacity_elements,
    void* const cuda_stream) noexcept {
  constexpr std::size_t first_output_size =
      static_cast<std::size_t>(FirstPanels) *
      kSm87A4W4AttentionCellPanelN;
  constexpr std::size_t second_output_size =
      static_cast<std::size_t>(SecondPanels) *
      kSm87A4W4AttentionCellPanelN;
  const Sm87A4W4AttentionSupermatrixPlan plan =
      sm87_a4w4_attention_supermatrix_plan(
          token_count, first_output_size, second_output_size, input_size);
  if (plan.launch_ctas == 0U || !aligned(packed_a, 16U) ||
      !aligned(a_k128_scales_bf16, 16U) ||
      !aligned(packed_first_b, 16U) ||
      !aligned(first_b_k128_scales_bf16, 16U) ||
      !aligned(packed_second_b, 16U) ||
      !aligned(second_b_k128_scales_bf16, 16U) ||
      !aligned(first_output_bf16, alignof(std::uint32_t)) ||
      !aligned(second_output_bf16, alignof(std::uint32_t)) ||
      first_output_bf16 == second_output_bf16 ||
      first_output_row_stride_elements < first_output_size ||
      second_output_row_stride_elements < second_output_size ||
      first_output_row_stride_elements % 2U != 0U ||
      second_output_row_stride_elements % 2U != 0U ||
      !product_fits(token_count, first_output_row_stride_elements) ||
      !product_fits(token_count, second_output_row_stride_elements) ||
      !consumer_capacity_fits(token_count, plan.physical_k64_groups) ||
      !consumer_capacity_fits(first_output_size,
                              plan.physical_k64_groups) ||
      !consumer_capacity_fits(second_output_size,
                              plan.physical_k64_groups) ||
      !shared_scale_capacity_fits(token_count, plan.k128_groups) ||
      !shared_scale_capacity_fits(first_output_size, plan.k128_groups) ||
      !shared_scale_capacity_fits(second_output_size, plan.k128_groups) ||
      plan.k128_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.pair_cells > std::numeric_limits<unsigned int>::max() ||
      plan.work_cells > std::numeric_limits<unsigned int>::max() ||
      plan.launch_ctas > std::numeric_limits<unsigned int>::max() ||
      first_output_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      second_output_row_stride_elements >
          std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_first_output_elements =
      token_count * first_output_row_stride_elements;
  const std::size_t required_second_output_elements =
      token_count * second_output_row_stride_elements;
  if (!product_fits(required_first_output_elements,
                    sizeof(std::uint16_t)) ||
      !product_fits(required_second_output_elements,
                    sizeof(std::uint16_t)) ||
      first_output_capacity_elements < required_first_output_elements ||
      second_output_capacity_elements < required_second_output_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_first_output_bytes =
      required_first_output_elements * sizeof(std::uint16_t);
  const std::size_t required_second_output_bytes =
      required_second_output_elements * sizeof(std::uint16_t);
  if (byte_ranges_overlap(first_output_bf16,
                          required_first_output_bytes,
                          second_output_bf16,
                          required_second_output_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_consumer_k128_scale_capacity_elements(
          token_count, input_size);
  const std::size_t required_first_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(first_output_size,
                                                input_size);
  const std::size_t required_first_b_scales =
      sm87_a4w4_consumer_k128_scale_capacity_elements(first_output_size,
                                                       input_size);
  const std::size_t required_second_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(second_output_size,
                                                input_size);
  const std::size_t required_second_b_scales =
      sm87_a4w4_consumer_k128_scale_capacity_elements(second_output_size,
                                                       input_size);
  if (required_a_bytes == 0U || required_a_scales == 0U ||
      required_first_b_bytes == 0U || required_first_b_scales == 0U ||
      required_second_b_bytes == 0U || required_second_b_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      packed_first_b_capacity_bytes < required_first_b_bytes ||
      first_b_scale_capacity_elements < required_first_b_scales ||
      packed_second_b_capacity_bytes < required_second_b_bytes ||
      second_b_scale_capacity_elements < required_second_b_scales ||
      !product_fits(required_a_scales, sizeof(std::uint16_t)) ||
      !product_fits(required_first_b_scales, sizeof(std::uint16_t)) ||
      !product_fits(required_second_b_scales, sizeof(std::uint16_t))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_scale_bytes =
      required_a_scales * sizeof(std::uint16_t);
  const std::size_t required_first_b_scale_bytes =
      required_first_b_scales * sizeof(std::uint16_t);
  const std::size_t required_second_b_scale_bytes =
      required_second_b_scales * sizeof(std::uint16_t);
  const auto output_overlaps_input = [&](const void* const input,
                                         const std::size_t input_bytes) {
    return byte_ranges_overlap(first_output_bf16,
                               required_first_output_bytes,
                               input, input_bytes) ||
           byte_ranges_overlap(second_output_bf16,
                               required_second_output_bytes,
                               input, input_bytes);
  };
  if (output_overlaps_input(packed_a, required_a_bytes) ||
      output_overlaps_input(a_k128_scales_bf16,
                            required_a_scale_bytes) ||
      output_overlaps_input(packed_first_b, required_first_b_bytes) ||
      output_overlaps_input(first_b_k128_scales_bf16,
                            required_first_b_scale_bytes) ||
      output_overlaps_input(packed_second_b, required_second_b_bytes) ||
      output_overlaps_input(second_b_k128_scales_bf16,
                            required_second_b_scale_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int target_status = validate_sm87();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_attention_pair_supermatrix_m128n64x2k128_kernel<
      FirstPanels, SecondPanels>
      <<<static_cast<unsigned int>(plan.launch_ctas),
         static_cast<unsigned int>(kSm87A4W4AttentionCellThreads), 0U,
         stream>>>(
          packed_a, a_k128_scales_bf16, packed_first_b,
          first_b_k128_scales_bf16, packed_second_b,
          second_b_k128_scales_bf16,
          static_cast<unsigned int>(plan.k128_groups),
          static_cast<unsigned int>(plan.physical_k64_groups),
          first_output_bf16,
          static_cast<unsigned int>(first_output_row_stride_elements),
          second_output_bf16,
          static_cast<unsigned int>(second_output_row_stride_elements),
          static_cast<unsigned int>(plan.m_tiles),
          static_cast<unsigned int>(plan.work_cells));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

int query_sm87_a4w4_linear_qkv_z_supermatrix_resources_cuda(
    Sm87A4W4AttentionSupermatrixResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4AttentionSupermatrixResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_sm87(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const auto kernel =
      q3x_sm87_a4w4_attention_pair_supermatrix_m128n64x2k128_kernel<
          static_cast<unsigned int>(kQwen36LinearQkvPanels),
          static_cast<unsigned int>(kQwen36LinearZPanels)>;
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel,
      static_cast<int>(kSm87A4W4AttentionCellThreads), 0U);
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

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(kSm87A4W4AttentionCellMaximumRegisters) ||
      resources->static_shared_bytes !=
          kSm87A4W4AttentionCellSharedBytes ||
      resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4AttentionCellCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4AttentionCellThreads)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_linear_qkv_z_supermatrix_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_qkv_b,
    const std::size_t packed_qkv_b_capacity_bytes,
    const std::uint16_t* const qkv_b_k128_scales_bf16,
    const std::size_t qkv_b_scale_capacity_elements,
    const std::uint8_t* const packed_z_b,
    const std::size_t packed_z_b_capacity_bytes,
    const std::uint16_t* const z_b_k128_scales_bf16,
    const std::size_t z_b_scale_capacity_elements,
    const std::size_t token_count,
    std::uint16_t* const qkv_output_bf16,
    const std::size_t qkv_output_row_stride_elements,
    const std::size_t qkv_output_capacity_elements,
    std::uint16_t* const z_output_bf16,
    const std::size_t z_output_row_stride_elements,
    const std::size_t z_output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_pair<
      static_cast<unsigned int>(kQwen36LinearQkvPanels),
      static_cast<unsigned int>(kQwen36LinearZPanels)>(
      packed_a, packed_a_capacity_bytes, a_k128_scales_bf16,
      a_scale_capacity_elements, packed_qkv_b,
      packed_qkv_b_capacity_bytes, qkv_b_k128_scales_bf16,
      qkv_b_scale_capacity_elements, packed_z_b,
      packed_z_b_capacity_bytes, z_b_k128_scales_bf16,
      z_b_scale_capacity_elements, token_count,
      kQwen36AttentionInputSize, qkv_output_bf16,
      qkv_output_row_stride_elements, qkv_output_capacity_elements,
      z_output_bf16, z_output_row_stride_elements,
      z_output_capacity_elements, cuda_stream);
}

int launch_sm87_a4w4_attention_pair_supermatrix_test_bf16_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_first_b,
    const std::size_t packed_first_b_capacity_bytes,
    const std::uint16_t* const first_b_k128_scales_bf16,
    const std::size_t first_b_scale_capacity_elements,
    const std::uint8_t* const packed_second_b,
    const std::size_t packed_second_b_capacity_bytes,
    const std::uint16_t* const second_b_k128_scales_bf16,
    const std::size_t second_b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t input_size,
    std::uint16_t* const first_output_bf16,
    const std::size_t first_output_row_stride_elements,
    const std::size_t first_output_capacity_elements,
    std::uint16_t* const second_output_bf16,
    const std::size_t second_output_row_stride_elements,
    const std::size_t second_output_capacity_elements,
    void* const cuda_stream) noexcept {
  return launch_pair<4U, 2U>(
      packed_a, packed_a_capacity_bytes, a_k128_scales_bf16,
      a_scale_capacity_elements, packed_first_b,
      packed_first_b_capacity_bytes, first_b_k128_scales_bf16,
      first_b_scale_capacity_elements, packed_second_b,
      packed_second_b_capacity_bytes, second_b_k128_scales_bf16,
      second_b_scale_capacity_elements, token_count, input_size,
      first_output_bf16, first_output_row_stride_elements,
      first_output_capacity_elements, second_output_bf16,
      second_output_row_stride_elements, second_output_capacity_elements,
      cuda_stream);
}

}  // namespace q3x::kernels
