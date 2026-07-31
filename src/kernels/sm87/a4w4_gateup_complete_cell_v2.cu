#include "q3x/kernels/sm87_a4w4_gateup_complete_cell_v2.h"

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
    kSm87A4W4GateUpCellV2TileM * kPackedK64Bytes;
inline constexpr unsigned int kBPlaneBytes =
    kSm87A4W4GateUpCellV2PhaseN * kPackedK64Bytes;

struct alignas(16) Sm87A4W4GateUpCellV2AStage final {
  std::uint8_t a[kPhysicalK64PerK128][kAPlaneBytes];
  std::uint16_t a_scales[kSm87A4W4GateUpCellV2TileM];
};

struct alignas(16) Sm87A4W4GateUpCellV2BPhase final {
  std::uint8_t b[kPhysicalK64PerK128][kBPlaneBytes];
  std::uint16_t b_scales[kSm87A4W4GateUpCellV2PhaseN];
};

struct alignas(16) Sm87A4W4GateUpCellV2Pipeline final {
  Sm87A4W4GateUpCellV2AStage a[kSm87A4W4GateUpCellV2AStages];
  Sm87A4W4GateUpCellV2BPhase b[kSm87A4W4GateUpCellV2BPhaseStages];
};

union alignas(16) Sm87A4W4GateUpCellV2PhaseOneWorkspace final {
  Sm87A4W4GateUpCellV2Pipeline pipeline;
  float product[kSm87A4W4GateUpCellV2TileM]
               [kSm87A4W4GateUpCellV2PhaseN];
};

struct alignas(16) Sm87A4W4GateUpCellV2Shared final {
  float phase_zero_product[kSm87A4W4GateUpCellV2TileM]
                          [kSm87A4W4GateUpCellV2PhaseN];
  Sm87A4W4GateUpCellV2PhaseOneWorkspace phase_one;
};

static_assert(sizeof(Sm87A4W4GateUpCellV2AStage) == 4'224U);
static_assert(sizeof(Sm87A4W4GateUpCellV2BPhase) == 4'224U);
static_assert(sizeof(Sm87A4W4GateUpCellV2Pipeline) == 25'344U);
static_assert(sizeof(Sm87A4W4GateUpCellV2PhaseOneWorkspace) == 25'344U);
static_assert(sizeof(Sm87A4W4GateUpCellV2Shared) ==
              kSm87A4W4GateUpCellV2SharedBytes);
static_assert(sizeof(Sm87A4W4GateUpCellV2Shared) *
                  kSm87A4W4GateUpCellV2CtasPerSm <=
              96U * 1'024U);

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
  // Codes and scales are one-use CTA data.  Bypassing L1 prevents the two
  // weight phases from evicting each other while retaining L2 reuse across
  // persistent M phases.
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

__device__ __forceinline__ void prefetch_a_stage(
    Sm87A4W4GateUpCellV2AStage& stage,
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
      kSm87A4W4GateUpCellV2TileM * sizeof(std::uint16_t) / 16U;
  static_assert(kCodeVectors == kSm87A4W4GateUpCellV2Threads);
  static_assert(kScaleVectors == 8U);

  if (threadIdx.x < kCodeVectors) {
    const unsigned int vector = threadIdx.x;
    const unsigned int half = vector / kVectorsPerPhysicalK64;
    const unsigned int half_vector =
        vector - half * kVectorsPerPhysicalK64;
    const unsigned int row = half_vector / 2U;
    const unsigned int row_vector = half_vector % 2U;
    const unsigned int physical_group = 2U * k128_group + half;
    cp_async_16(
        stage.a[half] + sm87_a4w4_swizzled_k64_byte_offset(
                            row, 16U * row_vector),
        packed_a + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(m_tile_start) + row,
                       physical_group, 16U * row_vector,
                       physical_k64_group_count));
  }

  if (threadIdx.x < kScaleVectors) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.a_scales + first_row,
        a_k128_scales_bf16 + sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + first_row,
            k128_group, k128_group_count));
  }
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_b_phase(
    Sm87A4W4GateUpCellV2BPhase& stage,
    const std::uint8_t* const packed_b,
    const std::uint16_t* const b_k128_scales_bf16,
    const unsigned int n_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  constexpr unsigned int kVectorsPerPhysicalK64 = kBPlaneBytes / 16U;
  constexpr unsigned int kCodeVectors =
      kPhysicalK64PerK128 * kVectorsPerPhysicalK64;
  constexpr unsigned int kScaleVectors =
      kSm87A4W4GateUpCellV2PhaseN * sizeof(std::uint16_t) / 16U;
  static_assert(kCodeVectors == kSm87A4W4GateUpCellV2Threads);
  static_assert(kScaleVectors == 8U);

  const unsigned int vector = threadIdx.x;
  const unsigned int half = vector / kVectorsPerPhysicalK64;
  const unsigned int half_vector =
      vector - half * kVectorsPerPhysicalK64;
  const unsigned int row = half_vector / 2U;
  const unsigned int row_vector = half_vector % 2U;
  const unsigned int physical_group = 2U * k128_group + half;
  cp_async_16(
      stage.b[half] + sm87_a4w4_swizzled_k64_byte_offset(
                          row, 16U * row_vector),
      packed_b + sm87_a4w4_consumer_packed_offset(
                     static_cast<std::size_t>(n_tile_start) + row,
                     physical_group, 16U * row_vector,
                     physical_k64_group_count));

  if (threadIdx.x < kScaleVectors) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.b_scales + first_row,
        b_k128_scales_bf16 + sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(n_tile_start) + first_row,
            k128_group, k128_group_count));
  }
  cp_async_commit();
}

__device__ __forceinline__ void prefetch_logical_group(
    Sm87A4W4GateUpCellV2Pipeline& pipeline,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int k128_group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  const unsigned int logical_slot =
      k128_group % kSm87A4W4GateUpCellV2AStages;
  const unsigned int gate_slot = 2U * logical_slot;
  prefetch_a_stage(
      pipeline.a[logical_slot], packed_a, a_k128_scales_bf16,
      m_tile_start, k128_group, physical_k64_group_count,
      k128_group_count);
  prefetch_b_phase(
      pipeline.b[gate_slot], packed_gate_b,
      gate_b_k128_scales_bf16, n_tile_start, k128_group,
      physical_k64_group_count, k128_group_count);
  prefetch_b_phase(
      pipeline.b[gate_slot + 1U], packed_up_b,
      up_b_k128_scales_bf16, n_tile_start, k128_group,
      physical_k64_group_count, k128_group_count);
}

__device__ __forceinline__ void accumulate_phase(
    const Sm87A4W4GateUpCellV2AStage& a_stage,
    const Sm87A4W4GateUpCellV2BPhase& b_stage,
    float (&accumulators)[1U][4U][4U]) noexcept {
  // These descriptors are intentionally rematerialized for each projection
  // phase.  Keeping them live across both 32-register accumulator sets made
  // ptxas preserve address state for the entire K loop.
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int warp_m = warp / 2U;
  const unsigned int warp_n = warp % 2U;
#pragma unroll
  for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
    const unsigned int fragment_n = warp_n * 32U + fragment * 8U;
    const unsigned int local_n0 =
        fragment_n + 2U * (lane % 4U);
    const unsigned int local_n1 = local_n0 + 1U;
    const float b_scale0 = decode_bf16(b_stage.b_scales[local_n0]);
    const float b_scale1 = decode_bf16(b_stage.b_scales[local_n1]);
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < 1U; ++m_panel) {
      const unsigned int panel_m = warp_m * 16U;
      Sm87A4W4Accumulator partial{};
#pragma unroll
      for (unsigned int half = 0U; half < kPhysicalK64PerK128; ++half) {
        const Sm87A4W4AFragment a_fragment =
            sm87_a4w4_load_a_fragment_swizzled_shared(
                a_stage.a[half] + panel_m * kPackedK64Bytes, lane);
        const Sm87A4W4BFragment b_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(
                b_stage.b[half] + fragment_n * kPackedK64Bytes, lane);
        sm87_a4w4_mma_m16n8k64(partial, a_fragment, b_fragment);
      }
      const unsigned int local_m0 = panel_m + lane / 4U;
      const unsigned int local_m1 = local_m0 + 8U;
      const float a_scale0 = decode_bf16(a_stage.a_scales[local_m0]);
      const float a_scale1 = decode_bf16(a_stage.a_scales[local_m1]);
      accumulators[m_panel][fragment][0U] +=
          static_cast<float>(partial.x0) * (a_scale0 * b_scale0);
      accumulators[m_panel][fragment][1U] +=
          static_cast<float>(partial.x1) * (a_scale0 * b_scale1);
      accumulators[m_panel][fragment][2U] +=
          static_cast<float>(partial.x2) * (a_scale1 * b_scale0);
      accumulators[m_panel][fragment][3U] +=
          static_cast<float>(partial.x3) * (a_scale1 * b_scale1);
    }
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpCellV2Threads,
                      kSm87A4W4GateUpCellV2CtasPerSm)
void q3x_sm87_a4w4_gateup_complete_cell_v2_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const unsigned int k128_group_count,
    const unsigned int physical_k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k128_scales_bf16,
    const unsigned int m_tile_count,
    const unsigned int n_phase_count,
    const unsigned int output_physical_k64_group_count) {
  __shared__ Sm87A4W4GateUpCellV2Shared shared;

  // On the target P2048 shape gridDim.x==m_tile_count==32.  A persistent CTA
  // therefore owns one M64 phase for all N128 cells; smaller contract shapes
  // retain the same mapping without correctness depending on launch order.
  for (unsigned int m_tile = blockIdx.x; m_tile < m_tile_count;
       m_tile += gridDim.x) {
    for (unsigned int n_phase = 0U; n_phase < n_phase_count; ++n_phase) {
      // Split one K128 output group into two N64 compute phases.  Only the
      // first product tile coexists with the second phase's copy pipeline.
#pragma unroll
      for (unsigned int output_half = 0U; output_half < 2U;
           ++output_half) {
        float gate_accumulators[1U][4U][4U]{};
        float up_accumulators[1U][4U][4U]{};
        const unsigned int phase_n_start =
            n_phase * kSm87A4W4GateUpCellV2TileN +
            output_half * kSm87A4W4GateUpCellV2PhaseN;

        prefetch_logical_group(
            shared.phase_one.pipeline, packed_a, a_k128_scales_bf16,
            packed_gate_b, gate_b_k128_scales_bf16, packed_up_b,
            up_b_k128_scales_bf16,
            m_tile * kSm87A4W4GateUpCellV2TileM,
            phase_n_start, 0U,
            physical_k64_group_count, k128_group_count);
        if (k128_group_count > 1U) {
          prefetch_logical_group(
              shared.phase_one.pipeline, packed_a,
              a_k128_scales_bf16,
              packed_gate_b, gate_b_k128_scales_bf16, packed_up_b,
              up_b_k128_scales_bf16,
              m_tile * kSm87A4W4GateUpCellV2TileM,
              phase_n_start, 1U,
              physical_k64_group_count, k128_group_count);
        }

        for (unsigned int group = 0U; group < k128_group_count; ++group) {
          const bool has_lookahead = group + 1U < k128_group_count;
          if (has_lookahead) {
            cp_async_wait<4U>();
          } else {
            cp_async_wait<1U>();
          }
          __syncthreads();

          const unsigned int logical_slot =
              group % kSm87A4W4GateUpCellV2AStages;
          const unsigned int gate_slot = 2U * logical_slot;
          accumulate_phase(
              shared.phase_one.pipeline.a[logical_slot],
              shared.phase_one.pipeline.b[gate_slot],
              gate_accumulators);

          if (has_lookahead) {
            cp_async_wait<3U>();
          } else {
            cp_async_wait<0U>();
          }
          __syncthreads();
          accumulate_phase(
              shared.phase_one.pipeline.a[logical_slot],
              shared.phase_one.pipeline.b[gate_slot + 1U],
              up_accumulators);
          __syncthreads();

          if (group + kSm87A4W4GateUpCellV2AStages <
              k128_group_count) {
            prefetch_logical_group(
                shared.phase_one.pipeline, packed_a,
                a_k128_scales_bf16, packed_gate_b,
                gate_b_k128_scales_bf16, packed_up_b,
                up_b_k128_scales_bf16,
                m_tile * kSm87A4W4GateUpCellV2TileM,
                phase_n_start,
                group + kSm87A4W4GateUpCellV2AStages,
                physical_k64_group_count, k128_group_count);
          }
        }

        // The phase pipeline is dead.  Phase zero writes the independent
        // product tile; phase one reinterprets the dead pipeline union.
        const unsigned int product_lane =
            threadIdx.x % kSm87A4W4WarpThreads;
        const unsigned int product_warp =
            threadIdx.x / kSm87A4W4WarpThreads;
        const unsigned int product_warp_m = product_warp / 2U;
        const unsigned int product_warp_n = product_warp % 2U;
        float* const product =
            output_half == 0U
                ? &shared.phase_zero_product[0U][0U]
                : &shared.phase_one.product[0U][0U];
#pragma unroll
        for (unsigned int m_panel = 0U; m_panel < 1U; ++m_panel) {
          const unsigned int panel_m = product_warp_m * 16U;
          const unsigned int local_m0 = panel_m + product_lane / 4U;
          const unsigned int local_m1 = local_m0 + 8U;
#pragma unroll
          for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
            const unsigned int local_n0 =
                product_warp_n * 32U + fragment * 8U +
                2U * (product_lane % 4U);
            const unsigned int local_n1 = local_n0 + 1U;
            product[local_m0 * kSm87A4W4GateUpCellV2PhaseN + local_n0] =
                silu_product(gate_accumulators[m_panel][fragment][0U],
                             up_accumulators[m_panel][fragment][0U]);
            product[local_m0 * kSm87A4W4GateUpCellV2PhaseN + local_n1] =
                silu_product(gate_accumulators[m_panel][fragment][1U],
                             up_accumulators[m_panel][fragment][1U]);
            product[local_m1 * kSm87A4W4GateUpCellV2PhaseN + local_n0] =
                silu_product(gate_accumulators[m_panel][fragment][2U],
                             up_accumulators[m_panel][fragment][2U]);
            product[local_m1 * kSm87A4W4GateUpCellV2PhaseN + local_n1] =
                silu_product(gate_accumulators[m_panel][fragment][3U],
                             up_accumulators[m_panel][fragment][3U]);
          }
        }
        __syncthreads();
      }

      const unsigned int epilogue_lane =
          threadIdx.x % kSm87A4W4WarpThreads;
      const unsigned int epilogue_warp =
          threadIdx.x / kSm87A4W4WarpThreads;
#pragma unroll
      for (unsigned int row_iteration = 0U; row_iteration < 8U;
           ++row_iteration) {
        const unsigned int local_m =
            epilogue_warp + row_iteration * 8U;
        const unsigned int global_m =
            m_tile * kSm87A4W4GateUpCellV2TileM + local_m;
        float value0 =
            shared.phase_zero_product[local_m][2U * epilogue_lane];
        float value1 = shared.phase_zero_product
            [local_m][2U * epilogue_lane + 1U];
        float value2 =
            shared.phase_one.product[local_m][2U * epilogue_lane];
        float value3 = shared.phase_one.product
            [local_m][2U * epilogue_lane + 1U];
        float maximum = fmaxf(
            fmaxf(fabsf(value0), fabsf(value1)),
            fmaxf(fabsf(value2), fabsf(value3)));
#pragma unroll
        for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
          maximum = fmaxf(
              maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
        }
        maximum = __shfl_sync(0xffffffffU, maximum, 0U);
        const float clipped_maximum = maximum * output_clip_ratio;
        std::uint16_t scale_bits = encode_bf16(
            maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
        float stored_scale = decode_bf16(scale_bits);
        if (maximum != 0.0F && stored_scale == 0.0F) {
          scale_bits = 1U;
          stored_scale = decode_bf16(scale_bits);
        }
        value0 = fminf(fmaxf(value0, -clipped_maximum), clipped_maximum);
        value1 = fminf(fmaxf(value1, -clipped_maximum), clipped_maximum);
        value2 = fminf(fmaxf(value2, -clipped_maximum), clipped_maximum);
        value3 = fminf(fmaxf(value3, -clipped_maximum), clipped_maximum);
        int code0 = stored_scale == 0.0F
                        ? 0
                        : __float2int_rn(value0 / stored_scale);
        int code1 = stored_scale == 0.0F
                        ? 0
                        : __float2int_rn(value1 / stored_scale);
        int code2 = stored_scale == 0.0F
                        ? 0
                        : __float2int_rn(value2 / stored_scale);
        int code3 = stored_scale == 0.0F
                        ? 0
                        : __float2int_rn(value3 / stored_scale);
        code0 = code0 < -7 ? -7 : (code0 > 7 ? 7 : code0);
        code1 = code1 < -7 ? -7 : (code1 > 7 ? 7 : code1);
        code2 = code2 < -7 ? -7 : (code2 > 7 ? 7 : code2);
        code3 = code3 < -7 ? -7 : (code3 > 7 ? 7 : code3);
        const unsigned int first_physical_group = 2U * n_phase;
        packed_output[sm87_a4w4_consumer_packed_offset(
            global_m, first_physical_group, epilogue_lane,
            output_physical_k64_group_count)] =
            sm87_a4w4_pack_signed_pair(code0, code1);
        packed_output[sm87_a4w4_consumer_packed_offset(
            global_m, first_physical_group + 1U, epilogue_lane,
            output_physical_k64_group_count)] =
            sm87_a4w4_pack_signed_pair(code2, code3);
        if (epilogue_lane == 0U) {
          output_k128_scales_bf16[
              sm87_a4w4_consumer_k128_scale_offset(
                  global_m, n_phase, n_phase_count)] = scale_bits;
        }
      }
      __syncthreads();
    }
  }
}

[[nodiscard]] int validate_target(
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
      local.multiProcessorCount != kRequiredSmCount) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_gateup_cell_v2_resources_cuda(
    Sm87A4W4GateUpCellV2Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpCellV2Resources{};
  cudaDeviceProp properties{};
  const int device_status = validate_target(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_gateup_complete_cell_v2_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_gateup_complete_cell_v2_kernel,
      static_cast<int>(kSm87A4W4GateUpCellV2Threads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(kSm87A4W4GateUpCellV2MaximumRegisters) ||
      resources->static_shared_bytes !=
          kSm87A4W4GateUpCellV2SharedBytes ||
      resources->dynamic_shared_bytes != 0U ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpCellV2Threads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4GateUpCellV2CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_cell_v2_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k128_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpCellV2Plan plan =
      sm87_a4w4_gateup_cell_v2_plan(
          token_count, intermediate_size, input_size);
  if (plan.launch_ctas == 0U ||
      !(output_clip_ratio > 0.0F && output_clip_ratio <= 1.0F) ||
      !aligned(packed_a, 16U) || !aligned(a_k128_scales_bf16, 16U) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k128_scales_bf16, 16U) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k128_scales_bf16, 16U) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k128_scales_bf16, alignof(std::uint16_t)) ||
      plan.k128_groups > std::numeric_limits<unsigned int>::max() ||
      plan.physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      plan.m_tiles > std::numeric_limits<unsigned int>::max() ||
      plan.n_phases > std::numeric_limits<unsigned int>::max() ||
      plan.launch_ctas > std::numeric_limits<unsigned int>::max() ||
      plan.output_physical_k64_groups >
          std::numeric_limits<unsigned int>::max() ||
      !product_fits(token_count, input_size) ||
      !product_fits(intermediate_size, input_size) ||
      !product_fits(token_count, intermediate_size)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_a_scales =
      sm87_a4w4_consumer_k128_scale_capacity_elements(
          token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(
          intermediate_size, input_size);
  const std::size_t required_b_scales =
      sm87_a4w4_consumer_k128_scale_capacity_elements(
          intermediate_size, input_size);
  const std::size_t required_output_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(
          token_count, intermediate_size);
  const std::size_t required_output_scales =
      sm87_a4w4_consumer_k128_scale_capacity_elements(
          token_count, intermediate_size);
  if (required_a_bytes == 0U || required_a_scales == 0U ||
      required_b_bytes == 0U || required_b_scales == 0U ||
      required_output_bytes == 0U || required_output_scales == 0U ||
      packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scales ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      gate_b_scale_capacity_elements < required_b_scales ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      up_b_scale_capacity_elements < required_b_scales ||
      packed_output_capacity_bytes < required_output_bytes ||
      output_scale_capacity_elements < required_output_scales) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int target_status = validate_target();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  Sm87A4W4GateUpCellV2Resources resources{};
  const int resource_status =
      query_sm87_a4w4_gateup_cell_v2_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_gateup_complete_cell_v2_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4GateUpCellV2Threads), 0U,
      stream>>>(
      packed_a, a_k128_scales_bf16, packed_gate_b,
      gate_b_k128_scales_bf16, packed_up_b, up_b_k128_scales_bf16,
      static_cast<unsigned int>(plan.k128_groups),
      static_cast<unsigned int>(plan.physical_k64_groups),
      output_clip_ratio, packed_output, output_k128_scales_bf16,
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.n_phases),
      static_cast<unsigned int>(plan.output_physical_k64_groups));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
