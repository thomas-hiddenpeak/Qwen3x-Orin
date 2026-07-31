#include "q3x/kernels/sm87_a4w4_gateup_projection_v3.h"

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
    kSm87A4W4GateUpProjectionV3TileM * kPackedK64Bytes;
inline constexpr unsigned int kBPlaneBytes =
    kSm87A4W4GateUpProjectionV3TileN * kPackedK64Bytes;

struct alignas(16) Sm87A4W4GateUpProjectionV3AStage final {
  std::uint8_t code[kPhysicalK64PerK128][kAPlaneBytes];
  std::uint16_t scale[kSm87A4W4GateUpProjectionV3TileM];
};

struct alignas(16) Sm87A4W4GateUpProjectionV3BStage final {
  std::uint8_t code[kPhysicalK64PerK128][kBPlaneBytes];
  std::uint16_t scale[kSm87A4W4GateUpProjectionV3TileN];
};

struct alignas(16) Sm87A4W4GateUpProjectionV3LogicalStage final {
  Sm87A4W4GateUpProjectionV3AStage a;
  Sm87A4W4GateUpProjectionV3BStage gate;
  Sm87A4W4GateUpProjectionV3BStage up;
};

struct alignas(16) Sm87A4W4GateUpProjectionV3Pipeline final {
  Sm87A4W4GateUpProjectionV3LogicalStage
      stage[kSm87A4W4GateUpProjectionV3Stages];
};

union alignas(16) Sm87A4W4GateUpProjectionV3Shared final {
  Sm87A4W4GateUpProjectionV3Pipeline pipeline;
  float product[kSm87A4W4GateUpProjectionV3TileM]
               [kSm87A4W4GateUpProjectionV3TileN];
};

static_assert(sizeof(Sm87A4W4GateUpProjectionV3AStage) ==
              kSm87A4W4GateUpProjectionV3AStageBytes);
static_assert(sizeof(Sm87A4W4GateUpProjectionV3BStage) ==
              kSm87A4W4GateUpProjectionV3BStageBytes);
static_assert(sizeof(Sm87A4W4GateUpProjectionV3LogicalStage) ==
              kSm87A4W4GateUpProjectionV3LogicalStageBytes);
static_assert(sizeof(Sm87A4W4GateUpProjectionV3Pipeline) ==
              kSm87A4W4GateUpProjectionV3SharedBytes);
static_assert(sizeof(Sm87A4W4GateUpProjectionV3Shared) ==
              kSm87A4W4GateUpProjectionV3SharedBytes);

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
    Sm87A4W4GateUpProjectionV3AStage& stage,
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
      kSm87A4W4GateUpProjectionV3TileM * sizeof(std::uint16_t) / 16U;
  static_assert(kCodeVectors == kSm87A4W4GateUpProjectionV3Threads);
  static_assert(kScaleVectors == 8U);

  const unsigned int vector = threadIdx.x;
  const unsigned int half = vector / kVectorsPerPhysicalK64;
  const unsigned int half_vector = vector - half * kVectorsPerPhysicalK64;
  const unsigned int row = half_vector / 2U;
  const unsigned int row_vector = half_vector % 2U;
  const unsigned int physical_group = 2U * k128_group + half;
  cp_async_16(
      stage.code[half] +
          sm87_a4w4_swizzled_k64_byte_offset(row, 16U * row_vector),
      packed_a + sm87_a4w4_consumer_packed_offset(
                     static_cast<std::size_t>(m_tile_start) + row,
                     physical_group, 16U * row_vector,
                     physical_k64_group_count));

  if (threadIdx.x < kScaleVectors) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.scale + first_row,
        a_k128_scales_bf16 + sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(m_tile_start) + first_row,
            k128_group, k128_group_count));
  }
}

__device__ __forceinline__ void issue_b_stage(
    Sm87A4W4GateUpProjectionV3BStage& stage,
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
      kSm87A4W4GateUpProjectionV3TileN * sizeof(std::uint16_t) / 16U;
  static_assert(kCodeVectors ==
                2U * kSm87A4W4GateUpProjectionV3Threads);
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
        stage.code[half] +
            sm87_a4w4_swizzled_k64_byte_offset(row, 16U * row_vector),
        packed_b + sm87_a4w4_consumer_packed_offset(
                       static_cast<std::size_t>(n_tile_start) + row,
                       physical_group, 16U * row_vector,
                       physical_k64_group_count));
  }
  if (threadIdx.x < kScaleVectors) {
    const unsigned int first_row = 8U * threadIdx.x;
    cp_async_16(
        stage.scale + first_row,
        b_k128_scales_bf16 + sm87_a4w4_consumer_k128_scale_offset(
            static_cast<std::size_t>(n_tile_start) + first_row,
            k128_group, k128_group_count));
  }
}

__device__ __forceinline__ void issue_logical_stage(
    Sm87A4W4GateUpProjectionV3LogicalStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k128_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k128_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k128_scales_bf16,
    const unsigned int m_tile_start,
    const unsigned int n_tile_start,
    const unsigned int group,
    const unsigned int physical_k64_group_count,
    const unsigned int k128_group_count) noexcept {
  issue_a_stage(stage.a, packed_a, a_k128_scales_bf16, m_tile_start,
                group, physical_k64_group_count, k128_group_count);
  issue_b_stage(stage.gate, packed_gate_b, gate_b_k128_scales_bf16,
                n_tile_start, group, physical_k64_group_count,
                k128_group_count);
  issue_b_stage(stage.up, packed_up_b, up_b_k128_scales_bf16,
                n_tile_start, group, physical_k64_group_count,
                k128_group_count);
  cp_async_commit();
}

__device__ __forceinline__ void accumulate_projection_group(
    const Sm87A4W4GateUpProjectionV3LogicalStage& stage,
    float (&accumulators)[16U][4U]) noexcept {
  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const unsigned int projection_warp =
      warp % kSm87A4W4GateUpProjectionV3ProjectionWarps;
  const unsigned int fragment_m_start = projection_warp * 16U;
  const Sm87A4W4GateUpProjectionV3BStage& b_stage =
      warp < kSm87A4W4GateUpProjectionV3ProjectionWarps
          ? stage.gate
          : stage.up;
  const Sm87A4W4AFragment a_fragments[kPhysicalK64PerK128] = {
      sm87_a4w4_load_a_fragment_swizzled_shared(
          stage.a.code[0U] + fragment_m_start * kPackedK64Bytes, lane),
      sm87_a4w4_load_a_fragment_swizzled_shared(
          stage.a.code[1U] + fragment_m_start * kPackedK64Bytes, lane)};

#pragma unroll
  for (unsigned int fragment_n = 0U; fragment_n < 16U; ++fragment_n) {
    const unsigned int fragment_n_start = fragment_n * 8U;
    Sm87A4W4Accumulator partial{};
#pragma unroll
    for (unsigned int half = 0U; half < kPhysicalK64PerK128; ++half) {
      const Sm87A4W4BFragment b_fragment =
          sm87_a4w4_load_b_fragment_swizzled_shared(
              b_stage.code[half] + fragment_n_start * kPackedK64Bytes,
              lane);
      sm87_a4w4_mma_m16n8k64(partial, a_fragments[half], b_fragment);
    }
    const std::int32_t integer_partial[4U] = {
        partial.x0, partial.x1, partial.x2, partial.x3};
#pragma unroll
    for (unsigned int output = 0U; output < 4U; ++output) {
      const Sm87A4W4AccumulatorCoordinate coordinate =
          sm87_a4w4_accumulator_coordinate(lane, output);
      const unsigned int local_m = fragment_m_start + coordinate.m;
      const unsigned int local_n = fragment_n_start + coordinate.n;
      const float scale_product =
          decode_bf16(stage.a.scale[local_m]) *
          decode_bf16(b_stage.scale[local_n]);
      accumulators[fragment_n][output] +=
          static_cast<float>(integer_partial[output]) * scale_product;
    }
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpProjectionV3Threads,
                      kSm87A4W4GateUpProjectionV3CtasPerSm)
void q3x_sm87_a4w4_gateup_projection_v3_warp_specialized_kernel(
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
    const unsigned int n_tile_count,
    const unsigned int output_physical_k64_group_count) {
  __shared__ Sm87A4W4GateUpProjectionV3Shared shared;

  // At P2048 the 32 persistent CTAs own exactly one M64 tile each.  Gate and
  // Up are separated by warp crew, never by a second A load or a second K
  // traversal.  The complete N128 output scale remains CTA-local.
  for (unsigned int m_tile = blockIdx.x; m_tile < m_tile_count;
       m_tile += gridDim.x) {
    const unsigned int m_tile_start =
        m_tile * kSm87A4W4GateUpProjectionV3TileM;
    for (unsigned int n_tile = 0U; n_tile < n_tile_count; ++n_tile) {
      const unsigned int n_tile_start =
          n_tile * kSm87A4W4GateUpProjectionV3TileN;
      float accumulators[16U][4U]{};

      issue_logical_stage(
          shared.pipeline.stage[0U], packed_a, a_k128_scales_bf16,
          packed_gate_b, gate_b_k128_scales_bf16, packed_up_b,
          up_b_k128_scales_bf16, m_tile_start, n_tile_start, 0U,
          physical_k64_group_count, k128_group_count);
      if (k128_group_count > 1U) {
        issue_logical_stage(
            shared.pipeline.stage[1U], packed_a, a_k128_scales_bf16,
            packed_gate_b, gate_b_k128_scales_bf16, packed_up_b,
            up_b_k128_scales_bf16, m_tile_start, n_tile_start, 1U,
            physical_k64_group_count, k128_group_count);
      }

      for (unsigned int group = 0U; group < k128_group_count; ++group) {
        if (group + 1U < k128_group_count) {
          cp_async_wait<1U>();
        } else {
          cp_async_wait<0U>();
        }
        __syncthreads();
        accumulate_projection_group(
            shared.pipeline.stage[
                group % kSm87A4W4GateUpProjectionV3Stages],
            accumulators);
        __syncthreads();

        if (group + kSm87A4W4GateUpProjectionV3Stages <
            k128_group_count) {
          const unsigned int future =
              group + kSm87A4W4GateUpProjectionV3Stages;
          issue_logical_stage(
              shared.pipeline.stage[
                  future % kSm87A4W4GateUpProjectionV3Stages],
              packed_a, a_k128_scales_bf16, packed_gate_b,
              gate_b_k128_scales_bf16, packed_up_b,
              up_b_k128_scales_bf16, m_tile_start, n_tile_start,
              future, physical_k64_group_count, k128_group_count);
        }
      }

      const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
      const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
      const unsigned int projection_warp =
          warp % kSm87A4W4GateUpProjectionV3ProjectionWarps;
      const unsigned int fragment_m_start = projection_warp * 16U;

      // The drained copy ring is now a cross-crew exchange tile.  Gate owns
      // the first publication; the matching Up crew reads each coordinate,
      // applies the exact SiLU product, and overwrites it in place.
      if (warp < kSm87A4W4GateUpProjectionV3ProjectionWarps) {
#pragma unroll
        for (unsigned int fragment_n = 0U; fragment_n < 16U; ++fragment_n) {
#pragma unroll
          for (unsigned int output = 0U; output < 4U; ++output) {
            const Sm87A4W4AccumulatorCoordinate coordinate =
                sm87_a4w4_accumulator_coordinate(lane, output);
            shared.product[fragment_m_start + coordinate.m]
                          [fragment_n * 8U + coordinate.n] =
                accumulators[fragment_n][output];
          }
        }
      }
      __syncthreads();

      if (warp >= kSm87A4W4GateUpProjectionV3ProjectionWarps) {
#pragma unroll
        for (unsigned int fragment_n = 0U; fragment_n < 16U; ++fragment_n) {
#pragma unroll
          for (unsigned int output = 0U; output < 4U; ++output) {
            const Sm87A4W4AccumulatorCoordinate coordinate =
                sm87_a4w4_accumulator_coordinate(lane, output);
            float& gate_or_product =
                shared.product[fragment_m_start + coordinate.m]
                              [fragment_n * 8U + coordinate.n];
            gate_or_product = silu_product(
                gate_or_product, accumulators[fragment_n][output]);
          }
        }
      }
      __syncthreads();

#pragma unroll
      for (unsigned int row_iteration = 0U; row_iteration < 8U;
           ++row_iteration) {
        const unsigned int local_m = warp + row_iteration * 8U;
        const unsigned int global_m = m_tile_start + local_m;
        float value0 = shared.product[local_m][2U * lane];
        float value1 = shared.product[local_m][2U * lane + 1U];
        float value2 =
            shared.product[local_m][64U + 2U * lane];
        float value3 =
            shared.product[local_m][64U + 2U * lane + 1U];
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
        const unsigned int first_physical_group = 2U * n_tile;
        packed_output[sm87_a4w4_consumer_packed_offset(
            global_m, first_physical_group, lane,
            output_physical_k64_group_count)] =
            sm87_a4w4_pack_signed_pair(code0, code1);
        packed_output[sm87_a4w4_consumer_packed_offset(
            global_m, first_physical_group + 1U, lane,
            output_physical_k64_group_count)] =
            sm87_a4w4_pack_signed_pair(code2, code3);
        if (lane == 0U) {
          output_k128_scales_bf16[
              sm87_a4w4_consumer_k128_scale_offset(
                  global_m, n_tile, n_tile_count)] = scale_bits;
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
      local.multiProcessorCount != static_cast<int>(kRequiredSmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_gateup_projection_v3_resources_cuda(
    Sm87A4W4GateUpProjectionV3Resources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpProjectionV3Resources{};
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes,
      q3x_sm87_a4w4_gateup_projection_v3_warp_specialized_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      q3x_sm87_a4w4_gateup_projection_v3_warp_specialized_kernel,
      static_cast<int>(kSm87A4W4GateUpProjectionV3Threads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  // The kernel launches with no dynamic shared allocation.  The CUDA
  // attribute is the opt-in ceiling, not bytes consumed by this launch.
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;

  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4GateUpProjectionV3MaximumRegisters) ||
      resources->static_shared_bytes !=
          kSm87A4W4GateUpProjectionV3SharedBytes ||
      resources->dynamic_shared_bytes != 0U ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpProjectionV3Threads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4GateUpProjectionV3CtasPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_projection_v3_cuda(
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
  const Sm87A4W4GateUpProjectionV3Plan plan =
      sm87_a4w4_gateup_projection_v3_plan(
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
      plan.n_tiles > std::numeric_limits<unsigned int>::max() ||
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

  Sm87A4W4GateUpProjectionV3Resources resources{};
  const int resource_status =
      query_sm87_a4w4_gateup_projection_v3_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_gateup_projection_v3_warp_specialized_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas),
      static_cast<unsigned int>(kSm87A4W4GateUpProjectionV3Threads), 0U,
      stream>>>(
      packed_a, a_k128_scales_bf16, packed_gate_b,
      gate_b_k128_scales_bf16, packed_up_b, up_b_k128_scales_bf16,
      static_cast<unsigned int>(plan.k128_groups),
      static_cast<unsigned int>(plan.physical_k64_groups),
      output_clip_ratio, packed_output, output_k128_scales_bf16,
      static_cast<unsigned int>(plan.m_tiles),
      static_cast<unsigned int>(plan.n_tiles),
      static_cast<unsigned int>(plan.output_physical_k64_groups));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
