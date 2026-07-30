#include "q3x/kernels/sm87_a4w4_gateup_paired.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr std::size_t kPackedK64Bytes =
    kSm87A4W4GateUpTileK / 2U;
inline constexpr std::size_t kStageABytes =
    kSm87A4W4GateUpTileM * kPackedK64Bytes;
inline constexpr std::size_t kStageBBytes =
    kSm87A4W4GateUpTileN * kPackedK64Bytes;
inline constexpr std::size_t kPackedVectorsPerStage =
    (kStageABytes + 2U * kStageBBytes) / 16U;
inline constexpr int kRequiredSmCount = 16;

struct alignas(16) Sm87A4W4GateUpPipelineStage final {
  std::uint8_t a[kStageABytes];
  std::uint8_t gate_b[kStageBBytes];
  std::uint8_t up_b[kStageBBytes];
  std::uint16_t a_scales[kSm87A4W4GateUpTileM];
  std::uint16_t gate_b_scales[kSm87A4W4GateUpTileN];
  std::uint16_t up_b_scales[kSm87A4W4GateUpTileN];
};

struct alignas(16) Sm87A4W4GateUpSharedStorage final {
  Sm87A4W4GateUpPipelineStage
      pipeline[kSm87A4W4GateUpPipelineStages];
  float product[kSm87A4W4GateUpTileM][kSm87A4W4GateUpTileN];
};

static_assert(kPackedVectorsPerStage == 576U);
static_assert(sizeof(Sm87A4W4GateUpPipelineStage) == 9'792U);
static_assert(sizeof(Sm87A4W4GateUpSharedStorage) == 45'760U);
static_assert(sizeof(Sm87A4W4GateUpSharedStorage) *
                  kSm87A4W4GateUpCtasPerSm <=
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

[[nodiscard]] constexpr bool consumer_capacity_fits(
    const std::size_t outer_count,
    const std::size_t k64_group_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return blocks != 0U && k64_group_count != 0U &&
         product_fits(blocks, k64_group_count) &&
         product_fits(blocks * k64_group_count,
                      kSm87A4W4ConsumerOuterBlock) &&
         product_fits(blocks * k64_group_count *
                          kSm87A4W4ConsumerOuterBlock,
                      kSm87A4W4ConsumerPackedKBlockBytes);
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
  // This branch avoids exp overflow for large negative Gate values while
  // preserving an FP32 SiLU boundary before output quantization.
  if (gate >= 0.0F) {
    return (gate / (1.0F + expf(-gate))) * up;
  }
  const float exponential = expf(gate);
  return (gate * exponential / (1.0F + exponential)) * up;
}

__device__ __forceinline__ void cp_async_16(
    void* const destination, const void* const source,
    const unsigned int source_bytes) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
               :
               : "r"(shared_address), "l"(source), "r"(source_bytes)
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

template <int PendingGroups>
__device__ __forceinline__ void cp_async_wait() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;"
               :
               : "n"(PendingGroups)
               : "memory");
#else
  asm volatile("trap;");
#endif
}

__device__ __forceinline__ void prefetch_stage(
    Sm87A4W4GateUpPipelineStage& stage,
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t token_count,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t k64_group,
    const std::size_t k64_group_count) noexcept {
  // The three global consumer blocks are contiguous in N64 halves. A is
  // staged once and subsequently consumed by both projection branches.
  for (std::size_t vector = threadIdx.x; vector < kPackedVectorsPerStage;
       vector += blockDim.x) {
    if (vector < kStageABytes / 16U) {
      const std::size_t row = vector / 2U;
      const std::size_t row_vector = vector % 2U;
      const std::size_t global_row = m_tile_start + row;
      const bool valid = global_row < token_count;
      const std::uint8_t* const source =
          valid ? packed_a + sm87_a4w4_consumer_packed_offset(
                                 global_row, k64_group,
                                 row_vector * 16U, k64_group_count)
                : packed_a;
      cp_async_16(
          stage.a + sm87_a4w4_swizzled_k64_byte_offset(
                        row, row_vector * 16U),
          source, valid ? 16U : 0U);
    } else if (vector < (kStageABytes + kStageBBytes) / 16U) {
      const std::size_t gate_vector = vector - kStageABytes / 16U;
      const std::size_t row = gate_vector / 2U;
      const std::size_t row_vector = gate_vector % 2U;
      const std::uint8_t* const source =
          packed_gate_b + sm87_a4w4_consumer_packed_offset(
                              n_tile_start + row, k64_group,
                              row_vector * 16U, k64_group_count);
      cp_async_16(
          stage.gate_b + sm87_a4w4_swizzled_k64_byte_offset(
                             row, row_vector * 16U),
          source, 16U);
    } else {
      const std::size_t up_vector =
          vector - (kStageABytes + kStageBBytes) / 16U;
      const std::size_t row = up_vector / 2U;
      const std::size_t row_vector = up_vector % 2U;
      const std::uint8_t* const source =
          packed_up_b + sm87_a4w4_consumer_packed_offset(
                            n_tile_start + row, k64_group,
                            row_vector * 16U, k64_group_count);
      cp_async_16(
          stage.up_b + sm87_a4w4_swizzled_k64_byte_offset(
                           row, row_vector * 16U),
          source, 16U);
    }
  }

  if (threadIdx.x < kSm87A4W4GateUpTileM) {
    const std::size_t global_row = m_tile_start + threadIdx.x;
    stage.a_scales[threadIdx.x] =
        global_row < token_count
            ? a_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                  global_row, k64_group, k64_group_count)]
            : static_cast<std::uint16_t>(0U);
  }
  if (threadIdx.x < kSm87A4W4GateUpTileN) {
    const std::size_t global_row = n_tile_start + threadIdx.x;
    stage.gate_b_scales[threadIdx.x] =
        gate_b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_row, k64_group, k64_group_count)];
    stage.up_b_scales[threadIdx.x] =
        up_b_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
            global_row, k64_group, k64_group_count)];
  }
  cp_async_commit();
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpThreads,
                      kSm87A4W4GateUpCtasPerSm)
void q3x_sm87_a4w4_gateup_paired_kernel(
    const std::uint8_t* const packed_a,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::uint8_t* const packed_gate_b,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::uint8_t* const packed_up_b,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t token_count,
    const std::size_t k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    std::uint16_t* const output_k64_scales_bf16,
    const std::size_t output_k64_group_count,
    const std::size_t m_tile_count,
    const std::size_t work_tile_count) {
  __shared__ Sm87A4W4GateUpSharedStorage shared;

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const std::size_t warp_m = warp % 2U;
  const std::size_t warp_n = warp / 2U;

  for (std::size_t work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const std::size_t n_tile = work_tile / m_tile_count;
    const std::size_t m_tile = work_tile - n_tile * m_tile_count;
    const std::size_t m_tile_start = m_tile * kSm87A4W4GateUpTileM;
    const std::size_t n_tile_start = n_tile * kSm87A4W4GateUpTileN;

    float gate_accumulators[4U][4U]{};
    float up_accumulators[4U][4U]{};

    prefetch_stage(
        shared.pipeline[0U], packed_a, a_k64_scales_bf16, packed_gate_b,
        gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
        token_count, m_tile_start, n_tile_start, 0U, k64_group_count);
    if (k64_group_count > 1U) {
      prefetch_stage(
          shared.pipeline[1U], packed_a, a_k64_scales_bf16, packed_gate_b,
          gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
          token_count, m_tile_start, n_tile_start, 1U, k64_group_count);
    }

    for (std::size_t group = 0U; group < k64_group_count; ++group) {
      if (group + 2U < k64_group_count) {
        prefetch_stage(
            shared.pipeline[(group + 2U) %
                            kSm87A4W4GateUpPipelineStages],
            packed_a, a_k64_scales_bf16, packed_gate_b,
            gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
            token_count, m_tile_start, n_tile_start, group + 2U,
            k64_group_count);
      }
      if (group + 1U == k64_group_count) {
        cp_async_wait<0>();
      } else {
        cp_async_wait<1>();
      }
      __syncthreads();

      const Sm87A4W4GateUpPipelineStage& stage =
          shared.pipeline[group % kSm87A4W4GateUpPipelineStages];
      const std::uint8_t* const warp_a =
          stage.a + warp_m * 16U * kPackedK64Bytes;
      const Sm87A4W4AFragment a =
          sm87_a4w4_load_a_fragment_swizzled_shared(warp_a, lane);
      const std::size_t local_m0 = warp_m * 16U + lane / 4U;
      const std::size_t local_m1 = local_m0 + 8U;
      const float a_scale0 = decode_bf16(stage.a_scales[local_m0]);
      const float a_scale1 = decode_bf16(stage.a_scales[local_m1]);

#pragma unroll
      for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
        const std::size_t fragment_n = warp_n * 32U + fragment * 8U;
        const std::uint8_t* const gate_b =
            stage.gate_b + fragment_n * kPackedK64Bytes;
        const std::uint8_t* const up_b =
            stage.up_b + fragment_n * kPackedK64Bytes;
        const Sm87A4W4BFragment gate_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(gate_b, lane);
        const Sm87A4W4BFragment up_fragment =
            sm87_a4w4_load_b_fragment_swizzled_shared(up_b, lane);
        Sm87A4W4Accumulator gate_partial{};
        Sm87A4W4Accumulator up_partial{};
        sm87_a4w4_mma_m16n8k64(gate_partial, a, gate_fragment);
        sm87_a4w4_mma_m16n8k64(up_partial, a, up_fragment);

        const std::size_t local_n0 = fragment_n + 2U * (lane % 4U);
        const std::size_t local_n1 = local_n0 + 1U;
        const float gate_scale0 =
            decode_bf16(stage.gate_b_scales[local_n0]);
        const float gate_scale1 =
            decode_bf16(stage.gate_b_scales[local_n1]);
        const float up_scale0 = decode_bf16(stage.up_b_scales[local_n0]);
        const float up_scale1 = decode_bf16(stage.up_b_scales[local_n1]);

        gate_accumulators[fragment][0U] +=
            static_cast<float>(gate_partial.x0) * a_scale0 * gate_scale0;
        gate_accumulators[fragment][1U] +=
            static_cast<float>(gate_partial.x1) * a_scale0 * gate_scale1;
        gate_accumulators[fragment][2U] +=
            static_cast<float>(gate_partial.x2) * a_scale1 * gate_scale0;
        gate_accumulators[fragment][3U] +=
            static_cast<float>(gate_partial.x3) * a_scale1 * gate_scale1;
        up_accumulators[fragment][0U] +=
            static_cast<float>(up_partial.x0) * a_scale0 * up_scale0;
        up_accumulators[fragment][1U] +=
            static_cast<float>(up_partial.x1) * a_scale0 * up_scale1;
        up_accumulators[fragment][2U] +=
            static_cast<float>(up_partial.x2) * a_scale1 * up_scale0;
        up_accumulators[fragment][3U] +=
            static_cast<float>(up_partial.x3) * a_scale1 * up_scale1;
      }
      __syncthreads();
    }

    const std::size_t local_m0 = warp_m * 16U + lane / 4U;
    const std::size_t local_m1 = local_m0 + 8U;
#pragma unroll
    for (unsigned int fragment = 0U; fragment < 4U; ++fragment) {
      const std::size_t local_n0 =
          warp_n * 32U + fragment * 8U + 2U * (lane % 4U);
      const std::size_t local_n1 = local_n0 + 1U;
      shared.product[local_m0][local_n0] = silu_product(
          gate_accumulators[fragment][0U], up_accumulators[fragment][0U]);
      shared.product[local_m0][local_n1] = silu_product(
          gate_accumulators[fragment][1U], up_accumulators[fragment][1U]);
      shared.product[local_m1][local_n0] = silu_product(
          gate_accumulators[fragment][2U], up_accumulators[fragment][2U]);
      shared.product[local_m1][local_n1] = silu_product(
          gate_accumulators[fragment][3U], up_accumulators[fragment][3U]);
    }
    __syncthreads();

    // Each warp owns four M rows and quantizes both N64 halves. A complete
    // output K64 group stays inside this CTA; no BF16 Gate or Up tensor
    // reaches global memory.
#pragma unroll
    for (unsigned int row_iteration = 0U; row_iteration < 4U;
         ++row_iteration) {
      const std::size_t local_m = warp + row_iteration * 8U;
      const std::size_t global_m = m_tile_start + local_m;
      if (global_m < token_count) {
        for (unsigned int output_group = 0U; output_group < 2U;
             ++output_group) {
          float even =
              shared.product[local_m][output_group * 64U + 2U * lane];
          float odd = shared.product[local_m]
                                    [output_group * 64U + 2U * lane + 1U];
          float maximum = fmaxf(fabsf(even), fabsf(odd));
#pragma unroll
          for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
            maximum = fmaxf(
                maximum,
                __shfl_down_sync(0xffffffffU, maximum, delta));
          }
          maximum = __shfl_sync(0xffffffffU, maximum, 0U);
          const float clipped_maximum = maximum * output_clip_ratio;
          const float scale =
              clipped_maximum > 0.0F ? clipped_maximum / 7.0F : 1.0F;
          const float inverse_scale = 1.0F / scale;
          even = fminf(fmaxf(even, -clipped_maximum), clipped_maximum);
          odd = fminf(fmaxf(odd, -clipped_maximum), clipped_maximum);
          int even_code = __float2int_rn(even * inverse_scale);
          int odd_code = __float2int_rn(odd * inverse_scale);
          even_code = even_code < -7 ? -7
                                     : (even_code > 7 ? 7 : even_code);
          odd_code = odd_code < -7 ? -7
                                   : (odd_code > 7 ? 7 : odd_code);
          const std::size_t global_output_group =
              n_tile_start / 64U + output_group;
          packed_output[sm87_a4w4_consumer_packed_offset(
              global_m, global_output_group, lane,
              output_k64_group_count)] =
              sm87_a4w4_pack_signed_pair(even_code, odd_code);
          if (lane == 0U) {
            output_k64_scales_bf16[sm87_a4w4_consumer_scale_offset(
                global_m, global_output_group, output_k64_group_count)] =
                encode_bf16(scale);
          }
        }
      }
    }
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
      local.multiProcessorCount != kRequiredSmCount) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (properties != nullptr) {
    *properties = local;
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_gateup_paired_resources_cuda(
    Sm87A4W4GateUpPairedResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4GateUpPairedResources{};
  cudaDeviceProp properties{};
  const int device_status = validate_sm87(&properties);
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_gateup_paired_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_gateup_paired_kernel,
      static_cast<int>(kSm87A4W4GateUpThreads), 0U);
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

  if (resources->local_bytes != 0U ||
      resources->active_blocks_per_sm <
          static_cast<int>(kSm87A4W4GateUpCtasPerSm) ||
      resources->maximum_threads_per_block <
          static_cast<int>(kSm87A4W4GateUpThreads) ||
      resources->static_shared_bytes !=
          sizeof(Sm87A4W4GateUpSharedStorage)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_gateup_paired_cuda(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_capacity_bytes,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::size_t gate_b_scale_capacity_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_capacity_bytes,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t up_b_scale_capacity_elements,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity_bytes,
    std::uint16_t* const output_k64_scales_bf16,
    const std::size_t output_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const Sm87A4W4GateUpPairedPlan plan =
      sm87_a4w4_gateup_paired_plan(token_count, intermediate_size,
                                  input_size);
  if (plan.launch_ctas == 0U ||
      !(output_clip_ratio > 0.0F && output_clip_ratio <= 1.0F) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_gate_b, 16U) ||
      !aligned(gate_b_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_up_b, 16U) ||
      !aligned(up_b_k64_scales_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_output, 16U) ||
      !aligned(output_k64_scales_bf16, alignof(std::uint16_t)) ||
      !consumer_capacity_fits(token_count, plan.k64_groups) ||
      !consumer_capacity_fits(intermediate_size, plan.k64_groups) ||
      !consumer_capacity_fits(
          token_count, intermediate_size / kSm87A4W4ConsumerKBlock)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t required_a_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count, input_size);
  const std::size_t required_a_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(token_count, input_size);
  const std::size_t required_b_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(intermediate_size, input_size);
  const std::size_t required_b_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(intermediate_size,
                                                 input_size);
  const std::size_t required_output_bytes =
      sm87_a4w4_consumer_packed_capacity_bytes(token_count,
                                               intermediate_size);
  const std::size_t required_output_scale_elements =
      sm87_a4w4_consumer_scale_capacity_elements(token_count,
                                                 intermediate_size);
  if (packed_a_capacity_bytes < required_a_bytes ||
      a_scale_capacity_elements < required_a_scale_elements ||
      packed_gate_b_capacity_bytes < required_b_bytes ||
      gate_b_scale_capacity_elements < required_b_scale_elements ||
      packed_up_b_capacity_bytes < required_b_bytes ||
      up_b_scale_capacity_elements < required_b_scale_elements ||
      packed_output_capacity_bytes < required_output_bytes ||
      output_scale_capacity_elements < required_output_scale_elements) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int device_status = validate_sm87();
  if (device_status != static_cast<int>(cudaSuccess)) {
    return device_status;
  }
  Sm87A4W4GateUpPairedResources resources{};
  const int resource_status =
      query_sm87_a4w4_gateup_paired_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  q3x_sm87_a4w4_gateup_paired_kernel<<<
      static_cast<unsigned int>(kSm87A4W4GateUpPersistentCtas),
      static_cast<unsigned int>(kSm87A4W4GateUpThreads), 0U, stream>>>(
      packed_a, a_k64_scales_bf16, packed_gate_b,
      gate_b_k64_scales_bf16, packed_up_b, up_b_k64_scales_bf16,
      token_count, plan.k64_groups, output_clip_ratio, packed_output,
      output_k64_scales_bf16,
      intermediate_size / kSm87A4W4ConsumerKBlock,
      plan.m_tiles, plan.work_tiles);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
