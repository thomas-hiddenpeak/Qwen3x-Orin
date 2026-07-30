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

static_assert(kPackedVectorsPerStage == 384U);
static_assert(sizeof(Sm87A4W4GateUpPipelineStage) == 6'528U);
static_assert(sizeof(Sm87A4W4GateUpSharedStorage) == 35'968U);
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
    const std::size_t packed_a_row_stride_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_row_stride_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_row_stride_bytes,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::size_t gate_b_scale_row_stride_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_row_stride_bytes,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t up_b_scale_row_stride_elements,
    const std::size_t token_count,
    const std::size_t m_tile_start,
    const std::size_t n_tile_start,
    const std::size_t k64_group) noexcept {
  // There are 384 packed 16-byte vectors.  Every thread issues at least one
  // copy and the first 128 threads issue two.  A occupies exactly one shared
  // segment and is subsequently consumed by both projection branches.
  for (std::size_t vector = threadIdx.x; vector < kPackedVectorsPerStage;
       vector += blockDim.x) {
    if (vector < kStageABytes / 16U) {
      const std::size_t row = vector / 2U;
      const std::size_t row_vector = vector % 2U;
      const std::size_t global_row = m_tile_start + row;
      const bool valid = global_row < token_count;
      const std::uint8_t* const source =
          valid ? packed_a + global_row * packed_a_row_stride_bytes +
                      k64_group * kPackedK64Bytes + row_vector * 16U
                : packed_a;
      cp_async_16(stage.a + row * kPackedK64Bytes + row_vector * 16U,
                  source, valid ? 16U : 0U);
    } else if (vector < (kStageABytes + kStageBBytes) / 16U) {
      const std::size_t gate_vector = vector - kStageABytes / 16U;
      const std::size_t row = gate_vector / 2U;
      const std::size_t row_vector = gate_vector % 2U;
      const std::uint8_t* const source =
          packed_gate_b +
          (n_tile_start + row) * packed_gate_b_row_stride_bytes +
          k64_group * kPackedK64Bytes + row_vector * 16U;
      cp_async_16(
          stage.gate_b + row * kPackedK64Bytes + row_vector * 16U,
          source, 16U);
    } else {
      const std::size_t up_vector =
          vector - (kStageABytes + kStageBBytes) / 16U;
      const std::size_t row = up_vector / 2U;
      const std::size_t row_vector = up_vector % 2U;
      const std::uint8_t* const source =
          packed_up_b +
          (n_tile_start + row) * packed_up_b_row_stride_bytes +
          k64_group * kPackedK64Bytes + row_vector * 16U;
      cp_async_16(stage.up_b + row * kPackedK64Bytes + row_vector * 16U,
                  source, 16U);
    }
  }

  // Scale tensors are strided by K64 group across rows, so a 16-byte async
  // vector would fetch unrelated groups.  One thread owns each scalar global
  // load and publishes it to the same three-stage shared slot.  In particular
  // each A scale reaches global memory once, then serves Gate and Up together.
  if (threadIdx.x < 3U * kSm87A4W4GateUpTileN) {
    const std::size_t table = threadIdx.x / kSm87A4W4GateUpTileN;
    const std::size_t row = threadIdx.x % kSm87A4W4GateUpTileN;
    if (table == 0U) {
      const std::size_t global_row = m_tile_start + row;
      stage.a_scales[row] =
          global_row < token_count
              ? a_k64_scales_bf16[
                    global_row * a_scale_row_stride_elements + k64_group]
              : static_cast<std::uint16_t>(0U);
    } else if (table == 1U) {
      stage.gate_b_scales[row] = gate_b_k64_scales_bf16[
          (n_tile_start + row) * gate_b_scale_row_stride_elements +
          k64_group];
    } else {
      stage.up_b_scales[row] = up_b_k64_scales_bf16[
          (n_tile_start + row) * up_b_scale_row_stride_elements +
          k64_group];
    }
  }
  cp_async_commit();
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpThreads,
                      kSm87A4W4GateUpCtasPerSm)
void q3x_sm87_a4w4_gateup_paired_kernel(
    const std::uint8_t* const packed_a,
    const std::size_t packed_a_row_stride_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_row_stride_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_row_stride_bytes,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::size_t gate_b_scale_row_stride_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_row_stride_bytes,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t up_b_scale_row_stride_elements,
    const std::size_t token_count,
    const std::size_t k64_group_count,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_row_stride_bytes,
    std::uint16_t* const output_k64_scales_bf16,
    const std::size_t output_scale_row_stride_elements,
    const std::size_t m_tile_count,
    const std::size_t work_tile_count) {
  __shared__ Sm87A4W4GateUpSharedStorage shared;

  const unsigned int lane = threadIdx.x % kSm87A4W4WarpThreads;
  const unsigned int warp = threadIdx.x / kSm87A4W4WarpThreads;
  const std::size_t warp_m = warp % 4U;
  const std::size_t warp_n = warp / 4U;

  for (std::size_t work_tile = blockIdx.x; work_tile < work_tile_count;
       work_tile += gridDim.x) {
    const std::size_t n_tile = work_tile / m_tile_count;
    const std::size_t m_tile = work_tile - n_tile * m_tile_count;
    const std::size_t m_tile_start = m_tile * kSm87A4W4GateUpTileM;
    const std::size_t n_tile_start = n_tile * kSm87A4W4GateUpTileN;

    float gate_accumulators[4U][4U]{};
    float up_accumulators[4U][4U]{};

    prefetch_stage(
        shared.pipeline[0U], packed_a, packed_a_row_stride_bytes,
        a_k64_scales_bf16, a_scale_row_stride_elements, packed_gate_b,
        packed_gate_b_row_stride_bytes, gate_b_k64_scales_bf16,
        gate_b_scale_row_stride_elements, packed_up_b,
        packed_up_b_row_stride_bytes, up_b_k64_scales_bf16,
        up_b_scale_row_stride_elements, token_count, m_tile_start,
        n_tile_start, 0U);
    if (k64_group_count > 1U) {
      prefetch_stage(
          shared.pipeline[1U], packed_a, packed_a_row_stride_bytes,
          a_k64_scales_bf16, a_scale_row_stride_elements, packed_gate_b,
          packed_gate_b_row_stride_bytes, gate_b_k64_scales_bf16,
          gate_b_scale_row_stride_elements, packed_up_b,
          packed_up_b_row_stride_bytes, up_b_k64_scales_bf16,
          up_b_scale_row_stride_elements, token_count, m_tile_start,
          n_tile_start, 1U);
    }

    for (std::size_t group = 0U; group < k64_group_count; ++group) {
      if (group + 2U < k64_group_count) {
        prefetch_stage(
            shared.pipeline[(group + 2U) %
                            kSm87A4W4GateUpPipelineStages],
            packed_a, packed_a_row_stride_bytes, a_k64_scales_bf16,
            a_scale_row_stride_elements, packed_gate_b,
            packed_gate_b_row_stride_bytes, gate_b_k64_scales_bf16,
            gate_b_scale_row_stride_elements, packed_up_b,
            packed_up_b_row_stride_bytes, up_b_k64_scales_bf16,
            up_b_scale_row_stride_elements, token_count, m_tile_start,
            n_tile_start, group + 2U);
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
      const Sm87A4W4AFragment a = sm87_a4w4_load_a_fragment(
          warp_a, kPackedK64Bytes, 0U, lane);
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
            sm87_a4w4_load_b_fragment(gate_b, kPackedK64Bytes, 0U, lane);
        const Sm87A4W4BFragment up_fragment =
            sm87_a4w4_load_b_fragment(up_b, kPackedK64Bytes, 0U, lane);
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

    // Each warp quantizes eight complete N64 rows.  A row's 64 FP32 products
    // stay inside this CTA: no BF16 Gate or Up tensor reaches global memory.
#pragma unroll
    for (unsigned int row_iteration = 0U; row_iteration < 8U;
         ++row_iteration) {
      const std::size_t local_m = warp + row_iteration * 8U;
      const std::size_t global_m = m_tile_start + local_m;
      if (global_m < token_count) {
        float even = shared.product[local_m][2U * lane];
        float odd = shared.product[local_m][2U * lane + 1U];
        float maximum = fmaxf(fabsf(even), fabsf(odd));
#pragma unroll
        for (unsigned int delta = 16U; delta != 0U; delta /= 2U) {
          maximum = fmaxf(maximum,
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
        even_code = even_code < -7 ? -7 :
                    (even_code > 7 ? 7 : even_code);
        odd_code = odd_code < -7 ? -7 :
                   (odd_code > 7 ? 7 : odd_code);
        packed_output[global_m * packed_output_row_stride_bytes +
                      n_tile_start / 2U + lane] =
            sm87_a4w4_pack_signed_pair(even_code, odd_code);
        if (lane == 0U) {
          output_k64_scales_bf16[
              global_m * output_scale_row_stride_elements + n_tile] =
              encode_bf16(scale);
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
    const std::size_t packed_a_row_stride_bytes,
    const std::uint16_t* const a_k64_scales_bf16,
    const std::size_t a_scale_row_stride_elements,
    const std::uint8_t* const packed_gate_b,
    const std::size_t packed_gate_b_row_stride_bytes,
    const std::uint16_t* const gate_b_k64_scales_bf16,
    const std::size_t gate_b_scale_row_stride_elements,
    const std::uint8_t* const packed_up_b,
    const std::size_t packed_up_b_row_stride_bytes,
    const std::uint16_t* const up_b_k64_scales_bf16,
    const std::size_t up_b_scale_row_stride_elements,
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const float output_clip_ratio,
    std::uint8_t* const packed_output,
    const std::size_t packed_output_row_stride_bytes,
    std::uint16_t* const output_k64_scales_bf16,
    const std::size_t output_scale_row_stride_elements,
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
      packed_a_row_stride_bytes < input_size / 2U ||
      packed_gate_b_row_stride_bytes < input_size / 2U ||
      packed_up_b_row_stride_bytes < input_size / 2U ||
      packed_a_row_stride_bytes % 16U != 0U ||
      packed_gate_b_row_stride_bytes % 16U != 0U ||
      packed_up_b_row_stride_bytes % 16U != 0U ||
      a_scale_row_stride_elements < plan.k64_groups ||
      gate_b_scale_row_stride_elements < plan.k64_groups ||
      up_b_scale_row_stride_elements < plan.k64_groups ||
      packed_output_row_stride_bytes < plan.packed_output_row_bytes ||
      output_scale_row_stride_elements <
          plan.output_scale_row_elements ||
      !product_fits(token_count, packed_a_row_stride_bytes) ||
      !product_fits(token_count, a_scale_row_stride_elements) ||
      !product_fits(intermediate_size,
                    packed_gate_b_row_stride_bytes) ||
      !product_fits(intermediate_size,
                    gate_b_scale_row_stride_elements) ||
      !product_fits(intermediate_size, packed_up_b_row_stride_bytes) ||
      !product_fits(intermediate_size, up_b_scale_row_stride_elements) ||
      !product_fits(token_count, packed_output_row_stride_bytes) ||
      !product_fits(token_count, output_scale_row_stride_elements)) {
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
      packed_a, packed_a_row_stride_bytes, a_k64_scales_bf16,
      a_scale_row_stride_elements, packed_gate_b,
      packed_gate_b_row_stride_bytes, gate_b_k64_scales_bf16,
      gate_b_scale_row_stride_elements, packed_up_b,
      packed_up_b_row_stride_bytes, up_b_k64_scales_bf16,
      up_b_scale_row_stride_elements, token_count, plan.k64_groups,
      output_clip_ratio, packed_output, packed_output_row_stride_bytes,
      output_k64_scales_bf16, output_scale_row_stride_elements,
      plan.m_tiles, plan.work_tiles);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
