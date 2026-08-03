#include "q3x/kernels/sm87_a4w4_prefill_handoff.h"

#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"
#include "q3x/kernels/sm87_a4w4_factorized_lane_quantize.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87A4W4PrefillHandoffThreads);
inline constexpr unsigned int kReductionThreads = 256U;
inline constexpr unsigned int kWarps = kThreads / 32U;
inline constexpr unsigned int kReductionWarps = kReductionThreads / 32U;
inline constexpr unsigned int kHidden =
    static_cast<unsigned int>(kSm87A4W4PrefillHandoffHiddenSize);
inline constexpr unsigned int kK256Groups =
    kHidden / static_cast<unsigned int>(kSm87A4W4AttentionK256ScaleK);
inline constexpr unsigned int kPhysicalK64Groups =
    kHidden /
    static_cast<unsigned int>(kSm87A4W4AttentionK256PhysicalK64);
inline constexpr int kRequiredSmCount = 16;

static_assert(kThreads == 512U);
static_assert(kReductionThreads == 256U);
static_assert(kWarps == 16U);
static_assert(kReductionWarps == 8U);
static_assert(kHidden == 5'120U);
static_assert(kK256Groups == 20U);
static_assert(kPhysicalK64Groups == 80U);

struct alignas(16) HandoffShared final {
  std::uint16_t row[kHidden];
  float partial[kReductionThreads];
  float warp_maxima[kReductionWarps];
  float clipped_maximum;
  float stored_scale;
  std::uint16_t scale_bits;
  std::uint16_t reserved;
};

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

[[nodiscard]] __device__ __forceinline__ float warp_max(
    float value) noexcept {
#pragma unroll
  for (unsigned int delta = 16U; delta != 0U; delta >>= 1U) {
    value = fmaxf(value,
                  __shfl_down_sync(0xffff'ffffU, value, delta));
  }
  return value;
}

__device__ __forceinline__ void publish_residual_and_normalized_row(
    HandoffShared& shared,
    const std::uint16_t* const left_bf16,
    const std::uint16_t* const right_bf16,
    const std::uint16_t* const centered_norm_weight_bf16,
    const unsigned int row,
    const float epsilon,
    std::uint16_t* const residual_output_bf16,
    std::uint16_t* const normalized_output_bf16) noexcept {
  const std::size_t row_offset = static_cast<std::size_t>(row) * kHidden;
  for (unsigned int dimension = threadIdx.x; dimension < kHidden;
       dimension += kThreads) {
    const std::size_t index = row_offset + dimension;
    const std::uint16_t residual_bits = encode_bf16(
        decode_bf16(left_bf16[index]) + decode_bf16(right_bf16[index]));
    residual_output_bf16[index] = residual_bits;
    shared.row[dimension] = residual_bits;
  }
  __syncthreads();

  if (threadIdx.x < kReductionThreads) {
    float sum = 0.0F;
#pragma unroll
    for (unsigned int pair = 0U; pair < 10U; ++pair) {
      const unsigned int first = threadIdx.x + 2U * pair * kReductionThreads;
      const unsigned int second = first + kReductionThreads;
      const float first_value = decode_bf16(shared.row[first]);
      const float second_value = decode_bf16(shared.row[second]);
      sum = fmaf(first_value, first_value, sum);
      sum = fmaf(second_value, second_value, sum);
    }
    shared.partial[threadIdx.x] = sum;
  }
  __syncthreads();
  for (unsigned int stride = kReductionThreads / 2U; stride != 0U;
       stride >>= 1U) {
    if (threadIdx.x < stride) {
      shared.partial[threadIdx.x] += shared.partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x < kReductionThreads) {
    const float inverse_rms =
        rsqrtf(shared.partial[0] / static_cast<float>(kHidden) + epsilon);
#pragma unroll
    for (unsigned int pair = 0U; pair < 10U; ++pair) {
      const unsigned int first = threadIdx.x + 2U * pair * kReductionThreads;
      const unsigned int second = first + kReductionThreads;
      const std::uint16_t first_bits = encode_bf16(
          decode_bf16(shared.row[first]) * inverse_rms *
          (decode_bf16(centered_norm_weight_bf16[first]) + 1.0F));
      const std::uint16_t second_bits = encode_bf16(
          decode_bf16(shared.row[second]) * inverse_rms *
          (decode_bf16(centered_norm_weight_bf16[second]) + 1.0F));
      shared.row[first] = first_bits;
      shared.row[second] = second_bits;
      if (normalized_output_bf16 != nullptr) {
        normalized_output_bf16[row_offset + first] = first_bits;
        normalized_output_bf16[row_offset + second] = second_bits;
      }
    }
  }
  __syncthreads();
}

__device__ __forceinline__ void publish_r1_row(
    HandoffShared& shared,
    const float* const inverse_alpha_fp32,
    const unsigned int row,
    const bool valid_row,
    const float clip_ratio,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_lane_scales_bf16) noexcept {
  float maximum = 0.0F;
  if (valid_row && threadIdx.x < kReductionThreads) {
    for (unsigned int k = threadIdx.x; k < kHidden;
         k += kReductionThreads) {
      maximum = fmaxf(
          maximum,
          fabsf(decode_bf16(shared.row[k]) * inverse_alpha_fp32[k]));
    }
  }
  if (threadIdx.x < kReductionThreads) {
    maximum = warp_max(maximum);
    const unsigned int lane = threadIdx.x & 31U;
    const unsigned int warp = threadIdx.x >> 5U;
    if (lane == 0U) {
      shared.warp_maxima[warp] = maximum;
    }
  }
  __syncthreads();

  if (threadIdx.x < 32U) {
    const unsigned int lane = threadIdx.x;
    float block_maximum =
        lane < kReductionWarps ? shared.warp_maxima[lane] : 0.0F;
    block_maximum = warp_max(block_maximum);
    if (lane == 0U) {
      const float clipped_maximum = block_maximum * clip_ratio;
      std::uint16_t scale_bits = encode_bf16(
          block_maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (block_maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      shared.clipped_maximum = clipped_maximum;
      shared.stored_scale = stored_scale;
      shared.scale_bits = scale_bits;
    }
  }
  __syncthreads();

  if (threadIdx.x < kReductionThreads) {
    constexpr unsigned int kPairCount = kHidden / 2U;
    for (unsigned int pair = threadIdx.x; pair < kPairCount;
         pair += kReductionThreads) {
      const unsigned int even_k = pair * 2U;
      float even = 0.0F;
      float odd = 0.0F;
      if (valid_row) {
        even = decode_bf16(shared.row[even_k]) * inverse_alpha_fp32[even_k];
        odd = decode_bf16(shared.row[even_k + 1U]) *
              inverse_alpha_fp32[even_k + 1U];
        even = fminf(fmaxf(even, -shared.clipped_maximum),
                     shared.clipped_maximum);
        odd = fminf(fmaxf(odd, -shared.clipped_maximum),
                    shared.clipped_maximum);
      }
      const int even_rounded =
          valid_row ? __float2int_rn(even / shared.stored_scale) : 0;
      const int odd_rounded =
          valid_row ? __float2int_rn(odd / shared.stored_scale) : 0;
      const int even_code = even_rounded < -7
                                ? -7
                                : (even_rounded > 7 ? 7 : even_rounded);
      const int odd_code = odd_rounded < -7
                               ? -7
                               : (odd_rounded > 7 ? 7 : odd_rounded);
      packed_a[sm87_a4w4_consumer_packed_offset(
          row, even_k / 64U, (even_k & 63U) / 2U,
          kPhysicalK64Groups)] =
          sm87_a4w4_pack_signed_pair(even_code, odd_code);
    }
    if (threadIdx.x == 0U) {
      a_lane_scales_bf16[
          sm87_a4w4_factorized_lane_scale_offset(row, 0U, 1U)] =
          shared.scale_bits;
    }
  }
}

__device__ __forceinline__ void publish_k256_group(
    const HandoffShared& shared,
    const unsigned int row,
    const bool valid_row,
    const unsigned int group,
    const float clip_ratio,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_k256_scales_bf16) noexcept {
  const unsigned int lane = threadIdx.x & 31U;
  float values[8U];
  float maximum = 0.0F;
  if (valid_row) {
    const unsigned int offset = group * 256U + lane * 8U;
#pragma unroll
    for (unsigned int index = 0U; index < 8U; ++index) {
      values[index] = decode_bf16(shared.row[offset + index]);
      maximum = fmaxf(maximum, fabsf(values[index]));
    }
  } else {
#pragma unroll
    for (unsigned int index = 0U; index < 8U; ++index) {
      values[index] = 0.0F;
    }
  }
  maximum = warp_max(maximum);
  maximum = __shfl_sync(0xffff'ffffU, maximum, 0U);
  const float clipped_maximum = maximum * clip_ratio;
  std::uint16_t scale_bits =
      encode_bf16(maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
  float stored_scale = decode_bf16(scale_bits);
  if (maximum != 0.0F && stored_scale == 0.0F) {
    scale_bits = 1U;
    stored_scale = decode_bf16(scale_bits);
  }
  const unsigned int physical_group = group * 4U + lane / 8U;
  const unsigned int first_byte = 4U * (lane % 8U);
#pragma unroll
  for (unsigned int pair = 0U; pair < 4U; ++pair) {
    const float even = fminf(fmaxf(values[2U * pair], -clipped_maximum),
                             clipped_maximum);
    const float odd = fminf(
        fmaxf(values[2U * pair + 1U], -clipped_maximum), clipped_maximum);
    const int even_rounded =
        stored_scale == 0.0F ? 0 : __float2int_rn(even / stored_scale);
    const int odd_rounded =
        stored_scale == 0.0F ? 0 : __float2int_rn(odd / stored_scale);
    const int even_code = even_rounded < -7
                              ? -7
                              : (even_rounded > 7 ? 7 : even_rounded);
    const int odd_code = odd_rounded < -7
                             ? -7
                             : (odd_rounded > 7 ? 7 : odd_rounded);
    packed_a[sm87_a4w4_attention_k256_packed_offset(
        row, physical_group, first_byte + pair, kPhysicalK64Groups)] =
        sm87_a4w4_pack_signed_pair(even_code, odd_code);
  }
  if (lane == 0U) {
    a_k256_scales_bf16[sm87_a4w4_attention_k256_scale_offset(
        row, group, kK256Groups)] = scale_bits;
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4PrefillHandoffThreads,
                      kSm87A4W4PrefillHandoffMinimumBlocksPerSm)
void q3x_sm87_attention_residual_post_norm_r1_quantize_kernel(
    const std::uint16_t* const left_bf16,
    const std::uint16_t* const right_bf16,
    const std::uint16_t* const centered_norm_weight_bf16,
    const float* const inverse_alpha_fp32,
    const unsigned int logical_token_count,
    const float epsilon,
    const float clip_ratio,
    std::uint16_t* const residual_output_bf16,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_lane_scales_bf16) {
  __shared__ HandoffShared shared;
  const unsigned int row = blockIdx.x;
  const bool valid_row = row < logical_token_count;
  if (valid_row) {
    publish_residual_and_normalized_row(
        shared, left_bf16, right_bf16, centered_norm_weight_bf16, row,
        epsilon, residual_output_bf16, nullptr);
  }
  publish_r1_row(shared, inverse_alpha_fp32, row, valid_row, clip_ratio,
                 packed_a, a_lane_scales_bf16);
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4PrefillHandoffThreads,
                      kSm87A4W4PrefillHandoffMinimumBlocksPerSm)
void q3x_sm87_mlp_residual_next_norm_k256_quantize_kernel(
    const std::uint16_t* const left_bf16,
    const std::uint16_t* const right_bf16,
    const std::uint16_t* const centered_norm_weight_bf16,
    const unsigned int logical_token_count,
    const float epsilon,
    const float clip_ratio,
    std::uint16_t* const residual_output_bf16,
    std::uint16_t* const normalized_output_bf16,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_k256_scales_bf16) {
  __shared__ HandoffShared shared;
  const unsigned int row = blockIdx.x;
  const bool valid_row = row < logical_token_count;
  if (valid_row) {
    publish_residual_and_normalized_row(
        shared, left_bf16, right_bf16, centered_norm_weight_bf16, row,
        epsilon, residual_output_bf16, normalized_output_bf16);
  }
  const unsigned int warp = threadIdx.x >> 5U;
  for (unsigned int group = warp; group < kK256Groups; group += kWarps) {
    publish_k256_group(shared, row, valid_row, group, clip_ratio, packed_a,
                       a_k256_scales_bf16);
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4PrefillHandoffThreads,
                      kSm87A4W4PrefillHandoffMinimumBlocksPerSm)
void q3x_sm87_mlp_residual_next_norm_attention_r1_quantize_kernel(
    const std::uint16_t* const left_bf16,
    const std::uint16_t* const right_bf16,
    const std::uint16_t* const centered_norm_weight_bf16,
    const float* const inverse_alpha_fp32,
    const unsigned int logical_token_count,
    const float epsilon,
    const float clip_ratio,
    std::uint16_t* const residual_output_bf16,
    std::uint16_t* const normalized_output_bf16,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_lane_scales_bf16) {
  __shared__ HandoffShared shared;
  const unsigned int row = blockIdx.x;
  const bool valid_row = row < logical_token_count;
  if (valid_row) {
    publish_residual_and_normalized_row(
        shared, left_bf16, right_bf16, centered_norm_weight_bf16, row,
        epsilon, residual_output_bf16, normalized_output_bf16);
  }
  publish_r1_row(shared, inverse_alpha_fp32, row, valid_row, clip_ratio,
                 packed_a, a_lane_scales_bf16);
}

[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

struct ByteRange final {
  const void* pointer{};
  std::size_t bytes{};
};

[[nodiscard]] bool byte_ranges_overlap(
    const ByteRange& first, const ByteRange& second) noexcept {
  if (first.pointer == nullptr || second.pointer == nullptr ||
      first.bytes == 0U || second.bytes == 0U) {
    return true;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first.pointer);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second.pointer);
  constexpr std::uintptr_t maximum =
      std::numeric_limits<std::uintptr_t>::max();
  if (first.bytes > maximum - first_begin ||
      second.bytes > maximum - second_begin) {
    return true;
  }
  const std::uintptr_t first_end = first_begin + first.bytes;
  const std::uintptr_t second_end = second_begin + second.bytes;
  return first_begin < second_end && second_begin < first_end;
}

template <std::size_t Count>
[[nodiscard]] bool ranges_pairwise_disjoint(
    const ByteRange (&ranges)[Count]) noexcept {
  for (std::size_t first = 0U; first < Count; ++first) {
    for (std::size_t second = first + 1U; second < Count; ++second) {
      if (byte_ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool valid_scalar_arguments(
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const float epsilon,
    const float clip_ratio) noexcept {
  return logical_token_count != 0U &&
         logical_token_count <= launch_token_count &&
         launch_token_count <= kSm87A4W4PrefillHandoffMaximumTokens &&
         launch_token_count % kSm87A4W4ConsumerOuterBlock == 0U &&
         epsilon > 0.0F && epsilon < std::numeric_limits<float>::infinity() &&
         clip_ratio > 0.0F && clip_ratio <= 1.0F;
}

[[nodiscard]] bool checked_multiply(
    const std::size_t first,
    const std::size_t second,
    std::size_t* const output) noexcept {
  if (output == nullptr ||
      (first != 0U &&
       second > std::numeric_limits<std::size_t>::max() / first)) {
    return false;
  }
  *output = first * second;
  return true;
}

[[nodiscard]] int validate_target(
    cudaDeviceProp* const output = nullptr) noexcept {
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
      properties.multiProcessorCount != kRequiredSmCount) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (output != nullptr) {
    *output = properties;
  }
  return static_cast<int>(cudaSuccess);
}

template <class Kernel>
[[nodiscard]] int query_one_kernel(
    const Kernel kernel,
    int* const registers,
    std::size_t* const shared_bytes,
    std::size_t* const local_bytes,
    int* const maximum_threads,
    int* const active_blocks) noexcept {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int occupancy = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &occupancy, kernel, static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers = attributes.numRegs;
  *shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads = attributes.maxThreadsPerBlock;
  *active_blocks = occupancy;
  return static_cast<int>(cudaSuccess);
}

}  // namespace

int query_sm87_a4w4_prefill_handoff_resources_cuda(
    Sm87A4W4PrefillHandoffResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaDeviceProp properties{};
  int status = validate_target(&properties);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  int attention_maximum_threads = 0;
  int mlp_maximum_threads = 0;
  int mlp_r1_maximum_threads = 0;
  status = query_one_kernel(
      q3x_sm87_attention_residual_post_norm_r1_quantize_kernel,
      &resources->attention_to_gate_registers_per_thread,
      &resources->attention_to_gate_static_shared_bytes,
      &resources->attention_to_gate_local_bytes,
      &attention_maximum_threads,
      &resources->attention_to_gate_active_blocks_per_sm);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = query_one_kernel(
      q3x_sm87_mlp_residual_next_norm_k256_quantize_kernel,
      &resources->mlp_to_attention_registers_per_thread,
      &resources->mlp_to_attention_static_shared_bytes,
      &resources->mlp_to_attention_local_bytes,
      &mlp_maximum_threads,
      &resources->mlp_to_attention_active_blocks_per_sm);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = query_one_kernel(
      q3x_sm87_mlp_residual_next_norm_attention_r1_quantize_kernel,
      &resources->mlp_to_attention_r1_registers_per_thread,
      &resources->mlp_to_attention_r1_static_shared_bytes,
      &resources->mlp_to_attention_r1_local_bytes,
      &mlp_r1_maximum_threads,
      &resources->mlp_to_attention_r1_active_blocks_per_sm);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  resources->maximum_threads_per_block =
      attention_maximum_threads < mlp_maximum_threads
          ? (attention_maximum_threads < mlp_r1_maximum_threads
                 ? attention_maximum_threads
                 : mlp_r1_maximum_threads)
          : (mlp_maximum_threads < mlp_r1_maximum_threads
                 ? mlp_maximum_threads
                 : mlp_r1_maximum_threads);
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->multiprocessor_count = properties.multiProcessorCount;
  if (resources->attention_to_gate_registers_per_thread <= 0 ||
      resources->attention_to_gate_registers_per_thread >
          static_cast<int>(kSm87A4W4PrefillHandoffMaximumRegisters) ||
      resources->mlp_to_attention_registers_per_thread <= 0 ||
      resources->mlp_to_attention_registers_per_thread >
          static_cast<int>(kSm87A4W4PrefillHandoffMaximumRegisters) ||
      resources->mlp_to_attention_r1_registers_per_thread <= 0 ||
      resources->mlp_to_attention_r1_registers_per_thread >
          static_cast<int>(kSm87A4W4PrefillHandoffMaximumRegisters) ||
      resources->attention_to_gate_local_bytes != 0U ||
      resources->mlp_to_attention_local_bytes != 0U ||
      resources->mlp_to_attention_r1_local_bytes != 0U ||
      resources->maximum_threads_per_block < static_cast<int>(kThreads) ||
      resources->attention_to_gate_active_blocks_per_sm <
          static_cast<int>(kSm87A4W4PrefillHandoffMinimumBlocksPerSm) ||
      resources->mlp_to_attention_active_blocks_per_sm <
          static_cast<int>(kSm87A4W4PrefillHandoffMinimumBlocksPerSm) ||
      resources->mlp_to_attention_r1_active_blocks_per_sm <
          static_cast<int>(kSm87A4W4PrefillHandoffMinimumBlocksPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_attention_residual_post_norm_r1_quantize_cuda(
    const std::uint16_t* const left_bf16,
    const std::uint16_t* const right_bf16,
    const std::uint16_t* const centered_norm_weight_bf16,
    const float* const authenticated_inverse_alpha_fp32,
    const std::size_t inverse_alpha_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const float epsilon,
    const float clip_ratio,
    std::uint16_t* const residual_output_bf16,
    const std::size_t residual_output_capacity_elements,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  std::size_t logical_elements = 0U;
  const std::size_t required_packed =
      sm87_a4w4_consumer_packed_capacity_bytes(
          launch_token_count, kSm87A4W4PrefillHandoffHiddenSize);
  const std::size_t required_scales =
      sm87_a4w4_factorized_lane_scale_capacity_elements(
          launch_token_count, 1U);
  if (!valid_scalar_arguments(logical_token_count, launch_token_count,
                              epsilon, clip_ratio) ||
      !checked_multiply(logical_token_count,
                        kSm87A4W4PrefillHandoffHiddenSize,
                        &logical_elements) ||
      launch_token_count !=
          sm87_a4w4_attention_k256_launch_token_count(
              logical_token_count) ||
      inverse_alpha_capacity_elements <
          kSm87A4W4PrefillHandoffHiddenSize ||
      residual_output_capacity_elements < logical_elements ||
      required_packed == 0U || packed_a_capacity_bytes < required_packed ||
      required_scales == 0U || a_scale_capacity_elements < required_scales ||
      !aligned(left_bf16, alignof(std::uint16_t)) ||
      !aligned(right_bf16, alignof(std::uint16_t)) ||
      !aligned(centered_norm_weight_bf16, alignof(std::uint16_t)) ||
      !aligned(authenticated_inverse_alpha_fp32, alignof(float)) ||
      !aligned(residual_output_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_a, 16U) || !aligned(a_lane_scales_bf16, 16U) ||
      logical_token_count > std::numeric_limits<unsigned int>::max() ||
      launch_token_count > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t logical_bytes =
      logical_elements * sizeof(std::uint16_t);
  const ByteRange ranges[] = {
      {left_bf16, logical_bytes},
      {right_bf16, logical_bytes},
      {centered_norm_weight_bf16,
       kSm87A4W4PrefillHandoffHiddenSize * sizeof(std::uint16_t)},
      {authenticated_inverse_alpha_fp32,
       kSm87A4W4PrefillHandoffHiddenSize * sizeof(float)},
      {residual_output_bf16, logical_bytes},
      {packed_a, required_packed},
      {a_lane_scales_bf16, required_scales * sizeof(std::uint16_t)}};
  if (!ranges_pairwise_disjoint(ranges)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int target_status = validate_target();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_attention_residual_post_norm_r1_quantize_kernel<<<
      static_cast<unsigned int>(launch_token_count), kThreads, 0U, stream>>>(
      left_bf16, right_bf16, centered_norm_weight_bf16,
      authenticated_inverse_alpha_fp32,
      static_cast<unsigned int>(logical_token_count), epsilon, clip_ratio,
      residual_output_bf16, packed_a, a_lane_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_a4w4_mlp_residual_next_norm_k256_quantize_cuda(
    const std::uint16_t* const left_bf16,
    const std::uint16_t* const right_bf16,
    const std::uint16_t* const next_centered_norm_weight_bf16,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const float epsilon,
    const float clip_ratio,
    std::uint16_t* const residual_output_bf16,
    const std::size_t residual_output_capacity_elements,
    std::uint16_t* const normalized_output_bf16,
    const std::size_t normalized_output_capacity_elements,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const a_k256_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  std::size_t logical_elements = 0U;
  const std::size_t required_packed =
      sm87_a4w4_attention_k256_packed_capacity_bytes(
          launch_token_count, kSm87A4W4PrefillHandoffHiddenSize);
  const std::size_t required_scales =
      sm87_a4w4_attention_k256_scale_capacity_elements(
          launch_token_count, kSm87A4W4PrefillHandoffHiddenSize);
  if (!valid_scalar_arguments(logical_token_count, launch_token_count,
                              epsilon, clip_ratio) ||
      launch_token_count !=
          sm87_a4w4_attention_k256_launch_token_count(
              logical_token_count) ||
      !checked_multiply(logical_token_count,
                        kSm87A4W4PrefillHandoffHiddenSize,
                        &logical_elements) ||
      residual_output_capacity_elements < logical_elements ||
      normalized_output_capacity_elements < logical_elements ||
      required_packed == 0U || packed_a_capacity_bytes < required_packed ||
      required_scales == 0U || a_scale_capacity_elements < required_scales ||
      !aligned(left_bf16, alignof(std::uint16_t)) ||
      !aligned(right_bf16, alignof(std::uint16_t)) ||
      !aligned(next_centered_norm_weight_bf16, alignof(std::uint16_t)) ||
      !aligned(residual_output_bf16, alignof(std::uint16_t)) ||
      !aligned(normalized_output_bf16, alignof(std::uint16_t)) ||
      !aligned(packed_a, 16U) || !aligned(a_k256_scales_bf16, 16U) ||
      logical_token_count > std::numeric_limits<unsigned int>::max() ||
      launch_token_count > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t logical_bytes =
      logical_elements * sizeof(std::uint16_t);
  // In-place publication of the rounded MLP residual over `left` is the
  // production contract.  Every other ownership domain remains disjoint.
  const bool residual_aliases_left = residual_output_bf16 == left_bf16;
  const ByteRange independent_ranges[] = {
      {right_bf16, logical_bytes},
      {next_centered_norm_weight_bf16,
       kSm87A4W4PrefillHandoffHiddenSize * sizeof(std::uint16_t)},
      {normalized_output_bf16, logical_bytes},
      {packed_a, required_packed},
      {a_k256_scales_bf16, required_scales * sizeof(std::uint16_t)}};
  if (!ranges_pairwise_disjoint(independent_ranges)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ByteRange left_range{left_bf16, logical_bytes};
  const ByteRange residual_range{residual_output_bf16, logical_bytes};
  for (const ByteRange& range : independent_ranges) {
    if (byte_ranges_overlap(left_range, range) ||
        byte_ranges_overlap(residual_range, range)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  if (!residual_aliases_left &&
      byte_ranges_overlap(left_range, residual_range)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int target_status = validate_target();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_mlp_residual_next_norm_k256_quantize_kernel<<<
      static_cast<unsigned int>(launch_token_count), kThreads, 0U, stream>>>(
      left_bf16, right_bf16, next_centered_norm_weight_bf16,
      static_cast<unsigned int>(logical_token_count), epsilon, clip_ratio,
      residual_output_bf16, normalized_output_bf16, packed_a,
      a_k256_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_a4w4_mlp_residual_next_norm_attention_r1_quantize_cuda(
    const std::uint16_t* const left_bf16,
    const std::uint16_t* const right_bf16,
    const std::uint16_t* const next_centered_norm_weight_bf16,
    const float* const authenticated_inverse_alpha_fp32,
    const std::size_t inverse_alpha_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const float epsilon,
    const float clip_ratio,
    std::uint16_t* const residual_output_bf16,
    const std::size_t residual_output_capacity_elements,
    std::uint16_t* const normalized_output_bf16,
    const std::size_t normalized_output_capacity_elements,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  std::size_t logical_elements = 0U;
  const std::size_t required_packed =
      sm87_a4w4_consumer_packed_capacity_bytes(
          launch_token_count, kSm87A4W4PrefillHandoffHiddenSize);
  const std::size_t required_scales =
      sm87_a4w4_factorized_lane_scale_capacity_elements(
          launch_token_count, 1U);
  if (!valid_scalar_arguments(logical_token_count, launch_token_count,
                              epsilon, clip_ratio) ||
      launch_token_count !=
          sm87_a4w4_attention_k256_launch_token_count(logical_token_count) ||
      !checked_multiply(logical_token_count,
                        kSm87A4W4PrefillHandoffHiddenSize,
                        &logical_elements) ||
      inverse_alpha_capacity_elements <
          kSm87A4W4PrefillHandoffHiddenSize ||
      residual_output_capacity_elements < logical_elements ||
      ((normalized_output_bf16 == nullptr) !=
       (normalized_output_capacity_elements == 0U)) ||
      (normalized_output_bf16 != nullptr &&
       normalized_output_capacity_elements < logical_elements) ||
      required_packed == 0U || packed_a_capacity_bytes < required_packed ||
      required_scales == 0U || a_scale_capacity_elements < required_scales ||
      !aligned(left_bf16, alignof(std::uint16_t)) ||
      !aligned(right_bf16, alignof(std::uint16_t)) ||
      !aligned(next_centered_norm_weight_bf16, alignof(std::uint16_t)) ||
      !aligned(authenticated_inverse_alpha_fp32, alignof(float)) ||
      !aligned(residual_output_bf16, alignof(std::uint16_t)) ||
      (normalized_output_bf16 != nullptr &&
       !aligned(normalized_output_bf16, alignof(std::uint16_t))) ||
      !aligned(packed_a, 16U) || !aligned(a_lane_scales_bf16, 16U) ||
      logical_token_count > std::numeric_limits<unsigned int>::max() ||
      launch_token_count > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::size_t logical_bytes =
      logical_elements * sizeof(std::uint16_t);
  const bool residual_aliases_left = residual_output_bf16 == left_bf16;
  const ByteRange independent_ranges[] = {
      {right_bf16, logical_bytes},
      {next_centered_norm_weight_bf16,
       kSm87A4W4PrefillHandoffHiddenSize * sizeof(std::uint16_t)},
      {authenticated_inverse_alpha_fp32,
       kSm87A4W4PrefillHandoffHiddenSize * sizeof(float)},
      {packed_a, required_packed},
      {a_lane_scales_bf16, required_scales * sizeof(std::uint16_t)}};
  if (!ranges_pairwise_disjoint(independent_ranges)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ByteRange left_range{left_bf16, logical_bytes};
  const ByteRange residual_range{residual_output_bf16, logical_bytes};
  const ByteRange normalized_range{normalized_output_bf16,
                                   normalized_output_bf16 == nullptr
                                       ? 0U
                                       : logical_bytes};
  for (const ByteRange& range : independent_ranges) {
    if (byte_ranges_overlap(left_range, range) ||
        byte_ranges_overlap(residual_range, range) ||
        (normalized_output_bf16 != nullptr &&
         byte_ranges_overlap(normalized_range, range))) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  if (!residual_aliases_left &&
      byte_ranges_overlap(left_range, residual_range)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (normalized_output_bf16 != nullptr &&
      (byte_ranges_overlap(left_range, normalized_range) ||
       byte_ranges_overlap(residual_range, normalized_range))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const int target_status = validate_target();
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_mlp_residual_next_norm_attention_r1_quantize_kernel<<<
      static_cast<unsigned int>(launch_token_count), kThreads, 0U, stream>>>(
      left_bf16, right_bf16, next_centered_norm_weight_bf16,
      authenticated_inverse_alpha_fp32,
      static_cast<unsigned int>(logical_token_count), epsilon, clip_ratio,
      residual_output_bf16, normalized_output_bf16, packed_a,
      a_lane_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
