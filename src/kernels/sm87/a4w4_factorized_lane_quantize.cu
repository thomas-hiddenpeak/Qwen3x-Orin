#include "q3x/kernels/sm87_a4w4_factorized_lane_quantize.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr int kRequiredSmCount = 16;
inline constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87A4W4FactorizedLaneQuantizeThreads);
inline constexpr unsigned int kWarps = kThreads / 32U;

struct alignas(16) ReductionStorage final {
  float warp_maxima[kWarps];
  float clipped_maximum;
  float stored_scale;
  std::uint16_t scale_bits;
  std::uint16_t reserved;
};

static_assert(kThreads == 256U);
static_assert(kWarps == 8U);
static_assert(sizeof(ReductionStorage) == 48U);

[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] constexpr bool checked_multiply(
    const std::size_t first, const std::size_t second,
    std::size_t* const output) noexcept {
  if (output == nullptr ||
      (first != 0U &&
       second > std::numeric_limits<std::size_t>::max() / first)) {
    return false;
  }
  *output = first * second;
  return true;
}

[[nodiscard]] constexpr bool checked_add(
    const std::size_t first, const std::size_t second,
    std::size_t* const output) noexcept {
  if (output == nullptr ||
      second > std::numeric_limits<std::size_t>::max() - first) {
    return false;
  }
  *output = first + second;
  return true;
}

[[nodiscard]] constexpr bool required_strided_span_elements(
    const std::size_t rows, const std::size_t row_stride,
    const std::size_t row_width, std::size_t* const elements) noexcept {
  if (rows == 0U || row_width == 0U || row_stride < row_width ||
      elements == nullptr) {
    return false;
  }
  std::size_t row_base = 0U;
  return checked_multiply(rows - 1U, row_stride, &row_base) &&
         checked_add(row_base, row_width, elements);
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
[[nodiscard]] bool ranges_are_pairwise_disjoint(
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
                  __shfl_down_sync(0xffffffffU, value, delta));
  }
  return value;
}

template <bool Split>
[[nodiscard]] __device__ __forceinline__ float load_transformed(
    const std::uint16_t* const primary_bf16,
    const unsigned int primary_row_stride_elements,
    const unsigned int primary_size,
    const std::uint16_t* const secondary_bf16,
    const unsigned int secondary_row_stride_elements,
    const float* const inverse_alpha_fp32, const unsigned int row,
    const unsigned int global_k) noexcept {
  const std::uint16_t bits = [&]() {
    if constexpr (!Split) {
      return primary_bf16[row * primary_row_stride_elements + global_k];
    } else {
      return global_k < primary_size
                 ? primary_bf16[row * primary_row_stride_elements +
                                global_k]
                 : secondary_bf16[
                       row * secondary_row_stride_elements + global_k -
                       primary_size];
    }
  }();
  return decode_bf16(bits) * inverse_alpha_fp32[global_k];
}

[[nodiscard]] __device__ __forceinline__ unsigned int packed_offset(
    const unsigned int row, const unsigned int global_k,
    const unsigned int physical_k64_group_count) noexcept {
  const unsigned int outer_block = row >> 6U;
  const unsigned int row_in_block = row & 63U;
  const unsigned int physical_group = global_k >> 6U;
  const unsigned int byte_in_group = (global_k & 63U) >> 1U;
  return (((outer_block * physical_k64_group_count + physical_group) *
               64U +
           row_in_block) *
              32U +
          byte_in_group);
}

[[nodiscard]] __device__ __forceinline__ unsigned int scale_offset(
    const unsigned int row, const unsigned int lane,
    const unsigned int lane_count) noexcept {
  return (((row >> 6U) * lane_count + lane) * 64U + (row & 63U));
}

template <bool Split>
__device__ __forceinline__ void factorized_lane_quantize_body(
    ReductionStorage& shared, const std::uint16_t* const primary_bf16,
    const unsigned int primary_row_stride_elements,
    const unsigned int primary_size,
    const std::uint16_t* const secondary_bf16,
    const unsigned int secondary_row_stride_elements,
    const float* const inverse_alpha_fp32,
    const unsigned int logical_token_count, const unsigned int input_size,
    const unsigned int lane_count, const float clip_ratio,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_lane_scales_bf16) noexcept {
  const unsigned int row = blockIdx.x / lane_count;
  const unsigned int lane_index = blockIdx.x - row * lane_count;
  const unsigned int lane_input_size = input_size / lane_count;
  const unsigned int lane_begin = lane_index * lane_input_size;
  const unsigned int lane_end = lane_begin + lane_input_size;
  const bool valid_row = row < logical_token_count;

  float maximum = 0.0F;
  if (valid_row) {
    for (unsigned int global_k = lane_begin + threadIdx.x;
         global_k < lane_end; global_k += kThreads) {
      maximum = fmaxf(
          maximum,
          fabsf(load_transformed<Split>(
              primary_bf16, primary_row_stride_elements, primary_size,
              secondary_bf16, secondary_row_stride_elements,
              inverse_alpha_fp32, row, global_k)));
    }
  }

  maximum = warp_max(maximum);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  if (lane == 0U) {
    shared.warp_maxima[warp] = maximum;
  }
  __syncthreads();

  if (warp == 0U) {
    float block_maximum = lane < kWarps ? shared.warp_maxima[lane] : 0.0F;
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

  const unsigned int pair_count = lane_input_size >> 1U;
  const unsigned int physical_k64_group_count = input_size >> 6U;
  for (unsigned int pair = threadIdx.x; pair < pair_count;
       pair += kThreads) {
    const unsigned int global_even = lane_begin + (pair << 1U);
    float even = 0.0F;
    float odd = 0.0F;
    if (valid_row) {
      even = load_transformed<Split>(
          primary_bf16, primary_row_stride_elements, primary_size,
          secondary_bf16, secondary_row_stride_elements,
          inverse_alpha_fp32, row, global_even);
      odd = load_transformed<Split>(
          primary_bf16, primary_row_stride_elements, primary_size,
          secondary_bf16, secondary_row_stride_elements,
          inverse_alpha_fp32, row, global_even + 1U);
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
    packed_a[packed_offset(row, global_even,
                           physical_k64_group_count)] =
        sm87_a4w4_pack_signed_pair(even_code, odd_code);
  }
  if (threadIdx.x == 0U) {
    a_lane_scales_bf16[scale_offset(row, lane_index, lane_count)] =
        shared.scale_bits;
  }
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4FactorizedLaneQuantizeThreads,
                      kSm87A4W4FactorizedLaneQuantizeMinimumActiveBlocksPerSm)
void q3x_sm87_a4w4_factorized_lane_quantize_contiguous_kernel(
    const std::uint16_t* const primary_bf16,
    const unsigned int primary_row_stride_elements,
    const unsigned int primary_size,
    const std::uint16_t* const secondary_bf16,
    const unsigned int secondary_row_stride_elements,
    const float* const inverse_alpha_fp32,
    const unsigned int logical_token_count, const unsigned int input_size,
    const unsigned int lane_count, const float clip_ratio,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_lane_scales_bf16) {
  __shared__ ReductionStorage shared;
  factorized_lane_quantize_body<false>(
      shared, primary_bf16, primary_row_stride_elements, primary_size,
      secondary_bf16, secondary_row_stride_elements, inverse_alpha_fp32,
      logical_token_count, input_size, lane_count, clip_ratio, packed_a,
      a_lane_scales_bf16);
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4FactorizedLaneQuantizeThreads,
                      kSm87A4W4FactorizedLaneQuantizeMinimumActiveBlocksPerSm)
void q3x_sm87_a4w4_factorized_lane_quantize_split_kernel(
    const std::uint16_t* const primary_bf16,
    const unsigned int primary_row_stride_elements,
    const unsigned int primary_size,
    const std::uint16_t* const secondary_bf16,
    const unsigned int secondary_row_stride_elements,
    const float* const inverse_alpha_fp32,
    const unsigned int logical_token_count, const unsigned int input_size,
    const unsigned int lane_count, const float clip_ratio,
    std::uint8_t* const packed_a,
    std::uint16_t* const a_lane_scales_bf16) {
  __shared__ ReductionStorage shared;
  factorized_lane_quantize_body<true>(
      shared, primary_bf16, primary_row_stride_elements, primary_size,
      secondary_bf16, secondary_row_stride_elements, inverse_alpha_fp32,
      logical_token_count, input_size, lane_count, clip_ratio, packed_a,
      a_lane_scales_bf16);
}

[[nodiscard]] int validate_target(cudaDeviceProp* const output = nullptr,
                                  int* const device_output = nullptr) noexcept {
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
  if (device_output != nullptr) {
    *device_output = device;
  }
  return static_cast<int>(cudaSuccess);
}

struct ResourceAccumulator final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{std::numeric_limits<int>::max()};
  int active_blocks_per_sm{std::numeric_limits<int>::max()};
};

template <class Kernel>
[[nodiscard]] cudaError_t merge_resources(
    const Kernel kernel, ResourceAccumulator* const resources) noexcept {
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return status;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, kernel, static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    return status;
  }
  resources->registers_per_thread =
      std::max(resources->registers_per_thread, attributes.numRegs);
  resources->static_shared_bytes =
      std::max(resources->static_shared_bytes, attributes.sharedSizeBytes);
  resources->local_bytes =
      std::max(resources->local_bytes, attributes.localSizeBytes);
  resources->maximum_threads_per_block = std::min(
      resources->maximum_threads_per_block, attributes.maxThreadsPerBlock);
  resources->active_blocks_per_sm =
      std::min(resources->active_blocks_per_sm, active_blocks);
  return cudaSuccess;
}

struct ResourceCache final {
  int device{-2};
  int status{static_cast<int>(cudaErrorUnknown)};
};

thread_local ResourceCache g_resource_cache{};

[[nodiscard]] int ensure_resources_cached_for_current_device() noexcept {
  int device = -1;
  const cudaError_t device_status = cudaGetDevice(&device);
  if (device_status != cudaSuccess) {
    return static_cast<int>(device_status);
  }
  if (g_resource_cache.device == device) {
    return g_resource_cache.status;
  }
  Sm87A4W4FactorizedLaneQuantizeResources resources{};
  const int status =
      query_sm87_a4w4_factorized_lane_quantize_resources_cuda(&resources);
  g_resource_cache = {device, status};
  return status;
}

struct CommonValidation final {
  Sm87A4W4FactorizedLaneQuantizePlan plan{};
  std::size_t required_packed_bytes{};
  std::size_t required_scale_elements{};
  std::size_t required_inverse_bytes{};
  std::size_t required_packed_scale_bytes{};
};

[[nodiscard]] bool validate_common(
    const float* const authenticated_inverse_alpha_fp32,
    const std::size_t inverse_alpha_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t input_size,
    const std::size_t lane_count, const float clip_ratio,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    CommonValidation* const output) noexcept {
  if (output == nullptr || !aligned(authenticated_inverse_alpha_fp32, 16U) ||
      !aligned(packed_a, 16U) ||
      !aligned(a_lane_scales_bf16, alignof(std::uint16_t)) ||
      !(clip_ratio > 0.0F && clip_ratio <= 1.0F)) {
    return false;
  }
  const auto plan = sm87_a4w4_factorized_lane_quantize_plan(
      logical_token_count, launch_token_count, input_size, lane_count);
  if (!plan.valid() ||
      inverse_alpha_capacity_elements < input_size ||
      packed_a_capacity_bytes < plan.packed_capacity_bytes ||
      a_scale_capacity_elements < plan.scale_capacity_elements ||
      !checked_multiply(input_size, sizeof(float),
                        &output->required_inverse_bytes) ||
      !checked_multiply(plan.scale_capacity_elements,
                        sizeof(std::uint16_t),
                        &output->required_packed_scale_bytes)) {
    return false;
  }
  output->plan = plan;
  output->required_packed_bytes = plan.packed_capacity_bytes;
  output->required_scale_elements = plan.scale_capacity_elements;
  return true;
}

}  // namespace

int query_sm87_a4w4_factorized_lane_quantize_resources_cuda(
    Sm87A4W4FactorizedLaneQuantizeResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = Sm87A4W4FactorizedLaneQuantizeResources{};
  cudaDeviceProp properties{};
  const int target_status = validate_target(&properties);
  if (target_status != static_cast<int>(cudaSuccess)) {
    return target_status;
  }

  ResourceAccumulator accumulator{};
  cudaError_t status = merge_resources(
      q3x_sm87_a4w4_factorized_lane_quantize_contiguous_kernel,
      &accumulator);
  if (status == cudaSuccess) {
    status = merge_resources(
        q3x_sm87_a4w4_factorized_lane_quantize_split_kernel,
        &accumulator);
  }
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->registers_per_thread = accumulator.registers_per_thread;
  resources->static_shared_bytes = accumulator.static_shared_bytes;
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = accumulator.local_bytes;
  resources->maximum_threads_per_block =
      accumulator.maximum_threads_per_block;
  resources->active_blocks_per_sm = accumulator.active_blocks_per_sm;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->multiprocessor_count = properties.multiProcessorCount;
  if (resources->registers_per_thread <= 0 ||
      resources->registers_per_thread >
          static_cast<int>(
              kSm87A4W4FactorizedLaneQuantizeMaximumRegisters) ||
      resources->local_bytes != 0U ||
      resources->maximum_threads_per_block < static_cast<int>(kThreads) ||
      resources->active_blocks_per_sm <
          static_cast<int>(
              kSm87A4W4FactorizedLaneQuantizeMinimumActiveBlocksPerSm)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_a4w4_factorized_lane_quantize_bf16_cuda(
    const std::uint16_t* const input_bf16,
    const std::size_t input_row_stride_elements,
    const std::size_t input_capacity_elements,
    const float* const authenticated_inverse_alpha_fp32,
    const std::size_t inverse_alpha_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t input_size,
    const std::size_t lane_count, const float clip_ratio,
    std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  CommonValidation validation{};
  std::size_t required_input_elements = 0U;
  std::size_t required_input_bytes = 0U;
  if (!aligned(input_bf16, alignof(std::uint16_t)) ||
      input_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      !required_strided_span_elements(
          logical_token_count, input_row_stride_elements, input_size,
          &required_input_elements) ||
      input_capacity_elements < required_input_elements ||
      !checked_multiply(required_input_elements, sizeof(std::uint16_t),
                        &required_input_bytes) ||
      !validate_common(
          authenticated_inverse_alpha_fp32,
          inverse_alpha_capacity_elements, logical_token_count,
          launch_token_count, input_size, lane_count, clip_ratio, packed_a,
          packed_a_capacity_bytes, a_lane_scales_bf16,
          a_scale_capacity_elements, &validation)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ByteRange ranges[] = {
      {input_bf16, required_input_bytes},
      {authenticated_inverse_alpha_fp32,
       validation.required_inverse_bytes},
      {packed_a, validation.required_packed_bytes},
      {a_lane_scales_bf16,
       validation.required_packed_scale_bytes}};
  if (!ranges_are_pairwise_disjoint(ranges)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int resource_status = ensure_resources_cached_for_current_device();
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_factorized_lane_quantize_contiguous_kernel<<<
      static_cast<unsigned int>(validation.plan.launch_ctas), kThreads, 0U,
      stream>>>(
      input_bf16, static_cast<unsigned int>(input_row_stride_elements),
      static_cast<unsigned int>(input_size), nullptr, 0U,
      authenticated_inverse_alpha_fp32,
      static_cast<unsigned int>(logical_token_count),
      static_cast<unsigned int>(input_size),
      static_cast<unsigned int>(lane_count), clip_ratio, packed_a,
      a_lane_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_a4w4_factorized_lane_quantize_bf16_split_cuda(
    const std::uint16_t* const primary_bf16,
    const std::size_t primary_row_stride_elements,
    const std::size_t primary_capacity_elements,
    const std::size_t primary_size,
    const std::uint16_t* const secondary_bf16,
    const std::size_t secondary_row_stride_elements,
    const std::size_t secondary_capacity_elements,
    const std::size_t secondary_size,
    const float* const authenticated_inverse_alpha_fp32,
    const std::size_t inverse_alpha_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count, const std::size_t lane_count,
    const float clip_ratio, std::uint8_t* const packed_a,
    const std::size_t packed_a_capacity_bytes,
    std::uint16_t* const a_lane_scales_bf16,
    const std::size_t a_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  if (primary_size !=
          kSm87A4W4FactorizedLaneQuantizeDownPrimaryInput ||
      secondary_size !=
          kSm87A4W4FactorizedLaneQuantizeDownSecondaryInput) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  CommonValidation validation{};
  std::size_t required_primary_elements = 0U;
  std::size_t required_secondary_elements = 0U;
  std::size_t required_primary_bytes = 0U;
  std::size_t required_secondary_bytes = 0U;
  if (!aligned(primary_bf16, alignof(std::uint16_t)) ||
      !aligned(secondary_bf16, alignof(std::uint16_t)) ||
      primary_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      secondary_row_stride_elements >
          std::numeric_limits<unsigned int>::max() ||
      !required_strided_span_elements(
          logical_token_count, primary_row_stride_elements, primary_size,
          &required_primary_elements) ||
      !required_strided_span_elements(
          logical_token_count, secondary_row_stride_elements, secondary_size,
          &required_secondary_elements) ||
      primary_capacity_elements < required_primary_elements ||
      secondary_capacity_elements < required_secondary_elements ||
      !checked_multiply(required_primary_elements, sizeof(std::uint16_t),
                        &required_primary_bytes) ||
      !checked_multiply(required_secondary_elements,
                        sizeof(std::uint16_t),
                        &required_secondary_bytes) ||
      !validate_common(
          authenticated_inverse_alpha_fp32,
          inverse_alpha_capacity_elements, logical_token_count,
          launch_token_count,
          kSm87A4W4FactorizedLaneQuantizeDownInput, lane_count, clip_ratio,
          packed_a, packed_a_capacity_bytes, a_lane_scales_bf16,
          a_scale_capacity_elements, &validation)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ByteRange ranges[] = {
      {primary_bf16, required_primary_bytes},
      {secondary_bf16, required_secondary_bytes},
      {authenticated_inverse_alpha_fp32,
       validation.required_inverse_bytes},
      {packed_a, validation.required_packed_bytes},
      {a_lane_scales_bf16,
       validation.required_packed_scale_bytes}};
  if (!ranges_are_pairwise_disjoint(ranges)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int resource_status = ensure_resources_cached_for_current_device();
  if (resource_status != static_cast<int>(cudaSuccess)) {
    return resource_status;
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  q3x_sm87_a4w4_factorized_lane_quantize_split_kernel<<<
      static_cast<unsigned int>(validation.plan.launch_ctas), kThreads, 0U,
      stream>>>(
      primary_bf16,
      static_cast<unsigned int>(primary_row_stride_elements),
      static_cast<unsigned int>(primary_size), secondary_bf16,
      static_cast<unsigned int>(secondary_row_stride_elements),
      authenticated_inverse_alpha_fp32,
      static_cast<unsigned int>(logical_token_count),
      static_cast<unsigned int>(
          kSm87A4W4FactorizedLaneQuantizeDownInput),
      static_cast<unsigned int>(lane_count), clip_ratio, packed_a,
      a_lane_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
