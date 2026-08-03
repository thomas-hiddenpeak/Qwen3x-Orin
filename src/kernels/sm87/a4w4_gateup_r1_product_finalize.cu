#include "q3x/kernels/sm87_a4w4_gateup_r1_product_finalize.h"

#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

inline constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87A4W4GateUpR1ProductFinalizeThreads);
inline constexpr unsigned int kWarps = kThreads / 32U;
inline constexpr unsigned int kInputSize = static_cast<unsigned int>(
    kSm87A4W4GateUpFactorizedModelIntermediate);
inline constexpr unsigned int kPrimarySize = static_cast<unsigned int>(
    kSm87A4W4GateUpFactorizedPrimaryWidth);
inline constexpr unsigned int kPartialTiles = static_cast<unsigned int>(
    kSm87A4W4GateUpFactorizedR1ProductPartialTiles);
inline constexpr unsigned int kPhysicalK64Groups =
    kInputSize / static_cast<unsigned int>(kSm87A4W4ConsumerKBlock);
inline constexpr int kRequiredSmCount = 16;

struct alignas(16) ReductionStorage final {
  float warp_maxima[kWarps];
  float clipped_maximum;
  float stored_scale;
  std::uint16_t scale_bits;
  std::uint16_t reserved;
};

static_assert(kThreads == 256U);
static_assert(kWarps == 8U);
static_assert(kInputSize == 17'408U);
static_assert(kPrimarySize == 12'288U);
static_assert(kPartialTiles == 136U);
static_assert(kPhysicalK64Groups == 272U);
static_assert(sizeof(ReductionStorage) == 48U);

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

[[nodiscard]] __device__ __forceinline__ std::uint16_t load_product_bf16(
    const std::uint16_t* const primary_product_bf16,
    const unsigned int primary_product_row_stride_elements,
    const std::uint16_t* const secondary_product_bf16,
    const unsigned int secondary_product_row_stride_elements,
    const unsigned int row,
    const unsigned int global_k) noexcept {
  return global_k < kPrimarySize
             ? primary_product_bf16[
                   static_cast<std::size_t>(row) *
                       primary_product_row_stride_elements +
                   global_k]
             : secondary_product_bf16[
                   static_cast<std::size_t>(row) *
                       secondary_product_row_stride_elements +
                   global_k - kPrimarySize];
}

extern "C" __global__
    __launch_bounds__(kSm87A4W4GateUpR1ProductFinalizeThreads,
                      kSm87A4W4GateUpR1ProductFinalizeMinimumBlocksPerSm)
void q3x_sm87_a4w4_gateup_r1_product_finalize_kernel(
    const std::uint16_t* const primary_product_bf16,
    const unsigned int primary_product_row_stride_elements,
    const std::uint16_t* const secondary_product_bf16,
    const unsigned int secondary_product_row_stride_elements,
    const float* const down_inverse_alpha_fp32,
    const float* const r1_product_tile_maxima_fp32,
    const unsigned int logical_token_count,
    const float clip_ratio,
    std::uint8_t* const packed_down_a,
    std::uint16_t* const down_a_lane_scales_bf16) {
  __shared__ ReductionStorage shared;
  const unsigned int row = blockIdx.x;
  const bool valid_row = row < logical_token_count;

  float maximum = 0.0F;
  // The Down M256 launch may extend beyond the GateUp M128 producer launch.
  // Every such row is also logically padded, so this validity gate prevents
  // all product, inverse-alpha, and partial-sidecar reads for both padding
  // regions while the loops below still publish zero codes and BF16 one.
  if (valid_row && threadIdx.x < kPartialTiles) {
    maximum = r1_product_tile_maxima_fp32[
        static_cast<std::size_t>(row) * kPartialTiles + threadIdx.x];
  }
  maximum = warp_max(maximum);
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  if (lane == 0U) {
    shared.warp_maxima[warp] = maximum;
  }
  __syncthreads();

  if (warp == 0U) {
    float row_maximum = lane < kWarps ? shared.warp_maxima[lane] : 0.0F;
    row_maximum = warp_max(row_maximum);
    if (lane == 0U) {
      const float clipped_maximum = row_maximum * clip_ratio;
      std::uint16_t scale_bits = encode_bf16(
          row_maximum == 0.0F ? 1.0F : clipped_maximum / 7.0F);
      float stored_scale = decode_bf16(scale_bits);
      if (row_maximum != 0.0F && stored_scale == 0.0F) {
        scale_bits = 1U;
        stored_scale = decode_bf16(scale_bits);
      }
      shared.clipped_maximum = clipped_maximum;
      shared.stored_scale = stored_scale;
      shared.scale_bits = scale_bits;
    }
  }
  __syncthreads();

  constexpr unsigned int kPairCount = kInputSize / 2U;
  for (unsigned int pair = threadIdx.x; pair < kPairCount;
       pair += kThreads) {
    const unsigned int even_k = pair * 2U;
    float even = 0.0F;
    float odd = 0.0F;
    if (valid_row) {
      // Each BF16 product element is loaded exactly once in the finalizer.
      // The transformed values are consumed immediately and never staged.
      even = decode_bf16(load_product_bf16(
                 primary_product_bf16,
                 primary_product_row_stride_elements,
                 secondary_product_bf16,
                 secondary_product_row_stride_elements, row, even_k)) *
             down_inverse_alpha_fp32[even_k];
      odd = decode_bf16(load_product_bf16(
                primary_product_bf16,
                primary_product_row_stride_elements,
                secondary_product_bf16,
                secondary_product_row_stride_elements, row,
                even_k + 1U)) *
            down_inverse_alpha_fp32[even_k + 1U];
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
    packed_down_a[sm87_a4w4_consumer_packed_offset(
        row, even_k / kSm87A4W4ConsumerKBlock,
        (even_k & (kSm87A4W4ConsumerKBlock - 1U)) / 2U,
        kPhysicalK64Groups)] =
        sm87_a4w4_pack_signed_pair(even_code, odd_code);
  }
  if (threadIdx.x == 0U) {
    down_a_lane_scales_bf16[
        sm87_a4w4_factorized_lane_scale_offset(row, 0U, 1U)] =
        shared.scale_bits;
  }
}

[[nodiscard]] constexpr bool aligned(
    const void* const pointer,
    const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] constexpr bool checked_multiply(
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

struct ByteRange final {
  const void* pointer{};
  std::size_t bytes{};
};

[[nodiscard]] bool byte_ranges_overlap(
    const ByteRange& first,
    const ByteRange& second) noexcept {
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
  return first_begin < second_begin + second.bytes &&
         second_begin < first_begin + first.bytes;
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

struct ResourceCache final {
  Sm87A4W4GateUpR1ProductFinalizeResources resources{};
  int status{static_cast<int>(cudaErrorUnknown)};
};

[[nodiscard]] ResourceCache build_resource_cache() noexcept {
  ResourceCache cache{};
  cudaDeviceProp properties{};
  cache.status = validate_target(&properties);
  if (cache.status != static_cast<int>(cudaSuccess)) {
    return cache;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, q3x_sm87_a4w4_gateup_r1_product_finalize_kernel);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, q3x_sm87_a4w4_gateup_r1_product_finalize_kernel,
      static_cast<int>(kThreads), 0U);
  if (status != cudaSuccess) {
    cache.status = static_cast<int>(status);
    return cache;
  }
  cache.resources.registers_per_thread = attributes.numRegs;
  cache.resources.static_shared_bytes = attributes.sharedSizeBytes;
  cache.resources.dynamic_shared_bytes = 0U;
  cache.resources.local_bytes = attributes.localSizeBytes;
  cache.resources.maximum_threads_per_block = attributes.maxThreadsPerBlock;
  cache.resources.active_blocks_per_sm = active_blocks;
  cache.resources.compute_major = properties.major;
  cache.resources.compute_minor = properties.minor;
  cache.resources.multiprocessor_count = properties.multiProcessorCount;
  cache.status =
      cache.resources.registers_per_thread <= 0 ||
              cache.resources.registers_per_thread >
                  static_cast<int>(
                      kSm87A4W4GateUpR1ProductFinalizeMaximumRegisters) ||
              cache.resources.local_bytes != 0U ||
              cache.resources.maximum_threads_per_block <
                  static_cast<int>(kThreads) ||
              cache.resources.active_blocks_per_sm <
                  static_cast<int>(
                      kSm87A4W4GateUpR1ProductFinalizeMinimumBlocksPerSm)
          ? static_cast<int>(cudaErrorLaunchOutOfResources)
          : static_cast<int>(cudaSuccess);
  return cache;
}

[[nodiscard]] const ResourceCache& resource_cache() noexcept {
  static const ResourceCache cache = build_resource_cache();
  return cache;
}

std::atomic<bool> g_resources_ready{false};

}  // namespace

int query_sm87_a4w4_gateup_r1_product_finalize_resources_cuda(
    Sm87A4W4GateUpR1ProductFinalizeResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ResourceCache& cache = resource_cache();
  *resources = cache.resources;
  if (cache.status == static_cast<int>(cudaSuccess)) {
    g_resources_ready.store(true, std::memory_order_release);
  }
  return cache.status;
}

int launch_sm87_a4w4_gateup_r1_product_finalize_cuda(
    const std::uint16_t* const primary_product_bf16,
    const std::size_t primary_product_row_stride_elements,
    const std::size_t primary_product_capacity_elements,
    const std::uint16_t* const secondary_product_bf16,
    const std::size_t secondary_product_row_stride_elements,
    const std::size_t secondary_product_capacity_elements,
    const float* const authenticated_down_inverse_alpha_fp32,
    const std::size_t down_inverse_alpha_capacity_elements,
    const float* const r1_product_tile_maxima_fp32,
    const std::size_t r1_product_tile_maxima_capacity_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    const float clip_ratio,
    std::uint8_t* const packed_down_a,
    const std::size_t packed_down_a_capacity_bytes,
    std::uint16_t* const down_a_lane_scales_bf16,
    const std::size_t down_a_scale_capacity_elements,
    void* const cuda_stream) noexcept {
  const auto plan = sm87_a4w4_gateup_r1_product_finalize_plan(
      logical_token_count, launch_token_count);
  std::size_t required_primary_elements = 0U;
  std::size_t required_secondary_elements = 0U;
  std::size_t required_primary_bytes = 0U;
  std::size_t required_secondary_bytes = 0U;
  std::size_t required_inverse_bytes = 0U;
  std::size_t required_partial_bytes = 0U;
  std::size_t required_scale_bytes = 0U;
  if (!plan.valid() || !(clip_ratio > 0.0F && clip_ratio <= 1.0F) ||
      primary_product_row_stride_elements !=
          kSm87A4W4GateUpFactorizedPrimaryStride ||
      secondary_product_row_stride_elements !=
          kSm87A4W4GateUpFactorizedSecondaryStride ||
      !aligned(primary_product_bf16, 16U) ||
      !aligned(secondary_product_bf16, 16U) ||
      !aligned(authenticated_down_inverse_alpha_fp32, 16U) ||
      !aligned(r1_product_tile_maxima_fp32, 16U) ||
      !aligned(packed_down_a, 16U) ||
      !aligned(down_a_lane_scales_bf16, alignof(std::uint16_t)) ||
      !checked_multiply(logical_token_count,
                        primary_product_row_stride_elements,
                        &required_primary_elements) ||
      !checked_multiply(logical_token_count,
                        secondary_product_row_stride_elements,
                        &required_secondary_elements) ||
      !checked_multiply(required_primary_elements, sizeof(std::uint16_t),
                        &required_primary_bytes) ||
      !checked_multiply(required_secondary_elements,
                        sizeof(std::uint16_t),
                        &required_secondary_bytes) ||
      !checked_multiply(plan.input_size, sizeof(float),
                        &required_inverse_bytes) ||
      !checked_multiply(plan.partial_capacity_elements, sizeof(float),
                        &required_partial_bytes) ||
      !checked_multiply(plan.scale_capacity_elements,
                        sizeof(std::uint16_t), &required_scale_bytes) ||
      primary_product_capacity_elements < required_primary_elements ||
      secondary_product_capacity_elements < required_secondary_elements ||
      down_inverse_alpha_capacity_elements < plan.input_size ||
      r1_product_tile_maxima_capacity_elements <
          plan.partial_capacity_elements ||
      packed_down_a_capacity_bytes < plan.packed_capacity_bytes ||
      down_a_scale_capacity_elements < plan.scale_capacity_elements ||
      launch_token_count > std::numeric_limits<unsigned int>::max()) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ByteRange ranges[] = {
      {primary_product_bf16, required_primary_bytes},
      {secondary_product_bf16, required_secondary_bytes},
      {authenticated_down_inverse_alpha_fp32, required_inverse_bytes},
      {r1_product_tile_maxima_fp32, required_partial_bytes},
      {packed_down_a, plan.packed_capacity_bytes},
      {down_a_lane_scales_bf16, required_scale_bytes}};
  if (!ranges_are_pairwise_disjoint(ranges)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  if (stream != nullptr) {
    const cudaError_t capture_query =
        cudaStreamIsCapturing(stream, &capture_status);
    if (capture_query != cudaSuccess) {
      return static_cast<int>(capture_query);
    }
  }
  if (capture_status != cudaStreamCaptureStatusNone &&
      !g_resources_ready.load(std::memory_order_acquire)) {
    return static_cast<int>(cudaErrorNotReady);
  }
  if (!g_resources_ready.load(std::memory_order_acquire)) {
    Sm87A4W4GateUpR1ProductFinalizeResources resources{};
    const int status =
        query_sm87_a4w4_gateup_r1_product_finalize_resources_cuda(
            &resources);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }

  (void)cudaGetLastError();
  q3x_sm87_a4w4_gateup_r1_product_finalize_kernel<<<
      static_cast<unsigned int>(plan.launch_ctas), kThreads, 0U, stream>>>(
      primary_product_bf16,
      static_cast<unsigned int>(primary_product_row_stride_elements),
      secondary_product_bf16,
      static_cast<unsigned int>(secondary_product_row_stride_elements),
      authenticated_down_inverse_alpha_fp32,
      r1_product_tile_maxima_fp32,
      static_cast<unsigned int>(logical_token_count), clip_ratio,
      packed_down_a, down_a_lane_scales_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
