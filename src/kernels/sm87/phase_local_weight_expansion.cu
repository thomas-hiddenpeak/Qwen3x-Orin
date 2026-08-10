#include "q3x/kernels/sm87_phase_local_weight_expansion.h"

#include "q3x/quantization/nvfp4.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87PhaseLocalWeightThreads);
constexpr unsigned int kWarpSize = 32U;

[[nodiscard]] bool pointer_is_aligned(const void* const pointer,
                                      const std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0U &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return pointer == nullptr ||
         bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

[[nodiscard]] bool byte_ranges_overlap(const void* const first,
                                       const std::size_t first_bytes,
                                       const void* const second,
                                       const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return true;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

[[nodiscard]] int exact_sm87_device_status() noexcept {
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
  return properties.major == 8 && properties.minor == 7
             ? static_cast<int>(cudaSuccess)
             : static_cast<int>(cudaErrorNotSupported);
}

[[nodiscard]] std::uint16_t float_to_bf16_rne_bits(
    const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t rounding_bias =
      0x0000'7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

// These device decoders are the audited bit mappings already used by the
// production FP8 supermatrix and NVFP4 cuBLASLt-reference kernels.  The host
// witnesses below call the canonical quantization helpers independently, so
// every raw encoding remains cross-checked against the shared semantics.
[[nodiscard]] __device__ __forceinline__ std::uint16_t
decode_e4m3fn_to_bf16_bits(const std::uint8_t code) {
  const std::uint16_t sign =
      static_cast<std::uint16_t>(code & 0x80U) << 8U;
  const std::uint16_t magnitude =
      static_cast<std::uint16_t>(code & 0x7fU);
  if (magnitude == 0x7fU) {
    return static_cast<std::uint16_t>(sign | 0x7fc0U);
  }
  const std::uint16_t exponent = magnitude >> 3U;
  const std::uint16_t mantissa = magnitude & 0x07U;
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return sign;
    }
    const std::uint16_t leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const std::uint16_t bf16_exponent = 118U + leading;
    const std::uint16_t bf16_mantissa =
        (mantissa - (1U << leading)) << (7U - leading);
    return static_cast<std::uint16_t>(
        sign | (bf16_exponent << 7U) | bf16_mantissa);
  }
  return static_cast<std::uint16_t>(
      sign | ((120U + exponent) << 7U) | (mantissa << 4U));
}

[[nodiscard]] __device__ __forceinline__ float decode_e4m3fn(
    const std::uint8_t bits) {
  const unsigned int sign =
      static_cast<unsigned int>(bits & 0x80U) << 24U;
  const unsigned int magnitude = static_cast<unsigned int>(bits & 0x7fU);
  const unsigned int exponent = magnitude >> 3U;
  const unsigned int mantissa = magnitude & 0x07U;
  if (magnitude == 0x7fU) {
    return __uint_as_float(sign | 0x7fc0'0000U);
  }
  if (exponent == 0U) {
    if (mantissa == 0U) {
      return __uint_as_float(sign);
    }
    const unsigned int leading =
        mantissa >= 4U ? 2U : (mantissa >= 2U ? 1U : 0U);
    const unsigned int fp32_exponent = 118U + leading;
    const unsigned int fp32_mantissa =
        (mantissa - (1U << leading)) << (23U - leading);
    return __uint_as_float(sign | (fp32_exponent << 23U) | fp32_mantissa);
  }
  return __uint_as_float(sign | ((120U + exponent) << 23U) |
                         (mantissa << 20U));
}

[[nodiscard]] __device__ __forceinline__ float decode_e2m1(
    const std::uint8_t nibble) {
  const unsigned int sign =
      static_cast<unsigned int>(nibble & 0x08U) << 28U;
  const unsigned int magnitude = static_cast<unsigned int>(nibble & 0x07U);
  const unsigned int nonzero_mask =
      0U - static_cast<unsigned int>(magnitude != 0U);
  const unsigned int mantissa =
      ((magnitude & 1U) & static_cast<unsigned int>(magnitude > 1U)) << 22U;
  const unsigned int finite_bits =
      ((126U + (magnitude >> 1U)) << 23U) | mantissa;
  return __uint_as_float(sign | (finite_bits & nonzero_mask));
}

template <unsigned int kInputFeatures, unsigned int kPassBase,
          unsigned int kWindowPasses>
__device__ __forceinline__ void expand_nvfp4_window(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    std::uint32_t* const output_pairs,
    const std::size_t packed_base,
    const std::size_t scale_base,
    const unsigned int lane,
    const unsigned int warp) {
  constexpr unsigned int kPackedPerRow = kInputFeatures / 2U;
  constexpr unsigned int kPasses = kPackedPerRow / kThreads;
  static_assert(kInputFeatures % 16U == 0U);
  static_assert(kPackedPerRow % kThreads == 0U);
  static_assert(kWindowPasses == 2U || kWindowPasses == 8U ||
                kWindowPasses == 10U);
  static_assert(kPassBase + kWindowPasses <= kPasses);

  std::uint8_t packed_values[kWindowPasses];
  std::uint32_t scale_words[kWindowPasses];
#pragma unroll
  for (unsigned int slot = 0U; slot < kWindowPasses; ++slot) {
    const unsigned int pass = kPassBase + slot;
    const unsigned int packed_column = threadIdx.x + pass * kThreads;
    packed_values[slot] = packed_weights[packed_base + packed_column];
    scale_words[slot] = 0U;
    if (lane == 0U) {
      const std::size_t word_index =
          scale_base + pass * (kThreads / 8U) + warp * 4U;
      scale_words[slot] = *reinterpret_cast<const std::uint32_t*>(
          block_scales + word_index);
    }
  }

#pragma unroll
  for (unsigned int slot = 0U; slot < kWindowPasses; ++slot) {
    const unsigned int pass = kPassBase + slot;
    const unsigned int packed_column = threadIdx.x + pass * kThreads;
    const std::uint32_t scale_word =
        __shfl_sync(0xffff'ffffU, scale_words[slot], 0);
    const std::uint8_t scale_code = static_cast<std::uint8_t>(
        scale_word >> ((lane >> 3U) * 8U));
    const float block_scale = decode_e4m3fn(scale_code);
    const std::uint8_t packed = packed_values[slot];
    const __nv_bfloat16 low = __float2bfloat16_rn(
        decode_e2m1(packed & 0x0fU) * block_scale);
    const __nv_bfloat16 high = __float2bfloat16_rn(
        decode_e2m1(packed >> 4U) * block_scale);
    output_pairs[packed_base + packed_column] =
        static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
        (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
  }
}

template <unsigned int kInputFeatures>
__global__ __launch_bounds__(kThreads, 4)
void expand_nvfp4_canonical_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    std::uint16_t* const bf16_scratch) {
  constexpr unsigned int kPackedPerRow = kInputFeatures / 2U;
  constexpr unsigned int kScalesPerRow = kInputFeatures / 16U;
  static_assert(kInputFeatures == 5'120U || kInputFeatures == 17'408U);
  const std::size_t packed_base =
      static_cast<std::size_t>(blockIdx.x) * kPackedPerRow;
  const std::size_t scale_base =
      static_cast<std::size_t>(blockIdx.x) * kScalesPerRow;
  const unsigned int lane = threadIdx.x & (kWarpSize - 1U);
  const unsigned int warp = threadIdx.x / kWarpSize;
  auto* const output_pairs =
      reinterpret_cast<std::uint32_t*>(bf16_scratch);

  if constexpr (kInputFeatures == 5'120U) {
    expand_nvfp4_window<kInputFeatures, 0U, 10U>(
        packed_weights, block_scales, output_pairs, packed_base, scale_base,
        lane, warp);
  } else {
    expand_nvfp4_window<kInputFeatures, 0U, 8U>(
        packed_weights, block_scales, output_pairs, packed_base, scale_base,
        lane, warp);
    expand_nvfp4_window<kInputFeatures, 8U, 8U>(
        packed_weights, block_scales, output_pairs, packed_base, scale_base,
        lane, warp);
    expand_nvfp4_window<kInputFeatures, 16U, 8U>(
        packed_weights, block_scales, output_pairs, packed_base, scale_base,
        lane, warp);
    expand_nvfp4_window<kInputFeatures, 24U, 8U>(
        packed_weights, block_scales, output_pairs, packed_base, scale_base,
        lane, warp);
    expand_nvfp4_window<kInputFeatures, 32U, 2U>(
        packed_weights, block_scales, output_pairs, packed_base, scale_base,
        lane, warp);
  }
}

template <unsigned int kInputFeatures>
__global__ __launch_bounds__(kThreads, 4)
void expand_fp8_canonical_kernel(
    const std::uint8_t* const weights,
    std::uint16_t* const bf16_scratch) {
  constexpr unsigned int kPairsPerRow = kInputFeatures / 2U;
  constexpr unsigned int kPasses = kPairsPerRow / kThreads;
  static_assert(kInputFeatures == 5'120U || kInputFeatures == 6'144U);
  static_assert(kPairsPerRow % kThreads == 0U);
  const std::size_t pair_base =
      static_cast<std::size_t>(blockIdx.x) * kPairsPerRow;
  const auto* const input_pairs =
      reinterpret_cast<const std::uint16_t*>(weights);
  auto* const output_pairs =
      reinterpret_cast<std::uint32_t*>(bf16_scratch);

  std::uint16_t packed_values[kPasses];
#pragma unroll
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    packed_values[pass] = input_pairs[pair_base + threadIdx.x +
                                      pass * kThreads];
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    const std::uint16_t packed = packed_values[pass];
    const std::uint16_t low = decode_e4m3fn_to_bf16_bits(
        static_cast<std::uint8_t>(packed & 0x00ffU));
    const std::uint16_t high = decode_e4m3fn_to_bf16_bits(
        static_cast<std::uint8_t>(packed >> 8U));
    output_pairs[pair_base + threadIdx.x + pass * kThreads] =
        static_cast<std::uint32_t>(low) |
        (static_cast<std::uint32_t>(high) << 16U);
  }
}

template <unsigned int kInputFeatures>
[[nodiscard]] cudaError_t inspect_nvfp4_expansion_kernel(
    cudaFuncAttributes* const attributes,
    int* const active_blocks) noexcept {
  cudaError_t status = cudaFuncGetAttributes(
      attributes, expand_nvfp4_canonical_kernel<kInputFeatures>);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_blocks, expand_nvfp4_canonical_kernel<kInputFeatures>,
      static_cast<int>(kThreads), 0U);
}

template <unsigned int kInputFeatures>
[[nodiscard]] cudaError_t inspect_fp8_expansion_kernel(
    cudaFuncAttributes* const attributes,
    int* const active_blocks) noexcept {
  cudaError_t status = cudaFuncGetAttributes(
      attributes, expand_fp8_canonical_kernel<kInputFeatures>);
  if (status != cudaSuccess) {
    return status;
  }
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_blocks, expand_fp8_canonical_kernel<kInputFeatures>,
      static_cast<int>(kThreads), 0U);
}

[[nodiscard]] cudaError_t inspect_expansion_role(
    const Sm87PhaseLocalWeightRole role,
    cudaFuncAttributes* const attributes,
    int* const active_blocks) noexcept {
  switch (role) {
    case Sm87PhaseLocalWeightRole::kNvFp4Gate:
    case Sm87PhaseLocalWeightRole::kNvFp4Up:
      return inspect_nvfp4_expansion_kernel<5'120U>(attributes,
                                                    active_blocks);
    case Sm87PhaseLocalWeightRole::kNvFp4Down:
      return inspect_nvfp4_expansion_kernel<17'408U>(attributes,
                                                     active_blocks);
    case Sm87PhaseLocalWeightRole::kFp8LinearQkv:
    case Sm87PhaseLocalWeightRole::kFp8LinearZ:
    case Sm87PhaseLocalWeightRole::kFp8FullQuery:
    case Sm87PhaseLocalWeightRole::kFp8FullKey:
    case Sm87PhaseLocalWeightRole::kFp8FullValue:
      return inspect_fp8_expansion_kernel<5'120U>(attributes,
                                                  active_blocks);
    case Sm87PhaseLocalWeightRole::kFp8LinearOutput:
    case Sm87PhaseLocalWeightRole::kFp8FullOutput:
      return inspect_fp8_expansion_kernel<6'144U>(attributes,
                                                  active_blocks);
  }
  return cudaErrorInvalidValue;
}

[[nodiscard]] int validate_plan_and_device(
    const Sm87PhaseLocalWeightExpansionPlan& plan,
    const Sm87PhaseLocalWeightFormat required_format) noexcept {
  if (!plan.valid() || plan.format != required_format) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return exact_sm87_device_status();
}

}  // namespace

std::uint16_t sm87_phase_local_nvfp4_expanded_bf16_reference(
    const std::uint8_t packed_weight,
    const bool high_nibble,
    const std::uint8_t block_scale) noexcept {
  const float value = q3x::quantization::decode_e2m1(
                          q3x::quantization::unpack_e2m1_nibble(
                              packed_weight, high_nibble)) *
                      q3x::quantization::decode_e4m3fn(block_scale);
  return float_to_bf16_rne_bits(value);
}

std::uint16_t sm87_phase_local_fp8_expanded_bf16_reference(
    const std::uint8_t weight) noexcept {
  return float_to_bf16_rne_bits(
      q3x::quantization::decode_e4m3fn(weight));
}

int query_sm87_phase_local_weight_expansion_resources_cuda(
    const Sm87PhaseLocalWeightRole role,
    Sm87PhaseLocalWeightExpansionResources* const resources) noexcept {
  if (resources == nullptr ||
      !sm87_phase_local_weight_shape(role).supported) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  resources->role = role;
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
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  if (properties.major != 8 || properties.minor != 7) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  status = inspect_expansion_role(role, &attributes, &active_blocks);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = 0U;
  resources->local_bytes = attributes.localSizeBytes;
  resources->active_blocks_per_sm = active_blocks;
  resources->admitted =
      attributes.binaryVersion == 87 &&
      attributes.numRegs == sm87_phase_local_weight_expected_registers(role) &&
      attributes.sharedSizeBytes == 0U && attributes.localSizeBytes == 0U &&
      active_blocks == kSm87PhaseLocalWeightExpectedActiveBlocksPerSm &&
      properties.maxThreadsPerMultiProcessor == 1'536;
  return resources->valid() ? static_cast<int>(cudaSuccess)
                            : static_cast<int>(cudaErrorNotSupported);
}

int launch_sm87_phase_local_nvfp4_weight_expansion_test_cuda(
    const Sm87PhaseLocalWeightExpansionPlan& plan,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    std::uint16_t* const bf16_scratch,
    const std::size_t scratch_bytes,
    void* const cuda_stream) noexcept {
  const int admission = validate_plan_and_device(
      plan, Sm87PhaseLocalWeightFormat::kNvFp4ModelOptBlock16);
  if (admission != static_cast<int>(cudaSuccess)) {
    return admission;
  }
  if (!pointer_is_aligned(packed_weights, 16U) ||
      !pointer_is_aligned(block_scales, 4U) ||
      !pointer_is_aligned(bf16_scratch, 4U) ||
      scratch_bytes < plan.scratch_bytes ||
      byte_ranges_overlap(packed_weights, plan.canonical_weight_bytes,
                          block_scales, plan.block_scale_bytes) ||
      byte_ranges_overlap(packed_weights, plan.canonical_weight_bytes,
                          bf16_scratch, plan.scratch_bytes) ||
      byte_ranges_overlap(block_scales, plan.block_scale_bytes,
                          bf16_scratch, plan.scratch_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  (void)cudaGetLastError();
  const dim3 grid(static_cast<unsigned int>(plan.blocks));
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  switch (plan.role) {
    case Sm87PhaseLocalWeightRole::kNvFp4Gate:
    case Sm87PhaseLocalWeightRole::kNvFp4Up:
      expand_nvfp4_canonical_kernel<5'120U>
          <<<grid, kThreads, 0U, stream>>>(packed_weights, block_scales,
                                          bf16_scratch);
      break;
    case Sm87PhaseLocalWeightRole::kNvFp4Down:
      expand_nvfp4_canonical_kernel<17'408U>
          <<<grid, kThreads, 0U, stream>>>(packed_weights, block_scales,
                                          bf16_scratch);
      break;
    default:
      return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaGetLastError());
}

int launch_sm87_phase_local_fp8_weight_expansion_test_cuda(
    const Sm87PhaseLocalWeightExpansionPlan& plan,
    const std::uint8_t* const weights,
    std::uint16_t* const bf16_scratch,
    const std::size_t scratch_bytes,
    void* const cuda_stream) noexcept {
  const int admission = validate_plan_and_device(
      plan, Sm87PhaseLocalWeightFormat::kFp8E4M3Fn);
  if (admission != static_cast<int>(cudaSuccess)) {
    return admission;
  }
  if (!pointer_is_aligned(weights, 16U) ||
      !pointer_is_aligned(bf16_scratch, 4U) ||
      scratch_bytes < plan.scratch_bytes ||
      byte_ranges_overlap(weights, plan.canonical_weight_bytes,
                          bf16_scratch, plan.scratch_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  (void)cudaGetLastError();
  const dim3 grid(static_cast<unsigned int>(plan.blocks));
  const cudaStream_t stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  switch (plan.role) {
    case Sm87PhaseLocalWeightRole::kFp8LinearQkv:
    case Sm87PhaseLocalWeightRole::kFp8LinearZ:
    case Sm87PhaseLocalWeightRole::kFp8FullQuery:
    case Sm87PhaseLocalWeightRole::kFp8FullKey:
    case Sm87PhaseLocalWeightRole::kFp8FullValue:
      expand_fp8_canonical_kernel<5'120U>
          <<<grid, kThreads, 0U, stream>>>(weights, bf16_scratch);
      break;
    case Sm87PhaseLocalWeightRole::kFp8LinearOutput:
    case Sm87PhaseLocalWeightRole::kFp8FullOutput:
      expand_fp8_canonical_kernel<6'144U>
          <<<grid, kThreads, 0U, stream>>>(weights, bf16_scratch);
      break;
    default:
      return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::kernels
