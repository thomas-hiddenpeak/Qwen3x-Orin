#include "q3x/kernels/sm87_macrofeed_v4_full_attention_preprocess.h"

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
#include "sm87_macrofeed_v4_bound_launch_internal.h"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kFullWarpMask = 0xffff'ffffU;

[[nodiscard]] __device__ __forceinline__ float decode_bf16(
    const std::uint16_t bits) noexcept {
  return __uint_as_float(static_cast<unsigned int>(bits) << 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) noexcept {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

// This is the established prompt-wide-128 tree with only its address map
// changed.  Q is published back to its native [Q256,Gate256] head slot and K
// is addressed through the full private [position,8,128] NHD cache.
__global__ __launch_bounds__(
    kSm87MacroFeedV4FullAttentionPreprocessThreads)
void sm87_macrofeed_v4_full_attention_preprocess_prompt_wide_128_kernel(
    std::uint16_t* const q_gate_scratch,
    const std::size_t scratch_row_stride,
    std::uint16_t* const key_cache,
    const std::size_t key_cache_row_stride,
    const std::uint16_t* const q_norm_weight,
    const std::uint16_t* const k_norm_weight,
    const float* const cosines,
    const float* const sines,
    const std::size_t rope_row_stride,
    const std::size_t first_position,
    const std::uint32_t epsilon_fp32_bits) {
  __shared__ float pair_squares
      [kSm87MacroFeedV4FullAttentionPreprocessThreads];
  __shared__ float inverse_rms_shared;

  const std::size_t token = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token_head = static_cast<std::size_t>(blockIdx.y);
  const bool is_query =
      token_head < kSm87MacroFeedV4FullAttentionPreprocessQueryHeads;
  const std::size_t head =
      is_query
          ? token_head
          : token_head -
                kSm87MacroFeedV4FullAttentionPreprocessQueryHeads;
  const std::size_t low_dimension = threadIdx.x;
  const std::size_t high_dimension =
      low_dimension + kSm87MacroFeedV4FullAttentionPreprocessThreads;

  std::uint16_t* const head_output =
      is_query
          ? q_gate_scratch + token * scratch_row_stride +
                head *
                    kSm87MacroFeedV4FullAttentionPreprocessQGateHeadStride
          : key_cache + (first_position + token) * key_cache_row_stride +
                head *
                    kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
  const std::uint16_t* const weight =
      is_query ? q_norm_weight : k_norm_weight;

  const float low_value = decode_bf16(head_output[low_dimension]);
  const float high_value = decode_bf16(head_output[high_dimension]);
  float pair_sum = fmaf(low_value, low_value, 0.0F);
  pair_sum += fmaf(high_value, high_value, 0.0F);
  pair_squares[low_dimension] = pair_sum;
  __syncthreads();

  if (low_dimension < 32U) {
    const unsigned int lane = static_cast<unsigned int>(low_dimension);
    float sum = pair_squares[lane];
    float rhs = pair_squares[lane + 64U];
    sum += rhs;
    float sibling = pair_squares[lane + 32U];
    rhs = pair_squares[lane + 96U];
    sibling += rhs;
    sum += sibling;
#pragma unroll
    for (unsigned int stride = 16U; stride != 0U; stride >>= 1U) {
      rhs = __shfl_down_sync(kFullWarpMask, sum, stride);
      if (lane < stride) {
        sum += rhs;
      }
    }
    if (lane == 0U) {
      inverse_rms_shared = rsqrtf(
          sum /
                  static_cast<float>(
                      kSm87MacroFeedV4FullAttentionPreprocessHeadDimension) +
              __uint_as_float(epsilon_fp32_bits));
    }
  }
  __syncthreads();

  const float low_gamma = decode_bf16(weight[low_dimension]) + 1.0F;
  const float high_gamma = decode_bf16(weight[high_dimension]) + 1.0F;
  head_output[low_dimension] =
      encode_bf16_rne(low_value * inverse_rms_shared * low_gamma);
  head_output[high_dimension] =
      encode_bf16_rne(high_value * inverse_rms_shared * high_gamma);

  __syncthreads();
  if (low_dimension <
      kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf) {
    const std::size_t table_offset =
        (first_position + token) * rope_row_stride + low_dimension;
    const float cosine = cosines[table_offset];
    const float sine = sines[table_offset];
    const float first = decode_bf16(head_output[low_dimension]);
    const float second = decode_bf16(
        head_output[low_dimension +
                    kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf]);
    const float rotated_first = fmaf(first, cosine, -(second * sine));
    const float rotated_second = fmaf(second, cosine, first * sine);
    head_output[low_dimension] = encode_bf16_rne(rotated_first);
    head_output[low_dimension +
                kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf] =
        encode_bf16_rne(rotated_second);
  }
}

// Independent T1 oracle.  It directly retains the predecessor's 256-thread
// shared-memory reduction stages 128..1 and the same BF16/RoPE boundary.
__global__ __launch_bounds__(256)
void sm87_macrofeed_v4_full_attention_preprocess_reference_256_kernel(
    std::uint16_t* const q_gate_scratch,
    const std::size_t scratch_row_stride,
    std::uint16_t* const key_cache,
    const std::size_t key_cache_row_stride,
    const std::uint16_t* const q_norm_weight,
    const std::uint16_t* const k_norm_weight,
    const float* const cosines,
    const float* const sines,
    const std::size_t rope_row_stride,
    const std::size_t first_position,
    const std::uint32_t epsilon_fp32_bits) {
  __shared__ float partial
      [kSm87MacroFeedV4FullAttentionPreprocessHeadDimension];

  const std::size_t token = static_cast<std::size_t>(blockIdx.x);
  const std::size_t token_head = static_cast<std::size_t>(blockIdx.y);
  const bool is_query =
      token_head < kSm87MacroFeedV4FullAttentionPreprocessQueryHeads;
  const std::size_t head =
      is_query
          ? token_head
          : token_head -
                kSm87MacroFeedV4FullAttentionPreprocessQueryHeads;
  const std::size_t dimension = threadIdx.x;
  std::uint16_t* const head_output =
      is_query
          ? q_gate_scratch + token * scratch_row_stride +
                head *
                    kSm87MacroFeedV4FullAttentionPreprocessQGateHeadStride
          : key_cache + (first_position + token) * key_cache_row_stride +
                head *
                    kSm87MacroFeedV4FullAttentionPreprocessHeadDimension;
  const std::uint16_t* const weight =
      is_query ? q_norm_weight : k_norm_weight;
  const float value = decode_bf16(head_output[dimension]);

  partial[dimension] = fmaf(value, value, 0.0F);
  __syncthreads();
  for (unsigned int stride = 128U; stride != 0U; stride >>= 1U) {
    if (dimension < stride) {
      partial[dimension] += partial[dimension + stride];
    }
    __syncthreads();
  }
  const float inverse_rms = rsqrtf(
      partial[0U] /
              static_cast<float>(
                  kSm87MacroFeedV4FullAttentionPreprocessHeadDimension) +
          __uint_as_float(epsilon_fp32_bits));
  const float gamma = decode_bf16(weight[dimension]) + 1.0F;
  head_output[dimension] = encode_bf16_rne(value * inverse_rms * gamma);

  __syncthreads();
  if (dimension < kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf) {
    const std::size_t table_offset =
        (first_position + token) * rope_row_stride + dimension;
    const float cosine = cosines[table_offset];
    const float sine = sines[table_offset];
    const float first = decode_bf16(head_output[dimension]);
    const float second = decode_bf16(
        head_output[dimension +
                    kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf]);
    const float rotated_first = fmaf(first, cosine, -(second * sine));
    const float rotated_second = fmaf(second, cosine, first * sine);
    head_output[dimension] = encode_bf16_rne(rotated_first);
    head_output[dimension +
                kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf] =
        encode_bf16_rne(rotated_second);
  }
}

[[nodiscard]] constexpr bool pointer_aligned(
    const void* const pointer) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) %
                 kSm87MacroFeedV4FullAttentionPreprocessPointerAlignment ==
             0U;
}

template <class Arguments>
[[nodiscard]] bool structural_arguments_valid(
    const Arguments& arguments, const bool production_extent) noexcept {
  if ((production_extent &&
       arguments.token_count !=
           kSm87MacroFeedV4FullAttentionPreprocessTokens) ||
      (!production_extent &&
       (arguments.token_count == 0U ||
        arguments.token_count >
            kSm87MacroFeedV4FullAttentionPreprocessOracleMaximumTokens)) ||
      !sm87_macrofeed_v4_full_attention_preprocess_first_position_supported(
          arguments.first_position) ||
      arguments.first_position >
          kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions -
              arguments.token_count ||
      arguments.scratch_row_stride !=
          kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride ||
      arguments.key_cache_position_capacity !=
          kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions ||
      arguments.key_cache_row_stride !=
          kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride ||
      arguments.rope_position_capacity !=
          kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions ||
      arguments.rope_row_stride !=
          kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf ||
      arguments.epsilon_fp32_bits !=
          kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits ||
      arguments.cuda_stream == nullptr) {
    return false;
  }
  const void* const pointers[] = {
      arguments.q_gate_scratch, arguments.key_cache,
      arguments.q_norm_weight,  arguments.k_norm_weight,
      arguments.cosines,        arguments.sines,
  };
  for (const void* const pointer : pointers) {
    if (!pointer_aligned(pointer)) {
      return false;
    }
  }

  const std::uint64_t scratch_bytes =
      arguments.token_count * arguments.scratch_row_stride *
      sizeof(std::uint16_t);
  const Sm87MacroFeedV4FullAttentionPreprocessByteRange ranges[] = {
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.q_gate_scratch, scratch_bytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.key_cache,
          kSm87MacroFeedV4FullAttentionPreprocessKeyCacheBytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.q_norm_weight,
          kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.k_norm_weight,
          kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.cosines,
          kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.sines,
          kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes),
  };
  for (std::size_t left = 0U; left < 6U; ++left) {
    if (!ranges[left].valid) {
      return false;
    }
    for (std::size_t right = left + 1U; right < 6U; ++right) {
      if (!sm87_macrofeed_v4_full_attention_preprocess_ranges_disjoint(
              ranges[left], ranges[right])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool device_allocation_range_owned(
    const Sm87MacroFeedV4FullAttentionPreprocessByteRange& range,
    const int device_ordinal) noexcept {
  if (!range.valid || range.begin == 0U || range.end <= range.begin ||
      device_ordinal < 0) {
    return false;
  }
  cudaPointerAttributes attributes{};
  const auto* const pointer = reinterpret_cast<const void*>(range.begin);
  if (cudaPointerGetAttributes(&attributes, pointer) != cudaSuccess ||
      attributes.type != cudaMemoryTypeDevice ||
      attributes.device != device_ordinal) {
    return false;
  }
  CUdeviceptr allocation_base = 0U;
  std::size_t allocation_bytes = 0U;
  if (cuMemGetAddressRange(&allocation_base, &allocation_bytes,
                           static_cast<CUdeviceptr>(range.begin)) !=
          CUDA_SUCCESS ||
      allocation_base == 0U || allocation_bytes == 0U) {
    return false;
  }
  const auto allocation_begin =
      static_cast<std::uintptr_t>(allocation_base);
  if (allocation_begin > std::numeric_limits<std::uintptr_t>::max() -
                             allocation_bytes) {
    return false;
  }
  const auto allocation_end = allocation_begin + allocation_bytes;
  return range.begin >= allocation_begin && range.end <= allocation_end;
}

template <class Arguments>
[[nodiscard]] bool argument_device_allocation_ranges_owned(
    const Arguments& arguments, const int device_ordinal) noexcept {
  const std::uint64_t scratch_bytes =
      arguments.token_count * arguments.scratch_row_stride *
      sizeof(std::uint16_t);
  const Sm87MacroFeedV4FullAttentionPreprocessByteRange ranges[] = {
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.q_gate_scratch, scratch_bytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.key_cache,
          kSm87MacroFeedV4FullAttentionPreprocessKeyCacheBytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.q_norm_weight,
          kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.k_norm_weight,
          kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.cosines,
          kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes),
      sm87_macrofeed_v4_full_attention_preprocess_byte_range(
          arguments.sines,
          kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes),
  };
  for (const auto& range : ranges) {
    if (!device_allocation_range_owned(range, device_ordinal)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool kernel_resources_equal(
    const Sm87MacroFeedV4FullAttentionPreprocessKernelResources& left,
    const Sm87MacroFeedV4FullAttentionPreprocessKernelResources& right)
    noexcept {
  return left.registers_per_thread == right.registers_per_thread &&
         left.static_shared_bytes == right.static_shared_bytes &&
         left.local_bytes == right.local_bytes &&
         left.maximum_threads_per_block == right.maximum_threads_per_block &&
         left.active_blocks_per_sm == right.active_blocks_per_sm &&
         left.threads_per_block == right.threads_per_block &&
         left.grid_x == right.grid_x && left.grid_y == right.grid_y &&
         left.physical_grid_ctas == right.physical_grid_ctas;
}

[[nodiscard]] constexpr bool admission_resources_equal(
    const Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        left,
    const Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        right) noexcept {
  return left.identity == right.identity &&
         left.device_ordinal == right.device_ordinal &&
         left.compute_major == right.compute_major &&
         left.compute_minor == right.compute_minor &&
         left.sm_count == right.sm_count &&
         left.binary_version == right.binary_version &&
         kernel_resources_equal(left.kernel, right.kernel) &&
         left.kernel_compiled == right.kernel_compiled &&
         left.exact_geometry == right.exact_geometry &&
         left.static_resource_gate_passed ==
             right.static_resource_gate_passed &&
         left.numerical_contract_qualified ==
             right.numerical_contract_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible &&
         left.startup_package_unbound == right.startup_package_unbound &&
         left.execution_capability == right.execution_capability &&
         left.caller_snapshot_grants_production_authority ==
             right.caller_snapshot_grants_production_authority;
}

[[nodiscard]] int launch_oracle(
    const Sm87MacroFeedV4FullAttentionPreprocessOracleArguments& arguments,
    const bool reference_256) noexcept {
  if (!structural_arguments_valid(arguments, false)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const dim3 grid(static_cast<unsigned int>(arguments.token_count),
                  static_cast<unsigned int>(
                      kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads));
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  (void)cudaGetLastError();
  if (reference_256) {
    sm87_macrofeed_v4_full_attention_preprocess_reference_256_kernel
        <<<grid, 256U, 0U, stream>>>(
            arguments.q_gate_scratch, arguments.scratch_row_stride,
            arguments.key_cache, arguments.key_cache_row_stride,
            arguments.q_norm_weight, arguments.k_norm_weight,
            arguments.cosines, arguments.sines, arguments.rope_row_stride,
            arguments.first_position, arguments.epsilon_fp32_bits);
  } else {
    sm87_macrofeed_v4_full_attention_preprocess_prompt_wide_128_kernel
        <<<grid, kSm87MacroFeedV4FullAttentionPreprocessThreads, 0U,
           stream>>>(arguments.q_gate_scratch,
                     arguments.scratch_row_stride, arguments.key_cache,
                     arguments.key_cache_row_stride,
                     arguments.q_norm_weight, arguments.k_norm_weight,
                     arguments.cosines, arguments.sines,
                     arguments.rope_row_stride, arguments.first_position,
                     arguments.epsilon_fp32_bits);
  }
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace

bool sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
    const Sm87MacroFeedV4FullAttentionPreprocessArguments& arguments)
    noexcept {
  return structural_arguments_valid(arguments, true);
}

int query_sm87_macrofeed_v4_full_attention_preprocess_admission_resources_cuda(
    Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot* const
        resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
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
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(
              kSm87MacroFeedV4FullAttentionPreprocessSmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes,
      sm87_macrofeed_v4_full_attention_preprocess_prompt_wide_128_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks,
      sm87_macrofeed_v4_full_attention_preprocess_prompt_wide_128_kernel,
      static_cast<int>(
          kSm87MacroFeedV4FullAttentionPreprocessThreads),
      0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->identity =
      kSm87MacroFeedV4FullAttentionPreprocessIdentity;
  resources->device_ordinal = device;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->sm_count = properties.multiProcessorCount;
  resources->binary_version = attributes.binaryVersion;
  resources->kernel.registers_per_thread = attributes.numRegs;
  resources->kernel.static_shared_bytes = attributes.sharedSizeBytes;
  resources->kernel.local_bytes = attributes.localSizeBytes;
  resources->kernel.maximum_threads_per_block =
      attributes.maxThreadsPerBlock;
  resources->kernel.active_blocks_per_sm = active_blocks;
  resources->kernel.threads_per_block = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessThreads);
  resources->kernel.grid_x = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessTokens);
  resources->kernel.grid_y = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads);
  resources->kernel.physical_grid_ctas = static_cast<std::int32_t>(
      kSm87MacroFeedV4FullAttentionPreprocessPhysicalCtas);
  resources->kernel_compiled = true;
  resources->exact_geometry = true;
  resources->static_resource_gate_passed = true;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  resources->startup_package_unbound = true;
  resources->execution_capability = false;
  resources->caller_snapshot_grants_production_authority = false;
  if (!sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
          *resources)) {
    resources->static_resource_gate_passed = false;
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v4_full_attention_preprocess_admission_cuda(
    const Sm87MacroFeedV4FullAttentionPreprocessArguments& arguments,
    const Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        resources,
    Sm87MacroFeedV4FullAttentionPreprocessAdmissionLaunchReceipt* const
        receipt) noexcept {
  if (receipt == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *receipt = {};
  const auto plan = sm87_macrofeed_v4_full_attention_preprocess_plan(
      arguments.first_position, arguments.token_count);
  if (!plan.valid() ||
      !sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(
          arguments) ||
      !sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
          resources)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot observed{};
  const int query_status =
      query_sm87_macrofeed_v4_full_attention_preprocess_admission_resources_cuda(
          &observed);
  if (query_status != static_cast<int>(cudaSuccess)) {
    return query_status;
  }
  if (!admission_resources_equal(resources, observed) ||
      !argument_device_allocation_ranges_owned(arguments,
                                               observed.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const dim3 grid(static_cast<unsigned int>(arguments.token_count),
                  static_cast<unsigned int>(
                      kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads));
  const auto stream = reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  (void)cudaGetLastError();
  sm87_macrofeed_v4_full_attention_preprocess_prompt_wide_128_kernel
      <<<grid, kSm87MacroFeedV4FullAttentionPreprocessThreads, 0U,
         stream>>>(arguments.q_gate_scratch, arguments.scratch_row_stride,
                   arguments.key_cache, arguments.key_cache_row_stride,
                   arguments.q_norm_weight, arguments.k_norm_weight,
                   arguments.cosines, arguments.sines,
                   arguments.rope_row_stride, arguments.first_position,
                   arguments.epsilon_fp32_bits);
  const cudaError_t launch_status = cudaPeekAtLastError();
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }

  receipt->identity = plan.identity;
  receipt->device_ordinal = observed.device_ordinal;
  receipt->first_position = plan.first_position;
  receipt->token_count = plan.token_count;
  receipt->scratch_row_stride = plan.scratch_row_stride;
  receipt->physical_kernel_launches = 1U;
  receipt->centered_qk_rmsnorm = true;
  receipt->exact_prompt_wide_reduction_tree = true;
  receipt->bf16_boundary_before_rope = true;
  receipt->partial_d64_neox_rope = true;
  receipt->q_in_place = true;
  receipt->k_in_place = true;
  receipt->gate_bitwise_preserved = true;
  receipt->scratch_gap_bitwise_preserved = true;
  receipt->private_nhd_key_cache = true;
  receipt->value_cache_unaddressable = true;
  receipt->current_device_revalidated = true;
  receipt->caller_snapshot_exact_observed_match = true;
  receipt->device_allocation_ranges_owned = true;
  receipt->caller_stream_non_null = true;
  receipt->stream_owner_verified = false;
  receipt->launch_enqueued = true;
  receipt->completion_observed = false;
  receipt->numerical_contract_qualified = false;
  receipt->production_dispatch_eligible = false;
  receipt->startup_package_unbound = true;
  receipt->execution_capability = false;
  receipt->caller_snapshot_grants_production_authority = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v4_full_attention_preprocess_candidate_128_oracle_cuda(
    const Sm87MacroFeedV4FullAttentionPreprocessOracleArguments& arguments)
    noexcept {
  return launch_oracle(arguments, false);
}

int launch_sm87_macrofeed_v4_full_attention_preprocess_reference_256_oracle_cuda(
    const Sm87MacroFeedV4FullAttentionPreprocessOracleArguments& arguments)
    noexcept {
  return launch_oracle(arguments, true);
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
namespace sm87_macrofeed_v4_bound_launch_detail {

int enqueue_full_attention_preprocess_c8000_prevalidated(
    const Sm87MacroFeedV4LockedSubmitToken& token,
    const Sm87MacroFeedV4FullAttentionPreprocessC8000Arguments& arguments,
    const Sm87MacroFeedV4FullAttentionPreprocessAdmissionResourceSnapshot&
        resources,
    std::size_t* const submitted_launches) noexcept {
  if (submitted_launches == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *submitted_launches = 0U;
  const Sm87MacroFeedV4FullAttentionPreprocessArguments fixed{
      arguments.q_gate_scratch,
      kSm87MacroFeedV4FullAttentionPreprocessTokens,
      kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride,
      arguments.key_cache_origin,
      kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
      kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride,
      arguments.q_norm_weight,
      arguments.k_norm_weight,
      arguments.cosines,
      arguments.sines,
      kSm87MacroFeedV4FullAttentionPreprocessMaximumPositions,
      kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf,
      arguments.first_position,
      kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits,
      token.cuda_stream_};
  const auto plan = sm87_macrofeed_v4_full_attention_preprocess_plan(
      arguments.first_position,
      kSm87MacroFeedV4FullAttentionPreprocessTokens);
  if (!plan.valid() ||
      !sm87_macrofeed_v4_full_attention_preprocess_arguments_valid(fixed) ||
      !resources.static_resource_gate_passed ||
      !sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
          resources) ||
      resources.identity !=
          kSm87MacroFeedV4FullAttentionPreprocessIdentity) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const cudaError_t prior_status = cudaPeekAtLastError();
  if (prior_status != cudaSuccess) {
    return static_cast<int>(prior_status);
  }

  const dim3 grid(
      static_cast<unsigned int>(
          kSm87MacroFeedV4FullAttentionPreprocessTokens),
      static_cast<unsigned int>(
          kSm87MacroFeedV4FullAttentionPreprocessCombinedHeads));
  sm87_macrofeed_v4_full_attention_preprocess_prompt_wide_128_kernel
      <<<grid, kSm87MacroFeedV4FullAttentionPreprocessThreads, 0U,
         reinterpret_cast<cudaStream_t>(token.cuda_stream_)>>>(
          arguments.q_gate_scratch,
          kSm87MacroFeedV4FullAttentionPreprocessScratchRowStride,
          arguments.key_cache_origin,
          kSm87MacroFeedV4FullAttentionPreprocessKeyRowStride,
          arguments.q_norm_weight, arguments.k_norm_weight,
          arguments.cosines, arguments.sines,
          kSm87MacroFeedV4FullAttentionPreprocessRotaryHalf,
          arguments.first_position,
          kSm87MacroFeedV4FullAttentionPreprocessEpsilonFp32Bits);
  const cudaError_t launch_status = cudaPeekAtLastError();
  if (launch_status != cudaSuccess) {
    return static_cast<int>(launch_status);
  }
  *submitted_launches = 1U;
  return static_cast<int>(cudaSuccess);
}

}  // namespace sm87_macrofeed_v4_bound_launch_detail
#endif

}  // namespace q3x::kernels
