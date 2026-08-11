#include "q3x/kernels/sm87_nvfp4_marlin_p40_parity.h"

#include "third_party/vllm_marlin/marlin_template.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

#if defined(Q3X_ENABLE_NVFP4_MARLIN_P40_PARITY_ADMISSION)
inline constexpr bool kP40ParityAdmitted = true;
#else
inline constexpr bool kP40ParityAdmitted = false;
#endif

inline constexpr int kThreads = 256;
inline constexpr int kThreadMBlocks = 4;
inline constexpr int kThreadNBlocks = 16;
inline constexpr int kThreadKBlocks = 4;
inline constexpr int kStages = 4;
inline constexpr int kGroupBlocks = 1;
inline constexpr int kGroupSize = 16;

static_assert(kThreadMBlocks * 16 ==
              static_cast<int>(kSm87NvFp4MarlinThreadM));
static_assert(kThreadNBlocks * 16 ==
              static_cast<int>(kSm87NvFp4MarlinThreadN));
static_assert(kThreadKBlocks * 16 ==
              static_cast<int>(kSm87NvFp4MarlinThreadK));
static_assert(kStages ==
              static_cast<int>(kSm87NvFp4MarlinPipelineStages));
static_assert(kSm87NvFp4MarlinP40ParityTokens <=
              static_cast<std::size_t>(std::numeric_limits<int>::max()));
static_assert(kSm87NvFp4MarlinP40ParityTokens *
                      kSm87NvFp4MarlinGateUpOutput <=
                  static_cast<std::size_t>(
                      std::numeric_limits<std::ptrdiff_t>::max()) /
                      sizeof(std::uint16_t));

using LegacyKernel = decltype(
    marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                   vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                   kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                   kStages, kGroupBlocks, false, false,
                   marlin::MarlinTileRasterPolicy::kLegacyStripe, false>);

struct ByteRange {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] ByteRange byte_range(const void* const pointer,
                                   const std::size_t bytes) noexcept {
  if (pointer == nullptr || bytes == 0U) {
    return {};
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (bytes > std::numeric_limits<std::uintptr_t>::max() - begin) {
    return {};
  }
  return {begin, begin + bytes, true};
}

[[nodiscard]] bool overlaps(const ByteRange& first,
                            const ByteRange& second) noexcept {
  return first.valid && second.valid && first.begin < second.end &&
         second.begin < first.end;
}

[[nodiscard]] int validate_fixed_device() noexcept {
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
          static_cast<int>(kSm87NvFp4MarlinP40ParityPersistentCtas) ||
      properties.sharedMemPerBlockOptin <
          kSm87NvFp4MarlinDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaSuccess);
}

template <typename Kernel>
[[nodiscard]] int configure_kernel(const Kernel kernel) noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4MarlinDynamicSharedBytes)));
}

template <typename Kernel>
[[nodiscard]] int read_resources(
    const Kernel kernel,
    Sm87NvFp4MarlinP40ParityKernelResources* const resources) noexcept {
  const int configured = configure_kernel(kernel);
  if (configured != static_cast<int>(cudaSuccess)) {
    return configured;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kernel, kThreads, kSm87NvFp4MarlinDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87NvFp4MarlinDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active;
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool valid_projection_ranges(
    const Sm87NvFp4MarlinP40ParityPlan& plan,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::uint16_t* const output, float* const reduction_workspace,
    const std::size_t reduction_workspace_bytes,
    std::int32_t* const locks, const std::size_t lock_bytes) noexcept {
  if (!plan.valid() || !aligned(input, 16U) ||
      !aligned(marlin_weight, 16U) || !aligned(marlin_scales, 16U) ||
      !aligned(marlin_global_scale, alignof(float)) ||
      !aligned(output, 16U) ||
      !aligned(reduction_workspace, 16U) ||
      !aligned(locks, alignof(std::int32_t)) ||
      reduction_workspace_bytes != plan.required_reduction_workspace_bytes ||
      lock_bytes != plan.required_lock_bytes) {
    return false;
  }
  const ByteRange input_range = byte_range(
      input, plan.token_count * plan.input_features * sizeof(std::uint16_t));
  const ByteRange weight_range = byte_range(
      marlin_weight,
      sm87_nvfp4_marlin_weight_bytes(plan.weight_output_features,
                                     plan.input_features));
  const ByteRange scale_range = byte_range(
      marlin_scales,
      sm87_nvfp4_marlin_scale_bytes(plan.weight_output_features,
                                    plan.input_features));
  const ByteRange global_scale_range =
      byte_range(marlin_global_scale, sizeof(float));
  const ByteRange output_range = byte_range(
      output,
      plan.token_count * plan.published_output_features *
          sizeof(std::uint16_t));
  const ByteRange reduction_range =
      byte_range(reduction_workspace, reduction_workspace_bytes);
  const ByteRange lock_range = byte_range(locks, lock_bytes);
  return input_range.valid && weight_range.valid && scale_range.valid &&
         global_scale_range.valid && output_range.valid &&
         reduction_range.valid && lock_range.valid &&
         !overlaps(output_range, input_range) &&
         !overlaps(output_range, weight_range) &&
         !overlaps(output_range, scale_range) &&
         !overlaps(output_range, global_scale_range) &&
         !overlaps(reduction_range, input_range) &&
         !overlaps(reduction_range, weight_range) &&
         !overlaps(reduction_range, scale_range) &&
         !overlaps(reduction_range, global_scale_range) &&
         !overlaps(reduction_range, output_range) &&
         !overlaps(lock_range, input_range) &&
         !overlaps(lock_range, weight_range) &&
         !overlaps(lock_range, scale_range) &&
         !overlaps(lock_range, global_scale_range) &&
         !overlaps(lock_range, output_range) &&
         !overlaps(lock_range, reduction_range);
}

[[nodiscard]] int launch_segment(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count, const int output_features,
    const int input_features, std::uint16_t* const output,
    float* const reduction_workspace, std::int32_t* const locks,
    void* const cuda_stream) noexcept {
  const auto kernel =
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                     vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                     kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                     kStages, kGroupBlocks, false, false,
                     marlin::MarlinTileRasterPolicy::kLegacyStripe, false>;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  kernel<<<static_cast<unsigned int>(
               kSm87NvFp4MarlinP40ParityPersistentCtas),
           kThreads, kSm87NvFp4MarlinDynamicSharedBytes, stream>>>(
      reinterpret_cast<const int4*>(input),
      reinterpret_cast<const int4*>(marlin_weight),
      reinterpret_cast<int4*>(output),
      reinterpret_cast<int4*>(reduction_workspace), nullptr, nullptr,
      reinterpret_cast<const int4*>(marlin_scales), marlin_global_scale,
      nullptr, nullptr, input_features / kGroupSize,
      static_cast<int>(token_count), output_features, input_features,
      input_features, locks, false, false, true,
      static_cast<int>(kSm87NvFp4MarlinDynamicSharedBytes));
  return static_cast<int>(cudaPeekAtLastError());
}

[[nodiscard]] int launch_projection(
    const Sm87NvFp4MarlinP40ParityRole role,
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    std::uint16_t* const output, float* const reduction_workspace,
    const std::size_t reduction_workspace_bytes,
    std::int32_t* const locks, const std::size_t lock_bytes,
    Sm87NvFp4MarlinP40ParityLaunchCounters* const counters,
    void* const cuda_stream) noexcept {
  if (counters == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *counters = {};
  if (!kP40ParityAdmitted) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const Sm87NvFp4MarlinP40ParityPlan plan =
      sm87_nvfp4_marlin_p40_parity_plan(
          role, kSm87NvFp4MarlinP40ParityTokens);
  if (!valid_projection_ranges(plan, input, marlin_weight, marlin_scales,
                               marlin_global_scale, output,
                               reduction_workspace,
                               reduction_workspace_bytes, locks,
                               lock_bytes)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device = validate_fixed_device();
  if (device != static_cast<int>(cudaSuccess)) {
    return device;
  }
  int status = configure_kernel(
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                     vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                     kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                     kStages, kGroupBlocks, false, false,
                     marlin::MarlinTileRasterPolicy::kLegacyStripe, false>);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }

  const int output_features = static_cast<int>(plan.weight_output_features);
  const int input_features = static_cast<int>(plan.input_features);
  for (std::size_t index = 0U;
       index < kSm87NvFp4MarlinP40ParityLegacySegmentCount; ++index) {
    const Sm87NvFp4MarlinP40ParitySegment segment =
        sm87_nvfp4_marlin_p40_parity_segment(plan, index);
    if (!segment.valid() ||
        segment.kind !=
            Sm87NvFp4MarlinP40ParitySegmentKind::
                kLegacyStripeFullKM1024) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    status = launch_segment(
        input + segment.token_offset * plan.input_features, marlin_weight,
        marlin_scales, marlin_global_scale, segment.token_count,
        output_features, input_features,
        output + segment.token_offset * plan.published_output_features,
        reduction_workspace, locks, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    ++counters->legacy_full_k_m1024_launches;
    ++counters->physical_projection_launches;
  }

  const Sm87NvFp4MarlinP40ParitySegment tail =
      sm87_nvfp4_marlin_p40_parity_segment(
          plan, kSm87NvFp4MarlinP40ParityLegacySegmentCount);
  if (!tail.valid() ||
      tail.kind !=
          Sm87NvFp4MarlinP40ParitySegmentKind::kLegacyStripeSplitKM64) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  status = launch_segment(
      input + tail.token_offset * plan.input_features, marlin_weight,
      marlin_scales, marlin_global_scale, tail.token_count, output_features,
      input_features,
      output + tail.token_offset * plan.published_output_features,
      reduction_workspace, locks, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  ++counters->legacy_split_k_m64_launches;
  ++counters->physical_projection_launches;
  counters->tail_split_k_output_tiles = tail.split_k_output_tile_count;
  counters->tail_split_k_partial_slices = tail.split_k_partial_slice_count;
  counters->complete = counters->counts_match(plan);
  return counters->matches(plan) ? static_cast<int>(cudaSuccess)
                                 : static_cast<int>(cudaErrorUnknown);
}

__device__ __forceinline__ float decode_bf16_bits(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<std::uint32_t>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16_bits(
    const float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void p40_parity_silu_kernel(
    const std::uint16_t* const merged_gate_up_row_major,
    const std::size_t elements, std::uint16_t* const output) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < elements;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    const std::size_t token = index / kSm87NvFp4MarlinIntermediate;
    const std::size_t column = index % kSm87NvFp4MarlinIntermediate;
    const std::size_t gate_index =
        token * kSm87NvFp4MarlinP40ParityGateUpRowStrideElements +
        kSm87NvFp4MarlinP40ParityGateColumnOffset + column;
    const std::size_t up_index =
        token * kSm87NvFp4MarlinP40ParityGateUpRowStrideElements +
        kSm87NvFp4MarlinP40ParityUpColumnOffset + column;
    const float gate =
        decode_bf16_bits(merged_gate_up_row_major[gate_index]);
    const float up = decode_bf16_bits(merged_gate_up_row_major[up_index]);
    output[index] =
        encode_bf16_bits(gate / (1.0F + expf(-gate)) * up);
  }
}

}  // namespace

int prepare_sm87_nvfp4_marlin_p40_parity_gate_up_cuda(
    const std::uint8_t* const canonical_gate_weight,
    const std::uint8_t* const canonical_up_weight,
    const std::uint8_t* const canonical_gate_scales,
    const std::uint8_t* const canonical_up_scales,
    const float* const canonical_shared_weight_scale_2_device,
    const float scale_factor, std::uint8_t* const marlin_weight,
    std::uint8_t* const marlin_scales, float* const marlin_global_scale,
    void* const transpose_scratch,
    const std::size_t transpose_scratch_bytes,
    void* const cuda_stream) noexcept {
  if (!kP40ParityAdmitted) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return prepare_sm87_nvfp4_marlin_gate_up_cuda(
      canonical_gate_weight, canonical_up_weight, canonical_gate_scales,
      canonical_up_scales, canonical_shared_weight_scale_2_device,
      scale_factor, marlin_weight, marlin_scales, marlin_global_scale,
      transpose_scratch, transpose_scratch_bytes, cuda_stream, false);
}

int launch_sm87_nvfp4_marlin_p40_parity_gate_up_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const canonical_marlin_weight,
    const std::uint8_t* const canonical_marlin_scales,
    const float* const marlin_global_scale,
    std::uint16_t* const merged_gate_up_row_major_bf16,
    float* const reduction_workspace,
    const std::size_t reduction_workspace_bytes,
    std::int32_t* const locks, const std::size_t lock_bytes,
    Sm87NvFp4MarlinP40ParityLaunchCounters* const counters,
    void* const cuda_stream) noexcept {
  return launch_projection(Sm87NvFp4MarlinP40ParityRole::kGateUp, input,
                           canonical_marlin_weight,
                           canonical_marlin_scales, marlin_global_scale,
                           merged_gate_up_row_major_bf16,
                           reduction_workspace, reduction_workspace_bytes,
                           locks, lock_bytes, counters, cuda_stream);
}

int launch_sm87_nvfp4_marlin_p40_parity_silu_cuda(
    const std::uint16_t* const merged_gate_up_row_major_bf16,
    std::uint16_t* const activated_bf16,
    void* const cuda_stream) noexcept {
  if (!kP40ParityAdmitted) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (!aligned(merged_gate_up_row_major_bf16, 16U) ||
      !aligned(activated_bf16, 16U)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const ByteRange merged = byte_range(
      merged_gate_up_row_major_bf16,
      kSm87NvFp4MarlinP40ParityMergedGateUpBytes);
  const ByteRange activated = byte_range(
      activated_bf16, kSm87NvFp4MarlinP40ParityActivatedBytes);
  if (!merged.valid || !activated.valid || overlaps(merged, activated)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device = validate_fixed_device();
  if (device != static_cast<int>(cudaSuccess)) {
    return device;
  }
  constexpr unsigned int kBlockThreads = 256U;
  constexpr unsigned int kBlocks = 4'096U;
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  p40_parity_silu_kernel<<<kBlocks, kBlockThreads, 0U, stream>>>(
      merged_gate_up_row_major_bf16,
      kSm87NvFp4MarlinP40ParityActivatedElements, activated_bf16);
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_nvfp4_marlin_p40_parity_down_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    std::uint16_t* const branch_bf16,
    float* const reduction_workspace,
    const std::size_t reduction_workspace_bytes,
    std::int32_t* const locks, const std::size_t lock_bytes,
    Sm87NvFp4MarlinP40ParityLaunchCounters* const counters,
    void* const cuda_stream) noexcept {
  return launch_projection(Sm87NvFp4MarlinP40ParityRole::kDown, input,
                           marlin_weight, marlin_scales,
                           marlin_global_scale, branch_bf16,
                           reduction_workspace, reduction_workspace_bytes,
                           locks, lock_bytes, counters, cuda_stream);
}

int query_sm87_nvfp4_marlin_p40_parity_resources_cuda(
    const Sm87NvFp4MarlinP40ParityRole role,
    Sm87NvFp4MarlinP40ParityResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  resources->role = role;
  if (!kP40ParityAdmitted ||
      !sm87_nvfp4_marlin_p40_parity_plan(
           role, kSm87NvFp4MarlinP40ParityTokens)
           .valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const int device = validate_fixed_device();
  if (device != static_cast<int>(cudaSuccess)) {
    return device;
  }
  const auto legacy =
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                     vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                     kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                     kStages, kGroupBlocks, false, false,
                     marlin::MarlinTileRasterPolicy::kLegacyStripe, false>;
  const Sm87NvFp4MarlinP40ParityPlan plan =
      sm87_nvfp4_marlin_p40_parity_plan(
          role, kSm87NvFp4MarlinP40ParityTokens);
  const int status = read_resources(legacy, &resources->legacy_stripe);
  resources->reduction_workspace_bytes =
      plan.required_reduction_workspace_bytes;
  resources->lock_bytes = plan.required_lock_bytes;
  resources->tail_split_k_output_tiles = plan.tail_split_k_output_tiles;
  resources->tail_split_k_partial_slices =
      plan.tail_split_k_partial_slices;
  resources->bulk_and_tail_share_kernel = true;
  resources->requires_zero_initialized_locks =
      plan.requires_zero_initialized_locks;
  resources->atomic_add = plan.atomic_add;
  resources->fp32_reduce = plan.fp32_reduce;
  resources->supported = status == static_cast<int>(cudaSuccess);
  return status;
}

}  // namespace q3x::kernels
