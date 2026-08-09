#include "q3x/kernels/sm87_nvfp4_prefill_persistent.h"

#include "third_party/vllm_marlin/marlin_template.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

#if defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
inline constexpr bool kPersistentPrefillAdmitted = true;
#else
inline constexpr bool kPersistentPrefillAdmitted = false;
#endif

inline constexpr int kThreads =
    static_cast<int>(kSm87NvFp4PersistentPrefillThreads);
inline constexpr int kThreadMBlocks = 4;
inline constexpr int kThreadNBlocks = 16;
inline constexpr int kThreadKBlocks = 4;
inline constexpr int kStages = 4;
inline constexpr int kGroupBlocks = 1;
inline constexpr int kGroupSize = 16;

using GateKernel = decltype(
    marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                   vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                   kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                   kStages, kGroupBlocks, false, true,
                   marlin::MarlinTileRasterPolicy::kGroupedM4NMajor>);
using DownKernel = decltype(
    marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                   vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                   kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                   kStages, kGroupBlocks, false, false,
                   marlin::MarlinTileRasterPolicy::kBStationaryNMajor,
                   true>);

static_assert(kThreadMBlocks * 16 ==
              static_cast<int>(kSm87NvFp4PersistentPrefillTileM));
static_assert(kThreadNBlocks * 16 ==
              static_cast<int>(kSm87NvFp4PersistentPrefillTileN));
static_assert(kThreadKBlocks * 16 ==
              static_cast<int>(kSm87NvFp4PersistentPrefillTileK));
static_assert(kStages ==
              static_cast<int>(kSm87NvFp4PersistentPrefillPipelineStages));
static_assert(kSm87NvFp4PersistentPrefillP40Tokens <=
              static_cast<std::size_t>(std::numeric_limits<int>::max()));
static_assert(kSm87NvFp4PersistentPrefillP40Tokens *
                      kSm87NvFp4PersistentPrefillMergedGateUp <=
                  static_cast<std::size_t>(
                      std::numeric_limits<std::ptrdiff_t>::max()) /
                      sizeof(std::uint16_t));

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
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
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

[[nodiscard]] int validate_device() noexcept {
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
          static_cast<int>(kSm87NvFp4PersistentPrefillCtas) ||
      properties.sharedMemPerBlockOptin <
          kSm87NvFp4PersistentPrefillDynamicSharedBytes) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  return static_cast<int>(cudaSuccess);
}

template <typename Kernel>
[[nodiscard]] int configure_kernel(const Kernel kernel) noexcept {
  return static_cast<int>(cudaFuncSetAttribute(
      kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87NvFp4PersistentPrefillDynamicSharedBytes)));
}

template <typename Kernel>
[[nodiscard]] int read_resources(
    const Kernel kernel,
    Sm87NvFp4PersistentPrefillResources* const resources) noexcept {
  const int configure = configure_kernel(kernel);
  if (configure != static_cast<int>(cudaSuccess)) {
    return configure;
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(&attributes, kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, kernel, kThreads,
      kSm87NvFp4PersistentPrefillDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87NvFp4PersistentPrefillDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active;
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool valid_common_ranges(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale, const std::size_t token_count,
    const std::size_t input_features, const std::size_t output_features,
    const std::size_t weight_output_features,
    const std::uint16_t* const output) noexcept {
  if (!aligned(input, 16U) || !aligned(marlin_weight, 16U) ||
      !aligned(marlin_scales, 16U) ||
      !aligned(marlin_global_scale, alignof(float)) ||
      !aligned(output, 16U)) {
    return false;
  }
  const ByteRange input_range = byte_range(
      input, token_count * input_features * sizeof(std::uint16_t));
  const ByteRange weight_range = byte_range(
      marlin_weight, weight_output_features * input_features / 2U);
  const ByteRange scale_range = byte_range(
      marlin_scales, weight_output_features * input_features / 16U);
  const ByteRange global_scale_range =
      byte_range(marlin_global_scale, sizeof(float));
  const ByteRange output_range = byte_range(
      output, token_count * output_features * sizeof(std::uint16_t));
  if (!input_range.valid || !weight_range.valid || !scale_range.valid ||
      !global_scale_range.valid || !output_range.valid) {
    return false;
  }
  return !overlaps(output_range, input_range) &&
         !overlaps(output_range, weight_range) &&
         !overlaps(output_range, scale_range) &&
         !overlaps(output_range, global_scale_range);
}

}  // namespace

int query_sm87_nvfp4_persistent_prefill_capability_cuda(
    const Sm87NvFp4PersistentPrefillRole role,
    const std::size_t token_count,
    Sm87NvFp4PersistentPrefillCapability* const capability) noexcept {
  if (capability == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *capability = {};
  capability->plan = sm87_nvfp4_persistent_prefill_plan(role, token_count);
  if (!kPersistentPrefillAdmitted || !capability->plan.valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }

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
  capability->device = device;
  capability->compute_major = properties.major;
  capability->compute_minor = properties.minor;
  capability->sm_count = properties.multiProcessorCount;
  capability->optin_shared_bytes_per_block =
      properties.sharedMemPerBlockOptin;
  capability->supported =
      validate_device() == static_cast<int>(cudaSuccess);
  return capability->supported ? static_cast<int>(cudaSuccess)
                               : static_cast<int>(cudaErrorNotSupported);
}

int query_sm87_nvfp4_persistent_prefill_resources_cuda(
    const Sm87NvFp4PersistentPrefillRole role,
    const std::size_t token_count,
    Sm87NvFp4PersistentPrefillResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};
  if (!kPersistentPrefillAdmitted ||
      !sm87_nvfp4_persistent_prefill_plan(role, token_count).valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const int device = validate_device();
  if (device != static_cast<int>(cudaSuccess)) {
    return device;
  }
  if (role == Sm87NvFp4PersistentPrefillRole::kGateUpPaired) {
    const auto kernel =
        marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                       vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                       kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                       kStages, kGroupBlocks, false, true,
                       marlin::MarlinTileRasterPolicy::kGroupedM4NMajor>;
    return read_resources(kernel, resources);
  }
  const auto kernel =
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                     vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                     kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                     kStages, kGroupBlocks, false, false,
                     marlin::MarlinTileRasterPolicy::kBStationaryNMajor,
                     true>;
  return read_resources(kernel, resources);
}

int launch_sm87_nvfp4_persistent_prefill_gate_up_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const interleaved_marlin_weight,
    const std::uint8_t* const interleaved_marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const activated_output,
    void* const cuda_stream) noexcept {
  const auto plan = sm87_nvfp4_persistent_prefill_plan(
      Sm87NvFp4PersistentPrefillRole::kGateUpPaired, token_count);
  if (!kPersistentPrefillAdmitted || !plan.valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (!valid_common_ranges(
          input, interleaved_marlin_weight, interleaved_marlin_scales,
          marlin_global_scale, token_count,
          kSm87NvFp4PersistentPrefillHidden,
          kSm87NvFp4PersistentPrefillIntermediate,
          kSm87NvFp4PersistentPrefillMergedGateUp, activated_output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device = validate_device();
  if (device != static_cast<int>(cudaSuccess)) {
    return device;
  }
  const auto kernel =
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                     vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                     kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                     kStages, kGroupBlocks, false, true,
                     marlin::MarlinTileRasterPolicy::kGroupedM4NMajor>;
  // Binding queries resources once per role and sets the sticky function
  // attribute before admitting this route.  Keep the timed launch path free
  // of repeated cudaFuncSetAttribute control traffic.
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  kernel<<<static_cast<unsigned int>(kSm87NvFp4PersistentPrefillCtas),
           kThreads, kSm87NvFp4PersistentPrefillDynamicSharedBytes, stream>>>(
      reinterpret_cast<const int4*>(input),
      reinterpret_cast<const int4*>(interleaved_marlin_weight), nullptr,
      nullptr, reinterpret_cast<const int4*>(activated_output), nullptr,
      reinterpret_cast<const int4*>(interleaved_marlin_scales),
      marlin_global_scale, nullptr, nullptr,
      static_cast<int>(kSm87NvFp4PersistentPrefillHidden / kGroupSize),
      static_cast<int>(token_count),
      static_cast<int>(kSm87NvFp4PersistentPrefillMergedGateUp),
      static_cast<int>(kSm87NvFp4PersistentPrefillHidden),
      static_cast<int>(kSm87NvFp4PersistentPrefillHidden), nullptr, false,
      false, true,
      static_cast<int>(kSm87NvFp4PersistentPrefillDynamicSharedBytes));
  return static_cast<int>(cudaPeekAtLastError());
}

int launch_sm87_nvfp4_persistent_prefill_down_residual_cuda(
    const std::uint16_t* const input,
    const std::uint8_t* const marlin_weight,
    const std::uint8_t* const marlin_scales,
    const float* const marlin_global_scale,
    const std::size_t token_count,
    std::uint16_t* const residual_in_out,
    void* const cuda_stream) noexcept {
  const auto plan = sm87_nvfp4_persistent_prefill_plan(
      Sm87NvFp4PersistentPrefillRole::kDown, token_count);
  if (!kPersistentPrefillAdmitted || !plan.valid()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  if (!valid_common_ranges(input, marlin_weight, marlin_scales,
                           marlin_global_scale, token_count,
                           kSm87NvFp4PersistentPrefillIntermediate,
                           kSm87NvFp4PersistentPrefillHidden,
                           kSm87NvFp4PersistentPrefillHidden,
                           residual_in_out)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const int device = validate_device();
  if (device != static_cast<int>(cudaSuccess)) {
    return device;
  }
  const auto kernel =
      marlin::Marlin<vllm::kBFloat16.id(), vllm::kFE2M1f.id(),
                     vllm::kBFloat16.id(), vllm::kFE4M3fn.id(), kThreads,
                     kThreadMBlocks, kThreadNBlocks, kThreadKBlocks, false,
                     kStages, kGroupBlocks, false, false,
                     marlin::MarlinTileRasterPolicy::kBStationaryNMajor,
                     true>;
  // The capability/resource bind configured this kernel before request
  // execution; see the paired GateUp launch above.
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  kernel<<<static_cast<unsigned int>(kSm87NvFp4PersistentPrefillCtas),
           kThreads, kSm87NvFp4PersistentPrefillDynamicSharedBytes, stream>>>(
      reinterpret_cast<const int4*>(input),
      reinterpret_cast<const int4*>(marlin_weight),
      reinterpret_cast<int4*>(residual_in_out), nullptr, nullptr, nullptr,
      reinterpret_cast<const int4*>(marlin_scales), marlin_global_scale,
      nullptr, nullptr,
      static_cast<int>(kSm87NvFp4PersistentPrefillIntermediate / kGroupSize),
      static_cast<int>(token_count),
      static_cast<int>(kSm87NvFp4PersistentPrefillHidden),
      static_cast<int>(kSm87NvFp4PersistentPrefillIntermediate),
      static_cast<int>(kSm87NvFp4PersistentPrefillIntermediate), nullptr,
      false, false, true,
      static_cast<int>(kSm87NvFp4PersistentPrefillDynamicSharedBytes));
  return static_cast<int>(cudaPeekAtLastError());
}

}  // namespace q3x::kernels
