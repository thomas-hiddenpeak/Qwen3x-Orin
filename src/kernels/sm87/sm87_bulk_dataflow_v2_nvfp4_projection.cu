#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_projection.h"

#include <cooperative_groups.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {
namespace {

namespace cg = cooperative_groups;

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87BulkV2NvFp4Threads);
constexpr unsigned int kPersistentCtas =
    static_cast<unsigned int>(kSm87BulkV2NvFp4PersistentCtas);
constexpr unsigned int kDownOwners =
    static_cast<unsigned int>(kSm87BulkV2NvFp4DownOwnerCtas);
constexpr unsigned int kRows =
    static_cast<unsigned int>(kSm87BulkV2NvFp4RowsPerGroup);
constexpr unsigned int kReadinessSlots =
    static_cast<unsigned int>(kSm87BulkV2NvFp4ReadinessSlots);
constexpr unsigned int kGateTasksPerRow =
    static_cast<unsigned int>(kSm87BulkV2NvFp4GateNTiles);
constexpr unsigned int kStageBytes =
    static_cast<unsigned int>(kSm87BulkV2NvFp4DownBytesPerStage);
constexpr unsigned int kStageVectors = kStageBytes / sizeof(uint4);
constexpr unsigned int kActivationVectors =
    static_cast<unsigned int>(kSm87BulkV2NvFp4ActivationBytesPerStage) /
    sizeof(uint4);
constexpr unsigned int kPayloadVectors =
    kStageVectors - kActivationVectors;

static_assert(kThreads == 256U && kPersistentCtas == 32U);
static_assert(kDownOwners == 20U &&
              kSm87BulkV2NvFp4DedicatedProducerCtas == 12U);
static_assert(kRows == 4U && kReadinessSlots == 32U);
static_assert(kStageBytes == 17'408U && kStageVectors == 1'088U);
static_assert(kActivationVectors == 512U && kPayloadVectors == 576U);

// Compile-time storage proof for the common, larger Down envelope. Gate/Up
// reuses the same three-stage allocation and leaves its excess bytes idle.
// This is intentionally not a numerical pipeline implementation.
struct alignas(16) ThreeStageStorage final {
  uint4 stage[kSm87BulkV2NvFp4PipelineStages][kStageVectors];
};

static_assert(sizeof(ThreeStageStorage) ==
              kSm87BulkV2NvFp4DynamicSharedBytes);

template <bool kCacheAll>
__device__ __forceinline__ void cp_async_16(
    void* const shared_destination, const void* const global_source,
    const bool valid) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(
          shared_destination));
  if constexpr (kCacheAll) {
    asm volatile("cp.async.ca.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(shared_address), "l"(global_source),
                   "r"(valid ? 16U : 0U)
                 : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;"
                 :
                 : "r"(shared_address), "l"(global_source),
                   "r"(valid ? 16U : 0U)
                 : "memory");
  }
#else
  *reinterpret_cast<uint4*>(shared_destination) =
      valid ? *reinterpret_cast<const uint4*>(global_source)
            : make_uint4(0U, 0U, 0U, 0U);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_all() noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

struct GateClaim final {
  unsigned int row = kRows;
  unsigned int q = kGateTasksPerRow;
  bool valid = false;
};

[[nodiscard]] __device__ __forceinline__ bool cancellation_requested(
    const std::uint32_t* const signal) noexcept {
  return signal != nullptr &&
         *reinterpret_cast<const volatile std::uint32_t*>(signal) != 0U;
}

// Claims exactly one (row,q) task. The per-row CAS cursor and retired cursor
// jointly enforce the 32-credit ring; there is no split-K or reduction queue.
[[nodiscard]] __device__ __forceinline__ GateClaim try_claim_gate_task(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const unsigned int active_rows, const unsigned int scan_seed) noexcept {
  for (unsigned int offset = 0U; offset < active_rows; ++offset) {
    const unsigned int row = (scan_seed + offset) % active_rows;
    unsigned int next = atomicAdd(&control->gate_next_q[row], 0U);
    const unsigned int retired = atomicAdd(&control->retired_q[row], 0U);
    while (next < kGateTasksPerRow &&
           next < retired + kReadinessSlots) {
      const unsigned int observed =
          atomicCAS(&control->gate_next_q[row], next, next + 1U);
      if (observed == next) {
        atomicAdd(&control->claimed_gate_tasks, 1U);
        return {row, next, true};
      }
      next = observed;
    }
  }
  return {};
}

// Skeleton-only publication represents completion of the future paired
// Gate/Up numerical tile. A real body must first write and fence the exact
// BF16 H tile; this compile-only symbol has no output pointer and cannot do so.
__device__ __forceinline__ void publish_skeleton_gate_claim(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const GateClaim claim) noexcept {
  const unsigned int slot = claim.q % kReadinessSlots;
  const unsigned int generation = claim.q / kReadinessSlots + 1U;
  atomicExch(&control->readers_remaining[claim.row][slot], kDownOwners);
  __threadfence();
  atomicExch(&control->readiness_generation[claim.row][slot], generation);
  atomicAdd(&control->completed_gate_tasks, 1U);
}

[[nodiscard]] __device__ __forceinline__ bool consume_ready_claim(
    Sm87BulkV2NvFp4DeviceControl* const control, const unsigned int row,
    const unsigned int q) noexcept {
  const unsigned int slot = q % kReadinessSlots;
  const unsigned int generation = q / kReadinessSlots + 1U;
  if (atomicAdd(&control->readiness_generation[row][slot], 0U) !=
      generation) {
    return false;
  }
  unsigned int readers_before =
      atomicAdd(&control->readers_remaining[row][slot], 0U);
  while (readers_before != 0U) {
    const unsigned int observed = atomicCAS(
        &control->readers_remaining[row][slot], readers_before,
        readers_before - 1U);
    if (observed == readers_before) {
      if (readers_before == 1U) {
        __threadfence();
        atomicExch(&control->retired_q[row], q + 1U);
      }
      return true;
    }
    readers_before = observed;
  }
  return false;
}

// This kernel exists only to compile and inspect the proposed fixed-32
// control/resource shape. It performs no projection arithmetic and is never
// reachable through the fail-closed public launcher below.
__global__ __launch_bounds__(256, 2)
void work_conserving_nvfp4_compile_only_kernel(
    Sm87BulkV2NvFp4DeviceControl* const control,
    const std::uint32_t* const cancellation_signal,
    const unsigned int active_rows, const unsigned int group_epoch,
    const uint4* const activation_probe,
    const uint4* const payload_probe, std::uint32_t* const trace) {
  extern __shared__ __align__(16) unsigned char shared_bytes[];
  auto* const pipeline =
      reinterpret_cast<ThreeStageStorage*>(shared_bytes);

  // Preserve the intended A.ca / payload.cg policy and all three global to
  // shared stages in the compiled code. These probe bytes have no numerical
  // meaning and the public ABI never launches this symbol.
#pragma unroll
  for (unsigned int stage = 0U;
       stage < kSm87BulkV2NvFp4PipelineStages; ++stage) {
    for (unsigned int vector = threadIdx.x; vector < kStageVectors;
         vector += blockDim.x) {
      if (vector < kActivationVectors) {
        const uint4* const activation_source =
            activation_probe == nullptr ? activation_probe
                                        : activation_probe + vector;
        cp_async_16<true>(&pipeline->stage[stage][vector],
                          activation_source,
                          activation_probe != nullptr);
      } else {
        const unsigned int payload_vector = vector - kActivationVectors;
        const uint4* const payload_source =
            payload_probe == nullptr ? payload_probe
                                     : payload_probe + payload_vector;
        cp_async_16<false>(&pipeline->stage[stage][vector],
                           payload_source,
                           payload_probe != nullptr);
      }
    }
    cp_async_commit_group();
  }
  cp_async_wait_all();
  __syncthreads();

  if (threadIdx.x == 0U && blockIdx.x < kPersistentCtas &&
      control != nullptr && active_rows > 0U && active_rows <= kRows &&
      control->group_epoch == group_epoch) {
    const bool down_owner = blockIdx.x < kDownOwners;
    if (down_owner) {
      // A consumer CTA owns one fixed N256 tile. It walks every row and all
      // 272 J64 tiles in ascending full-K order. Whenever the next H tile is
      // unavailable it executes eligible paired Gate/Up producer work.
      for (unsigned int row = 0U; row < active_rows; ++row) {
        for (unsigned int q = 0U; q < kGateTasksPerRow;) {
          if (cancellation_requested(cancellation_signal)) {
            atomicExch(&control->cancellation_observed, 1U);
            break;
          }
          if (consume_ready_claim(control, row, q)) {
            ++q;
          } else {
            const GateClaim claim = try_claim_gate_task(
                control, active_rows, blockIdx.x + row + q);
            if (claim.valid) {
              publish_skeleton_gate_claim(control, claim);
            } else {
              __nanosleep(64U);
            }
          }
        }
        if (atomicAdd(&control->cancellation_observed, 0U) != 0U) {
          break;
        }
        atomicAdd(&control->completed_down_tasks, 1U);
      }
    } else {
      // The remaining 12 CTAs are dedicated paired Gate/Up producers. Both
      // classes call the same CAS claim path, so work stealing cannot create
      // duplicate ownership.
      const unsigned int target = active_rows * kGateTasksPerRow;
      unsigned int retry = 0U;
      while (atomicAdd(&control->completed_gate_tasks, 0U) < target) {
        if (cancellation_requested(cancellation_signal)) {
          atomicExch(&control->cancellation_observed, 1U);
          break;
        }
        const GateClaim claim = try_claim_gate_task(
            control, active_rows, blockIdx.x + retry);
        if (claim.valid) {
          publish_skeleton_gate_claim(control, claim);
        } else {
          ++retry;
          __nanosleep(64U);
        }
      }
    }
  }

  cg::this_grid().sync();
  if (trace != nullptr && threadIdx.x == 0U &&
      blockIdx.x < kPersistentCtas) {
    trace[blockIdx.x] =
        control == nullptr
            ? 0U
            : (atomicAdd(&control->cancellation_observed, 0U) << 31U) |
                  (atomicAdd(&control->completed_gate_tasks, 0U) &
                   0x7fff'ffffU);
  }
}

[[nodiscard]] cudaError_t validate_fixed_sm87_device(
    cudaDeviceProp* const properties) noexcept {
  if (properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    return status;
  }
  status = cudaGetDeviceProperties(properties, device);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     static_cast<int>(kSm87BulkV2NvFp4SmCount) &&
                 properties->cooperativeLaunch != 0 &&
                 properties->sharedMemPerBlockOptin >=
                     kSm87BulkV2NvFp4DynamicSharedBytes
             ? cudaSuccess
             : cudaErrorNotSupported;
}

}  // namespace

int query_sm87_bulk_dataflow_v2_nvfp4_resources_cuda(
    const Sm87BulkV2NvFp4CodeEvidence* const code_evidence,
    Sm87BulkV2NvFp4KernelResources* const resources) noexcept {
  if (resources == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *resources = {};

  cudaDeviceProp properties{};
  cudaError_t status = validate_fixed_sm87_device(&properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = cudaFuncSetAttribute(
      work_conserving_nvfp4_compile_only_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87BulkV2NvFp4DynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes, work_conserving_nvfp4_compile_only_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, work_conserving_nvfp4_compile_only_kernel,
      static_cast<int>(kThreads),
      kSm87BulkV2NvFp4DynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87BulkV2NvFp4DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->cooperative_grid_capacity =
      active_blocks * properties.multiProcessorCount;
  if (code_evidence != nullptr) {
    resources->code = *code_evidence;
  }
  resources->kernel_compiled = true;
  resources->cooperative_launch_supported =
      properties.cooperativeLaunch != 0;
  resources->numerical_contract_qualified = false;
  resources->production_dispatch_eligible = false;
  resources->resource_and_code_gate_passed =
      resources->binary_version == 87 &&
      resources->registers_per_thread > 0 &&
      resources->registers_per_thread <=
          static_cast<int>(kSm87BulkV2NvFp4MaximumRegisters) &&
      resources->static_shared_bytes == 0U &&
      resources->local_bytes == 0U &&
      resources->maximum_threads_per_block >=
          static_cast<int>(kThreads) &&
      resources->active_blocks_per_sm >=
          static_cast<int>(kSm87BulkV2NvFp4RequiredCtasPerSm) &&
      resources->cooperative_grid_capacity >=
          static_cast<int>(kPersistentCtas) &&
      sm87_bulk_v2_nvfp4_code_evidence_valid(resources->code);
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_bulk_dataflow_v2_nvfp4_compile_only_cuda(
    const Sm87BulkV2NvFp4MacroArguments& arguments) noexcept {
  // The compiled kernel has no numerical body and this symbol deliberately
  // exposes no cudaLaunchCooperativeKernel call. A new reviewed ABI is
  // required before any execution, numerical qualification, or production
  // selector can exist.
  (void)arguments;
  return static_cast<int>(cudaErrorNotSupported);
}

}  // namespace q3x::kernels
