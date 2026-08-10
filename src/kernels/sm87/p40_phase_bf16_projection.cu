#include "q3x/kernels/sm87_p40_phase_bf16_projection.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cmath>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads =
    static_cast<unsigned int>(kSm87P40PhaseBf16Threads);
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kTileM =
    static_cast<unsigned int>(kSm87P40PhaseBf16TileM);
constexpr unsigned int kTileN =
    static_cast<unsigned int>(kSm87P40PhaseBf16TileN);
constexpr unsigned int kTileK =
    static_cast<unsigned int>(kSm87P40PhaseBf16TileK);
constexpr unsigned int kPipelineStages =
    static_cast<unsigned int>(kSm87P40PhaseBf16PipelineStages);
constexpr unsigned int kSharedLeadingDimension =
    static_cast<unsigned int>(kSm87P40PhaseBf16SharedLeadingDimension);
constexpr unsigned int kVectorsPerGlobalRow =
    kTileK * sizeof(std::uint16_t) / sizeof(uint4);
constexpr unsigned int kVectorsPerSharedRow =
    kSharedLeadingDimension * sizeof(std::uint16_t) / sizeof(uint4);
constexpr unsigned int kK16PerStage = kTileK / 16U;
constexpr unsigned int kWarpM = 32U;
constexpr unsigned int kWarpN = 64U;
constexpr unsigned int kM16PanelsPerWarp = kWarpM / 16U;
constexpr unsigned int kN8PanelsPerWarp = kWarpN / 8U;

static_assert(kThreads == 8U * kWarpSize);
static_assert(kVectorsPerGlobalRow == 8U);
static_assert(kVectorsPerSharedRow == 9U);
static_assert(kK16PerStage == 4U);
static_assert(kM16PanelsPerWarp == 2U);
static_assert(kN8PanelsPerWarp == 8U);

struct alignas(32) PipelineStorage final {
  uint4 activations[kPipelineStages][kTileM][kVectorsPerSharedRow];
  uint4 weights[kPipelineStages][kTileN][kVectorsPerSharedRow];
};

static_assert(sizeof(PipelineStorage) ==
              kSm87P40PhaseBf16DynamicSharedBytes);

struct M16K16Activation final {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct K16N8Weight final {
  std::uint32_t x0;
  std::uint32_t x1;
};

struct M16N8Accumulator final {
  float x0;
  float x1;
  float x2;
  float x3;
};

[[nodiscard]] bool pointer_is_aligned(const void* const pointer,
                                      const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
}

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  if (pointer == nullptr) {
    return true;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

[[nodiscard]] bool byte_ranges_overlap(const void* const first,
                                       const std::size_t first_bytes,
                                       const void* const second,
                                       const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return true;
  }
  const std::uintptr_t first_begin = reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

template <bool kPredicate>
__device__ __forceinline__ void cp_async_cg_shared_global_16(
    void* const destination, const void* const source,
    const bool valid = true) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  if constexpr (kPredicate) {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" :
                 : "r"(shared_address), "l"(source),
                   "r"(valid ? 16U : 0U)
                 : "memory");
  } else {
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
                 : "r"(shared_address), "l"(source)
                 : "memory");
  }
#else
  *reinterpret_cast<uint4*>(destination) =
      valid ? *reinterpret_cast<const uint4*>(source)
            : make_uint4(0U, 0U, 0U, 0U);
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

template <unsigned int kRemaining>
__device__ __forceinline__ void cp_async_wait_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group %0;" : : "n"(kRemaining) : "memory");
#endif
}

__device__ __forceinline__ void load_m16k16_activation_fragment(
    M16K16Activation* const fragment,
    const std::uint16_t* const shared_activations,
    const unsigned int absolute_m16_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      absolute_m16_panel * 16U + (lane % 8U) + (quadrant & 1U) * 8U;
  const unsigned int column = k16 * 16U + (quadrant >> 1U) * 8U;
  const std::uint16_t* const source =
      shared_activations + row * kSharedLeadingDimension + column;
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment->x0), "=r"(fragment->x1),
        "=r"(fragment->x2), "=r"(fragment->x3)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_activations;
  (void)absolute_m16_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void load_k16n8_weight_fragment(
    K16N8Weight* const fragment,
    const std::uint16_t* const shared_weights,
    const unsigned int absolute_n8_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  // Canonical W is [N,K] row-major, which is already the column-major
  // backing of logical MMA matrix B[K,N].  A non-transposed ldmatrix maps
  // source W[group_id][2*thread_id + {0,1}] directly to
  // B[2*thread_id + {0,1}][group_id].  Using .trans here would be correct
  // for a row-major [K,N] backing, but would transpose canonical W twice and
  // mix independent output rows inside the MMA B fragment.
  const unsigned int row = absolute_n8_panel * 8U + (lane & 7U);
  const unsigned int column =
      k16 * 16U + ((lane >> 3U) & 1U) * 8U;
  const std::uint16_t* const source =
      shared_weights + row * kSharedLeadingDimension + column;
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.shared.b16 "
      "{%0, %1}, [%2];"
      : "=r"(fragment->x0), "=r"(fragment->x1)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_weights;
  (void)absolute_n8_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    M16N8Accumulator* const accumulator,
    const M16K16Activation& activation, const K16N8Weight& weight) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator->x0), "+f"(accumulator->x1),
        "+f"(accumulator->x2), "+f"(accumulator->x3)
      : "r"(activation.x0), "r"(activation.x1), "r"(activation.x2),
        "r"(activation.x3), "r"(weight.x0), "r"(weight.x1));
#else
  (void)accumulator;
  (void)activation;
  (void)weight;
#endif
}

[[nodiscard]] __device__ __forceinline__ std::uint16_t encode_bf16_rne(
    const float value) {
  unsigned int bits = __float_as_uint(value);
  const unsigned int magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_bf16_pair(
    const float low, const float high, const float alpha) {
  return static_cast<std::uint32_t>(encode_bf16_rne(low * alpha)) |
         (static_cast<std::uint32_t>(encode_bf16_rne(high * alpha)) << 16U);
}

template <unsigned int kColumns>
__device__ __forceinline__ void issue_pipeline_stage(
    PipelineStorage* const storage, const unsigned int slot,
    const std::uint16_t* const activations,
    const std::uint16_t* const weights, const unsigned int first_token,
    const unsigned int first_output, const unsigned int first_k) {
  static_assert(kColumns == 5'120U || kColumns == 6'144U ||
                kColumns == 17'408U);
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    const unsigned int row = index / kVectorsPerGlobalRow;
    const unsigned int vector = index % kVectorsPerGlobalRow;
    const bool valid =
        first_token + row < kSm87P40PhaseBf16Tokens;
    const unsigned int source_token = valid ? first_token + row : 0U;
    cp_async_cg_shared_global_16<true>(
        &storage->activations[slot][row][vector],
        reinterpret_cast<const uint4*>(
            activations +
            static_cast<std::size_t>(source_token) * kColumns + first_k) +
            vector,
        valid);
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 4U; ++pass) {
    const unsigned int index = threadIdx.x + pass * kThreads;
    const unsigned int row = index / kVectorsPerGlobalRow;
    const unsigned int vector = index % kVectorsPerGlobalRow;
    cp_async_cg_shared_global_16<false>(
        &storage->weights[slot][row][vector],
        reinterpret_cast<const uint4*>(
            weights +
            static_cast<std::size_t>(first_output + row) * kColumns +
            first_k) +
            vector);
  }
  cp_async_commit_group();
}

template <unsigned int kColumns>
__global__ __launch_bounds__(kThreads, 1)
void p40_phase_bf16_projection_m128_n128_k64_kernel(
    const std::uint16_t* __restrict__ activations,
    const std::uint16_t* __restrict__ weights, const unsigned int rows,
    const float alpha, std::uint16_t* __restrict__ output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const storage = reinterpret_cast<PipelineStorage*>(dynamic_storage);
  constexpr unsigned int kKStages = kColumns / kTileK;
  static_assert(kKStages == 80U || kKStages == 96U || kKStages == 272U);

  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int warp_m = warp & 3U;
  const unsigned int warp_n = warp >> 2U;
  const unsigned int first_token = blockIdx.y * kTileM;
  const unsigned int first_output = blockIdx.x * kTileN;

  M16N8Accumulator accumulators[kM16PanelsPerWarp][kN8PanelsPerWarp];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16PanelsPerWarp; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp; ++n_panel) {
      accumulators[m_panel][n_panel] = {0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  issue_pipeline_stage<kColumns>(storage, 0U, activations, weights,
                                 first_token, first_output, 0U);
  issue_pipeline_stage<kColumns>(storage, 1U, activations, weights,
                                 first_token, first_output, kTileK);

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kKStages; ++stage) {
    if (stage + 1U < kKStages) {
      cp_async_wait_group<1U>();
    } else {
      cp_async_wait_group<0U>();
    }
    __syncthreads();
    const unsigned int slot = stage & 1U;
    const auto* const shared_activations =
        reinterpret_cast<const std::uint16_t*>(storage->activations[slot]);
    const auto* const shared_weights =
        reinterpret_cast<const std::uint16_t*>(storage->weights[slot]);

#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16PerStage; ++k16) {
      M16K16Activation activation_fragments[kM16PanelsPerWarp];
#pragma unroll
      for (unsigned int m_panel = 0U; m_panel < kM16PanelsPerWarp;
           ++m_panel) {
        load_m16k16_activation_fragment(
            &activation_fragments[m_panel], shared_activations,
            warp_m * kM16PanelsPerWarp + m_panel, k16, lane);
      }
#pragma unroll
      for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
           ++n_panel) {
        K16N8Weight weight_fragment{};
        load_k16n8_weight_fragment(
            &weight_fragment, shared_weights,
            warp_n * kN8PanelsPerWarp + n_panel, k16, lane);
#pragma unroll
        for (unsigned int m_panel = 0U; m_panel < kM16PanelsPerWarp;
             ++m_panel) {
          mma_m16n8k16_bf16(&accumulators[m_panel][n_panel],
                            activation_fragments[m_panel], weight_fragment);
        }
      }
    }
    __syncthreads();
    if (stage + kPipelineStages < kKStages) {
      const unsigned int future_stage = stage + kPipelineStages;
      issue_pipeline_stage<kColumns>(
          storage, slot, activations, weights, first_token, first_output,
          future_stage * kTileK);
    }
  }
  cp_async_wait_group<0U>();
  __syncthreads();

  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kM16PanelsPerWarp; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kN8PanelsPerWarp;
         ++n_panel) {
      const unsigned int local_token0 =
          warp_m * kWarpM + m_panel * 16U + lane_group;
      const unsigned int local_token1 = local_token0 + 8U;
      const unsigned int output_column =
          first_output + warp_n * kWarpN + n_panel * 8U +
          2U * lane_in_group;
      if (first_token + local_token0 < kSm87P40PhaseBf16Tokens) {
        *reinterpret_cast<std::uint32_t*>(
            output +
            static_cast<std::size_t>(first_token + local_token0) * rows +
            output_column) =
            pack_bf16_pair(accumulators[m_panel][n_panel].x0,
                           accumulators[m_panel][n_panel].x1, alpha);
      }
      if (first_token + local_token1 < kSm87P40PhaseBf16Tokens) {
        *reinterpret_cast<std::uint32_t*>(
            output +
            static_cast<std::size_t>(first_token + local_token1) * rows +
            output_column) =
            pack_bf16_pair(accumulators[m_panel][n_panel].x2,
                           accumulators[m_panel][n_panel].x3, alpha);
      }
    }
  }
}

template <unsigned int kColumns>
[[nodiscard]] cudaError_t set_dynamic_shared_attribute() noexcept {
  return cudaFuncSetAttribute(
      p40_phase_bf16_projection_m128_n128_k64_kernel<kColumns>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87P40PhaseBf16DynamicSharedBytes));
}

template <unsigned int kColumns>
[[nodiscard]] cudaError_t inspect_kernel(
    cudaFuncAttributes* const attributes, int* const active_blocks) noexcept {
  cudaError_t status = cudaFuncGetAttributes(
      attributes,
      p40_phase_bf16_projection_m128_n128_k64_kernel<kColumns>);
  if (status != cudaSuccess) {
    return status;
  }
  status = set_dynamic_shared_attribute<kColumns>();
  if (status != cudaSuccess) {
    return status;
  }
  return cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      active_blocks,
      p40_phase_bf16_projection_m128_n128_k64_kernel<kColumns>,
      static_cast<int>(kThreads), kSm87P40PhaseBf16DynamicSharedBytes);
}

template <unsigned int kColumns>
[[nodiscard]] cudaError_t launch_kernel(
    const Sm87P40PhaseBf16ProjectionPlan& plan,
    const std::uint16_t* const activations,
    const std::uint16_t* const weights, const float alpha,
    std::uint16_t* const output, cudaStream_t const stream) noexcept {
  cudaError_t status = set_dynamic_shared_attribute<kColumns>();
  if (status != cudaSuccess) {
    return status;
  }
  const dim3 grid(static_cast<unsigned int>(plan.grid_n),
                  static_cast<unsigned int>(plan.grid_m), 1U);
  p40_phase_bf16_projection_m128_n128_k64_kernel<kColumns>
      <<<grid, kThreads, kSm87P40PhaseBf16DynamicSharedBytes, stream>>>(
          activations, weights,
          static_cast<unsigned int>(plan.output_features), alpha, output);
  return cudaGetLastError();
}

[[nodiscard]] int invalid_value() noexcept {
  return static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] cudaError_t inspect_role_kernel(
    const std::size_t columns, cudaFuncAttributes* const attributes,
    int* const active_blocks) noexcept {
  switch (columns) {
    case 5'120U:
      return inspect_kernel<5'120U>(attributes, active_blocks);
    case 6'144U:
      return inspect_kernel<6'144U>(attributes, active_blocks);
    case 17'408U:
      return inspect_kernel<17'408U>(attributes, active_blocks);
    default:
      return cudaErrorInvalidValue;
  }
}

[[nodiscard]] cudaError_t launch_role_kernel(
    const Sm87P40PhaseBf16ProjectionPlan& plan,
    const std::uint16_t* const activations,
    const std::uint16_t* const weights, const float alpha,
    std::uint16_t* const output, cudaStream_t const stream) noexcept {
  switch (plan.input_features) {
    case 5'120U:
      return launch_kernel<5'120U>(plan, activations, weights, alpha, output,
                                   stream);
    case 6'144U:
      return launch_kernel<6'144U>(plan, activations, weights, alpha, output,
                                   stream);
    case 17'408U:
      return launch_kernel<17'408U>(plan, activations, weights, alpha, output,
                                    stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

int query_sm87_p40_phase_bf16_projection_resources_cuda(
    const Sm87P40PhaseBf16ProjectionRole role,
    const std::size_t token_count,
    Sm87P40PhaseBf16ProjectionResources* const resources) noexcept {
  if (resources == nullptr) {
    return invalid_value();
  }
  *resources = {};
  const auto plan =
      make_sm87_p40_phase_bf16_projection_plan(role, token_count);
  if (!plan.valid()) {
    return invalid_value();
  }

  int device = 0;
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
  resources->sm_count = properties.multiProcessorCount;
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kSm87P40PhaseBf16SmCount)) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  cudaFuncAttributes attributes{};
  int active_blocks = 0;
  status = inspect_role_kernel(plan.input_features, &attributes,
                               &active_blocks);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87P40PhaseBf16DynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->active_blocks_per_sm = active_blocks;
  resources->admitted =
      attributes.binaryVersion == 87 &&
      properties.multiProcessorCount ==
          static_cast<int>(kSm87P40PhaseBf16SmCount) &&
      attributes.numRegs ==
          kSm87P40PhaseBf16ExpectedRegistersPerThread &&
      attributes.sharedSizeBytes == 0U && attributes.localSizeBytes == 0U &&
      active_blocks == kSm87P40PhaseBf16ExpectedActiveBlocksPerSm &&
      properties.warpSize == static_cast<int>(kWarpSize) &&
      properties.maxThreadsPerBlock >= static_cast<int>(kThreads) &&
      properties.sharedMemPerBlockOptin >=
          kSm87P40PhaseBf16DynamicSharedBytes;
  return resources->valid() ? static_cast<int>(cudaSuccess)
                            : static_cast<int>(cudaErrorNotSupported);
}

int launch_sm87_p40_phase_bf16_projection_cuda(
    const Sm87P40PhaseBf16ProjectionRole role,
    const std::uint16_t* const activations,
    const std::uint16_t* const weights, const std::size_t token_count,
    const float alpha, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const auto plan =
      make_sm87_p40_phase_bf16_projection_plan(role, token_count);
  if (!plan.valid() || !pointer_is_aligned(activations, 16U) ||
      !pointer_is_aligned(weights, 16U) ||
      !pointer_is_aligned(output, 4U) || !std::isfinite(alpha) ||
      alpha < 0.0F) {
    return invalid_value();
  }
  const std::size_t activation_bytes =
      plan.token_count * plan.input_features * sizeof(std::uint16_t);
  const std::size_t weight_bytes =
      plan.output_features * plan.input_features * sizeof(std::uint16_t);
  const std::size_t output_bytes =
      plan.token_count * plan.output_features * sizeof(std::uint16_t);
  if (byte_ranges_overlap(output, output_bytes, activations,
                          activation_bytes) ||
      byte_ranges_overlap(output, output_bytes, weights, weight_bytes)) {
    return invalid_value();
  }

  Sm87P40PhaseBf16ProjectionResources resources{};
  int status = query_sm87_p40_phase_bf16_projection_resources_cuda(
      role, token_count, &resources);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  return static_cast<int>(launch_role_kernel(
      plan, activations, weights, alpha, output,
      static_cast<cudaStream_t>(cuda_stream)));
}

}  // namespace q3x::kernels
