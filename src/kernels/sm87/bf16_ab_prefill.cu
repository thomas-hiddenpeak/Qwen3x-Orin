#include "q3x/kernels/sm87_bf16_ab_prefill.h"

#include "q3x/kernels/reference_gemv.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarpSize = 32U;
constexpr unsigned int kRowsPerProjection = 48U;
constexpr unsigned int kLogicalRows = 2U * kRowsPerProjection;
constexpr unsigned int kColumns = 5'120U;
constexpr unsigned int kTokenTile = 64U;
constexpr unsigned int kColumnTile = 64U;
constexpr unsigned int kK16PerStage = kColumnTile / 16U;
constexpr unsigned int kStageCount = kColumns / kColumnTile;
constexpr unsigned int kPipelineStages = 2U;
constexpr unsigned int kSharedLeadingDimension = 72U;
constexpr unsigned int kVectorsPerSharedRow =
    kSharedLeadingDimension * sizeof(std::uint16_t) / sizeof(uint4);
constexpr unsigned int kVectorsPerGlobalRow =
    kColumnTile * sizeof(std::uint16_t) / sizeof(uint4);
constexpr unsigned int kActivationVectors =
    kTokenTile * kVectorsPerGlobalRow;
constexpr unsigned int kWeightVectors =
    kLogicalRows * kVectorsPerGlobalRow;

static_assert(kThreads == 8U * kWarpSize);
static_assert(kStageCount == 80U);
static_assert(kK16PerStage == 4U);
static_assert(kVectorsPerSharedRow == 9U);
static_assert(kVectorsPerGlobalRow == 8U);
static_assert(kActivationVectors + kWeightVectors == 5U * kThreads);

struct alignas(32) PipelineStorage final {
  uint4 activations[kPipelineStages][kTokenTile]
                   [kVectorsPerSharedRow];
  uint4 weights[kPipelineStages][kLogicalRows]
               [kVectorsPerSharedRow];
};

constexpr std::size_t kDynamicSharedBytes = sizeof(PipelineStorage);
static_assert(kDynamicSharedBytes == 46'080U);

struct InlineM16K16Activation final {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct InlineK16N8Weight final {
  std::uint32_t x0;
  std::uint32_t x1;
};

struct InlineM16N8Accumulator final {
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
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

__device__ __forceinline__ void cp_async_cg_shared_global_16(
    void* const destination, const void* const source) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :
               : "r"(shared_address), "l"(source)
               : "memory");
#else
  (void)destination;
  (void)source;
#endif
}

__device__ __forceinline__ void cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_1() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 1;" ::: "memory");
#endif
}

__device__ __forceinline__ void cp_async_wait_group_0() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;" ::: "memory");
#endif
}

__device__ __forceinline__ void load_m16k16_activation_fragment(
    InlineM16K16Activation& fragment,
    const std::uint16_t* const shared_activations,
    const unsigned int m_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row = (lane % 8U) + (quadrant & 1U) * 8U;
  const unsigned int column = k16 * 16U + (quadrant >> 1U) * 8U;
  const std::uint16_t* const source =
      shared_activations +
      (m_panel * 16U + row) * kSharedLeadingDimension + column;
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
      "{%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1),
        "=r"(fragment.x2), "=r"(fragment.x3)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_activations;
  (void)m_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void load_k16n8_weight_fragment(
    InlineK16N8Weight& fragment,
    const std::uint16_t* const shared_weights,
    const unsigned int n_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  // The canonical [N,K] row-major weights are exactly the desired logical
  // [K,N] column-major matrix. ldmatrix.trans receives one address per
  // canonical N row for each K8 half and emits the BF16 matrix-B fragment.
  const unsigned int row = lane & 7U;
  const unsigned int column = k16 * 16U + ((lane >> 3U) & 1U) * 8U;
  const std::uint16_t* const source =
      shared_weights +
      (n_panel * 8U + row) * kSharedLeadingDimension + column;
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
      "{%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_weights;
  (void)n_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void mma_m16n8k16_bf16(
    InlineM16N8Accumulator& accumulator,
    const InlineM16K16Activation& activation,
    const InlineK16N8Weight& weight) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(activation.x0), "r"(activation.x1),
        "r"(activation.x2), "r"(activation.x3),
        "r"(weight.x0), "r"(weight.x1));
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
    const float low, const float high) {
  return static_cast<std::uint32_t>(encode_bf16_rne(low)) |
         (static_cast<std::uint32_t>(encode_bf16_rne(high)) << 16U);
}

__device__ __forceinline__ void issue_pipeline_stage(
    PipelineStorage* const pipeline, const unsigned int slot,
    const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input, const unsigned int first_token,
    const unsigned int first_column) {
  const unsigned int thread = threadIdx.x;
#pragma unroll
  for (unsigned int pass = 0U; pass < 2U; ++pass) {
    const unsigned int index = thread + pass * kThreads;
    const unsigned int token = index / kVectorsPerGlobalRow;
    const unsigned int vector = index % kVectorsPerGlobalRow;
    cp_async_cg_shared_global_16(
        &pipeline->activations[slot][token][vector],
        reinterpret_cast<const uint4*>(
            input + static_cast<std::size_t>(first_token + token) * kColumns +
            first_column) + vector);
  }
#pragma unroll
  for (unsigned int pass = 0U; pass < 3U; ++pass) {
    const unsigned int index = thread + pass * kThreads;
    const unsigned int logical_row = index / kVectorsPerGlobalRow;
    const unsigned int vector = index % kVectorsPerGlobalRow;
    const std::uint16_t* const selected_weights =
        logical_row < kRowsPerProjection ? first_weights : second_weights;
    const unsigned int selected_row =
        logical_row < kRowsPerProjection
            ? logical_row
            : logical_row - kRowsPerProjection;
    cp_async_cg_shared_global_16(
        &pipeline->weights[slot][logical_row][vector],
        reinterpret_cast<const uint4*>(
            selected_weights +
            static_cast<std::size_t>(selected_row) * kColumns + first_column) +
            vector);
  }
  cp_async_commit_group();
}

__global__ __launch_bounds__(kThreads, 2)
void bf16_ab_prefill_m64_n96_k64_kernel(
    const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output) {
  extern __shared__ __align__(32) std::uint8_t dynamic_storage[];
  auto* const pipeline =
      reinterpret_cast<PipelineStorage*>(dynamic_storage);
  const unsigned int warp = threadIdx.x / kWarpSize;
  const unsigned int lane = threadIdx.x % kWarpSize;
  const unsigned int m_panel = warp & 3U;
  const unsigned int projection = warp >> 2U;
  const unsigned int first_n_panel = projection * 6U;
  const unsigned int first_token = blockIdx.x * kTokenTile;

  InlineM16N8Accumulator accumulators[6U];
#pragma unroll
  for (unsigned int panel = 0U; panel < 6U; ++panel) {
    accumulators[panel] = InlineM16N8Accumulator{0.0F, 0.0F, 0.0F, 0.0F};
  }

  issue_pipeline_stage(pipeline, 0U, first_weights, second_weights, input,
                       first_token, 0U);
  issue_pipeline_stage(pipeline, 1U, first_weights, second_weights, input,
                       first_token, kColumnTile);

#pragma unroll 1
  for (unsigned int stage = 0U; stage < kStageCount; ++stage) {
    if (stage + 1U < kStageCount) {
      cp_async_wait_group_1();
    } else {
      cp_async_wait_group_0();
    }
    __syncthreads();

    const unsigned int slot = stage & 1U;
    const auto* const shared_activations =
        reinterpret_cast<const std::uint16_t*>(
            pipeline->activations[slot]);
    const auto* const shared_weights =
        reinterpret_cast<const std::uint16_t*>(pipeline->weights[slot]);
#pragma unroll
    for (unsigned int k16 = 0U; k16 < kK16PerStage; ++k16) {
      InlineM16K16Activation activation{};
      load_m16k16_activation_fragment(
          activation, shared_activations, m_panel, k16, lane);
#pragma unroll
      for (unsigned int panel = 0U; panel < 6U; ++panel) {
        InlineK16N8Weight weight{};
        load_k16n8_weight_fragment(
            weight, shared_weights, first_n_panel + panel, k16, lane);
        mma_m16n8k16_bf16(accumulators[panel], activation, weight);
      }
    }
    __syncthreads();

    if (stage + kPipelineStages < kStageCount) {
      const unsigned int future_stage = stage + kPipelineStages;
      issue_pipeline_stage(
          pipeline, slot, first_weights, second_weights, input,
          first_token, future_stage * kColumnTile);
    }
  }
  cp_async_wait_group_0();
  __syncthreads();

  std::uint16_t* const selected_output =
      projection == 0U ? first_output : second_output;
  const unsigned int lane_group = lane / 4U;
  const unsigned int lane_in_group = lane % 4U;
  const unsigned int token0 = first_token + m_panel * 16U + lane_group;
  const unsigned int token1 = token0 + 8U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 6U; ++panel) {
    const unsigned int output_column = panel * 8U + 2U * lane_in_group;
    *reinterpret_cast<std::uint32_t*>(
        selected_output +
        static_cast<std::size_t>(token0) * kRowsPerProjection +
        output_column) =
        pack_bf16_pair(accumulators[panel].x0, accumulators[panel].x1);
    *reinterpret_cast<std::uint32_t*>(
        selected_output +
        static_cast<std::size_t>(token1) * kRowsPerProjection +
        output_column) =
        pack_bf16_pair(accumulators[panel].x2, accumulators[panel].x3);
  }
}

[[nodiscard]] int invalid_value() noexcept {
  return static_cast<int>(cudaErrorInvalidValue);
}

}  // namespace

int launch_sm87_bf16_ab_large_m_prefill_cuda(
    const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input,
    const std::size_t token_count,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output,
    void* const cuda_stream) noexcept {
  if (token_count < 2U || token_count > 512U ||
      !pointer_is_aligned(first_weights, alignof(uint4)) ||
      !pointer_is_aligned(second_weights, alignof(uint4)) ||
      !pointer_is_aligned(input, alignof(uint4)) ||
      !pointer_is_aligned(first_output, alignof(std::uint32_t)) ||
      !pointer_is_aligned(second_output, alignof(std::uint32_t))) {
    return invalid_value();
  }
  constexpr std::size_t kWeightBytes =
      static_cast<std::size_t>(kRowsPerProjection) * kColumns *
      sizeof(std::uint16_t);
  const std::size_t input_bytes =
      token_count * kColumns * sizeof(std::uint16_t);
  const std::size_t output_bytes =
      token_count * kRowsPerProjection * sizeof(std::uint16_t);
  if (byte_range_overflows(first_weights, kWeightBytes) ||
      byte_range_overflows(second_weights, kWeightBytes) ||
      byte_range_overflows(input, input_bytes) ||
      byte_range_overflows(first_output, output_bytes) ||
      byte_range_overflows(second_output, output_bytes) ||
      byte_ranges_overlap(first_output, output_bytes,
                          first_weights, kWeightBytes) ||
      byte_ranges_overlap(first_output, output_bytes,
                          second_weights, kWeightBytes) ||
      byte_ranges_overlap(first_output, output_bytes, input, input_bytes) ||
      byte_ranges_overlap(second_output, output_bytes,
                          first_weights, kWeightBytes) ||
      byte_ranges_overlap(second_output, output_bytes,
                          second_weights, kWeightBytes) ||
      byte_ranges_overlap(second_output, output_bytes, input, input_bytes) ||
      byte_ranges_overlap(first_output, output_bytes,
                          second_output, output_bytes)) {
    return invalid_value();
  }

  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  const std::size_t tensor_core_tokens =
      (token_count / kTokenTile) * kTokenTile;
  (void)cudaGetLastError();
  if (tensor_core_tokens != 0U) {
    cudaError_t status = cudaFuncSetAttribute(
        bf16_ab_prefill_m64_n96_k64_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(kDynamicSharedBytes));
    if (status != cudaSuccess) {
      return static_cast<int>(status);
    }
    const unsigned int blocks =
        static_cast<unsigned int>(tensor_core_tokens / kTokenTile);
    bf16_ab_prefill_m64_n96_k64_kernel
        <<<blocks, kThreads, kDynamicSharedBytes, stream>>>(
            first_weights, second_weights, input,
            first_output, second_output);
    const cudaError_t launch_status = cudaGetLastError();
    if (launch_status != cudaSuccess) {
      return static_cast<int>(launch_status);
    }
  }

  for (std::size_t token_offset = tensor_core_tokens;
       token_offset < token_count; token_offset += 16U) {
    const std::size_t remaining = token_count - token_offset;
    const std::size_t launch_tokens = remaining < 16U ? remaining : 16U;
    const std::uint16_t* const tile_input =
        input + token_offset * kColumns;
    std::uint16_t* const first_tile_output =
        first_output + token_offset * kRowsPerProjection;
    std::uint16_t* const second_tile_output =
        second_output + token_offset * kRowsPerProjection;
    const int status =
        launch_tokens == 16U
            ? launch_bf16_gemv_pair_m16_projection_fused_cuda(
                  first_weights, second_weights, tile_input,
                  first_tile_output, second_tile_output, cuda_stream)
            : launch_bf16_gemv_pair_tile_bf16_cuda(
                  first_weights, second_weights, tile_input,
                  launch_tokens, kRowsPerProjection, kColumns,
                  first_tile_output, second_tile_output, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return static_cast<int>(cudaSuccess);
}

int query_sm87_bf16_ab_large_m_prefill_resources_cuda(
    int* const registers_per_thread,
    std::size_t* const static_shared_bytes,
    std::size_t* const dynamic_shared_bytes,
    std::size_t* const local_bytes,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      dynamic_shared_bytes == nullptr || local_bytes == nullptr ||
      active_blocks_per_sm == nullptr) {
    return invalid_value();
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, bf16_ab_prefill_m64_n96_k64_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = cudaFuncSetAttribute(
      bf16_ab_prefill_m64_n96_k64_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, bf16_ab_prefill_m64_n96_k64_kernel,
      static_cast<int>(kThreads), kDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *dynamic_shared_bytes = kDynamicSharedBytes;
  *local_bytes = attributes.localSizeBytes;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
