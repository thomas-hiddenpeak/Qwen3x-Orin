#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_bf16_ab_prefill.h"

#include "q3x/kernels/reference_gemv.h"

#include "sm87_target_aot_bf16_ab_launch_internal.h"
#endif
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION)
#include "q3x/kernels/sm87_macrofeed_v4_bf16_ab.h"
#endif

#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
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
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
static_assert(kThreads == kSm87Bf16AbPromptWideP40Threads);
static_assert(kRowsPerProjection ==
              kSm87Bf16AbPromptWideP40RowsPerProjection);
static_assert(kColumns == kSm87Bf16AbPromptWideP40InputFeatures);
static_assert(kTokenTile == kSm87Bf16AbPromptWideP40TileTokens);
static_assert(kDynamicSharedBytes ==
              kSm87Bf16AbPromptWideP40DynamicSharedBytes);
#endif
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION)
static_assert(kThreads == kSm87MacroFeedV4Bf16AbThreads);
static_assert(kRowsPerProjection ==
              kSm87MacroFeedV4Bf16AbRowsPerProjection);
static_assert(kLogicalRows == kSm87MacroFeedV4Bf16AbLogicalRows);
static_assert(kColumns == kSm87MacroFeedV4Bf16AbInputFeatures);
static_assert(kTokenTile == kSm87MacroFeedV4Bf16AbTileTokens);
static_assert(kDynamicSharedBytes ==
              kSm87MacroFeedV4Bf16AbDynamicSharedBytes);
#endif

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

#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
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
#endif

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
    std::uint16_t* const second_output,
    const unsigned int output_row_stride) {
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
        static_cast<std::size_t>(token0) * output_row_stride +
        output_column) =
        pack_bf16_pair(accumulators[panel].x0, accumulators[panel].x1);
    *reinterpret_cast<std::uint32_t*>(
        selected_output +
        static_cast<std::size_t>(token1) * output_row_stride +
        output_column) =
        pack_bf16_pair(accumulators[panel].x2, accumulators[panel].x3);
  }
}

[[nodiscard]] int invalid_value() noexcept {
  return static_cast<int>(cudaErrorInvalidValue);
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION)
[[nodiscard]] bool macrofeed_v4_bf16_ab_structural_arguments_valid(
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const input,
    const std::size_t token_count,
    std::uint16_t* const scratch,
    const std::size_t scratch_row_stride,
    const void* const cuda_stream,
    const bool production_extent) noexcept {
  if ((production_extent &&
       token_count != kSm87MacroFeedV4Bf16AbTokens) ||
      (!production_extent &&
       token_count != kSm87MacroFeedV4Bf16AbOracleTokens) ||
      scratch_row_stride != kSm87MacroFeedV4Bf16AbScratchRowStride ||
      cuda_stream == nullptr ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(a_weights) ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(b_weights) ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(input) ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(scratch)) {
    return false;
  }

  const std::uint64_t input_bytes =
      token_count * kSm87MacroFeedV4Bf16AbInputFeatures *
      sizeof(std::uint16_t);
  const std::uint64_t scratch_bytes =
      token_count * scratch_row_stride * sizeof(std::uint16_t);
  const Sm87MacroFeedV4Bf16AbByteRange ranges[] = {
      sm87_macrofeed_v4_bf16_ab_byte_range(
          a_weights, kSm87MacroFeedV4Bf16AbWeightBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(
          b_weights, kSm87MacroFeedV4Bf16AbWeightBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(input, input_bytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(scratch, scratch_bytes),
  };
  for (std::size_t left = 0U; left < 4U; ++left) {
    if (!ranges[left].valid) {
      return false;
    }
    for (std::size_t right = left + 1U; right < 4U; ++right) {
      if (!sm87_macrofeed_v4_bf16_ab_ranges_disjoint(
              ranges[left], ranges[right])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] bool macrofeed_v4_bf16_ab_compact_structural_arguments_valid(
    const Sm87MacroFeedV4Bf16AbCompactOracleArguments& arguments) noexcept {
  if (arguments.token_count != kSm87MacroFeedV4Bf16AbOracleTokens ||
      arguments.cuda_stream == nullptr ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(arguments.a_weights) ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(arguments.b_weights) ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(arguments.input) ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(arguments.a_output) ||
      !sm87_macrofeed_v4_bf16_ab_pointer_aligned(arguments.b_output)) {
    return false;
  }
  constexpr std::uint64_t kInputBytes =
      kSm87MacroFeedV4Bf16AbOracleTokens *
      kSm87MacroFeedV4Bf16AbInputFeatures * sizeof(std::uint16_t);
  constexpr std::uint64_t kOutputBytes =
      kSm87MacroFeedV4Bf16AbOracleTokens *
      kSm87MacroFeedV4Bf16AbRowsPerProjection * sizeof(std::uint16_t);
  const Sm87MacroFeedV4Bf16AbByteRange ranges[] = {
      sm87_macrofeed_v4_bf16_ab_byte_range(
          arguments.a_weights, kSm87MacroFeedV4Bf16AbWeightBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(
          arguments.b_weights, kSm87MacroFeedV4Bf16AbWeightBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(arguments.input, kInputBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(arguments.a_output,
                                           kOutputBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(arguments.b_output,
                                           kOutputBytes),
  };
  for (std::size_t left = 0U; left < 5U; ++left) {
    if (!ranges[left].valid) {
      return false;
    }
    for (std::size_t right = left + 1U; right < 5U; ++right) {
      if (!sm87_macrofeed_v4_bf16_ab_ranges_disjoint(
              ranges[left], ranges[right])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] cudaError_t macrofeed_v4_bf16_ab_validate_device(
    int* const device_ordinal, cudaDeviceProp* const properties) noexcept {
  if (device_ordinal == nullptr || properties == nullptr) {
    return cudaErrorInvalidValue;
  }
  *device_ordinal = -1;
  *properties = {};
  cudaError_t status = cudaGetDevice(device_ordinal);
  if (status != cudaSuccess || *device_ordinal < 0) {
    return status == cudaSuccess ? cudaErrorInvalidDevice : status;
  }
  status = cudaGetDeviceProperties(properties, *device_ordinal);
  if (status != cudaSuccess) {
    return status;
  }
  return properties->major == 8 && properties->minor == 7 &&
                 properties->multiProcessorCount ==
                     static_cast<int>(kSm87MacroFeedV4Bf16AbSmCount)
             ? cudaSuccess
             : cudaErrorNotSupported;
}

[[nodiscard]] bool macrofeed_v4_bf16_ab_device_allocation_range_owned(
    const Sm87MacroFeedV4Bf16AbByteRange& range,
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
  if (cuMemGetAddressRange(
          &allocation_base, &allocation_bytes,
          static_cast<CUdeviceptr>(range.begin)) != CUDA_SUCCESS ||
      allocation_base == 0U || allocation_bytes == 0U) {
    return false;
  }
  const auto allocation_begin =
      static_cast<std::uintptr_t>(allocation_base);
  if (allocation_bytes >
      std::numeric_limits<std::uintptr_t>::max() - allocation_begin) {
    return false;
  }
  const auto allocation_end = allocation_begin + allocation_bytes;
  return range.begin >= allocation_begin && range.end <= allocation_end;
}

template <class Arguments>
[[nodiscard]] bool macrofeed_v4_bf16_ab_device_ranges_owned(
    const Arguments& arguments, const int device_ordinal) noexcept {
  const std::uint64_t input_bytes =
      arguments.token_count * kSm87MacroFeedV4Bf16AbInputFeatures *
      sizeof(std::uint16_t);
  const std::uint64_t scratch_bytes =
      arguments.token_count * arguments.scratch_row_stride *
      sizeof(std::uint16_t);
  const Sm87MacroFeedV4Bf16AbByteRange ranges[] = {
      sm87_macrofeed_v4_bf16_ab_byte_range(
          arguments.a_weights, kSm87MacroFeedV4Bf16AbWeightBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(
          arguments.b_weights, kSm87MacroFeedV4Bf16AbWeightBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(arguments.input, input_bytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(arguments.scratch,
                                           scratch_bytes),
  };
  for (const auto& range : ranges) {
    if (!macrofeed_v4_bf16_ab_device_allocation_range_owned(
            range, device_ordinal)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool macrofeed_v4_bf16_ab_compact_device_ranges_owned(
    const Sm87MacroFeedV4Bf16AbCompactOracleArguments& arguments,
    const int device_ordinal) noexcept {
  constexpr std::uint64_t kInputBytes =
      kSm87MacroFeedV4Bf16AbOracleTokens *
      kSm87MacroFeedV4Bf16AbInputFeatures * sizeof(std::uint16_t);
  constexpr std::uint64_t kOutputBytes =
      kSm87MacroFeedV4Bf16AbOracleTokens *
      kSm87MacroFeedV4Bf16AbRowsPerProjection * sizeof(std::uint16_t);
  const Sm87MacroFeedV4Bf16AbByteRange ranges[] = {
      sm87_macrofeed_v4_bf16_ab_byte_range(
          arguments.a_weights, kSm87MacroFeedV4Bf16AbWeightBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(
          arguments.b_weights, kSm87MacroFeedV4Bf16AbWeightBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(arguments.input, kInputBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(arguments.a_output,
                                           kOutputBytes),
      sm87_macrofeed_v4_bf16_ab_byte_range(arguments.b_output,
                                           kOutputBytes),
  };
  for (const auto& range : ranges) {
    if (!macrofeed_v4_bf16_ab_device_allocation_range_owned(
            range, device_ordinal)) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool macrofeed_v4_bf16_ab_resource_equal(
    const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& left,
    const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& right) noexcept {
  return left.identity == right.identity &&
         left.device_ordinal == right.device_ordinal &&
         left.compute_major == right.compute_major &&
         left.compute_minor == right.compute_minor &&
         left.sm_count == right.sm_count &&
         left.binary_version == right.binary_version &&
         left.registers_per_thread == right.registers_per_thread &&
         left.static_shared_bytes == right.static_shared_bytes &&
         left.dynamic_shared_bytes == right.dynamic_shared_bytes &&
         left.local_bytes == right.local_bytes &&
         left.maximum_threads_per_block == right.maximum_threads_per_block &&
         left.active_blocks_per_sm == right.active_blocks_per_sm &&
         left.threads_per_block == right.threads_per_block &&
         left.physical_grid_ctas == right.physical_grid_ctas &&
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

[[nodiscard]] cudaError_t macrofeed_v4_bf16_ab_enqueue(
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const input,
    std::uint16_t* const scratch,
    const unsigned int blocks,
    const cudaStream_t stream) noexcept {
  bf16_ab_prefill_m64_n96_k64_kernel<<<
      blocks, kThreads, kDynamicSharedBytes, stream>>>(
      a_weights, b_weights, input,
      scratch + kSm87MacroFeedV4Bf16AbAOffset,
      scratch + kSm87MacroFeedV4Bf16AbBOffset,
      static_cast<unsigned int>(
          kSm87MacroFeedV4Bf16AbScratchRowStride));
  return cudaPeekAtLastError();
}

[[nodiscard]] cudaError_t macrofeed_v4_bf16_ab_enqueue_compact(
    const Sm87MacroFeedV4Bf16AbCompactOracleArguments& arguments,
    const cudaStream_t stream) noexcept {
  bf16_ab_prefill_m64_n96_k64_kernel<<<
      1U, kThreads, kDynamicSharedBytes, stream>>>(
      arguments.a_weights, arguments.b_weights, arguments.input,
      arguments.a_output, arguments.b_output,
      static_cast<unsigned int>(
          kSm87MacroFeedV4Bf16AbRowsPerProjection));
  return cudaPeekAtLastError();
}
#endif

}  // namespace

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_BF16_AB_ADMISSION)
bool sm87_macrofeed_v4_bf16_ab_arguments_valid(
    const Sm87MacroFeedV4Bf16AbArguments& arguments) noexcept {
  return sm87_macrofeed_v4_bf16_ab_plan(
             arguments.token_count, arguments.scratch_row_stride)
             .valid() &&
         macrofeed_v4_bf16_ab_structural_arguments_valid(
             arguments.a_weights, arguments.b_weights, arguments.input,
             arguments.token_count, arguments.scratch,
             arguments.scratch_row_stride, arguments.cuda_stream, true);
}

int query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
    Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot* const
        resources) noexcept {
  if (resources == nullptr) {
    return invalid_value();
  }
  *resources = {};

  int device = -1;
  cudaDeviceProp properties{};
  cudaError_t status =
      macrofeed_v4_bf16_ab_validate_device(&device, &properties);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = cudaFuncSetAttribute(
      bf16_ab_prefill_m64_n96_k64_kernel,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSm87MacroFeedV4Bf16AbDynamicSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
      &attributes, bf16_ab_prefill_m64_n96_k64_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, bf16_ab_prefill_m64_n96_k64_kernel,
      static_cast<int>(kSm87MacroFeedV4Bf16AbThreads),
      kSm87MacroFeedV4Bf16AbDynamicSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  resources->identity = kSm87MacroFeedV4Bf16AbIdentity;
  resources->device_ordinal = device;
  resources->compute_major = properties.major;
  resources->compute_minor = properties.minor;
  resources->sm_count = properties.multiProcessorCount;
  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes =
      kSm87MacroFeedV4Bf16AbDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->maximum_threads_per_block = attributes.maxThreadsPerBlock;
  resources->active_blocks_per_sm = active_blocks;
  resources->threads_per_block =
      static_cast<std::int32_t>(kSm87MacroFeedV4Bf16AbThreads);
  resources->physical_grid_ctas =
      static_cast<std::int32_t>(kSm87MacroFeedV4Bf16AbGridCtas);
  resources->kernel_compiled = true;
  resources->exact_geometry = true;
  resources->startup_package_unbound = true;
  resources->execution_capability = false;
  resources->caller_snapshot_grants_production_authority = false;
  resources->static_resource_gate_passed = true;
  resources->static_resource_gate_passed =
      sm87_macrofeed_v4_bf16_ab_admission_resource_gate(*resources);
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
    const Sm87MacroFeedV4Bf16AbArguments& arguments,
    const Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot& resources,
    Sm87MacroFeedV4Bf16AbAdmissionLaunchReceipt* const receipt) noexcept {
  if (receipt == nullptr) {
    return invalid_value();
  }
  *receipt = {};
  if (!sm87_macrofeed_v4_bf16_ab_arguments_valid(arguments)) {
    return invalid_value();
  }
  if (!sm87_macrofeed_v4_bf16_ab_admission_resource_gate(resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot observed{};
  const int query_status =
      query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
          &observed);
  if (query_status != static_cast<int>(cudaSuccess)) {
    return query_status;
  }
  if (resources.device_ordinal != observed.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!macrofeed_v4_bf16_ab_resource_equal(resources, observed)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  const auto stream =
      reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  int stream_device = -1;
  cudaError_t status = cudaStreamGetDevice(stream, &stream_device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (stream_device != observed.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!macrofeed_v4_bf16_ab_device_ranges_owned(
          arguments, observed.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }

  (void)cudaGetLastError();
  status = macrofeed_v4_bf16_ab_enqueue(
      arguments.a_weights, arguments.b_weights, arguments.input,
      arguments.scratch,
      static_cast<unsigned int>(kSm87MacroFeedV4Bf16AbGridCtas), stream);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  receipt->identity = kSm87MacroFeedV4Bf16AbIdentity;
  receipt->token_count = arguments.token_count;
  receipt->scratch_row_stride = arguments.scratch_row_stride;
  receipt->a_offset = kSm87MacroFeedV4Bf16AbAOffset;
  receipt->b_offset = kSm87MacroFeedV4Bf16AbBOffset;
  receipt->physical_kernel_launches = static_cast<std::uint32_t>(
      kSm87MacroFeedV4Bf16AbPhysicalKernelLaunches);
  receipt->established_exact_body_reused = true;
  receipt->shared_input_dual_projection = true;
  receipt->direct_scratch_scatter = true;
  receipt->compact_bridge_absent = true;
  receipt->current_device_revalidated = true;
  receipt->caller_snapshot_exact_observed_match = true;
  receipt->caller_supplied_live_stream_required = true;
  receipt->live_stream_device_observed = true;
  receipt->device_allocation_ranges_owned = true;
  receipt->launch_enqueued = true;
  receipt->completion_observed = false;
  receipt->numerical_contract_qualified = false;
  receipt->production_dispatch_eligible = false;
  receipt->startup_package_unbound = true;
  receipt->execution_capability = false;
  receipt->caller_snapshot_grants_production_authority = false;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_macrofeed_v4_bf16_ab_oracle_cuda(
    const Sm87MacroFeedV4Bf16AbOracleArguments& arguments) noexcept {
  if (!macrofeed_v4_bf16_ab_structural_arguments_valid(
          arguments.a_weights, arguments.b_weights, arguments.input,
          arguments.token_count, arguments.scratch,
          arguments.scratch_row_stride, arguments.cuda_stream, false)) {
    return invalid_value();
  }
  Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot resources{};
  const int query_status =
      query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
          &resources);
  if (query_status != static_cast<int>(cudaSuccess)) {
    return query_status;
  }
  if (!sm87_macrofeed_v4_bf16_ab_admission_resource_gate(resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  const auto stream =
      reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  int stream_device = -1;
  cudaError_t status = cudaStreamGetDevice(stream, &stream_device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (stream_device != resources.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!macrofeed_v4_bf16_ab_device_ranges_owned(
          arguments, resources.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  (void)cudaGetLastError();
  return static_cast<int>(macrofeed_v4_bf16_ab_enqueue(
      arguments.a_weights, arguments.b_weights, arguments.input,
      arguments.scratch, 1U, stream));
}

int launch_sm87_macrofeed_v4_bf16_ab_compact_oracle_cuda(
    const Sm87MacroFeedV4Bf16AbCompactOracleArguments& arguments) noexcept {
  if (!macrofeed_v4_bf16_ab_compact_structural_arguments_valid(arguments)) {
    return invalid_value();
  }
  Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot resources{};
  const int query_status =
      query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
          &resources);
  if (query_status != static_cast<int>(cudaSuccess)) {
    return query_status;
  }
  if (!sm87_macrofeed_v4_bf16_ab_admission_resource_gate(resources)) {
    return static_cast<int>(cudaErrorLaunchOutOfResources);
  }
  const auto stream =
      reinterpret_cast<cudaStream_t>(arguments.cuda_stream);
  int stream_device = -1;
  cudaError_t status = cudaStreamGetDevice(stream, &stream_device);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  if (stream_device != resources.device_ordinal) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }
  if (!macrofeed_v4_bf16_ab_compact_device_ranges_owned(
          arguments, resources.device_ordinal)) {
    return static_cast<int>(cudaErrorInvalidDevicePointer);
  }
  (void)cudaGetLastError();
  return static_cast<int>(
      macrofeed_v4_bf16_ab_enqueue_compact(arguments, stream));
}
#endif

#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
int query_sm87_bf16_ab_prompt_wide_p40_resources_cuda(
    Sm87Bf16AbPromptWideP40Resources* const resources) noexcept {
  if (resources == nullptr) {
    return invalid_value();
  }
  *resources = {};

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
  if (properties.major != 8 || properties.minor != 7) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(
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

  resources->binary_version = attributes.binaryVersion;
  resources->registers_per_thread = attributes.numRegs;
  resources->static_shared_bytes = attributes.sharedSizeBytes;
  resources->dynamic_shared_bytes = kDynamicSharedBytes;
  resources->local_bytes = attributes.localSizeBytes;
  resources->active_blocks_per_sm = active_blocks;

  const bool supported =
      properties.major == 8 && properties.minor == 7 &&
      properties.warpSize == static_cast<int>(kWarpSize) &&
      properties.maxThreadsPerBlock >= static_cast<int>(kThreads) &&
      properties.maxThreadsDim[0] >= static_cast<int>(kThreads) &&
      properties.maxGridSize[0] >=
          static_cast<int>(kSm87Bf16AbPromptWideP40GridBlocks) &&
      properties.sharedMemPerBlockOptin >= kDynamicSharedBytes &&
      attributes.binaryVersion == 87 &&
      attributes.maxThreadsPerBlock >= static_cast<int>(kThreads) &&
      attributes.numRegs > 0 && active_blocks >= 2;
  if (!supported) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  resources->admitted = true;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_bf16_ab_prompt_wide_p40_cuda(
    const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input,
    const std::size_t token_count,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output,
    void* const cuda_stream) noexcept {
  const Sm87Bf16AbPromptWideP40Plan plan =
      make_sm87_bf16_ab_prompt_wide_p40_plan(token_count);
  if (!plan.valid() ||
      !pointer_is_aligned(first_weights, alignof(uint4)) ||
      !pointer_is_aligned(second_weights, alignof(uint4)) ||
      !pointer_is_aligned(input, alignof(uint4)) ||
      !pointer_is_aligned(first_output, alignof(std::uint32_t)) ||
      !pointer_is_aligned(second_output, alignof(std::uint32_t))) {
    return invalid_value();
  }

  constexpr std::size_t kWeightBytes =
      kSm87Bf16AbPromptWideP40WeightElements * sizeof(std::uint16_t);
  constexpr std::size_t kInputBytes =
      kSm87Bf16AbPromptWideP40InputElements * sizeof(std::uint16_t);
  constexpr std::size_t kOutputBytes =
      kSm87Bf16AbPromptWideP40OutputElements * sizeof(std::uint16_t);
  if (byte_range_overflows(first_weights, kWeightBytes) ||
      byte_range_overflows(second_weights, kWeightBytes) ||
      byte_range_overflows(input, kInputBytes) ||
      byte_range_overflows(first_output, kOutputBytes) ||
      byte_range_overflows(second_output, kOutputBytes) ||
      byte_ranges_overlap(first_output, kOutputBytes,
                          first_weights, kWeightBytes) ||
      byte_ranges_overlap(first_output, kOutputBytes,
                          second_weights, kWeightBytes) ||
      byte_ranges_overlap(first_output, kOutputBytes, input, kInputBytes) ||
      byte_ranges_overlap(second_output, kOutputBytes,
                          first_weights, kWeightBytes) ||
      byte_ranges_overlap(second_output, kOutputBytes,
                          second_weights, kWeightBytes) ||
      byte_ranges_overlap(second_output, kOutputBytes, input, kInputBytes) ||
      byte_ranges_overlap(first_output, kOutputBytes,
                          second_output, kOutputBytes)) {
    return invalid_value();
  }

  Sm87Bf16AbPromptWideP40Resources resources{};
  const int capability_status =
      query_sm87_bf16_ab_prompt_wide_p40_resources_cuda(&resources);
  if (capability_status != static_cast<int>(cudaSuccess) ||
      !resources.valid()) {
    return capability_status == static_cast<int>(cudaSuccess)
               ? static_cast<int>(cudaErrorNotSupported)
               : capability_status;
  }

  (void)cudaGetLastError();
  bf16_ab_prefill_m64_n96_k64_kernel
      <<<static_cast<unsigned int>(plan.grid_blocks), kThreads,
         kDynamicSharedBytes, static_cast<cudaStream_t>(cuda_stream)>>>(
          first_weights, second_weights, input,
          first_output, second_output, kRowsPerProjection);
  return static_cast<int>(cudaGetLastError());
}

namespace sm87_target_aot_bf16_ab_execution_detail {

namespace {

constexpr std::uint64_t kInterleavedP40SealTag =
    0x5142'4132'5034'3041ULL;

[[nodiscard]] bool exact_device_pointer(const void* const pointer,
                                        const int device) noexcept {
  cudaPointerAttributes attributes{};
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  return status == cudaSuccess && attributes.type == cudaMemoryTypeDevice &&
         attributes.device == device;
}

[[nodiscard]] bool exact_device_allocation_range(
    const void* const pointer, const std::size_t bytes,
    const int device) noexcept {
  if (byte_range_overflows(pointer, bytes) || bytes == 0U) {
    return false;
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  const auto end = begin + bytes;
  const std::array<const void*, 2U> endpoints{{
      pointer, reinterpret_cast<const void*>(end - 1U),
  }};
  std::array<CUdeviceptr, 2U> bases{{0U, 0U}};
  std::array<std::size_t, 2U> allocation_bytes{{0U, 0U}};
  for (std::size_t index = 0U; index < endpoints.size(); ++index) {
    if (!exact_device_pointer(endpoints[index], device) ||
        cuMemGetAddressRange(
            &bases[index], &allocation_bytes[index],
            static_cast<CUdeviceptr>(
                reinterpret_cast<std::uintptr_t>(endpoints[index]))) !=
            CUDA_SUCCESS ||
        bases[index] == 0U || allocation_bytes[index] == 0U) {
      return false;
    }
  }
  if (bases[0U] != bases[1U] ||
      allocation_bytes[0U] != allocation_bytes[1U]) {
    return false;
  }
  const auto allocation_begin =
      static_cast<std::uintptr_t>(bases[0U]);
  if (allocation_bytes[0U] >
      std::numeric_limits<std::uintptr_t>::max() - allocation_begin) {
    return false;
  }
  const auto allocation_end = allocation_begin + allocation_bytes[0U];
  return begin >= allocation_begin && end <= allocation_end;
}

[[nodiscard]] bool binding_valid(
    const InterleavedP40Binding& binding,
    const std::size_t ordinal,
    const int device) noexcept {
  constexpr std::size_t kWeightBytes =
      kSm87Bf16AbPromptWideP40WeightElements * sizeof(std::uint16_t);
  constexpr std::size_t kInputBytes =
      kSm87Bf16AbPromptWideP40InputElements * sizeof(std::uint16_t);
  constexpr std::size_t kInterleavedBytes =
      kSm87Bf16AbPromptWideP40Tokens * kLogicalRows *
      sizeof(std::uint16_t);
  if (binding.layer >= 64U || (binding.layer + 1U) % 4U == 0U ||
      binding.layer - binding.layer / 4U != ordinal ||
      !pointer_is_aligned(binding.a_weights, alignof(uint4)) ||
      !pointer_is_aligned(binding.b_weights, alignof(uint4)) ||
      !pointer_is_aligned(binding.input, alignof(uint4)) ||
      !pointer_is_aligned(binding.interleaved_ab_output,
                          alignof(std::uint32_t)) ||
      byte_range_overflows(binding.a_weights, kWeightBytes) ||
      byte_range_overflows(binding.b_weights, kWeightBytes) ||
      byte_range_overflows(binding.input, kInputBytes) ||
      byte_range_overflows(binding.interleaved_ab_output,
                           kInterleavedBytes) ||
      byte_ranges_overlap(binding.interleaved_ab_output,
                          kInterleavedBytes, binding.a_weights,
                          kWeightBytes) ||
      byte_ranges_overlap(binding.interleaved_ab_output,
                          kInterleavedBytes, binding.b_weights,
                          kWeightBytes) ||
      byte_ranges_overlap(binding.interleaved_ab_output,
                          kInterleavedBytes, binding.input,
                          kInputBytes)) {
    return false;
  }
  return exact_device_allocation_range(binding.a_weights, kWeightBytes,
                                       device) &&
         exact_device_allocation_range(binding.b_weights, kWeightBytes,
                                       device) &&
         exact_device_allocation_range(binding.input, kInputBytes, device) &&
         exact_device_allocation_range(binding.interleaved_ab_output,
                                       kInterleavedBytes, device);
}

}  // namespace

bool SealedInterleavedP40Access::valid() const noexcept {
  return device_ordinal_ >= 0 && cuda_stream_ != nullptr &&
         seal_nonce_ != 0U;
}

SealedInterleavedP40Result seal_interleaved_p40(
    const InterleavedP40Binding* const bindings,
    const std::size_t binding_count, void* const cuda_stream) noexcept {
  SealedInterleavedP40Result result;
  if (bindings == nullptr ||
      binding_count != kInterleavedP40LayerBindings ||
      cuda_stream == nullptr) {
    result.cuda_error = static_cast<int>(cudaErrorInvalidValue);
    return result;
  }
  int device = -1;
  cudaError_t status = cudaGetDevice(&device);
  if (status != cudaSuccess || device < 0) {
    result.cuda_error = static_cast<int>(status);
    return result;
  }
  unsigned int stream_flags = 0U;
  status = cudaStreamGetFlags(static_cast<cudaStream_t>(cuda_stream),
                              &stream_flags);
  if (status != cudaSuccess ||
      (stream_flags & cudaStreamNonBlocking) == 0U) {
    result.cuda_error = static_cast<int>(
        status == cudaSuccess ? cudaErrorInvalidResourceHandle : status);
    return result;
  }
  Sm87Bf16AbPromptWideP40Resources resources{};
  const int resource_status =
      query_sm87_bf16_ab_prompt_wide_p40_resources_cuda(&resources);
  if (resource_status != static_cast<int>(cudaSuccess) ||
      !resources.valid()) {
    result.cuda_error =
        resource_status == static_cast<int>(cudaSuccess)
            ? static_cast<int>(cudaErrorNotSupported)
            : resource_status;
    return result;
  }
  std::array<InterleavedP40Binding, kInterleavedP40LayerBindings> sealed{};
  std::uint64_t nonce = kInterleavedP40SealTag ^
                        static_cast<std::uint64_t>(device + 1);
  for (std::size_t ordinal = 0U; ordinal < sealed.size(); ++ordinal) {
    if (!binding_valid(bindings[ordinal], ordinal, device)) {
      result.cuda_error = static_cast<int>(cudaErrorInvalidDevicePointer);
      return result;
    }
    sealed[ordinal] = bindings[ordinal];
    nonce ^= reinterpret_cast<std::uintptr_t>(bindings[ordinal].a_weights) +
             0x9e37'79b9'7f4a'7c15ULL + (nonce << 6U) + (nonce >> 2U);
    nonce ^= reinterpret_cast<std::uintptr_t>(bindings[ordinal].b_weights) +
             0x9e37'79b9'7f4a'7c15ULL + (nonce << 6U) + (nonce >> 2U);
  }
  if (nonce == 0U) {
    nonce = kInterleavedP40SealTag;
  }
  result.access =
      SealedInterleavedP40Access(sealed, device, cuda_stream, nonce);
  return result;
}

int enqueue_interleaved_p40_prevalidated(
    const SealedInterleavedP40Access& access,
    const std::size_t gdn_ordinal,
    std::size_t* const submitted_launches) noexcept {
  if (submitted_launches == nullptr) {
    return invalid_value();
  }
  *submitted_launches = 0U;
  if (!access.valid() || gdn_ordinal >= access.bindings_.size()) {
    return invalid_value();
  }
  const auto& binding = access.bindings_[gdn_ordinal];
  bf16_ab_prefill_m64_n96_k64_kernel
      <<<kSm87Bf16AbPromptWideP40GridBlocks, kThreads,
         kDynamicSharedBytes,
         static_cast<cudaStream_t>(access.cuda_stream_)>>>(
          binding.a_weights, binding.b_weights, binding.input,
          binding.interleaved_ab_output,
          binding.interleaved_ab_output + kRowsPerProjection,
          kLogicalRows);
  const cudaError_t status = cudaPeekAtLastError();
  if (status == cudaSuccess) {
    *submitted_launches = 1U;
  }
  return static_cast<int>(status);
}

int launch_interleaved_p40(
    const std::uint16_t* const a_weights,
    const std::uint16_t* const b_weights,
    const std::uint16_t* const input,
    std::uint16_t* const interleaved_ab_output,
    void* const cuda_stream) noexcept {
  if (!pointer_is_aligned(a_weights, alignof(uint4)) ||
      !pointer_is_aligned(b_weights, alignof(uint4)) ||
      !pointer_is_aligned(input, alignof(uint4)) ||
      !pointer_is_aligned(interleaved_ab_output, alignof(std::uint32_t))) {
    return invalid_value();
  }
  constexpr std::size_t kWeightBytes =
      kSm87Bf16AbPromptWideP40WeightElements * sizeof(std::uint16_t);
  constexpr std::size_t kInputBytes =
      kSm87Bf16AbPromptWideP40InputElements * sizeof(std::uint16_t);
  constexpr std::size_t kInterleavedBytes =
      kSm87Bf16AbPromptWideP40Tokens * kLogicalRows *
      sizeof(std::uint16_t);
  if (byte_range_overflows(a_weights, kWeightBytes) ||
      byte_range_overflows(b_weights, kWeightBytes) ||
      byte_range_overflows(input, kInputBytes) ||
      byte_range_overflows(interleaved_ab_output, kInterleavedBytes) ||
      byte_ranges_overlap(interleaved_ab_output, kInterleavedBytes,
                          a_weights, kWeightBytes) ||
      byte_ranges_overlap(interleaved_ab_output, kInterleavedBytes,
                          b_weights, kWeightBytes) ||
      byte_ranges_overlap(interleaved_ab_output, kInterleavedBytes,
                          input, kInputBytes)) {
    return invalid_value();
  }
  Sm87Bf16AbPromptWideP40Resources resources{};
  const int capability_status =
      query_sm87_bf16_ab_prompt_wide_p40_resources_cuda(&resources);
  if (capability_status != static_cast<int>(cudaSuccess) ||
      !resources.valid()) {
    return capability_status == static_cast<int>(cudaSuccess)
               ? static_cast<int>(cudaErrorNotSupported)
               : capability_status;
  }
  const auto stream = static_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  bf16_ab_prefill_m64_n96_k64_kernel
      <<<kSm87Bf16AbPromptWideP40GridBlocks, kThreads,
         kDynamicSharedBytes, stream>>>(
          a_weights, b_weights, input, interleaved_ab_output,
          interleaved_ab_output + kRowsPerProjection, kLogicalRows);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace sm87_target_aot_bf16_ab_execution_detail

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
            first_output, second_output, kRowsPerProjection);
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
#endif

}  // namespace q3x::kernels
