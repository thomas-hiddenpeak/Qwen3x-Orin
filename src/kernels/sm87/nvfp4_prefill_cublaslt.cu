#include "q3x/kernels/sm87_nvfp4_prefill_cublaslt.h"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>

namespace q3x::kernels {

struct Sm87Nvfp4PrefillCublasLtContext {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatrixLayout_t activation_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  int device = -1;
  int heuristic_rank = -1;
};

struct Sm87Nvfp4PrefillDownCublasLtContext {
  cublasLtHandle_t handle = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t weight_layout = nullptr;
  cublasLtMatrixLayout_t activation_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  int device = -1;
  int heuristic_rank = -1;
};

namespace {

constexpr std::size_t kTokens = 512U;
constexpr std::size_t kRows = 17'408U;
constexpr std::size_t kColumns = 5'120U;
constexpr std::size_t kPackedBytes = kRows * (kColumns / 2U);
constexpr std::size_t kScaleBytes = kRows * (kColumns / 16U);
constexpr std::size_t kActivationBytes =
    kTokens * kColumns * sizeof(std::uint16_t);
constexpr std::size_t kOutputBytes =
    kTokens * kRows * sizeof(std::uint16_t);
constexpr std::size_t kScratchBytes =
    kRows * kColumns * sizeof(std::uint16_t);
constexpr std::size_t kWorkspaceBytes = 0U;
constexpr int kMaximumHeuristics = 16;
constexpr unsigned int kThreads = 256U;
constexpr unsigned int kPackedPerRow = kColumns / 2U;
constexpr unsigned int kScalesPerRow = kColumns / 16U;
constexpr unsigned int kPasses = kPackedPerRow / kThreads;

constexpr std::size_t kDownTokens = 512U;
constexpr std::size_t kDownRows = 5'120U;
constexpr std::size_t kDownColumns = 17'408U;
constexpr std::size_t kDownPackedBytes =
    kDownRows * (kDownColumns / 2U);
constexpr std::size_t kDownScaleBytes =
    kDownRows * (kDownColumns / 16U);
constexpr std::size_t kDownActivationBytes =
    kDownTokens * kDownColumns * sizeof(std::uint16_t);
constexpr std::size_t kDownOutputBytes =
    kDownTokens * kDownRows * sizeof(std::uint16_t);
constexpr std::size_t kDownScratchBytes =
    kDownRows * kDownColumns * sizeof(std::uint16_t);
constexpr unsigned int kDownPackedPerRow = kDownColumns / 2U;
constexpr unsigned int kDownScalesPerRow = kDownColumns / 16U;
constexpr unsigned int kDownPasses = kDownPackedPerRow / kThreads;

static_assert(kPackedBytes == 44'564'480U);
static_assert(kScaleBytes == 5'570'560U);
static_assert(kActivationBytes == 5'242'880U);
static_assert(kOutputBytes == 17'825'792U);
static_assert(kScratchBytes == 178'257'920U);
static_assert(kScratchBytes == 170U * 1024U * 1024U);
static_assert(kPackedPerRow == kPasses * kThreads);
static_assert(kDownPackedBytes == 44'564'480U);
static_assert(kDownScaleBytes == 5'570'560U);
static_assert(kDownActivationBytes == 17'825'792U);
static_assert(kDownOutputBytes == 5'242'880U);
static_assert(kDownScratchBytes == 178'257'920U);
static_assert(kDownScratchBytes == kScratchBytes);
static_assert(kDownPasses == 34U);
static_assert(kDownPackedPerRow == kDownPasses * kThreads);

struct AddressSpan {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
};

[[nodiscard]] bool make_span(const void* const pointer,
                             const std::size_t bytes,
                             AddressSpan* const span) noexcept {
  if (pointer == nullptr || span == nullptr) {
    return false;
  }
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }
  span->begin = begin;
  span->end = begin + bytes;
  return true;
}

[[nodiscard]] bool overlaps(const AddressSpan left,
                            const AddressSpan right) noexcept {
  return left.begin < right.end && right.begin < left.end;
}

[[nodiscard]] bool aligned(const void* const pointer,
                           const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
}

[[nodiscard]] int map_cublas_status(const cublasStatus_t status) noexcept {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS:
      return static_cast<int>(cudaSuccess);
    case CUBLAS_STATUS_NOT_INITIALIZED:
      return static_cast<int>(cudaErrorInitializationError);
    case CUBLAS_STATUS_ALLOC_FAILED:
      return static_cast<int>(cudaErrorMemoryAllocation);
    case CUBLAS_STATUS_INVALID_VALUE:
      return static_cast<int>(cudaErrorInvalidValue);
    case CUBLAS_STATUS_ARCH_MISMATCH:
    case CUBLAS_STATUS_NOT_SUPPORTED:
      return static_cast<int>(cudaErrorNotSupported);
    case CUBLAS_STATUS_EXECUTION_FAILED:
      return static_cast<int>(cudaErrorLaunchFailure);
    default:
      return static_cast<int>(cudaErrorUnknown);
  }
}

__device__ __forceinline__ float decode_e4m3fn(const std::uint8_t bits) {
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

__device__ __forceinline__ float decode_e2m1(const std::uint8_t nibble) {
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

// One CTA owns one canonical N row. Ten packed spans and their four-scale
// words are prefetched before conversion, exposing enough independent loads
// to reach the screened Orin memory/compute balance without a physical
// transpose or a persistent BF16 weight copy.
__global__ __launch_bounds__(kThreads, 4)
void dequantize_nvfp4_contiguous_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    std::uint16_t* const canonical_bf16) {
  const unsigned int row = blockIdx.x;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const std::size_t packed_base =
      static_cast<std::size_t>(row) * kPackedPerRow;
  const std::size_t scale_base =
      static_cast<std::size_t>(row) * kScalesPerRow;
  auto* const output_pairs =
      reinterpret_cast<std::uint32_t*>(canonical_bf16);

  std::uint8_t packed_values[kPasses];
  std::uint32_t scale_words[kPasses];
#pragma unroll
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    const unsigned int packed_column = threadIdx.x + pass * kThreads;
    packed_values[pass] = packed_weights[packed_base + packed_column];
    scale_words[pass] = 0U;
    if (lane == 0U) {
      const std::size_t word_index =
          scale_base + pass * (kThreads / 8U) + warp * 4U;
      scale_words[pass] = *reinterpret_cast<const std::uint32_t*>(
          block_scales + word_index);
    }
  }

#pragma unroll
  for (unsigned int pass = 0U; pass < kPasses; ++pass) {
    const unsigned int packed_column = threadIdx.x + pass * kThreads;
    const std::uint32_t scale_word =
        __shfl_sync(0xffff'ffffU, scale_words[pass], 0);
    const std::uint8_t scale_code = static_cast<std::uint8_t>(
        scale_word >> ((lane >> 3U) * 8U));
    const float scale = decode_e4m3fn(scale_code);
    const std::uint8_t packed = packed_values[pass];
    const __nv_bfloat16 low = __float2bfloat16_rn(
        decode_e2m1(packed & 0x0fU) * scale);
    const __nv_bfloat16 high = __float2bfloat16_rn(
        decode_e2m1(packed >> 4U) * scale);
    output_pairs[packed_base + packed_column] =
        static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
        (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
  }
}

template <unsigned int kPassBase, unsigned int kWindowPasses>
__device__ __forceinline__ void dequantize_nvfp4_down_window(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    std::uint32_t* const output_pairs, const std::size_t packed_base,
    const std::size_t scale_base, const unsigned int lane,
    const unsigned int warp) {
  static_assert(kWindowPasses == 8U || kWindowPasses == 2U);
  static_assert(kPassBase + kWindowPasses <= kDownPasses);
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
    const float scale = decode_e4m3fn(scale_code);
    const std::uint8_t packed = packed_values[slot];
    const __nv_bfloat16 low = __float2bfloat16_rn(
        decode_e2m1(packed & 0x0fU) * scale);
    const __nv_bfloat16 high = __float2bfloat16_rn(
        decode_e2m1(packed >> 4U) * scale);
    output_pairs[packed_base + packed_column] =
        static_cast<std::uint32_t>(__bfloat16_as_ushort(low)) |
        (static_cast<std::uint32_t>(__bfloat16_as_ushort(high)) << 16U);
  }
}

__global__ __launch_bounds__(kThreads, 4)
void dequantize_nvfp4_down_window8_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales,
    std::uint16_t* const canonical_bf16) {
  const unsigned int row = blockIdx.x;
  const unsigned int lane = threadIdx.x & 31U;
  const unsigned int warp = threadIdx.x >> 5U;
  const std::size_t packed_base =
      static_cast<std::size_t>(row) * kDownPackedPerRow;
  const std::size_t scale_base =
      static_cast<std::size_t>(row) * kDownScalesPerRow;
  auto* const output_pairs =
      reinterpret_cast<std::uint32_t*>(canonical_bf16);
  dequantize_nvfp4_down_window<0U, 8U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
  dequantize_nvfp4_down_window<8U, 8U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
  dequantize_nvfp4_down_window<16U, 8U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
  dequantize_nvfp4_down_window<24U, 8U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
  dequantize_nvfp4_down_window<32U, 2U>(
      packed_weights, block_scales, output_pairs, packed_base, scale_base,
      lane, warp);
}

__global__ void fill_down_autotune_bf16_kernel(
    std::uint16_t* const values, const std::size_t value_count,
    const std::uint32_t seed) {
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < value_count; index += stride) {
    std::uint32_t bits = static_cast<std::uint32_t>(index) ^ seed;
    bits ^= bits >> 16U;
    bits *= 0x7feb'352dU;
    bits ^= bits >> 15U;
    bits *= 0x846c'a68bU;
    bits ^= bits >> 16U;
    // BF16 exponent 126 or 127 with a hashed sign/mantissa produces finite,
    // normal, nonzero values in roughly [-2, 2), avoiding an all-zero tuning
    // workload without requiring an additional host-to-device transfer.
    values[index] = static_cast<std::uint16_t>(
        ((bits & 1U) << 15U) | ((126U + ((bits >> 1U) & 1U)) << 7U) |
        ((bits >> 2U) & 0x7fU));
  }
}

[[nodiscard]] int validate_launch(
    const Sm87Nvfp4PrefillCublasLtContext* const context,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const bf16_scratch,
    const std::size_t scratch_bytes,
    std::uint16_t* const output) noexcept {
  if (context == nullptr || context->heuristic_rank < 0 ||
      token_count != kTokens || rows != kRows || columns != kColumns ||
      !std::isfinite(weight_scale_2) || weight_scale_2 <= 0.0F ||
      scratch_bytes < kScratchBytes ||
      !aligned(packed_weights, 16U) || !aligned(block_scales, 4U) ||
      !aligned(activations, 16U) || !aligned(bf16_scratch, 16U) ||
      !aligned(output, 16U)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  AddressSpan packed_span{};
  AddressSpan scale_span{};
  AddressSpan activation_span{};
  AddressSpan scratch_span{};
  AddressSpan output_span{};
  if (!make_span(packed_weights, kPackedBytes, &packed_span) ||
      !make_span(block_scales, kScaleBytes, &scale_span) ||
      !make_span(activations, kActivationBytes, &activation_span) ||
      !make_span(bf16_scratch, kScratchBytes, &scratch_span) ||
      !make_span(output, kOutputBytes, &output_span)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::array<AddressSpan, 5U> spans{{
      packed_span, scale_span, activation_span, scratch_span, output_span}};
  for (std::size_t left = 0U; left < spans.size(); ++left) {
    for (std::size_t right = left + 1U; right < spans.size(); ++right) {
      if (overlaps(spans[left], spans[right])) {
        return static_cast<int>(cudaErrorInvalidValue);
      }
    }
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int validate_down_launch(
    const Sm87Nvfp4PrefillDownCublasLtContext* const context,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const bf16_scratch,
    const std::size_t scratch_bytes,
    std::uint16_t* const output) noexcept {
  if (context == nullptr || context->heuristic_rank < 0 ||
      token_count != kDownTokens || rows != kDownRows ||
      columns != kDownColumns || !std::isfinite(weight_scale_2) ||
      weight_scale_2 <= 0.0F || scratch_bytes < kDownScratchBytes ||
      !aligned(packed_weights, 16U) || !aligned(block_scales, 4U) ||
      !aligned(activations, 16U) || !aligned(bf16_scratch, 16U) ||
      !aligned(output, 16U)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  AddressSpan packed_span{};
  AddressSpan scale_span{};
  AddressSpan activation_span{};
  AddressSpan scratch_span{};
  AddressSpan output_span{};
  if (!make_span(packed_weights, kDownPackedBytes, &packed_span) ||
      !make_span(block_scales, kDownScaleBytes, &scale_span) ||
      !make_span(activations, kDownActivationBytes, &activation_span) ||
      !make_span(bf16_scratch, kDownScratchBytes, &scratch_span) ||
      !make_span(output, kDownOutputBytes, &output_span)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  const std::array<AddressSpan, 5U> spans{{
      packed_span, scale_span, activation_span, scratch_span, output_span}};
  for (std::size_t left = 0U; left < spans.size(); ++left) {
    for (std::size_t right = left + 1U; right < spans.size(); ++right) {
      if (overlaps(spans[left], spans[right])) {
        return static_cast<int>(cudaErrorInvalidValue);
      }
    }
  }
  return static_cast<int>(cudaSuccess);
}

// Down's exact-C512 Lt problem has multiple zero-workspace heuristics whose
// runtime order is not a performance order on SM87.  Tune against the exact
// BF16 operands once while constructing the context; launches remain
// allocation-free and retain only the winning algorithm.
[[nodiscard]] int autotune_down_algorithm(
    Sm87Nvfp4PrefillDownCublasLtContext* const context,
    const std::array<cublasLtMatmulHeuristicResult_t,
                     kMaximumHeuristics>& results,
    const int result_count) noexcept {
  if (context == nullptr || result_count < 0 ||
      result_count > kMaximumHeuristics) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  cudaStream_t stream = nullptr;
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  void* scratch = nullptr;
  void* activations = nullptr;
  void* output = nullptr;
  int tuning_status = static_cast<int>(cudaSuccess);

  const auto record_cuda_failure = [&tuning_status](
                                       const cudaError_t candidate) {
    if (candidate != cudaSuccess &&
        tuning_status == static_cast<int>(cudaSuccess)) {
      tuning_status = static_cast<int>(candidate);
    }
    return candidate == cudaSuccess;
  };

  const bool resources_ready = record_cuda_failure(
                                   cudaStreamCreateWithFlags(
                                       &stream, cudaStreamNonBlocking)) &&
                               record_cuda_failure(cudaEventCreate(&start)) &&
                               record_cuda_failure(cudaEventCreate(&stop)) &&
                               record_cuda_failure(
                                   cudaMalloc(&scratch, kDownScratchBytes)) &&
                               record_cuda_failure(cudaMalloc(
                                   &activations, kDownActivationBytes)) &&
                               record_cuda_failure(
                                   cudaMalloc(&output, kDownOutputBytes));
  if (resources_ready) {
    constexpr unsigned int kFillBlocks = 1'024U;
    (void)cudaGetLastError();
    fill_down_autotune_bf16_kernel<<<kFillBlocks, kThreads, 0U, stream>>>(
        static_cast<std::uint16_t*>(scratch),
        kDownScratchBytes / sizeof(std::uint16_t), 0x9e37'79b9U);
    (void)record_cuda_failure(cudaGetLastError());
    if (tuning_status == static_cast<int>(cudaSuccess)) {
      fill_down_autotune_bf16_kernel<<<kFillBlocks, kThreads, 0U, stream>>>(
          static_cast<std::uint16_t*>(activations),
          kDownActivationBytes / sizeof(std::uint16_t), 0x243f'6a88U);
      (void)record_cuda_failure(cudaGetLastError());
    }
    if (tuning_status == static_cast<int>(cudaSuccess)) {
      fill_down_autotune_bf16_kernel<<<kFillBlocks, kThreads, 0U, stream>>>(
          static_cast<std::uint16_t*>(output),
          kDownOutputBytes / sizeof(std::uint16_t), 0xb7e1'5163U);
      (void)record_cuda_failure(cudaGetLastError());
    }
    if (tuning_status == static_cast<int>(cudaSuccess)) {
      (void)record_cuda_failure(cudaStreamSynchronize(stream));
    }
  }

  constexpr int kWarmupRounds = 2;
  constexpr int kTimingRounds = 3;
  constexpr int kTimedIterations = 2;
  constexpr float kAlpha = 1.0F;
  constexpr float kBeta = 0.0F;
  constexpr int kMinimumEligibleAlgorithms = 2;
  std::array<int, kMaximumHeuristics> candidate_indices{};
  std::array<bool, kMaximumHeuristics> candidate_active{};
  std::array<float, kMaximumHeuristics> best_average_ms{};
  best_average_ms.fill(std::numeric_limits<float>::infinity());
  int candidate_count = 0;

  for (int index = 0; index < result_count; ++index) {
    const auto& result = results[static_cast<std::size_t>(index)];
    if (result.state == CUBLAS_STATUS_SUCCESS &&
        result.workspaceSize == kWorkspaceBytes) {
      candidate_indices[static_cast<std::size_t>(candidate_count)] = index;
      candidate_active[static_cast<std::size_t>(index)] = true;
      ++candidate_count;
    }
  }

  // Alternate forward/reverse order and rotate the third timing round.  This
  // keeps clock ramp and neighboring GPU work from systematically favoring
  // the first or last heuristic returned by cuBLASLt.
  const auto ordered_candidate =
      [&candidate_indices, candidate_count](const int round,
                                            const int position) {
        int ordered_position = position;
        if ((round & 1) != 0) {
          ordered_position = candidate_count - 1 - ordered_position;
        } else if (round > 0) {
          ordered_position = (ordered_position + 1) % candidate_count;
        }
        return candidate_indices[static_cast<std::size_t>(ordered_position)];
      };

  // A heuristic may be reported as supported yet fail when executed on a
  // particular driver.  Drain its stream work and clear the runtime error so
  // the remaining independent zero-workspace candidates can still compete.
  // If the CUDA context carries an unrecoverable error, subsequent candidates
  // will fail as well and the factory will correctly return not-supported.
  const auto discard_candidate_failure = [stream]() {
    (void)cudaStreamSynchronize(stream);
    (void)cudaGetLastError();
  };

  const auto warmup_candidate =
      [&](const int candidate_index) {
        const auto& result =
            results[static_cast<std::size_t>(candidate_index)];
        const cublasStatus_t lt_status = cublasLtMatmul(
            context->handle, context->operation, &kAlpha, scratch,
            context->weight_layout, activations, context->activation_layout,
            &kBeta, output, context->output_layout, output,
            context->output_layout, &result.algo, nullptr, kWorkspaceBytes,
            stream);
        if (lt_status != CUBLAS_STATUS_SUCCESS ||
            cudaStreamSynchronize(stream) != cudaSuccess) {
          discard_candidate_failure();
          return false;
        }
        return true;
      };

  const auto time_candidate =
      [&](const int candidate_index, float* const average_ms) {
        if (average_ms == nullptr || cudaEventRecord(start, stream) !=
                                         cudaSuccess) {
          discard_candidate_failure();
          return false;
        }
        const auto& result =
            results[static_cast<std::size_t>(candidate_index)];
        for (int iteration = 0; iteration < kTimedIterations; ++iteration) {
          const cublasStatus_t lt_status = cublasLtMatmul(
              context->handle, context->operation, &kAlpha, scratch,
              context->weight_layout, activations,
              context->activation_layout, &kBeta, output,
              context->output_layout, output, context->output_layout,
              &result.algo, nullptr, kWorkspaceBytes, stream);
          if (lt_status != CUBLAS_STATUS_SUCCESS) {
            discard_candidate_failure();
            return false;
          }
        }
        if (cudaEventRecord(stop, stream) != cudaSuccess ||
            cudaEventSynchronize(stop) != cudaSuccess) {
          discard_candidate_failure();
          return false;
        }
        float elapsed_ms = 0.0F;
        if (cudaEventElapsedTime(&elapsed_ms, start, stop) != cudaSuccess) {
          discard_candidate_failure();
          return false;
        }
        *average_ms = elapsed_ms / static_cast<float>(kTimedIterations);
        return std::isfinite(*average_ms) && *average_ms > 0.0F;
      };

  if (tuning_status == static_cast<int>(cudaSuccess) &&
      candidate_count >= kMinimumEligibleAlgorithms) {
    for (int round = 0; round < kWarmupRounds; ++round) {
      for (int position = 0; position < candidate_count; ++position) {
        const int index = ordered_candidate(round, position);
        if (candidate_active[static_cast<std::size_t>(index)] &&
            !warmup_candidate(index)) {
          candidate_active[static_cast<std::size_t>(index)] = false;
        }
      }
    }

    for (int round = 0; round < kTimingRounds; ++round) {
      for (int position = 0; position < candidate_count; ++position) {
        const int index = ordered_candidate(round, position);
        if (!candidate_active[static_cast<std::size_t>(index)]) {
          continue;
        }
        float average_ms = 0.0F;
        if (!time_candidate(index, &average_ms)) {
          candidate_active[static_cast<std::size_t>(index)] = false;
          best_average_ms[static_cast<std::size_t>(index)] =
              std::numeric_limits<float>::infinity();
          continue;
        }
        best_average_ms[static_cast<std::size_t>(index)] = std::min(
            best_average_ms[static_cast<std::size_t>(index)], average_ms);
      }
    }
  }

  float fastest_average_ms = std::numeric_limits<float>::infinity();
  cublasLtMatmulAlgo_t fastest_algorithm{};
  int fastest_rank = -1;
  for (int position = 0; position < candidate_count; ++position) {
    const int index =
        candidate_indices[static_cast<std::size_t>(position)];
    if (!candidate_active[static_cast<std::size_t>(index)]) {
      continue;
    }
    const float candidate_ms =
        best_average_ms[static_cast<std::size_t>(index)];
    if (candidate_ms < fastest_average_ms) {
      fastest_average_ms = candidate_ms;
      fastest_algorithm =
          results[static_cast<std::size_t>(index)].algo;
      fastest_rank = index;
    }
  }

  // Heuristic ranks are not a stable cuBLASLt ABI, so admission deliberately
  // does not depend on rank zero (or on the ratio between two rank numbers).
  // The measured latency is only useful for ordering candidates.  It is not a
  // support predicate: GPU contention, power state, and concurrent context
  // construction can scale every valid candidate past any fixed millisecond
  // ceiling.  Availability therefore depends only on finding meaningful
  // zero-workspace choice and successfully timing at least one candidate.
  // Performance admission remains the responsibility of the repeatable
  // ceiling benchmark, not a noisy one-shot production factory call.
  if (tuning_status == static_cast<int>(cudaSuccess) &&
      (candidate_count < kMinimumEligibleAlgorithms || fastest_rank < 0)) {
    tuning_status = static_cast<int>(cudaErrorNotSupported);
  }

  // Synchronize before releasing operands even when an algorithm reports a
  // launch failure.  Cleanup errors are returned if tuning itself succeeded,
  // while an earlier, more diagnostic failure keeps precedence.
  if (stream != nullptr) {
    (void)record_cuda_failure(cudaStreamSynchronize(stream));
  }
  if (output != nullptr) {
    (void)record_cuda_failure(cudaFree(output));
  }
  if (activations != nullptr) {
    (void)record_cuda_failure(cudaFree(activations));
  }
  if (scratch != nullptr) {
    (void)record_cuda_failure(cudaFree(scratch));
  }
  if (stop != nullptr) {
    (void)record_cuda_failure(cudaEventDestroy(stop));
  }
  if (start != nullptr) {
    (void)record_cuda_failure(cudaEventDestroy(start));
  }
  if (stream != nullptr) {
    (void)record_cuda_failure(cudaStreamDestroy(stream));
  }

  if (tuning_status == static_cast<int>(cudaSuccess)) {
    context->algorithm = fastest_algorithm;
    context->heuristic_rank = fastest_rank;
  }
  return tuning_status;
}

}  // namespace

int create_sm87_nvfp4_prefill_cublaslt_context(
    Sm87Nvfp4PrefillCublasLtContext** const context) noexcept {
  if (context == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *context = nullptr;

  int device = -1;
  cudaError_t cuda_status = cudaGetDevice(&device);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  cudaDeviceProp properties{};
  cuda_status = cudaGetDeviceProperties(&properties, device);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  if (properties.major != 8 || properties.minor != 7) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  auto* const created =
      new (std::nothrow) Sm87Nvfp4PrefillCublasLtContext{};
  if (created == nullptr) {
    return static_cast<int>(cudaErrorMemoryAllocation);
  }
  created->device = device;

  cublasStatus_t status = cublasLtCreate(&created->handle);
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescCreate(
        &created->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  }
  const cublasOperation_t transpose_weight = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        created->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_weight,
        sizeof(transpose_weight));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &created->weight_layout, CUDA_R_16BF, kColumns, kRows, kColumns);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &created->activation_layout, CUDA_R_16BF, kColumns, kTokens,
        kColumns);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &created->output_layout, CUDA_R_16BF, kRows, kTokens, kRows);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceCreate(&created->preference);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    std::size_t maximum_workspace_bytes = kWorkspaceBytes;
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &maximum_workspace_bytes, sizeof(maximum_workspace_bytes));
  }
  // The public launch contract guarantees 16-byte alignment for every Lt
  // operand, not cudaMalloc's stronger incidental alignment. Filter the
  // heuristic set against that exact guarantee so an admitted launch cannot
  // enqueue dequantization and only then discover an incompatible Lt algo.
  constexpr std::uint32_t kMinimumOperandAlignmentBytes = 16U;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        &kMinimumOperandAlignmentBytes, sizeof(kMinimumOperandAlignmentBytes));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        &kMinimumOperandAlignmentBytes, sizeof(kMinimumOperandAlignmentBytes));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        &kMinimumOperandAlignmentBytes, sizeof(kMinimumOperandAlignmentBytes));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        &kMinimumOperandAlignmentBytes, sizeof(kMinimumOperandAlignmentBytes));
  }

  std::array<cublasLtMatmulHeuristicResult_t, kMaximumHeuristics> results{};
  int result_count = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(
        created->handle, created->operation, created->weight_layout,
        created->activation_layout, created->output_layout,
        created->output_layout, created->preference, kMaximumHeuristics,
        results.data(), &result_count);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    for (int index = 0; index < result_count; ++index) {
      const auto& result = results[static_cast<std::size_t>(index)];
      if (result.state == CUBLAS_STATUS_SUCCESS &&
          result.workspaceSize == kWorkspaceBytes) {
        created->algorithm = result.algo;
        created->heuristic_rank = index;
        break;
      }
    }
    if (created->heuristic_rank < 0) {
      status = CUBLAS_STATUS_NOT_SUPPORTED;
    }
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    const int mapped = map_cublas_status(status);
    destroy_sm87_nvfp4_prefill_cublaslt_context(created);
    return mapped;
  }

  *context = created;
  return static_cast<int>(cudaSuccess);
}

int query_sm87_nvfp4_prefill_cublaslt_context(
    const Sm87Nvfp4PrefillCublasLtContext* const context,
    std::size_t* const scratch_bytes, std::size_t* const workspace_bytes,
    int* const heuristic_rank) noexcept {
  if (context == nullptr || context->heuristic_rank < 0 ||
      scratch_bytes == nullptr || workspace_bytes == nullptr ||
      heuristic_rank == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *scratch_bytes = kScratchBytes;
  *workspace_bytes = kWorkspaceBytes;
  *heuristic_rank = context->heuristic_rank;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_prefill_cublaslt_gate_c512(
    Sm87Nvfp4PrefillCublasLtContext* const context,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const bf16_scratch,
    const std::size_t scratch_bytes, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_launch(
      context, packed_weights, block_scales, weight_scale_2, activations,
      token_count, rows, columns, bf16_scratch, scratch_bytes, output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  int active_device = -1;
  cudaError_t cuda_status = cudaGetDevice(&active_device);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  if (active_device != context->device) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  dequantize_nvfp4_contiguous_kernel<<<static_cast<unsigned int>(kRows),
                                       kThreads, 0U, stream>>>(
      packed_weights, block_scales, bf16_scratch);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }

  constexpr float kBeta = 0.0F;
  const cublasStatus_t lt_status = cublasLtMatmul(
      context->handle, context->operation, &weight_scale_2, bf16_scratch,
      context->weight_layout, activations, context->activation_layout, &kBeta,
      output, context->output_layout, output, context->output_layout,
      &context->algorithm, nullptr, kWorkspaceBytes, stream);
  if (lt_status != CUBLAS_STATUS_SUCCESS) {
    return map_cublas_status(lt_status);
  }
  return static_cast<int>(cudaGetLastError());
}

void destroy_sm87_nvfp4_prefill_cublaslt_context(
    Sm87Nvfp4PrefillCublasLtContext* const context) noexcept {
  if (context == nullptr) {
    return;
  }
  if (context->preference != nullptr) {
    (void)cublasLtMatmulPreferenceDestroy(context->preference);
  }
  if (context->output_layout != nullptr) {
    (void)cublasLtMatrixLayoutDestroy(context->output_layout);
  }
  if (context->activation_layout != nullptr) {
    (void)cublasLtMatrixLayoutDestroy(context->activation_layout);
  }
  if (context->weight_layout != nullptr) {
    (void)cublasLtMatrixLayoutDestroy(context->weight_layout);
  }
  if (context->operation != nullptr) {
    (void)cublasLtMatmulDescDestroy(context->operation);
  }
  if (context->handle != nullptr) {
    (void)cublasLtDestroy(context->handle);
  }
  delete context;
}

int create_sm87_nvfp4_prefill_down_cublaslt_context(
    Sm87Nvfp4PrefillDownCublasLtContext** const context) noexcept {
  if (context == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *context = nullptr;

  int device = -1;
  cudaError_t cuda_status = cudaGetDevice(&device);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  cudaDeviceProp properties{};
  cuda_status = cudaGetDeviceProperties(&properties, device);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  if (properties.major != 8 || properties.minor != 7) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  auto* const created =
      new (std::nothrow) Sm87Nvfp4PrefillDownCublasLtContext{};
  if (created == nullptr) {
    return static_cast<int>(cudaErrorMemoryAllocation);
  }
  created->device = device;

  cublasStatus_t status = cublasLtCreate(&created->handle);
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescCreate(
        &created->operation, CUBLAS_COMPUTE_32F, CUDA_R_32F);
  }
  const cublasOperation_t transpose_weight = CUBLAS_OP_T;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulDescSetAttribute(
        created->operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose_weight,
        sizeof(transpose_weight));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &created->weight_layout, CUDA_R_16BF, kDownColumns, kDownRows,
        kDownColumns);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &created->activation_layout, CUDA_R_16BF, kDownColumns, kDownTokens,
        kDownColumns);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatrixLayoutCreate(
        &created->output_layout, CUDA_R_16BF, kDownRows, kDownTokens,
        kDownRows);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceCreate(&created->preference);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    std::size_t maximum_workspace_bytes = kWorkspaceBytes;
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &maximum_workspace_bytes, sizeof(maximum_workspace_bytes));
  }
  constexpr std::uint32_t kMinimumOperandAlignmentBytes = 16U;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES,
        &kMinimumOperandAlignmentBytes, sizeof(kMinimumOperandAlignmentBytes));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES,
        &kMinimumOperandAlignmentBytes, sizeof(kMinimumOperandAlignmentBytes));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES,
        &kMinimumOperandAlignmentBytes, sizeof(kMinimumOperandAlignmentBytes));
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulPreferenceSetAttribute(
        created->preference, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES,
        &kMinimumOperandAlignmentBytes, sizeof(kMinimumOperandAlignmentBytes));
  }

  std::array<cublasLtMatmulHeuristicResult_t, kMaximumHeuristics> results{};
  int result_count = 0;
  if (status == CUBLAS_STATUS_SUCCESS) {
    status = cublasLtMatmulAlgoGetHeuristic(
        created->handle, created->operation, created->weight_layout,
        created->activation_layout, created->output_layout,
        created->output_layout, created->preference, kMaximumHeuristics,
        results.data(), &result_count);
  }
  if (status == CUBLAS_STATUS_SUCCESS) {
    const int tuning_status =
        autotune_down_algorithm(created, results, result_count);
    if (tuning_status != static_cast<int>(cudaSuccess)) {
      destroy_sm87_nvfp4_prefill_down_cublaslt_context(created);
      return tuning_status;
    }
  }
  if (status != CUBLAS_STATUS_SUCCESS) {
    const int mapped = map_cublas_status(status);
    destroy_sm87_nvfp4_prefill_down_cublaslt_context(created);
    return mapped;
  }

  *context = created;
  return static_cast<int>(cudaSuccess);
}

int query_sm87_nvfp4_prefill_down_cublaslt_context(
    const Sm87Nvfp4PrefillDownCublasLtContext* const context,
    std::size_t* const scratch_bytes, std::size_t* const workspace_bytes,
    int* const heuristic_rank) noexcept {
  if (context == nullptr || context->heuristic_rank < 0 ||
      scratch_bytes == nullptr || workspace_bytes == nullptr ||
      heuristic_rank == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *scratch_bytes = kDownScratchBytes;
  *workspace_bytes = kWorkspaceBytes;
  *heuristic_rank = context->heuristic_rank;
  return static_cast<int>(cudaSuccess);
}

int launch_sm87_nvfp4_prefill_cublaslt_down_c512(
    Sm87Nvfp4PrefillDownCublasLtContext* const context,
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations,
    const std::size_t token_count, const std::size_t rows,
    const std::size_t columns, std::uint16_t* const bf16_scratch,
    const std::size_t scratch_bytes, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  const int validation = validate_down_launch(
      context, packed_weights, block_scales, weight_scale_2, activations,
      token_count, rows, columns, bf16_scratch, scratch_bytes, output);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  int active_device = -1;
  cudaError_t cuda_status = cudaGetDevice(&active_device);
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }
  if (active_device != context->device) {
    return static_cast<int>(cudaErrorInvalidDevice);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  dequantize_nvfp4_down_window8_kernel<<<
      static_cast<unsigned int>(kDownRows), kThreads, 0U, stream>>>(
      packed_weights, block_scales, bf16_scratch);
  cuda_status = cudaGetLastError();
  if (cuda_status != cudaSuccess) {
    return static_cast<int>(cuda_status);
  }

  constexpr float kBeta = 0.0F;
  const cublasStatus_t lt_status = cublasLtMatmul(
      context->handle, context->operation, &weight_scale_2, bf16_scratch,
      context->weight_layout, activations, context->activation_layout, &kBeta,
      output, context->output_layout, output, context->output_layout,
      &context->algorithm, nullptr, kWorkspaceBytes, stream);
  if (lt_status != CUBLAS_STATUS_SUCCESS) {
    return map_cublas_status(lt_status);
  }
  return static_cast<int>(cudaGetLastError());
}

void destroy_sm87_nvfp4_prefill_down_cublaslt_context(
    Sm87Nvfp4PrefillDownCublasLtContext* const context) noexcept {
  if (context == nullptr) {
    return;
  }
  if (context->preference != nullptr) {
    (void)cublasLtMatmulPreferenceDestroy(context->preference);
  }
  if (context->output_layout != nullptr) {
    (void)cublasLtMatrixLayoutDestroy(context->output_layout);
  }
  if (context->activation_layout != nullptr) {
    (void)cublasLtMatrixLayoutDestroy(context->activation_layout);
  }
  if (context->weight_layout != nullptr) {
    (void)cublasLtMatrixLayoutDestroy(context->weight_layout);
  }
  if (context->operation != nullptr) {
    (void)cublasLtMatmulDescDestroy(context->operation);
  }
  if (context->handle != nullptr) {
    (void)cublasLtDestroy(context->handle);
  }
  delete context;
}

int query_sm87_nvfp4_prefill_down_cublaslt_dequant_resources_test_cuda(
    int* const registers_per_thread, std::size_t* const static_shared_bytes,
    std::size_t* const local_bytes_per_thread,
    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes_per_thread == nullptr || active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status = cudaFuncGetAttributes(
      &attributes, dequantize_nvfp4_down_window8_kernel);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active_blocks = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active_blocks, dequantize_nvfp4_down_window8_kernel, kThreads, 0U);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes_per_thread = attributes.localSizeBytes;
  *active_blocks_per_sm = active_blocks;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::kernels
