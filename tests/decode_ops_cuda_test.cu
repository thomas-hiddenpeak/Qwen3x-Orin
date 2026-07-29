#include "q3x/core/sha256.h"
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/layout_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace q3x::runtime {

int query_residual_add_headwise_centered_rms_norm_m32_5120_test_cuda_resources(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int launch_attention_scores_baseline_24_4_256_test_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    std::size_t query_head_count, std::size_t kv_head_count,
    std::size_t sequence_length, std::size_t head_dimension,
    float attention_scale, float* scores, void* cuda_stream) noexcept;

int launch_attention_scores_warp_positions_24_4_256_test_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    std::size_t query_head_count, std::size_t kv_head_count,
    std::size_t sequence_length, std::size_t head_dimension,
    float attention_scale, float* scores, void* cuda_stream) noexcept;

bool use_attention_scores_warp_positions_24_4_256_test(
    std::size_t query_head_count, std::size_t kv_head_count,
    std::size_t sequence_length, std::size_t head_dimension) noexcept;

int query_attention_scores_baseline_24_4_256_test_cuda_resources(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int query_attention_scores_warp_positions_24_4_256_test_cuda_resources(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int launch_attention_values_baseline_24_4_256_test_cuda(
    const std::uint16_t* value_cache, const float* probabilities,
    std::size_t query_head_count, std::size_t kv_head_count,
    std::size_t sequence_length, std::size_t head_dimension,
    std::uint16_t* output, void* cuda_stream) noexcept;

int launch_attention_values_exact_24_4_256_test_cuda(
    const std::uint16_t* value_cache, const float* probabilities,
    std::size_t query_head_count, std::size_t kv_head_count,
    std::size_t sequence_length, std::size_t head_dimension,
    std::uint16_t* output, void* cuda_stream) noexcept;

int query_attention_values_baseline_24_4_256_test_cuda_resources(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int query_attention_values_exact_24_4_256_test_cuda_resources(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int query_attention_values_exact_24_4_256_test_cuda_selection(
    std::size_t query_head_count, std::size_t kv_head_count,
    std::size_t sequence_length, std::size_t head_dimension,
    int* selected) noexcept;

int launch_gqa_attention_sigmoid_gate_warp_positions_24_4_256_test_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, std::size_t sequence_length,
    float attention_scale, float* probabilities_scratch,
    std::size_t scratch_elements, const std::uint16_t* gate,
    std::uint16_t* output, void* cuda_stream) noexcept;

int launch_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_test_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, std::size_t sequence_length,
    float attention_scale, float* probabilities_scratch,
    std::size_t scratch_elements, const std::uint16_t* gate,
    std::uint16_t* output, void* cuda_stream) noexcept;

int query_gqa_attention_sigmoid_gate_24_4_256_resources_test_cuda(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int query_gqa_attention_sigmoid_gate_warp_positions_24_4_256_resources_test_cuda(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int query_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_resources_test_cuda(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int launch_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_test_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate,
    std::size_t first_position, std::size_t token_count,
    float attention_scale, std::uint16_t* output,
    void* cuda_stream) noexcept;

int query_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_resources_test_cuda(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int launch_full_attention_preprocess_warp_rms_24_4_256_64_test_cuda(
    const std::uint16_t* interleaved_q_gate, std::uint16_t* key,
    const std::uint16_t* q_weight, const std::uint16_t* k_weight,
    float epsilon, std::uint16_t* query_output,
    std::uint16_t* gate_output, const float* cosines, const float* sines,
    std::size_t first_position, std::size_t token_count,
    void* cuda_stream) noexcept;

int query_full_attention_preprocess_24_4_256_64_resources_test_cuda(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

int query_full_attention_preprocess_warp_rms_24_4_256_64_resources_test_cuda(
    int* registers, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads,
    int* active_blocks_per_multiprocessor) noexcept;

}  // namespace q3x::runtime

namespace {

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const cudaError_t status,
                             const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }

  void expect_near(const float actual, const float expected,
                   const float tolerance, const std::string& message) {
    if (!(std::fabs(actual - expected) <= tolerance)) {
      ++failures_;
      std::cerr << "FAIL: " << message << ": expected " << expected
                << ", got " << actual << ", tolerance " << tolerance << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T>
class ManagedBuffer {
 public:
  ManagedBuffer() = default;
  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;

  ~ManagedBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) {
    count_ = count;
    return cudaMallocManaged(reinterpret_cast<void**>(&data_),
                             count * sizeof(T));
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t size() const noexcept { return count_; }

  [[nodiscard]] T& operator[](const std::size_t index) noexcept {
    return data_[index];
  }
  [[nodiscard]] const T& operator[](const std::size_t index) const noexcept {
    return data_[index];
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

class GuardedBf16Buffer {
 public:
  static constexpr std::size_t kGuardElements = 17U;

  [[nodiscard]] bool allocate(TestContext& test, const std::size_t count,
                              const std::string& label) {
    count_ = count;
    return test.cuda_ok(storage_.allocate(count + 2U * kGuardElements),
                        label + " allocate");
  }

  void initialize(const std::uint16_t payload_value,
                  const std::uint16_t prefix_canary,
                  const std::uint16_t suffix_canary) {
    prefix_canary_ = prefix_canary;
    suffix_canary_ = suffix_canary;
    std::fill_n(storage_.data(), kGuardElements, prefix_canary_);
    std::fill_n(data(), count_, payload_value);
    std::fill_n(data() + count_, kGuardElements, suffix_canary_);
  }

  [[nodiscard]] std::uint16_t* data() noexcept {
    return storage_.data() + kGuardElements;
  }
  [[nodiscard]] const std::uint16_t* data() const noexcept {
    return storage_.data() + kGuardElements;
  }

  [[nodiscard]] std::vector<std::uint16_t> snapshot() const {
    return {storage_.data(), storage_.data() + storage_.size()};
  }

  void expect_guards(TestContext& test, const std::string& label) const {
    const bool prefix_ok =
        std::all_of(storage_.data(), storage_.data() + kGuardElements,
                    [&](const std::uint16_t value) {
                      return value == prefix_canary_;
                    });
    const bool suffix_ok =
        std::all_of(data() + count_, data() + count_ + kGuardElements,
                    [&](const std::uint16_t value) {
                      return value == suffix_canary_;
                    });
    test.expect(prefix_ok, label + " prefix canary is intact");
    test.expect(suffix_ok, label + " suffix canary is intact");
  }

  void expect_snapshot(TestContext& test,
                       const std::vector<std::uint16_t>& expected,
                       const std::string& label) const {
    const auto mismatch = std::mismatch(storage_.data(),
                                        storage_.data() + storage_.size(),
                                        expected.data());
    const bool equal = mismatch.first == storage_.data() + storage_.size();
    test.expect(equal, label + " is unchanged" +
                           (equal ? std::string{}
                                  : " at storage element " +
                                        std::to_string(static_cast<std::size_t>(
                                            mismatch.first -
                                            storage_.data()))));
  }

 private:
  ManagedBuffer<std::uint16_t> storage_;
  std::size_t count_ = 0U;
  std::uint16_t prefix_canary_ = 0U;
  std::uint16_t suffix_canary_ = 0U;
};

void expect_bf16_bits_equal(TestContext& test,
                            const std::uint16_t* const actual,
                            const std::uint16_t* const expected,
                            const std::size_t count,
                            const std::string& label) {
  const auto mismatch = std::mismatch(actual, actual + count, expected);
  const bool equal = mismatch.first == actual + count;
  test.expect(equal, label + " is bitwise identical" +
                         (equal ? std::string{}
                                : " at element " +
                                      std::to_string(static_cast<std::size_t>(
                                          mismatch.first - actual)) +
                                      ", actual=" +
                                      std::to_string(*mismatch.first) +
                                      ", expected=" +
                                      std::to_string(expected[
                                          mismatch.first - actual])));
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fffffffU) > 0x7f800000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t value) {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] q3x::runtime::Bf16GreedyArgmaxResult
bf16_greedy_argmax_cpu_oracle(const std::uint16_t* const input,
                              const std::size_t element_count) {
  q3x::runtime::Bf16GreedyArgmaxResult result;
  result.index = std::numeric_limits<std::uint32_t>::max();
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0U; index < element_count; ++index) {
    const std::uint16_t bits = input[index];
    if ((bits & 0x7f80U) == 0x7f80U) {
      result.has_nonfinite = 1U;
      continue;
    }
    const float value = decode_bf16(bits);
    if (result.index == std::numeric_limits<std::uint32_t>::max() ||
        value > maximum) {
      result.index = static_cast<std::uint32_t>(index);
      result.value_bits = bits;
      maximum = value;
    }
  }
  return result;
}

[[nodiscard]] std::uint32_t next_deterministic_random(
    std::uint32_t& state) noexcept {
  state ^= state << 13U;
  state ^= state >> 17U;
  state ^= state << 5U;
  return state;
}

[[nodiscard]] float float_from_bits(const std::uint32_t bits) {
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

template <typename Launch>
[[nodiscard]] bool launch_after_stale(TestContext& test, cudaStream_t stream,
                                      const std::string& label,
                                      Launch&& launch) {
  const cudaError_t stale =
      cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
  test.expect(stale == cudaErrorInvalidValue,
              label + " seeds stale CUDA last-error");
  bool ready = test.cuda_ok(static_cast<cudaError_t>(launch()),
                            label + " launch ignores stale error");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  return ready;
}

template <typename Launch>
[[nodiscard]] float measure_cuda_span_milliseconds(
    TestContext& test, cudaStream_t stream, const std::size_t iteration_count,
    const std::string& label, Launch&& launch) {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;
  bool ready = test.cuda_ok(cudaEventCreate(&start), label + " create start");
  ready = ready &&
          test.cuda_ok(cudaEventCreate(&stop), label + " create stop");
  if (!ready) {
    if (start != nullptr) {
      (void)cudaEventDestroy(start);
    }
    return std::numeric_limits<float>::quiet_NaN();
  }

  ready = test.cuda_ok(cudaEventRecord(start, stream), label + " record start");
  for (std::size_t iteration = 0U; ready && iteration < iteration_count;
       ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch()),
                         label + " measured launch");
  }
  ready = ready &&
          test.cuda_ok(cudaEventRecord(stop, stream), label + " record stop");
  ready = ready && test.cuda_ok(cudaEventSynchronize(stop),
                                label + " synchronize stop");

  float average_milliseconds = std::numeric_limits<float>::quiet_NaN();
  if (ready) {
    float total_milliseconds = 0.0F;
    ready = test.cuda_ok(
        cudaEventElapsedTime(&total_milliseconds, start, stop),
        label + " elapsed time");
    if (ready) {
      average_milliseconds =
          total_milliseconds / static_cast<float>(iteration_count);
    }
  }
  (void)test.cuda_ok(cudaEventDestroy(start), label + " destroy start");
  (void)test.cuda_ok(cudaEventDestroy(stop), label + " destroy stop");
  return average_milliseconds;
}

void expect_bf16_match(TestContext& test, const std::uint16_t actual,
                       const std::uint16_t expected,
                       const std::string& message) {
  const float actual_value = decode_bf16(actual);
  const float expected_value = decode_bf16(expected);
  if (std::isnan(expected_value)) {
    test.expect(std::isnan(actual_value), message + " NaN classification");
    return;
  }
  if (std::isinf(expected_value)) {
    test.expect(std::isinf(actual_value) &&
                    std::signbit(actual_value) == std::signbit(expected_value),
                message + " infinity classification");
    return;
  }
  const float tolerance =
      std::max(2.0e-3F, std::fabs(expected_value) * 8.0e-3F);
  test.expect_near(actual_value, expected_value, tolerance, message);
}

void test_launch_validation(TestContext& test) {
  constexpr std::size_t kMaximum = std::numeric_limits<std::size_t>::max();
  constexpr std::size_t kFusedHiddenSize = 5'120U;
  static std::array<std::uint16_t, 5U * kFusedHiddenSize>
      fused_validation_storage{};
  constexpr std::size_t kPreprocessQueryElements = 24U * 256U;
  constexpr std::size_t kPreprocessKeyElements = 4U * 256U;
  static std::array<std::uint16_t,
                    4U * kPreprocessQueryElements +
                        kPreprocessKeyElements + 2U * 256U>
      preprocess_validation_storage{};
  static std::array<float, 64U> preprocess_cosines{};
  static std::array<float, 64U> preprocess_sines{};
  static q3x::runtime::Bf16GreedyArgmaxResult argmax_validation_storage{};
  const std::uint16_t* const fused_left = fused_validation_storage.data();
  const std::uint16_t* const fused_right =
      fused_left + kFusedHiddenSize;
  const std::uint16_t* const fused_weight =
      fused_right + kFusedHiddenSize;
  std::uint16_t* const fused_residual =
      fused_validation_storage.data() + 3U * kFusedHiddenSize;
  std::uint16_t* const fused_normalized =
      fused_validation_storage.data() + 4U * kFusedHiddenSize;
  const std::uint16_t* const preprocess_interleaved =
      preprocess_validation_storage.data();
  std::uint16_t* const preprocess_key =
      preprocess_validation_storage.data() +
      2U * kPreprocessQueryElements;
  const std::uint16_t* const preprocess_q_weight =
      preprocess_key + kPreprocessKeyElements;
  const std::uint16_t* const preprocess_k_weight =
      preprocess_q_weight + 256U;
  std::uint16_t* const preprocess_query =
      const_cast<std::uint16_t*>(preprocess_k_weight + 256U);
  std::uint16_t* const preprocess_gate =
      preprocess_query + kPreprocessQueryElements;
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_embedding_gather_reference_cuda(
                      nullptr, 1U, 1U, 1U, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA embedding rejects out-of-range token");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_centered_rms_norm_reference_cuda(
                      nullptr, nullptr, 1U,
                      std::numeric_limits<float>::quiet_NaN(), nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA RMSNorm rejects NaN epsilon");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
                      nullptr, nullptr, kMaximum, 2U, 1.0e-6F, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA headwise RMSNorm rejects overflow");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
                      nullptr, nullptr, 0U, 5'120U, 1.0e-6F, nullptr)) ==
                  cudaSuccess,
              "empty CUDA headwise RMSNorm is a no-op");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
                      nullptr, nullptr, 8U, 5'120U, 1.0e-6F, nullptr)) ==
                  cudaErrorInvalidValue,
              "non-empty CUDA headwise RMSNorm rejects null storage");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::
                      launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                          nullptr, nullptr, nullptr, 1U, 1U, 1.0e-6F,
                          nullptr)) == cudaErrorInvalidValue,
              "CUDA fused headwise RMSNorm rejects null gate");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                      nullptr, 1U, nullptr)) == cudaErrorInvalidValue,
              "CUDA FP32 conversion rejects null storage");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_bf16_greedy_argmax_cuda(
                      nullptr, 1U, &argmax_validation_storage)) ==
                  cudaErrorInvalidValue,
              "CUDA BF16 greedy argmax rejects null input");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_bf16_greedy_argmax_cuda(
                      reinterpret_cast<const std::uint16_t*>(
                          &argmax_validation_storage),
                      1U, nullptr)) == cudaErrorInvalidValue,
              "CUDA BF16 greedy argmax rejects null result storage");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_bf16_greedy_argmax_cuda(
                      reinterpret_cast<const std::uint16_t*>(
                          &argmax_validation_storage),
                      0U, &argmax_validation_storage)) ==
                  cudaErrorInvalidValue,
              "CUDA BF16 greedy argmax rejects empty input");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_bf16_greedy_argmax_cuda(
                      reinterpret_cast<const std::uint16_t*>(
                          &argmax_validation_storage),
                      sizeof(argmax_validation_storage) /
                          sizeof(std::uint16_t),
                      &argmax_validation_storage)) ==
                  cudaErrorInvalidValue,
              "CUDA BF16 greedy argmax rejects overlapping result");
  if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
    test.expect(static_cast<cudaError_t>(
                    q3x::runtime::launch_bf16_greedy_argmax_cuda(
                        preprocess_interleaved,
                        static_cast<std::size_t>(
                            std::numeric_limits<std::uint32_t>::max()) +
                            1U,
                        &argmax_validation_storage)) ==
                    cudaErrorInvalidValue,
                "CUDA BF16 greedy argmax rejects an unrepresentable index range");
  }
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_sigmoid_gate_reference_cuda(
                      nullptr, nullptr, 1U, nullptr)) == cudaErrorInvalidValue,
              "CUDA sigmoid gate rejects null storage");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_l2_normalize_heads_reference_cuda(
                      nullptr, kMaximum, 2U, 1.0e-6F, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA L2 norm rejects overflow");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_partial_neox_rope_256_64_reference_cuda(
                      nullptr, nullptr, nullptr,
                      kMaximum /
                              q3x::runtime::kFullAttentionHeadDimension +
                          1U,
                      nullptr)) == cudaErrorInvalidValue,
              "CUDA RoPE rejects overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
              nullptr, nullptr, nullptr, nullptr, 0U, 0U)) ==
          cudaErrorInvalidValue,
      "CUDA Q/K RoPE tile rejects M=0");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
              nullptr, nullptr, nullptr, nullptr, 0U,
              q3x::runtime::kQkRopeTileMaximumTokens + 1U)) ==
          cudaErrorInvalidValue,
      "CUDA Q/K RoPE tile rejects M=17");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
              nullptr, nullptr, nullptr, nullptr, 0U, 1U)) ==
          cudaErrorInvalidValue,
      "CUDA Q/K RoPE tile rejects null storage");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
              nullptr, nullptr, nullptr, nullptr, kMaximum, 1U)) ==
          cudaErrorInvalidValue,
      "CUDA Q/K RoPE tile rejects position addition overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
              nullptr, nullptr, nullptr, nullptr,
              kMaximum /
                  ((q3x::runtime::kQwenRotaryDimension / 2U) *
                   sizeof(float)),
              1U)) == cudaErrorInvalidValue,
      "CUDA Q/K RoPE tile rejects table byte-offset overflow");
  const auto launch_preprocess =
      [&](const std::uint16_t* const interleaved_q_gate,
          std::uint16_t* const key,
          const std::uint16_t* const q_weight,
          const std::uint16_t* const k_weight,
          const float epsilon,
          std::uint16_t* const query,
          std::uint16_t* const gate,
          const float* const cosines,
          const float* const sines,
          const std::size_t first_position,
          const std::size_t token_count) {
        return static_cast<cudaError_t>(
            q3x::runtime::
                launch_full_attention_preprocess_24_4_256_64_cuda(
                    interleaved_q_gate, key, q_weight, k_weight, epsilon,
                    query, gate, cosines, sines, first_position,
                    token_count));
      };
  test.expect(
      launch_preprocess(
          preprocess_interleaved, preprocess_key, preprocess_q_weight,
          preprocess_k_weight, 1.0e-6F, preprocess_query, preprocess_gate,
          preprocess_cosines.data(), preprocess_sines.data(), 0U, 0U) ==
          cudaErrorInvalidValue,
      "full-attention preprocess rejects M=0");
  test.expect(
      launch_preprocess(
          preprocess_interleaved, preprocess_key, preprocess_q_weight,
          preprocess_k_weight, 1.0e-6F, preprocess_query, preprocess_gate,
          preprocess_cosines.data(), preprocess_sines.data(), 0U,
          q3x::runtime::kQkRopeTileMaximumTokens + 1U) ==
          cudaErrorInvalidValue,
      "full-attention preprocess rejects M=17");
  test.expect(
      launch_preprocess(
          preprocess_interleaved, preprocess_key, preprocess_q_weight,
          preprocess_k_weight, std::numeric_limits<float>::quiet_NaN(),
          preprocess_query, preprocess_gate, preprocess_cosines.data(),
          preprocess_sines.data(), 0U, 1U) == cudaErrorInvalidValue,
      "full-attention preprocess rejects invalid epsilon");
  test.expect(
      launch_preprocess(
          preprocess_interleaved, preprocess_key, preprocess_q_weight,
          preprocess_k_weight, 1.0e-6F, preprocess_query, preprocess_gate,
          preprocess_cosines.data(), preprocess_sines.data(), kMaximum,
          1U) == cudaErrorInvalidValue,
      "full-attention preprocess rejects position addition overflow");
  test.expect(
      launch_preprocess(
          preprocess_interleaved, preprocess_key, preprocess_q_weight,
          preprocess_k_weight, 1.0e-6F, preprocess_query, preprocess_gate,
          preprocess_cosines.data(), preprocess_sines.data(),
          kMaximum /
              ((q3x::runtime::kQwenRotaryDimension / 2U) * sizeof(float)),
          1U) == cudaErrorInvalidValue,
      "full-attention preprocess rejects table byte-offset overflow");
  test.expect(
      launch_preprocess(
          nullptr, preprocess_key, preprocess_q_weight, preprocess_k_weight,
          1.0e-6F, preprocess_query, preprocess_gate,
          preprocess_cosines.data(), preprocess_sines.data(), 0U, 1U) ==
              cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, nullptr, preprocess_q_weight,
              preprocess_k_weight, 1.0e-6F, preprocess_query,
              preprocess_gate, preprocess_cosines.data(),
              preprocess_sines.data(), 0U, 1U) == cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, preprocess_key, nullptr,
              preprocess_k_weight, 1.0e-6F, preprocess_query,
              preprocess_gate, preprocess_cosines.data(),
              preprocess_sines.data(), 0U, 1U) == cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, preprocess_key, preprocess_q_weight,
              nullptr, 1.0e-6F, preprocess_query, preprocess_gate,
              preprocess_cosines.data(), preprocess_sines.data(), 0U, 1U) ==
              cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, preprocess_key, preprocess_q_weight,
              preprocess_k_weight, 1.0e-6F, nullptr, preprocess_gate,
              preprocess_cosines.data(), preprocess_sines.data(), 0U, 1U) ==
              cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, preprocess_key, preprocess_q_weight,
              preprocess_k_weight, 1.0e-6F, preprocess_query, nullptr,
              preprocess_cosines.data(), preprocess_sines.data(), 0U, 1U) ==
              cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, preprocess_key, preprocess_q_weight,
              preprocess_k_weight, 1.0e-6F, preprocess_query,
              preprocess_gate, nullptr, preprocess_sines.data(), 0U, 1U) ==
              cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, preprocess_key, preprocess_q_weight,
              preprocess_k_weight, 1.0e-6F, preprocess_query,
              preprocess_gate, preprocess_cosines.data(), nullptr, 0U, 1U) ==
              cudaErrorInvalidValue,
      "full-attention preprocess rejects each null storage argument");
  test.expect(
      launch_preprocess(
          preprocess_interleaved, preprocess_key, preprocess_q_weight,
          preprocess_k_weight, 1.0e-6F, preprocess_query, preprocess_query,
          preprocess_cosines.data(), preprocess_sines.data(), 0U, 1U) ==
              cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, preprocess_key, preprocess_q_weight,
              preprocess_k_weight, 1.0e-6F,
              const_cast<std::uint16_t*>(preprocess_interleaved),
              preprocess_gate, preprocess_cosines.data(),
              preprocess_sines.data(), 0U, 1U) == cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved, preprocess_key, preprocess_q_weight,
              preprocess_k_weight, 1.0e-6F, preprocess_key,
              preprocess_gate, preprocess_cosines.data(),
              preprocess_sines.data(), 0U, 1U) == cudaErrorInvalidValue &&
          launch_preprocess(
              preprocess_interleaved,
              const_cast<std::uint16_t*>(preprocess_q_weight),
              preprocess_q_weight, preprocess_k_weight, 1.0e-6F,
              preprocess_query, preprocess_gate, preprocess_cosines.data(),
              preprocess_sines.data(), 0U, 1U) == cudaErrorInvalidValue,
      "full-attention preprocess rejects critical writable aliases");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_softmax_reference_cuda(
                      nullptr, kMaximum, 2U, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA softmax rejects overflow");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_gqa_attention_reference_cuda(
                      nullptr, nullptr, nullptr, 3U, 2U, 1U, 1U, 1.0F,
                      nullptr, 3U, nullptr)) == cudaErrorInvalidValue,
              "CUDA GQA rejects non-divisible head mapping");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_gqa_attention_reference_cuda(
                      nullptr, nullptr, nullptr, 2U, 1U, 3U, 1U, 1.0F,
                      nullptr, 5U, nullptr)) == cudaErrorInvalidValue,
              "CUDA GQA rejects undersized scratch");
  test.expect(static_cast<cudaError_t>(
                  q3x::runtime::launch_residual_add_reference_cuda(
                      nullptr, nullptr, 0U, nullptr)) == cudaSuccess,
              "empty CUDA pointwise op is a no-op");
  const auto launch_fused = [&](const std::uint16_t* const left,
                                const std::uint16_t* const right,
                                const std::uint16_t* const weight,
                                const float epsilon,
                                std::uint16_t* const residual,
                                std::uint16_t* const normalized) {
    return static_cast<cudaError_t>(
        q3x::runtime::launch_residual_add_centered_rms_norm_5120_cuda(
            left, right, weight, epsilon, residual, normalized));
  };
  test.expect(launch_fused(nullptr, fused_right, fused_weight, 1.0e-6F,
                           fused_residual, fused_normalized) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects null left input");
  test.expect(launch_fused(fused_left, nullptr, fused_weight, 1.0e-6F,
                           fused_residual, fused_normalized) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects null right input");
  test.expect(launch_fused(fused_left, fused_right, nullptr, 1.0e-6F,
                           fused_residual, fused_normalized) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects null weight");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           nullptr, fused_normalized) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects null residual output");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           fused_residual, nullptr) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects null normalized output");
  for (const float invalid_epsilon :
       {0.0F, -1.0e-6F, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    test.expect(launch_fused(fused_left, fused_right, fused_weight,
                             invalid_epsilon, fused_residual,
                             fused_normalized) == cudaErrorInvalidValue,
                "fused residual RMSNorm rejects invalid epsilon");
  }
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           fused_residual, fused_residual) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects identical outputs");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           fused_residual, fused_residual + 1U) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects partially overlapping outputs");
  const std::uintptr_t near_max_address =
      std::numeric_limits<std::uintptr_t>::max() - 1U;
  const auto* const near_max_input =
      reinterpret_cast<const std::uint16_t*>(near_max_address);
  auto* const near_max_output =
      reinterpret_cast<std::uint16_t*>(near_max_address);
  test.expect(launch_fused(near_max_input, fused_right, fused_weight, 1.0e-6F,
                           fused_residual, fused_normalized) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects overflowing left range");
  test.expect(launch_fused(fused_left, near_max_input, fused_weight, 1.0e-6F,
                           fused_residual, fused_normalized) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects overflowing right range");
  test.expect(launch_fused(fused_left, fused_right, near_max_input, 1.0e-6F,
                           fused_residual, fused_normalized) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects overflowing weight range");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           near_max_output, fused_normalized) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects overflowing residual range");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           fused_residual, near_max_output) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects overflowing normalized range");

  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           const_cast<std::uint16_t*>(fused_left),
                           fused_normalized) == cudaErrorInvalidValue,
              "fused residual RMSNorm rejects residual/left overlap");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           const_cast<std::uint16_t*>(fused_right),
                           fused_normalized) == cudaErrorInvalidValue,
              "fused residual RMSNorm rejects residual/right overlap");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           const_cast<std::uint16_t*>(fused_weight),
                           fused_normalized) == cudaErrorInvalidValue,
              "fused residual RMSNorm rejects residual/weight overlap");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           fused_residual,
                           const_cast<std::uint16_t*>(fused_left)) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects normalized/left overlap");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           fused_residual,
                           const_cast<std::uint16_t*>(fused_weight)) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects normalized/weight overlap");
  test.expect(launch_fused(fused_left, fused_right, fused_weight, 1.0e-6F,
                           fused_residual,
                           const_cast<std::uint16_t*>(fused_right) + 1U) ==
                  cudaErrorInvalidValue,
              "fused residual RMSNorm rejects normalized/right partial "
              "overlap");
}

void test_residual_rms_m32_launch_validation(TestContext& test) {
  constexpr std::size_t kTokenCount = 32U;
  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr std::size_t kElementCount = kTokenCount * kHiddenSize;
  static std::array<std::uint16_t, 4U * kElementCount + kHiddenSize>
      storage{};
  const std::uint16_t* const left = storage.data();
  const std::uint16_t* const right = left + kElementCount;
  const std::uint16_t* const weight = right + kElementCount;
  std::uint16_t* const residual =
      storage.data() + 2U * kElementCount + kHiddenSize;
  std::uint16_t* const normalized = residual + kElementCount;
  const auto launch = [&](const std::uint16_t* const launch_left,
                          const std::uint16_t* const launch_right,
                          const std::uint16_t* const launch_weight,
                          const std::size_t token_count,
                          const std::size_t hidden_size,
                          const float epsilon,
                          std::uint16_t* const launch_residual,
                          std::uint16_t* const launch_normalized) {
    return static_cast<cudaError_t>(q3x::runtime::
        launch_residual_add_headwise_centered_rms_norm_m32_5120_cuda(
            launch_left, launch_right, launch_weight, token_count,
            hidden_size, epsilon, launch_residual, launch_normalized,
            nullptr));
  };

  test.expect(launch(nullptr, right, weight, kTokenCount, kHiddenSize,
                     1.0e-6F, residual, normalized) ==
                  cudaErrorInvalidValue &&
                  launch(left, nullptr, weight, kTokenCount, kHiddenSize,
                         1.0e-6F, residual, normalized) ==
                      cudaErrorInvalidValue &&
                  launch(left, right, nullptr, kTokenCount, kHiddenSize,
                         1.0e-6F, residual, normalized) ==
                      cudaErrorInvalidValue &&
                  launch(left, right, weight, kTokenCount, kHiddenSize,
                         1.0e-6F, nullptr, normalized) ==
                      cudaErrorInvalidValue &&
                  launch(left, right, weight, kTokenCount, kHiddenSize,
                         1.0e-6F, residual, nullptr) ==
                      cudaErrorInvalidValue,
              "M32 residual RMSNorm rejects each null storage argument");
  constexpr std::array<std::pair<std::size_t, std::size_t>, 6U>
      kInvalidShapes{{{31U, kHiddenSize},
                      {33U, kHiddenSize},
                      {kTokenCount, 5'119U},
                      {kTokenCount, 5'121U},
                      {0U, kHiddenSize},
                      {kTokenCount, 0U}}};
  for (const std::pair<std::size_t, std::size_t>& invalid_shape :
       kInvalidShapes) {
    test.expect(launch(left, right, weight, invalid_shape.first,
                       invalid_shape.second, 1.0e-6F, residual,
                       normalized) == cudaErrorInvalidValue,
                "M32 residual RMSNorm rejects non-fixed shape");
  }
  for (const float invalid_epsilon :
       {0.0F, -1.0e-6F, std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::quiet_NaN()}) {
    test.expect(launch(left, right, weight, kTokenCount, kHiddenSize,
                       invalid_epsilon, residual, normalized) ==
                    cudaErrorInvalidValue,
                "M32 residual RMSNorm rejects invalid epsilon");
  }

  const std::uintptr_t near_max_address =
      std::numeric_limits<std::uintptr_t>::max() - 1U;
  const auto* const near_max_input =
      reinterpret_cast<const std::uint16_t*>(near_max_address);
  auto* const near_max_output =
      reinterpret_cast<std::uint16_t*>(near_max_address);
  test.expect(launch(near_max_input, right, weight, kTokenCount, kHiddenSize,
                     1.0e-6F, residual, normalized) ==
                  cudaErrorInvalidValue &&
                  launch(left, near_max_input, weight, kTokenCount,
                         kHiddenSize, 1.0e-6F, residual, normalized) ==
                      cudaErrorInvalidValue &&
                  launch(left, right, near_max_input, kTokenCount,
                         kHiddenSize, 1.0e-6F, residual, normalized) ==
                      cudaErrorInvalidValue &&
                  launch(left, right, weight, kTokenCount, kHiddenSize,
                         1.0e-6F, near_max_output, normalized) ==
                      cudaErrorInvalidValue &&
                  launch(left, right, weight, kTokenCount, kHiddenSize,
                         1.0e-6F, residual, near_max_output) ==
                      cudaErrorInvalidValue,
              "M32 residual RMSNorm rejects overflowing byte ranges");

  test.expect(
      launch(left, right, weight, kTokenCount, kHiddenSize, 1.0e-6F,
             const_cast<std::uint16_t*>(left), normalized) ==
              cudaErrorInvalidValue &&
          launch(left, right, weight, kTokenCount, kHiddenSize, 1.0e-6F,
                 const_cast<std::uint16_t*>(right), normalized) ==
              cudaErrorInvalidValue &&
          launch(left, right, weight, kTokenCount, kHiddenSize, 1.0e-6F,
                 const_cast<std::uint16_t*>(weight), normalized) ==
              cudaErrorInvalidValue &&
          launch(left, right, weight, kTokenCount, kHiddenSize, 1.0e-6F,
                 residual, residual) == cudaErrorInvalidValue &&
          launch(left, right, weight, kTokenCount, kHiddenSize, 1.0e-6F,
                 residual, residual + 1U) == cudaErrorInvalidValue,
      "M32 residual RMSNorm keeps residual output independent");
  test.expect(
      launch(left, right, weight, kTokenCount, kHiddenSize, 1.0e-6F,
             residual, const_cast<std::uint16_t*>(left)) ==
              cudaErrorInvalidValue &&
          launch(left, right, weight, kTokenCount, kHiddenSize, 1.0e-6F,
                 residual, const_cast<std::uint16_t*>(weight)) ==
              cudaErrorInvalidValue &&
          launch(left, right, weight, kTokenCount, kHiddenSize, 1.0e-6F,
                 residual, const_cast<std::uint16_t*>(right) + 1U) ==
              cudaErrorInvalidValue,
      "M32 residual RMSNorm rejects normalized input overlap except exact "
      "right alias");

  int registers = 0;
  std::size_t shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks = 0;
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          query_residual_add_headwise_centered_rms_norm_m32_5120_test_cuda_resources(
              nullptr, &shared_bytes, &local_bytes, &maximum_threads,
              &active_blocks)) == cudaErrorInvalidValue &&
          static_cast<cudaError_t>(q3x::runtime::
              query_residual_add_headwise_centered_rms_norm_m32_5120_test_cuda_resources(
                  &registers, nullptr, &local_bytes, &maximum_threads,
                  &active_blocks)) == cudaErrorInvalidValue &&
          static_cast<cudaError_t>(q3x::runtime::
              query_residual_add_headwise_centered_rms_norm_m32_5120_test_cuda_resources(
                  &registers, &shared_bytes, nullptr, &maximum_threads,
                  &active_blocks)) == cudaErrorInvalidValue &&
          static_cast<cudaError_t>(q3x::runtime::
              query_residual_add_headwise_centered_rms_norm_m32_5120_test_cuda_resources(
                  &registers, &shared_bytes, &local_bytes, nullptr,
                  &active_blocks)) == cudaErrorInvalidValue &&
          static_cast<cudaError_t>(q3x::runtime::
              query_residual_add_headwise_centered_rms_norm_m32_5120_test_cuda_resources(
                  &registers, &shared_bytes, &local_bytes, &maximum_threads,
                  nullptr)) == cudaErrorInvalidValue,
      "M32 residual RMSNorm resource query rejects null outputs");
}

void test_embedding(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kVocabulary = 7U;
  constexpr std::size_t kHidden = 259U;
  constexpr std::size_t kToken = 6U;
  ManagedBuffer<std::uint16_t> table;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(table.allocate(kVocabulary * kHidden),
                            "embedding allocate table");
  ready = ready &&
          test.cuda_ok(output.allocate(kHidden), "embedding allocate output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < table.size(); ++index) {
    table[index] = encode_bf16(static_cast<float>(index % 101U) / 32.0F);
  }
  ready = launch_after_stale(test, stream, "embedding", [&]() {
    return q3x::runtime::launch_embedding_gather_reference_cuda(
        table.data(), kVocabulary, kHidden, kToken, output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < kHidden; ++index) {
      test.expect(output[index] == table[kToken * kHidden + index],
                  "CUDA embedding element " + std::to_string(index));
    }
  }
}

void test_vector_ops(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kCount = 777U;
  constexpr float kEpsilon = 1.0e-6F;
  ManagedBuffer<std::uint16_t> input;
  ManagedBuffer<std::uint16_t> weight;
  ManagedBuffer<std::uint16_t> other;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(input.allocate(kCount), "vector allocate input");
  ready = ready &&
          test.cuda_ok(weight.allocate(kCount), "vector allocate weight");
  ready = ready &&
          test.cuda_ok(other.allocate(kCount), "vector allocate other");
  ready = ready &&
          test.cuda_ok(output.allocate(kCount), "vector allocate output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < kCount; ++index) {
    input[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 7U) % 47U) - 23) /
        16.0F);
    weight[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 5U) % 17U) - 8) /
        32.0F);
    other[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 3U) % 29U) - 14) /
        16.0F);
  }
  std::vector<std::uint16_t> cpu(kCount);

  (void)q3x::runtime::centered_rms_norm_reference_cpu(
      input.data(), weight.data(), kCount, kEpsilon, cpu.data());
  ready = launch_after_stale(test, stream, "centered RMSNorm", [&]() {
    return q3x::runtime::launch_centered_rms_norm_reference_cuda(
        input.data(), weight.data(), kCount, kEpsilon, output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < kCount; ++index) {
      expect_bf16_match(test, output[index], cpu[index],
                        "CUDA centered RMSNorm " + std::to_string(index));
    }
  }

  (void)q3x::runtime::plain_rms_norm_reference_cpu(
      input.data(), weight.data(), kCount, kEpsilon, cpu.data());
  ready = launch_after_stale(test, stream, "plain RMSNorm", [&]() {
    return q3x::runtime::launch_plain_rms_norm_reference_cuda(
        input.data(), weight.data(), kCount, kEpsilon, output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < kCount; ++index) {
      expect_bf16_match(test, output[index], cpu[index],
                        "CUDA plain RMSNorm " + std::to_string(index));
    }
  }

  (void)q3x::runtime::residual_add_reference_cpu(
      input.data(), other.data(), kCount, cpu.data());
  ready = launch_after_stale(test, stream, "residual add", [&]() {
    return q3x::runtime::launch_residual_add_reference_cuda(
        input.data(), other.data(), kCount, output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < kCount; ++index) {
      test.expect(output[index] == cpu[index],
                  "CUDA residual exact element " + std::to_string(index));
    }
  }

  (void)q3x::runtime::silu_mul_reference_cpu(
      input.data(), other.data(), kCount, cpu.data());
  ready = launch_after_stale(test, stream, "SiLU multiply", [&]() {
    return q3x::runtime::launch_silu_mul_reference_cuda(
        input.data(), other.data(), kCount, output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < kCount; ++index) {
      expect_bf16_match(test, output[index], cpu[index],
                        "CUDA SiLU multiply " + std::to_string(index));
    }
  }

  (void)q3x::runtime::sigmoid_gate_reference_cpu(
      input.data(), other.data(), kCount, cpu.data());
  ready = launch_after_stale(test, stream, "sigmoid gate in-place", [&]() {
    return q3x::runtime::launch_sigmoid_gate_reference_cuda(
        input.data(), other.data(), kCount, input.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < kCount; ++index) {
      expect_bf16_match(test, input[index], cpu[index],
                        "CUDA in-place sigmoid gate " +
                            std::to_string(index));
    }
  }
}

enum class ResidualRmsFixture : std::uint8_t {
  kStructured,
  kRandomLike,
  kNonfinite,
};

[[nodiscard]] const char* residual_rms_fixture_name(
    const ResidualRmsFixture fixture) {
  switch (fixture) {
    case ResidualRmsFixture::kStructured:
      return "structured";
    case ResidualRmsFixture::kRandomLike:
      return "random-like";
    case ResidualRmsFixture::kNonfinite:
      return "nonfinite";
  }
  return "unknown";
}

void fill_residual_rms_fixture(const ResidualRmsFixture fixture,
                               std::uint16_t* const left,
                               std::uint16_t* const right,
                               std::uint16_t* const weight,
                               const std::size_t count) {
  std::uint32_t left_state = 0x1234abcdU;
  std::uint32_t right_state = 0x89abcdefU;
  std::uint32_t weight_state = 0x31415926U;
  const auto advance = [](std::uint32_t& state) {
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
  };
  for (std::size_t index = 0U; index < count; ++index) {
    if (fixture == ResidualRmsFixture::kStructured) {
      const int left_value =
          static_cast<int>((index * 17U + index / 13U) % 257U) - 128;
      const int right_value =
          static_cast<int>((index * 29U + index / 7U) % 193U) - 96;
      const int weight_value = static_cast<int>((index * 11U) % 65U) - 32;
      left[index] = encode_bf16(static_cast<float>(left_value) / 32.0F);
      right[index] = encode_bf16(static_cast<float>(right_value) / 32.0F);
      weight[index] = encode_bf16(static_cast<float>(weight_value) / 128.0F);
    } else {
      const int left_value = static_cast<int>(advance(left_state) % 2049U) -
                             1024;
      const int right_value =
          static_cast<int>(advance(right_state) % 1537U) - 768;
      const int weight_value =
          static_cast<int>(advance(weight_state) % 257U) - 128;
      left[index] = encode_bf16(static_cast<float>(left_value) / 128.0F);
      right[index] = encode_bf16(static_cast<float>(right_value) / 128.0F);
      weight[index] = encode_bf16(static_cast<float>(weight_value) / 256.0F);
    }
  }
  if (fixture != ResidualRmsFixture::kNonfinite) {
    return;
  }

  // Exercise the exact BF16 NaN quieting and infinity propagation performed
  // at the residual boundary, followed by NaN canonicalization in RMSNorm.
  left[0U] = 0x7f81U;
  right[0U] = encode_bf16(1.0F);
  left[1U] = 0x7f80U;
  right[1U] = 0xff80U;
  left[2U] = 0x7f80U;
  right[2U] = encode_bf16(-0.5F);
  left[3U] = 0xff80U;
  right[3U] = encode_bf16(0.25F);
  left[4U] = 0x8000U;
  right[4U] = 0x0000U;
  weight[0U] = 0x7f81U;
  weight[1U] = 0x7f80U;
  weight[2U] = 0xff80U;
}

void run_residual_rms_fused_case(TestContext& test, cudaStream_t stream,
                                 const ResidualRmsFixture fixture) {
  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr float kEpsilon = 1.0e-6F;
  const std::string label =
      std::string("fused residual RMSNorm ") +
      residual_rms_fixture_name(fixture);

  GuardedBf16Buffer left;
  GuardedBf16Buffer right;
  GuardedBf16Buffer weight;
  GuardedBf16Buffer baseline_residual;
  GuardedBf16Buffer baseline_normalized;
  GuardedBf16Buffer candidate_residual;
  GuardedBf16Buffer candidate_normalized;
  GuardedBf16Buffer alias_right;
  GuardedBf16Buffer alias_residual;
  bool ready = left.allocate(test, kHiddenSize, label + " left");
  ready = ready && right.allocate(test, kHiddenSize, label + " right");
  ready = ready && weight.allocate(test, kHiddenSize, label + " weight");
  ready = ready && baseline_residual.allocate(
                       test, kHiddenSize, label + " baseline residual");
  ready = ready && baseline_normalized.allocate(
                       test, kHiddenSize, label + " baseline normalized");
  ready = ready && candidate_residual.allocate(
                       test, kHiddenSize, label + " candidate residual");
  ready = ready && candidate_normalized.allocate(
                       test, kHiddenSize, label + " candidate normalized");
  ready = ready &&
          alias_right.allocate(test, kHiddenSize, label + " alias right");
  ready = ready && alias_residual.allocate(
                       test, kHiddenSize, label + " alias residual");
  if (!ready) {
    return;
  }

  left.initialize(0U, 0x1111U, 0x1112U);
  right.initialize(0U, 0x2221U, 0x2222U);
  weight.initialize(0U, 0x3331U, 0x3332U);
  baseline_residual.initialize(0xa55aU, 0x4411U, 0x4412U);
  baseline_normalized.initialize(0x5aa5U, 0x5511U, 0x5512U);
  candidate_residual.initialize(0xdeadU, 0x6611U, 0x6612U);
  candidate_normalized.initialize(0xbeefU, 0x7711U, 0x7712U);
  alias_right.initialize(0U, 0x8811U, 0x8812U);
  alias_residual.initialize(0xc33cU, 0x9911U, 0x9912U);
  fill_residual_rms_fixture(fixture, left.data(), right.data(), weight.data(),
                            kHiddenSize);
  std::copy_n(right.data(), kHiddenSize, alias_right.data());

  const std::vector<std::uint16_t> left_before = left.snapshot();
  const std::vector<std::uint16_t> right_before = right.snapshot();
  const std::vector<std::uint16_t> weight_before = weight.snapshot();

  ready = launch_after_stale(test, stream, label + " baseline", [&]() {
    const int residual_status =
        q3x::runtime::launch_residual_add_reference_cuda(
            left.data(), right.data(), kHiddenSize, baseline_residual.data(),
            static_cast<void*>(stream));
    if (residual_status != static_cast<int>(cudaSuccess)) {
      return residual_status;
    }
    return q3x::runtime::launch_centered_rms_norm_reference_cuda(
        baseline_residual.data(), weight.data(), kHiddenSize, kEpsilon,
        baseline_normalized.data(), static_cast<void*>(stream));
  });
  ready = ready && launch_after_stale(
                       test, stream, label + " candidate out-of-place", [&]() {
                         return q3x::runtime::
                             launch_residual_add_centered_rms_norm_5120_cuda(
                                 left.data(), right.data(), weight.data(),
                                 kEpsilon, candidate_residual.data(),
                                 candidate_normalized.data(),
                                 static_cast<void*>(stream));
                       });
  if (!ready) {
    return;
  }

  expect_bf16_bits_equal(test, candidate_residual.data(),
                         baseline_residual.data(), kHiddenSize,
                         label + " out-of-place residual");
  expect_bf16_bits_equal(test, candidate_normalized.data(),
                         baseline_normalized.data(), kHiddenSize,
                         label + " out-of-place normalized");
  left.expect_snapshot(test, left_before, label + " left input");
  right.expect_snapshot(test, right_before, label + " right input");
  weight.expect_snapshot(test, weight_before, label + " weight");
  baseline_residual.expect_guards(test, label + " baseline residual");
  baseline_normalized.expect_guards(test, label + " baseline normalized");
  candidate_residual.expect_guards(test, label + " candidate residual");
  candidate_normalized.expect_guards(test, label + " candidate normalized");

  ready = launch_after_stale(test, stream, label + " candidate alias-right",
                             [&]() {
                               return q3x::runtime::
                                   launch_residual_add_centered_rms_norm_5120_cuda(
                                       left.data(), alias_right.data(),
                                       weight.data(), kEpsilon,
                                       alias_residual.data(),
                                       alias_right.data(),
                                       static_cast<void*>(stream));
                             });
  if (!ready) {
    return;
  }
  expect_bf16_bits_equal(test, alias_residual.data(), baseline_residual.data(),
                         kHiddenSize, label + " alias-right residual");
  expect_bf16_bits_equal(test, alias_right.data(), baseline_normalized.data(),
                         kHiddenSize, label + " alias-right normalized");
  left.expect_snapshot(test, left_before, label + " alias left input");
  weight.expect_snapshot(test, weight_before, label + " alias weight");
  alias_residual.expect_guards(test, label + " alias residual");
  alias_right.expect_guards(test, label + " alias normalized/right");
}

void test_residual_rms_fused(TestContext& test, cudaStream_t stream) {
  run_residual_rms_fused_case(test, stream,
                              ResidualRmsFixture::kStructured);
  run_residual_rms_fused_case(test, stream,
                              ResidualRmsFixture::kRandomLike);
  run_residual_rms_fused_case(test, stream,
                              ResidualRmsFixture::kNonfinite);
}

void test_residual_rms_fused_perf(TestContext& test, cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_RESIDUAL_RMS_5120_FUSED_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: fused residual RMSNorm performance gate; set "
                 "Q3X_RUN_RESIDUAL_RMS_5120_FUSED_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr float kEpsilon = 1.0e-6F;
  constexpr std::size_t kWarmupIterations = 128U;
  constexpr std::size_t kMeasuredIterations = 2'048U;
  constexpr int kMeasurementRounds = 3;
  constexpr double kMinimumSpeedup = 1.08;
  const std::string label = "fused residual RMSNorm 5120 perf";

  ManagedBuffer<std::uint16_t> left;
  ManagedBuffer<std::uint16_t> baseline_right;
  ManagedBuffer<std::uint16_t> candidate_right;
  ManagedBuffer<std::uint16_t> weight;
  ManagedBuffer<std::uint16_t> baseline_residual;
  ManagedBuffer<std::uint16_t> candidate_residual;
  bool ready = test.cuda_ok(left.allocate(kHiddenSize), label + " left");
  ready = ready && test.cuda_ok(baseline_right.allocate(kHiddenSize),
                                label + " baseline right/output");
  ready = ready && test.cuda_ok(candidate_right.allocate(kHiddenSize),
                                label + " candidate right/output");
  ready = ready &&
          test.cuda_ok(weight.allocate(kHiddenSize), label + " weight");
  ready = ready && test.cuda_ok(baseline_residual.allocate(kHiddenSize),
                                label + " baseline residual");
  ready = ready && test.cuda_ok(candidate_residual.allocate(kHiddenSize),
                                label + " candidate residual");
  if (!ready) {
    return;
  }

  fill_residual_rms_fixture(ResidualRmsFixture::kRandomLike, left.data(),
                            baseline_right.data(), weight.data(), kHiddenSize);
  std::copy_n(baseline_right.data(), kHiddenSize, candidate_right.data());
  const std::vector<std::uint16_t> left_before(left.data(),
                                               left.data() + kHiddenSize);
  const std::vector<std::uint16_t> weight_before(weight.data(),
                                                 weight.data() + kHiddenSize);
  const auto launch_baseline = [&]() {
    const int residual_status =
        q3x::runtime::launch_residual_add_reference_cuda(
            left.data(), baseline_right.data(), kHiddenSize,
            baseline_residual.data(), static_cast<void*>(stream));
    if (residual_status != static_cast<int>(cudaSuccess)) {
      return residual_status;
    }
    return q3x::runtime::launch_centered_rms_norm_reference_cuda(
        baseline_residual.data(), weight.data(), kHiddenSize, kEpsilon,
        baseline_right.data(), static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() {
    return q3x::runtime::launch_residual_add_centered_rms_norm_5120_cuda(
        left.data(), candidate_right.data(), weight.data(), kEpsilon,
        candidate_residual.data(), candidate_right.data(),
        static_cast<void*>(stream));
  };

  for (std::size_t iteration = 0U;
       ready && iteration < kWarmupIterations; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " baseline warmup");
    ready = ready &&
            test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                         label + " candidate warmup");
  }
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync");
  if (!ready) {
    return;
  }

  double baseline_total = 0.0;
  double candidate_total = 0.0;
  bool timing_finite = true;
  for (int round = 0; round < kMeasurementRounds; ++round) {
    const std::string round_label =
        label + " round=" + std::to_string(round + 1);
    // A/B/B/A balances both launch spans against within-round clock drift.
    const float baseline_first = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, round_label + " baseline pass 1",
        launch_baseline);
    const float candidate_first = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, round_label + " candidate pass 1",
        launch_candidate);
    const float candidate_second = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, round_label + " candidate pass 2",
        launch_candidate);
    const float baseline_second = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, round_label + " baseline pass 2",
        launch_baseline);
    const bool round_finite =
        std::isfinite(baseline_first) && baseline_first > 0.0F &&
        std::isfinite(candidate_first) && candidate_first > 0.0F &&
        std::isfinite(candidate_second) && candidate_second > 0.0F &&
        std::isfinite(baseline_second) && baseline_second > 0.0F;
    timing_finite = timing_finite && round_finite;
    if (round_finite) {
      baseline_total += baseline_first + baseline_second;
      candidate_total += candidate_first + candidate_second;
    }
    std::cout << "PERF_RESIDUAL_RMS_5120_FUSED_ROUND: round=" << round + 1
              << " measured_iterations=" << kMeasuredIterations
              << " baseline_pair_pass1_ms=" << baseline_first
              << " candidate_fused_pass1_ms=" << candidate_first
              << " candidate_fused_pass2_ms=" << candidate_second
              << " baseline_pair_pass2_ms=" << baseline_second << '\n';
  }

  constexpr double kTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const double baseline_milliseconds = baseline_total / kTimedPasses;
  const double candidate_milliseconds = candidate_total / kTimedPasses;
  const double speedup = baseline_milliseconds / candidate_milliseconds;
  const bool gate = timing_finite && std::isfinite(speedup) &&
                    baseline_milliseconds > 0.0 &&
                    candidate_milliseconds > 0.0 &&
                    speedup >= kMinimumSpeedup;
  std::cout << "PERF_RESIDUAL_RMS_5120_FUSED: baseline_pair_span_ms="
            << baseline_milliseconds
            << " candidate_fused_span_ms=" << candidate_milliseconds
            << " speedup=" << speedup
            << " required_speedup=" << kMinimumSpeedup
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, label + " clears the 1.08x total-span gate");

  expect_bf16_bits_equal(test, left.data(), left_before.data(), kHiddenSize,
                         label + " left input");
  expect_bf16_bits_equal(test, weight.data(), weight_before.data(), kHiddenSize,
                         label + " weight");
  expect_bf16_bits_equal(test, candidate_right.data(), baseline_right.data(),
                         kHiddenSize,
                         label + " repeated production-layout normalized");
  expect_bf16_bits_equal(test, candidate_residual.data(),
                         baseline_residual.data(), kHiddenSize,
                         label + " repeated production-layout residual");
}

constexpr std::size_t kResidualRmsM32TokenCount = 32U;
constexpr std::size_t kResidualRmsM32HiddenSize = 5'120U;
constexpr std::size_t kResidualRmsM32ElementCount =
    kResidualRmsM32TokenCount * kResidualRmsM32HiddenSize;
constexpr float kResidualRmsM32Epsilon = 1.0e-6F;

void fill_residual_rms_m32_fixture(const bool stress,
                                   std::uint16_t* const left,
                                   std::uint16_t* const right,
                                   std::uint16_t* const weight) {
  std::uint32_t left_state = stress ? 0x31415926U : 0x9e3779b9U;
  std::uint32_t right_state = stress ? 0x27182818U : 0x243f6a88U;
  for (std::size_t index = 0U; index < kResidualRmsM32ElementCount;
       ++index) {
    const std::size_t token = index / kResidualRmsM32HiddenSize;
    const std::size_t dimension = index - token * kResidualRmsM32HiddenSize;
    const int left_code = static_cast<int>(
                              next_deterministic_random(left_state) % 2049U) -
                          1024;
    const int right_code = static_cast<int>(
                               next_deterministic_random(right_state) %
                               2049U) -
                           1024;
    if (stress) {
      const float scale = (dimension & 1U) == 0U ? 16.0F : 0.0625F;
      const float left_value = static_cast<float>(left_code) * scale / 256.0F;
      const float cancellation =
          static_cast<float>(static_cast<int>((token + dimension) % 9U) - 4) /
          128.0F;
      left[index] = encode_bf16(left_value);
      right[index] = encode_bf16(-left_value + cancellation);
    } else {
      left[index] = encode_bf16(static_cast<float>(left_code) / 192.0F);
      right[index] = encode_bf16(static_cast<float>(right_code) / 256.0F);
    }
  }
  for (std::size_t dimension = 0U;
       dimension < kResidualRmsM32HiddenSize; ++dimension) {
    const int code = static_cast<int>((dimension * 37U + 11U) % 257U) - 128;
    weight[dimension] = encode_bf16(
        static_cast<float>(code) / (stress ? 160.0F : 256.0F));
  }
}

int launch_residual_rms_m32_baseline(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    std::uint16_t* const residual,
    std::uint16_t* const normalized,
    cudaStream_t stream) {
  const int residual_status = q3x::runtime::launch_residual_add_reference_cuda(
      left, right, kResidualRmsM32ElementCount, residual,
      static_cast<void*>(stream));
  if (residual_status != static_cast<int>(cudaSuccess)) {
    return residual_status;
  }
  return q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
      residual, weight, kResidualRmsM32TokenCount,
      kResidualRmsM32HiddenSize, kResidualRmsM32Epsilon, normalized,
      static_cast<void*>(stream));
}

int launch_residual_rms_m32_candidate(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight,
    std::uint16_t* const residual,
    std::uint16_t* const normalized,
    cudaStream_t stream) {
  return q3x::runtime::
      launch_residual_add_headwise_centered_rms_norm_m32_5120_cuda(
          left, right, weight, kResidualRmsM32TokenCount,
          kResidualRmsM32HiddenSize, kResidualRmsM32Epsilon, residual,
          normalized, static_cast<void*>(stream));
}

void test_residual_rms_m32_fused_exact(TestContext& test,
                                       cudaStream_t stream) {
  const std::string label = "M32 residual-add/headwise RMSNorm fusion";
  int registers = 0;
  std::size_t shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks = 0;
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_residual_add_headwise_centered_rms_norm_m32_5120_test_cuda_resources(
              &registers, &shared_bytes, &local_bytes, &maximum_threads,
              &active_blocks)),
      label + " query resources");
  if (!ready) {
    return;
  }
  std::cout << "RESIDUAL_RMS_M32_5120_RESOURCES: registers=" << registers
            << " static_shared=" << shared_bytes
            << " local=" << local_bytes
            << " maximum_threads=" << maximum_threads
            << " active_blocks_per_sm=" << active_blocks << '\n';
  test.expect(shared_bytes ==
                  5'120U * sizeof(std::uint16_t) + 256U * sizeof(float),
              label + " stages BF16 residuals and retains the exact "
                      "256-float reduction tree");
  test.expect(local_bytes == 0U, label + " has no local-memory spills");
  test.expect(registers <= 42, label + " stays below register hard cap");
  test.expect(maximum_threads >= 512,
              label + " supports the fixed 512-thread block");
  test.expect(active_blocks >= 3, label + " retains occupancy floor");

  GuardedBf16Buffer left;
  GuardedBf16Buffer right;
  GuardedBf16Buffer weight;
  GuardedBf16Buffer baseline_residual;
  GuardedBf16Buffer baseline_normalized;
  GuardedBf16Buffer candidate_residual;
  GuardedBf16Buffer candidate_normalized;
  GuardedBf16Buffer replay_residual;
  GuardedBf16Buffer replay_normalized;
  GuardedBf16Buffer alias_right;
  GuardedBf16Buffer alias_residual;
  GuardedBf16Buffer graph_residual;
  GuardedBf16Buffer graph_normalized;
  ready = left.allocate(test, kResidualRmsM32ElementCount, label + " left");
  ready = ready &&
          right.allocate(test, kResidualRmsM32ElementCount, label + " right");
  ready = ready &&
          weight.allocate(test, kResidualRmsM32HiddenSize, label + " weight");
  ready = ready && baseline_residual.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " baseline residual");
  ready = ready && baseline_normalized.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " baseline normalized");
  ready = ready && candidate_residual.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " candidate residual");
  ready = ready && candidate_normalized.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " candidate normalized");
  ready = ready && replay_residual.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " replay residual");
  ready = ready && replay_normalized.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " replay normalized");
  ready = ready && alias_right.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " alias right/normalized");
  ready = ready && alias_residual.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " alias residual");
  ready = ready && graph_residual.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " graph residual");
  ready = ready && graph_normalized.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " graph normalized");
  if (!ready) {
    return;
  }

  left.initialize(0U, 0x1011U, 0x1012U);
  right.initialize(0U, 0x2021U, 0x2022U);
  weight.initialize(0U, 0x3031U, 0x3032U);
  baseline_residual.initialize(0xa1a1U, 0x4011U, 0x4012U);
  baseline_normalized.initialize(0xb2b2U, 0x5021U, 0x5022U);
  candidate_residual.initialize(0xc3c3U, 0x6031U, 0x6032U);
  candidate_normalized.initialize(0xd4d4U, 0x7041U, 0x7042U);
  replay_residual.initialize(0xe5e5U, 0x8051U, 0x8052U);
  replay_normalized.initialize(0xf6f6U, 0x9061U, 0x9062U);
  alias_right.initialize(0U, 0xa071U, 0xa072U);
  alias_residual.initialize(0x1717U, 0xb081U, 0xb082U);
  graph_residual.initialize(0x2828U, 0xc091U, 0xc092U);
  graph_normalized.initialize(0x3939U, 0xd0a1U, 0xd0a2U);
  fill_residual_rms_m32_fixture(false, left.data(), right.data(),
                                weight.data());
  std::copy_n(right.data(), kResidualRmsM32ElementCount, alias_right.data());
  const std::vector<std::uint16_t> left_before = left.snapshot();
  const std::vector<std::uint16_t> right_before = right.snapshot();
  const std::vector<std::uint16_t> weight_before = weight.snapshot();

  ready = launch_after_stale(test, stream, label + " baseline", [&]() {
    return launch_residual_rms_m32_baseline(
        left.data(), right.data(), weight.data(), baseline_residual.data(),
        baseline_normalized.data(), stream);
  });
  ready = ready &&
          launch_after_stale(test, stream, label + " candidate", [&]() {
            return launch_residual_rms_m32_candidate(
                left.data(), right.data(), weight.data(),
                candidate_residual.data(), candidate_normalized.data(),
                stream);
          });
  ready = ready &&
          launch_after_stale(test, stream, label + " direct replay", [&]() {
            return launch_residual_rms_m32_candidate(
                left.data(), right.data(), weight.data(),
                replay_residual.data(), replay_normalized.data(), stream);
          });
  if (!ready) {
    return;
  }
  expect_bf16_bits_equal(test, candidate_residual.data(),
                         baseline_residual.data(),
                         kResidualRmsM32ElementCount,
                         label + " finite residual");
  expect_bf16_bits_equal(test, candidate_normalized.data(),
                         baseline_normalized.data(),
                         kResidualRmsM32ElementCount,
                         label + " finite normalized");
  expect_bf16_bits_equal(test, replay_residual.data(),
                         baseline_residual.data(),
                         kResidualRmsM32ElementCount,
                         label + " direct replay residual");
  expect_bf16_bits_equal(test, replay_normalized.data(),
                         baseline_normalized.data(),
                         kResidualRmsM32ElementCount,
                         label + " direct replay normalized");

  ready = launch_after_stale(test, stream, label + " exact right alias", [&]() {
    return launch_residual_rms_m32_candidate(
        left.data(), alias_right.data(), weight.data(), alias_residual.data(),
        alias_right.data(), stream);
  });
  if (!ready) {
    return;
  }
  expect_bf16_bits_equal(test, alias_residual.data(), baseline_residual.data(),
                         kResidualRmsM32ElementCount,
                         label + " alias-right residual");
  expect_bf16_bits_equal(test, alias_right.data(), baseline_normalized.data(),
                         kResidualRmsM32ElementCount,
                         label + " alias-right normalized");

  cudaGraph_t baseline_graph = nullptr;
  ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      label + " baseline graph begin capture");
  if (ready) {
    const cudaError_t baseline_capture_status =
        static_cast<cudaError_t>(launch_residual_rms_m32_baseline(
            left.data(), right.data(), weight.data(), graph_residual.data(),
            graph_normalized.data(), stream));
    test.expect(baseline_capture_status == cudaSuccess,
                label + " baseline graph launch succeeds");
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &baseline_graph),
                         label + " baseline graph end capture");
  }
  cudaGraphExec_t baseline_graph_exec = nullptr;
  if (ready) {
    std::size_t node_count = 0U;
    ready = test.cuda_ok(
        cudaGraphGetNodes(baseline_graph, nullptr, &node_count),
        label + " baseline graph count nodes");
    test.expect(node_count == 2U,
                label + " baseline graph captures two kernel nodes");
    std::size_t root_count = 0U;
    ready = ready && test.cuda_ok(
                         cudaGraphGetRootNodes(baseline_graph, nullptr,
                                               &root_count),
                         label + " baseline graph count roots");
    test.expect(root_count == 1U,
                label + " baseline graph has one residual root");
    if (ready && root_count == 1U) {
      cudaGraphNode_t root = nullptr;
      std::size_t root_capacity = 1U;
      ready = test.cuda_ok(
          cudaGraphGetRootNodes(baseline_graph, &root, &root_capacity),
          label + " baseline graph fetch root");
      cudaGraphNodeType root_type = cudaGraphNodeTypeEmpty;
      ready = ready && test.cuda_ok(cudaGraphNodeGetType(root, &root_type),
                                    label + " baseline graph root type");
      test.expect(root_type == cudaGraphNodeTypeKernel,
                  label + " baseline root is residual kernel");
      std::array<cudaGraphNode_t, 2U> nodes{};
      std::size_t node_capacity = nodes.size();
      ready = ready && test.cuda_ok(
                           cudaGraphGetNodes(baseline_graph, nodes.data(),
                                             &node_capacity),
                           label + " baseline graph fetch both nodes");
      test.expect(node_capacity == nodes.size(),
                  label + " baseline graph returns both nodes");
      cudaGraphNode_t dependent = nullptr;
      if (ready && node_capacity == nodes.size()) {
        dependent = nodes[0U] == root ? nodes[1U] : nodes[0U];
        test.expect(dependent != nullptr && dependent != root,
                    label + " baseline graph has one non-root node");
      }
      // A two-node DAG with exactly one root necessarily has the non-root as
      // that root's dependent. Checking both launch shapes identifies the
      // residual -> norm edge without relying on CUDA 13's extended
      // cudaGraphNodeGetDependentNodes signature.
      if (ready && dependent != nullptr && dependent != root) {
        cudaGraphNodeType dependent_type = cudaGraphNodeTypeEmpty;
        ready = ready && test.cuda_ok(
                             cudaGraphNodeGetType(dependent, &dependent_type),
                             label + " baseline graph dependent type");
        test.expect(dependent_type == cudaGraphNodeTypeKernel,
                    label + " baseline dependent is norm kernel");
        cudaKernelNodeParams root_parameters{};
        cudaKernelNodeParams dependent_parameters{};
        ready = ready && test.cuda_ok(
                             cudaGraphKernelNodeGetParams(
                                 root, &root_parameters),
                             label + " baseline graph root parameters");
        ready = ready && test.cuda_ok(
                             cudaGraphKernelNodeGetParams(
                                 dependent, &dependent_parameters),
                             label + " baseline graph dependent parameters");
        if (ready) {
          test.expect(root_parameters.gridDim.x == 640U &&
                          root_parameters.gridDim.y == 1U &&
                          root_parameters.gridDim.z == 1U &&
                          root_parameters.blockDim.x == 256U &&
                          root_parameters.blockDim.y == 1U &&
                          root_parameters.blockDim.z == 1U &&
                          root_parameters.sharedMemBytes == 0U,
                      label + " baseline root is the 640-CTA residual add");
          test.expect(dependent_parameters.gridDim.x == 32U &&
                          dependent_parameters.gridDim.y == 1U &&
                          dependent_parameters.gridDim.z == 1U &&
                          dependent_parameters.blockDim.x == 256U &&
                          dependent_parameters.blockDim.y == 1U &&
                          dependent_parameters.blockDim.z == 1U &&
                          dependent_parameters.sharedMemBytes == 0U,
                      label + " baseline dependent is the 32-CTA norm");
        }
      } else {
        ready = false;
      }
    } else {
      ready = false;
    }
  }
  if (ready) {
    ready = test.cuda_ok(
        cudaGraphInstantiate(&baseline_graph_exec, baseline_graph, nullptr,
                             nullptr, 0U),
        label + " baseline graph instantiate");
  }
  if (ready) {
    ready = test.cuda_ok(cudaGraphLaunch(baseline_graph_exec, stream),
                         label + " baseline graph replay");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  label + " baseline graph synchronize");
  }
  if (baseline_graph_exec != nullptr) {
    (void)test.cuda_ok(cudaGraphExecDestroy(baseline_graph_exec),
                       label + " baseline graph exec destroy");
  }
  if (baseline_graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(baseline_graph),
                       label + " baseline graph destroy");
  }
  if (ready) {
    expect_bf16_bits_equal(test, graph_residual.data(),
                           baseline_residual.data(),
                           kResidualRmsM32ElementCount,
                           label + " baseline graph residual");
    expect_bf16_bits_equal(test, graph_normalized.data(),
                           baseline_normalized.data(),
                           kResidualRmsM32ElementCount,
                           label + " baseline graph normalized");
  }

  cudaGraph_t graph = nullptr;
  ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      label + " graph begin capture");
  cudaError_t capture_launch = cudaErrorInvalidValue;
  if (ready) {
    capture_launch = static_cast<cudaError_t>(launch_residual_rms_m32_candidate(
        left.data(), right.data(), weight.data(), graph_residual.data(),
        graph_normalized.data(), stream));
    test.expect(capture_launch == cudaSuccess,
                label + " graph candidate launch succeeds");
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         label + " graph end capture");
  }
  cudaGraphExec_t graph_exec = nullptr;
  if (ready) {
    std::size_t node_count = 0U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                         label + " graph count nodes");
    test.expect(node_count == 1U, label + " graph captures one node");
    std::size_t root_count = 0U;
    ready = ready && test.cuda_ok(
                         cudaGraphGetRootNodes(graph, nullptr, &root_count),
                         label + " graph count roots");
    test.expect(root_count == 1U,
                label + " candidate graph has one kernel root");
    if (ready && node_count == 1U && root_count == 1U) {
      cudaGraphNode_t node = nullptr;
      std::size_t capacity = 1U;
      ready = test.cuda_ok(cudaGraphGetNodes(graph, &node, &capacity),
                           label + " graph fetch node");
      cudaGraphNodeType node_type = cudaGraphNodeTypeEmpty;
      ready = ready && test.cuda_ok(cudaGraphNodeGetType(node, &node_type),
                                    label + " graph read node type");
      test.expect(node_type == cudaGraphNodeTypeKernel,
                  label + " graph node is a kernel");
      cudaKernelNodeParams parameters{};
      ready = ready && test.cuda_ok(
                           cudaGraphKernelNodeGetParams(node, &parameters),
                           label + " graph read kernel parameters");
      if (ready) {
        test.expect(parameters.gridDim.x == 32U &&
                        parameters.gridDim.y == 1U &&
                        parameters.gridDim.z == 1U,
                    label + " graph grid is exactly 32x1x1");
        test.expect(parameters.blockDim.x == 512U &&
                        parameters.blockDim.y == 1U &&
                        parameters.blockDim.z == 1U,
                    label + " graph block is exactly 512x1x1");
        test.expect(parameters.sharedMemBytes == 0U,
                    label + " graph uses no dynamic shared memory");
      }
    } else {
      ready = false;
    }
  }
  if (ready) {
    ready = test.cuda_ok(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0U),
        label + " graph instantiate");
  }
  bool graph_launch_queued = false;
  for (int replay = 0; ready && replay < 2; ++replay) {
    const cudaError_t replay_status = cudaGraphLaunch(graph_exec, stream);
    graph_launch_queued = graph_launch_queued || replay_status == cudaSuccess;
    ready = test.cuda_ok(replay_status, label + " graph replay " +
                                            std::to_string(replay + 1));
  }
  if (graph_launch_queued) {
    const bool synchronize_ok = test.cuda_ok(
        cudaStreamSynchronize(stream), label + " graph replay synchronize");
    ready = synchronize_ok && ready;
  }
  if (graph_exec != nullptr) {
    (void)test.cuda_ok(cudaGraphExecDestroy(graph_exec),
                       label + " graph exec destroy");
  }
  if (graph != nullptr) {
    (void)test.cuda_ok(cudaGraphDestroy(graph), label + " graph destroy");
  }
  if (ready) {
    expect_bf16_bits_equal(test, graph_residual.data(),
                           baseline_residual.data(),
                           kResidualRmsM32ElementCount,
                           label + " graph replay residual");
    expect_bf16_bits_equal(test, graph_normalized.data(),
                           baseline_normalized.data(),
                           kResidualRmsM32ElementCount,
                           label + " graph replay normalized");
  }

  struct InvalidGraphCase {
    const char* name;
    const std::uint16_t* left;
    const std::uint16_t* right;
    const std::uint16_t* weight;
    std::size_t token_count;
    std::size_t hidden_size;
    float epsilon;
    std::uint16_t* residual;
    std::uint16_t* normalized;
  };
  const std::uintptr_t near_max_address =
      std::numeric_limits<std::uintptr_t>::max() - 1U;
  const auto* const overflowing_left =
      reinterpret_cast<const std::uint16_t*>(near_max_address);
  const std::array<InvalidGraphCase, 7U> invalid_cases{{
      {"null-left", nullptr, right.data(), weight.data(),
       kResidualRmsM32TokenCount, kResidualRmsM32HiddenSize,
       kResidualRmsM32Epsilon, graph_residual.data(), graph_normalized.data()},
      {"M31", left.data(), right.data(), weight.data(), 31U,
       kResidualRmsM32HiddenSize, kResidualRmsM32Epsilon,
       graph_residual.data(), graph_normalized.data()},
      {"hidden-5119", left.data(), right.data(), weight.data(),
       kResidualRmsM32TokenCount, 5'119U, kResidualRmsM32Epsilon,
       graph_residual.data(), graph_normalized.data()},
      {"epsilon-zero", left.data(), right.data(), weight.data(),
       kResidualRmsM32TokenCount, kResidualRmsM32HiddenSize, 0.0F,
       graph_residual.data(), graph_normalized.data()},
      {"overflow-left", overflowing_left, right.data(), weight.data(),
       kResidualRmsM32TokenCount, kResidualRmsM32HiddenSize,
       kResidualRmsM32Epsilon, graph_residual.data(), graph_normalized.data()},
      {"residual-alias-left", left.data(), right.data(), weight.data(),
       kResidualRmsM32TokenCount, kResidualRmsM32HiddenSize,
       kResidualRmsM32Epsilon, const_cast<std::uint16_t*>(left.data()),
       graph_normalized.data()},
      {"normalized-partial-alias-right", left.data(), right.data(),
       weight.data(), kResidualRmsM32TokenCount, kResidualRmsM32HiddenSize,
       kResidualRmsM32Epsilon, graph_residual.data(), right.data() + 1U},
  }};
  const auto expect_invalid_graph = [&](const InvalidGraphCase& invalid_case) {
    const std::string case_label =
        label + " invalid graph " + invalid_case.name;
    if (!test.cuda_ok(
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
            case_label + " begin capture")) {
      return;
    }
    const cudaError_t invalid_status = static_cast<cudaError_t>(q3x::runtime::
        launch_residual_add_headwise_centered_rms_norm_m32_5120_cuda(
            invalid_case.left, invalid_case.right, invalid_case.weight,
            invalid_case.token_count, invalid_case.hidden_size,
            invalid_case.epsilon, invalid_case.residual,
            invalid_case.normalized, static_cast<void*>(stream)));
    test.expect(invalid_status == cudaErrorInvalidValue,
                case_label + " returns cudaErrorInvalidValue");
    cudaGraph_t invalid_graph = nullptr;
    const bool invalid_ready = test.cuda_ok(
        cudaStreamEndCapture(stream, &invalid_graph),
        case_label + " end capture");
    if (invalid_ready) {
      std::size_t invalid_nodes = 0U;
      if (test.cuda_ok(
              cudaGraphGetNodes(invalid_graph, nullptr, &invalid_nodes),
              case_label + " count nodes")) {
        test.expect(invalid_nodes == 0U,
                    case_label + " captures zero graph nodes");
      }
    }
    if (invalid_graph != nullptr) {
      (void)test.cuda_ok(cudaGraphDestroy(invalid_graph),
                         case_label + " destroy graph");
    }
  };
  for (const InvalidGraphCase& invalid_case : invalid_cases) {
    expect_invalid_graph(invalid_case);
  }

  left.expect_snapshot(test, left_before, label + " preserves left input");
  right.expect_snapshot(test, right_before, label + " preserves right input");
  weight.expect_snapshot(test, weight_before,
                         label + " preserves shared weight");
  baseline_residual.expect_guards(test, label + " baseline residual");
  baseline_normalized.expect_guards(test, label + " baseline normalized");
  candidate_residual.expect_guards(test, label + " candidate residual");
  candidate_normalized.expect_guards(test, label + " candidate normalized");
  replay_residual.expect_guards(test, label + " replay residual");
  replay_normalized.expect_guards(test, label + " replay normalized");
  alias_residual.expect_guards(test, label + " alias residual");
  alias_right.expect_guards(test, label + " alias right/normalized");
  graph_residual.expect_guards(test, label + " graph residual");
  graph_normalized.expect_guards(test, label + " graph normalized");
}

void fill_residual_rms_m32_prefix_tail_fixture(
    const std::size_t token_count, std::uint16_t* const left,
    std::uint16_t* const right, std::uint16_t* const weight) {
  std::uint32_t left_state = 0x7f4a7c15U;
  std::uint32_t right_state = 0x94d049bbU;
  const std::size_t element_count =
      token_count * kResidualRmsM32HiddenSize;
  for (std::size_t index = 0U; index < element_count; ++index) {
    const int left_code = static_cast<int>(
                              next_deterministic_random(left_state) % 2049U) -
                          1024;
    const int right_code = static_cast<int>(
                               next_deterministic_random(right_state) %
                               2049U) -
                           1024;
    left[index] = encode_bf16(static_cast<float>(left_code) / 192.0F);
    right[index] = encode_bf16(static_cast<float>(right_code) / 256.0F);
  }
  for (std::size_t dimension = 0U;
       dimension < kResidualRmsM32HiddenSize; ++dimension) {
    const int code = static_cast<int>((dimension * 53U + 19U) % 257U) - 128;
    weight[dimension] = encode_bf16(static_cast<float>(code) / 256.0F);
  }
}

int launch_residual_rms_m32_prefix_tail_baseline(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight, const std::size_t token_count,
    std::uint16_t* const residual, std::uint16_t* const normalized,
    cudaStream_t stream) {
  const int residual_status = q3x::runtime::launch_residual_add_reference_cuda(
      left, right, token_count * kResidualRmsM32HiddenSize, residual,
      static_cast<void*>(stream));
  if (residual_status != static_cast<int>(cudaSuccess)) {
    return residual_status;
  }
  return q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
      residual, weight, token_count, kResidualRmsM32HiddenSize,
      kResidualRmsM32Epsilon, normalized, static_cast<void*>(stream));
}

int launch_residual_rms_m32_prefix_tail_candidate(
    const std::uint16_t* const left,
    const std::uint16_t* const right,
    const std::uint16_t* const weight, const std::size_t token_count,
    std::uint16_t* const residual, std::uint16_t* const normalized,
    cudaStream_t stream) {
  const std::size_t prefix_tokens =
      token_count - token_count % kResidualRmsM32TokenCount;
  for (std::size_t token_offset = 0U; token_offset < prefix_tokens;
       token_offset += kResidualRmsM32TokenCount) {
    const std::size_t element_offset =
        token_offset * kResidualRmsM32HiddenSize;
    const int status = q3x::runtime::
        launch_residual_add_headwise_centered_rms_norm_m32_5120_cuda(
            left + element_offset, right + element_offset, weight,
            kResidualRmsM32TokenCount, kResidualRmsM32HiddenSize,
            kResidualRmsM32Epsilon, residual + element_offset,
            normalized + element_offset, static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  const std::size_t tail_tokens = token_count - prefix_tokens;
  if (tail_tokens == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  const std::size_t element_offset =
      prefix_tokens * kResidualRmsM32HiddenSize;
  const int residual_status = q3x::runtime::launch_residual_add_reference_cuda(
      left + element_offset, right + element_offset,
      tail_tokens * kResidualRmsM32HiddenSize, residual + element_offset,
      static_cast<void*>(stream));
  if (residual_status != static_cast<int>(cudaSuccess)) {
    return residual_status;
  }
  return q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
      residual + element_offset, weight, tail_tokens,
      kResidualRmsM32HiddenSize, kResidualRmsM32Epsilon,
      normalized + element_offset, static_cast<void*>(stream));
}

void test_residual_rms_m32_prefix_tail_exact(TestContext& test,
                                             cudaStream_t stream) {
  constexpr std::array<std::size_t, 4U> kTokenCounts{33U, 63U, 407U, 481U};
  for (const std::size_t token_count : kTokenCounts) {
    const std::string label =
        "M32 residual/RMS prefix-tail M" + std::to_string(token_count);
    const std::size_t element_count =
        token_count * kResidualRmsM32HiddenSize;
    GuardedBf16Buffer left;
    GuardedBf16Buffer right;
    GuardedBf16Buffer weight;
    GuardedBf16Buffer baseline_residual;
    GuardedBf16Buffer baseline_normalized;
    GuardedBf16Buffer candidate_residual;
    GuardedBf16Buffer alias_right;
    bool ready = left.allocate(test, element_count, label + " left");
    ready = ready && right.allocate(test, element_count, label + " right");
    ready = ready &&
            weight.allocate(test, kResidualRmsM32HiddenSize,
                            label + " weight");
    ready = ready && baseline_residual.allocate(
                         test, element_count, label + " baseline residual");
    ready = ready && baseline_normalized.allocate(
                         test, element_count, label + " baseline normalized");
    ready = ready && candidate_residual.allocate(
                         test, element_count, label + " candidate residual");
    ready = ready &&
            alias_right.allocate(test, element_count,
                                 label + " alias right/normalized");
    if (!ready) {
      return;
    }

    left.initialize(0U, 0x1111U, 0x1112U);
    right.initialize(0U, 0x2221U, 0x2222U);
    weight.initialize(0U, 0x3331U, 0x3332U);
    baseline_residual.initialize(0xa1a1U, 0x4441U, 0x4442U);
    baseline_normalized.initialize(0xb2b2U, 0x5551U, 0x5552U);
    candidate_residual.initialize(0xc3c3U, 0x6661U, 0x6662U);
    alias_right.initialize(0U, 0x7771U, 0x7772U);
    fill_residual_rms_m32_prefix_tail_fixture(
        token_count, left.data(), right.data(), weight.data());
    std::copy_n(right.data(), element_count, alias_right.data());
    const std::vector<std::uint16_t> left_before = left.snapshot();
    const std::vector<std::uint16_t> right_before = right.snapshot();
    const std::vector<std::uint16_t> weight_before = weight.snapshot();

    ready = launch_after_stale(test, stream, label + " baseline", [&]() {
      return launch_residual_rms_m32_prefix_tail_baseline(
          left.data(), right.data(), weight.data(), token_count,
          baseline_residual.data(), baseline_normalized.data(), stream);
    });
    ready = ready &&
            launch_after_stale(test, stream, label + " candidate alias-right",
                               [&]() {
                                 return launch_residual_rms_m32_prefix_tail_candidate(
                                     left.data(), alias_right.data(),
                                     weight.data(), token_count,
                                     candidate_residual.data(),
                                     alias_right.data(), stream);
                               });
    if (!ready) {
      return;
    }
    expect_bf16_bits_equal(test, candidate_residual.data(),
                           baseline_residual.data(), element_count,
                           label + " residual");
    expect_bf16_bits_equal(test, alias_right.data(),
                           baseline_normalized.data(), element_count,
                           label + " normalized alias-right");
    left.expect_snapshot(test, left_before, label + " preserves left");
    right.expect_snapshot(test, right_before, label + " preserves right");
    weight.expect_snapshot(test, weight_before, label + " preserves weight");
    baseline_residual.expect_guards(test, label + " baseline residual");
    baseline_normalized.expect_guards(test, label + " baseline normalized");
    candidate_residual.expect_guards(test, label + " candidate residual");
    alias_right.expect_guards(test, label + " alias right/normalized");
  }
}

void test_residual_rms_m32_nan_operand_order(TestContext& test,
                                              cudaStream_t stream) {
  const std::string label = "M32 residual RMSNorm NaN operand order";
  GuardedBf16Buffer left;
  GuardedBf16Buffer right;
  GuardedBf16Buffer weight;
  GuardedBf16Buffer baseline_residual;
  GuardedBf16Buffer baseline_normalized;
  GuardedBf16Buffer candidate_residual;
  GuardedBf16Buffer candidate_normalized;
  GuardedBf16Buffer swapped_residual;
  bool ready = left.allocate(test, kResidualRmsM32ElementCount,
                             label + " left");
  ready = ready && right.allocate(test, kResidualRmsM32ElementCount,
                                  label + " right");
  ready = ready &&
          weight.allocate(test, kResidualRmsM32HiddenSize, label + " weight");
  ready = ready && baseline_residual.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " baseline residual");
  ready = ready && baseline_normalized.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " baseline normalized");
  ready = ready && candidate_residual.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " candidate residual");
  ready = ready && candidate_normalized.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " candidate normalized");
  ready = ready && swapped_residual.allocate(
                       test, kResidualRmsM32ElementCount,
                       label + " swapped-order residual");
  if (!ready) {
    return;
  }
  left.initialize(0U, 0x1211U, 0x1212U);
  right.initialize(0U, 0x2321U, 0x2322U);
  weight.initialize(0U, 0x3431U, 0x3432U);
  baseline_residual.initialize(0x4545U, 0x4541U, 0x4542U);
  baseline_normalized.initialize(0x5656U, 0x5651U, 0x5652U);
  candidate_residual.initialize(0x6767U, 0x6761U, 0x6762U);
  candidate_normalized.initialize(0x7878U, 0x7871U, 0x7872U);
  swapped_residual.initialize(0x8989U, 0x8981U, 0x8982U);
  fill_residual_rms_m32_fixture(false, left.data(), right.data(),
                                weight.data());
  constexpr std::size_t kBf16NanCount = 2U * 127U;
  constexpr std::size_t kNanPairCount = kBf16NanCount * kBf16NanCount;
  static_assert(kNanPairCount <= kResidualRmsM32ElementCount);
  // Exhaust the complete signed BF16 signaling/quiet NaN payload cross
  // product. This is stronger than selecting a few payloads: candidate bits
  // must match the runtime residual-left + projection-right baseline for
  // every pair. The reverse-order launch records whether operand order is
  // observable on this compiled SM87 path; it is not itself the oracle.
  std::size_t nan_pair = 0U;
  for (unsigned int left_sign = 0U; left_sign < 2U; ++left_sign) {
    for (unsigned int left_payload = 1U; left_payload < 128U;
         ++left_payload) {
      const std::uint16_t left_bits = static_cast<std::uint16_t>(
          (left_sign << 15U) | 0x7f80U | left_payload);
      for (unsigned int right_sign = 0U; right_sign < 2U; ++right_sign) {
        for (unsigned int right_payload = 1U; right_payload < 128U;
             ++right_payload) {
          left.data()[nan_pair] = left_bits;
          right.data()[nan_pair] = static_cast<std::uint16_t>(
              (right_sign << 15U) | 0x7f80U | right_payload);
          ++nan_pair;
        }
      }
    }
  }
  test.expect(nan_pair == kNanPairCount,
              label + " constructs the complete BF16 NaN cross product");
  const std::vector<std::uint16_t> left_before = left.snapshot();
  const std::vector<std::uint16_t> right_before = right.snapshot();
  const std::vector<std::uint16_t> weight_before = weight.snapshot();

  ready = launch_after_stale(test, stream, label + " runtime-order baseline",
                             [&]() {
                               return launch_residual_rms_m32_baseline(
                                   left.data(), right.data(), weight.data(),
                                   baseline_residual.data(),
                                   baseline_normalized.data(), stream);
                             });
  ready = ready && launch_after_stale(
                       test, stream, label + " runtime-order candidate", [&]() {
                         return launch_residual_rms_m32_candidate(
                             left.data(), right.data(), weight.data(),
                             candidate_residual.data(),
                             candidate_normalized.data(), stream);
                       });
  ready = ready && launch_after_stale(
                       test, stream, label + " reversed-order diagnostic", [&]() {
                         return q3x::runtime::launch_residual_add_reference_cuda(
                             right.data(), left.data(),
                             kResidualRmsM32ElementCount,
                             swapped_residual.data(),
                             static_cast<void*>(stream));
                       });
  if (!ready) {
    return;
  }
  expect_bf16_bits_equal(test, candidate_residual.data(),
                         baseline_residual.data(),
                         kResidualRmsM32ElementCount,
                         label + " runtime-order residual");
  expect_bf16_bits_equal(test, candidate_normalized.data(),
                         baseline_normalized.data(),
                         kResidualRmsM32ElementCount,
                         label + " runtime-order normalized");
  std::size_t exact_nan_matches = 0U;
  std::size_t reversed_order_differences = 0U;
  for (std::size_t index = 0U; index < kNanPairCount; ++index) {
    const std::uint16_t baseline_bits = baseline_residual.data()[index];
    const bool is_nan = (baseline_bits & 0x7f80U) == 0x7f80U &&
                        (baseline_bits & 0x007fU) != 0U;
    exact_nan_matches +=
        is_nan && candidate_residual.data()[index] == baseline_bits ? 1U : 0U;
    reversed_order_differences +=
        swapped_residual.data()[index] != baseline_bits ? 1U : 0U;
  }
  test.expect(exact_nan_matches == kNanPairCount,
              label + " matches every signed BF16 NaN-pair residual bit");
  std::cout << "RESIDUAL_RMS_M32_5120_NAN_ORDER: runtime_order="
               "residual_left_plus_projection_right exact_nan_matches="
            << exact_nan_matches << '/' << kNanPairCount
            << " reversed_order_bit_differences="
            << reversed_order_differences << '/' << kNanPairCount
            << " reversed_order_observable="
            << (reversed_order_differences != 0U ? "yes" : "no")
            << " gate="
            << (exact_nan_matches == kNanPairCount ? "PASS" : "FAIL")
            << '\n';
  left.expect_snapshot(test, left_before, label + " preserves left input");
  right.expect_snapshot(test, right_before, label + " preserves right input");
  weight.expect_snapshot(test, weight_before, label + " preserves weight");
  baseline_residual.expect_guards(test, label + " baseline residual");
  baseline_normalized.expect_guards(test, label + " baseline normalized");
  candidate_residual.expect_guards(test, label + " candidate residual");
  candidate_normalized.expect_guards(test, label + " candidate normalized");
  swapped_residual.expect_guards(test, label + " reversed-order residual");
}

struct PrefillM32ResidualRmsTiming {
  std::array<double, 6U> round_speedups{};
  double baseline_mean_milliseconds =
      std::numeric_limits<double>::quiet_NaN();
  double candidate_mean_milliseconds =
      std::numeric_limits<double>::quiet_NaN();
  double mean_speedup = std::numeric_limits<double>::quiet_NaN();
  double median_speedup = std::numeric_limits<double>::quiet_NaN();
  double minimum_speedup = std::numeric_limits<double>::quiet_NaN();
  bool timings_finite = false;
  bool every_round_nonregressing = false;
  bool final_state_bitwise = false;
  bool inputs_preserved = false;
  bool gate = false;
};

PrefillM32ResidualRmsTiming benchmark_prefill_m32_residual_rms_fixture(
    TestContext& test,
    cudaStream_t stream,
    const std::string& fixture_name,
    ManagedBuffer<std::uint16_t>& left,
    ManagedBuffer<std::uint16_t>& right_seed,
    ManagedBuffer<std::uint16_t>& weight,
    ManagedBuffer<std::uint16_t>& baseline_right,
    ManagedBuffer<std::uint16_t>& candidate_right,
    ManagedBuffer<std::uint16_t>& baseline_residual,
    ManagedBuffer<std::uint16_t>& candidate_residual) {
  constexpr int kWarmupChains = 128;
  constexpr std::size_t kChainsPerPass = 512U;
  constexpr int kRounds = 6;
  constexpr double kRequiredMedianSpeedup = 1.15;
  PrefillM32ResidualRmsTiming timing;
  const std::string label =
      "Prefill M32 residual RMS formal perf " + fixture_name;

  std::copy_n(right_seed.data(), kResidualRmsM32ElementCount,
              baseline_right.data());
  std::copy_n(right_seed.data(), kResidualRmsM32ElementCount,
              candidate_right.data());
  const std::vector<std::uint16_t> left_before(
      left.data(), left.data() + kResidualRmsM32ElementCount);
  const std::vector<std::uint16_t> right_seed_before(
      right_seed.data(), right_seed.data() + kResidualRmsM32ElementCount);
  const std::vector<std::uint16_t> weight_before(
      weight.data(), weight.data() + kResidualRmsM32HiddenSize);

  const auto launch_baseline = [&]() {
    return launch_residual_rms_m32_baseline(
        left.data(), baseline_right.data(), weight.data(),
        baseline_residual.data(), baseline_right.data(), stream);
  };
  const auto launch_candidate = [&]() {
    return launch_residual_rms_m32_candidate(
        left.data(), candidate_right.data(), weight.data(),
        candidate_residual.data(), candidate_right.data(), stream);
  };

  bool warmup_ready = true;
  for (int chain = 0; chain < kWarmupChains && warmup_ready; ++chain) {
    warmup_ready = test.cuda_ok(
        static_cast<cudaError_t>(launch_baseline()),
        label + " alternating baseline warmup chain=" +
            std::to_string(chain + 1));
    if (warmup_ready) {
      warmup_ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_candidate()),
          label + " alternating candidate warmup chain=" +
              std::to_string(chain + 1));
    }
  }
  const bool warmup_synchronize_ok = test.cuda_ok(
      cudaStreamSynchronize(stream), label + " warmup synchronize");
  warmup_ready = warmup_synchronize_ok && warmup_ready;
  if (!warmup_ready) {
    return timing;
  }

  std::array<float, kRounds> baseline_first{};
  std::array<float, kRounds> candidate_first{};
  std::array<float, kRounds> candidate_second{};
  std::array<float, kRounds> baseline_second{};
  bool all_passes_finite = true;
  for (int round = 0; round < kRounds; ++round) {
    const std::size_t index = static_cast<std::size_t>(round);
    const std::string round_label =
        label + " round=" + std::to_string(round + 1);
    baseline_first[index] = measure_cuda_span_milliseconds(
        test, stream, kChainsPerPass, round_label + " B1", launch_baseline);
    candidate_first[index] = measure_cuda_span_milliseconds(
        test, stream, kChainsPerPass, round_label + " C1", launch_candidate);
    candidate_second[index] = measure_cuda_span_milliseconds(
        test, stream, kChainsPerPass, round_label + " C2", launch_candidate);
    baseline_second[index] = measure_cuda_span_milliseconds(
        test, stream, kChainsPerPass, round_label + " B2", launch_baseline);
    const bool round_passes_finite =
        std::isfinite(baseline_first[index]) &&
        baseline_first[index] > 0.0F &&
        std::isfinite(candidate_first[index]) &&
        candidate_first[index] > 0.0F &&
        std::isfinite(candidate_second[index]) &&
        candidate_second[index] > 0.0F &&
        std::isfinite(baseline_second[index]) &&
        baseline_second[index] > 0.0F;
    all_passes_finite = all_passes_finite && round_passes_finite;
    std::cout << "PREFILL_M32_RESIDUAL_RMS_PERF_PASS: fixture="
              << fixture_name << " round=" << round + 1
              << " order=B-C-C-B chains_per_pass=" << kChainsPerPass
              << " B1_chain_ms=" << baseline_first[index]
              << " B1_span_ms="
              << baseline_first[index] * static_cast<float>(kChainsPerPass)
              << " C1_chain_ms=" << candidate_first[index]
              << " C1_span_ms="
              << candidate_first[index] * static_cast<float>(kChainsPerPass)
              << " C2_chain_ms=" << candidate_second[index]
              << " C2_span_ms="
              << candidate_second[index] * static_cast<float>(kChainsPerPass)
              << " B2_chain_ms=" << baseline_second[index]
              << " B2_span_ms="
              << baseline_second[index] * static_cast<float>(kChainsPerPass)
              << " finite_positive="
              << (round_passes_finite ? "true" : "false") << '\n';
  }
  test.expect(all_passes_finite,
              label + " records only finite positive B-C-C-B passes");
  if (!all_passes_finite) {
    std::cout << "PREFILL_M32_RESIDUAL_RMS_PERF_FIXTURE: fixture="
              << fixture_name << " timings_finite=false gate=FAIL\n";
    return timing;
  }

  timing.timings_finite = true;
  timing.every_round_nonregressing = true;
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  double speedup_sum = 0.0;
  for (int round = 0; round < kRounds; ++round) {
    const std::size_t index = static_cast<std::size_t>(round);
    const double baseline_pair =
        (static_cast<double>(baseline_first[index]) +
         static_cast<double>(baseline_second[index])) /
        2.0;
    const double candidate_pair =
        (static_cast<double>(candidate_first[index]) +
         static_cast<double>(candidate_second[index])) /
        2.0;
    timing.round_speedups[index] = baseline_pair / candidate_pair;
    baseline_sum += baseline_pair;
    candidate_sum += candidate_pair;
    speedup_sum += timing.round_speedups[index];
    timing.every_round_nonregressing =
        timing.every_round_nonregressing &&
        std::isfinite(timing.round_speedups[index]) &&
        timing.round_speedups[index] >= 1.0;
    std::cout << "PREFILL_M32_RESIDUAL_RMS_PERF_ROUND: fixture="
              << fixture_name << " round=" << round + 1
              << " baseline_pair_chain_ms=" << baseline_pair
              << " candidate_pair_chain_ms=" << candidate_pair
              << " speedup=" << timing.round_speedups[index]
              << " nonregression="
              << (timing.round_speedups[index] >= 1.0 ? "PASS" : "FAIL")
              << '\n';
  }
  timing.baseline_mean_milliseconds =
      baseline_sum / static_cast<double>(kRounds);
  timing.candidate_mean_milliseconds =
      candidate_sum / static_cast<double>(kRounds);
  timing.mean_speedup = speedup_sum / static_cast<double>(kRounds);
  std::array<double, kRounds> sorted_speedups = timing.round_speedups;
  std::sort(sorted_speedups.begin(), sorted_speedups.end());
  timing.minimum_speedup = sorted_speedups.front();
  timing.median_speedup =
      (sorted_speedups[2U] + sorted_speedups[3U]) / 2.0;

  timing.final_state_bitwise =
      std::equal(candidate_right.data(),
                 candidate_right.data() + kResidualRmsM32ElementCount,
                 baseline_right.data()) &&
      std::equal(candidate_residual.data(),
                 candidate_residual.data() + kResidualRmsM32ElementCount,
                 baseline_residual.data());
  timing.inputs_preserved =
      std::equal(left.data(),
                 left.data() + kResidualRmsM32ElementCount,
                 left_before.data()) &&
      std::equal(right_seed.data(),
                 right_seed.data() + kResidualRmsM32ElementCount,
                 right_seed_before.data()) &&
      std::equal(weight.data(), weight.data() + kResidualRmsM32HiddenSize,
                 weight_before.data());
  test.expect(timing.final_state_bitwise,
              label + " final residual and normalized states are bitwise");
  test.expect(timing.inputs_preserved,
              label + " preserves left/right seed/weight inputs");
  const bool performance_gate =
      timing.median_speedup >= kRequiredMedianSpeedup &&
      timing.every_round_nonregressing;
  timing.gate = performance_gate && timing.final_state_bitwise &&
                timing.inputs_preserved;
  test.expect(performance_gate,
              label + " clears 1.15x median and every-round 1.0x floors");
  std::cout << "PREFILL_M32_RESIDUAL_RMS_PERF_FIXTURE: fixture="
            << fixture_name
            << " baseline_mean_chain_ms="
            << timing.baseline_mean_milliseconds
            << " candidate_mean_chain_ms="
            << timing.candidate_mean_milliseconds
            << " speedup_mean=" << timing.mean_speedup
            << " speedup_median=" << timing.median_speedup
            << " speedup_min=" << timing.minimum_speedup
            << " required_median=1.15 required_round_min=1.0"
            << " final_state_bitwise="
            << (timing.final_state_bitwise ? "true" : "false")
            << " inputs_preserved="
            << (timing.inputs_preserved ? "true" : "false")
            << " gate=" << (timing.gate ? "PASS" : "FAIL") << '\n';
  return timing;
}

void test_prefill_m32_residual_rms_formal_perf(TestContext& test,
                                               cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_PREFILL_M32_RESIDUAL_RMS_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: formal Prefill M32 residual RMS performance gate; "
                 "set Q3X_RUN_PREFILL_M32_RESIDUAL_RMS_PERF=1 and "
                 "Q3X_PREFILL_M32_RESIDUAL_RMS_CHECKPOINT_FILE\n";
    return;
  }

  constexpr std::uint64_t kExpectedCheckpointBytes = 9'965'652'512ULL;
  constexpr std::size_t kWeightBytes =
      kResidualRmsM32HiddenSize * sizeof(std::uint16_t);
  struct ActualWeightPayload {
    std::string_view fixture_name;
    std::string_view tensor_name;
    std::uint64_t absolute_offset;
    std::string_view expected_sha256;
  };
  constexpr std::array<ActualWeightPayload, 2U> kActualPayloads{{
      {"actual_layer0_post_attention_norm",
       "model.language_model.layers.0.post_attention_layernorm.weight",
       2'544'000'192ULL,
       "8a672b9c5681d05057f2318e2a4ed48764fff341cd99afd660011766a8356359"},
      {"actual_layer1_input_norm",
       "model.language_model.layers.1.input_layernorm.weight",
       2'544'010'432ULL,
       "89b1d66c33ed1a46813b12d4ca0757fcdd01f1e6fb90d47a569f37e0603a193d"},
  }};
  const std::string label = "formal Prefill M32 residual RMS performance";

  const q3x::core::Sha256FileResult binary_identity =
      q3x::core::sha256_file("/proc/self/exe");
  test.expect(binary_identity.ok(), label + " hashes running test binary");
  std::cout << "PREFILL_M32_RESIDUAL_RMS_BINARY: sha256="
            << (binary_identity.ok() ? binary_identity.digest->hex()
                                     : "unavailable")
            << " error="
            << (binary_identity.ok() ? "none" : binary_identity.error)
            << " gate=" << (binary_identity.ok() ? "PASS" : "FAIL")
            << '\n';
  if (!binary_identity.ok()) {
    return;
  }

  const char* const checkpoint_value =
      std::getenv("Q3X_PREFILL_M32_RESIDUAL_RMS_CHECKPOINT_FILE");
  const bool checkpoint_set =
      checkpoint_value != nullptr && checkpoint_value[0] != '\0';
  test.expect(checkpoint_set, label + " requires actual checkpoint file");
  if (!checkpoint_set) {
    std::cout << "PREFILL_M32_RESIDUAL_RMS_CHECKPOINT: source=missing"
                 " required_env=Q3X_PREFILL_M32_RESIDUAL_RMS_CHECKPOINT_FILE"
                 " gate=FAIL\n";
    return;
  }
  const std::string checkpoint_path = checkpoint_value;
  std::ifstream checkpoint(checkpoint_path, std::ios::binary);
  test.expect(checkpoint.is_open(), label + " opens actual checkpoint");
  if (!checkpoint.is_open()) {
    return;
  }
  checkpoint.seekg(0, std::ios::end);
  const std::streamoff checkpoint_size = checkpoint.tellg();
  const bool checkpoint_size_gate =
      checkpoint_size >= 0 &&
      static_cast<std::uint64_t>(checkpoint_size) ==
          kExpectedCheckpointBytes;
  test.expect(checkpoint_size_gate, label + " pins checkpoint byte size");
  std::cout << "PREFILL_M32_RESIDUAL_RMS_CHECKPOINT: path="
            << checkpoint_path << " size=" << checkpoint_size
            << " expected_size=" << kExpectedCheckpointBytes
            << " gate=" << (checkpoint_size_gate ? "PASS" : "FAIL")
            << '\n';
  if (!checkpoint_size_gate) {
    return;
  }

  ManagedBuffer<std::uint16_t> left;
  ManagedBuffer<std::uint16_t> right_seed;
  ManagedBuffer<std::uint16_t> weight;
  ManagedBuffer<std::uint16_t> baseline_right;
  ManagedBuffer<std::uint16_t> candidate_right;
  ManagedBuffer<std::uint16_t> baseline_residual;
  ManagedBuffer<std::uint16_t> candidate_residual;
  bool ready = test.cuda_ok(left.allocate(kResidualRmsM32ElementCount),
                            label + " allocate left");
  ready = ready && test.cuda_ok(
                       right_seed.allocate(kResidualRmsM32ElementCount),
                       label + " allocate right seed");
  ready = ready &&
          test.cuda_ok(weight.allocate(kResidualRmsM32HiddenSize),
                       label + " allocate weight");
  ready = ready && test.cuda_ok(
                       baseline_right.allocate(kResidualRmsM32ElementCount),
                       label + " allocate baseline right/output");
  ready = ready && test.cuda_ok(
                       candidate_right.allocate(kResidualRmsM32ElementCount),
                       label + " allocate candidate right/output");
  ready = ready && test.cuda_ok(
                       baseline_residual.allocate(kResidualRmsM32ElementCount),
                       label + " allocate baseline residual");
  ready = ready && test.cuda_ok(
                       candidate_residual.allocate(kResidualRmsM32ElementCount),
                       label + " allocate candidate residual");
  if (!ready) {
    return;
  }

  const auto read_and_verify_payload =
      [&](const ActualWeightPayload& payload) {
        const bool range_valid =
            payload.absolute_offset <= kExpectedCheckpointBytes &&
            kWeightBytes <=
                kExpectedCheckpointBytes - payload.absolute_offset &&
            payload.absolute_offset <= static_cast<std::uint64_t>(
                                           std::numeric_limits<
                                               std::streamoff>::max()) &&
            kWeightBytes <= static_cast<std::size_t>(
                                std::numeric_limits<std::streamsize>::max());
        test.expect(range_valid,
                    label + " contains " + std::string(payload.tensor_name));
        if (!range_valid) {
          return false;
        }
        checkpoint.clear();
        checkpoint.seekg(static_cast<std::streamoff>(payload.absolute_offset),
                         std::ios::beg);
        const bool seek_ok = static_cast<bool>(checkpoint);
        test.expect(seek_ok,
                    label + " seeks " + std::string(payload.tensor_name));
        if (!seek_ok) {
          return false;
        }
        checkpoint.read(reinterpret_cast<char*>(weight.data()),
                        static_cast<std::streamsize>(kWeightBytes));
        const bool read_complete =
            checkpoint.gcount() == static_cast<std::streamsize>(kWeightBytes);
        test.expect(read_complete,
                    label + " reads " + std::string(payload.tensor_name));
        if (!read_complete) {
          return false;
        }
        q3x::core::Sha256 hasher;
        const bool hash_update = hasher.update(weight.data(), kWeightBytes);
        const std::string actual_sha256 = hasher.finalize().hex();
        const bool hash_gate =
            hash_update &&
            std::string_view(actual_sha256) == payload.expected_sha256;
        test.expect(hash_gate,
                    label + " pins " + std::string(payload.tensor_name) +
                        " payload SHA-256");
        std::cout << "PREFILL_M32_RESIDUAL_RMS_PAYLOAD: fixture="
                  << payload.fixture_name << " tensor=" << payload.tensor_name
                  << " weight_source=actual_checkpoint"
                  << " only_actual_field=norm_weight"
                  << " activation_source=deterministic"
                  << " residual_source=deterministic"
                  << " absolute_offset=" << payload.absolute_offset
                  << " bytes=" << kWeightBytes
                  << " sha256=" << actual_sha256
                  << " expected_sha256=" << payload.expected_sha256
                  << " gate=" << (hash_gate ? "PASS" : "FAIL") << '\n';
        return hash_gate;
      };

  std::array<bool, kActualPayloads.size()> actual_gates{};
  for (std::size_t fixture = 0U; fixture < kActualPayloads.size();
       ++fixture) {
    fill_residual_rms_m32_fixture(false, left.data(), right_seed.data(),
                                  weight.data());
    if (!read_and_verify_payload(kActualPayloads[fixture])) {
      std::cout << "PREFILL_M32_RESIDUAL_RMS_STOP: fixture="
                << kActualPayloads[fixture].fixture_name
                << " reason=payload_validation gate=FAIL\n";
      return;
    }
    const PrefillM32ResidualRmsTiming timing =
        benchmark_prefill_m32_residual_rms_fixture(
            test, stream, std::string(kActualPayloads[fixture].fixture_name),
            left, right_seed, weight, baseline_right, candidate_right,
            baseline_residual, candidate_residual);
    actual_gates[fixture] = timing.gate;
    if (!actual_gates[fixture]) {
      std::cout << "PREFILL_M32_RESIDUAL_RMS_STOP: fixture="
                << kActualPayloads[fixture].fixture_name
                << " reason=actual_gate_failed stress=SKIP gate=FAIL\n";
      return;
    }
  }

  fill_residual_rms_m32_fixture(true, left.data(), right_seed.data(),
                                weight.data());
  const PrefillM32ResidualRmsTiming stress =
      benchmark_prefill_m32_residual_rms_fixture(
          test, stream, "deterministic_cancellation_stress", left,
          right_seed, weight, baseline_right, candidate_right,
          baseline_residual, candidate_residual);
  const bool selected_gate = actual_gates[0U] && actual_gates[1U] &&
                             stress.gate;
  std::cout << "PREFILL_M32_RESIDUAL_RMS_SELECTED: actual_layer0="
            << (actual_gates[0U] ? "PASS" : "FAIL")
            << " actual_layer1="
            << (actual_gates[1U] ? "PASS" : "FAIL")
            << " stress=" << (stress.gate ? "PASS" : "FAIL")
            << " gate=" << (selected_gate ? "PASS" : "FAIL") << '\n';
  test.expect(selected_gate,
              label + " clears both actual weights and stress gates");
}

void run_outer_rms_tile_exact_case(TestContext& test, cudaStream_t stream,
                                   const std::size_t row_count) {
  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr float kEpsilon = 1.0e-6F;
  const std::size_t element_count = row_count * kHiddenSize;
  const std::string label =
      "outer centered RMSNorm tile M=" + std::to_string(row_count);

  ManagedBuffer<std::uint16_t> input;
  ManagedBuffer<std::uint16_t> weight;
  ManagedBuffer<std::uint16_t> scalar_output;
  ManagedBuffer<std::uint16_t> tile_output;
  bool ready = test.cuda_ok(input.allocate(element_count),
                            label + " allocate input");
  ready = ready && test.cuda_ok(weight.allocate(kHiddenSize),
                                label + " allocate weight");
  ready = ready && test.cuda_ok(scalar_output.allocate(element_count),
                                label + " allocate scalar output");
  ready = ready && test.cuda_ok(tile_output.allocate(element_count),
                                label + " allocate tile output");
  if (!ready) {
    return;
  }

  for (std::size_t row = 0U; row < row_count; ++row) {
    for (std::size_t dimension = 0U; dimension < kHiddenSize; ++dimension) {
      const int centered = static_cast<int>(
                               (row * 29U + dimension * 17U) % 127U) -
                           63;
      input[row * kHiddenSize + dimension] =
          encode_bf16(static_cast<float>(centered) / 32.0F);
    }
  }
  for (std::size_t dimension = 0U; dimension < kHiddenSize; ++dimension) {
    const int centered = static_cast<int>((dimension * 11U) % 61U) - 30;
    weight[dimension] =
        encode_bf16(static_cast<float>(centered) / 64.0F);
  }

  ready = launch_after_stale(test, stream, label + " repeated scalar", [&]() {
    for (std::size_t row = 0U; row < row_count; ++row) {
      const int status =
          q3x::runtime::launch_centered_rms_norm_reference_cuda(
              input.data() + row * kHiddenSize, weight.data(), kHiddenSize,
              kEpsilon, scalar_output.data() + row * kHiddenSize,
              static_cast<void*>(stream));
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  });
  if (!ready) {
    return;
  }
  ready = launch_after_stale(test, stream, label + " tiled", [&]() {
    return q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
        input.data(), weight.data(), row_count, kHiddenSize, kEpsilon,
        tile_output.data(), static_cast<void*>(stream));
  });
  if (!ready) {
    return;
  }

  const auto mismatch = std::mismatch(
      scalar_output.data(), scalar_output.data() + element_count,
      tile_output.data());
  test.expect(mismatch.first == scalar_output.data() + element_count,
              label + " is bitwise identical to repeated scalar launches" +
                  (mismatch.first == scalar_output.data() + element_count
                       ? std::string{}
                       : " at element " + std::to_string(static_cast<std::size_t>(
                                                mismatch.first -
                                                scalar_output.data()))));
}

void test_outer_rms_tile_exact(TestContext& test, cudaStream_t stream) {
  run_outer_rms_tile_exact_case(test, stream, 1U);
  run_outer_rms_tile_exact_case(test, stream, 2U);
  run_outer_rms_tile_exact_case(test, stream, 8U);
}

void test_fp32_conversion(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kCount = 519U;
  ManagedBuffer<float> input;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(input.allocate(kCount), "convert allocate input");
  ready = ready &&
          test.cuda_ok(output.allocate(kCount), "convert allocate output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < kCount; ++index) {
    input[index] = static_cast<float>(static_cast<int>(index) - 259) /
                   37.0F;
  }
  input[0] = float_from_bits(0x3f808000U);
  input[1] = float_from_bits(0x3f818000U);
  input[2] = float_from_bits(0x7f800001U);
  std::vector<std::uint16_t> cpu(kCount);
  (void)q3x::runtime::fp32_to_bf16_reference_cpu(
      input.data(), kCount, cpu.data());
  ready = launch_after_stale(test, stream, "FP32 to BF16", [&]() {
    return q3x::runtime::launch_fp32_to_bf16_reference_cuda(
        input.data(), kCount, output.data(), static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < kCount; ++index) {
      test.expect(output[index] == cpu[index],
                  "CUDA BF16 RNE element " + std::to_string(index));
    }
    test.expect(output[0] == 0x3f80U && output[1] == 0x3f82U,
                "CUDA BF16 conversion implements ties-to-even");
    test.expect(std::isnan(decode_bf16(output[2])) &&
                    (output[2] & 0x0040U) != 0U,
                "CUDA BF16 conversion quiets tiny-payload NaN");
  }
}

void test_bf16_greedy_argmax(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kVocabularySize = 248'320U;
  ManagedBuffer<std::uint16_t> input;
  ManagedBuffer<q3x::runtime::Bf16GreedyArgmaxResult> result;
  bool ready = test.cuda_ok(input.allocate(kVocabularySize),
                            "BF16 greedy argmax allocate input");
  ready = ready && test.cuda_ok(
                                result.allocate(
                                    q3x::runtime::
                                        kBf16GreedyArgmaxWorkspaceResults),
                                "BF16 greedy argmax allocate result");
  if (!ready) {
    return;
  }

  const auto launch = [&](const std::size_t count,
                          const std::string& label) {
    return launch_after_stale(test, stream, label, [&]() {
      return q3x::runtime::launch_bf16_greedy_argmax_cuda(
          input.data(), count, result.data(), static_cast<void*>(stream));
    });
  };

  for (std::size_t index = 0U; index < kVocabularySize; ++index) {
    const int centered = static_cast<int>((index * 131U) % 2'047U) - 1'023;
    input[index] = encode_bf16(static_cast<float>(centered) / 32.0F);
  }
  constexpr std::size_t kEarliestMaximum = 17U;
  constexpr std::size_t kRepeatedMaximum = 200'003U;
  input[kEarliestMaximum] = 0x7f7fU;
  input[kRepeatedMaximum] = 0x7f7fU;
  ready = launch(kVocabularySize, "BF16 greedy argmax vocabulary");
  if (ready) {
    test.expect(result[0].index == kEarliestMaximum &&
                    result[0].value_bits == 0x7f7fU &&
                    result[0].has_nonfinite == 0U,
                "CUDA BF16 greedy argmax selects earliest finite maximum");
  }

  input[0] = 0x8000U;
  input[1] = 0x0000U;
  ready = launch(2U, "BF16 greedy argmax signed zero");
  if (ready) {
    test.expect(result[0].index == 0U && result[0].value_bits == 0x8000U &&
                    result[0].has_nonfinite == 0U,
                "CUDA BF16 greedy argmax preserves earliest signed-zero tie");
  }

  constexpr std::array<std::size_t, 7U> kTailLengths{{
      1U, 255U, 256U, 257U, 2'047U, 2'048U, 2'049U,
  }};
  for (const std::size_t count : kTailLengths) {
    for (std::size_t index = 0U; index < count; ++index) {
      input[index] = encode_bf16(
          -1'000.0F + static_cast<float>((index * 73U) % 4'093U) / 8.0F);
    }
    const std::size_t first_maximum = std::min<std::size_t>(17U, count - 1U);
    input[first_maximum] = encode_bf16(-1.0F);
    if (count > 1'000U) {
      input[1'000U] = encode_bf16(-1.0F);
    }
    ready = launch(count, "BF16 greedy argmax tail length");
    if (ready) {
      test.expect(result[0].index == first_maximum &&
                      result[0].value_bits == encode_bf16(-1.0F) &&
                      result[0].has_nonfinite == 0U,
                  "CUDA BF16 greedy argmax handles singleton, warp, block, and tail lengths");
    }
  }

  constexpr std::array<std::size_t, 8U> kRandomLengths{{
      3U, 31U, 33U, 511U, 513U, 4'093U, 65'537U, kVocabularySize,
  }};
  std::uint32_t random_state = 0x6d2b'79f5U;
  for (const std::size_t count : kRandomLengths) {
    for (std::size_t index = 0U; index < count; ++index) {
      std::uint16_t bits =
          static_cast<std::uint16_t>(next_deterministic_random(random_state));
      if ((bits & 0x7f80U) == 0x7f80U) {
        bits = static_cast<std::uint16_t>(bits & ~0x0080U);
      }
      input[index] = bits;
    }
    const q3x::runtime::Bf16GreedyArgmaxResult expected =
        bf16_greedy_argmax_cpu_oracle(input.data(), count);
    const std::string label =
        "BF16 greedy argmax deterministic random count=" +
        std::to_string(count);
    ready = launch(count, label);
    if (ready) {
      test.expect(result[0].index == expected.index &&
                      result[0].value_bits == expected.value_bits &&
                      result[0].has_nonfinite == expected.has_nonfinite,
                  label + " matches the independent CPU oracle");
    }
  }

  constexpr std::array<std::uint16_t, 6U> kNonfiniteValues{{
      0x7f80U, 0xff80U, 0x7f81U, 0xff81U, 0x7fc1U, 0xffc1U,
  }};
  input[0] = encode_bf16(-2.0F);
  input[1] = encode_bf16(4.0F);
  for (std::size_t index = 0U; index < kNonfiniteValues.size(); ++index) {
    input[index + 2U] = kNonfiniteValues[index];
  }
  ready = launch(kNonfiniteValues.size() + 2U,
                 "BF16 greedy argmax nonfinite");
  if (ready) {
    test.expect(result[0].index == 1U &&
                    result[0].value_bits == encode_bf16(4.0F) &&
                    result[0].has_nonfinite != 0U,
                "CUDA BF16 greedy argmax reports every nonfinite class");
  }
  for (std::size_t index = 0U; index < kNonfiniteValues.size(); ++index) {
    input[index] = kNonfiniteValues[index];
  }
  ready = launch(kNonfiniteValues.size(),
                 "BF16 greedy argmax all nonfinite");
  if (ready) {
    test.expect(result[0].index ==
                        std::numeric_limits<std::uint32_t>::max() &&
                    result[0].value_bits == 0U &&
                    result[0].has_nonfinite != 0U,
                "CUDA BF16 greedy argmax safely rejects an all-nonfinite vector");
  }

  if (std::getenv("Q3X_RUN_BF16_GREEDY_ARGMAX_PERF") != nullptr) {
    for (std::size_t index = 0U; index < kVocabularySize; ++index) {
      input[index] = static_cast<std::uint16_t>(
          0x3b00U + ((index * 17U) % 0x0400U));
    }
    for (int warmup = 0; warmup < 10; ++warmup) {
      ready = ready && test.cuda_ok(
          static_cast<cudaError_t>(
              q3x::runtime::launch_bf16_greedy_argmax_cuda(
                  input.data(), kVocabularySize, result.data(),
                  static_cast<void*>(stream))),
          "BF16 greedy argmax performance warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  "BF16 greedy argmax warmup synchronize");
    if (ready) {
      constexpr std::size_t kMeasuredIterations = 200U;
      const float milliseconds = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          "BF16 greedy argmax performance", [&]() {
            return q3x::runtime::launch_bf16_greedy_argmax_cuda(
                input.data(), kVocabularySize, result.data(),
                static_cast<void*>(stream));
          });
      std::cout << "PERF_BF16_GREEDY_ARGMAX: length=" << kVocabularySize
                << " kernel_ms=" << milliseconds
                << " measured_iterations=" << kMeasuredIterations << '\n';
    }
  }
}

enum class HeadwiseNormKind : std::uint8_t {
  kCentered,
  kPlain,
  kPlainSiluGate,
};

void run_headwise_norm_case(TestContext& test, cudaStream_t stream,
                            const std::size_t heads,
                            const std::size_t dimension,
                            const HeadwiseNormKind kind,
                            const std::string& label) {
  constexpr float kEpsilon = 1.0e-6F;
  const std::size_t count = heads * dimension;
  ManagedBuffer<std::uint16_t> input;
  ManagedBuffer<std::uint16_t> weight;
  ManagedBuffer<std::uint16_t> gate;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(input.allocate(count), label + " allocate input");
  ready = ready &&
          test.cuda_ok(weight.allocate(dimension), label + " allocate weight");
  ready = ready &&
          test.cuda_ok(gate.allocate(count), label + " allocate gate");
  ready = ready &&
          test.cuda_ok(output.allocate(count), label + " allocate output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < count; ++index) {
    input[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 13U) % 43U) - 21) /
        16.0F);
    gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 7U) % 31U) - 15) /
        16.0F);
  }
  for (std::size_t index = 0; index < dimension; ++index) {
    weight[index] = encode_bf16(
        0.25F + static_cast<float>(index % 11U) / 32.0F);
  }
  std::vector<std::uint16_t> cpu(count);
  switch (kind) {
    case HeadwiseNormKind::kCentered:
      (void)q3x::runtime::headwise_centered_rms_norm_reference_cpu(
          input.data(), weight.data(), heads, dimension, kEpsilon,
          cpu.data());
      break;
    case HeadwiseNormKind::kPlain:
      (void)q3x::runtime::headwise_plain_rms_norm_reference_cpu(
          input.data(), weight.data(), heads, dimension, kEpsilon,
          cpu.data());
      break;
    case HeadwiseNormKind::kPlainSiluGate:
      (void)q3x::runtime::headwise_plain_rms_norm_silu_gate_reference_cpu(
          input.data(), weight.data(), gate.data(), heads, dimension, kEpsilon,
          cpu.data());
      break;
  }
  ready = launch_after_stale(test, stream, label, [&]() {
    switch (kind) {
      case HeadwiseNormKind::kCentered:
        return q3x::runtime::
            launch_headwise_centered_rms_norm_reference_cuda(
                input.data(), weight.data(), heads, dimension, kEpsilon,
                output.data(), static_cast<void*>(stream));
      case HeadwiseNormKind::kPlain:
        return q3x::runtime::launch_headwise_plain_rms_norm_reference_cuda(
            input.data(), weight.data(), heads, dimension, kEpsilon,
            output.data(), static_cast<void*>(stream));
      case HeadwiseNormKind::kPlainSiluGate:
        return q3x::runtime::
            launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                input.data(), weight.data(), gate.data(), heads, dimension,
                kEpsilon, output.data(), static_cast<void*>(stream));
    }
    return static_cast<int>(cudaErrorInvalidValue);
  });
  if (ready) {
    for (std::size_t index = 0; index < count; ++index) {
      expect_bf16_match(test, output[index], cpu[index],
                        label + " element " + std::to_string(index));
    }
  }
}

void run_l2_case(TestContext& test, cudaStream_t stream,
                 const std::size_t heads, const std::size_t dimension,
                 const std::string& label) {
  const std::size_t count = heads * dimension;
  ManagedBuffer<std::uint16_t> input;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(input.allocate(count), label + " allocate input");
  ready = ready &&
          test.cuda_ok(output.allocate(count), label + " allocate output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < count; ++index) {
    input[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 11U) % 31U) - 15) /
        32.0F);
  }
  std::vector<std::uint16_t> cpu(count);
  (void)q3x::runtime::l2_normalize_heads_reference_cpu(
      input.data(), heads, dimension, 1.0e-6F, cpu.data());
  ready = launch_after_stale(test, stream, label, [&]() {
    return q3x::runtime::launch_l2_normalize_heads_reference_cuda(
        input.data(), heads, dimension, 1.0e-6F, output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < count; ++index) {
      expect_bf16_match(test, output[index], cpu[index],
                        label + " element " + std::to_string(index));
    }
  }
}

void test_rope(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kHeads = 24U;
  constexpr std::size_t kDimension =
      q3x::runtime::kFullAttentionHeadDimension;
  constexpr std::size_t kHalfRotary =
      q3x::runtime::kQwenRotaryDimension / 2U;
  ManagedBuffer<std::uint16_t> input;
  ManagedBuffer<std::uint16_t> output;
  ManagedBuffer<float> cosines;
  ManagedBuffer<float> sines;
  bool ready = test.cuda_ok(input.allocate(kHeads * kDimension),
                            "RoPE allocate input");
  ready = ready && test.cuda_ok(output.allocate(kHeads * kDimension),
                                "RoPE allocate output");
  ready = ready &&
          test.cuda_ok(cosines.allocate(kHalfRotary), "RoPE allocate cosine");
  ready = ready &&
          test.cuda_ok(sines.allocate(kHalfRotary), "RoPE allocate sine");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < input.size(); ++index) {
    input[index] = encode_bf16(
        static_cast<float>(static_cast<int>(index % 37U) - 18) / 32.0F);
  }
  for (std::size_t index = 0; index < kHalfRotary; ++index) {
    const float angle = static_cast<float>(index) * 0.01953125F;
    cosines[index] = std::cos(angle);
    sines[index] = std::sin(angle);
  }
  std::vector<std::uint16_t> cpu(input.size());
  (void)q3x::runtime::partial_neox_rope_256_64_reference_cpu(
      input.data(), cosines.data(), sines.data(), kHeads, cpu.data());
  ready = launch_after_stale(test, stream, "partial NeoX RoPE", [&]() {
    return q3x::runtime::launch_partial_neox_rope_256_64_reference_cuda(
        input.data(), cosines.data(), sines.data(), kHeads, output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < input.size(); ++index) {
      expect_bf16_match(test, output[index], cpu[index],
                        "CUDA RoPE element " + std::to_string(index));
    }
  }
}

enum class FullAttentionPreprocFixture {
  kFinite,
  kNonfinite,
};

void run_full_attention_preproc_fusion_exact_case(
    TestContext& test, cudaStream_t stream, const std::size_t token_count,
    const FullAttentionPreprocFixture fixture) {
  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKeyHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kHalfRotary = 32U;
  constexpr std::size_t kFirstPosition = 7U;
  constexpr float kEpsilon = 1.0e-6F;
  constexpr std::size_t kQueryElementsPerToken =
      kQueryHeads * kDimension;
  constexpr std::size_t kKeyElementsPerToken = kKeyHeads * kDimension;
  const std::size_t query_elements = token_count * kQueryElementsPerToken;
  const std::size_t key_elements = token_count * kKeyElementsPerToken;
  const std::size_t interleaved_elements = 2U * query_elements;
  const std::size_t table_elements =
      (kFirstPosition + token_count) * kHalfRotary;
  const std::string label =
      "full-attention preprocess fusion M=" +
      std::to_string(token_count) +
      (fixture == FullAttentionPreprocFixture::kFinite ? " finite"
                                                       : " nonfinite");

  GuardedBf16Buffer interleaved_q_gate;
  GuardedBf16Buffer q_weight;
  GuardedBf16Buffer k_weight;
  GuardedBf16Buffer baseline_split_query;
  GuardedBf16Buffer baseline_query;
  GuardedBf16Buffer baseline_gate;
  GuardedBf16Buffer baseline_key;
  GuardedBf16Buffer candidate_query;
  GuardedBf16Buffer candidate_gate;
  GuardedBf16Buffer candidate_key;
  GuardedBf16Buffer warp_query;
  GuardedBf16Buffer warp_gate;
  GuardedBf16Buffer warp_key;
  ManagedBuffer<float> cosines;
  ManagedBuffer<float> sines;
  bool ready = interleaved_q_gate.allocate(
      test, interleaved_elements, label + " interleaved Q/gate");
  ready = ready &&
          q_weight.allocate(test, kDimension, label + " Q weight");
  ready = ready &&
          k_weight.allocate(test, kDimension, label + " K weight");
  ready = ready && baseline_split_query.allocate(
                       test, query_elements, label + " baseline split Q");
  ready = ready && baseline_query.allocate(
                       test, query_elements, label + " baseline query");
  ready = ready && baseline_gate.allocate(
                       test, query_elements, label + " baseline gate");
  ready = ready && baseline_key.allocate(
                       test, key_elements, label + " baseline key");
  ready = ready && candidate_query.allocate(
                       test, query_elements, label + " candidate query");
  ready = ready && candidate_gate.allocate(
                       test, query_elements, label + " candidate gate");
  ready = ready && candidate_key.allocate(
                       test, key_elements, label + " candidate key");
  ready = ready &&
          warp_query.allocate(test, query_elements, label + " warp query");
  ready = ready &&
          warp_gate.allocate(test, query_elements, label + " warp gate");
  ready = ready &&
          warp_key.allocate(test, key_elements, label + " warp key");
  ready = ready &&
          test.cuda_ok(cosines.allocate(table_elements),
                       label + " cosine table");
  ready = ready &&
          test.cuda_ok(sines.allocate(table_elements), label + " sine table");
  if (!ready) {
    return;
  }

  interleaved_q_gate.initialize(0U, 0x11a1U, 0xee5eU);
  q_weight.initialize(0U, 0x22b2U, 0xdd4dU);
  k_weight.initialize(0U, 0x33c3U, 0xcc3cU);
  baseline_split_query.initialize(0x7915U, 0x44d4U, 0xbb2bU);
  baseline_query.initialize(0x6a26U, 0x55e5U, 0xaa1aU);
  baseline_gate.initialize(0x5b37U, 0x66f6U, 0x9909U);
  baseline_key.initialize(0U, 0x7707U, 0x88f8U);
  candidate_query.initialize(0x4c48U, 0x18a8U, 0xe757U);
  candidate_gate.initialize(0x3d59U, 0x29b9U, 0xd646U);
  candidate_key.initialize(0U, 0x3acaU, 0xc535U);
  warp_query.initialize(0x2e6aU, 0x4b1dU, 0xb4e2U);
  warp_gate.initialize(0x1f7bU, 0x5c2eU, 0xa3d1U);
  warp_key.initialize(0U, 0x6d3fU, 0x92c0U);

  for (std::size_t head = 0U; head < token_count * kQueryHeads; ++head) {
    const std::size_t input_offset = head * 2U * kDimension;
    for (std::size_t dimension = 0U; dimension < kDimension; ++dimension) {
      const std::size_t seed = head * kDimension + dimension;
      interleaved_q_gate.data()[input_offset + dimension] = encode_bf16(
          static_cast<float>(static_cast<int>((seed * 37U + 19U) % 509U) -
                             254) /
          128.0F);
      interleaved_q_gate.data()[input_offset + kDimension + dimension] =
          static_cast<std::uint16_t>((seed * 6'541U + 0x1357U) & 0xffffU);
    }
  }
  for (std::size_t index = 0U; index < key_elements; ++index) {
    baseline_key.data()[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 43U + 23U) % 503U) -
                           251) /
        96.0F);
    candidate_key.data()[index] = baseline_key.data()[index];
    warp_key.data()[index] = baseline_key.data()[index];
  }
  for (std::size_t dimension = 0U; dimension < kDimension; ++dimension) {
    q_weight.data()[dimension] = encode_bf16(
        static_cast<float>(static_cast<int>((dimension * 17U) % 61U) - 30) /
        64.0F);
    k_weight.data()[dimension] = encode_bf16(
        static_cast<float>(static_cast<int>((dimension * 29U) % 67U) - 33) /
        64.0F);
  }

  if (fixture == FullAttentionPreprocFixture::kNonfinite) {
    constexpr std::array<std::uint16_t, 8U> kSpecialBits = {
        0x7f80U, 0xff80U, 0x7fc1U, 0xffc1U,
        0x0000U, 0x8000U, 0x0001U, 0x8001U};
    for (std::size_t sample = 0U; sample < kSpecialBits.size(); ++sample) {
      const std::size_t flat_head =
          (sample * 7U + 3U) % (token_count * kQueryHeads);
      const std::size_t dimension = (sample * 31U + 5U) % kDimension;
      const std::size_t input_offset =
          flat_head * 2U * kDimension + dimension;
      interleaved_q_gate.data()[input_offset] = kSpecialBits[sample];
      interleaved_q_gate.data()[input_offset + kDimension] =
          kSpecialBits[(sample + 3U) % kSpecialBits.size()];
      const std::size_t key_index =
          (sample * 127U + 11U) % key_elements;
      baseline_key.data()[key_index] =
          kSpecialBits[(sample + 1U) % kSpecialBits.size()];
      candidate_key.data()[key_index] = baseline_key.data()[key_index];
      warp_key.data()[key_index] = baseline_key.data()[key_index];
    }
    q_weight.data()[13U] = 0x7fc1U;
    k_weight.data()[29U] = 0xff80U;
  }

  for (std::size_t index = 0U; index < table_elements; ++index) {
    const float angle =
        static_cast<float>(static_cast<int>((index * 13U) % 1'009U) - 504) /
        2'048.0F;
    cosines[index] = std::cos(angle);
    sines[index] = std::sin(angle);
  }

  const auto original_interleaved = interleaved_q_gate.snapshot();
  const auto original_q_weight = q_weight.snapshot();
  const auto original_k_weight = k_weight.snapshot();
  const std::vector<float> original_cosines(cosines.data(),
                                             cosines.data() + table_elements);
  const std::vector<float> original_sines(sines.data(),
                                           sines.data() + table_elements);

  const auto launch_baseline = [&]() {
    int status =
        q3x::runtime::launch_split_interleaved_q_gate_reference_cuda(
            interleaved_q_gate.data(), token_count * kQueryHeads, kDimension,
            baseline_split_query.data(), baseline_gate.data(),
            static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
        baseline_split_query.data(), q_weight.data(),
        token_count * kQueryHeads, kDimension, kEpsilon,
        baseline_query.data(), static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
        baseline_key.data(), k_weight.data(), token_count * kKeyHeads,
        kDimension, kEpsilon, baseline_key.data(),
        static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    return q3x::runtime::
        launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
            baseline_query.data(), baseline_key.data(), cosines.data(),
            sines.data(), kFirstPosition, token_count,
            static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() {
    return q3x::runtime::launch_full_attention_preprocess_24_4_256_64_cuda(
        interleaved_q_gate.data(), candidate_key.data(), q_weight.data(),
        k_weight.data(), kEpsilon, candidate_query.data(),
        candidate_gate.data(), cosines.data(), sines.data(), kFirstPosition,
        token_count, static_cast<void*>(stream));
  };
  const auto launch_warp = [&]() {
    return q3x::runtime::
        launch_full_attention_preprocess_warp_rms_24_4_256_64_test_cuda(
            interleaved_q_gate.data(), warp_key.data(), q_weight.data(),
            k_weight.data(), kEpsilon, warp_query.data(), warp_gate.data(),
            cosines.data(), sines.data(), kFirstPosition, token_count,
            static_cast<void*>(stream));
  };

  const bool baseline_ready = launch_after_stale(
      test, stream, label + " baseline four launches", launch_baseline);
  const bool candidate_ready = launch_after_stale(
      test, stream, label + " candidate one launch", launch_candidate);
  const bool warp_ready = launch_after_stale(
      test, stream, label + " warp-RMS one launch", launch_warp);
  if (baseline_ready && candidate_ready && warp_ready) {
    expect_bf16_bits_equal(test, candidate_query.data(),
                           baseline_query.data(), query_elements,
                           label + " normalized and rotated query");
    expect_bf16_bits_equal(test, candidate_key.data(), baseline_key.data(),
                           key_elements,
                           label + " in-place normalized and rotated key");
    expect_bf16_bits_equal(test, candidate_gate.data(), baseline_gate.data(),
                           query_elements, label + " raw packed gate");
    expect_bf16_bits_equal(test, warp_query.data(), candidate_query.data(),
                           query_elements,
                           label + " warp-RMS normalized and rotated query");
    expect_bf16_bits_equal(test, warp_key.data(), candidate_key.data(),
                           key_elements,
                           label + " warp-RMS normalized and rotated key");
    expect_bf16_bits_equal(test, warp_gate.data(), candidate_gate.data(),
                           query_elements,
                           label + " warp-RMS raw packed gate");
  }

  interleaved_q_gate.expect_snapshot(test, original_interleaved,
                                     label + " interleaved input");
  q_weight.expect_snapshot(test, original_q_weight, label + " Q weight");
  k_weight.expect_snapshot(test, original_k_weight, label + " K weight");
  baseline_split_query.expect_guards(test, label + " baseline split Q");
  baseline_query.expect_guards(test, label + " baseline query");
  baseline_gate.expect_guards(test, label + " baseline gate");
  baseline_key.expect_guards(test, label + " baseline key");
  candidate_query.expect_guards(test, label + " candidate query");
  candidate_gate.expect_guards(test, label + " candidate gate");
  candidate_key.expect_guards(test, label + " candidate key");
  warp_query.expect_guards(test, label + " warp query");
  warp_gate.expect_guards(test, label + " warp gate");
  warp_key.expect_guards(test, label + " warp key");
  test.expect(std::memcmp(cosines.data(), original_cosines.data(),
                          table_elements * sizeof(float)) == 0,
              label + " cosine table is unchanged");
  test.expect(std::memcmp(sines.data(), original_sines.data(),
                          table_elements * sizeof(float)) == 0,
              label + " sine table is unchanged");
}

void test_full_attention_preproc_fusion_exact(TestContext& test,
                                              cudaStream_t stream) {
  constexpr std::array<std::size_t, 4U> kTokenCounts = {1U, 2U, 8U, 16U};
  for (const std::size_t token_count : kTokenCounts) {
    run_full_attention_preproc_fusion_exact_case(
        test, stream, token_count, FullAttentionPreprocFixture::kFinite);
    run_full_attention_preproc_fusion_exact_case(
        test, stream, token_count, FullAttentionPreprocFixture::kNonfinite);
  }
}

struct FullPreprocessKernelResources {
  int registers = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks_per_multiprocessor = 0;
};

using FullPreprocessResourceQuery = int (*)(
    int*, std::size_t*, std::size_t*, int*, int*) noexcept;

void test_full_attention_preproc_warp_rms_contract(TestContext& test,
                                                   cudaStream_t stream) {
  constexpr std::size_t kQueryElements = 24U * 256U;
  constexpr std::size_t kKeyElements = 4U * 256U;
  constexpr std::size_t kInterleavedElements = 2U * kQueryElements;
  constexpr std::size_t kHalfRotary = 32U;
  constexpr float kEpsilon = 1.0e-6F;
  const std::string label = "full-attention preprocess warp-RMS contract";

  FullPreprocessKernelResources production_resources;
  FullPreprocessKernelResources candidate_resources;
  constexpr std::array<std::pair<const char*, FullPreprocessResourceQuery>, 2U>
      kQueries{{
          {"production",
           q3x::runtime::
               query_full_attention_preprocess_24_4_256_64_resources_test_cuda},
          {"warp-RMS",
           q3x::runtime::
               query_full_attention_preprocess_warp_rms_24_4_256_64_resources_test_cuda},
      }};
  for (const auto& [name, query] : kQueries) {
    FullPreprocessKernelResources scratch;
    for (std::size_t null_index = 0U; null_index < 5U; ++null_index) {
      const cudaError_t status = static_cast<cudaError_t>(query(
          null_index == 0U ? nullptr : &scratch.registers,
          null_index == 1U ? nullptr : &scratch.static_shared_bytes,
          null_index == 2U ? nullptr : &scratch.local_bytes,
          null_index == 3U ? nullptr : &scratch.maximum_threads,
          null_index == 4U
              ? nullptr
              : &scratch.active_blocks_per_multiprocessor));
      test.expect(status == cudaErrorInvalidValue,
                  label + " " + name + " resource query rejects null " +
                      std::to_string(null_index));
    }
  }
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_full_attention_preprocess_24_4_256_64_resources_test_cuda(
              &production_resources.registers,
              &production_resources.static_shared_bytes,
              &production_resources.local_bytes,
              &production_resources.maximum_threads,
              &production_resources.active_blocks_per_multiprocessor)),
      label + " query production resources");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(q3x::runtime::
                           query_full_attention_preprocess_warp_rms_24_4_256_64_resources_test_cuda(
                               &candidate_resources.registers,
                               &candidate_resources.static_shared_bytes,
                               &candidate_resources.local_bytes,
                               &candidate_resources.maximum_threads,
                               &candidate_resources
                                    .active_blocks_per_multiprocessor)),
                       label + " query candidate resources");
  if (!ready) {
    return;
  }
  const bool resource_gate =
      production_resources.local_bytes == 0U &&
      candidate_resources.registers <= 40 &&
      candidate_resources.static_shared_bytes <= sizeof(float) &&
      candidate_resources.local_bytes == 0U &&
      candidate_resources.maximum_threads >= 256 &&
      candidate_resources.active_blocks_per_multiprocessor >= 6;
  test.expect(resource_gate,
              label + " keeps the <=40r/4B/0-local/6-CTA resource envelope");
  std::cout << "DECODE_FULL_PREPROCESS_WARP_RMS_RESOURCES: production_regs="
            << production_resources.registers
            << " production_shared_bytes="
            << production_resources.static_shared_bytes
            << " production_local_bytes=" << production_resources.local_bytes
            << " production_active_blocks_per_sm="
            << production_resources.active_blocks_per_multiprocessor
            << " candidate_regs=" << candidate_resources.registers
            << " candidate_shared_bytes="
            << candidate_resources.static_shared_bytes
            << " candidate_local_bytes=" << candidate_resources.local_bytes
            << " candidate_active_blocks_per_sm="
            << candidate_resources.active_blocks_per_multiprocessor
            << " gate=" << (resource_gate ? "PASS" : "FAIL") << '\n';

  ManagedBuffer<std::uint16_t> interleaved;
  ManagedBuffer<std::uint16_t> production_key;
  ManagedBuffer<std::uint16_t> candidate_key;
  ManagedBuffer<std::uint16_t> q_weight;
  ManagedBuffer<std::uint16_t> k_weight;
  ManagedBuffer<std::uint16_t> production_query;
  ManagedBuffer<std::uint16_t> production_gate;
  ManagedBuffer<std::uint16_t> candidate_query;
  ManagedBuffer<std::uint16_t> candidate_gate;
  ManagedBuffer<float> cosines;
  ManagedBuffer<float> sines;
  ready = test.cuda_ok(interleaved.allocate(kInterleavedElements),
                       label + " allocate interleaved");
  ready = ready && test.cuda_ok(production_key.allocate(kKeyElements),
                                label + " allocate production key");
  ready = ready && test.cuda_ok(candidate_key.allocate(kKeyElements),
                                label + " allocate candidate key");
  ready = ready && test.cuda_ok(q_weight.allocate(256U),
                                label + " allocate Q weight");
  ready = ready && test.cuda_ok(k_weight.allocate(256U),
                                label + " allocate K weight");
  ready = ready && test.cuda_ok(production_query.allocate(kQueryElements),
                                label + " allocate production query");
  ready = ready && test.cuda_ok(production_gate.allocate(kQueryElements),
                                label + " allocate production gate");
  ready = ready && test.cuda_ok(candidate_query.allocate(kQueryElements),
                                label + " allocate candidate query");
  ready = ready && test.cuda_ok(candidate_gate.allocate(kQueryElements),
                                label + " allocate candidate gate");
  ready = ready && test.cuda_ok(cosines.allocate(kHalfRotary),
                                label + " allocate cosines");
  ready = ready && test.cuda_ok(sines.allocate(kHalfRotary),
                                label + " allocate sines");
  if (!ready) {
    return;
  }
  std::fill_n(interleaved.data(), interleaved.size(), encode_bf16(0.25F));
  std::fill_n(production_key.data(), production_key.size(),
              encode_bf16(0.5F));
  std::copy_n(production_key.data(), production_key.size(),
              candidate_key.data());
  std::fill_n(q_weight.data(), q_weight.size(), encode_bf16(0.0F));
  std::fill_n(k_weight.data(), k_weight.size(), encode_bf16(0.0F));
  std::fill_n(cosines.data(), cosines.size(), 1.0F);
  std::fill_n(sines.data(), sines.size(), 0.0F);

  struct Topology {
    void* function = nullptr;
    dim3 grid{};
    dim3 block{};
    std::size_t dynamic_shared_bytes = 0U;
  };
  const auto capture_valid = [&](const std::string& route,
                                 auto&& launch,
                                 Topology& topology) {
    bool capture_ready = test.cuda_ok(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        label + " " + route + " begin capture");
    const cudaError_t launch_status =
        capture_ready ? static_cast<cudaError_t>(launch())
                      : cudaErrorUnknown;
    test.expect(launch_status == cudaSuccess,
                label + " " + route + " capture launch succeeds");
    cudaGraph_t graph = nullptr;
    capture_ready = capture_ready &&
                    test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                                 label + " " + route + " end capture");
    if (!capture_ready) {
      return false;
    }
    std::size_t node_count = 0U;
    capture_ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                                 label + " " + route + " count nodes") &&
                    capture_ready;
    test.expect(node_count == 1U,
                label + " " + route + " captures one node");
    cudaGraphNode_t node = nullptr;
    std::size_t capacity = 1U;
    if (node_count == 1U) {
      capture_ready = test.cuda_ok(cudaGraphGetNodes(graph, &node, &capacity),
                                   label + " " + route + " fetch node") &&
                      capture_ready;
      cudaGraphNodeType node_type = cudaGraphNodeTypeEmpty;
      capture_ready = test.cuda_ok(cudaGraphNodeGetType(node, &node_type),
                                   label + " " + route + " node type") &&
                      capture_ready;
      test.expect(node_type == cudaGraphNodeTypeKernel,
                  label + " " + route + " node is a kernel");
      cudaKernelNodeParams parameters{};
      capture_ready = test.cuda_ok(
                          cudaGraphKernelNodeGetParams(node, &parameters),
                          label + " " + route + " node parameters") &&
                      capture_ready;
      if (capture_ready) {
        topology.function = parameters.func;
        topology.grid = parameters.gridDim;
        topology.block = parameters.blockDim;
        topology.dynamic_shared_bytes = parameters.sharedMemBytes;
      }
    } else {
      capture_ready = false;
    }
    cudaGraphExec_t executable = nullptr;
    if (capture_ready) {
      capture_ready = test.cuda_ok(
                          cudaGraphInstantiate(&executable, graph, nullptr,
                                               nullptr, 0U),
                          label + " " + route + " instantiate") &&
                      capture_ready;
    }
    if (capture_ready) {
      capture_ready = test.cuda_ok(cudaGraphLaunch(executable, stream),
                                   label + " " + route + " replay") &&
                      capture_ready;
      capture_ready = test.cuda_ok(cudaStreamSynchronize(stream),
                                   label + " " + route + " replay sync") &&
                      capture_ready;
    }
    if (executable != nullptr) {
      (void)test.cuda_ok(cudaGraphExecDestroy(executable),
                         label + " " + route + " destroy executable");
    }
    (void)test.cuda_ok(cudaGraphDestroy(graph),
                       label + " " + route + " destroy graph");
    return capture_ready;
  };

  const auto launch_production = [&]() {
    return q3x::runtime::launch_full_attention_preprocess_24_4_256_64_cuda(
        interleaved.data(), production_key.data(), q_weight.data(),
        k_weight.data(), kEpsilon, production_query.data(),
        production_gate.data(), cosines.data(), sines.data(), 0U, 1U,
        static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() {
    return q3x::runtime::
        launch_full_attention_preprocess_warp_rms_24_4_256_64_test_cuda(
            interleaved.data(), candidate_key.data(), q_weight.data(),
            k_weight.data(), kEpsilon, candidate_query.data(),
            candidate_gate.data(), cosines.data(), sines.data(), 0U, 1U,
            static_cast<void*>(stream));
  };
  Topology production_topology;
  Topology candidate_topology;
  const bool production_graph =
      capture_valid("production", launch_production, production_topology);
  const bool candidate_graph =
      capture_valid("warp-RMS", launch_candidate, candidate_topology);
  const bool topology_gate =
      production_graph && candidate_graph &&
      production_topology.function != candidate_topology.function &&
      production_topology.grid.x == 28U &&
      candidate_topology.grid.x == 28U &&
      production_topology.block.x == 256U &&
      candidate_topology.block.x == 256U &&
      production_topology.dynamic_shared_bytes == 0U &&
      candidate_topology.dynamic_shared_bytes == 0U;
  test.expect(topology_gate,
              label + " preserves one-node 28x256 launch topology");
  std::cout << "DECODE_FULL_PREPROCESS_WARP_RMS_GRAPH: production_nodes=1 "
               "candidate_nodes=1 production_grid="
            << production_topology.grid.x
            << " candidate_grid=" << candidate_topology.grid.x
            << " production_block=" << production_topology.block.x
            << " candidate_block=" << candidate_topology.block.x
            << " distinct_functions="
            << (production_topology.function != candidate_topology.function
                    ? "true"
                    : "false")
            << " replay=pass gate=" << (topology_gate ? "PASS" : "FAIL")
            << '\n';

  const auto launch_invalid = [&](const std::size_t invalid_case) {
    const std::uint16_t* input = interleaved.data();
    std::uint16_t* key = candidate_key.data();
    const std::uint16_t* q_scale = q_weight.data();
    const std::uint16_t* k_scale = k_weight.data();
    float epsilon = kEpsilon;
    std::uint16_t* query = candidate_query.data();
    std::uint16_t* gate = candidate_gate.data();
    const float* cosine = cosines.data();
    const float* sine = sines.data();
    std::size_t position = 0U;
    std::size_t tokens = 1U;
    switch (invalid_case) {
      case 0U: tokens = 0U; break;
      case 1U: tokens = 17U; break;
      case 2U: epsilon = std::numeric_limits<float>::quiet_NaN(); break;
      case 3U: position = std::numeric_limits<std::size_t>::max(); break;
      case 4U: input = nullptr; break;
      case 5U: key = nullptr; break;
      case 6U: q_scale = nullptr; break;
      case 7U: k_scale = nullptr; break;
      case 8U: query = nullptr; break;
      case 9U: gate = nullptr; break;
      case 10U: cosine = nullptr; break;
      case 11U: sine = nullptr; break;
      case 12U: gate = query; break;
      case 13U:
        query = const_cast<std::uint16_t*>(input);
        break;
      case 14U:
        key = const_cast<std::uint16_t*>(q_scale);
        break;
      default: break;
    }
    return q3x::runtime::
        launch_full_attention_preprocess_warp_rms_24_4_256_64_test_cuda(
            input, key, q_scale, k_scale, epsilon, query, gate, cosine, sine,
            position, tokens, static_cast<void*>(stream));
  };
  bool invalid_gate = true;
  for (std::size_t invalid_case = 0U; invalid_case < 15U; ++invalid_case) {
    const std::string case_label =
        label + " invalid=" + std::to_string(invalid_case);
    bool case_ready = test.cuda_ok(
        cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
        case_label + " begin capture");
    const cudaError_t status =
        case_ready ? static_cast<cudaError_t>(launch_invalid(invalid_case))
                   : cudaErrorUnknown;
    test.expect(status == cudaErrorInvalidValue,
                case_label + " returns cudaErrorInvalidValue");
    cudaGraph_t graph = nullptr;
    case_ready = case_ready &&
                 test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                              case_label + " end capture");
    std::size_t node_count = std::numeric_limits<std::size_t>::max();
    if (case_ready) {
      case_ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                               case_label + " count nodes") &&
                   case_ready;
      test.expect(node_count == 0U,
                  case_label + " rejects before enqueue");
      (void)test.cuda_ok(cudaGraphDestroy(graph), case_label + " destroy graph");
    }
    invalid_gate = invalid_gate && case_ready &&
                   status == cudaErrorInvalidValue && node_count == 0U;
  }
  test.expect(invalid_gate,
              label + " all 15 invalid/alias calls capture zero nodes");
  std::cout << "DECODE_FULL_PREPROCESS_WARP_RMS_INVALID: cases=15 "
               "zero_node_cases="
            << (invalid_gate ? 15 : 0)
            << " gate=" << (invalid_gate ? "PASS" : "FAIL") << '\n';
}

void test_full_attention_preproc_rope_fusion_perf(TestContext& test,
                                                  cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_FULL_ATTENTION_PREPROC_ROPE_FUSION_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: full-attention preproc+RoPE fusion performance "
                 "gate; set Q3X_RUN_FULL_ATTENTION_PREPROC_ROPE_FUSION_PERF="
                 "1 to enable\n";
    return;
  }

  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKeyHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kHalfRotary = 32U;
  constexpr std::size_t kMaximumTokens = 16U;
  constexpr std::size_t kFirstPosition = 11U;
  constexpr std::size_t kQueryElementsPerToken =
      kQueryHeads * kDimension;
  constexpr std::size_t kKeyElementsPerToken = kKeyHeads * kDimension;
  constexpr float kEpsilon = 1.0e-6F;
  constexpr std::size_t kWarmupIterations = 64U;
  constexpr std::size_t kMeasuredIterations = 512U;
  constexpr int kMeasurementRounds = 3;
  constexpr double kMinimumCellSpeedup = 1.10;
  constexpr double kMinimumWeightedSpeedup = 1.10;
  constexpr double kMinimumSavedMilliseconds = 1.5;
  const std::string label = "full-attention preproc+RoPE fusion perf";

  ManagedBuffer<std::uint16_t> interleaved_q_gate;
  ManagedBuffer<std::uint16_t> q_weight;
  ManagedBuffer<std::uint16_t> k_weight;
  ManagedBuffer<std::uint16_t> baseline_split_query;
  ManagedBuffer<std::uint16_t> baseline_query;
  ManagedBuffer<std::uint16_t> baseline_gate;
  ManagedBuffer<std::uint16_t> baseline_key;
  ManagedBuffer<std::uint16_t> candidate_query;
  ManagedBuffer<std::uint16_t> candidate_gate;
  ManagedBuffer<std::uint16_t> candidate_key;
  ManagedBuffer<float> cosines;
  ManagedBuffer<float> sines;
  bool ready = test.cuda_ok(
      interleaved_q_gate.allocate(2U * kMaximumTokens *
                                  kQueryElementsPerToken),
      label + " interleaved Q/gate");
  ready = ready &&
          test.cuda_ok(q_weight.allocate(kDimension), label + " Q weight");
  ready = ready &&
          test.cuda_ok(k_weight.allocate(kDimension), label + " K weight");
  ready = ready && test.cuda_ok(
                       baseline_split_query.allocate(kMaximumTokens *
                                                     kQueryElementsPerToken),
                       label + " baseline split query");
  ready = ready && test.cuda_ok(
                       baseline_query.allocate(kMaximumTokens *
                                               kQueryElementsPerToken),
                       label + " baseline query");
  ready = ready && test.cuda_ok(
                       baseline_gate.allocate(kMaximumTokens *
                                              kQueryElementsPerToken),
                       label + " baseline gate");
  ready = ready && test.cuda_ok(
                       baseline_key.allocate(kMaximumTokens *
                                             kKeyElementsPerToken),
                       label + " baseline key");
  ready = ready && test.cuda_ok(
                       candidate_query.allocate(kMaximumTokens *
                                                kQueryElementsPerToken),
                       label + " candidate query");
  ready = ready && test.cuda_ok(
                       candidate_gate.allocate(kMaximumTokens *
                                               kQueryElementsPerToken),
                       label + " candidate gate");
  ready = ready && test.cuda_ok(
                       candidate_key.allocate(kMaximumTokens *
                                              kKeyElementsPerToken),
                       label + " candidate key");
  ready = ready && test.cuda_ok(
                       cosines.allocate((kFirstPosition + kMaximumTokens) *
                                        kHalfRotary),
                       label + " cosine table");
  ready = ready && test.cuda_ok(
                       sines.allocate((kFirstPosition + kMaximumTokens) *
                                      kHalfRotary),
                       label + " sine table");
  if (!ready) {
    return;
  }

  for (std::size_t index = 0U; index < interleaved_q_gate.size(); ++index) {
    interleaved_q_gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 37U + 19U) % 509U) -
                           254) /
        128.0F);
  }
  for (std::size_t dimension = 0U; dimension < kDimension; ++dimension) {
    q_weight[dimension] = encode_bf16(
        static_cast<float>(static_cast<int>((dimension * 17U) % 61U) - 30) /
        64.0F);
    k_weight[dimension] = encode_bf16(
        static_cast<float>(static_cast<int>((dimension * 29U) % 67U) - 33) /
        64.0F);
  }
  for (std::size_t index = 0U; index < cosines.size(); ++index) {
    const float angle =
        static_cast<float>(static_cast<int>((index * 13U) % 1'009U) - 504) /
        2'048.0F;
    cosines[index] = std::cos(angle);
    sines[index] = std::sin(angle);
  }

  struct PerfCell {
    std::size_t token_count;
    double baseline_milliseconds =
        std::numeric_limits<double>::quiet_NaN();
    double candidate_milliseconds =
        std::numeric_limits<double>::quiet_NaN();
    bool timing_finite = false;
  };
  std::array<PerfCell, 4U> cells = {
      PerfCell{1U}, PerfCell{2U}, PerfCell{8U}, PerfCell{16U}};
  bool all_cell_gates = true;

  for (PerfCell& cell : cells) {
    for (std::size_t index = 0U; index < baseline_key.size(); ++index) {
      const std::uint16_t value = encode_bf16(
          static_cast<float>(static_cast<int>((index * 43U + 23U) % 503U) -
                             251) /
          96.0F);
      baseline_key[index] = value;
      candidate_key[index] = value;
    }
    const auto launch_baseline = [&]() {
      int status =
          q3x::runtime::launch_split_interleaved_q_gate_reference_cuda(
              interleaved_q_gate.data(), cell.token_count * kQueryHeads,
              kDimension, baseline_split_query.data(), baseline_gate.data(),
              static_cast<void*>(stream));
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
      status = q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
          baseline_split_query.data(), q_weight.data(),
          cell.token_count * kQueryHeads, kDimension, kEpsilon,
          baseline_query.data(), static_cast<void*>(stream));
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
      status = q3x::runtime::launch_headwise_centered_rms_norm_reference_cuda(
          baseline_key.data(), k_weight.data(),
          cell.token_count * kKeyHeads, kDimension, kEpsilon,
          baseline_key.data(), static_cast<void*>(stream));
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
      return q3x::runtime::
          launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
              baseline_query.data(), baseline_key.data(), cosines.data(),
              sines.data(), kFirstPosition, cell.token_count,
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() {
      return q3x::runtime::
          launch_full_attention_preprocess_24_4_256_64_cuda(
              interleaved_q_gate.data(), candidate_key.data(),
              q_weight.data(), k_weight.data(), kEpsilon,
              candidate_query.data(), candidate_gate.data(), cosines.data(),
              sines.data(), kFirstPosition, cell.token_count,
              static_cast<void*>(stream));
    };
    const std::string cell_label =
        label + " M=" + std::to_string(cell.token_count);

    for (std::size_t iteration = 0U;
         ready && iteration < kWarmupIterations; ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           cell_label + " baseline warmup");
      ready = ready &&
              test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                           cell_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  cell_label + " warmup sync");
    if (!ready) {
      return;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    cell.timing_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          cell_label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          round_label + " baseline pass 1", launch_baseline);
      const float candidate_first = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          round_label + " candidate pass 1", launch_candidate);
      const float candidate_second = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          round_label + " candidate pass 2", launch_candidate);
      const float baseline_second = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          round_label + " baseline pass 2", launch_baseline);
      const bool round_finite =
          std::isfinite(baseline_first) && baseline_first > 0.0F &&
          std::isfinite(candidate_first) && candidate_first > 0.0F &&
          std::isfinite(candidate_second) && candidate_second > 0.0F &&
          std::isfinite(baseline_second) && baseline_second > 0.0F;
      cell.timing_finite = cell.timing_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_FULL_PREPROC_ROPE_FUSION_ROUND: M="
                << cell.token_count << " round=" << round + 1
                << " iterations=" << kMeasuredIterations
                << " baseline1_ms=" << baseline_first
                << " candidate1_ms=" << candidate_first
                << " candidate2_ms=" << candidate_second
                << " baseline2_ms=" << baseline_second << '\n';
    }

    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    cell.baseline_milliseconds = baseline_total / kTimedPasses;
    cell.candidate_milliseconds = candidate_total / kTimedPasses;
    const double speedup =
        cell.baseline_milliseconds / cell.candidate_milliseconds;
    const bool cell_gate =
        cell.timing_finite && std::isfinite(speedup) &&
        cell.baseline_milliseconds > 0.0 &&
        cell.candidate_milliseconds > 0.0 &&
        speedup >= kMinimumCellSpeedup;
    all_cell_gates = all_cell_gates && cell_gate;
    std::cout << "PERF_FULL_PREPROC_ROPE_FUSION_CELL: M="
              << cell.token_count
              << " baseline_four_launches_ms=" << cell.baseline_milliseconds
              << " candidate_one_launch_ms=" << cell.candidate_milliseconds
              << " speedup=" << speedup
              << " required_speedup=" << kMinimumCellSpeedup
              << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
    test.expect(cell_gate,
                cell_label + " clears the 1.10x per-cell gate");
    expect_bf16_bits_equal(test, candidate_query.data(),
                           baseline_query.data(),
                           cell.token_count * kQueryElementsPerToken,
                           cell_label + " repeated query");
    expect_bf16_bits_equal(test, candidate_key.data(), baseline_key.data(),
                           cell.token_count * kKeyElementsPerToken,
                           cell_label + " repeated key");
    expect_bf16_bits_equal(test, candidate_gate.data(), baseline_gate.data(),
                           cell.token_count * kQueryElementsPerToken,
                           cell_label + " repeated raw gate");
  }

  // One measured C=16 run contains 416 decode M=1 calls and 32 prefill
  // calls split evenly between the M=2 remainder and full M=16 tiles.
  constexpr std::array<double, 4U> kProfileCallCounts = {
      416.0, 16.0, 0.0, 16.0};
  double weighted_baseline_milliseconds = 0.0;
  double weighted_candidate_milliseconds = 0.0;
  bool all_timings_finite = true;
  for (std::size_t index = 0U; index < cells.size(); ++index) {
    all_timings_finite = all_timings_finite && cells[index].timing_finite;
    weighted_baseline_milliseconds +=
        cells[index].baseline_milliseconds * kProfileCallCounts[index];
    weighted_candidate_milliseconds +=
        cells[index].candidate_milliseconds * kProfileCallCounts[index];
  }
  const double weighted_speedup =
      weighted_baseline_milliseconds / weighted_candidate_milliseconds;
  const double saved_milliseconds =
      weighted_baseline_milliseconds - weighted_candidate_milliseconds;
  const bool gate =
      all_cell_gates && all_timings_finite &&
      std::isfinite(weighted_speedup) && std::isfinite(saved_milliseconds) &&
      weighted_speedup >= kMinimumWeightedSpeedup &&
      saved_milliseconds >= kMinimumSavedMilliseconds;
  std::cout << "PERF_FULL_PREPROC_ROPE_FUSION_WEIGHTED: decode_M1_calls=416 "
               "prefill_M2_calls=16 prefill_M16_calls=16 "
            << "baseline_estimated_ms=" << weighted_baseline_milliseconds
            << " candidate_estimated_ms=" << weighted_candidate_milliseconds
            << " saved_estimated_ms=" << saved_milliseconds
            << " speedup=" << weighted_speedup
            << " required_speedup=" << kMinimumWeightedSpeedup
            << " required_saved_ms=" << kMinimumSavedMilliseconds
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, label + " clears weighted and 1.5 ms saved gates");
}

void test_full_attention_preproc_warp_rms_perf(TestContext& test,
                                               cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_FULL_ATTENTION_PREPROCESS_WARP_RMS_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout
        << "SKIP: Decode full-attention preprocess warp-RMS performance gate; "
           "set Q3X_RUN_FULL_ATTENTION_PREPROCESS_WARP_RMS_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKeyHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kHalfRotary = 32U;
  constexpr std::size_t kMaximumTokens = 16U;
  constexpr std::size_t kFirstPosition = 11U;
  constexpr std::size_t kQueryElementsPerToken =
      kQueryHeads * kDimension;
  constexpr std::size_t kKeyElementsPerToken = kKeyHeads * kDimension;
  constexpr float kEpsilon = 1.0e-6F;
  constexpr std::size_t kWarmupIterations = 128U;
  constexpr std::size_t kMeasuredIterations = 4'096U;
  constexpr std::size_t kMeasurementRounds = 5U;
  constexpr double kRequiredDecodeMedianSpeedup = 1.10;
  constexpr double kRequiredDecodePerCallDeltaMs = 0.00125;
  constexpr double kDecodeLayersPerToken = 16.0;
  const std::string label =
      "Decode full-attention preprocess warp-RMS formal perf";

  ManagedBuffer<std::uint16_t> interleaved;
  ManagedBuffer<std::uint16_t> q_weight;
  ManagedBuffer<std::uint16_t> k_weight;
  ManagedBuffer<std::uint16_t> production_key;
  ManagedBuffer<std::uint16_t> candidate_key;
  ManagedBuffer<std::uint16_t> production_query;
  ManagedBuffer<std::uint16_t> candidate_query;
  ManagedBuffer<std::uint16_t> production_gate;
  ManagedBuffer<std::uint16_t> candidate_gate;
  ManagedBuffer<float> cosines;
  ManagedBuffer<float> sines;
  bool ready = test.cuda_ok(
      interleaved.allocate(2U * kMaximumTokens * kQueryElementsPerToken),
      label + " allocate interleaved");
  ready = ready && test.cuda_ok(q_weight.allocate(kDimension),
                                label + " allocate Q weight");
  ready = ready && test.cuda_ok(k_weight.allocate(kDimension),
                                label + " allocate K weight");
  ready = ready && test.cuda_ok(
                       production_key.allocate(kMaximumTokens *
                                               kKeyElementsPerToken),
                       label + " allocate production key");
  ready = ready && test.cuda_ok(
                       candidate_key.allocate(kMaximumTokens *
                                              kKeyElementsPerToken),
                       label + " allocate candidate key");
  ready = ready && test.cuda_ok(
                       production_query.allocate(kMaximumTokens *
                                                 kQueryElementsPerToken),
                       label + " allocate production query");
  ready = ready && test.cuda_ok(
                       candidate_query.allocate(kMaximumTokens *
                                                kQueryElementsPerToken),
                       label + " allocate candidate query");
  ready = ready && test.cuda_ok(
                       production_gate.allocate(kMaximumTokens *
                                                kQueryElementsPerToken),
                       label + " allocate production gate");
  ready = ready && test.cuda_ok(
                       candidate_gate.allocate(kMaximumTokens *
                                               kQueryElementsPerToken),
                       label + " allocate candidate gate");
  ready = ready && test.cuda_ok(
                       cosines.allocate((kFirstPosition + kMaximumTokens) *
                                        kHalfRotary),
                       label + " allocate cosine table");
  ready = ready && test.cuda_ok(
                       sines.allocate((kFirstPosition + kMaximumTokens) *
                                      kHalfRotary),
                       label + " allocate sine table");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0U; index < interleaved.size(); ++index) {
    interleaved[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 37U + 19U) % 509U) -
                           254) /
        128.0F);
  }
  for (std::size_t dimension = 0U; dimension < kDimension; ++dimension) {
    q_weight[dimension] = encode_bf16(
        static_cast<float>(static_cast<int>((dimension * 17U) % 61U) - 30) /
        64.0F);
    k_weight[dimension] = encode_bf16(
        static_cast<float>(static_cast<int>((dimension * 29U) % 67U) - 33) /
        64.0F);
  }
  std::fill_n(cosines.data(), cosines.size(), 1.0F);
  std::fill_n(sines.data(), sines.size(), 0.0F);

  constexpr std::array<std::size_t, 4U> kTokenCounts = {1U, 2U, 8U, 16U};
  bool secondary_cells_positive = true;
  bool decode_gate = false;
  for (const std::size_t token_count : kTokenCounts) {
    const std::size_t query_elements =
        token_count * kQueryElementsPerToken;
    const std::size_t key_elements = token_count * kKeyElementsPerToken;
    for (std::size_t index = 0U; index < key_elements; ++index) {
      const std::uint16_t value = encode_bf16(
          static_cast<float>(static_cast<int>((index * 43U + 23U) % 503U) -
                             251) /
          96.0F);
      production_key[index] = value;
      candidate_key[index] = value;
    }
    const auto launch_production = [&]() {
      return q3x::runtime::launch_full_attention_preprocess_24_4_256_64_cuda(
          interleaved.data(), production_key.data(), q_weight.data(),
          k_weight.data(), kEpsilon, production_query.data(),
          production_gate.data(), cosines.data(), sines.data(), kFirstPosition,
          token_count, static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() {
      return q3x::runtime::
          launch_full_attention_preprocess_warp_rms_24_4_256_64_test_cuda(
              interleaved.data(), candidate_key.data(), q_weight.data(),
              k_weight.data(), kEpsilon, candidate_query.data(),
              candidate_gate.data(), cosines.data(), sines.data(),
              kFirstPosition, token_count, static_cast<void*>(stream));
    };
    const std::string cell_label =
        label + " M=" + std::to_string(token_count);
    for (std::size_t iteration = 0U;
         ready && iteration < kWarmupIterations; ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_production()),
                           cell_label + " production warmup");
      ready = ready &&
              test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                           cell_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  cell_label + " warmup synchronize");
    if (!ready) {
      return;
    }

    std::array<double, kMeasurementRounds> paired_speedups{};
    std::array<double, kMeasurementRounds> paired_deltas{};
    bool all_rounds_positive = true;
    for (std::size_t round = 0U; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          cell_label + " round=" + std::to_string(round + 1U);
      float production_first = 0.0F;
      float production_second = 0.0F;
      float candidate_first = 0.0F;
      float candidate_second = 0.0F;
      const bool baseline_outer = (round % 2U) == 0U;
      if (baseline_outer) {
        production_first = measure_cuda_span_milliseconds(
            test, stream, kMeasuredIterations, round_label + " B1",
            launch_production);
        candidate_first = measure_cuda_span_milliseconds(
            test, stream, kMeasuredIterations, round_label + " C1",
            launch_candidate);
        candidate_second = measure_cuda_span_milliseconds(
            test, stream, kMeasuredIterations, round_label + " C2",
            launch_candidate);
        production_second = measure_cuda_span_milliseconds(
            test, stream, kMeasuredIterations, round_label + " B2",
            launch_production);
      } else {
        candidate_first = measure_cuda_span_milliseconds(
            test, stream, kMeasuredIterations, round_label + " C1",
            launch_candidate);
        production_first = measure_cuda_span_milliseconds(
            test, stream, kMeasuredIterations, round_label + " B1",
            launch_production);
        production_second = measure_cuda_span_milliseconds(
            test, stream, kMeasuredIterations, round_label + " B2",
            launch_production);
        candidate_second = measure_cuda_span_milliseconds(
            test, stream, kMeasuredIterations, round_label + " C2",
            launch_candidate);
      }
      const double production_pair =
          (static_cast<double>(production_first) +
           static_cast<double>(production_second)) /
          2.0;
      const double candidate_pair =
          (static_cast<double>(candidate_first) +
           static_cast<double>(candidate_second)) /
          2.0;
      paired_speedups[round] = production_pair / candidate_pair;
      paired_deltas[round] = production_pair - candidate_pair;
      const bool round_positive =
          std::isfinite(production_pair) && production_pair > 0.0 &&
          std::isfinite(candidate_pair) && candidate_pair > 0.0 &&
          paired_deltas[round] > 0.0;
      all_rounds_positive = all_rounds_positive && round_positive;
      std::cout << "DECODE_FULL_PREPROCESS_WARP_RMS_PERF_ROUND: M="
                << token_count << " round=" << round + 1U
                << " order=" << (baseline_outer ? "B-C-C-B" : "C-B-B-C")
                << " iterations=" << kMeasuredIterations
                << " B1_call_ms=" << production_first
                << " C1_call_ms=" << candidate_first
                << " C2_call_ms=" << candidate_second
                << " B2_call_ms=" << production_second
                << " paired_speedup=" << paired_speedups[round]
                << " paired_delta_ms_per_call=" << paired_deltas[round]
                << " positive=" << (round_positive ? "true" : "false")
                << '\n';
    }
    auto sorted_speedups = paired_speedups;
    auto sorted_deltas = paired_deltas;
    std::sort(sorted_speedups.begin(), sorted_speedups.end());
    std::sort(sorted_deltas.begin(), sorted_deltas.end());
    const double median_speedup = sorted_speedups[2U];
    const double median_delta = sorted_deltas[2U];
    const double projected_decode_delta =
        median_delta * kDecodeLayersPerToken;
    const bool cell_gate =
        all_rounds_positive &&
        (token_count != 1U ||
         (median_speedup >= kRequiredDecodeMedianSpeedup &&
          median_delta >= kRequiredDecodePerCallDeltaMs));
    if (token_count == 1U) {
      decode_gate = cell_gate;
    } else {
      secondary_cells_positive = secondary_cells_positive && cell_gate;
    }
    test.expect(cell_gate,
                cell_label +
                    (token_count == 1U
                         ? " clears 1.10x, 1.25-us, every-round Decode gates"
                         : " is positive in every supporting round"));
    expect_bf16_bits_equal(test, candidate_query.data(),
                           production_query.data(), query_elements,
                           cell_label + " repeated query");
    expect_bf16_bits_equal(test, candidate_key.data(), production_key.data(),
                           key_elements, cell_label + " repeated key");
    expect_bf16_bits_equal(test, candidate_gate.data(), production_gate.data(),
                           query_elements, cell_label + " repeated gate");
    std::cout << "DECODE_FULL_PREPROCESS_WARP_RMS_PERF_CELL: M="
              << token_count << " paired_median_speedup=" << median_speedup
              << " paired_median_delta_ms_per_call=" << median_delta
              << " projected_16_layer_delta_ms_per_token="
              << projected_decode_delta
              << " all_rounds_positive="
              << (all_rounds_positive ? "true" : "false")
              << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
  }
  const bool overall_gate = decode_gate && secondary_cells_positive;
  test.expect(overall_gate,
              label + " clears Decode selection and supporting-cell gates");
  std::cout << "DECODE_FULL_PREPROCESS_WARP_RMS_PERF_SUMMARY: "
               "decode_M1_gate="
            << (decode_gate ? "PASS" : "FAIL")
            << " supporting_M2_M8_M16_gate="
            << (secondary_cells_positive ? "PASS" : "FAIL")
            << " required_decode_median_speedup="
            << kRequiredDecodeMedianSpeedup
            << " required_decode_delta_ms_per_call="
            << kRequiredDecodePerCallDeltaMs
            << " required_projected_delta_ms_per_token="
            << kRequiredDecodePerCallDeltaMs * kDecodeLayersPerToken
            << " gate=" << (overall_gate ? "PASS" : "FAIL") << '\n';
}

enum class QkRopeTileFixture {
  kFinite,
  kNonfinite,
};

void run_qk_rope_tile_exact_case(TestContext& test, cudaStream_t stream,
                                 const std::size_t token_count,
                                 const std::size_t first_position,
                                 const QkRopeTileFixture fixture) {
  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKeyHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kHalfRotary = 32U;
  constexpr std::size_t kQueryElementsPerToken =
      kQueryHeads * kDimension;
  constexpr std::size_t kKeyElementsPerToken = kKeyHeads * kDimension;
  const std::size_t query_elements = token_count * kQueryElementsPerToken;
  const std::size_t key_elements = token_count * kKeyElementsPerToken;
  const std::size_t table_elements =
      (first_position + token_count) * kHalfRotary;
  const std::string fixture_name =
      fixture == QkRopeTileFixture::kFinite ? "finite" : "nonfinite";
  const std::string label =
      "Q/K RoPE tile M=" + std::to_string(token_count) +
      " first_position=" + std::to_string(first_position) + " " +
      fixture_name;

  GuardedBf16Buffer baseline_query;
  GuardedBf16Buffer baseline_key;
  GuardedBf16Buffer candidate_query;
  GuardedBf16Buffer candidate_key;
  ManagedBuffer<float> cosines;
  ManagedBuffer<float> sines;
  bool ready = baseline_query.allocate(test, query_elements,
                                       label + " baseline query");
  ready = ready && baseline_key.allocate(test, key_elements,
                                          label + " baseline key");
  ready = ready && candidate_query.allocate(test, query_elements,
                                             label + " candidate query");
  ready = ready && candidate_key.allocate(test, key_elements,
                                           label + " candidate key");
  ready = ready && test.cuda_ok(cosines.allocate(table_elements),
                                label + " cosine table");
  ready = ready && test.cuda_ok(sines.allocate(table_elements),
                                label + " sine table");
  if (!ready) {
    return;
  }

  baseline_query.initialize(0U, 0x2a15U, 0xd4e9U);
  baseline_key.initialize(0U, 0x35a6U, 0xcb59U);
  candidate_query.initialize(0U, 0x4b27U, 0xb4d8U);
  candidate_key.initialize(0U, 0x5c38U, 0xa3c7U);
  for (std::size_t index = 0U; index < query_elements; ++index) {
    baseline_query.data()[index] = encode_bf16(
        static_cast<float>(
            static_cast<int>((index * 37U + token_count * 11U) % 257U) -
            128) /
        64.0F);
  }
  for (std::size_t index = 0U; index < key_elements; ++index) {
    baseline_key.data()[index] = encode_bf16(
        static_cast<float>(
            static_cast<int>((index * 43U + first_position * 7U) % 251U) -
            125) /
        32.0F);
  }

  if (fixture == QkRopeTileFixture::kNonfinite) {
    constexpr std::array<std::uint16_t, 8U> kSpecialBf16Bits = {
        0x7f80U, 0xff80U, 0x7fc1U, 0xffc1U,
        0x0000U, 0x8000U, 0x0001U, 0x8001U};
    for (std::size_t token = 0U; token < token_count; ++token) {
      for (std::size_t sample = 0U; sample < kSpecialBf16Bits.size();
           ++sample) {
        const std::size_t query_head =
            (sample * 5U + token) % kQueryHeads;
        const std::size_t key_head = (sample + token) % kKeyHeads;
        const std::size_t pair = (sample * 3U + token) % kHalfRotary;
        const std::size_t query_offset =
            token * kQueryElementsPerToken + query_head * kDimension + pair;
        const std::size_t key_offset =
            token * kKeyElementsPerToken + key_head * kDimension + pair;
        baseline_query.data()[query_offset] = kSpecialBf16Bits[sample];
        baseline_query.data()[query_offset + kHalfRotary] =
            kSpecialBf16Bits[(sample + 3U) % kSpecialBf16Bits.size()];
        baseline_key.data()[key_offset] =
            kSpecialBf16Bits[(sample + 1U) % kSpecialBf16Bits.size()];
        baseline_key.data()[key_offset + kHalfRotary] =
            kSpecialBf16Bits[(sample + 4U) % kSpecialBf16Bits.size()];
      }
    }
  }

  std::copy_n(baseline_query.data(), query_elements,
              candidate_query.data());
  std::copy_n(baseline_key.data(), key_elements, candidate_key.data());
  const std::vector<std::uint16_t> original_query(
      baseline_query.data(), baseline_query.data() + query_elements);
  const std::vector<std::uint16_t> original_key(
      baseline_key.data(), baseline_key.data() + key_elements);
  for (std::size_t index = 0U; index < table_elements; ++index) {
    const float angle =
        static_cast<float>(static_cast<int>(index % 1'009U) - 504) /
        2'048.0F;
    cosines[index] = std::cos(angle);
    sines[index] = std::sin(angle);
  }
  const std::vector<float> original_cosines(cosines.data(),
                                             cosines.data() + table_elements);
  const std::vector<float> original_sines(sines.data(),
                                           sines.data() + table_elements);

  const auto launch_baseline = [&]() {
    for (std::size_t token = 0U; token < token_count; ++token) {
      const std::size_t table_offset =
          (first_position + token) * kHalfRotary;
      const int query_status =
          q3x::runtime::launch_partial_neox_rope_256_64_reference_cuda(
              baseline_query.data() + token * kQueryElementsPerToken,
              cosines.data() + table_offset, sines.data() + table_offset,
              kQueryHeads,
              baseline_query.data() + token * kQueryElementsPerToken,
              static_cast<void*>(stream));
      if (query_status != static_cast<int>(cudaSuccess)) {
        return query_status;
      }
      const int key_status =
          q3x::runtime::launch_partial_neox_rope_256_64_reference_cuda(
              baseline_key.data() + token * kKeyElementsPerToken,
              cosines.data() + table_offset, sines.data() + table_offset,
              kKeyHeads,
              baseline_key.data() + token * kKeyElementsPerToken,
              static_cast<void*>(stream));
      if (key_status != static_cast<int>(cudaSuccess)) {
        return key_status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  const auto launch_candidate = [&]() {
    return q3x::runtime::
        launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
            candidate_query.data(), candidate_key.data(), cosines.data(),
            sines.data(), first_position, token_count,
            static_cast<void*>(stream));
  };

  const bool baseline_ready =
      launch_after_stale(test, stream, label + " old 2xM launches",
                         launch_baseline);
  const bool candidate_ready =
      launch_after_stale(test, stream, label + " one tile launch",
                         launch_candidate);
  if (baseline_ready && candidate_ready) {
    expect_bf16_bits_equal(test, candidate_query.data(),
                           baseline_query.data(), query_elements,
                           label + " query");
    expect_bf16_bits_equal(test, candidate_key.data(), baseline_key.data(),
                           key_elements, label + " key");

    bool query_nonrotary_unchanged = true;
    bool key_nonrotary_unchanged = true;
    for (std::size_t token = 0U; token < token_count; ++token) {
      for (std::size_t head = 0U; head < kQueryHeads; ++head) {
        const std::size_t offset =
            token * kQueryElementsPerToken + head * kDimension;
        query_nonrotary_unchanged =
            query_nonrotary_unchanged &&
            std::equal(candidate_query.data() + offset + 2U * kHalfRotary,
                       candidate_query.data() + offset + kDimension,
                       original_query.data() + offset + 2U * kHalfRotary);
      }
      for (std::size_t head = 0U; head < kKeyHeads; ++head) {
        const std::size_t offset =
            token * kKeyElementsPerToken + head * kDimension;
        key_nonrotary_unchanged =
            key_nonrotary_unchanged &&
            std::equal(candidate_key.data() + offset + 2U * kHalfRotary,
                       candidate_key.data() + offset + kDimension,
                       original_key.data() + offset + 2U * kHalfRotary);
      }
    }
    test.expect(query_nonrotary_unchanged,
                label + " query non-rotary raw BF16 is unchanged");
    test.expect(key_nonrotary_unchanged,
                label + " key non-rotary raw BF16 is unchanged");
  }

  baseline_query.expect_guards(test, label + " baseline query");
  baseline_key.expect_guards(test, label + " baseline key");
  candidate_query.expect_guards(test, label + " candidate query");
  candidate_key.expect_guards(test, label + " candidate key");
  test.expect(std::memcmp(cosines.data(), original_cosines.data(),
                          table_elements * sizeof(float)) == 0,
              label + " cosine table is unchanged");
  test.expect(std::memcmp(sines.data(), original_sines.data(),
                          table_elements * sizeof(float)) == 0,
              label + " sine table is unchanged");
}

void test_qk_rope_tile_exact(TestContext& test, cudaStream_t stream) {
  run_qk_rope_tile_exact_case(test, stream, 1U, 0U,
                              QkRopeTileFixture::kFinite);
  run_qk_rope_tile_exact_case(test, stream, 2U, 5U,
                              QkRopeTileFixture::kFinite);
  run_qk_rope_tile_exact_case(test, stream, 8U, 17U,
                              QkRopeTileFixture::kFinite);
  run_qk_rope_tile_exact_case(test, stream, 16U, 33U,
                              QkRopeTileFixture::kFinite);
  run_qk_rope_tile_exact_case(test, stream, 2U, 9U,
                              QkRopeTileFixture::kNonfinite);
}

void test_qk_rope_tile_perf(TestContext& test, cudaStream_t stream) {
  const char* const enabled = std::getenv("Q3X_RUN_QK_ROPE_TILE_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: Q/K RoPE tile performance gate; set "
                 "Q3X_RUN_QK_ROPE_TILE_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKeyHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kHalfRotary = 32U;
  constexpr std::size_t kMaximumTokens = 16U;
  constexpr std::size_t kFirstPosition = 11U;
  constexpr std::size_t kQueryElementsPerToken =
      kQueryHeads * kDimension;
  constexpr std::size_t kKeyElementsPerToken = kKeyHeads * kDimension;
  constexpr std::size_t kWarmupIterations = 64U;
  constexpr std::size_t kMeasuredIterations = 512U;
  constexpr int kMeasurementRounds = 3;
  constexpr double kMinimumCellSpeedup = 1.10;
  constexpr double kMinimumWeightedSpeedup = 1.20;
  constexpr double kMinimumSavedMilliseconds = 1.5;
  const std::string label = "Q/K RoPE tile perf";

  ManagedBuffer<std::uint16_t> baseline_query;
  ManagedBuffer<std::uint16_t> baseline_key;
  ManagedBuffer<std::uint16_t> candidate_query;
  ManagedBuffer<std::uint16_t> candidate_key;
  ManagedBuffer<float> cosines;
  ManagedBuffer<float> sines;
  bool ready = test.cuda_ok(
      baseline_query.allocate(kMaximumTokens * kQueryElementsPerToken),
      label + " baseline query");
  ready = ready && test.cuda_ok(
                       baseline_key.allocate(kMaximumTokens *
                                             kKeyElementsPerToken),
                       label + " baseline key");
  ready = ready && test.cuda_ok(
                       candidate_query.allocate(kMaximumTokens *
                                                kQueryElementsPerToken),
                       label + " candidate query");
  ready = ready && test.cuda_ok(
                       candidate_key.allocate(kMaximumTokens *
                                              kKeyElementsPerToken),
                       label + " candidate key");
  ready = ready && test.cuda_ok(
                       cosines.allocate((kFirstPosition + kMaximumTokens) *
                                        kHalfRotary),
                       label + " cosine table");
  ready = ready && test.cuda_ok(
                       sines.allocate((kFirstPosition + kMaximumTokens) *
                                      kHalfRotary),
                       label + " sine table");
  if (!ready) {
    return;
  }

  for (std::size_t index = 0U; index < baseline_query.size(); ++index) {
    baseline_query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 17U + 5U) % 127U) -
                           63) /
        64.0F);
    candidate_query[index] = baseline_query[index];
  }
  for (std::size_t index = 0U; index < baseline_key.size(); ++index) {
    baseline_key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 7U) % 113U) -
                           56) /
        64.0F);
    candidate_key[index] = baseline_key[index];
  }
  std::fill_n(cosines.data(), cosines.size(), 1.0F);
  std::fill_n(sines.data(), sines.size(), 0.0F);

  struct PerfCell {
    std::size_t token_count;
    double baseline_milliseconds =
        std::numeric_limits<double>::quiet_NaN();
    double candidate_milliseconds =
        std::numeric_limits<double>::quiet_NaN();
    bool timing_finite = false;
  };
  std::array<PerfCell, 4U> cells = {
      PerfCell{1U}, PerfCell{2U}, PerfCell{8U}, PerfCell{16U}};
  bool all_cell_gates = true;

  for (PerfCell& cell : cells) {
    const auto launch_baseline = [&]() {
      for (std::size_t token = 0U; token < cell.token_count; ++token) {
        const std::size_t table_offset =
            (kFirstPosition + token) * kHalfRotary;
        const int query_status =
            q3x::runtime::launch_partial_neox_rope_256_64_reference_cuda(
                baseline_query.data() + token * kQueryElementsPerToken,
                cosines.data() + table_offset, sines.data() + table_offset,
                kQueryHeads,
                baseline_query.data() + token * kQueryElementsPerToken,
                static_cast<void*>(stream));
        if (query_status != static_cast<int>(cudaSuccess)) {
          return query_status;
        }
        const int key_status =
            q3x::runtime::launch_partial_neox_rope_256_64_reference_cuda(
                baseline_key.data() + token * kKeyElementsPerToken,
                cosines.data() + table_offset, sines.data() + table_offset,
                kKeyHeads,
                baseline_key.data() + token * kKeyElementsPerToken,
                static_cast<void*>(stream));
        if (key_status != static_cast<int>(cudaSuccess)) {
          return key_status;
        }
      }
      return static_cast<int>(cudaSuccess);
    };
    const auto launch_candidate = [&]() {
      return q3x::runtime::
          launch_qk_partial_neox_rope_tile_24_4_256_64_cuda(
              candidate_query.data(), candidate_key.data(), cosines.data(),
              sines.data(), kFirstPosition, cell.token_count,
              static_cast<void*>(stream));
    };
    const std::string cell_label =
        label + " M=" + std::to_string(cell.token_count);

    for (std::size_t iteration = 0U;
         ready && iteration < kWarmupIterations; ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           cell_label + " baseline warmup");
      ready = ready &&
              test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                           cell_label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  cell_label + " warmup sync");
    if (!ready) {
      return;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    cell.timing_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          cell_label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          round_label + " baseline pass 1", launch_baseline);
      const float candidate_first = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          round_label + " candidate pass 1", launch_candidate);
      const float candidate_second = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          round_label + " candidate pass 2", launch_candidate);
      const float baseline_second = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations,
          round_label + " baseline pass 2", launch_baseline);
      const bool round_finite =
          std::isfinite(baseline_first) && baseline_first > 0.0F &&
          std::isfinite(candidate_first) && candidate_first > 0.0F &&
          std::isfinite(candidate_second) && candidate_second > 0.0F &&
          std::isfinite(baseline_second) && baseline_second > 0.0F;
      cell.timing_finite = cell.timing_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_QK_ROPE_TILE_ROUND: M=" << cell.token_count
                << " round=" << round + 1
                << " iterations=" << kMeasuredIterations
                << " baseline1_ms=" << baseline_first
                << " candidate1_ms=" << candidate_first
                << " candidate2_ms=" << candidate_second
                << " baseline2_ms=" << baseline_second << '\n';
    }

    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    cell.baseline_milliseconds = baseline_total / kTimedPasses;
    cell.candidate_milliseconds = candidate_total / kTimedPasses;
    const double speedup =
        cell.baseline_milliseconds / cell.candidate_milliseconds;
    const bool cell_gate =
        cell.timing_finite && std::isfinite(speedup) &&
        cell.baseline_milliseconds > 0.0 &&
        cell.candidate_milliseconds > 0.0 &&
        speedup >= kMinimumCellSpeedup;
    all_cell_gates = all_cell_gates && cell_gate;
    std::cout << "PERF_QK_ROPE_TILE_CELL: M=" << cell.token_count
              << " baseline_old_2xM_ms=" << cell.baseline_milliseconds
              << " candidate_one_tile_ms=" << cell.candidate_milliseconds
              << " speedup=" << speedup
              << " required_speedup=" << kMinimumCellSpeedup
              << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
    test.expect(cell_gate,
                cell_label + " clears the 1.10x per-cell gate");
  }

  // One real C=16 profile: 416 decode tiles at M=1 and 32 prefill tiles,
  // split evenly between the full M=16 tile and the M=2 remainder.
  constexpr std::array<double, 4U> kProfileCallCounts = {
      416.0, 16.0, 0.0, 16.0};
  double weighted_baseline_milliseconds = 0.0;
  double weighted_candidate_milliseconds = 0.0;
  bool all_timings_finite = true;
  for (std::size_t index = 0U; index < cells.size(); ++index) {
    all_timings_finite = all_timings_finite && cells[index].timing_finite;
    weighted_baseline_milliseconds +=
        cells[index].baseline_milliseconds * kProfileCallCounts[index];
    weighted_candidate_milliseconds +=
        cells[index].candidate_milliseconds * kProfileCallCounts[index];
  }
  const double weighted_speedup =
      weighted_baseline_milliseconds / weighted_candidate_milliseconds;
  const double saved_milliseconds =
      weighted_baseline_milliseconds - weighted_candidate_milliseconds;
  const bool gate =
      all_cell_gates && all_timings_finite &&
      std::isfinite(weighted_speedup) &&
      std::isfinite(saved_milliseconds) &&
      weighted_speedup >= kMinimumWeightedSpeedup &&
      saved_milliseconds >= kMinimumSavedMilliseconds;
  std::cout << "PERF_QK_ROPE_TILE_WEIGHTED: decode_M1_calls=416 "
               "prefill_M2_calls=16 prefill_M16_calls=16 "
            << "baseline_estimated_ms=" << weighted_baseline_milliseconds
            << " candidate_estimated_ms=" << weighted_candidate_milliseconds
            << " saved_estimated_ms=" << saved_milliseconds
            << " speedup=" << weighted_speedup
            << " required_speedup=" << kMinimumWeightedSpeedup
            << " required_saved_ms=" << kMinimumSavedMilliseconds
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate,
              label + " clears weighted 1.20x and 1.5 ms saved gates");

  expect_bf16_bits_equal(test, candidate_query.data(), baseline_query.data(),
                         baseline_query.size(),
                         label + " repeated identity query");
  expect_bf16_bits_equal(test, candidate_key.data(), baseline_key.data(),
                         baseline_key.size(),
                         label + " repeated identity key");
}

void test_softmax(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kRows = 5U;
  constexpr std::size_t kColumns = 37U;
  ManagedBuffer<float> values;
  ManagedBuffer<float> output;
  bool ready = test.cuda_ok(values.allocate(kRows * kColumns),
                            "softmax allocate input");
  ready = ready && test.cuda_ok(output.allocate(kRows * kColumns),
                                "softmax allocate output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < values.size(); ++index) {
    values[index] = 1000.0F +
                    static_cast<float>(static_cast<int>(index % 43U) - 21) /
                        8.0F;
  }
  std::vector<float> cpu(values.size());
  (void)q3x::runtime::softmax_reference_cpu(
      values.data(), kRows, kColumns, cpu.data());
  ready = launch_after_stale(test, stream, "stable softmax", [&]() {
    return q3x::runtime::launch_softmax_reference_cuda(
        values.data(), kRows, kColumns, output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < values.size(); ++index) {
      test.expect_near(output[index], cpu[index], 3.0e-6F,
                       "CUDA softmax element " + std::to_string(index));
    }
  }
}

void test_attention(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kSequence = 17U;
  constexpr std::size_t kDimension = 256U;
  constexpr float kScale = 0.0625F;
  const std::size_t query_elements = kQueryHeads * kDimension;
  const std::size_t cache_elements = kSequence * kKvHeads * kDimension;
  const std::size_t scratch_elements = kQueryHeads * kSequence;
  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<float> scratch;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(query.allocate(query_elements),
                            "attention allocate query");
  ready = ready &&
          test.cuda_ok(key.allocate(cache_elements), "attention allocate key");
  ready = ready && test.cuda_ok(value.allocate(cache_elements),
                                "attention allocate value");
  ready = ready && test.cuda_ok(scratch.allocate(scratch_elements),
                                "attention allocate scratch");
  ready = ready && test.cuda_ok(output.allocate(query_elements),
                                "attention allocate output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0; index < query.size(); ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 5U) % 29U) - 14) /
        32.0F);
  }
  for (std::size_t index = 0; index < key.size(); ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 7U) % 31U) - 15) /
        32.0F);
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 11U) % 37U) - 18) /
        32.0F);
  }
  std::vector<float> cpu_scratch(scratch_elements);
  std::vector<std::uint16_t> cpu_output(query_elements);
  (void)q3x::runtime::gqa_attention_reference_cpu(
      query.data(), key.data(), value.data(), kQueryHeads, kKvHeads,
      kSequence, kDimension, kScale, cpu_scratch.data(), cpu_scratch.size(),
      cpu_output.data());
  ready = launch_after_stale(test, stream, "GQA attention", [&]() {
    return q3x::runtime::launch_gqa_attention_reference_cuda(
        query.data(), key.data(), value.data(), kQueryHeads, kKvHeads,
        kSequence, kDimension, kScale, scratch.data(), scratch.size(),
        output.data(), static_cast<void*>(stream));
  });
  if (ready) {
    for (std::size_t index = 0; index < scratch.size(); ++index) {
      test.expect_near(scratch[index], cpu_scratch[index], 8.0e-6F,
                       "CUDA GQA probability " + std::to_string(index));
    }
    for (std::size_t index = 0; index < output.size(); ++index) {
      expect_bf16_match(test, output[index], cpu_output[index],
                        "CUDA GQA output " + std::to_string(index));
    }
  }
}

struct AttentionScoreKernelResources {
  int registers = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks_per_multiprocessor = 0;
};

using AttentionScoreTestLaunch = int (*)(
    const std::uint16_t*, const std::uint16_t*, std::size_t, std::size_t,
    std::size_t, std::size_t, float, float*, void*) noexcept;

using AttentionValueTestLaunch = int (*)(
    const std::uint16_t*, const float*, std::size_t, std::size_t,
    std::size_t, std::size_t, std::uint16_t*, void*) noexcept;

struct AttentionScoreGraphTopology {
  void* function = nullptr;
  dim3 grid{0U, 0U, 0U};
  dim3 block{0U, 0U, 0U};
  unsigned int dynamic_shared_bytes = 0U;
};

struct GqaGraphTopology {
  AttentionScoreGraphTopology root;
  std::vector<void*> non_root_functions;
};

void expect_invalid_attention_score_graph(
    TestContext& test, cudaStream_t stream, const std::string& label,
    const AttentionScoreTestLaunch launch,
    const std::uint16_t* const query,
    const std::uint16_t* const key,
    const std::size_t query_heads,
    const std::size_t kv_heads,
    const std::size_t sequence_length,
    const std::size_t dimension,
    const float scale,
    float* const scores) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return;
  }
  const cudaError_t launch_status = static_cast<cudaError_t>(launch(
      query, key, query_heads, kv_heads, sequence_length, dimension, scale,
      scores, static_cast<void*>(stream)));
  test.expect(launch_status == cudaErrorInvalidValue,
              label + " returns cudaErrorInvalidValue");
  cudaGraph_t graph = nullptr;
  if (!test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                    label + " end capture")) {
    return;
  }
  std::size_t node_count = 0U;
  if (test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                   label + " count nodes")) {
    test.expect(node_count == 0U, label + " captures zero graph nodes");
  }
  (void)test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph");
}

[[nodiscard]] bool capture_attention_score_graph_topology(
    TestContext& test, cudaStream_t stream, const std::string& label,
    const AttentionScoreTestLaunch launch,
    const std::uint16_t* const query,
    const std::uint16_t* const key,
    const std::size_t sequence_length,
    float* const scores,
    AttentionScoreGraphTopology& topology) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return false;
  }
  const cudaError_t launch_status = static_cast<cudaError_t>(launch(
      query, key, 24U, 4U, sequence_length, 256U, 0.0625F, scores,
      static_cast<void*>(stream)));
  test.expect(launch_status == cudaSuccess, label + " launch succeeds");
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                            label + " end capture");
  if (!ready) {
    return false;
  }
  std::size_t node_count = 0U;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                       label + " count nodes");
  test.expect(node_count == 1U, label + " captures exactly one node");
  if (ready && node_count == 1U) {
    cudaGraphNode_t node = nullptr;
    std::size_t capacity = 1U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, &node, &capacity),
                         label + " fetch node");
    test.expect(capacity == 1U, label + " fetches one node");
    cudaGraphNodeType node_type = cudaGraphNodeTypeEmpty;
    ready = ready && test.cuda_ok(cudaGraphNodeGetType(node, &node_type),
                                  label + " read node type");
    test.expect(node_type == cudaGraphNodeTypeKernel,
                label + " node is a kernel");
    cudaKernelNodeParams parameters{};
    ready = ready && test.cuda_ok(
                         cudaGraphKernelNodeGetParams(node, &parameters),
                         label + " read kernel parameters");
    if (ready) {
      topology.function = parameters.func;
      topology.grid = parameters.gridDim;
      topology.block = parameters.blockDim;
      topology.dynamic_shared_bytes = parameters.sharedMemBytes;
    }
  } else {
    ready = false;
  }
  ready = test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph") &&
          ready;
  return ready;
}

void expect_invalid_attention_value_graph(
    TestContext& test, cudaStream_t stream, const std::string& label,
    const AttentionValueTestLaunch launch,
    const std::uint16_t* const value_cache,
    const float* const probabilities,
    const std::size_t query_heads,
    const std::size_t kv_heads,
    const std::size_t sequence_length,
    const std::size_t dimension,
    std::uint16_t* const output) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return;
  }
  const cudaError_t launch_status = static_cast<cudaError_t>(launch(
      value_cache, probabilities, query_heads, kv_heads, sequence_length,
      dimension, output, static_cast<void*>(stream)));
  test.expect(launch_status == cudaErrorInvalidValue,
              label + " returns cudaErrorInvalidValue");
  cudaGraph_t graph = nullptr;
  if (!test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                    label + " end capture")) {
    return;
  }
  std::size_t node_count = 0U;
  if (test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                   label + " count nodes")) {
    test.expect(node_count == 0U, label + " captures zero graph nodes");
  }
  (void)test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph");
}

[[nodiscard]] bool capture_attention_value_graph_topology(
    TestContext& test, cudaStream_t stream, const std::string& label,
    const AttentionValueTestLaunch launch,
    const std::uint16_t* const value_cache,
    const float* const probabilities,
    const std::size_t sequence_length,
    std::uint16_t* const output,
    AttentionScoreGraphTopology& topology) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return false;
  }
  const cudaError_t launch_status = static_cast<cudaError_t>(launch(
      value_cache, probabilities, 24U, 4U, sequence_length, 256U, output,
      static_cast<void*>(stream)));
  test.expect(launch_status == cudaSuccess, label + " launch succeeds");
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                            label + " end capture");
  if (!ready) {
    return false;
  }
  std::size_t node_count = 0U;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                       label + " count nodes");
  test.expect(node_count == 1U, label + " captures exactly one node");
  if (ready && node_count == 1U) {
    cudaGraphNode_t node = nullptr;
    std::size_t capacity = 1U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, &node, &capacity),
                         label + " fetch node");
    test.expect(capacity == 1U, label + " fetches one node");
    cudaGraphNodeType node_type = cudaGraphNodeTypeEmpty;
    ready = ready && test.cuda_ok(cudaGraphNodeGetType(node, &node_type),
                                  label + " read node type");
    test.expect(node_type == cudaGraphNodeTypeKernel,
                label + " node is a kernel");
    cudaKernelNodeParams parameters{};
    ready = ready && test.cuda_ok(
                         cudaGraphKernelNodeGetParams(node, &parameters),
                         label + " read kernel parameters");
    if (ready) {
      topology.function = parameters.func;
      topology.grid = parameters.gridDim;
      topology.block = parameters.blockDim;
      topology.dynamic_shared_bytes = parameters.sharedMemBytes;
    }
  } else {
    ready = false;
  }
  ready = test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph") &&
          ready;
  return ready;
}

[[nodiscard]] bool capture_gqa_graph_topology(
    TestContext& test, cudaStream_t stream, const std::string& label,
    const std::uint16_t* const query,
    const std::uint16_t* const key,
    const std::uint16_t* const value,
    const std::size_t sequence_length,
    float* const scratch,
    const std::size_t scratch_elements,
    std::uint16_t* const output,
    GqaGraphTopology& topology,
    const std::size_t query_head_count = 24U,
    const std::size_t kv_head_count = 4U,
    const std::size_t head_dimension = 256U) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return false;
  }
  const cudaError_t launch_status = static_cast<cudaError_t>(
      q3x::runtime::launch_gqa_attention_reference_cuda(
          query, key, value, query_head_count, kv_head_count, sequence_length,
          head_dimension, 0.0625F, scratch, scratch_elements, output,
          static_cast<void*>(stream)));
  test.expect(launch_status == cudaSuccess, label + " launch succeeds");
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                            label + " end capture");
  if (!ready) {
    return false;
  }

  std::size_t node_count = 0U;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                       label + " count nodes");
  test.expect(node_count == 3U, label + " captures three kernel nodes");
  std::vector<cudaGraphNode_t> nodes(node_count);
  std::size_t node_capacity = nodes.size();
  if (ready && node_count != 0U) {
    ready = test.cuda_ok(
        cudaGraphGetNodes(graph, nodes.data(), &node_capacity),
        label + " fetch nodes");
    test.expect(node_capacity == node_count,
                label + " fetches every graph node");
  }

  std::size_t root_count = 0U;
  ready = ready && test.cuda_ok(
                       cudaGraphGetRootNodes(graph, nullptr, &root_count),
                       label + " count roots");
  test.expect(root_count == 1U, label + " has exactly one root");
  cudaGraphNode_t root = nullptr;
  std::size_t root_capacity = 1U;
  if (ready && root_count == 1U) {
    ready = test.cuda_ok(cudaGraphGetRootNodes(graph, &root, &root_capacity),
                         label + " fetch root");
    test.expect(root_capacity == 1U, label + " fetches one root");
  } else {
    ready = false;
  }

  topology.non_root_functions.clear();
  if (ready) {
    for (const cudaGraphNode_t node : nodes) {
      cudaGraphNodeType node_type = cudaGraphNodeTypeEmpty;
      ready = test.cuda_ok(cudaGraphNodeGetType(node, &node_type),
                           label + " read node type") &&
              ready;
      test.expect(node_type == cudaGraphNodeTypeKernel,
                  label + " contains only kernel nodes");
      cudaKernelNodeParams parameters{};
      if (node_type == cudaGraphNodeTypeKernel) {
        ready = test.cuda_ok(cudaGraphKernelNodeGetParams(node, &parameters),
                             label + " read kernel parameters") &&
                ready;
        if (node == root) {
          topology.root.function = parameters.func;
          topology.root.grid = parameters.gridDim;
          topology.root.block = parameters.blockDim;
          topology.root.dynamic_shared_bytes = parameters.sharedMemBytes;
        } else {
          topology.non_root_functions.push_back(parameters.func);
        }
      }
    }
  }
  test.expect(topology.root.function != nullptr,
              label + " root has a kernel identity");
  test.expect(topology.non_root_functions.size() == 2U,
              label + " has two non-root kernels");
  ready = test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph") &&
          ready;
  return ready;
}

void test_attention_scores_warp_positions_exact(TestContext& test,
                                                cudaStream_t stream) {
  const int failures_before = test.failures();
  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kMaximumSequence = 544U;
  constexpr float kScale = 0.0625F;
  constexpr std::size_t kGuardElements = 17U;
  constexpr std::size_t kQueryElements = kQueryHeads * kDimension;
  constexpr std::size_t kKeyElements =
      kMaximumSequence * kKvHeads * kDimension;
  constexpr std::size_t kMaximumScoreElements =
      kQueryHeads * kMaximumSequence;
  constexpr std::array<std::size_t, 29U> kSequenceLengths{
      1U,   2U,   3U,   4U,   5U,   6U,   7U,   8U,   9U,   63U,
      64U,  65U,  66U,  67U,  68U,  69U,  70U,  71U,  72U,  127U,
      128U, 129U, 255U, 256U, 257U, 511U, 512U, 513U, 544U};
  const float canary = float_from_bits(0x4a55aa55U);

  AttentionScoreKernelResources baseline_resources;
  AttentionScoreKernelResources candidate_resources;
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_scores_baseline_24_4_256_test_cuda_resources(
              nullptr, &baseline_resources.static_shared_bytes,
              &baseline_resources.local_bytes,
              &baseline_resources.maximum_threads,
              &baseline_resources.active_blocks_per_multiprocessor)) ==
          cudaErrorInvalidValue,
      "attention score baseline resource query rejects null output");
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_scores_warp_positions_24_4_256_test_cuda_resources(
              &candidate_resources.registers,
              &candidate_resources.static_shared_bytes,
              &candidate_resources.local_bytes, nullptr,
              &candidate_resources.active_blocks_per_multiprocessor)) ==
          cudaErrorInvalidValue,
      "attention score candidate resource query rejects null output");
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_scores_baseline_24_4_256_test_cuda_resources(
              &baseline_resources.registers,
              &baseline_resources.static_shared_bytes,
              &baseline_resources.local_bytes,
              &baseline_resources.maximum_threads,
              &baseline_resources.active_blocks_per_multiprocessor)),
      "attention score baseline resource query");
  ready = ready && test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_scores_warp_positions_24_4_256_test_cuda_resources(
              &candidate_resources.registers,
              &candidate_resources.static_shared_bytes,
              &candidate_resources.local_bytes,
              &candidate_resources.maximum_threads,
              &candidate_resources.active_blocks_per_multiprocessor)),
      "attention score warp-position resource query");
  if (!ready) {
    return;
  }
  std::cout << "ATTENTION_SCORE_RESOURCES: baseline_regs="
            << baseline_resources.registers
            << " baseline_static_shared="
            << baseline_resources.static_shared_bytes
            << " baseline_local=" << baseline_resources.local_bytes
            << " baseline_active_blocks="
            << baseline_resources.active_blocks_per_multiprocessor
            << " candidate_regs=" << candidate_resources.registers
            << " candidate_static_shared="
            << candidate_resources.static_shared_bytes
            << " candidate_local=" << candidate_resources.local_bytes
            << " candidate_active_blocks="
            << candidate_resources.active_blocks_per_multiprocessor << '\n';
  test.expect(baseline_resources.static_shared_bytes ==
                  256U * sizeof(float),
              "attention score baseline retains the exact shared tree");
  test.expect(baseline_resources.local_bytes == 0U,
              "attention score baseline has no local-memory spills");
  test.expect(candidate_resources.static_shared_bytes == 0U,
              "attention score candidate uses no static shared memory");
  test.expect(candidate_resources.local_bytes == 0U,
              "attention score candidate has no local-memory spills");
  test.expect(candidate_resources.registers <= 64,
              "attention score candidate stays below the hard register cap");
  test.expect(candidate_resources.registers <= 40,
              "attention score candidate meets the target register cap");
  test.expect(candidate_resources.maximum_threads >= 256,
              "attention score candidate supports a 256-thread block");
  test.expect(candidate_resources.active_blocks_per_multiprocessor >= 4,
              "attention score candidate retains hard occupancy floor");
  test.expect(candidate_resources.active_blocks_per_multiprocessor >= 6,
              "attention score candidate meets the target occupancy floor");

  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<float> baseline_storage;
  ManagedBuffer<float> candidate_storage;
  ManagedBuffer<float> replay_storage;
  ready = test.cuda_ok(query.allocate(kQueryElements),
                       "attention score exact allocate query");
  ready = ready && test.cuda_ok(key.allocate(kKeyElements),
                                "attention score exact allocate key");
  ready = ready && test.cuda_ok(
      baseline_storage.allocate(kMaximumScoreElements +
                                2U * kGuardElements),
      "attention score exact allocate baseline");
  ready = ready && test.cuda_ok(
      candidate_storage.allocate(kMaximumScoreElements +
                                 2U * kGuardElements),
      "attention score exact allocate candidate");
  ready = ready && test.cuda_ok(
      replay_storage.allocate(kMaximumScoreElements + 2U * kGuardElements),
      "attention score exact allocate replay");
  if (!ready) {
    return;
  }

  const auto finite_bf16_pattern = [](const std::size_t index,
                                      const std::uint32_t salt) {
    std::uint32_t mixed = static_cast<std::uint32_t>(index) * 747796405U +
                          salt;
    mixed ^= mixed >> 16U;
    mixed *= 2246822519U;
    mixed ^= mixed >> 13U;
    const std::uint16_t sign =
        static_cast<std::uint16_t>((mixed >> 31U) << 15U);
    const std::uint16_t exponent = static_cast<std::uint16_t>(
        (115U + ((mixed >> 8U) % 25U)) << 7U);
    const std::uint16_t mantissa =
        static_cast<std::uint16_t>(mixed & 0x007fU);
    return static_cast<std::uint16_t>(sign | exponent | mantissa);
  };
  for (std::size_t index = 0U; index < query.size(); ++index) {
    query[index] = finite_bf16_pattern(index, 0x9e3779b9U);
  }
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = finite_bf16_pattern(index, 0x85ebca6bU);
  }
  const std::vector<std::uint16_t> query_snapshot(query.data(),
                                                   query.data() + query.size());
  const std::vector<std::uint16_t> key_snapshot(key.data(),
                                                 key.data() + key.size());
  float* const baseline = baseline_storage.data() + kGuardElements;
  float* const candidate = candidate_storage.data() + kGuardElements;
  float* const replay = replay_storage.data() + kGuardElements;

  const auto reference_tree_score = [&](const std::size_t query_head,
                                        const std::size_t position) {
    std::array<float, kDimension> partial{};
    const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
    const std::size_t query_offset = query_head * kDimension;
    const std::size_t key_offset =
        (position * kKvHeads + kv_head) * kDimension;
    for (std::size_t dimension = 0U; dimension < kDimension; ++dimension) {
      partial[dimension] =
          std::fma(decode_bf16(query[query_offset + dimension]),
                   decode_bf16(key[key_offset + dimension]), 0.0F);
    }
    for (std::size_t stride = kDimension / 2U; stride != 0U; stride >>= 1U) {
      for (std::size_t dimension = 0U; dimension < stride; ++dimension) {
        partial[dimension] =
            partial[dimension] + partial[dimension + stride];
      }
    }
    return partial[0] * kScale;
  };
  const auto linear_fma_score = [&](const std::size_t query_head,
                                    const std::size_t position) {
    const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
    const std::size_t query_offset = query_head * kDimension;
    const std::size_t key_offset =
        (position * kKvHeads + kv_head) * kDimension;
    float score = 0.0F;
    for (std::size_t dimension = 0U; dimension < kDimension; ++dimension) {
      score = std::fma(decode_bf16(query[query_offset + dimension]),
                       decode_bf16(key[key_offset + dimension]), score);
    }
    return score * kScale;
  };
  bool reduction_order_sensitive = false;
  for (std::size_t query_head = 0U;
       query_head < kQueryHeads && !reduction_order_sensitive;
       ++query_head) {
    for (std::size_t position = 0U;
         position < 8U && !reduction_order_sensitive; ++position) {
      const float tree = reference_tree_score(query_head, position);
      const float linear = linear_fma_score(query_head, position);
      reduction_order_sensitive =
          std::memcmp(&tree, &linear, sizeof(float)) != 0;
    }
  }
  test.expect(reduction_order_sensitive,
              "attention score finite fixture detects reduction reordering");

  const auto guards_intact = [&](const ManagedBuffer<float>& storage,
                                 const std::size_t active_elements) {
    const float* const payload = storage.data() + kGuardElements;
    return std::all_of(storage.data(), payload,
                       [&](const float value) { return value == canary; }) &&
           std::all_of(payload + active_elements,
                       storage.data() + storage.size(),
                       [&](const float value) { return value == canary; });
  };

  for (const std::size_t sequence_length : kSequenceLengths) {
    const std::string label =
        "attention score exact S=" + std::to_string(sequence_length);
    const std::size_t active_elements = kQueryHeads * sequence_length;
    std::fill_n(baseline_storage.data(), baseline_storage.size(), canary);
    std::fill_n(candidate_storage.data(), candidate_storage.size(), canary);
    std::fill_n(replay_storage.data(), replay_storage.size(), canary);

    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_scores_baseline_24_4_256_test_cuda(
                query.data(), key.data(), kQueryHeads, kKvHeads,
                sequence_length, kDimension, kScale, baseline,
                static_cast<void*>(stream))),
        label + " baseline launch");
    ready = ready && test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_scores_warp_positions_24_4_256_test_cuda(
                query.data(), key.data(), kQueryHeads, kKvHeads,
                sequence_length, kDimension, kScale, candidate,
                static_cast<void*>(stream))),
        label + " candidate launch");
    ready = ready && test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_scores_warp_positions_24_4_256_test_cuda(
                query.data(), key.data(), kQueryHeads, kKvHeads,
                sequence_length, kDimension, kScale, replay,
                static_cast<void*>(stream))),
        label + " replay launch");
    ready = ready &&
            test.cuda_ok(cudaStreamSynchronize(stream), label + " sync");
    if (!ready) {
      return;
    }
    test.expect(std::memcmp(candidate, baseline,
                            active_elements * sizeof(float)) == 0,
                label + " candidate is raw-FP32 bitwise exact");
    test.expect(std::memcmp(replay, candidate,
                            active_elements * sizeof(float)) == 0,
                label + " candidate replay is deterministic");
    const std::size_t oracle_query_head = sequence_length % kQueryHeads;
    const std::size_t oracle_position = sequence_length - 1U;
    const float oracle =
        reference_tree_score(oracle_query_head, oracle_position);
    test.expect(
        std::memcmp(&baseline[oracle_query_head * sequence_length +
                              oracle_position],
                    &oracle, sizeof(float)) == 0,
        label + " baseline matches the independent shared-tree oracle");
    test.expect(guards_intact(baseline_storage, active_elements),
                label + " baseline canaries are intact");
    test.expect(guards_intact(candidate_storage, active_elements),
                label + " candidate canaries are intact");
    test.expect(guards_intact(replay_storage, active_elements),
                label + " replay canaries are intact");
    test.expect(std::memcmp(query.data(), query_snapshot.data(),
                            query.size() * sizeof(std::uint16_t)) == 0,
                label + " query is unchanged");
    test.expect(std::memcmp(key.data(), key_snapshot.data(),
                            key.size() * sizeof(std::uint16_t)) == 0,
                label + " key cache is unchanged");
  }

  constexpr std::size_t kSpecialSequence = 8U;
  constexpr std::array<std::uint16_t, 12U> kSpecialBf16{
      0x0000U,  // +0
      0x8000U,  // -0
      0x0001U,  // smallest positive subnormal
      0x8001U,  // smallest negative subnormal
      0x7f7fU,  // maximum positive finite
      0xff7fU,  // maximum negative finite
      0x7f80U,  // +infinity
      0xff80U,  // -infinity
      0x7fc1U,  // positive quiet NaN
      0xffc1U,  // negative quiet NaN
      0x7f81U,  // positive signaling NaN
      0xff81U   // negative signaling NaN
  };
  constexpr std::array<std::size_t, 8U> kTreeSlots{
      0U, 32U, 64U, 96U, 128U, 160U, 192U, 224U};
  constexpr std::array<std::size_t, 4U> kShuffleBoundaryLanes{
      0U, 15U, 16U, 31U};
  constexpr std::size_t kSpecialKeyElements =
      kSpecialSequence * kKvHeads * kDimension;
  const std::size_t special_active_elements =
      kQueryHeads * kSpecialSequence;
  std::size_t isolated_special_cases = 0U;
  for (std::size_t raw_index = 0U; raw_index < kSpecialBf16.size();
       ++raw_index) {
    for (std::size_t slot_index = 0U; slot_index < kTreeSlots.size();
         ++slot_index) {
      for (std::size_t lane_index = 0U;
           lane_index < kShuffleBoundaryLanes.size(); ++lane_index) {
        for (std::size_t operand = 0U; operand < 2U; ++operand) {
          const std::string label =
              "attention score isolated special raw=" +
              std::to_string(raw_index) + " slot=" +
              std::to_string(slot_index) + " lane=" +
              std::to_string(lane_index) +
              (operand == 0U ? " query" : " key");
          std::fill_n(query.data(), query.size(),
                      static_cast<std::uint16_t>(0x0000U));
          std::fill_n(key.data(), kSpecialKeyElements,
                      static_cast<std::uint16_t>(0x0000U));
          const std::size_t query_head = isolated_special_cases % kQueryHeads;
          const std::size_t kv_head =
              query_head / (kQueryHeads / kKvHeads);
          const std::size_t position =
              (isolated_special_cases / kQueryHeads) % kSpecialSequence;
          const std::size_t dimension =
              kTreeSlots[slot_index] + kShuffleBoundaryLanes[lane_index];
          const std::size_t query_index =
              query_head * kDimension + dimension;
          if (operand == 0U) {
            query[query_index] = kSpecialBf16[raw_index];
            for (std::size_t key_position = 0U;
                 key_position < kSpecialSequence; ++key_position) {
              key[(key_position * kKvHeads + kv_head) * kDimension +
                  dimension] = encode_bf16(1.0F);
            }
          } else {
            query[query_index] = encode_bf16(1.0F);
            key[(position * kKvHeads + kv_head) * kDimension + dimension] =
                kSpecialBf16[raw_index];
          }
          const std::vector<std::uint16_t> special_query_snapshot(
              query.data(), query.data() + query.size());
          const std::vector<std::uint16_t> special_key_snapshot(
              key.data(), key.data() + kSpecialKeyElements);
          std::fill_n(baseline_storage.data(), baseline_storage.size(),
                      canary);
          std::fill_n(candidate_storage.data(), candidate_storage.size(),
                      canary);
          std::fill_n(replay_storage.data(), replay_storage.size(), canary);

          ready = test.cuda_ok(
              static_cast<cudaError_t>(q3x::runtime::
                  launch_attention_scores_baseline_24_4_256_test_cuda(
                      query.data(), key.data(), kQueryHeads, kKvHeads,
                      kSpecialSequence, kDimension, kScale, baseline,
                      static_cast<void*>(stream))),
              label + " baseline launch");
          ready = ready && test.cuda_ok(
              static_cast<cudaError_t>(q3x::runtime::
                  launch_attention_scores_warp_positions_24_4_256_test_cuda(
                      query.data(), key.data(), kQueryHeads, kKvHeads,
                      kSpecialSequence, kDimension, kScale, candidate,
                      static_cast<void*>(stream))),
              label + " candidate launch");
          ready = ready && test.cuda_ok(
              static_cast<cudaError_t>(q3x::runtime::
                  launch_attention_scores_warp_positions_24_4_256_test_cuda(
                      query.data(), key.data(), kQueryHeads, kKvHeads,
                      kSpecialSequence, kDimension, kScale, replay,
                      static_cast<void*>(stream))),
              label + " replay launch");
          ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                        label + " sync");
          if (!ready) {
            return;
          }
          test.expect(std::memcmp(candidate, baseline,
                                  special_active_elements * sizeof(float)) ==
                          0,
                      label + " is raw-FP32 bitwise exact");
          test.expect(std::memcmp(replay, candidate,
                                  special_active_elements * sizeof(float)) ==
                          0,
                      label + " replay is deterministic");
          const float classified =
              baseline[query_head * kSpecialSequence + position];
          const std::uint16_t raw = kSpecialBf16[raw_index];
          if ((raw & 0x7f80U) == 0x7f80U) {
            if ((raw & 0x007fU) == 0U) {
              test.expect(std::isinf(classified),
                          label + " keeps isolated infinity observable");
            } else {
              test.expect(std::isnan(classified),
                          label + " keeps isolated NaN observable");
            }
          } else {
            test.expect(std::isfinite(classified),
                        label + " is not masked by a nonfinite class");
          }
          test.expect(guards_intact(baseline_storage,
                                    special_active_elements),
                      label + " baseline canaries are intact");
          test.expect(guards_intact(candidate_storage,
                                    special_active_elements),
                      label + " candidate canaries are intact");
          test.expect(guards_intact(replay_storage, special_active_elements),
                      label + " replay canaries are intact");
          test.expect(std::memcmp(query.data(), special_query_snapshot.data(),
                                  query.size() * sizeof(std::uint16_t)) == 0,
                      label + " query is unchanged");
          test.expect(std::memcmp(key.data(), special_key_snapshot.data(),
                                  kSpecialKeyElements *
                                      sizeof(std::uint16_t)) == 0,
                      label + " key cache is unchanged");
          ++isolated_special_cases;
        }
      }
    }
  }
  std::cout << "ATTENTION_SCORE_BINARY_IDENTITY: finite_boundary_lengths="
            << kSequenceLengths.size()
            << " special_raw_bf16=" << kSpecialBf16.size()
            << " tree_slots=" << kTreeSlots.size()
            << " shuffle_boundary_lanes=" << kShuffleBoundaryLanes.size()
            << " isolated_operand_cases=" << isolated_special_cases
            << " replay=enabled status="
            << (test.failures() == failures_before ? "PASS" : "FAIL")
            << '\n';
}

void test_attention_scores_warp_positions_graph_contract(
    TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kSequence = 513U;
  constexpr std::size_t kQueryElements = 24U * 256U;
  constexpr std::size_t kKeyElements = kSequence * 4U * 256U;
  constexpr std::size_t kScoreElements = 24U * kSequence;
  test.expect(
      !q3x::runtime::use_attention_scores_warp_positions_24_4_256_test(
          24U, 4U, 64U, 256U),
      "attention score production selector keeps S64 on the reference path");
  test.expect(
      q3x::runtime::use_attention_scores_warp_positions_24_4_256_test(
          24U, 4U, 65U, 256U),
      "attention score production selector enables S65");
  test.expect(
      q3x::runtime::use_attention_scores_warp_positions_24_4_256_test(
          24U, 4U, 65535U * 8U, 256U),
      "attention score production selector accepts the maximum grid");
  test.expect(
      !q3x::runtime::use_attention_scores_warp_positions_24_4_256_test(
          24U, 4U, 65535U * 8U + 1U, 256U),
      "attention score production selector falls back above the maximum grid");
  test.expect(
      !q3x::runtime::use_attention_scores_warp_positions_24_4_256_test(
          23U, 4U, 65U, 256U) &&
          !q3x::runtime::use_attention_scores_warp_positions_24_4_256_test(
              24U, 5U, 65U, 256U) &&
          !q3x::runtime::use_attention_scores_warp_positions_24_4_256_test(
              24U, 4U, 65U, 255U),
      "attention score production selector rejects near-miss shapes");
  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<float> scores;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(query.allocate(kQueryElements),
                            "attention score graph allocate query");
  ready = ready && test.cuda_ok(key.allocate(kKeyElements),
                                "attention score graph allocate key");
  ready = ready && test.cuda_ok(value.allocate(kKeyElements),
                                "attention score graph allocate value");
  ready = ready && test.cuda_ok(scores.allocate(kScoreElements),
                                "attention score graph allocate scores");
  ready = ready && test.cuda_ok(output.allocate(kQueryElements),
                                "attention score graph allocate output");
  if (!ready) {
    return;
  }

  constexpr std::array<std::pair<const char*, AttentionScoreTestLaunch>, 2U>
      kLaunchers{{
          {"baseline", q3x::runtime::
                           launch_attention_scores_baseline_24_4_256_test_cuda},
          {"candidate", q3x::runtime::
                            launch_attention_scores_warp_positions_24_4_256_test_cuda},
      }};
  for (const auto& [name, launch] : kLaunchers) {
    const std::string prefix =
        std::string("attention score ") + name + " invalid graph ";
    expect_invalid_attention_score_graph(
        test, stream, prefix + "Q23", launch, query.data(), key.data(), 23U,
        4U, kSequence, 256U, 0.0625F, scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "KV5", launch, query.data(), key.data(), 24U,
        5U, kSequence, 256U, 0.0625F, scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "S0", launch, query.data(), key.data(), 24U,
        4U, 0U, 256U, 0.0625F, scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "S above grid", launch, query.data(),
        key.data(), 24U, 4U, 65535U * 8U + 1U, 256U, 0.0625F,
        scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "D255", launch, query.data(), key.data(), 24U,
        4U, kSequence, 255U, 0.0625F, scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "negative scale", launch, query.data(),
        key.data(), 24U, 4U, kSequence, 256U, -1.0F, scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "infinite scale", launch, query.data(),
        key.data(), 24U, 4U, kSequence, 256U,
        std::numeric_limits<float>::infinity(), scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "NaN scale", launch, query.data(), key.data(),
        24U, 4U, kSequence, 256U,
        std::numeric_limits<float>::quiet_NaN(), scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "null query", launch, nullptr, key.data(), 24U,
        4U, kSequence, 256U, 0.0625F, scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "null key", launch, query.data(), nullptr, 24U,
        4U, kSequence, 256U, 0.0625F, scores.data());
    expect_invalid_attention_score_graph(
        test, stream, prefix + "null scores", launch, query.data(), key.data(),
        24U, 4U, kSequence, 256U, 0.0625F, nullptr);
    expect_invalid_attention_score_graph(
        test, stream, prefix + "query-output alias", launch, query.data(),
        key.data(), 24U, 4U, kSequence, 256U, 0.0625F,
        reinterpret_cast<float*>(query.data()));
    expect_invalid_attention_score_graph(
        test, stream, prefix + "key-output alias", launch, query.data(),
        key.data(), 24U, 4U, kSequence, 256U, 0.0625F,
        reinterpret_cast<float*>(key.data()));
  }

  AttentionScoreGraphTopology baseline_topology;
  AttentionScoreGraphTopology candidate_topology;
  ready = capture_attention_score_graph_topology(
      test, stream, "attention score baseline positive graph",
      q3x::runtime::launch_attention_scores_baseline_24_4_256_test_cuda,
      query.data(), key.data(), kSequence, scores.data(), baseline_topology);
  ready = capture_attention_score_graph_topology(
              test, stream, "attention score candidate positive graph",
              q3x::runtime::
                  launch_attention_scores_warp_positions_24_4_256_test_cuda,
              query.data(), key.data(), kSequence, scores.data(),
              candidate_topology) &&
          ready;
  if (!ready) {
    return;
  }
  test.expect(baseline_topology.function != nullptr,
              "attention score baseline graph has a kernel identity");
  test.expect(candidate_topology.function != nullptr,
              "attention score candidate graph has a kernel identity");
  test.expect(baseline_topology.function != candidate_topology.function,
              "attention score graph kernel identities are distinct");
  test.expect(baseline_topology.grid.x == 24U &&
                  baseline_topology.grid.y == 1U &&
                  baseline_topology.grid.z == 1U,
              "attention score baseline graph grid is (24,1,1)");
  test.expect(candidate_topology.grid.x == 24U &&
                  candidate_topology.grid.y == 65U &&
                  candidate_topology.grid.z == 1U,
              "attention score candidate graph grid is (24,65,1)");
  test.expect(baseline_topology.block.x == 256U &&
                  baseline_topology.block.y == 1U &&
                  baseline_topology.block.z == 1U,
              "attention score baseline graph block is (256,1,1)");
  test.expect(candidate_topology.block.x == 256U &&
                  candidate_topology.block.y == 1U &&
                  candidate_topology.block.z == 1U,
              "attention score candidate graph block is (256,1,1)");
  test.expect(baseline_topology.dynamic_shared_bytes == 0U &&
                  candidate_topology.dynamic_shared_bytes == 0U,
              "attention score graphs request zero dynamic shared memory");
  std::cout << "ATTENTION_SCORE_GRAPH_TOPOLOGY: baseline_func="
            << baseline_topology.function << " baseline_grid=("
            << baseline_topology.grid.x << ',' << baseline_topology.grid.y
            << ',' << baseline_topology.grid.z << ") candidate_func="
            << candidate_topology.function << " candidate_grid=("
            << candidate_topology.grid.x << ',' << candidate_topology.grid.y
            << ',' << candidate_topology.grid.z << ") block=("
            << candidate_topology.block.x << ',' << candidate_topology.block.y
            << ',' << candidate_topology.block.z << ") dynamic_shared="
            << candidate_topology.dynamic_shared_bytes << '\n';

  GqaGraphTopology production_s64;
  GqaGraphTopology production_s65;
  ready = capture_gqa_graph_topology(
      test, stream, "attention score production S64 graph", query.data(),
      key.data(), value.data(), 64U, scores.data(), scores.size(),
      output.data(), production_s64);
  ready = capture_gqa_graph_topology(
              test, stream, "attention score production S65 graph",
              query.data(), key.data(), value.data(), 65U, scores.data(),
              scores.size(), output.data(), production_s65) &&
          ready;
  if (!ready) {
    return;
  }
  test.expect(production_s64.root.function == baseline_topology.function,
              "attention score production S64 dispatches the reference score");
  test.expect(production_s65.root.function == candidate_topology.function,
              "attention score production S65 dispatches the warp-position score");
  test.expect(production_s64.root.function != candidate_topology.function &&
                  production_s65.root.function != baseline_topology.function,
              "attention score production threshold selects distinct kernels");
  test.expect(production_s64.root.grid.x == 24U &&
                  production_s64.root.grid.y == 1U &&
                  production_s64.root.grid.z == 1U,
              "attention score production S64 root grid is (24,1,1)");
  test.expect(production_s65.root.grid.x == 24U &&
                  production_s65.root.grid.y == 9U &&
                  production_s65.root.grid.z == 1U,
              "attention score production S65 root grid is (24,9,1)");
  test.expect(production_s64.root.block.x == 256U &&
                  production_s64.root.block.y == 1U &&
                  production_s64.root.block.z == 1U &&
                  production_s65.root.block.x == 256U &&
                  production_s65.root.block.y == 1U &&
                  production_s65.root.block.z == 1U,
              "attention score production roots retain 256-thread blocks");
  test.expect(production_s64.root.dynamic_shared_bytes == 0U &&
                  production_s65.root.dynamic_shared_bytes == 0U,
              "attention score production roots use no dynamic shared memory");
  test.expect(std::is_permutation(
                  production_s64.non_root_functions.begin(),
                  production_s64.non_root_functions.end(),
                  production_s65.non_root_functions.begin(),
                  production_s65.non_root_functions.end()),
              "attention score production keeps softmax and value kernels");
  std::cout << "ATTENTION_SCORE_PRODUCTION_DISPATCH: S64_func="
            << production_s64.root.function << " S64_grid=("
            << production_s64.root.grid.x << ','
            << production_s64.root.grid.y << ','
            << production_s64.root.grid.z << ") S65_func="
            << production_s65.root.function << " S65_grid=("
            << production_s65.root.grid.x << ','
            << production_s65.root.grid.y << ','
            << production_s65.root.grid.z
            << ") downstream_kernel_identities=unchanged\n";
}

void test_attention_scores_warp_positions_perf(TestContext& test,
                                               cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_ATTENTION_SCORES_WARP_POSITIONS_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: attention score warp-position performance gate; set "
                 "Q3X_RUN_ATTENTION_SCORES_WARP_POSITIONS_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kMaximumSequence = 544U;
  constexpr float kScale = 0.0625F;
  constexpr std::size_t kWarmupIterations = 10U;
  constexpr std::size_t kMeasuredIterations = 80U;
  constexpr int kMeasurementRounds = 5;
  constexpr std::array<std::size_t, 5U> kSequenceLengths{
      65U, 128U, 257U, 513U, 544U};
  constexpr std::size_t kQueryElements = kQueryHeads * kDimension;
  constexpr std::size_t kKeyElements =
      kMaximumSequence * kKvHeads * kDimension;
  constexpr std::size_t kScoreElements =
      kQueryHeads * kMaximumSequence;

  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<float> baseline;
  ManagedBuffer<float> candidate;
  bool ready = test.cuda_ok(query.allocate(kQueryElements),
                            "attention score perf allocate query");
  ready = ready && test.cuda_ok(key.allocate(kKeyElements),
                                "attention score perf allocate key");
  ready = ready && test.cuda_ok(baseline.allocate(kScoreElements),
                                "attention score perf allocate baseline");
  ready = ready && test.cuda_ok(candidate.allocate(kScoreElements),
                                "attention score perf allocate candidate");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0U; index < query.size(); ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 17U + 5U) % 127U) -
                           63) /
        64.0F);
  }
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 7U) % 113U) -
                           56) /
        64.0F);
  }

  for (const std::size_t sequence_length : kSequenceLengths) {
    const std::string label =
        "attention score perf S=" + std::to_string(sequence_length);
    const auto launch_baseline = [&]() {
      return q3x::runtime::
          launch_attention_scores_baseline_24_4_256_test_cuda(
              query.data(), key.data(), kQueryHeads, kKvHeads,
              sequence_length, kDimension, kScale, baseline.data(),
              static_cast<void*>(stream));
    };
    const auto launch_candidate = [&]() {
      return q3x::runtime::
          launch_attention_scores_warp_positions_24_4_256_test_cuda(
              query.data(), key.data(), kQueryHeads, kKvHeads,
              sequence_length, kDimension, kScale, candidate.data(),
              static_cast<void*>(stream));
    };
    for (std::size_t iteration = 0U;
         ready && iteration < kWarmupIterations; ++iteration) {
      ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                           label + " baseline warmup");
      ready = ready &&
              test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                           label + " candidate warmup");
    }
    ready = ready &&
            test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync");
    if (!ready) {
      return;
    }

    double baseline_total = 0.0;
    double candidate_total = 0.0;
    bool timing_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          label + " round=" + std::to_string(round + 1);
      const float baseline_first = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " baseline 1",
          launch_baseline);
      const float candidate_first = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " candidate 1",
          launch_candidate);
      const float candidate_second = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " candidate 2",
          launch_candidate);
      const float baseline_second = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " baseline 2",
          launch_baseline);
      const bool round_finite =
          std::isfinite(baseline_first) && baseline_first > 0.0F &&
          std::isfinite(candidate_first) && candidate_first > 0.0F &&
          std::isfinite(candidate_second) && candidate_second > 0.0F &&
          std::isfinite(baseline_second) && baseline_second > 0.0F;
      timing_finite = timing_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_ATTENTION_SCORES_WARP_POSITIONS_ROUND: S="
                << sequence_length << " round=" << round + 1
                << " iterations=" << kMeasuredIterations
                << " baseline1_ms=" << baseline_first
                << " candidate1_ms=" << candidate_first
                << " candidate2_ms=" << candidate_second
                << " baseline2_ms=" << baseline_second << '\n';
    }
    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    const double baseline_milliseconds = baseline_total / kTimedPasses;
    const double candidate_milliseconds = candidate_total / kTimedPasses;
    const double speedup = baseline_milliseconds / candidate_milliseconds;
    const double required_speedup = sequence_length == 513U ? 1.20 : 1.05;
    const bool gate_passed =
        timing_finite && std::isfinite(speedup) &&
        baseline_milliseconds > 0.0 && candidate_milliseconds > 0.0 &&
        speedup >= required_speedup;
    std::cout << "PERF_ATTENTION_SCORES_WARP_POSITIONS: S="
              << sequence_length << " baseline_ms=" << baseline_milliseconds
              << " candidate_ms=" << candidate_milliseconds
              << " speedup=" << speedup
              << " required_speedup=" << required_speedup
              << " gate=" << (gate_passed ? "PASS" : "FAIL") << '\n';
    test.expect(gate_passed, label + " clears its span gate");
    test.expect(std::memcmp(candidate.data(), baseline.data(),
                            kQueryHeads * sequence_length * sizeof(float)) ==
                    0,
                label + " repeated output stays bitwise exact");
  }

  constexpr std::size_t kChainFirstSequence = 65U;
  constexpr std::size_t kChainLastSequence = 513U;
  constexpr std::size_t kChainWarmupIterations = 10U;
  constexpr std::size_t kChainMeasuredIterations = 1U;
  constexpr double kChainRequiredSpeedup = 1.20;
  const std::string chain_label = "attention score P513 chain S65..513";
  const auto launch_baseline_chain = [&]() {
    for (std::size_t sequence_length = kChainFirstSequence;
         sequence_length <= kChainLastSequence; ++sequence_length) {
      const int status = q3x::runtime::
          launch_attention_scores_baseline_24_4_256_test_cuda(
              query.data(), key.data(), kQueryHeads, kKvHeads,
              sequence_length, kDimension, kScale, baseline.data(),
              static_cast<void*>(stream));
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  const auto launch_candidate_chain = [&]() {
    for (std::size_t sequence_length = kChainFirstSequence;
         sequence_length <= kChainLastSequence; ++sequence_length) {
      const int status = q3x::runtime::
          launch_attention_scores_warp_positions_24_4_256_test_cuda(
              query.data(), key.data(), kQueryHeads, kKvHeads,
              sequence_length, kDimension, kScale, candidate.data(),
              static_cast<void*>(stream));
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  for (std::size_t iteration = 0U;
       ready && iteration < kChainWarmupIterations; ++iteration) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(launch_baseline_chain()),
        chain_label + " baseline warmup");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate_chain()),
                         chain_label + " candidate warmup");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                chain_label + " warmup sync");
  if (!ready) {
    return;
  }

  double chain_baseline_total = 0.0;
  double chain_candidate_total = 0.0;
  bool chain_timing_finite = true;
  for (int round = 0; round < kMeasurementRounds; ++round) {
    const std::string round_label =
        chain_label + " round=" + std::to_string(round + 1);
    const float baseline_first = measure_cuda_span_milliseconds(
        test, stream, kChainMeasuredIterations,
        round_label + " baseline 1", launch_baseline_chain);
    const float candidate_first = measure_cuda_span_milliseconds(
        test, stream, kChainMeasuredIterations,
        round_label + " candidate 1", launch_candidate_chain);
    const float candidate_second = measure_cuda_span_milliseconds(
        test, stream, kChainMeasuredIterations,
        round_label + " candidate 2", launch_candidate_chain);
    const float baseline_second = measure_cuda_span_milliseconds(
        test, stream, kChainMeasuredIterations,
        round_label + " baseline 2", launch_baseline_chain);
    const bool round_finite =
        std::isfinite(baseline_first) && baseline_first > 0.0F &&
        std::isfinite(candidate_first) && candidate_first > 0.0F &&
        std::isfinite(candidate_second) && candidate_second > 0.0F &&
        std::isfinite(baseline_second) && baseline_second > 0.0F;
    chain_timing_finite = chain_timing_finite && round_finite;
    if (round_finite) {
      chain_baseline_total += baseline_first + baseline_second;
      chain_candidate_total += candidate_first + candidate_second;
    }
    std::cout << "PERF_ATTENTION_SCORES_WARP_POSITIONS_CHAIN_ROUND: round="
              << round + 1 << " kernels_per_chain="
              << (kChainLastSequence - kChainFirstSequence + 1U)
              << " baseline1_ms=" << baseline_first
              << " candidate1_ms=" << candidate_first
              << " candidate2_ms=" << candidate_second
              << " baseline2_ms=" << baseline_second << '\n';
  }
  constexpr double kChainTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const double chain_baseline_milliseconds =
      chain_baseline_total / kChainTimedPasses;
  const double chain_candidate_milliseconds =
      chain_candidate_total / kChainTimedPasses;
  const double chain_speedup =
      chain_baseline_milliseconds / chain_candidate_milliseconds;
  const bool chain_gate_passed =
      chain_timing_finite && std::isfinite(chain_speedup) &&
      chain_baseline_milliseconds > 0.0 &&
      chain_candidate_milliseconds > 0.0 &&
      chain_speedup >= kChainRequiredSpeedup;
  std::cout << "PERF_ATTENTION_SCORES_WARP_POSITIONS_CHAIN: first_S="
            << kChainFirstSequence << " last_S=" << kChainLastSequence
            << " kernels_per_chain="
            << (kChainLastSequence - kChainFirstSequence + 1U)
            << " baseline_ms=" << chain_baseline_milliseconds
            << " candidate_ms=" << chain_candidate_milliseconds
            << " speedup=" << chain_speedup
            << " required_speedup=" << kChainRequiredSpeedup
            << " gate=" << (chain_gate_passed ? "PASS" : "FAIL") << '\n';
  test.expect(chain_gate_passed,
              chain_label + " clears the hard 1.20x aggregate gate");
  test.expect(std::memcmp(candidate.data(), baseline.data(),
                          kQueryHeads * kChainLastSequence * sizeof(float)) ==
                  0,
              chain_label + " final S513 output stays bitwise exact");
}

void test_attention_values_exact(TestContext& test, cudaStream_t stream) {
  const int failures_before = test.failures();
  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kMaximumSequence = 544U;
  constexpr std::size_t kValueElements =
      kMaximumSequence * kKvHeads * kDimension;
  constexpr std::size_t kProbabilityElements =
      kQueryHeads * kMaximumSequence;
  constexpr std::size_t kOutputElements = kQueryHeads * kDimension;
  constexpr std::size_t kMaximumSupportedSequence =
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) /
      (kKvHeads * kDimension);
  constexpr std::array<std::size_t, 29U> kSequenceLengths{
      1U,   2U,   3U,   4U,   5U,   6U,   7U,   8U,   9U,   63U,
      64U,  65U,  66U,  67U,  68U,  69U,  70U,  71U,  72U,  127U,
      128U, 129U, 255U, 256U, 257U, 511U, 512U, 513U, 544U};
  constexpr std::array<std::size_t, 8U> kMappingHeads{
      0U, 5U, 6U, 11U, 12U, 17U, 18U, 23U};
  constexpr std::array<std::size_t, 12U> kBoundaryDimensions{
      0U, 1U, 31U, 32U, 63U, 64U, 127U, 128U, 191U, 192U, 254U, 255U};

  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_values_exact_24_4_256_test_cuda_selection(
              kQueryHeads, kKvHeads, 1U, kDimension, nullptr)) ==
          cudaErrorInvalidValue,
      "attention value selector query rejects null output");
  const auto expect_selection = [&](const std::size_t query_heads,
                                    const std::size_t kv_heads,
                                    const std::size_t sequence_length,
                                    const std::size_t dimension,
                                    const bool expected,
                                    const std::string& label) {
    int selected = -1;
    const cudaError_t status = static_cast<cudaError_t>(
        q3x::runtime::
            query_attention_values_exact_24_4_256_test_cuda_selection(
                query_heads, kv_heads, sequence_length, dimension,
                &selected));
    test.expect(status == cudaSuccess, label + " selector query succeeds");
    test.expect(selected == (expected ? 1 : 0),
                label + " selector result matches");
  };
  expect_selection(kQueryHeads, kKvHeads, 0U, kDimension, false,
                   "attention value S0");
  expect_selection(kQueryHeads, kKvHeads, 1U, kDimension, true,
                   "attention value S1");
  expect_selection(kQueryHeads, kKvHeads, kMaximumSupportedSequence,
                   kDimension, true, "attention value maximum S");
  expect_selection(kQueryHeads, kKvHeads, kMaximumSupportedSequence + 1U,
                   kDimension, false, "attention value maximum S plus one");
  expect_selection(23U, kKvHeads, 513U, kDimension, false,
                   "attention value Q23");
  expect_selection(kQueryHeads, 5U, 513U, kDimension, false,
                   "attention value KV5");
  expect_selection(kQueryHeads, kKvHeads, 513U, 255U, false,
                   "attention value D255");

  AttentionScoreKernelResources baseline_resources;
  AttentionScoreKernelResources candidate_resources;
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_values_baseline_24_4_256_test_cuda_resources(
              nullptr, &baseline_resources.static_shared_bytes,
              &baseline_resources.local_bytes,
              &baseline_resources.maximum_threads,
              &baseline_resources.active_blocks_per_multiprocessor)) ==
          cudaErrorInvalidValue,
      "attention value baseline resource query rejects null output");
  test.expect(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_values_exact_24_4_256_test_cuda_resources(
              &candidate_resources.registers,
              &candidate_resources.static_shared_bytes,
              &candidate_resources.local_bytes, nullptr,
              &candidate_resources.active_blocks_per_multiprocessor)) ==
          cudaErrorInvalidValue,
      "attention value candidate resource query rejects null output");
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_values_baseline_24_4_256_test_cuda_resources(
              &baseline_resources.registers,
              &baseline_resources.static_shared_bytes,
              &baseline_resources.local_bytes,
              &baseline_resources.maximum_threads,
              &baseline_resources.active_blocks_per_multiprocessor)),
      "attention value baseline resource query");
  ready = ready && test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_attention_values_exact_24_4_256_test_cuda_resources(
              &candidate_resources.registers,
              &candidate_resources.static_shared_bytes,
              &candidate_resources.local_bytes,
              &candidate_resources.maximum_threads,
              &candidate_resources.active_blocks_per_multiprocessor)),
      "attention value exact resource query");
  if (!ready) {
    return;
  }
  std::cout << "ATTENTION_VALUE_RESOURCES: baseline_regs="
            << baseline_resources.registers
            << " baseline_static_shared="
            << baseline_resources.static_shared_bytes
            << " baseline_local=" << baseline_resources.local_bytes
            << " baseline_active_blocks="
            << baseline_resources.active_blocks_per_multiprocessor
            << " candidate_regs=" << candidate_resources.registers
            << " candidate_static_shared="
            << candidate_resources.static_shared_bytes
            << " candidate_local=" << candidate_resources.local_bytes
            << " candidate_active_blocks="
            << candidate_resources.active_blocks_per_multiprocessor << '\n';
  test.expect(baseline_resources.static_shared_bytes == 0U &&
                  baseline_resources.local_bytes == 0U,
              "attention value baseline has no shared memory or spills");
  test.expect(candidate_resources.static_shared_bytes == 0U &&
                  candidate_resources.local_bytes == 0U,
              "attention value candidate has no shared memory or spills");
  test.expect(baseline_resources.maximum_threads >= 256 &&
                  candidate_resources.maximum_threads >= 256,
              "attention value kernels support 256-thread blocks");
  test.expect(candidate_resources.registers <= baseline_resources.registers,
              "attention value candidate does not increase registers");
  test.expect(candidate_resources.active_blocks_per_multiprocessor >=
                  baseline_resources.active_blocks_per_multiprocessor,
              "attention value candidate preserves predecessor occupancy");

  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<float> probabilities;
  GuardedBf16Buffer baseline;
  GuardedBf16Buffer candidate;
  GuardedBf16Buffer replay;
  ready = test.cuda_ok(value.allocate(kValueElements),
                       "attention value exact allocate value cache");
  ready = ready && test.cuda_ok(
      probabilities.allocate(kProbabilityElements),
      "attention value exact allocate probabilities");
  ready = ready && baseline.allocate(test, kOutputElements,
                                     "attention value exact baseline");
  ready = ready && candidate.allocate(test, kOutputElements,
                                      "attention value exact candidate");
  ready = ready && replay.allocate(test, kOutputElements,
                                   "attention value exact replay");
  if (!ready) {
    return;
  }

  for (std::size_t index = 0U; index < value.size(); ++index) {
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 11U) % 67U) -
                           33) /
        64.0F);
  }
  for (std::size_t index = 0U; index < probabilities.size(); ++index) {
    probabilities[index] =
        static_cast<float>(static_cast<int>((index * 37U + 19U) % 61U) -
                           30) /
        256.0F;
  }
  const std::vector<std::uint16_t> value_snapshot(
      value.data(), value.data() + value.size());
  const std::vector<float> probability_snapshot(
      probabilities.data(), probabilities.data() + probabilities.size());

  for (const std::size_t sequence_length : kSequenceLengths) {
    const std::string label =
        "attention value exact S=" + std::to_string(sequence_length);
    baseline.initialize(0x7fc1U, 0x1357U, 0x2468U);
    candidate.initialize(0x7fc2U, 0x369cU, 0x147aU);
    replay.initialize(0x7fc3U, 0x55aaU, 0xaa55U);
    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_values_baseline_24_4_256_test_cuda(
                value.data(), probabilities.data(), kQueryHeads, kKvHeads,
                sequence_length, kDimension, baseline.data(),
                static_cast<void*>(stream))),
        label + " baseline launch");
    ready = ready && test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_values_exact_24_4_256_test_cuda(
                value.data(), probabilities.data(), kQueryHeads, kKvHeads,
                sequence_length, kDimension, candidate.data(),
                static_cast<void*>(stream))),
        label + " candidate launch");
    ready = ready && test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_values_exact_24_4_256_test_cuda(
                value.data(), probabilities.data(), kQueryHeads, kKvHeads,
                sequence_length, kDimension, replay.data(),
                static_cast<void*>(stream))),
        label + " replay launch");
    ready = ready &&
            test.cuda_ok(cudaStreamSynchronize(stream), label + " sync");
    if (!ready) {
      return;
    }
    expect_bf16_bits_equal(test, candidate.data(), baseline.data(),
                           kOutputElements,
                           label + " candidate matches predecessor");
    expect_bf16_bits_equal(test, replay.data(), candidate.data(),
                           kOutputElements,
                           label + " candidate replay is deterministic");
    for (const std::size_t query_head : kMappingHeads) {
      const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
      for (const std::size_t dimension : kBoundaryDimensions) {
        float accumulated = 0.0F;
        for (std::size_t position = 0U; position < sequence_length;
             ++position) {
          const std::size_t value_index =
              (position * kKvHeads + kv_head) * kDimension + dimension;
          accumulated = std::fma(
              probabilities[query_head * sequence_length + position],
              decode_bf16(value[value_index]), accumulated);
        }
        const std::uint16_t expected = encode_bf16(accumulated);
        const std::size_t output_index =
            query_head * kDimension + dimension;
        test.expect(baseline.data()[output_index] == expected,
                    label + " independent oracle Q=" +
                        std::to_string(query_head) + " D=" +
                        std::to_string(dimension));
      }
    }
    baseline.expect_guards(test, label + " baseline");
    candidate.expect_guards(test, label + " candidate");
    replay.expect_guards(test, label + " replay");
    test.expect(std::memcmp(value.data(), value_snapshot.data(),
                            value.size() * sizeof(std::uint16_t)) == 0,
                label + " value cache is unchanged");
    test.expect(std::memcmp(probabilities.data(), probability_snapshot.data(),
                            probabilities.size() * sizeof(float)) == 0,
                label + " probabilities are unchanged");
  }

  constexpr std::size_t kSpecialSequence = 8U;
  constexpr std::array<std::uint16_t, 16U> kSpecialValues{
      0x0000U, 0x8000U, 0x0001U, 0x8001U, 0x0080U, 0x8080U,
      0x3f80U, 0xbf80U, 0x7f7fU, 0xff7fU, 0x7f80U, 0xff80U,
      0x7fc1U, 0xffc1U, 0x7f81U, 0xff81U};
  constexpr std::array<std::uint32_t, 14U> kSpecialProbabilityBits{
      0x00000000U, 0x80000000U, 0x00000001U, 0x80000001U,
      0x00800000U, 0x80800000U, 0x3f800000U, 0xbf800000U,
      0x3eaaaaabU, 0xbeaaaaabU, 0x7f800000U, 0xff800000U,
      0x7fc12345U, 0xffc12345U};
  const auto run_special_case = [&](const std::string& label,
                                    const std::size_t query_head,
                                    const std::size_t dimension,
                                    const bool expect_infinite,
                                    const bool expect_nan,
                                    const bool expected_sign) {
    const std::vector<std::uint16_t> special_value_snapshot(
        value.data(), value.data() + value.size());
    const std::vector<float> special_probability_snapshot(
        probabilities.data(), probabilities.data() + probabilities.size());
    baseline.initialize(0x7fc1U, 0x1357U, 0x2468U);
    candidate.initialize(0x7fc2U, 0x369cU, 0x147aU);
    replay.initialize(0x7fc3U, 0x55aaU, 0xaa55U);
    bool case_ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_values_baseline_24_4_256_test_cuda(
                value.data(), probabilities.data(), kQueryHeads, kKvHeads,
                kSpecialSequence, kDimension, baseline.data(),
                static_cast<void*>(stream))),
        label + " baseline launch");
    case_ready = case_ready && test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_values_exact_24_4_256_test_cuda(
                value.data(), probabilities.data(), kQueryHeads, kKvHeads,
                kSpecialSequence, kDimension, candidate.data(),
                static_cast<void*>(stream))),
        label + " candidate launch");
    case_ready = case_ready && test.cuda_ok(
        static_cast<cudaError_t>(q3x::runtime::
            launch_attention_values_exact_24_4_256_test_cuda(
                value.data(), probabilities.data(), kQueryHeads, kKvHeads,
                kSpecialSequence, kDimension, replay.data(),
                static_cast<void*>(stream))),
        label + " replay launch");
    case_ready = case_ready &&
                 test.cuda_ok(cudaStreamSynchronize(stream), label + " sync");
    if (!case_ready) {
      ready = false;
      return;
    }
    expect_bf16_bits_equal(test, candidate.data(), baseline.data(),
                           kOutputElements,
                           label + " candidate matches predecessor");
    expect_bf16_bits_equal(test, replay.data(), candidate.data(),
                           kOutputElements,
                           label + " replay is deterministic");
    const float selected =
        decode_bf16(baseline.data()[query_head * kDimension + dimension]);
    if (expect_nan) {
      test.expect(std::isnan(selected), label + " preserves NaN class");
    } else if (expect_infinite) {
      test.expect(std::isinf(selected) &&
                      std::signbit(selected) == expected_sign,
                  label + " preserves infinity sign and class");
    } else {
      test.expect(std::isfinite(selected), label + " remains finite");
    }
    baseline.expect_guards(test, label + " baseline");
    candidate.expect_guards(test, label + " candidate");
    replay.expect_guards(test, label + " replay");
    test.expect(std::memcmp(value.data(), special_value_snapshot.data(),
                            value.size() * sizeof(std::uint16_t)) == 0,
                label + " value cache is unchanged");
    test.expect(std::memcmp(probabilities.data(),
                            special_probability_snapshot.data(),
                            probabilities.size() * sizeof(float)) == 0,
                label + " probabilities are unchanged");
  };

  for (std::size_t index = 0U; ready && index < kSpecialValues.size();
       ++index) {
    std::fill_n(value.data(), value.size(), static_cast<std::uint16_t>(0U));
    std::fill_n(probabilities.data(), probabilities.size(), 0.0F);
    const std::size_t query_head = kMappingHeads[index % kMappingHeads.size()];
    const std::size_t dimension =
        kBoundaryDimensions[index % kBoundaryDimensions.size()];
    const std::size_t position = index % kSpecialSequence;
    const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
    value[(position * kKvHeads + kv_head) * kDimension + dimension] =
        kSpecialValues[index];
    probabilities[query_head * kSpecialSequence + position] = 1.0F;
    const std::uint16_t raw = kSpecialValues[index];
    const bool nonfinite = (raw & 0x7f80U) == 0x7f80U;
    const bool infinite = nonfinite && (raw & 0x007fU) == 0U;
    const bool nan = nonfinite && !infinite;
    run_special_case("attention value special BF16 index=" +
                         std::to_string(index),
                     query_head, dimension, infinite, nan,
                     (raw & 0x8000U) != 0U);
  }
  for (std::size_t index = 0U;
       ready && index < kSpecialProbabilityBits.size(); ++index) {
    std::fill_n(value.data(), value.size(), static_cast<std::uint16_t>(0U));
    std::fill_n(probabilities.data(), probabilities.size(), 0.0F);
    const std::size_t query_head =
        kMappingHeads[(index + 3U) % kMappingHeads.size()];
    const std::size_t dimension =
        kBoundaryDimensions[(index + 5U) % kBoundaryDimensions.size()];
    const std::size_t position = index % kSpecialSequence;
    const std::size_t kv_head = query_head / (kQueryHeads / kKvHeads);
    value[(position * kKvHeads + kv_head) * kDimension + dimension] =
        encode_bf16(1.0F);
    const float raw = float_from_bits(kSpecialProbabilityBits[index]);
    probabilities[query_head * kSpecialSequence + position] = raw;
    run_special_case("attention value special FP32 probability index=" +
                         std::to_string(index),
                     query_head, dimension, std::isinf(raw), std::isnan(raw),
                     std::signbit(raw));
  }
  std::cout << "ATTENTION_VALUE_BINARY_IDENTITY: finite_boundary_lengths="
            << kSequenceLengths.size() << " mapping_heads="
            << kMappingHeads.size() << " dimension_boundaries="
            << kBoundaryDimensions.size() << " special_bf16="
            << kSpecialValues.size() << " special_fp32="
            << kSpecialProbabilityBits.size() << " replay=enabled status="
            << (test.failures() == failures_before ? "PASS" : "FAIL")
            << '\n';
}

void test_attention_values_exact_graph_contract(TestContext& test,
                                                cudaStream_t stream) {
  constexpr std::size_t kSequence = 513U;
  constexpr std::size_t kQueryElements = 24U * 256U;
  constexpr std::size_t kCacheElements = kSequence * 4U * 256U;
  constexpr std::size_t kProbabilityElements = 24U * kSequence;
  constexpr std::size_t kMaximumSupportedSequence =
      static_cast<std::size_t>(std::numeric_limits<unsigned int>::max()) /
      (4U * 256U);
  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<float> probabilities;
  ManagedBuffer<std::uint16_t> output;
  bool ready = test.cuda_ok(query.allocate(kQueryElements),
                            "attention value graph allocate query");
  ready = ready && test.cuda_ok(key.allocate(kCacheElements),
                                "attention value graph allocate key");
  ready = ready && test.cuda_ok(value.allocate(kCacheElements),
                                "attention value graph allocate value");
  ready = ready && test.cuda_ok(
      probabilities.allocate(kProbabilityElements),
      "attention value graph allocate probabilities");
  ready = ready && test.cuda_ok(output.allocate(kQueryElements),
                                "attention value graph allocate output");
  if (!ready) {
    return;
  }

  constexpr std::array<std::pair<const char*, AttentionValueTestLaunch>, 2U>
      kLaunchers{{
          {"baseline", q3x::runtime::
                           launch_attention_values_baseline_24_4_256_test_cuda},
          {"candidate", q3x::runtime::
                            launch_attention_values_exact_24_4_256_test_cuda},
      }};
  for (const auto& [name, launch] : kLaunchers) {
    const std::string prefix =
        std::string("attention value ") + name + " invalid graph ";
    expect_invalid_attention_value_graph(
        test, stream, prefix + "Q23", launch, value.data(),
        probabilities.data(), 23U, 4U, kSequence, 256U, output.data());
    expect_invalid_attention_value_graph(
        test, stream, prefix + "KV5", launch, value.data(),
        probabilities.data(), 24U, 5U, kSequence, 256U, output.data());
    expect_invalid_attention_value_graph(
        test, stream, prefix + "S0", launch, value.data(),
        probabilities.data(), 24U, 4U, 0U, 256U, output.data());
    expect_invalid_attention_value_graph(
        test, stream, prefix + "S overflow", launch, value.data(),
        probabilities.data(), 24U, 4U, kMaximumSupportedSequence + 1U,
        256U, output.data());
    expect_invalid_attention_value_graph(
        test, stream, prefix + "D255", launch, value.data(),
        probabilities.data(), 24U, 4U, kSequence, 255U, output.data());
    expect_invalid_attention_value_graph(
        test, stream, prefix + "null value", launch, nullptr,
        probabilities.data(), 24U, 4U, kSequence, 256U, output.data());
    expect_invalid_attention_value_graph(
        test, stream, prefix + "null probabilities", launch, value.data(),
        nullptr, 24U, 4U, kSequence, 256U, output.data());
    expect_invalid_attention_value_graph(
        test, stream, prefix + "null output", launch, value.data(),
        probabilities.data(), 24U, 4U, kSequence, 256U, nullptr);
    expect_invalid_attention_value_graph(
        test, stream, prefix + "value-output alias", launch, value.data(),
        probabilities.data(), 24U, 4U, kSequence, 256U, value.data());
    expect_invalid_attention_value_graph(
        test, stream, prefix + "probability-output alias", launch,
        value.data(), probabilities.data(), 24U, 4U, kSequence, 256U,
        reinterpret_cast<std::uint16_t*>(probabilities.data()));
  }

  AttentionScoreGraphTopology baseline_topology;
  AttentionScoreGraphTopology candidate_topology;
  ready = capture_attention_value_graph_topology(
      test, stream, "attention value baseline positive graph",
      q3x::runtime::launch_attention_values_baseline_24_4_256_test_cuda,
      value.data(), probabilities.data(), kSequence, output.data(),
      baseline_topology);
  ready = capture_attention_value_graph_topology(
              test, stream, "attention value candidate positive graph",
              q3x::runtime::launch_attention_values_exact_24_4_256_test_cuda,
              value.data(), probabilities.data(), kSequence, output.data(),
              candidate_topology) &&
          ready;
  if (!ready) {
    return;
  }
  test.expect(baseline_topology.function != nullptr &&
                  candidate_topology.function != nullptr &&
                  baseline_topology.function != candidate_topology.function,
              "attention value Graph kernel identities are non-null and distinct");
  test.expect(baseline_topology.grid.x == 24U &&
                  baseline_topology.grid.y == 1U &&
                  baseline_topology.grid.z == 1U,
              "attention value baseline Graph grid is (24,1,1)");
  test.expect(candidate_topology.grid.x == 6U &&
                  candidate_topology.grid.y == 4U &&
                  candidate_topology.grid.z == 1U,
              "attention value exact Graph grid is (6,4,1)");
  test.expect(baseline_topology.block.x == 256U &&
                  baseline_topology.block.y == 1U &&
                  baseline_topology.block.z == 1U &&
                  candidate_topology.block.x == 256U &&
                  candidate_topology.block.y == 1U &&
                  candidate_topology.block.z == 1U,
              "attention value Graph kernels use 256-thread blocks");
  test.expect(baseline_topology.dynamic_shared_bytes == 0U &&
                  candidate_topology.dynamic_shared_bytes == 0U,
              "attention value Graph kernels use zero dynamic shared memory");
  std::cout << "ATTENTION_VALUE_GRAPH_TOPOLOGY: baseline_func="
            << baseline_topology.function << " baseline_grid=("
            << baseline_topology.grid.x << ',' << baseline_topology.grid.y
            << ',' << baseline_topology.grid.z << ") candidate_func="
            << candidate_topology.function << " candidate_grid=("
            << candidate_topology.grid.x << ',' << candidate_topology.grid.y
            << ',' << candidate_topology.grid.z << ") block=("
            << candidate_topology.block.x << ',' << candidate_topology.block.y
            << ',' << candidate_topology.block.z << ") dynamic_shared="
            << candidate_topology.dynamic_shared_bytes << '\n';

  GqaGraphTopology production;
  ready = capture_gqa_graph_topology(
      test, stream, "attention value production S513 graph", query.data(),
      key.data(), value.data(), kSequence, probabilities.data(),
      probabilities.size(), output.data(), production);
  if (!ready) {
    return;
  }
  const bool production_has_baseline =
      std::find(production.non_root_functions.begin(),
                production.non_root_functions.end(),
                baseline_topology.function) !=
      production.non_root_functions.end();
  const bool production_has_candidate =
      std::find(production.non_root_functions.begin(),
                production.non_root_functions.end(),
                candidate_topology.function) !=
      production.non_root_functions.end();
  test.expect(!production_has_baseline,
              "attention value production no longer dispatches the predecessor");
  test.expect(production_has_candidate,
              "attention value production dispatches the exact kernel");
  std::cout << "ATTENTION_VALUE_PRODUCTION_DISPATCH: S=513 baseline_present="
            << (production_has_baseline ? "yes" : "no")
            << " candidate_present="
            << (production_has_candidate ? "yes" : "no") << '\n';

  GqaGraphTopology fallback_production;
  ready = capture_gqa_graph_topology(
      test, stream, "attention value production Q12 fallback graph",
      query.data(), key.data(), value.data(), kSequence,
      probabilities.data(), probabilities.size(), output.data(),
      fallback_production, 12U, 4U, 256U);
  if (!ready) {
    return;
  }
  const bool fallback_has_baseline =
      std::find(fallback_production.non_root_functions.begin(),
                fallback_production.non_root_functions.end(),
                baseline_topology.function) !=
      fallback_production.non_root_functions.end();
  const bool fallback_has_candidate =
      std::find(fallback_production.non_root_functions.begin(),
                fallback_production.non_root_functions.end(),
                candidate_topology.function) !=
      fallback_production.non_root_functions.end();
  test.expect(fallback_has_baseline,
              "attention value Q12 production fallback uses predecessor");
  test.expect(!fallback_has_candidate,
              "attention value Q12 production fallback excludes exact kernel");
  std::cout << "ATTENTION_VALUE_PRODUCTION_FALLBACK: Q=12 KV=4 D=256 S=513 "
               "baseline_present="
            << (fallback_has_baseline ? "yes" : "no")
            << " candidate_present="
            << (fallback_has_candidate ? "yes" : "no") << '\n';
}

void test_attention_values_exact_perf(TestContext& test,
                                      cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_ATTENTION_VALUES_EXACT_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: attention value exact performance gate; set "
                 "Q3X_RUN_ATTENTION_VALUES_EXACT_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kMaximumSequence = 544U;
  constexpr std::size_t kWarmupIterations = 10U;
  constexpr std::size_t kMeasuredIterations = 80U;
  constexpr int kMeasurementRounds = 5;
  constexpr std::array<std::size_t, 12U> kSequenceLengths{
      1U,  2U,  4U,  8U,  16U, 32U,
      64U, 65U, 128U, 257U, 513U, 544U};
  constexpr std::size_t kValueElements =
      kMaximumSequence * kKvHeads * kDimension;
  constexpr std::size_t kProbabilityElements =
      kQueryHeads * kMaximumSequence;
  constexpr std::size_t kOutputElements = kQueryHeads * kDimension;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<float> probabilities;
  ManagedBuffer<std::uint16_t> baseline;
  ManagedBuffer<std::uint16_t> candidate;
  bool ready = test.cuda_ok(value.allocate(kValueElements),
                            "attention value perf allocate value");
  ready = ready && test.cuda_ok(
      probabilities.allocate(kProbabilityElements),
      "attention value perf allocate probabilities");
  ready = ready && test.cuda_ok(baseline.allocate(kOutputElements),
                                "attention value perf allocate baseline");
  ready = ready && test.cuda_ok(candidate.allocate(kOutputElements),
                                "attention value perf allocate candidate");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0U; index < value.size(); ++index) {
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 43U + 13U) % 97U) -
                           48) /
        64.0F);
  }
  for (std::size_t index = 0U; index < probabilities.size(); ++index) {
    probabilities[index] =
        static_cast<float>(static_cast<int>((index * 31U + 7U) % 89U) -
                           44) /
        256.0F;
  }

  const auto launch_baseline = [&](const std::size_t sequence_length) {
    return q3x::runtime::
        launch_attention_values_baseline_24_4_256_test_cuda(
            value.data(), probabilities.data(), kQueryHeads, kKvHeads,
            sequence_length, kDimension, baseline.data(),
            static_cast<void*>(stream));
  };
  const auto launch_candidate = [&](const std::size_t sequence_length) {
    return q3x::runtime::launch_attention_values_exact_24_4_256_test_cuda(
        value.data(), probabilities.data(), kQueryHeads, kKvHeads,
        sequence_length, kDimension, candidate.data(),
        static_cast<void*>(stream));
  };

  for (const std::size_t sequence_length : kSequenceLengths) {
    const std::string label =
        "attention value hot perf S=" + std::to_string(sequence_length);
    for (std::size_t iteration = 0U;
         ready && iteration < kWarmupIterations; ++iteration) {
      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_baseline(sequence_length)),
          label + " baseline warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(
                               launch_candidate(sequence_length)),
                           label + " candidate warmup");
    }
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  label + " warmup sync");
    if (!ready) {
      return;
    }
    double baseline_total = 0.0;
    double candidate_total = 0.0;
    bool timing_finite = true;
    for (int round = 0; round < kMeasurementRounds; ++round) {
      const std::string round_label =
          label + " round=" + std::to_string(round + 1);
      const auto baseline_launch = [&]() {
        return launch_baseline(sequence_length);
      };
      const auto candidate_launch = [&]() {
        return launch_candidate(sequence_length);
      };
      const float baseline_first = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " baseline 1",
          baseline_launch);
      const float candidate_first = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " candidate 1",
          candidate_launch);
      const float candidate_second = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " candidate 2",
          candidate_launch);
      const float baseline_second = measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " baseline 2",
          baseline_launch);
      const bool round_finite =
          std::isfinite(baseline_first) && baseline_first > 0.0F &&
          std::isfinite(candidate_first) && candidate_first > 0.0F &&
          std::isfinite(candidate_second) && candidate_second > 0.0F &&
          std::isfinite(baseline_second) && baseline_second > 0.0F;
      timing_finite = timing_finite && round_finite;
      if (round_finite) {
        baseline_total += baseline_first + baseline_second;
        candidate_total += candidate_first + candidate_second;
      }
      std::cout << "PERF_ATTENTION_VALUES_EXACT_ROUND: cache=hot S="
                << sequence_length << " round=" << round + 1
                << " iterations=" << kMeasuredIterations
                << " baseline1_ms=" << baseline_first
                << " candidate1_ms=" << candidate_first
                << " candidate2_ms=" << candidate_second
                << " baseline2_ms=" << baseline_second << '\n';
    }
    constexpr double kTimedPasses =
        2.0 * static_cast<double>(kMeasurementRounds);
    const double baseline_milliseconds = baseline_total / kTimedPasses;
    const double candidate_milliseconds = candidate_total / kTimedPasses;
    const double speedup = baseline_milliseconds / candidate_milliseconds;
    const bool measurement_valid =
        timing_finite && std::isfinite(speedup) &&
        baseline_milliseconds > 0.0 && candidate_milliseconds > 0.0;
    std::cout << "PERF_ATTENTION_VALUES_EXACT: cache=hot S="
              << sequence_length << " baseline_ms=" << baseline_milliseconds
              << " candidate_ms=" << candidate_milliseconds
              << " speedup=" << speedup
              << " selection=REPORT_ONLY valid="
              << (measurement_valid ? "yes" : "no") << '\n';
    test.expect(measurement_valid, label + " produces a valid measurement");
    expect_bf16_bits_equal(test, candidate.data(), baseline.data(),
                           kOutputElements,
                           label + " final output remains bitwise exact");
  }

  constexpr std::size_t kColdSequence = 513U;
  constexpr std::size_t kColdBankCount = 16U;
  constexpr std::size_t kColdBankElements =
      kColdSequence * kKvHeads * kDimension;
  constexpr std::size_t kColdBankBytes =
      kColdBankElements * sizeof(std::uint16_t);
  constexpr std::size_t kColdWorkingSetBytes =
      kColdBankCount * kColdBankBytes;
  ManagedBuffer<std::uint16_t> cold_values;
  ready = test.cuda_ok(
      cold_values.allocate(kColdBankCount * kColdBankElements),
      "attention value cold perf allocate rotating value banks");
  if (!ready) {
    return;
  }
  for (std::size_t bank = 0U; bank < kColdBankCount; ++bank) {
    std::copy_n(value.data(), kColdBankElements,
                cold_values.data() + bank * kColdBankElements);
  }
  const auto launch_cold_baseline = [&](const std::size_t bank) {
    return q3x::runtime::
        launch_attention_values_baseline_24_4_256_test_cuda(
            cold_values.data() + bank * kColdBankElements,
            probabilities.data(), kQueryHeads, kKvHeads, kColdSequence,
            kDimension, baseline.data(), static_cast<void*>(stream));
  };
  const auto launch_cold_candidate = [&](const std::size_t bank) {
    return q3x::runtime::launch_attention_values_exact_24_4_256_test_cuda(
        cold_values.data() + bank * kColdBankElements, probabilities.data(),
        kQueryHeads, kKvHeads, kColdSequence, kDimension, candidate.data(),
        static_cast<void*>(stream));
  };
  for (std::size_t bank = 0U; ready && bank < kColdBankCount; ++bank) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(launch_cold_baseline(bank)),
        "attention value cold perf baseline first-touch");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(
                             launch_cold_candidate(bank)),
                         "attention value cold perf candidate first-touch");
  }
  ready = ready && test.cuda_ok(
                       cudaStreamSynchronize(stream),
                       "attention value cold perf first-touch sync");
  if (!ready) {
    return;
  }
  const auto measure_cold_baseline = [&](const std::string& label) {
    std::size_t bank = 0U;
    return measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, label, [&]() {
          const int status = launch_cold_baseline(bank);
          bank = (bank + 1U) % kColdBankCount;
          return status;
        });
  };
  const auto measure_cold_candidate = [&](const std::string& label) {
    std::size_t bank = 0U;
    return measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, label, [&]() {
          const int status = launch_cold_candidate(bank);
          bank = (bank + 1U) % kColdBankCount;
          return status;
        });
  };
  double cold_baseline_total = 0.0;
  double cold_candidate_total = 0.0;
  bool cold_timing_finite = true;
  for (int round = 0; round < kMeasurementRounds; ++round) {
    const std::string round_label =
        "attention value cold S513 round=" + std::to_string(round + 1);
    const float baseline_first =
        measure_cold_baseline(round_label + " baseline 1");
    const float candidate_first =
        measure_cold_candidate(round_label + " candidate 1");
    const float candidate_second =
        measure_cold_candidate(round_label + " candidate 2");
    const float baseline_second =
        measure_cold_baseline(round_label + " baseline 2");
    const bool round_finite =
        std::isfinite(baseline_first) && baseline_first > 0.0F &&
        std::isfinite(candidate_first) && candidate_first > 0.0F &&
        std::isfinite(candidate_second) && candidate_second > 0.0F &&
        std::isfinite(baseline_second) && baseline_second > 0.0F;
    cold_timing_finite = cold_timing_finite && round_finite;
    if (round_finite) {
      cold_baseline_total += baseline_first + baseline_second;
      cold_candidate_total += candidate_first + candidate_second;
    }
    std::cout << "PERF_ATTENTION_VALUES_EXACT_COLD_ROUND: "
                 "cache=rotating-cold-value banks="
              << kColdBankCount << " bank_bytes=" << kColdBankBytes
              << " working_set_bytes=" << kColdWorkingSetBytes
              << " S=" << kColdSequence << " iterations="
              << kMeasuredIterations
              << " round=" << round + 1
              << " baseline1_ms=" << baseline_first
              << " candidate1_ms=" << candidate_first
              << " candidate2_ms=" << candidate_second
              << " baseline2_ms=" << baseline_second << '\n';
  }
  constexpr double kColdTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const double cold_baseline_milliseconds =
      cold_baseline_total / kColdTimedPasses;
  const double cold_candidate_milliseconds =
      cold_candidate_total / kColdTimedPasses;
  const double cold_speedup =
      cold_baseline_milliseconds / cold_candidate_milliseconds;
  const bool cold_measurement_valid =
      cold_timing_finite && std::isfinite(cold_speedup) &&
      cold_baseline_milliseconds > 0.0 &&
      cold_candidate_milliseconds > 0.0;
  std::cout << "PERF_ATTENTION_VALUES_EXACT_COLD: "
               "cache=rotating-cold-value banks="
            << kColdBankCount << " bank_bytes=" << kColdBankBytes
            << " working_set_bytes=" << kColdWorkingSetBytes
            << " S=" << kColdSequence
            << " baseline_ms=" << cold_baseline_milliseconds
            << " candidate_ms=" << cold_candidate_milliseconds
            << " speedup=" << cold_speedup
            << " selection=REPORT_ONLY valid="
            << (cold_measurement_valid ? "yes" : "no") << '\n';
  test.expect(cold_measurement_valid,
              "attention value cold S513 produces a valid measurement");
  expect_bf16_bits_equal(test, candidate.data(), baseline.data(),
                         kOutputElements,
                         "attention value cold S513 stays bitwise exact");

  constexpr std::size_t kChainFirstSequence = 65U;
  constexpr std::size_t kChainLastSequence = 513U;
  constexpr std::size_t kChainWarmupIterations = 10U;
  constexpr std::size_t kChainMeasuredIterations = 1U;
  const std::string chain_label =
      "attention value hot chain S65..513";
  const auto launch_baseline_chain = [&]() {
    for (std::size_t sequence_length = kChainFirstSequence;
         sequence_length <= kChainLastSequence; ++sequence_length) {
      const int status = launch_baseline(sequence_length);
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  const auto launch_candidate_chain = [&]() {
    for (std::size_t sequence_length = kChainFirstSequence;
         sequence_length <= kChainLastSequence; ++sequence_length) {
      const int status = launch_candidate(sequence_length);
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  for (std::size_t iteration = 0U;
       ready && iteration < kChainWarmupIterations; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline_chain()),
                         chain_label + " baseline warmup");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate_chain()),
                         chain_label + " candidate warmup");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                chain_label + " warmup sync");
  if (!ready) {
    return;
  }
  double chain_baseline_total = 0.0;
  double chain_candidate_total = 0.0;
  bool chain_timing_finite = true;
  for (int round = 0; round < kMeasurementRounds; ++round) {
    const std::string round_label =
        chain_label + " round=" + std::to_string(round + 1);
    const float baseline_first = measure_cuda_span_milliseconds(
        test, stream, kChainMeasuredIterations,
        round_label + " baseline 1", launch_baseline_chain);
    const float candidate_first = measure_cuda_span_milliseconds(
        test, stream, kChainMeasuredIterations,
        round_label + " candidate 1", launch_candidate_chain);
    const float candidate_second = measure_cuda_span_milliseconds(
        test, stream, kChainMeasuredIterations,
        round_label + " candidate 2", launch_candidate_chain);
    const float baseline_second = measure_cuda_span_milliseconds(
        test, stream, kChainMeasuredIterations,
        round_label + " baseline 2", launch_baseline_chain);
    const bool round_finite =
        std::isfinite(baseline_first) && baseline_first > 0.0F &&
        std::isfinite(candidate_first) && candidate_first > 0.0F &&
        std::isfinite(candidate_second) && candidate_second > 0.0F &&
        std::isfinite(baseline_second) && baseline_second > 0.0F;
    chain_timing_finite = chain_timing_finite && round_finite;
    if (round_finite) {
      chain_baseline_total += baseline_first + baseline_second;
      chain_candidate_total += candidate_first + candidate_second;
    }
    std::cout << "PERF_ATTENTION_VALUES_EXACT_CHAIN_ROUND: cache=hot round="
              << round + 1 << " kernels_per_chain="
              << (kChainLastSequence - kChainFirstSequence + 1U)
              << " baseline1_ms=" << baseline_first
              << " candidate1_ms=" << candidate_first
              << " candidate2_ms=" << candidate_second
              << " baseline2_ms=" << baseline_second << '\n';
  }
  constexpr double kChainTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const double chain_baseline_milliseconds =
      chain_baseline_total / kChainTimedPasses;
  const double chain_candidate_milliseconds =
      chain_candidate_total / kChainTimedPasses;
  const double chain_speedup =
      chain_baseline_milliseconds / chain_candidate_milliseconds;
  const bool chain_measurement_valid =
      chain_timing_finite && std::isfinite(chain_speedup) &&
      chain_baseline_milliseconds > 0.0 &&
      chain_candidate_milliseconds > 0.0;
  std::cout << "PERF_ATTENTION_VALUES_EXACT_CHAIN: cache=hot first_S="
            << kChainFirstSequence << " last_S=" << kChainLastSequence
            << " kernels_per_chain="
            << (kChainLastSequence - kChainFirstSequence + 1U)
            << " baseline_ms=" << chain_baseline_milliseconds
            << " candidate_ms=" << chain_candidate_milliseconds
            << " speedup=" << chain_speedup
            << " selection=REPORT_ONLY valid="
            << (chain_measurement_valid ? "yes" : "no") << '\n';
  test.expect(chain_measurement_valid,
              chain_label + " produces a valid aggregate measurement");
  expect_bf16_bits_equal(test, candidate.data(), baseline.data(),
                         kOutputElements,
                         chain_label + " final S513 output is bitwise exact");
}

using FusedGqaSigmoidTestLaunch = int (*)(
    const std::uint16_t*, const std::uint16_t*, const std::uint16_t*,
    std::size_t, float, float*, std::size_t, const std::uint16_t*,
    std::uint16_t*, void*) noexcept;

using FusedGqaSigmoidResourceQuery = int (*)(
    int*, std::size_t*, std::size_t*, int*, int*) noexcept;

[[nodiscard]] bool capture_fused_gqa_sigmoid_graph(
    TestContext& test, cudaStream_t stream, const std::string& label,
    const FusedGqaSigmoidTestLaunch launch,
    const std::uint16_t* const query,
    const std::uint16_t* const key,
    const std::uint16_t* const value,
    const std::size_t sequence_length,
    float* const scratch,
    const std::size_t scratch_elements,
    const std::uint16_t* const gate,
    std::uint16_t* const output,
    AttentionScoreGraphTopology& topology) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return false;
  }
  const cudaError_t launch_status = static_cast<cudaError_t>(launch(
      query, key, value, sequence_length, 0.0625F, scratch,
      scratch_elements, gate, output, static_cast<void*>(stream)));
  test.expect(launch_status == cudaSuccess, label + " launch succeeds");
  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                            label + " end capture");
  if (!ready) {
    return false;
  }

  std::size_t node_count = 0U;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                       label + " count nodes");
  test.expect(node_count == 1U, label + " captures exactly one node");
  if (ready && node_count == 1U) {
    cudaGraphNode_t node = nullptr;
    std::size_t capacity = 1U;
    ready = test.cuda_ok(cudaGraphGetNodes(graph, &node, &capacity),
                         label + " fetch node");
    test.expect(capacity == 1U, label + " fetches one node");
    cudaGraphNodeType node_type = cudaGraphNodeTypeEmpty;
    ready = ready && test.cuda_ok(cudaGraphNodeGetType(node, &node_type),
                                  label + " read node type");
    test.expect(node_type == cudaGraphNodeTypeKernel,
                label + " node is a kernel");
    cudaKernelNodeParams parameters{};
    ready = ready && test.cuda_ok(
                         cudaGraphKernelNodeGetParams(node, &parameters),
                         label + " read kernel parameters");
    if (ready) {
      topology.function = parameters.func;
      topology.grid = parameters.gridDim;
      topology.block = parameters.blockDim;
      topology.dynamic_shared_bytes = parameters.sharedMemBytes;
    }
  } else {
    ready = false;
  }

  cudaGraphExec_t graph_exec = nullptr;
  if (ready) {
    ready = test.cuda_ok(
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0U),
        label + " instantiate");
  }
  if (ready) {
    ready = test.cuda_ok(cudaGraphLaunch(graph_exec, stream),
                         label + " replay") &&
            ready;
    ready = test.cuda_ok(cudaStreamSynchronize(stream),
                         label + " replay synchronize") &&
            ready;
  }
  if (graph_exec != nullptr) {
    (void)test.cuda_ok(cudaGraphExecDestroy(graph_exec),
                       label + " graph exec destroy");
  }
  ready = test.cuda_ok(cudaGraphDestroy(graph), label + " graph destroy") &&
          ready;
  return ready;
}

void expect_invalid_fused_gqa_sigmoid_graph(
    TestContext& test, cudaStream_t stream, const std::string& label,
    const FusedGqaSigmoidTestLaunch launch,
    const std::uint16_t* const query,
    const std::uint16_t* const key,
    const std::uint16_t* const value,
    const std::size_t sequence_length,
    const float scale,
    float* const scratch,
    const std::size_t scratch_elements,
    const std::uint16_t* const gate,
    std::uint16_t* const output) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return;
  }
  const cudaError_t launch_status = static_cast<cudaError_t>(launch(
      query, key, value, sequence_length, scale, scratch, scratch_elements,
      gate, output, static_cast<void*>(stream)));
  test.expect(launch_status == cudaErrorInvalidValue,
              label + " returns cudaErrorInvalidValue");
  cudaGraph_t graph = nullptr;
  if (!test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                    label + " end capture")) {
    return;
  }
  std::size_t node_count = 0U;
  if (test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                   label + " count nodes")) {
    test.expect(node_count == 0U,
                label + " fails before enqueue and captures zero nodes");
  }
  (void)test.cuda_ok(cudaGraphDestroy(graph), label + " graph destroy");
}

void test_fused_gqa_sigmoid_gate_warp_positions_contract(
    TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kSequence = 44U;
  constexpr std::size_t kQueryElements = kQueryHeads * kDimension;
  constexpr std::size_t kCacheElements =
      kSequence * kKvHeads * kDimension;
  constexpr std::size_t kScratchElements = kQueryHeads * kSequence;

  AttentionScoreKernelResources predecessor_resources;
  AttentionScoreKernelResources production_resources;
  AttentionScoreKernelResources direct_resources;
  constexpr std::array<std::pair<const char*,
                                 FusedGqaSigmoidResourceQuery>, 3U>
      kResourceQueries{{
          {"predecessor", q3x::runtime::
                              query_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_resources_test_cuda},
          {"production", q3x::runtime::
                             query_gqa_attention_sigmoid_gate_24_4_256_resources_test_cuda},
          {"direct", q3x::runtime::
                         query_gqa_attention_sigmoid_gate_warp_positions_24_4_256_resources_test_cuda},
      }};
  for (const auto& [name, query_resources] : kResourceQueries) {
    AttentionScoreKernelResources outputs;
    for (std::size_t null_index = 0U; null_index < 5U; ++null_index) {
      const cudaError_t status = static_cast<cudaError_t>(query_resources(
          null_index == 0U ? nullptr : &outputs.registers,
          null_index == 1U ? nullptr : &outputs.static_shared_bytes,
          null_index == 2U ? nullptr : &outputs.local_bytes,
          null_index == 3U ? nullptr : &outputs.maximum_threads,
          null_index == 4U
              ? nullptr
              : &outputs.active_blocks_per_multiprocessor));
      test.expect(status == cudaErrorInvalidValue,
                  std::string("fused GQA ") + name +
                      " resource query rejects null output " +
                      std::to_string(null_index));
    }
  }
  bool ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_resources_test_cuda(
              &predecessor_resources.registers,
              &predecessor_resources.static_shared_bytes,
              &predecessor_resources.local_bytes,
              &predecessor_resources.maximum_threads,
              &predecessor_resources.active_blocks_per_multiprocessor)),
      "fused GQA predecessor resource query");
  ready = ready && test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_gqa_attention_sigmoid_gate_24_4_256_resources_test_cuda(
              &production_resources.registers,
              &production_resources.static_shared_bytes,
              &production_resources.local_bytes,
              &production_resources.maximum_threads,
              &production_resources.active_blocks_per_multiprocessor)),
      "fused GQA production resource query");
  ready = ready && test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_gqa_attention_sigmoid_gate_warp_positions_24_4_256_resources_test_cuda(
              &direct_resources.registers,
              &direct_resources.static_shared_bytes,
              &direct_resources.local_bytes,
              &direct_resources.maximum_threads,
              &direct_resources.active_blocks_per_multiprocessor)),
      "fused GQA direct warp-position resource query");
  if (!ready) {
    return;
  }
  const auto expect_promoted_resources = [&](const std::string& name,
                                             const auto& resources) {
    const bool gate_passed =
        resources.registers == 40 &&
        resources.static_shared_bytes == 1'280U &&
        resources.local_bytes == 0U &&
        resources.maximum_threads >= 256 &&
        resources.active_blocks_per_multiprocessor == 6;
    test.expect(resources.registers == 40,
                "fused GQA " + name + " retains exactly 40 registers");
    test.expect(resources.static_shared_bytes == 1'280U,
                "fused GQA " + name + " retains 1280 static shared bytes");
    test.expect(resources.local_bytes == 0U,
                "fused GQA " + name + " has no local-memory spills");
    test.expect(resources.maximum_threads >= 256,
                "fused GQA " + name + " supports 256 threads");
    test.expect(resources.active_blocks_per_multiprocessor == 6,
                "fused GQA " + name + " retains six CTAs per SM");
    return gate_passed;
  };
  const bool resources_gate =
      expect_promoted_resources("predecessor", predecessor_resources) &
      expect_promoted_resources("production", production_resources) &
      expect_promoted_resources("direct warp hook", direct_resources);
  std::cout << "GQA_SIGMOID_WARP_POSITIONS_RESOURCES: predecessor_regs="
            << predecessor_resources.registers
            << " predecessor_static_shared="
            << predecessor_resources.static_shared_bytes
            << " predecessor_local=" << predecessor_resources.local_bytes
            << " predecessor_active_blocks="
            << predecessor_resources.active_blocks_per_multiprocessor
            << " production_regs="
            << production_resources.registers
            << " production_static_shared="
            << production_resources.static_shared_bytes
            << " production_local=" << production_resources.local_bytes
            << " production_active_blocks="
            << production_resources.active_blocks_per_multiprocessor
            << " direct_regs=" << direct_resources.registers
            << " direct_static_shared="
            << direct_resources.static_shared_bytes
            << " direct_local=" << direct_resources.local_bytes
            << " direct_active_blocks="
            << direct_resources.active_blocks_per_multiprocessor
            << " gate=" << (resources_gate ? "PASS" : "FAIL") << '\n';

  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<std::uint16_t> gate;
  ManagedBuffer<float> predecessor_scratch;
  ManagedBuffer<float> production_scratch;
  ManagedBuffer<float> direct_scratch;
  ManagedBuffer<std::uint16_t> predecessor_output;
  ManagedBuffer<std::uint16_t> production_output;
  ManagedBuffer<std::uint16_t> direct_output;
  ready = test.cuda_ok(query.allocate(kQueryElements),
                       "fused GQA graph allocate query");
  ready = ready && test.cuda_ok(key.allocate(kCacheElements),
                                "fused GQA graph allocate key");
  ready = ready && test.cuda_ok(value.allocate(kCacheElements),
                                "fused GQA graph allocate value");
  ready = ready && test.cuda_ok(gate.allocate(kQueryElements),
                                "fused GQA graph allocate gate");
  ready = ready && test.cuda_ok(predecessor_scratch.allocate(kScratchElements),
                                "fused GQA graph allocate predecessor scratch");
  ready = ready && test.cuda_ok(production_scratch.allocate(kScratchElements),
                                "fused GQA graph allocate production scratch");
  ready = ready && test.cuda_ok(direct_scratch.allocate(kScratchElements),
                                "fused GQA graph allocate direct scratch");
  ready = ready && test.cuda_ok(predecessor_output.allocate(kQueryElements),
                                "fused GQA graph allocate predecessor output");
  ready = ready && test.cuda_ok(production_output.allocate(kQueryElements),
                                "fused GQA graph allocate production output");
  ready = ready && test.cuda_ok(direct_output.allocate(kQueryElements),
                                "fused GQA graph allocate direct output");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0U; index < query.size(); ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 17U + 5U) % 127U) -
                           63) /
        64.0F);
    gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 23U + 3U) % 97U) -
                           48) /
        16.0F);
  }
  for (std::size_t index = 0U; index < key.size(); ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 7U) % 113U) -
                           56) /
        64.0F);
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 31U + 11U) % 109U) -
                           54) /
        32.0F);
  }
  const std::vector<std::uint16_t> query_snapshot(query.data(),
                                                   query.data() + query.size());
  const std::vector<std::uint16_t> key_snapshot(key.data(),
                                                 key.data() + key.size());
  const std::vector<std::uint16_t> value_snapshot(
      value.data(), value.data() + value.size());
  const std::vector<std::uint16_t> gate_snapshot(gate.data(),
                                                  gate.data() + gate.size());

  constexpr FusedGqaSigmoidTestLaunch kDirectLaunch =
      q3x::runtime::launch_gqa_attention_sigmoid_gate_warp_positions_24_4_256_test_cuda;
  constexpr std::array<std::pair<const char*, FusedGqaSigmoidTestLaunch>, 2U>
      kPromotedLaunches{{
          {"production",
           q3x::runtime::launch_gqa_attention_sigmoid_gate_24_4_256_cuda},
          {"direct", kDirectLaunch},
      }};
  struct NullPointerCase {
    const char* name;
    int pointer;
  };
  constexpr std::array<NullPointerCase, 6U> kNullCases{{
      {"query", 0}, {"key", 1}, {"value", 2},
      {"scratch", 3}, {"gate", 4}, {"output", 5},
  }};
  for (const auto& [route_name, launch] : kPromotedLaunches) {
    const std::string prefix = std::string("fused GQA ") + route_name +
                               " invalid ";
    expect_invalid_fused_gqa_sigmoid_graph(
        test, stream, prefix + "S0", launch, query.data(), key.data(),
        value.data(), 0U, 0.0625F, direct_scratch.data(),
        direct_scratch.size(), gate.data(), direct_output.data());
    expect_invalid_fused_gqa_sigmoid_graph(
        test, stream, prefix + "S65", launch, query.data(), key.data(),
        value.data(), 65U, 0.0625F, direct_scratch.data(),
        direct_scratch.size(), gate.data(), direct_output.data());
    expect_invalid_fused_gqa_sigmoid_graph(
        test, stream, prefix + "negative scale", launch, query.data(),
        key.data(), value.data(), kSequence, -1.0F, direct_scratch.data(),
        direct_scratch.size(), gate.data(), direct_output.data());
    expect_invalid_fused_gqa_sigmoid_graph(
        test, stream, prefix + "NaN scale", launch, query.data(), key.data(),
        value.data(), kSequence,
        std::numeric_limits<float>::quiet_NaN(), direct_scratch.data(),
        direct_scratch.size(), gate.data(), direct_output.data());
    expect_invalid_fused_gqa_sigmoid_graph(
        test, stream, prefix + "short scratch", launch, query.data(),
        key.data(), value.data(), kSequence, 0.0625F,
        direct_scratch.data(), kScratchElements - 1U, gate.data(),
        direct_output.data());
    for (const NullPointerCase& null_case : kNullCases) {
      expect_invalid_fused_gqa_sigmoid_graph(
          test, stream, prefix + "null " + null_case.name, launch,
          null_case.pointer == 0 ? nullptr : query.data(),
          null_case.pointer == 1 ? nullptr : key.data(),
          null_case.pointer == 2 ? nullptr : value.data(), kSequence,
          0.0625F,
          null_case.pointer == 3 ? nullptr : direct_scratch.data(),
          direct_scratch.size(),
          null_case.pointer == 4 ? nullptr : gate.data(),
          null_case.pointer == 5 ? nullptr : direct_output.data());
    }
  }

  AttentionScoreGraphTopology predecessor_topology;
  AttentionScoreGraphTopology production_topology;
  AttentionScoreGraphTopology direct_topology;
  ready = capture_fused_gqa_sigmoid_graph(
      test, stream, "fused GQA predecessor graph",
      q3x::runtime::
          launch_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_test_cuda,
      query.data(), key.data(), value.data(), kSequence,
      predecessor_scratch.data(), predecessor_scratch.size(), gate.data(),
      predecessor_output.data(), predecessor_topology);
  ready = capture_fused_gqa_sigmoid_graph(
              test, stream, "fused GQA production graph",
              q3x::runtime::launch_gqa_attention_sigmoid_gate_24_4_256_cuda,
              query.data(), key.data(), value.data(), kSequence,
              production_scratch.data(), production_scratch.size(),
              gate.data(), production_output.data(), production_topology) &&
          ready;
  ready = capture_fused_gqa_sigmoid_graph(
              test, stream, "fused GQA direct warp graph", kDirectLaunch,
              query.data(), key.data(), value.data(), kSequence,
              direct_scratch.data(), direct_scratch.size(), gate.data(),
              direct_output.data(), direct_topology) &&
          ready;
  if (!ready) {
    return;
  }
  const bool function_identity_gate =
      predecessor_topology.function != nullptr &&
      production_topology.function != nullptr &&
      direct_topology.function != nullptr &&
      predecessor_topology.function != production_topology.function &&
      production_topology.function == direct_topology.function;
  test.expect(predecessor_topology.function != nullptr &&
                  production_topology.function != nullptr &&
                  direct_topology.function != nullptr &&
                  predecessor_topology.function != production_topology.function,
              "fused GQA predecessor and promoted production functions differ");
  test.expect(production_topology.function == direct_topology.function,
              "fused GQA public and direct hooks select the promoted kernel");
  const auto fixed_topology = [](const auto& topology) {
    return topology.grid.x == 24U && topology.grid.y == 1U &&
           topology.grid.z == 1U && topology.block.x == 256U &&
           topology.block.y == 1U && topology.block.z == 1U &&
           topology.dynamic_shared_bytes == 0U;
  };
  const bool topology_gate = fixed_topology(predecessor_topology) &&
                             fixed_topology(production_topology) &&
                             fixed_topology(direct_topology);
  test.expect(topology_gate,
              "fused GQA Graph routes are one 24x256 zero-dynamic kernel");
  const bool output_gate =
      std::memcmp(production_output.data(), predecessor_output.data(),
                  kQueryElements * sizeof(std::uint16_t)) == 0 &&
      std::memcmp(direct_output.data(), predecessor_output.data(),
                  kQueryElements * sizeof(std::uint16_t)) == 0;
  test.expect(output_gate,
              "fused GQA promoted Graph outputs are bitwise exact");
  const bool scratch_gate =
      std::memcmp(production_scratch.data(), predecessor_scratch.data(),
                  kScratchElements * sizeof(float)) == 0 &&
      std::memcmp(direct_scratch.data(), predecessor_scratch.data(),
                  kScratchElements * sizeof(float)) == 0;
  test.expect(scratch_gate,
              "fused GQA promoted Graph probabilities are bitwise exact");
  const bool inputs_gate =
      std::memcmp(query.data(), query_snapshot.data(),
                  query.size() * sizeof(std::uint16_t)) == 0 &&
      std::memcmp(key.data(), key_snapshot.data(),
                  key.size() * sizeof(std::uint16_t)) == 0 &&
      std::memcmp(value.data(), value_snapshot.data(),
                  value.size() * sizeof(std::uint16_t)) == 0 &&
      std::memcmp(gate.data(), gate_snapshot.data(),
                  gate.size() * sizeof(std::uint16_t)) == 0;
  test.expect(inputs_gate,
              "fused GQA Graph inputs are preserved");
  const bool graph_gate = function_identity_gate && topology_gate &&
                          output_gate && scratch_gate && inputs_gate;
  std::cout << "GQA_SIGMOID_WARP_POSITIONS_GRAPH: predecessor_nodes=1 "
               "production_nodes=1 direct_nodes=1 predecessor_distinct=true "
               "public_direct_same=true grid=24 block=256 dynamic_shared=0 "
               "replay=bitwise gate="
            << (graph_gate ? "PASS" : "FAIL") << '\n';
}

void run_fused_gqa_sigmoid_gate_exact_case(TestContext& test,
                                           cudaStream_t stream,
                                           const std::size_t sequence_length) {
  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr float kScale = 0.0625F;
  constexpr std::uint16_t kPoison = 0x7fc1U;
  const std::size_t query_elements = kQueryHeads * kDimension;
  const std::size_t cache_elements =
      sequence_length * kKvHeads * kDimension;
  const std::size_t scratch_elements = kQueryHeads * sequence_length;
  const std::string label =
      "fused GQA sigmoid S=" + std::to_string(sequence_length);

  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<std::uint16_t> gate;
  ManagedBuffer<float> baseline_scratch;
  ManagedBuffer<float> predecessor_scratch;
  ManagedBuffer<float> candidate_scratch;
  ManagedBuffer<float> warp_candidate_scratch;
  ManagedBuffer<std::uint16_t> baseline_output;
  ManagedBuffer<std::uint16_t> predecessor_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  ManagedBuffer<std::uint16_t> warp_candidate_output;
  bool ready = test.cuda_ok(query.allocate(query_elements), label + " query");
  ready = ready && test.cuda_ok(key.allocate(cache_elements), label + " key");
  ready = ready &&
          test.cuda_ok(value.allocate(cache_elements), label + " value");
  ready = ready && test.cuda_ok(gate.allocate(query_elements), label + " gate");
  ready = ready && test.cuda_ok(baseline_scratch.allocate(scratch_elements),
                                label + " baseline scratch");
  ready = ready &&
          test.cuda_ok(predecessor_scratch.allocate(scratch_elements),
                       label + " predecessor scratch");
  ready = ready && test.cuda_ok(candidate_scratch.allocate(scratch_elements),
                                label + " candidate scratch");
  ready = ready &&
          test.cuda_ok(warp_candidate_scratch.allocate(scratch_elements),
                       label + " warp-position candidate scratch");
  ready = ready && test.cuda_ok(baseline_output.allocate(query_elements),
                                label + " baseline output");
  ready = ready &&
          test.cuda_ok(predecessor_output.allocate(query_elements),
                       label + " predecessor output");
  ready = ready && test.cuda_ok(candidate_output.allocate(query_elements),
                                label + " candidate output");
  ready = ready &&
          test.cuda_ok(warp_candidate_output.allocate(query_elements),
                       label + " warp-position candidate output");
  if (!ready) {
    return;
  }

  for (std::size_t index = 0U; index < query_elements; ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 17U + 5U) % 127U) -
                           63) /
        64.0F);
    gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 23U + 3U) % 97U) -
                           48) /
        16.0F);
  }
  for (std::size_t index = 0U; index < cache_elements; ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 7U) % 113U) -
                           56) /
        64.0F);
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 31U + 11U) % 109U) -
                           54) /
        32.0F);
  }
  std::fill_n(baseline_scratch.data(), scratch_elements,
              std::numeric_limits<float>::quiet_NaN());
  std::fill_n(predecessor_scratch.data(), scratch_elements,
              std::numeric_limits<float>::quiet_NaN());
  std::fill_n(candidate_scratch.data(), scratch_elements,
              std::numeric_limits<float>::quiet_NaN());
  std::fill_n(warp_candidate_scratch.data(), scratch_elements,
              std::numeric_limits<float>::quiet_NaN());
  std::fill_n(baseline_output.data(), query_elements, kPoison);
  std::fill_n(predecessor_output.data(), query_elements, kPoison);
  std::fill_n(candidate_output.data(), query_elements, kPoison);
  std::fill_n(warp_candidate_output.data(), query_elements, kPoison);
  const std::vector<std::uint16_t> query_snapshot(
      query.data(), query.data() + query.size());
  const std::vector<std::uint16_t> key_snapshot(key.data(),
                                                key.data() + key.size());
  const std::vector<std::uint16_t> value_snapshot(
      value.data(), value.data() + value.size());
  const std::vector<std::uint16_t> gate_snapshot(gate.data(),
                                                  gate.data() + gate.size());

  ready = launch_after_stale(test, stream, label + " baseline", [&]() {
    const int attention_status =
        q3x::runtime::launch_gqa_attention_reference_cuda(
            query.data(), key.data(), value.data(), kQueryHeads, kKvHeads,
            sequence_length, kDimension, kScale, baseline_scratch.data(),
            baseline_scratch.size(), baseline_output.data(),
            static_cast<void*>(stream));
    if (attention_status != static_cast<int>(cudaSuccess)) {
      return attention_status;
    }
    return q3x::runtime::launch_sigmoid_gate_reference_cuda(
        baseline_output.data(), gate.data(), query_elements,
        baseline_output.data(), static_cast<void*>(stream));
  });
  ready = ready &&
          launch_after_stale(test, stream, label + " predecessor", [&]() {
            return q3x::runtime::
                launch_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_test_cuda(
                    query.data(), key.data(), value.data(), sequence_length,
                    kScale, predecessor_scratch.data(),
                    predecessor_scratch.size(), gate.data(),
                    predecessor_output.data(), static_cast<void*>(stream));
          });
  ready = ready &&
          launch_after_stale(test, stream, label + " candidate", [&]() {
            return q3x::runtime::
                launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                    query.data(), key.data(), value.data(), sequence_length,
                    kScale, candidate_scratch.data(),
                    candidate_scratch.size(), gate.data(),
                    candidate_output.data(), static_cast<void*>(stream));
          });
  ready = ready && launch_after_stale(
                       test, stream, label + " warp-position candidate",
                       [&]() {
                         return q3x::runtime::
                             launch_gqa_attention_sigmoid_gate_warp_positions_24_4_256_test_cuda(
                                 query.data(), key.data(), value.data(),
                                 sequence_length, kScale,
                                 warp_candidate_scratch.data(),
                                 warp_candidate_scratch.size(), gate.data(),
                                 warp_candidate_output.data(),
                                 static_cast<void*>(stream));
                       });
  if (!ready) {
    return;
  }

  expect_bf16_bits_equal(test, predecessor_output.data(),
                         baseline_output.data(), query_elements,
                         label + " predecessor gated output");
  test.expect(
      std::memcmp(predecessor_scratch.data(), baseline_scratch.data(),
                  scratch_elements * sizeof(float)) == 0,
      label + " predecessor probability scratch is bitwise identical");
  expect_bf16_bits_equal(test, candidate_output.data(),
                         baseline_output.data(), query_elements,
                         label + " gated output");
  test.expect(std::memcmp(candidate_scratch.data(), baseline_scratch.data(),
                          scratch_elements * sizeof(float)) == 0,
              label + " probability scratch is bitwise identical");
  expect_bf16_bits_equal(test, warp_candidate_output.data(),
                         baseline_output.data(), query_elements,
                         label + " warp-position gated output");
  expect_bf16_bits_equal(test, warp_candidate_output.data(),
                         candidate_output.data(), query_elements,
                         label + " direct and public gated output");
  test.expect(
      std::memcmp(warp_candidate_scratch.data(), baseline_scratch.data(),
                  scratch_elements * sizeof(float)) == 0,
      label + " warp-position probability scratch is bitwise identical");
  test.expect(
      std::memcmp(warp_candidate_scratch.data(), candidate_scratch.data(),
                  scratch_elements * sizeof(float)) == 0,
      label + " direct and public probability scratch are bitwise identical");
  test.expect(std::memcmp(query.data(), query_snapshot.data(),
                          query.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(key.data(), key_snapshot.data(),
                              key.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(value.data(), value_snapshot.data(),
                              value.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(gate.data(), gate_snapshot.data(),
                              gate.size() * sizeof(std::uint16_t)) == 0,
              label + " preserves query, caches, and gate inputs");
}

void test_fused_gqa_sigmoid_gate_exact(TestContext& test,
                                       cudaStream_t stream) {
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                  nullptr, nullptr, nullptr, 0U, 0.0625F, nullptr, 0U,
                  nullptr, nullptr)) == cudaErrorInvalidValue,
      "fused GQA rejects empty sequence");
  test.expect(
      static_cast<cudaError_t>(
          q3x::runtime::launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                  nullptr, nullptr, nullptr, 65U, 0.0625F, nullptr, 0U,
                  nullptr, nullptr)) == cudaErrorInvalidValue,
      "fused GQA rejects sequence above 64");
  for (const std::size_t sequence_length :
       {1U, 16U, 20U, 32U, 44U, 64U}) {
    run_fused_gqa_sigmoid_gate_exact_case(test, stream, sequence_length);
  }
}

void test_fused_gqa_sigmoid_gate_perf(TestContext& test,
                                      cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_GQA_SIGMOID_FUSED_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: fused GQA sigmoid performance gate; set "
                 "Q3X_RUN_GQA_SIGMOID_FUSED_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kSequence = 44U;
  constexpr float kScale = 0.0625F;
  constexpr std::size_t kWarmupIterations = 64U;
  constexpr std::size_t kMeasuredIterations = 512U;
  constexpr int kMeasurementRounds = 3;
  constexpr double kMinimumSpeedup = 1.05;
  constexpr std::size_t kQueryElements = kQueryHeads * kDimension;
  constexpr std::size_t kCacheElements =
      kSequence * kKvHeads * kDimension;
  constexpr std::size_t kScratchElements = kQueryHeads * kSequence;
  const std::string label = "fused GQA sigmoid perf S=44";

  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<std::uint16_t> gate;
  ManagedBuffer<float> baseline_scratch;
  ManagedBuffer<float> candidate_scratch;
  ManagedBuffer<std::uint16_t> baseline_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(query.allocate(kQueryElements), label + " query");
  ready = ready && test.cuda_ok(key.allocate(kCacheElements), label + " key");
  ready = ready &&
          test.cuda_ok(value.allocate(kCacheElements), label + " value");
  ready = ready && test.cuda_ok(gate.allocate(kQueryElements), label + " gate");
  ready = ready && test.cuda_ok(baseline_scratch.allocate(kScratchElements),
                                label + " baseline scratch");
  ready = ready && test.cuda_ok(candidate_scratch.allocate(kScratchElements),
                                label + " candidate scratch");
  ready = ready && test.cuda_ok(baseline_output.allocate(kQueryElements),
                                label + " baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(kQueryElements),
                                label + " candidate output");
  if (!ready) {
    return;
  }

  for (std::size_t index = 0U; index < kQueryElements; ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 17U + 5U) % 127U) -
                           63) /
        64.0F);
    gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 23U + 3U) % 97U) -
                           48) /
        16.0F);
  }
  for (std::size_t index = 0U; index < kCacheElements; ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 7U) % 113U) -
                           56) /
        64.0F);
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 31U + 11U) % 109U) -
                           54) /
        32.0F);
  }

  const auto launch_baseline = [&]() {
    const int attention_status =
        q3x::runtime::launch_gqa_attention_reference_cuda(
            query.data(), key.data(), value.data(), kQueryHeads, kKvHeads,
            kSequence, kDimension, kScale, baseline_scratch.data(),
            baseline_scratch.size(), baseline_output.data(),
            static_cast<void*>(stream));
    if (attention_status != static_cast<int>(cudaSuccess)) {
      return attention_status;
    }
    return q3x::runtime::launch_sigmoid_gate_reference_cuda(
        baseline_output.data(), gate.data(), kQueryElements,
        baseline_output.data(), static_cast<void*>(stream));
  };
  const auto launch_candidate = [&]() {
    return q3x::runtime::launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
            query.data(), key.data(), value.data(), kSequence, kScale,
            candidate_scratch.data(), candidate_scratch.size(), gate.data(),
            candidate_output.data(), static_cast<void*>(stream));
  };

  for (std::size_t iteration = 0U;
       ready && iteration < kWarmupIterations; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " baseline warmup");
    ready = ready &&
            test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                         label + " candidate warmup");
  }
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " warmup sync");
  if (!ready) {
    return;
  }

  double baseline_total = 0.0;
  double candidate_total = 0.0;
  bool timing_finite = true;
  for (int round = 0; round < kMeasurementRounds; ++round) {
    const std::string round_label =
        label + " round=" + std::to_string(round + 1);
    const float baseline_first = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, round_label + " baseline pass 1",
        launch_baseline);
    const float candidate_first = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, round_label + " candidate pass 1",
        launch_candidate);
    const float candidate_second = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, round_label + " candidate pass 2",
        launch_candidate);
    const float baseline_second = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations, round_label + " baseline pass 2",
        launch_baseline);
    const bool round_finite =
        std::isfinite(baseline_first) && baseline_first > 0.0F &&
        std::isfinite(candidate_first) && candidate_first > 0.0F &&
        std::isfinite(candidate_second) && candidate_second > 0.0F &&
        std::isfinite(baseline_second) && baseline_second > 0.0F;
    timing_finite = timing_finite && round_finite;
    if (round_finite) {
      baseline_total += baseline_first + baseline_second;
      candidate_total += candidate_first + candidate_second;
    }
    std::cout << "PERF_GQA_SIGMOID_FUSED_ROUND: round=" << round + 1
              << " iterations=" << kMeasuredIterations
              << " baseline1_ms=" << baseline_first
              << " candidate1_ms=" << candidate_first
              << " candidate2_ms=" << candidate_second
              << " baseline2_ms=" << baseline_second << '\n';
  }

  constexpr double kTimedPasses =
      2.0 * static_cast<double>(kMeasurementRounds);
  const double baseline_milliseconds = baseline_total / kTimedPasses;
  const double candidate_milliseconds = candidate_total / kTimedPasses;
  const double speedup = baseline_milliseconds / candidate_milliseconds;
  const bool gate_passed =
      timing_finite && std::isfinite(speedup) &&
      baseline_milliseconds > 0.0 && candidate_milliseconds > 0.0 &&
      speedup >= kMinimumSpeedup;
  std::cout << "PERF_GQA_SIGMOID_FUSED: baseline_ms="
            << baseline_milliseconds
            << " candidate_ms=" << candidate_milliseconds
            << " speedup=" << speedup
            << " required_speedup=" << kMinimumSpeedup
            << " gate=" << (gate_passed ? "PASS" : "FAIL") << '\n';
  test.expect(gate_passed, label + " clears the 1.05x span gate");
  expect_bf16_bits_equal(test, candidate_output.data(),
                         baseline_output.data(), kQueryElements,
                         label + " repeated gated output");
  test.expect(std::memcmp(candidate_scratch.data(), baseline_scratch.data(),
                          kScratchElements * sizeof(float)) == 0,
              label + " repeated probability scratch is bitwise identical");
}

void test_fused_gqa_sigmoid_gate_warp_positions_perf(
    TestContext& test, cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_GQA_SIGMOID_WARP_POSITIONS_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout
        << "SKIP: fused GQA warp-position performance gate; set "
           "Q3X_RUN_GQA_SIGMOID_WARP_POSITIONS_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kQueryHeads = 24U;
  constexpr std::size_t kKvHeads = 4U;
  constexpr std::size_t kDimension = 256U;
  constexpr std::size_t kFirstSequence = 20U;
  constexpr std::size_t kLastSequence = 44U;
  constexpr std::size_t kSequenceCount =
      kLastSequence - kFirstSequence + 1U;
  constexpr float kScale = 0.0625F;
  constexpr std::size_t kWarmupIterations = 10U;
  constexpr std::size_t kIndividualMeasuredIterations = 80U;
  constexpr std::size_t kChainMeasuredIterations = 20U;
  constexpr std::size_t kMeasurementRounds = 5U;
  constexpr std::array<std::size_t, 3U> kIndividualSequences{
      20U, 32U, 44U};
  constexpr double kRequiredChainSpeedup = 1.10;
  constexpr double kRequiredPerLayerDeltaMs = 0.0125;
  constexpr double kRequiredProjected16LayerDeltaMs = 0.20;
  constexpr std::size_t kQueryElements = kQueryHeads * kDimension;
  constexpr std::size_t kCacheElements =
      kLastSequence * kKvHeads * kDimension;
  constexpr std::size_t kScratchElements =
      kQueryHeads * kLastSequence;
  const std::string label = "fused GQA warp-position Decode S20..44";

  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<std::uint16_t> gate;
  ManagedBuffer<float> shared_scratch;
  ManagedBuffer<std::uint16_t> shared_output;
  bool ready = test.cuda_ok(query.allocate(kQueryElements),
                            label + " allocate query");
  ready = ready && test.cuda_ok(key.allocate(kCacheElements),
                                label + " allocate key");
  ready = ready && test.cuda_ok(value.allocate(kCacheElements),
                                label + " allocate value");
  ready = ready && test.cuda_ok(gate.allocate(kQueryElements),
                                label + " allocate gate");
  ready = ready && test.cuda_ok(shared_scratch.allocate(kScratchElements),
                                label + " allocate shared scratch");
  ready = ready && test.cuda_ok(shared_output.allocate(kQueryElements),
                                label + " allocate shared output");
  if (!ready) {
    return;
  }

  for (std::size_t index = 0U; index < kQueryElements; ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 17U + 5U) % 127U) -
                           63) /
        64.0F);
    gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 23U + 3U) % 97U) -
                           48) /
        16.0F);
  }
  for (std::size_t index = 0U; index < kCacheElements; ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 7U) % 113U) -
                           56) /
        64.0F);
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 31U + 11U) % 109U) -
                           54) /
        32.0F);
  }
  const std::vector<std::uint16_t> query_snapshot(query.data(),
                                                   query.data() + query.size());
  const std::vector<std::uint16_t> key_snapshot(key.data(),
                                                 key.data() + key.size());
  const std::vector<std::uint16_t> value_snapshot(
      value.data(), value.data() + value.size());
  const std::vector<std::uint16_t> gate_snapshot(gate.data(),
                                                  gate.data() + gate.size());

  const auto launch_predecessor = [&](const std::size_t sequence_length) {
    return q3x::runtime::
        launch_gqa_attention_sigmoid_gate_shared_tree_predecessor_24_4_256_test_cuda(
            query.data(), key.data(), value.data(), sequence_length, kScale,
            shared_scratch.data(), shared_scratch.size(), gate.data(),
            shared_output.data(), static_cast<void*>(stream));
  };
  const auto launch_production = [&](const std::size_t sequence_length) {
    return q3x::runtime::launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
        query.data(), key.data(), value.data(), sequence_length, kScale,
        shared_scratch.data(), shared_scratch.size(), gate.data(),
        shared_output.data(), static_cast<void*>(stream));
  };
  const auto median_five = [](std::array<double, kMeasurementRounds> values) {
    std::sort(values.begin(), values.end());
    return values[kMeasurementRounds / 2U];
  };

  bool every_individual_round_nonregressing = true;
  for (const std::size_t sequence_length : kIndividualSequences) {
    const std::string cell =
        label + " individual S=" + std::to_string(sequence_length);
    for (std::size_t iteration = 0U;
         iteration < kWarmupIterations && ready; ++iteration) {
      ready = test.cuda_ok(
          static_cast<cudaError_t>(launch_predecessor(sequence_length)),
          cell + " predecessor warmup");
      ready = ready && test.cuda_ok(
                           static_cast<cudaError_t>(
                               launch_production(sequence_length)),
                           cell + " production warmup");
    }
    ready = ready &&
            test.cuda_ok(cudaStreamSynchronize(stream), cell + " warmup sync");
    if (!ready) {
      return;
    }

    std::array<double, kMeasurementRounds> paired_speedups{};
    std::array<double, kMeasurementRounds> paired_deltas{};
    bool cell_nonregressing = true;
    for (std::size_t round = 0U; round < kMeasurementRounds; ++round) {
      const bool predecessor_first = (round & 1U) == 0U;
      const std::string round_label =
          cell + " round=" + std::to_string(round + 1U);
      double predecessor1 = std::numeric_limits<double>::quiet_NaN();
      double predecessor2 = std::numeric_limits<double>::quiet_NaN();
      double production1 = std::numeric_limits<double>::quiet_NaN();
      double production2 = std::numeric_limits<double>::quiet_NaN();
      const auto measure_predecessor = [&]() {
        return static_cast<double>(measure_cuda_span_milliseconds(
            test, stream, kIndividualMeasuredIterations,
            round_label + " predecessor", [&]() {
              return launch_predecessor(sequence_length);
            }));
      };
      const auto measure_production = [&]() {
        return static_cast<double>(measure_cuda_span_milliseconds(
            test, stream, kIndividualMeasuredIterations,
            round_label + " production", [&]() {
              return launch_production(sequence_length);
            }));
      };
      if (predecessor_first) {
        predecessor1 = measure_predecessor();
        production1 = measure_production();
        production2 = measure_production();
        predecessor2 = measure_predecessor();
      } else {
        production1 = measure_production();
        predecessor1 = measure_predecessor();
        predecessor2 = measure_predecessor();
        production2 = measure_production();
      }
      const double predecessor_pair =
          0.5 * (predecessor1 + predecessor2);
      const double production_pair = 0.5 * (production1 + production2);
      paired_speedups[round] = predecessor_pair / production_pair;
      paired_deltas[round] = predecessor_pair - production_pair;
      const bool round_gate =
          std::isfinite(paired_speedups[round]) &&
          std::isfinite(paired_deltas[round]) &&
          paired_speedups[round] >= 1.0;
      cell_nonregressing = cell_nonregressing && round_gate;
      std::cout << "PERF_GQA_SIGMOID_WARP_POSITIONS_INDIVIDUAL_ROUND: S="
                << sequence_length << " round=" << round + 1U << " order="
                << (predecessor_first ? "B-C-C-B" : "C-B-B-C")
                << " predecessor1_ms=" << predecessor1
                << " production1_ms=" << production1
                << " production2_ms=" << production2
                << " predecessor2_ms=" << predecessor2
                << " paired_speedup=" << paired_speedups[round]
                << " paired_delta_ms=" << paired_deltas[round]
                << " gate=" << (round_gate ? "PASS" : "FAIL") << '\n';
    }
    const double median_speedup = median_five(paired_speedups);
    const double median_delta = median_five(paired_deltas);
    const bool cell_gate =
        cell_nonregressing && std::isfinite(median_speedup) &&
        std::isfinite(median_delta) && median_speedup >= 1.0 &&
        median_delta >= 0.0;
    every_individual_round_nonregressing =
        every_individual_round_nonregressing && cell_gate;
    test.expect(cell_gate, cell + " keeps all five rounds nonregressing");
    std::cout << "PERF_GQA_SIGMOID_WARP_POSITIONS_INDIVIDUAL: S="
              << sequence_length << " paired_median_speedup="
              << median_speedup << " paired_median_delta_ms=" << median_delta
              << " every_round_nonregressing="
              << (cell_nonregressing ? "true" : "false")
              << " gate=" << (cell_gate ? "PASS" : "FAIL") << '\n';
  }

  const auto launch_predecessor_chain = [&]() {
    for (std::size_t sequence_length = kFirstSequence;
         sequence_length <= kLastSequence; ++sequence_length) {
      const int status = launch_predecessor(sequence_length);
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  const auto launch_production_chain = [&]() {
    for (std::size_t sequence_length = kFirstSequence;
         sequence_length <= kLastSequence; ++sequence_length) {
      const int status = launch_production(sequence_length);
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  };
  for (std::size_t iteration = 0U;
       iteration < kWarmupIterations && ready; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_predecessor_chain()),
                         label + " predecessor chain warmup");
    ready = ready &&
            test.cuda_ok(static_cast<cudaError_t>(launch_production_chain()),
                         label + " production chain warmup");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " chain warmup sync");
  if (!ready) {
    return;
  }

  std::array<double, kMeasurementRounds> chain_predecessor_pairs{};
  std::array<double, kMeasurementRounds> chain_production_pairs{};
  std::array<double, kMeasurementRounds> chain_speedups{};
  std::array<double, kMeasurementRounds> chain_deltas{};
  bool every_chain_round_nonregressing = true;
  for (std::size_t round = 0U; round < kMeasurementRounds; ++round) {
    const bool predecessor_first = (round & 1U) == 0U;
    const std::string round_label =
        label + " chain round=" + std::to_string(round + 1U);
    double predecessor1 = std::numeric_limits<double>::quiet_NaN();
    double predecessor2 = std::numeric_limits<double>::quiet_NaN();
    double production1 = std::numeric_limits<double>::quiet_NaN();
    double production2 = std::numeric_limits<double>::quiet_NaN();
    const auto measure_predecessor_chain = [&]() {
      return static_cast<double>(measure_cuda_span_milliseconds(
          test, stream, kChainMeasuredIterations,
          round_label + " predecessor", launch_predecessor_chain));
    };
    const auto measure_production_chain = [&]() {
      return static_cast<double>(measure_cuda_span_milliseconds(
          test, stream, kChainMeasuredIterations,
          round_label + " production", launch_production_chain));
    };
    if (predecessor_first) {
      predecessor1 = measure_predecessor_chain();
      production1 = measure_production_chain();
      production2 = measure_production_chain();
      predecessor2 = measure_predecessor_chain();
    } else {
      production1 = measure_production_chain();
      predecessor1 = measure_predecessor_chain();
      predecessor2 = measure_predecessor_chain();
      production2 = measure_production_chain();
    }
    chain_predecessor_pairs[round] =
        0.5 * (predecessor1 + predecessor2);
    chain_production_pairs[round] = 0.5 * (production1 + production2);
    chain_speedups[round] =
        chain_predecessor_pairs[round] / chain_production_pairs[round];
    chain_deltas[round] =
        chain_predecessor_pairs[round] - chain_production_pairs[round];
    const bool round_gate =
        std::isfinite(chain_speedups[round]) &&
        std::isfinite(chain_deltas[round]) && chain_speedups[round] >= 1.0;
    every_chain_round_nonregressing =
        every_chain_round_nonregressing && round_gate;
    std::cout << "PERF_GQA_SIGMOID_WARP_POSITIONS_CHAIN_ROUND: round="
              << round + 1U << " order="
              << (predecessor_first ? "B-C-C-B" : "C-B-B-C")
              << " chains_per_pass=" << kChainMeasuredIterations
              << " kernels_per_chain=" << kSequenceCount
              << " predecessor1_ms=" << predecessor1
              << " production1_ms=" << production1
              << " production2_ms=" << production2
              << " predecessor2_ms=" << predecessor2
              << " paired_speedup=" << chain_speedups[round]
              << " paired_delta_ms=" << chain_deltas[round]
              << " gate=" << (round_gate ? "PASS" : "FAIL") << '\n';
  }

  const double predecessor_chain_median =
      median_five(chain_predecessor_pairs);
  const double production_chain_median =
      median_five(chain_production_pairs);
  const double paired_median_speedup = median_five(chain_speedups);
  const double paired_median_chain_delta_ms = median_five(chain_deltas);
  const double paired_median_per_layer_delta_ms =
      paired_median_chain_delta_ms / static_cast<double>(kSequenceCount);
  const double projected_16_layer_delta_ms =
      paired_median_per_layer_delta_ms * 16.0;
  const bool gate_passed =
      every_individual_round_nonregressing &&
      every_chain_round_nonregressing &&
      std::isfinite(paired_median_speedup) &&
      paired_median_speedup >= kRequiredChainSpeedup &&
      std::isfinite(paired_median_per_layer_delta_ms) &&
      paired_median_per_layer_delta_ms >= kRequiredPerLayerDeltaMs &&
      std::isfinite(projected_16_layer_delta_ms) &&
      projected_16_layer_delta_ms >= kRequiredProjected16LayerDeltaMs;
  test.expect(gate_passed,
              label + " clears the frozen 0.20-ms/token projection gate");
  std::cout << "PERF_GQA_SIGMOID_WARP_POSITIONS_CHAIN: first_S="
            << kFirstSequence << " last_S=" << kLastSequence
            << " kernels_per_chain=" << kSequenceCount
            << " predecessor_chain_median_ms=" << predecessor_chain_median
            << " production_chain_median_ms=" << production_chain_median
            << " paired_median_speedup=" << paired_median_speedup
            << " paired_median_chain_delta_ms="
            << paired_median_chain_delta_ms
            << " paired_median_per_layer_delta_ms="
            << paired_median_per_layer_delta_ms
            << " projected_16_layer_delta_ms="
            << projected_16_layer_delta_ms
            << " every_individual_round_nonregressing="
            << (every_individual_round_nonregressing ? "true" : "false")
            << " every_chain_round_nonregressing="
            << (every_chain_round_nonregressing ? "true" : "false")
            << " required_speedup=" << kRequiredChainSpeedup
            << " required_per_layer_delta_ms=" << kRequiredPerLayerDeltaMs
            << " required_projected_16_layer_delta_ms="
            << kRequiredProjected16LayerDeltaMs
            << " shared_output_scratch_addresses=true public_route=production "
               "promotion_gate="
            << (gate_passed ? "PASS" : "FAIL") << '\n';
  if (!gate_passed) {
    std::cout << "PERF_GQA_SIGMOID_WARP_POSITIONS_STOP_LOSS: "
                 "reason=first_frozen_process_failed second_process=NOT_RUN "
                 "public_route=production rollback_required=true "
                 "end_to_end=NOT_RUN "
                 "gate=FAIL\n";
  } else {
    std::cout << "PERF_GQA_SIGMOID_WARP_POSITIONS_SELECTION: "
                 "selected=production predecessor=shared_tree "
                 "production=warp_positions gate=PASS\n";
  }
  test.expect(std::memcmp(query.data(), query_snapshot.data(),
                          query.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(key.data(), key_snapshot.data(),
                              key.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(value.data(), value_snapshot.data(),
                              value.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(gate.data(), gate_snapshot.data(),
                              gate.size() * sizeof(std::uint16_t)) == 0,
              label + " preserves every input during timing");
}

constexpr std::size_t kBulkGqaQueryHeads = 24U;
constexpr std::size_t kBulkGqaKvHeads = 4U;
constexpr std::size_t kBulkGqaDimension = 256U;
constexpr std::size_t kBulkGqaQueryElementsPerToken =
    kBulkGqaQueryHeads * kBulkGqaDimension;
constexpr std::size_t kBulkGqaKvElementsPerToken =
    kBulkGqaKvHeads * kBulkGqaDimension;
constexpr float kBulkGqaScale = 0.0625F;
constexpr std::size_t kBulkGqaGuardElements = 16U;
constexpr std::uint16_t kBulkGqaGuard = 0x7fc1U;

struct BulkGqaResources {
  int registers = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks_per_multiprocessor = 0;
};

struct BulkGqaGraphTopology {
  std::size_t node_count = 0U;
  void* function = nullptr;
  dim3 grid{};
  dim3 block{};
  unsigned int dynamic_shared_bytes = 0U;
};

[[nodiscard]] int launch_bulk_gqa_baseline(
    const std::uint16_t* const query,
    const std::uint16_t* const key,
    const std::uint16_t* const value,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    float* const scratch,
    const std::size_t scratch_elements,
    std::uint16_t* const output,
    cudaStream_t stream) {
  for (std::size_t token = 0U; token < token_count; ++token) {
    const int status = q3x::runtime::launch_gqa_attention_reference_cuda(
        query + token * kBulkGqaQueryElementsPerToken, key, value,
        kBulkGqaQueryHeads, kBulkGqaKvHeads,
        first_position + token + 1U, kBulkGqaDimension, kBulkGqaScale,
        scratch, scratch_elements,
        output + token * kBulkGqaQueryElementsPerToken,
        static_cast<void*>(stream));
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return q3x::runtime::launch_sigmoid_gate_reference_cuda(
      output, gate, token_count * kBulkGqaQueryElementsPerToken, output,
      static_cast<void*>(stream));
}

template <typename Launch>
[[nodiscard]] bool capture_bulk_gqa_graph(
    TestContext& test,
    cudaStream_t stream,
    const std::string& label,
    Launch&& launch,
    const bool replay,
    BulkGqaGraphTopology& topology) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return false;
  }
  const cudaError_t launch_status = static_cast<cudaError_t>(launch());
  test.expect(launch_status == cudaSuccess, label + " launch succeeds");
  cudaGraph_t graph = nullptr;
  if (!test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                    label + " end capture")) {
    return false;
  }
  bool ready = true;
  ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr,
                                         &topology.node_count),
                       label + " count nodes") &&
          ready;
  std::vector<cudaGraphNode_t> nodes(topology.node_count);
  std::size_t copied_nodes = topology.node_count;
  if (ready && copied_nodes != 0U) {
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nodes.data(),
                                           &copied_nodes),
                         label + " read nodes") &&
            ready;
  }
  if (ready && copied_nodes == 1U) {
    cudaGraphNodeType type = cudaGraphNodeTypeEmpty;
    ready = test.cuda_ok(cudaGraphNodeGetType(nodes[0], &type),
                         label + " node type") &&
            ready;
    if (ready) {
      test.expect(type == cudaGraphNodeTypeKernel,
                  label + " sole node is a kernel");
    }
    cudaKernelNodeParams parameters{};
    if (ready && type == cudaGraphNodeTypeKernel) {
      ready = test.cuda_ok(cudaGraphKernelNodeGetParams(nodes[0], &parameters),
                           label + " kernel parameters") &&
              ready;
      if (ready) {
        topology.function = parameters.func;
        topology.grid = parameters.gridDim;
        topology.block = parameters.blockDim;
        topology.dynamic_shared_bytes = parameters.sharedMemBytes;
      }
    }
  }
  cudaGraphExec_t executable = nullptr;
  if (ready && replay) {
    ready = test.cuda_ok(
                cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0U),
                label + " instantiate") &&
            ready;
    if (ready) {
      ready = test.cuda_ok(cudaGraphLaunch(executable, stream),
                           label + " replay") &&
              ready;
      ready = test.cuda_ok(cudaStreamSynchronize(stream),
                           label + " replay synchronize") &&
              ready;
    }
  }
  if (executable != nullptr) {
    (void)test.cuda_ok(cudaGraphExecDestroy(executable),
                       label + " destroy executable");
  }
  ready = test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph") &&
          ready;
  return ready;
}

void expect_invalid_bulk_gqa_graph(
    TestContext& test,
    cudaStream_t stream,
    const std::string& label,
    const std::uint16_t* const query,
    const std::uint16_t* const key,
    const std::uint16_t* const value,
    const std::uint16_t* const gate,
    const std::size_t first_position,
    const std::size_t token_count,
    std::uint16_t* const output) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return;
  }
  const cudaError_t status = static_cast<cudaError_t>(
      q3x::runtime::launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
          query, key, value, gate, first_position, token_count, output,
          static_cast<void*>(stream)));
  test.expect(status == cudaErrorInvalidValue,
              label + " returns cudaErrorInvalidValue");
  cudaGraph_t graph = nullptr;
  if (!test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                    label + " end capture")) {
    return;
  }
  std::size_t node_count = 0U;
  if (test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                   label + " count nodes")) {
    test.expect(node_count == 0U,
                label + " rejects before enqueue and captures zero nodes");
  }
  (void)test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph");
}

void expect_invalid_bulk_gqa_compatibility_scale_graph(
    TestContext& test,
    cudaStream_t stream,
    const std::string& label,
    const std::uint16_t* const query,
    const std::uint16_t* const key,
    const std::uint16_t* const value,
    const std::uint16_t* const gate,
    std::uint16_t* const output) {
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          label + " begin capture")) {
    return;
  }
  const cudaError_t status = static_cast<cudaError_t>(q3x::runtime::
      launch_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_test_cuda(
          query, key, value, gate, 0U, 256U, 0.125F, output,
          static_cast<void*>(stream)));
  test.expect(status == cudaErrorInvalidValue,
              label + " returns cudaErrorInvalidValue");
  cudaGraph_t graph = nullptr;
  if (!test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                    label + " end capture")) {
    return;
  }
  std::size_t node_count = 0U;
  if (test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &node_count),
                   label + " count nodes")) {
    test.expect(node_count == 0U,
                label + " rejects before enqueue and captures zero nodes");
  }
  (void)test.cuda_ok(cudaGraphDestroy(graph), label + " destroy graph");
}

void test_bulk_gqa_near_max_graph_contract(TestContext& test,
                                           cudaStream_t stream) {
  constexpr std::uintptr_t kAddressStride = 0x1'0000'0000ULL;
  const auto* const query =
      reinterpret_cast<const std::uint16_t*>(1U * kAddressStride);
  const auto* const key =
      reinterpret_cast<const std::uint16_t*>(2U * kAddressStride);
  const auto* const value =
      reinterpret_cast<const std::uint16_t*>(3U * kAddressStride);
  const auto* const gate =
      reinterpret_cast<const std::uint16_t*>(4U * kAddressStride);
  auto* const output =
      reinterpret_cast<std::uint16_t*>(5U * kAddressStride);
  for (const std::size_t token_count : {256U, 512U}) {
    const std::string label =
        "bulk GQA near-max C" + std::to_string(token_count);
    BulkGqaGraphTopology topology;
    const std::size_t first_position =
        q3x::runtime::kBulkCausalGqaMaximumSequenceLength - token_count;
    const bool ready = capture_bulk_gqa_graph(
        test, stream, label,
        [&]() {
          return q3x::runtime::
              launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
                  query, key, value, gate, first_position, token_count,
                  output, static_cast<void*>(stream));
        },
        false, topology);
    if (!ready) {
      return;
    }
    test.expect(topology.node_count == 1U,
                label + " admits the final legal append as one node");
    test.expect(topology.grid.x == token_count / 2U &&
                    topology.grid.y == 4U && topology.block.x == 192U,
                label + " preserves the fixed bulk topology");
  }
}

struct BulkGqaErrorMetrics {
  double maximum_absolute = 0.0;
  double p99_absolute = 0.0;
  double normalized_rmse = 0.0;
  double cosine = 0.0;
  double p99_relative = 0.0;
  std::size_t relative_samples = 0U;
  std::size_t nonfinite_mismatches = 0U;
};

[[nodiscard]] double percentile_99(std::vector<float>& values) {
  if (values.empty()) {
    return 0.0;
  }
  const std::size_t index = 99U * (values.size() - 1U) / 100U;
  std::nth_element(values.begin(), values.begin() + index, values.end());
  return static_cast<double>(values[index]);
}

[[nodiscard]] BulkGqaErrorMetrics bulk_gqa_error_metrics(
    const std::uint16_t* const candidate,
    const std::uint16_t* const reference,
    const std::size_t count) {
  BulkGqaErrorMetrics metrics;
  std::vector<float> samples;
  samples.reserve(count);
  double squared_error = 0.0;
  double squared_reference = 0.0;
  double squared_candidate = 0.0;
  double dot = 0.0;
  for (std::size_t index = 0U; index < count; ++index) {
    const double expected = static_cast<double>(decode_bf16(reference[index]));
    const double actual = static_cast<double>(decode_bf16(candidate[index]));
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
      const bool same_nan = std::isnan(expected) && std::isnan(actual);
      const bool same_infinity =
          std::isinf(expected) && std::isinf(actual) &&
          std::signbit(expected) == std::signbit(actual);
      if (!same_nan && !same_infinity) {
        ++metrics.nonfinite_mismatches;
      }
      continue;
    }
    const double difference = actual - expected;
    const double absolute = std::fabs(difference);
    metrics.maximum_absolute =
        std::max(metrics.maximum_absolute, absolute);
    samples.push_back(static_cast<float>(absolute));
    squared_error += difference * difference;
    squared_reference += expected * expected;
    squared_candidate += actual * actual;
    dot += actual * expected;
  }
  metrics.p99_absolute = percentile_99(samples);
  metrics.normalized_rmse =
      squared_reference == 0.0
          ? std::sqrt(squared_error)
          : std::sqrt(squared_error / squared_reference);
  metrics.cosine =
      squared_reference == 0.0 || squared_candidate == 0.0
          ? (squared_reference == squared_candidate ? 1.0 : 0.0)
          : dot / std::sqrt(squared_reference * squared_candidate);

  samples.clear();
  for (std::size_t index = 0U; index < count; ++index) {
    const double expected = static_cast<double>(decode_bf16(reference[index]));
    const double actual = static_cast<double>(decode_bf16(candidate[index]));
    if (std::isfinite(expected) && std::isfinite(actual) &&
        std::fabs(expected) >= 1.0 / 32.0) {
      samples.push_back(static_cast<float>(
          std::fabs(actual - expected) / std::fabs(expected)));
    }
  }
  metrics.relative_samples = samples.size();
  metrics.p99_relative = percentile_99(samples);
  return metrics;
}

void run_bulk_gqa_correctness_case(
    TestContext& test,
    cudaStream_t stream,
    const std::size_t first_position,
    const std::size_t token_count,
    const bool graph_contract) {
  const std::string label =
      "bulk GQA C" + std::to_string(token_count) + " P" +
      std::to_string(first_position);
  const std::size_t query_elements =
      token_count * kBulkGqaQueryElementsPerToken;
  const std::size_t future_positions = first_position == 17U ? 3U : 0U;
  const std::size_t cache_elements =
      (first_position + token_count) * kBulkGqaKvElementsPerToken;
  const std::size_t allocated_cache_elements =
      cache_elements + future_positions * kBulkGqaKvElementsPerToken;
  const std::size_t scratch_elements =
      (first_position + token_count) * kBulkGqaQueryHeads;
  const std::size_t guarded_query_elements =
      query_elements + 2U * kBulkGqaGuardElements;
  const std::size_t guarded_cache_elements =
      allocated_cache_elements + 2U * kBulkGqaGuardElements;

  ManagedBuffer<std::uint16_t> query_storage;
  ManagedBuffer<std::uint16_t> key_storage;
  ManagedBuffer<std::uint16_t> value_storage;
  ManagedBuffer<std::uint16_t> gate_storage;
  ManagedBuffer<std::uint16_t> baseline_storage;
  ManagedBuffer<std::uint16_t> candidate_storage;
  ManagedBuffer<std::uint16_t> replay_storage;
  ManagedBuffer<float> scratch;
  bool ready = test.cuda_ok(query_storage.allocate(guarded_query_elements),
                            label + " allocate query");
  ready = ready && test.cuda_ok(key_storage.allocate(guarded_cache_elements),
                                label + " allocate key");
  ready = ready && test.cuda_ok(value_storage.allocate(guarded_cache_elements),
                                label + " allocate value");
  ready = ready && test.cuda_ok(gate_storage.allocate(guarded_query_elements),
                                label + " allocate gate");
  ready = ready &&
          test.cuda_ok(baseline_storage.allocate(guarded_query_elements),
                       label + " allocate baseline");
  ready = ready &&
          test.cuda_ok(candidate_storage.allocate(guarded_query_elements),
                       label + " allocate candidate");
  ready = ready && test.cuda_ok(replay_storage.allocate(guarded_query_elements),
                                label + " allocate replay");
  ready = ready && test.cuda_ok(scratch.allocate(scratch_elements),
                                label + " allocate scratch");
  if (!ready) {
    return;
  }

  std::uint16_t* const query =
      query_storage.data() + kBulkGqaGuardElements;
  std::uint16_t* const key =
      key_storage.data() + kBulkGqaGuardElements;
  std::uint16_t* const value =
      value_storage.data() + kBulkGqaGuardElements;
  std::uint16_t* const gate =
      gate_storage.data() + kBulkGqaGuardElements;
  std::uint16_t* const baseline =
      baseline_storage.data() + kBulkGqaGuardElements;
  std::uint16_t* const candidate =
      candidate_storage.data() + kBulkGqaGuardElements;
  std::uint16_t* const replay =
      replay_storage.data() + kBulkGqaGuardElements;
  std::fill_n(query_storage.data(), query_storage.size(), kBulkGqaGuard);
  std::fill_n(key_storage.data(), key_storage.size(), kBulkGqaGuard);
  std::fill_n(value_storage.data(), value_storage.size(), kBulkGqaGuard);
  std::fill_n(gate_storage.data(), gate_storage.size(), kBulkGqaGuard);
  std::fill_n(baseline_storage.data(), baseline_storage.size(),
              kBulkGqaGuard);
  std::fill_n(candidate_storage.data(), candidate_storage.size(),
              kBulkGqaGuard);
  std::fill_n(replay_storage.data(), replay_storage.size(), kBulkGqaGuard);
  for (std::size_t index = 0U; index < query_elements; ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 17U + 5U) % 127U) -
                           63) /
        64.0F);
    gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 23U + 3U) % 97U) -
                           48) /
        16.0F);
  }
  for (std::size_t index = 0U; index < cache_elements; ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 7U) % 113U) -
                           56) /
        64.0F);
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 31U + 11U) % 109U) -
                           54) /
        32.0F);
  }
  std::fill_n(key + cache_elements,
              allocated_cache_elements - cache_elements, 0x7fc1U);
  std::fill_n(value + cache_elements,
              allocated_cache_elements - cache_elements, 0xff80U);
  const std::vector<std::uint16_t> query_snapshot(
      query_storage.data(), query_storage.data() + query_storage.size());
  const std::vector<std::uint16_t> key_snapshot(
      key_storage.data(), key_storage.data() + key_storage.size());
  const std::vector<std::uint16_t> value_snapshot(
      value_storage.data(), value_storage.data() + value_storage.size());
  const std::vector<std::uint16_t> gate_snapshot(
      gate_storage.data(), gate_storage.data() + gate_storage.size());

  ready = test.cuda_ok(
      static_cast<cudaError_t>(launch_bulk_gqa_baseline(
          query, key, value, gate, first_position, token_count,
          scratch.data(), scratch.size(), baseline, stream)),
      label + " baseline launch");
  ready = ready && test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::runtime::launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
              query, key, value, gate, first_position, token_count, candidate,
              static_cast<void*>(stream))),
      label + " production launch");
  ready = ready && test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          launch_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_test_cuda(
              query, key, value, gate, first_position, token_count,
              kBulkGqaScale, replay, static_cast<void*>(stream))),
      label + " compatibility wrapper launch");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (!ready) {
    return;
  }
  test.expect(std::memcmp(replay, candidate,
                          query_elements * sizeof(std::uint16_t)) == 0,
              label + " compatibility wrapper matches production API");

  BulkGqaGraphTopology candidate_topology;
  ready = capture_bulk_gqa_graph(
      test, stream, label + " candidate graph",
      [&]() {
        return q3x::runtime::
            launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
                query, key, value, gate, first_position, token_count, replay,
                static_cast<void*>(stream));
      },
      true, candidate_topology);
  if (!ready) {
    return;
  }
  test.expect(candidate_topology.node_count == 1U,
              label + " candidate graph has one node");
  test.expect(candidate_topology.function != nullptr,
              label + " candidate graph has a kernel identity");
  const bool tensor_core_c512 =
      first_position == 0U && token_count == 512U;
  const bool expected_topology =
      tensor_core_c512
          ? candidate_topology.grid.x ==
                    token_count * 6U / 64U &&
                candidate_topology.grid.y == 1U &&
                candidate_topology.grid.z == 4U &&
                candidate_topology.block.x == 128U &&
                candidate_topology.block.y == 1U &&
                candidate_topology.block.z == 1U &&
                candidate_topology.dynamic_shared_bytes == 64U * 1024U
          : candidate_topology.grid.x == (token_count + 1U) / 2U &&
                candidate_topology.grid.y == 4U &&
                candidate_topology.grid.z == 1U &&
                candidate_topology.block.x == 192U &&
                candidate_topology.block.y == 1U &&
                candidate_topology.block.z == 1U &&
                candidate_topology.dynamic_shared_bytes == 0U;
  test.expect(expected_topology,
              label +
                  (tensor_core_c512
                       ? " candidate graph has fixed grouped-Q64 Tensor Core topology"
                       : " candidate graph has fixed QT2 topology"));

  if (graph_contract) {
    BulkGqaGraphTopology baseline_topology;
    ready = capture_bulk_gqa_graph(
        test, stream, label + " baseline graph",
        [&]() {
          return launch_bulk_gqa_baseline(
              query, key, value, gate, first_position, token_count,
              scratch.data(), scratch.size(), baseline, stream);
        },
        false, baseline_topology);
    if (!ready) {
      return;
    }
    test.expect(baseline_topology.node_count == 3U * token_count + 1U,
                label + " baseline graph has three attention nodes per token plus Gate");
  }

  const BulkGqaErrorMetrics metrics =
      bulk_gqa_error_metrics(candidate, baseline, query_elements);
  const bool numerical_gate =
      metrics.nonfinite_mismatches == 0U &&
      metrics.maximum_absolute <= 0.03125 &&
      metrics.p99_absolute <= 0.00390625 &&
      metrics.normalized_rmse <= 0.005 && metrics.cosine >= 0.9999 &&
      metrics.relative_samples != 0U && metrics.p99_relative <= 0.02;
  test.expect(numerical_gate, label + " clears online-softmax error gates");
  test.expect(std::memcmp(replay, candidate,
                          query_elements * sizeof(std::uint16_t)) == 0,
              label + " candidate Graph replay is bitwise deterministic");
  const auto guards_intact = [](const ManagedBuffer<std::uint16_t>& storage,
                                const std::size_t payload_elements) {
    return std::all_of(storage.data(),
                       storage.data() + kBulkGqaGuardElements,
                       [](const std::uint16_t value_bits) {
                         return value_bits == kBulkGqaGuard;
                       }) &&
           std::all_of(storage.data() + kBulkGqaGuardElements +
                           payload_elements,
                       storage.data() + storage.size(),
                       [](const std::uint16_t value_bits) {
                         return value_bits == kBulkGqaGuard;
                       });
  };
  test.expect(guards_intact(baseline_storage, query_elements),
              label + " baseline guards are intact");
  test.expect(guards_intact(candidate_storage, query_elements),
              label + " candidate guards are intact");
  test.expect(guards_intact(replay_storage, query_elements),
              label + " replay guards are intact");
  test.expect(std::memcmp(query_storage.data(), query_snapshot.data(),
                          query_storage.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(key_storage.data(), key_snapshot.data(),
                              key_storage.size() * sizeof(std::uint16_t)) ==
                      0 &&
                  std::memcmp(value_storage.data(), value_snapshot.data(),
                              value_storage.size() * sizeof(std::uint16_t)) ==
                      0 &&
                  std::memcmp(gate_storage.data(), gate_snapshot.data(),
                              gate_storage.size() * sizeof(std::uint16_t)) ==
                      0,
              label + " preserves query, K/V cache, Gate, and input guards");
  std::cout << "BULK_GQA_CORRECTNESS: C=" << token_count
            << " first_position=" << first_position
            << " max_abs=" << metrics.maximum_absolute
            << " p99_abs=" << metrics.p99_absolute
            << " nrmse=" << metrics.normalized_rmse
            << " cosine=" << metrics.cosine
            << " p99_relative=" << metrics.p99_relative
            << " relative_samples=" << metrics.relative_samples
            << " nonfinite_mismatches=" << metrics.nonfinite_mismatches
            << " baseline_nodes=" << (3U * token_count + 1U)
            << " candidate_nodes=" << candidate_topology.node_count
            << " future_poison_positions=" << future_positions
            << " replay=bitwise gate="
            << (numerical_gate ? "PASS" : "FAIL") << '\n';

  if (first_position == 0U && token_count == 256U) {
    const int invalid_failures_before = test.failures();
    const std::string invalid = "bulk GQA invalid graph ";
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "C1", query, key, value, gate, 0U, 1U,
        candidate);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "C513", query, key, value, gate, 0U, 513U,
        candidate);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "sequence overflow", query, key, value, gate,
        q3x::runtime::kBulkCausalGqaMaximumSequenceLength - 255U, 256U,
        candidate);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "null query", nullptr, key, value, gate, 0U,
        256U, candidate);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "null key", query, nullptr, value, gate, 0U,
        256U, candidate);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "null value", query, key, nullptr, gate, 0U,
        256U, candidate);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "null gate", query, key, value, nullptr, 0U,
        256U, candidate);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "null output", query, key, value, gate, 0U,
        256U, nullptr);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "misaligned query", query + 1U, key, value,
        gate, 0U, 256U, candidate);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "query-output alias", query, key, value,
        gate, 0U, 256U, query);
    expect_invalid_bulk_gqa_graph(
        test, stream, invalid + "key-value alias", query, key, key, gate, 0U,
        256U, candidate);
    expect_invalid_bulk_gqa_compatibility_scale_graph(
        test, stream, invalid + "compatibility wrong scale", query, key,
        value, gate, candidate);
    std::cout << "BULK_GQA_INVALID_GRAPH: production_cases=11 "
                 "compatibility_cases=1 "
                 "zero_node_contract=true "
                 "gate="
              << (test.failures() == invalid_failures_before ? "PASS"
                                                              : "FAIL")
              << '\n';
  }
}

void test_bulk_causal_gqa_prefill_contract(TestContext& test,
                                           cudaStream_t stream) {
  BulkGqaResources resources;
  for (std::size_t null_index = 0U; null_index < 5U; ++null_index) {
    const cudaError_t status = static_cast<cudaError_t>(
        q3x::runtime::
            query_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_resources_test_cuda(
                null_index == 0U ? nullptr : &resources.registers,
                null_index == 1U ? nullptr : &resources.static_shared_bytes,
                null_index == 2U ? nullptr : &resources.local_bytes,
                null_index == 3U ? nullptr : &resources.maximum_threads,
                null_index == 4U
                    ? nullptr
                    : &resources.active_blocks_per_multiprocessor));
    test.expect(status == cudaErrorInvalidValue,
                "bulk GQA resource query rejects null output " +
                    std::to_string(null_index));
  }
  const bool resources_ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::runtime::
          query_bulk_causal_gqa_sigmoid_24_4_256_qt2_bk16_resources_test_cuda(
              &resources.registers, &resources.static_shared_bytes,
              &resources.local_bytes, &resources.maximum_threads,
              &resources.active_blocks_per_multiprocessor)),
      "bulk GQA resource query");
  if (!resources_ready) {
    return;
  }
  const bool resource_gate =
      resources.static_shared_bytes <= 32U * 1024U &&
      resources.local_bytes == 0U && resources.maximum_threads >= 192 &&
      resources.active_blocks_per_multiprocessor >= 3;
  test.expect(resources.static_shared_bytes == 16U * 1024U,
              "bulk GQA QT2/BK16 uses exactly 16 KiB static shared memory");
  test.expect(resources.local_bytes == 0U,
              "bulk GQA has no local-memory spills");
  test.expect(resources.maximum_threads >= 192,
              "bulk GQA supports its 192-thread block");
  test.expect(resources.active_blocks_per_multiprocessor >= 3,
              "bulk GQA keeps at least three CTAs resident per SM");
  std::cout << "BULK_GQA_RESOURCES: registers=" << resources.registers
            << " static_shared=" << resources.static_shared_bytes
            << " local=" << resources.local_bytes
            << " max_threads=" << resources.maximum_threads
            << " active_blocks="
            << resources.active_blocks_per_multiprocessor
            << " gate=" << (resource_gate ? "PASS" : "FAIL") << '\n';

  test_bulk_gqa_near_max_graph_contract(test, stream);
  run_bulk_gqa_correctness_case(test, stream, 0U, 256U, true);
  run_bulk_gqa_correctness_case(test, stream, 0U, 512U, true);
  run_bulk_gqa_correctness_case(test, stream, 0U, 407U, true);
  run_bulk_gqa_correctness_case(test, stream, 0U, 481U, true);
  run_bulk_gqa_correctness_case(test, stream, 17U, 256U, false);
}

void run_bulk_gqa_perf_case(TestContext& test,
                            cudaStream_t stream,
                            const std::size_t token_count) {
  constexpr std::size_t kWarmupIterations = 2U;
  constexpr std::size_t kMeasuredIterations = 2U;
  constexpr std::size_t kRounds = 6U;
  const std::string label =
      "bulk GQA Prefill perf C" + std::to_string(token_count);
  const std::size_t query_elements =
      token_count * kBulkGqaQueryElementsPerToken;
  const std::size_t cache_elements =
      token_count * kBulkGqaKvElementsPerToken;
  const std::size_t scratch_elements =
      token_count * kBulkGqaQueryHeads;
  ManagedBuffer<std::uint16_t> query;
  ManagedBuffer<std::uint16_t> key;
  ManagedBuffer<std::uint16_t> value;
  ManagedBuffer<std::uint16_t> gate;
  ManagedBuffer<std::uint16_t> baseline_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  ManagedBuffer<float> scratch;
  bool ready = test.cuda_ok(query.allocate(query_elements),
                            label + " allocate query");
  ready = ready && test.cuda_ok(key.allocate(cache_elements),
                                label + " allocate key");
  ready = ready && test.cuda_ok(value.allocate(cache_elements),
                                label + " allocate value");
  ready = ready && test.cuda_ok(gate.allocate(query_elements),
                                label + " allocate gate");
  ready = ready && test.cuda_ok(baseline_output.allocate(query_elements),
                                label + " allocate baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(query_elements),
                                label + " allocate candidate output");
  ready = ready && test.cuda_ok(scratch.allocate(scratch_elements),
                                label + " allocate scratch");
  if (!ready) {
    return;
  }
  for (std::size_t index = 0U; index < query_elements; ++index) {
    query[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 17U + 5U) % 127U) -
                           63) /
        64.0F);
    gate[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 23U + 3U) % 97U) -
                           48) /
        16.0F);
  }
  for (std::size_t index = 0U; index < cache_elements; ++index) {
    key[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 29U + 7U) % 113U) -
                           56) /
        64.0F);
    value[index] = encode_bf16(
        static_cast<float>(static_cast<int>((index * 31U + 11U) % 109U) -
                           54) /
        32.0F);
  }
  const std::vector<std::uint16_t> query_snapshot(
      query.data(), query.data() + query.size());
  const std::vector<std::uint16_t> key_snapshot(
      key.data(), key.data() + key.size());
  const std::vector<std::uint16_t> value_snapshot(
      value.data(), value.data() + value.size());
  const std::vector<std::uint16_t> gate_snapshot(
      gate.data(), gate.data() + gate.size());

  const auto launch_baseline = [&]() {
    return launch_bulk_gqa_baseline(
        query.data(), key.data(), value.data(), gate.data(), 0U, token_count,
        scratch.data(), scratch.size(), baseline_output.data(), stream);
  };
  const auto launch_candidate = [&]() {
    return q3x::runtime::
        launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
            query.data(), key.data(), value.data(), gate.data(), 0U,
            token_count, candidate_output.data(), static_cast<void*>(stream));
  };
  for (std::size_t iteration = 0U;
       iteration < kWarmupIterations && ready; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " baseline warmup");
    ready = ready &&
            test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                         label + " candidate warmup");
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                label + " warmup synchronize");
  if (!ready) {
    return;
  }

  std::array<double, kRounds> baseline_pairs{};
  std::array<double, kRounds> candidate_pairs{};
  std::array<double, kRounds> speedups{};
  bool every_round_nonregressing = true;
  for (std::size_t round = 0U; round < kRounds; ++round) {
    const bool baseline_first = (round & 1U) == 0U;
    const std::string round_label =
        label + " round=" + std::to_string(round + 1U);
    double baseline1 = std::numeric_limits<double>::quiet_NaN();
    double baseline2 = std::numeric_limits<double>::quiet_NaN();
    double candidate1 = std::numeric_limits<double>::quiet_NaN();
    double candidate2 = std::numeric_limits<double>::quiet_NaN();
    const auto measure_baseline = [&]() {
      return static_cast<double>(measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " baseline",
          launch_baseline));
    };
    const auto measure_candidate = [&]() {
      return static_cast<double>(measure_cuda_span_milliseconds(
          test, stream, kMeasuredIterations, round_label + " candidate",
          launch_candidate));
    };
    if (baseline_first) {
      baseline1 = measure_baseline();
      candidate1 = measure_candidate();
      candidate2 = measure_candidate();
      baseline2 = measure_baseline();
    } else {
      candidate1 = measure_candidate();
      baseline1 = measure_baseline();
      baseline2 = measure_baseline();
      candidate2 = measure_candidate();
    }
    baseline_pairs[round] = 0.5 * (baseline1 + baseline2);
    candidate_pairs[round] = 0.5 * (candidate1 + candidate2);
    speedups[round] = baseline_pairs[round] / candidate_pairs[round];
    const bool round_gate =
        std::isfinite(baseline_pairs[round]) &&
        baseline_pairs[round] > 0.0 &&
        std::isfinite(candidate_pairs[round]) &&
        candidate_pairs[round] > 0.0 && std::isfinite(speedups[round]) &&
        speedups[round] >= 1.0;
    every_round_nonregressing = every_round_nonregressing && round_gate;
    std::cout << "PERF_BULK_GQA_ROUND: C=" << token_count
              << " round=" << round + 1U << " order="
              << (baseline_first ? "B-C-C-B" : "C-B-B-C")
              << " iterations=" << kMeasuredIterations
              << " baseline1_ms=" << baseline1
              << " candidate1_ms=" << candidate1
              << " candidate2_ms=" << candidate2
              << " baseline2_ms=" << baseline2
              << " paired_speedup=" << speedups[round]
              << " gate=" << (round_gate ? "PASS" : "FAIL") << '\n';
  }
  std::array<double, kRounds> sorted_speedups = speedups;
  std::sort(sorted_speedups.begin(), sorted_speedups.end());
  const double median_speedup =
      0.5 * (sorted_speedups[kRounds / 2U - 1U] +
             sorted_speedups[kRounds / 2U]);
  const bool speed_gate =
      every_round_nonregressing && std::isfinite(median_speedup) &&
      (token_count != 512U || median_speedup >= 2.0);
  test.expect(every_round_nonregressing,
              label + " keeps every mirrored round nonregressing");
  if (token_count == 512U) {
    test.expect(median_speedup >= 2.0,
                label + " clears the 2.0x C512 gate");
  }
  test.expect(std::memcmp(query.data(), query_snapshot.data(),
                          query.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(key.data(), key_snapshot.data(),
                              key.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(value.data(), value_snapshot.data(),
                              value.size() * sizeof(std::uint16_t)) == 0 &&
                  std::memcmp(gate.data(), gate_snapshot.data(),
                              gate.size() * sizeof(std::uint16_t)) == 0,
              label + " timing preserves all inputs");
  std::cout << "PERF_BULK_GQA: C=" << token_count
            << " paired_median_speedup=" << median_speedup
            << " every_round_nonregressing="
            << (every_round_nonregressing ? "true" : "false")
            << " required_C512_speedup="
            << (token_count == 512U ? 2.0 : 1.0)
            << " baseline_nodes=" << (3U * token_count + 1U)
            << " candidate_nodes=1 gate="
            << (speed_gate ? "PASS" : "FAIL") << '\n';
}

void test_bulk_causal_gqa_prefill_perf(TestContext& test,
                                       cudaStream_t stream) {
  const char* const enabled =
      std::getenv("Q3X_RUN_BULK_CAUSAL_GQA_PREFILL_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: bulk causal GQA Prefill performance gate; set "
                 "Q3X_RUN_BULK_CAUSAL_GQA_PREFILL_PERF=1 to enable\n";
    return;
  }
  run_bulk_gqa_perf_case(test, stream, 256U);
  run_bulk_gqa_perf_case(test, stream, 512U);
}

void test_nonfinite(TestContext& test, cudaStream_t stream) {
  ManagedBuffer<std::uint16_t> left;
  ManagedBuffer<std::uint16_t> right;
  ManagedBuffer<std::uint16_t> output;
  ManagedBuffer<float> softmax;
  bool ready = test.cuda_ok(left.allocate(3U), "nonfinite allocate left");
  ready = ready &&
          test.cuda_ok(right.allocate(3U), "nonfinite allocate right");
  ready = ready &&
          test.cuda_ok(output.allocate(3U), "nonfinite allocate output");
  ready = ready &&
          test.cuda_ok(softmax.allocate(3U), "nonfinite allocate softmax");
  if (!ready) {
    return;
  }
  left[0] = encode_bf16(std::numeric_limits<float>::infinity());
  left[1] = encode_bf16(std::numeric_limits<float>::quiet_NaN());
  left[2] = encode_bf16(1.0F);
  std::fill_n(right.data(), right.size(), encode_bf16(1.0F));
  ready = launch_after_stale(test, stream, "nonfinite residual", [&]() {
    return q3x::runtime::launch_residual_add_reference_cuda(
        left.data(), right.data(), left.size(), output.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    test.expect(std::isinf(decode_bf16(output[0])),
                "CUDA residual propagates infinity");
    test.expect(std::isnan(decode_bf16(output[1])),
                "CUDA residual propagates NaN");
  }
  softmax[0] = 0.0F;
  softmax[1] = std::numeric_limits<float>::quiet_NaN();
  softmax[2] = 1.0F;
  ready = launch_after_stale(test, stream, "nonfinite softmax", [&]() {
    return q3x::runtime::launch_softmax_reference_cuda(
        softmax.data(), 1U, softmax.size(), softmax.data(),
        static_cast<void*>(stream));
  });
  if (ready) {
    test.expect(std::all_of(softmax.data(), softmax.data() + softmax.size(),
                            [](const float value) {
                              return std::isnan(value);
                            }),
                "CUDA softmax propagates NaN across row");
  }
}

}  // namespace

int main() {
  TestContext test;
  test_launch_validation(test);
  test_residual_rms_m32_launch_validation(test);

  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA decode-op tests (no CUDA device)\n";
    (void)cudaGetLastError();
    return test.failures() == 0 ? 0 : 1;
  }
  cudaDeviceProp properties{};
  if (test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                   "read CUDA device properties")) {
    test.expect(properties.major == 8 && properties.minor == 7,
                "decode ops run on required SM87 device");
    std::cout << "CUDA decode-op device: " << properties.name << " (sm_"
              << properties.major << properties.minor << ")\n";
  }
  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create decode-op stream")) {
    return 1;
  }

  test_embedding(test, stream);
  test_vector_ops(test, stream);
  test_residual_rms_fused(test, stream);
  test_residual_rms_fused_perf(test, stream);
  test_residual_rms_m32_fused_exact(test, stream);
  test_residual_rms_m32_prefix_tail_exact(test, stream);
  test_residual_rms_m32_nan_operand_order(test, stream);
  test_prefill_m32_residual_rms_formal_perf(test, stream);
  test_outer_rms_tile_exact(test, stream);
  test_fp32_conversion(test, stream);
  test_bf16_greedy_argmax(test, stream);
  run_headwise_norm_case(test, stream, 24U,
                         q3x::runtime::kFullAttentionHeadDimension,
                         HeadwiseNormKind::kCentered,
                         "full-attention Q centered norm 24x256");
  run_headwise_norm_case(test, stream, 4U,
                         q3x::runtime::kFullAttentionHeadDimension,
                         HeadwiseNormKind::kCentered,
                         "full-attention K centered norm 4x256");
  run_headwise_norm_case(test, stream, 48U,
                         q3x::runtime::kLinearAttentionHeadDimension,
                         HeadwiseNormKind::kPlain,
                         "GDN output plain norm 48x128");
  run_headwise_norm_case(test, stream, 48U,
                         q3x::runtime::kLinearAttentionHeadDimension,
                         HeadwiseNormKind::kPlainSiluGate,
                         "GDN fused plain norm SiLU gate 48x128");
  run_l2_case(test, stream, 5U, 127U, "L2 awkward 5x127");
  run_l2_case(test, stream, 16U,
              q3x::runtime::kLinearAttentionHeadDimension,
              "GDN L2 target 16x128");
  test_full_attention_preproc_fusion_exact(test, stream);
  test_full_attention_preproc_warp_rms_contract(test, stream);
  test_full_attention_preproc_rope_fusion_perf(test, stream);
  test_full_attention_preproc_warp_rms_perf(test, stream);
  test_rope(test, stream);
  test_qk_rope_tile_exact(test, stream);
  test_qk_rope_tile_perf(test, stream);
  test_softmax(test, stream);
  test_attention(test, stream);
  test_attention_scores_warp_positions_exact(test, stream);
  test_attention_scores_warp_positions_graph_contract(test, stream);
  test_attention_scores_warp_positions_perf(test, stream);
  test_attention_values_exact(test, stream);
  test_attention_values_exact_graph_contract(test, stream);
  test_attention_values_exact_perf(test, stream);
  test_bulk_causal_gqa_prefill_contract(test, stream);
  test_bulk_causal_gqa_prefill_perf(test, stream);
  test_fused_gqa_sigmoid_gate_warp_positions_contract(test, stream);
  test_fused_gqa_sigmoid_gate_exact(test, stream);
  test_fused_gqa_sigmoid_gate_perf(test, stream);
  test_fused_gqa_sigmoid_gate_warp_positions_perf(test, stream);
  test_nonfinite(test, stream);

  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy decode-op stream");
  if (test.failures() != 0) {
    std::cerr << test.failures() << " CUDA decode-op assertion(s) failed\n";
    return 1;
  }
  std::cout << "CUDA decode-op reference tests passed\n";
  return 0;
}
