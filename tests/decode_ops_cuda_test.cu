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
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

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

  const bool baseline_ready = launch_after_stale(
      test, stream, label + " baseline four launches", launch_baseline);
  const bool candidate_ready = launch_after_stale(
      test, stream, label + " candidate one launch", launch_candidate);
  if (baseline_ready && candidate_ready) {
    expect_bf16_bits_equal(test, candidate_query.data(),
                           baseline_query.data(), query_elements,
                           label + " normalized and rotated query");
    expect_bf16_bits_equal(test, candidate_key.data(), baseline_key.data(),
                           key_elements,
                           label + " in-place normalized and rotated key");
    expect_bf16_bits_equal(test, candidate_gate.data(), baseline_gate.data(),
                           query_elements, label + " raw packed gate");
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
  test.expect(std::memcmp(cosines.data(), original_cosines.data(),
                          table_elements * sizeof(float)) == 0,
              label + " cosine table is unchanged");
  test.expect(std::memcmp(sines.data(), original_sines.data(),
                          table_elements * sizeof(float)) == 0,
              label + " sine table is unchanged");
}

void test_full_attention_preproc_fusion_exact(TestContext& test,
                                              cudaStream_t stream) {
  run_full_attention_preproc_fusion_exact_case(
      test, stream, 1U, FullAttentionPreprocFixture::kFinite);
  run_full_attention_preproc_fusion_exact_case(
      test, stream, 2U, FullAttentionPreprocFixture::kNonfinite);
  run_full_attention_preproc_fusion_exact_case(
      test, stream, 8U, FullAttentionPreprocFixture::kFinite);
  run_full_attention_preproc_fusion_exact_case(
      test, stream, 16U, FullAttentionPreprocFixture::kFinite);
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
  ManagedBuffer<float> candidate_scratch;
  ManagedBuffer<std::uint16_t> baseline_output;
  ManagedBuffer<std::uint16_t> candidate_output;
  bool ready = test.cuda_ok(query.allocate(query_elements), label + " query");
  ready = ready && test.cuda_ok(key.allocate(cache_elements), label + " key");
  ready = ready &&
          test.cuda_ok(value.allocate(cache_elements), label + " value");
  ready = ready && test.cuda_ok(gate.allocate(query_elements), label + " gate");
  ready = ready && test.cuda_ok(baseline_scratch.allocate(scratch_elements),
                                label + " baseline scratch");
  ready = ready && test.cuda_ok(candidate_scratch.allocate(scratch_elements),
                                label + " candidate scratch");
  ready = ready && test.cuda_ok(baseline_output.allocate(query_elements),
                                label + " baseline output");
  ready = ready && test.cuda_ok(candidate_output.allocate(query_elements),
                                label + " candidate output");
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
  std::fill_n(candidate_scratch.data(), scratch_elements,
              std::numeric_limits<float>::quiet_NaN());
  std::fill_n(baseline_output.data(), query_elements, kPoison);
  std::fill_n(candidate_output.data(), query_elements, kPoison);

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
          launch_after_stale(test, stream, label + " candidate", [&]() {
            return q3x::runtime::
                launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                    query.data(), key.data(), value.data(), sequence_length,
                    kScale, candidate_scratch.data(),
                    candidate_scratch.size(), gate.data(),
                    candidate_output.data(), static_cast<void*>(stream));
          });
  if (!ready) {
    return;
  }

  expect_bf16_bits_equal(test, candidate_output.data(),
                         baseline_output.data(), query_elements,
                         label + " gated output");
  test.expect(std::memcmp(candidate_scratch.data(), baseline_scratch.data(),
                          scratch_elements * sizeof(float)) == 0,
              label + " probability scratch is bitwise identical");
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
  for (const std::size_t sequence_length : {1U, 16U, 44U, 64U}) {
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
  test_full_attention_preproc_rope_fusion_perf(test, stream);
  test_rope(test, stream);
  test_qk_rope_tile_exact(test, stream);
  test_qk_rope_tile_perf(test, stream);
  test_softmax(test, stream);
  test_attention(test, stream);
  test_fused_gqa_sigmoid_gate_exact(test, stream);
  test_fused_gqa_sigmoid_gate_perf(test, stream);
  test_nonfinite(test, stream);

  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy decode-op stream");
  if (test.failures() != 0) {
    std::cerr << test.failures() << " CUDA decode-op assertion(s) failed\n";
    return 1;
  }
  std::cout << "CUDA decode-op reference tests passed\n";
  return 0;
}
