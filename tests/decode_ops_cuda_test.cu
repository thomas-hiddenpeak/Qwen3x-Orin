#include "q3x/runtime/decode_ops.h"

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
                                          mismatch.first - actual))));
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
  const std::uint16_t* const fused_left = fused_validation_storage.data();
  const std::uint16_t* const fused_right =
      fused_left + kFusedHiddenSize;
  const std::uint16_t* const fused_weight =
      fused_right + kFusedHiddenSize;
  std::uint16_t* const fused_residual =
      fused_validation_storage.data() + 3U * kFusedHiddenSize;
  std::uint16_t* const fused_normalized =
      fused_validation_storage.data() + 4U * kFusedHiddenSize;
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
  test_rope(test, stream);
  test_softmax(test, stream);
  test_attention(test, stream);
  test_nonfinite(test, stream);

  (void)test.cuda_ok(cudaStreamDestroy(stream), "destroy decode-op stream");
  if (test.failures() != 0) {
    std::cerr << test.failures() << " CUDA decode-op assertion(s) failed\n";
    return 1;
  }
  std::cout << "CUDA decode-op reference tests passed\n";
  return 0;
}
