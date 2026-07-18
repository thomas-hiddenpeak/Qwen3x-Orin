#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
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
