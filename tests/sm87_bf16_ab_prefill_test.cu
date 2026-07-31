#include "q3x/kernels/reference_gemv.h"
#include "q3x/kernels/sm87_bf16_ab_prefill.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;

constexpr std::size_t kRows = 48U;
constexpr std::size_t kColumns = 5'120U;
constexpr std::size_t kMaximumTestTokens = 513U;
constexpr std::size_t kGuardElements = 16U;
constexpr std::size_t kSparseEntries = 8U;
constexpr std::uint16_t kGuardValue = 0xa5a5U;

static_assert(kernels::kSm87Bf16AbLargeMPrefillMaximumTokens == 4'096U);

class TestContext final {
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

  [[nodiscard]] bool launch_ok(const int status,
                               const std::string& operation) {
    return cuda_ok(static_cast<cudaError_t>(status), operation);
  }

  void expect_invalid(const int status, const std::string& operation) {
    expect(status == static_cast<int>(cudaErrorInvalidValue),
           operation + " returns cudaErrorInvalidValue (actual=" +
               std::to_string(status) + ")");
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t elements) {
    return cudaMalloc(reinterpret_cast<void**>(&data_),
                      elements * sizeof(T));
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
};

class StreamOwner final {
 public:
  StreamOwner() = default;
  StreamOwner(const StreamOwner&) = delete;
  StreamOwner& operator=(const StreamOwner&) = delete;

  ~StreamOwner() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] cudaError_t create() {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits, sizeof(result));
  return result;
}

[[nodiscard]] int activation_integer(const std::size_t token,
                                     const std::size_t column) noexcept {
  return static_cast<int>((17U * token + 13U * column + 3U) % 7U) - 3;
}

[[nodiscard]] std::size_t first_weight_column(
    const std::size_t row, const std::size_t entry) noexcept {
  return (73U * row + 613U * entry + 7U) % kColumns;
}

[[nodiscard]] std::size_t second_weight_column(
    const std::size_t row, const std::size_t entry) noexcept {
  return (151U * row + 337U * entry + 19U) % kColumns;
}

[[nodiscard]] int first_weight_integer(const std::size_t row,
                                       const std::size_t entry) noexcept {
  return ((row + entry) & 1U) == 0U ? 1 : -1;
}

[[nodiscard]] int second_weight_integer(const std::size_t row,
                                        const std::size_t entry) noexcept {
  return ((3U * row + entry) & 1U) == 0U ? 2 : -2;
}

[[nodiscard]] std::uint16_t expected_sparse_dot(
    const std::size_t token, const std::size_t row,
    const bool first_projection) noexcept {
  int sum = 0;
  for (std::size_t entry = 0U; entry < kSparseEntries; ++entry) {
    const std::size_t column =
        first_projection ? first_weight_column(row, entry)
                         : second_weight_column(row, entry);
    const int weight =
        first_projection ? first_weight_integer(row, entry)
                         : second_weight_integer(row, entry);
    sum += activation_integer(token, column) * weight;
  }
  return encode_bf16(static_cast<float>(sum));
}

[[nodiscard]] bool device_is_target(TestContext& test) {
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status == cudaErrorNoDevice ||
      count_status == cudaErrorInsufficientDriver || device_count == 0) {
    (void)cudaGetLastError();
    std::cout << "SKIP: no CUDA device is available\n";
    return false;
  }
  if (!test.cuda_ok(count_status, "cudaGetDeviceCount")) {
    return false;
  }
  int device = 0;
  if (!test.cuda_ok(cudaGetDevice(&device), "cudaGetDevice")) {
    return false;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, device),
                    "cudaGetDeviceProperties")) {
    return false;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: requires the 16-SM SM87 target\n";
    return false;
  }
  return true;
}

struct Payload final {
  std::vector<std::uint16_t> first_weights;
  std::vector<std::uint16_t> second_weights;
  std::vector<std::uint16_t> input;
};

[[nodiscard]] Payload make_payload() {
  Payload payload;
  payload.first_weights.assign(kRows * kColumns, 0U);
  payload.second_weights.assign(kRows * kColumns, 0U);
  payload.input.resize(kMaximumTestTokens * kColumns);
  for (std::size_t row = 0U; row < kRows; ++row) {
    for (std::size_t entry = 0U; entry < kSparseEntries; ++entry) {
      payload.first_weights[row * kColumns +
                            first_weight_column(row, entry)] =
          encode_bf16(static_cast<float>(
              first_weight_integer(row, entry)));
      payload.second_weights[row * kColumns +
                             second_weight_column(row, entry)] =
          encode_bf16(static_cast<float>(
              second_weight_integer(row, entry)));
    }
  }
  for (std::size_t token = 0U; token < kMaximumTestTokens; ++token) {
    for (std::size_t column = 0U; column < kColumns; ++column) {
      payload.input[token * kColumns + column] = encode_bf16(
          static_cast<float>(activation_integer(token, column)));
    }
  }
  return payload;
}

[[nodiscard]] bool reset_guarded_output(
    TestContext& test, DeviceBuffer<std::uint16_t>& output) {
  return test.cuda_ok(
      cudaMemset(output.get(), 0xa5,
                 (2U * kGuardElements + kMaximumTestTokens * kRows) *
                     sizeof(std::uint16_t)),
      "reset guarded output");
}

[[nodiscard]] std::uint16_t* output_data(
    DeviceBuffer<std::uint16_t>& output) noexcept {
  return output.get() + kGuardElements;
}

[[nodiscard]] bool check_guards(
    TestContext& test, const DeviceBuffer<std::uint16_t>& output,
    const std::size_t token_count, const std::string& label) {
  const std::size_t elements =
      kGuardElements + token_count * kRows + kGuardElements;
  std::vector<std::uint16_t> host(elements);
  if (!test.cuda_ok(cudaMemcpy(host.data(), output.get(),
                               elements * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    label + " copy guarded output")) {
    return false;
  }
  for (std::size_t index = 0U; index < kGuardElements; ++index) {
    test.expect(host[index] == kGuardValue,
                label + " preserves the leading guard");
    test.expect(host[kGuardElements + token_count * kRows + index] ==
                    kGuardValue,
                label + " preserves the trailing guard");
  }
  return true;
}

[[nodiscard]] bool copy_output(
    TestContext& test, const DeviceBuffer<std::uint16_t>& source,
    const std::size_t token_count, std::vector<std::uint16_t>& destination,
    const std::string& operation) {
  destination.resize(token_count * kRows);
  return test.cuda_ok(
      cudaMemcpy(destination.data(), source.get() + kGuardElements,
                 destination.size() * sizeof(std::uint16_t),
                 cudaMemcpyDeviceToHost),
      operation);
}

[[nodiscard]] bool launch_established_exact_pair(
    TestContext& test, const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input, const std::size_t token_count,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output, cudaStream_t stream) {
  for (std::size_t offset = 0U; offset < token_count; offset += 16U) {
    const std::size_t remaining = token_count - offset;
    const std::size_t count = remaining < 16U ? remaining : 16U;
    const int status =
        count == 16U
            ? kernels::launch_bf16_gemv_pair_m16_projection_fused_cuda(
                  first_weights, second_weights,
                  input + offset * kColumns,
                  first_output + offset * kRows,
                  second_output + offset * kRows, stream)
            : kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
                  first_weights, second_weights,
                  input + offset * kColumns, count, kRows, kColumns,
                  first_output + offset * kRows,
                  second_output + offset * kRows, stream);
    if (!test.launch_ok(status, "launch established exact pair")) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool verify_resource_contract(TestContext& test) {
  int registers = 0;
  std::size_t static_shared = 0U;
  std::size_t dynamic_shared = 0U;
  std::size_t local = 0U;
  int active_blocks = 0;
  if (!test.launch_ok(
          kernels::query_sm87_bf16_ab_large_m_prefill_resources_cuda(
              &registers, &static_shared, &dynamic_shared, &local,
              &active_blocks),
          "query BF16 A/B resources")) {
    return false;
  }
  test.expect(registers <= 96, "BF16 A/B uses at most 96 registers/thread");
  test.expect(static_shared == 0U,
              "BF16 A/B has no static shared-memory allocation");
  test.expect(dynamic_shared == 46'080U,
              "BF16 A/B retains the two-stage 46080-byte pipeline");
  test.expect(local == 0U, "BF16 A/B has no local-memory spill");
  test.expect(active_blocks >= 2, "BF16 A/B admits at least 2 CTA/SM");
  return true;
}

[[nodiscard]] bool verify_invalid_calls_enqueue_nothing(
    TestContext& test, const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output, cudaStream_t stream) {
  cudaGraph_t graph = nullptr;
  if (!test.cuda_ok(
          cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
          "begin invalid-call capture")) {
    return false;
  }
  test.expect_invalid(
      kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
          first_weights, second_weights, input, 1U,
          first_output, second_output, stream),
      "M1");
  test.expect_invalid(
      kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
          first_weights, second_weights, input,
          kernels::kSm87Bf16AbLargeMPrefillMaximumTokens + 1U,
          first_output, second_output, stream),
      "M4097");
  test.expect_invalid(
      kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
          nullptr, second_weights, input, 2U,
          first_output, second_output, stream),
      "null first weights");
  const auto* const misaligned_input =
      reinterpret_cast<const std::uint16_t*>(
          reinterpret_cast<const std::uint8_t*>(input) + 1U);
  test.expect_invalid(
      kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
          first_weights, second_weights, misaligned_input, 2U,
          first_output, second_output, stream),
      "misaligned input");
  test.expect_invalid(
      kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
          first_weights, second_weights, input, 2U,
          const_cast<std::uint16_t*>(input), second_output, stream),
      "input/output alias");
  test.expect_invalid(
      kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
          first_weights, second_weights, input, 2U,
          first_output, first_output, stream),
      "cross-output alias");
  if (!test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                    "end invalid-call capture")) {
    return false;
  }
  std::size_t node_count = 0U;
  const bool graph_ok = test.cuda_ok(
      cudaGraphGetNodes(graph, nullptr, &node_count),
      "query invalid-call graph nodes");
  test.expect(node_count == 0U,
              "all rejected calls leave a zero-node CUDA graph");
  if (graph != nullptr) {
    (void)cudaGraphDestroy(graph);
  }
  return graph_ok;
}

[[nodiscard]] bool verify_m63_exact_fallback(
    TestContext& test, const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input,
    DeviceBuffer<std::uint16_t>& first_output,
    DeviceBuffer<std::uint16_t>& second_output,
    DeviceBuffer<std::uint16_t>& first_reference,
    DeviceBuffer<std::uint16_t>& second_reference, cudaStream_t stream) {
  constexpr std::size_t kTokens = 63U;
  if (!reset_guarded_output(test, first_output) ||
      !reset_guarded_output(test, second_output) ||
      !test.launch_ok(
          kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
              first_weights, second_weights, input, kTokens,
              output_data(first_output), output_data(second_output), stream),
          "launch M63 BF16 A/B") ||
      !launch_established_exact_pair(
          test, first_weights, second_weights, input, kTokens,
          first_reference.get(), second_reference.get(), stream) ||
      !test.cuda_ok(cudaStreamSynchronize(stream), "synchronize M63")) {
    return false;
  }
  std::vector<std::uint16_t> first_actual;
  std::vector<std::uint16_t> second_actual;
  std::vector<std::uint16_t> first_expected(kTokens * kRows);
  std::vector<std::uint16_t> second_expected(kTokens * kRows);
  if (!copy_output(test, first_output, kTokens, first_actual,
                   "copy M63 first output") ||
      !copy_output(test, second_output, kTokens, second_actual,
                   "copy M63 second output") ||
      !test.cuda_ok(cudaMemcpy(first_expected.data(), first_reference.get(),
                               first_expected.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "copy M63 first reference") ||
      !test.cuda_ok(cudaMemcpy(second_expected.data(), second_reference.get(),
                               second_expected.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "copy M63 second reference")) {
    return false;
  }
  test.expect(first_actual == first_expected,
              "M63 first projection is bit-exact to the established path");
  test.expect(second_actual == second_expected,
              "M63 second projection is bit-exact to the established path");
  return check_guards(test, first_output, kTokens, "M63 first") &&
         check_guards(test, second_output, kTokens, "M63 second");
}

[[nodiscard]] bool verify_m65_tensor_core_and_exact_tail(
    TestContext& test, const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input,
    DeviceBuffer<std::uint16_t>& first_output,
    DeviceBuffer<std::uint16_t>& second_output,
    DeviceBuffer<std::uint16_t>& first_reference,
    DeviceBuffer<std::uint16_t>& second_reference, cudaStream_t stream) {
  constexpr std::size_t kTokens = 65U;
  if (!reset_guarded_output(test, first_output) ||
      !reset_guarded_output(test, second_output) ||
      !test.launch_ok(
          kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
              first_weights, second_weights, input, kTokens,
              output_data(first_output), output_data(second_output), stream),
          "launch M65 BF16 A/B") ||
      !test.launch_ok(
          kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input + 64U * kColumns,
              1U, kRows, kColumns, first_reference.get(),
              second_reference.get(), stream),
          "launch M65 exact tail reference") ||
      !test.cuda_ok(cudaStreamSynchronize(stream), "synchronize M65")) {
    return false;
  }
  std::vector<std::uint16_t> first_actual;
  std::vector<std::uint16_t> second_actual;
  std::vector<std::uint16_t> first_tail(kRows);
  std::vector<std::uint16_t> second_tail(kRows);
  if (!copy_output(test, first_output, kTokens, first_actual,
                   "copy M65 first output") ||
      !copy_output(test, second_output, kTokens, second_actual,
                   "copy M65 second output") ||
      !test.cuda_ok(cudaMemcpy(first_tail.data(), first_reference.get(),
                               first_tail.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "copy M65 first tail") ||
      !test.cuda_ok(cudaMemcpy(second_tail.data(), second_reference.get(),
                               second_tail.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "copy M65 second tail")) {
    return false;
  }
  std::size_t first_mismatches = 0U;
  std::size_t second_mismatches = 0U;
  for (std::size_t token = 0U; token < 64U; ++token) {
    for (std::size_t row = 0U; row < kRows; ++row) {
      const std::uint16_t first_expected =
          expected_sparse_dot(token, row, true);
      const std::uint16_t second_expected =
          expected_sparse_dot(token, row, false);
      const std::uint16_t first_value = first_actual[token * kRows + row];
      const std::uint16_t second_value = second_actual[token * kRows + row];
      if (first_value != first_expected) {
        if (first_mismatches < 8U) {
          std::cerr << "M65 first mismatch token=" << token
                    << " row=" << row
                    << " expected=" << decode_bf16(first_expected)
                    << " actual=" << decode_bf16(first_value) << '\n';
        }
        ++first_mismatches;
      }
      if (second_value != second_expected) {
        if (second_mismatches < 8U) {
          std::cerr << "M65 second mismatch token=" << token
                    << " row=" << row
                    << " expected=" << decode_bf16(second_expected)
                    << " actual=" << decode_bf16(second_value) << '\n';
        }
        ++second_mismatches;
      }
    }
  }
  test.expect(first_mismatches == 0U,
              "M65 first Tensor Core prefix matches the exact sparse dot (" +
                  std::to_string(first_mismatches) + " mismatches)");
  test.expect(second_mismatches == 0U,
              "M65 second Tensor Core prefix matches the exact sparse dot (" +
                  std::to_string(second_mismatches) + " mismatches)");
  for (std::size_t row = 0U; row < kRows; ++row) {
    test.expect(first_actual[64U * kRows + row] == first_tail[row],
                "M65 first tail is bit-exact to the established M1 path");
    test.expect(second_actual[64U * kRows + row] == second_tail[row],
                "M65 second tail is bit-exact to the established M1 path");
  }
  return check_guards(test, first_output, kTokens, "M65 first") &&
         check_guards(test, second_output, kTokens, "M65 second");
}

[[nodiscard]] bool verify_m513_extension_and_exact_tail(
    TestContext& test, const std::uint16_t* const first_weights,
    const std::uint16_t* const second_weights,
    const std::uint16_t* const input,
    DeviceBuffer<std::uint16_t>& first_output,
    DeviceBuffer<std::uint16_t>& second_output,
    DeviceBuffer<std::uint16_t>& first_reference,
    DeviceBuffer<std::uint16_t>& second_reference, cudaStream_t stream) {
  constexpr std::size_t kTokens = 513U;
  if (!reset_guarded_output(test, first_output) ||
      !reset_guarded_output(test, second_output) ||
      !test.launch_ok(
          kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
              first_weights, second_weights, input, kTokens,
              output_data(first_output), output_data(second_output), stream),
          "launch M513 BF16 A/B") ||
      !test.launch_ok(
          kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input + 512U * kColumns,
              1U, kRows, kColumns, first_reference.get(),
              second_reference.get(), stream),
          "launch M513 exact tail reference") ||
      !test.cuda_ok(cudaStreamSynchronize(stream), "synchronize M513")) {
    return false;
  }
  std::vector<std::uint16_t> first_actual;
  std::vector<std::uint16_t> second_actual;
  std::vector<std::uint16_t> first_tail(kRows);
  std::vector<std::uint16_t> second_tail(kRows);
  if (!copy_output(test, first_output, kTokens, first_actual,
                   "copy M513 first output") ||
      !copy_output(test, second_output, kTokens, second_actual,
                   "copy M513 second output") ||
      !test.cuda_ok(cudaMemcpy(first_tail.data(), first_reference.get(),
                               first_tail.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "copy M513 first tail") ||
      !test.cuda_ok(cudaMemcpy(second_tail.data(), second_reference.get(),
                               second_tail.size() * sizeof(std::uint16_t),
                               cudaMemcpyDeviceToHost),
                    "copy M513 second tail")) {
    return false;
  }
  for (std::size_t row = 0U; row < kRows; ++row) {
    test.expect(first_actual[512U * kRows + row] == first_tail[row],
                "M513 first tail is bit-exact to the established M1 path");
    test.expect(second_actual[512U * kRows + row] == second_tail[row],
                "M513 second tail is bit-exact to the established M1 path");
  }
  return check_guards(test, first_output, kTokens, "M513 first") &&
         check_guards(test, second_output, kTokens, "M513 second");
}

[[nodiscard]] bool run_test(TestContext& test) {
  const Payload payload = make_payload();
  DeviceBuffer<std::uint16_t> device_first_weights;
  DeviceBuffer<std::uint16_t> device_second_weights;
  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint16_t> device_first_output;
  DeviceBuffer<std::uint16_t> device_second_output;
  DeviceBuffer<std::uint16_t> device_first_reference;
  DeviceBuffer<std::uint16_t> device_second_reference;
  StreamOwner stream;
  const std::size_t guarded_output_elements =
      2U * kGuardElements + kMaximumTestTokens * kRows;
  if (!test.cuda_ok(device_first_weights.allocate(
                        payload.first_weights.size()),
                    "allocate first weights") ||
      !test.cuda_ok(device_second_weights.allocate(
                        payload.second_weights.size()),
                    "allocate second weights") ||
      !test.cuda_ok(device_input.allocate(payload.input.size()),
                    "allocate input") ||
      !test.cuda_ok(device_first_output.allocate(guarded_output_elements),
                    "allocate first output") ||
      !test.cuda_ok(device_second_output.allocate(guarded_output_elements),
                    "allocate second output") ||
      !test.cuda_ok(device_first_reference.allocate(63U * kRows),
                    "allocate first reference") ||
      !test.cuda_ok(device_second_reference.allocate(63U * kRows),
                    "allocate second reference") ||
      !test.cuda_ok(stream.create(), "create stream") ||
      !test.cuda_ok(cudaMemcpy(device_first_weights.get(),
                               payload.first_weights.data(),
                               payload.first_weights.size() *
                                   sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice),
                    "copy first weights") ||
      !test.cuda_ok(cudaMemcpy(device_second_weights.get(),
                               payload.second_weights.data(),
                               payload.second_weights.size() *
                                   sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice),
                    "copy second weights") ||
      !test.cuda_ok(cudaMemcpy(device_input.get(), payload.input.data(),
                               payload.input.size() * sizeof(std::uint16_t),
                               cudaMemcpyHostToDevice),
                    "copy input")) {
    return false;
  }

  return verify_resource_contract(test) &&
         verify_invalid_calls_enqueue_nothing(
             test, device_first_weights.get(), device_second_weights.get(),
             device_input.get(), output_data(device_first_output),
             output_data(device_second_output), stream.get()) &&
         verify_m63_exact_fallback(
             test, device_first_weights.get(), device_second_weights.get(),
             device_input.get(), device_first_output, device_second_output,
             device_first_reference, device_second_reference, stream.get()) &&
         verify_m65_tensor_core_and_exact_tail(
             test, device_first_weights.get(), device_second_weights.get(),
             device_input.get(), device_first_output, device_second_output,
             device_first_reference, device_second_reference, stream.get()) &&
         verify_m513_extension_and_exact_tail(
             test, device_first_weights.get(), device_second_weights.get(),
             device_input.get(), device_first_output, device_second_output,
             device_first_reference, device_second_reference, stream.get());
}

}  // namespace

int main() {
  TestContext test;
  if (!device_is_target(test)) {
    return test.failures() == 0 ? 77 : 1;
  }
  (void)run_test(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " BF16 A/B whole-span assertion(s) failed\n";
    return 1;
  }
  std::cout << "SM87 BF16 A/B whole-span correctness/guard test passed\n";
  return 0;
}
