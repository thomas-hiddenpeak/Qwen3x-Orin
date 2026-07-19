#include "q3x/kernels/reference_gemv.h"
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
                   const std::size_t columns, const std::string& message) {
    const float tolerance = 5.0e-5F * std::sqrt(static_cast<float>(columns)) +
                            5.0e-4F * std::fabs(expected);
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
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t count) {
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
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

[[nodiscard]] bool seed_stale_error(TestContext& test) {
  const cudaError_t status =
      cudaMemcpy(nullptr, nullptr, 1U, cudaMemcpyHostToDevice);
  test.expect(status == cudaErrorInvalidValue,
              "invalid CUDA copy seeds stale last-error");
  return status == cudaErrorInvalidValue;
}

void compare_outputs(TestContext& test, const std::vector<float>& gpu,
                     const std::vector<float>& cpu,
                     const std::size_t columns, const std::string& label) {
  for (std::size_t row = 0; row < cpu.size(); ++row) {
    test.expect_near(gpu[row], cpu[row], columns,
                     label + " row " + std::to_string(row));
  }
}

void run_bf16_case(TestContext& test, cudaStream_t stream,
                   const std::size_t rows, const std::size_t columns,
                   const std::string& label) {
  std::vector<std::uint16_t> activation(columns);
  std::vector<std::uint16_t> weights(rows * columns);
  for (std::size_t column = 0; column < columns; ++column) {
    const int centered = static_cast<int>(column % 17U) - 8;
    activation[column] = encode_bf16(static_cast<float>(centered) / 16.0F);
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    const int centered = static_cast<int>((index * 7U + 3U) % 29U) - 14;
    weights[index] = encode_bf16(static_cast<float>(centered) / 32.0F);
  }
  std::vector<float> cpu(rows);
  std::vector<float> gpu(rows, std::numeric_limits<float>::quiet_NaN());
  (void)q3x::kernels::bf16_gemv_reference_cpu(
      weights.data(), activation.data(), rows, columns, cpu.data());

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_weights;
  DeviceBuffer<float> device_output;
  bool ready = test.cuda_ok(device_activation.allocate(activation.size()),
                            label + " allocate activation");
  ready = ready && test.cuda_ok(device_weights.allocate(weights.size()),
                                label + " allocate weights");
  ready = ready && test.cuda_ok(device_output.allocate(gpu.size()),
                                label + " allocate output");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      cudaMemcpyAsync(device_activation.get(), activation.data(),
                      activation.size() * sizeof(activation[0]),
                      cudaMemcpyHostToDevice, stream),
      label + " copy activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_weights.get(), weights.data(),
                                       weights.size() * sizeof(weights[0]),
                                       cudaMemcpyHostToDevice, stream),
                       label + " copy weights");
  if (!ready) {
    return;
  }

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::launch_bf16_gemv_reference_cuda(
          device_weights.get(), device_activation.get(), rows, columns,
          device_output.get(), static_cast<void*>(stream))),
      label + " launch after stale error");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(gpu.data(), device_output.get(),
                                       gpu.size() * sizeof(gpu[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy output");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (ready) {
    compare_outputs(test, gpu, cpu, columns, label);
  }
}

void run_bf16_pair_tile_case(TestContext& test, cudaStream_t stream,
                             const std::size_t token_count,
                             const std::size_t rows,
                             const std::size_t columns,
                             const std::string& label) {
  std::vector<std::uint16_t> input(token_count * columns);
  std::vector<std::uint16_t> first_weights(rows * columns);
  std::vector<std::uint16_t> second_weights(rows * columns);
  for (std::size_t index = 0; index < input.size(); ++index) {
    const int centered = static_cast<int>((index * 11U + 5U) % 31U) - 15;
    input[index] = encode_bf16(static_cast<float>(centered) / 32.0F);
  }
  for (std::size_t index = 0; index < first_weights.size(); ++index) {
    const int first_centered =
        static_cast<int>((index * 7U + 3U) % 29U) - 14;
    const int second_centered =
        static_cast<int>((index * 13U + 9U) % 37U) - 18;
    first_weights[index] =
        encode_bf16(static_cast<float>(first_centered) / 64.0F);
    second_weights[index] =
        encode_bf16(static_cast<float>(second_centered) / 128.0F);
  }

  constexpr std::size_t kGuardElements = 8U;
  constexpr std::uint16_t kGuard = 0xa55aU;
  const std::size_t output_elements = token_count * rows;
  const std::size_t output_storage_elements =
      output_elements + 2U * kGuardElements;
  std::vector<std::uint16_t> first_expected(output_elements);
  std::vector<std::uint16_t> second_expected(output_elements);
  std::vector<std::uint16_t> first_actual(output_storage_elements, kGuard);
  std::vector<std::uint16_t> second_actual(output_storage_elements, kGuard);
  std::vector<std::uint16_t> input_after(input.size());
  std::vector<std::uint16_t> first_weights_after(first_weights.size());
  std::vector<std::uint16_t> second_weights_after(second_weights.size());
  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint16_t> device_first_weights;
  DeviceBuffer<std::uint16_t> device_second_weights;
  DeviceBuffer<float> device_first_reference;
  DeviceBuffer<float> device_second_reference;
  DeviceBuffer<std::uint16_t> device_first_expected;
  DeviceBuffer<std::uint16_t> device_second_expected;
  DeviceBuffer<std::uint16_t> device_first_output;
  DeviceBuffer<std::uint16_t> device_second_output;
  bool ready = test.cuda_ok(device_input.allocate(input.size()),
                            label + " allocate input");
  ready = ready && test.cuda_ok(
                       device_first_weights.allocate(first_weights.size()),
                       label + " allocate first weights");
  ready = ready && test.cuda_ok(
                       device_second_weights.allocate(second_weights.size()),
                       label + " allocate second weights");
  ready = ready && test.cuda_ok(
                       device_first_reference.allocate(output_elements),
                       label + " allocate first reference");
  ready = ready && test.cuda_ok(
                       device_second_reference.allocate(output_elements),
                       label + " allocate second reference");
  ready = ready && test.cuda_ok(
                       device_first_expected.allocate(output_elements),
                       label + " allocate first expected");
  ready = ready && test.cuda_ok(
                       device_second_expected.allocate(output_elements),
                       label + " allocate second expected");
  ready = ready && test.cuda_ok(
                       device_first_output.allocate(output_storage_elements),
                       label + " allocate first output");
  ready = ready && test.cuda_ok(
                       device_second_output.allocate(output_storage_elements),
                       label + " allocate second output");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      cudaMemcpyAsync(device_input.get(), input.data(),
                      input.size() * sizeof(input[0]), cudaMemcpyHostToDevice,
                      stream),
      label + " copy input");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_first_weights.get(), first_weights.data(),
                           first_weights.size() * sizeof(first_weights[0]),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy first weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_second_weights.get(), second_weights.data(),
                           second_weights.size() * sizeof(second_weights[0]),
                           cudaMemcpyHostToDevice, stream),
                       label + " copy second weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_first_output.get(), first_actual.data(),
                           output_storage_elements * sizeof(first_actual[0]),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize first output guards");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_second_output.get(), second_actual.data(),
                           output_storage_elements * sizeof(second_actual[0]),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize second output guards");
  if (!ready) {
    return;
  }

  for (std::size_t token = 0U; token < token_count && ready; ++token) {
    ready = test.cuda_ok(
        static_cast<cudaError_t>(q3x::kernels::launch_bf16_gemv_reference_cuda(
            device_first_weights.get(), device_input.get() + token * columns,
            rows, columns, device_first_reference.get() + token * rows,
            static_cast<void*>(stream))),
        label + " launch first reference token " + std::to_string(token));
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(
                             q3x::kernels::launch_bf16_gemv_reference_cuda(
                                 device_second_weights.get(),
                                 device_input.get() + token * columns, rows,
                                 columns,
                                 device_second_reference.get() + token * rows,
                                 static_cast<void*>(stream))),
                         label + " launch second reference token " +
                             std::to_string(token));
  }
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               device_first_reference.get(), output_elements,
                               device_first_expected.get(),
                               static_cast<void*>(stream))),
                       label + " convert first reference to BF16");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               device_second_reference.get(), output_elements,
                               device_second_expected.get(),
                               static_cast<void*>(stream))),
                       label + " convert second reference to BF16");
  (void)seed_stale_error(test);
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
                               device_first_weights.get(),
                               device_second_weights.get(), device_input.get(),
                               token_count, rows, columns,
                               device_first_output.get() + kGuardElements,
                               device_second_output.get() + kGuardElements,
                               static_cast<void*>(stream))),
                       label + " launch pair after stale error");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           first_expected.data(), device_first_expected.get(),
                           output_elements * sizeof(first_expected[0]),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy first expected");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(second_expected.data(),
                                       device_second_expected.get(),
                                       output_elements *
                                           sizeof(second_expected[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy second expected");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(first_actual.data(),
                                       device_first_output.get(),
                                       output_storage_elements *
                                           sizeof(first_actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy first output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(second_actual.data(),
                                       device_second_output.get(),
                                       output_storage_elements *
                                           sizeof(second_actual[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy second output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(input_after.data(), device_input.get(),
                                       input.size() * sizeof(input_after[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy input after launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           first_weights_after.data(),
                           device_first_weights.get(),
                           first_weights.size() *
                               sizeof(first_weights_after[0]),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy first weights after launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           second_weights_after.data(),
                           device_second_weights.get(),
                           second_weights.size() *
                               sizeof(second_weights_after[0]),
                           cudaMemcpyDeviceToHost, stream),
                       label + " copy second weights after launch");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (!ready) {
    return;
  }

  for (std::size_t index = 0; index < output_elements; ++index) {
    test.expect(first_actual[kGuardElements + index] ==
                    first_expected[index],
                label + " first output " + std::to_string(index) +
                    " matches two-stage CUDA oracle bitwise");
    test.expect(second_actual[kGuardElements + index] ==
                    second_expected[index],
                label + " second output " + std::to_string(index) +
                    " matches two-stage CUDA oracle bitwise");
  }
  for (std::size_t index = 0; index < kGuardElements; ++index) {
    test.expect(first_actual[index] == kGuard &&
                    first_actual[kGuardElements + output_elements + index] ==
                        kGuard,
                label + " first output canaries are intact");
    test.expect(second_actual[index] == kGuard &&
                    second_actual[kGuardElements + output_elements + index] ==
                        kGuard,
                label + " second output canaries are intact");
  }
  test.expect(input_after == input, label + " leaves input immutable");
  test.expect(first_weights_after == first_weights,
              label + " leaves first weights immutable");
  test.expect(second_weights_after == second_weights,
              label + " leaves second weights immutable");
}

enum class Bf16DirectFixture : std::uint8_t {
  kStructured,
  kNumericEdges,
};

void fill_bf16_direct_fixture(const Bf16DirectFixture fixture,
                              std::vector<std::uint16_t>* const activation,
                              std::vector<std::uint16_t>* const weights,
                              const std::size_t rows,
                              const std::size_t columns) {
  if (fixture == Bf16DirectFixture::kStructured) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered =
          static_cast<int>((column * 17U + 11U) % 61U) - 30;
      (*activation)[column] =
          encode_bf16(static_cast<float>(centered) / 64.0F);
    }
    for (std::size_t index = 0U; index < weights->size(); ++index) {
      const int centered =
          static_cast<int>((index * 29U + 7U) % 67U) - 33;
      (*weights)[index] =
          encode_bf16(static_cast<float>(centered) / 128.0F);
    }
    return;
  }

  // The first rows force both ties-to-even directions, signed zeros,
  // infinities, and NaN quieting while retaining the production matrix
  // dimensions and reduction tree.
  std::fill(activation->begin(), activation->end(), 0x0080U);
  std::fill(weights->begin(), weights->end(), 0x0000U);
  (*activation)[0U] = encode_bf16(1.0F);
  (*activation)[1U] = encode_bf16(1.0F / 256.0F);

  (*weights)[0U * columns] = encode_bf16(1.0F);
  (*weights)[0U * columns + 1U] = encode_bf16(1.0F);
  (*weights)[1U * columns] = encode_bf16(1.0F);
  (*weights)[1U * columns + 1U] = encode_bf16(3.0F);
  (*weights)[2U * columns] = encode_bf16(-1.0F);
  (*weights)[2U * columns + 1U] = encode_bf16(-1.0F);
  (*weights)[3U * columns] = encode_bf16(-1.0F);
  (*weights)[3U * columns + 1U] = encode_bf16(-3.0F);
  std::fill_n(weights->data() + 4U * columns, columns, 0x8080U);
  std::fill_n(weights->data() + 5U * columns, columns, 0x0080U);
  (*weights)[4U * columns] = 0x0000U;
  (*weights)[4U * columns + 1U] = 0x0000U;
  (*weights)[5U * columns] = 0x0000U;
  (*weights)[5U * columns + 1U] = 0x0000U;
  (*weights)[6U * columns] = 0x7f81U;
  (*weights)[7U * columns] = 0x7f80U;
  (*weights)[8U * columns] = 0xff80U;

  for (std::size_t row = 9U; row < rows; ++row) {
    for (std::size_t column = 0U; column < columns; ++column) {
      const int centered = static_cast<int>(
                               (row * 13U + column * 7U + 3U) % 19U) -
                           9;
      (*weights)[row * columns + column] =
          encode_bf16(static_cast<float>(centered) / 32.0F);
    }
  }
}

void run_bf16_direct_case(TestContext& test, cudaStream_t stream,
                          const Bf16DirectFixture fixture,
                          const std::string& label) {
  constexpr std::size_t kRows = 48U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kGuardElements = 8U;
  constexpr std::uint16_t kCanary = 0xa55aU;
  constexpr std::uint16_t kPoison = 0xdeadU;
  std::vector<std::uint16_t> activation(kColumns);
  std::vector<std::uint16_t> weights(kRows * kColumns);
  fill_bf16_direct_fixture(fixture, &activation, &weights, kRows, kColumns);

  const std::size_t output_storage_elements =
      kRows + 2U * kGuardElements;
  std::vector<std::uint16_t> initialized(output_storage_elements, kCanary);
  std::fill_n(initialized.begin() +
                  static_cast<std::ptrdiff_t>(kGuardElements),
              kRows, kPoison);
  std::vector<std::uint16_t> expected(kRows);
  std::vector<std::uint16_t> actual(output_storage_elements);
  std::vector<std::uint16_t> activation_after(kColumns);
  std::vector<std::uint16_t> weights_after(kRows * kColumns);

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_weights;
  DeviceBuffer<float> device_reference;
  DeviceBuffer<std::uint16_t> device_expected;
  DeviceBuffer<std::uint16_t> device_output;
  bool ready = test.cuda_ok(device_activation.allocate(kColumns),
                            label + " allocate activation");
  ready = ready && test.cuda_ok(
                       device_weights.allocate(kRows * kColumns),
                       label + " allocate weights");
  ready = ready && test.cuda_ok(device_reference.allocate(kRows),
                                label + " allocate FP32 reference");
  ready = ready && test.cuda_ok(device_expected.allocate(kRows),
                                label + " allocate BF16 reference");
  ready = ready && test.cuda_ok(
                       device_output.allocate(output_storage_elements),
                       label + " allocate guarded output");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      cudaMemcpyAsync(device_activation.get(), activation.data(),
                      activation.size() * sizeof(activation.front()),
                      cudaMemcpyHostToDevice, stream),
      label + " upload activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_weights.get(), weights.data(),
                           weights.size() * sizeof(weights.front()),
                           cudaMemcpyHostToDevice, stream),
                       label + " upload weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_output.get(), initialized.data(),
                           initialized.size() * sizeof(initialized.front()),
                           cudaMemcpyHostToDevice, stream),
                       label + " initialize poison and canaries");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::kernels::launch_bf16_gemv_reference_cuda(
                               device_weights.get(), device_activation.get(),
                               kRows, kColumns, device_reference.get(),
                               static_cast<void*>(stream))),
                       label + " launch FP32 baseline");
  ready = ready && test.cuda_ok(
                       static_cast<cudaError_t>(
                           q3x::runtime::launch_fp32_to_bf16_reference_cuda(
                               device_reference.get(), kRows,
                               device_expected.get(),
                               static_cast<void*>(stream))),
                       label + " convert baseline to BF16");
  if (!ready) {
    return;
  }

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_bf16_cuda(
              device_weights.get(), device_activation.get(), kRows, kColumns,
              device_output.get() + kGuardElements,
              static_cast<void*>(stream))),
      label + " launch direct BF16 after stale error");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           expected.data(), device_expected.get(),
                           expected.size() * sizeof(expected.front()),
                           cudaMemcpyDeviceToHost, stream),
                       label + " download baseline");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           actual.data(), device_output.get(),
                           actual.size() * sizeof(actual.front()),
                           cudaMemcpyDeviceToHost, stream),
                       label + " download guarded output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activation_after.data(), device_activation.get(),
                           activation_after.size() *
                               sizeof(activation_after.front()),
                           cudaMemcpyDeviceToHost, stream),
                       label + " download activation after launch");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           weights_after.data(), device_weights.get(),
                           weights_after.size() * sizeof(weights_after.front()),
                           cudaMemcpyDeviceToHost, stream),
                       label + " download weights after launch");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (!ready) {
    return;
  }

  for (std::size_t row = 0U; row < kRows; ++row) {
    test.expect(actual[kGuardElements + row] == expected[row],
                label + " row " + std::to_string(row) +
                    " matches GEMV-plus-convert bitwise");
  }
  for (std::size_t index = 0U; index < kGuardElements; ++index) {
    test.expect(actual[index] == kCanary &&
                    actual[kGuardElements + kRows + index] == kCanary,
                label + " output canaries remain intact");
  }
  test.expect(activation_after == activation,
              label + " leaves activation immutable");
  test.expect(weights_after == weights, label + " leaves weights immutable");

  if (fixture == Bf16DirectFixture::kNumericEdges) {
    test.expect(actual[kGuardElements + 0U] == 0x3f80U &&
                    actual[kGuardElements + 1U] == 0x3f82U,
                label + " preserves positive RNE midpoint behavior");
    test.expect(actual[kGuardElements + 2U] == 0xbf80U &&
                    actual[kGuardElements + 3U] == 0xbf82U,
                label + " preserves negative RNE midpoint behavior");
    test.expect(actual[kGuardElements + 4U] == 0x8000U &&
                    actual[kGuardElements + 5U] == 0x0000U,
                label + " preserves both signed zeros");
    test.expect(actual[kGuardElements + 7U] == 0x7f80U &&
                    actual[kGuardElements + 8U] == 0xff80U,
                label + " preserves signed infinities");
    const std::uint16_t nan = actual[kGuardElements + 6U];
    test.expect((nan & 0x7f80U) == 0x7f80U &&
                    (nan & 0x007fU) != 0U && (nan & 0x0040U) != 0U,
                label + " quiets signaling NaN");
  }
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
  for (std::size_t iteration = 0U;
       ready && iteration < iteration_count; ++iteration) {
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

void test_bf16_direct_performance(TestContext& test, cudaStream_t stream) {
  const char* const enabled = std::getenv("Q3X_RUN_BF16_DIRECT_GEMV_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: direct BF16 GEMV performance gate; set "
                 "Q3X_RUN_BF16_DIRECT_GEMV_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kRows = 48U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kWarmupIterations = 128U;
  constexpr std::size_t kMeasuredIterations = 2'048U;
  constexpr int kRounds = 3;
  constexpr double kMinimumSpeedup = 1.08;
  const std::string label = "direct BF16 GEMV 48x5120 perf";
  std::vector<std::uint16_t> activation(kColumns);
  std::vector<std::uint16_t> weights(kRows * kColumns);
  fill_bf16_direct_fixture(Bf16DirectFixture::kStructured, &activation,
                           &weights, kRows, kColumns);

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_weights;
  DeviceBuffer<float> first_reference;
  DeviceBuffer<float> second_reference;
  DeviceBuffer<std::uint16_t> first_baseline;
  DeviceBuffer<std::uint16_t> second_baseline;
  DeviceBuffer<std::uint16_t> first_candidate;
  DeviceBuffer<std::uint16_t> second_candidate;
  bool ready = test.cuda_ok(device_activation.allocate(kColumns),
                            label + " activation");
  ready = ready && test.cuda_ok(device_weights.allocate(kRows * kColumns),
                                label + " weights");
  ready = ready &&
          test.cuda_ok(first_reference.allocate(kRows), label + " scratch A");
  ready = ready && test.cuda_ok(second_reference.allocate(kRows),
                                label + " scratch B");
  ready = ready &&
          test.cuda_ok(first_baseline.allocate(kRows), label + " baseline A");
  ready = ready && test.cuda_ok(second_baseline.allocate(kRows),
                                label + " baseline B");
  ready = ready && test.cuda_ok(first_candidate.allocate(kRows),
                                label + " candidate A");
  ready = ready && test.cuda_ok(second_candidate.allocate(kRows),
                                label + " candidate B");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_activation.get(), activation.data(),
                           activation.size() * sizeof(activation.front()),
                           cudaMemcpyHostToDevice, stream),
                       label + " upload activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_weights.get(), weights.data(),
                           weights.size() * sizeof(weights.front()),
                           cudaMemcpyHostToDevice, stream),
                       label + " upload weights");
  if (!ready) {
    return;
  }

  // Two consecutive projections model the production A/B use site.
  const auto launch_baseline = [&]() {
    int status = q3x::kernels::launch_bf16_gemv_reference_cuda(
        device_weights.get(), device_activation.get(), kRows, kColumns,
        first_reference.get(), static_cast<void*>(stream));
    if (status == static_cast<int>(cudaSuccess)) {
      status = q3x::runtime::launch_fp32_to_bf16_reference_cuda(
          first_reference.get(), kRows, first_baseline.get(),
          static_cast<void*>(stream));
    }
    if (status == static_cast<int>(cudaSuccess)) {
      status = q3x::kernels::launch_bf16_gemv_reference_cuda(
          device_weights.get(), device_activation.get(), kRows, kColumns,
          second_reference.get(), static_cast<void*>(stream));
    }
    if (status == static_cast<int>(cudaSuccess)) {
      status = q3x::runtime::launch_fp32_to_bf16_reference_cuda(
          second_reference.get(), kRows, second_baseline.get(),
          static_cast<void*>(stream));
    }
    return status;
  };
  const auto launch_candidate = [&]() {
    int status = q3x::kernels::launch_bf16_gemv_bf16_cuda(
        device_weights.get(), device_activation.get(), kRows, kColumns,
        first_candidate.get(), static_cast<void*>(stream));
    if (status == static_cast<int>(cudaSuccess)) {
      status = q3x::kernels::launch_bf16_gemv_bf16_cuda(
          device_weights.get(), device_activation.get(), kRows, kColumns,
          second_candidate.get(), static_cast<void*>(stream));
    }
    return status;
  };

  for (std::size_t iteration = 0U;
       ready && iteration < kWarmupIterations; ++iteration) {
    ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                         label + " baseline warmup");
    ready = ready && test.cuda_ok(
                         static_cast<cudaError_t>(launch_candidate()),
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
  for (int round = 0; round < kRounds; ++round) {
    const std::string round_label =
        label + " round=" + std::to_string(round + 1);
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
    timing_finite = timing_finite && std::isfinite(baseline_first) &&
                    std::isfinite(candidate_first) &&
                    std::isfinite(candidate_second) &&
                    std::isfinite(baseline_second) && baseline_first > 0.0F &&
                    candidate_first > 0.0F && candidate_second > 0.0F &&
                    baseline_second > 0.0F;
    baseline_total +=
        (static_cast<double>(baseline_first) + baseline_second) * 0.5;
    candidate_total +=
        (static_cast<double>(candidate_first) + candidate_second) * 0.5;
  }
  test.expect(timing_finite, label + " timings are finite and positive");
  if (timing_finite) {
    const double baseline_average = baseline_total / kRounds;
    const double candidate_average = candidate_total / kRounds;
    const double speedup = baseline_average / candidate_average;
    std::cout << label << ": baseline=" << baseline_average
              << " ms, candidate=" << candidate_average
              << " ms, speedup=" << speedup << "x\n";
    test.expect(speedup >= kMinimumSpeedup,
                label + " speedup is at least " +
                    std::to_string(kMinimumSpeedup) + "x");
  }
}

void test_bf16_m1_pair_performance(TestContext& test,
                                   cudaStream_t stream) {
  const char* const enabled = std::getenv("Q3X_RUN_BF16_M1_PAIR_PERF");
  if (enabled == nullptr || std::strcmp(enabled, "1") != 0) {
    std::cout << "SKIP: BF16 M1 pair performance gate; set "
                 "Q3X_RUN_BF16_M1_PAIR_PERF=1 to enable\n";
    return;
  }

  constexpr std::size_t kRows = 48U;
  constexpr std::size_t kColumns = 5'120U;
  constexpr std::size_t kWarmupIterations = 128U;
  constexpr std::size_t kMeasuredIterations = 2'048U;
  constexpr int kRounds = 4;
  constexpr double kMinimumSpeedup = 1.20;
  const std::string label = "BF16 M1 A/B pair production 48x5120 perf";

  std::vector<std::uint16_t> activation(kColumns);
  std::vector<std::uint16_t> first_weights(kRows * kColumns);
  fill_bf16_direct_fixture(Bf16DirectFixture::kStructured, &activation,
                           &first_weights, kRows, kColumns);
  std::vector<std::uint16_t> second_weights(kRows * kColumns);
  for (std::size_t index = 0U; index < second_weights.size(); ++index) {
    const int centered =
        static_cast<int>((index * 29U + index / kColumns * 11U + 7U) %
                         127U) -
        63;
    second_weights[index] =
        encode_bf16(static_cast<float>(centered) / 128.0F);
  }

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint16_t> device_first_weights;
  DeviceBuffer<std::uint16_t> device_second_weights;
  DeviceBuffer<std::uint16_t> baseline_first;
  DeviceBuffer<std::uint16_t> baseline_second;
  DeviceBuffer<std::uint16_t> candidate_first;
  DeviceBuffer<std::uint16_t> candidate_second;
  bool ready = test.cuda_ok(device_activation.allocate(kColumns),
                            label + " activation");
  ready = ready && test.cuda_ok(
                       device_first_weights.allocate(kRows * kColumns),
                       label + " first row-major weights");
  ready = ready && test.cuda_ok(
                       device_second_weights.allocate(kRows * kColumns),
                       label + " second row-major weights");
  ready = ready &&
          test.cuda_ok(baseline_first.allocate(kRows), label + " baseline A");
  ready = ready && test.cuda_ok(baseline_second.allocate(kRows),
                                label + " baseline B");
  ready = ready && test.cuda_ok(candidate_first.allocate(kRows),
                                label + " candidate A");
  ready = ready && test.cuda_ok(candidate_second.allocate(kRows),
                                label + " candidate B");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_activation.get(), activation.data(),
                           activation.size() * sizeof(activation.front()),
                           cudaMemcpyHostToDevice, stream),
                       label + " upload activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_first_weights.get(), first_weights.data(),
                           first_weights.size() *
                               sizeof(first_weights.front()),
                           cudaMemcpyHostToDevice, stream),
                       label + " upload first weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_second_weights.get(), second_weights.data(),
                           second_weights.size() *
                               sizeof(second_weights.front()),
                           cudaMemcpyHostToDevice, stream),
                       label + " upload second weights");
  if (!ready) {
    return;
  }

  const auto launch_baseline = [&]() {
    int status = q3x::kernels::launch_bf16_gemv_bf16_cuda(
        device_first_weights.get(), device_activation.get(), kRows, kColumns,
        baseline_first.get(), static_cast<void*>(stream));
    if (status == static_cast<int>(cudaSuccess)) {
      status = q3x::kernels::launch_bf16_gemv_bf16_cuda(
          device_second_weights.get(), device_activation.get(), kRows,
          kColumns, baseline_second.get(), static_cast<void*>(stream));
    }
    return status;
  };
  const auto launch_candidate = [&]() {
    return q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
        device_first_weights.get(), device_second_weights.get(),
        device_activation.get(), 1U, kRows, kColumns,
        candidate_first.get(), candidate_second.get(),
        static_cast<void*>(stream));
  };

  ready = test.cuda_ok(static_cast<cudaError_t>(launch_baseline()),
                       label + " bitwise baseline");
  ready = ready &&
          test.cuda_ok(static_cast<cudaError_t>(launch_candidate()),
                       label + " bitwise candidate");
  std::array<std::uint16_t, kRows> host_baseline_first{};
  std::array<std::uint16_t, kRows> host_baseline_second{};
  std::array<std::uint16_t, kRows> host_candidate_first{};
  std::array<std::uint16_t, kRows> host_candidate_second{};
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           host_baseline_first.data(), baseline_first.get(),
                           sizeof(host_baseline_first), cudaMemcpyDeviceToHost,
                           stream),
                       label + " download baseline A");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           host_baseline_second.data(), baseline_second.get(),
                           sizeof(host_baseline_second),
                           cudaMemcpyDeviceToHost, stream),
                       label + " download baseline B");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           host_candidate_first.data(), candidate_first.get(),
                           sizeof(host_candidate_first),
                           cudaMemcpyDeviceToHost, stream),
                       label + " download candidate A");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(host_candidate_second.data(),
                                       candidate_second.get(),
                                       sizeof(host_candidate_second),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " download candidate B");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " bitwise sync");
  if (!ready) {
    return;
  }
  test.expect(host_candidate_first == host_baseline_first,
              label + " first projection is raw-BF16 identical");
  test.expect(host_candidate_second == host_baseline_second,
              label + " second projection is raw-BF16 identical");

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
  for (int round = 0; round < kRounds; ++round) {
    const std::string round_label =
        label + " round=" + std::to_string(round + 1);
    const float baseline_first_ms = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations,
        round_label + " baseline pass 1", launch_baseline);
    const float candidate_first_ms = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations,
        round_label + " candidate pass 1", launch_candidate);
    const float candidate_second_ms = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations,
        round_label + " candidate pass 2", launch_candidate);
    const float baseline_second_ms = measure_cuda_span_milliseconds(
        test, stream, kMeasuredIterations,
        round_label + " baseline pass 2", launch_baseline);
    const bool round_finite =
        std::isfinite(baseline_first_ms) && baseline_first_ms > 0.0F &&
        std::isfinite(candidate_first_ms) && candidate_first_ms > 0.0F &&
        std::isfinite(candidate_second_ms) && candidate_second_ms > 0.0F &&
        std::isfinite(baseline_second_ms) && baseline_second_ms > 0.0F;
    timing_finite = timing_finite && round_finite;
    if (round_finite) {
      baseline_total += baseline_first_ms + baseline_second_ms;
      candidate_total += candidate_first_ms + candidate_second_ms;
    }
    std::cout << "PERF_BF16_M1_PAIR_ROUND: round=" << round + 1
              << " baseline1_ms=" << baseline_first_ms
              << " candidate1_ms=" << candidate_first_ms
              << " candidate2_ms=" << candidate_second_ms
              << " baseline2_ms=" << baseline_second_ms << '\n';
  }

  constexpr double kTimedPasses = 2.0 * static_cast<double>(kRounds);
  const double baseline_average = baseline_total / kTimedPasses;
  const double candidate_average = candidate_total / kTimedPasses;
  const double speedup = baseline_average / candidate_average;
  const bool gate = timing_finite && std::isfinite(speedup) &&
                    speedup >= kMinimumSpeedup;
  std::cout << "PERF_BF16_M1_PAIR: baseline_two_direct_ms="
            << baseline_average << " candidate_pair_ms=" << candidate_average
            << " speedup=" << speedup
            << " required_speedup=" << kMinimumSpeedup
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, label + " clears the 1.20x span gate");
}

void run_fp8_case(TestContext& test, cudaStream_t stream,
                  const std::size_t rows, const std::size_t columns,
                  const std::string& label) {
  constexpr float kWeightScale = 0.03125F;
  constexpr std::uint8_t kFiniteCodes[] = {
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U, 0x3cU,
      0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU};
  constexpr std::size_t kCodeCount = sizeof(kFiniteCodes) / sizeof(kFiniteCodes[0]);
  std::vector<std::uint16_t> activation(columns);
  std::vector<std::uint8_t> weights(rows * columns);
  for (std::size_t column = 0; column < columns; ++column) {
    const int centered = static_cast<int>((column * 3U) % 19U) - 9;
    activation[column] = encode_bf16(static_cast<float>(centered) / 32.0F);
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = kFiniteCodes[(index * 5U + 1U) % kCodeCount];
  }
  std::vector<float> cpu(rows);
  std::vector<float> gpu(rows, std::numeric_limits<float>::quiet_NaN());
  (void)q3x::kernels::fp8_gemv_reference_cpu(
      weights.data(), kWeightScale, activation.data(), rows, columns,
      cpu.data());

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint8_t> device_weights;
  DeviceBuffer<float> device_output;
  bool ready = test.cuda_ok(device_activation.allocate(activation.size()),
                            label + " allocate activation");
  ready = ready && test.cuda_ok(device_weights.allocate(weights.size()),
                                label + " allocate weights");
  ready = ready && test.cuda_ok(device_output.allocate(gpu.size()),
                                label + " allocate output");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      cudaMemcpyAsync(device_activation.get(), activation.data(),
                      activation.size() * sizeof(activation[0]),
                      cudaMemcpyHostToDevice, stream),
      label + " copy activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_weights.get(), weights.data(),
                                       weights.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy weights");
  if (!ready) {
    return;
  }

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::launch_fp8_gemv_reference_cuda(
          device_weights.get(), kWeightScale, device_activation.get(), rows,
          columns, device_output.get(), static_cast<void*>(stream))),
      label + " launch after stale error");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(gpu.data(), device_output.get(),
                                       gpu.size() * sizeof(gpu[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy output");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (ready) {
    compare_outputs(test, gpu, cpu, columns, label);
  }
}

void run_fp8_static_case(TestContext& test, cudaStream_t stream,
                         const std::size_t rows,
                         const std::size_t columns,
                         const std::string& label) {
  constexpr float kWeightScale = 0.03125F;
  constexpr float kInputScale = 0.00390625F;
  constexpr std::uint8_t kFiniteCodes[] = {
      0x00U, 0x80U, 0x01U, 0x07U, 0x30U, 0x38U, 0x3cU,
      0x40U, 0xb8U, 0x70U, 0x78U, 0x7eU, 0xfeU};
  constexpr std::size_t kCodeCount =
      sizeof(kFiniteCodes) / sizeof(kFiniteCodes[0]);
  std::vector<std::uint16_t> activation(columns);
  std::vector<std::uint8_t> weights(rows * columns);
  for (std::size_t column = 0; column < columns; ++column) {
    const int centered = static_cast<int>((column * 11U) % 257U) - 128;
    activation[column] =
        encode_bf16(static_cast<float>(centered) / 64.0F);
  }
  for (std::size_t index = 0; index < weights.size(); ++index) {
    weights[index] = kFiniteCodes[(index * 7U + 3U) % kCodeCount];
  }
  std::vector<float> cpu(rows);
  std::vector<float> gpu(rows, std::numeric_limits<float>::quiet_NaN());
  (void)q3x::kernels::fp8_static_gemv_reference_cpu(
      weights.data(), kWeightScale, kInputScale, activation.data(), rows,
      columns, cpu.data());

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint8_t> device_weights;
  DeviceBuffer<float> device_output;
  bool ready = test.cuda_ok(device_activation.allocate(activation.size()),
                            label + " allocate activation");
  ready = ready && test.cuda_ok(device_weights.allocate(weights.size()),
                                label + " allocate weights");
  ready = ready && test.cuda_ok(device_output.allocate(gpu.size()),
                                label + " allocate output");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      cudaMemcpyAsync(device_activation.get(), activation.data(),
                      activation.size() * sizeof(activation[0]),
                      cudaMemcpyHostToDevice, stream),
      label + " copy activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_weights.get(), weights.data(),
                                       weights.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy weights");
  if (!ready) {
    return;
  }

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::kernels::launch_fp8_static_gemv_reference_cuda(
              device_weights.get(), kWeightScale, kInputScale,
              device_activation.get(), rows, columns, device_output.get(),
              static_cast<void*>(stream))),
      label + " launch after stale error");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(gpu.data(), device_output.get(),
                                       gpu.size() * sizeof(gpu[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy output");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (ready) {
    compare_outputs(test, gpu, cpu, columns, label);
  }
}

void run_nvfp4_case(TestContext& test, cudaStream_t stream,
                    const std::size_t rows, const std::size_t columns,
                    const std::string& label) {
  constexpr float kScale2 = 0.00390625F;
  std::vector<std::uint16_t> activation(columns);
  std::vector<std::uint8_t> packed(rows * columns / 2U);
  std::vector<std::uint8_t> scales(rows * columns / 16U);
  for (std::size_t column = 0; column < columns; ++column) {
    const int centered = static_cast<int>((column * 5U) % 23U) - 11;
    activation[column] = encode_bf16(static_cast<float>(centered) / 16.0F);
  }
  for (std::size_t index = 0; index < packed.size(); ++index) {
    const std::uint8_t low = static_cast<std::uint8_t>((index * 3U) & 0x0fU);
    const std::uint8_t high = static_cast<std::uint8_t>((index * 7U + 1U) & 0x0fU);
    packed[index] = static_cast<std::uint8_t>(low | (high << 4U));
  }
  constexpr std::uint8_t kScaleCodes[] = {
      0x30U, 0x38U, 0x3cU, 0x40U, 0x44U, 0x48U};
  constexpr std::size_t kScaleCodeCount =
      sizeof(kScaleCodes) / sizeof(kScaleCodes[0]);
  for (std::size_t index = 0; index < scales.size(); ++index) {
    scales[index] = kScaleCodes[(index * 5U + 2U) % kScaleCodeCount];
  }
  std::vector<float> cpu(rows);
  std::vector<float> gpu(rows, std::numeric_limits<float>::quiet_NaN());
  (void)q3x::kernels::nvfp4_gemv_reference_cpu(
      packed.data(), scales.data(), kScale2, activation.data(), rows, columns,
      cpu.data());

  DeviceBuffer<std::uint16_t> device_activation;
  DeviceBuffer<std::uint8_t> device_packed;
  DeviceBuffer<std::uint8_t> device_scales;
  DeviceBuffer<float> device_output;
  bool ready = test.cuda_ok(device_activation.allocate(activation.size()),
                            label + " allocate activation");
  ready = ready && test.cuda_ok(device_packed.allocate(packed.size()),
                                label + " allocate packed weights");
  ready = ready && test.cuda_ok(device_scales.allocate(scales.size()),
                                label + " allocate block scales");
  ready = ready && test.cuda_ok(device_output.allocate(gpu.size()),
                                label + " allocate output");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      cudaMemcpyAsync(device_activation.get(), activation.data(),
                      activation.size() * sizeof(activation[0]),
                      cudaMemcpyHostToDevice, stream),
      label + " copy activation");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_packed.get(), packed.data(),
                                       packed.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy packed weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_scales.get(), scales.data(),
                                       scales.size(), cudaMemcpyHostToDevice,
                                       stream),
                       label + " copy block scales");
  if (!ready) {
    return;
  }

  (void)seed_stale_error(test);
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::launch_nvfp4_gemv_reference_cuda(
          device_packed.get(), device_scales.get(), kScale2,
          device_activation.get(), rows, columns, device_output.get(),
          static_cast<void*>(stream))),
      label + " launch after stale error");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(gpu.data(), device_output.get(),
                                       gpu.size() * sizeof(gpu[0]),
                                       cudaMemcpyDeviceToHost, stream),
                       label + " copy output");
  ready = ready &&
          test.cuda_ok(cudaStreamSynchronize(stream), label + " synchronize");
  if (ready) {
    compare_outputs(test, gpu, cpu, columns, label);
  }
}

void test_bf16_pair_numeric_edges(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kTokenCount = 2U;
  constexpr std::size_t kRows = 6U;
  constexpr std::size_t kColumns = 256U;
  std::vector<std::uint16_t> input(kTokenCount * kColumns, 0x0000U);
  input[0U] = encode_bf16(1.0F);
  input[1U] = encode_bf16(1.0F / 256.0F);
  std::fill(input.begin() + static_cast<std::ptrdiff_t>(kColumns), input.end(),
            0x0080U);

  std::vector<std::uint16_t> first_weights(kRows * kColumns, 0x0000U);
  std::vector<std::uint16_t> second_weights(kRows * kColumns, 0x0000U);
  first_weights[0U * kColumns] = encode_bf16(1.0F);
  first_weights[0U * kColumns + 1U] = encode_bf16(1.0F);
  first_weights[1U * kColumns] = encode_bf16(1.0F);
  first_weights[1U * kColumns + 1U] = encode_bf16(3.0F);
  second_weights[0U * kColumns] = encode_bf16(-1.0F);
  second_weights[0U * kColumns + 1U] = encode_bf16(-1.0F);
  second_weights[1U * kColumns] = encode_bf16(-1.0F);
  second_weights[1U * kColumns + 1U] = encode_bf16(-3.0F);
  std::fill_n(first_weights.data() + 2U * kColumns, kColumns, 0x8080U);
  std::fill_n(first_weights.data() + 3U * kColumns, kColumns, 0x0080U);
  std::fill_n(second_weights.data() + 2U * kColumns, kColumns, 0x0080U);
  std::fill_n(second_weights.data() + 3U * kColumns, kColumns, 0x8080U);
  first_weights[4U * kColumns] = 0x7f81U;
  second_weights[4U * kColumns] = 0xff81U;
  first_weights[5U * kColumns] = 0x7f80U;
  second_weights[5U * kColumns] = 0xff80U;

  DeviceBuffer<std::uint16_t> device_input;
  DeviceBuffer<std::uint16_t> device_first_weights;
  DeviceBuffer<std::uint16_t> device_second_weights;
  DeviceBuffer<std::uint16_t> device_first_output;
  DeviceBuffer<std::uint16_t> device_second_output;
  bool ready = test.cuda_ok(device_input.allocate(input.size()),
                            "BF16 pair numeric allocate input");
  ready = ready && test.cuda_ok(
                       device_first_weights.allocate(first_weights.size()),
                       "BF16 pair numeric allocate first weights");
  ready = ready && test.cuda_ok(
                       device_second_weights.allocate(second_weights.size()),
                       "BF16 pair numeric allocate second weights");
  ready = ready && test.cuda_ok(
                       device_first_output.allocate(kTokenCount * kRows),
                       "BF16 pair numeric allocate first output");
  ready = ready && test.cuda_ok(
                       device_second_output.allocate(kTokenCount * kRows),
                       "BF16 pair numeric allocate second output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_input.get(), input.data(),
                                       input.size() * sizeof(input[0]),
                                       cudaMemcpyHostToDevice, stream),
                       "BF16 pair numeric copy input");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_first_weights.get(), first_weights.data(),
                           first_weights.size() * sizeof(first_weights[0]),
                           cudaMemcpyHostToDevice, stream),
                       "BF16 pair numeric copy first weights");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_second_weights.get(), second_weights.data(),
                           second_weights.size() * sizeof(second_weights[0]),
                           cudaMemcpyHostToDevice, stream),
                       "BF16 pair numeric copy second weights");
  if (!ready) {
    return;
  }

  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              device_first_weights.get(), device_second_weights.get(),
              device_input.get(), kTokenCount, kRows, kColumns,
              device_first_output.get(), device_second_output.get(),
              static_cast<void*>(stream))),
      "BF16 pair numeric launch");
  std::vector<std::uint16_t> first_output(kTokenCount * kRows);
  std::vector<std::uint16_t> second_output(kTokenCount * kRows);
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           first_output.data(), device_first_output.get(),
                           first_output.size() * sizeof(first_output[0]),
                           cudaMemcpyDeviceToHost, stream),
                       "BF16 pair numeric copy first output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           second_output.data(), device_second_output.get(),
                           second_output.size() * sizeof(second_output[0]),
                           cudaMemcpyDeviceToHost, stream),
                       "BF16 pair numeric copy second output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "BF16 pair numeric synchronize");
  if (!ready) {
    return;
  }

  test.expect(first_output[0U] == 0x3f80U,
              "BF16 pair RNE midpoint rounds down to even");
  test.expect(first_output[1U] == 0x3f82U,
              "BF16 pair RNE midpoint rounds up to even");
  test.expect(second_output[0U] == 0xbf80U &&
                  second_output[1U] == 0xbf82U,
              "BF16 pair preserves RNE midpoint behavior for negatives");
  test.expect(first_output[kRows + 2U] == 0x8000U &&
                  first_output[kRows + 3U] == 0x0000U &&
                  second_output[kRows + 2U] == 0x0000U &&
                  second_output[kRows + 3U] == 0x8000U,
              "BF16 pair preserves signed zero from FP32 accumulation");
  test.expect(first_output[5U] == 0x7f80U &&
                  second_output[5U] == 0xff80U,
              "BF16 pair preserves signed infinity");
  const auto is_quiet_nan = [](const std::uint16_t bits) {
    return (bits & 0x7f80U) == 0x7f80U && (bits & 0x007fU) != 0U &&
           (bits & 0x0040U) != 0U;
  };
  test.expect(is_quiet_nan(first_output[4U]) &&
                  is_quiet_nan(second_output[4U]),
              "BF16 pair canonicalizes signaling NaNs to quiet BF16 NaNs");
}

void test_bf16_pair_readonly_alias(TestContext& test, cudaStream_t stream) {
  constexpr std::size_t kColumns = 256U;
  std::vector<std::uint16_t> shared(kColumns, encode_bf16(1.0F));
  DeviceBuffer<std::uint16_t> device_shared;
  DeviceBuffer<std::uint16_t> device_first_output;
  DeviceBuffer<std::uint16_t> device_second_output;
  bool ready = test.cuda_ok(device_shared.allocate(shared.size()),
                            "BF16 pair alias allocate shared input");
  ready = ready && test.cuda_ok(device_first_output.allocate(1U),
                                "BF16 pair alias allocate first output");
  ready = ready && test.cuda_ok(device_second_output.allocate(1U),
                                "BF16 pair alias allocate second output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(device_shared.get(), shared.data(),
                                       shared.size() * sizeof(shared[0]),
                                       cudaMemcpyHostToDevice, stream),
                       "BF16 pair alias copy shared input");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              device_shared.get(), device_shared.get(), device_shared.get(),
              1U, 1U, kColumns, device_first_output.get(),
              device_second_output.get(), static_cast<void*>(stream))),
      "BF16 pair permits exact alias among read-only inputs");
  std::uint16_t first = 0U;
  std::uint16_t second = 0U;
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(&first, device_first_output.get(),
                                       sizeof(first), cudaMemcpyDeviceToHost,
                                       stream),
                       "BF16 pair alias copy first output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(&second, device_second_output.get(),
                                       sizeof(second), cudaMemcpyDeviceToHost,
                                       stream),
                       "BF16 pair alias copy second output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "BF16 pair alias synchronize");
  if (ready) {
    test.expect(first == second && first == encode_bf16(256.0F),
                "BF16 pair read-only alias produces both projections");
  }
}

void test_bf16_direct_readonly_alias(TestContext& test,
                                     cudaStream_t stream) {
  constexpr std::size_t kColumns = 256U;
  std::vector<std::uint16_t> shared(kColumns, encode_bf16(1.0F));
  DeviceBuffer<std::uint16_t> device_shared;
  DeviceBuffer<std::uint16_t> device_output;
  bool ready = test.cuda_ok(device_shared.allocate(shared.size()),
                            "BF16 direct alias allocate shared input");
  ready = ready && test.cuda_ok(device_output.allocate(1U),
                                "BF16 direct alias allocate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           device_shared.get(), shared.data(),
                           shared.size() * sizeof(shared.front()),
                           cudaMemcpyHostToDevice, stream),
                       "BF16 direct alias upload shared input");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_bf16_cuda(
              device_shared.get(), device_shared.get(), 1U, kColumns,
              device_output.get(), static_cast<void*>(stream))),
      "BF16 direct permits alias among read-only inputs");
  std::uint16_t output = 0U;
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(&output, device_output.get(),
                                       sizeof(output), cudaMemcpyDeviceToHost,
                                       stream),
                       "BF16 direct alias download output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "BF16 direct alias synchronize");
  if (ready) {
    test.expect(output == encode_bf16(256.0F),
                "BF16 direct read-only alias produces expected output");
  }
}

void test_launch_validation(TestContext& test) {
  constexpr std::size_t kMaximum = std::numeric_limits<std::size_t>::max();
  constexpr std::size_t kBeyondGridX =
      static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1U;
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_reference_cuda(
                      nullptr, nullptr, 0U, 7U, nullptr)) == cudaSuccess,
              "empty CUDA BF16 shape is a no-op");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_reference_cuda(
                      nullptr, nullptr, 1U, 1U, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA BF16 rejects null non-empty pointers");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_reference_cuda(
                      nullptr, nullptr, kMaximum, 2U, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA BF16 rejects dimension overflow");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_bf16_cuda(
                      nullptr, nullptr, 0U, 7U, nullptr)) == cudaSuccess,
              "empty CUDA direct BF16 shape is a no-op");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_bf16_cuda(
                      nullptr, nullptr, 1U, 1U, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA direct BF16 rejects null non-empty pointers");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_bf16_cuda(
                      nullptr, nullptr, kMaximum, 2U, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA direct BF16 rejects element-count overflow");
  std::uint16_t direct_alias_storage[32]{};
  const auto* const direct_weights = direct_alias_storage;
  const auto* const direct_input = direct_alias_storage + 12U;
  auto* const direct_output = direct_alias_storage + 20U;
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_bf16_cuda(
                      direct_weights, direct_input, 2U, 4U,
                      const_cast<std::uint16_t*>(direct_weights + 1U))) ==
                  cudaErrorInvalidValue,
              "CUDA direct BF16 rejects output/weight overlap");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_bf16_cuda(
                      direct_weights, direct_input, 2U, 4U,
                      const_cast<std::uint16_t*>(direct_input + 1U))) ==
                  cudaErrorInvalidValue,
              "CUDA direct BF16 rejects output/input overlap");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_bf16_cuda(
                      direct_weights, direct_input, 1U,
                      kMaximum / 2U + 1U, direct_output)) ==
                  cudaErrorInvalidValue,
              "CUDA direct BF16 rejects BF16 byte-count overflow");
  const auto* const direct_overflowing_pointer =
      reinterpret_cast<const std::uint16_t*>(
          std::numeric_limits<std::uintptr_t>::max() - 1U);
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_bf16_gemv_bf16_cuda(
                      direct_overflowing_pointer, direct_input, 1U, 2U,
                      direct_output)) == cudaErrorInvalidValue,
              "CUDA direct BF16 rejects pointer range overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              nullptr, nullptr, nullptr, 1U, 0U, 7U, nullptr, nullptr)) ==
          cudaSuccess,
      "empty CUDA BF16 pair shape is a no-op");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              nullptr, nullptr, nullptr, 1U, kBeyondGridX, 0U, nullptr,
              nullptr)) == cudaSuccess,
      "empty CUDA BF16 pair ignores the non-launched grid dimension");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              nullptr, nullptr, nullptr, 0U, 1U, 1U, nullptr, nullptr)) ==
          cudaErrorInvalidValue,
      "CUDA BF16 pair rejects M=0");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              nullptr, nullptr, nullptr, 17U, 1U, 1U, nullptr, nullptr)) ==
          cudaErrorInvalidValue,
      "CUDA BF16 pair rejects M=17");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              nullptr, nullptr, nullptr, 1U, 1U, 1U, nullptr, nullptr)) ==
          cudaErrorInvalidValue,
      "CUDA BF16 pair rejects null non-empty pointers");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              nullptr, nullptr, nullptr, 1U, kMaximum, 2U, nullptr,
              nullptr)) == cudaErrorInvalidValue,
      "CUDA BF16 pair rejects dimension overflow");

  std::uint16_t alias_storage[40]{};
  const auto* const first_weights = alias_storage;
  const auto* const second_weights = alias_storage + 8U;
  const auto* const input = alias_storage + 16U;
  auto* const first_output = alias_storage + 24U;
  auto* const second_output = alias_storage + 28U;
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 2U, 2U, 4U,
              const_cast<std::uint16_t*>(first_weights), second_output)) ==
          cudaErrorInvalidValue,
      "CUDA BF16 pair rejects first output/weight overlap");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 2U, 2U, 4U,
              const_cast<std::uint16_t*>(second_weights + 1U),
              second_output)) == cudaErrorInvalidValue,
      "CUDA BF16 pair rejects cross-projection output/weight overlap");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 2U, 2U, 4U, first_output,
              const_cast<std::uint16_t*>(input + 1U))) ==
          cudaErrorInvalidValue,
      "CUDA BF16 pair rejects second output/input overlap");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 2U, 2U, 4U, first_output,
              first_output + 1U)) == cudaErrorInvalidValue,
      "CUDA BF16 pair rejects partial output/output overlap");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 2U, 2U, 4U, first_output,
              first_output)) == cudaErrorInvalidValue,
      "CUDA BF16 pair rejects exact output/output alias");
  const auto* const overflowing_pointer =
      reinterpret_cast<const std::uint16_t*>(
          std::numeric_limits<std::uintptr_t>::max() - 1U);
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              overflowing_pointer, second_weights, input, 2U, 2U, 4U,
              first_output, second_output)) == cudaErrorInvalidValue,
      "CUDA BF16 pair rejects pointer range overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 1U, kBeyondGridX, 1U,
              first_output, second_output)) == cudaErrorInvalidValue,
      "CUDA BF16 pair rejects grid-x overflow for non-empty shapes");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 1U, 1U,
              kMaximum / 2U + 1U, first_output, second_output)) ==
          cudaErrorInvalidValue,
      "CUDA BF16 pair rejects BF16 byte-count overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 16U, 1U,
              kMaximum / 16U + 1U, first_output, second_output)) ==
          cudaErrorInvalidValue,
      "CUDA BF16 pair rejects M*K overflow");
  test.expect(
      static_cast<cudaError_t>(
          q3x::kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
              first_weights, second_weights, input, 16U,
              kMaximum / 16U + 1U, 1U, first_output, second_output)) ==
          cudaErrorInvalidValue,
      "CUDA BF16 pair rejects M*N overflow");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_fp8_gemv_reference_cuda(
                      nullptr, std::numeric_limits<float>::quiet_NaN(), nullptr,
                      1U, 1U, nullptr)) == cudaErrorInvalidValue,
              "CUDA FP8 rejects NaN scale");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_fp8_static_gemv_reference_cuda(
                      nullptr, 1.0F, 0.0F, nullptr, 1U, 1U, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA static W8A8 rejects zero input scale");
  test.expect(static_cast<cudaError_t>(
                  q3x::kernels::launch_nvfp4_gemv_reference_cuda(
                      nullptr, nullptr, 1.0F, nullptr, 1U, 31U, nullptr)) ==
                  cudaErrorInvalidValue,
              "CUDA NVFP4 rejects non-group-aligned K");
}

void test_nonfinite_cuda(TestContext& test, cudaStream_t stream) {
  const std::uint16_t host_weight = encode_bf16(1.0F);
  const std::uint16_t host_activation =
      encode_bf16(std::numeric_limits<float>::infinity());
  DeviceBuffer<std::uint16_t> weight;
  DeviceBuffer<std::uint16_t> activation;
  DeviceBuffer<float> output;
  bool ready = test.cuda_ok(weight.allocate(1U), "nonfinite allocate weight");
  ready = ready && test.cuda_ok(activation.allocate(1U),
                                "nonfinite allocate activation");
  ready = ready && test.cuda_ok(output.allocate(1U),
                                "nonfinite allocate output");
  ready = ready && test.cuda_ok(cudaMemcpy(weight.get(), &host_weight,
                                           sizeof(host_weight),
                                           cudaMemcpyHostToDevice),
                                "nonfinite copy weight");
  ready = ready && test.cuda_ok(cudaMemcpy(activation.get(), &host_activation,
                                           sizeof(host_activation),
                                           cudaMemcpyHostToDevice),
                                "nonfinite copy activation");
  if (!ready) {
    return;
  }
  ready = test.cuda_ok(
      static_cast<cudaError_t>(q3x::kernels::launch_bf16_gemv_reference_cuda(
          weight.get(), activation.get(), 1U, 1U, output.get(),
          static_cast<void*>(stream))),
      "nonfinite launch");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                "nonfinite synchronize");
  float host_output = 0.0F;
  ready = ready && test.cuda_ok(cudaMemcpy(&host_output, output.get(),
                                           sizeof(host_output),
                                           cudaMemcpyDeviceToHost),
                                "nonfinite copy output");
  if (ready) {
    test.expect(std::isinf(host_output) && host_output > 0.0F,
                "CUDA BF16 infinity propagates");
  }
}

}  // namespace

int main() {
  TestContext test;
  test_launch_validation(test);

  int device_count = 0;
  const cudaError_t device_status = cudaGetDeviceCount(&device_count);
  if (device_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: CUDA reference GEMV tests (no CUDA device)\n";
    (void)cudaGetLastError();
    return test.failures() == 0 ? 0 : 1;
  }

  cudaDeviceProp properties{};
  if (test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                   "read CUDA device properties")) {
    test.expect(properties.major == 8 && properties.minor == 7,
                "reference GEMV runs on the required SM87 device");
    std::cout << "CUDA GEMV device: " << properties.name << " (sm_"
              << properties.major << properties.minor << ")\n";
  }

  cudaStream_t stream = nullptr;
  if (!test.cuda_ok(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking),
                    "create non-blocking stream")) {
    return 1;
  }

  // Awkward shapes exercise tails; target-K cases cover Qwen3.6 projection
  // dimensions without allocating full production matrices in a unit test.
  run_bf16_case(test, stream, 7U, 37U, "BF16 awkward 7x37");
  run_bf16_case(test, stream, 3U, 6144U, "BF16 target-K 3x6144");
  run_bf16_direct_case(test, stream, Bf16DirectFixture::kStructured,
                       "BF16 direct production structured 48x5120");
  run_bf16_direct_case(test, stream, Bf16DirectFixture::kNumericEdges,
                       "BF16 direct production numeric edges 48x5120");
  for (const std::size_t token_count : {1U, 2U, 8U, 16U}) {
    run_bf16_pair_tile_case(
        test, stream, token_count, 7U, 37U,
        "BF16 pair M" + std::to_string(token_count) + " awkward 7x37");
    run_bf16_pair_tile_case(
        test, stream, token_count, 48U, 5120U,
        "BF16 pair M" + std::to_string(token_count) +
            " production 48x5120");
  }
  test_bf16_pair_numeric_edges(test, stream);
  test_bf16_pair_readonly_alias(test, stream);
  test_bf16_direct_readonly_alias(test, stream);
  test_bf16_direct_performance(test, stream);
  test_bf16_m1_pair_performance(test, stream);
  run_fp8_case(test, stream, 7U, 37U, "FP8 awkward 7x37");
  run_fp8_case(test, stream, 3U, 5120U, "FP8 target-K 3x5120");
  run_fp8_static_case(test, stream, 7U, 37U,
                      "static W8A8 awkward 7x37");
  run_fp8_static_case(test, stream, 3U, 5120U,
                      "static W8A8 target-K 3x5120");
  run_nvfp4_case(test, stream, 5U, 48U, "NVFP4 awkward 5x48");
  run_nvfp4_case(test, stream, 3U, 5120U, "NVFP4 target-K 3x5120");
  run_nvfp4_case(test, stream, 248320U, 16U,
                  "NVFP4 lm-head-N grid-stride 248320x16");
  test_nonfinite_cuda(test, stream);

  (void)test.cuda_ok(cudaStreamDestroy(stream),
                     "destroy non-blocking stream");
  if (test.failures() != 0) {
    std::cerr << test.failures() << " CUDA GEMV assertion(s) failed\n";
    return 1;
  }
  std::cout << "CUDA reference GEMV tests passed\n";
  return 0;
}
