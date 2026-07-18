#include "q3x/kernels/reference_gemv.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

void test_launch_validation(TestContext& test) {
  constexpr std::size_t kMaximum = std::numeric_limits<std::size_t>::max();
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
