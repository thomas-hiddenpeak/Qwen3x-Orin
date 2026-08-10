#include "q3x/kernels/sm87_phase_local_weight_expansion.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using q3x::kernels::Sm87PhaseLocalWeightExpansionCapability;
using q3x::kernels::Sm87PhaseLocalWeightExpansionPlan;
using q3x::kernels::Sm87PhaseLocalWeightExpansionResources;
using q3x::kernels::Sm87PhaseLocalWeightRole;

constexpr Sm87PhaseLocalWeightExpansionCapability kCapability{
    8, 7, true, true, true};
constexpr std::size_t kCanaryElements = 4'096U;
constexpr std::uint16_t kCanary = 0xa5a5U;
constexpr unsigned int kThreads = 256U;
constexpr const char* kRunEnvironment =
    "Q3X_RUN_SM87_PHASE_LOCAL_WEIGHT_EXPANSION_NUMERICAL";

struct TestContext final {
  int failures = 0;

  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const cudaError_t status,
                             const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }
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

  [[nodiscard]] cudaError_t allocate(const std::size_t count) noexcept {
    if (data_ != nullptr ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return cudaErrorInvalidValue;
    }
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T));
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept {
    return count_ * sizeof(T);
  }

 private:
  T* data_ = nullptr;
  std::size_t count_ = 0U;
};

class Stream final {
 public:
  Stream() = default;
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;

  ~Stream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }

  [[nodiscard]] cudaError_t create() noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
  }

  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

struct SampleCoordinate final {
  std::uint32_t row = 0U;
  std::uint32_t column = 0U;
};

static_assert(sizeof(SampleCoordinate) == 2U * sizeof(std::uint32_t));

[[nodiscard]] unsigned int grid_blocks(const std::size_t work) noexcept {
  return static_cast<unsigned int>((work + kThreads - 1U) / kThreads);
}

__global__ void initialize_nvfp4_weights_kernel(
    std::uint8_t* const packed_weights, const std::size_t packed_bytes,
    const std::size_t input_features) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= packed_bytes) {
    return;
  }
  const std::size_t packed_per_row = input_features / 2U;
  const unsigned int packed_column =
      static_cast<unsigned int>(index % packed_per_row);
  const std::uint8_t low =
      static_cast<std::uint8_t>((2U * packed_column) & 0x0fU);
  const std::uint8_t high =
      static_cast<std::uint8_t>((2U * packed_column + 1U) & 0x0fU);
  packed_weights[index] =
      static_cast<std::uint8_t>(low | static_cast<std::uint8_t>(high << 4U));
}

__global__ void initialize_nvfp4_scales_kernel(
    std::uint8_t* const block_scales, const std::size_t scale_bytes,
    const std::size_t input_features) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= scale_bytes) {
    return;
  }
  const std::size_t scales_per_row = input_features / 16U;
  block_scales[index] =
      static_cast<std::uint8_t>((index % scales_per_row) & 0xffU);
}

__global__ void initialize_fp8_weights_kernel(
    std::uint8_t* const weights, const std::size_t elements,
    const std::size_t input_features) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= elements) {
    return;
  }
  weights[index] =
      static_cast<std::uint8_t>((index % input_features) & 0xffU);
}

__global__ void gather_samples_kernel(
    const std::uint16_t* const expanded,
    const std::size_t input_features,
    const SampleCoordinate* const coordinates,
    const std::size_t coordinate_count,
    std::uint16_t* const samples) {
  const std::size_t index =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= coordinate_count) {
    return;
  }
  const SampleCoordinate coordinate = coordinates[index];
  samples[index] =
      expanded[static_cast<std::size_t>(coordinate.row) * input_features +
               coordinate.column];
}

void append_coordinate(std::vector<SampleCoordinate>& coordinates,
                       const std::size_t row, const std::size_t column,
                       const Sm87PhaseLocalWeightExpansionPlan& plan) {
  if (row < plan.output_features && column < plan.input_features) {
    coordinates.push_back(
        {static_cast<std::uint32_t>(row),
         static_cast<std::uint32_t>(column)});
  }
}

void append_boundary_coordinates(
    std::vector<SampleCoordinate>& coordinates,
    const Sm87PhaseLocalWeightExpansionPlan& plan) {
  const std::array<std::size_t, 9U> rows{
      0U, 1U, 7U, 8U, 127U, 128U, plan.output_features / 2U,
      plan.output_features - 2U, plan.output_features - 1U};
  const std::array<std::size_t, 25U> columns{
      0U,
      1U,
      7U,
      8U,
      15U,
      16U,
      17U,
      31U,
      32U,
      63U,
      64U,
      127U,
      128U,
      255U,
      256U,
      511U,
      512U,
      1'023U,
      1'024U,
      4'095U,
      4'096U,
      plan.input_features / 2U,
      plan.input_features - 3U,
      plan.input_features - 2U,
      plan.input_features - 1U};
  for (const std::size_t row : rows) {
    for (const std::size_t column : columns) {
      append_coordinate(coordinates, row, column, plan);
    }
  }
}

[[nodiscard]] std::vector<SampleCoordinate> nvfp4_coordinates(
    const Sm87PhaseLocalWeightExpansionPlan& plan) {
  std::vector<SampleCoordinate> coordinates;
  coordinates.reserve(4'096U + 9U * 25U);
  // In row zero, columns [0, 4096) are exactly the Cartesian product of all
  // 16 E2M1 codes and all 256 E4M3FN block-scale codes.
  for (std::size_t column = 0U; column < 4'096U; ++column) {
    append_coordinate(coordinates, 0U, column, plan);
  }
  append_boundary_coordinates(coordinates, plan);
  return coordinates;
}

[[nodiscard]] std::vector<SampleCoordinate> fp8_coordinates(
    const Sm87PhaseLocalWeightExpansionPlan& plan) {
  std::vector<SampleCoordinate> coordinates;
  coordinates.reserve(256U + 9U * 25U);
  // Row zero columns [0, 256) cover every E4M3FN encoding, including both
  // NaNs, before adding row/K-boundary witnesses.
  for (std::size_t column = 0U; column < 256U; ++column) {
    append_coordinate(coordinates, 0U, column, plan);
  }
  append_boundary_coordinates(coordinates, plan);
  return coordinates;
}

[[nodiscard]] bool exact_resource_receipt(
    TestContext& test, const Sm87PhaseLocalWeightRole role,
    const std::string& name) {
  Sm87PhaseLocalWeightExpansionResources resources{};
  const int status =
      q3x::kernels::query_sm87_phase_local_weight_expansion_resources_cuda(
          role, &resources);
  test.expect(status == static_cast<int>(cudaSuccess),
              name + " resource query failed: " +
                  cudaGetErrorString(static_cast<cudaError_t>(status)));
  if (status != static_cast<int>(cudaSuccess)) {
    return false;
  }
  const bool exact =
      resources.valid() && resources.role == role &&
      resources.compute_major == 8 && resources.compute_minor == 7 &&
      resources.binary_version == 87 &&
      resources.registers_per_thread ==
          q3x::kernels::sm87_phase_local_weight_expected_registers(role) &&
      resources.static_shared_bytes == 0U &&
      resources.dynamic_shared_bytes == 0U && resources.local_bytes == 0U &&
      resources.active_blocks_per_sm == 6;
  test.expect(exact, name + " expansion resource receipt drifted");
  return exact;
}

template <typename Expected>
[[nodiscard]] bool validate_samples_and_canary(
    TestContext& test, const std::string& name,
    const Sm87PhaseLocalWeightExpansionPlan& plan,
    const std::vector<SampleCoordinate>& coordinates,
    const DeviceBuffer<std::uint16_t>& scratch,
    const cudaStream_t stream, Expected expected) {
  DeviceBuffer<SampleCoordinate> device_coordinates;
  DeviceBuffer<std::uint16_t> device_samples;
  if (!test.cuda_ok(device_coordinates.allocate(coordinates.size()),
                    name + " allocate coordinates") ||
      !test.cuda_ok(device_samples.allocate(coordinates.size()),
                    name + " allocate samples")) {
    return false;
  }
  if (!test.cuda_ok(cudaMemcpyAsync(
                        device_coordinates.data(), coordinates.data(),
                        coordinates.size() * sizeof(SampleCoordinate),
                        cudaMemcpyHostToDevice, stream),
                    name + " upload coordinates")) {
    return false;
  }
  gather_samples_kernel
      <<<grid_blocks(coordinates.size()), kThreads, 0U, stream>>>(
          scratch.data(), plan.input_features, device_coordinates.data(),
          coordinates.size(), device_samples.data());
  if (!test.cuda_ok(cudaGetLastError(), name + " gather samples launch")) {
    return false;
  }

  std::vector<std::uint16_t> actual(coordinates.size(), 0U);
  std::array<std::uint16_t, kCanaryElements> canary{};
  if (!test.cuda_ok(cudaMemcpyAsync(
                        actual.data(), device_samples.data(),
                        device_samples.bytes(), cudaMemcpyDeviceToHost,
                        stream),
                    name + " download samples") ||
      !test.cuda_ok(cudaMemcpyAsync(
                        canary.data(),
                        scratch.data() + plan.scratch_elements,
                        canary.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost, stream),
                    name + " download canary") ||
      !test.cuda_ok(cudaStreamSynchronize(stream), name + " synchronize")) {
    return false;
  }

  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < coordinates.size(); ++index) {
    const std::uint16_t reference = expected(coordinates[index]);
    if (actual[index] != reference) {
      if (mismatches < 16U) {
        std::cerr << "FAIL: " << name << " expanded["
                  << coordinates[index].row << ','
                  << coordinates[index].column << "] expected=0x" << std::hex
                  << reference << " actual=0x" << actual[index] << std::dec
                  << '\n';
      }
      ++mismatches;
    }
  }
  test.expect(mismatches == 0U,
              name + " sampled oracle mismatches=" +
                  std::to_string(mismatches));
  const auto damaged =
      std::find_if(canary.begin(), canary.end(),
                   [](const std::uint16_t value) { return value != kCanary; });
  test.expect(damaged == canary.end(), name + " scratch canary changed");
  return mismatches == 0U && damaged == canary.end();
}

void run_nvfp4_case(TestContext& test,
                    const Sm87PhaseLocalWeightRole role,
                    const std::size_t expected_input_features,
                    const std::string& name, const cudaStream_t stream) {
  const Sm87PhaseLocalWeightExpansionPlan plan =
      q3x::kernels::sm87_phase_local_weight_expansion_plan(role, kCapability);
  test.expect(plan.valid() && plan.input_features == expected_input_features,
              name + " plan is invalid");
  if (!plan.valid() || !exact_resource_receipt(test, role, name)) {
    return;
  }

  DeviceBuffer<std::uint8_t> packed_weights;
  DeviceBuffer<std::uint8_t> block_scales;
  DeviceBuffer<std::uint16_t> scratch;
  if (!test.cuda_ok(packed_weights.allocate(plan.canonical_weight_bytes),
                    name + " allocate packed weights") ||
      !test.cuda_ok(block_scales.allocate(plan.block_scale_bytes),
                    name + " allocate block scales") ||
      !test.cuda_ok(
          scratch.allocate(plan.scratch_elements + kCanaryElements),
          name + " allocate scratch")) {
    return;
  }
  if (!test.cuda_ok(cudaMemsetAsync(scratch.data(), 0xa5, scratch.bytes(),
                                    stream),
                    name + " seed scratch canary")) {
    return;
  }
  initialize_nvfp4_weights_kernel
      <<<grid_blocks(plan.canonical_weight_bytes), kThreads, 0U, stream>>>(
          packed_weights.data(), plan.canonical_weight_bytes,
          plan.input_features);
  if (!test.cuda_ok(cudaGetLastError(), name + " initialize weights launch")) {
    return;
  }
  initialize_nvfp4_scales_kernel
      <<<grid_blocks(plan.block_scale_bytes), kThreads, 0U, stream>>>(
          block_scales.data(), plan.block_scale_bytes, plan.input_features);
  if (!test.cuda_ok(cudaGetLastError(), name + " initialize scales launch")) {
    return;
  }

  const int launch_status =
      q3x::kernels::launch_sm87_phase_local_nvfp4_weight_expansion_test_cuda(
          plan, packed_weights.data(), block_scales.data(), scratch.data(),
          plan.scratch_bytes, stream);
  test.expect(launch_status == static_cast<int>(cudaSuccess),
              name + " expansion launch failed: " +
                  cudaGetErrorString(
                      static_cast<cudaError_t>(launch_status)));
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return;
  }

  const auto coordinates = nvfp4_coordinates(plan);
  (void)validate_samples_and_canary(
      test, name, plan, coordinates, scratch, stream,
      [input_features = plan.input_features](
          const SampleCoordinate coordinate) noexcept {
        const std::size_t packed_column = coordinate.column / 2U;
        const std::uint8_t low = static_cast<std::uint8_t>(
            (2U * packed_column) & 0x0fU);
        const std::uint8_t high = static_cast<std::uint8_t>(
            (2U * packed_column + 1U) & 0x0fU);
        const std::uint8_t packed = static_cast<std::uint8_t>(
            low | static_cast<std::uint8_t>(high << 4U));
        const std::uint8_t scale = static_cast<std::uint8_t>(
            ((coordinate.column / 16U) % (input_features / 16U)) & 0xffU);
        return q3x::kernels::
            sm87_phase_local_nvfp4_expanded_bf16_reference(
                packed, (coordinate.column & 1U) != 0U, scale);
      });
}

void run_fp8_case(TestContext& test, const Sm87PhaseLocalWeightRole role,
                  const std::size_t expected_input_features,
                  const std::string& name, const cudaStream_t stream) {
  const Sm87PhaseLocalWeightExpansionPlan plan =
      q3x::kernels::sm87_phase_local_weight_expansion_plan(role, kCapability);
  test.expect(plan.valid() && plan.input_features == expected_input_features,
              name + " plan is invalid");
  if (!plan.valid() || !exact_resource_receipt(test, role, name)) {
    return;
  }

  DeviceBuffer<std::uint8_t> weights;
  DeviceBuffer<std::uint16_t> scratch;
  if (!test.cuda_ok(weights.allocate(plan.canonical_weight_bytes),
                    name + " allocate weights") ||
      !test.cuda_ok(
          scratch.allocate(plan.scratch_elements + kCanaryElements),
          name + " allocate scratch")) {
    return;
  }
  if (!test.cuda_ok(cudaMemsetAsync(scratch.data(), 0xa5, scratch.bytes(),
                                    stream),
                    name + " seed scratch canary")) {
    return;
  }
  initialize_fp8_weights_kernel
      <<<grid_blocks(plan.canonical_weight_bytes), kThreads, 0U, stream>>>(
          weights.data(), plan.canonical_weight_bytes, plan.input_features);
  if (!test.cuda_ok(cudaGetLastError(), name + " initialize weights launch")) {
    return;
  }

  const int launch_status =
      q3x::kernels::launch_sm87_phase_local_fp8_weight_expansion_test_cuda(
          plan, weights.data(), scratch.data(), plan.scratch_bytes, stream);
  test.expect(launch_status == static_cast<int>(cudaSuccess),
              name + " expansion launch failed: " +
                  cudaGetErrorString(
                      static_cast<cudaError_t>(launch_status)));
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return;
  }

  const auto coordinates = fp8_coordinates(plan);
  (void)validate_samples_and_canary(
      test, name, plan, coordinates, scratch, stream,
      [](const SampleCoordinate coordinate) noexcept {
        return q3x::kernels::sm87_phase_local_fp8_expanded_bf16_reference(
            static_cast<std::uint8_t>(coordinate.column & 0xffU));
      });
}

[[nodiscard]] bool explicitly_enabled() noexcept {
  const char* const value = std::getenv(kRunEnvironment);
  return value != nullptr && std::strcmp(value, "1") == 0;
}

}  // namespace

int main() {
  if (!explicitly_enabled()) {
    std::cout << "SKIP: set " << kRunEnvironment
              << "=1 only after clean tegrastats/process/GPU-handle preflight\n";
    return 77;
  }

  int device_count = 0;
  cudaError_t status = cudaGetDeviceCount(&device_count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver ||
      device_count == 0) {
    std::cout << "SKIP: CUDA device unavailable\n";
    return 77;
  }
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceCount failed: " << cudaGetErrorString(status)
              << '\n';
    return 1;
  }
  int device = 0;
  status = cudaGetDevice(&device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(status) << '\n';
    return 1;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed: "
              << cudaGetErrorString(status) << '\n';
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: exact 16-SM SM87 target required\n";
    return 77;
  }

  TestContext test;
  Stream stream;
  if (!test.cuda_ok(stream.create(), "create nonblocking stream")) {
    return 1;
  }
  run_nvfp4_case(test, Sm87PhaseLocalWeightRole::kNvFp4Gate, 5'120U,
                  "NVFP4-K5120", stream.get());
  run_nvfp4_case(test, Sm87PhaseLocalWeightRole::kNvFp4Down, 17'408U,
                  "NVFP4-K17408", stream.get());
  run_fp8_case(test, Sm87PhaseLocalWeightRole::kFp8FullKey, 5'120U,
               "FP8-K5120", stream.get());
  run_fp8_case(test, Sm87PhaseLocalWeightRole::kFp8LinearOutput, 6'144U,
               "FP8-K6144", stream.get());

  if (test.failures != 0) {
    std::cerr << test.failures
              << " phase-local expansion CUDA checks failed\n";
    return 1;
  }
  std::cout << "phase-local expansion CUDA checks passed\n";
  return 0;
}
