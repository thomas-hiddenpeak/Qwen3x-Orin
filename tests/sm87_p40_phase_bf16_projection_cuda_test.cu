#include "q3x/kernels/sm87_p40_phase_bf16_projection.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

namespace {

using q3x::kernels::Sm87P40PhaseBf16ProjectionResources;
using q3x::kernels::Sm87P40PhaseBf16ProjectionRole;

constexpr std::size_t kTokens =
    q3x::kernels::kSm87P40PhaseBf16Tokens;
constexpr float kAlpha = 0.5F;
constexpr std::size_t kCanaryElements = 4'096U;
constexpr std::uint16_t kCanary = 0xa5a5U;
constexpr unsigned int kThreads = 256U;
constexpr const char* kRunEnvironment =
    "Q3X_RUN_SM87_P40_PHASE_BF16_NUMERICAL";

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
    return cudaMalloc(reinterpret_cast<void**>(&data_),
                      count * sizeof(T));
  }

  [[nodiscard]] T* data() noexcept { return data_; }
  [[nodiscard]] const T* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }
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
  std::uint32_t token = 0U;
  std::uint32_t output = 0U;
};

struct ProjectionCase final {
  const char* name = nullptr;
  Sm87P40PhaseBf16ProjectionRole role =
      Sm87P40PhaseBf16ProjectionRole::kFp8K5120N1024;
  std::size_t input_features = 0U;
  std::size_t output_features = 0U;
};

[[nodiscard]] __host__ __device__ int activation_coefficient(
    const unsigned int token, const unsigned int selected_index) noexcept {
  const unsigned int magnitude =
      1U + (3U * token + 5U * selected_index +
            selected_index * selected_index) %
               7U;
  const int sign = token % 5U == 0U ? -1 : 1;
  return sign * static_cast<int>(magnitude);
}

[[nodiscard]] __host__ __device__ int weight_coefficient(
    const unsigned int output, const unsigned int selected_index) noexcept {
  const unsigned int magnitude =
      1U + (5U * output + 3U * selected_index +
            2U * selected_index * selected_index) %
               7U;
  const int sign = output % 7U == 0U ? -1 : 1;
  return sign * static_cast<int>(magnitude);
}

__global__ void initialize_sparse_activations_kernel(
    std::uint16_t* const activations, const std::size_t input_features,
    const std::uint32_t* const selected_k,
    const unsigned int selected_count) {
  const std::size_t linear =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t work = kTokens * selected_count;
  if (linear >= work) {
    return;
  }
  const unsigned int token =
      static_cast<unsigned int>(linear / selected_count);
  const unsigned int selected_index =
      static_cast<unsigned int>(linear % selected_count);
  const float value = static_cast<float>(
      activation_coefficient(token, selected_index));
  activations[static_cast<std::size_t>(token) * input_features +
              selected_k[selected_index]] =
      __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void initialize_sparse_weights_kernel(
    std::uint16_t* const weights, const std::size_t input_features,
    const std::size_t output_features,
    const std::uint32_t* const selected_k,
    const unsigned int selected_count) {
  const std::size_t linear =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t work = output_features * selected_count;
  if (linear >= work) {
    return;
  }
  const unsigned int output =
      static_cast<unsigned int>(linear / selected_count);
  const unsigned int selected_index =
      static_cast<unsigned int>(linear % selected_count);
  const float value = static_cast<float>(
      weight_coefficient(output, selected_index));
  weights[static_cast<std::size_t>(output) * input_features +
          selected_k[selected_index]] =
      __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void gather_samples_kernel(
    const std::uint16_t* const output, const std::size_t output_features,
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
      output[static_cast<std::size_t>(coordinate.token) * output_features +
             coordinate.output];
}

[[nodiscard]] std::uint16_t bf16_rne_bits(const float value) noexcept {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return static_cast<std::uint16_t>(
      (bits + 0x0000'7fffU + ((bits >> 16U) & 1U)) >> 16U);
}

void append_unique(std::vector<std::uint32_t>& values,
                   const std::size_t value,
                   const std::size_t limit) {
  if (value >= limit) {
    return;
  }
  const auto narrowed = static_cast<std::uint32_t>(value);
  if (std::find(values.begin(), values.end(), narrowed) == values.end()) {
    values.push_back(narrowed);
  }
}

[[nodiscard]] std::vector<std::uint32_t> selected_k_positions(
    const std::size_t input_features) {
  std::vector<std::uint32_t> positions;
  positions.reserve(input_features / 64U + 40U);

  // Every K64 pipeline stage owns at least one distinct nonzero product. The
  // rotating offset prevents a duplicated/misaligned A/B stage from being
  // hidden by a constant per-stage coefficient.
  for (std::size_t stage = 0U; stage < input_features / 64U; ++stage) {
    append_unique(positions, stage * 64U + (17U * stage + 7U) % 64U,
                  input_features);
  }

  const auto add_boundary_cluster = [&](const std::size_t center) {
    for (const int delta : {-65, -64, -63, -17, -16, -15, -1, 0, 1,
                            7, 8, 15, 16, 31, 32, 63, 64, 65}) {
      if (delta < 0) {
        const std::size_t distance = static_cast<std::size_t>(-delta);
        if (center >= distance) {
          append_unique(positions, center - distance, input_features);
        }
      } else {
        append_unique(positions,
                      center + static_cast<std::size_t>(delta),
                      input_features);
      }
    }
  };
  add_boundary_cluster(0U);
  add_boundary_cluster(input_features / 2U);
  add_boundary_cluster(input_features - 1U);
  append_unique(positions, input_features - 1U, input_features);
  std::sort(positions.begin(), positions.end());
  return positions;
}

[[nodiscard]] std::vector<std::uint32_t> token_samples() {
  std::vector<std::uint32_t> values;
  for (const std::size_t value :
       {0U, 1U, 3U, 7U, 8U, 15U, 16U, 31U, 32U, 63U, 64U, 65U,
        127U, 128U, 129U, 255U, 256U, 257U, 39'934U, 39'935U,
        39'936U, 39'937U, 39'998U, 39'999U}) {
    append_unique(values, value, kTokens);
  }
  return values;
}

[[nodiscard]] std::vector<std::uint32_t> output_samples(
    const std::size_t output_features) {
  std::vector<std::uint32_t> values;
  for (const std::size_t value :
       {0U, 1U, 6U, 7U, 8U, 9U, 15U, 16U, 31U, 32U, 63U, 64U,
        65U, 127U, 128U, 129U}) {
    append_unique(values, value, output_features);
  }
  for (const int delta : {-129, -128, -127, -65, -64, -63, -9, -8, -1,
                          0, 1}) {
    const std::size_t center = output_features / 2U;
    if (delta < 0) {
      const std::size_t distance = static_cast<std::size_t>(-delta);
      if (center >= distance) {
        append_unique(values, center - distance, output_features);
      }
    } else {
      append_unique(values, center + static_cast<std::size_t>(delta),
                    output_features);
    }
  }
  for (const std::size_t distance :
       {129U, 128U, 127U, 65U, 64U, 63U, 9U, 8U, 1U}) {
    if (output_features >= distance) {
      append_unique(values, output_features - distance, output_features);
    }
  }
  std::sort(values.begin(), values.end());
  return values;
}

[[nodiscard]] std::vector<SampleCoordinate> sample_coordinates(
    const std::size_t output_features) {
  const auto tokens = token_samples();
  const auto outputs = output_samples(output_features);
  std::vector<SampleCoordinate> coordinates;
  coordinates.reserve(tokens.size() * outputs.size());
  for (const std::uint32_t token : tokens) {
    for (const std::uint32_t output : outputs) {
      coordinates.push_back({token, output});
    }
  }
  return coordinates;
}

[[nodiscard]] std::uint16_t expected_output(
    const SampleCoordinate coordinate,
    const std::size_t selected_count) noexcept {
  int sum = 0;
  for (std::size_t selected_index = 0U;
       selected_index < selected_count; ++selected_index) {
    const auto index = static_cast<unsigned int>(selected_index);
    sum += activation_coefficient(coordinate.token, index) *
           weight_coefficient(coordinate.output, index);
  }
  return bf16_rne_bits(static_cast<float>(sum) * kAlpha);
}

[[nodiscard]] bool exact_resource_receipt(
    TestContext& test, const ProjectionCase& projection) {
  Sm87P40PhaseBf16ProjectionResources resources{};
  const int status =
      q3x::kernels::query_sm87_p40_phase_bf16_projection_resources_cuda(
          projection.role, kTokens, &resources);
  test.expect(status == static_cast<int>(cudaSuccess),
              std::string(projection.name) + " resource query failed: " +
                  cudaGetErrorString(static_cast<cudaError_t>(status)));
  if (status != static_cast<int>(cudaSuccess)) {
    return false;
  }
  const bool exact =
      resources.valid() && resources.compute_major == 8 &&
      resources.compute_minor == 7 && resources.sm_count == 16 &&
      resources.binary_version == 87 &&
      resources.registers_per_thread == 158 &&
      resources.static_shared_bytes == 0U &&
      resources.dynamic_shared_bytes == 73'728U &&
      resources.local_bytes == 0U && resources.active_blocks_per_sm == 1;
  test.expect(exact, std::string(projection.name) +
                         " resource receipt drifted from "
                         "SM87/SM16/REG158/SHARED73728/LOCAL0/CTA1");
  return exact;
}

[[nodiscard]] bool initialize_case(
    TestContext& test, const ProjectionCase& projection,
    const std::vector<std::uint32_t>& selected_k,
    DeviceBuffer<std::uint16_t>& activations,
    DeviceBuffer<std::uint16_t>& weights,
    DeviceBuffer<std::uint16_t>& output,
    DeviceBuffer<std::uint32_t>& device_selected_k,
    const cudaStream_t stream) {
  const std::size_t activation_elements =
      kTokens * projection.input_features;
  const std::size_t weight_elements =
      projection.output_features * projection.input_features;
  const std::size_t output_elements =
      kTokens * projection.output_features;
  if (!test.cuda_ok(activations.allocate(activation_elements),
                    std::string(projection.name) + " allocate A") ||
      !test.cuda_ok(weights.allocate(weight_elements),
                    std::string(projection.name) + " allocate W") ||
      !test.cuda_ok(output.allocate(output_elements + kCanaryElements),
                    std::string(projection.name) + " allocate C+canary") ||
      !test.cuda_ok(device_selected_k.allocate(selected_k.size()),
                    std::string(projection.name) + " allocate K positions")) {
    return false;
  }

  if (!test.cuda_ok(cudaMemsetAsync(activations.data(), 0,
                                    activations.bytes(), stream),
                    std::string(projection.name) + " zero A") ||
      !test.cuda_ok(cudaMemsetAsync(weights.data(), 0, weights.bytes(),
                                    stream),
                    std::string(projection.name) + " zero W") ||
      !test.cuda_ok(cudaMemsetAsync(output.data(), 0xa5, output.bytes(),
                                    stream),
                    std::string(projection.name) + " seed C canary") ||
      !test.cuda_ok(cudaMemcpyAsync(
                        device_selected_k.data(), selected_k.data(),
                        selected_k.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice, stream),
                    std::string(projection.name) + " upload K positions")) {
    return false;
  }

  const auto selected_count =
      static_cast<unsigned int>(selected_k.size());
  const std::size_t activation_work = kTokens * selected_k.size();
  const std::size_t weight_work =
      projection.output_features * selected_k.size();
  initialize_sparse_activations_kernel
      <<<static_cast<unsigned int>((activation_work + kThreads - 1U) /
                                   kThreads),
         kThreads, 0U, stream>>>(activations.data(), projection.input_features,
                                 device_selected_k.data(), selected_count);
  if (!test.cuda_ok(cudaGetLastError(),
                    std::string(projection.name) + " initialize A launch")) {
    return false;
  }
  initialize_sparse_weights_kernel
      <<<static_cast<unsigned int>((weight_work + kThreads - 1U) / kThreads),
         kThreads, 0U, stream>>>(
          weights.data(), projection.input_features,
          projection.output_features, device_selected_k.data(),
          selected_count);
  return test.cuda_ok(cudaGetLastError(),
                      std::string(projection.name) + " initialize W launch");
}

[[nodiscard]] bool validate_samples_and_canary(
    TestContext& test, const ProjectionCase& projection,
    const std::size_t selected_count,
    const DeviceBuffer<std::uint16_t>& output,
    const cudaStream_t stream) {
  const auto coordinates = sample_coordinates(projection.output_features);
  DeviceBuffer<SampleCoordinate> device_coordinates;
  DeviceBuffer<std::uint16_t> device_samples;
  if (!test.cuda_ok(device_coordinates.allocate(coordinates.size()),
                    std::string(projection.name) +
                        " allocate sample coordinates") ||
      !test.cuda_ok(device_samples.allocate(coordinates.size()),
                    std::string(projection.name) +
                        " allocate sample results")) {
    return false;
  }
  if (!test.cuda_ok(cudaMemcpyAsync(
                        device_coordinates.data(), coordinates.data(),
                        coordinates.size() * sizeof(SampleCoordinate),
                        cudaMemcpyHostToDevice, stream),
                    std::string(projection.name) +
                        " upload sample coordinates")) {
    return false;
  }
  gather_samples_kernel
      <<<static_cast<unsigned int>((coordinates.size() + kThreads - 1U) /
                                   kThreads),
         kThreads, 0U, stream>>>(
          output.data(), projection.output_features,
          device_coordinates.data(), coordinates.size(),
          device_samples.data());
  if (!test.cuda_ok(cudaGetLastError(),
                    std::string(projection.name) + " gather samples launch")) {
    return false;
  }

  std::vector<std::uint16_t> actual(coordinates.size(), 0U);
  std::array<std::uint16_t, kCanaryElements> canary{};
  const std::size_t output_elements =
      kTokens * projection.output_features;
  if (!test.cuda_ok(cudaMemcpyAsync(
                        actual.data(), device_samples.data(),
                        device_samples.bytes(), cudaMemcpyDeviceToHost,
                        stream),
                    std::string(projection.name) + " download samples") ||
      !test.cuda_ok(cudaMemcpyAsync(
                        canary.data(), output.data() + output_elements,
                        canary.size() * sizeof(std::uint16_t),
                        cudaMemcpyDeviceToHost, stream),
                    std::string(projection.name) + " download canary") ||
      !test.cuda_ok(cudaStreamSynchronize(stream),
                    std::string(projection.name) + " synchronize")) {
    return false;
  }

  std::size_t mismatches = 0U;
  for (std::size_t index = 0U; index < coordinates.size(); ++index) {
    const std::uint16_t expected =
        expected_output(coordinates[index], selected_count);
    if (actual[index] != expected) {
      if (mismatches < 16U) {
        std::cerr << "FAIL: " << projection.name << " C["
                  << coordinates[index].token << ','
                  << coordinates[index].output << "] expected=0x"
                  << std::hex << expected << " actual=0x" << actual[index]
                  << std::dec << '\n';
      }
      ++mismatches;
    }
  }
  test.expect(mismatches == 0U,
              std::string(projection.name) + " sampled oracle mismatches=" +
                  std::to_string(mismatches));
  const auto damaged =
      std::find_if(canary.begin(), canary.end(),
                   [](const std::uint16_t value) { return value != kCanary; });
  test.expect(damaged == canary.end(),
              std::string(projection.name) + " output tail canary changed");
  return mismatches == 0U && damaged == canary.end();
}

void run_projection_case(TestContext& test,
                         const ProjectionCase& projection,
                         const cudaStream_t stream) {
  if (!exact_resource_receipt(test, projection)) {
    return;
  }
  const auto selected_k = selected_k_positions(projection.input_features);
  test.expect(!selected_k.empty(),
              std::string(projection.name) + " selected no K positions");
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> weights;
  DeviceBuffer<std::uint16_t> output;
  DeviceBuffer<std::uint32_t> device_selected_k;
  if (!initialize_case(test, projection, selected_k, activations, weights,
                       output, device_selected_k, stream)) {
    return;
  }

  const int launch_status =
      q3x::kernels::launch_sm87_p40_phase_bf16_projection_cuda(
          projection.role, activations.data(), weights.data(), kTokens,
          kAlpha, output.data(), stream);
  test.expect(launch_status == static_cast<int>(cudaSuccess),
              std::string(projection.name) + " projection launch failed: " +
                  cudaGetErrorString(
                      static_cast<cudaError_t>(launch_status)));
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return;
  }
  (void)validate_samples_and_canary(test, projection, selected_k.size(),
                                    output, stream);
}

[[nodiscard]] const std::uint16_t* const_pointer(
    const std::uintptr_t address) noexcept {
  return reinterpret_cast<const std::uint16_t*>(address);
}

[[nodiscard]] std::uint16_t* mutable_pointer(
    const std::uintptr_t address) noexcept {
  return reinterpret_cast<std::uint16_t*>(address);
}

void test_invalid_alpha_contracts(TestContext& test) {
  constexpr std::uintptr_t kActivations = 0x0000'0010'0000'0000ULL;
  constexpr std::uintptr_t kWeights = 0x0000'0020'0000'0000ULL;
  constexpr std::uintptr_t kOutput = 0x0000'0040'0000'0000ULL;
  const auto launch = [&](const float alpha) {
    return q3x::kernels::launch_sm87_p40_phase_bf16_projection_cuda(
        Sm87P40PhaseBf16ProjectionRole::kDownK17408N5120,
        const_pointer(kActivations), const_pointer(kWeights), kTokens, alpha,
        mutable_pointer(kOutput), nullptr);
  };
  for (const float invalid :
       {std::numeric_limits<float>::quiet_NaN(),
        std::numeric_limits<float>::infinity(),
        -std::numeric_limits<float>::infinity(), -0.5F}) {
    test.expect(launch(invalid) == static_cast<int>(cudaErrorInvalidValue),
                "NaN/Inf/negative alpha must fail before enqueue");
  }
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
    std::cerr << "cudaGetDevice failed: " << cudaGetErrorString(status)
              << '\n';
    return 1;
  }
  cudaDeviceProp properties{};
  status = cudaGetDeviceProperties(&properties, device);
  if (status != cudaSuccess) {
    std::cerr << "cudaGetDeviceProperties failed: "
              << cudaGetErrorString(status) << '\n';
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: exact SM87 device required\n";
    return 77;
  }

  TestContext test;
  test_invalid_alpha_contracts(test);
  Stream stream;
  if (!test.cuda_ok(stream.create(), "create nonblocking stream")) {
    return 1;
  }
  constexpr std::array<ProjectionCase, 4U> kCases{{
      {"K5120-N1024", Sm87P40PhaseBf16ProjectionRole::kFp8K5120N1024,
       5'120U, 1'024U},
      {"K6144-N5120", Sm87P40PhaseBf16ProjectionRole::kFp8K6144N5120,
       6'144U, 5'120U},
      {"K5120-N17408",
       Sm87P40PhaseBf16ProjectionRole::kGateOrUpK5120N17408, 5'120U,
       17'408U},
      {"K17408-N5120", Sm87P40PhaseBf16ProjectionRole::kDownK17408N5120,
       17'408U, 5'120U},
  }};
  for (const ProjectionCase& projection : kCases) {
    run_projection_case(test, projection, stream.get());
  }

  if (test.failures != 0) {
    std::cerr << test.failures
              << " exact-P40000 BF16 projection CUDA checks failed\n";
    return 1;
  }
  std::cout << "exact-P40000 BF16 projection CUDA checks passed\n";
  return 0;
}
