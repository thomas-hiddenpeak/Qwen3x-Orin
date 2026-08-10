#include "q3x/kernels/sm87_p40_packed_projection.h"

#include <cuda_bf16.h>
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

using q3x::kernels::Sm87P40PackedCanonicalSource;
using q3x::kernels::Sm87P40PackedLogicalRole;
using q3x::kernels::Sm87P40PackedProjectionDeviceView;
using q3x::kernels::Sm87P40PackedProjectionPlan;
using q3x::kernels::Sm87P40PackedProjectionResources;
using q3x::kernels::Sm87P40PackedProjectionRole;
using q3x::kernels::Sm87P40PackedTactic;

inline constexpr std::size_t kTokens =
    q3x::kernels::kSm87P40PackedProjectionTokens;
inline constexpr std::size_t kMaximumSources =
    q3x::kernels::kSm87P40PackedProjectionMaximumSources;
inline constexpr std::size_t kPayloadGuard =
    q3x::kernels::kSm87P40PackedProjectionPayloadAlignment;
inline constexpr std::size_t kOutputGuardElements = 4'096U;
inline constexpr std::uint8_t kPayloadCanary = 0xcdU;
inline constexpr std::uint16_t kOutputCanary = 0xa5a5U;
inline constexpr unsigned int kThreads = 256U;
inline constexpr unsigned int kWarpSize = 32U;
inline constexpr const char* kRunEnvironment =
    "Q3X_RUN_SM87_P40_PACKED_FP8_NUMERICAL";

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
    if (data_ != nullptr || count == 0U ||
        count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      return cudaErrorInvalidValue;
    }
    count_ = count;
    return cudaMalloc(reinterpret_cast<void**>(&data_), bytes());
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

[[nodiscard]] __host__ __device__ int activation_sign(
    const unsigned int token, const unsigned int selected_index) noexcept {
  return ((3U * token + 5U * selected_index +
           selected_index * selected_index) &
          1U) != 0U
             ? -1
             : 1;
}

[[nodiscard]] __host__ __device__ int weight_sign(
    const unsigned int source_index, const unsigned int output,
    const unsigned int selected_index) noexcept {
  return ((7U * source_index + 5U * output + 3U * selected_index +
           output * selected_index) &
          1U) != 0U
             ? -1
             : 1;
}

[[nodiscard]] __host__ __device__ std::uint8_t fp8_one_code(
    const int sign) noexcept {
  return sign < 0 ? 0xb8U : 0x38U;
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
  const float value =
      static_cast<float>(activation_sign(token, selected_index));
  activations[static_cast<std::size_t>(token) * input_features +
              selected_k[selected_index]] =
      __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__global__ void initialize_sparse_fp8_weights_kernel(
    std::uint8_t* const weights, const std::size_t input_features,
    const std::size_t output_features,
    const std::uint32_t* const selected_k,
    const unsigned int selected_count, const unsigned int source_index) {
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
  weights[static_cast<std::size_t>(output) * input_features +
          selected_k[selected_index]] = fp8_one_code(
      weight_sign(source_index, output, selected_index));
}

__global__ void gather_samples_kernel(
    const std::uint16_t* const output, const std::size_t output_features,
    const SampleCoordinate* const coordinates,
    const std::size_t coordinate_count, std::uint16_t* const samples) {
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
  positions.reserve(input_features / 16U);
  for (std::size_t k16 = 0U; k16 < input_features / 16U; ++k16) {
    append_unique(positions, k16 * 16U + (7U * k16 + 3U) % 16U,
                  input_features);
  }
  return positions;
}

[[nodiscard]] std::vector<std::uint32_t> token_samples() {
  std::vector<std::uint32_t> values;
  for (const std::size_t value :
       {0U, 1U, 7U, 8U, 15U, 16U, 31U, 32U, 63U, 64U, 65U,
        127U, 128U, 255U, 256U, 39'934U, 39'935U, 39'936U,
        39'937U, 39'998U, 39'999U}) {
    append_unique(values, value, kTokens);
  }
  return values;
}

[[nodiscard]] std::vector<std::uint32_t> output_samples(
    const std::size_t output_features) {
  std::vector<std::uint32_t> values;
  for (const std::size_t value :
       {0U, 1U, 7U, 8U, 15U, 16U, 31U, 32U, 63U, 64U, 65U,
        127U, 128U, 129U, 255U, 256U}) {
    append_unique(values, value, output_features);
  }
  const auto add_cluster = [&](const std::size_t center) {
    for (const int delta : {-129, -128, -127, -65, -64, -63, -17, -16,
                            -15, -1, 0, 1, 15, 16, 17, 63, 64, 65,
                            127, 128, 129}) {
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
  };
  add_cluster(output_features / 2U);
  add_cluster(output_features - 1U);
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
    const SampleCoordinate coordinate, const std::size_t selected_count,
    const unsigned int source_index, const float scale) noexcept {
  int sum = 0;
  for (std::size_t selected_index = 0U;
       selected_index < selected_count; ++selected_index) {
    const auto narrowed = static_cast<unsigned int>(selected_index);
    sum += activation_sign(coordinate.token, narrowed) *
           weight_sign(source_index, coordinate.output, narrowed);
  }
  return bf16_rne_bits(static_cast<float>(sum) * scale);
}

[[nodiscard]] bool exact_resource_receipt(
    TestContext& test, const Sm87P40PackedProjectionRole role) {
  Sm87P40PackedProjectionResources resources{};
  const int status =
      q3x::kernels::query_sm87_p40_packed_projection_resources_cuda(
          role, &resources);
  test.expect(status == static_cast<int>(cudaSuccess),
              "FP8 resource query failed for role " +
                  std::to_string(static_cast<unsigned int>(role)) + ": " +
                  cudaGetErrorString(static_cast<cudaError_t>(status)));
  if (status != static_cast<int>(cudaSuccess)) {
    return false;
  }
  const bool full = role == Sm87P40PackedProjectionRole::kFp8FullQkv;
  const bool exact =
      resources.registers_per_thread > 0 &&
      resources.registers_per_thread <= (full ? 254 : 196) &&
      resources.static_shared_bytes == 512U &&
      resources.dynamic_shared_bytes == (full ? 73'728U : 65'536U) &&
      resources.local_bytes == 0U &&
      resources.maximum_threads_per_block >= 128 &&
      resources.active_blocks_per_sm == 2;
  test.expect(exact,
              "FP8 resource receipt must retain CTA2/local0 and fixed shared "
              "geometry for role " +
                  std::to_string(static_cast<unsigned int>(role)));
  return exact;
}

[[nodiscard]] std::uint8_t expected_weight_code(
    const unsigned int source_index, const unsigned int output,
    const unsigned int column,
    const std::vector<std::uint32_t>& selected_k) {
  const auto found =
      std::lower_bound(selected_k.begin(), selected_k.end(), column);
  if (found == selected_k.end() || *found != column) {
    return 0U;
  }
  const unsigned int selected_index =
      static_cast<unsigned int>(found - selected_k.begin());
  return fp8_one_code(weight_sign(source_index, output, selected_index));
}

[[nodiscard]] bool validate_fragment(
    TestContext& test, const Sm87P40PackedProjectionPlan& plan,
    const std::uint8_t* const payload, const std::uint32_t n_tile,
    const std::uint32_t global_k16, const std::uint32_t warp,
    const std::vector<std::uint32_t>& selected_k,
    const cudaStream_t stream) {
  const auto fragment = q3x::kernels::sm87_p40_packed_fragment(
      plan, n_tile, global_k16, warp);
  test.expect(fragment.valid, "requested FP8 fragment must be valid");
  if (!fragment.valid) {
    return false;
  }
  std::vector<std::uint8_t> actual(fragment.weight_bytes, 0U);
  if (!test.cuda_ok(cudaMemcpyAsync(
                        actual.data(), payload + fragment.weight_offset,
                        actual.size(), cudaMemcpyDeviceToHost, stream),
                    "download packed FP8 fragment") ||
      !test.cuda_ok(cudaStreamSynchronize(stream),
                    "synchronize packed FP8 fragment")) {
    return false;
  }
  const unsigned int n_panels =
      fragment.weight_bytes / (kWarpSize * sizeof(std::uint32_t));
  std::size_t mismatches = 0U;
  for (unsigned int n_panel = 0U; n_panel < n_panels; ++n_panel) {
    for (unsigned int lane = 0U; lane < kWarpSize; ++lane) {
      const unsigned int output = fragment.output_column + n_panel * 8U +
                                  lane / 4U;
      const unsigned int column = global_k16 * 16U + 2U * (lane % 4U);
      const std::array<std::uint8_t, 4U> expected{{
          expected_weight_code(fragment.partition_index, output, column,
                               selected_k),
          expected_weight_code(fragment.partition_index, output,
                               column + 8U, selected_k),
          expected_weight_code(fragment.partition_index, output,
                               column + 1U, selected_k),
          expected_weight_code(fragment.partition_index, output,
                               column + 9U, selected_k),
      }};
      const std::size_t offset =
          (n_panel * kWarpSize + lane) * sizeof(std::uint32_t);
      for (std::size_t byte = 0U; byte < expected.size(); ++byte) {
        if (actual[offset + byte] != expected[byte]) {
          ++mismatches;
        }
      }
    }
  }
  test.expect(mismatches == 0U,
              "packed FP8 fragment mismatches=" +
                  std::to_string(mismatches));
  return mismatches == 0U;
}

[[nodiscard]] bool validate_layout_samples(
    TestContext& test, const Sm87P40PackedProjectionPlan& plan,
    const std::uint8_t* const payload,
    const std::vector<std::uint32_t>& selected_k,
    const cudaStream_t stream) {
  for (std::size_t source_index = 0U; source_index < plan.source_count;
       ++source_index) {
    const auto& partition = plan.partitions[source_index];
    const std::array<std::uint32_t, 3U> n_tiles{{
        partition.first_task_n_tile,
        partition.first_task_n_tile + partition.task_n_tiles / 2U,
        partition.first_task_n_tile + partition.task_n_tiles - 1U,
    }};
    const std::array<std::uint32_t, 3U> k16s{{
        0U, partition.input_features / 32U,
        partition.input_features / 16U - 1U,
    }};
    const std::array<std::uint32_t, 2U> warps{{0U,
                                               partition.warps - 1U}};
    for (const std::uint32_t n_tile : n_tiles) {
      for (const std::uint32_t k16 : k16s) {
        for (const std::uint32_t warp : warps) {
          if (!validate_fragment(test, plan, payload, n_tile, k16, warp,
                                 selected_k, stream)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool validate_samples_and_canaries(
    TestContext& test, const Sm87P40PackedProjectionPlan& plan,
    const std::array<DeviceBuffer<std::uint16_t>, kMaximumSources>& outputs,
    const std::array<float, kMaximumSources>& scales,
    const std::size_t selected_count, const std::uint8_t* const payload_base,
    const std::size_t payload_bytes, const cudaStream_t stream) {
  std::array<std::uint8_t, kPayloadGuard> prefix{};
  std::array<std::uint8_t, kPayloadGuard> suffix{};
  if (!test.cuda_ok(cudaMemcpyAsync(
                        prefix.data(), payload_base, prefix.size(),
                        cudaMemcpyDeviceToHost, stream),
                    "download packed payload prefix canary") ||
      !test.cuda_ok(cudaMemcpyAsync(
                        suffix.data(),
                        payload_base + kPayloadGuard + payload_bytes,
                        suffix.size(), cudaMemcpyDeviceToHost, stream),
                    "download packed payload suffix canary")) {
    return false;
  }

  for (std::size_t source_index = 0U; source_index < plan.source_count;
       ++source_index) {
    const auto& partition = plan.partitions[source_index];
    const auto coordinates = sample_coordinates(partition.output_features);
    DeviceBuffer<SampleCoordinate> device_coordinates;
    DeviceBuffer<std::uint16_t> device_samples;
    if (!test.cuda_ok(device_coordinates.allocate(coordinates.size()),
                      "allocate FP8 sample coordinates") ||
        !test.cuda_ok(device_samples.allocate(coordinates.size()),
                      "allocate FP8 sample values") ||
        !test.cuda_ok(cudaMemcpyAsync(
                          device_coordinates.data(), coordinates.data(),
                          coordinates.size() * sizeof(SampleCoordinate),
                          cudaMemcpyHostToDevice, stream),
                      "upload FP8 sample coordinates")) {
      return false;
    }
    gather_samples_kernel
        <<<static_cast<unsigned int>((coordinates.size() + kThreads - 1U) /
                                     kThreads),
           kThreads, 0U, stream>>>(
            outputs[source_index].data(), partition.output_features,
            device_coordinates.data(), coordinates.size(),
            device_samples.data());
    if (!test.cuda_ok(cudaGetLastError(), "gather FP8 output samples")) {
      return false;
    }
    std::vector<std::uint16_t> actual(coordinates.size(), 0U);
    std::array<std::uint16_t, kOutputGuardElements> output_canary{};
    const std::size_t output_elements =
        kTokens * partition.output_features;
    if (!test.cuda_ok(cudaMemcpyAsync(
                          actual.data(), device_samples.data(),
                          actual.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
                      "download FP8 output samples") ||
        !test.cuda_ok(cudaMemcpyAsync(
                          output_canary.data(),
                          outputs[source_index].data() + output_elements,
                          output_canary.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost, stream),
                      "download FP8 output canary") ||
        !test.cuda_ok(cudaStreamSynchronize(stream),
                      "synchronize FP8 samples")) {
      return false;
    }
    std::size_t mismatches = 0U;
    for (std::size_t index = 0U; index < coordinates.size(); ++index) {
      const std::uint16_t expected = expected_output(
          coordinates[index], selected_count,
          static_cast<unsigned int>(source_index), scales[source_index]);
      if (actual[index] != expected) {
        if (mismatches < 12U) {
          std::cerr << "FAIL: FP8 source " << source_index << " C["
                    << coordinates[index].token << ','
                    << coordinates[index].output << "] expected=0x"
                    << std::hex << expected << " actual=0x" << actual[index]
                    << std::dec << '\n';
        }
        ++mismatches;
      }
    }
    test.expect(mismatches == 0U,
                "FP8 sampled oracle mismatches=" +
                    std::to_string(mismatches));
    test.expect(std::all_of(output_canary.begin(), output_canary.end(),
                            [](const std::uint16_t value) {
                              return value == kOutputCanary;
                            }),
                "FP8 output suffix canary changed");
  }
  test.expect(std::all_of(prefix.begin(), prefix.end(),
                          [](const std::uint8_t value) {
                            return value == kPayloadCanary;
                          }),
              "FP8 packed payload prefix canary changed");
  test.expect(std::all_of(suffix.begin(), suffix.end(),
                          [](const std::uint8_t value) {
                            return value == kPayloadCanary;
                          }),
              "FP8 packed payload suffix canary changed");
  return test.failures == 0;
}

void run_role(TestContext& test, const Sm87P40PackedProjectionRole role,
              const cudaStream_t stream) {
  if (!exact_resource_receipt(test, role)) {
    return;
  }
  const Sm87P40PackedProjectionPlan plan =
      q3x::kernels::sm87_p40_packed_projection_plan(role);
  test.expect(plan.valid(), "FP8 plan must be valid");
  if (!plan.valid()) {
    return;
  }
  const std::size_t input_features = plan.partitions[0U].input_features;
  const auto selected_k = selected_k_positions(input_features);

  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint32_t> device_selected_k;
  DeviceBuffer<float> device_scales;
  DeviceBuffer<std::uint8_t> payload_storage;
  std::array<DeviceBuffer<std::uint8_t>, kMaximumSources> weights;
  std::array<DeviceBuffer<std::uint16_t>, kMaximumSources> outputs;
  if (!test.cuda_ok(activations.allocate(kTokens * input_features),
                    "allocate packed FP8 activations") ||
      !test.cuda_ok(device_selected_k.allocate(selected_k.size()),
                    "allocate packed FP8 K positions") ||
      !test.cuda_ok(device_scales.allocate(plan.source_count),
                    "allocate packed FP8 scales") ||
      !test.cuda_ok(payload_storage.allocate(
                        plan.payload_bytes + 2U * kPayloadGuard),
                    "allocate packed FP8 payload+guards")) {
    return;
  }
  if (!test.cuda_ok(cudaMemsetAsync(activations.data(), 0,
                                    activations.bytes(), stream),
                    "zero packed FP8 activations") ||
      !test.cuda_ok(cudaMemcpyAsync(
                        device_selected_k.data(), selected_k.data(),
                        selected_k.size() * sizeof(std::uint32_t),
                        cudaMemcpyHostToDevice, stream),
                    "upload packed FP8 K positions") ||
      !test.cuda_ok(cudaMemsetAsync(payload_storage.data(), kPayloadCanary,
                                    payload_storage.bytes(), stream),
                    "seed packed FP8 payload canaries")) {
    return;
  }
  const std::size_t activation_work = kTokens * selected_k.size();
  initialize_sparse_activations_kernel
      <<<static_cast<unsigned int>((activation_work + kThreads - 1U) /
                                   kThreads),
         kThreads, 0U, stream>>>(
          activations.data(), input_features, device_selected_k.data(),
          static_cast<unsigned int>(selected_k.size()));
  if (!test.cuda_ok(cudaGetLastError(),
                    "initialize packed FP8 activations")) {
    return;
  }

  std::array<float, kMaximumSources> scales{};
  std::array<Sm87P40PackedCanonicalSource, kMaximumSources> sources{};
  std::array<std::uint16_t*, kMaximumSources> output_views{};
  for (std::size_t index = 0U; index < plan.source_count; ++index) {
    const auto& partition = plan.partitions[index];
    const std::size_t weight_bytes =
        static_cast<std::size_t>(partition.output_features) *
        partition.input_features;
    const std::size_t output_elements =
        kTokens * static_cast<std::size_t>(partition.output_features);
    if (!test.cuda_ok(weights[index].allocate(weight_bytes),
                      "allocate canonical FP8 weights") ||
        !test.cuda_ok(outputs[index].allocate(output_elements +
                                               kOutputGuardElements),
                      "allocate packed FP8 output+canary") ||
        !test.cuda_ok(cudaMemsetAsync(weights[index].data(), 0,
                                      weights[index].bytes(), stream),
                      "zero canonical FP8 weights") ||
        !test.cuda_ok(cudaMemsetAsync(outputs[index].data(), 0xa5,
                                      outputs[index].bytes(), stream),
                      "seed packed FP8 output canary")) {
      return;
    }
    const std::size_t weight_work =
        static_cast<std::size_t>(partition.output_features) *
        selected_k.size();
    initialize_sparse_fp8_weights_kernel
        <<<static_cast<unsigned int>((weight_work + kThreads - 1U) /
                                     kThreads),
           kThreads, 0U, stream>>>(
            weights[index].data(), partition.input_features,
            partition.output_features, device_selected_k.data(),
            static_cast<unsigned int>(selected_k.size()),
            static_cast<unsigned int>(index));
    if (!test.cuda_ok(cudaGetLastError(),
                      "initialize canonical FP8 weights")) {
      return;
    }
    scales[index] = std::ldexp(1.0F, static_cast<int>(index) - 1);
    sources[index] = {partition.role,
                      weights[index].data(),
                      nullptr,
                      device_scales.data() + index,
                      partition.output_features,
                      partition.input_features};
    output_views[index] = outputs[index].data();
  }
  if (!test.cuda_ok(cudaMemcpyAsync(
                        device_scales.data(), scales.data(),
                        plan.source_count * sizeof(float),
                        cudaMemcpyHostToDevice, stream),
                    "upload packed FP8 scales")) {
    return;
  }

  auto* const payload = payload_storage.data() + kPayloadGuard;
  const int prepare_status =
      q3x::kernels::prepare_sm87_p40_packed_projection_cuda(
          role, sources.data(), plan.source_count, payload,
          plan.payload_bytes, stream);
  test.expect(prepare_status == static_cast<int>(cudaSuccess),
              "prepare packed FP8 artifact failed: " +
                  std::string(cudaGetErrorString(
                      static_cast<cudaError_t>(prepare_status))));
  if (prepare_status != static_cast<int>(cudaSuccess)) {
    return;
  }
  if (!validate_layout_samples(test, plan, payload, selected_k, stream)) {
    return;
  }

  Sm87P40PackedProjectionDeviceView artifact{};
  artifact.payload = payload;
  artifact.payload_bytes = plan.payload_bytes;
  artifact.artifact_identity = 1U + static_cast<std::uint64_t>(role);
  artifact.role = role;
  artifact.tactic = plan.tactic;
  artifact.source_count = plan.source_count;
  artifact.scalar_scales = scales;

  const int short_status = q3x::kernels::launch_sm87_p40_packed_fp8_cuda(
      activations.data(), artifact, kTokens - 1U, output_views, stream);
  test.expect(short_status == static_cast<int>(cudaErrorInvalidValue),
              "packed FP8 launch rejects non-P40000 token count");
  auto invalid_artifact = artifact;
  invalid_artifact.source_count = plan.source_count - 1U;
  const int source_status =
      q3x::kernels::launch_sm87_p40_packed_fp8_cuda(
          activations.data(), invalid_artifact, kTokens, output_views,
          stream);
  test.expect(source_status == static_cast<int>(cudaErrorInvalidValue),
              "packed FP8 launch rejects source-count drift");

  const int launch_status =
      q3x::kernels::launch_sm87_p40_packed_fp8_cuda(
          activations.data(), artifact, kTokens, output_views, stream);
  test.expect(launch_status == static_cast<int>(cudaSuccess),
              "launch packed FP8 projection failed: " +
                  std::string(cudaGetErrorString(
                      static_cast<cudaError_t>(launch_status))));
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return;
  }
  (void)validate_samples_and_canaries(
      test, plan, outputs, scales, selected_k.size(), payload_storage.data(),
      plan.payload_bytes, stream);
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
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: exact SM87/16-SM device required\n";
    return 77;
  }

  TestContext test;
  Stream stream;
  if (!test.cuda_ok(stream.create(), "create packed FP8 test stream")) {
    return 1;
  }
  constexpr std::array<Sm87P40PackedProjectionRole, 3U> kRoles{{
      Sm87P40PackedProjectionRole::kFp8LinearQkvZ,
      Sm87P40PackedProjectionRole::kFp8FullQkv,
      Sm87P40PackedProjectionRole::kFp8AttentionOutput,
  }};
  for (const Sm87P40PackedProjectionRole role : kRoles) {
    run_role(test, role, stream.get());
  }

  if (test.failures != 0) {
    std::cerr << test.failures << " packed-P40 FP8 CUDA checks failed\n";
    return 1;
  }
  std::cout << "packed-P40 FP8 CUDA correctness/canary/resource checks "
               "passed\n";
  return 0;
}
