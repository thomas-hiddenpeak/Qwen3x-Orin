#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "pinned_checkpoint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

namespace {

namespace kernels = q3x::kernels;
namespace support = q3x::test::support;

constexpr std::size_t kTokens = 407U;
constexpr std::size_t kRows = 10'240U;
constexpr std::size_t kColumns = 5'120U;

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t bytes) {
    bytes_ = bytes;
    return cudaMalloc(&pointer_, bytes) == cudaSuccess;
  }
  [[nodiscard]] void* get() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(pointer_);
  }

 private:
  void* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
};

[[nodiscard]] bool cuda_ok(const cudaError_t status, const char* stage) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "FP8_MARLIN_QKV_ERROR stage=" << stage
            << " cuda=" << cudaGetErrorName(status)
            << " detail=" << cudaGetErrorString(status) << '\n';
  return false;
}

[[nodiscard]] bool launch_ok(const int status, const char* stage) {
  return cuda_ok(static_cast<cudaError_t>(status), stage);
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  bits += 0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t value) noexcept {
  const std::uint32_t bits = static_cast<std::uint32_t>(value) << 16U;
  float output = 0.0F;
  std::memcpy(&output, &bits, sizeof(output));
  return output;
}

[[nodiscard]] bool launch_canonical(
    const std::uint8_t* const weight, const float weight_scale,
    const std::uint16_t* const input, std::uint16_t* const output,
    void* const stream) {
  std::size_t offset = 0U;
  while (kTokens - offset >= 32U) {
    if (!launch_ok(kernels::launch_sm87_fp8_w8a16_m32_gemm_bf16_cuda(
                       weight, weight_scale, input + offset * kColumns,
                       kRows, kColumns, output + offset * kRows, stream),
                   "canonical_m32")) {
      return false;
    }
    offset += 32U;
  }
  if (kTokens - offset >= 16U) {
    if (!launch_ok(kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
                       weight, weight_scale, input + offset * kColumns,
                       kRows, kColumns, output + offset * kRows, stream),
                   "canonical_m16")) {
      return false;
    }
    offset += 16U;
  }
  if (offset < kTokens) {
    if (!launch_ok(kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
                       weight, weight_scale, input + offset * kColumns,
                       kTokens - offset, kRows, kColumns,
                       output + offset * kRows, stream),
                   "canonical_tail")) {
      return false;
    }
  }
  return true;
}

template <typename Launch>
[[nodiscard]] bool measure(Launch launch, const char* const stage,
                           float* const milliseconds) {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  if (!cuda_ok(cudaEventCreate(&begin), "event_begin") ||
      !cuda_ok(cudaEventCreate(&end), "event_end")) {
    return false;
  }
  bool ok = launch() && cuda_ok(cudaDeviceSynchronize(), "warmup_sync");
  std::vector<float> samples;
  for (int round = 0; ok && round < 5; ++round) {
    ok = cuda_ok(cudaEventRecord(begin), "event_record_begin") && launch() &&
         cuda_ok(cudaEventRecord(end), "event_record_end") &&
         cuda_ok(cudaEventSynchronize(end), "event_sync");
    float elapsed = 0.0F;
    if (ok) {
      ok = cuda_ok(cudaEventElapsedTime(&elapsed, begin, end), stage);
      samples.push_back(elapsed);
    }
  }
  (void)cudaEventDestroy(begin);
  (void)cudaEventDestroy(end);
  if (!ok || samples.empty()) {
    return false;
  }
  std::sort(samples.begin(), samples.end());
  *milliseconds = samples[samples.size() / 2U];
  return true;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc != 2) {
    std::cerr << "usage: q3x_sm87_fp8_marlin_real_qkv_test MODEL_DIRECTORY\n";
    return 77;
  }

  int device = 0;
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDevice(&device), "get_device") ||
      !cuda_ok(cudaGetDeviceProperties(&properties, device),
               "device_properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kernels::kSm87Fp8MarlinSmCount)) {
    return 77;
  }

  support::PinnedBundleLoadOptions options;
  options.tensor_names = {
      std::string(support::kQwen36Layer0LinearQkvWeight),
      std::string(support::kQwen36Layer0LinearQkvScale),
  };
  options.maximum_total_payload_bytes = 64U * 1024U * 1024U;
  const auto loaded = support::load_pinned_checkpoint_bundle(
      std::filesystem::path(argv[1]),
      support::qwen36_27b_fp8_attention_bundle(), options);
  if (!loaded) {
    std::cerr << "FP8_MARLIN_QKV_ERROR stage=load_checkpoint detail="
              << (loaded.error.has_value()
                      ? support::describe_pinned_checkpoint_error(*loaded.error)
                      : "missing structured error")
              << '\n';
    return 1;
  }
  const auto* const weight = loaded.value->find_tensor(
      support::kQwen36Layer0LinearQkvWeight);
  const auto* const scale = loaded.value->find_tensor(
      support::kQwen36Layer0LinearQkvScale);
  if (weight == nullptr || scale == nullptr || !scale->f32_value.has_value()) {
    std::cerr << "FP8_MARLIN_QKV_ERROR stage=tensor_contract\n";
    return 1;
  }

  constexpr std::size_t kInputElements = kTokens * kColumns;
  constexpr std::size_t kOutputElements = kTokens * kRows;
  std::vector<std::uint16_t> host_input(kInputElements);
  for (std::size_t index = 0U; index < host_input.size(); ++index) {
    host_input[index] = encode_bf16(
        std::sin(static_cast<float>((index * 17U) % 4093U) * 0.001F) * 0.25F);
  }

  DeviceAllocation canonical_weight;
  DeviceAllocation canonical_scale;
  DeviceAllocation marlin_weight;
  DeviceAllocation marlin_scale;
  DeviceAllocation transpose_scratch;
  DeviceAllocation input;
  DeviceAllocation candidate;
  DeviceAllocation reference;
  DeviceAllocation reduction;
  DeviceAllocation locks;
  const std::size_t weight_bytes = kernels::sm87_fp8_marlin_weight_bytes(
      kRows, kColumns);
  if (!canonical_weight.allocate(weight_bytes) ||
      !canonical_scale.allocate(sizeof(float)) ||
      !marlin_weight.allocate(weight_bytes) ||
      !marlin_scale.allocate(kernels::sm87_fp8_marlin_scale_bytes(kRows)) ||
      !transpose_scratch.allocate(weight_bytes) ||
      !input.allocate(kInputElements * sizeof(std::uint16_t)) ||
      !candidate.allocate(kOutputElements * sizeof(std::uint16_t)) ||
      !reference.allocate(kOutputElements * sizeof(std::uint16_t)) ||
      !reduction.allocate(kernels::kSm87Fp8MarlinReductionBytes) ||
      !locks.allocate(kernels::kSm87Fp8MarlinLockBytes)) {
    std::cerr << "FP8_MARLIN_QKV_ERROR stage=allocate\n";
    return 1;
  }
  if (!cuda_ok(cudaMemcpy(canonical_weight.get(), weight->bytes.data(),
                          weight_bytes, cudaMemcpyHostToDevice),
               "upload_weight") ||
      !cuda_ok(cudaMemcpy(canonical_scale.get(), scale->bytes.data(),
                          sizeof(float), cudaMemcpyHostToDevice),
               "upload_scale") ||
      !cuda_ok(cudaMemcpy(input.get(), host_input.data(), input.bytes(),
                          cudaMemcpyHostToDevice),
               "upload_input") ||
      !launch_ok(kernels::prepare_sm87_fp8_marlin_projection_cuda(
                     canonical_weight.as<std::uint8_t>(),
                     canonical_scale.as<float>(), kRows, kColumns,
                     marlin_weight.as<std::uint8_t>(),
                     marlin_scale.as<std::uint16_t>(), transpose_scratch.get(),
                     transpose_scratch.bytes()),
                 "prepare") ||
      !cuda_ok(cudaDeviceSynchronize(), "prepare_sync")) {
    return 1;
  }

  const auto launch_candidate = [&]() {
    return cuda_ok(cudaMemsetAsync(locks.get(), 0, locks.bytes()),
                   "clear_locks") &&
           launch_ok(kernels::launch_sm87_fp8_marlin_projection_cuda(
                         input.as<std::uint16_t>(),
                         marlin_weight.as<std::uint8_t>(),
                         marlin_scale.as<std::uint16_t>(), kTokens, kRows,
                         kColumns, candidate.as<std::uint16_t>(),
                         reduction.as<float>(), locks.as<std::int32_t>()),
                     "candidate");
  };
  const auto launch_reference = [&]() {
    return launch_canonical(canonical_weight.as<std::uint8_t>(),
                            *scale->f32_value, input.as<std::uint16_t>(),
                            reference.as<std::uint16_t>(), nullptr);
  };
  if (!launch_candidate() || !launch_reference() ||
      !cuda_ok(cudaDeviceSynchronize(), "correctness_sync")) {
    return 1;
  }

  std::vector<std::uint16_t> host_candidate(kOutputElements);
  std::vector<std::uint16_t> host_reference(kOutputElements);
  if (!cuda_ok(cudaMemcpy(host_candidate.data(), candidate.get(),
                          candidate.bytes(), cudaMemcpyDeviceToHost),
               "download_candidate") ||
      !cuda_ok(cudaMemcpy(host_reference.data(), reference.get(),
                          reference.bytes(), cudaMemcpyDeviceToHost),
               "download_reference")) {
    return 1;
  }
  long double squared_error = 0.0L;
  long double squared_reference = 0.0L;
  double max_abs = 0.0;
  std::size_t finite = 0U;
  std::size_t bitwise = 0U;
  for (std::size_t index = 0U; index < kOutputElements; ++index) {
    bitwise += host_candidate[index] == host_reference[index] ? 1U : 0U;
    const double expected = decode_bf16(host_reference[index]);
    const double actual = decode_bf16(host_candidate[index]);
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
      continue;
    }
    const double error = actual - expected;
    squared_error += static_cast<long double>(error) * error;
    squared_reference += static_cast<long double>(expected) * expected;
    max_abs = std::max(max_abs, std::abs(error));
    ++finite;
  }
  const double nrmse =
      squared_reference == 0.0L
          ? std::sqrt(static_cast<double>(squared_error / finite))
          : std::sqrt(static_cast<double>(squared_error / squared_reference));

  float baseline_ms = 0.0F;
  float candidate_ms = 0.0F;
  if (!measure(launch_reference, "baseline_elapsed", &baseline_ms) ||
      !measure(launch_candidate, "candidate_elapsed", &candidate_ms)) {
    return 1;
  }
  const auto plan = kernels::sm87_fp8_marlin_execution_plan(kTokens);
  std::cout << "FP8_MARLIN_REAL_QKV_M407 weight=real_checkpoint"
            << " activation=deterministic_bf16"
            << " finite=" << finite << '/' << kOutputElements
            << " nrmse=" << nrmse << " max_abs=" << max_abs
            << " bitwise_fraction="
            << static_cast<double>(bitwise) / kOutputElements
            << " baseline_ms=" << baseline_ms
            << " candidate_ms=" << candidate_ms
            << " speedup=" << baseline_ms / candidate_ms
            << " plan=" << plan.primary_tokens << '+' << plan.remainder_tokens
            << " launches=" << plan.launch_count << '\n';
  if (finite != kOutputElements || nrmse > 0.03) {
    std::cerr << "FP8_MARLIN_QKV_ERROR stage=correctness_gate\n";
    return 1;
  }
  return 0;
}
