#include "q3x/kernels/sm87_nvfp4_marlin.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "pinned_checkpoint.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace support = q3x::test::support;
namespace kernels = q3x::kernels;

class DeviceAllocation {
 public:
  DeviceAllocation() = default;
  DeviceAllocation(const DeviceAllocation&) = delete;
  DeviceAllocation& operator=(const DeviceAllocation&) = delete;
  ~DeviceAllocation() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t bytes) {
    bytes_ = bytes;
    return cudaMalloc(&data_, bytes) == cudaSuccess;
  }
  [[nodiscard]] void* get() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  template <typename T>
  [[nodiscard]] T* as() const noexcept {
    return static_cast<T*>(data_);
  }

 private:
  void* data_ = nullptr;
  std::size_t bytes_ = 0U;
};

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

[[nodiscard]] bool cuda_ok(const cudaError_t status, const char* stage) {
  if (status == cudaSuccess) {
    return true;
  }
  std::cerr << "MARLIN_LAYER0_ERROR stage=" << stage
            << " cuda=" << cudaGetErrorName(status)
            << " detail=" << cudaGetErrorString(status) << '\n';
  return false;
}

[[nodiscard]] bool launch_ok(const int status, const char* stage) {
  return cuda_ok(static_cast<cudaError_t>(status), stage);
}

[[nodiscard]] bool parse_token_count(const int argc, char** const argv,
                                     std::size_t* const token_count) {
  if (token_count == nullptr) {
    return false;
  }
  if (argc == 2) {
    *token_count = kernels::kSm87NvFp4MarlinTokens;
    return true;
  }
  if (argc != 3) {
    return false;
  }
  constexpr std::string_view kPrefix = "--tokens=";
  const std::string_view argument(argv[2]);
  if (argument.size() < kPrefix.size() ||
      argument.compare(0U, kPrefix.size(), kPrefix) != 0) {
    return false;
  }
  const std::string_view value = argument.substr(kPrefix.size());
  std::size_t parsed = 0U;
  const auto result =
      std::from_chars(value.data(), value.data() + value.size(), parsed);
  if (result.ec != std::errc{} || result.ptr != value.data() + value.size() ||
      !kernels::sm87_nvfp4_marlin_supports_token_count(parsed)) {
    return false;
  }
  *token_count = parsed;
  return true;
}

// Correctness-only canonical oracle. Performance gates never use this split
// path: it deliberately favors an already-authenticated M32/small-M sequence
// so arbitrary Marlin spans can be compared against the checkpoint layout
// without introducing another candidate implementation.
[[nodiscard]] bool launch_canonical_projection(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales, const float weight_scale_2,
    const std::uint16_t* const activations, const std::size_t token_count,
    const std::size_t rows, const std::size_t columns,
    std::uint16_t* const output, const char* const stage) {
  std::size_t token_offset = 0U;
  while (token_count - token_offset >= 32U) {
    if (!launch_ok(kernels::launch_sm87_nvfp4_w4a16_m32_gemm_bf16_cuda(
                       packed_weights, block_scales, weight_scale_2,
                       activations + token_offset * columns, rows, columns,
                       output + token_offset * rows),
                   stage)) {
      return false;
    }
    token_offset += 32U;
  }
  while (token_offset < token_count) {
    const std::size_t launch_tokens =
        std::min<std::size_t>(8U, token_count - token_offset);
    if (!launch_ok(
            kernels::launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
                packed_weights, block_scales, weight_scale_2,
                activations + token_offset * columns, launch_tokens, rows,
                columns, output + token_offset * rows),
            stage)) {
      return false;
    }
    token_offset += launch_tokens;
  }
  return true;
}

[[nodiscard]] const support::PinnedTensorPayload* tensor(
    const support::LoadedPinnedBundle& bundle,
    const std::string_view name) {
  const support::PinnedTensorPayload* const result = bundle.find_tensor(name);
  if (result == nullptr) {
    std::cerr << "MARLIN_LAYER0_ERROR stage=find_tensor tensor=" << name
              << '\n';
  }
  return result;
}

[[nodiscard]] bool upload(const support::PinnedTensorPayload& payload,
                          DeviceAllocation& allocation) {
  return allocation.allocate(payload.bytes.size()) &&
         cuda_ok(cudaMemcpy(allocation.get(), payload.bytes.data(),
                            payload.bytes.size(), cudaMemcpyHostToDevice),
                 payload.name.c_str());
}

struct ErrorStatistics {
  double rmse = 0.0;
  double nrmse = 0.0;
  double maximum_absolute = 0.0;
  std::size_t bitwise_equal = 0U;
  std::size_t finite = 0U;
};

template <typename CandidateIndex>
[[nodiscard]] ErrorStatistics compare_output(
    const std::vector<std::uint16_t>& reference,
    const std::vector<std::uint16_t>& candidate,
    CandidateIndex candidate_index) {
  long double squared_error = 0.0L;
  long double squared_reference = 0.0L;
  ErrorStatistics result;
  for (std::size_t index = 0U; index < reference.size(); ++index) {
    const std::uint16_t candidate_bits = candidate[candidate_index(index)];
    result.bitwise_equal += candidate_bits == reference[index] ? 1U : 0U;
    const double expected = decode_bf16(reference[index]);
    const double actual = decode_bf16(candidate_bits);
    if (!std::isfinite(expected) || !std::isfinite(actual)) {
      continue;
    }
    const double error = actual - expected;
    squared_error += static_cast<long double>(error) * error;
    squared_reference += static_cast<long double>(expected) * expected;
    result.maximum_absolute =
        std::max(result.maximum_absolute, std::abs(error));
    ++result.finite;
  }
  if (result.finite != 0U) {
    result.rmse =
        std::sqrt(static_cast<double>(squared_error / result.finite));
    const double reference_rms =
        std::sqrt(static_cast<double>(squared_reference / result.finite));
    result.nrmse = reference_rms == 0.0 ? result.rmse
                                        : result.rmse / reference_rms;
  }
  return result;
}

void print_statistics(const char* name, const ErrorStatistics& statistics,
                      const std::size_t elements) {
  std::cout << "MARLIN_LAYER0_CORRECTNESS projection=" << name
            << " elements=" << elements << " finite=" << statistics.finite
            << " rmse=" << statistics.rmse
            << " nrmse=" << statistics.nrmse
            << " max_abs=" << statistics.maximum_absolute
            << " bitwise_fraction="
            << static_cast<double>(statistics.bitwise_equal) /
                   static_cast<double>(elements)
            << '\n';
}

}  // namespace

int main(const int argc, char** const argv) {
  std::size_t token_count = 0U;
  if (!parse_token_count(argc, argv, &token_count)) {
    std::cerr << "usage: q3x_sm87_nvfp4_marlin_layer0_test MODEL_DIRECTORY "
                 "[--tokens=1..512]\n";
    return 77;
  }

  int device = 0;
  if (!cuda_ok(cudaGetDevice(&device), "get_device")) {
    return 1;
  }
  cudaDeviceProp properties{};
  if (!cuda_ok(cudaGetDeviceProperties(&properties, device),
               "device_properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount !=
          static_cast<int>(kernels::kSm87NvFp4MarlinSmCount)) {
    return 77;
  }

  support::PinnedBundleLoadOptions options;
  options.tensor_names = {
      std::string(support::kQwen36Layer0GateWeight),
      std::string(support::kQwen36Layer0GateBlockScale),
      std::string(support::kQwen36Layer0GateWeightScale2),
      std::string(support::kQwen36Layer0UpWeight),
      std::string(support::kQwen36Layer0UpBlockScale),
      std::string(support::kQwen36Layer0UpWeightScale2),
      std::string(support::kQwen36Layer0DownWeight),
      std::string(support::kQwen36Layer0DownBlockScale),
      std::string(support::kQwen36Layer0DownWeightScale2),
  };
  options.maximum_total_payload_bytes = 160U * 1024U * 1024U;
  const support::PinnedBundleLoadResult loaded =
      support::load_pinned_checkpoint_bundle(
          std::filesystem::path(argv[1]),
          support::qwen36_27b_nvfp4_layer0_mlp_bundle(), options);
  if (!loaded) {
    std::cerr << "MARLIN_LAYER0_ERROR stage=load_checkpoint detail="
              << (loaded.error.has_value()
                      ? support::describe_pinned_checkpoint_error(*loaded.error)
                      : "missing structured error")
              << '\n';
    return 1;
  }
  const support::LoadedPinnedBundle& bundle = *loaded.value;
  const auto* const gate_weight =
      tensor(bundle, support::kQwen36Layer0GateWeight);
  const auto* const gate_scales =
      tensor(bundle, support::kQwen36Layer0GateBlockScale);
  const auto* const gate_global =
      tensor(bundle, support::kQwen36Layer0GateWeightScale2);
  const auto* const up_weight =
      tensor(bundle, support::kQwen36Layer0UpWeight);
  const auto* const up_scales =
      tensor(bundle, support::kQwen36Layer0UpBlockScale);
  const auto* const up_global =
      tensor(bundle, support::kQwen36Layer0UpWeightScale2);
  const auto* const down_weight =
      tensor(bundle, support::kQwen36Layer0DownWeight);
  const auto* const down_scales =
      tensor(bundle, support::kQwen36Layer0DownBlockScale);
  const auto* const down_global =
      tensor(bundle, support::kQwen36Layer0DownWeightScale2);
  if (gate_weight == nullptr || gate_scales == nullptr ||
      gate_global == nullptr || up_weight == nullptr || up_scales == nullptr ||
      up_global == nullptr || down_weight == nullptr || down_scales == nullptr ||
      down_global == nullptr || !gate_global->f32_value.has_value() ||
      !up_global->f32_value.has_value() ||
      !down_global->f32_value.has_value()) {
    return 1;
  }
  if (gate_global->f32_bits != up_global->f32_bits) {
    std::cerr << "MARLIN_LAYER0_ERROR stage=gate_up_global_scale_contract\n";
    return 1;
  }

  float gate_up_factor = 0.0F;
  float down_factor = 0.0F;
  if (!kernels::derive_sm87_nvfp4_marlin_scale_factor(
          gate_scales->bytes.data(), gate_scales->bytes.size(),
          up_scales->bytes.data(), up_scales->bytes.size(),
          &gate_up_factor) ||
      !kernels::derive_sm87_nvfp4_marlin_scale_factor(
          down_scales->bytes.data(), down_scales->bytes.size(), nullptr, 0U,
          &down_factor)) {
    std::cerr << "MARLIN_LAYER0_ERROR stage=scale_factor\n";
    return 1;
  }

  std::size_t free_before = 0U;
  std::size_t total_memory = 0U;
  if (!cuda_ok(cudaMemGetInfo(&free_before, &total_memory),
               "memory_before")) {
    return 1;
  }

  DeviceAllocation gate_weight_device;
  DeviceAllocation gate_scales_device;
  DeviceAllocation gate_global_device;
  DeviceAllocation up_weight_device;
  DeviceAllocation up_scales_device;
  DeviceAllocation down_weight_device;
  DeviceAllocation down_scales_device;
  DeviceAllocation down_global_device;
  if (!upload(*gate_weight, gate_weight_device) ||
      !upload(*gate_scales, gate_scales_device) ||
      !upload(*gate_global, gate_global_device) ||
      !upload(*up_weight, up_weight_device) ||
      !upload(*up_scales, up_scales_device) ||
      !upload(*down_weight, down_weight_device) ||
      !upload(*down_scales, down_scales_device) ||
      !upload(*down_global, down_global_device)) {
    return 1;
  }

  constexpr std::size_t kGateWeightBytes =
      kernels::sm87_nvfp4_marlin_weight_bytes(
          kernels::kSm87NvFp4MarlinIntermediate,
          kernels::kSm87NvFp4MarlinHidden);
  constexpr std::size_t kGateScaleBytes =
      kernels::sm87_nvfp4_marlin_scale_bytes(
          kernels::kSm87NvFp4MarlinIntermediate,
          kernels::kSm87NvFp4MarlinHidden);
  constexpr std::size_t kGateUpWeightBytes = 2U * kGateWeightBytes;
  constexpr std::size_t kGateUpScaleBytes = 2U * kGateScaleBytes;
  constexpr std::size_t kDownWeightBytes =
      kernels::sm87_nvfp4_marlin_weight_bytes(
          kernels::kSm87NvFp4MarlinHidden,
          kernels::kSm87NvFp4MarlinIntermediate);
  constexpr std::size_t kDownScaleBytes =
      kernels::sm87_nvfp4_marlin_scale_bytes(
          kernels::kSm87NvFp4MarlinHidden,
          kernels::kSm87NvFp4MarlinIntermediate);

  DeviceAllocation marlin_gate_up_weight;
  DeviceAllocation marlin_gate_up_scales;
  DeviceAllocation marlin_gate_up_global;
  DeviceAllocation marlin_down_weight;
  DeviceAllocation marlin_down_scales;
  DeviceAllocation marlin_down_global;
  DeviceAllocation transpose_scratch;
  DeviceAllocation c_tmp;
  DeviceAllocation locks;
  if (!marlin_gate_up_weight.allocate(kGateUpWeightBytes) ||
      !marlin_gate_up_scales.allocate(kGateUpScaleBytes) ||
      !marlin_gate_up_global.allocate(sizeof(float)) ||
      !marlin_down_weight.allocate(kDownWeightBytes) ||
      !marlin_down_scales.allocate(kDownScaleBytes) ||
      !marlin_down_global.allocate(sizeof(float)) ||
      !transpose_scratch.allocate(kGateUpWeightBytes) ||
      !c_tmp.allocate(kernels::kSm87NvFp4MarlinReductionBytes) ||
      !locks.allocate(kernels::kSm87NvFp4MarlinLockBytes) ||
      !cuda_ok(cudaMemset(locks.get(), 0, locks.bytes()), "clear_locks")) {
    std::cerr << "MARLIN_LAYER0_ERROR stage=allocate_sidecars\n";
    return 1;
  }

  if (!launch_ok(kernels::prepare_sm87_nvfp4_marlin_gate_up_cuda(
                     gate_weight_device.as<std::uint8_t>(),
                     up_weight_device.as<std::uint8_t>(),
                     gate_scales_device.as<std::uint8_t>(),
                     up_scales_device.as<std::uint8_t>(),
                     gate_global_device.as<float>(), gate_up_factor,
                     marlin_gate_up_weight.as<std::uint8_t>(),
                     marlin_gate_up_scales.as<std::uint8_t>(),
                     marlin_gate_up_global.as<float>(), transpose_scratch.get(),
                     transpose_scratch.bytes()),
                 "prepare_gate_up") ||
      !launch_ok(kernels::prepare_sm87_nvfp4_marlin_down_cuda(
                     down_weight_device.as<std::uint8_t>(),
                     down_scales_device.as<std::uint8_t>(),
                     down_global_device.as<float>(), down_factor,
                     marlin_down_weight.as<std::uint8_t>(),
                     marlin_down_scales.as<std::uint8_t>(),
                     marlin_down_global.as<float>(), transpose_scratch.get(),
                     transpose_scratch.bytes()),
                 "prepare_down") ||
      !cuda_ok(cudaDeviceSynchronize(), "prepare_synchronize")) {
    return 1;
  }

  kernels::Sm87NvFp4MarlinKernelResources resources;
  if (!launch_ok(
          kernels::query_sm87_nvfp4_marlin_m512_resources_cuda(&resources),
          "query_resources")) {
    return 1;
  }
  std::cout << "MARLIN_LAYER0_RESOURCES registers_per_thread="
            << resources.registers_per_thread
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << " dynamic_shared_bytes="
            << kernels::kSm87NvFp4MarlinDynamicSharedBytes
            << " tile=M64N256K64 stages=4 persistent_ctas=16\n";
  if (resources.active_blocks_per_sm != 1) {
    std::cerr << "MARLIN_LAYER0_ERROR stage=occupancy_contract\n";
    return 1;
  }

  const std::size_t kGateInputElements =
      token_count * kernels::kSm87NvFp4MarlinHidden;
  const std::size_t kGateElements =
      token_count * kernels::kSm87NvFp4MarlinIntermediate;
  const std::size_t kMergedElements = 2U * kGateElements;
  const std::size_t kDownInputElements = kGateElements;
  const std::size_t kDownOutputElements = kGateInputElements;
  std::vector<std::uint16_t> gate_input(kGateInputElements);
  std::vector<std::uint16_t> down_input(kDownInputElements);
  for (std::size_t index = 0U; index < gate_input.size(); ++index) {
    gate_input[index] = encode_bf16(
        std::sin(static_cast<float>((index * 17U) % 4093U) * 0.001F) * 0.25F);
  }
  for (std::size_t index = 0U; index < down_input.size(); ++index) {
    down_input[index] = encode_bf16(
        std::cos(static_cast<float>((index * 29U) % 8191U) * 0.0007F) *
        0.125F);
  }

  DeviceAllocation gate_input_device;
  DeviceAllocation down_input_device;
  DeviceAllocation merged_output_device;
  DeviceAllocation gate_reference_device;
  DeviceAllocation up_reference_device;
  DeviceAllocation down_output_device;
  DeviceAllocation down_reference_device;
  if (!gate_input_device.allocate(gate_input.size() * sizeof(std::uint16_t)) ||
      !down_input_device.allocate(down_input.size() * sizeof(std::uint16_t)) ||
      !merged_output_device.allocate(kMergedElements * sizeof(std::uint16_t)) ||
      !gate_reference_device.allocate(kGateElements * sizeof(std::uint16_t)) ||
      !up_reference_device.allocate(kGateElements * sizeof(std::uint16_t)) ||
      !down_output_device.allocate(kDownOutputElements * sizeof(std::uint16_t)) ||
      !down_reference_device.allocate(kDownOutputElements *
                                      sizeof(std::uint16_t)) ||
      !cuda_ok(cudaMemcpy(gate_input_device.get(), gate_input.data(),
                          gate_input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "upload_gate_input") ||
      !cuda_ok(cudaMemcpy(down_input_device.get(), down_input.data(),
                          down_input.size() * sizeof(std::uint16_t),
                          cudaMemcpyHostToDevice),
               "upload_down_input")) {
    return 1;
  }

  if (!launch_ok(kernels::launch_sm87_nvfp4_marlin_gate_up_cuda(
                     gate_input_device.as<std::uint16_t>(),
                     marlin_gate_up_weight.as<std::uint8_t>(),
                     marlin_gate_up_scales.as<std::uint8_t>(),
                     marlin_gate_up_global.as<float>(),
                     token_count,
                     merged_output_device.as<std::uint16_t>(), c_tmp.as<float>(),
                     locks.as<std::int32_t>()),
                 "candidate_gate_up") ||
      !launch_canonical_projection(
          gate_weight_device.as<std::uint8_t>(),
          gate_scales_device.as<std::uint8_t>(), *gate_global->f32_value,
          gate_input_device.as<std::uint16_t>(), token_count,
          kernels::kSm87NvFp4MarlinIntermediate,
          kernels::kSm87NvFp4MarlinHidden,
          gate_reference_device.as<std::uint16_t>(), "reference_gate") ||
      !launch_canonical_projection(
          up_weight_device.as<std::uint8_t>(),
          up_scales_device.as<std::uint8_t>(), *up_global->f32_value,
          gate_input_device.as<std::uint16_t>(), token_count,
          kernels::kSm87NvFp4MarlinIntermediate,
          kernels::kSm87NvFp4MarlinHidden,
          up_reference_device.as<std::uint16_t>(), "reference_up") ||
      !launch_ok(kernels::launch_sm87_nvfp4_marlin_down_cuda(
                     down_input_device.as<std::uint16_t>(),
                     marlin_down_weight.as<std::uint8_t>(),
                     marlin_down_scales.as<std::uint8_t>(),
                     marlin_down_global.as<float>(),
                     token_count,
                     down_output_device.as<std::uint16_t>(), c_tmp.as<float>(),
                     locks.as<std::int32_t>()),
                 "candidate_down") ||
      !launch_canonical_projection(
          down_weight_device.as<std::uint8_t>(),
          down_scales_device.as<std::uint8_t>(), *down_global->f32_value,
          down_input_device.as<std::uint16_t>(), token_count,
          kernels::kSm87NvFp4MarlinHidden,
          kernels::kSm87NvFp4MarlinIntermediate,
          down_reference_device.as<std::uint16_t>(), "reference_down") ||
      !cuda_ok(cudaDeviceSynchronize(), "gemm_synchronize")) {
    return 1;
  }

  std::vector<std::uint16_t> merged_output(kMergedElements);
  std::vector<std::uint16_t> gate_reference(kGateElements);
  std::vector<std::uint16_t> up_reference(kGateElements);
  std::vector<std::uint16_t> down_output(kDownOutputElements);
  std::vector<std::uint16_t> down_reference(kDownOutputElements);
  if (!cuda_ok(cudaMemcpy(merged_output.data(), merged_output_device.get(),
                          merged_output.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download_gate_up") ||
      !cuda_ok(cudaMemcpy(gate_reference.data(), gate_reference_device.get(),
                          gate_reference.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download_gate_reference") ||
      !cuda_ok(cudaMemcpy(up_reference.data(), up_reference_device.get(),
                          up_reference.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download_up_reference") ||
      !cuda_ok(cudaMemcpy(down_output.data(), down_output_device.get(),
                          down_output.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download_down") ||
      !cuda_ok(cudaMemcpy(down_reference.data(), down_reference_device.get(),
                          down_reference.size() * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost),
               "download_down_reference")) {
    return 1;
  }

  const auto gate_index = [](const std::size_t index) {
    const std::size_t token =
        index / kernels::kSm87NvFp4MarlinIntermediate;
    const std::size_t column =
        index % kernels::kSm87NvFp4MarlinIntermediate;
    return token * kernels::kSm87NvFp4MarlinGateUpOutput + column;
  };
  const auto up_index = [](const std::size_t index) {
    const std::size_t token =
        index / kernels::kSm87NvFp4MarlinIntermediate;
    const std::size_t column =
        index % kernels::kSm87NvFp4MarlinIntermediate;
    return token * kernels::kSm87NvFp4MarlinGateUpOutput +
           kernels::kSm87NvFp4MarlinIntermediate + column;
  };
  const ErrorStatistics gate_error =
      compare_output(gate_reference, merged_output, gate_index);
  const ErrorStatistics up_error =
      compare_output(up_reference, merged_output, up_index);
  const ErrorStatistics down_error = compare_output(
      down_reference, down_output, [](const std::size_t index) { return index; });
  print_statistics("gate", gate_error, kGateElements);
  print_statistics("up", up_error, kGateElements);
  print_statistics("down", down_error, kDownOutputElements);
  std::cout << "MARLIN_LAYER0_CASE token_count=" << token_count
            << " primary_tokens="
            << kernels::sm87_nvfp4_marlin_execution_plan(token_count)
                   .primary_tokens
            << " remainder_tokens="
            << kernels::sm87_nvfp4_marlin_execution_plan(token_count)
                   .remainder_tokens
            << '\n';

  std::size_t free_after = 0U;
  if (!cuda_ok(cudaMemGetInfo(&free_after, &total_memory), "memory_after")) {
    return 1;
  }
  std::cout << "MARLIN_LAYER0_MEMORY free_before=" << free_before
            << " free_after=" << free_after
            << " fixture_delta=" << (free_before - free_after)
            << " gate_up_sidecar="
            << (kGateUpWeightBytes + kGateUpScaleBytes + sizeof(float))
            << " down_sidecar="
            << (kDownWeightBytes + kDownScaleBytes + sizeof(float))
            << " scale_factor_gate_up=" << gate_up_factor
            << " scale_factor_down=" << down_factor
            << " authenticated_revision=" << bundle.revision << '\n';

  constexpr double kNrmseLimit = 0.025;
  constexpr double kMaximumAbsoluteLimit = 2.0;
  if (gate_error.finite != kGateElements || up_error.finite != kGateElements ||
      down_error.finite != kDownOutputElements ||
      gate_error.nrmse > kNrmseLimit || up_error.nrmse > kNrmseLimit ||
      down_error.nrmse > kNrmseLimit ||
      gate_error.maximum_absolute > kMaximumAbsoluteLimit ||
      up_error.maximum_absolute > kMaximumAbsoluteLimit ||
      down_error.maximum_absolute > kMaximumAbsoluteLimit) {
    std::cerr << "MARLIN_LAYER0_ERROR stage=correctness_threshold\n";
    return 1;
  }
  return 0;
}
