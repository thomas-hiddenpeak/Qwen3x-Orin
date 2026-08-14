#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_down_whole_p40.h"

#include "sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_oracle_internal.h"
#include "sm87_target_aot_projection_nvfp4_oracle_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernels = q3x::kernels;

namespace {

constexpr int kSkip = 77;
constexpr std::size_t kRows = 64U;
constexpr std::size_t kIntermediate = 17'408U;
constexpr std::size_t kHidden = 5'120U;
constexpr std::size_t kGuardBytes = 256U;
constexpr std::uint8_t kGuard = 0xa5U;
constexpr std::uint8_t kScaleOne = 0x38U;
constexpr std::uint8_t kWeightOne = 0x02U;
constexpr std::uint16_t kBf16One = 0x3f80U;
constexpr std::uint16_t kResidualPositive = 0x3e80U;
constexpr std::uint16_t kResidualNegative = 0xbe00U;

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

  [[nodiscard]] bool create() noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) ==
           cudaSuccess;
  }
  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class GuardedBuffer final {
 public:
  GuardedBuffer() = default;
  GuardedBuffer(const GuardedBuffer&) = delete;
  GuardedBuffer& operator=(const GuardedBuffer&) = delete;
  ~GuardedBuffer() {
    if (allocation_ != nullptr) {
      (void)cudaFree(allocation_);
    }
  }

  [[nodiscard]] bool allocate(const std::size_t bytes) noexcept {
    bytes_ = bytes;
    if (cudaMalloc(reinterpret_cast<void**>(&allocation_),
                   bytes + 2U * kGuardBytes) != cudaSuccess) {
      return false;
    }
    data_ = allocation_ + kGuardBytes;
    return true;
  }

  [[nodiscard]] bool initialize(const cudaStream_t stream) noexcept {
    return cudaMemsetAsync(allocation_, kGuard,
                           bytes_ + 2U * kGuardBytes, stream) ==
           cudaSuccess;
  }

  [[nodiscard]] std::uint8_t* data() noexcept { return data_; }
  [[nodiscard]] const std::uint8_t* data() const noexcept { return data_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }

  [[nodiscard]] bool guards_unchanged() const {
    std::array<std::uint8_t, kGuardBytes> prefix{};
    std::array<std::uint8_t, kGuardBytes> suffix{};
    return cudaMemcpy(prefix.data(), allocation_, prefix.size(),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           cudaMemcpy(suffix.data(), data_ + bytes_, suffix.size(),
                      cudaMemcpyDeviceToHost) == cudaSuccess &&
           std::all_of(prefix.begin(), prefix.end(),
                       [](const auto value) { return value == kGuard; }) &&
           std::all_of(suffix.begin(), suffix.end(),
                       [](const auto value) { return value == kGuard; });
  }

 private:
  std::uint8_t* allocation_ = nullptr;
  std::uint8_t* data_ = nullptr;
  std::size_t bytes_ = 0U;
};

[[nodiscard]] std::uint16_t encode_bf16_rne(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] float decode_bf16(const std::uint16_t bits) noexcept {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

void initialize_unit_scales(
    std::vector<std::uint8_t>& payload,
    const kernels::Sm87TargetAotProjectionPackedLayout& layout) {
  std::fill(payload.begin(), payload.end(), 0U);
  const auto& partition = layout.partitions[0U];
  for (std::size_t n_tile = 0U; n_tile < partition.n_tiles; ++n_tile) {
    for (std::size_t k_tile = 0U; k_tile < partition.k_tiles; ++k_tile) {
      const std::size_t cell =
          static_cast<std::size_t>(partition.payload_offset) +
          (n_tile * partition.k_tiles + k_tile) * partition.cell_bytes;
      std::fill_n(payload.begin() + static_cast<std::ptrdiff_t>(
                                      cell +
                                      partition.weight_bytes_per_cell),
                  partition.block_scale_bytes_per_cell, kScaleOne);
    }
  }
}

[[nodiscard]] bool set_weight(
    std::vector<std::uint8_t>& payload,
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t n, const std::size_t k,
    const std::uint8_t code) {
  const auto address =
      kernels::sm87_target_aot_projection_packed_weight_address(
          layout, 0U, n, k);
  if (!address.valid || address.byte_offset >= payload.size()) {
    return false;
  }
  auto& byte = payload[static_cast<std::size_t>(address.byte_offset)];
  const std::uint8_t shift =
      static_cast<std::uint8_t>(address.nibble * 4U);
  byte = static_cast<std::uint8_t>(
      (byte & static_cast<std::uint8_t>(~(0x0fU << shift))) |
      static_cast<std::uint8_t>((code & 0x0fU) << shift));
  return true;
}

[[nodiscard]] bool copy_to_device(GuardedBuffer& destination,
                                  const void* const source,
                                  const cudaStream_t stream) noexcept {
  return cudaMemcpyAsync(destination.data(), source, destination.bytes(),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

[[nodiscard]] bool copy_from_device(void* const destination,
                                    const GuardedBuffer& source,
                                    const cudaStream_t stream) noexcept {
  return cudaMemcpyAsync(destination, source.data(), source.bytes(),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

struct Inventory final {
  GuardedBuffer h;
  GuardedBuffer payload;
  GuardedBuffer candidate_residual;
  GuardedBuffer v1_residual;
  GuardedBuffer control;
  GuardedBuffer cancellation;

  [[nodiscard]] bool allocate(const std::size_t payload_bytes) {
    return h.allocate(kRows * kIntermediate * sizeof(std::uint16_t)) &&
           payload.allocate(payload_bytes) &&
           candidate_residual.allocate(kRows * kHidden *
                                       sizeof(std::uint16_t)) &&
           v1_residual.allocate(kRows * kHidden * sizeof(std::uint16_t)) &&
           control.allocate(sizeof(
               kernels::Sm87BulkV2NvFp4DownWholeP40DeviceControl)) &&
           cancellation.allocate(sizeof(std::uint32_t));
  }

  [[nodiscard]] bool guards_unchanged() const {
    return h.guards_unchanged() && payload.guards_unchanged() &&
           candidate_residual.guards_unchanged() &&
           v1_residual.guards_unchanged() && control.guards_unchanged() &&
           cancellation.guards_unchanged();
  }
};

[[nodiscard]] bool compare_exact(const std::vector<std::uint16_t>& observed,
                                 const std::vector<std::uint16_t>& expected,
                                 const char* const label) {
  for (std::size_t index = 0U; index < observed.size(); ++index) {
    if (observed[index] != expected[index]) {
      std::cerr << label << " mismatch row=" << index / kHidden
                << " column=" << index % kHidden << " expected=0x"
                << std::hex << expected[index] << " observed=0x"
                << observed[index] << std::dec << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool run_exact_case(
    const char* const label, const std::size_t selected_k,
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    Inventory& device, const cudaStream_t stream,
    const std::uint64_t epoch) {
  std::vector<std::uint16_t> h(kRows * kIntermediate, 0U);
  std::vector<std::uint16_t> residual(kRows * kHidden, 0U);
  std::vector<std::uint8_t> payload(layout.payload_bytes, 0U);
  initialize_unit_scales(payload, layout);
  for (std::size_t row = 0U; row < kRows; ++row) {
    h[row * kIntermediate + selected_k] = kBf16One;
    for (std::size_t column = 0U; column < kHidden; ++column) {
      residual[row * kHidden + column] =
          (column & 1U) == 0U ? kResidualPositive : kResidualNegative;
    }
  }
  for (std::size_t column = 0U; column < kHidden; ++column) {
    if (!set_weight(payload, layout, column, selected_k, kWeightOne)) {
      std::cerr << label << ": payload encoding failed\n";
      return false;
    }
  }
  if (!device.h.initialize(stream) || !device.payload.initialize(stream) ||
      !device.candidate_residual.initialize(stream) ||
      !device.v1_residual.initialize(stream) ||
      !device.control.initialize(stream) ||
      !device.cancellation.initialize(stream) ||
      !copy_to_device(device.h, h.data(), stream) ||
      !copy_to_device(device.payload, payload.data(), stream) ||
      !copy_to_device(device.candidate_residual, residual.data(), stream) ||
      !copy_to_device(device.v1_residual, residual.data(), stream) ||
      cudaMemsetAsync(device.control.data(), 0, device.control.bytes(),
                      stream) != cudaSuccess ||
      cudaMemsetAsync(device.cancellation.data(), 0,
                      device.cancellation.bytes(), stream) != cudaSuccess) {
    std::cerr << label << ": initialization failed\n";
    return false;
  }

  kernels::sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail::RawArguments
      candidate;
  candidate.transaction_epoch = epoch;
  candidate.h = reinterpret_cast<const std::uint16_t*>(device.h.data());
  candidate.down_payload = device.payload.data();
  candidate.tensor_scale = 1.0F;
  candidate.residual = reinterpret_cast<std::uint16_t*>(
      device.candidate_residual.data());
  candidate.device_control = reinterpret_cast<
      kernels::Sm87BulkV2NvFp4DownWholeP40DeviceControl*>(
      device.control.data());
  candidate.cancellation_signal =
      reinterpret_cast<const std::uint32_t*>(device.cancellation.data());
  candidate.m_tiles = 1U;
  candidate.n_tiles = 20U;
  candidate.cuda_stream = reinterpret_cast<void*>(stream);
  if (kernels::sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail::launch_raw(
          candidate) != static_cast<int>(cudaSuccess)) {
    std::cerr << label << ": candidate launch failed\n";
    return false;
  }

  kernels::sm87_target_aot_nvfp4_oracle_detail::RawV1Arguments v1;
  v1.role = kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
  v1.input = reinterpret_cast<const std::uint16_t*>(device.h.data());
  v1.payload = device.payload.data();
  v1.token_count = kRows;
  v1.tensor_scale0 = 1.0F;
  v1.output_or_residual = reinterpret_cast<std::uint16_t*>(
      device.v1_residual.data());
  v1.cuda_stream = reinterpret_cast<void*>(stream);
  if (kernels::sm87_target_aot_nvfp4_oracle_detail::launch_raw_v1(v1) !=
          static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << label << ": v1 launch/sync failed: "
              << cudaGetErrorString(cudaGetLastError()) << '\n';
    return false;
  }

  std::vector<std::uint16_t> candidate_output(residual.size(), 0U);
  std::vector<std::uint16_t> v1_output(residual.size(), 0U);
  std::vector<std::uint16_t> h_after(h.size(), 0U);
  std::vector<std::uint8_t> payload_after(payload.size(), 0U);
  kernels::Sm87BulkV2NvFp4DownWholeP40DeviceControl control{};
  if (!copy_from_device(candidate_output.data(), device.candidate_residual,
                        stream) ||
      !copy_from_device(v1_output.data(), device.v1_residual, stream) ||
      !copy_from_device(h_after.data(), device.h, stream) ||
      !copy_from_device(payload_after.data(), device.payload, stream) ||
      !copy_from_device(&control, device.control, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << label << ": result copy failed\n";
    return false;
  }

  std::vector<std::uint16_t> expected(residual.size(), 0U);
  for (std::size_t index = 0U; index < expected.size(); ++index) {
    expected[index] =
        encode_bf16_rne(1.0F + decode_bf16(residual[index]));
  }
  if (!compare_exact(candidate_output, v1_output, label) ||
      !compare_exact(candidate_output, expected, label) ||
      h_after != h || payload_after != payload ||
      control.transaction_epoch != epoch ||
      control.requested_m_tiles != 1U ||
      control.requested_n_tiles != 20U ||
      control.completed_super_waves != 5U ||
      control.completed_output_tiles != 20U ||
      control.cancellation_observed != 0U || control.wave_cancelled != 0U ||
      control.first_unfinished_super_wave != 5U ||
      control.error_code != 0U ||
      control.policy !=
          kernels::kSm87BulkV2NvFp4DownWholeP40RequiredPolicy ||
      !std::all_of(control.cta_completed_super_waves.begin(),
                   control.cta_completed_super_waves.end(),
                   [](const auto value) { return value == 5U; }) ||
      !device.guards_unchanged()) {
    std::cerr << label << ": exact/control/guard contract failed\n";
    return false;
  }

  // Pre-launch cancellation must suppress every residual publication. All
  // CTAs still reach the one terminal grid barrier, so divergent observation
  // cannot deadlock the oracle.
  const std::uint32_t cancel = 1U;
  if (!copy_to_device(device.candidate_residual, residual.data(), stream) ||
      cudaMemsetAsync(device.control.data(), 0, device.control.bytes(),
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(device.cancellation.data(), &cancel, sizeof(cancel),
                      cudaMemcpyHostToDevice, stream) != cudaSuccess ||
      kernels::sm87_bulk_v2_nvfp4_down_whole_p40_oracle_detail::launch_raw(
          candidate) != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !copy_from_device(candidate_output.data(), device.candidate_residual,
                        stream) ||
      !copy_from_device(&control, device.control, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      candidate_output != residual || control.cancellation_observed != 1U ||
      control.wave_cancelled != 1U ||
      control.completed_output_tiles != 0U ||
      control.completed_super_waves != 0U ||
      control.first_unfinished_super_wave != 0U ||
      !std::all_of(control.cta_completed_super_waves.begin(),
                   control.cta_completed_super_waves.end(),
                   [](const auto value) { return value == 0U; }) ||
      !device.guards_unchanged()) {
    std::cerr << label << ": cancellation contract failed\n";
    return false;
  }
  return true;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
    return kSkip;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, 0) != cudaSuccess ||
      properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16 ||
      properties.cooperativeLaunch == 0) {
    return kSkip;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    return 1;
  }

  kernels::Sm87BulkV2NvFp4DownWholeP40Resources resources{};
  const int resource_status =
      kernels::query_sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_resources_cuda(
          nullptr, &resources);
  if (resource_status != static_cast<int>(cudaSuccess) ||
      resources.kernel_symbol_identity !=
          kernels::kSm87BulkV2NvFp4DownWholeP40KernelSymbolIdentity ||
      resources.device_ordinal != 0 ||
      resources.binary_version != 87 ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 52'224U ||
      resources.local_bytes != 0U || resources.active_blocks_per_sm < 2 ||
      resources.cooperative_grid_capacity < 32 ||
      !resources.dynamic_shared_attribute_configured ||
      resources.resource_gate_passed ||
      kernels::sm87_bulk_v2_nvfp4_down_whole_p40_resources_valid(resources)) {
    std::cerr << "whole-P40 Down resource observation failed\n";
    return 1;
  }
  std::cout << "whole-P40 Down resources: regs="
            << resources.registers_per_thread
            << " local=" << resources.local_bytes
            << " dynamic_smem=" << resources.dynamic_shared_bytes
            << " active_cta_per_sm=" << resources.active_blocks_per_sm
            << '\n';

  const auto layout = kernels::sm87_target_aot_projection_packed_layout(
      kernels::Sm87TargetAotProjectionRole::kNvFp4Down);
  if (!layout.valid() || layout.payload_bytes != 50'135'040ULL) {
    return 1;
  }
  Inventory device;
  Stream stream;
  if (!device.allocate(static_cast<std::size_t>(layout.payload_bytes)) ||
      !stream.create()) {
    std::cerr << "oracle allocation failed\n";
    return 1;
  }
  if (!run_exact_case("K-first", 0U, layout, device, stream.get(), 41U) ||
      !run_exact_case("K-last", kIntermediate - 1U, layout, device,
                      stream.get(), 42U)) {
    return 1;
  }
  return 0;
}
