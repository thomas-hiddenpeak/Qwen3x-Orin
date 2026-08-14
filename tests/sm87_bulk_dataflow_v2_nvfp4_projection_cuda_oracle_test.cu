#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_projection.h"

#include "sm87_bulk_dataflow_v2_nvfp4_projection_oracle_internal.h"
#include "sm87_target_aot_projection_nvfp4_oracle_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

namespace kernels = q3x::kernels;

namespace {

constexpr int kSkip = 77;
constexpr std::size_t kRows = kernels::kSm87BulkV2NvFp4TailTokens;
constexpr std::size_t kHidden = kernels::kSm87BulkV2NvFp4Hidden;
constexpr std::size_t kIntermediate =
    kernels::kSm87BulkV2NvFp4Intermediate;
constexpr std::size_t kGuardBytes = 256U;
constexpr std::uint8_t kGuardValue = 0xa5U;
constexpr std::uint8_t kScaleOne = 0x38U;
constexpr std::uint8_t kWeightOne = 0x02U;
constexpr std::uint16_t kBf16One = 0x3f80U;
constexpr std::uint16_t kResidualPositive = 0x3e80U;
constexpr std::uint16_t kResidualNegative = 0xbe00U;

static_assert(kRows == 64U && kHidden == 5'120U &&
              kIntermediate == 17'408U);
static_assert(sizeof(kernels::Sm87BulkV2NvFp4DeviceControl) == 1'152U);

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

class GuardedDeviceBuffer final {
 public:
  GuardedDeviceBuffer() = default;
  GuardedDeviceBuffer(const GuardedDeviceBuffer&) = delete;
  GuardedDeviceBuffer& operator=(const GuardedDeviceBuffer&) = delete;
  ~GuardedDeviceBuffer() {
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
    return allocation_ != nullptr && stream != nullptr &&
           cudaMemsetAsync(allocation_, kGuardValue,
                           bytes_ + 2U * kGuardBytes,
                           stream) == cudaSuccess;
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
           std::all_of(prefix.begin(), prefix.end(), [](const auto value) {
             return value == kGuardValue;
           }) &&
           std::all_of(suffix.begin(), suffix.end(), [](const auto value) {
             return value == kGuardValue;
           });
  }

 private:
  std::uint8_t* allocation_ = nullptr;
  std::uint8_t* data_ = nullptr;
  std::size_t bytes_ = 0U;
};

[[nodiscard]] bool copy_to_device(GuardedDeviceBuffer& destination,
                                  const void* const source,
                                  const cudaStream_t stream) noexcept {
  return cudaMemcpyAsync(destination.data(), source, destination.bytes(),
                         cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

[[nodiscard]] bool copy_from_device(void* const destination,
                                    const GuardedDeviceBuffer& source,
                                    const cudaStream_t stream) noexcept {
  return cudaMemcpyAsync(destination, source.data(), source.bytes(),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

[[nodiscard]] float decode_bf16(const std::uint16_t bits) noexcept {
  const std::uint32_t wide = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

[[nodiscard]] std::uint16_t encode_bf16_rne(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

void initialize_unit_scales(
    std::vector<std::uint8_t>& payload,
    const kernels::Sm87TargetAotProjectionPackedLayout& layout) {
  std::fill(payload.begin(), payload.end(), 0U);
  for (std::size_t partition_index = 0U;
       partition_index < layout.partition_count; ++partition_index) {
    const auto& partition = layout.partitions[partition_index];
    for (std::size_t n_tile = 0U; n_tile < partition.n_tiles; ++n_tile) {
      for (std::size_t k_tile = 0U; k_tile < partition.k_tiles; ++k_tile) {
        const std::size_t cell =
            static_cast<std::size_t>(partition.payload_offset) +
            (n_tile * partition.k_tiles + k_tile) * partition.cell_bytes;
        std::fill_n(payload.begin() +
                        static_cast<std::ptrdiff_t>(
                            cell + partition.weight_bytes_per_cell),
                    partition.block_scale_bytes_per_cell, kScaleOne);
      }
    }
  }
}

[[nodiscard]] bool set_weight(
    std::vector<std::uint8_t>& payload,
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const std::size_t partition, const std::size_t n,
    const std::size_t k, const std::uint8_t code) {
  const auto address =
      kernels::sm87_target_aot_projection_packed_weight_address(
          layout, partition, n, k);
  if (!address.valid || address.byte_offset >= payload.size()) {
    return false;
  }
  auto& byte = payload[static_cast<std::size_t>(address.byte_offset)];
  const std::uint8_t shift = static_cast<std::uint8_t>(address.nibble * 4U);
  byte = static_cast<std::uint8_t>(
      (byte & static_cast<std::uint8_t>(~(0x0fU << shift))) |
      static_cast<std::uint8_t>((code & 0x0fU) << shift));
  return true;
}

[[nodiscard]] bool compare_exact(
    const std::vector<std::uint16_t>& observed,
    const std::vector<std::uint16_t>& expected,
    const std::size_t width, const char* const case_name,
    const char* const output_name) {
  for (std::size_t index = 0U; index < observed.size(); ++index) {
    if (observed[index] != expected[index]) {
      std::cerr << case_name << ' ' << output_name << " mismatch at row "
                << index / width << " column " << index % width
                << " expected 0x" << std::hex << expected[index]
                << " observed 0x" << observed[index] << std::dec << '\n';
      return false;
    }
  }
  return true;
}

struct DeviceInventory final {
  GuardedDeviceBuffer input;
  GuardedDeviceBuffer gate_payload;
  GuardedDeviceBuffer down_payload;
  GuardedDeviceBuffer h;
  GuardedDeviceBuffer residual;
  GuardedDeviceBuffer control;
  GuardedDeviceBuffer cancellation;

  [[nodiscard]] bool allocate(const std::size_t gate_bytes,
                              const std::size_t down_bytes) {
    return input.allocate(kRows * kHidden * sizeof(std::uint16_t)) &&
           gate_payload.allocate(gate_bytes) &&
           down_payload.allocate(down_bytes) &&
           h.allocate(kRows * kIntermediate * sizeof(std::uint16_t)) &&
           residual.allocate(kRows * kHidden * sizeof(std::uint16_t)) &&
           control.allocate(
               sizeof(kernels::Sm87BulkV2NvFp4DeviceControl)) &&
           cancellation.allocate(sizeof(std::uint32_t));
  }

  [[nodiscard]] bool guards_unchanged() const {
    return input.guards_unchanged() && gate_payload.guards_unchanged() &&
           down_payload.guards_unchanged() && h.guards_unchanged() &&
           residual.guards_unchanged() && control.guards_unchanged() &&
           cancellation.guards_unchanged();
  }
};

[[nodiscard]] kernels::sm87_bulk_v2_nvfp4_oracle_detail::RawTailArguments
candidate_arguments(DeviceInventory& device, const cudaStream_t stream,
                    const std::uint32_t epoch) {
  kernels::sm87_bulk_v2_nvfp4_oracle_detail::RawTailArguments arguments;
  arguments.normalized_input = reinterpret_cast<const std::uint16_t*>(
      device.input.data());
  arguments.gate_up_payload = device.gate_payload.data();
  arguments.gate_tensor_scale = 1.0F;
  arguments.up_tensor_scale = 1.0F;
  arguments.down_payload = device.down_payload.data();
  arguments.down_tensor_scale = 1.0F;
  arguments.residual =
      reinterpret_cast<std::uint16_t*>(device.residual.data());
  arguments.group_h_scratch =
      reinterpret_cast<std::uint16_t*>(device.h.data());
  arguments.device_control =
      reinterpret_cast<kernels::Sm87BulkV2NvFp4DeviceControl*>(
          device.control.data());
  arguments.cancellation_signal =
      reinterpret_cast<const std::uint32_t*>(device.cancellation.data());
  arguments.group_epoch = epoch;
  arguments.cuda_stream = reinterpret_cast<void*>(stream);
  return arguments;
}

[[nodiscard]] bool initialize_case_device(
    DeviceInventory& device, const std::vector<std::uint16_t>& input,
    const std::vector<std::uint8_t>& gate_payload,
    const std::vector<std::uint8_t>& down_payload,
    const std::vector<std::uint16_t>& residual,
    const std::uint32_t cancellation, const cudaStream_t stream) {
  return device.input.initialize(stream) &&
         device.gate_payload.initialize(stream) &&
         device.down_payload.initialize(stream) && device.h.initialize(stream) &&
         device.residual.initialize(stream) &&
         device.control.initialize(stream) &&
         device.cancellation.initialize(stream) &&
         copy_to_device(device.input, input.data(), stream) &&
         copy_to_device(device.gate_payload, gate_payload.data(), stream) &&
         copy_to_device(device.down_payload, down_payload.data(), stream) &&
         copy_to_device(device.residual, residual.data(), stream) &&
         cudaMemsetAsync(device.h.data(), 0, device.h.bytes(), stream) ==
             cudaSuccess &&
         cudaMemsetAsync(device.control.data(), 0, device.control.bytes(),
                         stream) == cudaSuccess &&
         cudaMemcpyAsync(device.cancellation.data(), &cancellation,
                         sizeof(cancellation), cudaMemcpyHostToDevice,
                         stream) == cudaSuccess;
}

[[nodiscard]] bool run_case(
    const char* const case_name, const bool encoded,
    const std::size_t input_k, const std::size_t intermediate_k,
    const kernels::Sm87TargetAotProjectionPackedLayout& gate_layout,
    const kernels::Sm87TargetAotProjectionPackedLayout& down_layout,
    DeviceInventory& device, const cudaStream_t stream,
    const std::uint32_t epoch) {
  std::vector<std::uint16_t> input(kRows * kHidden, 0U);
  std::vector<std::uint16_t> residual(kRows * kHidden, 0U);
  std::vector<std::uint8_t> gate_payload(gate_layout.payload_bytes, 0U);
  std::vector<std::uint8_t> down_payload(down_layout.payload_bytes, 0U);
  if (encoded) {
    initialize_unit_scales(gate_payload, gate_layout);
    initialize_unit_scales(down_payload, down_layout);
  }
  for (std::size_t row = 0U; row < kRows; ++row) {
    input[row * kHidden + input_k] = kBf16One;
    for (std::size_t column = 0U; column < kHidden; ++column) {
      residual[row * kHidden + column] =
          (column & 1U) == 0U ? kResidualPositive : kResidualNegative;
    }
  }
  if (encoded) {
    if (!set_weight(gate_payload, gate_layout, 0U, intermediate_k,
                    input_k, kWeightOne) ||
        !set_weight(gate_payload, gate_layout, 1U, intermediate_k,
                    input_k, kWeightOne)) {
      std::cerr << case_name << ": Gate/Up one-hot encoding failed\n";
      return false;
    }
    for (std::size_t n = 0U; n < kHidden; ++n) {
      if (!set_weight(down_payload, down_layout, 0U, n,
                      intermediate_k, kWeightOne)) {
        std::cerr << case_name << ": Down one-hot encoding failed\n";
        return false;
      }
    }
  }

  if (!initialize_case_device(device, input, gate_payload, down_payload,
                              residual, 0U, stream)) {
    std::cerr << case_name << ": device initialization failed\n";
    return false;
  }
  auto candidate = candidate_arguments(device, stream, epoch);
  const int candidate_status =
      kernels::sm87_bulk_v2_nvfp4_oracle_detail::launch_raw_tail(candidate);
  if (candidate_status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << case_name << ": v2 launch failed: " << candidate_status
              << " / " << cudaGetErrorString(cudaGetLastError()) << '\n';
    return false;
  }

  std::vector<std::uint16_t> candidate_h(kRows * kIntermediate, 0U);
  std::vector<std::uint16_t> candidate_residual(kRows * kHidden, 0U);
  std::vector<std::uint16_t> input_after(input.size(), 0U);
  std::vector<std::uint8_t> gate_after(gate_payload.size(), 0U);
  std::vector<std::uint8_t> down_after(down_payload.size(), 0U);
  kernels::Sm87BulkV2NvFp4DeviceControl observed_control{};
  if (!copy_from_device(candidate_h.data(), device.h, stream) ||
      !copy_from_device(candidate_residual.data(), device.residual, stream) ||
      !copy_from_device(input_after.data(), device.input, stream) ||
      !copy_from_device(gate_after.data(), device.gate_payload, stream) ||
      !copy_from_device(down_after.data(), device.down_payload, stream) ||
      !copy_from_device(&observed_control, device.control, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << case_name << ": v2 result copy failed\n";
    return false;
  }
  if (input_after != input || gate_after != gate_payload ||
      down_after != down_payload) {
    std::cerr << case_name << ": v2 modified immutable input bytes\n";
    return false;
  }
  if (observed_control.gate_next_q[0U] != 272U ||
      observed_control.retired_q[0U] != 272U ||
      observed_control.claimed_gate_tasks != 272U ||
      observed_control.completed_gate_tasks != 272U ||
      observed_control.completed_down_tasks != 20U ||
      observed_control.cancellation_observed != 0U ||
      observed_control.group_epoch != epoch) {
    std::cerr << case_name << ": v2 completion control mismatch\n";
    return false;
  }
  if (!device.guards_unchanged()) {
    std::cerr << case_name << ": v2 guard corruption\n";
    return false;
  }

  // Reinitialize publications only.  Input and payload remain in place so a
  // second candidate execution catches stale shared/readiness dependence.
  if (cudaMemsetAsync(device.h.data(), 0, device.h.bytes(), stream) !=
          cudaSuccess ||
      copy_to_device(device.residual, residual.data(), stream) == false ||
      cudaMemsetAsync(device.control.data(), 0, device.control.bytes(),
                      stream) != cudaSuccess) {
    std::cerr << case_name << ": replay initialization failed\n";
    return false;
  }
  const int replay_status =
      kernels::sm87_bulk_v2_nvfp4_oracle_detail::launch_raw_tail(candidate);
  std::vector<std::uint16_t> replay_h(candidate_h.size(), 0U);
  std::vector<std::uint16_t> replay_residual(candidate_residual.size(), 0U);
  if (replay_status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !copy_from_device(replay_h.data(), device.h, stream) ||
      !copy_from_device(replay_residual.data(), device.residual, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      replay_h != candidate_h || replay_residual != candidate_residual) {
    std::cerr << case_name << ": v2 deterministic replay failed\n";
    return false;
  }

  // Same-ELF v1 control, fed the identical raw payload addresses, tensor
  // scales, input, stream and M64 logical shape.
  if (cudaMemsetAsync(device.h.data(), 0, device.h.bytes(), stream) !=
          cudaSuccess ||
      !copy_to_device(device.residual, residual.data(), stream)) {
    std::cerr << case_name << ": v1 initialization failed\n";
    return false;
  }
  kernels::sm87_target_aot_nvfp4_oracle_detail::RawV1Arguments control;
  control.role = kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
  control.input = reinterpret_cast<const std::uint16_t*>(device.input.data());
  control.payload = device.gate_payload.data();
  control.token_count = kRows;
  control.tensor_scale0 = 1.0F;
  control.tensor_scale1 = 1.0F;
  control.output_or_residual =
      reinterpret_cast<std::uint16_t*>(device.h.data());
  control.cuda_stream = reinterpret_cast<void*>(stream);
  const int gate_status =
      kernels::sm87_target_aot_nvfp4_oracle_detail::launch_raw_v1(control);
  control.role = kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
  control.input = reinterpret_cast<const std::uint16_t*>(device.h.data());
  control.payload = device.down_payload.data();
  control.tensor_scale0 = 1.0F;
  control.tensor_scale1 = 0.0F;
  control.output_or_residual =
      reinterpret_cast<std::uint16_t*>(device.residual.data());
  const int down_status =
      gate_status == static_cast<int>(cudaSuccess)
          ? kernels::sm87_target_aot_nvfp4_oracle_detail::launch_raw_v1(
                control)
          : gate_status;
  std::vector<std::uint16_t> control_h(candidate_h.size(), 0U);
  std::vector<std::uint16_t> control_residual(candidate_residual.size(), 0U);
  if (gate_status != static_cast<int>(cudaSuccess) ||
      down_status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !copy_from_device(control_h.data(), device.h, stream) ||
      !copy_from_device(control_residual.data(), device.residual, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !compare_exact(candidate_h, control_h, kIntermediate, case_name,
                     "v2/v1 H") ||
      !compare_exact(candidate_residual, control_residual, kHidden,
                     case_name, "v2/v1 residual")) {
    std::cerr << case_name << ": same-ELF v1 differential failed\n";
    return false;
  }

  std::vector<std::uint16_t> expected_h(kRows * kIntermediate, 0U);
  auto expected_residual = residual;
  if (encoded) {
    const std::uint16_t h_bits = encode_bf16_rne(
        1.0F / (1.0F + std::exp(-1.0F)));
    for (std::size_t row = 0U; row < kRows; ++row) {
      expected_h[row * kIntermediate + intermediate_k] = h_bits;
      for (std::size_t column = 0U; column < kHidden; ++column) {
        expected_residual[row * kHidden + column] = encode_bf16_rne(
            decode_bf16(h_bits) +
            decode_bf16(residual[row * kHidden + column]));
      }
    }
  }
  return compare_exact(candidate_h, expected_h, kIntermediate, case_name,
                       "independent H") &&
         compare_exact(candidate_residual, expected_residual, kHidden,
                       case_name, "independent residual") &&
         device.guards_unchanged();
}

[[nodiscard]] bool run_cancellation_case(
    DeviceInventory& device,
    const kernels::Sm87TargetAotProjectionPackedLayout& gate_layout,
    const kernels::Sm87TargetAotProjectionPackedLayout& down_layout,
    const cudaStream_t stream) {
  std::vector<std::uint16_t> input(kRows * kHidden, kBf16One);
  std::vector<std::uint16_t> residual(kRows * kHidden,
                                      kResidualPositive);
  std::vector<std::uint8_t> gate_payload(gate_layout.payload_bytes, 0U);
  std::vector<std::uint8_t> down_payload(down_layout.payload_bytes, 0U);
  if (!initialize_case_device(device, input, gate_payload, down_payload,
                              residual, 1U, stream)) {
    return false;
  }
  std::vector<std::uint8_t> h_before(device.h.bytes(), 0U);
  if (cudaMemsetAsync(device.h.data(), 0x5a, device.h.bytes(), stream) !=
          cudaSuccess ||
      !copy_from_device(h_before.data(), device.h, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }
  auto arguments = candidate_arguments(device, stream, 91U);
  const int status =
      kernels::sm87_bulk_v2_nvfp4_oracle_detail::launch_raw_tail(arguments);
  std::vector<std::uint8_t> h_after(h_before.size(), 0U);
  std::vector<std::uint16_t> residual_after(residual.size(), 0U);
  kernels::Sm87BulkV2NvFp4DeviceControl control{};
  if (status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !copy_from_device(h_after.data(), device.h, stream) ||
      !copy_from_device(residual_after.data(), device.residual, stream) ||
      !copy_from_device(&control, device.control, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }
  const bool h_unchanged = h_after == h_before;
  const bool residual_unchanged = residual_after == residual;
  const bool guards = device.guards_unchanged();
  if (!h_unchanged || !residual_unchanged ||
      control.cancellation_observed != 1U ||
      control.completed_down_tasks != 0U || !guards) {
    std::cerr << "cancellation state: h=" << h_unchanged
              << " residual=" << residual_unchanged
              << " observed=" << control.cancellation_observed
              << " down=" << control.completed_down_tasks
              << " claimed=" << control.claimed_gate_tasks
              << " completed_gate=" << control.completed_gate_tasks
              << " guards=" << guards << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] bool run_m1024_macro_oracle(
    DeviceInventory& payload_device,
    const kernels::Sm87TargetAotProjectionPackedLayout& gate_layout,
    const kernels::Sm87TargetAotProjectionPackedLayout& down_layout,
    const cudaStream_t stream) {
  constexpr std::size_t kMacroRows =
      kernels::kSm87BulkV2NvFp4MacroTokens;
  constexpr std::size_t kGroupRows =
      kernels::kSm87BulkV2NvFp4GroupTokens;
  std::vector<std::uint16_t> input(kMacroRows * kHidden, 0U);
  std::vector<std::uint16_t> residual(kMacroRows * kHidden, 0U);
  std::vector<std::uint8_t> gate_payload(gate_layout.payload_bytes, 0U);
  std::vector<std::uint8_t> down_payload(down_layout.payload_bytes, 0U);
  initialize_unit_scales(gate_payload, gate_layout);
  initialize_unit_scales(down_payload, down_layout);
  constexpr std::size_t kInputK = kHidden - 1U;
  constexpr std::size_t kIntermediateK = kIntermediate - 1U;
  if (!set_weight(gate_payload, gate_layout, 0U, kIntermediateK,
                  kInputK, kWeightOne) ||
      !set_weight(gate_payload, gate_layout, 1U, kIntermediateK,
                  kInputK, kWeightOne)) {
    return false;
  }
  for (std::size_t n = 0U; n < kHidden; ++n) {
    if (!set_weight(down_payload, down_layout, 0U, n, kIntermediateK,
                    kWeightOne)) {
      return false;
    }
  }
  for (std::size_t row = 0U; row < kMacroRows; ++row) {
    input[row * kHidden + kInputK] = kBf16One;
    for (std::size_t column = 0U; column < kHidden; ++column) {
      residual[row * kHidden + column] =
          (column & 1U) == 0U ? kResidualPositive : kResidualNegative;
    }
  }

  GuardedDeviceBuffer macro_input;
  GuardedDeviceBuffer macro_residual;
  GuardedDeviceBuffer group_h;
  GuardedDeviceBuffer v1_h;
  GuardedDeviceBuffer control_buffer;
  GuardedDeviceBuffer cancellation;
  if (!macro_input.allocate(input.size() * sizeof(std::uint16_t)) ||
      !macro_residual.allocate(residual.size() * sizeof(std::uint16_t)) ||
      !group_h.allocate(kGroupRows * kIntermediate *
                        sizeof(std::uint16_t)) ||
      !v1_h.allocate(kMacroRows * kIntermediate * sizeof(std::uint16_t)) ||
      !control_buffer.allocate(
          sizeof(kernels::Sm87BulkV2NvFp4DeviceControl)) ||
      !cancellation.allocate(sizeof(std::uint32_t)) ||
      !macro_input.initialize(stream) || !macro_residual.initialize(stream) ||
      !group_h.initialize(stream) || !v1_h.initialize(stream) ||
      !control_buffer.initialize(stream) || !cancellation.initialize(stream) ||
      !payload_device.gate_payload.initialize(stream) ||
      !payload_device.down_payload.initialize(stream) ||
      !copy_to_device(macro_input, input.data(), stream) ||
      !copy_to_device(macro_residual, residual.data(), stream) ||
      !copy_to_device(payload_device.gate_payload, gate_payload.data(),
                      stream) ||
      !copy_to_device(payload_device.down_payload, down_payload.data(),
                      stream) ||
      cudaMemsetAsync(group_h.data(), 0, group_h.bytes(), stream) !=
          cudaSuccess ||
      cudaMemsetAsync(control_buffer.data(), 0, control_buffer.bytes(),
                      stream) != cudaSuccess ||
      cudaMemsetAsync(cancellation.data(), 0, cancellation.bytes(), stream) !=
          cudaSuccess) {
    std::cerr << "M1024 initialization failed\n";
    return false;
  }

  kernels::sm87_bulk_v2_nvfp4_oracle_detail::RawTailArguments candidate;
  candidate.normalized_input = reinterpret_cast<const std::uint16_t*>(
      macro_input.data());
  candidate.gate_up_payload = payload_device.gate_payload.data();
  candidate.gate_tensor_scale = 1.0F;
  candidate.up_tensor_scale = 1.0F;
  candidate.down_payload = payload_device.down_payload.data();
  candidate.down_tensor_scale = 1.0F;
  candidate.residual =
      reinterpret_cast<std::uint16_t*>(macro_residual.data());
  candidate.group_h_scratch =
      reinterpret_cast<std::uint16_t*>(group_h.data());
  candidate.device_control =
      reinterpret_cast<kernels::Sm87BulkV2NvFp4DeviceControl*>(
          control_buffer.data());
  candidate.cancellation_signal =
      reinterpret_cast<const std::uint32_t*>(cancellation.data());
  candidate.group_epoch = 101U;
  candidate.token_count = kMacroRows;
  candidate.cuda_stream = reinterpret_cast<void*>(stream);
  const int candidate_status =
      kernels::sm87_bulk_v2_nvfp4_oracle_detail::launch_raw_macro(candidate);
  if (candidate_status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << "M1024 candidate launch failed: " << candidate_status
              << '\n';
    return false;
  }

  std::vector<std::uint16_t> candidate_h(kGroupRows * kIntermediate, 0U);
  std::vector<std::uint16_t> candidate_residual(residual.size(), 0U);
  kernels::Sm87BulkV2NvFp4DeviceControl observed{};
  if (!copy_from_device(candidate_h.data(), group_h, stream) ||
      !copy_from_device(candidate_residual.data(), macro_residual, stream) ||
      !copy_from_device(&observed, control_buffer, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      observed.active_group != 3U || observed.active_rows != 4U ||
      observed.claimed_gate_tasks != 4U * 272U ||
      observed.completed_gate_tasks != 4U * 272U ||
      observed.completed_down_tasks != 4U * 20U ||
      observed.macro_completed_groups != 4U ||
      observed.macro_claimed_gate_tasks != 16U * 272U ||
      observed.macro_completed_gate_tasks != 16U * 272U ||
      observed.macro_completed_down_tasks != 16U * 20U ||
      observed.cancellation_observed != 0U) {
    std::cerr << "M1024 completion control mismatch\n";
    return false;
  }

  if (cudaMemsetAsync(v1_h.data(), 0, v1_h.bytes(), stream) != cudaSuccess ||
      !copy_to_device(macro_residual, residual.data(), stream)) {
    return false;
  }
  kernels::sm87_target_aot_nvfp4_oracle_detail::RawV1Arguments v1;
  v1.role = kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
  v1.input = reinterpret_cast<const std::uint16_t*>(macro_input.data());
  v1.payload = payload_device.gate_payload.data();
  v1.token_count = kMacroRows;
  v1.tensor_scale0 = 1.0F;
  v1.tensor_scale1 = 1.0F;
  v1.output_or_residual = reinterpret_cast<std::uint16_t*>(v1_h.data());
  v1.cuda_stream = reinterpret_cast<void*>(stream);
  const int gate_status =
      kernels::sm87_target_aot_nvfp4_oracle_detail::launch_raw_v1(v1);
  v1.role = kernels::Sm87TargetAotProjectionRole::kNvFp4Down;
  v1.input = reinterpret_cast<const std::uint16_t*>(v1_h.data());
  v1.payload = payload_device.down_payload.data();
  v1.tensor_scale0 = 1.0F;
  v1.tensor_scale1 = 0.0F;
  v1.output_or_residual =
      reinterpret_cast<std::uint16_t*>(macro_residual.data());
  const int down_status =
      gate_status == static_cast<int>(cudaSuccess)
          ? kernels::sm87_target_aot_nvfp4_oracle_detail::launch_raw_v1(v1)
          : gate_status;
  std::vector<std::uint16_t> v1_residual(residual.size(), 0U);
  std::vector<std::uint16_t> v1_last_h(candidate_h.size(), 0U);
  if (gate_status != static_cast<int>(cudaSuccess) ||
      down_status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !copy_from_device(v1_residual.data(), macro_residual, stream) ||
      cudaMemcpyAsync(
          v1_last_h.data(),
          v1_h.data() +
              3U * kGroupRows * kIntermediate * sizeof(std::uint16_t),
          v1_last_h.size() * sizeof(std::uint16_t), cudaMemcpyDeviceToHost,
          stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !compare_exact(candidate_h, v1_last_h, kIntermediate, "M1024",
                     "v2/v1 last-group H") ||
      !compare_exact(candidate_residual, v1_residual, kHidden, "M1024",
                     "v2/v1 residual")) {
    std::cerr << "M1024 same-ELF differential failed\n";
    return false;
  }

  const std::uint16_t h_bits =
      encode_bf16_rne(1.0F / (1.0F + std::exp(-1.0F)));
  std::vector<std::uint16_t> expected_h(candidate_h.size(), 0U);
  auto expected_residual = residual;
  for (std::size_t row = 0U; row < kGroupRows; ++row) {
    expected_h[row * kIntermediate + kIntermediateK] = h_bits;
  }
  for (std::size_t row = 0U; row < kMacroRows; ++row) {
    for (std::size_t column = 0U; column < kHidden; ++column) {
      expected_residual[row * kHidden + column] = encode_bf16_rne(
          decode_bf16(h_bits) +
          decode_bf16(residual[row * kHidden + column]));
    }
  }
  return compare_exact(candidate_h, expected_h, kIntermediate, "M1024",
                       "independent last-group H") &&
         compare_exact(candidate_residual, expected_residual, kHidden,
                       "M1024", "independent residual") &&
         macro_input.guards_unchanged() &&
         macro_residual.guards_unchanged() && group_h.guards_unchanged() &&
         v1_h.guards_unchanged() && control_buffer.guards_unchanged() &&
         cancellation.guards_unchanged() &&
         payload_device.gate_payload.guards_unchanged() &&
         payload_device.down_payload.guards_unchanged();
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

  // Runtime CUDA attributes qualify the hardware half of the resource gate.
  // Offline SASS identity/row evidence remains a separate retained artifact,
  // so pass no fabricated code evidence here and require that admission stay
  // fail-closed despite the observed two-CTA residency.
  kernels::Sm87BulkV2NvFp4TailNumericalResources resources{};
  const int resource_status =
      kernels::query_sm87_bulk_dataflow_v2_nvfp4_tail_numerical_resources_cuda(
          nullptr, &resources);
  if (resource_status != static_cast<int>(cudaSuccess) ||
      resources.kernel.binary_version != 87 ||
      resources.kernel.registers_per_thread <= 0 ||
      resources.kernel.registers_per_thread > 128 ||
      resources.kernel.static_shared_bytes != 0U ||
      resources.kernel.dynamic_shared_bytes != 52'224U ||
      resources.kernel.local_bytes != 0U ||
      resources.kernel.active_blocks_per_sm < 2 ||
      resources.kernel.cooperative_grid_capacity < 32 ||
      resources.kernel.resource_and_code_gate_passed ||
      !resources.exact_control_stepping_stone ||
      resources.cross_group_weight_residency_qualified ||
      resources.p40_hot_path_qualified ||
      kernels::sm87_bulk_v2_nvfp4_tail_numerical_resources_valid(
          resources)) {
    std::cerr << "NVFP4 v2 runtime resource observation failed\n";
    return 1;
  }
  std::cout << "NVFP4 v2 tail resources: regs="
            << resources.kernel.registers_per_thread
            << " local=" << resources.kernel.local_bytes
            << " dynamic_smem=" << resources.kernel.dynamic_shared_bytes
            << " active_cta_per_sm="
            << resources.kernel.active_blocks_per_sm << '\n';

  const auto gate_layout =
      kernels::sm87_target_aot_projection_packed_layout(
          kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp);
  const auto down_layout =
      kernels::sm87_target_aot_projection_packed_layout(
          kernels::Sm87TargetAotProjectionRole::kNvFp4Down);
  if (!gate_layout.valid() || !down_layout.valid()) {
    return 1;
  }
  DeviceInventory device;
  Stream stream;
  if (!device.allocate(static_cast<std::size_t>(gate_layout.payload_bytes),
                       static_cast<std::size_t>(down_layout.payload_bytes)) ||
      !stream.create()) {
    std::cerr << "NVFP4 oracle allocation failed\n";
    return 1;
  }

  struct OracleCase final {
    const char* name;
    bool encoded;
    std::size_t input_k;
    std::size_t intermediate_k;
  };
  constexpr std::array<OracleCase, 3U> cases{{
      {"all-zero-payload", false, 0U, 0U},
      {"first-k16-boundary", true, 0U, 0U},
      {"last-k16-boundary", true, kHidden - 1U, kIntermediate - 1U},
  }};
  std::uint32_t epoch = 1U;
  for (const auto& oracle_case : cases) {
    if (!run_case(oracle_case.name, oracle_case.encoded,
                  oracle_case.input_k, oracle_case.intermediate_k,
                  gate_layout, down_layout, device, stream.get(), epoch++)) {
      return 1;
    }
  }
  if (!run_cancellation_case(device, gate_layout, down_layout, stream.get())) {
    std::cerr << "cancellation/discard oracle failed\n";
    return 1;
  }
  if (!run_m1024_macro_oracle(device, gate_layout, down_layout,
                              stream.get())) {
    std::cerr << "M1024 macro oracle failed\n";
    return 1;
  }
  std::cout << "SM87 bulk-dataflow-v2 NVFP4 exact CUDA oracle passed\n";
  return 0;
}
