#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40.h"

#include "sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_oracle_internal.h"
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
constexpr std::size_t kRows =
    kernels::kSm87BulkV2NvFp4GateUpWholeP40TileM;
constexpr std::size_t kHidden = kernels::kSm87BulkV2NvFp4Hidden;
constexpr std::size_t kIntermediate =
    kernels::kSm87BulkV2NvFp4Intermediate;
constexpr std::size_t kGuardBytes = 256U;
constexpr std::uint8_t kGuardValue = 0xa5U;
constexpr std::uint8_t kPoisonValue = 0x6dU;
constexpr std::uint8_t kScaleOne = 0x38U;
constexpr std::uint8_t kWeightOne = 0x02U;
constexpr std::uint16_t kBf16One = 0x3f80U;
constexpr std::size_t kInputK = kHidden - 1U;
constexpr std::size_t kOutputN = 63U;

static_assert(kRows == 64U && kHidden == 5'120U &&
              kIntermediate == 17'408U);
static_assert(sizeof(kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl) ==
              64U);

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

  [[nodiscard]] bool initialize_guards(const cudaStream_t stream) noexcept {
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

[[nodiscard]] std::uint16_t encode_bf16_rne(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
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
  const auto shift = static_cast<std::uint8_t>(address.nibble * 4U);
  byte = static_cast<std::uint8_t>(
      (byte & static_cast<std::uint8_t>(~(0x0fU << shift))) |
      static_cast<std::uint8_t>((code & 0x0fU) << shift));
  return true;
}

[[nodiscard]] bool compare_exact(
    const std::vector<std::uint16_t>& observed,
    const std::vector<std::uint16_t>& expected,
    const char* const observed_name, const char* const expected_name) {
  for (std::size_t index = 0U; index < observed.size(); ++index) {
    if (observed[index] != expected[index]) {
      std::cerr << observed_name << " differs from " << expected_name
                << " at row " << index / kIntermediate << " column "
                << index % kIntermediate << " expected 0x" << std::hex
                << expected[index] << " observed 0x" << observed[index]
                << std::dec << '\n';
      return false;
    }
  }
  return true;
}

struct DeviceInventory final {
  GuardedDeviceBuffer input;
  GuardedDeviceBuffer payload;
  GuardedDeviceBuffer h;
  GuardedDeviceBuffer control;
  GuardedDeviceBuffer cancellation;

  [[nodiscard]] bool allocate(const std::size_t payload_bytes) {
    return input.allocate(kRows * kHidden * sizeof(std::uint16_t)) &&
           payload.allocate(payload_bytes) &&
           h.allocate(kRows * kIntermediate * sizeof(std::uint16_t)) &&
           control.allocate(sizeof(
               kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl)) &&
           cancellation.allocate(sizeof(std::uint32_t));
  }

  [[nodiscard]] bool initialize_guards(const cudaStream_t stream) {
    return input.initialize_guards(stream) &&
           payload.initialize_guards(stream) && h.initialize_guards(stream) &&
           control.initialize_guards(stream) &&
           cancellation.initialize_guards(stream);
  }

  [[nodiscard]] bool guards_unchanged() const {
    return input.guards_unchanged() && payload.guards_unchanged() &&
           h.guards_unchanged() && control.guards_unchanged() &&
           cancellation.guards_unchanged();
  }
};

[[nodiscard]] kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail::
    RawArguments
candidate_arguments(DeviceInventory& device, const cudaStream_t stream,
                    const std::uint64_t epoch) {
  kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail::RawArguments
      arguments;
  arguments.transaction_epoch = epoch;
  arguments.normalized_input =
      reinterpret_cast<const std::uint16_t*>(device.input.data());
  arguments.gate_up_payload = device.payload.data();
  arguments.gate_tensor_scale = 1.0F;
  arguments.up_tensor_scale = 1.0F;
  arguments.h = reinterpret_cast<std::uint16_t*>(device.h.data());
  arguments.device_control = reinterpret_cast<
      kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl*>(
      device.control.data());
  arguments.cancellation_signal =
      reinterpret_cast<const std::uint32_t*>(device.cancellation.data());
  arguments.m_tiles = 1U;
  arguments.n_tiles = 1U;
  arguments.cuda_stream = reinterpret_cast<void*>(stream);
  return arguments;
}

[[nodiscard]] bool control_is_complete(
    const kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl& control,
    const std::uint64_t epoch) {
  return control.transaction_epoch == epoch && control.expected_cells == 1U &&
         control.started_cells == 1U && control.completed_cells == 1U &&
         control.completed_ctas == 32U &&
         control.cancellation_observed == 0U &&
         control.launch_completed == 1U &&
         control.first_incomplete_cohort == 0xffff'ffffU;
}

[[nodiscard]] bool run_exact_oracle(
    DeviceInventory& device, const cudaStream_t stream,
    const kernels::Sm87TargetAotProjectionPackedLayout& layout,
    const std::vector<std::uint16_t>& input,
    const std::vector<std::uint8_t>& payload) {
  constexpr std::uint64_t kEpoch = 0x4'7557'5034ULL;
  const std::uint32_t cancellation = 0U;
  if (!device.initialize_guards(stream) ||
      !copy_to_device(device.input, input.data(), stream) ||
      !copy_to_device(device.payload, payload.data(), stream) ||
      cudaMemsetAsync(device.h.data(), 0, device.h.bytes(), stream) !=
          cudaSuccess ||
      cudaMemsetAsync(device.control.data(), 0, device.control.bytes(),
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(device.cancellation.data(), &cancellation,
                      sizeof(cancellation), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    std::cerr << "candidate initialization failed\n";
    return false;
  }

  auto candidate = candidate_arguments(device, stream, kEpoch);
  const int candidate_status =
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail::
          launch_raw(candidate);
  if (candidate_status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << "candidate launch failed: " << candidate_status << " / "
              << cudaGetErrorString(cudaGetLastError()) << '\n';
    return false;
  }

  std::vector<std::uint16_t> observed(kRows * kIntermediate, 0U);
  std::vector<std::uint16_t> input_after(input.size(), 0U);
  std::vector<std::uint8_t> payload_after(payload.size(), 0U);
  kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl control{};
  if (!copy_from_device(observed.data(), device.h, stream) ||
      !copy_from_device(input_after.data(), device.input, stream) ||
      !copy_from_device(payload_after.data(), device.payload, stream) ||
      !copy_from_device(&control, device.control, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << "candidate result copy failed\n";
    return false;
  }
  if (input_after != input || payload_after != payload) {
    std::cerr << "candidate modified immutable input or payload bytes\n";
    return false;
  }
  if (!control_is_complete(control, kEpoch) ||
      !device.guards_unchanged()) {
    std::cerr << "candidate receipt or guard validation failed\n";
    return false;
  }

  // Replay the identical candidate without replacing its immutable inputs.
  if (cudaMemsetAsync(device.h.data(), 0, device.h.bytes(), stream) !=
          cudaSuccess ||
      cudaMemsetAsync(device.control.data(), 0, device.control.bytes(),
                      stream) != cudaSuccess) {
    return false;
  }
  const int replay_status =
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail::
          launch_raw(candidate);
  std::vector<std::uint16_t> replay(observed.size(), 0U);
  kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl replay_control{};
  if (replay_status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !copy_from_device(replay.data(), device.h, stream) ||
      !copy_from_device(&replay_control, device.control, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess || replay != observed ||
      !control_is_complete(replay_control, kEpoch)) {
    std::cerr << "candidate deterministic replay failed\n";
    return false;
  }

  // Same-ELF v1 consumes the identical M64 activation and authenticated raw
  // Gate+Up payload.  It is an exact differential oracle, not a production
  // alternative or a performance reference.
  if (cudaMemsetAsync(device.h.data(), 0, device.h.bytes(), stream) !=
      cudaSuccess) {
    return false;
  }
  kernels::sm87_target_aot_nvfp4_oracle_detail::RawV1Arguments v1;
  v1.role = kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp;
  v1.input =
      reinterpret_cast<const std::uint16_t*>(device.input.data());
  v1.payload = device.payload.data();
  v1.token_count = kRows;
  v1.tensor_scale0 = 1.0F;
  v1.tensor_scale1 = 1.0F;
  v1.output_or_residual =
      reinterpret_cast<std::uint16_t*>(device.h.data());
  v1.cuda_stream = reinterpret_cast<void*>(stream);
  const int v1_status =
      kernels::sm87_target_aot_nvfp4_oracle_detail::launch_raw_v1(v1);
  std::vector<std::uint16_t> v1_output(observed.size(), 0U);
  if (v1_status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !copy_from_device(v1_output.data(), device.h, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !compare_exact(observed, v1_output, "candidate", "same-ELF v1")) {
    std::cerr << "same-ELF v1 differential failed\n";
    return false;
  }

  std::vector<std::uint16_t> independent_expected(observed.size(), 0U);
  const std::uint16_t expected_silu =
      encode_bf16_rne(1.0F / (1.0F + std::exp(-1.0F)));
  for (std::size_t row = 0U; row < kRows; ++row) {
    independent_expected[row * kIntermediate + kOutputN] = expected_silu;
  }
  if (!compare_exact(observed, independent_expected, "candidate",
                     "independent SiLU(1)*1") ||
      !device.guards_unchanged()) {
    std::cerr << "independent mathematical oracle failed\n";
    return false;
  }

  (void)layout;
  return true;
}

[[nodiscard]] bool run_cancellation_oracle(DeviceInventory& device,
                                           const cudaStream_t stream) {
  constexpr std::uint64_t kEpoch = 0x4'7557'5035ULL;
  const std::uint32_t cancellation = 1U;
  if (cudaMemsetAsync(device.h.data(), kPoisonValue, device.h.bytes(),
                      stream) != cudaSuccess ||
      cudaMemsetAsync(device.control.data(), 0, device.control.bytes(),
                      stream) != cudaSuccess ||
      cudaMemcpyAsync(device.cancellation.data(), &cancellation,
                      sizeof(cancellation), cudaMemcpyHostToDevice,
                      stream) != cudaSuccess) {
    return false;
  }
  const auto candidate = candidate_arguments(device, stream, kEpoch);
  const int status =
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_oracle_detail::
          launch_raw(candidate);
  std::vector<std::uint16_t> observed(kRows * kIntermediate, 0U);
  kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl control{};
  if (status != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      !copy_from_device(observed.data(), device.h, stream) ||
      !copy_from_device(&control, device.control, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    std::cerr << "pre-cancelled candidate failed to return cleanly\n";
    return false;
  }
  if (!std::all_of(observed.begin(), observed.end(), [](const auto value) {
        return value == 0x6d6dU;
      }) ||
      control.transaction_epoch != kEpoch || control.expected_cells != 1U ||
      control.started_cells != 0U || control.completed_cells != 0U ||
      control.completed_ctas != 32U ||
      control.cancellation_observed != 1U ||
      control.launch_completed != 0U ||
      control.first_incomplete_cohort != 0U ||
      !device.guards_unchanged()) {
    std::cerr << "pre-cancelled publication or receipt mismatch\n";
    return false;
  }
  return true;
}

[[nodiscard]] bool runtime_resource_envelope() {
  kernels::Sm87BulkV2NvFp4GateUpWholeP40Resources resources;
  const int status =
      kernels::query_sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_resources_cuda(
          nullptr, &resources);
  if (status != static_cast<int>(cudaSuccess) ||
      resources.binary_version != 87 ||
      resources.registers_per_thread <= 0 ||
      resources.registers_per_thread > 128 ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 38'400U ||
      resources.local_bytes != 0U ||
      resources.maximum_threads_per_block < 256 ||
      resources.active_blocks_per_sm < 2 ||
      resources.cooperative_grid_capacity < 32 ||
      !resources.cooperative_launch_supported ||
      resources.exact_oracle_attached || resources.resource_gate_passed ||
      resources.numerical_contract_qualified ||
      resources.production_dispatch_eligible ||
      kernels::sm87_bulk_v2_nvfp4_gate_up_whole_p40_resources_valid(
          resources)) {
    std::cerr << "runtime resource envelope failed: status=" << status
              << " regs=" << resources.registers_per_thread
              << " local=" << resources.local_bytes
              << " active_blocks=" << resources.active_blocks_per_sm
              << " cooperative_capacity="
              << resources.cooperative_grid_capacity << '\n';
    return false;
  }
  std::cout << "runtime resource envelope: "
            << resources.registers_per_thread
            << " registers/thread, " << resources.active_blocks_per_sm
            << " CTA/SM, zero runtime local bytes; static code gate remains "
               "open pending hash-bound ptxas/cuobjdump evidence\n";
  return true;
}

}  // namespace

int main() {
  int device_count = 0;
  if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
      device_count == 0) {
    return kSkip;
  }
  int device = -1;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16 ||
      properties.cooperativeLaunch == 0) {
    return kSkip;
  }

  const auto layout = kernels::sm87_target_aot_projection_packed_layout(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp);
  if (!layout.valid() || layout.partition_count != 2U ||
      layout.payload_bytes !=
          kernels::kSm87BulkV2NvFp4GateUpPayloadBytes) {
    std::cerr << "authenticated Gate+Up layout mismatch\n";
    return 1;
  }

  std::vector<std::uint16_t> input(kRows * kHidden, 0U);
  for (std::size_t row = 0U; row < kRows; ++row) {
    input[row * kHidden + kInputK] = kBf16One;
  }
  std::vector<std::uint8_t> payload(
      static_cast<std::size_t>(layout.payload_bytes), 0U);
  initialize_unit_scales(payload, layout);
  if (!set_weight(payload, layout, 0U, kOutputN, kInputK, kWeightOne) ||
      !set_weight(payload, layout, 1U, kOutputN, kInputK, kWeightOne)) {
    std::cerr << "one-hot Gate/Up payload encoding failed\n";
    return 1;
  }

  Stream stream;
  DeviceInventory device_inventory;
  if (!stream.create() ||
      !device_inventory.allocate(static_cast<std::size_t>(
          layout.payload_bytes))) {
    std::cerr << "Gate+Up oracle allocation failed\n";
    return 1;
  }
  if (!run_exact_oracle(device_inventory, stream.get(), layout, input,
                        payload) ||
      !run_cancellation_oracle(device_inventory, stream.get()) ||
      !runtime_resource_envelope()) {
    return 1;
  }
  std::cout << "SM87 Gate+Up whole-P40 exact CUDA oracle: PASS\n";
  return 0;
}
