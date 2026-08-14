#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_projection.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_whole_p40.h"

#include "sm87_bulk_dataflow_v2_fp8_whole_p40_oracle_internal.h"
#include "sm87_target_aot_projection_fp8_oracle_internal.h"

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

using Role = kernels::Sm87TargetAotProjectionRole;

constexpr int kSkip = 77;
constexpr std::size_t kRows = 64U;
constexpr std::size_t kGuardBytes = 256U;
constexpr std::uint8_t kGuardValue = 0xa5U;

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
    pointer_ = allocation_ + kGuardBytes;
    return true;
  }
  [[nodiscard]] bool initialize(const cudaStream_t stream,
                                const std::uint8_t value = 0U) noexcept {
    return allocation_ != nullptr &&
           cudaMemsetAsync(allocation_, kGuardValue,
                           bytes_ + 2U * kGuardBytes,
                           stream) == cudaSuccess &&
           cudaMemsetAsync(pointer_, value, bytes_, stream) == cudaSuccess;
  }
  [[nodiscard]] std::uint8_t* data() noexcept { return pointer_; }
  [[nodiscard]] const std::uint8_t* data() const noexcept { return pointer_; }
  [[nodiscard]] std::size_t bytes() const noexcept { return bytes_; }
  [[nodiscard]] bool guards_unchanged() const {
    std::array<std::uint8_t, kGuardBytes> prefix{};
    std::array<std::uint8_t, kGuardBytes> suffix{};
    if (cudaMemcpy(prefix.data(), allocation_, prefix.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess ||
        cudaMemcpy(suffix.data(), pointer_ + bytes_, suffix.size(),
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return false;
    }
    return std::all_of(prefix.begin(), prefix.end(), [](const auto value) {
             return value == kGuardValue;
           }) &&
           std::all_of(suffix.begin(), suffix.end(), [](const auto value) {
             return value == kGuardValue;
           });
  }

 private:
  std::uint8_t* allocation_ = nullptr;
  std::uint8_t* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
};

[[nodiscard]] bool upload(GuardedDeviceBuffer& destination,
                          const void* const source,
                          const std::size_t bytes,
                          const cudaStream_t stream) noexcept {
  return destination.bytes() == bytes &&
         cudaMemcpyAsync(destination.data(), source, bytes,
                         cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

[[nodiscard]] bool download(void* const destination,
                            const GuardedDeviceBuffer& source,
                            const cudaStream_t stream) noexcept {
  return cudaMemcpyAsync(destination, source.data(), source.bytes(),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

[[nodiscard]] const char* role_name(const Role role) noexcept {
  if (role == Role::kFp8GdnQkvZ) {
    return "GDN-QKVZ";
  }
  if (role == Role::kFp8FullQkv) {
    return "Full-QKV";
  }
  return "Attention-O";
}

[[nodiscard]] float float_from_bf16(const std::uint16_t bits) noexcept {
  const std::uint32_t bits32 = static_cast<std::uint32_t>(bits) << 16U;
  float result = 0.0F;
  std::memcpy(&result, &bits32, sizeof(result));
  return result;
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

[[nodiscard]] std::uint16_t expected_product(
    const std::uint8_t code,
    const std::uint16_t compensated_scale_bits) noexcept {
  if ((code & 0x7fU) == 0U) {
    return 0U;
  }
  const float biased = float_from_bf16(
      kernels::sm87_bulk_v2_fp8_whole_p40_raw_code_to_biased_bf16_bits(
          code));
  return encode_bf16_rne(
      std::fma(1.0F, biased, 0.0F) *
      float_from_bf16(compensated_scale_bits));
}

[[nodiscard]] std::uint8_t asymmetric_code(
    const std::size_t partition, const std::size_t column,
    const std::size_t k) noexcept {
  if (column % 257U == 0U) {
    return 0x7fU;
  }
  if (column % 257U == 1U) {
    return 0xffU;
  }
  if (column % 257U == 2U) {
    return 0x00U;
  }
  if (column % 257U == 3U) {
    return 0x80U;
  }
  return static_cast<std::uint8_t>(
      partition * 67U + (column / 128U) * 43U + column * 13U +
      (k / 16U) * 29U + (k % 16U) * 7U);
}

struct HostOutputs final {
  std::vector<std::uint16_t> primary;
  std::vector<std::uint16_t> secondary;
  std::vector<std::uint16_t> tertiary;
};

struct DeviceOutputs final {
  GuardedDeviceBuffer primary;
  GuardedDeviceBuffer secondary;
  GuardedDeviceBuffer tertiary;
};

[[nodiscard]] bool allocate_outputs(
    DeviceOutputs& device, const kernels::Sm87BulkV2Fp8WholeP40RolePlan& plan,
    HostOutputs& host) {
  host.primary.resize(kRows * plan.primary_output_features);
  host.secondary.resize(kRows * plan.secondary_output_features);
  host.tertiary.resize(kRows * plan.tertiary_output_features);
  return device.primary.allocate(host.primary.size() * sizeof(std::uint16_t)) &&
         (host.secondary.empty() ||
          device.secondary.allocate(
              host.secondary.size() * sizeof(std::uint16_t))) &&
         (host.tertiary.empty() ||
          device.tertiary.allocate(
              host.tertiary.size() * sizeof(std::uint16_t)));
}

[[nodiscard]] bool initialize_outputs(DeviceOutputs& device,
                                      const HostOutputs& host,
                                      const cudaStream_t stream,
                                      const std::uint8_t value = 0U) {
  return device.primary.initialize(stream, value) &&
         (host.secondary.empty() ||
          device.secondary.initialize(stream, value)) &&
         (host.tertiary.empty() ||
          device.tertiary.initialize(stream, value));
}

[[nodiscard]] bool download_outputs(HostOutputs& host,
                                    const DeviceOutputs& device,
                                    const cudaStream_t stream) {
  return download(host.primary.data(), device.primary, stream) &&
         (host.secondary.empty() ||
          download(host.secondary.data(), device.secondary, stream)) &&
         (host.tertiary.empty() ||
          download(host.tertiary.data(), device.tertiary, stream));
}

[[nodiscard]] bool output_guards_unchanged(
    const HostOutputs& host, const DeviceOutputs& device) {
  return device.primary.guards_unchanged() &&
         (host.secondary.empty() || device.secondary.guards_unchanged()) &&
         (host.tertiary.empty() || device.tertiary.guards_unchanged());
}

[[nodiscard]] std::uint16_t* device_secondary(
    DeviceOutputs& outputs, const HostOutputs& host) noexcept {
  return host.secondary.empty()
             ? nullptr
             : reinterpret_cast<std::uint16_t*>(outputs.secondary.data());
}

[[nodiscard]] std::uint16_t* device_tertiary(
    DeviceOutputs& outputs, const HostOutputs& host) noexcept {
  return host.tertiary.empty()
             ? nullptr
             : reinterpret_cast<std::uint16_t*>(outputs.tertiary.data());
}

void set_expected(HostOutputs& expected,
                  const kernels::Sm87TargetAotProjectionPackedLayout& layout,
                  const std::size_t partition, const std::size_t column,
                  const std::uint16_t value) {
  const auto role = layout.role;
  const auto& descriptor = layout.partitions[partition];
  for (std::size_t row = 0U; row < kRows; ++row) {
    if (role == Role::kFp8FullQkv) {
      auto* destination = partition == 0U
                              ? &expected.primary
                              : (partition == 1U ? &expected.secondary
                                                 : &expected.tertiary);
      const std::size_t width = descriptor.output_features;
      (*destination)[row * width + column] = value;
    } else {
      expected.primary[
          row * layout.projected_output_features +
          descriptor.global_n_offset + column] = value;
    }
  }
}

[[nodiscard]] bool compare_vector(const std::vector<std::uint16_t>& left,
                                  const std::vector<std::uint16_t>& right,
                                  const char* const role,
                                  const char* const name) {
  if (left.size() != right.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (left[index] != right[index]) {
      std::cerr << role << ' ' << name << " mismatch index=" << index
                << " expected=0x" << std::hex << right[index]
                << " observed=0x" << left[index] << std::dec << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool compare_outputs(const HostOutputs& left,
                                   const HostOutputs& right,
                                   const char* const role,
                                   const char* const name) {
  return compare_vector(left.primary, right.primary, role, name) &&
         compare_vector(left.secondary, right.secondary, role, name) &&
         compare_vector(left.tertiary, right.tertiary, role, name);
}

[[nodiscard]] bool run_role(const Role role, const cudaStream_t stream) {
  const auto plan =
      kernels::sm87_bulk_v2_fp8_whole_p40_role_plan(role);
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  if (!plan.valid || !layout.valid()) {
    return false;
  }

  std::vector<std::uint16_t> input(kRows * plan.input_features, 0U);
  std::vector<std::uint16_t> input_after(input.size(), 0U);
  std::vector<std::uint8_t> payload(layout.payload_bytes, 0U);
  std::vector<std::uint8_t> payload_after(payload.size(), 0U);
  GuardedDeviceBuffer device_input;
  GuardedDeviceBuffer device_payload;
  GuardedDeviceBuffer device_control;
  GuardedDeviceBuffer device_cancellation;
  DeviceOutputs device_outputs;
  HostOutputs observed;
  if (!device_input.allocate(input.size() * sizeof(std::uint16_t)) ||
      !device_payload.allocate(payload.size()) ||
      !device_control.allocate(
          sizeof(kernels::Sm87BulkV2Fp8WholeP40DeviceControl)) ||
      !device_cancellation.allocate(sizeof(std::uint32_t)) ||
      !allocate_outputs(device_outputs, plan, observed)) {
    std::cerr << role_name(role) << ": allocation failed\n";
    return false;
  }

  constexpr std::array<std::uint16_t, 3U> scales{{
      kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
          0x3f80'0000U),
      kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
          0x3f00'0000U),
      kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
          0x4000'0000U)}};
  static_assert(scales[0U] != scales[1U] && scales[1U] != scales[2U]);
  const std::array<std::size_t, 5U> selected_k{{
      0U, 16U, 32U, 48U,
      static_cast<std::size_t>(plan.input_features - 1U)}};

  kernels::sm87_bulk_v2_fp8_whole_p40_oracle_detail::RawArguments candidate;
  candidate.transaction_epoch = 0x1000U +
                                static_cast<std::uint64_t>(role);
  candidate.role = role;
  candidate.input = reinterpret_cast<const std::uint16_t*>(
      device_input.data());
  candidate.payload = device_payload.data();
  candidate.compensated_scale_bf16_bits = scales;
  candidate.primary_output =
      reinterpret_cast<std::uint16_t*>(device_outputs.primary.data());
  candidate.secondary_output = device_secondary(device_outputs, observed);
  candidate.tertiary_output = device_tertiary(device_outputs, observed);
  candidate.device_control =
      reinterpret_cast<kernels::Sm87BulkV2Fp8WholeP40DeviceControl*>(
          device_control.data());
  candidate.m_tiles = 1U;
  candidate.n_tiles = plan.n_tiles;
  candidate.cuda_stream = reinterpret_cast<void*>(stream);

  for (const auto k : selected_k) {
    std::fill(input.begin(), input.end(), 0U);
    std::fill(payload.begin(), payload.end(), 0U);
    for (std::size_t row = 0U; row < kRows; ++row) {
      input[row * plan.input_features + k] = 0x3f80U;
    }
    HostOutputs expected;
    expected.primary.assign(observed.primary.size(), 0U);
    expected.secondary.assign(observed.secondary.size(), 0U);
    expected.tertiary.assign(observed.tertiary.size(), 0U);
    for (std::size_t partition = 0U;
         partition < layout.partition_count; ++partition) {
      for (std::size_t column = 0U;
           column < layout.partitions[partition].output_features; ++column) {
        const auto address =
            kernels::sm87_target_aot_projection_packed_weight_address(
                layout, partition, column, k);
        if (!address.valid || address.byte_offset >= payload.size()) {
          return false;
        }
        const auto code = asymmetric_code(partition, column, k);
        payload[address.byte_offset] = code;
        set_expected(expected, layout, partition, column,
                     expected_product(code, scales[partition]));
      }
    }

    if (!device_input.initialize(stream) ||
        !device_payload.initialize(stream) ||
        !device_control.initialize(stream) ||
        !device_cancellation.initialize(stream) ||
        !initialize_outputs(device_outputs, observed, stream) ||
        !upload(device_input, input.data(),
                input.size() * sizeof(std::uint16_t), stream) ||
        !upload(device_payload, payload.data(), payload.size(), stream)) {
      return false;
    }
    const int candidate_status =
        kernels::sm87_bulk_v2_fp8_whole_p40_oracle_detail::launch_raw(
            candidate);
    kernels::Sm87BulkV2Fp8WholeP40DeviceControl control{};
    if (candidate_status != static_cast<int>(cudaSuccess) ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        !download_outputs(observed, device_outputs, stream) ||
        !download(input_after.data(), device_input, stream) ||
        !download(payload_after.data(), device_payload, stream) ||
        !download(&control, device_control, stream) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << role_name(role) << " candidate launch failed k=" << k
                << " status=" << candidate_status << '\n';
      return false;
    }
    if (control.transaction_epoch != candidate.transaction_epoch ||
        control.role != static_cast<std::uint32_t>(role) ||
        control.expected_cells != plan.n_tiles ||
        control.started_cells != plan.n_tiles ||
        control.completed_cells != plan.n_tiles ||
        control.completed_ctas != 32U || control.cancellation_observed != 0U ||
        control.launch_completed != 1U ||
        control.first_incomplete_cohort != 0xffff'ffffU ||
        control.policy != kernels::kSm87BulkV2Fp8WholeP40RequiredPolicy) {
      std::cerr << role_name(role) << " invalid completion receipt k=" << k
                << '\n';
      return false;
    }
    if (input_after != input || payload_after != payload ||
        !compare_outputs(observed, expected, role_name(role),
                         "candidate-expected")) {
      return false;
    }
    const HostOutputs first = observed;

    // Deterministic replay of the same cooperative kernel and payload.
    if (!device_control.initialize(stream) ||
        !initialize_outputs(device_outputs, observed, stream) ||
        kernels::sm87_bulk_v2_fp8_whole_p40_oracle_detail::launch_raw(
            candidate) != static_cast<int>(cudaSuccess) ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        !download_outputs(observed, device_outputs, stream) ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        !compare_outputs(observed, first, role_name(role),
                         "deterministic-replay")) {
      return false;
    }

    // Same-ELF exact control uses the identical raw payload, scales and M64
    // publication while retaining its independent N256 execution body.
    if (!initialize_outputs(device_outputs, observed, stream)) {
      return false;
    }
    kernels::sm87_target_aot_projection_execution_detail::
        Sm87TargetAotFp8RawOracleArguments exact_control;
    exact_control.role = role;
    exact_control.input = candidate.input;
    exact_control.payload = candidate.payload;
    exact_control.compensated_scale_bf16_bits = scales;
    exact_control.token_count = kRows;
    exact_control.primary_output = candidate.primary_output;
    exact_control.secondary_output = candidate.secondary_output;
    exact_control.tertiary_output = candidate.tertiary_output;
    exact_control.cuda_stream = candidate.cuda_stream;
    const int control_status =
        kernels::sm87_target_aot_projection_execution_detail::
            launch_raw_fp8_v1_oracle(exact_control);
    if (control_status != static_cast<int>(cudaSuccess) ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        !download_outputs(observed, device_outputs, stream) ||
        !download(input_after.data(), device_input, stream) ||
        !download(payload_after.data(), device_payload, stream) ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        input_after != input || payload_after != payload ||
        !compare_outputs(observed, first, role_name(role),
                         "same-elf-exact-control")) {
      std::cerr << role_name(role) << " exact control failed k=" << k
                << " status=" << control_status << '\n';
      return false;
    }
    if (!device_input.guards_unchanged() ||
        !device_payload.guards_unchanged() ||
        !device_control.guards_unchanged() ||
        !device_cancellation.guards_unchanged() ||
        !output_guards_unchanged(observed, device_outputs)) {
      std::cerr << role_name(role) << " redzone corruption k=" << k << '\n';
      return false;
    }
  }

  // Cancellation is observed CTA-uniformly before any load/MMA/store.  The
  // output poison and incomplete receipt must survive.
  const std::uint32_t cancel = 1U;
  candidate.cancellation_signal =
      reinterpret_cast<const std::uint32_t*>(device_cancellation.data());
  candidate.n_tiles = 1U;
  if (!device_control.initialize(stream) ||
      !device_cancellation.initialize(stream) ||
      !upload(device_cancellation, &cancel, sizeof(cancel), stream) ||
      !initialize_outputs(device_outputs, observed, stream, kGuardValue) ||
      kernels::sm87_bulk_v2_fp8_whole_p40_oracle_detail::launch_raw(
          candidate) != static_cast<int>(cudaSuccess) ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }
  kernels::Sm87BulkV2Fp8WholeP40DeviceControl cancelled_control{};
  if (!download_outputs(observed, device_outputs, stream) ||
      !download(&cancelled_control, device_control, stream) ||
      !download(input_after.data(), device_input, stream) ||
      !download(payload_after.data(), device_payload, stream) ||
      cudaStreamSynchronize(stream) != cudaSuccess ||
      cancelled_control.transaction_epoch != candidate.transaction_epoch ||
      cancelled_control.role != static_cast<std::uint32_t>(role) ||
      cancelled_control.policy !=
          kernels::kSm87BulkV2Fp8WholeP40RequiredPolicy ||
      cancelled_control.expected_cells != 1U ||
      cancelled_control.cancellation_observed != 1U ||
      cancelled_control.launch_completed != 0U ||
      cancelled_control.started_cells != 0U ||
      cancelled_control.completed_cells != 0U ||
      cancelled_control.completed_ctas != 32U ||
      cancelled_control.first_incomplete_cohort != 0U ||
      input_after != input || payload_after != payload ||
      !std::all_of(observed.primary.begin(), observed.primary.end(),
                   [](const auto value) { return value == 0xa5a5U; }) ||
      !std::all_of(observed.secondary.begin(), observed.secondary.end(),
                   [](const auto value) { return value == 0xa5a5U; }) ||
      !std::all_of(observed.tertiary.begin(), observed.tertiary.end(),
                   [](const auto value) { return value == 0xa5a5U; })) {
    std::cerr << role_name(role) << " cancellation/receipt guard failed\n";
    return false;
  }
  return output_guards_unchanged(observed, device_outputs) &&
         device_input.guards_unchanged() &&
         device_payload.guards_unchanged() &&
         device_control.guards_unchanged() &&
         device_cancellation.guards_unchanged();
}

[[nodiscard]] kernels::Sm87BulkV2Fp8WholeP40CodeEvidence
oracle_static_record(const std::uint64_t identity) noexcept {
  kernels::Sm87BulkV2Fp8WholeP40CodeEvidence result;
  result.elf_identity = identity;
  result.sass_identity = identity ^ 0x5133'5832'5034'3046ULL;
  result.launch_bounds_256_2 = true;
  result.contains_cp_async_cg = true;
  result.contains_ldmatrix = true;
  result.contains_bf16_mma = true;
  result.same_kernel_exact_oracle = true;
  result.no_partial_c_symbol = true;
  result.valid = true;
  return result;
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

  const std::array<kernels::Sm87BulkV2Fp8WholeP40CodeEvidence, 3U> code{{
      oracle_static_record(1U), oracle_static_record(2U),
      oracle_static_record(3U)}};
  kernels::Sm87BulkV2Fp8WholeP40FamilyResources resources{};
  const int resource_status =
      kernels::query_sm87_bulk_dataflow_v2_fp8_whole_p40_resources_cuda(
          &code, &resources);
  if (resource_status != static_cast<int>(cudaSuccess) ||
      !kernels::
          sm87_bulk_v2_fp8_whole_p40_family_resource_observation_consistent(
              resources)) {
    std::cerr << "whole-P40 runtime resource observation failed: "
              << resource_status << '\n';
    for (const auto& role : resources.roles) {
      std::cerr << role_name(role.role) << " regs="
                << role.registers_per_thread << " local=" << role.local_bytes
                << " active_cta_per_sm=" << role.active_blocks_per_sm << '\n';
    }
    return 1;
  }
  for (const auto& role : resources.roles) {
    std::cout << role_name(role.role) << " regs="
              << role.registers_per_thread << " local=" << role.local_bytes
              << " dynamic_smem=" << role.dynamic_shared_bytes
              << " active_cta_per_sm=" << role.active_blocks_per_sm << '\n';
  }

  Stream stream;
  if (!stream.create()) {
    return 1;
  }
  constexpr std::array<Role, 3U> roles{{
      Role::kFp8GdnQkvZ, Role::kFp8FullQkv,
      Role::kFp8AttentionOutput}};
  for (const auto role : roles) {
    if (!run_role(role, stream.get())) {
      return 1;
    }
  }
  std::cout << "SM87 bulk-v2 FP8 whole-P40000 same-kernel exact CUDA oracle "
               "passed\n";
  return 0;
}
