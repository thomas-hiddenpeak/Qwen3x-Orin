#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_projection.h"

#include "sm87_target_aot_projection_fp8_oracle_internal.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace kernels = q3x::kernels;

namespace {

using Role = kernels::Sm87TargetAotProjectionRole;

constexpr int kSkip = 77;
constexpr std::size_t kGuardBytes = 256U;
constexpr std::uint8_t kGuardValue = 0xa5U;
constexpr std::uint16_t kCompensatedScaleOne =
    kernels::sm87_target_aot_fp8_compensated_tensor_scale_bf16_bits(
        0x3f80'0000U);

static_assert(kCompensatedScaleOne == 0x7b80U);

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
    const cudaError_t status = cudaMalloc(
        reinterpret_cast<void**>(&allocation_), bytes + 2U * kGuardBytes);
    if (status != cudaSuccess) {
      return false;
    }
    pointer_ = allocation_ + kGuardBytes;
    return true;
  }

  [[nodiscard]] bool initialize_on(
      const cudaStream_t stream) noexcept {
    // The oracle stream is intentionally cudaStreamNonBlocking.  A plain
    // cudaMemset here would use the legacy default stream and is not ordered
    // before the H2D copies below.  The original harness did exactly that;
    // its 0xa5 guard initialization could race the payload copy, and the
    // reported BF16 0xbe50 is precisely raw FP8 code 0xa5.  Keep every byte
    // of initialization on the same explicit stream.
    return allocation_ != nullptr && stream != nullptr &&
           cudaMemsetAsync(allocation_, kGuardValue,
                           bytes_ + 2U * kGuardBytes, stream) == cudaSuccess;
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
    for (const auto value : prefix) {
      if (value != kGuardValue) {
        return false;
      }
    }
    for (const auto value : suffix) {
      if (value != kGuardValue) {
        return false;
      }
    }
    return true;
  }

 private:
  std::uint8_t* allocation_ = nullptr;
  std::uint8_t* pointer_ = nullptr;
  std::size_t bytes_ = 0U;
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
  [[nodiscard]] bool create() noexcept {
    return cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) ==
           cudaSuccess;
  }
  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

[[nodiscard]] bool copy_to_device(GuardedDeviceBuffer& destination,
                                  const void* const source,
                                  const std::size_t bytes,
                                  const cudaStream_t stream) noexcept {
  return bytes == destination.bytes() &&
         cudaMemcpyAsync(destination.data(), source, bytes,
                         cudaMemcpyHostToDevice, stream) == cudaSuccess;
}

[[nodiscard]] bool copy_from_device(void* const destination,
                                    const GuardedDeviceBuffer& source,
                                    const cudaStream_t stream) noexcept {
  return cudaMemcpyAsync(destination, source.data(), source.bytes(),
                         cudaMemcpyDeviceToHost, stream) == cudaSuccess;
}

[[nodiscard]] const char* role_name(const Role role) noexcept {
  if (role == Role::kFp8GdnQkvZ) {
    return "GDN QKVZ";
  }
  if (role == Role::kFp8FullQkv) {
    return "Full QKV";
  }
  return "Attention O";
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

[[nodiscard]] std::uint16_t expected_one_hot_product(
    const std::uint8_t code) noexcept {
  if ((code & 0x7fU) == 0U) {
    // MMA accumulates into +0, so either FP8 zero sign has the same published
    // arithmetic result even though the raw decoder itself preserves sign.
    return 0U;
  }
  const float biased = float_from_bf16(
      kernels::sm87_bulk_v2_fp8_raw_code_to_biased_bf16_bits(code));
  const float compensation = float_from_bf16(kCompensatedScaleOne);
  return encode_bf16_rne(std::fma(1.0F, biased, 0.0F) * compensation);
}

void fill_expected_partition(std::vector<std::uint16_t>& expected,
                             const std::size_t width,
                             const std::size_t column_offset) {
  for (std::size_t row = 0U;
       row < kernels::kSm87BulkV2Fp8TailTokens; ++row) {
    for (std::size_t column = 0U; column < 256U; ++column) {
      expected[row * width + column_offset + column] =
          expected_one_hot_product(static_cast<std::uint8_t>(column));
    }
  }
}

[[nodiscard]] bool compare_exact(
    const std::vector<std::uint16_t>& observed,
    const std::vector<std::uint16_t>& expected, const std::size_t width,
    const char* const role, const char* const case_name,
    const char* const output_name) {
  if (observed.size() != expected.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < observed.size(); ++index) {
    if (observed[index] != expected[index]) {
      std::cerr << role << ' ' << case_name << ' ' << output_name
                << " mismatch at row "
                << index / width << " column " << index % width
                << " expected 0x" << std::hex << expected[index]
                << " observed 0x" << observed[index] << std::dec << '\n';
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool run_role_oracle(const Role role,
                                   const cudaStream_t stream) {
  const auto plan = kernels::sm87_bulk_v2_fp8_role_plan(role);
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  if (!plan.valid || !layout.valid()) {
    std::cerr << role_name(role) << ": invalid frozen plan\n";
    return false;
  }

  std::vector<std::uint16_t> input(
      kernels::kSm87BulkV2Fp8TailTokens * plan.input_features, 0U);
  std::vector<std::uint8_t> payload(layout.payload_bytes, 0U);

  const std::size_t primary_values =
      kernels::kSm87BulkV2Fp8TailTokens * plan.primary_output_features;
  const std::size_t secondary_values =
      kernels::kSm87BulkV2Fp8TailTokens * plan.secondary_output_features;
  const std::size_t tertiary_values =
      kernels::kSm87BulkV2Fp8TailTokens * plan.tertiary_output_features;
  std::vector<std::uint16_t> primary(primary_values, 0U);
  std::vector<std::uint16_t> secondary(secondary_values, 0U);
  std::vector<std::uint16_t> tertiary(tertiary_values, 0U);
  std::vector<std::uint16_t> input_after(input.size(), 0U);
  std::vector<std::uint8_t> payload_after(payload.size(), 0U);

  GuardedDeviceBuffer device_input;
  GuardedDeviceBuffer device_payload;
  GuardedDeviceBuffer device_primary;
  GuardedDeviceBuffer device_secondary;
  GuardedDeviceBuffer device_tertiary;
  if (!device_input.allocate(input.size() * sizeof(std::uint16_t)) ||
      !device_payload.allocate(payload.size()) ||
      !device_primary.allocate(primary.size() * sizeof(std::uint16_t)) ||
      (secondary_values != 0U &&
       !device_secondary.allocate(secondary.size() * sizeof(std::uint16_t))) ||
      (tertiary_values != 0U &&
       !device_tertiary.allocate(tertiary.size() * sizeof(std::uint16_t)))) {
    std::cerr << role_name(role) << ": device allocation failed\n";
    return false;
  }

  kernels::Sm87BulkV2Fp8OracleArguments arguments;
  arguments.role = role;
  arguments.input = reinterpret_cast<const std::uint16_t*>(
      device_input.data());
  arguments.payload = device_payload.data();
  for (std::size_t partition = 0U;
       partition < layout.partition_count; ++partition) {
    arguments.compensated_scale_bf16_bits[partition] =
        kCompensatedScaleOne;
  }
  arguments.token_count = kernels::kSm87BulkV2Fp8TailTokens;
  arguments.primary_output =
      reinterpret_cast<std::uint16_t*>(device_primary.data());
  if (secondary_values != 0U) {
    arguments.secondary_output =
        reinterpret_cast<std::uint16_t*>(device_secondary.data());
    arguments.tertiary_output =
        reinterpret_cast<std::uint16_t*>(device_tertiary.data());
  }
  arguments.cuda_stream = reinterpret_cast<void*>(stream);

  struct OracleCase final {
    const char* name;
    std::size_t input_k;
    bool encoded_payload;
  };
  const std::array<OracleCase, 3U> cases{{
      {"all-zero-payload", 0U, false},
      {"first-k-tile", 0U, true},
      {"last-k-tile", plan.input_features - 1U, true},
  }};

  for (const auto& oracle_case : cases) {
    std::fill(input.begin(), input.end(), 0U);
    std::fill(payload.begin(), payload.end(), 0U);
    for (std::size_t row = 0U;
         row < kernels::kSm87BulkV2Fp8TailTokens; ++row) {
      input[row * plan.input_features + oracle_case.input_k] = 0x3f80U;
    }
    if (oracle_case.encoded_payload) {
      for (std::size_t partition = 0U;
           partition < layout.partition_count; ++partition) {
        for (std::size_t column = 0U; column < 256U; ++column) {
          const auto address =
              kernels::sm87_target_aot_projection_packed_weight_address(
                  layout, partition, column, oracle_case.input_k);
          if (!address.valid || address.byte_offset >= payload.size()) {
            std::cerr << role_name(role) << ' ' << oracle_case.name
                      << ": invalid payload address\n";
            return false;
          }
          payload[address.byte_offset] =
              static_cast<std::uint8_t>(column);
        }
      }
    }

    if (!device_input.initialize_on(stream) ||
        !device_payload.initialize_on(stream) ||
        !device_primary.initialize_on(stream) ||
        (secondary_values != 0U &&
         !device_secondary.initialize_on(stream)) ||
        (tertiary_values != 0U &&
         !device_tertiary.initialize_on(stream)) ||
        !copy_to_device(device_input, input.data(),
                        input.size() * sizeof(std::uint16_t), stream) ||
        !copy_to_device(device_payload, payload.data(), payload.size(),
                        stream) ||
        cudaMemsetAsync(device_primary.data(), 0,
                        device_primary.bytes(), stream) != cudaSuccess ||
        (secondary_values != 0U &&
         cudaMemsetAsync(device_secondary.data(), 0,
                         device_secondary.bytes(), stream) != cudaSuccess) ||
        (tertiary_values != 0U &&
         cudaMemsetAsync(device_tertiary.data(), 0,
                         device_tertiary.bytes(), stream) != cudaSuccess)) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": initialization failed\n";
      return false;
    }

    const int launch_status =
        kernels::launch_sm87_bulk_dataflow_v2_fp8_oracle_segment_cuda(
            arguments);
    if (launch_status != static_cast<int>(cudaSuccess) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": launch failed: " << launch_status << '\n';
      return false;
    }

    if (!copy_from_device(primary.data(), device_primary, stream) ||
        !copy_from_device(input_after.data(), device_input, stream) ||
        !copy_from_device(payload_after.data(), device_payload, stream) ||
        (secondary_values != 0U &&
         !copy_from_device(secondary.data(), device_secondary, stream)) ||
        (tertiary_values != 0U &&
         !copy_from_device(tertiary.data(), device_tertiary, stream)) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": result copy failed\n";
      return false;
    }
    if (input_after != input) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": input was modified\n";
      return false;
    }
    if (payload_after != payload) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": payload was modified\n";
      return false;
    }
    if (!device_input.guards_unchanged() ||
        !device_payload.guards_unchanged() ||
        !device_primary.guards_unchanged() ||
        (secondary_values != 0U && !device_secondary.guards_unchanged()) ||
        (tertiary_values != 0U && !device_tertiary.guards_unchanged())) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": guard corruption\n";
      return false;
    }

    const auto bulk_primary = primary;
    const auto bulk_secondary = secondary;
    const auto bulk_tertiary = tertiary;

    // Deterministic candidate replay.  Reinitialize every byte and enqueue
    // the same v2 kernel again before involving the v1 control.  This catches
    // stage-ring/stale-shared dependence that a single successful run could
    // hide.  The all-zero case is replayed too: it is the strongest detector
    // for stray guard or prior-case bytes.
    if (!device_input.initialize_on(stream) ||
        !device_payload.initialize_on(stream) ||
        !device_primary.initialize_on(stream) ||
        (secondary_values != 0U &&
         !device_secondary.initialize_on(stream)) ||
        (tertiary_values != 0U &&
         !device_tertiary.initialize_on(stream)) ||
        !copy_to_device(device_input, input.data(),
                        input.size() * sizeof(std::uint16_t), stream) ||
        !copy_to_device(device_payload, payload.data(), payload.size(),
                        stream) ||
        cudaMemsetAsync(device_primary.data(), 0,
                        device_primary.bytes(), stream) != cudaSuccess ||
        (secondary_values != 0U &&
         cudaMemsetAsync(device_secondary.data(), 0,
                         device_secondary.bytes(), stream) != cudaSuccess) ||
        (tertiary_values != 0U &&
         cudaMemsetAsync(device_tertiary.data(), 0,
                         device_tertiary.bytes(), stream) != cudaSuccess)) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v2 replay initialization failed\n";
      return false;
    }
    const int replay_status =
        kernels::launch_sm87_bulk_dataflow_v2_fp8_oracle_segment_cuda(
            arguments);
    if (replay_status != static_cast<int>(cudaSuccess) ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        !copy_from_device(primary.data(), device_primary, stream) ||
        !copy_from_device(input_after.data(), device_input, stream) ||
        !copy_from_device(payload_after.data(), device_payload, stream) ||
        (secondary_values != 0U &&
         !copy_from_device(secondary.data(), device_secondary, stream)) ||
        (tertiary_values != 0U &&
         !copy_from_device(tertiary.data(), device_tertiary, stream)) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v2 replay failed: " << replay_status << '\n';
      return false;
    }
    if (input_after != input || payload_after != payload ||
        primary != bulk_primary || secondary != bulk_secondary ||
        tertiary != bulk_tertiary) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v2 deterministic replay mismatch\n";
      return false;
    }
    if (!device_input.guards_unchanged() ||
        !device_payload.guards_unchanged() ||
        !device_primary.guards_unchanged() ||
        (secondary_values != 0U && !device_secondary.guards_unchanged()) ||
        (tertiary_values != 0U && !device_tertiary.guards_unchanged())) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v2 replay guard corruption\n";
      return false;
    }

    // Same-ELF v1 M128 control: identical input, payload bytes, scales,
    // stream, and M64 publication.  If v2 and v1 diverge, the failure belongs
    // to the new executor; if both see the same bad bytes, the shared harness
    // or payload path is at fault.
    if (!device_primary.initialize_on(stream) ||
        (secondary_values != 0U &&
         !device_secondary.initialize_on(stream)) ||
        (tertiary_values != 0U &&
         !device_tertiary.initialize_on(stream)) ||
        cudaMemsetAsync(device_primary.data(), 0,
                        device_primary.bytes(), stream) != cudaSuccess ||
        (secondary_values != 0U &&
         cudaMemsetAsync(device_secondary.data(), 0,
                         device_secondary.bytes(), stream) != cudaSuccess) ||
        (tertiary_values != 0U &&
         cudaMemsetAsync(device_tertiary.data(), 0,
                         device_tertiary.bytes(), stream) != cudaSuccess)) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v1 control initialization failed\n";
      return false;
    }
    kernels::sm87_target_aot_projection_execution_detail::
        Sm87TargetAotFp8RawOracleArguments control;
    control.role = role;
    control.input = arguments.input;
    control.payload = arguments.payload;
    control.compensated_scale_bf16_bits =
        arguments.compensated_scale_bf16_bits;
    control.token_count = arguments.token_count;
    control.primary_output = arguments.primary_output;
    control.secondary_output = arguments.secondary_output;
    control.tertiary_output = arguments.tertiary_output;
    control.cuda_stream = arguments.cuda_stream;
    const int control_status =
        kernels::sm87_target_aot_projection_execution_detail::
            launch_raw_fp8_v1_oracle(control);
    if (control_status != static_cast<int>(cudaSuccess) ||
        cudaStreamSynchronize(stream) != cudaSuccess ||
        !copy_from_device(primary.data(), device_primary, stream) ||
        !copy_from_device(input_after.data(), device_input, stream) ||
        !copy_from_device(payload_after.data(), device_payload, stream) ||
        (secondary_values != 0U &&
         !copy_from_device(secondary.data(), device_secondary, stream)) ||
        (tertiary_values != 0U &&
         !copy_from_device(tertiary.data(), device_tertiary, stream)) ||
        cudaStreamSynchronize(stream) != cudaSuccess) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v1 control failed: " << control_status << '\n';
      return false;
    }
    if (input_after != input || payload_after != payload) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v1 control modified an input payload\n";
      return false;
    }
    if (!device_input.guards_unchanged() ||
        !device_payload.guards_unchanged() ||
        !device_primary.guards_unchanged() ||
        (secondary_values != 0U && !device_secondary.guards_unchanged()) ||
        (tertiary_values != 0U && !device_tertiary.guards_unchanged())) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v1 control guard corruption\n";
      return false;
    }
    if (primary != bulk_primary || secondary != bulk_secondary ||
        tertiary != bulk_tertiary) {
      std::cerr << role_name(role) << ' ' << oracle_case.name
                << ": v2/v1 same-ELF differential mismatch\n";
      return false;
    }

    std::vector<std::uint16_t> expected_primary(primary.size(), 0U);
    std::vector<std::uint16_t> expected_secondary(secondary.size(), 0U);
    std::vector<std::uint16_t> expected_tertiary(tertiary.size(), 0U);
    if (oracle_case.encoded_payload) {
      fill_expected_partition(expected_primary,
                              plan.primary_output_features, 0U);
      if (role == Role::kFp8GdnQkvZ) {
        fill_expected_partition(expected_primary,
                                plan.primary_output_features, 10'240U);
      } else if (role == Role::kFp8FullQkv) {
        fill_expected_partition(expected_secondary,
                                plan.secondary_output_features, 0U);
        fill_expected_partition(expected_tertiary,
                                plan.tertiary_output_features, 0U);
      }
    }
    if (!compare_exact(bulk_primary, expected_primary,
                       plan.primary_output_features, role_name(role),
                       oracle_case.name, "v2-primary") ||
        (secondary_values != 0U &&
         !compare_exact(bulk_secondary, expected_secondary,
                        plan.secondary_output_features, role_name(role),
                        oracle_case.name, "v2-secondary")) ||
        (tertiary_values != 0U &&
         !compare_exact(bulk_tertiary, expected_tertiary,
                        plan.tertiary_output_features, role_name(role),
                        oracle_case.name, "v2-tertiary")) ||
        !compare_exact(primary, expected_primary,
                       plan.primary_output_features, role_name(role),
                       oracle_case.name, "v1-primary") ||
        (secondary_values != 0U &&
         !compare_exact(secondary, expected_secondary,
                        plan.secondary_output_features, role_name(role),
                        oracle_case.name, "v1-secondary")) ||
        (tertiary_values != 0U &&
         !compare_exact(tertiary, expected_tertiary,
                        plan.tertiary_output_features, role_name(role),
                        oracle_case.name, "v1-tertiary"))) {
      return false;
    }
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
      properties.multiProcessorCount != 16) {
    return kSkip;
  }
  if (cudaSetDevice(0) != cudaSuccess) {
    return 1;
  }

  kernels::Sm87BulkV2Fp8FamilyResources resources{};
  const int resource_status =
      kernels::query_sm87_bulk_dataflow_v2_fp8_family_resources_cuda(
          &resources);
  if (resource_status != static_cast<int>(cudaSuccess) ||
      !kernels::sm87_bulk_v2_fp8_family_resources_valid(resources)) {
    std::cerr << "family resource gate failed: " << resource_status << '\n';
    return 1;
  }
  for (const auto& role : resources.roles) {
    std::cout << role_name(role.role) << ": regs="
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
    if (!run_role_oracle(role, stream.get())) {
      return 1;
    }
  }
  std::cout << "SM87 bulk-dataflow v2 FP8 exact CUDA oracle passed\n";
  return 0;
}
