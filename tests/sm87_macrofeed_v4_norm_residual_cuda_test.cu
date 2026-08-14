#include "q3x/kernels/sm87_macrofeed_v4_norm_residual.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;

[[nodiscard]] bool expect(const bool condition,
                          const std::string_view message) {
  if (!condition) {
    std::cerr << "FAIL: " << message << '\n';
  }
  return condition;
}

[[nodiscard]] bool expect_cuda(const cudaError_t status,
                               const std::string_view message) {
  if (status != cudaSuccess) {
    std::cerr << "FAIL: " << message << ": "
              << cudaGetErrorString(status) << '\n';
    return false;
  }
  return true;
}

[[nodiscard]] std::uint16_t encode_bf16(const float value) {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

template <class T>
class DeviceBuffer final {
 public:
  DeviceBuffer() = default;
  explicit DeviceBuffer(const std::size_t count) { (void)allocate(count); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  DeviceBuffer(DeviceBuffer&& other) noexcept
      : pointer_(std::exchange(other.pointer_, nullptr)),
        count_(std::exchange(other.count_, 0U)) {}
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
    if (this != &other) {
      release();
      pointer_ = std::exchange(other.pointer_, nullptr);
      count_ = std::exchange(other.count_, 0U);
    }
    return *this;
  }
  ~DeviceBuffer() { release(); }

  [[nodiscard]] bool allocate(const std::size_t count) {
    release();
    if (count == 0U ||
        cudaMalloc(reinterpret_cast<void**>(&pointer_),
                   count * sizeof(T)) != cudaSuccess) {
      pointer_ = nullptr;
      count_ = 0U;
      return false;
    }
    count_ = count;
    return true;
  }

  void release() noexcept {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
      pointer_ = nullptr;
      count_ = 0U;
    }
  }

  [[nodiscard]] bool upload(const std::vector<T>& values,
                            const cudaStream_t stream) const {
    return pointer_ != nullptr && values.size() == count_ &&
           cudaMemcpyAsync(pointer_, values.data(), count_ * sizeof(T),
                           cudaMemcpyHostToDevice, stream) == cudaSuccess;
  }

  [[nodiscard]] bool download(std::vector<T>* const values,
                              const cudaStream_t stream) const {
    if (pointer_ == nullptr || values == nullptr) {
      return false;
    }
    values->resize(count_);
    return cudaMemcpyAsync(values->data(), pointer_, count_ * sizeof(T),
                           cudaMemcpyDeviceToHost, stream) == cudaSuccess;
  }

  [[nodiscard]] T* get(const std::size_t offset = 0U) const noexcept {
    return pointer_ == nullptr || offset > count_ ? nullptr : pointer_ + offset;
  }
  [[nodiscard]] std::size_t count() const noexcept { return count_; }

 private:
  T* pointer_ = nullptr;
  std::size_t count_ = 0U;
};

class Stream final {
 public:
  Stream() {
    if (cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking) !=
        cudaSuccess) {
      stream_ = nullptr;
    }
  }
  Stream(const Stream&) = delete;
  Stream& operator=(const Stream&) = delete;
  ~Stream() {
    if (stream_ != nullptr) {
      (void)cudaStreamDestroy(stream_);
    }
  }
  [[nodiscard]] cudaStream_t get() const noexcept { return stream_; }

 private:
  cudaStream_t stream_ = nullptr;
};

class ManagedBuffer final {
 public:
  ManagedBuffer() = default;
  ManagedBuffer(const ManagedBuffer&) = delete;
  ManagedBuffer& operator=(const ManagedBuffer&) = delete;
  ~ManagedBuffer() {
    if (pointer_ != nullptr) {
      (void)cudaFree(pointer_);
    }
  }
  [[nodiscard]] bool allocate(const std::size_t bytes) noexcept {
    return bytes != 0U && pointer_ == nullptr &&
           cudaMallocManaged(&pointer_, bytes) == cudaSuccess;
  }
  [[nodiscard]] void* get() const noexcept { return pointer_; }

 private:
  void* pointer_ = nullptr;
};

[[nodiscard]] bool host_contract_test() {
  bool ok = true;
  const auto plan = kernels::sm87_macrofeed_v4_norm_residual_plan(
      kernels::kSm87MacroFeedV4NormResidualTokens,
      kernels::kSm87MacroFeedV4NormResidualHidden);
  ok &= expect(plan.valid() && plan.established_exact_norm_body_reused &&
                   plan.established_exact_residual_norm_body_reused &&
                   plan.input_norm_threads_per_cta == 256U &&
                   plan.fused_residual_norm_threads_per_cta == 512U &&
                   plan.residual_published_in_place_to_right &&
                   plan.normalized_published_in_place_to_left &&
                   !plan.third_hidden_plane_present &&
                   !plan.copy_kernel_present && !plan.selector_present &&
                   !plan.fallback_permitted && !plan.jit_permitted &&
                   !plan.runtime_repack_permitted &&
                   !plan.autotune_permitted && plan.default_off &&
                   !plan.numerical_contract_qualified &&
                   !plan.production_dispatch_eligible &&
                   plan.startup_package_unbound &&
                   !plan.execution_capability &&
                   !plan.caller_snapshot_grants_production_authority,
               "V4 norm/residual plan must expose the exact two-plane seam");
  ok &= expect(
      !kernels::sm87_macrofeed_v4_norm_residual_plan(7'999U, 5'120U)
           .valid() &&
          !kernels::sm87_macrofeed_v4_norm_residual_plan(8'000U, 5'119U)
               .valid(),
      "non-C8000/non-H5120 plans must fail closed");

  const auto* const input = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'1000'0000'0000ULL));
  const auto* const weight = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'2000'0000'0000ULL));
  auto* const output = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'3000'0000'0000ULL));
  kernels::Sm87MacroFeedV4InputNormArguments norm{
      input,
      weight,
      output,
      8'000U,
      5'120U,
      kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
      reinterpret_cast<void*>(0x1000U)};
  ok &= expect(kernels::sm87_macrofeed_v4_input_norm_arguments_valid(norm),
               "coherent host-only input norm ranges must pass T0");
  auto changed_norm = norm;
  changed_norm.token_count -= 1U;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_input_norm_arguments_valid(changed_norm),
      "wrong norm token count must fail closed");
  changed_norm = norm;
  changed_norm.output_hidden = const_cast<std::uint16_t*>(input);
  ok &= expect(
      !kernels::sm87_macrofeed_v4_input_norm_arguments_valid(changed_norm),
      "input/output alias must fail the norm seam");
  changed_norm = norm;
  changed_norm.cuda_stream = nullptr;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_input_norm_arguments_valid(changed_norm),
      "null norm stream must fail closed");

  kernels::Sm87MacroFeedV4ResidualPostNormArguments fused{
      const_cast<std::uint16_t*>(input),
      output,
      weight,
      8'000U,
      5'120U,
      kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
      reinterpret_cast<void*>(0x1000U)};
  ok &= expect(
      kernels::sm87_macrofeed_v4_residual_post_norm_arguments_valid(fused),
      "coherent two-plane fused alias contract must pass T0");
  auto changed_fused = fused;
  changed_fused.right_branch_then_residual =
      changed_fused.left_residual_then_normalized;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_residual_post_norm_arguments_valid(
          changed_fused),
      "one physical plane cannot impersonate the exact two-plane contract");
  changed_fused = fused;
  changed_fused.epsilon_fp32_bits = 0U;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_residual_post_norm_arguments_valid(
          changed_fused),
      "changed epsilon bits must fail closed");

  kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot synthetic{};
  synthetic.identity = kernels::kSm87MacroFeedV4NormResidualIdentity;
  synthetic.device_ordinal = 0;
  synthetic.compute_major = 8;
  synthetic.compute_minor = 7;
  synthetic.sm_count = 16;
  synthetic.binary_version = 87;
  synthetic.input_norm = {
      32,
      kernels::kSm87MacroFeedV4NormResidualInputNormStaticSharedBytes,
      0U,
      1'024,
      2,
      256,
      8'000};
  synthetic.fused_residual_norm = {
      32,
      kernels::kSm87MacroFeedV4NormResidualFusedStaticSharedBytes,
      0U,
      1'024,
      2,
      512,
      8'000};
  synthetic.kernels_compiled = true;
  synthetic.exact_geometry = true;
  synthetic.static_resource_gate_passed = true;
  synthetic.startup_package_unbound = true;
  ok &= expect(
      kernels::sm87_macrofeed_v4_norm_residual_resource_gate(synthetic),
      "caller-fillable resource evidence may describe but not authorize T1");
  synthetic.fused_residual_norm.local_bytes = 8U;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_norm_residual_resource_gate(synthetic),
      "local-memory regression must fail the resource gate");

  kernels::Sm87MacroFeedV4NormResidualAdmissionLaunchReceipt receipt{};
  ok &= expect(!receipt.valid_enqueue_receipt(),
               "default caller receipt must grant no authority");
  return ok;
}

[[nodiscard]] std::vector<std::uint16_t> make_hidden_pattern(
    const std::size_t tokens, const std::uint32_t salt) {
  const std::size_t count =
      tokens * kernels::kSm87MacroFeedV4NormResidualHidden;
  std::vector<std::uint16_t> values(count);
  for (std::size_t index = 0U; index < count; ++index) {
    const std::int32_t centered = static_cast<std::int32_t>(
        (index * 37U + salt * 101U + index / 5U) % 257U) -
                                  128;
    values[index] = encode_bf16(static_cast<float>(centered) / 64.0F);
  }
  if (tokens >= 65U) {
    const std::size_t offset =
        64U * kernels::kSm87MacroFeedV4NormResidualHidden;
    constexpr std::array<std::uint16_t, 10U> special{
        0x0000U, 0x8000U, 0x0001U, 0x8001U, 0x007fU,
        0x807fU, 0x7f80U, 0xff80U, 0x7fc1U, 0xffc1U};
    std::copy(special.begin(), special.end(), values.begin() + offset);
  }
  return values;
}

[[nodiscard]] std::vector<std::uint16_t> make_weight_pattern() {
  std::vector<std::uint16_t> values(
      kernels::kSm87MacroFeedV4NormResidualHidden);
  for (std::size_t index = 0U; index < values.size(); ++index) {
    const std::int32_t centered =
        static_cast<std::int32_t>((index * 13U + 17U) % 65U) - 32;
    values[index] = encode_bf16(static_cast<float>(centered) / 256.0F);
  }
  return values;
}

[[nodiscard]] bool bounded_bit_oracles(const cudaStream_t stream,
                                       const std::size_t token_count) {
  if (token_count != 1U && token_count != 65U) {
    return expect(false, "bounded oracle admits only C1 or C65");
  }
  const std::size_t kTokens = token_count;
  constexpr std::size_t kHidden =
      kernels::kSm87MacroFeedV4NormResidualHidden;
  const std::size_t kElements = kTokens * kHidden;
  constexpr std::size_t kGuard = 8U;
  constexpr std::uint16_t kGuardBits = 0x5a5aU;
  bool ok = true;

  const auto host_left = make_hidden_pattern(kTokens, 3U);
  const auto host_right = make_hidden_pattern(kTokens, 11U);
  const auto host_weight = make_weight_pattern();
  std::vector<std::uint16_t> guarded_left(kElements + 2U * kGuard,
                                          kGuardBits);
  std::vector<std::uint16_t> guarded_right(kElements + 2U * kGuard,
                                           kGuardBits);
  std::copy(host_left.begin(), host_left.end(),
            guarded_left.begin() + kGuard);
  std::copy(host_right.begin(), host_right.end(),
            guarded_right.begin() + kGuard);

  DeviceBuffer<std::uint16_t> input(kElements + 2U * kGuard);
  DeviceBuffer<std::uint16_t> norm_reference(kElements + 2U * kGuard);
  DeviceBuffer<std::uint16_t> norm_candidate(kElements + 2U * kGuard);
  DeviceBuffer<std::uint16_t> weight(host_weight.size());
  if (input.get() == nullptr || norm_reference.get() == nullptr ||
      norm_candidate.get() == nullptr || weight.get() == nullptr) {
    return expect(false, "bounded norm allocations");
  }
  std::vector<std::uint16_t> guarded_output(kElements + 2U * kGuard,
                                            kGuardBits);
  ok &= expect(input.upload(guarded_left, stream) &&
                   norm_reference.upload(guarded_output, stream) &&
                   norm_candidate.upload(guarded_output, stream) &&
                   weight.upload(host_weight, stream),
               "bounded norm uploads");
  constexpr float kEpsilon = 1.0e-6F;
  const int reference_status =
      runtime::launch_headwise_centered_rms_norm_reference_cuda(
          input.get(kGuard), weight.get(), kTokens, kHidden, kEpsilon,
          norm_reference.get(kGuard), stream);
  kernels::Sm87MacroFeedV4InputNormArguments norm_arguments{
      input.get(kGuard),
      weight.get(),
      norm_candidate.get(kGuard),
      kTokens,
      kHidden,
      kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
      stream};
  const int candidate_status =
      kernels::launch_sm87_macrofeed_v4_input_norm_oracle_cuda(
          norm_arguments);
  ok &= expect(reference_status == static_cast<int>(cudaSuccess) &&
                   candidate_status == static_cast<int>(cudaSuccess),
               "bounded norm launches");

  std::vector<std::uint16_t> norm_reference_host;
  std::vector<std::uint16_t> norm_candidate_host;
  std::vector<std::uint16_t> input_after;
  std::vector<std::uint16_t> weight_after;
  ok &= expect(norm_reference.download(&norm_reference_host, stream) &&
                   norm_candidate.download(&norm_candidate_host, stream) &&
                   input.download(&input_after, stream) &&
                   weight.download(&weight_after, stream),
               "bounded norm downloads");
  ok &= expect_cuda(cudaStreamSynchronize(stream),
                    "bounded norm synchronization");
  ok &= expect(norm_reference_host == norm_candidate_host,
               "C1/C65 input norm must be bitwise equal to established body");
  ok &= expect(input_after == guarded_left && weight_after == host_weight,
               "input norm must preserve input and centered weight");
  ok &= expect(std::all_of(norm_candidate_host.begin(),
                           norm_candidate_host.begin() + kGuard,
                           [](const std::uint16_t value) {
                             return value == kGuardBits;
                           }) &&
                   std::all_of(norm_candidate_host.end() - kGuard,
                               norm_candidate_host.end(),
                               [](const std::uint16_t value) {
                                 return value == kGuardBits;
                               }),
               "input norm guards must remain intact");

  DeviceBuffer<std::uint16_t> baseline_left(kElements + 2U * kGuard);
  DeviceBuffer<std::uint16_t> baseline_right(kElements + 2U * kGuard);
  DeviceBuffer<std::uint16_t> baseline_residual(kElements + 2U * kGuard);
  DeviceBuffer<std::uint16_t> baseline_normalized(kElements + 2U * kGuard);
  DeviceBuffer<std::uint16_t> candidate_left(kElements + 2U * kGuard);
  DeviceBuffer<std::uint16_t> candidate_right(kElements + 2U * kGuard);
  if (baseline_left.get() == nullptr || baseline_right.get() == nullptr ||
      baseline_residual.get() == nullptr ||
      baseline_normalized.get() == nullptr || candidate_left.get() == nullptr ||
      candidate_right.get() == nullptr) {
    return expect(false, "bounded fused allocations");
  }
  ok &= expect(baseline_left.upload(guarded_left, stream) &&
                   baseline_right.upload(guarded_right, stream) &&
                   baseline_residual.upload(guarded_output, stream) &&
                   baseline_normalized.upload(guarded_output, stream) &&
                   candidate_left.upload(guarded_left, stream) &&
                   candidate_right.upload(guarded_right, stream),
               "bounded fused uploads");
  int status = runtime::launch_residual_add_reference_cuda(
      baseline_left.get(kGuard), baseline_right.get(kGuard), kElements,
      baseline_residual.get(kGuard), stream);
  if (status == static_cast<int>(cudaSuccess)) {
    status = runtime::launch_headwise_centered_rms_norm_reference_cuda(
        baseline_residual.get(kGuard), weight.get(), kTokens, kHidden,
        kEpsilon, baseline_normalized.get(kGuard), stream);
  }
  kernels::Sm87MacroFeedV4ResidualPostNormArguments fused_arguments{
      candidate_left.get(kGuard),
      candidate_right.get(kGuard),
      weight.get(),
      kTokens,
      kHidden,
      kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
      stream};
  const int fused_status =
      kernels::launch_sm87_macrofeed_v4_residual_post_norm_oracle_cuda(
          fused_arguments);
  ok &= expect(status == static_cast<int>(cudaSuccess) &&
                   fused_status == static_cast<int>(cudaSuccess),
               "bounded fused/two-step launches");

  std::vector<std::uint16_t> residual_host;
  std::vector<std::uint16_t> normalized_host;
  std::vector<std::uint16_t> candidate_left_host;
  std::vector<std::uint16_t> candidate_right_host;
  std::vector<std::uint16_t> baseline_left_after;
  std::vector<std::uint16_t> baseline_right_after;
  std::vector<std::uint16_t> fused_weight_after;
  ok &= expect(baseline_residual.download(&residual_host, stream) &&
                   baseline_normalized.download(&normalized_host, stream) &&
                   candidate_left.download(&candidate_left_host, stream) &&
                   candidate_right.download(&candidate_right_host, stream) &&
                   baseline_left.download(&baseline_left_after, stream) &&
                   baseline_right.download(&baseline_right_after, stream) &&
                   weight.download(&fused_weight_after, stream),
               "bounded fused downloads");
  ok &= expect_cuda(cudaStreamSynchronize(stream),
                    "bounded fused synchronization");
  ok &= expect(candidate_right_host == residual_host,
               "right plane must contain exact BF16 residual bits");
  ok &= expect(candidate_left_host == normalized_host,
               "left plane must contain exact post-residual norm bits");
  ok &= expect(baseline_left_after == guarded_left &&
                   baseline_right_after == guarded_right,
               "independent two-step source inputs must remain immutable");
  ok &= expect(fused_weight_after == host_weight,
               "fused route must preserve centered weight");
  return ok;
}

[[nodiscard]] bool full_c8000_smoke_and_negatives(
    const cudaStream_t stream,
    const kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot&
        resources) {
  constexpr std::size_t kTokens =
      kernels::kSm87MacroFeedV4NormResidualTokens;
  constexpr std::size_t kHidden =
      kernels::kSm87MacroFeedV4NormResidualHidden;
  constexpr std::size_t kElements = kTokens * kHidden;
  bool ok = true;
  DeviceBuffer<std::uint16_t> left(kElements);
  DeviceBuffer<std::uint16_t> right(kElements);
  DeviceBuffer<std::uint16_t> weight(kHidden);
  if (left.get() == nullptr || right.get() == nullptr ||
      weight.get() == nullptr) {
    return expect(false, "complete C8000 allocations");
  }
  ok &= expect_cuda(cudaMemsetAsync(left.get(), 0, kElements * 2U, stream),
                    "clear C8000 left") &&
        expect_cuda(cudaMemsetAsync(right.get(), 0, kElements * 2U, stream),
                    "clear C8000 right") &&
        expect_cuda(cudaMemsetAsync(weight.get(), 0, kHidden * 2U, stream),
                    "clear centered weight");

  kernels::Sm87MacroFeedV4InputNormArguments norm{
      left.get(),
      weight.get(),
      right.get(),
      kTokens,
      kHidden,
      kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
      stream};
  kernels::Sm87MacroFeedV4NormResidualAdmissionLaunchReceipt norm_receipt{};
  int status = kernels::launch_sm87_macrofeed_v4_input_norm_admission_cuda(
      norm, resources, &norm_receipt);
  ok &= expect(status == static_cast<int>(cudaSuccess) &&
                   norm_receipt.valid_enqueue_receipt() &&
                   norm_receipt.operation ==
                       kernels::Sm87MacroFeedV4NormResidualOperation::
                           kInputCenteredRmsNorm,
               "complete C8000 norm enqueue receipt");
  auto forged_operation_receipt = norm_receipt;
  forged_operation_receipt.operation =
      static_cast<kernels::Sm87MacroFeedV4NormResidualOperation>(0xffU);
  ok &= expect(!forged_operation_receipt.valid_enqueue_receipt(),
               "unknown operation cannot validate an enqueue receipt");
  ok &= expect_cuda(cudaStreamSynchronize(stream),
                    "complete C8000 norm synchronization");

  ok &= expect_cuda(cudaMemsetAsync(left.get(), 0, kElements * 2U, stream),
                    "reclear C8000 left") &&
        expect_cuda(cudaMemsetAsync(right.get(), 0, kElements * 2U, stream),
                    "reclear C8000 right");
  kernels::Sm87MacroFeedV4ResidualPostNormArguments fused{
      left.get(),
      right.get(),
      weight.get(),
      kTokens,
      kHidden,
      kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
      stream};
  kernels::Sm87MacroFeedV4NormResidualAdmissionLaunchReceipt fused_receipt{};
  status =
      kernels::launch_sm87_macrofeed_v4_residual_post_norm_admission_cuda(
          fused, resources, &fused_receipt);
  ok &= expect(status == static_cast<int>(cudaSuccess) &&
                   fused_receipt.valid_enqueue_receipt() &&
                   fused_receipt.operation ==
                       kernels::Sm87MacroFeedV4NormResidualOperation::
                           kBranchResidualPostCenteredRmsNorm,
               "complete C8000 fused enqueue receipt");
  ok &= expect_cuda(cudaStreamSynchronize(stream),
                    "complete C8000 fused synchronization");

  const std::array<std::size_t, 3U> sample_indices{
      0U, kElements / 2U, kElements - 1U};
  for (const std::size_t index : sample_indices) {
    std::uint16_t left_value = 1U;
    std::uint16_t right_value = 1U;
    ok &= expect_cuda(cudaMemcpy(&left_value, left.get(index),
                                 sizeof(left_value), cudaMemcpyDeviceToHost),
                      "sample C8000 left") &&
          expect_cuda(cudaMemcpy(&right_value, right.get(index),
                                 sizeof(right_value), cudaMemcpyDeviceToHost),
                      "sample C8000 right");
    ok &= expect(left_value == 0U && right_value == 0U,
                 "zero C8000 smoke must remain exactly zero");
  }

  auto tampered = resources;
  ++tampered.input_norm.registers_per_thread;
  kernels::Sm87MacroFeedV4NormResidualAdmissionLaunchReceipt rejected{};
  ok &= expect(
      kernels::launch_sm87_macrofeed_v4_input_norm_admission_cuda(
          norm, tampered, &rejected) ==
              static_cast<int>(cudaErrorLaunchOutOfResources) &&
          !rejected.valid_enqueue_receipt(),
      "caller resource forgery must fail before enqueue");

  DeviceBuffer<std::uint16_t> undersized_weight(128U);
  kernels::Sm87MacroFeedV4ResidualPostNormArguments undersized{
      left.get(),
      right.get(),
      undersized_weight.get(),
      kTokens,
      kHidden,
      kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
      stream};
  rejected = {};
  ok &= expect(
      kernels::launch_sm87_macrofeed_v4_residual_post_norm_admission_cuda(
          undersized, resources, &rejected) ==
              static_cast<int>(cudaErrorInvalidDevicePointer) &&
          !rejected.valid_enqueue_receipt(),
      "undersized live allocations must fail before enqueue");

  ManagedBuffer managed;
  if (managed.allocate(kernels::kSm87MacroFeedV4NormResidualHiddenBytes)) {
    auto managed_norm = norm;
    managed_norm.output_hidden =
        static_cast<std::uint16_t*>(managed.get());
    rejected = {};
    ok &= expect(
        kernels::launch_sm87_macrofeed_v4_input_norm_admission_cuda(
            managed_norm, resources, &rejected) ==
                static_cast<int>(cudaErrorInvalidDevicePointer),
        "managed output cannot impersonate a device-owned hidden plane");
  }
  return ok;
}

}  // namespace

int main() {
  bool ok = host_contract_test();
  int device = -1;
  cudaDeviceProp properties{};
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 ||
      cudaGetDeviceProperties(&properties, device) != cudaSuccess ||
      properties.major != 8 || properties.minor != 7 ||
      properties.multiProcessorCount != 16) {
    std::cout << "SKIP: exact SM87/16-SM device unavailable\n";
    return 77;
  }
  Stream stream;
  if (stream.get() == nullptr) {
    std::cerr << "FAIL: stream creation\n";
    return 1;
  }
  kernels::Sm87MacroFeedV4NormResidualAdmissionResourceSnapshot resources{};
  const int resource_status =
      kernels::query_sm87_macrofeed_v4_norm_residual_admission_resources_cuda(
          &resources);
  ok &= expect(resource_status == static_cast<int>(cudaSuccess) &&
                   kernels::sm87_macrofeed_v4_norm_residual_resource_gate(
                       resources),
               "live norm/residual resource snapshot must pass");
  if (resource_status == static_cast<int>(cudaSuccess)) {
    std::cout << "input_norm_regs="
              << resources.input_norm.registers_per_thread
              << " input_norm_shared="
              << resources.input_norm.static_shared_bytes
              << " input_norm_local=" << resources.input_norm.local_bytes
              << " input_norm_cta_per_sm="
              << resources.input_norm.active_blocks_per_sm
              << " fused_regs="
              << resources.fused_residual_norm.registers_per_thread
              << " fused_shared="
              << resources.fused_residual_norm.static_shared_bytes
              << " fused_local="
              << resources.fused_residual_norm.local_bytes
              << " fused_cta_per_sm="
              << resources.fused_residual_norm.active_blocks_per_sm << '\n';
  }
  ok &= bounded_bit_oracles(stream.get(), 1U);
  ok &= bounded_bit_oracles(stream.get(), 65U);
  if (resource_status == static_cast<int>(cudaSuccess)) {
    ok &= full_c8000_smoke_and_negatives(stream.get(), resources);
  }
  return ok ? 0 : 1;
}
