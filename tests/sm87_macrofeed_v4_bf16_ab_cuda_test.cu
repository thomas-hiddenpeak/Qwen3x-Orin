#include "q3x/kernels/sm87_macrofeed_v4_bf16_ab.h"

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

constexpr std::uint16_t kScratchSentinel = 0x5a5aU;

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

  [[nodiscard]] bool memset_byte(const int byte,
                                 const cudaStream_t stream) const {
    return pointer_ != nullptr &&
           cudaMemsetAsync(pointer_, byte, count_ * sizeof(T), stream) ==
               cudaSuccess;
  }

  [[nodiscard]] T* get() const noexcept { return pointer_; }
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
  const auto plan = kernels::sm87_macrofeed_v4_bf16_ab_plan(
      kernels::kSm87MacroFeedV4Bf16AbTokens,
      kernels::kSm87MacroFeedV4Bf16AbScratchRowStride);
  ok &= expect(plan.valid() && plan.established_exact_body_reused &&
                   plan.shared_input_dual_projection &&
                   plan.direct_scratch_scatter &&
                   !plan.compact_bridge_present && !plan.tail_present &&
                   !plan.selector_present && !plan.fallback_permitted &&
                   !plan.jit_permitted && !plan.runtime_repack_permitted &&
                   !plan.autotune_permitted && plan.default_off &&
                   !plan.numerical_contract_qualified &&
                   !plan.production_dispatch_eligible &&
                   plan.startup_package_unbound &&
                   !plan.execution_capability &&
                   !plan.caller_snapshot_grants_production_authority,
               "fixed C8000 plan must expose only the admission dataflow");
  ok &= expect(
      !kernels::sm87_macrofeed_v4_bf16_ab_plan(
           kernels::kSm87MacroFeedV4Bf16AbTokens - 1U,
           kernels::kSm87MacroFeedV4Bf16AbScratchRowStride)
           .valid() &&
          !kernels::sm87_macrofeed_v4_bf16_ab_plan(
               kernels::kSm87MacroFeedV4Bf16AbTokens,
               kernels::kSm87MacroFeedV4Bf16AbScratchRowStride - 1U)
               .valid(),
      "non-C8000/non-V4 layouts must fail closed");

  const auto* const a = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'1000'0000'0000ULL));
  const auto* const b = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'2000'0000'0000ULL));
  const auto* const input = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'3000'0000'0000ULL));
  auto* const scratch = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'5000'0000'0000ULL));
  kernels::Sm87MacroFeedV4Bf16AbArguments arguments{
      a,
      b,
      input,
      scratch,
      kernels::kSm87MacroFeedV4Bf16AbTokens,
      kernels::kSm87MacroFeedV4Bf16AbScratchRowStride,
      reinterpret_cast<void*>(0x1000U)};
  ok &= expect(kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(arguments),
               "coherent host-only fake ranges must pass T0 structure");
  auto changed = arguments;
  changed.token_count -= 1U;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(changed),
      "wrong token count must fail structural validation");
  changed = arguments;
  changed.scratch_row_stride -= 1U;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(changed),
      "wrong row stride must fail structural validation");
  changed = arguments;
  changed.cuda_stream = nullptr;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(changed),
      "null stream must fail structural validation");
  changed = arguments;
  changed.a_weights = reinterpret_cast<const std::uint16_t*>(
      reinterpret_cast<std::uintptr_t>(a) + 2U);
  ok &= expect(
      !kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(changed),
      "misaligned payload must fail structural validation");
  changed = arguments;
  changed.scratch = const_cast<std::uint16_t*>(input);
  ok &= expect(
      !kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(changed),
      "scratch/input overlap must fail structural validation");

  kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot synthetic{};
  synthetic.identity = kernels::kSm87MacroFeedV4Bf16AbIdentity;
  synthetic.device_ordinal = 0;
  synthetic.compute_major = 8;
  synthetic.compute_minor = 7;
  synthetic.sm_count =
      static_cast<std::int32_t>(kernels::kSm87MacroFeedV4Bf16AbSmCount);
  synthetic.binary_version = 87;
  synthetic.registers_per_thread = 64;
  synthetic.dynamic_shared_bytes =
      kernels::kSm87MacroFeedV4Bf16AbDynamicSharedBytes;
  synthetic.local_bytes = 128U;
  synthetic.maximum_threads_per_block = 1'024;
  synthetic.active_blocks_per_sm = 2;
  synthetic.threads_per_block =
      static_cast<std::int32_t>(kernels::kSm87MacroFeedV4Bf16AbThreads);
  synthetic.physical_grid_ctas =
      static_cast<std::int32_t>(kernels::kSm87MacroFeedV4Bf16AbGridCtas);
  synthetic.kernel_compiled = true;
  synthetic.exact_geometry = true;
  synthetic.static_resource_gate_passed = true;
  synthetic.startup_package_unbound = true;
  ok &= expect(
      kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(synthetic),
      "resource gate must observe, not invent, a local-byte value");
  synthetic.active_blocks_per_sm = 1;
  ok &= expect(
      !kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(synthetic),
      "less than two resident CTAs must fail closed");
  return ok;
}

[[nodiscard]] bool device_is_supported() {
  int device = -1;
  cudaDeviceProp properties{};
  return cudaGetDevice(&device) == cudaSuccess && device >= 0 &&
         cudaGetDeviceProperties(&properties, device) == cudaSuccess &&
         properties.major == 8 && properties.minor == 7 &&
         properties.multiProcessorCount ==
             static_cast<int>(kernels::kSm87MacroFeedV4Bf16AbSmCount);
}

[[nodiscard]] std::vector<std::uint16_t> make_input(
    const std::size_t token_count) {
  std::vector<std::uint16_t> values(
      token_count * kernels::kSm87MacroFeedV4Bf16AbInputFeatures);
  for (std::size_t token = 0U; token < token_count; ++token) {
    for (std::size_t column = 0U;
         column < kernels::kSm87MacroFeedV4Bf16AbInputFeatures; ++column) {
      const int pattern = static_cast<int>(
                              (token * 17U + column * 7U + column / 11U) %
                              19U) -
                          9;
      values[token * kernels::kSm87MacroFeedV4Bf16AbInputFeatures +
             column] = encode_bf16(static_cast<float>(pattern) / 64.0F);
    }
  }
  return values;
}

[[nodiscard]] std::vector<std::uint16_t> make_weights(
    const bool second_projection) {
  std::vector<std::uint16_t> values(
      kernels::kSm87MacroFeedV4Bf16AbWeightElements);
  for (std::size_t row = 0U;
       row < kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection; ++row) {
    for (std::size_t column = 0U;
         column < kernels::kSm87MacroFeedV4Bf16AbInputFeatures; ++column) {
      const std::size_t mixed = second_projection
                                    ? row * 23U + column * 11U + column / 5U
                                    : row * 13U + column * 3U + column / 7U;
      const int pattern = static_cast<int>(mixed % 17U) - 8;
      const float denominator = second_projection ? 96.0F : 128.0F;
      values[row * kernels::kSm87MacroFeedV4Bf16AbInputFeatures +
             column] =
          encode_bf16(static_cast<float>(pattern) / denominator);
    }
  }
  return values;
}

[[nodiscard]] bool bounded_same_body_test(const cudaStream_t stream) {
  constexpr std::size_t kTokens =
      kernels::kSm87MacroFeedV4Bf16AbOracleTokens;
  const auto host_input = make_input(kTokens);
  const auto host_a = make_weights(false);
  const auto host_b = make_weights(true);
  std::vector<std::uint16_t> host_scratch(
      kTokens * kernels::kSm87MacroFeedV4Bf16AbScratchRowStride,
      kScratchSentinel);

  DeviceBuffer<std::uint16_t> input(host_input.size());
  DeviceBuffer<std::uint16_t> a(host_a.size());
  DeviceBuffer<std::uint16_t> b(host_b.size());
  DeviceBuffer<std::uint16_t> scratch(host_scratch.size());
  DeviceBuffer<std::uint16_t> compact_a(
      kTokens * kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection);
  DeviceBuffer<std::uint16_t> compact_b(
      kTokens * kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection);
  if (!expect(input.get() != nullptr && a.get() != nullptr &&
                  b.get() != nullptr && scratch.get() != nullptr &&
                  compact_a.get() != nullptr && compact_b.get() != nullptr,
              "allocate bounded same-body buffers") ||
      !expect(input.upload(host_input, stream), "upload patterned input") ||
      !expect(a.upload(host_a, stream), "upload A weights") ||
      !expect(b.upload(host_b, stream), "upload B weights") ||
      !expect(scratch.upload(host_scratch, stream),
              "upload sentinel scratch")) {
    return false;
  }

  const kernels::Sm87MacroFeedV4Bf16AbCompactOracleArguments
      compact_arguments{a.get(),
                        b.get(),
                        input.get(),
                        compact_a.get(),
                        compact_b.get(),
                        kTokens,
                        reinterpret_cast<void*>(stream)};
  const int compact_status =
      kernels::launch_sm87_macrofeed_v4_bf16_ab_compact_oracle_cuda(
          compact_arguments);
  if (!expect(compact_status == static_cast<int>(cudaSuccess),
              "launch established compact M64 body")) {
    return false;
  }
  const kernels::Sm87MacroFeedV4Bf16AbOracleArguments arguments{
      a.get(),
      b.get(),
      input.get(),
      scratch.get(),
      kTokens,
      kernels::kSm87MacroFeedV4Bf16AbScratchRowStride,
      reinterpret_cast<void*>(stream)};
  const int direct_status =
      kernels::launch_sm87_macrofeed_v4_bf16_ab_oracle_cuda(arguments);
  if (!expect(direct_status == static_cast<int>(cudaSuccess),
              "launch direct-strided M64 body") ||
      !expect_cuda(cudaStreamSynchronize(stream),
                   "synchronize bounded same-body comparison")) {
    return false;
  }

  std::vector<std::uint16_t> observed_input;
  std::vector<std::uint16_t> observed_a;
  std::vector<std::uint16_t> observed_b;
  std::vector<std::uint16_t> observed_scratch;
  std::vector<std::uint16_t> observed_compact_a;
  std::vector<std::uint16_t> observed_compact_b;
  if (!expect(input.download(&observed_input, stream), "download input") ||
      !expect(a.download(&observed_a, stream), "download A weights") ||
      !expect(b.download(&observed_b, stream), "download B weights") ||
      !expect(scratch.download(&observed_scratch, stream),
              "download direct scratch") ||
      !expect(compact_a.download(&observed_compact_a, stream),
              "download compact A") ||
      !expect(compact_b.download(&observed_compact_b, stream),
              "download compact B") ||
      !expect_cuda(cudaStreamSynchronize(stream),
                   "synchronize bounded downloads")) {
    return false;
  }

  bool ok = true;
  ok &= expect(observed_input == host_input,
               "direct/compact bodies must not mutate input");
  ok &= expect(observed_a == host_a,
               "direct/compact bodies must not mutate A weights");
  ok &= expect(observed_b == host_b,
               "direct/compact bodies must not mutate B weights");
  bool projections_differ = false;
  bool a_bitwise_identical = true;
  bool b_bitwise_identical = true;
  bool scratch_preserved = true;
  for (std::size_t token = 0U; token < kTokens; ++token) {
    const std::size_t scratch_row =
        token * kernels::kSm87MacroFeedV4Bf16AbScratchRowStride;
    const std::size_t compact_row =
        token * kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection;
    for (std::size_t column = 0U;
         column < kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection;
         ++column) {
      const auto direct_a = observed_scratch[
          scratch_row + kernels::kSm87MacroFeedV4Bf16AbAOffset + column];
      const auto direct_b = observed_scratch[
          scratch_row + kernels::kSm87MacroFeedV4Bf16AbBOffset + column];
      a_bitwise_identical &=
          direct_a == observed_compact_a[compact_row + column];
      b_bitwise_identical &=
          direct_b == observed_compact_b[compact_row + column];
      projections_differ |= direct_a != direct_b;
    }
    for (std::size_t column = 0U;
         column < kernels::kSm87MacroFeedV4Bf16AbScratchRowStride;
         ++column) {
      const bool written =
          column >= kernels::kSm87MacroFeedV4Bf16AbAOffset &&
          column < kernels::kSm87MacroFeedV4Bf16AbBOffset +
                       kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection;
      if (!written) {
        scratch_preserved &=
            observed_scratch[scratch_row + column] == kScratchSentinel;
      }
    }
  }
  ok &= expect(a_bitwise_identical,
               "A direct scatter must be bitwise compact-identical");
  ok &= expect(b_bitwise_identical,
               "B direct scatter must be bitwise compact-identical");
  ok &= expect(scratch_preserved,
               "direct body must preserve every non-A/B scratch slot");
  ok &= expect(projections_differ,
               "different A/B patterns must produce distinguishable output");
  return ok;
}

[[nodiscard]] bool production_admission_test(const cudaStream_t stream) {
  kernels::Sm87MacroFeedV4Bf16AbAdmissionResourceSnapshot resources{};
  if (!expect(
          kernels::query_sm87_macrofeed_v4_bf16_ab_admission_resource_snapshot_cuda(
              &resources) == static_cast<int>(cudaSuccess),
          "query observed V4 BF16 A/B resources") ||
      !expect(
          kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
              resources),
          "observed V4 BF16 A/B resources must pass fixed gate")) {
    return false;
  }
  std::cout << "SM87_MACROFEED_V4_BF16_AB_RESOURCES: regs="
            << resources.registers_per_thread
            << " static_shared=" << resources.static_shared_bytes
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " local=" << resources.local_bytes
            << " active_ctas_per_sm=" << resources.active_blocks_per_sm
            << '\n';

  DeviceBuffer<std::uint16_t> a(
      kernels::kSm87MacroFeedV4Bf16AbWeightElements);
  DeviceBuffer<std::uint16_t> b(
      kernels::kSm87MacroFeedV4Bf16AbWeightElements);
  DeviceBuffer<std::uint16_t> input(
      kernels::kSm87MacroFeedV4Bf16AbInputBytes / sizeof(std::uint16_t));
  DeviceBuffer<std::uint16_t> scratch(
      kernels::kSm87MacroFeedV4Bf16AbScratchBytes /
      sizeof(std::uint16_t));
  if (!expect(a.get() != nullptr && b.get() != nullptr &&
                  input.get() != nullptr && scratch.get() != nullptr,
              "allocate complete C8000 admission ranges") ||
      !expect(a.memset_byte(0, stream), "zero production A weights") ||
      !expect(b.memset_byte(0, stream), "zero production B weights") ||
      !expect(input.memset_byte(0, stream), "zero production input") ||
      !expect(scratch.memset_byte(0x5a, stream),
              "initialize production scratch sentinel")) {
    return false;
  }

  kernels::Sm87MacroFeedV4Bf16AbArguments arguments{
      a.get(),
      b.get(),
      input.get(),
      scratch.get(),
      kernels::kSm87MacroFeedV4Bf16AbTokens,
      kernels::kSm87MacroFeedV4Bf16AbScratchRowStride,
      reinterpret_cast<void*>(stream)};
  if (!expect(kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(arguments),
              "complete C8000 arguments must pass structural validation")) {
    return false;
  }
  kernels::Sm87MacroFeedV4Bf16AbAdmissionLaunchReceipt receipt{};
  const int launch_status =
      kernels::launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
          arguments, resources, &receipt);
  if (!expect(launch_status == static_cast<int>(cudaSuccess),
              "enqueue complete C8000 direct-scatter grid") ||
      !expect(receipt.valid_enqueue_receipt() &&
                  !receipt.completion_observed &&
                  !receipt.production_dispatch_eligible &&
                  !receipt.execution_capability,
              "receipt must attest enqueue without completion/production") ||
      !expect_cuda(cudaStreamSynchronize(stream),
                   "synchronize complete C8000 admission run")) {
    return false;
  }

  bool ok = true;
  constexpr std::array<std::size_t, 2U> kObservedTokens{{
      0U, kernels::kSm87MacroFeedV4Bf16AbTokens - 1U}};
  for (const std::size_t token : kObservedTokens) {
    const std::size_t row =
        token * kernels::kSm87MacroFeedV4Bf16AbScratchRowStride;
    std::array<std::uint16_t,
               kernels::kSm87MacroFeedV4Bf16AbLogicalRows>
        outputs{};
    std::uint16_t before = 0U;
    std::uint16_t after = 0U;
    ok &= expect_cuda(
        cudaMemcpy(outputs.data(),
                   scratch.get() + row +
                       kernels::kSm87MacroFeedV4Bf16AbAOffset,
                   sizeof(outputs), cudaMemcpyDeviceToHost),
        "copy first/last production A/B row");
    ok &= expect_cuda(
        cudaMemcpy(&before,
                   scratch.get() + row +
                       kernels::kSm87MacroFeedV4Bf16AbAOffset - 1U,
                   sizeof(before), cudaMemcpyDeviceToHost),
        "copy production prefix sentinel");
    ok &= expect_cuda(
        cudaMemcpy(&after,
                   scratch.get() + row +
                       kernels::kSm87MacroFeedV4Bf16AbBOffset +
                       kernels::kSm87MacroFeedV4Bf16AbRowsPerProjection,
                   sizeof(after), cudaMemcpyDeviceToHost),
        "copy production suffix sentinel");
    ok &= expect(std::all_of(outputs.begin(), outputs.end(),
                             [](const std::uint16_t bits) {
                               return bits == 0U;
                             }),
                 "zero production payload must produce exact BF16 zero");
    ok &= expect(before == kScratchSentinel && after == kScratchSentinel,
                 "production grid must preserve adjacent scratch slots");
  }

  kernels::Sm87MacroFeedV4Bf16AbAdmissionLaunchReceipt rejected{};
  ok &= expect(
      kernels::launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
          arguments, resources, nullptr) ==
          static_cast<int>(cudaErrorInvalidValue),
      "null receipt must fail before enqueue");

  auto changed_arguments = arguments;
  changed_arguments.cuda_stream = nullptr;
  ok &= expect(
      kernels::launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
          changed_arguments, resources, &rejected) ==
          static_cast<int>(cudaErrorInvalidValue),
      "null stream must fail before enqueue");

  auto changed_resources = resources;
  changed_resources.registers_per_thread += 1;
  ok &= expect(
      kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          changed_resources) &&
          kernels::launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
              arguments, changed_resources, &rejected) ==
              static_cast<int>(cudaErrorLaunchOutOfResources),
      "caller resource snapshot must exactly match the observation");

  changed_resources = resources;
  changed_resources.device_ordinal += 1;
  ok &= expect(
      kernels::sm87_macrofeed_v4_bf16_ab_admission_resource_gate(
          changed_resources) &&
          kernels::launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
              arguments, changed_resources, &rejected) ==
              static_cast<int>(cudaErrorInvalidDevice),
      "wrong snapshot device must fail closed");

  changed_arguments = arguments;
  changed_arguments.scratch += 8U;
  ok &= expect(
      kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(changed_arguments) &&
          kernels::launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
              changed_arguments, resources, &rejected) ==
              static_cast<int>(cudaErrorInvalidDevicePointer),
      "offset scratch origin must fail complete allocation coverage");

  ManagedBuffer managed;
  ok &= expect(managed.allocate(kernels::kSm87MacroFeedV4Bf16AbInputBytes),
               "allocate managed-memory negative");
  if (managed.get() != nullptr) {
    changed_arguments = arguments;
    changed_arguments.input =
        static_cast<const std::uint16_t*>(managed.get());
    ok &= expect(
        kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(
            changed_arguments) &&
            kernels::launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
                changed_arguments, resources, &rejected) ==
                static_cast<int>(cudaErrorInvalidDevicePointer),
        "managed input must fail exact device-allocation ownership");
  }

  const auto* const fake_a = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'1000'0000'0000ULL));
  const auto* const fake_b = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'2000'0000'0000ULL));
  const auto* const fake_input = reinterpret_cast<const std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'3000'0000'0000ULL));
  auto* const fake_scratch = reinterpret_cast<std::uint16_t*>(
      static_cast<std::uintptr_t>(0x0000'5000'0000'0000ULL));
  changed_arguments = {fake_a,
                       fake_b,
                       fake_input,
                       fake_scratch,
                       kernels::kSm87MacroFeedV4Bf16AbTokens,
                       kernels::kSm87MacroFeedV4Bf16AbScratchRowStride,
                       reinterpret_cast<void*>(stream)};
  ok &= expect(
      kernels::sm87_macrofeed_v4_bf16_ab_arguments_valid(changed_arguments) &&
          kernels::launch_sm87_macrofeed_v4_bf16_ab_admission_cuda(
              changed_arguments, resources, &rejected) ==
              static_cast<int>(cudaErrorInvalidDevicePointer),
      "host-only fake ranges must never cross the T1 device gate");
  return ok;
}

}  // namespace

int main() {
  if (!host_contract_test()) {
    return 1;
  }
  if (!device_is_supported()) {
    std::cout << "SKIP: fixed MacroFeed V4 BF16 A/B admission requires "
                 "the 16-SM SM87 target\n";
    return 77;
  }
  Stream stream;
  if (!expect(stream.get() != nullptr,
              "create nonblocking CUDA correctness stream")) {
    return 1;
  }
  if (!bounded_same_body_test(stream.get()) ||
      !production_admission_test(stream.get())) {
    return 1;
  }
  std::cout << "PASS: fixed C8000 BF16 A/B reuses the established exact "
               "M64N96K64 body and scatters directly into V4 scratch\n";
  return 0;
}
