#include "q3x/kernels/sm87_weight_only_gemv.h"

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace q3x::kernels {

// Test-compatibility query implemented beside the production launcher. It is
// intentionally not part of the public installed API.
[[nodiscard]] int
query_sm87_nvfp4_w4a16_whole_chunk_gate_m128_b_reuse_resources_test_cuda(
    std::size_t token_count, std::size_t rows, std::size_t columns,
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::kernels

namespace {

constexpr std::size_t kRows = 17'408U;
constexpr std::size_t kColumns = 5'120U;
constexpr std::size_t kScaleColumns = kColumns / 16U;
constexpr std::size_t kPackedBytes = kRows * (kColumns / 2U);
constexpr std::size_t kScaleBytes = kRows * kScaleColumns;
constexpr std::size_t kGuardElements = 64U;
constexpr std::size_t kScrubBytes = 32U * 1024U * 1024U;
constexpr std::size_t kRequestedPersistingBytes = 2'883'584U;
constexpr std::size_t kNominalPersistentBudget = 2'621'440U;
constexpr int kWarmups = 10;
constexpr int kIterations = 24;
constexpr int kRounds = 6;
constexpr double kRequiredC512Speedup = 1.02;
constexpr std::array<std::uint8_t, 32U> kCheckpointLikeScaleCodes{{
    0x4eU, 0x50U, 0x52U, 0x54U, 0x55U, 0x56U, 0x57U, 0x58U,
    0x58U, 0x58U, 0x59U, 0x59U, 0x59U, 0x5aU, 0x5aU, 0x5bU,
    0x5bU, 0x5cU, 0x5cU, 0x5dU, 0x5dU, 0x5eU, 0x5fU, 0x60U,
    0x60U, 0x61U, 0x62U, 0x63U, 0x64U, 0x65U, 0x66U, 0x67U,
}};

class TestContext {
 public:
  void expect(const bool condition, const std::string& message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] bool cuda_ok(const cudaError_t status,
                             const std::string& operation) {
    expect(status == cudaSuccess,
           operation + ": " + cudaGetErrorString(status));
    return status == cudaSuccess;
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <typename T>
class DeviceBuffer {
 public:
  DeviceBuffer() = default;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;

  ~DeviceBuffer() {
    if (data_ != nullptr) {
      (void)cudaFree(data_);
    }
  }

  [[nodiscard]] bool allocate(TestContext& test, const std::size_t count,
                              const std::string& label) {
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
      test.expect(false, label + " allocation size is representable");
      return false;
    }
    return test.cuda_ok(
        cudaMalloc(reinterpret_cast<void**>(&data_), count * sizeof(T)),
        label);
  }

  [[nodiscard]] T* get() noexcept { return data_; }
  [[nodiscard]] const T* get() const noexcept { return data_; }

 private:
  T* data_ = nullptr;
};

class StreamAndEvents {
 public:
  StreamAndEvents() = default;
  StreamAndEvents(const StreamAndEvents&) = delete;
  StreamAndEvents& operator=(const StreamAndEvents&) = delete;

  ~StreamAndEvents() {
    if (stop_ != nullptr) {
      (void)cudaEventDestroy(stop_);
    }
    if (start_ != nullptr) {
      (void)cudaEventDestroy(start_);
    }
    if (done_ != nullptr) {
      (void)cudaEventDestroy(done_);
    }
    if (ready_ != nullptr) {
      (void)cudaEventDestroy(ready_);
    }
    if (auxiliary_ != nullptr) {
      (void)cudaStreamDestroy(auxiliary_);
    }
    if (main_ != nullptr) {
      (void)cudaStreamDestroy(main_);
    }
  }

  [[nodiscard]] bool create(TestContext& test) {
    bool ready = test.cuda_ok(
        cudaStreamCreateWithFlags(&main_, cudaStreamNonBlocking),
        "create main stream");
    ready = ready && test.cuda_ok(
                         cudaStreamCreateWithFlags(&auxiliary_,
                                                   cudaStreamNonBlocking),
                         "create auxiliary stream");
    ready = ready && test.cuda_ok(
                         cudaEventCreateWithFlags(&ready_,
                                                  cudaEventDisableTiming),
                         "create branch-ready event");
    ready = ready && test.cuda_ok(
                         cudaEventCreateWithFlags(&done_,
                                                  cudaEventDisableTiming),
                         "create branch-done event");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&start_), "create timing start");
    ready = ready &&
            test.cuda_ok(cudaEventCreate(&stop_), "create timing stop");
    return ready;
  }

  [[nodiscard]] cudaStream_t main() const noexcept { return main_; }
  [[nodiscard]] cudaStream_t auxiliary() const noexcept { return auxiliary_; }
  [[nodiscard]] cudaEvent_t ready() const noexcept { return ready_; }
  [[nodiscard]] cudaEvent_t done() const noexcept { return done_; }
  [[nodiscard]] cudaEvent_t start() const noexcept { return start_; }
  [[nodiscard]] cudaEvent_t stop() const noexcept { return stop_; }

 private:
  cudaStream_t main_ = nullptr;
  cudaStream_t auxiliary_ = nullptr;
  cudaEvent_t ready_ = nullptr;
  cudaEvent_t done_ = nullptr;
  cudaEvent_t start_ = nullptr;
  cudaEvent_t stop_ = nullptr;
};

[[nodiscard]] std::uint16_t encode_bf16(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t magnitude = bits & 0x7fff'ffffU;
  if (magnitude > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  bits += 0x7fffU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>(bits >> 16U);
}

[[nodiscard]] bool is_bf16_finite(const std::uint16_t value) noexcept {
  return (value & 0x7f80U) != 0x7f80U;
}

[[nodiscard]] bool same_window(const cudaAccessPolicyWindow& left,
                               const cudaAccessPolicyWindow& right) noexcept {
  return left.base_ptr == right.base_ptr &&
         left.num_bytes == right.num_bytes &&
         left.hitRatio == right.hitRatio && left.hitProp == right.hitProp &&
         left.missProp == right.missProp;
}

[[nodiscard]] cudaStreamAttrValue neutral_stream_attribute() noexcept {
  cudaStreamAttrValue value{};
  value.accessPolicyWindow.base_ptr = nullptr;
  value.accessPolicyWindow.num_bytes = 0U;
  value.accessPolicyWindow.hitRatio = 0.0F;
  value.accessPolicyWindow.hitProp = cudaAccessPropertyNormal;
  value.accessPolicyWindow.missProp = cudaAccessPropertyNormal;
  return value;
}

struct PolicySnapshot {
  std::size_t limit = 0U;
  cudaStreamAttrValue main{};
  cudaStreamAttrValue auxiliary{};
};

enum class PersistPolicy {
  kBalanced,
  kMainOwner,
};

[[nodiscard]] const char* policy_name(const PersistPolicy policy) noexcept {
  return policy == PersistPolicy::kBalanced ? "balanced" : "main_owner";
}

class PolicyController {
 public:
  PolicyController(TestContext& test, const cudaStream_t main,
                   const cudaStream_t auxiliary,
                   const cudaDeviceProp& properties)
      : test_(test),
        main_(main),
        auxiliary_(auxiliary),
        maximum_persisting_bytes_(
            static_cast<std::size_t>(properties.persistingL2CacheMaxSize)),
        maximum_window_bytes_(
            static_cast<std::size_t>(properties.accessPolicyMaxWindowSize)) {}

  PolicyController(const PolicyController&) = delete;
  PolicyController& operator=(const PolicyController&) = delete;

  ~PolicyController() {
    if (captured_ && !restored_) {
      (void)restore(false);
    }
  }

  [[nodiscard]] bool capture() {
    bool ready = test_.cuda_ok(
        cudaDeviceGetLimit(&original_.limit,
                           cudaLimitPersistingL2CacheSize),
        "read original persisting-L2 limit");
    ready = ready && test_.cuda_ok(
                         cudaStreamGetAttribute(
                             main_, cudaStreamAttributeAccessPolicyWindow,
                             &original_.main),
                         "read original main APW");
    ready = ready && test_.cuda_ok(
                         cudaStreamGetAttribute(
                             auxiliary_,
                             cudaStreamAttributeAccessPolicyWindow,
                             &original_.auxiliary),
                         "read original auxiliary APW");
    captured_ = ready;
    std::cout << "L2_APW_ORIGINAL: limit_bytes=" << original_.limit
              << " main_window_bytes="
              << original_.main.accessPolicyWindow.num_bytes
              << " auxiliary_window_bytes="
              << original_.auxiliary.accessPolicyWindow.num_bytes << '\n';
    return ready;
  }

  [[nodiscard]] bool snapshot(PolicySnapshot& result,
                              const std::string& label) {
    bool ready = test_.cuda_ok(
        cudaDeviceGetLimit(&result.limit, cudaLimitPersistingL2CacheSize),
        label + " read limit");
    ready = ready && test_.cuda_ok(
                         cudaStreamGetAttribute(
                             main_, cudaStreamAttributeAccessPolicyWindow,
                             &result.main),
                         label + " read main APW");
    ready = ready && test_.cuda_ok(
                         cudaStreamGetAttribute(
                             auxiliary_,
                             cudaStreamAttributeAccessPolicyWindow,
                             &result.auxiliary),
                         label + " read auxiliary APW");
    return ready;
  }

  [[nodiscard]] bool unchanged_from(
      const PolicySnapshot& expected, const PolicySnapshot& actual) const {
    return expected.limit == actual.limit &&
           same_window(expected.main.accessPolicyWindow,
                       actual.main.accessPolicyWindow) &&
           same_window(expected.auxiliary.accessPolicyWindow,
                       actual.auxiliary.accessPolicyWindow);
  }

  [[nodiscard]] bool apply_baseline(const std::string& label) {
    ++apply_count_;
    return prepare_common(original_.limit, neutral_stream_attribute(),
                          neutral_stream_attribute(), label, "baseline");
  }

  [[nodiscard]] bool apply_candidate(const std::uint16_t* const activations,
                                     const std::size_t token_count,
                                     const PersistPolicy policy,
                                     const std::string& label) {
    ++apply_count_;
    const std::size_t activation_bytes =
        token_count * kColumns * sizeof(std::uint16_t);
    if (activations == nullptr ||
        (token_count != 256U && token_count != 512U) ||
        activation_bytes > maximum_window_bytes_ ||
        maximum_persisting_bytes_ < kRequestedPersistingBytes) {
      test_.expect(false, label + " candidate capability gate");
      return false;
    }

    const float balanced_ratio = token_count == 512U ? 0.25F : 0.5F;
    const float owner_ratio = token_count == 512U ? 0.5F : 1.0F;
    const float main_ratio =
        policy == PersistPolicy::kBalanced ? balanced_ratio : owner_ratio;
    const float auxiliary_ratio =
        policy == PersistPolicy::kBalanced ? balanced_ratio : 0.0F;
    cudaStreamAttrValue main_value = make_candidate_attribute(
        activations, activation_bytes, main_ratio);
    cudaStreamAttrValue auxiliary_value =
        auxiliary_ratio == 0.0F
            ? neutral_stream_attribute()
            : make_candidate_attribute(activations, activation_bytes,
                                       auxiliary_ratio);
    const double aggregate_selected =
        static_cast<double>(activation_bytes) *
        static_cast<double>(main_ratio + auxiliary_ratio);
    const bool budget_safe =
        aggregate_selected <=
        static_cast<double>(kNominalPersistentBudget);
    test_.expect(budget_safe, label + " nominal persistent budget is safe");
    std::cout << "L2_APW_POLICY: label=" << label
              << " policy=" << policy_name(policy)
              << " tokens=" << token_count
              << " window_bytes_per_stream=" << activation_bytes
              << " main_hit_ratio=" << main_ratio
              << " auxiliary_hit_ratio=" << auxiliary_ratio
              << " main_selected_bytes="
              << static_cast<std::size_t>(
                     static_cast<double>(activation_bytes) * main_ratio)
              << " auxiliary_selected_bytes="
              << static_cast<std::size_t>(
                     static_cast<double>(activation_bytes) * auxiliary_ratio)
              << " aggregate_selected_bytes="
              << static_cast<std::size_t>(aggregate_selected)
              << " requested_limit_bytes=" << kRequestedPersistingBytes
              << " requested_capacity_margin_bytes="
              << kRequestedPersistingBytes - kNominalPersistentBudget
              << " budget_gate=" << (budget_safe ? "PASS" : "FAIL")
              << '\n';
    return budget_safe &&
           prepare_common(kRequestedPersistingBytes, main_value,
                          auxiliary_value, label, policy_name(policy));
  }

  [[nodiscard]] bool restore(const bool report = true) {
    if (!captured_ || restored_) {
      return true;
    }
    PolicySnapshot after{};
    const bool exact = restore_snapshot(original_, after, "restore");
    if (report) {
      std::cout << "L2_APW_RESTORE: original_limit_bytes=" << original_.limit
                << " restored_limit_bytes=" << after.limit
                << " stream_attributes_exact="
                << (exact ? "true" : "false")
                << " cache_contents=normalized_not_restored"
                << " gate=" << (exact ? "PASS" : "FAIL") << '\n';
    }
    restored_ = exact;
    return exact;
  }

  [[nodiscard]] std::size_t apply_count() const noexcept {
    return apply_count_;
  }

 private:
  [[nodiscard]] static cudaStreamAttrValue make_candidate_attribute(
      const std::uint16_t* const activations,
      const std::size_t activation_bytes, const float hit_ratio) noexcept {
    cudaStreamAttrValue value{};
    value.accessPolicyWindow.base_ptr =
        const_cast<std::uint16_t*>(activations);
    value.accessPolicyWindow.num_bytes = activation_bytes;
    value.accessPolicyWindow.hitRatio = hit_ratio;
    value.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    value.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    return value;
  }

  [[nodiscard]] bool synchronize(const std::string& label) {
    bool ready = test_.cuda_ok(cudaStreamSynchronize(main_),
                               label + " synchronize main");
    ready = test_.cuda_ok(cudaStreamSynchronize(auxiliary_),
                          label + " synchronize auxiliary") &&
            ready;
    return ready;
  }

  [[nodiscard]] bool restore_snapshot(const PolicySnapshot& target,
                                      PolicySnapshot& after,
                                      const std::string& label) {
    bool ready = synchronize(label);
    if (!ready) {
      test_.expect(false, label + " refuses mutation before quiescence");
      return false;
    }
    ready = test_.cuda_ok(
                cudaStreamSetAttribute(
                    main_, cudaStreamAttributeAccessPolicyWindow,
                    &target.main),
                label + " restore main APW") &&
            ready;
    ready = test_.cuda_ok(
                cudaStreamSetAttribute(
                    auxiliary_, cudaStreamAttributeAccessPolicyWindow,
                    &target.auxiliary),
                label + " restore auxiliary APW") &&
            ready;
    ready = test_.cuda_ok(cudaCtxResetPersistingL2Cache(),
                          label + " normalize persisting lines") &&
            ready;
    ready = test_.cuda_ok(
                cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize,
                                   target.limit),
                label + " restore persisting-L2 limit") &&
            ready;
    ready = snapshot(after, label + " verify restored policy") && ready;
    const bool exact = ready && unchanged_from(target, after);
    test_.expect(exact,
                 label + " restores exact stream attributes and limit");
    return exact;
  }

  [[nodiscard]] bool prepare_common(
      const std::size_t target_limit, const cudaStreamAttrValue& main_value,
      const cudaStreamAttrValue& auxiliary_value, const std::string& label,
      const std::string& policy) {
    const cudaStreamAttrValue neutral = neutral_stream_attribute();
    bool ready = synchronize(label + " pre-policy");
    if (!ready) {
      test_.expect(false,
                   label + " refuses policy mutation before quiescence");
      return false;
    }
    ready = test_.cuda_ok(
        cudaStreamSetAttribute(main_,
                               cudaStreamAttributeAccessPolicyWindow,
                               &neutral),
        label + " neutralize main APW");
    if (ready) {
      ready = test_.cuda_ok(
          cudaStreamSetAttribute(auxiliary_,
                                 cudaStreamAttributeAccessPolicyWindow,
                                 &neutral),
          label + " neutralize auxiliary APW");
    }
    if (ready) {
      ready = test_.cuda_ok(cudaCtxResetPersistingL2Cache(),
                            label + " reset persisting lines");
    }
    if (ready) {
      ready = test_.cuda_ok(
          cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, target_limit),
          label + " set persisting-L2 limit");
    }
    std::size_t actual_limit = 0U;
    if (ready) {
      ready = test_.cuda_ok(
          cudaDeviceGetLimit(&actual_limit,
                             cudaLimitPersistingL2CacheSize),
          label + " read applied limit");
    }
    if (ready) {
      ready = test_.cuda_ok(
          cudaStreamSetAttribute(main_,
                                 cudaStreamAttributeAccessPolicyWindow,
                                 &main_value),
          label + " set main APW");
    }
    if (ready) {
      const cudaError_t auxiliary_status = cudaStreamSetAttribute(
          auxiliary_, cudaStreamAttributeAccessPolicyWindow,
          &auxiliary_value);
      ready = test_.cuda_ok(auxiliary_status,
                            label + " set auxiliary APW") &&
              ready;
    }
    PolicySnapshot applied{};
    if (ready) {
      ready = snapshot(applied, label + " verify applied policy");
    }
    const bool baseline_limit_exact =
        target_limit != original_.limit || actual_limit == target_limit;
    const bool candidate_limit_sufficient =
        target_limit == original_.limit ||
        actual_limit >= kRequestedPersistingBytes;
    const bool exact = ready && applied.limit == actual_limit &&
                       baseline_limit_exact && candidate_limit_sufficient &&
                       same_window(applied.main.accessPolicyWindow,
                                   main_value.accessPolicyWindow) &&
                       same_window(applied.auxiliary.accessPolicyWindow,
                                   auxiliary_value.accessPolicyWindow);
    test_.expect(exact, label + " applied policy reads back exactly");
    std::cout << "L2_APW_APPLY: label=" << label
              << " policy=" << policy
              << " requested_limit_bytes=" << target_limit
              << " actual_limit_bytes=" << actual_limit
              << " exact=" << (exact ? "true" : "false") << '\n';
    if (!exact) {
      PolicySnapshot rollback_after{};
      const bool rollback = restore_snapshot(
          original_, rollback_after, label + " partial-failure rollback");
      std::cout << "L2_APW_ROLLBACK: label=" << label
                << " exact=" << (rollback ? "true" : "false") << '\n';
    }
    return exact;
  }

  TestContext& test_;
  cudaStream_t main_ = nullptr;
  cudaStream_t auxiliary_ = nullptr;
  std::size_t maximum_persisting_bytes_ = 0U;
  std::size_t maximum_window_bytes_ = 0U;
  PolicySnapshot original_{};
  std::size_t apply_count_ = 0U;
  bool captured_ = false;
  bool restored_ = false;
};

__global__ void scrub_l2_kernel(std::uint32_t* const words,
                                const std::size_t count) {
  for (std::size_t index =
           static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       index < count;
       index += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
    words[index] = words[index] + static_cast<std::uint32_t>(index) + 1U;
  }
}

struct Fixture {
  std::size_t token_count = 512U;
  DeviceBuffer<std::uint8_t> gate_packed;
  DeviceBuffer<std::uint8_t> up_packed;
  DeviceBuffer<std::uint8_t> gate_scales;
  DeviceBuffer<std::uint8_t> up_scales;
  DeviceBuffer<std::uint16_t> activations;
  DeviceBuffer<std::uint16_t> gate_output_store;
  DeviceBuffer<std::uint16_t> up_output_store;
  DeviceBuffer<std::uint32_t> scrub;
  std::vector<std::uint8_t> host_gate_packed;
  std::vector<std::uint8_t> host_up_packed;
  std::vector<std::uint8_t> host_gate_scales;
  std::vector<std::uint8_t> host_up_scales;
  std::vector<std::uint16_t> host_activations;

  [[nodiscard]] std::size_t output_elements() const noexcept {
    return token_count * kRows;
  }

  [[nodiscard]] std::size_t guarded_output_elements() const noexcept {
    return output_elements() + 2U * kGuardElements;
  }

  [[nodiscard]] std::uint16_t* gate_output() noexcept {
    return gate_output_store.get() + kGuardElements;
  }

  [[nodiscard]] std::uint16_t* up_output() noexcept {
    return up_output_store.get() + kGuardElements;
  }

  [[nodiscard]] bool initialize(TestContext& test, const cudaStream_t stream,
                                const std::size_t tokens) {
    token_count = tokens;
    const std::size_t activation_elements = token_count * kColumns;
    host_gate_packed.resize(kPackedBytes);
    host_up_packed.resize(kPackedBytes);
    host_gate_scales.resize(kScaleBytes);
    host_up_scales.resize(kScaleBytes);
    host_activations.resize(activation_elements);
    for (std::size_t index = 0U; index < kPackedBytes; ++index) {
      const std::uint8_t low =
          static_cast<std::uint8_t>((index * 3U + (index >> 7U)) & 0x0fU);
      const std::uint8_t high = static_cast<std::uint8_t>(
          (index * 5U + (index >> 5U) + 1U) & 0x0fU);
      host_gate_packed[index] =
          static_cast<std::uint8_t>(low | (high << 4U));
      host_up_packed[index] = static_cast<std::uint8_t>(
          (low ^ 0x09U) | ((high ^ 0x05U) << 4U));
    }
    for (std::size_t index = 0U; index < kScaleBytes; ++index) {
      const std::size_t row = index / kScaleColumns;
      const std::size_t scale_column = index - row * kScaleColumns;
      host_gate_scales[index] = kCheckpointLikeScaleCodes[
          (scale_column * 5U + row * 11U + (scale_column >> 3U)) %
          kCheckpointLikeScaleCodes.size()];
      host_up_scales[index] = kCheckpointLikeScaleCodes[
          (scale_column * 7U + row * 13U + (scale_column >> 2U) + 3U) %
          kCheckpointLikeScaleCodes.size()];
    }
    for (std::size_t index = 0U; index < activation_elements; ++index) {
      const int centered = static_cast<int>(index % 29U) - 14;
      host_activations[index] =
          encode_bf16(static_cast<float>(centered) / 16.0F);
    }

    bool ready = gate_packed.allocate(test, kPackedBytes,
                                      "allocate Gate packed weights");
    ready = ready && up_packed.allocate(test, kPackedBytes,
                                        "allocate Up packed weights");
    ready = ready && gate_scales.allocate(test, kScaleBytes,
                                          "allocate Gate scales");
    ready = ready && up_scales.allocate(test, kScaleBytes,
                                        "allocate Up scales");
    ready = ready && activations.allocate(test, activation_elements,
                                          "allocate activations");
    ready = ready && gate_output_store.allocate(
                         test, guarded_output_elements(),
                         "allocate guarded Gate output");
    ready = ready && up_output_store.allocate(
                         test, guarded_output_elements(),
                         "allocate guarded Up output");
    ready = ready && scrub.allocate(test, kScrubBytes / sizeof(std::uint32_t),
                                    "allocate L2 scrub buffer");
    if (!ready) {
      return false;
    }
    ready = test.cuda_ok(
        cudaMemcpyAsync(gate_packed.get(), host_gate_packed.data(),
                        kPackedBytes, cudaMemcpyHostToDevice, stream),
        "upload Gate packed weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(up_packed.get(),
                                         host_up_packed.data(), kPackedBytes,
                                         cudaMemcpyHostToDevice, stream),
                         "upload Up packed weights");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(gate_scales.get(),
                                         host_gate_scales.data(), kScaleBytes,
                                         cudaMemcpyHostToDevice, stream),
                         "upload Gate scales");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(up_scales.get(),
                                         host_up_scales.data(), kScaleBytes,
                                         cudaMemcpyHostToDevice, stream),
                         "upload Up scales");
    ready = ready && test.cuda_ok(
                         cudaMemcpyAsync(
                             activations.get(), host_activations.data(),
                             activation_elements * sizeof(std::uint16_t),
                             cudaMemcpyHostToDevice, stream),
                         "upload activations");
    ready = ready && test.cuda_ok(
                         cudaMemsetAsync(scrub.get(), 0, kScrubBytes, stream),
                         "initialize L2 scrub buffer");
    ready = ready && test.cuda_ok(cudaStreamSynchronize(stream),
                                  "fixture upload synchronize");
    return ready;
  }
};

[[nodiscard]] bool run_resource_gate(TestContext& test) {
  bool complete = true;
  for (const std::size_t token_count : {256U, 512U}) {
    int registers = -1;
    std::size_t shared = std::numeric_limits<std::size_t>::max();
    std::size_t local = std::numeric_limits<std::size_t>::max();
    int threads = -1;
    int active = -1;
    const int status = q3x::kernels::
        query_sm87_nvfp4_w4a16_whole_chunk_gate_m128_b_reuse_resources_test_cuda(
            token_count, kRows, kColumns, &registers, &shared, &local,
            &threads, &active);
    const bool c256 = token_count == 256U;
    const int expected_registers = c256 ? 126 : 244;
    const std::size_t expected_shared = c256 ? 37'376U : 512U;
    const int expected_active = c256 ? 2 : 1;
    const bool gate = status == static_cast<int>(cudaSuccess) &&
                      registers == expected_registers &&
                      shared == expected_shared && local == 0U &&
                      threads == 256 && active == expected_active;
    std::cout << "L2_APW_PRODUCTION_RESOURCES: tokens=" << token_count
              << " status=" << status << " registers=" << registers
              << " static_shared_bytes=" << shared
              << " local_bytes=" << local << " threads=" << threads
              << " active_blocks_per_sm=" << active
              << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
    test.expect(gate, "production Gate M128 resource identity");
    complete = complete && gate;
  }
  return complete;
}

[[nodiscard]] bool run_invalid_graph_gate(TestContext& test,
                                          const cudaStream_t stream) {
  constexpr std::size_t kTokens = 512U;
  constexpr std::uintptr_t kWeightAddress = 0x1'0000'0000ULL;
  constexpr std::uintptr_t kScaleAddress = 0x2'0000'0000ULL;
  constexpr std::uintptr_t kActivationAddress = 0x3'0000'0000ULL;
  constexpr std::uintptr_t kOutputAddress = 0x4'0000'0000ULL;
  constexpr std::uintptr_t kMaximumAddress =
      std::numeric_limits<std::uintptr_t>::max();
  const auto* const weights =
      reinterpret_cast<const std::uint8_t*>(kWeightAddress);
  const auto* const scales =
      reinterpret_cast<const std::uint8_t*>(kScaleAddress);
  const auto* const activations =
      reinterpret_cast<const std::uint16_t*>(kActivationAddress);
  auto* const output = reinterpret_cast<std::uint16_t*>(kOutputAddress);
  auto* const packed_alias =
      reinterpret_cast<std::uint16_t*>(kWeightAddress + 128U);
  auto* const scale_alias =
      reinterpret_cast<std::uint16_t*>(kScaleAddress + 128U);
  auto* const activation_alias = reinterpret_cast<std::uint16_t*>(
      kActivationAddress + 128U * kColumns * sizeof(std::uint16_t));
  const auto* const wrapping_weights =
      reinterpret_cast<const std::uint8_t*>(kMaximumAddress - 15U);
  const auto* const wrapping_scales =
      reinterpret_cast<const std::uint8_t*>(kMaximumAddress - 1U);
  const auto* const wrapping_activations =
      reinterpret_cast<const std::uint16_t*>(kMaximumAddress - 7U);
  auto* const wrapping_output =
      reinterpret_cast<std::uint16_t*>(kMaximumAddress - 1U);
  const auto launch = [&](const std::uint8_t* const w,
                          const std::uint8_t* const s, const float scale,
                          const std::uint16_t* const a,
                          const std::size_t tokens, const std::size_t rows,
                          const std::size_t columns,
                          std::uint16_t* const o) noexcept {
    return q3x::kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
            w, s, scale, a, tokens, rows, columns, o,
            static_cast<void*>(stream));
  };

  cudaGraph_t graph = nullptr;
  bool ready = test.cuda_ok(
      cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal),
      "invalid begin capture");
  std::array<int, 21U> statuses{};
  statuses.fill(static_cast<int>(cudaErrorUnknown));
  if (ready) {
    statuses[0] = launch(nullptr, scales, 1.0F, activations, kTokens, kRows,
                         kColumns, output);
    statuses[1] = launch(weights, nullptr, 1.0F, activations, kTokens, kRows,
                         kColumns, output);
    statuses[2] = launch(weights, scales, 1.0F, nullptr, kTokens, kRows,
                         kColumns, output);
    statuses[3] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                         kColumns, nullptr);
    statuses[4] = launch(weights, scales,
                         std::numeric_limits<float>::quiet_NaN(), activations,
                         kTokens, kRows, kColumns, output);
    statuses[5] = launch(weights, scales, -1.0F, activations, kTokens, kRows,
                         kColumns, output);
    statuses[6] = launch(weights, scales, 1.0F, activations, 128U, kRows,
                         kColumns, output);
    statuses[7] = launch(weights, scales, 1.0F, activations, 513U, kRows,
                         kColumns, output);
    statuses[8] = launch(weights, scales, 1.0F, activations, kTokens,
                         kRows - 1U, kColumns, output);
    statuses[9] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                         kColumns - 16U, output);
    statuses[10] = launch(weights + 1U, scales, 1.0F, activations, kTokens,
                          kRows, kColumns, output);
    statuses[11] = launch(weights, scales + 1U, 1.0F, activations, kTokens,
                          kRows, kColumns, output);
    statuses[12] = launch(
        weights, scales, 1.0F,
        reinterpret_cast<const std::uint16_t*>(kActivationAddress + 2U),
        kTokens, kRows, kColumns, output);
    statuses[13] = launch(
        weights, scales, 1.0F, activations, kTokens, kRows, kColumns,
        reinterpret_cast<std::uint16_t*>(kOutputAddress + 1U));
    statuses[14] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                          kColumns, activation_alias);
    statuses[15] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                          kColumns, packed_alias);
    statuses[16] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                          kColumns, scale_alias);
    statuses[17] = launch(wrapping_weights, scales, 1.0F, activations, kTokens,
                          kRows, kColumns, output);
    statuses[18] = launch(weights, wrapping_scales, 1.0F, activations, kTokens,
                          kRows, kColumns, output);
    statuses[19] = launch(weights, scales, 1.0F, wrapping_activations, kTokens,
                          kRows, kColumns, output);
    statuses[20] = launch(weights, scales, 1.0F, activations, kTokens, kRows,
                          kColumns, wrapping_output);
    ready = test.cuda_ok(cudaStreamEndCapture(stream, &graph),
                         "invalid end capture") &&
            ready;
  }
  std::size_t nodes = 0U;
  if (ready && graph != nullptr) {
    ready = test.cuda_ok(cudaGraphGetNodes(graph, nullptr, &nodes),
                         "invalid count graph nodes") &&
            ready;
  }
  if (graph != nullptr) {
    ready = test.cuda_ok(cudaGraphDestroy(graph), "invalid destroy graph") &&
            ready;
  }
  const std::size_t invalid_count = static_cast<std::size_t>(std::count(
      statuses.begin(), statuses.end(),
      static_cast<int>(cudaErrorInvalidValue)));
  const bool gate = ready && invalid_count == statuses.size() && nodes == 0U;
  std::cout << "L2_APW_INVALID_GRAPH: invalid_statuses=" << invalid_count
            << '/' << statuses.size() << " kernel_nodes=" << nodes
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "invalid calls fail before any kernel enqueue");
  return gate;
}

[[nodiscard]] cudaError_t launch_pair(Fixture& fixture,
                                      const StreamAndEvents& execution) {
  cudaError_t status = cudaEventRecord(execution.ready(), execution.main());
  if (status == cudaSuccess) {
    status = cudaStreamWaitEvent(execution.auxiliary(), execution.ready(), 0U);
  }
  if (status == cudaSuccess) {
    status = static_cast<cudaError_t>(q3x::kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
            fixture.gate_packed.get(), fixture.gate_scales.get(), 1.0F,
            fixture.activations.get(), fixture.token_count, kRows, kColumns,
            fixture.gate_output(),
            static_cast<void*>(execution.main())));
  }
  if (status == cudaSuccess) {
    status = static_cast<cudaError_t>(q3x::kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
            fixture.up_packed.get(), fixture.up_scales.get(), 1.0F,
            fixture.activations.get(), fixture.token_count, kRows, kColumns,
            fixture.up_output(),
            static_cast<void*>(execution.auxiliary())));
  }
  if (status == cudaSuccess) {
    status = cudaEventRecord(execution.done(), execution.auxiliary());
  }
  if (status == cudaSuccess) {
    status = cudaStreamWaitEvent(execution.main(), execution.done(), 0U);
  }
  return status;
}

[[nodiscard]] bool scrub_l2(TestContext& test, Fixture& fixture,
                            const StreamAndEvents& execution,
                            const std::string& label) {
  constexpr unsigned int kThreads = 256U;
  constexpr unsigned int kBlocks = 256U;
  scrub_l2_kernel<<<kBlocks, kThreads, 0U, execution.main()>>>(
      fixture.scrub.get(), kScrubBytes / sizeof(std::uint32_t));
  bool ready = test.cuda_ok(cudaGetLastError(), label + " launch L2 scrub");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " synchronize L2 scrub");
  return ready;
}

[[nodiscard]] bool poison_outputs(TestContext& test, Fixture& fixture,
                                  const StreamAndEvents& execution,
                                  const int byte,
                                  const std::string& label) {
  const std::size_t bytes =
      fixture.guarded_output_elements() * sizeof(std::uint16_t);
  bool ready = test.cuda_ok(
      cudaMemsetAsync(fixture.gate_output_store.get(), byte, bytes,
                      execution.main()),
      label + " poison Gate output");
  ready = ready && test.cuda_ok(
                       cudaMemsetAsync(fixture.up_output_store.get(), byte,
                                       bytes, execution.main()),
                       label + " poison Up output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " poison synchronize");
  return ready;
}

[[nodiscard]] bool copy_outputs(TestContext& test, const Fixture& fixture,
                                const StreamAndEvents& execution,
                                std::vector<std::uint16_t>& gate,
                                std::vector<std::uint16_t>& up,
                                const std::string& label) {
  gate.resize(fixture.guarded_output_elements());
  up.resize(fixture.guarded_output_elements());
  const std::size_t bytes = gate.size() * sizeof(std::uint16_t);
  bool ready = test.cuda_ok(
      cudaMemcpyAsync(gate.data(), fixture.gate_output_store.get(), bytes,
                      cudaMemcpyDeviceToHost, execution.main()),
      label + " copy Gate output");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(up.data(),
                                       fixture.up_output_store.get(), bytes,
                                       cudaMemcpyDeviceToHost,
                                       execution.main()),
                       label + " copy Up output");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " copy output synchronize");
  return ready;
}

[[nodiscard]] bool run_correctness(TestContext& test, Fixture& fixture,
                                   const StreamAndEvents& execution,
                                   PolicyController& policies,
                                   const PersistPolicy policy) {
  std::vector<std::uint16_t> baseline_gate;
  std::vector<std::uint16_t> baseline_up;
  std::vector<std::uint16_t> candidate_gate;
  std::vector<std::uint16_t> candidate_up;
  std::vector<std::uint16_t> replay_gate;
  std::vector<std::uint16_t> replay_up;
  bool ready = policies.apply_baseline("correctness_baseline");
  ready = ready && poison_outputs(test, fixture, execution, 0x3c,
                                  "correctness baseline");
  ready = ready && test.cuda_ok(launch_pair(fixture, execution),
                                "correctness baseline launch pair");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness baseline synchronize");
  ready = ready && copy_outputs(test, fixture, execution, baseline_gate,
                                baseline_up, "correctness baseline");

  ready = ready && policies.apply_candidate(
                       fixture.activations.get(), fixture.token_count, policy,
                       "correctness_candidate");
  ready = ready && poison_outputs(test, fixture, execution, 0xa5,
                                  "correctness candidate");
  ready = ready && test.cuda_ok(launch_pair(fixture, execution),
                                "correctness candidate launch pair");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness candidate synchronize");
  ready = ready && copy_outputs(test, fixture, execution, candidate_gate,
                                candidate_up, "correctness candidate");

  ready = ready && poison_outputs(test, fixture, execution, 0x5a,
                                  "correctness replay");
  ready = ready && test.cuda_ok(launch_pair(fixture, execution),
                                "correctness replay launch pair");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "correctness replay synchronize");
  ready = ready && copy_outputs(test, fixture, execution, replay_gate,
                                replay_up, "correctness replay");

  std::size_t candidate_mismatches = 0U;
  std::size_t replay_mismatches = 0U;
  std::size_t unexpected_nonfinite = 0U;
  bool guards = ready;
  if (ready) {
    for (std::size_t index = 0U; index < fixture.output_elements(); ++index) {
      const std::size_t guarded_index = kGuardElements + index;
      candidate_mismatches +=
          baseline_gate[guarded_index] != candidate_gate[guarded_index] ? 1U
                                                                        : 0U;
      candidate_mismatches +=
          baseline_up[guarded_index] != candidate_up[guarded_index] ? 1U
                                                                    : 0U;
      replay_mismatches +=
          candidate_gate[guarded_index] != replay_gate[guarded_index] ? 1U
                                                                      : 0U;
      replay_mismatches +=
          candidate_up[guarded_index] != replay_up[guarded_index] ? 1U : 0U;
      unexpected_nonfinite +=
          !is_bf16_finite(baseline_gate[guarded_index]) ||
                  !is_bf16_finite(baseline_up[guarded_index]) ||
                  !is_bf16_finite(candidate_gate[guarded_index]) ||
                  !is_bf16_finite(candidate_up[guarded_index])
              ? 1U
              : 0U;
    }
    for (std::size_t index = 0U; index < kGuardElements; ++index) {
      const std::size_t tail = kGuardElements + fixture.output_elements() +
                               index;
      guards = guards && baseline_gate[index] == 0x3c3cU &&
               baseline_gate[tail] == 0x3c3cU &&
               baseline_up[index] == 0x3c3cU &&
               baseline_up[tail] == 0x3c3cU &&
               candidate_gate[index] == 0xa5a5U &&
               candidate_gate[tail] == 0xa5a5U &&
               candidate_up[index] == 0xa5a5U &&
               candidate_up[tail] == 0xa5a5U &&
               replay_gate[index] == 0x5a5aU &&
               replay_gate[tail] == 0x5a5aU && replay_up[index] == 0x5a5aU &&
               replay_up[tail] == 0x5a5aU;
    }
  }

  std::vector<std::uint8_t> gate_packed_after(kPackedBytes);
  std::vector<std::uint8_t> up_packed_after(kPackedBytes);
  std::vector<std::uint8_t> gate_scales_after(kScaleBytes);
  std::vector<std::uint8_t> up_scales_after(kScaleBytes);
  std::vector<std::uint16_t> activations_after(
      fixture.host_activations.size());
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(gate_packed_after.data(),
                                       fixture.gate_packed.get(), kPackedBytes,
                                       cudaMemcpyDeviceToHost,
                                       execution.main()),
                       "copy Gate weights after correctness");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(up_packed_after.data(),
                                       fixture.up_packed.get(), kPackedBytes,
                                       cudaMemcpyDeviceToHost,
                                       execution.main()),
                       "copy Up weights after correctness");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(gate_scales_after.data(),
                                       fixture.gate_scales.get(), kScaleBytes,
                                       cudaMemcpyDeviceToHost,
                                       execution.main()),
                       "copy Gate scales after correctness");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(up_scales_after.data(),
                                       fixture.up_scales.get(), kScaleBytes,
                                       cudaMemcpyDeviceToHost,
                                       execution.main()),
                       "copy Up scales after correctness");
  ready = ready && test.cuda_ok(
                       cudaMemcpyAsync(
                           activations_after.data(), fixture.activations.get(),
                           activations_after.size() * sizeof(std::uint16_t),
                           cudaMemcpyDeviceToHost, execution.main()),
                       "copy activations after correctness");
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                "copy immutable inputs synchronize");
  const bool inputs_preserved =
      ready && gate_packed_after == fixture.host_gate_packed &&
      up_packed_after == fixture.host_up_packed &&
      gate_scales_after == fixture.host_gate_scales &&
      up_scales_after == fixture.host_up_scales &&
      activations_after == fixture.host_activations;
  const bool gate = ready && candidate_mismatches == 0U &&
                    replay_mismatches == 0U &&
                    unexpected_nonfinite == 0U && guards && inputs_preserved;
  std::cout << "L2_APW_CORRECTNESS: tokens=" << fixture.token_count
            << " policy=" << policy_name(policy)
            << " candidate_mismatches=" << candidate_mismatches << '/'
            << 2U * fixture.output_elements()
            << " replay_mismatches=" << replay_mismatches << '/'
            << 2U * fixture.output_elements()
            << " unexpected_nonfinite=" << unexpected_nonfinite
            << " guards=" << (guards ? "intact" : "BAD")
            << " inputs_preserved="
            << (inputs_preserved ? "true" : "false")
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "APW preserves exact Gate/Up results and all inputs");
  return gate;
}

enum class TimedPolicy {
  kBaseline,
  kCandidate,
};

[[nodiscard]] float measure_policy_pass(
    TestContext& test, Fixture& fixture, const StreamAndEvents& execution,
    PolicyController& policies, const TimedPolicy timed_policy,
    const PersistPolicy persist_policy, const std::string& label,
    const int warmups = kWarmups, const int iterations = kIterations,
    const bool profiler_range = false) {
  bool ready = timed_policy == TimedPolicy::kBaseline
                   ? policies.apply_baseline(label)
                   : policies.apply_candidate(fixture.activations.get(),
                                              fixture.token_count,
                                              persist_policy, label);
  ready = ready && scrub_l2(test, fixture, execution, label);
  for (int warmup = 0; ready && warmup < warmups; ++warmup) {
    ready = test.cuda_ok(launch_pair(fixture, execution),
                         label + " warmup pair") &&
            ready;
  }
  ready = ready && test.cuda_ok(cudaStreamSynchronize(execution.main()),
                                label + " warmup synchronize");
  if (ready && profiler_range) {
    ready = test.cuda_ok(cudaProfilerStart(),
                         label + " start profiler range");
  }
  const auto wall_start = std::chrono::steady_clock::now();
  ready = ready && test.cuda_ok(cudaEventRecord(execution.start(),
                                                execution.main()),
                                label + " record start");
  for (int iteration = 0; ready && iteration < iterations; ++iteration) {
    ready = test.cuda_ok(launch_pair(fixture, execution),
                         label + " measured pair") &&
            ready;
  }
  ready = ready && test.cuda_ok(cudaEventRecord(execution.stop(),
                                                execution.main()),
                                label + " record stop");
  ready = ready && test.cuda_ok(cudaEventSynchronize(execution.stop()),
                                label + " synchronize stop");
  const auto wall_stop = std::chrono::steady_clock::now();
  if (profiler_range) {
    ready = test.cuda_ok(cudaProfilerStop(),
                         label + " stop profiler range") &&
            ready;
  }
  float total_ms = std::numeric_limits<float>::quiet_NaN();
  ready = ready && test.cuda_ok(
                       cudaEventElapsedTime(&total_ms, execution.start(),
                                            execution.stop()),
                       label + " elapsed time");
  const double wall_ms =
      std::chrono::duration<double, std::milli>(wall_stop - wall_start)
          .count();
  const float average_ms =
      ready && iterations > 0
          ? total_ms / static_cast<float>(iterations)
          : std::numeric_limits<float>::quiet_NaN();
  std::cout << "L2_APW_PASS: label=" << label
            << " timed_policy="
            << (timed_policy == TimedPolicy::kBaseline ? "baseline"
                                                       : "candidate")
            << " persist_policy=" << policy_name(persist_policy)
            << " tokens=" << fixture.token_count
            << " warmups=" << warmups << " iterations=" << iterations
            << " profiler_range="
            << (profiler_range ? "true" : "false")
            << " average_pair_ms=" << average_ms
            << " host_wall_average_ms="
            << (iterations > 0 ? wall_ms / static_cast<double>(iterations)
                               : std::numeric_limits<double>::quiet_NaN())
            << '\n';
  return average_ms;
}

[[nodiscard]] bool run_screen(TestContext& test, Fixture& fixture,
                              const StreamAndEvents& execution,
                              PolicyController& policies,
                              const PersistPolicy policy) {
  double baseline_sum = 0.0;
  double candidate_sum = 0.0;
  bool every_round = true;
  for (int round = 0; round < kRounds; ++round) {
    const std::string prefix =
        "round_" + std::to_string(round + 1) + '_';
    const float b1 = measure_policy_pass(
        test, fixture, execution, policies, TimedPolicy::kBaseline, policy,
        prefix + "B1");
    const float c1 = measure_policy_pass(
        test, fixture, execution, policies, TimedPolicy::kCandidate, policy,
        prefix + "C1");
    const float c2 = measure_policy_pass(
        test, fixture, execution, policies, TimedPolicy::kCandidate, policy,
        prefix + "C2");
    const float b2 = measure_policy_pass(
        test, fixture, execution, policies, TimedPolicy::kBaseline, policy,
        prefix + "B2");
    const bool finite = std::isfinite(b1) && std::isfinite(c1) &&
                        std::isfinite(c2) && std::isfinite(b2) && b1 > 0.0F &&
                        c1 > 0.0F && c2 > 0.0F && b2 > 0.0F;
    const double speedup =
        finite ? static_cast<double>(b1 + b2) /
                     static_cast<double>(c1 + c2)
               : std::numeric_limits<double>::quiet_NaN();
    every_round = every_round && finite && speedup > 1.0;
    if (finite) {
      baseline_sum += static_cast<double>(b1 + b2);
      candidate_sum += static_cast<double>(c1 + c2);
    }
    std::cout << "PERF_L2_APW_ROUND: tokens=" << fixture.token_count
              << " policy=" << policy_name(policy)
              << " round=" << round + 1 << " order=B-C-C-B"
              << " B1_ms=" << b1 << " C1_ms=" << c1
              << " C2_ms=" << c2 << " B2_ms=" << b2
              << " speedup=" << speedup
              << " strict_positive_gate="
              << (finite && speedup > 1.0 ? "PASS" : "FAIL") << '\n';
  }
  const double baseline_ms = baseline_sum / (2.0 * kRounds);
  const double candidate_ms = candidate_sum / (2.0 * kRounds);
  const double speedup = baseline_sum / candidate_sum;
  const bool gate = fixture.token_count == 512U && every_round &&
                    std::isfinite(speedup) &&
                    speedup >= kRequiredC512Speedup;
  std::cout << "PERF_L2_APW_AGGREGATE: tokens=" << fixture.token_count
            << " policy=" << policy_name(policy)
            << " baseline_pair_ms=" << baseline_ms
            << " candidate_pair_ms=" << candidate_ms
            << " speedup=" << speedup
            << " required_speedup=" << kRequiredC512Speedup
            << " every_round_strict_positive="
            << (every_round ? "true" : "false")
            << " rounds=" << kRounds << " iterations=" << kIterations
            << " measurement_scope=steady_state_same_buffer_no_producer"
            << " eager_direct_launch=true graph_apw_not_tested=true"
            << " gate=" << (gate ? "PASS" : "FAIL") << '\n';
  test.expect(gate, "C512 APW clears frozen pair performance gate");
  return gate;
}

enum class Mode {
  kValidate,
  kScreen,
  kMeasureBaseline,
  kMeasureCandidate,
  kProfileBaseline,
  kProfileCandidate,
};

struct Options {
  Mode mode = Mode::kValidate;
  PersistPolicy policy = PersistPolicy::kBalanced;
  std::size_t token_count = 512U;
};

[[nodiscard]] bool parse_options(const int argc, char** argv,
                                 Options& options) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--mode=validate") {
      options.mode = Mode::kValidate;
    } else if (argument == "--mode=screen") {
      options.mode = Mode::kScreen;
    } else if (argument == "--mode=measure-baseline") {
      options.mode = Mode::kMeasureBaseline;
    } else if (argument == "--mode=measure-candidate") {
      options.mode = Mode::kMeasureCandidate;
    } else if (argument == "--mode=profile-baseline") {
      options.mode = Mode::kProfileBaseline;
    } else if (argument == "--mode=profile-candidate") {
      options.mode = Mode::kProfileCandidate;
    } else if (argument == "--policy=balanced") {
      options.policy = PersistPolicy::kBalanced;
    } else if (argument == "--policy=main-owner") {
      options.policy = PersistPolicy::kMainOwner;
    } else if (argument == "--tokens=256") {
      options.token_count = 256U;
    } else if (argument == "--tokens=512") {
      options.token_count = 512U;
    } else {
      std::cerr << "unknown argument: " << argument << '\n';
      return false;
    }
  }
  if (options.mode == Mode::kScreen && options.token_count != 512U) {
    std::cerr << "screen mode is frozen to --tokens=512\n";
    return false;
  }
  return true;
}

[[nodiscard]] const char* mode_name(const Mode mode) noexcept {
  switch (mode) {
    case Mode::kValidate:
      return "validate";
    case Mode::kScreen:
      return "screen";
    case Mode::kMeasureBaseline:
      return "measure_baseline";
    case Mode::kMeasureCandidate:
      return "measure_candidate";
    case Mode::kProfileBaseline:
      return "profile_baseline";
    case Mode::kProfileCandidate:
      return "profile_candidate";
  }
  return "unknown";
}

}  // namespace

int main(const int argc, char** argv) {
  Options options{};
  if (!parse_options(argc, argv, options)) {
    return 2;
  }
  TestContext test;
  int device_count = 0;
  const cudaError_t count_status = cudaGetDeviceCount(&device_count);
  if (count_status != cudaSuccess || device_count == 0) {
    std::cout << "SKIP: Gate/Up L2 APW screen requires a CUDA device\n";
    (void)cudaGetLastError();
    return 77;
  }
  cudaDeviceProp properties{};
  if (!test.cuda_ok(cudaGetDeviceProperties(&properties, 0),
                    "read CUDA device properties")) {
    return 1;
  }
  if (properties.major != 8 || properties.minor != 7) {
    std::cout << "SKIP: Gate/Up L2 APW screen requires SM87; got sm_"
              << properties.major << properties.minor << '\n';
    return 77;
  }
  const bool capability =
      properties.l2CacheSize >= 4 * 1024 * 1024 &&
      static_cast<std::size_t>(properties.persistingL2CacheMaxSize) >=
          kRequestedPersistingBytes &&
      static_cast<std::size_t>(properties.accessPolicyMaxWindowSize) >=
          options.token_count * kColumns * sizeof(std::uint16_t);
  std::cout << "L2_APW_DEVICE: name=" << properties.name
            << " sm=" << properties.major << properties.minor
            << " sm_count=" << properties.multiProcessorCount
            << " l2_bytes=" << properties.l2CacheSize
            << " maximum_persisting_bytes="
            << properties.persistingL2CacheMaxSize
            << " maximum_window_bytes="
            << properties.accessPolicyMaxWindowSize
            << " mode=" << mode_name(options.mode)
            << " policy=" << policy_name(options.policy)
            << " tokens=" << options.token_count
            << " capability_gate=" << (capability ? "PASS" : "FAIL")
            << '\n';
  if (!capability) {
    std::cout << "SKIP: device lacks the frozen APW capacity\n";
    return 77;
  }

  StreamAndEvents execution;
  if (!execution.create(test)) {
    return 1;
  }
  PolicyController policies(test, execution.main(), execution.auxiliary(),
                            properties);
  if (!policies.capture()) {
    return 1;
  }
  PolicySnapshot before_invalid{};
  PolicySnapshot after_invalid{};
  bool ready = policies.snapshot(before_invalid, "before invalid gate");
  ready = run_resource_gate(test) && ready;
  ready = run_invalid_graph_gate(test, execution.main()) && ready;
  ready = policies.snapshot(after_invalid, "after invalid gate") && ready;
  const bool invalid_policy_untouched =
      ready && policies.apply_count() == 0U &&
      policies.unchanged_from(before_invalid, after_invalid);
  std::cout << "L2_APW_INVALID_POLICY_STATE: policy_apply_count="
            << policies.apply_count()
            << " limit_and_stream_attributes_unchanged="
            << (invalid_policy_untouched ? "true" : "false")
            << " gate=" << (invalid_policy_untouched ? "PASS" : "FAIL")
            << '\n';
  test.expect(invalid_policy_untouched,
              "invalid validation precedes every policy mutation");
  if (!ready || !invalid_policy_untouched) {
    (void)policies.restore();
    return 1;
  }

  Fixture fixture;
  if (!fixture.initialize(test, execution.main(), options.token_count)) {
    (void)policies.restore();
    return 1;
  }

  switch (options.mode) {
    case Mode::kValidate:
      (void)run_correctness(test, fixture, execution, policies,
                            options.policy);
      break;
    case Mode::kScreen:
      if (run_correctness(test, fixture, execution, policies,
                          options.policy)) {
        (void)run_screen(test, fixture, execution, policies, options.policy);
      }
      break;
    case Mode::kMeasureBaseline:
      (void)measure_policy_pass(test, fixture, execution, policies,
                                TimedPolicy::kBaseline, options.policy,
                                "standalone_baseline");
      break;
    case Mode::kMeasureCandidate:
      (void)measure_policy_pass(test, fixture, execution, policies,
                                TimedPolicy::kCandidate, options.policy,
                                "standalone_candidate");
      break;
    case Mode::kProfileBaseline:
    case Mode::kProfileCandidate: {
      const TimedPolicy timed_policy =
          options.mode == Mode::kProfileBaseline ? TimedPolicy::kBaseline
                                                 : TimedPolicy::kCandidate;
      const float milliseconds = measure_policy_pass(
          test, fixture, execution, policies, timed_policy, options.policy,
          "single_pair_profile", 0, 1, true);
      std::cout << "L2_APW_PROFILE_MARKER: mode=" << mode_name(options.mode)
                << " policy=" << policy_name(options.policy)
                << " tokens=" << options.token_count
                << " pair_ms=" << milliseconds
                << " production_kernel_launches=2"
                << " scrub_kernel_launches_outside_range=1"
                << " profiler_range_kernel_launches=2"
                << " ncu_replay_mode=app-range"
                << " ncu_profile_from_start=false"
                << " kernel_replay_forbidden=true\n";
      break;
    }
  }

  const bool restored = policies.restore();
  if (test.failures() != 0 || !restored) {
    std::cerr << test.failures() << " Gate/Up L2 APW assertion(s) failed\n";
    return 1;
  }
  std::cout << "Gate/Up L2 APW SM87 screen passed\n";
  return 0;
}
