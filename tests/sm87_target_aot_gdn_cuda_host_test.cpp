#include "q3x/kernels/sm87_target_aot_gdn_cuda.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace kernels = q3x::kernels;

namespace {

constexpr auto kP40Plan = kernels::sm87_target_aot_gdn_plan(40'000U);

static_assert(kP40Plan.valid());
static_assert(kP40Plan.first_position == 0U);
static_assert(kP40Plan.owner_ctas == 16U);
static_assert(kP40Plan.threads_per_cta == 256U);
static_assert(kP40Plan.warps_per_cta == 8U);
static_assert(kP40Plan.kernel_launches_per_layer == 1U);
static_assert(kP40Plan.value_heads_per_owner == 3U);
static_assert(kP40Plan.private_shared_payload_bytes == 65'920U);
static_assert(kernels::kSm87TargetAotGdnCudaDynamicSharedBytes == 100'252U);
static_assert(kernels::Sm87TargetAotGdnCudaArguments{}.first_position == 0U);
static_assert(std::is_same_v<
              decltype(kernels::Sm87TargetAotGdnCudaArguments{}.
                           interleaved_ab),
              const std::uint16_t*>);
static_assert(std::is_same_v<
              decltype(kernels::Sm87TargetAotGdnCudaArguments{}.
                           final_recurrent_state),
              std::uint16_t*>);

struct TestContext final {
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

template <class T>
[[nodiscard]] T* fake_pointer(const std::uintptr_t address) noexcept {
  return reinterpret_cast<T*>(address);
}

[[nodiscard]] kernels::Sm87TargetAotGdnCudaArguments make_arguments()
    noexcept {
  kernels::Sm87TargetAotGdnCudaArguments arguments;
  arguments.raw_qkvz = fake_pointer<const std::uint16_t>(
      0x0000'0001'0000'0000ULL);
  arguments.interleaved_ab = fake_pointer<const std::uint16_t>(
      0x0000'0001'6000'0000ULL);
  arguments.conv_weight = fake_pointer<const std::uint16_t>(
      0x0000'0001'6100'0000ULL);
  arguments.a_log = fake_pointer<const std::uint16_t>(
      0x0000'0001'6200'0000ULL);
  arguments.dt_bias = fake_pointer<const std::uint16_t>(
      0x0000'0001'6200'1000ULL);
  arguments.norm_weight = fake_pointer<const std::uint16_t>(
      0x0000'0001'6200'2000ULL);
  arguments.l2_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.first_position = 0U;
  arguments.token_count = kernels::kSm87TargetAotGdnCudaTokenCount;
  arguments.output =
      fake_pointer<std::uint16_t>(0x0000'0001'7000'0000ULL);
  arguments.final_conv_history =
      fake_pointer<std::uint16_t>(0x0000'0001'9000'0000ULL);
  arguments.final_recurrent_state =
      fake_pointer<std::uint16_t>(0x0000'0001'9100'0000ULL);
  arguments.cancellation_signal =
      fake_pointer<const std::uint32_t>(0x0000'0001'9200'0000ULL);
  arguments.cuda_stream = fake_pointer<void>(0x0000'0000'0000'0100ULL);
  return arguments;
}

void test_argument_contract(TestContext& test) {
  auto arguments = make_arguments();
  test.expect(kernels::sm87_target_aot_gdn_cuda_arguments_valid(arguments),
              "exact cold P40000 arguments validate");

  auto changed = arguments;
  changed.token_count = 39'984U;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "only the frozen P40000 token count is admitted");
  changed = arguments;
  changed.first_position = 1U;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "continuation is rejected by the cold-only kernel");
  changed = arguments;
  ++changed.l2_epsilon_fp32_bits;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "L2 epsilon raw bits are exact");
  changed = arguments;
  ++changed.norm_epsilon_fp32_bits;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "norm epsilon raw bits are exact");
  changed = arguments;
  changed.cuda_stream = nullptr;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "an owner-bound non-default stream is required");
  changed = arguments;
  changed.interleaved_ab = arguments.raw_qkvz;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "AB producer cannot alias raw QKVZ");
  changed = arguments;
  changed.output = const_cast<std::uint16_t*>(arguments.raw_qkvz);
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "output cannot overwrite a live producer");
  changed = arguments;
  changed.final_recurrent_state = changed.final_conv_history;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "final transaction spans are disjoint");
  changed = arguments;
  changed.norm_weight = fake_pointer<const std::uint16_t>(
      reinterpret_cast<std::uintptr_t>(arguments.norm_weight) + 2U);
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "every bound tensor observes the fixed alignment");
  changed = arguments;
  changed.final_conv_history = nullptr;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "cold execution still requires a final history span");
  changed = arguments;
  changed.final_recurrent_state = nullptr;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "cold execution still requires a final recurrent span");
  changed = arguments;
  changed.cancellation_signal = nullptr;
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "C16 cancellation observation is mandatory");
  changed = arguments;
  changed.cancellation_signal = fake_pointer<const std::uint32_t>(
      reinterpret_cast<std::uintptr_t>(arguments.cancellation_signal) + 2U);
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "mapped cancellation control is uint32 aligned");
  changed = arguments;
  changed.cancellation_signal = reinterpret_cast<const std::uint32_t*>(
      arguments.final_recurrent_state);
  test.expect(!kernels::sm87_target_aot_gdn_cuda_arguments_valid(changed),
              "cancellation word has an independent owner range");

  const auto ab_range = kernels::sm87_target_aot_gdn_cuda_byte_range(
      arguments.interleaved_ab,
      kernels::kSm87TargetAotGdnCudaInterleavedAbBytes);
  test.expect(ab_range.valid &&
                  ab_range.end - ab_range.begin == 7'680'000ULL,
              "AB range covers [T,96], not one [T,48] plane");
}

void test_resource_contract(TestContext& test) {
  kernels::Sm87TargetAotGdnCudaResources unavailable;
  unavailable.token_count = kernels::kSm87TargetAotGdnCudaTokenCount;
  test.expect(
      kernels::sm87_target_aot_gdn_cuda_resources_structurally_valid(
          unavailable),
      "a default-off uncompiled resource record remains fail closed");

  kernels::Sm87TargetAotGdnCudaResources compiled;
  compiled.token_count = kernels::kSm87TargetAotGdnCudaTokenCount;
  compiled.binary_version = 87;
  compiled.registers_per_thread = 224;
  compiled.static_shared_bytes = 0U;
  compiled.dynamic_shared_bytes =
      kernels::kSm87TargetAotGdnCudaDynamicSharedBytes;
  compiled.local_bytes = 64U;
  compiled.maximum_threads_per_block = 256;
  compiled.active_blocks_per_sm = 1;
  compiled.physical_grid_ctas = 16;
  compiled.kernel_compiled = true;
  compiled.exact_owner_geometry = true;
  test.expect(
      kernels::sm87_target_aot_gdn_cuda_resources_structurally_valid(
          compiled),
      "compiled resources bind one owner CTA per SM structurally");

  auto changed = compiled;
  changed.active_blocks_per_sm = 2;
  test.expect(
      !kernels::sm87_target_aot_gdn_cuda_resources_structurally_valid(
          changed),
      "resource record rejects a geometry inconsistent with 100 KiB shared");
  changed = compiled;
  changed.physical_grid_ctas = 48;
  test.expect(
      !kernels::sm87_target_aot_gdn_cuda_resources_structurally_valid(
          changed),
      "resource record rejects value-head rather than QK-owner topology");
  changed = compiled;
  changed.numerical_contract_qualified = true;
  test.expect(
      !kernels::sm87_target_aot_gdn_cuda_resources_structurally_valid(
          changed),
      "T0 resources cannot forge numerical qualification");
  changed = compiled;
  changed.production_dispatch_eligible = true;
  test.expect(
      !kernels::sm87_target_aot_gdn_cuda_resources_structurally_valid(
          changed),
      "T0 resources cannot forge production eligibility");
}

}  // namespace

int main() {
  TestContext test;
  test_argument_contract(test);
  test_resource_contract(test);
  if (test.failures() != 0) {
    return 1;
  }
  std::cout << "SM87 target-AOT exact P40000 GDN CUDA host contract checks "
               "passed\n";
  return 0;
}
