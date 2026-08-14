#include "q3x/kernels/sm87_macrofeed_v3_gdn_p40.h"

#include <cstddef>
#include <cstdint>
#include <iostream>

namespace {

using q3x::kernels::Sm87MacrofeedV3GdnKernelResources;
using q3x::kernels::Sm87MacrofeedV3GdnP40Arguments;
using q3x::kernels::Sm87MacrofeedV3GdnResources;

struct TestContext final {
  int failures = 0;

  void expect(const bool condition, const char* const message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

template <class T>
[[nodiscard]] T* pointer(const std::uintptr_t address) noexcept {
  return reinterpret_cast<T*>(address);
}

[[nodiscard]] Sm87MacrofeedV3GdnP40Arguments passing_arguments() noexcept {
  Sm87MacrofeedV3GdnP40Arguments arguments;
  // Deliberately sparse synthetic addresses exercise only pure-host range
  // validation; no address is dereferenced by this test.
  arguments.raw_qkv = pointer<const std::uint16_t>(0x1'0000'0000ULL);
  arguments.a = pointer<const std::uint16_t>(0x1'4000'0000ULL);
  arguments.b = pointer<const std::uint16_t>(0x1'4100'0000ULL);
  arguments.conv_weight =
      pointer<const std::uint16_t>(0x1'4200'0000ULL);
  arguments.a_log = pointer<const std::uint16_t>(0x1'4300'0000ULL);
  arguments.dt_bias = pointer<const std::uint16_t>(0x1'4301'0000ULL);
  arguments.norm_weight =
      pointer<const std::uint16_t>(0x1'4302'0000ULL);
  arguments.z = pointer<const std::uint16_t>(0x1'5000'0000ULL);
  arguments.conv_history = pointer<std::uint16_t>(0x1'9000'0000ULL);
  arguments.recurrent_state = pointer<std::uint16_t>(0x1'9100'0000ULL);
  arguments.conv_qkv = pointer<std::uint16_t>(0x2'0000'0000ULL);
  arguments.output = pointer<std::uint16_t>(0x2'4000'0000ULL);
  arguments.cancellation_signal = pointer<const std::uint32_t>(0x2'8000'0000ULL);
  arguments.l2_epsilon_fp32_bits =
      q3x::kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.norm_epsilon_fp32_bits =
      q3x::kernels::kSm87TargetAotGdnEpsilonFp32Bits;
  arguments.cuda_stream = pointer<void>(0x2'8001'0000ULL);
  return arguments;
}

[[nodiscard]] Sm87MacrofeedV3GdnKernelResources passing_kernel(
    const int ctas, const int active) noexcept {
  Sm87MacrofeedV3GdnKernelResources resources;
  resources.registers_per_thread = 64;
  resources.static_shared_bytes = 0U;
  resources.local_bytes = 0U;
  resources.maximum_threads_per_block = 1'024;
  resources.active_blocks_per_sm = active;
  resources.threads_per_block = 256;
  resources.physical_grid_ctas = ctas;
  return resources;
}

[[nodiscard]] Sm87MacrofeedV3GdnResources passing_resources() noexcept {
  Sm87MacrofeedV3GdnResources resources;
  resources.binary_version = 87;
  resources.convolution = passing_kernel(40, 4);
  resources.recurrence_epilogue = passing_kernel(48, 3);
  resources.recurrence_epilogue.static_shared_bytes = 34'316U;
  resources.kernels_compiled = true;
  resources.exact_geometry = true;
  resources.resource_gate_passed = true;
  return resources;
}

}  // namespace

int main() {
  using namespace q3x::kernels;
  TestContext test;

  static_assert(kSm87MacrofeedV3GdnP40Tokens == 40'000U);
  static_assert(kSm87MacrofeedV3GdnMacrochunkTokens == 5'000U);
  static_assert(kSm87MacrofeedV3GdnPhysicalKernels == 9U);
  static_assert(kSm87MacrofeedV3GdnRawQkvBytes == 819'200'000ULL);
  static_assert(kSm87MacrofeedV3GdnOutputBytes == 491'520'000ULL);

  const Sm87MacrofeedV3GdnP40Arguments baseline = passing_arguments();
  test.expect(sm87_macrofeed_v3_gdn_p40_arguments_valid(baseline),
              "the exact disjoint P40 binding is admitted");

  Sm87MacrofeedV3GdnP40Arguments changed = baseline;
  changed.conv_qkv = const_cast<std::uint16_t*>(baseline.raw_qkv);
  test.expect(!sm87_macrofeed_v3_gdn_p40_arguments_valid(changed),
              "in-place P40 convolution is rejected for independent raw reads");
  changed = baseline;
  changed.output = const_cast<std::uint16_t*>(baseline.z);
  test.expect(!sm87_macrofeed_v3_gdn_p40_arguments_valid(changed),
              "the gate and final output cannot alias");
  changed = baseline;
  changed.cancellation_signal = pointer<const std::uint32_t>(
      reinterpret_cast<std::uintptr_t>(baseline.cancellation_signal) + 1U);
  test.expect(!sm87_macrofeed_v3_gdn_p40_arguments_valid(changed),
              "a misaligned cancellation signal is rejected");
  changed = baseline;
  changed.cancellation_signal = reinterpret_cast<const std::uint32_t*>(
      baseline.raw_qkv);
  test.expect(!sm87_macrofeed_v3_gdn_p40_arguments_valid(changed),
              "the cancellation word cannot alias a numerical operand");
  changed = baseline;
  changed.l2_epsilon_fp32_bits = 0U;
  test.expect(!sm87_macrofeed_v3_gdn_p40_arguments_valid(changed),
              "the exact L2 epsilon identity is mandatory");
  changed = baseline;
  changed.cuda_stream = nullptr;
  test.expect(!sm87_macrofeed_v3_gdn_p40_arguments_valid(changed),
              "the admitted route requires an owned nonblocking stream");

  const Sm87MacrofeedV3GdnResources resources = passing_resources();
  test.expect(sm87_macrofeed_v3_gdn_resources_valid(resources),
              "the exact SM87 resource envelope passes");
  Sm87MacrofeedV3GdnResources changed_resources = resources;
  changed_resources.recurrence_epilogue.local_bytes = 4U;
  test.expect(!sm87_macrofeed_v3_gdn_resources_valid(changed_resources),
              "any recurrence spill fails closed");
  changed_resources = resources;
  changed_resources.recurrence_epilogue.active_blocks_per_sm = 2;
  test.expect(!sm87_macrofeed_v3_gdn_resources_valid(changed_resources),
              "the 48-head recurrence must close in one three-CTA wave");
  changed_resources = resources;
  changed_resources.numerical_contract_qualified = true;
  test.expect(!sm87_macrofeed_v3_gdn_resources_valid(changed_resources),
              "a constituent resource query cannot claim numerical qualification");
  changed_resources = resources;
  changed_resources.production_dispatch_eligible = true;
  test.expect(!sm87_macrofeed_v3_gdn_resources_valid(changed_resources),
              "the standalone slice cannot claim production dispatch");

  if (test.failures == 0) {
    std::cout << "sm87 MacroFeed v3 exact-GDN P40 host contract passed\n";
  }
  return test.failures == 0 ? 0 : 1;
}
