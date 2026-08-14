#include "sm87_target_aot_bf16_ab_launch_internal.h"

#include <iostream>
#include <type_traits>

namespace detail =
    q3x::kernels::sm87_target_aot_bf16_ab_execution_detail;

static_assert(detail::interleaved_p40_prevalidated_contract().valid());
static_assert(!std::is_default_constructible_v<
              detail::SealedInterleavedP40Access>);

int main() {
  constexpr auto contract = detail::interleaved_p40_prevalidated_contract();
  auto wrong = contract;
  ++wrong.hot_static_cuda_queries;
  if (wrong.valid()) {
    std::cerr << "request-time static CUDA query escaped startup seal\n";
    return 1;
  }
  wrong = contract;
  wrong.startup_complete_device_ranges_per_binding = 1U;
  if (wrong.valid()) {
    std::cerr << "pointer-byte checks replaced complete allocation ranges\n";
    return 1;
  }
  wrong = contract;
  wrong.startup_exact_layer_order_gate = false;
  if (wrong.valid()) {
    std::cerr << "the 48-layer binding order was not sealed\n";
    return 1;
  }
  std::cout << "SM87 BF16 A/B P40 startup-seal/prevalidated contract passed\n";
  return 0;
}
