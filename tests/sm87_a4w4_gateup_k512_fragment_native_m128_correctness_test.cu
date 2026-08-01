#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m128.h"

// Reuse the already-audited v2 payload constructor and scalar K512 oracle.
// Only the launcher/resource surface and case matrix are replaced here.
#define Sm87A4W4GateUpK512FragmentNativeResources \
  Sm87A4W4GateUpK512FragmentNativeM128Resources
#define query_sm87_a4w4_gateup_k512_fragment_native_resources_cuda \
  query_sm87_a4w4_gateup_k512_fragment_native_m128_resources_cuda
#define launch_sm87_a4w4_gateup_k512_fragment_native_test_bf16_cuda \
  launch_sm87_a4w4_gateup_k512_fragment_native_m128_test_bf16_cuda
#define main q3x_imported_m64_fragment_native_main
#include "sm87_a4w4_gateup_k512_fragment_native_correctness_test.cu"
#undef main
#undef launch_sm87_a4w4_gateup_k512_fragment_native_test_bf16_cuda
#undef query_sm87_a4w4_gateup_k512_fragment_native_resources_cuda
#undef Sm87A4W4GateUpK512FragmentNativeResources

int main() {
  const int target = target_status();
  if (target != 0) {
    return target;
  }
  kernels::Sm87A4W4GateUpK512FragmentNativeM128Resources resources{};
  const int resource_status =
      kernels::
          query_sm87_a4w4_gateup_k512_fragment_native_m128_resources_cuda(
              &resources);
  if (!cuda_ok(static_cast<cudaError_t>(resource_status),
               "M128N32 resource query")) {
    return 1;
  }
  std::cout << "resources: regs=" << resources.registers_per_thread
            << " local=" << resources.local_bytes
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  if (resources.registers_per_thread > 128 ||
      resources.local_bytes != 0U ||
      resources.dynamic_shared_bytes != 24'576U ||
      resources.active_blocks_per_sm < 2) {
    std::cerr << "M128N32 resource hard gate failed\n";
    return 1;
  }
  return run_case("M128_N32_K512", 128U, 64U, 0U, 32U, 512U) &&
                 run_case("M256_N32_window_K1024", 256U, 64U, 32U,
                          32U, 1'024U) &&
                 run_case("M1920_N96_persistent_K512", 1'920U, 128U,
                          32U, 96U, 512U) &&
                 run_case("M128_N32_model_K5120", 128U, 64U, 0U,
                          32U, 5'120U)
             ? 0
             : 1;
}
