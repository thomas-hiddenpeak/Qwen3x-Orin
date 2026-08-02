#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta.h"

// Reuse the audited v2 publication constructor and scalar K512 oracle.
#define Sm87A4W4GateUpK512FragmentNativeResources \
  Sm87A4W4GateUpK512FragmentNativeM64N1281CtaResources
#define query_sm87_a4w4_gateup_k512_fragment_native_resources_cuda \
  query_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_resources_cuda
#define launch_sm87_a4w4_gateup_k512_fragment_native_test_bf16_cuda \
  launch_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_test_bf16_cuda
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
  kernels::Sm87A4W4GateUpK512FragmentNativeM64N1281CtaResources
      resources{};
  const int resource_status =
      kernels::
          query_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_resources_cuda(
              &resources);
  if (!cuda_ok(static_cast<cudaError_t>(resource_status),
               "M64N128/1CTA resource query")) {
    return 1;
  }
  std::cout << "resources: regs=" << resources.registers_per_thread
            << " local=" << resources.local_bytes
            << " dynamic_shared=" << resources.dynamic_shared_bytes
            << " active_blocks_per_sm=" << resources.active_blocks_per_sm
            << '\n';
  if (resources.registers_per_thread > 128 ||
      resources.local_bytes != 0U ||
      resources.static_shared_bytes != 0U ||
      resources.dynamic_shared_bytes != 12'288U ||
      resources.maximum_threads_per_block < 512 ||
      resources.active_blocks_per_sm < 1) {
    std::cerr << "M64N128/1CTA resource hard gate failed\n";
    return 1;
  }
  return run_case("M64_N128_K512", 64U, 128U, 0U, 128U, 512U) &&
                 run_case("M128_N128_window_K1024", 128U, 256U, 128U,
                          128U, 1'024U) &&
                 run_case("M1088_N128_persistent_K512", 1'088U, 128U,
                          0U, 128U, 512U) &&
                 run_case("M64_N128_model_K5120", 64U, 128U, 0U,
                          128U, 5'120U)
             ? 0
             : 1;
}
