#include "q3x/kernels/sm87_nvfp4_a8w4_gate_admission.h"

#include <cuda_runtime.h>

#include <cstddef>

int main() {
  namespace kernels = q3x::kernels;
  static_assert(kernels::kSm87NvFp4A8W4GateAdmissionTokens == 512U);
  static_assert(kernels::kSm87NvFp4A8W4GateAdmissionRows == 17'408U);
  static_assert(kernels::kSm87NvFp4A8W4GateAdmissionColumns == 5'120U);

  constexpr bool kAdmissionDisabled = false;
  const int expected = static_cast<int>(cudaErrorInvalidValue);
  if (kernels::prepare_sm87_nvfp4_a8w4_gate_m512_admission_cuda(
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, 0U,
          kAdmissionDisabled) != expected) {
    return 1;
  }
  if (kernels::launch_sm87_dynamic_a8_gate_m512_admission_cuda(
          nullptr, nullptr, nullptr, nullptr, kAdmissionDisabled) != expected) {
    return 2;
  }
  if (kernels::launch_sm87_nvfp4_a8w4_gate_m512_admission_cuda(
          nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
          kAdmissionDisabled) != expected) {
    return 3;
  }
  if (kernels::query_sm87_nvfp4_a8w4_gate_m512_admission_resources_cuda(
          nullptr, kAdmissionDisabled) != expected) {
    return 4;
  }
  return 0;
}
