#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

// Compile-only sentinel.  CMake emits PTX for compute_87 and rejects the
// build unless this kernel contains the exact native S4 x S4 instruction.
// This target is never timed and is not registered as a GPU test.
extern "C" __global__ void q3x_sm87_a4w4_mma_ptx_sentinel(
    const std::uint8_t* const packed_a,
    const std::uint8_t* const packed_b,
    std::int32_t* const exported_accumulators) {
  const unsigned int lane = threadIdx.x;
  if (lane >= q3x::kernels::kSm87A4W4WarpThreads) {
    return;
  }
  const q3x::kernels::Sm87A4W4AFragment a =
      q3x::kernels::sm87_a4w4_load_a_fragment(
          packed_a, q3x::kernels::kSm87A4W4MmaK / 2U, 0U, lane);
  const q3x::kernels::Sm87A4W4BFragment b =
      q3x::kernels::sm87_a4w4_load_b_fragment(
          packed_b, q3x::kernels::kSm87A4W4MmaK / 2U, 0U, lane);
  q3x::kernels::Sm87A4W4Accumulator accumulator{};
  q3x::kernels::sm87_a4w4_mma_m16n8k64(accumulator, a, b);
  exported_accumulators[lane * 4U + 0U] = accumulator.x0;
  exported_accumulators[lane * 4U + 1U] = accumulator.x1;
  exported_accumulators[lane * 4U + 2U] = accumulator.x2;
  exported_accumulators[lane * 4U + 3U] = accumulator.x3;
}
