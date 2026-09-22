#ifndef Q3X_KERNELS_SM87_NVFP4_DEQUANT_CUTLASS_GATE_UP_H_
#define Q3X_KERNELS_SM87_NVFP4_DEQUANT_CUTLASS_GATE_UP_H_

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// NVFP4 dequant (vec16) + CUTLASS 2.x GEMM (128x256x64, 27.1 TF vs cuBLAS 25.4 TF)
// + SiLU-mul for the gate/up MLP projection.
// gate_packed/up_packed: [n, k/2] u8 (NVFP4), gate_scale/up_scale: [n, k/16] u8 (E4M3),
// input: [m, k] BF16 (row-major), activated: [m, n] BF16 (row-major).
// Returns 0 on success, cudaError_t otherwise.
int launch_nvfp4_dequant_cutlass_gate_up(
    const std::uint8_t* const gate_packed,
    const std::uint8_t* const gate_scale, const float gate_ws2,
    const std::uint8_t* const up_packed, const std::uint8_t* const up_scale,
    const float up_ws2, const std::uint16_t* const input,
    std::uint16_t* const activated, const std::size_t m, const std::size_t n,
    const std::size_t k, void* const cuda_stream) noexcept;

}  // namespace q3x::kernels

#endif  // Q3X_KERNELS_SM87_NVFP4_DEQUANT_CUTLASS_GATE_UP_H_
