#ifndef Q3X_KERNELS_SM87_NVFP4_DEQUANT_CUBLAS_GATE_UP_H
#define Q3X_KERNELS_SM87_NVFP4_DEQUANT_CUBLAS_GATE_UP_H

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// On-the-fly NVFP4 dequant + cuBLAS GEMM + SiLU-mul for the Gate/Up MLP.
// Dequantizes gate and up weights into a merged [2N,K] BF16 buffer,
// runs M-chunked cuBLAS GEMM, then applies SiLU-mul activation.
// Peak transient: 358MB (dequant buffer) + 558MB (GEMM temp) = 916MB.
//
// Parameters:
//   gate_packed:  [N, K/2] U8 packed NVFP4 (2 FP4 E2M1 per byte)
//   gate_scale:   [N, K/16] U8 block scales (FP8 E4M3)
//   gate_ws2:     scalar weight_scale_2 for gate
//   up_packed:    [N, K/2] U8 packed NVFP4
//   up_scale:     [N, K/16] U8 block scales
//   up_ws2:       scalar weight_scale_2 for up
//   input:        [M, K] BF16 row-major
//   activated:    [M, N] BF16 row-major (output: silu(gate)*up)
//   m, n, k:      dimensions
//   stream:       CUDA stream
int launch_nvfp4_dequant_cublas_gate_up(
    const std::uint8_t* const gate_packed,
    const std::uint8_t* const gate_scale,
    const float gate_ws2,
    const std::uint8_t* const up_packed,
    const std::uint8_t* const up_scale,
    const float up_ws2,
    const std::uint16_t* const input,
    std::uint16_t* const activated,
    const std::size_t m, const std::size_t n, const std::size_t k,
    void* const cuda_stream) noexcept;

}  // namespace q3x::kernels

#endif  // Q3X_KERNELS_SM87_NVFP4_DEQUANT_CUBLAS_GATE_UP_H
