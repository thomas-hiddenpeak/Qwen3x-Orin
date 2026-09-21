#ifndef Q3X_KERNELS_SM87_FP8_DEQUANT_CUBLAS_PROJECTION_H_
#define Q3X_KERNELS_SM87_FP8_DEQUANT_CUBLAS_PROJECTION_H_

#include <cstddef>
#include <cstdint>

// On-the-fly FP8(E4M3, per-tensor scale) -> BF16 dequant + cuBLAS BF16 GEMM
// for the exact-P40000 whole-core fat-N FP8 projections (input_size == 5120:
// GDN in_proj_qkv/z, full-attn q/k/v). The dequant result is written into a
// process-lifetime reused BF16 buffer (peak = largest fat-N weight, ~126 MB),
// NOT a per-projection cache, so device memory stays bounded. The skinny-N
// projections (input_size == 6144: GDN out, full-attn o) are deliberately NOT
// routed here (cuBLAS is not faster than Marlin on those shapes).
namespace q3x::kernels {

// out[M,N] = input[M,K] * dequant(weight[N,K])^T, all BF16, FP32 accumulate.
// weight is canonical row-major [N,K] FP8 E4M3; weight_scale is the per-tensor
// scalar. Returns a cudaError_t code (cudaSuccess on success).
int launch_fp8_dequant_cublas_projection(
    const std::uint8_t* const fp8_weight, const float weight_scale,
    const std::uint16_t* const input_bf16, std::uint16_t* const output_bf16,
    const std::size_t m, const std::size_t n, const std::size_t k,
    void* const cuda_stream) noexcept;

}  // namespace q3x::kernels

#endif  // Q3X_KERNELS_SM87_FP8_DEQUANT_CUBLAS_PROJECTION_H_
