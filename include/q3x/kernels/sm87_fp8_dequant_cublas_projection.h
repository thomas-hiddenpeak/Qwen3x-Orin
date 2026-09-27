#ifndef Q3X_KERNELS_SM87_FP8_DEQUANT_CUBLAS_PROJECTION_H_
#define Q3X_KERNELS_SM87_FP8_DEQUANT_CUBLAS_PROJECTION_H_

#include <cstddef>
#include <cstdint>

// FP8 dequantization plus CUTLASS BF16 GEMM with FP32 accumulation.
// The historical function name is retained for source/ABI compatibility;
// neither entry calls cuBLAS. The workspace entry consumes request-owned
// scratch instead of the legacy process-global allocation.
namespace q3x::kernels {

// out[M,N] = input[M,K] * dequant(weight[N,K])^T, all BF16, FP32 accumulate.
// weight is canonical row-major [N,K] FP8 E4M3; weight_scale is the per-tensor
// scalar. Returns a cudaError_t code (cudaSuccess on success).
int launch_fp8_dequant_cublas_projection(
    const std::uint8_t* const fp8_weight, const float weight_scale,
    const std::uint16_t* const input_bf16, std::uint16_t* const output_bf16,
    const std::size_t m, const std::size_t n, const std::size_t k,
    void* const cuda_stream) noexcept;

// Allocation-free entry. The caller owns aligned scratch until stream completion.
int launch_fp8_dequant_cublas_projection_with_workspace(
    const std::uint8_t* const fp8_weight, const float weight_scale,
    const std::uint16_t* const input_bf16, std::uint16_t* const output_bf16,
    const std::size_t m, const std::size_t n, const std::size_t k,
    void* const workspace, std::size_t workspace_bytes,
    void* const cuda_stream) noexcept;

}  // namespace q3x::kernels

#endif  // Q3X_KERNELS_SM87_FP8_DEQUANT_CUBLAS_PROJECTION_H_
