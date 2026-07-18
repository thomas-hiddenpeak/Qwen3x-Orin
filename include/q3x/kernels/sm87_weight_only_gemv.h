#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// SM87 batch-one weight-only GEMV prototypes. Both entry points consume the
// canonical row-major checkpoint layout, accumulate in FP32, round the final
// row result to BF16 RNE, and write the raw BF16 bits to output.
//
// The launch is asynchronous on cuda_stream and performs no allocation,
// copying, or synchronization. cuda_stream is a cudaStream_t represented as
// void*, with nullptr selecting the legacy default stream. Empty shapes are
// successful no-ops. Invalid dimensions, scales, pointers, or output/input
// overlap return cudaErrorInvalidValue represented as int.
//
// These functions require an sm_87 CUDA image. Production callers reach them
// only through an explicitly selected ProjectionBackend::kSm87WeightOnly;
// the default projection policy remains the correctness reference.
// Their parallel reduction and final global-scale order need not be bitwise
// identical to the scalar-shaped CUDA reference, so callers must use the
// documented numerical and end-to-end gates rather than compare FP32 scratch.

// ModelOpt FP8 W8A16:
//
//   output[n] = BF16(sum_k BF16(input[k]) *
//                    E4M3FN(weight[n,k]) * weight_scale)
//
// input_scale is intentionally absent: the activation remains BF16 on Orin.
[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_bf16_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// ModelOpt NVFP4 W4A16 canonical layout. packed_weights is [rows, columns/2],
// block_scales is E4M3FN [rows, columns/16], low nibble precedes high nibble,
// and columns must be a multiple of 16:
//
//   output[n] = BF16(sum_k BF16(input[k]) * E2M1(weight[n,k]) *
//                    E4M3FN(block_scale[n,k/16]) * weight_scale_2)
//
// input_scale is intentionally absent: the activation remains BF16 on Orin.
[[nodiscard]] int launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
