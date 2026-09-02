#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Status returned by the host reference implementations. CUDA launch helpers
// return cudaError_t as int so this header remains usable without CUDA headers.
enum class GemvStatus : std::uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kInvalidColumnCount,
  kSizeOverflow,
};

[[nodiscard]] const char* gemv_status_string(GemvStatus status) noexcept;

// Batch-one row-major GEMV correctness references. BF16 values are passed as
// their raw IEEE bfloat16 bit patterns. Every implementation accumulates in
// FP32 and writes one FP32 value per row.
//
// Empty shapes are successful no-ops and may use null pointers. Non-empty
// shapes require all relevant pointers. rows * columns must fit size_t.
[[nodiscard]] GemvStatus bf16_gemv_reference_cpu(
    const std::uint16_t* weights, const std::uint16_t* activation,
    std::size_t rows, std::size_t columns, float* output) noexcept;

// Canonical ModelOpt FP8 layout: weights is E4M3FN [rows, columns], and
// weight_scale is the finite, non-negative per-tensor multiplier:
//
//   output[row] = sum_k activation[k] * E4M3FN(weight[row,k]) * weight_scale
[[nodiscard]] GemvStatus fp8_gemv_reference_cpu(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    float* output) noexcept;

// Canonical ModelOpt static W8A8 path. Before the dot product, each BF16
// activation is divided by input_scale, saturated to finite E4M3FN, rounded
// to nearest-even, and expanded back to FP32 with input_scale. This mirrors
// ModelOpt quant_algo="FP8" rather than the W8A16 diagnostic helper above.
[[nodiscard]] GemvStatus fp8_static_gemv_reference_cpu(
    const std::uint8_t* weights, float weight_scale, float input_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    float* output) noexcept;

// Canonical ModelOpt NVFP4 layout before backend repacking. packed_weights is
// [rows, columns / 2], block_scales is E4M3FN [rows, columns / 16], and
// weight_scale_2 is a finite, non-negative per-tensor multiplier. Even K uses
// the low nibble and odd K uses the high nibble. columns must be a multiple of
// 16 (including for empty shapes).
[[nodiscard]] GemvStatus nvfp4_gemv_reference_cpu(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, float* output) noexcept;

// Asynchronous CUDA correctness references. Pointer arguments must reference
// device-accessible storage; no allocation or synchronization is performed.
// cuda_stream is a cudaStream_t represented as void*, with nullptr selecting
// the legacy default stream. Invalid host-visible arguments return
// cudaErrorInvalidValue. The launch helpers clear an unrelated stale CUDA
// last-error immediately before launching and return the status of their own
// launch boundary.
[[nodiscard]] int launch_bf16_gemv_reference_cuda(
    const std::uint16_t* weights, const std::uint16_t* activation,
    std::size_t rows, std::size_t columns, float* output,
    void* cuda_stream = nullptr) noexcept;

// Single-token BF16 projection with direct BF16 output. This preserves the
// FP32 FMA and shared-memory reduction order of
// launch_bf16_gemv_reference_cuda, then rounds the completed row result to
// BF16 RNE. Empty shapes are successful no-ops and may use null pointers. For
// a non-empty shape, output must be disjoint from both read-only input ranges;
// weights and activation may alias one another.
[[nodiscard]] int launch_bf16_gemv_bf16_cuda(
    const std::uint16_t* weights, const std::uint16_t* activation,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fused pair of BF16 projection tiles. Both weights are row-major
// [rows, columns], input is token-major [token_count, columns], and each
// output is token-major [token_count, rows]. token_count must be in [1, 16].
// Every CTA computes exactly one (projection, token, row) dot product using
// the same FP32 FMA and shared-memory reduction order as the scalar reference
// above, then rounds the result directly to BF16 RNE. With a valid token
// count, an empty shape is a successful no-op and may use null pointers. For
// a non-empty shape, the two outputs must be disjoint from one another and
// from every input range; the read-only weights and input may alias.
[[nodiscard]] int launch_bf16_gemv_pair_tile_bf16_cuda(
    const std::uint16_t* first_weights,
    const std::uint16_t* second_weights,
    const std::uint16_t* input, std::size_t token_count, std::size_t rows,
    std::size_t columns, std::uint16_t* first_output,
    std::uint16_t* second_output, void* cuda_stream = nullptr) noexcept;

// Exact production M16/N48/K5120 projection pair. One 256-thread CTA owns both
// projections for a row, so every token activation is decoded once and fed to
// independent A/B FP32 FMA chains. Each chain preserves the generic pair's
// column order and 256-thread shared-memory reduction tree before BF16 RNE.
[[nodiscard]] int launch_bf16_gemv_pair_m16_projection_fused_cuda(
    const std::uint16_t* first_weights,
    const std::uint16_t* second_weights, const std::uint16_t* input,
    std::uint16_t* first_output, std::uint16_t* second_output,
    void* cuda_stream = nullptr) noexcept;

// Exact P40000/N48/K5120 projection pair. This is the same per-row, per-M16
// arithmetic body as launch_bf16_gemv_pair_m16_projection_fused_cuda, with
// all 2,500 independent M16 tiles submitted in one two-dimensional grid.
// Consequently each output preserves the Legacy scalar-FMA column order,
// 256-thread FP32 reduction tree, and BF16 RNE without 2,500 host launches.
[[nodiscard]] int launch_bf16_gemv_pair_p40000_projection_fused_cuda(
    const std::uint16_t* first_weights,
    const std::uint16_t* second_weights, const std::uint16_t* input,
    std::size_t token_count, std::uint16_t* first_output,
    std::uint16_t* second_output, void* cuda_stream = nullptr) noexcept;

// Kernel resource probes used by the exact-route and regression gates.
// active_blocks_per_sm uses a 256-thread block and zero dynamic shared memory;
// all destinations are required.
[[nodiscard]] int query_bf16_gemv_pair_tile_bf16_cuda_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* active_blocks_per_sm) noexcept;
[[nodiscard]] int query_bf16_gemv_pair_m16_projection_fused_cuda_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* active_blocks_per_sm) noexcept;

[[nodiscard]] int launch_fp8_gemv_reference_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    float* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_fp8_static_gemv_reference_cuda(
    const std::uint8_t* weights, float weight_scale, float input_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    float* output, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_nvfp4_gemv_reference_cuda(
    const std::uint8_t* packed_weights, const std::uint8_t* block_scales,
    float weight_scale_2, const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, float* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
