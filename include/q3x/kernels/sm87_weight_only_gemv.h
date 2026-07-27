#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// SM87 single-request weight-only projection kernels. Unless an exact derived
// layout is documented explicitly, every entry point consumes the canonical
// row-major checkpoint layout, accumulates in FP32, rounds each final row
// result to BF16 RNE, and writes the raw BF16 bits to output.
//
// The launch is asynchronous on cuda_stream and performs no allocation,
// copying, or synchronization. cuda_stream is a cudaStream_t represented as
// void*, with nullptr selecting the legacy default stream. Except for
// exact-only entry points documented below, empty shapes are successful
// no-ops. Invalid dimensions, scales, pointers, or output/input overlap return
// cudaErrorInvalidValue represented as int.
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

// Exact M=1 output projection over the derived AoSoA4 FP8 layout for
// [rows=5120, columns=6144]. For each adjacent row quad and packed four-column
// word, sidecar_weights stores one uint4 whose x/y/z/w components correspond
// to rows 0/1/2/3. Every byte is pre-swizzled as code ^ (code >> 5). The
// sidecar therefore has the same 31,457,280-byte extent as the canonical
// matrix but is not a canonical checkpoint tensor.
//
// sidecar_weights requires 16-byte alignment, activation 8-byte alignment,
// and output 2-byte alignment. Their complete spans must be pairwise
// disjoint. Only the exact shape is accepted; invalid arguments return
// cudaErrorInvalidValue before work is enqueued. The launch uses a fixed
// 1024-CTA grid and is asynchronous on cuda_stream.
[[nodiscard]] int
launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_bf16_cuda(
    const std::uint8_t* sidecar_weights, float weight_scale,
    const std::uint16_t* activation, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Builds the exact AoSoA4 sidecar above from one canonical row-major FP8
// [5120, 6144] matrix. canonical_weights is read-only and requires 4-byte
// alignment; sidecar_weights requires 16-byte alignment. Both complete
// 31,457,280-byte spans must be disjoint. Each destination uint4 is written
// once after applying the byte-wise code ^ (code >> 5) transform. The launch
// is asynchronous, allocation-free, and accepts no other shape.
[[nodiscard]] int
launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_pack_cuda(
    const std::uint8_t* canonical_weights,
    std::uint8_t* sidecar_weights, std::size_t rows,
    std::size_t columns, void* cuda_stream = nullptr) noexcept;

// Fused checkpoint QKV/Z projection for the exact FP8 [10240, 5120] and
// [6144, 5120] M=1 shapes. Both matrices consume the same activation and
// retain the single-projection BF16 result for every output row. Unsupported
// shapes, unsafe aliases, or unaligned pointers return cudaErrorInvalidValue
// represented as int. Weights require 4-byte alignment, activation 8-byte
// alignment, and outputs 2-byte alignment.
[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_qkv_z_bf16_cuda(
    const std::uint8_t* qkv_weights, float qkv_weight_scale,
    const std::uint8_t* z_weights, float z_weight_scale,
    const std::uint16_t* activation, std::size_t qkv_rows,
    std::size_t z_rows, std::size_t columns, std::uint16_t* qkv_output,
    std::uint16_t* z_output, void* cuda_stream = nullptr) noexcept;

// Exact M=1 linear-attention input projection for FP8 QKV [10240, 5120],
// FP8 Z [6144, 5120], and BF16 A/B [48, 5120]. Twenty-four light QKV tail
// CTAs compute two adjacent A/B rows while sharing the activation load. QKV,
// Z, A, and B outputs are bitwise-identical to the established QKV/Z then A/B
// two-kernel chain. All four output spans must be mutually disjoint and must
// not overlap any input. FP8 weights require 4-byte alignment, activation
// 8-byte alignment, and BF16 weights and outputs 2-byte alignment.
[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_qkv_z_bf16_ab_pair_cuda(
    const std::uint8_t* qkv_weights, float qkv_weight_scale,
    const std::uint8_t* z_weights, float z_weight_scale,
    const std::uint16_t* a_weights, const std::uint16_t* b_weights,
    const std::uint16_t* activation, std::size_t qkv_rows,
    std::size_t z_rows, std::size_t ab_rows, std::size_t columns,
    std::uint16_t* qkv_output, std::uint16_t* z_output,
    std::uint16_t* a_output, std::uint16_t* b_output,
    void* cuda_stream = nullptr) noexcept;

// Fused checkpoint K/V projection for the exact FP8 [1024, 5120] M=1
// shape. Both matrices consume the same activation while retaining the
// single-projection BF16 result for every output row. Unsupported shapes or
// unsafe output aliases or unaligned pointers return cudaErrorInvalidValue
// represented as int. Weights require 4-byte alignment, activation 8-byte
// alignment, and outputs 2-byte alignment.
[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_pair_bf16_cuda(
    const std::uint8_t* first_weights, float first_weight_scale,
    const std::uint8_t* second_weights, float second_weight_scale,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* first_output, std::uint16_t* second_output,
    void* cuda_stream = nullptr) noexcept;

// Fused full-attention Q/K/V projection for the exact ordered FP8
// [12288, 5120], [1024, 5120], and [1024, 5120] M=1 shapes. Q retains its
// production row-quad order; K and V retain the paired row reduction order.
// All three outputs are bitwise-identical to the existing Q-then-K/V chain.
// Unsupported shapes, unsafe aliases, or unaligned pointers return
// cudaErrorInvalidValue before work is enqueued. Weights require 4-byte
// alignment, activation 8-byte alignment, and outputs 2-byte alignment.
[[nodiscard]] int launch_sm87_fp8_w8a16_gemv_q_kv_bf16_cuda(
    const std::uint8_t* q_weights, float q_weight_scale,
    const std::uint8_t* key_weights, float key_weight_scale,
    const std::uint8_t* value_weights, float value_weight_scale,
    const std::uint16_t* activation, std::size_t q_rows,
    std::size_t kv_rows, std::size_t columns,
    std::uint16_t* q_output, std::uint16_t* key_output,
    std::uint16_t* value_output, void* cuda_stream = nullptr) noexcept;

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

// Exact M=1 NVFP4 down projection for checkpoint shape [5120, 17408], fused
// with the following residual add and centered RMSNorm. The three outputs
// preserve the unfused BF16 boundaries in order:
//
//   raw_down_output = BF16(down_projection)
//   residual_output = BF16(residual_left + raw_down_output)
//   normalized_output = BF16(residual_output * inverse_rms * (1 + weight))
//
// Only rows=5120 and columns=17408 are accepted. weight_scale_2 must be
// finite and nonnegative, and epsilon must be finite and positive. Packed
// weights require 4-byte alignment, activation requires 8-byte alignment,
// and every other BF16 pointer requires 2-byte alignment. The three output
// spans must be mutually disjoint and disjoint from every input, packed-
// weight, and block-scale span. Invalid arguments return
// cudaErrorInvalidValue before any work is enqueued.
[[nodiscard]] int launch_sm87_nvfp4_w4a16_down_residual_norm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activation,
    const std::uint16_t* residual_left,
    const std::uint16_t* norm_weight, float epsilon,
    std::size_t rows, std::size_t columns,
    std::uint16_t* raw_down_output,
    std::uint16_t* residual_output,
    std::uint16_t* normalized_output,
    void* cuda_stream = nullptr) noexcept;

// Production scale6 twin of the exact down/residual/centered-RMSNorm launch.
// scale6_sidecar contains 4,177,920 bytes: 128 six-bit scale deltas in each
// 96-byte row-quad/K512 tile. scale_base reconstructs the canonical E4M3FN
// code and must be at most 192. The sidecar requires 32-byte alignment; every
// shape, output boundary, alias, and cooperative launch contract otherwise
// matches launch_sm87_nvfp4_w4a16_down_residual_norm_bf16_cuda.
[[nodiscard]] int
launch_sm87_nvfp4_w4a16_down_residual_norm_scale6_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* scale6_sidecar, unsigned int scale_base,
    float weight_scale_2, const std::uint16_t* activation,
    const std::uint16_t* residual_left,
    const std::uint16_t* norm_weight, float epsilon,
    std::size_t rows, std::size_t columns,
    std::uint16_t* raw_down_output,
    std::uint16_t* residual_output,
    std::uint16_t* normalized_output,
    void* cuda_stream = nullptr) noexcept;

// Static-resource query for the exact production scale6 Function selected by
// the launcher above.
[[nodiscard]] int
query_sm87_nvfp4_w4a16_m1_down_residual_norm_scale6_resources_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* local_bytes, int* maximum_threads_per_block,
    int* active_blocks_per_sm) noexcept;

// Fused checkpoint gate/up projection for the exact NVFP4 [17408, 5120] M=1
// shape. Both projections first round independently to BF16. gate_output is
// then overwritten with BF16(SiLU(rounded_gate) * rounded_up), while up_output
// retains the independently rounded up projection.
//
// The two scales must be finite and nonnegative. Packed weights require
// 4-byte alignment, activation 8-byte alignment, and outputs 2-byte
// alignment. Both output spans must be disjoint from each other, activation,
// and both matrices' packed weights and block scales. Any other shape or
// invalid argument returns cudaErrorInvalidValue before work is enqueued.
[[nodiscard]] int launch_sm87_nvfp4_w4a16_gemv_gate_up_silu_bf16_cuda(
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* gate_block_scales, float gate_weight_scale_2,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* up_block_scales, float up_weight_scale_2,
    const std::uint16_t* activation, std::size_t rows, std::size_t columns,
    std::uint16_t* gate_output, std::uint16_t* up_output,
    void* cuda_stream = nullptr) noexcept;

// Fuses the post-attention residual add and centered RMSNorm into the exact
// NVFP4 gate/up/SiLU projection above. residual_output receives
// BF16(residual_left + residual_right). The normalized BF16 activation is
// consumed directly from CTA-local shared memory and is not materialized as
// a global output. gate_output and up_output retain the same contracts as the
// gate/up/SiLU entry point.
//
// Only [17408, 5120] is accepted. epsilon must be finite and positive. Packed
// weights require 4-byte alignment; all BF16 pointers require 2-byte
// alignment. The three output spans must be mutually disjoint and disjoint
// from every input, packed-weight, and block-scale span. Invalid arguments
// return cudaErrorInvalidValue before work is enqueued.
[[nodiscard]] int
launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_bf16_cuda(
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* gate_block_scales, float gate_weight_scale_2,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* up_block_scales, float up_weight_scale_2,
    const std::uint16_t* residual_left,
    const std::uint16_t* residual_right,
    const std::uint16_t* norm_weight, float epsilon,
    std::size_t rows, std::size_t columns,
    std::uint16_t* residual_output,
    std::uint16_t* gate_output, std::uint16_t* up_output,
    void* cuda_stream = nullptr) noexcept;

// Decode-runner-only variant of the exact fused operation above. It preserves
// the independently rounded gate/up BF16 arithmetic boundary internally, but
// publishes only residual_output and the final gate_output. up_workspace is
// validated with the same size, alignment, and alias contract as up_output,
// then remains untouched on this exact route. Callers must prove that the up
// value is dead and must not observe up_workspace before overwriting it.
//
// This entry point exists so the Decode runner can avoid a dead global up
// publication without weakening the generic double-output ABI. It accepts
// only [17408, 5120], and every invalid argument returns
// cudaErrorInvalidValue before work is enqueued.
[[nodiscard]] int
launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_dead_up_bf16_cuda(
    const std::uint8_t* gate_packed_weights,
    const std::uint8_t* gate_block_scales, float gate_weight_scale_2,
    const std::uint8_t* up_packed_weights,
    const std::uint8_t* up_block_scales, float up_weight_scale_2,
    const std::uint16_t* residual_left,
    const std::uint16_t* residual_right,
    const std::uint16_t* norm_weight, float epsilon,
    std::size_t rows, std::size_t columns,
    std::uint16_t* residual_output,
    std::uint16_t* gate_output, std::uint16_t* up_workspace,
    void* cuda_stream = nullptr) noexcept;

// Small-M sequence-tile projections over the same canonical weight layouts.
// activations is contiguous token-major BF16 [token_count, columns] and output
// is contiguous token-major BF16 [token_count, rows]. token_count must be in
// [1, 8]; M=1 delegates to the corresponding GEMV entry point above.
//
// The optimized M=2..8 kernels assign one output row to a block/warp and reuse
// each streamed weight across every token in the tile. Unsupported alignment
// or K shapes retain correctness by enqueueing the existing M=1 path once per
// token on the same stream.
[[nodiscard]] int launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-M16 FP8 sequence-tile projection. activations is contiguous
// token-major BF16 [16, columns] and output is contiguous token-major BF16
// [16, rows]. The complete input/output spans are validated before any work
// is enqueued. The four checkpoint production shapes use the BF16 tensor-core
// path when weights are 16-byte aligned and activations are 8-byte aligned;
// every other valid non-empty shape falls back to two ordered M=8 launches.
// Empty rows or columns retain the successful no-op contract above.
[[nodiscard]] int launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-M32 FP8 sequence-tile projection. activations is contiguous
// token-major BF16 [32, columns] and output is contiguous token-major BF16
// [32, rows]. The complete input/output spans are validated before any work
// is enqueued. The four checkpoint production shapes use one BF16 tensor-core
// kernel when weights are 16-byte aligned and activations are 8-byte aligned;
// every other valid non-empty shape falls back to two ordered public M16
// launches. Empty rows or columns retain the successful no-op contract above.
[[nodiscard]] int launch_sm87_fp8_w8a16_m32_gemm_bf16_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-M64 FP8 attention-output projection. activations is contiguous
// token-major BF16 [64, 6144] and output is contiguous token-major BF16
// [64, 5120]. The exact aligned checkpoint shape uses one tensor-core kernel
// whose four M16 accumulator panels reuse each decoded weight tile. Every
// near-miss shape, invalid range, or unsupported alignment is rejected before
// enqueue; the projection dispatcher owns the ordered two-M32 fallback for
// every other M64 FP8 projection.
[[nodiscard]] int
launch_sm87_fp8_w8a16_m64_attention_output_gemm_bf16_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Exact C256/C512 FP8 Prefill projection. activations is contiguous
// token-major BF16 [token_count, columns] and output is contiguous token-major
// BF16 [token_count, rows]. The accepted checkpoint shapes are linear-
// attention QKV [10240,5120] and Z [6144,5120], full-attention Q
// [12288,5120] and K/V [1024,5120], and attention output [5120,6144]. The
// complete chunk is validated before one fixed whole-chunk grid is enqueued.
// Q and C512 K/V use N-major ordering; C256 K/V uses the measured-faster
// M-major ordering to avoid N-major's locality penalty at the same 32-CTA
// under-filled grid size.
// Every shape, token-count, range, alias, or production-alignment near miss
// returns cudaErrorInvalidValue before enqueue.
[[nodiscard]] int launch_sm87_fp8_w8a16_whole_chunk_gemm_bf16_cuda(
    const std::uint8_t* weights, float weight_scale,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-M16 NVFP4 sequence-tile projection. activations is contiguous
// token-major BF16 [16, columns] and output is contiguous token-major BF16
// [16, rows]. The complete input/output spans are validated before any work
// is enqueued. The two checkpoint MLP shapes use the BF16 tensor-core path
// when packed weights are 16-byte aligned, block scales are 2-byte aligned,
// and activations are 8-byte aligned; every other valid non-empty shape falls
// back to two ordered M=8 launches. Empty rows or columns retain the
// successful no-op contract above.
[[nodiscard]] int launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-M18 NVFP4 sequence-tile projection. activations is contiguous
// token-major BF16 [18, columns] and output is contiguous token-major BF16
// [18, rows]. The complete input/output spans are validated before any work
// is enqueued. The two checkpoint MLP shapes use one masked-M32 tensor-core
// kernel when packed weights are 16-byte aligned, block scales are 2-byte
// aligned, and activations are 8-byte aligned. That kernel reads and writes
// exactly 18 token rows, so no padded capacity is required. Every other valid
// non-empty shape falls back to one ordered public M16 launch followed by one
// public M2 launch. Empty rows or columns retain the successful no-op
// contract above.
[[nodiscard]] int launch_sm87_nvfp4_w4a16_m18_gemm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Runtime-count NVFP4 sequence-tile projection for M=17 and M=19..31.
// activations is contiguous token-major BF16 [token_count, columns] and output
// is contiguous token-major BF16 [token_count, rows]. The complete exact
// input/output spans are validated before any work is enqueued. The two
// checkpoint MLP shapes use one runtime-masked M32 tensor-core kernel when
// packed weights are 16-byte aligned, block scales are 2-byte aligned, and
// activations are 8-byte aligned. That kernel reads and writes exactly
// token_count rows; no padded capacity is required. Every other valid
// non-empty shape or alignment falls back to one ordered public M16 launch
// followed by public small-M launches of at most eight rows. M=18 retains its
// fixed API above and is rejected here. Empty rows or columns retain the
// successful no-op contract.
[[nodiscard]] int launch_sm87_nvfp4_w4a16_m17_m31_gemm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-M32 NVFP4 sequence-tile projection. activations is contiguous
// token-major BF16 [32, columns] and output is contiguous token-major BF16
// [32, rows]. The complete input/output spans are validated before any work
// is enqueued. The two checkpoint MLP shapes use one BF16 tensor-core kernel
// when packed weights are 16-byte aligned, block scales are 2-byte aligned,
// and activations are 8-byte aligned; every other valid non-empty shape falls
// back to two ordered public M16 launches. Empty rows or columns retain the
// successful no-op contract above.
[[nodiscard]] int launch_sm87_nvfp4_w4a16_m32_gemm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Fixed-M64 NVFP4 down projection. activations is contiguous token-major
// BF16 [64, 17408] and output is contiguous token-major BF16 [64, 5120].
// The exact aligned checkpoint shape uses one tensor-core kernel that reuses
// each decoded weight tile across four M16 panels. Every near-miss shape,
// invalid range, or unsupported alignment is rejected before enqueue; the
// projection dispatcher owns the ordered two-M32 fallback for other M64
// weights.
[[nodiscard]] int launch_sm87_nvfp4_w4a16_m64_down_gemm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t rows,
    std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Exact C256/C512 NVFP4 dense-MLP Gate or Up branch. activations is
// contiguous token-major BF16 [token_count, 5120] and output is contiguous
// token-major BF16 [token_count, 17408]. One production CTA owns an M128xN128
// tile and reuses each decoded B fragment across eight ordered M16 panels.
// The complete chunk is validated before the one-grid launch (272 CTAs for
// C256, 544 for C512). Only token_count=256 or 512 and the exact aligned
// checkpoint shape are accepted; every near miss fails closed.
[[nodiscard]] int
launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Exact C256/C512 NVFP4 dense-MLP Down branch. activations is contiguous
// token-major BF16 [token_count, 17408] and output is contiguous token-major
// BF16 [token_count, 5120]. The complete chunk is validated before one
// N-major grid is enqueued. Only token_count=256 or 512 and the exact aligned
// checkpoint shape are accepted; every near miss fails closed.
[[nodiscard]] int
launch_sm87_nvfp4_w4a16_whole_chunk_down_gemm_bf16_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
