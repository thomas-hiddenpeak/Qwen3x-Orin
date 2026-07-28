#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Opaque, device-bound exact-C512 NVFP4 Prefill context. The factory creates
// all cuBLASLt objects, queries the exact BF16 problem with a zero-workspace
// preference, and retains the first successful zero-workspace heuristic.
// It never creates or owns a persistent BF16 weight copy.
struct Sm87Nvfp4PrefillCublasLtContext;

// Creates one context for the current SM87 CUDA device. `context` must be
// non-null and is set to null before any fallible work. Returns a CUDA status
// represented as int. No exception escapes this function.
[[nodiscard]] int create_sm87_nvfp4_prefill_cublaslt_context(
    Sm87Nvfp4PrefillCublasLtContext** context) noexcept;

// Reports the caller-owned transient BF16 scratch extent (170 MiB), the
// selected algorithm workspace requirement (always zero), and its runtime
// heuristic rank. Every output pointer and context must be non-null.
[[nodiscard]] int query_sm87_nvfp4_prefill_cublaslt_context(
    const Sm87Nvfp4PrefillCublasLtContext* context,
    std::size_t* scratch_bytes, std::size_t* workspace_bytes,
    int* heuristic_rank) noexcept;

// Exact Gate/Up C512 launch. Inputs use canonical checkpoint layout:
//
//   packed_weights: E2M1 pairs, row-major [17408, 5120/2]
//   block_scales:   E4M3FN, row-major [17408, 5120/16]
//   activations:    BF16 token-major [512, 5120]
//   output:         BF16 token-major [512, 17408]
//
// `bf16_scratch` is caller-owned contiguous BF16 [17408, 5120] with at least
// the queried extent. The asynchronous launch enqueues exactly two ordered
// operations on `cuda_stream`: canonical direct dequantization into scratch,
// then cuBLASLt with FP32 alpha=weight_scale_2 and beta=0. It allocates no
// memory and performs no synchronization.
//
// Only token_count=512, rows=17408, columns=5120, non-overlapping aligned
// spans, and finite weight_scale_2>0 are accepted. Invalid calls fail before
// enqueue with cudaErrorInvalidValue represented as int.
[[nodiscard]] int launch_sm87_nvfp4_prefill_cublaslt_gate_c512(
    Sm87Nvfp4PrefillCublasLtContext* context,
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* bf16_scratch,
    std::size_t scratch_bytes, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Releases host-side cuBLASLt resources only. It deliberately does not
// synchronize any stream; the owner must ensure all launches using `context`
// have completed before destruction. Null is accepted.
void destroy_sm87_nvfp4_prefill_cublaslt_context(
    Sm87Nvfp4PrefillCublasLtContext* context) noexcept;

// Independent device-bound exact-C512 NVFP4 Down context. It owns a distinct
// cuBLASLt handle, problem descriptors, preference, and selected algorithm;
// it never owns the caller-provided transient BF16 scratch.
struct Sm87Nvfp4PrefillDownCublasLtContext;

// Creates one Down context for the current SM87 CUDA device. `context` must be
// non-null and is cleared before any fallible work.
[[nodiscard]] int create_sm87_nvfp4_prefill_down_cublaslt_context(
    Sm87Nvfp4PrefillDownCublasLtContext** context) noexcept;

// Reports the caller-owned canonical BF16 [5120,17408] scratch extent
// (170 MiB), the selected zero workspace requirement, and heuristic rank.
[[nodiscard]] int query_sm87_nvfp4_prefill_down_cublaslt_context(
    const Sm87Nvfp4PrefillDownCublasLtContext* context,
    std::size_t* scratch_bytes, std::size_t* workspace_bytes,
    int* heuristic_rank) noexcept;

// Exact Down C512 launch over canonical checkpoint tensors:
//
//   packed_weights: E2M1 pairs, row-major [5120, 17408/2]
//   block_scales:   E4M3FN, row-major [5120, 17408/16]
//   activations:    BF16 token-major [512, 17408]
//   output:         BF16 token-major [512, 5120]
//
// `bf16_scratch` is caller-owned BF16 [5120,17408] with at least the queried
// extent. The allocation-free asynchronous launch enqueues the selected
// Window8 canonical dequantization kernel followed by zero-workspace
// cuBLASLt. Only the exact shape, pairwise-disjoint spans, finite positive
// scale, and documented alignments are admitted before enqueue.
[[nodiscard]] int launch_sm87_nvfp4_prefill_cublaslt_down_c512(
    Sm87Nvfp4PrefillDownCublasLtContext* context,
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales, float weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* bf16_scratch,
    std::size_t scratch_bytes, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Releases host-side Down cuBLASLt resources without synchronizing. Null is
// accepted; the owner must first complete all launches using the context.
void destroy_sm87_nvfp4_prefill_down_cublaslt_context(
    Sm87Nvfp4PrefillDownCublasLtContext* context) noexcept;

}  // namespace q3x::kernels
