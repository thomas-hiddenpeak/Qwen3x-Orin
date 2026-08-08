---
q3x_document:
  id: q3x-reference-gemv
  class: contract
  status: active
  owner: kernel-maintainers
  authority: batch-one reference GEMV numerical, ownership, validation, and launch contract
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: reference GEMV layouts, arithmetic, validation, ownership, and launch behavior
  review_trigger: any reference GEMV ABI, dtype, layout, arithmetic, validation, ownership, or launch-contract change
---

# Batch-one reference GEMV contract

> **Authority boundary.** This component contract refines the numerical kernel
> boundary in the [system SDD](SDD.md) and is subordinate to it and the
> [engineering constitution](ENGINEERING_CONSTITUTION.md). Current route,
> qualification, performance, and Production lifecycle truth belongs in
> [`CURRENT_STATUS.md`](CURRENT_STATUS.md). CTA shapes, staging, cache policy,
> kernel fusion, and timing thresholds are excluded and may govern only a
> named active local work package.

The public API is `include/q3x/kernels/reference_gemv.h`. It supplies
allocation-free CPU and CUDA correctness operations for canonical checkpoint
layouts. These functions define numerical and failure behavior, not the
project's optimized scheduling architecture.

## Data and numerical contract

All matrices are row-major `[rows,columns]`. The activation is one BF16 vector
`[columns]`; BF16 is passed as raw IEEE bfloat16 `uint16_t` bits.

- BF16 consumes BF16 weights directly.
- FP8 consumes E4M3FN `[rows,columns]` and a finite, non-negative per-tensor
  `weight_scale`.
- static FP8 W8A8 first divides each BF16 activation by a finite positive
  `input_scale`, saturates and rounds it to finite E4M3FN, then expands the
  code back to FP32 **and multiplies by `input_scale`** before the dot product.
- NVFP4 consumes canonical packed U8 `[rows,columns/2]`, E4M3FN block scales
  `[rows,columns/16]`, and a finite, non-negative `weight_scale_2`. Even K is
  the low nibble, odd K the high nibble, and columns must be a multiple of 16.

Every dot product accumulates in FP32. FP32-output APIs write one FP32 value
per row. Direct BF16-output APIs round the completed FP32 result to BF16
round-to-nearest-even. Encoded/BF16 NaNs and infinities follow IEEE arithmetic
and are not silently clamped, except for the explicitly defined finite
static-W8A8 activation conversion.

CUDA reduction order may differ from the sequential CPU oracle, so finite
cross-device comparisons use the applicable numerical accuracy policy rather
than assuming bitwise FP32 equality.

## Public launch and ownership behavior

CPU operations return `GemvStatus`. CUDA operations return `cudaError_t` as
`int` so CUDA types stay out of the header.

CUDA pointers and optional streams are caller-owned. A launch allocates,
copies, and synchronizes nothing. `cuda_stream == nullptr` selects the legacy
default stream; otherwise the `void*` must represent a valid `cudaStream_t`.
Writable outputs must satisfy the alias rules documented by the corresponding
function.

The generic BF16 pair-tile API accepts token-major input and two independent
outputs for `token_count` in `[1,16]`; it validates the complete
two-projection operation before enqueue. The exact
`launch_bf16_gemv_pair_m16_projection_fused_cuda` entry is the ordinary SM87
runtime subroute for the pinned BF16 A/B pair at `M=16, N=48, K=5120`; it is
not merely a diagnostic surface. The resource-query entries are diagnostic
gates and do not establish a runner route or an optimization mandate.

Empty shapes are successful no-ops where the header permits them. Non-empty
null pointers, invalid scales, overflowing `rows * columns`, invalid NVFP4
column grouping, illegal token counts, or prohibited aliases return a
structured host error or `cudaErrorInvalidValue`. Before a valid launch, the
wrapper clears unrelated stale CUDA last-error state and returns the status of
its own launch boundary.

## Verification boundary

Host tests cover fixed and deterministic randomized shapes, independent
double-precision formulas, zero matrices, all encoding boundaries,
malformed arguments, overflow, and non-finite propagation. CUDA tests compare
all public numerical paths with host references, including model K sizes,
awkward tails, large row counts, non-default streams, stale CUDA errors,
aliases, and non-finite values.

Synthetic matrices are valid for correctness enumeration and smoke tests only.
Historical kernel mechanisms, resource observations, and timings belong in
[`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md). Exact lineage for the
ordinary M16 BF16 pair subroute is the
[`M16 projection-fused record`](metadata/qwen36-27b-bf16-m16-projection-fused-benchmark.json).
It does not amend this contract.
