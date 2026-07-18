# Batch-one reference GEMV contract

`q3x/kernels/reference_gemv.h` defines the first allocation-free matrix-vector
boundary used by the text runtime. It is a correctness implementation for
canonical checkpoint layouts, not the final tensor-core or Marlin throughput
path.

## Data and numerical contract

All matrices are row-major `[rows, columns]`, the activation is one BF16
vector `[columns]`, accumulation is FP32, and output is FP32 `[rows]`. BF16 is
represented by its raw IEEE bfloat16 `uint16_t` bit pattern.

- `bf16_gemv_reference_cpu` consumes BF16 weights directly.
- `fp8_gemv_reference_cpu` consumes E4M3FN bytes and one finite,
  non-negative `weight_scale`. The decoded weight is
  `E4M3FN(byte) * weight_scale`.
- `fp8_static_gemv_reference_cpu` is a separate ModelOpt W8A8 diagnostic:
  BF16 activations are scaled, saturated and rounded to E4M3FN before the
  dot product. Its `input_scale` must be finite and strictly positive. The
  SM87 production dispatcher intentionally uses the preceding W8A16 API,
  matching vLLM's `MarlinFP8ScaledMMLinearKernel`, which discards the static
  activation scale on GPUs without native FP8 support.
- `nvfp4_gemv_reference_cpu` consumes canonical ModelOpt `uint8_t`
  `[rows, columns / 2]` weights, E4M3FN `[rows, columns / 16]` block scales,
  and one finite, non-negative `weight_scale_2`. Even K is the low nibble and
  odd K is the high nibble. K must be a multiple of 16.

The corresponding `launch_*_cuda` functions implement the same formulas.
They use one 256-thread block per row, cap the grid at 65,535 blocks, and
grid-stride over additional rows. This covers the pinned model's projection K
dimensions (including 5120 and 6144) and `lm_head` N=248320 without imposing a
65,535-row limit.

CUDA reduction order differs from the sequential host reference, so finite
results agree within an FP32 accumulation tolerance rather than bit-for-bit.
Encoded or BF16 NaNs and infinities follow IEEE arithmetic and are not silently
clamped, except that the explicit static-W8A8 activation conversion saturates
to finite E4M3FN by definition. Host weight scales must be finite and
non-negative; static-W8A8 `input_scale` must additionally be nonzero.

## Ownership and launch behavior

CUDA pointers and the optional stream are caller-owned. A launch performs no
allocation, copy, or synchronization. `cuda_stream == nullptr` selects the
legacy default stream; otherwise pass a `cudaStream_t` as `void*`. Output must
not alias the activation or weight storage.

Empty shapes are successful no-ops. Non-empty null pointers, overflowing
`rows * columns`, invalid scalar scales, and non-group-aligned NVFP4 K return a
structured host status or `cudaErrorInvalidValue`. Immediately before a valid
kernel launch, the wrapper clears an unrelated stale CUDA last-error and then
returns the status of its own launch boundary.

## Verification

`reference_gemv_host` checks fixed fixtures, 24 deterministic randomized shape
sets, independent double-precision dot products, zero matrices, encoded
boundary values, malformed arguments, dimension overflow, and NaN/infinity
propagation. `reference_gemv_cuda` compares all four GPU paths with the host
FP32 references on SM87, including awkward tails, K=5120/6144, N=248320
grid-striding, a non-default stream, stale CUDA errors, and non-finite values.

Current intentional limitations are batch size one, F32 output only, no bias,
and no fused epilogue. Activation quantization exists only in the explicitly
named static-W8A8 diagnostic; the production SM87 FP8 route remains W8A16.
These keep this path useful as a numerical oracle while optimized kernels are
added behind a distinct API.
