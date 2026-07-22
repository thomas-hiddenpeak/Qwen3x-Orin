# Qwen3.6-27B resident weight binding

`q3x::runtime::ModelWeights` is the boundary between the authenticated
resident arena and the future 64-layer decoder. It converts string-keyed
`DeviceTensorView` records into a fixed, typed, non-owning graph once at model
startup. The graph contains no names, vectors, maps, owning allocations, or
lazy scalar reads, so projection dispatch does not allocate after binding.

This component does not execute a layer and does not expand the current
support claim to native inference.

## Lifetime and production entry point

Production code binds the exact pinned arena:

```cpp
q3x::runtime::ResidentLoadResult loaded =
    q3x::runtime::load_pinned_qwen36_27b(model_directory);
if (!loaded) {
  // Report loaded.diagnostic.
}

q3x::runtime::WeightBindResult bound =
    q3x::runtime::bind_qwen36_27b_weights(*loaded.value);
if (!bound) {
  // Report bound.diagnostic.
}
```

The `ResidentWeights` object and its CUDA arena must outlive `ModelWeights`
and every queued CUDA operation that uses a bound pointer. `ModelWeights`
does not retain an owner pointer and does not extend this lifetime. Moving the
owner transfers its CUDA allocation, but destroying it or move-assigning over
it invalidates all bound views. The production overload rejects any arena
whose size is not exactly `20,150,786,560` bytes or whose text view table does
not contain exactly `1,846` entries.

The callback-based `WeightBindingSource` overload exists for deterministic
tests and already-validated alternative arena adapters. It does not weaken
per-tensor checking: every requested view must still have the exact dtype,
shape, byte count, non-null pointer, 256-byte-aligned offset and pointer, and
`arena_base + arena_offset` address.

## Typed graph

The graph fixes the catalogued Qwen3.6-27B constants and exposes:

- BF16 `[248320,5120]` token embeddings, BF16 `[5120]` final centered norm,
  and the independently typed `[248320,5120]` LM head;
- 64 decoder layers with the exact repeated schedule of three
  linear-attention layers and one full-attention layer;
- BF16 input/post-attention norms plus typed gate/up/down MLP projections in
  every layer;
- linear-attention QKV, Z, BF16 A/B, `[10240,1,4]` convolution, A-log,
  dt-bias, plain norm, and output projection in 48 layers;
- full-attention Q/K/V/O projections and BF16 `[256]` Q/K centered norms in
  16 layers.

Projection type is selected from the actual weight view dtype, never from a
module-name guess:

| Weight dtype | Bound alternative | Required companions |
| --- | --- | --- |
| `BF16 [N,K]` | `Bf16LinearWeight` | none |
| `F8_E4M3 [N,K]` | `Fp8LinearWeight` | F32 scalar `weight_scale`, F32 scalar `input_scale` |
| `U8 [N,K/2]` | `NvFp4LinearWeight` | F8_E4M3 `[N,K/16]` `weight_scale`, F32 scalar `weight_scale_2`, F32 scalar `input_scale` |

All F32 companions are copied synchronously from device once during binding
and must be finite and non-negative. Their device pointers and host values
are both retained. The reference W8A16 and W4A16 paths validate but do not
apply `input_scale`, because their activation input is already BF16.

## Allocation-free projection dispatch

`launch_projection_reference_cuda` accepts caller-owned BF16 device input,
writes caller-owned FP32 device output, and selects the existing BF16, FP8, or
canonical NVFP4 reference GEMV from the bound variant. It neither allocates
nor synchronizes.

`launch_projection_to_bf16_reference_cuda` additionally invokes the common
FP32-to-BF16 round-to-nearest-even kernel. The caller supplies FP32 scratch of
at least `linear_output_size(weight)` elements and the BF16 output buffer.
Both APIs accept a CUDA stream and isolate their launch result from an
unrelated stale CUDA last-error.

The production boundary `launch_projection_to_bf16_cuda` takes the strongly
typed `ProjectionBackend`. `kReference` is the default and preserves the
above behavior. Explicit `kSm87WeightOnly` dispatches FP8 and NVFP4 directly
to BF16 with the checked SM87 kernels; BF16 deliberately falls back to the
reference path, and LM-head FP32 projection is unaffected. Direct quantized
launches do not use FP32 scratch. Unknown backends, unknown/invalid variants,
and incomplete companion payloads return `cudaErrorInvalidValue`; no dispatch
path allocates or synchronizes.

`launch_projection_tile_to_bf16_cuda` extends that boundary to caller-owned
row-major BF16 input/output tiles with `M=1..32`. M1 delegates to the scalar
projection API. For explicit SM87 FP8/NVFP4 weights, M2..M8 use small-M
kernels that reuse each loaded weight across all M activation rows; M9..M15
are split into M8 plus the remaining M1..M7 rows. M16 first reaches a
format-specific fixed-tile launcher: exact aligned production shapes use
Ampere BF16 Tensor Core MMA and every other valid case falls back to two
ordered M8 launches. M17..M31 use an M16-first composite followed by an M8 and
M1..M7 tail as required; M18 is M16+M2. At M32, the four exact aligned FP8
production shapes use one fixed-M32 Tensor Core kernel; other FP8 cases and all
NVFP4 cases use two ordered M16 launches. BF16 weights and the reference
backend enqueue M checked M1 projections. The launcher validates the complete
tile spans, scratch capacity, overflow, and input/output overlap before
enqueueing any work.

The fixed-M16 FP8 route accepts `[10240,5120]`, `[5120,6144]`,
`[6144,5120]`, and `[12288,5120]` when weights are 16-byte aligned and
activations are 8-byte aligned. It expands the canonical E4M3FN checkpoint
bytes into BF16 shared-memory tiles and uses FP32 Tensor Core accumulation;
the measured `[1024,5120]` shape deliberately retains the two-M8 fallback.
The fixed-M16 NVFP4 route accepts `[17408,5120]` and `[5120,17408]` when
packed weights are 16-byte aligned, block scales are 2-byte aligned, and
activations are 8-byte aligned. It combines canonical E2M1 values with their
E4M3FN block scales before the same BF16 MMA. Neither route repacks or mutates
the resident checkpoint.

Within the SM87 NVFP4 launcher, aligned canonical weights with K divisible by
256 use a packed-x8 route: one 32-bit load supplies eight E2M1 values per lane,
and adjacent lanes share one 16-value block scale. Non-vector K and unaligned
weight pointers retain the original scalar kernel. This shape dispatch does not
change the public projection API or require an offline weight layout.

Three exact aligned M1 NVFP4 shapes refine that packed-x8 route. Down
`[5120,17408]`, gate/up `[17408,5120]`, and lm-head `[248320,5120]` use the
adjacent-lane XOR-dual arithmetic while staging their BF16 activation once per
CTA for reuse by grid-stride row quads. Down stages 34 KiB; gate/up and lm-head
stage 10 KiB. The cooperative global copy is 8 bytes wide, so all three routes
preserve the existing 8-byte activation alignment contract; they add no repack
or 16-byte public alignment requirement. Near-miss shapes, packed-weight or
activation misalignment, M2 through M32, and prefill retain their preceding
routes.

Within the SM87 FP8 launcher, canonical weights with K divisible by 1,024,
4-byte-aligned weights, and an 8-byte-aligned BF16 activation use a packed-x4
route. Each lane consumes four E4M3FN weights with one 32-bit load and four
activations with one 64-bit load, retaining four FP32 accumulators. Its
branchless decoder is exact for finite E4M3FN values and preserves signed zero
and signed canonical quiet NaNs. Non-vector K and either unaligned pointer
retain the scalar kernel. This dispatch likewise requires no public API or
checkpoint-layout change.

`supports_fp8_projection_pair` recognizes only two valid FP8 `[1024,5120]`
matrices under the explicit SM87 backend. For M1 and aligned operands,
`launch_projection_pair_tile_to_bf16_cuda` uses one K/V cross-matrix row-quad
kernel that shares activation decode and codebook setup while preserving each
single projection's BF16 bits. Eligible unaligned calls, M2..M32 tiles, other
shapes/types, and other backends retain the existing first-then-second path.
The runner applies the pair only to full-attention K/V on prompt-final and
decode steps; chunked prefix projection remains unchanged.

`supports_fp8_qkv_z_projection_pair` recognizes the ordered linear-attention
FP8 `[10240,5120]` QKV plus `[6144,5120]` Z pair under the same explicit
backend. At aligned M1, one 1,536-CTA two-phase launch first preserves the QKV
row-quad grid/stride and then lets its first 768 CTAs preserve the Z
grid/stride. Both phases reuse one decoded E4M3FN codebook and retain the two
production single-projection BF16 results bit-for-bit. Reversed or near-miss
shapes, M2..M32, other backends, 4-byte weight misalignment, 8-byte activation
misalignment, or 2-byte output misalignment retain two ordered projections.
The scalar runner step, including `prefill_prefix_tile(..., 1)`, uses the pair
dispatcher; layer-major C2..C32 prefix tiles remain independent QKV and Z
projections.

The SM87 kernels preserve the documented FP32-accumulation/BF16-RNE formula,
but their warp reduction and global-scale multiplication order are not
required to be bitwise identical to the deliberately scalar-shaped CUDA
reference. Optimized results must instead pass the independent per-operation
tolerance gate and the fixed full-model exact-token/text gate. The current
deterministic gate covers FP8 and NVFP4 K=5120/6144/17408, scalar and
unaligned fallbacks, all 256 E4M3FN codes in all four packed byte positions,
both reserved encodings with NaN-class checks, all E2M1 codes and packed
nibble positions, and adjacent-lane scale selection. Its 1,237 BF16 outputs
produced zero bit mismatches against the CUDA reference; the independent host
oracle checks reserved E4M3FN outputs by NaN class. The default remains
`kReference`;
passing one prompt does not make the optimized backend a universal
floating-point oracle.

The small-M gate covers M1 through M8 for all three bound weight alternatives,
production K=5120/6144/17408 shapes, awkward K fallback, unaligned inputs and
weights, host-double/CUDA references, repeated-M1 equivalence, and deterministic
replay. The full-model C8 gate separately retains the exact 19/26-token text and
44-step oracle contract. The fixed-M16 gates compare the Tensor Core output
with two production M8 launches per shape, report BF16 mismatch and
absolute/relative-error statistics, classify reserved FP8 NaNs separately,
and require deterministic replay. Their different reduction grouping is not
a universal bitwise-equivalence promise. The C16 full-model gate nonetheless
retains the exact 19 prompt IDs, 26 output IDs, decoded text, `<|im_end|>`, and
44 runner steps.

## Validation coverage

`model_weights_host` builds a complete synthetic 64-layer view table without
allocating model payloads. It covers all three projection alternatives, exact
48/16 scheduling, companion shapes and scalar copies, pointer/range/alignment
checks, missing tensors, overflow-facing metadata checks, move/copy policy,
and structured failure diagnostics.

`model_weights_cuda` compares all three dispatch alternatives with the CPU
GEMV references on a real CUDA device, checks stale-error isolation, and
checks the caller-scratch FP32-to-BF16 convenience path. It does not reload
the official 20 GB checkpoint.

`projection_backend_dispatch` is a small SM87 CUDA gate for all three
production routes at M1 through M32, BF16/reference fallback, M9..M15
segmentation, M16 Tensor Core selection/two-M8 fallback, M17..M32 M16-first
composition, scratch behavior,
whole-tile overlap/span validation, and fail-closed backend and variant
handling. It also covers exact-shape FP8 M1 K/V and QKV/Z pair selection,
near-miss and unaligned ordered fallbacks, different output sizes,
cross-output alias rejection, fail-before-enqueue behavior, CUDA Graph node
counts, and stale-error isolation. It uses only synthetic buffers.

`sm87_weight_only_gemv` covers awkward dimensions, aligned/unaligned dispatch,
all packed E4M3FN and E2M1 positions, the model reduction lengths, independent
host formulas, deterministic replay, and direct optimized-versus-reference
BF16 mismatch/error statistics. Its optional production-shape segment performs
a mirrored scalar/vector CUDA-event comparison and enforces a 1.15x minimum
speedup for the dominant NVFP4 shapes and FP8 shapes with at least 5,120 rows;
the smaller FP8 shape may regress by at most 2%. It is enabled with
`Q3X_RUN_SM87_WEIGHT_ONLY_GEMV_PERF=1`. The fixed-M16 same-binary gates use
`Q3X_RUN_SM87_FP8_M16_WMMA_PERF=1` and
`Q3X_RUN_SM87_NVFP4_M16_WMMA_PERF=1`; their final production-call-weighted
speedups over two M8 launches are 2.41756x and 1.56406x, respectively.
The FP8 M1 K/V pair correctness segment runs by default and covers all 254
finite E4M3FN codes in each packed byte position, isolated `0x7f`/`0xff` NaNs,
bitwise comparison, and output canaries. Its optional mirrored timing gate is
enabled with `Q3X_RUN_SM87_FP8_M1_KV_PAIR_PERF=1` and requires at least 1.10x
for both checkpoint-like and same-bank-stress fixtures.
The FP8 M1 QKV/Z two-phase gate is enabled with
`Q3X_RUN_SM87_FP8_M1_QKV_Z_FUSION_PERF=1`; production evidence additionally
requires `Q3X_FP8_M1_QKV_Z_ACTUAL_CHECKPOINT_FILE` and the decimal
`Q3X_FP8_M1_QKV_Z_ACTUAL_CHECKPOINT_OFFSETS=QKV:Z` pair. It compares the exact
production kernel with two production row-quad launches in five
baseline/candidate/candidate/baseline rounds, requires at least 1.02x on the
actual checkpoint and 1.00x on the same-bank stress guard, and reports the
stricter symmetric 1.01x result as an exploratory diagnostic rather than a
promotion criterion. It also covers all 254 finite codes in all four packed
positions for both matrices, isolated signed NaNs, replay, guards, input
preservation, resources, alignment, shape, scale, cap, and alias contracts.
The exact M1 NVFP4 data-reuse gates are enabled with
`Q3X_RUN_SM87_NVFP4_M1_DOWN_XOR_DUAL_PERF=1` and
`Q3X_RUN_SM87_NVFP4_M1_LM_HEAD_ACTIVATION_STAGED_PERF=1`. The gate/up staged
gate is enabled with
`Q3X_RUN_SM87_NVFP4_M1_GATE_UP_ACTIVATION_STAGED_PERF=1`, and the down staged
gate is enabled with
`Q3X_RUN_SM87_NVFP4_M1_DOWN_ACTIVATION_STAGED_PERF=1`. They compare the
preserved indexed-dual or direct-activation baselines with their production
candidates in the same binary, require bitwise finite/NaN and canary equality,
verify scalar fallback for packed-weight or activation misalignment, and
compare public/direct CUDA Graph kernel identities. The default down and
gate/up segments additionally capture the exact production shapes without
execution and verify `func`, grid, block, and dynamic-shared launch identity,
including scalar function identity for packed-weight `+1` and activation `+2`
fallbacks. The down gate also proves that the preserved direct XOR baseline
remains a distinct callable kernel.
