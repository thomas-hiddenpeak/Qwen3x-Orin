---
q3x_document:
  id: q3x-model-weight-binding
  class: contract
  status: active
  owner: runtime-maintainers
  authority: typed resident-weight graph, numerical payload, lifetime, and dispatch-interface contract
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: ModelWeights types, binding validation, non-owning lifetime, and projection API behavior
  review_trigger: any ModelWeights ABI, tensor graph, payload formula, binding validation, ownership, or public dispatch change
---

# Qwen3.6-27B resident weight binding contract

> **Authority boundary.** This contract refines the model-asset and projection
> boundaries in the [system SDD](SDD.md) and is subordinate to it and the
> [engineering constitution](ENGINEERING_CONSTITUTION.md). Current route,
> qualification, performance, and Production lifecycle truth belongs in
> [`CURRENT_STATUS.md`](CURRENT_STATUS.md). Kernel tiles, cache policy,
> stream topology, fusion selection, and experiment thresholds are excluded;
> they may govern only a named active local work package.

The public API is `include/q3x/runtime/model_weights.h`. `ModelWeights`
converts the authenticated, string-keyed resident view table into one fixed,
typed, non-owning 64-layer graph at startup. After binding, graph traversal
requires no tensor-name lookup, scalar device read, or owning allocation.

## Ownership and binding

The ordinary runtime path binds the exact object returned by
`load_pinned_qwen36_27b` through `bind_qwen36_27b_weights`. The pinned overload
requires the 20,150,786,560-byte resident arena and exactly 1,846 text tensor
views.

`ResidentWeights` and its CUDA allocation must outlive `ModelWeights` and
every queued operation that consumes a bound pointer. `ModelWeights` does not
retain or extend owner lifetime. Moving `ResidentWeights` transfers its arena;
destroying or move-assigning over that owner invalidates all bound views.

The callback-based `WeightBindingSource` overload exists for deterministic
tests and already-validated adapters. It does not weaken tensor validation.
For every view, binding checks name presence, dtype, shape, byte count,
non-null device pointer, **256-byte-aligned arena offset and device pointer**,
arithmetic/range safety, and `arena_base + arena_offset` identity before
publishing a graph.

## Typed model graph

The bound graph exposes:

- BF16 `[248320,5120]` token embeddings, BF16 `[5120]` final centered norm,
  and a typed `[248320,5120]` lm-head;
- 64 decoder layers in the fixed three-linear-attention/one-full-attention
  schedule;
- BF16 input and post-attention norms plus gate/up/down projections in every
  layer;
- QKV, Z, BF16 A/B, `[10240,1,4]` convolution, A-log, dt-bias, plain norm,
  and output projection in each of 48 linear-attention layers; and
- Q/K/V/O plus BF16 `[256]` Q/K centered norms in each of 16 full-attention
  layers.

Projection type is selected from the authenticated payload dtype, never from a
module-name guess:

| Payload | Bound alternative | Required companions |
| --- | --- | --- |
| BF16 `[N,K]` | `Bf16LinearWeight` | none |
| E4M3FN `[N,K]` | `Fp8LinearWeight` | F32 scalar `weight_scale`, F32 scalar `input_scale` |
| packed U8 `[N,K/2]` | `NvFp4LinearWeight` | E4M3FN `[N,K/16]` block scale, F32 scalar `weight_scale_2`, F32 scalar `input_scale` |

F32 companions are read once during binding, retained both as device pointers
and host values, and must be finite and non-negative. The W8A16 and W4A16
projection formulas consume BF16 activations and therefore validate but do not
apply the checkpoint activation scale.

## Numerical payload contract

The stable projection interpretation is:

```text
BF16:  y[n] = sum_k BF16(w[n,k]) * BF16(x[k])
FP8:   y[n] = sum_k E4M3FN(w[n,k]) * weight_scale * BF16(x[k])
NVFP4: y[n] = sum_k E2M1(w[n,k]) * E4M3FN(block_scale[n,k/16])
                    * weight_scale_2 * BF16(x[k])
```

NVFP4 even K occupies the low nibble and odd K the high nibble. Accumulation
is FP32 and direct-to-BF16 APIs round their completed observable outputs to
BF16 round-to-nearest-even. NaNs and infinities are not silently clamped.
Different conforming reduction groupings are not automatically a bitwise
cross-backend oracle; whole-runner accuracy eligibility is controlled by the
constitution and real-model policy, not by a local dispatch document.

## Projection interfaces

All projection launchers use caller-owned device storage, accept an optional
`void*` CUDA stream, allocate nothing, and do not synchronize. Writable output
ranges must not alias another operand unless the specific public operation
explicitly documents that behavior.

- `launch_projection_reference_cuda` writes FP32 and dispatches by the bound
  variant.
- `launch_projection_to_bf16_reference_cuda` uses caller-owned FP32 scratch
  and emits BF16.
- `launch_projection_to_bf16_cuda` selects the strongly typed
  `ProjectionBackend`; `kReference` is the compatibility default and
  `kSm87WeightOnly` is explicit. BF16 remains on the reference formula and
  lm-head FP32 behavior is unchanged.
- `launch_projection_tile_to_bf16_cuda` accepts token-major BF16 input/output
  for `M=1..64`. This is a projection-component capacity, not the runner's
  Prefill capacity.

The narrow public launch families retain these stable ABI boundaries:

| Public family | Token/shape boundary | Observable contract |
| --- | --- | --- |
| exact FP8 whole-chunk | `M in {256,512}`; `[10240,5120]` QKV, `[6144,5120]` Z, `[12288,5120]` Q, `[1024,5120]` K/V, or `[5120,6144]` attention output | direct BF16 output; does not widen the generic M64 API |
| exact NVFP4 whole-chunk branch | `M in {256,512}`; `[17408,5120]` Gate/Up or `[5120,17408]` Down | direct BF16 output; no FP32 scratch; does not widen the generic M64 API |
| projection pair tile | `M=1..64`; common input K and two independently sized outputs | validates both projections and all cross-ranges before first enqueue; exact BF16 A/B is two `[48,5120]` matrices and uses the ordinary M16 fused subroute |
| linear-attention QKV/Z/A/B group | `M=1`; FP8 `[10240,5120]`, FP8 `[6144,5120]`, then two BF16 `[48,5120]` matrices | four distinct BF16 outputs; unsupported exact-route eligibility is reported before enqueue |
| full-attention Q/K/V group | `M=1`; exact fusion eligibility is FP8 `[12288,5120]`, `[1024,5120]`, `[1024,5120]` | three distinct BF16 outputs; valid non-fused combinations preserve ordered fallback |
| MLP Gate/Up/SiLU group | `M=1`; exact fusion eligibility is two NVFP4 `[17408,5120]` matrices | Gate output becomes rounded `SiLU(Gate) * Up`; the ordinary form also publishes rounded Up |
| residual/norm plus Gate/Up/SiLU | `M=1`, hidden 5,120, same Gate/Up shapes | publishes rounded residual and Gate; ordinary form publishes Up, while the decode-only dead-Up form makes that workspace unobservable until overwritten |
| Down plus residual/norm | `M=1`; exact fusion eligibility is NVFP4 `[5120,17408]` | publishes three distinct BF16 boundaries: raw Down, residual, and normalized output |

Generic BF16 operands require natural 2-byte alignment. Exact FP8 whole-chunk
eligibility requires 16-byte-aligned weight, 8-byte-aligned BF16 input, and
2-byte-aligned output. Exact NVFP4 whole-chunk eligibility additionally
requires a 16-byte-aligned packed weight and 2-byte-aligned block scales.
Single-token FP8/NVFP4 grouped fusion eligibility uses 4-byte-aligned
quantized weights, 8-byte-aligned activation, and naturally aligned BF16
outputs; BF16 A/B weights are naturally aligned. These stricter exact-route
near misses return `cudaErrorNotSupported` where the API defines caller
fallback, while null, under-capacity, overflowing, unnaturally aligned, or
overlapping operands return `cudaErrorInvalidValue`.

Static shape dispatch is an interface/implementation constraint only. It does
not prescribe project priority, architecture, or promotion. A structurally
unsupported optional route returns `cudaErrorNotSupported` before enqueue so
its caller may choose a documented fallback. Malformed payloads, pointers,
scalars, sizes, ranges, aliases, or backend values return
`cudaErrorInvalidValue` and are not fallback signals. Composite launchers
validate the complete operation before their first enqueue so a bad later
operand cannot leave a partial prefix queued.

## Optional sidecar attachment boundary

The header exposes transactional attachment APIs for optional FP8 and NVFP4
derived layouts. Those APIs are stable only in these respects:

- canonical null/zero input detaches the applicable set where documented;
- all descriptors, shapes, alignments, ranges, and complete-set invariants are
  validated before any existing binding changes;
- failure preserves the previous attachment set;
- descriptor fields are copied during attachment, while device arenas remain
  non-owning and must outlive all consumers; and
- replacement or detach requires a globally quiescent point because the call
  does not synchronize earlier CUDA work.

An attached pointer does not by itself authorize scheduling or imply that a
route is default, qualified, or Production. Sidecars prepared by the ordinary
engine construction path are real runtime dependencies of the attached
`ModelWeights`, even when the consuming route is not yet Production-qualified.
Only sidecars not selected by the current DeploymentPlan/default route are
dormant; named local work packages may study them without turning their
mechanisms into global rules.

## Failure and verification boundary

Binding errors distinguish invalid sources/arenas, missing tensors,
unsupported or mismatched dtypes, shape/byte mismatches, null or misaligned
pointers, arena-range mismatch, invalid scalar, layer-schedule and arithmetic
errors, CUDA failure, and allocation failure. A failed bind publishes no
partially valid `ModelWeights`.

Host and CUDA tests cover the complete typed graph, all three numerical
payload alternatives, schedule and companion validation, owner lifetime,
alignment/range/overflow/alias failures, stale CUDA-error isolation, and
allocation-free dispatch. Synthetic kernel matrices remain correctness and
smoke tools only. Historical mechanism measurements and gates live in
[`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md); exact widened-prefix
lineage includes the
[`M17/M19..M31 runtime-masked record`](metadata/qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json).
It has no active authority here.
