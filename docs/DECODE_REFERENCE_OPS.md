---
q3x_document:
  id: q3x-decode-reference-ops
  class: contract
  status: active
  owner: runtime-maintainers
  authority: Decode common-operation numerical and dimension contract
  effective: 2026-08-09
  last_reviewed: 2026-09-09
  supersedes: []
  superseded_by: []
  ssot_for: reference Decode common-op dimensions, arithmetic, and error behavior
  review_trigger: any Decode common-op ABI, numerical rule, dimension, or error-contract change
---

# Decode common-op reference contract

> **Authority boundary.** This component contract refines the
> [system SDD](SDD.md) and is subordinate to it and the
> [engineering constitution](ENGINEERING_CONSTITUTION.md). Current
> implementation, qualification, and default-route truth belongs in
> [`CURRENT_STATUS.md`](CURRENT_STATUS.md). Mechanism-level tuning or
> benchmark rules here apply only inside a named active local optimization
> work package; they cannot set global priority or select production.

`q3x/runtime/decode_ops.h` provides allocation-free CPU and CUDA correctness
paths for the operations surrounding batch-one projection GEMVs. Checkpoint,
activation, and cache values use raw IEEE BF16 `uint16_t` storage. Reductions
and nonlinear arithmetic use FP32; BF16 outputs use round-to-nearest-even.
FP32 NaNs are always retained as quiet BF16 NaNs, including tiny payloads that
would otherwise truncate to infinity.

## Dimension constants that must not be mixed

- `kLinearAttentionHeadDimension = 128`: Q/K/V head width for the 48
  Gated-DeltaNet layers. `l2_normalize_heads_*` computes
  `x / sqrt(sum(x^2) + epsilon)` independently per head. This is an L2 norm,
  not an RMSNorm. The caller applies the DeltaNet-specific extra
  `1 / sqrt(128)` to Q only.
- `kFullAttentionHeadDimension = 256`: Q/K/V head width for the 16 full
  attention layers (24 query heads, 4 KV heads). Full-attention Q/K use
  `headwise_centered_rms_norm_*` independently on each 256-value head, with
  one shared `[256]` weight and effective gamma `1 + weight`. Q's 24 heads and
  K's 4 heads are each normalized in one launch; concatenating all heads into
  one RMS reduction is incorrect.
- `kQwenRotaryDimension = 64`: partial NeoX RoPE rotates dimensions 0..63
  of each 256-value full-attention head. Dimensions 0..31 pair with 32..63;
  dimensions 64..255 pass through unchanged.

The outer decoder and final norms use the single-vector centered RMSNorm. The
internal GDN output uses `headwise_plain_rms_norm_*` over 48 independent
128-value heads with one shared `[128]` weight; effective gamma is the stored
weight without adding one. The fused
`headwise_plain_rms_norm_silu_gate_*` path then applies the 48x128 `SiLU(z)`
gate without an intermediate launch.

## Operations

- BF16 embedding row gather with checked vocabulary and token bounds.
- Single-vector and headwise centered/plain RMSNorm, fused GDN norm+SiLU,
  residual add, `SiLU(gate) * up`, and
  `value * sigmoid(gate)` for the full-attention output gate.
- FP32 GEMV-output to BF16 RNE conversion.
- BF16 greedy argmax with full-vector finite validation and earliest-index tie
  breaking. Its CUDA path uses caller-provided
  `kBf16GreedyArgmaxWorkspaceResults` scratch, writes the final compact result
  at element zero, and performs no allocation or copy.
- Per-head GDN L2 normalization and fixed full-attention 256/64 partial NeoX
  RoPE.
- Stable row-wise FP32 softmax.
- Single-token causal GQA over BF16 KV cache layout
  `[sequence, kv_heads, head_dim]`.

GQA requires `query_heads % kv_heads == 0`. The caller supplies an FP32
`[query_heads, sequence]` scratch buffer; the operation leaves normalized
probabilities there and writes BF16 `[query_heads, head_dim]` output. No CPU or
CUDA API allocates internal storage.

CUDA calls are asynchronous on a caller-supplied stream, perform no copies or
synchronization, and clear an unrelated stale last-error before their first
kernel. Exact input/output aliasing is supported for RMSNorm input, pointwise
gates, L2 input, RoPE input, and softmax input. The ordinary residual-add
launcher additionally permits its output to equal the complete left or right
input span. The two inputs must be disjoint, and every shifted/partial output
overlap is rejected before enqueue. This does not relax the fused
residual-plus-RMSNorm launcher's disjoint-output contract. Weight/cache storage
and attention scratch/output must remain disjoint unless an API explicitly
states otherwise.

## Verification shape gates

Host tests use independent double-precision formulas for normalization, RoPE,
softmax, and awkward 6-query/2-KV GQA. They cover RNE halfway cases, signed
zero, infinity, tiny-payload NaN quieting, dimension overflow, bad epsilon,
bad head mapping, and undersized scratch.

SM87 CUDA tests compare every primitive with the CPU oracle on a non-default
stream while injecting stale CUDA errors. Target gates include full-attention
Q/K centered norms at 24x256 and 4x256, GDN plain/fused norms at 48x128, GDN
16x128 L2, full-attention 24-head 256/64 RoPE, and
24-query/4-KV/head-dim-256 GQA with
attention scale `1/sqrt(256) = 1/16`; awkward tails and non-finite propagation
are also covered.

The residual-add alias gate uses a 777-element BF16 payload, which leaves a
nine-element CTA tail, and requires bitwise agreement for out-of-place, exact
left-alias, and exact right-alias execution. Both exact aliases are also
captured and replayed twice as one-kernel CUDA Graphs with guard regions and
peer-input preservation.

## Scoped ordinary Prefill raw-score feed v2

The isolated `WP-EXACT-ATTENTION-SCORE-FEED-20260909` correction is subordinate
to the [active package](ROADMAP.md#2026-09-09-bounded-engineering-window).
It changes only the ordinary nonfixed generic QT2 suffix. The original QT2
remains the explicit test oracle and fixed-launcher implementation; GroupQ64,
scalar final prompt/Decode, liveness, state/reset, and startup are unchanged.
This is the package's unique correction, not a selected production route.

Each CTA retains two query tokens, one KV head, six producer warps and six
consumer warps, with the existing two KV16 buffers. Producer lane ownership
and the eight ordered `fmaf(Q_i, K_i, score)` updates per lane are unchanged.
The masked warp reduction is still 16, 8, 4, 2, 1; lane zero multiplies that
same FP32 sum by `1/16` and stores the raw FP32 result to shared memory. The
store/load adds no dtype conversion or arithmetic. Producers perform no
maximum, denominator, exponential, or PV recurrence.

Every consumer lane owns its original eight value dimensions and replicates
the original QT2 scalar state. Keys and the two queries are consumed in the
original order. Starting from `maximum=-inf`, `denominator=+0` and `PV=+0`,
the operations are exactly:

- When `score > maximum`, compute `correction=expf(maximum-score)`, update
  `denominator=denominator*correction+1`, then
  `PV=fmaf(PV, correction, V)` per dimension and set `maximum=score`.
- Otherwise compute `probability=expf(score-maximum)`, add it to the
  denominator, then use `PV=fmaf(probability, V, PV)` per dimension.
- After the final key, divide by that consumer's denominator, publish BF16,
  decode that rounded value, apply the unchanged two-branch sigmoid gate,
  and publish BF16 again.

The comparison, two FMA operand orders, exponential expression, denominator
update, final division, and BF16 boundaries must not be unified or
reassociated. Infinity, NaN, subnormal, signed-zero, odd-query, and causal-tail
paths retain those same operations and predicates. An invalid/masked query-key
pair is neither written nor read. The source equivalence argument is not a
substitute for the existing focused CUDA and complete live-state P40000/O16
oracles on the new ELF.

At iteration `n`, every thread loads K/V slot `n&1`; the top CTA barrier makes
those words visible. Producers publish scores for tile `n` while consumers
read scores/V for tile `n-1`. The bottom CTA barrier both publishes tile `n`
and retires tile `n-1` before its slot can be overwritten at `n+1`. Priming and
draining retain both barriers for every thread. The only shared objects are
the two packed K/V buffers (32,768 bytes) and two FP32 score buffers (1,536
bytes), totaling 34,304 bytes. No global workspace, allocation, lock, new
stream, or public ABI is added. All other launch validation and zero-enqueue
failure behavior is unchanged.

The existing `q3x_decode_ops_cuda_test --bulk-score-feed-exact-only` mode
retains its eleven bitwise/guard/special-value/Graph/route fixtures and adds
the exact shared-layout assertion. Ordinary ON explicit `baseline`/`liveness`
still select incumbent QT2; `score-feed` selects v2, and `combined` selects the
same compiled default as OFF. Mutable selectors and hit/resource seams remain
test-only. The distinct profile, plan and v21 policy values are owned by the
[external witness contract](EVALSCOPE_EVALUATION.md); compiled policy is not an
observed launch count. Only a passing same-ELF P40000/O16 live-state comparison
followed by the actual installed OFF API can determine this correction's
fitness; it remains production-ineligible and release-unqualified meanwhile.
