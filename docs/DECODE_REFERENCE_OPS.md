---
q3x_document:
  id: q3x-decode-reference-ops
  class: contract
  status: active
  owner: runtime-maintainers
  authority: Decode common-operation numerical and dimension contract
  effective: 2026-08-09
  last_reviewed: 2026-08-09
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
kernel. Exact input/output aliasing is supported for RMSNorm input, residual
operands, pointwise gates, L2 input, RoPE input, and softmax input. Weight/cache
storage and attention scratch/output must remain disjoint unless an API
explicitly states otherwise.

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
