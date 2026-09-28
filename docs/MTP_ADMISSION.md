---
q3x_document:
  id: q3x-mtp-admission
  class: active
  status: active
  owner: runtime-maintainers
  authority: isolated greedy MTP development boundaries and transaction contract
  effective: 2026-09-28
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: native MTP admission design and first composition scope
  review_trigger: draft execution, verifier, state transaction, or admission API change
---

# Native MTP admission

This subsystem refines the generation boundary in [SDD](SDD.md). The owner
authorized MTP development with configured draft lengths **2 and 3**, followed
by business-driven scanning later. `WP-MTP-20260928` owns
`AC-MTP-GREEDY-v1`. The existing non-MTP targets and production artifact remain
separate; an MTP result cannot claim to close a non-MTP performance gap.
[Roadmap](ROADMAP.md) owns delivery order and [Current Status](CURRENT_STATUS.md)
owns what actually executes. This design does not imply a working GPU backend.

## Product trace and bounded composition

The product question is whether committed-token throughput improves on the
same real generation/API requests without changing greedy output, capacity,
stream semantics or numerical state. Current long-context Decode cost selects
one small draft layer plus a weight-reusing target verifier; faster drafting
alone cannot select the architecture.

The first composition consists of exactly these dependencies:

1. Validate the 15 BF16 MTP tensors and implement the bounded accept/commit
   controller with rejection, EOS, cancellation and failure tests.
2. Bind authenticated MTP weights, shared embedding/lm-head, prompt hidden
   capture, draft KV and complete target-state prefix selection in a separate
   non-installable native admission. A scalar verifier may establish the state
   oracle here, but is only a correctness prerequisite.
3. Compose a multi-row target verifier with reuse of projection weights and
   unchanged per-token recurrence/publication boundaries. Return immediately
   to the real P65 sanity and P8192/P40000 O256 API panel, for lengths 2 and 3.

These three integration stages are the composition bound. No draft-length
scan, local kernel tuning campaign or production promotion precedes that API
return. If the composed direction is negative, record the acceptance and cost
breakdown and either close v1 or declare one materially changed architecture;
do not keep collecting qualification evidence for a losing composition.
Serial target execution plus drafting is not an acceleration architecture.

## Checkpoint and draft model

The pinned revision is `0893e1606ff3d5f97a441f405d5fc541a6bdf404` of
`nvidia/Qwen3.6-27B-NVFP4`. It has one MTP full-attention layer and no dedicated
embedding. All 15 MTP tensors are **BF16**, totaling **849,398,784 bytes**.
The private [catalog](../src/model/mtp_weights_internal.h) lists exact names,
ranks and shapes; the [planner](../src/model/mtp_weights_internal.cpp) rejects
missing/extra MTP entries, wrong category/dtype/shape, byte sizes and ranges.
Its 256-byte-aligned offsets describe a prospective resident arena, not a
device allocation or payload authentication. The source manifest must outlive
its non-owning pointers. GPU startup must use the authenticated resident-loader
ownership/security contract; passing this header check is insufficient.

The primary reference studied locally is vLLM's `qwen3_5_mtp.py`, together
with `qwen3_next.py` and `v1/spec_decode/llm_base_proposer.py`. Exact source
hashes are pinned in the [initial evidence](metadata/qwen36-27b-mtp-foundation-2026-09-28.json).
They are architecture references, not runtime dependencies or imported code.

For target row t, pair its **final normalized hidden** with embedding of
token t+1; keep position t. Normalize the embedding and hidden separately,
concatenate **embedding first**, apply BF16 `fc` (10240 to 5120), then the one
full-attention decoder layer and MTP final centered RMSNorm. Subsequent draft
steps feed the MTP hidden and predicted next token back through that layer.
Use the shared base lm-head for logits. Q projection width is 12288, including
the gate; Q attention width is 6144, K/V width 1024, head size 256.

Prompt initialization needs the corresponding target hidden rows and shifted
tokens throughout the prefix, including the target's first prediction in the
last pair. Reusing only the last hidden with an empty draft cache is not the
reference algorithm. MTP's pre-FC hidden norm does not authorize passing an
unnormalized target residual. A proposed EOS is not a committed EOS.

## Greedy verification and state transaction

At round entry, the seed is already emitted but has not yet been consumed by
the target. Given d drafts, verify `[seed, draft0, ..., draft(d-1)]` and obtain
d+1 target predictions. Accept only the longest contiguous matching prefix,
then emit the first mismatch's target prediction or the all-accepted bonus.
Later predictions after a mismatch are invalid regardless of matching IDs.

The private [controller](../src/runtime/mtp_control.h) admits configured d=2
or 3. Effective d is at most remaining output minus one, so a final one-token
tail executes no draft. Full request capacity is checked before a transaction;
speculation never admits a larger `P+O-1` boundary. All returned token IDs must
be in vocabulary, and the backend must reject non-finite target logits.

The backend contract stages all speculative work, then synchronously selects
the prefix state **before each token observer**. Emitting n round outputs
requires exactly n new target state rows: seed plus the first n-1 outputs.
The last output remains pending. EOS, length and cancellation may select an
earlier prefix even when later drafts would have been accepted. EOS wins over
observer cancellation on that same token, matching the existing engine.

Prefix selection must cover all 48 GDN states, all convolution histories,
16 target KV live lengths/content, target hidden boundaries and draft cache
reconciliation. Truncating KV length alone does not undo a hybrid recurrence.
Draft hidden/KV generated from speculative hidden cannot replace entries
conditioned on accepted **target** hidden. Saved prefix states remain immutable
until the transaction retires; later accepted prefixes must still be selectable.
Discarded physical KV rows may remain only if logically inaccessible and
overwritten before reuse. No rejected token reaches text, SSE or usage.

Failure after transaction entry always invokes abort, even if begin failed.
Abort drains work and poisons uncertain state; it never grants normal reuse.
Failure after token publication terminates the request, preserving its partial
publication receipt. It cannot retract already emitted text or report success.
Only a successful full reset may recover the backend. Device allocation and
workspace growth during a round are forbidden. The host controller allocates
no dynamic storage and requires serialized, non-reentrant use.

## Numerical and memory obligations for the device stage

Target verification must preserve the incumbent scalar Decode semantics for
the identical token history, including projection reduction, BF16 publication,
ordered Attention, recurrent updates and argmax tie breaking. Existing Prefill
tile execution is not automatically a valid multi-token verifier. Complete
accepted-prefix GDN/Conv/KV/hidden and full logits must be compared to a scalar
teacher-forced oracle before using that verifier to select output. The draft
model also requires an independently assessed reference; low acceptance is not
automatically a model limitation if hidden shifting or normalization is wrong.

A conservative resource floor at 44,095 positions includes BF16 MTP weights,
4096 bytes/position of draft KV, 10240 bytes/position of retained target hidden,
and entry plus d+1 snapshots of the 78,446,592-byte recurrent/Conv state.
For d=3 that totals 1,873,777,664 bytes **before** draft projection/Attention,
logit, prefix-hidden and transaction scratch. This is a planning floor, not
an admitted arena or a claim of fitting the retained 8-GiB reserve. A composed
startup must calculate all lifetimes and prove actual free-memory headroom.
Pointer selection, layer-local snapshots or recomputation may reduce the
storage only after their full state contract is demonstrated.

## Evidence and API exit

The ordinary service remains MTP-disabled. The admission must identify its
different model inventory, plan, physical verifier work, committed tokens and
MTP policy; it cannot reuse a witness claiming zero MTP steps. No public
sampling feature or speculative token streaming is introduced.

Report together: total Prefill (including required draft initialization),
external TTFT, committed Decode token/s, end-to-end request duration, proposed
and accepted counts, acceptance by draft position, verified target rows,
round count, draft/verify/reconcile costs and peak/reserved memory. Exclude
the first Prefill-produced token from the Decode numerator. Never count
discarded proposals or bonus tokens as accepted draft tokens. Streaming
timestamps must expose token bursts rather than fabricating uniform latency.

The first host milestone's exhaustive state-machine tests and real checkpoint
header check have no numerical, GPU, API, acceptance-rate or speedup authority.
The exact completion boundary is in its
[evidence record](metadata/qwen36-27b-mtp-foundation-2026-09-28.json).
