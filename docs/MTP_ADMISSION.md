---
q3x_document:
  id: q3x-mtp-admission
  class: active
  status: active
  owner: runtime-maintainers
  authority: isolated greedy MTP development boundaries and transaction contract
  effective: 2026-09-28
  last_reviewed: 2026-10-09
  supersedes: []
  superseded_by: []
  ssot_for: native MTP admission design and first composition scope
  review_trigger: draft execution, verifier, state transaction, or admission API change
---

# Native MTP admission

This subsystem refines the generation boundary in [SDD](SDD.md). The owner
authorized MTP development with configured draft lengths **2 and 3**, followed
by business-driven scanning later. `WP-MTP-20260928` owns
`AC-MTP-GREEDY-v1`; its efficiency successor is specified below. The existing non-MTP targets and production artifact remain
separate; an MTP result cannot claim to close a non-MTP performance gap.
[Roadmap](ROADMAP.md) owns delivery order and [Current Status](CURRENT_STATUS.md)
owns what actually executes. The native transaction below has scalar-oracle
and isolated multi-row modes. The service composition below is a separate
development API route; the default production service remains non-MTP.

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

## Efficiency composition v2

`WP-MTP-EFFICIENCY-20260928` responds to v1's measured 87–91% verification
share and the owner's 50%–200% Decode speedup objective. At 8K, d3 commits
255 tokens in 84 rounds; 1.5x the 9.223-token/s baseline allows about 219 ms
per round including draft/reconciliation, compared with roughly 701 ms in
verification alone. This budget selects operand delivery and projection
ownership, not an acceptance-rate or length sweep.

Stage one groups four output channels and M2..4 token rows in each projection
worker. FP8 reads packed four-byte words, consuming the existing AoSoA4
preswizzled output sidecar where attached. NVFP4 consumes the existing coupled
Gate/Up feed and Down consumer-order/scale6 inventory; ineligible Down layers
retain canonical bytes with vector reads. Weights/scales are decoded once for
all M rows; each activation load serves four output channels. The expression
remains four independent FP32 FMA chains, `(a0+a1)+(a2+a3)`, the same warp
and (FP8) eight-warp reduction, final tensor scale and BF16 publication. No
Tensor Core reassociation, new weight arena, persistent-state boundary or
approximate verifier is admitted. GDN/Conv, ordered Attention, snapshots and
lm-head remain the established exact operators at this stage.

The transfer follows the already studied vLLM/Marlin operand reuse and
consumer-order delivery principle, translated to SM87 vector loads and scalar
FMA to preserve Decode arithmetic. FlashInfer's query/KV reuse is a separate
remaining Attention opportunity; FLA/Triton and Mamba chunk composition cannot
silently replace this BF16 recurrent commit contract. The source assessment
in the [non-MTP architecture record](analysis/non-mtp-architecture-assessment-2026-09-28/README.md)
and pinned MTP references above remain reference-only.

The complete per-prefix scalar state/logit harness supplies minimum admission,
then the existing full-capacity service returns immediately to the same three
API buckets for drafts 2/3. One operand-dataflow repair is the stop-loss bound;
negative whole-path results close this version rather than trigger tuning.
Stage two may reduce the measured 3.24s/15.8s draft initialization only after
stage-one API evidence; it needs an explicit exact reduction/ownership ledger
before mutation. Artifacts live under `.q3x-work/mtp-efficiency-20260928/`.
The installed ordinary service is the rollback and remains unchanged until
separate production qualification.

The first API stage reduces v1 verification cost but does not beat non-MTP
across the tested contexts.
One same-ELF P65/O16 profile assigns approximately 108 ms/round to FP8 and
105 ms/round to NVFP4, about 83% of verification. The bounded operand repair
therefore moves speculative rows from each thread's accumulator array into
separate cooperating thread groups of the same CTA. Each group retains the
original scalar K/reduction ownership and four output channels. Read-only
cached loads let those groups consume the same adjacent weight lines; no new
layout or persistent memory is introduced. This trades repeated cache/decode
instructions for lower register pressure and more active warps; only the
complete API decides whether it is beneficial. It is the one allowed repair,
not the start of a launch-shape scan.

The second stage composes two exact draft changes at that same API return.
Initialization widens weight reuse from eight to 32 independent shifted rows,
keeping each thread's K-strided FMA sequence and the 256-thread binary tree.
Its construction-owned scratch is 1,310,720 bytes (983,040 additional bytes);
cancellation is checked before work and after every completed batch of at most
32 rows. The complete cache oracle includes 31/32/33-row boundaries.
Reconciliation appends only FC/input norm/K/V/K norm/RoPE for one target-hidden
row. The next proposal always starts from the selected target hidden, so the
replayed Q/Attention/O/MLP/final draft hidden has no later consumer. Those dead
outputs are not published; `Draft::step` remains the full-layer cache oracle.
The exact live K/V and position complete before each observer. Failure still
drains/poisons, and complete target-state/full-logit and draft-KV comparisons
remain required. Full drafting is unchanged. This composition returns directly
to P65/8K/40K for d2/d3 and cannot claim speedup before that return.

The one CTA-row operand repair regressed the first API sanity case and is
rejected. Its source, oracle and observed requests remain frozen. The final
composition restores the admitted stage-one vector-feed verifier and retains
the independent M32 initialization/cache-only reconciliation changes. This is
a bounded composition of already admitted mechanisms, not another projection
variant or parameter scan. Profile v4 identifies that composition and returns
to the complete d2/d3 API panel before any performance claim.

## Shared verification composition v3

`WP-MTP-SHARED-VERIFY-20261009` owns `AC-MTP-GREEDY-v3`. The previous
composition's 8K/40K d2 verification alone exceeds the entire 1.5x Decode
budget. Two coupled changes target that boundary, using the existing real API,
M2..4 verifier and all eligible layers. Production and the scalar oracle stay
unchanged. Artifacts belong under `.q3x-work/mtp-shared-verify-20261009/`.

The exact computation ledger is:

- Projection remains `RNE_BF16(scale * reduce((a0+a1)+(a2+a3)))`.
  Each `ac` consumes exactly the scalar K subsequence through FP32 `fmaf`.
  A single producer group decodes each quantized K tile into shared FP32
  operands. Separate token groups consume those operands before a CTA barrier
  permits overwrite. This removes repeated decoding without the rejected
  cached-load variant's duplicate codebook work. FP8 keeps eight-warp and
  NVFP4 one-warp reduction ownership. Additional shared stores/reads and
  barriers are charged to the complete verifier; no permanent weight copy or
  request allocation is added. The shared tile is CTA-owned, disjoint from
  the cross-warp reduction storage, and expires after all row consumers.
- Attention computes each row m on exactly `[0, entry+m+1)`. All Q/K/V
  preprocessing completes first on the same stream. QK and ordered PV grids
  interleave independent speculative rows over the same KV tiles, enabling
  cache reuse and concurrent query consumers. Every score reduction,
  per-row actual-length softmax, increasing-position PV FMA and BF16 output
  boundary is unchanged; later KV rows never enter an earlier causal domain.
  Probability storage aliases dead projection buffer 2 only during full
  Attention, after preceding GDN consumers and before MLP reuse. Its capacity
  must be checked for `rows * 24 * last_sequence * sizeof(float)`; otherwise
  the established scalar-row route is used before any batch enqueue. No
  partial failure falls back. Short-context arithmetic remains unchanged.

The source translation uses the pinned vLLM/Marlin separation of packed
operand delivery from consumption, and FlashInfer's grouped query/KV
scheduling. SM87 realizes these using ordinary shared memory, barriers and
existing asynchronous KV staging, without new-device ISA or changed tensor
arithmetic. The prior [reference assessment](analysis/non-mtp-architecture-assessment-2026-09-28/README.md)
also covers Triton/FLA and Mamba: their chunk reassociation does not satisfy
this exact recurrent prefix contract and is not introduced. This is a native
implementation, with no new imported source or runtime dependency.

The fresh-host resource gate exposed a construction-time reserve shortfall.
The composition therefore also removes duplicate transaction allocations in
whole-core mode: its five immutable slots and normalized prompt capture borrow
disjoint subranges of the existing `linear.prompt_wide_workspace`. Its producer
lifetime begins only after successful target Prefill commit. All Prefill
projection/GDN/Attention/MLP scratch consumers have completed then; scalar and
multi-row Decode use the physically disjoint C512 bundle and persistent state.
The next Prefill may overwrite the borrowed region only after the transaction
retires or abort drains. No arithmetic or live state is deleted. The public
prompt-hidden borrow expires on reset/new generation as before. Legacy mode
retains its independent snapshot allocation. Geometry, range and alignment
checks precede binding; destruction never frees request-owned storage. The
service still checks actual free memory against the unchanged 8-GiB reserve.
This is a bounded resource prerequisite in the same composition, not a reserve
waiver or another projection variant.

The [rejection record](metadata/qwen36-27b-mtp-shared-verification-rejection-2026-10-09.json)
closes both v3 projection implementations without promotion. The first
shared-FP32 operand composition is rejected after P65 and 8K API
verification time regressed. P65 does not execute batched Attention, isolating
the dominant change to projection delivery. A bounded Nsight attempt could not
start under the unchanged reserve with profiler overhead; no profile or kernel
attribution is claimed. The complete phase receipts and source traffic ledger
still establish a negative direction. Expanded shared operands add four bytes
per weight per producer and per row consumer, on top of codebook traffic.

The one permitted repair restores register-resident row reuse and replaces
random FP8/FP4 codebook accesses with exact register bit construction. FP8
uses the same existing 256-code function, undoing the output sidecar's
involutive swizzle first. E2M1 magnitudes 0/1 map to FP32 0/0.5; codes 2..7
map to `(252+code)<<22`, then the sign bit is restored, including negative
zero. Block scaling, all four FMA chains and reduction/publication stay fixed.
No expanded decoded tile crosses shared memory. Only the 256-entry NVFP4
block-scale table remains. This charges extra integer decode instructions
against removed shared lookups and barriers. The admitted batch Attention and
post-Prefill lifetime reuse compose with this repair in profile v6 and return
immediately to the same API; there is no further repair or launch scan in v3.

The smallest correctness gate is complete per-prefix state/full logits for
M2/3/4 plus existing rejection/cancel/failure transactions. Long Attention also
requires a P509 transition and P8192 same-history full-state oracle. The first
composed service returns immediately to the three existing API lengths for
both configured drafts, preserving startup reserve and reporting Prefill,
TTFT, committed Decode and external total. One initial dataflow plus at most
one causally explained correction bounds this version; a negative result
closes or redesigns it instead of triggering a parameter scan. Target attainment
and release qualification are separate from this reversible direction screen.

## Draft matrix composition v4

The shared-FP32 and register-bit-decode projections above both regressed the
API and are rejected. `WP-MTP-DRAFT-MMA-20261009` restores the earlier exact
vector verifier and composes the admitted query scheduling and storage reuse
with a separate BF16 draft executor. This is not another weight-decoding
repair. The 1.5x–3x target and complete verifier budget remain outstanding.

The draft computes `RNE_BF16(sum_k BF16(A)*BF16(W))` with native CUTLASS
SM80 Tensor Core FP32 accumulation, no split-K, and one fixed 32x128x32 CTA /
32x32x32 warp / 16x8x16 instruction mapping for M1..32. It consumes original
BF16 weights directly. No new packing, permanent weight copy, runtime
allocation, algorithm search or cuBLAS dependency is introduced. The existing
1,310,720-byte batch scratch, row normalization/RoPE and cancellation bounds
remain unchanged. Launch validation checks the complete shape, alignment and
disjoint spans before enqueue; CUTLASS nonzero workspace fails closed.

This changes the private draft reduction tree, identified as
`draft-bf16-mma-v1`; it does not claim old draft bit identity. Full draft steps,
cache-only reconciliation and batch initialization all use the same fixed K
order and MMA mapping, so their live K/V must agree exactly with each other.
The independent CPU FP64 hidden/K/V oracle retains its predeclared 0.02 bound.
Every main-model verification projection, recurrence, state snapshot, full
logit and greedy commit remains exactly the existing scalar computation.
Draft differences may change acceptance/work, but no unverified prediction
reaches the API. The normal target-output and complete transaction checks
remain mandatory. One executor implementation returns promptly to the existing
real API panel for d2/d3; no performance claim precedes that composition.

## Cooperative verifier composition v5

`WP-MTP-COOPERATIVE-VERIFY-20261009` / `AC-MTP-GREEDY-v5` is the bounded
successor to draft matrix composition. Its d2 API receipts still charge
19.32/25.73 seconds to 8K/40K verification. The complete 1.5x budgets are
18.45/21.62 seconds; after measured proposal/reconciliation, verification
must fit approximately 15.7/17.9 seconds. This selects two residency changes,
not a draft-length or launch-parameter sweep:

- In projection, speculative rows become subgroups inside each warp. One
  subgroup reads/decodes packed weights and broadcasts them with warp shuffles.
  Each thread owns one token's four output channels and four original FMA
  chains. Original K-lane identities are retained explicitly, then complete
  chain merges are staged once at the end and reduced in the original warp
  and FP8 eight-warp tree. There is no per-K-tile expanded shared-weight
  publication or CTA barrier. This trades shuffle and final-partial movement
  for fewer live accumulators and shared codebook conflicts. M2 uses two
  token groups; M3/M4 use four with the fourth M3 group masked. Scale6
  extraction preserves the exact packed bit fields without cross-token state.
- Ordered Attention PV retains all speculative queries inside one CTA so each
  V tile is physically loaded once for their six grouped query heads. Every
  row has its own FP32 probability and two independent FP32 output chains,
  consuming positions in exactly increasing order within its actual causal
  end. Three physical buffers stage two future groups; M4 uses at most
  43,008 shared bytes. Tail masks exclude future KV values before FMA.
  QK, actual-length softmax, final BF16 rounding and public state are unchanged.

This transfers the studied Marlin warp operand reuse and FlashInfer grouped
query/V residency into native SM87 shuffle/cp.async primitives. FLA/Mamba
recurrence remains unchanged. The two source-private kernels form one coupled
candidate with the admitted draft matrix path; no new persistent allocation
or weight sidecar is added. Minimum admission is all nine exact scalar prefix
state/logit comparisons plus the long Attention oracle. One initial composition
and at most one correctness repair return immediately to P65/8K/40K d2/d3 API.
A negative direction archives this version rather than opening a parameter scan.
Production and independent non-MTP baselines are unchanged.

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

The isolated [transaction verifier](../src/runtime/mtp_verify_internal.cpp)
implements a layer-major verifier for at most four rows. Dedicated FP8/NVFP4 projections reuse weights across rows and preserve
scalar Decode's four independent accumulation chains, parenthesized merge,
warp/block reduction, final scale and BF16 publication. The generic small-M
projection dispatcher has a different one-chain reduction and is not an exact
Decode verifier. BF16 A/B and causal Attention retain the scalar numerical
path. Conv/GDN updates remain token-ordered, with each layer's complete state
copied into its corresponding immutable prefix slot after every update. Slots
are assembled across all 64 layers and gain no publication authority until
normalized hidden, full logits and finiteness checks have also completed.
Existing request-owned C512 scratch supplies the bounded row buffers, with no
new request-time allocation. The scalar verifier remains an explicit oracle.
Every prefix for M2/M3/M4 must match scalar state/full logits before service
composition; existing Prefill equivalence is not assumed to qualify this path.

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


## Native scalar correctness backend

The source-private [device implementation](../src/runtime/mtp_device_internal.cpp)
is linked into explicit MTP device tests and the isolated service admission.
Its `Weights` owner validates the
catalog, requires each tensor at its compiled source offset relative to byte
59,416 of shard 3, opens every root component and the shard without following
symlinks, and copies/hash-authenticates the same sequential bytes. All of shard
3 must match the existing pinned full-file SHA-256, exact size and stable file
metadata before the owner is returned. This separate 849,398,784-byte arena
contains only MTP. The base owner continues to authenticate all three shards.
The ordinary resident loader, text arena and installed ABI are unchanged.

`Draft` borrows the base embedding/lm-head, MTP weights and target RoPE tables.
Those owners must outlive it. Its single bounded workspace includes its own
K/V. The draft matrix composition supplies private BF16 projections; existing
normalization, RoPE, Attention, MLP and argmax operations remain unchanged. Every step completes on its owned stream before position
publication; there is no allocation or implicit fallback in a step. Reset
clears complete draft K/V and poison. Rewind changes only logical draft length;
rejected rows are inaccessible and overwritten before reuse. `hidden()` is a
borrowed last-step value, invalid for use after reset, rewind, cache-only append or failure until
a new successful step. Prompt initialization consumes shifted normalized target hidden rows. The
whole-core adapter below adds prompt-wide capture and bounded batched live-KV
initialization; scalar full steps remain the independent cache oracle.

`TargetTransaction` binds one exact Legacy-C512 or whole-core runner/state
pair using SM87 quantized-lm-head BF16 logits. Its explicit multi-row mode
requires at least four rows of validated request scratch. Scalar mode remains
the oracle; multi-row mode assembles the same complete prefix slots layer by
layer, preserving token-ordered Conv/GDN and causal Attention. It binds five immutable prefix
slots, each 78,446,592 recurrent/Conv bytes, 10,240 hidden bytes and 496,640
full-logit bytes. Restoring logits prevents a rejected later speculative row
from remaining visible at an earlier accepted boundary. Target KV is append-only
within a round, so prefix selection preserves its earlier rows and publishes
only the accepted logical length. The adapter copies the selected recurrent
and hidden snapshot, rebuilds one draft row from the corresponding target
hidden, and completes both streams before publishing that token. It never
borrows the ordinary engine's successful-request prefix-reset authority.
Abort drains execution, poisons target and draft, and requires explicit full
reset. Both owners must outlive the adapter, including destruction/abort.

The [device harness](../tests/mtp_device_test.cpp) compares complete live target
Conv/GDN/K/V/final-hidden bytes and draft K/V against independent scalar replay,
including scripted full acceptance, every rejection position, cancellation
and failure after target verification. Scripted proposals are test inputs,
never checkpoint acceptance measurements. The
[CPU oracle](../tools/evaluation/validate_mtp_draft.py) separately evaluates the
whole MTP layer with FP64 projections/causal Attention and BF16 publication
boundaries on captured real target hidden. Its predeclared maximum per-row
relative-L2 bound is 0.02 for draft hidden/K/V; this bounds a draft-layer check
and does not relax the target verifier's bitwise state/output contract.

Exact device results and limitations are recorded in the
[device milestone](metadata/qwen36-27b-mtp-device-2026-09-28.json). These checks
establish the scalar correctness substrate only. The subsequent multi-row
implementation does not inherit API acceleration authority from these tests.
Synchronous snapshot restoration and target-conditioned cache-only draft
reconciliation remain in the composition budget. Service integration,
API receipts, cancellation/stream accounting and real API selection
remain required before the architecture can be selected.


## Whole-core prompt handoff

### Batched draft-cache initialization work package

Within `WP-MTP-20260928`, the next dependency removes serial full-layer
draft initialization before service composition. The owner explicitly requires
completing these prerequisites before API integration; a short-context scalar
service is not a substitute for the planned architecture.

For prompt rows t<P-1, the only live draft outputs are
`K[t]=RoPE(Norm_K(W_K Norm_input(W_fc concat(Norm_E(E[token[t+1]]),
Norm_H(target_hidden[t]))))))` and the corresponding V projection before
K normalization/RoPE. Each input hidden comes from the target, not the previous
draft output. Query/gate, causal Attention, O, residual/MLP and final draft
hidden therefore have no path to a later live value during this initialization.
They remain mandatory during proposal steps. Reconciliation now uses the
same live-KV deletion described in the efficiency composition above. No draft hidden
export is valid after cache-only initialization until a successful full step.

The implementation batches at most 32 independent rows. The original FC/K/V
executor retained scalar 256-thread strided FMA and binary reduction. The
active draft matrix composition above replaces that private draft arithmetic
with one fixed MMA reduction for both batch and scalar execution, preserving
BF16 publication and the independently checked draft error bound. Normalization
and RoPE keep their existing per-row kernels and absolute positions. Dedicated
1,310,720-byte construction-owned scratch does not alias scalar or cache storage.
Each batch completes on the draft stream before its length is published;
cancellation is polled before the first batch and after each completed batch.
Cancellation/failure drains and poisons both owners through the transaction;
complete reset is required for reuse. No request-time allocation is permitted.

The bounded gate compares complete K/V against full scalar draft replay on
real whole-core prompt hidden, including masked tails, next-row canaries,
cancellation and reset, then reuses the complete target-state transaction
panel. This is one dependency implementation, not a tuning sweep. Its value
returns at the already-declared multi-row verifier plus service/API composition;
local initialization time cannot select MTP or change production.

The source-private `EngineAccess` definitions exist only with internal test
seams. They borrow the engine's exact model, runner and state; they add no
installed API, owner fields or production selector. All owners must outlive
the draft/transaction, and generation and peer operations remain serialized.
The explicit `q3x_mtp_whole_core_test` requires the corrected whole-core/exact
Decode admission. It uses the engine's ordinary O1 whole-core generation to
complete and validate Prefill before invoking the peer.

`initialize_whole_core_prefill` requires that completed prompt boundary, its
exact token IDs, no active transaction, and a non-poisoned runner. It binds
one normalized `[capacity,5120]` BF16 capture at construction. In the shared
verification composition this and the snapshot slots borrow the dead
post-Prefill workspace described above; the request state remains their owner. The
complete layer-63 residual is still live; terminal-prefix deletion cannot be
used for this mode. The adapter widens the existing independent row-wise final
RMSNorm launch to P rows without changing its per-row reduction or BF16
publication. It copies the final normalized row to the scalar hidden workspace,
then initializes draft positions 0 through P-2 from hidden[t] and token[t+1]
using the batched live-KV path above.
The first round supplies the seed at draft position P-1. Persistent target
state, prompt residual and full first-token logits are preserved. No allocation
occurs during initialization or a round. Failure after enqueue drains and
poisons both participants; recovery requires complete reset.

The capture is borrowed scratch, valid only immediately after successful
initialization until another initialization, reset, transaction or generation.
It is not a public hidden-export API. The adapter now initializes only live
draft K/V in bounded batches, with cancellation before execution and after
each batch. It remains an isolated prerequisite; service readiness additionally
requires composing the verifier with the service controller and evaluating
the full-context API path.
The complete numerical/state and resource checks do not themselves establish
API speed, long-prompt admission or production eligibility.

The [whole-core handoff record](metadata/qwen36-27b-mtp-whole-core-2026-09-28.json)
binds the exact P65/O16 harness artifact, complete state/full-logit comparisons,
CPU prompt-normalization/draft oracle and unchanged production boundary.

## Isolated service composition

`Q3X_BUILD_MTP_SERVICE_ADMISSION=ON` binds the existing corrected whole-core
Prefill, batched shifted draft cache and exact multi-row verifier into the
ordinary generation controller and HTTP gateway. It requires testing, excludes
production/install, and identifies itself as
`q3x.sm87.admission.mtp-multirow-api.v7`. The startup-only
`Q3X_MTP_DRAFT_LENGTH` must be exactly 2 or 3. Capacity remains
`P+O-1<=44095`, O1..4096, with the complete target acceleration inventory and
an additional post-composition 8-GiB free-memory check.

The engine owns the service after its model/runner/state owners and destroys
it first. A serialized thread-local scope binds only that engine's generation
control. The handoff runs after whole-request state commit and before first
observer publication. At this internal point the request-level route record
is still active: a private service-only handoff accepts it only with a completed
layer pass, no route/boundary error, exact committed prompt length and no
active whole-request stage. The ordinary completed-request harness boundary
remains supported. The engine finalizes the original Prefill record after
generation; MTP never fabricates a non-MTP witness.

Total Prefill includes normalized prompt capture and draft-cache initialization.
Decode measures elapsed time through all rounds, including proposal, snapshots,
verification, reconciliation and observer work; its numerator is committed
output minus the first token. Per-token SSE timestamps retain real bursts.
`mtp-multirow-api-witness-v1` binds the exact request and prompt hashes, actual
output IDs/count, target Prefill physical counts, proposed/accepted/verified
rows, per-position acceptance, phase costs and startup free bytes. No proposal
is counted as output before complete prefix selection. Cancellation returns
through existing engine/gateway handling; uncertain initialization poisons and
requires full reset. A failed transaction terminates the request/service through
the existing fatal-health boundary.

The owner now prioritizes completing this chain before further local tuning.
The [API driver](../tools/evaluation/validate_mtp_service.py) returns each stage's
Prefill, external TTFT and committed Decode immediately, with a matched ordinary
service baseline. P65/O16 is the first integration stage, then P8192/P40000
O256 for both draft lengths, followed by bounded stream/text/chat/cancellation
checks. These single-process observations select an engineering direction,
not production promotion or a statistically repeated performance baseline.

Reproduce the isolated service (all build/cache artifacts stay in the workspace):

```bash
cmake --preset orin-mtp-service-admission
cmake --build --preset orin-mtp-service-admission --parallel 3
Q3X_MTP_DRAFT_LENGTH=2 \
  .q3x-work/build/orin-mtp-service-admission/qwen3x-eval-server-mtp-admission \
  MODEL_DIR --candidate-profile whole-core-exact-decode --port 18080
```

Use a separate ordinary whole-core admission build for the scalar/device
harnesses; they are excluded from this service build to avoid attaching a
second independent MTP owner to an already composed engine.
