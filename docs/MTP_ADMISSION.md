---
q3x_document:
  id: q3x-mtp-admission
  class: active
  status: active
  owner: runtime-maintainers
  authority: isolated greedy MTP development boundaries and transaction contract
  effective: 2026-09-28
  last_reviewed: 2026-10-10
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

The [cooperative rejection](metadata/qwen36-27b-mtp-cooperative-projection-rejection-2026-10-09.json)
archives this projection after negative P65 API direction; the following v6
composition was its shared-V dependency expiry point.

## Accumulator ownership composition v6

The cooperative verifier preserves all P65/P8192 state/logit bits but its
warp-broadcast projection regresses P65 Decode from 10.72 to 3.90 token/s;
that version is rejected and removed. P65 does not execute the shared-V
kernel, so it cannot select or reject that independent mechanism. The exact
shared-V implementation has only dependency authority until this next API
composition; its local numerical evidence is not a performance win.

`WP-MTP-ACCUMULATOR-OWNERSHIP-20261009` keeps every speculative token's four
FMA chains in the same thread, as in the admitted vector verifier. For M3/M4,
each thread owns two output channels instead of four, splitting each original
packed output quad into two aligned uint2 consumers. Canonical weights use two
uint32 loads; coupled/consumer-order layouts retain the exact original bytes
and scale fields. Adjacent output-pair CTAs can reuse packed cache lines.
No token-group broadcast or per-K shared-weight staging remains. Each weight
still serves every speculative row; only independent output ownership changes.
Activation reads may increase, so lower registers alone cannot select it.
M2 retains four output channels. Four chain merges, warp/eight-warp reductions,
final tensor scales and BF16 publication are unchanged for every output.

One fixed implementation (plus at most one correctness repair) composes with
the exact shared-V kernel and draft MMA; complete scalar prefix state/logits
admit it before P65/8K/40K d2/d3 API selection. The same 1.5x–3x goal and
15.7/17.9-second verifier budgets control the decision. Negative whole-path
fitness closes this version without a parameter sweep. No production route,
accuracy bound, capacity or retained-free gate changes.

The [accumulator-ownership rejection](metadata/qwen36-27b-mtp-accumulator-ownership-rejection-2026-10-09.json)
closes the complete d2 API panel as negative versus retained v7. Both the
output-pair projection and shared-V implementation were removed.

## Packed FP8 and draft Attention composition v7

Output-pair ownership plus shared-V regresses d2 API Decode at all three
contexts and is archived. The restored v7 vector-quad/interleaved-query
composition is the incumbent for `WP-MTP-PACKED-FP8-20261009`.

The pinned Marlin `dequant.h` in vLLM commit
`ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb` maps E4M3 bits into FP16 and
multiplies by exactly 256 to correct the exponent bias. Both FP8 subnormals
and all finite normal values are represented exactly in FP16, then exactly
converted to FP32. The private verifier implements this in two-value pairs
with native half2 multiplication, replacing scalar shared-codebook lookup.
The speed-only reference's bias folding into a tensor scale is not used:
weights are restored to their full exact values before every existing FP32
FMA, and the original tensor scale stays after reduction. FN NaN codes are
explicitly restored to the original signed quiet-NaN bits; signed zeros are
preserved. Every two-code combination is checked on-device against independent
host formulas before the real-model oracle. Existing preswizzled FP8 O bytes
are unswizzled first. NVFP4 codebooks and arithmetic are unchanged.

The other dependency binds draft Attention at S512..44095 to the already
qualified production ordered GQA operation. Q/K normalization, RoPE, causal
span, FP32 QK/softmax/PV arithmetic, BF16 output and subsequent sigmoid gate
stay identical. It borrows the existing draft probability scratch on the same
stream; no allocation or fallback after enqueue is permitted. Short draft
Attention retains the public reference. This transfers the production KV
residency to the real draft critical path rather than changing model precision.

The two dependencies return together to the same API panel after exhaustive
encoding and complete prefix state/logit checks. The 1.5x–3x objective remains
unchanged; one initial composition and one correctness repair bound the work.
Neither component observations nor preserved generated text select production.

The [packed-conversion rejection](metadata/qwen36-27b-mtp-packed-fp8-rejection-2026-10-09.json)
closes this direction: every encoding/state check passes, but P65/8K slows
and 40K is effectively tied. Draft ordered Attention had dependency authority
only through the immediately following composition.

## Full-K operand residency composition v8

`WP-MTP-FULLK-RESIDENCY-20261009` moves FP8 operand lifetime across the entire
K loop. A CTA decodes canonical FP8 weights for four K5120 outputs or two K6144
outputs into exact BF16 shared storage. A complete producer barrier precedes
all consumers. M independent 256-thread groups then execute one token each:
original K-thread identities, four FP32 chains, original merge, warp and
eight-warp reduction, tensor scale and BF16 publication. The final reduction
uses disjoint shared partials. Complete operand storage is at most 40,960
bytes plus codebook/partials, below the ordinary 48-KiB static limit; there
is no runtime shared-memory attribute change. Group size derives from K
capacity, not a performance parameter scan. Canonical source weights are
already authenticated and resident; output sidecars remain attached for the
scalar oracle and other consumers.

Compared with rejected v3, operands are BF16 rather than FP32, there is no
inner-K producer barrier, and each token consumer has only its own output
accumulators. Compared with v5, there are no warp-broadcast instructions in
the FMA loop. This transfers full-operand residency/producer-consumer ownership
from the studied Marlin dataflow while retaining exact scalar arithmetic.
The cost of shared publication, rereads and lower CTA concurrency is charged
to the full API; reduced registers alone cannot select it. NVFP4, recurrence,
causal target Attention and transaction ownership stay incumbent. Admitted
production ordered draft GQA remains a coupled dependency only. Complete
prefix state/logits admit one fixed composition before the same d2/d3 API
panel. The 1.5x–3x target and whole-round budgets remain unchanged; at most
one correctness repair is permitted and a negative composition ends the
version. No production numerical or routing contract changes.

The [full-K rejection](metadata/qwen36-27b-mtp-fullk-residency-rejection-2026-10-09.json)
closes this version after exact state/logit admission and negative P65 API.
Neither new projection nor draft Attention remains selected. The retained
implementation is the v7 draft matrix composition; the active bottleneck reset
and next composition decision are owned by [Roadmap](ROADMAP.md).

## Persistent activation and lookup composition v9

`WP-MTP-PERSISTENT-OPERANDS-20261010` retains each packed weight once per
M-row register consumer while extending activation/codebook lifetime across
output groups. FP8 CTAs stage all M BF16 activation rows once and use a
32-lane-replicated FP32 codebook: index `32*code+lane` has conflict-free bank
ownership for all codes. It does not decode or store an expanded weight tile.
A fixed 32-CTA grid traverses disjoint four-output groups with the same
256-thread K ownership and four-chain/eight-warp reduction. Dynamic shared
storage is at most 81,920 bytes, plus disjoint static partials. Device capability
and attributes are bound at transaction construction, before execution;
launches allocate nothing. Replication/init cost and residency are charged to
the complete API; the unsuccessful counter collection does not establish
that bank conflicts are the measured cause.

NVFP4 K5120 uses a fixed maximum 64-CTA persistent output loop and stages M
complete activation rows once. The current production lm-head already uses
this activation-lifetime pattern for M1; the native M2..4 extension preserves
all original K subsequences, FP32 block multiplication, FMA chains, warp sums,
final tensor scale and BF16 publication. Down K17408 retains the incumbent
executor. A batched canonical NVFP4 lm-head reuses its weights across M target
hidden rows, publishing contiguous complete logits into dead projection-0
scratch and then copying each complete row into its immutable prefix slot.
The buffer capacity is checked before dispatch; smaller non-service test
arenas retain scalar lm-head before enqueue. Argmax and non-finite validation
remain per complete vocabulary. No persistent target state aliases the logit
scratch, and no partially completed slot gains commit authority.

The admitted ordered draft GQA transfers the same exact production arithmetic
to S512 and above. It shares no stream or cache state with target execution.
These dependencies form one bounded composition, with complete scalar-prefix
state/full-logit checks before the same d2/d3 API panel. One implementation and
at most one correctness repair are permitted; no parameter scan is opened.
The target and resource reserve are unchanged, and production remains non-MTP.

The [persistent-operand rejection](metadata/qwen36-27b-mtp-persistent-operands-rejection-2026-10-10.json)
closes this composition after complete P65 numerical admission and negative
P65 API direction. All new runtime paths were removed together; no component
receives a separate performance claim. Retained v7 remains the isolated
incumbent, and the 1.5x–3x goal remains outstanding.

## Certified-publication feasibility boundary

The closed Roadmap package used a numerical-only observation seam in the
frozen whole-core test executable. It was absent from serving and was removed
after the bounded capture; the frozen source is retained in its evidence. After a successful projection enqueue the callback synchronizes the
owned stream, copies only selected real inputs, canonical weight/scale rows and
BF16 results, then returns before scratch reuse. Capture failure fails the
verification and drains through the existing poison boundary. It cannot alter
device data, selected tokens or request state. Its altered timing has no
performance authority. The ordinary complete scalar-prefix comparison still
checks the captured transaction.

A prospective fast reduction may publish only when a conservative interval
containing the original scalar result lies strictly inside one BF16 rounding
cell. Otherwise it must execute the original scalar operation for that output.
The current host feasibility audit uses a double-precision center and a
conservative FP32 reduction-depth bound; it implements no fast GPU path and
grants no certificate to production. Exceptional values, subnormal arithmetic,
scale rounding, candidate error and correct-or-repair execution must all be
covered before an executor is admitted. A favorable sample is not proof of
universal certification or positive API performance.

The [feasibility record](metadata/qwen36-27b-mtp-certified-publication-feasibility-2026-10-10.json)
closes the coarse norm-bound design without a new GPU executor. It does not
exclude a different proven certificate and does not weaken scalar equivalence.

## Independent token CTA composition v10

Each FP8 output quad or NVFP4 output group uses separate 256/128-thread CTAs
for the M2..4 token rows. The block index interleaves token rows before moving
to the next output group. Each block's activation/output pointers select one
row; packed weight/scale pointers select the shared group. Four FP32 FMA chains,
K subsequences, warp/eight-warp merges, scale and BF16 rounding are unchanged.
No block reads another block's scratch, signals a peer, or depends on cache
contents for correctness. The cache can miss without changing semantics.
Only the scheduler/cache may reuse packed weights between independently live
blocks. Per-block codebooks and reduction storage remain private and bounded.
The admitted ordered production draft GQA is included at the same service
composition point. Complete prefix/transaction oracles and the existing
real API d2/d3 panel select the result; lower register counts alone do not.

The [independent-CTA rejection](metadata/qwen36-27b-mtp-independent-ctas-rejection-2026-10-10.json)
closes this version after complete P65 numerical admission and negative real
API direction. All new runtime paths were removed. Lower register counts and
independent scheduling did not establish whole-product value.

## Asynchronous packed-feed composition v11

Three CTA-private packed buffers hold the current and two future K blocks.
FP8 uses 1,024 K values per block, NVFP4 uses 512; each buffer is 4,096 bytes.
Canonical four-byte streams use aligned cp.async.ca copies and existing
16-byte sidecar vectors use cp.async.cg. No expanded decoded weight crosses
shared memory. Scale/codebook lookup, activation loads and original M-row
FP32 accumulation remain unchanged. Each thread owns its packed destination;
commit/wait plus a uniform CTA barrier precede consumption and reuse. Tail
waits drain all outstanding copies before shared storage expires. There is no
inter-CTA dependency or request allocation. All admitted K values contain at
least two complete blocks. The original final reduction and BF16 publication
remain exact; numerical and real API gates select the complete composition.

The [asynchronous-feed rejection](metadata/qwen36-27b-mtp-async-packed-rejection-2026-10-10.json)
closes this composition after complete numerical admission and negative P65
API direction. All new runtime paths are removed; retained v7 remains selected
only in isolated development. No pipeline sweep or production change follows.

## Bounded reduction and sparse repair composition v12

The candidate applies only to FP8 and NVFP4 K5120 Gate/Up M2..4. Canonical
quantized weights decode exactly into FP16: all finite E4M3 and E2M1-times-E4M3
values fit its precision/range. Each BF16 activation is multiplied by 256
exactly, converted to FP16 RN, and any FP16 subnormal is replaced with zero.
A second positive operand rounds the absolute conversion residual upward to
FP16, using its smallest normal for nonzero smaller residuals. Inputs outside
zero or `[2^-32,128]`, invalid weights/scales and exceptional intermediates
force exact repair. Nonfinite operands are sanitized only inside the untrusted
fast calculation; the original repair preserves their scalar behavior.

Each K16 FP16 MMA starts with zero FP32 accumulation. Shared operands use
a fixed 16-half row skew, preserving 32-byte row alignment while avoiding
the unskewed K128 row-bank alias. This is the fixed initial composition, not
a timing-selected layout sweep. The
[NVIDIA PTX contract](https://docs.nvidia.com/cuda/archive/12.6.3/parallel-thread-execution/index.html#warp-level-matrix-instructions-mma)
provides at least single-precision product and accumulation, without promising
an order or rounding mode. Products here fit FP32 exactly. A conservative
`gamma(64,2^-24)` covers sixteen arbitrary-rounding adds, three RN binary-tree
levels within K128, at most six RN partition-tree levels, and final scaling.
A separate MMA with absolute weights computes both absolute fast products and
conversion-error products. Dividing each positive sum upward by `1-gamma`
bounds its exact sum; dividing by 256 is exact in the admitted range.
The original BF16/quantized product lattice and guarded exponents exclude
FP32 subnormal intermediate sums and overflow. The scalar gamma depth remains
`K/1024+11` for FP8 and `K/128+8` for NVFP4. Tensor scaling stays after reduction.

Let A bound the absolute fast products, R bound the conversion-error products,
and Gf/Gs be fast/scalar gamma. The outward error radius
`scale * ((Gf+Gs)*(A+R)+R)` encloses the original scalar result around the fast
scaled candidate. Only a finite normal candidate whose complete closed error
interval lies strictly inside one BF16 midpoint cell may publish. Zero,
subnormal, overflow, NaN and boundary-touching candidates always repair.
Repair executes the unchanged four FMA chains, merge, warp/eight-warp reduction,
tensor scale and BF16 rounding for each uncertified element on the same stream.
The complete projection is visible only after repair. No statistical tolerance,
argmax-only comparison or sampled certificate substitutes for this runtime proof.

At most 64 MiB scratch follows the aligned immutable snapshot and prompt-hidden
ranges in whole-core post-Prefill GDN workspace. It is disjoint from live state
and C512 buffers, reused serially per projection, and expires before the next
Prefill. Legacy/no-scratch transactions keep the existing exact implementation
before enqueue. Full scalar-prefix state/logit admission and real API selection
remain required; neither the proof nor reduced arithmetic implies a speedup.

The [certified sparse rejection](metadata/qwen36-27b-mtp-certified-sparse-rejection-2026-10-10.json)
closes this composition after exact synthetic/real-state admission and negative
P65 API direction. The measured fast partial generator dominates its cost;
certification feasibility is not an acceleration result. All new paths and
scratch bindings are removed. The frozen proof/code remain reproduction evidence
only, and the isolated retained v7 route remains the incumbent.

## Streamed certified execution composition v13

One CTA owns 32 output channels across the complete K span. Four warps use
native FP16 m16n8k16 with zero FP32 initial accumulators, the established
row-major A and canonical `[N,K]` column-major B ldmatrix coordinates, and
fixed 16-half shared row skew. Packed vector loads decode eight weights per
producer. Signed and absolute/residual MMAs keep the same operands and guards
as the preceding proof. Only accumulator entries for logical rows 0..7 are
live; the other eight rows are padding, not target or certificate outputs.

Within K128, eight K16 leaves combine through three explicit RN tree levels.
Across the at most 48 real K128 tiles, zero padding to 64 leaves and a six-level
binary stack preserve the same conservative gamma(64) bound. Each thread owns
its stack cells; no peer reads them. A CTA barrier retires all shared operand
consumers before the next producer overwrite. The final live error row reaches
its matching target row by a fixed same-warp shuffle. Certification publishes
only the proved BF16 value and a byte mask directly; no global partial matrix
or separate merge/certificate pass exists. Uncertified elements execute the
original scalar tree with vector packed/activation loads, after the producing
kernel completes on the same stream.

At most 1 MiB after the aligned post-Prefill snapshots/capture stores the mask;
its actual maximum is 69,632 bytes. Full-K ownership introduces no persistent
weight expansion, allocation, inter-CTA dependency or new numerical tolerance.
The exact proof/source and real-path checks of the previous version are
prerequisites, not a transferred performance claim. Only this complete
composition's real API result selects retention.

The [streamed-certificate rejection](metadata/qwen36-27b-mtp-streamed-certificate-rejection-2026-10-10.json)
closes this composition after all 30 synthetic cases, complete P65 prefix
state/full-logit and transaction/recovery checks, and negative API direction.
No new path or mask binding remains selected; frozen proof/source is evidence
only. Retained isolated v7 and the production non-MTP route are unchanged.

## Direct fragment delivery composition v14

A projection-wide preparation kernel converts the same guarded BF16 inputs to
scaled FP16 and upward residuals once, retaining eight rows (four target and
four error, with unused rows zero). The complete array and one invalid flag
precede the consumer on the same stream. They occupy aligned disjoint ranges
after the output mask, within the 1-MiB borrowed workspace. Preparation and
repair always consume original inputs; no target activation is overwritten.

Each warp owns eight output channels. Lane group `lane/4` addresses the
canonical weight row, and `2*(lane%4)` addresses two adjacent K operands;
the second pair is eight K positions later. Those are the native m16n8k16
column-major B-fragment coordinates. A fragments load the corresponding target
or residual row directly from prepared storage, with the upper eight matrix
rows explicitly zero. FP8 pairs restore the exact exponent bias with FP16
multiplication by 256. FP4 pairs restore bias by 16384 before multiplication
by the separately decoded exact FP8 block scale. Every finite result is exactly
representable in FP16. Nonfinite codes poison the certificate and are sanitized
only in untrusted fast operands. Original scalar repair remains authoritative.

This transfers the pinned Marlin representation principle already studied in
v7, but connects direct canonical loads to MMA registers rather than scalar
SIMT or shared expanded tiles. No upstream source is imported. The zero-start
K16 signed/absolute calculations, balanced carry tree, gamma bound, strict
BF16-cell test and vector exact repairs stay unchanged. The original scalar
full-prefix oracle and same-request API remain the selection boundaries.

The [direct-fragment rejection](metadata/qwen36-27b-mtp-register-certificate-rejection-2026-10-10.json)
closes this version. Exhaustive pair conversion, complete P65 state/logits and
transaction/recovery checks pass, but real API direction is negative. The
fast executor and repair both retain material cost. No new path remains
selected; proof and source are frozen evidence, not production qualification.

## Partitioned execution and compact repair composition v15

The direct packed-to-fragment conversion is retained, but one independent CTA
owns N32/K128 and publishes signed, absolute-product and residual partials.
There is no cross-CTA wait or mutable target output from a producer. A separate
same-stream merge uses the identical six-level zero-padded binary tree; its
certificate retains the original gamma and strict BF16-cell proof. An invalid
input or local weight marks that output's absolute bound infinite, forcing
original scalar repair. The preceding exhaustive decoder proof remains relevant;
complete real-prefix checks still admit the composed executor independently.

Each certificate warp ballots failed cells, atomically reserves exactly that
many worklist entries once, and writes each rejected output's unique index.
The list has capacity M*N, so the all-repair case is bounded. The following
kernel starts only after all producers and the certificate kernel finish on
the same stream. Its 64 fixed CTAs consume disjoint grid-strided list indices;
there is no host count readback, dynamic launch or unfinished-list consumer.
FP8 uses one original 256-thread reduction per listed output; NVFP4 uses one
original warp reduction per listed output. Shared codebooks persist across
that worker's list traversal. FP8 barriers retire partial storage before reuse;
NVFP4 warps have independent lists and no post-initialization CTA barrier.

At most 64 MiB after immutable snapshots and prompt capture stores prepared
inputs, three partial arrays, byte masks, M*N indices and initialized counters.
These ranges are aligned/disjoint, expire before new Prefill and allocate
nothing during a round. Original inputs remain available for all exact repairs.
There is no numerical waiver, altered target recurrence or production switch.
Only the full composition's real API result may select retention.

The [partition/repair rejection](metadata/qwen36-27b-mtp-partition-repair-rejection-2026-10-10.json)
closes this version after exact worklist/output and complete real P65 state,
logit and transaction/recovery checks, followed by negative API direction.
All new paths are removed. No further certificate/executor/worklist variation
is active, and retained v7 plus the production non-MTP route are unchanged.

## Shared multi-query KV composition v16

This composition retains v7's exact projection executor and full target state
transaction. It changes only multi-row Attention at S512 and above, plus the
previously admitted ordered draft GQA dependency. The studied FlashInfer
multi-query KV ownership transfers through SM87 cp.async and ordinary CTA
barriers; no different dot-product, softmax or PV reduction is introduced.

QK uses a 64-position tile. For each speculative row, eight warps represent
four independent positions and two groups of three heads. This bounds resident
query operands to 24 FP32 values per thread; the fixed M2/M3/M4 CTA contains
512/768/1024 threads. Only the producer subset stages four K positions per
iteration into a four-buffer ring. All row/head consumers finish before reuse.
Every row consumes positions below its own causal end, preserves the exact
8-product and warp reduction tree, and publishes its scores into a disjoint
row/head tile before coalesced writes. The three-buffer lookahead uses the
latest row's extent only for safe staging; it never extends an earlier row's
arithmetic domain. Actual-length softmax remains per row.

PV uses 32 output dimensions and six query-head warps per CTA. All speculative
rows have independent FP32 accumulators in each thread. A four-buffer ring
stages each V slice once and the distinct rows' FP32 probabilities. Every
accumulator consumes positions in increasing order, skipping future positions
before FMA, and publishes BF16 exactly once. Shared storage is at most 40,960
bytes; 32 CTAs cover four KV heads and eight dimension slices. There is no
inter-CTA dependency, new request storage, probability precision change or
request-time allocation. Existing scratch capacity/alias checks remain intact.
The complete same-history prefix/state/logit and real API checks select the
composition; the previous losing projection combinations do not confer a
separate positive or negative Attention performance claim.

The [shared-KV rejection](metadata/qwen36-27b-mtp-shared-kv-rejection-2026-10-10.json) closes this composition after complete
P513/P8192 numerical admission and the three d2 context API requests. All new
paths are removed; physical KV sharing alone did not improve complete Decode.
The isolated retained implementation remains v7, with no production change.

## Exact PV work elimination composition v17

For one ordered FP32 accumulator `a`, let `B` be an upward-rounded bound on
`abs(p*v)` for every position in the next 64-position causal block. For finite
normal `a` with biased exponent E>25, `2^(E-127-25)` is no larger than half
the distance to either adjacent FP32 value, including powers of two. If
`B` is strictly smaller, every RN FMA returns exactly `a`, by induction across
the complete block. Equality never skips. Zero, subnormal, small-exponent,
infinite and NaN accumulators never skip. Nonfinite P or V forces an infinite
bound; no exceptional operation is removed. This uses a conservative strict
rounding-cell proof, not a numerical tolerance or an approximate attention mask.

An initialized `[16,4,ceil(capacity/64)]` FP32 array bounds absolute BF16 V
for each target layer, KV head and position block. It is constructed from all
actual prompt values after Prefill commit and reset on every new initialization.
Each later projection updates the affected block maxima on the target stream.
Rejected rows may leave a larger maximum: that remains conservative and never
changes model state or the causal domain. No subtraction on rewind is allowed.
New blocks begin at zero; nonfinite values map to positive infinity. The array
and two 64-bit group counters borrow aligned storage strictly after immutable
snapshots and prompt capture, expire before the next Prefill and allocate nothing.
Legacy transactions without that borrow retain the original batch path.

Within the existing PV CTA, each head warp reduces the absolute probabilities
of its actual causal 64-position tile. The outward product of that maximum and
its persistent V maximum bounds every product. The warp skips its original
loop only if both output accumulators in every lane satisfy the strict test.
Otherwise all original increasing-position FMAs execute unchanged. Async V
staging and its barriers remain intact; the transformation removes arithmetic
and shared reads, not the already issued global V copies. QK, actual-length
softmax and final BF16 publication remain unchanged. Counters record one tested
head/dimension block and one eliminated block, accumulated once per warp at
kernel completion; they do not count committed tokens. A single readback at
Decode completion is included in full Decode time. Failure uses the existing
poison boundary, and complete scalar-prefix state/logits remain mandatory.

The [exact-PV rejection](metadata/qwen36-27b-mtp-exact-pv-elision-rejection-2026-10-10.json) closes this version without retention.
The initial bounds setup polled cancellation before the existing draft reset
and batch boundary. One repair moved setup after successful draft initialization;
the established cancellation boundary and complete prefix/transaction checks
then passed. All eight d2 API/lifecycle checks also pass, but actual elision is
negligible and no Decode gain appears at 8K or 40K. All new bounds, counters,
kernels and bindings are removed. The proof and both source snapshots remain
frozen evidence only; the isolated implementation is retained v7.

## Startup-owned pair layout composition v18

An isolated build prepares one immutable private NVFP4 arena directly from
all authenticated canonical Gate, Up and Down tensors. Every output pair and
K256 span stores 256 lane-major weight bytes followed by 32 full scale bytes.
A thread reads its two four-byte row words through one aligned uint2; paired
lanes share one two-byte scale record. The exact original eight K positions,
four accumulator chains, warp reduction, final tensor scale and BF16 rounding
remain unchanged. Packed codes and scales are permuted without decoding or
requantization. Startup checks every packed payload against canonical input
before publication. No request-time repacking or allocation is permitted.

The owning private transaction constructs the complete arena before service
readiness and releases it only after target work drains. It never substitutes
its pointer into an existing public quad-layout field. The explicit admission
build omits old Gate/Up, Down consumer and scale6 preparation and reports their
absence truthfully; the private owner has a separate layout/byte identity.
Scalar/M1 execution, including startup Graphs and the independent full-state
oracle, uses canonical NVFP4. Ordinary production preparation is unchanged.
The old and new owners may not coexist and consume the memory margin. Failed
packing, checking, resource admission or construction releases partial private
ownership and never publishes readiness. The [design record](metadata/qwen36-27b-mtp-pair-layout-design-2026-10-10.json)
binds exact sizes and the original source-only hypothesis. The complete
owner/inventory/consumer composition subsequently passed full-size pack/check
and consumer correctness, complete P65 transactions and all P8192 prefixes.
The [pair-layout rejection](metadata/qwen36-27b-mtp-pair-layout-rejection-2026-10-10.json) closes it after negative P65 API direction despite valid output,
route and startup reserve. All pair paths and the earlier pack/check
prerequisite are removed. Frozen sources retain reproduction authority only;
isolated v7 and the production non-MTP route remain unchanged.

## Startup-owned FP8 input layout composition v19

The private startup owner covers 144 input projections: QKV/Z on 48 linear
layers and Q/K/V on 16 full-attention layers. Each `[output_quad][K_quad]`
record holds four consecutive four-byte output-row words, with the existing
involutive `code ^ (code >> 5)` lookup-index swizzle applied independently
per byte. A separate checker reconstructs each canonical byte before readiness.
No decoding, requantization or tensor scaling enters packing. Every code,
including signed zero and NaN, keeps the exact existing FP32 lookup value.
The original four FMA chains, warp/eight-warp tree, final scale and BF16 RNE
remain unchanged. The scalar oracle uses authenticated canonical weights.

The owner is private to multi-row whole-core transactions, with 192 fixed
layer/role view slots and exactly 144 populated entries. It constructs before
readiness, validates every payload, and publishes views only after completion.
Failed allocation, copy/check, reserve or construction drains and releases
partial ownership. Destruction drains its stream before freeing the arena.
No request-time packing, allocation or public sidecar-layout reinterpretation
is permitted. The isolated build omits FP8 O and NVFP4 Down auxiliary assets;
those roles use existing canonical scalar/multi-row consumers. Gate/Up and
Prefill Down assets remain intact. Startup and request receipts distinguish
base inventory from the private 5,200,936,960-byte owner. Production is unchanged.

The [design record](metadata/qwen36-27b-mtp-fp8-input-layout-design-2026-10-10.json)
fixes the resource exchange and source-derived instruction hypothesis. It has
no standalone performance authority; full-prefix exact oracles admit the
composition before its immediate real API direction screen.

The [FP8 input-layout rejection](metadata/qwen36-27b-mtp-fp8-input-layout-rejection-2026-10-10.json)
closes this version after exact packing/consumer and complete P65/P8192 state
checks, plus all eight d2 API/lifecycle checks. Its 8K/40K throughput is
essentially tied with retained v7, so the additional owner/inventory exchange
is removed. Frozen source and binaries retain reproduction authority only;
no d3 qualification or production switch follows. The isolated implementation
remains v7 and the 1.5x–3x objective remains unmet.

## Compact full-K residency composition v20

FP8 CTAs stage four complete compressed output rows into private quad-word
shared storage, applying only the incumbent lookup-index swizzle. The existing
O sidecar is copied directly. Each of M2..4 independent 256-thread groups then
executes its original K subsequences, four FP32 chains, warp/eight-warp tree,
scale and BF16 publication. Packed weights are immutable after the producer
barrier; per-row reduction partials occupy disjoint storage. No inner-K barrier
or expanded BF16/FP32 weight array exists. Maximum shared storage is 26,112 bytes.

Gate/Up CTAs copy four full existing K5120 quad records, including exact scale
bytes, into 46,080 private shared bytes. Each independent 128-thread row group
uses its four original warps and four-output/four-chain accumulation. Codebooks
add 1,088 bytes; all reads remain in the immutable compact records. Down retains
v7 because its longer K would consume excessive full-record shared capacity.
The complete original arithmetic/state boundary is unchanged for all rows.

This transfers compact shared operand delivery from the studied Marlin source,
without importing its MMA tree or code. A separate consumer group per row
reduces live accumulators but repeats shared decoding. Resource counts do not
select performance: complete same-history state/logits and the real API remain
the admission and selection boundaries. The design adds no request allocation,
weight owner, host synchronization or new production route. One bounded
composition returns directly to the API under the Roadmap stop condition.

The [compact-residency rejection](metadata/qwen36-27b-mtp-compact-residency-rejection-2026-10-10.json)
closes this composition after complete synthetic/real-prefix admission and a
negative P65 API result. No new compact executor remains selected. Reduced
registers did not compensate for the composed execution cost; evidence does
not isolate a particular memory/stall cause. All new paths are removed and
frozen for reproduction only. Production and retained isolated v7 are unchanged.

## Phase-borrowed FP8 input layout composition v21

The lossless v19 pack/check and exact vector consumer are reused with a new
ownership composition. All original O/Down/GateUp layouts remain attached.
The source-private transaction binds two aligned segments of the existing
whole-core family arena, excluding all five immutable snapshot slots. No raw
family view is added to the public API. Complete checked plan extents and typed
snapshot-pointer identity must hold before binding. Whole tensors occupy one
segment each; a separate aligned checker flag occupies the final 256 bytes.
The 144 input layouts total 5,200,936,960 bytes, within the 5,591,000,064 bytes
available after excluding snapshots. Persistent state, residual, RoPE and the
C512 Decode bundle remain physically disjoint.

The first multi-row transaction begins only after successful whole-core
Prefill and synchronized draft-cache initialization. The normalized prompt
capture's existing borrow expires at that transaction boundary; its bytes may
then hold layouts. Canonical weights are immutable. Each request prepares and
independently checks all layouts once on the target stream; readiness publishes
only after checking and synchronization. This explicit isolated-admission
experiment permits that fixed request-phase transformation, unlike v19's
startup-only owner; it does not amend the production prohibition on request-time
checkpoint repacking. No allocation, tactic search or public weight-field
reinterpretation occurs. Preparation time/count/bytes are reported separately
and included in complete Decode. Short output without a multi-row transaction
does not prepare layouts. Legacy and scalar-oracle transactions stay canonical.

New initialization invalidates readiness; abort drains and invalidates it.
Successful rounds retain the prepared bytes until the request ends; next
Prefill may overwrite them only outside a transaction. A failure never falls
back after partial enqueue. Complete prefix/state/logit, repeated initialization,
cancellation and recovery checks admit one composition before immediate API
selection. The exact arithmetic and independent baseline remain unchanged.

The [phase-layout rejection](metadata/qwen36-27b-mtp-phase-fp8-layout-rejection-2026-10-10.json)
closes the experiment after full numerical, reuse/cancellation and d2 API
checks. The source-private borrowed layouts and preparation witness are removed;
the request-state/system lifetime exception is not retained. Full inventory
preservation did not produce a useful verifier or API gain. Frozen source has
reproduction authority only; isolated v7 and production remain unchanged.

## Ordered draft composition v22

The retained v7 target verifier and draft MMA compose with the existing exact
production ordered GQA only for draft full steps at S512..44095. Original
QK/softmax/PV arithmetic, BF16 publication, sigmoid gate and remaining draft
layer remain unchanged. The private draft owner validates the fixed device
before readiness and requires the operator's complete 4,233,120-byte scratch
extent, even for smaller S. A smaller draft arena keeps the reference before
enqueue. Once selected, failure drains/poisons rather than falling back.

No new allocation, target operation, cache initialization or reconciliation
change is introduced. An actual ordered-step counter resets with draft state
and is exported in the request receipt; eligible full-context requests require
one hit per proposal. The composition returns directly through P513 full
transactions and the real d2/d3 API. Earlier losing target combinations do not
supply standalone performance authority. Any retained benefit must participate
in the next complete verifier architecture; the full 1.5x–3x objective remains.

The [direction record](metadata/qwen36-27b-mtp-draft-ordered-composition-direction-2026-10-10.json)
binds complete same-input draft and P513 state/transaction checks plus both
full d2/d3 API panels. The service wiring remains dependency-only for the next
verifier composition; single-process observations do not grant architecture
qualification or production eligibility. The private constructor's original
Attention option is used only by the numerical comparator, with no server
selector or installed ABI. Short/inadequate-scratch cases retain reference
execution before enqueue, and the request counter records actual route hits.

## Combined immutable decoder composition v23

For every four-bit weight code q and eight-bit block-scale code s, construction
computes `C[16*s+q] = FP4(q) * FP8(s)` with the original FP32 multiplication.
The complete 4,096-entry table includes signed zeros and non-finite codes.
Each verifier weight now reads this exact product once from private immutable
device storage instead of two shared lookups and multiplication. The scalar
oracle retains its existing arithmetic. All K subsequences, four FMA chains,
warp merge, final tensor scale and BF16 rounding stay identical; no scale is
moved across accumulation. Canonical, Gate/Up coupled and Down consumer-order
formats still supply the same nibble and full block-scale code.

The multi-row transaction owns exactly 16,384 additional bytes, initialized
and synchronized before readiness and never modified during requests. It
releases them only after its target work drains. Failure during construction
cleans partial ownership; request failures retain existing poison semantics.
No global mutable table, checkpoint rewrite, request allocation or production
selector is introduced. Per-CTA codebook initialization and its barrier vanish;
read-only cache behavior and indexing become explicit costs. Exhaustive product
and full-output checks plus real scalar state/logits admit one composition with
v25 ordered draft before the same complete API direction panel.

The [combined-decoder rejection](metadata/qwen36-27b-mtp-combined-codebook-rejection-2026-10-10.json)
closes this version after exact exhaustive/full-prefix admission and negative
P65 API direction. The immutable table, owner and consumers are removed;
frozen source retains reproduction authority only. Fewer decoder operations
did not select a faster runner. The retained isolated implementation returns
to v25 ordered draft plus the original target verifier, with no production change.

## Batched vocabulary finalization composition v24

The v25 transformer verifier remains unchanged. For M2..4, the complete target
head consumes all normalized hidden rows together, reading and decoding each
canonical NVFP4 weight once per row group. The earlier exact persistent head
consumer is reused only at N248320/K5120. A fixed 64-CTA executor stages all
M activation rows once in CTA-private shared memory, then retains original
K subsequences, four FMA chains, parenthesized merge, warp reduction, tensor
scale and BF16 publication for every output. No quantization or reassociation
is introduced. Its shared extent is at most 42,048 bytes.

Projection-0 is dead after the final transformer layer. Its validated C512
extent can hold the complete M-by-vocabulary BF16 matrix (at most 1,986,560
bytes). Smaller oracle arenas select the existing scalar finalization before
enqueue. The batched producer finishes on the target stream before each whole
logit row is copied to its immutable prefix slot. Each row retains full-vocabulary
finite validation and earliest-index argmax, and no slot gains commit authority
before the existing synchronized boundary. Failed enqueue drains/poisons;
there is no fallback after partial execution. No allocation or state layout
changes. Complete scalar-prefix state/logits and real API direction select this
composition independently of the earlier losing projection bundle.

The [completed batched-head direction](metadata/qwen36-27b-mtp-batched-head-direction-2026-10-10.json)
passes exhaustive full-vocabulary and complete P65 transaction admission, then
both complete API panels. It remains in isolated development as a bounded
prerequisite for the next verifier composition. Preserved full output and a
single-process positive direction grant no production or target-attainment claim.

## Projection lifetime composition v25

Gate/Up's original four warps independently read every M activation row.
The candidate stages `min(M,3)` complete K5120 BF16 rows once per CTA, sharing
those exact bits across the unchanged four consumers. M4 keeps its fourth row
in global memory; the three-row bound preserves room for four CTAs in SM87's
shared capacity. Producer copies and codebook construction precede one uniform
barrier; inputs are immutable thereafter and expire with the CTA. At most
31,808 static shared bytes are used. Startup sets a fixed maximum-shared
carveout on the three Gate kernels before publishing the transaction; failures
fail construction. Original packed weights, grid, K ownership, four FMA chains,
scale placement, warp merge and BF16 output stay unchanged. No new global
allocation, persistent state or cross-CTA dependency is introduced.

FP8 retains its entire mainloop and per-warp partial publication. After the
existing uniform barrier, each warp owns independent `(token, output)` final
reductions in a stride-eight loop. Every reducer reads the identical eight
partial values into lanes 0..7, zero elsewhere, then executes the same full
warp tree, scale and BF16 publication. All finite and exceptional boundaries
remain the original ones; no reassociation or omitted zero additions occurs.
This removes serialization in warp zero while preserving all required barriers.
The two mechanisms compose with retained ordered draft and batched head under
a distinct isolated API identity. Complete scalar-state/full-logit admission
and the real API determine whether the execution trade is useful.

The [completed lifetime direction](metadata/qwen36-27b-mtp-projection-lifetime-direction-2026-10-10.json)
passes complete synthetic/real-prefix checks and both API panels. The isolated
implementation retains this composition with bounded dependency authority;
there is no numerical waiver, statistical architecture selection or production
switch. The full 1.5x–3x objective remains unmet.

## Register lookahead composition v26

FP8 uses fixed input-width instantiations for K5120 and K6144. Each original
thread retains the current raw 16-byte four-output record plus one next record.
The next K1024-strided weight load is issued before current codebook decoding
and M-row FMA consumption; activations are loaded only for the current block.
The first record precedes codebook readiness, and the final iteration performs
no lookahead. Records are thread-private registers with no peer visibility,
shared producer storage, inter-iteration barrier or global allocation.

Canonical and output-sidecar addresses, cache operators, lookup swizzle,
FP8 expansion, BF16 activation expansion and every original four-chain FMA
subsequence remain unchanged. Final partial publication uses retained v28
finalization. Tail validation admits only the existing fixed complete K shapes;
no out-of-range prefetch or discarded speculative arithmetic is permitted.
Static launch bounds preserve the incumbent M2/M3/M4 register-limited CTA
floors of four/three/two; a spill fails admission instead of silently changing
the execution cost. The candidate composes with all retained v28 mechanisms
and preserves the complete scalar-prefix/full-logit and real API gates.

The [completed direction](metadata/qwen36-27b-mtp-register-lookahead-rejection-2026-10-10.json)
closes this executor without retention after complete numerical and d2 API
checks. Compiler scheduling retains only partial overlap: canonical M3 next
weight loads occur during current FMA consumption, not before its first FMA.
All nine fixed instantiations have no spills, but API differences of only
+0.30% to +0.46% do not establish a useful gain. The described v26 executor
is removed; isolated service remains v28. No d3 or further lookahead scan follows.

## Live reduction ownership composition v27

The FP8 mainloop and four-chain merge are unchanged. Let `s[r,l]` be the
original merged value for channel r and lane l. The stage-16 tree computes
`s[r,i] + s[r,i+16]` for i0..15. Lower-half lanes retain channels 0/1;
upper-half lanes retain channels 2/3 at matching i, using XOR shuffles and
explicit lower-operand-first addition. Stage 8 partitions each half into
channels, computing the original `t[r,i] + t[r,i+8]` for i0..7. Four 8-lane
groups now own the four independent channel trees; offsets 4, 2 and 1 complete
the identical ordered tree. Lanes 0/8/16/24 publish channels 0/1/2/3 into the
original `[M,4,8]` partial array. No dead-lane result is observed.

After the existing barrier, one warp per token reads the four channel partials
in four groups of eight. Each leaf explicitly adds positive zero twice,
preserving the original warp tree's offsets 16 and 8 including signed-zero
and non-finite behavior. Grouped offsets 4/2/1, tensor scale and BF16 rounding
then publish the same outputs. Every invoked shuffle uses the full active warp
mask; ownership branches are warp-uniform. This changes neither accumulation
order nor persistent state and requires no new scratch, layout or allocation.
Directed tree tests and complete scalar state/logits precede the real API.

The [completed live-reduction direction](metadata/qwen36-27b-mtp-live-reduction-direction-2026-10-10.json)
passes directed exceptional-value trees, complete projection outputs, full P65
state/logits/transactions and both API panels. The exact mapping remains in
the isolated v30 service as a bounded composition prerequisite. Numerical
agreement and single-process positive direction do not select production or
establish the owner's full speedup objective.

## NV live reduction composition v28

Gate/Up, Down and the batched target vocabulary head reuse the exact
`warp_sum_four` mapping proved by v27. Each warp retains all four original
four-chain leaves and the unchanged K traversal. Ordered stage16/8/4/2/1
additions migrate roots to lanes 0/8/16/24, followed by the original tensor
scale and BF16 publication. There is no cross-warp merge in these families.
The persistent head advances its existing output-row loop only after this
same warp has published all four roots. Shared inputs/codebooks, grids,
allocations and all request-state boundaries remain unchanged. Full canonical,
packed and vocabulary output checks precede the complete real-prefix oracle
and the next API return. This is a numerical-preserving composition hypothesis,
not a performance or production claim.

The [completed NV direction](metadata/qwen36-27b-mtp-nv-live-reduction-direction-2026-10-10.json)
passes all 30 complete synthetic outputs/guards, full P65 state/logit and
transaction/recovery checks, and both complete API panels. The exact mapping
remains dependency-only in isolated v31 through the next complete verifier
API return. No numerical waiver, new allocation or production switch follows;
the single-process positive direction does not establish the 1.5x–3x objective.

## Softmax lifetime composition v29

One 256-thread CTA owns each `(speculative row, head)` with its original actual
causal length and fixed maximum-length row slot. A single grid replaces serial
row launches. Thread-local maximum and denominator traverse the same strided
columns. Shared stages 128/64/32 followed by warp stages 16/8/4/2/1 preserve
the original ordered 256-leaf tree; explicit barriers protect the root broadcast
and later scratch reuse. No online normalization or changed exp approximation
is introduced.

After maximum publication, each thread computes the original `expf(score-max)`
once and overwrites only its own now-dead score cells. These FP32 values feed
the unchanged sequential denominator and final division, replacing duplicate
exponential evaluation. The existing workspace already has exactly this
producer/consumer lifetime; padding and earlier-row causal ends remain untouched.
QK/PV, BF16 output/state, failure, capacity and production contracts remain
unchanged. Complete FP32 probability equality and real-prefix state/logits
admit the composition before its immediate API return.

The [completed rejection](metadata/qwen36-27b-mtp-softmax-lifetime-rejection-2026-10-10.json)
closes this version after complete probability/state/logit checks and all eight
d2 API/lifecycle requests. The small single-process Decode differences do not
establish material runner value. All described new softmax paths are removed;
isolated source is restored to v31, with no d3 or further normalization scan.
Frozen source retains reproduction authority only; production is unchanged.

## NV register feed composition v30

Gate/Up and Down retain one current and one next raw four-output weight record
per thread. The next K256 record is read before current decoding/consumption;
block-scale ownership and scale6 phase exchange retain the original K512 loop.
Only independent immutable read timing changes. The last phase reads no future
record; head and FP8 execution remain v31. Original decoders, four FMA chains,
K subsequences, live-root reductions, tensor scaling and BF16 publication are
unchanged. No shared producer ring, per-tile barrier, layout or allocation is
introduced. Fixed per-role launch bounds preserve the incumbent CTA floors;
nonzero spills fail static admission. Complete full-output and real-prefix
state/logit checks precede the immediate complete API comparison.

The [completed rejection](metadata/qwen36-27b-mtp-nv-register-feed-rejection-2026-10-10.json)
closes this version after static resource, complete synthetic/real-prefix and
all eight d2 API checks. Earlier raw loads preserve the arithmetic but do not
improve the complete runner. All new feed/profile/auditor paths are removed,
restoring isolated v31. No d3 or further resource/feed scan follows. Frozen
source has reproduction authority only; production is unchanged.

## Independent accumulation chain composition v31

Each NVFP4 128-thread CTA owns one original four-output group. Its lane mapping
is `old_lane=tid/4`, `chain=tid%4`; every thread retains all M token accumulators
for that chain. Its two products per K256 phase are at `old_lane*8+chain` and
four positions later, preserving the original per-chain increasing K FMA order.
Packed weight and scale values remain unchanged; no operand broadcast or
inner-loop barrier is introduced. Chain pairs merge lower operand first, then
`(a0+a1)+(a2+a3)` publishes one complete old-lane leaf to private shared storage.
A uniform CTA barrier precedes one final warp per token, which reads the same
32 leaves and uses the unchanged live-root reduction, tensor scale and BF16
rounding. Shared partials occupy at most 2,048 bytes plus the existing codebooks.
Gate/Up and canonical/packed Down change together; vocabulary and FP8 stay v31.
No allocation, state, probability, numerical tolerance or production change is
introduced. Full output/prefix checks admit the complete mapping before API
selection; reduced register counts alone cannot select it.

The [completed rejection](metadata/qwen36-27b-mtp-chain-partition-rejection-2026-10-10.json)
closes this version after complete numerical admission and negative P65 API
direction. All new chain-partition/profile/auditor paths are removed. Frozen
source proves only the tested mapping and rejected direction; no long-context,
d3 or production qualification follows. Retained isolated service is v31.

## Direct PV consumption composition v32

The target verifier's long Attention retains original QK and row-wise softmax.
PV assigns one 128-thread CTA to one query head and speculative row, with two
adjacent output dimensions per thread. Four raw BF16 V pairs and FP32
probabilities are prefetched in thread-private registers through read-only
global loads. Each accumulator still consumes every actual causal position
once in increasing order with the original FP32 FMA, then publishes BF16 once.
The final partial group consumes only valid positions. Later speculative KV
rows never enter an earlier query, and probability slots retain their actual
per-head stride inside the existing fixed maximum-length row slot.

There is no shared producer, async-copy ring, inter-warp data dependence or
CTA barrier in this PV executor. The old shared V reuse across six heads is
traded for independent cached reads. Cache misses change cost, never results.
Total warp count remains 96 per speculative row. No allocation, state layout,
inventory, reserve, failure or production boundary changes. Complete P513/P8192
scalar state/logits admit one composition before its immediate real API return;
resource savings alone provide no performance authority.

The [completed rejection](metadata/qwen36-27b-mtp-direct-pv-rejection-2026-10-10.json)
passes complete P513/P8192 state/logit admission but regresses d2 P8192/O256
Decode by 9.01% versus retained v31, with identical acceptance and work.
The new direct consumer and profile/auditor wiring are removed. No 40K/d3
or full API lifecycle qualification follows; frozen source is reproduction
evidence only. Removing shared publication did not select a faster runner.

## Gate/Up paired publication composition v33

One 256-thread CTA owns sixteen Gate and sixteen matching Up channels. Each
four-warp half retains the original K subsequences, four FP32 FMA chains,
ordered live-root reduction, independent tensor scale and BF16 publication.
The halves share one immutable `min(M,3)`-row input and codebook publication;
M4's fourth row retains its original global reads. Neither half reads peer
accumulators. Both rounded BF16 result vectors enter a private at-most-256-byte
pair buffer before a uniform CTA barrier. Each output then evaluates the
original `g/(1+expf(-g))*u` in FP32 from those BF16 operands and rounds once to
BF16 into the existing Gate output buffer. The Up intermediate has no later
consumer and need not reach global memory.

This preserves every numerical publication while removing duplicate input/table
initialization and two kernel boundaries per layer. It introduces no allocation,
weight layout, state or observer change. Absent paired sidecars or M1 selects the
established path before enqueue; a selected paired launch failure drains and
poisons rather than falling back. Complete original-output guards and real
scalar-prefix/full-logit admission precede the immediate real API comparison.
The implementation has no performance or production authority before that return.

The [completed rejection](metadata/qwen36-27b-mtp-gate-pair-rejection-2026-10-10.json)
passes full output/state/logit and d2 API/lifecycle checks, but 8K/40K complete
verification is effectively unchanged. The paired executor and wiring are
removed, restoring v31. Frozen source retains reproduction authority only;
no d3 or further producer/consumer mapping scan follows.

## Batched cache reconciliation composition v34

After complete target verification, the first prefix commit prepares all
contiguous valid input-token/previous-target-hidden pairs together. The existing
cache-only M1..32 FC/K/V executor and scratch handle at most four independent
rows, with unchanged BF16 arithmetic and absolute RoPE positions. Preparation
does not advance logical draft length. A checked prepared watermark permits
exactly one successive row to become visible at each commit, after the matching
target recurrent/hidden/full-logit restoration and before its observer.

Cancellation or EOS may leave physically prepared rows beyond the live prefix;
they are inaccessible and overwritten before reuse. Reset, rewind, full step,
ordinary append, poison and transaction retirement invalidate prepared authority.
No new allocation, stream, capacity or numerical tolerance is introduced.
Scalar transactions retain independent per-row replay. Complete live-cache and
target-state comparisons admit the composition before immediate API selection;
the bounded reconciliation budget does not establish the full speedup goal.

The [completed direction](metadata/qwen36-27b-mtp-batch-reconcile-rejection-2026-10-10.json)
passes complete state/cache and API checks, but its small phase reduction does
not produce a useful complete-runner gain. Prepared-prefix authority and its
executor/wiring are removed, restoring v31. Frozen source retains reproduction
authority only; no d3, batching scan or production change follows.

## Exact activation publication composition v35

Every transformer projection expands each original BF16 input bit pattern once
into existing request-owned FP32 scratch. Same-stream consumers load those
exact bits with unchanged quantized decoding, four FMA chains, ordered merges,
tensor scaling and BF16 output. Maximum storage is 278,528 bytes, disjoint from
inputs/outputs/weights; earlier scratch consumers have retired and later
Attention/argmax consumers begin after projection completion. Validation rejects
insufficient, misaligned or overlapping scratch before enqueue. No allocation,
new stream or fallback after enqueue is introduced. The vocabulary head retains
its original BF16 executor. Complete output/state/logit checks precede the real
API tradeoff between removed unpacking and increased activation traffic.

The [completed rejection](metadata/qwen36-27b-mtp-activation-publication-rejection-2026-10-10.json)
passes complete synthetic and real-prefix numerical admission but regresses
P65 d2 API Decode. The producer, scratch binding, FP32 consumers and distinct
profile are removed; retained isolated source is v31. Frozen source has
reproduction authority only, with no long-context or production claim.

## QK live reduction composition v36

Each original QK warp retains all six unchanged eight-product subtrees.
Heads 0..3 use the proved four-root live mapping: ordered stage16/stage8
additions move their eight leaves into groups at lanes 0/8/16/24 before
stages4/2/1. Heads4/5 use two sixteen-lane groups: ordered stage16 places
head4 in the lower half and head5 in the upper half, then stages8/4/2/1
preserve each original tree. Root lanes publish their own score rows with the
unchanged 1/16 scale. Every valid position is warp-uniform and all shuffle
lanes remain active. No extra zero, changed operand order, causal read,
shared storage, launch geometry or request-state boundary is introduced.
Directed raw FP32 roots and full scalar-prefix state/logits admit this exact
execution change before real API selection.

The [completed direction](metadata/qwen36-27b-mtp-qk-live-reduction-direction-2026-10-10.json) preserves complete numerical
and API behavior for both policies. The exact QK execution is retained only
as a bounded development dependency, with no production or full-target claim.
The service composition below identifies this cumulative implementation as v39.

## Exact short-chain matrix composition v37 — removed

The following describes the removed v40 experiment for reproduction only.
The [rejection](metadata/qwen36-27b-mtp-short-chain-executor-rejection-2026-10-10.json)
closes it after exact numerical admission but negative P65/8K API direction.
All runtime and borrowed-bound ownership changes are removed; the retained
service remains v39. Numerical eligibility alone did not produce fast execution.

The native/current-payload proof admitted an isolated FP8 executor at M2..4.
A 256-thread CTA owns sixteen output channels, staging one original warp's
128 chains at a time. Canonical packed inputs are read coalesced and decoded
once into shared BF16. Two zero-start BF16 m16n8k16 instructions compute the
four original five/six-product chains for each original thread and all M rows.
Disjoint K/column blocks preserve chain identity. Every eligible signed subset
is exactly normal FP32 or zero under the proved exponent envelope. Zero roots
are canonicalized to positive zero. Uncertified roots run their original
ordered FMAs before the unchanged four-chain, 32-lane and eight-warp merges,
tensor scale and BF16 publication. Nonfinite weight in either paired block
forces both roots to repair because zero multiplication can propagate NaN.
Nonfinite and nonzero subnormal BF16 inputs fail eligibility before zero tests.
No approximate intermediate gains publication authority.

Static minimum-LSB and ceil-L1 bounds occupy nine bits per chain in separate
low-byte/high-bit arrays. Their 1,547,698,176 bytes borrow aligned whole-core
GDN scratch strictly after all five immutable snapshots and complete prompt
hidden. Both earlier ranges retain their existing lifetimes. Construction
checks every source shape and borrowed extent. The first multirow verification
prepares every bound once on the target stream; original weights never change.
Initialization and abort invalidate readiness; new Prefill cannot overlap an
active transaction. No allocation or public ABI is added, and legacy owners
retain the old executor before enqueue. Complete verification timing includes
preparation. Dynamic per-input-chain bounds use dead, range-checked C512 FP32
scratch and expire before later Attention/argmax consumers. Errors drain and
poison, with no fallback after partial enqueue. Full scalar-prefix state/logits
and real API direction select the composed executor; coverage alone is no gain.

## Compact shared-row QK composition v38

One CTA owns two independent key positions and all speculative rows for one KV
head. Its2*M warps each retain the original six queries and complete eight-product
and live-ancestor trees. A producer pair stages each K position once into a
four-buffer ring; every row consumer completes before reuse. The128-position
tile executes64 two-position iterations. Per-row actual causal extents guard
products and publication, while staging safely uses the latest extent. Separate
M*6*128 FP32 score storage precedes coalesced global publication. M2/3/4 uses
128/192/256 threads and10240/13312/16384 shared bytes. This avoids the rejected
v19 head split and8*M-warp CTA. Softmax, PV, state, workspace and failure
semantics remain unchanged. Complete scalar-prefix admission and the real API
select the composition; shared bytes or traffic alone establish no gain.

The [completed direction](metadata/qwen36-27b-mtp-qk-row-share-direction-2026-10-10.json)
passes complete P513/P8192 state/logits and both API panels. Small consistent
verification reductions retain this exact executor only as a bounded
next-composition dependency. No noise-qualified selection or production claim
follows; the service below identifies the cumulative admission as v41.

## Shared-row PV recomposition v39 — removed

The following describes the removed experiment for reproduction only.
The [completed rejection](metadata/qwen36-27b-mtp-pv-row-compose-rejection-2026-10-10.json)
passes full numerical and d2 API checks but finds no useful gain. All new paths
are removed; the current service remains v41.

The compact QK was composed once with the historical exact shared-row PV
component. A192-thread CTA owns one KV head and32 dimensions; six warps retain
one independent ordered accumulator per speculative row and lane. A four-buffer
ring stages64 V positions once for all rows and each row's distinct original
FP32 probabilities. Every row applies its actual causal end before FMA and
publishes BF16 once. Staging to the latest causal extent does not authorize
future-position arithmetic. All consumers retire before buffer reuse.
M2/3/4 uses28672/34816/40960 shared bytes and no new request storage. This
recomposition retains current QK and softmax, original alias/capacity/failure
checks and all persistent state boundaries. Its API comparison against v41
separates PV from the earlier failed large-CTA QK/PV combination; it does not
reverse that historical result or permit a geometry scan.

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
`q3x.sm87.admission.mtp-qk-row-share-api.v41`. The startup-only
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
