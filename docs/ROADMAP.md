---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-08-20
  supersedes: [docs/ROADMAP_LEGACY.md]
  superseded_by: []
  ssot_for: active unfinished delivery slices and their ordering
  review_trigger: delivery dependency, active phase, or milestone-exit change
---

# Qwen3x-Orin active delivery roadmap

This roadmap contains only unfinished delivery slices and their exit criteria.
It does not own current performance numbers, experiment history, subsystem
design, or evidence. Those facts belong to Current Status, the SDD, subsystem
contracts, and immutable evidence respectively.

## Ordering rule

Work is ordered from the final runner boundary inward:

```text
callable product API
  -> exact request/capacity contract
  -> attested release and execution plan
  -> target-length Prefill and Decode budgets
  -> subsystem architecture candidates
  -> scoped local optimization work packages
  -> kernels, layouts, and local mechanisms
```

A local mechanism does not enter this roadmap independently. It appears only
as a required mutation inside a named architecture candidate selected by an
API-visible product constraint. Local work-package rules stop at that package;
they cannot reorder the delivery slices below.

Inside every Prefill architecture candidate, the work order is also fixed:

```text
real-number equivalence and live graph
  -> finite-precision operands, reduction tree, rounding and state boundaries
  -> production observables, buffer/control-state lifetime and alias ownership
  -> engineering map to SM87 residency, ownership, pipeline and synchronization
```

The
[`PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md`](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md)
owns the proof classes. A local kernel parameter cannot skip or retroactively
justify an earlier step.

## Locked outcome

The owner-set constraints are normative in the
[Constitution](ENGINEERING_CONSTITUTION.md#8-locked-business-targets), and the
exact external contract is owned by the
[system SDD](SDD.md#2-end-state-product-contract). This roadmap sequences the
unfinished work needed to satisfy them; it neither copies nor recalibrates
the targets. Current support and gaps are reported only in Current Status.

## P0. Documentation control plane

Purpose: preserve project intent and system truth across agents, branches, and
context compaction before further architecture work.

Remaining deliverable:

- carry the documentation-control commit into every actual development
  baseline, because an `AGENTS.md` in another linked worktree is not a Codex
  entry point for a session started elsewhere.

Exit criteria:

- every tracked Markdown file is classified;
- the `document_control` host test rejects registry, required-header,
  identity, reciprocal-supersession, and local-link drift;
- every active task can trace product constraint -> budget -> architecture
  candidate -> local work package -> API return point;
- no active document calls a historical component proxy the product target;
- README, SDD, Current Status, Roadmap, policies, contracts, and evidence each
  have one non-overlapping truth owner.

## P1. Product API and target-capacity boundary

Purpose: make the final workload executable and observable before selecting a
performance architecture. This phase may use one pinned exact development
artifact; it does not call that artifact a release candidate.

Deliverables:

- a versioned OpenAI-compatible production API contract distinct from the
  loopback evaluation adapter;
- exact definitions for the 40K, 60K, and approximately 130K token witnesses,
  including maximum output tokens and `prompt + output - 1` capacity;
- capacity admission before partial execution and no silent truncation;
- versioned memory profiles for each witness;
- request and startup state machines defined by the SDD;
- streaming first-token, usage, error, overload, disconnect, cancellation, and
  shutdown semantics;
- structured per-request observability for queue, tokenize, admission, pure
  Prefill, final prompt/logits, first commit, first byte, Decode, and total;
- one named exact measurement artifact and route identity that excludes MTP,
  approximate research routes, cuBLASLt, and undeclared fallbacks while the
  release packaging contract is still being implemented.

Exit criteria:

- all target witnesses are accepted or rejected before execution with an exact
  reason;
- successful requests prove full prompt consumption and complete output
  accounting;
- target-length cold/no-cache API smoke tests run through the pinned exact
  measurement artifact without claiming release qualification;
- the evaluation adapter and product trust boundary cannot be confused in
  documentation or telemetry.

## P2. Canonical exact release artifact

Purpose: package the now-executable product boundary into one binary whose
measured route is the route users receive.

Deliverables:

- a `BUILD_TESTING=OFF` release profile;
- no test-only admission flag or environment-variable route composition;
- authenticated checkpoint, executable, kernel ABI, sidecar, workspace, and
  execution-plan identities;
- explicit `DeploymentPlan`, `PrefillExecutionPlan`, `DecodeExecutionPlan`,
  and `RequestMemoryPlan` boundaries;
- startup route warmup and attestation;
- fail-closed startup on plan, artifact, capacity, workspace, or route mismatch;
- proof that MTP, approximate research routes, cuBLASLt, and unintended
  fallbacks are unreachable from the release artifact.

Exit criteria:

- one documented command builds the release with testing disabled;
- the server becomes ready only after plan and route attestation;
- target-length API smoke tests and deterministic exact-output oracles use the
  same executable that will enter external performance selection;
- health/telemetry identifies the release and route without relying on an
  experiment harness.

## P3. Exact Prefill parity, then specialization advantage

Purpose: first reach the useful vLLM starting line and then justify the
specialized runner by exceeding it.

P3 implementation preparation may proceed in the same pinned development
route while P1/P2 capacity and release seams are being closed. It cannot
select an architecture, change the default, or promote production before the
P1 and P2 exits provide the target-length API and canonical release artifact.

Current implementation facts, the P40 incumbent, rejected candidate versions,
and their measurements are owned by
[Current Status](CURRENT_STATUS.md) and immutable evidence. This roadmap does
not repeat them. A new candidate compares against the incumbent identified
there without inheriting a rejected kernel skeleton or treating an old
measurement as a current plan.

### Active architecture program

`AC-PREFILL-SM87-AOT-SYSTEM-v1` reached the exact real-model P40 API and was
closed after its first request returned no bytes before the 840.000399-second
client timeout. Its frozen evidence identifies whole-composition
serialization rather than an admissible local tuning gap. The v1 chain is now
a default-off correctness/diagnostic control and cannot receive an unchanged
rerun or parameter scan.

`AC-PREFILL-SM87-BULK-DATAFLOW-v2` subsequently reached the same real P40 API
boundary and is also closed. EvalScope 1.9.1 received zero bytes and recorded
0/1 success after its 680.73-second timeout, so V2 has no TTFT or throughput
result. The cancellation line's layer 37 / quantum 38 values are host
submission progress, not completed-layer timing. The one permitted bounded
causal profile then captured a valid 120.002145-second request window:
Gate+Up plus Down account for 84.2735% of aggregate kernel time, all
projections account for 87.3325%, and 16,760 `cudaLaunchKernel` calls contain
long host-blocking intervals behind queued work. The profile selects a new
dataflow; it does not authorize V2 tile, stage, cache, or launch scanning. V2
remains a default-off, accuracy-unqualified, non-production exact/diagnostic
control. It receives no repetition, P60/P130, NCU tuning campaign, or
production promotion. Exact API and profile evidence is frozen in the
[`Bulk V2 P40 rejection record`](metadata/qwen36-27b-sm87-bulk-v2-p40-rejection-2026-08-14.json).

The last complete executable development composition is
`AC-PREFILL-SM87-MACROFEED-v3`. It remains a default-off diagnostic/control
route: its attempted local P40 run did not reach completion, and its host was
not eligible for performance authority under the clean-host policy. V3 has no
P40 timing, numerical qualification, release authority, or production
dispatch. It is frozen against local tile, cache, stage, or launch scanning.

The reserved performance successor lineage is
**`AC-PREFILL-SM87-MACROFEED-v4`**. Owner direction on 2026-08-20 pauses this
lineage while the project returns to delivery mainline development. Its
recoverable, explicitly incomplete construction checkpoint is isolated on
`archive/v4-construction-ownership-20260820`; nothing in that branch is a
delivery dependency or production route. At the present boundary it is an
architecture foundation/local work package, not yet the Constitution's
`architecture_candidate`: that term becomes valid only after the complete
dataflow is runnable on the real API route. V4 changes the
whole-request traversal and ownership model before adding another kernel
mechanism. Exact P40000 is partitioned into five contiguous C8000 panels; the
panel loop is outermost and the natural layer-0--63 loop is innermost. Two
C8000 hidden planes ping-pong across layers and one C8000×17408 scratch plane
is reused by mutually exclusive phases. No full-P40 activation plane is part
of the candidate. The architecture retains the real OpenAI P40 request and
the exact finite-precision/state contract, then requires these coupled
changes:

- C8000 shape-specific Gate+Up, K-heavy Down, and FP8 feeds consume
  authenticated canonical AOT payloads without request-time repack, JIT,
  autotune, fallback, cuBLASLt, MTP, or approximate arithmetic;
- full Attention retains the projection-native per-head `[Q256, Gate256]`
  interleave, preprocesses Q in place, overwrites those Q slots with the exact
  online-Attention result, leaves Gate slots in place, and lets the FP8 O
  projection gather the logical 6,144-wide input directly; a compact Q/G
  repack or separate Attention-output plane is forbidden because the incumbent
  live set cannot fit the single scratch plane;
- GDN must alias consumed QKV/Z workspace while preserving every per-token
  BF16 state boundary; and
- convolution/GDN recurrent state uses two explicit 78,446,592-byte private
  epochs, KV visibility uses a private valid-end fence, and canonical state is
  published only after panel five, with sequence length as the final
  non-fallible visibility fence.

The tracked V4 foundation is default-off and not route-executable. It contains
the host traversal/workspace/state contract, the owner/epoch-bound host request
transaction, a private authenticated startup foundation package, independent
C8000 NVFP4 Gate+Up and Down constituents, four layout-specific FP8 admissions,
a fixed direct-scratch BF16 A/B admission, and a two-kernel exact-GDN C8000
continuation admission. It also contains an exact two-plane input-norm
and fused residual/post-norm admission that reuses the established
finite-precision bodies without a copy or third hidden plane, plus a private
three-stream/nine-event owner that enforces the device-ordered five-panel and
48-cycle-per-panel dependency graph. A first isolated execution package now
owns the fixed transient and recurrent arenas and executes one discarded,
complete synthetic-T1 GDN layer 0:
`InputNorm → (QKVZ || A/B) → continuation → GDN-O → ResidualPostNorm →`
`GateUp → Down`.
The FP8 tactics directly publish GDN
QKV/Z, interleaved Full Q/G plus private NHD K/V, Attention O from interleaved
Q slots, and GDN O from the fixed contiguous scratch slice `[4096,10240)`;
none owns a compact Q/G or GDN-O bridge. The BF16 A/B tactic reuses the exact
two-stage M64N96K64 body in one 125-CTA C8000 grid and publishes both 48-row
results directly to scratch `[16384,16480)` without a compact bridge or tail.
An independent in-place Full-Attention preprocess admission preserves the
established centered RMSNorm/BF16/partial-D64 NeoX order directly on those Q
slots and the active private-NHD K slice without touching Gate, V, the scratch
gap, or prior/future K rows.
An independent fixed-C8000 Full-Attention admission then uses one unsplit
Q128/KV32 two-stage online-softmax kernel over the complete private NHD K/V
allocation origins and overwrites those Q slots only after the CTA's complete
Q128 tile is staged. Its bounded C1/C65 oracle is bitwise equal to the existing
public compact Q128-v4 body on nonzero Q/K/V/Gate data; the production instance
uses 254 registers, 128 KiB shared memory, zero local bytes, and one CTA/SM.
The GDN body preserves per-token BF16 state and active/candidate ownership in
bounded C1/C65 bit oracles. The startup package binds all 48 real BF16 A/B
pairs and all 64 real outer-norm pairs and exposes only construction-time
private full-catalog seals. It also implements an all-or-nothing natural-order
48-complete-GDN execution catalog (QKVZ, GDN-O, and four continuation weights)
and a 64-layer Gate+Up/Down catalog with exact live CUDA
allocation/upload-receipt matching. The normal path now also seals all 16
natural Full layers `3,7,...,63`: Full-QKV, interleaved-Q Full-O, exact Q/K
norm pairs, and one common four-resource bundle covering Full-QKV, Full-O,
in-place preprocess, and fixed C8000 online Attention. The fake host catalog
fails closed at ordinal zero. Its
normal startup and execution factories are private, the core static library
contains no named synthetic-T1 construction wrapper, and the remaining
private synthetic branch is confined to the BUILD_TESTING-only CUDA fixture
without production authority. One Engine-private
composition root now wires those normal catalogs into the existing
default-off V3 real-owner harness. The normal ExecutionPackage owns the exact
442,368,000-byte transient, 156,893,184-byte recurrent, and
2,621,440,000-byte private Full-KV arenas, exactly 3,220,701,184 device bytes.
It additionally owns one exact 160,008-byte portable pinned request-boundary
staging allocation, for 3,220,861,192 total package-owned bytes, without
charging those host bytes to the device reserve. It
borrows an exact 67,108,864-byte, 262,144-position × 32-pair cosine/sine table
from one shared Engine RoPE owner, for 3,287,970,056 total anchored
device-plus-host bytes. The build-time reserve chain preserves the
8,640,542,976-byte legacy P40 request arena and caller reserve at every
preceding allocation: complete target-AOT leaves 11,928,353,024 bytes plus
the caller reserve, RoPE creation leaves 11,861,244,160 bytes plus it, and V4
creation leaves the legacy arena plus it. Teardown is `execution → startup →
Engine RoPE → ModelWeights → complete target-AOT owner → resident`. A fixed
schema-v5 probe at `4f34c19` has now run this construction root through the
pinned real checkpoint. Engine creation/validation/destruction, the complete
target-AOT inventory, the normal 48-GDN/64-MLP/16-Full/one-boundary catalogs,
KV/RoPE/boundary ownership, pinned-staging release, reserve arithmetic, and
the owner/allocation/device chain were exact. The source result nevertheless
remains `status=fail` solely because its 419,917,824-byte post-destruction
free-memory gap exceeds the 33,554,432-byte tolerance. The passing root
post-exit resource preflight does not supply a Jetson `nvmap`
`no_owner_leak` classification. This is a bounded real-checkpoint
construction/catalog/ownership observation, not normal request execution,
numerical, generation, API, timing, performance, release, or production
authority.
The isolated package consumes layer-0
input norm, QKVZ, A/B, the two-kernel continuation plus one asynchronous
61,440-byte convolution-history D2D copy, GDN-O, residual/post-norm, Gate+Up,
and Down through typed owner-locked submissions. Its complete layer receipt
counts exactly nine kernels plus that copy. Raw ready-event records and
caller-filled QKVZ submissions are not exposed by the execution driver.
The reusable complete-GDN Events transaction now consumes a const move-only
GDN state grant and separates normal sealed-catalog authority from explicit
Synthetic-T1 authority. Its opaque authenticated receipt and fieldwise
semantic digest bind package/catalog/binding/norm/BF16-A/B/MLP identities,
grant-owned recurrent slices, the actual typed arguments, and seven concrete
resource snapshots; the issuing owner re-matches all of them before commit.
One direct O(1) `panel * 48 + gdn_ordinal` slot over the five-by-48 domain is
reserved before the first enqueue, so replay cannot add work. Accepted-prefix
evidence covers all nine kernels and the one history copy, including partial
continuation progress; a failure terminally drains and discards, while success
returns enqueue-only authority without a drain or publication.

The normal ExecutionPackage now has a reusable GDN composer over the sealed
48-GDN, 48-BF16-A/B, 64-norm, and 64-MLP catalogs. It derives natural model
layer, hidden-plane parity, assets, recurrent slices, and resources in O(1),
submits the authenticated nine-kernel/one-copy transaction, owner-matches the
receipt, and move-commits the same RequestState grant. Success does not drain,
swap banks, publish state, or complete the panel/model; failure terminally
drains/discards. No runtime-positive normal-package invocation has run. The old
one-shot package operation is hard-gated to Synthetic-T1 and always ends in a
physical drain/discard; it is not a fallback for normal authority.

The private Full Events transaction strictly separates
`NormalSealedCatalog` from `SyntheticT1`: normal authority requires every
production package/catalog/binding/RoPE/resource identity and zero synthetic
source, while Synthetic-T1 requires zero production identities and one
nonzero synthetic source. Its semantic digest, opaque receipt, owner matcher,
and direct O(1) `panel * 16 + full_ordinal` replay slot bind that choice. It
submits one complete Full DAG on Main as exactly eight kernels and zero copies,
records every accepted prefix 0--8, and drains all streams plus discards
RequestState after an accepted-prefix or replay failure. It leaves the GDN
BF16 A/B event/cycle state unchanged. The normal ExecutionPackage has the
private construction-sealed composer for this exact transaction.
It derives layer/parity/aliases, KV offsets, first position, RoPE, assets, and
resources in O(1), performs no request-time query or scan, authenticates the
eight-kernel receipt, and commits the same move-only KV grant. Its outer normal
receipt cross-checks all package, Full/MLP/norm/RoPE/resource, request, grant,
KV-range, layer, authority-domain, synthetic-source, and nested opaque Events
identities. Success does not drain or publish; failure terminal-drains/
discards. The synthetic package branch still proves only zero-enqueue
fail-closed rejection, and neither normal composer has a runtime-positive or
real-checkpoint invocation.

One joined Events CUDA T1 fixture now supplies the physical dependency that
the formerly isolated transactions lacked. On one request and panel it reuses
the natural hidden parity/scratch, a complete 16-layer KV arena and complete
RoPE tables, then executes `GDN0 -> GDN1 -> GDN2 -> Full3` under strict
Synthetic-T1 authority as exactly 35 kernels plus three 61,440-byte D2D
history copies. Three GDN grants and one Full grant are owner-authenticated and
move-committed; one normal combined dual-tail drain/discard then issues the
only physical completion receipt. No bank swap, KV valid-end, canonical state,
sequence fence, Decode access, panel/model completion, or publication occurs.
The minimum failure retains the exact 27-kernel/three-copy GDN prefix, rejects
Full3 before its first enqueue, poison-drains all streams, and invalidates the
pending Full grant.
No complete layer is numerically qualified at full C8000 with the real
checkpoint. The startup
package regenerates the canonical plan, validates the live
256-artifact/400-source catalog once, retains typed payload/scale capabilities,
and mints V4-local resource seals with no launcher authority.
Execution-package construction may
perform the expensive BF16/model/CUDA validation once; request execution may
not rescan it. These ownership facts, bounded bit
oracles, projection resource gates, event graph, and the Attention one-CTA/SM
gate do not create a whole-product result. There is no V4 selector, whole-model
launcher, 64-layer/request execution package, real-checkpoint oracle, API
timing, numerical qualification, or
production eligibility yet. The
joined synthetic receipt proves four-layer enqueue order and physical
quiescence while explicitly denying panel, model, numerical, API, performance,
release, and production completion. These boundaries are blockers, not
implied future facts.

Four additional fixed outer-edge leaves exist behind source-private,
construction-prevalidated launch seams: one exact BF16 C8000 Embedding gather,
one centered RMSNorm M1 intended for the final hidden row, one exact
activation-staged NVFP4 LM-head M1 producing BF16 vocabulary logits, and a
two-kernel BF16 greedy argmax. Their BUILD_TESTING-only Synthetic-T1 CUDA
fixture supplies narrow resource, argument/alias/error, and synthetic
correctness evidence, including deterministic earliest-tie and nonfinite
reporting.

Normal Startup has now closed their immutable source boundary independently
from the 256-entry projection catalog and against the authenticated real
Resident owner. Its capability-free seal binds Embedding, final norm,
canonical LM-head packed weight/block scale, `weight_scale_2` and
`input_scale` raw-bit provenance, and the fixed greedy specification;
`input_scale` is retained but not consumed. Host-test Resident authority may
exercise the T0 source inventory but cannot mint or upgrade to a normal
execution binding. The construction-only normal Execution resource-seal API
also exists and fails all-or-nothing. Normal Execution construction now
invokes it exactly once after four source-private resource queries, retains the
sealed catalog and independent fold, acquires and zeroes exact 160,008-byte
`cudaHostAllocPortable` staging before the device reserve query, and binds the
exact 507,144-byte phase-aliased scratch span. Synthetic-T1 keeps these facts
explicitly zero/unbound. Every construction rollback checks `cudaFreeHost`;
the fixed `4f34c19` probe observed stale pinned-address invalidation after
Engine destruction, while its overall source result remains failed solely at
the strict memory-recovery gate described below. There is therefore still no
fixed request-edge submission, request-boundary leaf
invocation, or normal runtime-positive invocation. The leaves grant no
whole-panel, whole-request,
real-weight boundary oracle, real-checkpoint numerical, API, timing, release,
or production authority.

Its P40 allocation remains:

- no more than 5.0 s for all NVFP4 and FP8 projections;
- no more than 1.8 s for exact full Attention;
- no more than 1.5 s for exact GDN plus BF16 A/B work;
- about 1.0 s for all remaining work; and
- no more than 9.302326 s total, equivalent to the 4,300 prompt tok/s starting
  line on exactly 40,000 consumed prompt tokens.

The v1 diagnostic control exposes host-visible per-layer progress and
propagates client cancellation through bounded device-safe points. Its 64
host waits are diagnostic serialization, not a V4 schedule. V4 must retain
bounded progress/cancellation through device-ordered work, discard the private
candidate recurrent epoch on panel failure, and keep KV/sequence visibility
private until the final request commit. A future pre-P40 admission must prove
those operations rather than infer them from the host plan.

These are planning allocations derived from the owner-set target, not
hardware-bound claims. The candidate must cover Gate/Up, Down, FP8 QKV/Z/O,
Attention, GDN/SSM, residual/layout consumers, state publication, and their
live producer/consumer boundaries in one authenticated AOT execution plan.
It may not introduce a full-model BF16 weight copy, cuBLASLt production path,
MTP, approximate arithmetic, request-time JIT/repack/autotune, silent
fallback, or request-time tactic discovery.

Two candidate classes are now closed before implementation. An impossible
one-pass dense INT4 mapping of the listed P40 projection work still exceeds
the 5.0-second projection allocation, so exact dense integer limb/bit-plane
re-expression is not an active successor. A real-checkpoint
early/middle/terminal screen also found zero repeated packed E2M1 K16 keys in
more than fifty million Gate/Up/Down role-block instances, including the optimistic
scale-ignored form, so dictionary and identical/proportional-code cross-role
reuse are not successors. These results preserve rather than lower the 4.3K
target. They made matched route and executed-work reconciliation the P0 for
projection selection.

That matched ledger is now complete. It confirms all 64 layer bodies, 192
NVFP4 MLP roles, 208 FP8 checkpoint roles, 96 BF16 A/B roles, and the existing
`1.948-Pop` projection total. The retained SM87 vLLM route is W8A16/W4A16 and
does not remove the work through activation quantization, MTP, cache, or
terminal liveness. It also separates 128 fused outer calls from the pinned
Marlin source's expected 5,120 P40 physical launches per quantized family;
Q3X v10 records 1,040 FP8 and 128 NVFP4 physical launches. Launch count is
therefore not the missing architecture. The exact closure and remaining route
unknowns are frozen in the
[matched-work evidence](metadata/qwen36-27b-prefill-p40-matched-work-ledger-2026-08-14.json).
Its default-off executable checker binds every exact role/shape/scale
partition to outer-operation and physical-launch identities, so subsequent V3
API receipts can fail closed on missing or substituted work instead of
reconstructing the ledger after timing.

The first bounded package is
**[`WP-PREFILL-REFERENCE-TRANSLATION-v1`](PREFILL_REFERENCE_TRANSLATION_MATRIX.md)**:

1. retain the completed vLLM, FlashInfer SM8x, Humming/Triton, FLA, and Mamba
   source translation and the explicit SM87 load, buffering,
   synchronization, MMA, residency, and AOT map;
2. retain the thermally invalid stock-vLLM P40 attempt as route and metric-
   semantics feedback only. It has no performance result and does not replace
   the owner-established 4.3K starting line;
3. do not repeat the unchanged stock reference or let reconstruction of the
   unknown optimized reference route delay the native V4 gate. A future
   reference run requires a materially different, hash-bound configuration;
4. retain the selected role-specific macro feeds and exact-GDN macrochunk
   geometry in the C8000 panel-major V4 AOT candidate covering the complete
   family set;
   and
5. return the first executable composition directly to the incumbent P40 API
   gate. A material whole-path step advances qualification; a small complete
   result reopens the global dataflow instead of starting a parameter scan.

FlashInfer and the other engines remain source and behavioral references, not
runtime dependencies. Offline JIT/autotune may reveal a tactic; the native
release contains only the authenticated AOT result. The package records added
work, eliminated movement, resource transfer, exactness, and API effect under
the Constitution's compute-for-movement rule.

### Constituent plans and dependency order

The reference-translation package must produce one source/dataflow matrix and
then freeze the following mutually dependent constituents before CUDA
implementation is selected:

- **SM87 AOT projection plan:** distinct packed-operand ownership and
  load/decode/MMA schedules for NVFP4 Gate/Up, K-heavy NVFP4 Down, and FP8
  QKV/Z/O, selected only after the matched production work ledger closes the
  current 1.948-Pop/reference-route discrepancy. One universal tile,
  persistent-grid skeleton, or nominal low-bit ISA is not assumed.
- **`WP-P40-EXACT-ATTENTION-v1`:** exact whole-prompt online-softmax
  Attention with ordered KV publication, consumer-native Q/gate/layout
  handling, and a declared sequence-parallel work plan. FlashInfer and
  FlashAttention supply reference mechanisms; the provisional P40 allocation
  is 1.8 s.
- **`WP-P40-EXACT-GDN-v1`:** one prompt-span recurrent work graph with
  chunk-local parallel work, only the mathematical boundary-state dependency
  serialized, exact BF16 state/publication semantics, and panel-wide BF16 A/B
  tactics. V2 proved the 48 independent value-head ownership and exact C64
  arithmetic, but its 30,000 prepare/recurrence/epilogue chunks plus
  cancellation samples are rejected as a production submission topology. V3
  retains every token's BF16 state publication and pre-round same-token output
  use while V4 maps the recurrence to a panel-sized layer-persistent or
  large-macrochunk graph targeting O(10) physical kernels per GDN layer.
  WY/KKT/SSD and FP32 authoritative
  chunk state remain forbidden. FLA and Mamba selective-scan supply
  organization mechanisms only; the provisional GDN plus BF16 A/B allocation
  is 1.5 s.
- **Handoff and composition plan:** one final exact Prefill state publication,
  bounded arena/control-state lifetimes, no illegal aliasing, and explicit
  overlap only where the dependency graph permits it.

Exact arithmetic-class qualification and terminal-layer liveness deletion
remain documented in the
[mathematical-equivalence ledger](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md).
They are not independent active scans. The architecture may compose them only
when the source/dataflow matrix shows that they serve the complete candidate
and its current critical path.

The matched work ledger, v10 control, V2 closure, and complete default-off V3
control are no longer unfinished Roadmap items. The V4 host contract, its
owner/epoch-bound request transaction, its authenticated startup foundation
package, its isolated execution package, its normal 16-Full catalog/KV/RoPE
lifetime foundation, and its admitted constituents are implementation facts
only: the shared
canonical decoder has an asymmetric nibble/scale bit oracle, both projection
tests cover the two N128 halves and M64/M37, and both production kernels pass
their two-CTA/SM resource gate. The request transaction additionally proves
five atomic panel swaps, private KV valid-end, cancellation/poisoning, and the
logical final fence. A separate CUDA owner now supplies three nonblocking
streams and nine private events, enforces 48 AB readiness cycles per panel,
keeps panel closure device ordered, physically observes only final/safe-drain
boundaries, and permanently poisons itself after exceptional drain. Its narrow
driver now executes one complete synthetic-T1 GDN layer 0: one InputNorm, one
GDN-QKVZ, one BF16 A/B, two continuation kernels, one GDN-O, one
ResidualPostNorm, one GateUp, and one Down, plus one asynchronous 61,440-byte
D2D copy. The QKVZ and A/B write sets are disjoint and their dependency graph
permits physical overlap; no performance overlap is claimed. Combined
Events/RequestState drain proves physical quiescence, candidate discard, and
pending-grant invalidation on poison without a bank swap or publication. Its
owner-mediated drain identity is consumed by RequestState and cross-checked by
the complete receipt together with `physical_execution_receipt_issued`. The
startup package retains no execution-owning capability, and the receipt grants
layer completion only—not panel, model, production, or Decode authority. The
fixture uses zero-valued synthetic weights; none of these tests has real-model,
numerical, or performance authority.

The secure complete-GDN Events path and normal package composer are now narrow
implementation facts rather than remaining design work. The Events receipt is
grant-bound, opaque, fieldwise-digested, owner-authenticated, and protected by
a direct five-panel-by-48-ordinal O(1) at-most-once ledger. Same-panel
Synthetic-T1 transactions for GDN0, GDN1, and GDN2 accept 27 kernels plus
three 61,440-byte history copies. The joined fixture continues from that exact
physical prefix into Full3 on the same request/panel, reaching 35 kernels and
three copies before one normal combined drain/discard. The normal package
composer derives the natural layer and hidden parity from sealed
GDN/BF16-A/B/norm/MLP catalogs, then records nine kernels, one copy, owner
match, and move-commit without a success drain or publication. Its runtime-
positive path has not run. The old one-shot package fixture remains
Synthetic-T1-only and terminally drains/discards.

The private Events T1 Full transaction independently submits
`InputNorm → FullQKV → preprocess → Attention → FullO → ResidualPostNorm →`
`GateUp → Down` as eight Main kernels and zero copies. Its move-only KV grant,
strict normal/Synthetic-T1 authority split, semantic submission digest,
direct five-by-16 O(1) at-most-once ledger, exact 0--8 accepted-prefix
failures, terminal drain/discard, and unchanged BF16 A/B boundary pass
synthetic CUDA tests. The normal source/build path separately
seals 16 Full bindings plus the four Full resource observations, owns the exact
2,621,440,000-byte KV arena, and borrows the exact 67,108,864-byte shared
Engine RoPE owner under the closed reserve/lifetime chain. The private normal
composer now joins those facts to the eight-kernel Events transaction and
commits the matching RequestState KV grant in O(1). Its outer receipt binds
the normal domain, zero synthetic source, all package/catalog/binding/RoPE/
resource and request/grant/KV-range identities, and the exact nested opaque
Events receipt. Its synthetic package test is
deliberately negative because that fixture has no Full physical owner: it
fails before enqueue with zero added kernels, then drains and discards. No
runtime-positive normal-package invocation of either composer and no
real-checkpoint numerical invocation has run. The joined Events T1 cohort
instead uses the
strict Synthetic-T1 domain, complete KV/RoPE, and the same request/panel to
physically execute natural GDN0--2 and Full3 as 35 kernels plus three copies.
Its success has one normal combined drain/discard and no publication; its
minimum failure retains 27/3 and poison-invalidates the pending Full grant.
The separate fixed schema-v5 real-checkpoint Engine probe closed the exact
construction/catalog/ownership and pinned-release observations but preserved
`status=fail` at its sole free-memory-recovery check. It did not execute either
normal composer. These transaction fixtures remain synthetic
correctness/dependency facts only; none has whole-model, API, numerical,
performance, release, or production authority. The owner-only panel-commit
bridge is now implemented separately: runtime rearm enqueues the exact
156,893,184-byte Main zero and atomically creates a fresh epoch plus panel 0;
exact 48-GDN/16-Full, 560-kernel, 48-copy/2,949,120-byte and 48-A/B ledgers
gate device-ordered `PanelDone` before Events directly commits RequestState.
Panel capabilities are move-only optional values with no per-panel heap.
Healthy physical discard can rearm; poison-terminal and successful-request
reuse cannot. The focused lifecycle fixture seeds the 560-kernel counter
ledger, so it is not a physical whole-panel witness.

The Events name `FinalRepresentationReady` remains a diagnostic computation
endpoint only. It does not attest canonical state publication and cannot be
used as the pure-Prefill stop. The normative interval still ends at the atomic
RequestState handoff, `PrefillStateCommitted`. LM-head/argmax may overlap once
their final-row dependency is satisfied, but qualification must bracket the
normative state-publication path with independent timing-enabled Control
events and exclude those finalizer kernels; neither a representation-ready
timestamp nor duration subtraction is admissible.

Frozen recovery order (inactive until the owner explicitly resumes Prefill
performance work):

1. place the owner-atomic panel lifecycle behind the normal package's fixed natural
   64-layer loop and execute the loop for all five C8000 panels, with one
   Embedding H2D/gather entry per panel using the already construction-sealed
   request-boundary catalog, portable staging, and scratch aliases. Add
   complete failure/cancellation rollback without a per-layer or per-panel
   host drain;
2. close the final state transition: exact epoch-five `B -> A` canonical copy
   when required by parity, `FinalPublish`, physical completion receipt, and
   the logical sequence-length fence as the last non-fallible visibility
   transition, then make physically successful requests eligible for cold
   reuse. Add independent timing-enabled Control events around canonical
   RequestState publication so qualified pure Prefill ends at
   `PrefillStateCommitted` and excludes any overlapped LM-head/argmax/D2H work.
   `FinalRepresentationReady` remains diagnostic only. Preserve the frozen
   pinned-real-checkpoint Engine closeout exactly: its construction/catalog,
   KV/RoPE/boundary ownership, and pinned-release observations passed, while
   the source remains `fail` because 419,917,824 bytes did not recover within
   the 33,554,432-byte tolerance. If formal lifecycle acceptance is required
   after resume, reconcile that gap with an immediate canonical Jetson
   `nvmap`/owner snapshot or a separately admitted replacement probe; the
   post-exit resource preflight alone is not `no_owner_leak`. Then run the
   remaining proportional C8000 GDN/Full/cohort/loop/boundary numerical gates.
   Synthetic T1 cannot substitute for those results, and correctness/lifetime
   probes have no performance authority; and
3. expose exactly one default-off V4 route through the existing real OpenAI
   P40 API, then run one clean-host cold/no-cache EvalScope direction witness.
   A negative or merely small whole-path result reopens the V4 global
   dataflow; it does not authorize a local parameter scan. Do not add a
   synthetic timing campaign or a normal-prefix probe before this return
   point. P60 and P130 remain locked until the competitive,
   accuracy-admissible P40 gate passes. cuBLASLt remains reference-only and
   cannot supply any missing request binding or runtime route.

An earlier admission stopped before execution because Xorg, GNOME, and Mutter
held GPU-device handles and produced no timing. The final construction
closeout later used passing root `tegrastats`/process/device-handle preflights
after the interfering vLLM process group was terminated; it ran correctness
and Engine-lifetime probes only, not a performance request or profiler. The
fixed source lifetime result remains `fail` at its sole recovery check, and
the passing post-exit resource preflight does not rewrite it. Future
performance work remains inactive until explicit owner resumption and must
still fail closed on the same clean-host gate.

### Promotion and stop gates

Before CUDA mapping, the candidate records its real-number equations,
finite-precision reduction/publication/state boundaries,
production-observable live graph, buffer/control-state lifetime and alias
ownership, communication lower bound, and intended operand residence/reuse.
The stable requirements are owned by the
[Prefill subsystem SDD](PREFILL_ARCHITECTURE_RESET.md#8-stable-implementation-constraints).

A P40 direction is retained locally only under the real-model performance
policy. It advances to correctness and statistical qualification only after a
material positive whole-API result. A negative complete composition closes its
own version after at most one bounded causal profile. No P60, P130, EvalScope
dataset matrix, or low-yield tile/stage/cache scan may displace that return
point. Only a competitive, accuracy-admissible P40 result unlocks P60 and then
the approximately-130K witness.

Every candidate must answer:

- how prompt spans and layers are traversed without repeated dead work;
- where weights, scales, activations, KV, recurrent state, and scratch remain
  resident over their useful lifetimes;
- how Gate/Up, Down, FP8 projection, Attention, and GDN receive shape-specific
  ownership while composing in natural model order;
- which exact dependency justifies every stream, event, buffer, and overlap;
- how 40K, 60K, and approximately 130K capacities fit without request-time
  allocation growth or silent truncation; and
- which offline-selected tactics enter the authenticated release plan.

Exit criteria, in order:

- same-workload vLLM parity on the shortest target witness;
- no regression on longer target witnesses, accuracy, Decode, or memory;
- the 40K--60K and approximately 130K locked SLOs; and
- a measured specialization advantage over the matched general engine.

P513, component timings, NSys, and NCU remain attribution tools. They do not
replace these exits.

## P4. Accuracy, stability, and release qualification

Purpose: turn a positive architecture into a deliverable release.

Deliverables:

- pinned deterministic token/logit/state oracles;
- parseable public capability evaluation with no output truncation;
- independent-process mirrored performance repetitions and tail latency;
- long-output, cancellation, malformed-request, OOM, and shutdown stability;
- resource-preflight, temperature, clocks, memory, artifact, corpus, raw
  EvalScope, server-log, and route evidence in one bundle;
- release installation and startup documentation from a clean checkout.

Exit criteria:

- hard accuracy and API constraints pass before performance is considered;
- all release evidence is reproducible from tracked commands and retained raw
  artifacts;
- a fresh process reaches attested ready state and serves every declared
  capacity profile without hidden tuning or fallback.

## P5. Startup and operational hardening

This slice follows, rather than precedes, a viable target-length runner unless
startup behavior directly blocks its deployment contract.

- authenticate and load a static AOT deployment plan;
- bound cold load, sidecar preparation, memory peak, and warmup;
- remove request-path compilation, autotuning, repacking, and workspace growth;
- add packaging, service supervision, telemetry retention, and upgrade/rollback
  procedures.

## P6. Additional model and feature families

Only after the dense 27B runner clears the product gates:

1. Qwen3.5/Qwen3.6 35B-A3B MoE on a separately pinned checkpoint and SDD
   extension;
2. continuous batching or multi-request service where a real product workload
   selects it;
3. MTP, speculative decoding, vision, or other features under explicit new
   product contracts.

MTP remains ineligible to satisfy the current Prefill or Decode target.

## Work-package admission template

The admission fields and lifecycle are owned by the
[system SDD](SDD.md#7-local-work-package-boundary) and the
[real-model performance policy](REAL_MODEL_PERFORMANCE_POLICY.md#named-local-optimization-work-packages).
Each active phase above records only its concrete package instance and return
gate; this roadmap does not maintain a second generic template.

## Historical record

The pre-SDD component-first and append-only roadmap is retained unchanged in
[`ROADMAP_LEGACY.md`](ROADMAP_LEGACY.md). Its completed milestones and
measurements remain valid only for their recorded protocols. They do not own
current status, priority, production eligibility, or terminal targets.
