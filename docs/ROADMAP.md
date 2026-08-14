---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-08-14
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

The reserved active successor lineage is
**`AC-PREFILL-SM87-MACROFEED-v4`**. At the present boundary it is an
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

The tracked V4 foundation is deliberately non-executable and default-off. It
contains the host traversal/workspace/state contract, the owner/epoch-bound
host request transaction, a private authenticated startup foundation package,
independent C8000 NVFP4 Gate+Up and Down constituents, four layout-specific FP8
admissions, a fixed direct-scratch BF16 A/B admission, and an admission-only
two-kernel exact-GDN C8000 continuation. The FP8 tactics directly publish GDN
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
bounded C1/C65 bit oracles. None of these constituents is privately bound or
numerically qualified at full C8000. The startup package
regenerates the canonical plan, validates the live 256-artifact/400-source
catalog once, retains typed payload/scale capabilities, and mints V4-local
resource seals with no launcher authority. These ownership facts, bounded bit
oracles, projection resource gates, and the Attention one-CTA/SM gate do not
create a whole-product result. There is no V4 selector, whole-model launcher,
authenticated device execution
package, real-checkpoint oracle, API timing, numerical qualification, or
production eligibility yet. Those boundaries are blockers, not implied future
facts.

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
owner/epoch-bound host-only request transaction, its authenticated startup
foundation package, and its first two NVFP4 constituents are implementation
facts only: the shared
canonical decoder has an asymmetric nibble/scale bit oracle, both projection
tests cover the two N128 halves and M64/M37, and both production kernels pass
their two-CTA/SM resource gate. The request transaction additionally proves
five atomic panel swaps, private KV valid-end, cancellation/poisoning, and the
logical final fence, but its event completions are explicitly test-only and it
issues neither a CUDA execution receipt nor Decode access. The startup package
retains no execution-owning capability and binds no device event or missing
executor. None of those tests has real-model or performance authority.

Remaining execution order:

1. compose the existing private startup foundation package and host request
   transaction under an Engine-owned lifetime root, then bind device-owned
   panel/final events while retaining two recurrent epochs, private KV
   valid-end, rollback, reverse destruction order, and no production selector;
2. bind all 48 real BF16 A/B pairs plus the admitted preprocess → fixed C8000
   Attention → O chain with per-layer physical readiness events, then privately
   bind and qualify the complete C8000 BF16 A/B/FP8/GDN/Attention bodies,
   residual/layout, finalizer, rollback, and receipt paths;
3. expose exactly one default-off V4 route through the existing real OpenAI
   P40 API and verify that its receipt reports executed physical work rather
   than a forecast; and
4. only then run one clean-host, real-checkpoint, cold/no-cache P40
   OpenAI/EvalScope direction witness after
   `tegrastats`/process/device-handle preflight. A negative or merely small
   whole-path result reopens the V4 global dataflow; it does not authorize a
   local parameter scan. P60 and P130 remain locked until the competitive,
   accuracy-admissible P40 gate passes.

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
