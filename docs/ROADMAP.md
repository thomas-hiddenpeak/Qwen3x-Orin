---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-08-12
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

### Active architecture candidate

The active successor is **`AC-PREFILL-SM87-AOT-SYSTEM-v1`**. It is one
complete Prefill composition, not a projection-first sequence of independently
selected kernels. Its P40 allocation is:

- no more than 5.0 s for all NVFP4 and FP8 projections;
- no more than 1.8 s for exact full Attention;
- no more than 1.5 s for exact GDN plus BF16 A/B work;
- about 1.0 s for all remaining work; and
- no more than 9.302326 s total, equivalent to the 4,300 prompt tok/s starting
  line on exactly 40,000 consumed prompt tokens.

These are planning allocations derived from the owner-set target, not
hardware-bound claims. The candidate must cover Gate/Up, Down, FP8 QKV/Z/O,
Attention, GDN/SSM, residual/layout consumers, state publication, and their
live producer/consumer boundaries in one authenticated AOT execution plan.
It may not introduce a full-model BF16 weight copy, cuBLASLt production path,
MTP, approximate arithmetic, request-time JIT/repack/autotune, silent
fallback, or request-time tactic discovery.

The first bounded package is
**[`WP-PREFILL-REFERENCE-TRANSLATION-v1`](PREFILL_REFERENCE_TRANSLATION_MATRIX.md)**:

1. read the relevant vLLM, FlashInfer SM8x, Humming/Triton, FLA, and Mamba
   source paths and separate invariant mathematics, dataflow, ownership,
   scheduling, specialization, and tuning from their ISA/resource shell;
2. map TMA/WGMMA/cluster/TMEM/native-FP4 or other unavailable mechanisms to an
   explicit SM87 load, buffering, synchronization, MMA, residency, and AOT
   plan instead of excluding the reference;
3. after source analysis, run one clean-host, real-weight, cold/no-cache P40
   API geometry witness for the proven reference route at explicit macrochunk
   budgets `2048`, `4096`, `8192`, and near-whole-prompt after its
   JIT/autotune warmup;
4. freeze the selected macrochunk geometry and role-specific plans into a
   Q3X-only AOT candidate covering the complete family set above; and
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
  QKV/Z/O. One universal tile or persistent-grid skeleton is not assumed.
- **`WP-P40-EXACT-ATTENTION-v1`:** exact whole-prompt online-softmax
  Attention with ordered KV publication, consumer-native Q/gate/layout
  handling, and a declared sequence-parallel work plan. FlashInfer and
  FlashAttention supply reference mechanisms; the provisional P40 allocation
  is 1.8 s.
- **`WP-P40-EXACT-GDN-v1`:** one prompt-span recurrent work graph with
  chunk-local parallel work, only the mathematical boundary-state dependency
  serialized, exact BF16 state/publication semantics, and panel-wide BF16 A/B
  tactics. FLA and Mamba selective-scan supply reference mechanisms; the
  provisional GDN plus BF16 A/B allocation is 1.5 s.
- **Handoff and composition plan:** one final exact Prefill state publication,
  bounded arena/control-state lifetimes, no illegal aliasing, and explicit
  overlap only where the dependency graph permits it.

Exact arithmetic-class qualification and terminal-layer liveness deletion
remain documented in the
[mathematical-equivalence ledger](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md).
They are not independent active scans. The architecture may compose them only
when the source/dataflow matrix shows that they serve the complete candidate
and its current critical path.

Execution order:

1. complete the reference/source dataflow matrix and SM87 translation;
2. choose macrochunk geometry and full-family ownership from that matrix;
3. freeze numerical, state, buffer, synchronization, and AOT-plan identities;
4. implement the mutually required projection, Attention, GDN, and handoff
   seams in one candidate binary;
5. return immediately to one clean-host real P40 API direction witness;
6. use a bounded profile only to answer a predeclared causal question that can
   select a materially different complete dataflow; and
7. qualify a positive composition or close and redesign the architecture
   version. Do not convert a negative result into a local parameter scan.

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
