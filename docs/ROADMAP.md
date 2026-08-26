---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-08-27
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

### Post-absorption whole-product acceptance gate

Every selected cross-branch absorption or integration batch must be closed on
the then-current mainline with the real pinned model, even when static review
shows that the selected delta cannot affect runtime. The minimum closeout is a
fresh `BUILD_TESTING=OFF` binary and one clean-host default-route run that
records exact source/binary/model/workload identity, hardware and locked-clock
state, process-spawn-to-ready time, process and system memory observations,
EvalScope request/latency/throughput results, route and streaming-protocol
receipts, thermal validity, bounded shutdown, and post-exit resource state.
Any rejected or invalid attempt remains explicitly classified and cannot
supply timing.

This gate proves integration health only. A performance-improvement claim
still requires a runtime-relevant delta, a matched retained comparator, the
applicable request count and target-length workload, and the repetition and
qualification required by the real-model performance policy. The gate cannot
waive P1/P2/P3 exit criteria or turn
the loopback evaluation adapter into the final product API.

The bounded 2026-08-21 owner-directed prompt-wide absorption batch has closed
this gate at runtime commit `ff47f179`. It selected only exact, allocation-free
Legacy-C512 Embedding gather and full-Attention preprocessing, rejected the
state/logit-inexact exact-C512 final-token candidate, retained matched
4K/8K/16K/32K direction evidence, and passed the final `BUILD_TESTING=OFF`
eight-request short EvalScope integration proxy. The exact result and its
non-statistical, non-target-length, non-release boundary are frozen in the
[`prompt-wide absorption record`](metadata/qwen36-27b-prompt-wide-mainline-absorption-2026-08-21.json).
At the time this was a branch-closeout exception rather than a P3 restart;
the later 2026-08-22 owner direction now supersedes that scheduling boundary.

The subsequent owner-directed v10 P40 retention exception made the exact
nine-item whole-core inventory reproducible as one typed, default-OFF,
`BUILD_TESTING=OFF`, non-installing development artifact. Current main
`b0c0c837` / tree `2963cb99` now closes the integration-health gate with a
strict four-valid-cell sustainable BCCB: candidate mean 325.983493208 tok/s,
baseline mean 326.111753524 tok/s, and C/B 0.9996066983. It passes the
predeclared 0.99 matched-baseline floor but is not a speedup and does not
reproduce the historical one-sample 392.804397 tok/s max-clock result. Full
max-GPU r4/r5 are invalid from over-current counter increases, not temperature;
all remained below 85C. Near-max GPU r6-r8 contain four individually valid C
cells with descriptive mean 375.956428659 tok/s and three valid B cells with
descriptive mean 375.935680902 tok/s (descriptive C/B 1.00005519), but external
CPU samples prevent any bundle from completing strict BCCB, so those cells are
not stitched into a performance decision. The route remains accuracy-
unqualified, non-release, non-production, and default-OFF. The exact boundary
and bundle hashes are frozen in the
[`P40 v10 mainline absorption record`](metadata/qwen36-27b-p40-v10-mainline-absorption-2026-08-21.json).

The first resumed production-conversion batch is now closed on current main
`c6c34ef` / tree `6a2df18`. The ordinary Release/OFF installed ELF
`ab3492a...` retains exact-span GDN, prompt-wide preprocessing, and both
Decode Gate/Up coupled-feed plus Down consumer-order layouts; split-KV remains
rejected. Same-ELF Decode BCCB is positive at P1024/P4096, the corrected
installed short EvalScope run passes 8/8, and a fresh C-only P40000 request
reports 57.230764 exact-default prompt tok/s. These are integration and
direction observations, not release qualification. Exact identities,
invalid-r1 classification, metrics, and limitations are frozen in the
[`current-main production-default closeout`](metadata/qwen36-27b-production-default-mainline-closeout-2026-08-23.json).

The follow-on 0.7.0 request-reuse conversion is now closed on current main
`8886119` / tree `fbbcac6` and installed Release/OFF ELF `d70ba913...`.
P1024/P4096/P8192 BCCB accepts 12/12 cells; every one of the six matched pairs
reduces the external-TTFT-minus-pure-Prefill bridge, and all 24 warmup plus
measured requests preserve output, SSE, usage, finish, and route receipts.
Pure Prefill and Decode are mixed or noise-scale and receive no improvement
claim. The same-server P40000 lifecycle measures committed-prefix cleanup at
15.185065 ms versus 17.504564 ms for conservative full cleanup, while proving
post-execution cancellation forces the next full reset and preserves exact
output/protocol/route receipts. This supports retaining the compiled policy as
the ordinary default across the verified lengths; it is not P40-throughput,
accuracy, release, or production qualification. Exact invalid-run boundaries,
identities, metrics, and bundle hashes are frozen in the
[`request-reset mainline closeout`](metadata/qwen36-27b-request-reset-mainline-closeout-2026-08-23.json).

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

## P3. Exact Prefill parity, then specialization advantage — active

Purpose: convert every accuracy-admissible whole-product gain into the
ordinary installed mainline, then close the remaining gap to the locked
Prefill targets.

Status: **resumed by explicit project-owner direction on 2026-08-22**. Work is
conversion-first: inventory retained real-API gains, re-evaluate them on the
then-current mainline, select accurate positive changes into the ordinary
Release default, and immediately return to the same installed API for
Prefill, TTFT, Decode, startup, and memory measurement. Ordinary P3 iteration
advances on the smallest decision-complete current-main API comparison;
hardware telemetry records and diagnoses the environment but does not become
the optimization objective or a blanket veto. Only confirmed interfering
workloads are quiesced, and the active control path remains available.
Architecture selection and release qualification pay the strict environment
and repetition cost defined by the
[`real-model performance policy`](REAL_MODEL_PERFORMANCE_POLICY.md).

The first active batch has promoted the exact C256/C512 GDN whole-span path
and the already-qualified prompt-wide Embedding/Attention preprocessing path
into the sealed ordinary profile. It retains both proven Decode coupled-feed
and consumer-order layouts as ordinary defaults, while explicitly rejecting
the split-KV Decode selector after its S65 BF16 boundary oracle differed from
the exact fallback. The next P0 inside P3 is the unconverted P40 result: the
current 0.7 same-server exact route samples 57.391208--57.585234 tok/s without
throughput-comparison authority, while the retained v10 transaction reached a
historical 392.804397 tok/s. Its known Attention-state mismatch must be
repaired or separately qualified, followed by a fresh current-main complete-
route comparison, before that arithmetic can become a production default.

The 0.7.0 request-reuse conversion is now compiled into the ordinary
`q3x.sm87.production.p40.legacy-c512-exact.v3` lifecycle: creation is already
clean, a prior exact
successful request clears only its committed K/V prefix plus complete
Conv/GDN state, and every uncertain boundary performs the existing full
reset. Its reset mode, positions, bytes, and synchronized duration are emitted
without an environment, CLI, request, or test selector. Host policy, protocol,
package-consumer, bounded CUDA correctness, and installed real-model API gates
now close the verified P1024/P4096/P8192/P40000 scope. The short BCCB is 6/6
positive on the targeted external-TTFT-minus-pure-Prefill bridge, and the P40
same-server lifecycle measures the prefix reset itself 2.319499 ms below the
full reset. Output, SSE, usage, finish, and route receipts remain exact. The
policy is therefore retained as the ordinary default across those lengths;
the evidence deliberately makes no material TTFT, Prefill-throughput, Decode,
accuracy, release, or production claim. This bounded qualification does not
displace the accuracy-admissible P40 v10 conversion as the next architecture-
scale P0.

### Retained recovery boundary

The exact recovery anchor is
`archive/v4-construction-ownership-20260820@f3545240075651eaa54a5bea6c0f15ee9dfd9a3e`.
It is an explicitly incomplete, default-off architecture foundation/local work
package, not a Constitution-defined runnable `architecture_candidate`, and it
is neither contained in nor selected by this mainline. The archive's
schema-v5 closeout source remains `status=fail` because the 419,917,824-byte
post-destruction free-memory gap exceeds the fixed 33,554,432-byte tolerance.
The later holder-free post-exit preflight is not a Jetson `nvmap`
`no_owner_leak` classification. The archive grants no normal request
execution, numerical, generation, API, timing, performance, release, or
production authority.

The explicit owner direction has resumed this work. Start every conversion
from the then-current mainline and use the archive only as a source/evidence
reference; do not merge it wholesale or infer capability from branch-local
tests.
The then-current mainline now already contains the narrow exact prompt-wide
Embedding/Attention absorption at `ff47f179`; treat it as the incumbent for
that eligible Legacy-C512 scope and use its
[`closeout record`](metadata/qwen36-27b-prompt-wide-mainline-absorption-2026-08-21.json)
instead of reopening the rejected final-token route or repeating branch-local
qualification by assumption.
The frozen first sequence is:

1. fixed normal 64-layer by five-C8000-panel loop plus the Embedding edge and
   complete failure/cancellation rollback;
2. exact epoch-five B-to-A/FinalPublish/sequence fence, successful-request
   reuse, and independent Control events around canonical
   `PrefillStateCommitted`;
3. preserve and reconcile the strict lifetime failure if formal lifecycle
   acceptance is needed, then run the remaining proportional real-checkpoint
   numerical gates; and
4. use the now-tracked, default-off `p40-whole-core-v10` development artifact
   as the exact historical-route recovery surface and complete one valid
   clean-host direction witness before any performance conclusion. The invalid
   2026-08-21 B1 diagnostic does not satisfy this step. A negative or small
   valid result reopens the global dataflow, never a local parameter scan.

### Promotion and stop gates

These gates remain resume-time exit criteria; they do not make P3 or any
candidate active.

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
