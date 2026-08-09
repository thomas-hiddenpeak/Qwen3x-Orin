---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-09
  last_reviewed: 2026-08-09
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

## Locked outcome

The first proof vehicle is the pinned Qwen3.6-27B-NVFP4 checkpoint on Jetson
AGX Orin SM87. The runner must:

- expose the declared OpenAI-compatible text API;
- preserve the production numerical and generated-behavior contract;
- serve cold, no-cache 40K--60K prompts to first response within two seconds;
- serve the pinned approximately 130K prompt contract within four seconds;
- decode one request at no less than 10 token/s without MTP;
- use no cuBLASLt production dependency or fallback; and
- match, then exceed, the useful same-workload vLLM starting line.

The exact token counts, output capacity, endpoint, cache state, model identity,
and release artifact for each witness are defined in the SDD and evaluation
contract. Current support and gaps are reported only in Current Status.

## P0. Documentation control plane

Purpose: preserve project intent and system truth across agents, branches, and
context compaction before further architecture work.

Deliverables:

- [complete] Constitution updated with end-state-first, downward constraint
  leakage, upward value leakage, and controlled evolution.
- [complete] Canonical API-first SDD and Current Status.
- [complete] Complete registry for every tracked Markdown document.
- [complete] Candidate taxonomy separating local mutations, architecture
  candidates, and release candidates.
- [complete] Legacy Roadmap retained as history; active Roadmap contains no
  append-only experiment ledger.
- [pending] Machine-checkable documentation rules for registry coverage,
  status headers, links, and supersession.

Exit criteria:

- every tracked Markdown file is classified;
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

Selection sequence:

1. Establish a clean-host real API baseline on the canonical release.
2. Decompose target TTFT into externally visible and engine phase budgets.
3. Select the dominant complete execution-plan boundary, not an isolated
   kernel chosen from historical convenience.
4. Define one architecture candidate and its bounded local mutations.
5. Combine the required mutations in one binary and return to a target-length
   API witness as soon as dependencies close or the composition budget expires.
6. Qualify a positive composition, or close/redesign a negative architecture
   version after at most one bounded causal profile.

Prepared architecture lineage and current integration state (executable
compatibility route, not yet selected or production):

- `AC-PREFILL-LAYERMAJOR-8K-v1` replaces repeated public C512 full-model walks
  with one whole-request layer-major pass and bounded logical C8192 panels.
  `C8192` is internal scheduling capacity, not a public tile or partial
  state-commit boundary.
- At implementation baseline `606ea1d`, an explicit development route
  provisions the layer-major request profile and executes true
  `layer -> logical panel -> physical segment` traversal through
  `ReferenceEngine` and `qwen3x-eval-server`. It still lowers every logical
  panel to ordered physical segments of at most C512, so it closes the system
  transaction/API shape without yet closing the large-M performance design.
- Engine creation seals the model, state, arena, typed views, main/auxiliary
  streams, two completion events, and all 17 required operator-role receipts.
  The request receipt is move-only. Transcript allocation precedes GPU work;
  final hidden/logits remain uncommitted until one final state publication;
  every earlier failure drains and resets the runner.
- The API supplies disconnect/shutdown cancellation through a two-slot bounded
  submission window. It polls after retiring the oldest `layer x panel`
  quantum and again after final norm, before logits, and before commit. A
  cancelled request publishes no partial position or success witness.
- The compatibility route fails closed unless the SM87 backend, C512
  compatibility workspace, layer-major memory profile, and both exact FP8 and
  NVFP4 Marlin Prefill inventories are present. Those inventories remain
  development/test admissions; legacy stays the default route.
- The indivisible composition boundary still includes shape-specific NVFP4
  Gate/Up and Down, FP8 projections, exact causal Attention/KV progress, exact
  GDN/SSM progress, residual/layout consumers, and one final
  `PrefillStateCommitted` event. No isolated subset may claim the candidate
  result.
- The host workspace planner reports exact request-arena bytes for the target
  tactic profile: C8192 C64-native GDN with in-place convolution, current
  Release/default legacy C16 GDN, and separate Gate+Up then SiLU. C64-native
  and native large-M admissions remain non-production today.

  | Prompt tokens | Caller-selected conditional profile | Conservative disjoint profile |
  | ---: | ---: | ---: |
  | 40,000 | 3,975,374,848 | 5,324,963,840 |
  | 60,000 | 5,496,014,848 | 7,052,323,840 |
  | 130,000 | 10,818,254,848 | 13,098,083,840 |

  `selected` here names only the caller's explicit host-planner strategies; it
  is not architecture or production selection. The disjoint profile separates
  the three operator families and legacy workspace but still requires each
  tactic's named phase layout. Token-parallel C64 convolution raises its rows
  to 5,335,449,600 / 7,062,809,600 / 13,108,569,600 bytes; fused Gate+Up raises
  the overlay from 855,638,016 to 940,572,928 bytes. All shown rows fit the
  17,437,720,576-byte request-arena limit, but model bytes, sidecar bytes, and
  total whole-process bytes remain unknown, so whole-process capacity is
  `kIndeterminate`. These totals include a disjoint 10,240-byte final-hidden
  handoff. The runtime compatibility `RequestState` combination—C8192 family
  overlay plus physically disjoint legacy C512 workspace—reserves
  4,066,344,960 / 5,588,904,960 / 10,917,864,960 bytes for 40K/60K/130K.
- Isolated true-C8192 NVFP4/FP8 projection surfaces remain outside the bound
  compatibility executor; it deliberately uses the admitted exact M512
  Marlin paths. The next architecture change must bind native large-M tactics
  rather than mistaking logical-panel scheduling alone for the performance
  candidate.
- The immediate gate is a clean-host short exact real-model API sanity request,
  followed by capacity/resource closure and the exact 40K, 60K, and
  approximately-130K real-Agent API witnesses. No panel throughput or
  component profile substitutes for those results.

The complete subsystem design and non-production status are recorded in
[`PREFILL_ARCHITECTURE_RESET.md`](PREFILL_ARCHITECTURE_RESET.md#8-designed-candidate-lineage-ac-prefill-layermajor-8k-v1).

Required architecture questions include:

- layer-major prompt-span execution versus repeated tile-major layer walks;
- host and stream synchronization boundaries;
- weight and sidecar residency across target-length Prefill;
- shape-specific Gate/Up, Down, FP8 projection, Attention, and GDN ownership;
- exact recurrent-state update and consumer-native fusion boundaries;
- real double/multi-stage overlap where dependency analysis permits it;
- AOT deployment tactics learned from vLLM, Humming, FlashInfer, Triton, FLA,
  and Mamba without introducing runtime JIT as a requirement.

Exit criteria, in order:

- same-workload vLLM parity on the shortest target witness;
- no regression on longer target witnesses, accuracy, Decode, or memory;
- the 40K--60K and approximately 130K locked SLOs;
- a measured specialization advantage over the matched general engine.

P513, component timings, NSys, and NCU remain necessary attribution tools.
They do not replace these exits.

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

Every local optimization work package must record:

- originating product requirement and API symptom;
- attested release route and affected execution-plan node;
- baseline phase and local component budgets;
- architecture candidate and required local mutations;
- numerical and state contract;
- real weight/activation/state payload;
- maximum mutation count, elapsed budget, and stop loss;
- composition dependencies and deadline;
- target API witness and expected measurable system delta;
- rollback and evidence location.

Local mechanism documents become authoritative only through such an active
record. A package closes when it wins and is inherited by a qualified
architecture, loses at composition, exceeds its budget, or is superseded.

## Historical record

The pre-SDD component-first and append-only roadmap is retained unchanged in
[`ROADMAP_LEGACY.md`](ROADMAP_LEGACY.md). Its completed milestones and
measurements remain valid only for their recorded protocols. They do not own
current status, priority, production eligibility, or terminal targets.
