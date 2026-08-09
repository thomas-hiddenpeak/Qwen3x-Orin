---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-08-10
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

P3 implementation preparation may proceed in the same pinned development
route while P1/P2 capacity and release seams are being closed. It cannot
select an architecture, change the default, or promote production before the
P1 and P2 exits provide the target-length API and canonical release artifact.

Current P0 decision: both WP-V2-C1-v1 and its GateUpG2/DownD2 v1 successor are
implemented and rejected. The clean real-API G2/D2 P40K request consumed all
40,000 tokens and attested 320+320 coupled role hits, but server pure Prefill
was 115.041752 s / 347.699851 prompt tok/s versus 108.981855 s / 367.033577
tok/s for retained `c45b7c5`: +5.560464% latency and -5.267563% throughput.
EvalScope TTFT was 115.08576 s, so the API boundary remains negligible.
Bounded NSys measured G2+D2 at 60.904184288 s versus 53.864024544 s for the
retained native Marlin main plus SiLU scope, a +7.040159744-s role gap and
+6.110398080 s in complete GPU kernel time. Defaults are unchanged and no
accuracy promotion was run. Exact evidence is in the
[G2/D2 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-g2-d2-rejection-2026-08-10.json).

The active P0 is now a projection-ownership architecture reset, not another
parameter scan on either rejected raster skeleton. The successor must make
B/scale residency and decoded-fragment reuse persist across multiple M rows or
output tiles, and keep Gate/Up and Down shape-specific. Freeze the dispatch
contract for M8192/M7712 and the future M5424 tail now, but implement
M8192/M7712 and return first to the same P40K API witness. Only a competitive,
accuracy-admissible P40K composition unlocks M5424 implementation/validation
and P60; P130 remains locked behind those gates.

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
- At implementation baseline `18363ad`, an explicit development route
  provisions the layer-major request profile and executes true
  `layer -> logical panel -> physical segment` traversal through
  `ReferenceEngine` and `qwen3x-eval-server`. It balances the final logical and
  physical panel pairs, but still lowers every logical panel to ordered
  physical segments of at most C512, so it closes the system transaction/API
  shape without yet closing the large-M performance design.
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
  NVFP4 Marlin Prefill inventories are present. It also seals the real native
  GDN workspace, validates the 48 linear-attention layer/weight shapes, and
  requires exact native C64 GDN for each eligible M32--M512 segment; M1--M31
  alone may use the sealed exact fallback. Those inventories remain
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
- At current baseline `bbd8ac3`, explicit, default-off
  `native-group-q64-panel` Attention and
  `segmented-marlin-operator-panel` projection tactics are bound to the
  layer-major executor and emit v4 accuracy-unqualified witnesses. The exact
  layer-major tactics and server-wide legacy route remain the defaults. The
  projection tactic is deliberately named `segmented`: it groups logical
  operator panels but still lowers them to bounded physical Marlin launches;
  it is not a native large-M implementation or the indivisible architecture
  candidate.
- Revision `21d5c28` connects a distinct default-off
  `native-quantized-large-m-operator-panel` v5 tactic. It preserves the
  immutable balanced logical-panel topology and exact recurrent/Attention
  route. A complete M8192 panel issues one frozen Marlin launch for each
  logical FP8 or NVFP4 projection; every partial panel executes the complete
  authenticated oracle span ledger. Clean P513/P8192/P7712 state screens
  passed. The underlying Marlin body still decomposes large M internally, so
  this closes a host-submission seam rather than true cross-row weight reuse.
- A clean-host one-request P1025 OpenAI API/EvalScope direction screen on
  `18363ad` is cumulatively positive against its older greedy/fallback
  layer-major witness and remains slower than the older legacy observation.
  The comparison is not causal: the older witness used exact FP8, Attention,
  and GDN fallbacks, whereas `18363ad` reports every required role on its
  admitted production disposition. It retains the complete sealed composition
  direction only and does not select `AC-PREFILL-LAYERMAJOR-8K-v1`; current
  numbers are owned by [`CURRENT_STATUS.md`](CURRENT_STATUS.md) and exact
  evidence by the
  [machine-readable direction record](metadata/qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json).
- A clean-host real-API P40K direction run of that partial tactic observed
  222.204547 pure Prefill tok/s versus 59.974810 tok/s for the exact segmented
  Attention context, but generated a different first token and remains
  accuracy-unqualified. The following T4 trace records 221.796604 tok/s and
  places 46.8% of kernel time in grouped-Q64 Attention, 43.4% in the two main
  segmented Marlin signatures, and 3.0% in 120,000 BF16 M16-pair launches.
  It observes one CUDA stream and zero kernel overlap; external EvalScope TTFT
  exceeds server TTFT by only 3.738965 ms. This is diagnostic evidence only,
  not P40K target passage or architecture selection. P60K/P130K were not run.
  Exact evidence is frozen in the
  [P40K profile record](metadata/qwen36-27b-prefill-p40k-native-group-q64-panel-nsys-2026-08-09.json).
- The following clean-host P40K real-API screen at `bbd8ac3` exercised the
  segmented projection wrapper with grouped-Q64 Attention. EvalScope reported
  179,511.19 ms TTFT and 222.827281 prompt tok/s; server pure Prefill was
  179,395.678907 ms. External TTFT exceeded server TTFT by only 3.775881 ms,
  so the API is not the active measured bottleneck. The v4 witness recorded
  five logical panels, 1,680 segmented projection hits, 12,992 physical
  projection launches, 80 grouped-Q64 Attention hits, and no Prefix, MTP,
  cuBLASLt, external-reference, or generic approximate-route hits. The route
  remains accuracy-unqualified.
- That 179.51119-s result crossed the predeclared `>=165 s` stop-loss. The
  segmented-wrapper direction is rejected; P60K, P130K, NSys, and NCU were not
  run for it. The decision and raw-artifact hashes are frozen in the
  [P40K API record](metadata/qwen36-27b-prefill-p40k-segmented-marlin-q64-api-2026-08-09.json).
- The clean-host cold/no-cache P40K real-API gate for `21d5c28` consumed all
  40,000 tokens as `3x8192 + 2x7712`. EvalScope TTFT was 670.53071 s and server
  pure Prefill was 670.486890 s, or 59.658139 prompt tok/s. Pure Prefill was
  99.994017% of server TTFT; external overhead was 3.702638 ms. The witness
  recorded 1,008 bulk launches, 672 partial-oracle hits, 13,104 physical
  projection launches, and zero forbidden routes. It was about 0.53% slower
  than the prior same-payload exact-Attention observation. The M8192-only
  composition is therefore closed and P60K/P130K were not run. Exact evidence
  is frozen in the
  [v5 P40K API record](metadata/qwen36-27b-prefill-p40k-native-large-m-exact-api-2026-08-09.json).
- The existing same-payload T4 trace already establishes the architecture-level
  symptom: one CUDA stream and zero kernel overlap, with 46.8% of kernel time
  in the grouped-Q64 Attention upper-bound path, 43.4% in the two main Marlin
  signatures, and 120,000 BF16 M16-pair launches. The new exact route is far
  slower because exact Attention repeatedly scans causal K/V through bounded
  spans. A duplicate full trace is not required before redesign.
- Revision `c45b7c5` implements WP-V2-A as the default-off
  `native-flashinfer-exact-panel` route. Its clean-host cold/no-cache P40K
  OpenAI API/EvalScope screen consumed all 40,000 tokens, issued 80 exact
  FlashInfer logical-panel Attention calls, and issued zero QT2/Q64/Q128
  Attention calls. EvalScope TTFT was 109.02622 s; server pure Prefill was
  108.981855 s, or 367.033577 prompt tok/s. That is 6.15x faster than the
  preceding exact route and 1.65x faster than the grouped-Q64 direction, so
  the Attention dataflow is retained. It is not selected for production: the
  P513 full-state hash differs, and the P40K output `The` matches the preceding
  exact route but is only one token. The route remains default-off and
  accuracy-unqualified, and P60K and P130K were not run. Evidence is frozen in
  the
  [v6 P40K API record](metadata/qwen36-27b-prefill-p40k-flashinfer-exact-panel-api-2026-08-09.json).
- Revision `da2b9f6` implements WP-V2-C1-v1 as the distinct default-off
  `native-nvfp4-true-large-m-operator-panel` route. Its clean-host P40K API
  screen consumed all 40,000 tokens with exact FlashInfer Attention and zero
  Prefix, MTP, cuBLASLt, external, approximate, or forbidden fallback hits.
  Server pure Prefill was 136.929918 s, or 292.120 prompt tok/s, versus
  108.981855 s and 367.034 prompt tok/s for retained `c45b7c5`: a 25.64%
  latency regression. Pure Prefill was 99.97% of TTFT and the external/server
  boundary was 4.047 ms, so the API is not the bottleneck.
- The matched NSys pair changes the architecture decision. It attributes
  99.30% of the whole-request interval increase to NVFP4. Gate+Up increased
  from 35.039284 s to 52.600776 s (1.501x), and Down from 17.051106 s to
  27.094866 s (1.589x), despite reducing their combined physical launches from
  4,992 to 640. The M128N256K64, three-stage v1 consumes 203/220 registers per
  thread, reaches only one CTA/SM, duplicates decoded B across M warp rows, and
  uses two full-CTA barriers per K64 step. WP-V2-C1-v1 is rejected; defaults
  are unchanged, and accuracy promotion, P60K, P130K, and NCU were not run.
  Exact evidence is frozen in the
  [v1 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-true-large-m-rejection-2026-08-10.json).

Active architecture candidate: `AC-PREFILL-PROMPT-WIDE-v2`.

- **WP-V2-A — exact logical-panel Attention — implemented, retained, not
  accuracy-admissible:** `c45b7c5` replaces repeated C512 exact Attention with
  an M8192/M7712 FlashInfer logical-panel path using exact causal masking and
  FP32 online-softmax state. The P40K product interval moved by 6.15x, which
  closes the architecture-direction exit. The P513 state divergence and
  unexecuted accuracy gate keep its accuracy and production exits open; no
  weaker numerical contract may be used to close them.
- **WP-V2-C1-v1 — coupled true-large-M NVFP4 projections — implemented,
  rejected, default unchanged:** `da2b9f6` implements the M128N256K64,
  three-stage skeleton for both Gate+Up and Down. It returned through the real
  P40K API path but regressed pure-Prefill latency by 25.64%. The matched
  profile assigns 99.30% of the regression to the new NVFP4 kernels. It is not
  accuracy-promoted and must not receive further parameter scans.
- **WP-V2-C1-G2 / WP-V2-C1-D2 v1 — shape-specific NVFP4 redesign —
  implemented, rejected, default unchanged:** the explicit route implements
  merged Gate+Up (`K=5120,N=34816`) as an M128 paired Gate64+Up64 K64 raster
  and Down (`K=17408,N=5120`) as an M128N128K64 N-major raster. It preserves
  the authenticated packed-weight/scale identities, FP32 accumulation, and
  existing BF16 boundaries while fusing SiLU and residual. The resource gate
  reaches 127/126 registers per thread, 41,984 bytes dynamic shared memory,
  and static two-CTA/SM admission. The clean P40K route hit G2/D2 320+320
  times but regressed pure Prefill by 5.560464%; bounded NSys shows their role
  scope 7.040160 s slower than retained Marlin main plus SiLU. Meeting the
  resource envelope did not create persistent cross-CTA B/scale reuse, so this
  one-raster-CTA version is closed and must not receive parameter scans.
- **WP-V2-C1-v2 — persistent NVFP4 ownership reset — active design, not
  implemented:** freeze one global Gate/Up and Down dataflow that retains or
  reuses packed B, scales, and decoded fragments across multiple M rows or
  output tiles rather than exiting after one raster tile. Derive distinct
  ownership for the asymmetric Gate/Up and Down shapes from the proven
  Marlin/vLLM/Triton feed structures. Its dispatch design must reserve an exact
  M5424/general-tail extension, but the first implementation slice admits
  M8192/M7712 and returns directly to P40K. Static occupancy is only an
  admission fact; the design must quantify the useful reuse and pipeline
  lifetime that reaches the API budget. M5424 code and P60 execution follow
  only after that P40K composition is competitive and accuracy-admissible.
- **WP-V2-C2 — shape-specific FP8 projections:** after the coupled NVFP4
  package returns to P40K, build separate QKV, Z, and O tactics instead of
  inheriting one universal tile. Humming, Triton, vLLM and cuBLASLt may inform
  the dataflow, but cuBLASLt remains reference-only and never enters production
  dispatch.
- **WP-V2-B — prompt-wide recurrent and BF16 path:** submit one panel's C64
  hierarchy as one GDN work graph, expose chunk-local KKT/WY work in parallel,
  serialize only the mathematical boundary-state dependency, and replace
  recursive BF16 M16 A/B dispatch with panel-wide exact tactics. FLA and Mamba
  selective-scan mechanisms are design references; copied code or changed
  state precision is outside scope.
- **Composition deadline:** G2/D2 v1 met its API return point and is closed.
  WP-V2-C1-v2 must freeze its whole-dataflow/shape contract before local timing,
  implement the mutually dependent ownership and feed changes as one package,
  and return to one unprofiled clean-host P40K API direction against retained
  `c45b7c5` as soon as both roles execute end to end. A positive result earns
  correctness repetition and matched NSys/real-weight NCU; a negative result
  earns at most one bounded causal profile before closure or another material
  architecture reset. Do not let low-yield scans displace this return point.
- Only a competitive, accuracy-admissible P40K result unlocks P60K and
  approximately-130K execution, followed by complete capacity/resource and
  architecture-witness qualification. P60's balanced geometry is
  `6x8192 + 2x5424`; the rejected G2/D2 v1 lacks M5424, so P60 was not run and
  no P60 performance conclusion exists.

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
