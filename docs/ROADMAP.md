---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-09-28
  supersedes: [docs/ROADMAP_LEGACY.md]
  superseded_by: []
  ssot_for: active unfinished delivery slices and their ordering
  review_trigger: delivery dependency, active phase, or milestone-exit change
---

# Qwen3x-Orin active delivery roadmap

This page owns unfinished work and dependency order. [Current Status](CURRENT_STATUS.md)
owns delivered behavior and metrics; [SDD](SDD.md) owns architecture. The
[Constitution](ENGINEERING_CONSTITUTION.md) retains all owner-set targets.

## 2026-09-28 MTP — active, configured drafts 2 and 3

The owner's new authorization opens `WP-MTP-20260928` and
`AC-MTP-GREEDY-v1`, governed by [MTP Admission](MTP_ADMISSION.md).
Non-MTP optimization packages remain closed and their targets remain intact.
The first host milestone delivers the exact BF16 weight catalog and a bounded
transaction controller, with scalar-state oracle, rejection, EOS, cancellation,
capacity and failure tests. It does not yet execute a GPU draft or an API
request. The [milestone record](metadata/qwen36-27b-mtp-foundation-2026-09-28.json)
identifies that scope.

Remaining work, in order: authenticated native draft execution and prompt
hidden/KV initialization; complete device prefix-state restoration and a
scalar correctness oracle; then weight-reusing multi-row verification composed
with the real API. Return to P65 sanity and P8192/P40000 O256 for both configured
lengths immediately after those dependencies compose. Report Prefill, TTFT,
committed Decode, acceptance, reconciliation cost and memory together. No
length sweep or production switch is selected by host-only results. The three
integration stages and negative-direction stop are bounded in the subsystem
contract; no additional qualification-only campaign is opened.

## 2026-09-28 publication fusion — closed without promotion

`WP-PUBLICATION-FUSION-20260928` implemented both selected epilogues in
`AC-PREFILL-PUBLICATION-FUSION-v1`: Up/SiLU and FP8 O/residual, preserving
the GEMM mainloops and intermediate BF16 boundaries. Resource admission and
32 real-weight numerical cases passed, but the composed API direction was
mixed: P8192 Prefill improved 0.56%, while P40000 took 0.95% longer; Decode
was effectively unchanged. This single-process comparison does not establish
a statistically qualified regression or a promotable gain. Under the
predeclared negative-composition stop, the version is closed. The
[direction record](analysis/publication-fusion-direction-2026-09-28/README.md)
retains the reproducible patch, outputs, hashes and scope. Candidate integration
was removed, production is unchanged, and no qualification-only campaign or
tile scan remains active. FP8 preparation reuse remains a separate unimplemented
opportunity; this closed package reopened neither Down nor MTP. The later
owner-authorized MTP package above is separate.

## 2026-09-28 alternative-path source assessment — complete

The owner's follow-up to examine other paths is recorded in the
[alternative assessment](analysis/alternative-paths-2026-09-28/README.md).
It recommends preserving current GEMM mainloops while fusing Up/SiLU and
FP8 O/residual publication, with exact intermediate BF16 rounding; FP8
per-phase decoded-weight reuse is the next separate opportunity. It does not
claim measured gains, activate implementation, reopen Down, or alter production.
The report defines the bounded composition proposal for a subsequent package.
Decode's approximately 10 ms/step interim gap remains open; existing KV sharing
and rejected PV variants must not be presented as new opportunities.

## 2026-09-28 bounded non-MTP architecture work — closed

The owner-authorized `WP-NON-MTP-ARCHITECTURE-20260928` assessment and its
`WP-DOWN-ORDERED-20260928` implementation are complete. The
[assessment](analysis/non-mtp-architecture-assessment-2026-09-28/README.md)
selected Down's 17.56-second profiled Prefill budget. The
[implementation record](analysis/ordered-down-direction-2026-09-28/README.md)
closes both permitted dataflows: first external FP32 partials, then a corrected
block-local two-partition merge. Both preserved the bounded numerical panel
and compared API outputs, but regressed real 8K and 40K Prefill. Neither is
retained for production or future composition.

Candidate runner/build/profile branches have been removed; frozen reproducible
patches and all failures remain in evidence. The default 0.8.1 artifact,
numerical contract, capacity and paired performance baseline are unchanged.
No third variant, local tuning, extra profile or qualification-only campaign
is authorized by this closed package. It grants no MTP execution authority. A materially new
architecture requires a new bounded product-connected package; the deferred
product targets below are neither lowered nor claimed achieved.

## 2026-09-28 product audit — performance optimization paused

The owner paused performance optimization for this now-completed product
audit. The subsequent bounded architecture package above has also closed;
the earlier milestone list does not authorize unrelated tuning. The 8.55 interim
and long-term targets remain recorded, with no waiver or achievement claim.

The [product audit](analysis/product-readiness-audit-2026-09-28/README.md)
identifies Unicode output-cap handling, HTTP control-plane availability,
zero-test release validation, error-to-readiness propagation, reproducible
qualification tools, install notices and bounded sustained-service coverage.
The owner has authorized one delivery batch, `WP-SERVICE-INDUSTRIALIZATION-20260928`.
Its originating constraints are correct text serialization, available control
plane, fail-closed runtime health, reproducible validation and deployable
packaging. These select bounded UTF-8 carry, staged incomplete connections,
fatal-error health latching, production host tests, a versioned API driver and
installed notices/supervision templates. Value returns at the same installed
0.8.1 service through the unified protocol in
[EVALSCOPE_EVALUATION](EVALSCOPE_EVALUATION.md#service-industrialization-validation).
The batch is complete: six host/package CTests, 36 Python tests, multilingual/
slow-client/cancel/capacity checks, the pinned 98-question panel, two hours /
1,708 bounded reuse requests, fresh-process recovery and complete route/cleanup
checks passed. The [delivery record](metadata/qwen36-27b-service-industrialization-2026-09-28.json)
closes this package; no further qualification-only loop is active. It does not
reopen kernel optimization or change model arithmetic. The one-second reuse
cadence and 60-second capacity spacing do not qualify continuous maximum-load
saturation; the earlier thermal stop remains recorded as a deployment limit.

## 2026-09-27 product convergence — delivered scope and retained gaps

The owner's later numerical-repair and independent-baseline direction closes
`WP-PREFILL-REFERENCE-REPAIR-20260928` and the bounded service integration of
`AC-WHOLE-CORE-SERVICE-20260927`. Corrected whole-core Prefill plus ordered v7
Decode is installed and selected by the ordinary release preset. No further
Prefill numerical variant or qualification-only loop is open. The
[production record](metadata/qwen36-27b-whole-core-service-production-2026-09-28.json)
is the exact completion boundary.

The reference decision is explicit: checkpoint-faithful independent FP32 GDN
and same-input FP64 Decode checks assess the reference itself. Legacy remains
a regression comparator rather than universal truth. Complete prompt-boundary
state/logits, public capability and real installed API behavior qualify the
corrected service. Do not revive the invalid error denominator or compare
free-running states after token histories diverge.

A retained performance gap is long-context Decode: current correct
40K service is below the owner's approximately 8.55 token/s interim target.
Retain that target and report Prefill and Decode together. The four-chain PV
arithmetic variant and 64-CTA spatial-ownership variant are both closed for
slower real-payload performance. They are not pending tests and must not be
reopened as another local parameter scan.

Any next Decode optimization needs a new whole-runner bottleneck observation,
a named accuracy-preserving architecture change, a downward latency budget,
and a bounded return to the actual P40000/O256 production API. Preserve the
selected Prefill contract, BF16 KV, no-MTP scope and independent reference
checks. A component improvement without a positive composed API result does
not select production. A negative bounded composition closes its own version.

## Deferred product milestones

1. Resolve the remaining 40K Decode gap without numerical regression; retain
   the long-term 10 token/s target separately from the interim 8.55 target.
2. Expand capacity to the declared 60K and approximately 130K workloads through
   a new state/memory plan and complete real API witness. Current 44,095-state
   service qualification does not extend by extrapolation.
3. Resolve the locked 40K–60K 2-second and approximately 130K 4-second TTFT
   targets, starting from matched request-bound vLLM evidence and a complete
   architecture/dataflow budget. Do not infer impossibility from the present
   implementation or a dense-BF16 proxy.
4. Complete full product release qualification across those capacities/SLOs,
   longer-running stability and the required broader capability coverage.
   Preserve exact installed artifact, numerical, route, resource, dependency,
   package and public API attestation.
5. Basic service-supervision templates are delivered. Add further startup/AOT
   or deployment improvements only when selected by an
   actual deployment constraint. Additional models, batching and media
   remain later separately authorized product scopes. MTP is now separately
   authorized in the active package above.

These milestones do not authorize an open-ended experiment campaign. Each
new package must identify the product constraint, finite-precision/state
contract, one bounded implementation/composition point and real API exit.

## 2026-09-09 bounded engineering window

This window is closed and has no current execution authority. Its accepted
terminal-prefix and Graph ownership work, rejected score-feed and other
lineages remain documented in immutable metadata and Git history. The
[legacy roadmap](ROADMAP_LEGACY.md) is historical only. Old inbound links to
this anchor do not reactivate a package or make its profile the default.

## Evidence and stop discipline

Use the smallest safe product-connected check for a reversible iteration.
Freeze independent-process mirrored comparisons for architecture selection
and installed-artifact promotion. Stop collecting evidence once the declared
engineering question is answered. All performance work uses real prompts and
weights, clean GPU ownership, recorded preflight/cache preparation and the
[performance policy](REAL_MODEL_PERFORMANCE_POLICY.md). Preserve raw rejected
runs and source/binary identity; do not rewrite history as a pass.
