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

## 2026-09-28 product audit — performance optimization paused

The owner has explicitly paused performance optimization and requested a
product audit. Do not start Decode tuning, Prefill architecture experiments
or performance-only reruns under the earlier milestone list. The 8.55 interim
and long-term targets remain recorded, with no waiver or achievement claim.

The [product audit](analysis/product-readiness-audit-2026-09-28/README.md)
identifies Unicode output-cap handling, HTTP control-plane availability,
zero-test release validation, error-to-readiness propagation, reproducible
qualification tools, install notices and bounded sustained-service coverage.
Its recommended repair order is API correctness/availability, repeatable
production tests, then recovery/packaging and sustained-service checks. These
are findings and proposed closures; the audit itself makes no runtime repair.

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
5. Add startup/AOT/service-supervision improvements only when selected by an
   actual deployment constraint. Additional models, batching, MTP and media
   remain later separately authorized product scopes.

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
