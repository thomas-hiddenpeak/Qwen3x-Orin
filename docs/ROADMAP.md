---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-10-10
  supersedes: [docs/ROADMAP_LEGACY.md]
  superseded_by: []
  ssot_for: active unfinished delivery slices and their ordering
  review_trigger: delivery dependency, active phase, or milestone-exit change
---

# Qwen3x-Orin active delivery roadmap

This page owns unfinished work and dependency order. [Current Status](CURRENT_STATUS.md)
owns delivered behavior and metrics; [SDD](SDD.md) owns architecture. The
[Constitution](ENGINEERING_CONSTITUTION.md) retains all owner-set targets.

## 2026-10-10 MTP asynchronous packed feed — closed negative

`WP-MTP-ASYNC-PACKED-FEED-20261010` / `AC-MTP-GREEDY-v11` addresses the
same 1.5x–3x objective and 18.45/21.62-second full Decode budgets. The measured
16.29-second verifier projection cost selects overlapping packed-weight movement
with the existing exact register-local M-row consumption. Unlike rejected
expanded shared-weight tiles or token-ownership variants, a three-buffer
cp.async pipeline retains FP8/NVFP4 bytes unchanged, prefetches two future K
blocks, and keeps the original four FMA chains and output ownership. It
transfers the proven Marlin/production Attention asynchronous producer-consumer
lifetime to this private SIMT verifier without importing MMA reassociation.
No hardware-stall cause is asserted from the unavailable counters.

One fixed composition, including the admitted ordered draft GQA dependency,
returns through full P65 state/logit/transaction checks to P65/8K/40K d2/d3 API.
At most one correctness repair is permitted; negative API direction closes
this version without buffer/grid/tile scans. Shared packed storage is 12,288
bytes per block, no new arena or runtime allocation. Artifacts stay under
`.q3x-work/mtp-async-packed-20261010/`. The 8-GiB reserve and production
non-MTP path remain unchanged.

The [API rejection](metadata/qwen36-27b-mtp-async-packed-rejection-2026-10-10.json)
closes the composition: complete P65 numerical checks pass, but P65 Decode is
10.308 token/s versus retained v7 10.717. New runtime paths are removed; no
8K/40K/d3 qualification or pipeline-parameter scan follows. Retained v7 remains
the development incumbent. The next implementation must address the complete
verification budget with a different execution architecture, preserving exact
publication and the 1.5x–3x target; this closed package authorizes no variant.

## 2026-10-10 MTP independent token CTAs — closed negative

`WP-MTP-INDEPENDENT-CTAS-20261010` / `AC-MTP-GREEDY-v10` retains the
1.5x–3x objective and 18.45/21.62-second full 8K/40K Decode budgets. The
16.29-second projection observation selects independent token CTAs with
M-fast output-tile scheduling: adjacent blocks consume the same packed weight
span through shared L2, while each thread retains only one token's original
four FP32 chains. No token-group barrier or shuffle broadcast is introduced.
This is distinct from the rejected v3 CTA-row implementation, which forces
M*256 FP8 or M*128 NVFP4 threads into each block with shared codebooks and
multi-token reduction storage. Here the unchanged 256/128-thread blocks can
reside and retire independently, trading duplicated cached reads/conversion
for fewer registers and independently schedulable consumers. L2 reuse is a
hypothesis, not a measured counter claim. Packed weight layout and decoder
remain unchanged; there is no grid/tile sweep. Ordered draft Attention composes
at this immediate API return with dependency-only authority.

One fixed composition plus at most one correctness repair must pass complete
P65 scalar prefix state/logits and transactions, then return to P65/8K/40K d2/d3
API in fail-fast order. Negative direction closes this version. No new arena,
resource-margin reduction, approximate output or production change is allowed.
Artifacts stay under `.q3x-work/mtp-independent-ctas-20261010/`.
The [API rejection](metadata/qwen36-27b-mtp-independent-ctas-rejection-2026-10-10.json)
records P65 Decode 7.509 token/s versus retained v7 10.717. All numerical and
recovery checks pass, but the composition is removed. No long-context or d3
qualification follows. Independent-CTA, cooperative-CTA and per-thread row
ownership have now all been compared; another ownership/cache/grid variation
is not the next package. The overall 1.5x–3x goal remains active and unmet.
A successor requires a different complete execution/data-representation
architecture and a traceable exact-publication proof before implementation.


## 2026-10-10 MTP certified publication — feasibility closed

`WP-MTP-CERTIFIED-PUBLICATION-20261010` addresses the same 1.5x–3x API goal
and measured 16.29-second projection cost. Repeated packed-feed variants are
closed. The alternative is a fast reduction followed by a rigorous rounding
certificate and exact scalar repair of uncertified outputs. A certificate must
prove the identical BF16 publication, including scale and reduction error;
statistical error or matching argmax is insufficient. Recurrent state and all
observable target outputs remain exact. This does not authorize approximate
production arithmetic or a weaker numerical baseline.

Before implementing a new executor, one whole-core real-prompt M3 capture at
P65 samples 32 output rows per projection in layers 0/3/20/23/40/43/60/63.
The numerical-only harness records canonical weights/scales, real activations,
exact BF16 outputs and identities. A host error-bound audit measures the
optimistic certification and required repair fractions, both element-wise and
for shared M-row output ownership. This is a bounded feasibility prerequisite,
not a local performance screen. It may reject an uneconomic certificate but
cannot select a runner or claim speedup. At most one capture and one audit
precede either closure or a fully specified executor/repair composition with
an immediate d2/d3 real API return. No quantized-decoder or launch sweep opens.
Artifacts remain in `.q3x-work/mtp-certified-publication-20261010/`.
The [bounded audit](metadata/qwen36-27b-mtp-certified-publication-feasibility-2026-10-10.json)
finds that a cheap norm certificate would repair at least one row in 46.4%
of FP8, 59.2% of Gate/Up and 97.7% of Down sampled output groups. Even an
optimistic exact center cannot remove most Down repairs with this bound.
The coarse-bound whole-verifier composition is closed before GPU executor
implementation; no tighter-bound or parameter sweep follows under this package.
The capture seam is frozen numerical-only evidence and was removed after this
completed gate; it was never part of serving.


## 2026-10-10 MTP persistent operands — closed negative

`WP-MTP-PERSISTENT-OPERANDS-20261010` / `AC-MTP-GREEDY-v9` uses the
[complete 40K attribution](metadata/qwen36-27b-mtp-bottleneck-reset-2026-10-10.json):
16.29 GPU seconds in verifier projections, 5.40 in QK/PV, and 2.30 in combined
target/draft lm-head after the first verifier. No valid NCU hardware-stall
counters were obtained, and that collection is closed. The selected composition
retains packed global weights and register-local M reuse, but makes activation
and lookup storage persist across output groups. It also batches target lm-head
weights across all verified rows and reuses ordered production draft Attention.
This follows the production lm-head's proven persistent activation lifetime;
it is not another expanded-weight tile or register-decoder variant. Main-model
FMA trees/state/logits remain exact. One fixed composition plus at most one
correctness repair returns to complete prefix admission and P65/8K/40K d2/d3.
The 1.5x–3x objective and 18.45/21.62-second full Decode budgets remain fixed;
negative API direction closes the version without tile, grid or draft scans.

The [API rejection](metadata/qwen36-27b-mtp-persistent-operands-rejection-2026-10-10.json)
closes this version: complete scalar-prefix state/logit checks pass, but P65
Decode falls from retained v7's 10.717 to 9.536 token/s. The new runtime paths
are removed; long-context/d3 qualification and grid/tile scans do not follow.
The overall MTP goal remains active and unmet. A successor must change the
complete verifier execution architecture rather than reopen another packed
weight-decoder, output-group or persistence variation. It must first reconcile
its numerical contract and expected whole-round budget against the measured
16.29-second projection cost, then implement one composition and return to the
same real API. No new kernel variant is selected by this closure.

## 2026-10-09 MTP bottleneck reset — complete

`WP-MTP-BOTTLENECK-RESET-20261009` restores the retained v7 composition after
full-K residency regressed the real P65 API. No losing projection or draft
route remains selected. One bounded same-ELF P40000/O256 d2 Nsight attempt
will distinguish projection execution, long-context Attention and controller
cost on the complete matched real workload. The existing fresh v7 API record
is its timing authority; profile timings cannot replace it. Startup reserve
and clean ownership stay mandatory; a resource failure stops that attempt.
The completed trace attributes 16.29 seconds to verifier projections and
5.40 seconds to verifier QK/PV (softmax separate). The bounded NCU collection produced no hardware counters: filtering and
permission failures were followed by replay-backup memory exhaustion under
root. It is closed without reducing the reserve or claiming a measured stall
cause. The subsequent persistent composition above also returned negative.
The 1.5x–3x goal remains active.

## 2026-10-09 MTP full-K operand residency — closed negative

`WP-MTP-FULLK-RESIDENCY-20261009` / `AC-MTP-GREEDY-v8` follows the
packed-conversion composition's negative P65/8K direction. Repeated scalar
weight conversion changes did not close the verifier budget. This fixed
architecture instead decodes a complete FP8 output group into CTA-owned BF16
once, publishes it once, then lets independent token groups consume the whole
K span without an inner producer barrier. Unlike rejected shared-FP32 tiles,
there is no per-tile synchronization or expanded FP32 operand traffic. BF16
holds every FP8 code exactly. Output ownership is four channels for K5120 and
two for K6144, solely to fit the static shared-memory limit. All original
four-chain, warp/eight-warp reductions and BF16 publication remain exact.
NVFP4 stays incumbent. Production ordered draft Attention remains dependency-only
until this composition's API result. One implementation and at most one
correctness repair return to full-prefix admission and P65/8K/40K d2/d3;
negative direction closes it without a launch/shape sweep. The same 1.5x–3x
goal and 15.7/17.9-second verifier budgets select this work.

## 2026-10-09 MTP packed FP8 conversion — negative direction

`WP-MTP-PACKED-FP8-20261009` / `AC-MTP-GREEDY-v7` restores the v7
vector/output-quad and interleaved-query implementation after v9 regressed all
three d2 API buckets. Shared-V and output-pair ownership are archived, with
no d3 qualification-only continuation. The next fixed composition transfers
the pinned Marlin pairwise FP8-to-FP16 conversion to exact FP32 verifier
operands and reuses production ordered GQA for draft S512..44095. It changes
neither target FMA trees nor draft outputs. Exhaustive 256-code/all-pair device
checks precede complete real-prefix state/logit admission, then immediate
P65/8K/40K d2/d3 API return. One composition plus one correctness repair is
the bound; no parameter scan or numerical waiver is opened. The unchanged
1.5x–3x goal and full-round budgets remain the selection constraint.

## 2026-10-09 MTP accumulator ownership — closed negative

`WP-MTP-ACCUMULATOR-OWNERSHIP-20261009` / `AC-MTP-GREEDY-v6` follows the
negative P65 API result of warp-broadcast verification. The v8 projection is
removed; its exact shared-V implementation is a dependency only, bounded to
this immediate composition. Retain register-local reuse across all M rows,
but partition the existing four-channel packed sidecar into two aligned
output pairs for M3/M4. This reduces per-thread accumulator liveness without
multiplying decode/broadcast instructions across token subgroups. M2 retains
the existing four-output route. This is one fixed ownership transformation,
not a tile/launch sweep. It composes with the exact shared-V kernel and admitted
draft MMA before the next P65/8K/40K d2/d3 API return. The 15.7/17.9-second
verifier budgets and 1.5x–3x target remain unchanged. One implementation plus
at most one correctness repair bounds this package; no third weight-decoder
variant or repeated qualification of a losing policy is opened.

## 2026-10-09 MTP cooperative verifier — projection rejected

`WP-MTP-COOPERATIVE-VERIFY-20261009` / `AC-MTP-GREEDY-v5` follows the
completed d2/d3 API return of the draft matrix path.
[MTP Admission](MTP_ADMISSION.md#cooperative-verifier-composition-v5) defines
warp-local packed-weight broadcast and actual shared-V query consumption.
Exactly these two dependencies compose before the next API direction screen;
no separate local-timing campaign or parameter scan is opened. The unchanged
1.5x–3x product constraint selects a 15.7/17.9-second 8K/40K verifier budget.
One composition plus at most one correctness repair is the bound; complete
scalar state/logit identity, causal masks and the reserve remain mandatory.

## 2026-10-09 MTP draft matrix execution — API return complete

`WP-MTP-DRAFT-MMA-20261009` / `AC-MTP-GREEDY-v4` addresses the same
1.5x–3x owner target. The v4 verifier is restored; numerically admitted causal
query scheduling and post-Prefill workspace borrowing compose with one native
BF16 draft matrix executor. The measured 2.83/13.80-second 8K/40K draft
initialization and 3.41/4.54-second proposal costs select this dependency.
The complete Decode budgets remain 18.45/21.62 seconds, so a draft-only win
cannot close the verifier gap. The bounded composition is one fixed CUTLASS
M1..32 executor, independent draft-oracle admission, then the existing
P65/8K/40K d2/d3 API panel. No kernel-parameter scan is opened. Target state
and logits remain bitwise scalar-equivalent; only the private draft reduction
gets a separate numerical identity, subject to the existing independent
FP64 0.02 relative-L2 bound and exact within-route cache replay.

## 2026-10-09 MTP shared verification — closed negative

`WP-MTP-SHARED-VERIFY-20261009` / `AC-MTP-GREEDY-v3` implements the next
bounded composition described in [MTP Admission](MTP_ADMISSION.md#shared-verification-composition-v3).
The originating owner constraint remains 1.5x–3x committed Decode versus the
same-request ordinary production service. The previous 8K/40K full-Decode
budgets of 18.43/21.61 seconds select shared decoded projection operands and
causal multi-query Attention, with unchanged exact state/logit semantics.
One initial composition and at most one causally justified repair must return
to P65/8K/40K d2/d3 API; no launch-parameter or draft-length sweep is opened.
Both weight-delivery implementations regressed API Decode and are removed.
No third projection repair is permitted by this package. This is isolated
development, not a production switch. Prefill initialization
remains an explicitly measured unresolved cost; this package first returns the
new verifier to the API before selecting its next prompt-side dependency.

## 2026-09-28 MTP — active, configured drafts 2 and 3

`WP-MTP-EFFICIENCY-20260928` / `AC-MTP-GREEDY-v2` has completed its bounded
implementation and API return, recorded in the
[efficiency direction](metadata/qwen36-27b-mtp-efficiency-direction-2026-09-28.json).
Vector-feed verification, M32 initialization and cache-only reconciliation
remain in the isolated v4 admission; the single CTA-row repair is rejected and
removed. Both drafts pass the declared numerical/API panel, but neither
provides the owner's 1.5x–3x Decode objective across contexts. No production
switch, draft-length sweep or qualification-only campaign follows this result.

The MTP efficiency goal remains active. The downward budget now requires
8K d2's entire Decode to fit about 18.43 seconds, versus 19.63 seconds in
verification alone; at 40K the full d2 budget is about 21.61 seconds versus
26.91 seconds in verification alone. This selects a materially new verifier
operand/Attention dataflow and prompt/draft work elimination, not another
launch-parameter variant of the rejected CTA design. Any successor must name
its exact arithmetic/state/lifetime ledger, bounded composition point and
return to the same P65/8K/40K d2/d3 API panel, reporting Prefill, TTFT, committed
Decode and external total time. The numerical contract and 8-GiB reserve remain
hard constraints; the production service remains the speedup denominator.

The earlier owner authorization opened `WP-MTP-20260928` and
`AC-MTP-GREEDY-v1`, governed by [MTP Admission](MTP_ADMISSION.md).
Non-MTP optimization packages remain closed and their targets remain intact.
The host foundation and the [native scalar device milestone](metadata/qwen36-27b-mtp-device-2026-09-28.json)
are delivered. Authenticated draft execution, shifted scalar hidden/KV
initialization, complete recurrent prefix restoration and target-conditioned
draft reconciliation pass the bounded real-model correctness harness. That scalar
milestone executed the main model serially and had no API or acceleration claim.

The whole-core prompt handoff now captures all final normalized hidden rows
and initializes shifted draft KV in the non-installable engine harness. Full
logits are included in prefix restoration. The
[handoff record](metadata/qwen36-27b-mtp-whole-core-2026-09-28.json) bounds these
checks. The [batched initialization milestone](metadata/qwen36-27b-mtp-batch-prefill-2026-09-28.json)
now closes cache-only FC/K/V batching and bounded initialization cancellation
on the real P65 state panel. The [multi-row verifier milestone](metadata/qwen36-27b-mtp-multirow-2026-09-28.json)
now closes exact FP8/NVFP4 weight reuse and complete immutable prefix snapshots.
P61/P65/P509 pass every M2/M3/M4 prefix plus rejection, cancellation and recovery
against full scalar state/logits, including the S64 and S512 Attention switches.
The generic small-M reduction mismatch is fixed; no additional local verifier
scan is pending.

The [service composition](metadata/qwen36-27b-mtp-service-direction-2026-09-28.json)
now completes that API return: P65/O16 and P8192/P40000 O256 run for both draft
lengths, with matched responses and bounded lifecycle/route checks. Both
policies worsen Prefill and Decode relative to the ordinary service. This
closes the performance version of `AC-MTP-GREEDY-v1` without selection; no
additional v1 qualification or draft-length sweep is pending. The isolated
controller, full-capacity admission and truthful physical witness are retained
as development infrastructure, not enabled in production.

The efficiency package above supersedes v1's next-step proposal. Its one
CTA-row operand repair is rejected after a valid P65/P8192 slowdown; it is not
an outstanding tuning branch. The final composition uses stage-one vector
projection feed with M32 initialization and cache-only reconciliation. Both
configured draft lengths have returned to the complete API direction panel,
closing this bounded package. Numerical admission does not authorize production or a
50% speedup claim. A remaining gap must select a materially new verifier or
Attention/draft dataflow with a downward whole-round budget; repeating these
projection variants, qualifying a losing policy or scanning draft lengths is
not the next task.

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
