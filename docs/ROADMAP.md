---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-10-20
  supersedes: [docs/ROADMAP_LEGACY.md]
  superseded_by: []
  ssot_for: active unfinished delivery slices and their ordering
  review_trigger: delivery dependency, active phase, or milestone-exit change
---

# Qwen3x-Orin active delivery roadmap

This page owns unfinished work and dependency order. [Current Status](CURRENT_STATUS.md)
owns delivered behavior and metrics; [SDD](SDD.md) owns architecture. The
[Constitution](ENGINEERING_CONSTITUTION.md) retains all owner-set targets.

## 2026-10-20 complete GDN owner composition — selected; installation next

The [v127 design](metadata/qwen36-27b-mtp-gdn-owner-design-2026-10-20.json)
selects complete GDN prefix ownership coupled with the previously admitted
startup FP8 input owner. This is a materially different composition from the
closed first-prefix-only assessment and fixed-VA remapping. First admit a
source-private indirect GDN input with unchanged arithmetic and Graph replay;
then compose all typed-state, scalar-tail, observer, reset and failure consumers
before the immediate short cap 2 API return. No production or speed claim
follows from the prerequisite. The design bounds it to two further milestones.

The [indirect-input prerequisite](metadata/qwen36-27b-mtp-gdn-owner-prerequisite-2026-10-20.json)
now passes twelve complete state/output comparisons, nine changing-pointer
Graph replays, four exact-alias cases, invalid zero-enqueue and guard checks,
and bounded memcheck with zero errors. This closed milestone 1; the complete
integration and API returns below close its original composition deadline.
No further isolated measurement is selected.

The v127 integration is implemented behind a default-off build option;
retained candidate selection remains v122. The complete selected-storage
exception is specified in [Request State](REQUEST_STATE.md#isolated-v127-selected-gdn-ownership)
and the runner/system contracts. The [native admission](metadata/qwen36-27b-mtp-gdn-owner-integration-2026-10-20.json)
now passes complete P65 state/logit/draft-KV and EOS/cancel/failure checks.
Milestones 2 and 3 are complete: the
[cap 2 API return](metadata/qwen36-27b-mtp-gdn-owner-cap2-direction-2026-10-20.json)
passes P65/O16 at 14.398 Decode token/s and both 8K/40K regression gates with
exact output/work. The [cap 3 API return](metadata/qwen36-27b-mtp-gdn-owner-cap3-direction-2026-10-20.json)
also passes all three buckets. Retain the default-off composition for affected
ownership lifecycle/resource checks and independent performance selection.
The [lifecycle protocol](metadata/qwen36-27b-mtp-gdn-owner-lifecycle-design-2026-10-20.json)
has a [recorded process-boundary repair](metadata/qwen36-27b-mtp-gdn-owner-lifecycle-process-repair-2026-10-20.json):
same-process reconstruction attempts hit the unchanged reserve guard after
successful CUDA release. Fresh-process fault checks preserve the reserve and
match the service restart boundary; same-process reconstruction is unqualified.
The [completed lifecycle panel](metadata/qwen36-27b-mtp-gdn-owner-lifecycle-complete-2026-10-20.json)
now passes seven native ownership cases and both-cap API cancellation/recovery,
with normal exit throughout. Stop lifecycle collection. The
[bounded independent selection](metadata/qwen36-27b-mtp-gdn-owner-repeat-2026-10-20.json)
now passes all six N-C2-C3-C3-C2-N processes without retries. Every cap/bucket
clears the worst-observed >=1.5x floor and the same-cap Prefill regression guard;
Current Status owns the joint metrics and their bounded workload scope.
Stop architecture measurement and select v127 for independent executable-only
packaging. Its RequestState ABI must not enter the ordinary 0.8.2 exported
libraries/headers. Next complete the exact installed artifact's route, API,
capacity, capability, resource/restart and performance attestation, then audit
the full goal. No new parameter scan, production switch or full-goal claim
follows from this architecture selection.

The [package design](metadata/qwen36-27b-mtp-package-design-2026-10-20.json)
activates fixed-cap executable-only build/install implementation. The
[host admission](metadata/qwen36-27b-mtp-package-host-2026-10-20.json) now passes
both builds, four CTests per cap and negative configuration/manifest checks.
Clean source `65e13db` installed artifacts now pass the
[two-cap API/lifecycle panel](metadata/qwen36-27b-mtp-package-api-2026-10-20.json).
Stop integration collection. Execute the [six-process installed performance gate](metadata/qwen36-27b-mtp-installed-repeat-design-2026-10-20.json),
then only on success the [eight-process remaining delivery panel](metadata/qwen36-27b-mtp-installed-qualification-design-2026-10-20.json):
both-cap public capability, full capacity, two-hour sustained reuse and fresh
restart. These are explicit completion gates, not a new optimization campaign.
Do not transfer the frozen candidate's timing or v122 sustained result to a new ELF.

## 2026-10-20 startup FP8 input composition — closed below API floor

The [startup ownership composition](metadata/qwen36-27b-mtp-startup-input-design-2026-10-20.json)
passes bounded complete numerical/state checks but closes at its
[short API gate](metadata/qwen36-27b-mtp-startup-input-rejection-2026-10-20.json).
Cap 2 P65/O16 observes 14.238 Decode token/s, below the declared 14.389 floor;
Prefill is 0.89223s / 72.85 token/s, TTFT 0.91098s and total 1.96430s.
Exact output and work match v122. The +0.70% single-process direction versus
v122 is insufficient for this composition's gate and is not independent
performance selection. The ownership repair is feasible, but no runtime
prerequisite is retained without a new bounded composition. All candidate
integration is removed and v122 source/binaries restored. Stop before long
contexts/cap 3 and do not scan layouts. Full goal remains open.

## 2026-10-20 same-thread Gate/Up composition — closed at short API floor

The [bounded API successor](metadata/qwen36-27b-mtp-gate-register-pair-api-design-2026-10-20.json)
now has a [complete short API rejection](metadata/qwen36-27b-mtp-gate-register-pair-api-rejection-2026-10-20.json).
The unchanged prototype passes actual four/three-CTA resource admission,
complete projection/Graph/synchronization checks and bounded real-model
scalar state/logit/draft-KV transactions. This corrects the planning inference
from the earlier 160-register heuristic without rewriting its frozen result.

Cap 2 P65/O16 Decode is 14.036 token/s, below the predeclared 14.389 floor
and the frozen v122 observation of 14.140. Prefill is 0.89175s / 72.89 token/s,
external TTFT 0.91061s and total 1.97909s. Output IDs, acceptance and physical
work remain exact. This is one-process direction, not a statistical regression
or a hardware-stall attribution. The short stop ends before long/cap 3 checks.

All integration and unused prerequisites are removed; v122 source and frozen
binaries are restored. No register/geometry/stage scan or qualification-only
continuation follows. The full goal and cap 2 short deficit remain open;
ordinary non-MTP production is unchanged.

## 2026-10-20 MTP sustained service — complete; short-cap2 performance next

The [completed two-cap sustained and restart record](metadata/qwen36-27b-mtp-sustained-complete-2026-10-20.json)
closes the [bounded protocol](metadata/qwen36-27b-mtp-sustained-design-2026-10-20.json)
on both frozen v122 binaries. Cap 2 completes 7,201 seconds/1,907 measured
requests; cap 3 completes 7,204 seconds/1,911. Every response preserves the
native response contract and full MTP token sequence, including cross-cap and
fresh-process recovery. All four processes exit normally with bounded resource
growth. Current Status owns paired metrics and reference limitations.

Stop sustained collection: capability, capacity and this reliability panel are
complete. The next implementation must close cap 2's short-context >=1.5x
phase budget using a materially different architecture and an immediate real
API return. Preserve the closed negative lineages; no parameter scan is
selected. Final independent performance selection and delivery/installation
qualification remain open. Neither this reliability pass nor cap 3's retained
performance closes the cap 2 gap or changes ordinary non-MTP production.

## 2026-10-20 MTP capacity — complete for both caps

The [capacity protocol](metadata/qwen36-27b-mtp-capacity-design-2026-10-20.json)
and [completed two-cap record](metadata/qwen36-27b-mtp-capacity-complete-2026-10-20.json)
close this bounded service-range check on the frozen retained v122 binaries.
Each cap completes P40000/O4096 and P44095/O1, rejects P44095/O2, then recovers
P65/O16, with normal process exit and the startup reserve intact. Complete
native-reference text/usage/finish contracts match. All 4,113 output token IDs
match between caps; historical native output IDs are absent, so this is not
claimed as direct native-ID equality. O1 correctly reports no Decode rate.

Stop capacity collection. This pass does not close sustained delivery, final
independent selection, installation or the cap 2 short-context performance gap.
The next implementation must address that phase budget under a materially
different architecture with an immediate real API return. Preserve rejected
combined decoder tables, input layouts, accumulation-chain splits and other
closed lineages; no parameter scan is selected. The full 50%–200% objective
remains active and ordinary production stays non-MTP.

## 2026-10-20 EOS frontier — composed and retained

The [bounded design](metadata/qwen36-27b-mtp-eos-frontier-design-2026-10-20.json)
is complete through its [native/initial API return](metadata/qwen36-27b-mtp-eos-frontier-initial-direction-2026-10-20.json)
and [remaining two-cap regression](metadata/qwen36-27b-mtp-eos-frontier-regression-2026-10-20.json).
Retain v122 in the cumulative non-installable MTP candidate. All six affected
EOS requests per cap preserve exact output and remove two target rows and one
private step. Representative noEOS, short/8K/40K, authentication, invalid
requests, cancellation/recovery and normal exit pass. Stop this collection;
no further EOS measurement or threshold scan is selected.

This is bounded development retention, not independent performance selection
or production promotion. Joint metrics and their limits belong to
[Current Status](CURRENT_STATUS.md). P65 never triggers EOS truncation, so cap2
still misses1.5x. The next implementation must address that phase budget with
a materially different architecture, preserving the closed lineages below.
Capacity is closed above; sustained delivery, final selection and installation remain open;
the full50%–200% goal is not complete and ordinary production stays nonMTP.

## 2026-10-20 sealed MTP capability — complete; EOS work audit next

The [bounded design](metadata/qwen36-27b-mtp-sealed-capability-design-2026-10-20.json)
and [completed two-cap panel](metadata/qwen36-27b-mtp-sealed-capability-2026-10-20.json)
close this capability check. Each cap passes 98 complete native-response and
external-answer comparisons with 79 correct answers; cross-cap output token
IDs are exact. Historical native records lack output token IDs, so the native
comparison is complete text/usage, not a direct token-ID oracle. Both fresh
processes exit normally. Stop this collection; no repetition is selected.

The panel exposes six stop-finished requests per cap with accepted EOS, two
rounds and six verified rows for only three committed Decode tokens. The next
bounded efficiency assessment must establish which EOS suffix work can be
removed while preserving mismatch fallback, complete target prefixes and
observer semantics. A proposed EOS must never stop the actual request before
target verification. No saving or P65 cap 2 deficit closure is claimed yet.
That deficit, capacity, sustained delivery and production selection remain open.
The reviewed FP8 input-layout mechanism remains closed by v19/v21 evidence.

## 2026-10-20 sealed MTP build — both caps composed

The [bounded design](metadata/qwen36-27b-mtp-sealed-service-design-2026-10-20.json)
now has a [complete cap 3 candidate/API return](metadata/qwen36-27b-mtp-sealed-service-2026-10-20.json).
The `orin-mtp-sealed-candidate` preset builds with `BUILD_TESTING=OFF`, a fixed
compiled cap and an independent unqualified profile. No engine/runner test
seams, MTP/fused test admissions, ambient draft selection or installation
surface is enabled. The ordinary non-MTP production preset remains unchanged.

The first API attempt exposed a hardcoded old receipt identity and was stopped.
The repair emits the selected service profile and rejects mismatched receipts
immediately. The repaired cap 3 binary passes all eight API response/token/work
comparisons, authentication, invalid requests, cancellation and recovery.
This composes the preceding service-access prerequisite. Current joint metrics
belong to [Current Status](CURRENT_STATUS.md); this one-process check does not
transfer earlier repeated performance qualification to the new ELF.

The [fixed-cap 2 build and API regression](metadata/qwen36-27b-mtp-sealed-cap2-2026-10-20.json)
is also complete: clean source 261bc9d, no test seams or cuBLAS dynamic dependency,
fixed-cap CLI and environment rejection, eight exact response/token/work checks,
authentication, invalid requests, cancellation/recovery and normal shutdown.
This closes the two-cap build composition; do not repeat this panel without a
new engineering question. Current joint metrics belong to Current Status.

Cap 2 short performance remains below the 1.5x goal. The next performance
implementation must select a materially different architecture with a downward
phase budget and immediate real API return, respecting the closed lineages
below. Repeated performance selection and delivery qualification remain open;
this build closure does not select installation or a production switch.

## 2026-10-20 service ownership boundary — validated delivery prerequisite

The [bounded design](metadata/qwen36-27b-mtp-service-boundary-design-2026-10-20.json)
and [completed API regression](metadata/qwen36-27b-mtp-service-boundary-2026-10-20.json)
separate source-private `ServiceAccess` from test-only runner/state/model access.
The engine retains ownership; HTTP borrows its service under serialized request
execution. The engine object compiles without test seams and the complete linked
service contains no `EngineAccess` symbols. Eight successful API requests preserve
exact responses, token IDs and physical work; authentication, invalid-request,
cancellation and recovery checks pass. Arithmetic and output policy are unchanged.

Only the engine object has test seams disabled; other admission flags remain.
That boundary-only artifact was not a complete `BUILD_TESTING=OFF` or
production artifact. Its prerequisite has now composed into the sealed cap 3
candidate above. Cap 2's short deficit remains open and production stays non-MTP.

## 2026-10-20 Gate/Up scale residency — closed negative API direction

The [startup-owned scale design](metadata/qwen36-27b-mtp-gate-scale-residency-design-2026-10-20.json)
first failed peak retained-free admission with its 16-bit owner. The one
[bounded memory repair](metadata/qwen36-27b-mtp-gate-scale-residency-repair-2026-10-20.json)
packed invariant-zero BF16 scale bits into 12-bit fields, preserving exact
reconstruction in 1,069,547,520 bytes. Both artifacts retain their evidence;
the first has no real-model numerical or API pass.

The [completed repaired direction](metadata/qwen36-27b-mtp-gate-scale-residency-rejection-2026-10-20.json)
passes complete scale/projection/Graph checks, native state/logit/KV transactions
and fault recovery. Short API output and work match v119, and actual startup
free memory is 9,178,890,240 bytes. Decode is 13.825 token/s versus v119's
14.104, failing the 1.5x floor. Prefill is 0.89275s (72.809 token/s), TTFT
0.91129s and total 1.99618s. Verification increases about 22.8ms; this does not
establish a specific cache, bank or instruction bottleneck.

The predeclared stop ends before long-context/cap 3 and lifecycle qualification.
All integration and unused prerequisites are removed; exact v119 source and
binaries are restored. No storage-width, scale, layout, register or profile
scan follows. Retained cap 3 results and the full objective remain unchanged.


## 2026-10-20 cap 2 architecture assessment — two directions closed

The [source and real-output assessment](metadata/qwen36-27b-mtp-cap2-architecture-assessment-2026-10-20.json)
rejects terminal-private-KV-only work as sufficient to close the short budget:
the entire measured reconciliation interval is smaller than the gap, and only
a subset is terminal private KV. It also rejects the pinned vLLM default
5-token prompt lookup for this short workload: no match exists at any of the
15 possible proposal positions. No GPU process or threshold scan is selected.
These are scoped feasibility conclusions, not speed measurements or hardware
bounds. The next implementation still requires a materially different dominant
projection or private-proposal computation and an immediate real API return.

## 2026-10-20 retained cap 3 repetition — complete

The [prospective protocol](metadata/qwen36-27b-mtp-output-tail-repeat-design-2026-10-20.json)
and [independent-process validation](metadata/qwen36-27b-mtp-output-tail-repeat-2026-10-20.json) are complete. Four fresh processes pass
exact outputs, physical work, resource/preflight and normal shutdown checks.
The slower cap 3 rate divided by the faster non-MTP rate exceeds 1.5 in all
three pinned buckets. Joint metrics are owned by [Current Status](CURRENT_STATUS.md).
Stop this collection; no retry or wider evidence campaign is selected.

Cap 2's short deficit remains the next implementation constraint. The rejected
balanced tail is not reopened. A materially different composition must name
its phase budget and return promptly to the API. This bounded cap 3 selection
is not full-goal completion or production-release authority.

## 2026-10-20 balanced cap 2 tail — closed at short API gate

The [bounded design](metadata/qwen36-27b-mtp-balanced-tail-design-2026-10-20.json)
replaced the four-output cap 2 tail's M3 plus native M1 with two M2 rounds.
The [completed direction](metadata/qwen36-27b-mtp-balanced-tail-rejection-2026-10-20.json)
passes host and native full-state/logit/KV checks and preserves all output IDs.
Short Decode is 14.126 token/s, below the declared 1.5x floor; Prefill is
0.89241s (72.836 token/s), TTFT 0.91139s and total 1.97309s. Seven rounds and
19 verified rows remain; eliminating the scalar tail alone is insufficient.
This is a single-request direction, not a repeated performance comparison.

The early stop closes before long-context and lifecycle qualification. Remove
the cap 2 rule, unused proposal-limit interface and new tests; restore v119
source and frozen binaries. No horizon, shape or profile scan follows.
The retained cap 3 direction and remaining full objective are unchanged.

## 2026-10-20 output-tail composition — positive API direction

[WP-MTP-OUTPUT-TAIL-20261020](metadata/qwen36-27b-mtp-output-tail-design-2026-10-20.json)
selects one fixed request-horizon rule. Configured cap 2 stays unchanged; cap 3
uses budget 2 except when four output tokens remain, when budget 3 may complete
with its bonus and remove a subsequent full scalar model pass. Existing 0.4
confidence stopping, state commitment and all arithmetic remain unchanged.
This removes work if accepted; it does not optimize the singleton's submission.
A rejected extension may cost more. No prompt/token-specific rule or threshold
scan is permitted. At most one requested 3 round occurs per request.

The [completed direction](metadata/qwen36-27b-mtp-output-tail-direction-2026-10-20.json)
passes mixed-budget host/native full-state admission and both configured-cap
API/lifecycle panels. Cap3 crosses 1.5x in short / 8K / 40K in its first process,
with exact output. This is bounded retention, not repeated selection. Current
joint metrics and limitations are owned by [Current Status](CURRENT_STATUS.md).

Remaining delivery is cap 2's short budget and prospective repeated selection
of the complete composition. Keep this rule as a bounded constituent through
that next API return; do not scan horizons, confidence thresholds or kernels.
The next implementation requires its own downward budget and composition
point. No production route or numerical contract changes, and the full
50%–200% objective remains active.

## 2026-10-20 current short verifier budget — diagnostic closed

The [one-capture protocol](metadata/qwen36-27b-mtp-v113-short-budget-design-2026-10-20.json)
is [complete](metadata/qwen36-27b-mtp-v113-short-budget-2026-10-20.json), with a
[classifier-label erratum](metadata/qwen36-27b-mtp-v113-short-budget-erratum-2026-10-20.json)
that leaves the timings and decision unchanged.
Frozen v113 P65/O16 d2 preserves all16 output tokens, response and work counts.
In the instrumented interval from first verifier FP8 through the last kernel,
multirow projection families total802.211ms: Gate/Up307.106ms, FP8308.891ms,
Down153.124ms and vocabulary33.091ms. This is diagnostic attribution, not a
new unprofiled phase budget, hardware bound or speed claim.

The native singleton embedding-to-argmax interval is104.624ms with104.050ms
of kernel execution and0.574ms outside kernels. It does not support treating
singleton host submission as the remaining17.903ms saving. Existing v96 already
selected native over generic M1. No singleton Graph-only retry is selected.
Standalone A/B kernels total9.882ms; v76's batching was already neutral.
Gate/Up fusion and NV prefetch lineages remain closed. The trace does not
reopen their parameter searches merely because their families are dominant.

Close collection now. The next implementation must change major projection
execution under the exact numerical contract and return promptly to the API;
no implementation is selected by this diagnostic alone. V113, joint metrics,
production and the full50%–200% objective remain unchanged.

## 2026-10-20 complete private draft Graph — closed neutral API direction

The [complete private-step design](metadata/qwen36-27b-mtp-draft-step-graph-design-2026-10-20.json)
was implemented as v118 with six startup-owned templates, dynamic token/position
inputs, bounded hidden/KV staging, exact actual-length Attention and complete
MLP/head/argmax/optional-confidence execution. The
[completed direction](metadata/qwen36-27b-mtp-draft-step-graph-rejection-2026-10-20.json)
passes 21 input and12 Attention replays, bounded root memory/race/synchronization
checks, and P65/P511 recursive full hidden/KV/logit/confidence and target-prefix
oracles. P65 transaction and failure-recovery checks also pass.

One complete d2 P65/O16 request preserves all16 output tokens and unchanged
acceptance/work, but Decode is14.1618 versus14.1490 token/s: only+0.0907%,
below the predeclared+0.1% direction gate. Pure Prefill is0.90115s
(72.1304token/s), external TTFT0.91995s and total1.97900s. Draft time falls
1.690ms while complete Decode falls0.961ms, far below the remaining17.903ms
budget. The whole draft phase was not removable Graph submission overhead.
These are single-request observations, not a statistically selected speedup.

The short stop closes before8K/40K/d3 and lifecycle qualification. The server
exits normally after the interrupted driver; the completed-request audit passes.
All integration and unused prerequisites are removed, and the frozen v113
server/oracle are restored exactly. No submission, shape, geometry or policy
scan follows. The full50%–200% objective remains unmet; the next architecture
must address whole-runner value under a materially different dataflow.
Production and numerical contracts remain unchanged.

## 2026-10-20 budgeted alternate branch — closed negative API direction

The [fixed-budget design](metadata/qwen36-27b-mtp-budgeted-branch-design-2026-10-20.json)
was fully composed into the isolated v117 service: GPU first-step top2,
confidence stopping, tree controller, exact parent Conv/GDN, ancestor-only
Attention, selected-path KV/draft reconciliation and ten startup verifier Graphs.
Host, kernel and P65/P513 full-state scalar oracles passed; the
[completed direction](metadata/qwen36-27b-mtp-budgeted-branch-rejection-2026-10-20.json)
retains their exact scope and the frozen source/binaries.

One complete d2 P65/O16 API request preserves non-MTP responses/usage and v113
token IDs, but Decode is 12.5256 versus 14.1490 token/s (about -11.47%). Two
alternate rounds reduce net accepted drafts from8 to7; rounds rise7 to8 and
verified rows19 to21. Prior linear-path rank coverage did not predict net work
under the changed topology. Top2/last-confidence and unused primary work are
included in the observed request. Pure Prefill is0.89696s (72.467token/s),
TTFT0.91553s and total2.11303s. This is a valid negative direction screen,
not repeated selection or a hardware/algorithm impossibility claim.

The predeclared short gate closes this version. The next8K request was cancelled
during Prefill and has no timing authority; 40K/d3/lifecycle qualification were
not run. All tree integration and unused prerequisites are removed. Source
matches package-entry ca2abcb/v113; the original frozen server/oracle are restored.
Rebuilds differed only in nine CUDA temporary-source filename bytes; exact
rebuilt and restored identities remain recorded rather than conflated.

No top2, rank, probability threshold, width or depth scan follows. The complete
50%–200% objective and remaining short-context gap stay active. V113 remains
current; the next direction must address whole-runner value under a materially
different architecture. Production and numerical contracts are unchanged.

## 2026-10-20 canonical singleton state — closed neutral

`WP-MTP-SINGLETON-CANONICAL-20261020` implemented one
[sole-prefix ownership change](metadata/qwen36-27b-mtp-singleton-canonical-design-2026-10-20.json):
retain the complete canonical state from the native zero-draft target step and
remove its redundant snapshot/restore. The
[completed direction](metadata/qwen36-27b-mtp-singleton-canonical-rejection-2026-10-20.json)
preserves ten complete prefixes, fifteen transactions and failure/cancel
recovery; six prompt/hidden/KV captures equal v113. One complete d2 P65/O16
request attests one singleton and 157,906,944 fewer copied bytes, but Decode is
14.15036 token/s, only +0.0098% versus v113, below the declared +0.1% gate.
No useful gain is selected. The next 8K request was cancelled at a bounded
Prefill safe point and supplies no timing evidence; 40K/d3 were not run.

Remove the branch and counters; restore v113 source and binaries. Preserve
all evidence, including the initial executable-mode launch failure before any
model process started. No singleton, overlap, kernel or policy scan follows.
The approximately17.903ms short gap and complete50%–200% objective remain.
This is a direction rejection, not a repeated performance or hardware-bound
claim. Production and numerical contracts remain unchanged.

## 2026-10-20 compensated private MLP — closed without API gain

`WP-MTP-COMPENSATED-MLP-20261020` implemented the
[two-component integer composition](metadata/qwen36-27b-mtp-compensated-mlp-design-2026-10-20.json):
primary/residual activation components share two MMA columns, Gate/Up share one
quantization, and Down requantizes after SiLU in existing dead scratch. The
[prerequisite](metadata/qwen36-27b-mtp-compensated-mlp-prerequisite-2026-10-20.json)
and [complete integration decision](metadata/qwen36-27b-mtp-compensated-mlp-rejection-2026-10-20.json)
preserve numerical bounds: native P65 hidden maximum 1.3693%, three recursive
P513 hidden/K/V maxima 1.4164%/1.2866%/0.7227%, and complete target state/logits,
transactions and failure recovery pass.

Real d2 short/8K Decode is 14.1319/15.6224 token/s, approximately
0.121%/0.083% below v113, with unchanged acceptance and verification work.
This is no useful direction gain, not a repeated statistical regression claim.
The predeclared 8K stop closes before 40K/d3/lifecycle qualification. The driver
interrupt occurs during post-request spacing; both requests are complete and
the server exits normally. Integration, executor and unused test target are
removed; frozen source/binaries and raw failures remain in evidence. No
component, group, quantizer or tile scan follows. Retained v113 and the full
50%–200% target remain unchanged; a materially new architecture must address
the remaining short-context gap and return to the real API.

## 2026-10-20 private integer vocabulary — API direction complete

`WP-MTP-INTEGER-HEAD-20261020` selects one
[packed integer dataflow](metadata/qwen36-27b-mtp-integer-head-design-2026-10-20.json)
for the complete private draft vocabulary. Preserve canonical NVFP4 weight
bytes; map codes to exact signed integers, quantize each input group of sixteen,
and accumulate four integer dot products before applying that group's original
scale. Reuse 6400 dead scratch bytes with no weight expansion or omitted rows.
This explicitly changes private head arithmetic, not target state or committed
outputs. Independent complete-logit FP64 error must remain within 2%.
The [completed native admission](metadata/qwen36-27b-mtp-integer-head-native-2026-10-20.json)
passes exhaustive packed-code and integer-dot checks, nine complete real-weight
vectors (FP64 maximum relative L2 0.6592%), guarded changing-input Graph replay,
and bounded root memory/race/synchronization checks. P65 preserves ten complete
prefixes, fifteen transactions, failure recovery and unchanged hidden/KV captures.
The private stream owns quantization and projection; confidence may reuse its
scratch only after projection retires. Nonfinite inputs propagate to the
existing finiteness failure path. Actual successful head calls are counted.

The [complete API return](metadata/qwen36-27b-mtp-integer-head-api-2026-10-20.json)
passes all sixteen d2/d3 API/lifecycle requests with exact committed output.
All timing requests attest integer-head calls equal to proposed tokens. Short
and 8K work is unchanged; each 40K policy accepts one additional token and removes
one round/two verifier rows, so gains include acceptance changes. D2 Decode is
14.149/15.635/11.853 token/s; joint Prefill/TTFT/Decode/total and d3 observations
belong to [Current Status](CURRENT_STATUS.md#mtp-development-status).

Retain v113 through the next complete architecture API return or archival.
This is one-process-per-policy direction, not repeated selection or production
promotion. No integer-head quantizer/geometry/profile scan follows. Short d2
still needs approximately 17.903 ms of Decode savings to reach 1.5x against the
faster fresh non-MTP observation. The 40K d2 observation crosses 1.5x, but the full
50%–200% goal remains unmet and final repetition remains required. Production
stays non-MTP; target semantics and private error bound remain unchanged.

## 2026-10-14 complete verifier Graph — bounded API retention complete

`WP-MTP-FULL-GRAPH-20261014` selects the
[complete verifier dataflow](metadata/qwen36-27b-mtp-full-graph-design-2026-10-14.json).
Bind token IDs and absolute position in a stable device capsule, stage K/V
projection outputs before exact dynamic cache publication, and capture all
64 layers plus full prefix outputs at startup. Preserve every arithmetic and
state boundary. Long Attention updates only the sixteen QK grid extents before
one graph launch; actual causal lengths remain device inputs. Singleton, tiny
and cross-family boundary transactions retain explicit incumbent dispatch.
First admit changing-length Attention replay; within the next two implementation
milestones complete ownership/composition/state checks and return to d2 API.
Useful full direction unlocks d3. Failed API selection removes the composition
and unused prerequisites. No parameter scan, new default or speed claim follows
from a prerequisite. v101 was the package-entry incumbent; the full objective remains active.

The [2026-10-20 Attention prerequisite](metadata/qwen36-27b-mtp-full-graph-attention-2026-10-20.json)
adds the fixed 32-byte token/position capsule, device-length short/long Attention
entry and QK-node identification. Exact arithmetic and the static entry remain
intact. Against frozen v101 kernels, 63 graphs and 252 changing-length replays
preserve complete outputs, long-path probabilities, padding and guards; the
static entry is checked separately. Root memory/race/synchronization checks
pass on the bounded long-path panel. The explicit `q3x_mtp_graph_attention_test` GPU regression
target covers the same 63-graph/252-replay sequence against the static entry.
The [full verifier composition](metadata/qwen36-27b-mtp-full-graph-composition-design-2026-10-20.json)
now binds six startup-owned graphs, a fixed device token/position capsule and
pinned input/results storage. Exact dynamic embedding and Q/K preprocessing
publish K/V at the actual position; the captured body preserves complete prefix
snapshots and full logits. Long replay updates sixteen QK grid extents. Graph
objects outlive borrowed parameter arrays; destruction drains before release.
Tiny, singleton and cross-family transactions retain explicit fallback routes.

The [native admission](metadata/qwen36-27b-mtp-full-graph-native-2026-10-20.json)
passes 21 complete embedding/preprocess/cache cases and bounded memory, race
and synchronization checks. P65, P513 and P8192 pass all ten complete prefix
state/logit checks with actual Graph execution; P65 also passes fifteen
transactions, four verifier-failure recoveries and eight initialization-cancel
recoveries. Six P65 draft capture hashes match v101, preserving that capture's
independent FP64 qualification. This is numerical admission, not speed selection.
The [completed API return](metadata/qwen36-27b-mtp-full-graph-api-2026-10-20.json)
passes all sixteen d2/d3 API/lifecycle requests, matching v101/non-MTP output
and v101 acceptance/work counters. Compared with fresh v101 processes, d2
Decode changes +0.392%/+0.546%/+0.411% at P65/8K/40K; d3 changes
+0.410%/+0.447%/+0.356%. Complete joint metrics belong to
[Current Status](CURRENT_STATUS.md#mtp-development-status). These are single
process direction observations; no noise-qualified selection or production
promotion is claimed by that forward-only record.

The [bounded reverse-order protocol](metadata/qwen36-27b-mtp-full-graph-repeat-design-2026-10-20.json)
is now [complete](metadata/qwen36-27b-mtp-full-graph-repeat-2026-10-20.json).
Eighteen fresh API requests preserve exact outputs, counters and physical routes;
two fresh installed non-MTP processes bracket the C/B continuation. Including
the prior B/C pairs, candidate Decode minima exceed incumbent maxima at all
three workloads for both policies, satisfying the declared engineering
retention rule. Median increments are 0.38%–0.53%. Retain v112 as the development
incumbent and close the Graph package without further scanning or profiling.
This remains bounded engineering evidence, not fully prospective release
qualification. Complete joint metrics and individual limitations are linked
from Current Status.

The 50%–200% goal remains unmet: conservative d2 speedups are
1.461x/1.680x/1.484x versus fresh non-MTP. The next architecture must remove at
least approximately 27.945 ms from short Decode and 236.358 ms from 40K Decode,
while preserving current Prefill, numerical/state, output and resource contracts.
Choose a materially different complete dataflow using those budgets; repeated
Graph measurements and local parameter scans are closed. D3 remains supported
and measured but slower than d2. Production is unchanged.

## 2026-10-14 private BF16 Down — closed without API gain

`WP-MTP-BF16-DOWN-20261014` selects one
[private precision-scope composition](metadata/qwen36-27b-mtp-bf16-down-design-2026-10-14.json).
Keep the Gate/Up INT8 register executor, but return private Down to its
already-resident original BF16 matrix and retained projection executor.
Extra draft cost must be repaid by accepted tokens and fewer target rounds;
closer weights alone do not guarantee better acceptance or speed. Preserve
target arithmetic, state, cache construction and controller. Current P65
state/transaction and independent original-weight FP64 admission return
immediately to d2 API; useful full direction unlocks d3. Stop for over 3%
short regression or nonpositive 8K/40K Decode. One fixed composition, no
precision/quantizer/profile scan. v101 remains incumbent until real API selection.

The [completed composition](metadata/qwen36-27b-mtp-bf16-down-rejection-2026-10-14.json)
passes ten complete prefixes, fifteen transactions, four failure recoveries,
eight initialization-cancel recoveries and independent original-weight FP64.
Hidden maximum row error falls from 1.369% to 1.094%, but both completed API
requests retain exactly the same acceptance and work counts. P65/8K Prefill is
0.892/17.340 s (72.89/472.43 token/s), TTFT 0.910/17.363 s, Decode
13.885/15.330 token/s and total 1.990/33.997 s. Decode changes -0.599%/-0.499%
relative to v101; 8K draft time rises 84.950 ms. These are direction observations,
not a statistically repeated regression. All output IDs match v101; text and usage match both v101
and the non-MTP response oracle. Close before 40K/d3/lifecycle, restore all
runtime and identity files and the measured v101 server, and retain only
evidence. No precision-scope or profile scan follows. Current v101 joint metrics
and the complete unmet objective remain unchanged.

## 2026-10-14 GDN publication density — closed at traffic gate

`WP-MTP-STATE-DENSITY-20261014` selects one
[real API density check](metadata/qwen36-27b-mtp-state-density-design-2026-10-14.json).
Compare canonical and selected next-prefix GDN states at the first eight
commits of one P40000/O16 d2 request, across all 48 layers. Preserve every
ordinary state copy and output boundary. This instrumented process has no
performance authority. Only an aggregate changed 32-byte-sector fraction
below 75% unlocks a separately bounded producer-mask implementation; otherwise
close without a mask or parameter scan. The historical copy budget alone
cannot close the remaining 40K gap. v101 and the full objective stay current.

The [completed diagnostic](metadata/qwen36-27b-mtp-state-density-rejection-2026-10-14.json)
finds 293,311,892 of 301,989,888 BF16 words changed (97.1264%), and
18,863,831 of 18,874,368 sectors changed (99.9442%). Every layer/sample
has at least 99.3245% changed sectors. Adjacent state hashes chain exactly;
all sixteen generated IDs match the retained 40K output prefix. This fails
the declared traffic-opportunity gate, so no bitmap or selective-copy kernel
is implemented. Remove all instrumentation, rebuild restored source and restore
the measured v101 server. Instrumented timings have no performance authority;
current joint metrics and the full unmet objective remain unchanged. No
granularity, mask or threshold scan follows.

## 2026-10-14 shared-prefix feedback — closed negative

`WP-MTP-SHARED-FEEDBACK-20261014` selects one
[request-local rate controller](metadata/qwen36-27b-mtp-shared-feedback-design-2026-10-14.json).
Choose budgets 1..configured 2/3 from exactly inferable prefix rewards and
measured costs of executed budgets. Keep the current confidence stop; never
invent a larger all-accepted continuation or counterfactual latency. Each budget
is tried once in descending order, then empirical committed tokens per complete
round time select it. Feedback eligibility is restricted to budgets no larger than the chosen budget,
independent of the current outcome, to avoid rejection-only sampling bias.
No cross-request learning. Apply only to requested outputs
at least 64; short requests stay unchanged. This experimental policy has no
statistical optimality guarantee and cannot close the short gap. Host shared-
feedback/state checks and native mixed-budget transactions return immediately
to d2 API; nonpositive 8K/40K closes, useful full direction unlocks d3. One policy,
no threshold/length/profile scan or production promotion. v101 remains incumbent.

The [completed API direction](metadata/qwen36-27b-mtp-shared-feedback-rejection-2026-10-14.json)
passes ten complete prefixes and fifteen transactions (four with mixed budgets), four failure
recoveries and eight initialization-cancel checks. The one host-only repair
restricts feedback eligibility to avoid rejection-only sampling bias. Both
completed API outputs and generated IDs match v101/non-MTP. P65/8K Prefill is
0.890/17.334 s (72.99/472.60 token/s), TTFT 0.909/17.356 s, Decode
13.949/12.982 token/s and total 1.984/36.999 s. At 8K Decode falls 15.740%;
rounds rise from 98 to 140 despite verified rows falling from 285 to 280.
The first budget-2 observation is confidence-shortened and rejected; the
empirical greedy policy then stays at budget 1. This is a failure of this
policy, not a general scheduling or hardware limit. Stop before 40K/d3/lifecycle,
restore all runtime/control/identity files to v101, and preserve the evidence.
No warmup, horizon, exploration or threshold scan follows. Existing v101 joint
metrics and the complete unmet objective remain unchanged.

## 2026-10-14 softmax warp tree — closed without local gain

`WP-MTP-SOFTMAX-WARP-TREE-20261014` selects one
[exact synchronization-scope change](metadata/qwen36-27b-mtp-softmax-warp-tree-design-2026-10-14.json).
Preserve the current aligned softmax expressions, operand trees and lifetime
barrier; move only the final five reduction stages to the first warp. The
older negative composition also cached exponentials, so it did not separately
select this component. One fixed implementation passes full numerical/Graph
and race/synchronization checks, then one mirrored four-layer historical real
QK-score comparator. Its single-row groups are local evidence, not M2..4 API
qualification. Require at least 5% aggregate saving and no role regression above
3% before composition. Any retained dependency must return through current
numerical/state checks and d2 API in the next milestone; useful full direction
unlocks d3. No parameter/profile scan or production change. v101 stays current.

The [completed local check](metadata/qwen36-27b-mtp-softmax-warp-tree-rejection-2026-10-14.json)
preserves all 72 complete probability/guard cases and 216 Graph replays, with
zero root racecheck and synccheck errors. Registers fall from 30 to 25; shared
storage remains 1,024 bytes, with no stack or spills. The mirrored four-layer
real-payload comparison has no useful gain: aggregate baseline/candidate times
are 0.575585/0.577330 ms, a 0.3031% increase rather than the required 5% saving.
This small difference is not a statistically established regression. All 96
blocks preserve first/last complete outputs. These historical single-row groups
are local evidence, not actual M2..4 API timings. Close before runtime or API
composition; no further variant or profiler run. v101 and the full unmet
objective remain unchanged.

## 2026-10-14 private W8A8 representation — closed at numerical gate

`WP-MTP-PRIVATE-W8A8-FEASIBILITY-20261014` selects one
[host representation check](metadata/qwen36-27b-mtp-private-w8a8-design-2026-10-14.json).
Keep current group128 INT8 MLP weight codes and scales, quantize only private
MLP activations per group128, and assess integer-dot scaling instead of BF16
weight reconstruction. FC, complete Attention and target arithmetic stay fixed.
An authenticated 64-row full-layer comparison against original-weight FP64
must satisfy the unchanged 2% hidden/K/V maximum row bound before GPU work.
One fixed quantizer, no scan. A pass unlocks a separately bounded native
executor with current recursive numerical/state admission and immediate d2
API return; it does not prove native arithmetic, acceptance or speed. v101
and the complete 1.5x–3x objective remain unchanged.

The [completed host check](metadata/qwen36-27b-mtp-private-w8a8-rejection-2026-10-14.json)
rejects this representation: hidden maximum row error is 2.8484%, above the
unchanged 2% limit. K/V remain identical to the original FP64 layer because
FC/Attention are unchanged. The original reference, capture and retained six
weight payload hashes are authenticated. This teacher-forced representation
check does not simulate native FP32 group reduction or prove recursive behavior.
Close before any GPU executor, runtime or API change; no quantizer/group scan.
Current v101 joint API metrics remain unchanged and the full objective remains
unmet. No performance or hardware-limit conclusion follows.

## 2026-10-14 lossless draft register consumption — closed at static gate

`WP-MTP-LOSSLESS-REGISTER-20261014` selects one
[compact-register executor](metadata/qwen36-27b-mtp-lossless-register-design-2026-10-14.json)
for original BF16 FC/Q/K/V/O; retained INT8 MLP stays fixed. Preserve the old
lossless packet format but reconstruct pairs directly in MMA registers after
compact async staging, removing the rejected decoded shared tile and separate
producer warps. The fixed 128-thread M64C8K128 executor must use at most 96
registers, 37,376 shared bytes and no stack/spills. A failed static gate closes
before GPU with no resource scan. Full words/output/Graph admission and one
five-matrix real-weight comparison must improve aggregate time by at least 5%
without a role regressing above 3%. Otherwise close before runtime. A positive
dependency must return through startup ownership, current state checks and d2
API in the next implementation milestone; d3 requires useful whole direction.
No format, geometry or profile scan, target reduction change or production
promotion is selected. v101 and the full objective remain unchanged.

The [static compilation](metadata/qwen36-27b-mtp-lossless-register-rejection-2026-10-14.json)
uses 168 registers, exceeding the declared 96-register limit. Shared storage
is 37,376 bytes, with zero stack and spills. Register capacity permits at most
three 128-thread CTAs, whereas the shared-storage bound permits four; these
are static limits, not measured occupancy. Close before any GPU process,
numerical admission, real-weight timing or runtime/API integration. Keep only
the prototype and resource evidence. No register-cap, unroll, geometry, format
or profiler variants follow. v101 and its complete joint API metrics remain
unchanged; this result neither completes the goal nor establishes a hardware
limit.

## 2026-10-14 stable-address state publication — closed negative

`WP-MTP-STATE-VMM-FEASIBILITY-20261014` owns one
[VMM capability check](metadata/qwen36-27b-mtp-state-vmm-feasibility-design-2026-10-14.json).
Assess whether the current driver can remap a stable device address between
two bounded allocations while exposing the selected complete bytes. This is
distinct from pointer-only state selection: canonical observer addresses must
still expose the selected full state. Current cudaMalloc arenas cannot support
this directly; any successor needs explicit arena/pool ownership, immutable
prefix rotation and graph/state lifecycle design. The 217.858 ms older copy
family is an opportunity ceiling, not a saving prediction or full-goal closure.
One capability/primitive process, no model timing or runtime changes. Unsupported
operation closes before implementation; a pass only unlocks a bounded real-state
composition. v101 and the complete performance objective remain unchanged.

The [capability probe](metadata/qwen36-27b-mtp-state-vmm-feasibility-2026-10-14.json)
passes on SM87/driver 13020: two stable-address switches and three full-byte
checks, with complete cleanup. Minimum allocation granularity is 2 MiB, so the
78,446,592-byte recurrent region needs 1,245,184 bytes of page padding.
This does not establish a performance gain. The selected
[real-state publication comparator](metadata/qwen36-27b-mtp-state-vmm-publication-design-2026-10-14.json)
authenticates historical full Conv/GDN capture bytes and includes padding-copy
cost. One process uses three B-C-C-B rounds with 32 publications per block;
all bytes must match and mapping must save at least 25% in every round before
runtime ownership work proceeds. Negative results close without an allocator
or driver scan. Current model numerical qualification and real API return
remain mandatory for any successor.

The [real-state comparator](metadata/qwen36-27b-mtp-state-vmm-publication-rejection-2026-10-14.json)
rejects this direction before runtime edits. Complete copy publication averages
0.875–0.878 ms per state; remap publication takes 3.554–3.920 ms, or 4.06–4.46x
copy cost across all three mirrored rounds. All 12 timed block comparisons and
four warmup comparisons preserve the full 79,691,776 bytes; 384 publications
were timed. Cleanup passes with no GPU handles remaining. Driver support does
not imply lower latency. No arena/pool/Graph implementation or allocator,
granularity or driver scan follows. These local real-state timings do not
replace the unchanged v101 Prefill/TTFT/Decode/total API metrics or prove a
hardware limit. The complete 1.5x–3x objective remains unmet.

## 2026-10-14 certified QK execution — closed negative

`WP-MTP-QK-CERTIFIED-API-20261014` selects one
[certificate/executor composition](metadata/qwen36-27b-mtp-qk-certified-api-design-2026-10-14.json).
Bind the proved complete-product domain to startup-owned per-layer flags.
Check existing KV once, then every appended or overwritten range before QK;
request initialization and abort invalidate certificates. Sticky invalid flags
may conservatively disable acceleration after rejection but never enable it.
Queries are checked outside the position loop. Preserve original arithmetic
when either certificate fails. Complete Attention/Graph and current full-state
prefix checks return immediately to d2 API, then d3 only for useful direction.
One executor and one correctness repair; no parameter/profile scan. Short
slowdown above 3% or nonpositive 8K/40K closes. v101 remains incumbent; this
long-only candidate has no claim on the short gap or production selection.

The [completed API screen](metadata/qwen36-27b-mtp-qk-certified-api-rejection-2026-10-14.json)
passes 90 complete Attention cases, 270 Graph replays and certificate lifecycle
controls, ten complete prefixes at both P513 and P8192, 15 state transactions,
four failure recoveries and eight initialization-cancel checks. Kernel usage
is 93 registers and zero stack/local memory. Both completed API outputs and
all acceptance/work counters match v101. P65/8K Decode is 13.926/15.306 token/s,
changing -0.304%/-0.654%; 8K verification adds 116.169 ms. Prefill is 0.899/17.309 s
(72.29/473.27 token/s), TTFT 0.918/17.331 s, total 1.995/33.992 s. Unchanged short,
draft and Prefill variation is not attributed to this edit. The declared 8K
stop closes before 40K, d3 or API lifecycle qualification; server cleanup exits 0.
Remove all runtime/identity changes and restore v101. No certificate, compiler,
guard, geometry or profiler scan follows. Mathematical equivalence survives;
this executor does not improve complete API performance.

## 2026-10-14 QK complete product domain — arithmetic prerequisite

`WP-MTP-QK-EXACT-PAIR-DOMAIN-20261014` owns one
[complete-product proof](metadata/qwen36-27b-mtp-qk-exact-pair-domain-design-2026-10-14.json).
Unlike the closed nonzero-even-product certificate, constrain both original
products to exact finite FP32 values, allowing BF16 signed zeros. This excludes
the odd negative-underflow zero that invalidated the earlier unguarded fusion.
Check all dimensions in the authenticated historical captures and native raw
bits across all BF16 codes and directed corner triples. Higher reductions and
publication stay fixed. One native correctness process, no parameter scan or
performance claim. A pass only unlocks a separately bounded current-payload
certificate/executor/API composition; v101 remains the runtime incumbent.

The [completed prerequisite](metadata/qwen36-27b-mtp-qk-exact-pair-domain-2026-10-14.json)
passes all 163,864,576 historical BF16 operands, including the original zeros.
One SM87 process checks 1,073,741,824 directed combinations: 532,709,376 admitted
combinations have zero raw-bit differences; excluded-domain and signed-zero
controls pass. Each BF16 code occupies each operand position against fixed
corner triples; this is not exhaustive enumeration of all four operands.
The algebraic proof supplies the complete-domain argument. Native machine code
contains separate original FADD and candidate FFMA instructions. No runtime
or performance baseline changes. The next composition must bind certificates
to actual KV append/reset/rollback lifetimes, validate current full-state
prefixes, and return immediately to d2 API before any d3 qualification.

## 2026-10-14 QK cache certificate — closed at host screen

`WP-MTP-QK-CACHE-CERTIFICATE-20261014` owns one
[host feasibility check](metadata/qwen36-27b-mtp-qk-cache-certificate-design-2026-10-14.json).
The earlier exact QK pair fusion had negative API value with guards inside
every position iteration; its separate guard cost was not measured. Assess
whether an immutable full-prefix KV certificate can move those checks to
append/reset boundaries. Authenticate all original even-offset key/query
operands in the four historical 40K layer captures. All four complete prefixes
and queries must satisfy the unchanged exact-product predicate before a
separate current-payload/native composition is selected. Failure closes this
whole-prefix certificate without granularity or threshold scanning. Historical
Legacy captures cannot admit current MTP execution. No GPU run or speed claim
belongs to this screen; v101 and the full objective remain unchanged.

The [completed host screen](metadata/qwen36-27b-mtp-qk-cache-certificate-rejection-2026-10-14.json)
authenticates all eight captures and checks 81,932,288 BF16 operands. Queries
pass; each complete key prefix fails on exact zeros (124/122/129/111 by layer).
The frozen whole-prefix predicate therefore closes before native implementation
or GPU tests. This is no current-MTP numerical result or hardware bound. No
runtime, binary or performance baseline changes. A proof covering both products
of each pair, including signed zeros, is a separate unselected mathematical
question; it does not reopen this predicate or admit a new executor.

## 2026-10-14 private Attention PV partition — closed neutral

`WP-MTP-PRIVATE-PV-SPLIT-20261014` / `AC-MTP-GREEDY-v101` selects one
[private PV composition](metadata/qwen36-27b-mtp-private-pv-split-design-2026-10-14.json).
The current d2 short/40K gaps are 31/348 ms. Existing real-API node evidence
attributes only 11.61 ms to confidence, so that direction is not implemented.
Private PV accounts for 136.68 ms in that older instrumented interval: a partial
opportunity, not a promise to close either gap. Partition only private long PV
into four contiguous segments, preserve original QK/softmax and final BF16/gate
boundaries, and merge FP32 partials. Target arithmetic stays exact; private
reassociation requires unchanged original-weight FP64 hidden/KV admission,
including recursive long-position captures with the retained INT8 MLP.
One executor, one correctness repair, no parameter/profile scan. Full shape,
guard/Graph and real numerical checks return immediately to d2 API; useful long
direction unlocks d3. Short >3% slowdown or nonpositive 8K/40K closes. No new
performance result or production change is claimed. v101 remains incumbent.

The [completed direction](metadata/qwen36-27b-mtp-private-pv-split-rejection-2026-10-14.json)
passes ten synthetic PV cases, 30 Graph replays and root synchronization checks,
plus five real-operand prefix cases and 15 Graph replays. Recursive original-weight
FP64 checks at 8K, 40K and through position 44095 pass the unchanged 2% bound;
maximum hidden errors are 1.147%, 1.269% and 1.562%. These recursive references
share a captured initial KV; they do not independently regenerate that cache.
Ten complete target prefixes, fifteen transactions and failure/cancel checks pass.
One static correctness repair handles empty final partitions before async staging.

All three completed d2 responses and generated IDs match v101, with unchanged
acceptance/work. Decode is 13.965/15.417/11.614 token/s at P65/8K/40K, changing
-0.0240%/+0.0688%/-0.0528%. Prefill is 0.891/17.339/92.706 seconds,
72.95/472.46/431.47 token/s; TTFT0.910/17.361/92.741 seconds and
complete request1.984/33.901/114.699 seconds. At40K private draft time falls
43.161 ms, but whole Decode does not improve. Do not attribute unchanged target
or Prefill variation to the partition. The declared nonpositive40K stop closes
before API lifecycle requests or d3; server cleanup exits0. Restore v101 runtime
and measured server. Retain only the diagnostic long-capture/FP64 tools, with
no new speed or production claim. No partition/tile/profile scan follows.
The retained capture/FP64 utility also passes on restored v101 at 8K: three
recursive steps reach hidden/K/V maxima 1.083%/1.029%/0.665%, conditional on
the captured initial KV. This is a numerical tool check, not a new timing run.


## 2026-10-14 private MLP register consumption — API return complete

`WP-MTP-MLP-REGISTER-20261014` / `AC-MTP-GREEDY-v100` owns one
[register-consumer composition](metadata/qwen36-27b-mtp-mlp-register-design-2026-10-14.json).
The prior INT8 MLP passed accuracy but lost API performance. M1 has no second
token warp group to reuse its decoded shared tile. Decode fixed packed bytes
directly into MMA A registers, reuse two scales per K128 tile, and remove that
intermediate and barrier. Preserve original BF16 FC/Attention, quantizer,
startup hashes, rounding and ascending MMA. One M64C8K128/128-thread executor
must use at most 96 registers, 20 KiB shared and no stack/spills. Exhaustive actual
scale/pair checks, complete projection/Graph and P65 native FP64/state gates
return immediately to d2 API; useful direction unlocks d3. Short >3% slowdown
or nonpositive 8K/40K closes. One correctness repair; no format/geometry/profile
scan. The full 1.5x–3x objective and production boundary remain; v98 is the frozen comparator.

The [complete API return](metadata/qwen36-27b-mtp-mlp-register-direction-2026-10-14.json)
passes 44,171,264 decoder cases, eight full projection/guard cases and 24 Graph
replays. The consumer uses 80 registers, 20 KiB shared and no spills/stack.
Native original-weight FP64 hidden maximum is 0.013687553141055709; ten complete
prefixes, fifteen transactions and bounded failure/cancel checks pass. Both
policies complete eight API/lifecycle requests with identical target IDs and
responses. Private acceptance changes are reported rather than hidden.
Retain the register consumer as a development dependency through the next
complete MTP architecture API return or archival. No format/geometry/profile
scan follows. Current Status owns joint metrics and remaining gaps; this is
single-process direction evidence, with no production promotion or full-goal claim.


## 2026-10-14 invariant softmax divisor — closed neutral

`WP-MTP-SOFTMAX-RECIPROCAL-20261014` / `AC-MTP-GREEDY-v99` owns one
[exact divisor composition](metadata/qwen36-27b-mtp-softmax-reciprocal-design-2026-10-14.json).
Frozen v98 machine code repeats reciprocal/refinement inside each probability
loop. Hoist only denominator-dependent work; preserve the compiler's three-FMA
normal quotient sequence in a conservative finite range, with original division
elsewhere. No exponential, reduction, causal or state boundary changes. This is
separate from the rejected exponential-lifetime path. Exhaustive mantissa/edge,
complete probability/Graph and P513 state checks precede immediate d2 API.
Short >3% or 8K >1% slowdown stops; nonpositive 40K closes. Useful full direction
unlocks d3. One correctness repair; no profile or parameter scan. The measured
0.515-second long softmax family bounds scope, not promised saving or closure of
the 0.460-second long and 0.051-second short gaps. Production remains unchanged.

The [completed direction](metadata/qwen36-27b-mtp-softmax-reciprocal-rejection-2026-10-14.json)
passes 260,047,137 division pairs, 72 complete probability cases, 216 Graph
replays, ten P513 prefixes, fifteen transactions, four failure recoveries and
eight initialization-cancel checks. The actual server hoists the reciprocal,
using 31 registers, 1 KiB shared and no stack/local memory. All eight d2 API
and lifecycle requests complete with identical target IDs, output and work.
8K/40K Decode is 15.183/11.567 token/s, only +0.0346%/+0.0507% versus v98;
this is neutral, not a useful gain. Prefill is 17.312/92.798 seconds,
473.19/431.04 token/s; TTFT17.334/92.833 seconds, total34.129/114.880 seconds.
P65 is 13.752 token/s but does not execute the changed long softmax. Unchanged
Prefill and short-path variation are not attributed. The d2 driver had already
finished normally when the neutral stop was inspected; no process was signalled.
Remove all runtime/profile changes, restore v98, and stop before d3 or profiling.
The full goal stays unmet. A separately bounded successor may remove the prior
private M1 INT8 consumer's unshared decoded-weight tile and scale reloads;
that ownership question is recorded as unselected, not a tuning continuation
or performance claim.

## 2026-10-14 native private INT8 MLP — closed negative

`WP-MTP-MLP-INT8-NATIVE-20261014` / `AC-MTP-GREEDY-v98` owns the
[fixed native composition](metadata/qwen36-27b-mtp-mlp-int8-native-design-2026-10-14.json).
The BF16 Attention boundary passes the host prerequisite at 1.145% maximum
hidden relative L2. Startup packs and authenticates only Gate/Up/Down as the
unchanged group128 INT8 format; FC, Attention and cache construction remain BF16.
One M64C8K128 executor uses async packed double buffering and a single decoded
BF16 tile, preserving ascending K16 MMA. Original weights remain resident;
271,564,800 extra bytes obey the 8-GiB reserve. No request allocation or fallback.
Compile resource admission and independent original-weight native FP64 plus
complete target-state checks return immediately to d2 P65/8K/40K API; useful
direction unlocks d3. Short >3% slowdown or nonpositive 8K stops. One correctness
repair, no geometry/format/profiler scan. Draft acceptance may change; target
outputs remain exact. The full objective and production boundary remain intact.

The [completed native direction](metadata/qwen36-27b-mtp-mlp-int8-native-rejection-2026-10-14.json)
passes all six packed payload hashes, native original-weight FP64 hidden max
0.013687553141055709, ten complete prefixes, fifteen transactions, four failure
recoveries and eight initialization-cancel checks. K/V captures match v98
bitwise. The two completed d2 API outputs and all acceptance/work fields match
v98 and baseline output. Decode is 13.565/14.958 token/s at P65/8K, down
1.090%/1.448%; private draft time increases 13.937/232.690 ms. Prefill is
0.891 s / 72.93 token/s and 17.312 s / 473.20 token/s; TTFT is 0.910/17.334 s,
total 2.016/34.382 s. Unchanged Prefill variation is not attributed to this edit.
The declared 8K stop closes before 40K/d3/lifecycle; server cleanup exits 0.
All runtime/integration changes are removed and v98 restored. No compression,
geometry, pipeline or profiler scan follows. Numerical feasibility survives
as evidence only; storage saving did not return API value. Current metrics,
production and the unmet complete objective remain unchanged.

## 2026-10-14 BF16 Attention boundary with private INT8 MLP — host prerequisite complete

`WP-MTP-DRAFT-MLP-INT8-20261014` / `AC-MTP-GREEDY-v97` owns one
[fixed precision partition](metadata/qwen36-27b-mtp-draft-mlp-int8-design-2026-10-14.json).
The error-origin audit reproduces the old failure and finds amplified Attention
residual error already entering MLP. Preserve FC and the complete Attention
branch in BF16; quantize only Gate/Up/Down with the unchanged prior group128
format. This is one residual-boundary partition, not a role/quantizer sweep.
One authenticated 64-row full-layer FP64 check must pass the unchanged 0.02
hidden/K/V bound. Failure closes before GPU; a pass only unlocks a separately
bounded native consumer/API composition. Storage arithmetic saves about 31%;
no speedup, numerical waiver or production change follows from that estimate.
The [completed host check](metadata/qwen36-27b-mtp-draft-mlp-int8-2026-10-14.json)
passes at hidden max 0.011448350779191842, with exact original K/V. The native
composition above is the bounded successor; host PASS grants no speed claim.

## 2026-10-14 private draft representation error origin — complete

`WP-MTP-DRAFT-ERROR-ORIGIN-20261014` owns one
[host attribution](metadata/qwen36-27b-mtp-draft-error-origin-design-2026-10-14.json).
The earlier fixed group128 mixed draft failed hidden accuracy on two of 64 rows;
K/V passed. Reproduce its exact saved original/mixed outputs, then distinguish
per-operator same-input representation error from propagated stage error.
Original FC/norms, quantizer, inputs and 0.02 bound stay fixed. Existing reference
hidden-flow audits already found no semantic mismatch. This is no new GPU run,
format/role sweep or revival of rejected execution. The diagnostic must either
select one evidence-backed precision-ownership question for a separate bounded
composition or close; no runtime retention or performance claim follows.
The full objective and current v98 metrics remain unchanged.
The [completed attribution](metadata/qwen36-27b-mtp-draft-error-origin-2026-10-14.json)
authenticates every old quantized payload and reproduces all six saved outputs
bitwise. Rows 18/62 expose Attention residual cancellation; propagated MLP input
error exceeds same-input direct MLP weight error. The fixed BF16 Attention
boundary above is the selected successor, not an individual-role scan.

## 2026-10-14 singleton target-tail dispatch — API return complete

`WP-MTP-SINGLE-TAIL-20261014` / `AC-MTP-GREEDY-v96` owns one
[singleton composition](metadata/qwen36-27b-mtp-single-tail-design-2026-10-14.json).
P65 d2 contains one zero-draft final round. Reuse the existing native scalar
Decode transaction for that single row, preserving its complete snapshot,
commit, draft-cache and failure semantics. M2..4 and non-MTP code stay unchanged.
A new scalar-row counter distinguishes physical work from direct GDN writes.
Ten complete prefixes and existing transactions precede immediate d2 P65 API.
A short gain above 0.5% unlocks full d2/d3 checks; >3% regression or neutral
short direction removes. No kernel, graph, policy or profiler scan. This can
address the 59-ms short gap only; the 450-ms long gap and full goal remain.

The [complete API return](metadata/qwen36-27b-mtp-single-tail-direction-2026-10-14.json)
passes ten complete M1..4 prefixes, fifteen transactions, four verification
failure recoveries, eight initialization-cancel checks and all sixteen d2/d3
API/lifecycle requests. Outputs, generated IDs, acceptance and semantic work
match v94. Four d2 P65-equivalent requests use one scalar tail and improve
Decode by 0.68%–0.90%; no-tail long/d3 observations are effectively unchanged.
Physical GDN/copy-elision counters now exclude scalar snapshots. Retain this
small route dependency through the next complete verifier API return or
archival, without mirrored selection or production authority. No single-tail
kernel/graph/policy scan follows. [Current Status](CURRENT_STATUS.md) owns the
joint metrics and remaining gaps; the complete 1.5x–3x goal remains unmet.


## 2026-10-14 paired Gate/Up with bounded input ring — closed negative

`WP-MTP-GATE-RING-PAIR-20261014` / `AC-MTP-GREEDY-v95` owns one
[projection composition](metadata/qwen36-27b-mtp-gate-ring-pair-design-2026-10-14.json).
Node attribution selects Gate/Up from the remaining 59/450-ms short/40K gap.
The old v33 fusion already owns four warps per role. This successor composes
its exact BF16-to-SiLU publication with the later paired decoder and M2/M3
K512 two-slot ring. M4 and scalar paths remain unchanged. One 256-thread CTA
must use at most 128 registers, 8,384 shared bytes and no stack/spills, with
at least two resident CTAs. Complete output/Graph/race and P65 state checks
return immediately to d2 API, then d3 for useful direction. Short >3% slowdown
or nonpositive 8K closes. One correctness repair; no resource/geometry/profile
scan. Numerical contracts, production and the full 1.5x–3x objective remain.

The [completed direction](metadata/qwen36-27b-mtp-gate-ring-pair-rejection-2026-10-14.json)
passes eight complete output cases, 24 Graph replays, root race/sync checks,
nine model prefixes and fifteen state/cache transactions. A startup identity
omission was caught before generation and repaired without changing arithmetic.
The completed P65/O16 d2 request preserves all output IDs and work, but Decode
falls from 13.621 to 12.841 token/s (-5.727%); verification adds 68.228 ms.
Prefill is 0.892 seconds / 72.91 token/s, TTFT 0.910 seconds and total 2.078.
The declared short stop closes the composition before 8K/40K/d3 or lifecycle
qualification. Remove all kernel/integration/identity changes and restore v94.
No resource, geometry, stage or profiler variant follows. The smaller shared
buffer did not yield API value; current paired metrics and the full goal remain.


## 2026-10-14 GDN graph-node attribution — complete

`WP-MTP-GRAPH-NODE-BUDGET-20261014` owns one
[node-level API attribution](metadata/qwen36-27b-mtp-graph-node-budget-design-2026-10-14.json).
The two latest GDN attempts yielded no retained improvement. The old graph
aggregate cannot identify recurrence, projections, Conv or copy costs. One
unchanged measured v94 P40000/O256 d2 API capture enables node tracing and
requires exact output/ID/work reconciliation. It is a new bounded diagnostic,
not a repeat or extension of the closed v79 aggregate package. Node overhead
qualifies every timing; unprofiled v94 remains the performance anchor. No NCU,
second capture, runtime change or d3 qualification-only loop is selected.
The result must select a concrete next implementation question or close.

The [completed node trace](metadata/qwen36-27b-mtp-graph-node-budget-2026-10-14.json)
reconciles all output IDs and work with v94. All 14,640 GDN and Conv calls are
visible. Recurrence costs 0.424 seconds and Conv 0.045; target projections cost
13.884 seconds, including Gate/Up 5.309 and FP8 5.335. QK/PV cost 4.324 seconds.
These are instrumented duration sums, not a new benchmark or hardware bound.
Return priority to projections; no further GDN register/Conv variant or profile
is selected. The concrete next question is whether the admitted paired NV
weight decoder and M2/M3 K512 ring can now compose with the old Gate/Up/SiLU
publication. Plain v33 fusion and its warp mapping are already tried; only the
changed producer lifetime/decoder composition qualifies as a successor. Its
M3 shared-storage estimate is 8,384 bytes versus old 32,000, not a performance
claim. A separately bounded composition must preserve all arithmetic and return
to v94's real API comparison; no local resource win selects production.
The complete objective and unprofiled paired performance remain unchanged.

## 2026-10-14 complete value-row register ownership — closed at compilation

`WP-MTP-ROW-OWNED-GDN-20261014` / `AC-MTP-GREEDY-v94` owns one
[row-owned GDN composition](metadata/qwen36-27b-mtp-row-owned-gdn-design-2026-10-14.json).
The same 59/450-ms short/40K gap selects removing shared state staging and
inactive dot lanes. Each thread holds one complete BF16 state row in packed
registers, retaining both ascending scalar FMA chains and all rounding points.
This differs from the old scalar path's repeated global reads and v81's
key-partitioned cross-token state. The unchanged Conv and per-token prefix
schedule remain. One fixed 128-thread composition must compile without spills,
with at most 160 registers and 2 KiB shared memory. Complete state/Graph/race
and P65 admission return immediately to d2 API, then d3 for useful direction.
One correctness repair maximum; short >3% slowdown or nonpositive 8K stops.
No register, launch, layout or profiler scan; no production or goal-completion claim.

The [compile admission record](metadata/qwen36-27b-mtp-row-owned-gdn-rejection-2026-10-14.json)
closes this construction before GPU execution. Both the initial source and one
bounded exact operand-lifetime correction use 255 registers, 72 bytes of stack
and 68 bytes each of spill stores/loads. Neither meets the declared 160-register,
zero-spill gate. The specific allocation cause is not established; the first
CSE hypothesis is not confirmed by the correction. All candidate code and
identities are removed. No numerical, model or performance pass is claimed;
no register-cap, unroll, launch or profiler scan follows. Retained v94 and the
complete 1.5x–3x objective remain unchanged.

## 2026-10-14 direct Conv/GDN consumption — closed neutral

`WP-MTP-CONV-GDN-20261014` / `AC-MTP-GREEDY-v93` owns one
[Conv/GDN composition](metadata/qwen36-27b-mtp-conv-gdn-design-2026-10-14.json).
The remaining d2 short/40K Decode gaps are 59/450 ms. Original per-token GDN
consumes the exact BF16-rounded convolution inside its CTA, eliminating the
separate Conv publication and history copy. Immutable Conv prefixes have unique
writers; canonical history stays untouched until the existing full commit.
Q/K convolution is recomputed for three value heads, charged to the whole API.
This changes neither GDN row-state ownership nor its ordered scalar arithmetic;
the closed v81 row-reuse version remains closed. One implementation and at most
one correctness repair return through complete output/state/Graph/race and P65
transaction admission directly to d2 API, then d3 only for useful direction.
Short >3% slowdown or nonpositive 8K closes; no launch/tile/profile scan.
The full objective and production boundary remain unchanged.

The [completed API direction](metadata/qwen36-27b-mtp-conv-gdn-rejection-2026-10-14.json)
passes fifteen full output/state/Conv cases, 45 Graph replays, bounded root race/
sync checks, nine model prefixes, fifteen transactions and all eight d2 API
requests. Short/8K/40K Decode differences are only +0.096%/+0.076%/+0.076%;
verification saves about 1.5/21.6/21.8 ms. This establishes no useful composed
improvement. Remove the private kernel and integration, restoring v94. D3,
profiling and further Conv/launch/graph variants are not selected. Complete
output, state and resource passes do not select performance or complete the
1.5x–3x goal. The existing retained metrics remain unchanged.

## 2026-10-14 lazy accepted-tail KV composition — API return complete

`WP-MTP-TAIL-KV-20261014` / `AC-MTP-GREEDY-v92` owns one
[seed-preserving tail composition](metadata/qwen36-27b-mtp-tail-kv-design-2026-10-14.json).
The closed v34 batch rebuilt the whole prefix before first publication. This
successor keeps the now-reusable seed and waits until the second commit before
batching two/three valid target-conditioned tail rows through the current
weight-major C8 kernel. One-row tails retain scalar append. Prepared future rows
remain inaccessible until individually committed; cancellation and failure
invalidate readiness. Original canonical state copies and observer order stay.
The 344-ms 40K reconciliation interval includes mandatory copies and cannot
alone close the 515-ms complete-Decode gap. One implementation and one correctness
repair pass P65 full state/cache/recovery plus explicit lazy-publication checks,
then return directly to d2 API. Short >3% slowdown or nonpositive 8K closes;
useful long direction unlocks d3. No batch-size, kernel, stream or profiler scan.
The complete objective and non-MTP production boundary remain unchanged.

The [complete API return](metadata/qwen36-27b-mtp-tail-kv-direction-2026-10-14.json)
passes nine full prefixes with explicit lazy-tail assertions, fifteen state/cache
transactions, four verification-failure recoveries, eight initialization-cancel
recoveries and all sixteen d2/d3 API/lifecycle requests. Generated IDs, output,
acceptance, target work and seed reuse match v93. D2 at 8K/40K uses 71/64
batches for 142/128 tail rows, reducing reconciliation by 57.3/50.0 ms.
Full Decode improves about 0.45%/0.30%; short direction is also positive.
Retain this small development dependency through the next complete verifier
architecture API return or archival, not mirrored selection or production.
The v93 short path also passes this composition. No further batching, stream,
kernel or profiler scan follows. [Current Status](CURRENT_STATUS.md) owns the
paired metrics and remaining gap; the full 1.5x–3x objective is still unmet.

## 2026-10-14 short multi-row Attention — API return complete

`WP-MTP-SHORT-ATTENTION-20261014` / `AC-MTP-GREEDY-v91` owns one
[short-interval composition](metadata/qwen36-27b-mtp-short-attention-design-2026-10-14.json).
At S65..511, predecessor v86 dispatches QK/softmax/PV/gate separately for each
verified row. One independent row/head CTA keeps exact intermediate scores and
probabilities on chip, preserving all scalar trees, ordered FMAs and the BF16
boundary before gating. The existing exact S<=64 fused source supplies the
arithmetic construction; long-context kernels remain unchanged. This targets
the 72.798-ms short gap and cannot close the independent 505.033-ms 40K gap.
One implementation and one correctness repair pass full output/Graph/race and
P65/P509 state admission, then return immediately to the short d2 API. Useful
short direction unlocks the complete d2/d3 panel; >3% short regression stops.
No parameter or profiler scan. Production and the full objective stay unchanged.

The [completed composition](metadata/qwen36-27b-mtp-short-attention-direction-2026-10-14.json)
passes 72 complete output/guard cases with 216 Graph comparisons, root race/sync
checks, eighteen full prefixes, fifteen transactions and all sixteen d2/d3
API/lifecycle requests. All generated IDs, output, usage and non-timing work
match v86. Short Decode improves from 13.447/12.560 to 13.583/12.699 token/s
for d2/d3 (about 1.0%/1.1%); streaming, nonstream and recovery observations
have the same useful direction. Long Decode is effectively unchanged. Retain
only as a bounded development dependency through the next complete verifier
architecture API return or archival, not noise-qualified architecture selection
or production. No short tile/unroll/launch/profile scan follows. Prefill, TTFT,
Decode and total observations are owned by [Current Status](CURRENT_STATUS.md).
The complete 1.5x–3x target remains unmet.

## 2026-10-14 private packed-weight pipeline — closed negative

`WP-MTP-DRAFT-PACKED-PIPELINE-20261014` / `AC-MTP-GREEDY-v90` implements the
[fixed producer/consumer composition](metadata/qwen36-27b-mtp-draft-packed-pipeline-design-2026-10-14.json)
selected below. Compile/resource admission precedes GPU word/output and race
checks, full P65 state/cache/recovery, then immediate d2 P65/8K/40K API. Useful
direction unlocks d3. One implementation, one correctness repair maximum,
no format/tile/producer-count or profiler scan. Original Prefill and target
arithmetic remain; sidecars add resident bytes and must preserve the reserve.
Short >3% slowdown or nonpositive 8K closes this version. The full goal remains.

The [completed composition](metadata/qwen36-27b-mtp-draft-packed-pipeline-rejection-2026-10-14.json)
passes all BF16 bit patterns, complete projection/guard checks, root race/sync
checks, every real packed weight, nine target prefixes and fifteen state/cache
transactions. P65/O16 d2 Decode falls from 13.447 to 12.273 token/s (-8.737%).
Drafting adds 90.631 ms and cache reconciliation 16.700 ms; verification remains
unchanged within the observed variation. Prefill is 0.892 seconds / 72.88 token/s,
TTFT 0.910 seconds and total 2.132 seconds. Startup retains 10,219,192,320 free
bytes, above the reserve. The short stop prevents further completed API cases.
All packet/owner/pipeline/dispatch/identity code is removed, restoring v86.
No format, producer-count, tile, stage or profiler scan follows. Correct lossless
compression did not yield full-runner benefit; the full goal remains unmet.

## 2026-10-14 private BF16 traffic assessment — complete, composition closed

`WP-MTP-DRAFT-LOSSLESS-BF16-20261014` owns one
[fixed lossless packet assessment](metadata/qwen36-27b-mtp-draft-lossless-bf16-design-2026-10-14.json).
Authenticate all eight original draft dense matrices. Each 256-word packet
uses sign/mantissa bytes plus four-bit exponent deltas if its exponent range
fits; otherwise retain raw BF16. Count all headers, offsets, alignment and raw
fallback, and roundtrip every source word. Less than 20% total saving closes
before GPU work, without a format scan. A pass only selects a separately
bounded owner/decoder/executor and real API composition. The 1.004-second
historical private-matrix interval is not proof of bandwidth limitation, and
20% of that interval cannot alone close the 0.505-second 40K gap. No runtime,
production or performance claim changes; the full 1.5x–3x goal stays active.

The [complete checkpoint assessment](metadata/qwen36-27b-mtp-draft-lossless-bf16-assessment-2026-10-14.json)
reduces 849,346,560 dense-weight bytes to 667,009,984 bytes including every
header, offset and raw fallback: 21.468% saving. Every original word roundtrips,
with 1,215 explicit wire-packet checks. No GPU process or API measurement runs.
Select one private M1 sidecar plus producer/consumer composition described in
the assessment: 128 producer and 128 consumer threads, original M64/C8/K128
MMA, two buffers and explicit ready/retired ownership. Original weights remain
for Prefill, so sidecars add resident bytes; eight-byte alignment needs safe
raw loads. One implementation and one correctness repair return directly to
state admission and d2 API, then d3 only for useful direction. No format/tile/
producer-count scan. The composition is now closed negative above and has no
retained runtime code.

## 2026-10-14 arithmetic FP8 reconstruction — closed negative

`WP-MTP-FP8-ARITHMETIC-DECODE-20261014` / `AC-MTP-GREEDY-v89` owns one
[exact arithmetic composition](metadata/qwen36-27b-mtp-fp8-arithmetic-decode-design-2026-10-14.json).
Replace verifier FP8 shared lookup/init with exact FP32 bit assembly and binary
scaling, including subnormals, signed zero and signed qNaN. Existing global
weights, launch geometry, scalar chains and state remain unchanged. This removes
the lookup boundary rather than changing its layout; no table or decoder scan.
All raw/sidecar codes, complete projections and P65 state admission return
immediately to d2 P65/8K/40K API; useful direction unlocks d3. One composition,
one correctness repair maximum, no profiling. Short >3% slowdown or nonpositive
8K closes this version. Retained v86 and the full 1.5x–3x goal stay unchanged.

The [completed direction](metadata/qwen36-27b-mtp-fp8-arithmetic-decode-rejection-2026-10-14.json)
passes all 512 raw/sidecar encoding cases, 72 full projection/guard checks,
nine complete target prefixes and fifteen state/cache transactions. Frozen API
machine code preserves non-FTZ binary scaling and has no spills. P65/O16 d2
Decode falls from 13.447 to 12.355 token/s (-8.125%), with identical output,
acceptance and physical work; verification adds 99.619 ms. Prefill is 0.891
seconds / 72.99 token/s, TTFT 0.917 seconds and total 2.131 seconds. The short
stop cancels further qualification; no completed 8K/40K/d3 result is claimed.
Remove all candidate code and identities, restoring v86. Replacing shared
traffic with exact arithmetic did not yield API benefit. No decoder instruction,
lookup, launch or profiler scan follows. The full goal remains unmet.

The [lineage erratum](metadata/qwen36-27b-mtp-fp8-arithmetic-lineage-erratum-2026-10-14.json)
links the older half2 exponent-bias reconstruction. The unary FP32 path is a
related instruction variant, not a new mathematical family; no further
scalar/packed arithmetic conversion variants are selected.

## 2026-10-14 sign-separated NVFP4 codebooks — closed negative

`WP-MTP-NV-SIGNED-CODEBOOKS-20261014` / `AC-MTP-GREEDY-v88` implements the
[fixed composition](metadata/qwen36-27b-mtp-nv-signed-codebooks-design-2026-10-14.json)
selected by the completed NVFP4 attribution. Only M3 coupled Gate/Up changes:
positive pair and packed scale words are replicated per lane, restoring exact
signs before the unchanged BF16 multiplication and FP32 chains. Shared storage
is 30,720 bytes; compilation must preserve at most 96 registers and zero spills.
One composition and at most one correctness repair return through full state
admission to P65/8K/40K d2 API, then d3 only for useful direction. Short >3%
slowdown or nonpositive 8K closes this version. No profile or parameter scan.
The full goal, corrected v86 comparator and production boundary remain unchanged.

The [completed direction](metadata/qwen36-27b-mtp-nv-signed-codebooks-rejection-2026-10-14.json)
passes 95-register/no-spill admission, twelve full projection/guard cases, nine
complete target prefixes and fifteen state/cache transactions. P65/O16 d2
Decode falls from 13.447 to 12.152 token/s (-9.637%), with identical output,
acceptance and physical work; verification adds 119.084 ms. Prefill is 0.892
seconds / 72.88 token/s, TTFT 0.918 seconds and total 2.152 seconds. The short
stop cancels further qualification; no complete 8K/40K or d3 result is claimed.
All candidate runtime code and identities are removed, restoring v86. Reduced
lookup conflicts and preserved compiled residency did not yield API benefit.
No table, occupancy, launch or profiler scan follows. The full goal is unmet.

## 2026-10-14 NVFP4 verifier feed attribution — complete

`WP-MTP-NV-FEED-AUDIT-20261014` owns the
[fixed Gate attribution](metadata/qwen36-27b-mtp-nv-feed-audit-design-2026-10-14.json).
The current real API budget selects NVFP4 verification after private-head and
static Graph changes fail to close the gap. Choose four of all 64 real Gate
weights by complete code entropy before timing; retain scales and tensor scale.
Run unchanged M3 coupled-feed projection with deterministic BF16 inputs and
complete scalar-output checks, then one small-resident median-layer NCU capture.
This is checkpoint-weight-only attribution, not prompt activation or API speed.
No full-model replay, new kernel, parameter scan or repeated profile is opened.
The result must select a bounded dataflow or reject a proposed mechanism; the
full goal and corrected v86 comparison remain unchanged.

The [completed attribution](metadata/qwen36-27b-mtp-nv-feed-audit-2026-10-14.json)
preserves complete scalar outputs across four preselected Gate layers and one
small-resident NCU capture. Shared transactions are 2.008x ideal, with 50.187%
excess; L1/TEX throughput is 82.66%. These are not latency or DRAM-bandwidth
fractions. Registers limit residency to five blocks; actual shared partition
167,936 bytes leaves room for a different exact representation. The bounded
host proof reconstructs all original FP4-pair/E4M3 operands and lane bank
addresses over 2,097,152 combinations. Select one M3 coupled Gate/Up composition:
sign-free pair/scale codebooks with lane-owned words, restoring original sign
bits before unchanged scaled multiplication. Its 30,720-byte shared allocation
preserves five-block capacity if compilation stays at most 96 registers without
spills. Extra initialization/bit work must still win at the real API. The fixed composition is now closed negative above; no more profiling or
table/launch scanning follows. Retained API metrics and the full goal are unchanged.

## 2026-10-14 full-Attention static tails — closed neutral

`WP-MTP-FA-TAIL-GRAPHS-20261014` / `AC-MTP-GREEDY-v87` implements the
[fixed static-island composition](metadata/qwen36-27b-mtp-fa-tail-graphs-design-2026-10-14.json).
Capture each full-Attention O/MLP tail together with following three GDN layers;
position-dependent Q/K/V, preprocessing and Attention remain outside. Original
arithmetic and state ordering are unchanged. The resulting 51 startup-owned
execs remove 112 direct submissions per round at one additional graph launch.
This is the first captured FA-tail boundary, not a GDN granularity scan.
P65 full transactions and P513 prefix admission return directly to d2 API;
useful direction unlocks d3. One composition, one correctness repair maximum,
no profiler or granularity scan. Short >3% slowdown or nonpositive 8K closes
this version. The full goal and 72.798/505.033-ms short/40K gaps remain active.

The [completed direction](metadata/qwen36-27b-mtp-fa-tail-graphs-rejection-2026-10-14.json)
passes eighteen complete prefixes, fifteen state/cache transactions and all
eight d2 API/lifecycle requests, preserving outputs, token IDs and physical work.
Decode changes only +0.151%/+0.120%/+0.050% at P65/8K/40K. At 40K, Prefill is
92.786 seconds / 431.10 token/s, TTFT 92.821 seconds, Decode 11.543 token/s and
total 114.913 seconds. This single-process near-tie establishes no useful gain;
Prefill variation is not attributed to unchanged Prefill code. Remove the new
islands and restore the original 48 GDN graphs and v86 identity. D3 is not run;
no granularity, dynamic-update or profiler scan follows. The full goal remains.

## 2026-10-14 native private draft vocabulary — closed negative

`WP-MTP-NATIVE-DRAFT-HEAD-20261014` / `AC-MTP-GREEDY-v86` implements the
[fixed native head](metadata/qwen36-27b-mtp-native-draft-head-design-2026-10-14.json).
The existing real 40K trace assigns 0.876 seconds to private vocabulary work.
Original canonical NVFP4 weights feed a weight-major BF16 MMA directly, without
Marlin repacking or its additional 715 MB retained copy. Full vocabulary,
private FP64 error bound and exact target state/output remain mandatory.
One composition and at most one correctness repair return immediately to
P65/8K/40K d2 API; useful direction unlocks d3. Short >3% slowdown or nonpositive
8K closes the version. No parameter/profile scan is selected. The complete
1.5x–3x goal and 72.798/505.033-ms short/40K floor gaps remain unchanged.

The [completed direction](metadata/qwen36-27b-mtp-native-draft-head-rejection-2026-10-14.json)
passes all quantized code pairs, nine complete real vocabulary FP64 checks,
nine target prefixes, fifteen state/cache transactions and recovery checks.
P65/O16 d2 Decode is 13.025 token/s, 3.144% below retained v86, with unchanged
output and acceptance. Prefill is 0.892 seconds / 72.90 token/s, TTFT 0.910
seconds and total 2.062 seconds. Private drafting adds 35.982 ms. The short
stop closes this version; 8K is cancelled before completion and 40K/d3 are not
run. All new code and identities are removed, restoring v86. No head mapping,
stage or profiler scan follows. Original packed storage without a repacked
copy did not make the complete runner faster; the full goal remains active.

## 2026-10-14 seed-prefix/draft overlap — closed negative

`WP-MTP-SEED-PREFIX-20261014` / `AC-MTP-GREEDY-v85` implements the
[bounded scheduling composition](metadata/qwen36-27b-mtp-seed-prefix-design-2026-10-14.json).
Overlap the known seed's first eight target layers with private drafting, backfill
remaining rows through those layers and join ordinary multirow verification.
Full state/logit/cache semantics, target arithmetic and publication remain exact.
Additional prefix weight reads and GPU contention are charged to complete API
Decode. One fixed composition and one correctness repair return directly to
P65/8K/40K d2, then d3 only for useful direction. No prefix-length or profiler
scan; the 72.798/505.033-ms short/40K floor gaps and full 1.5x–3x goal remain.

The [completed direction](metadata/qwen36-27b-mtp-seed-prefix-rejection-2026-10-14.json)
passes eighteen complete prefix comparisons, fifteen state/cache transactions,
four verification-failure and eight initialization-cancel recoveries. P65/O16 d2
then reaches only12.585 Decode token/s,6.414% below retained v86; Prefill is
0.891 seconds /72.92 token/s, TTFT0.918 seconds and total2.109 seconds.
Output IDs and non-timing work are unchanged. This meets the predeclared short
stop. The subsequent8K request is cancelled;40K/d3 are not run. Remove all new
runtime paths and restore v86. No prefix-depth/stream-priority or profiler scan
follows. The full1.5x–3x objective remains; a successor must remove work or a
material dependency without repeating this duplicated-prefix composition.

## 2026-10-14 softmax corrected package — complete

`WP-SOFTMAX-PACKAGE-20261014` completes the remaining installed-artifact
boundary of the synchronization repair, under this [bounded delivery contract](metadata/qwen36-27b-softmax-package-design-2026-10-14.json).
Package 0.8.2 changes version/export identity only beyond the admitted repair.
Keep the predecessor immutable; check the new exact installation, full supported
capacity, 98 pinned capability answers, cancellation and fresh-process recovery.
A dedicated numerical-patch mode performs those gates plus bounded request reuse;
it does not claim a renewed two-hour reliability qualification or relax the
existing full qualification mode. This is a correctness delivery prerequisite,
not another performance experiment. The [completed delivery](metadata/qwen36-27b-softmax-package-2026-10-14.json)
passes six applicable host/package tests, all 184 patch and 11 restart requests,
complete predecessor response comparisons and owned shutdown. No further patch
qualification loop is open. Resume MTP architecture work against the corrected
v86 comparator: the short/40K full-Decode deficits remain 72.798/505.033 ms for
the 1.5x floor. The full 1.5x–3x objective stays active. The seed-prefix/draft overlap
successor above is closed negative and adds no retained runtime or gain.

## 2026-10-14 softmax synchronization repair — source/API and package complete

`WP-SOFTMAX-SYNC-20261014` preserves the independently discovered shared-memory
lifetime repair after the PV experiment closes. Maximum reduction readers must
all retire before the same shared array becomes denominator storage. One CTA
barrier fixes the MTP aligned softmax and three scalar/reference sites without
changing operands, arithmetic, reductions or publication. The source retains the
v79 PV/GDN architecture with isolated MTP identity
`q3x.sm87.admission.mtp-softmax-sync-api.v86`.

The [completed repair/API record](metadata/qwen36-27b-softmax-sync-2026-10-14.json)
passes CPU comparison, root racecheck, complete P513 state/transactions and all
24 API/lifecycle requests across corrected non-MTP/d2/d3. Complete responses
match their predecessors; MTP token IDs and all non-timing work remain equal.
Current Status owns the paired measurements and remaining target gap. The source
repair is retained. The separate exact-package work above closes corrective
installation without a deployed process switch or full release claim. Subsequent MTP optimization uses
the corrected comparator and unchanged full 1.5x–3x goal. No repeated PV work or
extra hardware profile is selected by this closure.

## 2026-10-14 partitioned PV pipeline — closed negative

`WP-MTP-PV-PARTITIONED-20261014` / `AC-MTP-GREEDY-v84` implements the
[bounded producer/consumer composition](metadata/qwen36-27b-mtp-pv-partitioned-design-2026-10-14.json)
selected by the real-payload diagnosis. Two producer warps independently fill
four stages; the existing six head warps consume original ordered FP32 FMAs.
Stage completion and reader retirement govern visibility and reuse. Grid,
causal lengths, operands and BF16 publication remain fixed. Complete synthetic,
Graph/synchronization and captured-output checks plus P513 state admission
return immediately to d2 API, then d3 for useful direction. One implementation,
no profiler or role/tile/stage/carveout scan. Extra barriers and producer resources
must pay for themselves in complete Decode; the full 1.5x–3x goal remains.

Admission exposes an independent softmax lifetime race in both MTP and its
reference predecessor. Preserve this finding separately from PV performance:
add the missing maximum-read retirement barrier, validate it with instrumented
and CPU comparisons, and revalidate scalar/API behavior. The existing production
binary is not silently replaced, and the non-MTP denominator must be refreshed
before full MTP target qualification. This correctness prerequisite may survive
PV rejection; it is not another pipeline/parameter variant.

The [completed negative direction](metadata/qwen36-27b-mtp-pv-partitioned-rejection-2026-10-14.json) passes 72 synthetic/Graph cases, root
synchronization/race checks, captured 40K PV output, nine full model prefixes
and fifteen state/cache transactions. P8192 d2 API Decode is 15.008 token/s,
0.90% below v79; Prefill is 17.296 seconds / 473.64 token/s, TTFT 17.318 seconds,
and total 34.309 seconds. Output and non-timing work are unchanged. This meets
the predeclared nonpositive-8K stop: remove the partitioned pipeline, retain the
independent synchronization fix, and do not run 40K/d3 for this rejected version.
The composition includes that fix, so no isolated PV API attribution is claimed.
The real-payload PV component itself is slower (1.793 versus 1.341 ms median).

## 2026-10-14 ordered PV feed attribution — complete

`WP-MTP-PV-FEED-AUDIT-20261014` owns the
[bounded current-path diagnosis](metadata/qwen36-27b-mtp-pv-feed-audit-design-2026-10-14.json).
The current 40K d2 trace assigns 2.180 seconds to ordered PV. Earlier shared-row
variants did not transfer fewer logical V loads to API gains. Capture one complete
real first-round V/probability/output payload through the existing P40000/O256
API and require unchanged output/work. Instrumented timing is not a baseline.
Restore ordinary source, then use that small resident payload for one unchanged
PV timing/check and one NCU capture. Resolve actual residency, stalls and traffic
before selecting another architecture. No full-model replay or parameter scan;
this work neither changes production nor closes the 1.5x–3x goal.

The [completed real-payload diagnosis](metadata/qwen36-27b-mtp-pv-feed-audit-2026-10-14.json)
preserves full API output/work and checks every replay against the captured
complete PV output. Eight unchanged-kernel samples have median 1.340736 ms.
One small-resident NCU capture succeeds with over 58 GiB free: the actual shared
partition is 167,936 bytes and permits four CTAs, disproving the initial
small-partition suspicion. The 48-CTA grid supplies three CTAs per SM on average;
achieved occupancy is 32.12%. Shared conflicts are only 245 excess wavefronts
out of 19,565,140. Neither carveout nor bank-layout scanning is selected.
Barrier and wait diagnostics motivate one independent producer/consumer
pipeline proposal: 64 producer threads, the existing 192 consumers, unchanged
four buffers and original ordered FMAs. Stage completion and reader retirement
replace lockstep copying/compute ownership. This differs from the removed
single-barrier deletion and shared-row variants. Extra synchronization and
producer resources must still pay for themselves at API return. No new runtime
is retained by this diagnosis; restore v79, do not repeat the profile, and
implement this one bounded composition next. Counters do not predict speedup;
current Prefill/TTFT/Decode/total and the full goal remain unchanged.

## 2026-10-14 compact exact FP8 table — closed negative

`WP-MTP-COMPACT-FP8-TABLE-20261014` / `AC-MTP-GREEDY-v83` implements the
[bounded compact representation](metadata/qwen36-27b-mtp-compact-fp8-table-design-2026-10-14.json)
selected by the preceding hardware attribution. Store exact FP8 values in a
16-KiB replicated BF16 table, publish through four vector-store rounds, and
expand to unchanged FP32 operands at consumption. Original global feed,
scalar chains and all state boundaries remain fixed. Resource-derived shared
partition preferences preserve the original launch-bound residency targets;
there is no carveout/layout sweep. Complete lookup/output/state admission
returns directly to d2 API, and useful long direction unlocks d3. One
implementation, no further profiler or codebook variant. Measured conflicts
motivate the change but do not predict the full 1.5x–3x goal.

The [completed direction](metadata/qwen36-27b-mtp-compact-fp8-table-rejection-2026-10-14.json)
passes all 8192 lookup cells, 72 complete projection/guard cases, nine model
prefixes and fifteen state/cache transactions. P65 d2 Decode is 13.048 token/s,
3.028% below retained v79, crossing the predeclared short-request stop. Prefill
is 0.892 seconds / 72.89 token/s, TTFT 0.918 seconds and total 2.067 seconds;
output and non-timing work are unchanged. Verification adds 35.027 ms. Remove
the table, shared preferences and route identity, restoring v79. The already
started 8K request is cancelled at a bounded Prefill safe point; no long-context
or d3 result is claimed. The server exits cleanly. No repeats, further profiler
or codebook/carveout variants follow. Reduced theoretical bank multiplicity did
not establish whole-product value. Current retained metrics and the full
1.5x–3x objective remain unchanged.

## 2026-10-14 exact FP8 operand-feed attribution — complete

`WP-MTP-PROJECTION-FEED-AUDIT-20261014` owns a
[bounded component diagnosis](metadata/qwen36-27b-mtp-projection-feed-audit-design-2026-10-14.json)
of the unchanged M3 canonical GDN QKV verifier. The real v79 API cost record
selects this projection family; four actual layers are fixed from code entropy
before timing. A checkpoint-weight-only rotation precedes one median-layer NCU
capture. Deterministic inputs do not claim real-prompt activation coverage or
API speed. Only 200 MiB of weights are resident, with the 8-GiB reserve;
full-model replay, parameter scans and repeated counter attempts are excluded.
The question is whether memory delivery, shared-table traffic or instruction
issue limits this exact kernel. Counters must guide a bounded implementation,
not become a hardware impossibility or a whole-product performance claim.

The [completed small-resident capture](metadata/qwen36-27b-mtp-projection-feed-audit-2026-10-14.json)
succeeds without the earlier full-model replay exhaustion. Four layer medians
are 0.3715–0.3726 ms; one median-layer NCU capture records 4,122,311 shared
wavefronts against 1,728,000 ideal (2.386x), 80 registers and 50% theoretical
occupancy. These are component facts, not model speedups or DRAM saturation.
The next bounded proposal is an exact BF16 codebook representation with
32 lane replicas: 16 KiB rather than the rejected 32-KiB FP32 table, four
128-bit publication rounds rather than 32 scalar rounds, and FP32 expansion
at consumption. Host proof covers all 256 codes and the subword bank mapping;
GPU numerical/resource admission and full API composition are still pending.
No runtime change or new API metric is selected here. Do not reuse the
profiler's heuristic speedup estimate as a forecast or repeat its collection.

## 2026-10-14 draft FC embedding-prefix cache — closed neutral

`WP-MTP-FC-EMBEDDING-CACHE-20261014` / `AC-MTP-GREEDY-v82` follows the
[current budget](metadata/qwen36-27b-mtp-v79-budget-2026-10-14.json).
The [bounded design](metadata/qwen36-27b-mtp-fc-embedding-cache-design-2026-10-14.json)
memoizes the FC accumulator after its immutable normalized-embedding half.
A full-token-tag hit continues the same FP32 MMA sequence at the hidden half;
a miss publishes the prefix and completes the original sequence. This removes
repeated computation without changing arithmetic, proposal policy or target
state. A fixed 256-slot owner cache costs 5 MiB and invalidates on reset.
Synthetic full-FC/cache guards and complete P65 state admission return directly
to d2 API, then d3 only for useful direction. One implementation, no cache-size
scan or profiler. The 1.004-second matrix-family budget includes other work;
cache hits cannot be assumed to close the full 1.5x–3x objective.

The [completed direction](metadata/qwen36-27b-mtp-fc-embedding-cache-rejection-2026-10-14.json)
passes 18 complete FC rows, nine model prefixes, fifteen state/cache transactions
and all eight d2 API/lifecycle checks. Hits are 5/21, 242/344 and 192/350 at
P65/8K/40K, with unchanged output and proposal/verification work. At 8K/40K,
Decode is 15.184/11.566 token/s, only +0.269%/+0.074% without noise qualification.
Prefill is 17.213/92.056 seconds (475.91/434.52 token/s), TTFT 17.236/92.091
seconds and total 34.030/114.139 seconds. The 40K total increases 0.071 seconds.
Close this neutral product direction, remove cache/dispatch/profile/counters
and restore v79 without d3 or a cache scan. Refined source-correlated timing
isolates FC at 0.199 seconds, not the whole 1.004-second matrix family; even
ideal half-FC reuse is only 0.099 seconds before misses and overhead. Select
future implementations from the exact removable role before coding. The full
goal remains unchanged; mathematical work elimination alone is not enough.

## 2026-10-14 current MTP budget — reconciled

`WP-MTP-V79-BUDGET-20261014` freezes one unchanged v79 P40000/O256 d2
NSys capture under its [design](metadata/qwen36-27b-mtp-v79-budget-design-2026-10-14.json).
The earlier v64 trace predates the draft matrix, Gate input and GDN graph
changes. Reconcile current kernel families, copies and uncovered intervals
against the same-output/work API anchor before selecting another architecture.
This is diagnostic only: no NCU replay, parameter scan, performance baseline
update or production change. One capture is the bound; the full goal remains.

The [completed trace](metadata/qwen36-27b-mtp-v79-budget-2026-10-14.json)
preserves the exact request, output IDs and complete work. Its low-overhead
graph granularity measures 1,680 fixed GDN segments at 10.800 seconds in
aggregate, with QK/PV separately at 2.143/2.180 seconds. Graph-external draft
matrix work is 1.004 seconds and draft vocabulary is 0.876 seconds. Graph
spans include internal gaps; neither their kernel breakdown nor total GPU
idle is measured. Graph-external D2D plus outside-span time is 0.342 seconds,
less than the retained 0.440-second 40K d2 gap, but it is not a global upper
bound on copies/gaps. No new runtime architecture or speedup is selected.
Return to materially changed projection/operand consumption; the source audit
confirms simple Gate/Up/SiLU fusion and full-input FP8 persistence were already
implemented in closed experiments. No second trace or geometry scan follows.
Normal v79 Prefill/TTFT/Decode/total observations remain unchanged, and the
full 1.5x–3x objective remains open.

## 2026-10-14 GDN row-state reuse — closed neutral

`WP-MTP-GDN-ROW-REUSE-20261014` / `AC-MTP-GREEDY-v81` owns the
[bounded recurrence composition](metadata/qwen36-27b-mtp-gdn-row-reuse-design-2026-10-14.json).
Retain packed BF16 state across speculative rows within each independent
value-row batch. Publish every complete prefix and use unrounded FP32 updates
for current outputs. Original Conv, reduction trees and scalar FMA order stay
fixed. Precompute independent Q/K/gates, then recur, then apply the original
head normalization. Additional registers/shared inputs may offset saved state
reads; the historical 0.438-second GDN budget cannot alone establish the full
goal. Complete synthetic and state admission returns directly to d2 API;
3% slowdown or nonpositive 8K closes, useful long direction unlocks d3.
One implementation, no geometry/profiler scan or production change.

The [completed direction](metadata/qwen36-27b-mtp-gdn-row-reuse-rejection-2026-10-14.json)
passes twelve complete synthetic cases, nine complete model prefixes, fifteen
state/cache transactions and all bounded failure/cancel recovery checks.
At 8K d2, Prefill is 17.218 seconds / 475.79 token/s, TTFT 17.240 seconds,
Decode 15.150 token/s and total 34.072 seconds. Output and work match v79,
but Decode differs only +0.04% without noise qualification. Close this neutral
construction after the completed 8K request; the subsequent 40K request is
interrupted, and d3/full lifecycle are not qualified. Remove the new kernel,
dispatch and profile, restore v79, and do not scan row/warp/register geometry.
Fewer logical state loads did not establish useful API value or a DRAM saving.
Production and the full 1.5x–3x objective remain unchanged.

## 2026-10-14 direct Down scales — closed neutral

`WP-MTP-DOWN-SCALE-DIRECT-20261014` / `AC-MTP-GREEDY-v80` owns the
[bounded scale delivery change](metadata/qwen36-27b-mtp-down-scale-direct-design-2026-10-14.json).
Keep consumer-order packed Down weights and read existing canonical eight-bit
scales, removing six-bit extraction and warp shuffles. Exact arithmetic and
state remain unchanged; additional scale loads may offset the saved work.
The historical packed-Down budget is 2.206 seconds; the remaining 40K complete
Decode gap is 0.440 seconds. Neither predicts a gain. Complete synthetic and
state admission returns immediately to d2 API; 3% slowdown or nonpositive 8K
closes, useful long direction unlocks d3. One implementation, no layout or
profiler scan, no allocation, policy or production change.

The [completed direction](metadata/qwen36-27b-mtp-down-scale-direct-rejection-2026-10-14.json)
passes 30 complete synthetic cases, nine complete prefixes, fifteen state/cache
transactions and all eight d2 API/lifecycle checks with unchanged output/work.
At 8K/40K, Decode is 15.188/11.573 token/s, only +0.29%/+0.13% in one process
without noise qualification. Prefill is 17.150/92.031 seconds (477.66/434.64
token/s), TTFT is 17.172/92.066 seconds and total is 33.963/114.103 seconds.
The 40K total is 0.034 seconds longer. This does not select useful retention;
remove the new kernel dispatch/profile and restore v79 without d3, repetition,
profiler or layout scanning. Production and the full goal remain unchanged.

## 2026-10-14 prefix commit overlap — closed

`WP-MTP-COMMIT-OVERLAP-20261014` / `AC-MTP-GREEDY-v79` owns one
[bounded synchronization change](metadata/qwen36-27b-mtp-commit-overlap-design-2026-10-14.json).
Canonical target-state copies and target-conditioned draft KV append read
completed immutable snapshots and write disjoint owners. Enqueue both before
joining, preserving full canonical state and draft cache before every observer.
Pointer-only selection is not admitted by the existing raw-arena contract.
The historical 0.264-second total D2D cost is only a loose ceiling; this cannot
alone establish the remaining 0.440-second 40K budget or the complete goal.
Complete state/failure admission returns immediately to d2 API; 3% slowdown
or nonpositive 8K closes. Useful long direction unlocks d3. One implementation,
no stream/priority scan, no numerical, policy or production change.

The [completed direction](metadata/qwen36-27b-mtp-commit-overlap-rejection-2026-10-14.json)
passes nine complete prefixes, fifteen state/cache transactions, four verification
failure recoveries and eight initialization-cancel recoveries. At8K d2,
Prefill is17.220s /475.72token/s, TTFT17.242s, Decode15.124token/s and
total34.103s. Output and work match v79, but reconciliation saves only1.922ms
and full Decode adds22.129ms. This is no useful API gain, not a statistically
established regression. Stop before40K/d3 and the remaining lifecycle panel;
remove the scheduling/profile change and restore v79. No stream/priority scan
follows. Production and the full1.5x–3x objective remain unchanged.

## 2026-10-14 fixed GDN segments — API composition complete

`WP-MTP-GDN-GRAPHS-20261014` / `AC-MTP-GREEDY-v78` owns the
[startup graph composition](metadata/qwen36-27b-mtp-gdn-graphs-design-2026-10-14.json).
Capture 16 three-GDN-layer segments for each M2/M3/M4 at owner construction,
then replay unchanged kernel sequences at fixed addresses. Attention remains
dynamic. No graph creation or allocation enters a round; partial construction
and destruction have explicit drain/release boundaries. The 0.242s historical
uncovered interval cannot alone establish the remaining 0.662s goal.
Complete state admission returns immediately to d2 API; 3% slowdown or
nonpositive 8K closes, useful long direction unlocks d3. One fixed granularity,
no graph/profiler scan, no arithmetic, policy or production change.

The [completed composition](metadata/qwen36-27b-mtp-gdn-graphs-direction-2026-10-14.json) passes complete
prefix/state/cache/failure admission and all sixteen d2/d3 API/lifecycle checks,
with identical output IDs and work against v75. D2 8K/40K Decode reaches
15.144/11.558 token/s, saving 0.192/0.223 seconds of full Decode. D3 improves
at all three buckets and remains slower than d2. At 40K d2, Prefill is
91.968 seconds / 434.93 token/s, TTFT 92.004 seconds and total 114.068 seconds.

Retain this fixed graph composition as a bounded development dependency
through the next complete architecture API return or archival. It is one
process per maximum, not statistical selection or production qualification.
D2 reaches 1.644x/1.470x non-MTP at 8K/40K; the remaining 40K full-Decode gap
is 0.440 seconds and the full 1.5x–3x goal remains open. No graph granularity,
threshold or profiler scan follows. Production remains unchanged.

## 2026-10-14 private vocabulary coverage — closed

`WP-MTP-VOCAB-COVERAGE-20261014` / `AC-MTP-GREEDY-v77` owns one
[fixed coverage check](metadata/qwen36-27b-mtp-vocab-coverage-design-2026-10-14.json).
The unchanged private head scans248,320 words, historically0.874s at40K d2.
Audit whether entry-target top8,192 includes at least99% of accepted drafts
in every bucket, with zero exclusions in the short bucket. Preserve actual
full-vocabulary drafting, confidence, target verification and output throughout.
Additional D2H/ranking/logging makes timing diagnostic only. One d2 API panel;
passing coverage admits d3, then one sparse-head implementation or archival.
No vocabulary-size/history/threshold scan, model retraining or production change.

The [completed diagnosis](metadata/qwen36-27b-mtp-vocab-coverage-rejection-2026-10-14.json)
preserves all eight d2 API/lifecycle outputs and complete work. The fixed list
covers only6/8,150/157 and139/150 accepted proposals atP65/8192/40000,
failing every predeclared coverage gate. Some accepted words rank far below
the cutoff in entry logits; the next-token distribution is not a sufficient
selector for later draft tokens under this construction.

Close this specific shortlist without d3, sparse-head implementation or a
size/history/threshold scan. All diagnostic code is removed and v75 restored.
Timing includes extra copies/ranking and cannot select performance. This is
observed membership, not counterfactual acceptance or a bound on all vocabulary
reduction methods. Production and the full1.5x–3x objective remain unchanged.

## 2026-10-14 A/B row dispatch — closed neutral

`WP-MTP-AB-BATCH-20261014` / `AC-MTP-GREEDY-v76` owns the
[bounded dispatch composition](metadata/qwen36-27b-mtp-ab-batch-design-2026-10-14.json).
Each GDN layer currently launches the same BF16 A/B kernel separately per row.
One existing tile dispatch combines independent token CTAs with identical
per-output FMA/reduction/publication. In the40K d2 workload this removes9,600
launches; the old kernel budget is only0.184s, so this cannot alone claim the
remaining0.662s goal. Full state admission returns immediately to API;
3% slowdown or nonpositive8K closes, useful long direction unlocksd3.
One dispatch edit, no kernel/shape/profiler scan or production change.

The [completed direction](metadata/qwen36-27b-mtp-ab-batch-rejection-2026-10-14.json)
passes nine complete prefixes, fifteen state/cache transactions and four
verification-failure recoveries. P8192/O256 d2 output and work match v75;
Decode14.973215 versus14.973271 token/s is effectively identical, with only
0.0000643s full-Decode difference. Prefill is17.184s /476.73token/s,
TTFT17.206s and total34.236s; its variation is not caused by this dispatch edit.
The predeclared nonpositive stop closes before40K/d3 and full lifecycle checks.
The dispatch/profile edit is removed and v75 restored, with no parameter or
profiler scan. Fewer launches alone did not transfer value to the API.
Production and the full1.5x–3x objective remain unchanged.

## 2026-10-14 Down input reuse — closed negative

`WP-MTP-DOWN-INPUT-RING-20261014` / `AC-MTP-GREEDY-v75` owns the
[bounded input producer](metadata/qwen36-27b-mtp-down-input-ring-design-2026-10-14.json).
The remaining 0.662-second 40K full-Decode gap selects unchanged Down input
ownership: four warp consumers share two K512 BF16 slices for M2/M3.
Gate and M4 remain unchanged. Added barriers may outweigh reduced global
input demand. Exact arithmetic/state admission returns immediately to d2 API;
3% slowdown or nonpositive 8K closes, useful long direction unlocks d3.
One implementation and one possible correctness repair, no parameter/profile
scan. Production and the full 1.5x–3x objective remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-down-input-ring-rejection-2026-10-14.json)
passes 30 complete projection output/guard cases, nine complete target prefixes,
fifteen state/cache transactions and four verification-failure recoveries.
At P8192/O256 d2, output and complete work match v75, but Decode falls from
14.973 to 14.727 token/s (-1.65%). Prefill is 17.248 seconds / 474.94 token/s,
TTFT 17.270 seconds and total 34.586 seconds. Verification adds 287.315ms;
complete Decode adds 285.280ms despite lower M3 registers.

The predeclared stop closes this architecture before 40K/d3 and full lifecycle
qualification. All candidate runtime/profile paths are removed, restoring v75.
No further input-ring tuning or profiler campaign follows. Production and the
remaining 0.662-second v75 40K budget stay unchanged; the full goal remains open.

## 2026-10-13 Gate/Up input lifetime — API composition complete

`WP-MTP-GATE-INPUT-RING-20261013` / `AC-MTP-GREEDY-v73` owns the
[bounded input producer](metadata/qwen36-27b-mtp-gate-input-ring-design-2026-10-13.json).
The remaining 0.992-second 40K full-Decode gap selects the unchanged Gate/Up
family. Replace complete shared inputs with two K512 slices, preserving all
four warp consumers and exact scalar arithmetic. M3/M4 shared storage falls
from 32 KiB to 8 KiB; extra barriers and asynchronous copies may offset that
resource benefit, so only composed API fitness selects it. This is input
lifetime replacement, not the rejected compressed-weight staging lineage.
One fixed structure, one possible correctness repair, no parameter/profiler
scan. Complete numerical/state admission returns immediately to d2 API;
3% slowdown or nonpositive 8K closes, useful long direction unlocks d3.
Production and the full objective remain unchanged.

The [first API return](metadata/qwen36-27b-mtp-gate-input-ring-direction-2026-10-13.json)
passes all eight d2 checks with 8K/40K Decode 14.954/11.440 token/s, but d3
8K falls from 14.008 to 13.862 token/s and stops before 40K. This complete
replacement is not retained. D2 has no M4 verification, while d3 mostly does.

`WP-MTP-GATE-INPUT-COMPOSITION-20261013` / `AC-MTP-GREEDY-v74` owns one
[bounded composition](metadata/qwen36-27b-mtp-gate-input-composition-design-2026-10-13.json):
keep the admitted M2/M3 ring and restore original M4 full-input publication.
These exact shape bodies already passed complete state/cache admission; the
combined binary repeats full projection guards, then returns immediately to
both d2/d3 API panels. No new tile, stage, launch or threshold variant. This is
a temporary dependency until that return, not a production or full-goal claim.

The [completed composition](metadata/qwen36-27b-mtp-gate-input-composition-direction-2026-10-13.json) passes all sixteen d2/d3 API and recovery
checks. GPU instruction words prove exact reuse of admitted M2/M3 and original
M4 bodies; complete synthetic outputs/guards also pass. Final d2 8K/40K Decode
is 14.973/11.442 token/s, saving 0.323/0.330 seconds of complete Decode versus v73.
At 40K, Prefill is 92.160 seconds / 434.03 token/s, TTFT 92.196 seconds and
external total 114.483 seconds. D3's original long-context performance is preserved.

Retain only this shape composition as a bounded development dependency through
the next complete MTP architecture API return or archival. No production switch,
statistical selection, new stall cause or further input-ring tuning is claimed.
The full 1.5x–3x objective remains open; the remaining 40K full-Decode gap is
approximately 0.662 seconds. Next work must address that whole-runner budget.

## 2026-10-13 draft weight-major matrix — API return complete

`WP-MTP-DRAFT-WEIGHT-MAJOR-20261013` / `AC-MTP-GREEDY-v72` owns the
[common private draft executor](metadata/qwen36-27b-mtp-draft-weight-major-design-2026-10-13.json).
The preceding matrix path padded M1 into 32 token rows. A native weight-major
BF16 executor uses64 weight rows,8/16/32 token-column buckets and one K128
shared double buffer. Every count uses the same per-output K16 order, so
initialization, drafting and cache reconciliation must still agree exactly.
The private draft numerical identity changes and must pass the unchanged
independent FP64 hidden/K/V0.02 bound; target state/output remain bitwise exact.
One implementation plus one possible correctness repair returns directly to
API. No tile/stage scan;3% slowdown or nonpositive8K stops, useful long direction
unlocks d3. This matrix budget alone cannot claim the full goal; production stays fixed.

The [completed direction](metadata/qwen36-27b-mtp-draft-weight-major-direction-2026-10-13.json) passes complete projection, independent
FP64 draft, full target-state/cache and all sixteen d2/d3 API/lifecycle checks.
D2 8K/40K Decode is 14.694/11.276 token/s, about 1.3%/1.2% above v64;
complete Decode saves 0.229/0.273 seconds with identical output and work.
At 40K, Prefill is 92.175 seconds / 433.96 token/s, TTFT 92.210 seconds and
external total 114.827 seconds. The record reports every d2/d3 phase together.

Retain the fixed executor as a bounded development composition through the
next complete MTP architecture API return or archival. This is one process per
maximum, not statistical architecture selection or production qualification.
The full 1.5x–3x target is still open: d2 reaches 1.595x/1.434x non-MTP at
8K/40K, with about 0.992 seconds still to remove from 40K complete Decode.
Next work must address that remaining composed budget. No matrix tile/stage
scan or repetition-only campaign follows; production is unchanged.

## 2026-10-13 retained MTP budget — complete diagnostic

`WP-MTP-V64-BUDGET-20261013` owns one
[unchanged-v64 API timeline](metadata/qwen36-27b-mtp-v64-budget-design-2026-10-13.json).
The previous family profile predates retained FP8 prefetch, probability
publication and confidence work. Reconcile the current P40000/O256 d2 verifier
budget with exact prior output and work before selecting another architecture.
One Nsight Systems capture, no kernel replay, NCU, runtime mutation or scan.
This diagnostic has no speedup or production authority; one analysis selects
or excludes a next implementation and closes the package.

The [completed timeline](metadata/qwen36-27b-mtp-v64-budget-2026-10-13.json)
authenticates identical request, output and complete work. In the declared
23.000-second diagnostic window, Gate/Up costs 5.666s, FP8 aggregate 5.346s,
QK 2.142s and PV 2.180s. Copy plus uncovered intervals total 0.507s, below the
remaining 1.264s full-Decode budget. These are profile costs, not updated API
baselines or stall attribution. No repeated profile/NCU follows.

## 2026-10-13 PV row warps — closed negative

`WP-MTP-PV-ROW-WARPS-20261013` / `AC-MTP-GREEDY-v71` owns the
[shared V / independent row consumers](metadata/qwen36-27b-mtp-pv-row-warps-design-2026-10-13.json).
The current PV budget selects one 32-dimension CTA with 6*M head warps and one
ordered accumulator per lane. Distinct speculative rows consume one shared V
ring concurrently. This differs from the closed 192-thread serial-M consumer.
Probability demand doubles relative to the 64-dimension incumbent, so fewer V
loads or static concurrency cannot select it. Complete numerical/state admission
returns directly to d2 API; 3% slowdown or nonpositive 8K closes it, useful long
direction unlocks d3. One structure/one possible correctness repair, no tile,
launch or profiler scan. Production and the full goal remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-pv-row-warps-rejection-2026-10-13.json)
passes 72 complete Attention score/probability/output/guard cases, eighteen
P513/P8192 prefixes, fifteen state/cache transactions and four failure recoveries.
All three kernels use 39 registers without spills. At P8192/O256 d2, output
and work remain exact but Decode falls from 14.503 to 14.332 token/s (-1.18%).
Prefill is 17.255 seconds / 474.75 token/s, TTFT 17.277 seconds and total
35.070 seconds. Verification adds 214.398ms; complete Decode adds 209.099ms.

The predeclared nonpositive 8K stop prevents 40K/d3 and full lifecycle
qualification. All new producer/consumer/profile paths are removed and v64
restored. P65 does not execute this PV route, and its small timing variation
is not attributed to the change. No new stall or bandwidth cause, carveout,
stage or launch scan follows. Production and the full goal remain unchanged.

## 2026-10-13 FP8 output ownership — closed negative

`WP-MTP-FP8-OUTPUT-PARTITION-20261013` / `AC-MTP-GREEDY-v70` owns the
[fixed output partition](metadata/qwen36-27b-mtp-fp8-output-partition-design-2026-10-13.json).
The remaining 40K full-Decode budget selects lower accumulator liveness in the
FP8 verifier. Two 256-thread groups share one CTA and split four independent
outputs, preserving every scalar chain and reduction. This doubles input
demand between groups but may admit more active warps for M3/M4. Compile
and exact numerical admission return immediately to the API. One fixed512-thread
structure, no register/launch scan; spills, >3% slowdown or nonpositive8K close
it. Useful long direction unlocks d3. Production and the full goal stay fixed.

The [completed rejection](metadata/qwen36-27b-mtp-fp8-output-partition-rejection-2026-10-13.json)
passes 72 complete projection outputs/guards, nine real-model prefixes, fifteen
state/cache transactions and four failure recoveries. M2/M3/M4 registers fall
to 40/52/60–61 without spills, but P65/O16 d2 Decode falls to 11.842 token/s
from 12.796 (-7.46%). Prefill is 0.892 seconds / 72.83 token/s, TTFT 0.911
seconds and external total 2.178 seconds; verification adds 94.061ms.

The 3% stop closes the fixed structure. The already-entered 8K request is
cancelled; 40K/d3 and full lifecycle qualification do not follow. All candidate
runtime/profile paths are removed and v64 restored. No measured stall cause,
actual occupancy gain or universal hardware limit is inferred; no output-count,
register or launch scan follows. Production and the full target stay unchanged.

## 2026-10-13 weight-major certificate — closed negative

`WP-MTP-WEIGHT-MAJOR-CERTIFICATE-20261013` / `AC-MTP-GREEDY-v69` owns the
[weight-major producer](metadata/qwen36-27b-mtp-weight-major-certificate-design-2026-10-13.json).
The prior certified Tensor Core producer wastes token padding on the 16-row
axis and reconstructs FP8 values in its inner loop. Move weights to 16 rows
and token inputs to 8 columns, halve MMA calls, and publish the existing exact
BF16 code lookup once per CTA. This is one coupled dataflow, with unchanged
gamma64, strict certificate and scalar repair; no tolerance or bound scan.
The original full-state/output contract remains mandatory. PTX's unspecified
rounding still prevents claiming universal scalar-FMA identity or production
proof from sampled tests. Compile/numerical/state admission returns immediately
to d2 API; 3% slowdown or nonpositive 8K closes it, useful long direction unlocks
d3. One implementation and one correctness repair; no profiler campaign.
Production and the full 1.5x–3x goal remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-weight-major-certificate-rejection-2026-10-13.json)
passes 72 complete FP8 outputs/guards, nine real-model prefixes, fifteen full
state/cache transactions and four failure recoveries. P65/O16 d2 preserves
output and work but Decode falls from 12.796 to 8.586 token/s (-32.91%).
Prefill is 0.892 seconds / 72.84 token/s, TTFT 0.911 seconds and total 2.658
seconds. Verification adds 574.962ms and complete Decode adds 574.886ms.

The predeclared stop loss closes this composition. The already-entered 8K
request is cancelled; no completed 8K/40K/d3 or full lifecycle result follows.
All candidate runtime/profile changes are removed and v64 restored. Resource
improvements and sampled numerical passes do not qualify the proposed interval
bound for production, and no gamma/tile/threshold or profiler scan follows.
The full 1.5x–3x objective remains open; production is unchanged.

## 2026-10-13 FP8 activation publication — closed negative

`WP-MTP-FP8-ACTIVATION-STAGE-20261013` / `AC-MTP-GREEDY-v68` owns the
[complete-input publication](metadata/qwen36-27b-mtp-fp8-activation-stage-design-2026-10-13.json).
The remaining 40K full-Decode budget and historical 5.869-second FP8 family
select this fixed operand-delivery change. Publish unchanged BF16 input rows
with 128-bit copies before the existing CTA barrier; consume them from shared
memory through the unchanged scalar FMA/reduction trees. M4 stages three rows
and retains its last global row. No extra intra-CTA reuse or measured stall
cause is assumed. Fixed shared carveout, no spill, complete numerical and
state admission precede immediate API selection. One implementation and one
possible correctness repair; no parameter/profiler scan. Nonpositive8K or
>3% slowdown closes it; useful long direction unlocks d3. Production stays fixed.

The [completed rejection](metadata/qwen36-27b-mtp-fp8-activation-stage-rejection-2026-10-13.json)
passes 72 full FP8 output/guard cases, nine complete prefixes, fifteen state/cache
transactions and four failure recoveries. P65/O16 d2 preserves output and work
but Decode falls to 12.037 token/s from 12.796 (-5.93%). Prefill is 0.890 seconds /
73.02 token/s, TTFT 0.909 seconds and total 2.155 seconds. Verification adds
73.047ms; complete Decode adds 73.941ms.

The predeclared 3% stop loss closes this composition. The already-entered 8K
request is interrupted and has no completed timing/output claim; 40K/d3 and
full lifecycle qualification do not follow. All candidate paths are removed
and v64 program sections restored exactly. There is no new profiler attribution
for the added shared traffic or changed cache partition, and no subsequent
carveout/launch scan. The full 1.5x–3x goal and production remain unchanged.

## 2026-10-13 lookahead cost — diagnosis archived

`WP-MTP-LOOKAHEAD-COST-20261013` / `AC-MTP-GREEDY-v67` owns the
[round-cost composition](metadata/qwen36-27b-mtp-lookahead-cost-design-2026-10-13.json).
The unchanged 1.5x–3x target selects removal of wasted verification work.
D3 reduces 40K rounds but adds rows and loses Decode; aggregate acceptance
cannot choose a useful next proposal. First retain already-computed confidence
and phase costs in a bounded host trace, without additional GPU operations.
One existing API panel for each configured maximum must preserve outputs and
all work counts. The diagnostic then either selects one cost-based policy
for immediate full-state/API composition or closes; no threshold scan or
further trace-only campaign. Production and target arithmetic stay fixed.

The [completed diagnosis](metadata/qwen36-27b-mtp-lookahead-cost-closeout-2026-10-13.json)
preserves all sixteen d2/d3 API outputs and work counts. At 40K, observed
mean verification costs for two/three/four target rows are approximately
155/196/245ms. These shape-conditioned means are diagnostic, not matched
counterfactual timings. More accepted proposals do not by themselves overcome
this cost. A 0.987-confidence short-prompt proposal is rejected while a
0.311-confidence proposal is accepted: draft softmax is not target acceptance
probability. Later confidence and ungenerated counterfactuals remain unobserved.

No calibrated cost policy is justified by this single prompt's conditional
samples. The package closes through its explicit archival branch, without
fitting or scanning request-specific thresholds. This does not reject all
future adaptive scheduling or establish a hardware bound. Temporary telemetry
is removed and v64 is restored; no further trace-only campaign remains.
The next implementation must address the remaining whole-verifier budget or
supply a materially new acceptance mechanism; the 1.5x–3x goal is unchanged.

## 2026-10-13 private draft matrix head — closed without retention

`WP-MTP-DRAFT-MATRIX-HEAD-20261013` / `AC-MTP-GREEDY-v66` owns the
[packed draft vocabulary composition](metadata/qwen36-27b-mtp-draft-matrix-head-design-2026-10-13.json).
Extend the private BF16 draft matrix executor to the full NVFP4 vocabulary
using existing Marlin preparation and one fixed M1 mapping. No token shortlist.
The draft reduction changes; independent FP64 full-logit relative L2 must stay
at or below 0.02 on the fixed nine-row real capture panel before API use.
Every target state, full logit and committed token retains the scalar contract.
Startup-only sidecars preserve the 8-GiB reserve. One implementation, one possible
correctness repair, no profiler or parameter scan. Full-state admission returns
immediately to d2 API; nonpositive 8K or >3% slowdown closes it, useful long
results unlock d3. The full 1.5x–3x goal remains active; production is unchanged.

The [completed direction](metadata/qwen36-27b-mtp-draft-matrix-head-rejection-2026-10-13.json)
passes nine full-vocabulary independent FP64 comparisons (maximum relative L2
0.001666), complete P65 state/cache transactions and all eight d2 API/lifecycle
checks. D2 Decode is 12.855/14.563/11.193 token/s at P65/8K/40K, only
0.45%/0.42%/0.47% above retained v64. At 40K, Prefill is 93.046 seconds /
429.89 token/s, TTFT 93.082 seconds and external total 115.865 seconds.
Complete Decode saves 106.380ms, including 61.204ms less draft time, while
external total adds 113.111ms. Work and committed outputs are unchanged.

This single-process direction does not establish a useful robust gain for
715,293,184 additional retained bytes. It does not prove a statistical
regression or a hardware ceiling. Unchanged Prefill/verification timing
variation is not causally attributed to the head. The useful-long-direction
condition for d3 is not established; no d3 run or further tuning follows.
All candidate runtime/profile changes are removed and v64 is restored.
The full 1.5x–3x objective remains open; production is unchanged.

## 2026-10-13 exact QK pair fusion — closed without retention

`WP-MTP-QK-PAIR-FMA-20261013` / `AC-MTP-GREEDY-v65` owns one
[guarded pair-fusion composition](metadata/qwen36-27b-mtp-qk-pair-fma-design-2026-10-13.json).
The current 40K d2 full-Decode gap is 1.264 seconds. Exact nonzero BF16 products
can fuse into their original pair sums without changing a higher reduction.
Runtime range guards and the original exceptional path preserve arbitrary
inputs. This removes arithmetic rather than changing Attention tile sizes.
One implementation, at most one correctness repair, no profiler or parameter
scan. Directed numerical and full-prefix checks precede immediate d2 API;
>3% slowdown or nonpositive 8K closes it. Useful long direction unlocks d3.
Production stays fixed; the full 1.5x–3x goal remains active.

The [completed rejection](metadata/qwen36-27b-mtp-qk-pair-fma-rejection-2026-10-13.json) passes 62,914,560 native raw pair cases,
72 complete score/probability/output/guard cases, eighteen P513/P8192 prefixes,
fifteen full-state transactions and four failure recoveries. P8192/O256 d2
preserves output and work but Decode is 14.496 token/s versus v64 at 14.503
(-0.05%). Prefill is 17.241 seconds / 475.15 token/s, TTFT 17.263 seconds and
total 34.855 seconds. Unchanged Prefill code does not justify attributing its
50.571ms lower observation to QK; complete Decode adds 8.762ms.

This is neutral direction, not a statistically established regression. The
predeclared nonpositive 8K stop prevents 40K/d3 and the full lifecycle panel.
All candidate runtime/profile paths are removed and v64 program sections restored
exactly. No further guard, launch, tile or profiler scan follows this package.
Production and the full 1.5x–3x goal remain unchanged.

## 2026-10-10 retained MTP Attention counters — closed inconclusive

`WP-MTP-V64-ATTENTION-COUNTERS-20261010` owns the
[fixed two-launch diagnostic](metadata/qwen36-27b-mtp-v64-attention-counters-design-2026-10-10.json).
The retained v64 40K d2 complete-Decode gap is 1.264 seconds. Historical QK/PV
GPU intervals total 4.532 seconds but do not identify the current limiting
instruction or movement. Capture the first QK and PV launches in one unchanged
v64 P40000/O256 API request. Authenticate output and complete work against the
unprofiled request. No profiler timing is a performance selection result.
One capture, two launches, no sweep or runtime mutation; choose one causal
implementation hypothesis or close as inconclusive. Production stays fixed.

The [2026-10-13 recovery closeout](metadata/qwen36-27b-mtp-v64-attention-counters-closeout-2026-10-13.json)
finds an interrupted capture after NCU reported allocation error 12 and disk
backup during kernel replay. No report or complete API receipt survived; no
counter, speed, or reboot-cause claim follows. Owned processes are absent and
GPU handles empty; the stale key and tool pipes are removed. This fixed capture
is closed without repetition. Future profiling must bound replay memory and
stop on allocation errors. Retained v64 and production are unchanged.

## 2026-10-10 MTP verification frontier — closed without retention

`WP-MTP-FRONTIER-20261010` / `AC-MTP-GREEDY-v64` selects the
[verification-frontier composition](metadata/qwen36-27b-mtp-frontier-design-2026-10-10.json).
The remaining 40K d2 Decode gap is 1.264 seconds. Keep threshold 0.4 and maxima 2/3,
but exclude the low-confidence trigger itself from target verification, including
at the final configured draft position. Empty submitted prefixes remain valid
seed-only target steps; produced seed KV is reusable independently of submission.
Separate produced/submitted/pruned work so no draft cost disappears from metrics.
One composition, at most one correctness repair; complete host/P65 state checks
return immediately to d2 API. >3% slowdown or nonpositive 8K closes it; useful long
direction unlocks d3. No threshold/length/launch scan or production change.

The [completed rejection](metadata/qwen36-27b-mtp-frontier-rejection-2026-10-10.json) passes eighteen confidence CPU/device cases,
nine complete P65 prefixes, fifteen state/logit transactions and four failure
recoveries, including one real non-tail empty submission. P65/O16 d2 output is
unchanged but Decode falls from 12.796 to 12.100 token/s (-5.44%). Prefill is
0.889 seconds / 73.14 token/s, TTFT 0.907 seconds and total 2.147 seconds.
The >3% stop cancels the already-entered 8K request; no completed 8K/40K/d3 or
full lifecycle claim follows.

Verified rows fall from 19 to 18, but acceptance falls from 8 to 7, rounds rise from 7 to 8 and actual
drafting rises from 12 to 13. Dropping a low-confidence proposal loses useful work and
adds another round here; row count alone did not predict API value. All new
runtime/profile paths are removed and v64 is restored exactly. No threshold
scan, further native variant or qualification follows this fixed package.
Production and the full 1.5x–3x goal remain unchanged; v64's 40K gap is still 1.264 seconds.

## 2026-10-10 MTP confidence lookahead — direction complete, retained dependency

`WP-MTP-CONFIDENCE-20261010` / `AC-MTP-GREEDY-v63` selects the
[bounded confidence composition](metadata/qwen36-27b-mtp-confidence-design-2026-10-10.json).
The 40K complete Decode gap is 1.767 seconds. Use current draft logits to
stop further proposals below fixed softmax confidence 0.4, keeping configured
maxima 2/3 and the triggering proposal. Actual count selects exact target
verification and full prefix commitment. This differs from rejected prior-round
acceptance backoff. One composition, at most one correctness repair, no threshold
scan; host/device state admission returns immediately to d2 API. A >3% slowdown
or nonpositive 8K direction closes it before further long qualification.
Useful long direction unlocks d3. Production and the full goal remain fixed.

The [completed direction](metadata/qwen36-27b-mtp-confidence-direction-2026-10-10.json) passes eighteen confidence checks, complete
P65 state/logit and dynamic transaction/cancellation/recovery admission, and all
sixteen API/lifecycle checks. D2 8K/40K Decode is 14.503/11.141 token/s;
Prefill is 17.292/92.827 seconds (473.76/430.91 token/s), TTFT
17.314/92.862 seconds and total 34.897/115.751 seconds.
Both maxima improve long Decode with fewer actual rows and unchanged outputs.
D2 P65 is slightly slower; no blanket all-context improvement is claimed.

Retain this one-process direction as a bounded development composition through
the next complete MTP API return or archival, without threshold/length/launch
scanning. This closes the fixed package. The full 1.5x–3x goal remains active;
40K d2 still needs 1.264 seconds removed from full Decode. Production is
unchanged; subsequent work must address that remaining whole-path budget.

## 2026-10-10 MTP uniform short-chain dispatch — closed at host feasibility

`WP-MTP-UNIFORM-CHAIN-20261010` / `AC-MTP-GREEDY-v62` screens one
[uniform dispatch architecture](metadata/qwen36-27b-mtp-uniform-chain-design-2026-10-10.json).
The rejected direct-chain executor interleaves per-element scalar repair with
MMA outputs; that is a source fact, not measured stall attribution. Reuse the
unchanged nine-bit predicate across complete N16/two-chain/M3 native units.
At least 50% complete-unit coverage in authenticated historical real captures
is required before one native implementation and immediate API composition.
Lower coverage closes the fixed design without GPU work. No guard, grouping,
threshold or launch scan. Production, exact state and the full goal stay fixed.

The [completed host screen](metadata/qwen36-27b-mtp-uniform-chain-feasibility-2026-10-10.json)
authenticates all 28 historical P65 M3 FP8 captures. The unchanged cheap guard
admits 2,448,199/2,752,512 individual roots (88.94%), but only 3,130/28,672 complete
96-root native units (10.92%). The 50% complete-unit floor fails; no native
implementation, GPU process or new API timing is opened. The earlier 73.09%
complete-unit observation used the exact-product lattice predicate, not this
cheap runtime guard. Neither rate establishes executor speed.

This closes the fixed uniform-dispatch proposal without a guard/group/threshold
scan. It does not relax the numerical contract, prove a hardware limit or
supersede retained v62 performance. Production and the full goal stay fixed.

## 2026-10-10 MTP exact draft vocabulary — closed without retention

`WP-MTP-DRAFT-HEAD-20261010` / `AC-MTP-GREEDY-v61` composes the
[exact draft-head executor](metadata/qwen36-27b-mtp-draft-head-design-2026-10-10.json)
from admitted paired decoding and live-ancestor reductions. Draft still uses
the old scalar head although target verification already uses this arithmetic.
The historical draft-head budget is 0.914 seconds; remaining 40K full Decode
needs 1.767 seconds, so this component alone cannot claim the whole target.
Six complete vocabulary/guard checks and P65 full-state transactions precede
immediate d2 API. A >3% slowdown stops; useful long direction unlocks d3.
One composition, at most one correctness repair, no parameter/profile scan.
Production, full vocabulary, proposals, state and the full goal stay fixed.

The [completed rejection](metadata/qwen36-27b-mtp-draft-head-rejection-2026-10-10.json) passes six full-vocabulary output/guard
cases, nine complete P65 prefixes and eleven transactions including full state,
logits, draft cache and failure recovery. P8192/O256 d2 retains exact output/work
but Decode is 14.190 token/s versus retained v62 at 14.223 (-0.23%). Prefill
is 17.321 seconds / 472.94 token/s, TTFT 17.344 seconds and total 35.314 seconds.
Draft adds 44.81 ms, complete Decode 41.73 ms and total 62.62 ms. The subsequent
40K request entered before the stop arrived and was cancelled; it supplies no
completed result. D3 API and remaining lifecycle are not run.

The nonnegative 8K screen closes this version. All new runtime/profile paths
are removed and retained v62 is restored. This composed executor transfer did
not return API value; decoder, reduction and launch geometry are not separately
attributed. No further launch/tile/decoder scan or profile follows. Production
and the full unmet 1.5x–3x target stay fixed.

## 2026-10-10 MTP probability publication — direction complete, retained dependency

`WP-MTP-PROBABILITY-PUBLICATION-20261010` / `AC-MTP-GREEDY-v60` owns one
[aligned probability publication](metadata/qwen36-27b-mtp-probability-publication-design-2026-10-10.json)
composition across QK, exact softmax and ordered PV. Four-float aligned head
strides permit 16-byte probability copies; padding never enters arithmetic.
This retains the V producer and all original numerical operations. Complete
probability/output and P513/P8192 state admission precede immediate d2 API;
>3% slowdown stops, useful long direction unlocks d3, otherwise remove.
No stride, tile, launch or profiler scan. Production and full goal stay fixed.

The [completed direction](metadata/qwen36-27b-mtp-probability-publication-direction-2026-10-10.json) passes complete numerical admission and
all sixteen d2/d3 API/lifecycle checks. D2 8K/40K Decode is
14.223/10.902 token/s, versus v52 at 14.143/10.672. Corresponding Prefill is
17.300/93.570 seconds (473.51/427.49 token/s), TTFT
17.323/93.605 seconds and total 35.252/116.997 seconds. D2 40K total is
0.228 seconds longer despite lower Decode; d3 similarly adds 0.357 seconds.
No 40K total-latency win is claimed.
The unchanged Prefill code does not explain or erase its observed variation.

Both policies preserve output/work and improve the two long Decode intervals.
This one-process direction is retained only as a bounded development dependency
through the next complete verifier API return or archival. No further stride,
tile, launch, profile or qualification-only scan belongs to this package.
Production stays fixed. The full 1.5x–3x target is still active, with roughly
1.77 seconds remaining in the 40K d2 complete-Decode budget; the next composition
must remove that actual cost and report all four API metrics.

## 2026-10-10 MTP fused certificate — closed without retention

`WP-MTP-FUSED-CERTIFICATE-20261010` / `AC-MTP-GREEDY-v59` owns one
[CTA-resident producer/certificate composition](metadata/qwen36-27b-mtp-fused-certificate-design-2026-10-10.json).
The previous API rejection remains frozen. Replace its external K128 partials,
finish and certificate stages with one shared-memory lifetime, preserving both
balanced trees and exact original repair. This removes global intermediates
and two launch boundaries; it does not change gamma or assume a speedup.
Full projection/P65 admission precedes immediate d2 API; >3% slowdown stops,
useful direction unlocks d3, otherwise remove. No local timing, bound, tile,
repair-layout or launch scan follows. Production and the full goal stay fixed.

The [completed rejection](metadata/qwen36-27b-mtp-fused-certificate-rejection-2026-10-10.json)
passes 72 full projection/guard cases, nine complete P65 prefixes and eleven
transactions including state/logits, draft cache and failure recovery.
P65/O16 d2 preserves output and work but Decode is 7.465 token/s versus
retained v52 at 12.828 (-41.80%). Prefill is 0.888 seconds / 73.21 token/s,
TTFT 0.906 seconds and total 2.916 seconds. Verification adds 839.51 ms and
complete Decode adds 839.95 ms. The declared 3% stop removes this composition;
long contexts, d3 API and remaining lifecycle checks are not run.

All new runtime/profile paths are removed and retained v52 restored. Removing
external intermediate publications did not yield upward API value; this is
not an isolated bandwidth/occupancy attribution or hardware-limit claim.
No further bound, tile, repair-layout, launch scan or profile follows this
closed version. Production and the full unmet 1.5x–3x target remain unchanged.

## 2026-10-10 MTP direct-BF16 certificate — closed without retention

`WP-MTP-BF16-CERTIFICATE-20261010` / `AC-MTP-GREEDY-v58` completes the
[runtime composition](metadata/qwen36-27b-mtp-bf16-certificate-api-design-2026-10-10.json).
It reopened the earlier host-only 25% repair screen through actual BF16 MMA,
outward-rounded native norms/certificates and original scalar element repair.
Construction owned norms for 208 FP8 views and fixed 16-MiB scratch; no
request-time allocation or numerical waiver was introduced.
The [native centers](metadata/qwen36-27b-mtp-bf16-certificate-native-2026-10-10.json)
and [native pipeline](metadata/qwen36-27b-mtp-bf16-certificate-pipeline-2026-10-10.json)
passed their bounded checks before the completed API return below.

The [completed API rejection](metadata/qwen36-27b-mtp-bf16-certificate-api-rejection-2026-10-10.json)
passes 72 complete projection output/guard cases, nine full P65 prefixes and
eleven transactions including state/logits, draft cache and failure recovery.
P65/O16 d2 preserves output and work but Decode falls from 12.828 to
7.597 token/s (-40.78%). Prefill is 0.889 seconds / 73.10 token/s,
TTFT 0.908 seconds and external total 2.882 seconds. Verification adds
804.86 ms and complete Decode adds 805.17 ms. Startup free memory is
9,788,293,120 bytes, above the unchanged 8-GiB floor.

The declared 3% stop removes the entire norm-owner/certificate/repair runtime
and its profile. Retained v52 is restored; long-context and d3 API checks are
not run. The actual BF16-center and native-certificate prerequisites are now
archived after their mandatory API return, not pending integration. The full
composition loses; no isolated component or hardware-limit claim follows.
No bound, repair-layout, launch scan or additional profile follows this closed
version. Production and the unmet 1.5x–3x objective remain unchanged.

## 2026-10-10 MTP FP8 normal consumer — closed without retention

`WP-MTP-FP8-NORMAL-CONSUMER-20261010` / `AC-MTP-GREEDY-v57` implements
one [exact warp-uniform consumer](metadata/qwen36-27b-mtp-fp8-normal-consumer-design-2026-10-10.json)
following the passed host screen. Every lane checks all sixteen weight codes;
only an all-normal warp bypasses the incumbent lookup. Preserve codebook
swizzle, special values and all scalar arithmetic. Full-output and complete
P65 prefix admission precede immediate d2 API; >3% slowdown stops, useful
direction unlocks d3, otherwise remove. No converter or launch scan follows.
Production and the complete 1.5x–3x goal remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-fp8-normal-consumer-rejection-2026-10-10.json)
passes 72 complete projection cases, nine full P65 prefixes and eleven
transactions, including state/logits, cache and failure recovery. P65/O16 d2
preserves output and work but Decode falls from 12.828 to 10.811 token/s
(-15.73%). Prefill is 0.888 seconds / 73.19 token/s, TTFT 0.907 seconds and
total 2.294 seconds. Verification adds 217.83 ms and full Decode adds 218.19 ms.
The declared 3% stop closes this composition without long-context or d3 API
qualification. All new runtime/profile paths are removed and retained v52
restored. High sampled ordinary-code coverage did not establish consumer
speedup; individual predicate/branch/conversion costs are not isolated.
No converter or launch scan follows. Production and the unmet 1.5x–3x goal
remain unchanged.

## 2026-10-10 MTP FP8 ordinary-code groups — host screen passed, consumer closed

`WP-MTP-FP8-NORMAL-GROUP-20261010` screens one
[exact codebook-bypass opportunity](metadata/qwen36-27b-mtp-fp8-normal-group-design-2026-10-10.json).
Authenticate real captured FP8 weights and count warp-uniform ordinary finite
codes. Sparse sampled output rows provide only single-output group coverage,
not the actual four-output consumer coverage. Below 50% closes this host
screen without GPU work; a pass requires exact consumer/state admission and
immediate full API composition. No code/shape scan or performance claim follows
from this host statistic. The full objective and production remain unchanged.

The [completed host screen](metadata/qwen36-27b-mtp-fp8-normal-group-feasibility-2026-10-10.json)
authenticates 28 projections and 37,888 sampled single-output groups. Of these,
33,597 (88.67%) contain only normal finite codes; all 238 ordinary-code
integer conversions are exact. This passes the predeclared 50% screen, but
says nothing measured about four-output coverage, branch cost or API speed.
The completed consumer above supplies the bounded API return. Its design required
one warp-uniform exact consumer with incumbent fallback, exhaustive
full-output admission, complete target-prefix state/logits and immediate
P65/8K/40K d2/d3 API return. Preserve sidecar xor decoding, signed zeros,
subnormals and NaNs. One composition and at most one correctness repair;
no conversion/launch scan. Production and current performance are unchanged.

## 2026-10-10 MTP QK position parallelism — closed without retention

`WP-MTP-QK-POSITION-PARALLEL-20261010` / `AC-MTP-GREEDY-v56` owns one
[spatial QK composition](metadata/qwen36-27b-mtp-qk-position-parallel-design-2026-10-10.json).
Four independent positions per speculative row halve the serial position loop,
retaining exact score arithmetic and sharing each K record across all rows.
This is distinct from the rejected bulk-copy scheme's unchanged two-position
consumer. Full P513/P8192 state/logit admission precedes immediate d2 API;
>3% slowdown stops, useful direction unlocks d3, otherwise remove. One fixed
composition, no launch/stage scan or profile. Production and the full
1.5x–3x goal remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-qk-position-parallel-rejection-2026-10-10.json)
passes eighteen complete P513/P8192 prefixes and all eleven P513 transactions,
including full state/logits, draft cache and failure recovery. At 8K d2,
Prefill is 17.295 seconds / 473.67 token/s, TTFT 17.317 seconds,
Decode 14.120 token/s and total 35.376 seconds. Output and work match v52,
but verification adds 22.60 ms and full Decode adds 29.15 ms; the -0.16%
direction misses the declared nonnegative 8K screen. Stop during post-8K wait,
without 40K, d3 API or remaining lifecycle qualification. This is not a
noise-qualified regression or a conclusion about unmeasured contexts.
All new runtime/profile paths are removed and retained v52 restored.
Production and the unmet 1.5x–3x objective remain unchanged; no position-count
or launch scan follows.

## 2026-10-10 MTP acceptance backoff — closed without retention

`WP-MTP-ACCEPTANCE-BACKOFF-20261010` / `AC-MTP-GREEDY-v55` owns one
[request-local draft-budget composition](metadata/qwen36-27b-mtp-acceptance-backoff-design-2026-10-10.json).
Keep configured maxima 2/3; after rejection reduce the next budget by one to
minimum one, and after full acceptance increase by one up to the maximum.
Existing exact operators and commit/observer semantics stay fixed. Host replay
and full P65 mixed-budget state/cache/recovery checks precede immediate d2 API;
>3% slowdown stops, useful direction unlocks d3, otherwise remove. Report changed
work and exact output separately. No heuristic/length/threshold scan or profile
follows. Production and the 1.5x–3x objective remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-acceptance-backoff-rejection-2026-10-10.json)
passes all nine P65 complete prefixes and eleven mixed-budget transactions,
including cache and failure recovery. P65/O16 d2 preserves output IDs but
Decode falls from 12.828 to 11.971 token/s (-6.68%), triggering the declared
3% stop. Prefill is 0.890 seconds / 73.07 token/s, TTFT 0.908 seconds and
external total 2.161 seconds. Proposals remain twelve, but accepted proposals
fall from eight to seven, rounds rise from seven to eight and verified rows
from nineteen to twenty. Acceptance history did not reduce work here.
All new controller/service/report/test paths are removed and retained v52
restored. Long-context and d3 API checks are not run; no heuristic scan follows.
This short-request rejection makes no claim about unmeasured long contexts.
Production and the unmet 1.5x–3x objective remain unchanged.

## 2026-10-10 MTP static-gate API reconciliation — complete without retention

`WP-MTP-STATIC-GATE-API-20261010` / `AC-MTP-GREEDY-v54` [reopens the exact frozen v56 binary](metadata/qwen36-27b-mtp-static-gate-api-design-2026-10-10.json)
without a source/parameter mutation. SASS shows no backward branches and only
four/six fixed spill accesses per thread in two M3 kernels. The earlier local
no-spill screen remains recorded; it did not establish API fitness. Complete
projection/P65 checks precede one d2 API return; >3% slowdown stops, useful
direction unlocks d3, otherwise close. Current source/production stays unchanged
until useful API evidence. No stage/register scan or profile follows. This
corrects decision scope without weakening numerical or device-safety gates.

The [completed API reconciliation](metadata/qwen36-27b-mtp-static-gate-api-rejection-2026-10-10.json)
passes 36 complete projection cases, all P65 prefix/cache/transaction checks
and eight d2 API checks, preserving output and logical/physical work. At 8K,
Prefill is 17.303 seconds / 473.45 token/s, TTFT 17.325 seconds,
Decode 14.269 token/s and total 35.196 seconds. At 40K these are
92.757 seconds / 431.23 token/s, 92.793 seconds, 10.758 token/s and
116.497 seconds. Decode directions of +0.89%/+0.81% correct any inference
that spill presence proved a slowdown, but do not meet this package's 1%
40K floor or provide noise qualification. Candidate source is not restored;
d3 is not run and no parameter scan follows. Retained v52, production and
the complete unmet 1.5x–3x target remain unchanged.

## 2026-10-10 MTP FP8 static stages — closed at static gate

`WP-MTP-FP8-STATIC-STAGES-20261010` / `AC-MTP-GREEDY-v53` owns one
[fixed-stage FP8 composition](metadata/qwen36-27b-mtp-fp8-static-stages-design-2026-10-10.json).
Specialize admitted K5120/K6144 into five/six ordered K1024 stages, preserving
native hints, original demand loads and scalar arithmetic. Native no-loop/
no-spill inspection precedes complete projection/P65 admission and immediate
d2 API. >3% slowdown stops; useful direction unlocks d3, otherwise remove.
Code size and scheduling costs belong to whole API fitness. No unroll-factor,
register scan or profile follows. Useful retention expires at the next full
verifier API return or archival. Production and the 1.5x–3x goal stay fixed.

The [completed static rejection](metadata/qwen36-27b-mtp-fp8-static-stages-rejection-2026-10-10.json)
builds both executables and the synthetic harness but fails the declared
no-spill gate. The actual M3 canonical K5120/K6144 kernels use 8/16-byte stack
frames and four/six local-memory instructions. This is a static resource
rejection, not an observed API slowdown. All new source/profile paths are
removed and retained v52 rebuilt with identical program sections. No GPU
numerical/API run, profile or unroll/register scan follows. Current performance,
production and the unmet 1.5x–3x objective remain unchanged.

## 2026-10-10 MTP element certificate — closed at host feasibility

`WP-MTP-ELEMENT-CERTIFICATE-20261010` checks one [element-granular repair design](metadata/qwen36-27b-mtp-element-certificate-design-2026-10-10.json).
A direct-BF16 fast path would remove the old FP16 conversion/residual-MMAs;
a cheap norm envelope would replace absolute-product MMAs, with original scalar
repair only for uncertified elements. Before any GPU implementation, reuse pinned
real FP8 inputs and screen one conservative envelope at <=25% sampled repair.
This is a host feasibility surrogate, not runtime certification or timing.
Failure closes the direction without a bound/tile scan; success requires a
concrete runtime proof and immediate numerical/API composition. The full
1.5x–3x target and production default stay unchanged.

The [completed feasibility screen](metadata/qwen36-27b-mtp-element-certificate-rejection-2026-10-10.json)
authenticates 28 real FP8 projections and 2,688 sampled M3 elements. The fixed
norm envelope certifies 1,852 elements around the FP64 central-dot surrogate;
31.1% require repair, exceeding the predeclared 25% screen. Close this design
without runtime changes, a GPU test or bound/tile scanning. The surrogate is
not an actual MMA evaluation or an upper bound on runtime certification rate;
this does not establish a hardware limit or target impossibility. Current v52
API observations and the unmet 1.5x–3x goal remain unchanged.

## 2026-10-10 MTP QK bulk publication — closed without retention

`WP-MTP-QK-BULK-PUBLICATION-20261010` / `AC-MTP-GREEDY-v52` owns one
[immutable K-tile composition](metadata/qwen36-27b-mtp-qk-bulk-publication-design-2026-10-10.json).
Publish 32 K positions once and consume sixteen original two-position steps
before retiring the tile. This replaces the fine-grained copy ring, preserving
all scalar arithmetic, causal domains and query/output ownership. The historical
2.215-second QK family cannot alone close the remaining 2.27-second Decode gap;
this is a bounded composition prerequisite. Full P513/P8192 admission returns
immediately to d2 API; >3% slowdown stops, useful direction unlocks d3, otherwise
remove. No tile/pipeline scan or profile follows. Useful retention expires at
the next complete verifier API return or archival; production stays unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-qk-bulk-publication-rejection-2026-10-10.json) preserves all eighteen complete P513/P8192
prefix state/logit comparisons and P513 transactions/recovery. D2 8K observes
Prefill 17.297 seconds / 473.62 token/s, TTFT 17.319 seconds,
Decode 14.112 token/s and total 35.388 seconds.
The declared useful-direction screen is not met. All new QK publication and
profile paths are removed and v52 rebuilt. The driver stops in its post-8K
idle interval; 40K, d3 and remaining API lifecycle checks are not run. No bulk tile,
copy-stage or register scan follows; retained v52 observations remain current.
These single-process observations do not qualify a speedup or hardware cause.
The complete 1.5x–3x goal remains unmet.

## 2026-10-10 MTP NVFP4 native prefetch — closed without retention

`WP-MTP-NV-NATIVE-PREFETCH-20261010` / `AC-MTP-GREEDY-v51` owns one
[native operand-prefetch composition](metadata/qwen36-27b-mtp-nv-native-prefetch-design-2026-10-10.json)
against the remaining 2.27-second 40K Decode gap. Gate/Up and Down issue
cache hints for one valid next packed weight/scale and unstaged activation
record. Demand reads and arithmetic remain original. Native early hints and
no spills precede complete projection/P65 admission and immediate d2 API.
>3% slowdown stops; useful direction unlocks d3, otherwise remove. No profile,
distance/cache-level or register scan follows. Useful retention expires at the
next complete verifier API return or archival; production and the goal stay fixed.

The [completed rejection](metadata/qwen36-27b-mtp-nv-native-prefetch-rejection-2026-10-10.json) passes native instruction/no-spill gates, 45 complete projection cases,
all P65 prefix state/logit/cache comparisons and transactions/recovery. D2 8K observes
Prefill 17.299 seconds / 473.56 token/s, TTFT 17.321 seconds,
Decode 14.061 token/s and total 35.457 seconds.
The declared useful-direction screen is not met. All new NV prefetch and
profile paths are removed and v52 rebuilt. The driver stops in its post-8K
idle interval; 40K, d3 and remaining API lifecycle checks are not run. No prefetch-distance,
cache-level or register scan follows; retained v52 observations remain current.
These single-process observations do not qualify a speedup or hardware cause.
The complete 1.5x–3x goal remains unmet.

## 2026-10-10 MTP Attention rendezvous — closed without retention

`WP-MTP-ATTENTION-RENDEZVOUS-20261010` / `AC-MTP-GREEDY-v50` owns one
[QK/PV synchronization composition](metadata/qwen36-27b-mtp-attention-rendezvous-design-2026-10-10.json)
against the remaining 2.27-second 40K Decode gap. One retained post-wait CTA
barrier retires current readers and publishes next operands; the redundant
pre-wait barrier is removed. Ring ownership and scalar arithmetic are unchanged.
Full P513/P8192 state/logit admission returns immediately to d2 API; >3%
slowdown stops, neutral removes, useful direction unlocks d3. No profile or
parameter scan follows. Useful retention expires at the next full verifier API
return or archival. Production and the complete 1.5x–3x goal remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-attention-rendezvous-rejection-2026-10-10.json) preserves all eighteen complete P513/P8192
prefix state/logit comparisons and P513 transactions/recovery. D2 8K observes
Prefill 17.319 seconds / 473.01 token/s, TTFT 17.341 seconds,
Decode 14.139 token/s and total 35.376 seconds.
The declared useful-direction screen is not met. All new synchronization and
profile paths are removed and v52 rebuilt. The following 40K request is
interrupted before completion; d3 and remaining API lifecycle checks are not
run. No further barrier,
ring or tile scan follows; retained v52 observations remain current.
These single-process observations do not qualify a speedup or hardware cause.
The complete 1.5x–3x goal remains unmet.

## 2026-10-10 MTP native FP8 cache prefetch — API return complete

`WP-MTP-NATIVE-PREFETCH-20261010` / `AC-MTP-GREEDY-v49` owns one
[native cache-prefetch composition](metadata/qwen36-27b-mtp-native-prefetch-design-2026-10-10.json)
against the remaining 2.67-second 40K Decode gap. An explicit cache hint for
the next weight/activation records avoids v51's folded operand-register
lifetime. Original demand loads, arithmetic and state remain authoritative.
The exact compiled server must show early native prefetch and no spills before
full numerical admission and immediate d2 API. >3% slowdown stops, neutral
removes, useful direction unlocks d3. No distance/cache-level scan or profile
follows. Production and the 1.5x–3x target remain unchanged.

The [completed direction](metadata/qwen36-27b-mtp-native-prefetch-direction-2026-10-10.json)
passes exact compiled-instruction/no-spill gates, 36 complete output/guard cases,
all P65 numerical/cache/transaction checks and sixteen d2/d3 API checks.
Both policies reduce complete Decode and verification at 8K/40K with identical
output/work. D2 reaches 14.143/10.672 token/s, or
1.535x/1.357x non-MTP. Retain this composition as a bounded
next-verifier dependency, without a distance/cache-level scan or production
promotion. The full 1.5x–3x goal remains unmet; 40K still needs approximately
2.27 seconds removed from full Decode. These are single-process directions,
not noise-qualified speedups or measured cache-hit attribution.

## 2026-10-10 MTP complete FP8 operand feed — closed at static gate

`WP-MTP-COMPLETE-FEED-20261010` / `AC-MTP-GREEDY-v48` owns one
[complete operand pipeline](metadata/qwen36-27b-mtp-complete-feed-design-2026-10-10.json)
against the remaining 2.67-second 40K Decode gap. Both next weights and next
activations become register-resident before current ordered FMA consumption;
the rejected weight-only v29 left activation reads on the current dependency
path. Current epilogues, scalar arithmetic and storage ownership remain fixed.
Historical L2 miss bytes closely match one weight pass, narrowing the traffic
hypothesis without asserting a current DRAM or PC-stall cause. One no-spill/
full-numerical gate returns immediately to d2 API; >3% slowdown stops, neutral
removes, useful direction unlocks d3. No profile or parameter scan follows.
Production and the complete 1.5x–3x target remain unchanged.

The [completed static rejection](metadata/qwen36-27b-mtp-complete-feed-rejection-2026-10-10.json)
passes builds and no-spill resource checks but fails its mechanism gate: in
the actual server's M3/K5120 loop, all next-record global loads follow the
last current FFMA. The compiler removed the intended overlap. This construction
is removed before GPU numerical or API runs; no new performance result follows.
Both retained-v48 executables are rebuilt with identical program sections.
A successor must prove its native producer/consumer overlap before GPU testing;
source ordering alone is insufficient. No launch/feed parameter scan follows.
Production, retained metrics and the complete 1.5x–3x objective are unchanged.

## 2026-10-10 MTP FP8 lane-owned codebook — closed negative

`WP-MTP-FP8-LANE-TABLE-20261010` / `AC-MTP-GREEDY-v47` addresses the
remaining 2.67-second 40K Decode gap with one [packed FP8 consumer composition](metadata/qwen36-27b-mtp-fp8-lane-table-design-2026-10-10.json).
A lane-owned codebook removes possible lookup bank aliasing while preserving
all scalar arithmetic and current output ownership. Unlike the rejected v12
bundle, it adds neither full-K input staging nor a persistent output grid.
The 5.869-second family is an opportunity budget, not evidence of bank stalls.
One static/full-numerical gate returns directly to the d2 API; >3% slowdown
stops, neutral removes and useful direction unlocks d3. No parameter scan or
profile follows. Production and the complete 1.5x–3x target remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-fp8-lane-table-rejection-2026-10-10.json)
passes all 36 complete FP8 output/guard cases and full P65 prefix, state/logit,
cache and transaction checks. P8192/O256 d2 nevertheless falls to 13.398
Decode token/s versus v48 13.833, with identical output and work and
0.602 seconds more verification. Prefill is 17.299 seconds / 473.55 token/s,
external TTFT 17.321 seconds and total 36.354 seconds. The declared stop
interrupts 40K; d3 and the remaining API lifecycle panel are not run.
All lane-table and profile paths are removed and retained v48 rebuilt.
No lookup-layout scan or profiler follows. This closes the consumer-table
hypothesis without establishing a hardware cause or changing the unmet goal.

## 2026-10-10 retained-v48 budget reconciliation — complete

`WP-MTP-V48-BUDGET-20261010` uses one [bounded diagnostic](metadata/qwen36-27b-mtp-v48-budget-design-2026-10-10.json)
on the exact retained v48 API binary and P40000/O256 d2 request. It asks which
verifier families still dominate the remaining 2.67-second full Decode gap
after QK, paired weights, prefix publication and row-boundary changes. One
NSys capture reconciles kernel families, copies and uncovered intervals with
the old v31 ranking. No NCU, runtime mutation or parameter scan is included.
The historical analysis window is retained explicitly; profiler durations do
not replace unprofiled API metrics. Completion must select or exclude the next
architecture directions, not open a repetition campaign. The full 1.5x–3x
target and production remain unchanged.

The [completed diagnosis](metadata/qwen36-27b-mtp-v48-budget-2026-10-10.json)
preserves exact request/output/work and normal shutdown. Current FP8 costs
5.869 seconds, Gate/Up 5.802 and QK/PV 4.532 in the declared diagnostic window.
Copies plus uncovered intervals total 0.516 seconds; a launch/copy-only change
cannot supply the remaining 2.67-second budget in this opportunity comparison.
Prioritize a distinct packed FP8 consumer composition, preserving original
scalar chains, rather than repeating the closed Gate/Up/PV/storage variants.
No hardware-stall cause or new runtime implementation is selected by NSys.
No further capture, NCU or parameter scan follows this diagnostic package.
Unprofiled v48 metrics, production and the unmet goal remain unchanged.

## 2026-10-10 MTP Gate/Up input residency — closed without retention

`WP-MTP-GATE-RESIDENCY-20261010` / `AC-MTP-GREEDY-v46` preserves the
1.5x–3x objective and remaining 2.67-second 40K full Decode gap. The [design](metadata/qwen36-27b-mtp-gate-residency-design-2026-10-10.json)
keeps existing Gate/Up shared inputs and paired codebooks resident across
disjoint output groups. The rejected v9 bundle did not isolate this dependency;
its FP8 executor is not reintroduced. Current exact paired scaling, K chains
and publication remain unchanged. One fixed resident-wave grid passes native
resource, complete projection and full P65 state/cache gates before immediately
returning to P65/8K/40K d2 API. Greater-than-3% slowdown stops, neutral removes,
useful direction unlocks d3. No geometry scan or profile follows. Useful
retention expires at the next complete verifier API return or archival;
production and its reserve remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-gate-residency-rejection-2026-10-10.json)
passes native occupancy, 45 complete projection/guard cases, all P65 numerical/
transaction checks and all eight d2 API checks. Outputs and physical/logical
work match v48, but 8K/40K Decode is 13.744/10.439 token/s versus
13.833/10.498. These single-process differences establish no useful direction,
not a statistical regression. Lower register use and staging requests do not
select the composition. All new Gate/Up/resource-plan/profile paths are removed
and v48 rebuilt. No d3, profile or geometry scan follows; the full 1.5x–3x
objective remains active and unmet, and production remains unchanged.

## 2026-10-10 MTP row boundary composition — API return complete

`WP-MTP-ROW-BOUNDARY-20261010` / `AC-MTP-GREEDY-v45` preserves the
1.5x–3x objective and remaining 0.41/3.14-second full Decode gap. The [design](metadata/qwen36-27b-mtp-row-boundary-design-2026-10-10.json)
reuses the existing exact independent-row residual/norm operator in both
per-layer boundaries, removing repeated scalar cooperative-grid launches.
The historical family costs 0.605 seconds and cannot alone close 40K; this
is a bounded scheduling prerequisite, not a reduced product objective.
One composition passes full P65 state/logit/cache and recovery admission,
then immediately returns to P65/8K/40K d2 API. Greater-than-3% slowdown stops,
neutral removes, useful direction unlocks d3. No kernel/geometry scan or
profile follows. Useful retention expires at the next complete verifier API
return or archival; production remains unchanged.

The [completed direction](metadata/qwen36-27b-mtp-row-boundary-direction-2026-10-10.json)
preserves every P65 state/cache prefix and transaction/recovery check plus all
sixteen d2/d3 API outputs. Both policies reduce verification and complete Decode
at 8K/40K. D2 reaches 13.833/10.498 token/s, or
1.502x/1.335x non-MTP. The narrow single-process 8K crossing is not
repeated qualification. Retain this scheduling composition as a bounded next-
verifier dependency, without further geometry scans or production promotion.
The full 1.5x–3x goal remains active and unmet; 40K still needs approximately
2.67 seconds removed from full Decode.

## 2026-10-10 MTP register codebook — closed negative

`WP-MTP-REGISTER-CODEBOOK-20261010` / `AC-MTP-GREEDY-v44` preserves the
1.5x–3x objective and remaining 0.41/3.14-second full Decode gap. The [design](metadata/qwen36-27b-mtp-register-codebook-design-2026-10-10.json)
replaces repeated shared-memory E2M1 pair lookups with exact register byte
selection, retaining v46 state ownership and v45 paired scaling. The historical
NV projection family supplies a roughly 9.5-second opportunity budget, not
predicted savings. One composition passes exhaustive pair/product and complete
projection checks plus full P65 numerical admission, then returns immediately
to P65/8K/40K d2 API. Greater-than-3% slowdown stops, neutral removes, useful
direction unlocks d3. No decoder parameter scan or profile follows. Useful
retention expires at the next complete verifier API return or archival.
Production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-register-codebook-rejection-2026-10-10.json)
passes exhaustive raw pairs/scaled products, 45 complete projection/guard cases,
full P65 target/cache prefix and transaction/recovery checks, but P65 d2 API
Decode falls to 10.917 token/s versus v46 12.283.
Prefill is 0.887 seconds / 73.27 token/s, external TTFT
0.906 seconds and total 2.280 seconds. Output and physical/logical
work match. The stop prevents a completed long-context/d3 panel. All new
runtime paths are removed and v46 rebuilt; no codebook scan or profile follows.
The 1.5x–3x goal remains active and unmet, with production unchanged.

## 2026-10-10 MTP expanded QKV representation — closed host assessment

`WP-MTP-EXPANDED-QKV-ASSESSMENT-20261010` assesses exact BF16 expansion
of the 48 GDN input-QKV matrices in dead Prefill storage against the remaining
0.41/3.14-second 8K/40K full Decode budgets. The [completed assessment](metadata/qwen36-27b-mtp-expanded-qkv-assessment-2026-10-10.json)
proves the proposed 5,033,164,800-byte layout fits, but does not authorize a
storage borrow. The unchanged FP8 consumer's historical same-model 40K cell
costs 2.051 seconds over 5,040 calls. Expanded weight traffic alone takes
2.168 seconds in an optimistic model allowing 16 MiB of useful cached weights
per call and full documented peak bandwidth, before any preparation or other
traffic. This is a conditional host screen, not measured candidate performance
or a hardware impossibility claim. Do not implement this representation or
repeat a GPU screen. No runtime or current API metrics change; retained v46
and the 1.5x–3x objective remain active. Further verifier work must address
the complete remaining budget without assuming decoded storage is free.

## 2026-10-10 MTP prefix publication — API return complete

`WP-MTP-PREFIX-PUBLICATION-20261010` / `AC-MTP-GREEDY-v43` preserves the
1.5x–3x goal and remaining 0.75/3.51-second full Decode gap. The [design](metadata/qwen36-27b-mtp-prefix-publication-design-2026-10-10.json)
removes dead entry snapshots, publishes original GDN states directly into
immutable prefix slots, and reuses the exact first seed-conditioned draft KV.
This bounded transaction prerequisite cannot alone close the 40K gap. One
composition passes full P513 state/logit/cache and recovery checks, then
immediately returns to P65/8K/40K d2 API. Greater-than-3% slowdown stops,
neutral removes, useful direction unlocks d3. No profile or parameter scan
follows; useful retention expires at the next complete verifier API return or
archival. Production and observer-visible state semantics remain unchanged.

The [completed direction](metadata/qwen36-27b-mtp-prefix-publication-direction-2026-10-10.json)
preserves every P513 target/cache prefix and transaction/recovery check plus
all sixteen d2/d3 API outputs. Both policies reduce verification, reconciliation
and complete Decode at 8K/40K. D2 reaches 13.518/10.299 token/s,
only 1.467x/1.310x non-MTP. Retain this exact ownership composition as a
bounded next-verifier dependency, without further transaction scans or production
promotion. The 1.5x–3x goal remains active and unmet; full Decode still needs
approximately 0.41/3.14 seconds removed at 8K/40K.

## 2026-10-10 MTP paired weight decoding — API return complete

`WP-MTP-PAIRED-WEIGHT-DECODE-20261010` / `AC-MTP-GREEDY-v42` preserves the
1.5x–3x objective and remaining 1.16/3.94-second full Decode gap. The [design](metadata/qwen36-27b-mtp-paired-weight-decode-design-2026-10-10.json)
replaces repeated scalar NVFP4 decoding/scaling with exact paired products,
keeping all FP32 accumulation and state boundaries. One fixed composition
passes exhaustive product and full numerical checks, then immediately returns
to P65/8K/40K d2 API. Greater-than-3% slowdown stops, neutral removes, useful
direction unlocks d3. No parameter scan or profile follows. Any useful dependency
expires at the next complete verifier API return or archival. Production is
unchanged; family timing is an opportunity ceiling, not predicted savings.

The [completed direction](metadata/qwen36-27b-mtp-paired-weight-decode-direction-2026-10-10.json)
passes exhaustive native products, 45 complete output/guard cases, full P65
numerical/transaction checks and all sixteen d2/d3 API checks. Both policies
reduce verification and complete Decode at 8K/40K with unchanged output/work.
D2 reaches 13.275/10.147 token/s, only 1.441x/1.291x non-MTP.
Retain this exact decoder as a bounded next-composition dependency, with no
further decoder/resource scan or production promotion. The 1.5x–3x goal remains
active and unmet; the remaining full Decode gap is 0.75/3.51 seconds.

## 2026-10-10 MTP direct short-chain execution — closed negative

`WP-MTP-DIRECT-CHAIN-20261010` / `AC-MTP-GREEDY-v41` preserves the
1.5x–3x goal and remaining 1.16/3.94-second full Decode gap. The [design](metadata/qwen36-27b-mtp-direct-chain-design-2026-10-10.json)
reuses proved exact short-chain eligibility but replaces the rejected executor's
expanded shared operands and serial eight-group loop with direct packed register
fragments and independent original-warp producers. A separate original-tree
merge consumes bounded FP32 roots. Preparation and repairs remain charged.
One fixed composition passes full synthetic/P65 admission, then returns
immediately to 8K/40K d2 API; greater-than-3% slowdown stops, neutral removes,
useful direction unlocks d3. No guard, layout, resource or geometry scan follows.
Any useful dependency expires at the next complete verifier API return or
archival. Production and its numerical contract remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-direct-chain-rejection-2026-10-10.json)
passes all synthetic and full P65 numerical/transaction checks, but P8192/O256
d2 Decode falls to 3.061 token/s versus retained v41 13.001.
Prefill is 17.302 seconds / 473.46 token/s, external TTFT
17.325 seconds and total 100.624 seconds. Both completed API outputs
and acceptance/work match; verification adds 63.684 seconds at 8K.
The declared stop prevents 40K/d3 and the remaining API lifecycle panel.
All new executor, metadata ownership and profile paths are removed and v41
rebuilt. Direct fragments and independent warp producers do not make this
exact short-chain executor competitive; numerical eligibility alone cannot
select it. No profile or further guard/layout/grid scan follows. The 1.5x–3x
goal remains active and unmet, and production remains unchanged.

## 2026-10-10 MTP Gate/Up shared-input chain composition — closed negative

`WP-MTP-GATE-CHAIN-SHARE-20261010` / `AC-MTP-GREEDY-v40` preserves the
1.5x–3x goal and remaining 1.16/3.94-second complete Decode gap. The
[fixed design](metadata/qwen36-27b-mtp-gate-chain-share-design-2026-10-10.json)
composes independent Gate/Up chains with retained full-K input sharing and
original N16 output ownership. The closed v34 removed that producer reuse and
also changed Down; this composition retains Down and all other v41 execution.
One fixed implementation must pass static resource and full numerical gates,
then immediately return to P65/8K/40K d2 API. Greater-than-3% slowdown stops,
neutral removes, useful direction unlocks d3. No geometry or resource scan
follows. A useful dependency expires at the next full verifier API return or
archival. Production remains unchanged; family cost is not projected savings.

The [completed rejection](metadata/qwen36-27b-mtp-gate-chain-share-rejection-2026-10-10.json)
passes static resource, nine complete output/guard cases and all P65 numerical
and transaction checks, but P65/O16 d2 Decode falls to 10.748 token/s versus
retained v41 11.879. Output and acceptance/work are identical; verification
adds 133.3 ms. The declared stop interrupts the following 8K request before
completion; no 40K, d3 or full API lifecycle panel follows. All new runtime
paths are removed and v41 rebuilt. Retaining input reuse does not rescue this
chain-consumer composition. No further chain, geometry or resource scan opens.
The 1.5x–3x goal remains active and unmet; production is unchanged.

## 2026-10-10 MTP shared-row PV recomposition — closed without retention

`WP-MTP-PV-ROW-COMPOSE-20261010` / `AC-MTP-GREEDY-v39` preserves the1.5x–3x
goal and remaining1.16/3.94-second full Decode gap. The [fixed composition](metadata/qwen36-27b-mtp-pv-row-compose-design-2026-10-10.json)
adds the existing exact shared-row PV component to retained compact QK. The
historical v19 combined failure did not identify PV's separate contribution;
this one composition answers that question without a tile or staging scan.
Full P513/P8192 admission returns immediately to8K/40K d2 API; greater-than3%
slowdown stops, neutral removes, useful direction unlocks d3. A useful dependency
expires at the next complete verifier API return or archival. PV alone cannot
close the40K gap. Production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-pv-row-compose-rejection-2026-10-10.json)
preserves full P513/P8192 numerical checks and all eight d2 API checks, with
unchanged output and acceptance/work. Decode is12.989/9.957 token/s at8K/40K,
respectively0.09%/0.19% below retained v41 in one process. This provides no
useful gain or statistical regression claim. All new PV/profile/auditor paths
are removed and v41 rebuilt. No d3, profile or staging/geometry scan follows.
Retained v41 and the unmet1.5x–3x objective remain unchanged.

## 2026-10-10 MTP compact QK row sharing — API return complete

`WP-MTP-QK-ROW-SHARE-20261010` / `AC-MTP-GREEDY-v38` preserves the1.5x–3x
objective and remaining1.19/4.21-second complete Decode gap. The [design](metadata/qwen36-27b-mtp-qk-row-share-design-2026-10-10.json)
shares each K position across all speculative rows in a compact2*M-warp CTA,
keeping all six query heads and original numerical trees. This replaces the
rejected large8*M-warp/head-split ownership, not its parameters. One fixed
implementation passes P513/P8192 full state/logits and immediately returns to
8K/40K d2 API; useful direction unlocks d3, greater-than3% slowdown stops and
neutral removes. No geometry or staging scan follows. Any useful dependency
expires at the next complete verifier API return or archival. Production is
unchanged; this QK budget alone cannot close the40K target gap.

The [completed direction](metadata/qwen36-27b-mtp-qk-row-share-direction-2026-10-10.json)
preserves all complete numerical/transaction checks and all sixteen d2/d3 API
outputs and work counts. Verification falls by34/51ms at8K and269/306ms at40K.
D2 Decode is13.001/9.976 token/s; its non-MTP ratios remain only1.411x/1.269x.
This is a small bounded dependency through the next complete verifier API
return or archival, not noise-qualified architecture selection. Prefill changes
are outside the edited code; no whole-request attribution or production
promotion follows. No mapping or staging scan remains active. The1.5x–3x
goal remains unmet, with approximately1.16/3.94 seconds left in full Decode.

## 2026-10-10 MTP lossless FP8 traffic — closed at host gate

`WP-MTP-LOSSLESS-FP8-20261010` preserves the 1.5x–3x objective and remaining
1.19/4.21-second complete Decode gap. The [fixed packet design](metadata/qwen36-27b-mtp-lossless-fp8-design-2026-10-10.json)
assesses all208 real FP8 tensors using six-bit palette codes, literal exceptions
and bounded packet offsets. It preserves raw bytes and original arithmetic.
Less than20% complete payload saving closes before GPU, without a format scan.
A pass requires one separately bounded owner/decoder/executor composition and
immediate real API return; compression alone grants no speedup or retention.
Production and retained v39 remain unchanged.

The [completed assessment](metadata/qwen36-27b-mtp-lossless-fp8-assessment-2026-10-10.json)
authenticates all three shards and all208 FP8 tensors. Including literal
exceptions, packet offsets and palettes, 7,214,202,880 raw bytes become
6,702,302,636 bytes: only7.096% saving, below the declared20% floor. All3328
sampled packet roundtrips and12 controls preserve raw bytes. The format closes
before native implementation or GPU work; no alternate format scan follows.
No speedup, hardware bound or new API measurement is claimed. The unchanged
1.5x–3x goal still requires a materially larger complete-verifier improvement.

## 2026-10-10 MTP short-chain executor — closed negative

`WP-MTP-SHORT-CHAIN-EXECUTOR-20261010` / `AC-MTP-GREEDY-v37` implements the
[fixed complete executor](metadata/qwen36-27b-mtp-short-chain-executor-design-2026-10-10.json)
after native/current-payload admission. The unchanged 1.5x–3x goal and
1.19/4.21-second remaining complete Decode gap select compact operands,
nine-bit metadata, cheap runtime guards, exact short-chain repairs and on-chip
original reductions together. Preparation and repair remain charged to Decode.
One implementation plus at most one correctness repair returns through complete
synthetic/P65 state admission immediately to 8K/40K d2 API; useful direction
unlocks d3. Greater-than-3% long-context slowdown stops; neutral removes.
No further guard/layout/grid scan is active. Production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-short-chain-executor-rejection-2026-10-10.json)
preserves all synthetic and full P65 state/logit/transaction checks, plus both
completed API outputs and work counts. P8192/O256 d2 Decode falls to 4.291
token/s versus retained v39 12.978, so 40K/d3 are not run. One same-ELF P65
profile assigns 2.745 GPU seconds to repeated FP8 execution, versus 0.167 to
one-time metadata and 0.006 to input bounds. Preparation amortization cannot
rescue this version. The executor and borrowed ownership are removed and v39
rebuilt; no guard/layout/grid scan or additional certificate campaign follows.
The 1.5x–3x objective remains unmet. A successor must eliminate repeated
execution cost, not merely increase numerical eligibility or move preparation.

## 2026-10-10 MTP short-chain native admission — prerequisite complete

`WP-MTP-SHORT-CHAIN-DEVICE-20261010` preserves the 1.5x–3x objective and
remaining 1.19/4.21-second Decode gap. The [design](metadata/qwen36-27b-mtp-short-chain-device-design-2026-10-10.json)
checks one native BF16 matrix instruction that pairs two original short chains
in disjoint K/column blocks. The earlier 128-cell host grouping was a proxy,
not a realizable dense fragment; the explicit M3 mapping has 96 useful cells.
One numerical device process must pass before one current P65 adjacent-row
capture. Product, root, fragment and guard checks do not establish speedup.
Coverage below 50% or numerical failure closes this version without scanning.
After both gates, a bounded complete guard/operand/on-chip reduction/fallback
executor must return promptly to the real API. Production remains unchanged.

The [completed admission](metadata/qwen36-27b-mtp-short-chain-device-admission-2026-10-10.json)
passes 16,255,180 native valid product pairs and 581,632 matrix output checks,
with zero mismatches. Current P65 capture preserves all three complete prefix
states/full logits and all 2,688 saved projection outputs. Actual paired M3
complete-unit eligibility is 20,957/28,672 (73.09%), above the declared floor.
One fixed separable exponent guard admits 2,448,199/2,752,512 individual roots
(88.94%), with no false positives against the exact host predicate. This closes
numerical feasibility, not runtime guard economics or a speedup. Temporary
capture seams are removed and the exact retained v39 oracle is rebuilt.
The next work is one composed executor with cheap guards, compact operands,
on-chip original reductions and exact fallback, followed immediately by the
same real API. No further standalone admission campaign is pending. The
1.5x–3x target remains unmet and production is unchanged.

## 2026-10-10 MTP original short-chain exactness — host feasibility passed

`WP-MTP-SHORT-CHAIN-LATTICE-20261010` keeps the 1.5x–3x goal and remaining
1.19/4.21-second full Decode gap. The [design](metadata/qwen36-27b-mtp-short-chain-lattice-design-2026-10-10.json)
checks exact integer-lattice closure of the original five/six-product FP8
chains, preserving every later reduction and publication. This differs from
approximate full-K certificates and rounded-carry MMA substitution. One
hash-authenticated historical real-input host panel reproduces complete scalar
outputs and assesses 128-cell cooperative units. Below 50% complete-unit
coverage, close without GPU; a pass requires separate current-payload and
native-instruction proof before a bounded executor and immediate API return.
No numerical waiver, timing or parameter scan opens here. Production remains
unchanged; artifacts stay under `.q3x-work/mtp-short-chain-lattice-20261010/`.

The [completed host assessment](metadata/qwen36-27b-mtp-short-chain-lattice-feasibility-2026-10-10.json)
reproduces all 2,688 captured outputs and finds 2,734,028 of 2,752,512 chains
exact (99.33%). Complete 128-cell coverage is 13,616/21,504 (63.32%), above
the declared floor; every admitted root matches its independent integer sum.
This closes the host assessment positively, with no GPU or performance claim.
Next admission must bind current adjacent output groups and prove native
zero-start instruction behavior. Runtime guard cost, compact operand delivery,
on-chip original-tree merging and fallback remain unresolved; repeating the
old costly certificate executor is not selected. The 1.5x–3x goal is unmet.

## 2026-10-10 MTP QK live reductions — API return complete

`WP-MTP-QK-LIVE-REDUCTION-20261010` / `AC-MTP-GREEDY-v36` retains the
1.5x–3x objective and complete 18.45/21.62-second Decode budgets. The
[design](metadata/qwen36-27b-mtp-qk-live-reduction-design-2026-10-10.json)
transfers the retained exact live-ancestor projection reduction to six QK head
trees, eliminating dead-lane work without changing products, grids, causal
bounds or KV ownership. Current matched QK work is 3.077 GPU seconds; this
alone cannot close the remaining 40K gap. One fixed implementation plus at
most one correctness repair passes directed trees and complete P513/P8192
state/logits, then returns immediately to the same d2 API. Greater-than-3%
regression stops; neutral removes; useful direction unlocks d3. No mapping
or resource scan follows. Any useful prerequisite expires at the next full
verifier API return or archival. Production remains unchanged.

The [completed direction](metadata/qwen36-27b-mtp-qk-live-reduction-direction-2026-10-10.json) passes 24,576 raw FP32 roots,
complete P513/P8192 state/logits and all sixteen d2/d3 API/lifecycle checks.
Output and acceptance/work remain unchanged. Verification falls by about
0.10 seconds at 8K and 0.57/0.63 seconds at 40K; d2 Decode is 12.978/9.872
token/s. This exact reduction remains a bounded development dependency through
the next full verifier API return or archival. There is no further QK mapping
scan or qualification-only campaign. The single-process 40K total is mixed
because Prefill is longer; no production selection follows. The 1.5x–3x goal
remains unmet, with roughly 1.20/4.21 seconds still to remove from complete
8K/40K Decode. The next architecture must address dominant target computation.

## 2026-10-10 retained MTP v31 budget — diagnosis complete

`WP-MTP-V31-BUDGET-20261010` preserves the 1.5x–3x goal and complete
18.45/21.62-second Decode budgets. The [design](metadata/qwen36-27b-mtp-v31-budget-design-2026-10-10.json)
permits one same-frozen-v31-ELF P40000/O256 d2 real API capture after its
completed unprofiled direction. The old full-model attribution predates the
retained draft/head/projection changes. This capture updates family/copy/gap
attribution to select the next complete execution response; it changes no
performance baseline, runtime or production route. Close on completion or
resource failure, with no NCU replay or parameter scan. Artifacts remain under
`.q3x-work/mtp-v31-budget-20261010/`.

The [completed diagnosis](metadata/qwen36-27b-mtp-v31-budget-2026-10-10.json)
preserves output, acceptance and work counts at the same frozen v31 ELF.
From first verifier projection to last generation kernel, current verifier
projections take 14.706 GPU seconds, QK/PV 5.395, all copies 0.522 and the
uncovered activity interval 0.317. The latter two together are far below the
remaining 4.774-second unprofiled 40K Decode gap. Collection is closed; no
hardware-stall cause, ceiling or new performance baseline is claimed. The next
complete computation/executor response must address dominant verification;
copy/coordination work and repeated rejected layout/feed variants cannot
substitute for that budget. Production, retained v31 and the unmet goal remain
unchanged.

## 2026-10-10 MTP activation publication — closed negative

`WP-MTP-ACTIVATION-PUBLICATION-20261010` / `AC-MTP-GREEDY-v35` targets the
unchanged 1.5x–3x objective and 18.45/21.62-second complete Decode budgets.
The [design](metadata/qwen36-27b-mtp-activation-publication-design-2026-10-10.json)
publishes exact FP32 inputs once per transformer projection for all output
consumers. Original FMA/reduction/publication boundaries remain unchanged;
doubled input traffic and producer overhead must be repaid in the actual API.
One implementation plus at most one correctness repair passes resource and
full numerical admission, then immediately returns to P65/8K/40K d2 API.
Greater-than-3% slowdown stops; neutral removes; useful direction unlocks d3.
No parameter scan follows. Any retained dependency expires at the next full
verifier API return or archival. Production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-activation-publication-rejection-2026-10-10.json)
passes all 65,536 BF16 patterns, 72 complete projection cases and full P65
state/logit/transaction checks. P65/O16 d2 preserves output and work, but
Decode falls to 10.737 token/s versus v31 11.902, a 9.78% negative direction.
Verification adds 136.775 ms. The stop occurs before 8K; no 40K, d3 or full
API lifecycle claim follows. All new runtime/profile/auditor paths are removed
and v31 rebuilt. No activation representation/cache/grid scan follows. The
1.5x–3x goal remains active and unmet; removing unpack instructions without
improving the complete projection execution cost did not deliver the target.

## 2026-10-10 MTP batched cache reconciliation — closed without retention

`WP-MTP-BATCH-RECONCILE-20261010` / `AC-MTP-GREEDY-v34` preserves the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The [design](metadata/qwen36-27b-mtp-batch-reconcile-design-2026-10-10.json)
batches already available target-conditioned cache rows before publishing each
accepted logical prefix. It removes repeated FC/K/V weight traversals while
preserving per-token observer state and the original draft arithmetic. The
approximately 0.447-second reconciliation phase bounds this prerequisite;
it cannot alone close the remaining complete-verifier gap.

One implementation plus at most one correctness repair returns through full
P65 state/logit/cache/transaction admission immediately to the same d2 API.
Greater-than-3% slowdown stops; neutral removes; useful direction unlocks d3.
No batch-size or draft-length scan follows. Any useful dependency expires at
the next complete verifier API return or archival. Artifacts remain under
`.q3x-work/mtp-batch-reconcile-20261010/`; production is unchanged.

The [completed direction](metadata/qwen36-27b-mtp-batch-reconcile-rejection-2026-10-10.json)
passes all nine complete target and draft-cache prefix comparisons, tail and
prepared-authority guards, full transaction/recovery admission and all eight
d2 API/lifecycle checks. Work and acceptance match v31. Reconciliation falls
by 131/125 ms at 8K/40K, but complete Decode changes only +0.61%/+0.36% in
one process, and 40K external total is effectively unchanged. This does not
select useful whole-runner retention. All new runtime/profile/test/auditor paths
are removed and v31 rebuilt. No d3, repetition or batching scan follows. The
1.5x–3x objective remains unmet; dominant target verification, rather than this
small coordination phase, must supply the remaining complete Decode savings.

## 2026-10-10 MTP exact QK lattice — broad-cooperative proposal closed

`WP-MTP-QK-LATTICE-20261010` preserves the 1.5x–3x real API objective and
18.45/21.62-second complete Decode budgets. Its [design](metadata/qwen36-27b-mtp-qk-lattice-design-2026-10-10.json)
checks one exact integer-lattice predicate for the original eight-product QK
subtrees. All signed intermediate sums must fit FP32 exactly before any
order-independent executor could be considered. One authenticated historical
real-input panel checks coverage and the original scalar tree; its older
Legacy Prefill inputs cannot establish current MTP eligibility or performance.
Below 50% complete cooperative-unit coverage, close this broad-cooperative
proposal without a GPU or parameter scan. Above that floor, current MTP capture
and an independent native-instruction proof must precede a separately bounded
executor with immediate real API return. The old 3.077-second QK attribution
cannot alone close the 40K gap. Artifacts remain under
`.q3x-work/mtp-qk-lattice-20261010/`; runtime and production are unchanged.

The [completed assessment](metadata/qwen36-27b-mtp-qk-lattice-feasibility-2026-10-10.json)
authenticates the four-layer P40000 payload panel and checks 786,432 original
eight-product groups. Of these, 672,226 satisfy the exact predicate, with zero
admitted scalar-tree mismatches; independent integer checks cover every subset
of 64 selected admitted groups. Only 990 of 16,384 complete cooperative units
pass (6.04%), below the declared floor. This broad-cooperative proposal closes
before current-MTP capture or GPU implementation. No per-cell repair or mapping
scan follows. The historical-input scope cannot establish current MTP eligibility,
native instruction equivalence or a performance ceiling. Retained v31 and its
performance observations remain unchanged; the 1.5x–3x objective stays active.

## 2026-10-10 MTP Gate/Up producer-consumer fusion — closed without retention

`WP-MTP-GATE-PAIR-20261010` / `AC-MTP-GREEDY-v33` preserves the 1.5x–3x
goal and 18.45/21.62-second complete Decode budgets. The [design](metadata/qwen36-27b-mtp-gate-pair-design-2026-10-10.json)
selects one paired Gate/Up executor: separate original warp consumers share
input/table publication, then original BF16-rounded values feed exact SiLU
inside the CTA. It halves duplicate input production without doubling each
thread's accumulators and removes dead global intermediate publication. The
older 5.874-second family attribution is a ceiling, not projected savings.
State-copy removal and draft precision retuning are not selected by the bounded
source/evidence audit. No state, numerical, acceptance or inventory change occurs.

One fixed implementation plus at most one correctness repair passes no-spill
and resource-floor checks, complete producer/SiLU output guards and P65 full
prefix/state/logit transactions, then immediately returns to the same P65/8K/40K
d2 API. Greater-than-3% slowdown stops; neutral/negative removes; useful direction
unlocks d3. No launch/register/stage scan follows. A useful dependency expires
at the next complete verifier API return or archival. Artifacts stay under
`.q3x-work/mtp-gate-pair-20261010/`; production remains unchanged.

The [completed direction](metadata/qwen36-27b-mtp-gate-pair-rejection-2026-10-10.json)
passes all static, twelve full-output/guard, complete P65 state/logit/transaction
and eight d2 API/lifecycle checks. Work and acceptance match v31. At 8K/40K,
Decode is 12.915/9.656 token/s versus 12.914/9.660, only +0.014%/-0.045% in
one process; verification differs by -4.8/+9.9 ms. No useful whole-runner gain
is established. All new runtime/profile/auditor paths are removed and v31
rebuilt. No d3, profiler, producer-grid or register scan follows. The 1.5x–3x
goal remains active and unmet; fewer publications and launches did not address
the material verifier cost. A successor needs a different complete computation
or execution response with an immediate real API return.

## 2026-10-10 MTP direct PV consumption — closed negative

`WP-MTP-DIRECT-PV-20261010` / `AC-MTP-GREEDY-v32` keeps the 1.5x–3x goal
and complete 18.45/21.62-second Decode budgets. The [design](metadata/qwen36-27b-mtp-direct-pv-design-2026-10-10.json)
removes the shared PV producer/copy ring and CTA barriers, retaining the exact
increasing-position FMA chains in independent full-head consumers. Repeated
global V loads replace physical shared reuse, so only the complete API selects
the trade. This differs from the rejected shared-KV ownership and rounding
elision mechanisms. The older 2.318-second PV attribution cannot alone close
the 40K gap and is not a projected gain.

One fixed implementation plus at most one correctness repair passes static
resource review and complete P513/P8192 scalar state/logit admission, then
returns immediately to P65/8K/40K d2 API; useful direction unlocks d3. A
greater-than-3% long-context regression stops; neutral/negative removes the
new route. No prefetch/cache/grid scan follows. A useful dependency expires at
the next complete verifier API return or archival. Artifacts remain under
`.q3x-work/mtp-direct-pv-20261010/`; production is unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-direct-pv-rejection-2026-10-10.json)
passes the static no-spill gate, all eighteen P513/P8192 prefix comparisons
and full P513 transaction/recovery checks. P65 output/performance is unchanged,
but P8192/O256 d2 Decode falls to 11.750 token/s versus v31 12.914, with
identical acceptance/work. Verification adds 1.950 seconds; external total
adds 1.981 seconds. The declared stop occurs during the idle gap before 40K;
40K, d3 and the full API lifecycle panel are not run. All new runtime paths
are removed and v31 restored. No cache/prefetch/grid scan follows. This closes
direct global consumption without a hardware-stall claim; the 1.5x–3x goal
remains active and unmet, and the next architecture must address a materially
different complete verification budget rather than repeat PV ownership variants.

## 2026-10-10 MTP scalar-chain Tensor Core mapping — direct substitution rejected

`WP-MTP-SINGLE-PRODUCT-MMA-20261010` retains the 1.5x–3x goal and complete
18.45/21.62-second Decode budgets. The [bounded design](metadata/qwen36-27b-mtp-single-product-mma-design-2026-10-10.json)
tests a different executor prerequisite: one nonzero product per MMA output,
with the original FP32 accumulator carried in original chain order. NVIDIA's
ISA does not guarantee identical rounding to scalar FMA. One directed SM87
numerical process therefore checks finite rounding and final BF16 publication
before any runtime integration. A finite counterexample closes this direct
substitution; a sample pass alone is not equivalence proof or selection. No
timing, profiler or geometry scan follows this screen. Artifacts remain under
`.q3x-work/mtp-single-product-mma-20261010/`; production is unchanged.

The [completed numerical screen](metadata/qwen36-27b-mtp-single-product-mma-2026-10-10.json)
confirms ordinary finite counterexamples on SM87. Exact layout controls and
guards pass, and GPU scalar FMA agrees with independent host/directed bits.
The zero-start four-product chain yields scalar FP32 `0x3f818000` versus MMA
`0x3f817fff`, publishing different BF16 values `0x3f82` versus `0x3f81` on
all 128 outputs. Every operand is an exactly representable FP8 weight or BF16
activation. Thus eliminating multi-product reassociation does not make this
instruction a drop-in scalar FMA. The direct substitution is closed before
runtime integration, with no timing or hardware-performance conclusion. A
sample cannot quantify real-model mismatch frequency. Retained v31 and the
unmet 1.5x–3x objective remain unchanged; a successor must supply a different
proved computation or exact execution dataflow and a bounded real API return.

## 2026-10-10 MTP role-aware draft representation — rejected before GPU

`WP-MTP-MIXED-DRAFT-20261010` keeps the 1.5x–3x goal and exact target
verification. The [bounded design](metadata/qwen36-27b-mtp-mixed-draft-design-2026-10-10.json)
responds to the measured FC diagonal risk by preserving the complete FC matrix
in BF16 and using one fixed K128-group INT8 representation for the other seven
draft matrices. Its host screen includes BF16 operand reconstruction before
independent FP64 projections and compares complete hidden/K/V to the original
reference at the unchanged 0.02 bound. One fixed full-layer evaluation either
closes this representation or proceeds directly to a separately identified
native packed owner/consumer and the same real d2/d3 API. No group-size or
quantizer scan follows. Draft savings alone cannot close the 40K budget gap;
any retained dependency must compose with further verifier work. Artifacts
stay under `.q3x-work/mtp-mixed-draft-20261010/`; production is unchanged.

The [completed assessment](metadata/qwen36-27b-mtp-mixed-draft-2026-10-10.json)
authenticates all reused captures, original independent outputs and checkpoint
payloads. K/V pass at 0.008094/0.006036 maximum-row relative L2, but complete
hidden fails at 0.036985, with two of 64 rows above 0.02. Preserving FC and
using grouped scales substantially reduces this representation's error but
does not meet the unchanged full-layer contract. The conditional native
owner/consumer is therefore not implemented; no GPU, acceptance or performance
result follows. This fixed representation is closed without a quantizer or
group-size scan. Retained v31, production and performance observations are
unchanged. The 1.5x–3x objective remains active and unmet; a successor must
address the complete verifier budget with a new bounded architecture and real
API return, rather than treat lower weight error as a delivered improvement.

## 2026-10-10 MTP draft INT8 feasibility — fixed quantizer rejected

`WP-MTP-DRAFT-INT8-FEASIBILITY-20261010` preserves the 1.5x–3x target,
18.45/21.62-second Decode budgets and exact target verification. The
[design](metadata/qwen36-27b-mtp-draft-int8-feasibility-design-2026-10-10.json)
assesses one fixed per-channel INT8 draft representation against the original
independent FP64 full-layer reference on retained real P65 target hidden.
The unchanged 0.02 maximum-row hidden/K/V bound is a feasibility screen;
no runtime arithmetic, accuracy contract or numerical baseline changes here.
The measured 2.06/2.45-second draft cost is only a bounded opportunity and
cannot alone close the 40K gap. One original/quantized host pass either closes
the fixed quantizer or selects a separately bounded native owner/consumer
composition with immediate real API return and acceptance accounting. No
quantizer/group-size scan or GPU process opens. Artifacts stay under
`.q3x-work/mtp-draft-int8-feasibility-20261010/`; production is unchanged.

The [completed assessment](metadata/qwen36-27b-mtp-draft-int8-feasibility-2026-10-10.json)
reproduces the original BF16 draft's independent-reference pass but rejects
this fixed INT8 representation before GPU work. Maximum-row hidden/K/V
relative L2 is 0.19197/0.13003/0.11821 against the unchanged 0.02 bound.
FC has 5,113 of 5,120 row maxima on its hidden-half diagonal; its one-scale
quantization zeros 23.0% of codes and gives 12.77% weight relative L2 error.
Other matrices have 0.93–1.50% weight error. This is a specific representation
risk, not proof that every INT8 method fails or attribution of all output error
to FC. No runtime, numerical contract, GPU process or performance baseline
changes. The 1.5x–3x goal remains active; a future outlier-preserving proposal
requires a new bounded design and real API return, and is not selected by
this host-only closure. No quantizer/group-size scan follows here.

## 2026-10-10 MTP independent accumulation chains — closed negative

`WP-MTP-CHAIN-PARTITION-20261010` / `AC-MTP-GREEDY-v31` keeps the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The
[design](metadata/qwen36-27b-mtp-chain-partition-design-2026-10-10.json)
partitions four independent NVFP4 FMA chains across adjacent lanes while
preserving per-weight reuse across all speculative rows. Unlike closed token
ownership or lookahead variants, it changes ownership of independent scalar
chains and reconstructs the identical original tree after K completes. This
addresses the older 8.845-second NV family ceiling; it is not a projected gain.
FP8 and head stay v31. One fixed mapping, no spill and complete synthetic/P65
state admission precede immediate P65/8K/40K d2 API; greater-than-3% regression
stops, neutral/negative removes, useful direction unlocks d3. At most one
correctness repair is allowed. No chain, grid, decoder, layout or resource scan
follows. A useful dependency expires at the next complete verifier API return
or archival. Artifacts stay under `.q3x-work/mtp-chain-partition-20261010/`;
production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-chain-partition-rejection-2026-10-10.json)
passes the static no-spill gate, 30 complete output/guard cases and full P65
prefix/state/logit/transaction checks. P65/O16 d2 preserves baseline output and
work counts but Decode falls to 9.691 token/s versus v31 11.902, triggering
the declared early stop. Verification adds 287.332 ms. The following 8K
request is interrupted; there is no 40K, d3 or full API lifecycle claim.
All new runtime/profile/auditor paths are removed and v31 rebuilt. Lower
register pressure and exact arithmetic did not select a faster runner. No
chain, ownership, grid or resource scan follows. The 1.5x–3x objective remains
unmet; the next package must remove material complete-verifier work or change
its producer/consumer boundaries, rather than repeat accumulator redistribution.

## 2026-10-10 MTP NV register feed — closed without retention

`WP-MTP-NV-REGISTER-FEED-20261010` / `AC-MTP-GREEDY-v30` retains the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The
[design](metadata/qwen36-27b-mtp-nv-register-feed-design-2026-10-10.json)
selects one raw-record lookahead in NVFP4 Gate/Up and Down, preserving scale
ownership and original arithmetic. This family has a much longer consumption
loop than the closed FP8 lookahead; its existing real-input counter evidence
supports a bounded test, not a transferred performance claim. The old 8.845-second
family attribution is a ceiling only. FP8 and vocabulary execution stay v31.

Fixed launch bounds preserve the incumbent per-role M2/3/4 CTA floors. Any
spill or resource-floor loss closes this implementation before GPU execution;
no resource, launch, decoder or layout scan is admitted. If admitted, complete
synthetic and P65 state/logit checks precede immediate P65/8K/40K d2 API.
Greater-than-3% regression stops; neutral/negative removes; useful direction
unlocks d3. One implementation plus at most one correctness repair is the bound.
A useful result expires at the next complete verifier API return or archival.
Artifacts stay under `.q3x-work/mtp-nv-register-feed-20261010/`; production
is unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-nv-register-feed-rejection-2026-10-10.json)
passes the static no-spill/resource floor, 30 complete synthetic outputs/guards,
full P65 state/logit/transaction checks and all eight d2 API/lifecycle requests.
Work and acceptance match v31. D2 Decode is 12.698/9.527 token/s at 8K/40K,
versus retained 12.914/9.660; external total adds 0.305/0.301 seconds. This is
a negative direction screen, not a statistically qualified regression. All new
runtime/profile/auditor changes are removed and v31 restored. D3 is not run;
no register, launch, feed or resource scan follows. Earlier weight issue and
no spills do not establish whole-runner benefit. The 1.5x–3x goal remains
active and unmet; the next package must address the complete verifier budget
with a materially different execution response and immediate real API return.

## 2026-10-10 MTP softmax lifetime composition — closed without retention

`WP-MTP-SOFTMAX-LIFETIME-20261010` / `AC-MTP-GREEDY-v29` keeps the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The
[design](metadata/qwen36-27b-mtp-softmax-lifetime-design-2026-10-10.json)
selects one complete normalization-lifetime change: batch all speculative
head rows in one grid, compute exponentials once in existing dead score cells,
and retain the exact original denominator tree with warp-local final ancestors.
The older matched softmax attribution is below one second and includes draft;
this is a bounded prerequisite, not a claim to close the full remaining gap.
QK/PV, projections, draft and all state/publication semantics stay v31.

One implementation plus at most one correctness repair passes complete FP32
probability/guard tests and P513/P8192 state/logit admission, then returns
immediately to P65/8K/40K d2 API. Greater-than-3% regression stops; neutral or
negative removes it. Useful direction unlocks d3 and retains only a bounded
dependency through the next complete verifier API return or archival. No local
parameter/timing or profiler campaign opens; artifacts stay under
`.q3x-work/mtp-softmax-lifetime-20261010/`. Production is unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-softmax-lifetime-rejection-2026-10-10.json)
passes 60 full FP32 probability/guard cases, all eighteen P513/P8192 prefix
comparisons, full P513 transactions/recovery and eight d2 API/lifecycle checks.
Work and acceptance match v31. D2 observes 12.984/9.724 token/s at 8K/40K
versus 12.914/9.660: only +0.55%/+0.67% in one process, without noise
separation. The 40K external request is 0.403 seconds longer. This does not
establish a material whole-product gain; all new runtime/profile/auditor paths
are removed and v31 restored. D3 is not run. No further softmax mapping,
exponential-storage or launch scan follows. The 1.5x–3x goal remains active
and unmet; the next package must address a materially larger complete verifier
budget, rather than compound additional sub-percent normalization variants.

## 2026-10-10 MTP NV live reduction composition — API return complete

`WP-MTP-NV-LIVE-REDUCTION-20261010` / `AC-MTP-GREEDY-v28` retains the
1.5x–3x objective and 18.45/21.62-second complete Decode budgets. The
[design](metadata/qwen36-27b-mtp-nv-live-reduction-design-2026-10-10.json)
composes the proved v30 live-ancestor reduction with NVFP4 Gate/Up, Down and
target vocabulary finalization. Longer NV mainloops limit the source-level
opportunity; instruction ratios are not measured latency or a speedup claim.
One fixed reuse of the existing helper, without new grids/layouts/decoders,
returns through complete synthetic and P65 state/logit admission to the real
P65/8K/40K d2 API. Greater-than-3% regression stops; neutral/negative removes
the new mapping, and useful direction unlocks d3. At most one correctness
repair is allowed. A useful result is only a bounded prerequisite through the
next complete verifier API return or archival. Artifacts remain under
`.q3x-work/mtp-nv-live-reduction-20261010/`; production is unchanged.

The [completed direction](metadata/qwen36-27b-mtp-nv-live-reduction-direction-2026-10-10.json)
passes all numerical checks and both complete API panels. D2 observes
12.914/9.660 token/s at 8K/40K versus v30 12.708/9.533; d3 observes
11.310/8.665 versus 11.164/8.574, with identical work and acceptance counts.
The exact mapping remains a bounded development dependency through the next
complete verifier architecture's first API return or archival. No further
mapping/grid scan or qualification-only campaign follows. This is a small
single-process direction, not statistical architecture selection or production
promotion. The 1.5x–3x objective remains active and unmet: even d2 still needs
about 1.30/4.78 seconds less complete Decode at 8K/40K. A successor must name
a materially different complete verifier response and promptly return to the
same real API rather than repeat tail-reduction variants.

## 2026-10-10 MTP causal proposal reuse — assessment complete

`WP-MTP-CAUSAL-REUSE-20261010` keeps the 1.5x–3x objective and the
18.45/21.62-second complete Decode budgets. The [design](metadata/qwen36-27b-mtp-causal-reuse-design-2026-10-10.json)
selects one host-only feasibility audit of repeated committed-history proposals
at fixed draft lengths 2/3. The pinned vLLM default supplies one fixed five-token
match with earliest occurrence. Selection may read only prompt and committed
history; retained future target tokens are comparison-only after selection.
Every possible entry in the existing P65/8K/40K output is assessed. This cannot
simulate unknown MTP proposals at changed fallback positions or establish API
speedup. One audit either closes the direction or selects a separately bounded
hybrid-proposal composition with unchanged exact target verification and prompt
API return. No n-gram/draft-length scan, GPU run or runtime mutation is opened.
Artifacts stay under `.q3x-work/mtp-causal-reuse-20261010/`.

The [completed assessment](metadata/qwen36-27b-mtp-causal-reuse-assessment-2026-10-10.json)
finds zero matched entries at P65, about 54% at P8192 and only 7.1% at P40000.
All 1,035 entry decisions pass an independent earliest-match/causal-bound scan.
This is insufficient evidence to select proposal reuse as the response to the
full target-domain gap; no runtime composition, GPU test or matching-parameter
scan follows. The 8K opportunity remains workload-specific, without a hybrid
schedule or speedup claim. Retained v30 and the 1.5x–3x objective are unchanged;
a successor must address the complete verifier budget and return to the real
40K API, rather than substitute a repetition-rich subset for the goal.

## 2026-10-10 MTP live reduction ownership — API return complete

`WP-MTP-LIVE-REDUCTION-20261010` / `AC-MTP-GREEDY-v27` retains the
1.5x–3x objective and 18.45/21.62-second complete Decode budgets. Weight
expansion was assessed but is not selected: complete decoded FP8 inputs exceed
the borrowed arena and double repeated weight bytes. The [design](metadata/qwen36-27b-mtp-live-reduction-design-2026-10-10.json)
instead removes dead-lane reduction work while preserving every live ordered
add. Four original FP8 output trees migrate into four 8-lane groups; packed
final partial reduction retains both leading zero additions. Packed weights,
mainloops, launch grids and all other v28 mechanisms remain unchanged.

The [MTP ledger](MTP_ADMISSION.md#live-reduction-ownership-composition-v27)
defines exact mapping. One implementation plus at most one correctness repair
passes directed tree/full-output and P65 transaction checks, then returns
immediately to P65/8K/40K d2 API. Greater-than-3% regression stops; neutral
or negative direction removes it. Useful direction unlocks d3 and grants only
bounded prerequisite authority through the next complete verifier API return.
No kernel grid, layout or decoder scan opens. Artifacts remain under
`.q3x-work/mtp-live-reduction-20261010/`; production is unchanged.

The [completed direction](metadata/qwen36-27b-mtp-live-reduction-direction-2026-10-10.json)
passes all directed/full-output and real-model numerical gates plus both
complete API panels. D2 observes 12.708/9.533 token/s at 8K/40K versus v28
12.377/9.337, with identical work and acceptance. D3 improves to 11.164/8.574
but remains slower than d2. The exact reduction mapping remains a bounded
development dependency through the next complete verifier's first API return
or archival. No repeat qualification or reduction/launch sweep follows. The
1.5x–3x objective remains active and unmet; the next architecture still needs
to remove substantial complete verifier cost, especially at 40K. Static
instruction counts neither establish a hardware limit nor project further gains.

## 2026-10-10 MTP register lookahead — closed without retention

`WP-MTP-REGISTER-LOOKAHEAD-20261010` / `AC-MTP-GREEDY-v26` responds to
bounded real-input FP8 long-scoreboard evidence and the recorded current-block
load/consume SASS. The [design](metadata/qwen36-27b-mtp-register-lookahead-design-2026-10-10.json)
keeps the 1.5x–3x goal and 18.45/21.62-second complete Decode budgets. One
fixed K5120/6144 executor issues the next raw weight record into registers
before consuming the current one. It retains current activations, the original
decoder/grid/FMA ownership, and the incumbent register-limited CTA floors.
This changes producer/consumer overlap without the rejected shared packed
ring's stores and per-tile barriers. No new layout, allocation or parameter
scan opens; counter ratios are not projected API gains.

The [MTP ledger](MTP_ADMISSION.md#register-lookahead-composition-v26) binds
finite-precision and tail safety. One implementation plus at most one correctness
repair passes static no-spill, full synthetic and P65 transaction admission,
then returns immediately to P65/8K/40K d2 API versus frozen v28. Greater-than-3%
regression stops; neutral/negative direction removes this executor. Useful
direction unlocks d3 and retains only a bounded dependency through the next
full verifier API return. Artifacts remain under
`.q3x-work/mtp-register-lookahead-20261010/`; production stays non-MTP.

The [completed direction](metadata/qwen36-27b-mtp-register-lookahead-rejection-2026-10-10.json)
passes 42 complete synthetic cases, full P65 numerical/transaction admission
and all eight d2 API/lifecycle requests. Work and acceptance counts match v28.
Decode observes 12.434/9.365 token/s at 8K/40K versus v28 12.377/9.337:
only +0.46%/+0.30% in one process, without measured noise separation. The 40K
external total is 0.222 seconds longer. This provides no useful retention
signal or statistically qualified speedup. All new runtime/profile/auditor
paths are removed, restoring v28; d3 is not run. No register, pipeline, grid
or qualification-only scan follows. The complete 1.5x–3x objective remains
active and unmet. Retained v28 mechanisms have now reached their next API
composition point; they remain development-only. Any successor must identify
a materially different complete verifier response and its API return budget.

## 2026-10-10 MTP projection lifetime composition — API return complete

`WP-MTP-PROJECTION-LIFETIME-20261010` / `AC-MTP-GREEDY-v25` composes
v27 with shared Gate/Up input publication and distributed FP8 finalization.
The [design record](metadata/qwen36-27b-mtp-projection-lifetime-design-2026-10-10.json)
binds the newly available component counters and the unchanged 1.5x–3x goal.
Original grids, packed weights, decoders and FMA ownership stay fixed. Gate
input publication removes three of four repeated global activation traversals
for up to three rows; all eight FP8 warps perform independent final reductions.
This is one coupled execution-lifetime response, not another packed-weight,
register-decoder or launch sweep. Neither counter ratios nor byte reductions
are claimed as API gains. The [MTP ledger](MTP_ADMISSION.md#projection-lifetime-composition-v25)
owns exact arithmetic and lifetime.

One composition plus at most one correctness repair passes complete synthetic
output/guard and real P65 transaction checks, then immediately returns to
P65/8K/40K d2 API against frozen v27. Greater-than-3% regression stops; neutral
or negative direction removes the bundle. Useful direction unlocks d3 and
retains only a bounded dependency through the next full verifier API return.
No allocation, inventory, numerical, reserve or production change is admitted.
Artifacts stay under `.q3x-work/mtp-projection-lifetime-20261010/`.

The [completed direction](metadata/qwen36-27b-mtp-projection-lifetime-direction-2026-10-10.json)
passes all numerical checks and both complete API panels. D2 observes
12.377/9.337 token/s at 8K/40K versus v27 11.984/9.090, with identical work
and acceptance. The exact execution changes remain a bounded development
dependency through the next complete verifier API return. No qualification-only
campaign, input-row/carveout/grid sweep or production switch follows. The
1.5x–3x goal remains active and unmet; the next architecture must address the
remaining complete verifier budget using the now-available bounded counters,
without treating local stall ratios as whole-API savings.

## 2026-10-10 MTP projection counters — complete

`WP-MTP-PROJECTION-COUNTERS-20261010` preserves the 1.5x–3x objective and
18.45/21.62-second complete Decode budgets. The prior full-model NCU attempt
exhausted replay-backup memory. One separate, source-identical component owner
loads only authenticated complete layer-0 QKV/Gate weights and retained real M3
inputs, avoiding that full-model backup footprint. The
[diagnostic design](metadata/qwen36-27b-mtp-projection-counters-design-2026-10-10.json)
permits one unprofiled correctness/timing process and one bounded root NCU
process with unchanged preflight/safety rules. No runtime mutation, local
retention panel, performance baseline or production change is selected by these
two cells. Close collection on success or resource failure, then use its
bounded attribution to select a materially different complete verifier response
and return promptly to the same d2/d3 API. No full-model replay retry or
layout/grid/decoder scan opens. Artifacts remain under
`.q3x-work/mtp-projection-counters-20261010/`.

The [completed diagnostic](metadata/qwen36-27b-mtp-projection-counters-2026-10-10.json)
obtains counters without resource failure. Both real-payload cells reproduce
all 96 captured elements and guards. Long-scoreboard waits dominate the sampled
stall ratios; no spills are observed, and Gate MIO throttle is negligible.
FP8 achieved occupancy is 37.29% versus theoretical 50%, including its
single-warp finalization. These component facts do not establish full-model
stall percentages or a bank-conflict cause. Collection is closed.

## 2026-10-10 MTP batched vocabulary finalization — API return complete

`WP-MTP-BATCHED-HEAD-20261010` / `AC-MTP-GREEDY-v24` composes one
previously coupled dependency with retained v25: exact full-vocabulary target
finalization across all verified rows. The earlier persistent-operand rejection
changed several projection families and did not isolate this dependency. The
matched 40K trace assigns 2.300 GPU seconds to target plus draft heads; even
eliminating that entire budget cannot alone close the unchanged 1.5x–3x target.
The [design record](metadata/qwen36-27b-mtp-batched-head-design-2026-10-10.json)
binds one complete API composition, not a new weight-decoder or layout scan.

All transformer projections remain v25. Complete logits reuse dead projection-0
storage and reach each immutable prefix slot before finite/argmax validation.
Original scalar arithmetic, resource reserve and committed-token accounting
remain mandatory. Full-vocabulary synthetic checks and complete P65 transaction
admission precede immediate P65/8K/40K d2 API; useful direction unlocks d3.
A greater-than-3% regression stops; neutral direction removes this version.
At most one correctness repair is permitted. Any useful result has bounded
prerequisite authority only through the next full verifier API return, never
a production or 50% claim. Artifacts stay in `.q3x-work/mtp-batched-head-20261010/`.

The [completed direction](metadata/qwen36-27b-mtp-batched-head-direction-2026-10-10.json)
passes complete numerical admission and all eight API/lifecycle requests for
both policies. D2 Decode observes 11.984/9.090 token/s at 8K/40K versus v25
11.630/8.874; d3 observes 10.299/8.034 versus 10.026/7.859. Identical acceptance
and verification counts accompany about 0.65–0.71 seconds less verification.
The isolated code retains this as a bounded dependency, not a statistically
selected baseline or production route. No further qualification-only campaign
or head/grid/draft-length scan is pending. The next complete verifier architecture
must compose this dependency at its first API return or archive it. The dominant
projection and long-context Attention budgets still need a materially different
execution response; the 1.5x–3x objective remains active and unmet.

## 2026-10-10 MTP combined immutable decoder — closed negative

`WP-MTP-COMBINED-CODEBOOK-20261010` / `AC-MTP-GREEDY-v23` composes the
ordered draft dependency with a different exact NVFP4 computation boundary.
The matched 40K verifier spends 8.845 GPU seconds in this family. Instead of
repeating two shared lookups and block-scale multiplication per weight, one
construction-owned 4,096-entry table materializes their exact finite-domain
product. Every original FMA and publication remains unchanged. This also removes
per-CTA table construction/barriers, charging read-only cache traffic and index
work to the complete API. It adds only 16,384 immutable device bytes, with no
request-time allocation or model layout change.

The [design record](metadata/qwen36-27b-mtp-combined-codebook-design-2026-10-10.json)
and [equivalence ledger](MTP_ADMISSION.md#combined-immutable-decoder-composition-v23)
bind one composition plus at most one correctness repair. Exhaustive decoder/
full-output guards and complete P65 transaction admission precede immediate
P65/8K/40K d2 API; useful direction unlocks d3. A greater-than-3% regression
against v25 stops early; neutral direction removes this version without a
lookup/layout/grid scan. The unchanged 1.5x–3x goal and 18.45/21.62-second
complete Decode budgets remain controlling. No hardware-stall claim or target
attainment follows from instruction counts. Artifacts stay under
`.q3x-work/mtp-combined-codebook-20261010/`. This is the ordered-draft dependency's
next complete API composition point; negative new decoding returns to the
independently positive v25 direction, without production selection.

The [combined-decoder rejection](metadata/qwen36-27b-mtp-combined-codebook-rejection-2026-10-10.json)
closes this composition after exhaustive decoder/consumer checks and complete
P65 numerical/transaction admission. P65/O16 d2 output and accounting match,
but Decode falls to 9.252 token/s versus v25 10.704. The early-stop gate
interrupts 8K; no 40K/d3 or complete lifecycle claim follows. All new table,
owner and consumer paths are removed. The independently positive v25 wiring
remains in isolated development, and its next API composition point is now
fulfilled. This result closes another weight-decoding substitution; a successor
must address the complete verification execution budget rather than open a
table/cache/layout scan. The 1.5x–3x objective remains active and unmet.

## 2026-10-10 MTP ordered draft composition — API return complete

`WP-MTP-DRAFT-ORDERED-COMPOSITION-20261010` / `AC-MTP-GREEDY-v22`
returns an earlier coupled dependency through the retained v7 target verifier.
The prior losing projection/Attention packages did not select the independent
whole-path effect of exact production ordered draft GQA. This is a complete
service wiring change, without another target kernel variant. The
[design record](metadata/qwen36-27b-mtp-draft-ordered-composition-design-2026-10-10.json)
binds the 3.258-second 40K draft budget and unchanged 18.45/21.62-second
complete Decode budgets. Draft savings alone cannot establish the 1.5x–3x goal.

One composition uses the existing fixed production operator on eligible draft
steps, with explicit scratch/device validation and actual hit receipts. Full
P513 transaction admission precedes immediate P65/8K/40K d2 API/lifecycle;
useful long-context direction unlocks the same d3 panel. Neutral/negative
composition is removed. A positive result is only a bounded prerequisite for
the next complete verifier architecture's API return, never a production or
50% claim. No local kernel/layout scan opens. Artifacts stay in
`.q3x-work/mtp-draft-ordered-composition-20261010/`.

The [completed direction](metadata/qwen36-27b-mtp-draft-ordered-composition-direction-2026-10-10.json)
passes the new same-input original-draft comparison, complete P513 numerical/
transaction admission and all eight API/lifecycle requests for each policy.
D2 observes 11.630/8.874 token/s at 8K/40K versus v7 11.560/8.636; d3 observes
10.026/7.859 versus 9.954/7.624. Target verification stays effectively unchanged.
The private wiring remains as a bounded dependency for the next complete
verifier composition, not a statistically selected baseline or production
route. No further qualification-only campaign or draft-length scan is pending.
The unchanged 1.5x–3x goal still selects the dominant verifier budget; the next
architecture must compose this dependency at its first real API return or
archive it. The negative layout, shared-residency and certificate-executor
lineages remain closed.

## 2026-10-10 MTP phase-borrowed FP8 input layout — closed without retention

`WP-MTP-PHASE-FP8-LAYOUT-20261010` / `AC-MTP-GREEDY-v21` preserves the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The earlier
startup-owned layout exchanged O/Down acceleration for input layouts; its
neutral result did not isolate input benefit. This composition instead borrows
the dead Prefill family arena after draft initialization and prompt-capture
expiry, retaining the complete original acceleration inventory. Its
[design record](metadata/qwen36-27b-mtp-phase-fp8-layout-design-2026-10-10.json)
and [lifetime ledger](MTP_ADMISSION.md#phase-borrowed-fp8-input-layout-composition-v21)
bind 5,200,936,960 packed bytes within 5,591,000,064 available bytes excluding
snapshots. Preparation and independent byte checking occur once per request,
are included in complete Decode, and receive a separate receipt. This changes
cross-phase ownership rather than a local layout or launch parameter.

One composition plus at most one correctness repair returns through packing,
range/lifetime, full P65 transaction and P8192 prefix admission immediately to
P65/8K/40K d2 API; d3 follows useful d2 direction. P65 reports the new fixed preparation cost but cannot alone reject its O256
amortization. A greater-than-3% 8K/40K regression stops early; neutral complete
long-context direction closes the version. No allocation, canonical-weight mutation, inventory removal,
reserve reduction or production switch is admitted. Artifacts remain under
`.q3x-work/mtp-phase-fp8-layout-20261010/`. The measured 5.454-second input
projection budget alone cannot guarantee the full target; only the complete
API can select this prerequisite.

The [phase-layout rejection](metadata/qwen36-27b-mtp-phase-fp8-layout-rejection-2026-10-10.json)
closes this composition after complete P65/P8192 numerical admission and all
eight d2 API/lifecycle requests. Preparation costs about 159 ms per request;
8K/40K verification is effectively unchanged from v7. Decode is 11.477/8.587
versus retained 11.560/8.636 token/s. The startup reserve passes with the full
inventory, but no useful API gain emerges. All new runtime/test/auditor paths
are removed. No d3 qualification, profile or layout/ownership scan follows.
The 1.5x–3x objective remains unmet; this closes the proposed cross-phase FP8
layout prerequisite rather than retaining it indefinitely. A successor needs
a different complete verifier computation/execution response, with the exact
publication boundary and prompt API return preserved.

## 2026-10-10 MTP compact full-K residency — closed negative

`WP-MTP-COMPACT-RESIDENCY-20261010` / `AC-MTP-GREEDY-v20` targets the
unchanged 1.5x–3x objective and 18.45/21.62-second complete Decode budgets.
The matched 40K profile assigns 13.321 GPU seconds to FP8 plus Gate/Up
verification. One complete composition stages compressed full-K records once
per CTA, then uses independent original consumer groups for each speculative
row. This changes both weight lifetime and accumulator ownership across both
families. It avoids the rejected full-K BF16 expansion and per-tile FP32
barriers, but charges repeated shared decode and large-block residency to the
API. Down, Attention and draft execution remain retained v7.

The [design record](metadata/qwen36-27b-mtp-compact-residency-design-2026-10-10.json)
and [MTP ledger](MTP_ADMISSION.md#compact-full-k-residency-composition-v20)
bind one fixed implementation plus at most one correctness repair. T1 full
output/guard checks, P65 full transactions and P8192 complete prefixes precede
immediate P65/8K/40K d2/d3 API return. Neutral/negative direction closes this
version without a shape/grid/stage scan; d3 follows only useful d2 direction.
No new allocation, inventory exchange or reserve reduction is allowed.
Artifacts remain under `.q3x-work/mtp-compact-residency-20261010/`.

The [compact-residency rejection](metadata/qwen36-27b-mtp-compact-residency-rejection-2026-10-10.json)
closes this version after all 42 synthetic output/guard cases and complete
P65/P8192 numerical admission. P65/O16 d2 output and accounting match, but
Decode falls to 7.459 token/s versus retained v7 10.717; verification rises
to 1.854 seconds. The early-stop gate interrupts the following 8K API request.
All new runtime paths are removed, with no 40K/d3 or full lifecycle claim.
Compact residency and lower registers do not select a faster runner. No further
shared-residency/row-ownership variant is active. The 1.5x–3x goal remains
unmet; a successor needs a different complete execution/computation response
and an immediate real-API return, preserving the exact publication contract.

## 2026-10-10 MTP startup-owned FP8 input layout — closed without retention

`WP-MTP-FP8-INPUT-LAYOUT-20261010` / `AC-MTP-GREEDY-v19` retains the
1.5x–3x objective and 18.45/21.62-second complete Decode budgets. The matched
40K trace assigns 5.454 GPU seconds to canonical FP8 input verification.
One startup-owned compact quad layout removes four scalar weight loads and
four index swizzles per K iteration in favor of the existing exact vector
consumer. It does not expand weight bytes. To retain the 8-GiB reserve, the
composition replaces FP8 O and NVFP4 Down auxiliary layouts with canonical
consumers, retaining NVFP4 Gate/Up. Only the complete API selects this trade.
The [design record](metadata/qwen36-27b-mtp-fp8-input-layout-design-2026-10-10.json)
and [MTP ledger](MTP_ADMISSION.md#startup-owned-fp8-input-layout-composition-v19)
bind its private ownership and exact arithmetic. The 5,200,936,960-byte owner
replaces 4,596,613,120 bytes, adding 604,323,840 bytes. No canonical payload or
Prefill operator is changed. A full BF16 expansion is not selected because it
doubles the repeated input-weight traffic.

One implementation plus at most one correctness repair passes exhaustive
pack/consumer T1, P65 full transactions and P8192 complete prefixes, then
returns directly to P65/8K/40K d2/d3 API. Negative direction closes this version
without a layout/grid scan. No local timing campaign or production switch is
opened. Artifacts stay under `.q3x-work/mtp-fp8-input-layout-20261010/`.

The [FP8 input-layout rejection](metadata/qwen36-27b-mtp-fp8-input-layout-rejection-2026-10-10.json)
closes this complete composition after exhaustive packing/consumer checks,
complete P65/P8192 numerical admission and all eight d2 API/lifecycle requests.
D2 Decode is 11.581 at 8K and 8.642 at 40K versus retained v7 11.560/8.636:
these single-process differences provide no useful gain or qualified speedup.
Startup retains 9,666,056,192 free bytes, but adds 604,323,840 bytes of inventory.
All new runtime/build/auditor paths are removed. No d3 qualification, layout
scan or additional profile follows. The 1.5x–3x objective remains active and
unmet; a successor must address the complete verifier execution budget rather
than repeat an isolated weight-layout substitution.

## 2026-10-10 MTP startup-owned pair layout — closed negative

`WP-MTP-PAIR-LAYOUT-20261010` / `AC-MTP-GREEDY-v18` addresses the unchanged
1.5x–3x objective and 18.45/21.62-second complete Decode budgets. The matched
40K profile assigns 8.845 GPU seconds to NVFP4 verification. The earlier pair
consumer still used quad-packed records: its warp's 256 useful weight bytes
span 16 aligned 32-byte regions. A pair-packed producer makes these eight
regions. This is an address fact, not measured DRAM traffic or a stall claim;
cache reuse may already serve the old consumers. The
[design audit](metadata/qwen36-27b-mtp-pair-layout-design-2026-10-10.json)
proves the lossless Gate/Up permutation and binds the source identities.

One composition replaces the old Gate/Up, Down consumer and scale6 startup
sidecars with one private MTP pair-packed NVFP4 owner, then binds exact M2..4
pair consumers. Canonical weights and scalar arithmetic remain intact; FP8
stays v7. The new owner is 9,625,927,680 bytes, replacing 9,000,632,320 bytes,
so actual composed startup must retain the unchanged 8-GiB reserve after the
625,295,360-byte increase. The private layout is never attached under an old
public layout identity. M1 and the scalar oracle consume canonical NVFP4.

The [MTP ledger](MTP_ADMISSION.md#startup-owned-pair-layout-composition-v18)
controls lifecycle and proof. One implementation plus at most one correctness
repair includes startup identity, full payload permutation checking and the
consumer before full P65 transaction/P8192 prefix checks and immediate
P65/8K/40K d2/d3 API. No local timing or launch/layout scan precedes that return.
A neutral/negative API composition closes the version. This is a bounded
attempt on the dominant projection budget, not a guaranteed target claim.
Artifacts stay under `.q3x-work/mtp-pair-layout-20261010/`.
The [pair-layout rejection](metadata/qwen36-27b-mtp-pair-layout-rejection-2026-10-10.json) closes the complete owner/inventory/consumer composition. Both full-size
T1 layouts/consumers, full P65 transactions and all P8192 prefixes pass. P65/O16
d2 API output and accounting match, but Decode is 10.361 token/s versus retained
v7 10.717, triggering the predeclared direction stop. P8192 API is interrupted;
no 40K/d3 or complete lifecycle claim follows. Startup retains 9,728,929,792
free bytes, so resource fit is not the rejection reason. All new runtime paths,
including the previously committed pack/check prerequisite, are removed.
The 1.5x–3x goal remains active and unmet. This closes the pair-layout version
without a layout/grid scan; the next package must change the complete verifier
execution architecture and return promptly to the same real API.

## 2026-10-10 MTP exact PV work elimination — closed without retention

`WP-MTP-EXACT-PV-ELISION-20261010` / `AC-MTP-GREEDY-v17` retains the
1.5x–3x objective and 18.45/21.62-second full Decode budgets. The matched
40K profile assigns 2.318 GPU seconds to ordered verifier PV. Shared KV
movement did not improve the runner. This materially different computation
eliminates an entire ordered block only when a runtime bound proves that every
FMA leaves its current FP32 accumulator unchanged. It changes no reduction,
probability, state or output rounding. The maximum measured PV budget alone
cannot close the full target; this is a bounded composed prerequisite, not a
redefined speedup goal. Projection, QK, draft and all other paths stay v7.

The [MTP ledger](MTP_ADMISSION.md#exact-pv-work-elimination-composition-v17)
controls the proof, exception handling and conservative cache-bound lifetime.
One fixed implementation plus at most one correctness repair passes directed
rounding/exception checks, full P513 transactions and P8192 complete prefixes,
then immediately returns to P65/8K/40K d2/d3 API. If the real-prefix checks
prove zero eliminated groups, close the mechanism without a timing campaign;
otherwise the normal negative API stop applies. No threshold, tile, numerical
bound or launch scan is opened. Persistent bounds borrow less than 177 KiB of
otherwise dead post-Prefill storage; the 8-GiB reserve stays fixed. Every API
receipt counts tested and eliminated PV groups. Artifacts remain under
`.q3x-work/mtp-exact-pv-elision-20261010/`.

The [exact-PV rejection](metadata/qwen36-27b-mtp-exact-pv-elision-rejection-2026-10-10.json) closes this composition after the one
initialization-cancellation repair, complete P513/P8192 numerical checks and
all eight d2 API/lifecycle requests. D2 Decode is 11.502 at 8K and 8.522 at
40K versus retained v7 11.560/8.636, with negligible actual elision. These
single-process differences establish no qualified regression or gain. All new
runtime paths are removed; no d3 qualification or bound/tile scan follows.
The 1.5x–3x goal remains active and unmet. A successor must address the dominant
complete verifier execution budget with a materially different architecture;
this closure selects no further local variant.

## 2026-10-10 MTP shared multi-query KV — closed negative

`WP-MTP-SHARED-KV-20261010` / `AC-MTP-GREEDY-v16` returns to retained
v7 projections after the certificate lineage's negative API results. The same
1.5x–3x goal and 18.45/21.62-second complete Decode budgets remain controlling.
The matched 40K profile assigns 3.077 GPU seconds to QK and 2.318 to PV.
Current interleaved CTAs enable cache reuse but load each speculative row's KV
separately. This composition physically stages K and V once for all rows.
QK partitions three heads per warp to bound query registers while all row/head
groups share a causal K tile. PV uses 32-dimension ownership and all-row
accumulators, keeping 32 CTAs rather than the old shared-V experiment's 16.
These choices follow the complete shared-KV resource/ownership contract;
there is no launch or dimension sweep. Ordered production draft GQA composes
at the same return. The original projection, recurrence and logit paths remain.

One fixed implementation plus at most one correctness repair passes full P513
prefix/transaction and P8192 complete-prefix admission, then returns immediately
to the same P65/8K/40K d2/d3 API panel. Short P65 cannot select a long-Attention
win; it only guards unaffected execution. Negative long-context direction
closes the composition. No new allocation, sidecar, mask workspace or reserve
change is introduced. The [MTP ledger](MTP_ADMISSION.md#shared-multi-query-kv-composition-v16)
controls exact arithmetic and causal tails. Artifacts remain under
`.q3x-work/mtp-shared-kv-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-shared-kv-rejection-2026-10-10.json) closes this version after full P513 and P8192
numerical admission. D2 completes all three context requests with identical
output/accounting, but Decode is 11.422 at 8K and 8.296 at 40K versus retained
v7 11.560/8.636. New shared-KV and ordered draft paths are removed. No d3,
qualification-only continuation or KV tile/ownership scan follows. The exact
causal transformation is valid but does not select a faster runner. The
1.5x–3x objective remains active and unmet; subsequent work must address a
materially different complete verifier execution architecture rather than
reopen the rejected packed-feed, certificate-executor or shared-KV variants.

## 2026-10-10 MTP partitioned execution and compact repair — closed negative

`WP-MTP-PARTITION-REPAIR-20261010` / `AC-MTP-GREEDY-v15` responds to
v17's matched 1.697-second fast execution plus 0.737-second repair costs. The
unchanged 1.5x–3x objective and 18.45/21.62-second full Decode budgets select
a complete two-stage scheduling change. Independent K128 fragment producers
replace each full-K serial CTA; a fixed balanced merge certifies each output
and compacts only uncertified elements into a bounded device worklist. Fixed
persistent repair workers consume that list, preparing codebooks once instead
of launching the entire possible output grid. The original scalar tree and
all certificate/exception rules remain unchanged. This composes the measured
fast and repair dependencies, rather than reopening a tile/grid scan.

One fixed composition plus at most one correctness repair passes exhaustive
pair/boundary and complete P65 state/logit checks, then returns immediately to
P65/8K/40K d2/d3 API in fail-fast order. Negative direction closes the version.
The partials, mask, activation preparation and worst-case worklist fit a 64-MiB
post-Prefill borrow; no new allocation, host count readback, reserve relaxation
or production change occurs. Ordered draft GQA composes at the same return.
The [MTP ledger](MTP_ADMISSION.md#partitioned-execution-and-compact-repair-composition-v15)
controls proof and lifetime. Artifacts stay in
`.q3x-work/mtp-partition-repair-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-partition-repair-rejection-2026-10-10.json)
closes this composition: exhaustive conversion, complete worklist/output and
real P65 state/logit/transaction checks pass, but Decode is 4.702 token/s
against retained v7 10.717. All new paths are removed, with no long-context/d3
qualification or further certificate/executor/worklist variant. The 1.5x–3x
objective remains unmet. Subsequent work returns to the retained projection
architecture and the measured multi-query Attention budget; rejected projection
compositions did not independently select or reject an Attention dependency.

## 2026-10-10 MTP direct fragment delivery — closed negative

`WP-MTP-REGISTER-CERTIFICATE-20261010` / `AC-MTP-GREEDY-v14` retains
1.5x–3x committed Decode and 18.45/21.62-second complete Decode budgets.
The rejected streamed executor still expands every weight into shared storage,
then reads it back into matrix fragments. This successor replaces that entire
operand path: one projection-wide guarded activation conversion publishes a
bounded FP16 array; warps load canonical packed weights directly into the
native MMA B-fragment coordinates and convert pairs in registers. No decoded
weight tile, codebook lookup or per-K producer/consumer barrier remains.
The pinned Marlin bit-field/pair conversion supplies the proven representation
principle, with full bias restoration before multiplication. The certificate,
zero-start K16 tree and exact sparse repair retain their numerical contract.
This source-derived movement/instruction hypothesis is not a hardware-stall claim.

One fixed implementation plus at most one correctness repair passes pair/code
and complete P65 state/logit admission, then returns immediately to the same
P65/8K/40K d2/d3 API in fail-fast order. Negative direction closes this version
without a launch, shape or bound scan. Ordered draft GQA composes at that return.
Activation/mask storage fits the existing 1-MiB post-Prefill borrow; no weight
arena, reserve reduction or production switch is introduced. The detailed
[MTP ledger](MTP_ADMISSION.md#direct-fragment-delivery-composition-v14) controls
ownership. Artifacts stay under `.q3x-work/mtp-register-certificate-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-register-certificate-rejection-2026-10-10.json)
closes this composition after complete numerical admission: P65 d2 Decode is
4.901 token/s versus retained v7 10.717. All new paths are removed. One bounded
same-ELF profile assigns 1.697 GPU seconds to direct fragments and 0.737 to
exact repair; input preparation is only 0.008 seconds. This is family-level
attribution, not a hardware-stall diagnosis. No long-context/d3 qualification or
feed/launch parameter scan follows. Any successor must address both the full-K
executor and repair scheduling, retaining the same exact publication proof
and prompt API return. The 1.5x–3x goal remains unmet.

## 2026-10-10 MTP streamed certified execution — closed negative

`WP-MTP-STREAMED-CERTIFICATE-20261010` / `AC-MTP-GREEDY-v13` responds
to the measured 4.875-second fast-generator cost in the rejected P65 composition.
The unchanged 1.5x–3x goal and 18.45/21.62-second full Decode budgets select
whole-K CTA ownership, vector packed-weight delivery and on-chip bounded-depth
reduction. The composition eliminates all global signed/absolute/error partials
and the separate certificate kernel. Native m16n8k16 fragments retain only the
observable first eight rows; a thread-owned shared binary stack limits the
arithmetic depth without excessive register retention. Vector repair consumes
four FP8 or eight NVFP4 codes per load, preserving the original scalar tree.
This replaces the complete producer/publication path, not its tile parameters.

The preceding certificate proof and exception rules remain unchanged; the
[MTP ledger](MTP_ADMISSION.md#streamed-certified-execution-composition-v13)
binds the new ownership. One fixed composition plus at most one correctness
repair returns through boundary and complete P65 state/logit checks directly to
P65/8K/40K d2/d3 API. Negative direction closes this version without a shape,
register, pipeline or certificate-bound scan. Ordered draft GQA composes at the
same return. Only a 1-MiB borrowed mask workspace remains. The 8-GiB reserve,
production default and numerical contract are unchanged. Artifacts stay in
`.q3x-work/mtp-streamed-certificate-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-streamed-certificate-rejection-2026-10-10.json)
closes this version: complete numerical/transaction gates pass, but P65/O16
Decode is 3.956 token/s against retained v7 10.717. All new runtime paths are
removed. There is no 8K/40K/d3 result or follow-on tile, pipeline or bound scan.
Eliminating global partials and reducing workspace did not select a faster
runner. The 1.5x–3x objective remains active; a successor must replace the
complete packed-to-fragment feed/compute architecture, account for repair,
and return promptly to the same API. No new implementation is selected here.

## 2026-10-10 MTP bounded reduction with sparse repair — closed negative

`WP-MTP-CERTIFIED-SPARSE-20261010` / `AC-MTP-GREEDY-v12` targets the
unchanged 1.5x–3x API objective and 18.45/21.62-second full Decode budgets.
The measured FP8 plus Gate/Up verification cost selects a new computation,
not another packed-feed/ownership variant. Native Tensor Core K16 reductions
start from zero, combine by explicit balanced FP32 trees within K128 partitions,
and publish signed, absolute-product and input-conversion-error partials.
An outward-rounded per-element certificate proves the original scalar BF16
publication; every uncertified element executes the original scalar tree.
Down retains its exact incumbent because its longer scalar chain makes this
certificate less useful. This differs from the closed coarse-norm feasibility:
it computes data-dependent absolute-product bounds, limits fast reduction depth,
and repairs individual elements rather than rejecting a whole M-row group.
No approximate target value is permitted to escape the certificate/repair pair.

The [MTP ledger](MTP_ADMISSION.md#bounded-reduction-and-sparse-repair-composition-v12)
owns the proof, exception handling and scratch lifetime. One implementation
plus at most one correctness repair must pass full P65 state/logit/transaction
admission and return immediately to the same P65/8K/40K d2/d3 API panel. Negative
API direction closes this version without a tile or certificate-parameter scan.
The ordered draft GQA dependency composes at that return. Scratch borrows at
most 64 MiB of dead post-Prefill GDN storage after snapshots and prompt capture;
no new allocation, smaller reserve or production switch is authorized.
Artifacts remain under `.q3x-work/mtp-certified-sparse-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-certified-sparse-rejection-2026-10-10.json)
closes this version: numerical gates pass, but P65 Decode is 2.283 token/s
versus retained v7 10.717. One same-ELF P65 profile attributes 4.875 GPU seconds
to fast partial generation, 0.178 to certificates and 0.927 to exact repairs.
The fast executor is the dominant measured cost; this is not proof that exact
certification itself prevents acceleration. All new runtime paths are removed.
No 8K/40K/d3 qualification or tile/bound scan follows. A successor must replace
the complete fast operand/execution path and account for sparse repair costs;
the 1.5x–3x objective and full-round budget remain unchanged.

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
