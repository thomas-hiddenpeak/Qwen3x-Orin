---
q3x_document:
  id: q3x-current-status
  class: active
  status: active
  owner: project-maintainers
  authority: current implementation, qualification, production, metric, and blocker snapshot
  effective: 2026-08-12
  last_reviewed: 2026-10-10
  supersedes: []
  superseded_by: []
  ssot_for: current delivered state and open production gaps
  review_trigger: any default route, capability, qualification, metric, release, or blocker change
---

# Qwen3x-Orin current status

Snapshot: 2026-10-10. This replaceable page owns current defaults and measured
qualification; [SDD](SDD.md) owns design and [Roadmap](ROADMAP.md) owns remaining
work. Historical failures and measurements retain their original scope.

## Production switch

**The corrected whole-core Prefill service is the ordinary mainline default.**
`orin-release` and fresh `BUILD_TESTING=OFF` builds select
`Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=ON`, package 0.8.1, profile
`q3x.sm87.production.whole-core-service.v1`, without a candidate/tactic selector.
The exact installed binary is
`5ee029cc8877ad0086e5e6ea5bcf8d7674626e55cd96098d518252d33c14aa33`.
The 0.8.1 [industrialization record](metadata/qwen36-27b-service-industrialization-2026-09-28.json)
binds the current artifact and reliability checks. Model arithmetic is unchanged.
The preceding 0.8.0 [production record](metadata/qwen36-27b-whole-core-service-production-2026-09-28.json)
binds the frozen source, artifact, installed consumer, independent numerical
references, capability, route, lifecycle and mirrored API checks.

The service supports nonempty text, chat and token-ID prompts, streaming with
or without usage, and nonstream responses. O is 1..4096 and
`P+O-1<=44095`; the arena is 9,508,218,624 bytes with aligned family scratch
for 44,096 rows. Padding never advances model state. One GPU worker and one
queued inference request bound execution. Greedy non-MTP generation and the
8-GiB retained-free reserve remain fixed. Non-loopback binding requires an
owner-only Bearer key file; TLS termination is external.

The installed service has no admission/tactic CLI, forced-token or state
inspection hook, cuBLAS/cuBLASLt dynamic dependency, or request-time projection
workspace growth. It exports the exact 0.8.1 geometry to installed consumers.
The diagnostic `qwen3x-orin generate` CLI retains its explicit legacy options;
use `qwen3x-eval-server` for this production route.

This completes the owner's **bounded Prefill production switch**, not the
future complete product: `production_eligible=true`, `release_qualified=false`.
60K/130K capacity, the locked 2s/4s TTFT targets and the approximately 8.55
Decode token/s convergence target are not claimed achieved.

## MTP development status

The current isolated source composes retained v7 target verification with
exact ordered draft GQA under `q3x.sm87.admission.mtp-draft-ordered-api.v25`,
ELF `75fedc5a94c6d381071b6bce8e247d54e49d1afb8d03e6e8e45b0288ee84a4e0`.
The [ordered-draft direction](metadata/qwen36-27b-mtp-draft-ordered-composition-direction-2026-10-10.json)
passes three same-input recursive draft steps with bitwise complete hidden/KV
and predictions, all P513 prefix/transaction checks, and eight API/lifecycle
requests for each draft length. All sixteen successful responses match the
non-MTP baseline output, usage and finish. No target arithmetic, allocation,
production artifact or accuracy boundary changes.

This is a bounded development prerequisite for the next complete verifier
composition, not a statistically selected performance baseline. On 2026-10-10,
one fresh process per policy reports the following same-request observations.
Prefill includes draft initialization; Decode includes all controller, draft,
verification, reconciliation and observer work, excluding the first token.

| P / O | Policy | Prefill s | Prefill token/s | External TTFT s | Decode token/s | External total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | d2 | 0.885 | 73.41 | 0.904 | 10.704 | 2.305 |
| 8192 / 256 | d2 | 17.244 | 475.07 | 17.266 | 11.630 | 39.192 |
| 40000 / 256 | d2 | 92.631 | 431.82 | 92.667 | 8.874 | 121.405 |
| 65 / 16 | d3 | 0.888 | 73.20 | 0.907 | 8.659 | 2.639 |
| 8192 / 256 | d3 | 17.332 | 472.65 | 17.354 | 10.026 | 42.789 |
| 40000 / 256 | d3 | 92.843 | 430.83 | 92.878 | 7.859 | 125.328 |

D2 40K draft work falls from the v7 observation of 3.258 seconds to 2.461;
target verification remains about 25.735 seconds. D2's observed ratios versus
the non-MTP anchor below are about 1.263x at 8K and 1.129x at 40K. **The
1.5x–3x objective remains unmet.** D3 remains slower than d2. No further
qualification-only run is pending; [Roadmap](ROADMAP.md) requires this dependency
to return in the next complete verifier architecture, while production remains
non-MTP. Earlier milestone and v7-anchor statements below describe their dated
closures rather than superseding this current isolated-code snapshot.

### Earlier milestones and v7 comparison anchor

The owner-authorized candidate uses configured draft lengths 2 and 3. The
[host milestone](metadata/qwen36-27b-mtp-foundation-2026-09-28.json) is followed
by a [native device milestone](metadata/qwen36-27b-mtp-device-2026-09-28.json):
authenticated BF16 draft weights, shared embedding/lm-head, the complete GPU
draft layer and scalar target prefix restoration now execute in a separate
non-installable harness. On a real P17/O16 fixture, natural lengths 2/3,
scripted acceptance/rejection and cancellation pass 11 complete bitwise target
state/draft-KV comparisons; two injected post-verification failures recover
through full reset. An independent CPU draft-layer oracle also passes its
bounded hidden/K/V checks. This is scalar correctness evidence, not production
Prefill, a multi-row weight-reusing verifier or an API acceleration result.

The subsequent [whole-core handoff](metadata/qwen36-27b-mtp-whole-core-2026-09-28.json)
uses current corrected Prefill through the engine, captures every normalized
prompt hidden row and initializes shifted draft KV. P65/O16 passes 11 complete
target-state/full-logit/draft-KV comparisons and two fault-recovery cases for
configured lengths 2 and 3. These private implementations are linked only into
test targets. The subsequent
[batched draft-cache milestone](metadata/qwen36-27b-mtp-batch-prefill-2026-09-28.json)
replaces serial full-layer initialization with exact weight-reusing FC/K/V
batches. Real P65 tests pass complete cache comparisons at 0/1/7/8/9/64 rows,
next-row canaries, two initialization cancellation/reset cases and the same
11 complete transaction comparisons plus two injected-failure recoveries.
The subsequent [multi-row verifier milestone](metadata/qwen36-27b-mtp-multirow-2026-09-28.json)
adds layer-major verification with exact FP8/NVFP4 weight reuse for M2/M3/M4.
Generic small-M projection arithmetic failed the full-state oracle because its
single accumulation chain differs from scalar Decode's four chains; dedicated
kernels preserve the original reduction and publication boundaries. P61/P65/P509
cover both causal Attention path transitions and pass 27 complete per-prefix
state/logit comparisons, 33 target-state/full-logit/draft-KV transaction checks,
six post-verification failure recoveries and six initialization-cancel recoveries.
These close the bounded verifier correctness prerequisite. The subsequent
[service direction record](metadata/qwen36-27b-mtp-service-direction-2026-09-28.json)
connects those components through the ordinary HTTP generation controller at
full configured capacity. Draft lengths 2 and 3 match the baseline response
text, usage and finish on eight requests each, including stream/nonstream,
text/chat and recovery after real Prefill/Decode disconnects. Startup retains
over 8 GiB free memory with the complete target acceleration inventory.

The preceding [draft matrix composition](metadata/qwen36-27b-mtp-draft-matrix-direction-2026-10-09.json)
restores the exact vector verifier and adds private BF16 Tensor Core drafting,
causal query scheduling and post-Prefill workspace borrowing. Its isolated
profile is `q3x.sm87.admission.mtp-multirow-api.v7`, ELF
`614b480dc69e6718e13b83d524944826833b91a234d2d52018f4cb014f81d78a`.
Both intervening weight-decoding replacements were slower and removed; the
[rejection record](metadata/qwen36-27b-mtp-shared-verification-rejection-2026-10-09.json)
retains every failure and frozen source. Fewer registers did not establish
whole-runner gains. The earlier
[v4 composition](metadata/qwen36-27b-mtp-efficiency-direction-2026-09-28.json)
remains a reproduction anchor, not the current isolated implementation.

The new private draft reduction passes independent FP64 hidden/K/V maximum
row-relative-L2 0.009382/0.001897/0.001300 against the original 0.02 bound.
Target verification retains exact scalar arithmetic. P65/P509 pass all nine
complete M2/3/4 prefix state/logit comparisons each; P65 additionally passes
11 full transactions, initialization cancellation, injected-failure recovery,
and exact cache replay with tail guards. Both policies pass eight API/lifecycle
requests with matched baseline text/usage/finish. Workspace borrowing removes
846,300,160 duplicate allocated bytes while preserving the 8-GiB reserve.

**The 1.5x–3x target remains unmet; production is unchanged.** The following
2026-10-09 observations use one fresh process per policy, not mirrored repeated
means. Prefill includes draft initialization; MTP Decode includes the complete
controller/observer interval, while non-MTP retains its step timer. External
total time is retained separately. These are bounded development observations,
not release qualification or statistically repeated speedup claims.

| P / O | Policy | Prefill s | Prefill token/s | External TTFT s | Decode token/s | External total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | non-MTP | 0.880 | 73.86 | 0.905 | 9.581 | 2.471 |
| 65 / 16 | MTP d2 | 0.886 | 73.36 | 0.905 | 10.717 | 2.304 |
| 65 / 16 | MTP d3 | 0.885 | 73.45 | 0.904 | 8.625 | 2.643 |
| 8192 / 256 | non-MTP | 16.750 | 489.06 | 16.773 | 9.212 | 44.458 |
| 8192 / 256 | MTP d2 | 17.290 | 473.81 | 17.312 | 11.560 | 39.371 |
| 8192 / 256 | MTP d3 | 17.279 | 474.09 | 17.301 | 9.954 | 42.920 |
| 40000 / 256 | non-MTP | 90.265 | 443.14 | 90.300 | 7.862 | 122.740 |
| 40000 / 256 | MTP d2 | 92.996 | 430.13 | 93.031 | 8.636 | 122.560 |
| 40000 / 256 | MTP d3 | 93.354 | 428.48 | 93.389 | 7.624 | 126.839 |

D2's observed Decode improvements are about 25.5% at 8K and 9.8% at 40K.
Its 40K total request remains effectively tied with non-MTP; d3 is slower.
Draft initialization falls to 0.54/2.63 seconds, but d2 verification still
takes 19.32/25.73 seconds. Complete 1.5x Decode budgets are approximately
18.45/21.62 seconds. This composition is retained as a bounded dependency
of the next architecture selected by [Roadmap](ROADMAP.md), not a generally faster
production policy or a reason to lower the target.

The later cooperative-warp, output-pair/shared-V, packed-FP8 and full-K
shared-operand compositions all failed their bounded API direction screens
and were removed. Their [cooperative](metadata/qwen36-27b-mtp-cooperative-projection-rejection-2026-10-09.json),
[output-pair](metadata/qwen36-27b-mtp-accumulator-ownership-rejection-2026-10-09.json),
[packed conversion](metadata/qwen36-27b-mtp-packed-fp8-rejection-2026-10-09.json)
and [full-K](metadata/qwen36-27b-mtp-fullk-residency-rejection-2026-10-09.json)
records preserve exact numerical passes, completed API observations and
interrupted runs. Reduced registers and fewer operand decodes did not select
a better runner. The v7 table above remains the retained implementation's
observation; no losing variant or uncomposed draft dependency remains active.

The [40K attribution](metadata/qwen36-27b-mtp-bottleneck-reset-2026-10-10.json)
subsequently measures 16.29 GPU seconds in verifier projections and 5.40 in
QK/PV after the first verifier. Hardware-counter collection failed; no specific
hardware-stall cause is established. The
[persistent-operand composition](metadata/qwen36-27b-mtp-persistent-operands-rejection-2026-10-10.json)
passes complete P65 state/logit and transaction checks but reduces P65/O16
Decode to 9.536 token/s, with Prefill 0.885 s / 73.41 token/s, external TTFT
0.912 s and total 2.485 s. Long-context testing was stopped and all new runtime
paths were removed. This is a negative direction screen, not a replacement
performance baseline or proof that the target is unattainable.

The later [certificate feasibility audit](metadata/qwen36-27b-mtp-certified-publication-feasibility-2026-10-10.json)
closes a coarse rounding-bound design before implementing a new GPU path:
it would still require extensive exact repair on the sampled real inputs.
The [independent-CTA composition](metadata/qwen36-27b-mtp-independent-ctas-rejection-2026-10-10.json)
passes complete numerical/transaction checks but regresses P65/O16 Decode to
7.509 token/s, with Prefill 0.885 s / 73.41 token/s, external TTFT 0.904 s and
total 2.901 s. It is removed; no new long-context result is claimed. The v7
retained implementation and unmet 1.5x–3x objective remain unchanged.

The [asynchronous packed-feed composition](metadata/qwen36-27b-mtp-async-packed-rejection-2026-10-10.json)
also passes complete P65 numerical and transaction checks but reduces P65/O16
d2 Decode to 10.308 token/s versus retained v7 10.717. Prefill is 0.885 s /
73.41 token/s, external TTFT 0.904 s and total 2.359 s. Testing stopped at this
negative direction; no new long-context or d3 result is claimed. The new paths
are removed and production is unchanged. The 1.5x–3x target remains unmet.

The later [bounded Tensor Core reduction/certificate/repair composition](metadata/qwen36-27b-mtp-certified-sparse-rejection-2026-10-10.json)
preserves complete P65 state/logits and exception/recovery checks but regresses
P65/O16 d2 Decode to 2.283 token/s. Prefill is 0.887 s / 73.26 token/s, external
TTFT 0.906 s and total 7.476 s. A matched diagnostic assigns 4.875 GPU seconds
to fast partial generation versus 0.178 to certification and 0.927 to exact
repair; it does not establish a hardware-stall cause. This version is removed,
with no long-context or d3 claim. Retained v7 and the unmet objective are unchanged.

The [streamed certificate successor](metadata/qwen36-27b-mtp-streamed-certificate-rejection-2026-10-10.json)
also preserves complete P65 numerical/transaction semantics but remains slower:
Prefill 0.887 s / 73.25 token/s, external TTFT 0.906 s, Decode 3.956 token/s,
and external total 4.697 s for P65/O16 d2. It is removed after this completed
negative screen; 8K is interrupted and no 40K/d3 result is claimed. On-chip
partial merging and smaller workspace are not sufficient for an API win.
Retained v7, the production default and the unmet 1.5x–3x objective are unchanged.

The [direct-fragment successor](metadata/qwen36-27b-mtp-register-certificate-rejection-2026-10-10.json)
passes exhaustive pair conversion and complete P65 state/logit/transaction
checks, but P65/O16 d2 remains slower: Prefill 0.887 s / 73.27 token/s,
external TTFT 0.906 s, Decode 4.901 token/s and external total 3.966 s.
A matched profile assigns 1.697 GPU seconds to fast fragments, 0.737 to exact
repair and 0.008 to input preparation. The version is removed; 8K is interrupted
and no 40K/d3 claim follows. Retained v7 and the unmet objective remain unchanged.

The [partitioned-execution/compact-repair composition](metadata/qwen36-27b-mtp-partition-repair-rejection-2026-10-10.json)
also passes exhaustive code, exact worklist and complete P65 state/logit/
transaction checks but does not beat v7: P65/O16 d2 Prefill is 0.895 s /
72.66 token/s, external TTFT 0.913 s, Decode 4.702 token/s and total 4.103 s.
It is removed after this screen; 8K is interrupted and no 40K/d3 result is
claimed. The certificate execution lineage is closed without a retained speedup.
The 1.5x–3x objective remains unmet; production and isolated v7 are unchanged.

The [shared multi-query KV composition](metadata/qwen36-27b-mtp-shared-kv-rejection-2026-10-10.json) passes complete P513/P8192
state/logit admission, but d2 API direction is negative. P8192/O256 Prefill is
17.232 s / 475.38 token/s, external TTFT 17.255 s, Decode 11.422 token/s and
total 39.580 s. P40000/O256 Prefill is 92.670 s / 431.64 token/s, external
TTFT 92.705 s, Decode 8.296 token/s and total 123.445 s. All three completed
context requests preserve baseline output and accounting. The new paths are
removed; d3 and the remaining lifecycle panel are not run. The retained v7
observations above remain the incumbent, and the 1.5x–3x goal remains unmet.

The [exact-PV rejection](metadata/qwen36-27b-mtp-exact-pv-elision-rejection-2026-10-10.json) passes repaired P513/P8192 complete
state/logit checks and all eight d2 API/lifecycle requests, but provides no
improvement over retained v7. P8192/O256 Prefill is 17.264 s / 474.50 token/s,
external TTFT 17.287 s, Decode 11.502 token/s and total 39.457 s.
P40000/O256 Prefill is 92.744 s / 431.30 token/s, external TTFT 92.779 s,
Decode 8.522 token/s and total 122.703 s. Only 1,568 of 303,595,008 tested
40K PV groups are eliminated. The new paths are removed; d3 is not run.
These are single-process direction observations, not a new performance baseline.
The retained v7 implementation and unmet 1.5x–3x goal remain unchanged.

## Numerical baseline decision

Legacy Prefill is retained as a regression comparator, **not model truth**.
Its per-token BF16 GDN rounding is not required by the checkpoint's FP32 SSM
contract. Independent direct FP32 recurrence and an independent all-layer
FP32 chunk implementation assess the vLLM/FLA reference rather than merely
assuming that framework is correct. The corrected native route has a separate
numerical identity; it does not claim Legacy bitwise equivalence.

The fast candidate had real defects: A/B `ldmatrix` coordinates were
transposed, and FP8/Gate-Up tensor scales were applied before the proper FP32
accumulation boundary. Those are repaired. At the matched P8192 Prefill
boundary, full-logit KL to the independent all-layer FP32 reference is
0.00800 for corrected native, 0.00957 for FLA, 1.57837 for Legacy, and 4.38283
for the original fast v3. At P40000, corrected native versus independently
pinned vLLM/FLA has full-logit KL 0.00733 and aggregate GDN relative L2 0.01251.
These are bounded error observations, not exact-logit or exact-token identity.
See the [reference repair](analysis/prefill-reference-repair-2026-09-28/README.md).

Ordered v7 Decode preserves the scalar FP32 arithmetic on identical inputs.
An independent FP64 check over layers 3/23/43/63 and P576/8192/40000 finds
relative L2 near 0.0016–0.0017, largely the BF16 output-publication floor.
At P40000 only 5/10/7/14 of 6144 BF16 outputs differ from rounded FP64 for those
layers. It is a validated bounded comparator, not an exact-real oracle.
Two later PV variants were slower and are closed, with their evidence in the
[capacity/baseline record](metadata/qwen36-27b-whole-core-capacity-baselines-2026-09-28.json).

The installed native and pinned vLLM APIs return identical answers on every
one of the predeclared 98 C-Eval questions: both score 79/98. Exact HTTP
request hashes and prompt-token counts match. This is all validation rows of
four subjects with five-shot direct answers, **not all 52 subjects** or proof
that every future generation must be identical.

## Paired Prefill and Decode performance

The retained 0.8.0 numerical-route qualification table reports means of two fresh installed processes in mirrored
B-C-C-B order against the frozen correct v9-r5 candidate. Each process uses
the same real prompt prefixes and request order, successful OS cache
preparation, no Prefix/KV reuse and no MTP. The GPU is exclusive and fixed at
1300.5 MHz; recorded temperatures, CPU/EMC state, cleanup and exact route gates
pass. Later requests in each process are not represented as first-after-startup
measurements. Engine Prefill is the server's pure interval; external TTFT is
POST to first nonempty token; Decode is (O-1) divided by its engine interval.

| Prompt / output | Prefill seconds | Prefill token/s | External TTFT seconds | Decode token/s |
| --- | ---: | ---: | ---: | ---: |
| 1 / 16 | 0.819 | 1.22 | 0.838 | 9.676 |
| 65 / 16 | 0.880 | 73.83 | 0.898 | 9.586 |
| 513 / 16 | 1.587 | 323.18 | 1.605 | 9.612 |
| 1,089 / 16 | 2.701 | 403.23 | 2.719 | 9.588 |
| 8,192 / 256 | 16.827 | 486.84 | 16.849 | 9.218 |
| 40,000 / 256 | 91.971 | 434.92 | 92.007 | 7.864 |

All paired bodies and generated outputs match, and installed Prefill and
Decode pass the predeclared no-material-regression (>3%) gate at every tested
length. This establishes preservation of the **correct** fast candidate's
performance, not a newly matched speedup against every historical Legacy
length. Historical P40000 Legacy+v7 observed 182.20 Prefill token/s and 7.87
Decode token/s; the corrected composition previously observed 443.28 and 7.87.
Those single-run values retain their older artifact/protocol authority.

The 40K Decode result remains below approximately 8.55 token/s. The old
approximate route's 8.55 observation is not promoted by this work.

## Retained 0.8.0 numerical-route service verification

- P44080/O16 reaches the last served state position; ordinary text/chat,
  streaming and nonstream responses pass on the same installed artifact.
- The P44095/O1 maximum prompt succeeds, including the final non-C64 tail.
- P40000/O4096 requests the full output ceiling; exact completion is recorded
  below, with no hidden truncation or capacity relaxation.
- The full 98-question public capability panel is parseable and matches the
  pinned reference answers; no score is assigned to truncated answers.
- Malformed input and capacity/sampling/token errors return HTTP 400; bounded
  queue overload returns 429. Prefill and Decode disconnects recover with
  identical subsequent output and usage. Owned shutdown returns zero.
- EvalScope 1.9.1 completes one warmup/eight measured requests plus an
  independent raw-SSE usage check. Its choices-empty usage parsing limitation
  is handled in the audit, not by changing the server's stream contract.
- The installed 0.8.0 package consumer, selector rejection, dependency and
  symbol checks pass. CMake guards reject mixed production/admission builds.

| Prompt / actual output | Prefill seconds | Prefill token/s | External TTFT seconds | Decode token/s |
| --- | ---: | ---: | ---: | ---: |
| 44,095 / 1 | 102.375 | 430.72 | 102.412 | N/A |
| 40,000 / 4,096 | 92.091 | 434.35 | 92.126 | 7.792 |

The long-output finish is `length` with 4,096 actual generated tokens.

## Build, deployment and rollback

Use the root [quick start](../README.md#functional-evaluation-quick-start) or
`cmake --preset orin-release`, build, then install under `.q3x-work/install/`.
An existing hand-configured cache may retain OFF; the ordinary preset explicitly
sets the production option ON. Consumers must request exact package 0.8.1.

The frozen correct candidate `f322f10` remains the numerical/performance
reproduction anchor. To reproduce Legacy, explicitly configure
`Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=OFF` and all admission/development
options OFF in a separate build directory; do not call that historical route
an independently qualified accuracy baseline. Operational rollback should
restore a retained complete binary/plan tuple, not mix old libraries with
new headers. Historical fixed-P40 and v10 profiles remain opt-in reproductions
with withdrawn numerical qualification.

## Non-performance product audit, 2026-09-28

The bounded non-MTP architecture assessment and ordered Down implementation
are complete. The [implementation record](analysis/ordered-down-direction-2026-09-28/README.md)
rejects both permitted dataflows: they preserve the bounded numerical panel
and compared API outputs but make 8K/40K Prefill slower. Candidate integration
was removed; only reproducible evidence remains. The default artifact,
qualified numerical route and performance baselines are unchanged. That
non-MTP package has no outstanding performance or qualification run; the later
MTP authorization is tracked separately above.
All seven findings in the frozen
[product audit](analysis/product-readiness-audit-2026-09-28/README.md) are closed
within the 0.8.1 delivery scope:

- Byte-fragment output caps serialize valid UTF-8 in stream/nonstream responses.
- Incomplete clients use bounded staging instead of occupying response workers.
- Production builds run six checks; empty CTest selection fails.
- Fatal CUDA/engine failures latch unhealthy and terminate for supervisor recovery.
- Versioned qualification/comparison tools bind exact requests, tokens and witnesses.
- Installed notices/licenses and an optional systemd template accompany the package.
- Multilingual, cancellation, capacity, sustained reuse and fresh-process checks pass.

The installed artifact passed 24 multilingual stream/nonstream pairs, 12 slow
incomplete clients, eight Prefill/Decode cancellation cycles, queue/auth/error
checks, the 98-question panel (79/98; all reference answers equal), and the
maximum prompt/output checks. The two-hour soak completed 1,708 requests;
RSS stayed 50,283,108 KiB, descriptors 46 and threads 8. All 1,889 successful
responses have complete validated route receipts. A fresh process completed
11 further requests, with identical baseline and slow-reader output/usage/finish.
Both processes exited zero. Six CTests and 36 Python unit tests pass; six old
runs / 144 witnesses also pass the strengthened request-identity audit.

The following are single-process 0.8.1 reliability observations, not a new
mirrored performance baseline or a speedup claim:

| Prompt / output | Prefill seconds | Prefill token/s | External TTFT seconds | Decode token/s |
| --- | ---: | ---: | ---: | ---: |
| 8,192 / 256 | 16.753 | 488.99 | 16.773 | 9.220 |
| 40,000 / 256 | 90.964 | 439.73 | 90.999 | 7.867 |
| 40,000 / 4,096 | 91.518 | 437.07 | 91.551 | 7.764 |
| 44,095 / 1 | 103.070 | 427.82 | 103.104 | N/A |

Qualification used one-second idle gaps during capability/soak and 60 seconds
between capacity cases. Peak temperature was 87.656C with no clock errors.
An earlier back-to-back maximum-load sequence reached 90.156C and correctly
stopped; it remains a failed run. Continuous maximum-load saturation is not
qualified by the successful cadence-limited soak. Fatal-error classification
is host fault-tested; destructive real-GPU fault injection and activation of
the supplied systemd unit were not performed. CI is configured, with local
checks executed; no remote CI result is claimed. Full future release gates
and performance targets remain open. Exact scope and hashes are in the
[industrialization record](metadata/qwen36-27b-service-industrialization-2026-09-28.json).

## Code and documentation audit, 2026-09-28

The [audit record](metadata/qwen36-27b-code-documentation-audit-2026-09-28.json)
reconciles build defaults, server/diagnostic entry points, memory-profile binding,
package ABI and numerical authority with the source and installed artifact.
The README now starts from the production API. Stale factory/ABI claims and
historical work-package authority are corrected; the CLI identity check now
covers the installed service explicitly. Retained records were re-audited,
not remeasured. Runtime changes in that earlier code/documentation audit were comments only, so the
qualification and paired performance above retain their original scope.

## Decode convergence snapshot 2026-09-27

This retained anchor now points to the
[performance retrospective](analysis/decode-performance-lessons-2026-09-27/README.md)
and [numerical erratum](analysis/whole-core-production-switch-2026-09-27/README.md).
The old normalized-error denominator diluted error with tensor size; captures
also compared different generated histories. Sampled KV, top-five logits,
first-token agreement and component timing never established full-model
qualification. Wrong-target builds could leave stale server ELFs. Current
checks bind full prompt-boundary tensors, independent reference arithmetic,
exact binary/profile hashes, actual usage and real API intervals. Those
lessons remain in force; old failure records describe their original artifacts.
