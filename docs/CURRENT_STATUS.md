---
q3x_document:
  id: q3x-current-status
  class: active
  status: active
  owner: project-maintainers
  authority: current implementation, qualification, production, metric, and blocker snapshot
  effective: 2026-08-12
  last_reviewed: 2026-10-21
  supersedes: []
  superseded_by: []
  ssot_for: current delivered state and open production gaps
  review_trigger: any default route, capability, qualification, metric, release, or blocker change
---

# Qwen3x-Orin current status

Snapshot: 2026-10-21. This replaceable page owns current defaults and measured
qualification; [SDD](SDD.md) owns design and [Roadmap](ROADMAP.md) owns remaining
work. Historical failures and measurements retain their original scope.

## Production switch

**The corrected whole-core Prefill service is the ordinary mainline default.**
`orin-release` and fresh `BUILD_TESTING=OFF` builds select
`Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=ON`, package 0.8.2, profile
`q3x.sm87.production.whole-core-service.v1`, without a candidate/tactic selector.
The exact installed binary is
`6b65d472aa55584d7ab55ee7d618db54bee9c25f236f517879074573244bc93f`, installed under
`.q3x-work/install/softmax-sync-20261014/`. The [0.8.2 patch record](metadata/qwen36-27b-softmax-package-2026-10-14.json)
binds this artifact and its bounded corrective validation. No systemd unit or
running deployment was switched. The preserved 0.8.1 [industrialization record](metadata/qwen36-27b-service-industrialization-2026-09-28.json)
binds the predecessor and its two-hour reliability checks. Arithmetic is unchanged;
0.8.2 repairs shared-memory synchronization.
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
workspace growth. It exports the exact 0.8.2 geometry to installed consumers.
The diagnostic `qwen3x-orin generate` CLI retains its explicit legacy options;
use `qwen3x-eval-server` for this production route.

This completes the owner's **bounded Prefill production switch**, not the
future complete product: `production_eligible=true`, `release_qualified=false`.
60K/130K capacity, the locked 2s/4s TTFT targets and the approximately 8.55
Decode token/s convergence target are not claimed achieved.

## Softmax synchronization repair

The [2026-10-14 repair](metadata/qwen36-27b-softmax-sync-2026-10-14.json) fixes
a shared-memory lifetime race in MTP aligned softmax and three scalar/reference
sites. Every thread now finishes reading the reduced maximum before denominator
partials overwrite the shared array. Arithmetic, reductions, BF16 publication,
causal domains and state semantics are unchanged. CPU comparisons, root racecheck,
72 complete Attention/Graph cases and P513 full-state/transaction checks pass.
Prior ordinary bitwise equality did not prove absence of this race.

**The corrected 0.8.2 package is installed and its bounded checks are complete.**
All 184 patch requests and 11 fresh-process requests have complete route receipts.
The 98 pinned capability answers match the reference (79 correct); full capacity,
multilingual parity, cancellation, queue behavior and bounded reuse pass. The old
5ee artifact is preserved unchanged and contains the race; it is historical.
This does not renew two-hour reliability, select a performance improvement or
complete full release qualification.

The following same-request non-MTP patch observations are separate from the MTP
comparator below. Pure Prefill and Decode use engine intervals; TTFT and total
are external. Each row is one observation, with 60-second capacity spacing.

| P / O | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 8192 / 256 | 16.766 | 488.62 | 16.786 | 9.222 | 44.440 |
| 40000 / 256 | 89.991 | 444.49 | 90.026 | 7.866 | 122.446 |
| 40000 / 4096 | 90.175 | 443.58 | 90.209 | 7.779 | 616.645 |
| 44095 / 1 | 101.626 | 433.90 | 101.661 | N/A | 101.661 |

## MTP development status

### Additional Decode work opened

The owner has authorized a [new staged Decode goal](metadata/qwen36-27b-mtp-decode-next-design-2026-10-21.json)
against the exact installed v127 cap 2/3 artifacts below: additional 25%, then
50%, in all three pinned buckets. [Roadmap](ROADMAP.md#2026-10-21-additional-mtp-decode-gain--active)
owns the targets, 2% matched Prefill regression guard and bounded current-API
attribution before implementation. No new executor or performance gain is
qualified. The completed delivery metrics and ordinary non-MTP default below
remain authoritative; this new work does not reopen the old qualification panel.

The [current installed node profile](metadata/qwen36-27b-mtp-decode-node-budget-2026-10-21.json)
now closes short/40K attribution with exact output and work identity. Projection
work dominates both diagnostic windows, while Attention becomes material at
40K. The first driver stopped before its long request on its own orphaned
Nsight agent; one recorded long-only continuation completed. No unprofiled
performance baseline, runtime executor or production selection changed.

### Installed MTP delivery complete

The [final delivery audit](metadata/qwen36-27b-mtp-delivery-complete-2026-10-21.json)
selects both independent executable-only MTP packages for the bounded service
milestone. Clean source `65e13db` binds cap 2 SHA256
`74e7bb29383c6d62aa22c0ad746a1b2a8ca32821f32488e82795f45035cb04cf`
and cap 3 SHA256
`c4ab7583b1907ddfc4ded1ff3aa7696091610ffd211fb52cd70b39c87c7324ac`.
They are installed under `.q3x-work/install/mtp-package-20261020-cap2/` and
`.q3x-work/install/mtp-package-20261020-cap3/`. Ordinary non-MTP remains the
default; no running deployment is switched. Full future-product
`release_qualified` remains false.

The [exact installed six-process comparison](metadata/qwen36-27b-mtp-installed-repeat-2026-10-20.json)
passes both fixed caps across all three pinned API buckets, with no retries.
The conservative gain uses the slower MTP observation divided by the faster
non-MTP observation. Joint metrics below are medians of two fresh processes;
Prefill includes draft initialization, while TTFT and total are external.

| P / O | Configuration | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s | Conservative Decode gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | nonmtp | 0.888 | 73.17 | 0.907 | 9.582 | 2.472 | — |
| 65 / 16 | cap2 | 0.894 | 72.74 | 0.912 | 14.408 | 1.953 | +50.28% |
| 65 / 16 | cap3 | 0.899 | 72.30 | 0.918 | 15.264 | 1.900 | +59.13% |
| 8192 / 256 | nonmtp | 16.781 | 488.17 | 16.803 | 9.215 | 44.480 | — |
| 8192 / 256 | cap2 | 17.325 | 472.84 | 17.347 | 15.946 | 33.339 | +73.02% |
| 8192 / 256 | cap3 | 17.348 | 472.22 | 17.370 | 15.930 | 33.378 | +72.79% |
| 40000 / 256 | nonmtp | 90.064 | 444.13 | 90.099 | 7.863 | 122.533 | — |
| 40000 / 256 | cap2 | 92.667 | 431.65 | 92.702 | 12.018 | 113.921 | +52.57% |
| 40000 / 256 | cap3 | 92.698 | 431.51 | 92.733 | 12.009 | 113.968 | +52.53% |

Cap 2 short clears the floor by only 0.28 percentage points; this is bounded
workload qualification, not a universal prompt guarantee. MTP adds about
0.54–0.57 s of Prefill at 8K and 2.60–2.63 s at 40K versus non-MTP, while
reducing total time. Complete MTP IDs and physical work match the frozen
same-cap references; native historical IDs remain unavailable. All six
processes pass cache, ownership, fixed-clock, reserve and normal-exit checks.

Performance collection for that completed milestone is closed. The [remaining installed delivery panel](metadata/qwen36-27b-mtp-installed-qualification-design-2026-10-20.json)
now has [completed both-cap capability](metadata/qwen36-27b-mtp-installed-capability-2026-10-20.json):
98/98 native responses and pinned reference answers match per cap, with 79
correct answers and exact full MTP token sequences between caps. Both processes
exit normally. [Installed capacity](metadata/qwen36-27b-mtp-installed-capacity-2026-10-20.json)
also passes both caps: P40000/O4096, P44095/O1, over-capacity rejection and
short recovery, with exact cross-cap full token IDs and normal exit.
[Installed reliability](metadata/qwen36-27b-mtp-installed-reliability-2026-10-21.json)
now also passes: cap 2 completes 1,918 measured requests in 7,203.8 s and cap 3
completes 1,917 in 7,202.0 s, followed by successful fresh-process recovery.
Both have zero descriptor/thread growth; RSS growth is 1,688/2,052 KiB.
Every audited response and full MTP token sequence matches its pinned oracle.
This is P513/O16 with one-second gaps, not maximum-load saturation.

| Cap | Reliability Prefill s / token/s | TTFT s | Decode token/s | Total s |
| --- | ---: | ---: | ---: | ---: |
| 2 | 1.630 / 314.81 | 1.648 | 14.707 | 2.668 |
| 3 | 1.630 / 314.81 | 1.648 | 14.704 | 2.668 |

These are sustained-phase medians, not a new performance selection.
The [integrity audit](metadata/qwen36-27b-mtp-installed-integrity-audit-2026-10-21.json)
verifies 4,419 retained raw-file hashes and both installed executables. Two
historical mutable static archives were rebuilt; their original linked test
executable and logs remain intact. The record states this reproduction limit
without substituting current archives for historical inputs. All eight delivery
processes are complete. The final audit closes this MTP delivery scope.
Numerical authority remains independently qualified native references plus
complete same-input state/transaction checks; historical native output IDs
are unavailable, so response equality is not presented as native-ID equality.
No 200% gain, universal speedup, full 52-subject capability, same-process owner
recreation or future 60K/130K service qualification is claimed.

### Historical v127 architecture and packaging stages

The following records the completed intermediate stages; their pending wording
is historical. Installed delivery status is owned by the section above.

The [completed N-C2-C3-C3-C2-N selection](metadata/qwen36-27b-mtp-gdn-owner-repeat-2026-10-20.json)
passes all six fresh processes without retries. Both caps preserve exact full
MTP output IDs, response semantics, physical verification work, startup reserve,
fixed clocks, cold-cache preparation and normal exit. Every bucket clears the
predeclared worst-observed 1.5x floor. This selects the v127 architecture for
independent executable-only packaging; it does not switch ordinary non-MTP
production or qualify an installed MTP artifact.

Each row below is the median of two independent processes. Prefill includes
draft initialization for MTP; TTFT and total are external. The gain is the
slower MTP observation divided by the faster native observation, minus one.

| P / O | Configuration | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s | Conservative Decode gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | nonmtp | 0.888 | 73.18 | 0.907 | 9.580 | 2.473 | — |
| 65 / 16 | cap2 | 0.898 | 72.36 | 0.917 | 14.398 | 1.959 | +50.21% |
| 65 / 16 | cap3 | 0.894 | 72.72 | 0.912 | 15.286 | 1.894 | +59.46% |
| 8192 / 256 | nonmtp | 16.795 | 487.75 | 16.817 | 9.214 | 44.495 | — |
| 8192 / 256 | cap2 | 17.306 | 473.37 | 17.328 | 15.937 | 33.329 | +72.81% |
| 8192 / 256 | cap3 | 17.348 | 472.21 | 17.370 | 15.928 | 33.380 | +72.79% |
| 40000 / 256 | nonmtp | 89.936 | 444.76 | 89.971 | 7.863 | 122.404 | — |
| 40000 / 256 | cap2 | 92.599 | 431.97 | 92.634 | 12.009 | 113.870 | +52.49% |
| 40000 / 256 | cap3 | 92.771 | 431.17 | 92.807 | 12.021 | 114.021 | +52.80% |

Cap 2 short clears the floor by only 0.21 percentage points. These results
cover the pinned workloads and declared environment, not every future prompt.
MTP adds about 0.51–0.55 seconds of Prefill at 8K and 2.66–2.84 seconds at 40K
versus non-MTP, while reducing total time. Native historical token IDs remain
unavailable; direct native comparison covers full response/usage, with complete
MTP IDs compared to frozen v122 and numerical authority supplied by independent
native references plus full-state transaction checks. Installation, affected
delivery qualification and the final goal audit remain open.

Independent executable-only package implementation and [host admission](metadata/qwen36-27b-mtp-package-host-2026-10-20.json) now pass under
`orin-mtp-package-cap2` and `orin-mtp-package-cap3`. The new service identity
does not inherit the frozen candidate ELF's qualification; both installed
artifacts still require the [package delivery checks](metadata/qwen36-27b-mtp-package-design-2026-10-20.json).
The ordinary non-MTP release and default non-installable candidate remain
unchanged. The [clean installed API composition](metadata/qwen36-27b-mtp-package-api-2026-10-20.json)
now passes all 16 responses, complete output IDs, physical verification work,
auth/input checks, Prefill/Decode cancellation recovery and normal exit. Both
packages bind clean source `65e13db`; the record pins their distinct hashes.
The [final installed performance protocol](metadata/qwen36-27b-mtp-installed-repeat-design-2026-10-20.json)
has [passed](metadata/qwen36-27b-mtp-installed-repeat-2026-10-20.json), activating the [bounded remaining delivery panel](metadata/qwen36-27b-mtp-installed-qualification-design-2026-10-20.json).
Independent installed speedup is qualified above; complete delivery remains pending.

These installed API observations are one process per cap. Prefill includes
draft initialization; TTFT and total are external.

| Cap | P / O | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 65 / 16 | 0.894 | 72.68 | 0.913 | 14.385 | 1.956 |
| 2 | 8192 / 256 | 17.299 | 473.56 | 17.321 | 15.922 | 33.337 |
| 2 | 40000 / 256 | 92.601 | 431.96 | 92.636 | 12.003 | 113.882 |
| 3 | 65 / 16 | 0.895 | 72.61 | 0.914 | 15.289 | 1.895 |
| 3 | 8192 / 256 | 17.329 | 472.73 | 17.351 | 15.943 | 33.346 |
| 3 | 40000 / 256 | 92.562 | 432.14 | 92.597 | 12.015 | 113.822 |


### Preceding v127 direction and lifecycle evidence

The default-off [v127 complete GDN owner composition](metadata/qwen36-27b-mtp-gdn-owner-cap2-direction-2026-10-20.json)
passes native complete state/logit/draft-KV admission and the first sealed API
return for both caps; the [cap 3 result](metadata/qwen36-27b-mtp-gdn-owner-cap3-direction-2026-10-20.json)
completes the same three-bucket direction panel. It combines startup FP8 input ownership with selected GDN prefix
storage; every committed token receipts that ownership. All six requests
preserve full output IDs, response semantics and physical verification work.
Cap 2 startup free memory is 12,186,071,040 bytes; normal exit and thermal
checks pass for both artifacts.

| Cap | P / O | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 65 / 16 | 0.89337 | 72.76 | 0.91202 | 14.398 | 1.95380 |
| 2 | 8192 / 256 | 17.37361 | 471.52 | 17.39530 | 15.936 | 33.39714 |
| 2 | 40000 / 256 | 92.52087 | 432.33 | 92.55642 | 12.030 | 113.75438 |
| 3 | 65 / 16 | 0.89404 | 72.70 | 0.91279 | 15.281 | 1.89430 |
| 3 | 8192 / 256 | 17.28945 | 473.81 | 17.31121 | 15.922 | 33.32732 |
| 3 | 40000 / 256 | 92.56806 | 432.11 | 92.60288 | 12.000 | 113.85471 |

Prefill includes draft initialization; TTFT and total are external. These are
single-process direction observations. Short Decode is 1.50092x the frozen
same-request non-MTP comparator, only 0.00883 token/s above the gate; stable
1.5x remains unproven. The [affected ownership lifecycle panel](metadata/qwen36-27b-mtp-gdn-owner-lifecycle-complete-2026-10-20.json)
now passes normal/active destruction, five constructor failure points, and six
exact API responses plus auth/input/cancellation checks per cap. Every final
process exits normally. Earlier same-process reconstruction attempts remain
failed: successful CUDA release did not immediately restore reported free
memory, so the unchanged 8-GiB reserve rejected a new owner. Same-process owner
reconstruction is unqualified; service restart and surviving-runner recovery
are separately checked. The independent selection above is now complete. The retained default candidate is still
v122 pending separate packaging; this opt-in composition changes neither ordinary non-MTP production nor
installation, and does not complete the full goal.


The [completed two-cap sustained and fresh-restart panel](metadata/qwen36-27b-mtp-sustained-complete-2026-10-20.json)
passes two hours per frozen v122 cap, followed by independent fresh-process
recovery. Cap 2 completes 1,907 measured requests in 7,201 seconds; cap 3
completes 1,911 in 7,204 seconds. All 3,838 responses including baselines,
warmups and restarts match the full native response contract. Full MTP output
IDs match throughout and between caps; native historical IDs are unavailable.
All four processes exit normally. RSS grows 1,604/2,784 KiB for caps 2/3,
with no FD/thread growth; maximum temperatures are 77.343/76.062 C.

Measured P513/O16 medians follow; these are sustained-reuse observations,
not independent speed selection. Prefill includes draft initialization.

| Cap | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 1.631 | 314.52 | 1.650 | 14.431 | 2.689 |
| 3 | 1.628 | 315.10 | 1.647 | 14.441 | 2.685 |

Decode ranges are 14.271–14.439 and 14.321–14.446 token/s respectively.
This closes bounded sustained/restart validation. Cap 2 short still misses
1.5x; final independent selection and delivery qualification remain open.

The [completed two-cap capacity panel](metadata/qwen36-27b-mtp-capacity-complete-2026-10-20.json)
passes P40000/O4096, P44095/O1, P44095/O2 rejection and short recovery with normal
exit in both frozen v122 configurations. Complete native-reference text/usage
match; direct native output-ID comparison is unavailable in that historical
reference. All 4,113 output token IDs match between caps, and each short recovery
matches its previous same-cap MTP output IDs.

| Cap | P / O | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 40000 / 4096 | 92.674 | 431.62 | 92.710 | 12.745 | 414.003 |
| 3 | 40000 / 4096 | 92.461 | 432.61 | 92.496 | 12.745 | 413.812 |
| 2 | 44095 / 1 | 105.261 | 418.91 | 105.296 | N/A | 105.298 |
| 3 | 44095 / 1 | 104.015 | 423.93 | 104.050 | N/A | 104.052 |
| 2 | 65 / 16 recovery | 0.888 | 73.18 | 0.907 | 14.122 | 1.969 |
| 3 | 65 / 16 recovery | 0.888 | 73.19 | 0.907 | 14.976 | 1.908 |

These are one-process capacity observations per cap. Prefill includes draft
initialization; TTFT/total are external. Long-output Decode is 63.82%–63.84%
above the frozen same-request non-MTP 7.779 token/s observation, not a repeated
selection result. O1 has no Decode rate; cap 3's original derived 0 was corrected
to null without changing raw receipts or positive-token rates. Maximum observed
temperature was 79.968 C. No production switch or full-goal completion follows.

The [EOS-frontier v122 composition](metadata/qwen36-27b-mtp-eos-frontier-regression-2026-10-20.json)
is retained in the non-installable MTP candidate after its
[native state and initial API checks](metadata/qwen36-27b-mtp-eos-frontier-initial-direction-2026-10-20.json)
and bothcap remaining regressions. Profiles are
`q3x.sm87.candidate.mtp-eos-frontier-cap2.v122` and
`q3x.sm87.candidate.mtp-eos-frontier-cap3.v122`; the linked records bind exact
frozen binaries. This supersedes the preceding v119 sealed candidate in current
source, not the ordinary installed nonMTP0.8.2 production route.

All six affected EOS requests per cap preserve exact output and reduce target
verification from6 to4 rows and private head work from4 to3 steps. Decode gains
range15.67%–16.46% against frozen same-request MTP observations. These are
single-process direction observations, not a new comparison against nonMTP.
Across initial and regression panels,32 complete responses preserve exact token
IDs; authentication, invalid requests, cancellation/recovery and normal exit pass.
Native complete state/logit/draft-KV and EOS/cancel precedence evidence retains
its explicitly bounded scope. This does not repeat the full98-question suite.

The following v122 regression observations include draft initialization in
Prefill; TTFT and total are external. NoEOS representative and these three
buckets retain physical work and pass3% Prefill/Decode regression guards.

| Cap | P / O | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 65 / 16 | 0.888 | 73.21 | 0.906 | 14.140 | 1.967 |
| 3 | 65 / 16 | 0.888 | 73.17 | 0.907 | 14.978 | 1.908 |
| 2 | 8192 / 256 | 17.335 | 472.58 | 17.357 | 15.620 | 33.683 |
| 3 | 8192 / 256 | 17.339 | 472.47 | 17.361 | 15.606 | 33.701 |
| 2 | 40000 / 256 | 92.780 | 431.13 | 92.815 | 11.831 | 114.370 |
| 3 | 40000 / 256 | 92.718 | 431.41 | 92.753 | 11.831 | 114.308 |

Cap2 short remains below1.5x and has no EOS event. Final independent selection,
sustained delivery and production qualification remain open; capacity is closed above. Historical
v119 and original sealed evidence below retains its exact artifact scope.

The [sealed capability panel](metadata/qwen36-27b-mtp-sealed-capability-2026-10-20.json)
passes once per cap: 98/98 complete responses and reference answers match,
with 79/98 correct in each configuration. All output token IDs also match
between caps. The historical native reference contains full response text and
usage, but not output IDs; its numerical authority remains the independently
qualified non-MTP package, not the external answer score.

This panel covers P780..1197 with four output tokens. The following is its
first request, P849/O4, per cap; it is a capability observation, not a new
performance selection or a replacement for the three-bucket benchmark.

| Cap | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.278 | 372.62 | 2.312 | 18.421 | 2.480 |
| 3 | 2.213 | 383.58 | 2.247 | 18.439 | 2.415 |

Six requests per cap expose an unused verified suffix when an accepted EOS
ends generation. This is a new bounded efficiency finding, not a measured
optimization. Cap 2 short-performance closure and delivery qualification remain
open; production is unchanged.


The [sealed candidate composition](metadata/qwen36-27b-mtp-sealed-service-2026-10-20.json)
now runs both caps through complete `BUILD_TESTING=OFF` services. The
[cap 2 completion](metadata/qwen36-27b-mtp-sealed-cap2-2026-10-20.json) binds clean source
261bc9d and its independent artifact. Profiles are
`q3x.sm87.candidate.mtp-sealed-cap2.v1` and
`q3x.sm87.candidate.mtp-sealed-cap3.v1`; exact binary hashes and source identities
are in their respective records. It fixes the draft cap at build time,
contains no engine/runner test seams and rejects ambient draft selection.
Eight successful API requests per cap preserve v119 responses, token IDs and
physical work. Authentication, invalid requests, cancellation/recovery and cleanup pass.
The first attempt's hardcoded receipt identity was repaired; that interrupted
artifact has no candidate performance qualification.

One process per cap gives the following joint metrics. Cap 3 uses the repaired
receipt identity. Prefill includes draft initialization; TTFT and total are
external. These are build-composition regressions, not repeated performance
selection or production qualification.

| P / O | Cap | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | 2 | 0.892 | 72.90 | 0.910 | 14.129 | 1.972 |
| 8192 / 256 | 2 | 17.319 | 473.01 | 17.341 | 15.613 | 33.674 |
| 40000 / 256 | 2 | 92.675 | 431.62 | 92.710 | 11.848 | 114.233 |
| 65 / 16 | 3 | 0.892 | 72.89 | 0.910 | 14.986 | 1.911 |
| 8192 / 256 | 3 | 17.335 | 472.57 | 17.357 | 15.627 | 33.675 |
| 40000 / 256 | 3 | 93.064 | 429.81 | 93.099 | 11.844 | 114.630 |

The candidate remains noninstallable and reports `production_eligible=false`,
`release_qualified=false`. The ordinary non-MTP 0.8.2 service is unchanged.
Both fixed-cap build/API checks are complete. Cap 2's short-performance
deficit, repeated selection and delivery qualification remain open. Independent-process evidence below
belongs to the earlier v119 artifact and does not qualify this new binary.
The preceding [service-access boundary](metadata/qwen36-27b-mtp-service-boundary-2026-10-20.json)
has composed into this candidate.

### Cap 3 independent-process validation

The prospective N-C3-C3-N [independent-process validation](metadata/qwen36-27b-mtp-output-tail-repeat-2026-10-20.json) is complete.
Both frozen v119 cap 3 processes preserve exact response/token IDs, physical
work, route and resource checks. All four processes exit normally after fresh
preflight and successful cache preparation. No timing sample was retried.

Each performance row below is the median of two independent processes.
The last column is conservative: the slower MTP observation divided by the
faster non-MTP observation, minus one. Prefill includes draft initialization;
TTFT and total are external. These pinned three-bucket results do not claim
all prompts, all output lengths or production-release qualification.

| P / O | Configuration | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s | Conservative Decode gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | non-MTP | 0.886 | 73.33 | 0.905 | 9.592 | 2.469 | — |
| 65 / 16 | adaptive cap 3 | 0.891 | 72.93 | 0.910 | 14.983 | 1.911 | +56.17% |
| 8192 / 256 | non-MTP | 16.801 | 487.58 | 16.823 | 9.224 | 44.470 | — |
| 8192 / 256 | adaptive cap 3 | 17.330 | 472.71 | 17.352 | 15.615 | 33.682 | +69.28% |
| 40000 / 256 | non-MTP | 90.025 | 444.32 | 90.061 | 7.869 | 122.470 | — |
| 40000 / 256 | adaptive cap 3 | 92.853 | 430.79 | 92.888 | 11.839 | 114.430 | +50.42% |

Cap 3 clears the predeclared 1.5x floor in every repeated bucket. **The complete
objective remains open:** cap 2 short Decode is below 1.5x, and this bounded
cap 3 validation grants no production switch. The ordinary service stays
non-MTP 0.8.2. The tables below retain the preceding single-process direction
and cap 2 regression, rather than replacing this repeated evidence.


The preceding v119 performance artifact uses profile
`q3x.sm87.admission.mtp-output-tail-api.v119`, frozen server
`1dc8458a67231470344122c1f60a7aede6c199c7f50ce2b54b5cd37bc48e65dd`.
The [output-tail composition](metadata/qwen36-27b-mtp-output-tail-direction-2026-10-20.json)
keeps configured cap 2 unchanged and makes cap 3 adaptive: budget 2 ordinarily,
budget 3 only when four output tokens remain. Existing confidence stopping and
all arithmetic remain unchanged. On the short request it removes one whole
scalar target pass, changes rounds 7 to 6 and accepted drafts 8 to 9, and retains
19 verified rows and exact committed token IDs. Both long requests retain the
v113 d2 work counts; no tail extension is encountered there.

The following cap 3 observations are one process, not repeated selection.
Prefill includes private draft initialization; TTFT and total are external.
The comparator is the same-request non-MTP median linked in the preceding
v113 section below, not the slower historical fixed-d3 policy.

| P / O | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s | vs non-MTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | 0.891 | 72.91 | 0.910 | 14.999 | 1.910 | +56.39% |
| 8192 / 256 | 17.354 | 472.06 | 17.376 | 15.629 | 33.692 | +69.52% |
| 40000 / 256 | 92.720 | 431.40 | 92.755 | 11.838 | 114.298 | +50.52% |

The same frozen binary's configured cap 2 regression gives:

| P / O | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s | vs non-MTP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | 0.900 | 72.22 | 0.919 | 14.104 | 1.982 | +47.07% |
| 8192 / 256 | 17.302 | 473.48 | 17.324 | 15.622 | 33.648 | +69.43% |
| 40000 / 256 | 92.831 | 430.89 | 92.866 | 11.851 | 114.385 | +50.69% |

Host mixed-budget replay passes 373,026 checks. Native admission passes ten
complete target prefixes, 17 state/full-logit/draft-KV transactions and five
failure/reset recoveries. API responses, token IDs and lifecycle checks are
recorded for both caps in the linked direction record. Target and private
arithmetic are unchanged, so prior original-weight numerical admission retains
its existing scope. **The complete objective is not yet achieved:** configured 2
short Decode remains below 1.5x and cap 3 repeated selection does not close that gap. Production remains the installed non-MTP 0.8.2 artifact.

The [balanced cap 2 tail](metadata/qwen36-27b-mtp-balanced-tail-rejection-2026-10-20.json)
is closed after exact native admission and a short API result of 14.126 token/s,
below its 1.5x floor. Replacing M3 plus native M1 with two M2 rounds did not
close the budget. Integration and unused prerequisites are removed; v119 is
restored. This does not change the retained table or qualify long contexts.

The [Gate/Up scale-residency composition](metadata/qwen36-27b-mtp-gate-scale-residency-rejection-2026-10-20.json)
is closed after its one memory-contract repair. Complete numerical/state and
short output checks pass, but Decode falls to 13.825 token/s with unchanged
work. Executor, owner and integration are removed; v119 source and frozen
binaries are restored. This changes no retained metric or production route.

### Preceding v113 and closed directions

The following v113 metrics and closed lineages are historical comparators;
statements about their current source or pending return describe those stages.

The [complete private draft Graph](metadata/qwen36-27b-mtp-draft-step-graph-rejection-2026-10-20.json)
is closed neutral. Numerical/Graph/state checks pass, but d2 P65 Decode changes
only+0.0907% (14.1618 versus14.1490token/s), below the declared+0.1% gate.
Prefill is0.90115s (72.1304token/s), TTFT0.91995s and total1.97900s; output and
acceptance/work remain exact. The short stop closes before8K/40K/d3 or lifecycle
qualification. Runtime and unused prerequisites are removed, and frozen v113
artifacts are restored. The retained metrics below and production are unchanged.

The [fixed-budget tree composition](metadata/qwen36-27b-mtp-budgeted-branch-rejection-2026-10-20.json)
is closed after negative real API direction. Complete target-state/Graph checks
pass, but d2 P65 Decode falls from14.1490 to12.5256token/s with one fewer
accepted draft, one additional round and two additional verified rows. Output
and token IDs remain exact. The declared short stop ends before8K completion,
40K/d3 or lifecycle qualification. All tree runtime and unused prerequisites
are removed; source and frozen artifacts are restored to v113. No parameter
scan follows, and the full50%–200% objective remains unmet. These results do
not replace the retained v113 metrics below.

The [canonical singleton-state experiment](metadata/qwen36-27b-mtp-singleton-canonical-rejection-2026-10-20.json)
is closed neutral: complete state/output checks pass and one short request
elides 157,906,944 copied bytes, but Decode changes only +0.0098%, below its
predeclared gate. Its branch/counters are removed; v113 remains current.

The [compensated private-MLP integration](metadata/qwen36-27b-mtp-compensated-mlp-rejection-2026-10-20.json)
is closed without an API gain. Native full-layer and three-step recursive
numerical checks pass, but completed d2 short/8K Decode is 14.1319/15.6224
token/s (approximately -0.121%/-0.083% versus v113), with unchanged work.
The declared 8K stop ends this version before 40K/d3; integration and unused
executor are removed. This is a direction decision, not a statistically
qualified slowdown or a new performance baseline. Current v113 remains below.

The preceding v113 source retains the complete verifier Graph and adds a private
integer draft vocabulary projection. Profile is
`q3x.sm87.admission.mtp-integer-head-api.v113`, measured ELF
`4a2dac2971851bca6ada3111c423210af6f13362a28c2ab27dd89f761b339366`.
Canonical NVFP4 weights remain unchanged; blockwise signed integer dot products
consume quantized private activations, then restore the original scales and
publish BF16 logits. All vocabulary entries are evaluated. Target arithmetic,
state and committed output remain exact. The implementation reuses 6400 dead
scratch bytes and adds no weight copy or request allocation.

The [numerical admission](metadata/qwen36-27b-mtp-integer-head-native-2026-10-20.json)
passes all 65,536 packed code cases, integer dot checks, complete-logit guards,
changing-input Graph replay and bounded memory/race/synchronization checks.
Nine authenticated historical inputs have maximum full-vector relative L2
0.6592% against independent original-weight FP64, below the private 2% bound;
all nine argmax results match the scalar head. P65 passes ten complete prefixes,
fifteen transactions and four verifier-failure recoveries. Six hidden/KV capture
hashes equal v112. This numerical scope does not claim every future private
proposal or confidence value is identical.

The [API return](metadata/qwen36-27b-mtp-integer-head-api-2026-10-20.json)
passes all sixteen d2/d3 API/lifecycle requests, matching committed token IDs
and responses to v112/non-MTP. Short/8K acceptance and work are unchanged.
At 40K, each policy accepts one additional draft token, performs one fewer round
and verifies two fewer rows. Performance combines faster draft computation and
changed private proposals/confidence, rather than kernel execution alone.

The following 2026-10-20 candidate rows are **one process per policy**, not
repeated selection or release qualification. Non-MTP rows are the medians of
two fresh installed 0.8.2 processes from the
[preceding bounded validation](metadata/qwen36-27b-mtp-full-graph-repeat-2026-10-20.json).
Prefill includes draft initialization; TTFT and total are external. The final
column compares Decode with that same-request non-MTP median.

| P / O | Configuration | Prefill s | Prefill token/s | TTFT s | Decode token/s | Total s | vs non-MTP |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | non-MTP | 0.887 | 73.29 | 0.909 | 9.591 | 2.473 | — |
| 8192 / 256 | non-MTP | 16.783 | 488.10 | 16.806 | 9.220 | 44.465 | — |
| 40000 / 256 | non-MTP | 89.988 | 444.50 | 90.023 | 7.864 | 122.450 | — |
| 65 / 16 | d2 | 0.892 | 72.86 | 0.911 | 14.149 | 1.971 | +47.53% |
| 8192 / 256 | d2 | 17.298 | 473.58 | 17.320 | 15.635 | 33.630 | +69.58% |
| 40000 / 256 | d2 | 92.582 | 432.05 | 92.617 | 11.853 | 114.132 | +50.72% |
| 65 / 16 | d3 | 0.892 | 72.89 | 0.910 | 13.721 | 2.003 | +43.06% |
| 8192 / 256 | d3 | 17.345 | 472.30 | 17.367 | 14.675 | 34.744 | +59.16% |
| 40000 / 256 | d3 | 92.745 | 431.29 | 92.780 | 11.173 | 115.604 | +42.07% |

**The full-range 50%–200% objective remains unmet.** D2's 40K observation now
crosses 1.5x, but short Decode still needs approximately 17.903 ms removed against
the conservative faster non-MTP observation. D3 remains slower. Keep v113 as
a bounded constituent of the next complete architecture API return or archive
it; no head quantizer, geometry or profile scan follows. A repeated final
comparison remains necessary before goal or production claims. The ordinary
installed production service stays non-MTP.

The preceding v112 Graph increment was independently retained by the bounded
reverse-order validation linked above (0.38%–0.53% median Decode improvements
versus v101). Its exact state/Graph ownership remains active; its historical
metrics are preserved in that record rather than mixed into this candidate.

### Preceding v101 reference

The preceding v101 source retains the GDN graphs, Gate/Up input ring, fixed 0.4
lookahead, corrected softmax, exact short Attention, lazy accepted-tail KV
batching and native singleton transaction. Private draft Gate/Up/Down now use
fixed group128 INT8 weights decoded directly into BF16 MMA registers. FC,
complete Attention and draft-cache construction retain original BF16 weights.
Target arithmetic/state and the ordinary non-MTP service are unchanged.
The profile is `q3x.sm87.admission.mtp-mlp-register-api.v101`; measured server
ELF is `6d46918d3804e0db065529951292588abcb45c6cc96ae32b6ed9ddc2da215c09`.

The [register direction](metadata/qwen36-27b-mtp-mlp-register-direction-2026-10-14.json)
passes exhaustive decoder/full-projection/Graph checks and original-weight FP64
hidden maximum 1.369%, within the private 2% bound. Ten complete target prefixes,
fifteen transactions, failure recovery and all sixteen d2/d3 API/lifecycle
requests pass. Target IDs and responses match v98/non-MTP; private acceptance
changes are explicitly recorded. The additional 271,564,800 packed bytes retain
the startup 8-GiB free reserve and require no request allocation. This dependency
is bounded through the next complete MTP architecture API return or archival;
there is no mirrored selection or production promotion yet.

The corrected non-MTP denominator remains the
[fresh paired softmax record](metadata/qwen36-27b-softmax-sync-2026-10-14.json),
ELF `b320fc0bdf2d734591c180f1f3e71556f2ee4535fba373cb2bf68fbc9c459614`.
These 2026-10-14 MTP observations use one fresh process per policy. MTP Prefill
includes draft initialization; Decode includes the complete controller after
the first token. TTFT and total are external. Separate installed-patch timings
do not replace this denominator. Prefill variation is not attributed to the
unchanged Prefill implementation.

| P / O | Configuration | Prefill s | Prefill token/s | External TTFT s | Decode token/s | External total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | non-MTP | 0.895 | 72.60 | 0.914 | 9.591 | 2.478 |
| 8192 / 256 | non-MTP | 16.803 | 487.52 | 16.825 | 9.222 | 44.480 |
| 40000 / 256 | non-MTP | 90.109 | 443.91 | 90.144 | 7.872 | 122.542 |
| 65 / 16 | d2 | 0.892 | 72.83 | 0.911 | 13.968 | 1.985 |
| 8192 / 256 | d2 | 17.299 | 473.56 | 17.321 | 15.407 | 33.872 |
| 40000 / 256 | d2 | 92.722 | 431.39 | 92.757 | 11.620 | 114.704 |
| 65 / 16 | d3 | 0.892 | 72.87 | 0.911 | 13.513 | 2.020 |
| 8192 / 256 | d3 | 17.371 | 471.59 | 17.393 | 14.452 | 35.038 |
| 40000 / 256 | d3 | 92.818 | 430.95 | 92.853 | 10.954 | 116.134 |

D2 reaches 1.456x/1.671x/1.476x non-MTP Decode at P65/8K/40K; d3 reaches 1.409x/1.567x/1.392x.
**The full 1.5x–3x objective remains unmet.** At v101, d2 needed approximately
0.031 seconds removed from short Decode and 0.348 seconds from 40K Decode.
Product targets and numerical contracts remain intact; the ordinary installed
production service remains non-MTP.

The preceding [shared-tile private INT8 MLP direction](metadata/qwen36-27b-mtp-mlp-int8-native-rejection-2026-10-14.json)
passes independent original-weight FP64 and complete target-state admission,
but regresses P65/8K Decode by 1.09%/1.45% with unchanged acceptance. It is
removed before 40K/d3 qualification. BF16 Attention protects the fixed draft
numerical boundary, but this packed consumer does not improve whole-API fitness.
That rejected shared-tile consumer remains removed; the new register consumer
above is a distinct composition.

The subsequent [invariant-divisor direction](metadata/qwen36-27b-mtp-softmax-reciprocal-rejection-2026-10-14.json)
passes exhaustive division, full softmax/Graph, P513 state and eight d2 API
checks, but is removed for neutral whole-request performance: 8K/40K Decode
changes only +0.0346%/+0.0507%. D3 and further profiling are not run. This
closes that arithmetic-hoisting path; its v98 comparator is preserved in the
dated record.

The later [private PV partition](metadata/qwen36-27b-mtp-private-pv-split-rejection-2026-10-14.json) passes recursive
original-weight FP64 checks through position44095, but gives neutral complete
API results: 8K/40K Decode changes +0.0688%/-0.0528% versus v101. The candidate
is removed at its declared long stop before d3; only long-capture and independent
recursive numerical tools remain. Current runtime and joint metrics above stay
v101. No full-goal or production claim follows.

The subsequent [certified QK executor](metadata/qwen36-27b-mtp-qk-certified-api-rejection-2026-10-14.json)
preserves complete numerical/state/output checks but lowers 8K Decode by 0.654%
versus v101. It is removed at the declared 8K stop before 40K/d3/lifecycle tests.
The complete-product arithmetic proof remains valid; it does not select this
executor. Current runtime, joint metrics and unmet full objective above remain
unchanged. No production switch or further certificate/parameter scan follows.

### Preceding NV live-reduction direction

The following dated closures preserve their original scope; they do not
supersede the current implementation above.

The preceding isolated source extends the exact live-ancestor reduction to
NVFP4 Gate/Up, Down and batched target vocabulary outputs, on top of retained
v30. Mainloops and numerical/state boundaries remain unchanged. Profile is
`q3x.sm87.admission.mtp-nv-live-reduction-api.v31`, ELF
`0049caf79b62156af12b1b0b6d193df23def2283a6f9184b20b81b78ee2183b8`.
The [completed direction](metadata/qwen36-27b-mtp-nv-live-reduction-direction-2026-10-10.json)
passes 30 complete output/guard cases, full P65 prefix/state/logit and
transaction/recovery admission, and all eight API/lifecycle requests per
policy. All sixteen responses match the non-MTP baseline, with unchanged
work and acceptance counts versus v30. Production is unchanged.

These 2026-10-10 observations use one fresh process per policy, not mirrored
means or release qualification. Prefill includes draft initialization; Decode
includes the complete controller/observer interval after the first token.

| P / O | Policy | Prefill s | Prefill token/s | External TTFT s | Decode token/s | External total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | d2 | 0.887 | 73.27 | 0.913 | 11.902 | 2.173 |
| 8192 / 256 | d2 | 17.308 | 473.31 | 17.330 | 12.914 | 37.077 |
| 40000 / 256 | d2 | 92.699 | 431.51 | 92.734 | 9.660 | 119.133 |
| 65 / 16 | d3 | 0.885 | 73.41 | 0.904 | 9.842 | 2.428 |
| 8192 / 256 | d3 | 17.303 | 473.45 | 17.325 | 11.310 | 39.871 |
| 40000 / 256 | d3 | 93.117 | 429.57 | 93.152 | 8.665 | 122.582 |

D2 verification falls by approximately 0.316/0.348 seconds at 8K/40K against
v30; d3 falls by 0.291/0.313 seconds. D2 Decode ratios versus the non-MTP
anchor below are about 1.402x/1.229x. **The 1.5x–3x objective remains unmet.**
D3 remains slower than d2. The small consistent direction is retained only as
a bounded development dependency through the next complete verifier API return
or archival, not statistically qualified architecture selection. No further
reduction mapping scan or qualification-only campaign is active. Complete
Decode still exceeds the 18.45/21.62-second minimum-target budgets; next work
must address the remaining verifier cost, particularly at 40K.

The subsequent [softmax-lifetime direction](metadata/qwen36-27b-mtp-softmax-lifetime-rejection-2026-10-10.json)
passes complete FP32 probabilities, P513/P8192 state/logits and eight d2 API
checks, but is removed without retention. At 8K, Prefill is 17.311 s / 473.23
token/s, external TTFT 17.333 s, Decode 12.984 token/s and total 36.973 s;
at 40K, 93.277 s / 428.83 token/s, 93.312 s, 9.724 token/s and 119.537 s.
These single-process Decode differences of +0.55%/+0.67% versus v31 have no
noise qualification, and 40K total is 0.403 seconds longer. Prefill code is
unchanged; that interval difference is not attributed to this edit. D3 is not
run. The retained v31 table, production and unmet goal remain unchanged.

The subsequent [NV register-feed direction](metadata/qwen36-27b-mtp-nv-register-feed-rejection-2026-10-10.json)
also passes numerical and eight d2 API/lifecycle checks, but is removed. At 8K,
Prefill is 17.278 s / 474.13 token/s, external TTFT 17.300 s, Decode 12.698
token/s and total 37.382 s. At 40K, these are 92.631 s / 431.82 token/s,
92.666 s, 9.527 token/s and 119.434 s. Single-process Decode differences are
-1.67%/-1.38% versus v31 with unchanged work and acceptance; total adds
0.305/0.301 seconds. This is direction rejection, not statistical regression.
D3 is not run. Retained v31 metrics and the unmet 1.5x–3x goal are unchanged.

The later [independent-chain direction](metadata/qwen36-27b-mtp-chain-partition-rejection-2026-10-10.json)
passes 30 complete output/guard cases and full P65 state/logit/transaction
checks, but is removed after P65/O16 d2 regresses to 9.691 Decode token/s.
Prefill is 0.887 s / 73.26 token/s, external TTFT 0.913 s and total 2.461 s.
Output and work match the baseline, but verification adds 287.332 ms versus
v31. The following 8K request is interrupted; 40K/d3 and the complete API
lifecycle panel are not run. Retained v31 and the unmet goal remain unchanged.

The subsequent [direct-PV direction](metadata/qwen36-27b-mtp-direct-pv-rejection-2026-10-10.json)
passes complete P513/P8192 state/logit checks but is removed after negative
P8192/O256 d2 API direction. Prefill is 17.334 s / 472.60 token/s, external
TTFT 17.356 s, Decode 11.750 token/s and total 39.058 s. Work and acceptance
match v31, but verification adds 1.950 seconds and Decode falls by 9.01%.
The run stops before 40K, d3 and the full API lifecycle panel. Retained v31,
production and the unmet 1.5x–3x goal remain unchanged.

The subsequent [paired Gate/Up direction](metadata/qwen36-27b-mtp-gate-pair-rejection-2026-10-10.json)
passes complete numerical admission and all eight d2 API/lifecycle requests,
but is removed without retention. At 8K, Prefill is 17.302 s / 473.46 token/s,
external TTFT 17.324 s, Decode 12.915 token/s and total 37.069 s; at 40K,
92.650 s / 431.73 token/s, 92.685 s, 9.656 token/s and 119.096 s respectively.
Single-process Decode differences of +0.014%/-0.045% versus v31 establish no
useful gain or statistical regression. Output and work are unchanged; d3 is
not run. Retained v31, production and the unmet 1.5x–3x objective are unchanged.

The subsequent [batched cache-reconciliation direction](metadata/qwen36-27b-mtp-batch-reconcile-rejection-2026-10-10.json)
passes complete state/cache and eight d2 API checks but is removed without
retention. At 8K, Prefill is 17.298 s / 473.57 token/s, TTFT 17.321 s,
Decode 12.993 token/s and total 36.947 s; at 40K, 92.779 s / 431.13 token/s,
92.813 s, 9.694 token/s and 119.118 s respectively. Reconciliation saves
131/125 ms, but single-process Decode differences of +0.61%/+0.36% and a
nearly unchanged 40K total do not select useful whole-runner retention.
Work/output match v31; d3 is not run. Retained v31, production and the unmet
1.5x–3x objective remain unchanged.

The subsequent [activation-publication direction](metadata/qwen36-27b-mtp-activation-publication-rejection-2026-10-10.json)
passes complete numerical admission but is removed after negative P65/O16 d2
API direction. Prefill is 0.888 s / 73.18 token/s, external TTFT 0.914 s,
Decode 10.737 token/s and total 2.311 s. This is 9.78% slower Decode than
retained v31, with identical output and work; verification adds 136.775 ms.
The run stops before 8K, 40K and d3. Retained v31, production and the unmet
1.5x–3x objective remain unchanged.

### Preceding FP8 live-reduction direction

The preceding isolated source uses exact live-ancestor FP8 reductions on top of
retained v28. Four output trees share warp lanes as their live leaf count falls;
every original ordered addition and publication remains intact. Profile is
`q3x.sm87.admission.mtp-live-reduction-api.v30`, ELF
`222c174fe9c9b7f76fab1f15f4a68d635fb907f85da7cf08bc789dfe6a6adea1`.
The [completed direction](metadata/qwen36-27b-mtp-live-reduction-direction-2026-10-10.json)
passes 2,048 directed tree cases (four roots, two reduction stages), 42 complete
projection output/guard cases, full P65 prefix/state/logit and transaction
admission, and all eight API/lifecycle requests for each policy. All sixteen
responses match the non-MTP baseline; work and acceptance counts match v28.
The installed production artifact remains unchanged.

These 2026-10-10 observations use one fresh process per policy, not mirrored
means or release qualification. Prefill includes draft initialization; Decode
includes the complete controller/observer interval after the first token.

| P / O | Policy | Prefill s | Prefill token/s | External TTFT s | Decode token/s | External total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | d2 | 0.887 | 73.27 | 0.913 | 11.721 | 2.193 |
| 8192 / 256 | d2 | 17.309 | 473.28 | 17.331 | 12.708 | 37.398 |
| 40000 / 256 | d2 | 92.991 | 430.15 | 93.026 | 9.533 | 119.777 |
| 65 / 16 | d3 | 0.888 | 73.19 | 0.914 | 9.701 | 2.460 |
| 8192 / 256 | d3 | 17.339 | 472.46 | 17.361 | 11.164 | 40.202 |
| 40000 / 256 | d3 | 93.448 | 428.05 | 93.483 | 8.574 | 123.227 |

D2 verification falls by approximately 0.54/0.56 seconds at 8K/40K against
v28; d3 falls by about 0.75/0.79 seconds. D2 Decode ratios versus the non-MTP
anchor below are about 1.379x/1.213x. **The 1.5x–3x objective remains unmet.**
D3 remains slower than d2. The 40K Prefill interval is about 0.46% longer in
both observations, offsetting some Decode savings in total request time; no
Prefill implementation changed. This is a bounded development dependency for
the next complete verifier API return, not statistical architecture selection
or production promotion. No further reduction mapping scan is active.

### Preceding projection lifetime direction

The preceding isolated source composes exact shared Gate/Up activation publication
and distributed FP8 final reductions with retained ordered draft and batched head.
Profile is `q3x.sm87.admission.mtp-projection-lifetime-api.v28`, ELF
`3891e1a0db7947de5f72501ab069f6383bced7cafec7f9a8cbf29d9f1e87fc54`.
The [completed direction](metadata/qwen36-27b-mtp-projection-lifetime-direction-2026-10-10.json)
passes 42 complete synthetic output/guard cases, full P65 prefix/state/logit and
transaction/recovery admission, and eight API/lifecycle requests per policy.
All sixteen responses match the non-MTP baseline; work and acceptance counts
match v27. The installed production artifact remains unchanged.

These 2026-10-10 observations use one fresh process per policy, not mirrored
means or release qualification. Prefill includes draft initialization; Decode
includes the full controller/observer interval after the first token.

| P / O | Policy | Prefill s | Prefill token/s | External TTFT s | Decode token/s | External total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | d2 | 0.887 | 73.24 | 0.913 | 11.429 | 2.226 |
| 8192 / 256 | d2 | 17.314 | 473.15 | 17.336 | 12.377 | 37.938 |
| 40000 / 256 | d2 | 92.569 | 432.11 | 92.604 | 9.337 | 119.917 |
| 65 / 16 | d3 | 0.887 | 73.29 | 0.913 | 9.377 | 2.513 |
| 8192 / 256 | d3 | 17.295 | 473.66 | 17.317 | 10.810 | 40.907 |
| 40000 / 256 | d3 | 93.023 | 430.00 | 93.059 | 8.353 | 123.589 |

D2 verification falls by approximately 0.67/0.74 seconds at 8K/40K against
v27. Its Decode ratios to the non-MTP anchor below are about 1.344x/1.188x.
**The 1.5x–3x objective remains unmet.** D3 remains slower than d2. This is a
bounded development dependency for the next complete verifier API return,
not a statistical architecture selection or production switch. The
[component diagnosis](metadata/qwen36-27b-mtp-projection-counters-2026-10-10.json)
obtains hardware counters for two real-input projection cells; its scope does
not establish whole-model stall percentages or a hardware performance ceiling.

The subsequent [register-lookahead direction](metadata/qwen36-27b-mtp-register-lookahead-rejection-2026-10-10.json)
passes full numerical and all eight d2 API checks but is removed without
retention. It observes only +0.46%/+0.30% Decode differences at 8K/40K versus
v28 in one process, with no noise qualification and a 0.222-second longer 40K
request. At 8K, Prefill is 17.292 s / 473.76 token/s, external TTFT 17.314 s,
Decode 12.434 token/s and total 37.823 s; at 40K, 92.874 s / 430.69 token/s,
92.909 s, 9.365 token/s and 120.140 s respectively. These rejected-candidate
observations do not update the retained v28 table. D3 is not run, production
is unchanged, and the 1.5x–3x objective remains unmet.

### Preceding batched-head direction

The preceding isolated source adds exact batched target vocabulary finalization to
the retained ordered-draft composition, profile
`q3x.sm87.admission.mtp-batched-head-api.v27`, ELF
`22b7ae45b8bb19fe602b6a5026f60d7d7989dbc7209a8ca2b8f426d94508f83e`.
The [batched-head direction](metadata/qwen36-27b-mtp-batched-head-direction-2026-10-10.json)
passes full-vocabulary synthetic equality, complete P65 prefix/state/logit and
transaction/recovery checks, and eight API/lifecycle requests for each draft
length. All sixteen successful responses match the same non-MTP baseline.
Transformer kernels, numerical boundaries and the production artifact remain
unchanged. Existing dead scratch supplies batched logits without an allocation.

One fresh process per policy on 2026-10-10 reports the following direction
observations. Prefill includes draft initialization; Decode includes the full
controller/observer interval after the first token. These are not mirrored
means, statistical architecture selection or release qualification.

| P / O | Policy | Prefill s | Prefill token/s | External TTFT s | Decode token/s | External total s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | d2 | 0.894 | 72.70 | 0.913 | 11.076 | 2.267 |
| 8192 / 256 | d2 | 17.225 | 475.60 | 17.247 | 11.984 | 38.525 |
| 40000 / 256 | d2 | 92.626 | 431.85 | 92.661 | 9.090 | 120.715 |
| 65 / 16 | d3 | 0.894 | 72.68 | 0.913 | 8.901 | 2.598 |
| 8192 / 256 | d3 | 17.292 | 473.76 | 17.314 | 10.299 | 42.074 |
| 40000 / 256 | d3 | 92.783 | 431.12 | 92.818 | 8.034 | 124.559 |

D2 verification decreases by approximately 0.65/0.68 seconds at 8K/40K with
identical work and acceptance counts. D2 Decode ratios versus the non-MTP
anchor below are about 1.301x/1.156x. **The 1.5x–3x objective remains unmet.**
D3 remains slower than d2. This useful direction remains a bounded development
dependency for the next complete verifier composition, with no production
switch or qualification-only campaign. [Roadmap](ROADMAP.md) owns its expiry.

### Preceding ordered-draft direction

The preceding isolated source composes retained v7 target verification with
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
sets the production option ON. Consumers must request exact package 0.8.2.

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
