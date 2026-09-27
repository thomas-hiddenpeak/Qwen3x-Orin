---
q3x_document:
  id: q3x-current-status
  class: active
  status: active
  owner: project-maintainers
  authority: current implementation, qualification, production, metric, and blocker snapshot
  effective: 2026-08-12
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: current delivered state and open production gaps
  review_trigger: any default route, capability, qualification, metric, release, or blocker change
---

# Qwen3x-Orin current status

Snapshot: 2026-09-28. This replaceable page owns current defaults and measured
qualification; [SDD](SDD.md) owns design and [Roadmap](ROADMAP.md) owns remaining
work. Historical failures and measurements retain their original scope.

## Production switch

**The corrected whole-core Prefill service is the ordinary mainline default.**
`orin-release` and fresh `BUILD_TESTING=OFF` builds select
`Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=ON`, package 0.8.0, profile
`q3x.sm87.production.whole-core-service.v1`, without a candidate/tactic selector.
The exact installed binary is
`62178564d499e020a77d1753b43179fe7a6e3ad07cd9a206da153b7543368d58`.
The [production record](metadata/qwen36-27b-whole-core-service-production-2026-09-28.json)
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
workspace growth. It exports the exact 0.8.0 geometry to installed consumers.
The diagnostic `qwen3x-orin generate` CLI retains its explicit legacy options;
use `qwen3x-eval-server` for this production route.

This completes the owner's **bounded Prefill production switch**, not the
future complete product: `production_eligible=true`, `release_qualified=false`.
60K/130K capacity, the locked 2s/4s TTFT targets and the approximately 8.55
Decode token/s convergence target are not claimed achieved.

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

The table reports means of two fresh installed processes in mirrored
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

## Installed service verification

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
sets the production option ON. Consumers must request exact package 0.8.0.

The frozen correct candidate `f322f10` remains the numerical/performance
reproduction anchor. To reproduce Legacy, explicitly configure
`Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=OFF` and all admission/development
options OFF in a separate build directory; do not call that historical route
an independently qualified accuracy baseline. Operational rollback should
restore a retained complete binary/plan tuple, not mix old libraries with
new headers. Historical fixed-P40 and v10 profiles remain opt-in reproductions
with withdrawn numerical qualification.

## Non-performance product audit, 2026-09-28

Performance optimization is paused by owner direction. A subsequent
[product audit](analysis/product-readiness-audit-2026-09-28/README.md) reproduced
UTF-8 output-cap failure and HTTP worker starvation without loading the model,
and confirmed the release test preset runs zero tests. It also records
readiness failure-propagation, reproducibility, packaging and sustained-service
coverage gaps. These findings are open; the audit does not change runtime or
withdraw the earlier bounded numerical qualification. Follow the
[Roadmap](ROADMAP.md) for the revised work boundary.

## Code and documentation audit, 2026-09-28

The [audit record](metadata/qwen36-27b-code-documentation-audit-2026-09-28.json)
reconciles build defaults, server/diagnostic entry points, memory-profile binding,
package ABI and numerical authority with the source and installed artifact.
The README now starts from the production API. Stale factory/ABI claims and
historical work-package authority are corrected; the CLI identity check now
covers the installed service explicitly. Retained records were re-audited,
not remeasured. Runtime changes in this audit are comments only, so the
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
