---
q3x_document:
  id: q3x-decode-product-admission-20260927
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: bounded implementation, numerical admission, startup repair and API direction evidence
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: none; correct through a dated successor
---

# Decode product admission, 2026-09-27

This closes the first implementation batch of
`AC-DECODE-FUSED-GQA-PRODUCT-20260927`, based on `468b293`. It delivers a
separate compiled numerical admission and a bounded startup repair, **not a
new production Attention route or a completed product release**. Exact results,
source/build hashes, failed attempts and local artifact inventory are in the
[structured record](../../metadata/qwen36-27b-decode-product-admission-2026-09-27.json).
The [preceding retrospective](../decode-performance-lessons-2026-09-27/README.md)
remains unchanged.

## Implemented boundaries

The non-installable `Q3X_BUILD_FUSED_DECODE_ADMISSION=ON` build exposes
`qwen3x-eval-server-fused-decode-admission`. It requires testing, rejects other
admission/production compositions and the entire ambient `Q3X_` namespace,
and publishes its own unqualified profile, Decode route and witness schema.
It loads the complete ordinary production Decode sidecar inventory. The
ordinary binary retains scalar attention.

The internal Q24/KV4/D256 launcher uses 99,072 bytes of request-owned scratch
at S512..44095, the runner stream, fixed startup attributes and a deterministic
partition merge. It produces output only; the public reference API still
produces normalized probability scratch. The old scores/values split sources
are no longer linked or reachable through the public launcher. The ordinary
ELF contains neither their launcher symbols nor their environment selectors.
The startup identity's pre-existing self-reference initializer is also fixed.

The implementation specializes the pinned vendored FlashInfer kernel with
split-P high/residual BF16 operands. FP32 QK, online normalization and MMA
accumulation remain; partial and final output publication are BF16. No vendor
source, Prefill instantiation, weight format, KV precision or MTP boundary is
changed.

## Numerical admission and limits

The first BF16-probability version failed the predeclared independent FP64
cancellation fixture: relative L2 0.0111531933 exceeds 1/128. The one causal
split-P correction passes the same fixture and threshold at 0.00285950869.
Bounds, GQA mapping, six sequence lengths through 44095, partition tails,
output/scratch guards, eager/Graph repeatability and all-NaN V propagation pass.
An attempted compute-sanitizer run reports unavailable device debugging; it
is not a memory-sanitizer pass. Initial invalid LSE-argument and shared-offset
implementations remain recorded as failed attempts.

P1089/O16 observes every full-attention layer on both scalar-restored and
fused trajectories: 256 comparisons per process, exact repeated candidate
outputs and no non-finite values. Maximum same-input relative L2 is
0.00361330157346 on the scalar trajectory and 0.00353652683924 on the fused
trajectory. Four first-step layers (3, 23, 43, 63), heads 0 and 23, additionally
use an independent CPU FP64 softmax/PV reference. The candidate has larger
error than scalar on each of those sampled scalar-trajectory cells.

Both processes produce the same 16 token IDs and identical tiled Prefix
observations. Nevertheless **all 16 full-vocabulary logit hashes differ**, and
GDN, convolution and KV hashes differ after the final prompt step and at
return. Equal generated text is therefore not evidence of unchanged model
state or numerics. These naturally equal input trajectories are bounded
same-input evidence, not a general teacher-forcing harness. Full-logit hashes
detect differences without quantifying them. No public capability score,
long-context numerical qualification or changed production contract is granted.

## Startup resource repair

Two fresh ordinary starts fail the unchanged 256-MiB Graph increment gate at
362,614,784 and 335,777,792 bytes. A complete-layout fused start also fails.
Boundary instrumentation places the large increments at executable
instantiation rather than upload; inspected Graph kernels report no local
memory allocation. This does not identify a universal physical-memory cause.

The serialized runner now retains all 25 complete position plans but shares
one executable, using checked `cudaGraphExecUpdate` before each launch.
Reference-counted private ownership preserves public object layout, rollback,
reset and destruction. Standalone P1 behavior is retained. Position coverage,
390-node plans, the 256-MiB and one-second limits are unchanged.

The existing real-model Graph production test passes canonical replay, changed
embedding input, reset reuse, serial full-statistics/trace paths and the first
out-of-cache position. Preparation is 31.17 ms with a 2,818,048-byte free-memory
increment. Fresh ordinary API startup independently reports 38.39 ms and
4,669,440 bytes. These are scoped resource/correctness observations, not a
universal allocation bound or formal repeated-start reliability qualification.

## API return and qualification boundary

The ordinary and fused API cells use the same P19/P1089/P40000 token prefixes,
O16, LAZY loading, complete production sidecars and cache-drop procedure.
The structured record owns their rates, tail observations, text comparison and
cleanup. P19 exercises the short Graph route; the fused interval starts at 512.
The ordinary P1089/P40000 outputs match the corresponding first 16 events in
the retained pre-change investigation. The much older September 9 P40 text
is not used as the oracle for the subsequently changed mainline.

The first complete B/C observations are:

| Prompt / output | Ordinary Decode tok/s | Split-P admission tok/s |
| --- | ---: | ---: |
| 19 / 16 (outside fusion interval) | 9.7712 | 9.7661 |
| 1089 / 16 | 9.4125 | 9.6325 |
| 40000 / 16 | 5.1751 | 8.5545 |

At P40000, TPOT is 193.23 versus 116.90 ms. The generated text differs.
The direction survives corrected sidecar inventory and matched module loading;
it does not establish accuracy-preserving performance or reach 10 tok/s.

The installed ordinary artifact independently repeats all three O16 outputs and
usage exactly; its P40000 rate is 5.1774 tok/s. A fresh fused P40000/O256 run
completes 256 content events, usage 40000/256/40256 and normal stream termination
at 8.5518 tok/s, with p95 ITL 117.04 ms and maximum ITL 117.43 ms. This is one
Legacy long-output observation, not whole-core composition, output-quality
certification or general stability qualification.

The final witness schema is `target-prefill-witness-fused-decode-admission-v2`.
An audit caught the inherited `completed_exact_fallback_hits` field in the
unqualified v1 record. V2 reports `completed_fallback_hits_unqualified` instead,
with unchanged counts; a dedicated host test and fresh P1089 API smoke verify
the correction. Earlier v1 raw receipts remain unchanged and cannot qualify
exact fallback numerics. The final startup log also uses the admission profile
identity already published by health; the earlier O16 ready line still named
the configured Legacy profile alongside the correct fused Decode route.

This is an engineering direction comparison, not mirrored release timing.
CPU compilation overlapped part of the ordinary P40 Prefill, so no Prefill
speedup is attributed. Numerical differences remain unqualified regardless of
Decode speed. Legacy Prefill is used in these runs; the accepted whole-core
P40000/O16 profile has **not** been composed with this Decode admission.
Normal-output whole-core composition, public capability, comprehensive
teacher-forced logits/state, cancellation, installed fused qualification and the
10 tok/s target remain open.

## Additional errors prevented from guiding later work

- A first testing build omitted production Gate/Up and Down sidecars. Its
  startup memory and two completed short requests are unmatched diagnostics;
  its cancelled P40 request grants no timing authority. The build now requires
  the complete inventory. That owned child required recorded SIGKILL cleanup;
  the harness now escalates and waits for the exact child.
- Passing a short token/text check does not qualify full-model numerics.
- EAGER loading and smaller test inventories do not repair Graph ownership.
- The host protocol fixture still encoded whole-core P40000/O1, although
  `3c765164` changed the actual contract to O16 and its larger arena. The fixture
  is corrected; the contract is not weakened. The earlier failed host runs
  remain in the artifact inventory.

Reproduction helpers here retain preflight, sanitized telemetry, cache-drop
attempts, owned cleanup and exact request bodies. Raw outputs stay under
`.q3x-work/decode-product-20260927/`; the tracked structured record freezes
hashes and conclusions. The active dependency order remains owned by
[Roadmap](../../ROADMAP.md#2026-09-27-product-convergence--active).
