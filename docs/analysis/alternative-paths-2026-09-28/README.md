---
q3x_document:
  id: q3x-alternative-paths-20260928
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: source-based alternative assessment after ordered Down rejection
  effective: 2026-09-28
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: implementation or new measurements require a successor record
---

# Alternatives after the ordered Down rejection

Source snapshot: `89853c91a9cc5dc91503c420ba17e2f5def59bca`.
The owner's request to examine other paths is answered by this source review
and the existing installed-service profile. No new GPU process, runtime edit,
numerical qualification or performance result is claimed. MTP remains deferred.
The [source manifest](../../metadata/qwen36-27b-alternative-paths-2026-09-28.json)
pins the files and calculations used here.

## Product anchor

The [Down direction record](../ordered-down-direction-2026-09-28/README.md)
contains the latest unchanged-production B1 observations:

| Prompt / output | Prefill seconds | Prefill token/s | External TTFT seconds | Decode token/s |
| --- | ---: | ---: | ---: | ---: |
| 8,192 / 256 | 16.742 | 489.30 | 16.764 | 9.222 |
| 40,000 / 256 | 90.274 | 443.10 | 90.309 | 7.868 |

These are single-process direction observations, not a replacement for the
qualified table in [Current Status](../../CURRENT_STATUS.md). The existing
[installed profile](../non-mtp-architecture-assessment-2026-09-28/README.md)
is the diagnostic budget source. Neither Down variant remains a candidate.

## Recommended next path: preserve GEMM, remove output round trips

The current Gate/Up implementation runs separate CUTLASS GEMMs, publishes both
BF16 results, then launches SiLU-multiply. A less invasive alternative to a
paired-accumulator kernel is:

1. Keep Gate GEMM and its BF16 output unchanged.
2. Keep Up's input, weights, tile, stages, K order and FP32 accumulation.
3. Use CUTLASS's existing C-source epilogue interface to read the matching Gate
   value. Round `up_scale * up_accumulator` to BF16 **before** computing the
   existing `gate / (1 + expf(-gate)) * up`, then publish BF16 activation.

The source and destination iterators can carry separate leading dimensions;
Gate can retain its current `2*N` stride while activated output uses `N`.
The output operator must request the source explicitly. A generic alpha/beta
linear combination is insufficient: it would omit the intermediate Up BF16
boundary. Reuse the incumbent multiply/conversion semantics, including signed
zero and special values, rather than introducing an FMA with an added zero.

This removes only Up's intermediate write/read and the separate SiLU launch.
It does **not** eliminate Gate publication, either dot product, or all 29.454 s
of Gate/Up GEMM time. Across P40000 and 64 layers the removed logical traffic is
`40000 * 17408 * 2 bytes * 2 directions * 64 = 178,257,920,000 bytes`.
The 320 standalone SiLU kernels currently total 1.387 s, but their arithmetic,
Gate read and activation write move into the Up epilogue; 1.387 s is not a
guaranteed saving. Physical DRAM traffic and net latency have not been measured.

A second composition member applies the same principle to FP8 O projection:
round the scaled branch to BF16 in the epilogue, then compute the incumbent
`BF16(float(residual) + float(branch))` and publish residual directly.
The current drain already orders projection before the separate residual add.
This can remove 320 residual kernels (0.673 s total in the profile) and
`40000 * 5120 * 2 * 2 * 64 = 52,428,800,000` logical branch bytes.
Residual reads/writes remain. This is not CUTLASS `alpha*AB + C`: the branch
rounding must precede addition. In-place C/D ownership, aliases and all tail
rows require checking; trace modes that expose raw branch outputs cannot
silently take an elided path.

Both changes retain the incumbent GEMM mainloop and avoid introducing a second
accumulator set. They still add epilogue source loads and live registers; a
resource cliff or slower GEMM can outweigh the removed traffic. Compiler
resource/SASS checks and real API composition, not source reasoning, decide.
The combined logical deletion is 230.687 GB, **not** an additive speed forecast.

The historical [Marlin fused epilogue](../prefill-marlin-gate-up-fused-epilogue-2026-07-30/README.md)
used interleaved paired rows and a different reduction/writeback path. Its
result supports studying publication fusion but cannot qualify this CUTLASS
route. Decode already has Gate/Up/SiLU fusion; this proposal targets Prefill.

## Other paths and their limits

| Path | Source-backed opportunity | Disposition |
| --- | --- | --- |
| FP8 decode once per layer/phase | Five panels currently decode each weight five times at P40. 1,040 decodes can become 208; the existing 2.00261 s explicit decode budget gives at most 1.60209 s removed before added costs. | Useful second step; keep the current GEMMs. |
| Pair Gate and Up inside one GEMM | Could also remove Gate materialization and share A, beyond the one-sided epilogue proposal. | Defer: pairing/exchange and resource redesign are unnecessary for the first step. |
| Final-layer liveness | All layer-63 K/V stay live, while earlier Q/MLP outputs are dead for ordinary generation. | Valid narrow scope, approximately 2.15% of the ledger's arithmetic denominator; not a comparable latency promise. |
| Decode softmax exponential reuse | Current reference evaluates each exponential twice after the maximum reduction. | Bounded opportunity, not the main Decode solution. |
| Decode KV transpose or more sharing | Current QK already shares K across six heads; PV loads each owned V tile once for six warps, with contiguous 16-byte producers. | No demonstrated redundant full-KV load to delete; do not add a second cache merely on a bandwidth hypothesis. |
| More PV CTAs / independent sequence sums | Already rejected; independent sums also change the ordered arithmetic. | Remain closed. |
| GDN rewrite / launch-count reduction | Current Decode GDN+conv is only 1.756 ms/step; Prefill GPU kernels occupy nearly the entire phase. | Insufficient primary budget for the remaining gap. |

FP8 reuse is a lifetime change, not a global persistent BF16 cache. Linear
fill needs distinct QKV and Z buffers simultaneously (167,772,160 bytes total);
full-attention fill needs Q/K/V (146,800,640 bytes). O needs 62,914,560 bytes
after core execution. These fit in size within current family scratch, but
size alone is not an alias proof. Protect the live token-ID prefix; retire
fill weights before GDN core reuse; prepare O after core; retire it before MLP.
No decoded pointer may survive an intervening scratch writer. Short prompts
with one panel have no repeated-decode saving. All existing scale placement
and output bits remain mandatory.

Decode at 7.86835 token/s spends about 127.09 ms/step; 8.55 requires 116.96 ms,
so roughly 10.13 ms must be removed. The profile assigns 96.846 ms to
projections/fused work and 23.782 ms to attention (QK 9.906, PV 11.037,
softmax 2.838). These profiled figures explain scope; they are not added to
the unprofiled API interval. A projection-only solution needs roughly a 10.5%
reduction of that diagnostic family, or an attention-only solution roughly
43%, as conditional planning sensitivities.

Softmax reuse must preserve each thread's position sequence, the 256-thread
denominator reduction tree, `expf` and division semantics. Saving exponentials
in global scratch adds traffic; keeping up to 173 values/thread at capacity
risks registers/spills. Even removing the entire measured softmax budget would
not close the roughly 10 ms gap. No such kernel is implemented or timed here.

## Decision and bounded next implementation

Recommend Prefill output-publication fusion first, followed by FP8 preparation
reuse if it remains worthwhile. The new hypothesis removes actual work at
existing numerical boundaries; it does not reopen Down's consumer rewrite.
This is incremental convergence, not a claimed solution to the 2-second TTFT
target. A large-gap successor still needs the matched reference/dataflow and
arithmetic-class reconciliation required by the constitution.

Before implementation, activate one package for the two epilogues, with no
tile/stage sweep: prove the exact BF16 boundaries and source/destination
lifetime, compile-check resource changes, check real checkpoint layers and
tails, then compose promptly into an isolated service admission. Use at most
these two mutations plus one correctness/resource repair, and a 30-minute
real-model process budget before the first API direction decision. Return to
the existing P65/P8192/P40000 requests, reporting Prefill, TTFT and Decode
together. A negative composition closes it; a positive direction unlocks
complete state, service-envelope and mirrored installed-artifact qualification.
No production change, outstanding qualification campaign or implementation
activation is implied by this assessment alone.
