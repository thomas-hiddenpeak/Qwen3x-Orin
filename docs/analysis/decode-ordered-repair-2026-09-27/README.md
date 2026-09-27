---
q3x_document:
  id: q3x-decode-ordered-repair-20260927
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: bounded ordered Decode numerical repair and API evidence
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: immutable evidence; add a successor for later results
---

# Ordered Decode numerical repair

Source parent: `66f3916`. Owner request: finish remaining numerical work while
retaining the approximately 8.55 tok/s interim delivery goal. Work package:
`WP-DECODE-ORDERED-PIPELINE-20260927`, under the active
[Decode architecture](../../ROADMAP.md#2026-09-27-product-convergence--active).
The ordinary default and the public scalar comparator remain independent.

## Equivalence ledger

The v3 precision correction removed extra BF16 publication, but still changed
QK, softmax and PV reduction order. The successor changes operand ownership
while preserving the scalar finite-precision computation:

| Boundary | Preserved numerical operation | Changed movement |
| --- | --- | --- |
| QK | BF16 operands, FP32 products and the original 128/64/32/16/8/4/2/1 tree; scale 1/16 | v4 uses the original score kernel; v5 retains grouped queries across positions and reuses decoded K |
| Softmax | Original FP32 maximum/denominator reduction, `expf` and division | Existing probability arena is reused |
| PV | One FP32 accumulator per output, `fmaf` at positions 0 through S-1 in that order | Six query warps share a V tile; double-buffered `cp.async` stages the next tile |
| Publication | One final BF16 RNE conversion, including quiet-NaN handling | No partition result or intermediate BF16 probability |

Each CTA owns one KV head and 32 output dimensions. The 32 CTAs retain 6,144
independent ordered accumulators. Shared storage is 11,264 bytes; no atomics,
request allocation, runtime search, or tensorcore reassociation occurs. Each
buffer is read only after its asynchronous copies and CTA barrier complete;
it is reused only after every consumer reaches the next barrier. Partial
tiles consume valid positions only. The maximum 4,233,120-byte FP32 workspace
fits the existing request arena; the arena itself is unchanged.

The reference translation takes shared KV and asynchronous operand staging
from the studied FlashInfer dataflow, while retaining scalar reductions that
the tensorcore numerical class does not preserve. This is a numerical-contract
repair, not a reopening of the old unroll/partition parameter scans.

## Numerical result

The frozen scalar teacher-forced fixtures are unchanged. Every bucket captures
16 complete 248,320-element BF16 logit vectors, complete Conv/GDN state at the
first and final steps, and every changed K/V row. Byte comparison distinguishes
signed zero. The prior KL 0.001 and state relative-L2 0.01 alarms are retained;
strict equality is the stronger gate used here.

| Prompt tokens (v4 and v5 independently) | Raw spans equal | Attention calls exactly equal | Maximum KL | Maximum state-span relative L2 |
| ---: | ---: | ---: | ---: | ---: |
| 576 | 84/84 | 256/256 | 0 | 0 |
| 8,192 | 84/84 | 256/256 | 0 | 0 |
| 40,000 | 84/84 | 256/256 | 0 | 0 |

Each version passes all 768 Attention comparisons, repeats bitwise and remains
finite. Both also match the complete Prefill-commit/generation-return live-state
digests, normalized-hidden/residual/logit step records and generated text. Prefix observations,
forced inputs, and raw argmax match the scalar baseline. This closes the
previous bounded numerical failures without changing an acceptance threshold.
The test panel is finite evidence; the preservation argument above supplies
the mechanism, rather than treating equal generated text as proof.

Synthetic tests independently check the cancellation residual that rejected
v2, the FP64 oracle, tails, guards, invalid arguments, non-finites, and Graph
replay. A separate exact test compares every output and FP32 probability at
S512, 513, 575, 576, 577, 2047, 8192, 40000 and 44095. It passes all nine.

## Causal correction

The v4 API completes P40000/O256 with exact usage and SSE at 159.011193 ms/token,
or 6.28887 tok/s. It is numerically admissible but does not meet the interim
speed goal. The sole correction, v5, keeps six query vectors in registers
across 16 independent positions per warp and decodes each K once for the six
heads. At S40000, the QK grid becomes 1,252 CTAs instead of 120,000. These are
source-derived work/ownership facts, not measured physical DRAM savings.
Every score retains the original product/add tree, and softmax/PV are unchanged.
The kernels move into the dedicated admission translation unit, leaving the
public scalar implementation source unchanged. QK uses 72 registers and no
shared/local memory; PV uses 40 registers and 11,264 shared bytes, no local
memory or stack. The real-model observer now fails on any differing output bit.

The v4 capability phase did not run: its helper failed to locate the pinned
case file after the completed long-output request. This is a harness failure,
not an accuracy score or an invalidation of the completed request. The exact
original cases were copied and SHA-256 checked before the v5 run; the failed
v4 record remains preserved.

## API and delivery decision

The v5 P40000/O256 API completes 256 content events, correct usage and DONE
at **135.888518 ms/token, 7.358973 tok/s**. The pinned 20-case direct-answer
screen is 20/20 parseable and 15/20 correct; answers, text, request identities
and usage match the frozen scalar screen exactly. Malformed-input HTTP 400,
cancellation after three content events, subsequent identical recovery output
and health checks pass. Server shutdown returns zero. This is a bounded
screen, not full public or long-context capability qualification.

One NCU attempt on a fresh P40000/O1 request fails the device HWPM secure
profiling permission check. Its completed one-token response does not grant
counter or performance authority; the profiler exits after SIGINT with -2.
No QK/PV bottleneck, hardware ceiling, or pipeline-stage tuning decision can
be inferred from this failed collection. The raw error is retained.

The package closes with exact numerical repair retained in the noninstallable
admission build and **no production selection**: 7.36 remains below the
approximately 8.55 tok/s interim goal. The default public scalar source is
unchanged. OFF preprocessing of the two modified server files is identical
to the parent and OFF symbols exclude the new kernels. The rebuilt OFF ELF
has a different hash; no binary-identity or speedup claim is made.
A fresh OFF P1089/O16 integration request and the same 20-case screen plus
cancellation/recovery check pass; this is default-route integration health,
not an installed release qualification.

The [machine-readable closeout](../../metadata/qwen36-27b-decode-ordered-repair-2026-09-27.json)
retains artifact identities, numerical comparisons and request receipts. Numerical admission alone does not imply a
performance pass, production promotion, whole-core composition, or release
qualification. Prior v2/v3 evidence remains immutable.

Raw artifacts and the frozen protocol are retained under
`.q3x-work/decode-ordered-repair-20260927/`.
