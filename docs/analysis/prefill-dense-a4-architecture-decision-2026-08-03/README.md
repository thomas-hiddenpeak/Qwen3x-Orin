# Prefill dense-A4 architecture decision (SM87)

Status: active implementation decision after the real-weight architecture
stop gates on 2026-08-03.  This document changes no production selector by
itself.

## Decision

Keep the locked production route unchanged while building one new dense-A4
projection plane.  The implementation has two coupled parts:

1. shape-specific operand-resident kernels: an R1 Down `M256N128` performance
   upper-bound cell and an R4 Down `M192N128` quality cell, each presenting a
   complete B panel to more tokens without a grid barrier or cross-CTA
   partials; and
2. a separately versioned, equalized scale-lane sidecar which can remove the
   repeated K512/K256 FP32 scale epochs without dropping any weight values.

This is not another tile/cache/stage scan.  A cell is retained only if it fits
the complete projection-plane dataflow below.  The first performance authority
remains a real checkpoint request through the OpenAI-compatible API driven by
external EvalScope.

MTP remains excluded.  cuBLASLt remains a comparison oracle and has no
production eligibility.

## Locked whole-request budget

The authority is the retained real P1853 request:

| Item | Locked time |
| --- | ---: |
| server Prefill | 1,911.30 ms |
| all GPU kernels | 1,911.120448 ms |
| Gate + Up | 722.001952 ms |
| Down | 288.796640 ms |
| Attention projections | 404.839904 ms |
| everything outside those projections | about 495.481952 ms |

The 2,000-token/s P1853 budget is 926.50 ms.  If the current non-projection
time did not change, the projection plane would have only 431.018048 ms, which
is not a credible dense-S4 target.  Closure therefore requires two coupled
budgets: the three projection families must move together toward about 721 ms
while everything outside them moves toward about 205 ms.  A Gate-only or
Down-only result has no terminal budget authority, and a 721-ms projection
plane is not by itself a 2K result.

## Architecture stop gates already closed

### Existing vLLM-faithful Marlin

The repository already carries the upstream-derived W4A16 Marlin scheduler,
four-stage pipeline, persistent stripes, load-time repack, and a fused
Gate/Up epilogue.  This is useful source material, not a missing production
switch.

The real Qwen3.6 P513 profile records 128 MLP Marlin calls at 666.4768 ms,
plus 26.5935 ms for the independent SiLU.  Even deleting the entire SiLU and
scaling only by token count gives

```text
666.4768 * 1853 / 512 = 2,412.1 ms
```

for the P1853 MLP alone.  The current dense-A4 MLP is 1,010.798592 ms.  This
token-linear extrapolation is not a P1853 measurement, but it is a strong
negative architecture screen: extending the old runner seam has no credible
route to a qualitative Prefill improvement.  Marlin remains an implementation
reference and compatibility oracle only.

### SM87 structured sparse S4

The current production K256 A4 publication was audited directly:

- payload SHA-256
  `cdb3b1f54d0a1f406a0d055eb4fd5cba9f272b8b423b47a8746678e4cdb8f1d7`;
- policy SHA-256
  `f345581fd2bec39a10c33a831befafd19ddbd7599f391836eef580b2a6c03717`;
- layer-0 Gate/Up/Down ordinals `0/1/2`, at payload offsets
  `0/45260800/90521600`.

SM87 `mma.sp.m16n8k128.s4.s4` uses the INT4 4:8 pair contract: four adjacent
INT4 pairs select two pairs.  Choosing the minimum-energy legal mask for every
real weight group and weighting the dequantized code energy by its stored BF16
scale gives:

| Projection | weight energy removed | relative L2 error | cosine | already lossless groups |
| --- | ---: | ---: | ---: | ---: |
| Gate | 21.0664% | 45.8982% | 0.88845 | 0.8657% |
| Up | 21.0880% | 45.9217% | 0.88832 | 0.7956% |
| Down | 20.9048% | 45.7218% | 0.88935 | 0.8748% |

Even a hardware-ineligible relaxation that may keep any four of each eight
scalar weights removes 11.3674%/11.3676%/11.2355% of Gate/Up/Down energy and
has about 0.335--0.337 relative L2 error.  The fixed-K-order,
value-preserving mask therefore fails the real-weight architecture screen.
Sparse execution is deferred unless a separately scoped permutation,
calibration/refitting, or fine-tuning project first supplies weights that pass
the real-API capability gates.  No sparse kernel is implemented on this
mainline.

## Dense projection dataflow

### Down: direct K512 skeleton closed; coupled implementation required

Current Down is `M128N128`, 512 threads, 16 warps, 128 registers/thread and
one CTA/SM.  At P1853 it has 15 M tiles and 40 N tiles.  Every N panel therefore
presents the complete B matrix 15 times; NCU reports about 79% DRAM activity.

The first direct-K512 skeleton made one CTA own two adjacent M128 panels for
one N128 panel.  It verified the traffic arithmetic: B presentation falls
from 15 to 8 while A presentation is unchanged, reducing requested packed
operand and scale traffic from 1,347,379,200 to 1,032,990,720 bytes per layer
at P1853 (23.33%).

It also closed the idea of treating M enlargement as an independent change.
The final 512-thread/16-warp implementation still compiled to 128
registers/thread plus 144 bytes of local memory/thread and 392/268 bytes of
spill stores/loads after explicit named scalarization.  Its 99,072-byte full
K512 shared cell was necessarily single buffered.  The zero-local resource
gate rejected it before bitwise correctness and before any API performance
run.  The complete evidence is recorded in
[`../prefill-down-m256n128-k512resident-resource-rejection-2026-08-03/README.md`](../prefill-down-m256n128-k512resident-resource-rejection-2026-08-03/README.md).

The successor therefore keeps the same operand-reuse goal but changes the
scale and output lifecycle with it.  R1 and R4 do not share one forced tile:

- R1 uses `M256N128`, keeps only 64 S32 accumulators per thread across the
  complete `K=17408` dot, and applies its single lane scale at the end.  The
  exact integer upper bound is `49 * 17408 = 852992`.
- R4 uses `M192N128`, 12 warps and 384 threads.  Each thread keeps 64 S32
  current-lane partials and 64 FP32 cross-lane outputs.  The SM87 block limit
  leaves about 168 registers/thread, rather than the roughly 128 available to
  a 512-thread M256 block.  At P1920 it launches exactly ten M tiles, so B
  presentation falls from 15 to 10 with no tail MMA.

Both forms must have:

- no `grid.sync`, split-K, global FP32 partial, or inter-CTA wait;
- one owner and one final BF16 write for every output element;
- factorized scale lanes so a repeated K512 FP32-output/S32-partial pair is not
  simultaneously live across 34 Down epochs;
- at most one CTA/SM, no spill/local memory, and enough independent MMA chains
  for the compiled warp ownership to issue continuously;
- an overlapped A/B pipeline whose compiled shared-memory and register
  contracts fit the actual SM87 per-block limits.

No 220--225 ms credit is retained from the rejected skeleton.  Only a
zero-local factorized-lane cell that reaches the real OpenAI API/EvalScope
direction gate may establish a new Down timing.

### Gate/Up, the product boundary, and Attention

The same skeleton is applied by shape rather than copied literally:

- Gate/Up shares activation fragments and keeps paired output ownership, but
  R1 and R4 again use different cells.  R1 uses `M128N128`: the 512-thread
  CTA keeps 64 S32 outputs/thread across the complete K dot, halving B
  presentation without adding the FP32 cross-lane plane.  Against the current
  `M64N128` cell its requested A+Gate-B+Up-B operand traffic falls by 40% at
  an aligned prompt.  R4 uses paired `M128N64`: 32 S32 current-lane partials
  and 32 FP32 cross-lane outputs/thread retain the current 64-accumulator
  resource floor while reducing the same ideal operand traffic by 20%.  This
  deliberately carries forward the measured M128N64 core mechanism without
  its rejected projection-serial/shared-Gate handoff.
- Each epilogue writes every `BF16(SiLU(Gate) * Up)` value once into the
  existing whole-M projection slabs.  A separate factorized quantizer then
  computes the Down lane-wide dynamic scale and publishes the packed A4
  consumer operand.
- This BF16 product boundary is required by the numerical contract, not an
  optional launch-fusion experiment.  A Gate/Up CTA owns only N128 while one
  R4 Down scale lane covers 4352 product columns.  Without a materialized
  product, static scales, grid synchronization, or recomputation, that CTA
  cannot know the lane-wide maximum.  The previously rejected detached
  M128N64 publication kernel therefore remains closed; it does not forbid the
  single-write Gate/Up epilogue followed by the required lane quantizer.
- Attention projections retain their grouped descriptors (linear QKV+Z and
  full Q/K/V) and online Attention consumer layouts.  A wider-M cell may be
  used only when it removes operand presentation without introducing a global
  synchronization seam.
- Down never inherits Gate's N tile or pipeline simply because both are S4.

### Equalized scale lanes (v4 throughput contract)

The current numerical contract applies a distinct dynamic activation scale
and weight-row scale at every K512/K256 epoch.  Gate/Attention are issue-bound
and Down combines that scale cadence with repeated B traffic.  Replacing FP32
instructions with integer shifts would not remove the `(m,n,group)` scale
dependency, scale loads, shuffles, or barriers, and is rejected.

The v4 sidecar instead declares authenticated positive factors `alpha[k]` and
uses

```text
W'[n,k] = W[n,k] * alpha[k]
A'[m,k] = A[m,k] / alpha[k]
C[m,n] ~= sum_lane Sa[m,lane] * Sb[n,lane]
                         * sum_(k in lane) Qa[m,k] * Qb[n,k]
```

All dense weight values remain represented.  The factor cancels algebraically
before quantization; only the explicitly measured W4A4 recoding error changes.
The ABI supports lane counts 1, 2, and 4, but they have fixed roles rather than
forming a performance sweep:

- R1 is an end-to-end performance upper-bound probe.  The complete K dot fits
  safely in S32 (`49 * 17408 = 852992`) and applies scale once.
- R4 is the predeclared quality-oriented candidate if R1 proves the execution
  architecture but fails the numerical/capability gate.
- R2 is reserved in the ABI so an artifact is not reformatted merely to add a
  lane count; it has no independent scan or promotion role in this plan.
- a small, explicit outlier-channel correction plane is allowed only after a
  real-prompt calibration audit demonstrates that it is necessary and bounded.

The existing equalization parser, factor SHA validation, shared-boundary
policy checks, and converter multiplication are reused.  Production's current
equalization rejection is not deleted.  A new, independent 192-projection MLP
v4 overlay, layout, receipt and authenticated inverse-factor metadata handle
are required, and the existing 400-projection K256 artifact continues to own
Attention.  Old K512 MLP and new v4 MLP overlays are mutually exclusive; the
existing K64/K128/K256 artifacts remain fail-closed.

A real-weight alpha=1 diagnostic on the first 512 rows shows why R1 cannot be
promoted on performance alone: re-quantizing the current A4 Gate/Down weights
adds about 0.153/0.187 weight NRMSE; R4 adds about 0.117/0.156.  Real-prompt
equalization/calibration and the external capability gate are mandatory.

## Secondary layer-boundary package

The GDN/consumer audit found a credible first 90--125 ms package, but not a
credible 150-ms primary architecture:

- retain chunk/head parallel C512 WY/state work;
- fuse GDN RMSNorm+SiLU directly into O-input A4 packing;
- make O/Down epilogues own residual publication; and
- fuse centered RMS with the next K256/K512 A4 producer.

The rejected one-CTA/value-head prompt-span macro serialized 29 C64 chunks and
must not be reopened.  This consumer-native package is P1 after the projection
plane establishes its first positive structural result.  It covers only part
of the required reduction from about 495 ms to about 205 ms: even its 125-ms
upper estimate leaves roughly 165 ms unclosed.  That remainder requires a new
chunk/head-parallel GDN/Attention-core structure and cannot be silently
credited to launch fusion.

## Qualification order

1. Synthetic inputs are used only for decoder-bit exhaustiveness, canaries,
   invalid arguments, non-default streams and CUDA Graph correctness.
2. The first performance observation is one natural P1804 warmup followed by
   one natural P1853 measured request, real checkpoint and authenticated
   sidecars, OpenAI API, external EvalScope 1.9.1, concurrency one and one
   generated token.
3. Route counters and NSys names must prove that every intended layer used the
   candidate and that incumbent/cublasLt/MTP fallbacks were absent.
4. A negative whole-request direction result gets one request-scoped NSys
   attribution and is closed.  There is no tile, stage, cache-hint or CTA-order
   scan around a negative architecture.
5. A positive route proceeds to the P512/P1K/P2K/P4K performance matrix, then
   hidden-state and public EvalScope capability gates.  Aggregate hidden NRMSE
   must remain at most 0.01 with cosine at least 0.9999; no public capability
   score may fall by more than one percentage point or its baseline confidence
   interval.
6. Production remains locked until the complete default-off package passes
   Release tests, Decode graph/oracle checks, memory accounting, real API
   matrices and independent review.

The first complete direction slice is deliberately narrow but end to end:

1. generate an authenticated R1/alpha=1 MLP-v4 artifact from the locked real
   K256 publication (itself bound to the real checkpoint), explicitly labelled
   as a performance upper bound rather than a quality candidate; R4 quality
   conversion later returns to the original checkpoint plus real-prompt
   equalization statistics rather than compounding this R1 recode;
2. attach it over the locked K256 base, with old K512 MLP, cuBLASLt and MTP
   routes forbidden;
3. run factorized hidden quantize, paired Gate/Up plus single BF16 product
   publication, factorized product quantize, and R1 M256 Down in every layer;
4. expose exact route counters and authenticated identities through the same
   OpenAI-compatible server ELF; and
5. run one natural P1804 warmup then one P1853 measurement using external
   `evalscope[perf]==1.9.1`, concurrency one and one generated token.

Only a positive whole-request result unlocks the R4/M192 quality artifact and
kernel, repeated statistics, the longer prompt matrix, and capability gates.
