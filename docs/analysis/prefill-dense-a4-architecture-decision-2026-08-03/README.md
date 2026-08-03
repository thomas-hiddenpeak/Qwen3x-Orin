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

## R1 external result and post-gate decision

The authenticated R1 upper-bound slice passed its performance direction gate
on 2026-08-03.  This is an architecture result, not a production promotion.
The same server ELF was used for both arms:

- ELF SHA-256:
  `ee07214be25ab12ae497dfce31827cf402464be590b40d73e9337b0ba8207c18`;
- model: the pinned real `nvidia/Qwen3.6-27B-NVFP4` checkpoint;
- client: external `evalscope[perf]==1.9.1` through `/v1/completions`;
- concurrency one, one warmup, four measured requests per bucket, one output
  token, no Prefix cache and `mtp=false`;
- baseline MLP: authenticated K256 `M128N256` pairfeed package;
- candidate MLP: authenticated factorized R1 package; GDN, Attention, server,
  prompts and all other selectors remained identical.

The four-bucket result was:

| Bucket | Baseline TTFT | R1 TTFT | TTFT saved | Baseline total throughput | R1 total throughput | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P512 | 1,295.81 ms | 522.63 ms | 773.18 ms | 407.5799 token/s | 1,010.3874 token/s | +147.90% |
| P1K | 1,652.34 ms | 991.88 ms | 660.46 ms | 638.2150 token/s | 1,063.0795 token/s | +66.57% |
| P2K | 2,424.93 ms | 1,751.03 ms | 673.90 ms | 794.0269 token/s | 1,099.5693 token/s | +38.48% |
| P4K | 4,456.86 ms | 3,629.75 ms | 827.11 ms | 854.2648 token/s | 1,048.8411 token/s | +22.78% |

Every bucket completed 4/4 measured requests.  Every candidate request proved
`factorized_lane_r1_package_launch_hits=64`, zero incumbent/K512 MLP hits, the
same Attention and GDN counts, and `mtp=false`.  The external direction
validator independently recomputed every summary and returned
`advance_to_internal_validation` in all four buckets.  Its records are:

| Bucket | Direction record SHA-256 |
| --- | --- |
| P512 | `8510d3b25721fdf6a3db79b98645ed8838570ca0292e9421d8155c22b14cbaff` |
| P1K | `a97e2715f979ba66cb1f0288d7f0107abe34aeb6cd494deb5ebf3df47cfe5488` |
| P2K | `868e2f93cf6962c44fe7b236366029323ef2d86dd5c962798bdf5c7f703ca618` |
| P4K | `43730266aa81e6285caa6fcbde49783b5181bb0d5863779770350f093fd02f1e` |

The candidate output set matched the native baseline in 0/4 requests in every
bucket.  That is expected for this explicitly approximate R1 recode and is a
hard reminder that the throughput result has no capability or quality
authority.  R1 remains `quality_production_eligible=false`,
`production_residency_eligible=false` and
`performance_upper_bound_only=true`.  It must not become a default or
production route by changing metadata flags.

The matrix initially exposed a harness-only boundary error at P1025.  Native
C512 deliberately leaves a final C1--C31 tail on the exact recurrence, while
the harness had incorrectly expected a rounded-up native launch.  Commit
`55639e7` corrected the accounting to full C512 tiles plus a native tail only
when the tail has at least 32 tokens, added P1025/P1055/P1056 boundary tests,
and passed all 86 external-harness host tests before the matrix was rerun.

### Request-scoped R1 NSys closure

One real P1853 measured API request was captured with
`--profile-request-index 2`.  Profiling overhead makes its external TTFT
non-authoritative, but the server recorded 1,697.08 ms of Prefill and the GPU
kernel span was 1,697.071872 ms.  The 2,744 kernel intervals had no overlap:

- kernel raw sum and interval union: 1,679.437568 ms;
- kernel span idle: 17.634304 ms;
- all GPU-operation raw sum and union: 1,694.903808 ms;
- all GPU-operation span: 1,699.099616 ms;
- memory operations: 15.466240 ms.

The complete kernel budget is:

| Category | Time | Kernel share |
| --- | ---: | ---: |
| R1 Gate+Up | 487.604160 ms | 29.03% |
| R1 Down | 221.828032 ms | 13.21% |
| R1 hidden/product quantizers | 89.017024 ms | 5.30% |
| Attention K256 projections | 404.077856 ms | 24.06% |
| Attention K256 quantizers | 28.579904 ms | 1.70% |
| GDN hierarchy | 249.540672 ms | 14.86% |
| residual and RMSNorm | 117.727968 ms | 7.01% |
| FlashInfer Attention core/pre/post | 56.228928 ms | 3.35% |
| fixed and other kernels | 24.833024 ms | 1.48% |

The raw NSys report SHA-256 is
`071ad5bd10f637d3f9f4992dfb00626b88d1c56f40bde1d48347dc59dfbcbd10`.
The exported kernel summary and GPU trace SHA-256 values are respectively
`ee0231e26807b7e9ee1e108622f5bd14b0b5ff8b72468e66c86fc52549d8cb1a`
and
`e9e5705f0d7a8ec3e4ea4ffff6c4f1c32636a66468ae601f0133a42d122c0449`.
They remain outside the repository with the raw EvalScope databases because
they contain run-local evidence.

### Consequence

The R1 MLP package still owns 798.449216 ms, and Attention projections plus
their quantizer own another 432.657760 ms.  The new end-to-end throughput also
plateaus near 1.05--1.10K token/s from P1K through P4K.  The next milestone is
therefore not an R1 tile or cache-policy scan.  It is a quality-preserving,
A-resident high-M/low-N projection plane:

1. build the first K256-semantics strip engine to separate scheduling gain
   from recoding quality;
2. move the proven dataflow to a calibrated R4 artifact generated directly
   from the original checkpoint, never from R1 or K256;
3. extend the same operand-resident projection plane to grouped Attention;
4. then reduce the 249.540672-ms GDN hierarchy and its consumer boundaries.

The P1853 2K budget remains 926.50 ms.  The measured 1,697.08-ms profiled
request therefore requires a further 1.832x whole-request improvement.  The
MLP, Attention-projection and GDN changes must be treated as one system budget;
none can close the target alone.

## K256 M240N64 strip resource closure

The first quality-preserving K256-semantics strip engine was closed at its
resource gate before correctness, runner integration, or timing.  Its fixed
P1920 topology used sixteen cooperative CTAs arranged as eight M240 owners and
two N lanes.  Each lane owned two N64 cells of every N256 edge.  This reduced
Gate/Up B presentation from fifteen M owners to eight while retaining the
current K256 scale epochs and exact BF16 product seam.

The resource audit corrected an important SM87 limit.  A 480-thread block has
fifteen logical warps, but register allocation across the four SM
subpartitions rounds that block to sixteen warps.  The launchable ceiling is
therefore 128 registers/thread, not 136: `4096 * 16 == 65536` registers per
SM.  Two independently compiled lifetime forms failed the zero-local gate:

| Partial lifetime | Registers | Stack/thread | Spill stores | Spill loads |
| --- | ---: | ---: | ---: | ---: |
| paired N32 retire | 128 | 392 B | 948 B | 868 B |
| paired N8 retire | 128 | 296 B | 688 B | 632 B |

Replacing nested aggregates, reference-returning fragment accessors and
runtime-looking arrays with explicitly named `Float4` outputs and named N8
partials did not change the second result.  The blocker is therefore the
complete N64 Gate/Up FP32 cross-K256 state plus address, fragment and control
state under the 128-register ceiling, not an accidental C++ aggregate alone.
The local-memory gate was not weakened.

Static review found no coverage, K256-order, M240 canonical-gather,
split-plane store, three-stage pipeline, or uniform-grid-barrier defect.  The
architecture nevertheless offered only a bounded traffic change: packed
operand traffic fell by 23.33% because A presentation was unchanged, and the
estimated edge traffic improvement fell to about 18.3% after the BF16 product
and split quantizer were counted.  N8 retirement also repeated shared-A
`ldmatrix` work.  No correctness harness, selector, OpenAI API route, or
performance claim was created for this failed skeleton.

This closure removes the K256 M240N64 paired strip from the active route.  The
next Gate/Up implementation changes the resource model instead of shrinking
another lifetime inside the same block: the predeclared calibrated R4 plane
uses a lower-thread paired M128N64 Gate/Up cell and an independent M192N128
Down cell.  Its four scale lanes are generated directly from the original
checkpoint with real-prompt equalization statistics, never by recoding R1 or
the current K256 sidecar.  It must first prove a positive complete-request
direction through the real OpenAI API and external EvalScope; only then does
it earn the longer performance and capability matrices.

## R4 resource-plane closure

The first paired R4 Gate/Up implementation tried to share the Down cell's
`M192N64`, 384-thread ownership.  Even after moving three paired N8 S32
fragments to shared memory, the final device image used 168 registers/thread,
8 bytes of stack, 12 bytes of spill stores and 16 bytes of spill loads.  It
therefore failed the zero-local hard gate.  That source is not retained as a
runtime candidate and receives no correctness or performance credit.

The shape-separated resource plane now consists of:

| Consumer | Cell | Threads | Pipeline | Dynamic shared | Registers | Local/spill | Orin residency |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| paired Gate+Up | `M128N64` | 256 | 4 stages | 163,840 B | 242 | 0 B / 0 B | 1 CTA/SM, 16 grid CTAs resident |
| Down | `M192N128` | 384 | 3 stages | 122,880 B | 168 | 0 B / 0 B | 1 CTA/SM, 16 grid CTAs resident |

Gate and Up share every staged A tile and publish the BF16-RNE
`SiLU(Gate) * Up` product once.  The complete current-lane S32 state remains
in registers; four paired N8 cross-lane FP32 fragments live in a field-major
shared tail which is touched only at the four lane folds and the epilogue.
Down keeps 64 current-lane S32 plus 64 cross-lane FP32 values per thread.  At
P1920 its ten M owners reduce Down B presentation from fifteen to ten without
split-K, a grid barrier, or a global FP32 partial.

These numbers close only the compile/device resource gate.  They are not a
Prefill throughput result and do not authorize a selector.  The next ordered
gates remain full release-shape bitwise CUDA correctness, a direct publication
from the pinned original NVFP4 checkpoint, and then one real P1853 request
through the OpenAI-compatible API driven by external EvalScope.  A negative
whole-request direction closes the R4 cell before any microbenchmark campaign;
a positive direction unlocks the prompt matrix and profiler attribution.

## Direct R4 external closure

The complete direct-checkpoint R4 path reached the real API gate and was
rejected.  External EvalScope measured 832.1429 token/s and 2,227.94-ms TTFT
at P1853; the server measured 2,223.02-ms Prefill.  The locked R1 reference is
1,106.6259 token/s and 1,670.42-ms server Prefill, so R4 regressed throughput
by 24.80% and Prefill latency by 33.08%.

Request-scoped NSys isolated the regression to the R4 cells.  Gate+Up rose
from 487.604160 to 860.028928 ms (+76.38%) and Down rose from 221.828032 to
390.843168 ms (+76.19%).  Every unchanged category remained within 0.5% of
the R1 trace.  This validates the resource-plane stop rule: the lower R4 lane
count per cell did not compensate for N64 A replay, cross-lane FP32 lifetime,
one-CTA/eight-warp Gate residency or increased operand issue.

The M128N64/M192N128 skeleton is closed.  Its code remains default off as an
authenticated negative architecture and quality reference, but no local
parameter scan follows.  The next `projection-plane v2` must change the
resource/occupancy model and cover shape-specific MLP plus Attention
projections and legal layer-boundary fusions as one system budget.  Complete
evidence, hashes and the kernel top 20 are recorded in
[`../prefill-factorized-r4-direct-rejection-2026-08-03/README.md`](../prefill-factorized-r4-direct-rejection-2026-08-03/README.md).

## Direct R4 two-CTA cancellation closure

The resource-successor package also reached the real API and is closed.  Its
M64N64 Gate+Up cell reached two CTA/SM and improved Gate by 77.113888 ms, but
the M128N64 Down cell increased staged operand presentation by about 80% and
regressed by 72.722592 ms.  The complete MLP slice improved only 4.472256 ms;
same-ELF external EvalScope moved from 829.7341 to 833.9381 token/s, still
24.64% below the locked R1 baseline.

This is a cancellation result, not a new baseline.  No repeat matrix or local
scan follows.  The active plan returns to R1 and requires projection-plane and
legal GDN/residual/RMS/quantize boundary changes as one package.  Full route,
resource, profile, hashes and kernel-top-20 evidence are in
[`../prefill-factorized-r4-2cta-cancellation-2026-08-03/README.md`](../prefill-factorized-r4-2cta-cancellation-2026-08-03/README.md).
