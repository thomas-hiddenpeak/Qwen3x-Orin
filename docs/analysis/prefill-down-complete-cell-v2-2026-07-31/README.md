# SM87 NVFP4 Down complete-cell v2 design

Status: standalone structural admission passed, a default-off runtime vertical
slice is wired, and the first real-weight generation gate is repeat-positive.
The production default is unchanged; the candidate is retained for cumulative
work but has not passed the same-ELF statistical or EvalScope promotion gates.

## Pinned shape and contract

- Real Down shape: `M=2048, N=5120, K=17408`.
- Both operands retain the authenticated consumer layout
  `[ceil(outer/64), K/64, 64, 32]` packed signed nibbles.
- Two ordered physical K64 code planes form one logical K128 group.  The two
  integer MMA partials accumulate into the same S32 value before the one BF16
  A-scale times BF16 B-scale product is applied.
- The output is BF16 and must be bitwise equal to the established K128 Down
  contract.  There is no split-K workspace and no intermediate global store.

## Complete-cell dataflow

The rejected M128N256 experiment retained 128 FP32 outputs per thread and
compiled to one CTA/SM.  This cell changes the ownership rather than tuning
that skeleton:

| Item | Ownership / residence | Lifetime |
|---|---|---|
| Output | one CTA owns M128N128; each of eight warps owns M16N128 | 64 FP32 accumulator registers/thread for the full K loop |
| A codes/scales | two shared slots, each M128K128 = 8192 code bytes + 256 scale bytes | current group and one-group lookahead |
| B codes/scales | three shared slots, each N128K128 = 8192 code bytes + 256 scale bytes | current group and two-group lookahead |
| Physical MMA | M16N8K64 S4xS4->S32 | low K64 then high K64 in the same S32 accumulator |
| Dequantization | registers plus shared BF16 scales | exactly once per logical K128 group |

Shared memory is therefore `2*8448 + 3*8448 = 42240` bytes.  Two resident
CTAs consume 84480 bytes, below the frozen 96-KiB/SM engineering budget.  The
asymmetric ring gives the cold weight plane a two-group lead while A has a
one-group lead.  It also avoids the 50688-byte three-full-stage layout, which
cannot satisfy the two-CTA budget.

All code copies use `cp.async.cg`: each shared value is reused by the complete
CTA tile, while the full-K reuse distance between consecutive persistent work
items is much larger than L1.  L2, not competing A/B `.ca` traffic, is the
appropriate inter-cell reuse level.

## Persistent mapping at the real shape

The launch has 32 persistent CTAs (two per each of the 16 Orin SMs).  The
M128N128 grid has 16 M tiles and 40 N tiles.  CTA `b` fixes
`m_tile=b mod 16` and `n_parity=floor(b/16)`, then visits
`n_tile=n_parity,n_parity+2,...,39`.  Consequently:

- every CTA executes exactly 20 complete cells (no 32+8 CTA tail wave);
- all 640 real-shape cells are covered once;
- the A M128 tile is stable for a CTA while adjacent resident CTAs cover the
  two N parities;
- each scheduling phase exposes all 16 M tiles for two adjacent B stripes,
  allowing their packed weights to be shared through L2 across CTAs.

The minimal contract shape falls back to one work item per CTA, but the target
mapping is explicit and does not rely on launch order for correctness.

## Hard admission gates

1. ptxas reports zero stack, zero local memory, and zero spills.
2. The compiled kernel reports at least two active CTAs/SM on the 16-SM SM87
   target and no more than 128 registers/thread.
3. Small-K tests exercise K128 ring fill, steady-state recycle, and drain
   (`K=128,256,384,512`) with exact candidate/baseline/CPU agreement.
4. A real-address shape check covers `M=2048,N=5120,K=17408` capacities and
   the final logical output coordinate without allocating a synthetic full
   performance fixture.
5. Invalid alignment, capacity, and non-tile-multiple shapes fail closed.

These gates permit the narrow runtime admission below.  They do not admit the
candidate as a production default; that decision requires real-weight
generation timing.

## Default-off runtime vertical slice

The build option
`Q3X_BUILD_SM87_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=ON` compiles the
candidate and its runtime selector.  The selector remains disabled unless
`Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=1` is present and optimized
Prefill dispatch is enabled.

Admission is deliberately model-specific and fail-closed.  A launch requires:

- the immutable full-model inventory result to be authenticated K128;
- exact Down `N=5120,K=17408`;
- the runner's internal ceil64-padded projection M to be a nonzero multiple
  of 128 (for example P2048); and
- complete packed-A, A-scale, packed-weight, weight-scale, and BF16-output
  capacities.

The logical prompt length is not passed to the selector.  An internal padded
span such as P1853 -> M1856 therefore remains on the established K128 route;
M64 tail handling is outside this slice.  When both experimental environment
switches are present, complete-cell v2 has priority over the archived Down
M128 stage-major route.  Once selected, any launch failure is returned to the
runner directly and cannot silently fall back.  K64 and all ineligible K128
shapes retain their existing paths.

Dedicated enable and successful-launch counters keep this experiment
independent from the old Down M128 and aggregate full-A4 accounting.  Host
tests cover the exact positive shape, each negative shape/capacity dimension,
environment parsing, selector priority, switch orthogonality, and both
candidate-off and candidate-on builds.  The four small-K GPU cases and PTX
resource sentinel remain mandatory.  These checks establish dispatch safety,
not a speed result.

## Measured structural result

Built on the target host with CUDA 13.3.  The isolated `sm_87` compilation
reported:

```text
0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
Used 124 registers, used 1 barriers, 42240 bytes smem
```

The device resource query on the 16-SM Orin reported:

```text
registers=124 static_shared=42240 dynamic_shared_limit=6912
local=0 active_blocks_per_sm=2 gate=PASS
```

The host contract proved the full `M=2048,N=5120,K=17408` final packed-code,
K128-scale, and BF16-output addresses equal their computed capacities.  It
also enumerated all 640 work cells: each is visited once and each of the 32
CTAs owns exactly 20 cells.

Candidate, established K128 baseline, and CPU reference were bitwise equal at
`M=128,N=256` for `K=128,256,384,512`.  These four cases respectively cover
single-group fill/drain, two-group fill/drain, first A-slot recycle, and the
steady A2/B3 recycle.  Output-row guards and invalid alignment/capacity/tail
calls also passed.  The PTX sentinel found the required SM87 S4 MMA and
`cp.async.cg` commit/wait instructions.

`compute-sanitizer` could not add a memcheck result because GPU debugging
features are disabled on this target configuration; the executable itself
still completed all four exactness cases during that invocation.  This is an
environmental coverage limitation, not a sanitizer pass.

## Real-generation direction gate

The runtime slice was measured with the pinned real
`nvidia/Qwen3.6-27B-NVFP4@0893e160` checkpoint, authenticated full-model K128
publication, a natural ShareGPT P4K request truncated to exactly 2,048 token
IDs, and the OpenAI-compatible `/v1/completions` path with one generated token.
Synthetic matrices were not used for this performance decision.

The same candidate process enabled only the retained cumulative bundle plus
the new Down selector:

```text
Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1
Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1
Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION=1
Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=1
```

Gate+Up complete-cell, old M128, and short-Prefill selectors remained off.
The measured ELF SHA-256 was
`3c75e0ea9614fff42651da25bbae9ca5cf02dd2b079cc7596aa04a71b556666b`.

| P2048 request | Server-side Prefill | Prompt rate | HTTP total |
|---|---:|---:|---:|
| run 1 | 3458.01 ms | 592.25 tok/s | 3.461642 s |
| run 2 | 3446.27 ms | 594.26 tok/s | 3.454676 s |
| run 3 | 3446.57 ms | 594.21 tok/s | 3.451667 s |
| mean | 3450.283 ms | 593.574 tok/s | 3.456 s |

Against the frozen cumulative server-side baseline mean of `3554.425 ms`
(`576.18 tok/s`), the candidate saves `104.142 ms`, reduces Prefill latency by
`2.930%`, and improves prompt throughput by `3.018%`.  All three requests were
positive against the frozen mean, so the mechanism passes the requested
real-path direction gate and is retained.

This comparison intentionally remains an experiment gate rather than a
production promotion: the frozen baseline came from an earlier ELF.  A
same-ELF interleaved baseline/candidate replay, natural-length matrix, and
public capability checks are still required before a default change.  The
profiler proof below closes only the dispatch/attribution question.  The gain
is useful but not structural enough to alter the main priority: the projection
plane still needs a new compressed-weight consumer architecture to reach the
2,000-token/s system target.

## Request-scoped profiler proof

A subsequent NSys capture used the same real request and candidate process,
with model loading outside `cudaProfilerStart`/`cudaProfilerStop`.  Profiling
overhead raised the request's reported Prefill time to `3481.72 ms`; that wall
number is not used as the performance result.  The kernel trace proves the
selected route and its exact contribution:

| Kernel family | Calls | Total time |
|---|---:|---:|
| Gate+Up K128 paired | 64 | 1316.942 ms |
| Generic K128 (Attention projections only) | 208 | 866.181 ms |
| Down complete-cell v2 | 64 | 502.655 ms |

The frozen profile had 272 generic calls totalling `1471.572 ms`, of which the
same fixed-shape attribution assigned 64 Down calls and `607.355 ms`.  The new
trace instead has exactly 208 generic calls, so all 64 Down launches moved to
the complete cell.  Its `104.700-ms` kernel saving matches the `104.142-ms`
unprofiled wall saving within measurement noise.  Down effective throughput
therefore rises from `38.469` to approximately `46.483 TOPS` (`1.2083x`).

The complete projection plane is now `2685.778 ms` at approximately
`37.100 TOPS`, a `1.0382x` improvement over the frozen `2788.393-ms` profile.
Gate+Up and Attention projections remain the dominant structural targets.

The NSys report SHA-256 is
`198b1018b5bb54179264754ec42f53abade0af3bbb03a1331a58cf2926e76aad`;
the exported SQLite SHA-256 is
`25b7d6c241163a8b336362467ca37f6abd087684e04c51be6924266e4800d066`.
