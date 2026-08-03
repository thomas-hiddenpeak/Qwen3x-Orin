# Direct R4 two-CTA MLP cancellation

Date: 2026-08-03
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill

Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Decision

Reject the combined direct-R4 two-CTA MLP leaf as a performance baseline.  It
remains default off and keeps no production, quality-promotion, or fallback
eligibility.  The resource mechanism is real, but its Gate gain is cancelled
by its Down regression; the resulting complete-request movement is only about
0.5% and the candidate remains 24.64% slower than the locked R1 performance
baseline.

There is no repeat matrix and no stage, cache, raster, or tile scan around this
result.  The two kernels and fail-closed leaf remain in the tree as auditable
resource/correctness evidence.  The active performance program returns to R1
and requires a package-sized projection plus boundary change.  MTP remains
disabled and cuBLASLt remains reference-only with no production eligibility.

## Qualified implementation

Commit `ce59d203b09334971604b9e793681609c872a164` introduced two independent,
default-off cells.  Commit `ca008d14d9b539f216a245faa6a70f562dac412e`
wired them beneath the existing authenticated R4 master as the immutable leaf
`Q3X_RUN_A4W4_FACTORIZED_LANE_R4_2CTA_ADMISSION`.

| Cell | Threads | Dynamic shared | Registers | Local/spill | Orin residency |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gate+Up `M64N64`, 2-stage, shared Up cross-tail | 256 | 65,536 B | 128 | 0 B / 0 B | 2 CTA/SM, 32 resident |
| Down `M128N64`, 3-stage | 256 | 73,728 B | 115 | 0 B / 0 B | 2 CTA/SM, 32 resident |

Both release-shape P1853-to-P1920 tests are bit exact and cover tail guards,
the Gate primary/secondary seam, nondefault streams, CUDA Graph capture and
replay, invalid arguments, and capacity failures.  The combined factory
resource-gates both cells and never silently falls back.  The artifact ABI is
unchanged: the same 8,583,954,432-byte R4 payload, manifest, policy, and receipt
are reused.

## Same-ELF real API direction gate

The baseline and candidate used the identical server ELF
`8fdf38d997f3af63d9f78b341205525ec8cc10d631ad567eb4458b0d3a4381c7`,
the natural P2K corpus
`41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af`,
one P1804 warmup, one P1853 measurement, concurrency one, one output token,
and no MTP.  Only the subordinate two-CTA leaf changed.

| Route | P1853 TTFT | Total throughput | Server Prefill |
| --- | ---: | ---: | ---: |
| incumbent direct R4 | 2,234.41 ms | 829.7341 token/s | 2,229.24 ms |
| R4 two-CTA Gate+Down | 2,223.15 ms | 833.9381 token/s | 2,218.17 ms |
| apparent movement | -11.26 ms (-0.504%) | +4.2040 token/s (+0.507%) | -11.07 ms (-0.497%) |

Both candidate requests prove 64 direct-R4 master hits, 64 subordinate two-CTA
hits, 128 Attention A-exchange-B4 physical launches covering 208 logical
projections, 192 native GDN launches, and `mtp=false`.  Baseline requests prove
the same composition with zero subordinate hits.  The locked R1 reference is
1,106.6259 token/s, so 833.9381 token/s is not a new experimental performance
baseline even before statistical expansion.

## Request-scoped NSys attribution

One NSys capture profiled only the measured P1853 candidate request.  Profiler
TTFT is not performance authority; the kernel totals and category split are
used only for attribution.  The trace contains the same 2,744 kernel calls as
the incumbent R4 trace.

| Category | incumbent R4 | two-CTA leaf | Change |
| --- | ---: | ---: | ---: |
| Gate+Up | 860.028928 ms | 782.915040 ms | -77.113888 ms (-8.97%) |
| Down | 390.843168 ms | 463.565760 ms | +72.722592 ms (+18.61%) |
| two factorized quantizers | 86.423744 ms | 86.342784 ms | -0.080960 ms |
| complete MLP slice | 1,337.295840 ms | 1,332.823584 ms | -4.472256 ms (-0.334%) |
| all kernels | 2,219.219456 ms | 2,213.942592 ms | -5.276864 ms (-0.238%) |

The two-resident Gate mechanism therefore works, but only modestly.  Its
`M64N64` cell raises staged A+B presentation from about 2.674 to 4.011 GB per
layer (+50%) while doubling resident CTAs, yielding a net 8.97% Gate gain.
Down raises staged presentation from about 1.114 to 2.005 GB per layer (+80%);
two-CTA residency cannot repay it and Down regresses by 18.61%.  Their
cancellation explains the whole-request result without a speculative NCU
story.

The complete kernel top 20 is retained in
[`kernel-top20.csv`](kernel-top20.csv).

## Carry-forward

Do not tune this combined M64N64/M128N64 leaf.  The M64N64 Gate resource
mechanism may be reconsidered only as part of a later R4 quality path paired
with a non-regressing Down design; it is not relevant to the current fastest
R1 performance route.

The next performance package must start from R1 and remove work in more than
one plane:

1. preserve R1's `M128N128` Gate and `M256N128` Down operand reuse while
   eliminating producer/quantizer boundaries;
2. replace the 404-ms Attention projection backbone only with a design that
   reduces work, not one that merely trades B4 for 33% more staged traffic;
3. fuse legal GDN/residual/RMS/quantize consumer boundaries, whose audited
   full-package ceiling is about 125 ms;
4. take the first verdict through the same real API and external EvalScope,
   and compare with the current R1 baseline rather than this rejected R4 path.

## Evidence

Direction root:
`/home/rm01/q3x-r4-2cta-real-api-eval-20260803`

```text
baseline summary SHA-256
  89e199f7fa74c14625566c171cc5376b885c748de43d261bac2a7fcff587c5d7
baseline server log SHA-256
  369d7c43df93e01a0cbfd9672bea96d34d2309bd3ebdbdd0a426207b13f0a843
candidate summary SHA-256
  16ab4e66b6a89a8691c291147ca3c9d808d574e9621e9651ccaec00dc84dc2c2
candidate server log SHA-256
  7fe15e33052dc6e17e6d170a71a0b5d2abdf4dbb14c1d8917882a125a8615dd7
NSys report SHA-256
  6839ac3ece8c9f8aafa32a223892da15eb5543080010e344feb62f759e686796
NSys kernel summary SHA-256
  277336ae93825ec2353612c5f45492ca0ff345507bb709fce629032ede16441d
profile server log SHA-256
  56235711f631925067d4d3b210b1ea309191d1536478f5ccb1725649d7afb2e0
```
