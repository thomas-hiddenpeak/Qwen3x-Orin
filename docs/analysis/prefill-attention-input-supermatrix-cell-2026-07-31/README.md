# Prefill Attention input-supermatrix complete cell (2026-07-31)

## Status and boundary

This change now has an independent, default-off runtime admission slice. It
implements fixed Linear QKV+Z, Full Q/K/V, and Attention O specializations and
links them into `q3x_kernels` only when
`Q3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=ON`. The exact-value
runtime selector is `Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=1`. This is
still an experimental runner route, not a production performance result.

The candidate does not use cuBLASLt, MTP, a generic GEMM wrapper, or a new
weight layout. It consumes the authenticated signed-A4 K128 sidecar ABI and
writes the existing independent row-major BF16 projection outputs.

## Why this is the first attention candidate

The frozen P2048 cumulative NSys trace attributes 864.216 ms to attention
projections:

| Projection family | Calls | Time (ms) |
|---|---:|---:|
| Full K/V, N1024 K5120 | 32 | 19.906 |
| Linear Z/O plus Full O | 112 | 415.323 |
| Linear QKV, N10240 K5120 | 48 | 304.653 |
| Full Q, N12288 K5120 | 16 | 124.334 |

The Linear QKV and Z projections have the same A and K=5120. They also cover
the largest input-side topology: 160 QKV N64 panels and 96 Z N64 panels.
Consequently Linear was the first vertical slice; Full Q/K/V and O now use the
same physical cell with their own compile-time topology and ABI.

Source trace: `docs/analysis/prefill-p2048-cumulative-nsys-2026-07-31/README.md`.

## Complete-cell dataflow

One 256-thread CTA owns M128 and two independent N64 output panels. Every warp
owns one M16 strip and eight N8 fragments from each panel, for 64 long-lived
FP32 accumulators per thread. Two K64 S4 MMA operations form one S32 K128
partial before one pair of BF16 A/B scales is applied.

The shared-memory pipeline is asymmetric:

```text
A stage:       M128 x packed-K128 codes + 128 BF16 scales = 8,448 B
B pair stage:  2 x N64 packed-K128 codes + 2 x 64 scales  = 8,448 B
A2 + B3:       5 x 8,448 B                               = 42,240 B
```

The prologue commits A0+B0, A1+B1, and B2. During the steady state, consuming
group `g` releases A for `g+2` and B for `g+3`. Codes and scales use aligned
`cp.async.cg`; K128 groups remain strictly ordered.

For the same 16,384 output values as the incumbent M64N256 stage, staged
global traffic changes from 4,224 B A + 16,896 B B = 21,120 B to 8,448 B A +
8,448 B B = 16,896 B, a 20% reduction. This is a different complete-cell
dataflow, not launch aggregation around the generic kernel.

Stores pack adjacent BF16 accumulator values into aligned 32-bit writes. The
launcher therefore requires 4-byte output alignment and even row strides.

## Fixed Linear topology

For P2048, M has 16 M128 tiles. Each M tile has 128 pair cells:

- cells 0..95 pair QKV panel `c` with Z panel `c`;
- cells 96..127 pair the remaining QKV panels two at a time.

There are 2,048 work cells. Thirty-two persistent CTAs preserve one M tile and
one pair-cell parity per CTA, so every CTA executes exactly 64 cells. No
atomic scheduler or dynamic partition descriptor appears in the K loop.

## Fixed Full and O topologies

Full Q/K/V has 192 Q panels and 16 K/V panels each. Cells 0..15 pair Q with
K, cells 16..31 pair Q with V, and cells 32..111 pair the remaining Q panels.
At P2048 that is 1,792 work cells, exactly 56 per persistent CTA. K and V stay
as separate payload and BF16 output pointers; no physical concatenation or
dynamic descriptor appears in the K loop.

Attention O has 80 N64 panels at K6144. Its dedicated one-output kernel pairs
adjacent panels into 40 cells per M tile, or 640 P2048 work cells and exactly
20 cells per CTA.

At P2048 the complete attention family contains 2,048 Linear input, 1,792
Full input, and 640 O work cells. The 29.549 TOP attention projection
arithmetic lower bound at 170.459 TOP/s is 173.352 ms. Therefore attention
alone cannot reach the project-wide 2K token/s target; it is one structural
component of the projection-wide rewrite.

## Admission gates

Configured with:

```text
-DBUILD_TESTING=ON
-DQ3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=ON
```

At runtime the route remains disabled unless the exact selector is present:

```text
Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=1
```

The selector requires an authenticated all-400-projection K128 inventory,
complete M128 projection spans, the exact Qwen3.6 projection shapes, and every
A/weight/scale/output capacity. An ineligible shape remains on the incumbent
route. Once a candidate launcher is selected, any CUDA error is returned
directly; there is no silent retry through the incumbent implementation.

Linear QKV+Z and Full Q/K/V each reuse the one A quantization already owned by
their layer. Attention O reuses its existing post-attention quantization. A
fully selected 64-layer execution therefore preserves 192 A quantizations and
replaces 208 logical generic projections with exactly 48 Linear-input, 16
Full-input, and 64 O physical launches. Independent success-only counters
enforce that 48/16/64/208 ratio.

The host contract proves the real P2048 topology, exact panel coverage,
persistent-CTA balance, final consumer-layout addresses, invalid tails, and a
null resource-query fast rejection. Because device row coordinates are
uint32, every plan also rejects a row count beyond `UINT_MAX+1`; the boundary
contract covers the first wrapping M128 tile.

The candidate compilation reports:

| Specialization | Registers/thread | Static shared | Stack | Spill stores/loads |
|---|---:|---:|---:|---:|
| Test N256+N128 | 123 | 42,240 B | 0 B | 0 B / 0 B |
| Real Linear QKV+Z | 123 | 42,240 B | 0 B | 0 B / 0 B |
| Test Full Q256/K64/V64 | 124 | 42,240 B | 0 B | 0 B / 0 B |
| Real Full Q/K/V | 123 | 42,240 B | 0 B | 0 B / 0 B |
| Test O N256 | 128 | 42,240 B | 0 B | 0 B / 0 B |
| Real O N5120/K6144 | 128 | 42,240 B | 0 B | 0 B / 0 B |

The N256+N128 test topology covers both mixed QKV/Z cells and the
QKV-only remainder cell used by the real specialization. The Full test covers
Q+K, Q+V, and Q-only cells; O covers two panels of one output. The CUDA test
hooks deliberately cap their grids at 2/2/1 CTAs, forcing at least one CTA to
execute multiple cells and reuse the shared pipeline. The test is serialized
by both CTest `RESOURCE_LOCK` and
`flock -x /tmp/q3x-gpu-bench.lock`. It checks K={128,256,384,512} against a
CPU K128 oracle, every output bitwise, row guards, short capacities,
misalignment, M tails, short/even-stride requirements, output capacity, and
output aliasing. The runtime resource query additionally requires SM87/16 SM,
at most 128 registers, zero local bytes, and at least 2 active CTAs/SM.

## Promotion order

1. Pass the small-K bitwise/guard/resource admission for all three families.
2. Exercise the experimental runner selector and measure the first real P2048
   API request with authenticated model weights.
3. Keep the selector only for a positive whole-request result; then run the
   repeated real-request and external EvalScope gates before production
   promotion.

No synthetic timing is an admission or performance result.
