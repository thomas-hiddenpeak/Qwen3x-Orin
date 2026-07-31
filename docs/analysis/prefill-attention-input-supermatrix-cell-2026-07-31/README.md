# Prefill Attention input-supermatrix complete cell (2026-07-31)

## Status and boundary

This change is a standalone, test-only admission candidate. It implements the
Linear Attention QKV+Z input side only. It is not linked into `q3x_kernels`,
has no runner selector, and is not a production performance result.

The candidate does not use cuBLASLt, MTP, a generic GEMM wrapper, or a new
weight layout. It consumes the authenticated signed-A4 K128 sidecar ABI and
writes the existing independent row-major BF16 QKV and Z outputs.

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
Consequently this vertical slice tests the reusable dataflow before adding the
Full Q/K/V and O fixed-shape launchers.

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

## Whole-attention topology that follows this gate

The same physical cell can cover all attention projections, but those fixed
launchers are intentionally not claimed as implemented here:

- Full Q/K/V: Q has 192 N64 panels and K/V have 16 each. The fixed launcher
  needs 112 pair cells per M tile and three independent BF16 store ABIs.
- Attention O: N5120 is 40 N128 cells with K6144. It needs a dedicated
  one-output specialization, not a duplicated virtual output.

At P2048 the complete attention family would contain 2,048 Linear input,
1,792 Full input, and 640 O work cells. The 29.549 TOP attention projection
arithmetic lower bound at 170.459 TOP/s is 173.352 ms. Therefore attention
alone cannot reach the project-wide 2K token/s target; it is one structural
component of the projection-wide rewrite.

## Admission gates

Configured with:

```text
-DBUILD_TESTING=ON
-DQ3X_BUILD_SM87_A4W4_ATTENTION_SUPERMATRIX_ADMISSION=ON
```

The host contract proves the real P2048 topology, exact panel coverage,
persistent-CTA balance, final consumer-layout addresses, invalid tails, and a
null resource-query fast rejection.

The candidate compilation reports:

| Specialization | Registers/thread | Static shared | Stack | Spill stores/loads |
|---|---:|---:|---:|---:|
| Test N256+N128 | 123 | 42,240 B | 0 B | 0 B / 0 B |
| Real Linear QKV+Z | 123 | 42,240 B | 0 B | 0 B / 0 B |

The N256+N128 test topology covers both mixed QKV/Z cells and the
QKV-only remainder cell used by the real specialization. The CUDA test is
serialized by both CTest `RESOURCE_LOCK` and
`flock -x /tmp/q3x-gpu-bench.lock`. It checks K={128,256,384,512} against a
CPU K128 oracle, both outputs bitwise, row guards, short capacities,
misalignment, M tails, short/even-stride requirements, output capacity, and
output aliasing. The runtime resource query additionally requires SM87/16 SM,
at most 128 registers, zero local bytes, and at least 2 active CTAs/SM.

## Promotion order

1. Pass the small-K bitwise/guard/resource admission.
2. Add the fixed Full Q/K/V and O specializations to the same standalone
   library and repeat their resource/correctness gates.
3. Add an experimental runner selector and measure the first real P2048 API
   request with authenticated model weights.
4. Keep the selector only for a positive whole-request result; then run the
   repeated real-request and external EvalScope gates before production
   promotion.

No synthetic timing is an admission or performance result.
