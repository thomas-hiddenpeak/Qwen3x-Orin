# Shape-separated high-register Prefill MLP package plan

Date: 2026-08-03
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Why this is the next package

The locked production P1853 request is 1,911.30 ms server Prefill and
967.4983 tok/s total throughput. Its request-scoped GPU budget is:

| Plane | Time |
|---|---:|
| Gate+Up | 722.001952 ms |
| Down | 288.796640 ms |
| Attention projections | 404.839904 ms |
| All remaining GPU kernels | 495.481952 ms |
| All GPU kernels | 1,911.120448 ms |

The 2,000 prompt-token/s boundary for P1853 is 926.5 ms. Removing the entire
Gate+Up block would still leave about 1.19 seconds, so no Gate-only local
optimization can close the target. Conversely, the M32N512 owner proved that
occupancy without operand ownership is not useful: it doubled presentation
work and made Gate+Up 19.17% slower despite reaching two CTA/SM.

The next experiment is therefore one shape-separated MLP package. Gate+Up
and Down receive different dataflows, but enter the real runner under one
default-off package selector and receive one end-to-end verdict.

## Gate+Up: M64N256 high-register paired owner

One 256-thread CTA owns the complete M64N512 output-quantization edge and
computes it as two temporal M64N256 cells. Eight warps each own M64N32. A
decoded B fragment is reused across four ordered M16 A panels, while matching
Gate and Up accumulators remain in the same warp through the exact K512
scale/FMA boundary. After each N256 half, the CTA writes BF16-RNE SiLU
products into its local M64N512 edge plane; after both halves it publishes
the canonical A4/K512 Down input directly.

This differs structurally from the production M64N128 pair-feed cell:

- B-fragment consumption moves from M16N32 to M64N32 ownership;
- four M16 panels share each B feed instead of assigning one panel per warp;
- an edge needs two compute cells instead of four;
- 256 high-register threads replace 512 threads capped at 128 registers;
- the product and K512 publication remain producer-local, so no global seam
  is introduced.

It also differs from the rejected M32N512 owner: M64 ownership is preserved,
so adjacent CTAs do not duplicate all Gate/Up weights, and N512 is not walked
as eight narrow N64 cells.

The final resource-feasible operand layout uses three K64 stages so that
only two N8 partial stripes remain register-resident:

```text
one K64 A+GateB+UpB stage    18,432 bytes
three stages                 55,296 bytes
two A/Gate/Up scale slots     2,304 bytes
M64N512 BF16 edge plane      65,536 bytes
two transient N8 scratch     32,768 bytes
total                       155,904 bytes
```

The original all-register transient layout compiled at 255 registers/thread
but spilled.  The lifetime rewrite moves one Gate/Up N8 pair into auxiliary
CTA scratch and another pair into the not-yet-published 32-KiB half-edge.
The hard resource gate is 256 threads, at most 255 registers/thread, zero
stack/local/spill, at most 165,376 dynamic shared bytes, and one active CTA
per SM.  Disabling temporal-cell unrolling prevents NVCC from hoisting the
second cell's address state across the first cell's accumulator lifetime;
the admitted build therefore has no compiler local state.

## Down: M64N256 16-warp pair-ring owner

Down has K=17,408, N=5,120 and is already at 79.23% measured DRAM read
throughput. It must not inherit the Gate pipeline. One 512-thread CTA owns an
M64N256 cell. Its 2-by-8 warp grid gives every warp an M32N32 output tile,
the same 32 persistent FP32 outputs and 32 transient S32 partials per thread
as the admitted production M128N128/16-warp consumer. All N256 outputs remain
live across one complete ordered K traversal; there is no repeated K pass or
global partial-output seam.

The operand pipeline uses four combined K256 A+B stages and two K512 scale
slots:

```text
one M64K256 A plane/stage       8,192 bytes
one N256K256 B plane/stage     32,768 bytes
four combined stages          163,840 bytes
two M64+N256 scale slots        1,280 bytes
total                         165,120 bytes
```

Across the M128N256 area covered by four production M128N128 cells, two new
cells request 32 KiB of aggregate A codes and 128 KiB of aggregate B codes,
versus 128 KiB of A and 128 KiB of B for production. That is a 37.5% logical
operand reduction without sacrificing the production kernel's sixteen active
warps. The persistent work queue is flat over complete cells, so odd natural
M64 tails are distributed across all sixteen SMs rather than assigned to one
fixed M owner.

The hard resource gate is 512 threads, at most 128 registers/thread, zero
stack/local/spill, exactly 165,120 dynamic shared bytes, and one active CTA
per SM. This is not the resource-rejected M128N256 A1+B2 skeleton: it halves
M ownership so the admitted production M32N32 warp accumulator footprint is
preserved, instead of either spilling 64-output warps or repeating the full K
traversal.

## Direction gate

Both kernels must pass only the minimum safe prerequisites before timing:
compile/resource admission, one bounded bit-exact production-shape oracle,
capacity/alias guards, nondefault stream, route isolation, and first-capture
resource preflight.

The first performance verdict is one natural P1804 warmup followed by one
natural P1853 measured request, concurrency one and one generated token,
using the real checkpoint, authenticated real-weight sidecars, OpenAI API,
and external EvalScope. The same request must prove 64 package Gate launches,
64 package Down launches, zero incumbent/sibling MLP launches, 128 retained
Attention A-exchange B4 launches, and prompt-derived native GDN accounting.

The package continuation gate is intentionally material:

```text
production server Prefill       1,911.30 ms
maximum candidate Prefill       1,711.30 ms
minimum absolute improvement      200.00 ms
Gate+Up diagnostic target       <=500.00 ms
Down diagnostic target          <=210.00 ms
```

If the endpoint improvement is less than 200 ms or negative, collect at most
one request-scoped NSys profile to distinguish Gate from Down, archive the
package, and stop. Do not scan tile sizes, stage counts, cache hints, or CTA
orders. If it passes, run the natural P2K four-shape paired closure and only
then use NCU for further attribution.

Even a passing package does not close 2,000 tok/s. The next package must then
attack the remaining Attention projection and GDN/layer-boundary budgets,
with the same real-path-first rule.
