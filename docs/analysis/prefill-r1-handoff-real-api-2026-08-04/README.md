# Prefill R1 producer/consumer handoff real-API result

Date: 2026-08-04

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill

Performance authority: the pinned real checkpoint through the
OpenAI-compatible `/v1/completions` endpoint, driven by external EvalScope
1.9.1. Synthetic inputs are correctness evidence only.

## Decision

Retain the complete default-off R1 handoff package as the new experimental
self-baseline. The same-ELF P1853 direction request lowers server Prefill by
64.02 ms and raises external total throughput by 3.94%. This clears the
incremental self-baseline gate, but it does not clear the package's 100-ms
design target and is not a qualitative Prefill closure.

R1 remains `performance_upper_bound_only=true`,
`quality_production_eligible=false`, and
`production_residency_eligible=false`. The result does not authorize R1 as a
default or production model path. MTP was false. cuBLASLt was neither linked
to nor called by the production route.

## Package

The selected leaf keeps the authenticated R1 Gate/Up and Down GEMMs, then
changes four producer/consumer boundaries:

1. native GDN rows-8 normalization and SiLU gate publish K256 A4 directly for
   Attention-O;
2. Attention residual plus post-norm publish Gate R1 A4 directly;
3. Gate/Up publishes per-N128 product maxima and one finalizer publishes Down
   R1 A4;
4. Down residual plus the next centered RMS publish the next layer's K256
   Attention input directly when the request has one projection span.

The measured P1853 request proved 64 R1 package hits, 64 in-layer handoff
hits, 63 cross-layer handoff hits, 192 native GDN tiles, and 192 direct GDN
K256 publications. Incumbent MLP, R4, prompt-span GDN macro, and MTP counters
were zero.

## Same-ELF real API direction

Both arms used server ELF SHA-256
`d3ea64d96e6d9514044f1292558a2663ef8d427d157f7c41ea6e9beeaeb5ca4e`,
the same natural P1804 warmup and P1853 measurement, concurrency one, one
output token, the same authenticated checkpoint/sidecars, and external
EvalScope.

| Route | Server Prefill | EvalScope throughput | TTFT |
| --- | ---: | ---: | ---: |
| R1 baseline | 1,685.68 ms | 1,096.7227 token/s | 1,690.45 ms |
| R1 handoff | 1,621.66 ms | 1,139.8824 token/s | 1,626.45 ms |
| Change | **-64.02 ms (-3.80%)** | **+43.1597 token/s (+3.94%)** | **-64.00 ms** |

Both requests completed 1/1. The benchmark-summary SHA-256 values are
`bfde072916325ea74b00d67907a066eb8f0bfd4ce56c4784adcb113f2740d80d`
for baseline and
`31748681c1b458941d5ad8eb0d840e90a4365e07f012033a4de98b3713801b91`
for the candidate.

## Request-scoped NSys attribution

One additional real P1853 request was captured with
`--profile-request-index 2`. Its profiler-inflated external TTFT and
throughput are not performance authority. The server recorded 1,643.67 ms of
Prefill. The trace has 2,442 kernel calls and a 1,638.628736-ms raw kernel
sum, versus 2,744 calls and 1,679.437568 ms in the preceding R1 request
profile.

| Kernel family | Current request time | Share |
| --- | ---: | ---: |
| R1 Gate+Up tile-max | 515.064000 ms | 31.4% |
| Attention K256 projections | 403.835136 ms | 24.6% |
| R1 Down | 220.180096 ms | 13.4% |
| Gate product finalizer | 64.396256 ms | 3.9% |
| GDN chunk-o | 62.798176 ms | 3.8% |
| GDN persistent state | 54.836672 ms | 3.3% |
| Down-to-next-layer residual/RMS/K256 publisher | 50.469216 ms | 3.1% |
| Attention-to-Gate residual/RMS/R1 publisher | 39.714624 ms | 2.4% |

The complete top 20 is checked in beside this record. The profile shows why
the handoff result is not a terminal architecture:

- Gate tile-max raises the dominant Gate kernel from the preceding
  487.604160 ms to 515.064000 ms. The direct product finalizer removes most
  of the old two-quantizer cost, but this product-boundary submechanism is
  approximately neutral by itself.
- The material gain comes from eliminating launches and full-buffer traffic
  at the GDN/Attention-O, Attention/MLP, and Down/next-layer boundaries.
- Gate+Up, Attention projections, Down, and the still multi-stage GDN core
  remain the only budgets large enough to support the next qualitative step.
- A scheduler-order-only 4x4 L2 macro-wave was already rejected on the real
  endpoint; it is not reopened around this R1 kernel.

NSys report SHA-256:
`35fe451ac74f4fe0a56f3b74897ef5de7489c2456e15d3b1af6069814fa3690c`.
Exported kernel summary SHA-256:
`afad46945bafcee790fb19abcbf4b3f9ebecda5bb8f2792f8ff7cd19d43253cd`.
Server log SHA-256:
`788108201188f26fcb54c3a12e5a064720da249e40a789f75b779e5b08e772f6`.

## Correctness and capacity closure

The follow-up closes three integration gaps before the next performance
architecture:

- a dedicated `[A4 workspace tokens,136]` FP32 region replaces the fixed
  1-MiB K512 CTA scratch as R1 product-partial backing, so P1921..P4096 no
  longer fail capacity checks;
- the handoff-only native GDN contract admits true C1..C31 state tiles while
  ordinary native admission retains its C32 lower bound;
- a handoff-enabled build without native GDN now fails in the runner factory
  instead of constructing a runner that fails on the first request.

P1853/P2048 Gate product publication is byte-for-byte identical to the old
Gate plus split quantizer over both BF16 planes, the complete packed output,
all scales, the K=12288 plane boundary, padding, guards, and a nondefault
stream. Direct GDN K256 publication is byte-for-byte identical to BF16
norm/SiLU plus the incumbent K256 quantizer for C1, C31, C317, and C512,
including destination slices, padding, guards, capacity rejection, and a
nondefault stream.

## Next architecture gate

No cache-hint, stage-count, CTA-order, or tile-size scan follows this result.
The next candidate must remove work in at least two of the remaining dominant
planes: a new Attention/MLP projection backbone and an exact-BF16-state GDN
fusion that removes global intermediate publication. Its first verdict stays
one real OpenAI API request driven by external EvalScope; only a positive
whole-request direction earns broader statistics and NCU attribution.
