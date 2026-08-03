# Prefill Attention R1 consumer-native architecture gate

Date: 2026-08-04

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill.

Performance authority is one real checkpoint request through the
OpenAI-compatible API, driven by external EvalScope. Synthetic matrices have
correctness and smoke authority only.

## Decision

The next qualitative candidate is the complete Attention projection family
under a factorized-lane R1 performance upper bound, together with native
producer/consumer boundaries. A projection-only result is a mechanism probe;
it is not the package that receives the qualitative gate.

The current real P1853 handoff self-baseline is 1,621.66 ms of server Prefill
and 1,139.8824 token/s from EvalScope. The complete candidate must remove at
least 150.00 ms in its first same-ELF direction run:

- required server Prefill: at most 1,471.66 ms;
- P1804 natural warmup followed by P1853 measurement;
- concurrency one and `max_tokens=1`;
- identical model, request corpus, server ELF, and non-candidate selectors;
- all 208 Attention logical projections must hit the new route;
- K256 Attention projection, cuBLASLt, MTP, and fallback hits must be zero.

If the complete package misses this line, capture one request-scoped NSys
attribution and close the architecture. Do not follow it with cache, tile,
stage, or CTA-order scans. A positive projection-only probe may be retained as
an implementation self-baseline, but it does not relax the complete-package
gate.

## Real API result

The first consumer-native subset was measured through the real
OpenAI-compatible endpoint with external EvalScope 1.9.1.  It includes the
Linear-Attention normalized-input handoff and direct GDN-to-Linear-O R1
publisher; Full-Attention producer fusion and an Attention-O residual
publisher are not implemented and must not be described as deployed.

| Path | Server Prefill | EvalScope total throughput | TTFT |
| --- | ---: | ---: | ---: |
| locked MLP R1 handoff baseline | 1,621.66 ms | 1,139.8824 token/s | 1,626.45 ms |
| Attention R1 projection mechanism | 1,556.36 ms | 1,187.4899 token/s | 1,561.25 ms |
| consumer-native subset | **1,508.59 ms** | **1,224.6199 token/s** | **1,513.90 ms** |

The subset saves 113.07 ms (6.97%) from the locked baseline and 47.77 ms
from the projection-only mechanism.  All request-local route contracts were
observed on both the natural P1804 warmup and measured P1853 request:

- Attention R1: 128 launches / 208 logical projections;
- MLP R1 package / handoff / cross-layer handoff: 64 / 64 / 63;
- native GDN launches / direct R1 publications: 192 / 192;
- old Attention K256, GDN K256 direct publication, R4, and MTP: zero.

The result misses the 1,471.66 ms package gate by 36.93 ms.  Therefore the
M128N256 R1-v1 projection-plus-current-boundary architecture is closed as a
positive experimental self-baseline, not promoted.  The authoritative
artifact is:

`/home/rm01/q3x-results/attention-r1-consumer-native-real-api-p1853-run1`

The server ELF SHA-256 is
`6a7720452e89579140acfab74533a2e66b904589777f891b2a483fe499b5669c`.

## Next structural candidate

Do not shrink M merely to claim two resident CTAs.  At P1920 that duplicates
B presentation by 25--67% for the model's Gate, Down, and Attention shapes,
and prior real measurements already rejected the M128N128 Attention two-CTA
shape against M128N256/B4.

The next projection-plane package keeps each shape's proven reuse geometry:

- Gate: M128N128, 512 threads, K256 ping-pong;
- Down: M256N128, 512 threads;
- Attention: M128N256, 256 threads, A1+B4.

It changes the offline R1 B publication and all three consumers together to
an equal-byte fragment-native paired feed.  Gate pairs Gate/Up fragments;
Down and Attention pair adjacent N8 fragments.  This preserves global A/B
bytes, wide-M B reuse, CTA raster, whole-K S32 order, and the one-scale
epilogue while reducing hot B `ldmatrix` issue count by 8--16% depending on
family.  A previous real-model API candidate demonstrated a 9.68% Gate
reduction from the same paired-feed mechanism, so the package has a measured
directional basis rather than a synthetic estimate.

The v2 package is accepted only after all 400 R1 projections use the new
publication and the same real API/EvalScope gate is rerun.  A positive result
updates the self-baseline; cuBLASLt remains reference-only and can never enter
the runtime path.

## Why this is one system candidate

The current request-scoped profile assigns 403.835136 ms to K256 Attention
projections. Those kernels fold one S32 K256 partial into FP32 with A/B scales
20 times for K=5120 and 24 times for K=6144. The R1 upper-bound kernel keeps
the same signed-W4 consumer order and ordered K64 MMA sequence, retains S32
for the whole K dimension, then applies one activation scale and one weight
scale in the epilogue.

Measured MLP R1 effective rates predict only 78--125 ms from the projection
replacement itself. Therefore the first qualitative candidate also removes
whole-plane boundaries:

1. Linear QKV projection owns an N cell while walking M in order, retains the
   three-row causal-convolution halo, applies the exact BF16 convolution and
   SiLU seam, and publishes the GDN consumer layout without a raw-QKV plane.
2. Full Q/K/V projection publishes the exact BF16 preprocess/gate and cache
   layouts without rereading complete raw Q/K/V planes.
3. Attention O publishes the residual row contribution and reduction partial
   needed by the existing centered-RMS/R1 Gate handoff, avoiding a complete O
   write/read boundary.
4. GDN, full-attention output, and the previous layer's Down publisher emit
   R1 inputs directly. K=6144 is part of the activation-producer ABI; no O
   projection is allowed to fall back to K256 quantization.

The credible combined pool is approximately 152--214 ms. This estimate is a
go/no-go budget, not performance evidence.

## Fixed sidecar inventory

The overlay is Attention-only and independent from the 192-projection MLP R1
overlay. It contains 208 projections in the model's actual layer order:

- 48 Linear layers: QKV `[10240,5120]`, Z `[6144,5120]`, O `[5120,6144]`;
- 16 Full layers: Q `[12288,5120]`, K `[1024,5120]`, V `[1024,5120]`,
  O `[5120,6144]`.

With the existing 256-byte factorized-lane projection contract, the R1
payload is exactly 3,614,363,648 bytes:

| Projection | Bytes |
| --- | ---: |
| Linear QKV | 26,255,616 |
| Linear Z | 15,761,664 |
| Full Q | 31,502,592 |
| Full K or V | 2,644,224 |
| O | 15,763,712 |
| One Linear layer | 57,780,992 |
| One Full layer | 52,554,752 |

The converter consumes only an authenticated K256 publication and binds its
manifest, policy, payload, and exact receipt bytes before deriving R1 N64
blocks. Linear QKV/Z must share one authenticated activation factor and clip;
Full Q/K/V must share another; O has its independent K=6144 factor.

## Qualification boundary

R1 is permanently marked `performance_upper_bound_only=true`,
`quality_production_eligible=false`, and
`production_residency_eligible=false`. It cannot become a default or
production path even if it passes the performance gate.

Passing the gate authorizes a calibrated R2/R4 or mixed-lane replacement and
the external capability matrix. Failing the gate means the full-K
factorization plus consumer-boundary hypothesis is insufficient; it does not
authorize production use of R1 or cuBLASLt.
