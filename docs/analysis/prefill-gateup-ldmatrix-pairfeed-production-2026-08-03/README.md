# Prefill Gate+Up LDSM pair-feed production result

Date: 2026-08-03
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Authority: real checkpoint, OpenAI `/v1/completions`, external EvalScope 1.9.1

## Decision

The M64N128 K256 LDSM pair-feed Gate+Up kernel passes the cumulative
production gate and replaces the alternating scalar-feed Gate+Up kernel in
the current-best experimental bundle. It remains default-off and
fail-closed; the exact route is unchanged unless its selector is present.

The promoted mode is:

```text
cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4
```

The incumbent and candidate are sibling implementations. Every successful
request must prove exactly one of these contracts:

```text
baseline:  gateup_alternating_launch_hits=64
           gateup_ldmatrix_pairfeed_launch_hits=0
candidate: gateup_alternating_launch_hits=0
           gateup_ldmatrix_pairfeed_launch_hits=64
both:      down_m128n128_16warp_pairring_launch_hits=64
           attention_k256_m128n256_a_exchange_b4_launch_hits=128
```

The complete default-off kernel, production selector, engine/runner
fail-closed route, request telemetry, and external EvalScope mode were added
by `64d198a`. Synthetic inputs contributed only bit-exact correctness,
capacity/alias guards, CUDA Graph replay, downstream Down-BF16 equivalence,
and resource validation. No synthetic timing contributed to this decision.

## Same-ELF P1853 direction gate

Both routes used the same Release server ELF, authenticated K256 A4 and K512
MLP publications, natural P1804 warmup, natural P1853 measured request, and
the external EvalScope OpenAI API client. The only selector delta was
alternating Gate+Up out and LDSM pair-feed Gate+Up in.

```text
server ELF SHA256 9e380df38c682573c1535cd4e1fcfdcbde4cdc366223e2aea784ce453fa92089
corpus SHA256     41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af
```

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| Alternating feed | 1,945.01 ms | 953.1856 tok/s | 1,939.96 ms |
| **LDSM pair-feed** | **1,916.24 ms** | **967.4983 tok/s** | **1,911.30 ms** |
| Change | **-28.77 ms (-1.479%)** | **+14.3127 tok/s (+1.502%)** | **-28.66 ms (-1.477%)** |

The P1804 warmup also improved from 1,933.44 to 1,902.98 ms, a 30.46-ms
reduction. Both warmup and measured requests passed the mutually exclusive
64/0 Gate accounting contract while retaining Down16, Attention A-exchange,
and native GDN accounting.

```text
baseline root    /tmp/q3x-pairfeed-real-p1853-baseline-64d198a
baseline summary c6965db75f3aa835e92e006e802f9ee6ff7d416375c4974a093f5aeb861137e2
candidate root   /tmp/q3x-pairfeed-real-p1853-candidate-64d198a
candidate summary f0c7e33cdecdf91e25d2d3189023e6ae7a5631640eb91d9aa481ace674663bbc
```

## Natural P2K C-B closure

The four-request closure ran candidate first and baseline second, reversing
the direction-gate order. All four natural prompt shapes improved, all eight
measured requests succeeded, and all ten requests including warmups passed
the exact Gate, Down16, Attention, and GDN accounting contracts.

| Prompt | Alternating server Prefill | Pair-feed server Prefill | Change |
|---:|---:|---:|---:|
| P1853 | 1,941.48 ms | 1,911.53 ms | -29.95 ms (-1.543%) |
| P1792 | 1,815.29 ms | 1,790.34 ms | -24.95 ms (-1.374%) |
| P2148 | 2,226.17 ms | 2,190.97 ms | -35.20 ms (-1.581%) |
| P1906 | 1,952.89 ms | 1,923.48 ms | -29.41 ms (-1.506%) |
| **Mean** | **1,983.9575 ms** | **1,954.0800 ms** | **-29.8775 ms (-1.506%)** |

External EvalScope reported:

| Route | Mean TTFT | Total throughput | Success |
|---|---:|---:|---:|
| Alternating feed | 1,989.73 ms | 967.6572 tok/s | 4/4 |
| **LDSM pair-feed** | **1,959.55 ms** | **982.5678 tok/s** | **4/4** |
| Change | **-30.18 ms (-1.517%)** | **+14.9106 tok/s (+1.541%)** | -- |

P1853 endpoint drift was only +1.52 ms for the baseline and +0.23 ms for the
candidate. The observed 28.66--29.95-ms effect is therefore larger than
same-route drift and preserves its sign in B-C-C-B order.

```text
candidate root    /tmp/q3x-pairfeed-real-p1853-candidate-n4-64d198a
candidate summary b4f8c2f65f3b6c487aa0c5312d3c21626387af01758de82751ad6d37b5a69af6
baseline root     /tmp/q3x-pairfeed-real-p1853-baseline-n4-64d198a
baseline summary  9d76508f7aeca4d86f9cca67b2b2442d59f032ae7a055bc62bc029185e117957
```

## Request-scoped NSys attribution

NSys captured request index 2, the natural P1853 measured request after
warmup. Loading and authentication remained outside the capture range.
Profiler-perturbed EvalScope latency is not a performance verdict.

The incumbent column below reuses the previously promoted A-exchange/B4
request profile. It is a cross-ELF attribution comparison; the unprofiled
same-ELF API results above remain authoritative.

| Kernel family | Alternating profile | Pair-feed profile | Change |
|---|---:|---:|---:|
| Gate+Up | 746.902656 ms | 722.001952 ms | **-24.900704 ms (-3.334%)** |
| Down16 | 285.900544 ms | 288.796640 ms | +2.896096 ms (+1.013%) |
| Attention projections | 404.520736 ms | 404.839904 ms | +0.319168 ms (+0.079%) |
| All GPU kernels | 1,933.605536 ms | 1,911.120448 ms | **-22.485088 ms (-1.163%)** |

The target Gate block is the only large family that improved. Its 24.90-ms
reduction accounts for the cross-profile GPU saving, while Down and Attention
moved slightly in the opposite direction. This closes causal attribution to
the intended shared-memory consumer-feed replacement.

```text
result root        /tmp/q3x-pairfeed-real-p1853-candidate-nsys-64d198a
NSys report SHA256 80aecad0ab6941e5a36860eac6d133b2ab18daa699cf19760a9240d877a02244
SQLite SHA256      f3a123cf3fecd4fd54f90ecdd48f356145ed7c1f7ae7a8531d4aef9c81640096
kernel CSV SHA256  fa332f53675432c46fe002cc1bc52d5688d140569c9d8e70f1ff5c81f9cd3788
```

## Correctness and resource gate

The candidate preserves the incumbent tile, scheduler, K512 arithmetic
order, shared footprint, output quantization, and authenticated v1 payload.
It replaces the complete shared operand feed as one coherent mechanism:

- A fragments use `ldmatrix.x4`;
- independent Gate and Up B fragments use `ldmatrix.x2`;
- scale owners load once and distribute values by warp shuffle.

Static SASS retains 64 IMMA and 26 LDGSTS instructions while replacing the
incumbent's 181 scalar LDS instructions with 72 LDSM instructions and 86
remaining LDS instructions. The candidate uses 128 registers/thread,
148,736 bytes of dynamic shared memory, one active CTA/SM, and no stack,
local memory, or spill. Candidate and incumbent tests both pass bit-exact
K1536/K2048 shapes, tails, guards, alias rejection, nondefault stream, two
CUDA Graph replays, and downstream Down-BF16 equivalence.

## Remaining target and next structural step

The promoted P1853 result is 1,916.24-ms TTFT and about 967 prompt token/s.
At P1853, 2,000 prompt token/s permits 926.50 ms, leaving about 989.74 ms to
remove and another 2.068x latency reduction. Pair-feed is a stable positive
baseline update, not the requested qualitative leap.

No further pair-feed tile or cache-policy scans are justified. The next Gate
work must reduce structural M-direction weight reloads rather than merely
change the shared consumer instruction sequence. The first design gate is a
large-M, real-shape-specific skeleton that extends B/scale lifetime across
multiple M rows while preserving exact FP32 accumulation and selecting a
different configuration for Gate/Up and Down. Global API/EvalScope impact
remains the first performance verdict for that skeleton.
