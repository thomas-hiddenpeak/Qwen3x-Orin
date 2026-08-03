# Prefill Attention/Down L2 macro4x4 rejection

Date: 2026-08-03

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill

Performance authority: the real checkpoint through the OpenAI-compatible
`/v1/completions` endpoint, driven by external EvalScope 1.9.1.  Synthetic
inputs were used only for bit-exact correctness and guards.

## Decision

Reject the cross-CTA 4-by-4 L2 macro-wave for both Attention projections and
Down.  The real P1853 direction request regressed server Prefill by 24.96 ms
(+1.31%), regressed external TTFT by 24.92 ms (+1.30%), and reduced total
throughput by 12.4200 token/s (-1.28%).  The implementation remains
default-off and does not replace the current production route.

The rejected EvalScope mode is:

```text
cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-a-exchange-b4-l2-macro4x4
```

## Structural hypothesis that failed

The candidate preserves the incumbent arithmetic, tiles, thread counts and
weight layouts.  It changes the persistent-CTA work order only:

- Attention uses 16 fixed CTAs arranged as four local-M by four local-N
  siblings.  All siblings traverse the same M macro before advancing, aiming
  to keep A and B code/scale planes resident in L2.
- Down uses the same four-M by four-N macro mapping and traverses complete M
  waves before the tail.

There is deliberately no cross-CTA software barrier.  CUDA block scheduling
does not guarantee phase lock among the 16 siblings, so the intended cache
co-residency was a hypothesis that required a real endpoint verdict.

Both implementations passed their pre-performance admission:

| Kernel | Threads | Registers | Dynamic shared | Active CTA/SM | Local/spill |
|---|---:|---:|---:|---:|---:|
| Attention M128N256 A-exchange B4 macro4x4 | 256 | 254 | 149,760 B | 1 | 0 B |
| Down M128N128 16-warp pairring macro4x4 | 512 | 128 | 132,096 B | 1 | 0 B |

Attention was bit exact against its incumbent at P1920 with K5120 and K6144.
Down was bit exact at the exact-model P1909/P1920, N5120, K17408 shapes.
Both suites covered output guards, a nondefault stream and two CUDA Graph
replays.  Launchers fail closed on unsupported grids, shapes and capacities;
the default selector state is unchanged.

## Real OpenAI API and external EvalScope result

The direction gate used one natural P1804 warm-up, one natural P1853 measured
request, one generated token, concurrency one and the authenticated K256/K512
sidecars.

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| Production | 1,916.24 ms | 967.4983 token/s | 1,911.30 ms |
| L2 macro4x4 | 1,941.16 ms | 955.0783 token/s | 1,936.26 ms |
| Change | +24.92 ms (+1.30%) | -12.4200 token/s (-1.28%) | +24.96 ms (+1.31%) |

The candidate warm-up server Prefill was 1,928.39 ms.  Both real requests
proved the intended mutually exclusive production route: 64 Gate+Up pairfeed
launches, 64 macro4x4 Down launches, 128 macro4x4 Attention launches / 208
logical projections, no incumbent Attention or Down launches, and the
expected dynamic GDN token accounting.

## Request-scoped NSys attribution

NSys captured only request index two through the same endpoint and external
EvalScope driver.  Profiler-perturbed EvalScope latency is not used as a
performance result; kernel time attributes the already-negative direction.

| Kernel family | Production | Macro4x4 | Change |
|---|---:|---:|---:|
| Gate+Up pairfeed | 722.001952 ms | 725.475680 ms | +3.473728 ms (+0.48%) |
| Down16 | 288.796640 ms | 292.412832 ms | +3.616192 ms (+1.25%) |
| Attention projections | 404.839904 ms | 423.828896 ms | **+18.988992 ms (+4.69%)** |
| All GPU kernels | 1,911.120448 ms | 1,937.457728 ms | **+26.337280 ms (+1.38%)** |

Attention explains 72.10% of the total kernel-time regression.  The result
rejects scheduler-order-only L2 reuse as a qualitative optimization: it adds
no durable ownership or synchronization mechanism and makes every major
projection family slower.  No macro dimension, cache-hint or phase-offset
scan follows.

The checked-in `kernel-top20.csv` records the candidate profile.  The next
bounded architecture instead changes the Attention resource model itself:
M128N128, 256 threads, half the accumulator width and a hard target of two
resident CTAs per SM with zero spills.

## Evidence identity

```text
direction result root
  /home/rm01/q3x-results/l2-macro4x4-p2k-evalscope-directional-run1
server ELF SHA256
  4ed24dd104bd7bea34ce4d3b56143bfac0b83f98c293939c3fb333603ffe2a03
benchmark summary SHA256
  c1dbd8019cafb0ca1de34b048c3b46813fe5502564da67b0922ac94dc2d8ee1e
EvalScope stdout SHA256
  ce896552d4830ad1ef3abf7ef2dfc00b68319fd3bcd1d4da5b81213e61554403
server log SHA256
  dcb80a675c7a9ce412dbda336794e71adeaca8c02492503699dcaa6491640989
provenance SHA256
  b01b3554e17a5603563a5ee4b7413d89c8a70fe4a23628f7680f50977aa3d7b3

candidate NSys report
  /home/rm01/q3x-results/l2-macro4x4-p1853-nsys-run1.nsys-rep
candidate NSys report SHA256
  94d2b19f109ab5d484e1022f5af5feb70a2c5e96edec7357202ef677e5561edf
candidate NSys SQLite SHA256
  a238cc938026ef456637c367bb9fa762becf3576375b162dc0686e0b0da6bdc0

production NSys report
  /tmp/q3x-gateup-ldmatrix-pairfeed-p1853-profile-64d198a.nsys-rep
production NSys report SHA256
  80aecad0ab6941e5a36860eac6d133b2ab18daa699cf19760a9240d877a02244
```
