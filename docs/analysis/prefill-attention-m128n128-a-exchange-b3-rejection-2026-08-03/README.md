# Prefill Attention M128N128 A-exchange B3 rejection

Date: 2026-08-03

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill

Performance authority: the real checkpoint through the OpenAI-compatible
`/v1/completions` endpoint, driven by external EvalScope 1.9.1. Synthetic
inputs were used only for bit-exact correctness and guard coverage.

## Decision

Reject the M128N128 A-exchange B3 Attention candidate at the first real-path
direction gate. The real P1853 request regressed server Prefill by 28.13 ms
(+1.472%), regressed external TTFT by 28.20 ms (+1.47%), and reduced total
throughput by 14.0329 token/s (-1.45%). The implementation remains default
off and does not replace the current M128N256 A-exchange B4 production route.

Do not follow this result with B-stage-count, cache-hint, CTA-order or
phase-offset scans on the same M128N128 skeleton. The candidate met its hard
two-CTA-per-SM resource objective, but that resource reduction did not
translate into an end-to-end gain.

The rejected EvalScope mode is:

```text
cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-m128n128-a-exchange-b3
```

## Structural hypothesis that failed

The candidate splits each incumbent M128N256 Attention projection cell into
two M128N128 cells. Eight warps own an M128N128 cell, preserve the incumbent
K256 arithmetic and bit ordering, exchange A once per K step, and pipeline B
through three shared-memory slots. A fixed 32-CTA grid retains the production
N-major work order. The intended qualitative gain was to halve accumulator
width and move from one to two resident CTAs per SM without introducing a
global publication seam.

All three exact-model projection topologies met the release resource gate:

| Topology kernel | Threads | Registers | Dynamic shared | Active CTA/SM | Stack/local | Spill loads/stores |
|---|---:|---:|---:|---:|---:|---:|
| Linear QKV/Z | 256 | 128 | 66,560 B | 2 | 0 B | 0 B / 0 B |
| Full QKV | 256 | 128 | 66,560 B | 2 | 0 B | 0 B / 0 B |
| Attention O | 256 | 128 | 66,560 B | 2 | 0 B | 0 B / 0 B |

The focused CUDA suite was bit exact against the M128N256 incumbent for
Linear QKV/Z and Full QKV at K5120 and Attention O at K6144, using logical
P1853 with launch padding P1920. It covered a nondefault stream, two CUDA
Graph replays, immutable input/output guards, padded-row guards, invalid
unpadded tails, undersized capacities, input/output aliases and all three
projection topologies.

The runtime selector is a child modifier of the production A-exchange B4
selector. Missing-parent, orphan and conflicting-selector configurations fail
closed; the default production selector state is unchanged.

## Real OpenAI API and external EvalScope result

The direction gate used one natural P1804 warm-up, one natural P1853 measured
request, one generated token, concurrency one and the authenticated K256/K512
sidecars.

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| M128N256 A-exchange B4 production | 1,916.24 ms | 967.4983 token/s | 1,911.30 ms |
| M128N128 A-exchange B3 | 1,944.44 ms | 953.4654 token/s | 1,939.43 ms |
| Change | +28.20 ms (+1.47%) | -14.0329 token/s (-1.45%) | +28.13 ms (+1.472%) |

The candidate P1804 warm-up reports 1,931.96 ms server Prefill. Both real
requests proved the intended composition: 64 Gate+Up pair-feed launches, 64
Down16 launches, 128 Attention launches / 208 logical projections through
the selected B3 implementation, no incumbent Attention route, and the
expected prompt-derived native GDN accounting. The four required B3 runtime
stage names and the added child selector are recorded in provenance.

## Request-scoped NSys attribution

NSys captured request index two through the same real endpoint and external
EvalScope driver. Profiler-perturbed TTFT is not used as a performance result;
kernel time only attributes the already-negative direction.

| Kernel family | Production | M128N128 B3 | Change |
|---|---:|---:|---:|
| Gate+Up pair-feed | 722.001952 ms | 719.463808 ms | -2.538144 ms (-0.352%) |
| Down16 | 288.796640 ms | 288.037568 ms | -0.759072 ms (-0.263%) |
| Attention projections | 404.839904 ms | 425.693568 ms | **+20.853664 ms (+5.151%)** |
| All GPU kernels | 1,911.120448 ms | 1,929.539168 ms | **+18.418720 ms (+0.964%)** |

The B3 projection kernels account for 113.22% of the total kernel-time
regression; small Gate+Up and Down movements offset part of it. This isolates
the failure to the candidate Attention structure rather than another model
plane. The checked-in `kernel-top20.csv` records the candidate profile. No
B-stage, cache-hint, CTA-order or phase-offset scan follows.

## Evidence identity

```text
direction result root
  /home/rm01/q3x-results/attention-m128n128-b3-p2k-evalscope-directional-run1

server ELF SHA256
  1d69fed294163a576753191b38aae106f57bf52a93857afe58ce3695b8248ed2
server log
  /home/rm01/q3x-results/attention-m128n128-b3-p2k-evalscope-directional-run1/server.log
server log SHA256
  ed1d6d8ea5dfc26af315c59e6a0497221e13c6cea26a8cbee7862cde52835341
provenance
  /home/rm01/q3x-results/attention-m128n128-b3-p2k-evalscope-directional-run1/provenance.txt
provenance SHA256
  3d2e45458cabf7a34910a8fa2d08f8bd1f84d8248b6836ba8be3c71a83924d93
EvalScope stdout
  /home/rm01/q3x-results/attention-m128n128-b3-p2k-evalscope-directional-run1/p2k/evalscope.stdout
EvalScope stdout SHA256
  0b825f8bb573cee1d35ca178a3b0707390877d4dd0c7b0c0f93596667287f450
benchmark summary
  /home/rm01/q3x-results/attention-m128n128-b3-p2k-evalscope-directional-run1/p2k/pure-prefill-p2k-cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-ldmatrix-pairfeed-down-16warp-pairring-attention-k256-m128n128-a-exchange-b3/parallel_1_number_1/benchmark_summary.json
benchmark summary SHA256
  d125a6629ba1638f48dc1d626ffa81443ab00a3a3edc4f7c776eed1c035d279a
corpus P2K SHA256
  41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af

candidate NSys report
  /home/rm01/q3x-results/attention-m128n128-b3-p1853-nsys-run1.nsys-rep
candidate NSys report SHA256
  979fc4376835e66a87d2fa16e35c2d3e790112caca7b538bdb9a50b696d9e186
candidate NSys SQLite SHA256
  326696aec0e27ee0974af91c0c2b59b180a3eca396c6e3f1e117e83675163399

production NSys report
  /tmp/q3x-gateup-ldmatrix-pairfeed-p1853-profile-64d198a.nsys-rep
production NSys report SHA256
  80aecad0ab6941e5a36860eac6d133b2ab18daa699cf19760a9240d877a02244
```
