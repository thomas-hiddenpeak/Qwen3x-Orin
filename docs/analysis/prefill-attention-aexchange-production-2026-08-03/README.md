# Prefill Attention A-exchange/B4 production result

Date: 2026-08-03
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Authority: real checkpoint, OpenAI `/v1/completions`, external EvalScope 1.9.1

## Decision

The M128N256 K256 Attention A-exchange/B4 kernel passes the cumulative
production gate and replaces the incumbent M128N256 K256 Attention projection
kernel in the current-best experimental bundle.  It remains default-off and
fail-closed; the exact route is unchanged when its selector is absent.

The promoted mode is:

```text
cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256-a-exchange-b4
```

Every successful request must prove one of the following mutually exclusive
Attention accounting contracts while retaining the promoted MLP route:

```text
baseline:
  attention_k256_m128n256_incumbent_launch_hits=128
  attention_k256_m128n256_incumbent_logical_projection_hits=208
  attention_k256_m128n256_a_exchange_b4_launch_hits=0
  attention_k256_m128n256_a_exchange_b4_logical_projection_hits=0

candidate:
  attention_k256_m128n256_incumbent_launch_hits=0
  attention_k256_m128n256_incumbent_logical_projection_hits=0
  attention_k256_m128n256_a_exchange_b4_launch_hits=128
  attention_k256_m128n256_a_exchange_b4_logical_projection_hits=208

both:
  gateup_alternating_launch_hits=64
  down_m128n128_ldmatrix_pairring_launch_hits=0
  down_m128n128_16warp_pairring_launch_hits=64
```

The structural cell was introduced by `a55f8c0`; the independent production
selector, fail-closed engine/runner route, request accounting, and EvalScope
mode were added by `a61d5e8`.  Synthetic inputs were used only for bit-exact
correctness, padding, guards, CUDA Graph replay, and resource validation.  No
synthetic timing contributed to the production decision.

## Same-ELF P1853 direction gate

Both routes used the same Release server ELF, real authenticated K256 A4 and
K512 MLP publications, P1804 warmup, natural P1853 measured prompt, and the
same external EvalScope command.  The only selector delta was incumbent
Attention out and A-exchange/B4 Attention in; the 16-warp Down baseline stayed
enabled in both routes.

Server ELF SHA-256:

```text
33dc63590ea3b920373c6122e2f20b8285341056bd687d57ae8b6876c5879169
```

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| Incumbent Attention | 1,988.76 ms | 932.2195 tok/s | 1,983.80 ms |
| **A-exchange/B4 Attention** | **1,946.45 ms** | **952.4815 tok/s** | **1,941.56 ms** |
| Change | **-42.31 ms (-2.127%)** | **+20.2620 tok/s (+2.174%)** | **-42.24 ms (-2.129%)** |

The P1804 warmup also improved from 1,976.26 to 1,932.78 ms, a 43.48-ms
reduction (-2.200%).  Both measured and warmup requests satisfied the exact
128/208 Attention accounting swap and retained Gate=64 and Down16=64.

Evidence roots and benchmark-summary hashes:

```text
baseline  /tmp/q3x-attention-aexchange-same-elf-baseline-p2k1-a61d
summary   53ff32e27bb616db082d90d576e47208b3283f58195993d0c79b8da5ce4d01f0
candidate /tmp/q3x-attention-aexchange-same-elf-candidate-p2k1-a61d
summary   d0d8ddcea5359741c4b26fe20b146f528a21722277b90e665975ab81bcd78ca6
```

## Natural P2K C-B closure

The four-request closure ran candidate first and baseline second, reversing
the direction-gate order.  All four natural prompt shapes improved, all eight
measured requests succeeded, and every request passed the mutually exclusive
128/208 Attention accounting contract plus Gate=64 and Down16=64.

| Prompt | Incumbent server Prefill | A-exchange/B4 server Prefill | Change |
|---:|---:|---:|---:|
| P1853 | 1,987.30 ms | 1,937.47 ms | -49.83 ms (-2.507%) |
| P1792 | 1,858.45 ms | 1,813.33 ms | -45.12 ms (-2.428%) |
| P2148 | 2,278.38 ms | 2,226.06 ms | -52.32 ms (-2.296%) |
| P1906 | 1,997.95 ms | 1,949.37 ms | -48.58 ms (-2.431%) |
| **Mean** | **2,030.52 ms** | **1,981.5575 ms** | **-48.9625 ms (-2.411%)** |

External EvalScope reported:

| Route | Mean TTFT | Total throughput | Success |
|---|---:|---:|---:|
| Incumbent Attention | 2,035.92 ms | 945.7218 tok/s | 4/4 |
| **A-exchange/B4 Attention** | **1,986.93 ms** | **969.0290 tok/s** | **4/4** |
| Change | **-48.99 ms (-2.406%)** | **+23.3072 tok/s (+2.464%)** | -- |

The reverse-order P1853 result still saved 49.83 ms.  Between the direction
and closure rounds, baseline P1853 server drift was 3.50 ms and candidate
drift was 4.09 ms, both far below the observed 42.24--49.83-ms benefit.

```text
candidate root    /tmp/q3x-attention-aexchange-same-elf-candidate-p2k4-a61d
candidate summary 3a665102539c93c3ad8e7d3b4131ec58620c622bc9d58ba8be478223acf2fd5f
baseline root     /tmp/q3x-attention-aexchange-same-elf-baseline-p2k4-a61d
baseline summary  8c0f69772ee41caddecee8d31c784e1261fca72c42bc8dc6ee84e7f2596aa644
```

This is sufficient for cumulative-baseline promotion.  It covers four
natural prompt shapes rather than four repetitions of one shape.  Longer
buckets and release-grade statistical coverage remain future validation and
are not claimed here.

## Request-scoped NSys attribution

NSys captured request index 2, the measured natural P1853 request after
warmup.  Model loading and authentication were outside the capture range.
The candidate profile's server Prefill span was 1,951.12 ms; profiler-perturbed
EvalScope throughput is not used as a performance verdict.

The incumbent column below is the previously promoted Down16 request profile,
whereas the candidate column is the new A-exchange/B4 request profile.  This
cross-profile comparison is attribution evidence, not the same-ELF production
gate; the unprofiled API runs above remain authoritative for the decision.

| Kernel family | Incumbent profile | A-exchange/B4 profile | Change |
|---|---:|---:|---:|
| Alternating Gate/Up | 747.108064 ms | 746.902656 ms | -0.205408 ms (-0.027%) |
| Down 16-warp pair-ring | 287.331872 ms | 285.900544 ms | -1.431328 ms (-0.498%) |
| Attention projections | 447.114848 ms | 404.520736 ms | **-42.594112 ms (-9.526%)** |
| All GPU kernels | 1,976.480224 ms | 1,933.605536 ms | **-42.874688 ms (-2.169%)** |

The candidate Attention total is composed of 228.659808 ms over 48 Linear
launches, 66.818080 ms over 16 Full launches, and 109.042848 ms over 64 O
launches.  Gate is effectively unchanged, and the 42.594-ms Attention saving
accounts for almost all of the 42.875-ms all-kernel reduction.  This directly
attributes the whole-request improvement to the intended structural swap.

Profile evidence:

```text
result root         /tmp/q3x-attention-aexchange-same-elf-candidate-p2k1-nsys-a61d
NSys report SHA256  700b74cea08e573ed124a100a9b76050967e55ccfc4440e70f17fceea38d5170
SQLite SHA256       d1a0d2b84115d1a6d345c5f4f0ba74f092a3222ae6a498865f07cac3b8a17541
kernel CSV SHA256   0003140e1210adbb3e784a16d58f823c830253b2b4852fa6e011a5abd7a1a48a
memop CSV SHA256    d94b93b5b4ab214e050579b2b402cfd894a501a651183832e1369dcfbd89f313
```

## Correctness and integration validation

Synthetic correctness and production performance are separate gates.  The
synthetic test compares the candidate bit-for-bit with the incumbent across
all three fixed production topologies: Linear and Full at K5120, and O at
K6144, using P1920 padding.  It also checks nondefault-stream execution,
invalid padding/capacity rejection, input immutability, inner and outer
guards, and two CUDA Graph replays.

The structural kernel uses 256 threads (eight warps), an M32N128 per-warp
consumer tile, one register-handoff A slot, four long-lived B slots, and
149,760 bytes of dynamic shared memory.  Its resource gate requires no local
memory or spill, at most 255 registers/thread, and one active CTA/SM on SM87.
These synthetic results establish correctness and launch viability only;
they do not establish or contribute any timing claim.

The production integration is independently selectable, mutually exclusive
with the incumbent Attention selector, and rejected during engine preflight
when the build, global Prefill enable, or selector contract is invalid.  Host
runner/engine-control coverage, OpenAI server telemetry, EvalScope harness
coverage, and Bash syntax validation cover the route and its fail-closed
accounting surface.

## Remaining 2K closure

The external P1853 baseline is now 1,946.45-ms EvalScope TTFT with 952.4815
total token/s.  At P1853, 2,000 prompt token/s permits 926.50 ms, so the
external result still needs 1,019.95 ms removed, or another 2.1009x latency
reduction.  The corroborating server span is 1,941.56 ms, leaving 1,015.06 ms
and a 2.0956x requirement.  The project has not reached 2K token/s.

This gap remains structural, not a micro-tuning regime.  The next priorities
are the 746.903-ms Gate/Up block and the GDN prompt-span macro, where removing
cross-kernel traffic, launch boundaries, and repeated state materialization
can change the whole-request dataflow.  Further small Attention or Down
adjustments should remain deferred until the cumulative runner is close to
the target.
