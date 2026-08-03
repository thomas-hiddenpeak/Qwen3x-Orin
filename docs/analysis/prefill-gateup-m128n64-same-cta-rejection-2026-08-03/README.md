# Prefill Gate+Up M128N64 same-CTA rejection

Date: 2026-08-03
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Decision

The M128N64 same-CTA Gate+Up candidate is rejected at the first real-path
direction gate. It remains default off and fail closed; it does not replace
the M64N128 K256 LDSM pair-feed production baseline. No four-request closure,
stability matrix, or synthetic performance sweep follows the negative
end-to-end result.

This rejection is narrower than “M128 reuse failed.” Request-scoped NSys
shows that the two same-CTA projection kernels reduce Gate+Up compute by
60.044352 ms (-8.316%) relative to pair-feed. The separate exact K512
publication kernel costs 67.121088 ms and reverses that saving. The useful
M128 mechanism is retained as evidence, but this publication seam is not a
production architecture.

## Structural hypothesis

One 256-thread CTA owns an M128N64 cell. It computes the complete Gate
projection, hands an exact FP32 Gate plane through shared memory to the same
CTA, reuses the accumulator/register lifetime for the complete Up projection,
and writes the BF16 SiLU product. Two persistent N-major launches cover the
12,288-wide primary and 5,120-wide secondary windows. The existing split
K512 quantizer then publishes canonical A4 codes and BF16 scales for the
unchanged Down16 kernel.

The candidate is selected only by
`Q3X_RUN_A4W4_GATEUP_K512_M128N64_SAME_CTA_ADMISSION`. It changes only the
Gate selector in the current production composition; native GDN, Attention
K256 A-exchange B4, K512 input quantization, and Down16 remain identical.
Each request must prove 64 same-CTA route hits and zero hits for every sibling
Gate implementation.

## Correctness, resources, and integration gates

The focused logical-M117/launch-M128/N-window/K1536 test is bit exact and
covers a persistent CTA consuming two N cells, tails, and guards. The release
kernel satisfies the hard Orin resource contract:

```text
threads/CTA       256
registers/thread  128
dynamic shared    82,688 bytes
active CTA/SM     2
stack             0 bytes
local             0 bytes
spill loads       0 bytes
spill stores      0 bytes
```

The candidate contract, resource, CUDA correctness, runner-host,
engine-control, and server-CLI tests pass 6/6. The complete external
EvalScope harness suite passes 77/77, and `git diff --check` plus shell syntax
validation pass. Independent read-only reviews found no selector, stride,
workspace lifetime, Down handoff, accounting, resource-gate, or harness
defect.

## External EvalScope P1853 direction result

The real-path direction gate used the authenticated checkpoint and
publications, the fixed natural P2K corpus, one P1804 warmup, one measured
P1853 request, concurrency one, and one generated token. Both requests
proved the intended route: same-CTA=64, pair-feed=0, every other Gate leaf=0,
Down16=64, Attention A-exchange B4=128, and prompt-derived native GDN
accounting.

```text
candidate server ELF SHA256 5d43e6a950b8e48fd6f72994f57734984c8e41962fe9f6f4f302c382bb08bc7f
corpus SHA256               41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af
```

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| LDSM pair-feed production baseline | 1,916.24 ms | 967.4983 tok/s | 1,911.30 ms |
| M128N64 same-CTA | 1,928.21 ms | 961.4925 tok/s | 1,923.41 ms |
| Candidate change | +11.97 ms (+0.625%) | -6.0058 tok/s (-0.621%) | +12.11 ms (+0.634%) |

The candidate P1804 warmup reports 1,907.01 ms server Prefill. The measured
regression is on the selected production path rather than a fallback or
synthetic proxy.

```text
result root
  /home/rm01/q3x-results/same-cta-p2k-evalscope1-run1

provenance.txt SHA256
  4df96c79a7288ee99f736e24c8a02d9181e3dd1f000bda9785a7672b12182b6b
readiness.json SHA256
  e07718eab8740e44a01a73f963db0c22e8cf85e15b4156cfbea3ed8a7a28481b
server.log SHA256
  69fecac99d9c5f3998385b605785a12ec523dee27314717f22cec4a24a6a854a
EvalScope stdout SHA256
  361ce2b6539c9a462670a409fd6c88567d190ef2ea5e257efe6ba70a8635ccfb
benchmark summary SHA256
  44363cab5bdf1d2d2452f4c49c797918d5f2bf8013778148c0bf1c1be67be43b
performance summary SHA256
  bb5bf5d3fe08ca4cd19b5d40193b248f696c7b59f8efa333290caa86a88e322a
```

## Request-scoped NSys attribution

NSys captured request index 2 only, after the natural warmup. Loading and
authentication are outside the capture range. Profiler-perturbed API latency
is not used as a performance verdict. The checked-in `kernel-top20.csv`
contains the candidate aggregation.

| Kernel family | Pair-feed profile | Same-CTA profile | Change |
|---|---:|---:|---:|
| Gate+Up compute | 722.001952 ms | 661.957600 ms | **-60.044352 ms (-8.316%)** |
| Separate K512 split publication | 0 ms | 67.121088 ms | +67.121088 ms |
| Gate+Up plus publication | 722.001952 ms | 729.078688 ms | **+7.076736 ms (+0.980%)** |
| Down16 | 288.796640 ms | 288.091168 ms | -0.705472 ms (-0.244%) |
| Attention projections | 404.839904 ms | 404.490944 ms | -0.348960 ms (-0.086%) |
| All GPU kernels | 1,911.120448 ms | 1,917.879008 ms | **+6.758560 ms (+0.354%)** |

The Gate row is 64 fused pair-feed launches versus 128 same-CTA kernel
instances: primary and secondary are separate physical launches, while
request telemetry counts one successful route ownership per model layer.
The unchanged Down and Attention families stay within sub-millisecond
cross-profile movement. The added exact publication pass explains the sign
reversal without relying on timing noise.

```text
baseline NSys report
  /tmp/q3x-gateup-ldmatrix-pairfeed-p1853-profile-64d198a.nsys-rep
baseline report SHA256
  80aecad0ab6941e5a36860eac6d133b2ab18daa699cf19760a9240d877a02244

candidate NSys report
  /home/rm01/q3x-results/same-cta-p1853-nsys-run1.nsys-rep
candidate report bytes
  310024
candidate report SHA256
  f3f868a3d994de811bd0cccf50793fa949e5791dc51107595c6094e38d642cf9
candidate SQLite SHA256
  871f82c34af0b005394b1141c3d795e3d556ea763a0d37852ab3bb072e54a89a
profile provenance SHA256
  c19d107bc09db2a753e34d5d1d2b9460bffbf2b483d4b4bbcc3925e29b1757e9
profile server log SHA256
  dde435f58edd7560fb114472010ca6fdcccc98100010a4cbc2179b281e9c30d9
```

## Carry-forward

Do not scan cache operators, stage count, or CTA ordering on this exact
same-CTA-plus-global-publication composition. Its negative end-to-end result
is already explained.

The measured M128N64 projection core is nevertheless 8.3% faster than the
production Gate kernel and therefore remains a valid mechanism. A successor
must preserve large-M reuse while avoiding a standalone 67-ms publication
pass. It must not repeat the rejected one-CTA/SM M128N512 serialized owner.
The architectural options are a different device-level producer/publication
decomposition or a quantization-group owner that stays at useful occupancy
without retaining an M128N512 product plane.

Even eliminating the entire current 722-ms Gate block would leave about
1.19 seconds at P1853, above the 926.5-ms budget for 2,000 prompt tok/s.
Therefore Gate redesign must proceed as one part of a global plan that also
attacks the 405-ms Attention projection block and the GDN/runner remainder.
Every successor still receives its first performance verdict from the real
checkpoint through the OpenAI API and external EvalScope.
