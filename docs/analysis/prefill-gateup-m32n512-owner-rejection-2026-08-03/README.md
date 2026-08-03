# Prefill Gate+Up M32N512 owner rejection

Date: 2026-08-03
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Decision

Reject the M32N512 owner Gate+Up architecture at its first real production
direction gate. It remains default off and does not replace the M64N128 K256
LDSM pair-feed production baseline. No B-C-C-B closure, long-output matrix,
synthetic performance sweep, or local cache/stage scan follows this negative
result.

The candidate regressed P1853 server Prefill by 145.57 ms (+7.6163%),
regressed TTFT by 145.54 ms (+7.5951%), and reduced total throughput by
68.2946 tok/s (-7.0589%). A request-scoped NSys capture attributes the entire
regression to the candidate Gate+Up kernel: its 64-layer total is 138.420672
ms (+19.1718%) slower than production, while all non-Gate kernels together
are 2.500768 ms faster.

## Structural hypothesis

One 256-thread CTA owns an M32N512 output quantization group. It temporally
visits eight M32N64 subcells, retains each matching Gate/Up accumulator pair
through the exact K512 scale boundary, places the resulting BF16 products in
one CTA-owned 32-KiB edge plane, and publishes the canonical A4/K512 input for
the unchanged Down kernel.

The operand pipeline uses four independently committed K128 `cp.async.cg`
stages. Eight warps compute M32N64 at a time. Thirty-two persistent CTAs form
16 logical adjacent-M32 teams; the pairing is scheduling order only and has
no inter-CTA correctness dependency. The intended structural gain was two
resident CTAs per SM plus opportunistic L2 reuse between the two owners,
without changing the current exact fused Gate+Up/product/K512-publication
contract.

The route is a fail-closed child of the production pair-feed package. It is
selected only when both
`Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_LDMATRIX_PAIRFEED_ADMISSION`
and
`Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M32N512_OWNER_ADMISSION`
are present. Merely compiling or linking the candidate cannot change
production.

## Correctness and resources

The bounded correctness suite compares the candidate's packed A4 codes and
BF16 K512 scales bit-for-bit with the established reference. It covers:

- logical M117 / launch M128 / N512 / K512;
- M128 / N512 / K512 with two CUDA Graph replays;
- logical M117 / launch M128 / N1024 / K1536 with two Graph replays;
- nondefault streams, tails, guards, short capacities, misalignment, alias
  rejection, and input immutability.

This is deliberately described as bounded bit-exact correctness. It does not
exercise the real P1920 scheduler's `base_waves > 0` branch or the complete
N17408/K5120 shape. Because the real endpoint direction is negative, the
direction-first rule does not justify expanding that test matrix.

The release SM87 kernel satisfies its hard resource gate:

```text
threads/CTA       256
warps/CTA           8
registers/thread  120
dynamic shared    74368 bytes
active CTA/SM         2
stack                 0 bytes
local                 0 bytes
spill loads           0 bytes
spill stores          0 bytes
```

The focused resource, correctness, and host suites pass 3/3. The external
EvalScope harness unit suite passes 79/79.

## External EvalScope P1853 direction result

The direction gate used the real Qwen3.6-27B-NVFP4 checkpoint and
authenticated real-weight sidecars through the runner's OpenAI-compatible
API. External EvalScope issued one P1804 warmup and one measured P1853
request at concurrency one with one generated token. Outside Gate+Up, the
candidate retained the same production GDN, Attention K256 A-exchange B4,
FlashInfer attention, K512 activation contract, and Down16 pair-ring route.

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| LDSM pair-feed production baseline | 1,916.24 ms | 967.4983 tok/s | 1,911.30 ms |
| M32N512 owner | 2,061.78 ms | 899.2037 tok/s | 2,056.87 ms |
| Candidate change | +145.54 ms (+7.5951%) | -68.2946 tok/s (-7.0589%) | +145.57 ms (+7.6163%) |

The warmup server Prefill was 2,056.39 ms. Both requests passed every runtime
route contract. The child intentionally reuses the parent's
`gateup_ldmatrix_pairfeed_launch_hits=64` request counter, so that counter is
not independent evidence of owner execution. The provenance stage marker
identifies `m32n512_owner_k128_b4`, and the profiled kernel name
`q3x_sm87_a4w4_gateup_down_k512_edge_m32n512_owner_kernel` proves that the
child executed 64 times. Down16 executed 64 times and Attention A-exchange
B4 executed 128 launches, as required.

```text
direction result root
  /home/rm01/q3x-results/m32n512-owner-p1853-candidate-run2
candidate server ELF SHA256
  a97a1394370f74302ea4efe72ce63389aec3057a096abdfffd4d8db36b616481
corpus SHA256
  41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af
provenance.txt SHA256
  f06cab9f29186fa7f99c7fb3789290066a274d093e9473e178d0cd030bf43efe
readiness.json SHA256
  c7731a9aa080b46589e6ddb5276c1173210bb20a437a3c8103e64aeabf859578
server.log SHA256
  c066d40391fd1d44fe54e3e80bd12ca9a4e574e3b522bb29d53061203e07be18
EvalScope stdout SHA256
  d0e8e43951fc4b7028667a35bcb5f86eb8b9fa9080a10ffde3f7f6c8a9132835
benchmark_summary.json SHA256
  34d9908f5872695b510d1cc280ffa15f8b65ceb8967cbf4f3f1165f0f97b6edb
performance_summary.txt SHA256
  9ff7024ca7df4752faea7a38501dbef67dab067a4fe8a36c0f2701f7c9ed7872
```

## Request-scoped NSys attribution

The same real model, API, P1853 request, selectors, and external EvalScope
driver were captured after the endpoint rejection. Profiler-inflated API
latency is not used as a performance verdict; only request-scoped GPU kernel
totals are compared with the locked production report.

| GPU family | Production | M32N512 owner | Change |
|---|---:|---:|---:|
| Gate+Up, 64 calls | 722.001952 ms | 860.422624 ms | +138.420672 ms (+19.1718%) |
| Down16, 64 calls | 288.796640 ms | 286.829728 ms | -1.966912 ms |
| Attention projections, 128 calls | 404.839904 ms | 404.366624 ms | -0.473280 ms |
| All GPU kernels | 1,911.120448 ms | 2,047.040352 ms | +135.919904 ms (+7.1121%) |
| All non-Gate kernels | 1,189.118496 ms | 1,186.617728 ms | -2.500768 ms |

Gate+Up accounts for 101.84% of the request-level GPU regression because
the rest of the request became slightly faster. This rules out API overhead,
Down, Attention, and GDN as explanations for the candidate's failure.

The structural reason is now bounded tightly enough to stop. Relative to one
M64N128 pair-feed owner, splitting the same M64 span into two independent M32
owners duplicates Gate/Up weight delivery across the pair. Each owner also
walks eight temporal N64 subcells, doubling A delivery for an equivalent
M64N512 macrocell. Adjacent scheduling may recover some bytes from L2, but it
cannot remove duplicated shared-memory publication, decode, and instruction
work. Two-CTA residency does not repay those costs.

The complete top-20 kernel table is retained in `kernel-top20.csv`.

```text
production NSys report
  /tmp/q3x-gateup-ldmatrix-pairfeed-p1853-profile-64d198a.nsys-rep
production report SHA256
  80aecad0ab6941e5a36860eac6d133b2ab18daa699cf19760a9240d877a02244
candidate profile root
  /home/rm01/q3x-results/m32n512-owner-p1853-profile-run1
candidate NSys report
  /home/rm01/q3x-results/m32n512-owner-p1853-nsys-run1.nsys-rep
candidate report bytes
  298952
candidate report SHA256
  4b219a92e3273b23e641b87b98972823709e42dbfe8e0ae50b2ebb1d511f51a8
candidate SQLite SHA256
  50de9cf7381af1eaf52c89bc5ba2295930eba58e2577473fad9b7cca481bc1c6
profile provenance SHA256
  a038626bff12515b7e8249216853126ecd1c7fd5a1c4722313cd5ec4dea6e588
profile server log SHA256
  c9c36041abbd0b1c4d5b31812fd44b093889a4257daf1b0bfc01a99fb63d2951
profile EvalScope stdout SHA256
  75b951e27274f1e92c335a72f833f00e22197a6617bbac7bcbe1343195c1766f
```

## Carry-forward

Keep the M64N128 K256 LDSM pair-feed route as production. Do not tune cache
operators, pipeline depth, CTA phase ordering, or the M32N512 scheduler on
this rejected skeleton.

The next qualitative design must preserve at least M64 reuse while changing
the dataflow, not merely the tile constants. It must be evaluated against the
whole P1853 budget: even deleting all 722 ms of Gate+Up leaves about 1.19 s,
above the 926.5-ms boundary for 2,000 prompt tok/s. Therefore the next active
plan must couple a credible Gate+Up replacement with a second architectural
lever in the 404.84-ms Attention projection block or the remaining GDN/runtime
path. Its first verdict remains the real checkpoint through the OpenAI API
and external EvalScope; only a positive endpoint direction earns wider
correctness, stability, and promotion work.
