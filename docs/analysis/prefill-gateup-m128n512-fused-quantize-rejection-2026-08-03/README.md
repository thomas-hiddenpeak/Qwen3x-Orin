# Prefill Gate+Up M128N512 fused-quantize rejection

Date: 2026-08-03
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Decision

The M128N512 fused-quantize Gate+Up candidate is rejected at the first
real-production-path direction gate. It remains default off and does not
replace the M64N128 K256 LDSM pair-feed production baseline. No P2K
four-request closure, stability run, or synthetic performance sweep follows
this negative result.

The candidate was correctly selected in the full model, but regressed P1853
server Prefill by 377.43 ms (+19.75%), regressed TTFT by 377.55 ms (+19.70%),
and reduced total throughput by 159.2422 tok/s (-16.46%). This is too large to
justify local tuning of the skeleton.

## Baseline correction

The production pair-feed path already fuses the Gate+Up SiLU product and
canonical K512 activation quantization. It does **not** materialize an
external BF16 product seam or invoke a separate global product quantizer.

Consequently, this candidate did not test removal of such a seam relative to
production. Its actual structural hypothesis was narrower:

- widen ownership from M64 to M128 so Gate and Up can reuse each published A
  slice across a larger output tile;
- make one persistent CTA own the complete M128N512 K512 output group;
- reduce total CTA work and approximately halve Gate/Up weight traffic for the
  equivalent macro-tile while preserving the incumbent fused publication
  contract.

The earlier projection-serial carry-forward phrase about removing a global
BF16 product scratch was prospective for that rejected serial skeleton. It
must not be counted as a benefit over the current pair-feed production path.

## Candidate structure

One persistent 512-thread CTA owns an M128N512 output quantization cell. The
hybrid implementation computes the first three N128 subcells with M16N64 warp
ownership, stores two completed BF16 product planes in 64 KiB of shared
memory, and aliases the retired wide pipeline for the third plane. It then
computes the final N128 as two serialized N64 halves with a reduced pipeline,
derives one exact K512 scale per row, and publishes authenticated-v1 A4 codes
and BF16 scales directly.

The design-time input-traffic estimate is about 65% of the equivalent
pair-feed traffic: Gate/Up B traffic is halved and A requires five sweeps
instead of the pair-feed equivalent of eight. This estimate is an
architectural model, not a measured performance result.

The route is selected only by
`Q3X_RUN_A4W4_GATEUP_K512_M128N512_FUSED_QUANTIZE_ADMISSION`; the build option
and runtime selector are default off.

## Correctness, contracts, and resources

The candidate passed exact A4-code and BF16-scale comparison against the
scalar Gate+Up reference followed by the canonical standalone K512 quantizer.
The cases include:

- M128/N512/K512 with CUDA Graph replay;
- logical-M117/P128/N512/K1536 with canonical tail publication;
- logical-M129/P256/N1024/K512 with persistent-CTA work offsets.

Canaries, invalid capacities and alias rejection, tail codes/scales,
nondefault-stream execution, and graph replay are covered. Five focused tests
passed: the host contract, SM87 resource gate, CUDA correctness test,
reference-runner host test, and reference-engine control test. The external
EvalScope harness unit suite passed 69/69.

The release SM87 kernel satisfies the hard resource contract:

```text
threads/CTA       512
registers/thread  128
dynamic shared    165376 bytes
active CTA/SM     1
stack             0 bytes
local             0 bytes
spill loads       0 bytes
spill stores      0 bytes
```

## External EvalScope P1853 direction result

The direction gate used the real Qwen3.6-27B-NVFP4 checkpoint through the
runner's OpenAI-compatible API and external EvalScope 1.9.1. The workload was
concurrency one, one P1804 warmup, one measured P1853 request, and one
generated token. The candidate warmup server Prefill was 2,288.25 ms.

Outside Gate+Up, both routes retained the same production composition,
including native GDN, whole-span BF16 A/B, FlashInfer attention, Attention
K256 A-exchange B4, MLP K512, and the Down16 pair-ring kernel.

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| LDSM pair-feed production baseline | 1,916.24 ms | 967.4983 tok/s | 1,911.30 ms |
| M128N512 fused quantize | 2,293.79 ms | 808.2561 tok/s | 2,288.73 ms |
| Candidate change | +377.55 ms (+19.70%) | -159.2422 tok/s (-16.46%) | +377.43 ms (+19.75%) |

Both warmup and measured requests prove the intended route:

```text
GateUp M128N512 fused-quantize launches  64
GateUp LDSM pair-feed launches             0
GateUp projection-serial launches          0
GateUp alternating launches                0
Down M128N128 16-warp pair-ring launches  64
Attention K256 A-exchange B4 launches    128
```

The result therefore is not a selector miss or a microbenchmark artifact.
The real service path rejects the structural composition.

## Attribution boundary

The candidate's modeled traffic reduction did not survive its execution
structure. A source-level schedule audit identifies two concrete structural
differences from the production pair-feed kernel:

- pair-feed publishes the even and odd K256 stages as separate async groups;
  `wait_group<1>` exposes even while odd remains eligible to overlap compute,
  and the next even publication overlaps odd compute. The candidate publishes
  both halves of each K512 group and executes `wait_group 0` before any MMA,
  serializing copy and compute at every group;
- pair-feed keeps four M16N32 Gate/Up partial pairs live and traverses the
  physical K64 planes outside the four N8 fragments, so one A fragment feeds
  four B fragments. The candidate's M16N64 ownership keeps only one S32
  Gate/Up partial pair live at a time, traverses the eight N8 fragments on the
  outside, and consequently reloads the same A fragment from shared memory
  once per N8 fragment. Keeping all eight Gate and eight Up FP32 accumulators
  plus all eight Gate/Up S32 partial pairs live would exceed the practical
  SM87 register budget of this already-128-register kernel.

The one-CTA/SM occupancy ceiling, repeated CTA-wide synchronization between
shared-memory phases, and serialized final N128/K512
reduction-and-publication pipeline are additional design-level costs. In
combination, this execution structure overwhelmed the projected A/B traffic
savings.

The schedule observations above come directly from the two kernel sources;
they are not an NCU attribution or a measured latency decomposition. No NCU
metric breakdown is required to reject a +19.75% real-path regression.
Profiling may still be used later to quantify the negative result, but it
cannot promote this skeleton and must not delay the next structural design.

## Evidence and hashes

```text
evidence root
  /home/rm01/q3x-evidence/fused-quantize-real-p1853-20260803-run1

candidate server ELF SHA256
  8b5d240c60ebfa5273c146270d1f4f370ed7377951282ff9237ec1038af39204
corpus SHA256
  41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af

provenance.txt SHA256
  930af20c3b268d1b06db07ca914308ee9f75c3d453e456827cd8e92771a3dd36
readiness.json SHA256
  ba1d75439b84faa71ea8e076efc9e45f54c0f3c9c24d6e39dce690cf8f9be15a
server.log SHA256
  c86fbeb7123632ed1525f66db778eb7608bbfd6d9f10a7817776c9e0745c75eb
EvalScope stdout SHA256
  20d9428318ae5832681568748dfa9953eab73d6cc8a90e934bb561c02c7ffe7f
benchmark_args.json SHA256
  987af1abb10c01f88ca13f65b7cbee09a70622a45f5a1348aaf27b0741848413
benchmark_data.db SHA256
  129a38ebc4904b441d2c459a02b36173965a0b88ad88bf73e758a81536530d35
benchmark_summary.json SHA256
  651d11cecbef33c202f42da0967ef866bc7d3ac5e8c3a0468a21007b9d1198a3
performance_summary.txt SHA256
  1464c014ae7fceb16b080b4b91a7a0cb5683787ae332e7d0171617af27a9bb09
```

The evidence directory retains the harness's historical `p2k` path label,
but this run intentionally contains only the one-warmup/one-measured P1853
direction gate. It is not a P2K four-request closure.

## Carry-forward

Keep the pair-feed production baseline unchanged. Preserve this experiment
default off as a correctness and negative-architecture record, but do not
scan cache policy, pipeline stage count, or barrier placement on it. A future
qualitative candidate must reduce weight traffic without concentrating the
entire N512 quantization group behind one 165,376-byte, one-CTA/SM serialized
owner. Its first performance verdict remains the real checkpoint through the
OpenAI API and external EvalScope; only a positive direction result earns
P2K stability closure and detailed promotion work.
