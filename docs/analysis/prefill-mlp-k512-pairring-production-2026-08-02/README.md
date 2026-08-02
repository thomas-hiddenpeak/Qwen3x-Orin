# Prefill K512 pair-ring Down production result

Date: 2026-08-02
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Authority: real checkpoint, OpenAI `/v1/completions`, external EvalScope 1.9.1

## Decision

The M128N128 LDSM pair-ring Down kernel is retained and combined with the
current M64N128 K256-alternating Gate/Up edge on the authenticated v1 MLP K512
publication.  This is the new production candidate.  It remains explicit and
fail-closed; the exact/default route is unchanged when its selector is absent.

The selected mode is:

```text
cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-pairring-attention-k256
```

It requires both request-scoped launch counters to be exactly 64 for the warmup
and every measured request:

```text
gateup_alternating_launch_hits=64
down_m128n128_ldmatrix_pairring_launch_hits=64
```

No synthetic timing contributed to this decision.

## Real API direction result

The final v1 composition used one P1804 warmup followed by one natural P1853
measurement.  The full production build had SHA-256
`b2eefa3c51b049dc63f5bc5fe3b332a7c013ee0baebe893c88e80fc1ef4ead01`.

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| Retained alternating Gate + canonical Down | 2,218.11 ms | 835.8324 tok/s | 2,212.96 ms |
| Paired M128N512 Gate + canonical Down | 2,250.31 ms | 823.8725 tok/s | 2,245.47 ms |
| Paired M128N512 Gate + pair-ring Down | 2,132.57 ms | 869.3565 tok/s | 2,127.55 ms |
| **Alternating Gate + pair-ring Down** | **2,103.35 ms** | **881.4315 tok/s** | **2,098.45 ms** |

The promoted v1 composition improves the retained baseline by 114.76 ms TTFT
(-5.17%) and 45.5991 token/s (+5.46%).  It also recovers 29.22 ms over the
hybrid Gate+Down composition by removing the rejected M128N512 Gate skeleton.

The pair-ring leaf itself first passed a same-ELF structural direction test:
with the paired Gate held fixed, measured server Prefill fell from 2,245.47 ms
to 2,127.55 ms, a 117.92 ms improvement.  This is consistent with a roughly
23% reduction of the historical 513.211 ms canonical Down budget; the latter
percentage is an inference until the promoted request receives its own NSys
capture.

Evidence:

```text
result root     /tmp/q3x-v1-alternating-pairring-p2k1-b2ee
summary SHA256  124308479914ef06e0db44a55cb43fa51b83933558a63f776916a4e5d09afb2e
server SHA256   ac3aa9092c1f91a8a88f2be5178fb739b8d72f9ec422126b46b172e0bdec7469
provenance SHA  670798d1ef49064527765213c66b77818fdee0d8ec448551e74ead1211383c90
```

## Natural P2K stability gate

Before the v1 composition was wired, the pair-ring leaf was tested with the
paired Gate route over the fixed four-request natural P2K matrix.  Four of four
requests succeeded and every measured prompt length improved server Prefill.

| Route | Mean TTFT | Total throughput | Success |
|---|---:|---:|---:|
| Alternating Gate + canonical Down comparator | 2,261.38 ms | 851.4416 tok/s | 4/4 |
| Paired Gate + pair-ring Down | 2,196.37 ms | 876.6357 tok/s | 4/4 |
| Change | -65.01 ms (-2.87%) | +2.96% | unchanged |

Measured server-Prefill improvements at P1853, P1792, P2148, and P1906 were
79.12, 81.13, 14.47, and 85.43 ms respectively.  The smaller P2148 gain does
not reverse direction.

```text
result root     /tmp/q3x-paired-same-elf-p2k4-gate-down-cf97-rerun
summary SHA256  f686ed2feadfac6ad6f9aa5387ab7021524c329c76a4e1920d37fd0351c30868
server SHA256   a8206dfe3182467fb9afd01e56948afd6623ff1aab3e38ffda4d88ff2e502410
```

One preceding four-request attempt stopped after readiness and before warmup
because the shared harness was edited while Bash was still reading it.  It
contains no benchmark summary or performance sample and is excluded from all
decisions.  The rerun above used a syntax-checked, 53-test harness and a fresh
output root.

## Why pair-ring advanced

For each K512 group, the retained Down mainloop presented operands through
about 640 scalar LDS operations.  The pair-ring consumer replaces that feed
with 16 `LDSM.x4` plus 64 `LDSM.x2` operations while preserving 128 IMMA
instructions and exact K512 arithmetic.  Its dynamic barriers fall from about
69 to 35 per group.  The compiled kernel uses 238 registers/thread, 131,072 B
dynamic shared memory, one CTA/SM, and no stack or local spill.

Bit-exact tests cover K512, K1024, and the real K17408 Down shape, including
natural padded-M scheduling.  The production selector launches this kernel
directly and has no internal fallback.

## Promoted-route NSys budget

A request-scoped NSys capture then profiled the promoted v1 composition on the
same natural P1853 OpenAI request.  The request reported 2,108.37 ms server
Prefill while profiled.  Profiler-perturbed EvalScope throughput is not a
performance verdict.

| Kernel family | Calls | Total time | Request share |
|---|---:|---:|---:|
| Alternating Gate/Up + K512 publication | 64 | 748.269 ms | 35.8% |
| Pair-ring Down | 64 | 399.490 ms | 19.1% |
| Linear-attention QKV+Z projections | 48 | 249.412 ms | 11.9% |
| Attention-O projections | 64 | 124.358 ms | 5.9% |
| Full-attention Q/K/V projections | 16 | 73.410 ms | 3.5% |
| All GPU kernels | - | 2,090.784 ms | 100% |

The pair-ring kernel is 113.721 ms (-22.16%) below the retained canonical Down
profile of 513.211 ms.  This closes the earlier request-level inference with a
direct kernel measurement.  Gate/Up plus Down still consume 1,147.759 ms, or
54.9% of all captured GPU time; the three Attention projection families add
447.179 ms.  The next qualitative steps therefore remain Gate/Up dataflow and
then the grouped Attention projection plane, not the already-small 32.593 ms
FlashInfer attention core.

```text
NSys report SHA256  5a0a55544ac93c8b332400374c7c9f02434ba9af69d6dc573337c24a9343abe5
SQLite SHA256       91c49499e4e6fb3b80807421e6bc5fe6d4451052da88428d1f20052bf2296eab
kernel CSV SHA256   fd387475d832afc4349b788f82f62ac51c6fff7cc29756d0e94ac00fa50da596
```

## Rejected paired Gate skeleton

The paired M128N512 Gate candidate is not promoted even though its publication
and LDSM mechanisms are valid.  On the same real P1853 request it regressed
TTFT by 32.20 ms.  Request-level NSys measured its Gate family at 779.102 ms
versus the retained 773.339 ms family.

The structural cause is the exact K512 product boundary.  M128 output needs a
128-KiB BF16 product plane while the mainloop also needs 66,560 B of pipeline
shared memory.  The implementation therefore writes and reloads one 64-KiB
M-half through global scratch.  Across 64 layers this adds about 4.28 GB of
scratch traffic on the serial publication path.  Cache hints, stage-count
scans, or L2 persistence cannot remove that traffic, so this skeleton is
closed.

## Production closure

Two independent reviews found no P0/P1 issue.  Direct-engine and one-shot
startup both reject missing masters, missing compile support, fragment/hybrid
mixing, and conflicting Down selectors before model I/O.  The v1 and hybrid
routes remain disjoint.  Request accounting synchronizes the CUDA stream and
requires exact Gate and Down deltas before returning a successful API result.

Validation completed:

- candidate-ON and candidate-OFF builds;
- runner host/control tests in both build closures;
- EvalScope harness 53/53;
- Bash syntax and `git diff --check`;
- full production ELF dry-run with the authenticated v1 and K256 triplets;
- real OpenAI/EvalScope warmup and measured request with exact 64/64 hits.

## Next structural target

The current P1853 rate is 881.43 token/s, still 2.27x below the 2,000 token/s
checkpoint.  The next Gate/Up candidate will not tune the closed M128N512
scratch skeleton.  It will test M128N128 full-K512 ownership with a
shape-specific A/B pipeline and an explicit BF16 product boundary, accepting
one CTA/SM when required to obtain real M reuse.  Its first performance gate
remains one real OpenAI request through external EvalScope; only a positive
whole-request result earns broader correctness, NSys/NCU, and stability work.
