# Prefill Gate+Up M64N128 register-pipeline rejection

Date: 2026-08-03  
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill  
Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Decision

Reject the projection-major M64N128 register-pipeline Gate+Up route. It was
selected correctly in the complete runner, but the first real P1853 direction
gate regressed server Prefill by 709.56 ms (+37.12%), regressed external TTFT
by 709.64 ms (+37.03%), and reduced total throughput by 261.4603 token/s
(-27.02%). It does not replace the production M64N128 K256 LDSM pair-feed
route, and no tile, cache-hint, or stage-count scan follows this result.

The rejected mode is:

```text
cumulative-prefill-current-best-mlp-k512-projection-major-gateup-down-16warp-pairring-attention-k256-a-exchange-b4
```

## Structural hypothesis that failed

One persistent 512-thread CTA owns M64N512. Warps 0--7 compute Gate and
warps 8--15 compute Up. A uses a two-slot K512 `cp.async` ring, while the
lossless projection-major-v3 Gate/Up B fragments are loaded directly into
registers. Four consecutive M64N128 cells form one producer-owned K512 edge,
which is quantized directly to the signed-A4/BF16-scale Down-input ABI.

The implementation is bit exact and removes neither a hidden fallback nor a
selector ambiguity. Its release resource contract is:

```text
threads/CTA       512
registers/thread  124
dynamic shared    99,584 bytes
active CTA/SM     1
stack/local/spill 0 / 0 / 0 bytes
```

This data flow tried to reduce B shared-memory presentation and retain A
across both projections. In practice, projection-split warp crews, direct
global B fragment consumption, four serialized N128 cells, and one-CTA/SM
latency hiding made the Gate+Up kernel almost twice as slow as production.

## Real OpenAI API and external EvalScope result

The candidate and comparator use the same real checkpoint, authenticated
K256 Attention and K512 MLP publications, natural P1804 warmup, natural P1853
measured request, one generated token, concurrency one, and external
EvalScope. The candidate request proved this exact mutually exclusive route:

```text
GateUp M64N128 register-pipeline launches  64
GateUp LDSM pair-feed launches               0
Down M128N128 16-warp pair-ring launches    64
Attention K256 A-exchange B4 launches      128
```

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| Production LDSM pair-feed | 1,916.24 ms | 967.4983 tok/s | 1,911.30 ms |
| Projection-major register pipeline | 2,625.88 ms | 706.0380 tok/s | 2,620.86 ms |
| Change | +709.64 ms (+37.03%) | -261.4603 tok/s (-27.02%) | +709.56 ms (+37.12%) |

The candidate warmup server Prefill was 2,616.16 ms. Both warmup and measured
requests passed the Gate, Down, Attention, and dynamic GDN accounting
contracts. Synthetic data contributed only bit-exact correctness, guards,
tail coverage, nondefault-stream execution, and leaf CUDA Graph replay; it
had no performance authority.

## Request-scoped NSys attribution

NSys captured request index two, after warmup, through the same real API
route. Profiler-perturbed TTFT is not a performance verdict. Kernel-family
time is used only to locate the real-path regression.

| Kernel family | Production profile | Candidate profile | Change |
|---|---:|---:|---:|
| Gate+Up | 722.001952 ms | 1,433.636416 ms | **+711.634464 ms (+98.56%)** |
| Down16 | 288.796640 ms | 285.497888 ms | -3.298752 ms |
| Attention projections | 404.839904 ms | 404.308704 ms | -0.531200 ms |
| All GPU kernels | 1,911.120448 ms | 2,614.618272 ms | **+703.497824 ms (+36.81%)** |

Gate+Up accounts for 101.16% of the total kernel-time regression; all other
kernel families collectively improved by about 8.14 ms. The negative API
result is therefore entirely attributable to the new Gate+Up data flow, not
to model loading, EvalScope, Down, Attention, GDN, or a routing fallback.

The checked-in `kernel-top20.csv` contains the candidate's first 20 NSys
kernel rows. The production profile is retained by the promoted LDSM
pair-feed record.

## Evidence identity

```text
candidate result root
  /home/rm01/q3x-results/projection-major-p2k-evalscope1-93c08ce-run1

candidate benchmark summary SHA256
  cd2856ead72e503fc13987a796914251d75ccbac4e7254e48592e5c45e656219
candidate server log SHA256
  7842edcd6a777d8a0eeb0278f5f9947852edfb628dc982aaf6de7104f31fda2f
candidate provenance SHA256
  d832cea34d2538088cf9df40b301c85297cabf9f6a7bf55faee6394f4d15c480
candidate EvalScope stdout SHA256
  1bf0b576f72ae1e7c09c32137e171e98206e22a7d28cb430b964e9f63aede1b8

NSys report
  /home/rm01/q3x-results/projection-major-p1853-candidate.nsys-rep
NSys report SHA256
  5912ac8e00d73ace4eb8f1903d4013c38dba861c19b22d26d0d084a426fa2a19
NSys SQLite SHA256
  d43c9c59c88db4ff92872928cd06b6242670209d726aabfa78ed8ddad7a9cc34
```

## Carry-forward

Keep the LDSM pair-feed production baseline unchanged. Preserve the rejected
kernel and authenticated publication as a default-off correctness and
architecture record, but stop work on this skeleton.

The next Gate candidate must change the residency model, not its constants:
two active CTAs/SM, same-warp Gate/Up ownership, staged B reuse, and a bounded
epilogue that does not reserve a complete M64K512 shared edge. Gate/Up and
Down remain separate shape-specific designs. Its first timing verdict is one
real P1853 request through the OpenAI API and external EvalScope; only a
positive whole-run result earns broader closure and profiler work.
