# Prefill Down 16-warp production result

Date: 2026-08-02
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Authority: real checkpoint, OpenAI `/v1/completions`, external EvalScope 1.9.1

## Decision

The 16-warp M128N128 K512 pair-ring Down kernel passes the cumulative
production gate and replaces the incumbent 8-warp pair-ring Down kernel in
the current-best experimental bundle.  It remains default-off and fail-closed;
the exact route is unchanged when its selector is absent.

The promoted mode is:

```text
cumulative-prefill-current-best-mlp-k512-edge-m64n128-k256-alternating-down-16warp-pairring-attention-k256
```

Every successful request must prove:

```text
gateup_alternating_launch_hits=64
down_m128n128_ldmatrix_pairring_launch_hits=0
down_m128n128_16warp_pairring_launch_hits=64
```

The structural cell was added by `a55f8c0`; the independent production
selector, accounting, and EvalScope route were added by `49c8ba9`.  Synthetic
inputs were used only for bitwise correctness, padding, CUDA Graph, and
resource tests.  No synthetic timing contributed to this decision.

## Same-ELF P1853 direction gate

Both routes used the same Release server ELF, real authenticated K256 A4 and
K512 MLP publications, P1804 warmup, natural P1853 measured prompt, and the
same external EvalScope command.  The only selector delta was incumbent Down
out and 16-warp Down in.

Server ELF SHA-256:

```text
0925cba6856f9fef46a69e2aa48815165cdb32c34bf6373004c7236d43480ff1
```

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| Incumbent pair-ring Down | 2,103.10 ms | 881.5373 tok/s | 2,098.04 ms |
| **16-warp pair-ring Down** | **1,989.46 ms** | **931.8931 tok/s** | **1,984.45 ms** |
| Change | **-113.64 ms (-5.403%)** | **+5.712%** | **-113.59 ms (-5.414%)** |

The P1804 warmup also improved from 2,090.67 to 1,976.55 ms, a 114.12-ms
reduction.  The two request-level savings differ by only 0.53 ms.

Evidence roots and summary hashes:

```text
baseline  /tmp/q3x-down16-same-elf-baseline-p2k1-49c8
summary   56defe4e0d3b451c7255c18a452fb391d7a9a1b9061adba2538227356ece6dc2
candidate /tmp/q3x-down16-same-elf-candidate-p2k1-49c8
summary   146f9f8e4451493fc728e806b6eb47af0a8f56fb572b0eb4ec094c4df8b32e8e
```

## Natural P2K C-B closure

After the B-C direction run, the candidate and baseline were run in reverse
order over the four natural measured prompts.  All four candidate requests
improved and all five request logs, including the warmup, passed the exact
64/0/64 accounting contract.

| Prompt | Incumbent server Prefill | 16-warp server Prefill | Change |
|---:|---:|---:|---:|
| P1853 | 2,099.03 ms | 1,989.49 ms | -109.54 ms (-5.219%) |
| P1792 | 1,952.46 ms | 1,857.19 ms | -95.27 ms (-4.879%) |
| P2148 | 2,414.87 ms | 2,278.22 ms | -136.65 ms (-5.659%) |
| P1906 | 2,109.40 ms | 1,997.30 ms | -112.10 ms (-5.314%) |
| **Mean** | **2,143.94 ms** | **2,030.55 ms** | **-113.39 ms (-5.289%)** |

External EvalScope reported:

| Route | Mean TTFT | Prompt throughput | Total throughput | Success |
|---|---:|---:|---:|---:|
| Incumbent | 2,149.26 ms | 895.3900 tok/s | 895.8552 tok/s | 4/4 |
| **16-warp** | **2,036.07 ms** | **945.1569 tok/s** | **945.6480 tok/s** | **4/4** |

The reverse-order P1853 results still saved 109.54 ms.  Across the two rounds,
baseline drift was 0.99 ms and candidate drift was about 5.08 ms, far below
the observed 109.5--113.6-ms benefit.

```text
baseline root     /tmp/q3x-down16-same-elf-baseline-p2k4-49c8
baseline summary  1e036c2e7890ce4c36e20b00e706c56907f835f36281aa29a01950fc890dc3ac
candidate root    /tmp/q3x-down16-same-elf-candidate-p2k4-49c8
candidate summary abc7ea8f6737cccdf7a6708a7b1051057c1770a42cfb2117d545a64230bb5b33
```

This is sufficient for cumulative-baseline promotion.  It is four natural
prompt shapes, not four repetitions of one shape, so longer buckets and
release-grade statistics remain later coverage rather than a claim already
made here.

## Request-scoped NSys attribution

NSys captured only request index 2, the measured natural P1853 request after
warmup.  Model loading and authentication are outside the capture range.  The
profiled server Prefill span was 1,994.03 ms; profiler-perturbed EvalScope
throughput is not used as a performance verdict.

| Kernel family | Incumbent profile | 16-warp profile | Change |
|---|---:|---:|---:|
| Alternating Gate/Up | 748.269 ms | 747.108 ms | -1.161 ms |
| Down pair-ring | 399.490 ms | 287.332 ms | **-112.158 ms (-28.075%)** |
| Attention projections | 447.179 ms | 447.115 ms | -0.064 ms |
| All GPU kernels | 2,090.784 ms | 1,976.480 ms | **-114.304 ms (-5.467%)** |

The new Down kernel averages 4.48956 ms over 64 layers and is 1.39034x faster
than the incumbent profile.  Gate and Attention are effectively unchanged,
so the whole-request improvement is directly attributed to the intended
structural replacement.  The new profile also contains 224 D2D copies
totalling 13.441 ms and one 1.912-ms memset.

Profile evidence:

```text
NSys report SHA256  e40fc14572e18b20e3ade42b27d0682bd54fa03620545d28bce168f13a67d7c6
SQLite SHA256       66fd05ec0168850ad200a270fa0deb62ceacdb17dc586334c4836db62bb2b7a3
kernel CSV SHA256   575dba7c248f5a9faf90271067373cf4108dcd24ac632de768ce525a3ce1a8fe
memop CSV SHA256    302448e1096651f5be03a4c0de2c5645e1fb6485bef0ff77bcc7bdc9b11cc5a7
```

## Validation closure

The candidate uses 512 threads, 128 registers/thread, 132,096 bytes of
dynamic shared memory, one CTA/SM, and zero stack/local spill.  Bit-exact tests
cover K512, K1024, and the real Down shape M1853/P1920 N5120 K17408, including
padding, nondefault stream, guards, immutable inputs, and CUDA Graph replay.

The full production build passed the candidate correctness tests plus runner,
engine-control, OpenAI protocol, server CLI/help, Bash syntax, and all 56
EvalScope harness tests.  Two independent reviews found no blocking selector,
padding, accounting, or same-ELF issue.

## Remaining 2K closure

The promoted P1853 result is 931.8931 token/s with 1,989.46-ms TTFT.  The
2,000-token/s budget is 926.50 ms, leaving 1,062.96 ms to remove and a further
2.146x throughput requirement.  This is not close enough for micro-tuning.

The new profile keeps the structural priority order:

1. Gate/Up: 747.108 ms;
2. all Attention projections: 447.115 ms;
3. Down: 287.332 ms;
4. the GDN/layer-boundary macro package.

The independent Attention A-exchange/B4 cell is still correctness-only and
has not entered the production runtime.  It must receive its own selector,
request accounting, and real EvalScope direction gate on top of this promoted
Down16 baseline.  The GDN prompt-span macro must first close its 29-chunk,
tail, state-round/reload, non-degenerate solve-oracle, and SFU-flow contracts
before it is compiled or tested on the production path.
