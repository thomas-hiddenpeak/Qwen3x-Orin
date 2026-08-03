# Prefill direct R4 factorized-lane rejection

Date: 2026-08-03

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill

Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Decision

Reject the current direct R4 `M128N64` Gate+Up plus `M192N128` Down execution
skeleton at its first real-API direction gate.  It remains default off,
`performance_candidate_only=true`, and ineligible for production and quality
promotion.  Its publication, admission, correctness and route-accounting code
is retained as an authenticated negative architecture and as a future quality
reference; it does not replace R1 or any locked default route.

There is no tile, pipeline-stage, cache-hint or CTA-order scan around this
negative result.  The next candidate must change the projection-plane resource
and producer-consumer structure.  MTP remains excluded and cuBLASLt remains a
reference oracle with no production eligibility.

## Authenticated real-weight candidate

Commit `a018c1fc841083b01fb86e4bf41f0ae745cb12c7` adds the direct-checkpoint
publisher and the complete model-weights, engine, runner and API path.  The
publisher authenticated every byte of all three pinned checkpoint shards
before it created the payload, then revalidated the sources after conversion
and hashed the complete payload before publishing four read-only files.

| Item | Value |
| --- | --- |
| pinned source bytes hashed | 21,921,697,184 |
| source MLP bytes converted | 9,625,928,448 |
| projections | 192 |
| N64 blocks | 39,936 |
| payload bytes | 8,583,954,432 |
| payload SHA-256 | `a472d8c0fbbe31add532835bfb1104bb384ee23c36cf400006afeaa6e6ae153e` |
| manifest file SHA-256 | `f6c87568df03108b5b84aff08ef9f069f9f004480b83881aefbb95968e6ef10e` |
| policy file SHA-256 | `ab1c13e29a44bfc90242fd408a31cfcd829450a43b16dfeb4edf5d4d5ca3a235` |
| receipt file SHA-256 | `0122980c8690699074ae3941811f55b54dcdc79ea7e2aae3fd8cee3f6d1e4404` |
| factor scheme | `identity_alpha_f32_v1` |

The server independently verified all 192 metadata records, synthesized and
authenticated only the two fixed built-in identity factor vectors, performed
two complete payload digest passes around H2D, and reported 64/64 attached
layers.  The measured requests proved `factor_files=0`,
`authenticated_builtin_factors=2`, R4 hits 64, R1 hits zero, Attention K256
A-exchange-B4 hits 128/208, and `mtp=false`.

## External EvalScope direction result

The test used the natural ShareGPT P2K corpus with SHA-256
`41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af`.
P1804 was the only warmup and P1853 the only measured request.  Concurrency was
one, output length one, Prefix cache was absent and the client called the real
OpenAI-compatible completion endpoint.  The candidate ELF SHA-256 was
`d6cf356b8498d0c4461fa2efb5a7322d588f411400018da0a5249d1025b703b1`.

| Route | P1853 TTFT | Total throughput | Server Prefill |
| --- | ---: | ---: | ---: |
| locked R1 P1853 reference | 1,675.33 ms | 1,106.6259 token/s | 1,670.42 ms |
| direct R4 candidate | 2,227.94 ms | 832.1429 token/s | 2,223.02 ms |
| R4 change | +552.61 ms (+32.99%) | -274.4830 token/s (-24.80%) | +552.60 ms (+33.08%) |

The locked R1 row is the prior real P1853 direction result with the same
corpus and runtime composition; it is not relabelled as a contemporaneous
same-ELF A/B.  The request-scoped kernel comparison below makes the attribution
stronger: every unchanged category stayed within 0.5%, while both R4 GEMMs
regressed by about 76%.

The R4 warmup server Prefill was 2,208.32 ms.  The result is 2.403x below the
2,000-token/s milestone and is too negative to justify repeated timing or a
length matrix.

## Request-scoped NSys attribution

After the negative API result, one second server launch profiled only the
measured P1853 request with `--profile-request-index 2`.  Profiler-inflated
external TTFT is not a performance authority; the server recorded 2,237.41 ms
of Prefill and route counters again proved the exact R4 path.

The trace contains 2,744 kernel intervals with no overlap:

| Item | Time |
| --- | ---: |
| kernel raw sum and interval union | 2,219.219456 ms |
| kernel span | 2,237.349824 ms |
| kernel span idle | 18.130368 ms |
| all GPU-operation raw sum and union | 2,235.021280 ms |
| all GPU-operation span | 2,239.389760 ms |
| all GPU-operation span idle | 4.368480 ms |
| memcpy plus memset | 15.801824 ms |

The category comparison against the locked R1 request is decisive:

| Category | R1 | R4 | Change | R4 kernel share |
| --- | ---: | ---: | ---: | ---: |
| Gate+Up | 487.604160 ms | 860.028928 ms | +76.38% | 38.75% |
| Down | 221.828032 ms | 390.843168 ms | +76.19% | 17.61% |
| factorized quantizers | 89.017024 ms | 86.423744 ms | -2.91% | 3.89% |
| Attention projections | 404.077856 ms | 404.363648 ms | +0.07% | 18.22% |
| Attention quantizers | 28.579904 ms | 28.643072 ms | +0.22% | 1.29% |
| GDN hierarchy | 249.540672 ms | 249.537376 ms | -0.00% | 11.24% |
| residual and RMSNorm | 117.727968 ms | 118.208672 ms | +0.41% | 5.33% |
| FlashInfer Attention core/pre/post | 56.228928 ms | 56.326528 ms | +0.17% | 2.54% |
| fixed and other kernels | 24.833024 ms | 24.844320 ms | +0.05% | 1.12% |

R4 MLP therefore owns 1,337.295840 ms versus R1's 798.449216 ms.  The
538.846624-ms MLP increase explains the whole-request regression; Attention,
GDN and residual work are stable controls rather than alternative causes.
The exact top-20 kernel list is in
[`kernel-top20.csv`](kernel-top20.csv).

## Structural cause

The regression matches the compiled resource and issue geometry:

| Shape | R1 cell | R4 cell | R1 resources | R4 resources |
| --- | --- | --- | --- | --- |
| Gate+Up K5120/N17408 | M128N128, 16 warps | M128N64, 8 warps | 128 regs, 99,072 B shared | 242 regs, 163,840 B shared |
| Down K17408/N5120 | M256N128, 16 warps | M192N128, 12 warps | 122 regs, 99,072 B shared | 168 regs, 122,880 B shared |

R4 Gate doubles the work-cell count from 2,040 to 4,080.  Its four-lane
quality semantics require current S32 partials plus cross-lane FP32 state; the
shared cross-tail is read and written at every lane fold.  Gate A+B issue rises
from about 2.005 to 2.674 GB/layer (+33.3%), while the register/shared-memory
cliffs leave only one eight-warp CTA per SM.  Four async stages hide latency
but cannot remove this traffic or supply missing resident warps.

R4 Down changes eight M owners into ten and raises issued A+B bytes from about
1.070 to 1.114 GB/layer (+4.2%).  The additional FP32 lane folds and reduced
warp residency amplify that modest traffic increase into the measured 76.19%
latency regression.

## Carry-forward

Do not tune the current R4 M128N64/M192N128 skeleton.  The next
`projection-plane v2` milestone has these system constraints:

1. Gate, Down and Attention use shape-specific schedules; no common tile is
   inherited across K5120/N17408, K17408/N5120, linear/full Attention and O.
2. The first resource proof must escape the current one-CTA/eight-warp cliff,
   preferably by reaching two CTA/SM or by demonstrating an equivalent
   latency-hiding and issue-rate change.  Zero stack/local/spill remains hard.
3. Gate's current N64 A replay and cross-lane accumulator lifetime must change
   together.  Merely changing `M`, stage count or cache operators is not a new
   architecture.
4. Attention projections plus their quantizer already cost 433.006720 ms;
   GDN, residual/RMS and Attention core add another 424.072576 ms.  The P1853
   2K budget is 926.50 ms, so an MLP-only package cannot close the target.
5. The minimum system package must therefore include a low-resource large-M
   Attention projection backbone and at least the legal GDN/O/residual/RMS
   producer-consumer boundary fusions alongside the new MLP plane.

The first verdict for that package remains one real P1804 warmup plus one real
P1853 measurement through the OpenAI API and external EvalScope.  A positive
whole-request result earns repeatability and length/capability matrices; a
negative result gets one profile and is closed.

## Evidence

Direction result root:
`/home/rm01/q3x-r4-real-api-eval-20260803/candidate-r4-p1853-run1`

```text
provenance.txt SHA-256
  307c8adf6d7097ba3f9ec29cb016d74e45212b828633796b9d675dc0aeef3dce
server.log SHA-256
  37c4120d137cb53002545631f256805b11a709118021754b3ea1219e71be151e
benchmark_summary.json SHA-256
  05922b4cb78de0ae3d5c17aadda8d701ef823feebf0f02a2005c53ffd8e49df4
EvalScope stdout SHA-256
  7a3304439cf347b4fb8783b202afd301cbe61c15060c7a3fa05029acf27b379d
```

Profile result root:
`/home/rm01/q3x-r4-real-api-eval-20260803/profile-r4-p1853-run1`

```text
NSys report SHA-256
  7625b0886385cc899e5fc989acc69623d7be3d28d78407265b4a81e68ca916c8
kernel summary SHA-256
  09eb072747da47d79c032d12b3cb64f6403b5252d055d6690f22563f9ced4895
server.log SHA-256
  46228d9fb06f691e30830e983fe647adc2465d816e8070988558f68b5668d186
profile benchmark_summary.json SHA-256
  c903f975d7796904f33abed7d91095614b7399e9c5e92750be3fad75d73022e9
```
