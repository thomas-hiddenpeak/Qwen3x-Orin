# Qwen3.6 27B Phase 3 performance evidence

This document records the first kernel-level baseline and matched SM87
projection-backend comparison for the native Qwen3.6-27B-NVFP4 runner on
Jetson AGX Orin. These are diagnostic Phase 3 results, not serving-throughput
claims. The machine-readable records are
[`qwen36-27b-reference-nsys-baseline.json`](metadata/qwen36-27b-reference-nsys-baseline.json)
and
[`qwen36-27b-projection-backend-benchmark.json`](metadata/qwen36-27b-projection-backend-benchmark.json).
The subsequent startup diagnosis and authenticated-loader result are retained
in
[`qwen36-27b-afalg-loader-benchmark.json`](metadata/qwen36-27b-afalg-loader-benchmark.json).
The later parallel-shard and tokenizer/resident startup records are retained in
[`qwen36-27b-parallel-loader-benchmark.json`](metadata/qwen36-27b-parallel-loader-benchmark.json)
and
[`qwen36-27b-startup-overlap-benchmark.json`](metadata/qwen36-27b-startup-overlap-benchmark.json).
The next M=1 NVFP4 milestone is recorded in
[`qwen36-27b-nvfp4-packedx8-benchmark.json`](metadata/qwen36-27b-nvfp4-packedx8-benchmark.json),
and the subsequent FP8 milestone is recorded in
[`qwen36-27b-fp8-packedx4-benchmark.json`](metadata/qwen36-27b-fp8-packedx4-benchmark.json).
The first bounded multi-token prefill result is recorded in
[`qwen36-27b-c8-prefill-benchmark.json`](metadata/qwen36-27b-c8-prefill-benchmark.json).
The subsequent C8 kernel-optimization milestone is recorded in
[`qwen36-27b-c8-kernel-optimization-benchmark.json`](metadata/qwen36-27b-c8-kernel-optimization-benchmark.json).
The bounded C16 runtime and FP8/NVFP4 Tensor Core milestone is recorded in
[`qwen36-27b-c16-tensor-core-prefill-benchmark.json`](metadata/qwen36-27b-c16-tensor-core-prefill-benchmark.json).
The subsequent exact-shape FP8 K/V decode-pair diagnostic is recorded in
[`qwen36-27b-fp8-kv-pair-benchmark.json`](metadata/qwen36-27b-fp8-kv-pair-benchmark.json).
The aligned M1 NVFP4 down-projection dual-iteration diagnostic is recorded in
[`qwen36-27b-nvfp4-down-dual-benchmark.json`](metadata/qwen36-27b-nvfp4-down-dual-benchmark.json).
The aligned M1 NVFP4 gate/up adjacent-lane XOR-dual diagnostic is recorded in
[`qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json`](metadata/qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json).
The aligned M1 NVFP4 lm-head adjacent-lane XOR-dual diagnostic is recorded in
[`qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json`](metadata/qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json).
The subsequent down/lm-head data-reuse diagnostic is recorded in
[`qwen36-27b-nvfp4-data-reuse-benchmark.json`](metadata/qwen36-27b-nvfp4-data-reuse-benchmark.json).
The follow-up gate/up activation-reuse diagnostic is recorded in
[`qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json`](metadata/qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json).
The follow-up down activation-reuse diagnostic is recorded in
[`qwen36-27b-nvfp4-down-activation-staged-benchmark.json`](metadata/qwen36-27b-nvfp4-down-activation-staged-benchmark.json).
The subsequent eight-row lane-striped GDN diagnostic is recorded in
[`qwen36-27b-gdn-eight-row-benchmark.json`](metadata/qwen36-27b-gdn-eight-row-benchmark.json).
The subsequent exact FP8 M1 linear-attention QKV/Z fusion diagnostic is
recorded in
[`qwen36-27b-fp8-qkv-z-fusion-benchmark.json`](metadata/qwen36-27b-fp8-qkv-z-fusion-benchmark.json).
The subsequent exact NVFP4 M1 dense-MLP gate/up/SiLU fusion diagnostic is
recorded in
[`qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json`](metadata/qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json).
The follow-up post-attention residual/norm/gate/up/SiLU fusion diagnostic is
recorded in
[`qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json`](metadata/qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json).
The reduction-only warp-tail follow-up inside that fused kernel is recorded in
[`qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json`](metadata/qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json).
The subsequent exact FP8 M1 full-attention Q+K/V fusion diagnostic is recorded
in
[`qwen36-27b-fp8-q-kv-fusion-benchmark.json`](metadata/qwen36-27b-fp8-q-kv-fusion-benchmark.json).
The subsequent exact NVFP4 M1 down/residual/centered-RMSNorm cooperative fusion
diagnostic is recorded in
[`qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json`](metadata/qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json).
The subsequent canonical M1 GDN/plain-RMSNorm/SiLU-gate fusion diagnostic is
recorded in
[`qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json`](metadata/qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json).
The reduction-only warp-tail follow-up inside that fused GDN kernel is recorded
in
[`qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json`](metadata/qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json).

## Method

The profiled source was commit
`b6f8d805bd8409f191ad5d9872fe8b7a751b9dd9`, using the pinned model revision
`0893e1606ff3d5f97a441f405d5fc541a6bdf404`. Nsight Systems captured CUDA
activity only:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=true \\
  -o q3x-27b-reference-baseline \\
  qwen3x-orin generate MODEL_DIR \\
  --prompt '用一句话解释 CUDA 是什么。' --max-tokens 2
```

The run used `MAXN`, but clocks were not locked because `jetson_clocks --show`
requires root in this environment. Swap was enabled. The results therefore
select optimization work and define comparison fields; they are not a
release-grade absolute performance number.

The baseline used Nsight Systems 2026.1.3. Its checked-in JSON contains the
compiler, CUDA, CMake, kernel, and L4T versions. The raw `.nsys-rep` is not
checked in; its size and SHA-256 are retained, and its two checked-in summaries
can be regenerated with:

```bash
nsys stats --report cuda_gpu_kern_sum --format csv \
  q3x-27b-reference-baseline.nsys-rep
nsys stats --report cuda_gpu_kern_gb_sum --format csv \
  q3x-27b-reference-baseline.nsys-rep
```

The 19-token prompt plus one post-first-token decode step completed in
21,799.394 ms. Prompt prefill was 20,648.739 ms and the measured subsequent
token was 1,150.655 ms. Cold authenticated loading was 205,175.575 ms and is
reported separately from generation.

## Kernel evidence

The 20 runner steps launched 32,042 kernels. Nsight attributed the GPU kernel
time as follows:

| Kernel group | Instances | GPU time | Share |
| --- | ---: | ---: | ---: |
| NVFP4 reference GEMV | 3,842 | 16,348.908 ms | 75.2% |
| FP8 reference GEMV | 4,160 | 5,041.971 ms | 23.2% |
| Gated DeltaNet update | 960 | 186.203 ms | 0.9% |
| FP32 to BF16 conversion | 9,920 | 37.601 ms | 0.2% |
| BF16 reference GEMV | 1,920 | 31.774 ms | 0.1% |

The weight-only reference GEMVs therefore account for approximately 98.4% of
GPU kernel time. Optimizing GDN, RoPE, attention softmax, copies, or the BF16
epilogue first cannot materially improve end-to-end decode.

The two repeated NVFP4 MLP shapes dominate:

| Shape `[N,K]` | Calls per step | Mean kernel | Total profile share |
| --- | ---: | ---: | ---: |
| `[17408,5120]` | 128 | 4.262 ms | 50.2% |
| `[5120,17408]` | 64 | 4.154 ms | 24.5% |

Together they consume about 811.4 ms per runner step. The NVFP4 language head
`[248320,5120]` averaged 60.069 ms, but ran only on the two logits-producing
steps and represented 0.6% of this profile.

One full logits step streams approximately 17.602 GB of encoded weights. At
1.151 seconds per decode token, the reference path realizes only about
15.3 GB/s of logical encoded-weight bandwidth versus the device-reported
83.2 GB/s DDR ceiling. This number includes neither profiler overhead
correction nor a claim that every encoded byte maps to exactly one DRAM
transaction.

## Matched projection-backend comparison

Commit `fb09f4245763e73e5b50c34b09c300fedc4119ac` adds an explicit,
default-off `sm87` layer-projection backend and a benchmark harness that reuses
one loaded engine. Both backends ran from the same Release binary with the
same two prompts, one warmup round, two measured rounds, two generated tokens,
and a 64-position request arena:

```bash
qwen3x-orin benchmark MODEL_DIR \
  --prompt '用一句话解释 CUDA 是什么。' \
  --prompt 'Explain CUDA in one sentence.' \
  --max-tokens 2 --warmup 1 --iterations 2 \
  --max-sequence-length 64 \
  --projection-backend reference  # repeat with sm87
```

Every warmup and measured invocation replayed identical prompt IDs, generated
IDs, decoded text, stop reason, and step sequence. The four measured samples
were tightly clustered:

| Median metric | Reference | SM87 | Reduction | Speedup |
| --- | ---: | ---: | ---: | --- |
| Time to first token | 20,521.948 ms | 11,163.214 ms | 45.60% | 1.838x |
| Total two-token generation | 21,666.056 ms | 11,814.763 ms | 45.47% | 1.834x |
| Subsequent token | 1,144.108 ms | 651.554 ms | 43.05% | 1.756x |

Neither paired run reported a persistent device-free-memory drop. Cold load
remained separate and nearly equal at 204,926.476 ms for reference and
204,837.861 ms for SM87.

The production-shape CUDA-event segment uses five warmups and twenty measured
launches. Its optimized means, compared with the matching reference shape
means from the Nsight baseline, were:

| Shape `[N,K]` | Reference | SM87 | Speedup |
| --- | ---: | ---: | ---: |
| FP8 `[10240,5120]` | 1.824 ms | 1.600 ms | 1.140x |
| NVFP4 `[17408,5120]` | 4.262 ms | 1.774 ms | 2.403x |
| NVFP4 `[5120,17408]` | 4.154 ms | 1.818 ms | 2.286x |

## Post-SM87 profile and authenticated startup

A second Nsight run at commit
`e287e581638762b9c2a712d956c9d44797528de1` profiled the optimized backend with
the same 19-token prompt and two generated tokens. It launched 24,042 kernels;
optimized NVFP4 and FP8 layer projections still accounted for 58.3% and 38.1%
of GPU kernel time, respectively. This confirms that quantized projections
remain the decode target after the first SM87 kernel milestone.

The same trace exposed a separate, larger usability problem at startup. The
profile's H2D GPU memory operations occupied only 564.086 ms, and the two
`cudaMalloc` API calls occupied about 1.598 seconds, while authenticated resident
loading took 204,003.741 ms. Repeated warm-cache loads were similar.
The portable scalar SHA-256 pass, rather than storage or H2D bandwidth, was the
dominant load cost. On this Orin, Linux reports the generic `sha256` algorithm's
highest-priority provider as `sha256-ce` with a passed self-test.

Commit `06f16c924c47c3d876f42fba872b0172b676161b` therefore adds a default
`auto` hash backend. It prepares one Linux AF_ALG SHA-256 operation for every
shard before reading any checkpoint bytes. Setup failure may fall back to the
portable implementation only before loading begins; update or finalization
failure remains fail-closed. Every byte is still covered by the same three
pinned full-file digests, and no OpenSSL dependency or second file pass was
introduced.

The official loader fixture then authenticated all 21,921,697,184 source bytes,
copied 20,150,569,096 bytes, and completed in 21.4772 seconds with
`sha256_backend=linux_af_alg`. A full two-token CLI run reported 21,484.838 ms
for resident loading and 22,637.825 ms for total engine loading, while retaining
the exact 19 prompt IDs and generated `77517,220` / `"CUDA "` result.

Against the previous unprofiled SM87 artifact, the diagnostic historical
comparison is:

| Load metric | Portable baseline | AF_ALG run | Reduction | Speedup |
| --- | ---: | ---: | ---: | ---: |
| Resident load | 203,676.733 ms | 21,484.838 ms | 89.45% | 9.480x |
| Total engine load | 204,837.861 ms | 22,637.825 ms | 88.95% | 9.048x |
| Complete fixed-oracle CTest | 234.35 s | 52.22 s | 77.72% | 4.488x |

The baseline load fields came from the prior same-device benchmark command,
whereas the accelerated fields came from `generate`; this is therefore a
diagnostic historical comparison, not a randomized paired release claim. The
full-model row is the same `reference_engine_e2e` gate at the two commits.

## Packed-x8 NVFP4 decode

Commit `177c560c88710e93a452f58e3a9ae70d2da41c7f` vectorizes the canonical
NVFP4 path without repacking the checkpoint. For K divisible by 256 and a
4-byte-aligned packed-weight pointer, each lane loads one `uint32`, decodes
eight E2M1 values, shares one 16-value scale with its adjacent lane, and uses
four FP32 accumulators. Other K values and unaligned pointers retain the
previous scalar SM87 kernel.

The same-binary, mirrored-order CUDA-event gate used identical buffers for the
scalar and vector paths:

| Shape `[N,K]` | Scalar | Packed x8 | Speedup | Gate |
| --- | ---: | ---: | ---: | ---: |
| `[17408,5120]` | 1.77453 ms | 1.00099 ms | 1.773x | pass |
| `[5120,17408]` | 1.80440 ms | 1.00064 ms | 1.803x | pass |

Both exceed the required 1.15x speedup. The generated vector kernel uses 39
registers per thread, no stack/local/shared memory, and no spills; its packed
weight access is one 32-bit global load per lane and loop iteration.

Repeating the prior two-prompt benchmark command produced exact replay for all
four measured samples and no persistent device-memory drop above the 64 MiB
gate:

| Median metric | Reference | First SM87 | Packed x8 | Reduction vs first SM87 | Speedup vs reference |
| --- | ---: | ---: | ---: | ---: | ---: |
| Time to first token | 20,521.948 ms | 11,163.214 ms | 8,279.876 ms | 25.83% | 2.479x |
| Total two-token generation | 21,666.056 ms | 11,814.763 ms | 8,779.179 ms | 25.69% | 2.468x |
| Subsequent token | 1,144.108 ms | 651.554 ms | 499.086 ms | 23.40% | 2.292x |

The reference and first-SM87 columns come from the earlier `fb09f42` artifact;
the packed-x8 column comes from `177c560`. This end-to-end table is a
cross-commit historical comparison, not a same-binary paired trial. Only the
preceding scalar/vector microbenchmark is same-binary.

The last row is approximately 2.004 subsequent tokens per second, a material
interactive-decode improvement. It is still a batch-one, two-short-prompt result
with unlocked clocks, not a serving-throughput claim.

The post-change Nsight trace attributes 51.2% of GPU kernel time to FP8 layer
projections and 43.8% to packed-x8 NVFP4 layer projections. The two NVFP4
production shapes averaged 1.0035 ms and 1.0017 ms in the full-model trace,
down from approximately 1.782 ms and 1.818 ms. This result selected FP8 as the
next M=1 decode target; multi-token prefill remained a separate small-M
milestone.

## Packed-x4 FP8 decode

Commit `ad22fdda2925bc12d60c319296646e4449dc11a3` vectorizes the canonical
FP8 E4M3FN path without repacking the checkpoint. When K is divisible by
1,024, the weight pointer is 4-byte aligned, and the BF16 activation pointer
is 8-byte aligned, each lane loads four encoded weights with one 32-bit load
and four activations with one 64-bit load. A branchless decoder preserves the
exact finite E4M3FN values, signed zero, and the signed canonical quiet-NaN
class. Other shapes or alignments retain the scalar SM87 kernel.

The same-binary CUDA-event gate used identical buffers and mirrored
scalar/vector measurement order. All four shapes with at least 5,120 rows
exceed their required 1.15x gate; the smaller shape also improves instead of
using its allowed 2% regression budget:

| Shape `[N,K]` and payload | Scalar | Packed x4 | Speedup | Encoded-weight bandwidth |
| --- | ---: | ---: | ---: | ---: |
| `[10240,5120]`, uniform finite | 1.61403 ms | 0.795254 ms | 2.030x | 65.93 GB/s |
| `[5120,6144]`, uniform finite | 0.969309 ms | 0.469730 ms | 2.064x | 66.97 GB/s |
| `[6144,5120]`, uniform finite | 0.966847 ms | 0.479402 ms | 2.017x | 65.62 GB/s |
| `[12288,5120]`, uniform finite | 1.91765 ms | 0.951666 ms | 2.015x | 66.11 GB/s |
| `[1024,5120]`, uniform finite | 0.179198 ms | 0.086185 ms | 2.079x | 60.83 GB/s |
| `[10240,5120]`, mixed finite codes | 1.87785 ms | 0.794232 ms | 2.364x | 66.01 GB/s |

The generated packed-x4 kernel uses 36 registers per thread and 32 bytes of
shared memory per block, with no stack, local memory, or spills. SASS retains
the intended 32-bit encoded-weight load and 64-bit activation load. The scalar
baseline uses 17 registers per thread.

Repeating the same two-prompt benchmark command reproduced every prompt ID,
generated ID, decoded string, stop reason, and runner-step count. All four
measured samples passed the 64 MiB persistent-memory gate:

| Median metric | Reference | First SM87 | Packed x8 | Packed x4 | Reduction vs packed x8 | Speedup vs reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Time to first token | 20,521.948 ms | 11,163.214 ms | 8,279.876 ms | 6,107.361 ms | 26.24% | 3.360x |
| Total two-token generation | 21,666.056 ms | 11,814.763 ms | 8,779.179 ms | 6,492.535 ms | 26.05% | 3.337x |
| Subsequent token | 1,144.108 ms | 651.554 ms | 499.086 ms | 385.181 ms | 22.82% | 2.970x |

The four packed-x4 samples ranged from 6,106.646 to 6,107.479 ms for TTFT,
6,491.486 to 6,493.014 ms total, and 384.827 to 385.535 ms for the subsequent
token. The last median is approximately 2.596 subsequent tokens per second.
AF_ALG resident loading remained separate at 21,579.657 ms. Device free memory
ended above its starting value, so no persistent drop was detected.

Only the preceding scalar/vector microbenchmark is a same-binary A/B. The
end-to-end reference, first-SM87, packed-x8, and packed-x4 columns are retained
from their respective commits and form a cross-commit historical comparison,
not a randomized same-binary release trial. The batch-one, two-short-prompt
run also used unlocked clocks with swap enabled.

The post-change Nsight trace launched the same 24,042 kernels over 20 runner
steps. Packed-x4 FP8 projections consumed 2,226.552 ms across 4,160 instances,
50.52% less GPU time than the preceding profile, while packed-x8 NVFP4 consumed
3,850.493 ms across 3,840 instances. Their new profile shares are 34.2% and
59.1%, respectively; Gated DeltaNet and the reference NVFP4 language head
account for another 2.9% and 1.8%. Total kernel GPU time was 6,512.443 ms,
25.89% below the preceding packed-x8 profile. The profiled generation reported
6,175.789 ms TTFT and 386.433 ms for its subsequent token. The raw report is
not checked in; its 1,769,234-byte size and SHA-256
`e5854b3c163153204c8b871cb8e1cd1614a09622b441e12e1f7bdef1d234df32`
are retained in the machine-readable record.

## C8 chunked prompt prefill

Commit `44aa676fc9f77d11d0e48f19d9a8caf80561204f` adds the SM87 small-M
projection substrate for `M=1..8`. The following runtime integration groups
only the prompt prefix into tiles of at most eight tokens. Quantized layer
projections share each streamed weight row across the tile; causal Conv/GDN
updates and per-token attention lengths retain token order. The final prompt
token still runs as an M=1 logits step, and every later decode step remains
M=1. Chunk size 1 is the compatibility default.

The comparison reused the packed-x4 two-prompt/two-output-token Release
benchmark shape. The two commands differed only in the final chunk-size
option:

```bash
qwen3x-orin benchmark MODEL_DIR \
  --prompt '用一句话解释 CUDA 是什么。' \
  --prompt 'Explain CUDA in one sentence.' \
  --max-tokens 2 --warmup 1 --iterations 2 \
  --max-sequence-length 64 --projection-backend sm87 \
  --prefill-chunk-size 1

qwen3x-orin benchmark MODEL_DIR \
  --prompt '用一句话解释 CUDA 是什么。' \
  --prompt 'Explain CUDA in one sentence.' \
  --max-tokens 2 --warmup 1 --iterations 2 \
  --max-sequence-length 64 --projection-backend sm87 \
  --prefill-chunk-size 8
```

The benchmark's strict replay gate passed. Aggregate medians were:

| Median metric | C1 | C8 | Reduction | C1/C8 speedup |
| --- | ---: | ---: | ---: | ---: |
| Time to first token | 6,107.420 ms | 2,005.784 ms | 67.16% | 3.045x |
| Total two-token generation | 6,492.908 ms | 2,389.125 ms | 63.20% | 2.718x |
| Subsequent token | 385.467 ms | 383.320 ms | 0.56% | 1.006x |

The nearly flat subsequent-token row is the expected guardrail: chunking
changes prompt-prefix work, not decode dispatch. At a 64-position request
capacity, reserving C8 activation workspace increased the request arena from
83,821,056 to 85,011,968 bytes, an exact 1,190,912-byte (about 1.136 MiB)
increment. Persistent Conv/GDN/KV state and the model arena are unchanged.

The independent full-model C8 gate used the pinned 19-token fixture and
matched all 19 prompt IDs, all 26 generated IDs, exact decoded text,
`<|im_end|>`, and all 44 transcript steps. For that prompt, the 18-token prefix
is scheduled as `8+8+2`; the nineteenth prompt token remains the scalar logits
step. Trace capture intentionally forces effective C1 execution because the
trace ABI is defined at every token boundary.

This remains a diagnostic batch-one result from two short prompts, without a
new locked-clock environment capture or any large-prefill/concurrent-request
claim. The checked-in metadata contains the reported aggregates and explicit
limitations; raw per-sample output was not retained in that artifact.

## Post-C8 kernel optimization

Commits `1ca41d3` through `5fe0ae0` retain the same C8 execution policy while
removing overhead inside it. The first sequence through `4f23fdb` batches
prefix norms and the logits transfer, tiles the GDN prefix recurrence, caches
FP8 and NVFP4 decode codebooks per block, makes E2M1 decode branchless,
vectorizes the NVFP4 M=1 activation load, coalesces GDN state rows by warp, and
pairs adjacent NVFP4 M=8 output rows. The follow-on sequence `ce0d289`,
`3c2f481`, `4f00796`, `19d8208`, `3e61ae1`, `35acf9f`, `eea6567`,
`5e3b62c`, `4ada2c1`, and `5fe0ae0` adds a small-shape guard, caches NVFP4
block-scale decodes for M=8/M=1/M=2, pairs FP8 M=8 and GDN state rows, reuses
FP8 codebooks across rows for M=1/M=2, and specializes the two production
NVFP4 M=8 shapes plus the five production FP8 M=8 shapes.

The final dispatch remains deliberately narrow. Exact `[17408,5120]` and
`[5120,17408]` NVFP4 M=8 projections use compile-time shape specializations;
exact `[10240,5120]`, `[5120,6144]`, `[6144,5120]`, `[12288,5120]`, and
`[1024,5120]` FP8 M=8 projections do the same. Other aligned M=8 shapes retain
their generic row-pair kernels, and M=1 plus unaligned/fallback shapes retain
their separate dispatches. The C1 compatibility default and the default
`reference` projection backend are unchanged.

Repeating the same two-prompt, two-output-token C8 benchmark shape at
`5fe0ae0` produced the following medians. The original C1 column is retained
only as context; the optimization comparison is the matched-shape initial C8
versus optimized C8 pair.

| Median metric | Original C1 context | Initial C8 (`883a962`) | Optimized C8 (`5fe0ae0`) | Reduction vs initial C8 | Speedup vs initial C8 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Time to first token | 6,107.420 ms | 2,005.784 ms | 1,020.755 ms | 49.11% | 1.965x |
| Total two-token generation | 6,492.908 ms | 2,389.125 ms | 1,205.989 ms | 49.52% | 1.981x |
| Subsequent token | 385.467 ms | 383.320 ms | 185.108 ms | 51.71% | 2.071x |

Relative to the original C1 context, the optimized C8 run reduces TTFT by
83.29% (5.983x), total generation by 81.43% (5.384x), and subsequent-token
latency by 51.98% (2.082x). Its 85,011,968-byte request arena is unchanged
from the first C8 milestone and remains 1,190,912 bytes above the original C1
context.

The same-binary CUDA-event gates isolate several of the larger kernel changes:

| NVFP4 M=8 shape `[N,K]` | Single-row kernel | Row-pair kernel | Speedup |
| --- | ---: | ---: | ---: |
| `[17408,5120]` | 1.36294 ms | 1.16454 ms | 1.17037x |
| `[5120,17408]` | 1.91616 ms | 1.17258 ms | 1.63414x |

The final fixed-shape gate compares the generic scale-codebook row-pair kernel
with the specialization in mirrored B/C/C/B order. Both checkpoint-like and
same-bank-stress scale distributions match every production BF16 output
bit-for-bit and clear the required 1.03x gate:

| NVFP4 M=8 shape `[N,K]` | Generic row pair | Fixed shape | Aggregate speedup | Distribution speedup range |
| --- | ---: | ---: | ---: | ---: |
| `[17408,5120]` | 1.04221 ms | 0.906097 ms | 1.15022x | 1.14536x–1.15513x |
| `[5120,17408]` | 1.04346 ms | 0.882412 ms | 1.18251x | 1.17688x–1.18823x |

Weighting the shapes by their 256:128 calls in the profiled C8 run gives
`1.16079x`. The specialization retains 64 registers per thread, 1,088 bytes
of shared memory per block, and zero stack/local memory. Its normalized SASS
instruction count is 1,144 versus 1,272 for the generic kernel; the 128 FFMA
operations and numerical reduction order remain unchanged.

The same-binary FP8 M=8 gate compares each generic row-pair kernel with its
exact production-shape specialization. Every checked BF16 output is bitwise
equal:

| FP8 M=8 shape `[N,K]` | Speedup | Calls in final profile |
| --- | ---: | ---: |
| `[10240,5120]` | 1.13073x | 96 |
| `[5120,6144]` | 1.14011x | 128 |
| `[6144,5120]` | 1.13936x | 96 |
| `[12288,5120]` | 1.12711x | 32 |
| `[1024,5120]` | 1.22060x | 64 |

Weighting by those 416 profile calls gives `1.13694x`. The fixed kernels use
48 registers per thread and 1,536 bytes of shared memory per block, with zero
stack, local memory, or spills. Exact-shape compilation reduces normalized
SASS from 1,864 to 784 instructions.

| GDN token tile | Prior kernel | Warp-coalesced rows | Speedup |
| --- | ---: | ---: | ---: |
| M=1 | 0.197281 ms | 0.0965267 ms | 2.04379x |
| M=8 | 1.49476 ms | 0.729171 ms | 2.04995x |

The independent C8 full-model oracle at `5fe0ae0` again matched all 19 prompt
IDs, all 26 generated IDs, exact decoded text, `<|im_end|>`, and all 44 runner
steps. These end-to-end numbers remain a cross-commit historical diagnostic,
not a randomized same-binary trial. They cover batch one, two short prompts,
two generated tokens, and unlocked clocks; they do not establish large-prefill,
continuous-batching, or concurrent-request throughput.

The final CUDA-only Nsight Systems 2026.1.3 trace provides a separate hotspot
view for one 19-token prompt and two generated tokens. It reported 1,032.914 ms
prompt prefill, 186.659 ms after the first token, and 1,219.573 ms total under
profiling. Its 10,107 CUDA kernel instances consumed 1,192.638976 ms. The two
fixed NVFP4 M=8 specializations total 343.824640 ms across 384 calls (28.8%),
followed by NVFP4 M=1 at 227.911488 ms (19.1%). The five fixed FP8 M=8
specializations total 196.364096 ms across 416 calls (16.5%); NVFP4 M=2 uses
116.766080 ms (9.8%) and FP8 M=1 uses 113.207808 ms (9.5%). Relative to the
immediately prior `4ada2c1` profile, aggregate kernel time fell 1.71% and FP8
M=8 time fell 9.60%. Across the complete follow-on sequence from `4f23fdb` to
`5fe0ae0`, aggregate kernel time fell 17.44% and NVFP4 M=8 time fell 23.31%;
these profile comparisons do not assign every aggregate change to a single
commit. The raw 871,724-byte report is not checked in; SHA-256
`da7e5c82cdf38c0471e60ec317f095a7f83924103cf8b9c19c88467b067145aa`
and the complete hotspot table are retained in the machine-readable record.

## C16 Tensor Core prompt prefill

Commits `c90f37e` and `dda4e3a` raise the causal GDN/Conv and request/runtime
tile contracts to C16. For the 19-token oracle prompt, the 18-token prefix is
now scheduled as `16+2`; the final prompt/logits step and all decode work remain
M1. Quantized M9..M15 projections are split into M8 plus the remaining M1..M7
rows, preserving the existing small-M kernels and fallbacks.

Commits `e7283d6` and `33948e3` add fixed-M16 FP8 and NVFP4 kernels. They decode
the canonical checkpoint layouts into BF16 shared-memory tiles and use Ampere
BF16 Tensor Core MMA with FP32 accumulation; no offline repack or native
Blackwell NVFP4 instruction is involved. FP8 selects the path for
`[10240,5120]`, `[5120,6144]`, `[6144,5120]`, and `[12288,5120]` but retains
two M8 launches for the measured `[1024,5120]` regression and every other
shape/alignment. NVFP4 selects `[17408,5120]` and `[5120,17408]`; its other
valid cases likewise retain two M8 launches. Whole-C16 spans are validated
before the first launch.

The final same-binary CUDA-event gates compare one M16 Tensor Core launch with
two production M8 launches. Weighting by the production call mix gives
2.41756x for the selected FP8 shapes and 1.56406x for the two NVFP4 shapes.
Each gate retains deterministic replay, per-shape BF16 mismatch and
absolute/relative-error checks, exhaustive encoded-value/NaN coverage where
applicable, and safe alignment/shape fallbacks.
The generated cubin contains `HMMA.16816.F32.BF16`. The FP8 kernel uses 43
registers and 21,248 bytes of shared memory per block; NVFP4 uses 39 registers
and 28,928 bytes. Both report zero stack/local memory and zero spills.

The end-to-end comparison used four separate same-binary processes in strict
mirrored order: C8 round 1, C16 round 1, C16 round 2, then C8 round 2. Each
process ran two prompts, one warmup, and two measured iterations, producing
four samples; the two retained logs per policy therefore contribute eight
samples. Every process used this template with `C` set by that order:

```bash
qwen3x-orin benchmark MODEL_DIR \
  --prompt '用一句话解释 CUDA 是什么。' \
  --prompt 'Explain CUDA in one sentence.' \
  --max-tokens 2 --warmup 1 --iterations 2 \
  --max-sequence-length 64 --projection-backend sm87 \
  --prefill-chunk-size C
```

| Median metric | C8 | C16 | C16 reduction vs C8 |
| --- | ---: | ---: | ---: |
| Time to first token | 1,021.088 ms | 761.037 ms | 25.468% |
| Total two-token generation | 1,206.170 ms | 946.217 ms | 21.552% |
| Subsequent token | 185.084 ms | 185.186 ms | effectively flat |

Strict replay passed. The independent C16 full-model gate also retained all
19 prompt IDs, 26 output IDs, exact decoded text, `<|im_end|>`, and 44 runner
steps. At a 64-position capacity, the request arena grows from 85,011,968 bytes
at C8 to 86,373,376 bytes at C16, an exact 1,361,408-byte workspace increment;
persistent Conv/GDN/KV state and resident weights do not change.

A final CUDA-only Nsight Systems diagnostic attributes 929.615 ms to 9,210
kernel instances at C16. The historical optimized-C8 trace above recorded
1,192.639 ms across 10,107 instances. This comparison is useful hotspot
context, but it is cross-commit, used unlocked clocks, and does not assign the
entire difference to Tensor Core projection kernels. Neither profile
establishes large-prefill, continuous-batching, concurrent-request, or serving
throughput.

## FP8 K/V paired M1 decode

The full-attention K and V projections are both FP8 `[1024,5120]` matrices
that consume the same activation. The SM87 M1 pair path replaces their two
independent launches with one 256-thread cross-matrix row-quad launch. Each CTA
computes two K and two V rows, sharing activation decode and codebook setup
while retaining the individual production projection's BF16 bits. The selected
128-block persistent cap uses 63 registers, 1,152 bytes of shared memory, and
zero local memory per thread, with four active blocks per SM. The path requires
aligned exact-shape FP8 operands; eligible unaligned calls and every other valid
shape/token count retain the ordered independent-projection fallback.

The optional same-binary synthetic gate runs three mirrored timing rounds with
eight warmups and 80 measured launch pairs per timed pass:

| Distribution | Two production launches | Fused K/V launch | Speedup |
| --- | ---: | ---: | ---: |
| Checkpoint-like | 0.117589 ms | 0.0674598 ms | 1.74310x |
| Same-bank stress | 0.156288 ms | 0.0637426 ms | 2.45187x |

Both clear the required 1.10x gate. The default, non-performance test covers
all 254 finite E4M3FN codes in every packed byte position for both matrices,
isolates `0x7f`/`0xff` NaNs for class/sign checks, compares every candidate
output bit with the two production projections, checks output canaries, and
exercises alias rejection plus the unaligned dispatcher fallback.

A separate real-model max-26-token Nsight comparison used the exact command
template below for the historical base and candidate binaries:

```bash
nsys profile --trace=cuda --sample=none --cpuctxsw=none --stats=false \
  --force-overwrite=true -o REPORT \
  qwen3x-orin generate MODEL_DIR \
  --prompt '用一句话解释 CUDA 是什么。' --max-tokens 26 \
  --prefill-chunk-size 16 --projection-backend sm87
```

The 832 separate K/V kernels occupying 39.328192 ms became 416 fused kernels
occupying 31.316608 ms: 8.011584 ms saved, or an equivalent 1.255825407x
speedup. Total launches fell from 23,542 to 23,126. Aggregate CUDA kernel time
moved only from 3,620.029504 to 3,618.054560 ms because other hotspot families
increased by 6.036640 ms between the two reports. Their SHA-256 values are
`2982f1fef356e1281d274a56b5a38ee5626ad56564a70f15515c4fd7e1c7920e`
and `58f194d05118d1d0efe5b1fd95e7ea3960251339a67e2fb700a6389a1707e6c8`.
This cross-report comparison is hotspot evidence, not a same-binary trial.

The generation gate used four single-load processes in baseline/candidate/
candidate/baseline order. Each process loaded once, ran one warmup, then five
measured generations with this command shape:

```bash
qwen3x-orin benchmark MODEL_DIR \
  --prompt '用一句话解释 CUDA 是什么。' \
  --max-tokens 26 --warmup 1 --iterations 5 \
  --max-sequence-length 64 --projection-backend sm87 \
  --prefill-chunk-size 16
```

| Average of two process medians | Baseline | Fused K/V | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,628.073 ms | 3,620.950 ms | 7.123 ms (0.196330118%) |
| Time to first token | 576.0375 ms | 575.6235 ms | 0.414 ms (0.071870321%) |
| Subsequent token | 122.0775 ms | 121.8100 ms | 0.2675 ms (0.219123098%) |

The total-generation speedup is 1.001967163x. Every warmup and measured run
retained the exact 19 prompt IDs, 26 generated IDs, decoded text,
`<|im_end|>`, and 44 steps. These unlocked-clock batch-one results are a
diagnostic for one exact shape and prompt, not a release or serving-throughput
claim.

## NVFP4 down dual-iteration M1 decode

The long-K NVFP4 MLP down projection is the exact M1 `[5120,17408]` shape.
For aligned packed weights and BF16 activation, its SM87 row-quad kernel now
processes two adjacent packed-x8 K iterations per loop trip and broadcasts the
four rows' raw scale codes from the corresponding lane in each lane pair. The
kernel uses 64 registers per thread, 1,088 bytes of shared memory, zero local
memory, 256 threads, and four active blocks per SM. Gate/up uses the separately
gated specialization below. At this down-projection milestone, lm-head still
used the prior exact path; its subsequent specialization is also recorded
below. Near-miss shapes, unaligned operands, M2 through M16, and prefill retain
their previous routes.

The same-binary gate preserves the previous exact row-quad kernel as its
baseline and runs three baseline/candidate/candidate/baseline rounds. Two
independent production repeats cleared the required 1.025x gate for both
fixtures:

| Repeat and distribution | Preserved exact row quad | Dual iteration | Speedup |
| --- | ---: | ---: | ---: |
| 1, checkpoint-like | 0.340377 ms | 0.331082 ms | 1.02808x |
| 1, same-bank stress | 0.344832 ms | 0.332624 ms | 1.03670x |
| 2, checkpoint-like | 0.341337 ms | 0.332121 ms | 1.02775x |
| 2, same-bank stress | 0.345657 ms | 0.333736 ms | 1.03572x |

For both finite distributions, the preserved baseline, direct candidate, and
public production dispatch matched all 5,120 BF16 outputs bit-for-bit and kept
all output canaries intact. CUDA graph capture produced one kernel node. The
packed-weight `+1` and activation `+2` cases each matched the direct scalar
fallback at all 5,120 outputs. A separate fixture placed `0x7f` and `0xff`
scale NaNs in both dual sub-iterations: all four outputs retained the baseline
class/sign in direct and public paths, while the remaining 5,116 outputs stayed
finite and bitwise exact.

The matched max-26-token Nsight reports retain 1,664 down launches in both
profiles. Their time moved from 586.411200 to 568.161568 ms, saving 18.249632
ms for a 1.032120497x down-projection speedup. Aggregate CUDA kernel time moved
from 3,618.054560 to 3,600.652608 ms over the same 23,126 launches, saving
17.401952 ms (0.480975389%, 1.004832999x). The baseline report is 1,626,708
bytes with SHA-256
`58f194d05118d1d0efe5b1fd95e7ea3960251339a67e2fb700a6389a1707e6c8`;
the candidate is 1,624,642 bytes with SHA-256
`9c5cbe1f082864516726220f8f692bacce6e4a308cf5a1da862bee710824bd95`.
The reports are local evidence and are not checked in.

The mirrored single-load baseline/candidate/candidate/baseline benchmark gave:

| Average of two process medians | Baseline | Dual iteration | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,619.0065 ms | 3,603.5495 ms | 15.457 ms (0.427106169%) |
| Time to first token | 575.5240 ms | 574.9025 ms | 0.6215 ms (0.107988546%) |
| Subsequent token | 121.7455 ms | 121.1505 ms | 0.595 ms (0.488724429%) |

Every run retained the exact 19 prompt IDs, 26 generated IDs, decoded text,
`<|im_end|>`, and 44 runner steps. Release validation passed 52 tests with four
skips; ASAN/UBSAN with `detect_leaks=0`, excluding `package_consumer`, passed
51 tests with four skips. Clocks were unlocked. These measurements are
diagnostic evidence for one aligned M1 shape, not a release, randomized, or
serving-throughput claim.

## NVFP4 gate/up adjacent-lane XOR-dual M1 decode

The NVFP4 MLP gate and up projections share the exact M1 `[17408,5120]`
shape. For aligned packed weights and BF16 activation, each adjacent lane pair
now loads the four rows' raw scale codes for alternating packed-x8 phases. One
XOR shuffle reconstructs the ordered phase-zero and phase-one payloads for
both lanes, after which every lane retains the preserved phase, half, value,
FMA, reduction, scale, and BF16-encoding order. The kernel uses 64 registers
per thread, 1,088 bytes of shared memory, zero local memory, 256 threads, and
four active blocks per SM. Down projection retains its production dual-
iteration path. At this gate/up milestone, lm-head remained unchanged; the
subsequent exact-shape specialization is recorded in the next section. Every
other shape, alignment, token count, and prefill route remains unchanged.

The same-binary gate preserves the previous exact row-quad kernel as its
baseline and runs three baseline/candidate/candidate/baseline rounds. Two
independent production repeats cleared the required 1.01x gate for both
fixtures:

| Repeat and distribution | Preserved exact row quad | XOR dual | Speedup |
| --- | ---: | ---: | ---: |
| 1, checkpoint-like | 0.332866 ms | 0.317900 ms | 1.04708x |
| 1, same-bank stress | 0.342709 ms | 0.322024 ms | 1.06423x |
| 2, checkpoint-like | 0.333352 ms | 0.318523 ms | 1.04656x |
| 2, same-bank stress | 0.343205 ms | 0.322447 ms | 1.06438x |

For both finite distributions, the preserved baseline, direct candidate, and
public production dispatch matched all 17,408 BF16 outputs bit-for-bit and
kept all output canaries intact. CUDA graph capture produced one kernel node.
The packed-weight `+1` and activation `+2` cases each matched the direct scalar
fallback at all 17,408 outputs. A separate fixture placed `0x7f` and `0xff`
scale NaNs in both phases: all four outputs retained the baseline bits,
class, and sign in direct and public paths, while the remaining 17,404 outputs
stayed finite and bitwise exact.

The matched max-26-token Nsight reports retain 3,328 gate/up launches in both
profiles. Their time moved from 1,123.488832 to 1,073.110560 ms, saving
50.378272 ms for a 1.046946022x gate/up speedup. Aggregate CUDA kernel time
moved from 3,600.652608 to 3,551.045600 ms over the same 23,126 launches,
saving 49.607008 ms (1.377722691%, 1.013969691x). Non-gate/up kernels increased
by 0.771264 ms between these separate profiles. The baseline report is
1,624,642 bytes with SHA-256
`9c5cbe1f082864516726220f8f692bacce6e4a308cf5a1da862bee710824bd95`;
the candidate is 1,625,714 bytes with SHA-256
`652cf9d64e48caf487dee2d999f2740dcbb201155f6bc2482eff2aa748443dc1`.
The reports are local evidence and are not checked in.

The mirrored single-load baseline/candidate/candidate/baseline benchmark gave:

| Average of two process medians | Baseline | XOR dual | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,603.9395 ms | 3,553.0160 ms | 50.9235 ms (1.412995418%) |
| Time to first token | 574.8655 ms | 572.9845 ms | 1.8810 ms (0.327206973%) |
| Subsequent token | 121.1585 ms | 119.2120 ms | 1.9465 ms (1.606573208%) |

Every run retained the exact 19 prompt IDs, 26 generated IDs, decoded text,
`<|im_end|>`, and 44 runner steps. Release validation passed 52 tests with four
skips; ASAN/UBSAN with `detect_leaks=0`, excluding `package_consumer`, passed
51 tests with four skips. Clocks were unlocked. These measurements are
diagnostic evidence for one aligned M1 shape, not a release, randomized, or
serving-throughput claim.

## NVFP4 lm-head adjacent-lane XOR-dual M1 decode

The NVFP4 language head is the exact M1 `[248320,5120]` shape. It now uses a
separately gated instance of the K=5120 adjacent-lane XOR-dual kernel described
above. The larger row count changes only the grid-stride row-quad schedule: the
ordered phase, half, value, FMA, reduction, scale, and BF16-encoding sequence
remains bit-for-bit identical to the preserved exact row-quad kernel. Resources
remain 64 registers per thread, 1,088 bytes of shared memory, zero local memory,
256 threads, and four active blocks per SM. The down dual-iteration and gate/up
XOR-dual routes remain active; near-miss shapes, unaligned operands, M2 through
M16, and prefill retain their preceding routes.

The optional same-binary gate runs three mirrored
baseline/candidate/candidate/baseline timing rounds with 10 warmups and 24
measured launches per pass. The production full-shape run cleared its required
1.02x threshold for both distributions:

| Distribution | XOR-dual speedup | Required |
| --- | ---: | ---: |
| Checkpoint-like | 1.05649x | 1.02x |
| Same-bank stress | 1.07370x | 1.02x |

For both distributions, the preserved baseline, direct candidate, and public
production dispatch matched all 248,320 BF16 outputs bit-for-bit and kept all
output canaries intact. CUDA graph capture produced one kernel node. The
packed-weight `+1` and activation `+2` paths each matched the scalar fallback
at all 248,320 outputs. A separate fixture covered both `0x7f` and `0xff` scale
NaNs in both XOR phases: all four classified outputs and the remaining 248,316
finite outputs matched the preserved exact path in direct and public dispatch.
The gate/up regression run still measured 1.04808x checkpoint-like and
1.06432x same-bank speedups.

The matched max-26-token Nsight reports retain 26 lm-head launches in both
profiles. Their time moved from 125.008736 to 117.825600 ms, saving 7.183136 ms
for a 1.060964137x lm-head speedup. Aggregate CUDA kernel time moved from
3,551.045600 to 3,543.031168 ms over the same 23,126 launches, saving 8.014432
ms (0.225692174%, 1.002262027x). Non-lm-head kernels decreased by 0.831296 ms
between the separate reports, so the aggregate difference is not attributed
entirely to this specialization. The baseline report is 1,625,714 bytes with
SHA-256
`652cf9d64e48caf487dee2d999f2740dcbb201155f6bc2482eff2aa748443dc1`;
the candidate is 1,626,252 bytes with SHA-256
`5e55f08d6b53d65683699281695b2a33130c933a42ab5cc4b817a050f6b05983`.
The reports are local evidence and are not checked in.

The mirrored single-load baseline/candidate/candidate/baseline benchmark gave
the following average of two per-process medians:

| Average of two process medians | Baseline | XOR dual | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,554.7025 ms | 3,547.2890 ms | 7.4135 ms (0.208554724%) |
| Time to first token | 573.1200 ms | 572.7695 ms | 0.3505 ms (0.061156477%) |
| Subsequent token | 119.2710 ms | 119.0040 ms | 0.2670 ms (0.223859949%) |

Every warmup and measured generation retained the exact 19 prompt IDs, 26
generated IDs, decoded text, `<|im_end|>`, and 44 runner steps. Release
validation passed 52 tests with four skips; ASAN/UBSAN with `detect_leaks=0`,
excluding `package_consumer`, passed 51 tests with four skips. Independent
review found zero blockers. Clocks were unlocked. These measurements are
diagnostic evidence for one aligned M1 shape and short prompt, not a release,
randomized, or serving-throughput claim.

## NVFP4 down/lm-head data-reuse follow-up

The follow-up worktree based on `64da1a19f9a6e959c33b6e0721fac83a77458b38`
changes two exact aligned M1 routes without repacking the checkpoint. The
`[5120,17408]` down projection replaces its indexed-broadcast dual-iteration
kernel with the adjacent-lane XOR-dual specialization already proven for
gate/up. Even and odd lanes own alternating packed-x8 phases, exchange one
raw-scale payload, and then consume both phases in their original order. The
kernel retains 64 registers per thread, 1,088 bytes of shared memory, zero
local memory, 256 threads, and four active blocks per SM.

The `[248320,5120]` language head retains the same XOR-dual arithmetic but now
cooperatively copies its 10-KiB BF16 activation into shared memory once per CTA.
The global copy uses 8-byte loads and preserves the existing 8-byte activation
alignment contract; 16-byte shared loads feed every grid-stride row quad. With
the existing 1,088-byte codebooks, the kernel uses 11,328 bytes of static
shared memory, 64 registers per thread, zero local memory, 256 threads, and
four active blocks per SM. Gate/up remains on its prior XOR-dual instance.
Near-miss shapes, unaligned operands, M2 through M16, and prefill keep their
preceding routes.

Both same-binary gates use 10 warmups and three mirrored
baseline/candidate/candidate/baseline rounds with 24 measured launches per
pass:

| Route and distribution | Preserved baseline | Production candidate | Speedup |
| --- | ---: | ---: | ---: |
| Down, checkpoint-like | 0.331558 ms | 0.313948 ms | 1.05609x |
| Down, same-bank stress | 0.333291 ms | 0.315587 ms | 1.05610x |
| Lm-head, checkpoint-like | 4.48449 ms | 4.38675 ms | 1.02228x |
| Lm-head, same-bank stress | 4.53985 ms | 4.41034 ms | 1.02937x |

For each route and distribution, the preserved baseline, direct candidate,
and public production dispatch matched every BF16 output bit-for-bit and kept
both output canaries intact. Public and direct CUDA Graph captures each
contained one kernel node, and their `func` identities matched. Packed-weight
`+1` and activation `+2` cases matched the scalar fallback at all 5,120 down
and 248,320 lm-head outputs. Isolated `0x7f`/`0xff` block-scale NaNs retained
their bits, class, and sign; every remaining output stayed finite and exact.

The matched max-26-token profiles retained all 23,126 kernel launches. The
separately grouped target work changed as follows:

| Kernel group | Instances | Baseline | Candidate | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Down projection | 1,664 | 567.723200 ms | 537.979392 ms | 29.743808 ms (5.239139073%) |
| Lm-head | 26 | 117.825600 ms | 115.945952 ms | 1.879648 ms (1.595279803%) |
| Combined target work | 1,690 | 685.548800 ms | 653.925344 ms | 31.623456 ms (4.612867239%) |

Aggregate CUDA-kernel time moved from 3,543.031168 to 3,512.152960 ms,
saving 30.878208 ms (0.871519514%, 1.008791818x). Non-target kernels increased
by 0.745248 ms between the separate traces, so the combined target row is the
isolated attribution. The baseline report is 1,626,252 bytes with SHA-256
`5e55f08d6b53d65683699281695b2a33130c933a42ab5cc4b817a050f6b05983`;
the candidate is 1,623,010 bytes with SHA-256
`6dd711812a35b486a7e3452af25753af238f662954c8dd14eb90083bb30566d3`.
Both reports are local evidence and are not checked in.

An independent `git archive` rebuild of the base commit provided a standalone
baseline binary for a B-C-C-B single-load benchmark. Each process performed
one warmup and five measured generations:

| Average of two process medians | Base commit | Data reuse | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,548.701 ms | 3,514.877 ms | 33.824 ms (0.953137500%) |
| Time to first token | 572.837 ms | 571.639 ms | 1.198 ms (0.209134536%) |
| Subsequent token | 119.0265 ms | 117.7150 ms | 1.3115 ms (1.101855469%) |

Every warmup and measured run retained the exact 19 prompt IDs, 26 generated
IDs, decoded text, `<|im_end|>`, and 44 runner steps; the dedicated 27B C16
oracle also passed. Release validation passed 52 tests with four skips;
ASAN/UBSAN with `detect_leaks=0`, excluding `package_consumer`, passed 51 tests
with four skips. Independent review found no blockers, and its suggested down
Graph identity gate was added before commit. Clocks were unlocked. This remains
single-prompt diagnostic evidence, not a randomized, release, or serving-
throughput claim.

## NVFP4 gate/up activation-reuse follow-up

The follow-up worktree based on `e02905d64c053e3e67d726fc4a89bd4e331a8003`
moves the exact aligned M1 `[17408,5120]` gate/up projection from direct global
activation reads to the CTA activation-staged kernel already used by lm-head.
Each CTA cooperatively copies the 10-KiB BF16 activation with 8-byte global
loads and reuses it through 16-byte shared loads across every grid-stride row
quad. The arithmetic and checkpoint layout are unchanged. The production
kernel uses 64 registers per thread, 11,328 bytes of static shared memory, zero
local memory, 256 threads, and four active blocks per SM. Down remains on its
adjacent-lane XOR-dual instance; lm-head retains its staged instance. Near-miss
shapes, unaligned operands, M2 through M16, and prefill keep their prior routes.

The final same-binary gate used 10 warmups and three mirrored
baseline/candidate/candidate/baseline rounds with 24 launches per timed pass:

| Distribution | Direct-activation XOR baseline | CTA-staged candidate | Speedup |
| --- | ---: | ---: | ---: |
| Checkpoint-like | 0.318883 ms | 0.314369 ms | 1.01436x |
| Same-bank stress | 0.322921 ms | 0.316299 ms | 1.02093x |

Both exceed the per-distribution 1.005x gate. The preserved baseline, direct
candidate, and public production route matched all 17,408 BF16 outputs for both
distributions, retained finite outputs and both canaries, and preserved all
isolated `0x7f`/`0xff` NaN bits, classes, and signs. Public and direct staged
CUDA Graph captures each contained one kernel and matched by `func` identity.
Packed-weight `+1` and activation `+2` public calls matched the scalar fallback
at every output.

Default CTest now also capture-checks the production-sized route without
allocating or executing its weight fixture. It compares public and direct
staged `func`, grid, block, and dynamic-shared launch fields, then verifies that
both unaligned cases select the same scalar function and launch configuration
as the direct scalar test entry. This closes exact-selector and fallback
coverage even when the performance segment is not enabled.

The matched max-26-token Nsight profiles retained all 23,126 kernel launches:

| Kernel group | Instances | Base commit | Staged gate/up | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Gate/up projection | 3,328 | 1,074.533504 ms | 1,054.402944 ms | 20.130560 ms (1.873423204%) |
| All CUDA kernels | 23,126 | 3,512.152960 ms | 3,490.693120 ms | 21.459840 ms (0.611016668%) |

Non-target kernels decreased by 1.329280 ms between the separate profiles, so
only the gate/up row isolates this specialization. The baseline report is
1,623,010 bytes with SHA-256
`6dd711812a35b486a7e3452af25753af238f662954c8dd14eb90083bb30566d3`;
the candidate is 1,633,258 bytes with SHA-256
`85d0bbefe19b0d7193b4b5045d70ba07f6d934c58ce1673bea6cdc91a283916a`.
Both reports are local evidence and are not checked in.

An independent `git archive` rebuild of the base commit supplied the standalone
baseline for a B-C-C-B process comparison. Each process loaded one engine,
performed one warmup, and measured five generations:

| Average of two process medians | Base commit | Staged gate/up | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,515.5365 ms | 3,498.1105 ms | 17.4260 ms (0.495685%) |
| Time to first token | 571.5735 ms | 570.8440 ms | 0.7295 ms (0.127630%) |
| Subsequent token | 117.7605 ms | 117.0925 ms | 0.6680 ms (0.567253%) |

Every process retained the exact 19 prompt IDs, 26 generated IDs, decoded
text, `<|im_end|>`, and 44 runner steps; the dedicated 27B C16 oracle also
passed. Release validation passed 52 tests with four skips. ASan/UBSan with
`detect_leaks=0`, excluding `package_consumer`, passed 51 tests with four
skips; leak detection is disabled for the CUDA driver's known process-exit
allocations. Independent review found no blocker, and its default exact Graph
and scalar-function identity recommendations were added before commit. Clocks
were unlocked. These results remain single-prompt diagnostic evidence, not a
randomized, release, or serving-throughput claim.

## NVFP4 down activation-reuse follow-up

The follow-up worktree based on `b7418c1bb2d40f0bff408bf7ae4d4719a7585490`
moves exact aligned M1 `[5120,17408]` down from direct activation reads to CTA
activation staging. Each CTA copies the 34-KiB BF16 activation once with
8-byte global loads and reuses it across its grid-stride row quads. The direct
XOR kernel remains the same-binary baseline. Candidate resources are
64 registers, 35,904 static-shared bytes, zero local bytes, 256 threads, and
four active blocks per SM, versus 64/1,088/0/256/four for the baseline.
Near-miss shapes, packed-weight or activation misalignment, M2 through M16,
and prefill retain their preceding routes.

| Distribution | Direct XOR | CTA staged | Speedup |
| --- | ---: | ---: | ---: |
| Checkpoint-like | 0.314312 ms | 0.305565 ms | 1.02862x |
| Same-bank stress | 0.316203 ms | 0.306915 ms | 1.03026x |

Both clear the 1.005x gate with zero BF16 mismatches. Default capture-only
coverage verifies public/direct staged `func`, grid, block, and dynamic-shared
launch-field identity; direct XOR remains distinct, while packed-weight `+1`
and activation `+2` select the scalar function and launch configuration.

| Kernel group | Instances | Base commit | Staged down | Reduction |
| --- | ---: | ---: | ---: | ---: |
| Down projection | 1,664 | 536.467904 ms | 516.386464 ms | 20.081440 ms (3.743270%) |
| All CUDA kernels | 23,126 | 3,490.693120 ms | 3,471.580320 ms | 19.112800 ms (0.547536%) |

Non-target kernels increased by 0.968640 ms between these separate Nsight
reports, so only the down row isolates the change. The saved base-commit binary
and candidate were measured in B-C-C-B process order:

| Average of two process medians | Base commit | Staged down | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,498.3615 ms | 3,478.2960 ms | 20.0655 ms (0.573569%) |
| Time to first token | 570.8060 ms | 570.0515 ms | 0.7545 ms (0.132182%) |
| Subsequent token | 117.1025 ms | 116.3305 ms | 0.7720 ms (0.659252%) |

Every run retained the exact 19/26-token text, `<|im_end|>`, and 44-step
oracle. Release passed 52/52 tests; ASan/UBSan with `detect_leaks=0`, excluding
`package_consumer`, passed 51/51. Independent review found no blocker, raised
the initial 1.002 gate to 1.005, and corrected diagnostic wording. Clocks were
unlocked, so this remains diagnostic rather than a release or serving claim.

## GDN eight-row lane-striped follow-up

The worktree based on `52af2e06f14a0215f7e490cb37854a6cb3486851`
advances the fixed Qwen3.6 Gated DeltaNet state update from four to eight rows
per warp batch. Lanes 0 through 7 retain eight independent left-to-right
scalar FMA chains while all 32 lanes coalesce state loads and updates. This
halves each warp's outer state-row loop without changing per-row arithmetic,
BF16 persistence boundaries, token recurrence, validation, or alias behavior.
The public M1 and bounded C2 through C16 launches now use row8; row4 remains a
same-binary test predecessor.

Both kernels use 40 registers per thread and zero stack/local memory. Static
shared memory grows from 18,056 to 34,568 bytes per block. A direct
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` query reports six active row4
blocks versus four row8 blocks per SM; the 48-block row8 launch averages three
blocks per SM across the target Orin's 16 SMs.
The optional gate compares identical in-place and disjoint-state fixtures over
three baseline/candidate/candidate/baseline rounds with 10 warmups and 24
measured launches per pass:

| Token tile | Four-row predecessor | Eight-row production | Speedup |
| --- | ---: | ---: | ---: |
| M1 | 0.0476869 ms | 0.0335836 ms | 1.41995x |
| M2 | 0.0849027 ms | 0.0581104 ms | 1.46106x |
| M8 | 0.308216 ms | 0.202025 ms | 1.52564x |
| M16 | 0.605771 ms | 0.394754 ms | 1.53455x |

Every shape clears the 1.03x per-shape gate and matches the predecessor's
output plus persisted state bit-for-bit in both alias modes. Weighting M1/M2/
M8/M16 by the current max-26 profile calls `1248:48:0:48` gives
92.6656 versus 63.6498 ms, or 1.45587x, above the 1.20x aggregate gate.
Four recorded runs all clear that aggregate gate; the worst observed weighted
speedup is 1.32234x.

The matched max-26-token profiles retain all kernel launches:

| Kernel group | Instances | Four-row base | Eight-row | Reduction |
| --- | ---: | ---: | ---: | ---: |
| GDN update | 1,344 | 87.167840 ms | 60.100480 ms | 27.067360 ms (31.052003%) |
| All CUDA kernels | 23,126 | 3,471.580320 ms | 3,446.859392 ms | 24.720928 ms (0.712094%) |

Non-target kernels increased by 2.346432 ms between the separate reports, so
only the GDN row isolates this change. The baseline report is 1,633,875 bytes
with SHA-256
`534be81e6e3cc048c5c52862305fed97b0ada8caca9a288c68e56e401f0857fd`;
the row8 report is 1,633,214 bytes with SHA-256
`a5d908ddfba0fa441df70efe7048cb2b0f4138fbdb3bb46c48ea97db1bbe5c6a`.

An independent detached build of the base commit and the candidate were run in
B-C-C-B process order. Each process loaded once, warmed up once, and measured
five 26-token generations:

| Average of two process medians | Four-row base | Eight-row | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,479.3700 ms | 3,457.1855 ms | 22.1845 ms (0.637601%) |
| Time to first token | 570.1095 ms | 558.2255 ms | 11.8840 ms (2.084512%) |
| Subsequent token | 116.3880 ms | 115.8255 ms | 0.5625 ms (0.483297%) |

All warmup and measured generations retained the exact 19 prompt IDs, 26
generated IDs, decoded text, `<|im_end|>`, and 44 steps. The dedicated 27B C16
oracle passed. Release reported zero failures across 52 discovered tests, with
four skipped; ASan/UBSan reported zero failures across 51 discovered tests,
with four skipped, `detect_leaks=0`, and `package_consumer` excluded. The
sanitizers instrument host code; CUDA compute-sanitizer debugging features are
disabled on this device. Independent review found no blocker or medium issue.
Clocks were unlocked, and this single-prompt, batch-one result is diagnostic
rather than randomized, release, or serving-throughput evidence.

## FP8 linear-attention QKV/Z two-phase fusion

The worktree based on `e103c7f655b75a4b2c74e53b156902ea5261ffc5`
replaces the two exact aligned M1 linear-attention projections
`[10240,5120]` QKV then `[6144,5120]` Z with one two-phase launch. Its fixed
1,536-CTA grid preserves the production QKV grid/stride across 2,560 row
quads; the first 768 CTAs then preserve the production Z grid/stride across
1,536 row quads. Both phases reuse one decoded E4M3FN codebook. The row-quad
FMA, reduction, scale, and BF16-RNE order remains the same as the two preceding
production launches, so their outputs compare bit-for-bit.

The route is deliberately narrow. It requires the explicit SM87 backend,
M1, the ordered exact shapes, 4-byte-aligned weights, an 8-byte-aligned BF16
activation, and 2-byte-aligned outputs. M2 through M16, reversed or near-miss
shapes, unaligned operands, and other backends retain two ordered projections.
The scalar runner step and `prefill_prefix_tile(..., 1)` may select the fused
route; layer-major C2 through C16 prefix tiles remain two independent tile
projections.

The frozen promotion policy was stated before the five independent process
repeats. Actual checkpoint bytes carry a 1.02x production-value threshold; the
synthetic same-bank distribution is a 1.00x non-regression guard. The earlier
symmetric 1.01x threshold remains visible as an exploratory diagnostic but is
not the promotion criterion: it fails in four of the five stress repeats.
The fixture reads layer 0 QKV and Z at absolute shard offsets 3,485,125,152 and
3,537,553,952. Their payload SHA-256 values are
`66eeb8a7cfd3f577a4f7bafdb5b68f4f7ba3cb1aa9717801082791b2de696ed7`
and
`79a60d790f4ca146c05ea2efeff51964f825b1cdd92a83c8ee11b9fe9cfafdae`.

Each process uses 10 warmups, 80 logical QKV/Z pairs per timed pass, and five
baseline/candidate/candidate/baseline measurement rounds. All five clear the
frozen policy:

| Independent process | Actual checkpoint | Same-bank stress | Frozen gate | Exploratory symmetric 1.01x |
| ---: | ---: | ---: | --- | --- |
| 1 | 1.05824x | 1.00985x | pass | fail |
| 2 | 1.05542x | 1.00992x | pass | fail |
| 3 | 1.05868x | 1.01092x | pass | pass |
| 4 | 1.05501x | 1.00832x | pass | fail |
| 5 | 1.05575x | 1.00833x | pass | fail |

The actual-checkpoint minimum/median/maximum is
1.05501x/1.05575x/1.05868x. The stress minimum/median/maximum is
1.00832x/1.00985x/1.01092x. The fused kernel uses 64 registers per thread,
1,152 static-shared bytes, zero local bytes, 256 threads, and four active
blocks per SM. The same-binary correctness gate covers all 254 finite E4M3FN
codes in all four packed byte positions for both matrices, isolated signed NaN
classifications, bitwise output equality, deterministic replay, output
canaries, input preservation, and null/alignment/shape/scale/cap/alias
contracts.

Matched max-26-token Nsight profiles use an independently detached base build
and the candidate with the same command and workload. They are not a
same-binary trial:

| Profile group | Base | Fused QKV/Z | Reduction |
| --- | ---: | ---: | ---: |
| QKV/Z kernel instances | 2,496 | 1,248 | 1,248 (50.000000%) |
| QKV/Z kernel time | 634.147712 ms | 615.753920 ms | 18.393792 ms (2.900553%) |
| All CUDA kernel instances | 23,126 | 21,878 | 1,248 (5.396523%) |
| All CUDA kernel time | 3,446.077312 ms | 3,427.491392 ms | 18.585920 ms (0.539336%) |
| Profiled generation | 3,486.520 ms | 3,465.799 ms | 20.721 ms (0.594318%) |

Non-target kernels decreased by 0.192128 ms between the separate profiles, so
only the QKV/Z row isolates the specialization. The baseline report is
1,633,525 bytes with SHA-256
`7b9a6c5bffacd524d22214a8efeb7807f5894d6aed156b8d82b4db4effdd1143`;
the candidate is 1,550,518 bytes with SHA-256
`0a5ce00f21ffa6f14a370be6a6651f34b6368ce724b75b9ff768e948750c88af`.
Both reports are local evidence and are not checked in.

A detached-base B-C-C-B process comparison loaded one engine per process,
warmed up once, and measured five generations:

| Process order | Total generation median | TTFT median | Subsequent-token median |
| --- | ---: | ---: | ---: |
| Base 1 | 3,453.668 ms | 558.294 ms | 115.808 ms |
| Candidate 1 | 3,435.461 ms | 557.490 ms | 115.082 ms |
| Candidate 2 | 3,433.312 ms | 557.280 ms | 115.034 ms |
| Base 2 | 3,452.026 ms | 558.089 ms | 115.764 ms |

| Average of two process medians | Two-launch base | Fused QKV/Z | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,452.8470 ms | 3,434.3865 ms | 18.4605 ms (0.534646%) |
| Time to first token | 558.1915 ms | 557.3850 ms | 0.8065 ms (0.144484%) |
| Subsequent token | 115.7860 ms | 115.0580 ms | 0.7280 ms (0.628746%) |

Every warmup and measured generation retained the exact 19 prompt IDs, 26
generated IDs, decoded text, `<|im_end|>`, and 44 runner steps. Dedicated C1,
C8, and C16 27B oracle runs all passed. Release reported zero failures across
52 discovered tests, with four skipped; ASan/UBSan with `detect_leaks=0` and
`package_consumer` excluded reported zero failures across 51 discovered tests,
with four skipped. Clocks were unlocked. The microbenchmark, separate Nsight
profiles, and two-process-per-policy B-C-C-B comparison remain batch-one,
single-prompt diagnostic evidence rather than randomized, release,
concurrent-request, or serving-throughput claims.

## NVFP4 dense-MLP gate/up/SiLU fusion

The worktree based on `1dc309729da6df248e7eee7254db4c9a0350a41c`
replaces the exact aligned M1 NVFP4 gate projection, up projection, and SiLU
multiply with one launch. A 64-CTA, 256-thread kernel stages the BF16
activation and E2M1/E4M3FN decode codebooks once. Each warp computes four gate
rows and four up rows in rolled phases, preserving the existing projection
FMA, reduction, scale, and BF16-RNE order. After both independent BF16 stores,
the CTA synchronizes and redistributes its rows across all threads for the
SiLU multiply. The gate buffer is overwritten with the final BF16 result; the
up buffer retains its independently rounded projection.

The route requires the explicit SM87 backend, ordered valid NVFP4 gate/up
weights with exact `[17408,5120]` shapes, 4-byte-aligned packed weights, an
8-byte-aligned activation, and 2-byte-aligned disjoint outputs. Near-miss
shapes, unaligned operands, other dtypes/backends, and C2 through C16 retain
the previous ordered projections followed by the reference SiLU kernel.

The frozen same-binary gate reads layer-0 gate/up packed weights, block scales,
and scale2 values directly from the pinned shard. The four payload SHA-256
values are pinned in the machine-readable record. Actual checkpoint bytes
must reach 1.02x; a deterministic synthetic same-bank fixture must not regress
below 1.00x. Each process uses 10 warmups, 64 logical chains per timed pass,
and five baseline/candidate/candidate/baseline rounds:

| Independent process | Actual checkpoint | Same-bank stress | Frozen gate | Exploratory symmetric 1.01x |
| ---: | ---: | ---: | --- | --- |
| 1 | 1.02604x | 1.02555x | pass | pass |
| 2 | 1.02704x | 1.02733x | pass | pass |
| 3 | 1.02800x | 1.02764x | pass | pass |
| 4 | 1.02752x | 1.02738x | pass | pass |
| 5 | 1.02593x | 1.02602x | pass | pass |

The actual-checkpoint minimum/median/maximum is
1.02593x/1.02704x/1.02800x; the stress minimum/median/maximum is
1.02555x/1.02733x/1.02764x. The fused kernel uses 64 registers per thread,
11,328 static-shared bytes, zero local bytes, 256 threads, and four active
blocks per SM. Correctness covers raw pair projections, the fused final and up
outputs, signed NaN class/sign, deterministic replay, guards, preservation of
both matrices/scales/activation, public/test kernel identity, and
fail-before-enqueue validation for shape, alignment, scale, and aliases.

Matched max-26-token Nsight profiles compare an independently rebuilt detached
base with the candidate:

| Profile group | Three-launch base | Fused gate/up/SiLU | Reduction |
| --- | ---: | ---: | ---: |
| Target-chain kernel instances | 4,992 | 1,664 | 3,328 (66.666667%) |
| Target-chain kernel time | 1,063.999712 ms | 1,037.277440 ms | 26.722272 ms (2.511492%) |
| All CUDA kernel instances | 21,878 | 18,550 | 3,328 (15.211628%) |
| All CUDA kernel time | 3,426.051552 ms | 3,402.646528 ms | 23.405024 ms (0.683149%) |
| Profiled generation | 3,467.049 ms | 3,439.356 ms | 27.693 ms (0.798748%) |

The detached-base report is 1,551,018 bytes with SHA-256
`5b4ce9371d756daaee42996fdde2015a8fb2c6b4fc447eb90ddf727fa4f9b5ea`;
the candidate is 1,346,486 bytes with SHA-256
`d2dbe90a1c093d118d81cf82262ddaeea03f53d021c34b043a4643753897c46b`.
Non-target GPU kernel time increased by 3.317248 ms between the separate,
unlocked-clock processes, so the isolated target-chain row is the reliable
kernel attribution.

The independent B-C-C-B process comparison loaded once per process, warmed up
once, and measured five generations:

| Process order | Total generation median | TTFT median | Subsequent-token median |
| --- | ---: | ---: | ---: |
| Base 1 | 3,434.372 ms | 557.498 ms | 115.081 ms |
| Candidate 1 | 3,404.371 ms | 556.286 ms | 113.925 ms |
| Candidate 2 | 3,404.997 ms | 556.221 ms | 113.958 ms |
| Base 2 | 3,434.048 ms | 557.480 ms | 115.057 ms |

| Average of two process medians | Three-launch base | Fused gate/up/SiLU | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,434.2100 ms | 3,404.6840 ms | 29.5260 ms (0.859761%) |
| Time to first token | 557.4890 ms | 556.2535 ms | 1.2355 ms (0.221619%) |
| Subsequent token | 115.0690 ms | 113.9415 ms | 1.1275 ms (0.979847%) |

Every warmup and measured generation retained the exact 19 prompt IDs, 26
generated IDs, decoded text, `<|im_end|>`, and 44 runner steps. Dedicated C1,
C8, and C16 27B oracle runs passed. Release reported zero failures across 52
discovered tests with four skipped; ASan/UBSan reported zero failures across
51 discovered tests with four skipped, `detect_leaks=0`, and
`package_consumer` excluded. Clocks were unlocked, and these results remain
batch-one, single-prompt diagnostic evidence rather than randomized, release,
concurrent-request, or serving-throughput claims.

## NVFP4 post-attention residual/norm/gate/up/SiLU fusion

The worktree based on `06d0a53f1cafd6ee232c13c48e44391310704b20`
extends the exact aligned M1 NVFP4 `[17408,5120]` gate/up/SiLU kernel across
the preceding post-attention BF16 residual-add and centered-RMSNorm boundary.
The result is one 64-CTA, 256-thread launch. Every CTA reproduces the previous
256-thread reduction exactly: each thread processes its same 20 columns with
the same `fmaf` accumulation and reduction tree, then applies `rsqrtf`, gamma,
and BF16-RNE in the same order. CTA 0 alone writes the public residual; all
CTAs place the normalized 5,120-element activation only in 10,240 bytes of
shared memory before the rolled gate/up phases and CTA-parallel SiLU epilogue.
The reused shared allocation keeps the kernel at 11,328 static-shared bytes,
64 registers per thread, zero local bytes, and four active blocks per SM.

Eligibility remains explicit SM87, exact ordered NVFP4 gate/up
`[17408,5120]` weights, positive finite epsilon, and the required operand
alignments and disjoint writable ranges. The whole composite call validates
before its first enqueue, including device scalar-weight aliases and, when a
fallback needs it, FP32 scratch overlap with the residual-left and norm-weight
ranges. Aligned exact M1 uses one node. Unaligned or near-miss SM87 NVFP4 uses
four; reference NVFP4 uses six; aligned SM87 BF16 and FP8 pairs use three;
unaligned FP8 uses four. C2 through C16 never select this exact M1 fusion.

The final same-binary gate uses the pinned layer-0 gate/up checkpoint payloads
and the deterministic same-bank fixture. Each process performs 10 warmups,
64 logical chains per timed pass, and five baseline/candidate/candidate/base
rounds. The frozen thresholds are 1.005x for checkpoint bytes and 1.00x for
the stress guard:

| Independent process | Checkpoint baseline | Single kernel | Speedup | Stress speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 0.639913 ms | 0.623738 ms | 1.02593x | 1.02548x |
| 2 | 0.639629 ms | 0.623983 ms | 1.02507x | 1.02534x |
| 3 | 0.639832 ms | 0.623807 ms | 1.02569x | 1.02558x |
| 4 | 0.640239 ms | 0.624272 ms | 1.02558x | 1.02543x |
| 5 | 0.639722 ms | 0.623726 ms | 1.02565x | 1.02556x |

Checkpoint speedup min/median/mean/max is
1.02507x/1.02565x/1.025584x/1.02593x. Stress speedup is
1.02534x/1.02548x/1.025478x/1.02558x. Finite and nonfinite bounded fixtures
match the old two-call chain bitwise: residual, final gate, and up mismatches
are zero; all 4,096 nonfinite outputs preserve NaN class/sign; guards and
inputs remain intact. The public/test exact launches capture one identical
kernel node, while invalid calls capture zero.

Matched max-26-token Nsight reports compare an independently rebuilt detached
base with the candidate:

| Profile group | Detached base | Candidate | Change |
| --- | ---: | ---: | ---: |
| Gate/up/SiLU or new full fusion | 1,664 / 1,036.740480 ms | 1,664 / 1,053.055264 ms | absorbs residual/norm work |
| Standalone residual-add + norm | 3,328 / 74.653472 ms | 1,664 / 39.305440 ms | -1,664 launches / -35.348032 ms |
| Two tracked groups | 4,992 / 1,111.393952 ms | 3,328 / 1,092.360704 ms | -1,664 / -19.033248 ms (1.712556%) |
| All CUDA kernels | 18,550 / 3,400.911136 ms | 16,886 / 3,383.698144 ms | -1,664 / -17.212992 ms (0.506129%) |
| Profiled generation | 3,437.927 ms | 3,418.050 ms | -19.877 ms (0.578168%) |

The base report is 1,348,090 bytes with SHA-256
`727baa40d726211f58337a11b33d5beb59f63934fc25738f55aa40c61523fe28`;
the candidate is 1,252,092 bytes with SHA-256
`9cbb447e76a74db09b89c78f6b33ecd768ab31e9668ec6fd691cc491c6b429ab`.
The remaining 1,664 standalone norm calls are the other decoder norm boundary,
not a failure to dispatch the post-attention fusion.

The independent B-C-C-B comparison loads one engine per process, warms once,
and measures five generations:

| Process order | Total generation median | TTFT median | Subsequent-token median |
| --- | ---: | ---: | ---: |
| Base 1 | 3,405.924 ms | 556.400 ms | 113.996 ms |
| Candidate 1 | 3,381.458 ms | 555.453 ms | 113.036 ms |
| Candidate 2 | 3,380.342 ms | 555.367 ms | 113.008 ms |
| Base 2 | 3,405.890 ms | 556.151 ms | 113.991 ms |

| Average of two process medians | Detached base | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,405.9070 ms | 3,380.9000 ms | 25.0070 ms (0.734224%) |
| Time to first token | 556.2755 ms | 555.4100 ms | 0.8655 ms (0.155588%) |
| Subsequent token | 113.9935 ms | 113.0220 ms | 0.9715 ms (0.852242%) |

Every measured generation and dedicated C1/C8/C16 run retained the exact
19/26-token, decoded-text, `<|im_end|>`, and 44-step oracle. Release reported
zero failures across 52 discovered tests with four skips; ASan/UBSan reported
zero failures across 51 discovered tests with four skips and
`package_consumer` excluded. Clocks were unlocked. The
microbenchmark, separate profiles, and two-process-per-policy B-C-C-B result
remain batch-one, local-artifact diagnostics rather than randomized, release,
concurrent-request, or serving-throughput claims.

## NVFP4 post-attention RMSNorm warp-tail reduction

The reduction-only follow-up based on
`7770128a4fd68890817a830d5a79d60e8b08125f` keeps the public exact aligned
SM87 M1 `[17408,5120]` residual/norm/gate/up/SiLU launch at 64 CTAs and 256
threads. Each thread still accumulates its same 20 centered-residual squares
with the same left-to-right `fmaf` order. The production reduction retains the
shared-memory tree at strides 128, 64, and 32, then warp zero performs the same
pairwise additions with shuffle-down strides 16, 8, 4, 2, and 1. Lane zero
publishes the result to shared memory before one final block barrier. This
replaces five per-level tail barriers with the shuffle chain plus one publish
barrier, removing four reduction barriers per CTA without changing any
addition pairing, `rsqrtf`, gamma, BF16-RNE, projection, or SiLU operation.

The former eight-level shared-memory tree remains a test-only template
instance in the same binary. Production public and test launchers instantiate
the warp-tail kernel; graph capture verifies that they resolve to the same
kernel function and `64x256` topology, while the test-only shared-tree
predecessor is a distinct function with the same topology. Both variants use
64 registers per thread, 11,328 static-shared bytes, zero local bytes, and
permit four active blocks per SM on the tested Orin.

The direct optional gate compares those two single kernels on separate output
buffers. Each process warms both variants ten times, then runs five symmetric
B-C-C-B rounds with 64 launches per timed pass. B is the shared-tree
predecessor and C is the production warp tail. The statistic is the median of
ten B pass means divided by the median of ten C pass means. Both the pinned
checkpoint fixture and deterministic same-bank stress fixture have a frozen
minimum direct speedup of 1.005x:

| Process | Checkpoint shared tree | Checkpoint warp tail | Speedup | Stress shared tree | Stress warp tail | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.623346 ms | 0.613568 ms | 1.01594x | 0.626713 ms | 0.617878 ms | 1.01430x |
| 2 | 0.624090 ms | 0.614271 ms | 1.01598x | 0.641316 ms | 0.628109 ms | 1.02103x |
| 3 | 0.623477 ms | 0.614366 ms | 1.01483x | 0.628674 ms | 0.619247 ms | 1.01522x |
| 4 | 0.625712 ms | 0.616100 ms | 1.01560x | 0.627949 ms | 0.619546 ms | 1.01356x |
| 5 | 0.623689 ms | 0.614093 ms | 1.01563x | 0.627186 ms | 0.618388 ms | 1.01423x |

Checkpoint speedup min/median/mean/max is
1.01483x/1.01563x/1.015596x/1.01598x. Stress speedup is
1.01356x/1.01430x/1.015668x/1.02103x. All ten fixture/process cells clear the
1.005x threshold. The median per-process saving is 9.612 microseconds on the
checkpoint fixture and 8.835 microseconds on the stress fixture. Several r2
stress passes rose together under unlocked-clock thermal drift; the B-C-C-B
order and ten-pass median preserve a paired comparison, but the absolute r2
latencies are not treated as stable clock-locked measurements.

The measured binary passed the existing default exact-shape launcher identity,
zero-node invalid-call, bounded direct shared-tree/warp-tail bitwise, signed
Inf/NaN class/sign, canary, and input-preservation gates. Its optional fixture
also matched the old two-kernel chain bitwise at the full production shape for
both finite fixtures. The final closeout binary additionally matches the
shared-tree predecessor directly at full shape for actual finite, stress
finite, and signed Inf/NaN inputs: all residual/final/up mismatch counts are
zero, all 34,816 expected nonfinite projection outputs preserve NaN class and
sign, and every guard and input remains intact. Its hard-gate rerun measures
1.01474x actual and 1.01355x stress, clearing both frozen 1.005x thresholds.

A matched max-26 profile reuses the immediately preceding `7770128` capture
as the shared-tree baseline and independently captures the warp-tail candidate:

| Profile group | Shared tree | Warp tail | Change |
| --- | ---: | ---: | ---: |
| Post-attention fused kernel | 1,664 / 1,054.426816 ms | 1,664 / 1,044.102176 ms | -10.324640 ms (0.979171%, 1.009888534x) |
| Non-target kernels | 13,142 / 2,306.443104 ms | 13,142 / 2,313.795744 ms | +7.352640 ms |
| All CUDA kernels | 14,806 / 3,360.869920 ms | 14,806 / 3,357.897920 ms | -2.972000 ms (0.088429%, 1.000885078x) |
| Profiled generation | 3,392.935 ms | 3,390.068 ms | -2.867 ms (0.084499%, 1.000845706x) |

The target saving is directly visible, but unrelated kernels move in the
opposite direction under unlocked clocks, so the all-CUDA net is reported
alongside the target result rather than attributed wholly to the reduction.
The candidate profile preserves the exact 19/26-token, text, stop, and 44-step
contract.

An independent B-C-C-B comparison uses the `7770128` shared-tree binary and
the candidate warp-tail binary. Each process loads one engine, warms once, and
measures five generations with the same 19-token prompt and 26-token cap:

| Process order | Total generation median | TTFT median | Subsequent-token median |
| --- | ---: | ---: | ---: |
| Shared tree 1 | 3,375.874 ms | 555.941 ms | 112.824 ms |
| Warp tail 1 | 3,345.556 ms | 554.045 ms | 111.760 ms |
| Warp tail 2 | 3,344.837 ms | 553.941 ms | 111.663 ms |
| Shared tree 2 | 3,359.825 ms | 554.631 ms | 112.189 ms |

| Average of two process medians | Shared tree | Warp tail | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,367.8495 ms | 3,345.1965 ms | 22.6530 ms (0.672625%, 1.006771800x) |
| Time to first token | 555.2860 ms | 553.9930 ms | 1.2930 ms (0.232853%, 1.002333965x) |
| Subsequent token | 112.5065 ms | 111.7115 ms | 0.7950 ms (0.706626%, 1.007116546x) |

All four processes preserve the exact prompt IDs, generated IDs, decoded text,
`<|im_end|>` stop, and 44 runner steps. The six canonical contract lines hash
to `db918ca4ff54455c4b035d4ae851d38a02b97df9a3088414de1d354366c9585a`
in every log. A separate base/candidate `--trace` comparison requests C16 and
intentionally executes C1 trace order; both logs contain 44 traces and 5,905
canonical prompt/generated/boundary lines. Both canonical streams hash to
`8e33629f357c95fc26eb380c180e9645dddd47f5933adc6759aa01d69ca6d789`,
and their diff is empty. Release validation passes all 52 discovered tests
with four model-dependent skips. ASan/UBSan with
`detect_leaks=0:halt_on_error=1`, `UBSAN_OPTIONS=halt_on_error=1`, and
`package_consumer` excluded passes all 51 discovered tests with the same four
skips. The direct, profile, trace, and end-to-end
measurements use unlocked clocks and remain single-prompt diagnostics, not
serving-throughput or release claims; see the
[warp-tail reduction record](metadata/qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json).

## FP8 full-attention Q+K/V one-kernel fusion

The worktree based on `23ef5ab225ec53721d035027cb4940384f375a3a`
extends the exact full-attention M1 decode boundary from a standalone FP8 Q
projection followed by the existing fused K/V pair to one 2,048-CTA,
256-thread kernel. The ordered checkpoint shapes are Q `[12288,5120]`, K
`[1024,5120]`, and V `[1024,5120]`. The Q blocks preserve the production
row-quad accumulation order, and the K/V blocks preserve the paired reduction
order, so every output retains the old chain's independent BF16-RNE result.
The production kernel uses 64 registers per thread, 1,152 static-shared bytes,
zero local bytes, and four active blocks per SM.

Eligibility remains explicit SM87, M1, valid ordered FP8 payloads at the three
exact shapes, 4-byte-aligned weights, an 8-byte-aligned BF16 activation, and
2-byte-aligned mutually disjoint outputs. The composite validates Q, K, V,
all input/output/weight spans, required fallback scratch, and cross-projection
aliases before its first enqueue. An exact eligible call captures one kernel
node; invalid shapes and unsafe aliases capture zero. Near-miss or unaligned
calls, other weight types, and other backends retain the old validated Q
projection followed by the existing K/V pair, which itself falls back to two
ordered projections when ineligible. C2 through C16 prompt-prefix tiles retain
their preceding paths.

The isolated performance gate is a synthetic same-binary microbenchmark, not
a checkpoint-payload or end-to-end measurement. It uses deterministic
checkpoint-like E4M3FN code distributions, 10 warmups, 80 logical chains per
timed pass, and five baseline/candidate/candidate/base rounds in each of five
independent processes. Each row below is the median of that process's ten
baseline and ten candidate passes; the frozen acceptance threshold is 1.005x:

| Independent process | Q plus paired K/V | One kernel | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.421340 ms | 0.408413 ms | 1.03165x |
| 2 | 0.419815 ms | 0.409136 ms | 1.02610x |
| 3 | 0.420127 ms | 0.407571 ms | 1.03080x |
| 4 | 0.421085 ms | 0.409546 ms | 1.02818x |
| 5 | 0.421049 ms | 0.408431 ms | 1.03089x |

Speedup min/median/mean/max is
1.02610x/1.03080x/1.029524x/1.03165x. Q, K, and V report zero BF16-bit
mismatches across 12,288/1,024/1,024 outputs. The one-node production capture,
resource bounds, and invalid-call zero-node gate all pass. Dispatch coverage
also rejects an invalid third weight, mutually aliased outputs,
cross-weight/output aliases, output or scratch overlap with device-scalar
ranges, and unsafe scratch aliases before enqueue.

The separate max-26-token Nsight profiles compare an independently rebuilt
detached base with the candidate. Unlike the synthetic microbenchmark, these
profiles use the pinned model, but they are independent unlocked-clock
processes rather than a same-binary timing trial:

| Profile group | Detached base | Candidate | Change |
| --- | ---: | ---: | ---: |
| Full-attention Q | 416 / 157.074528 ms | absorbed | -416 launches |
| Full-attention K/V pair | 416 / 31.433728 ms | absorbed | -416 launches |
| Full-attention Q+K/V fusion | absent | 416 / 184.886112 ms | one kernel per layer/step |
| Replaced Q+K/V chain | 832 / 188.508256 ms | 416 / 184.886112 ms | -416 / -3.622144 ms (1.921478%) |
| All CUDA kernels | 16,886 / 3,449.954816 ms | 16,470 / 3,414.459904 ms | -416 / -35.494912 ms (1.028852%) |
| Profiled generation | 3,484.492 ms | 3,446.214 ms | -38.278 ms (1.098525%) |

The directly replaced-chain row is the reliable kernel attribution. Aggregate
kernel and generation differences also include process-to-process clock and
unrelated-kernel variation. The base report is 1,252,279 bytes with SHA-256
`7a94473f41ece92a3d35ef654920fb03f43c18317f4015b1d1c09506368a1052`;
the candidate is 1,222,499 bytes with SHA-256
`50f29dc69a70b5b73923ccbc127b56f94172793c6ffe8f60326734b71c463f46`.

The detached-base B-C-C-B comparison is a third, separate measurement. It
loads one engine per process, warms once, and measures five generations:

| Process order | Total generation median | TTFT median | Subsequent-token median |
| --- | ---: | ---: | ---: |
| Base 1 | 3,381.952 ms | 555.451 ms | 113.063 ms |
| Candidate 1 | 3,376.378 ms | 555.093 ms | 112.848 ms |
| Candidate 2 | 3,376.302 ms | 555.191 ms | 112.826 ms |
| Base 2 | 3,381.028 ms | 555.512 ms | 113.018 ms |

| Average of two process medians | Detached base | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,381.4900 ms | 3,376.3400 ms | 5.1500 ms (0.152300%) |
| Time to first token | 555.4815 ms | 555.1420 ms | 0.3395 ms (0.061118%) |
| Subsequent token | 113.0405 ms | 112.8370 ms | 0.2035 ms (0.180024%) |

Every warmup and measured generation retained the exact 19 prompt IDs, 26
generated IDs, decoded text, `<|im_end|>`, and 44 runner steps. Dedicated C1,
C8, and C16 oracle runs passed the same gate. Release reported zero failures across
52 discovered tests with four skips; ASan/UBSan reported zero failures across
51 discovered tests with four skips, `detect_leaks=0`, and `package_consumer`
excluded. Clocks were unlocked. The synthetic same-binary microbenchmark,
independent model profiles, and detached-base B-C-C-B result remain distinct
batch-one, local-artifact diagnostics rather than randomized, release,
concurrent-request, or serving-throughput claims.

## NVFP4 down/residual/centered-RMSNorm cooperative fusion

The worktree based on `d047007d37264793153b0d81d1c5f30bd4bf30cc`
extends the exact aligned M1 NVFP4 `[5120,17408]` dense-MLP down projection
across its following BF16 residual-add and centered-RMSNorm boundary. The
projection phase retains the production activation-staged accumulation and
BF16-RNE raw-down store. Each CTA then publishes its assigned rounded residual
rows, a cooperative grid synchronization makes all 5,120 residual values
visible, and every CTA repeats the exact 256-thread RMS reduction. The first
20 CTAs publish disjoint 256-element normalized slices. This preserves three
separate public BF16 boundaries--raw down, rounded residual, and normalized
next-layer input--rather than hiding the trace-visible raw projection.

The kernel launches 64 cooperative CTAs of 256 threads. It uses 64 registers
per thread, 35,904 static-shared bytes, zero local bytes, and permits four
active CTAs per SM. The tested Jetson AGX Orin has 16 SMs, so its resident-grid
capacity is exactly `16 * 4 = 64` CTAs: sufficient for the grid-wide barrier,
but with no capacity margin. Any resource change that reduced residency, or a
different SM count/configuration, would require revalidation. This is therefore
an AGX Orin `sm_87` result, not a general cooperative-kernel portability claim.

Eligibility is explicit SM87, M1, a valid NVFP4 payload at the exact
`[5120,17408]` shape, 4-byte-aligned packed weights, an 8-byte-aligned BF16
activation, and naturally aligned BF16 residual/norm inputs and three mutually
disjoint outputs. The runtime validates the weight and auxiliary payloads,
device-scalar ranges, activation, residual input, norm weight, all three output
spans, fallback scratch, overflows, and cross-aliases before its first enqueue;
safe read-only overlaps remain allowed. Exact aligned dispatch captures one
kernel node. Unaligned or near-miss SM87 NVFP4 retains the old down projection
then residual/norm sequence with two nodes, while reference NVFP4 retains its
projection, BF16 conversion, and residual/norm sequence with three. Other
types/backends and C2 through C16 prefill retain their prior valid routes.

The decode workspace deliberately reuses the dead post-SiLU up buffer
`projection[1]` for raw down, writes the public residual to `hidden[0]`, and
writes the normalized next-layer input to `hidden[1]`. `trace_layer_hidden`
copies from `projection[1]`, preserving the raw-down trace topology; the final
layer selects final-norm weights before dispatch, while other layers select the
next layer's input-norm weights.

The default production gate captures one node at grid 64/block 256 and checks
the exact resource/capacity contract. Its finite fixture reports zero bitwise
mismatches for all 5,120 raw, residual, and normalized BF16 outputs with intact
guards. Two signed-nonfinite fixtures--one injected through the residual input
and one through the norm weight--also report zero bitwise mismatches for all
three outputs, correct class/sign behavior, and intact guards. Forty-two bad
shape, null, alignment, overflow, input/output-alias, and output/output-alias
cases all return before enqueue and capture zero graph nodes. The higher-level
dispatch gate additionally covers malformed weight kinds/payloads, both device
scalars, scratch sizing and aliases, and the one-node/two-node/three-node route
matrix. A small `[5120,16]` fallback fixture matches the old raw, residual, and
normalized chain bitwise.

The isolated performance gate uses deterministic synthetic NVFP4 packed
weights, checkpoint-like E4M3FN block scales, and finite BF16 activation,
residual, and norm inputs. Each independent process runs ten paired warmups,
then five baseline/candidate/candidate/baseline rounds with 40 launches per
timed pass. Each baseline/candidate cell is the arithmetic mean of its ten
pass means across the five rounds, and speedup is the ratio of those means.
The frozen acceptance threshold is 1.005x:

| Independent process | Down plus residual/norm | Cooperative fusion | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.329520 ms | 0.319087 ms | 1.03270x |
| 2 | 0.331475 ms | 0.322251 ms | 1.02862x |
| 3 | 0.329008 ms | 0.318182 ms | 1.03403x |
| 4 | 0.329976 ms | 0.319720 ms | 1.03208x |
| 5 | 0.329373 ms | 0.318930 ms | 1.03274x |

Speedup min/median/mean/max is
1.02862x/1.03270x/1.032034x/1.03403x. Baseline-time
min/median/mean/max is 0.329008/0.329520/0.329870/0.331475 ms; candidate-time
min/median/mean/max is 0.318182/0.319087/0.319634/0.322251 ms. This is a
synthetic same-binary gate rather than checkpoint-payload or end-to-end timing.

The matched max-26-token Nsight profiles compare an independently rebuilt
detached `d047007` base with the candidate under the same pinned-model command.
They are separate unlocked-clock processes:

| Profile group | Detached base | Candidate | Change |
| --- | ---: | ---: | ---: |
| NVFP4 down projection | 1,664 / 521.471936 ms | absorbed | -1,664 launches |
| Residual add + centered RMSNorm | 1,664 / 38.932800 ms | absorbed | -1,664 launches |
| Down/residual/norm cooperative fusion | absent | 1,664 / 539.154752 ms | one kernel per layer/step |
| Directly replaced chain | 3,328 / 560.404736 ms | 1,664 / 539.154752 ms | -1,664 / -21.249984 ms (3.791899%) |
| All CUDA kernels | 16,470 / 3,380.586880 ms | 14,806 / 3,360.869920 ms | -1,664 / -19.716960 ms (0.583241%) |
| Profiled generation | 3,415.619 ms | 3,392.935 ms | -22.684 ms (0.664126%) |

The directly replaced-chain row is the reliable kernel attribution. Aggregate
non-target kernels increased by 1.533024 ms in aggregate, offsetting the
21.249984 ms target saving to the 19.716960 ms all-CUDA reduction. Aggregate
kernel and generation deltas also contain process-to-process clock and
unrelated-kernel variation. The base report is 1,224,976 bytes with SHA-256
`5cbff073881a6db409c25d0676bbf3f655ee15d8ab8cc42b3ffb6bfb419af3fb`;
the candidate is 1,121,911 bytes with SHA-256
`2f9f99bb05e6fd1925a0565e2fa9ef294f488ca84af5d1ad8df60370dfa06e5f`.

The detached-base B-C-C-B comparison is a third measurement. Each process
loads one engine, warms once, and measures five max-26 generations:

| Process order | Total generation median | TTFT median | Subsequent-token median |
| --- | ---: | ---: | ---: |
| Base 1 | 3,378.228 ms | 555.270 ms | 112.920 ms |
| Candidate 1 | 3,358.844 ms | 554.479 ms | 112.171 ms |
| Candidate 2 | 3,360.510 ms | 554.637 ms | 112.226 ms |
| Base 2 | 3,376.944 ms | 555.207 ms | 112.890 ms |

| Average of two process medians | Detached base | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,377.5860 ms | 3,359.6770 ms | 17.9090 ms (0.530231%) |
| Time to first token | 555.2385 ms | 554.5580 ms | 0.6805 ms (0.122560%) |
| Subsequent token | 112.9050 ms | 112.1985 ms | 0.7065 ms (0.625747%) |

Every warmup and measured generation retained the exact 19 prompt IDs, 26
generated IDs, decoded text, `<|im_end|>`, and 44 runner steps. A separate
detached-base/candidate max-26 `--trace` comparison requested C16 and, as the
trace contract requires, reported effective C1; its full 5,902-line contract,
covering every layer-boundary hash in all 44 steps plus the prompt/generated
contract, matched line by line. Dedicated C1, C8, and C16
oracle processes passed. Release reported zero failures across 52 discovered
tests with four skips. ASan/UBSan reported zero failures across 51 discovered
tests with four skips and `package_consumer` excluded. Clocks were unlocked.
The synthetic microbenchmark, independent profiles, detached-base B-C-C-B,
and trace comparison remain distinct batch-one local-artifact diagnostics,
not randomized, release, concurrent-request, or serving-throughput claims.

## GDN/plain-RMSNorm/SiLU-gate one-kernel fusion

Commit `71be7d3ec6a3d9a48510b4c66b01846fde757a16` extends the canonical
single-token eight-row lane-striped GDN update from base
`3175e9d66fc12e15c743df1c8081df76f8b4daa9` across its following headwise
plain-RMSNorm and SiLU-gate boundary. One CTA owns each of the 48 value heads,
so the exact 48x128 route needs only a block barrier: the GDN phase publishes
its raw result to global BF16, the same CTA reads that rounded boundary back,
performs the existing 256-thread RMS reduction, applies the shared plain norm
weight and elementwise SiLU gate, and overwrites the output in BF16. The launch
is 48 CTAs by 256 threads and needs neither a cooperative launch nor grid
synchronization.

The final kernel uses 40 registers per thread, 34,568 static-shared bytes,
zero local bytes, supports 1,024 threads per block, and permits four active
blocks per SM. All 48 CTAs therefore fit one resident wave across the tested
16-SM Orin. The composite supports exact in-place state or disjoint
state-input/state-output buffers, rejects partial state overlap, and validates
all pointers, dimensions, epsilons, byte-range overflows, alignments, writable
spans, aliases, and fallback requirements before enqueue. Canonical M1 48x128
captures one kernel node; a valid 24x256 partition and the explicit ordered
reference capture two; invalid calls capture zero. Multi-token prefill remains
on the ordered path.

The production same-binary gate rotates 24 independent 1.5 MiB persistent
states per variant, or 36 MiB per bank versus the device's 4 MiB L2. It resets
each bank before a timed pass, runs 48 warmup and 480 measured logical chains,
and repeats five B-C-C-B rounds per process. Each row is the ratio of the
medians of ten ordered and ten fused pass means. After timing, the complete
state bank and final output are compared bitwise. All five runs cleared the
frozen 1.005x threshold:

| Independent process | Ordered GDN + norm/gate | One fused kernel | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.0356057 ms | 0.0320374 ms | 1.11138x |
| 2 | 0.0356020 ms | 0.0320513 ms | 1.11078x |
| 3 | 0.0357769 ms | 0.0320749 ms | 1.11542x |
| 4 | 0.0356452 ms | 0.0320201 ms | 1.11321x |
| 5 | 0.0357787 ms | 0.0320570 ms | 1.11610x |

The ordered and fused means are 0.035681700 and 0.032048140 ms, a
3.633560 us saving and 1.113378187x ratio of means. Speedup
min/median/mean/max is 1.11078x/1.11321x/1.113378x/1.11610x. Every run also
passed the post-timing bitwise and resource gates. The measured test binary is
776,120 bytes with SHA-256
`d5ad8847800ccb45637d5cdb429b8a2db9bfacd29594bbff8fb571665954da4a`.
The earlier fixed-order, single-state prototype probes remain catalogued in
metadata but are explicitly excluded from production acceptance.

Matched max-26 Nsight profiles compare the immediately preceding post-attention
binary with the candidate under the same exact generation contract:

| Profile group | Detached base | Candidate | Change |
| --- | ---: | ---: | ---: |
| Decode GDN + standalone norm/gate | 2,496 / 44.600480 ms | 1,248 / 40.273216 ms | -1,248 / -4.327264 ms (9.702281%, 1.107447689x) |
| Prefill GDN + norm/gate | 192 / 22.383360 ms | 192 / 22.386752 ms | +0.003392 ms |
| All CUDA kernels | 14,806 / 3,357.897920 ms | 13,558 / 3,345.617760 ms | -1,248 / -12.280160 ms (0.365710%, 1.003670521x) |
| Profiled generation | 3,390.068 ms | 3,373.952 ms | -16.116 ms (0.475389%, 1.004776594x) |

The decode target row is the isolated fusion attribution: its base side is
1,248 GDN launches taking 39.241600 ms plus 1,248 norm/gate launches taking
5.358880 ms. Unrelated kernels independently move by -7.956288 ms, so neither
the all-kernel nor generation delta is attributed wholly to this fusion. Both
profiles preserve the exact 19 prompt IDs, 26 generated IDs, decoded text,
`<|im_end|>` stop, and 44 runner steps.

The detached-base B-C-C-B comparison loads one engine per process, warms once,
and measures five max-26 generations:

| Process order | Total generation median | TTFT median | Subsequent-token median |
| --- | ---: | ---: | ---: |
| Base 1 | 3,346.852 ms | 554.031 ms | 111.715 ms |
| Candidate 1 | 3,337.188 ms | 553.630 ms | 111.353 ms |
| Candidate 2 | 3,341.281 ms | 554.053 ms | 111.494 ms |
| Base 2 | 3,343.525 ms | 553.961 ms | 111.592 ms |

| Average of two process medians | Detached base | Candidate | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,345.1885 ms | 3,339.2345 ms | 5.9540 ms (0.177987%, 1.001783043x) |
| Time to first token | 553.9960 ms | 553.8415 ms | 0.1545 ms (0.027888%, 1.000278961x) |
| Subsequent token | 111.6535 ms | 111.4235 ms | 0.2300 ms (0.205994%, 1.002064197x) |

All four processes preserve the exact generation contract and share canonical
contract SHA-256
`db918ca4ff54455c4b035d4ae851d38a02b97df9a3088414de1d354366c9585a`.
Dedicated C1, C8, and C16 oracle runs also pass. An independent base/candidate
trace comparison requests C16, follows the intentional effective-C1 trace
order, and produces 5,905 canonical prompt/generated/boundary lines per log;
both streams hash to
`8e33629f357c95fc26eb380c180e9645dddd47f5933adc6759aa01d69ca6d789`
with an empty diff. Release reports zero failures across 52 discovered tests
with four skips; ASan/UBSan reports zero failures across 51 discovered tests
with four skips and
`package_consumer` excluded.

Clocks remained unlocked. The hardened microbenchmark, matched profiles,
two-process-per-policy end-to-end comparison, and single-prompt exact gates
remain separate batch-one diagnostics rather than randomized-clock,
continuous-batching, concurrent-request, release, or serving-throughput
claims. Exact paths, byte counts, SHA-256 identities, and calculation details
are in the
[GDN fusion metadata record](metadata/qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json).

## Fused GDN RMSNorm warp-tail reduction

Commit `c4625b6bf8c31d9b23de030e4bfff04aa5cc3035` follows base
`21bb8ac47c4d2c68ca2ebd07a8a7b0c47a7865ad` by changing only the exact
256-thread RMS reduction inside the canonical M1 fused GDN/plain-RMSNorm/
SiLU-gate kernel. Production retains the shared-memory 128/64/32 pairings,
then warp zero performs the same 16/8/4/2/1 pairings with shuffle-down. Lane
zero publishes the sum to shared element zero before one final block barrier.
The complete shared-memory tree is preserved behind a test-local entry point;
it is neither a public route nor part of the production claim.

Both variants launch 48 CTAs by 256 threads with zero dynamic-shared bytes,
use 40 registers per thread, 34,568 static-shared bytes and zero local bytes,
and permit four active blocks per SM. CUDA graph capture confirms the public
and detail exact routes use the same production function and launch parameters,
while the test-only shared-tree function is distinct but has the same topology.
A three-step direct test compares all 786,432 BF16 state elements and all 6,144
BF16 output elements bitwise after every step. Disassembly counts 28 `BAR`
instructions in the full-tree instantiation and 24 in production, matching the
four removed reduction barriers.

The final same-binary gate rotates 24 independent 1.5 MiB states per variant,
or 36 MiB per bank against the 4 MiB device L2. Every timed pass resets the
bank, runs 48 warmup and 480 measured logical chains, and each process repeats
five symmetric B-C-C-B rounds. The reported process statistic divides the
median of ten shared-tree pass means by the median of ten production pass
means. The complete state bank and final output are compared bitwise after
timing. All five independent processes clear the frozen 1.005x threshold:

| Independent process | Test-only shared tree | Production warp tail | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.0320791 ms | 0.0317545 ms | 1.01022x |
| 2 | 0.0320421 ms | 0.0317470 ms | 1.00930x |
| 3 | 0.0320664 ms | 0.0317624 ms | 1.00957x |
| 4 | 0.0320384 ms | 0.0317359 ms | 1.00953x |
| 5 | 0.0320527 ms | 0.0317946 ms | 1.00812x |

The shared-tree and warp-tail means are 0.032055740 and 0.031758880 ms,
a 0.296860 us saving and 1.009347307x ratio of means. The speedup
min/median/arithmetic-mean/max is
1.00812x/1.00953x/1.009348x/1.01022x. A separate regression run still measures
the ordered GDN-plus-norm/gate chain at 0.0357226 ms and the production fused
kernel at 0.0317504 ms, or 1.12511x; its full state/output bitwise and resource
gates pass.

Matched max-26 Nsight profiles retain the same launch count and isolate the
reduction change within the production fused kernel:

| Profile group | Fused shared-tree base | Warp-tail candidate | Change |
| --- | ---: | ---: | ---: |
| Decode fused GDN target | 1,248 / 40.273216 ms | 1,248 / 39.977728 ms | -0.295488 ms (0.733708%, 1.007391315x) |
| All CUDA kernels | 13,558 / 3,345.617760 ms | 13,558 / 3,338.262016 ms | -7.355744 ms (0.219862%, 1.002203465x) |
| Profiled generation | 3,373.952 ms | 3,367.721 ms | -6.231 ms (0.184680%, 1.001850213x) |

Only the decode-target row is directly attributed to this change. Other
kernels account for 7.060256 ms of the all-kernel movement, so the all-kernel
and generation rows remain diagnostic rather than attributable results. Both
profiles preserve the exact 19 prompt IDs, 26 generated IDs, decoded text,
`<|im_end|>` stop, and all 44 runner steps.

The detached-base B-C-C-B comparison loads one engine per process, warms once,
and measures five max-26 generations:

| Process order | Total generation median | TTFT median | Subsequent-token median |
| --- | ---: | ---: | ---: |
| Base 1 | 3,338.588 ms | 553.915 ms | 111.392 ms |
| Candidate 1 | 3,338.842 ms | 553.905 ms | 111.390 ms |
| Candidate 2 | 3,340.697 ms | 553.697 ms | 111.478 ms |
| Base 2 | 3,339.659 ms | 553.920 ms | 111.428 ms |

| Average of two process medians | Fused shared-tree base | Warp-tail candidate | Reduction |
| --- | ---: | ---: | ---: |
| Total generation | 3,339.1235 ms | 3,339.7695 ms | -0.6460 ms (-0.019346%, 0.999806573x) |
| Time to first token | 553.9175 ms | 553.8010 ms | 0.1165 ms (0.021032%, 1.000210364x) |
| Subsequent token | 111.4100 ms | 111.4340 ms | -0.0240 ms (-0.021542%, 0.999784626x) |

The mixed sub-millisecond movements include a 0.646 ms total-generation
regression and therefore establish no end-to-end gain; they are reported as
unlocked-clock noise. All four processes still preserve the exact generation
contract. Dedicated C1, C8 and C16 oracle runs pass. An independent trace
comparison emits 5,905 canonical lines per binary; both streams hash to
`8e33629f357c95fc26eb380c180e9645dddd47f5933adc6759aa01d69ca6d789`
and have an empty diff. The synchronized main Release suite reports zero
failures across 52 discovered tests with four skips. ASan/UBSan reports zero
failures across 51 discovered tests with four skips and `package_consumer`
excluded; leak detection remains disabled for the CUDA driver environment.

Clocks remained unlocked. The hardened same-binary gate, matched profiles,
two-process-per-policy end-to-end comparison and single-prompt exact gates are
separate batch-one diagnostics, not randomized-clock, continuous-batching,
concurrent-request, release or serving-throughput claims. Early exploratory
prototype timings and the test-only shared-tree implementation are explicitly
excluded from production acceptance. Exact paths, byte counts, SHA-256
identities and calculation details are in the
[GDN warp-tail metadata record](metadata/qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json).

## Correctness gate

The historical `reference_engine_e2e` gate at `5fe0ae0` loaded the pinned 27B
checkpoint with the SM87 backend at both C1 and C8 and matched all 19 prompt
IDs, 26 generated IDs, exact UTF-8 text, `<|im_end|>`, and all 44 runner steps.
The lightweight kernel gate also compared SM87 directly with the CUDA reference for FP8 and
NVFP4 M=1 through M=8 awkward/fallback/vector shapes. It covers every one of
the 256 E4M3FN codes in all four positions of the packed FP8 load, including both reserved
NaN encodings; all 16 E2M1 codes in all eight packed nibble positions;
adjacent-lane scale selection; both unaligned FP8 fallbacks; and
K=5120/6144/17408. The fixed-shape gates additionally compare the two NVFP4
production shapes under checkpoint-like and same-bank-stress scales and all
five FP8 production shapes, with zero BF16-bit mismatches. The deterministic
CUDA outputs match the reference bit-for-bit, while the independent host
oracle checks the reserved E4M3FN outputs by NaN class. The non-model test
suite also passes.
Parallel reduction order is still not a general bitwise-equivalence promise,
so `reference` remains the default projection backend. The accelerated loader
is independent of that projection choice and is now the default when AF_ALG
is available. The complete packed-x4 fixed-oracle CTest passed in 40.60
seconds.

The current sequence through `33948e3` adds C16 projection, causal-state,
controller, request-plan, and full-model gates. C1, C8, and C16 all retain the
same fixed 19/26-token, text, stop, and 44-step result; M16 Tensor Core kernels
are accepted by per-operation tolerance and deterministic-replay gates rather
than a blanket bitwise-equivalence claim.
For the latest down/residual/norm milestone, independent detached `d047007`
and candidate max-26 trace runs (requested C16, effective C1) also produced an
empty diff across the full 5,902-line trace contract: every layer-boundary hash
in all 44 steps is line-for-line identical.

## Phase 3 decision

The optimized path now includes SM87 single-token decode plus bounded C16
prompt-prefix projection reuse and fixed-shape Tensor Core dispatch:

1. keep `sm87` explicit while prompt coverage expands;
2. retain shape-driven packed-x8 dispatch for aligned canonical NVFP4 M=1;
3. retain shape-driven packed-x4 dispatch for aligned canonical FP8 M=1;
4. retain C1 as the compatibility default, expose validated C2 through C16
   prompt-prefix tiles explicitly, and retain measured C8/C16 evidence;
5. use fixed NVFP4 M=8 kernels only for exact `[17408,5120]` and
   `[5120,17408]`, while retaining the generic row-pair kernel and independent
   fallbacks for every other shape;
6. use fixed FP8 M=8 kernels only for exact `[10240,5120]`, `[5120,6144]`,
   `[6144,5120]`, `[12288,5120]`, and `[1024,5120]`, with the same generic and
   fallback coverage elsewhere;
7. use fixed-M16 Tensor Core kernels only for the four measured FP8 shapes and
   two measured NVFP4 shapes, retaining two-M8 fallback for every other valid
   shape/alignment and for FP8 `[1024,5120]`;
8. fuse full-attention FP8 Q+K/V into one kernel only for aligned ordered M1
   `[12288,5120]`, `[1024,5120]`, and `[1024,5120]` operands; otherwise retain
   the validated Q projection followed by paired K/V when eligible or two
   independent K/V projections for every other valid case;
9. pair linear-attention FP8 QKV/Z only for aligned ordered M1
   `[10240,5120]` and `[6144,5120]` operands, retaining two ordered
   projections for C2 through C16 and every other valid case;
10. fuse dense-MLP NVFP4 gate/up/SiLU only for aligned ordered M1
    `[17408,5120]` operands, retaining two ordered projections plus SiLU for
    C2 through C16 and every other valid case;
11. fuse post-attention BF16 residual add and centered RMSNorm into that exact
    aligned NVFP4 M1 gate/up/SiLU route, retaining the fully prevalidated
    ordered chain for every fallback;
12. fuse the exact aligned NVFP4 M1 `[5120,17408]` down projection and its
    following BF16 residual-add/centered-RMSNorm boundary only on the validated
    64-CTA cooperative AGX Orin route, retaining three public BF16 boundaries
    and the prevalidated ordered fallback everywhere else;
13. use CTA activation staging for aligned M1 down `[5120,17408]`, gate/up
    `[17408,5120]`, and lm-head `[248320,5120]`, retaining the direct down XOR
    test baseline and all other fallbacks;
14. use the eight-row lane-striped GDN update for M1 and bounded C2 through C16,
    retaining the four-row lane-striped predecessor as a same-binary test
    baseline;
15. fuse the canonical M1 GDN update with its 48x128 headwise plain-RMSNorm
    and SiLU gate in one 48-CTA kernel, retaining the ordered two-launch path
    for other valid norm partitions and multi-token prefill;
16. retain shared-memory strides 128/64/32 but finish the fused GDN kernel's
    exact RMS reduction with warp-zero shuffle-down strides 16/8/4/2/1,
    retaining the full shared tree as a test-only same-binary predecessor;
17. retain exact-token, numerical, replay, and memory gates for every dispatch
    change;
18. lock clocks when privileged access is available before making a formal
    release performance claim.

The small-M and bounded `C<=16` gates are complete. The next prefill work
is broader prompt/shape coverage and a measured dispatch boundary between the
current bounded tiles and future larger-prefill kernels. `.q3x` repacking
waits for a stable physical layout.

## Prefill/Decode control-plan split

Commit `21bafb4` adds an internal host-control seam without changing the device
schedule. `PrefillPlan` owns scalar/tiled prefix routing and the final
prompt/logits step; `DecodePlan` owns only later generated-token feedback.
Production binds both plans to the same `EngineStepContext`, `ReferenceRunner`,
request state, workspace, and CUDA stream. The compatibility control entry
point remains available. No runner, request-state, kernel, workspace, stream,
or buffering policy changed, and no performance gain is attributed to this
milestone.

The pure-host gate now covers explicit phase contexts and 44 combinations of
11 prompt lengths with C1/C2/C8/C16. The canonical 19-token route remains 18
scalar prefix steps at C1, `8+8+2` at C8, and `16+2` at C16, followed by exactly
one `finish_prefill` step and, when required, `decode_step`. Trace capture still
forces the prefix to C1 without erasing the logical phase boundary.

Independent SM87 C1/C8/C16 27B processes retain the exact 19 prompt IDs, 26
generated IDs, decoded text, `<|im_end|>`, and 44-step sequence. Their one-shot
sequence capacity remains 44 and their request arenas remain 82,505,216,
83,696,128, and 85,057,536 bytes respectively. A requested-C16/effective-C1
trace retains all 5,905 canonical lines with SHA-256
`8e33629f357c95fc26eb380c180e9645dddd47f5933adc6759aa01d69ca6d789`.
Nsight comparison also retains the exact ordered 13,558-entry launch contract
(API, grid, block, and kernel base name), with identical SHA-256
`7a0968fdf449115148537ca5ec9594029f68747fae6bb92998cdafad21da6d41`.

The detached-base B-C-C-B diagnostic averages are:

| Metric | Baseline | Candidate | Candidate change |
|---|---:|---:|---:|
| Total generation | 3,341.5435 ms | 3,341.0340 ms | -0.01525% |
| TTFT | 553.9785 ms | 554.1095 ms | +0.02365% |
| Subsequent token | 111.5050 ms | 111.4875 ms | -0.01569% |

All movements are below 0.024% and are treated as unlocked-clock noise. Release
passes all 52 discovered tests with four fixture/model skips; ASan/UBSan passes
all 51 selected tests with the same four skips. The three SM87 model E2E cases
were run separately rather than counted as skips. This milestone does not claim
broader real-prompt coverage, independent executors, CUDA Graphs, double/triple
buffering, multi-stream overlap, or improved utilization. Full commands, local
artifact identities, and limitations are in the
[Prefill/Decode plan-split record](metadata/qwen36-27b-prefill-decode-plan-split-benchmark.json).

## Frozen SM87 shape/chunk/prompt matrix

Commit `471b7a0` replaces six duplicated single-projection decision chains with
one private constexpr registry. The registry resolves format, exact shape,
alignment, leaf route, block cap, and recursive fallback count without changing
the public ABI, validation order, launchers, fusion eligibility, or fallback
semantics. Release passes 49 of 53 discovered tests with four fixture/model
skips; ASan/UBSan passes 48 of 52 selected tests with the same four skips.
Independent real-model C1/C8/C16 runs retain the exact 19 prompt IDs, 26 output
IDs, decoded text, stop token, and 44 steps. Baseline and candidate max-26 C16
profiles also match the complete ordered 13,558-launch contract byte-for-byte.

The intended direct-kernel atlas contains 29 production-route cells:

| Tile | FP8 cells | NVFP4 cells | Call-weighted result |
| --- | ---: | ---: | ---: |
| M1 | 5 | 3 | all latency/identity gates pass |
| M2 | 5 | 2 | 1.58984x versus two M1, required 1.50x |
| M8 | 5 | 2 | 2.99347x versus eight M1, required 2.75x |
| M16 | 5 | 2 | FP8 2.41820x and NVFP4 1.69116x versus two M8 |

All 29 pass. FP8 `[1024,5120]` is an important negative selector case: its raw
M16 WMMA candidate reaches only 0.559068x versus two M8, so production correctly
retains the 0.147214 ms two-M8 route. The optional test process nevertheless
exits 1 because an additional historical M4 aggregate check reports 1.50844x
against a 2.50x threshold, even though its seven individual cells pass. That
separate gate is frozen as a measurement-definition issue pending mirrored
reruns; it is neither silently relaxed nor reported as a production failure.

With one warmup and three measured maximum-one-token iterations, the exact
19-token prompt gives:

| Requested chunk | 18-token prefix schedule | TTFT |
| ---: | --- | ---: |
| C1 | 18 x M1 | 2,031.901 ms |
| C2 | 9 x M2 | 1,366.633 ms |
| C8 | M8 + M8 + M2 | 831.525 ms |
| C16 | M16 + M2 | 554.386 ms |

C16 is 3.664x faster than C1 and 1.500x faster than C8 in this diagnostic.
Tokenizer-pinned repeated-text P33/P65/P129/P513 cases produce exact 32/64/
128/512-token C16 prefixes and record 721.468, 1,347.115, 2,660.541, and
11,927.270 ms TTFT respectively. Their raw text, rendered chat, token IDs,
hashes, schedules, and first generated token are checked in as
`benchmarks/qwen36-27b-sm87-prefill-prompts-v1.json`.

Matched P19 Nsight profiles reduce launches from 8,249 at C1 to 2,633 at C16
and summed kernel time from 2,039.659200 to 568.984352 ms. Projection remains
dominant: 519.757216 ms, or 91.348%, at C16. The C16 `16+2` tail attributes
83.854 ms to NVFP4 M2 and 44.623 ms to FP8 M2. A maximum-two-token C16 profile
shares the exact first 2,633 ordered launches with the maximum-one profile;
their delta is one logical decode step with 437 launches, 109.942112 ms summed
kernel time, and 111.600 ms wall latency.

These measurements keep C16 as the largest validated production tile while
selecting C32/M17-M32 composition or tensor work as the next implementation.
M2 tail optimization is the secondary kernel target. A dense cuBLASLt/Marlin
crossover still needs an actual candidate, while double/triple buffering and
multi-stream overlap remain deferred until NCU exposes a concrete stall and
overlap opportunity. The complete values, hashes, local artifacts, and limits
are in the [shape/chunk/prompt matrix record](metadata/qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json).

## Historical C32 composite prefill baseline

Commit `e6fac6b` adds a C32 composite outer prefill tile and advances the
project/package ABI to 0.2.0. Release validation reports 49 passes and 5 skips
from 54 tests; the selected ASan/UBSan run reports 48 passes and 5 skips from
53. Real-model C1/C8/C16/C32 oracle runs all match the pinned 19 prompt IDs, 26
generated IDs, exact text and `im_end`, and all 44 logical steps.

The P19 same-binary B-C-C-B diagnostic uses one warmup and five measured
maximum-one-token iterations. Its C16 medians are 553.753 and 553.988 ms; its
C32 medians are 549.143 and 548.750 ms. Averaging each mirrored pair gives
553.8705 versus 548.9465 ms, a C32 delta of -4.924 ms (-0.8890%). The run was
made before the source commit, but the measured executable's 3,288,680 bytes
and SHA-256 match the subsequently identified `e6fac6b` build product exactly.

| Prompt | Prefix tokens | C16 TTFT | C32 TTFT | Delta |
| --- | ---: | ---: | ---: | ---: |
| P33 | 32 | 721.286 ms | 713.792 ms | -1.0390% |
| P65 | 64 | 1,348.561 ms | 1,331.483 ms | -1.2664% |
| P129 | 128 | 2,661.077 ms | 2,628.096 ms | -1.2394% |
| P513 | 512 | 11,930.745 ms | 11,762.383 ms | -1.4112% |

Matched P19 Nsight profiles reduce launches from 2,633 to 2,264, summed kernel
time from 566.903840 to 563.373792 ms, and kernel span from 577.084768 to
572.684320 ms. Projection is unchanged in structure: 1,089 launches consume
517.721824 ms at C16 and 517.828832 ms at C32. The only launch-count reductions
are headwise RMSNorm (-177), residual add (-128), and SiLU multiply (-64).

The P33/C32 profile supplies the target baseline for a future true M32
projection: 2,534 launches, 728.236864 ms summed kernel time, 736.294432 ms
kernel span, and 1,121 projection launches consuming 651.548672 ms. The current
`M16+M16` path includes 384 NVFP4 M16 launches taking 393.150656 ms and 352 FP8
M16 launches taking 124.776832 ms; that repeated M16 work is the concrete next
projection target, not evidence that a single-pass M32 kernel already exists.

This is deliberately a composite baseline, not a single-pass M32 result. P19
changes the outer schedule from `M16+M2` to one 18-row outer tile, while its
projection still executes `M16+M2`; a full 32-row tile projects as `M16+M16`.
Conv/GDN and QK/RoPE also remain bounded subtiles on one serial CUDA stream.
There is no double/triple buffering, multi-stream overlap, or dense-GEMM
crossover yet. Clocks were not locked, so the timing gains are diagnostic.
The next kernel priority is a true M17-M32 projection path; overlap work should
follow measured NCU stall evidence. Full commands, artifact hashes, ABI notes,
and limits are in the [C32 composite prefill record](metadata/qwen36-27b-c32-composite-prefill-benchmark.json).

## Production FP8 M32 projection

Commit `5c4845b` promotes the fixed-M32 FP8 WMMA candidate into public and
runtime dispatch. The four aligned production shapes `[10240,5120]`,
`[5120,6144]`, `[6144,5120]`, and `[12288,5120]` now validate the complete
32-row tile and execute one kernel. Other valid FP8 M32 shapes or alignments
fall back to two ordered public M16 calls; NVFP4 remains two M16 calls. M17
through M31, causal subtiles, the logical Prefill/Decode plan split, and the
single-stream execution policy are unchanged.

All four direct shapes are bit-exact to two public M16 launches across
1,081,344 BF16 outputs, deterministic on replay, preserve both output guards,
and capture exactly one CUDA kernel node each. The four instances use 46
registers/thread, 21,248 bytes static shared memory, and zero stack/local
memory. Generic, 4-byte-but-not-16-byte-aligned, and exact-shape near-miss M32
cases retain four-node ordered fallback graphs; NVFP4 retains two nodes;
cross-half alias and M33 failures capture zero nodes. Release passes 49 of 54
tests with five environment skips, and the selected ASan/UBSan suite passes 48
of 53 with the same five skips. Independent C1/C8/C16/C32 model processes all
retain the pinned 19 input IDs, 26 generated IDs, exact text/stop semantics,
and all 44 logical steps.

The same-cubin preproduction gate measures the four kernels at 1.51313x,
1.42923x, 1.50555x, and 1.50818x versus two production M16 calls. Weighting by
the P33 logical call mix gives 1.48217x. The detached-binary P33/C32 B-C-C-B
diagnostic then records:

| Arm | TTFT median |
| --- | ---: |
| Baseline 1 | 713.465 ms |
| Candidate 1 | 674.498 ms |
| Candidate 2 | 674.819 ms |
| Baseline 2 | 713.339 ms |

The mirrored-pair averages are 713.4020 and 674.6585 ms: a candidate delta of
-38.7435 ms (-5.4308%, 1.05743x). A matched P33/C32 Nsight comparison attributes
that movement directly:

| Metric | Composite baseline | FP8 M32 | Delta |
| --- | ---: | ---: | ---: |
| Kernel launches | 2,534 | 2,358 | -176 |
| Summed kernel time | 728.236864 ms | 688.875296 ms | -39.361568 ms |
| Kernel span | 736.294432 ms | 698.223712 ms | -38.070720 ms |
| Projection launches | 1,121 | 945 | -176 |
| Projection time | 651.548672 ms | 613.072480 ms | -38.476192 ms |
| FP8 M16/M32 tile launches | 352 M16 | 176 M32 | -176 |
| FP8 M16/M32 tile time | 124.776832 ms | 86.635360 ms | 1.44025x |

At that milestone NVFP4 was the clear next target: its unchanged 384 M16
launches consumed 392.846528 ms in the candidate profile. That result selected
the K64/LD72 versus K128/LD136 M32 gate completed in the following milestone.
Buffering and multi-stream work remained behind NCU stall evidence. Clocks
were not locked, so the timing values are diagnostic.
Full commands, identities, hashes, and limitations are in the
[FP8 M32 production record](metadata/qwen36-27b-fp8-m32-production-benchmark.json).

## Production NVFP4 M32 projection

Commit `2b98063` promotes the winning fixed-M32 NVFP4 WMMA candidate into the
public and runtime dispatch paths. Aligned `[17408,5120]` gate/up and
`[5120,17408]` down projections now validate the complete 32-row span and run
one K64/LD72 dual-resident-A kernel. Other valid NVFP4 M32 shapes or alignments
fall back to two ordered public M16 calls. M17 through M31, the logical
Prefill/Decode plan split, and the one-stream serial execution policy remain
unchanged; this milestone does not add double or triple buffering.

The production path is bit-exact to two public M16 launches across 720,896
BF16 outputs, deterministic on replay, exact at the token-15/16 boundary,
finite, guard-safe, and one CUDA graph node for each direct shape. Each direct
instance uses 47 registers/thread, 31,232 bytes static shared memory, zero
local/stack memory, and admits five active blocks per SM. Full-span aliases
fail before enqueue. Generic and minimally misaligned fallbacks retain their
expected ordered 4- or 32-node graphs. Release passes 49 of 54 tests with five
environment skips; the selected ASan/UBSan suite passes 48 of 53 with the same
five skips. Independent C1/C8/C16/C32 model processes preserve the pinned 19
prompt IDs, 26 generated IDs, exact text/`im_end`, and all 44 logical steps.

The same-cubin four-round B-C-C-B gate compares K64/LD72 dual-A and K128/LD136
single-A candidates across both exact shapes and checkpoint-like plus
same-bank-stress scales. Their production-call-weighted speedups over two M16
launches are 1.55193x and 1.42609x respectively, selecting K64/LD72. A detached
binary P33/C32 B-C-C-B diagnostic using the exact manifest user text records:

| Arm | TTFT median |
| --- | ---: |
| Baseline 1 | 674.828 ms |
| Candidate 1 | 530.884 ms |
| Candidate 2 | 530.805 ms |
| Baseline 2 | 674.508 ms |

The mirrored-pair averages are 674.6680 and 530.8445 ms: a candidate delta of
-143.8235 ms (-21.3177%, 1.27093x). Every run renders 33 prompt tokens, uses one
32-token prefix tile, and generates token 9419 (`Hello`). Matched P33/C32
Nsight profiles attribute the change directly:

| Metric | FP8-M32 baseline | NVFP4 M32 | Delta |
| --- | ---: | ---: | ---: |
| Kernel launches | 2,358 | 2,166 | -192 |
| Summed kernel time | 687.272192 ms | 545.447136 ms | -141.825056 ms |
| Kernel span | 694.953280 ms | 551.600992 ms | -143.352288 ms |
| Projection launches | 945 | 753 | -192 |
| Projection time | 611.334176 ms | 469.641312 ms | -141.692864 ms |
| NVFP4 M16/M32 tile launches | 384 M16 | 192 M32 | -192 |
| NVFP4 M16/M32 tile time | 393.042464 ms | 249.394720 ms | 1.57599x |

The target kernels alone save 143.647744 ms, while the complete kernel span
saves 143.352288 ms and mirrored TTFT saves 143.823500 ms. That agreement is
strong attribution evidence despite unlocked clocks. Both profiles still show
one kernel stream: Prefill and Decode are logically separated, but execution
is not yet overlapped. The next gate is replay-scoped NCU evidence for the
remaining P33/C32 hotspots; buffering or multi-stream work begins only if it
exposes independent work and a concrete scheduling gap. Full commands,
binary identities, hashes, excluded literal-label probes, and limitations are
in the [NVFP4 M32 production record](metadata/qwen36-27b-nvfp4-m32-production-benchmark.json).

## NVFP4 M32 K256 scale-window

Commit `9690129` keeps the production N128, K64/LD72, dual-resident-A kernel
but coalesces its block-scale traffic. Each output row now loads one contiguous
16-byte scale segment per K256 window into eight otherwise-unused BF16 words in
the shared-B padding `[64,72)`, then reuses that segment across four ordered K64
WMMA stages. The two exact `[17408,5120]` and `[5120,17408]` routes are
unchanged, accumulation order is unchanged, and no shared memory is added. The
candidate uses 46 registers/thread, 31,232 bytes static shared memory, zero
local/stack memory, 256 threads/block, and five active blocks/SM.

The direct gate is bit-exact to both the previous M32 production kernel and two
public M16 launches across 720,896 BF16 outputs. Replay, token-15/16 boundary,
finite-output, guard, two-byte-aligned-scale, and single-node CUDA Graph checks
all pass. Release reports 49 passes and five environment skips from 54 tests.
A later configuration audit found that the historical `build/orin-asan` run
reported in this section had no sanitizer compile or link flags, so it is only
additional non-sanitized evidence. The following factorized milestone reruns
the same 53-test selection with a verified ASan/UBSan build. Separate C1/C8/
C16/C32 model processes preserve the pinned 19 prompt IDs, 26 generated IDs,
exact text/`im_end`, and all 44 logical steps.

The same-cubin four-round B-C-C-B gate records:

| Shape | Scale distribution | Previous M32 | K256 window | Speedup |
| --- | --- | ---: | ---: | ---: |
| `[17408,5120]` gate/up | checkpoint-like | 1.19710 ms | 1.02512 ms | 1.16777x |
| `[17408,5120]` gate/up | same-bank stress | 1.16842 ms | 0.995491 ms | 1.17372x |
| `[5120,17408]` down | checkpoint-like | 1.34318 ms | 1.18348 ms | 1.13494x |
| `[5120,17408]` down | same-bank stress | 1.30996 ms | 1.14579 ms | 1.14328x |

Applying the 128:64 P33 gate/up-to-down call mix across both distributions
reduces the aggregate from 472.588 to 407.711 ms, or 1.15913x. Replay-scoped
NCU supplies the mechanism rather than relying on timing alone:

| Shape | Duration | Excess global sectors | Excess shared wavefronts | Memory / compute throughput |
| --- | ---: | ---: | ---: | ---: |
| Gate/up baseline | 1.25 ms | 1,218,560 | 4,513,852 | 63.60% / 30.94% |
| Gate/up K256 window | 1.03 ms | 174,080 | 3,670,912 | 71.00% / 38.94% |
| Down baseline | 1.39 ms | 1,218,560 | 4,441,924 | 56.80% / 27.86% |
| Down K256 window | 1.15 ms | 174,080 | 2,887,680 | 60.14% / 34.98% |

Thus each exact shape removes 1,044,480 excessive global sectors (85.714%)
without reducing achieved occupancy. The remaining source-counter pressure is
primarily shared lookup/WMMA access, not repeated global block-scale loads.

The detached-base P33/C32 B-C-C-B diagnostic moves mirrored-pair TTFT from
530.6365 to 499.0395 ms, a -31.5970 ms change (-5.9545%, 1.06332x), while all
runs generate `Hello`. Matched Nsight Systems profiles retain 2,166 launches;
target gate/up plus down time falls from 249.133952 to 217.687904 ms
(-31.446048 ms, 1.14446x), accounting for essentially the complete 30.586848
ms summed-kernel and 31.5970 ms TTFT reductions.

Three follow-up directions were rejected and removed. Full and shift-5 XOR
product-table swizzles improved only the synthetic same-bank distribution and
regressed checkpoint-like weighted performance to 0.98530x and 0.98649x.
Down-only N64 and N96 tiling reached only 0.98493x and 0.90586x aggregate,
because extra staging, barriers, padded work, and lost B/accumulator reuse cost
more than their larger grids recovered. The production route therefore remains
N128 with K256 scale windows. Prefill and Decode are still logically separate
plans on one serial CUDA stream; this milestone adds no double/triple buffering
or multi-stream overlap. The next kernel-local gate should reduce decoded-
product shared traffic or pipeline stages while preserving B reuse and dual-
accumulator ILP. Full artifacts and limitations are in the
[NVFP4 M32 scale-window record](metadata/qwen36-27b-nvfp4-m32-scale-window-benchmark.json).

## NVFP4 M32 factorized product lookup

Commit `51ca634` keeps the production N128, K64/LD72, dual-resident-A K256
scale-window kernel and changes only decoded-product construction. The previous
8,192-byte table materialized every `E4M3 scale x signed E2M1 value` BF16
product. The selected path stores 256 packed E2M1 byte pairs in 1,024 bytes,
stores all E4M3 scales in 512 bytes, and uses exact BF16x2 multiply after each
packed lookup. An exhaustive 256-scale x 256-packed-byte device gate compares
65,536 words and 131,072 BF16 halves, including signed zero and 1,024 NaN
halves per path, with zero mismatches.

Both exact shapes use 46 registers/thread, 24,576 bytes static shared memory,
zero local/stack memory, 256 threads/block, and five active blocks/SM. Relative
to the full table, static shared memory falls 6,656 bytes (21.31%). SASS falls
from 784 to 728 static instructions and from 49 to 35 LDS instructions; the
new path adds 16 `HFMA2.BF16_V2` instructions, leaves LDG/HMMA/barrier counts
unchanged, and introduces no local loads or stores. The default direct suite,
two exact-shape smoke tests, replay, token boundary, guards, invalid routes,
and one-node CUDA Graph contracts all pass. Release reports 49 passes and five
environment skips from 54 tests. A genuinely instrumented `build-sanitize`
(`-fsanitize=address,undefined`, linked to libasan/libubsan) reports 48 passes,
five skips, and zero failures from the selected 53 tests. C1/C8/C16/C32 model
processes preserve the exact 19 prompt IDs, 26 generated IDs, text, `im_end`,
and 44 logical steps.

The persisted same-cubin four-round B-C-C-B gate records:

| Shape | Scale distribution | K256 full table | Factorized | Speedup |
| --- | --- | ---: | ---: | ---: |
| `[17408,5120]` gate/up | checkpoint-like | 1.02615 ms | 0.924178 ms | 1.11033x |
| `[17408,5120]` gate/up | same-bank stress | 0.996987 ms | 0.926639 ms | 1.07592x |
| `[5120,17408]` down | checkpoint-like | 1.18354 ms | 1.08811 ms | 1.08770x |
| `[5120,17408]` down | same-bank stress | 1.14503 ms | 1.09019 ms | 1.05030x |

Applying the 128:64 P33 gate/up-to-down call mix across both distributions
reduces the weighted aggregate from 407.989648 to 376.316384 ms, or 1.08417x.
Exact shape-and-template-filtered NCU explains both the win and the remaining
gap:

| NCU fixture | Full-table duration | Factorized duration | Executed instructions | Excess shared wavefronts |
| --- | ---: | ---: | ---: | ---: |
| Gate/up checkpoint-like | 1.042176 ms | 0.930656 ms (1.11983x) | 32,916,432 -> 27,831,936 | 3,670,912 -> 5,709,824 |
| Down same-bank stress | 1.163552 ms | 1.100448 ms (1.05734x) | 32,385,552 -> 27,510,912 | 2,887,680 -> 5,730,880 |

Global-access counters are unchanged and achieved occupancy rises slightly.
The speedup comes from roughly 15% fewer executed instructions, fewer LDS
instructions, and the smaller footprint—not from fewer bank conflicts. Total
and excessive shared wavefronts increase, so bank-aware E2M1-pair layout and
recovering the former four `STS.128` decoded-tile stores are the next bounded
kernel-local targets. The down NCU row is explicitly same-bank stress; it is
not presented as checkpoint-like evidence.

An alignment audit also resolves an apparent cross-milestone global-sector
increase. The two rows above deliberately pass `cudaMalloc + 2` scales to
stress the public two-byte alignment contract, so alternating K256 windows
touch four/eight sectors and both specializations report 348,160 excess
sectors. Frozen scale-window and current `bool=false` SASS are identical. A
dedicated checkpoint-like gate/up replay with the cudaMalloc-aligned scale
buffer—matching the production resident arena's mandatory 256-byte tensor
alignment—reports 174,080 excess sectors for both `bool=false` and `bool=true`,
while the candidate remains 1.10576x faster (1.027168 to 0.928928 ms). Thus
there is no factorization-induced global-load regression; the two alignment
fixtures are retained as separate production and public-contract evidence.

The detached scale-window baseline and factorized candidate P33/C32 B-C-C-B
diagnostic records 499.087, 486.068, 486.056, and 499.070 ms. The mirrored
averages are 499.0785 and 486.0620 ms, a -13.0165 ms change (-2.6081%,
1.02678x), with all runs generating token 9419 (`Hello`). Matched Nsight
Systems profiles retain 2,166 launches and reduce target gate/up plus down time
from 217.672096 to 204.422208 ms (-13.249888 ms, 1.06482x). Summed kernel time
falls 11.457632 ms and kernel span falls 11.666848 ms; the small difference is
unlocked-clock non-target noise.

Prefill and Decode remain logically separate execution plans on one serial
CUDA stream. This milestone adds no double/triple buffering or multi-stream
overlap. Packed-weight prefetch variants regressed to 0.9817x and 0.93859x
weighted speedup and were removed. Protocols, hashes, NCU/Nsys artifacts,
limitations, and the sanitizer evidence correction are in the
[NVFP4 M32 factorized lookup record](metadata/qwen36-27b-nvfp4-m32-factorized-lookup-benchmark.json).

## NVFP4 M32 vectorized decoded stores

Commit `9280474` keeps the selected factorized K64/LD72 M32 kernel and changes
only how each thread writes its 32 decoded BF16 values into the shared WMMA
tile. Four groups of eight BF16 values are now packed as `uint4` and emitted
as four aligned `STS.128` instructions instead of sixteen scalar `STS`
instructions. Static SASS falls from 728 to 712 instructions. Both exact shapes
remain at 46 registers/thread, 24,576 bytes static shared memory, zero
local/stack memory, and five active blocks/SM; the 35 `LDS`, 16
`HFMA2.BF16_V2`, global-load, HMMA, and barrier instruction counts are
unchanged.

The default executable passes the exhaustive 65,536-word factorized-lookup
gate and compares 720,896 direct-shape BF16 outputs against two public M16
launches with zero candidate or replay mismatches. Token-15/16, finite-output,
guard, invalid-route, resource, and one-node CUDA Graph contracts pass, and the
public exact-shape route resolves to the vector-store specialization. Release
reports 49 passes and five environment skips from 54 tests. The verified
ASan/UBSan selection reports 48 passes, five skips, and zero failures from 53
tests. Separate C1/C8/C16/C32 model processes preserve the pinned 19 prompt
IDs, 26 generated IDs, exact text/`im_end`, and all 44 logical steps.

The same-cubin four-round B-C-C-B gate records:

| Shape | Scale distribution | Scalar stores | Vector stores | Speedup |
| --- | --- | ---: | ---: | ---: |
| `[17408,5120]` gate/up | checkpoint-like | 0.924128 ms | 0.754692 ms | 1.22451x |
| `[17408,5120]` gate/up | same-bank stress | 0.926465 ms | 0.757198 ms | 1.22354x |
| `[5120,17408]` down | checkpoint-like | 1.09238 ms | 0.913389 ms | 1.19597x |
| `[5120,17408]` down | same-bank stress | 1.09503 ms | 0.915340 ms | 1.19631x |

Applying the 128:64 P33 gate/up-to-down call mix across both distributions
reduces the weighted aggregate from 376.870 to 310.561 ms, or 1.21352x.
Exact-template replay-scoped NCU confirms the shared-store mechanism:

| NCU fixture | Duration | Executed instructions | Excess shared wavefronts |
| --- | ---: | ---: | ---: |
| Gate/up checkpoint-like | 0.932416 -> 0.763968 ms (1.22049x) | 27,831,936 -> 26,787,456 | 5,709,824 -> 1,531,904 |
| Down same-bank stress | 1.100224 -> 0.929728 ms (1.18338x) | 27,505,440 -> 26,460,960 | 5,730,880 -> 1,552,960 |

Each shape removes exactly 4,177,920 excessive shared wavefronts while ideal
shared wavefronts, global sectors, resources, and CTA residency remain
unchanged. These NCU rows retain the legal `cudaMalloc + 2` block-scale stress
fixture from the factorized milestone; the down row is deliberately
same-bank-stress rather than checkpoint-like evidence.

The detached factorized baseline and vector-store candidate P33/C32 B-C-C-B
diagnostic records 485.913, 454.215, 454.080, and 486.327 ms. Mirrored TTFT
averages fall from 486.1200 to 454.1475 ms (-31.9725 ms, -6.5771%, 1.07040x),
with all runs generating token 9419 (`Hello`). Matched Nsight Systems profiles
retain 2,166 launches and reduce target gate/up plus down time from 204.363040
to 172.409120 ms (-31.953920 ms, 1.18534x). Summed kernel time falls from
500.473024 to 468.811104 ms and kernel span from 509.762336 to 478.095424 ms,
so the target rows explain essentially the complete TTFT improvement.

Two bounded follow-ups widened the shared E4M3 scale table to U32 entries.
Direct indexing regressed the four-cell weighted aggregate to 0.99676x;
high-bit XOR bank indexing recovered the stress cells but reached only
1.00126x overall, below the 1.005x gate, with both checkpoint-like cells at or
below parity. Both candidates were removed. This exhausts the cheap U32
scale-table subpath, not every possible E2M1-pair layout.

The two bounded explicit-buffering follow-ups also failed decisively and were
removed:

| Candidate | Resources | Four-cell speedup range | Weighted speedup |
| --- | --- | ---: | ---: |
| Two-window scale ping-pong | 46 regs, 26,624 B shared, 0 local, 5 CTA/SM | 0.83829-0.90991x | 0.86449x |
| Activation-only `cp.async` two-panel pipeline | 46 regs, 29,184 B shared, 0 local, 5 CTA/SM | 0.70623-0.75350x | 0.72349x |

Scale ping-pong reduces the static barrier count from five to four, but grows
the kernel from 712 to 768 SASS instructions and makes every cell slower. The
activation candidate also grows to 768 instructions. It does generate four
`LDGSTS.E.64` instructions and removes the synchronous activation
`LDG.E.64`/`STS.64` pairs while retaining the five barriers, four decoded-tile
`STS.128`, 16 `HFMA2`, exact output, and five-CTA residency. Its two
`LDGDEPBAR`/`DEPBAR` pairs and pipeline control nevertheless dominate the
small activation panel's latency. Halving the copy count with a stronger
16-byte alignment contract cannot credibly recover the observed 28% weighted
gap, so that near-duplicate was not pursued.

Prefill and Decode remain logically separate plans on one serial CUDA stream.
With both low-footprint kernel-local buffering paths now measured, explicit
M32 buffering is evidence-exhausted for this route. The next gate is a matched
trace that quantifies dependency-independent execution and its theoretical
overlap ceiling before any multi-stream policy is implemented. Full binary
identities, protocols, hashes, raw profiler artifacts, rejected follow-ups,
and diagnostic limits are in the
[vector-store record](metadata/qwen36-27b-nvfp4-m32-vector-store-benchmark.json).

## C32 NVFP4 MLP gate/up dual-stream overlap

Commit `c58b797` is the first production execution-overlap milestone. It does
not assign separate streams to Prefill and Decode. Instead, it exploits the
one clean branch inside each exact 32-token MLP tile: after post-attention
normalization, gate runs on the existing main stream while up runs on one
owned nonblocking auxiliary stream. A reusable ready event publishes the
normalized activation, a reusable done event joins the branches, and SiLU,
down, residual, and the next layer remain ordered on the main stream.

Eligibility is deliberately narrow: the SM87 backend, exactly 32 tokens, two
aligned NVFP4 `[17408,5120]` projections, and distinct BF16 outputs. C1-C31,
other dtypes or shapes, misaligned inputs, Decode, and every fallback remain
serial. The auxiliary stream and two `cudaEventDisableTiming` events are
created once as best-effort resources. A partial allocation is destroyed and
its CUDA last-error is cleared, leaving a valid serial runner. Move, reset,
failure, and release paths explicitly transfer or drain both streams. The
request arena, production kernels, SASS, registers, and shared-memory
footprints do not change; this is neither double/triple buffering nor
Prefill/Decode overlap.

The final same-binary performance gate now calls the production tile
dispatcher rather than a test-only launcher. It includes the ready/wait/done/
join envelope and remains bit-exact:

| Scale distribution | Serial gate+up | Dual-stream envelope | Speedup |
| --- | ---: | ---: | ---: |
| Synthetic checkpoint-like | 1.77959 ms | 1.64649 ms | 1.08084x |
| Same-bank stress | 1.78546 ms | 1.65180 ms | 1.08092x |
| Two-distribution aggregate | 3.56504 ms | 3.29829 ms | 1.08088x |

Both cells and the aggregate clear the 1.03x scheduling gate with zero gate
or up mismatches. Two independent unprofiled P33/C32 B-C-C-B rounds also
clear the separate broader 1.01x end-to-end gate without a reversed round: their
combined mirrored mean moves from 474.06675 to 468.36000 ms (-5.70675 ms,
-1.2038%, 1.01218x). A separate exact-commit binary confirmation records
475.7985 to 466.3375 ms (-9.4610 ms, -1.9884%, 1.02029x), with every process
generating token 9419 (`Hello`).

The fixed prompt matrix shows that the absolute saving scales with complete
C32 prefix tiles and never reverses:

| Prompt | C32 prefix tiles | Serial TTFT | Dual-stream TTFT | Change |
| --- | ---: | ---: | ---: | ---: |
| P33 | 1 | 453.9335 ms | 446.8070 ms | -7.1265 ms (-1.5699%) |
| P65 | 2 | 812.2235 ms | 797.7420 ms | -14.4815 ms (-1.7829%) |
| P129 | 4 | 1,589.3165 ms | 1,562.1545 ms | -27.1620 ms (-1.7090%) |
| P513 | 16 | 7,608.5265 ms | 7,503.6680 ms | -104.8585 ms (-1.3782%) |

Matched P33 Nsight Systems evidence confirms real concurrency rather than two
nominal stream handles. All 64 gate/up pairs overlap across streams 17 and 18.
Their combined envelope falls from 107.754336 to 99.308160 ms (-8.446176 ms,
1.08505x), with 29.192032 ms of actual interval intersection. Contention
lengthens the raw target-kernel duration sum from 107.666976 to 128.500192 ms,
but that sum double-counts concurrent time. The all-inference kernel span
falls from 474.910176 to 471.603136 ms, while profiled TTFT moves from 476.042
to 473.018 ms. Raw duration growth is therefore a contention signal, not a
critical-path, power, or energy measurement.

Release reports 49 passes and five environment skips from 54 tests. The
verified ASan/UBSan suite reports 48 passes, five skips, and zero failures from
53 tests. Separate final C1/C8/C16/C32 model processes preserve the pinned 19
prompt IDs, 26 generated IDs, exact text/`im_end`, and all 44 logical steps.
The next bounded scheduling candidate is the lower-footprint Linear-Attention
A/B sidecar against QKV/Z; Decode and broad multi-stream scheduling remain
lower priority until their own gates pass. Full hashes, protocols, trace
definitions, limitations, and rollback thresholds are in the
[gate/up dual-stream record](metadata/qwen36-27b-nvfp4-m32-gate-up-dual-stream-benchmark.json).

## Rejected C32 Linear-Attention A/B sidecar

The follow-up remained test-only and used the production dispatchers at the
exact C32 checkpoint shapes: FP8 QKV `[10240,5120]`, FP8 Z `[6144,5120]`, and
the BF16 A/B pair `[48,5120]`. Serial and auxiliary-stream schedules produced
zero bit mismatches across all four outputs. Each timing included reusable
ready/done event waits and the final main-stream join, with ten warmups and
four 24-iteration B-C-C-B rounds.

Three scheduling variants establish the practical limit:

| Schedule | Serial | Concurrent envelope | Speedup | Result |
| --- | ---: | ---: | ---: | --- |
| A/B submitted between QKV and Z | 1.35208 ms | 1.32996 ms | 1.01663x | reject |
| Main-chain-first QKV/Z, run 1 | 1.35220 ms | 1.30479 ms | 1.03634x | marginal |
| Main-chain-first QKV/Z, run 2 | 1.35470 ms | 1.31523 ms | 1.03001x | marginal |
| QKV serial, then Z parallel with A/B | 1.35042 ms | 1.35591 ms | 0.99595x | reject |

The two main-chain-first repetitions combine to 1.03316x, but save only
0.04344 ms per Linear-Attention layer. Even multiplying that synthetic saving
across all 48 such layers gives 2.08512 ms, about 0.45% of the preceding
466.3375 ms P33 TTFT, before runner overhead and cross-layer effects. The
Z-only isolation also contains reversed passes and regresses in aggregate.
The large 1,536-CTA A/B grids contend with the fixed-M32 QKV/Z kernels for SM,
cache, and memory resources, so nominally independent work does not translate
into robust critical-path reduction on this device.

The candidate therefore did not enter `ReferenceRunner`; its optional test
code was removed, and the committed runtime remains unchanged. A third stream,
later join, and broader Prefill/Decode overlap are rejected at this stage
because they add control complexity without a measured envelope capable of
clearing the end-to-end promotion gate. The logical Prefill/Decode plan split
remains useful for independent phase benchmarks and kernel selection. Full
rounds, decision thresholds, and limitations are in the
[A/B sidecar rejection record](metadata/qwen36-27b-linear-ab-sidecar-rejection.json).
