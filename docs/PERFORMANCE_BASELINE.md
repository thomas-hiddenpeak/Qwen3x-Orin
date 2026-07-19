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
8. pair full-attention FP8 K/V only for aligned M1 `[1024,5120]` operands,
   retaining two ordered projections for every other valid case;
9. retain exact-token, numerical, replay, and memory gates for every dispatch
   change;
10. lock clocks when privileged access is available before making a formal
   release performance claim.

The small-M and bounded `C<=16` gates are complete. The next prefill work
is broader prompt/shape coverage and a measured dispatch boundary between the
current bounded tiles and future larger-prefill kernels. `.q3x` repacking
waits for a stable physical layout.
