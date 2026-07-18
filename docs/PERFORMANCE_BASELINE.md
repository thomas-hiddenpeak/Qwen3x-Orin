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
The next M=1 NVFP4 milestone is recorded in
[`qwen36-27b-nvfp4-packedx8-benchmark.json`](metadata/qwen36-27b-nvfp4-packedx8-benchmark.json),
and the subsequent FP8 milestone is recorded in
[`qwen36-27b-fp8-packedx4-benchmark.json`](metadata/qwen36-27b-fp8-packedx4-benchmark.json).
The first bounded multi-token prefill result is recorded in
[`qwen36-27b-c8-prefill-benchmark.json`](metadata/qwen36-27b-c8-prefill-benchmark.json).

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

## Correctness gate

The current `reference_engine_e2e` gate loaded the pinned 27B checkpoint with
the SM87 backend at both C1 and C8 and matched all 19 prompt IDs, 26 generated
IDs, exact UTF-8 text, `<|im_end|>`, and all 44 runner steps. The lightweight
kernel gate also compared SM87 directly with the CUDA reference for FP8 and
NVFP4 M=1 through M=8 awkward/fallback/vector shapes. It covers every one of
the 256 E4M3FN codes in all four positions of the packed FP8 load, including both reserved
NaN encodings; all 16 E2M1 codes in all eight packed nibble positions;
adjacent-lane scale selection; both unaligned FP8 fallbacks; and
K=5120/6144/17408. All 1,237 deterministic BF16 outputs matched the CUDA
reference bit-for-bit, while the independent host oracle checks the reserved
E4M3FN outputs by NaN class. The remaining 41 non-model CTests also pass.
Parallel reduction order is still not a general bitwise-equivalence promise,
so `reference` remains the default projection backend. The accelerated loader
is independent of that projection choice and is now the default when AF_ALG
is available. The complete packed-x4 fixed-oracle CTest passed in 40.60
seconds.

## Phase 3 decision

The first optimized path now includes SM87 single-token decode plus bounded
small-M prompt-prefix projection reuse:

1. keep `sm87` explicit while prompt coverage expands;
2. retain shape-driven packed-x8 dispatch for aligned canonical NVFP4 M=1;
3. retain shape-driven packed-x4 dispatch for aligned canonical FP8 M=1;
4. retain C1 as the compatibility default and expose measured C2 through C8
   prompt-prefix tiles explicitly;
5. retain exact-token, numerical, replay, and memory gates for every dispatch
   change;
6. lock clocks when privileged access is available before making a formal
   release performance claim.

The small-M and first chunked `C<=8` gates are complete. The next prefill work
is broader prompt/shape coverage and a measured dispatch boundary between the
current bounded tiles and future larger-prefill kernels. `.q3x` repacking
waits for a stable physical layout.
