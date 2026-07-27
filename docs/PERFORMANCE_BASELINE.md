# Qwen3.6 27B Phase 3 performance evidence

This document records the first kernel-level baseline and matched SM87
projection-backend comparison for the native Qwen3.6-27B-NVFP4 runner on
Jetson AGX Orin. These are diagnostic Phase 3 results, not serving-throughput
claims. The machine-readable records are
[`qwen36-27b-reference-nsys-baseline.json`](metadata/qwen36-27b-reference-nsys-baseline.json)
and
[`qwen36-27b-projection-backend-benchmark.json`](metadata/qwen36-27b-projection-backend-benchmark.json).
The current Phase 3 handoff and direct Prefill anchor are recorded in
[`qwen36-27b-current-head-prefill-baseline.json`](metadata/qwen36-27b-current-head-prefill-baseline.json)
and summarized in [Decode baseline lock and Prefill phase
handoff](#decode-baseline-lock-and-prefill-phase-handoff); earlier uses of
"current" remain historical to their individual milestones.
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
The rejected Decode M1 QKV/Z full-activation-staging follow-up is recorded in
[`qwen36-27b-fp8-m1-qkv-z-activation-staged-rejection.json`](metadata/qwen36-27b-fp8-m1-qkv-z-activation-staged-rejection.json).
The promoted Decode M1 QKV/Z reduction-scratch ping-pong follow-up is recorded
in
[`qwen36-27b-fp8-m1-qkv-z-reduction-scratch-ping-pong-benchmark.json`](metadata/qwen36-27b-fp8-m1-qkv-z-reduction-scratch-ping-pong-benchmark.json).
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
The promoted Decode M1 Q+K/V reduction-scratch ping-pong follow-up is recorded
in
[`qwen36-27b-fp8-m1-q-kv-reduction-scratch-ping-pong-benchmark.json`](metadata/qwen36-27b-fp8-m1-q-kv-reduction-scratch-ping-pong-benchmark.json).
The subsequent exact NVFP4 M1 down/residual/centered-RMSNorm cooperative fusion
diagnostic is recorded in
[`qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json`](metadata/qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json).
The subsequent canonical M1 GDN/plain-RMSNorm/SiLU-gate fusion diagnostic is
recorded in
[`qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json`](metadata/qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json).
The reduction-only warp-tail follow-up inside that fused GDN kernel is recorded
in
[`qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json`](metadata/qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json).
The promoted long-context GQA attention-score kernel and its separate Prefill
and Decode evidence are recorded in
[`qwen36-27b-prefill-attention-score-warp-positions-benchmark.json`](metadata/qwen36-27b-prefill-attention-score-warp-positions-benchmark.json).
The production promotion of the selected Decode gate/up and down streaming
loads is recorded in
[`qwen36-27b-decode-streaming-production-benchmark.json`](metadata/qwen36-27b-decode-streaming-production-benchmark.json).

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
14. use the eight-row lane-striped GDN update for M1 and bounded C2 through
    C15, select the exact register-resident state kernel for C16, and retain
    row8 as the same-binary C16 test predecessor plus the bounded fallback;
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

This paragraph is retained as the historical `e6fac6b` baseline. Commit
`09aa7f7` supersedes its M18 projection description only for the two exact
aligned NVFP4 MLP shapes; see the current M18 section below.

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

## Exact NVFP4 M18 masked-M32 prefill

Commit `09aa7f7` promotes a single-pass M18 projection for the exact aligned
NVFP4 MLP shapes `[17408,5120]` and `[5120,17408]`. The detached baseline is
`965ebb4`; its runtime is equivalent to `c58b797` because the intervening
Linear-Attention sidecar work was documentation-only after its test candidate
was removed. Other M18 shapes and insufficiently aligned operands preserve
the ordered public M16+M2 fallback.

The kernel uses an M32 WMMA tile internally but has an exact C18 external
capacity contract. It globally reads only input rows 0 through 17, zero-fills
the internal activation rows 18 through 31, and globally stores only output
rows 0 through 17. Exact-C18 allocations, output-tail canaries, replay, finite
checks, token-15/16 boundaries, invalid-route rejection, and graph-node counts
all pass. No caller-visible 32-row input or output padding is required.

The production API microbenchmark uses ten warmups, 24 measured iterations per
pass, and four B-C-C-B rounds. It compares the candidate with explicit public
M16+M2 calls across checkpoint-like and same-bank-stress scale distributions:

| Shape and distribution | M16+M2 | Masked M32 | Speedup |
| --- | ---: | ---: | ---: |
| `[17408,5120]` gate/up, checkpoint-like | 1.38387 ms | 0.746593 ms | 1.85358x |
| `[17408,5120]` gate/up, same-bank stress | 1.35469 ms | 0.749178 ms | 1.80824x |
| `[5120,17408]` down, checkpoint-like | 1.45716 ms | 0.907094 ms | 1.60640x |
| `[5120,17408]` down, same-bank stress | 1.42400 ms | 0.910023 ms | 1.56479x |

Applying the production 128:64 gate/up-to-down call mix across both
distributions reduces the weighted aggregate from 534.930 to 307.754 ms, or
1.73817x.

The P19/C32 detached-binary B-C-C-B run uses one warmup and five measured
maximum-one-token iterations per process:

| Process order | TTFT median |
| --- | ---: |
| Baseline 1 | 548.801 ms |
| Candidate 1 | 439.642 ms |
| Candidate 2 | 439.554 ms |
| Baseline 2 | 548.764 ms |

The mirrored means move from 548.7825 to 439.5980 ms, a -109.1845 ms
(-19.8958%, 1.2483735x) improvement with no reversed round. The non-timing
generation contract is unchanged.

Matched Nsight Systems profiles attribute the mechanism directly:

| Metric | M16+M2 baseline | Exact M18 | Change |
| --- | ---: | ---: | ---: |
| All kernel launches | 2,264 | 2,072 | -192 |
| Summed all-kernel time | 562.744480 ms | 454.835648 ms | -107.908832 ms |
| All-kernel span | 572.676640 ms | 460.716960 ms | -111.959680 ms |
| Gate/up projection | 128 M16 / 125.570368 ms | 128 M18 / 106.655584 ms | -18.914784 ms |
| Down projection | 64 M16 / 71.424832 ms | 64 M18 / 64.636960 ms | -6.787872 ms |
| M2 projection tails | 192 / 83.358688 ms | 0 | -192 launches / -83.358688 ms |
| Complete projection target | 384 / 280.353888 ms | 192 / 171.292544 ms | 1.636696x |

Release validation reports 49 passes, five environment skips, and zero
failures from 54 tests. The host C++ ASan/UBSan selection reports 48 passes,
five skips, and zero failures from 53 tests; this is not a sanitizer claim for
CUDA device code. Separate C1/C8/C16/C32 real-model processes preserve the
pinned 19 prompt IDs, 26 generated IDs, exact decoded text, `im_end`, and all
44 logical steps. Device `compute-sanitizer` could not run because this Orin
reports that GPU debugging features are disabled; that check is
platform-blocked, not passed. Exact-C18 allocation/canary coverage and the
full-model oracle are the available device-memory-safety evidence.

M18 gate and up remain ordered single kernels on the main stream. The route
does not use the C32-only gate/up auxiliary stream, does not introduce
double/triple buffering, and does not overlap Prefill with Decode. With exact
M18 promoted and broader scheduling candidates exhausted or deferred, the next
priority is a matched test-only gate for a Decode M1 NVFP4 factorized
scale-codebook candidate; production integration follows only if that gate
passes. Full binary identities, commands, hashes, profiler attribution, and
limits are in the
[M18 masked-M32 record](metadata/qwen36-27b-nvfp4-m18-masked-m32-benchmark.json).

## Rejected Decode M1 NVFP4 factorized lookup

The next bounded Decode probe tested a BF16 pair/scale factorization in
test-only clones of the raw down, gate/up-plus-SiLU, and post-attention
residual/norm/gate/up/SiLU M1 kernels. The working-tree base was `1607e76`,
whose production runtime is `09aa7f7`. The candidate never entered production
dispatch, and all candidate source was removed after the gate failed. A clean
rebuild from the exact base HEAD then passed the default test; its 36,900-byte
log has the same SHA-256 as the prior M18 default log.

The semantic and resource gates passed. An exhaustive BF16x2 product lookup
covered all 65,536 E2M1-pair/E4M3-scale combinations with zero word or half
mismatches and the same 1,024 NaN halves. The bounded gate/up, down, and fused
paths were bit-exact, including matching NaNs. Both exact full-shape fused
fixtures had zero mismatches, finite outputs, intact guards, and exact replay.
The factorized kernels retain zero local memory, 256 threads per block, and
four active CTAs/SM:

| Test-only kernel | Registers/thread | Static shared | Local | Active CTA/SM |
| --- | ---: | ---: | ---: | ---: |
| Raw down | 60 | 36,352 B | 0 B | 4 |
| Gate/up + SiLU | 61 | 11,776 B | 0 B | 4 |
| Residual/norm/gate/up + SiLU | 61 | 12,288 B | 0 B | 4 |

The selected performance gate compared the public production
residual/norm/gate/up/SiLU path with its exact test-only clone at
`[17408,5120]`. It used synthetic checkpoint-like and same-bank-stress data,
ten warmups, four B-C-C-B rounds, 64 launches per pass, and the median of the
eight per-launch pass averages:

| Synthetic distribution | Production | Factorized clone | Speedup | Latency change | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Checkpoint-like | 0.674431 ms | 0.744299 ms | 0.906129x | +10.36% | reject |
| Same-bank stress | 0.616738 ms | 0.670177 ms | 0.920262x | +8.66% | reject |

Every measured candidate pass was slower; neither cell reversed direction.
The required gates were 1.03x for the selected hotspot and 1.00x for each
cell, so the candidate failed decisively before an end-to-end run was
warranted. These fixtures did not use actual checkpoint tensor data. Raw down
was checked only for correctness and resources, not timing, so this result
must not be generalized into a raw-down performance claim.

Static SASS inspection gives a plausible implementation-level explanation,
not an NCU counter-based causal result. For the hottest fused path, FFMA and
FADD remain 101 and 76, while FMUL falls from 81 to 17 and LDS from 87 to 55.
However, the clone adds 32 HFMA2 instructions, grows PRMT from 3 to 77, and
IMAD from 217 to 276. The compact lookup's arithmetic savings appear to be
consumed by BF16 packing/unpacking, permutations, and integer addressing.
Separately, the four audited production SASS sections remain byte-normalized
identical; the hottest production section remains 2,693 normalized lines.

The next priority returns to Prefill: measure a general M17/M19-M31
masked-M32 dispatch, starting with M17/M19 boundary and dynamic-valid-count
gates before long-prompt evaluation. Broader multi-stream scheduling remains
deferred. Full rounds, hashes, compiler evidence, scope boundaries, and local
artifact identities are in the
[M1 factorized rejection record](metadata/qwen36-27b-nvfp4-m1-factorized-rejection.json).

## Runtime-masked NVFP4 M17/M19-M31 prefill tails

Commit `8b19d2aa27370f7daafb743bf024fbcfc8b25950` promotes the runtime-valid-count
masked-M32 kernel for M17 and M19 through M31 at the exact aligned NVFP4 MLP
shapes `[17408,5120]` and `[5120,17408]`. The public
`launch_sm87_nvfp4_w4a16_m17_m31_gemm_bf16_cuda` API accepts exactly M17 and
M19..M31 and owns only the exact M-row activation and output spans: the second
internal 16-row panel reads only valid rows, zero-fills rows M through 31, and
the epilogue stores only rows below M. It rejects M18, which continues to use
its fixed masked specialization and API; M32
continues to use its fixed factorized/vector-store specialization. Shape
near-misses or insufficient packed-weight, block-scale, or activation alignment
retain the public M16 plus at-most-M8 tail decomposition.

Production dispatch selects one kernel node per eligible projection. Pair
dispatch validates both complete tiles, natural input/output alignment,
cross-weight/output ranges, aliases, and overflows before its first enqueue,
then submits the two single-kernel projections in first-then-second order. If
either projection misses the direct gate, both retain the existing recursive
subtile path. The audited 4-byte-only packed-weight M24/M25 fallback boundaries
capture three/four nodes, and an M25 pair with either only the first or only
the second weight ineligible captures the established seven-node ordered
fallback. Odd pointers, aliases, wrapping ranges, and a malformed second
projection all fail with zero captured nodes, so the pair cannot be
half-enqueued by a deterministic host validation error.

The production-call-weighted per-M microbenchmark reports:

| Token count | Weighted speedup over M16 plus tail |
| ---: | ---: |
| M17 | 1.59909x |
| M19 | 2.22925x |
| M20 | 2.31321x |
| M21 | 2.84755x |
| M22 | 2.70354x |
| M23 | 3.10906x |
| M24 | 2.31787x |
| M25 | 2.70434x |
| M26 | 2.85538x |
| M27 | 3.32989x |
| M28 | 3.41481x |
| M29 | 3.94367x |
| M30 | 3.80147x |
| M31 | 4.19900x |

The end-to-end promotion uses ten tokenizer-pinned C32 prompts whose prefix
schedules exercise `17,19,24,25,31` alone and after one fixed M32 tile:
P18/P20/P25/P26/P32 and P50/P52/P57/P58/P64. The four independent process
contracts in B-C-C-B order are byte-identical, 10,210 bytes each, and share
SHA-256
`5b154c9af44dc2b7297eacf75e78664e4c56bf496d59111fce63d15357cc9028`.
All ten prompt cells improve in both candidate processes, with no reversal in
10/10 comparisons. The equally weighted ten-prompt aggregate is 1.349789x,
or a 25.914% TTFT reduction. The short-prompt endpoints improve from 16.711%
at P18 to 43.145% at P32; the long-prompt endpoints improve from 10.066% at
P50 to 32.994% at P64. This is a per-prompt equal-weight aggregate, not a
token-weighted serving-throughput result. The exact prompt text, IDs, hashes,
and schedules are frozen in the
[prefill-tail prompt manifest](../benchmarks/qwen36-27b-sm87-prefill-tail-prompts-v1.json).

Matched Nsight Systems captures isolate the mechanism at one single-tail,
one longer single-tail, and one fixed-M32-plus-tail schedule:

| Prompt | All kernels: baseline -> candidate | Kernel span: baseline -> candidate | Runtime-tail target: baseline -> candidate |
| --- | --- | --- | --- |
| P18 (`17`) | 2,247 / 536.938944 ms -> 2,055 / 449.431232 ms | 546.287424 -> 458.982528 ms | 384 / 257.606496 ms -> 192 / 170.093504 ms (1.514499x) |
| P26 (`25`) | 2,783 / 823.055936 ms -> 2,399 / 564.453184 ms | 833.204 -> 574.560160 ms | 576 / 429.425312 ms -> 192 / 171.027040 ms (2.510862x) |
| P64 (`32+31`) | 4,614 / 1,531.614336 ms -> 4,230 / 1,041.613344 ms | 1,514.018240 -> 1,026.293760 ms | 576 / 663.078592 ms -> 192 / 171.773312 ms (3.860196x) |

The P64 fixed-M32 portion remains 192 launches and moves only from 193.332160
to 193.260160 ms. The launch reduction and target-tail attribution therefore
come from replacing each M17/M19-M31 M16-plus-tail projection with one runtime
kernel, not from changing M32.

Release reports 49 passes, five environment skips, and zero failures; the
verified host C++ ASan/UBSan suite reports 48 passes, five skips, and zero
failures. Separate C1/C8/C16/C32 exact-model oracle processes pass. The four
fixed M18/M32 production SASS sections are unchanged and share normalized
SHA-256
`2574bd41a76f112e753f5784a97e38d54362394a2f2ab9c3b8ee32c6074bdf32`
between the frozen and candidate binaries.

Prefill and Decode remain logically separate host-control plans in one runner.
This milestone adds neither a general double/triple buffer nor Prefill/Decode
overlap. M17 through M31 gate/up projections remain ordered on the main stream;
only the pre-existing exact M32 gate/up route owns a narrow layer-local
auxiliary-stream overlap. These are unlocked-clock, batch-one local diagnostics
rather than a concurrent-request or serving-throughput claim. Full protocols,
binary/report hashes, raw profile attribution, correctness contracts, and
limitations are in the
[runtime-masked M17/M19-M31 record](metadata/qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json).

## Rejected M17-M32 NVFP4 MLP gate/up dual-stream generalization

Commit `8902f0e` retains a test-only same-binary probe for extending the exact
M32 gate/up auxiliary-stream topology across M17 through M32. The production
baseline is the `d22663b1...3bfe` application binary from that exact source
boundary: M17 through M31, including fixed M18, remain serial on the main
stream, while exact M32 keeps its already-promoted auxiliary-stream path. The
candidate was an uncommitted three-file selector-and-tests patch using the
existing event topology on that boundary. Static review of the selector and
ready/wait/done/join topology was clean, and its Release suite reported 49
passes, five environment skips, and zero failures. The patch was nevertheless
withdrawn after the performance gate.

The isolated probe used two synthetic scale distributions, ten warmups, four
serial-concurrent-concurrent-serial rounds, and 24 launches per pass. Every
cell required at least 1.03x, all four rounds in the positive direction, and a
minimum round speedup of 1.00x. All 32 correctness/cell gates, 128 rounds, and
16 per-M gates passed:

| M | Speedup | M | Speedup |
| ---: | ---: | ---: | ---: |
| 17 | 1.08188x | 25 | 1.08135x |
| 18 | 1.08124x | 26 | 1.08148x |
| 19 | 1.08152x | 27 | 1.08146x |
| 20 | 1.08166x | 28 | 1.08157x |
| 21 | 1.08157x | 29 | 1.08122x |
| 22 | 1.08170x | 30 | 1.08136x |
| 23 | 1.08148x | 31 | 1.08161x |
| 24 | 1.08154x | 32 | 1.08078x |

The observed round range is 1.08030x to 1.08240x, the cell range is 1.08074x
to 1.08216x, and the two-distribution aggregate moves from 56.6910 to 52.4207
ms (1.08146x). This is a synthetic scheduling ceiling, not a model result and
not evidence that all-count dual-stream dispatch should enter production.

The whole-model gate used four independent B-C-C-B processes, one warmup and
five maximum-one-token samples per prompt, C32, and twelve representative
prompts. It reused the ten-prompt tail manifest, then added the historical
P19 fixed-M18 prompt and an unchanged P33 fixed-M32 control. The byte-identical
non-timing contracts are 11,566 bytes each with SHA-256
`b808fc56760a1d5d863608c9c4d94ce0b806b6511f90b50e2d2c137f69bddc61`.
The matrix covers workload/schedule counts `17,18,19,24,25,31,32`; it is not
direct whole-model coverage of every M. The exhaustive per-M evidence above is
synthetic.

| Prompt | C32 schedule | Baseline mean | Candidate mean | Speedup |
| --- | --- | ---: | ---: | ---: |
| P18 | `17` | 434.6480 ms | 429.5075 ms | 1.011968x |
| P20 | `19` | 477.0205 ms | 471.8955 ms | 1.010860x |
| P25 | `24` | 503.4070 ms | 498.2450 ms | 1.010360x |
| P26 | `25` | 550.6840 ms | 545.9065 ms | 1.008751x |
| P32 | `31` | 647.6460 ms | 642.6815 ms | 1.007725x |
| P50 | `32+17` | 778.6705 ms | 773.2830 ms | 1.006967x |
| P52 | `32+19` | 821.7205 ms | 816.6415 ms | 1.006219x |
| P57 | `32+24` | 850.3060 ms | 845.2105 ms | 1.006029x |
| P58 | `32+25` | 898.0900 ms | 893.0785 ms | 1.005611x |
| P64 | `32+31` | 997.6640 ms | 992.6135 ms | 1.005088x |
| P19 | `18` | 439.5300 ms | 435.3570 ms | 1.009585x |
| P33 control | `32` | 446.9155 ms | 446.8830 ms | 1.000073x |

All eleven affected rows improve in both mirrored pairs. The unchanged P33
control improves by 0.0073% in mean; its second pair reverses by only 0.241 ms,
well inside the existing 0.5% non-trigger regression limit. At aggregate level,
the first and second mirrored pairs remain positive at 1.008136x and 1.005983x.
However, the combined mean moves only from 7,846.302 to 7,791.303 ms, saving
54.999 ms (0.7010%, 1.007059x). The affected-only result is 1.007484x. Both are
below the required 1.01x whole-model threshold, so the production candidate is
rejected despite its clean correctness result and positive microbenchmark.

No Nsight Systems profile was collected for the withdrawn candidate after this
hard gate failed. Stopping there avoids selectively appending candidate
attribution evidence after a negative promotion decision. Production therefore
retains serial M17-M31/M18 gate-up execution and the existing M32-only overlap.

After that decision, an independent Nsight Compute diagnostic profiled the
retained production runtime-mask kernel at its first representative M17
gate/up launch. The `<17408,5120,72>` kernel measured 751.552 us, 34.95% DRAM
read throughput, 45.03% SM throughput, 44.85% issue active, 19.11% tensor-pipe
active, 70.51% active warps, and 5.26 long-scoreboard stalled warps per active
issue. It used 48 registers per thread and 24,576 static shared bytes with five
active blocks per SM. These unlocked-clock counters do not prove a speedup, but
they show that the kernel is not saturating DRAM and justify one bounded
test-only experiment: prefetch one 4 KiB packed-weight stage while WMMA consumes
the already-decoded stage. This baseline diagnostic is not evidence for the
rejected dual-stream candidate. Its local report is 1,595,682 bytes with
SHA-256 `e99a80aa071a33d16dc200900faaae22b245f2a4c691b8f2bece467cc6bc26b2`;
the 44,011-byte log has SHA-256
`836a7237184a213dbfb0e97adb4c989d1bfe2b3e8f7c269fd88bc6a42d9986fd`.

That bounded 4 KiB raw-weight `cp.async` experiment is now measured and closed.
The test-only kernel uses one shared raw-weight slot: after waiting for a K64 x
N128 packed tile, each thread moves its aligned `uint4` into registers before
the next asynchronous copy reuses the slot. It is logical register/shared
double buffering, not a second shared slot or a three-buffer schedule. On the
exact `17408x5120` gate/up shape, all 28 synthetic cells and all 112 mirrored
rounds pass bitwise-output, canary, no-reversal, and 1.03x gates. The aggregate
moves from 21.0109 to 19.8148 ms (1.06036x); the 14 per-M results range from
1.05542x to 1.06570x.

The result does not generalize to down projection. All six screened
`5120x17408` cells at M17/M25/M31 remain correct but regress, ranging from
0.985482x to 0.997861x, and every cell contains a mirrored-round reversal.
Compiler resources remain within the five-CTA/SM gate: 48 registers per thread
are unchanged, while static shared memory rises from 24,576 to 28,672 bytes.

A matched unlocked-clock M17 Nsight Compute diagnostic measures 751.552 us for
the retained kernel and 706.912 us for the candidate (1.06315x). Long-scoreboard
stalls fall from 5.26 to 2.67 warps per active issue, SM throughput rises from
45.03% to 51.33%, issue active from 44.85% to 48.24%, and tensor-pipe active
from 19.11% to 20.23%. The candidate executes 87,040 `LDGSTS` instructions for
44,564,480 bytes with zero reported shared `LDGSTS` bank conflicts. Occupancy
remains five CTAs/SM and 83.33%. These counters support the isolated mechanism;
they are not a serving result and do not override the whole-model gate.

The formal whole-model gate used four independent B-C-C-B processes, C32, one
warmup and five maximum-one-token measurements for each of twelve prompts. All
four non-timing contracts are byte-identical at 11,566 bytes with SHA-256
`b808fc56760a1d5d863608c9c4d94ce0b806b6511f90b50e2d2c137f69bddc61`.
All ten affected prompts improve in both mirrored pairs; the fixed-M18 P19 and
fixed-M32 P33 controls stay within 0.035% pair regression. The first and second
aggregate pairs improve by 1.008139x and 1.007036x, but their combined result is
only 1.007587x (affected-only 1.008529x), below the required 1.01x production
threshold.

The temporary gate/up selector is therefore rejected and fully withdrawn.
Production retains serial runtime-mask gate/up execution for exact M17 and
M19-M31, serial fixed M18, the existing exact-M32 auxiliary-stream route, and
unchanged near-miss fallbacks. The retained test-only probe documents the local
kernel opportunity but is not selected by production. The reverted state
passes all 49 runnable CTest cases; the five skips are pre-existing.

Local raw-weight prefetch widening is no longer the next priority. The next
step is to quantify the full Prefill/Decode timeline and scheduler architecture
with representative end-to-end traces, separating unavoidable same-request
dependency from overlap available across requests, streams, copy engines, and
continuous-batching boundaries. Phase separation, double/triple buffering, or
Prefill/Decode overlap should be implemented only if that trace-backed ceiling
is material and its memory, fairness, correctness, and tail-latency gates are
explicit. All timings and counters here use unlocked clocks; the whole-model
runs are batch-one and one generated token. Full identities, hashes, rows,
thresholds, and claim limits are in the
[raw-weight cp.async rejection record](metadata/qwen36-27b-nvfp4-m17-m31-gate-up-raw-weight-cp-async-rejection.json);
the preceding rationale remains in the
[M17-M32 dual-stream rejection record](metadata/qwen36-27b-nvfp4-m17-m32-gate-up-dual-stream-rejection.json).

## Prefill/Decode phase-trace baseline

Commit `8323e6e` adds default-off, registered-string NVTX ranges for the whole
generation, Prefill tiles/finalization, and Decode steps, together with matching
benchmark phase timings. A final P33 NVTX-only capture resolves all five names
through registered `textId` values and observes exactly one generation, one
prefix tile, and one finish range. The main performance grid intentionally
retains one earlier frozen observer binary for all four cells so that its rows
remain directly comparable; the final registered capture validates the
committed instrumentation and does not replace the grid's P33 timing row.

CUDA kernels are attributed through their launching Runtime API inside the
same-thread NVTX range, then joined by correlation and process identity. `raw`
is the sum of kernel durations, `union` merges concurrent GPU intervals,
`span` is first-kernel start through last-kernel end, `idle = span - union`, and
`overlap = raw - union`:

| Workload | Phase shape | Host | Raw | Union | Span | Idle | Overlap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| P19, max26 | `18 + finish + 25 decode` | 3272.371 ms | 3243.683 ms | 3243.683 ms | 3271.041 ms | 27.358 ms (0.836%) | 0.000 ms |
| P33, max1 | `32 + finish` | 467.828 ms | 486.313 ms | 457.107 ms | 466.568 ms | 9.461 ms (2.028%) | 29.206 ms |
| P64, max1 | `32+31 + finish` | 1025.832 ms | 1040.272 ms | 1011.049 ms | 1024.331 ms | 13.282 ms (1.297%) | 29.223 ms |
| P513, max1 | `16x32 + finish` | 7562.280 ms | 7927.875 ms | 7461.361 ms | 7561.023 ms | 99.662 ms (1.318%) | 466.514 ms |

Range and kernel closure is exact. P19 contains 1/1/1/25
generation/prefix/finish/decode ranges and 12,997 kernels; P33 contains
1/1/1/0 and 2,166; P64 contains 1/2/1/0 and 4,230; P513 contains 1/16/1/0
and 42,709. Every generation kernel belongs to exactly one leaf range, with
zero missing, extra, or duplicate assignments, and leaf raw durations sum
exactly to generation raw duration in every row.

The profile redirects optimization toward phase-local compute rather than a
blanket buffering rewrite. P19 Decode raw time is led by NVFP4 gate/up (35.71%),
FP8 QKV/Z (21.25%), NVFP4 down (18.63%), and other FP8 projections. At P33,
M32 gate/up and down are 35.45% and 17.87% of prefix raw time. At P513, NVFP4
gate/up is 26.38%, attention scores 24.08%, NVFP4 down 13.31%, and GDN 7.59%.
The four GPU spans expose only 0.84% to 2.03% idle; this is not evidence for a
large same-request double/triple-buffer ceiling.

A matched event-enabled P33 calibration observes 128 event records and 128
waits: 64 edges from stream 17 to 18 and 64 back from 18 to 17, matching the
existing M32 ready/done handshake. It also changes generation host/raw/union
time by +0.621%/+0.843%/+0.899%, while prefix host time falls 0.284% and finish
host time rises 3.105%. The event trace therefore establishes topology, but its
single perturbed sample cannot prove microsecond-scale removable critical path
or justify deleting an ordering edge.

The final default-off B-C-C-B gate uses two prompts, one warmup, and five
measurements per prompt/process. All four 3,367-byte semantic contracts are
identical with SHA-256
`a10e79070a83fa20884dc557bb1ab00d31c549a36af1c39f9fe6093c98fef72a`.
Across all 20 samples per side, candidate deltas are -0.176260% TTFT,
-0.111676% total generation, and -0.092987% Decode; the largest mirrored
regression is only +0.063398% in pair-two Decode. These unlocked-clock changes
are noise, not a speedup claim, and show no material default-off regression.

Prefill and Decode are now logically separable and independently measurable,
but production still uses one dependency-serialized runner. There is no
general double/triple buffer, same-request phase overlap, continuous batching,
or serving scheduler; only exact M32 gate/up retains its narrow layer-local
auxiliary-stream overlap. The next priority is bounded long-context Prefill
attention/projection work and M1 Decode projection work under separate exact
gates. Multi-request phase scheduling should follow only with explicit KV
memory, fairness, TTFT, inter-token-latency, and tail-latency gates. These are
unlocked-clock, batch-one local diagnostics, not serving-throughput results.
All binary/report/log identities, raw rows, event boundaries, and limitations
are in the
[phase-trace baseline record](metadata/qwen36-27b-prefill-decode-phase-trace-baseline.json).

## Decode M1 down/residual/norm warp-tail rejection

The first post-trace Decode experiment tested the same shared-prefix/warp-zero
reduction tail that benefits the separate post-attention gate/up fusion inside
the exact `[5120,17408]` NVFP4 M1 down/residual/norm cooperative kernel. The
test-only shared tree and warp-tail instances are bitwise identical for finite
and two signed-nonfinite fixtures, reject all 42 invalid cases before enqueue,
and have identical 64-register, 35,904-byte-static-shared, zero-local, four-CTA
resources at a cooperative `64x256` launch.

The performance gate rejects it decisively. Five independent same-binary
B-C-C-B processes report baseline/candidate speedups from 0.972242x to
0.974540x (median 0.974232x; ratio of means 0.97378931x), so all five fail the
required 1.005x threshold. Matched six-pass NCU capture measures 325.312 versus
336.032 us (0.9680983x). Although the candidate removes 2,048 `BAR`, 5,056
`LDS`, and 2,496 `STS` opcode instances, barrier, long-scoreboard, and
short-scoreboard stall ratios rise 20.67%, 10.31%, and 14.72%, while issue
activity falls 2.74%. The serial shuffle dependency immediately before the
final synchronization offsets the reduced explicit barrier count.

No end-to-end run was warranted after the mandatory micro gate failed. All
test-only candidate code was removed, no probe or runtime/dispatch change is
retained, and production keeps the full shared-memory reduction tree. This
does not invalidate the earlier successful `[17408,5120]` post-attention
gate/up warp-tail kernel, whose shape and shared-memory layout differ. Exact
logs, hashes, NCU filter requirements, counters, and claim limits are in the
[down/residual/norm warp-tail rejection record](metadata/qwen36-27b-nvfp4-m1-down-residual-norm-warp-tail-rejection.json).

## Decode M1 FP8 QKV/Z activation-staging rejection

The next bounded Decode experiment tested whether the existing exact M1 FP8
QKV/Z two-phase kernel should stage its complete 5,120-element BF16 activation
once per CTA. The test-only candidate copied 1,280 coalesced 64-bit words
(10,240 bytes, five words per thread), shared the existing first barrier with
the E4M3FN codebook setup, and reused that shared activation across every row
quad assigned to the CTA in both ordered phases. The production row-quad FMA,
decode, reduction, scale, and BF16-RNE operation order was unchanged. The
public ABI, production launcher, and dispatch never selected the candidate.

Correctness and resources pass. Against the current fused kernel at the same
requested cap, all fourteen actual-checkpoint/same-bank cells have zero QKV/Z
BF16 mismatches, zero replay mismatches, intact output canaries, preserved
inputs, and finite outputs. Exhaustive finite E4M3FN byte positions and
isolated signed-weight-NaN classification also pass, as do the return-value
invalid contracts. The candidate uses 56 registers, exactly 11,392 bytes of
static shared memory, zero local memory, 256 threads, and four active CTAs per
SM; production uses 64 registers and 1,152 static-shared bytes at the same
four-CTA occupancy.

One same-binary process used the actual layer-0 checkpoint tensors, 10 warmups,
80 logical QKV/Z pairs per timed pass, five B-C-C-B rounds, and separate
checkpoint/stress measurements. Every cap regresses:

| Requested cap | Actual checkpoint | Same-bank stress |
| ---: | ---: | ---: |
| 384 | 0.925641x | 0.909042x |
| 512 | 0.923279x | 0.905295x |
| 768 | 0.920369x | 0.973138x |
| 1,024 | 0.920192x | 0.909874x |
| 1,280 | 0.916682x | 0.907831x |
| 1,536 | 0.913651x | 0.906337x |
| 2,048 | 0.912063x | 0.905724x |

The frozen production-topology cap-1,536 retest measures 0.483885 versus
0.529748 ms on actual checkpoint bytes, or 0.913426x against the mandatory
1.02x gate. Same-bank stress measures 0.441462 versus 0.487527 ms, or
0.905513x against the mandatory 1.00x non-regression gate. Thus the candidate
raises latency by 9.48% and 10.43% respectively; correctness and unchanged
occupancy do not rescue the failed production-value gate.

A production-only NCU baseline collected before implementing the candidate
helps bound the mechanism. The exact cap-1,536 kernel requests 125,829,120
global-load bytes: 83,886,080 bytes are the one-pass QKV/Z weights and
41,943,040 bytes are the logically repeated activation. NCU reports 41,761,920
L1-hit bytes and 84,067,200 L1-miss bytes. Under the explicit inference that
the streaming weights account for the misses, 99.568% of activation bytes
already hit L1; the remaining 181,120 miss bytes are only 0.432% of the
activation traffic. Thus full per-CTA staging cannot remove the weight-dominant
DRAM path.

The likely candidate mechanism is therefore fixed staging cost without useful
new locality: static shared memory grows by the activation's full 10,240 bytes,
each CTA must populate it, and all later activation reads become shared-memory
traffic. This remains an inference because no matched candidate NCU profile was
collected. Both mandatory micro gates fail decisively, so the stop-loss
intentionally excludes candidate NCU and end-to-end model runs; neither could
promote the candidate after the hard gate failure. The production-only report
is 1,649,414 bytes with SHA-256
`9496eee8ffb66656a0a782599cbf71d739c05e299382df96a1765b9bb7b8d928`.

All test-only code and hooks were removed. The kernel and test source blobs
match base HEAD `5db3fb0d474c0c539d4fc4a3453d34b1663b1670` exactly, the restored binary
passes the default device test, and production retains direct activation reads.
The complete local rejection log is
`/tmp/q3x-fp8-qkvz-activation-staged-run1.log` (101,940 bytes, 445 lines,
SHA-256
`02c91f88672e4455063b13220d031eb82fc28216b7564131196ee084cdca750d`).
Exact per-cap latencies, checkpoint offsets and independently verified payload
hashes, artifact identities, hardening omissions, and claim limits are in the
[QKV/Z activation-staging rejection record](metadata/qwen36-27b-fp8-m1-qkv-z-activation-staged-rejection.json).

## Long-context GQA attention-score warp-position promotion

Commit `87573f3` promotes an exact Q24/KV4/D256 attention-score kernel for
sequence lengths at least 65. Eight warps per CTA independently own eight
positions, preserving the reference reduction order while replacing repeated
block-wide shared-memory trees with warp shuffles. The public GQA ABI,
probability scratch, softmax, value kernel, streams, and buffers are unchanged;
S64 and all non-matching shapes keep the prior route. This is a phase-local
kernel change, not a double/triple-buffer or scheduler implementation.

The same public selector serves both phases. P513 max1 is the direct Prefill
promotion workload, while P64 max26 is the direct long-context Decode workload:
its first token completes at S64 and subsequent tokens use S>=65. P19 max26
never reaches S65 and therefore proves only short-context fallback stability,
not measured long-context Decode throughput.

The same-binary five-round B-C-C-B microbenchmark computes each row from the
arithmetic means of ten timed passes. It measures 9.065x at S65, 12.0287x at
S128, 14.4892x at S257, 16.3467x at S513, and 16.511x at S544. The full
S65-S513 chain improves 14.6162x. Exact finite-boundary, raw-BF16 special,
replay, graph, S64/S65 dispatch, resource, and downstream-identity gates pass.

Matched S513 NCU uses the pre-promotion test symbol; after normalizing only the
function name, its SASS is byte-identical to the final production kernel.
It preserves 12,607,488 global-load bytes, 196,992 global-load instructions,
and 3,151,872 FFMA thread instructions. Duration falls from
690.624 to 34.688 us; shared load/store instructions fall from
1,588,248/886,464 to zero, barrier stalls fall from 3.55 to zero warps per
active issue, and issue activity rises from 17.02% to 64.89%. Both reports
use an earlier same-source test cubin rather than the final gate binary; after
normalizing the function name, its candidate SASS is byte-identical to the
committed production kernel. Both reports warn that clocks were not fixed, so
these counters establish the mechanism, not release-grade absolute latency.

The same-workload P513 Nsight comparison uses the frozen phase-observer
precursor as baseline and the committed production runner as candidate; it is
cross-process structural attribution, while the B-C-C-B result below is the
promotion timing gate. It retains 7,184 score launches and all 42,709
generation kernels. Score time falls from 1,886.442 to 126.063 ms
(14.9643x), reducing its raw-kernel share from 23.80% to 2.05%. The independent
B-C-C-B whole-model gate moves the ten-sample P513 median from 7,504.189 to
5,732.909 ms: 1.308967x, or 23.60% lower TTFT. P64/P33/P19 max1 controls stay
within 0.06% pair regression.

The direct P64-max26 Decode gate moves the ten-sample total-generation median
from 3,371.574 to 3,334.155 ms (1.011223x). Pooling 210 subsequent-token
samples per binary moves the median from 112.779 to 111.204 ms (1.014163x);
the mean of the two process medians improves 1.015401x for Decode-after-first.
The combined total-generation and pooled subsequent-token results both clear
the 1.01 decision gate; the first total-generation pair alone is 1.009637x and
is not claimed as an independent 1.01x pass.
All four contracts are byte-identical and all four processes report zero
persistent memory drop. The P19-max26 fallback control stays within 0.056% and
also preserves its exact four-process contract.

One of the twelve B-C-C-B processes, main C2, reports a 105,021,440-byte
device-wide `cudaMemGetInfo` persistent-drop warning; the other eleven do not,
and the process returns `status=ok`. An independent production Nsys candidate
process reports 125,648,896 bytes, while the independent baseline phase
profile reports 18,292,736 bytes and stays below threshold. These observations
are retained as separate caveats and are not causally attributed to the
implementation. All 51 runnable tests in the 56-test suite pass, with five
dependency/model/tokenizer skips.

The production selector is retained. Since attention score is no longer the
leading P513 hotspot, the next priority is separately gated Prefill
projection/GDN work and Decode M1/long-context work. General buffering or
multi-request phase scheduling remains later architecture work with explicit
KV-memory, fairness, TTFT, inter-token, and tail-latency gates. Exact binaries,
raw rows, report hashes, contracts, warnings, and claim limits are in the
[warp-position benchmark record](metadata/qwen36-27b-prefill-attention-score-warp-positions-benchmark.json).

## Decode M1 NVFP4 full-product-table rejection

The first Decode projection screen after the attention promotion tested only
the exact `[17408,5120]` residual-add/centered-RMSNorm/gate/up/SiLU hotspot. A
test-only phase clone initialized a canonical 4,096-entry BF16 product table,
indexed as `table[scale_code * 16 + nvfp4_nibble]`, then replaced dynamic
scale/nibble multiplication with direct shared `uint16` lookup. The 8,192-byte
table reused the earlier 256-float norm scratch phase and coexisted with the
10,240-byte activation stage, for 18,432 bytes of static shared memory. The
candidate never entered the public selector, production dispatch, or headers.

Correctness and resource gates pass. The exhaustive table dump has zero
finite-bit mismatches across 4,064 finite entries and zero NaN-class
mismatches across 32 NaNs. Host arithmetic is not a raw-bit oracle for SM87
FMUL NaN canonicalization: the observed 16 sign and 32 payload differences are
diagnostic, while the authoritative bounded production-versus-candidate
full-path fixture is bitwise exact, including signed-NaN scale codes. The
checkpoint-like, same-bank, and signed-NaN bounded fixtures have zero residual,
gate, up, and replay mismatches, intact canaries, and preserved inputs. Exact
full-shape replay after a different poison also passes for both timed fixtures.
The candidate retains 64 registers, zero local memory, 256 threads, and four
active CTAs/SM, while static shared memory grows from production's 11,328 bytes
to 18,432 bytes.

The corrected screen uses synthetic data only (`actual_payload=false`), 10
warmups, 64 launches per pass, four B-C-C-B rounds, and the median of eight
per-launch pass averages. Every candidate pass is slower:

| Synthetic fixture | Production | Full table | Speedup | Candidate latency |
| --- | ---: | ---: | ---: | ---: |
| Checkpoint-like | 0.614472 ms | 0.935502 ms | 0.656836x | +52.244854% |
| Same-bank stress | 0.618422 ms | 1.005210 ms | 0.615217x | +62.544347% |

Both cells fail the required 1.00x non-regression gate, and the selected cell
also fails its 1.03x promotion gate. These corrected values supersede the
earlier preliminary fixture run; no actual checkpoint payload, model, TTFT,
Decode-throughput, NCU, or end-to-end claim is attached to this rejection.

Static SASS gives a bounded implementation hypothesis, not NCU causality. The
full candidate section grows from 1,344 to 1,520 instructions. Its hot
projection loop grows from 393 to 531 instructions: removing 64 `FMUL` and
eight `LDS` while leaving 64 `FFMA` and 12 global loads unchanged adds 100
`IMAD.SHL`, 61 `IMAD.U32`, and 38 `LOP3` instructions for table addressing,
index construction, and BF16 expansion. The candidate section's two `HFMA2`
instructions sit outside this hot loop in constant/control code; the product
projection loop itself has none and is not the earlier factorized `HFMA2`
chain. Production's normalized section hash is
unchanged between candidate and clean binaries. The failed hard gate therefore
triggered stop-loss before candidate NCU or end-to-end evaluation.

All candidate implementation, ABI, environment, and test hooks were removed.
The kernel and test sources match base HEAD `dcab3d3` byte-for-byte, the clean
Release target rebuilds, and the default device test passes. This stops the
full-product-table family unless materially new evidence changes its ceiling.
At that point the design gate selected Prefill GDN M16 shared-resident BF16
state as the next bounded screen, with Decode FP8 QKV/Z reduction-scratch
ping-pong as an independent quick screen. Both screens were subsequently
implemented and measured; the GDN result is closed below. Exact binaries,
mirrored rows, log hashes, SASS hashes, removal proof, and limitations for the
full-product-table screen are in the
[full-product-table rejection record](metadata/qwen36-27b-nvfp4-m1-full-product-table-rejection.json).

## Prefill GDN M16 shared-resident BF16 state rejection

The bounded Prefill GDN screen tested whether one value-head CTA should keep
its complete `[128,128]` recurrent BF16 state in 32,768 bytes of opt-in dynamic
shared memory across an M16 tile. Two test-only branches preserved the existing
four-row and eight-row lane-striped arithmetic. Both copied global state into
shared memory once, rounded every token update back to BF16 before the next
token, and copied the final state back once. Production row8 was the direct
baseline for both branches; neither candidate entered the public header,
production selector, runtime, or engine.

The first implementation exposed a CUDA translation-unit isolation hazard
before formal timing. Merely placing a 16-byte-aligned dynamic-shared candidate
in `gdn_decode.cu` padded every production GDN static-shared section by eight
bytes: row4 moved from 18,056 to 18,064 bytes and row8 from 34,568 to 34,576
bytes. That artifact was excluded. The candidates were moved into a CUDA source
linked only into the GDN device test, and the production source, resources, and
SASS were restored before measurement. Production row4 remains 40 registers,
18,056 static-shared bytes, zero dynamic/local bytes, and six active CTAs/SM;
production row8 remains 40 registers, 34,568 static-shared bytes, zero
dynamic/local bytes, and four active CTAs/SM. Their normalized SASS hashes are
`a2e442b6c6f376fff44cb6a11d7f75606f1ab39ee112c296229e817398b9477e`
and
`a91e5113444940c3781c6c92eed7572ea2c7b1486d21b41ba646580843eaba90`,
identical to base HEAD `8e85584`.

Correctness, contract, and resource gates all pass in the isolated binary.
For candidate token counts M1/M2/M8/M16, row4 and row8 are bitwise equal to
production for output and persistent state in both in-place and disjoint modes.
Disjoint input state and all other inputs remain unchanged; output tails and
canaries remain intact. M16 poison/replay and ordered C8+C8 versus C16 are exact.
Each valid candidate captures as one distinct `48x256` kernel node with 32,768
dynamic-shared bytes, while invalid paths return `cudaErrorInvalidValue` and
capture zero nodes. Row4 uses 40 registers, 18,064 static plus 32,768 dynamic
shared bytes, zero local memory, and clears the three-CTA/SM minimum. Row8 uses
40 registers, 34,576 static plus 32,768 dynamic shared bytes, zero local memory,
and clears the two-CTA/SM minimum.

The hard performance gate deliberately exceeds the 4 MiB L2 state footprint.
Each variant rotates equally across 24 independent state banks, or 36 MiB of
state, and fully resets all state/output banks before every pass. Each pass has
48 warmup and 480 measured M16 launches, giving every bank two warmups and 20
timed launches. Five same-binary B-C-C-B rounds produce ten per-launch pass
averages per side; promotion requires the candidate median to be at least
1.20x faster and the candidate mean to beat the baseline mean in every round.

| Branch versus production row8 | Production | Resident candidate | Speedup | Candidate latency |
| --- | ---: | ---: | ---: | ---: |
| Row4 resident | 0.390997 ms | 0.623302 ms | 0.627300x | +59.413499% |
| Row8 resident | 0.390995 ms | 0.519849 ms | 0.752132x | +32.955409% |

Both branches fail decisively: all ten candidate passes are slower for each
branch, no round favors either candidate, and the selected branch is `none`.
The final 24-bank state and last output remain bitwise identical after every
timed round, so the failure is performance-only. The mandatory micro gate
therefore triggered stop-loss before candidate NCU, Nsys, or end-to-end model
evaluation. Bank conflicts and lower residency are plausible design-level
explanations, not measured stall attribution.

All candidate implementation, ABI, tests, and build wiring were removed.
`CMakeLists.txt`, the production GDN CUDA source, and the GDN CUDA test match
base HEAD `8e85584` byte-for-byte. Production keeps its eight-row lane-striped
path. This closes direct row-major BF16 shared-resident state; GDN residency
should reopen only with a materially different bank-friendly packed layout or
new measured evidence. That condition was later met by the exact-M16 packed
register-state promotion recorded at the end of this document and in the
[register-resident benchmark record](metadata/qwen36-27b-gdn-m16-register-resident-bf16-state-benchmark.json).
The independently passed Decode FP8 QKV/Z
reduction-scratch ping-pong candidate was subsequently productionized and
validated below. The retained candidate binary is 983,976
bytes with SHA-256
`15b375eef5b4cfa17f5a150cb75ec7ec46d0667b632ac4f0fd3e7d5a8f0b63b6`;
the 6,215-byte log has SHA-256
`c69cd2d43a8c7b58e22b8805d4f109ee8abe84a83d0b5a2d101beed3bd9f08f4`.
Exact raw rounds, artifact identities, resource gates, removal proof, and
limitations are in the
[shared-resident rejection record](metadata/qwen36-27b-gdn-m16-shared-resident-bf16-state-rejection.json).

## Decode M1 FP8 QKV/Z reduction-scratch ping-pong promotion

The passed QKV/Z follow-up now replaces the production two-phase kernel's
single reduction-scratch slot and tail barrier with two CTA-local slots on the
M1 route used by Decode and final-prompt/finish-prefill. Shared
scratch is `float[2][4][8]`: adjacent row-quad bodies alternate slots, and the
slot sequence continues across the ordered QKV-to-Z phase boundary. The
producer barrier after lane-0 warp stores remains in every body; only the
trailing barrier after warp 0 consumes a body is removed. The retained barrier
in body `i+1` cannot release before warp 0 finishes body `i`, so body `i+2`
cannot reuse body `i`'s slot early. Weight/BF16 decode, FFMA order, reduction,
scale, BF16-RNE, the 1,536/768 QKV/Z topology, public ABI, validation, and
fallbacks remain unchanged.

This mechanism is deliberately narrow. It is a two-slot reduction-scratch
pipeline inside one CTA and one kernel launch, not a whole-runtime double
buffer, triple buffer, cross-kernel/stream schedule, or Prefill/Decode overlap.
The runtime already has logically distinct Prefill and Decode host-control
plans, but their execution remains dependency-serialized.

All promotion gates pass. The public launcher, cap-1536 performance launcher,
and direct ping-pong test hook capture the same function and launch
configuration; the tail-barrier predecessor is a distinct function with the
same topology. Invalid calls capture zero CUDA Graph nodes, and null resource
destinations are rejected. Exhaustive coverage includes all 254 finite E4M3FN
codes plus `0x7f`/`0xff` in all four packed byte positions for both QKV and Z.
Caps 512, 1,024, 1,536, and 4,096 have zero QKV/Z or replay mismatches, intact
canaries, and a dedicated race-signature fixture with one initial execution
plus one replay per cap; the cap-512 case also verifies input preservation.
Signed-NaN class/sign checks pass.

| Kernel | Registers | Static shared | Local | Active CTA/SM | SASS words / instructions | `BAR` / `FFMA` / `FADD` / `SHFL` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production two-slot ping-pong | 64 | 1,280 B | 0 B | 4 | 2,416 / 1,208 | 2 / 128 / 88 / 41 |
| Tail-barrier predecessor | 64 | 1,152 B | 0 B | 4 | 2,240 / 1,120 | 3 / 128 / 70 / 41 |

The normalized SASS hashes are
`46e3f218deb005bb3feea4e23dc57bde7aa3695a756beb27754bc9847a48b602`
and
`37be9574c5b6d3257820952e3624c21ea39ce8ba355539e198109481c96f36e3`.
The frozen base and post-promotion engines each contain 144 CUDA functions.
After excluding the two QKV/Z target symbols, the remaining 142 full mangled
function-name sets are identical, and every same-name function has the same
`(normalized encoding hash, word count)` pair: zero per-name mismatches. The two
target hashes and word counts are also identical across engines; only their
test/production roles and production dispatch swap. This rules out observed
unrelated kernel code-generation drift, while static SASS still does not
establish stall causality.

The formal same-binary test uses the actual layer-0 checkpoint payload, ten
warmups, 80 logical QKV/Z pairs per timed pass, and five B-C-C-B rounds at the
frozen cap 1,536. The tail-barrier predecessor is the baseline:

| Fixture | Tail predecessor | Production ping-pong | Speedup | Frozen gate |
| --- | ---: | ---: | ---: | ---: |
| Actual checkpoint | 0.482807 ms | 0.471458 ms | 1.02407x | >=1.01x PASS |
| Same-bank stress | 0.442070 ms | 0.439011 ms | 1.00697x | >=1.00x PASS |

The exploratory symmetric 1.01x stress threshold fails, but it is explicitly
not the selection policy. Both frozen production gates pass.

Separate-process whole-model validation compares a pre-promotion worktree based
on base HEAD with the post-promotion runner in B-C-C-B order. Both binaries
contain the A/B kernels; their production dispatch selects the old and promoted
roles respectively. Each process uses one warmup and five measured generations,
C32 requested/effective, and exact output contracts:

| Workload and metric | Base average | Production average | Speedup |
| --- | ---: | ---: | ---: |
| P19/max26 prompt prefill | 439.559 ms | 438.941 ms | 1.001407934x |
| P19/max26 Decode-after-first | 2,785.346 ms | 2,774.686 ms | 1.003841876x |
| P19/max26 subsequent token | 111.416 ms | 110.8645 ms | 1.004974541x |
| P19/max26 total | 3,224.8845 ms | 3,213.5545 ms | 1.003525691x |
| P64/max1 prefix | 885.947 ms | 886.205 ms | 0.999708871x |
| P64/max1 finish-prefill | 112.068 ms | 111.5585 ms | 1.004567111x |
| P64/max1 total | 998.027 ms | 997.759 ms | 1.000268602x |
| P513/max1 prefix | 5,626.5645 ms | 5,627.345 ms | 0.999861302x |
| P513/max1 finish-prefill | 113.812 ms | 113.190 ms | 1.005495185x |
| P513/max1 total | 5,741.0275 ms | 5,740.598 ms | 1.000074818x |

The frozen end-to-end gate allows at most 0.5% regression for every workload
and stage. Independent review finds the worst regression is prompt-prefix only:
0.0140% at P19, 0.0291% at P64, and 0.0139% at P513. Every stage gate passes.

All four P19 processes produce the same 26 token IDs and text, stop on
`im_end`, and execute 44 steps. All P64 and P513 processes produce token 9419
(`Hello`), stop at `max_new_tokens`, and execute 64 and 513 steps respectively.
One P19 candidate process reports a 285,569,024-byte device-wide
`cudaMemGetInfo` persistent-drop warning against a 67,108,864-byte tolerance,
but returns `status=ok` with valid exact output and timing rows. It is retained
as an external memory-watermark limitation, not attributed as evidence of a
causal code leak.

Compute Sanitizer racecheck was attempted for the first 12 promoted-kernel
launches, but this Orin/driver reports `GPU debugging features are disabled`
and exits 99. Its one summarized error is tool unavailability, not a reported
kernel hazard; racecheck is therefore neither passed nor failed. The promotion's
race evidence remains the explicit happens-before argument, four-cap
race-signature/replay fixtures, and Graph contracts.

The post-promotion Release build completes all targets. CTest discovers 56
tests, passes 51, skips one tokenizer plus four configured model fixtures, and
fails none. The real-model P19/P64/P513 runs above are separate completed
evidence; the five configured skips are not counted as model coverage.

Production keeps the two-slot kernel. The immediate main line remains
phase-local kernel and dispatch optimization through the already completed
logical Prefill/Decode plan seam. For batch-one single-request generation,
Prefill must causally produce the first token before Decode feedback begins,
and the measured phase trace does not expose a useful general-buffering idle
window. Independent executors/queues belong to future multi-request continuous
batching, after ownership, KV/state lifetime, fairness, TTFT, inter-token,
tail-latency, cancellation, and memory contracts are explicit. Double/triple
buffers should be evaluated only inside that cross-kernel/stream dependency
model. Frozen binaries, all 12 end-to-end log hashes, checkpoint offsets and
payload hashes, raw SASS identities, warning, and claim limits are in the
[reduction-scratch ping-pong benchmark record](metadata/qwen36-27b-fp8-m1-qkv-z-reduction-scratch-ping-pong-benchmark.json).

## Decode M1 FP8 Q+K/V reduction-scratch ping-pong promotion

The corresponding full-attention M1 route now also uses two reduction-scratch
slots. The exact aligned Q `[12288,5120]` plus K/V `[1024,5120]` fusion keeps
its `2048x256` topology, shared FP8 codebook, ordered Q then K/V execution,
FFMA/reduction/scaling/BF16-RNE arithmetic, public ABI, validation, and
fallbacks. Its scratch changes from `float[4][8]` to `float[2][4][8]`.
Consecutive Q row quads, and the Q-to-K/V handoff in blocks 1024 through 1535,
alternate slots. Every CTA executes at most two logical bodies, so no slot is
reused before kernel completion. The producer barrier remains in each body;
the post-consumer tail barrier is removed.

This is a narrow CTA-local mechanism. It does not allocate two runtime
workspaces, create a second stream, overlap kernels, or turn the
dependency-serialized single-request Prefill/Decode path into a system-level
double- or triple-buffered scheduler.

All correctness and dispatch gates pass. The public and direct promoted hooks
capture the same `2048x256` function, while the frozen tail-barrier predecessor
is distinct with the same topology. All 41 invalid cases are checked through
both launchers: 82 captures contain zero kernel nodes. Q, K, and V cover all
254 finite E4M3FN codes plus `0x7f` and `0xff` in every packed byte position.
The full-shape, poison/replay, ordered Q-row-quad and Q-to-K/V race signature,
canary, input-preservation, actual-checkpoint, same-bank, and signed-NaN gates
are exact. The NaN fixture classifies 24 outputs with correct class and sign.

| Kernel | Registers | Static shared | Local | Active CTA/SM | SASS words / instructions | `BAR` / `FFMA` / `FADD` / `SHFL` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Production two-slot ping-pong | 64 | 1,280 B | 0 B | 4 | 4,304 / 2,152 | 3 / 256 / 158 / 81 |
| Tail-barrier predecessor | 64 | 1,152 B | 0 B | 4 | 4,144 / 2,072 | 5 / 256 / 140 / 81 |

The exact raw `cuobjdump` instruction-line hashes are
`5333899d89c2d3bfcf48a4a584e1947ac3b396fb7084a49887390308f9f8abac`
and
`bd5a0af0bfb91743ba0c9db1c6888a02d22da9a8019bb308ca8ea8d9af2ae946`.
The predecessor hash is identical to the frozen base public kernel. To prevent
canonicalization ambiguity, the metadata separately records text-normalized
64-bit word hashes and hashes of the same words decoded as big-endian bytes.
The symmetric tail-public and ping-pong-public engines each contain 145 CUDA
functions; their full mangled-name sets, every per-function encoding hash, and
every word count match. Only the host public launch stub selects the opposite
Q+K/V role. Static SASS proves the intended mechanism and production isolation,
not runtime stall causality.

Five independent same-binary processes use ten warmups, 80 logical chains per
timed pass, and five B-C-C-B rounds per fixture. Promotion retains the frozen
1.01x actual-checkpoint and 1.00x same-bank gates:

| Fixture | Minimum speedup | Median speedup | Maximum speedup | Frozen gate |
| --- | ---: | ---: | ---: | ---: |
| Layer-3 actual checkpoint | 1.02169x | 1.02387x | 1.02576x | >=1.01x PASS |
| Same-bank stress | 1.01031x | 1.01259x | 1.01336x | >=1.00x PASS |

Separate-process whole-model validation uses symmetric engines in B-C-C-B
order, one warmup plus five measured generations per process, and requested
and effective C32. Every process reproduces the exact output contract.

| Workload and metric | Tail-public average | Ping-pong-public average | Speedup |
| --- | ---: | ---: | ---: |
| P19/max26 prompt prefill | 439.141 ms | 438.7775 ms | 1.000828438x |
| P19/max26 Decode-after-first | 2,773.517 ms | 2,766.751 ms | 1.002445468x |
| P19/max26 subsequent token | 110.939 ms | 110.669 ms | 1.002439708x |
| P19/max26 total | 3,212.649 ms | 3,205.5615 ms | 1.002211001x |
| P64/max1 prefix | 885.6755 ms | 886.049 ms | 0.999578466x |
| P64/max1 finish-prefill | 111.6305 ms | 111.451 ms | 1.001610573x |
| P64/max1 total | 997.299 ms | 997.492 ms | 0.999806515x |
| P513/max1 prefix | 5,623.407 ms | 5,625.5905 ms | 0.999611863x |
| P513/max1 finish-prefill | 113.3545 ms | 113.1135 ms | 1.002130603x |
| P513/max1 total | 5,736.764 ms | 5,738.704 ms | 0.999661945x |

The frozen end-to-end gate permits at most 0.5% regression at every workload
stage. The worst observed regression is the P64 prefix at 0.042171%; every
stage passes. P19 emits the same 26 IDs and text, stops on `im_end`, and takes
44 steps. P64 and P513 both emit token 9419 (`Hello`), stop at
`max_new_tokens`, and take 64 and 513 steps. All 12 processes report
`status=ok` and `persistent_drop_detected=0`.

The Release build passes, CTest discovers 56 tests, passes 51, skips five
configured fixtures, and fails none. A final rebuild retains the formal
artifact's GNU Build ID and loaded content; its full-file difference is only
four `.strtab` bytes from an nvcc temporary name, and its independent optional
retest remains positive at 1.02290x actual and 1.00596x stress.

Production keeps the two-slot Q+K/V kernel. The subsequent bounded FP8 M32
dual-resident-A screen is recorded below. The broader main line remains
phase-local kernel and dispatch work through the existing logical
Prefill/Decode seam. General double/triple buffering remains deferred to a
future explicit multi-request scheduler with ownership, KV/state/workspace
lifetime, handoff, fairness, TTFT, inter-token, tail-latency, cancellation, and
memory gates. Frozen binaries, all five micro logs, all 12 end-to-end log
hashes, checkpoint offsets and payload hashes, SASS identities, and claim
limits are in the
[Q+K/V reduction-scratch benchmark record](metadata/qwen36-27b-fp8-m1-q-kv-reduction-scratch-ping-pong-benchmark.json).

## Prefill FP8 M32 dual-resident-A promotion

The four exact FP8 M32 projections now keep both 16-token A panels resident
through each K64 stage. A decoded B fragment is loaded once and reused by two
independent accumulator chains; each chain preserves the predecessor K/MMA
order. The public validation, route registry, fallback, ABI, streams, and
scheduler are unchanged.

| Kernel | Registers | Static shared | Local | Active CTA/SM | Instructions | `BAR` | Dynamic barriers K5120 / K6144 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Fixed single-resident predecessor | 46 | 21,248 B | 0 B | 5 | 656 | 8 | 324 / 388 |
| Production dual-resident A | 47 | 23,552 B | 0 B | 5 | 632 | 4 | 162 / 194 |

All four production shapes pass public-versus-predecessor bitwise comparison,
different-poison direct replay, token-15/16, canary, and input-preservation
gates. The first shape covers all 256 E4M3FN raw codes in all four packed-byte
positions; `0x7f` and `0xff` produce all 256 expected signed NaN outputs with
no unexpected nonfinite result. Invalid shape, alias, and input-alignment
calls capture zero kernel nodes. After promotion, the public and direct dual-A
hooks capture the same function, while the preserved predecessor remains a
distinct function with the same launch topology.

Five independent same-binary processes use ten warmups, 80 launches per timed
pass, and five B-C-C-B rounds. The frozen gates require every shape to be
non-regressing and the `48:64:48:16` P33-weighted result to reach 1.03x:

| Shape N x K | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| 10240 x 5120 | 1.13158x | 1.13172x | 1.13287x |
| 5120 x 6144 | 1.16197x | 1.16853x | 1.17505x |
| 6144 x 5120 | 1.13904x | 1.13990x | 1.14078x |
| 12288 x 5120 | 1.13363x | 1.13465x | 1.13673x |
| P33-weighted | 1.14418x | 1.14569x | 1.14799x |

The symmetric predecessor-public and dual-A-public runners each contain 149
CUDA functions. Their complete mangled-name sets, instruction-word counts,
and every per-function encoding hash are identical; only the host public M32
role changes. Separate-process P33 and P513 B-C-C-B validation uses one warmup
and five measured generations per process:

| Workload and metric | Predecessor-public average | Dual-A-public average | Speedup |
| --- | ---: | ---: | ---: |
| P33 prefix | 335.7175 ms | 322.4050 ms | 1.041291x |
| P33 finish-prefill | 110.6360 ms | 110.7820 ms | 0.998682x |
| P33 TTFT / total | 446.3555 ms | 433.2275 ms | 1.030303x |
| P513 prefix | 5,627.4165 ms | 5,411.3180 ms | 1.039935x |
| P513 finish-prefill | 113.1935 ms | 113.2690 ms | 0.999333x |
| P513 TTFT / total | 5,740.5980 ms | 5,524.5655 ms | 1.039104x |

Every process emits token 9419 (`Hello`) with exact prompt and step contracts,
uses one P33 or sixteen P513 prefix-execution entries per sample, and reports
`persistent_drop_detected=0`. The averaged finish-prefill regressions are
0.132% and 0.067%; the worst paired-process regressions are 0.303% and 0.128%.
All remain below the frozen 0.5% stage limit, while overall TTFT falls by 2.94%
and 3.76%. Release builds all targets, and CTest discovers 56 tests, passes 51,
skips five configured fixtures, and fails none. Clocks were not locked, so the
ratios are diagnostic promotion evidence rather than a release-latency claim.
An exact-tree rebuild after commit retains both GNU Build IDs; the runner differs
from the symmetric dual-A binary only in four `.strtab` bytes from an nvcc
temporary filename, and the rebuilt device test independently replays at
1.14411x weighted. The frozen five-process screen predates a mechanical
internal-symbol neutralization; production code and the exact-tree replay use
the neutral symbol.

This promotion is shared-memory residency inside one kernel. It is not a
system double/triple buffer and does not overlap causally ordered batch-one
Prefill and Decode. The next step is refreshed phase-local attribution followed
by a bounded attention-values screen. Multi-request independent executors and
queues remain a future scheduler project with explicit request/KV/workspace
ownership, fairness, latency, cancellation, and memory contracts. Full binary,
SASS, micro, and end-to-end identities are in the
[FP8 M32 dual-resident-A benchmark record](metadata/qwen36-27b-fp8-m32-dual-resident-a-benchmark.json).

## Prefill FP8 M32 U16 codebook-swizzle rejection

A bounded follow-up tested two byte-bijective U16 codebook layouts inside the
four exact FP8 M32 dual-resident-A kernels. Mode 1 maps an FP8 byte `x` to
`x ^ (x >> 5)`; mode 2 maps it to `x ^ (x >> 5) ^ (x >> 6)`. Matching packed
U32 masks keep all four byte transforms independent. Both candidates were
test-only, and the production launcher continued to select the original
unswizzled codebook.

The Release device-test target build, default suite, resource, one-node Graph,
bitwise/replay, guard, input-preservation, token-15/16, exhaustive 256-code by
four-byte-position, and signed-NaN gates all pass. The first exhaustive shape
reports 0/327,680 candidate mismatches for both modes and both replays,
1,024/1,024 covered code positions, and 256/256 expected NaN outputs. Resources
are shape-invariant:

| Layout | Registers/thread | Static shared | Local | Active CTA/SM |
| --- | ---: | ---: | ---: | ---: |
| Production mode 0 | 47 | 23,552 B | 0 B | 5 |
| Test-only mode 1 | 48 | 23,552 B | 0 B | 5 |
| Test-only mode 2 | 48 | 23,552 B | 0 B | 5 |

Complete mode-0 instruction words match the frozen `115a068` comparator on
all four shapes. For `[10240,5120]`, static SASS grows from 632 instructions in
mode 0 to 648 in mode 1 and 672 in mode 2. `BAR`, `HMMA`, shared-load/store,
and global-load counts remain `4/4/48/15/12`, while `LOP3` grows
`115 -> 125 -> 134` and `SHF` grows `83 -> 92 -> 101`. This is compiled
mechanism evidence, not a dynamic attribution.

The formal fixed-frequency screen is one same-binary process with ten warmups,
80 launches per timed pass, and six rounds. Baseline launches bracket each
round; candidate order alternates `B-C2-C1-B` and `B-C1-C2-B`. Each shape uses
one complete hash-pinned real shard-1 weight tensor with the same deterministic
finite BF16 activation fixture, and the aggregate uses the `48:64:48:16` P33
call weights:

| Actual tensor shape N x K | Mode 1 speedup | Mode 1 minimum round | Mode 2 speedup | Mode 2 minimum round |
| --- | ---: | ---: | ---: | ---: |
| 10240 x 5120 | 1.01310x | 1.01280x | 0.992083x | 0.991781x |
| 5120 x 6144 | 0.994624x | 0.994048x | 0.986594x | 0.986075x |
| 6144 x 5120 | 0.994016x | 0.993277x | 0.982087x | 0.981719x |
| 12288 x 5120 | 1.01410x | 1.01292x | 0.991548x | 0.990612x |
| P33-weighted | 1.00308x | required 1.03x | 0.988108x | required 1.03x |

Mode 1 reverses in all six rounds of two shapes; mode 2 reverses in all 24
actual-tensor rounds. Both therefore fail the mandatory per-cell,
no-round-reversal, and weighted actual-checkpoint selection path. A synthetic
four-code same-bank stress fixture reaches 1.55792x and 1.52486x weighted, but
that deliberately adversarial fixture proves only that swizzling can remove
the constructed conflict. It is not representative production evidence and
does not override the real-payload failures.

A separate read-only source-level bank audit scans all 178,257,920 bytes of
the four pinned payloads. It models every N128 CTA, K64 stage, thread/pass,
four U32 words, and four byte lookups: 5,570,560 warp lookup instructions in
total, with exact-slot repeats collapsed as broadcasts. Mode 0 needs
18,301,967 total and 12,731,407 extra wavefronts. Mode 1 reduces those totals
by only 1.985% and 2.853%; mode 2 reduces them by 2.224% and 3.197%. All three
retain `p50=3`, `p90/p95/p99=4`, `p99.9=5`, and maximum 6. This static model
excludes codebook construction, WMMA shared accesses, and candidate ALU and
scheduling cost; it is not NCU. The weak unchanged-tail opportunity, together
with mode 2's additional shift/mask/XOR instructions, explains why the large
synthetic benefit does not transfer to the real tensors.

Both candidates are rejected and all source/test hooks are removed; production
remains the original dual-resident-A mode 0. The hard gate failed before a
five-process screen, P513 end-to-end run, or candidate NCU study. The next
forced rebuild and default device test pass, and all four rollback mode-0 SASS
word streams again match the frozen comparator. The subsequent bounded Decode
FP8 M1 `[5120,6144]` exact single-body/no-tail-barrier screen is now closed as
well. Its compiled SASS removes the generic control and third barrier without
changing resources, but all six fixed-clock actual-checkpoint rounds regress
and the paired median is 0.994914x. The candidate was removed before
synthetic, repeated-process, end-to-end, or profiling work. Codebook evidence
remains frozen in the
[FP8 M32 codebook-swizzle rejection record](metadata/qwen36-27b-fp8-m32-codebook-swizzle-rejection.json),
and the later Decode result is recorded in the
[attention O-projection no-tail rejection record](metadata/qwen36-27b-fp8-m1-attention-o-proj-no-tail-rejection.json).

## Decode NVFP4 M1 Gate/Up balanced-tail rejection

The first current-HEAD Decode screen targeted the fused exact-M1
`[17408,5120]` residual/norm/gate/up/SiLU kernel, which accounts for
35.788149% of P19/C32/max26 Decode kernel time. Production uses 64 CTAs and
eight warps per CTA. Its first eight row-quad rounds cover rows 0-16,383 on
the full grid; only the first 32 CTAs execute the ninth row-quad round that
covers the final 1,024 rows.

The test-only candidate left the first eight rounds unchanged and assigned one
two-row tail to every warp:

```text
tail_row0 = 16384 + 2 * (blockIdx.x * 8 + warp)
tail_row1 = tail_row0 + 1
```

Every warp therefore processed 34 rows instead of the production 36/32 split.
The row-pair retained the four accumulator chains per row, packed-column and
two-phase order, warp reduction, weight-scale multiply, BF16-RNE boundary,
residual/norm behavior, and SiLU order. Production dispatch and the public ABI
were unchanged throughout the screen.

Static resources were neutral:

| Exact M1 fused route | Registers/thread | Static shared | Local | Active CTA/SM |
| --- | ---: | ---: | ---: | ---: |
| Production | 64 | 11,328 B | 0 B | 4 |
| Balanced tail | 64 | 11,328 B | 0 B | 4 |

The actual checkpoint, same-bank stress, and signed Inf/NaN comparisons report
zero residual, final-gate, and up mismatches; canaries remain intact and all
inputs are preserved. The fixed-clock same-binary protocol used ten warmups,
64 launches per timed pass, five `B-C-C-B` rounds, and current production as
the direct baseline.

| Fixture | Paired speedup range | Paired median | Production pass median | Candidate pass median |
| --- | ---: | ---: | ---: | ---: |
| Actual layer-0 weights/scales | 0.950984x-0.951884x | 0.951273x | 0.622637 ms | 0.654414 ms |
| Same-bank stress | 0.954611x-0.955693x | 0.955264x | 0.626957 ms | 0.656310 ms |

All ten rounds fail the 1.00x non-regression requirement, and actual misses the
frozen 1.03x early gate. The result shows that the old half-grid tail already
retains sufficient resident warps to sustain its weight stream; distributing
the same bytes across twice as many CTAs adds row-pair/control overhead without
unlocking useful parallelism.

Stop-loss removed the candidate and all test hooks before a clean rebuild and
default SM87 device-test pass. Production behavior is unchanged. Full fixture
hashes, per-round measurements, artifact identities, rollback proof, and claim
limits are in the
[balanced-tail rejection record](metadata/qwen36-27b-nvfp4-m1-gate-up-balanced-tail-rejection.json).

## Decode NVFP4 M1 K256 packed-weight pipeline rejection

The next exact-M1 gate/up screen preserved the production 64-CTA row-quad
mapping and all arithmetic, but replaced each phase's four canonical global
`uint32` loads with a cooperative packed-weight stage. For every K256 tile,
loader lane `l` copied one aligned `uint4` selected by `row=l/8` and
`vector=l%8`. Each warp staged four 128-byte rows, so the single shared slot
was 512 bytes/warp and 4,096 bytes/CTA. After `cp.async` wait and a warp
synchronization, compute lane `c` loaded shared words `c`, `32+c`, `64+c`, and
`96+c` into registers. A second warp synchronization made the same-slot
overwrite safe, and the next asynchronous copy then overlapped the current
tile's FFMA chains. This is logical register/shared double buffering, not two
shared slots or executor-level concurrency.

Static gates pass. The candidate remains at the 64-register ceiling, adds only
the expected 4-KiB slot, allocates no local or stack memory, and preserves four
active CTAs/SM:

| Exact M1 fused route | Registers/thread | Static shared | Local | Active CTA/SM |
| --- | ---: | ---: | ---: | ---: |
| Production | 64 | 11,328 B | 0 B | 4 |
| K256 packed-weight pipeline | 64 | 15,424 B | 0 B | 4 |

The candidate SASS contains `LDGSTS.E.BYPASS.128`, `LDGDEPBAR`, and
`DEPBAR.LE`; wait precedes four cross-lane shared loads and slot overwrite
follows the second warp synchronization. There is no hot-loop CTA barrier and
no `LDL`/`STL`. The production Function remains byte-identical to its frozen
comparator at SHA-256
`4d8893b3e0d4328c4fc464cdb12563541466e121ede524edd9b386003ce5ab95`.
One important code-generation detail is retained for the follow-up: ptxas
outlines each source `__syncwarp()` through a `CALL.REL` to a shared
`WARPSYNC/RET` body.

The test-only direct ABI rejects either 4-byte-only packed-weight base with
`cudaErrorInvalidValue` and captures zero Graph nodes. Actual checkpoint and
same-bank stress fixtures both report 0/5,120 residual, 0/17,408 final-gate,
and 0/17,408 up mismatches, zero unexpected nonfinite values, intact canaries,
and preserved inputs.

The fixed-clock same-binary screen used ten warmups, 64 launches per timed
pass, and five `B-C-C-B` rounds:

| Fixture | Paired speedup range | Paired median | Production pass median | Candidate pass median |
| --- | ---: | ---: | ---: | ---: |
| Actual layer-0 weights/scales | 0.932326x-0.933307x | 0.932869x | 0.619377 ms | 0.663949 ms |
| Same-bank stress | 0.928641x-0.929438x | 0.928905x | 0.623363 ms | 0.671072 ms |

All ten rounds fail non-regression; the candidate raises pass latency by about
7.20% actual and 7.65% stress. The canonical four-row global accesses were
already coalesced, so the experiment does not reduce bytes or transactions;
it adds shared-memory round trips and synchronization. Stop-loss removes the
candidate and test hooks before a clean Release rebuild and default device-
test pass. Production dispatch and the public ABI remain unchanged.

The higher-register K512 form is skipped because halving synchronization
cannot credibly close a seven-percent gap and holding both phases' packed words
risks spill at 64 registers. The narrower SASS experiment replaced both
source synchronizations with volatile inline PTX
`bar.warp.sync 0xffffffff`, but ptxas emitted a Function byte-identical to the
measured candidate: SHA-256
`0c5ec4ced9ef546f6f176d3e298e057ac94b3c0ffdfcb818aa08f5232e7a21bf`
for both, including the same four `CALL.REL` sites and
`WARPSYNC R30; RET` helper. Static-equivalence stop-loss therefore skipped a
redundant correctness and performance rerun. That result is recorded in the
[inline warp-barrier static rejection](metadata/qwen36-27b-nvfp4-m1-gate-up-k256-inline-warp-barrier-static-rejection.json).

The next materially different candidate is a single-layer Decode-only AoSoA4
row-quad-interleaved weight sidecar. It makes the four row words one direct
`uint4` global load, removes shared staging and both warp barriers, keeps block
scales canonical, and leaves canonical weights serving Prefill and 4-byte
fallback. The full 64-layer gate/up layout costs 5.3125 GiB, so loader and
production integration remain forbidden until the single-layer screen reaches
at least 1.03x without a reversed round. Full K256 round data, binary/SASS
identities, fixture hashes, cleanup proof, and claim limits are in the
[K256 packed-weight pipeline rejection record](metadata/qwen36-27b-nvfp4-m1-gate-up-k256-packed-weight-pipeline-rejection.json).

## Decode NVFP4 M1 Gate/Up AoSoA4 sidecar rejection

The materially different follow-up changed the physical packed-weight layout
instead of staging canonical rows. For every packed `uint32` word, an AoSoA4
sidecar stores the same word from four consecutive rows in one `uint4`:

```text
dst[row_quad][word] = {row0[word], row1[word], row2[word], row3[word]}
```

The test-only exact-M1 kernel therefore replaces four independent 32-bit
weight loads per lane and phase with one aligned 128-bit load. Block scales
remain canonical, as do row mapping, scale XOR pairing, activation indexing,
FFMA order, reductions, BF16 boundaries, residual/norm staging, and SiLU. One
projection sidecar is 44,564,480 bytes; gate plus up is 85 MiB per layer and
5.3125 GiB for all 64 layers. The experiment allocates only the single-layer
pair and never changes the loader or production dispatcher.

Static resources return exactly to the production envelope:

| Exact M1 fused route | Registers/thread | Static shared | Local | Active CTA/SM |
| --- | ---: | ---: | ---: | ---: |
| Production canonical | 64 | 11,328 B | 0 B | 4 |
| AoSoA4 sidecar | 64 | 11,328 B | 0 B | 4 |

The unrolled projection hot loop contains two `LDG.E.128` sidecar weight
loads and zero 32-bit weight loads. Its four byte loads and three `PRMT`
instructions belong to unchanged canonical scale-code assembly; no weight
deinterleave is present. The loop has no `WARPSYNC`, CTA barrier, `LDL`, or
`STL`. Production SASS remains frozen at
`4d8893b3e0d4328c4fc464cdb12563541466e121ede524edd9b386003ce5ab95`.

Both actual and stress host sidecars pass a full inverse-layout `memcmp` before
upload. The direct ABI rejects either 4-byte-only sidecar base and captures
zero Graph nodes. Both fixtures then report zero residual/final/up mismatches,
zero unexpected nonfinite outputs, intact canaries, and preserved canonical
inputs and sidecars.

Ten warmups and five 64-launch `B-C-C-B` rounds give:

| Fixture | Paired speedup range | Paired median | Production pass median | Candidate pass median |
| --- | ---: | ---: | ---: | ---: |
| Actual layer-0 weights/scales | 0.994655x-0.995677x | 0.994987x | 0.612615 ms | 0.615702 ms |
| Same-bank stress | 1.00346x-1.00477x | 1.00353x | 0.616882 ms | 0.614712 ms |

The stress improvement is real but cannot override the hash-pinned actual
payload, whose five rounds all fail non-regression and whose median misses the
1.03x gate. Combining loads without reducing packed-weight bytes or sectors
does not improve the real workload. Stop-loss removes the kernel, sidecars,
and test hooks before a forced Release rebuild and default device-test pass.
It also forbids the 5.3125-GiB full-model allocation, scale interleave, loader
ownership, and offline-cache work.

This closes the current tail-scheduling, canonical `cp.async`, inline-barrier,
K512-extension, and packed-weight-only sidecar branches for gate/up. The next
priority is re-ranked from the remaining current-production Decode hotspots,
starting with FP8 linear-attention QKV/Z and fused NVFP4 down. Full round data,
layout hashes, static identities, artifacts, rollback proof, and limitations
are in the
[AoSoA4 sidecar rejection record](metadata/qwen36-27b-nvfp4-m1-gate-up-aosoa4-sidecar-rejection.json).

## Prefill exact-M32 NVFP4 down/residual epilogue-fusion rejection

The next bounded Prefill screen targeted the exact-M32 NVFP4 down projection
`[5120,17408]`, whose 1,024 launches expose 931.620448 ms, or 19.816929% of
the post-GDN-M16 prefix kernel union. Production rounds the down result to BF16
and then launches a standalone BF16 residual add. The test-only candidate kept
the production table-free Tensor Core body, decoded its rounded BF16 result in
the epilogue, loaded the residual, added in FP32, rounded again to BF16, and
wrote the residual output directly.

The finite smoke, replay, and Graph comparisons report 0/163,840 mismatches,
with intact guards, preserved inputs, one candidate Graph node, and zero nodes
for invalid calls. Production and candidate both use 23,552 bytes of static
shared memory, zero local memory, 256 threads, and five active CTAs/SM; the
candidate uses 47 registers/thread versus 48. Its projection body retains all
five barriers, four `HMMA`, 56 `PRMT`, 16 `HFMA2`, 19 shared loads/stores, and
16 global stores while adding 16 global residual loads and 16 `FADD`:

| Exact-M32 down SASS | Production projection | Fused candidate |
| --- | ---: | ---: |
| Instructions / ordered words | 712 / 1,424 | 960 / 1,920 |
| `LDG` / `FADD` | 7 / 0 | 23 / 16 |
| Registers/thread | 48 | 47 |
| Static shared / local | 23,552 B / 0 B | 23,552 B / 0 B |
| Active CTA/SM | 5 | 5 |

The authoritative screen uses the hash-pinned actual layer-0 `mlp.down_proj`
packed weights and scales with deterministic finite BF16 activation/residual
fixtures. At fixed 1.3005 GHz GPU and 3.2 GHz EMC, one process ran ten warmups
and six `B-C-C-B` rounds of 80 logical launches per timed pass:

| Round | Paired speedup | Per-round non-regression |
| ---: | ---: | --- |
| 1 | 1.00036x | pass |
| 2 | 1.00108x | pass |
| 3 | 1.00046x | pass |
| 4 | 1.00055x | pass |
| 5 | 1.00027x | pass |
| 6 | 1.00068x | pass |

The paired-round median is 1.00050x and the minimum is 1.00027x. Thus every
round is non-regressing, but the aggregate misses both the 1.005x technical
threshold and the 1.02x promotion threshold. Separately printed baseline and
candidate means are 0.904394 and 0.903883 ms (diagnostic ratio 1.00057x).

An independent audit then found that the candidate and its test baseline add
rounded raw-down on the left and residual on the right, whereas the real
Prefill call chain adds residual-left plus raw-down-right. The measured finite
fixture and performance workload remain valid for rejection, but NaN payload
propagation can depend on operand order. The test's 4/4 NaN bit match therefore
does not establish full runtime bitwise semantics. Because performance already
fails decisively relative to promotion, the candidate was removed rather than
patched and rerun. Stress, independent repeats, NCU, Nsys, and end-to-end work
were not run. Source and tests match `1de2e20`; a forced clean build/default
test passes and rollback SASS is byte-identical to the frozen base. The
relinked rollback ELF differs, so binary byte identity is not claimed.

The next bounded priority is a standalone per-token residual-add plus centered-
RMSNorm fusion shared by attention and MLP boundaries. It must preserve the
runtime's residual-left plus projection-right order and clear 1.15x on both
actual-like and stress fixtures before wider validation. Full commands,
payload offsets and hashes, per-pass timings, SASS/resource identities,
rollback proof, and claim boundaries are in the
[M32 down/residual rejection record](metadata/qwen36-27b-nvfp4-m32-down-residual-epilogue-fusion-rejection.json).

## Exact Q24/KV4/D256 attention-values promotion

The refreshed post-M32 P513 profile attributes 7,184 `attention_values`
launches and 206.459 ms of raw kernel time to this phase. That is 3.469% of
generation raw-kernel time, or a 3.695% host-span upper bound. The production
Q24/KV4/D256 path now uses a fixed-shape kernel with grid `(6,4,1)` and block
`(256,1,1)`. The grid exposes `query_within_kv` and `kv_head` directly, uses
32-bit indexes through the proven `S <= UINT_MAX/1024` bound, and unrolls four
positions while retaining one loop-carried accumulator. The generic kernel is
unchanged for every other shape and for sequences outside that bound.

The exact kernel preserves the predecessor's position order, `fmaf` operand
order, BF16 decode, and final BF16-RNE encode. It removes the dynamic 64-bit
division helper and most generic indexing. `cuobjdump` reports:

| Kernel | Registers | Static shared | Local | Active CTA/SM | Function-block SASS lines | Dynamic-division helper |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Generic predecessor | 40 | 0 B | 0 B | 6 | 376 | present |
| Exact production | 26 | 0 B | 0 B | 6 | 112 | absent |

Default tests cover 29 sequence boundaries through S544, all six-query GQA
mapping boundaries, dimension boundaries around 32/128/256, finite values,
signed zero, subnormals, infinities, signed NaNs, different poison patterns,
deterministic replay, guards, and input preservation. Invalid captures produce
zero Graph nodes. Positive Graph capture proves distinct predecessor/exact
functions, predecessor grid `(24,1,1)`, exact grid `(6,4,1)`, 256 threads, and
zero dynamic shared memory. After promotion, the P513 production Graph contains
the exact function and no predecessor value function. Selector tests freeze
S0/S1/Smax/Smax+1 plus near-shape decisions, and a Q12/KV4/D256 production
Graph contains the predecessor value function while excluding the exact one.

Five independent same-binary processes use ten warmups, 80 launches per timed
pass, five B-C-C-B rounds, and report-only selection. The aggregate chain
covers every S from 65 through 513. The cold guard rotates sixteen identical
S513 V banks: each bank is 1,050,624 bytes and the 16,809,984-byte read-only
working set is about four times Orin L2; probabilities stay hot to model the
softmax-to-value handoff.

| Direct fixture | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| Hot S65 | 1.30170x | 1.31206x | 1.31587x |
| Hot S128 | 1.38370x | 1.39595x | 1.40753x |
| Hot S257 | 1.44175x | 1.44647x | 1.44876x |
| Hot S513 | 1.49055x | 1.49090x | 1.49766x |
| Hot S544 | 1.49560x | 1.49916x | 1.50671x |
| Hot S65..513 chain | 1.39585x | 1.39745x | 1.39948x |
| Rotating-cold V, S513 | 1.79801x | 1.80853x | 1.80948x |

The additional S1/2/4/8/16/32/64 cells are positive in every final process;
S1 is launch-noise dominated and spans 1.01245x to 1.11607x. No sequence
threshold is therefore needed inside the exact-shape selector.

Symmetric predecessor-public and exact-public runners have identical
canonicalized device SASS (`7f9b9e8e512eb9c27f47215837e176deec56722fbe37c7a994719e2bb3b18132`);
only the host value dispatch differs. Separate-process B-C-C-B results use one
warmup and five measured generations for max1, and one warmup plus three
measured generations for P513/max8:

| Workload and metric | Predecessor average | Exact average | Speedup |
| --- | ---: | ---: | ---: |
| P33/max1 TTFT / total | 432.9695 ms | 433.0620 ms | 0.999786405x |
| P513/max1 prefix | 5,408.1410 ms | 5,342.3690 ms | 1.012311392x |
| P513/max1 finish-prefill | 113.2960 ms | 112.0940 ms | 1.010723143x |
| P513/max1 TTFT / total | 5,521.4435 ms | 5,454.4910 ms | 1.012274748x |
| P513/max8 TTFT | 5,523.2725 ms | 5,454.4005 ms | 1.012626869x |
| P513/max8 Decode-after-first | 792.7390 ms | 784.8970 ms | 1.009991120x |
| P513/max8 subsequent token | 113.2415 ms | 112.1235 ms | 1.009971148x |
| P513/max8 total | 6,316.0940 ms | 6,239.2395 ms | 1.012317928x |

P33's 0.0214% diagnostic regression is below the frozen 0.5% stage limit.
P513 max1 TTFT falls by 66.953 ms (1.213%); P513 max8 subsequent-token latency
falls by 0.987%, and total generation falls by 76.855 ms (1.217%). All twelve
processes reproduce their exact token IDs/text/stop/step contracts and report
`persistent_drop_detected=0`. Clocks were not locked, so these ratios are
promotion diagnostics rather than release-latency claims.

The final Release build completes every target. CTest discovers 56 tests,
passes 51, skips five configured fixtures, and fails none. This optimization
changes one phase-local kernel and its production selector. It does not add a
runtime double/triple buffer or cross-kernel streams: Prefill and Decode remain
logically separate plans but causally serialized for a batch-one request. The
next main-line step is a fresh production profile and bounded screen of the new
largest phase-local hotspot; multi-request overlap remains a scheduler project.
Full artifact hashes, five-process logs, SASS/resource identities, and B-C-C-B
rows are in the
[attention-values exact benchmark record](metadata/qwen36-27b-attention-values-exact-benchmark.json).

## Prefill BF16 M16 A/B projection-fused promotion

The fresh post-attention-values P513/C32/max1 profile exposed the linear-
attention BF16 `in_proj_a`/`in_proj_b` pair as the next bounded exact-shape
target. Sixteen prefix tiles across 48 linear-attention layers issue 1,536
M16 pair kernels; they consumed 229.592608 ms and were fully exposed in the
measured GPU union. The 48 finish-prefill M1 calls consumed another 0.917440
ms, but are intentionally outside this Prefill specialization.

The production `M16/N48/K5120` kernel maps one CTA to one output row and owns
both projections for all sixteen tokens. It loads the two row weights once,
retains 32 independent FP32 accumulator chains, and reuses every decoded
activation across A and B. Shared memory is `float[2][16][256]`, so each token
and projection retains the predecessor's column order, `fmaf` operands,
256-thread reduction tree, and final BF16 round-to-nearest-even conversion.
The public-call count is unchanged, while the CTA grid per M16 call falls from
`(48,16,2)` to `(48,1,1)`.

The selector is deliberately narrow. M1-M15 remain one generic pair kernel,
M16 selects one exact kernel, M17-M31 run an exact M16 prefix followed by a
generic tail, and M32 runs two ordered exact kernels. Full validation of both
projections and all cross-ranges precedes recursive enqueue. Other backends,
weight kinds, and shapes preserve their existing fallbacks; Decode M1 is
unchanged.

Two exact candidates were screened in the same test binary. The row-resident
candidate reached 2.83063x hot and 2.98126x rotating-cold. The selected
projection-fused candidate reached 4.48082x and 3.41249x, respectively. The
losing row-resident kernel and all temporary test aliases were then removed.
Final `cuobjdump` and runtime-resource results are:

| Kernel | Grid | Registers | Static shared | Local | Active CTA/SM | Static instructions | `LDL` / `STL` / `CALL` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| Generic predecessor | `(48,16,2)` | 22 | 1,024 B | 0 B | 6 | 144 | 0 / 0 / 0 |
| Exact projection-fused | `(48,1,1)` | 55 | 32,768 B | 0 B | 4 | 1,520 | 0 / 0 / 0 |

The exact function is intentionally fully unrolled: its larger static block
performs the work of 32 predecessor CTAs per output row. It contains 32 static
`FFMA` chains, 18 `LDG` instructions, and no spill or out-of-line helper call.
The predecessor/exact kernels each have one SASS function and resource entry
in both the final candidate runner and reference GEMV device-test binary; no
BF16 M16 row-resident or projection-fused `_test` kernel alias remains.

Default tests compare every output bit against the predecessor and cover
finite structured inputs, signed zero, subnormals, infinities, signaling and
quiet NaNs, exact cancellation, different poison replays, head/tail canaries,
and input plus both weight buffers. Invalid low-level calls capture zero Graph
nodes. Production-dispatch tests freeze function identity, order, grid, block,
and zero dynamic shared memory for M1, M15, M16, M17, and M32, including the
fail-before-enqueue cross-subtile and invalid-second-projection cases.

Five independent same-binary processes use 64 hot warmups, sixteen cold-bank
first touches, 512 launches per timed pass, and four B-C-C-B rounds. Each cold
bank contains both weights and the input, so the 18,350,080-byte rotating
working set is 4.375 times the reported 4 MiB L2.

| Direct fixture | Minimum | Median | Mean | Maximum |
| --- | ---: | ---: | ---: | ---: |
| Hot M16/N48/K5120 | 4.48049x | 4.48613x | 4.48788x | 4.49640x |
| Rotating 16-bank cold | 3.40114x | 3.41783x | 3.42279x | 3.44638x |

Every mirrored round is non-regressing and clears the frozen 1.35x gate.
Separate-process B-C-C-B generation compares the frozen `c7cfdde` runner with
the final candidate. Max1 uses one warmup and five measured rounds per process;
P513/max8 uses one warmup and three measured rounds:

| Workload and metric | Predecessor average | Projection-fused average | Speedup |
| --- | ---: | ---: | ---: |
| P33/max1 prefix | 321.9895 ms | 312.0575 ms | 1.031827x |
| P33/max1 finish-prefill | 110.6190 ms | 110.5925 ms | 1.000240x |
| P33/max1 TTFT / total | 432.6075 ms | 422.6400 ms | 1.023584x |
| P513/max1 prefix | 5,343.0075 ms | 5,184.0450 ms | 1.030664x |
| P513/max1 finish-prefill | 111.9990 ms | 111.9915 ms | 1.000067x |
| P513/max1 TTFT / total | 5,454.9925 ms | 5,296.0260 ms | 1.030016x |
| P513/max8 TTFT | 5,457.7485 ms | 5,299.2295 ms | 1.029914x |
| P513/max8 Decode-after-first | 784.4325 ms | 784.6645 ms | 0.999704x |
| P513/max8 subsequent token | 112.0755 ms | 112.0995 ms | 0.999786x |
| P513/max8 total | 6,242.5510 ms | 6,083.9260 ms | 1.026073x |

Thus P33 and P513 max1 TTFT fall by 9.9675 ms (2.304%) and 158.9665
ms (2.914%). P513/max8 preserves the complete eight-token output while
Decode-after-first and subsequent-token latency remain neutral within 0.03%,
as expected from the unchanged M1 route. One first-baseline max8 process saw
a 215,093,248-byte global-free-memory decrease. Both candidate processes, the
second baseline, every max1 process, and an immediate same-command baseline
audit were clean; the isolated baseline-only reading is retained as an
environmental warning.

The production P513 profile confirms the mechanism rather than relying only
on end-to-end timing:

| Profile metric | Predecessor | Projection-fused | Change |
| --- | ---: | ---: | ---: |
| Target Prefix launches | 1,536 | 1,536 | 0 |
| Target Prefix raw time | 229.592608 ms | 66.677600 ms | -162.915008 ms, 3.44332x |
| All generation kernel launches | 42,709 | 42,709 | 0 |
| All generation raw time | 5,893.497216 ms | 5,729.023008 ms | -164.474208 ms |
| All generation kernel union | 5,425.624032 ms | 5,261.599744 ms | -164.024288 ms |
| Generation NVTX range | 5,528.166304 ms | 5,366.616384 ms | -161.549920 ms, 1.03010x |

The final Release build completes every target. CTest discovers 56 tests,
passes 51, skips five configured external fixtures, and fails none. Independent
actual-checkpoint chunk 1/8/16/32 runs each preserve all 19 prompt IDs, 26
generated IDs, exact text/`im_end`, and 44 steps.

The refreshed critical-path ranking is now led by the already heavily tuned
NVFP4 M32 gate/up route (1,589.152 ms marginal union), NVFP4 M32 down
(1,037.387 ms), GDN update (590.966 ms), and the largest FP8 M32 rows
(390.640/382.470 ms). The next gate therefore starts with a mechanism audit:
the rejected M32 scale-window ping-pong and activation-only `cp.async` paths
stay closed, and NVFP4 is revisited only for a materially new mechanism. If
that audit finds none, the next bounded implementation target is GDN.

This promotion adds neither a system double/triple buffer nor Prefill/Decode
overlap. Prefill and Decode remain logically separated plans, but one batch-one
request is still causally dependency-serialized. Full hashes, per-process
rows, SASS inventory, profiler artifacts, and limitations are in the
[BF16 M16 projection-fused benchmark record](metadata/qwen36-27b-bf16-m16-projection-fused-benchmark.json).

## Prefill NVFP4 M32 table-free E2M1 promotion

The post-BF16 profile left exact-M32 NVFP4 MLP projections as the dominant
bounded Prefill path: gate/up contributed 1,589.152 ms of marginal kernel
union and down another 1,037.387 ms in P513/C32/max1. The retained factorized
kernel decoded E2M1 through a 1,024-byte shared pair table plus a 512-byte E4M3
scale table. Its E2M1 lookup generated sixteen static `LDS` operations in the
stage loop, making shared lookup pressure a concrete mechanism candidate when
gate and up ran on separate streams.

The promoted specialization constructs each eight-value E2M1 BF16 vector with
register `PRMT` operations and keeps only the 512-byte E4M3 scale table. The
existing BF16x2 multiply, WMMA sequence, accumulation order, and vectorized
decoded-weight stores are unchanged. Code 8 remains exact BF16 negative zero.
Dispatch changes only aligned exact M32 for `17408x5120` and `5120x17408`;
M18, M17-M31, Decode M1, and fallback routes remain on their prior kernels.

Exhaustive validation covers `256 scale codes x 256 packed bytes x 4 byte
positions = 262,144` combinations with zero word or half mismatch. Both
production shapes then compare 720,896 BF16 outputs, poison replay,
token-15/16 boundaries, guards, single-kernel Graph capture, and zero-node
invalid calls. The two-stream A/B additionally compares gate and up
independently under both checkpoint-like and same-bank-stress scales.

Final `cuobjdump` and runtime resources are:

| Metric | Factorized-vector predecessor | Table-free E2M1 |
| --- | ---: | ---: |
| Registers/thread | 46 | 48 |
| Static shared | 24,576 B | 23,552 B |
| Local/stack | 0 B | 0 B |
| Active CTA/SM | 5 | 5 |
| Static instructions | 712 | 712 |
| `LDS` | 35 | 19 |
| `PRMT` | 16 | 56 |
| `HFMA2.BF16_V2` | 16 | 16 |
| `BAR` | 5 | 5 |

All sixteen BF16 multiply instructions retain `-RZ.H0_H0`; `HMMA`, `LDG`,
`STS.128`, and barrier counts are unchanged, and there is no `LDL` or `STL`.
The preserved bool-false specialization is encoding-identical to its
pre-change function. Nsight Compute counters are unavailable because the
target vGPU denies this user performance-counter access, so no counter-derived
bank-conflict claim is made.

Two independent final-binary processes use ten warmups, 24 logical launches
per timed pass, and four B-C-C-B rounds for each scale distribution. One
logical launch is one kernel for the single-kernel gate and one joined pair for
the pair gate:

| Same-cubin gate | Run 1 | Run 2 | Required |
| --- | ---: | ---: | ---: |
| Profile-weighted single-kernel speedup | 1.04790x | 1.04841x | 1.02x |
| Absolute concurrent gate/up-pair speedup | 1.20403x | 1.21929x | 1.02x |
| Minimum pair cell speedup | 1.19966x | 1.21726x | 1.00x |
| Minimum pair round speedup | 1.18629x | 1.20634x | 1.00x |
| Current serial-to-concurrent retention | 1.06266x | 1.04450x | 1.01x |

The absolute pair gate compares the old and new kernels with the same cubin,
inputs and workload, equivalent output buffers, streams, ready/done events,
and complete joined envelope. It is the production promotion metric. The
separate current-kernel serial-to-concurrent retention diagnostic remains
positive in both final processes, so the M32 auxiliary stream stays enabled.
A retained consecutive hot-state calibration reached 1.01850x aggregate,
1.01737x minimum cell, and 1.01413x minimum round with all eight rounds
positive. M32 therefore uses a 1.01 per-cell and aggregate threshold plus an
all-round non-reversal gate;
the unpromoted M17-M31 scheduling generalization keeps its 1.03 candidate
threshold and excludes the M32 control from its aggregate.

Detached-base whole-model B-C-C-B fixes the predecessor at `41f1c5f`, uses one
warmup round and five measured rounds per prompt (two prompts, ten measured
generations per max1 process), and preserves every token, text, stop, step, and
persistent-memory contract:

| Workload and metric | Predecessor average | Table-free average | Speedup |
| --- | ---: | ---: | ---: |
| P33/max1 TTFT | 422.6490 ms | 401.6080 ms | 1.052392x |
| P513/max1 prefix | 5,184.7810 ms | 4,846.3925 ms | 1.069823x |
| P513/max1 finish-prefill | 111.8940 ms | 111.8965 ms | 0.999978x |
| P513/max1 TTFT | 5,296.7645 ms | 4,958.3225 ms | 1.068257x |
| P513/max8 TTFT | 5,298.4240 ms | 5,048.3965 ms | 1.049526x |
| P513/max8 Decode-after-first | 782.3915 ms | 782.8130 ms | 0.999462x |
| P513/max8 subsequent token | 111.7710 ms | 111.8275 ms | 0.999495x |
| P513/max8 total | 6,080.7485 ms | 5,831.1780 ms | 1.042799x |

P33 and P513 max1 TTFT fall by 21.041 ms and 338.442 ms with no mirrored
reversal. Max8 Decode-after-first moves by only +0.054%, inside the frozen 0.5%
neutrality band, while the one-time Prefill saving remains. Chunk 1/8/16/32
model oracles all reproduce 19 prompt IDs, 26 generated IDs, exact text,
`im_end`, and 44 steps.

The matched P513/C32/max1 Nsys timeline closes the causal attribution:

| Profile metric | Predecessor | Table-free | Change |
| --- | ---: | ---: | ---: |
| Generation NVTX | 5,355.389728 ms | 5,079.708704 ms | -275.681024 ms |
| All-kernel union | 5,258.089248 ms | 4,978.907136 ms | -279.182112 ms |
| Gate/up pair envelope/marginal | 1,588.556192 ms | 1,410.349120 ms | -178.207072 ms, 1.126357x |
| Down raw/union/marginal | 1,036.605824 ms | 932.155840 ms | -104.449984 ms, 1.112052x |
| Exact-M32 total marginal | 2,625.162016 ms | 2,342.504960 ms | -282.657056 ms, 1.120664x |

Both profiles contain 42,709 kernel launches. Gate/up remains exactly 1,024
connected components with two launches each; its overlap structure does not
degrade. Other-kernel union plus generation gaps pay back 6.976032 ms, so
`282.657056 - 6.976032 = 275.681024 ms` exactly explains the generation delta.
Repeated B-C-C-B remains the primary performance result because tracing
perturbs the candidate more strongly than the predecessor.

The Release suite discovers 56 tests and fails none; five configured external
fixtures skip. The actually instrumented host ASan/UBSan build links
`libasan.so.8` and `libubsan.so.1`, discovers 55 tests after excluding the
package consumer, and fails none with five configured skips. This promotion
does not introduce system double/triple buffering or Prefill/Decode overlap.

The bounded exact-M32 pair-fused CTA follow-up is now measured and closed. All
three variants were test-only, used the promoted dual-stream pair as their
same-cubin joined-envelope baseline, and left the production selector, public
ABI, stream/event topology, and output contracts unchanged:

| Pair-fused variant | Registers / shared / local | Residency | Result | Decision |
| --- | ---: | ---: | ---: | --- |
| 256-thread, one decoded-B slot | 64 / 25,600 B / 0 B | 4 CTA/SM | 1.01383x aggregate; 1.00310x minimum round | Rejected below the 1.02x aggregate gate |
| 256-thread, two decoded-B slots | 64 / 41,984 B / 0 B | 3 CTA/SM | Performance not run | Rejected below the required 4 CTA/SM resource gate |
| 512-thread, two decoded-B slots | 56 / 41,984 B / 0 B | 2 CTA/SM, 32 warps/SM | 0.899017x aggregate; 0.894528x minimum round | Rejected; both cells and all eight rounds regress |

The single-B result is directionally positive but only one unlocked-clock
process and remains below its predeclared threshold, so the gate was not
lowered after measurement. The 512-thread route is bit-exact for both 557,056-
element outputs under both synthetic distributions, but its larger CTA loses
about 10% against the existing two-stream envelope. The two-slot 256-thread
route stops before timing because its 41,984-byte footprint admits only three
CTAs/SM. All candidate and harness code was removed, and the rebuilt default
SM87 weight-only test passes with no pair-fused symbols remaining. No NCU,
Nsys, or end-to-end run is claimed after the mandatory micro gate failed.

The subsequent exact-M32 down dual-A/dual-decoded-B screen is also closed. Its
synchronous ping-pong schedule merges the current-consumed and next-published
barriers, reducing the modeled dynamic CTA barriers from 614 to 343 across the
68 K256 windows without changing K order, WMMA order, or BF16 boundaries.
Correctness, replay, token15/16, guards, valid/invalid Graph, and resources all
pass:

| Down route | Registers | Shared | Local | Active CTA / warps per SM |
| --- | ---: | ---: | ---: | ---: |
| Production table-free | 48 | 23,552 B | 0 B | 5 / 40 |
| Dual-A/dual-B candidate | 56 | 46,592 B | 0 B | 3 / 24 |

The occupancy cost dominates the 44.1% barrier reduction. Checkpoint-like
scales measure 0.873096x with a 0.872322x minimum round; same-bank stress
measures 0.872016x with a 0.871715x minimum round. The 0.872556x aggregate is
below the frozen 1.03x gate, and all eight mirrored rounds regress. The
candidate and its harness were removed, the rebuilt default test passes, and
no NCU, Nsys, or end-to-end work is claimed after this mandatory gate failure.

The production table-free kernels and existing M32 gate/up dual stream
therefore remained selected at that point. The next bounded mechanism selected
for measurement was Prefill GDN M16 state held in a static per-thread BF16
register partition across the complete C16 chain, with zero local spill, at
least three CTAs/SM, exact per-token BF16 state boundaries, and a 1.20x early
micro gate. That mechanism subsequently passed and is recorded below. These
remain intra-kernel experiments, not general system double/triple buffering or
multi-request Prefill/Decode overlap.
Full hashes, per-process rows, SASS identities, profiling summaries, and
limitations are in the [table-free E2M1 benchmark record](metadata/qwen36-27b-nvfp4-m32-table-free-e2m1-benchmark.json).

## Prefill GDN exact-M16 register-resident BF16 state promotion

The materially different follow-up to the rejected shared-resident layout now
owns one complete C16 recurrence per value-head CTA. Each of 256 threads loads
64 BF16 state elements into 32 packed U32 words, rounds every token update back
to BF16x2, and writes the state only after token 15. Warp 0 and warp 1 retain the
exact Q and K normalization trees, while thread 64 produces the recurrence
scalars. The public selector uses this `48x256` kernel only for C16; M1 and
C2-C15 retain their existing routes. No stream, event, persistent workspace, or
public API changes.

One complete 48-head state is 1,572,864 bytes. Production row8 previously read
and wrote that state at all 16 token boundaries, or 48 MiB per C16 call. The new
kernel performs one initial read and one final write, or 3 MiB, removing 45 MiB
(93.75%) without removing any per-token BF16 rounding boundary.

The default device test passes exact output and final state in both in-place and
disjoint modes, disjoint-input preservation, replay/poison, guarded canaries,
C8+C8 versus C16 recurrence, stale-last-error isolation, and valid/invalid Graph
contracts. Public C16 captures the exact function as one `48x256` node; public
C15 remains the row8 function. Null, aliasing, and invalid-epsilon calls capture
zero nodes. The C1/C8/C16/C32 full-model oracle also retains exact IDs, text,
stop, and step contracts.

| C16 GDN route | Registers/thread | Static shared | Stack/local | Active CTA/SM |
| --- | ---: | ---: | ---: | ---: |
| Row8 predecessor | 40 | 34,568 B | 0 / 0 B | 4 |
| Register-resident production | 64 | 34,056 B | 0 / 0 B | 4 |

The extra 24 registers do not reduce the occupancy-API active-block ceiling,
and the packed route uses 512 fewer shared bytes. Final `cuobjdump` sections
contain 1,680 row8 and 2,160 exact static instructions. After removing
addresses/encodings and
normalizing opcode-plus-operand rows, their hashes are
`60b9f6f66ee7d77874c19f1f46304591353b74da37a6b501d2af297f79a24a95`
and
`b363d6dabc33196435399837433f89619423d696883576d03fd4b685edd48386`.
A second canonical-text method strips address prefixes and encoding comments,
collapses whitespace, and yields
`6aba6bcee9b958292338f96ed932515068cfce1d200c97b8bec6b14ce55c41f0`
and
`50f7d26942ccb48d492d12e07d264f7bee59185f8e2fa9c9a78573007dc6add8`.
The exact section contains three static `BAR.SYNC`, 32
`F2FP.BF16.PACK_AB`, 84 `PRMT`, and 288 `LOP3` instructions. These counts
establish the compiled mechanism and zero-spill identity, not dynamic stall
causality.

An incremental-build hazard was found before formal promotion. Three early
runs reported both labels near 0.321 ms, and the diagnostic profile contained
10,567 exact launches but only 136 row8 launches. The test executable still
contained an older test object in which both timing labels reached the public
launcher; after the public C16 selector changed, both therefore ran exact. A
forced rebuild of `gdn_decode_cuda_test.cu.o` and relink restored the explicit
row8 B route. An orphan earlier candidate object remains in the build directory
but is absent from the final link command and was not the root cause. All
pre-rebuild approximately-1.0x logs and their profile are explicitly excluded.

The formal screen uses the corrected same binary at fixed 1,300,500,000 Hz GPU
and 3,200,000,000 Hz EMC clocks. Each pass rotates through 24 state banks (36
MiB, nine times the 4 MiB L2), with 48 warmup and 480 measured launches. Five
independent processes each run five B-C-C-B rounds; the per-process statistic is
the median of ten pass means on each side. Every one of the 25 rounds favors the
candidate and every process clears the frozen 1.20x gate:

| Process | Row8 median | Exact median | Speedup |
| --- | ---: | ---: | ---: |
| 1 | 0.391646 ms | 0.321668 ms | 1.21755x |
| 2 | 0.391354 ms | 0.321536 ms | 1.21714x |
| 3 | 0.391427 ms | 0.321604 ms | 1.21711x |
| 4 | 0.391274 ms | 0.321612 ms | 1.21660x |
| 5 | 0.391298 ms | 0.321273 ms | 1.21796x |

The speedup min/median/mean/max is
1.21660x/1.21714x/1.217272x/1.21796x. The aggregate rows and clock readback were
retained only in the controlling console, so this record deliberately does not
invent per-process log hashes.

The fixed-frequency P33/max1 end-to-end screen uses an independent `ec9ac1a`
row8 runner and the `b09614c` exact runner in B-C-C-B order, with one warmup and
five measured samples per process:

| P33 median | B1 | C1 | C2 | B2 | Mirrored B | Mirrored C | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt prefix | 599.586 ms | 593.332 ms | 593.411 ms | 599.634 ms | 599.610 ms | 593.3715 ms | 1.010514x |
| Finish-prefill | 110.764 ms | 110.530 ms | 110.620 ms | 110.666 ms | 110.715 ms | 110.575 ms | 1.001266x |
| Prompt prefill / TTFT | 710.349 ms | 703.843 ms | 704.065 ms | 710.325 ms | 710.337 ms | 703.954 ms | 1.009067x |

All four runs produce the same 33 prompt IDs, generated ID `9419`, text
`Hello`, `max_new_tokens` stop, and 33-step trace; all report
`persistent_drop_detected=0`. Prompt-prefix latency falls by 6.2385 ms and TTFT
falls by 6.383 ms.

The longer fixed-frequency P513/C32/max1 screen uses the same independent
runners and B-C-C-B order, with one warmup and three measured samples per
process:

| P513 median | B1 | C1 | C2 | B2 | Mirrored B | Mirrored C | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Prompt prefix | 4,858.838 ms | 4,750.071 ms | 4,759.712 ms | 4,850.911 ms | 4,854.8745 ms | 4,754.8915 ms | 1.0210274x |
| Prompt prefill / TTFT | 4,970.824 ms | 4,861.946 ms | 4,871.458 ms | 4,962.815 ms | 4,966.8195 ms | 4,866.7020 ms | 1.02057194x |

All four runs retain the identical 513 prompt IDs and steps, generated ID
`9419`, text `Hello`, and `max_new_tokens` stop. Prompt-prefix and TTFT fall by
99.9830 and 100.1175 ms (2.0594% and 2.0157%). Persistent-drop bytes are
0/15,355,904/0/22,716,416 in B1/C1/C2/B2 order, all below the 64 MiB tolerance;
all four report `persistent_drop_detected=0`.

Matched P33 profiles preserve exactly 96 target launches, corresponding to two
C16 prefix tiles across 48 value heads:

| Nsys target | Launches | Total | Per launch | Change |
| --- | ---: | ---: | ---: | ---: |
| Row8 predecessor | 96 | 37.047840 ms | 0.385915 ms | baseline |
| Register-resident exact | 96 | 30.724128 ms | 0.320043 ms | -6.323712 ms, 1.205822x |

The generation NVTX range falls from 737.890112 to 731.369344 ms, a 6.520768 ms
saving, while all CUDA kernel time falls from 725.422496 to 718.581056 ms. The
target accounts for 96.9780% of the generation-range saving. That traced saving
closely matches the untraced P33 prefix result, but the separate traced
processes do not justify an exact causal subtraction. Full
artifact hashes, binary/build identities, stale-object exclusion, five-process
rows, and limitations are in the
[register-resident benchmark record](metadata/qwen36-27b-gdn-m16-register-resident-bf16-state-benchmark.json).
The clean-first Release build and all 56 discovered CTest entries complete
with 51 passes, five configured skips, and zero failures. The closeout does not
claim a fresh host-sanitizer suite, candidate NCU, device racecheck, Decode
improvement, system double/triple buffering, or multi-request overlap. Prefill
and Decode remain logically separated and the batch-one request remains
dependency-serialized.

## Post-GDN-M16 production phase profile

The fresh production profile after the exact-C16 GDN promotion uses source
commit `b09614c6c0ff3897f99434650de36893d5c06ac0`. The later `d9e40d9`
commit changes documentation only. Both captures use the same Release binary:
4,128,456 bytes, SHA-256
`298e603e133069dbc65dcc6d7eb71d266eebedf5ea917c6b381fa8e020bd46ec`,
ELF build ID `1995be1f538fd7a90183bd5df370c05055084c38`, and effective
`compute_87,sm_87` CUDA code generation. A target dry-run between the captures
requested no compile or link action.

The GPU and EMC min/max/current readbacks were fixed at 1,300,500,000 and
3,200,000,000 Hz. The runs use `--nvtx-phase-ranges` and do not use the CLI
`--trace` option, so the requested C32 schedule remains effective. The
normalized commands are:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --stats=false --force-overwrite=true \
  -o /tmp/q3x-phase-prefill-p513-c32-b09614c \
  build/orin-release/qwen3x-orin generate MODEL_DIR \
  --prompt "${P513_USER_TEXT}" --max-tokens 1 \
  --prefill-chunk-size 32 --projection-backend sm87 --nvtx-phase-ranges

nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --stats=false --force-overwrite=true \
  -o /tmp/q3x-phase-decode-p19-c32-b09614c \
  build/orin-release/qwen3x-orin generate MODEL_DIR \
  --prompt '用一句话解释 CUDA 是什么。' --max-tokens 26 \
  --prefill-chunk-size 32 --projection-backend sm87 --nvtx-phase-ranges
```

`P513_USER_TEXT` is exactly 501 space-separated copies of `hello`, producing
513 rendered tokens and sixteen C32 prefix tiles. Its observed oracle is
generated ID `9419`, text `Hello`, `max_new_tokens`, and 513 steps. The raw
Prefill console log was not retained. The independently retained Decode log
contains all 19 prompt IDs, all 26 expected output IDs, the exact Chinese
text, `im_end`, 44 steps, and `status=ok`.

The P513 phase and kernel closure is:

| P513 phase | NVTX ranges | NVTX host | GPU projection | Kernels | Kernel raw | Kernel union |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete generation | 1 | 4,939.968672 ms | 4,938.810720 ms | 42,709 | 5,237.841408 ms | 4,838.720512 ms |
| Prefix tiles | 16 | 4,800.798240 ms | 4,797.268256 ms | 42,224 | 5,100.255264 ms | 4,701.134368 ms |
| Finish-prefill | 1 | 138.998816 ms | 138.889792 ms | 485 | 137.586144 ms | 137.586144 ms |

The 42,224 prefix kernels plus 485 finish-prefill kernels close exactly to the
42,709 generation kernels. Prefix raw time exceeds its union by 399.120896 ms,
all from the intentional M32 gate/up main/auxiliary-stream overlap. Its kernel
span is 4,799.785312 ms, leaving 98.650944 ms, or 2.055320%, outside the
kernel union. Consequently the built-in raw kernel summary is not a valid
critical-path ranking for the gate/up pair.

For each row below, marginal exposure is the complete prefix kernel union
minus the union after removing every interval in that kernel-name group. It is
a current-timeline opportunity ranking, not a prediction that a candidate can
eliminate the whole interval:

| Prefill group | Instances | Raw time | Marginal exposure | Prefix-union share |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 M32 gate/up `[17408,5120]` pair | 2,048 | 1,766.932000 ms | 1,367.811104 ms | 29.095342% |
| NVFP4 M32 down `[5120,17408]` | 1,024 | 931.620448 ms | 931.620448 ms | 19.816929% |
| Exact GDN M16 register state | 1,536 | 487.589952 ms | 487.589952 ms | 10.371751% |
| FP8 M32 `[10240,5120]` | 768 | 390.644896 ms | 390.644896 ms | 8.309588% |
| FP8 M32 `[5120,6144]` | 1,024 | 382.405664 ms | 382.405664 ms | 8.134327% |
| FP8 M32 `[6144,5120]` | 768 | 247.019840 ms | 247.019840 ms | 5.254473% |

The first gate/up row contains 1,024 main-stream and 1,024 auxiliary-stream
launches. Its raw-to-marginal difference is exactly the 399.120896 ms prefix
overlap. Gate/up plus down therefore expose 2,299.431552 ms, or 48.912270% of
the prefix kernel union; including the newly promoted GDN route raises the top
three to 59.284021%.

The P19/max26 capture reports 454.074 ms TTFT, 2,786.014 ms Decode after the
first token, and 3,240.088 ms total generation. Its 25 Decode NVTX ranges total
2,786.236192 ms and average 111.449448 ms. All 10,925 associated Decode
kernels run on the same stream. Their raw and union times are identical at
2,765.851648 ms, or 110.634066 ms per step; the 2,786.299712 ms kernel span
contains 20.448064 ms idle (0.733879%) and zero overlap.

| Decode group | Instances | GPU time | Per Decode step | Decode-kernel share |
| --- | ---: | ---: | ---: | ---: |
| NVFP4 fused residual/norm/gate/up/SiLU | 1,600 | 995.135872 ms | 39.805435 ms | 35.979365% |
| FP8 linear-attention QKV/Z scratch ping-pong | 1,200 | 575.893568 ms | 23.035743 ms | 20.821564% |
| NVFP4 fused down/residual/norm | 1,600 | 522.837632 ms | 20.913505 ms | 18.903314% |
| FP8 row-quad output projection | 1,600 | 299.047168 ms | 11.961887 ms | 10.812119% |
| FP8 full-attention Q+K/V scratch ping-pong | 400 | 172.002016 ms | 6.880081 ms | 6.218772% |
| NVFP4 language head | 25 | 109.969568 ms | 4.398783 ms | 3.975975% |
| Fused Decode GDN/plain-RMSNorm/SiLU gate | 1,200 | 38.401568 ms | 1.536063 ms | 1.388417% |

The first three rows account for 75.704244% of Decode kernel time and the
first six account for 96.711110%. NVFP4 MLP gate/up plus down alone account for
54.882680%; the three FP8 attention-projection rows account for 37.852455%.

This closes the immediate scheduling-priority question. Prefill and Decode are
already logically separated plans, but batch-one execution remains causally
ordered and has no general double/triple buffer. The measured Decode scheduling
hole is below 1%, and Prefill already uses its one proven narrow two-stream
overlap. Phase-local kernel mechanisms therefore remain ahead of general
buffering. Prefill candidates must be screened against marginal exposure, not
the overlapped raw gate/up sum. The rejected pair-fused gate/up CTA and
dual-A/dual-decoded-B down pipeline remain closed unless a materially different
mechanism is proposed. Independent queues and buffering belong to the later
multi-request scheduler work with explicit request, KV/state, workspace,
fairness, cancellation, latency, and memory gates.

The Prefill report is 3,286,423 bytes with SHA-256
`32a9fbae00d93714c4328ce99b4d1be591dff2513f50e4dc8615f6c8effe35c7`.
The Decode report is 1,021,755 bytes with SHA-256
`7bdf228c74c93a31906ff25261b19a08395db505dcf83da002bcc6b556531f57`;
its 3,061-byte console log hashes to
`5cec6c4d71b4210fac78d02f15cfb0a456d807e63696b38dec0f15f0831305a9`.
These are one-process Nsight diagnostic captures, not repeated latency
distributions, an NCU causality study, a before/after promotion comparison, or
a serving-throughput claim. Full commands, SQLite attribution, oracle,
binary/environment identities, and limitations are in the
[post-GDN-M16 phase-profile record](metadata/qwen36-27b-post-gdn-m16-phase-profile.json).

## Decode FP8 M1 attention O-projection single-body/no-tail rejection

The first bounded follow-up to the refreshed Decode ranking targeted the FP8
row-quad output projection `[5120,6144]`, which accounts for 1,600 launches and
299.047168 ms, or 10.812119% of P19 Decode kernel time. Production launches
1,280 CTAs with 256 threads. Because 5,120 rows are exactly 1,280 complete row
quads and the registry cap is also 1,280, every CTA executes the generic
row-grid-stride loop exactly once.

The test-only candidate specialized that topology to one body per CTA. It
removed runtime rows/columns and the outer row loop, kept the decoded-codebook
producer barrier and the eight-warp reduction producer barrier, and removed
the trailing reduction-scratch lifetime barrier because no second body can
reuse the scratch. The public launcher continued to select the production
`fp8_w8a16_gemv_bf16_row_quad_kernel<true>` throughout the experiment.

The default device test passes 1,024/1,024 FP8 code/packed-position coverage,
0/5,120 candidate and replay BF16 mismatches, 8/8 NaN classifications for
signed `0x7f`/`0xff` codes,
zero unexpected nonfinite outputs, intact canaries, and preserved inputs.
Public, direct predecessor, and candidate CUDA Graph captures each contain one
`1280x256` kernel node; public and predecessor functions are identical and the
candidate is distinct. Four invalid shape, alias, and alignment calls return
`cudaErrorInvalidValue` and capture zero nodes.

Resources do not change:

| FP8 M1 `[5120,6144]` route | Registers/thread | Static shared | Local | Active CTA/SM |
| --- | ---: | ---: | ---: | ---: |
| Production row-quad predecessor | 64 | 1,152 B | 0 B | 4 |
| Test-only single-body/no-tail | 64 | 1,152 B | 0 B | 4 |

The complete production Function section contains 1,152 instructions and
2,304 ordered 64-bit encoding/control words. Its normalized word-text SHA-256
is `59d3521f9f64d81ee03d16503cef4096eef63e754fc9331c736765680cd85e40`
in the independent frozen binary, the candidate binary, the forced clean
rollback, and an older pre-exact SASS dump. The candidate section contains 960
instructions and 1,920 words with SHA-256
`80b703273a00a551d77bd5eb71e945625c7a33f352f5839259ce453b25cdacba`:

| Static SASS count | Production | Candidate |
| --- | ---: | ---: |
| `BAR.SYNC.DEFER_BLOCKING` | 3 | 2 |
| `FFMA` | 128 | 128 |
| `FADD` | 70 | 52 |
| `FMUL` | 7 | 4 |
| `SHFL.DOWN` | 41 | 40 |
| `BRA` + `BRA.DIV` | 19 | 12 |
| `CALL.REL.NOINC` | 22 | 1 |
| `RET.REL.NODEC` | 1 | 0 |
| `LDG.E` + `LDG.E.64` | 40 | 40 |
| `LDS` / `STS` / `STG.E.U16` | 132 / 5 / 4 | 132 / 5 / 4 |
| `LDL` / `STL` | 0 / 0 | 0 / 0 |

The candidate's lone predicated `CALL.REL.NOINC` at `0x2110` targets `0x2130`
inside the same Function section, has no `RET`, and allocates no stack or local
memory. It is a ptxas intra-section control transfer, not evidence of an
out-of-line source call; a textual zero-`CALL` gate would also reject the
production predecessor incorrectly. All 142 pre-existing Function sections
match between the frozen and candidate dumps by symbol, raw-block size, line
count, and SHA-256. The candidate adds exactly one Function section. This is a
per-function isolation result, not a claim that the whole SASS file is an
append-only byte stream.

The authoritative performance screen used the hash-pinned layer-0
`linear_attn.out_proj.weight` payload at absolute shard offset 3,569,011,232,
31,457,280 bytes, SHA-256
`e49a9f770a84cfcf7c2eb60f041aa7f1af8b84f5ab6323d3b8a9151588ff2bb9`.
GPU and EMC sysfs readbacks were fixed at 1,300,500,000 and 3,200,000,000 Hz
before and after the run. These readbacks prove the requested frequency lock,
not thermal or power stability. One same-binary process ran ten warmups and six
`B-C-C-B` rounds of 80 logical launches per timed pass:

| Round | Paired speedup | BF16 mismatches | Guards | Non-regression |
| ---: | ---: | ---: | --- | --- |
| 1 | 0.994295x | 0/5,120 | intact | fail |
| 2 | 0.995110x | 0/5,120 | intact | fail |
| 3 | 0.994488x | 0/5,120 | intact | fail |
| 4 | 0.995209x | 0/5,120 | intact | fail |
| 5 | 0.994857x | 0/5,120 | intact | fail |
| 6 | 0.994971x | 0/5,120 | intact | fail |

The selection statistic is the median of the six paired per-round
`Bmean/Cmean` speedups: 0.994914x, versus the required 1.01x. The minimum round
is 0.994295x versus the required 1.00x. Separately printed predecessor and
candidate medians are 0.189387 and 0.190360 ms; they are diagnostic values, not
the ratio used for selection. All seven reported failures are exactly the six
round non-regression failures plus the actual-checkpoint aggregate failure;
no correctness, resource, Graph, or fixture-identity gate failed.

The actual-checkpoint-first stop-loss therefore fired. The synthetic same-bank
fixture, five-process confirmation, P19 end-to-end run, Nsys, and NCU were not
run. This is consistent with two older same-family screens: conditional final
barrier skipping reached only 1.00207x/1.00109x on checkpoint-like/stress data,
and the older broad exact-shape candidate reached only 0.913770x/0.915366x for
this shape despite using 56 registers. Those historical runs are directional
context, not substitutes for the fixed-clock real-payload result.

The candidate and its test hooks were removed. A forced clean rebuild and
default device test pass; source and test blobs match `1c930d7`, and the
rollback production block remains byte-identical to both frozen comparators.
Production dispatch, runtime behavior, and public ABI never changed. The next
step is to re-rank the remaining phase-local opportunities from the post-M16
profile before proposing a materially different bounded mechanism; this
rejection does not preselect one. The performance command, symbols, word hashes,
per-pass timings, artifact identities, historical log hashes, rollback proof,
and limitations are in the
[attention O-projection no-tail rejection record](metadata/qwen36-27b-fp8-m1-attention-o-proj-no-tail-rejection.json).

## Decode FP8 M1 output-projection AoSoA4/preswizzled test-only selection

The materially different exact `[5120,6144]` follow-up places each four-row
packed word in one `uint4` and pretransforms every FP8 byte as
`code ^ (code >> 5)`. Its test-only fixed-1,024-CTA kernel clears the
hash-pinned actual layer-0 gate at a 1.05155x paired median with all five
`B-C-C-B` rounds non-regressing, and clears same-bank stress at 1.01465x.
Actual candidate/replay mismatches are 0/5,120, inputs, sidecar, and guards are
preserved, and resources remain 64 registers, 1,152 B shared, zero local, and
four active CTAs/SM. Diagnostic production/candidate pass medians are
0.187221/0.178149 ms.

One persistent sidecar costs 30 MiB; 64 layers cost 1.875 GiB. Multiplying the
isolated 0.009072-ms delta by 64 gives only a projection estimate of 0.580608
ms/token, moving the 110.951-ms planning baseline arithmetically to 110.370 ms
or about 9.060 token/s. No loader, allocation, production selector,
model-oracle, end-to-end, Nsys, or NCU gate has run, so this selects only the
test-only persistent-sidecar integration experiment and does not claim an
achieved runtime improvement. Full rounds, hashes, SASS, memory obligations,
and promotion gates are in the
[AoSoA4/preswizzled selection record](metadata/qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-selection.json).

## Decode FP8 M1 output-projection persistent-sidecar production promotion

The selected exact `[5120,6144]` route is now integrated for production SM87
M1 dispatch. Model construction allocates one persistent arena, launches one
GPU AoSoA4/preswizzle pack per layer, synchronizes it before runner creation,
and atomically attaches all 64 sidecars. Canonical weights remain resident for
Prefill, M2+, and fallback routes. Admission preserves the configured 8-GiB
post-create free-memory margin; memory pressure falls back to canonical
weights, while an actual pack or attachment failure fails construction.

The production formal run repeats the hash-pinned actual payload and stress
gates and validates the GPU pack against a full 30-MiB CPU oracle while
preserving canonical input. All seven pack invalid cases and all eleven direct
production-GEMV invalid cases return `cudaErrorInvalidValue` with zero
captured nodes. The kernel retains 64 registers/thread, 1,152 B static shared
memory, zero local memory, and four active CTAs/SM.

| Formal fixture | Production median | Sidecar median | Paired speedup | Result |
| --- | ---: | ---: | ---: | --- |
| Actual layer-0 checkpoint | 0.185194 ms | 0.176046 ms | 1.05212x | pass |
| Same-bank stress | 0.168455 ms | 0.166611 ms | 1.00853x | pass |

The fixed-frequency end-to-end gate used separate frozen Release binaries in
`B1-C1-C2-B2` process order, with one warmup and five measured generations per
process on the P19/C32/max26 workload:

| Process | Subsequent-token median | Decode-after-first median | Cold sidecar pack |
| --- | ---: | ---: | ---: |
| B1 | 110.082 ms | 2,752.664 ms | n/a |
| C1 | 109.786 ms | 2,744.812 ms | 437.460 ms |
| C2 | 109.731 ms | 2,743.777 ms | 497.171 ms |
| B2 | 110.130 ms | 2,757.813 ms | n/a |

The mean of mirrored process medians improves subsequent-token latency from
110.1060 to **109.7585 ms**, a 0.3475-ms reduction or 1.003166x speedup. Both
mirrored pairs improve independently. Decode-after-first moves from
2,755.2385 to 2,744.2945 ms, a 10.944-ms reduction or 1.003988x speedup. The
current hot single-request rate is therefore **9.1109 token/s**. It remains
9.7585 ms/token above the 100-ms target, requiring another 8.89% latency
reduction from the current value.

This promotion deliberately does not hide its system cost. The 64-layer arena
is 2,013,265,920 bytes, or 1.875 GiB, and the two benchmark processes spend
437.460/497.171 ms constructing it (467.316 ms mean). Those are cold startup
costs and are excluded from the hot Decode numbers above. The final Release
binary differs from the timed candidate only by `noexcept` and memory-margin
exception-path repairs; it separately passes exact 19-prompt-ID, 26-generated-
ID, text, `im_end`, and 44-step replay with all 64 sidecars attached and a
460.648-ms cold pack.

This is a narrow persistent weight-format sidecar, not a runtime double/triple
buffer or Prefill/Decode overlap. The immediately following NVFP4 down CTA-
prune screen is closed below; a fresh production profile and hotspot
re-ranking now follow it.
Full binary and log hashes, aggregation arithmetic, validation scope, memory
policy, and limitations are in the
[production benchmark record](metadata/qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-production-benchmark.json).

## Decode NVFP4 M1 down/norm post-sync CTA-prune rejection

The test-only exact `[5120,17408]` candidate keeps the complete 64-CTA
projection, BF16 raw/residual boundaries, and cooperative grid sync, then
returns CTAs 20–63 before the repeated RMS reduction because only CTAs 0–19
publish normalized slices. It retains 64 registers/thread, 35,904 B shared,
zero local, and four active CTAs/SM. Production and candidate are distinct
single-node `64x256` Graphs. Both checkpoint-like and same-bank synthetic
fixtures match raw/residual/normalized at `0/5,120`, with intact guards and
preserved inputs.

Five 80-launch `B-C-C-B` rounds all non-regress, but paired medians are only
1.00135x and 1.00107x versus the required 1.002x. Pass medians move merely
0.320397→0.320088 and 0.321472→0.321102 ms, an arithmetic 64-layer saving of
about 0.020–0.024 ms/token rather than an achieved end-to-end result. Stop-loss
therefore rejects and removes the candidate without actual checkpoint, NCU,
Nsys, model-oracle, or P19 timing. Full evidence is in the
[CTA-prune rejection record](metadata/qwen36-27b-nvfp4-m1-down-norm-cta-prune-rejection.json).

## Post-FP8-output-sidecar Decode phase profile

The required current-production refresh profiles `aa7312b`, whose latest
runtime source is the `77931b8` output-sidecar promotion. The documentation-only
commit between those identities does not alter the executable. The captured
Release binary is 4,199,264 bytes with SHA-256
`52968768870629826635b395e679b0500fb16b2692a55b79b58697ad19317462`.
The fixed-GPU-frequency P19/C32/max26 run contains 25 completed
`q3x.decode.step` ranges and closes over 10,925 unique kernel rows on stream
18. Raw and interval-union time are both 2,745.814816 ms, so measured overlap
is zero. The associated spans contain 17.226272 ms idle, or 0.623453%.

The current exact-name ranking is:

| Rank | Decode group | Launches | Mean launch | Per Decode step | Raw share |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | NVFP4 residual/norm/gate/up/SiLU | 1,600 | 0.619935 ms | 39.675852 ms | 36.123932% |
| 2 | FP8 linear-attention QKV/Z | 1,200 | 0.477048 ms | 22.898294 ms | 20.848359% |
| 3 | NVFP4 down/residual/norm | 1,600 | 0.324463 ms | 20.765622 ms | 18.906612% |
| 4 | FP8 output AoSoA4/preswizzled sidecar | 1,600 | 0.181293 ms | 11.602725 ms | 10.564009% |
| 5 | FP8 full-attention Q+K/V | 400 | 0.427392 ms | 6.838275 ms | 6.226089% |
| 6 | NVFP4 language head | 25 | 4.381676 ms | 4.381676 ms | 3.989413% |
| 7 | Fused Decode GDN/plain-RMSNorm/SiLU gate | 1,200 | 0.032070 ms | 1.539343 ms | 1.401536% |
| 8 | BF16 projection pair tile | 1,200 | 0.019425 ms | 0.932381 ms | 0.848911% |
| 9 | GQA sigmoid gate | 400 | 0.049282 ms | 0.788512 ms | 0.717922% |
| 10 | Causal Conv1D/SiLU | 1,200 | 0.005566 ms | 0.267174 ms | 0.243256% |

The top three still account for 75.878904% of raw Decode kernel time; the top
ten account for 99.870039%. Compared directionally with the earlier matched-
workload `9bddbda` trace, raw time is 11.810816 ms lower across 25 steps
(0.472433 ms/step), and the profiled host-range average moves from 110.953001
to 110.519286 ms. The output-projection row itself moves from 299.916576 to
290.068128 ms across 1,600 launches: 0.187448→0.181293 ms/launch and
11.996663→11.602725 ms/step. Other unchanged rows move in both directions,
so separate-process Nsys drift prevents attributing the complete trace delta
to the sidecar or claiming intrinsic changes in those rows.

This single profiler run is hotspot-selection and topology evidence. It does
not replace the fixed-frequency, unprofiled `B1-C1-C2-B2` release result of
**109.7585 ms/token and 9.1109 token/s**. That remains 9.7585 ms/token above
the target. The new ranking keeps materially different, actual-payload-gated
phase-local work ahead of general batch-one double/triple buffering: the trace
is single-stream and its 0.623453% idle fraction is much smaller than the
leading kernel exposures. Rejected gate/up load-shape and down CTA-prune
mechanisms remain closed. Commands, exact arithmetic, report/SQLite hashes,
same-binary oracle anchor, and limitations are retained in the
[post-sidecar Decode phase-profile record](metadata/qwen36-27b-post-fp8-output-sidecar-decode-phase-profile.json).

## Decode FP8 M1 linear-attention QKV/Z AoSoA4 sidecar rejection

The first bounded follow-up to that refreshed ranking targeted the second-place
FP8 linear-attention QKV/Z row: 22.898294 ms per Decode step and 20.848359% of
raw Decode kernel time. The test-only exact `[10240,5120]` QKV plus
`[6144,5120]` Z candidate preserves the production one-kernel two-phase
`1536/768` reduction-scratch ping-pong topology. Its sidecars interleave the
same packed word from four adjacent rows in one `uint4` and preswizzle every
FP8 byte with `code ^ (code >> 5)`.

Production and candidate remain distinct one-node `1536x256` Graphs and both
retain 64 registers/thread, 1,280 B static shared, zero local memory, and four
active CTAs/SM. All nine invalid calls capture zero nodes. The hash- and
offset-pinned layer-0 QKV/Z payload plus same-bank stress fixture each match
QKV, Z, and replay outputs bitwise, with intact output/sidecar guards and
preserved canonical inputs and activation. Static inspection also proves the
production function unchanged at 2,416 normalized words while the candidate
falls to 2,224 words.

Five 80-launch `B-C-C-B` rounds produce:

| Fixture | Production median | Candidate median | Paired median | Frozen gate | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Actual layer-0 checkpoint | 0.472820 ms | 0.470228 ms | 1.00562x | 1.03x | fail |
| Same-bank stress | 0.439653 ms | 0.446634 ms | 0.984394x | 1.00x | fail |

All actual-payload rounds improve, but by far too little for the residency
cost, while every stress round regresses. The isolated actual pass-median
delta multiplied by 48 linear-attention layers is only **0.124416 ms/token**
in an ideal arithmetic projection, not an achieved end-to-end result. The
sidecars would add 83,886,080 bytes per layer and 4,026,531,840 bytes
(3.75 GiB) across 48 layers while canonical weights remain required.

Stop-loss therefore removes the candidate and retains production at
`1ab756a`. The 48-layer sidecar, loader/model integration, full-model oracle,
P19 benchmark, Nsys, NCU, and cross-process replication were not run. The
default restored device test passes. Full round data, payload identities,
SASS/artifact hashes, cleanup proof, and limitations are retained in the
[QKV/Z AoSoA4 sidecar rejection record](metadata/qwen36-27b-fp8-m1-qkv-z-aosoa4-preswizzled-sidecar-rejection.json).

## Decode NVFP4 M1 gate/up 32x512 CTA-coarsening selection

The next bounded screen returns to the leading Decode row with a materially
different mechanism. The test-only exact `[17408,5120]` candidate regroups the
same 512 projection warps from production's 64 CTAs of eight warps into 32
CTAs of 16 warps. The global 2,048-row stride, warp-to-row-quad mapping,
packed-weight/scale accesses, FFMA and reduction order, and BF16 boundaries are
unchanged. Threads 0–255 also retain the exact production residual and
centered-RMSNorm accumulation/reduction order; threads 256–511 skip that
arithmetic but participate in every CTA barrier. The change therefore halves
the number of CTAs that redundantly prepare the same per-token residual/norm
activation without adding a sidecar or model-resident memory.

The candidate retains 64 registers/thread, 11,328 B static shared, and zero
local memory. Its two 512-thread CTAs per SM preserve the production total of
32 resident warps/SM. Production and candidate capture as distinct single
kernel nodes at `64x256` and `32x512`, respectively; all nine invalid calls
capture zero nodes. The actual layer-0 checkpoint and same-bank stress
fixtures match residual, final gate, up, and deterministic replay outputs
bitwise, with intact guards and preserved inputs. The signed Inf/NaN case also
matches all three outputs bitwise and preserves all 34,816 classified NaN
outputs, including class and sign.

Three independent same-binary processes each run five 64-launch `B-C-C-B`
rounds per fixture:

| Fixture | Per-process paired medians | Cross-process median | Frozen per-process gate | Result |
| --- | --- | ---: | ---: | --- |
| Actual layer-0 checkpoint | 1.02410x / 1.02456x / 1.02181x | 1.02410x | 1.005x plus no regressing round | pass |
| Same-bank stress | 1.02312x / 1.02295x / 1.02106x | 1.02295x | 1.000x plus no regressing round | pass |

All 30 paired rounds improve, and all three processes exit zero. Static
inspection independently keeps the
production function identical between the clean `8026b29` baseline and the
candidate build at 2,688 normalized instruction words with SHA-256
`1e5139d45e1cec02a7d416f7dbc8776098e4e2faa5f5d56fb4dc2a6483eca98a`;
the distinct test-only candidate contains 2,704 words. The successful test
binary also rejects a null resource-query destination, but its launcher is
still unreachable from production.

The three isolated actual pass-median deltas are 0.014469, 0.014486, and
0.013029 ms/layer. Multiplying their median by 64 dense layers gives an ideal
**0.926016 ms/token** arithmetic saving, with a 0.833856–0.927104-ms/token
process range. Those are only planning bounds, not achieved end-to-end
reductions: the formal result remains **109.7585 ms/token and 9.1109 token/s**,
still 9.7585 ms/token from the stage target.
The candidate is selected for production integration next. Promotion requires
the final Release resource/Graph/invalid/static gates, pinned full-model exact
oracle, fixed-frequency P19/C32/max26 mirrored end-to-end measurement, and a
fresh Decode Nsys closure before the formal performance anchor changes. The
complete 15-plus-15 rounds, payload identities, artifacts, gates, and
limitations are retained in the
[CTA-coarsening selection record](metadata/qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-selection.json).

## Decode NVFP4 M1 gate/up 32x512 CTA-coarsening production promotion

Commit `15cbb28` promotes the selected exact `[17408,5120]` route without
changing its public API, dispatcher eligibility, model format, loader, or
resident-memory footprint. The public residual/norm/gate/up/SiLU launch now
uses 32 CTAs of 512 threads. The old 64-CTA, 256-thread implementation remains
reachable only as a same-binary predecessor for regression gates, while a
second test entry replays the production function identity directly.

The final production screen retains the frozen layer-0 checkpoint payload,
same-bank stress fixture, five 64-launch `B-C-C-B` rounds per fixture, and the
per-round no-regression requirement:

| Fixture | 64x256 predecessor | 32x512 production | Paired median | Result |
| --- | ---: | ---: | ---: | --- |
| Actual layer-0 checkpoint | 0.615768 ms | 0.601257 ms | 1.02388x | pass |
| Same-bank stress | 0.619698 ms | 0.605754 ms | 1.02302x | pass |

All ten paired rounds improve. Both finite fixtures match residual at
`0/5,120`, final gate and up at `0/17,408` each, and deterministic replay at
`0/39,936`, with intact guards and preserved inputs. The signed Inf/NaN case
matches both implementations bitwise and preserves all 34,816 expected NaN
outputs, including class and sign. The production node is exactly `32x512`,
the predecessor is a distinct `64x256` node, and all nine invalid calls capture
zero nodes.

The promoted kernel retains 64 registers/thread, 11,328 B static shared
memory, zero per-thread local memory, and two active CTAs/SM, or the same 32
resident warps/SM as the predecessor. Normalized SASS remains identical to the
selected candidate: 2,704 words with SHA-256
`134202f948d757928aa744d20fa7064bdb28b12334659e88046a0fecd071d2ab`.
The retained predecessor remains 2,688 words with SHA-256
`1e5139d45e1cec02a7d416f7dbc8776098e4e2faa5f5d56fb4dc2a6483eca98a`.
Release build, the four targeted CTests, and the pinned full-model oracle all
pass. The oracle reproduces 19 prompt IDs, 26 generated IDs, exact text,
`im_end`, and 44 steps with requested/effective C32 and all 64 FP8 output
sidecars attached.

The fixed-frequency end-to-end gate freezes separate Release binaries and
runs one warmup plus five measured generations per process in
`B1-C1-C2-B2` order on P19/C32/max26:

| Process | Subsequent-token median | Decode-after-first | TTFT | Total generation |
| --- | ---: | ---: | ---: | ---: |
| B1 predecessor | 109.868 ms | 2,745.622 ms | 428.830 ms | 3,174.413 ms |
| C1 production | 108.959 ms | 2,723.965 ms | 427.775 ms | 3,151.794 ms |
| C2 production | 109.153 ms | 2,728.987 ms | 428.088 ms | 3,156.942 ms |
| B2 predecessor | 109.766 ms | 2,744.400 ms | 429.043 ms | 3,173.748 ms |

Both mirrored pairs improve independently. The mean of process medians moves
hot subsequent-token latency from 109.817 to **109.056 ms/token**, a
0.761-ms reduction or 1.006978x speedup. Decode-after-first moves from
2,745.011 to 2,726.476 ms, a reduction of 18.535 ms across 25 later-token
steps. The new formal single-request rate is **9.1696 token/s**. Compared with
the previous 109.7585-ms release anchor, the accumulated production result is
0.7025 ms/token lower. The remaining stage gap is 9.056 ms/token, or
0.8304 token/s; reaching 100 ms now requires another 8.304% latency reduction.

A fresh same-binary Nsys capture closes 25 Decode ranges over 10,925 distinct
kernel rows, exactly 437 per step. The one generation range contains 12,997
kernels and closes exactly over the prefix, finish, and Decode leaves with no
missing, extra, or duplicate rows. Decode remains on one stream with
2,725.022432 ms raw time equal to interval union, zero overlap, and
17.434496 ms idle across the associated span. The promoted gate/up kernel is
present exactly 1,600 times at `32x512`; the predecessor is absent. The FP8
output sidecar remains present 1,600 times, with no canonical output fallback
or pack work inside generation.

| Rank | Decode group | Launches | Mean launch | Per Decode step | Raw share |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | NVFP4 residual/norm/gate/up/SiLU 32x512 | 1,600 | 0.605825 ms | 38.772791 ms | 35.571075% |
| 2 | FP8 linear-attention QKV/Z | 1,200 | 0.477056 ms | 22.898703 ms | 21.007812% |
| 3 | NVFP4 down/residual/norm | 1,600 | 0.326449 ms | 20.892739 ms | 19.167492% |
| 4 | FP8 output AoSoA4/preswizzled sidecar | 1,600 | 0.180820 ms | 11.572457 ms | 10.616846% |
| 5 | FP8 full-attention Q+K/V | 400 | 0.428260 ms | 6.852165 ms | 6.286338% |
| 6 | NVFP4 language head | 25 | 4.382528 ms | 4.382528 ms | 4.020635% |

Against the immediately preceding matched-workload profiler capture, the
gate/up row falls from 39.675852 to 38.772791 ms/step, a 0.903060-ms reduction
or 1.023291x directional speedup. Complete Decode raw time is 0.831695 ms/step
lower. These profiler values diagnose topology and attribution only; they do
not replace the unprofiled 109.056-ms release anchor. Nsight also reports a
14,155,776-byte context-level `localMemoryTotal` field for both old and new
captures; the kernel-specific per-thread local field, resource query, and SASS
audit all remain zero, so that context field is not evidence of a spill.

The promotion adds no double/triple buffer and does not overlap Prefill with
Decode. Gate/up remains the largest Decode row, but QKV/Z and down remain close
behind; the next bounded mechanism must be selected from the refreshed
production profile without restoring the rejected QKV/Z sidecar, down CTA
prune, or earlier gate/up load-shape branches. Full binary identities,
commands, arithmetic, logs, and limitations are retained in the
[production benchmark record](metadata/qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-production-benchmark.json)
and the
[post-promotion Decode profile](metadata/qwen36-27b-post-gate-up-cta-coarsen-decode-phase-profile.json).

## Decode NVFP4 M1 down 32x512 CTA-coarsening selection

The next bounded Decode screen targets the rank-three exact `[5120,17408]`
down/residual/centered-RMSNorm cooperative kernel. The test-only candidate
regroups the same 512 projection warps from production's 64 CTAs by 256
threads into 32 CTAs by 512 threads. It preserves projection, raw-down BF16,
residual BF16, grid-sync, 256-lane RMS reduction, gamma, and normalized BF16
order while halving complete 34-KiB activation staging and redundant RMS
reductions. Production dispatch and the public ABI remain unchanged.

Three independent processes used one frozen Release binary, the pinned actual
layer-0 down payload, same-bank stress, ten warmups, 64 launches per timed
pass, and five `B-C-C-B` rounds per fixture:

| Fixture | Process paired medians | Cross-process median | Frozen gate |
| --- | --- | ---: | ---: |
| Actual checkpoint | 1.01181x / 1.01842x / 1.01176x | 1.01181x | >=1.005x |
| Same-bank stress | 1.01218x / 1.01876x / 1.01453x | 1.01453x | >=1.000x |

All 30 paired rounds improve; the minimum actual and stress rounds are
1.01028x and 1.01191x. Both finite fixtures match production bitwise at
`0/5,120` for raw, residual, and normalized output and `0/15,360` on candidate
replay, with intact guards and all five inputs preserved. Independent
residual-only and norm-weight-only signed Inf/NaN cases pass the same
three-output, replay, class/sign, guard, and input gates. The candidate Graph
is a distinct one-node `32x512` cooperative launch; all nine invalid calls
capture zero nodes.

The candidate uses 64 registers/thread, 35,904 B static shared memory, zero
local memory, and two active CTAs/SM. Its 32-CTA grid therefore exactly equals
the 16-SM resident capacity, preserving 32 resident warps/SM but leaving no
occupancy margin. Final frozen-binary extraction retains production's
historical 1,280-instruction body SHA-256
`17b02b92fa3404948dd695d3fee3f34d7ddf7d0d4912357063cacb83c09674e5`;
the distinct candidate canonical encoding SHA-256 is
`fde55ae44d31797fe93f31a682c74bb6515133f75a68c7189879ca9057055ad8`.

The median actual pass delta is 0.003851 ms/layer. Multiplying it by 64 gives
an ideal **0.246464 ms/token** arithmetic saving, only 2.721555% of the current
9.056-ms stage gap. It is not an achieved runtime result: the formal anchor
remains **109.056 ms/token and 9.1696 token/s**. The candidate is selected for
production integration next, followed by final Release/static gates, the
pinned full-model oracle, fixed-frequency mirrored end-to-end measurement,
and a fresh Decode trace. Complete rounds, payload identities, binary/log/SASS
hashes, and claim limits are in the
[down CTA-coarsening selection record](metadata/qwen36-27b-nvfp4-m1-down-cta-coarsen-selection.json).

## Decode NVFP4 M1 down 32x512 CTA-coarsening production promotion

Commit `0372d41` (tree `aa341b9`) promotes the selected exact
`[5120,17408]` down/residual/centered-RMSNorm route. The production cooperative
launch is now `32x512`; the distinct `64x256` implementation remains only as a
test predecessor. The change preserves 512 projection warps, 64
registers/thread, 35,904 B static shared memory, zero local memory, and 32
resident warps/SM. Two active CTAs on each of 16 SMs give a resident capacity
of exactly 32 CTAs, equal to the cooperative grid and therefore without extra
residency margin. It adds no allocation, stream, system double/triple buffer,
or Prefill/Decode overlap.

The final production screen uses three independent processes, ten warmups per
route, and five 64-launch `B-C-C-B` rounds for both the pinned actual payload
and same-bank stress:

| Fixture | Process 1 | Process 2 | Process 3 | Cross-process median | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Actual checkpoint | 1.01511x | 1.01184x | 1.01419x | 1.01419x | pass |
| Same-bank stress | 1.01975x | 1.01416x | 1.01544x | 1.01544x | pass |

All 30 paired rounds improve. Raw-down, residual, normalized output, and
deterministic replay remain bitwise equal on both finite fixtures; the split
residual-only and norm-weight-only signed Inf/NaN cases preserve output class
and sign. Guards and inputs are intact. Production and predecessor capture as
distinct one-node Graphs at `32x512` and `64x256`; all nine invalid calls
capture zero nodes. Default validation ends in `SM87 weight-only GEMV tests
passed`, all four focused CTests pass, and the pinned full-model oracle
reproduces 19 prompt IDs, 26 generated IDs, exact text, `im_end`, and 44
steps. The promoted canonical encoding SHA-256 is
`fde55ae44d31797fe93f31a682c74bb6515133f75a68c7189879ca9057055ad8`;
the predecessor remains
`fe10e377d8377678ad6fe3dcc43b86011476f839171bd5ffacc7d6584f6c2922`.

The median actual pass-median difference is 0.004303 ms/layer. Its linear
64-layer value is **0.275392 ms/token**, an isolated arithmetic projection and
not an end-to-end claim.

The fixed-clock end-to-end gate first encountered a baseline inference that
completed but whose stdout was truncated. It is excluded as an incomplete
formal sample, and the complete sequence restarts from B1:

| Process | Subsequent token | Decode after first | TTFT | Total generation |
| --- | ---: | ---: | ---: | ---: |
| B1 predecessor | 109.431 ms | 2,735.798 ms | 428.921 ms | 3,164.748 ms |
| C1 production | 109.066 ms | 2,726.918 ms | 428.227 ms | 3,155.146 ms |
| C2 production | 109.165 ms | 2,728.960 ms | 428.556 ms | 3,157.503 ms |
| B2 predecessor | 109.316 ms | 2,732.469 ms | 428.502 ms | 3,161.081 ms |

Both mirrored pairs improve. The mean of complete process medians moves from
109.3735 to 109.1155 ms/token, a **0.258-ms** same-run reduction or
1.002364x speedup. Decode-after-first falls 6.1945 ms across the 25 subsequent
steps. Functional canonicalization is byte-identical for all four processes
at SHA-256
`f66b837ed8f5b17f1307b424062c432c7ed02befdc97d99147f34c7675decee8`.

This relative win promotes the implementation, but it does not raise the
absolute release claim: the fresh 109.1155-ms candidate is 0.0595 ms slower
than the previous formal 109.056-ms result. The conservative anchor therefore
remains **109.056 ms/token and 9.169600939 token/s**, leaving 9.056 ms/token
to the 100-ms stage target. Applying the same-run -0.258-ms delta to the old
anchor gives **108.798 ms/token and 9.191345429 token/s only as planning
normalization**; those values were not measured and must not be cited as
achieved.

The fresh Nsys capture closes all 25 Decode ranges over 10,925 distinct kernel
rows, exactly 437 per step. Raw time equals interval union at 2,725.072960 ms;
the associated span is 2,740.776000 ms with 15.703040 ms idle (0.572941%) and
zero overlap on stream 18. Prefix, finish, and Decode leaves also close the
generation range exactly at 12,997 rows and 3,171.275488 ms.

| Rank | Decode group | Launches | Per Decode step | Raw share | Delta vs prior profile |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | NVFP4 residual/norm/gate/up/SiLU `32x512` | 1,600 | 38.834222 ms | 35.626773% | +0.061431 ms |
| 2 | FP8 linear-attention QKV/Z | 1,200 | 22.963224 ms | 21.066614% | +0.064521 ms |
| 3 | NVFP4 down/residual/norm `32x512` | 1,600 | 20.813704 ms | 19.094630% | -0.079035 ms |
| 4 | FP8 output AoSoA4/preswizzled | 1,600 | 11.506221 ms | 10.555883% | -0.066236 ms |
| 5 | FP8 full-attention Q+K/V | 400 | 6.872178 ms | 6.304582% | +0.020013 ms |
| 6 | NVFP4 language head | 25 | 4.380984 ms | 4.019144% | -0.001544 ms |

Production Down appears exactly 1,600 times at `32x512`, or 64 launches in
every Decode step. The predecessor symbol and every down `64x256` topology are
absent. The Down row moves directionally lower by 0.079035 ms/step, but total
Decode raw time changes by only +0.002021 ms/step across separate profiler
processes. This is a topology and hotspot result, not causal attribution for
unchanged rows and not a replacement release benchmark.

The refreshed order remains gate/up, QKV/Z, Down, then output; the top three
account for 75.788017% of Decode raw time. Continue with a materially new
gate/up or QKV/Z mechanism before returning to blind Down CTA tuning. The
single-stream, zero-overlap trace exposes too little idle time to prioritize a
general same-request double/triple-buffer rewrite. Full artifacts and claim
boundaries are retained in the
[production benchmark](metadata/qwen36-27b-nvfp4-m1-down-cta-coarsen-production-benchmark.json)
and
[post-promotion Decode profile](metadata/qwen36-27b-post-down-cta-coarsen-decode-phase-profile.json).

## Decode NVFP4 M1 gate/up 16x1024 balanced-tail rejection

The next bounded screen asked whether the production exact `[17408,5120]`
fused residual/norm/gate/up/SiLU route could profitably coarsen one final time.
The test-only candidate keeps the production `32x512` kernel's 512 global
projection warps, 2,048-row stride, packed-weight arithmetic, and BF16
boundaries, but places two 512-thread logical CTAs in each 1,024-thread
physical CTA. Physical CTA `b` pairs logical blocks `b` and `b+16`; physical
warps 0–15 serve the former and warps 16–31 serve the latter. The final 1,024
rows belong to logical blocks 0–15, so this pairing gives every physical CTA
one tail-bearing and one non-tail logical block. It also reduces repeated
residual/RMSNorm setup from 32 physical CTAs to 16. Low threads 0–255 preserve
the production accumulation and reduction tree, and all 1,024 threads traverse
every CTA barrier.

This is specifically the `b`/`b+16` balanced-tail mapping. A naive contiguous
`16x1024` row mapping was neither implemented nor timed and is not rejected by
this result. The candidate remained test-only throughout; production dispatch,
the public ABI, model storage, and the production `32x512` route did not
change.

Static and launch-contract gates pass:

| Exact M1 fused route | Registers/thread | Static shared | Local | Active CTA/SM | Resident capacity | Graph |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Production `32x512` | 64 | 11,328 B | 0 B | 2 | 32 CTAs | one node, `32x512` |
| Test-only `16x1024` | 64 | 11,328 B | 0 B | 1 | 16 CTAs | one distinct node, `16x1024` |

The candidate grid exactly equals the measured 16-SM resident capacity, with
no CTA margin, while retaining 32 resident warps/SM. Its null resource query
is rejected, and all nine invalid launches capture zero Graph nodes. Actual
layer-0 checkpoint and same-bank-stress fixtures match production bitwise at
0/5,120 residual, 0/17,408 final-gate, 0/17,408 up, and 0/39,936 replay
mismatches, with finite outputs, intact guards, and preserved inputs. The
combined signed Inf/NaN fixture also matches all three outputs bitwise and
preserves class and sign for all 34,816 classified NaN outputs.

The frozen candidate ELF is 5,435,784 bytes, SHA-256
`e4024bfc5932344d565724ca7c57182cf4f0eeeca97d51ba5e3ebb1e2102cc87`,
and Build ID `79c4e43d2f0d36bbf3be896fca87f9e9c59ebf4e`. Inside that same
binary, production's normalized encoding remains its promoted 2,704-word
stream at SHA-256
`134202f948d757928aa744d20fa7064bdb28b12334659e88046a0fecd071d2ab`.
The distinct candidate has 2,720 words at SHA-256
`4bdb0246107e16c68c75e00803452d446d193c182d701d4e0d1649b81bdc8b2c`
and contains no `LDL` or `STL`.

One fixed-clock same-binary process used ten warmups per route and five
64-launch `B-C-C-B` rounds per fixture:

| Fixture | Production pass median | Candidate pass median | Paired range | Paired median | Frozen median gate | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Actual layer-0 checkpoint | 0.603352 ms | 0.606524 ms | 0.994464x–0.995194x | 0.994805x | 1.005x | fail |
| Same-bank stress | 0.607962 ms | 0.609870 ms | 0.996282x–0.997435x | 0.996853x | 1.000x | fail |

All ten rounds regress; pass-median latency rises by approximately 0.526% on
the actual payload and 0.314% on stress. The expected nonzero screen exit is
therefore a performance-gate rejection, not a correctness or resource
failure. Stop-loss ends the experiment after the first process: no
three-process confirmation, production integration, full-model oracle,
end-to-end benchmark, Nsys, or NCU study is attached. Without NCU counters,
the result does not assign causality to the exact-capacity launch, scheduling,
barriers, or memory traffic.

The candidate and all test hooks are removed, and source/test blobs again
match clean commit `31716af`; production remains `32x512`. Consequently the
formal single-request result remains **109.056 ms/token and 9.169600939
token/s**, still 9.056 ms/token from the stage target. Complete rounds,
payload identities, artifact hashes, cleanup proof, and scope limits are in
the
[16x1024 rejection record](metadata/qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-1024-rejection.json).

## Decode FP8 M1 QKV/Z ping-pong QKV-grid-cap 1024 rejection

This bounded screen keeps the exact production QKV/Z reduction-scratch
ping-pong function and compiled SASS, changing only its test policy from
QKV1536/Z768 to QKV1024/Z768. The block remains 256 threads, the Z cap remains
768, and the two CTA-local scratch slots, arithmetic order, phase order, BF16
boundaries, public ABI, and production dispatch are unchanged. This is not an
executor-level double buffer and adds no cross-kernel or Prefill/Decode
overlap.

The shared production kernel remains at 64 registers/thread, 1,280 B static
shared memory, zero local memory, 256 maximum threads/block, and four active
CTAs/SM. Public, performance, and test launchers resolve to the same function;
the production Graph contract remains `1536x256`, null resource queries are
rejected, and invalid calls pass. Because the candidate is only a runtime cap
on the same function, it has no distinct SASS body. The promoted production
2,416-word encoding remains the canonical reference at SHA-256
`46e3f218deb005bb3feea4e23dc57bde7aa3695a756beb27754bc9847a48b602`.

Finite exhaustive coverage spans 254 E4M3 codes, four packed byte positions,
both matrices, and requested caps 512/1024/1536/4096. Full 256-code coverage,
the two-execution/replay race signature at all four caps, and isolated signed
NaN class/sign coverage pass. Frozen actual-checkpoint and same-bank-stress
cap-1024 cells each report 0/10,240 QKV, 0/6,144 Z, and 0/16,384 replay
mismatches, finite outputs, and intact output guards. Input preservation was
checked independently in the exhaustive cap-512 contract, not repeated in
the two frozen performance cells.

The frozen candidate ELF is 5,298,184 bytes, SHA-256
`80898ecdb7b9bbb2264f9a64f68dda902ee33f5473e48556dee5d0fd98e6acaf`,
and Build ID `e1096bfdb5957ca7195b6168d13ae7af82149ad1`. One same-binary process
used ten warmups per route, 80 logical QKV/Z pairs per timed pass, and five
`B-C-C-B` rounds per fixture. The selection statistic is the median of ten
baseline pass means divided by the median of ten candidate pass means:

| Fixture | Production pass median | Candidate pass median | Frozen speedup | Gate | Round behavior | Result |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| Actual layer-0 checkpoint | 0.467083 ms | 0.464898 ms | 1.0047x | 1.005x | 5/5 improve | fail |
| Same-bank stress | 0.432375 ms | 0.432900 ms | 0.998787x | 1.000x | 5/5 regress | fail |

An initial sweep was directionally consistent at 1.0043x actual and
0.998711x stress, but the separate frozen retest is authoritative. The
expected nonzero screen exit is solely the performance-policy rejection:
source identity, resources, and correctness pass. Stop-loss ends work after
process 1, so there is no replication, production integration, full-model
oracle, end-to-end benchmark, Nsys, or NCU result. The retained logs do not
contain independent clock, thermal, power-mode, or energy readbacks.

The temporary test policy is removed and source/test blobs match base commit
`2529a317`; production stays QKV1536/Z768. The formal single-request anchor
therefore remains **109.056 ms/token and 9.169600939 token/s**, still
**9.056 ms/token** from the 100-ms stage target. Complete rounds, fixture and
artifact identities, cleanup proof, and claim limits are in the
[QKV grid-cap 1024 rejection record](metadata/qwen36-27b-fp8-m1-qkv-z-ping-pong-grid-cap-1024-rejection.json).

## Decode NVFP4 M1 gate/up dead-up shared-pair selection

The next bounded mechanism uses the Decode runner's actual dataflow rather
than another CTA-width or grid-cap change. The reference runner does not
observe the independently rounded up projection after the exact-M1 fused
gate/up boundary: the down projection overwrites that workspace before its
next read. The test-only candidate therefore preserves the production
`32x512` topology and all arithmetic boundaries, but retains the rounded gate
and up intermediates in two CTA-local `BF16[576]` arrays. A CTA barrier then
precedes the same balanced 512-thread SiLU epilogue, which publishes only the
final BF16 `SiLU(gate)*up` vector. The supplied up buffer remains poison and
untouched.

This is a selection for a new explicit **Decode-runner-only** contract. It is
not a replacement for the existing low- or high-level public APIs, whose
observable contract still requires an independently rounded up output. The
candidate, launcher, and resource query remain test-only at commit `544d695`;
production dispatch, the runner, model layout, public ABI, and formal
end-to-end result are unchanged. It is also not executor double/triple
buffering or cross-kernel overlap.

Both routes use 32 CTAs, 512 threads/CTA, 16 warps/CTA, 512 projection warps,
two active CTAs/SM, and 32 resident warps/SM. Production uses 11,328 B static
shared memory; the candidate's two local arrays add 2,304 B for a total of
13,632 B. Both report 64 registers/thread, zero local memory, and a 512-thread
maximum block. Logically, avoiding the two global intermediate writes and
reads removes `17,408 * 2 B * 2 tensors * 2 directions = 139,264 B` per layer,
or 8.5 MiB across 64 layers. These are logical bytes, not measured DRAM
transactions, and no persistent model memory is added.

The frozen Release test ELF is 5,435,464 bytes, SHA-256
`63a38ab803a41c21e3dc46b4d5e11727c45ecfa68d48281b2fabc871f9e85e3f`,
and Build ID `f3fdefe504c4da5894635880fedcc47d97b0abf7`. Inside that same binary,
production retains its established 2,704 normalized SASS words at SHA-256
`134202f948d757928aa744d20fa7064bdb28b12334659e88046a0fecd071d2ab`.
The distinct dead-up candidate also has 2,704 words, at SHA-256
`38e45f532ce63c539d8e735699c146b046967c16ead6596b402045db6a228b5d`,
with no `LDL` or `STL` rows.

Actual layer-0 checkpoint and same-bank-stress fixtures each report 0/5,120
residual, 0/17,408 final-gate, and 0/22,528 replay mismatches, finite published
outputs, intact guards, preserved seven inputs, and an untouched dead-up
buffer. The signed Inf/NaN fixture preserves residual/final bits and class/sign
for 17,408/17,408 published NaNs. Candidate and production capture as distinct
single-node `32x512` Graphs with zero dynamic shared memory; all nine
representative invalid launches enqueue zero nodes, and a null resource query
is rejected. The Graph gate at this selection stage proves capture topology
with aligned sentinel pointers; real-buffer instantiate/replay and the full
invalid matrix remain production-integration gates.

Three independent clean processes of the frozen binary use ten warmups per
route, five 64-launch `B-C-C-B` rounds per fixture, and the median paired-round
speedup as the selection statistic:

| Process | Actual P/C pass medians | Actual paired median | Stress P/C pass medians | Stress paired median | Result |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.602499 / 0.596105 ms | 1.01041x | 0.607264 / 0.601584 ms | 1.00929x | pass |
| 2 | 0.599853 / 0.593660 ms | 1.01034x | 0.604504 / 0.598810 ms | 1.00940x | pass |
| 3 | 0.597774 / 0.592036 ms | 1.00975x | 0.603131 / 0.597697 ms | 1.00932x | pass |
| Cross-process median | 0.599853 / 0.593660 ms | **1.01034x** | 0.604504 / 0.598810 ms | **1.00932x** | pass |

All 30 clean paired rounds improve. Actual round speedups span
1.00503x–1.01126x, and stress spans 1.00603x–1.01076x. One earlier complete
process is deliberately excluded from those statistics: an unrelated,
already-promoted 64x256-versus-32x512 sentinel suffered one 0.966306x round
after a 0.677115-ms production pass, so the whole suite exited nonzero. The
dead-up cells in that excluded log still passed all ten rounds; the replacement
reran the complete binary and exited zero. Both logs and hashes remain in the
[selection record](metadata/qwen36-27b-nvfp4-m1-gate-up-dead-up-shared-pair-selection.json),
so no dead-up regression is silently omitted.

The cross-process median actual pass delta is 0.006193 ms/layer. Multiplying
by 64 gives **0.396352 ms/token**, with a 0.367232–0.409216-ms/token process
range. That is only an isolated arithmetic projection, equal to 4.376678% of
the remaining 9.056-ms stage gap. Applying it to the old anchor would give
108.659648 ms/token and 9.203048 token/s as planning normalization only; those
values were not measured. The formal achieved result remains **109.056
ms/token and 9.169600939 token/s**.

The immediate production gate is narrow: add an internal runner-only API while
leaving the generic double-output route and every fallback intact; prove the
workspace is dead until down overwrites it; run real-buffer Graph replay and
the complete invalid matrix; then run Release/default/focused CTests, pinned
C1/C8/C16/C32 model oracles, P19/C32/max26 `B1-C1-C2-B2`, and a fresh Decode
Nsys closure. Only that final evidence can update the formal anchor or begin
the larger Prefill optimization program.

## Decode NVFP4 M1 gate/up dead-up production promotion

Commit `2dbd832` promotes the selected mechanism through two explicit
runner-only APIs without weakening the generic double-output contract. Only
the exact aligned SM87 NVFP4 `[17408,5120]` post-attention route keeps the
independently BF16-rounded gate/up pair in CTA-local `BF16[576]` arrays and
leaves `up_workspace` untouched. The generic low- and high-level APIs still
publish rounded up; every near-miss, unaligned, BF16, FP8, reference, or other
fallback follows the old chain and may write the workspace. In
`ReferenceRunner::step()`, no consumer observes that workspace between the
fused boundary and the following same-stream down projection, which
overwrites it before trace or any later read. C2–C32 prefix-tile routes are
unchanged; C1 finish-prefill shares the ordinary runner step.

Validation commit `798582c` expands the new low-level contract to a complete
54-case pre-enqueue matrix: 10 null pointers, six scalar predicates, three
shapes, eight alignments, 24 writable/read-only or writable/writable aliases,
and three overflowing address ranges. Every case returns
`cudaErrorInvalidValue` and captures zero Graph nodes. The full Release CTest
run reports 51 passes, zero failures, and five configured skips out of 56;
the four skipped model-dependent chunk cases were run directly against the
pinned checkpoint at C1/C8/C16/C32. All four match 19 prompt IDs, 26 generated
IDs, exact text, `im_end`, and 44 steps, with the existing 64 FP8 sidecars and
no fallback.

The final production microbenchmark preserves the public and dead-up SASS
identities selected earlier: both have 2,704 normalized words, at SHA-256
`134202f948d757928aa744d20fa7064bdb28b12334659e88046a0fecd071d2ab`
and
`38e45f532ce63c539d8e735699c146b046967c16ead6596b402045db6a228b5d`
respectively. The dead-up kernel reports 64 registers/thread, 13,632 B static
shared memory, zero local memory, 512 maximum threads/block, and two active
CTAs/SM. Aligned sentinel captures prove distinct one-node `32x512` launch
topologies. Separately, real checkpoint and stress buffers complete CUDA Graph
capture, instantiate, and replay with 0/5,120 residual, 0/17,408 final, and
0/22,528 replay mismatches, intact guards, all seven inputs preserved, and
different direct/replay workspace poisons left untouched. The real replay
itself was not queried for node parameters, and signed Inf/NaN coverage is
direct-launch only.

One final fixed-clock, same-binary `B-C-C-B` screen retains ten warmups and
five 64-launch rounds per fixture:

| Fixture | Public pass median | Dead-up pass median | Paired median | Every round non-regress | Result |
| --- | ---: | ---: | ---: | --- | --- |
| Actual layer-0 checkpoint | 0.599918 ms | 0.593990 ms | 1.00985x | yes | pass |
| Same-bank stress | 0.604584 ms | 0.599014 ms | 1.00921x | yes | pass |

The actual pass-median delta projects to 0.379392 ms/token across 64 layers,
but that multiplication is phase-local evidence only. The formal result comes
from four independent complete-engine processes in strict `B1-C1-C2-B2`
order:

| Process | Subsequent token | Decode after first | TTFT | Total generation |
| --- | ---: | ---: | ---: | ---: |
| B1 | 109.074 ms | 2,726.544 ms | 428.348 ms | 3,154.910 ms |
| C1 | 108.782 ms | 2,719.279 ms | 428.538 ms | 3,147.832 ms |
| C2 | 108.557 ms | 2,713.763 ms | 428.134 ms | 3,141.927 ms |
| B2 | 109.033 ms | 2,725.636 ms | 428.210 ms | 3,153.845 ms |

The mirrored baseline/candidate hot medians are 109.0535 and **108.6695
ms/token**, a 0.384-ms same-run reduction or 1.00353365x speedup. Both mirrored
pairs improve independently by 0.292 and 0.476 ms/token. Decode-after-first
falls from 2,726.090 to 2,716.521 ms, while TTFT changes only from 428.279 to
428.336 ms. All four logs contain 163 lines, five samples of exactly 25 later
tokens, no persistent device-memory drop, and the same 29-row functional
canonical SHA-256
`f66b837ed8f5b17f1307b424062c432c7ed02befdc97d99147f34c7675decee8`,
which also matches the historical golden.

Fresh Nsys evidence closes all 25 Decode ranges over 10,925 distinct kernel
rows, exactly 437 per range. The generation range closes over 12,997 prefix,
finish, and Decode leaves with no missing, extra, or duplicate rows. The new
exact symbol appears 1,600 times—64 in every Decode step—while the retired
full-output boundary symbol appears zero times. It uses `32x512`, 64
registers/thread, 13,632 B static shared memory, and zero local memory per
thread. Decode raw and interval-union time are both 2,705.937632 ms on stream
18; overlap is zero, and the 2,721.827360-ms span contains only 15.889728 ms
idle (0.583789%). Thus this win is dead-publication removal, not hidden
multi-stream overlap or system buffering.

Because the measured 108.6695-ms candidate is 0.3865 ms below the previous
109.056-ms formal result, the new achieved single-request anchor is
**108.6695 ms/token and 9.202214053 token/s**. The remaining stage gap is
8.6695 ms/token or 0.797785947 token/s; reaching 100 ms requires another
7.977859% latency reduction. The separate 108.672-ms planning normalization
is not an achieved value. Complete machine-readable evidence is in the
[production benchmark](metadata/qwen36-27b-nvfp4-m1-gate-up-dead-up-production-benchmark.json)
and
[post-promotion Decode profile](metadata/qwen36-27b-post-gate-up-dead-up-decode-phase-profile.json).

## Decode FP8 QKV/Z plus BF16 A/B tail-composite production promotion

Commit `f64abc2` promotes the candidate selected in `8e84265` for the exact
aligned SM87 M1 linear-attention input group: FP8 QKV `[10240,5120]`, FP8 Z
`[6144,5120]`, and BF16 A/B `[48,5120]`. The physical `1536x256` QKV/Z launch
and its arithmetic are unchanged. Twenty-four otherwise-light tail CTAs each
compute two adjacent rows from both A and B while sharing the activation load,
so the established QKV/Z kernel and BF16 pair kernel become one launch. M2–M32
prefix tiles are unchanged; ordinary M1 Decode and finish-prefill use the new
route.

The high-level wrapper validates each projection and all six cross-projection
pairs before checking fusion-only alignment. This covers four writable
outputs against four matrices, the activation, four FP8 device scalars, and
one another. Invalid payload, BF16 alignment, range, or alias cases return
`cudaErrorInvalidValue` with zero Graph nodes; only a semantically safe
fusion alignment miss returns `cudaErrorNotSupported` for the ordered old
fallback. The full Release suite reports 51 passes, five configured skips,
and zero failures out of 56. Pinned C1/C8/C16/C32 model oracles all match 19
prompt IDs, 26 generated IDs, exact text, `im_end`, and 44 steps; C1 and C32
were rerun after final wrapper hardening.

The production composite has 64 registers/thread, 1,280 B static shared
memory, zero local memory, 256 threads/block, and four active CTAs/SM. Its
3,504 normalized 64-bit SASS words hash to
`c7b810b4effa7223274c28e7569f9218f451f9c96bce24ba4b179b1c95596a8d`,
identical to the selected test route. Actual checkpoint, same-bank stress,
Graph replay, canary, resource, and invalid-call gates pass bitwise. Three
independent actual-checkpoint screens project savings of 0.566158, 0.586158,
and 0.609119 ms/token across 48 linear-attention layers; the independent
stress screen projects 0.556704 ms/token.

The formal fixed-clock gate uses four independently loaded engines in strict
`B1-C1-C2-B2` order, with one warmup and five measured P19/C32/max26
generations per process:

| Process | Subsequent token | Decode after first | TTFT | Total generation |
| --- | ---: | ---: | ---: | ---: |
| B1 | 108.820 ms | 2,720.705 ms | 428.238 ms | 3,148.944 ms |
| C1 | 108.337 ms | 2,708.661 ms | 427.693 ms | 3,136.407 ms |
| C2 | 108.192 ms | 2,704.985 ms | 427.499 ms | 3,132.496 ms |
| B2 | 108.964 ms | 2,724.023 ms | 428.134 ms | 3,152.184 ms |

The mirrored baseline/candidate hot medians are 108.892 and **108.2645
ms/token**, a 0.6275-ms reduction or 1.005795990x speedup. Both pairs improve
independently by 0.483 and 0.772 ms/token. Decode-after-first falls from
2,722.364 to 2,706.823 ms. Prefix execution is effectively unchanged
(319.5835 versus 319.5845 ms), while the M1 finish-prefill boundary improves
from 108.6000 to 108.0035 ms. All four logs share the same 21-row functional
SHA-256 `99e91d052157ac48608a4c6e46f62536db0f8376f1ce37276abcdbbaf3c92841`.

Fresh Nsys closes all 25 Decode ranges over 9,725 kernel rows, exactly 389 per
range, versus the preceding 10,925/437. The composite appears exactly 1,200
times (`48x25`); both predecessor symbols appear zero times. Finish-prefill
falls from 437 to 389 rows, and the complete generation from 12,997 to 11,749.
Execution remains on one stream with zero kernel overlap. This establishes a
deterministic one-launch-per-linear-layer reduction, not hidden concurrency or
system buffering.

The achieved single-request anchor is therefore **108.2645 ms/token and
9.236638048 token/s**, 0.405 ms/token below the prior 108.6695-ms formal
anchor. The remaining stage gap is 8.2645 ms/token or 0.763361952 token/s;
reaching 100 ms requires another 7.633620% latency reduction. This is hot
single-request Decode on one fixed-clock Orin, excludes model load, and is not
a Prefill, multi-request throughput, tail-latency, power, or energy claim.
Complete evidence is in the
[production benchmark](metadata/qwen36-27b-fp8-m1-qkv-z-bf16-ab-tail-composite-production-benchmark.json).

## Rejected Decode output-projection residual handoff

Commit d42963a adds a bounded, test-only two-kernel comparison without
changing runtime dispatch. The baseline publishes the independently rounded
FP8 attention output, then the selected 32x512 residual/RMSNorm/dead-up
gate-up kernel recomputes and publishes the residual. The candidate instead
keeps raw=BF16_RNE(projection) in the output-projection kernel, publishes
BF16_RNE(hidden0+raw), and lets a second test-only gate-up kernel consume that
pre-rounded residual. Both routes remain exactly two launches; generic
projection behavior, runner code, M2–M32 Prefill, and every production symbol
are unchanged.

The candidate preserves the required arithmetic boundary. On the pinned
layer-0 output, gate, and up weights it reports zero mismatches over 5,120
rounded residual values and 17,408 final gate values, plus zero replay
mismatches over all 22,528 outputs. Both dead-up workspaces remain untouched,
all output guards pass, and all outputs are finite. Resource limits also pass:
the output epilogue remains at 64 registers/thread, 1,152 B shared, zero local,
and four active 256-thread CTAs/SM; the pre-rounded gate-up kernel remains at
64 registers/thread, 13,632 B shared, zero local, and two active
512-thread CTAs/SM.

The first fixed-clock actual-checkpoint process uses ten warmups and five
64-chain rounds in strict B1-C1-C2-B2 order:

| Round | B1 | C1 | C2 | B2 | Paired delta | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.762578 ms | 0.766447 ms | 0.766235 ms | 0.762486 ms | -0.00380945 ms | fail |
| 2 | 0.762338 ms | 0.766020 ms | 0.765776 ms | 0.762123 ms | -0.00366724 ms | fail |
| 3 | 0.762399 ms | 0.766737 ms | 0.767038 ms | 0.761948 ms | -0.00471348 ms | fail |
| 4 | 0.763025 ms | 0.765953 ms | 0.766252 ms | 0.762098 ms | -0.00354099 ms | fail |
| 5 | 0.761992 ms | 0.766145 ms | 0.766361 ms | 0.763161 ms | -0.00367701 ms | fail |

The baseline/candidate pass medians are 0.762369 and 0.766243 ms/layer,
respectively: **0.994944x**, with a -0.00367701-ms median delta. Across 64
layers that is a projected **0.235329-ms/token regression**, not the required
0.5-ms/token saving. An earlier console-only diagnostic also regressed in all
five rounds at 0.993985x, so the retained formal result is not an isolated
reversal.

The mandatory first-process stop-loss therefore rejects this mechanism before
stress, signed nonfinite, Graph, invalid-contract, model-oracle, end-to-end, or
Nsys work. The test-only probe remains available for reproducibility, but no
production path calls it. The achieved anchor remains **108.2645 ms/token and
9.236638048 token/s**. The next bounded Decode experiment changes only the
cache policy of one-pass packed-weight/block-scale loads in the dominant
NVFP4 projections. Complete evidence is in the
[rejection record](metadata/qwen36-27b-decode-o-proj-prerounded-residual-chain-rejection.json).

## Rejected Decode gate/up cache-global loads

Commit 2623339 adds a test-only twin of the production runner-only `32x512`
residual/RMSNorm/gate/up/SiLU dead-up kernel. The candidate changes only the
one-pass packed-weight U32 and block-scale U8 loads from the compiler-default
cache policy to cache-global (`.cg`); arithmetic, BF16 boundaries, shared
staging, topology, launch count, stream, runtime dispatch, and Prefill remain
unchanged.

The static isolation gate passes. Production remains byte-identical at 2,704
encoding words and SASS SHA-256 `38e45f53...228b5d`. The candidate also has
2,704 words and changes exactly eight `LDG.E` instructions to
`LDG.E.STRONG.GPU` plus four `LDG.E.U8` instructions to
`LDG.E.U8.STRONG.GPU`; all other global/shared load counts match and there are
no local loads or stores. Both kernels use 64 registers/thread, 13,632 B
shared, zero local/stack, 512 threads, and two active CTAs/SM.

Actual checkpoint correctness reports zero residual mismatches over 5,120
elements, zero gate mismatches over 17,408 elements, and zero candidate replay
mismatches over all 22,528 outputs. Outputs are finite, all guards pass, and
the baseline, candidate, and replay dead-up workspaces remain untouched.

The first fixed-clock process uses ten warmups and five 64-kernel rounds in
strict B1-C1-C2-B2 order:

| Round | B1 | C1 | C2 | B2 | Paired delta | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 0.596318 ms | 0.600479 ms | 0.610862 ms | 0.608900 ms | -0.00306123 ms | fail |
| 2 | 0.600843 ms | 0.599312 ms | 0.596852 ms | 0.596765 ms | +0.000722229 ms | fail |
| 3 | 0.596629 ms | 0.596790 ms | 0.597003 ms | 0.596520 ms | -0.000321746 ms | fail |
| 4 | 0.596133 ms | 0.596524 ms | 0.596677 ms | 0.596116 ms | -0.000476003 ms | fail |
| 5 | 0.595873 ms | 0.596580 ms | 0.596425 ms | 0.595859 ms | -0.000636578 ms | fail |

The baseline/candidate pass medians are 0.596419 and 0.596821 ms/layer:
**0.999327x**, with a -0.000476003-ms median paired delta and a projected
**0.0304642-ms/token regression** over 64 layers. No round improves in both
directions. First-process stop-loss therefore skips stress, nonfinite, Graph,
invalid-contract, model, end-to-end, and profiler work. Production remains at
**108.2645 ms/token and 9.236638048 token/s**. The next independent cell tests
the evict-first streaming (`.cs`) policy rather than repeating `.cg`.
Complete evidence is in the
[rejection record](metadata/qwen36-27b-decode-gate-up-cache-global-rejection.json).

## Selected test-only Decode gate/up streaming loads

Commit `4aa74f2` closes the bounded follow-up to the rejected `.cg` screen. The
test-only candidate changes only the one-pass packed-weight U32 and block-scale
U8 loads in the production-shaped `32x512` residual/RMSNorm/gate/up/SiLU
dead-up kernel to evict-first streaming (`.cs`). Runtime dispatch, the
production symbol, arithmetic and BF16 boundaries, launch count, stream, and
Prefill are unchanged.

Static isolation passes. The frozen production kernel remains 2,704 normalized
64-bit encoding words with SASS SHA-256
`38e45f532ce63c539d8e735699c146b046967c16ead6596b402045db6a228b5d`.
The candidate also has 2,704 words and hashes to
`2176900d654384440b638338cbbfe7e3ae2a02d59643d1ce30aa882fb47d4ae3`;
exactly eight `LDG.E` instructions become `LDG.E.EF`, and four `LDG.E.U8`
instructions become `LDG.E.EF.U8`, while all 25 `LDG.E.U16` instructions are
unchanged. Both routes retain 64 registers/thread, 13,632 B static shared
memory, zero local memory, 512 threads/block, and two active CTAs/SM.

Actual-checkpoint and same-bank-stress finite gates each report 0/5,120
residual, 0/17,408 final-gate, and 0/22,528 CUDA Graph replay mismatches. The
candidate preserves all seven inputs, leaves the dead-up buffer under distinct
direct/replay poisons untouched, and keeps every output guard intact. The
signed Inf/NaN direct-launch gate reports 0/5,120 residual mismatches,
0/17,408 final-gate mismatches, and correct class/sign for all 17,408 NaN
outputs. The focused CTest also validates distinct one-node `32x512`, zero
dynamic-shared Graph topology; the real checkpoint and stress paths capture,
instantiate, and replay successfully.

The formal same-binary closeout primes both routes, then uses ten warmups and
five 64-launch paired rounds per fixture, alternating `B-C-C-B` and `C-B-B-C`
order:

| Fixture | Baseline pass median | Candidate pass median | Paired median | Every round non-regress | Result |
| --- | ---: | ---: | ---: | --- | --- |
| Actual layer-0 checkpoint | 0.589491 ms | 0.584337 ms | **1.00895x** | yes | pass |
| Same-bank stress | 0.595607 ms | 0.590388 ms | **1.00882x** | yes | pass |

All ten closeout rounds are non-regressing. The selected actual-checkpoint
paired delta is 0.00523198 ms/layer and projects to 0.334846 ms/token over 64
layers, above the 0.25-ms phase-local threshold. The exact test binary hashes
to `db1dc7d50068f5203a30103d3a70e75baa7cbf39ee91640002b3c3ac28b688af`;
the independent closeout process exits zero, and its log is
`/tmp/q3x-decode-cache-policy.8mr8So/gate-up-cs-closeout.log` with SHA-256
`12d96816c6e0a454b519132d9b02d41782ce9f7b6e32d22a991094ac078170be`.

This is a **test-only selection**, not a production or end-to-end result. No
runtime path calls the candidate, so the achieved production anchor remains
**108.2645 ms/token and 9.236638048 token/s**. Production integration, full
model oracles, formal end-to-end benchmarking, and profiler closure remain
required before changing that anchor. Complete evidence is in the
[selection record](metadata/qwen36-27b-decode-gate-up-streaming-selection.json).

## Selected test-only Decode down streaming loads and gate+down projection

Commits `f12b598` and `527f9a7` add and close a test-only twin of the
production `32x512` cooperative NVFP4 down/residual/RMSNorm kernel. The source
change applies evict-first streaming (`.cs`) only to the one-pass packed-weight
U32 and block-scale U8 projection loads. Arithmetic, BF16 publication
boundaries, residual/RMSNorm work, cooperative topology, output addresses used
for timed A/B, launch count, stream, runtime dispatch, and Prefill remain
unchanged. Correctness uses disjoint guarded outputs; timing deliberately uses
the same three candidate output addresses for both routes to remove allocator
and output-cache placement bias.

All three independent same-binary processes pass actual-checkpoint and
same-bank-stress finite comparisons with zero raw, residual, or normalized
mismatches over 5,120 elements and zero CUDA Graph replay mismatches over all
15,360 outputs. Each process also passes distinct one-node `32x512`, zero
dynamic-shared Graph topology, complete input preservation, output guards,
and split residual/norm-weight signed Inf/NaN class-and-sign gates. Both routes
use 64 registers/thread, 35,904 B static shared memory, zero local memory, 512
threads/block, and two active CTAs/SM.

The final binary confirms the intended load policy: eight production `LDG.E`
and four `LDG.E.U8` operations become eight `LDG.E.EF` and four
`LDG.E.EF.U8`; the eight `LDG.E.64` and 37 `LDG.E.U16` loads remain. Static
identity is not same-length, however. Production has 1,280 instructions/2,560
encoding words with SHA-256
`fde55ae44d31797fe93f31a682c74bb6515133f75a68c7189879ca9057055ad8`,
whereas the candidate has 1,288/2,576 and hashes to
`46be0b08c4dd8660928cd33e2696ca5d703326d77b88f5527d03e1ed36771da6`.
Seven extra instructions are unreachable tail-alignment NOPs; reachable code
also has a net one-instruction zero-materialization/scheduling delta. The
arithmetic, store, and synchronization census is preserved, but the measured
result necessarily includes this compiler scheduling artifact as well as the
target `.cs` qualifiers.

Each process uses ten warmups, one unmeasured `B-C-C-B` prime, and five
64-launch rounds per fixture in alternating `B-C-C-B`/`C-B-B-C` order:

| Process | Actual paired median | Stress paired median | Actual delta/layer | Projected 64-layer delta | Positive rounds |
| --- | ---: | ---: | ---: | ---: | ---: |
| Closeout | **1.01538x** | 1.01721x | 0.00494376 ms | **0.316401 ms/token** | 10/10 |
| Replication 1 | **1.01762x** | 1.01887x | 0.00537601 ms | **0.344065 ms/token** | 10/10 |
| Replication 2 | **1.01386x** | 1.01344x | 0.00429049 ms | **0.274591 ms/token** | 10/10 |

Thus all 30/30 measured actual-plus-stress rounds have positive paired deltas;
every process clears the 1.005x actual, 1.0x stress/every-round, and
0.169-ms/token projected-absolute gates. The median down projection is
**0.316401 ms/token**. Added to the separately selected gate/up projection of
0.334846 ms/token, the planning sum is **0.651247 ms/token**; using the lowest
replicated down projection instead gives a conservative **0.609437
ms/token**. These sums are arithmetic phase-local projections, not a measured
combined-engine or end-to-end result.

The three logs hash respectively to
`696e837a70fa419d3ed59e0b3cb1b2bd33c4455aeb697cfac23970cfca897f83`,
`7d98cece3ccd19037f112c8c21795410624ef465c8b768d10b7c5ee1f99b863f`,
and `d03811c1c21e0ef09faec302c0ab04a37f7d728ec13723d3db471cfc143650e0`.
Their exact test binary hashes to
`c785b99f98fa00371357e48997a8e6f2a6870f42491bdc2e885985a28a7eec79`.

Both down and gate/up remain **test-only selections**: no production dispatch
or Prefill path calls either candidate, and the combined projection has not
been benchmarked end to end. The achieved production anchor therefore remains
**108.2645 ms/token and 9.236638048 token/s**. Production integration, model
oracles, independent-process end-to-end A/B, and profiler closure remain
required before changing it. Complete evidence is in the
[selection record](metadata/qwen36-27b-decode-down-streaming-selection.json).

## Decode gate/up and down streaming-load production promotion

Production commit `9aab3c3` promotes the selected evict-first streaming
(`.cs`) policy for the one-pass packed-weight and block-scale loads in both
exact M1 Decode NVFP4 projections: residual/norm/gate/up/SiLU dead-up
`[17408,5120]` and cooperative down/residual/RMSNorm `[5120,17408]`.
Arithmetic, BF16 publication boundaries, `32x512` topology, launch count,
workspace layout, stream policy, and all non-exact fallbacks remain unchanged.
Coverage fix `072ea5e` restores full finite, CUDA Graph replay, and signed
Inf/NaN exercise of the explicit compiler-default rollback launchers in the
production test closure. Captured Graph identity and resource gates prove
`public == selected .cs != default rollback` for both kernels; the rollback
paths are test-only.

The final production closeouts compare default rollback B against the public
selected route C, with timing routes writing the same output addresses:

| Exact route | Actual B/C pass medians | Actual paired median | Stress paired median | Projected 64-layer delta | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Gate/up dead-up | 0.594817 / 0.589816 ms | 1.00843x | 1.00860x | 0.318230 ms/token | pass |
| Down/residual/norm | 0.315761 / 0.312179 ms | 1.00922x | 1.01262x | 0.184895 ms/token | pass |
| Arithmetic sum | — | — | — | **0.503125 ms/token** | diagnostic only |

The sum is an isolated microbenchmark projection, not a whole-engine claim.
Static extraction retains the selected SASS identities. Gate/up default and
selected each contain 2,704 normalized 64-bit words, hashing respectively to
`38e45f532ce63c539d8e735699c146b046967c16ead6596b402045db6a228b5d`
and
`2176900d654384440b638338cbbfe7e3ae2a02d59643d1ce30aa882fb47d4ae3`.
Down default contains 2,560 words at
`fde55ae44d31797fe93f31a682c74bb6515133f75a68c7189879ca9057055ad8`;
the selected down route retains the known compiler scheduling artifact at
2,576 words and
`46be0b08c4dd8660928cd33e2696ca5d703326d77b88f5527d03e1ed36771da6`.

The default device test passes. Direct pinned-model oracles at C1, C8, C16,
and C32 all match 19 prompt IDs, 26 generated IDs, exact text, `im_end`, and
44 steps. The formal fixed-clock gate then uses four independently loaded
engines in strict `B1-C1-C2-B2` order, one warmup and five measured
P19/C32/max26 generations per process:

| Process | Subsequent token | Decode after first | TTFT | Total generation |
| --- | ---: | ---: | ---: | ---: |
| B1 | 108.301 ms | 2,707.365 ms | 427.540 ms | 3,134.839 ms |
| C1 | 107.919 ms | 2,696.961 ms | 426.966 ms | 3,124.021 ms |
| C2 | 107.860 ms | 2,696.277 ms | 426.954 ms | 3,123.198 ms |
| B2 | 108.138 ms | 2,702.793 ms | 427.258 ms | 3,130.060 ms |

The mirrored baseline/candidate subsequent-token medians are 108.219500 and
**107.889500 ms/token**, a 0.330000-ms reduction or 1.003058685x speedup.
Both independent pairs improve (`0.382` and `0.278` ms). Decode-after-first
falls from 2,705.079 to 2,696.619 ms, TTFT from 427.399 to 426.960 ms, and
total generation from 3,132.4495 to 3,123.6095 ms. Every process reports
`status=ok`, five measured samples, 125 pooled subsequent-token intervals,
25 values in sample 4, and no persistent device-memory drop. All four retain
the exact 29-row functional canonical SHA-256
`f66b837ed8f5b17f1307b424062c432c7ed02befdc97d99147f34c7675decee8`.

Matched baseline and candidate Nsys captures each close 25 Decode ranges over
9,725 kernel rows (389 per range) and 11,749 generation rows. Both use one
stream, raw time equals interval union, and kernel overlap is zero. Baseline
contains the compiler-default gate/up and down kernels exactly 1,600 times
each and no selected kernels; candidate contains the selected streaming
kernels exactly 1,600 times each and no default kernels. Both console logs
retain the exact P19/26-token text, `im_end`, and 44-step result. This proves
dispatch replacement, but the separately profiled timings do not replace the
unprofiled mirrored end-to-end gate.

The previous active anchor was **108.2645 ms/token and 9.236638048 token/s**.
The new achieved hot single-request anchor is therefore **107.889500 ms/token
and 9.268742556 token/s**. It remains **7.889500 ms/token and 0.731257444
token/s short** of the 100-ms/10-token/s stage target, so the target is not
met. This remains a serial single-request path: no double or triple buffering,
additional stream, or cross-kernel overlap was added. Dedicated Prefill
work for the next optimization phase has not begun; this promotion changes no
Prefill tile path. The next priority remains bounded Decode work until the
stage target is reached. Complete hashes, commands, gates, and scope limits
are in the
[production benchmark](metadata/qwen36-27b-decode-streaming-production-benchmark.json).

## Rejected Decode NVFP4 6-bit block-scale sidecar

The next bounded Decode screen kept NVFP4 packed weights canonical and
compressed only E4M3FN block-scale codes into a test-only lossless sidecar.
Each four-row/K512 tile stores 128 `scale - base` values as 6-bit deltas in
96 bytes (24 aligned U32 words), versus 128 canonical U8 scale bytes. Gate,
up, and down layer-0 sidecars are each 4,177,920 bytes rather than 5,570,560
bytes. An all-tensor feasibility scan finds all 64 gate and 64 up projections
eligible and 53 of 64 down projections eligible; the other 11 down layers
retain canonical scales. Keeping all 181 eligible sidecars beside the
canonical payload would add 756,203,520 bytes (0.704269 GiB), because Prefill
and fallbacks still require the canonical representation.

The host packer covers all 64 deltas, round-trips and repacks byte-identically,
rejects a span of 64, and pins the three actual layer-0 sidecars by SHA-256.
Actual-checkpoint and same-bank-stress direct/replay comparisons are bitwise
exact for gate/up and all three down outputs. The down route also passes two
split signed Inf/NaN fixtures, guarded input/sidecar preservation, eight
fail-before-enqueue cases, and distinct one-node `32x512` CUDA Graph capture.
The candidate retains two active CTAs/SM and zero local memory: gate/up uses
63 registers and 13,632 B shared, while down uses 64 registers and 35,904 B.

One early exploratory process deadlocked before producing output. The cause
was a full-mask warp shuffle inside a lane-dependent branch; commit `7612bd5`
makes the shuffle warp-uniform. The zero-byte interrupted run is not a timing
sample. A completed post-fix confirmation also predates the hardened down-only
gate and is excluded; the formal three-process series starts afterward.

Each formal process compares the current public `.cs` baseline with the
test-only scale6 route in the same binary and stream. It uses ten warmups, an
unmeasured `B-C-C-B` prime, and five 64-launch rounds for both actual and
same-bank fixtures, alternating `B-C-C-B` and `C-B-B-C` order:

| Process | Gate actual/stress | Down actual/stress | Down actual delta/layer | Projected 53-layer down delta | Down gate |
| --- | ---: | ---: | ---: | ---: | --- |
| 1 | 0.964139x / 0.965743x | 1.02671x / 1.02637x | 0.00819799 ms | 0.434494 ms/token | pass |
| 2 | 0.966421x / 0.967750x | 1.01646x / 1.01603x | 0.00505701 ms | 0.268021 ms/token | pass |
| 3 | 0.963405x / 0.965035x | 1.01230x / 1.01172x | 0.00377023 ms | **0.199822 ms/token** | **fail** |

Gate/up regresses in all 30 formal actual-plus-stress rounds; its unpack work
more than offsets the 25% block-scale-byte reduction. Down improves in all 30
rounds and clears its relative gates, but production selection requires all
three independent processes to project at least 0.25 ms/token over the 53
eligible layers. Process 3 reaches only 0.199822 ms/token, so down-only is also
rejected. The mixed gate/up-plus-down projections are negative in every
process (-0.970036, -1.04729, and -1.23270 ms/token).

No candidate route entered production, and no candidate end-to-end or Nsys
claim is made. The formal anchor remains **107.889500 ms/token and
9.268742556 token/s**, still 7.889500 ms/token short of the stage target.
Decode remains serial on one stream without double/triple buffering or
cross-kernel overlap. The planned next Prefill optimization stage has not
started; bounded Decode work continues until the 100-ms/token and 10-token/s
gate is met. Complete hashes, the interrupted-run exclusion, formal logs, and
claim limits are in the
[scale6 rejection record](metadata/qwen36-27b-decode-nvfp4-scale6-sidecar-rejection.json).

## Rejected Decode linear QKV/Z FP8-weight streaming loads

The next bounded Decode cell applied evict-first streaming cache policy only
to the canonical FP8 weight words in the current public exact-M1
linear-attention QKV/Z plus BF16 A/B tail composite. The test-only candidate
retains the public one-kernel `1536x256` topology, arithmetic, reduction order,
scalar QKV/Z scales, BF16 A/B path, output boundaries, launch count, and
stream. Static extraction leaves both routes at 3,504 normalized 64-bit SASS
words: production hashes to
`c7b810b4effa7223274c28e7569f9218f451f9c96bce24ba4b179b1c95596a8d`
and the candidate to
`153979c36e233fd0a5da59d80ac0e1a79d4c135ec3073c1b91241db6e8f48312`.
The only targeted SASS change is 32 `LDG.E` FP8-weight loads becoming
`LDG.E.EF`; eight activation `LDG.E.64` and forty BF16 A/B-weight
`LDG.E.U16` loads remain unchanged. Both routes use 64 registers/thread,
1,280 B static shared memory, zero local memory, and four active CTAs/SM.

The pinned layer-0 actual checkpoint and same-bank stress fixtures match the
current public outputs bitwise for QKV, Z, A, and B with intact output guards.
The synthetic candidate also captures as a distinct one-node CUDA Graph,
replays bitwise with guards, and passes the four-case fail-before-enqueue
zero-node matrix. The default Release suite reports 51 passed and five skipped
tests out of 56. This screen did not independently hash or guard-check inputs,
add a signed-nonfinite fixture, or replay the candidate Graph on the actual
checkpoint; those scope limits do not weaken the first-process stop-loss
decision.

The frozen same-binary, same-stream screen uses ten warmups, an unmeasured
`B-C-C-B` prime, and five alternating 64-chain rounds per fixture:

| Fixture | Public median | Candidate median | Paired speedup | Paired delta/layer | Projected 48-layer delta | Improving rounds | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Actual layer-0 checkpoint | 0.479171 ms | 0.475652 ms | 1.00756x | 0.00359401 ms | **0.172513 ms/token** | 5/5 | **fail absolute floor** |
| Same-bank stress | 0.445017 ms | 0.440557 ms | 1.00979x | 0.00431475 ms | — | 5/5 | pass |

The candidate is directionally positive and clears the 1.005x actual and
1.0x stress relative gates, but the actual projection misses the required
0.20-ms/token first-process absolute floor. Stop-loss therefore rejects this
isolated mechanism without running the other two formal processes or the
planned 0.25-ms/token cross-process median gate. No candidate production
integration, full-model end-to-end benchmark, or Nsys closure was attempted.
The 0.172513-ms/token projection is arithmetic only and is not an achieved
whole-engine reduction.

The formal hot P19/C32/max26 anchor remains **107.889500 ms/token and
9.268742556 token/s**, still 7.889500 ms/token and 0.731257444 token/s short of
the stage target. Decode remains serial on one stream without double/triple
buffering or cross-kernel overlap, and dedicated Prefill optimization has not
started. The frozen binary, Build ID, exact per-round values, payload hashes,
SASS identities, and retained 7,258-byte log are recorded in the
[linear QKV/Z streaming rejection record](metadata/qwen36-27b-decode-fp8-linear-qkv-streaming-rejection.json).

## Rejected Decode FP8 output-projection streaming loads

The next bounded cell kept the production `[5120,6144]` output projection's
existing 31,457,280-byte AoSoA4/preswizzled sidecar, `1024x256` topology,
arithmetic, BF16 boundary, and activation cache policy unchanged. A distinct
test-only twin applied evict-first streaming policy only to the aligned
128-bit sidecar-weight loads. No new pack, sidecar, launch, stream, or
production dispatch was introduced.

Static extraction preserves the production image at 2,048 normalized 64-bit
words with SHA-256
`2f34439fe274eda16c845582962b12a1c95d82880ee99d5e95b0b3e664c1b392`;
the candidate also has 2,048 words and hashes to
`5e7646106ff3ba428ce757efbae06fdda585dca5f76a90aff83f75e470f8ded2`.
Production has eight `LDG.E.128` sidecar loads, while the candidate has eight
`LDG.E.EF.128`; both retain eight activation `LDG.E.64` loads. Each route uses
64 registers/thread, 1,152 B static shared memory, zero stack/local memory,
and four active CTAs/SM.

Actual-checkpoint and same-bank direct comparisons are bitwise exact over all
5,120 outputs, preserve both output guards and both immutable input buffers,
and retain the pinned 31,457,280-byte payload hashes. Public and candidate
routes capture as distinct single-root, single-kernel `1024x256` CUDA Graphs;
candidate replay is exact and guarded. Eleven null, shape, alignment, and
complete-span alias cases all fail before enqueue with zero captured nodes.
The complete default Release suite reports 51 passed and five skipped tests
out of 56.

The frozen actual-checkpoint process uses ten alternating warmup pairs, an
unmeasured `B-C-C-B` prime, and five 64-launch rounds in alternating
`B-C-C-B`/`C-B-B-C` order, with both routes writing the same output address:

| Round | Order | Public pair | Candidate pair | Paired speedup | Delta/layer |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | B-C-C-B | 0.1795145 ms | 0.1824665 ms | 0.983820x | -0.00295225 ms |
| 2 | C-B-B-C | 0.1797220 ms | 0.1824020 ms | 0.985313x | -0.00267902 ms |
| 3 | B-C-C-B | 0.1803695 ms | 0.1834160 ms | 0.983390x | -0.00304651 ms |
| 4 | C-B-B-C | 0.1804410 ms | 0.1831880 ms | 0.985003x | -0.00274724 ms |
| 5 | B-C-C-B | 0.1804635 ms | 0.1832100 ms | 0.985008x | -0.00274676 ms |

The pass medians are 0.180218 ms for public and 0.183088 ms for the
candidate. The paired median is **0.985003x**, and all five rounds regress.
Its -0.00274724-ms/layer delta projects to **-0.175823 ms/token** over 64
layers: an arithmetic phase-local loss, not an end-to-end measurement. This
fails the predeclared 1.005x, every-round, and 0.10-ms/token first-process
gates. Stop-loss therefore skips stress timing, the other two processes, the
external 0.125-ms/token median gate, production integration, end-to-end
benchmarking, and Nsys closure.

The formal hot P19/C32/max26 anchor remains **107.889500 ms/token and
9.268742556 token/s**, still 7.889500 ms/token and 0.731257444 token/s short of
the stage target. Decode remains serial on one stream without double/triple
buffering or cross-kernel overlap, and dedicated Prefill optimization has not
started. Complete commands, hashes, rounds, gates, and claim limits are in the
[output-projection streaming rejection record](metadata/qwen36-27b-decode-fp8-o-proj-streaming-rejection.json).

## Rejected Decode NVFP4 LM-head streaming loads

The final isolated streaming-load cell in this sequence cloned the production
activation-staged NVFP4 LM-head `[248320,5120]` kernel as a test-only twin.
Only eight packed-weight U32 loads and four block-scale U8 loads use
evict-first policy; the eight activation `LDG.E.64` loads, 10,240-byte shared
activation staging, arithmetic, reduction order, scale2, BF16-RNE boundary,
and `64x256` topology remain unchanged. The 715,161,600-byte encoded payload
is the existing checkpoint representation, not a new sidecar.

Production retains 1,584 normalized 64-bit SASS words and SHA-256
`9896967392032afdf31fc3593b777323225d2689d4070d3bdb7a5204204fc855`.
The candidate also has 1,584 words and hashes to
`a1bc4c089436377dea1868c8db6f9dcaf550e115e5d16a8513971816357e6234`.
Its target loads become eight `LDG.E.EF` plus four `LDG.E.EF.U8`, while both
routes retain eight activation `LDG.E.64` loads. Each uses 64 registers/thread,
11,328 B static shared memory, zero stack/local memory, and four CTAs/SM.

Bounded checkpoint-like, same-bank, and signed-NaN fixtures pass bitwise,
classification, finite, and guard gates. Public and direct production Graph
nodes are identical and distinct from the candidate; an intentionally odd
block-scale pointer remains a valid positive contract. Seventeen shape, null,
scale2, alignment, alias, and overflow cases fail before enqueue with a clean
stream. On the pinned shard-3 payload, all 248,320 direct and replay outputs
match bitwise, are finite and guarded, and full 635,699,200-byte weight plus
79,462,400-byte scale readbacks retain their hashes while activation and all
input guards remain exact.

The frozen process then uses ten alternating warmup pairs, an unmeasured
`B-C-C-B` prime, and five 64-launch rounds, always writing one timing output:

| Round | Order | Paired speedup | Delta/token |
| --- | --- | ---: | ---: |
| 1 | B-C-C-B | 0.993659x | -0.0280323 ms |
| 2 | C-B-B-C | 0.992887x | -0.0314708 ms |
| 3 | B-C-C-B | 0.993515x | -0.0286746 ms |
| 4 | C-B-B-C | 0.993680x | -0.0279393 ms |
| 5 | B-C-C-B | 0.993477x | -0.0288420 ms |

The public and candidate pass medians are 4.39286 and 4.42124 ms. All five
rounds regress; the paired median is **0.993515x**, a **0.0286746-ms/token
loss**. This fails the predeclared 1.005x, every-round, and 0.20-ms/token
first-process gates. Stop-loss skips stress timing, the other two processes,
the external 0.25-ms/token median gate, production integration, E2E, and Nsys.

The formal hot P19/C32/max26 anchor remains **107.889500 ms/token and
9.268742556 token/s**, still 7.889500 ms/token and 0.731257444 token/s short of
the stage target. Decode remains serial on one stream without double/triple
buffering or cross-kernel overlap, and dedicated Prefill optimization has not
started. Complete evidence is in the
[LM-head streaming rejection record](metadata/qwen36-27b-decode-nvfp4-lm-head-streaming-rejection.json).

## Decode projection palette-v2 production promotion

The production palette-v2 route combines evict-first FP8 QKV/Z weight loads
with lossless six-bit NVFP4 down-scale sidecars. The latter are admitted for
53 of 64 down projections; 11 layers retain the canonical-scale fallback.
Canonical scales remain resident for bulk Prefill and fallback, so the 53
sidecars add **221,429,760 bytes (0.206223 GiB)** rather than replacing model
storage. Their cold host-pack/upload step takes approximately **2 seconds**
and is excluded from the hot Decode number.

Pinned full-model oracles at C1, C8, C16, and C32 preserve all 19 prompt IDs,
26 generated IDs, exact text, `im_end`, and 44 steps. The fixed-clock
P19/C32/max26 end-to-end gate first runs `B1-C1-C2-B2`; a subsequent `B3-C3`
pair supplies the third independent confirmation:

| Pair | Baseline | Palette v2 | Saved |
| --- | ---: | ---: | ---: |
| B1/C1 | 107.186 ms/token | 106.666 ms/token | 0.520 ms/token |
| B2/C2 | 107.114 ms/token | 106.860 ms/token | 0.254 ms/token |
| B3/C3 | 107.157 ms/token | 106.769 ms/token | 0.388 ms/token |

The first four mirrored processes aggregate to **107.150 ms/token** for the
baseline and **106.763 ms/token** for palette v2, saving **0.387 ms/token**.
All three independent pairs improve by 0.520/0.254/0.388 ms/token, with a
**0.388-ms/token median**. The new achieved hot single-request Decode anchor
is therefore **106.763000 ms/token and 9.366540843 token/s**, replacing the
previous formal **107.314000 ms/token and 9.318448665 token/s** anchor. This
leaves **6.763000 ms/token and 0.633459157 token/s** to the 100-ms/token and
10-token/s target, so the stage target is not yet met.

These values come from full generation processes; the earlier phase-local
QKV-plus-down arithmetic sums are selection diagnostics and are not treated
as end-to-end timing. Execution remains dependency-ordered on one CUDA stream
with no system double/triple buffer or cross-kernel overlap. Bulk Prefill tile
dispatch is unchanged, while the finish-prefill M1 step shares the selected
M1 route. The next priority remains incremental Decode work toward 100
ms/token; the larger dedicated Prefill program follows that gate. Complete
commands, hashes, process results, startup cost, and claim limits are in the
[palette-v2 production benchmark](metadata/qwen36-27b-decode-projection-palette-v2-production-benchmark.json).

## Rejected Decode NVFP4 M1 Gate/Up scale-only Row-Quad AoSoA4 P1

The next bounded P1 screen changed only the block-scale layout of the current
exact-M1 `32x512` residual/RMSNorm/gate/up/SiLU dead-up route. Four adjacent
rows at one scale column are packed into one Row-Quad AoSoA4 U32, replacing
four strided U8 scale loads with one test-only U32 load. Packed weights retain
the current production streaming policy; arithmetic order, BF16 boundaries,
topology, launch count, stream, production dispatch, and Prefill remain
unchanged. This is an isolated kernel screen using pinned layer-0 checkpoint
weights with synthetic activation and residual inputs, not a full-checkpoint
activation, model-oracle, or end-to-end measurement.

The actual-checkpoint and same-bank-stress fixtures pass bitwise correctness,
finite-output, output-guard, scale-layout round-trip, and untouched dead-up
workspace checks. The formal P1 process uses ten warmups and five paired
64-launch rounds, alternating `B-C-C-B` and `C-B-B-C` order. Its frozen gates
require at least 1.02x on actual checkpoint weights, 1.00x on stress, strict
positive improvement in every round, and a projected 64-layer saving of at
least 0.50 ms/token:

| Fixture | Production pass median | Candidate pass median | Paired median | Delta/layer | Projected 64-layer delta | Improving rounds | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Layer-0 checkpoint weights + synthetic activation/residual | 0.589436 ms | 0.588577 ms | **1.00139x** | 0.000818014 ms | **0.0523529 ms/token** | 5/5 | **fail relative and absolute gates** |
| Same-bank stress | — | — | **1.0045x** | — | — | — | pass |

All five actual-checkpoint paired rounds are directionally positive, but the
gain is only 0.139%, far below the required 2%, and its 64-layer projection
reaches only about one tenth of the 0.50-ms/token floor. The scale-only
Row-Quad AoSoA4 mechanism is therefore rejected at P1. Stop-loss ends this
branch without P2, P3, full-model end-to-end, Nsys, or production-integration
work; the small phase-local projection is not an achieved Decode reduction.

No production path changes, so the formal hot single-request Decode anchor
remains **106.763000 ms/token and 9.366540843 token/s**. This result does not
alter the current production palette or the remaining distance to the
100-ms/token and 10-token/s target.

## Rejected Decode FP8 M1 O-proj CTA512 dual-worker/shared-activation P1

The next isolated P1 screen tested whether the exact-M1 `[5120,6144]` FP8
output projection benefits from doubling its CTA from the production 256
threads to 512 threads. The candidate assigns two projection workers to each
CTA and stages one shared copy of the 6,144-element BF16 activation for both
workers. It retains the production AoSoA4/preswizzled weight sidecar,
arithmetic, BF16 output boundary, launch count, stream, and output shape. This
is a real checkpoint-weight plus synthetic-activation kernel screen, not a
model-oracle or end-to-end measurement.

The formal binary is
`/tmp/q3x-o-proj-cta512-worktree/build/orin-release/q3x_sm87_weight_only_gemv_test`,
7,323,352 bytes with SHA-256
`b303d4fcf2ff5dcda3207dd02e0f5cd4245f1aac57cce36945b61a1c51e27cc7`.
Its 14-line, 3,221-byte P1 log is `/tmp/q3x-o-proj-cta512.P1.log`, with
SHA-256
`f65d42a42e035fd2858a2182964db62251ea1530aa05ad3c337283f3596720ab`.
The pinned layer-0 payload starts at byte 3,569,011,232, spans 31,457,280
bytes, and hashes to
`e49a9f770a84cfcf7c2eb60f041aa7f1af8b84f5ab6323d3b8a9151588ff2bb9`;
the equal-size AoSoA4 sidecar hashes to
`20f6875f163aefa869d7c96c436fdc63f90171793a88a0f1b3fefca55531782b`.

The resource and correctness gates pass:

| Route | Threads | Registers/thread | Static shared | Active CTAs/SM | Local memory |
| --- | ---: | ---: | ---: | ---: | ---: |
| Production | 256 | 64 | 1,152 B | 4 | 0 B |
| CTA512 candidate | 512 | 64 | 13,568 B | 2 | 0 B |

All 5,120 candidate outputs match production bitwise. Both output guards pass,
and the sidecar plus synthetic activation remain immutable. The formal first
process then uses five 64-launch paired rounds in alternating `B-C-C-B` and
`C-B-B-C` order:

| Round | Paired speedup | Paired delta/layer | Result |
| --- | ---: | ---: | --- |
| 1 | 0.913742x | -0.0166838 ms | fail |
| 2 | 0.915945x | -0.0161972 ms | fail |
| 3 | 0.914051x | -0.0166215 ms | fail |
| 4 | 0.923278x | -0.0147860 ms | fail |
| 5 | 0.924904x | -0.0143918 ms | fail |

Every round regresses. Production and candidate pass medians are 0.176767 and
0.192815 ms/layer, respectively; the paired median is **0.915945x**, with a
**-0.0161972-ms/layer** delta. Across 64 layers this projects to a
**-1.03662-ms/token delta**, a 1.03662-ms/token regression. The candidate
therefore fails all three frozen performance requirements: at least 1.02x
actual-checkpoint speedup, at least 0.30 ms/token projected saving, and strict
improvement in every round.

First-process stop-loss rejects the CTA512 dual-worker/shared-activation
mechanism. Stress timing, P2, P3, full-model validation, end-to-end timing,
and Nsys were not run. No production path changes, so the formal hot
single-request Decode anchor remains **106.763000 ms/token and 9.366540843
token/s**.

## Rejected Decode NVFP4 M1 Gate/Up exact-BF16 SiLU FP32 lookup P1

This P1 screen replaces only the production dead-up epilogue expression
`gate / (1 + expf(-gate))` with a global FP32 lookup indexed by all 16 raw bits
of the independently rounded BF16 gate. The cold initialization covers all
65,536 inputs and occupies **262,144 bytes**; initialization is outside warmup
and timing. Residual/RMSNorm, canonical NVFP4 weights and scales, both
projection phases, BF16 rounding boundaries, final multiplication by rounded
up, `32x512` topology, launch count, stream, and production dispatch are
unchanged.

The final Release binary is 7,464,992 bytes with SHA-256
`4d6e61bc198ab16980d03e79e1210ae5051f8193309440a52e00b959e643f0f5`.
Its 374-line hardened default log passes and hashes to
`a061853b352aec35a6d619fee9fbb9d493f08e80b7f2098d8480a4fc99317307`.
The authoritative 28-line focused P1 log is
`/tmp/q3x-gateup-silu-table.hardened-P1.log`, 6,632 bytes with SHA-256
`135f0ff4ec06345a168b3ae2bf6890015fcfa84625b25c88ff2a8786c2b7f266`.
The earlier `/tmp/q3x-gateup-silu-table.P1.log` is exploratory and supplies no
formal decision or anchor value. All 65,536 table FP32 values match an
independent direct device function bitwise, invalid initialization is rejected,
and the table remains immutable. For both actual-checkpoint and
same-bank-stress fixtures, residual mismatches are 0/5,120 and final-gate
mismatches are 0/17,408; outputs are finite, guards pass, all weights, scales,
residual, and norm inputs remain unchanged, and both dead-up workspaces remain
untouched.

The resource envelope is unchanged:

| Route | Grid/block | Registers/thread | Static shared | Local memory | Active CTAs/SM |
| --- | --- | ---: | ---: | ---: | ---: |
| Production streaming dead-up | `32x512` | 64 | 13,632 B | 0 B | 2 |
| FP32 SiLU table candidate | `32x512` | 64 | 13,632 B | 0 B | 2 |

The formal process uses ten warmups and five paired 64-launch rounds per
fixture, alternating `B-C-C-B` and `C-B-B-C`. Its frozen gates require at least
1.01x on actual checkpoint weights, 1.00x on stress, strict improvement in
every round, and at least 0.30 ms/token projected over 64 layers:

| Fixture | Production pass median | Candidate pass median | Paired median | Median delta/layer | Projected 64-layer delta | Regressing rounds | Result |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| Layer-0 checkpoint weights + synthetic dynamic inputs | 0.594244 ms | 0.594725 ms | **0.999793x** | -0.000123262 ms | **-0.00788879 ms/token** | 4/5 | **fail** |
| Same-bank stress | 0.598240 ms | 0.598310 ms | **1.00011x** | 0.0000650287 ms | — | 2/5 | **fail every-round gate** |

The hardened actual projection is negative and effectively noise-level: four
of five rounds reverse direction. Stress reverses in two rounds, so neither
fixture clears the full frozen contract. The pass-median difference and median
paired-round delta are distinct statistics and are not recomputed from one
another.

Static same-binary disassembly confirms the intended substitution. The
production dead-up function has 2,704 static instruction rows with
MUFU.EX2/RCP/RSQ/LDG/STG counts of 1/2/2/37/6; the table candidate has 2,624
rows and counts of 0/1/2/38/6. Thus one extra global load really does replace
the SiLU EX2/RCP work, but it provides no measurable benefit inside the much
larger bandwidth-dominated projection. This is static attribution, not NCU
counter evidence.

First-process stop-loss rejects this exact global FP32 lookup. P2/P3,
candidate-specific Graph expansion, model oracles, end-to-end timing, Nsys,
NCU, and production integration were not run. The -0.00788879-ms value is an
isolated arithmetic projection, not achieved Decode latency. Production memory
does not include the table, and the hot single-request anchor remains
**106.763000 ms/token and 9.366540843 token/s**. Full evidence and per-round
values are in the [machine-readable rejection record](metadata/qwen36-27b-decode-nvfp4-m1-gate-up-silu-fp32-table-rejection.json).

## Rejected Decode NVFP4 LM-head RP2 schedule-major AoSoA2 P1

This actual-first P1 evaluates a materially different, test-only LM-head row-
pair schedule. The production baseline remains the public activation-staged
NVFP4 `[248320,5120]` route over canonical weights and scales. The RP2
candidate assigns 640 persistent warp owners across an `80x256` launch; each
owner processes 194 logical row pairs. Same-size sidecars place the two rows'
packed U32 words into one U64 and their U8 block scales into one U16 in owner-
major execution order. The candidate retains the production arithmetic,
reduction, scale2, BF16-RNE boundary, launch count, output address, and stream.

The formal Release binary is
`/tmp/q3x-lmhead-rp2-build/q3x_sm87_weight_only_gemv_test`, 7,612,176 bytes
with SHA-256
`d590c818c88acaae225247f11d6c01ebe4389d97bd3383d9b590c218946bf912`.
The 19-line, 5,137-byte P1 log is `/tmp/q3x-lmhead-rp2.P1.log`, with SHA-256
`496dac50cfd28dc8658c79a288b370976479d6b93369686a9eb5f74b680d8464`.
It uses the pinned shard-3 LM-head payload: the 635,699,200-byte weight tensor
hashes to
`746c1d13e9cf69bfca6f5901a7dec5a7f2b252359696644a1ee55953b9680205`,
and the 79,462,400-byte scale tensor hashes to
`e20faadf62bd2b3bf88f2fc9fbf4f42462fdfdd0f4bc9f23d7eaabcc1b697f9b`.

Every non-performance gate passes. The candidate stays within its frozen
resource envelope:

| Route | Grid/block | Registers/thread | Static shared | Local memory | Active CTAs/SM |
| --- | --- | ---: | ---: | ---: | ---: |
| Production activation-staged baseline | `64x256` | 64 | 11,328 B | 0 B | 4 |
| RP2 schedule-major AoSoA2 candidate | `80x256` | 48 | 11,328 B | 0 B | 5 |

The host independently checks every byte of both sidecars. Weight and scale
sidecar mismatch counts are respectively 0/635,699,200 and 0/79,462,400;
their hashes are
`f9c34ee54183745de53049ef24126a7910542b06ab2b9d9e043cb8a6727b5515`
and
`d9e1b57d53a9ddb6684eca803a00728448b734c0b2b96d7461101ed588c0b8f7`.
The valid candidate captures exactly one `80x256` Graph node, while all 18
candidate-invalid and 18 pack-invalid/alias cases capture zero nodes. Direct
candidate and Graph replay outputs both match all 248,320 production BF16
outputs bitwise. Outputs are finite, guards pass, canonical inputs and both
sidecars remain immutable, and injected positive/negative E4M3FN NaN codes
`0x7f` and `0xff` also match all 248,320 baseline outputs bitwise.

The formal process uses ten warmup pairs and five 64-launch paired rounds,
alternating `B-C-C-B` and `C-B-B-C`. Frozen gates require at least 1.075x
actual-checkpoint speedup, at least 0.30 ms saved by the once-per-token LM-head
call, and strict improvement in every round:

| Round | Order | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 | Paired speedup | Paired delta | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | B-C-C-B | 5.20994 ms | 4.77344 ms | 4.76840 ms | 4.63467 ms | 1.03173x | +0.151387 ms | pass |
| 2 | C-B-B-C | 4.83678 ms | 4.77307 ms | 4.78875 ms | 5.34279 ms | 1.06461x | +0.308878 ms | pass |
| 3 | B-C-C-B | 4.87976 ms | 4.77362 ms | 4.74771 ms | 4.38912 ms | 0.973487x | -0.126222 ms | fail |
| 4 | C-B-B-C | 4.39591 ms | 4.74684 ms | 4.74627 ms | 4.74935 ms | 0.963360x | -0.173916 ms | fail |
| 5 | B-C-C-B | 4.38951 ms | 4.74649 ms | 4.74739 ms | 4.38924 ms | 0.924675x | -0.357563 ms | fail |

Only two of five rounds improve. Baseline and candidate pass medians are
4.69201 and 4.75805 ms, while the authoritative median of the paired-round
statistics is **0.973487x** and **-0.126222 ms/token**. These pass medians and
paired medians are distinct statistics. The negative paired result misses both
the 1.075x relative gate and the +0.30-ms absolute gate, and three reversals
fail the every-round requirement.

First-process stop-loss therefore rejects this exact RP2 schedule and sidecar
layout as a performance failure, not a resource or correctness failure.
Same-bank stress, P2, P3, model oracles, full-model timing, profiling, and
production integration were not run. Production allocates neither sidecar;
dispatch, runtime, ABI, Prefill, and the hot single-request Decode anchor remain
unchanged at **106.763000 ms/token and 9.366540843 token/s**. Full evidence is
in the [machine-readable rejection record](metadata/qwen36-27b-decode-nvfp4-lm-head-rp2-schedule-aosoa2-rejection.json).

## Decode short-position CUDA Graph cache production promotion

The selected P19-P43 CUDA Graph cache is now promoted from its P2 test harness
to an engine-lifetime production policy. Ordinary SM87 predicted-token-only,
non-trace CLI and benchmark requests with more than one generated token request
the bounded 25-slot cache; the generic engine API remains default-disabled.
Each hit executes the existing dependency-ordered 389-kernel DAG plus its
greedy-result D2H as one Graph launch, updating only the embedding root token
offset. Preparation occurs outside generation, stages the full bank before an
all-or-nothing publication, and checks the 1-second/256-MiB admission envelope.
Cache misses, incompatible modes, and P44 remain serial. A runtime Graph
failure is not retried serially for the same token and demotes subsequent work.

The production full-model gate compares enabled and disabled engines, including
a reset between two canonical generations. It checks generation semantics and
the exact enabled/disabled 87,846,400-byte arena layout; unlike the P2 harness,
it does not claim complete-arena byte equality:

| Production route | Graph replays | Serial fallbacks | Result |
| --- | ---: | ---: | --- |
| Canonical P19-P43 | 25 | 0 | exact |
| Post-reset P19-P43 | 25 | 0 | exact |
| Full statistics | 0 | 0 | exact serial path |
| Trace | 0 | 0 | exact serial path |
| Max27 through P44 | 25 | 1 | exact P44 miss fallback |

All 19 prompt IDs, 26 generated IDs, exact text, `im_end`, and 44 steps match
the pinned golden. The latest production preparation takes **75.758861 ms**
and observes a **76,607,488-byte** CUDA free-memory drop, passing the 1-second
and 268,435,456-byte gates. This device-wide free-memory delta is not added to
host RSS on Jetson's shared DRAM.

The formal performance comparison uses the ordinary benchmark production
entry, not the test-only prepared-cache hook. Four independent fixed-core
processes run `B1-C1-C2-B2`; each performs one warmup and five measured
P19/C32/max26 generations:

| Process | Decode median | Total-generation median | Candidate dispatch |
| --- | ---: | ---: | ---: |
| B1, `ba45011` | 106.763 ms/token | 3093.621 ms | — |
| C1, `fc9547c` | 105.865 ms/token | 3071.291 ms | 125/0 |
| C2, `fc9547c` | 105.876 ms/token | 3071.281 ms | 125/0 |
| B2, `ba45011` | 106.747 ms/token | 3092.894 ms | — |

Both mirrored pairs improve by **0.898/0.871 ms/token**. Their aggregate is
**106.755000 ms/token** baseline versus **105.870500 ms/token** candidate,
saving **0.884500 ms/token** for **1.008354546x** and reaching
**9.445501816 token/s**. Every candidate sample dispatches 25 Graph replays
and zero fallbacks, all four processes reproduce the golden, and none detects
a persistent device-memory drop. The new directly achieved hot single-request
Decode anchor is therefore **105.870500 ms/token / 9.445501816 token/s**,
replacing **106.763000 ms/token / 9.366540843 token/s**. It remains
**5.870500 ms/token and 0.554498184 token/s** short of the 100-ms/token and
10-token/s target.

Nsight Systems independently finds 25 Decode ranges and exactly one Graph
trace in each, with 25 distinct graph/GraphExec IDs on one stream and a
105.867008-ms median Graph trace. All 25 capture, end-capture, instantiate, and
upload calls occur before generation, with none inside Decode ranges. The full
Release suite passes 55 of 59 tests with four expected skips and no failures.
Execution still has no double/triple buffer, cross-kernel overlap, separate
Prefill/Decode executor, or Prefill/Decode overlap; bulk Prefill is unchanged.
Failure-injection coverage for transactional rollback and runtime demotion
remains test debt. Complete hashes, commands, gates, and claim limits are in
the [production benchmark record](metadata/qwen36-27b-decode-short-position-cuda-graph-cache-production-benchmark.json).

## Decode M1 gate/up table-free E2M1 rejection

The next actual-first P1 screen cloned the exact production-CS `32x512` M1
gate/up kernel and changed only E2M1 decoding: the baseline's shared 16-entry
FP32 LUT became the already exhaustive BF16 PRMT constructor. Packed weights
and block scales retain the same evict-first `__ldcs` loads, the E4M3FN scale
table remains shared, and residual/RMSNorm, projection arithmetic, independent
gate/up rounding, CTA-local staging, SiLU, stream, and launch topology are
unchanged. The 262,144-combination PRMT gate has zero word/half mismatches and
preserves signed zero. Actual residual and gate outputs, including a second
direct replay, are bit exact; finite, guard, and dead-workspace gates pass.

Resources do not explain a residency loss: baseline and candidate both use 64
registers, zero local memory, and two active CTAs/SM, while removal of the
E2M1 table lowers static shared memory from 13,632 to 13,568 bytes. Static SASS
does expose the tradeoff: total instructions rise from 1,352 to 1,568, `LDS`
falls from 89 to 25, `PRMT` rises from 3 to 131, and both retain twelve
`LDG.E.EF` plus zero `LDL`/`STL` rows.

Ten warmups precede five alternating 64-launch rounds. Frozen gates require
at least 1.0135x, at least 0.0078125 ms/layer (0.5 ms over 64 layers), and
strict improvement in every round:

| Round | Order | Baseline 1 | Candidate 1 | Candidate 2 | Baseline 2 | Paired speedup | Paired delta/layer | Result |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | B-C-C-B | 0.589707 ms | 0.949269 ms | 0.949386 ms | 0.589633 ms | 0.621145x | -0.359657 ms | fail |
| 2 | C-B-B-C | 0.589563 ms | 0.949135 ms | 0.949183 ms | 0.589683 ms | 0.621206x | -0.359536 ms | fail |
| 3 | B-C-C-B | 0.589787 ms | 0.948672 ms | 0.948848 ms | 0.590051 ms | 0.621779x | -0.358841 ms | fail |
| 4 | C-B-B-C | 0.589942 ms | 0.949548 ms | 0.949651 ms | 0.589505 ms | 0.621024x | -0.359876 ms | fail |
| 5 | B-C-C-B | 0.589285 ms | 0.949283 ms | 0.949280 ms | 0.589586 ms | 0.620928x | -0.359846 ms | fail |

Baseline/candidate medians are **0.589658/0.949274 ms/layer**. The paired
median is **0.621145x** and **-0.359657 ms/layer**, projecting to a
**23.0181-ms/token loss** over 64 layers. First-process stop-loss therefore
skips stress timing, candidate Graph/invalid matrices, full-model validation,
Nsys, and production integration. Candidate source and test hooks are removed;
production and the formal **105.870500-ms/token / 9.445501816-token/s** anchor
remain unchanged. Full identities and claim limits are in the
[machine-readable rejection record](metadata/qwen36-27b-decode-nvfp4-m1-gate-up-table-free-e2m1-rejection.json).

## Decode down adaptive scale-width ceiling rejection

An exact raw-code scan of all 64 pinned
`model.language_model.layers.i.mlp.down_proj.weight_scale` tensors finds
5,570,560 `F8_E4M3` bytes per layer. Direct base-plus-delta spans range from 46
to 69: 53 layers fit six bits, while layers 2, 24, 30, 31, 54-60 require seven
bits; none fits five bits. The current selected hot scale payload is
`53*4,177,920 + 11*5,570,560 = 282,705,920` bytes. Ideal padding-free adaptive
packing lowers that to 275,046,400 bytes, only **7,659,520 bytes or
2.709359606%**. Its incremental saving is 10.377358491% of the existing
73,809,920-byte scale6 saving.

Linearly scaling the three existing 53-layer projections
`0.434494/0.268021/0.199822` by that incremental ratio gives optimistic
`0.045089000/0.027813500/0.020736245 ms/token`, with a **0.027813500-ms/token
median** before seven-bit unpack overhead. A more aggressive dictionary ideal
uses six-bit indices for 63 layers and seven bits for one. Before even its
codebooks and lookup cost, it reaches 268,083,200 bytes: only **5.172413793%**
below current, with an optimistic **0.053098500-ms/token** median.

Both lower bounds miss the frozen 15% payload-reduction and 0.30-ms/token
admission gates. This is therefore a zero-code ceiling rejection: no branch,
candidate, build, GPU run, or production change exists, and the formal anchor
remains **105.870500 ms/token / 9.445501816 token/s**. The exact shell scanner,
unique-code distribution, equations, and claim limits are in the
[machine-readable rejection record](metadata/qwen36-27b-decode-down-adaptive-scale-bitwidth-ceiling-rejection.json).

## Non-MTP Decode global-traffic and state audit

MTP is explicitly excluded from the current 100-ms/token path. A source-level
traffic inventory and fresh root-enabled, current-production NCU pass instead
evaluate global traffic, cross-kernel movement, L2 persistence, and SSM state
update. The exact projection payload lower bound is **17,528,668,160 bytes per
token**. Adding one BF16 read and write of the 48-layer GDN state, logical
causal-conv traffic, and unique short-position KV bytes raises the tracked
minimum to **17,693,556,736 bytes per token**. At the formal 105.870500-ms
anchor, projection payload alone corresponds to **165.567 GB/s**. Reaching 100
ms requires at least **176.936 GB/s** for the tracked bytes, or 86.394% of the
204.8-GB/s theoretical LPDDR peak, before transaction, compute, synchronization,
and small-tensor costs.

The current Graph trace assigns 104.563872 ms/token to the six projection
groups. Gate/up, down, and LM head account for 38.361600, 20.192096, and
4.386208 ms/token; their payload-effective rates are 167.284, 155.250, and
163.048 GB/s. Even the optimistic sensitivity of moving all three to 175 GB/s
saves only **4.269736 ms/token**, so a non-projection contribution is also
needed. Fresh matched NCU narrows the diagnosis:

| Kernel | Duration | L1 global request | LTS total | DRAM read peak | SM peak | Resources |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| gate/up | 606.784 us | 101.253 MB | 100.866 MB | 81.12% | 71.85% | 64r, 13,632 B, 2 CTA/SM |
| down scale6 | 319.840 us | 50.303 MB | 49.807 MB | 74.90% | 71.76% | 64r, 35,904 B, 2 CTA/SM |
| LM head | 4,417.888 us | 715.817 MB | 723.252 MB | 79.42% | 69.27% | 64r, 11,328 B, 4 CTA/SM |
| GDN row8 | 37.376 us | 1.677 MB | 3.322 MB | 21.57% | 57.65% | 40r, 34,568 B, 4 CTA/SM |

The projection reports are already close to mandatory weight-plus-scale bytes;
small instruction substitutions therefore lack a credible 0.30-ms/token
ceiling. The QKV reduction scratch is CTA-local shared memory, not a global
cross-core buffer. Prior cross-stage fusion screens also regress, and the
single-stream Graph trace has only about 0.57% idle time, so same-request
double/triple buffering cannot recover the 5.870500-ms gap.

Payload entropy does not rescue LM head: even a free, metadata-free
Shannon-limit codec saves only 49.018 MB, an optimistic **0.302736 ms/token**,
so any real lossless codec misses the gate. Two projection ideas remain
zero-code admissions rather than implementations: a gate/up four-bit common
scale palette with sparse raw escapes must first fit all 128 scale tensors into
at most 60% of canonical bytes, and prediction-only exact progressive LM
pruning must certify the same argmax while saving at least 55 MB/call after all
bounds and survivor traffic. A down wavefront-major layout is lower confidence
still: it removes no bytes and would require a 2.20-GiB sidecar if selected.

L2 persistence is not the first implementation path. The device exposes 4 MiB
of L2 and at most 2.75 MiB of persisting cache. That cannot retain the
17.53-GB/token weight stream and covers only 3.819% of the 72-MiB GDN state,
whose optimistic linear ceiling is **0.058269 ms/token**. A later default-off
conv-state window may be screened, but only after stronger candidates.

GDN is the selected actual-first P1. Nsys measures 31.783 us/layer, or
1.525591 ms/token, while NCU shows only 21.57% DRAM-read throughput and a
shared/dependency bottleneck. The test-only candidate will retain each
thread's eight-row transient BF16 state in packed registers across prediction
and delta update, preserve the existing arithmetic/rounding order, and write
global state once. It advances only if all 786,432 state bits and 6,144 output
bits match, multi-step/in-place/disjoint/NaN/Graph gates pass, local memory
stays zero, residency remains at least three CTA/SM, and actual timing reaches
at least **1.2448x / 0.30 ms/token**. Otherwise it is removed immediately.

The formal production anchor remains **105.870500 ms/token / 9.445501816
token/s**. Full report identities, equations, priorities, and claim limits are
in the [non-MTP audit record](metadata/qwen36-27b-decode-non-mtp-global-traffic-state-audit.json).

## Decode GDN M1 transient register-state rejection

The first non-MTP state-update P1 targeted the production fused GDN/plain
RMSNorm/SiLU-gate kernel's second read of decayed FP32 row scratch. That read is
64 KiB per value head, 3 MiB per layer, and **144 MiB/token** over 48 GDN
layers. Two same-binary, test-only implementations preserve the production
48x256 launch, K=0..127 FMA order, BF16 state boundary, output arithmetic,
global state traffic, and production dispatcher:

| Candidate | Resources | Baseline | Candidate | Speedup | Projected saving | Gate |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| packed BF16 + recompute | 64r, 34,568 B, 0 local, 4 CTA/SM | 31.6932 us | 30.1427 us | 1.05144x | 0.074424 ms/token | fail |
| retained FP32 decay | 80r, 34,568 B, 0 local, 3 CTA/SM | 31.6569 us | 30.0652 us | 1.05294x | 0.076402 ms/token | fail |

Both variants pass four-step in-place, disjoint state, replay, directed signed
NaN/infinity, invalid capture, one-node 48x256 Graph, and complete 786,432-state
bit/6,144-output-bit comparisons. Static SASS confirms the intended mechanism:
both remove 32 `LDS` rows without changing `STS`, `LDG`, `STG`, `FFMA`, shuffle,
or barrier counts; pair rounding adds 16 `F2FP` and 23-25 `PRMT` rows.

The 24-state-bank working set is 36 MiB per variant, nine times L2. Each
process uses 48 warmups and 480 measured launches per pass over five `B-C-C-B`
rounds. Every round improves, but both candidates deliver only about one
quarter of the frozen **1.2448x / 0.30-ms/token** admission requirement. FP32
reuse removes the packed variant's decay recomputation yet barely changes the
result, so the single shared reload is not the dominant whole-kernel limiter.

First-process stop-loss skips full-model, candidate Nsys/NCU, and production
integration. Both candidate kernels, launchers, and tests are removed with
`apply_patch`; the production source/test blobs are restored exactly, the
clean target build and default GDN CUDA suite pass, and the formal anchor stays
**105.870500 ms/token / 9.445501816 token/s**. Complete rounds and claim limits
are in the [rejection record](metadata/qwen36-27b-decode-gdn-m1-transient-register-state-rejection.json).

## Decode down global-flow layout ceiling

A fresh 24-pass NCU transaction-table capture closes down-only preswizzle and
cross-core repacking whose sole mechanism is improved global coalescing. The
production scale6 kernel issues 1,573,753 load sectors at **31.846437 useful
bytes per 32-byte sector**, or **99.520116% utilization**. It records 48,797,696
L2 read-miss bytes against 48,742,400 mandatory packed-weight plus scale6
bytes. Stores occupy only 184,320 sector bytes and have zero L2 write lookup
misses.

NCU estimates only **0.4654% local speedup** from an ideal remaining load
pattern. Applied to the 20.192096-ms/token down stage, that is **0.093974
ms/token**. Even an impossible zero-cost removal of every unused load and store
sector byte projects to **0.157985 ms/token**, below the frozen 0.30-ms gate.
No candidate kernel, sidecar, or dispatch change is created, and the formal
anchor remains **105.870500 ms/token / 9.445501816 token/s**. Report identity,
raw counters, command, and claim limits are in the
[measured ceiling record](metadata/qwen36-27b-decode-down-global-flow-layout-ceiling-rejection.json).

## Decode gate/up P15E-T128 payload admission

The gate/up lossless-scale scan does clear its byte-only gate. Across all 128
`F8_E4M3 [17408,320]` tensors, a per-tensor top-15 palette with four-bit codes,
128-value escape-prefix tiles, raw escapes, headers, and padding occupies
**398,909,184 bytes**, or **55.945506%** of the 713,031,680 canonical bytes.
All tensors round-trip bit exactly and independently stay under 60%; the worst
is layer-50 gate at 58.363971%. A shared palette for each layer's gate/up pair
costs only 255,488 additional bytes over the full model and is the selected
test-only decoder fixture.

The **1.897253-ms/token** free-decode projection is only a byte ceiling. The
older scale6 gate/up sidecar, despite removing 25% of scale bytes, regressed in
all 30 formal rounds by about 0.0210-0.0224 ms/layer. P15E-T128 therefore enters
only a same-binary decoder/sector stop-loss; it is not allocated or reachable
from production. The formal anchor is unchanged. Exact byte decomposition,
hashes, risk comparison, and gates are in the
[capacity record](metadata/qwen36-27b-decode-gate-up-p15e-t128-scale-payload-admission.json).

## Decode gate/up P15E-T128 decoder rejection

The byte admission was followed by an actual layer-0, same-binary standalone
decoder screen. Host and device unpacking match every canonical E4M3 scale
byte; checksums, guards, and input immutability pass. The candidate uses 27
registers, 16 bytes of shared memory, zero local memory, and three active
CTA/SM. Nevertheless, every formal `B-C-C-B` round regresses:

| Metric | Canonical | P15E-T128 | Result |
| --- | ---: | ---: | ---: |
| Selection median | 122.343 us/layer | 197.127 us/layer | 0.620662x |
| L1 request sectors | 348,192 | 459,649 | +32.010% |
| LTS sector equivalents | 349,244 | 208,588 | -40.274% |
| Projected 64-layer delta | -- | -- | -4.785536 ms/token |

Palette lookup, tile-directory reads, sparse raw escape ranking, ballot, and
popcount more than erase the lower L2-facing payload. Layer 50 and full GEMV
integration are skipped by the predeclared first-fixture stop-loss. The
standalone source and CMake target are deleted; production dispatch and the
formal anchor remain unchanged. Full round and NCU identities are in the
[decoder rejection record](metadata/qwen36-27b-decode-gate-up-p15e-t128-decoder-rejection.json).

## Decode GDN transposed-state and native-encoder rejection

The follow-up tested whether a one-time Prefill-to-Decode transpose from
`[head][value][key]` to `[head][key][value]` could give each row-owner warp
coalesced state access and eliminate further shared-state flows. Correctness
passes bitwise over the complete state, three recurrence steps, directed
nonfinite patterns, and Graph replay. The phase-boundary transpose costs about
1.01--1.03 ms/request, but the decisive 24-bank cold-state results all regress:

| Candidate | Cold speedup | Projected 48-layer delta |
| --- | ---: | ---: |
| four-warp double-global state | 0.843005x | -0.283054 ms/token |
| four-warp BF16 shared retain | 0.902691x | -0.163734 ms/token |
| eight-warp cooperative load | 0.579271x | -1.103440 ms/token |
| canonical layout, native BF16 encoder only | 0.996115x | -0.005874 ms/token |

NCU confirms the second global state pass is almost entirely an L1 hit. The
failure instead comes from exposed cold first-read latency, too few useful
state-work warps, and, for cooperative loading, a phase barrier. Resident-only
speedups up to 1.50276x are therefore cache artifacts rather than promotion
evidence. Candidate code is removed, the production source/test hashes are
restored, the default CUDA GDN suite passes, and the formal anchor remains
**105.870500 ms/token / 9.445501816 token/s**. See the
[structural rejection record](metadata/qwen36-27b-decode-gdn-transposed-state-native-encoder-rejection.json).

## Decode LM-head exact progressive traffic audit

An offline audit captured all 25 P19--P43 final-norm activation vectors and
tested exact prediction-only branch-and-bound traffic. The best fixed-cut,
two-kernel scheme reads 4,864 of 5,120 columns before fully recomputing
survivors. Although survivor ratios are tiny, duplicated prefix work limits
the mean/worst saving to **34.584/33.912 MB per call**; no step reaches the
55-MB admission gate, so that route is closed without code.

A different q20 single-kernel simulation, with 20 256-column checkpoints and
register-resident partial accumulators, saves **89.591 MB/call** on average and
**59.539 MB/call** at worst. It remains conditional P2 only: no implementation,
timing, or production claim exists until per-row predicates demonstrably stop
future weight transactions, completed-wave incumbents are published
deterministically, exact native argmax/tie behavior passes, and measured
traffic stays above the gate after the 18.872-MB suffix-norm sidecar. See the
[progressive audit record](metadata/qwen36-27b-lm-head-exact-progressive-mips-admission.json).

## Decode QKV/Z P127E-W128 payload admission

An exact scan of all 48 linear-attention QKV and Z pairs selects a lossless
per-tensor 127-entry FP8 palette, seven-bit code stream, 128-value tile prefix,
row escape base, and sparse raw-byte escape pool for an actual-first test. All
96 tensors pass the frozen 92% capacity ceiling:

| Payload | Raw | Encoded | Saving |
| --- | ---: | ---: | ---: |
| all QKV/Z layers | 4,026,531,840 B | 3,641,497,600 B | 385,034,240 B (9.562429%) |
| actual layer 0 | 83,886,080 B | 75,948,032 B | 7,938,048 B (9.463%) |

At the current 175.182-GB/s effective stage rate, free decoding would save
**2.197907 ms/token**. Reaching the 0.30-ms admission requires only 1.013070x
stage speedup, or 6.25 us/layer, and leaves an ideal 39.54-us/layer budget for
new decode work. Those values are ceilings, not achieved performance. Each
112-byte row tile may still span four L1 sectors, while shuffle extraction,
escape rank, directory reads, and palette lookup can erase its L2 byte saving.

The first implementation is therefore test-only and layer-0 bounded. It must
reconstruct every FP8 byte, preserve QKV/Z/A/B BF16 outputs and the production
arithmetic order, use no local memory, retain four CTA/SM, pass actual and
stress non-regression in every round, and net at least 0.30 ms/token before any
3.391-GiB full-model sidecar or production integration. O projection remains a
rank-2 backup with a narrower 12.46-us/layer overhead budget. Full tensor
hashes, byte decomposition, closed-route deduplication, and stop-loss gates are
in the [capacity admission](metadata/qwen36-27b-decode-fp8-qkv-z-p127e-w128-payload-admission.json).

## Decode QKV/Z P127 decoder rejection

The actual-first implementation replaced the risky row-prefix/raw-pool design
with a stronger row-quad P127X hybrid: fixed aligned 112-byte code tiles,
sorted `(position, raw-byte)` exception pairs, and canonical 128-byte fallback
for tiles with at least eight escapes. Layer 0 packs 83,886,080 raw bytes into
76,997,608 sidecar bytes, reconstructs every host byte, and matches all 512
device warp checksums. Resources also pass at 26 registers, 256 bytes shared,
zero local memory, and six active CTA/SM. Performance does not:

| Decoder | Median paired speedup | Median delta/layer | 48-layer projection |
| --- | ---: | ---: | ---: |
| exact P127X hybrid | 0.369522x | -1,893.332 us | -90.879944 ms/token |
| fixed-direct impossible lower bound | 0.568148x | -843.482 us | -40.487129 ms/token |

The fixed-direct row deliberately omits every directory read, escape load,
exact escape reconstruction, and shared-palette lookup. Its failure is thus a
hard lower-bound rejection, not merely a poor exception encoding. Matched NCU
explains the exact P127X result:

| Metric | Canonical | P127X | Change |
| --- | ---: | ---: | ---: |
| L1 global request | 83,886,080 B | 182,734,432 B | +117.836% |
| L1 global miss | 83,886,080 B | 87,856,000 B | +4.733% |
| LTS total | 83,919,136 B | 87,915,648 B | +4.762% |
| duration | 1,099.2 us | 3,004.0 us | regression |

The candidate source, CMake target, binary, and object are removed; the default
SM87 GEMV suite passes and production is unchanged. This closes seven-bit W128
P127E/P127X/fixed-direct compression and its lower-margin O-projection backup
on SM87. Complete rounds, report hashes, SASS resources, and cleanup evidence
are in the [decoder rejection](metadata/qwen36-27b-decode-fp8-qkv-z-p127x-w128-decoder-rejection.json).

## Decode Gate/Up Delta4 row-fallback admission

The next lossless scale screen avoids both scale6 cross-word unpack and P15E
escape ranking. For each row's 32 scales in a K512 tile, it stores a raw-byte
base plus 32 four-bit deltas when the raw E4M3 span is at most 15; otherwise
that row reads its existing aligned canonical sector. All 713,031,680 source
bytes round-trip exactly in the offline scan:

| Inventory | Result |
| --- | ---: |
| compressed row tiles | 18,133,306 / 22,282,240 (81.380086%) |
| fallback row tiles | 4,148,934 (18.619914%) |
| fixed sidecar | 378,798,080 B (53.125% of canonical scales) |
| conservative L1 requests | 524,933,312 B, down 188,098,368 B |
| conservative traffic ceiling | 1.136086 ms/token |

The required physical layout is fixed: lanes 0--15 load one aligned U32 each
from a 64-byte `[lane-pair][row4]` tile, one indexed shuffle gives every lane
both phase nibbles, and lanes 0--9 preload the row quad's ten contiguous U32
metadata words in two sectors. Direct rows remove the canonical partner XOR;
fallback rows retain one known-address canonical sector and share one
conditional partner shuffle. Loading metadata separately for every tile would
cap the saving at only 0.271945 ms/token and is therefore not an admitted
implementation.

This remains a zero-code capacity/sector result. A standalone decoder must pass
both layer 0 and the lower-margin layer 50, reproduce all frozen hashes, use no
local memory, retain at least two CTA/SM, and save at least 4.6875 us/layer in
every five-round `B-C-C-B` series. No 361.25-MiB full sidecar or production
dispatch may be created before that gate. See the
[Delta4 admission record](metadata/qwen36-27b-decode-gate-up-delta4-row-fallback-admission.json).

## Decode Gate/Up Delta4 decoder rejection

The production-unreachable decoder reconstructs every host and device scale
byte exactly, preserves all guards and inputs, and passes the corrected
resource gate at 39 registers, zero local memory, and three active CTA/SM.
An initial apparent win was invalidated before selection because its canonical
baseline spilled 328 local bytes per thread. With both backends changed to the
same runtime ten-tile loop, the canonical baseline also uses zero local memory
and the valid layer-0 result reverses decisively:

| Metric | Canonical | Delta4 | Change |
| --- | ---: | ---: | ---: |
| Paired median | about 154.2 us | about 257.6 us | 0.598319x |
| L1 global-load sectors | 348,160 | 252,907 | -27.359% |
| LTS total sectors | 349,480 | 247,899 | -29.066% |
| Dynamic instructions | 6,072,064 | 11,704,600 | +92.761% |

All five `B-C-C-B` rounds regress, with a median delta of **-103.492233
us/layer** and a 64-layer projection of **-6.623503 ms/token**. The physical
traffic reduction is real, but base/nibble reconstruction, metadata broadcast,
fallback predicates, and conditional canonical reconstruction nearly double
the instruction count. The frozen first-fixture stop-loss therefore skips a
valid layer-50 run and full GEMV integration. The standalone source, target,
binary, and object directory are removed; production and the formal
**105.870500 ms/token / 9.445501816 token/s** anchor remain unchanged. See the
[decoder rejection record](metadata/qwen36-27b-decode-gate-up-delta4-row-fallback-decoder-rejection.json).

## Decode causal-convolution L2-persistence ceiling rejection

The bounded persisting-L2 follow-up is rejected before implementation. The 48
linear-attention histories occupy 2.8125 MiB and generate 5.625 MiB of unique
read-plus-write traffic per token. Orin can reserve at most 2.75 MiB, covering
44/45 of that state. Even granting the unrealistic assumption that every
covered byte removes both a read and a write, the ceiling is only **0.034833
ms/token** at the measured projection bandwidth.

Production Nsys provides a stronger bound: all 48 causal-convolution kernels
together cost **0.260622 ms/token** in a current-source-identical trace, and
**0.283672 ms/token** in an independent trace. Deleting the entire kernel for
free would still miss the frozen 0.30-ms admission gate. Reserving 2.75 MiB
would also leave only 1.25 MiB of ordinary L2 and could regress much larger
projection streams. No APW probe or production change is therefore created.
See the [ceiling rejection](metadata/qwen36-27b-decode-causal-conv-l2-persistence-ceiling-rejection.json).

## Decode cross-kernel/core-flow ceiling rejection

The remaining serial intermediate boundaries were re-audited after the GDN,
QKV compression, and Gate/Up decoder stop-losses. No individual boundary has a
credible 0.30-ms/token ceiling. QKV-to-convolution and convolution-to-GDN each
combine only about 0.065 ms of observed GPU gap with 1.966 MB/token of BF16
write/read traffic, or roughly **0.077 ms/token** under perfect removal.
O-to-Gate and Down-to-next are each below about **0.095 ms/token**.

The superficially large Gate-to-Down and Down-to-next request counts are not
removable DRAM traffic. Gate-to-Down needs a global all-to-all broadcast for 32
consumer CTAs, while production QKV/Z evidence puts the normalized activation
at **99.568176% inferred L1 hits**. Existing fusion and shared-staging screens
also regress. Even the impossible sum of both adjacent linear boundaries is
only about **0.154 ms/token**, and broader sums would double-count gaps while
requiring one cooperative mega-kernel across incompatible topologies. Generic
core-flow fusion is therefore closed without new code. See the
[core-flow ceiling record](metadata/qwen36-27b-decode-cross-kernel-core-flow-ceiling-rejection.json).

## Decode GDN row16 register-baton rejection

The production-unreachable row16 screen assigns two lanes to each state row
and passes the ordered accumulator between them, removing all five shared-state
transfers without changing the canonical state layout or global read/write
sectors. It passes the static/resource gate at 71 registers, 2,568 bytes of
shared memory, zero local/stack bytes, and three active CTA/SM. Four-step full
state/output comparisons, in-place and disjoint state, directed special values,
guards, input preservation, and one-node Graph replay are all bitwise exact.

The first 24-bank cold-state process improves from 31.5728 to 26.3388 us/layer,
or **1.19872x / 0.251234 ms/token** projected over 48 GDN layers. An independent
process reaches **1.19586x / 0.247622 ms/token**. Both are stable improvements,
but both miss the frozen **1.2448x / 0.30-ms/token** hard gate, so NCU and
production integration are skipped and the candidate is removed. The clean
default GDN test passes, production and the formal anchor remain unchanged,
and MTP was not used. See the
[row16 rejection record](metadata/qwen36-27b-decode-gdn-row16-register-baton-rejection.json).

## Decode LM-head q20 exact-sector rejection

The canonical P19--P43 activation/checkpoint hashes and directed BF16-RNE
suffix-bound proof pass, but the only schedule independent of global-progress
assumptions does not meet the frozen traffic gate. With a per-CTA incumbent and
the conservative `-1` delta, all 25 fixtures fail: exact request-sector savings
range from **19.244 to 34.044 MB/call**, average **26.311 MB/call**, versus the
required 55 MB on every fixture.

The ideal completed-wave diagnostic reaches 25/25, but its worst fixture clears
the gate by only **54,752 bytes** and requires **121 global completion
boundaries**. It is therefore not an independent admission path; a one-wave-lag
diagnostic already falls to 24/25. A contiguous global-seed sweep likewise has
no seed size that passes all 25 fixtures. The q20 schedule family is closed
before CUDA or NCU work. Production and the formal **105.870500 ms/token /
9.445501816 token/s** anchor remain unchanged, and MTP was not used. See the
[exact-sector rejection record](metadata/qwen36-27b-lm-head-q20-exact-sector-rejection.json).

## Decode baseline lock and Prefill phase handoff

The achieved non-MTP P19/C32/max26 fixed-clock result is frozen at
**105.870500 ms/token / 9.445501816 token/s** as the Decode regression anchor.
The original 100-ms/token / 10-token/s objective remains unmet and documented,
but it no longer blocks dedicated Prefill work. MTP remains outside this
optimization path. Earlier sections that placed Prefill behind that Decode
gate describe the policy active at those historical milestones and are
superseded by this handoff.

The current-HEAD direct Prefill lock uses source `edef543`, the Release SM87
binary SHA-256
`c63d6dbc187abace4539f7b0915e75b7abd31a89a93967390d947ac176a5e6bb`,
MAXN, fixed 1.3005-GHz GPU and 3.2-GHz EMC clocks, C32/max1, one warmup, and
five measured generations per prompt:

| Prompt | Prefix tokens | Prefix median | Finish-prefill median | TTFT median | Prefix throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| P33 | 32 | 271.159 ms | 106.890 ms | 378.074 ms | 118.011941 token/s |
| P65 | 64 | 543.646 ms | 107.161 ms | 650.889 ms | 117.723666 token/s |
| P129 | 128 | 1,102.542 ms | 107.413 ms | 1,209.974 ms | 116.095351 token/s |
| P513 | 512 | 4,605.071 ms | 108.853 ms | 4,713.890 ms | 111.181782 token/s |

All 20 measured generations produce ID `9419`, text `Hello`, the expected
prompt/step counts, and no persistent device-memory drop. The nearly linear
C32-tile accumulation and the latest overlap-aware phase profile make larger
Prefill tiles, cross-tile weight reuse, and a dense-prefill backend comparison
higher priorities than a general single-request buffering rewrite. The user's
reported 2k--8k token/s vLLM/FlashInfer result is retained as an external
opportunity signal, not a directly comparable project baseline until model,
hardware, prompt, concurrency, and accounting contracts are matched.

The separately executed P19/max26 control reproduces the exact 26 generated
IDs/text/stop contract and uses 125 prepared Graph replays with zero serial
fallback. Its first two measured Decode sequences remain around the frozen
anchor, but later samples show thermal drift and the process reports a
164,122,624-byte persistent free-memory decrease above the 64-MiB tolerance.
It is therefore a route/correctness diagnostic only and does not replace the
formal mirrored Decode anchor. Full identities, ranges, claim limits, and the
next-work decision are in the
[machine-readable baseline record](metadata/qwen36-27b-current-head-prefill-baseline.json).

## Prefill NVFP4 M64 down schedule selection

The first large-M Prefill screen validates cross-tile weight reuse on the
exact NVFP4 down shape `[M=64, N=5120, K=17408]`. One test-only CTA keeps four
M16 WMMA accumulators, stages only two activation panels at a time, and reuses
each decoded K64 weight tile across all 64 tokens. The comparison baseline is
two ordered calls to the current production exact-M32 table-free kernel;
production dispatch is unchanged.

Under MAXN with fixed 1.3005-GHz GPU and 3.2-GHz EMC clocks, 10 warmup pairs,
24 logical operations per pass, and six `B-C-C-B` rounds per distribution:

| Scale distribution | Two production M32 | Candidate M64 | Speedup | Worst round |
| --- | ---: | ---: | ---: | ---: |
| checkpoint-like synthetic | 1.78080 ms | 1.41531 ms | 1.25824x | 1.25704x |
| same-bank stress | 1.78056 ms | 1.41563 ms | 1.25778x | 1.25674x |
| aggregate | 3.56136 ms | 2.83094 ms | **1.25801x** | 1.25674x |

Both distributions match all 327,680 BF16 outputs bit-for-bit and capture as
one CUDA Graph kernel node. All 12 rounds improve. Release SASS reports the
production M32 at 48 registers, 23,552 bytes shared, and zero local memory;
M64 uses 76 registers, the same shared footprint, zero local memory, and three
active CTA/SM. The directed screen and default `sm87_weight_only_gemv` test
pass.

This selects large-M weight decode/reuse as the active Prefill mechanism, but
does not claim an end-to-end gain yet. Runtime promotion first needs a
phase-local path that presents 64 contiguous MLP intermediate tokens to down
projection without changing attention/GDN ordering, residual boundaries, or
tail behavior. Full protocol and claim limits are in the
[machine-readable M64 screen](metadata/qwen36-27b-prefill-nvfp4-m64-down-screen.json).

## C64 Prefill and NVFP4 M64 down production promotion

Commit `3f8dd10` promotes the selected schedule behind a C64 request/runner
boundary and advances the exact package ABI to 0.3.0. Only exact aligned NVFP4
`[M64,N5120,K17408]` down uses one M64 kernel. All other C64 projections retain
two ordered C32 schedules; gate/up keeps its narrow layer-local branch overlap,
residual/RMS uses two exact M32 operations, and causal Conv/GDN/QK+RoPE remains
on ordered subtiles of at most M16. A 33..63-token controller candidate is
issued as C32 plus a recomputed ordered tail, so partial-wide requests do not
lose the established C32 schedules. Decode Graph admission remains explicitly
pinned to the frozen P19/C32 baseline, and MTP is not used.

The formal screen uses MAXN, 1.3005-GHz GPU, locked 3.2-GHz EMC, one warmup,
five measured generations per prompt, and mirrored `B1-C1-C2-B2` processes.
Each process shares one loaded model across P33/P65/P97/P129/P513; the table is
the arithmetic mean of mirrored process medians:

| Prompt | C32 Prefix | C64-policy Prefix | Prefix speedup | C64-policy TTFT | Prefix throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| P33 | 271.646 ms | 271.367 ms | 1.001030x | 378.197 ms | 117.921486 token/s |
| P65 | 545.055 ms | 511.476 ms | **1.065650x** | 618.703 ms | 125.127938 token/s |
| P97 | 824.211 ms | 790.412 ms | **1.042762x** | 897.707 ms | 121.455723 token/s |
| P129 | 1,104.837 ms | 1,037.894 ms | **1.064499x** | 1,145.282 ms | 123.326660 token/s |
| P513 | 4,612.398 ms | 4,339.632 ms | **1.062855x** | 4,448.349 ms | 117.982368 token/s |

All 20 formal generations emit ID `9419`, text `Hello`, and exact prompt/step
counts. P65 is one C64 tile, P97 is C64+C32, P129 is two C64 tiles, and P513 is
eight C64 tiles; P33 remains the unchanged C32 schedule. The C64 request arena
adds 5,445,632 bytes. One C64 process reports a 239,554,560-byte free-memory
drop, but C1 and an independent C3 rerun report zero persistent drop; the
anomalous process is retained as a limitation rather than discarded.

Matched no-warmup P513 Nsight diagnostics confirm the route. C32 executes
1,024 down-M32 kernels totaling 934.292 ms; C64 executes 512 down-M64 kernels
totaling 737.822 ms, a 1.266283x cumulative-kernel improvement. The controlling
Prefix NVTX span falls from 4,644.437 to 4,376.232 ms (1.061287x), while
range-minus-projected-GPU time stays below 0.04% for both policies. The complete
Release suite reports 51 passes, 9 model/external-data skips, and zero failures;
the direct C64 fixed fixture and the P65/P97/P129/P513 formal outputs pass on
the target model. Full identities, hashes, per-process medians, memory evidence,
and limitations are in the
[C64 production record](metadata/qwen36-27b-prefill-c64-down-production-benchmark.json).

## M64 Prefill NVFP4 Gate/Up stop-loss

Commit `6be3a87` adds a test-only exact NVFP4 `[M64,N17408,K5120]`
table-free quad-A instance without changing production dispatch. It is
bitwise equal to two public production M32 raw-weight cp.async launches, has
zero replay mismatch, preserves guards and panel boundaries, captures one
grid-136 kernel, rejects eight invalid cases before enqueue, and reports 76
registers, 23,552 bytes static shared memory, zero local bytes, and three
active CTAs/SM.

At fixed 1.3005-GHz GPU and 3.2-GHz EMC clocks, six mirrored B-C-C-B rounds
per distribution put the isolated projection at 1.08934x aggregate. A second
screen retains the production ready/done/join topology, with Gate on the main
stream and Up on the auxiliary stream. It is exact on both branches and all
12 rounds improve, but the pair envelope reaches only 1.07884x. The current
P513 C64 profile contains 1,231.372 ms of Gate/Up union inside a 4,376.232-ms
Prefix range; supporting the later 1.03x full-Prefix gate requires at least
1.11547x at pair level, rounded to a 1.12x micro gate. The candidate therefore
stops before any full-model or production integration. A fixed-component
projection estimates only 1.02099x Prefix gain. Full protocol, hashes, and
claim limits are in the
[M64 Gate/Up rejection](metadata/qwen36-27b-prefill-nvfp4-m64-gate-up-rejection.json).

## FP8 M64 attention-output production promotion

Commit `5df6ca6` promotes the exact aligned FP8 attention-output projection
`[M64,N5120,K6144]` screened at base `e8f3ff5`. One four-accumulator M64
Tensor Core kernel now replaces two ordered production M32 dual-resident-A
launches for that shape only. Every other C64 FP8 projection retains the two-
M32 schedule, smaller tiles and Decode remain unchanged, and near-miss shapes
or unsupported alignments fail the exact low-level entry point before enqueue.
No MTP, system double/triple buffering, or Prefill/Decode overlap is added.

The production resource/correctness gate reports 69 registers/thread, 23,552
bytes static shared memory, zero local memory, and three active CTAs/SM. Both
the checkpoint-like finite fixture and exhaustive 256-code-by-four-byte-position
fixture match two public M32 launches, the frozen control, dispatcher output,
and replay bit-for-bit over 327,680 BF16 elements. All panel-boundary, guard,
input-preservation, classified-NaN, invalid-contract, and one-node CUDA Graph
checks pass. Six fixed-clock `B-C-C-B` micro rounds span 1.48995x--1.49154x
and aggregate at **1.49086x**, above the 1.10x exact-route gate.

The matched full-model protocol uses MAXN, a fixed 1.3005-GHz GPU, locked
3.2-GHz EMC, C64/max1/max-sequence-length 1024, one warmup, five measured
generations per prompt, and mirrored `B1-C1-C2-B2` processes. The table uses
the arithmetic mean of mirrored process medians:

| Prompt | Baseline Prefix | Candidate Prefix | Prefix speedup | TTFT speedup | Candidate Prefix throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| P33 | 272.3525 ms | 273.2145 ms | 0.99684497x | 0.99758798x | 117.124091 token/s |
| P65 | 515.5555 ms | 502.9500 ms | **1.02506313x** | 1.02059219x | 127.249230 token/s |
| P97 | 795.9610 ms | 783.6080 ms | **1.01576426x** | 1.01387197x | 122.510235 token/s |
| P129 | 1,046.2735 ms | 1,021.3555 ms | **1.02439699x** | 1.02206210x | 125.323651 token/s |
| P513 | 4,374.5310 ms | 4,272.4115 ms | **1.02390208x** | 1.02330551x | **119.838644 token/s** |

The unchanged P33 fallback is within its low-risk at-most-0.5% regression
gate in the formal run. A longer 10-warmup/10-measure mirrored control instead
reaches 1.001689x Prefix and 1.001418x TTFT, so short-route noise does not block
the exact dispatch. Every affected P65--P513 route clears its at-least-0.5%
Prefix gate. This admission policy is intentionally narrower than the rejected
cross-stream Gate/Up experiment's 1.03x full-Prefix requirement.

P513 Nsight closes the intended route: the target projection changes from
1,024 M32 launches totaling 378.191584 ms to 512 M64 launches totaling
279.108096 ms, a **1.355000x** cumulative-kernel improvement. The diagnostic
Prefix range falls from 4,399.928 to 4,278.702 ms. All 20 formal generations
retain ID `9419`, text `Hello`, exact 33/65/97/129/513 step counts, and zero
persistent device-memory drop. The complete Release suite reports 51 passes,
9 model/external-data skips, and zero failures. The non-MTP Decode anchor
remains frozen at **105.870500 ms/token / 9.445501816 token/s**. FlashInfer
has not been introduced by this native projection promotion. Full binary identities,
protocol, evidence hashes, gates, and limitations are in the
[FP8 M64 production record](metadata/qwen36-27b-prefill-fp8-m64-attention-output-production-benchmark.json).

## FP8 M256/M512 whole-chunk grid screen

Commit `0196751` adds a test-only whole-chunk form of the promoted FP8
attention-output CTA. M256/M512 execute the same total CTA arithmetic as
four/eight public production M64 launches. The candidate flattens all token
tiles into one grid and orders M64 CTAs N-major; a second M-major one-grid
variant separates grid-boundary/tail-wave gains from weight-locality gains.
Production dispatch, runner workspace, Decode, and the C64 request ABI are
unchanged.

Under MAXN, fixed 1.3005-GHz GPU and locked 3.2-GHz EMC clocks, each process
uses 10 warmups, 24 whole-chunk operations per timing pass, six mirrored
`B-C-C-B` and `S-C-C-S` rounds, and the exact public M64 production entry as
baseline:

| Shape | Process speedups | Median | All-round range | N-major / M-major |
| --- | --- | ---: | ---: | ---: |
| M256 | 1.26628x, 1.26451x, 1.26591x | **1.26591x** | 1.26413x--1.26706x | 1.01234x |
| M512 | 1.29047x, 1.28893x, 1.29167x | **1.29047x** | 1.28859x--1.29184x | 1.01153x |

The M512 M-major control is 1.27510x faster than repeated production M64, so
most of the selected result comes from eliminating repeated under-filled
40-CTA grid boundaries. N-major ordering adds a smaller, stable locality gain;
CUDA scheduling is not strict and performance-counter access is unavailable,
so this is not labeled a direct L2-hit-rate measurement.

Production M64 remains at 69 registers/thread, 23,552 bytes shared, zero local
memory, and three CTA/SM. N-major uses 70 registers and M-major 71 without
changing shared/local memory or occupancy. Finite M256/M512 results are
bit-exact. The exhaustive M512 fixture covers all E4M3FN byte codes and reports
zero mismatches across 2,621,440 elements, zero replay mismatches, exact class
and sign for 4,096 NaNs, intact guards, preserved inputs, and zero-node invalid
captures. The full Release suite remains 51 pass, 9 existing model/external
skips, and 0 fail.

This passes the 1.25x M512 mechanism gate, but is not an end-to-end result.
Output-only arithmetic projects to 1.01490x P513 Prefix; applying the median to
all current FP8 QKV/Z/output hotspots projects 1.05007x and still requires
direct shape measurements. The next bounded milestones are native NVFP4 and
direct QKV/Z whole-chunk screens before one public C512 workspace change. Full
evidence is in the
[whole-chunk screen](metadata/qwen36-27b-prefill-fp8-whole-chunk-grid-screen.json).

## NVFP4 M256/M512 whole-chunk down grid screen

Commit `5dc256b` applies the same one-grid mechanism to exact NVFP4 down
`[N5120,K17408]` without changing production dispatch, the C64 request ABI,
runner workspace, Decode, or MTP policy. Four/eight public production M64
launches form the baseline. M-major and N-major candidates execute the same
M64 CTA arithmetic in one 160/320-CTA grid; the N-major order places all
four/eight token CTAs for one N128 packed-weight panel next to each other.

Three independent MAXN processes use fixed 1.3005-GHz GPU and locked 3.2-GHz
EMC clocks, two scale distributions, 10 warmups, 24 logical operations per
timed pass, and six mirrored `B-C-C-B` plus `S-C-C-S` rounds:

| Shape | Process speedups | Median | All-round range | Median N-major / M-major |
| --- | --- | ---: | ---: | ---: |
| M256 | 1.29357x, 1.29708x, 1.29624x | **1.29624x** | 1.29303x--1.29764x | 1.03208x |
| M512 | 1.34513x, 1.34717x, 1.34655x | **1.34655x** | 1.34468x--1.34758x | **1.03288x** |

All 72 baseline-comparison rounds improve. The M512 control shows N-major is
about 3.29% faster than the otherwise identical one-grid M-major order, larger
than the earlier FP8 locality delta. This is stable locality evidence, but it
is not labeled a measured L2-hit-rate improvement because no hardware counter
was collected.

Production M64 and every candidate retain 23,552 bytes shared, zero local
memory, and three CTA/SM. Production reports 76 registers/thread; M256/M512
M-major and N-major report 79. Both checkpoint-like and same-bank-stress
fixtures match the repeated production M64 output bit-for-bit across
1,310,720 M256 and 2,621,440 M512 elements per route. Replay mismatches are
zero, all outputs are finite, guards and all inputs remain intact, valid Graph
topologies are 4/8 baseline nodes versus one candidate node, and 17 invalid
calls capture zero nodes. The complete Release suite reports 51 pass, 9
existing model/external-data skips, and 0 fail.

The latest P513 trace assigns 738.322 ms to down inside a 4,278.702-ms Prefix.
Applying the measured M512 median to that hotspot alone projects 190.016 ms
saved and a **1.04647x** Prefix opportunity. Combining it arithmetically with
the still-unintegrated all-FP8 whole-chunk hypothesis projects 1.10144x and
131.801 token/s, but neither number is an achieved end-to-end result. The
selected next step is isolated Gate followed by the production-like Gate/Up
pair, then direct QKV/Z screens. Only after these dominant shapes select their
routes will the public request boundary move once to C512, with C256 retained
as an internal canary rather than a second ABI release.

The shared templated kernel gained compile-time tile-order parameters, so the
precise claim is that production public dispatch, validation contract,
shared/local resources, occupancy, and default correctness remain unchanged;
cubin/SASS byte identity is not claimed. Full inputs, process measurements,
binary identity, projections, and limits are in the
[NVFP4 down screen](metadata/qwen36-27b-prefill-nvfp4-whole-chunk-down-grid-screen.json).

## NVFP4 M256/M512 whole-chunk Gate/Up pair screen

Commit `d9c8aa6` measures the actual production-like two-stream pair rather
than extrapolating from one branch. Baseline B is the public M32 chain on each
Gate/Up branch, control R is repeated test-only M64, and candidate C is one
N-major whole-chunk grid per branch. Ready, auxiliary completion, and join
events are included in every envelope; production dispatch and workspace are
unchanged.

| Shape | Process C/B speedups | Median | All-round range | Median C/R |
| --- | --- | ---: | ---: | ---: |
| M256 | 1.13439x, 1.13507x, 1.13824x | **1.13507x** | 1.13098x--1.14363x | 1.03323x |
| M512 | 1.13072x, 1.12867x, 1.12760x | **1.12867x** | 1.12617x--1.14327x | 1.02848x |

All 36 C/B rounds per shape improve, and M512 clears the predeclared 1.12x
pair gate in every process. Both scale distributions are bit-exact across B,
R, C, and C replay; outputs are finite, guards and inputs are preserved, and
17 invalid calls capture zero nodes. The candidate reports 79 registers versus
production's 76 while retaining 23,552-byte shared memory, zero local memory,
and three CTA/SM. Per branch, Graph topology changes from 8/16 public M32 nodes
at M256/M512 to one candidate node.

Applying the M512 median to the latest 1,231.372-ms Gate/Up union inside the
4,278.702-ms P513 Prefix projects 140.378 ms saved and **1.03392x** Prefix.
This is an arithmetic opportunity, not achieved full-model throughput. It
selects the route for C512 integration with C256 canary coverage; the existing
fallbacks remain required. Full evidence is in the
[Gate/Up pair screen](metadata/qwen36-27b-prefill-nvfp4-whole-chunk-gate-up-pair-screen.json).

## Native SM87 bulk causal GQA Prefill screen

Commit `3044ab5` adds a dependency-free test-only bulk full-attention kernel
using QT2, BK16, 192 threads, six Q-head warps per KV head, shared K/V tiles,
register-resident FP32 online softmax, and fused sigmoid Gate. It preserves the
production numerical boundary `FP32 attention -> BF16 -> Gate -> BF16`. The
baseline is the current Prefill-style loop of three attention kernels per
query token followed by one tile Gate kernel.

| Shape | Process speedups | Median | All-round range | Graph nodes B/C |
| --- | --- | ---: | ---: | ---: |
| C256 | 6.30241x, 6.33538x, 6.34034x | **6.33538x** | 6.27487x--6.35993x | 769 / 1 |
| C512 | 4.53722x, 4.53465x, 4.56828x | **4.53722x** | 4.52129x--4.57580x | 1,537 / 1 |

All 18 mirrored rounds per shape improve and C512 more than doubles the frozen
2x core gate. The kernel uses 64 registers, 16,384 bytes shared, zero local
memory, and permits five CTA/SM. Against the existing attention+Gate path,
C256/C512 max absolute error is 0.000244141, normalized RMSE stays below
7.43e-6, P99 absolute and relative error are zero, cosine is one, and Graph
replay is bitwise. A P17+C256 append case, guards, all inputs, and 12 zero-node
invalid captures also pass.

The latest 301.050-ms P513 attention+Gate hotspot would save 234.699 ms if the
single-layer C512 ratio transfers across the model, a **1.05804x** projected
Prefix opportunity. This projection excludes Q/K preprocessing, KV placement,
GDN, projection, and runner interactions. Production promotion therefore
requires full-path integration and token/memory/Prefix validation. Evidence is
in the [bulk GQA screen](metadata/qwen36-27b-prefill-bulk-causal-gqa-screen.json).

## FP8 C256/C512 whole-chunk QKV/Z screen

Commit `3134c38` adds test-only exact-shape M64 and whole-chunk launchers for
FP8 QKV `[N10240,K5120]` and Z `[N6144,K5120]`; production dispatch, public
runtime APIs, runner workspace, Decode, and MTP policy remain unchanged. One
fixed-clock MAXN process compares the public M32 chain (B), repeated test-only
M64 (R), one M-major grid (S), and one N-major grid (C) with 10 warmups, 24
logical operations per timed pass, and six `B-R-S-C-C-S-R-B` rounds.

| Sequential QKV then Z pair | B M32 | C N-major | C/B | Round range |
| --- | ---: | ---: | ---: | ---: |
| C256 checkpoint-like | 8.40219 ms | 5.41286 ms | 1.55227x | 1.55204x--1.55262x |
| C256 stress | 5.99729 ms | 4.26208 ms | 1.40713x | 1.40640x--1.40801x |
| C512 checkpoint-like | 16.80020 ms | 10.71280 ms | **1.56823x** | 1.56794x--1.56888x |
| C512 stress | 11.97630 ms | 8.38478 ms | **1.42834x** | 1.42824x--1.42847x |

Combining both C512 cells yields **1.50681x**, clearing the frozen 1.25x pair
gate. All 48 individual-shape and 24 pair rounds improve. Every finite B/R/S/C
and replay output is bit-exact; exhaustive M512 E4M3FN coverage for both
shapes, 4,096 classified NaNs per shape, guards, inputs, all Graph topologies,
30 zero-node invalid captures, and invalid resource queries pass. QKV and Z
share the same resource envelope: the N-major candidate uses 70 registers,
23,552 bytes shared, zero local memory, 256 threads, and three active CTA/SM;
the M-major control uses 71 registers and the same residency.

Applying the isolated ratio to the current 617.622304-ms P513 QKV+Z profile
row projects 207.735 ms saved. That is arithmetic opportunity only: the screen
uses deterministic synthetic scale fixtures, one process, and test-only entry
points. The result admits narrow exact C256/C512 production integration, after
which P257/P513 model-oracle, memory, mirrored fixed-clock Prefix, and fresh
Nsight gates remain mandatory. Full binary, log, cell, resource, and hash
evidence is in the [FP8 QKV/Z screen](metadata/qwen36-27b-prefill-fp8-whole-chunk-qkv-z-screen.json).

## GDN C256/C512 whole-span register-state rejection

Commit `9572c2a` compares the ordered production exact-M16 GDN chain with one
isolated test-only exact-C256 or exact-C512 kernel. The candidate retains each
thread's packed BF16 recurrent-state words in registers across the complete
span, while preserving per-token BF16 rounding. It reduces 16/32 production
kernel nodes to one and removes 15/31 intermediate state publication
boundaries without changing `q3x_kernels`, public dispatch, the runner,
Decode, or MTP policy.

| GDN span | Production M16 chain | Whole-span candidate | Speedup | Round range | Frozen gate |
| --- | ---: | ---: | ---: | ---: | ---: |
| C256 | 5.21686 ms | 5.08111 ms | **1.02672x** | 1.02670x--1.02681x | 1.03x |
| C512 | 10.4250 ms | 10.2336 ms | **1.01871x** | 1.01868x--1.01881x | 1.03x |

All five mirrored `B-R-R-B` rounds per shape improve. Output and final state
are bitwise against the production chain for in-place, disjoint, guarded, and
replay paths; two R256 launches also equal one R512 launch. Input, finite,
invalid zero-node capture, and Graph topology gates pass. Both candidates use
64 registers, 34,056 bytes shared, zero local memory, 256 threads, and retain
four active CTA/SM.

Both shapes nevertheless miss the frozen 1.03x first-process gate, so the
candidate is rejected. The performance-enabled test binary intentionally
returns failure with two threshold assertions; a trailing `tee` can hide that
status unless shell `pipefail` is enabled. The performance segment is skipped
in the passing default test. External replication, runner/full-model work,
NCU, and Nsys are omitted by stop-loss, and production remains unchanged. The
result closes longer sequential register lifetime with the same recurrence
body; a future GDN attempt must use a materially different algorithm or
dataflow. Full log, binary, round, resource, Graph, and exit-status evidence is
in the [GDN rejection record](metadata/qwen36-27b-prefill-gdn-whole-span-register-state-rejection.json).

## Matched vLLM plus FlashInfer Prefill reference

The offline raw-token probe now has three independent processes across
P65/P129/P257/P513/P1025. Each process uses three warmups and ten measured
batch-one/output-one requests, disables prefix cache, chunked Prefill, and
speculation/MTP, and verifies BF16 caches plus the actual loaded backends.
All 150 engine-core timing records are trusted and all return first token ID
9419.

| Profile | Process median token/s | Median of processes | Median scheduled-to-first-token |
| --- | --- | ---: | ---: |
| P65 | 236.380, 237.086, 236.115 | **236.380 token/s** | 274.981 ms |
| P129 | 313.128, 312.828, 312.532 | **312.828 token/s** | 412.367 ms |
| P257 | 375.178, 372.911, 373.579 | **373.579 token/s** | 687.940 ms |
| P513 | 410.797, 411.629, 411.385 | **411.385 token/s** | 1,247.007 ms |
| P1025 | 432.738, 436.037, 431.969 | **432.738 token/s** | 2,368.642 ms |

The worker reports FlashInfer for 16 full-attention layers, GDN for 48 layers,
Triton/FLA GDN Prefill, and Marlin FP8/NVFP4 projections. The vLLM-side
reference is therefore established. Native's current P513 `Prefix` divided by
vLLM TTFT is directionally about 3.43x, far smaller than the prior 16--67x
surface comparison. It is not yet a formal speedup ratio: native `Prefix`
omits the final prompt token and LM head, while vLLM scheduled-to-first-token
includes them. A native complete-prompt boundary and mirrored ordering remain
required. Full records and hashes are in the
[vLLM reference](metadata/qwen36-27b-vllm-flashinfer-prefill-reference.json).

## C512 request boundary and fallback baseline

Commit `6be943e` publishes package/ABI 0.4.0 with a 512-token Prefill request
capacity and an explicit `{512,256,64,32,tail<=31}` scheduler. This is a
correctness and ownership boundary, not a performance promotion: exact C256
and C512 tiles initially retain the existing M32/C64 fallbacks until their
narrow optimized routes pass full-model admission.

The generic projection contract remains capped at C64. FP8 attention-output
and NVFP4 down selectors now test exact C64 rather than inheriting the request
maximum; residual/RMS preserves ordered M32 arithmetic through C512. For large
full-attention tiles that start before position 64, the eligible prefix keeps
the existing fused per-token GQA/Gate route and only the suffix uses the
reference split path.

At default max sequence 128, C256/C512 arenas are exactly 131,426,304 and
174,991,360 bytes. The absolute C512 arena ceiling is 17,437,720,576 bytes.
Release and Werror builds pass; the default suite reports 51 passes, 11
expected model/external-data skips, and zero failures across 62 tests. C256 and
C512 model E2E registrations remain skipped without the model environment, so
no throughput or output promotion is claimed. See the
[C512 boundary record](metadata/qwen36-27b-prefill-c512-request-boundary.json).

## Persistent packed NVFP4 Down P0 rejection

Commit `03336b6` tests the smallest scheduling-only precursor to a
Marlin-inspired Down kernel. The exact M256 candidate reorders one Down
projection into NK64/NK256 sidecars and replaces the selected 160-CTA N-major
grid with 16 persistent CTAs statically striding over the same 160 M64xN128
logical tasks. It intentionally retains the old synchronous shared-memory
decode/WMMA body; it does not implement four-stage `cp.async`, register E2M1
MMA, or reuse B arithmetic across multiple M64 tasks.

All three scale distributions are bit-exact across 1,310,720 outputs, replay,
guards, inputs, invalid capture, and single-node Graph contracts. The candidate
uses 80 registers, 23,552 bytes shared, zero local memory, and has a theoretical
three-CTA/SM occupancy, but grid 16 launches only one CTA per SM. Performance
is **0.511096x**: baseline 4.383274 ms versus candidate 8.576229 ms, with all
18 rounds between 0.510571x and 0.511604x. It fails both the 1.0 stop-loss and
the frozen 1.20x selection gate.

This closes fixed-grid persistence plus byte reordering without computational
reuse. It does not reject a true persistent kernel: the next valid attempt must
reuse each staged B tile across at least M128/M256 work and/or implement the
actual asynchronous packed-B/register-decode pipeline before reducing the
grid. See the
[P0 rejection](metadata/qwen36-27b-prefill-nvfp4-down-persistent-packed-p0-rejection.json).

## FP8 C256/C512 whole-chunk production promotion

Commit `10c4c85` promotes one fixed N-major grid for each exact aligned FP8
QKV `[N10240,K5120]`, Z `[N6144,K5120]`, and attention-output
`[N5120,K6144]` projection at C256/C512. The generic projection API remains
capped at C64. Structural or alignment near misses preserve the established
ordered C32 route; malformed payloads and full-span aliases fail before
enqueue. Both device companion-scale pointers remain part of the validated
FP8 payload even though this narrow kernel uses the host weight scale. Decode,
default C1, request workspace, and persistent allocations are unchanged.

One frozen bulk-GQA-plus-Down baseline binary and one Release candidate binary
were measured at fixed 1.3005-GHz GPU and locked 3.2-GHz EMC clocks in
`B1-C1-C2-B2` order. Each process loaded the pinned model once, used batch one,
C512, max1, one warmup, and five measured generations per prompt:

| Prompt / phase | B1 | C1 | C2 | B2 | Mirrored B | Mirrored C | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| P257 Prefix | 1,983.008 ms | 1,805.168 ms | 1,804.541 ms | 1,981.938 ms | 1,982.4730 ms | 1,804.8545 ms | **1.098411534x** |
| P257 TTFT | 2,090.958 ms | 1,913.028 ms | 1,912.768 ms | 2,089.720 ms | 2,090.3390 ms | 1,912.8980 ms | **1.092760304x** |
| P513 Prefix | 3,970.878 ms | 3,612.650 ms | 3,608.373 ms | 3,968.961 ms | 3,969.9195 ms | 3,610.5115 ms | **1.099544898x** |
| P513 TTFT | 4,079.767 ms | 3,721.437 ms | 3,717.176 ms | 4,077.709 ms | 4,078.7380 ms | 3,719.3065 ms | **1.096639387x** |

Candidate prefix throughput is **141.839688 token/s at P257** and
**141.808162 token/s at P513**. Complete-prompt `P/TTFT` throughput is
**134.351126** and **137.928939 token/s**, respectively. All 40 measured
generations return token 9419 (`Hello`) with exact 257/513 step counts. A
streamed canonical extraction of routing, prompt IDs, generated output, and
complete steps has SHA-256
`6cc61feb53776048e4f9ea2542d8d95447c5643327a8cefcb77a3580b7980b5c`
for each of the four logs; no intermediate contract file was materialized.
Every process reports zero persistent memory drop.

The post-promotion P513/C512 trace closes a 3,625.119136-ms Prefix NVTX wall
range against 3,624.145984 ms of GPU-projected span. Exact whole-chunk rows are
QKV 48 launches / 262.804224 ms, Z 48 / 158.643360 ms, and attention output
64 / 212.131040 ms. Thus the intended 160 production nodes replace the exact
previous C32 contract's theoretical 2,560 nodes. This 16x launch-count
reduction is a topology result; the wall-time gain also includes decoded-weight
reuse across M64 panels and is not attributed to launch overhead alone.

Using the already measured stock vLLM complete-prompt accounting, P257/P513
reach 373.579/411.385 token/s versus native's 134.351126/137.928939 token/s,
leaving **2.780617x/2.982587x** gaps. The native implementation uses neither
FlashInfer nor MTP, and this result is not claimed to approach the user's
separately tuned 2k--8k token/s vLLM range. It is a fixed-clock batch-one
production promotion, not a serving-throughput, concurrency, tail-latency,
power, or energy claim. Full binary, log, test, Nsight, hash, and fallback
evidence is in the
[production benchmark](metadata/qwen36-27b-prefill-fp8-whole-chunk-production-benchmark.json).

## FP8 C256/C512 full-attention Q/K/V whole-chunk screen

Commit `de86613` adds a production-unreachable screen that applies the same
M64 arithmetic to full-attention Q `[N12288,K5120]` and K/V
`[N1024,K5120]`. It compares the
current Q M32 and K/V M8 CUDA chains (B), repeated test-only M64 launches (R),
one M-major grid (S), and one N-major grid (C). The timing protocol uses 10
warmups, 24 operations per pass, six mirrored `B-R-S-C-C-S-R-B` rounds, two
synthetic scale distributions, and fixed Orin clocks. Production dispatch,
the public runtime API, request workspace, Decode, and MTP policy are
unchanged.

| Shape / tokens / distribution | B | S | C | C versus B |
| --- | ---: | ---: | ---: | ---: |
| Q C256 checkpoint-like | 6.11637 ms | 4.08295 ms | 4.03269 ms | **1.51670x** |
| Q C256 stress | 4.32268 ms | 3.22455 ms | 3.13630 ms | **1.37827x** |
| Q C512 checkpoint-like | 12.23170 ms | 8.09723 ms | 8.00805 ms | **1.52743x** |
| Q C512 stress | 8.64306 ms | 6.38080 ms | 6.21283 ms | **1.39116x** |
| K C512 checkpoint-like | 5.94517 ms | 0.749165 ms | 0.746083 ms | **7.96850x** |
| K C512 stress | 5.21541 ms | 0.615514 ms | 0.608698 ms | **8.56814x** |
| V C512 checkpoint-like | 5.95843 ms | 0.750072 ms | 0.747286 ms | **7.97343x** |
| V C512 stress | 5.19512 ms | 0.617100 ms | 0.608949 ms | **8.53129x** |

The sequential C512 Q-then-K-then-V envelope reaches **2.54717x** for the
checkpoint-like fixture and **2.57488x** for the stress fixture; their combined
aggregate is **2.55933x**, above the frozen 1.25x gate. Every individual and
sequence round improves over its own production baseline. Q is bit-exact with
the M32 baseline. K/V are bit-exact across R/S/C and replay and pass the CUDA
M8 tolerance gate; their largest observed baseline differences are 0.00195312
absolute / 0.0052356 relative for K and 0.0000038147 / 0.0042735 for V.
Exhaustive M512 E4M3FN-by-four-byte-position coverage for Q and the shared K/V
kernel, 4,096 classified NaNs per representative shape, guards, immutable
inputs, Graph topology, 18 zero-node invalid captures, full-span aliases, and
resource/cross-shape rejection gates all pass.

The N-major candidates use 70 registers, 23,552 bytes shared, zero local
memory, 256 threads, and retain three active CTA/SM. C256 K/V expose only 32
CTAs and N-major is slightly slower than the M-major control there, although
both one-grid layouts are much faster than the 32-launch M8 baseline. The
screen therefore selects the exact shapes for a narrow production-promotion
gate, but does not prescribe one layout for every token count. No full-model,
TTFT, throughput, or end-to-end gain is attributed until dispatch integration
passes exact P257/P513 model, memory, mirrored fixed-clock, and fresh-profile
gates. Full evidence is in the
[full-attention screen](metadata/qwen36-27b-prefill-fp8-whole-chunk-full-attention-screen.json).
