# Qwen3x-Orin roadmap

This roadmap is ordered by technical dependency, not by calendar date. A phase
is complete only when its exit criteria are reproducible on a Jetson AGX Orin.
The project has a complete Phase 1 direct-load pipeline, a fixed-oracle native
Phase 2 path for the pinned Qwen3.6 27B model, and active Phase 3 SM87
performance work. Qwen3.6 dense and MoE metadata evidence is pinned and
reproducible; Qwen3.5 ModelOpt
packaging remains an explicit gap rather than being inferred from shape
compatibility.

## Phase 0 — Bootstrap and evidence capture

Deliverables:

- CMake/C++/CUDA repository skeleton targeting `sm_87`.
- Checked CUDA device discovery and a minimal test executable.
- Model-support descriptors separated from kernels.
- Scripts or tools that inspect target checkpoint headers and quantization
  metadata without loading all weights.
- Recorded representative dimensions and formats for dense FP4, dense FP8,
  and MoE expert microbenchmarks.
- Apache-2.0 license, source-provenance policy, design, and roadmap.

Exit criteria:

- The bootstrap tree configures, builds, and tests on the target Orin.
- Each planned model revision has a checked-in metadata report or reproducible
  command, and unknown revisions fail closed.
- Benchmark shapes come from inspected checkpoint metadata rather than memory
  or model-name assumptions.

Current evidence: the Orin build/device gate, Qwen3.6 pinned reports,
metadata-only inspector, negative-format/revision cases, and checkpoint-derived
benchmark shapes are recorded in [PHASE0_EVIDENCE.md](PHASE0_EVIDENCE.md).
The two Qwen3.5 rows remain catalogued architecture targets until a concrete
ModelOpt checkpoint revision is available and pinned.

## Phase 1 — Quantization references and weight pipeline

Deliverables:

- [done] Safetensors index/header reader with checked offsets and bounded
  metadata-only I/O.
- [done] Strict ModelOpt mixed-precision metadata parser and pinned per-module
  dispatch records for the Qwen3.6 artifacts.
- [done] Exhaustive E2M1 decoder plus E4M3 and global-scale reference handling.
- [done] CPU or straightforward CUDA references for NVFP4 W4A16 and FP8
  W8A16.
- [done] Strict ownership, membership, range, and aggregate-payload validation
  across every indexed shard in the fully materialized 27B checkpoint.
- [done] Authenticated one-pass, double-buffered direct loader with one
  256-byte-aligned text-only CUDA arena, a pre-allocation memory gate, and
  automatic Linux AF_ALG SHA-256 acceleration with a portable fallback.
- Lossless Marlin-oriented repack and scale preprocessing primitives.
- Versioned `.q3x` container prototype with source hashes and atomic output.

Exit criteria:

- Repacking round-trips on synthetic and real representative tensors.
- Reference W4A16/W8A16 results match an independent implementation within
  documented per-operation tolerances.
- Direct loading streams and authenticates data without retaining two complete
  model copies; the same property remains required of the future repacker.
- Corrupt, stale, truncated, or wrong-architecture caches are rejected.

## Phase 2 — 27B text-only correctness

Deliverables:

- [done, reference] BF16 common decode kernels, exact Q/Gate layout split, and
  BF16/FP8/NVFP4 allocation-free projection dispatch. cuBLASLt/optimized
  backends remain performance work.
- [done, native reference; fixed-oracle token/text gate passed] Hybrid sequence
  layers, attention, recurrent/DeltaNet state, normalization, positional
  encoding, embeddings, and output projection for the pinned 27B revision.
- [done, direct] Direct checkpoint loading; optional `.q3x` loading remains.
- [done, native reference; fixed-oracle token/text gate passed] Bounded BF16
  KV/state cache, pinned tokenizer/chat integration, sequential greedy
  generation, and a script-friendly command-line runner.
- [done, memory substrate] Single-arena batch-one Conv/GDN state, full-attention
  KV cache, allocation-free workspace, BF16-rounded RoPE cache, and async reset.
- [done, oracle] Pinned target-device greedy tokens and log probabilities from
  an independent runtime for the first fixed prompt.
- [done, oracle] Two-phase 64-layer boundary fixtures for both the original
  runtime cache policy and an explicitly all-BF16 cache/state policy.

Current native evidence: the fixed 19-token prompt produces all 26 oracle
output IDs, decoded UTF-8 text, and `<|im_end|>` stop exactly on the target
Orin. The first reference record used an 82,505,216-byte request arena, a
213.845-second portable-hash cold load, and 49.212 seconds of sequential
generation. The current AF_ALG loader reduced a measured resident load to
21.485 seconds; the SM87 full fixed-oracle CTest took 52.22 seconds after that
loader milestone and 45.56 seconds after packed-x8, without changing its exact
19/26-token and 44-step result. The packed-x4 FP8 milestone reduced the same
fixed-oracle CTest to 40.60 seconds, again with the exact 19/26-token and
44-step result. The first C8 chunked-prefix run preserves that same exact
19/26-token and 44-step oracle result; trace mode continues to use the scalar
C1 order. The post-C8 kernel sequence through `5fe0ae0` preserves the same
exact result at the optimized C8 dispatch. The C16 runtime/GDN sequence
(`dda4e3a`/`c90f37e`) and FP8/NVFP4 Tensor Core sequence
(`e7283d6`/`33948e3`) preserve it again with the prefix scheduled as `16+2`.
Native boundary hashes are not
required to equal vLLM hashes
because independent checkpoint
scales versus fused requantization and sequential versus chunk BF16 GDN updates
have different rounding/order. Tolerance-based boundary characterization and
broader prompt repeatability remain in progress.

Exit criteria:

- Fixed prompts produce reference-aligned intermediate activations and logits.
- Greedy generation is stable across repeated runs within the documented
  accumulation policy.
- Startup reports a complete memory budget and short-context generation fits
  on a 64 GB Orin without swap dependence.

## Phase 3 — Ampere weight-only performance

Deliverables:

- [done, explicit opt-in] `sm_87` NVFP4 and FP8 single-token GEMV kernels.
- [done, shape-gated] Canonical NVFP4 packed-x8 M=1 decode with scalar fallback.
- [done, shape-gated] Canonical FP8 packed-x4 M=1 decode with scalar fallback.
- [done, bounded path] SM87 W4A16/W8A16 weight-reuse kernels, C16 causal
  Conv/GDN state, and layer-major prompt-prefix dispatch for `C=2..16`, with
  C1 fallback/default and M9..M15 split into M8 plus the remainder.
- [done, control-plane seam] Explicit internal Prefill/Decode plans distinguish
  prefix work, the final prompt/logits step, and subsequent decode feedback
  while retaining the public engine contract and existing runner/device
  schedule.
- [done, production-gated] Eight-row lane-striped GDN state updates for
  batch-one M1 and bounded C2 through C16 tiles, retaining the four-row
  lane-striped predecessor as a same-binary test baseline and preserving
  validation, aliasing, per-row arithmetic, and token-recurrence contracts.
- [done, shape-gated] Aligned canonical NVFP4 M=8 output-row pairing with
  independent fallbacks for other shapes.
- [done, exact-shape gated] Compile-time NVFP4 M=8 specializations for
  `[17408,5120]` and `[5120,17408]`, with the generic row-pair path retained.
- [done, exact-shape gated] Compile-time FP8 M=8 specializations for
  `[10240,5120]`, `[5120,6144]`, `[6144,5120]`, `[12288,5120]`, and
  `[1024,5120]`, with the generic row-pair path retained.
- [done, exact-shape gated] Fixed-M16 FP8 BF16 Tensor Core kernels for
  `[10240,5120]`, `[5120,6144]`, `[6144,5120]`, and `[12288,5120]`, with
  two-M8 fallback for `[1024,5120]`, other shapes, and insufficient alignment.
- [done, exact-shape gated] Fixed-M16 NVFP4 decode-to-BF16 Tensor Core kernels
  for `[17408,5120]` and `[5120,17408]`, with two-M8 fallback elsewhere.
- [done, exact-shape gated] Fused full-attention FP8 M1 K/V projection for
  paired `[1024,5120]` matrices, with ordered independent fallbacks elsewhere.
- [done, exact-shape gated] Fused linear-attention FP8 M1 QKV/Z projection for
  ordered `[10240,5120]` and `[6144,5120]` matrices, with independent
  fallbacks for C2..C16, near-miss shapes, other backends, and insufficient
  alignment.
- [done, exact-shape gated] CTA activation-staged NVFP4 M1 down projection for
  `[5120,17408]`, retaining the direct XOR test baseline plus scalar and
  near-miss fallbacks.
- [done, exact-shape gated] CTA activation-staged NVFP4 M1 gate/up projection
  for `[17408,5120]`, preserving the 8-byte activation alignment contract and
  preceding fallbacks elsewhere.
- [done, exact-shape gated] CTA activation-staged NVFP4 M1 language head for
  `[248320,5120]`, preserving the 8-byte activation alignment contract and
  preceding fallbacks elsewhere.
- Shape-driven kernel registry and measured dispatch thresholds.
- Dense-prefill comparison among Marlin-style, cuBLASLt-assisted, and reference
  paths.
- [done, initial diagnostic] Reproducible single-load benchmark/replay harness
  with Jetson power/clock metadata.

Current evidence: Nsight attributes 98.4% of reference GPU kernel time to
weight-only GEMV. The first direct-BF16 SM87 backend passes awkward/K=5120/
K=17408 numerical gates and the complete 19/26-token fixed oracle. In a
same-binary, two-prompt comparison it reduced median TTFT by 45.60%, total
two-token generation by 45.47%, and subsequent-token latency by 43.05%
(1.838x, 1.834x, and 1.756x speedups). The packed-x8 NVFP4 milestone then
reduced those SM87 medians by another 25.83%, 25.69%, and 23.40%, reaching
8.280-second TTFT and 499.086 ms per subsequent token. Its follow-up profile
assigns 51.2% of GPU time to FP8 projections and 43.8% to NVFP4, selecting FP8
conversion as the next M=1 target. The packed-x4 FP8 milestone then reached
6.107-second TTFT and 385.181 ms per subsequent token, reducing the packed-x8
medians by 26.24% and 22.82%. Its same-binary kernel gate measures 2.02x to
2.36x speedups on the recorded production and mixed-code shapes. The follow-up
profile assigns 34.2% of GPU time to packed-x4 FP8 and 59.1% to packed-x8
NVFP4. The historical bounded small-M milestone batches only the prompt prefix
up to C8. In the same two-prompt/two-output-token diagnostic shape, C8 reduced
median TTFT from 6,107.420 to 2,005.784 ms and total generation from 6,492.908
to 2,389.125 ms, while median subsequent-token latency remained effectively
flat (385.467 versus 383.320 ms). The 64-position request arena increased by
1,190,912 bytes, and the C8 full-model gate retained the exact 19 prompt IDs,
26 output IDs, and 44 steps. The post-C8 kernel sequence through `5fe0ae0`
then reached 1,020.755 ms TTFT, 1,205.989 ms total generation, and 185.108 ms
for the subsequent token without increasing the C8 request arena. That is a
further 49.11%, 49.52%, and 51.71% reduction from the first C8 medians, and
the exact 19/26-token, text, stop, and 44-step gate still passes. The final
NVFP4 M=8 specialization provides a same-binary 1.16079x weighted speedup and
reduces normalized SASS from 1,272 to 1,144 instructions while retaining 64
registers, 1,088 bytes of shared memory, and zero stack/local memory. It is
limited to the two exact production shapes; other aligned M=8 shapes retain
the generic row-pair path. The final FP8 M=8 specialization provides a
same-binary 1.13694x call-weighted speedup across five exact production shapes
and reduces normalized SASS from 1,864 to 784 instructions while retaining 48
registers, 1,536 bytes of shared memory, and zero stack/local memory or spills.
Other aligned FP8 M=8 shapes retain the generic row-pair path.
The completed C16 milestone then provides 2.41756x FP8 and 1.56406x NVFP4
production-call-weighted fixed-M16 speedups over two M8 launches. In a mirrored
same-binary comparison with eight measured samples per policy, C16 reduced
median TTFT from 1,021.088 to 761.037 ms (25.468%) and total generation from
1,206.170 to 946.217 ms (21.552%); median subsequent-token latency remained
effectively flat at 185.084 versus 185.186 ms. The 64-position arena grows from
85,011,968 bytes at C8 to 86,373,376 bytes at C16, while the exact 19/26-token,
text, stop, and 44-step gate still passes. A final C16 Nsight diagnostic records
929.615 ms across 9,210 kernel instances versus the historical optimized-C8
trace's 1,192.639 ms across 10,107 instances; this is cross-commit hotspot
context rather than a release or serving-throughput claim. See the
[C16 metadata record](metadata/qwen36-27b-c16-tensor-core-prefill-benchmark.json).
The subsequent FP8 K/V pair clears same-binary 1.74310x checkpoint-like and
2.45187x stress micro-gates. In matched max-26-token profiles it reduces that
projection work from 39.328192 to 31.316608 ms and removes 416 launches; a
mirrored single-load generation diagnostic measures a smaller 7.123 ms
(0.196330118%) average-of-medians reduction while retaining the exact
19/26-token, text, stop, and 44-step gate. This is exact-shape diagnostic
evidence, not a release claim; see the
[FP8 K/V metadata record](metadata/qwen36-27b-fp8-kv-pair-benchmark.json).
The subsequent NVFP4 data-reuse milestone moves down projection from indexed
dual-iteration to adjacent-lane XOR-dual and stages the 10-KiB lm-head
activation once per CTA. Its matched max-26 profile saves 31.623456 ms across
the two target groups and 30.878208 ms (0.871519514%) across all CUDA kernels.
A B-C-C-B diagnostic against an independently rebuilt base commit reduces
average total generation by 33.824 ms (0.953137500%) and subsequent-token
latency by 1.3115 ms (1.101855469%), with the full exact oracle retained. See the
[NVFP4 data-reuse record](metadata/qwen36-27b-nvfp4-data-reuse-benchmark.json).
The next gate/up activation-reuse milestone stages the same 10-KiB input once
per CTA. Its same-binary gate clears 1.01436x checkpoint-like and 1.02093x
same-bank-stress speedups. The matched max-26 profile saves 20.130560 ms across
3,328 gate/up kernels and 21.459840 ms (0.611016668%) across all CUDA kernels.
The independent-base B-C-C-B diagnostic reduces average total generation by
17.426 ms (0.495685%) and subsequent-token latency by 0.668 ms (0.567253%),
while retaining the full exact oracle. See the
[NVFP4 gate/up activation-staging record](metadata/qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json).
The following down activation-reuse milestone stages its 34-KiB input once per
CTA. Its same-binary gate clears 1.02862x checkpoint-like and 1.03026x
same-bank-stress speedups. The matched max-26 profile saves 20.081440 ms across
1,664 down kernels and 19.112800 ms (0.547536%) across all CUDA kernels. The
saved-base B-C-C-B diagnostic reduces average total generation by
20.0655 ms (0.573569%) and subsequent-token latency by 0.7720 ms (0.659252%),
with the full exact oracle retained. See the
[NVFP4 down activation-staging record](metadata/qwen36-27b-nvfp4-down-activation-staged-benchmark.json).
The subsequent GDN row8 milestone retains 40 registers and zero local/stack
memory while increasing shared memory from 18,056 to 34,568 bytes. Direct
occupancy queries report six active row4 blocks versus four row8 blocks per SM;
the 48-block row8 grid still averages three blocks per SM. It clears 1.41995x
through 1.53455x same-binary M1/M2/M8/M16 gates and measures 1.45587x under the
current profile call weights. The matched max-26 profile reduces GDN time from
87.167840 to 60.100480 ms and all CUDA time by 24.720928 ms (0.712094%). Its
detached-base B-C-C-B diagnostic reduces average total generation by 22.1845
ms (0.637601%) and TTFT by 11.8840 ms (2.084512%), with the full exact oracle
retained. See the
[GDN row8 metadata record](metadata/qwen36-27b-gdn-eight-row-benchmark.json).
The following exact FP8 QKV/Z milestone replaces the two linear-attention M1
launches with one topology-preserving two-phase launch. Five frozen
same-binary processes measure 1.05501x to 1.05868x on actual checkpoint bytes
and 1.00832x to 1.01092x on the same-bank stress guard. The matched max-26
profile removes 1,248 launches, reduces target work from 634.147712 to
615.753920 ms, and reduces all CUDA time by 18.585920 ms (0.539336%). Its
detached-base B-C-C-B diagnostic reduces average total generation by 18.4605
ms (0.534646%) and subsequent-token latency by 0.7280 ms (0.628746%),
while C1/C8/C16 retain the exact 19/26-token, text, stop, and 44-step oracle.
This remains unlocked-clock diagnostic evidence; see the
[FP8 QKV/Z fusion metadata record](metadata/qwen36-27b-fp8-qkv-z-fusion-benchmark.json).
Separately, default AF_ALG authentication
reduced the resident-load phase by 89.45% in a diagnostic historical
comparison. The projection backend remains default-off while prompt and shape
coverage expands. See
[PERFORMANCE_BASELINE.md](PERFORMANCE_BASELINE.md).

Exit criteria:

- All optimized kernels pass Phase 1 numerical tests and awkward-shape tests.
- Representative kernels show a material, repeatable improvement over the
  reference path without increasing model-load peak beyond its budget.
- End-to-end 27B decode and time-to-first-token measurements are published with
  enough environment detail to reproduce them.

No throughput target is fixed before Phase 0 measurements. This prevents a
speculative estimate from becoming an accidental compatibility promise.

## Phase 4 — 35B-A3B MoE

Deliverables:

- Router, top-k, routing-weight normalization, shared-expert, and scatter/reduce
  semantics for a supported 35B-A3B revision.
- Simple per-expert execution path for numerical reference.
- Token-to-expert sorting and grouped/fused FP4 expert execution.
- Decode, prefill, and multi-token dispatch policies that avoid per-expert
  launch explosions.
- MoE-specific workspace and memory planning.

Exit criteria:

- Router choices, routing weights, expert intermediates, and final layer
  outputs match trusted fixtures.
- Grouped execution matches the simple path within documented tolerances.
- Full text generation runs within the 64 GB budget.
- Profiling demonstrates that optimized execution no longer launches a naive
  gate/up/down sequence independently for every selected expert and token.

## Phase 5 — Serving and long-context foundations

Deliverables:

- Paged KV/state allocation and request lifecycle management.
- FP8 KV option after independent accuracy validation.
- Continuous batching and cancellation-safe scheduler.
- Streaming OpenAI-compatible text API with explicit supported fields.
- Memory admission control, structured metrics, and graceful error responses.

Exit criteria:

- Mixed prompt lengths and cancellations do not leak cache pages or corrupt
  another request.
- Server and CLI use the same model/runtime interfaces and produce equivalent
  deterministic output.
- Context-limit and out-of-memory decisions happen before unsafe partial work.
- Concurrency benchmarks publish latency percentiles as well as throughput.

## Phase 6 — MTP, vision, and broader optimization

These features begin only after both text models have stable correctness and
performance baselines.

Possible work:

- MTP loading, verification, acceptance logic, and tuned small-batch kernels.
- Vision encoder and multimodal preprocessing for explicitly supported model
  revisions.
- Prefix caching and cache persistence where correctness and privacy boundaries
  are clear.
- CUDA graph capture for stable launch shapes.
- Auto-tuning across Orin power modes and JetPack releases.
- Additional Qwen3.5/3.6 revisions when their architecture and formats fit the
  engine's explicit compatibility contract.

Exit criteria are feature-specific and must include independent correctness,
memory, and end-to-end tests. MTP speedup is reported together with acceptance
rate and any output-policy difference; vision support is not declared from
text-only model compatibility.

## Cross-cutting release gates

Every versioned release must include:

- exact supported checkpoint revisions or hashes;
- build and runtime requirements;
- clean build and test results on a named Jetson/JetPack configuration;
- model memory budgets and benchmark methodology;
- known numerical and functional limitations;
- an attribution audit covering copied, adapted, and vendored code;
- no model weights or other separately licensed artifacts in release archives.

Work may move between phases when measurements justify it, but correctness,
metadata-driven compatibility, bounded memory, and license provenance remain
release gates rather than optional cleanup.
