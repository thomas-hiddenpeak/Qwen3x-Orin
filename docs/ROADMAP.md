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
  batch-one M1 and bounded C2 through C15 tiles, retaining the four-row
  lane-striped predecessor as a same-binary test baseline and preserving
  validation, aliasing, per-row arithmetic, and token-recurrence contracts.
- [done, exact-shape production promotion] Register-resident BF16 GDN state for
  exact C16. Each thread keeps 64 BF16 state elements in 32 packed U32 words
  across the full recurrence, preserves every token's BF16 boundary, and
  retains row8 as the explicit same-binary predecessor and C2-C15 fallback.
- [done, post-promotion diagnostic] Fixed-frequency P513/C32 Prefill and
  P19/C32/max26 Decode phase profiles with exact NVTX-to-kernel closure,
  two-stream-aware Prefill marginal exposure, Decode idle/overlap bounds, and
  a phase-local-before-general-buffering priority decision.
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
- [done, production promotion] Two-slot `float[2][4][8]` CTA-local QKV/Z
  reduction-scratch ping-pong, preserving the per-body producer barrier and
  1,536/768 topology while removing the tail barrier. Exhaustive code/race,
  replay, Graph, resource, frozen actual/stress, and P19/P64/P513 B-C-C-B gates
  pass. This is an intra-kernel scratch pipeline, not systemwide double/triple
  buffering or Prefill/Decode overlap.
- [done, production promotion] Two-slot `float[2][4][8]` CTA-local
  full-attention Q+K/V reduction-scratch ping-pong on the exact aligned
  `[12288,5120]` plus paired `[1024,5120]` M1 route. Exhaustive Q/K/V code,
  ordered handoff/race, replay, Graph, resource, five-process actual/stress,
  and P19/P64/P513 B-C-C-B gates pass. This is also an intra-kernel scratch
  pipeline, not a general runtime buffer or Prefill/Decode overlap.
- [done, exact-shape gated] CTA activation-staged NVFP4 M1 down projection for
  `[5120,17408]`, retaining the direct XOR test baseline plus scalar and
  near-miss fallbacks.
- [done, exact-shape gated] CTA activation-staged NVFP4 M1 gate/up projection
  for `[17408,5120]`, preserving the 8-byte activation alignment contract and
  preceding fallbacks elsewhere.
- [done, Decode-runner-only production promotion] The exact aligned M1
  post-attention residual/norm/gate/up/SiLU route keeps independently
  BF16-rounded gate/up values in CTA-local shared memory and elides only the
  runner-dead up publication. The generic double-output APIs and every
  fallback retain their observable up output; the following same-stream down
  projection overwrites the validated workspace before any read.
- [done, exact-shape gated] CTA activation-staged NVFP4 M1 language head for
  `[248320,5120]`, preserving the 8-byte activation alignment contract and
  preceding fallbacks elsewhere.
- [done, production-equivalent] Private shape-driven SM87 projection registry,
  with exhaustive route/alignment/near-miss coverage, an identical ordered
  13,558-launch contract, and noise-level detached B-C-C-B regression evidence.
- [done, initial matrix] Direct M1/M2/M8/M16 route measurements plus P19
  C1/C2/C8/C16, tokenizer-pinned P33/P65/P129/P513, and matched Nsight
  launch/time attribution.
- [done, composite baseline] Bounded C32 outer prefill with complete-span
  validation, M16-first projection composition, ordered C16 Conv/GDN and
  Q/K+RoPE subtiles, one stream, and one outer state commit.
- [done, exact-shape gated] Fixed-M32 FP8 Tensor Core projections for four
  production shapes, with two-M16 fallback elsewhere.
- [measured and rejected] Test-only U16 codebook swizzles for those four FP8
  M32 shapes. Mode 1 (`x ^ (x >> 5)`) reaches only 1.00308x on the
  hash-pinned, `48:64:48:16` weighted actual tensors; mode 2
  (`x ^ (x >> 5) ^ (x >> 6)`) regresses to 0.988108x. Large synthetic
  same-bank gains are retained only as mechanism evidence. Production keeps
  the unswizzled dual-resident-A route and all candidate hooks are removed.
- [done, productionized exact shape] Attention-values for Q24/KV4/D256, with
  the generic predecessor retained as a test baseline and fallback elsewhere.
- [done, exact-shape gated] Fixed-M32 NVFP4 Tensor Core projections for
  `[17408,5120]` and `[5120,17408]`, with two-M16 fallback elsewhere.
- [done, exact-shape gated] Masked-M32 NVFP4 Tensor Core projections for exact
  M18 `[17408,5120]` and `[5120,17408]`, with exact external C18 capacity and
  ordered M16+M2 fallback elsewhere.
- [done, exact-shape gated] Runtime-valid-count masked-M32 NVFP4 Tensor Core
  projections for exact M17 and M19-M31 `[17408,5120]` and `[5120,17408]`,
  with exact external M-row capacity and ordered M16-plus-at-most-M8 fallback
  elsewhere.
- [measured and rejected] Generalize gate/up dual-stream scheduling across exact
  M17-M32, including fixed M18 and the existing M32 control. All 16 per-M
  synthetic gates passed, but the representative whole-model result reached
  only 1.007059x against the required 1.01x, so production retains serial
  M17-M31/M18 scheduling and the existing M32-only auxiliary stream.
- [measured and rejected] Replace Decode M1 NVFP4 scale/nibble multiplication
  with a canonical 8-KiB full-product table in the exact fused
  residual/norm/gate/up/SiLU hotspot. Corrected synthetic screens reach only
  0.656836x and 0.615217x; the candidate was removed and production is
  unchanged.
- [measured and rejected] Add a lossless 6-bit block-scale sidecar to the
  production-selected M1 NVFP4 streaming gate/up and down kernels. Gate/up
  regresses in all 30 formal actual/stress rounds. Down improves in all 30,
  but its third independent 53-layer projection reaches only 0.199822
  ms/token against the required 0.25-ms/token gate. The pre-formal deadlock
  run is excluded after the `7612bd5` warp-uniform shuffle fix. Production
  remains serial on one stream at 107.889500 ms/token and 9.268742556 token/s;
  the next dedicated Prefill optimization stage has not started.
- [measured and rejected by first-process stop-loss] Apply evict-first
  streaming policy only to the 32 canonical FP8 weight loads in the current
  public exact-M1 linear QKV/Z plus BF16 A/B tail composite. Actual and
  same-bank fixtures are bitwise/guarded and improve in all five rounds, while
  SASS size and `64r/1,280B/0local/4CTA-SM` resources are preserved. The
  actual 1.00756x result projects to only 0.172513 ms/token over 48 linear
  layers, below the 0.20-ms first-process floor, so the other two processes,
  production integration, end-to-end benchmark, and Nsys closure are skipped.
  The 107.889500-ms/token / 9.268742556-token/s anchor, serial one-stream
  schedule, and not-yet-started dedicated Prefill phase remain unchanged.
- [measured and rejected by first-process stop-loss] Apply evict-first
  streaming policy only to the eight aligned 128-bit weight loads in the
  production FP8 output-projection AoSoA4 sidecar kernel. Actual and stress
  correctness, guards, immutable inputs, resources, Graph replay, and eleven
  fail-before-enqueue cases pass, but the actual candidate regresses in all
  five rounds to 0.985003x. Its 64-layer projection is a 0.175823-ms/token
  loss, so stress timing, later processes, production integration, E2E, and
  Nsys are skipped. The production anchor and serial/Prefill state remain
  unchanged.
- [measured and rejected by first-process stop-loss] Apply streaming policy
  only to the eight packed-weight and four block-scale loads in the exact
  activation-staged NVFP4 LM head. Bounded/actual correctness, signed NaNs,
  resources, Graph replay, 17 invalid cases, guards, and 715 MB input hashes
  pass, but the candidate regresses in all five actual rounds to 0.993515x,
  adding 0.0286746 ms/token. Stress timing, later processes, production, E2E,
  and Nsys are skipped; the anchor and serial/Prefill state remain unchanged.
- [done, Decode projection palette-v2 production promotion] Select
  evict-first FP8 QKV/Z loads and lossless six-bit NVFP4 down scales for 53
  eligible layers, retaining canonical fallback for 11 layers. C1/C8/C16/C32
  pinned model oracles pass. The first four mirrored full-generation
  processes move 107.150 to 106.763 ms/token, and three independent pairs
  improve by 0.520/0.254/0.388 ms/token (median 0.388). This establishes the
  **106.763000-ms/token / 9.366540843-token/s** hot single-request anchor,
  still 6.763000 ms/token and 0.633459157 token/s short of the stage target.
  The 221,429,760-byte (0.206223-GiB) persistent sidecar and roughly two-second
  cold pack are explicit costs. Execution remains serial on one stream with
  no double/triple buffer or overlap; bulk Prefill tiles are unchanged, while
  finish-prefill M1 shares the selected route. Earlier component arithmetic
  remains microbenchmark evidence rather than end-to-end timing.
- [done, Decode short-position CUDA Graph production promotion] Promote the
  selected P19-P43/25-slot cache through an engine-lifetime transactional bank
  and the ordinary SM87 predicted-only, non-trace CLI/benchmark policy.
  Canonical and reset paths dispatch 25/0 Graph/serial, full-statistics and
  trace remain 0/0 serial paths, and P44 is an exact 25/1 miss fallback. Cold
  preparation is 75.758861 ms with a 76,607,488-byte observed CUDA free drop,
  inside the 1-second/256-MiB gates. Ordinary-entry `B1-C1-C2-B2` moves
  **106.755000 to 105.870500 ms/token**, establishing the directly achieved
  **105.870500-ms/token / 9.445501816-token/s** anchor. The target remains
  5.870500 ms/token and 0.554498184 token/s away. Execution stays on one stream
  without double/triple buffering or phase overlap; bulk Prefill is unchanged.
  Transaction-rollback and runtime-demotion failure injection remain test debt.
- [measured and rejected by actual-first P1 stop-loss] Replace only the
  production-CS Decode M1 gate/up E2M1 LUT with the exhaustive PRMT constructor
  while preserving packed-weight and block-scale `__ldcs` loads. Resources and
  actual bitwise gates pass, but baseline/candidate medians are
  0.589658/0.949274 ms/layer and all five rounds regress. The 0.621145x paired
  median projects to a 23.0181-ms/token loss, so stress, Graph/invalid matrices,
  model, profiling, and production work are skipped. Candidate hooks are
  removed; the 105.870500-ms/token / 9.445501816-token/s anchor is unchanged.
- [rejected at zero-code ceiling] Adapt down-projection scale payload width per
  layer. Exact scanning yields 53 six-bit and 11 seven-bit direct layers; the
  ideal padding-free saving is only 2.709359606%, with a 0.027813500-ms/token
  optimistic median. Even dictionary indices save only 5.172413793% before
  codebooks and project to 0.053098500 ms/token. Both miss the 15% byte and
  0.30-ms/token gates, so no implementation or GPU run is admitted and the
  105.870500-ms/token / 9.445501816-token/s anchor remains unchanged.
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
The frozen registry/matrix follow-up at `471b7a0` preserves the complete
13,558-launch production contract and exact C1/C8/C16 model gates. In a newer
maximum-one-token P19 diagnostic, TTFT is 2,031.901, 1,366.633, 831.525, and
554.386 ms for C1/C2/C8/C16. Matched Nsight profiles reduce launches from
8,249 to 2,633, but projection still occupies 91.348% of C16 kernel time; the
M2 tail alone contributes 128.477 ms across FP8 and NVFP4. This selected bounded
C32/M17-M32 work first, M2-tail tuning second, and kept buffering/multi-stream
work behind an NCU stall-evidence gate. See the
[shape/chunk/prompt matrix record](metadata/qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json).
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
The later C32 sequence first adds the composite outer tile, then promotes four
exact FP8 M32 and two exact NVFP4 M32 projections. Against the frozen FP8-M32
binary, production NVFP4 M32 reduces P33/C32 mirrored TTFT from 674.6680 to
530.8445 ms (-21.3177%), replaces 384 M16 launches with 192 M32 launches, and
reduces the target work from 393.042464 to 249.394720 ms (1.57599x). C1/C8/
C16/C32 exact-model gates remain intact. Execution still has one kernel stream
and no double/triple buffering; replay-scoped NCU evidence now decides whether
the next work is kernel-local or scheduling overlap. See the
[NVFP4 M32 production record](metadata/qwen36-27b-nvfp4-m32-production-benchmark.json).
The follow-up K256 scale-window milestone (`9690129`) keeps that N128 kernel
and reuses one coalesced 16-byte scale segment per output row across four K64
stages. Its same-cubin weighted gate is 1.15913x; replay-scoped NCU reduces
excess global sectors from 1,218,560 to 174,080 per exact shape (85.714%); and
matched P33/C32 target time falls from 249.133952 to 217.687904 ms. Mirrored
TTFT moves from 530.6365 to 499.0395 ms (-5.9545%). Two XOR product-table
swizzles and down-only N64/N96 tiles failed checkpoint-distribution gates and
were removed, so smaller-N tiling and simple XOR layouts are no longer the
priority. The next bounded gate is a checkpoint-aware reduction in decoded-
product shared traffic or a stage pipeline that preserves B reuse and dual-
accumulator ILP. One-stream serial scheduling remains in place until a trace
shows independent work worth overlapping. See the
[NVFP4 M32 scale-window record](metadata/qwen36-27b-nvfp4-m32-scale-window-benchmark.json).

The next factorized-lookup milestone (`51ca634`) completes that decoded-product
construction gate. It replaces the 8,192-byte signed-product table with a
1,024-byte packed-E2M1-pair table plus 512 bytes of E4M3 scales and exact
BF16x2 multiply.
Both exact shapes remain at 46 registers, zero local memory, and five CTAs/SM,
while static shared memory falls from 31,232 to 24,576 bytes. The same-cubin
weighted gate is 1.08417x, P33/C32 mirrored TTFT moves from 499.0785 to
486.0620 ms (-2.6081%), and matched target-kernel time falls from 217.672096 to
204.422208 ms (1.06482x). NCU also exposes the next limit: dynamic instructions
fall about 15%, but excessive shared wavefronts rise because the factorized
table layout is bank-unfriendly. Bank-aware lookup layout and vectorized decoded
stores therefore outrank broad double/triple buffering or multi-stream work.
See the [factorized lookup record](metadata/qwen36-27b-nvfp4-m32-factorized-lookup-benchmark.json).

The vector-store milestone (`9280474`) completes the first of those shared-
traffic targets. It replaces sixteen scalar decoded-tile stores with four
aligned `STS.128` stores without changing registers, shared footprint, or
five-CTA/SM residency. The same-cubin weighted gate is 1.21352x, and NCU
removes exactly 4,177,920 excessive shared wavefronts for each exact shape.
P33/C32 mirrored TTFT moves from 486.1200 to 454.1475 ms (-6.5771%), while
matched target-kernel time falls from 204.363040 to 172.409120 ms (1.18534x).
Two immediate scale-table probes did not clear promotion: direct U32 indexing
measured 0.99676x weighted, while high-bit-XOR U32 indexing reached only
1.00126x against a 1.005x gate and left checkpoint-like cells at or below
parity. Both were removed. This exhausts the cheap U32 scale-table subpath,
not every possible E2M1-pair layout. The subsequent low-footprint pipeline
gates were also rejected and removed: scale-window ping-pong reached 0.86449x
weighted, and an activation-only `cp.async` two-panel pipeline reached
0.72349x, despite both retaining 46 registers, zero local memory, and five
CTA/SM. Kernel-local explicit buffering is therefore evidence-exhausted for
this M32 route. The next priority is a matched Prefill/Decode trace that
quantifies dependency-independent work and the maximum useful overlap before
changing the one-stream policy; the logical plan split already exists, but
execution is still serial. See the
[vector-store record](metadata/qwen36-27b-nvfp4-m32-vector-store-benchmark.json).

The matched trace then justified one narrow policy change. Commit `c58b797`
overlaps the exact aligned C32 NVFP4 MLP gate and up projections on two streams
inside each layer, with ready/done events and a main-stream join before SiLU.
All other Prefill routes and all Decode work remain single-stream; auxiliary
resource allocation is best-effort and falls back to the prior serial path.
The production-dispatch micro gate is 1.08088x, the exact-commit P33/C32
confirmation is 1.02029x, and P33/P65/P129/P513 mirrored TTFT values improve by
1.57%/1.78%/1.71%/1.38% with absolute savings growing from 7.13 to 104.86 ms.
Nsight confirms all 64 target pairs actually overlap, rather than merely using
two stream IDs. See the
[gate/up dual-stream record](metadata/qwen36-27b-nvfp4-m32-gate-up-dual-stream-benchmark.json).

The bounded production-dispatch Linear-Attention A/B sidecar experiment was
then rejected before runtime integration. All four outputs were bit-exact, but
the two main-chain-first repetitions combined to only 1.03316x and at most
2.085 ms of synthetic P33 saving, while the resource-isolating
`QKV -> (Z || A/B)` variant regressed to 0.99595x with reversed passes. The
large A/B grids contend with QKV/Z strongly enough that a third stream or a
broader join is not justified. The test-only code was removed; see the
[A/B sidecar rejection record](metadata/qwen36-27b-linear-ab-sidecar-rejection.json).

Scheduling work now yields to phase-local profiling and kernel/dispatch
optimization through the existing logical Prefill/Decode plan split. Broad
Prefill/Decode overlap remains invalid for one batch-one request because token
and persistent state dependencies are causal; multi-request continuous
batching is a later serving project, not an extension of local branch overlap.

The exact M18 Prefill milestone (`09aa7f7`) is now complete. For the two aligned
NVFP4 MLP shapes, an internally masked M32 kernel replaces M16+M2 without
requiring padded caller storage. Its four microbenchmark cells improve by
1.85358x/1.80824x for gate/up and 1.60640x/1.56479x for down; the 128:64
production-call-weighted result is 1.73817x. P19/C32 mirrored TTFT improves
from 548.7825 to 439.5980 ms (-19.8958%, 1.2483735x), and matched profiling
reduces the target from 384 launches / 280.353888 ms to 192 launches /
171.292544 ms (1.636696x). C1/C8/C16/C32 exact-model gates remain intact.
M18 gate/up remains serial on the main stream, so this milestone is not
double/triple buffering and does not extend the C32-only dual-stream policy.

The immediate Decode M1 NVFP4 factorized scale-codebook candidate has now been
measured and rejected. Its exhaustive and full-path bitwise/resource gates
pass, but the selected production hotspot regresses from 0.674431 to 0.744299
ms on synthetic checkpoint-like data and from 0.616738 to 0.670177 ms on the
same-bank fixture. Every B-C-C-B pass moves in the same losing direction. The
candidate never entered production dispatch, and all test-only source was
removed. This rejects that specific fused-kernel factorization; raw down was
not timed and no actual checkpoint tensor payload was used. See the
[M1 factorized rejection record](metadata/qwen36-27b-nvfp4-m1-factorized-rejection.json).

The general runtime-tail milestone (`8b19d2a`) is now complete. Exact aligned
NVFP4 M17 and M19-M31 projections use one exact-capacity runtime-masked kernel,
while fixed M18/M32 specializations and all near-miss fallbacks remain intact.
The per-M production-call-weighted microbenchmark ranges from 1.59909x at M17
to 4.19900x at M31. A tokenizer-pinned ten-prompt B-C-C-B comparison has no
reversal in 10/10 cells and improves the equally weighted TTFT aggregate by
25.914% (1.349789x). Matched P18/P26/P64 profiles reduce the runtime-tail target
from 384/576/576 launches to 192 each and measure 1.514499x/2.510862x/3.860196x
target speedups; the fixed P64 M32 portion remains 192 launches and moves only
from 193.332160 to 193.260160 ms. C1/C8/C16/C32 exact-model gates pass. See the
[runtime-tail metadata record](metadata/qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json)
and
[pinned prompt manifest](../benchmarks/qwen36-27b-sm87-prefill-tail-prompts-v1.json).

The M17-M32 gate/up dual-stream generalization was then measured and rejected.
The committed test-only probe clears all 32 correctness/cell gates and all 16
per-M synthetic performance gates, with a 1.08146x aggregate. A clean
three-file production selector candidate also passes the Release suite. Its
twelve-prompt B-C-C-B result, however, improves only from 7,846.302 to
7,791.303 ms (0.7010%, 1.007059x), below the required 1.01x whole-model gate.
Both aggregate mirrored pairs and all eleven affected prompt rows remain
positive; the unchanged P33 control is neutral. The selector patch was
withdrawn without a production-dispatch change, and no post-failure Nsight
Systems profile was added. See the
[dual-stream rejection record](metadata/qwen36-27b-nvfp4-m17-m32-gate-up-dual-stream-rejection.json).

An initial Nsight Compute diagnostic is now complete on the retained production
M17 gate/up runtime-mask kernel. It reports 34.95% DRAM read throughput, 45.03%
SM throughput, 44.85% issue active, 19.11% tensor-pipe active, 70.51% active
warps, and 5.26 long-scoreboard stalled warps per active issue. With 48
registers per thread and 24,576 static shared bytes, the kernel retains five
active blocks per SM. This is unlocked-clock diagnostic evidence, not a causal
speedup claim and not post-hoc evidence for the rejected dual-stream candidate.

The bounded 4 KiB single-slot raw-weight `cp.async` prototype is now complete.
Gate/up clears all 28 synthetic cells and 112 mirrored rounds at 1.06036x
aggregate; matched M17 Nsight Compute measures 1.06315x, cuts long-scoreboard
stalls from 5.26 to 2.67, and retains five CTAs/SM. The optimization is
shape-sensitive: all six down screening cells regress. More importantly, the
formal twelve-prompt B-C-C-B result is only 1.007587x (affected-only 1.008529x),
below the required 1.01x whole-model gate despite no affected-prompt reversal
and neutral fixed-M18/M32 controls. The temporary selector was fully withdrawn,
and production remains serial for M17/M19-M31 and fixed M18 while preserving
the existing M32-only auxiliary-stream route. See the
[raw-weight cp.async rejection record](metadata/qwen36-27b-nvfp4-m17-m31-gate-up-raw-weight-cp-async-rejection.json).

The subsequent Decode M1 canonical full-product-table screen is also closed.
Its exact fused residual/norm/gate/up/SiLU path passes bitwise, replay, canary,
graph, and resource gates, but corrected synthetic checkpoint-like and
same-bank measurements reach only 0.656836x and 0.615217x. The candidate was
removed without a production-dispatch change; no actual checkpoint payload,
candidate NCU, or end-to-end run was used. See the
[full-product-table rejection record](metadata/qwen36-27b-nvfp4-m1-full-product-table-rejection.json).

The Prefill GDN M16 shared-resident BF16 state screen is now closed and
rejected. Its row4 and row8 test-only branches pass exact output/state,
in-place/disjoint, replay, C8+C8, Graph, canary, input-preservation, and resource
gates, but the hardened 24-bank B-C-C-B screen reaches only 0.627300x and
0.752132x against production row8 and the required 1.20x gate. Every candidate
pass is slower. The candidates and build/test wiring were removed, production
row8 remains selected, and no candidate NCU or end-to-end run was warranted.
See the
[GDN shared-resident rejection record](metadata/qwen36-27b-gdn-m16-shared-resident-bf16-state-rejection.json).
That rejection remains scoped to direct row-major shared state; the later
packed register-resident mechanism satisfies its materially-different-layout
reopen condition and is promoted below.

The independently passed Decode FP8 QKV/Z reduction-scratch ping-pong is now
productionized. Its frozen actual-checkpoint and stress cells reach 1.02407x
and 1.00697x, exact P19/P64/P513 B-C-C-B contracts pass, and all three averaged
whole-model totals improve. The promoted two-slot scratch lives only inside
one CTA/kernel; the overall runner remains dependency-serialized and is not a
general double- or triple-buffered system. See the
[QKV/Z reduction-scratch promotion record](metadata/qwen36-27b-fp8-m1-qkv-z-reduction-scratch-ping-pong-benchmark.json).

The corresponding Decode FP8 full-attention Q+K/V reduction-scratch ping-pong
is now productionized as well. Five independent actual-checkpoint processes
reach a 1.02387x median speedup and five same-bank processes reach 1.01259x.
Exact P19/P64/P513 B-C-C-B contracts pass; the worst whole-model stage
regression is 0.042171% against the 0.5% limit. The two-slot scratch remains
inside one CTA/kernel, and the single-request runner remains
dependency-serialized. See the
[Q+K/V reduction-scratch promotion record](metadata/qwen36-27b-fp8-m1-q-kv-reduction-scratch-ping-pong-benchmark.json).

The exact-shape FP8 M32 dual-resident-A screen is now productionized. Both
16-token A panels remain resident across each K64 stage, so one decoded B
fragment feeds two independent accumulator chains. Static shared memory moves
from 21,248 to 23,552 bytes and registers from 46 to 47 while five CTAs/SM are
retained. Static `BAR` instructions fall from 8 to 4; the K5120/K6144 dynamic
barrier counts fall from 324/388 to 162/194. Exhaustive 256-code by four-byte
position, signed-NaN, bitwise/replay, token-15/16, canary/input, invalid Graph,
resource, and post-promotion public-function identity gates all pass.

Five independent same-binary processes produce a P33-weighted 1.14418x to
1.14799x speedup, with a 1.14569x median. Symmetric predecessor-public and
dual-A-public runners contain the same 149 CUDA functions and identical
per-function encodings. Exact P33/P513 B-C-C-B runs improve average TTFT from
446.3555 to 433.2275 ms (-2.94%) and from 5,740.5980 to 5,524.5655 ms
(-3.76%); all eight processes reproduce token 9419 (`Hello`) and report no
persistent memory drop. These unlocked-clock ratios are diagnostic promotion
evidence, not a release-latency claim. See the
[FP8 M32 dual-resident-A promotion record](metadata/qwen36-27b-fp8-m32-dual-resident-a-benchmark.json).

The bounded U16 codebook-swizzle follow-up is now closed. Both test-only byte
bijections pass exhaustive 256-code/four-position, signed-NaN, exact replay,
guard, resource, and Graph gates, and the production mode-0 SASS remains
word-identical on all four shapes. Real shard-1 payloads tell a different
performance story from the synthetic same-bank fixture: mode 1 reaches only
1.00308x weighted and reverses in two of four cells, while mode 2 reaches
0.988108x and reverses in every actual cell. A full 5,570,560-instruction
source-level bank audit finds only 1.985% and 2.224% total-wavefront reductions
and no tail-quantile change; it is not an NCU result. The 1.55792x/1.52486x
synthetic stress results are therefore not production benefits. Neither
candidate advances to five-process, P513, or NCU work, and production remains
the original dual-resident-A layout. See the
[FP8 M32 codebook-swizzle rejection record](metadata/qwen36-27b-fp8-m32-codebook-swizzle-rejection.json).

The Decode FP8 M1 `[5120,6144]` single-body/no-tail item named as the next
screen in that snapshot was deferred while the following profile-selected
promotions completed. Its eventual result is recorded below against the
refreshed post-GDN-M16 production ranking.

The subsequent exact Q24/KV4/D256 attention-values route is now productionized.
It reduces registers from 40 to 26 and the `cuobjdump` function-block count
from 376 to 112 instruction lines while retaining six active CTAs/SM. Across
five independent same-binary processes, the hot
S65/S128/S257/S513/S544 cells span 1.30170x-1.50671x; the complete hot
S65..S513 chain spans
1.39585x-1.39948x, and the rotating 16-bank cold S513 guard spans
1.79801x-1.80948x. Exact finite/nonfinite output, replay, guards, input
preservation, invalid-call, CUDA Graph topology/production identity, resource,
and full Release/model test gates pass. Symmetric B-C-C-B runs keep P33/max1
neutral within the guard (+0.021% TTFT), reduce P513/max1 TTFT by 1.213%, and
reduce P513/max8 subsequent-token latency by 0.987%. See the
[attention-values exact promotion record](metadata/qwen36-27b-attention-values-exact-benchmark.json).

The exact Prefill BF16 M16 A/B pair is now productionized as well. One CTA
owns both N48/K5120 projections for all sixteen tokens, reusing each activation
across two independent accumulator sets while preserving the predecessor's
per-output arithmetic and BF16-RNE tree. The public-call count is unchanged,
but the per-call grid contracts from `(48,16,2)` to `(48,1,1)`. The selected
kernel uses 55 registers, 32,768 bytes static shared memory, zero local memory,
and admits four CTAs/SM.

Five independent same-binary processes span 4.48049x-4.49640x hot and
3.40114x-3.44638x across a rotating 18,350,080-byte cold working set. Exact
finite/nonfinite output, replay, guards, input and both weight buffers,
fail-before-enqueue validation, M1/M15/M16/M17/M32 Graph identity, full Release,
and actual-checkpoint chunk 1/8/16/32 gates pass. Detached B-C-C-B generation
reduces P33/P513 max1 TTFT by 2.304%/2.914%; P513/max8 Decode-after-first and
subsequent-token rows remain neutral within 0.03% because Decode M1 stays on
the generic path. The post-promotion profile reduces the 1,536-launch exact
target from 229.592608 to 66.677600 ms (3.44332x) and the full generation NVTX
range by 161.549920 ms. See the
[BF16 M16 projection-fused promotion record](metadata/qwen36-27b-bf16-m16-projection-fused-benchmark.json).

The next exact-M32 gate/up pair-fused CTA screen is now measured and rejected.
The 256-thread single-B version is exact and directionally positive, but its
1.01383x aggregate misses the frozen 1.02x gate. A 256-thread dual-B version
falls from the required four to three CTAs/SM before timing, while the
512-thread dual-B version is exact but reaches only 0.899017x and loses all
eight mirrored rounds. All candidates and their test harness were removed;
the existing table-free two-stream path remains selected. This closes
activation/codebook reuse through these pair-CTA topologies without changing
the runtime scheduler. See the
[table-free E2M1 record](metadata/qwen36-27b-nvfp4-m32-table-free-e2m1-benchmark.json).

The bounded exact-M32 down dual-A/dual-decoded-B pipeline is now measured and
rejected as well. It preserves all output, replay, boundary, guard, Graph, and
resource contracts and cuts the modeled dynamic CTA barriers by 44.1%, but its
46,592-byte shared footprint reduces residency from 5 CTA/40 warps to 3 CTA/24
warps per SM. Both scale distributions and all eight mirrored rounds regress;
the aggregate is 0.872556x against the frozen 1.03x gate. The candidate was
removed without a production change. See the
[table-free E2M1 record](metadata/qwen36-27b-nvfp4-m32-table-free-e2m1-benchmark.json).

The selected exact-C16 Prefill GDN register-state mechanism is now
productionized. It reduces recurrent state traffic from 48 MiB to 3 MiB per
C16 call while preserving per-token BF16 rounding, exact output/final state,
in-place/disjoint, replay, guard, Graph, and C1/C8/C16/C32 model contracts. The
final route uses 64 registers, 34,056 shared bytes, zero stack/local bytes, and
retains four CTAs/SM versus row8's 40 registers, 34,568 shared bytes, and four
CTAs/SM. Five fixed-frequency same-binary processes reach 1.21660x through
1.21796x and every mirrored round improves. P33 B-C-C-B prompt-prefix latency
falls from a mirrored 599.610 to 593.3715 ms; matched Nsys target time falls
from 37.047840 to 30.724128 ms across the same 96 launches. A stale test-object
exact-versus-exact measurement was found, excluded, and corrected by a forced
test-object rebuild before the formal measurements. A longer P513/C32 B-C-C-B
screen reduces prompt-prefix latency from 4,854.8745 to 4,754.8915 ms and TTFT
from 4,966.8195 to 4,866.7020 ms while retaining all 513 IDs/steps and zero
persistent-drop detections. See the
[GDN exact-M16 register-state record](metadata/qwen36-27b-gdn-m16-register-resident-bf16-state-benchmark.json).

None of these exact-kernel or shared-residency promotions introduces a runtime
double/triple buffer. Prefill and Decode are logically separated by the
completed plan seam, but batch-one execution remains causally dependency-
serialized. The post-promotion production profile is now complete. Its
P513/C32 prefix contains 42,224 kernels and a 4,701.134368 ms kernel union.
Exact-M32 gate/up contributes 1,766.932000 ms raw but only 1,367.811104 ms
marginal exposure because 399.120896 ms overlaps across the existing main and
auxiliary streams. NVFP4 down follows at 931.620448 ms marginal exposure, then
the promoted GDN M16 route at 487.589952 ms and the leading FP8 M32 shapes at
390.644896 and 382.405664 ms. Gate/up plus down represent 48.912270% of the
prefix kernel union; raw overlapped time is no longer accepted as the Prefill
selection metric.

The corresponding P19/C32/max26 Decode profile contains 25 Decode ranges and
10,925 kernels, all on one stream. Its 2,765.851648 ms raw time equals its
union, while only 20.448064 ms (0.733879%) of the 2,786.299712 ms span is idle.
NVFP4 gate/up, FP8 QKV/Z, and NVFP4 down account for 35.979365%, 20.821564%,
and 18.903314% of Decode kernel time, or 75.704244% together. This evidence
keeps phase-local kernels ahead of general buffering: there is no useful
batch-one Decode overlap hole large enough to preempt those rows, and Prefill
already uses the one proven narrow dual-stream path. See the
[post-GDN-M16 phase-profile record](metadata/qwen36-27b-post-gdn-m16-phase-profile.json).

The deferred Decode FP8 M1 `[5120,6144]` exact
single-body/no-tail-barrier screen is now closed. Exhaustive/replay/NaN/guard/
Graph/resource gates pass, and SASS proves the intended 960-instruction,
two-barrier body against the 1,152-instruction, three-barrier production
predecessor without changing 64-register/four-CTA residency. The authoritative
fixed-clock layer-0 payload screen nevertheless regresses in all six
`B-C-C-B` rounds: paired median 0.994914x, minimum 0.994295x, versus required
1.01x/1.00x gates. The actual-first stop-loss skipped synthetic, five-process,
P19, Nsys, and NCU work; the candidate was removed and the forced clean
rollback passes. See the
[attention O-projection no-tail rejection record](metadata/qwen36-27b-fp8-m1-attention-o-proj-no-tail-rejection.json).

The first bounded Prefill follow-up is also closed. An exact-M32 NVFP4
`[5120,17408]` down/residual epilogue fusion preserves five-CTA residency and
passes its finite/replay/guard/Graph/resource checks, but its six fixed-clock
actual-payload rounds reach only a 1.00050x paired median against 1.005x
technical and 1.02x promotion gates. Independent audit also found that the
candidate/test add raw-down-left plus residual-right while the real Prefill
chain is residual-left plus raw-down-right. The finite performance rejection
remains valid, but the 4/4 test-defined NaN bit match is not a runtime bitwise
semantics claim. The candidate was removed without stress, NCU, Nsys, or
end-to-end work. See the
[M32 down/residual rejection record](metadata/qwen36-27b-nvfp4-m32-down-residual-epilogue-fusion-rejection.json).

The subsequent exact-M32 FP8 QKV/Z dual-N128 fusion is closed as well. Its
single grid64/block512 kernel replaces the public grid80 plus grid48 pair and
shares one decoded codebook and activation tile across two eight-warp N128
groups. Static mapping/SASS/resource audit, exhaustive E4M3FN, full-K,
replay, guard, input-preservation, Graph, and real-payload correctness all
pass. Fixed-clock timing reverses decisively, however: all six `B-C-C-B`
rounds regress, with a 0.908776x paired median and a
0.908670x-0.909103x range. The 512-thread/two-CTA topology exposes 32 resident
warps per SM versus 40 for the two 256-thread/five-CTA production kernels;
the stable result is consistent with that lost scheduling headroom outweighing
activation-traffic reuse. Actual-first stop-loss skips stress, NCU/Nsys,
end-to-end, and five-process work and restores the tracked source/test
files.
See the
[QKV/Z dual-N128 rejection record](metadata/qwen36-27b-fp8-m32-qkv-z-dual-n128-fusion-rejection.json).

The independent exact-M32 FP8 `[5120,6144]` K6144 half-tile single-slot
raw-weight `cp.async` screen is now closed. Its test-only grid40/block256
candidate targeted 382.219296 ms of post-C1 marginal exposure (8.3653% of
prefix kernel union) and passes static resource/SASS, exhaustive-code, full-K,
replay, guard, input-preservation, Graph, and actual-weight/scalar plus
deterministic-activation correctness gates.
All six fixed-clock `B-C-C-B` rounds regress, however: paired median 0.947746x
and range 0.947361x-0.948105x. The reported five CTAs per SM is occupancy
capacity; the 40-CTA grid averages only 2.5 CTAs per SM across this device's 16
SMs. The actual-first stop-loss skips stress, NCU/Nsys, end-to-end, and
five-process work, restores the tracked source/test files, and makes no
profiler-attribution claim. Both exact FP8 half-tile shapes are therefore
closed under their frozen gates. See the
[K6144 half-tile rejection record](metadata/qwen36-27b-fp8-m32-k6144-half-tile-raw-weight-cp-async-rejection.json).

The higher-risk exact-M32 NVFP4 asymmetric table-free-E4M3 concurrent screen
is now closed. Its packed E4M3x2 constructor is exact over all 65,536 ordered
byte pairs, and the corrected test-only kernel passes resource/SASS, full
557,056-element B/G/U and replay bit checks, guards, input preservation,
Graph, and pinned actual-payload gates. The final candidate retains 48
registers and five-CTA capacity, uses 29,184 shared bytes, and reduces the
static body from 736 to 688 instructions, but both one-sided concurrent routes
regress in every fixed-clock round. Gate/main reaches a 0.957097x paired
median (0.955298x-0.958556x); up/aux reaches 0.956026x
(0.953316x-0.957179x), with 0/6 rounds at or above 1.0 for either direction.
The actual-first stop-loss skips stress, profiling, end-to-end, and
replication, removes the candidate, and leaves production unchanged. See the
[asymmetric E4M3 rejection record](metadata/qwen36-27b-nvfp4-m32-asymmetric-table-free-e4m3-bgu-rejection.json).

The independent exact-C32 GDN register-state lifetime-extension screen is now
closed as well. Its test-only compile-time M32 body keeps the recurrent state
in the same packed BF16 register words across all 32 tokens, replacing two
ordered exact-M16 launches without removing any per-token BF16 rounding. Full
32-token output/final-state/replay bit checks, input preservation, Graph, and
resource gates pass; M32 retains the M16 envelope of 64 registers, 34,056 B
shared, zero local memory, and four active CTAs/SM. All six fixed-clock
`B-C-C-B` rounds improve, but only to a 1.02474x paired median and
1.02468x-1.02487x range, below the frozen 1.03x early gate. The threshold was
not lowered after measurement. Stop-loss removes the candidate and skips
runner integration, model-oracle, NCU/Nsys, end-to-end, and replication work.
See the
[GDN exact-C32 rejection record](metadata/qwen36-27b-gdn-m32-register-resident-bf16-state-rejection.json).

That required current-HEAD marginal refresh is now complete at `9bddbda`.
Against the earlier `b09614c` snapshot, the production raw-weight `cp.async`
route reduces the P513/C32 prefix union from 4,701.134368 to 4,580.975776 ms
and the gate/up marginal exposure from 1,367.811104 to 1,273.522080 ms. The
current Prefill gate/up and down rows contribute 27.800236% and 20.438737% of
prefix union; the top ten rows cover 93.891520%. The P19/C32/max26 Decode
refresh contains the same 10,925 kernels on one stream, 2,757.625632 ms raw
time equal to union, and only 16.223744 ms (0.584882%) idle in its kernel span.
Its NVFP4 gate/up, FP8 QKV/Z, and NVFP4 down rows account for 75.591051% of
Decode kernel time. See the
[current-HEAD phase-profile record](metadata/qwen36-27b-current-head-phase-profile.json).

The first bounded Decode follow-up screened the exact-N17,408 gate/up tail.
It retained the production mapping for rows 0-16,383 and replaced the old
half-grid ninth row-quad round with one row pair per warp across all 64 CTAs.
Resources and every actual/stress/nonfinite bitwise gate pass, but all five
fixed-clock rounds regress: 0.951273x actual and 0.955264x stress paired
medians. The candidate and test hooks were removed before a clean rebuild and
default device-test pass. See the
[balanced-tail rejection record](metadata/qwen36-27b-nvfp4-m1-gate-up-balanced-tail-rejection.json).

That packed-weight screen is now measured. Its exact-K256 single-slot
`cp.async` candidate uses one 4-KiB CTA slot and current packed words in
registers as a logical double buffer. It preserves 64 registers, zero local
memory, and four active CTAs/SM, and all actual/stress output gates are
bit-exact. However, all five rounds regress: paired medians are 0.932869x on
the actual layer-0 payload and 0.928905x on same-bank stress. Canonical weight
traffic was already coalesced, while the candidate adds four shared loads and
two cross-lane synchronizations per K256 tile. See the
[K256 packed-weight pipeline rejection record](metadata/qwen36-27b-nvfp4-m1-gate-up-k256-packed-weight-pipeline-rejection.json).

The smaller code-generation screen is also closed. Replacing both source
`__syncwarp` calls with volatile inline PTX
`bar.warp.sync 0xffffffff` produces a candidate Function byte-identical to the
measured 0.93x version: ptxas still emits the same four `CALL.REL` sites and
the same `WARPSYNC R30; RET` helper. The static-equivalence stop therefore
skips redundant correctness and timing. See the
[inline warp-barrier static rejection](metadata/qwen36-27b-nvfp4-m1-gate-up-k256-inline-warp-barrier-static-rejection.json).

A K512 canonical variant is not next: it can only halve this overhead, needs
four more live packed words, and cannot credibly recover a measured seven-
percent regression under the 64-register ceiling. The materially different
single-layer Decode-only AoSoA4 sidecar is now closed too. It directly loads
the four row words as one `uint4`, deletes shared staging and both warp
barriers, and returns to the production 64-register/11,328-byte resource
envelope. Actual and stress results remain bit-exact, and the stress median is
1.00353x, but every actual-payload round regresses; the paired median is
0.994987x against the frozen 1.03x gate. See the
[AoSoA4 sidecar rejection](metadata/qwen36-27b-nvfp4-m1-gate-up-aosoa4-sidecar-rejection.json).
The full 5.3125-GiB 64-layer sidecar, scale interleave, loader changes, and
cache format are therefore not built.

Gate/up load-shape work remains deprioritized, but a separate exact FP8 M1
`[5120,6144]` output-projection AoSoA4 plus byte-preswizzled sidecar now clears
the test-only formal gate: 1.05155x actual paired median with all five rounds
non-regressing, 1.01465x stress, and complete bitwise/guard/resource gates.
The next bounded step is persistent load/cache-time sidecar integration, not
production dispatch: 30 MiB per layer and 1.875 GiB for 64 layers must clear
loader, startup, peak-memory, model-oracle, and fixed-clock end-to-end gates.
The isolated 0.580608-ms/token projection moves the 110.951-ms planning value
only arithmetically to 110.370 ms, about 9.060 token/s, and is not an achieved
runtime result. If those integration gates fail, re-rank FP8 linear-attention
QKV/Z and fused NVFP4 down from the current-production profile; do not restore
closed gate/up mechanisms. See the
[test-only selection record](metadata/qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-selection.json).

That bounded integration has now cleared production. All 64 exact output
projections attach a persistent GPU-packed sidecar while retaining canonical
weights for Prefill, M2+, and fallback. The final Release formal gate reaches
1.05212x on the actual checkpoint and 1.00853x on same-bank stress; the full
GPU pack matches a 30-MiB CPU oracle, all seven pack-invalid and eleven direct-
GEMV-invalid cases pass, and the final Release binary passes the exact model
oracle. The fixed-frequency P19/C32/max26 `B1-C1-C2-B2` gate moves the mirrored
hot Decode median from 110.1060 to 109.7585 ms/token, or 9.1109 token/s, with
both pairs improving. This is an achieved runtime result, unlike the earlier
arithmetic projection.

The tradeoff is explicit: 2,013,265,920 persistent bytes (1.875 GiB) and
437.460/497.171 ms of cold construction in the timed candidate processes,
about 467 ms on average. The hot number excludes that startup work. The
single-request milestone remains 9.7585 ms/token away from 100 ms and needs a
further 8.89% latency reduction. The immediate bounded item is the NVFP4 down
CTA-prune screen described next. Prefill resumes after the 100-ms/10-token/s
Decode gate, and closed gate/up load-shape branches remain closed. See the
[production sidecar benchmark](metadata/qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-production-benchmark.json).

That post-grid-sync CTA-prune screen is now closed. It is bit-exact, preserves
the `64x256` cooperative shape and production resources, and non-regresses in
all ten paired rounds, but checkpoint-like/stress medians reach only
1.00135x/1.00107x against 1.002x. The roughly 0.02-ms/token arithmetic ceiling
triggers stop-loss before actual checkpoint, NCU, model-oracle, or end-to-end
work. The candidate is removed. See the
[CTA-prune rejection](metadata/qwen36-27b-nvfp4-m1-down-norm-cta-prune-rejection.json).

The required post-sidecar production Decode profile is now complete at
`aa7312b`, using the final `77931b8` Release binary. Its 25 ranges contain
10,925 unique kernels on one stream; 2,745.814816 ms raw equals union, and only
0.623453% of the associated span is idle. Gate/up, QKV/Z, and down remain the
first three rows at 36.123932%, 20.848359%, and 18.906612%, jointly 75.878904%.
The promoted output sidecar is fourth at 10.564009%, averaging 0.181293 ms per
launch and 11.602725 ms per step. Against the older `9bddbda` profiler capture,
that row is directionally 0.393938 ms/step lower, but cross-process Nsys drift
means neither the full-trace delta nor other row movement is assigned a cause.
The unprofiled 109.7585 ms/token and 9.1109 token/s `B1-C1-C2-B2` result remains
the release anchor. See the
[post-sidecar Decode phase profile](metadata/qwen36-27b-post-fp8-output-sidecar-decode-phase-profile.json).

The first bounded follow-up from that ranking is now closed. The test-only FP8
M1 `[10240,5120]` QKV plus `[6144,5120]` Z AoSoA4/preswizzled sidecar is
bit-exact on the hash-pinned layer-0 and stress fixtures, preserves the
one-node `1536x256` Graph and 64-register/four-CTA resource envelope, and
reduces the normalized candidate function from 2,416 to 2,224 words. Actual
payload improves in all five rounds but reaches only 1.00562x versus the 1.03x
gate; stress regresses in all five rounds to 0.984394x. Its ideal 48-layer
arithmetic saving is only 0.124416 ms/token against 3.75 GiB of extra residency.
The candidate is removed before full-model allocation, integration, profiling,
or end-to-end work. See the
[QKV/Z sidecar rejection](metadata/qwen36-27b-fp8-m1-qkv-z-aosoa4-preswizzled-sidecar-rejection.json).

That materially different leading-row mechanism is now selected at the
test-only microbenchmark stage. The exact M1 residual/norm/gate/up/SiLU
candidate changes production's `64x256` physical grouping to `32x512` while
preserving all 512 projection warps, the 2,048-row stride, exact arithmetic,
32 resident warps/SM, and zero extra model memory. It halves the CTAs that
repeat residual/RMSNorm setup. Across three independent same-binary processes,
actual-checkpoint and same-bank-stress cross-process paired medians are
1.02410x and 1.02295x; all 30 rounds improve. Exact finite/replay and signed
Inf/NaN outputs pass bitwise, and resource including null-query rejection,
one-node Graph, and nine zero-node invalid-call gates pass. Production SASS
remains identical in the test build, and all three processes exit zero.
The complete test-only stage remains in the
[CTA-coarsening selection](metadata/qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-selection.json).

That production promotion is now complete at `15cbb28`. The public exact M1
route is `32x512`; the retained `64x256` predecessor remains test-only. The
final actual/stress paired medians are 1.02388x/1.02302x, every round improves,
all finite/replay and signed Inf/NaN outputs remain bit-exact, and the final
resource, Graph, invalid-call, SASS, Release, CTest, and pinned full-model
oracle gates pass. No public API, model format, loader, sidecar, or persistent
memory changes are required.

The fixed-frequency P19/C32/max26 `B1-C1-C2-B2` gate moves mirrored hot Decode
from 109.817 to **109.056 ms/token**, or **9.1696 token/s**, with both pairs
improving. This is an achieved end-to-end result. The stage target is now
9.056 ms/token and 0.8304 token/s away, requiring another 8.304% latency
reduction. Fresh Nsys closure retains 25 ranges, 10,925 distinct kernel rows,
one stream, and zero overlap. The promoted gate/up row falls directionally from
39.675852 to 38.772791 ms/step while remaining rank one; QKV/Z and down remain
ranks two and three. See the
[production benchmark](metadata/qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-production-benchmark.json)
and
[post-promotion Decode profile](metadata/qwen36-27b-post-gate-up-cta-coarsen-decode-phase-profile.json).

The next rank-three down/residual/centered-RMSNorm mechanism is now selected at
the test-only stage. Its `32x512` grouping preserves the production `64x256`
route's 512 projection warps and 32 resident warps/SM while halving complete
activation staging and redundant RMS reductions. Three fixed-frequency
same-binary processes reach actual-checkpoint medians of
1.01181x/1.01842x/1.01176x and stress medians of
1.01218x/1.01876x/1.01453x; all 30 rounds improve, and bitwise/replay,
guards, inputs, split nonfinite, resource, Graph, invalid-call, and production
SASS gates pass. The 32-CTA cooperative grid exactly matches measured resident
capacity. Production remains unchanged, so 109.056 ms/token and 9.1696 token/s
remain authoritative; the 0.246464-ms/token 64-layer value is arithmetic only.
The immediate next step is bounded production integration, full-model and
end-to-end validation, then a fresh Decode trace. See the
[down CTA-coarsening selection](metadata/qwen36-27b-nvfp4-m1-down-cta-coarsen-selection.json).

That promotion is now complete at `0372d41` (tree `aa341b9`). Production uses
the selected `32x512` down/residual/centered-RMSNorm cooperative route; the
distinct `64x256` function is retained only as a test predecessor. Three final
same-binary processes report actual-checkpoint medians of
1.01511x/1.01184x/1.01419x and stress medians of
1.01975x/1.01416x/1.01544x, with all 30 rounds improving. Default, four-test
CTest, pinned model-oracle, finite/replay, split nonfinite, guards, inputs,
resource, Graph, invalid-call, and SASS gates pass. The 0.275392-ms/token
64-layer micro projection remains arithmetic only.

The valid fixed-clock end-to-end sequence is `B1-C1-C2-B2` at
109.431/109.066/109.165/109.316 ms/token. An earlier baseline inference
completed but had truncated stdout; it is excluded, and the sequence was
restarted from B1. Both complete mirrored pairs improve; their aggregate moves
from 109.3735 to 109.1155 ms/token, a same-run -0.258-ms delta, while all four
functional canonical files have SHA-256 `f66b837e...`. Because the fresh
candidate absolute median is still 0.0595 ms slower than the older formal
109.056-ms result, the conservative achieved anchor remains **109.056
ms/token and 9.1696 token/s**. The 108.798-ms/9.191345-token/s value obtained
by applying the same-run delta to the old anchor is planning normalization,
not a measurement.

Fresh production profiling closes 25 Decode ranges over 10,925 distinct rows
on one stream, with raw equal to union at 2,725.072960 ms, zero overlap, and
0.572941% span idle. Generation closes exactly over 12,997 leaf rows. Down
`32x512` is present 1,600 times, exactly 64 per step, at 20.813704 ms/step and
19.094630%; the predecessor and every down `64x256` launch are absent. The row
is directionally 0.079035 ms/step lower than the previous trace, while total
raw time is effectively flat (+0.002021 ms/step) under separate-run drift.
See the
[production benchmark](metadata/qwen36-27b-nvfp4-m1-down-cta-coarsen-production-benchmark.json)
and
[post-promotion profile](metadata/qwen36-27b-post-down-cta-coarsen-decode-phase-profile.json).

The bounded terminal gate/up CTA-coarsening screen is now closed. Its test-only
`16x1024` candidate pairs production logical blocks `b` and `b+16`, preserving
all 512 projection warps while balancing the final half-stride and halving
repeated residual/RMSNorm setup again. Resource, exact 16-CTA capacity,
distinct one-node Graph, nine-invalid, finite/replay, guard, input, combined
signed Inf/NaN, and SASS gates pass. Nevertheless, all ten fixed-clock rounds
regress: the actual-checkpoint and stress paired medians are 0.994805x and
0.996853x. First-process stop-loss removes the candidate and skips replication,
production, model-oracle, end-to-end, Nsys, and NCU work. The result closes
only this `b`/`b+16` balanced-tail mapping; a naive contiguous `16x1024`
mapping was not tested. See the
[16x1024 rejection record](metadata/qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-1024-rejection.json).

The bounded production-ping-pong QKV grid-cap screen is also closed. Its
test-only policy runs the exact same kernel and SASS at QKV1024/Z768 instead
of production QKV1536/Z768; the 256-thread block, two CTA-local scratch slots,
Z cap, ABI, and production route do not change. Source identity, resource,
invalid-call, exhaustive finite/code, replay/race-signature, isolated NaN,
and frozen actual/stress correctness gates pass. The authoritative
same-binary process reaches 1.0047x on the actual checkpoint against a 1.005x
gate, while stress reaches 0.998787x and all five stress rounds regress.
First-process stop-loss restores the test policy and skips replication,
production, model-oracle, end-to-end, Nsys, and NCU work. Production and the
formal **109.056 ms/token / 9.169600939 token/s** anchor remain unchanged. See
the
[QKV grid-cap 1024 rejection record](metadata/qwen36-27b-fp8-m1-qkv-z-ping-pong-grid-cap-1024-rejection.json).

The materially different gate/up mechanism was first selected at the test-only
stage. It keeps production's `32x512` topology but retains the independently
BF16-rounded gate/up pair in two CTA-local `BF16[576]` arrays and publishes
only final `SiLU(gate)*up`; the runner-dead up buffer remains untouched. Three
clean frozen-binary processes reach actual/stress cross-process paired medians
of **1.01034x / 1.00932x**, with all 30 rounds improving. Published-output
bitwise/replay, combined signed Inf/NaN, guards, inputs, resource
(`64r/13,632B/0local/2CTA`), one-node Graph capture, nine-invalid, and
production-SASS-identity gates pass. The 64-layer **0.396352-ms/token** value
was arithmetic only at selection time, so that record did not change the then
formal **109.056 ms/token / 9.169600939 token/s** anchor. See the
[selection record](metadata/qwen36-27b-nvfp4-m1-gate-up-dead-up-shared-pair-selection.json).

That production sequence is now complete at `2dbd832`, with the complete
54-case invalid matrix hardened at `798582c`. The generic public double-output
API continues to publish independently rounded up values; only the explicit
runner-dead exact route elides that publication, and every fallback preserves
the old behavior. Real-buffer CUDA Graph replay, Release/default/full CTest,
and pinned C1/C8/C16/C32 model oracles pass. Fixed-clock P19/C32/max26
`B1-C1-C2-B2` moves the mirrored hot Decode median from 109.0535 to
**108.6695 ms/token**, with both pairs improving, establishing
**9.202214053 token/s**. The new formal stage gap is 8.6695 ms/token.

The production trace also passes every closure gate: 25 Decode ranges contain
10,925 distinct rows, exactly 437 per step; the runner-only symbol appears
1,600 times, exactly 64 per step; the retired full-output boundary symbol is
absent; and all 12,997 generation leaves close with no missing, extra, or
duplicate rows. Decode remains one stream with raw equal to union and zero
overlap. See the
[production benchmark](metadata/qwen36-27b-nvfp4-m1-gate-up-dead-up-production-benchmark.json)
and
[post-promotion profile](metadata/qwen36-27b-post-gate-up-dead-up-decode-phase-profile.json).

This remains phase-local work, not a system double/triple buffer or
Prefill/Decode executor split. The measured single request is causally serial
on one stream, so general batch-one buffering stays behind the next
phase-local mechanisms.
Closed gate/up row-pair balanced-tail/shared-pipeline/AoSoA4/`16x1024`
variants, the down CTA-prune, the QKV/Z sidecar, and production-ping-pong
QKV1024/Z768 grid-cap policy remain closed. The next bounded mechanism must be
selected from the refreshed gate/up, QKV/Z, and down ranking. The 100-ms/token
and 10-token/s gate still precedes the larger Prefill program.

The later GQA warp-position promotion established the previous formal
**107.314000-ms/token / 9.318448665-token/s** anchor. Decode projection palette
v2 now supersedes it with a full-generation result of **106.763000 ms/token
and 9.366540843 token/s**: the first four mirrored processes move 107.150 to
106.763 ms/token, while three independent pair savings are
0.520/0.254/0.388 ms/token (median 0.388). This is end-to-end evidence, unlike
the earlier QKV-plus-down arithmetic selection projections. The C1/C8/C16/C32
oracles pass; 53 down layers use the new 221,429,760-byte (0.206223-GiB)
sidecar, 11 fall back, and cold packing costs about two seconds.

The target remains unmet by **6.763000 ms/token and 0.633459157 token/s**.
One-stream serial execution, the absence of system double/triple buffering or
overlap, and bulk Prefill tile dispatch are unchanged; finish-prefill M1 does
share the selected route. Continue bounded Decode optimization toward 100
ms/token before starting the larger dedicated Prefill stage. See the
[palette-v2 production benchmark](metadata/qwen36-27b-decode-projection-palette-v2-production-benchmark.json).

The P1 gate/up scale-only Row-Quad AoSoA4 candidate is now a stop-loss
rejection. Its isolated 64-layer projection is only **0.0523529 ms/token**
against the frozen **0.50-ms/token** gate, so it does not advance to a full-
model run and does not change production or the formal anchor. The active
anchor therefore remains **106.763000 ms/token / 9.366540843 token/s**, still
**6.763000 ms/token** above the 100-ms target.

The P1 test-only FP8 attention O-projection CTA512 candidate is now a stop-loss
rejection. Correctness passes, but the public path measures **0.176767
ms/layer** versus **0.192815 ms/layer** for the candidate, for a paired speedup
of only **0.915945** and a projected 64-layer regression of **1.03662
ms/token**. The first process therefore stops at the micro gate without a
full-model run; production remains unchanged and the formal anchor stays at
**106.763000 ms/token / 9.366540843 token/s**.

The exact-BF16 SiLU FP32 lookup is also closed at P1. Its 256-KiB table and
production-shaped `32x512` kernel pass exhaustive 65,536-entry FP32-bit,
output, guard, workspace, and unchanged-resource gates, but actual-checkpoint
timing reaches only **0.999793x** and projects a **0.00788879-ms/token
regression** over 64 layers versus the required **1.01x / +0.30 ms/token**.
Four of five actual rounds and two of five stress rounds regress. SASS confirms
that one LDG replaces the SiLU EX2/RCP work, but the bandwidth-dominated
projection sees no measurable gain. P2/P3, model E2E, and Nsys are skipped;
production and the **106.763000 ms/token / 9.366540843 token/s** anchor remain
unchanged. Do not reopen the same global exact-BF16 lookup without a materially
different mechanism and a credible at-least-0.30-ms/token ceiling.

The exact-shape LM-head RP2 schedule-major AoSoA2 candidate is now rejected by
the same actual-first P1 discipline. Its `80x256` persistent-owner kernel and
same-size U64/U16 row-pair sidecars pass the <=51-register, 11,328-byte shared,
zero-local, five-CTA/SM resource gate; every sidecar byte maps exactly; all
248,320 direct and Graph-replay BF16 outputs match the public activation-staged
baseline; guards, input/sidecar immutability, invalid zero-node cases, and both
`0x7f`/`0xff` signed-NaN fixtures pass. Performance does not: production and
candidate pass medians are 4.69201/4.75805 ms, the paired median is
**0.973487x** with a **-0.126222-ms/token** delta, and only 2/5 rounds improve
against the frozen 1.075x, +0.30-ms, every-round gate. Stop-loss skips stress,
P2/P3, model, profiling, and integration. Production remains unchanged and the
formal anchor stays **106.763000 ms/token / 9.366540843 token/s**. Do not repeat
this exact RP2 schedule/layout without a materially different mechanism and a
credible >=0.30-ms/token ceiling. See the
[RP2 rejection record](metadata/qwen36-27b-decode-nvfp4-lm-head-rp2-schedule-aosoa2-rejection.json).

The P19-P43 Decode Graph cache is now production-promoted through an
engine-lifetime transactional bank and the ordinary CLI/benchmark entry point.
The mirrored full-generation gate establishes a new directly achieved anchor
of **105.870500 ms/token / 9.445501816 token/s**. Its two independent pairs
save 0.898/0.871 ms/token, every candidate sample dispatches 25/0 Graph/serial,
all golden results are exact, and no process reports a persistent memory drop.
Production correctness closes P19-P43, full-statistics, trace, reset, and the
P44 serial miss; cold preparation and free-memory drop pass their budgets.
See the [production Decode Graph record](metadata/qwen36-27b-decode-short-position-cuda-graph-cache-production-benchmark.json).

The production-CS M1 gate/up table-free E2M1 probe is closed at actual-first
P1. Although exhaustive PRMT, exact output, guard/workspace, and resource gates
pass, its paired median is **0.621145x** and all five rounds regress, projecting
a **23.0181-ms/token loss** over 64 layers. Candidate hooks are removed and the
formal anchor remains **105.870500 ms/token / 9.445501816 token/s**. Do not
reopen the exact inline PRMT substitution without a materially different
dequantization mechanism and a credible at-least-0.5-ms/token ceiling. See the
[rejection record](metadata/qwen36-27b-decode-nvfp4-m1-gate-up-table-free-e2m1-rejection.json).

The 100-ms/token / 10-token/s gate is still open by **5.870500 ms/token and
0.554498184 token/s**. Re-rank structural Decode candidates from the fresh
production Graph profile and screen only mechanisms with a credible **at least
0.3--0.5 ms/token** contribution before advancing to full-model validation.
Keep the larger dedicated Prefill program behind the Decode stage gate.

Root-enabled current-production NCU is now available and closes the earlier
performance-counter gap. Matched one-launch reports put gate/up, down scale6,
and LM head near mandatory weight-plus-scale traffic, at 81.12%, 74.90%, and
79.42% of peak DRAM-read throughput. GDN is materially different: only 21.57%
DRAM-read and 37.97% issue-active, with a concrete shared-memory state round
trip. The next actual-first P1 is therefore an exact packed-register transient
GDN state update, gated at least 1.2448x / 0.30 ms/token, zero local memory,
and at least three CTA/SM. Projection physical-layout work follows only as a
bounded hypothesis; generic L2 persistence and same-request buffering remain
lower priority. MTP is explicitly outside this target path. See the
[non-MTP traffic/state audit](metadata/qwen36-27b-decode-non-mtp-global-traffic-state-audit.json).

The selected GDN single-reload P1 is now closed by actual-first stop-loss.
Packed-BF16 recompute and retained-FP32 variants both pass bitwise, Graph, and
resource gates and improve all five `B-C-C-B` rounds, but reach only
**1.05144x / 0.074424 ms/token** and **1.05294x / 0.076402 ms/token**. Both miss
the 1.2448x / 0.30-ms/token gate, so candidate code is removed and production
is unchanged. Do not repeat a one-reload-only state cache; a new GDN topology
must remove at least two scratch transfers and independently establish a
credible 0.30-ms/token ceiling. The immediate path returns to zero-code
projection payload admissions and measured global-flow changes. See the
[GDN rejection record](metadata/qwen36-27b-decode-gdn-m1-transient-register-state-rejection.json).

The first global-flow follow-up also closes down-only physical repacking.
Fresh transaction tables show **99.520% load-sector utilization** and
48,797,696 L2-miss bytes versus 48,742,400 mandatory bytes. Perfecting every
remaining load transaction projects to only **0.093974 ms/token**; an even more
optimistic bound that removes all store-sector waste reaches **0.157985
ms/token**, still below the 0.30-ms gate. Gate/up scale payload does pass a
separate capacity admission: lossless P15E-T128 uses **55.945506%** of canonical
bytes across all 128 tensors, but it remains test-only until decoder cost is
measured. See the [down-flow rejection](metadata/qwen36-27b-decode-down-global-flow-layout-ceiling-rejection.json)
and [gate/up payload admission](metadata/qwen36-27b-decode-gate-up-p15e-t128-scale-payload-admission.json).

The admitted gate/up payload does not survive its decoder gate. P15E-T128 is
bit exact and reduces LTS traffic by **40.27%**, but its sparse directory/raw
access raises L1 request traffic by **32.01%** and all five actual layer-0
rounds regress to a **0.620662x** paired median, projecting a **4.785536
ms/token loss**. The candidate is removed. The deeper GDN structural screen is
also closed: transposed-state cold results are 0.8430x, 0.9027x, and 0.5792x,
while the isolated native BF16 encoder is 0.996115x. Resident-only GDN wins are
cache artifacts and may not justify promotion. Fixed-cut LM pruning likewise
misses its traffic gate; q20 single-kernel progressive pruning remains a
conditional P2 without an implementation or performance claim. The immediate
mainline returns to decode-wide projection bytes and requires a new mechanism
with a measured **at least 0.30 ms/token** contribution. See the
[gate decoder rejection](metadata/qwen36-27b-decode-gate-up-p15e-t128-decoder-rejection.json),
[GDN structural rejection](metadata/qwen36-27b-decode-gdn-transposed-state-native-encoder-rejection.json),
and [LM progressive audit](metadata/qwen36-27b-lm-head-exact-progressive-mips-admission.json).

The next zero-code admission is QKV/Z P127E-W128 lossless FP8 payload. All 96
actual tensors fit in **90.437571%** of their raw bytes in aggregate, removing
**385,034,240 bytes/token** for a **2.197907-ms/token** zero-decode-overhead
ceiling. This is not yet a performance result: a 112-byte warp code tile can
still request four 32-byte L1 sectors, and escape ranking adds instructions and
metadata traffic. Advance only through a production-unreachable layer-0
actual/stress screen with bit-exact outputs, zero local memory, four CTA/SM,
every-round non-regression, and at least **6.25 us/layer / 0.30 ms/token** net
saving. Do not allocate the 3.391-GiB full sidecar before that gate. See the
[P127E admission](metadata/qwen36-27b-decode-fp8-qkv-z-p127e-w128-payload-admission.json).

That implementation gate is now closed. The bit-exact row-quad P127X hybrid
passes its resource screen (`26r / 256 B shared / 0 local / 6 CTA-SM`) but
regresses every cold actual round to **0.369522x**. Its L1 request traffic rises
**117.84%** and LTS traffic rises **4.76%**, so the logical 8.21% compression
does not reduce physical traffic. More decisively, a non-exact fixed-direct
lower bound that removes all directory and escape work still reaches only
**0.568148x**. Close P127E, P127X, fixed-direct P127, P63, and the narrower O
backup on SM87; do not build a full GEMV or sidecar from this admission. See
the [decoder rejection](metadata/qwen36-27b-decode-fp8-qkv-z-p127x-w128-decoder-rejection.json).

The next actual-first cell returns to Gate/Up scales with a mechanism distinct
from scale6 and P15E. Delta4-Row32 stores each four-row K512 tile in a fixed
64-byte `[lane-pair][row4]` code block; 81.380086% of rows use exact
`base + nibble`, while the rest read one retained canonical sector. Hoisting
all ten metadata words once per row quad gives a conservative **26.380086%**
L1-request reduction and **1.136086 ms/token** traffic ceiling. This is only a
test admission. Layer 0 and worst layer 50 must both be bit exact, preserve
`32x512`, zero local memory and at least two CTA/SM, and save **at least 4.6875
us/layer** in every `B-C-C-B` round before a full Gate/Up kernel is considered.
See the [Delta4 admission](metadata/qwen36-27b-decode-gate-up-delta4-row-fallback-admission.json).

That decoder gate is now closed. The first timings were discarded because the
canonical standalone kernel spilled 328 local bytes per thread. Under a
symmetric zero-local schedule, Delta4 reduces L1 sectors by **27.36%** and LTS
sectors by **29.07%**, but increases dynamic instructions by **92.76%**. Every
actual layer-0 round regresses; the paired median is **0.598319x** and projects
to a **6.623503-ms/token loss**. The candidate is removed without a full
Gate/Up kernel or production change. Further Decode work returns to measured
cross-kernel flow and state/L2 ceilings; MTP remains outside the current target
path. See the [Delta4 decoder rejection](metadata/qwen36-27b-decode-gate-up-delta4-row-fallback-decoder-rejection.json).

The bounded causal-convolution L2-persistence backup is also closed at the
admission stage. Its 2.8125-MiB history barely exceeds Orin's 2.75-MiB
persisting budget, but optimistically retaining 44/45 of all history reads and
writes is worth only **0.034833 ms/token**. More decisively, the complete
causal-convolution kernel costs only **0.260622--0.283672 ms/token** in two
production traces, below the 0.30-ms gate even under impossible free removal.
No default-off APW experiment is implemented. See the
[L2 ceiling rejection](metadata/qwen36-27b-decode-causal-conv-l2-persistence-ceiling-rejection.json).

The remaining cross-kernel/core-flow boundaries likewise fail admission.
Ordinary single boundaries top out near **0.077--0.095 ms/token** after adding
their observed GPU gaps and unique BF16 publication traffic. Gate-to-Down still
requires a cross-CTA global broadcast, and Down-to-next's large logical reads
are overwhelmingly L1 hits. Even an impossible additive treatment of both
adjacent linear boundaries reaches only about **0.154 ms/token**. Generic
fusion and a topology-mismatched mega-kernel are not next-step candidates. See
the [core-flow rejection](metadata/qwen36-27b-decode-cross-kernel-core-flow-ceiling-rejection.json).

The structurally distinct GDN row16 register-baton screen is also closed. It
removes all five shared-state transfers and passes the full bitwise correctness
and resource gates at **71r / 2,568 B shared / 0 local / 3 CTA/SM**. Two
independent 24-bank cold-state processes project **0.251234** and **0.247622
ms/token**, both below the frozen 0.30-ms/token hard gate. The candidate is
removed, production is unchanged, and MTP was not used. Reopen GDN state work
only for a materially different exact topology with a credible contribution
above 0.30 ms/token. See the
[row16 rejection](metadata/qwen36-27b-decode-gdn-row16-register-baton-rejection.json).

The LM-head q20 progressive schedule family is now closed by its formal exact
sector gate. Canonical inputs and the directed BF16 suffix-bound proof pass,
but the independently implementable per-CTA/`-1` schedule saves only
**19.244--34.044 MB/call**, averaging **26.311 MB/call**, so 0/25 fixtures reach
55 MB. Completed-wave/`-1` reaches 25/25 only as an idealized schedule: its
worst margin is **54,752 bytes** before pricing **121 global completion
boundaries**. One-wave lag reaches 24/25, and no contiguous-seed sweep point
passes all fixtures. No CUDA, NCU, or production work follows; the formal
**105.870500-ms/token / 9.445501816-token/s** anchor is unchanged and MTP was
not used. Reopen only for a materially different exact incumbent source or
row-search structure without an unpriced global-progress assumption. See the
[q20 exact-sector rejection](metadata/qwen36-27b-lm-head-q20-exact-sector-rejection.json).

- [active, Decode baseline lock and Prefill handoff] Freeze the achieved
  non-MTP P19/C32/max26 fixed-clock result at **105.870500 ms/token /
  9.445501816 token/s** as the Phase 3 Decode regression anchor. The original
  no-more-than-100-ms/token and at-least-10-token/s objective remains an unmet
  stretch target, but no longer gates dedicated Prefill work. Earlier Phase 3
  statements that placed Prefill behind that gate are historical decisions
  superseded by this handoff. MTP remains excluded.
- [done, current-HEAD Prefill baseline lock] At `edef543`, one fixed-frequency
  reusable-engine process with one warmup and five measured generations per
  prompt establishes the C32/max1 direct baseline. P33/P65/P129/P513 median
  prefix times are **271.159/543.646/1,102.542/4,605.071 ms**, median TTFTs
  are **378.074/650.889/1,209.974/4,713.890 ms**, and median-derived prefix
  throughputs are **118.011941/117.723666/116.095351/111.181782 token/s**.
  All 20 generations emit token `9419` (`Hello`) with zero persistent memory
  drop. The separately run P19/max26 control reproduces all 26 oracle tokens
  and 125/0 Graph/serial Decode steps, but late-process thermal drift and a
  164,122,624-byte free-memory drop make it diagnostic rather than a
  replacement for the frozen formal Decode anchor. See the
  [current-HEAD Prefill baseline](metadata/qwen36-27b-current-head-prefill-baseline.json).
- [done, Prefill NVFP4 M64 down schedule selection] A production-unreachable
  exact `[M=64,N=5120,K=17408]` candidate reuses every decoded K64 weight tile
  across four M16 WMMA accumulator panels while retaining the M32 shared-A/C
  footprint. Against two ordered production M32 launches it is bitwise exact
  for both synthetic scale distributions, captures as one Graph kernel, uses
  76 registers/23,552 shared bytes/zero local bytes, and improves all 12
  fixed-frequency rounds. Checkpoint-like and same-bank cells reach
  **1.25824x** and **1.25778x**; aggregate speedup is **1.25801x**, above the
  1.15x gate. Production dispatch remains unchanged. See the
  [M64 down screen](metadata/qwen36-27b-prefill-nvfp4-m64-down-screen.json).
- [done, C64 runtime and M64 down production promotion] ABI 0.3.0 raises the
  request/projection boundary to C64. Exact aligned NVFP4
  `[M64,N5120,K17408]` down uses one M64 kernel; other C64 projections retain
  two ordered C32 schedules, residual/RMS retains two M32 operations, causal
  work remains at most M16, and partial-wide 33..63 candidates are scheduled as
  C32 plus an ordered tail. Fixed-clock mirrored Prefix speedups are
  **1.065650x P65, 1.042762x P97, 1.064499x P129, and 1.062855x P513**;
  P33 is neutral at 1.001030x. All formal outputs remain exact. P513 Nsight
  confirms 1,024 M32 down launches become 512 M64 launches and the Prefix range
  falls from 4,644.437 to 4,376.232 ms. See the
  [C64 production record](metadata/qwen36-27b-prefill-c64-down-production-benchmark.json).
- [rejected, synchronous M64 NVFP4 Gate/Up] The exact test-only
  `[M64,N17408,K5120]` kernel is bitwise equal to two production M32
  raw-weight cp.async launches, uses one Graph node, and clears its
  76-register/23,552-byte/zero-local/3-CTA resource gate. Its isolated
  aggregate is only 1.08934x, and the joined production-like dual-stream pair
  is 1.07884x with all 12 rounds positive. That pair misses the 1.12x gate
  derived from the later 1.03x P513 Prefix threshold, so dispatcher and runner
  remain unchanged. See the
  [M64 Gate/Up rejection](metadata/qwen36-27b-prefill-nvfp4-m64-gate-up-rejection.json).
- [done, FP8 M64 attention output production promotion] Commit `5df6ca6`
  selects one exact aligned `[M64,N5120,K6144]` kernel in place of two ordered
  M32 dual-resident-A launches. The production gate passes at **69 registers /
  23,552 B shared / zero local / 3 CTA/SM**, exhaustive E4M3FN correctness is
  bitwise exact, and six micro rounds aggregate at **1.49086x**. Fixed-clock
  mirrored full-model Prefix speedups are **1.025063x P65, 1.015764x P97,
  1.024397x P129, and 1.023902x P513**; the unchanged P33 fallback remains
  within its 0.5% gate and improves in the longer control. P513 reaches
  **119.838644 token/s**, while Nsight confirms 1,024 M32 launches totaling
  378.191584 ms become 512 M64 launches totaling 279.108096 ms. See the
  [FP8 M64 production record](metadata/qwen36-27b-prefill-fp8-m64-attention-output-production-benchmark.json).
- [done, Prefill reference-architecture audit] FlashInfer commit `4b969c9`
  and qwen35-thor commit `57e2977` were inspected read-only. FlashInfer FA2's
  bulk causal GQA contract is a source-level match for contiguous BF16 NHD
  Q24/KV4/D256 attention on SM87, but its current GDN Prefill routes require
  SM90 or newer. qwen35-thor does not call FlashInfer in production; its main
  transferable mechanism is a layer-major T>1 path with up to 2,048 tokens per
  GEMM-bearing chunk, followed by bulk attention and an eight-token WY GDN
  recurrence. SM110 TMA/UMMA/PDL and external performance numbers are not
  portable evidence. See the
  [Prefill reference audit](PREFILL_REFERENCE_AUDIT.md).
- [done, whole-chunk large-M grid screen] Commit `0196751` keeps the promoted
  `[M64,N5120,K6144]` CTA arithmetic and compares repeated production M64
  launches with one M-major control grid and one N-major candidate grid. Three
  fixed-clock processes put M256 at 1.26591x median and M512 at **1.29047x**
  median; all 36 baseline-comparison rounds improve. N-major adds only 1.01153x
  over the M-major one-grid control at M512, so most gain is removal of repeated
  under-filled 40-CTA grid boundaries, not a standalone L2 claim. Exhaustive
  E4M3FN, 4,096 classified NaNs, replay, guards, inputs, invalid capture, and
  Graph topology pass. N-major adds one address register but preserves
  23,552-byte shared memory, zero local memory, and three CTA/SM. Production
  dispatch remains unchanged. See the
  [whole-chunk screen](metadata/qwen36-27b-prefill-fp8-whole-chunk-grid-screen.json).
- [done, NVFP4 whole-chunk down canary] Commit `5dc256b` compares four/eight
  exact production M64 down launches with one M/N-major M256/M512 grid. Three
  fixed-clock processes put M256 at **1.29624x** median and M512 at
  **1.34655x** median; all 72 baseline rounds improve. N-major adds 1.03288x
  over the M-major M512 control. Both scale distributions are bit-exact with
  replay, guards, inputs, invalid capture, and Graph topology passing. The
  candidate uses 79 registers versus production 76 while retaining 23,552 B
  shared, zero local, and three CTA/SM. Down alone projects to 1.04647x P513
  Prefix; production dispatch remains unchanged. See the
  [NVFP4 down screen](metadata/qwen36-27b-prefill-nvfp4-whole-chunk-down-grid-screen.json).
- [done, production-like NVFP4 Gate/Up whole-chunk screen] Commit `d9c8aa6`
  measures the true main/aux-stream pair against public M32 chains. Three
  fixed-clock processes put M256 at **1.13507x** and M512 at **1.12867x**;
  all 72 shape/distribution rounds improve and every process clears the frozen
  1.12x M512 gate. Candidate/control/baseline outputs are bit-exact, replay,
  guards, inputs, invalid capture, Graph topology, and resources pass without
  occupancy loss. The phase-local P513 opportunity is 1.03392x; production
  dispatch remains unchanged. See the
  [Gate/Up pair screen](metadata/qwen36-27b-prefill-nvfp4-whole-chunk-gate-up-pair-screen.json).
- [done, direct FP8 whole-chunk QKV/Z screen] Commit `3134c38` compares the public
  production M32 chains with repeated test-only M64, one M-major grid, and one
  N-major grid for exact C256/C512 QKV `[N10240,K5120]` and Z
  `[N6144,K5120]`. The sequential M512 pair reaches **1.50681x** overall,
  including 1.56823x checkpoint-like and 1.42834x stress cells; every
  individual and pair round improves. Exhaustive E4M3FN, replay, Graph,
  invalid-call, guard, input, and resource gates pass at 70 registers,
  23,552 B shared, zero local memory, and three CTA/SM. This admitted the narrow
  production route below; the screen itself did not change dispatch. See the
  [FP8 QKV/Z screen](metadata/qwen36-27b-prefill-fp8-whole-chunk-qkv-z-screen.json).
- [done, production FP8 C256/C512 QKV/Z/O whole chunks] Commit `10c4c85`
  routes exact aligned QKV `[10240,5120]`, Z `[6144,5120]`, and attention
  output `[5120,6144]` through one N-major grid while keeping the generic
  projection cap at C64 and every near-miss fallback. Frozen-binary
  `B1-C1-C2-B2` reaches **1.098411534x/1.099544898x Prefix** and
  **1.092760304x/1.096639387x TTFT** at P257/P513. All 40 generations retain
  ID 9419, `Hello`, exact steps, one common streamed contract hash, and zero
  persistent drop. P513 Nsight confirms 48/48/64 QKV/Z/O launches and the
  intended 2,560-to-160 node transition. Native complete-prompt throughput is
  now 134.351126/137.928939 token/s, still 2.780617x/2.982587x behind the
  measured stock vLLM reference. See the
  [production record](metadata/qwen36-27b-prefill-fp8-whole-chunk-production-benchmark.json).
- [selected, test-only FP8 full-attention Q/K/V whole chunks] Commit `de86613`
  tests exact C256/C512
  Q `[12288,5120]` and K/V `[1024,5120]` pass independent correctness,
  replay, Graph, invalid-call, guard, input, resource, and every-round-positive
  gates. At C512, Q reaches **1.52743x/1.39116x**, K reaches
  **7.96850x/8.56814x**, and V reaches **7.97343x/8.53129x** for
  checkpoint-like/stress fixtures. Sequential Q-then-K-then-V reaches
  **2.54717x/2.57488x** and **2.55933x** aggregate. Production dispatch is
  unchanged by the screen. Promotion must choose C256 K/V layout deliberately
  because its
  32-CTA N-major grid underfills the device and trails the M-major control;
  then pass exact model, memory, fixed-clock Prefix/TTFT, and fresh Nsight
  gates. See the
  [full-attention screen](metadata/qwen36-27b-prefill-fp8-whole-chunk-full-attention-screen.json).
- [done, production FP8 C256/C512 full-attention Q/K/V whole chunks] Commit
  `86d5843` promotes exact Q `[12288,5120]` and K/V `[1024,5120]`. Q uses
  N-major at C256/C512; K/V use M-major at C256 and N-major at C512. Frozen-
  binary `B1-C1-C2-B2` reaches **1.054381379x/1.055108640x Prefix** and
  **1.051134118x/1.053417066x TTFT** at P257/P513. Complete-prompt throughput
  is now **141.273450981/145.327508146 token/s**, leaving
  **2.644368049x/2.830744195x** gaps to matched stock vLLM. All 40 outputs and
  steps remain exact. Fresh P513 Nsight confirms the full-attention projection
  transition from 2,304 launches / 314.357120 ms to 48 launches / 124.991296
  ms. Both candidate processes have zero persistent drop; B1's 66,441,216-byte
  drop remains below tolerance. The remaining Prefix is still one-stream;
  Gate/Up contributes 2,048 serial launches / 1,346.373984 ms / 39.199622%,
  selecting the already-screened C512 whole-chunk main/aux pair as the next
  production integration. See the
  [production record](metadata/qwen36-27b-prefill-fp8-whole-chunk-full-attention-production-benchmark.json).
- [done, production NVFP4 C256/C512 whole-chunk Gate/Up] Commit `d1fa6c5`
  routes exact `[17408,5120]` Gate and Up through one whole-chunk kernel per
  branch on the existing main/aux event fork/join. Frozen-binary P257/P513
  Prefix improves **1.087506234x/1.086814433x** to
  **162.636231668/162.584844997 token/s**; TTFT improves
  **1.081889974x/1.083922378x**. All 40 formal outputs and steps remain exact
  and every process has zero persistent drop. Fresh P513 Nsight reduces the
  pair from 2,048 nodes / 1,346.373984-ms union to 128 nodes /
  1,065.953440 ms and reduces all Prefix nodes from 10,129 to 8,209. Only
  15.839296 ms of the new pair overlaps, so whole-chunk work accounts for
  94.35% of its union saving and the result is not described as buffering.
  The subsequent M128 B-tile-reuse screen is recorded below. See the
  [Gate/Up production record](metadata/qwen36-27b-prefill-nvfp4-whole-chunk-gate-up-production-benchmark.json).
- [done, historical test-only NVFP4 Gate/Up M128 B-reuse screen]
  Commit `1f8779c` reuses each decoded/staged B tile across two adjacent M64
  token panels. Three frozen-binary processes reach
  **1.275707061x/1.283725529x** aggregate single-branch speedup at M256/M512.
  The main/aux Gate/Up pair reaches process-minimum
  **1.27998x/1.28465x**; every one of 72 pair rounds improves, and its
  all-round minima are **1.27930x/1.28349x**. Branch exact/replay/guard/input,
  pair exact/replay/guard, invalid-call, Graph, and
  126-register/37,376-byte-shared/two-CTA resource gates pass. Production
  remained on M64 at screen time. Its conservative, then-unimplemented P513
  projection was **2,912.933316 ms / 175.767841 token/s**. The production
  promotion immediately below closes that plan while this record retains its
  historical screen status. See the
  [M128 screen](metadata/qwen36-27b-prefill-nvfp4-gate-m128-b-reuse-screen.json).
- [done, production NVFP4 C256/C512 Gate/Up M128 B reuse] Commit `676e8ad`
  promotes the screened kernel after pair Graph, partial/wrapping alias,
  exact-checkpoint, resource, memory, and full-suite gates. Formal M64 and
  M128 binaries contain the same span-safety fix. Mirrored P257/P513 Prefix
  improves **1.079886387949x/1.081515606806x** to
  **175.547351155/175.730487094 token/s**; complete-prompt throughput reaches
  **164.104109691/169.740778038 token/s**. All 40 outputs and canonical
  contracts match, every process has zero persistent drop, and the suite has
  52 passes/12 expected skips/zero failures. Fresh P513 Nsight keeps 8,209
  Prefix and 128 Gate/Up nodes, halves Gate/Up grid X from 1,088 to 544, and
  reduces its union from 1,065.953440 to 827.889280 ms
  (**1.287555553x**). The existing two-stream fork/join remains; the gain is
  M128 B reuse, not new buffering or overlap. See the
  [M128 production record](metadata/qwen36-27b-prefill-nvfp4-gate-m128-production-benchmark.json).
- [measured and rejected on real checkpoint, GDN sequential FP32-B8 and WY]
  Commit `eaa09f4` compares production M16 (`B`), sequential FP32-B8 (`S`), and a B8
  lower-triangular WY control (`W`). C256 reaches **2.76977x S** versus
  1.74177x W; C512 reaches **2.78551x S** versus 1.74228x W. Sequential
  latency is only **0.628852/0.625479** of WY, and S beats W in all three
  mirrored rounds at both shapes. CPU C1/C7/C8/C9/C15/C16, split-tail,
  numerical, immutable-input, invalid-contract, and 109-register/8,256-byte-
  shared/two-CTA gates pass. The 37,748,736-byte rotating state pool exceeds
  the 4-MiB L2 but makes no hit-rate claim. Commits `feb18a7`, `60f376e`, and
  `9c4be88` then prove real route execution at P257/P513/P769/P1025. Short
  outputs remain exact, but Prefix aggregate state NRMSE grows
  **0.0741172/0.115284/0.136871/0.148576** against the frozen 0.01 gate.
  FP32-B8 is therefore rejected alongside WY; production remains exact M16.
  The default OFF executable is bitwise identical to the M128 production
  binary. See the historical
  [GDN B8 screen](metadata/qwen36-27b-prefill-gdn-b8-block-transition-screen.json)
  and the
  [real-checkpoint rejection](metadata/qwen36-27b-prefill-gdn-b8-real-checkpoint-rejection.json).
- [measured and rejected, GDN whole-span register state] Commit `9572c2a`
  reduces exact C256/C512 production M16 chains from 16/32 nodes to one while
  retaining packed BF16 recurrent state across the complete span. Bitwise,
  replay, guard, input, Graph, invalid-call, and resource gates pass at
  64 registers, 34,056 B shared, zero local memory, and four CTA/SM. Every
  mirrored round improves, but C256 reaches only **1.02672x** and C512 only
  **1.01871x**, both below the frozen 1.03x gate. First-process stop-loss skips
  replication, runner/full-model, NCU, and Nsys work; production is unchanged.
  Do not repeat longer lifetime with the same sequential recurrence body. A
  future GDN screen required a materially different algorithm/dataflow; the
  B8 block-transition screen above now supplies that new test-only direction,
  while this whole-span result remains rejected. See the
  [GDN rejection](metadata/qwen36-27b-prefill-gdn-whole-span-register-state-rejection.json).
- [closed, scheduling-only persistent Down P0] Commit `03336b6` uses equal-byte
  NK64/NK256 sidecars plus 16 static-stride CTAs but reaches only
  **0.511096x**, with all 18 rounds regressing. Grid reduction without B
  arithmetic reuse is closed. Any future persistent attempt must reuse each
  staged B tile across at least M128/M256 work and/or implement the real
  four-stage `cp.async` plus register-side E2M1 MMA pipeline. See the
  [P0 rejection](metadata/qwen36-27b-prefill-nvfp4-down-persistent-packed-p0-rejection.json).
- [done, one C512 public Prefill boundary] Commit `6be943e` publishes
  ABI/package 0.4.0, explicit `{512,256,64,32,tail<=31}` scheduling, C256
  canary coverage, exact 131,426,304/174,991,360-byte default arenas, and the
  17,437,720,576-byte absolute ceiling. Generic projections remain capped at
  C64; FP8 output/down selectors no longer inherit the request maximum;
  residual/RMS and the first-64 fused GQA prefix retain their old arithmetic.
  The 62-test suite has 51 passes, 11 expected skips, and no failures. Default
  remains C1 and no candidate route is enabled, so this is not a performance
  result. See the
  [C512 boundary](metadata/qwen36-27b-prefill-c512-request-boundary.json).
- [done, current C512 optimized-route bundle] Commits `1f7d6be`, `6327733`,
  `10c4c85`, `86d5843`, `d1fa6c5`, and `676e8ad` route exact C256/C512 bulk
  full-attention compute, NVFP4 Down and Gate/Up, FP8 linear-attention
  QKV/Z/O, and FP8 full-attention Q/K/V behind the ABI-0.4 boundary.
  P257/P513 exact-token, persistent-state,
  memory, fixed-clock Prefix, and fresh-profile gates pass. C64/C32/tail
  fallbacks, Decode Graph, and default C1 remain. The early synchronous M64
  Gate/Up recheck was rejected, but the later production-like whole-chunk
  main/aux pair reached 1.12867x at C512 and then passed production admission.
  The current Gate/Up route uses one M128 B-reuse kernel per branch and the
  existing event join; it does not introduce an independent executor or
  buffering pipeline.
- [done, native bulk attention production route; queued new-mechanism GDN]
  Commit `3044ab5` validates
  the dependency-free QT2/BK16 SM87 bulk causal GQA plus fused Gate prototype.
  Three processes reach **6.33538x C256** and **4.53722x C512**, every round is
  positive, resources are 64 registers/16 KiB shared/zero local/five CTA-SM,
  and the declared numerical/append/replay/Graph/invalid contracts pass. The
  1.05804x P513 projection excludes Q/K preprocessing and KV placement, so
  production admission then passed in `1f7d6be` and in the combined C512
  full-model gates summarized above. See the
  [bulk GQA screen](metadata/qwen36-27b-prefill-bulk-causal-gqa-screen.json).
  A future GDN attempt must use a materially different algorithm or dataflow
  from the rejected whole-span register-lifetime extension. Decode remains
  frozen and non-MTP.

Closed
table-free, half-tile, pair-fused, and shared-pipeline variants are not
candidates for restoration. Graph replay now removes repeated host submission
for the selected positions, but adds no double/triple buffering, GPU overlap,
separate Prefill/Decode executor, or phase overlap; those remain later
multi-request/serving concerns rather than claims of this promotion.
Rejected
product-table, pair-fused CTA, shared A/B pipeline, scale-window ping-pong,
activation-only `cp.async`, generalized dual-stream, and GDN shared-resident
variants stay closed without a materially different mechanism. The existing
logical plan split is sufficient for independent phase tuning; it is not an
independent executor or queue.
Independent executors and queues remain a future multi-request continuous-
batching item after request/KV/workspace ownership, handoff, fairness, TTFT,
inter-token, tail-latency, cancellation, and memory gates are explicit.

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

The achieved non-MTP Decode result is frozen as the Phase 3 regression anchor.
The 100-ms/token / 10-token/s objective remains documented but is no longer a
prerequisite for Prefill work. The C256/C512 whole-chunk main/aux-stream
Gate/Up route now uses production M128 B reuse and passes exact model, memory,
mirrored-latency, resource/invalid, full-suite, and fresh-profile gates.
P257/P513 Prefix reaches **175.547351155/175.730487094 token/s** and complete-
prompt throughput reaches **164.104109691/169.740778038 token/s**. The layer
still has one Gate and one Up node on the existing two-stream fork/join; no
new double/triple buffering or Prefill/Decode overlap was introduced.
The parallel GDN screen measured sequential FP32-B8 at **2.76977x/2.78551x**
over production M16 and rejected WY, but the subsequent real-checkpoint gate
also rejects FP32-B8: Prefix aggregate state NRMSE rises from **0.0741172** at
P257 to **0.148576** at P1025 against a 0.01 threshold. Exact short token
output does not override recurrent-state drift. Production therefore stays
on exact per-token BF16 M16 GDN. The immediate main line is exact FP8 large-N
C256/C512 M128 B-tile reuse: QKV/Z/O/full-Q currently consume 739.064832 ms,
or 25.46% of P513 Prefix union. NVFP4 Down M128 reuse follows. Bounded
OpenAI-compatible API/EvalScope work starts in parallel so the kernel path and
external baseline advance together.
C1024 remains a low-priority single-kernel canary rather than the main route.
Exact output and Decode non-regression gates remain mandatory. Multi-request
serving throughput is a separate later target.

Before Phase 3.5, retain the completed offline same-token vLLM/FlashInfer
matrix as the cross-framework reference. Its batch-one/output-one comparison
uses `P / scheduled-to-first-token`; native direct timing uses complete-prompt
`P/TTFT`. Current native production reaches **164.104109691/169.740778038
token/s** at P257/P513, while matched stock vLLM reaches 373.579/411.385.
These are different runtime systems and leave directional
**2.276478024670x/2.423607680423x** gaps, not a same-kernel attribution. The
user's
separately tuned 2k--8k range has no matched raw protocol and is not treated as
a comparable result. GDN B8 admission is now closed as rejected. Introduce
the HTTP adapter and EvalScope in parallel with the FP8 M128 screen and its
traffic counters so the external baseline precedes later system-level
optimization. Phase 3.5 remains
where EvalScope and user-visible TTFT become first-class release evidence.

The first P65/P513 route smoke is complete: the exact checkpoint uses
FlashInfer for 16 full-attention layers, GDN for 48 layers, and Marlin for
FP8/NVFP4 projections. It returns the same first token ID 9419 and measures
235.977/414.612 prompt token/s at P65/P513. These one-warmup/one-measure values
validate the harness and narrow the practical comparison, but remain below the
formal three-process gate and below the separately optimized 2k--8k target.
See the
[vLLM smoke record](metadata/qwen36-27b-vllm-flashinfer-prefill-smoke.json).

The vLLM-side formal matrix is also complete: three 3-warmup/10-measure
processes reach 236.380/312.828/373.579/411.385/432.738 prompt token/s at
P65/P129/P257/P513/P1025, with 150/150 trusted timestamps and first token ID
9419. The current native production boundary and mirrored ordering are
complete: P257/P513 reach 164.104109691/169.740778038 token/s, leaving
directional stock-vLLM gaps of **2.276478024670x/2.423607680423x**. Native
`Prefix` still
excludes the final
prompt token and LM head, so cross-framework comparisons use complete-prompt
`P/TTFT`, not `(P-1)/Prefix`. See the
[formal vLLM record](metadata/qwen36-27b-vllm-flashinfer-prefill-reference.json).

## Phase 3.5 — External evaluation gateway

Deliverables:

- One resident model and one serialized batch-one GPU worker behind a bounded
  request queue; this is an evaluation adapter, not continuous batching.
- Text-only `POST /v1/chat/completions` and raw text/token-ID
  `POST /v1/completions`, with explicit supported-field validation.
- True per-token SSE, non-streaming responses, exact token usage, cancellation
  at a committed token boundary, `GET /v1/models`, and `GET /healthz`.
- Pinned EvalScope capability smoke tests plus single-request Prefill, Decode,
  and end-to-end latency matrices.

Exit criteria:

- CLI, non-streaming API, and reassembled streaming output are identical for
  deterministic supported requests.
- The first SSE content event is emitted only after a real generated token;
  post-hoc pseudo-streaming is rejected for TTFT claims.
- Direct CUDA/engine timing and HTTP-visible timing remain separate reports.
- Unsupported sampling, tool, media, and concurrency semantics fail clearly
  rather than being silently ignored.

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
- Harden and extend the Phase 3.5 evaluation gateway into a concurrency-safe
  production server.
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
