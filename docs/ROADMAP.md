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
serialized. The pre-promotion refreshed profile ranked NVFP4 M32 gate/up and
down first by marginal critical-path exposure, followed by GDN and FP8 M32.
Gate/up pair CTA fusion and down shared A/B ping-pong are now evidence-closed,
and the bounded
register-resident Prefill GDN M16 candidate has cleared its zero-spill,
four-CTA, exact-boundary, and 1.20x gates. The next main-line action is a fresh
production phase profile after this promotion, followed by a bounded mechanism
chosen from marginal critical-path exposure rather than raw overlapped time.
Rejected product-table,
pair-fused CTA, shared A/B pipeline, scale-window ping-pong, activation-only
`cp.async`, generalized dual-stream, and GDN shared-resident variants stay
closed without a new mechanism.
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
