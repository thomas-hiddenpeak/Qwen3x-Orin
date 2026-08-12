# Pinned checkpoint metadata evidence

> **Document control**
> - Class: `evidence_index`
> - Authority: artifact discovery and exact per-record authority only
> - Status: historical index; not current status or an active roadmap
> - Current truth: [`../CURRENT_STATUS.md`](../CURRENT_STATUS.md)

This directory records normalized facts used to design and test Qwen3x-Orin.
The reports are evidence artifacts, not model files and not end-to-end support
claims. A report reaches only `metadata-compatible` until all referenced shard
headers and tensor contracts pass the runtime inspector on the target Orin.

The first OpenAI/EvalScope whole-product direction baseline is recorded in
[`qwen36-27b-evalscope-external-directional-baseline-2026-07-29.json`](qwen36-27b-evalscope-external-directional-baseline-2026-07-29.json).
It compares 32 identical real ShareGPT token-ID requests against native and
stock vLLM, establishes the 2.768687x TTFT gap as a Prefill architecture
priority, and records the first C-Eval attempt as an invalid output-cap
measurement. Its authority is roadmap prioritization only: it is not a
release baseline, production promotion, threshold reset, or capability score.
The fresh stock-vLLM run is separately frozen in
[`qwen36-27b-evalscope-vllm-frozen-reference-2026-07-29.json`](qwen36-27b-evalscope-vllm-frozen-reference-2026-07-29.json).
It is a cumulative market floor and is not rerun for each native candidate.
The sealed balanced layer-major P1025 API/EvalScope direction screen is frozen
in
[`qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json`](qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json).
It retains only the complete sealed composition direction for the next
operator-panel integration step; its older comparator used FP8, Attention,
and GDN fallbacks, so the result has no constituent-level attribution,
architecture-selection, release, or production authority.

The first real-checkpoint preparation of the default-off SM87 whole-system AOT
candidate is frozen in
[`qwen36-27b-sm87-target-aot-real-checkpoint-preparation-2026-08-12.json`](qwen36-27b-sm87-target-aot-real-checkpoint-preparation-2026-08-12.json).
It records 128 independently read-back NVFP4 artifacts from 192 pinned source
tensors in one 9,625,927,680-byte device arena, their private `ModelWeights`
attachment, and the exact payload-catalog digest. The source probe remains
formally `fail` at its in-process `cudaMemGetInfo` recovery gate. A separate,
bounded post-exit `no_owner_leak` diagnosis combines the first hash-frozen
page-pool/IOVMM snapshot with a later complete process/IOVMM/FSI/handle audit;
it does not rewrite the source result. The record has preparation-dependency
authority only and grants no numerical, launcher, generation, API,
performance, release, or production authority.

The next default-off real-checkpoint gate is frozen in
[`qwen36-27b-sm87-target-aot-layer0-m192-oracle-2026-08-12.json`](qwen36-27b-sm87-target-aot-layer0-m192-oracle-2026-08-12.json).
The private layer-0 M192 Gate+Up and Down-plus-residual candidate is bitwise
identical to the canonical baseline and replay across the full M128 region and
predicated M64 tail. The same ELF passes its SM87 geometry/resource gates. The
child source status remains `fail` only at immediate in-process memory
recovery; the original parent lifecycle report remains `inconclusive`, while a
strict reparse of its immutable immediate Jetson `nvmap` snapshot yields
`no_owner_leak` with 20/20 criteria. This record grants only layer-0 M192
numerical, same-ELF resource, and bounded lifecycle-diagnostic authority. It
grants no complete-model, public-launcher, generation, API, performance,
release, or production authority.

The rejected real-API P40K screen of the explicit segmented-Marlin projection
wrapper plus grouped-Q64 Attention is frozen in
[`qwen36-27b-prefill-p40k-segmented-marlin-q64-api-2026-08-09.json`](qwen36-27b-prefill-p40k-segmented-marlin-q64-api-2026-08-09.json).
EvalScope reported 179,511.19 ms TTFT and 222.827281 prompt tok/s; server pure
Prefill was 179,395.678907 ms. The v4 witness records five logical panels,
1,680 segmented projection hits, 12,992 physical Marlin launches, and 80
grouped-Q64 Attention hits, with zero Prefix, MTP, cuBLASLt,
external-reference, or generic approximate-route hits. The route remains
default-off and accuracy-unqualified. Because it crossed the predeclared
`>=165 s` stop-loss, it was rejected without P60K, P130K, NSys, or NCU. The
record distinguishes this wrapper from true native large-M and sends the next
work to shape-specific M128 Gate/Up, Down, and FP8 projection ownership.

The first real-API P40K Nsight attribution of the explicit, default-off
native-group-Q64-panel route is frozen in
[`qwen36-27b-prefill-p40k-native-group-q64-panel-nsys-2026-08-09.json`](qwen36-27b-prefill-p40k-native-group-q64-panel-nsys-2026-08-09.json).
The single clean-host capture records 180,345.412463 ms of server pure
Prefill (221.796604 tok/s), with 46.8% of kernel time in grouped-Q64 causal
Attention and 43.4% in the two dominant still-segmented Marlin signatures.
It also records one CUDA stream, zero kernel overlap, and only 3.738965 ms
between external EvalScope TTFT and server TTFT, so the loopback API is not
the dominant observed budget. This record is T4 diagnostic evidence only:
the route is accuracy-unqualified, generated a different first token from
the exact segmented direction context, remains non-default, and has no
architecture-selection, release, or production authority. EvalScope 1.9.1
is the declared invocation version but is not self-attested by the retained
raw artifacts. P60K and P130K were not run.

The first executable `AC-PREFILL-PROMPT-WIDE-v2` slice and its clean-host
cold/no-cache P40K API direction screen are frozen in
[`qwen36-27b-prefill-p40k-flashinfer-exact-panel-api-2026-08-09.json`](qwen36-27b-prefill-p40k-flashinfer-exact-panel-api-2026-08-09.json).
The v6 route replaces repeated exact QT2 Attention with 80 FlashInfer
M8192/M7712 logical-panel calls and records zero QT2/Q64/Q128 hits. EvalScope
TTFT was 109.02622 s and server pure Prefill was 108.981855 s, or 367.033577
prompt tok/s: 6.15x faster than the preceding exact route. The P40K first
token `The` matches that route; the P513 full-state hash does not, and the
complete accuracy gate was not run. The result therefore retains only the
Attention dataflow direction. It remains default-off, accuracy-unqualified,
and far below the owner's vLLM starting line; P60K/P130K and production
promotion have no authority from this record.

The rejected first true-large-M NVFP4 operator-panel P40K API screen and its
matched Nsight attribution are frozen in
[`qwen36-27b-prefill-p40k-nvfp4-true-large-m-rejection-2026-08-10.json`](qwen36-27b-prefill-p40k-nvfp4-true-large-m-rejection-2026-08-10.json).
The candidate used native M128N256K64 Gate/Up and Down kernels, but regressed
server pure Prefill from 108,981.854892 ms to 136,929.917554 ms. The API
boundary contributed only 4.046819 ms, while matched profiling attributed
27.605252 s of added kernel time to NVFP4 projection kernels. The route is
therefore rejected, default-off, and accuracy-unqualified; its evidence sends
the next work to distinct G2 paired Gate/Up and D2 Down dataflows rather than
parameter scanning of the rejected skeleton.

The rejected coupled GateUpG2/DownD2 v1 P40K API direction and bounded Nsight
attribution are frozen in
[`qwen36-27b-prefill-p40k-nvfp4-g2-d2-rejection-2026-08-10.json`](qwen36-27b-prefill-p40k-nvfp4-g2-d2-rejection-2026-08-10.json).
The v8 request consumed all 40,000 tokens and attested 320 G2 plus 320 D2 hits
with zero Prefix, MTP, cuBLASLt, external, approximate, exact-fallback, or
forbidden-route hits. Server pure Prefill was 115,041.751913 ms / 347.699851
tok/s versus 108,981.854892 ms / 367.033577 tok/s for retained `c45b7c5`, a
+5.560464% latency regression. Bounded NSys measured G2+D2 at 60.904184288 s,
7.040159744 s above retained native Marlin main plus SiLU, and complete GPU
kernel time increased by 6.110398080 s. The matching first token is not an
accuracy qualification. P60 was not run because its balanced M5424 tail is
unsupported by the M8192/M7712-only candidate; this is a geometry blocker, not
a P60 performance result. G2/D2 v1 is rejected, default-off, and closed.

The rejected exact-P40000 persistent layer-wide MLP direction is frozen in
[`qwen36-27b-prefill-p40k-persistent-layerwide-mlp-rejection-2026-08-10.json`](qwen36-27b-prefill-p40k-persistent-layerwide-mlp-rejection-2026-08-10.json).
Its v9 witness consumed all 40,000 tokens through the OpenAI API and recorded
one 64-layer route pass, 64 persistent GateUp calls, 64 persistent
Down+residual calls, 128 persistent NVFP4 physical launches, and zero
forbidden routes. Server pure Prefill was 109,028.398424 ms / 366.876892
tok/s versus 108,981.854892 ms / 367.033577 tok/s for retained `c45b7c5`, a
0.042708% latency regression. The direction is therefore rejected before
full accuracy qualification or profiling. The result closes isolated MLP
replacement and sends the next candidate to a whole-core schedule with five
equal M8000 fill/drain panels, prompt-wide Attention/GDN, and aligned
single-launch FP8 panels.

The following exact-P40000 whole-core direction and bounded Nsight attribution
are frozen in
[`qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json`](qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json).
Its v10 witness consumed all 40,000 tokens as five M8000 panels, completed one
64-layer route pass and exactly 768 bounded retirements, and recorded zero
forbidden routes. Server pure Prefill reached 101,831.853876 ms / 392.804397
tok/s, +7.02138% versus retained `c45b7c5`; EvalScope TTFT was 101,870.53 ms.
NSys found 102,113.313600 ms of kernels in a 102,121.306528-ms request, with
Gate/Up, FP8, Down, and Attention dominant. The whole-core substrate is
retained, while its inherited 16-CTA old-Marlin NVFP4 bodies are designated
for architecture replacement. The candidate remains default-off and
accuracy-unqualified.

The first shape-wide Gate/Up and Down package returned through the same real
API P40K gate and is frozen in
[`qwen36-27b-prefill-p40k-nvfp4-shape-wide-v3-rejection-2026-08-10.json`](qwen36-27b-prefill-p40k-nvfp4-shape-wide-v3-rejection-2026-08-10.json).
It regressed server pure Prefill to 106,374.300578 ms / 376.030675 tok/s.
Bounded NSys measured Gate/Up +10.437332%, Down +1.302115%, and the pair
+7.511889% against the whole-core profile. The binary-pinned runner overlay
was removed; the independent kernel surface remains default-off evidence only.

The rejected phase-local BF16 projection direction and bounded Nsight
attribution are frozen in
[`qwen36-27b-prefill-p40k-phase-local-bf16-rejection-2026-08-10.json`](qwen36-27b-prefill-p40k-phase-local-bf16-rejection-2026-08-10.json).
It regressed real-API pure Prefill from 101,831.853876 ms / 392.804397 tok/s
to 124,815.475334 ms / 320.472999 tok/s. The profile assigns 98,999.424800 ms
to the ordinary BF16 dense consumers and only 363.332320 ms to canonical
weight expansion, closing request-time BF16 B materialization as a performance
architecture. The exact measured binary and raw v12 server witness were not
retained, and the experimental topology disagreed with physical lowering;
therefore the record supports negative timing and hotspot attribution only,
not exact replay, sealed-route attestation, accuracy, or production promotion.
Only its default-off correctness primitives remain in the tree.

The rejected complete packed-projection v1 P40K direction is frozen in
[`qwen36-27b-prefill-p40k-packed-projection-rejection-2026-08-10.json`](qwen36-27b-prefill-p40k-packed-projection-rejection-2026-08-10.json).
Its independent v13 witness authenticated 256 packed FP8/NVFP4 artifacts from
400 checkpoint sources and completed the exact 40,000-token OpenAI/EvalScope
route, but reached 161,410.929373 ms / 247.814694 pure prompt tok/s. The route
is default-off, accuracy-unqualified, and rejected against the v10 whole-core
incumbent; it is retained here as the direct non-incumbent comparator for the
NVFP4-only successor.

That successor and its clean-host real-API rejection are frozen in
[`qwen36-27b-prefill-p40k-packed-nvfp4-v2-rejection-2026-08-11.json`](qwen36-27b-prefill-p40k-packed-nvfp4-v2-rejection-2026-08-11.json).
The duplicate-key-free v14 witness consumed all 40,000 prompt tokens, restored
v10's 1,040/1,040 FP8 ledger, and isolated 64 Gate+Up plus 64 Down operations
in 128 packed NVFP4 v2 launches backed by 128 artifacts from 192 real
checkpoint sources. Every fallback, Prefix-cache, MTP, cuBLASLt,
external-reference, approximate-route, and forbidden-route counter was zero.
Server pure Prefill was 128,493.372123 ms / 311.300103 tok/s and EvalScope
TTFT was 128,532.05 ms. This recovers 20.393636% latency versus rejected v13,
but regresses latency 26.181904% and throughput 20.749333% versus the v10
incumbent. Packed NVFP4 v2 is therefore closed without a parameter scan,
remains default-off and accuracy-unqualified, and does not unlock P60/P130 or
production promotion.

The completed stock-vLLM-Marlin projection host-dispatch direction is frozen
in
[`qwen36-27b-prefill-p40k-vllm-marlin-parity-rejection-2026-08-11.json`](qwen36-27b-prefill-p40k-vllm-marlin-parity-rejection-2026-08-11.json).
After an initial commit-profile failure with no timing authority, commit
`986b9fd` corrected the request binding. An r2 sample with a non-allowlisted
CPU consumer was discarded; its timing has no authority. The replacement r3
passed the tightened resource preflight and completed the exact 40,000-token
v15 OpenAI/EvalScope route at 101,857.500727 ms / 392.705493 pure prompt
tok/s. Its 5,120 parity Marlin launches, 64 complete lifetime receipts, one
stable-owner lock clear, and all
zero fallback/forbidden counters prove the integrated schedule identity, not
accuracy. The single valid sample was 0.025179% below v10 throughput and
therefore supplied no positive direction under the predeclared early-stop
gate. The skeleton is closed before repetition, numerical/SASS/full-state
qualification, P60, or P130; it remains default-off reference evidence and
does not alter production.

The complete real-checkpoint MLP exact-zero inventory is frozen in
[`qwen36-27b-nvfp4-mlp-zero-structure-audit-2026-08-11.json`](qwen36-27b-nvfp4-mlp-zero-structure-audit-2026-08-11.json).
It scans all 192 Gate/Up/Down NVFP4 tensors across 64 layers and all three
checkpoint shards: only 7.945540% of E2M1 codes are exact signed-or-unsigned
zero, and only 3.540535% of consecutive K4 groups contain at least two exact
zero codes. This closes exact-zero/2:4 sparsity as a material successor for
the pinned model; it is a CPU structural audit, not a GPU performance or
accuracy result.

The current Prefill architecture reset and same-host stock-vLLM P513 kernel
profile are recorded in
[`qwen36-27b-vllm-p513-architecture-profile-2026-07-29.json`](qwen36-27b-vllm-p513-architecture-profile-2026-07-29.json).
It is an external architecture reference with no production-selection
authority. The promoted exact-C16 GDN/norm/gate route and current native P513
anchor are recorded in
[`qwen36-27b-prefill-gdn-c16-norm-gate-production-2026-07-29.json`](qwen36-27b-prefill-gdn-c16-norm-gate-production-2026-07-29.json).
The subsequent test-only Chunk64/WY architecture proof, its 347.539616 ms
real-P513 saving, full-state numerical characterization, and explicit
non-production decision are recorded in
[`qwen36-27b-prefill-gdn-chunk64-architecture-2026-07-29.json`](qwen36-27b-prefill-gdn-chunk64-architecture-2026-07-29.json).

Policy correction effective 2026-07-29: cuBLASLt is permanently an external
benchmark reference.  It has no production-dispatch, fallback, native
development-retention, or production-promotion eligibility.  Older evidence
artifacts and the historical prose below intentionally preserve the words
`production route`, `bridge`, `admission`, and `rejection` as measurement
provenance; every such cuBLASLt status is revoked and superseded by
[`REAL_MODEL_PERFORMANCE_POLICY.md`](../REAL_MODEL_PERFORMANCE_POLICY.md) and
[`LARGE_M_PROJECTION_DATAFLOW.md`](../LARGE_M_PROJECTION_DATAFLOW.md).  Their
numeric measurements remain useful reference observations, but their old
selection decisions have no current authority.

The current reports pin the official NVIDIA ModelOpt artifacts:

- `nvidia/Qwen3.6-27B-NVFP4` at
  `0893e1606ff3d5f97a441f405d5fc541a6bdf404`;
- `nvidia/Qwen3.6-35B-A3B-NVFP4` at
  `491c2f1ea524c639598bf8fa787a93fed5a6fbce`.

The machine-readable
[`qwen36-27b-native-reference-run.json`](qwen36-27b-native-reference-run.json)
records the first successful full native 27B generation: exact 19 prompt and
26 oracle output IDs, exact decoded text/stop semantics, cold-load and
generation timings, native position-18/19 trace hashes, and the canonical
reasons native and vLLM boundary hashes are not expected to be bitwise equal.

The diagnostic Phase 3 records are:

- [`qwen36-27b-reference-nsys-baseline.json`](qwen36-27b-reference-nsys-baseline.json),
  which pins the reference CUDA kernel and launch-shape profile, toolchain,
  report hash, and reproducible `nsys stats` commands;
- [`qwen36-27b-projection-backend-benchmark.json`](qwen36-27b-projection-backend-benchmark.json),
  which retains every paired reference/SM87 sample, strict replay results,
  exact full-model gate, memory watermarks, microbenchmarks, and limitations.
- [`qwen36-27b-afalg-loader-benchmark.json`](qwen36-27b-afalg-loader-benchmark.json),
  which records the post-SM87 startup diagnosis, observed kernel SHA-256
  provider, accelerated full-shard authentication/load timings, unchanged
  exact-generation gates, and historical-comparison limitations.
- [`qwen36-27b-parallel-loader-benchmark.json`](qwen36-27b-parallel-loader-benchmark.json),
  which records the same-binary warm-cache one-/two-/three-worker resident-load
  measurements, deterministic loader invariants, derived speedups, and
  measurement limitations.
- [`qwen36-27b-startup-overlap-benchmark.json`](qwen36-27b-startup-overlap-benchmark.json),
  which records the same-binary serial/overlapped tokenizer-and-resident
  startup comparison, exact full-model gates, wall-time contract, fallback
  behavior, and measurement limitations.
- [`qwen36-27b-nvfp4-packedx8-benchmark.json`](qwen36-27b-nvfp4-packedx8-benchmark.json),
  which records the canonical packed-x8 NVFP4 dispatch, same-binary scalar/
  vector gate, two-prompt replay benchmark, full-model exact gate, SASS
  resources, and post-change Nsight profile.
- [`qwen36-27b-fp8-packedx4-benchmark.json`](qwen36-27b-fp8-packedx4-benchmark.json),
  which records the canonical packed-x4 FP8 dispatch, exhaustive E4M3FN and
  fallback gates, same-binary scalar/vector measurements, exact 27B replay,
  SASS resources, and the post-change Nsight profile.
- [`qwen36-27b-c8-prefill-benchmark.json`](qwen36-27b-c8-prefill-benchmark.json),
  which records the first bounded C1/C8 prompt-prefix comparison, exact C8
  full-model oracle gate, request-arena increment, execution policy, and
  diagnostic limitations.
- [`qwen36-27b-c8-kernel-optimization-benchmark.json`](qwen36-27b-c8-kernel-optimization-benchmark.json),
  which records the subsequent C8 kernel-optimization chain, matched-shape
  end-to-end medians, GDN and NVFP4 row-pair microbenchmarks, unchanged exact
  full-model gate, dispatch scope, and diagnostic limitations.
- [`qwen36-27b-c16-tensor-core-prefill-benchmark.json`](qwen36-27b-c16-tensor-core-prefill-benchmark.json),
  which records the bounded C16 runtime and GDN extension, fixed-shape FP8 and
  NVFP4 Tensor Core dispatch, exhaustive correctness gates, mirrored same-binary
  C8/C16 samples, and the final C16 Nsight kernel profile.
- [`qwen36-27b-fp8-kv-pair-benchmark.json`](qwen36-27b-fp8-kv-pair-benchmark.json),
  which records the exact-shape M1 FP8 K/V projection pair, same-binary
  synthetic kernel gate, max-26-token Nsight comparison, mirrored single-load
  generation benchmark, exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-dual-benchmark.json`](qwen36-27b-nvfp4-down-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[5120,17408]` down-projection
  dual-iteration kernel, repeated same-binary gates, unchanged fallbacks,
  max-26-token Nsight comparison, mirrored single-load generation benchmark,
  exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json`](qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[17408,5120]` gate/up adjacent-lane
  XOR-dual kernel, repeated same-binary gates, unchanged fallbacks, max-26-token
  Nsight comparison, mirrored single-load generation benchmark, exact replay
  gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json`](qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[248320,5120]` lm-head adjacent-lane
  XOR-dual kernel, its same-binary production gate, unchanged fallbacks,
  max-26-token Nsight comparison, mirrored single-load generation benchmark,
  exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-data-reuse-benchmark.json`](qwen36-27b-nvfp4-data-reuse-benchmark.json),
  which records the follow-up aligned M1 `[5120,17408]` down adjacent-lane
  XOR-dual route and `[248320,5120]` lm-head CTA activation-staged route,
  same-binary identity/performance gates, B-C-C-B generation evidence against
  an independently rebuilt base commit, matched max-26-token profiles, exact
  replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json`](qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json),
  which records the follow-up aligned M1 `[17408,5120]` gate/up CTA
  activation-staged route, default capture-only exact/fallback dispatch
  identity gates, same-binary performance and bitwise gates, an independently
  rebuilt-base B-C-C-B generation benchmark, matched max-26-token profiles,
  exact replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-activation-staged-benchmark.json`](qwen36-27b-nvfp4-down-activation-staged-benchmark.json),
  which records the follow-up aligned M1 `[5120,17408]` down CTA
  activation-staged route, its preserved direct-XOR baseline and scalar
  fallbacks, same-binary performance and bitwise gates, a saved-base B-C-C-B
  generation benchmark, matched max-26-token profiles,
  exact replay, and diagnostic limitations.
- [`qwen36-27b-gdn-eight-row-benchmark.json`](qwen36-27b-gdn-eight-row-benchmark.json),
  which records the production eight-row lane-striped GDN update against its
  preserved four-row predecessor, M1/M2/M8/M16 bitwise state/output gates,
  mirrored same-binary measurements with current-profile call weights, matched
  max-26 profiles, compiler resources, exact replay, and diagnostic
  limitations.
- [`qwen36-27b-fp8-qkv-z-fusion-benchmark.json`](qwen36-27b-fp8-qkv-z-fusion-benchmark.json),
  which records the exact aligned FP8 M1 linear-attention QKV/Z two-phase
  fusion, five frozen actual-checkpoint/stress gates, compiler resources,
  matched max-26 profiles, detached-base B-C-C-B generation evidence, C1/C8/C16
  exact replay, ordered fallbacks, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json`](qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 dense-MLP gate/up/SiLU fusion,
  five frozen actual-checkpoint/same-bank gates, compiler resources, matched
  max-26 profiles, detached-base B-C-C-B generation evidence, C1/C8/C16 exact
  replay, ordered fallbacks, and diagnostic limitations.
- [`qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json`](qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 post-attention residual-add,
  centered-RMSNorm, gate/up, and SiLU fusion, its redundant per-CTA norm
  reduction, five frozen gates, dispatch/alias contracts, matched profiles,
  detached-base B-C-C-B evidence, exact replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json`](qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json),
  which records the reduction-only follow-up inside that post-attention
  fusion: shared-memory strides 128/64/32 followed by exact warp-zero shuffle
  strides 16/8/4/2/1, the preserved full-shared-tree same-binary predecessor,
  compiler resources, five direct actual-checkpoint/same-bank B-C-C-B gates,
  a matched max-26 profile, an independent B-C-C-B generation comparison,
  a bitwise-equal 5,905-line trace, full-shape finite/nonfinite closeout gates,
  and diagnostic limits.
- [`qwen36-27b-fp8-q-kv-fusion-benchmark.json`](qwen36-27b-fp8-q-kv-fusion-benchmark.json),
  which records the exact aligned FP8 M1 full-attention Q plus K/V fusion,
  five same-binary production-shaped gates, compiler resources, complete
  dispatch/alias contracts, matched profiles, detached-base B-C-C-B evidence,
  exact model gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json`](qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 dense-MLP down projection plus
  residual-add and centered-RMSNorm cooperative fusion, five production-shaped
  gates, finite and signed-nonfinite bitwise contracts, zero-node validation,
  matched profiles, detached-base B-C-C-B and full-trace equality, exact model
  gates, the zero-residency-margin AGX Orin constraint, and diagnostic limits.
- [`qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json`](qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json),
  which records the canonical M1 GDN plus 48x128 headwise plain-RMSNorm/SiLU
  gate implementation, its one-node/fallback/alias/resource contracts, the
  hardened 24-state same-binary gate, matched max-26 profiles, detached-base
  B-C-C-B and 5,905-line full-trace equality, C1/C8/C16 exact-model gates,
  release/sanitizer suites, and diagnostic limitations. Five earlier probes
  remain explicitly excluded as pre-production prototype inventory.
- [`qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json`](qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json),
  which records the exact shared-prefix/warp-zero reduction tail inside that
  production fused GDN kernel, the test-only full shared-tree predecessor,
  identical launch/resource and three-step bitwise contracts, the 28-to-24
  SASS `BAR` reduction, five hardened cold-state same-binary gates, the
  directly attributed max-26 target-kernel delta, exact model/trace suites,
  and the absence of an attributable detached-process end-to-end gain. Early
  prototypes and the test-only launcher are explicitly excluded from the
  production claim.
- [`qwen36-27b-prefill-decode-plan-split-benchmark.json`](qwen36-27b-prefill-decode-plan-split-benchmark.json),
  which records the host-control Prefill/Decode plan seam, explicit prefix,
  final-prompt, and feedback routing, the 44-case controller matrix, exact
  C1/C8/C16 model and arena gates, 5,905-line trace equality, identical ordered
  13,558-launch contracts, detached-base B-C-C-B regression evidence, and the
  explicit absence of new streams, buffers, kernels, or an attributed speedup.
- [`qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json`](qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json),
  which freezes the resolved SM87 projection route registry, its exact
  launch-contract and B-C-C-B equivalence gates, all 29 intended M1/M2/M8/M16
  direct-kernel cells, the P19 C1/C2/C8/C16 TTFT matrix, tokenizer-pinned
  P33/P65/P129/P513 prompts, and matched Nsight launch/time attribution. It
  records the unrelated historical M4 aggregate-threshold failure separately
  instead of treating that measurement gap as a production correctness error.
- [`qwen36-27b-c32-composite-prefill-benchmark.json`](qwen36-27b-c32-composite-prefill-benchmark.json),
  which records the ABI-0.2.0 C32 composite outer tile, exact real-model gates,
  P19 B-C-C-B and long-prompt C16/C32 timings, matched Nsight attribution, and
  the explicit single-stream M16-plus-tail projection and unlocked-clock limits.
- [`qwen36-27b-fp8-m32-production-benchmark.json`](qwen36-27b-fp8-m32-production-benchmark.json),
  which records the production fixed-M32 FP8 route, four-shape bitwise/guard/
  single-node gates, complete fallback validation, exact C1/C8/C16/C32 model
  replay, P33/C32 B-C-C-B TTFT evidence, and matched Nsight launch/time
  attribution against the frozen composite baseline.
- [`qwen36-27b-nvfp4-m32-production-benchmark.json`](qwen36-27b-nvfp4-m32-production-benchmark.json),
  which records the production fixed-M32 NVFP4 route, the K64/LD72 versus
  K128/LD136 same-cubin selection gate, two-shape bitwise/resource/graph and
  fallback contracts, exact C1/C8/C16/C32 model replay, the tokenizer-pinned
  P33/C32 B-C-C-B result, and matched baseline/candidate Nsight attribution.
- [`qwen36-27b-nvfp4-m32-scale-window-benchmark.json`](qwen36-27b-nvfp4-m32-scale-window-benchmark.json),
  which records the K256 coalesced block-scale window, same-cubin and NCU
  evidence, exact model replay, P33/C32 attribution, rejected swizzle/tile
  probes, and the later correction that its historical `build/orin-asan`
  directory was not actually sanitizer-instrumented.
- [`qwen36-27b-nvfp4-m32-factorized-lookup-benchmark.json`](qwen36-27b-nvfp4-m32-factorized-lookup-benchmark.json),
  which records the production factorized E2M1/E4M3 lookup, exhaustive bitwise
  gate, resource/SASS changes, same-cubin timing, exact-template NCU tradeoff,
  verified ASan/UBSan and model suites, and matched P33/C32 TTFT/Nsight
  attribution.
- [`qwen36-27b-nvfp4-m32-vector-store-benchmark.json`](qwen36-27b-nvfp4-m32-vector-store-benchmark.json),
  which records the production vectorized decoded-tile stores, direct and
  same-cubin bitwise gates, unchanged resources, replay-scoped shared-wavefront
  reduction, verified Release/sanitizer/model suites, and matched P33/C32
  TTFT/Nsight attribution.
- [`qwen36-27b-nvfp4-m32-gate-up-dual-stream-benchmark.json`](qwen36-27b-nvfp4-m32-gate-up-dual-stream-benchmark.json),
  which records the exact C32 NVFP4 MLP gate/up auxiliary-stream schedule,
  best-effort serial fallback and lifecycle contracts, production-dispatch
  bitwise/performance gates, real cross-stream overlap, P33 promotion rounds,
  P33/P65/P129/P513 scaling, and Release/sanitizer/model validation.
- [`qwen36-27b-nvfp4-m32-table-free-e2m1-benchmark.json`](qwen36-27b-nvfp4-m32-table-free-e2m1-benchmark.json),
  which records the exact-M32 table-free E2M1 PRMT promotion, the retained
  512-byte E4M3 scale table, exhaustive signed-zero/NaN validation, compiler
  resources and canonical SASS, repeated same-cubin single-kernel and absolute
  dual-stream pair gates, detached-base P33/P513 max1 and P513 max8 B-C-C-B,
  matched Nsys critical-path attribution, exact model/Release/sanitizer suites,
  and the platform-blocked Nsight Compute counter attempt.
- [`qwen36-27b-linear-ab-sidecar-rejection.json`](qwen36-27b-linear-ab-sidecar-rejection.json),
  which records the test-only exact-C32 production-dispatch A/B sidecar gate,
  three scheduling variants, four-output bitwise equality, marginal and
  regressing envelopes, and the decision to retain the serial runtime.
- [`qwen36-27b-nvfp4-m18-masked-m32-benchmark.json`](qwen36-27b-nvfp4-m18-masked-m32-benchmark.json),
  which records the exact aligned NVFP4 M18 masked-M32 Prefill route, exact-C18
  capacity and bitwise gates, four-cell synthetic kernel result, P19/C32
  B-C-C-B TTFT promotion, matched Nsight launch/time attribution, full model
  replay, host sanitizer coverage, and the platform-blocked device memcheck.
- [`qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json`](qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json),
  which records the production exact-capacity runtime-masked M17 and M19-M31
  routes for both NVFP4 MLP shapes, all 14 weighted microbenchmark gates,
  direct/pair/fallback and zero-node validation, the ten-prompt no-reversal
  B-C-C-B result, matched P18/P26/P64 Nsight attribution, exact model and host
  sanitizer suites, and unchanged fixed-M18/M32 production SASS. Its ten prompt
  texts, token IDs, hashes, and C32 schedules are frozen separately in the
  [`qwen36-27b-sm87-prefill-tail-prompts-v1.json` manifest](../../benchmarks/qwen36-27b-sm87-prefill-tail-prompts-v1.json).
  Prefill and Decode remain logically separate plans; only exact M32 gate/up
  has the existing layer-local dual-stream path, with no general double/triple
  buffering or Prefill/Decode overlap.
- [`qwen36-27b-nvfp4-m17-m32-gate-up-dual-stream-rejection.json`](qwen36-27b-nvfp4-m17-m32-gate-up-dual-stream-rejection.json),
  which records the rejected attempt to generalize that M32 gate/up schedule
  across M17-M32: all 32 synthetic correctness/cell gates, all 16 per-M gates,
  and the 1.08146x micro aggregate pass, but the twelve-prompt whole-model
  result reaches only 1.007059x against the required 1.01x. It freezes the
  baseline/candidate/withdrawn-patch identities, representative prompt and
  output contracts, the deliberate absence of a post-failure Nsight Systems
  profile for that candidate, and the decision to retain serial M17-M31/M18
  plus the existing M32-only auxiliary stream. It also separates a later
  Nsight Compute diagnostic of the retained production M17 kernel from the
  rejected candidate and records the evidence basis for a test-only 4 KiB
  raw-weight prefetch experiment.
- [`qwen36-27b-nvfp4-m17-m31-gate-up-raw-weight-cp-async-rejection.json`](qwen36-27b-nvfp4-m17-m31-gate-up-raw-weight-cp-async-rejection.json),
  which closes that 4 KiB single-slot experiment. The exact gate/up probe
  passes all 28 synthetic cells and 112 mirrored rounds at 1.06036x aggregate,
  while matched M17 Nsight Compute measures 1.06315x and lower long-scoreboard
  pressure without reducing five-CTA occupancy. All six screened down cells
  regress, however, and the formal twelve-prompt B-C-C-B result reaches only
  1.007587x against the required 1.01x. The temporary selector was fully
  withdrawn; production retains serial M17/M19-M31 and fixed-M18 gate/up plus
  the existing exact-M32 auxiliary-stream route. The next priority is a
  trace-backed Prefill/Decode phase and scheduler-overlap ceiling, not wider
  local raw-weight prefetch.
- [`qwen36-27b-prefill-decode-phase-trace-baseline.json`](qwen36-27b-prefill-decode-phase-trace-baseline.json),
  which records the default-off phase-aware timing/NVTX contract, exact
  P19/P33/P64/P513 range-to-kernel closure, CUDA-event topology calibration,
  phase-local hotspot split, registered-string validation, and the matched
  B-C-C-B proof that disabled instrumentation has no material regression. It
  also freezes the current boundary: Prefill and Decode are logically
  measurable but dependency-serialized, with no general double/triple buffer
  or multi-request scheduler.
- [`qwen36-27b-prefill-attention-score-warp-positions-benchmark.json`](qwen36-27b-prefill-attention-score-warp-positions-benchmark.json),
  which records the promoted exact Q24/KV4/D256 S>=65 warp-position attention
  score kernel: bitwise and S64/S65 dispatch gates, five mirrored same-binary
  micro rounds, matched S513 NCU, P513 Prefill Nsight and B-C-C-B promotion,
  direct P64-max26 long-context Decode evidence, P19 short-context fallback,
  complete test suites, memory-watermark caveats, and the unchanged public
  ABI, streams, buffers, softmax, value path, and serialized phase scheduler.
- [`qwen36-27b-nvfp4-m1-down-residual-norm-warp-tail-rejection.json`](qwen36-27b-nvfp4-m1-down-residual-norm-warp-tail-rejection.json),
  which records the rejected first post-trace Decode M1 experiment: full
  finite/nonfinite, invalid-call, graph, and resource contracts pass, but all
  five same-binary processes regress to 0.972242x-0.974540x against the 1.005x
  gate. Matched NCU explains why fewer `BAR`/shared-memory instructions do not
  win in this kernel, and the record freezes complete candidate removal with
  no retained test probe, runtime change, end-to-end run, or dispatch change.
- [`qwen36-27b-fp8-m1-qkv-z-activation-staged-rejection.json`](qwen36-27b-fp8-m1-qkv-z-activation-staged-rejection.json),
  which records the rejected test-only Decode M1 FP8 QKV/Z activation-staging
  experiment: actual layer-0 checkpoint payload identity, exhaustive bitwise/
  replay/canary/input and resource gates, all seven actual-checkpoint and
  same-bank cap cells, and the frozen 0.913426x/0.905513x failure against the
  1.02x/1.00x gates. It records the fixed 10-KiB-per-CTA staging mechanism,
  exact log and restored-artifact identities, complete candidate removal, the
  production-only NCU locality bound, and the stop-loss decision not to run
  candidate NCU or end-to-end evaluation.
- [`qwen36-27b-nvfp4-m1-factorized-rejection.json`](qwen36-27b-nvfp4-m1-factorized-rejection.json),
  which records the rejected test-only Decode M1 BF16 pair/scale factorization:
  exhaustive and full-path bitwise gates, compiler resources, unchanged
  production SASS, two consistently regressing full-shape synthetic cells, the
  static-instruction interpretation, removal of all candidate code, and the
  explicit absence of a production-dispatch change.
- [`qwen36-27b-nvfp4-m1-full-product-table-rejection.json`](qwen36-27b-nvfp4-m1-full-product-table-rejection.json),
  which records the rejected test-only Decode M1 canonical full-product-table
  candidate for the fused residual/norm/gate/up/SiLU hotspot. Corrected
  synthetic checkpoint-like and same-bank screens reach only 0.656836x and
  0.615217x despite bitwise full-path, replay, canary, graph, and resource
  gates. The record freezes candidate/clean binary identities, static-SASS
  attribution and its no-NCU limit, host/SM87 NaN-canonicalization boundaries,
  complete candidate removal, and unchanged production dispatch.
- [`qwen36-27b-gdn-m16-shared-resident-bf16-state-rejection.json`](qwen36-27b-gdn-m16-shared-resident-bf16-state-rejection.json),
  which records the rejected test-only Prefill GDN M16 row4/row8
  shared-resident BF16 state candidates. Both pass exact M1/M2/M8/M16
  in-place/disjoint output and state, poison replay, C16 versus C8+C8, Graph,
  canary, input-preservation, and resource gates, but the hardened 24-bank
  screen reaches only 0.627300x and 0.752132x against production row8. The
  record freezes the initial same-TU dynamic-shared padding hazard, final
  test-only-TU isolation, complete candidate removal, restored production
  source/resource/SASS identity, exact artifact hashes, and the stop-loss
  decision not to run candidate NCU, Nsys, or end-to-end evaluation.
- [`qwen36-27b-fp8-m1-qkv-z-reduction-scratch-ping-pong-benchmark.json`](qwen36-27b-fp8-m1-qkv-z-reduction-scratch-ping-pong-benchmark.json),
  which records the promoted two-slot CTA-local FP8 M1 QKV/Z reduction-scratch
  pipeline. It freezes exhaustive 256-code/four-byte-position coverage,
  four-cap race-signature/replay and Graph contracts, exact checkpoint payload
  identity, resources and per-function SASS, the 142-non-target-function
  production-isolation proof, frozen 1.02407x/1.00697x micro gates, and exact
  P19/P64/P513 B-C-C-B evidence with all 12 log hashes. The record retains the
  external P19 memory-watermark warning and unavailable racecheck as explicit
  limitations, and makes clear that this intra-kernel scratch ping-pong is not
  general double/triple buffering or Prefill/Decode overlap.
- [`qwen36-27b-fp8-m1-q-kv-reduction-scratch-ping-pong-benchmark.json`](qwen36-27b-fp8-m1-q-kv-reduction-scratch-ping-pong-benchmark.json),
  which records the promoted two-slot CTA-local FP8 M1 full-attention Q+K/V
  reduction-scratch pipeline. It freezes exhaustive 256-code/four-byte-position
  coverage across Q, K, and V, ordered Q-row-quad and Q-to-K/V race/replay
  contracts, exact checkpoint payloads, resources, three explicitly distinct
  SASS canonicalizations, five independent actual/stress micro processes, and
  exact P19/P64/P513 B-C-C-B evidence with all 12 log hashes. It also records
  the symmetric-engine device-code isolation proof, zero persistent-drop flags,
  the 0.042171% worst whole-model stage regression against the 0.5% limit, and
  the boundary between this intra-kernel scratch mechanism and system buffering.
- [`qwen36-27b-fp8-m32-dual-resident-a-benchmark.json`](qwen36-27b-fp8-m32-dual-resident-a-benchmark.json),
  which records the promoted exact-shape FP8 M32 dual-resident-A WMMA route.
  It freezes full 256-code/four-byte-position bitwise and NaN coverage, resource
  and Graph role gates, the preserved single-resident predecessor SASS, five
  independent same-binary micro processes, and symmetric P33/P513 B-C-C-B
  evidence. The record also proves that both end-to-end runners contain the
  same 149 CUDA functions with identical encodings, while P33 and P513 TTFT
  improve by 2.94% and 3.76% respectively without changing the serialized
  Prefill/Decode scheduler or adding system double/triple buffering.
- [`qwen36-27b-fp8-m32-codebook-swizzle-rejection.json`](qwen36-27b-fp8-m32-codebook-swizzle-rejection.json),
  which records the rejected test-only U16 codebook swizzles for the four exact
  FP8 M32 dual-resident-A shapes. It freezes exhaustive byte-bijection,
  256-code/four-position, signed-NaN, replay, guard, resource, Graph, mode-0
  all-word SASS, four pinned real-payload, and alternating six-round evidence.
  The record separates the non-representative 1.55792x/1.52486x synthetic
  stress result from the failed 1.00308x/0.988108x actual weighted gates and
  includes the full 5,570,560-instruction static bank-wavefront audit, its
  non-NCU limits, and the complete source/test removal. Its then-next Decode
  FP8 M1 `[5120,6144]` single-body/no-tail-barrier screen is resolved by the
  separate rejection record below rather than retroactively claimed as part of
  the M32 evidence.
- [`qwen36-27b-attention-values-exact-benchmark.json`](qwen36-27b-attention-values-exact-benchmark.json),
  which records the production exact Q24/KV4/D256 attention-values route, its
  40-to-26 register and 376-to-112 `cuobjdump` function-block reductions,
  exact output/replay/guard/Graph/resource and full-suite gates, five-process
  hot-cell (1.30170x-1.50671x), S65..S513 chain (1.39585x-1.39948x), and rotating
  16-bank cold-S513 (1.79801x-1.80948x) results, neutral P33/max1 guard, and
  P513 reductions of 1.213% for max1 TTFT and 0.987% for max8
  subsequent-token latency. It introduces no runtime double/triple buffering:
  Prefill and Decode remain logically separate plans but batch-one execution
  is dependency-serialized; the next step is a refreshed profile of the next
  hotspot.
- [`qwen36-27b-bf16-m16-projection-fused-benchmark.json`](qwen36-27b-bf16-m16-projection-fused-benchmark.json),
  which records the production exact M16/N48/K5120 linear-attention BF16 A/B
  pair. It freezes the rejected row-resident comparison, selected 32-chain
  projection-fused implementation, bitwise/special-value/replay/canary and
  production Graph gates, final resources and SASS, five independent hot and
  rotating-cold processes, P33/P513 max1 and P513 max8 B-C-C-B evidence, and
  the before/after P513 profile. The exact target falls from 229.592608 to
  66.677600 ms (3.44332x); P33/P513 max1 TTFT falls by 2.304%/2.914%, while
  Decode remains neutral because M1 keeps the generic route. The record also
  retains one baseline-only global-memory warning and its clean audit, and
  makes clear that no system double/triple buffer or Prefill/Decode overlap
  was introduced.
- [`qwen36-27b-gdn-m16-register-resident-bf16-state-benchmark.json`](qwen36-27b-gdn-m16-register-resident-bf16-state-benchmark.json),
  which records the promoted exact-C16 GDN register-state route. It freezes the
  packed per-thread BF16 state and 45-MiB traffic-removal mechanism, exact
  output/state/replay/guard/Graph/selector gates, final SASS and resources,
  five fixed-frequency same-binary processes at 1.21660x-1.21796x, strict P33
  and P513/C32 B-C-C-B gains, and matched 96-launch Nsys attribution. It also
  records and excludes the stale-test-object exact-versus-exact runs, proves
  the orphan candidate object was not linked, and preserves the boundary
  between this intra-kernel optimization and system buffering or
  Prefill/Decode overlap.
- [`qwen36-27b-post-gdn-m16-phase-profile.json`](qwen36-27b-post-gdn-m16-phase-profile.json),
  which records the fixed-frequency production phase profile after that
  promotion. It freezes the P513/C32 Prefill and P19/C32/max26 Decode commands,
  binary/report identities, the retained Decode log, observed exact oracles,
  and NVTX-to-kernel closure; corrects the M32 gate/up raw sum to its two-stream
  marginal exposure; and ranks the resulting Prefill and Decode hotspots. It
  also records the current scheduling boundary: logical phase plans exist, but
  batch-one has no general double/triple buffer; Decode has zero overlap and
  only 0.733879% kernel-span idle, so phase-local work remains ahead of
  multi-request scheduler work.
- [`qwen36-27b-fp8-m1-attention-o-proj-no-tail-rejection.json`](qwen36-27b-fp8-m1-attention-o-proj-no-tail-rejection.json),
  which records the rejected first bounded follow-up to that Decode ranking.
  The exact `[5120,6144]` single-body candidate passes exhaustive E4M3FN,
  replay, NaN, guard, input, invalid Graph, topology, resource, and compiled
  mechanism gates; complete SASS removes 192 instructions and the third static
  barrier while preserving 64-register/four-CTA residency, 128 FFMA, and all
  key global/shared/output-store counts. All six fixed-clock actual layer-0
  payload rounds regress, however, producing a 0.994914x paired median and
  0.994295x minimum round against required 1.01x/1.00x gates. The record freezes
  the actual-first stop-loss, two older same-family failures, 142-function
  production-isolation manifest, retained candidate artifacts, complete
  source/test removal, forced clean rollback, and the deliberate absence of
  synthetic, five-process, P19, Nsys, or NCU work.
- [`qwen36-27b-nvfp4-m32-down-residual-epilogue-fusion-rejection.json`](qwen36-27b-nvfp4-m32-down-residual-epilogue-fusion-rejection.json),
  which records the rejected exact-M32 Prefill down/residual epilogue fusion.
  The test-only kernel preserves five-CTA residency and passes its finite,
  replay, guard, Graph, and resource checks, but reaches only 1.00050x across
  six fixed-clock actual-payload rounds, below the 1.005x technical and 1.02x
  promotion gates. It also records the independent audit finding that the
  candidate and test reverse the real residual-add operand order: finite
  performance evidence remains valid, while the test's 4/4 NaN bit match is
  explicitly not treated as a full runtime semantics proof. The candidate was
  removed before stress, independent-repeat, Nsys, NCU, or end-to-end work.
- [`qwen36-27b-m32-residual-centered-rmsnorm-fusion-benchmark.json`](qwen36-27b-m32-residual-centered-rmsnorm-fusion-benchmark.json),
  which records the promoted exact-M32 residual-add plus centered-RMSNorm
  boundary fusion. It freezes the rounded-BF16 bitwise and NaN-order contract,
  alias/guard/replay/Graph and selector gates, actual checkpoint norm fixtures,
  final resources and SASS, P33/P64/P513 model oracles, exact launch-topology
  reduction, and two independent P513 B-C-C-B groups. Hot repeated chains reach
  about 1.497x, production raw target kernels reach 1.0703x, and the measured
  whole P513 prefix improves by 0.278%; the record explicitly confines each
  claim and retains the cross-capture Nsight drift warning.
- [`qwen36-27b-m32-residual-rms-staged-register-cache-benchmark.json`](qwen36-27b-m32-residual-rms-staged-register-cache-benchmark.json),
  which records the follow-up C1/C2/C3/C4 screen and promotion of the
  512-thread shared-staging plus low-lane register-cache C4 kernel. It preserves
  the exact BF16 boundary and reduction tree while raising producer parallelism
  and removing the output-phase residual reread. The 2,048 production launches
  fall from a bracketed 62.403 to 37.396 ms (1.6687x), and two unprofiled P513
  B-C-C-B pairs reduce the whole prefix by 27.559 ms, or 0.579%, with neutral
  finish-prefill. The record separates hot-chain, production-kernel, and
  end-to-end claims and explains why the other three variants were not selected.
- [`qwen36-27b-post-c4-prefill-phase-profile.json`](qwen36-27b-post-c4-prefill-phase-profile.json),
  which refreshes fixed-frequency P513/C32 Prefill attribution after C4. It
  closes all 40,176 prefix kernels, separates the 391.939 ms dual-stream
  overlap from raw gate/up duration, places the promoted residual/RMSNorm
  chain at only 0.803% of kernel union, and shows that exact-M32 NVFP4 gate/up
  plus down retain 49.001% marginal exposure. It also freezes the next three
  test-only mechanism gates and keeps general scheduler buffering behind the
  much larger phase-local opportunities.
- [`qwen36-27b-nvfp4-m32-table-free-e4m3-rejection.json`](qwen36-27b-nvfp4-m32-table-free-e4m3-rejection.json),
  which records the first bounded post-C4 NVFP4 screen. The exact integer
  E4M3FN-to-BF16 constructor removes 512 B of shared state, two static LDS,
  and one barrier while preserving five-CTA residency and all bitwise/Graph
  contracts, but every isolated cell regresses to 0.94947x-0.95670x. Its
  dual-stream pair reaches only 1.01952x and includes a checkpoint-like round
  reversal, so the candidate and all test hooks are removed before wider
  profiling or end-to-end work.
- [`qwen36-27b-nvfp4-m32-gate-up-raw-weight-cp-async-benchmark.json`](qwen36-27b-nvfp4-m32-gate-up-raw-weight-cp-async-benchmark.json),
  which records the promoted exact-M32 gate/up single-slot raw-weight
  `cp.async` pipeline. It preserves five-CTA residency and bit-exact output,
  improves the isolated kernel by at least 1.0799x across five processes, and
  improves the production dual-stream pair by at least 1.0732x. Matched
  P513/C32 profiling reduces gate/up marginal exposure by 126.236 ms
  (1.1001x) and whole-prefix kernel union by 123.220 ms (1.0270x), while the
  source- and selector-unchanged down kernel differs by 0.0140%. The 4 KiB
  CTA-local staging slot is explicitly not a runtime double buffer or
  Prefill/Decode scheduler.
- [`qwen36-27b-nvfp4-m32-down-n160-rejection.json`](qwen36-27b-nvfp4-m32-down-n160-rejection.json),
  which records the rejected exact-M32 down N160/320-thread balanced-grid
  screen. It reduces the grid from 40 to 32 CTAs and passes bitwise, replay,
  guard, Graph, resource, and compiled-code gates, but all six fixed-clock
  actual-checkpoint rounds regress to 0.84983x-0.85050x. The stop-loss omits
  same-bank, Nsys, and end-to-end work, removes the candidate and test hooks,
  and leaves the promoted gate/up `cp.async` source unchanged.
- [`qwen36-27b-fp8-m32-half-tile-raw-weight-cp-async-rejection.json`](qwen36-27b-fp8-m32-half-tile-raw-weight-cp-async-rejection.json),
  which records the bounded exact-M32 FP8 `[10240,5120]` half-tile raw-weight
  `cp.async` screen. Its 4 KiB single slot preserves five-CTA residency,
  bit-exact output, final-stage draining, and every-round non-regression, but
  the retained actual-checkpoint paired median is only 1.02225x against the
  frozen 1.03x gate. Stop-loss therefore removes the candidate and skips
  stress, profiling, end-to-end work, and five-process replication while
  retaining a complete recovery patch for possible independent shape study.
- [`qwen36-27b-fp8-m32-k6144-half-tile-raw-weight-cp-async-rejection.json`](qwen36-27b-fp8-m32-k6144-half-tile-raw-weight-cp-async-rejection.json),
  which records that independent exact-M32 FP8 `[5120,6144]` K6144 screen.
  The test-only 4 KiB half-tile slot passes resource, SASS, exhaustive-code,
  full-K, replay, guard, input-preservation, Graph, and actual-weight/scalar
  plus deterministic-activation correctness gates, but all six fixed-clock rounds regress to
  0.947361x-0.948105x and the paired median is 0.947746x. The actual-first
  stop-loss removes the candidate and skips stress, NCU/Nsys, end-to-end, and
  five-process work, closing this longer-K shape without changing production.
- [`qwen36-27b-nvfp4-m32-asymmetric-table-free-e4m3-bgu-rejection.json`](qwen36-27b-nvfp4-m32-asymmetric-table-free-e4m3-bgu-rejection.json),
  which closes the exact-M32 NVFP4 one-sided table-free-E4M3 gate/up screen.
  The corrected packed-x2 candidate passes 65,536-pair exhaustive, static,
  full B/G/U, replay, guard, input-preservation, Graph, and actual-payload
  correctness gates, but gate/main and up/aux regress in all six fixed-clock
  rounds to 0.957097x and 0.956026x paired medians. The actual-first stop-loss
  removes the candidate and skips stress, profiling, end-to-end, and
  replication without changing production.
- [`qwen36-27b-gdn-m32-register-resident-bf16-state-rejection.json`](qwen36-27b-gdn-m32-register-resident-bf16-state-rejection.json),
  which closes the test-only exact-C32 extension of the production M16
  register-state lifetime. Output, final state, replay, input, Graph, and
  resource gates pass, and every fixed-clock round improves, but the 1.02474x
  paired median misses the frozen 1.03x early gate. Stop-loss removes the
  candidate before runner/model/profiler work and keeps production unchanged;
  its required fresh current-HEAD phase profile is recorded immediately below.
- [`qwen36-27b-current-head-phase-profile.json`](qwen36-27b-current-head-phase-profile.json),
  which refreshes fixed-clock P513/C32 Prefill and P19/C32/max26 Decode
  attribution at `9bddbda`. It freezes the binary, exact output oracles,
  reports, SQLite exports, analysis script, NVTX closure, and interval-union
  method. The current Prefill gate/up and down rows account for 48.238973% of
  prefix critical-path exposure; Decode remains single-stream with 0.584882%
  span idle and its top three rows account for 75.591051% of kernel time. The
  result keeps phase-local, NCU-guided kernel work ahead of general batch-one
  double/triple buffering and forbids restoring already closed mechanisms.
- [`qwen36-27b-fp8-m32-packed-decode-static-rejection.json`](qwen36-27b-fp8-m32-packed-decode-static-rejection.json),
  which closes the CPU/compile/SASS-only exact-M32 FP8 packed-decode screen.
  The exact four-code BF16-bit constructor removes all 32 codebook `LDS.U16`,
  512 B of shared state, and one barrier while retaining four `STS.128` decoded-B
  stores and five-CTA inferred residency, but grows the function from 632 to
  760 instructions and the K64 body from 200 to 379 instructions. Both exceed
  the frozen static stop, so no GPU program runs and production remains
  unchanged. This M32 WMMA shared-B constructor is distinct from ad22fdd's
  earlier M1 packed-load/FP32-decode GEMV. The next bounded priority at that
  snapshot was the phase-local exact-M32 FP8 QKV/Z dual-N128-tile fusion,
  now closed below.
- [`qwen36-27b-fp8-m32-qkv-z-dual-n128-fusion-rejection.json`](qwen36-27b-fp8-m32-qkv-z-dual-n128-fusion-rejection.json),
  which closes that phase-local exact-M32 QKV/Z screen. One 512-thread CTA
  maps two independent eight-warp N128 groups, halves the baseline 128 CTAs
  to 64, and shares the decoded codebook and `[32,5120]` activation traffic.
  Static, exhaustive-codebook, full-K, replay, guard, input-preservation,
  Graph, and actual-payload correctness gates all pass. The fixed-clock real
  QKV/Z payload result nevertheless regresses in all six `B-C-C-B` rounds:
  paired median 0.908776x, range 0.908670x-0.909103x. Stop-loss skips stress,
  NCU/Nsys, end-to-end work, and five-process replication, restores the two
  tracked source/test files exactly to `ce98625`, and retains the complete
  recovery patch.
- [`qwen36-27b-nvfp4-m1-gate-up-balanced-tail-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-balanced-tail-rejection.json),
  which closes the first Decode follow-up to the current-HEAD profile. The
  exact-M1 test-only candidate preserves the first 16,384 rows and redistributes
  only the final 1,024 rows from 32 row quads to 64 row pairs. Resources remain
  64 registers, 11,328 B shared, zero local, and four active CTAs/SM; actual,
  stress, and signed-nonfinite outputs are bit-exact with intact guards and
  preserved inputs. Nevertheless, all five fixed-clock `B-C-C-B` rounds regress:
  paired medians are 0.951273x actual and 0.955264x stress. Stop-loss removes
  the candidate, restores both tracked files exactly, and skips production,
  model-oracle, end-to-end, Nsys, NCU, and replication work.
- [`qwen36-27b-nvfp4-m1-gate-up-k256-packed-weight-pipeline-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-k256-packed-weight-pipeline-rejection.json),
  which closes the next exact-M1 Decode gate/up screen. A canonical-layout
  K256 cooperative loader adds one 4-KiB CTA raw-weight slot and forms a
  logical register/shared double buffer with `cp.async`; resources remain 64
  registers, zero local memory, and four active CTAs/SM. Static SASS,
  alignment, actual/stress bitwise, canary, and input-preservation gates all
  pass, but every fixed-clock round regresses. Paired medians are 0.932869x
  actual and 0.928905x stress. Stop-loss restores source/test exactly and
  retains the observed out-of-line `CALL/WARPSYNC/RET` synchronization as one
  narrower code-generation follow-up before an interleaved Decode sidecar.
- [`qwen36-27b-nvfp4-m1-gate-up-k256-inline-warp-barrier-static-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-k256-inline-warp-barrier-static-rejection.json),
  which closes that narrower code-generation follow-up. Volatile inline PTX
  `bar.warp.sync 0xffffffff` preserves the candidate's 64-register/15,424-byte
  resource envelope but compiles to a Function byte-identical to the measured
  source-`__syncwarp` version, including the same four `CALL.REL` sites and
  `WARPSYNC R30; RET` helper. Static stop-loss skips redundant correctness and
  timing, restores source/test exactly, and moves the next bounded screen to a
  single-layer AoSoA4 row-quad-interleaved Decode weight sidecar.
- [`qwen36-27b-nvfp4-m1-gate-up-aosoa4-sidecar-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-aosoa4-sidecar-rejection.json),
  which closes that materially different layout screen. The single-layer
  85-MiB sidecar combines four row words into each `uint4`, generates two
  `LDG.E.128` weight loads and zero 32-bit weight loads in the unrolled phase,
  and retains the production 64-register/11,328-byte/four-CTA resource
  envelope. Actual and stress outputs are bit-exact, but actual regresses in
  all five rounds to a 0.994987x paired median; stress reaches only 1.00353x.
  Stop-loss skips the 5.3125-GiB full-model sidecar and restores source/test
  before re-ranking the remaining Decode hotspots.
- [`qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-selection.json`](qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-selection.json),
  which selects, at the test-only formal-microbenchmark stage, the exact FP8
  M1 `[5120,6144]` output-projection AoSoA4 plus byte-preswizzled sidecar. The
  pinned real layer-0 payload clears a 1.05155x paired-median gate with all
  five rounds non-regressing, same-bank stress clears 1.01465x, and bitwise,
  replay, guard, input/sidecar-preservation, and resource gates pass. The
  record also freezes the 30-MiB-per-layer/1.875-GiB-for-64-layers memory
  obligation and a projection-only 110.951-to-110.370 ms/token estimate. It
  does not claim production dispatch, loader integration, or an achieved
  end-to-end latency or throughput result.
- [`qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-production-benchmark.json`](qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-production-benchmark.json),
  which records the subsequent persistent-sidecar production promotion. It
  separates the timed candidate from the final exception-path-fix binary,
  pins the GPU pack oracle, seven pack-invalid and eleven GEMV-invalid cases,
  the exact final-model replay, and the P19/C32/max26 `B1-C1-C2-B2` result.
  The mirrored hot Decode value is
  109.7585 ms/token, or 9.1109 token/s, while the record also exposes the
  non-amortized 1.875-GiB residency and roughly 467-ms cold pack cost.
- [`qwen36-27b-nvfp4-m1-down-norm-cta-prune-rejection.json`](qwen36-27b-nvfp4-m1-down-norm-cta-prune-rejection.json),
  which closes the synthetic exact-M1 fused-down post-grid-sync CTA-prune
  screen. Resources, distinct one-node `64x256` Graphs, both distributions,
  all three BF16 outputs, guards, and inputs pass, and every timing round
  non-regresses. Paired medians reach only 1.00135x checkpoint-like and
  1.00107x same-bank stress against the frozen 1.002x gate, so stop-loss skips
  actual checkpoint, NCU, model-oracle, and end-to-end work.
- [`qwen36-27b-post-fp8-output-sidecar-decode-phase-profile.json`](qwen36-27b-post-fp8-output-sidecar-decode-phase-profile.json),
  which records the required fixed-frequency Decode-only refresh at `aa7312b`
  using the final `77931b8` Release binary. All 25 Decode ranges close over
  10,925 unique kernels on one stream, raw equals union at 2,745.814816 ms,
  and the top three gate/up, QKV/Z, and down rows retain 75.878904% of kernel
  time. The production output-sidecar row averages 0.181293 ms per launch and
  contributes 11.602725 ms per Decode step. Its comparison with the earlier
  `9bddbda` trace is explicitly directional; the formal performance anchor
  remains the unprofiled 109.7585 ms/token and 9.1109 token/s `B1-C1-C2-B2`
  result.
- [`qwen36-27b-fp8-m1-qkv-z-aosoa4-preswizzled-sidecar-rejection.json`](qwen36-27b-fp8-m1-qkv-z-aosoa4-preswizzled-sidecar-rejection.json),
  which closes the test-only exact-M1 linear-attention QKV/Z
  AoSoA4/preswizzled sidecar screen. The hash-pinned layer-0 payload, stress,
  replay, guard, input-preservation, invalid-call, resource, Graph, and static
  production-identity gates pass. Actual payload improves in all five rounds
  but reaches only 1.00562x against 1.03x, while stress regresses in all five
  rounds to 0.984394x. Stop-loss skips the 3.75-GiB 48-layer sidecar and all
  production, model, profiler, and replication work, removes the candidate,
  and retains the restored default-test pass.
- [`qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-selection.json`](qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-selection.json),
  which selects the test-only exact-M1 residual/norm/gate/up/SiLU `32x512`
  CTA grouping for production integration. It preserves the same 512
  projection warps and 32 resident warps/SM as production's `64x256` route
  while halving repeated per-CTA residual/RMSNorm setup. Across three
  independent same-binary processes, the pinned actual-checkpoint and
  same-bank-stress paired medians reach 1.02410x and 1.02295x, with all 30
  rounds improving and finite/replay, signed Inf/NaN, guard, input, resource,
  Graph, invalid-call, and production-static-identity gates passing. The
  0.926016-ms/token 64-layer value (0.833856–0.927104-ms/token process range)
  is an arithmetic projection; production remains at 109.7585 ms/token and
  9.1109 token/s pending final production integration, model-oracle,
  end-to-end, and Nsys evidence.
- [`qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-production-benchmark.json`](qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-production-benchmark.json),
  which records the subsequent zero-additional-sidecar production promotion.
  The final
  exact checkpoint/stress gates reach 1.02388x/1.02302x with every round
  improving; finite, replay, signed Inf/NaN, resource, Graph, invalid-call,
  SASS, Release, CTest, and full-model oracle checks pass. Fixed-frequency
  P19/C32/max26 `B1-C1-C2-B2` moves the mirrored hot Decode median from
  109.817 to 109.056 ms/token with both pairs improving, establishing
  9.1696 token/s and a remaining 9.056-ms/token stage gap.
- [`qwen36-27b-post-gate-up-cta-coarsen-decode-phase-profile.json`](qwen36-27b-post-gate-up-cta-coarsen-decode-phase-profile.json),
  which records the fresh production Decode closure. Across all 25 ranges,
  there are exactly 10,925 distinct kernel rows (437 per range) on one stream,
  with raw equal to union and zero overlap. The promoted `32x512` gate/up
  kernel appears 1,600 times and falls
  directionally from 39.675852 to 38.772791 ms/step; the old `64x256` kernel is
  absent. Generation/leaf closure and the retained FP8 output-sidecar route
  both pass, while the unprofiled 109.056-ms result remains authoritative.
- [`qwen36-27b-nvfp4-m1-down-cta-coarsen-selection.json`](qwen36-27b-nvfp4-m1-down-cta-coarsen-selection.json),
  which selects the test-only exact-M1 down/residual/centered-RMSNorm
  `32x512` CTA grouping while leaving production's `64x256` route unchanged.
  Three fixed-frequency same-binary processes reach actual-checkpoint paired
  medians of 1.01181x/1.01842x/1.01176x and stress medians of
  1.01218x/1.01876x/1.01453x; all 30 rounds improve. Three-output bitwise and
  replay, guards, inputs, split residual-only/norm-only Inf/NaN, resource,
  Graph, invalid-call, and production-SASS-identity gates pass. The isolated
  0.246464-ms/token 64-layer value is arithmetic only; the formal anchor
  remains 109.056 ms/token and 9.1696 token/s pending production integration.
- [`qwen36-27b-nvfp4-m1-down-cta-coarsen-production-benchmark.json`](qwen36-27b-nvfp4-m1-down-cta-coarsen-production-benchmark.json),
  which records the final `0372d41` production promotion at tree `aa341b9`.
  Three independent same-binary processes report actual-checkpoint paired
  medians of 1.01511x/1.01184x/1.01419x and stress medians of
  1.01975x/1.01416x/1.01544x; all 30 rounds improve. Default validation, the
  four focused CTests, the pinned full-model oracle, finite/replay, split
  nonfinite, guard, input, Graph, invalid-call, resource, and SASS gates pass.
  The complete fixed-clock `B1-C1-C2-B2` sequence moves its same-run mirrored
  median from 109.3735 to 109.1155 ms/token, a 0.258-ms reduction, and both
  pairs improve with canonical SHA-256 `f66b837e...` equal four ways. An
  earlier baseline inference completed but produced truncated stdout; it is
  excluded and the formal sequence restarts at B1. Because 109.1155 is still
  0.0595 ms slower than the older 109.056-ms formal result, the conservative
  release anchor remains **109.056 ms/token and 9.1696 token/s**. Applying the
  same-run delta to that anchor yields 108.798 ms/token and 9.191345 token/s
  only as a non-measured planning normalization, never as an achieved result.
- [`qwen36-27b-post-down-cta-coarsen-decode-phase-profile.json`](qwen36-27b-post-down-cta-coarsen-decode-phase-profile.json),
  which closes the fresh production trace over 25 Decode ranges and 10,925
  distinct kernel rows, exactly 437 per range. Raw equals union at
  2,725.072960 ms on one stream, overlap is zero, and the associated span has
  only 0.572941% idle. The generation range closes exactly over prefix,
  finish, and Decode leaves at 12,997 rows and 3,171.275488 ms. Production
  down `32x512` appears 1,600 times, exactly 64 per Decode step, at
  20.813704 ms/step and 19.094630% of raw time; every named or topological
  `64x256` predecessor count is zero. Against the prior profile the Down row
  is directionally 0.079035 ms/step lower, while total Decode raw time is
  effectively flat (+0.002021 ms/step) because separate-run drift moves other
  rows. The trace explicitly remains serial, single-stream, and without a
  system double/triple buffer or independent Prefill/Decode executors.
- [`qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-1024-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-1024-rejection.json),
  which closes the test-only exact-M1 fused gate/up `16x1024` balanced-tail
  screen. Each physical CTA pairs logical production blocks `b` and `b+16`,
  preserving all 512 projection warps and assigning one tail-bearing logical
  block to every CTA while halving repeated residual/RMSNorm setup. Resource,
  exact-capacity, distinct-Graph, nine-invalid, finite/replay, guard, input,
  signed Inf/NaN, and SASS gates pass. However, all ten fixed-clock paired
  rounds regress: actual-checkpoint and stress medians are 0.994805x and
  0.996853x. Stop-loss ends after one process, removes source/test hooks, and
  leaves production and the formal 109.056-ms/token / 9.169600939-token/s
  anchor unchanged. A naive contiguous `16x1024` mapping was not tested and
  is outside this rejection.
- [`qwen36-27b-fp8-m1-qkv-z-ping-pong-grid-cap-1024-rejection.json`](qwen36-27b-fp8-m1-qkv-z-ping-pong-grid-cap-1024-rejection.json),
  which closes the test-only production-ping-pong QKV grid-cap screen. The
  exact same kernel and SASS change only from production QKV1536/Z768 to
  QKV1024/Z768; exhaustive finite/code, replay/race-signature, NaN, source,
  invalid, null-query, resource, and frozen actual/stress correctness gates
  pass. The actual-checkpoint median reaches 1.0047x against 1.005x, while
  stress reaches 0.998787x and regresses in all five rounds. First-process
  stop-loss restores the test policy and leaves production and the formal
  109.056-ms/token / 9.169600939-token/s anchor unchanged.
- [`qwen36-27b-nvfp4-m1-gate-up-dead-up-shared-pair-selection.json`](qwen36-27b-nvfp4-m1-gate-up-dead-up-shared-pair-selection.json),
  which selects a test-only, runner-scoped gate/up mechanism for explicit
  production integration. Production and candidate both remain `32x512`; the
  candidate keeps the independently BF16-rounded gate/up pair in two
  CTA-local `BF16[576]` arrays, publishes only final `SiLU(gate)*up`, and
  leaves the dead up buffer untouched. It uses 64 registers, 13,632 B shared,
  zero local memory, and two active CTAs/SM. Three clean frozen-binary
  processes reach cross-process actual/stress paired medians of 1.01034x and
  1.00932x, with all 30 rounds improving. Published-output bitwise/replay,
  signed Inf/NaN, guard, input, Graph-capture, invalid-call, resource, and
  production-SASS-identity gates pass. One complete-suite outlier attempt is
  retained and excluded transparently; no dead-up regression was omitted.
  The 0.396352-ms/token 64-layer value is arithmetic only. Production and the
  formal 109.056-ms/token / 9.169600939-token/s anchor remain unchanged until
  an explicit runner-only API, full-model oracle, end-to-end benchmark, and
  fresh Nsys closure pass; the generic double-output API must continue to
  publish independently rounded up values.
- [`qwen36-27b-nvfp4-m1-gate-up-dead-up-production-benchmark.json`](qwen36-27b-nvfp4-m1-gate-up-dead-up-production-benchmark.json),
  which records the explicit runner-only production promotion at `2dbd832`
  and the complete 54-case invalid-contract hardening at `798582c`. The
  generic double-output APIs and all fallbacks remain intact; only the exact
  runner-dead boundary elides up publication. Real-buffer CUDA Graph replay,
  finite/direct-nonfinite correctness, resource/SASS identity, full Release
  CTest, and pinned C1/C8/C16/C32 model oracles pass. Fixed-clock
  P19/C32/max26 `B1-C1-C2-B2` moves the mirrored hot Decode median from
  109.0535 to **108.6695 ms/token** with both pairs improving, establishing
  **9.202214053 token/s** and an 8.6695-ms/token remaining stage gap.
- [`qwen36-27b-post-gate-up-dead-up-decode-phase-profile.json`](qwen36-27b-post-gate-up-dead-up-decode-phase-profile.json),
  which closes the promoted production route over 25 Decode ranges and 10,925
  distinct kernel rows, exactly 437 per range. The new symbol appears 1,600
  times, exactly 64 per step, while the retired full-output boundary appears
  zero times. Generation closes over 12,997 leaves without missing, extra, or
  duplicate rows. Decode remains on one stream with raw equal to union and
  zero overlap; this is not a double/triple-buffer or executor-overlap result.
- [`qwen36-27b-fp8-m1-qkv-z-bf16-ab-tail-composite-production-benchmark.json`](qwen36-27b-fp8-m1-qkv-z-bf16-ab-tail-composite-production-benchmark.json),
  which records the selected and promoted exact-M1 linear-attention QKV/Z plus
  BF16 A/B tail composite. It preserves the established QKV/Z arithmetic and
  assigns A/B to 24 otherwise-light tail CTAs, passes actual-checkpoint,
  stress, replay, nonfinite, resource, Graph, invalid-contract, model-oracle,
  Release, and profile gates, and removes 1,200 launches over 25 Decode steps.
  The fixed-clock P19/C32/max26 result establishes the current production
  anchor of **108.2645 ms/token and 9.236638048 token/s**.
- [`qwen36-27b-decode-o-proj-prerounded-residual-chain-rejection.json`](qwen36-27b-decode-o-proj-prerounded-residual-chain-rejection.json),
  which closes the test-only two-kernel output-projection residual-handoff
  screen. Residual and gate results are bitwise exact, guards and dead-up
  workspaces pass, and both kernels retain their required occupancy, but the
  actual-checkpoint candidate regresses in all five fixed-clock rounds to
  0.994944x, projecting a 0.235329-ms/token loss. First-process stop-loss
  therefore leaves production dispatch, Prefill, and the 108.2645-ms/token
  anchor unchanged.
- [`qwen36-27b-decode-gate-up-cache-global-rejection.json`](qwen36-27b-decode-gate-up-cache-global-rejection.json),
  which closes the test-only cache-global (`.cg`) screen for the production
  Decode gate/up dead-up kernel. Production SASS remains byte-identical and
  the candidate changes exactly the eight packed-weight and four block-scale
  load qualifiers while preserving resources and bitwise outputs. The first
  fixed-clock actual-checkpoint process nevertheless reaches only 0.999327x,
  with no fully improving round and a projected 0.0304642-ms/token regression,
  so stop-loss leaves the production anchor unchanged and routes the next cell
  to the independent evict-first streaming policy.
- [`qwen36-27b-decode-nvfp4-scale6-sidecar-rejection.json`](qwen36-27b-decode-nvfp4-scale6-sidecar-rejection.json),
  which closes the lossless 6-bit NVFP4 block-scale sidecar screen against the
  production-selected Decode streaming kernels. Gate/up regresses in all 30
  formal actual/stress rounds. Down improves in all 30 rounds, but its three
  independent 53-layer projections are 0.434494, 0.268021, and 0.199822
  ms/token; the third misses the required 0.25-ms/token absolute gate. The
  pre-formal deadlock exploration is recorded separately and excluded from
  performance after the warp-uniform shuffle fix. Production, the serial
  one-stream schedule, Prefill, and the **107.889500 ms/token / 9.268742556
  token/s** anchor remain unchanged.
- [`qwen36-27b-decode-fp8-linear-qkv-streaming-rejection.json`](qwen36-27b-decode-fp8-linear-qkv-streaming-rejection.json),
  which closes the test-only evict-first streaming-load screen for the current
  public linear-attention FP8 QKV/Z plus BF16 A/B tail composite. Production
  and candidate each contain 3,504 normalized SASS words and retain identical
  `64r/1,280B/0local/4CTA-SM` resources; only 32 FP8 weight `LDG.E` loads
  become `LDG.E.EF`. Actual and same-bank fixtures are bitwise/guarded and
  improve in all five rounds, but the first retained formal process projects
  only 0.172513 ms/token over 48 linear-attention layers, below its 0.20-ms
  stop-loss floor. The other two formal processes, end-to-end benchmark, and
  Nsys closure were therefore not run. Production, the serial one-stream
  schedule, Prefill, and the **107.889500 ms/token / 9.268742556 token/s**
  anchor remain unchanged.
- [`qwen36-27b-decode-fp8-o-proj-streaming-rejection.json`](qwen36-27b-decode-fp8-o-proj-streaming-rejection.json),
  which closes the test-only evict-first sidecar-load screen for the production
  FP8 output projection. Production and candidate each retain 2,048 normalized
  SASS words and identical `64r/1,152B/0local/4CTA-SM` resources; eight
  sidecar `LDG.E.128` loads become `LDG.E.EF.128`, while eight activation
  `LDG.E.64` loads remain. Correctness, guards, input immutability, Graph
  replay, and eleven invalid cases pass, but all five actual-checkpoint rounds
  regress. The 0.985003x median projects to a 0.175823-ms/token loss over 64
  layers, so first-process stop-loss skips stress timing and later gates.
  Production, the serial one-stream schedule, Prefill, and the **107.889500
  ms/token / 9.268742556 token/s** anchor remain unchanged.
- [`qwen36-27b-decode-nvfp4-lm-head-streaming-rejection.json`](qwen36-27b-decode-nvfp4-lm-head-streaming-rejection.json),
  which closes the test-only packed-weight/block-scale streaming screen for the
  activation-staged NVFP4 LM head. Production and candidate each retain 1,584
  normalized SASS words and identical `64r/11,328B/0local/4CTA-SM`
  resources. Bounded and actual correctness, signed NaNs, Graph replay, 17
  invalid cases, guards, full payload hashes, and activation exactness pass,
  but all five actual rounds regress. The 0.993515x median adds 0.0286746
  ms/token, so first-process
  stop-loss skips stress timing and later gates. Production, the serial
  one-stream schedule, Prefill, and the **107.889500 ms/token / 9.268742556
  token/s** anchor remain unchanged.
- [`qwen36-27b-decode-fp8-q-kv-aosoa4-sidecar-rejection.json`](qwen36-27b-decode-fp8-q-kv-aosoa4-sidecar-rejection.json),
  which closes the standalone layout-only full-attention FP8 Q plus K/V
  AoSoA4/preswizzled sidecar screen at the unchanged `2048x256` topology.
  Actual-checkpoint correctness, byte-exact GPU packing, resources, guarded
  Graph replay, and 25 invalid cases pass. Process 1 projects a 0.229120-ms/token
  reduction and also clears five stress rounds, but process 2 projects only a
  0.196940-ms/token reduction, below the frozen 0.20-ms/token absolute gate.
  Stop-loss
  skips process-2 stress, a third process, the unallocated 1.09375-GiB
  production sidecar, integration, and end-to-end work. The test substrate is
  retained for a materially distinct sidecar plus CTA-persistence/coarsening
  screen; production, Prefill, and the **107.889500 ms/token / 9.268742556
  token/s** anchor remain unchanged.
- [`qwen36-27b-decode-fp8-q-kv-aosoa4-cta512-rejection.json`](qwen36-27b-decode-fp8-q-kv-aosoa4-cta512-rejection.json),
  which closes the follow-up full-attention sidecar CTA-coarsening screen.
  The candidate maps the same 2,048 logical workers into `1024x512`, shares
  one decoded codebook between two 256-thread halves, and preserves bitwise
  outputs, byte-exact packing, guarded Graph replay, all 25 invalid cases, and
  a zero-spill 1,024-active-thread/SM envelope. All five actual-checkpoint
  rounds nevertheless regress; the 0.984332x median projects the isolated
  per-layer loss to a **0.106153-ms/token regression**, not an end-to-end
  result. First-process stop-loss skips stress, a second process, the
  unallocated 1.09375-GiB production sidecar, integration, and end-to-end
  work. The next independent cell is attention preprocess/GQA cooperative
  fusion; production, Prefill, and the **107.889500 ms/token / 9.268742556
  token/s** anchor remain unchanged.
- [`qwen36-27b-decode-gqa-sigmoid-warp-positions-selection.json`](qwen36-27b-decode-gqa-sigmoid-warp-positions-selection.json),
  which selects the test-only exact Q24/KV4/D256, S1-S64 fused GQA/sigmoid
  warp-position score path for production integration. Eight warps per
  query-head CTA process score positions in parallel while reproducing the
  original 256-product reduction tree bitwise; softmax, FP32 probability
  publication, value accumulation, the BF16 intermediate, and sigmoid gate
  remain unchanged. BF16 output and FP32 probabilities match at S1/S16/S20/
  S32/S44/S64, real-buffer Graph replay remains one distinct `24x256` node,
  and both routes retain `40r/1,280B/0local/0stack/6CTA-SM`. Two independent
  S20-S44 chain processes improve all 40 individual-plus-chain rounds, reach
  3.60462x-3.60602x, and project **0.386042-0.386268 ms/token** over 16
  full-attention layers against the frozen 0.20-ms gate. This remains an
  isolated synthetic test-only projection: production dispatch, Prefill, the
  serial one-stream schedule, and the **107.889500 ms/token / 9.268742556
  token/s** anchor are unchanged pending promotion, full suites/model oracles,
  whole-model B-C-C-B timing, and fresh Nsight Systems closure.
- [`qwen36-27b-decode-gqa-sigmoid-warp-positions-production-benchmark.json`](qwen36-27b-decode-gqa-sigmoid-warp-positions-production-benchmark.json),
  which records the subsequent exact Q24/KV4/D256 S1-S64 production
  promotion. The public and direct warp-position launchers resolve to one
  `24x256` Function while the shared-tree predecessor remains test-only;
  bitwise output/probability, Graph, resource, Release, verified ASan/UBSan,
  and C1/C8/C16/C32 model gates pass. Two production micro processes project
  0.385729-0.386008 ms/token over 16 layers, while the independent whole-model
  P19/C32/max26 `B1-C1-C2-B2` result directly moves the mirrored hot Decode
  mean from 107.9395 to **107.3140 ms/token**, establishing **9.318448665
  token/s**. Fresh Nsight Systems closes all 25 single-stream Decode ranges,
  finds 400 production and zero predecessor launches, and directionally cuts
  the target symbol from 0.7896896 to 0.2024768 ms/token. The formal gap is
  now **7.314 ms/token / 0.681551335 token/s**; Prefill remains deferred and no
  system double/triple buffering or Prefill/Decode overlap is claimed.
- [`qwen36-27b-decode-fp8-qkv-z-bf16-ab-causal-conv-epilogue-rejection.json`](qwen36-27b-decode-fp8-qkv-z-bf16-ab-causal-conv-epilogue-rejection.json),
  which closes the test-only 48-layer QKV/Z/A/B causal-convolution epilogue
  screen. Four sequential steps over five state arrays are bitwise exact;
  guards, input preservation, 28 zero-node invalid cases, the `2 -> 1` Graph
  replay, and the unchanged `64r/1,280B/0local/0stack/4CTA-SM` envelope pass.
  The candidate nevertheless regresses in all five formal rounds: its
  48-layer median is 26.9566 ms versus 21.5568 ms for production composite
  plus standalone convolution, or **0.800019x and a 5.38834-ms/token loss**.
  Static evidence explains the placement failure: 232 instructions are added
  to all 1,536 projection CTAs while the removed 224-instruction standalone
  kernel uses only 40 CTAs and accounts for about 0.284 ms/token. Process 2,
  production integration, end-to-end timing, and candidate Nsys are skipped;
  the **107.314000 ms/token / 9.318448665 token/s** anchor remains unchanged,
  and the next bounded Decode cell is standalone preprocess warp-tree work.
- [`qwen36-27b-decode-full-attention-preprocess-warp-rms-rejection.json`](qwen36-27b-decode-full-attention-preprocess-warp-rms-rejection.json),
  which closes the test-only Q24/KV4/D256/RoPE64 full-attention preprocess
  warp-RMS screen. Warp 0 reproduces the production `128 -> 64 -> 32 -> 16
  -> 8 -> 4 -> 2 -> 1` reduction order, reducing static shared memory from
  1,024 B to 4 B and primary-path CTA barriers from 10 to 2 while retaining
  six CTAs/SM, exact finite/nonfinite M1/M2/M8/M16 outputs, one distinct
  `M*28 x 256` Graph node, and 15 zero-node invalid cases. The authoritative
  M1 median is nevertheless only **1.03734x**, saving 0.000158074 ms/call and
  projecting just **0.00252919 ms/token** over 16 layers—7.91x below the
  frozen 0.020-ms gate. M2/M8/M16 remain positive supporting cells but do not
  override Decode M1. Process 2, production integration, end-to-end timing,
  and candidate Nsys are skipped; production and the **107.314000 ms/token /
  9.318448665 token/s** anchor remain unchanged. The next bounded cell is the
  attention output-projection exact-resident grid-64 screen.
- [`qwen36-27b-decode-fp8-o-proj-resident-grid64-rejection.json`](qwen36-27b-decode-fp8-o-proj-resident-grid64-rejection.json),
  which closes that exact output-projection resident-wave screen. The
  candidate launches the same production AoSoA4/preswizzled Function at
  `64x256` instead of `1024x256`, exactly filling the measured 16-SM by
  4-CTA/SM capacity while retaining `64r/1,152B/0local`. Same-bank and actual
  outputs, guards, and inputs are bitwise exact; both one-node Graphs replay
  bitwise with the same Function, all 11 invalid calls capture zero nodes,
  signed infinity/NaN fixtures pass, and the canonical and sidecar SHA gates
  pass. A separate legacy cache-streaming run also preserves its original
  functional gates and frozen stop-loss behavior after screen
  parameterization. Nevertheless, all five grid-64 actual-checkpoint rounds
  regress: the public median is 0.179963 ms/layer versus 0.186084 ms/layer at
  grid 64, for **0.967290x**, a 0.00609002-ms/layer loss, and an isolated
  **0.389761-ms/token projected regression** over 64 layers. Frozen stop-loss
  skips stress timing, processes 2/3, nearby grid-128/256 trials, production,
  end-to-end, and Nsys work. This rejects only exact grid 64, not all
  persistent-grid mechanisms; the **107.314000 ms/token / 9.318448665 token/s**
  anchor remains unchanged. The next priority is a new major-projection
  mechanism pending parallel audit.
- [`qwen36-27b-decode-fp8-qkv-z-register-lookahead-rejection.json`](qwen36-27b-decode-fp8-qkv-z-register-lookahead-rejection.json),
  which closes the test-only QKV/Z rolled direct-LDG register-lookahead
  screen. The pinned SASS contains the intended runtime loop and one-stage
  next-weight lookahead without full-K preload or local spills, contracts from
  1,752 to 1,144 static instructions, and preserves four CTAs/SM while reducing
  registers from 64 to 53. Actual layer-0 QKV/Z/A/B outputs, guards, all five
  inputs, distinct one-node Graph replay, 11 zero-node invalid calls, payload
  hashes, and resources pass. Performance nevertheless regresses in all five
  rounds: production is 0.475987 ms/layer versus 0.517730 ms/layer for the
  candidate, or **0.919387x**, a 0.0418743-ms/layer loss and an isolated
  **2.00997-ms/token projected regression** over 48 layers. Frozen stop-loss
  skips process 2, stress timing, production, end-to-end, Nsys, and NCU work;
  the projection is not an end-to-end result. Production and the **107.314000
  ms/token / 9.318448665 token/s** anchor remain unchanged. The next bounded
  Decode cell is NVFP4 gate/up warp specialization.
- [`qwen36-27b-decode-nvfp4-gate-up-eight-plus-eight-warp-specialization-static-rejection.json`](qwen36-27b-decode-nvfp4-gate-up-eight-plus-eight-warp-specialization-static-rejection.json),
  which closes that gate/up candidate at static audit, before build. The
  `32x512` mapping is complete and invertible: eight warps per matrix execute
  17 row quads, covering `32 * 8 * 17 * 4 = 17,408` rows with CTA-local index
  543 as the maximum, and it can preserve the existing numerical order.
  However, it removes no packed-weight, scale, activation, or intermediate
  bytes; no FFMA, launch, CTA barrier, or occupancy limit; and production
  already has no CTA-wide barrier between its gate and up phases. The closest
  M32 8+8 analog reached only **0.897969x/0.900066x** with 0/8 positive rounds,
  while the M1 balanced-tail repartition reached **0.951273x/0.955264x**. An
  incomplete 344-line probe was fully reverted without build, test, commit,
  or production reachability. This is therefore a static NO-GO, not a measured
  regression; the **107.314000 ms/token / 9.318448665 token/s** anchor remains
  unchanged, and the next priority is a down runner-only dead-raw
  inline-residual contract audit.
- [`qwen36-27b-decode-down-dead-raw-inline-residual-rejection.json`](qwen36-27b-decode-down-dead-raw-inline-residual-rejection.json),
  which closes that test-only `32x512` down candidate. It keeps the rounded raw
  BF16 value in a register, performs the existing left-then-raw residual add,
  and removes the raw workspace store/read while preserving 64 registers,
  35,904 static shared bytes, zero local bytes, two CTAs/SM, and one cooperative
  launch. Actual layer-0 residual and normalized outputs are bit exact; the
  candidate raw workspace stays poisoned; guards, five inputs, one-node/root
  Graph replay, 27 representative zero-node invalid calls, payload hashes, and
  static mechanism gates pass. Final-binary performance nevertheless regresses
  in all five rounds: production is 0.312246 ms/layer versus 0.312416 ms/layer,
  for **0.999333x** and an isolated **0.0133438-ms/token projected loss** over
  64 layers. The +0.169-ms gate fails, so frozen stop-loss skips stress,
  nonfinite, process 2, production, end-to-end, Nsys, and NCU work. An older
  binary's 1.000380x repeat is retained only as near-noise directional evidence.
  Normal-generation raw liveness was analyzed but not integrated, while trace
  execution retains production. The **107.314000 ms/token / 9.318448665
  token/s** anchor and serial one-stream schedule remain unchanged; the next
  priority is a fresh Decode cross-layer/launch-amortization audit pending.
- [`qwen36-27b-decode-qkv-cs-down-scale6-bundle-selection.json`](qwen36-27b-decode-qkv-cs-down-scale6-bundle-selection.json),
  which selects projection palette v2 for joint implementation, not production.
  Six refreshed component processes use the same binary but run QKV streaming
  and down-only scale6 sequentially in separate processes. Their 48-layer QKV
  projections are 0.117695/0.168936/0.183432 ms and their 53-layer down
  projections are 0.195385/0.232112/0.307122 ms. The paired arithmetic sums
  are 0.313080/0.401048/0.490554 ms, with a 0.401048-ms median; this is only an
  implementation-admission statistic, not a jointly timed bundle or E2E
  result. Down scale6 covers 53 layers, keeps 11 canonical fallbacks, and adds
  221,429,760 bytes (0.206223 GiB) because canonical scales remain resident.
  Gates registered after reviewing the refresh require a future joint
  candidate to save at least 0.30 ms in every process and 0.35 ms at the
  three-process median, with every component actual/stress round positive;
  future E2E must improve every process and save at least 0.25 ms at median.
  These are not pre-refresh frozen gates. The arithmetic-only projection is
  106.912952 ms/token / 9.35340 token/s and does not replace the **107.314000
  ms/token / 9.318448665 token/s** anchor. A 64-of-388 launch-amortization
  heuristic recovers only about 0.101 ms from the 0.613196-ms idle upper bound,
  so palette v2 is the next bounded candidate; production and the serial
  one-stream schedule remain unchanged.
- [`qwen36-27b-decode-projection-palette-v2-production-benchmark.json`](qwen36-27b-decode-projection-palette-v2-production-benchmark.json),
  which records the subsequent production promotion of the 48-layer FP8 QKV/Z
  `.cs` route plus exact down-only scale6 sidecars for 53 layers, with 11
  canonical down fallbacks. Three same-process/same-stream component screens
  pass all 60 selected actual/stress rounds and yield arithmetic-only savings
  of 0.447933/0.789277/1.199360 ms, with a 0.789277-ms median; those values are
  isolated projections, not E2E timing. C1/C8/C16/C32 model oracles pass, and
  six whole-model processes preserve the same 29-line/1,881-byte canonical
  generation contract and golden SHA. The `B1-C1-C2-B2-B3-C3` subsequent-token
  medians are 107.186/106.666/106.860/107.114/107.157/106.769 ms; all three
  pairwise savings are positive at 0.520/0.254/0.388 ms and have a 0.388-ms
  median. The first four mirrored processes directly establish **106.763000
  ms/token and 9.366540843 token/s**, saving 0.551 ms against the prior formal
  anchor and leaving **6.763 ms/token / 0.633459157 token/s** to the stage
  target. The 221,429,760-byte sidecar and its roughly two-second cold build
  remain outside hot Decode latency. Bulk tiled Prefill is unchanged, while
  the single-token finish-prefill step shares the new M1 route. Scheduling is
  still serial on one stream without double/triple buffering or overlap. A
  fresh palette-v2 Nsight Systems trace closes all 11,749 hot inference kernels
  on one stream and exactly counts 48-layer QKV, 53-layer scale6 down, and
  11-layer canonical down dispatch; its profiled timing remains supporting
  attribution and is not used to establish this E2E anchor.

- [`qwen36-27b-decode-nvfp4-m1-gate-up-scale-aosoa4-sidecar-rejection.json`](qwen36-27b-decode-nvfp4-m1-gate-up-scale-aosoa4-sidecar-rejection.json),
  which records the bounded rejection of a test-only, scale-only Row-Quad
  AoSoA4 Gate/Up sidecar while canonical packed weights remain unchanged. All
  ten actual/stress rounds improve and both fixtures remain bit-exact, finite,
  guarded, and workspace-safe, but the real-checkpoint median reaches only
  1.00139x and a projected 0.0523529 ms/token across 64 layers, missing the
  frozen 1.02x and 0.50-ms/token promotion gates; stress reaches 1.00450x and
  passes its 1.00x qualification gate. Productionizing the same-size sidecars
  would add 0.6640625 GiB while retaining canonical scales, so the process-1
  stop-loss skips replication, allocation, loader/model integration, E2E, and
  Nsys work. Production dispatch and ABI remain unchanged, and the formal
  Decode anchor remains 106.763 ms/token and 9.366540843 token/s.

- [`qwen36-27b-decode-fp8-o-proj-cta512-shared-activation-rejection.json`](qwen36-27b-decode-fp8-o-proj-cta512-shared-activation-rejection.json),
  which rejects a test-only exact-M1 FP8 output-projection CTA512 candidate.
  Two independent 256-thread workers share one staged 6,144-element BF16
  activation per CTA; production retains its `1024x256` route, while the
  candidate uses `640x512`. On pinned layer-0 checkpoint weights with a
  synthetic activation, resources, 0/5,120 bitwise correctness, guards, and
  input immutability pass. All five actual-checkpoint rounds regress: public
  and candidate medians are 0.176767 and 0.192815 ms/layer, the paired median
  is 0.915945x, and the -0.0161972-ms/layer delta projects to a
  **1.03662-ms/token loss** over 64 layers. First-process stop-loss therefore
  skips stress, P2/P3, model, E2E, and Nsys work. Source and test hooks remain
  test-only, production ABI/dispatch are unchanged, and the formal Decode
  anchor remains **106.763 ms/token and 9.366540843 token/s**.

- [`qwen36-27b-decode-gate-up-down-cooperative-fusion-static-rejection.json`](qwen36-27b-decode-gate-up-down-cooperative-fusion-static-rejection.json),
  which closes the proposed `32x512` cooperative Gate/Up-to-Down chain before
  build. A direct read of the existing palette-v2 Nsys SQLite trace finds all
  1,600 Gate/Up boundaries across 25 Decode steps immediately followed by the
  expected Down Function. Their complete device gaps total only 2,144,640 ns,
  or **0.0857856 ms/token**: 1.3404 us/layer on average, with 1.344-us median,
  1.408-us P95, and 1.728-us maximum. The 53 scale6 and 11 canonical layers
  contribute 0.0709632 and 0.0148224 ms/token respectively. Although Gate/Up
  host launch calls sum to roughly 0.419 ms/token, they finish a median 50.284
  ms before GPU execution and are therefore overlapped, not additive savings.
  Retaining the global BF16 gated activation eliminates no global bytes, while
  the candidate requires a new grid barrier and must fit a 32-block resident
  grid with zero occupancy margin. This is a static NO-GO with no candidate,
  build, GPU run, or production change. Reopen only for a materially different
  mechanism that removes real computation or global traffic and has a ceiling
  of at least 0.3 ms/token; the formal anchor remains **106.763 ms/token and
  9.366540843 token/s**.
- [`qwen36-27b-decode-nvfp4-m1-gate-up-silu-fp32-table-rejection.json`](qwen36-27b-decode-nvfp4-m1-gate-up-silu-fp32-table-rejection.json),
  which rejects a test-only exact-BF16 SiLU FP32 lookup in the production-shaped
  `32x512` Gate/Up dead-up epilogue. Its cold-initialized 65,536-entry table is
  262,144 bytes; all table values match an independent device direct reference
  bitwise, both actual/stress outputs are exact, and the candidate preserves 64
  registers, 13,632 shared bytes, zero local bytes, and two CTAs/SM. Performance
  is effectively flat and slightly negative: hardened actual production/candidate
  medians are 0.594244/0.594725 ms/layer, paired speedup is 0.999793x, and the
  64-layer projection is a **0.00788879-ms/token regression** against the
  +0.30-ms gate, with 4/5 rounds regressing. Stress reaches 1.00011x but also
  has two reversals. Static SASS confirms one extra LDG replaces the SiLU
  EX2/RCP sequence, but the bandwidth-dominated projection does not benefit.
  P1 stop-loss skips P2/P3, model E2E, Nsys, and production integration. The
  formal anchor remains
  **106.763 ms/token and 9.366540843 token/s**.
- [`qwen36-27b-decode-nvfp4-lm-head-rp2-schedule-aosoa2-rejection.json`](qwen36-27b-decode-nvfp4-lm-head-rp2-schedule-aosoa2-rejection.json),
  which rejects the test-only exact-shape LM-head RP2 schedule-major AoSoA2
  row-pair mechanism at the actual-first P1 gate. The candidate's `80x256`
  kernel uses 48 registers, 11,328 shared bytes, zero local bytes, and reaches
  five CTAs/SM. Full-byte mapping of the 635,699,200-byte weight sidecar and
  79,462,400-byte scale sidecar passes; all 248,320 direct and Graph-replay
  outputs are bit-exact, finite, guarded, and input/sidecar-safe; valid Graph,
  invalid zero-node, and signed `0x7f`/`0xff` NaN contracts also pass. Only 2/5
  actual rounds improve. Baseline/candidate pass medians are 4.69201/4.75805
  ms, while the paired median is **0.973487x** with a
  **-0.126222-ms/token** delta, failing the frozen 1.075x, +0.30-ms, and
  every-round gates. Stop-loss skips stress, P2/P3, model, profiling, and
  integration. Production remains unchanged, so the formal anchor stays
  **106.763 ms/token and 9.366540843 token/s**.
- [`qwen36-27b-decode-fixed-position-cuda-graph-p1-selection.json`](qwen36-27b-decode-fixed-position-cuda-graph-p1-selection.json),
  which validates host-launch amortization with a test-only full predicted-only
  Decode CUDA Graph specialized to P19. The graph contains exactly 390 nodes:
  the existing 389 serial kernels plus one greedy-result D2H, with only the
  embedding root token offset updated before replay. Primary input `77517`
  predicts `220`; alternate input `220` predicts the pinned independent oracle
  `52965`; both serial/Graph pairs match over the complete 89,096,192-byte
  request arena. The final feature-commit process measures **106.670894 ms**
  serial versus **105.747733 ms** Graph, saving **0.918392 ms** at **5/5**
  positive rounds and reaching **9.456467 token/s**. Three earlier passing
  processes save 0.905752-0.934808 ms; a separate unbound process with one
  111.131-ms Graph outlier is retained as a strict 4/5 environmental failure,
  and no gate was lowered. An initial alternate-token failure is also retained:
  it exposed a real default-stream/nonblocking-stream arena-restore race, fixed
  by explicit synchronization outside timing. This P1 does not install a
  production graph cache and does not meet 100 ms, so the formal anchor remains
  **106.763 ms/token / 9.366540843 token/s**. The next bounded step is a
  position-specialized GraphExec cache covering continuous P19-P43 Decode,
  with exact state/fallback gates plus cold-time and memory budgets.
- [`qwen36-27b-decode-short-position-cuda-graph-cache-p2-selection.json`](qwen36-27b-decode-short-position-cuda-graph-cache-p2-selection.json),
  which extends the proven P1 mechanism to a test-only 25-slot P19-P43
  GraphExec cache and exercises it through the real continuous generation
  dispatcher. All 25 predictions, final text/stop semantics, P31/P32 pinned
  transitions, and the complete 89,162,240-byte arena match serial execution.
  Empty-cache, full-statistics, trace, reference-backend, and P64 cache-miss
  paths take exact serial fallbacks; the separate P63 fixture predicts `0`
  through Graph and P64 predicts `59720` through the real dispatcher fallback.
  The exact feature-commit process has **106.943472 ms/token** serial and
  **105.978481 ms/token** Graph pair medians, a 0.964991-ms difference, while
  the median of paired per-round savings is **0.919298 ms/token** and all
  **5/5** rounds improve. The first distinct-slot chain saves 0.839136
  ms/token. The production-window cache prepares in 83.475048 ms and has a
  138,436,608-byte
  observed CUDA free-memory drop; adding the out-of-window P63 fixture brings
  the diagnostic total to 145,055,744 bytes. P2 therefore passes its exact,
  hot, cold, and resource gates, but remains default-off pending a transactional
  internal production cache manager. It reaches **9.435878 token/s**, not the
  10-token/s target, so the formal production anchor remains **106.763 ms/token
  / 9.366540843 token/s**.
- [`qwen36-27b-decode-short-position-cuda-graph-cache-production-benchmark.json`](qwen36-27b-decode-short-position-cuda-graph-cache-production-benchmark.json),
  which promotes the selected P19-P43 cache through an engine-lifetime SM87
  policy and the ordinary predicted-only, non-trace CLI and benchmark paths.
  The 25 slots are staged and published transactionally outside generation;
  admission failure preserves serial execution, while a runtime Graph failure
  is not retried serially for the same token and demotes later work. The
  production gate reproduces canonical and post-reset P19-P43 dispatch at
  **25/0** Graph/serial, keeps full-statistics and trace at **0/0**, and reaches
  P44 with **25/1**; all compared generation semantics are exact. Its latest
  cold preparation is **75.758861 ms** with a **76,607,488-byte** observed CUDA
  free-memory drop, inside the 1-second and 256-MiB budgets. Ordinary-entry
  `B1-C1-C2-B2` measures **106.755000** versus **105.870500 ms/token**, saving
  **0.884500 ms/token**; both pairs improve, every candidate sample dispatches
  25/0, and the new directly achieved formal anchor is **105.870500 ms/token /
  9.445501816 token/s**. The target remains unmet by 5.870500 ms/token and
  0.554498184 token/s. The generic engine API remains default-disabled;
  execution stays on one stream without double/triple buffering, overlap, or a
  dedicated Prefill performance change. Fault-injection coverage for
  transactional rollback and runtime demotion remains explicit test debt.
- [`qwen36-27b-decode-nvfp4-m1-gate-up-table-free-e2m1-rejection.json`](qwen36-27b-decode-nvfp4-m1-gate-up-table-free-e2m1-rejection.json),
  which rejects an exact production-CS M1 gate/up twin that replaces only the
  16-entry E2M1 LUT with the already exhaustive PRMT constructor. Resources
  remain `64r/0local/2CTA-SM` and shared memory falls from 13,632 to 13,568
  bytes; actual residual/gate and second-direct replay are bit exact. Static
  SASS changes from 1,352 to 1,568 instructions, with `LDS` 89 to 25 and
  `PRMT` 3 to 131. All five actual-first rounds regress: baseline/candidate
  medians are 0.589658/0.949274 ms/layer, paired speedup is 0.621145x, and the
  projected 64-layer delta is -23.0181 ms/token. Stop-loss skips stress, Graph,
  invalid-matrix, model, profiling, and production work; the candidate hooks
  are removed and the 105.870500-ms/token / 9.445501816-token/s anchor is
  unchanged.
- [`qwen36-27b-decode-down-adaptive-scale-bitwidth-ceiling-rejection.json`](qwen36-27b-decode-down-adaptive-scale-bitwidth-ceiling-rejection.json),
  which closes adaptive lossless down-scale widths without code. Exact scanning
  finds 53 direct six-bit and 11 seven-bit layers. An ideal padding-free direct
  payload saves only 2.709359606% and projects to a 0.027813500-ms/token median;
  even dictionary indices save only 5.172413793% before codebooks and project
  to 0.053098500 ms/token. Both miss the 15% byte and 0.30-ms/token gates, so no
  branch, build, GPU run, or production change is warranted.
- [`qwen36-27b-decode-non-mtp-global-traffic-state-audit.json`](qwen36-27b-decode-non-mtp-global-traffic-state-audit.json),
  which explicitly excludes MTP and audits the current 5.870500-ms/token gap
  through exact projection payloads, fresh root-enabled production NCU,
  cross-kernel movement, L2 persistence, and SSM state update. The tracked
  minimum is 17.694 GB/token and requires at least 176.936 GB/s at 100 ms.
  Projection traffic is already near mandatory bytes; 2.75 MiB of persisting
  L2 covers only 3.819% of the 72-MiB GDN state. GDN is therefore selected for
  a test-only packed-register transient-state P1 with bit-exact, zero-local,
  at-least-three-CTA/SM, and 1.2448x / 0.30-ms/token stop-loss gates. This audit
  changes no production code or formal performance anchor.
- [`qwen36-27b-decode-gdn-m1-transient-register-state-rejection.json`](qwen36-27b-decode-gdn-m1-transient-register-state-rejection.json),
  which closes the selected non-MTP GDN single-shared-reload P1. Packed-BF16
  recompute and retained-FP32 variants remove the intended 144 MiB/token of
  shared reads, stay bit-exact with zero local memory and three or four CTA/SM,
  and improve all five mirrored rounds. Their 1.05144x/1.05294x results project
  to only 0.074424/0.076402 ms/token, however, far below the frozen 1.2448x /
  0.30-ms/token gate. Stop-loss removes all candidate code, restores the exact
  production blobs, and leaves the formal anchor unchanged.
- [`qwen36-27b-decode-down-global-flow-layout-ceiling-rejection.json`](qwen36-27b-decode-down-global-flow-layout-ceiling-rejection.json),
  which closes down-projection global-flow repacking with fresh NCU transaction
  evidence. Loads use 31.846 of every 32-byte sector and measured L2-miss bytes
  are within 0.114% of mandatory packed-weight plus scale6 payload. NCU's ideal
  load-pattern estimate projects to only 0.093974 ms/token; even also removing
  every unused store-sector byte reaches only 0.157985 ms/token. Both miss the
  0.30-ms/token gate, so no 2.20-GiB preswizzle sidecar or production change is
  created.
- [`qwen36-27b-decode-gate-up-p15e-t128-scale-payload-admission.json`](qwen36-27b-decode-gate-up-p15e-t128-scale-payload-admission.json),
  which admits only the capacity of a lossless gate/up four-bit top-15 palette
  plus raw-escape scale sidecar. All 128 pinned tensors round-trip bit exactly;
  the exact 398,909,184-byte representation is 55.945506% of canonical and all
  tensors pass the frozen 60% bound. The free-decode byte ceiling is 1.897253
  ms/token, but the prior scale6 decoder regressed in all 30 rounds, so a
  same-binary decoder/sector stop-loss remains mandatory before any full
  gate/up candidate or production allocation.
- [`qwen36-27b-lm-head-exact-progressive-mips-admission.json`](qwen36-27b-lm-head-exact-progressive-mips-admission.json),
  which closes the prediction-only LM-head fixed-cut/two-kernel route on 25
  captured P19--P43 final-norm activations. Its best 4,864-column cut saves
  only 34.584 MB/call on average and 33.912 MB/call at worst, so 0/25 steps
  reach the frozen 55-MB traffic gate. A distinct single-kernel q20
  progressive design remains conditional P2 only: its zero-code simulation
  saves 89.591 MB/call on average and 59.539 MB/call at worst, but it has no
  implementation or performance claim until per-row transaction suppression,
  incumbent publication, exactness, resources, and measured traffic pass.
- [`qwen36-27b-decode-gate-up-p15e-t128-decoder-rejection.json`](qwen36-27b-decode-gate-up-p15e-t128-decoder-rejection.json),
  which rejects the admitted P15E-T128 payload after an actual layer-0
  standalone decoder screen. Full host/device decode is bit exact with zero
  local memory and three CTA/SM, and LTS traffic falls 40.27%, but sparse
  directory/raw access raises L1 request sectors 32.01% and the decoder body
  grows to 368 instructions. All five rounds regress; the paired median is
  0.620662x and projects to a 4.785536-ms/token loss. Candidate source and its
  CMake target are removed, with no production allocation or dispatch change.
- [`qwen36-27b-decode-gdn-transposed-state-native-encoder-rejection.json`](qwen36-27b-decode-gdn-transposed-state-native-encoder-rejection.json),
  which closes the deeper GDN transposed-state and native-BF16-encoder family.
  The second transposed-state global read does hit L1, but 24-bank cold-state
  screens reach only 0.8430x for double-global, 0.9027x for BF16 shared retain,
  and 0.5792x for cooperative loading. The isolated native encoder reaches
  only 0.996115x. All test-only candidates are removed, production source and
  tests are restored byte-for-byte, and the default CUDA GDN suite passes.
- [`qwen36-27b-decode-fp8-qkv-z-p127e-w128-payload-admission.json`](qwen36-27b-decode-fp8-qkv-z-p127e-w128-payload-admission.json),
  which selects only the byte-capacity of a lossless per-tensor top-127 FP8
  palette with seven-bit codes and sparse raw escapes. All 96 QKV/Z tensors
  pass the 92% cap: 4,026,531,840 raw bytes become 3,641,497,600 encoded bytes,
  saving 385,034,240 bytes/token with a 2.197907-ms zero-overhead ceiling. The
  candidate still requires a production-unreachable actual-payload screen;
  112-byte warp tiles may retain four L1 request sectors, so measured L2
  reduction, decode instructions, resources, and at least 0.30 ms/token net
  benefit remain hard gates. O projection is rank-2 only.
- [`qwen36-27b-decode-fp8-qkv-z-p127x-w128-decoder-rejection.json`](qwen36-27b-decode-fp8-qkv-z-p127x-w128-decoder-rejection.json),
  which closes the admitted seven-bit FP8 family on SM87. The stronger
  row-quad P127X hybrid is bit exact with 26 registers, 256 bytes shared, zero
  local memory, and six CTA/SM, but all five cold rounds regress to 0.369522x.
  NCU records 117.84% more L1 request bytes and 4.76% more LTS traffic. An
  impossible fixed-direct lower bound that omits every directory and escape
  operation still reaches only 0.568148x. The standalone source/target are
  removed; neither QKV/Z nor the narrower O-projection backup proceeds.
- [`qwen36-27b-decode-gate-up-delta4-row-fallback-admission.json`](qwen36-27b-decode-gate-up-delta4-row-fallback-admission.json),
  which admits one production-unreachable Gate/Up scale decoder screen. Across
  all 128 tensors, 81.380086% of 32-scale row tiles fit an exact unsigned
  four-bit delta; other rows read their retained canonical sector. A fixed
  64-byte `[lane-pair][row4]` code tile plus hoisted ten-tile metadata projects
  26.380086% fewer L1 request sectors and a 1.136086-ms/token traffic ceiling.
  Layer 0 and worst-case layer 50 must each save at least 4.6875 us/layer in
  every round; this record contains no CUDA timing or production change.
- [`qwen36-27b-decode-gate-up-delta4-row-fallback-decoder-rejection.json`](qwen36-27b-decode-gate-up-delta4-row-fallback-decoder-rejection.json),
  which closes that screen after correcting an unfair canonical spill. Both
  backends then use zero local memory and three active CTA/SM; Delta4 lowers L1
  sectors 27.36% and LTS sectors 29.07%, but raises dynamic instructions
  92.76%. All five actual layer-0 rounds regress to a 0.598319x median, so the
  valid layer-50 run and full GEMV integration are skipped and all candidate
  code/build artifacts are removed.
- [`qwen36-27b-decode-causal-conv-l2-persistence-ceiling-rejection.json`](qwen36-27b-decode-causal-conv-l2-persistence-ceiling-rejection.json),
  which rejects a persisting-L2 access-policy-window probe before coding. The
  2.8125-MiB history exceeds the 2.75-MiB persisting budget, and an optimistic
  44/45 coverage of all read/write bytes is worth only 0.034833 ms/token. Two
  production traces put the complete causal-convolution kernel at only
  0.260622--0.283672 ms/token, so even free removal misses the 0.30-ms gate.
- [`qwen36-27b-decode-cross-kernel-core-flow-ceiling-rejection.json`](qwen36-27b-decode-cross-kernel-core-flow-ceiling-rejection.json),
  which closes generic intermediate-flow fusion after an updated boundary
  inventory. Individual credible ceilings are at most about 0.095 ms/token;
  both adjacent linear boundaries total only about 0.154 ms/token even under
  impossible perfect removal. Gate-to-Down still requires global cross-CTA
  broadcast, while Down-to-next activation requests are 99.568176% inferred
  L1 hits, so neither logical byte count is a removable DRAM opportunity.
- [`qwen36-27b-decode-gdn-row16-register-baton-rejection.json`](qwen36-27b-decode-gdn-row16-register-baton-rejection.json),
  which closes the exact test-only GDN row16 accumulator-baton screen. Resource
  and full state/output correctness gates pass at 71 registers, 2,568 B shared,
  zero local memory, and three CTA/SM. Two independent cold-state processes
  project 0.251234 and 0.247622 ms/token, below the frozen 0.30-ms/token hard
  gate; candidate code is removed, production is unchanged, and MTP is unused.
- [`qwen36-27b-lm-head-q20-exact-sector-rejection.json`](qwen36-27b-lm-head-q20-exact-sector-rejection.json),
  which closes q20 before CUDA after canonical inputs and the exact directed
  BF16 suffix-bound proof pass. Per-CTA/`-1` saves 19.244--34.044 MB/call with a
  26.311-MB mean, so 0/25 fixtures reach 55 MB. Ideal completed-wave/`-1`
  reaches 25/25 but has only 54,752 B of worst-case margin and requires 121
  global completion boundaries; lag-one reaches 24/25, and the contiguous-seed
  sweep has no all-fixture pass. No CUDA/NCU or production change follows, the
  105.870500-ms/token / 9.445501816-token/s anchor is unchanged, and MTP is
  unused.
- [`qwen36-27b-prefill-c64-down-production-benchmark.json`](qwen36-27b-prefill-c64-down-production-benchmark.json),
  which promotes the selected exact NVFP4 `[M64,N5120,K17408]` down kernel and
  the C64 request/runner boundary under ABI 0.3.0. Fixed-clock mirrored C32/C64
  process medians put P65/P129/P513 Prefix speedups at 1.065650x/1.064499x/
  1.062855x, mixed P97 at 1.042762x, and P33 at 1.001030x. All 20 formal
  generations retain ID 9419, `Hello`, and exact step counts. P513 Nsight
  closes the intended route from 1,024 M32 down launches to 512 M64 launches;
  no MTP, double/triple buffering, or Prefill/Decode overlap is introduced.
- [`qwen36-27b-prefill-nvfp4-m64-gate-up-rejection.json`](qwen36-27b-prefill-nvfp4-m64-gate-up-rejection.json),
  which screens a test-only exact `[M64,N17408,K5120]` Gate/Up kernel against
  the promoted C64 policy's two production raw-weight cp.async M32 launches.
  Correctness, guards, single-node Graph, invalid-contract, and resource gates
  pass. The isolated projection reaches 1.08934x and the production-like
  dual-stream pair envelope reaches 1.07884x with all 12 rounds positive, but
  both miss their
  1.15x/1.12x promotion gates. Production remains unchanged and the next
  bounded candidate is FP8 M64 `[M64,N5120,K6144]`.
- [`qwen36-27b-prefill-fp8-m64-attention-output-production-benchmark.json`](qwen36-27b-prefill-fp8-m64-attention-output-production-benchmark.json),
  which promotes the exact aligned FP8 `[M64,N5120,K6144]` attention-output
  projection. Six fixed-clock micro rounds aggregate at 1.49086x with
  69 registers, 23,552 B shared, zero local memory, and three CTA/SM. Matched
  full-model Prefix speedups reach 1.025063x/1.015764x/1.024397x/1.023902x at
  P65/P97/P129/P513, while the unchanged P33 fallback passes its 0.5% gate.
  P513 Nsight confirms 1,024 M32 launches become 512 M64 launches and target
  projection time falls from 378.191584 to 279.108096 ms. Outputs and steps
  remain exact, persistent drop is zero, Decode stays frozen at 105.870500
  ms/token / 9.445501816 token/s, MTP is unused, and FlashInfer is not yet
  introduced.
- [`qwen36-27b-prefill-fp8-whole-chunk-qkv-z-screen.json`](qwen36-27b-prefill-fp8-whole-chunk-qkv-z-screen.json),
  which selects test-only N-major whole-chunk grids for exact C256/C512 FP8
  QKV `[10240,5120]` and Z `[6144,5120]`. The sequential M512 pair reaches
  **1.50681x** against the public production M32 chains; checkpoint-like and
  stress cells reach 1.56823x and 1.42834x, every round improves, and
  exhaustive E4M3FN, replay, Graph, invalid-call, guard, input, and resource
  gates pass. Production dispatch remains unchanged pending narrow exact-shape
  integration and full-model admission.
- [`qwen36-27b-prefill-fp8-whole-chunk-production-benchmark.json`](qwen36-27b-prefill-fp8-whole-chunk-production-benchmark.json),
  which records that production admission for exact aligned C256/C512 FP8 QKV,
  Z, and attention output. Frozen-binary P257/P513 `B1-C1-C2-B2` measurements
  reach **1.098411534x/1.099544898x Prefix** and
  **1.092760304x/1.096639387x TTFT** speedups; all 40 measured generations
  retain ID 9419, `Hello`, exact 257/513 steps, a common streamed contract hash,
  and zero persistent memory drop. The P513 production profile confirms
  48/48/64 QKV/Z/O whole-chunk launches totaling 633.578624 ms, replacing the
  exact prior contract's 2,560 C32 nodes with 160 nodes. The generic projection
  cap remains C64, all near misses retain their validated fallback, and no MTP,
  FlashInfer, new buffering, or Prefill/Decode overlap is introduced.
- [`qwen36-27b-prefill-fp8-whole-chunk-full-attention-screen.json`](qwen36-27b-prefill-fp8-whole-chunk-full-attention-screen.json),
  which selects test-only exact C256/C512 FP8 full-attention Q
  `[12288,5120]` and K/V `[1024,5120]` whole-chunk routes for narrow
  production promotion. C512 Q reaches **1.52743x/1.39116x**, K reaches
  **7.96850x/8.56814x**, and V reaches **7.97343x/8.53129x** across the two
  distributions; sequential Q-then-K-then-V reaches **2.54717x/2.57488x**
  and **2.55933x** aggregate. Correctness, exhaustive E4M3FN, replay, Graph,
  invalid-call, guard, input, and resources pass. Production dispatch remains
  unchanged, and the record makes no model-level or end-to-end claim.
- [`qwen36-27b-prefill-fp8-whole-chunk-full-attention-production-benchmark.json`](qwen36-27b-prefill-fp8-whole-chunk-full-attention-production-benchmark.json),
  which records production admission for those exact Q/K/V routes. Q uses
  N-major at C256/C512; K/V use M-major at C256 and N-major at C512. Frozen-
  binary P257/P513 `B1-C1-C2-B2` measurements reach **1.054381379x/
  1.055108640x Prefix** and **1.051134118x/1.053417066x TTFT** speedups.
  Complete-prompt throughput advances to **141.273450981/145.327508146
  token/s**, leaving **2.644368049x/2.830744195x** gaps to matched stock vLLM.
  All 40 outputs and steps remain exact. Fresh P513 Nsight confirms Q/K/V
  change from 2,304 launches / 314.357120 ms to 48 launches / 124.991296 ms.
  Both candidate processes have zero persistent drop; baseline B1's nonzero
  66,441,216-byte drop remains below the 64-MiB tolerance. Public ABI,
  workspace, Decode, MTP, FlashInfer, and buffering policy remain unchanged.
- [`qwen36-27b-prefill-gdn-whole-span-register-state-rejection.json`](qwen36-27b-prefill-gdn-whole-span-register-state-rejection.json),
  which rejects test-only exact C256/C512 GDN whole-span register-state
  lifetime extension. Bitwise output/state, replay, guards, inputs, Graph,
  invalid-call, and 64-register/34,056-byte-shared/four-CTA resource gates all
  pass, and every mirrored round improves. C256 reaches only 1.02672x and C512
  only 1.01871x, however, below the frozen 1.03x gate. The underlying
  performance test therefore fails its two expected threshold assertions;
  the default suite passes and production remains unchanged.
- [`qwen36-27b-prefill-gdn-c16-scalar-vector-qk-pingpong-rejection.json`](qwen36-27b-prefill-gdn-c16-scalar-vector-qk-pingpong-rejection.json),
  which retains a standalone exact-C16 GDN experiment that parallelizes all
  16 alpha/beta pairs and ping-pongs normalized Q/K across two shared slots.
  It reduces dynamic CTA barriers from 47 to 16 while preserving 64 registers,
  zero local memory, four CTA/SM, and bitwise production output/state,
  BF16x2 NaN-half, replay, Graph, guard, input, and invalid-call contracts.
  All six fixed-clock 24-bank rounds improve, but the **1.01990x median** and
  **1.01946x worst round** miss the frozen 1.06x/1.04x gates. Production,
  selector, runner, API/ABI, and MTP remain unchanged; this exact structure is
  retained only as reproducible rejection evidence and will not be micro-tuned.
- [`qwen36-27b-prefill-nvfp4-whole-chunk-gate-up-production-benchmark.json`](qwen36-27b-prefill-nvfp4-whole-chunk-gate-up-production-benchmark.json),
  which promotes exact aligned C256/C512 NVFP4 Gate and Up to one whole-chunk
  kernel per branch on the existing main/aux event fork/join. Frozen-binary
  P257/P513 Prefix improves by **1.087506234x/1.086814433x**, reaching
  **162.636231668/162.584844997 token/s**; complete-prompt throughput reaches
  **152.800878754/157.464097495 token/s**. All 40 formal outputs and steps are
  exact and persistent drop is zero. Fresh P513 Nsight reduces Gate/Up from
  2,048 nodes / 1,346.373984 ms to 128 nodes / 1,065.953440 ms. Only 5.65% of
  the union saving comes from stream overlap, so the record explicitly makes
  no double/triple-buffering or Prefill/Decode-overlap claim.
- [`qwen36-27b-prefill-nvfp4-gate-m128-b-reuse-screen.json`](qwen36-27b-prefill-nvfp4-gate-m128-b-reuse-screen.json),
  which selects the test-only exact C256/C512 M128 Gate/Up B-tile-reuse kernel
  for production admission. Three frozen-binary processes reach
  **1.275707061x/1.283725529x** aggregate single-branch speedup at M256/M512;
  the production-like main/aux pair reaches process-minimum
  **1.27998x/1.28465x**, and all 72 pair rounds improve. Single-branch
  exact/replay/guard/input and pair exact/replay/guard checks pass, as do
  Graph, invalid-call, and 126-register/37,376-byte-shared/two-CTA resource
  gates. At the time of this screen the production route was unchanged;
  applying the conservative M512 pair minimum projected, but had not yet
  achieved, **2,912.933316 ms / 175.767841 token/s** P513 Prefix. The later
  production record below closes that admission plan without rewriting this
  historical screen.
- [`qwen36-27b-prefill-nvfp4-gate-m128-production-benchmark.json`](qwen36-27b-prefill-nvfp4-gate-m128-production-benchmark.json),
  which promotes exact C256/C512 Gate/Up to M128 decoded/staged B-tile reuse.
  The formal M64 comparator and M128 candidate both contain the same output-
  span safety fix. Mirrored fixed-clock P257/P513 Prefix improves by
  **1.079886387949x/1.081515606806x** to **175.547351155/175.730487094
  token/s**; complete-prompt throughput reaches **164.104109691/
  169.740778038 token/s**. All 40 outputs, steps, canonical hashes, persistent-
  memory checks, E2E gates, and the 52-pass/12-skip suite remain exact. Fresh
  P513 attribution retains 8,209 total and 128 Gate/Up nodes, halves Gate/Up
  grid X from 1,088 to 544, and reduces its interval union from 1,065.953440
  to 827.889280 ms (**1.287555553x**). The gain is M128 B reuse, not new
  buffering or overlap; the existing two-stream fork/join remains unchanged.
- [`qwen36-27b-prefill-nvfp4-gate-c512-inclusive-cublaslt-admission.json`](qwen36-27b-prefill-nvfp4-gate-c512-inclusive-cublaslt-admission.json),
  which admits a test-only exact-C512 hybrid large-M route for the next
  production-integration gate. Direct canonical NVFP4-to-BF16 decode takes
  **1.236907 ms**, persistent-weight cuBLASLt takes **2.323137 ms**, and the
  inclusive **3.561288 ms** median is **1.842441x** faster than the fresh
  **6.561464 ms** production M128 Gate reference. All 89,128,960 decoded BF16
  values match the scalar oracle bitwise, non-unit weight scaling and the
  exact two-node Graph replay pass, and an independent process replicates
  **1.840714x**. Production remains unchanged: Gate+Up pair contention,
  production-M128 numerical comparison, Down reuse, memory admission, and
  pinned-checkpoint Prefix gates are still required. The native fused large-M
  kernel remains the parallel upper-bound route. The later pair-selection
  record below closes the zero-workspace production comparison and Gate/Up
  scheduling gates without rewriting this historical admission result.
- [`qwen36-27b-prefill-nvfp4-gate-up-c512-cublaslt-pair-selection.json`](qwen36-27b-prefill-nvfp4-gate-up-c512-cublaslt-pair-selection.json),
  which closes P0 and selects the exact-C512 production-integration schedule.
  Zero-workspace candidate output matches all 8,912,896 production-M128 values
  bitwise. The two-weight screen measures **12.894805 ms** production versus
  **7.155811 ms** for one-handle/one-scratch serial Gate then Up
  (**1.802005x**, 6/6 positive rounds). Naive dual-stream and staggered
  schedules reach only **0.980728x/0.984581x** relative to serial and are
  rejected together with the second 170-MiB scratch. All eager and Graph
  outputs are exact with intact guards. Production remains unchanged pending
  the request-workspace, runner, pinned-checkpoint, and mirrored Prefix gates.
- [`qwen36-27b-prefill-nvfp4-gate-up-c512-cublaslt-production-benchmark.json`](qwen36-27b-prefill-nvfp4-gate-up-c512-cublaslt-production-benchmark.json),
  which closes those gates and admits the narrow exact-C512 production route.
  Gate and Up execute serial direct canonical-NVFP4-to-BF16 decode plus
  zero-workspace cuBLASLt while overwriting one conditional 170-MiB request
  scratch. All 89,128,960 decoded values and 8,912,896 production-M128 outputs
  are bitwise exact. Mirrored fixed-clock P513 Prefix improves from
  **2557.970 ms / 200.158720 token/s** to **2208.733 ms / 231.807104 token/s**
  (**1.158116441x**); TTFT improves **1.150713102x** to **2317.622 ms**.
  Nsight sees 128 decode and 128 Lt calls and zero prior C512 Gate/Up M128
  calls. The 62-pass/12-skip suite has zero failures, and mirrored Decode is
  unchanged within noise at **105.8270/105.8995 ms per subsequent token**.
  C256, Down, finish-Prefill, Decode, MTP, and near misses remain unchanged;
  the next bounded screen is exact-C512 Down reuse of the same scratch.
- [`qwen36-27b-prefill-nvfp4-down-c512-cublaslt-p0-ceiling.json`](qwen36-27b-prefill-nvfp4-down-c512-cublaslt-p0-ceiling.json),
  which retains the test-only Down P0 and its two bounded decoder follow-ups
  without changing production. The fully sequential no-spill route is exact
  but regresses to **0.880460x** and is rejected. The selected compile-time
  `8+8+8+8+2` Window8 route reaches 32 registers, zero local bytes, and six
  CTAs/SM; its dequant screen improves **1.315078x** with all six rounds
  positive. After the entire formal candidate chain is switched to Window8,
  direct dequant takes **1.266897 ms**, zero-workspace Lt **3.268475 ms**, and
  the measured inclusive path **4.534723 ms** versus **6.656049 ms** live
  production M128 (**1.467796x**, worst round **1.466581x**). All 89,128,960
  decoded values, 2,621,440 production outputs, exact two-node Graph replays,
  guards, immutability, scale handling, and 16-byte Lt preference pass.
  Production remains unchanged until an independent Down context supplies the
  full null/shape/alias/overflow/device rejection contract. It may reuse the
  existing 170-MiB request scratch after SiLU, with no second scratch or
  persistent BF16 weights.
- [`qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-development-baseline.json`](qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-development-baseline.json),
  which freezes the self-developed exact-C512 M64xN256xK64 test-only baseline.
  K256 scale reuse, paired packed-weight copies, and aligned 32-bit BF16 stores
  move Gate from **6.471293 ms** to **5.532324 ms** (**1.169724x**) with 128
  registers, zero local memory, and two CTAs/SM. All 17,825,792 outputs and
  Graph replays are bitwise exact and 21/21 invalid cases fail closed. The
  native route remains outside production because it misses the existing
  1.22x gate; final NCU evidence rejects further scale-window, triple-pipeline,
  and shared-broadcast micro-tuning as incapable of credibly supplying the
  remaining 4.3%. Future native work requires a structural tile, decode, or
  data-layout change.
- [`qwen36-27b-prefill-nvfp4-gate-c512-native-m128n128-ca-development-cell.json`](qwen36-27b-prefill-nvfp4-gate-c512-native-m128n128-ca-development-cell.json),
  which retains the exact-C512 M128xN128 A-only `cp.async.ca` result as a
  positive test-only development cell, not a production admission. B and
  scales remain on `cp.async.cg`; SASS contains 12 non-bypass A and six bypass
  B/scale LDGSTS sites versus 18 bypass sites in the CG control. Exact
  0/17,825,792, two-node Graph replay, 23/23 invalid-call, guard, immutable-
  input, 128-register/49,152-byte-dynamic-shared/zero-local/two-CTA, full-suite,
  and independent-audit gates pass. All six formal rounds improve
  **5.531556 ms to 5.483505 ms** (**1.008763x**), but the result remains above
  the historical **5.304339 ms** absolute gate, so the old procedure stopped
  pair timing. That bridge-derived gate is now revoked: the positive native
  result is retained as development evidence, remains outside production, and
  was later superseded by faster cells. Uncontrolled-cache/clock NCU is causal
  evidence only: versus the differently tiled M64 control, matched L2 requests
  fall
  **11.23%** and bypass bytes **51.86%**, so the full counter delta is not
  attributed to `.ca` alone. The M128 arithmetic body closes and the next
  bounded cell transfers only the A-cache operator to the frozen M64xN256
  topology.
- [`qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-ca-development-cell.json`](qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-ca-development-cell.json),
  which retained the then-fastest native exact-C512 Gate kernel as a
  positive test-only baseline, not a production admission. The frozen
  M64xN256 topology is unchanged: only A's six LDGSTS sites move from
  `cp.async.cg` to `cp.async.ca`, while nine B/scale sites remain bypassing.
  All 17,825,792 outputs, Graph replay, 23/23 invalid calls, resource gates,
  the 69-pass/12-skip full suite, and independent audit pass. CPU-11-pinned
  six-round B-C-C-B timing is uniformly positive at **5.531551 ms to 5.388746
  ms** (**1.026501x**); matched same-topology NCU shows L2 requests down
  **14.9294%** and bypass bytes down **45.8234%**. Under the superseded policy,
  the **5.304339 ms** absolute gate missed by **0.084407 ms**/**1.566357%** and
  pair timing was skipped. Under the current policy this uniformly positive
  native result remains valid development evidence; CF3 and BS512 later
  superseded it, and the old bridge has no current production status.
- [`qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-abca-rejection.json`](qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-abca-rejection.json),
  which rejects extending cache-all from activation A to packed B on the
  otherwise identical M64xN256 native cell. Exact outputs, Graph replay,
  23/23 invalid calls, immutable inputs, 128-register/two-CTA resources, the
  69-pass/12-skip full suite, and independent audit pass. CPU-11-pinned
  six-round B-C-C-B timing regresses uniformly from **5.388882 ms to 5.698348
  ms** (**0.945692x**, **5.742675%** higher latency). Matched same-topology NCU
  records **17.6510%** fewer L2 request bytes and **84.5815%** fewer bypass
  bytes, yet duration rises **5.4785%**, issue active falls **2.56 points**,
  and MIO throttle rises **2.69 points**. Packed B therefore remains on
  `cp.async.cg`; the A-only-CA native baseline and production route are
  unchanged, and pair timing is skipped by stop-loss.
- [`qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-ca-seed-rejection.json`](qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-ca-seed-rejection.json),
  which rejects replacing the retained A-only-CA kernel's per-CTA scalar
  E4M3FN-to-BF16 table construction with a warp-coalesced 512-byte
  device-global seed. The 256/256 seed values, 17,825,792 projection outputs,
  Graph replay, resources, full 69-pass/12-skip suite, and independent audit
  pass. SASS removes 24 instructions while preserving 64 HMMA, 128 PRMT, and
  nine bypass LDGSTS sites, but the CPU-11 MAXN B-C-C-B short screen regresses
  **5.416544 ms to 5.450288 ms** (**0.993808768x**) and misses the preset
  **1.008x** continuation gate. Stop-loss therefore skips six formal rounds,
  NCU, and pair timing. Production dispatch semantics and the A-only baseline
  are unchanged, although the test module physically adds a 512-byte ordinary
  `.nv.global.init` seed and changes binary bytes.
- [`qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-ca-factored-rejection.json`](qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-ca-factored-rejection.json),
  which rejects the exact scale-factored x4 decoder in the retained
  M64xN256 A-only-CA topology. Its 131,072 candidate-specific exhaustive
  combinations, 17,825,792 projection outputs and Graph replay, 23/23 invalid
  calls, guards, immutable inputs, and 123-register/zero-local/two-CTA
  resources pass. SASS halves HFMA2 sites from 32 to 16 and removes 128 PRMT
  sites, but grows from 1416 to 2240 instructions. All six CPU-11 MAXN formal
  rounds regress from **5.387814 ms to 10.429699 ms** (**0.516584x**), the
  unchanged **5.304339 ms** absolute gate fails, and Gate+Up timing stops.
  The sole authoritative isolated serial suite is clean at 69 passes and 12
  expected skips; an earlier ceiling-test miss during concurrent GPU resource
  contention is excluded rather than classified as a regression. Production,
  Decode, and MTP remain unchanged.
- [`qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-ca-full-product-rejection.json`](qwen36-27b-prefill-nvfp4-gate-c512-native-m64n256-ca-full-product-rejection.json),
  which rejects the test-only runtime-built scale-major 256x16 full-product
  lookup at its compiled-mechanism stop-loss. Whole-projection and two-node
  Graph replay comparisons are bitwise exact, 23/23 invalid calls fail with
  zero captured nodes, and the kernel keeps 128 registers, zero local memory,
  and two CTAs/SM. Its 8,192 static plus 43,008 dynamic shared bytes are 51,200
  logical bytes/CTA, or 52,224 bytes after the driver's additional 1,024-byte
  overhead. The generic `REGISTER_FED_X4_EXHAUSTIVE` result validates only the
  retained decoder; the candidate's table-initialization logic covers all
  256x16 entries but has no independent candidate-specific exhaustive oracle.
  SASS changes from 1416 instructions/32 HFMA2/128 PRMT/70 LDS to 1472/0/48/118
  while preserving 64 HMMA and 15 LDGSTS sites. It therefore misses the preset
  at-most-1384 instruction gate and stops before performance timing, NCU, or
  Gate+Up pair work. The isolated full suite is clean at 69 passes and 12
  expected skips. Production dispatch, Decode, and MTP are unchanged, though
  the test binary physically changes.
- [`qwen36-27b-gate-c512-cf3-complete-cell-2026-07-29.json`](qwen36-27b-gate-c512-cf3-complete-cell-2026-07-29.json),
  which immutably preserves the real-weight Gate/Up CF3 complete-cell
  measurement and its original legacy bridge-derived decision wording.
- [`qwen36-27b-gate-c512-cf3-policy-correction-2026-07-29.json`](qwen36-27b-gate-c512-cf3-policy-correction-2026-07-29.json),
  which append-only corrects the historical decision semantics without
  rewriting that artifact. CF3 was a valid positive native development
  retention result, was later superseded by structured BS512, and was never
  promoted to production; every old bridge rejection field has no current
  authority.
- [`qwen36-27b-gate-c512-bs512-retention-2026-07-29.json`](qwen36-27b-gate-c512-bs512-retention-2026-07-29.json),
  which records the exact native-only structured-BS512 requalification on the
  isolation commit. The binary has no cuBLASLt dependency or import, allocates
  zero reference scratch, validates CF3 and BS512 eagerly and through CUDA
  Graph replay, and retains BS512 by **1.060981x** serial and **1.063372x**
  dual across six all-positive real-weight B-C-C-B rounds. Production
  promotion and dispatch remain unchanged.
- [`qwen36-27b-native-only-c512-p513-baseline-2026-07-29.json`](qwen36-27b-native-only-c512-p513-baseline-2026-07-29.json),
  which establishes the post-isolation current-production P513/C512 baseline
  from two independent native-only processes. The mean Prefix median is
  **2558.108500 ms / 200.147883 token/s** and mean TTFT median is
  **2666.888500 ms / 192.358998 complete-prompt token/s**; all ten measured
  generations produce token 9419 (`Hello`), no process reports a persistent
  drop, and the L1024 request arena is 233,940,992 bytes. The missing
  temperature envelope is explicit rather than inferred.
- [`qwen36-27b-gate-c512-bs512-vs-bf16-matched-ncu-2026-07-29.json`](qwen36-27b-gate-c512-bs512-vs-bf16-matched-ncu-2026-07-29.json),
  which records the post-isolation full matched-NCU diagnostic for one
  real-checkpoint layer-0 Gate launch. Structured BS512 takes **5.122560 ms**
  versus **2.378112 ms** for the external BF16 GEMM and **1.288832 ms** for
  its separate dequantizer. Native issues only **58.93%** of the inclusive
  external TEX-sourced L2 sectors but executes **5.687757x** the BF16 GEMM
  instructions for the same 22,282,240 HMMA executions. The result
  makes a 256-thread
  M128xN256 decoded-B-reuse cell the next test-only priority. The external
  path remains reference-only with no retention, promotion, fallback, or
  production authority.
- [`qwen36-27b-gate-c512-m128n256-bs512-256t-retention-2026-07-29.json`](qwen36-27b-gate-c512-m128n256-bs512-256t-retention-2026-07-29.json),
  which retains the 256-thread M128xN256 structural successor against
  structured BS512 on the pinned layer-0 Gate/Up checkpoint. Serial improves
  **10.184074 -> 9.372453 ms (1.086596x)** and dual improves
  **10.115721 -> 9.324912 ms (1.084806x)**, with all twelve rounds positive.
  The candidate is bitwise exact, uses 241 registers, 96,256 dynamic plus 512
  static shared bytes, zero local bytes, and one CTA/SM. cuBLASLt is absent;
  production promotion and dispatch remain unchanged.
- [`qwen36-27b-gate-c512-m128n256-bs512-256t-matched-ncu-2026-07-29.json`](qwen36-27b-gate-c512-m128n256-bs512-256t-matched-ncu-2026-07-29.json),
  which completes the source-identical full matched-NCU attribution for the
  retained M128 cell against structured BS512. On one real-checkpoint layer-0
  Gate launch, M128 takes **4.696736 ms** versus **5.119744 ms**
  (**1.090064x**) while preserving all 22,282,240 HMMA executions. It halves
  compressed-B decode/dependency opcodes, cuts warp instructions **33.56%**,
  TEX-sourced L2 read-sector bytes **26.47%**, and excessive shared wavefronts
  **50%**. The activation is deterministic BF16 rather than a captured layer
  tensor, so the report is diagnostic-only. cuBLASLt is `NOT_RUN`; production
  promotion and dispatch remain unchanged. The next native cell couples raw-B
  and packed-scale consumer order rather than reopening isolated cache tuning.
- [`qwen36-27b-prefill-gate-c512-a-stationary-production-2026-07-29.json`](qwen36-27b-prefill-gate-c512-a-stationary-production-2026-07-29.json),
  which promotes strict one-dimensional A-stationary CTA ordering for exact
  C512 NVFP4 Gate/Up. Formal real-model B-C-C-B lowers mean P513 Prefix from
  **2330.9795 to 2324.6885 ms** and raises throughput from **219.650151 to
  220.244562 token/s**; both candidate runs beat both native baselines and
  preserve token 9419 (`Hello`). Matched NCU lowers one real-weight Gate from
  **4.698528 to 4.665280 ms** despite more L2 read misses, while L2 write
  requests fall **9.45%** for unchanged output sectors. Gate/Up eager/Graph
  replays are bitwise exact and all 76 runnable Release tests pass with 12
  policy/environment skips. C256, Down, FP8, Decode, GDN, MTP, and the
  cuBLASLt reference-only production policy are unchanged.
- [`qwen36-27b-prefill-gate-c512-k128-double-production-2026-07-29.json`](qwen36-27b-prefill-gate-c512-k128-double-production-2026-07-29.json),
  which promotes the exact-C512 Gate/Up M128xN256 pipeline from 80 published
  K64 stages in three slots to 40 published K128 stages in two slots while
  preserving the same 80 ordered K64 `cp.async` groups. Formal real-model
  B-C-C-B lowers mean P513 Prefix from **2322.7290 to 2293.8755 ms** and raises
  throughput from **220.430364 to 223.203047 token/s**; both candidate runs
  beat both native baselines and preserve token 9419 (`Hello`). Matched NCU
  lowers one real Gate from **4.665280 to 4.406944 ms**, with identical global
  and L2 read sectors but 4.15% fewer instructions, 15.43% fewer barrier-stall
  samples, and 25.24% fewer MIO-throttle samples. The new LD136 row also
  exposes 5,570,560 shared-load conflicts, defining the next LD72 split-plane
  screen. All 76 runnable Release tests pass with 12 expected skips; C256,
  Down, FP8, GDN, Decode, MTP, and cuBLASLt's reference-only status are
  unchanged.
- [`qwen36-27b-prefill-gate-c512-k128-split-plane-rejection-2026-07-29.json`](qwen36-27b-prefill-gate-c512-k128-split-plane-rejection-2026-07-29.json),
  which rejects splitting each K128 activation slot into two physical LD72
  K64 planes. The required first real P513 direction is positive by 1.085 ms,
  so the candidate correctly advances to complete validation; formal B-C-C-B
  then retains only a 0.684-ms mean advantage without cross-run separation.
  Matched real-weight NCU is decisively negative: one Gate regresses
  **4.406944 -> 4.427744 ms**, while 5,570,560 load-bank conflicts, 38,993,920
  shared-load wavefronts, and 144,095,808 warp instructions are all unchanged.
  LD72 and LD136 are 128-byte congruent in shared-bank phase, so the candidate
  only adds 4 KiB of shared storage. It is fully reverted; the `d7aa73b` K128
  production route remains the native champion.
- [`qwen36-27b-prefill-gate-c512-k128-ld144-rejection-2026-07-29.json`](qwen36-27b-prefill-gate-c512-k128-ld144-rejection-2026-07-29.json),
  which rejects a non-congruent LD144 activation bank phase. The required real
  P513 direction regresses Prefix **2294.365 -> 2333.889 ms**, so no formal
  harness is built. Matched real-weight NCU attributes the loss directly: one
  Gate regresses **4.406944 -> 4.709184 ms**, load-bank conflicts increase
  fivefold to 27,852,800, and shared-load wavefronts rise 57.14% with unchanged
  instructions and occupancy. The 128 target launches explain 97.88% of the
  whole-prefix regression. The candidate is fully reverted and blind leading-
  dimension sweeps are closed.
- [`qwen36-27b-prefill-gate-c512-k128-k64-consumer-loop-rejection-2026-07-29.json`](qwen36-27b-prefill-gate-c512-k128-k64-consumer-loop-rejection-2026-07-29.json),
  which rejects replacing the fully unrolled two-half consumer with a runtime
  K64 loop. Real P513 Prefix regresses **2294.071 -> 2335.508 ms**. Matched NCU
  keeps all 5,570,560 load conflicts, adds 2.45% warp instructions, and
  regresses one Gate **4.406944 -> 4.746720 ms**, despite two fewer registers
  and a 64-KiB smaller runner. The source is reverted; the next bounded cell
  targets the merged 64-byte B row with two compile-time-selected K64 planes.
- [`qwen36-27b-prefill-gate-c512-k128-b32-planes-production-2026-07-29.json`](qwen36-27b-prefill-gate-c512-k128-b32-planes-production-2026-07-29.json),
  which promotes two independent 32-byte shared-B planes inside each exact-
  C512 K128 Gate/Up slot. Across two B-C-C-B sets every candidate Prefix beats
  every baseline; the combined mean falls **2293.33775 -> 2288.51025 ms** and
  reaches **223.726330 token/s**. Matched real-weight NCU eliminates all
  5,570,560 load-bank conflicts, lowers shared-load wavefronts 14.29%, reduces
  registers 249 -> 247, and improves one Gate **4.406944 -> 4.384128 ms** with
  identical global/L2 sectors. Fresh production NSys ranks Gate/Up at 556.839
  ms, GDN at 488.461 ms, and Down at 426.603 ms. All 76 runnable Release tests
  pass with 12 expected skips; cuBLASLt remains reference-only and MTP is not
  used.
- [`qwen36-27b-prefill-gdn-b8-block-transition-screen.json`](qwen36-27b-prefill-gdn-b8-block-transition-screen.json),
  which selects the test-only sequential FP32-B8 GDN dataflow and rejects the
  measured WY control. C256 reaches **2.76977x** versus production M16 while
  WY reaches 1.74177x; C512 reaches **2.78551x** versus 1.74228x. Sequential
  beats WY in all six mirrored rounds. Independent CPU C1/C7/C8/C9/C15/C16,
  split-tail, frozen numerical, immutable-input, invalid-call, and
  109-register/8,256-byte-shared/two-CTA resource gates pass. The 24-bank
  37,748,736-byte state pool exceeds the 4-MiB L2, but this is not an L2
  hit-rate claim. Production dispatch, runner, ABI, Decode, and MTP remain
  unchanged; real-checkpoint and fixed-clock full-model gates are still due.
- [`qwen36-27b-prefill-gdn-b8-real-checkpoint-rejection.json`](qwen36-27b-prefill-gdn-b8-real-checkpoint-rejection.json),
  which closes that admission by rejecting FP32-B8 on the pinned checkpoint.
  Exact route counters prove 48/48/96/96 linear-layer hits at
  P257/P513/P769/P1025, and short Engine outputs remain identical to the
  current production baseline. Recurrent state does not: Prefix aggregate
  NRMSE grows from **0.0741172** at P257 to **0.148576** at P1025 versus the
  frozen **0.01** gate, while aggregate cosine falls from 0.997252 to
  0.988911. The test-only option defaults OFF and cannot be enabled without
  `BUILD_TESTING`; the OFF executable remains bitwise identical to the frozen
  M128 production binary. Production stays on exact per-token BF16 M16 GDN;
  no MTP, public ABI, Decode, or buffering policy changes.
- [`qwen36-27b-prefill-fp8-m128-b-reuse-production-benchmark.json`](qwen36-27b-prefill-fp8-m128-b-reuse-production-benchmark.json),
  which promotes exact C256/C512 FP8 QKV/Z/O/full-Q to M128xN128 K64
  decoded-B reuse while retaining full-attention K/V on the screened M64
  layouts. Three production-admission processes reach **1.39169x--1.39285x**
  weighted kernel speedup. Frozen-binary P257/P513 Prefix improves by
  **1.064843195x/1.063438857x** to **186.975342/186.870419 token/s**, and
  TTFT improves by **1.060239389x/1.061019397x**. All 40 formal outputs and
  steps have one canonical hash; real-checkpoint C64-versus-C256/C512 E2E,
  exhaustive E4M3FN, Graph identity, invalid-call, memory, and the
  52-pass/12-skip suite pass. Fresh P513 Nsight proves exactly 176 promoted
  M128 calls and 32 retained M64 K/V calls, while NCU records 25% less L1
  global-load traffic for QKV/C512. No MTP, FlashInfer, new buffering,
  Prefill/Decode overlap, workspace, public-ABI, or Decode change is used.
- [`qwen36-27b-prefill-nvfp4-down-m128-production-benchmark.json`](qwen36-27b-prefill-nvfp4-down-m128-production-benchmark.json),
  which promotes only exact C256/C512 NVFP4 Down `[5120,17408]` to
  M128xN128 K64 decoded-B reuse. Three production-admission processes retain
  two CTA/SM and reach **1.29838x--1.29880x C256** and
  **1.26845x--1.26979x C512** cross-distribution minimum speedups. Mirrored
  frozen-binary P257/P513 Prefix improves by **1.050250272x/1.043924307x**
  to **196.261675/195.040523 token/s**; complete-prompt throughput reaches
  **181.985297/187.647869 token/s**. All 40 model results, exhaustive raw
  E4M3FN-by-E2M1 coverage, replay, Graph identity, invalid-call, memory, bulk
  E2E, and the 52-pass/12-skip suite pass. Fresh P513 attribution keeps 64
  Down nodes, halves grid X from 320 to 160, and reduces raw Down time from
  542.446976 to 426.948448 ms (**1.270521016x**). Matched NCU records 19.23%
  less L1 global-load traffic and 40.81% fewer SM instructions with unchanged
  Tensor instructions. No MTP, FlashInfer, new buffering, API, workspace,
  public-ABI, or Decode change is used.
- [`qwen36-27b-prefill-nvfp4-gate-m256-b-reuse-rejection.json`](qwen36-27b-prefill-nvfp4-gate-m256-b-reuse-rejection.json),
  which closes two test-only M256 Gate/Up mappings without changing
  production. The 256-thread/16-accumulator form uses 216 registers per
  thread and fails its resource gate before correctness or timing. A distinct
  512-thread/two-warp-group form reaches 128 registers, zero local memory,
  one CTA/SM, and exact C512 output/replay, but regresses from 6.48127 to
  8.19205 ms: **0.791166x** production-M128 throughput, with all six mirrored
  rounds between 0.790548x and 0.791389x. The first-C512 1.05x stop-loss skips
  C256, the remaining distributions, pair timing, NCU, Nsys, and full-model
  work. Gate/Down M128 SASS remains byte-for-byte identical to the frozen
  `c885d8e` binary; both rejected candidates were removed and MTP was unused.
- [`qwen36-27b-prefill-nvfp4-gate-m128-n256-a-reuse-rejection.json`](qwen36-27b-prefill-nvfp4-gate-m128-n256-a-reuse-rejection.json),
  which rejects a test-only Gate M128xN256 activation-reuse mapping at its
  first C512 checkpoint-like cell. Cold Graph, 21 invalid calls, resource,
  exact **0/8,912,896**, replay, guard, and input gates pass at 124 registers,
  55,808 total shared bytes, zero local memory, 512 threads, and one CTA/SM.
  Fixed-clock 6x24 `B-C-C-B` timing regresses from 6.47376 to 7.17060 ms:
  **0.902819x**, with all six rounds below one. The stop-loss skips pair/C256,
  other distributions, candidate NCU, Nsys, and full-model work; the candidate
  is removed and production SASS remains identical to the frozen binary. A
  separate production Gate baseline records 99.867% L1 global-load misses, a
  768-KiB context set-aside, and no activation access-policy window, selecting
  a two-CTA-preserving L2 window screen next. This is not candidate traffic
  evidence, and MTP remains unused.
- [`qwen36-27b-prefill-nvfp4-gate-up-l2-apw-rejection.json`](qwen36-27b-prefill-nvfp4-gate-up-l2-apw-rejection.json),
  which completes that two-CTA-preserving screen without changing production
  kernels, dispatch, runner, Decode, or MTP policy. C256/C512 exact/replay,
  guards, inputs, 21 zero-node invalid cases, resources, and cache-policy
  restoration pass. At C512, a 2.75-MiB set-aside with 2.5 MiB nominally
  selected reaches only **0.998762x** for balanced 0.25/0.25 and
  **0.998956x** for main-owner 0.5/0; all twelve rounds regress. The stop-loss
  skips NCU, Nsys, full-model, and production APW work, makes no measured
  L2-hit-rate claim, and selects register-fed MMA or removal of the decoded-B
  shared-memory round trip behind a 1.20x first-pair gate.
- [`qwen36-27b-prefill-nvfp4-gate-register-fed-sidecar-rejection.json`](qwen36-27b-prefill-nvfp4-gate-register-fed-sidecar-rejection.json),
  which closes that register-fed follow-up after two test-only stages.
  Sentinel commit `28ecd4f` freezes the bijective SM87 BF16 matrix-B
  lane/slot mapping and proves direct fragment feed bitwise against shared
  WMMA load. K16 commit `d28acb` regresses from 6.470355 to 6.788260 ms
  (**0.953168x**); K64-vectorized commit `0caed62` improves all six rounds but
  moves only from 6.465014 to 6.351387 ms (**1.017890x**) against the tightened
  **1.22x single-Gate** gate. Both C256/C512 layouts pass pair exact/Graph
  replay, Gate/Up sidecar oracles, exhaustive x4 decode, 21 zero-node invalid
  cases, guards, immutability, and 122-register/35,328-byte-shared/zero-local/
  two-CTA resources. Equal-byte per-tensor sidecars and their builders remain
  test-only; no 5.9766-GiB full-model allocation is admitted. Pair timing,
  NCU, Nsys, full-model, production integration, Decode, and MTP are unchanged.
  The selected next screen was a canonical-weight 512-thread M128 fused
  Gate+Up/shared-A pair behind a 1.22x C512 pair gate; bulk attention was
  already production at 2.904% of the current profile.
- [`qwen36-27b-prefill-nvfp4-gate-up-fused-m128-shared-a-rejection.json`](qwen36-27b-prefill-nvfp4-gate-up-fused-m128-shared-a-rejection.json),
  which closes that canonical-weight follow-up at commit `afafcd9`. The
  test-only 512-thread CTA keeps independent production-order Gate and Up
  groups, private B/C overlays, and one shared M128xK64 activation stage.
  C256/C512 resources, exact output, two Graph replays, finite/guard/input,
  one-node valid Graph, and 34 zero-node invalid-call gates pass at 124
  registers, 37,376 static plus 18,432 dynamic shared bytes, zero local memory,
  and one CTA/SM. Fixed-clock C512 6x24 `B-C-C-B` timing regresses from
  12.893605 to 14.485416 ms (**0.890109x**); all six rounds are negative versus
  the 1.22x gate. The result is only consistent with the 512-thread
  synchronization domain and one-CTA schedule—no skipped NCU/Nsys causal claim
  is made. C256 timing, profiling, full-model, and production work stop there;
  Prefill production, Decode, and MTP remain unchanged. Priority moves to an
  audit of existing FP8 QKV/Z/O candidates at 564.576448 ms / 21.425% of P513
  projected GPU time before choosing the next bounded first cell.
- [`qwen36-27b-prefill-gdn-c16-norm-gate-shared-boundary-t1-2026-07-29.json`](qwen36-27b-prefill-gdn-c16-norm-gate-shared-boundary-t1-2026-07-29.json),
  which records the standalone exact-C16 GDN plus plain-RMSNorm/SiLU-gate P0
  cell at commit `4c135d5`. Its synthetic T1 run proves finite shared/global
  final-output and recurrent-state equality plus the shared raw BF16 boundary;
  a separate NaN case proves the shared raw, final-output, and state boundaries.
  In-place/disjoint and one-node Graph replay, 22 guarded-buffer redzones,
  eight immutable inputs, seven invalid calls, and the zero-node invalid
  capture also pass. The shared candidate uses 64 registers, 38,184 static
  shared bytes, zero local bytes, and four CTAs/SM; the global control and
  frozen production exact-C16 body each report 64/34,056/0/four. Performance
  and production promotion are `NOT_RUN`; no full-model pinned-prompt path or
  pinned captured real-layer trajectory was used, and production remains
  unchanged. Device
  `compute-sanitizer` checking is unavailable because the target Orin reports
  its CUDA debug feature disabled, so no sanitizer pass is claimed.
- [`qwen36-27b-prefill-gdn-c16-norm-gate-p513-direction-rejection-2026-07-29.json`](qwen36-27b-prefill-gdn-c16-norm-gate-p513-direction-rejection-2026-07-29.json),
  which closes that P0 with the first real P513/C512 generation-path direction
  screen. Commit `6e668f5` proves 0/1,536 baseline/candidate route hits,
  complete 37,748,736-word GDN-state bitwise equality at committed positions
  512 and 513, and token 9419/text `Hello`/513-step semantics. A separate
  snapshot-free, same-engine/ELF measured B-C-C-B gives Prefix
  2558.744418 versus 2565.174709 ms (**0.997493235x**, 6.430291 ms slower)
  and TTFT 2667.414833 versus 2673.856789 ms (**0.997590763x**, 6.441956 ms
  slower). This one-round result has early-stop authority only: no noise
  calibration, formal six-round retention, or promotion work follows, and
  production remains unchanged. Nsys/NCU are `NOT_RUN_IN_THIS_CLOSURE`, but a
  later bounded profile remains admissible for an explicit causal question;
  it cannot reverse the rejection. cuBLASLt is neither linked nor executed,
  and MTP is unused.
- [`qwen36-27b-prefill-gdn-c16-warp-row-epilogue-retention-2026-07-29.json`](qwen36-27b-prefill-gdn-c16-warp-row-epilogue-retention-2026-07-29.json),
  which records the materially revised successor at commit `fc597e9`. A
  bounded NSys/NCU diagnosis attributes the predecessor's regression to
  sixteen serial 256-thread reduction trees per C16 CTA. The successor maps
  one row per warp in two eight-row batches while preserving the exact
  `(i,i+64)`, `(i+32,i+96)`, then 16/8/4/2/1 FP32 addition order. T1 and the
  complete 37,748,736-word P513 GDN-state gate remain bitwise, with unchanged
  64-register/38,184-byte-shared/zero-local/four-CTA resources. All six
  snapshot-free real P513 rounds are positive: Prefix falls from 2556.550133
  to 2529.929521 ms (**1.010522274x**, 26.620613 ms saved), while TTFT falls
  from 2665.298000 to 2638.661993 ms (**1.010094513x**). The gains clear the
  maximum matched noise by 126.4x/115.0x. NSys attributes 27.700000 ms per
  request to the closed kernel interval, and NCU shows fewer instructions and
  a 35.998% lower barrier-stall ratio. The candidate becomes the test-only
  native experimental incumbent; production still awaits the separate
  P257/P513/P769/P1025 promotion gate. cuBLASLt is neither linked nor
  executed, and MTP is unused.
- [`qwen36-27b-prefill-gdn-c16-warp-row-prompt-matrix-2026-07-29.json`](qwen36-27b-prefill-gdn-c16-warp-row-prompt-matrix-2026-07-29.json),
  which extends only the private test admission at commit `912897b` to C256
  and C512 tiles at 16-aligned first positions. One warm-up plus one measured
  B-C-C-B per profile is positive across P257/P513/P769/P1025: Prefix ratios
  are **1.010568820x/1.010578521x/1.010338310x/1.010329821x**, saving
  13.652206/26.785156/40.331183/53.824843 ms. All 32 invocations pass the
  generation and 768/1,536/2,304/3,072 route-hit oracles. A separate
  timing-inadmissible correctness mode compares every word in the
  37,748,736-element GDN state at ten C256/C512/final-step boundaries; all
  are bitwise with active, changing state. Default Release isolation passes.
  This confirms multi-tile applicability but does not replace the six-round
  P513 retention anchor or clear the 1.03x production margin, so production
  remains unchanged while another exact mechanism is stacked. cuBLASLt is
  neither linked nor executed, and MTP is unused.
- [`qwen36-27b-prefill-gdn-c16-constant-hoist-direction-rejection-2026-07-29.json`](qwen36-27b-prefill-gdn-c16-constant-hoist-direction-rejection-2026-07-29.json),
  which closes the next exact scalar cell. The candidate retains
  `exp(A_log)` and decoded `dt_bias` in two extra shared FP32 words per C16
  CTA, theoretically removing 23,040 repeated exponentiations and 46,080
  constant BF16 loads over P513. Resources remain 64 registers, zero local,
  and four CTAs/SM while shared grows 38,184 to 38,192 bytes. In the first
  same-engine/ELF real-path screen, all route and generation oracles pass but
  Prefix regresses 2530.701891 to 2531.008918 ms (**0.999878694x**) and TTFT
  regresses 2639.417501 to 2639.724687 ms (**0.999883629x**). The valid
  neutral-to-negative result triggers the direction stop-loss: no full noise,
  six-round, NSys, or NCU work follows, the patch is fully withdrawn, and the
  retained warp-row test incumbent plus production remain unchanged.
- [`qwen36-27b-prefill-gdn-c16-packed-prediction-direction-rejection-2026-07-29.json`](qwen36-27b-prefill-gdn-c16-packed-prediction-direction-rejection-2026-07-29.json),
  which closes the shared-scratch packing cell against the retained warp-row
  incumbent. Packing removes 768 logical shared bytes per token row, but exact
  sequential prediction must reload both halves and the update repeats BF16
  decode/alpha multiplication. Full unroll fails the zero-local resource gate
  at 56 bytes/thread; resource-clean unroll 1/4/8 variants are all negative on
  the real P513 path. The best unroll-8 cell moves Prefix 2537.545144 to
  2546.275917 ms (**0.996571160x**) and TTFT 2646.260633 to 2654.983662 ms
  (**0.996714470x**). A bounded NSys diagnostic shows 331.406 versus
  325.247 us per candidate/incumbent kernel call, explaining about 9.460 ms
  per request and the full profiled end-to-end loss. The patch is fully
  withdrawn; exhaustive correctness, noise, six-round retention, and NCU do
  not follow. Production, Decode, MTP, and the native incumbent are unchanged;
  cuBLASLt is neither linked nor executed.
- [`qwen36-27b-prefill-gdn-persistent-span-direction-rejection-2026-07-29.json`](qwen36-27b-prefill-gdn-persistent-span-direction-rejection-2026-07-29.json),
  which closes the last planned exact state-lifetime cell on the real P513
  path. One C512 CTA retains packed BF16 state and reuses a C16 shared raw
  window, removing 1,488 intermediate state-publication boundaries per
  request. The 78-register/three-CTA form and both four-CTA forms are
  negative. The best 64-register/zero-local/four-CTA variant moves Prefix
  2531.777600 to 2532.033922 ms (**0.999898768x**) and TTFT 2640.529857 to
  2640.769251 ms (**0.999909347x**). Bounded NSys shows the candidate C512
  kernels total 498.025816 ms/request versus 493.711440 ms for the incumbent
  C16 chain, so the kernel interval itself is 4.314376 ms slower. The patch is
  fully withdrawn; full correctness, noise, retention, and NCU do not follow.
  The retained warp-row C16 experimental incumbent, production, Decode, MTP,
  and cuBLASLt policy remain unchanged. Priority returns to the larger
  real-weight Prefill GEMM intervals.

The model-compatibility reports contain raw SHA-256 hashes for `config.json`,
`hf_quant_config.json`, and `model.safetensors.index.json`; normalized model and
quantization fields; index counts; representative shapes read from the first
safetensors shard header; and benchmark shapes derived from that evidence.
Tensor payloads were not downloaded to produce the header probes: the first
eight bytes were read to obtain the little-endian header length, followed by a
bounded HTTP byte-range request for the JSON header.

The pinned 27B artifact was subsequently materialized in a separate local
model directory. Git LFS and an independent `sha256sum` pass matched all three
published object IDs. `qwen3x-inspect checkpoint MODEL_DIR --require-shards`
then validated all 3 shard headers, all 2,194 index mappings, and exactly
21,921,428,072 payload bytes without reading payload contents. The 27B report
records all shard hashes/sizes and a text/vision/MTP storage split. No model
file is copied into this repository.

Once `qwen3x-inspect` is built, local evidence can be reproduced without loading
weight payloads into memory:

```bash
qwen3x-inspect index MODEL_DIR/model.safetensors.index.json
qwen3x-inspect header MODEL_DIR/model-00001-of-00003.safetensors
qwen3x-inspect checkpoint MODEL_DIR
qwen3x-inspect checkpoint MODEL_DIR --require-shards
qwen3x-inspect manifest MODEL_DIR
qwen3x-inspect load-plan MODEL_DIR
```

The strict form requires regular, non-symlink shard files and validates every
header, tensor membership/ownership, and aggregate payload size. The
non-strict form is intentionally metadata-only: it reports shard presence but
sets `shard_contract_validated=false`.

`load-plan` additionally assigns the exact deterministic text-only arena layout
against the three compiled full-file identities without allocating GPU memory.
The separate conditional resident-loader integration then read and SHA-256
authenticated all 21,921,697,184 file bytes in one sequential pass, copied
20,150,569,096 text bytes into one 20,150,786,560-byte CUDA arena, and skipped
1,771,128,088 bytes. Full details and the target-device reproduction command
are in [RESIDENT_WEIGHT_LOADER.md](../RESIDENT_WEIGHT_LOADER.md).

The inspector treats filenames and directory names as untrusted context. Model
series and quantization compatibility require exact pinned descriptors plus
semantic validation. A shape-compatible unknown revision remains unsupported.

## Known upstream metadata quirk

For the pinned 35B-A3B artifact, external `hf_quant_config.json` identifies
ModelOpt 0.44.0 and FP8 KV cache, while the embedded
`config.json.quantization_config` carries stale ModelOpt 0.37.0 metadata and no
KV declaration. Their 291 per-module `quantized_layers` entries agree exactly.
Qwen3x-Orin permits that discrepancy only for the pinned file hashes and treats
the external file as authoritative. The same disagreement on an unknown
revision fails closed.

Model weights, tokenizers, and source configuration files remain separately
licensed artifacts and are not copied into this repository.
