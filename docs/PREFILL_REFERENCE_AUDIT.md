# Prefill reference-architecture audit

This audit is a design reference, not a dependency-selection record. It pins
the inspected sources, compares their Prefill dataflow with the current
Qwen3x-Orin runtime, and changes the implementation order without importing
FlashInfer or qwen35-thor code.

## Scope and pinned sources

The audit was performed on 2026-07-27 against:

- [FlashInfer](https://github.com/flashinfer-ai/flashinfer) commit
  `4b969c9363e0d32a33f8fcccd9ef5a5fec51cd9f` and its
  [attention API](https://docs.flashinfer.ai/api/attention.html);
- [qwen35-thor](https://github.com/thomas-hiddenpeak/qwen35-thor) commit
  `57e29777c2aff8a97f42df6e3d9487b1327f014f`, including its pinned
  FlashInfer submodule commit `f521fe19ac387e8baffd7b5c925ef59d9f2ecc0c`;
- Qwen3x-Orin commit `516c22ed22568239e3407dff8a591d306334e61f`.

The external repositories were cloned read-only under `/tmp`. No external
source was copied into this repository, no new build dependency was added,
and production dispatch is unchanged.

## Current structural baseline

Qwen3x-Orin already separates Prefill and Decode at the control-plan level,
but the two phases still share one runner, request state, workspace, and main
CUDA stream. The relevant Prefill constraints are:

- the request and transcript ABI caps one tile at 64 tokens;
- most projections are split by the runner into ordered M32 launches before
  the low-level dispatcher can see the complete C64 request;
- only exact NVFP4 down and FP8 attention-output shapes currently have narrow
  M64 bypasses;
- full-attention Q/K/V projection and KV placement are tiled, but attention is
  still executed once per query token;
- only MLP Gate/Up has a main/auxiliary-stream branch; the rest of Prefill is
  ordered on the main stream and synchronizes at the end of each tile.

The current P513 trace makes the cost of those boundaries concrete:

| Group | Time | Launches | Prefix share |
| --- | ---: | ---: | ---: |
| NVFP4 Gate/Up projection | 1,735.606 ms | 2,048 | 40.563% |
| NVFP4 down projection | 738.322 ms | 512 | 17.256% |
| GDN update | 486.874 ms | 1,536 | 11.379% |
| FP8 linear QKV projection | 384.251 ms | 768 | 8.981% |
| FP8 attention output projection | 279.108 ms | 512 | 6.523% |
| FP8 linear Z projection | 243.086 ms | 768 | 5.681% |
| full-attention core | about 301.050 ms | about 22,528 | 7.036% |

The shares overlap because the projection table uses raw-kernel totals while
the denominator is the controlling Prefix range; they are ranking evidence,
not an additive accounting table. The important structural signal is that a
P513 request traverses eight complete C64 layer passes. Projection weights are
therefore presented to the machine eight times instead of one large-M layer
pass, while the attention core emits thousands of decode-style launches.

## What FlashInfer contributes as a reference

FlashInfer's `single_prefill_with_kv_cache` data contract is a close semantic
match for the full-attention part of this model:

- causal single-request Prefill/append is a bulk query operation;
- GQA is supported when the query-head count is a multiple of the KV-head
  count;
- separate NHD K/V tensors use `[sequence, heads, head_dim]`, matching the
  current contiguous BF16 cache shape `[sequence,4,256]`;
- already normalized and RoPE-transformed Q/K can use `pos_encoding_mode=NONE`;
- causal alignment covers a query suffix against a longer prefix-plus-query KV
  span, so an append tile need not be converted to a paged cache first;
- on non-Hopper devices backend selection falls to FA2, and the JIT accepts
  SM75 or newer. Official tests instantiate FA2 head dimension 256 and grouped
  heads.

This is source-level compatibility, not a measured Orin result. The repository
build and system Python environment do not contain PyTorch. A separate external
reference environment later located at `/home/rm01/setup/.venv` contains
PyTorch, vLLM, and FlashInfer, but it is not linked into native production.
The public Python wrapper allocates temporary/output tensors and uses a JIT/FFI
stack. A standalone probe must include setup, workspace, stream, KV placement,
gate, and any layout cost rather than timing only the attention kernel.

FlashInfer's current GDN Prefill implementation is not directly available on
SM87. Its dispatch accepts SM90, SM100/103, and SM120 families and uses a
64-token chunked delta-rule formulation with FP32 state on the SM90/SM120
paths. The algebra, checkpoint contract, varlen layout, and numerical tests
are valuable references, but the kernels cannot be adopted as an Orin
backend. Their correctness tests also use tolerance-based output/state gates,
not bitwise equality; any native SM87 WY experiment must define its own
explicit error and end-to-end token contract.

## What qwen35-thor contributes as a reference

qwen35-thor includes FlashInfer headers, but its production `src/` tree does
not call FlashInfer symbols. It uses FlashInfer and FLA primarily as algorithm
references and implements its own model-specific paths. The transferable
ideas are:

1. `forward_prefill` and `forward_decode` are separate model entry points.
2. A single Prefill chunk may contain 2,048 tokens by default. T>1 projections
   stay layer-major and use M-dependent GEMM/GEMV selection, so a P513 prompt
   is presented as one large-M layer pass rather than eight C64 passes.
3. Linear QKV and Z can be treated as one wider projection when weight layout
   permits; small A/B rows are handled separately.
4. Full attention uses a bulk T>=256 QK/softmax/PV route, and later chunks use
   a tiled prefix-plus-chunk attention path instead of one decode kernel per
   query token.
5. GDN Prefill uses an eight-token WY factorization with the state resident in
   shared memory across the enclosing token span. The present production
   implementation is scalar CUDA; SM100 UMMA types in the file are declared
   for a later phase rather than used by the kernel.
6. Workspace and library handles are prepared persistently rather than
   allocated in every hot call.

The reported qwen35-thor Qwen3.5-27B BF16 results reach 1,208 token/s at P512
and 1,046 token/s at P2048 on Jetson AGX Thor. Those numbers are not a
Qwen3x-Orin baseline: hardware, precision, weight representation, software,
prompt accounting, and model revision differ. They nevertheless validate the
architectural direction because the same report shows throughput peaking near
the whole-chunk GEMM region and falling when long prompts require repeated
chunks.

The following parts are not portable to Orin:

- SM110 TMA, UMMA, cluster, and programmatic-dependent-launch mechanisms;
- its CUTLASS SM110 kernels and hardware-specific thresholds;
- its materialized T-by-T attention score workspace as a final design;
- paged KV, continuous batching, SSD cache, MTP, and serving policy;
- its external performance numbers as promotion evidence.

## Decision and updated implementation order

The audit changes the active Prefill priority. The next test is no longer a
production-oriented M64 QKV promotion in isolation. It is a bounded
whole-chunk weight-reuse screen:

1. Build a test-only exact M256/M512 FP8 attention-output launcher from the
   already validated M64 CTA body. One launch must cover all M64 row tiles and
   order CTAs N-major so identical weight panels have a chance to be reused
   through L2. Compare it with four/eight ordered production M64 launches.
2. Require bitwise finite-output equality, classified-NaN agreement, guard and
   input preservation, one CUDA Graph node, no local memory, and no per-CTA
   resource regression. Advance the mechanism only if every timing round is
   non-regressive, M512 is at least 1.25x faster, and the measured hotspot
   table projects at least a 1.05x P513 Prefix opportunity across eligible
   projections.
3. If selected, generalize the test-only large-M grid across the dominant
   Gate/Up, down, QKV, Z, and output shapes before changing the public request
   ABI. Validate C256 and C512 in the same implementation, but publish one C512
   boundary rather than two successive ABI changes.
4. Keep exact FP8 M64 QKV and Z screens as fallback/control milestones. They
   remain valid low-risk improvements if the cross-CTA/L2 mechanism fails, but
   they are no longer the main route to the external 2k--8k token/s region.
5. In parallel, build a dependency-free native SM87 bulk-attention prototype
   following FlashInfer's online-softmax dataflow and current NHD cache layout.
   Its gate is at least 2x attention-core speedup and at least 1.03x P513 Prefix
   after preprocessing, KV placement, gating, and numerical validation.
6. After the larger workspace boundary exists, screen a native SM87 WY GDN
   prototype. It must compare output and final state against the serial path,
   publish absolute/relative error distributions, preserve generated tokens,
   and show a material full-Prefix gain before integration.

## Whole-chunk screen result

Commit `0196751` completed the first bounded experiment. Three fixed-clock
processes measured M256 at 1.26591x median and M512 at 1.29047x median against
four/eight ordered production M64 launches. All 18 rounds per shape improved.
An M-major single-grid control shows that about 1.27510x of the M512 result
comes from replacing repeated under-filled 40-CTA grids with one continuous
grid; N-major ordering contributes another 1.01153x. The result therefore
selects a whole-chunk scheduling boundary, not an unqualified L2 optimization.

The M512 exhaustive fixture covers all 256 E4M3FN codes in four byte
positions. M-major, N-major, and replay match 2,621,440 baseline BF16 elements
bit-for-bit, including 4,096 classified NaNs, while preserving guards and
inputs. Invalid calls capture zero nodes. The N-major instance uses 70
registers rather than the production CTA's 69, but shared memory remains
23,552 bytes, local memory remains zero, and residency remains three CTA/SM.
This is acceptable for the test-only mechanism screen; production integration
must retain the same occupancy and re-evaluate the address-register cost for
each shape.

The measured output shape alone projects to only 1.01490x P513 Prefix. Applying
the M512 median hypothetically to QKV, Z, and output projects 1.05007x, almost
exactly the 1.05 gate, but one of three processes is below the derived 1.28995x
all-FP8 requirement. The next step is therefore direct dominant-shape screens
followed by one C512 workspace change, not an output-only production promotion.
Full inputs, resources, per-process results, hashes, and claim limits are in the
[machine-readable screen](metadata/qwen36-27b-prefill-fp8-whole-chunk-grid-screen.json).

## NVFP4 down canary result

Commit `5dc256b` completes the first native-NVFP4 canary. The exact down
`[N5120,K17408]` shape compares four/eight production M64 launches with one
M-major grid and one N-major grid. Across three fixed-clock processes and two
scale distributions, M256 reaches 1.29624x median and M512 reaches **1.34655x**
median. All 36 rounds per shape improve. At M512, N-major adds 1.03288x over
the single-grid M-major control, so packed-weight locality contributes more
than it did in the FP8 output screen while grid consolidation remains useful.

Production reports 76 registers/thread; both whole-chunk orders report 79.
All retain 23,552-byte shared memory, zero local memory, and three CTA/SM.
M-major, N-major, and replay are bit-exact to repeated production M64 for
checkpoint-like and same-bank-stress scales at M256/M512. Guards, weights,
scales, and activations remain intact; 17 invalid calls capture zero nodes;
Graph topology is exactly 4/8 baseline nodes versus one candidate node. The
full 60-test Release suite has 51 passes, nine existing skips, and zero fails.

Down alone projects to a 1.04647x P513 Prefix opportunity from the latest
738.322-ms hotspot. Combining it arithmetically with the unintegrated FP8
hypothesis reaches 1.10144x, but this remains a phase-local projection. The
next canary is isolated Gate, followed by the actual main/aux-stream Gate/Up
pair. Direct QKV/Z and native bulk attention remain parallel necessities.
Machine-readable evidence is in the
[NVFP4 down screen](metadata/qwen36-27b-prefill-nvfp4-whole-chunk-down-grid-screen.json).

## External vLLM alignment status

The same Orin has a dedicated Python 3.13 reference environment with PyTorch
2.11.0, vLLM `ccd49f682`, installed FlashInfer 0.6.12, and the exact
`Qwen3.6-27B-NVFP4` checkpoint revision used by native fixtures. Existing
oracle records prove successful Marlin NVFP4/FP8 model loading and exact greedy
outputs, but no retained run proves a FlashInfer attention route or a vLLM
Prefill performance number. The source checkout and installed FlashInfer
versions also differ, so a formal run must pin the installed runtime rather
than mix its AOT binaries with a newer source tree.

The latest native production measurements are 127.249/125.324/119.839 token/s
at P65/P129/P513 under the repository's `(P-1)/Prefix` definition. The user's
2k--8k token/s result is therefore a surface distance of roughly 16--67x, not
a valid cross-framework ratio. Native `Prefix` excludes the final prompt token
and LM head, while standard vLLM throughput/TTFT accounting differs.

Before the OpenAI-compatible API and EvalScope gateway, run an offline matched
matrix using raw token IDs at P65/P129/P257/P513/P1025, batch one, output one,
no prefix cache, no chunked Prefill, no MTP/speculation, BF16 KV/state, and an
explicit FlashInfer attention request. The cross-framework primary metric is
`P / scheduled-to-first-token`; native maps that to complete prompt Prefill,
not the historical `(P-1)/Prefix` metric. Use three warmups, ten measurements,
three independent fixed-clock processes, and native-vLLM-vLLM-native mirrored
ordering. A P65/P513 smoke must first confirm the actual vLLM linear,
attention, and GDN backends. This converts the external target into a real
engineering gap without waiting for the HTTP evaluation adapter.

No MTP, FlashInfer dependency, paged-KV rewrite, generic double/triple
buffering, Prefill Graph, or Prefill/Decode overlap is admitted by this audit.
The Decode anchor remains frozen at 105.870500 ms/token and 9.445501816
token/s.

## Native follow-through after the reference audit

The first two architecture-level native screens now select their mechanisms.
The production-like NVFP4 Gate/Up pair replaces 8/16 public M32 launches per
branch with one N-major whole-chunk grid per branch. Across three fixed-clock
processes, M512 reaches **1.12867x** median versus production, clears the frozen
1.12x gate in all processes, and remains bit-exact with unchanged occupancy.
Its latest phase-local opportunity is 1.03392x P513 Prefix. See the
[Gate/Up pair record](metadata/qwen36-27b-prefill-nvfp4-whole-chunk-gate-up-pair-screen.json).

The dependency-free SM87 bulk causal GQA prototype follows FlashInfer's
online-softmax dataflow without importing FlashInfer. One QT2/BK16 kernel
replaces the decode-style per-token attention loop and tile Gate: C256 reaches
**6.33538x**, C512 reaches **4.53722x**, and Graph topology falls from 769/1537
nodes to one. Its FP32-attention/BF16/Gate/BF16 contract passes the declared
error distribution, append, replay, guard, input, invalid, and resource gates.
The 1.05804x P513 projection is not an achieved Prefix result because runner
integration, preprocessing, and KV placement have not yet been remeasured.
See the [bulk GQA record](metadata/qwen36-27b-prefill-bulk-causal-gqa-screen.json).

The Marlin audit also bounds what whole-chunk scheduling can accomplish. The
current CTA still decodes each weight panel independently for every M64 tile.
The next projection prototype should use a versioned prepacked K16-by-N64
tensor-core layout, a fixed 16-CTA persistent schedule, four-stage `cp.async`,
and register-side E2M1 decode/MMA. Start with exact Down M256 at M64xN128xK64;
advance to N256 and Gate/Up only if it is spill-free, numerically exact, and
materially faster. Even a 3x NVFP4 projection improvement cannot by itself
reach the external target, so FP8 projections, GDN, attention, and launch/global
traffic remain co-equal workstreams.

Commit `ee74ba2` adds the matched offline vLLM probe at
`tools/reference/qwen36_27b_vllm_prefill.py`. It constructs the repository's
raw-token P65/P129/P257/P513/P1025 profiles, enforces batch-one/output-one,
disables prefix cache, chunked Prefill, and speculation/MTP, requests BF16
cache/state plus explicit FlashInfer, verifies the loaded worker backends, and
records trusted engine-core scheduled-to-first-token timing with an explicit
wall-time fallback.

The first retained P65/P513 smoke now passes. The loaded worker reports 16
FlashInfer full-attention layers and 48 GDN layers; runtime logs select
MarlinFP8/NVFP4 linear kernels and Triton/FLA GDN Prefill. P65 measures
275.451 ms / 235.977 prompt token/s and P513 measures 1,237.301 ms / 414.612
prompt token/s; both trusted engine-core spans are within one millisecond of
the enclosing wall call and both produce first token ID 9419. This is only one
warmup and one measurement, after compilation caches were prepared. It proves
the matched route works but does not replace the P65--P1025 three-process
matrix. The native historical `Prefix` boundary also omits the last prompt
token and LM head, so no formal native/vLLM ratio is published yet. See the
[vLLM FlashInfer smoke](metadata/qwen36-27b-vllm-flashinfer-prefill-smoke.json).
