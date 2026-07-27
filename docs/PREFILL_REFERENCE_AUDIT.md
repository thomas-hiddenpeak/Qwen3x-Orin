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

This is source-level compatibility, not a measured Orin result. The current
environment does not contain PyTorch, and the public Python wrapper allocates
temporary/output tensors and uses a JIT/FFI stack. Installing that stack or
calling it from production would violate the scope of this audit. A later
standalone probe must include setup, workspace, stream, KV placement, gate,
and any layout cost rather than timing only the attention kernel.

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
3. If selected, raise the request/workspace boundary incrementally to C256 and
   then C512. Generalize the large-M grid across the dominant Gate/Up, down,
   QKV, Z, and output shapes before spending further effort on isolated M64
   variants.
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

No MTP, FlashInfer dependency, paged-KV rewrite, generic double/triple
buffering, Prefill Graph, or Prefill/Decode overlap is admitted by this audit.
The Decode anchor remains frozen at 105.870500 ms/token and 9.445501816
token/s.
