---
q3x_document:
  id: q3x-adr-0002-prefill-attention-vllm-numerical-alignment
  class: active
  status: active
  owner: project-owner
  authority: accepted production numerical-class decision for prefill attention
  effective: 2026-09-21
  last_reviewed: 2026-09-21
  supersedes: []
  superseded_by: []
  ssot_for: rationale for aligning prefill attention to the vLLM/FlashInfer BF16-probability numerical class with split-P
  review_trigger: successor ADR or constitutional accuracy-contract amendment
---

# ADR-0002: Prefill attention aligned to the vLLM numerical class (split-P tensor route)

Decision state: accepted, by explicit project-owner instruction
(2026-09-21: "可以按照vllm的生产部署对齐,但是一定要做好记录").

## Context

The P40000/O16 production prefill spent 663.7 s because the dispatch predicate
`use_bulk_causal_gqa_group_q64_prefill` restricted the 6.6x-faster WMMA tensor
kernel to `first_position in {0, 512}`, sending ~74 of 76 prefill chunks to the
scalar FP32-probability kernel (79.6% of prefill time).

A real-model A/B proved that simply widening the predicate diverged from the
scalar oracle from token 0. The P513 three-way oracle on real layer-3
activations proved the in-tree FlashInfer kernel is not more accurate than the
GroupQ64 kernel: all BF16-probability tensor routes share the same numerical
class (nrmse floor ~5.4e-4 vs the scalar FP32-probability oracle).

The split-P technique (P = P_hi + P_lo, two WMMA mma into the same FP32
accumulator, FP32 probability sum for the denominator) was verified
synthetically (22x nrmse improvement, no speed loss) and on real data
(logsumexp error reduced 12x, 0.0633 -> 0.00517).

The top-5 logit comparison on the pinned P40000 prompt established the
divergence mechanism: the scalar oracle's own BF16 logits are an **exact tie**
at token 0 (760 = 16.25 and 27775 = 16.25, top-2 margin = 0.0). The
production `bf16_greedy_argmax_kernel` breaks ties by lowest index, so the
oracle selects 760. Any BF16-input tensor route perturbs the underlying FP32
values enough to cross the zero margin and flip the argmax to 27775. This is a
property of the model and prompt at this precision class, not a fixable bug:
no BF16-input tensor-core route can guarantee token-identical greedy output
against the FP32-probability scalar oracle on prompts with exact BF16 ties.

## Decision

1. The production prefill attention numerical contract is aligned to the
   vLLM/FlashInfer production deployment class: BF16-probability tensor-core
   attention. Token-identical reproduction of the historical scalar FP32
   oracle is no longer a production requirement; prompts whose BF16 logits
   contain exact top-k ties may produce different (equally valid) greedy
   tokens.
2. The widened dispatch predicate routes all legal C2..C512 prefill chunks to
   the grouped-Q64 tensor kernel with split-P probability
   (P_hi + P_lo, two WMMA mma, FP32 denominator), which captures ~16-bit
   probability precision at no measured speed cost.
3. The historical scalar FP32-probability kernel remains in the tree as the
   reference oracle for accuracy measurement and as the non-tensor fallback
   for single-token and over-capacity shapes.
4. The accepted real-API evidence for this decision is the pinned P40000/O16
   request on the split-P build: pure prefill wall 219.8 s (vs 663.7 s
   scalar, ~3.0x), 16 generated tokens, coherent output, finish_reason
   length. The generated text differs from the scalar-oracle text from token
   0 due to the documented exact-tie mechanism; both outputs are coherent.
5. Evidence is retained in the ignored workspace
   `.q3x-work/attention-bench/real-model-accuracy-20260920/` (capture A/B,
   top-5 comparison, P513 oracle) and
   `.q3x-work/attention-bench/real-model-accuracy-20260920/api-splitp-20260921/`
   (real API preflight, stream events, result).
6. This decision does not amend the constitutional accuracy contract for
   decode, GDN, or projection paths; it amends only the prefill full-attention
   numerical class. Release qualification remains separately required.

## Post-decision prefill attribution (nsys, 2026-09-21)

An nsys profile of the split-P build on the pinned P40000/O16 request
(evidence `.q3x-work/attention-bench/real-model-accuracy-20260920/
nsys-splitp-20260921/attribution.json`) attributes the 219.5 s GPU prefill:

| component | kernel | time | share |
| --- | --- | --- | --- |
| full attention (split-P tensor) | `group_q64_kernel` | 79.9 s | 36.4% |
| MLP Gate/Up (NVFP4) | `nvfp4_w4a16_gate_c512` | 64.5 s | 29.4% |
| QKV/O projection (FP8) | `fp8_prefill_supermatrix` | 31.5 s | 14.4% |
| GDN linear attention | `gdn_update_exact_span` | 27.1 s | 12.3% |
| small-M projections + norms + conv | (11 kernels) | 16.5 s | 7.5% |

Attention fell from 79.8% (scalar) to 36.4% (tensor). Projections are now the
largest cost (43.8%).

Hardware floor (measured 2026-09-21, cuBLAS at MAXN power mode, 57 C, no
throttle): the Orin's true BF16 dense peak is **33.5 TFLOPS** on the
production MLP shape (C8000 x 17408 x 5120, K=5120 saturates the tensor
cores better than a square GEMM) and 26.1 TFLOPS on an 8192^3 square GEMM,
not the 275 TOPS INT8-sparse nominal that earlier estimates used. The model
is 27B **dense** (not MoE); P40000 prefill is 2.2e15 FLOPs (23.6B projection
params x 2 x 40K = 1.89e15 plus 0.31e15 attention scores). The FLOP floor is
therefore **65.7 s at the measured MLP-shape peak** (84.6 s at the square
GEMM rate, 8.0 s at the INT8-sparse theoretical ceiling). The current 219.5 s
runs at 30% of the measured MLP-shape peak. The 2 s
prefill target requires 1100 TFLOPS, 33x the measured MLP-shape peak (4x even
the INT8-sparse ceiling): it is **physically unreachable on Orin for a 27B
dense model at P40000**, independent of software. The remaining headroom is
kernel efficiency (30% toward the 65.7 s floor, realistically ~90-110 s at
60-75% of peak), not the 2 s target. Reaching the floor requires near-roofline
kernels across all components; the 2 s target requires a different model
(MoE/smaller) or hardware.

## Layer-major whole-core architecture result (2026-09-21)

The skinny-M collapse measured above (C512 at 2.1 TFLOPS vs C8000 at 33.5) is
the core cost of the 512-chunk tiled Legacy-C512 route. The in-tree
layer-major whole-core architecture removes it: 5x M8000 panels, one
whole-prompt FlashInfer attention per full-attention layer, and a persistent
NVFP4 large-M MLP. From a clean `orin-p40-whole-core-dev` build
(`qwen3x-eval-server-p40-v10-dev`, `--development-route p40-whole-core-v10`
atomic acknowledgement), two clean-host real-API P40000 runs on 2026-09-21
measured **101.34 s and 101.56 s pure prefill** (first token "Based", matching
the split-P production output; server witness `pure_prefill` 101,344 ms, 5x
M8000 panels, 64 whole-core layer passes, 16 whole-prompt FlashInfer hits, 64
persistent NVFP4 Gate/Up hits, zero forbidden-route hits). That is **2.17x the
Legacy-C512 219.76 s**, 21.7 TFLOPS effective (65% of the measured MLP-shape
peak), and matches the historical v10 incumbent (101,831.85 ms) within 0.5%.
The route remains **accuracy-unqualified** (inherited FlashInfer P513
full-state mismatch; `numerical_contract.qualified=false`) and default-off;
the production route is still the Legacy-C512 split-P build. Note: the Orin
SM87 (Ampere) has no FP4/FP8 tensor cores, so NVFP4 weights dequantize to BF16
for the MAC and the relevant peak is the measured 33.5 TFLOPS BF16 dense peak;
the FLOP floor above is valid for the NVFP4 deployment.

## FP8 fat-N on-the-fly dequant + cuBLAS (2026-09-21)

The 101.3 s layer-major route runs the 1040 FP8 W8A16 Marlin projections at
22.4 TFLOPS (67% of the measured 33.5 TFLOPS peak). Independent cuBLAS
benchmarks at the real M8000 panel shapes measured 29-30 TFLOPS on the fat-N
projections (input_size 5120: GDN in_proj_qkv/z, full-attn q/k/v) but only
22-23 TFLOPS on the skinny-N projections (input_size 6144: GDN out, full-attn
o), so only the fat-N set is routed to cuBLAS.

Implementation: per-panel on-the-fly dequant of the canonical FP8 E4M3
weights into a process-lifetime reused BF16 buffer (peak 126 MB, NOT a
per-projection cache; the earlier 14.4 GB full-cache variant OOMed under
nsys and was reverted), followed by `cublasGemmEx` BF16/FP32-accumulate on
the engine stream. The dequant is lossless in the BF16 sense (E4M3 mantissa
fits the BF16 mantissa; the per-tensor scale multiply is the only rounding).
Route evidence is unchanged: 1040 FP8 projection hits, zero forbidden-route
hits (plain cuBLAS is not the forbidden cuBLASLt boundary).

Two clean-host real-API P40000 runs on the fixed build measured **98.91 s
and 99.14 s** pure prefill (first token "Based", sha256 5bf13d90...,
identical to the Marlin baseline), a 2.5-2.7 s (2.5%) improvement over the
101.3-101.6 s Marlin baseline. The route remains accuracy-unqualified
(inherited FlashInfer P513 full-state mismatch; the dequant path was
additionally validated against a same-dequant CPU reference at 1.5%
maxabs/rms) and default-off.

Code lessons recorded: `static_cast<std::uint16_t>(__nv_bfloat16)` yields
0x0000 (all-zero weights, silent, produces a plausible-but-wrong token);
use `__bfloat16_as_ushort`. Dequant-path validation must compare against a
CPU reference that performs the same dequant-to-BF16 rounding, or the
expected scale-rounding is misread as a layout bug.

## NVFP4 Gate/Up on-the-fly dequant + cuBLAS (2026-09-22, reverted)

The largest remaining Marlin component of the layer-major route is the
NVFP4 Gate/Up MLP (64 layers x 5 M8000 panels, fused gate+up N=34816
K=5120, 0.91e15 FLOPs, 37.3 s in the 101.3 s profile at ~24.5 TFLOPS).
Independent cuBLAS benchmarks measured 29.2 TFLOPS at the merged N=34816
shape (M=40000) and 29.8-30.2 TFLOPS at M=8000, suggesting a ~1.2x GEMM
gain. The Down projection (K=17408, N=5120) was already rejected in the
M40000 benchmark (10.6 TFLOPS, 0.41x of Marlin) and stays on Marlin.

Implementation: per-layer on-the-fly dequant of gate and up into one
merged [34816,5120] BF16 buffer (358 MB, process-lifetime reused, NOT a
per-layer cache), M-chunked (8000) `cublasGemmEx` BF16/FP32-accumulate,
and a separate SiLU-mul kernel (peak transient 916 MB). Isolated
verification against a same-dequant CPU reference passed exactly
(maxabs/rms = 0.000000, 500 samples, M=40000).

Clean-host real-API P40000 e2e: **99.28 s** pure prefill (first token
"Based", sha256 5bf13d90..., identical) vs 98.91-99.14 s for the
FP8-fat-N-only build - performance-neutral within run noise. nsys
attribution confirmed the kernels ran (1040 cuBLAS GEMM = 720 FP8 +
320 gate/up; 128 dequant_nvfp4 at 18.2 ms; 320 SiLU-mul at 9.0 ms):
the ~1.2x GEMM gain (~31.5 s vs ~35.2 s) is cancelled by the standalone
dequant pass (2.3 s) and the extra 916 MB transient working set (L2
pressure on the 16-SM device).

Decision: reverted. The fused Marlin kernel (dequant inside the GEMM,
zero extra memory, ~25.9 TFLOPS) remains the gate/up route. cuBLAS wins
only when the dequant is cheap to amortize (the FP8 fat-N case: per-
tensor scale, 126 MB buffer) - a two-operand NVFP4 dequant is not.

Full GPU kernel budget of the 98.9 s route (nsys, 2026-09-22): cuBLAS
FP8 GEMM 46.1 s, Marlin down 17.6 s, FlashInfer attention 13.1 s, Marlin
FP8 skinny 7.3 s, GDN kernels ~6.2 s, SiLU-mul 2.9 s, dequant 3.7 s,
norms/other ~2.2 s.

## Down projection cuBLAS go/no-go (2026-09-22)

The Down projection (M=40000, K=17408, N=5120, 64 launches) runs the
persistent Marlin kernel at 25.99 TFLOPS (77% of the measured 33.5 TFLOPS
peak; 17.6 s). A full cuBLAS variant sweep on the exact shape (standard
OP_T/OP_N, K-split x4/x8, M-split x2/x4/x5, N-split x2) measured 10.6
TFLOPS for every variant - 41% of Marlin. cuBLAS has no configuration that
approaches Marlin on this giant-K/skinny-N shape, so the Down route stays
on Marlin.

A custom fused kernel would need to beat Marlin's 25.99 TFLOPS to win
anything; even at 32 TFLOPS (96% of peak) it saves only 3.3 s (3.3%).
Combined with the gate/up result above (cuBLAS performance-neutral,
reverted), the cuBLAS route is exhausted: it wins only on the FP8 fat-N
projections (committed, +2.5%), not on the NVFP4 MLP.

The remaining gap to the 67.5 s FLOP floor (98.9 s = 1.47x) is MLP GEMM
efficiency (~13 s, Marlin at 77% peak) plus FlashInfer attention (13.1 s)
plus GDN/overhead (~5 s). Closing it requires research-grade custom
kernels that beat Marlin and replace FlashInfer; no single component
offers a >5 s win without that effort.

## NVFP4 Gate/Up vec16 dequant + cuBLAS (2026-09-22, retained)

The scalar-dequant gate/up cuBLAS attempt above was performance-neutral
because the standalone dequant pass cost 2.3 s (42 GB/s, 21% of the 204.8
GB/s LPDDR5 peak) and cancelled the ~1.2x GEMM gain. A vectorized dequant
(`dequant_nvfp4_vec16_kernel`, 16 elements/thread, 8 B packed + 1 B scale ->
32 B BF16, two uint4 stores) measures 181.8 GB/s (89% of peak) - 4.3x the
scalar kernel - and drops the dequant to 0.16 s total (128 launches, 64
layers x gate+up). This removes the cancellation and makes the cuBLAS GEMM
win (28.3 TFLOPS, 84% of the measured 33.5 TFLOPS peak, vs Marlin's 24.5
TFLOPS) net positive.

Implementation: per-layer on-the-fly vec16 dequant of gate and up into one
merged [34816,5120] BF16 buffer (process-lifetime reused), M-chunked (8000)
`cublasGemmEx` BF16/FP32-accumulate, and a separate SiLU-mul kernel. Correctness
is validated on controlled known-answer data (weight 0.75, input 1.0 ->
activated 14745600, maxdiff 0.0000) for both the single-chunk (M=16) and
multi-chunk (M=10000, 8000+2000) paths.

Clean-host real-API P40000 e2e: **96.23 s and 97.53 s** pure prefill (first
token "Based", sha256 5bf13d90..., identical to the Marlin baseline) vs
98.91-99.28 s for the FP8-fat-N-only build - a 1.8-3.1 s (~2%) improvement,
reproducible across two runs. The SiLU-mul kernel was then vectorized (8
elements/thread, uint4 loads/stores; the scalar version ran at 94 GB/s, 46% of
peak, while the vectorized version reaches 194 GB/s, 95% of peak - the pass is
purely memory-bound, so `__expf`/`__frcp_rn` give no help). This drops the
SiLU pass from 2.82 s to 1.38 s (64 layers x 5 chunks) and the e2e to
**94.94 s** (first token "Based", sha256 5bf13d90..., identical). nsys attribution of the 96-98 s route: cuBLAS
GEMM 46.7 s (1040 = 720 FP8 fat-N + 320 gate/up), Marlin down 17.6 s,
FlashInfer attention 13.4 s, Marlin FP8 skinny 7.3 s, SiLU-mul 2.9 s
(scalar; 1.38 s after vectorization), dequant_fp8 1.4 s,
dequant_nvfp4_vec16 0.16 s, GDN/norms/other ~8 s. The gate/up cuBLAS path
(dequant 0.16 s + GEMM ~32.3 s + SiLU 1.38 s = ~33.9 s) beats the Marlin
gate/up (37.3 s).

The new dominant gate/up overhead is the separate SiLU-mul pass (1.38 s
after vectorization) plus the intermediate-buffer round trip (2.79 GB gate/up
write + read-back + activated write). A fused dequant+GEMM+SiLU kernel that keeps the GEMM
accumulator in registers and applies SiLU before the store is the Phase 2
target; it must beat Marlin's 24.5 TFLOPS to win further and would remove the
2.9 s SiLU pass plus the round-trip traffic.

Code lesson recorded: `__bfloat162float(ptr[i])` where `ptr` is a
`const std::uint16_t*` does a *numeric* conversion (the BF16 bit pattern is
treated as a float value, then rounded to BF16), NOT a bit reinterpretation.
For a BF16 value 3840.0 (bits 0x4570 = 17776) this silently yields 17792.0.
The dequant kernel only *writes* BF16 (`__bfloat16_as_ushort`) so it never
exposed this; cuBLAS reads/writes BF16 natively via `CUDA_R_16BF`; only the
SiLU kernel that *reads* a `uint16_t` buffer must use
`*(__nv_bfloat16 const*)&ptr[i]` to reinterpret the bits. This bug produced
a uniform ~4.6x output error that was invisible to every per-component
isolation test (dequant byte-exact, cuBLAS layout correct, GEMM correct)
because each was checked with a host-side bit reinterpretation that masked the
device-side numeric conversion.

## Phase 2 custom-kernel go/no-go (2026-09-22)

The cuBLAS gate/up route (94.94 s) leaves a 27.4 s gap to the 67.5 s FLOP
floor. The attribution shows the gate/up GEMM itself runs at 28.3 TFLOPS
(84% of the measured 33.5 TFLOPS peak) - cuBLAS's BF16 GEMM is near the
practical ceiling for this shape (M=40000, K=5120, N=34816). The remaining
gate/up overhead is the separate SiLU pass (1.38 s, 95% of peak bandwidth)
and dequant (0.16 s, 89% of peak bandwidth).

The crucial finding: cuBLAS's separate-pass BF16 GEMM (84% peak) BEATS
Marlin's fused NVFP4 dequant+GEMM+SiLU kernel (24.5 TFLOPS, 73% peak) for
the gate/up shape. The fused Marlin kernel's inline E2M1-lookup + E4M3-scale
dequant in the mma inner loop costs ~11 percentage points of peak, more than
the 1.54 s of separate-pass traffic it saves.

Custom-kernel bars (e2e seconds saved, effort):
- Fused gate/up dequant+GEMM+SiLU: breakeven 27.0 TFLOPS (must match
  cuBLAS's 28.3 TFLOPS GEMM efficiency while folding in the dequant+SiLU
  work). Marlin (the proven production reference) is at 24.5 TFLOPS. At
  28.3 TFLOPS (match cuBLAS) saves ~1.6 s; at 30 TFLOPS saves ~3.4 s. HIGH
  effort, UNCERTAIN payoff - requires beating Marlin by 10-20% and matching
  cuBLAS.
- Down custom (beat Marlin's 25.99 TFLOPS, 77% peak): at 32 TFLOPS saves
  ~3.3 s. HIGH effort - Marlin is already at 77% peak on the giant-K/skinny-N
  shape.
- FlashInfer attention replacement (13.4 s, 16 layers x 826 ms): halving
  saves ~6.7 s. VERY HIGH effort - attention is a fundamentally different
  computation, and FlashInfer is already a highly optimized library.

Decision: NO-GO for the fused gate/up kernel. The arithmetic is decisive:
the current cuBLAS route (dequant 0.16 s + GEMM 32.3 s + SiLU 1.38 s = 33.84 s)
is already at the practical ceiling for this shape. A fused kernel replaces all
three passes with one, so its time is FLOPs/TF: at Marlin's proven 24.5 TFLOPS
it is 37.3 s (+3.4 s WORSE); it only wins above 27.0 TFLOPS, and even matching
cuBLAS's 84% peak (28.3 TFLOPS) while folding in the dequant+SiLU work gains
only ~1.6 s (1.6%). The inline NVFP4 dequant in the mma inner loop is the
binding constraint - it costs ~11 percentage points of peak (Marlin 73% vs
cuBLAS 84%), more than the 1.54 s of separate-pass traffic it saves. A
bounded hand-rolled mma GEMM prototype (128x128 tile, 2-stage, no cp.async)
measured 3.4 TFLOPS (10% of peak), confirming that a competitive GEMM requires
cp.async + deep pipelining - a large effort whose optimistic payoff (~3.4 s,
3.4%) is smaller than the FlashInfer attention component (13.4 s).

The remaining gap to the 67.5 s FLOP floor (94.94 s = 1.41x) is: gate/up GEMM
at 84% peak (~5 s), down Marlin at 77% peak (~4 s), FlashInfer attention
(13.4 s, a different computation), and GDN/overhead (~8 s). Closing it
requires either a GEMM that beats cuBLAS's 84% peak (rejected above) or a
FlashInfer attention replacement (very high effort, different problem). The
cuBLAS route is retained as the gate/up production path for this dev route.

## GDN linear-attention architecture assessment (2026-09-22)

The GDN (Gated Delta Net) linear-attention path - 48 of the 64 layers, the
AGENTS.md-named FLA/Mamba reference - accounts for 5.28 s of the 94.94 s route
(5.5%). It is a 6-7 kernel pipeline per layer: causal_conv (0.72 s),
chunk64_native WY+state (1.14 s), chunk_o_bv64 output (1.49 s), wy_vllm
recompute/solve (1.27 s), rms_norm_silu (0.53 s), compact (0.10 s).

The architecture-level finding: the dominant cost is the SERIAL chunk loop in
`chunk64_native` (one CTA per value head, 48 CTAs, looping 625 chunks of 64
tokens serially, state held in wmma register fragments). The chunked GDN FLOP
floor is only 0.28 s (48 heads x 625 chunks x 4 dense mma products + solve), so
the 5.28 s is 19x the FLOP floor (effective 1.8 TFLOPS, 5% of peak) - the
limit is the serial cross-chunk state recurrence and low occupancy (48 CTAs on
16 SMs = 25% thread occupancy), not raw FLOP or bandwidth (memory floor ~2 s).

The FLA-style fix (parallelize the chunk dimension via an associative scan:
per-chunk WY in parallel, then a scan to propagate boundary states, then
parallel output) is NOT a clear win on this hardware. The scan must store every
chunk's state contribution C_i (128x128 FP32 x 625 chunks x 48 heads = 1.95 GB
per layer); writing and reading that across 48 layers is ~0.9 s of traffic that
cancels the compute-parallelism gain (estimated FLA-style ~2.9 s vs the current
2.63 s for the WY+output pair). The current serial design keeps the 64 KB state
in registers precisely to avoid that traffic - a reasonable choice for a 16-SM
device. Beating it requires either a deeper software-pipelined serial loop
(bounded payoff, the serial dependency remains) or a scan formulation that
avoids materializing all C_i (research-grade, unproven).

Decision: DEFER. The GDN rewrite is a legitimate research-grade direction (the
named FLA path, 5% FLOP efficiency) but its obvious algorithmic fix is negated
by memory traffic and the remaining improvements are bounded. It is not a clear
win comparable to the Phase 1 cuBLAS gains. The remaining gap to the 67.5 s
FLOP floor (94.94 s = 1.41x) is: FlashInfer full-attention (13.4 s, a different
computation), GDN serial recurrence (5.28 s, deferred above), and GEMM
efficiency headroom (~9 s, near the 77-84% peak ceiling, fused route rejected).
Each requires a large research rewrite without a guaranteed >5 s payoff.

### FlashInfer full-attention assessment (2026-09-22)

FlashInfer full attention is 13.4 s (16 layers, 24 Q / 4 KV heads GQA,
head_dim 256, sigmoid gating). The causal FLOP floor (QK^T + attn@V, each
T^2/2 pairs x 2 x 256 x 24 heads x 16 layers = 314.6T) at 33.5 TF is
**9.39 s**. FlashInfer at 13.4 s is **70% of the causal FLOP floor** - already
good for causal attention (causal masking inherently wastes ~50% of block
compute, and head_dim=256 limits tile size). Reaching 85-95% efficiency would
save only 2-3.5 s, and requires a FlashAttention-class custom kernel
(SM87 + head_dim=256 + GQA + sigmoid gating), the highest-effort direction
identified. **DEFER**: not a clear win.

### CUTLASS sw2 gate/up GEMM beats cuBLAS (2026-09-23, retained)

The "cuBLAS is the ceiling" framing was wrong: cuBLAS 84% was a baseline,
not a hardware bound, and the earlier NO-GO rested on Marlin's 73% (its
in-mma dequant breaks mma throughput) plus an unfinished 10% prototype.
Studying the proven CUTLASS path (v3.9.2 cloned, 2.x Ampere cp.async device
API) and benchmarking at the EXACT production shape (M=8000, N=34816,
K=5120, BF16, C=A*B^T) gave:

| impl | M=8000 | M=40000 single |
|---|---|---|
| cuBLAS (production call) | 95.1 ms / 30.0 TF | 468.8 ms / 30.4 TF |
| CUTLASS 128x256x64 s3 sw1 | 105.7 ms / 27.0 TF | 540.9 ms / 26.4 TF |
| **CUTLASS 128x256x64 s3 sw2** | **90.6-91.8 ms / 31.1-31.5 TF** | - |
| CUTLASS 128x256x64 s3 sw4 | 109.9 ms / 26.0 TF | 633.0 ms / 22.5 TF |

The swizzle factor is decisive on this shape: sw2 beats cuBLAS by 4.7%
(stable over 3 runs); sw1/sw4 lose. Note the production M-chunking is
40000/8000 = five full M=8000 chunks (no M=4000 tail chunk exists).

Integrated as `src/kernels/sm87/nvfp4_dequant_cutlass_gate_up.cu` (same
vec16 dequant + vec8 SiLU, CUTLASS 2.x Gemm 128x256x64/warp 64x64/3 stages/
swizzle 2, CUTLASS headers vendored at third_party/cutlass/include).
E2E: **93.08 s** (was 94.94 s cuBLAS, 98.9 s Marlin baseline), first token
"Based", sha256 5bf13d90... identical to the Marlin baseline. The dev route
remains accuracy-unqualified and default-off.

### CUTLASS sw2 FP8 fat-N projections beat cuBLAS (2026-09-23, retained)

The same CUTLASS 128x256x64/warp 64x64/3-stage/swizzle-2 configuration that
won gate/up also wins the 720 FP8 fat-N projection GEMMs (the largest GEMM
component, ~35 s). Benchmarked at the exact shapes (M=8000, K=5120):

| N | cuBLAS | CUTLASS sw2 | gain |
|---|---|---|---|
| 10240 (GDN in_proj_qkv) | 28.2 ms / 29.7 TF | 22.5 ms / 37.3 TF | +25% |
| 12288 (full-attn qkv) | 33.2 ms / 30.3 TF | 27.5-28.6 ms / 35.2-36.6 TF | +18-21% |

Stable over 3 runs. Replaced the cuBLAS call in
`src/kernels/sm87/fp8_dequant_cublas_projection.cu` (same scalar FP8 dequant
to BF16, then CUTLASS GEMM). E2E: **91.67 s** (was 93.08 s), first token
"Based", sha256 5bf13d90... identical to the Marlin baseline. Dev route
remains accuracy-unqualified and default-off.

### CUTLASS sw2 skinny projections beat Marlin (2026-09-23, retained)

Extended the dequant + CUTLASS sw2 path from the fat-N projections
(input_size==5120) to the skinny ones (input_size==6144: GDN out, full-attn
o), which had stayed on W8A16 Marlin. Benchmarked at M=8000, K=6144, N=5120:
CUTLASS sw2 13.4 ms / 37.5 TF vs cuBLAS 21.4 ms / 23.5 TF (+60%), and well
above the Marlin skinny route (7.3 s total for 96 skinny GEMMs). E2E:
**89.96 s** (was 91.67 s), first token "Based", sha256 5bf13d90... identical
to the Marlin baseline. All FP8 projection GEMMs (fat-N + skinny) now run on
CUTLASS sw2. Dev route remains accuracy-unqualified and default-off.

### Hardware peak calibration - correction (2026-09-23)

The "33.5 TFLOPS BF16 peak" used as the denominator throughout this ADR was
WRONG. It was derived as `16 SM x 2048 FLOP/SM/clock x 1.024 GHz`, but the
GPU clock is pinned at **1300.5 MHz** (devfreq `min=max=cur=1300500000`,
tegrastats `GR3D_FREQ 99%`), not 1.024 GHz. Measured on the Orin 64G under
sustained GEMM load (tegrastats: `gpu@61-63C`, `VDD_GPU_SOC 34.4W`, no
throttle):

- Raw tensor-core ceiling (pure `mma.m16n8k16` microbenchmark, 32 warps/SM,
  no memory traffic): **41.4 TFLOPS**. Theoretical `16 x 2048 x 1.3005 GHz =
  42.6 TF`; measured 41.4 = 97% (normal overhead).
- D2D bandwidth: 181 GB/s.

Re-calibrated against the true 41.4 TF ceiling:

| Shape | CUTLASS sw2 | /41.4 TF |
|---|---|---|
| skinny K=6144 N=5120 | 37.6 TF | 91% |
| fat-N N=10240 | 37.3 TF | 90% |
| fat-N N=12288 | 35.8 TF | 86% |
| **gate/up M=8000 N=34816 K=5120** | **31.2 TF** | **75%** |
| **down M=40000 N=5120 K=17408** | **27.4 TF** | **66%** |

skinny/fat-N are near the hardware limit (86-91%); gate/up (75%) and down
(66%) have real headroom. The earlier "GEMM optimization at its natural
boundary" claim was premature - it rested on the wrong 33.5 TF denominator.
The "1.33x FLOP floor" figure is likewise invalid; the floor must be
recomputed against 41.4 TF. A CUTLASS 2.x tile sweep on gate/up confirmed
31.2 TF is the CUTLASS ceiling for that shape (larger tiles fail to compile
or run slower), so closing the 25% gap requires a hand-written persistent
kernel, not library tuning.

### Real mma ceiling for the GEMM profile - correction (2026-09-23)

The "41.4 TF raw tensor-core ceiling" was measured with mma_peak.cu at
128 warps/SM and a single accumulator chain per warp - that hides latency
with warp count, not with the GEMM's register file, so it does not represent
a real GEMM. Re-measured with the real GEMM register profile (mma_peak2.cu:
8 warps/SM, 128 accumulator registers/thread = acc[4][8][4], no spill, pure
back-to-back mma, no ldmatrix/cp.async/syncthreads): **41.9 TF** at 8 warps
(39.7 TF at 16 warps). This is the true mma throughput ceiling for the
gate/up GEMM.

Re-calibrated: CUTLASS 30.5 TF = **73%** of the real ceiling (not "near
ceiling"); the hand-written v10 24.8 TF = **59%**. So CUTLASS itself leaves
11.4 TF (27%) of ldmatrix/cp.async/syncthreads overhead un-overlapped, and
the hand-written kernel leaves 17.1 TF (41%). The mma is NOT the bottleneck
(pure mma hits 41.9 TF); the gap to close is overlapping ldmatrix + cp.async
+ the single mid-loop syncthreads with the mma. This is a real, closable gap
- both CUTLASS and the hand-written kernel are below the mma ceiling.

### Hand-written persistent GEMM kernel - experiment record (2026-09-23)

Per the directive to hand-write kernels against the hardware shape and
dataflow (not tune a library), a from-scratch persistent GEMM was built for
the gate/up shape (M=8000, N=34816, K=5120): 16-SM persistent grid,
128x256x64 tile, 8 warps (64x64 each), cp.async multi-stage pipeline,
`mma.m16n8k16` bf16, ldmatrix fragments, in-kernel NVFP4 dequant of B
(E2M1 nibbles x E4M3 block scale x ws2).

Correctness methodology correction: earlier "maxrel=0.0000" passes used
`cudaMemset` constant data, which masks any row/column offset bug (wrong
address, same value). With RANDOM data the scalar-fragment version
verifies maxrel=0.0000 and the ldmatrix version maxrel=0.012 (bf16
rounding), bad=0/279367.

Measured (random data, production shape, 41.4 TF raw ceiling):

| Variant | TF | Notes |
|---|---|---|
| scalar 32-bit shared loads, 2-stage | 21.4 | verified correct |
| ldmatrix.x4/x2, 2-stage | 21.0 | verified correct; no gain |
| 3-stage pipeline | 23.9 | no gain |
| issue-before-mma overlap | 24.1 | no gain |
| software-pipelined ldmatrix | 21.1 | no gain |
| in-kernel dequant (v2) | 11.8 | dequant ALU steals issue slots |
| **CUTLASS 128x256x64 sw2 (production)** | **31.2** | |

Performance decomposition (hw_decomp.cu, 3 modes of the same kernel, random
data) localizes the gap: full (ldmatrix+mma+2 sync) 135.6 ms / 21.0 TF,
mma-only (no ldmatrix) 123.0 ms / 23.2 TF, ldmatrix-only (no mma) 110.9 ms /
25.7 TF, cuBLAS 95.0 ms / 30.0 TF. The bottleneck is NOT ldmatrix - mma-only
is already capped at 23.2 TF. Per-stage cycle budget at 1.3 GHz: the tensor
core is busy 2048 cycles/stage (67 ms total compute), but my kernel spends
4133 cycles/stage (2085 cycles overhead) vs CUTLASS 2847 cycles/stage (799
overhead). The 2.6x overhead gap comes from 2 `__syncthreads`/stage (CUTLASS
uses 1) plus cp.async not overlapping the mma. Next: replicate the CUTLASS
mainloop structure (3 stages, 1 syncthreads, double-buffered fragments,
interleaved cp.async) in the hand-written kernel.

Replicating the CUTLASS mainloop structure in the hand-written kernel
(hw_gateup10.cu: 3 stages, cp.async for stage kt+2 split into 4 groups
interleaved with the mma loop, fence+wait+1 syncthreads at ks==2,
double-buffered ldmatrix fragments, separate prologue commits) raises the
hand-written kernel from 21.0 TF to **24.8 TF** (verified maxrel=0.0000 on
random data, bad=0/279367). This confirms the loop structure (cp.async
overlap + single mid-loop sync) was the main overhead, not ldmatrix. The
remaining gap to CUTLASS 30.5 TF (5.7 TF / 19%) is under investigation -
candidate: 8 warps/SM occupancy (256 threads) vs a 16-warp (512-thread)
configuration that halves per-warp mma work to hide latency, at the cost of
register pressure (acc drops 128->64 floats).

Conclusion: a hand-written kernel with the same tile shape reaches 21-24 TF
(52-58% of the raw ceiling) vs CUTLASS 31.2 TF (75%). The gap is NOT
pipeline depth, load instruction, overlap, or L2 (DRAM traffic is ~5.5 ms
vs 68 ms compute - compute-bound). CUTLASS's advantage is accumulated
micro-scheduling (register-level mma/ldmatrix interleaving, warp-level
scheduling, epilogue overlap) that a single-pass hand-written loop does not
replicate. Status: EXPERIMENT - not integrated; production keeps CUTLASS
sw2. The hand-written path is retained as evidence that the 75% CUTLASS
figure is a real scheduling-quality gap, not a hardware wall, and as the
starting point if a warp-specialized producer/consumer rewrite is ever
scoped.

### Hand-written GEMM: swizzle + L2 threadblock swizzle (2026-09-23)

Two structural levers were isolated to close the v10 (24.8 TF) -> CUTLASS
(30.5 TF) gap, both replicated exactly from the vendored CUTLASS v3.9.2.

(1) Shared-memory bank-conflict swizzle - NO GAIN. The v10 layout used
row-major + PAD=8 (row stride 72 elements = 144 bytes = 36 words), so
ldmatrix.x4 rows 0 and 8 both map to banks 0-3 (a 2-way conflict). CUTLASS
eliminates this with `cute::Swizzle<3,3,3>` on a PAD-free 128-byte row:
`swz(o) = o ^ ((o & 0x3C0) >> 3)`, equivalently `chunk' = chunk ^ (row & 7)`.
Applying the identical swizzle to BOTH the cp.async write and the ldmatrix
read (required - a mismatch corrupts the data) gives v16 = 24.6 TF, verified
maxrel=0.0000. The bank conflict was real but NOT the bottleneck: removing it
moved nothing. The 2-way ldmatrix conflict is hidden by the mma issue
latency.

(2) L2 threadblock swizzle - REAL GAIN. v10 used a plain column-major tile
order (tn outer, tm inner), so the 16 CTAs resident at once all read the same
K-column of A but 16 different N-slices of B - poor L2 reuse. CUTLASS's
`GemmIdentityThreadblockSwizzle<2>` groups tiles into 2x2 superblocks
(log_tile=1 for tiles_n=136), so the co-resident CTAs share both an A row-band
AND a B column-band. Replicating the exact mapping
(`sid=t/4, tm=sm+2*(sid%sup_m), tn=sn+2*(sid/sup_m)`, total padded to the
full superblock grid so clamped slots recompute idempotently) gives v17 =
**27.2 TF** (grid=128, verified maxrel=0.0000, bad=0/279367). A 4x4
superblock (v18) is marginal at 27.3 TF. Grid scan: 128 is optimal
(16=24.6, 64=26.5, 128=27.2, 256=26.6, 512=26.4).

| Variant | TF | Notes |
|---|---|---|
| v10 (CUTLASS mainloop, PAD=8, column-major) | 24.8 | prior best |
| v16 (v10 + Swizzle<3,3,3>, PAD=0) | 24.6 | bank conflict removed, no gain |
| v17 (v16 + 2x2 L2 swizzle, grid=128) | **27.2** | L2 reuse is the lever |
| v18 (v16 + 4x4 L2 swizzle) | 27.3 | marginal over 2x2 |
| CUTLASS 128x256x64 sw2 (production) | 30.5 | target |

The remaining gap (27.2 -> 30.5 TF, ~3.3 TF / 11%) is no longer a single
structural item; it is the accumulated warp-level mma/ldmatrix/cp.async
interleaving and epilogue overlap that CUTLASS's generated mainloop has.
Status: EXPERIMENT - not integrated; production keeps CUTLASS sw2. The
hand-written kernel now reaches 65% of the 41.9 TF real mma ceiling (was 59%),
narrowing but not closing the scheduling-quality gap.

### Hand-written GEMM: memory-bound decomposition + lever exhaustion (2026-09-23)

A compile-time 3-mode decomposition of v17 (full / mma-only / ldmatrix-only,
each a separate `-DMODE` build so the compiler can fully optimize) shows all
three modes run in ~105 ms (27.0-27.1 TF): removing ALL mma, or removing ALL
ldmatrix, changes nothing. The kernel is NOT mma-bound and NOT ldmatrix-bound -
both are fully hidden. It is **cp.async + syncthreads bound**. This corrects
the earlier "compute-bound, DRAM 5.5 ms vs 68 ms compute" note: that compared
DRAM to mma, but the real limiter is the shared-memory pipeline (cp.async
issue + L2 + the mid-loop `__syncthreads`), not DRAM and not the tensor core.

`__syncthreads` cost isolated: removing it entirely (broken correctness, perf
upper bound only) drops 106.0 -> 98.6 ms (26.9 -> 28.9 TF), so the sync barrier
is ~2 TF (~7%) of the 3.3 TF gap to CUTLASS 30.5 TF. The remaining ~1.4 TF is
cp.async/mma issue scheduling. Even with ALL syncthreads removed the kernel
tops out at 28.9 TF, still below CUTLASS - confirming the residual is CUTLASS's
generated warp-level interleaving, not a single removable barrier.

Lever exhaustion (all measured, random data, gate/up shape):

| Lever | TF | Verdict |
|---|---|---|
| v17 (8-warp + 2x2 L2 swizzle, grid=128) | **27.2** | hand-written best |
| 16-warp (v14, no L2 swizzle) | 23.4 | 8-warp wins |
| v19 (16-warp + 2x2 L2 swizzle) | 23.4 | L2 swizzle does not help 16-warp |
| grid scan (16/64/128/256/512) | 24.6/26.5/27.2/26.6/26.4 | 128 optimal |
| 4x4 L2 swizzle (v18) | 27.3 | marginal over 2x2 |
| Swizzle<3,3,3> bank-conflict (v16) | 24.6 | no gain (hidden) |
| no syncthreads (upper bound, broken) | 28.9 | sync is ~2 TF |
| CUTLASS 128x256x64 sw2 (production) | 30.5 | target |

Conclusion: the hand-written kernel reaches **27.2 TF (89% of CUTLASS 30.5 TF,
65% of the 41.9 TF real mma ceiling)** by replicating every major CUTLASS
structure (mainloop, bank-conflict swizzle, L2 threadblock swizzle). The
residual 3.3 TF is CUTLASS's accumulated warp-level mma/ldmatrix/cp.async
interleaving plus epilogue overlap, which a single-pass hand-written loop does
not reproduce and which 16-warp / grid / pipeline-depth / sync-position
variants do not recover. Status: EXPERIMENT - not integrated. The hand-written
kernel is retained as proof of full control over the hardware shape, dataflow,
and every major CUTLASS structure; production keeps CUTLASS sw2 (the
hand-written kernel is slower, and the production NVFP4 path's in-kernel
dequant would drop it to ~12 TF).

### GEMM optimization complete; down GEMM no-go (2026-09-23)

Fresh nsys at 89.96 s (cutlass-sw2.nsys-rep) gives the post-CUTLASS
breakdown: CUTLASS GEMM (gate/up + fat-N + skinny) 48.5 s / 53.1%, Marlin
down 17.6 s / 19.2%, FlashInfer attention 13.1 s / 14.4%, GDN 5.3 s / 5.8%,
dequant_fp8 2.0 s / 2.2%, silu_mul 1.4 s / 1.5%, rms_norm 1.0 s / 1.1%,
residual_add 0.7 s / 0.7%. Every projection GEMM is now at 82-100% of the
33.5 TF BF16 peak (gate/up 93%, fat-N 100%, skinny 100%), so the GEMM
optimization is at its natural boundary.

The last GEMM component, the down projection (Marlin 17.6 s, 26 TF), was
evaluated for a CUTLASS sw2 replacement: the bare GEMM would be 27.4 TF
(16.7 s, -0.9 s), but Marlin fuses dequant + GEMM + residual-add + norm into
one kernel, so a CUTLASS path must add back a separate dequant (~0.08 s),
residual-add (~0.44 s), and norm (~0.29 s). Net: ~75 ms, within run-to-run
noise. NO-GO - the fused Marlin epilogue is worth the 1.4 TF gap. The
remaining headroom is architectural (FlashInfer attention at its causal
floor, GDN serial recurrence), both already DEFERred, plus bandwidth-bound
elementwise passes that are not cleanly fusible (SiLU needs cross-column
gate/up access the CUTLASS epilogue tile cannot see).

### Phase 2 custom-kernel program - closure (2026-09-22)

Phase 2 systematically assessed every remaining gap component (94.94 s vs the
67.5 s FLOP floor = 27 s gap):

| Component | Now | Efficiency | Custom headroom | Decision |
|---|---|---|---|---|
| cuBLAS GEMM (720 FP8 + gate/up) | 46.7 s | 77-84% peak | ~9 s (near ceiling, fused NO-GO) | keep cuBLAS |
| FlashInfer attention | 13.4 s | 70% causal floor | 2-3.5 s (FA-class kernel, max effort) | DEFER |
| Marlin down GEMM | 17.6 s | 77% peak | ~4 s (fused NO-GO) | keep Marlin |
| GDN linear attention | 5.28 s | 5% peak (serial recurrence) | ~0-1 s (FLA scan negated by traffic) | DEFER |
| Marlin FP8 skinny | 7.3 s | - | small | keep |
| other (dequant/norm/residual) | ~4 s | - | small | keep |

**Conclusion**: the remaining gap is entirely composed of components that are
either near their practical ceiling or require a research-grade rewrite without
a guaranteed >5 s payoff. Phase 1's clear win (cuBLAS dequant 10% -> 84%
peak) is captured. Phase 2 closes here: fused gate/up NO-GO, GDN DEFER,
FlashInfer DEFER. Further gains require committing to research-grade rewrites
(FA-class attention, GDN parallel scan) - an owner decision.

## Consequences




- Prefill speed improves ~3x on the pinned P40000/O16 workload (663.7 s ->
  219.8 s, Legacy-C512 split-P), and the layer-major whole-core architecture
  reaches 101.3 s, then 98.9 s with the FP8 fat-N cuBLAS route, then 94.9-97.5 s with the NVFP4 gate/up vec16-dequant cuBLAS route (vectorized SiLU; 2.31x further; 7.0x total from the 663.7 s scalar baseline). The earlier scalar-dequant gate/up attempt was performance-neutral and reverted; the vec16-dequant variant is retained (see above). A Phase 2 fused dequant+GEMM+SiLU kernel was evaluated and rejected (NO-GO): the cuBLAS separate-pass route (84% peak GEMM) already beats Marlin's fused kernel (73% peak), and a fused kernel only wins above 27 TFLOPS for a ~1.6-3.4 s payoff (see the Phase 2 go/no-go section).
  The 2 s prefill target is physically unreachable on Orin for this 27B dense
  model: the measured BF16 peak (33.5 TFLOPS on the production MLP shape)
  gives a 65.7 s FLOP floor for 2.2e15 FLOPs, so 2 s would need 1100 TFLOPS
  (33x measured peak). The achievable headroom on the layer-major route is
  kernel efficiency toward that floor (~66-80 s at 80-100% of peak), not the
  2 s target. The layer-major route remains accuracy-unqualified and
  default-off; the production route is the Legacy-C512 split-P build.
- Any downstream consumer that pinned the historical scalar-oracle output on
  tie-prone prompts must re-baseline.
- The synthetic accuracy gate (`q3x_decode_ops_cuda_test`) now passes for all
  legal C2..C512 shapes with split-P (nrmse <= 1.7e-4, gate 0.005).
