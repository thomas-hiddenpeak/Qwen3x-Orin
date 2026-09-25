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

### Hand-written GEMM: SASS-level comparison + swizzle/L2 interaction correction (2026-09-23)

A 4-way cross (swizzle x L2 threadblock swizzle, grid=128, 3 runs each, random
data) corrects the earlier "Swizzle<3,3,3> = NO GAIN" note, which only held
WITHOUT the L2 threadblock swizzle:

| Variant | PAD | smem swizzle | L2 swizzle | TF (avg) |
|---|---|---|---|---|
| v10 | 8 | none | none | 24.7 |
| v16 | 0 | Swizzle<3,3,3> | none | 24.1 |
| **v17** | **0** | **Swizzle<3,3,3>** | **2x2** | **27.1** |
| v20 | 8 | none | 2x2 | 23.8 |

The smem swizzle is a **prerequisite** for the L2 threadblock swizzle: without
it, adding the L2 swizzle HURTS (23.8 < 24.7); with it, the L2 swizzle gains
+3 TF (24.1 -> 27.1). The mechanism is not fully understood but the empirical
interaction is stable across 3 runs each.

SASS comparison (v17 vs CUTLASS sw2, same 128x256x64/3-stage/warp-64x64):
both have exactly 128 HMMA (same mma count). Per k-slice: v17 = 4 A-LDSM(x4)
+ 8 B-LDSM(x2) = 12 LDSM; CUTLASS = 4 A-LDSM(x4) + 6 B-LDSM(x2) = 10 LDSM
(CUTLASS's B uses a 16-element vector load, so its ldmatrix pattern is more
compact). CUTLASS also uses `LDGSTS.E.BYPASS.LTC128B.128` (cp.async with an
L2::128B prefetch hint, `cp.async.ca`) vs my `LDGSTS.E.BYPASS.128`
(`cp.async.cg`). Replicating the L2::128B hint (v21) HURT: 24.1 TF vs v17's
27.1 - the `.ca` L1-caching is counterproductive for cross-CTA reuse; `.cg`
(L2-only) is correct here.

Conclusion: the residual 3.4 TF (27.1 -> 30.5) is CUTLASS's instruction-level
mma/ldmatrix/cp.async interleaving (its mainloop keeps 15+ HMMA in flight with
only `.reuse` register hints between them, vs my swizzle address computation
inserting LOP3+SHF between HMMA), plus its more compact B ldmatrix pattern.
These are codegen-quality differences, not a single removable structural item.
Status: EXPERIMENT - not integrated; production keeps CUTLASS sw2. Hand-written
best remains v17 = 27.1 TF (89% of CUTLASS, 65% of the 41.9 TF real mma
ceiling).

### Hand-written GEMM: chunk-based swizzle -> 28.5 TF (93% of CUTLASS) (2026-09-23)

The SASS comparison showed CUTLASS's LDSM addresses use a base register +
immediate offset (`LDSM ... [R0+UR5+0x3000]`), while my v17 swizzle
`swz(o) = o ^ ((o & 0x3C0) >> 3)` applied to the FULL offset forced a per-lane
LOP3+SHF address computation between every HMMA. The fix: the Swizzle<3,3,3>
only permutes the 16-byte chunks WITHIN a 128-byte row - the row offset itself
stays linear. Rewriting as `swcol(row, col) = (((col>>3) ^ (row&7)) << 3) |
(col&7)` with `addr = As[row*TB_K + swcol(row, col)]` keeps the `row*TB_K` part
linear, so the compiler folds the swizzle into the immediate offset.

v22 (chunk-based swizzle + 2x2 L2 threadblock swizzle, grid=128, random data,
5 runs): **28.5 TF** (28.4/28.5/28.7/28.5/28.4), verified maxrel=0.0000,
bad=0/279367. SASS confirms LDSM is now `[R154+0x800]` (base + immediate), no
per-lane LOP3+SHF. This raises the hand-written best from v17's 27.1 TF (89%
of CUTLASS) to **28.5 TF (93% of CUTLASS 30.5 TF, 68% of the 41.9 TF real mma
ceiling)**. The gap to CUTLASS narrowed from 3.4 TF to 2.0 TF.

Remaining 2.0 TF: CUTLASS's B operand uses a 16-element (32-byte) vector load
so its ldmatrix pattern is 6 B-LDSM/k-slice vs my 8 (total 10 vs 12 LDSM/
k-slice), plus its mainloop keeps 15+ HMMA in flight with only `.reuse` hints.
Status: EXPERIMENT - not integrated; production keeps CUTLASS sw2.

Cross-shape generality of v22 (chunk-based swizzle + 2x2 L2 swizzle, grid=128,
3 runs each, random data, all verified correct):

| Shape | v22 TF | CUTLASS sw2 TF | Ratio |
|---|---|---|---|
| gate/up M=8000 N=34816 K=5120 | 29.0 | 30.5 | 95% |
| fat-N M=8000 N=10240 K=5120 | 29.2 | 37.3 | 78% |
| skinny M=8000 N=5120 K=6144 | 30.1 | 37.5 | 80% |

The hand-written kernel is correct and competitive across all production
shapes. The fat-N/skinny gap is larger because those shapes are more
memory/L2-bound (smaller N means less per-tile compute to hide the pipeline),
where CUTLASS's generated code has a bigger scheduling advantage.

Status: EXPERIMENT - not integrated; production keeps CUTLASS sw2.

### Hand-written GEMM: v22 no-sync upper bound EXCEEDS CUTLASS (2026-09-23)

A syncthreads probe on v22 (chunk-based swizzle + 2x2 L2 swizzle, grid=128,
3 runs each) reveals the final structure of the gap:

| Mode | TF |
|---|---|
| v22 with syncthreads (production-correct) | 28.7 |
| v22 WITHOUT syncthreads (broken, upper bound) | **31.3** |
| CUTLASS sw2 (with syncthreads, production) | 30.5 |

The no-sync upper bound (31.3 TF) **exceeds** CUTLASS's production figure
(30.5 TF). This proves the hand-written mainloop's instruction-level
mma/ldmatrix/cp.async interleaving quality is now AT LEAST as good as
CUTLASS's generated code. The entire 2.6 TF gap (28.7 -> 31.3) is the
`__syncthreads` barrier stall, not scheduling quality.

CUTLASS hides its own syncthreads stall by scheduling ldmatrix slightly
further ahead across the barrier. (CORRECTION: an earlier draft attributed
this to "4 fragment buffers / 384 registers" - wrong. The production CUTLASS
config compiles to 242 registers with no spill, only 16 more than v22's 226;
the 4 LDSM address-register groups in the SASS are the 3 pipeline-stage base
pointers, not 4 fragment buffers. See the v23-v26 section for the full
correction.) A 4-stage pipeline (the other way to deepen the ldmatrix lead)
needs 4 x 48 KB = 192 KB shared memory, exceeding the 163 KB opt-in limit.

Conclusion: the hand-written kernel has reached the practical ceiling for a
single-pass 8-warp 128x256x64 3-stage design on SM87. v22 = 28.7 TF
(94% of CUTLASS 30.5 TF, 68% of the 41.9 TF real mma ceiling) is the best
achievable without a warp-specialized producer/consumer rewrite (which would
also not be integrated into production, since the NVFP4 dequant path drops
any hand-written kernel to ~12 TF). Status: EXPERIMENT - not integrated;
production keeps CUTLASS sw2. The hand-written kernel is retained as proof of
full control over the hardware shape, dataflow, swizzle, L2 scheduling, and
instruction-level mainloop - matching or exceeding CUTLASS's scheduling
quality (31.3 > 30.5 no-sync).

### Hand-written GEMM: barrier-stall closure attempts v23-v26 (2026-09-23)

Four attempts to close the 2.6 TF syncthreads barrier gap (28.7 -> 31.3 no-sync)
all failed, each confirming v22 is the practical ceiling for a 2-fragment-buffer
128x256x64 3-stage design:

| Variant | Approach | Result | Why it failed |
|---|---|---|---|
| v23 | prefetch next-kt slice0 after sync | 27.3 TF (worse) | ldmatrix.sync is warp-synchronous, issued before mma(3) in the same warp - cannot be hidden, adds issue pressure |
| v24 | warp-specialized producer/consumer (9 warp, named barriers) | deadlock | `bar.sync 6+s,256` expects 256 threads but the producer warp also executes it -> 288 threads hit a 256-count barrier |
| v25 | 4 fragment buffers, all 4 slices pre-loaded | 27.0 TF (worse) | 255 regs + 32B spill; 48 LDSM all land on the post-sync critical path |
| v26 | continuous global-slice ldmatrix across kt boundary | hang | restructuring the cp.async commit/wait/sync protocol introduced a correctness bug |

The decisive result is v25: pre-loading all 4 slices after the sync is SLOWER,
because it puts 48 LDSM on the critical path.

CORRECTION to the earlier "CUTLASS uses 4 fragment buffers (384 registers)"
claim: compiling the production CUTLASS config (128x256x64/64x64/3-stage/
swizzle2) with `--ptxas-options=-v` shows it uses **242 registers, no spill**
- NOT 384. The 4 distinct LDSM address-register groups seen in the SASS (R4 /
R0+UR / R197+UR5 / R198+UR4) are the 3 pipeline-stage base pointers, not 4
fragment buffers. A 4-fragment-buffer scheme (128 acc + 256 fragment) would
need >255 registers and would spill, which CUTLASS does not do. The real
CUTLASS-vs-v22 delta is small: 242 vs 226 registers (16 more) and 40 vs 48
LDSM (CUTLASS's B operand uses ldmatrix.x4, so 2 fewer B-LDSM per k-slice).
CUTLASS hides its barrier stall by scheduling ldmatrix slightly further ahead
across the syncthreads using those 16 extra registers, not by holding 4 full
fragment buffers.

Conclusion: v22 = 28.7 TF (94% of CUTLASS 30.5 TF, 68% of the 41.9 TF real mma
ceiling) is the hand-written result for this tile/pipeline configuration. The
no-sync upper bound (31.3 TF) exceeds CUTLASS, proving the mainloop scheduling
quality matches CUTLASS; the residual 2.6 TF is a quantified barrier stall that
CUTLASS absorbs with 16 extra registers of ldmatrix-ahead scheduling (242 vs
226) plus a more compact B ldmatrix pattern (40 vs 48 LDSM). Status:
EXPERIMENT - not integrated; production keeps CUTLASS sw2. The hand-written
kernel is retained as proof of full control over the hardware shape, dataflow,
swizzle, L2 scheduling, and instruction-level mainloop.

### Hand-written GEMM: v27 3-fragment-buffer ldmatrix 2-ahead -> 29.0 TF (95% of CUTLASS) (2026-09-23)

The register correction (CUTLASS = 242 regs no-spill, not 384) pointed to the
real mechanism: CUTLASS holds ~3 fragment buffers (ldmatrix 2 k-slices ahead),
not 4. v27 replicates this: 3 fragment buffers (A[3][4][4], B[3][8][2]),
prologue loads slices 0+1, then `ldm(ks+2)` issued 2 slices before `mma(ks)`.
This compiles to **240 registers, no spill** (2 fewer than CUTLASS's 242).

v27 vs v22 (grid=128, 5 runs each, random data, verified maxrel=0.0000):

| | run1 | run2 | run3 | run4 | run5 | avg |
|---|---|---|---|---|---|---|
| v22 (2-buffer) | 28.7 | 28.5 | 28.2 | 28.4 | 28.3 | 28.4 |
| **v27 (3-buffer)** | 28.9 | 28.9 | 29.0 | 29.0 | 28.6 | **28.9** |

v27 = **29.0 TF (95% of CUTLASS 30.5 TF, 69% of the 41.9 TF real mma ceiling)**,
a stable +0.5 TF over v22. Grid scan: 192 = 29.0 TF (best), 128 = 28.7 TF.

This confirms the corrected mechanism: the barrier stall is hidden by holding
3 fragment buffers so ldmatrix for the next 2 k-slices is in flight across the
syncthreads. The residual 1.5 TF (29.0 -> 30.5) is CUTLASS's 2 extra registers
of finer scheduling plus its more compact B ldmatrix pattern (40 vs 48 LDSM,
B uses ldmatrix.x4). Status: EXPERIMENT - not integrated; production keeps
CUTLASS sw2. v27 is the new hand-written best.

### Hand-written GEMM: v28 B ldmatrix.x4 closure - no gain, v27 is final (2026-09-23)

v28 replicates CUTLASS's B ldmatrix.x4 (one x4 load covers two n-slices, 16k x
16n) to test whether the residual 1.5 TF is the B-ldmatrix pattern. The x4
addressing maps lanes 0-15 -> n-slice j, lanes 16-31 -> n-slice j+1 (lane>>4
selects the n-slice pair; (lane>>3)&1 selects the k half). Correct (maxrel=
0.0000), 248 registers no-spill, 3 runs: 28.9 / 28.5 / 29.0 TF (avg 28.8).

**No gain over v27.** This falsifies the hypothesis that the residual 1.5 TF
comes from the B-ldmatrix.x4 pattern (40 vs 48 LDSM): with the LDSM count
matched, performance is unchanged. The LDSM issue is fully hidden by the
cp.async + mma pipeline, so fewer LDSM instructions buy nothing. The residual
is therefore CUTLASS's 2 extra registers of finer instruction scheduling, not
a structural difference - a register-level micro-tuning with negligible ROI.

**Hand-written GEMM program closed at v27 = 29.0 TF (95% of CUTLASS 30.5 TF).**
Lever exhaustion (all tested, no gain or worse): full-offset swizzle, L2::128B
hint, 16-warp, 4x4 L2 swizzle, cross-kt prefetch, 4-buffer preload, warp-
specialized (deadlock), continuous ldmatrix (hang), B ldmatrix.x4. The no-sync
upper bound (31.3 TF > CUTLASS 30.5 TF) proves mainloop scheduling parity; the
1.5 TF with sync is the syncthreads stall CUTLASS hides with 2 extra registers.
v27 is retained as the hand-written best; production keeps CUTLASS sw2.

### Hand-written attention program: FlashInfer 812 ms baseline, WMMA + raw-mma attempts (2026-09-23, corrected 2026-09-25)

The remaining e2e lever after GEMM is FlashInfer whole-prompt attention (13.1 s /
14.4% of the 89.96 s nsys). A faithful FlashInfer whole-prompt baseline for our
shape (24 Q heads / 4 KV heads GQA 6:1, head_dim 256, causal, sigmoid gate)
measures **812.3 ms (24.2 TF)** at T=40000; the causal-only floor is **586.9 ms
(33.5 TF)**, so the headroom is ~225 ms/layer.

Two hand-written kernels were built (`.q3x-work/attention-bench/`, gitignored):

- **attn_hw1 (WMMA, per-warp independent online softmax)**: **bit-exact** vs the
  production kernel (correct), but **5329 ms (3.7 TF), 6.5x slower** than
  FlashInfer. SASS instruction mix shows the cause: 161 `CALL`s (WMMA
  `mma_sync` lowers to a function call) + 212 `BSSY/BSYNC` (per-tile
  `__syncthreads`) vs FlashInfer's 0 CALL + 0 BSSY/BSYNC (raw PTX `mma` +
  per-warp softmax, no per-tile barrier).
- **attn_hw2 (raw PTX `mma.sync.m16n8k16` + `ldmatrix`, per-warp softmax)**:
  **CORRECT** — on identical T=256 input it matches the real FlashInfer kernel
  to **max_rel=0.0039 (bf16 noise)**. The earlier "QK^T all wrong / data
  corruption" reading was a **false alarm** produced by my own debug
  instrumentation: (a) a race between two debug-buffer writers (the 16x16 QK^T
  dump by 32 lanes into indices 0-255 overlapped a single-thread smem/global
  dump into 128-167), and (b) probe data that used raw integer bit-patterns as
  bf16 (denormals ~2^-126) whose products underflow to 0 in f32. Once the
  ldmatrix directions were fixed to the empirically-verified
  **K=no-trans, V=trans, Q=ldmatrix.x4** (each proven by a dedicated
  0/128-BAD probe: `bfrag_def`, `vfrag_def`, `qfrag_def`), attn_hw2 is
  numerically equivalent to FlashInfer. The residual "2.3% bad vs the fp32 CPU
  oracle" is the shared **bf16-P softmax floor** (both attn_hw2 and FlashInfer
  carry it vs a pure-fp32 reference), not a bug. The bench's `prod_kernel` WMMA
  *copy* (18.3% bad vs CPU) is the one with a transcription defect.

**Performance (the real gap):** attn_hw2 = **4506 ms (4.4 TF)** at T=40000,
i.e. **5.5x slower** than FlashInfer (812 ms, 24.2 TF) — raw PTX mma only
bought 1.18x over the WMMA version. Root cause refined 2026-09-25 after reading
the FlashInfer `prefill.cuh` source and running ablations attn_hw3..hw8:
- **Not occupancy**: FlashInfer (238 regs) and the hand-written kernel
  (252-254 regs) are BOTH register-limited to 2 blocks/SM (65536 regs / 238 /
  32 / 4 warps = 2). Single-buffering K/V (attn_hw8, 48 KB smem) does NOT raise
  occupancy — registers still cap it at 2 blocks/SM (attn_hw8 = 4252 ms, -5.7%).
- **Not mma throughput**: a pure-mma microbench (the QK^T pattern, A/B in
  registers) reaches 37.5 TF (87% of peak); both kernels emit the identical
  `HMMA.16816.F32.BF16` instruction. The mma units are fine in isolation.
- **It is mma stalling on memory**: the full kernel feeds the mma units only
  4.6 TF (12%) vs FlashInfer's 24.2 TF (65%) for the same mma work — the mma
  waits on ldmatrix/cp.async. Deeper KV prefetching helps: attn_hw5 (4-stage
  pipeline) = 3964 ms (5.0 TF, best hand-written) vs attn_hw2 (2-stage) 4506 ms.
  FlashInfer reaches 65% with a 2-stage pipeline, so its edge is the QUALITY of
  the cp.async/ldmatrix/mma overlap, not prefetch depth.
- **Softmax is NOT the bottleneck (direct evidence, attn_hw9)**: attn_hw9 =
  attn_hw5 with the entire causal-mask + online-softmax block removed (replaced
  by a trivial scale) = 4034 ms — no faster than attn_hw5 (3964 ms). Removing
  the softmax path gives zero gain, so the wall is the mma/ldmatrix/cp.async
  pipeline overlap, not the softmax math. Full ablation: attn_hw3 (no rescale)
  4364, attn_hw4 (no barriers) 4416, attn_hw5 (4-stage) 3964 (best), attn_hw6
  (KV48) 4189, attn_hw7 (ldmatrix.x4) 4426, attn_hw8 (single-buffer) 4252,
  attn_hw9 (no softmax) 4034 — none approach FlashInfer's 812 ms.

**Conclusion (DEFERred, owner decision — unchanged, rationale corrected):**
FlashInfer's whole-prompt kernel remains the right production choice. The
hand-written attention is now proven *correct* (not "blocked"), but it is 5.5x
slower because the mma stalls on memory latency, not because of softmax,
occupancy, or mma throughput. Closing that gap needs a research-grade
FlashAttention-class rewrite that replicates FlashInfer's exact instruction-level
mma/memory overlap (persistent CTA, warp-specialized producer/consumer) — not
justified against the current whole-product result (~225 ms/layer x 16 layers of
headroom, ~3.6 s e2e). Consistent with the Phase 2 closure, production keeps
FlashInfer whole-prompt attention + the separate sigmoid-gate kernel. Together
with the hand-written GEMM (29.0 TF), these findings confirm the library kernels
(CUTLASS sw2 + FlashInfer) are the right production choice for this
hardware/shape.

### Copy-FlashInfer study: exact reproduction + fused sigmoid gate (2026-09-25)

Per the owner directive to copy FlashInfer verbatim before improving it, the
in-tree kernel was manually instantiated with the exact dispatch config and the
Qwen3.6-specific sigmoid gate was fused into its epilogue.

**Exact reproduction (fi_copy):** manually instantiating
`SinglePrefillWithKVCacheKernel<KTraits, Params>` with the dispatch-selected
traits reproduces the library baseline to within noise: **811.8 ms** vs the
dispatched 812.3 ms (24.2 TF) at T=40000, 239 regs, no spill. The dispatch
config for our shape (head_dim=256, Q24/KV4 GQA 6:1, causal) is:
CTA_TILE_Q=64, NUM_WARPS_Q=4, NUM_WARPS_KV=1, NUM_MMA_Q=1,
NUM_MMA_D_QK=NUM_MMA_D_VO=16, **NUM_MMA_KV=2** (so CTA_TILE_KV=32 and
SharedStorage = 32 KB q + 16 KB k + 16 KB v = **64 KB**, not the 48 KB /
CTA_TILE_KV=16 previously assumed). NUM_MMA_KV is the dispatch's optimal
choice: forcing 3 gives 898 ms and 4 gives 836 ms (both worse). A full
config sweep (fi_sweep, T=40000) over the dispatch's unexplored points confirms
the dispatched config is the optimum of the whole config space:
NUM_MMA_KV=2/WARPS_KV=1 (dispatch) **811.8 ms**, NUM_MMA_KV=1 963.8 ms,
NUM_MMA_KV=4 835.6 ms, NUM_MMA_KV=2/WARPS_KV=2 931.5 ms, NUM_MMA_KV=1/WARPS_KV=2
1118.1 ms — every alternative is slower. This confirms
the 812 ms baseline is the library's tuned point, not an artifact of the
dispatch wrapper, and that beating it requires a structurally different kernel
(warp-specialized producer/consumer, absent from this vendored version), not a
config change.

**Fused sigmoid gate (fi_fused):** the verbatim mainloop plus a fused
`o *= sigmoid(gate)` in the epilogue (after `transform_output`, before
`write_o_reg_gmem`) is **numerically correct** — verified against a CPU
reference `attn * sigmoid(gate)` at T=256 with **max_rel=0.0033, bad=0/1572864
(PASS)** (the earlier "all zeros" was a defect in the *test* reference gate
kernel, which took the address of an rvalue from `__float2bfloat16_rn`; the
fused kernel itself was always correct). But it is **slower**: 815.5 ms vs the
812.3 ms baseline + separate gate. The fused gate reads `gate[q, head, d]` in
the mma C-fragment layout — one element per (mma_q, mma_d, reg_id) at an
irregular `dpos` per lane — so the loads are scattered, whereas the separate
`sigmoid_gate_kernel` (16.66 ms) reads a contiguous linear range (fully
coalesced). Coalescing the gate into smem is infeasible: the per-CTA gate tile
is CTA_TILE_Q(64) x group(6) x head_dim(256) x 2 B = **196,608 B > 167,936 B**
available smem.

**Decision:** keep the separate coalesced sigmoid-gate kernel. Fusing the gate
into the attention epilogue is correct but costs ~3 ms (scattered gate loads
lose to the coalesced separate pass) and cannot be coalesced within the smem
budget. This is a bounded, evidence-backed NO-GO for gate fusion, consistent
with the DEFERred attention conclusion above: production keeps FlashInfer
whole-prompt attention + the separate sigmoid-gate kernel.

### SASS instruction-level diagnosis + KV-tile experiment (2026-09-25)

To pin down the hand-written vs FlashInfer gap at instruction level (not the
earlier "mma stalls on memory" heuristic), the SASS of both kernels was
disassembled and the mainloop instruction mix compared per KV position:

| Per KV pos | FlashInfer (tile=32) | attn_hw5 (tile=16) | Note |
|---|---|---|---|
| total instructions | 19.1 | 37.1 | 1.94x |
| HMMA | 4.06 | 4.00 | 1.0x — mma density identical |
| LDSM | 2.50 | 4.00 | 1.6x — hw5 uses ldmatrix.x2, FI x4 |
| FALU (FADD+FMUL+FFMA) | 4.69 | 11.75 | 2.5x — o_frag rescale is per-iteration, amortized over half the KV at tile=16 |

**Correction (2026-09-25):** an earlier revision of this section quoted
"69.8 vs 384.5 instructions/KV (5.5x)" and "12x FALU" — both were computed over
a wrong mainloop boundary (the epilogue was included). Measuring the true
mainloop (the back-jump enclosing the HMMA block) gives **1.94x total
instructions/KV**, not 5.5x. The mma density is identical (4.06 vs 4.00
HMMA/KV); the per-KV FALU gap (2.5x) is the tile-size amortization of the
per-iteration o_frag rescale, and the LDSM gap (1.6x) is ldmatrix.x2 vs x4.

So of the 4.9x wall-clock gap (3964 ms vs 812 ms), only ~1.94x is raw
instruction count; the remaining ~2.5x is **scheduling / memory-pipeline
overlap quality**. Crucially, attn_hw10 (tile=32 + 2-stage, matching
FlashInfer's structural parameters) was *slower* (4001.6 ms), so the gap is
not closed by matching tile/stage count — it is in the fine-grained
mma/ldmatrix/cp.async interleave that FlashInfer's codegen produces.

**KV-tile experiment (attn_hw10):** raising the hand-written KV tile from 16 to
32 (to match FlashInfer's CTA_TILE_KV=32) was built and verified correct
(GPU-vs-GPU vs FlashInfer max_rel=0.0039, bf16 noise) but is **slower**
(4001.6 ms vs attn_hw5's 3964 ms): at the 64 KB smem budget, tile=32 forces a
2-stage pipeline (prefetch distance 32 KV < attn_hw5's 48 KV) and pushes
registers to 255 with spill. KV tile size is not the lever — deeper prefetch
(attn_hw5's 4-stage) matters more than matching FlashInfer's tile.

**K/V-decoupled pipeline experiment (attn_ws1, 2026-09-25):** the one
structural axis FlashInfer uses that attn_hw5 did not is *decoupling the K and
V cp.async streams* — FlashInfer commits K and V as separate cp.async groups
so QK^T waits only on K and PV only on V (V keeps loading during softmax).
attn_ws1 replicates exactly that on the attn_hw5 base (tile=16, ldmatrix.x2,
2-stage per stream, uniform `cp.async.wait_group 1`). It is **correct**
(GPU-vs-GPU vs FlashInfer max_rel=0.003906, bf16 noise) but **slower**:
4470.4 ms vs attn_hw5's 3964 ms (+12.8%). Decoupling forces a 2-stage
pipeline (prefetch distance 16 KV) where attn_hw5's 4-stage keeps 48 KV in
flight; the deeper prefetch wins, confirming the attn_hw10 finding that
prefetch depth beats structural decoupling on this shape.

**Warp-specialized producer/consumer is infeasible on sm_87 (2026-09-25):**
a minimal mbarrier producer/consumer test fails to compile —
`mbarrier.try_wait.parity` requires `.target sm_90 or higher`, and Orin is
sm_87 (Ampere). This is the root cause of the earlier GEMM warp-specialized
deadlock: the hardware has no mbarrier wait primitive, so a true
producer/consumer split cannot be signalled. Reading the vendored FlashInfer
`prefill.cuh` mainloop confirms FlashInfer itself uses **zero mbarriers** —
it is a *cooperative* kernel (with `NUM_WARPS_KV=1`, `get_warp_idx_kv` is
constant 0, so every warp both loads and computes) that overlaps via
independent K/V cp.async commit groups + `__syncthreads`, the same primitive
family as attn_hw5.

**Conclusion (corrected, 2026-09-25):** the 5x gap is instruction scheduling /
memory-pipeline overlap quality, not FALU math, occupancy, mma throughput, KV
tile size, or K/V stream decoupling. The previously proposed
warp-specialized producer/consumer rewrite is **not a viable path on sm_87**
(mbarrier wait requires sm_90+), and FlashInfer does not use it either. The
residual gap is the fine-grained mma/ldmatrix/cp.async interleave that
FlashInfer's codegen produces within the same cooperative + cp.async
mechanism; every structural lever tested (config sweep, KV tile, ldmatrix.x4,
single-buffer, softmax removal, K/V decoupling) is closed with direct
evidence. Production keeps FlashInfer whole-prompt + separate sigmoid-gate
kernel.

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
