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

## Consequences

- Prefill speed improves ~3x on the pinned P40000/O16 workload (663.7 s ->
  219.8 s, Legacy-C512 split-P), and the layer-major whole-core architecture
  reaches 101.3 s (2.17x further; 6.5x total from the 663.7 s scalar baseline).
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
