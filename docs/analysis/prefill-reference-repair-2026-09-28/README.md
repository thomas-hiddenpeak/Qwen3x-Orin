---
q3x_document:
  id: q3x-prefill-reference-repair-20260928
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: matched P8192 baseline audit and A/B projection repair evidence
  effective: 2026-09-28
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: new reference arithmetic or production qualification
---

# Prefill reference audit and projection repair

The owner requires completion of numerical repair and production convergence,
including assessment of whether the incumbent is a valid numerical reference.
Legacy equality is a regression check; it is not independent model truth.
This record establishes a bounded repair and baseline assessment, **not a
production-switch or capability pass**. The active continuation belongs to
[Roadmap](../../ROADMAP.md).

## Independent reference

All comparisons use the exact 8192 prompt IDs with SHA-256
`1c62527726da9158c03006c561cfd1958fdf06a56086e6170b69fc49a3e5de70`,
one generated token, no Decode continuation, no prefix cache, and all 248320
logits. The model is the pinned local Qwen3.6-27B-NVFP4 checkpoint. The
read-only installed reference is vLLM 0.27.1 with Torch 2.13.0+cu130,
BF16 activations/KV/Conv, FP32 SSM cache, eager execution, FlashInfer Attention
and Triton/FLA GDN Prefill. This is distinct from the historical short vLLM
0.23 reference; its identity does not transfer to that fixture.

The checkpoint's `mamba_ssm_dtype` is FP32. Both installed Transformers
`modeling_qwen3_5.py` and vLLM's `qwen_gdn_linear_attn.py` retain FP32 recurrent
state across Prefill. A selected BF16 cache policy converts the final state
at the handoff; it does not prescribe per-token Prefill rounding.

Two independent checks assess the reference itself:

1. Capture layer 0's actual `prefill_state_indices=[1]`, normalized Q/K,
   V, gates and zero initial state. Recompute with Transformers' direct FP32
   token recurrence, transposing only the declared K/V state layout. FLA
   output relative L2 is 0.0019052 and final-state relative L2 is 0.0027273.
   No cache slot is selected by finding a favorable numerical match.
2. Replace GDN in **all 48 layers** with Transformers' FP32 chunk64 equations,
   disable TF32, and recompute the complete model. The rest of the independent
   vLLM loading/projection/Attention path stays fixed. Both reference variants
   select token 20960 (`paths`). This checks the GDN algorithm independently;
   it is not an independent implementation of every model operator.

The pure FP32 reference may use cuBLAS internally. It remains reference-only
and supplies no native production dependency or performance claim.

## Concrete A/B projection defect

`bf16_ab_prefill.cu` used `ldmatrix.x2.trans` for canonical `[N,K]` weights,
which already represent the MMA B operand in column-major `[K,N]` order.
The extra transpose permuted K/N coordinates. Replacing it with `ldmatrix.x2`
restores the correct operand mapping.

The independent basis-vector regression selects known K coordinates across
pipeline stages and both projections. Before repair, M64 has 5376 erroneous
outputs out of 6144; M65 and M127 expose the same broken full tile. All three
pass after repair. The fix and regression are committed as `efc7355`.
Earlier graph-topology and same-kernel shape comparisons did not test this
arithmetic property and could not establish correctness.

The corrected candidate additionally applies FP8 and NVFP4 tensor-global
scales after FP32 accumulation, preserving independent Gate/Up scales and
BF16 publication before SiLU. The historical implementation rounded globally
scaled weights to BF16 before the GEMM. These corrections remain separately
identified in admission until complete integration and qualification.

## Matched numerical observations

Relative L2 means `L2(candidate-reference)/L2(reference)`, never
`RMSE(error)/L2(reference)`. KL uses all vocabulary logits in FP64.

| Native/reference route | Layer-0 state relative L2 vs vLLM FLA | Full-logit KL vs independent all-layer FP32 GDN |
| --- | ---: | ---: |
| Legacy | 0.0972508 | 1.57837 |
| Original whole-core v3 | 0.332374 | 4.38283 |
| Scale repair plus Legacy per-token recurrence (diagnostic v4) | 0.0972525 | 1.18354 |
| Scale repair, original fast A/B and GDN (diagnostic v5) | 0.332274 | not recomputed in this table |
| Corrected A/B plus scale repair, fast GDN (diagnostic v6) | 0.0052168 | 0.0080018 |
| vLLM FLA itself | reference | 0.0095687 |

The corrected native full-logit relative L2 versus the independent FP32 model
is 0.0446087; cosine is 0.9990963. Its first-token ordering remains different:
5378 scores 17.5 and 20960 scores 17.375; the independent FP32 reference orders
those two scores oppositely. No exact-logit or exact-token equivalence is
claimed. This bounded result supports correcting the arithmetic and rejecting
blind Legacy alignment; it does not set a post-hoc release tolerance or
substitute for capability and served-context qualification.

The A/B fix reduces native gate differences against the independently captured
reference: log-decay relative L2 falls from 0.31504 to 0.0006934, and beta
from 0.41979 to 0.0005003. Conv inputs were already close. Thus the large
fast-path state error was not explained solely by a recurrence precision choice.

## Evidence and lessons

Raw scripts, frozen ELFs/patches, inputs, complete state/logits and reports:
`.q3x-work/prefill-convergence-20260928/`. Owned model runs and preflights:
`.q3x-work/whole-core-exact-composition-20260927/convergence-*`.
The [machine-readable record](../../metadata/qwen36-27b-prefill-reference-repair-2026-09-28.json)
pins the reports and reference sources. The earlier
[production-switch erratum](../whole-core-production-switch-2026-09-27/README.md)
retains its observed native differences and withdrawal of invalid qualification;
it did not independently prove the incumbent correct.

A further execution error was caught during this work: CMake's server
`OUTPUT_NAME` was mistaken for its target. Make accepted the existing file as
up to date, so the first nominal v6 API run actually executed the old v3 ELF.
That run is explicitly invalidated as v6 evidence. The correct build target is
`qwen3x-eval-server`; the follow-up harness checks `/healthz` profile identity
before submitting a request. Independently rebuilt numerical-capture and basis
regression targets are unaffected. This record makes no corrected API speed
claim; subsequent API evidence must identify its actual binary and report both
Prefill and Decode.
