---
q3x_document:
  id: q3x-decode-numerical-repair-20260927
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: FP32 partition and merge-staging repair and its bounded numerical and API results
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: this v3 numerical repair, precision regression, and remaining admission gap
  review_trigger: immutable evidence; correct through a dated successor
---

# Decode numerical repair: FP32 partition and merge state

## Result and scope

Parent `4139551`; owner instruction: advance resolution of the numerical
problem while accepting approximately 8.55 token/s. Work package
`WP-DECODE-FP32-PARTIAL-REPAIR-20260927` under
`AC-DECODE-FUSED-GQA-PRODUCT-20260927` implements one precision correction,
not a speed/tile/threshold scan.

Two avoidable BF16 rounding points in the isolated fused implementation are
removed: partition-output publication and the merge kernel's intermediate
cross-warp staging. The new v3 keeps both in FP32, then publishes the final
Attention output in BF16. A targeted cancellation regression demonstrates the
specific defect. The default scalar executable remains byte-identical.

**This repairs an actual precision loss but does not close whole-model
numerical admission.** The short/mid/40K teacher-forced comparison still
fails the unchanged engineering alarms and exact scalar contract. Local
Attention accuracy improves; whole-model distance is not uniformly better.
The version remains non-installable, unqualified, and not selected for
whole-core composition or production. The implementation and regression test
preserve the corrective finding without promoting a local norm over the
whole-model result.

- [Previous v2 qualification](../decode-qualification-2026-09-27/README.md).
- [Repair protocol](../../metadata/qwen36-27b-decode-numerical-repair-protocol-2026-09-27.json).
- [Results and artifact hashes](../../metadata/qwen36-27b-decode-numerical-repair-2026-09-27.json).
- Raw artifacts: `.q3x-work/decode-numerical-repair-20260927/`.

## Precision ledger and implementation

For partition j, the online kernel computes normalized vector `o_j` and
base-2 log-sum-exp `l_j`. The mathematical merge is the weighted mean of
`o_j` with weights proportional to `2**l_j`.

| Boundary | Rejected v2 | Corrected v3 |
| --- | --- | --- |
| Q/K/V storage | BF16 | unchanged |
| QK accumulation | FP32 Tensor Core accumulation | unchanged |
| Probability operand | BF16 high + BF16 residual | unchanged |
| Online numerator / denominator | FP32 | unchanged |
| Partition vector in global scratch | BF16 | FP32 |
| Cross-warp normalized vector in merge shared memory | BF16 | FP32 |
| Final public Attention output | BF16 RNE | unchanged |
| Partition count / mapping | up to eight, fixed geometry | unchanged |

The second lost-precision point is in vendored
`attention/cascade.cuh::threadblock_sync_state`: the intermediate `st.o`
is cast to the merge input type before reloading and merging the warp states.
Changing only a comment or final output type cannot fix that boundary. The
v3 instantiation uses FP32 for the partition type and merge input/staging.
No vendored source or Prefill template instantiation is edited.

The generic FP32 output store would let multiple KV warps publish the same
merged registers. A uniquely typed local specialization selects the owning
KV warp; the called generic float store has no CTA barrier. Merge shared
storage grows from 8,704 to 16,896 bytes; request scratch grows from 99,072
to 197,376 bytes and still fits the existing request-owned FP32 region.
There is no new allocation, API capacity, KV format, or recurrent-state dtype.
The complete model arithmetic outside this isolated Attention route is held
constant. Compiled resource inspection reports 255 registers and no local
memory/stack for the fused kernel, and 40 registers with no local memory/stack
for the merge. These are build observations, not performance bounds.

The server identifies the route as
`q3x.sm87.admission.fused-decode-split-p.v3` and
`fixed-gqa-tensorcore-split-p-fp32-partials-s512-44095.v3`. Its v3 witness
explicitly says `split-p-fp32-partials-unqualified`. It does not reuse v2's
numerical identity or claim exact fallback semantics. The public scalar
probability-producing API is unchanged.

## Regression and reference checks

The new deterministic cancellation case has zero Q/K and 512 positions.
The first partition contains 128 values of 1 and 128 values of 1.0078125;
the second contains 256 values of -1. All inputs are exactly BF16-representable.
The first partition's mean is 1.00390625, a BF16 midpoint. Premature RNE
publication rounds it to 1, destroying the final residual. The correct final
mean is exactly **0.001953125 (2^-9)**. The test requires that exact BF16 result
for every head and output dimension.

The corrected v3 passes this exact-output case. Linking the archived v2
kernel source with the same new test fails at output element 0 with the
expected regression exit 10. That reconstruction uses the larger allocated
scratch bound but unchanged v2 arithmetic; it is not relabelled as the old
production ELF. The initial negative-control attempt never launched a GPU
process because preflight flagged CPU activity from identified
`systemd-journald` (PID 265); the complete GPU audit was empty. Its record is
retained, the service was not quiesced, and a fresh scoped T1 run passes
preflight before reproducing the expected numerical failure.

The existing nonuniform independent FP64 synthetic test also improves from
relative L2 0.00285951 to 0.00167567. This supports the precision diagnosis,
not a model capability or performance claim. Bounds, tails, non-finites,
workspace/output guards, GQA mapping, and repeated CUDA Graph replay pass.
The v3 protocol serialization test requires the new unqualified identity.

## Full-model results and remaining cause

Use the previous frozen real Agent prefixes P576/P8192/P40000, 16 steps,
scalar token trajectories, all vocabulary logits at every step, and complete
Conv/GDN plus changed K/V spans at first/final steps. All captures use the real
engine and full sidecars. A fresh scalar-forced P576 capture reproduces the
previous scalar raw buffers exactly; the reference did not move with this
repair. Same-input repetition and finite checks remain mandatory.

| Prompt | v2 maximum KL | v3 maximum KL | v2 maximum state-span relative L2 | v3 maximum state-span relative L2 | v3 argmax changes |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 576 | 0.026106 | 0.009860 | 0.139774 | 0.106054 | 0/16 |
| 8192 | 0.006314 | 0.010484 | 0.018668 | 0.016122 | 0/16 |
| 40000 | 0.002447 | 0.001791 | 0.017501 | 0.015477 | 2/16 |

The predeclared alarms remain KL > 0.001 and state-span relative L2 > 0.01.
All three buckets fail both maxima; no tolerance is changed. Maximum ratios
refer to individual captured spans, not the norm of all model state. All
three runs are finite and repeat their same-input Attention outputs exactly.

P576's scalar-trajectory diagnostic has only 1,266 differing BF16 Attention
elements across 256 comparisons of 6,144 elements each (about 0.081%);
the worst comparison differs at 15 elements. Nevertheless, downstream
logits/state leave the old trajectory materially. On P8192, the maximum
logit KL is **worse** than v2 despite the local precision correction. Neither
an improved component norm nor equal greedy argmax can select production.

The code audit explains why FP32 staging does not promise exact scalar
identity: QK uses a different reduction tree, online softmax rescales partial
state instead of the scalar row-wide reduction, high+residual BF16 probability
operands approximate FP32 values, and partitioned PV accumulation/merge
reassociates the scalar position-ordered FMA chain. These remaining differences
can cross BF16 rounding boundaries. The sparse same-input differences followed
by differing downstream state are evidence of propagation, not proof that
one isolated remaining boundary is the sole cause or that task quality regresses.

Independent FP64 QK/softmax/PV checks use actual heads 0/23 at first-step
layers 3/23/43/63. The added mismatch counts use FP64 results converted through
FP32 to BF16 RNE; they are diagnostic, not a new oracle contract. Real-model
errors are now near the final BF16 publication floor on these sampled cells,
but a sample is not a whole-model equivalence proof. No production threshold
is inferred from it. Exact scalar equivalence and model-capability
non-regression remain separately reported requirements.

## Real API return and resource closure

One fresh, unprofiled Legacy API process returns P40000/O256 using the exact
same request hash as the previous v2 long-output run. It completes 256 content
events, exact 40000/256/40256 usage, and `[DONE]`. Observed TTFT is
219.431641 s and Decode TPOT 116.957877 ms, or **8.550087 token/s**. The full
256-token response text equals v2. This is a single-process regression
guardrail, not a new paired performance qualification or a whole-core Prefill
result; the owner's approximately 8.55 token/s acceptance remains intact.

The same v3 server then completes the unchanged 20-question, four-subject,
5-shot direct-answer C-Eval-derived screen. Results are **15/20**, identical
to the frozen scalar and v2 answers in every case; request hashes and usage
match and all responses stop naturally after four tokens. No baseline-correct
answer is lost. This is a bounded regression screen, not a complete official
C-Eval score, long-context capability measurement, or numerical waiver.

Malformed input is rejected with HTTP 400. Closing a P1089/O256 stream after
three content events is followed by a successful same-server recovery with
identical baseline text/usage, natural stop, `[DONE]`, and healthy readiness.
The recovery witness records conservative full reset; shutdown returns 0.

A fresh Release/OFF rebuild remains byte-identical to the scalar API baseline:
`4d1ecc59e0f5f67fd9408105520c0a8bf893454cfbe90941b5e2e4a2fce413e2`.
Symbol inspection finds neither the fused private seam nor cuBLASLt. The 84
raw buffers of the fresh scalar control also match the earlier scalar buffers
by SHA-256, including any signed-zero bits; the ordinary floating-value norm
comparator is not used as the sole proof of byte equality.

Every model process records sanitized preflight, GPU ownership, cache-drop
result, memory, exact executable and bounded cleanup. Cache-drop attempts
succeed. Maximum reported temperature is 77.468 C, including both
slash-separated values in retained telemetry. CPU-only active-control
observations qualify correctness captures; no instrumented capture supplies
model speed. The fresh API guardrail has a passing preflight and begins after
the temporary host builds have ended. Final preflight passes with no GPU
holders. Existing worktrees and user-owned model/environments are unchanged;
all generated evidence stays within `.q3x-work/`.

Final documentation validation and all seven host unit tests pass. A retained
initial documentation-unit invocation failed only because the new registry
entry had not yet been staged; the completed staged-tree checks supersede
that invocation, without rewriting its log.

No installed fused release, whole-core P40000/O256 expansion, composed
Prefill/Decode result, full benchmark certification, or numerical-contract
amendment is claimed. The existing Prefill-only ADR-0002 does not authorize
changed Decode numerics. The remaining product decision must address the
reduction/rounding contract explicitly; more scratch precision alone is not
a sufficient fix, and no claim is made that the accepted speed is impossible
under a lawful successor design.
