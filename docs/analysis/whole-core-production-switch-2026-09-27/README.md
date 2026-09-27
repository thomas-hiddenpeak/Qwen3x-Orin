---
q3x_document:
  id: q3x-whole-core-production-switch-20260927
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: dated numerical erratum and production-switch gate disposition
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: matched P8192 Prefill-boundary audit and withdrawal of historical whole-core state qualification
  review_trigger: preserve this record; qualify a repaired successor separately
---

# Whole-core production-switch disposition

**Decision: NO-GO for direct production replacement by the current whole-core
Prefill composition.** This is a concrete numerical failure, not a request
for another repetition of an otherwise qualified performance result. The
ordinary Legacy default is unchanged. The prior whole-core full-state accuracy
qualification is withdrawn, including its use to label the fixed-P40 named
deployment. The historical records remain intact.

The [machine-readable record](../../metadata/qwen36-27b-whole-core-production-switch-2026-09-27.json)
pins the captures, executable, requests, raw comparison hashes, environment,
source bridge and gate statuses. Parent source is `b09759f`; the implementation
adds a matched-boundary capture and correct audit, and prevents a stale runtime
qualification flag from restoring the withdrawn claim. No production numerical
kernel, admission predicate, or default dispatch is changed.

## Historical qualification erratum

The 2026-09-26 record under
`.q3x-work/evidence/whole-core-full-state-qualification-20260926/` cannot
qualify whole-core Prefill for three independent reasons:

1. Its reported KV errors reproduce `RMSE(error) / L2(reference)`. The
   dimension-invariant relative error is
   `L2(error) / L2(reference) = RMSE(error) / RMS(reference)`. The former
   shrinks by `sqrt(number_of_elements)` as the tensor grows. For layer-31
   value KV, the recorded `0.0005054882288403488` becomes **0.7320251483**
   relative L2; the raw tensors reproduce the erroneous value as
   `0.0005054882288403488`. Cosine is **0.7162306919**. This is not a pass at
   a size-independent `1e-3` threshold. The historical Conv aggregate shows
   the same denominator problem. This finding audits recorded numbers; the
   original comparison script is not required or claimed to have been recovered.
2. The old capture dumps recurrent/Conv state at **generation return after
   divergent generated tokens**, not at a common Prefill boundary. Those
   states cannot isolate a Prefill error. Aligning output-step indices after
   greedy divergence also does not make their next inputs equal.
3. Only the last 2048 prompt positions of KV and top-five logits were retained.
   That is not complete KV or a full-vocabulary distribution comparison, and
   coherent output or a few logit ties do not establish capability equivalence.

The later whole-core qualification statements appended to
[ADR-0002](../../decisions/0002-prefill-attention-vllm-numerical-alignment.md)
and the old deployment claims relied on this invalid evidence. This erratum
withdraws that inference without rewriting the accepted Attention decision or
its historical appendix. ADR-0002 permits the specified **Prefill full-Attention**
numerical class; it does not waive GDN, projection, or Decode contracts.

## Fresh matched Prefill boundary

Both routes use one capture executable, the pinned real model and the same
8192 prompt tokens. O1 captures the complete state after the prompt has been
consumed, before any generated token is fed into subsequent Decode. Legacy's
scalar final-prompt step is included in this logical boundary. Every capture
checks `sequence_length == 8192` and retains all 48 GDN states, all 48 Conv
histories, all 16 layers' K and V at **all 8192 positions**, and all 248320
BF16 logits. The full-logit argmax independently reproduces the emitted token.

| Observable | Fresh result |
| --- | ---: |
| Layer-0 GDN relative L2 | 0.3190460999 |
| Layer-0 Conv relative L2 | 0.0014409698 |
| Maximum per-layer GDN relative L2 | 0.9249414855 |
| Maximum per-layer Conv relative L2 | 0.8997630851 |
| Layer-3 key KV relative L2 | 0.2419565830 |
| Maximum per-layer K/V relative L2 | 0.5504970115 |
| Full first-logit relative L2 | 0.7962820770 |
| Full first-logit KL(reference to candidate) | 9.8696063968 |
| Legacy first token | 5378 (`close`) |
| Whole-core first token | 31750 (`operators`) |

Legacy's first and second logits are 19.625 and 18.125 (margin 1.5); the
candidate's are 19.625 and 19.25. This first divergence is not an exact tie.
Layer 0 precedes the first full-Attention layer (layer 3), so the early state
difference cannot be attributed to the accepted full-Attention class.

Source inspection identifies a contract difference: whole-core uses the
prompt-wide WY/chunk path with an FP32 recurrent-state lifetime, whereas
ordinary Prefill uses the exact-span ordered recurrence with BF16 publication
after every token. The
[normative GDN contract](../../PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md#42-per-token-bf16-gdnssm-state)
does not equate these operations. Projection feeds also differ. **This audit
does not assign the entire layer-0 error to one kernel**; identical-input
projection and recurrence isolation is the first repair step. Nor does it
measure a general capability score or establish that Legacy is a framework
ground truth. It decisively rejects inheriting Legacy's numerical contract.

Both processes complete with rc0; observed peak temperatures are 65.937C and
70.125C. Each has preflight, GPU ownership, cache-drop and memory records.
The candidate preflight qualifies CPU-only activity from the active Codex
control process; it is not a release timing record. These instrumented O1
captures establish numerical facts only, with no Decode rate or new API
performance claim.

## Complete switch-gate disposition

| Gate | Disposition | Evidence and consequence |
| --- | --- | --- |
| G0 service coverage | FAIL: incomplete implementation | The retained host panel admits 28/95 cases, rejects 67. Only P64..40000 divisible by 64, O1..4096, token-ID streaming with usage is implemented. Non-C64 MLP tails, P40001..44095 family storage, text/chat and other response surfaces remain missing. This is not 95 GPU tests. |
| G1 Prefill numerics | FAIL / historical qualification withdrawn | Fresh complete P8192/O1 state and logits diverge; the early GDN contract difference is outside ADR-0002. One supported-input failure already prevents universal replacement; no full length sweep is needed to reject this version. |
| G1 Decode numerics | BOUNDED PASS only | Existing exact-v7 identical-input Attention checks, including 240 calls after P8192 and P40000 Prefill, retain their original scope. They do not qualify a different Prefill state or every service/output position. |
| G2 API and lifecycle | PARTIAL; full qualification BLOCKED | Prior P40000/O4096 completion and bounded disconnect/reuse checks remain valid. General surface/length, cancellation and resource coverage depends on G0/G1 repair. |
| G3 performance | Existing P40 observation BELOW interim target; formal qualification BLOCKED | Positive Prefill direction is retained, but composed P40 engine Decode 7.864 is below approximately 8.55 token/s. Mirrored full-panel selection is unexecuted because its prerequisites fail. |
| G4 public capability | NOT QUALIFIED; BLOCKED | The old 20-case Decode smoke is not a complete or long-context Prefill capability comparison. No full suite result is invented. |
| G5 installed default | NOT PROMOTED; BLOCKED | Ordinary Release/OFF rebuild and host tests verify the reporting repair. They are not an installed whole-core default qualification. |

Earlier API observations are reported jointly below. These are unchanged
single-pair engineering observations, not new timings or release-qualified
speedups; output trajectories differ. Sources are the
[P8192 variable-length record](../../metadata/qwen36-27b-whole-core-variable-prefill-2026-09-27.json)
and [P40000 composition record](../../metadata/qwen36-27b-whole-core-exact-decode-composition-2026-09-27.json).

| P / O | Route | Pure Prefill s | Prefill token/s | External TTFT s | Engine Decode token/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 8192 / 256 | Legacy + exact v7 | 33.724 | 242.91 | 33.730 | 9.233 |
| 8192 / 256 | Whole-core + exact v7 | 17.052 | 480.41 | 17.082 | 9.215 |
| 40000 / 256 | Legacy + exact v7 | 219.541 | 182.20 | 219.559 | 7.873 |
| 40000 / 256 | Whole-core + exact v7 | 91.813 | 435.67 | 91.856 | 7.864 |

## Repairs delivered and reproduction

- The existing capture accepts `--capture-boundary prefill` with a P8192/O1
  request; historical O16 mode remains available and explicitly labelled
  `generation_return`. Raw files reside in
  `.q3x-work/production-switch-20260927/`.
- [The audit tool](../../../tools/evaluation/audit_prefill_state.py) verifies
  equal input/coverage and exact pinned payload sizes, rejects non-finite
  values, reports per-layer dimension-invariant errors and full-logit KL, and
  never grants qualification automatically. Repeated-tensor tests prevent
  size dilution; boundary tests reject post-generation and incomplete captures.
- The v10 witness emits `qualified:false` with reason
  `whole-core-state-qualification-withdrawn-2026-09-27`; even a stale caller
  setting `whole_core_production_qualified=true` cannot restore the claim.
  The OpenAI protocol regression test and ordinary Release/OFF build pass.

Run each real-model capture through the required preflight wrapper recorded in
the metadata's `runs[*].identity.command`; do not invoke a fresh GPU process
without the host/device checks. Offline reproduction needs no GPU:

```bash
OPENBLAS_NUM_THREADS=1 python3 -B tools/evaluation/audit_prefill_state.py \
  .q3x-work/production-switch-20260927/prefill-8192-legacy.json \
  .q3x-work/production-switch-20260927/prefill-8192-wholecore.json \
  --output .q3x-work/production-switch-20260927/recomputed-prefill.json
OPENBLAS_NUM_THREADS=1 python3 -B tools/evaluation/audit_prefill_state.py \
  .q3x-work/evidence/whole-core-full-state-qualification-20260926/state-legacy.json \
  .q3x-work/evidence/whole-core-full-state-qualification-20260926/state-wholecore.json \
  --historical --output .q3x-work/production-switch-20260927/recomputed-history.json
OPENBLAS_NUM_THREADS=1 python3 -B -m unittest tests.test_prefill_state_audit
```

The [active Roadmap](../../ROADMAP.md#2026-09-27-product-convergence--active)
owns the bounded numerical-repair successor and its immediate return to the
complete model/API, followed by service coverage and final qualification.
The current version is closed for promotion; the production-switch goal
remains unfinished. No accuracy relaxation or new target is inferred.
