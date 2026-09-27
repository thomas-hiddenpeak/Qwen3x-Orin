---
q3x_document:
  id: q3x-decode-qualification-20260927
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: bounded fused Decode numerical and API regression qualification on parent 5038344
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: this qualification protocol, results, failures, and rejection boundary
  review_trigger: immutable evidence; correct through a dated successor
---

# Fused Decode qualification closeout, 2026-09-27

## Decision and product boundary

The owner provisionally accepts approximately **8.55 Decode token/s** and
current Prefill performance. No additional speed search was performed.
The compiled split-P numerical version from parent `5038344` is **not admitted
for production composition**: all three frozen context buckets exceed the
predeclared logit/state divergence alarms, and the existing exact scalar
numerical contract is not preserved. This is a completed negative numerical
qualification, not a slower replacement or an assertion that 8.55 is inadequate.

The default scalar route, the accepted whole-core Prefill profile, GPU
arithmetic, capacity, and production numerical contracts are unchanged. The
fused implementation remains isolated, non-installable admission code. Its
previous real Legacy P40000/O256 result, **8.551773 token/s**, remains valid
only for the artifact and protocol in the
[product-admission record](../decode-product-admission-2026-09-27/README.md).
It is not a new measurement in this batch.

Originating constraint: remove the excessive long-context Decode decline
without degrading capability or state/API semantics. The downward contract is
`WP-DECODE-FUSED-GQA-NUMERICS-20260927` under
`AC-DECODE-FUSED-GQA-PRODUCT-20260927`. The upward milestone would compose the
accepted Prefill and qualified Decode on the real API. The predeclared
numerical stop prevents that composition for this version; no parameter sweep,
threshold relaxation, or promotion follows this result.

Protocols and machine-readable results:

- [Original frozen protocol](../../metadata/qwen36-27b-decode-qualification-protocol-2026-09-27.json).
- [Direct-answer successor protocol](../../metadata/qwen36-27b-decode-capability-direct-protocol-2026-09-27.json).
- [Final result and raw artifact hashes](../../metadata/qwen36-27b-decode-qualification-2026-09-27.json).
- Raw artifacts: `.q3x-work/decode-qualification-20260927/`.

## Numerical method and implementation

Use the pinned real Agent request, model revision
`0893e1606ff3d5f97a441f405d5fc541a6bdf404`, P576/P8192/P40000, and 16 output
positions. In one admission executable, the scalar observer restores the
ordered scalar Attention output on every call. Its greedy tokens are frozen
before the fused arm consumes those same tokens. Prefix observations must
match. No candidate-generated trajectory is mistaken for a same-input oracle.

Capture all 248,320 BF16 logits at each of 16 steps, complete Conv and GDN
state at the first and final steps, and every newly written K/V row in all
16 full-Attention layers. The unchanged prefix is covered by the existing
full-state capture/hash checks. An independent FP64 QK/softmax/PV calculation
uses actual Q/K/V from heads 0 and 23 at layers 3, 23, 43, 63 on the first
step of each trajectory. This diagnoses Attention rounding; it does not define
an independent full-model reference or a capability score.

A private compile-guarded test seam overrides selected next-token IDs **after**
raw logits/state are captured. The controller still checks actual consumed
IDs against the forced trajectory. No weights, activations, logits, or state
are overwritten by teacher forcing. The seam is absent from the ordinary OFF
build and exposes no API, environment, or public selector. Existing normal
capture keeps schema 3; the extended wrapper uses schema 4 and explicitly
labels forced predictions separately from raw-logit argmax.

The initial P576 captures preceded the schema-label addition and retain schema
3. Their CLI and frozen token file identify forcing; their `generated_ids`
are selected inputs, not proof of fused greedy agreement. This metadata erratum
does not change their raw buffers. Later P8192/P40000 captures use schema 4.
The initial alternative input-hook draft was never executed.

Before results, the protocol fixed engineering alarms at Attention relative
L2 > 1/128, full-logit KL(scalar || fused) > 0.001, or captured state-span
relative L2 > 0.01. These are conservative diagnostic stop conditions, **not an
approved production tolerance or proof of task degradation**. The strict
production equality result is reported independently. After P576 failed,
a bounded continuation completed the already frozen length panel, scalar
replay control, and capability distinction with no new kernel variant.

## Numerical results

| Prompt | Maximum full-logit KL | Maximum captured state-span relative L2 | Raw argmax changes / 16 | Numerical disposition |
| ---: | ---: | ---: | ---: | --- |
| 576 | 0.026106 | 0.139774 | 0 | Reject this version |
| 8192 | 0.006314 | 0.018668 | 0 | Reject this version |
| 40000 | 0.002447 | 0.017501 | 2 | Reject this version |

All 48 full-vocabulary logit vectors differ. P576's largest state-span ratio
is a newly written value-cache span, not a claim that the entire model state
has 14% error. Final complete Conv/GDN relative L2 is respectively
0.053305/0.005752 at P576, 0.011091/0.001165 at P8192, and
0.006011/0.001598 at P40000. Offline per-layer localization is retained as
diagnosis, not a retroactive acceptance threshold.

All six primary numerical captures complete with finite tensors and matching
prefix observations/forced input IDs. Their 1,536 same-input Attention
comparisons repeat bit-exactly. The largest local Attention relative L2 is
0.003885, below the gross 1/128 screen, even while downstream logit/state
alarms fail. A local passing norm therefore does not qualify a recurrent
whole-model trajectory.

All 24 sampled FP64 comparisons have greater fused error than scalar error:
fused relative L2 0.001903–0.002495 versus scalar 0.001487–0.001875. The
24 cells include both trajectories and are not independent statistical trials.
They do not support the explanation that the observed difference is merely a
more accurate fused result replacing an inaccurate scalar reference.

At P40000 step 0, the scalar winner 27775 leads token 760 by 0.125; fusion
makes them tie at 16.375, selecting the earlier ID 760. At step 5, the scalar
has an exact 24.375 tie between 3425 and 9017; fusion gives 9017 a 0.125 lead.
These are two argmax changes under the **same inputs**, with small/tied
margins. They do not alone demonstrate lost model capability.

Scalar free-greedy capture versus scalar forced replay is bit-exact in every
captured raw logit/state buffer, validating the teacher-forcing control. A fresh fused P576 process also
repeats every captured raw logit/state buffer bit-exactly, including the
schema-3-to-schema-4 metadata bridge.

## Capability and API closeout

Use the public [C-Eval dataset](https://huggingface.co/datasets/ceval/ceval-exam),
revision `617524a00b307ff6f9933702f724131fe12ca7ce`: first five validation
questions from computer network, business administration, law, and clinical
medicine, with the first five development examples per subject. Selection is
fixed before candidate answers. No validation answer or explanation enters
the prompt. The original EvalScope 1.9.1 template/extractor calibration fails
on the scalar baseline: its first response reaches the 1,024-token cap without
a final answer. That attempt remains a protocol failure, not a 0% score.

A separately frozen successor appends a direct-answer-only instruction to the
same questions and examples. This is a **C-Eval-derived 5-shot direct-answer
regression screen**, not the official default-format/full C-Eval score. Both
arms use the real `/v1/chat/completions` API, greedy decoding, identical
request hashes, the same 1,024-token cap, and native thinking-disabled chat
formatting.

| Check | Scalar | Fused |
| --- | ---: | ---: |
| Complete, parseable, natural-stop answers | 20/20 | 20/20 |
| Correct answers | 15/20 | 15/20 |
| Completion tokens per answer | 4 | 4 |
| Baseline-correct answers lost by fusion | — | 0 |
| Identical answer and usage pairs | 20/20 | 20/20 |

The frozen small-screen capability gate passes. It does not demonstrate
statistical noninferiority, long-context capability, or release qualification,
and does not override the failed numerical contract.

Each arm completes 20 consecutive requests in one server. Malformed
`max_tokens=-1` is rejected with HTTP 400. The first scalar lifecycle script
mistakenly reads `/v1/completions` events using the chat `delta.content`
field instead of `choices.text`; it fails its event-count assertion after the
stream finishes. This is a harness failure, not a server cancellation defect.
The successful 20-question results from that process remain valid; its
original lifecycle failure and clean server exit remain recorded.

The named r2 harness corrects only that field lookup. Fusion passes the full
panel plus lifecycle in one process; scalar receives a separate focused
lifecycle rerun, without repeating the completed capability panel. Each closes
a P1089/O256 Agent stream after three content events, then completes the
first benchmark question with identical baseline text and usage, natural
stop, `[DONE]`, and healthy readiness. The recovery witness explicitly records
`conservative_full`, 44,095 positions, and 2,968,256,512 cleared bytes.
Both servers shut down cleanly with return code 0. These are bounded reuse
and disconnect-recovery checks, not long-running service qualification.

## Verification boundary and engineering lessons

Fresh Release/OFF rebuild passes and produces the **same exact server ELF**
as this batch's baseline API and focused lifecycle runs:
`4d1ecc59e0f5f67fd9408105520c0a8bf893454cfbe90941b5e2e4a2fce413e2`.
Symbol inspection finds no fused teacher-forcing/Attention seam, archived
split scores/values launcher, or cuBLASLt symbol in that default executable.
This exact-artifact bridge closes the ordinary default regression check;
there is no newly installed fused production artifact.

The refreshed fused contract test passes bounds, GQA mapping, tails, guards,
and Graph repeatability at lengths 512, 513, 2047, 8192, 40000, and 44095.
The fused API protocol test also passes. Documentation validation and its
seven host unit tests pass in the final staged tree.

Each model process has its own sanitized Jetson preflight, GPU ownership
inspection, cache-drop attempt, before/after memory, binary identity, telemetry,
and bounded owned cleanup. Some preflights flag only the active Codex CPU
process; those are qualified ordinary correctness observations, not timing or
release results. No control process was quiesced. Maximum retained temperature
across model processes is 76.968 C; no thermal stop occurs. The final host
preflight passes with no GPU device holders. The additional numerical
continuation, including the final repeat, stays below its 20-minute budget.
No elapsed number from instrumented capture is advertised as model speed.

The following prior misconceptions are explicitly prevented by this record:

1. Equal short greedy output is not numerical equivalence: P576/P8192 keep
   all selected argmax IDs yet have different logits and recurrent state.
2. A small Attention-only error is not a bound on downstream model drift.
   Conversely, numerical drift alone is not a measured capability loss.
3. Deterministic repetition rules out one kind of instability; it does not
   establish correctness relative to the required numerical contract.
4. A capped response without a final answer is a failed evaluation protocol,
   not a zero model score. A revised direct-answer format must receive its
   own identity and cannot inherit the official default-format score label.
5. Approximately 90-second whole-core Prefill and Legacy's 8.55 Decode cannot
   be added into a product. The accepted whole-core profile is fixed
   P40000/O16 and disables the complete Decode Gate/Up and Down layout
   inventory required by this admission. Composition needs an explicit
   capacity/layout handoff and real API validation.
6. Owner acceptance of speed does not silently authorize changed numerics.
   No hardware impossibility claim, renewed speed scan, or MTP workaround
   follows a numerical rejection.

Not executed after the numerical stop: whole-core output-capacity expansion,
Prefill/Decode composition, mirrored composed performance selection, full
public benchmark certification, installed fused release, or default promotion.
Those are conditional downstream gates, not passed or silently waived tests.
The current version is closed; reopening requires a concrete successor
numerical design or an explicitly accepted changed numerical contract with
adequate capability evidence. The 20-case screen cannot justify such a
contract by itself.

The Python helpers beside this record preserve the exact local audit workflow,
including failed protocol versions and their explicitly named repairs. They
expect the pinned workspace/model and retained fixture paths and are not a
general-purpose benchmark package. Original raw attempts remain immutable.
