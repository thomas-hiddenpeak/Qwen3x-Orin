---
q3x_document:
  id: q3x-adr-0003-prefill-target-hardware-bound
  class: active
  status: active
  owner: project-owner
  authority: proposed hardware-bound analysis of the locked Prefill targets; requests owner target adjudication
  effective: 2026-09-26
  last_reviewed: 2026-09-26
  supersedes: []
  superseded_by: []
  ssot_for: hardware-bound reconciliation of the locked Prefill targets against the measured Orin SM87 ceiling
  review_trigger: owner target change, new measured hardware ceiling, or model-family change
---

# ADR-0003: Prefill target hardware-bound analysis and target adjudication request

Decision state: proposed. This ADR records a hardware-bound analysis and
requests an owner decision. It changes no target, no route, and no priority.
Per the Constitution, a target cannot be declared impossible from the
limitations of the current implementation; this record therefore supplies the
required same-workload hardware bound, matched profile evidence, and
reconciliation with the owner's own vLLM reference before asking for
adjudication.

## Context

The Constitution (section 8, locked business targets) requires:

- a cold/no-cache 40K-60K-token Agent prompt to reach first response in at
  most 2 seconds;
- a cold/no-cache prompt of about 130K tokens to reach first response in at
  most 4 seconds;
- single-request Decode of at least 10 token/s;
- no production accuracy regression.

The same section records the owner's vLLM reference: an `Avg prompt
throughput` log of about 4.3K token/s on the target Orin and model family,
described as supporting telemetry that does not by itself establish a single
request's Prefill latency.

Two production deployment profiles are now qualified and measured on the real
model through the OpenAI-compatible API (comprehensive e2e evaluation,
2026-09-26, `e2c975f`):

| Route | P40000/O16 cold TTFT | Prompt tok/s | Decode tok/s |
| --- | ---: | ---: | ---: |
| Legacy-C512 (default), warmed | 219.5 s | 179.09 | 3.94 |
| Legacy-C512 (default), first request after start (2026-09-09) | 663.7 s | 60.27 | 3.94 |
| Whole-core `p40-whole-core-v1` | 91.4 s | 419.75 | 3.87 |

The whole-core route is the best measured Prefill result on this hardware and
model. The question this ADR answers is whether the locked 2 s / 4 s Prefill
targets are reachable on the Orin AGX for the pinned 27B dense model family,
independent of software quality.

## Measured hardware ceiling (corrected values)

The 2026-09-23 calibration in ADR-0002 corrected the earlier 33.5 TFLOPS
denominator, which used a 1.024 GHz clock instead of the pinned 1300.5 MHz.
The values used below are the corrected, measured ceilings:

- raw tensor-core ceiling (pure `mma.m16n8k16`, no memory traffic):
  **41.4 TFLOPS** (97% of the theoretical 16 SM x 2048 FLOP/SM/clock x
  1.3005 GHz = 42.6 TF);
- real GEMM register-profile ceiling (`mma_peak2.cu`, 8 warps/SM, 128
  accumulator registers/thread, no spill): **41.9 TFLOPS**;
- best measured production GEMM: CUTLASS sw2 at 31.2 TFLOPS on the gate/up
  shape (75% of the raw ceiling); 37.6 TFLOPS on the skinny shape (91%);
- D2D bandwidth: 181 GB/s.

The Orin SM87 (Ampere) has no FP4 or FP8 tensor cores; NVFP4 weights
dequantize to BF16 for the MAC, so the BF16 tensor ceiling above is the
relevant bound for the NVFP4 deployment.

## FLOP floor for the pinned workload

The model is 27B dense (not MoE). P40000 prefill is 2.2e15 FLOPs: 23.6B
projection parameters x 2 x 40K = 1.89e15, plus 0.31e15 attention scores.
FLOPs scale linearly with prompt length for the projection term, so the
approximately-130K witness is about 7.15e15 FLOPs.

| Witness | FLOPs | Floor at 41.9 TF ceiling | Locked target | Target / floor |
| --- | ---: | ---: | ---: | ---: |
| P40000 | 2.2e15 | 52.5 s | 2 s | 0.038 |
| P130000 | 7.15e15 | 170.6 s | 4 s | 0.023 |

The floor assumes every FLOP executes at the measured mma ceiling, which no
real kernel achieves (attention, norms, conv, and dequant do not); the
achievable floor is therefore higher than the table. The 2 s target would
require 1,100 TFLOPS sustained, 26.3x the measured ceiling; the 4 s target
would require 1,788 TFLOPS, 42.7x the measured ceiling. No software change on
this hardware closes that gap. The best measured route (91.4 s) already runs
at 57% of the floor; realistic kernel-efficiency headroom on the whole-core
route is toward roughly 60-80 s, not toward 2 s.

## Reconciliation with the owner's vLLM reference

The Constitution's own vLLM reference of about 4.3K prompt token/s implies,
if read as a single cold request's Prefill rate, about 9.3 s for 40K tokens
and about 14 s for 60K tokens. That reading does not support a 2 s first
response for a 40K-60K cold prompt, so the locked 2 s target and the locked
vLLM reference are in tension for the stated workload.

The Constitution itself classifies that 4.3K figure as supporting telemetry
that "does not establish its exact elapsed interval or a single request's
Prefill latency," and the target's restrictions exclude Prefix/KV reuse. The
measured FLOP floor (52.5 s for P40000 at the 41.9 TF ceiling) is the
authoritative bound: no software on this board, including vLLM, can Prefill a
cold 40K 27B-dense prompt faster than the floor, so a 4.3K token/s single-
request cold rate would require 236 TFLOPS, 5.6x the measured ceiling, and is
therefore not a fair single-cold-request comparison on this hardware. The
4.3K figure most plausibly reflects batched or cached aggregate throughput,
which the target's no-reuse restriction excludes. Per the Constitution, this
discrepancy is reported without silently lowering the goal; the target stands
until the owner changes it, and the 4.3K figure is not used to claim that the
native runner is behind or ahead of vLLM.

## What remains reachable

- **Decode 10 tok/s:** the short-context measurement is 9.51 tok/s (95% of
  target); the P40000 long-context measurement is 3.87-3.94 tok/s, where the
  40K KV attention term dominates. This is the one locked target with a
  plausible engineering path on the current hardware.
- **Prefill toward the floor:** the whole-core route can plausibly move from
  91.4 s toward the 52.5 s floor (projections are 43.8% of the cost; gate/up
  at 75% and down at 66% of the ceiling leave real headroom). This is a
  bounded efficiency program, not a path to the locked target.
- **Model-family change (P6):** a 35B-A3B MoE reduces activated FLOPs by
  roughly an order of magnitude, moving the P40000 floor toward about 6-7 s.
  That still does not reach 2 s, but it changes the order of magnitude and is
  the only route that could make the 2 s target plausible on this hardware.

## Requested owner adjudication

The locked Prefill targets are retained as written. The owner is asked to
choose the direction for Prefill work:

1. **Amend the Prefill target** to a value consistent with the measured
   hardware bound (for example, the vLLM-referenced throughput class) for the
   40K-60K cold-prompt workload;
2. **Re-scope the witness** if the 2 s target was intended for a shorter
   Agent prompt or a different workload definition;
3. **Change the model family** (P6 MoE) as the means of approaching the
   target on this hardware;
4. **Retain the target and fund the floor program**, treating 2 s as a
   hardware-dependent goal that this Orin board cannot meet, with progress
   measured against the 52.5 s floor instead.

Until the owner decides, Prefill work continues under the existing P3
ordering (accuracy-preserving whole-product gains returned to the real API),
and no target is lowered in any tracked document.

## Consequences if accepted as the working analysis

- The 52.5 s / 170.6 s floors replace the earlier 65.7 s / 84.6 s figures
  (which used the superseded 33.5 TF denominator) wherever a hardware bound
  is cited;
- Prefill progress reports cite the floor percentage, not the locked-target
  percentage, as the engineering metric;
- the Decode 10 tok/s target remains the active locked target with a
  reachable path, and is the natural next P3/P4 focus.
