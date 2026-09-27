---
q3x_document:
  id: q3x-decode-performance-lessons-2026-09-27
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: dated erratum for Decode performance inferences at c4c8a34
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: successor audit with new implementation or qualification evidence
---

# Decode performance retrospective, 2026-09-27

The main mistake was converting the cost of the current implementation into a
hardware limit, then asking the product to accept that limit. Measured local
improvements remain useful. Their unsupported explanations and qualification
extensions must not select later work.

This is an explicit erratum, not a rewrite of historical measurements. Source
identity is `c4c8a34ba39c86e18b68e6a49a401fee43a20d2d`; the
[machine-readable findings and source hashes](../../metadata/qwen36-27b-decode-performance-lessons-2026-09-27.json)
make the corrections inspectable. No new GPU timing, racecheck, accuracy or
installed-artifact test was run for this audit. The
[active Roadmap](../../ROADMAP.md#2026-09-27-product-convergence--active)
owns delivery order, and [Current Status](../../CURRENT_STATUS.md#decode-convergence-snapshot-2026-09-27)
owns current qualification. Existing accuracy and non-MTP constraints remain.

## How the mistaken conclusion formed

| Commit | Recorded step | What later work must retain or correct |
| --- | --- | --- |
| `44842df`, `23e3b0a`, `216df21` | Real API attribution followed by unroll-8 and unroll-128 | Useful scalar-path improvements; faster serial work does not prove the architecture optimal. |
| `65a0186` | Requested-byte bandwidth interpreted as near-free L2 reuse; approximately 5.5 tok/s ceiling | The physical-traffic inference and general ceiling are unsupported. |
| `5f625d8` | Split values improved the API; interpretation flipped to sixfold real DRAM cost and approximately 6.1 tok/s ceiling | Keep improvement, reject attribution from timing alone and the new ceiling. |
| `574a216`, `d37cc90` | Matched kernel attribution, read calibration and split microbenchmarks | Useful bounded observations; a calibration is not universal peak bandwidth. |
| `0686340` | Choices reduced to batching, numerical compromise or lower target | This excluded materially different attention dataflows before evaluating them. |
| `158d04e`, `d6d5afb` | Scores split and repeated long-output split checks | Keep scope-specific results; 6.29 is not a universal optimum, repeated divergence is not a capability score. |
| `c4c8a34` | Fused GQA direction reached P40 8.557 tok/s | Reopens architecture decisively; still does not qualify production or prove 10 tok/s achievable. |

The historical [hardware-bound record](../../metadata/qwen36-27b-decode-hardware-bound-2026-09-26.json)
and [direction evaluation](../../metadata/qwen36-27b-decode-directions-evaluation-2026-09-27.json)
remain unchanged. Their field labels such as “no ceiling claim” do not override
contradictory impossibility assertions in their contents. The moving ceiling
was an implementation model repeatedly adjusted after local improvements,
not an independently established bound.

## Findings and corrections

### D01. Moving implementation cost presented as a hardware ceiling

Old inference (summary): The realistic batch-one ceiling is approximately 6.1 tok/s; no direction reaches single-request 10 tok/s.

Correction: A sum of incumbent kernel costs bounds that implementation, not all legal dataflows. The later fused real-API direction reaches 8.557 tok/s under its recorded unmatched loading protocol; this is sufficient to reopen architecture, not qualify 10 tok/s.

Retain: Retain measured unroll/split gains; retire 5.5/6.1/6.29 as planning ceilings.

Consequence: Qualify one composed fused architecture at the real API. Sources: [hardware-bound-2026-09-26](../../metadata/qwen36-27b-decode-hardware-bound-2026-09-26.json), [directions-evaluation-2026-09-27](../../metadata/qwen36-27b-decode-directions-evaluation-2026-09-27.json).

### D02. Calibration and checkpoint size promoted to a compulsory traffic bound

Old inference (summary): 18.529 GB / 182.7 GB/s gives a 101.4 ms hardware floor.

Correction: The division is a conditional estimate. A read microbenchmark is not a universal bandwidth upper bound, and checkpoint tensor bytes are not independently established mandatory DRAM transactions per step.

Retain: Retain the read calibration and tensor inventory as observations.

Consequence: Any bound needs same-workload actual traffic, operations, matched profile and known implementation reconciliation. Sources: [hardware-bound-2026-09-26](../../metadata/qwen36-27b-decode-hardware-bound-2026-09-26.json), [premise-remeasure-2026-09-27](../../metadata/qwen36-27b-decode-premise-remeasure-2026-09-27.json).

### D03. Opposite DRAM/L2 causal claims from the same indirect timing

Old inference (summary): Requested throughput above read calibration proves L2 absorption; corrected timing proves sixfold real DRAM cost.

Correction: Neither conclusion follows without actual transactions/counters or a discriminating control. Source reuse also changes instruction count, occupancy and scheduling. A slower 1 GB benchmark than a 4 GB benchmark does not establish partial L2 assistance.

Retain: Retain measured times and API changes, not the asserted physical traffic cause.

Consequence: Use traffic terminology precisely; collect counters only if this cause selects the next implementation. Sources: [hardware-bound-2026-09-26](../../metadata/qwen36-27b-decode-hardware-bound-2026-09-26.json), [premise-remeasure-2026-09-27](../../metadata/qwen36-27b-decode-premise-remeasure-2026-09-27.json).

### D04. Sixteen-layer totals mislabeled per layer

Old inference (summary): V unique volume 1.31 GB/layer, redundant volume 7.86 GB/layer, time 40.3 ms/layer.

Correction: At S=40000, KV heads=4, D=256, BF16: one-layer V=81920000 bytes; all 16 layers=1310720000 bytes; six logical query traversals across all layers=7864320000 bytes. The cited 40.3 ms is likewise an aggregate.

Retain: Ratios can accidentally remain plausible when numerator and denominator both contain the same factor-16 error.

Consequence: Label bytes, layer count, memory level and time scope explicitly. Sources: [hardware-bound-2026-09-26](../../metadata/qwen36-27b-decode-hardware-bound-2026-09-26.json), [premise-remeasure-2026-09-27](../../metadata/qwen36-27b-decode-premise-remeasure-2026-09-27.json).

### D05. Mixed-version denominators and retrospective predictions

Old inference (summary): 100.52 ms is 38% of the 165.2 ms integrated step.

Correction: 100.52/165.2=60.847%; 100.52/193=52.083%. Neither is 38%. A prediction revised after observing the result is an updated model, not a successful prior prediction.

Retain: The later matched per-kernel attribution is useful recovery; keep its own artifact and denominator.

Consequence: Freeze prediction separately; do not combine costs across binary/protocol versions. Sources: [split-values-bandwidth-2026-09-27](../../metadata/qwen36-27b-decode-split-values-bandwidth-2026-09-27.json), [scores-split-e2e-2026-09-27](../../metadata/qwen36-27b-decode-scores-split-e2e-2026-09-27.json).

### D06. Greedy divergence, nondeterminism and capability conflated

Old inference (summary): Changed output attributed directly to nondeterministic few-ULP accumulation.

Correction: A before/after string difference proves divergence, not repeatability failure or capability regression. Repeated identical-run divergence supports nondeterminism; same-input tensors/logits and capability evaluation address different questions.

Retain: Later [repeated split runs](../../metadata/qwen36-27b-decode-split-production-cost-measurement-2026-09-27.json) provide stronger repeatability evidence; scalar long-output observations retain their exact context scope.

Consequence: Keep separate numerical error, repeated execution, greedy margins and public capability checks. Sources: [split-values-e2e-2026-09-26](../../metadata/qwen36-27b-decode-split-values-e2e-2026-09-26.json), [longout-stability-2026-09-26](../../metadata/qwen36-27b-decode-longout-stability-2026-09-26.json).

### D07. A failed geometry generalized to shared GQA reuse

Old inference (summary): The single-block-per-KV-head failure rejects GQA sharing; atomic merging is its required accuracy tradeoff.

Correction: Four-CTA geometry does not reject segmented GQA reuse. The fused candidate uses staged partials and deterministic merge rather than unordered atomic addition.

Retain: Retain negative evidence for the exact single-block geometry and split implementation.

Consequence: Select a complete dataflow before scanning parameters; do not inherit an abandoned reduction mechanism. Sources: [hardware-bound-2026-09-26](../../metadata/qwen36-27b-decode-hardware-bound-2026-09-26.json).

### D08. Short or partial validation generalized to product qualification

Old inference (summary): A short-context long-output run or a short oracle closes long-context decode correctness.

Correction: P1089/O256 does not cover P40 long output. max-length-128 oracles cannot exercise S>=512 split routing. Text searches for nan/inf are not tensor-finiteness checks.

Retain: Retain each actual oracle and output check at its tested scope.

Consequence: Test the selected path and useful long-context output capacity, then qualify the installed composition. Sources: [longout-stability-2026-09-26](../../metadata/qwen36-27b-decode-longout-stability-2026-09-26.json).

### D09. New fusion direction must not inherit old overclaim patterns

Old inference (summary): 8.56 tok/s is a production-qualified rate or the 2% screen establishes accuracy.

Correction: It is one unprofiled O32 real-API direction with default/lazy baseline versus EAGER candidate. The shadow screen is gross admission, not an approved numerical tolerance. P40 text changed. No new successful vLLM API reference or composed accepted-Prefill runner was obtained.

Retain: Keep the real direction, finite shadow tensors, repeated same-input output and diagnostic profile as bounded evidence.

Consequence: Match loading protocol, declare numerical criteria before qualification, compose accepted Prefill and useful output length. Sources: [context-scaling-investigation-2026-09-27](../../metadata/qwen36-27b-decode-context-scaling-investigation-2026-09-27.json).

### D10. Source contracts do not match experimental-path assurances

Old inference (summary): Opt-in split code and sealed CLI imply production route isolation and read-once scratch ownership.

Correction: At c4c8a34 both split launchers consult getenv without a BUILD_TESTING guard; public GQA dispatch invokes them. The inspected server environment rejection is confined to the whole-core development branch. Values scratch is process-static cudaMalloc, and two thread groups write each shared tile element.

Retain: This is static source evidence, not a reproduced installed-release failure or measured physical traffic count. Prior cleared-environment timings remain scoped and unaffected.

Consequence: Seal low-level dispatch and use explicit engine-owned workspace; do not promote the legacy split implementation. Sources: [values split source](../../../src/kernels/reference/attention_values_split_shared.cu), [scores split source](../../../src/kernels/reference/attention_scores_split_shared.cu), [public GQA dispatch](../../../src/kernels/reference/decode_ops.cu), [server validation](../../../src/server/evaluation_server.cpp).

## Arithmetic and causality boundaries

All byte quantities below are decimal; none is a hardware-counter reading.
For P40000, KV heads 4, head dimension 256, BF16 KV, 16 full-attention layers:

| Quantity | Explicit calculation | Bytes |
| --- | --- | ---: |
| Unique V, one layer | 40000 × 4 × 256 × 2 | 81,920,000 |
| Unique V, all 16 layers | previous × 16 | 1,310,720,000 |
| Six logical traversals of V, all layers | previous × 6 | 7,864,320,000 |
| Unique K and V, all layers | 1,310,720,000 × 2 | 2,621,440,000 |

The old 40.3 ms and 7.86 GB can yield a plausible ratio even though both were
mislabeled “per layer.” This is why dimensions must be checked independently
of bandwidth ratios. Logical loads, unique bytes, cache transactions and DRAM
transactions are four different quantities.

The values-split source has 512 threads, `dimension = thread % 256`, and both
256-thread groups execute the shared-tile load/store loop without a producer
guard. Two threads therefore write each shared element from the same V address.
This exposes duplicate source-level loads and a shared-memory write-ownership
problem; it does not establish twice the DRAM transactions or prove the
observed token divergence came from that problem. No racecheck was run.
Its process-static allocation likewise does not provide engine/stream-scoped
ownership or teardown. These facts are reasons not to inherit that prototype
as a production resource design.

Both split selectors test environment-variable presence, so even the string
`0` selects them. The inspected production CLI sealing does not by itself
seal those lower-level paths. This audit identifies a source-level gap;
installed-binary reachability still needs its own qualification check. The
prior investigation explicitly cleared route-changing environment variables,
so this finding does not retroactively invalidate its default-route timings.

## Qualification mistakes that the new direction must avoid

The [fused investigation](../decode-context-scaling-2026-09-27/README.md)
records these limits explicitly:

- Native versus fused P40 TPOT was 192.952 versus 116.858 ms, or 5.183 versus
  8.557 tok/s. The baseline used default/lazy loading and the candidate EAGER;
  the matched EAGER baseline failed startup. This is direction evidence, not
  the final mirrored qualification comparison.
- The unchanged short-Graph free-memory check failed with both implementations.
  EAGER is not an established startup repair. Failed starts stay in the record.
- Same-input shadow relative L2 peaked at about 0.321%; finite tensors and
  repeatability passed the tested cells. The 2% gross screen is not a product
  accuracy tolerance. The P40 generated text changed. Neither automatic
  rejection as “quality loss” nor automatic acceptance as “small error” follows.
- One captured step attributed 13.7592 ms to fused attention including merge.
  Profiled API TPOT was distorted and is not a second performance measurement.
  Historical scalar attention attribution near 92 ms is a separate capture.
- No new vLLM API reference completed. Source study supports the mechanism,
  not a fabricated parity result. Future reference startup should use the
  required text workload and a bounded readiness check; the failed attempt's
  multimodal startup cost did not answer the Decode question.
- The wrapper reuses probability scratch for partial outputs/LSE. It does not
  satisfy the public normalized-probability postcondition and cannot become
  production by merely copying the wrapper into that API.
- Prefill ADR-0002 is an accepted Prefill numerical decision, not blanket
  Decode permission. The accepted whole-core P40/O16 profile and the Legacy
  fused Decode experiment are different compositions. Their best numbers
  cannot be added into a new delivered product.

Short successful completions, stable repeated strings, no atomic instructions,
and maximum-length admission are useful separate facts. None substitutes for
representative model capability, state fidelity, useful output capacity and
installed-artifact behavior. Archived microbenchmarks whose original source
or raw artifacts no longer exist remain observations with reduced
reproducibility; this audit does not invent replacements.

## Delivery consequence

Keep the measured scalar gains as a regression baseline. Pause new Prefill
kernel exploration under the owner's current direction. Close the old scalar
split/unroll search as the planning route and admit one fused Decode dataflow
with explicit workspace and numerical ownership. Then return it to the accepted
Prefill API with useful output length, fixed route identity and matched
performance/capability qualification. The related Prefill peak/FLOP ratios remain conditional arithmetic-model
estimates, not adopted universal bounds; no new Prefill bound audit or target
change is claimed here. API protocol matrices are also labeled conformance,
not model-capability scores. The detailed sequence and bounded first
batch are in the active Roadmap; this frozen erratum does not create a second
plan or revise targets.
