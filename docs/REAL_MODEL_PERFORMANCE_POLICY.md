# Real-model performance evidence policy

This policy defines which measurements may select, reject, or promote a model
kernel.  It applies to microbenchmarks, component screens, end-to-end
benchmarks, and Nsight evidence.

The [engineering constitution](ENGINEERING_CONSTITUTION.md) is the controlling
planning and target contract. This policy qualifies evidence; it may not be
used to debate away or silently lower a project-owner production observation.

## Authority rule

Direct project-owner observations of real production/API behavior are
authoritative for priority and target setting. They are not automatically
release-grade publication evidence, but a conflicting local benchmark must be
treated first as a harness, token-accounting, cache, endpoint, configuration,
or path-reconciliation problem. Until a same-workload reproduction is
accepted as superseding evidence, the owner-supplied target remains active.
Component cells, logger-window rates, short-prompt corpora, and unmatched
framework runs cannot lower it.

Any result used to continue or stop an optimization, change production
dispatch, set a performance threshold, or publish a model-performance claim
must use a pinned production payload.  Synthetic timing is non-admissible: it
may smoke-test a path or help form a hypothesis, but it may not pass or fail a
performance gate.

Synthetic inputs remain required for exhaustive correctness: all quantized
codes, NaNs and scale edge cases, same-bank stress, guards, aliases, invalid
zero-enqueue behavior, race checks, and CUDA Graph replay.  A synthetic timing
anomaly must be investigated with a broader real-payload panel before it can
affect a performance decision.

Pure device-health calibration, such as launch or memcpy smoke tests, may use
synthetic bytes when it is explicitly labelled `hardware_calibration` and no
model-performance conclusion is drawn from it.

## Test tiers

| Tier | Payload | Purpose | Performance authority |
|---|---|---|---|
| T0 | none | build, ABI, selector, SASS, and static resource contracts | none |
| T1 | synthetic | exhaustive correctness, robustness, Graph, and smoke tests | none |
| T2 | pinned model tensors | same-binary component timing and candidate selection | component authority |
| T3 | pinned model and prompts | natural layer order, route attribution, Prefix/TTFT, tokens, and state | formal protocol: production authority; direction screen: early-stop only |
| T4 | the exact T2 or T3 payload | NCU/NSys mechanism attribution after a valid timing run, including a rejected direction screen | diagnostic only |

T2 is necessary but not sufficient for production promotion.  Production
selection also requires the applicable T3 gates.
A direction screen is a restricted protocol class on a T3 payload, not a new
payload tier. It has early-stop authority only; T3 gains production authority
only after the formal correctness, noise, repetition, and promotion protocol
completes.

## External whole-product evaluation

Evidence tier and observation surface are orthogonal. EvalScope through the
project's OpenAI-compatible adapter is an external T3 protocol class, not a
new payload tier. Project-level performance, capability, and release claims
must ultimately exercise the pinned real model through such a publicly
recognized framework and user-visible API. An internal P513 run, component
timer, NSys trace, or NCU cell may screen direction and explain a result, but
it cannot replace the external whole-product observation.

The decision order is:

1. run the smallest predeclared real external workload that can reveal whether
   the complete architecture change matters;
2. if positive, validate correctness, state, resources, noise, and mechanism
   with the applicable T2--T4 tools;
3. return the completed architecture milestone to the same external protocol
   before retention or production promotion.

A one-process, one-round external run may set roadmap priority but has no
release, promotion, publication, or threshold-recalibration authority.
Release evidence must predeclare workload and framework versions, preserve
raw artifact hashes, repeat independent processes in mirrored order, record
success and output completeness, report tail latency, and obtain a parseable
public capability score. A framework-reported number is inadmissible when its
request contract fails; truncation before an answer marker is a protocol
failure, not an accuracy score.

## Required production payload

For weight-dependent kernels, T2--T4 must load the real weight, block scale,
and tensor scale directly from the checkpoint.  NVFP4 evidence includes the
packed weight, E4M3 block scale, and the exact raw bits of `weight_scale_2`;
FP8 evidence includes the weight and its tensor scale.  A generated tensor
with a checkpoint-like distribution is still synthetic.

When values affect control flow, addresses, sparsity, compression, or state
evolution, the activation, KV cache, or recurrent state must also come from a
pinned real-prompt capture at the relevant layer boundary.  Attention, SSM,
GDN, and any data-dependent fast path therefore cannot be promoted from real
weights combined only with arbitrary synthetic state.  A value-independent
dense projection may use a deterministic activation fixture at T2, but the
record must identify the evidence as `checkpoint_weight_only`, and T3 remains
mandatory.

The current model identity is:

- repository: `nvidia/Qwen3.6-27B-NVFP4`;
- revision: `0893e1606ff3d5f97a441f405d5fc541a6bdf404`.

Every evidence record must additionally pin the config, quantization config,
index and shard identities; tensor name, layer, role, dtype, shape, file range
and payload SHA256; and scalar scale bits.  Paths must be canonical, read-only,
non-symlink files inside the model root.  A real activation/state capture must
pin the prompt or token IDs, tokenizer/config revision, capture point, dtype,
shape, and payload SHA256.

The record must also identify the git tree, binary SHA256/build ID, CUDA and
driver versions, device, clocks, nvpmodel, CPU affinity, temperature envelope,
and cache protocol.  A hash or shape mismatch fails closed before enqueue.

## Modes and exit behavior

Performance harnesses must expose separate, unambiguous modes.  Names may be
adapted to an existing executable, but their behavior must match this contract:

- `validate-synthetic` (currently spelled `smoke` by the C512 harnesses): T1
  only; it contains no timing threshold.
- `performance-checkpoint`: T2 or T3; a checkpoint is mandatory and no
  synthetic fallback exists.
- `profile-checkpoint`: T4; it uses the same pinned payload and exact
  incumbent/candidate routes as the preceding valid timing run.

Exit `0` means that the requested tier completed and its applicable gates
passed.  Exit `1` means invalid evidence, a correctness failure, or a runtime
failure.  Exit `2` means invalid or incomplete command-line configuration.
Exit `3` distinguishes a valid, completed measurement that rejected the
performance candidate.  Exit `77` means that optional hardware or checkpoint
data is unavailable in a non-performance environment.  When strict evidence is
enabled, for example with `Q3X_REQUIRE_REAL_PERF=1`, missing model data,
incorrect hashes, unsupported hardware, or unlocked required clocks are
failures, not skips.  A missing checkpoint must never select a synthetic
performance path implicitly.  For native development screens, exit `3`
means that the candidate was not retained against the frozen native
incumbent; an external-library reference may never cause that exit.

## Current C512 migration state

The Gate-only cuBLASLt ceiling is synthetic T1 only.  Its legacy absolute
reference and directional speedup are diagnostic fields and cannot retain or
reject a route.  The Gate+Up pair and Down ceiling expose separate `smoke` and
`performance-checkpoint` modes.  Only the latter may emit a performance
decision, requires a checkpoint explicitly, and is registered as a serialized
CTest when `Q3X_REAL_PERF_MODEL_DIR` (or the E2E model directory) is supplied.

These component harnesses currently combine pinned layer-0 weight, block
scale, and tensor scale with a deterministic activation fixture.  Their
evidence class is therefore `checkpoint_weight_only`, not `full_model`.
Their layer-0 results are useful migration controls but remain provisional
until the candidate-independent four-layer panel and T3 full-model gates are
implemented.  No historical synthetic threshold is thereby re-certified.

The historical exact-C512 Down cuBLASLt module no longer times synthetic BF16
operands while constructing its reference context.  It scans the runtime
heuristic list for the zero-workspace configuration locked by the current T2
checkpoint evidence (`algorithm_id=6`, `tile_id=23`, `split_k=1`, no
reduction, `cta_swizzle=1`, `custom_option=0`, `stages_id=15`).  This module is
an external benchmark reference only: it has no production dispatch,
fallback, retention, or promotion eligibility.  Heuristic-list rank is
recorded but is not an ABI or native selection key.

This exclusion is permanent rather than a temporary migration state.  The
production executable and installed kernel library must have no cuBLASLt
dependency, symbol, selector, context, scratch allocation, or fallback.  A
native retention run must also remain executable when every cuBLASLt reference
component is absent; reference setup or timing failure cannot alter its result
or exit status.  Link/symbol/install audits enforce this boundary.

## CI lanes

Ordinary CI requires T0 and T1.  It may report T2--T4 as explicit skips when
the pinned model or SM87 device is unavailable; a skip is not a performance
pass.  Suggested labels are `correctness;synthetic`,
`correctness;checkpoint`, `performance;checkpoint;sm87;retention`,
`performance;checkpoint;sm87;promotion`, and `profile;checkpoint`.

The Orin performance lane requires the pinned checkpoint, strict evidence
validation, fixed GPU and EMC clocks, CPU affinity, and serialized ownership
of the GPU.  Kernel, selector, pipeline, cache-policy, or threshold changes
may not claim a performance completion without evidence from this lane.
Development retention and production promotion are separate test labels and
decisions.  Release promotion requires both T2 and T3 results.

## Layer panels and cache state

A single convenient layer, including layer 0, is not a general production
sample.  Before viewing a candidate result, each projection or stateful role
must freeze a layer panel chosen from all eligible layers using payload
features such as quantized-code and scale-exponent histograms, zero fraction,
entropy, and compressibility.  The development panel must contain at least
four layers spanning low, median, high, and extreme payloads.  Gate, Up, Down,
FP8 QKV/Z/O/KV, Attention, and SSM use independent panels.

Final promotion must either sweep every eligible layer or exercise all of them
in the natural T3 model path.  Component timing should rotate through real
layers or explicitly scrub cache so that repeated use of one tensor does not
create an accidental warm-cache gate.  If warm-cache behavior is relevant, it
is recorded as a separate steady-state result.

## Direction-first development and threshold recalibration

Synthetic thresholds and absolute timings are not transferred to real
payloads. Development follows a direction-first funnel so that a candidate
must first show useful movement in the natural model path before it consumes
complete qualification effort:

1. Freeze the live incumbent, payload manifest, layer panel, environment, and
   timing protocol before inspecting the candidate.
2. Before the first device run, complete only the minimum safe admission:
   build and launch contracts, bounds and resource sanity, route isolation,
   and at least one applicable correctness oracle. Exhaustive synthetic and
   full-state qualification are not prerequisites for the first direction
   screen unless safety or the candidate's numerical contract requires them.
3. Run the first performance screen in the pinned T3 generation path whenever
   that path exists. Use one engine and one ELF, the natural layer order, the
   real prompt-derived activations/state, explicit incumbent/candidate route
   hits, a deterministic generation oracle, and a snapshot-free mirrored
   order such as one B-C-C-B round. A T2 component screen may help diagnose a
   candidate, but it does not replace this first whole-path direction check.
4. A negative or neutral direction result has early-stop authority. Reject and
   archive the candidate without building exhaustive correctness, noise,
   six-round, profiler, or promotion harnesses. Alternatively, when a concrete
   causal question could change the next design, collect a bounded T4 NSys or
   NCU comparison on the same real payload and record it as diagnostic
   evidence. Profiling is optional and cannot reverse the rejection.
5. A positive direction result has no retention, promotion, or publication
   authority by itself. It unlocks complete correctness/state/Graph/resource
   validation and incumbent-versus-incumbent paired controls on the real panel
   to establish the local noise floor.
6. A development candidate is retained when the formal paired real-payload
   result is
   stably positive against the current native development incumbent and clears
   the measured noise allowance.  It then becomes the new native development
   incumbent.  Do not impose an external reference or the terminal production
   margin on this incremental step.
7. Use the same final binary and six B-C-C-B rounds for incumbent and
   candidate.  Preserve all raw rounds, including reversals and failures.
8. Production promotion is a separate cumulative decision against the current
   native production baseline.  It requires per-layer non-regression, the
   predeclared engineering margin (currently 1.03x where specified), an
   independent-process repeat, complete role/shape coverage, and T3.
9. Re-run the gates after the production binary and selector are finalized.

A direction screen may stop work but may never retain a candidate, update a
production selector, recalibrate a threshold, or serve as a published
performance anchor. A rejected candidate may be reopened only after a named
mechanism changes, at which point the new candidate starts at step 2.

Thresholds are role- and shape-specific.  Gate evidence does not set Down or
FP8 thresholds, and a microbenchmark gain is weighted by the number of actual
model calls before its end-to-end significance is claimed.

## Historical evidence

Historical artifacts remain immutable.  An evidence registry should classify
each result as `synthetic_only`, `checkpoint_weight_only`,
`checkpoint_full_payload`, or `full_model`, and mark its authority as
`non_admissible_legacy`, `provisional`, `confirmed`, or `superseded`.

- Synthetic microbenchmarks backed by later real full-model evidence may
  retain the production decision, but their mechanism claims are directional.
- A synthetic-only production selection is provisional and must be
  requalified in priority order; it is not reverted without replacement
  evidence.
- A synthetic-only rejection is no longer a permanent stop.  It is rerun only
  if that mechanism re-enters the active roadmap; old negative cells need not
  be migrated wholesale.

## Profiling order

NCU or NSys follows, never precedes, a completed and valid T2 or T3 timing run
or direction screen on the same real payload. A rejected candidate may be
profiled when an explicit causal question can guide the next design, but that
work must be bounded and the profile is diagnostic: it cannot reverse the
rejection. A negative candidate may instead be archived immediately; profiler
completion is not a rejection gate. The
authoritative T2 retention comparison is always native incumbent versus native
candidate.  For the C512 NVFP4 external-reference study, profile the
dequantization kernel, the immediately following cuBLASLt BF16 GEMM, and the
native NVFP4 kernel separately.  Both `dequantization + Lt` versus `native`
and Lt-only results are diagnostic reference observations; neither may retain,
reject, select, or promote a production route.

Collect occupancy and launch resources, pipeline/SASS structure, issue and
Tensor activity, HMMA count, stall reasons, global/L2/L1/shared traffic, and
bank conflicts.  Normalize counters by useful work and record counter
denominators.  Profiler percentages overlap and must not be added into a
latency explanation.  A gap is eligible for implementation only after it has
a quantified cycle or byte ceiling on the real payload; the resulting change
must return through T2 and T3.
