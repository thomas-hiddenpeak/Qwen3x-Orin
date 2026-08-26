---
q3x_document:
  id: q3x-real-model-performance-policy
  class: normative
  status: active
  owner: project-owner
  authority: performance evidence, candidate selection, retention, and release promotion
  effective: 2026-08-09
  last_reviewed: 2026-08-27
  supersedes: []
  superseded_by: []
  ssot_for: performance evidence tiers and decision-unit lifecycle
  review_trigger: evidence tier, candidate unit, selection gate, or production-promotion change
---

# Real-model performance evidence policy

This policy defines which measurements may retain a local mechanism, select an
architecture, or promote the complete runner. It applies to microbenchmarks,
component screens, end-to-end benchmarks, and Nsight evidence. Those are three
different decisions. An evidence record must name the decision it is allowed
to influence rather than using the unqualified word `candidate`.

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
| T2 | pinned model tensors | same-binary component timing inside a named local work package | local-retention authority only |
| T3 | pinned model and prompts | natural layer order, route attribution, Prefix/TTFT, tokens, and state | direction screen: early-stop; target-length architecture witness: development selection; formal release protocol: production authority |
| T4 | the exact T2 or T3 payload | NCU/NSys mechanism attribution after a valid timing run, including a rejected direction screen | diagnostic only |

T2 is necessary but not sufficient for production promotion.  Production
selection also requires the applicable T3 gates.
A direction screen and a target-length architecture witness are restricted
protocol classes on a T3 payload, not new payload tiers. A direction screen
has early-stop authority only. A completed target-length witness may select a
development architecture but cannot change production. T3 gains production
authority only after the formal correctness, noise, repetition, and promotion
protocol completes for a frozen `release_candidate`.

## Decision proportionality and environment authority

Evidence cost follows decision authority. An ordinary engineering direction
screen uses the minimum safe real-payload check that can decide the next
reversible step. A target-length architecture witness pays for a matched,
predeclared whole-product comparison. Release qualification pays for the full
repeated artifact, environment, accuracy, capability, resource, and packaging
protocol. Release-grade ceremony must not be imposed on every implementation
iteration, and a direction screen must not be relabelled as release evidence.

Core decision facts and supporting environment context are distinct. Model,
binary, workload and route identity; numerical/output correctness; API and
capacity semantics; fallback behavior; device ownership; resource safety; and
the metric named by the decision are core. Temperature, clock samples,
over-current and throttle counters, memory snapshots, utilization traces, and
similar host observations are normally supporting context. Record them when
available and use them to explain anomalies, but a benign deviation or a
missing auxiliary sample does not by itself stop ordinary engineering or
erase unaffected correctness, route, state, or output evidence.

Environment state becomes a hard invalidator when it demonstrates a material
safety risk, an unowned GPU consumer, confirmed workload contention, OOM or
device failure, actual throttle/downclock that destroys timing attribution, a
resource leak, or another major fault. It is also a hard gate when required by
a predeclared architecture-selection or release-qualification protocol. A
failure in an auxiliary monitor narrows the claims that depend on that monitor;
it does not automatically null the whole run.

The purpose of a measurement is to make an engineering decision. When the
stated question has been answered, stop collecting evidence and implement,
compose, retain, or reject the direction. A local result may select a local
mechanism only; it cannot override a negative composed API result or keep the
project optimizing a local maximum while the global product remains behind.

## Candidate units and decision authority

Every performance change belongs to exactly one of these units:

| Unit | Meaning | May be selected by | Authority it never has by itself |
|---|---|---|---|
| `local_mutation` | One mechanism or an intentionally coupled mechanism bundle inside a named local optimization work package | A same-binary, real-payload T2 comparison or a more representative T3 comparison against the package's frozen local incumbent | global priority, architecture selection, production dispatch, or a product-performance claim |
| `architecture_candidate` | A coherent executable dataflow whose mechanisms are connected on the real generation route, including their layouts, ownership, synchronization, state, and fallback contract | A predeclared T3 target-length API selection witness against the current native architecture | production promotion before complete release qualification |
| `release_candidate` | A frozen production binary, deployment plan, configuration, API contract, and model-artifact set | The complete repeated T3 release protocol plus accuracy, capability, resource, and packaging gates | changing the constitutional product or numerical contract |

The unit may grow only through an explicit transition recorded in evidence:
`local_mutation -> architecture_candidate -> release_candidate`. Local
retention is not architecture retention, and architecture selection is not a
release claim. The frozen native default/qualification baseline remains
unchanged until a `release_candidate` completes promotion; whether that
baseline is already production-qualified is reported only by
[`CURRENT_STATUS.md`](CURRENT_STATUS.md).

### Named local optimization work packages

Mechanism-level engineering rules apply only while work is explicitly inside
a **named local optimization work package**. They are tools for executing a
chosen architecture hypothesis; they must not set the project's global
priority, redefine the product target, or keep an architecture alive after
its whole-path evidence fails.

Before the first local timing run, a work-package record must freeze:

- its identifier, owning `architecture_candidate`, affected production route,
  exact shapes/roles, and numerical contract;
- the bottleneck observation that opened it and the expected path by which
  local savings can reach the target API witness;
- the real payload, local incumbent, timing/noise protocol, correctness oracle,
  and interactions that require two or more mechanisms to be tested together;
- an integration point and an explicit composition deadline expressed as a
  date, a maximum number of local variants, or a clean-host device-time budget;
  and
- a stop-loss budget for the composed architecture, not an invented demand
  that each mechanism independently move an end-to-end metric.

A `local_mutation` may be retained inside that package only when it is
numerically admissible, its selector/route attestation shows that it can
execute on the named production route without an undeclared fallback, and
either:

1. is stably positive over the package's real-payload local incumbent beyond
   the measured local noise; or
2. is a required dependency of a predeclared coupled dataflow, has a bounded
   mechanistic proof, and is labelled `dependency_only` until composition.

The first condition updates only the package-local incumbent. The second does
not update a performance baseline and cannot be called a speedup. Coupled
loads, layouts, decode, MMA, fusion, and pipeline stages may be compared as one
bundle when their interaction is the hypothesis; the policy does not require
artificially orthogonal single-variable experiments.

A local mutation is not required to pierce whole-API noise before it can be
retained for composition. Conversely, locally retained mechanisms may not
accumulate indefinitely. At the declared deadline the package must compose
them into its named `architecture_candidate` and return to the target API
witness, or archive the uncomposed mechanisms as non-production evidence. A
deadline extension requires a newly recorded causal finding and one new
bounded deadline. When the composed architecture loses, package-local wins do
not override that result; work resumes only under a materially changed named
architecture hypothesis.

cuBLASLt may provide an external component ceiling, while vLLM provides the
whole-product starting line and mechanism clues. Neither serves as a
per-mutation retention threshold, and cuBLASLt never enters a native package's
production route.

### Architecture and release selection

An `architecture_candidate` must connect its mechanisms in one real runner
route and expose route identity, pure Prefill timing, TTFT, output/state
identity, peak resource use, and fallback behavior. Prefill architecture
selection uses cold/no-cache 40K, 60K, and approximately 130K API witnesses on
the pinned model. These witnesses may run in fail-fast order, but a short
prompt or component result can only check sanity or explain a mechanism; it
cannot name the winning Prefill architecture.

The architecture comparison is cumulative native incumbent versus cumulative
native candidate. A stable improvement on the predeclared target-length
witness set selects the new development architecture; external vLLM remains
the starting-line and cumulative comparison, not the incremental rejection
gate. Selection still does not alter production dispatch.

A `release_candidate` freezes the selected architecture into the actual
deliverable API, release build, authenticated deployment plan, and default
configuration. It must complete the formal repeated T3 protocol on all target
lengths, exact/no-regression accuracy gates, public capability evaluation,
resource and startup checks, and installation/link/route attestation. Only
that decision may change the production baseline.

## External whole-product evaluation

Evidence tier and observation surface are orthogonal. EvalScope through the
project's OpenAI-compatible adapter is an external T3 protocol class, not a
new payload tier. Project-level performance, capability, and release claims
must ultimately exercise the pinned real model through such a publicly
recognized framework and user-visible API. An internal P513 run, component
timer, NSys trace, or NCU cell may screen direction and explain a result, but
it cannot replace the external whole-product observation.

The decision order is:

1. use the smallest safe real API request to establish that the route,
   streaming contract, token accounting, and output oracle function; this is a
   sanity gate, not Prefill architecture selection;
2. execute any named local work packages with T2--T4 evidence under their
   bounded composition contracts;
3. return the composed `architecture_candidate` to the predeclared 40K, 60K,
   and approximately 130K cold/no-cache T3 witnesses for architecture
   selection; and
4. freeze the selected architecture as a `release_candidate` and repeat the
   complete external capability, performance, accuracy, and packaging
   protocol before production promotion.

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

The record must also identify `decision_unit`, `work_package_id` when local,
`architecture_candidate_id` when applicable, the git tree, binary SHA256/build
ID, CUDA and driver versions, device, clocks, nvpmodel, CPU affinity,
temperature envelope, and cache protocol. A hash or shape mismatch fails
closed before enqueue.

## Modes and exit behavior

Performance harnesses must expose separate, unambiguous modes.  Names may be
adapted to an existing executable, but their behavior must match this contract:

- `validate-synthetic` (currently spelled `smoke` by the C512 harnesses): T1
  only; it contains no timing threshold.
- `performance-checkpoint`: T2 or T3; a checkpoint is mandatory and no
  synthetic fallback exists.
- `profile-checkpoint`: T4; it uses the same pinned payload and exact incumbent
  plus `local_mutation` or `architecture_candidate` routes as the preceding
  valid timing run.

Exit `0` means that the requested tier completed and its applicable gates
passed.  Exit `1` means invalid evidence, a correctness failure, or a runtime
failure.  Exit `2` means invalid or incomplete command-line configuration.
Exit `3` distinguishes a valid, completed measurement that rejected the
declared decision unit. Exit `77` means that optional hardware or checkpoint
data is unavailable in a non-performance environment.  When strict evidence is
enabled, for example with `Q3X_REQUIRE_REAL_PERF=1`, missing model data,
incorrect hashes, unsupported hardware, or unlocked required clocks are
failures, not skips.  A missing checkpoint must never select a synthetic
performance path implicitly. For native development screens, exit `3` means
that the declared `local_mutation` or `architecture_candidate` was not
retained against its corresponding frozen native incumbent; an
external-library reference may never cause that exit.

## Dormant C512 evidence boundary

C512-specific harness modes, tactic IDs, thresholds, and profile prescriptions
are historical local-work-package material, not normative policy. They remain
reachable through [`LARGE_M_PROJECTION_DATAFLOW.md`](LARGE_M_PROJECTION_DATAFLOW.md),
[`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md), and the
[`metadata/`](metadata/) evidence index. Reopening any of them requires a new
named work-package manifest with a current incumbent, real payload, evidence
protocol, composition deadline, and API return point; the old local rules do
not reactivate themselves.

The cuBLASLt boundary is project-wide rather than C512-specific: the
deliverable executable and installed kernel library have no cuBLASLt
dependency, symbol, selector, context, scratch allocation, dispatch, or
fallback. Reference setup or timing failure cannot alter a native decision.

## CI lanes

Ordinary CI requires T0 and T1.  It may report T2--T4 as explicit skips when
the pinned model or SM87 device is unavailable; a skip is not a performance
pass.  Suggested labels are `correctness;synthetic`,
`correctness;checkpoint`, `performance;checkpoint;sm87;retention`,
`performance;checkpoint;sm87;promotion`, and `profile;checkpoint`.

The Orin lane always records the pinned checkpoint, actual CPU/GPU/EMC state,
CPU affinity, and serialized GPU ownership. Ordinary direction screens may use
that lane with environment observations as supporting context. Architecture-
selection and release-promotion jobs additionally use their predeclared strict
clock, repetition, and host-envelope contracts. Kernel, selector, pipeline,
cache-policy, or threshold changes may not claim more authority than the lane
mode completed. Development retention and production promotion are separate
labels and decisions; release promotion requires both T2 and T3 results.

## Host preparation and environment observations

Every real-model timing and profiler process begins with a recorded host/device
preflight. On Jetson the record uses `tegrastats`, CPU/process inspection, and
GPU device-handle ownership inspection. Jetson's incomplete `nvidia-smi`
implementation is not an idle detector and must not attribute GPU consumers.
The runner must own the GPU; an unowned device handle or confirmed material
CPU/GPU contention invalidates timing. Benign background activity and other
environment observations follow the proportional authority above.

Quiesce only identified interfering workloads and preserve an independent,
recoverable control path. The active Codex/SSH session, evidence writer, and
recovery-critical services must not be suspended or terminated to make the
host appear idle. Before each fresh real-model performance or external-
evaluation process, run `sync` and attempt
`echo 3 | sudo -n tee /proc/sys/vm/drop_caches`; record the command result and
memory before and after. If it fails, do not claim cold-cache timing, but
unaffected correctness, route, state, and output observations may still guide
engineering.

Cooling is externally controlled. Harnesses must not inspect, modify, or gate
on system fan/controller state, and must remove incidental fan fields from
retained telemetry. Temperatures at or below 85C are normal. Above 85C through
90C, use actual clocks, over-current, and throttle observations to qualify
timing; temperature alone is not a rejection gate. Above 90C is an operational
stop. Final selection or release protocols may predeclare stricter matched-
start, clock, ownership, cache, and repetition requirements, but they may not
lower the normal-through-85C band or make a temperature in the 85C--90C band or
an isolated over-current counter change the sole rejection reason.

An ordinary direction screen with an auxiliary telemetry gap remains eligible
to direct the next implementation step when its core identity, correctness,
route, ownership, and metric are intact; label the affected timing authority.
Architecture-selection, release-promotion, and publication-grade comparisons
fail closed on their complete predeclared environment protocol. Profiler
attribution also fails closed when foreign ownership or overlap makes its
counters ambiguous.

Store preflight and run artifacts below `.q3x-work/` with the evidence record.
Do not create project-owned monitoring output in the project owner's home
directory, and do not leave large or persistent captures in `/tmp`.

## Layer panels and cache state

A single convenient layer, including layer 0, is not a general production
sample. Before viewing a `local_mutation` result, each projection or stateful
role must freeze a layer panel chosen from all eligible layers using payload
features such as quantized-code and scale-exponent histograms, zero fraction,
entropy, and compressibility.  The development panel must contain at least
four layers spanning low, median, high, and extreme payloads.  Gate, Up, Down,
FP8 QKV/Z/O/KV, Attention, and SSM use independent panels.

Final promotion must either sweep every eligible layer or exercise all of them
in the natural T3 model path.  Component timing should rotate through real
layers or explicitly scrub cache so that repeated use of one tensor does not
create an accidental warm-cache gate.  If warm-cache behavior is relevant, it
is recorded as a separate steady-state result.

## Staged development and threshold recalibration

Synthetic thresholds and absolute timings are not transferred to real
payloads. Development uses different rules for different decision units.

For a `local_mutation` inside a named work package:

1. Freeze the package-local incumbent, real payload manifest, applicable layer
   panel, decision-class environment record, and measurement protocol before
   inspecting the mutation.
2. Complete the minimum safe admission before the first device run: build and
   launch contracts, bounds/resource sanity, route isolation, and at least one
   applicable correctness oracle. Exhaustive qualification is deferred unless
   safety or the numerical contract requires it.
3. Run the cheapest representative real-payload T2 or T3 comparison capable of
   measuring the package's local hypothesis. Use the same engine and ELF where
   practical, explicit route hits, and a mirrored incumbent/mutation order.
   The work package may use a short prompt, P513, a layer panel, or a component
   boundary here because this result has local authority only.
4. A negative local result normally archives that mutation. A
   `dependency_only` mutation may remain until the declared coupled test when
   the interaction was predeclared; the record must not call it a win.
5. A positive first result unlocks the package's complete correctness,
   state/Graph/resource, and noise qualification. When the package specifies
   six B-C-C-B rounds or another paired protocol, preserve every raw round,
   reversal, and failure.
6. Retain a stably positive mutation beyond local noise as the new
   package-local incumbent. Gate evidence does not set Down or FP8 thresholds,
   and per-role results remain local until composition.
7. At the package deadline, build the coupled `architecture_candidate` or
   archive the package. Do not convert a collection of uncomposed local
   timings into an end-to-end claim.

These minimum-admission, pair-table, local noise, layer-panel, and profiler
rules are **local optimization engineering rules**. Outside a named local
work package they do not determine global priority, require the project to
continue scanning, or supersede a real API architecture result.

For an `architecture_candidate`:

1. Freeze the cumulative native architecture incumbent and one final binary;
   verify natural layer order, exact model/state, explicit route identity, and
   the product API semantics.
2. Run the cold/no-cache target-length witnesses in fail-fast 40K, 60K, 130K
   order. A neutral or negative whole-path result may close the architecture
   version immediately. A bounded T4 profile may answer a named causal
   question but cannot reverse the result.
3. Select a cumulative architecture only after it is stably positive over the
   predeclared witness set and preserves the numerical/output contract. Do not
   impose the vLLM starting-line margin or a cuBLASLt external-ceiling margin
   on this incremental native architecture decision.

For a `release_candidate`, repeat the formal protocol after the production
binary, selector, deployment plan, and default API configuration are frozen.
Production promotion is cumulative against the current qualified native
production baseline when one exists, or otherwise the frozen native release
baseline named by `CURRENT_STATUS.md`. It requires the predeclared engineering
margin where applicable, independent-process repetition, complete role/shape
and target-length coverage, public capability evidence, and T3. A direction
or architecture screen may never update production dispatch, recalibrate a
published threshold, or serve as release evidence.

## Historical evidence

Historical artifacts remain immutable.  An evidence registry should classify
each result as `synthetic_only`, `checkpoint_weight_only`,
`checkpoint_full_payload`, or `full_model`, and mark its authority as
`non_admissible_legacy`, `provisional`, `confirmed`, or `superseded`.

- Synthetic microbenchmarks backed by later real full-model evidence may
  retain the production decision, but their mechanism claims are directional.
- A historical synthetic-only default-route selection is provisional and must
  be requalified through the active Roadmap and release protocol; it is not
  reverted without replacement evidence.
- A synthetic-only rejection is no longer a permanent stop.  It is rerun only
  if that mechanism re-enters the active roadmap; old negative cells need not
  be migrated wholesale.

## Profiling order

NCU or NSys follows, never precedes, a completed and valid T2 or T3 timing run
or direction screen on the same real payload. A rejected `local_mutation` or
`architecture_candidate` may be profiled when an explicit causal question can
guide the next design, but that work must be bounded and the profile is
diagnostic: it cannot reverse the rejection. A negative decision unit may
instead be archived immediately; profiler completion is not a rejection gate.
The authoritative T2 retention comparison is always the native package-local
incumbent versus the native `local_mutation`. A named local-work-package
manifest owns the exact reference components and boundaries to profile. No
historical component-specific profiling prescription applies outside that
package, and an external-library profile remains diagnostic only.

Collect occupancy and launch resources, pipeline/SASS structure, issue and
Tensor activity, HMMA count, stall reasons, global/L2/L1/shared traffic, and
bank conflicts.  Normalize counters by useful work and record counter
denominators.  Profiler percentages overlap and must not be added into a
latency explanation.  A gap is eligible for implementation only after it has
a quantified cycle or byte ceiling on the real payload; the resulting change
must return through T2 and T3.
