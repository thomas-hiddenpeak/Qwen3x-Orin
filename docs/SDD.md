---
q3x_document:
  id: q3x-system-sdd
  class: active
  status: active
  owner: project-owner
  authority: end-to-end external-to-internal runner system design
  effective: 2026-08-09
  last_reviewed: 2026-08-27
  supersedes: []
  superseded_by: []
  ssot_for: runner product shape, system boundaries, lifecycle, and release architecture
  review_trigger: any product API, lifecycle, state, DeploymentPlan, or release-boundary change
---

# Qwen3x-Orin system design document

Status: active system design, effective 2026-08-09.

This document defines the intended product shape and the stable system
boundaries that must produce it. It starts at the externally observable
runner rather than at checkpoint loading or an individual kernel. Current
implementation and qualification state belongs in
[`CURRENT_STATUS.md`](CURRENT_STATUS.md); evidence and historical experiments
must not be copied here as if they were current capabilities.

The [engineering constitution](ENGINEERING_CONSTITUTION.md) controls mission,
targets, and conflict resolution. The
[real-model performance policy](REAL_MODEL_PERFORMANCE_POLICY.md) controls the
authority of evidence. This SDD controls how the product is decomposed so
those requirements can reach implementation and implementation gains can
reach the product.

## 1. Design method: begin with the delivered runner

Qwen3x-Orin is a specialized runner for a pinned model family, numerical
format, and Jetson AGX Orin SM87 target. Its primary artifact is not a loader,
library, benchmark, or collection of kernels. It is an externally usable,
accuracy-preserving OpenAI-compatible service whose behavior can be evaluated
by a public client without repository-internal knowledge.

The project uses **bidirectional leakage** as its engineering method:

1. final product constraints leak downward through API, admission, scheduling,
   memory, operators, layouts, and kernels;
2. implementation changes occur at the smallest coherent boundary that can
   answer the resulting problem;
3. local mechanisms are composed into an executable architecture candidate;
4. value leaks upward through the real generation path and is selected at the
   external API;
5. only a fully qualified candidate becomes a release.

This is constrained evolution, not random trial. Design fixes the habitat,
hard invariants, observability, and admissible state transitions. Real product
fitness selects among implementations. Profiler counters explain that
selection; they do not become the selection surface.

Three engineering units must remain distinct:

- a **local mutation** changes one bounded mechanism or subsystem inside a
  declared local work package;
- an **architecture candidate** composes the mutually dependent mutations
  needed to realize one executable end-to-end dataflow; and
- a **release candidate** packages that architecture in the exact production
  artifact and faces every product constraint.

A local mutation is selected with its real-route local comparator and does not
have to move the whole API above system noise independently. The architecture
candidate is the unit that must return to the target-representative API for
whole-product selection.

The governing shorthand is:

> Constraints leak down; changes may be local; attribution follows the whole
> path; selection is global; value must reach the API.

## 2. End-state product contract

### 2.1 External behavior

The release runner shall:

- load the pinned Qwen3.6 27B NVFP4 model family and advertise its authenticated
  identity;
- expose versioned OpenAI-compatible text completion and chat-completion
  endpoints to an external client;
- support streaming in which the first visible generated token is an actual
  committed model token, not a role-only, heartbeat, or empty event;
- return exact prompt, completion, finish, and error accounting;
- reject unsupported parameters, over-capacity requests, incompatible assets,
  and unplanned routes before partial execution whenever possible;
- preserve the production numerical and accuracy contract, with MTP excluded
  from the current Prefill and Decode target path;
- make cache policy explicit. A cold/no-cache request may not silently use
  Prefix/KV reuse or truncate input; and
- expose sufficient machine-readable identity and timing to reproduce the
  route that served each request.

The minimum endpoint surface is `GET /healthz`, `GET /v1/models`,
`POST /v1/completions`, and `POST /v1/chat/completions`. The production
network, authentication, authorization, TLS, tenant, overload, cancellation,
and shutdown contracts must be explicit in a release profile. A loopback-only
evaluation adapter is a useful implementation stage, but is not by itself the
delivered serving product.

### 2.2 Locked fitness contract

The production candidate is evaluated as a vector, not as one average:

| Dimension | Locked workload and requirement | Selection rule |
| --- | --- | --- |
| Accuracy | Pinned deterministic oracles plus a parseable public capability suite | No production regression; failure is disqualifying |
| 40K Prefill | Cold/no-cache, real Agent prompt, concurrency one, first visible generated token | TTFT at most 2 s |
| 60K Prefill | Same contract | TTFT at most 2 s |
| 130K Prefill | Same contract, no silent truncation | TTFT at most 4 s |
| Decode | Single request, committed output tokens, MTP disabled | At least 10 token/s and at most 100 ms/token |
| Protocol | Complete valid responses/streams, correct usage and finish semantics | No failed or incomplete request in the release panel |
| Capacity | Declared context fits the planned resident memory without oversubscription or unplanned growth | Fail closed before execution if it does not fit |
| Reproducibility | Pinned binary, plan, model, host state, request hashes, and repeated process runs | Required for qualification |

Accuracy, protocol validity, no-MTP, no-cache, exact token identity, and no
truncation are hard constraints. A faster invalid request is not a point on
the performance frontier. A mean may not hide failure in one required length
bucket. Tail latency, memory headroom, thermal state, and request success are
guardrails even when the headline target is TTFT or token/s.

The server's measured pure Prompt-Prefill interval, projection/GDN/Attention
budgets, kernel timings, bandwidth, occupancy, and stall counters are
diagnostic coordinates. They are not replacements for this vector.

### 2.3 Specialized release profile

One release must resolve to one auditable production route. The production
profile shall satisfy all of the following:

- configure and build with `BUILD_TESTING=OFF`;
- contain no test/admission selector, environment-dependent route composition,
  or implicit numerical fallback;
- contain no production dependency, symbol, context, allocation, dispatch, or
  fallback for cuBLASLt;
- perform no request-time JIT compilation, autotuning, checkpoint repacking,
  layout discovery, workspace growth, or algorithm search;
- use only accuracy-qualified numerical routes and keep MTP disabled for the
  locked targets;
- bind all optimized routes through one authenticated DeploymentPlan; and
- pass install-tree, dependency, symbol, route-coverage, API, accuracy,
  resource, and performance attestation using the exact installed artifacts.

A development binary with test-only admissions enabled may establish an
architecture result, but it cannot be renamed or described as the release.

## 3. DeploymentPlan contract

JIT and autotuning in proven engines may be studied offline to discover
structure. Qwen3x-Orin materializes the result as an authenticated AOT
`DeploymentPlan`. It is the stable bridge between checkpoint identity and the
release binary.

The plan shall bind at least:

- format version and generator version;
- source git revision, installed binary build ID and SHA-256;
- CUDA, driver, JetPack/L4T, SM, SM count, and declared power/clock envelope;
- checkpoint repository, revision, config, tokenizer, quantization config,
  index, shard hashes, and every consumed tensor identity;
- every supported API workload bucket and its admission limits;
- logical operation, layer family, exact M/N/K or bucket, dtypes, numerical
  contract, physical layout version, kernel/tactic ID, launch geometry, and
  workspace requirements;
- persistent derived sidecars or packed layouts, each authenticated against
  its source tensor and generator;
- request-state, KV/state, scratch, staging, stream, event, and buffer lifetime
  plans for each supported context bucket;
- graph/capture segments and explicit synchronization boundaries;
- complete route-coverage expectations and forbidden fallback routes; and
- the evidence registry entries that qualified this exact plan.

Production startup validates the plan and all referenced artifacts before it
opens the listener. A missing shape, hash mismatch, resource shortfall,
unsupported host, unknown tactic, or route-coverage gap fails closed. The
runtime may select among predeclared plan entries using request facts; it may
not invent a new route.

### 3.1 First installable production profile

The first ordinary installed profile is
`q3x.sm87.production.p40.legacy-c512-exact.v3`. It is built by the
`orin-release` preset as Release, SM87, and `BUILD_TESTING=OFF`, and installs
`qwen3x-eval-server`, `qwen3x-orin`, `qwen3x-inspect`, and the versioned 0.7.0
package. The server profile admits `prompt + output - 1 <= 44,095`, exposes a
40,000-token product prompt and a 4,096-token output ceiling, fixes the exact
Legacy-C512 request arena at 3,070,908,416 bytes, and retains an 8-GiB
post-create free-memory reserve.

This profile is a sealed route rather than a public tactic selector. Before
opening its listener it requires the complete 208-projection FP8 Prefill
supermatrix, the 64-layer FP8 Decode output layout, the 64-layer NVFP4
Gate/Up coupled-feed layout, the pinned 53-layer scale6 and Down
consumer-order layouts, and the 25-slot short-position Decode Graph cache.
Their planned retained acceleration payload is 18,228,101,120 bytes. Missing,
partial, memory-rejected, or silently demoted inventory fails startup; the
ordinary Release binary does not consult environment variables to compose
the route.

`GET /healthz` remains public. When a key file is configured, models and
generation require Bearer authentication; a non-loopback listener is rejected
without an owner-only key file. TLS termination is supplied by a trusted
local reverse proxy. This profile is the installed production-shaped artifact
used for real-model selection, but remains `release_qualified=false` until its
installed binary passes the target-length, Decode, accuracy, stability, and
capability gates in Sections 2.2 and 11.

## 4. System decomposition from the API inward

```text
external OpenAI-compatible client
  -> protocol parser / stream contract / request identity
  -> admission, capacity, queueing, cancellation, overload
  -> tokenizer or exact token-ID boundary
  -> generation controller
       -> Prefill plan
       -> exact Prefill-to-Decode state commit
       -> Decode plan and token publication
  -> model execution graph
       -> Attention and KV ownership
       -> GDN/SSM recurrence and state ownership
       -> MLP Gate/Up and Down ownership
       -> normalization, embedding, logits and sampling
  -> kernel registry and shape-specific tactics
  -> authenticated resident weights, derived layouts and workspaces
  -> CUDA streams, events, arenas and SM87 device
```

Every boundary has a product-facing reason:

| Boundary | Product constraint leaking downward | Required design response |
| --- | --- | --- |
| Protocol | External correctness and observable TTFT | Exact SSE semantics, request hashes, committed-token timestamp |
| Admission | 40K--130K capacity and predictable failure | Exact memory plan, no truncation, bounded queues, pre-execution rejection |
| Generation controller | Prefill and Decode have different fitness targets | Separate plans, metrics, state boundaries, and selectors |
| Execution graph | Whole-model latency must improve | Natural model order, call-count/traffic ownership, no hidden fallback |
| Operator dataflow | Long prompts must reuse work and expose parallelism | Prompt-span scheduling, planned residency, fusion only across exact boundaries |
| Kernel tactics | SM87 and asymmetric shapes demand specialization | Shape/family-specific layouts, pipelines, launch geometry, exact arithmetic |
| Asset/runtime ownership | Stable startup and request latency | Authenticated AOT plan, persistent derived layouts, allocation-free hot path |

Subsystem contracts refine these boundaries and are indexed from
[`DESIGN.md`](DESIGN.md). They may choose local physical layouts without
changing the API, numerical, state, or release contract here.

## 5. Lifecycle state machines

### 5.1 Startup state machine

```text
Created
  -> ConfigurationValidated
  -> DeploymentPlanAuthenticated
  -> ModelAndSidecarsAuthenticated
  -> ResidentAssetsLoaded
  -> RequestResourcesPlannedAndAllocated
  -> RoutesBoundAndCoverageChecked
  -> WarmedAndSelfTested
  -> Ready
```

Any transition may enter `StartupFailed`, which releases project-owned
resources and never exposes a ready listener. `Ready` may enter `Draining`,
then `Stopped`; fatal device or invariant failures enter `Unhealthy` and stop
new admission. Startup logs the identity and duration of every transition.

The listener opens only after `Ready`. Readiness proves authenticated identity
and route coverage, not merely that a socket accepts connections.

### 5.2 Request state machine

```text
Received
  -> ProtocolValidated
  -> TokenizedOrTokenIdsValidated
  -> CapacityAdmitted
  -> Queued
  -> Prefilling
  -> PrefillStateCommitted
  -> Decoding
  -> FirstTokenCommitted
  -> Streaming
  -> Finished
  -> ResourcesReleased
```

Before GPU work, a request can enter `Rejected`. Before a committed token, a
runtime failure returns an ordinary protocol error. After streaming starts it
enters `StreamFailed`, emits the defined terminal error if possible, closes,
and releases resources. Cancellation enters `Cancelled` at a declared safe
boundary; the target design must not require a full 130K Prefill to finish
before observing cancellation.

The request records queue, tokenize, admission, Prefill, state-commit, Decode,
first-token, stream, and cleanup intervals independently. It never derives a
pure Prefill metric from TTFT without accounting for the other intervals.

Ordinary legacy request reuse is lifecycle-derived rather than caller-selected.
A newly created or successfully full-reset arena is already clean; an exactly
committed successful request may clear complete Conv/GDN state plus only its
written K/V prefix; cancellation, poison, an uncommitted whole request, a
position mismatch, or any uncertain boundary requires the conservative full
reset. The request witness records the selected mode, cleared positions,
zeroed bytes, and synchronized cleanup duration. This public receipt and the
associated C++ object-layout changes define package ABI 0.7.0; 0.x consumers
must rebuild against that exact installed version.

The v3 profile identity supersedes v2 only for the current installed route.
Version 2 remains the immutable identity of the 0.6.0 evidence tuple; it cannot
name the 0.7.0 request-start cleanup and witness contract.

### 5.3 Engineering evolution state machine

```text
ProductObservationOrTargetGap
  -> WholePathReproduction
  -> DownwardLeakageTrace
  -> DeclaredLocalWorkPackage
  -> LocalMutationAndRealRouteLocalSelection
       -> RejectedAndArchived
       -> LocallyRetainedForNamedComposition
  -> ExecutableArchitectureCandidate
  -> MinimalTargetRepresentativeAPISelection
       -> RejectedAndArchived
       -> DirectionPositive
  -> LocalAndSystemQualification
  -> RepeatedRealAPIFitnessSelection
  -> ReleaseCandidate
  -> InstalledArtifactAttestation
  -> Production
```

`LocallyRetainedForNamedComposition` is not a whole-product result, and
`DirectionPositive` is not production qualification. A local mechanism may
remain an explicitly bounded constituent of an architecture candidate when
its product effect is temporarily masked, but it has no independent
production claim. It must participate in the named whole-path candidate by
its predeclared expiry point or be archived.

A production regression returns the release to `Unqualified`; it does not
silently lower the target or promote a fallback.

## 6. Bidirectional leakage traceability

Every material performance or capacity change owns one trace record with two
directions.

### 6.1 Downward trace

The record starts with a user-visible gap and identifies:

1. exact external workload, expected fitness component, and observed result;
2. API/queue/tokenization/engine interval decomposition;
3. Prefill or Decode plan and repeated model-layer topology;
4. dominant operator family, call count, state/synchronization boundaries,
   bytes, and compute;
5. data residency and ownership across host, DRAM, L2, shared memory,
   registers, streams, and buffers; and
6. for every relevant proven implementation, the invariant mathematics,
   dataflow, work decomposition, scheduling, specialization, and tuning policy
   separated from architecture-specific instructions and resources, together
   with the proposed SM87 translation; and
7. the smallest coherent architectural mechanism capable of moving the
   product budget.

When a gap is large, the trace covers the full Gate/Up, Down, FP8 projection,
Attention, and GDN/SSM paths and studies the proven vLLM, FlashInfer, Triton,
FLA, and Mamba structures before local parameter scanning.

### 6.2 Upward trace

The same record follows the implementation back upward:

1. kernel or subsystem mechanism and its exact numerical contract;
2. route hit and absence of fallback in the real model path;
3. operator/layer or scheduling effect under real weights and state;
4. complete Prefill or Decode budget change;
5. external API fitness-vector change; and
6. qualification and release-attestation identity.

If upward propagation stops, the trace names the boundary that absorbed the
gain: call topology, conversion, synchronization, memory movement, queueing,
another critical path, or measurement noise. The next action changes that
boundary or closes the work. A faster isolated kernel is evidence of a local
mechanism, not proof of runner value.

## 7. Local work-package boundary

Local optimization remains essential, but its engineering rules are scoped
to a declared **local work package**. They are not the default methodology for
project prioritization, architecture selection, or product release.

A work package must declare before implementation:

- the parent product gap and downward leakage trace;
- the exact real production route, operation family, layers, shapes, and
  context buckets it affects;
- the mechanism and why its gain can propagate upward;
- numerical/state contract and forbidden accuracy changes;
- incumbent and candidate identity in one executable architecture;
- an estimated whole-path ceiling or prerequisite value;
- smallest real-route local comparator and admission screen, local
  correctness/resource gates, stop-loss, architecture-candidate API return
  point, and expiry/combination point; and
- artifact location under `.q3x-work/`, tracked evidence destination, and
  rollback boundary.

Within that boundary, component harnesses, single-variable controls, paired
timing, noise calibration, tile or pipeline choices, NCU counter matching,
SASS review, synthetic correctness enumeration, and mechanism-specific gates
are appropriate. These rules answer a local causal question. They must not:

- redefine the product goal around the component harness;
- require every useful constituent to beat vLLM or a terminal reference alone;
- accumulate unrelated wins without a named architecture composition;
- promote a test/admission selector into the release; or
- claim product value before the upward trace reaches the real API.

Architecture work may contain several interdependent local mechanisms because
their variables are not necessarily orthogonal. The combined candidate is
the selection unit. Local evidence explains membership in that candidate;
the external fitness vector decides whether the candidate survives.

## 8. Prefill and Decode logical boundary

Prefill and Decode share authenticated weights and one model graph but are
separate execution products:

| Concern | Prefill | Decode |
| --- | --- | --- |
| Primary input | Prompt span/chunks | One committed token position |
| Primary fitness | Cold/no-cache TTFT and pure Prompt-Prefill interval | Inter-token latency/token/s |
| Parallelism | Large-M and sequence/chunk parallelism | Batch-one M=1 latency and state locality |
| Persistent state | Builds KV and recurrent boundary state | Consumes and updates that state once per token |
| Kernel ownership | Large-M projections, Prefill Attention, chunked GDN/SSM | GEMV/small-M projections, Decode Attention/GDN |
| Scheduling | Prompt-span/layer/dataflow plan with explicit overlap | Low-latency token plan, graph/launch reuse |
| Evidence | 40K/60K/130K API buckets plus internal Prefill spans | Short and long output API runs plus per-token spans |

The handoff is one explicit `PrefillStateCommitted` boundary. It defines KV
length/layout, recurrent state dtype/layout/rounding, convolution state,
position/RoPE state, next-token logits, stream/event ownership, and visibility
rules. A Prefill optimization may not change this boundary silently. Decode
must not depend on a test-only Prefill representation.

Separate planners and route identities are required even if one runner object
and some kernels are shared. Prefill work must not reactivate Decode tuning
unless the external fitness vector identifies Decode as the limiting
component.

## 9. Dataflow, buffering, and ownership requirements

The production hot path shall make ownership explicit for every persistent and
transient byte:

- checkpoint and derived layout lifetime is engine-wide;
- request state and KV/recurrent state lifetime is request-wide;
- layer/tile scratch has one declared producer, consumer, stream, reuse event,
  and maximum size;
- staging buffers are bounded and cannot grow during a request;
- stream/event dependencies are represented in the plan, not discovered by
  unconditional device synchronization; and
- double/triple buffering is selected only where a declared producer/consumer
  overlap exists and its memory cost fits every required context bucket.

The target Prefill planner must expose prompt-span reuse and avoid forcing
long prompts to repeat a complete weight traversal or global synchronization
for every small tile when an exact layer-major or otherwise fused dataflow can
preserve state semantics. Gate/Up and Down retain shape-specific ownership;
FP8 projections, Attention, and GDN/SSM retain their own dataflow contracts.
One universal tile or pipeline configuration is not assumed.

Fusion is a consequence of shared lifetime and eliminated boundaries. It is
not a goal by kernel count alone. A fusion that changes rounding, extends a
live range beyond capacity, serializes independent stages, or blocks an
upstream gain fails the system contract even if its isolated launch count is
lower.

### 9.1 Mathematical equivalence before physical mapping

Every dominant Prefill or Decode family has a four-level computation ledger:

1. the real-number tensor, causal reduction, or state-transition equation;
2. the actual finite-precision operator, including decode, accumulation order,
   special values, rounding, and publication points;
3. the observable output and persistent-state boundary consumed by the next
   subsystem or request phase; and
4. the physical ownership of the irreducible work across DRAM, L2, shared
   memory, registers, CTAs, warps, streams, and request lifetimes.

Design proceeds in that order. A candidate first looks for a lawful
mathematical reformulation that removes materialization, repeated transforms,
repeated history traversal, or unnecessary state publication. Only then does
it select layouts, tiles, stages, buffering, cache policy, or launch geometry.
The physical implementation must preserve level 2 and level 3; equality at
level 1 alone is insufficient because floating-point arithmetic and a rounded
recurrence are not generally associative.

Each active architecture work package records an **equivalence ledger**. For
every proposed fusion or reorder it names the original expression, transformed
expression, preserved accumulation and rounding boundaries, changed ownership,
added arithmetic and instructions, work or movement eliminated at every memory
level, expected resource/critical-path transfer, exact oracle, target API
effect, and any reason the transformation is research-only. This ledger is
also the stop rule: when a mechanism merely reschedules the same ownership
without removing a measured limiting dependency, or trades traffic for a new
feed/occupancy bottleneck, the architecture must be reconsidered rather than
parameter-scanned.

### 9.2 Cross-architecture reference translation

Reference eligibility is independent of an upstream device-support guard.
Each architecture candidate may study implementations for newer, data-center,
or otherwise different devices, but must record three separate layers:

1. invariant mathematics, dataflow, lifetimes, work partition, and planning;
2. unavailable ISA/resource mechanisms such as TMA, WGMMA, clusters, TMEM, or
   native block-scaled FP4; and
3. the SM87 realization using its own load, buffering, synchronization, MMA,
   residency, and AOT-plan mechanisms.

The translation is selected for eliminated movement and whole-path fitness,
not source similarity. Runtime JIT/autotuning from a reference engine may
discover a plan offline; the release receives only the authenticated AOT
result. An added-compute translation is admissible only under the Constitution's
compute-for-movement rule and the equivalence ledger above.

## 10. Observability contract

Every startup and request emits a stable request/run ID and structured fields
sufficient to reconstruct the selected path. Required records include:

- git revision, binary build ID/hash, release profile and DeploymentPlan hash;
- checkpoint/config/tokenizer/shard and derived-sidecar identities;
- device, CUDA/driver/JetPack, power/clock/thermal envelope and host-preflight
  record for performance evidence;
- API endpoint, request hash, token-ID hash, prompt/completion counts,
  concurrency, cache/MTP policy, maximum lengths, finish and error state;
- admission decision, planned arena bytes, actual peak memory and remaining
  headroom;
- Prefill planner, chunk/span sizes, every route family hit, fallback count,
  synchronization count, and Prefill state-boundary identity;
- `request_received`, `queued`, `tokenization`, `admission`, `prefill`,
  `prefill_state_commit`, `decode`, `first_token_committed`, `stream`, and
  `request_finished` timestamps or intervals; and
- generated-token identities for deterministic oracles and output-completeness
  checks.

The server reports both user-visible TTFT and its exact decomposition. Pure
Prompt-Prefill begins only when the admitted prompt execution begins and ends
when the complete Prefill state is visible to Decode. Logger-window throughput
is labelled as windowed aggregate telemetry and never substituted for a
single-request pure interval.

Missing identity, unexpected route, hidden fallback, or an incomplete stream
invalidates that request's timing, selection, and promotion record; only
independently attributable non-performance diagnostics may remain. An unowned
GPU consumer, confirmed material contention, or timing-overlap ambiguity also
invalidates timing. Supporting host telemetry otherwise qualifies ordinary
engineering evidence under the decision class in
[`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md); it does
not independently erase unaffected correctness, route, state, or output facts.

## 11. Release attestation

Release attestation operates on the installed `BUILD_TESTING=OFF` artifacts,
not a development build. One machine-readable manifest shall prove:

1. source revision, clean diff policy, compiler/linker options and binary
   hashes;
2. absence of test-only admissions, cuBLASLt dependencies/symbols, JIT and
   runtime autotuning hooks;
3. authenticated DeploymentPlan, checkpoint, tokenizer, derived sidecars and
   route coverage;
4. startup, malformed-input, overload, cancellation, shutdown, memory and
   long-running stability results;
5. deterministic numerical oracles and a valid public capability result;
6. cold/no-cache 40K, 60K, 130K and Decode fitness results under clean-host
   preflight, including raw artifacts and independent-process repetition;
7. exact API request/response/stream conformance and usage accounting; and
8. installation content, license/provenance, rollback identity and known
   limitations.

Qualification attaches to the exact tuple `(binary, DeploymentPlan, model,
host/software envelope, workload protocol)`. Changing any member invalidates
the affected portion and requires re-attestation. Historical evidence remains
immutable but cannot automatically qualify a new tuple.

## 12. Evidence and host execution boundary

Performance begins on the real model's production generation/API path.
Synthetic matrices have correctness, robustness, and device-health roles only.
Real-model runs and profiler captures record host/device preflight. On Jetson,
load and ownership are established with `tegrastats`, CPU/process inspection,
and GPU-device handle inspection; the incomplete Jetson `nvidia-smi` is not an
ownership authority. Device ownership, safety, and confirmed material
contention remain hard gates. Other environment observations qualify ordinary
direction evidence, while architecture-selection and release records fail
closed on the stricter protocol defined by the real-model performance policy.

All generated builds, profiles, temporary sources, evaluation state, and tool
environments stay under the repository's ignored `.q3x-work/` tree. User model
directories, `~/vllmEvn`, and shared caches are read-only external inputs.
Large or persistent `/tmp` artifacts are prohibited; bounded tool-required
files are removed when their exact ownership is known.

## 13. Change control

- Amend mission or locked product constraints in the constitution first.
- Amend stable cross-subsystem boundaries in this SDD.
- Record a point-in-time capability only in `CURRENT_STATUS.md` with source
  identity and evidence authority.
- Put subsystem physical/numerical detail in the owning contract indexed by
  `DESIGN.md`.
- Put pending ordered work in `ROADMAP.md`.
- Put decisions and rejected alternatives in tracked ADR/evidence records;
  do not turn an old result into a current status declaration.

A code change that crosses an SDD boundary must update the affected contract
in the same atomic milestone. Context compaction or contributor handoff does
not relax these requirements.
