---
q3x_document:
  id: q3x-prefill-architecture
  class: active
  status: active
  owner: prefill-maintainers
  authority: Prefill subsystem boundary, state contract, and architecture-candidate requirements
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: [docs/PREFILL_ARCHITECTURE_RESET_LEGACY.md, docs/PREFILL_REFERENCE_AUDIT.md]
  superseded_by: []
  ssot_for: Prefill inputs, outputs, ownership, synchronization, failure, and Decode handoff
  review_trigger: Prefill boundary, state ABI, execution-plan interface, or handoff change
---

# Prefill subsystem design

## 1. Authority and parent boundary

This subsystem SDD refines the Prefill boundary in
[`SDD.md`](SDD.md). It defines what Prefill must consume, produce, own,
synchronize, report, and commit so that the external runner can satisfy its
locked product contract. It does not define the project's delivery order,
record current implementation state, retain performance history, or prescribe
a kernel mechanism.

[`ROADMAP.md`](ROADMAP.md) is the only active delivery order.
[`CURRENT_STATUS.md`](CURRENT_STATUS.md) is the only current implementation
and qualification snapshot. Evidence and candidate selection follow
[`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md), while
the external witness protocol is owned by
[`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md).

The following product constraints leak into every Prefill design:

- cold, no-cache 40K and 60K real-Agent prompts must reach the first visible
  generated token within two seconds;
- the pinned approximately 130K prompt must do so within four seconds;
- the complete prompt must be consumed without silent truncation, Prefix/KV
  reuse, or an approximate numerical route;
- production accuracy and the declared Prefill-to-Decode state semantics must
  not regress;
- MTP is excluded from the current target path; and
- cuBLASLt is a reference only and has no production dependency, dispatch, or
  fallback eligibility.

These are system constraints, not local performance cells. A subsystem or
mechanism result has value only when it composes into a named architecture
candidate and returns through the real API path.

## 2. Prefill boundary

Prefill begins after protocol validation, exact tokenization or token-ID
validation, capacity admission, queue release, and request-resource binding.
It ends only when the complete prompt state and next-token inputs have been
atomically published at `PrefillStateCommitted` for Decode.

Queueing, tokenization, admission, first-token Decode, and response
publication are outside the pure Prefill interval even though they remain
part of user-visible TTFT. The runner must report those intervals separately;
it may not infer pure Prefill from TTFT by assumption.

### 2.1 Inputs

An admitted Prefill invocation receives:

- the exact prompt token IDs, their authenticated tokenizer/model identity,
  and their declared position range;
- the request's cache policy, context/output capacity, cancellation state,
  and preallocated memory-plan binding;
- the authenticated model weights, derived layouts, tactic identities, and
  supported context bucket from the release `DeploymentPlan`;
- an empty cold-request state for the locked witnesses, or an explicitly
  declared state only for a separately specified workload; and
- CUDA streams, events, arenas, and route identifiers allocated before the
  hot path begins.

The production Prefill path may not discover a new tactic, repack weights,
grow workspace, compile code, or silently change numerical mode during a
request.

### 2.2 Outputs

A successful Prefill invocation publishes one complete handoff containing:

- full-Attention KV state for every consumed prompt position in the declared
  production layout;
- GDN/SSM recurrent and convolution state at the exact prompt boundary;
- position and RoPE state needed by the first Decode step;
- next-token logits or the exact final representation from which the planned
  Decode entry computes them;
- completion events that make every handoff component visible to its declared
  Decode consumer; and
- route, interval, synchronization, fallback, resource, and prompt-consumption
  evidence for the request record.

No partially published state is a valid output. The externally visible first
token may be produced only after the complete handoff is committed.

## 3. State, lifetime, and ownership

Ownership is explicit at every lifetime:

| State or asset | Owner | Lifetime | Publication rule |
| --- | --- | --- | --- |
| Checkpoint weights and authenticated derived layouts | Engine / `DeploymentPlan` | Engine-wide | Read-only after readiness; no request-time replacement |
| Prompt token IDs and request metadata | Request controller | Request-wide | Immutable after admission |
| Full-Attention KV | Request state | Request-wide through Decode | Written by Prefill, transferred only at the handoff event |
| GDN/SSM recurrent and convolution state | Request state | Request-wide through Decode | Exact declared dtype, layout, and rounding at handoff |
| Position and RoPE state | Request state | Request-wide through Decode | Advances exactly by the admitted prompt length |
| Layer/span activations and scratch | Prefill execution plan | Bounded producer-consumer interval | Reused only after the declared consumer event |
| Streams and events | Runner execution plan | Engine- or request-wide as declared | Dependencies are plan entries, not implicit global barriers |
| Next-token representation/logits | Prefill/Decode boundary | Until first Decode consumption | Published atomically with the remaining handoff state |

Every buffer has one producer, one or more named consumers, a maximum size,
and a reuse event. A double-, triple-, or deeper buffer is allowed only when
the execution plan identifies a real producer/consumer overlap and the full
capacity profile retains required memory headroom. Buffer count is therefore
an architecture decision derived from dependencies, not a global mechanism
rule.

## 4. Synchronization and execution semantics

The Prefill plan must preserve model order and recurrent dependencies while
exposing independent work across prompt spans, operator families, layers, or
streams where the exact state contract permits it.

- Stream and event dependencies are declared in the `DeploymentPlan` and
  request execution plan.
- An unconditional device synchronization is not a normal layer, tile, or
  span boundary. If one remains, the architecture candidate must identify the
  correctness dependency it protects and its user-visible budget.
- Scratch may be recycled only after its last planned consumer completes.
- Cancellation is observed at bounded safe points without committing a
  partial Prefill state.
- The final state-commit event is the only transition that authorizes Decode
  to consume Prefill-owned state.

The plan may choose layer-major, span-major, pipelined, fused, or another
execution shape. This SDD does not select one in advance. The selected design
must explain prompt-wide reuse, state progression, memory capacity, and every
barrier for all locked context buckets.

## 5. Prefill-to-Decode handoff

`PrefillStateCommitted` is a versioned ABI, not an informal call boundary. Its
identity binds:

- the exact number and IDs of consumed prompt tokens;
- KV length, dtype, layout, addressing, and visibility;
- recurrent and convolution state dtype, layout, rounding, and position;
- position/RoPE state and the first Decode token index;
- next-token logits or the planned representation and owning producer;
- producing and consuming stream/event identities; and
- model, binary, release plan, numerical route, and state-schema versions.

Decode may not infer missing fields or depend on a test-only Prefill layout.
Prefill may not change this ABI to obtain a local speed result without an
atomic update to the parent SDD, Decode contract, implementation, oracles, and
release plan. Handoff equivalence is checked before performance selection.

## 6. Numerical route, failure, and cancellation contract

The active production contract is the accuracy-preserving route defined by
the parent SDD and pinned model/runtime contracts. An approximate activation,
recurrent-state, Attention, or handoff representation is not a Prefill
architecture candidate under this contract. Any opt-in numerical research
requires an explicit project-owner amendment and a separately identified work
package, route, evidence set, and non-production artifact; it cannot share an
implicit selector or promotion claim with the production route.

Unsupported context length, insufficient planned memory, unknown tactic,
artifact mismatch, route-coverage gap, forbidden cache mode, or incompatible
handoff schema fails closed before Prefill begins whenever the fact is known.

After Prefill begins:

- a device, numerical-invariant, cancellation, or execution-plan failure
  prevents `PrefillStateCommitted`;
- no partially written KV or recurrent state becomes visible to Decode;
- request-owned resources are released only after their producers are safely
  quiesced;
- the request state machine enters its declared error or cancellation state;
  and
- route identity, last completed safe point, resource state, and error cause
  are recorded without presenting the run as performance evidence.

The design must bound cancellation observation and resource cleanup even for
the longest supported prompt. A design that can only discover failure after
an uninterruptible full-prompt execution does not satisfy the product
boundary.

## 7. Architecture-candidate contract

A Prefill architecture candidate is a complete executable dataflow, not a
kernel list. Before activation it must declare:

1. the originating 40K, 60K, or approximately 130K API symptom and the
   downward budget assigned to Prefill;
2. one route from admitted token IDs through all model layers to the exact
   handoff, including fallback and numerical-mode exclusions;
3. tensor shapes, model order, state transitions, bytes, residency,
   synchronization, and producer/consumer ownership for every dominant
   operator family;
4. the mutually dependent local mechanisms required to realize that route,
   each inside a bounded Roadmap-activated work package;
5. a composition deadline and the target-length API return point;
6. output/state equivalence, resource, capacity, cancellation, and route
   attestation gates; and
7. the cumulative native incumbent and external vLLM starting-line evidence
   used at architecture and release selection.

When the Prefill gap is architectural, candidate design must inspect the
relevant proven vLLM, FlashInfer, Triton, FlashLinearAttention, and Mamba
dataflows before local parameter scanning. Their algorithms, scheduling, and
specialization mechanisms are references; they do not become production
dependencies by inspection.

The architecture candidate returns to the real OpenAI-compatible API and the
40K/60K/approximately-130K witness set as soon as its dependencies close or
its composition budget expires. Component timing and profiler evidence may
explain acceptance or rejection, but cannot select the whole architecture.

## 8. Designed candidate lineage: `AC-PREFILL-LAYERMAJOR-8K-v1`

`AC-PREFILL-LAYERMAJOR-8K-v1` is the first complete design response to the
repeated short-span layer traversal. Its status remains **designed and
non-executable as a complete architecture candidate**: supporting host
contracts and isolated projection surfaces now exist, but the end-to-end
candidate is not implemented, selected, or active in production. The
identifier freezes a lineage for Roadmap-controlled integration; it does not
open a local optimization work package or change the current route.

### 8.1 Execution shape and progress

The candidate changes the scheduling skeleton from repeated public C512
Prefill tiles to one whole-request, layer-major model pass. Each model layer
consumes the complete admitted prompt before the next layer begins. Work
inside a layer is divided into operator panels of at most 8,192 prompt tokens
so kernels and scratch remain bounded.

`C8192` is an internal operator-panel capacity, not a public Prefill tile,
request boundary, partial handoff, or permission to truncate a final panel.
For 40K, 60K, and 130K prompts the design therefore replaces approximately
79, 118, and 254 complete 64-layer walks at C512 with one 64-layer walk whose
layers contain approximately 5, 8, and 16 ordered panels. These counts state
the scheduling hypothesis only; they are not a speedup or latency promise.

The execution plan owns two monotonic progress vectors:

- `kv_progress[layer]` records the largest prompt position whose exact KV is
  visible to later Attention panels in that layer; and
- `gdn_progress[layer]` records the largest position whose exact recurrent
  and convolution state has advanced in that layer.

Attention panels may consume only positions made visible by the applicable KV
event. GDN panels advance in token order and preserve the declared recurrent
state dtype, layout, rounding, and convolution history across every panel
boundary. Panel progress is internal, request-owned state: neither vector
authorizes Decode. After every layer has consumed every admitted token, the
plan publishes exactly one final `PrefillStateCommitted` handoff containing
the complete KV, GDN, position/RoPE, and next-token state. Cancellation and
failure before that event expose no partial handoff.

The current tree implements this topology as a pure-host, immutable plan for
64 layers (48 linear-Attention and 16 full-Attention layers), at most 32
C8192 panels, request-owned progress vectors, and one planned final host-state
commit transition. That scaffold has no launcher, device pointer, stream,
event, allocation, or production selector. Its commit helper deliberately
does not call `RequestState::set_sequence_length()`, and its `executable()`
result therefore remains false. The existing production-route admission and
runner boundary remain public C512.

### 8.2 Indivisible composition boundary

The design is complete only when one attested route composes all dominant
families below in natural model order:

| Family | Candidate requirement |
| --- | --- |
| NVFP4 Gate/Up and Down | Shape-specific large-M ownership for both asymmetric roles; neither may inherit a tactic merely because it won on the other shape |
| FP8 QKV, Z/O, and KV projections | Panel-capable, shape-specific exact routes with authenticated weights, scales, layout, and workspace |
| Attention | Exact causal Attention over the complete prompt, including ordered KV publication and consumption across panels |
| GDN/SSM | Exact recurrent and convolution update across panels and exact final boundary state; no approximate state or altered rounding contract |
| Residual, normalization, embedding, logits, and handoff | Consumer-native layouts or explicit planned conversions, with no undeclared synchronization, allocation, or fallback |

A build that implements only a larger projection, Attention, or GDN kernel is
a local mutation, not this candidate. Missing family coverage, a return to the
public C512 loop, approximate arithmetic, MTP, cuBLASLt dispatch, or an
undeclared fallback invalidates the route rather than producing a partial
candidate result. Buffering and overlap remain dependency-derived plan
choices; the `C8192` name does not imply double or triple buffering by itself.

The current tree also implements a typed host binding contract for exactly 17
roles: NVFP4 Gate/Up and Down; linear-Attention FP8 QKV, Z, and O;
full-Attention FP8 Q, K, V, and O; linear BF16 A and B; exact GDN; exact
causal Attention; residual; normalization; embedding; and final handoff. The
C8192 target descriptors state exact/native/AOT requirements, but every
tactic, weight, sidecar, workspace, launcher, and completion-event identity
remains unbound with `kUnboundDesign` attestation. The legacy inventory is
capped at M512 and cannot satisfy the C8192 contract.

Candidate-only C8192 NVFP4 Gate/Up and Down surfaces and shape-specific FP8
projection surfaces now exist in the kernel tree. As architecture-candidate
assets they are completely unbound: they remain isolated from the binding
contract's physical identities and are not connected to `RequestState`, the
runner, or any production selector. Their presence has not widened production
admission, changed the selected route, or produced a new production
performance result. The remaining operator roles are likewise not composed
into a bound layer-major executor.

### 8.3 Implemented workspace requirements and capacity closure gate

The current tree implements a checked, host-only workspace requirements plan.
It neither reserves device memory nor authenticates model or sidecar
residency. A byte total exists only after the caller explicitly names the
C8192 GDN, legacy GDN, and MLP physical tactics. For C8192 exact C64-native GDN
with in-place convolution, current Release/default legacy exact C16 GDN, and
separate Gate+Up then SiLU, the target-bucket arithmetic is:

| Prompt tokens | Caller-selected conditional profile | Conservative disjoint profile |
| ---: | ---: | ---: |
| 40,000 | 3,975,374,848 bytes | 5,324,963,840 bytes |
| 60,000 | 5,496,014,848 bytes | 7,052,323,840 bytes |
| 130,000 | 10,818,254,848 bytes | 13,098,083,840 bytes |

Here `selected` means only the pair of strategies explicitly supplied to the
host planner: one prompt-wide hidden buffer under an unbound panelwise
in-place contract, plus sequential C8192 family-live-set overlay under
unbound completion and legacy-route-exclusion contracts. It does **not** mean
that the architecture candidate, a tactic, or a production route has been
selected. The conservative profile uses two disjoint prompt-wide hidden
buffers and separates three C8192 operator families plus the legacy C512
workspace. It still relies on each selected tactic's named phase-local layout
and therefore is not an alias-free executable reservation.

The corrected C8192 physical high-water ledger is:

- exact C64-native GDN: 446,365,696 bytes with in-place convolution, or
  456,851,456 bytes with one independent token-parallel C512 convolution
  output; the fixed 75,694,080-byte native workspace is reused serially and is
  never multiplied by physical segment count;
- Full Attention preprocess/core: 402,653,184 bytes for raw Q+gate, processed
  Q, and packed gate; K/V write directly to persistent cache;
- MLP: 855,638,016 bytes for separate Gate+Up then SiLU, or 940,572,928 bytes
  for the fused Gate+Up epilogue; and
- shared Marlin FP32 reduction plus locks: 1,048,832 bytes, placed only through
  an authenticated dead-span/phase layout.

The request total separately owns one stable 10,240-byte final-hidden handoff.
The native GDN layout and every CUDA partition offset now share one constexpr
host ABI, while the Marlin reduction and lock sizes come directly from the
public NVFP4/FP8 kernel ABIs. One 32,768-byte panel token-ID view may occupy the
operator-arena prefix only until embedding gather completes; its reuse is an
explicit event-gated alias rather than additional capacity.

Consequently separate-SiLU MLP still dominates the selected overlay. Choosing
token-parallel C64 convolution changes the conservative 40K/60K/130K rows to
5,335,449,600 / 7,062,809,600 / 13,108,569,600 bytes. Choosing a disjoint
test-only C64-native legacy route adds 75,694,080 bytes; the current Release
C16 legacy route has no such external workspace. C64-native and the large-M
projection admissions remain test-only today, so these are explicit candidate
requirements rather than deployed production facts.

The first `RequestState` allocation strategy is now separately expressible:
the three C8192 families share their sequential high-water, while the complete
legacy C512 workspace is physically disjoint. Its exact selected totals are
4,066,344,960 / 5,588,904,960 / 10,917,864,960 bytes at 40K/60K/130K. It still
requires typed subrange and completion-event binding before execution.

Both profiles for all three buckets fit the planner's declared
17,437,720,576-byte request-arena limit. This is only a request-arena verdict.
The planner intentionally leaves resident-model bytes, derived-sidecar bytes,
and total whole-process required bytes absent, and reports whole-process
capacity as `kIndeterminate`. Allocation reservations, alias/event contracts,
and operator bindings are also unbound, so the workspace plan's
`executable()` result remains false. These values are **not** evidence that
the model plus candidate fit the device or that a production capacity profile
exists; allocator fragmentation, runtime metadata, and measured
whole-process peak memory remain open.

P1 must turn these host requirements into versioned 40K/60K/130K
`RequestMemoryPlan` reservations and measured whole-process peaks. P2 must
bind one authenticated AOT layout/sidecar ownership model, all panel
workspaces, and the exact installed binary. Startup and admission fail closed
when the selected bucket cannot reserve the complete plan before Prefill
begins; the request path may not grow it.

### 8.4 Next integration seam, activation, and adjudication

A default-off whole-request host-control seam now exists. It accepts only a
complete, uncommitted 64-layer progress result, validates the transcript and
domains, invokes a dedicated uncommitted retained-hidden finalizer, calls one
no-throw final commit callback, and publishes the controller's local host
progress only after that callback returns OK. The callback returns status
rather than a caller-authored receipt: it must complete every fallible check
before its single no-fail initial-to-final state publication, and a non-OK
status guarantees no state change. `ReferenceEngine` supplies none of these
callbacks, so the seam cannot select or execute the candidate.

The immediate integration slice is now a distinct layer-major `RequestState`
profile with one full-prompt residual, a typed phase-layout scratch span, a
fixed final-hidden handoff slot, and the legacy C512 views retained disjointly
for the first safe integration. The host planner now names this exact combined
strategy, but its device allocation and phase offsets remain deliberately
unbound. The runner must extract a parameterized
single-layer/single-panel body and implement true outer-layer/inner-panel
traversal. Repeated calls to the existing public C512 64-layer loop do not
satisfy this architecture. A later bound execution/deployment plan must still
authenticate artifacts and bind typed launchers, resources, streams, events,
reservations, and the complete 17-role route before any selector may admit it.

The Roadmap may activate this lineage only after its P1/P2 prerequisites are
closed and after naming bounded local work packages for the mutually dependent
route changes. Implementation first returns through a small exact API sanity
request, then faces the clean-host cold/no-cache 40K, 60K, and approximately
130K real-Agent witness set against the frozen cumulative native incumbent.
The witness must retain exact prompt consumption, output/state equivalence,
route identity, pure Prefill and external TTFT intervals, peak resources,
single final state commit, and zero forbidden fallbacks.

Only that complete API result may select the development architecture. A
component timing, C8192 panel rate, C512 comparison, or profiler counter may
retain or explain a named local mutation but cannot select this candidate or
activate production. Production remains unchanged until a later frozen
`release_candidate` completes the full release protocol.

## 9. Global dataflow questions

Every candidate must answer these questions for the full Prefill route, not
only for a convenient operator or prompt tile:

| Question | Required architectural answer |
| --- | --- |
| Traversal | How are prompt spans and model layers ordered, and what work is repeated as context grows? |
| Residency | Where do weights, derived layouts, activations, KV, recurrent state, and scratch reside over their useful lifetimes? |
| Projection ownership | How do Gate/Up, Down, and FP8 projection shapes receive distinct ownership without forcing one universal tactic? |
| Attention | How are exact causal state, KV publication, and sequence-parallel work represented for the complete prompt? |
| GDN/SSM | How is exact recurrence advanced while parallel work is exposed, and where is the final boundary state rounded and committed? |
| Synchronization | Which dependency requires each event or barrier, and which independent stages can overlap? |
| Buffering | Which producer/consumer pair justifies each additional buffer, and what is its worst-case context-memory cost? |
| Capacity | How do 40K, 60K, and approximately 130K requests fit without request-time allocation growth or silent truncation? |
| Startup specialization | Which offline-selected layouts and tactics are authenticated in the AOT plan, with no request-time discovery? |
| Upward leakage | Which Prefill interval and external TTFT component must move, and how will route attribution prove the connection? |

These questions intentionally stop above tile size, pipeline stage count,
cache instruction, fusion recipe, profiler threshold, or kernel name. Such
choices belong to an explicitly active local optimization work package and
retain authority only inside its declared role, shape, numerical mode, and
composition budget.

## 10. Roadmap activation and historical boundary

This SDD owns no execution sequence. [`ROADMAP.md`](ROADMAP.md) alone selects
the active delivery slice, architecture candidate, and local work packages.
An activation must name a work-package ID, parent candidate, product symptom,
scope, incumbent, real payload, stop condition, composition deadline, and API
return witness before a mechanism document can guide implementation.

The following records are dormant and have no current planning authority:

- [`PREFILL_ARCHITECTURE_RESET_LEGACY.md`](PREFILL_ARCHITECTURE_RESET_LEGACY.md)
  preserves the former mixed architecture plan, historical measurements,
  physical-limit arguments, local budgets, and execution order;
- [`GDN_PREFILL_DATAFLOW.md`](GDN_PREFILL_DATAFLOW.md) preserves GDN mechanism
  designs and experiment lineage;
- [`LARGE_M_PROJECTION_DATAFLOW.md`](LARGE_M_PROJECTION_DATAFLOW.md) preserves
  projection mechanism designs and shape-specific experiment lineage; and
- [`PREFILL_REFERENCE_AUDIT.md`](PREFILL_REFERENCE_AUDIT.md) and
  [`FP8_MARLIN_W8A16_SOURCE_MAP.md`](FP8_MARLIN_W8A16_SOURCE_MAP.md) preserve
  reference analysis and provenance.

Roadmap activation may delegate a bounded section of these records to a local
work package. It does not make their old thresholds, priorities, feasibility
claims, or production wording current. Any reused mechanism must be restated
in the active package against the current product route, incumbent, numerical
contract, evidence protocol, and expiry point.
