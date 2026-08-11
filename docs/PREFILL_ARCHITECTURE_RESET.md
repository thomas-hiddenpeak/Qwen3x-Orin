---
q3x_document:
  id: q3x-prefill-architecture
  class: active
  status: active
  owner: prefill-maintainers
  authority: Prefill subsystem boundary, state contract, and architecture-candidate requirements
  effective: 2026-08-10
  last_reviewed: 2026-08-12
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

The same rule applies to small control workspaces. Locks, counters, barrier
state, and completion words that remain live across operator-family phases
must have one stable physical owner until their last consumer. Allocation,
pointer validity, and an initial clear are insufficient when a family overlay
can write the same bytes in between. The bound plan must either exclude every
intervening alias writer or explicitly re-establish the required state before
the next consumer; a receipt must cover the complete lifetime. The normative
proof obligation is owned by the
[`PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md`](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md).

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

```text
real-number function and legal rewrites
  -> finite-precision operands, reduction tree, rounding and state boundaries
  -> production observables, buffer/control-state lifetimes and alias owners
  -> SM87 engineering map: residency, ownership, pipeline and synchronization
```

This is the mandatory design order. The mathematical ledger owns the first
three proof layers; this subsystem SDD owns their executable state, lifetime,
and handoff mapping. CUDA layout or profiler evidence may refine the fourth
step only. If it changes an earlier premise, the candidate returns to that
proof layer under a new identity rather than treating the physical mechanism
as its own justification.

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

## 8. Stable implementation constraints

This SDD does not name the active candidate, repeat its measurements, or
retain its implementation history. [`ROADMAP.md`](ROADMAP.md) selects the
current architecture and work packages; [`CURRENT_STATUS.md`](CURRENT_STATUS.md)
owns the replaceable implementation snapshot; immutable evidence owns every
measured outcome. The following constraints remain stable across candidates.

### 8.1 Packed-operand and producer/consumer contract

Large-M quantized projections in a production Prefill candidate must preserve
the packed operand as a first-class execution asset:

- authenticated real-checkpoint tensors are transformed only at build or
  load time into role- and shape-specific consumer layouts;
- NVFP4 or FP8 B operands and their scales remain packed through global and
  shared-memory movement, with decode adjacent to the exact BF16 MMA consumer
  and FP32 accumulation;
- A, packed B, scales, and epilogue state have one declared producer/consumer
  pipeline rather than independent mechanisms whose composition is assumed;
- merged Gate+Up, K-heavy Down, and FP8 QKV/Z/O families may select different
  layouts and schedules while preserving each tensor's original scale,
  rounding, residual, and activation semantics;
- a request may not materialize a full BF16 copy of B, JIT compile or repack a
  weight, discover a tactic, grow an arena, or enter a fallback route;
- the physical launch and synchronization ledger must agree with the sealed
  DeploymentPlan and runtime witness; and
- rejected implementation mechanisms may remain as default-off correctness
  evidence but cannot remain reachable from the runner.

This contract selects no tile, stage count, grid size, performance budget, or
delivery order.

### 8.2 Reference translation and compute-for-movement

Every active candidate follows the cross-architecture translation contract in
[`SDD.md`](SDD.md#92-cross-architecture-reference-translation). An upstream
SM guard does not exclude a vLLM, FlashInfer, Triton, Humming, FLA, Mamba,
CUTLASS, or related path from source study. The work package separates its
invariant mathematics, dataflow, work partition, and planning from unavailable
ISA/resources, then states the SM87 realization explicitly.

Any candidate that adds arithmetic or decode/feed instructions to remove
movement records the added work, eliminated DRAM/L2/shared/materialization or
synchronization, resource and critical-path transfer, exact oracle, and target
API effect in its equivalence ledger. Lower traffic without a positive complete
route is not sufficient for selection.

### 8.3 Mathematical-equivalence owner

The normative proof classes, finite-precision obligations, quantitative P40
arithmetic, liveness boundary, and plan/receipt/oracle requirements are owned
only by
[`PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md`](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md).
This SDD links that owner instead of copying equations, experiment-specific
ceilings, or a mutable candidate derivation.

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
