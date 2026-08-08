---
q3x_document:
  id: q3x-design-entry
  class: active
  status: active
  owner: project-maintainers
  authority: legacy DESIGN path compatibility and subsystem-design navigation
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: DESIGN.md compatibility entry and subsystem-design index
  review_trigger: any system SDD or active subsystem-design addition, removal, or replacement
---

# Qwen3x-Orin design entry and subsystem index

Status: compatibility entry point and subsystem-design index. This file does not
declare current implementation or performance state.

Older repository links point to `DESIGN.md`, so this path remains stable. The
system-level design has moved to [`SDD.md`](SDD.md), which begins at the
externally delivered OpenAI-compatible runner and traces constraints inward to
the kernel. Point-in-time implementation, qualification, production, and
performance facts live only in [`CURRENT_STATUS.md`](CURRENT_STATUS.md).

## 1. Document authority and reading order

Repository operating instructions require these documents to be read in this
exact order before architecture or performance work:

1. [`ENGINEERING_CONSTITUTION.md`](ENGINEERING_CONSTITUTION.md) — mission,
   owner authority, locked targets and non-negotiable product constraints.
2. [`SDD.md`](SDD.md) — external product boundary and system decomposition.
3. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — current implementation, route,
   evidence authority, and blockers.
4. [`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md) —
   evidence authority and qualification rules.
5. [`PREFILL_ARCHITECTURE_RESET.md`](PREFILL_ARCHITECTURE_RESET.md) — active
   Prefill subsystem boundary, state, ownership, failure, and Decode-handoff
   contract, subject to the documents above.
6. [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md) — external evaluation
   protocol and evidence limitations.

For documentation work, also read
[`DOCUMENT_GOVERNANCE.md`](DOCUMENT_GOVERNANCE.md). Then read
[`ROADMAP.md`](ROADMAP.md) and the active subsystem contracts for the work.
[`README.md`](README.md) is the canonical documentation entry point and
[`DOCUMENT_REGISTRY.md`](DOCUMENT_REGISTRY.md) is the exhaustive Markdown
inventory.

[`ROADMAP.md`](ROADMAP.md) orders unfinished work. It cannot redefine a locked
target or promote a capability. Subsystem documents refine one boundary; they
cannot override the API, numerical, state, evidence, workspace, or release
contracts above.

When an older subsystem or evidence document uses words such as “current”,
“production”, “terminal”, or “supported”, interpret them at the recorded
revision and protocol. `CURRENT_STATUS.md` is the current-state authority.

## 2. Stable system map

```text
external client
  -> OpenAI-compatible API and streaming contract
  -> admission, capacity and request lifecycle
  -> Prefill plan -> exact state commit -> Decode plan
  -> model graph and numerical contracts
  -> shape/family-specific operator dataflows
  -> authenticated layouts, kernels, streams and arenas
  -> pinned checkpoint and SM87 device
```

The complete design and evolution rules are in [`SDD.md`](SDD.md). The index
below directs detail to one owning document instead of repeating dynamic facts
in several places.

## 3. Governance, delivery and evaluation documents

| Document | Owning responsibility | It must not be used as |
| --- | --- | --- |
| [`ENGINEERING_CONSTITUTION.md`](ENGINEERING_CONSTITUTION.md) | Mission, trust, product-first hierarchy, targets, hard feature/numerical boundaries, continuity and host hygiene | A benchmark log or subsystem implementation guide |
| [`DOCUMENT_GOVERNANCE.md`](DOCUMENT_GOVERNANCE.md) | Document classes, single-source ownership, lifecycle, supersession and registry completeness | A project target or subsystem implementation contract |
| [`README.md`](README.md) | Canonical documentation navigation and required entry points | A duplicate current-status or evidence ledger |
| [`DOCUMENT_REGISTRY.md`](DOCUMENT_REGISTRY.md) | Exhaustive classification of every tracked Markdown path | Narrative system design or benchmark interpretation |
| [`SDD.md`](SDD.md) | External product contract, DeploymentPlan, state machines, fitness vector, leakage trace, subsystem and release boundaries | A point-in-time performance report |
| [`CURRENT_STATUS.md`](CURRENT_STATUS.md) | Current default route, capability state, evidence authority, and open gaps | An immutable history ledger or delivery-order owner |
| [`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md) | Tiers, payload identity, direction/retention/promotion and profiler evidence rules | The project mission or a component design |
| [`ROADMAP.md`](ROADMAP.md) | Ordered unfinished outcomes, dependencies and exit criteria | Current capability truth or historical evidence authority |
| [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md) | External API workload, metric semantics, validation and release evaluation protocol | Proof that the evaluation-only gateway is a production server |
| [`VLLM_HUMMING_STARTUP_AUDIT.md`](VLLM_HUMMING_STARTUP_AUDIT.md) | Frozen snapshot of proven startup/tactic-selection mechanisms for AOT-specialization study | A runtime dependency decision, active plan, or performance result |

The repository-root `AGENTS.md` is the contributor entry contract and gives
the mandatory reading order, workspace boundary and clean-host performance
preflight.

## 4. Product, model and numerical contracts

| Document | Scope | Authority |
| --- | --- | --- |
| [`MODEL_SUPPORT.md`](MODEL_SUPPORT.md) | Model-family catalog, checkpoint descriptors and support vocabulary | Catalog/compatibility contract; not an end-to-end production claim |
| [`QWEN36_27B_RUNTIME_CONTRACT.md`](QWEN36_27B_RUNTIME_CONTRACT.md) | Pinned Qwen3.6 27B text-only execution semantics and layer schedule | Model/numerical ABI for the named revision |
| [`TOKENIZER.md`](TOKENIZER.md) | Pinned tokenizer identity, normalization, BPE and chat-template behavior | Text/token boundary contract |
| [`RESIDENT_WEIGHT_LOADER.md`](RESIDENT_WEIGHT_LOADER.md) | Checkpoint authentication, one-pass resident load, memory ownership and failure behavior | Loader subsystem contract |
| [`MODEL_WEIGHT_BINDING.md`](MODEL_WEIGHT_BINDING.md) | Typed immutable binding from authenticated tensors to the execution graph | Weight/view lifetime and model-graph boundary |
| [`REQUEST_STATE.md`](REQUEST_STATE.md) | Per-request arena, KV/recurrent state, scratch and position ownership | Request-memory and state ABI |

Quantized layouts remain checkpoint-authenticated and lossless unless an
explicit research contract says otherwise. A research numerical path cannot
gain production authority from a subsystem document.

## 5. Runtime and correctness contracts

| Document | Scope | Relationship to the SDD |
| --- | --- | --- |
| [`REFERENCE_ENGINE.md`](REFERENCE_ENGINE.md) | Correctness-first resident engine lifecycle and generation surface | Implements a bounded engine stage; not the final serving contract |
| [`REFERENCE_RUNNER.md`](REFERENCE_RUNNER.md) | 64-layer batch-one execution and prompt-tile semantics | Refines model execution and state transition boundaries |
| [`REFERENCE_BENCHMARK.md`](REFERENCE_BENCHMARK.md) | Repeatable engine reuse and internal benchmark harness | Diagnostic/correctness surface; never substitutes for the external API |
| [`REFERENCE_ORACLE.md`](REFERENCE_ORACLE.md) | Pinned oracle loading and comparison | Accuracy-evidence input, not an optimized route |
| [`REFERENCE_GEMV.md`](REFERENCE_GEMV.md) | Canonical BF16/NVFP4/FP8 batch-one projection reference | Local numerical oracle and fallback contract |
| [`DECODE_REFERENCE_OPS.md`](DECODE_REFERENCE_OPS.md) | Decode normalization, positional, cache and surrounding reference operations | Refines Decode numerical boundaries |
| [`GDN_DECODE_REFERENCE.md`](GDN_DECODE_REFERENCE.md) | Single-token/bounded-tile Gated DeltaNet semantics and state update | Refines Decode GDN state ownership |

Reference means correctness ownership, not performance authority. A component
can be production-eligible only through the complete release route and
attestation defined in the SDD.

## 6. Prefill and optimized-dataflow documents

| Document | Scope | Classification |
| --- | --- | --- |
| [`PREFILL_ARCHITECTURE_RESET.md`](PREFILL_ARCHITECTURE_RESET.md) | Prefill inputs, outputs, state, ownership, synchronization, failure, architecture candidates and Decode handoff | Active subsystem SDD; `ROADMAP.md` alone owns delivery order |
| [`PREFILL_ARCHITECTURE_RESET_LEGACY.md`](PREFILL_ARCHITECTURE_RESET_LEGACY.md) | Former mixed Prefill plan, measurements, physical-limit arguments, mechanism rules and execution order | Dormant historical design/evidence with no current planning authority |
| [`GDN_PREFILL_DATAFLOW.md`](GDN_PREFILL_DATAFLOW.md) | Exact and research GDN Prefill dataflows, state semantics and experiment lineage | Dormant historical mechanism design/evidence; numerical modes remain separate |
| [`LARGE_M_PROJECTION_DATAFLOW.md`](LARGE_M_PROJECTION_DATAFLOW.md) | Gate/Up, Down and FP8 large-M operand residency, shape ownership and candidate lineage | Dormant historical mechanism design/evidence; role/shape rules remain local |
| [`FP8_MARLIN_W8A16_SOURCE_MAP.md`](FP8_MARLIN_W8A16_SOURCE_MAP.md) | Frozen upstream-to-local mechanism/provenance map | Reference/admission source map, not production dispatch authority |
| [`PREFILL_REFERENCE_AUDIT.md`](PREFILL_REFERENCE_AUDIT.md) | FlashInfer/qwen35-thor and related Prefill architecture review | Historical/reference input, not a current target or dependency decision |

The legacy Prefill, GDN, and Large-M documents are dormant historical
mechanism/evidence records. Their mechanism-specific rules apply only after
`ROADMAP.md` activates a bounded section inside a named local work package.
They do not define the project-wide starting point, product fitness, roadmap
priority, or release promotion. The combined architecture returns to the real
API for selection.

## 7. Evidence and historical records

| Document or tree | Scope | Authority boundary |
| --- | --- | --- |
| [`PHASE0_EVIDENCE.md`](PHASE0_EVIDENCE.md) | Bootstrap host/toolchain/checkpoint evidence | Historical to its named revision |
| [`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md) | Append-only component, Prefix and Decode performance lineage | Each result applies only to its recorded tuple; it is not current status |
| [`metadata/README.md`](metadata/README.md) | Machine-readable checkpoint, test and evaluation evidence registry/index | Evidence facts; promotion authority is per record and policy |
| `docs/metadata/*.json` | Pinned identities, protocols, results and decisions | Machine-readable record for the exact hash/protocol |
| `docs/analysis/*` | Experiment reports, raw-artifact hashes, attribution and rejection/retention rationale | Immutable experiment evidence, not a global status surface |

Profiler output, synthetic correctness, component timings and external runs
retain their own evidence tiers. Moving a record into Git does not increase
its authority.

## 8. Documentation placement rules

Use this routing when a change needs documentation:

- mission, hard constraint or owner-authority change → constitution amendment;
- stable cross-subsystem product/architecture boundary → `SDD.md`;
- present implementation, default, metric or blocker → `CURRENT_STATUS.md`;
- evidence/admission/promotion method → performance or evaluation policy;
- unresolved ordered deliverable → `ROADMAP.md`;
- subsystem ABI, numerical behavior, layout or ownership → one owning
  subsystem contract;
- architecture choice and rejected alternative → ADR or named analysis record;
- raw identity/result → machine-readable metadata plus an evidence report;
- historical narrative → append-only evidence, clearly dated and scoped.

Do not copy a mutable number into the constitution, SDD, README, roadmap and
several subsystem documents. Link to `CURRENT_STATUS.md` or the evidence
record instead. Do not use “current” in an append-only history heading unless
it includes a revision/date and explicitly yields to `CURRENT_STATUS.md`.

## 9. Compatibility note

The previous `DESIGN.md` mixed intended architecture, bootstrap status,
current kernel implementation detail, and measurement method. Those
responsibilities are now separated:

- system intent and final delivery shape → [`SDD.md`](SDD.md);
- live state → [`CURRENT_STATUS.md`](CURRENT_STATUS.md);
- exact subsystem contracts → the index above; and
- measured history → evidence records.

Existing links to this file remain valid, but new cross-references should link
to the owning document directly.
