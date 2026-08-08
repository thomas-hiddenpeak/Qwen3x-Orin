---
q3x_document:
  id: q3x-adr-0001-end-state-first-leakage
  class: active
  status: active
  owner: project-owner
  authority: accepted methodology and system-boundary decision
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: rationale for end-state-first leakage and controlled evolution
  review_trigger: successor ADR or constitutional methodology amendment
---

# ADR-0001: End-state-first leakage and controlled evolution

Decision state: accepted.

## Context

Qwen3x-Orin is a runner, but its historical design and roadmap grew from the
checkpoint loader and individual kernels outward. Product evaluation was
eventually added at the end. That direction allowed component metrics,
historical P513 cells, and local mechanism rules to become de facto project
priorities even when the intended delivery was an externally callable API.

The first correction made real API behavior the highest measurement surface.
Applied too literally, however, it treated every local implementation change
as though it were a complete product candidate: a local change that could not
independently cross whole-API noise was rejected. That prevents construction
of coupled dataflows whose required mechanisms are non-orthogonal or whose
local improvement is temporarily hidden by another critical boundary.

The project owner identified a more complete engineering philosophy. For an
AI-executed specialized-runner project, the final delivery environment can be
fixed first and the implementation can evolve under explicit constraints.
Local optimization then gains meaning because it is selected by a global
need and has a defined path back to product value.

## Decision

1. The externally callable OpenAI-compatible runner is the logical system
   boundary and design starting point. API-first does not expand the project
   into a general serving framework.
2. Product constraints propagate downward through API observables, phase
   budgets, execution-plan nodes, subsystem contracts, and named local work
   packages.
3. Implementation value propagates upward from local mechanisms through a
   composed architecture candidate to the real API result.
4. `local_mutation`, `architecture_candidate`, and `release_candidate` are
   distinct engineering units with distinct evidence authority.
5. Local mechanism specifications and profiler/microbenchmark rules apply only
   inside an explicitly active local optimization work package. They cannot
   set global priority, product targets, or production eligibility.
6. A locally positive exact mutation may be retained as an architecture
   prerequisite below API noise only when it has an attested real route,
   real-payload evidence, a named dependency, and a bounded composition point.
7. Complete architecture candidates are selected on target-representative real
   API workloads. Local gains are not added arithmetically to excuse a
   negative composition.
8. Release candidates satisfy hard accuracy, API, capacity, route, non-MTP,
   cuBLASLt-exclusion, resource, and evidence constraints before performance
   can promote them.
9. Candidate lineage, rejected mechanisms, supersession, evidence, and current
   state are retained as engineering memory across agents and context
   compaction.

## Consequences

- Kernel, fusion, layout, buffering, or profiler work remains central, but it
  must be traceable to a product constraint and an architecture candidate.
- A single local experiment need not independently improve EvalScope, while a
  completed architecture composition must return to the API immediately.
- Architecture work must explicitly remove boundaries that block the upward
  propagation of real local improvements.
- Short requests and P513 remain useful for protocol sanity and attribution;
  they cannot select the long-context Prefill architecture without a calibrated
  relationship to target-length witnesses.
- Documentation needs separate owners for constitution, SDD, current status,
  roadmap, policy, subsystem contracts, decisions, and immutable evidence.
- Controlled evolution has budgets and stop conditions. It is not permission
  for unlimited micro-optimization accumulation.

## Implementation mapping

- Normative principles: `docs/ENGINEERING_CONSTITUTION.md`
- System traceability and state machines: `docs/SDD.md`
- Candidate lifecycle and evidence: `docs/REAL_MODEL_PERFORMANCE_POLICY.md`
- External workload protocol: `docs/EVALSCOPE_EVALUATION.md`
- Current product truth: `docs/CURRENT_STATUS.md`
- Active dependency order: `docs/ROADMAP.md`
- Document ownership: `docs/DOCUMENT_GOVERNANCE.md`
