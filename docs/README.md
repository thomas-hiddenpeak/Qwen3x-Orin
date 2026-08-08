---
q3x_document:
  id: q3x-documentation-index
  class: procedure
  status: active
  owner: project-maintainers
  authority: documentation navigation
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: documentation entry points and required reading order
  review_trigger: any controlling document addition, removal, rename, or reclassification
---

# Qwen3x-Orin documentation

This is the canonical documentation entry point. It separates current truth,
system design, contracts, procedures, evidence, history, and external source
material so that an old benchmark or local mechanism cannot silently become a
current product requirement.

## Required reading order

Before planning, implementing, or evaluating performance work, read:

1. [`../AGENTS.md`](../AGENTS.md) — repository-local work package;
2. [`ENGINEERING_CONSTITUTION.md`](ENGINEERING_CONSTITUTION.md) — mission,
   locked constraints, and engineering philosophy;
3. [`SDD.md`](SDD.md) — API-first, external-to-internal runner design;
4. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — what the default release can do
   now and the currently known gaps;
5. [`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md) —
   evidence, retention, and production-promotion rules;
6. [`PREFILL_ARCHITECTURE_RESET.md`](PREFILL_ARCHITECTURE_RESET.md) — active
   Prefill subsystem boundary and handoff contract;
7. [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md) — external-evaluation
   contract and procedure;
8. [`DOCUMENT_GOVERNANCE.md`](DOCUMENT_GOVERNANCE.md) — document authority and
   lifecycle;
9. [`ROADMAP.md`](ROADMAP.md) — active delivery order and exit criteria;
10. the detailed subsystem SDD or contract for the work being changed.

Items 2--7 preserve the mandatory order declared by `AGENTS.md`; the remaining
items complete the end-state-first project orientation before implementation.

The controlling authority and conflict order are defined in
[`DOCUMENT_GOVERNANCE.md`](DOCUMENT_GOVERNANCE.md). The exhaustive inventory
is [`DOCUMENT_REGISTRY.md`](DOCUMENT_REGISTRY.md).

## Single-source map

| Subject | Current owner |
| --- | --- |
| Mission, target hardware/model, owner-set performance and accuracy constraints | [`ENGINEERING_CONSTITUTION.md`](ENGINEERING_CONSTITUTION.md) |
| End-to-end runner architecture and API-to-kernel boundaries | [`SDD.md`](SDD.md) |
| Current default route, delivered capability, qualified metrics, and open gaps | [`CURRENT_STATUS.md`](CURRENT_STATUS.md) |
| Active dependency order and milestone exit criteria | [`ROADMAP.md`](ROADMAP.md) |
| Evidence tiers, local retention, architecture-candidate qualification, and release promotion | [`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md) |
| Documentation classes, lifecycle, supersession, and completeness | [`DOCUMENT_GOVERNANCE.md`](DOCUMENT_GOVERNANCE.md) |
| Accepted architecture decisions and their supersession chain | [`decisions/README.md`](decisions/README.md) |
| OpenAI-compatible EvalScope procedure and protocol limitations | [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md) |
| All tracked Markdown classifications | [`DOCUMENT_REGISTRY.md`](DOCUMENT_REGISTRY.md) |

## Active SDD set

The system SDD is [`SDD.md`](SDD.md). The following first-party documents
refine named subsystem boundaries and are subordinate to it:

- [`DESIGN.md`](DESIGN.md) — compatibility entry point and subsystem-design
  map;
- [`PREFILL_ARCHITECTURE_RESET.md`](PREFILL_ARCHITECTURE_RESET.md) — active
  Prefill subsystem SDD for inputs, outputs, state, ownership,
  synchronization, failure, and Decode handoff.

The detailed
[`GDN_PREFILL_DATAFLOW.md`](GDN_PREFILL_DATAFLOW.md) and
[`LARGE_M_PROJECTION_DATAFLOW.md`](LARGE_M_PROJECTION_DATAFLOW.md) files are
dormant local-design libraries, not active global plans. A named SDD/Roadmap
architecture candidate must activate a bounded section before it guides
implementation. The former mixed Prefill plan and mechanism history is
preserved in
[`PREFILL_ARCHITECTURE_RESET_LEGACY.md`](PREFILL_ARCHITECTURE_RESET_LEGACY.md)
with no current planning authority. [`VLLM_HUMMING_STARTUP_AUDIT.md`](VLLM_HUMMING_STARTUP_AUDIT.md)
is a frozen external architecture snapshot, not an SDD or production
dependency.

Mechanism-level prescriptions in the GDN and projection documents apply only
to their declared role, shape, numerical mode, hardware, and evidence scope.
They do not override the system SDD or select the whole runner. A local change
becomes system architecture only when the parent SDD records its boundary and
the real API path demonstrates upward value transfer.

## Contracts and procedures

Runtime, numerical, state, loader, tokenizer, and reference-component
contracts are indexed individually in the registry. They define behavior at a
specific boundary; they do not own project priorities or performance targets.

Procedures describe reproducible actions. In particular:

- [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md) owns the external
  evaluation procedure;
- [`../benchmarks/evalscope/README.md`](../benchmarks/evalscope/README.md) owns
  the pinned workload-manifest procedure;
- [`REFERENCE_BENCHMARK.md`](REFERENCE_BENCHMARK.md) owns the internal
  repeatability harness procedure.

## Evidence, history, and external material

Evidence under `docs/analysis/`, `docs/metadata/`, and the named evidence
records is immutable for its exact workload and protocol. It can explain a
decision but cannot redefine current intent, broaden a local conclusion, or
lower a product target.

Accepted decisions live under [`decisions/`](decisions/README.md). An ADR
records why a design boundary was selected; the SDD records the resulting
current design. Superseding an ADR requires an explicit successor rather than
silently rewriting the accepted decision.

[`ROADMAP_LEGACY.md`](ROADMAP_LEGACY.md),
[`PREFILL_ARCHITECTURE_RESET_LEGACY.md`](PREFILL_ARCHITECTURE_RESET_LEGACY.md),
and [`PREFILL_REFERENCE_AUDIT.md`](PREFILL_REFERENCE_AUDIT.md) are historical.
Their current decisions must be restated in an active owner document before
they guide implementation. Vendored documentation and first-party audits of
external implementations retain source/provenance value but have no native
production-selection authority.

## Maintaining the documentation set

When a Markdown file is added, moved, deleted, or reclassified:

1. use the standard control header from
   [`DOCUMENT_GOVERNANCE.md`](DOCUMENT_GOVERNANCE.md) for a new first-party
   document;
2. update [`DOCUMENT_REGISTRY.md`](DOCUMENT_REGISTRY.md) in the same commit;
3. update this index if a controlling document or SDD link changes;
4. verify that every `git ls-files '*.md'` path appears exactly once in the
   registry and that no registry path is absent from the tree;
5. preserve historical/evidence contents and third-party source boundaries.

Generated builds, profiles, raw evaluation outputs, downloaded tools, and
temporary source copies belong under the ignored `/.q3x-work/` tree and are
not part of the tracked documentation registry.
