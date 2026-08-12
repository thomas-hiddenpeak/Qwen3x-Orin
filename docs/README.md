---
q3x_document:
  id: q3x-documentation-index
  class: procedure
  status: active
  owner: project-maintainers
  authority: documentation navigation
  effective: 2026-08-09
  last_reviewed: 2026-08-12
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

## Required orientation and task routes

Codex enters through [`../AGENTS.md`](../AGENTS.md), which sends it here. Do
not recurse back to that file when it has already been applied. A human or
tool entering through this index directly must read it once for repository
hygiene and execution constraints.

Before planning, implementing, reviewing, or evaluating non-trivial work, and
again after context compaction or contributor handoff, read this core
orientation in order:

1. [`ENGINEERING_CONSTITUTION.md`](ENGINEERING_CONSTITUTION.md) — mission,
   locked constraints, and engineering philosophy;
2. [`SDD.md`](SDD.md) — API-first, external-to-internal runner design;
3. [`CURRENT_STATUS.md`](CURRENT_STATUS.md) — what the default release can do
   now and the currently known gaps; and
4. [`ROADMAP.md`](ROADMAP.md) — the active delivery slice and exit criteria.

Then read the complete route required by the work:

When more than one route applies, merge them in the bullet order below and
skip documents already read; later bullets do not replace earlier routes.

- performance implementation, retention, or promotion:
  [`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md);
- Prefill architecture or implementation:
  [`PREFILL_ARCHITECTURE_RESET.md`](PREFILL_ARCHITECTURE_RESET.md), then
  [`PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md`](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md),
  then the active local work package linked by [`ROADMAP.md`](ROADMAP.md);
- external API evaluation or metric reconciliation:
  [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md);
- Markdown addition, movement, reclassification, or controlling-document
  change: [`DOCUMENT_GOVERNANCE.md`](DOCUMENT_GOVERNANCE.md), then
  [`DOCUMENT_REGISTRY.md`](DOCUMENT_REGISTRY.md); and
- subsystem implementation: the detailed subsystem SDD and every active
  interface, numerical, state, or ownership contract named by that SDD.

The index owns this order. `AGENTS.md`, the root README, SDDs, and Roadmap link
here rather than maintaining parallel reading lists.

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
| Prefill proof order, finite-precision equivalence, P40 arithmetic, and production-observable liveness | [`PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md`](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md) |
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
4. run `python3 tools/docs/validate_document_control.py` from the repository
   root; it checks registry coverage, required headers, identities, and local
   links;
5. run `python3 -B -m unittest tests.test_document_control` when changing the
   validator itself; and
6. preserve historical/evidence contents and third-party source boundaries.

Generated builds, profiles, raw evaluation outputs, downloaded tools, and
temporary source copies belong under the ignored `/.q3x-work/` tree and are
not part of the tracked documentation registry.
