---
q3x_document:
  id: q3x-document-registry
  class: procedure
  status: active
  owner: project-maintainers
  authority: exhaustive tracked-Markdown inventory and classification
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: tracked Markdown paths, primary classes, roles, and lifecycle states
  review_trigger: any tracked Markdown addition, deletion, move, or reclassification
---

# Document registry

This registry assigns exactly one primary class to every tracked Markdown
document. Classes and authority rules are defined by
[`DOCUMENT_GOVERNANCE.md`](DOCUMENT_GOVERNANCE.md).

## Audit snapshot

- Audit date: 2026-08-09.
- Expected integrated-tree coverage: **66 Markdown paths**.
- Classified: **66**.
- Unclassified: **0**.
- Duplicate registrations: **0**.
- Inventory basis: every literal path expected from `git ls-files '*.md'`
  after the SDD/document-governance refactor is committed.
- Transitional rule: existing first-party documents without a standard
  control header inherit their class, role, lifecycle, and authority from this
  registry until materially rewritten. Historical evidence and third-party
  documents are not rewritten merely to add headers.

`Lifecycle` describes the document, not the implementation or experiment it
mentions. A dormant local design becomes active only through the named SDD and
Roadmap work package; it never activates itself. `frozen` evidence keeps
authority only for its exact recorded protocol.

## Local work package (1)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `AGENTS.md` | `repository_work_package` | active | Repository execution and hygiene instructions; cannot amend mission, numerical contract, or business targets. |

## Normative (3)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `docs/DOCUMENT_GOVERNANCE.md` | `documentation_policy` | active | Document classes, control headers, SSOT, supersession, SDD composition, and completeness. |
| `docs/ENGINEERING_CONSTITUTION.md` | `constitution` | active | Mission, owner-set constraints, end-to-start/leakage philosophy, product targets, and project-wide boundaries. |
| `docs/REAL_MODEL_PERFORMANCE_POLICY.md` | `evidence_policy` | active | Real-payload evidence, local retention, architecture-candidate qualification, and release promotion. |

## Active first-party documents (7)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `README.md` | `entry_point` | active | Concise product overview and navigation only; links to controlling docs for current claims. |
| `docs/CURRENT_STATUS.md` | `current_status` | active | SSOT for current default route, delivered capability, qualified metrics, and known gaps; replaceable, not an evidence ledger. |
| `docs/DESIGN.md` | `subsystem_design_index` | active | Compatibility entry point and map of detailed subsystem contracts; subordinate to `docs/SDD.md`. |
| `docs/PREFILL_ARCHITECTURE_RESET.md` | `subsystem_sdd` | active | Prefill input/output, state, ownership, synchronization, failure, handoff, and architecture-candidate contract; owns no delivery order or mechanism rule. |
| `docs/ROADMAP.md` | `active_plan` | active | SSOT for current dependency order, milestones, and exit criteria; not an experiment ledger. |
| `docs/SDD.md` | `system_sdd` | active | SSOT for the externally callable runner design from API and target workloads inward to kernels/deployment. |
| `docs/decisions/0001-end-state-first-leakage.md` | `accepted_adr` | active | Accepted decision recording end-state-first/leakage as the system design and evolution model; subordinate only to normative policy. |

## Contracts (12)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `docs/DECODE_REFERENCE_OPS.md` | `component_contract` | active | Decode common-op reference numerical and dimension contract. |
| `docs/GDN_DECODE_REFERENCE.md` | `component_contract` | active | Single-token/bounded-tile GDN semantics and reference ownership. |
| `docs/MODEL_SUPPORT.md` | `model_catalog_contract` | active | Pinned supported model/architecture facts and status vocabulary. |
| `docs/MODEL_WEIGHT_BINDING.md` | `weight_binding_contract` | active | Resident tensor-to-runtime typed binding and lifetime boundary. |
| `docs/QWEN36_27B_RUNTIME_CONTRACT.md` | `model_runtime_contract` | active | Pinned Qwen3.5/Qwen3.6 27B text runtime semantics and tensor/state contract. |
| `docs/REFERENCE_ENGINE.md` | `component_contract` | active | Correctness-first engine ownership, generation, timing, trace, and verification boundary. |
| `docs/REFERENCE_GEMV.md` | `component_contract` | active | Batch-one GEMV numerical and launch contract. |
| `docs/REFERENCE_ORACLE.md` | `oracle_contract` | active | BF16 oracle identity, schema, trust boundary, and comparison diagnostics. |
| `docs/REFERENCE_RUNNER.md` | `component_contract` | active | Batch-one reference runner sequence, state commit/reset, and fixture boundary. |
| `docs/REQUEST_STATE.md` | `state_contract` | active | Per-request persistent state, workspace, RoPE, memory plan, and lifecycle ownership. |
| `docs/RESIDENT_WEIGHT_LOADER.md` | `loader_contract` | active | Authenticated resident-weight I/O, identity, memory budget, and loader boundary. |
| `docs/TOKENIZER.md` | `tokenizer_contract` | active | Pinned tokenizer/chat formatting resource and error contract. |

## Procedures (6)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `benchmarks/evalscope/README.md` | `evaluation_fixture_procedure` | active | Pinned EvalScope workload-manifest generation and corpus handling. |
| `docs/DOCUMENT_REGISTRY.md` | `document_inventory` | active | Exhaustive Markdown path/class/role/lifecycle registry. |
| `docs/EVALSCOPE_EVALUATION.md` | `external_evaluation_procedure` | active | OpenAI-compatible EvalScope protocol, limitations, and reproduction commands; subordinate to evidence policy. |
| `docs/README.md` | `documentation_index` | active | Required reading order, SSOT map, active SDD links, and documentation navigation. |
| `docs/REFERENCE_BENCHMARK.md` | `benchmark_procedure` | active | Internal reference repeatability harness and CLI procedure; not product-performance authority. |
| `docs/decisions/README.md` | `decision_index` | active | ADR naming, status, supersession, and navigation procedure. |

## Historical (5)

| Path | Role | Lifecycle | Successor / authority boundary |
| --- | --- | --- | --- |
| `docs/GDN_PREFILL_DATAFLOW.md` | `dormant_local_design` | historical | GDN mechanism/design lineage. It becomes a bounded local work package only when activated by `docs/SDD.md`, `docs/ROADMAP.md`, and the active Prefill architecture candidate. |
| `docs/LARGE_M_PROJECTION_DATAFLOW.md` | `dormant_local_design` | historical | Large-M mechanism/design lineage. Gate/Up, Down, and FP8 rules remain dormant and role/shape scoped until explicitly activated. |
| `docs/PREFILL_ARCHITECTURE_RESET_LEGACY.md` | `historical_prefill_design` | historical | Preserved former mixed Prefill plan, measurements, feasibility arguments, local budgets, and execution order; superseded by `docs/PREFILL_ARCHITECTURE_RESET.md` and has no current planning authority. |
| `docs/PREFILL_REFERENCE_AUDIT.md` | `historical_architecture_audit` | historical | Superseded as current Prefill design authority by `docs/PREFILL_ARCHITECTURE_RESET.md`, which refines `docs/SDD.md`; retains pinned source-analysis provenance. |
| `docs/ROADMAP_LEGACY.md` | `historical_roadmap_ledger` | historical | Superseded by concise active `docs/ROADMAP.md`; retained for linked chronology, with Git history and evidence owning exact observations. |

## Evidence (28)

| Path | Role | Lifecycle | Evidence authority |
| --- | --- | --- | --- |
| `docs/PERFORMANCE_BASELINE.md` | `historical_performance_ledger` | frozen | Append-only accumulated component/runner observations for their exact commits/protocols; not current status or target SSOT. |
| `docs/PHASE0_EVIDENCE.md` | `milestone_evidence` | frozen | Phase-0 environment, checkpoint, oracle, and phase-boundary record. |
| `docs/analysis/decode-gate-up-coupled-feed-vllm-parity-2026-07-30/README.md` | `experiment_evidence` | frozen | Decode coupled-feed parity observation for its pinned real API protocol. |
| `docs/analysis/decode-gqa-splitkv-sm87-2026-07-30/README.md` | `experiment_evidence` | frozen | Decode split-KV direction/admission evidence for the recorded build and workload. |
| `docs/analysis/evalscope-prefill-cumulative-19e10f6-2026-07-30/README.md` | `experiment_evidence` | frozen | Cumulative external Prefill checkpoint at the named commit/protocol. |
| `docs/analysis/evalscope-prefill-cumulative-c92a2ef-2026-07-30/README.md` | `experiment_evidence` | frozen | Cumulative external Prefill checkpoint at the named commit/protocol. |
| `docs/analysis/evalscope-prefill-gdn-bv64-0189050-2026-07-30/README.md` | `experiment_evidence` | frozen | External GDN BV64 direction evidence for its exact route and workload. |
| `docs/analysis/evalscope-prefill-gdn-wy-a23bf1d-2026-07-30/README.md` | `experiment_evidence` | frozen | External GDN WY direction evidence for its exact route and workload. |
| `docs/analysis/prefill-evalscope8-down-m128n256-2026-07-29/README.md` | `experiment_evidence` | frozen | Down M128xN256 external direction evidence for its recorded candidate. |
| `docs/analysis/prefill-evalscope8-fp8-supermatrix-v2-2026-07-30/README.md` | `experiment_evidence` | frozen | FP8 supermatrix external direction/admission evidence; promotion wording remains protocol-scoped. |
| `docs/analysis/prefill-gdn-conv-compact-qk-fused-2026-07-30/README.md` | `experiment_evidence` | frozen | GDN convolution/compact-QK candidate evidence. |
| `docs/analysis/prefill-gdn-partial-c64-padding-2026-07-30/README.md` | `experiment_evidence` | frozen | Arbitrary-tail C64 padding candidate evidence. |
| `docs/analysis/prefill-marlin-gate-up-fused-epilogue-2026-07-30/README.md` | `experiment_evidence` | frozen | Gate/Up fused-epilogue direction/admission evidence. |
| `docs/analysis/prefill-p513-current-cumulative-nsys-2026-07-30/README.md` | `profile_evidence` | frozen | One cumulative real-model NSys attribution capture; diagnostic only. |
| `docs/analysis/prefill-p513-embedding-prompt-wide-2026-07-30/README.md` | `experiment_evidence` | frozen | Prompt-wide embedding admission evidence for the recorded route. |
| `docs/analysis/prefill-p513-fp8-supermatrix-ncu-2026-07-29/README.md` | `profile_evidence` | frozen | Matched FP8 supermatrix NCU architecture-selection evidence; diagnostic only. |
| `docs/analysis/prefill-p513-gate-m128n256-production-2026-07-29/README.md` | `experiment_evidence` | frozen | Gate/Up production-profile/admission evidence for the exact C512 scope. |
| `docs/analysis/prefill-p513-gdn-chunk-o-bv64-parity-2026-07-30/README.md` | `experiment_evidence` | frozen | GDN output BV64 parity/admission evidence for the named candidate. |
| `docs/analysis/prefill-p513-gdn-chunk64-architecture-2026-07-29/README.md` | `experiment_evidence` | frozen | Research-only Chunk64/WY architecture screen and numerical characterization. |
| `docs/analysis/prefill-p513-gdn-compact-qk-packless-2026-07-30/README.md` | `experiment_evidence` | frozen | Compact-QK packless admission evidence for the recorded route. |
| `docs/analysis/prefill-p513-gdn-conv-token-parallel-2026-07-30/README.md` | `experiment_evidence` | frozen | Token-parallel convolution admission evidence. |
| `docs/analysis/prefill-p513-gdn-recompute-m64-2026-07-30/README.md` | `experiment_evidence` | frozen | GDN M64 recompute candidate evidence. |
| `docs/analysis/prefill-p513-gdn-wy-vllm-layout-2026-07-30/README.md` | `experiment_evidence` | frozen | Value-head-owned WY layout candidate evidence. |
| `docs/analysis/prefill-p513-nsys-2026-07-28/README.md` | `profile_evidence` | frozen | Real P513 kernel-list/NSys audit for the exact old protocol; diagnostic only. |
| `docs/analysis/prefill-p513-vllm-architecture-2026-07-29/README.md` | `external_profile_evidence` | frozen | Same-host stock-vLLM architecture profile; reference only. |
| `docs/analysis/priority-assessment-2026-07-23/source_notes.md` | `decision_evidence` | frozen | Source notes for the dated priority assessment; no current planning authority. |
| `docs/metadata/README.md` | `metadata_evidence_ledger` | frozen | Accumulated normalized checkpoint/build/evaluation facts; individual records retain exact scope. |
| `tests/fixtures/README.md` | `fixture_evidence_catalog` | frozen | Meaning and provenance of checked-in normalized correctness fixtures. |

## External reference documents (2)

| Path | Role | Lifecycle | Ownership boundary |
| --- | --- | --- | --- |
| `docs/FP8_MARLIN_W8A16_SOURCE_MAP.md` | `external_source_map` | frozen | First-party provenance/mapping for a pinned vLLM direct-port reference; no independent native production authority. |
| `docs/VLLM_HUMMING_STARTUP_AUDIT.md` | `external_architecture_audit` | frozen | Fixed source/audit snapshot for AOT-specialization study; source revisions and observed mechanisms are external, while native decisions remain in the SDD. |

## Third-party documents (2)

| Path | Role | Lifecycle | Ownership boundary |
| --- | --- | --- | --- |
| `src/kernels/sm87/third_party/vllm_marlin/README.md` | `vendored_source_notice` | frozen | Vendored vLLM Marlin documentation/license provenance; third-party content boundary. |
| `third_party/flashinfer/README.q3x.md` | `vendored_source_notice` | frozen | FlashInfer subset provenance and local integration notice; source ownership remains upstream. |

## Integrity check

After all intended Markdown additions/deletions are staged, the literal paths
in the tables above must equal the sorted output of:

```bash
git ls-files '*.md' | sort
```

The check must fail on any missing, extra, or duplicate path. A wildcard such
as `docs/analysis/**` is not a registration. Generated Markdown under ignored
`.q3x-work/` is outside the tracked registry and cannot be an authoritative
project document.
