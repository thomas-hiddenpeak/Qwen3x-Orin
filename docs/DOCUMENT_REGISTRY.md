---
q3x_document:
  id: q3x-document-registry
  class: procedure
  status: active
  owner: project-maintainers
  authority: exhaustive tracked-Markdown inventory and classification
  effective: 2026-08-09
  last_reviewed: 2026-10-10
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

- Audit date: 2026-09-28.
- Expected integrated-tree coverage: **86 Markdown paths**.
- Classified: **86**.
- Unclassified: **0**.
- Duplicate registrations: **0**.
- Automated validator: **PASS** (`86` tracked, `86` registered, `41`
  required first-party headers, `57` headers checked, `1030` local links,
  `0` errors).
- Inventory basis: every literal path from `git ls-files '*.md'` in the
  integrated tree including the numerical repair and corrected whole-core
  service production switch, code/documentation audit, 0.8.1 reliability delivery,
  closed bounded non-MTP implementation and MTP host/device/whole-core handoff,
  batched-cache, multi-row verifier, full API direction and bounded MTP efficiency milestones; the paused reference-translation package is archived, with its original observations preserved.
- Header transition: every registered non-evidence first-party document now
  has a standard control header and is enforced by the validator. Frozen
  evidence and third-party documents are not rewritten merely to add headers.

`Lifecycle` describes the document, not the implementation or experiment it
mentions. A dormant local design becomes active only through the named SDD and
Roadmap work package; it never activates itself. `frozen` evidence keeps
authority only for its exact recorded protocol.

## Local work package (1)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `AGENTS.md` | `repository_work_package` | active | Repository execution and hygiene instructions; cannot amend mission, numerical contract, or business targets. |

## Normative (4)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `docs/DOCUMENT_GOVERNANCE.md` | `documentation_policy` | active | Document classes, control headers, SSOT, supersession, SDD composition, and completeness. |
| `docs/ENGINEERING_CONSTITUTION.md` | `constitution` | active | Mission, owner-set constraints, end-to-start/leakage philosophy, product targets, and project-wide boundaries. |
| `docs/PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md` | `numerical_equivalence_policy` | active | Prefill architecture proof order, finite-precision/reduction-tree identity, P40 arithmetic ledger, and production-observable liveness eligibility. |
| `docs/REAL_MODEL_PERFORMANCE_POLICY.md` | `evidence_policy` | active | Real-payload evidence, local retention, architecture-candidate qualification, and release promotion. |

## Active first-party documents (10)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `README.md` | `entry_point` | active | Production-service overview, paired performance summary, build/API quick start, and navigation; current capability and performance remain owned by Current Status. |
| `docs/CURRENT_STATUS.md` | `current_status` | active | SSOT for current default route, delivered capability, qualified metrics, and known gaps; replaceable, not an evidence ledger. |
| `docs/DESIGN.md` | `subsystem_design_index` | active | Compatibility entry point and map of detailed subsystem contracts; subordinate to `docs/SDD.md`. |
| `docs/PREFILL_ARCHITECTURE_RESET.md` | `subsystem_sdd` | active | Prefill input/output, state, ownership, synchronization, failure, handoff, and architecture-candidate contract; owns no delivery order or mechanism rule. |
| `docs/MTP_ADMISSION.md` | `subsystem_sdd` | active | Isolated greedy MTP draft/verification and transaction contract; configured drafts 2/3, bounded API composition, no default-route authority. |
| `docs/ROADMAP.md` | `active_plan` | active | SSOT for current dependency order, milestones, and exit criteria; not an experiment ledger. |
| `docs/SDD.md` | `system_sdd` | active | SSOT for the externally callable runner design from API and target workloads inward to kernels/deployment. |
| `docs/decisions/0001-end-state-first-leakage.md` | `accepted_adr` | active | Accepted decision recording end-state-first/leakage as the system design and evolution model; subordinate only to normative policy. |
| `docs/decisions/0002-prefill-attention-vllm-numerical-alignment.md` | `accepted_adr` | active | Accepted decision aligning prefill full-attention to the vLLM/FlashInfer BF16-probability numerical class with split-P tensor route; owner instruction 2026-09-21. |
| `docs/decisions/0003-prefill-target-hardware-bound.md` | `proposed_adr` | active | Proposed hardware-bound analysis reconciling the locked Prefill targets against the measured Orin SM87 ceiling; requests owner target adjudication; changes no target, route, or priority. |

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
| `docs/REFERENCE_RUNNER.md` | `component_contract` | active | Batch-one reference runner sequence, state commit/reset, prepared Graph executable/template ownership, and fixture boundary. |
| `docs/REQUEST_STATE.md` | `state_contract` | active | Per-request persistent state, workspace, RoPE, memory plan, and lifecycle ownership. |
| `docs/RESIDENT_WEIGHT_LOADER.md` | `loader_contract` | active | Authenticated resident-weight I/O, identity, memory budget, and loader boundary. |
| `docs/TOKENIZER.md` | `tokenizer_contract` | active | Pinned tokenizer/chat formatting resource and error contract. |

## Procedures (6)

| Path | Role | Lifecycle | Authority / ownership boundary |
| --- | --- | --- | --- |
| `benchmarks/evalscope/README.md` | `evaluation_fixture_procedure` | active | Pinned EvalScope workload-manifest generation and corpus handling. |
| `docs/DOCUMENT_REGISTRY.md` | `document_inventory` | active | Exhaustive Markdown path/class/role/lifecycle registry. |
| `docs/EVALSCOPE_EVALUATION.md` | `external_evaluation_procedure` | active | OpenAI-compatible EvalScope protocol, joint Prefill/Decode reporting, limitations, and reproduction commands; subordinate to evidence policy. |
| `docs/README.md` | `documentation_index` | active | Required reading order, SSOT map, active SDD links, and documentation navigation. |
| `docs/REFERENCE_BENCHMARK.md` | `benchmark_procedure` | active | Internal reference repeatability harness and CLI procedure; not product-performance authority. |
| `docs/decisions/README.md` | `decision_index` | active | ADR naming, status, supersession, and navigation procedure. |

## Historical (6)

| Path | Role | Lifecycle | Successor / authority boundary |
| --- | --- | --- | --- |
| `docs/PREFILL_REFERENCE_TRANSLATION_MATRIX.md` | `historical_translation_work_package` | historical | Archived reference-translation lineage, source pins and rejected candidates. Current route and numerical authority reside in Current Status and the numerical ledger; reuse requires a new Roadmap package. |
| `docs/GDN_PREFILL_DATAFLOW.md` | `dormant_local_design` | historical | GDN mechanism/design lineage. It becomes a bounded local work package only when activated by `docs/SDD.md`, `docs/ROADMAP.md`, and the active Prefill architecture candidate. |
| `docs/LARGE_M_PROJECTION_DATAFLOW.md` | `dormant_local_design` | historical | Large-M mechanism/design lineage. Gate/Up, Down, and FP8 rules remain dormant and role/shape scoped until explicitly activated. |
| `docs/PREFILL_ARCHITECTURE_RESET_LEGACY.md` | `historical_prefill_design` | historical | Preserved former mixed Prefill plan, measurements, feasibility arguments, local budgets, and execution order; superseded by `docs/PREFILL_ARCHITECTURE_RESET.md` and has no current planning authority. |
| `docs/PREFILL_REFERENCE_AUDIT.md` | `historical_architecture_audit` | historical | Superseded as current Prefill design authority by `docs/PREFILL_ARCHITECTURE_RESET.md`, which refines `docs/SDD.md`; retains pinned source-analysis provenance. |
| `docs/ROADMAP_LEGACY.md` | `historical_roadmap_ledger` | historical | Superseded by concise active `docs/ROADMAP.md`; retained for linked chronology, with Git history and evidence owning exact observations. |

## Evidence (43)

| Path | Role | Lifecycle | Evidence authority |
| --- | --- | --- | --- |
| `docs/analysis/publication-fusion-direction-2026-09-28/README.md` | `mixed_architecture_direction` | frozen | Up/SiLU and O/residual composition at 99e10db, bounded exact numerical admission and mixed P65/8K/40K API direction; closed without production promotion. |
| `docs/analysis/alternative-paths-2026-09-28/README.md` | `alternative_architecture_assessment` | frozen | Source assessment at 89853c9 of Prefill publication fusion, FP8 preparation reuse and bounded Decode opportunities; reuses existing profile, no new GPU run or production change. |
| `docs/analysis/ordered-down-direction-2026-09-28/README.md` | `negative_architecture_direction` | frozen | Two ordered Down implementations, bounded bitwise numerical admission and negative same-request 8K/40K API direction at 50f7323; candidate integration removed, no promotion. |
| `docs/analysis/non-mtp-architecture-assessment-2026-09-28/README.md` | `architecture_assessment` | frozen | Installed e3f90fd/0.8.1 P40000 profile attribution, historical failure audit and one unimplemented ordered Down producer/consumer hypothesis; no new performance baseline or production change. |
| `docs/analysis/product-readiness-audit-2026-09-28/README.md` | `product_readiness_audit` | frozen | Production-boundary source audit and host-only Unicode/ingress reproductions at 8c0eb23; test, readiness, reproducibility, packaging and coverage findings; no runtime repair or new model qualification. |
| `docs/PERFORMANCE_BASELINE.md` | `historical_performance_ledger` | frozen | Append-only accumulated component/runner observations for their exact commits/protocols; not current status or target SSOT. |
| `docs/PHASE0_EVIDENCE.md` | `milestone_evidence` | frozen | Phase-0 environment, checkpoint, oracle, and phase-boundary record. |
| `docs/analysis/decode-context-scaling-2026-09-27/README.md` | `experiment_evidence` | frozen | Source audit and isolated real-API fused-GQA direction at d6d5afb; numerical research only, no production selection. |
| `docs/analysis/decode-product-admission-2026-09-27/README.md` | `implementation_admission_evidence` | frozen | Compiled fused Decode numerical admission, bounded Graph ownership repair and API direction evidence; no numerical-contract change or production promotion. |
| `docs/analysis/decode-qualification-2026-09-27/README.md` | `qualification_evidence` | frozen | Frozen short/mid/40K numerical rejection, bounded direct-answer capability and lifecycle checks; no numerical-contract waiver or production promotion. |
| `docs/analysis/decode-numerical-repair-2026-09-27/README.md` | `numerical_repair_evidence` | frozen | FP32 partition/merge-staging correction, cancellation regression and bounded real-model/API results; remaining scalar-equivalence gap, no promotion. |
| `docs/analysis/decode-exact-performance-2026-09-27/README.md` | `exact_decode_performance_evidence` | frozen | Exact Decode scheduling and operand-pipeline comparison on real inputs and API; no automatic production promotion. |
| `docs/analysis/whole-core-service-replacement-2026-09-27/README.md` | `service_replacement_audit` | frozen | Source/history and host-contract assessment of full-range whole-core replacement; no new GPU timing or production selection. |
| `docs/analysis/prefill-reference-repair-2026-09-28/README.md` | `prefill_reference_repair_evidence` | frozen | Independent P8192 GDN/full-model baseline assessment, A/B coordinate repair, and stale-server evidence invalidation; no production promotion. |
| `docs/analysis/whole-core-production-switch-2026-09-27/README.md` | `numerical_qualification_erratum` | frozen | Historical error-normalization and capture-boundary erratum, fresh complete P8192/O1 state/logit comparison, and explicit negative production-switch disposition; no numerical waiver or default promotion. |
| `docs/analysis/decode-ordered-repair-2026-09-27/README.md` | `ordered_numerical_repair_evidence` | frozen | Scalar-equivalent ordered Decode dataflow and bounded numerical/API comparison; no implicit production or release promotion. |
| `docs/analysis/decode-performance-lessons-2026-09-27/README.md` | `retrospective_erratum` | frozen | Dated correction of Decode ceiling, causality and qualification inferences at c4c8a34; source audit only, no new performance or production selection. |
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

The 2026-10-09 shared-verification work package updates the existing MTP SDD
and Roadmap, followed by draft matrix, rejected verifier compositions and the
bounded bottleneck reset, rejected persistent/independent-CTA compositions,
closed certificate feasibility gate and rejected asynchronous packed-feed
composition, followed by rejected bounded-reduction/sparse-repair and streamed
certificate compositions, rejected direct-fragment successor and rejected partitioned
execution/compact-repair composition and rejected shared multi-query KV successor, followed by closed exact PV work elimination without retention and the rejected
startup-owned NVFP4 pair-layout composition, including removal of its earlier
pack/check prerequisite. The startup-owned compact FP8 input-layout successor
is also closed without retention after full d2 API/lifecycle checks; its runtime
paths are removed and its exact evidence remains linked from Roadmap and MTP.
The compact full-K residency successor is closed after negative P65 API
direction; its numerical passes and removed source remain frozen evidence. The registered set and
classes are unchanged. The phase-borrowed FP8 successor is also closed without retention after
complete numerical and d2 API checks; the temporary state/system exception
and runtime paths are removed. No Markdown path or classification changes.

The ordered-draft successor completes both API panels and remains a bounded
development dependency for the next verifier composition. Existing MTP SDD,
Current Status and Roadmap own its scope; no registry path or class changes.

The combined immutable decoder package updates the same MTP SDD and Roadmap;
it adds no Markdown path or classification. The package is closed after negative
P65 API direction; all new runtime paths are removed and frozen evidence retained.

The batched vocabulary finalization package updates the existing MTP SDD and
Roadmap and Current Status after both positive API direction panels. It remains
a bounded development dependency, with no Markdown path or class changes.

The bounded projection-counter diagnosis updates the existing Roadmap only;
no Markdown path or classification changes.

The projection-lifetime composition completes both API direction panels and
updates the existing MTP SDD, Current Status and Roadmap;
no Markdown path or classification changes.

The bounded register-lookahead composition closes without retention after full
d2 API checks; the existing MTP SDD, Current Status and Roadmap record removal
and frozen evidence without changing Markdown paths or classes.

The live-reduction ownership composition completes both API direction panels
and updates the existing MTP SDD, Current Status and Roadmap. It remains a
bounded development dependency without changing Markdown paths or classes.

The causal proposal-reuse feasibility assessment completes without selecting
a runtime composition and updates the existing Roadmap; no Markdown path or
classification changes.

The NV live-reduction composition completes both API direction panels and
updates the existing MTP SDD, Current Status and Roadmap. It remains a bounded
development dependency; no Markdown path or classification changes.

The softmax-lifetime composition closes without retention after numerical and
d2 API checks. Existing MTP SDD, Current Status and Roadmap record its removal
and frozen evidence; no Markdown path or classification changes.

The NV register-feed package closes without retention after complete numerical
and d2 API checks. Existing MTP SDD, Current Status and Roadmap record removal
and frozen evidence; no Markdown path or classification changes.

The independent accumulation-chain package closes after negative P65 API
direction. Existing MTP SDD, Current Status and Roadmap record removal and
frozen evidence; no Markdown path or classification changes.

The bounded draft INT8 host assessment rejects the fixed per-channel quantizer
before GPU work and updates the existing Roadmap only; no Markdown path or
classification changes.

The role-aware draft representation package updates the existing Roadmap and
closes at its failed host numerical gate. No native owner/consumer or MTP
numerical-contract extension is selected; no Markdown path or class changes.

The scalar-chain Tensor Core numerical feasibility package updates the existing
Roadmap and closes direct substitution after finite SM87 counterexamples;
no runtime integration, Markdown path or classification changes.

The direct PV composition closes after negative P8192 API direction. Existing
MTP SDD, Current Status and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The Gate/Up paired publication package closes without retention after numerical
and complete d2 API checks. Existing MTP SDD, Current Status and Roadmap record
removal and frozen evidence; no Markdown path or classification changes.

The bounded QK exact-lattice feasibility package closes its broad-cooperative
proposal at the historical real-input host screen and updates the existing
Roadmap; no runtime, Markdown path or classification changes.

The batched cache-reconciliation package closes without retention after complete
numerical and d2 API checks. Existing Roadmap, MTP SDD and Current Status record
removal and frozen evidence; no Markdown path or classification changes.

The activation-publication package closes after negative P65 API direction.
Existing Roadmap, MTP SDD and Current Status record removal and frozen evidence;
no Markdown path or classification changes.

The completed retained-v31 budget diagnosis updates the existing Roadmap only;
no Markdown path or classification changes.

The QK live-reduction package completes both API panels and updates existing
Roadmap, MTP SDD and Current Status. It remains a bounded development dependency;
no Markdown path or classification changes.

The original short-chain exactness assessment closes with a positive host
feasibility result and updates the existing Roadmap;
no Markdown path or classification changes.

The short-chain native instruction and current-payload admission closes with
positive numerical feasibility, including one conservative guard assessment.
The existing Roadmap records the immediate composed-executor/API successor;
no runtime, Markdown path or classification changes.

The short-chain executor closes after negative P8192 API direction and a bounded
same-ELF diagnostic. Existing MTP SDD and Roadmap retain the rejection; the
temporary Request State exception is removed. No Markdown path or class changes.

The lossless FP8 packet host gate closes below its payload-saving floor
and updates the existing Roadmap only;
no runtime, Markdown path or classification changes.

The compact QK row-sharing composition completes both API panels and updates
existing MTP SDD, Current Status and Roadmap as a bounded dependency;
no Markdown path or classification changes.

The shared-row PV recomposition closes without retention after complete numerical
and d2 API checks. Existing MTP SDD and Roadmap record removal and evidence;
no Markdown path or classification changes.

The Gate/Up shared-input chain composition closes after negative P65 API
direction. Existing MTP SDD and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The direct short-chain composition closes after negative P8192 API direction.
Existing MTP SDD and Roadmap retain the rejection; the temporary Request State
exception is removed. No Markdown path or classification changes.

The paired weight-decoding package completes both API panels and updates
existing MTP SDD, Current Status and Roadmap as a bounded dependency;
no Markdown path or classification changes.

The prefix-publication package completes both API panels and updates existing
MTP SDD, Request State, Current Status and Roadmap as a bounded dependency; no Markdown path or classification changes.

The expanded-QKV representation assessment closes at its host traffic screen
and updates the existing Roadmap only; no runtime, state contract, Markdown
path or classification changes.

The register-codebook package closes after negative P65 API direction; existing
MTP SDD and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The independent-row boundary package completes both API panels and updates
existing MTP SDD, Current Status and Roadmap as a bounded dependency;
no Markdown path or classification changes.

The Gate/Up input-residency package closes without retention after complete
d2 API checks. Existing MTP SDD and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The retained-v48 budget reconciliation updates the existing Roadmap only;
no Markdown path or classification changes.

The FP8 lane-owned codebook package closes after negative 8K API direction;
existing MTP SDD and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The complete FP8 operand-feed package closes at its compiled-mechanism gate;
existing MTP SDD and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The native FP8 cache-prefetch package completes both API panels and updates
existing MTP SDD, Current Status and Roadmap as a bounded dependency;
no Markdown path or classification changes.

The Attention rendezvous package closes without retention; existing MTP SDD and Roadmap
record removal and frozen evidence;
no Markdown path or classification changes.

The NVFP4 native-prefetch package closes without retention; existing MTP SDD and Roadmap
record removal and frozen evidence;
no Markdown path or classification changes.

The QK bulk-publication package closes without retention; existing MTP SDD and Roadmap
record removal and frozen evidence;
no Markdown path or classification changes.

The element-certificate package closes at host feasibility; the existing
Roadmap records its frozen result and lack of runtime changes;
no Markdown path or classification changes.

The FP8 static-stage package closes at its resource gate; existing MTP SDD
and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The static-gate API reconciliation completes without retention and records a
successor decision in the existing Roadmap; prior evidence stays frozen and
no Markdown classification changes.

The acceptance-backoff package closes after negative P65 API direction;
existing MTP SDD and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The QK position-parallel package closes after negative P8192 API direction;
existing MTP SDD and Roadmap record removal and frozen evidence;
no Markdown path or classification changes.

The FP8 ordinary-code group assessment updates the existing Roadmap only;
no Markdown path or classification changes.

## Integrity check

After all intended Markdown changes, run:

```bash
python3 tools/docs/validate_document_control.py
```

The check fails on any missing, extra, or duplicate registry path, required
first-party header defect, duplicate document ID, registry/header class or
lifecycle mismatch, missing or asymmetric supersession, repository-escaping
link, or missing local link target. A
wildcard such as `docs/analysis/**` is not a registration. Generated Markdown
under ignored `.q3x-work/` is outside the tracked registry and cannot be an
authoritative project document.
