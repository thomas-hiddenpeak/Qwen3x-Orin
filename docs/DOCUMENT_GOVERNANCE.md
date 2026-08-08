---
q3x_document:
  id: q3x-document-governance
  class: normative
  status: active
  owner: project-owner
  authority: documentation-control
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: document classes, authority, lifecycle, supersession, and registry completeness
  review_trigger: any change to the documentation model or controlling document set
---

# Documentation governance

## Purpose and scope

This policy controls every Markdown document tracked by the Qwen3x-Orin
repository. It makes the documentation set usable as durable project memory
across contributors, agents, handoffs, and context compaction. It governs
classification, authority, single-source ownership, lifecycle, supersession,
and inventory completeness. The canonical inventory is
[`DOCUMENT_REGISTRY.md`](DOCUMENT_REGISTRY.md).

Documentation is part of the runner, not a commentary layer added after the
implementation. The controlling chain starts at the externally useful product
boundary and narrows toward implementation:

```text
mission and locked constraints
  -> externally callable runner and workload contract
  -> system SDD and current delivery plan
  -> subsystem designs and interface/numerical contracts
  -> implementation
  -> real-path observation and qualified evidence
  -> current status, decisions, and the next bounded change
```

The system is selected at the real API boundary. Local changes remain the
normal unit of implementation and causal learning, but their claimed scope and
authority must remain local until their value reaches a named architecture
candidate and then the real runner.

## Authority order and single-source ownership

When tracked documents conflict, use this order:

1. an explicit current project-owner direction and the
   [`ENGINEERING_CONSTITUTION.md`](ENGINEERING_CONSTITUTION.md);
2. normative project policies, including
   [`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md) and
   this document;
3. the canonical system SDD and its explicitly linked active subsystem SDDs;
4. active numerical, API, state, ownership, and release contracts;
5. active procedures and the current delivery/status documents;
6. historical documents, evidence records, external references, and vendored
   documentation.

`AGENTS.md` is a repository-local work package. It can make execution safer or
more restrictive, but it does not independently amend the mission, numerical
contract, business target, or production promotion rules.

Every subject has exactly one current source of truth. Other documents link to
that owner and add only their narrower scope. In particular:

- the constitution owns mission, owner-set constraints, and engineering
  philosophy;
- `SDD.md` owns the end-to-end external-to-internal runner design;
- `CURRENT_STATUS.md` owns the replaceable snapshot of what the default
  release can do now and what remains unqualified;
- `ROADMAP.md` owns the active delivery sequence and exit criteria;
- the real-model performance policy owns evidence and promotion rules;
- subsystem SDDs own only their declared subsystem boundary;
- immutable evidence owns observations for its exact recorded protocol, never
  current intent or a universal target.

A summary may repeat a fact only with a link to its owner and without changing
its scope. A copied table, threshold, or status is not a second source of
truth.

## Document classes

Every tracked Markdown file has exactly one primary class in the registry.
Status is separate from class.

| Class | Purpose and authority | Mutation rule |
| --- | --- | --- |
| `normative` | Mission, hard constraints, governance, or project-wide policy. | Change only with the authority named in its control header. |
| `active` | Current SDD, subsystem design, delivery plan, status, or product entry point. | Keep synchronized with the implementation and its declared owner document. |
| `contract` | API, numerical, data-layout, state, ownership, identity, or component behavior. | Change atomically with implementation/tests or record the implementation gap. |
| `procedure` | Reproducible build, evaluation, benchmark, fixture, or operating instructions. | Commands must remain executable or be explicitly marked unavailable. |
| `local-work-package` | Scoped repository/session instructions such as `AGENTS.md`. | May constrain local work; cannot redefine project-wide truth. |
| `historical` | Superseded design, roadmap ledger, audit, or narrative retained for provenance. | Freeze; correct through an erratum or successor, not silent rewriting. |
| `evidence` | An observation tied to a pinned workload, artifact, commit, or protocol. | Append only where the format permits; never rewrite a measured outcome. |
| `external` | First-party commentary about a pinned external implementation or source map. | Preserve source revision, license, provenance, and the reference-only boundary. |
| `third-party` | Documentation imported with vendored or third-party source. | Do not impose the project header or edit except as required by license/vendor updates. |

The `active` class includes several roles, recorded separately in the
registry: `system_sdd`, `subsystem_sdd`, `current_status`, `active_plan`, and
`entry_point`. Those roles are not interchangeable.

## Local mechanism scope

Mechanism-level engineering rules belong to the local optimization scope in
which they were derived. A tile, occupancy, cache, stream, fusion, profiler,
or single-operator rule must declare all of the following before it can guide
work:

- subsystem and call path;
- model role and exact shapes;
- numerical mode and accuracy boundary;
- hardware and software identity;
- current incumbent and evidence protocol;
- upstream assumptions and downstream consumer;
- condition that retires or reopens the rule.

Such a rule may select local variants within that boundary. It may not become
a runner-wide architecture principle, reject a different role/shape, lower a
product target, or override the real API result. Promotion from local rule to
system rule requires an explicit SDD change showing the end-to-end dataflow
and a real-path result demonstrating upward leakage. Conversely, a system
constraint may flow downward and deliberately create stricter local
requirements.

This rule prevents two symmetric failures: treating a good local mechanism as
a universal architecture, and dismissing useful local work merely because an
unrelated system bottleneck currently masks it.

## Standard document-control header

All new or materially rewritten first-party Markdown documents must begin with
this YAML front matter. Existing documents without it remain registered under
the transitional classification in `DOCUMENT_REGISTRY.md` until touched.
Historical evidence and third-party documents are not rewritten solely to add
a header.

```yaml
---
q3x_document:
  id: stable-kebab-case-id
  class: normative | active | contract | procedure | local-work-package | historical | evidence | external
  status: draft | active | frozen | historical | superseded
  owner: project-owner | named-maintainer-role
  authority: short statement of what this document controls
  effective: YYYY-MM-DD
  last_reviewed: YYYY-MM-DD
  supersedes: []
  superseded_by: []
  ssot_for: one bounded subject, or none
  review_trigger: event that requires review
---
```

Rules for the fields:

- `id` is stable across renames.
- `class` is one primary class from the table above.
- `status` is lifecycle state, not a performance verdict.
- `owner` is an accountable role, not the agent that last edited the file.
- `authority` states what downstream decisions the document may control.
- `effective` is the date the current controlling version became effective.
- `supersedes` and `superseded_by` contain repository-relative document IDs or
  paths, never only a chat reference.
- `ssot_for` is deliberately narrow. `none` is valid for evidence and history.
- `review_trigger` is concrete: an API/ABI change, new default route, target
  amendment, model revision, or scheduled release review.

## SDD composition rules

The canonical SDD describes the complete runner from the external API inward.
It owns the client-visible contract, request lifecycle, admission, scheduling,
Prefill/Decode phase boundary, state and memory ownership, observability,
deployment identity, failure behavior, and the interfaces between dominant
subsystems. It must define what is delivered before explaining how weights or
kernels are implemented.

A subsystem SDD is admissible only when it:

1. links to the parent SDD boundary it refines;
2. declares inputs, outputs, state, ownership, synchronization, and failure
   behavior;
3. identifies which constraints flow down from the product boundary;
4. distinguishes compatibility/production paths from opt-in research paths;
5. names the real API observation through which local value can flow upward;
6. scopes local mechanism rules instead of generalizing them to the runner;
7. links exact evidence rather than embedding a mutable performance history.

When a subsystem design is superseded, the canonical SDD changes its link in
the same commit. Old evidence remains reachable through the registry and Git
history.

## Lifecycle and supersession

The lifecycle is explicit:

```text
draft -> active -> historical -> superseded
                  \
                   -> frozen evidence
```

- `draft` may guide implementation only when a controlling active document
  delegates a bounded work package to it.
- `active` may control only the scope in its header and registry entry.
- `frozen` records an immutable contract or evidence snapshot.
- `historical` remains readable but has no current planning authority.
- `superseded` must name its successor; the successor must name what it
  supersedes.

Supersession is never deletion by implication. A replacement commit must
update both control headers, the registry, the documentation index, and all
controlling inbound links. Deleting a tracked document requires proof that no
current owner or evidence record depends on it. Git history alone is not a
substitute for a reachable evidence chain, but it may retain a large legacy
ledger once its current truth has moved to a concise active document.

Historical/evidence correction uses an explicit erratum containing the old
claim, corrected claim, reason, date, and supporting evidence. Do not silently
rewrite a measured number or make an old experiment appear to have used a new
protocol.

## Registry completeness

`DOCUMENT_REGISTRY.md` is complete only when:

1. every path returned by `git ls-files '*.md'` appears exactly once;
2. every first-party entry has one class, role, lifecycle status, and bounded
   authority/SSOT statement;
3. every historical or superseded entry names its current successor when one
   exists;
4. every third-party/external entry identifies the ownership boundary;
5. no wildcard stands in for individual evidence records;
6. no untracked chat, Home-directory file, or `.q3x-work/` artifact is treated
   as controlling documentation.

Adding, deleting, moving, or reclassifying a Markdown file requires a registry
update in the same commit. CI should compare the registry's literal paths with
`git ls-files '*.md'`, reject duplicates or omissions, validate first-party
headers when required, and check internal links. Until that check is automated,
the author must record the exact coverage count and zero-unclassified result
in the change handoff.

## Change procedure

For a documentation or implementation milestone:

1. start at the current external contract, status, and delivery goal;
2. identify the single owner document for every fact being changed;
3. update the controlling SDD/contract before or with implementation;
4. retain experiment observations as evidence, not active design prose;
5. update `CURRENT_STATUS.md` only after the default route and qualification
   state actually change;
6. update `ROADMAP.md` when dependency order or exit criteria change;
7. update the registry and index for any document-set change;
8. validate links, registry coverage, and contradictions before commit.

This procedure does not require every local experiment to rewrite the system
SDD. It requires a named local scope and an SDD update when the system boundary,
default route, or upward-leakage path changes.
