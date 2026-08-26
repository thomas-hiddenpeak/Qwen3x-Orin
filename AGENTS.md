---
q3x_document:
  id: q3x-repository-work-package
  class: local-work-package
  status: active
  owner: project-maintainers
  authority: repository execution, continuity, workspace hygiene, and performance-run preflight
  effective: 2026-08-12
  last_reviewed: 2026-08-27
  supersedes: []
  superseded_by: []
  ssot_for: Codex repository entry and local operating constraints
  review_trigger: documentation entry, workspace boundary, or clean-host procedure change
---

# Repository operating instructions

Before planning, implementing, reviewing, or evaluating non-trivial project
work, and again after context compaction or contributor handoff, read the
complete canonical [`docs/README.md`](docs/README.md) entry point and follow
its required orientation plus the route for the active task. Identify the
current [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) snapshot and the
active [`docs/ROADMAP.md`](docs/ROADMAP.md) slice before acting. This file
intentionally does not duplicate the documentation index's reading list.

The engineering constitution is normative. Project-owner production
observations and targets are planning constraints, not claims to debate away
with a different local benchmark. When evidence conflicts, preserve the
target, audit the local protocol/configuration/path first, and report the
discrepancy without silently lowering the goal.

Performance work must begin on the real model's production generation/API
path. Component timing and Nsight evidence explain or qualify a direction;
they do not replace the whole-product result. Large gaps require an
architecture-level dataflow response and study of the proven vLLM,
FlashInfer, Triton, FLA, and Mamba paths before local parameter scanning.

Every implementation task must identify its originating product constraint,
the downward budget or contract that selects it, and the architecture
milestone at which its value returns to the real API. Local mechanism rules,
microbenchmarks, and profiler cells govern only an explicitly active local
optimization work package. They cannot set global priority or promote a
production route. A local mutation may be retained as a named architecture
prerequisite, but it must have a bounded composition point; production is
selected only from the whole runner.

This is an engineering-delivery program, not an evidence-production program.
Scale validation cost with the decision: use the smallest safe,
product-connected check for a reversible iteration; return a composed
architecture to the real API promptly; reserve strict repetition and the full
environment envelope for architecture selection and release qualification.
Once evidence answers the stated engineering question, implement, compose,
retain, or reject the direction instead of extending measurement for its own
sake. A local optimum never outranks whole-product fitness.

Do not claim a requested target is impossible from the limitations of the
current implementation. Such a claim requires a same-workload hardware bound,
matched profile evidence, and reconciliation with known implementations.

Preserve unrelated user changes in the worktree. Keep production accuracy,
non-MTP scope, reference-only cuBLASLt boundary, evidence requirements, and
commit history intact.

All project-generated artifacts must stay inside the repository workspace.
Use the ignored `/.q3x-work/` tree for temporary source copies,
builds, profiles, evaluation outputs, downloaded tool environments, and
retained experiment evidence. Do not create project-named files or
directories directly under `$HOME`. Treat user-owned model directories,
virtual environments (including `~/vllmEvn`), and shared caches as read-only
unless the project owner explicitly requests otherwise. Avoid `/tmp` for
large or persistent artifacts; when a tool requires `/tmp`, remove the exact
project-owned files promptly after the task completes.

Before each real-model performance or profiler process, record a lightweight
host/device preflight. On Jetson, use `tegrastats` plus CPU/process and
GPU-device-handle inspection; the incomplete Jetson `nvidia-smi` view is not
an idle or ownership authority. An unowned GPU handle, confirmed material
contention, a safety risk, or violation of a predeclared final-test contract is
a hard stop. Other environment observations qualify the record and guide
diagnosis; they do not
independently block an ordinary engineering iteration or erase unaffected
correctness, route, or output evidence. Follow
[`REAL_MODEL_PERFORMANCE_POLICY.md`](docs/REAL_MODEL_PERFORMANCE_POLICY.md) for
the exact decision-class authority.

Quiesce only identified interfering workloads, using recoverable actions.
Never suspend or terminate the active Codex/SSH control path, evidence writer,
or recovery-critical services to make the host look idle. After quiescence and
before each fresh real-model timing/evaluation process, run `sync` and attempt
`echo 3 | sudo -n tee /proc/sys/vm/drop_caches`; retain the result and
before/after memory state. A failed cache drop invalidates a cold-cache timing
claim, not unrelated engineering work.

Cooling is externally controlled. Do not inspect, modify, or gate on system
fan/controller state, and sanitize incidental fan fields from retained host
telemetry. Temperatures through 85C are normal; above 85C through 90C use
actual clock, over-current, and throttle observations as context; above 90C is
an operational stop.

Before adding, moving, or reclassifying Markdown, read
[`docs/DOCUMENT_GOVERNANCE.md`](docs/DOCUMENT_GOVERNANCE.md) and update the
complete [`docs/DOCUMENT_REGISTRY.md`](docs/DOCUMENT_REGISTRY.md). Historical
evidence and third-party documentation are registered but not rewritten to
make them appear current.
