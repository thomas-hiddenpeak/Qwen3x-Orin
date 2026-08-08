# Repository operating instructions

Before planning, implementing, or evaluating performance work, read these
documents in order:

1. [`docs/ENGINEERING_CONSTITUTION.md`](docs/ENGINEERING_CONSTITUTION.md)
2. [`docs/SDD.md`](docs/SDD.md)
3. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md)
4. [`docs/REAL_MODEL_PERFORMANCE_POLICY.md`](docs/REAL_MODEL_PERFORMANCE_POLICY.md)
5. [`docs/PREFILL_ARCHITECTURE_RESET.md`](docs/PREFILL_ARCHITECTURE_RESET.md)
6. [`docs/EVALSCOPE_EVALUATION.md`](docs/EVALSCOPE_EVALUATION.md)

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

Before every performance run or profiler capture, fail closed on a clean-host
resource preflight. On Jetson, use `tegrastats` plus CPU/process and GPU-device
handle inspection to establish idleness and ownership. The Jetson
`nvidia-smi` implementation is incomplete and must not be used to decide that
the GPU is idle or to attribute GPU consumers. Any unexpected CPU/GPU
consumer invalidates the run; do not retain or report its timing.

Before adding, moving, or reclassifying Markdown, read
[`docs/DOCUMENT_GOVERNANCE.md`](docs/DOCUMENT_GOVERNANCE.md) and update the
complete [`docs/DOCUMENT_REGISTRY.md`](docs/DOCUMENT_REGISTRY.md). Historical
evidence and third-party documentation are registered but not rewritten to
make them appear current.
