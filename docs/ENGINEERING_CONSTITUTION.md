---
q3x_document:
  id: q3x-engineering-constitution
  class: normative
  status: active
  owner: project-owner
  authority: highest project engineering authority below an explicit current owner direction
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: mission, locked product constraints, and engineering philosophy
  review_trigger: explicit project-owner amendment
---

# Qwen3x-Orin engineering constitution

This constitution governs project prioritization, performance methodology,
accuracy policy, and continuity across contributors, agents, sessions, and
context compaction. It controls planning when an older roadmap, audit, or
benchmark narrative conflicts with it. Historical measurements remain valid
only for the exact protocol they recorded; they do not retain authority to
lower a current project target.

`REAL_MODEL_PERFORMANCE_POLICY.md` defines the evidence required to retain or
promote code. This constitution defines what the project is trying to achieve
and how conflicting observations must influence the work. A project-owner
observation can therefore set priority and a target without automatically
becoming release-grade published evidence.

## 0. Project mission and paradigm

1. Qwen3x-Orin is not merely a one-off Orin engine or a collection of fast
   kernels. It explores and documents a repeatable engineering paradigm for a
   **specialized model, specialized numerical format, specialized hardware,
   and specialized runner**.
2. The pinned Qwen3.6/NVFP4/SM87 target is the concrete proof vehicle. The
   runner may deliberately give up general model and hardware compatibility to
   co-design weight representation, kernel dataflow, state ownership, memory
   lifetime, scheduling, API evaluation, and deployment. The reusable product
   is the method and evidence chain; the production binary need not be
   universal.
3. Matching vLLM is the minimum viability boundary for this proof, not its
   terminal ambition. A general engine that remains faster on the same useful
   workload indicates an unresolved specialized-runner design problem.
4. The project-owner's Qwen35-Thor work with GitHub Copilot and Opus is an
   established precedent for both the technical paradigm and productive
   human--AI engineering. It is relevant project evidence, not an incidental
   anecdote.
5. The collaboration process is itself part of the practice being evaluated.
   An engineering agent is expected to preserve owner intent across context
   changes, study proven implementations, reason at whole-system level,
   implement and validate autonomously, distinguish experiments from
   production, and deliver tracked commits. Repeatedly forcing the owner to
   re-establish accepted goals, defend production observations, or supervise
   local optimization choices is a process failure.
6. Source review, profiler evidence, and external frameworks serve this
   mission by exposing transferable mechanisms and validating the completed
   runner. They are not authorities that outrank project-owner knowledge or
   substitutes for engineering judgment.

## 1. End-state-first, leakage, and controlled evolution

1. The delivered product is an externally callable, OpenAI-compatible runner
   for the pinned model/hardware tuple. The external API and the first useful
   response are therefore the logical starting boundary of system design.
   Weight loading, execution plans, schedulers, memory, state, operators, and
   kernels are internal means by which that product contract is satisfied.
   API-first does not require a general-purpose, multi-model, multi-tenant
   serving engine; specialization remains intentional.
2. Product fitness is lexicographic rather than a single speed number. Model
   accuracy, API semantics, requested capacity without silent truncation,
   production route identity, non-MTP scope, the cuBLASLt exclusion, and
   bounded resource behavior are hard constraints. Performance selects among
   candidates only after those constraints hold.
3. Final constraints must **leak downward** through a traceable chain:
   `product scenario -> API observable -> runner phase budget -> execution
   plan node -> subsystem contract -> local work package`. A local task that
   cannot name this chain has no authority to consume the active performance
   program.
4. Implementation value must **leak upward** through the real production
   route: a local mechanism changes a subsystem budget, the composed
   architecture changes a runner phase, and the runner changes the external
   API result. Architecture has an additional duty to remove any
   synchronization, layout, ownership, or measurement boundary that prevents
   a real local improvement from propagating upward.
5. Three engineering units are distinct:
   - a `local_mutation` is a bounded change to one mechanism or subsystem;
   - an `architecture_candidate` is a runnable composition of the mutually
     dependent local mutations needed to realize one complete dataflow; and
   - a `release_candidate` is an architecture candidate packaged in the exact
     production artifact and qualified against every product constraint.
   A local mutation is not required to cross whole-API noise by itself, and it
   can never claim production value by itself.
6. Local mechanism engineering rules, microbenchmark thresholds, profiler
   cells, tile constraints, and kernel-specific stop losses apply only while
   an SDD/roadmap-selected **local optimization work package** is active. They
   may choose implementation variants inside that package. They may not set
   the product target, reorder global priorities, define the final delivery
   shape, or keep a package alive after its architecture stop condition.
7. A locally positive mutation may be retained as an experimental prerequisite
   only when it is on the attested production route, preserves the numerical
   contract, beats its real-payload local incumbent above measured noise,
   belongs to a named architecture candidate, and has a predeclared
   composition deadline or budget. It does not change the default release.
   Once the dependencies close or the deadline expires, the complete
   architecture candidate must return to the target API workload; local gains
   are never added arithmetically to excuse a globally negative composition.
8. Controlled evolution is not random parameter scanning. Design fixes the
   product environment, invariants, search boundary, candidate lineage,
   fitness vector, selection points, and stop conditions. Local mutations
   provide variation; complete architecture candidates face system selection;
   release candidates face production qualification. Rejected and superseded
   lineages remain recorded so later agents do not rediscover them blindly.
9. The governing shorthand is:

   > Constraints leak downward from the final runner; variation occurs
   > locally; mechanisms compose in an architecture candidate; value must leak
   > upward to the real API. Local work may be retained, but it may not remain
   > suspended indefinitely, and only whole-product fitness selects production.

## 2. Trust and planning authority

1. A direct project-owner statement about observed production behavior,
   required accuracy, or the target hardware/model is an authoritative
   planning constraint. Treat it as an input to the work, not as an invitation
   to litigate the observation unless the owner explicitly requests an audit.
2. If a local benchmark conflicts with an owner-supplied production
   observation, first audit the local harness, token accounting, cache state,
   endpoint, model revision, startup configuration, execution path, and metric
   definition. Do not silently lower the target or use the conflicting local
   result to declare the production observation impossible.
3. Record unresolved evidence as a measurement-reconciliation problem. Keep
   advancing work that is safe under the owner-supplied target while the
   discrepancy is isolated.
4. A planning constraint changes only through an explicit owner decision or a
   same-workload reproduction that the owner accepts as superseding evidence.
   Approximate prompts, another context length, a component microbenchmark,
   or a differently configured endpoint cannot supersede it.

## 3. Proven implementation is an existence proof

1. Working vLLM behavior on the target Orin and pinned model family is the
   performance starting line. It is an existence proof that the hardware and
   model can reach that region, not a terminal aspiration and not an optional
   comparison.
2. When the native runner trails that starting line materially, inspect and
   reproduce the relevant vLLM, FlashInfer, Triton, FlashLinearAttention, and
   Mamba dataflows before inventing a local explanation for why the target is
   unreachable. Reference their scheduling, tiling, fusion, state-update,
   attention, and memory-lifetime ideas; adapt source only with the required
   license and provenance.
3. Framework overhead in a general Python engine makes the native runner's
   specialization opportunity larger. It is not a reason to accept slower
   native behavior.
4. cuBLASLt remains a measurement and numerical reference only. It never has
   production dispatch, fallback, retention, or promotion eligibility. vLLM
   is also not a runtime dependency requirement; it is the external starting
   point and architectural reference.

## 4. Product-first measurement hierarchy

Use the following hierarchy to decide what to do next:

1. **Real product path:** cold/no-cache OpenAI-compatible API behavior on real
   Agent prompts, especially time to the first visible generated token and
   end-to-end usability.
2. **External reproducible path:** EvalScope or an equivalent public framework
   using the same real model, exact tokens, cache policy, concurrency, and
   output contract.
3. **Engine path:** computed prompt tokens divided by the server's actual
   Prefill interval, separated from queueing, HTTP, and Decode.
4. **Component path:** real-weight and real-state whole-layer or kernel timing.
5. **Diagnostic path:** NSys/NCU/SASS counters used to explain an already
   observed real-path result.

A lower level may diagnose a higher-level result, but it cannot override it
without a same-workload reconciliation. Logger-window rates, P513 cells,
synthetic matrices, and isolated kernel peaks must never be presented as the
product result.

The real product path first identifies the active system constraint and selects
the local optimization work package. A `local_mutation` then uses the smallest
safe real-route admission and the applicable real-payload local comparator;
it is governed by the package's composition budget rather than required to
move the whole API independently. A complete `architecture_candidate` must
return immediately to the smallest target-representative production/API run
capable of selecting the whole composition. Positive system direction unlocks
complete qualification. A negative composition stops or redesigns that
architecture version; bounded profiling is optional when it answers a named
causal question.

## 5. Architecture before parameter scanning

1. When the whole-product gap is at least 2x, or the target requires a
   qualitative step, stop low-yield single-variable scanning as the primary
   strategy. Write the global dataflow: tensor shapes, traffic, ownership,
   residency, synchronization, pipeline stages, and phase budgets for every
   dominant Gate/Up, Down, FP8 projection, Attention, and GDN/SSM path.
2. Select changes capable of moving a complete Prefix budget or removing an
   architectural boundary. Micro-optimizations are appropriate near the
   target or when they validate a named mechanism needed by the architecture.
3. Do not infer a global optimum by accumulating hundreds of unrelated local
   wins. Revisit the kernel or runner skeleton when the ideal dataflow does not
   fit it.
4. Incremental native improvements are retained against the current native
   incumbent when they clear real noise. The vLLM starting line and any
   cuBLASLt external component ceiling are checked after cumulative progress;
   neither is a per-experiment rejection threshold.
5. Mutually dependent mechanisms may be implemented and evaluated as one
   architecture candidate when their value is structurally non-orthogonal.
   Single-variable isolation is a diagnostic tool, not a rule that forbids the
   complete dataflow required by the design. The package must still have a
   bounded mutation count, time/budget limit, and API return point.

## 6. Accuracy and feature boundaries

1. Production mainline changes must not degrade model accuracy. A path that
   changes the numerical contract, activation precision, recurrent-state
   boundary, logits, or generated behavior remains research-only unless the
   project owner explicitly authorizes that product-contract change.
2. Synthetic inputs remain useful for exhaustive correctness and smoke tests,
   but never select performance work. Performance decisions use real model
   weights and, when data-dependent, real prompt-derived activations and state.
3. MTP is excluded from the current Prefill and Decode targets. It is a later
   roadmap feature and cannot be used to claim the present goals.
4. Prefill and Decode remain logically distinct optimization phases even when
   they share a runner. Their kernels, state ownership, buffering, metrics,
   and priorities must be separable.

## 7. Claim discipline

Do not say that a target is impossible, at the hardware ceiling, or outside
the feasible region merely because the current implementation stalls.

An impossibility or ceiling claim requires all of the following:

- an exact same-model, same-hardware, same-workload definition;
- a defensible roofline using the actual quantized operations and memory
  traffic rather than an inapplicable BF16 proxy;
- matched NSys/NCU evidence for the complete dominant path;
- reconciliation with the fastest known vLLM/reference behavior;
- documented evaluation of materially different algorithms and dataflows.

Without that evidence, report an implementation limitation and the next
architecture hypothesis, not a hardware impossibility.

## 8. Locked business targets

These targets remain active until the project owner changes them:

| Phase | Required user-visible behavior | Restrictions |
| --- | --- | --- |
| Prefill | A cold/no-cache 40K--60K-token Agent prompt reaches first response in at most 2 seconds | real API, real model, no Prefix/KV reuse, no MTP, no accuracy loss |
| Prefill long context | A cold/no-cache prompt of about 130K tokens reaches first response in at most 4 seconds | same restrictions; no silent truncation |
| Decode | Single-request Decode reaches at least 10 token/s, corresponding to at most 100 ms/token | no MTP as the means of compliance |
| Accuracy | Production output/capability does not regress | public evaluation plus pinned deterministic oracles |

The observed vLLM Agent experience is the starting reference for these
Prefill targets. A ten-second `Avg prompt throughput` log around 4.3K token/s
is supporting telemetry, not an upper bound and not permission to reduce the
business target. EvalScope is another product-facing observation surface; a
short-prompt corpus or unmatched endpoint cannot invalidate the long-context
Agent requirement.

The native runner must first match vLLM's useful performance on this
specialized hardware/model family. Its market justification then requires
specialization to exceed that general engine while preserving accuracy.

## 9. Continuity and change control

1. At the start of performance work, and after any context compaction or
   handoff, read `AGENTS.md`, this constitution, the canonical `SDD.md`,
   `CURRENT_STATUS.md`, `REAL_MODEL_PERFORMANCE_POLICY.md`, the active phase
   architecture document, and the external-evaluation contract before
   proposing work.
2. Every material architecture decision, target change, retained result, and
   rejected mechanism must be written into tracked repository evidence. Chat
   history is not the authoritative memory.
3. Commit material milestones atomically and push them to the active remote
   branch after relevant verification. Preserve unrelated worktree changes.
4. When an older document conflicts with this constitution, annotate the old
   statement as historical or superseded rather than allowing both statements
   to guide current work.
5. Amend this constitution only through an explicit project-owner direction.
   Record the date, reason, and affected targets or methods in the amendment.

## 10. Workspace and host hygiene

1. All project-generated builds, temporary source copies, linked worktrees,
   profiler captures, evaluation databases, generated corpora, logs, and tool
   environments belong under the repository workspace. Use the ignored
   `.q3x-work/` tree when an artifact is not source-controlled.
2. Do not create project-named files or directories directly under the
   project owner's home directory. User-owned checkpoints, virtual
   environments such as `~/vllmEvn`, and shared caches are external inputs;
   treat them as read-only unless the owner explicitly requests a mutation.
3. Do not use `/tmp` for large or persistent project artifacts. If a system
   tool requires `/tmp`, keep the allocation bounded and remove or relocate
   the exact project-owned artifact immediately after use. A test that creates
   a small temporary file must clean it on every normal and failure path.
4. Before completing a work phase, audit project-owned processes, temporary
   files, worktree registrations, and storage use. Retain material evidence
   inside the workspace and remove only artifacts whose ownership and
   recoverability are established.
5. Before every performance timing or profiler capture, run and retain a
   clean-host resource preflight. On Jetson, `tegrastats`, CPU/process
   sampling, and GPU-device handle inspection are the authority for load and
   ownership. The platform's incomplete `nvidia-smi` must not be used to
   declare the GPU idle or identify all GPU consumers. Any unexpected CPU or
   GPU consumer invalidates the run; abort before timing or discard the result.

## Amendments

- **2026-08-09:** Initial constitution. It codifies owner-supplied production
  observations as planning constraints; product-first and actual-first
  evaluation; vLLM as the starting-line existence proof; architecture-first
  response to large gaps; accuracy, non-MTP, and cuBLASLt boundaries; claim
  discipline; and the 40K--60K/130K long-context Prefill targets.
- **2026-08-09:** Project-owner mission clarification. The primary project is
  the reproducible specialized-model/specialized-hardware/specialized-runner
  paradigm, with Qwen3.6/NVFP4/Orin as its proof vehicle and Qwen35-Thor as an
  established human--AI engineering precedent. The method should transfer;
  the binary is not required to be general.
- **2026-08-09:** Host-hygiene requirement. Generated project artifacts are
  confined to the repository workspace, large or persistent `/tmp` use is
  prohibited, and user-owned home-directory inputs remain untouched.
- **2026-08-09:** End-state-first and leakage principle. The external runner
  API becomes the logical design boundary; final constraints propagate down
  to scoped local work, local mechanisms compose into architecture
  candidates, and only whole-product fitness can select production. Local
  mechanism rules are explicitly confined to an active local optimization
  work package.
- **2026-08-09:** Benchmark resource gate. Every timing/profile begins with a
  clean-host preflight; Jetson load authority is `tegrastats` plus process and
  device-handle inspection, never the incomplete `nvidia-smi` process view.
