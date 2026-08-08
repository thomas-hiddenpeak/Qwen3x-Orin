---
q3x_document:
  id: q3x-current-status
  class: active
  status: active
  owner: project-maintainers
  authority: current implementation, qualification, production, metric, and blocker snapshot
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: current delivered state and open production gaps
  review_trigger: any default route, capability, qualification, metric, release, or blocker change
---

# Qwen3x-Orin current status

Snapshot date: 2026-08-09. Implementation baseline audited:
`main@1d727c7`. Governance and system-design changes may exist on a separate
integration branch; they do not change the implementation state recorded
here.

This is the single point-in-time status page. It records what is target,
designed, implemented, qualified, and production. Architecture contracts
belong in [`SDD.md`](SDD.md), pending work in [`ROADMAP.md`](ROADMAP.md), and
immutable measurements in evidence records. A new result must update this
page explicitly before it may be described as the current project state.

## 1. Status vocabulary

| State | Meaning |
| --- | --- |
| **Target** | Required by the owner/constitution, but no claim of design or implementation follows |
| **Designed** | A reviewed contract or architecture exists; executable support does not follow |
| **Implemented** | Code exists on the named revision and can exercise the path; release qualification does not follow |
| **Qualified** | The exact implementation tuple passed its applicable correctness, API, performance, resource, repetition, and evidence gates |
| **Production** | The qualified route is the default installed `BUILD_TESTING=OFF` release, with an attested DeploymentPlan and no test-only composition |

These states are deliberately non-interchangeable. “Runs through the API”
means implemented. “Faster in a direction screen” means evidence about an
implemented candidate. Neither means production.

## 2. Answer-first state

Qwen3x-Orin currently has a real-model, batch-one native runner and a useful
loopback OpenAI-compatible evaluation gateway. It does **not** yet have a
uniquely defined, attested production release. Its strongest short-prompt
results were obtained from a cumulative production-like development build
whose optimized Prefill routes require `BUILD_TESTING=ON`, compile-time
admission options, and runtime route composition. Those results are not the
ordinary default release path.

The default evaluation server is limited to 8,192 sequence tokens and a 2 GiB
per-request arena. It therefore does not deliver the locked cold/no-cache
40K--60K or 130K Agent workloads. The current listener is explicitly an
evaluation adapter: loopback-only, unauthenticated, serialized batch one, and
without a production network, tenant, admission, or cancellation contract.

Accordingly, current product status is **implemented evaluation runner,
unqualified production runner**.

## 3. Capability state matrix

| Capability | State at the audited baseline | What exists | What prevents the next state |
| --- | --- | --- | --- |
| Pinned Qwen3.6-27B NVFP4 model identity and loader | Implemented | Exact revision/shard authentication, one resident arena and typed weight binding | Must be tied to the installed binary and DeploymentPlan in release attestation |
| Pure C++ tokenizer and greedy generation | Implemented | Pinned tokenizer, batch-one generation and deterministic test/oracle surfaces | Public capability qualification is incomplete |
| OpenAI-compatible evaluation API | Implemented | `/healthz`, `/v1/models`, completions/chat, non-streaming and committed-token SSE | Loopback/evaluation-only; no production exposure, security, cancellation, or multi-tenant contract |
| Production serving API | Designed | Product/API contract is defined in the SDD | No installed release profile or release attestation exists |
| Default context capacity | Implemented at 8,192 | Server default `max_sequence_length=8192`, maximum output 4,096, 2 GiB request-arena limit | Does not admit the locked long-context workloads |
| 40K/60K/130K cold/no-cache service | Target; partially designed | Runtime absolute limit and configurable planner can express larger values | Default admission/memory plan, API validation, cancellation, performance, and qualification are absent |
| Prefill/Decode logical separation | Implemented in part | Separate phase APIs/metrics and an explicit state transition exist | Shared runner and synchronization-heavy physical plan prevent independent utilization and overlap |
| Large-M Prefill specializations | Implemented as admissions | Native NVFP4/FP8/BF16 and Attention/GDN candidates exist in development builds | Options default off and are test-only; no unique exact release selection |
| Decode target | Directionally near target | Short API evidence reports about 104 ms TPOT | At least 10 token/s, long-output stability, and release repetition are not qualified |
| Production accuracy | Target with partial oracles | Exact deterministic outputs are available for selected prompts/routes | No complete public capability baseline and promotion gate has passed |
| AOT DeploymentPlan | Designed | Startup-specialization audit defines the intended mechanism | No authenticated plan artifact is loaded and attested by the default release |
| Unique `BUILD_TESTING=OFF` release | Not implemented | Installed targets can be built | No single build/route manifest reproduces the strongest evidence without admissions or environment composition |
| Automated release evidence lane | Designed | Local CTest and evidence policies exist | No checked-in remote CI/workflow enforces the Orin release gate |

## 4. Current API and capacity facts

At `main@1d727c7`, `EvaluationServerOptions` defaults to:

- bind address `127.0.0.1`;
- one serialized inference worker behind bounded ingress/inference queues;
- `max_sequence_length=8192`;
- `maximum_output_tokens=4096`;
- Prefill chunk size 512; and
- `request_max_arena_bytes=2 GiB`.

The adapter has no authentication or TLS and cannot interrupt a long Prefill
inside its tile sequence. The first response header is delayed until the first
committed token or an early error. These are honest evaluation-stage
properties, not production API guarantees.

Under the current request-state planner at chunk 512, approximate arena demand
is:

| Maximum sequence length | Planned request arena |
| ---: | ---: |
| 8,192 | 0.657 GiB |
| 40,000 | 2.609 GiB |
| 60,000 | 3.836 GiB |
| 130,000 | 8.131 GiB |

The 2 GiB default therefore fails the 40K target before performance is
considered. Merely increasing the command-line limit is not a production
solution: resident-weight/derived-layout footprint, transient Prefill
workspace, thermal headroom, cancellation, queue policy, and exact API
qualification must be planned together.

## 5. Current performance evidence and its authority

### 5.1 Latest retained main-line short external result

The latest retained cumulative comparison associated with main used eight
real requests, prompt lengths 32--1,025, 16 output tokens, concurrency one,
and a cumulative development build with all relevant Prefill admissions plus
runtime composition enabled.

| Metric | Native run 1 | Native run 2 | Frozen vLLM slice |
| --- | ---: | ---: | ---: |
| Workload prompt throughput | 183.341934 tok/s | 183.315553 tok/s | 181.896870 tok/s |
| Mean TTFT | 1,152.676220 ms | 1,153.721993 ms | 1,168.570642 ms |
| Mean TPOT | 104.083710 ms | 104.041991 ms | 104.466390 ms |
| Exact native outputs | 8/8 | 8/8 | not an accuracy oracle |

Source:
[`analysis/decode-gate-up-coupled-feed-vllm-parity-2026-07-30/README.md`](analysis/decode-gate-up-coupled-feed-vllm-parity-2026-07-30/README.md).

This is useful short-workload directional/parity evidence for the exact
admission composition. `Workload prompt throughput` includes the complete
request run and is not pure Prefill tok/s. The small panel, short contexts,
test-only build requirements, runtime selectors, and missing public capability
gate prevent a production or long-context claim.

The reported TPOT corresponds to roughly 9.61 token/s in the best of these
runs, which remains below the locked 10 token/s release target and is not a
long-output qualification.

### 5.2 Current cumulative internal Prefix attribution

One real-model P513 NSys capture reports:

| Scope | Wall time |
| --- | ---: |
| M512 Prefix | 1,217.934464 ms |
| M1 tail | 108.699616 ms |
| Prefix total | 1,326.634080 ms |
| Finish Prefill | 5.371744 ms |
| Prefix plus finish | 1,332.005824 ms |

Source:
[`analysis/prefill-p513-current-cumulative-nsys-2026-07-30/README.md`](analysis/prefill-p513-current-cumulative-nsys-2026-07-30/README.md).

This is one diagnostic capture, not a retention sample and not the target
workload. It attributes the admitted P513 path; it cannot establish 40K--130K
API fitness.

### 5.3 Historical external baseline

The earlier 32-request EvalScope 1.9.1 run used 20--1,160-token prompts and 16
output tokens. Native mean TTFT was 3,168.79 ms versus 1,144.51 ms for matched
stock vLLM; total workload prompt throughput was 102.8141 versus 182.1476
tok/s. It set Prefill architecture priority, but its native binary provenance
was incomplete and it was a single-process directional protocol. It remains
historical evidence, not current release qualification.

### 5.4 The 1,224.7335 tok/s number

The 1,224.7335 tok/s result belongs to an opt-in Factorized-R1 experimental
branch and a one-warmup/one-measurement P1853 `/v1/completions` direction
screen with `max_tokens=1`. It changes the numerical trajectory: its recorded
outputs do not match the native baseline, and its metadata explicitly marks
it `quality_production_eligible=false`,
`production_residency_eligible=false`, and
`performance_upper_bound_only=true`.

It is therefore a research upper-bound observation. It is not in main, not a
default path, not lossless, and not a production Prefill result.

## 6. Accuracy state

Production accuracy is a hard constraint; no lossy Attention, GDN state,
activation-quantization, or other changed numerical contract is eligible
without an explicit owner amendment.

Current evidence includes deterministic token/output and layer/component
oracles, but it does not close the product gate:

- the latest short cumulative route reproduced its native comparator on 8/8
  outputs;
- the first external native/vLLM comparison matched text on 26/32 requests,
  but neither runtime is the accuracy oracle; and
- the first public C-Eval attempt produced no parseable answer before its
  output cap, so the reported zero score is an invalid protocol result rather
  than a capability measurement.

A parseable public capability baseline, exact request/output contract,
deterministic production oracles, and post-integration repeat are still
required.

## 7. Current execution-architecture gap

Prefill and Decode are logically identifiable, but the current physical
runner is primarily serial. Prefill processes bounded prompt tiles through the
shared runner and synchronizes the stream at each tile boundary before
committing state. Gate/Up uses a limited layer-local auxiliary-stream fork and
join, but the system has no general double/triple-buffered cross-tile or
cross-layer pipeline.

At chunk 512, 40K tokens require about 79 tiles. Repeating a 64-layer weight
and synchronization traversal for every small tile is a first-class
architecture seam. The required response is a whole prompt-span execution
plan with explicit state semantics, residency, buffer ownership and overlap,
not a return to unrelated kernel parameter scans.

Unpinned or dirty experimental branches are intentionally excluded from this
status snapshot. A candidate affects current truth only after its exact commit,
route, numerical mode, and evidence authority are recorded; branch proximity
or a chat description is not implementation status.

## 8. Open gaps and roadmap ownership

The rows below are status facts, not an independent priority list. The sole
active dependency order and exit criteria are in
[`ROADMAP.md`](ROADMAP.md).

| Gap | Audited state | Controlling roadmap slice |
| --- | --- | --- |
| Product API and long-context admission | Evaluation adapter only; 40K/60K/130K do not fit the default contract | P1 |
| Exact deliverable identity | No unique `BUILD_TESTING=OFF` release or authenticated DeploymentPlan | P2 |
| Target-length performance and physical Prefill plan | No valid target witness; tile-major synchronization remains | P3 |
| Accuracy, capability, stability, and release evidence | Partial deterministic oracles; no complete qualification bundle | P4 |
| Packaging and operations | No attested install/startup/upgrade lane | P5 |

Subsystem and mechanism documents may explain these gaps but cannot reorder
them. When one changes, update this snapshot and let the Roadmap own the
resulting delivery sequence.

## 9. Measurement preflight

No performance run is valid while another unexpected host process owns
material CPU/GPU resources. Jetson resource preflight uses `tegrastats` and
process/device-handle inspection, never `nvidia-smi` as the idle authority.

## 10. Claim boundary

Until the gaps above close, use the following language:

- **Current:** real-model native evaluation runner with short-workload
  cumulative admission evidence.
- **Not current:** production server, 40K--130K support, release-grade vLLM
  parity, 1,224.7335 tok/s lossless Prefill, or a fully qualified 10 token/s
  Decode release.
- **Target:** accuracy-preserving, non-MTP OpenAI-compatible runner reaching
  40K--60K first response within 2 s, about 130K within 4 s, and Decode at
  least 10 token/s, first matching and then exceeding useful vLLM behavior.
