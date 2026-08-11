---
q3x_document:
  id: q3x-current-status
  class: active
  status: active
  owner: project-maintainers
  authority: current implementation, qualification, production, metric, and blocker snapshot
  effective: 2026-08-12
  last_reviewed: 2026-08-12
  supersedes: []
  superseded_by: []
  ssot_for: current delivered state and open production gaps
  review_trigger: any default route, capability, qualification, metric, release, or blocker change
---

# Qwen3x-Orin current status

Snapshot date: 2026-08-12.

This page is a replaceable state snapshot. It does not own architecture,
delivery order, or experiment history. The system design is in
[`SDD.md`](SDD.md), the only active dependency order is in
[`ROADMAP.md`](ROADMAP.md), and exact observations remain in their linked
metadata/evidence records.

## 1. Answer-first state

Qwen3x-Orin currently delivers a real-checkpoint, batch-one native evaluation
runner with a bounded loopback OpenAI-compatible adapter. It is not yet an
installed production service. The evaluation-adapter default capacity remains
8,192 tokens, and no canonical `BUILD_TESTING=OFF` binary plus authenticated
DeploymentPlan has passed the long-context, accuracy, stability, and packaging
gates.

The strongest whole-product P40 development observation remains the
default-off exact-P40000 whole-core route. It was measured from a binary-pinned
dirty tree above `a4f95ba`; the implementation was committed later as
`a46d165`:

| Observable | Current incumbent observation |
| --- | ---: |
| Consumed prompt tokens | 40,000 |
| Server pure Prefill | 101,831.853876 ms |
| Pure prompt throughput | 392.804397 tok/s |
| EvalScope TTFT | 101,870.53 ms |
| EvalScope minus server TTFT | 3.730501 ms |

That route improved the preceding retained direction by 7.02138%, but it is
not a production result. It was measured from a pinned dirty-tree binary, is
default-off, and inherits a known P513 full-state mismatch from its
FlashInfer Attention arithmetic. It therefore has neither production-accuracy
nor release authority. Its transaction, memory, route-receipt, and
whole-prompt control substrate is retained as development infrastructure.
Its timing authority is one clean-host real-API direction sample plus one
bounded NSys capture, not a repetition-qualified performance baseline.
The witness consumed all 40,000 prompt tokens and reported zero Prefix-cache,
MTP, cuBLASLt, external-reference, approximate, exact-fallback, and forbidden
route hits.
Exact evidence is frozen in the
[`v10` whole-core record](metadata/qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json).

The later independent stock-vLLM-Marlin projection reference completed the
same P40 API path at 101,857.500727 ms / 392.705493 tok/s. It supplied no
positive direction against the incumbent, so that skeleton is closed and
remains default-off and accuracy-unqualified. See the
[`v15` rejection record](metadata/qwen36-27b-prefill-p40k-vllm-marlin-parity-rejection-2026-08-11.json).

The incumbent is still 10.95x below the owner-established 4.3K tok/s useful
vLLM starting line. Pure Prefill alone consumes 50.92x the complete two-second
TTFT budget, while the measured TTFT is 50.94x the target. These are active
product constraints from the
[`ENGINEERING_CONSTITUTION.md`](ENGINEERING_CONSTITUTION.md), not limits
inferred from the current implementation. P60 and approximately P130 have not
been opened because no competitive, accuracy-admissible P40 composition
exists.

## 2. Current capability matrix

| Capability | Current state | Missing production condition |
| --- | --- | --- |
| Pinned Qwen3.6-27B NVFP4 checkpoint | Implemented | Installed artifact must bind checkpoint, binary, layouts, and DeploymentPlan in one attestation |
| Resident loader and typed weights | Implemented | Whole-process memory and release identity remain unqualified |
| Pure C++ tokenizer and greedy generation | Implemented | Public capability and long-run qualification remain incomplete |
| Loopback OpenAI-compatible evaluation API | Implemented | It has no authentication, TLS, multi-tenant admission, or production exposure contract |
| Final product API | Designed | No installed production server/profile or release attestation exists |
| Evaluation-adapter default maximum context | 8,192 tokens | Does not admit the locked 40K/60K/approximately-130K workloads |
| Target-length Prefill | P40 development route exercised | P40 is 392.804397 tok/s, accuracy-unqualified, and far below parity; P60/P130 remain unopened |
| Prefill/Decode phase identity | Logically separated | Physical scheduling and state ownership do not yet provide an independently optimized/overlapped production pipeline |
| Decode | Directionally near target | [Short API evidence](analysis/decode-gate-up-coupled-feed-vllm-parity-2026-07-30/README.md) is about 104 ms TPOT; at least 10 tok/s, long-output stability, and release repetition are not qualified |
| Production accuracy | Partial deterministic oracles | No complete public capability, hidden/state/logit, and release-repeat bundle has passed |
| Canonical release artifact | Not implemented | No unique `BUILD_TESTING=OFF` artifact reproduces the selected route without test admissions |
| Automated release lane | Designed only | Local tests and policies exist, but no checked-in Orin release workflow enforces the complete gate |

Status terms are strict:

- **implemented** means executable code exists;
- **default-off development** means an explicitly selected research route
  exists but cannot describe the release;
- **qualified** means the applicable real-model accuracy, performance,
  repetition, route, and resource evidence has passed;
- **production** means the installed default artifact and API are attested;
- **target** means required capability, not current implementation.

## 3. API and capacity snapshot

The evaluation adapter currently defaults to:

- loopback `127.0.0.1` binding;
- one serialized inference worker behind bounded ingress/inference queues;
- `max_sequence_length=8192`;
- `maximum_output_tokens=4096`;
- Prefill chunk size 512; and
- a 2 GiB request-arena admission limit.

It supports health/model discovery, completions/chat, non-streaming responses,
and committed-token SSE. It remains an evaluation instrument, not the final
serving boundary. The product API requirements are owned by
[`SDD.md`](SDD.md); the executable external procedure and metric semantics are
owned by [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md).

The current exact M512 request-state planner reports:

| Maximum sequence length | Planned request arena |
| ---: | ---: |
| 8,192 | 705,331,200 bytes |
| 40,000 | 2,801,096,704 bytes |
| 60,000 | 4,118,856,704 bytes |
| 130,000 | 8,731,016,704 bytes |

The 2 GiB default therefore rejects P40 before performance is considered.
Raising the command-line limit alone is not a capacity qualification: the
resident checkpoint, derived layouts, Prefill workspace, KV/recurrent state,
cancellation resources, thermal headroom, and request arena have not yet been
proven to fit simultaneously. Whole-process capacity remains indeterminate.

## 4. Current Prefill execution and attribution

The strongest P40 development route is layer-major and single-stream. For
each of 64 layers it performs:

```text
five M8000 fill panels
  -> one P40000 Attention or GDN core
  -> five M8000 drain panels
  -> one P40000 Gate+Up/SiLU and Down/residual MLP phase
```

The runner's two-slot submission window bounds cancellation and completion
retirement. It is not GPU double buffering: all kernels are submitted to one
CUDA stream, and the P40 whole-core path does not use the older auxiliary
branch stream. There is currently no general double- or triple-buffered
cross-panel/cross-layer pipeline.

The bounded whole-request NSys capture reports 102.121307 s around
102.113314 s of kernels; only 7.992928 ms, or 0.0078%, lies outside kernels.
The API and host launch gaps are therefore not the active P40 bottleneck.

| Dominant role | Calls | Total | Request share |
| --- | ---: | ---: | ---: |
| NVFP4 Gate/Up persistent Marlin | 64 | 37,273.068224 ms | 36.50% |
| FP8 Marlin | 1,040 | 25,864.646560 ms | 25.33% |
| NVFP4 Down persistent Marlin | 64 | 17,559.457280 ms | 17.19% |
| Whole-prompt FlashInfer Attention | 16 | 13,634.170272 ms | 13.35% |

These four roles account for about 92.37% of the request, and the current P40
path is kernel-dominated. Architecture selection, composition scope, and the
real-API return point are owned only by [`ROADMAP.md`](ROADMAP.md).

## 5. Retained and rejected Prefill code

The following selected routes are the minimum set needed to interpret the
current incumbent and the closed v11--v15 projection lineage. This is not an
experiment inventory; other earlier screens remain only in frozen evidence.
None below is a production path or an active parameter scan.

| Route | P40 pure prompt throughput | Current disposition |
| --- | ---: | --- |
| v10 whole-core substrate | 392.804397 tok/s | Strongest direction; retained infrastructure, default-off and accuracy-unqualified |
| Shape-wide NVFP4 v3 replacement | 376.030675 tok/s | Rejected; temporary runner overlay removed |
| v11 grouped projection reset | 205.951777 tok/s | Rejected |
| v12 phase-local BF16 projection | 320.472999 tok/s | Rejected; unsealed historical direction |
| v13 AOT packed projection v1 | 247.814694 tok/s | Rejected |
| v14 packed NVFP4 v2 | 311.300103 tok/s | Rejected |
| v15 stock-Marlin parity reference | 392.705493 tok/s | Rejected; no positive direction |

Exact negative observations and route limitations are frozen in the
[`shape-wide v3`](metadata/qwen36-27b-prefill-p40k-nvfp4-shape-wide-v3-rejection-2026-08-10.json),
[`v11`](metadata/qwen36-27b-prefill-p40k-projection-reset-rejection-2026-08-10.json),
[`v12`](metadata/qwen36-27b-prefill-p40k-phase-local-bf16-rejection-2026-08-10.json),
[`v13`](metadata/qwen36-27b-prefill-p40k-packed-projection-rejection-2026-08-10.json),
[`v14`](metadata/qwen36-27b-prefill-p40k-packed-nvfp4-v2-rejection-2026-08-11.json),
and [`v15`](metadata/qwen36-27b-prefill-p40k-vllm-marlin-parity-rejection-2026-08-11.json)
records. They remain evidence for their exact protocols; they do not select
the next architecture.

A previously discussed Factorized-R1 research direction changes the numerical
trajectory. No tracked qualification/evidence record currently gives it
current measurement authority. It is not mainline, lossless, or
production-eligible and must not be reported as current Prefill performance.

## 6. Accuracy and release state

Accuracy is a hard production constraint. Any changed precision, recurrent
state boundary, reduction tree, logits, or generated behavior must first pass
the declared exact/no-regression numerical and behavioral gates. An
approximate route or changed product numerical contract may not enter the
mainline without an explicit owner amendment.

Current evidence is incomplete:

- selected native routes have deterministic component, state, token, and
  output oracles;
- the short cumulative native route reproduced its comparator on 8/8 outputs;
- the first external native/vLLM comparison matched text on 26/32 requests,
  but vLLM is not the accuracy oracle;
- the retained FlashInfer P40 direction has a known P513 full-state mismatch;
  and
- the first public C-Eval attempt produced no parseable answer within its cap,
  so the zero score is a protocol failure rather than a model-capability
  measurement.

A release still needs a frozen public capability baseline, exact request and
output contract, full state/logit oracles for every changed numerical route,
independent-process repetition, and the installed-artifact attestation.

## 7. Open gaps

These rows report current facts only. They do not reorder the work; the active
sequence and successor identity live exclusively in
[`ROADMAP.md`](ROADMAP.md).

| Gap | Current fact | Roadmap owner |
| --- | --- | --- |
| Documentation-control propagation | Integrated here, but not yet carried to every active development baseline; Codex loads `AGENTS.md` only from the worktree in which a session starts | P0 |
| Product API and long-context admission | Validation and host planners exist, but the default contract cannot admit 40K/60K/130K | P1 |
| Exact deliverable identity | No unique release binary plus authenticated DeploymentPlan | P2 |
| Prefill parity and physical plan | P40 is 392.804397 tok/s, 10.95x below the useful vLLM line, and accuracy-unqualified | P3 |
| Accuracy, capability, stability, and release evidence | Partial oracles only; no complete qualification bundle | P4 |
| Packaging and operations | No attested install, startup, upgrade, or rollback lane | P5 |

## 8. Claim boundary

Use the following language until this snapshot changes:

- **Current:** real-model native evaluation runner; strongest default-off P40
  development direction is 392.804397 pure prompt tok/s and is
  accuracy-unqualified.
- **Not current:** production server, production-default 40K--130K support,
  lossless Factorized-R1 Prefill, vLLM parity, or a fully qualified
  10-token/s Decode release.
- **Target:** the accuracy-preserving, non-MTP, OpenAI-compatible runner and
  performance region locked by the Constitution.

Before any performance run or profiler capture, the clean-host preflight must
pass using `tegrastats`, CPU/process inspection, and GPU-device-handle
ownership. Jetson `nvidia-smi` is not an idle or attribution authority.
