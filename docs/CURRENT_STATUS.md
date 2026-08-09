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

Snapshot date: 2026-08-09. Implementation facts audited through code baseline
`18363ad`. Performance and qualification claims remain tied to their exact
artifacts and evidence authority; the short sealed layer-major result below is
a direction screen and does not revise target-length or production status.

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

The development tree now contains an executable, explicit layer-major
compatibility route. `ReferenceEngine` provisions the typed layer-major
`RequestState`, seals the exact model/state/arena/views/streams/two completion
events and 17 operator-role receipts, executes outer-layer/inner-logical-panel
traversal, finalizes logits from uncommitted hidden state, and publishes one
final sequence length through a move-only request receipt. A two-slot bounded
submission window carries API disconnect/shutdown cancellation into long
Prefill without exposing partial state; failures drain and reset through the
Engine transaction guard. Logical and physical tails are balanced without
changing the legacy scheduler, and eligible M32--M512 linear-attention layers
must bind the exact native C64 GDN tactic and its real workspace; only M1--M31
may use the sealed exact fallback.

This closes the system execution and API-observation shape, not the performance
architecture. Each logical C8192 panel still lowers to ordered physical
segments of at most C512 and uses the exact admitted M-at-most-512 FP8/NVFP4 Marlin
routes. Both inventories are development/test admissions. True C8192 NVFP4/FP8
surfaces remain isolated, the evaluation server selects layer-major only by an
explicit option, and the legacy route remains default. A clean-host real-model
P1025 OpenAI API/EvalScope screen is cumulatively positive against an older
greedy/fallback layer-major witness, but remains slower than the legacy path
and has no target-length selection or production authority.

Accordingly, current product status is **implemented evaluation runner,
unqualified production runner**.

## 3. Capability state matrix

| Capability | State at the audited baseline | What exists | What prevents the next state |
| --- | --- | --- | --- |
| Pinned Qwen3.6-27B NVFP4 model identity and loader | Implemented | Exact revision/shard authentication, one resident arena and typed weight binding | Must be tied to the installed binary and DeploymentPlan in release attestation |
| Pure C++ tokenizer and greedy generation | Implemented | Pinned tokenizer, batch-one generation and deterministic test/oracle surfaces | Public capability qualification is incomplete |
| OpenAI-compatible evaluation API | Implemented | `/healthz`, `/v1/models`, completions/chat, non-streaming and committed-token SSE; explicit layer-major mode has bounded Prefill cancellation | Loopback/evaluation-only; no production exposure, security, admission, or multi-tenant contract |
| Production serving API | Designed | Product/API contract is defined in the SDD | No installed release profile or release attestation exists |
| Default context capacity | Implemented at 8,192 | Server default `max_sequence_length=8192`, maximum output 4,096, 2 GiB request-arena limit | Does not admit the locked long-context workloads |
| 40K/60K/130K cold/no-cache service | Target; development compatibility route implemented | Token-ID ingress fails closed on capacity; layer-major startup reserves the request profile and binds a bounded exact segmented route | Whole-process capacity, clean-host target-length execution, native large-M tactics, performance, and qualification are absent |
| Prefill/Decode logical separation | Implemented in part | Separate phase APIs/metrics and an explicit state transition exist | Shared runner and synchronization-heavy physical plan prevent independent utilization and overlap |
| Layer-major C8192 candidate | Executable segmented compatibility route; native large-M candidate still designed | Sealed Engine transaction, balanced logical/physical tails, native exact C64 GDN receipt for eligible segments, true layer-major/logical-panel traversal, typed request profile, two-slot cancellation, 17 bound role receipts, explicit evaluation-server selection, v2 witness, and a clean-host P1025 direction screen | Physical execution remains M512-segmented/test-admitted; true C8192 surfaces, whole-process capacity, target-length real-model API result, release plan, and qualification remain open |
| Large-M Prefill specializations | Implemented as legacy admissions plus isolated C8192 surfaces | Native NVFP4/FP8/BF16 and Attention/GDN C512 candidates exist in development builds; isolated NVFP4/FP8 surfaces accept candidate-only C8192 panels | Existing options default off/test-only, C8192 surfaces are unbound, and no unique exact release selection exists |
| Decode target | Directionally near target | Short API evidence reports about 104 ms TPOT | At least 10 token/s, long-output stability, and release repetition are not qualified |
| Production accuracy | Target with partial oracles | Exact deterministic outputs are available for selected prompts/routes | No complete public capability baseline and promotion gate has passed |
| AOT DeploymentPlan | Implemented internally for the development route; release artifact still designed | Engine-lifetime sealed plan binds model/state/resources/operator identities and one-shot request receipts | No authenticated installed plan artifact is loaded and attested by the default release |
| Unique `BUILD_TESTING=OFF` release | Not implemented | Installed targets can be built | No single build/route manifest reproduces the strongest evidence without admissions or environment composition |
| Automated release evidence lane | Designed | Local CTest and evidence policies exist | No checked-in remote CI/workflow enforces the Orin release gate |

## 4. Current API and capacity facts

At implementation baseline `18363ad`, `EvaluationServerOptions` still defaults
to:

- bind address `127.0.0.1`;
- one serialized inference worker behind bounded ingress/inference queues;
- `max_sequence_length=8192`;
- `maximum_output_tokens=4096`;
- Prefill chunk size 512; and
- `request_max_arena_bytes=2 GiB`.

The adapter has no authentication or TLS. Legacy mode still observes
disconnect/shutdown only at committed-token boundaries. Explicit layer-major
mode uses a two-slot bounded window over `layer x logical-panel` quanta and
checks cancellation after retirement, final normalization, before logits, and
before the single commit; it drains and resets without publishing partial
position. The first response header is delayed until the first committed token
or an early error. These are honest evaluation-stage properties, not
production API guarantees.

Under the existing request-state planner at the maximum production-route
chunk M512, exact arena demand is:

| Maximum sequence length | Planned request arena bytes |
| ---: | ---: |
| 8,192 | 705,331,200 |
| 40,000 | 2,801,096,704 |
| 60,000 | 4,118,856,704 |
| 130,000 | 8,731,016,704 |

The 2 GiB default therefore fails the 40K target before performance is
considered. Merely increasing the command-line limit is not a production
solution: resident-weight/derived-layout footprint, transient Prefill
workspace, thermal headroom, cancellation, queue policy, and exact API
qualification must be planned together.

The unbound layer-major workspace planner separately reports the following
exact request-arena arithmetic for one explicit physical-tactic profile:
C8192 exact C64-native GDN with in-place convolution, current Release/default
legacy C16 GDN, and separate Gate+Up then SiLU. C64-native remains a
development admission and is not the default production route; at `18363ad`
it is nevertheless authenticated and mandatory for eligible M32--M512
segments in the explicit layer-major plan.

| Prompt tokens | Caller-selected conditional profile | Conservative disjoint profile |
| ---: | ---: | ---: |
| 40,000 | 3,975,374,848 bytes | 5,324,963,840 bytes |
| 60,000 | 5,496,014,848 bytes | 7,052,323,840 bytes |
| 130,000 | 10,818,254,848 bytes | 13,098,083,840 bytes |

The `selected` label in this table is a caller-selected host-planner strategy,
not selection of an architecture candidate or production route. It assumes
one prompt-wide hidden allocation and family-live-set overlay whose alias,
completion-event, and legacy-route-exclusion contracts are still unbound. The
conservative profile uses two prompt-wide hidden buffers and makes the three
C8192 operator families plus the legacy C512 workspace disjoint; it still
depends on the named phase-local layout contract inside each selected tactic.
Changing the tactic changes the exact total: token-parallel C64 convolution
raises the three conservative rows to 5,335,449,600, 7,062,809,600, and
13,108,569,600 bytes; fused Gate+Up epilogue raises the C8192 overlay from
855,638,016 to 940,572,928 bytes. A disjoint test-only native legacy GDN adds
another 75,694,080 bytes, whereas the current Release C16 route adds none.
Every profile also includes an independent 10,240-byte final-hidden handoff.
The initial `RequestState` shape keeps the C8192 family overlay but gives the
legacy C512 workspace disjoint storage; its exact 40K/60K/130K totals are
4,066,344,960, 5,588,904,960, and 10,917,864,960 bytes. A 32,768-byte C8192
token-ID staging view reuses the operator-arena prefix only after an explicit
embedding-consumed event, so it does not add another allocation.

All six request-arena values fit the planner's declared
17,437,720,576-byte limit, but no whole-process fit follows. Resident-model and
derived-sidecar byte requirements are absent, total whole-process bytes are
unknown, and the planner's whole-process capacity verdict is
`kIndeterminate`. The layer-major Engine create path can allocate the selected
request arena, expose typed phase views, and bind the segmented compatibility
executor and completion events. This is a fail-closed development admission,
not a production capacity verdict, and it does not replace the default M512
route or establish that model plus sidecars plus the largest request fit
concurrently.

## 5. Current performance evidence and its authority

### 5.1 Sealed layer-major P1025 direction screen

Commit `18363ad` was measured through the real OpenAI-compatible API with
EvalScope 1.9.1 on one hash-locked 1,025-token prompt and one greedy output
token after a clean-host Jetson preflight. The exact binary SHA-256 was
`de172fd1dca5d62241f6df5954bbc90efce4d8e3ba0d7cf1ccf8ca28f4aae8c0`.
The prior comparator is an older full-inventory build family using greedy
layer-major physical segmentation and exact fallback for FP8, Attention, and
GDN.

| Metric | Prior greedy/fallback layer-major | `18363ad` sealed composition |
| --- | ---: | ---: |
| EvalScope TTFT | 3,291.95 ms | **2,767.24 ms** |
| Server pure Prefill | 3,285.658588 ms | **2,761.173664 ms** |
| EvalScope prompt throughput | 311.36 tok/s | **370.40 tok/s** |
| Server pure Prefill throughput | 311.96 tok/s | **371.218954 tok/s** |

The cumulative sealed route saves 524.484924 ms of pure Prefill, a 1.189950x
speedup or 15.962855% latency reduction against that older witness. This is
not a single-mechanism comparison: the older witness reports production
dispositions only for NVFP4 Gate/Up and Down, with FP8 QKV/Z/O, Attention, and
GDN on exact fallback. The `18363ad` v2 witness records 64 NVFP4 Gate/Up hits,
64 NVFP4 Down hits, 96 FP8 QKV hits, 48 FP8 Z hits, 64 FP8 O hits, 16 Attention
hits, and 48 native GDN hits on their admitted production dispositions, with
zero exact fallback, forbidden fallback, Prefix cache, MTP, cuBLASLt,
external-reference, or approximate-route hits. The output token is exactly
`在` and finishes by the declared one-token length cap.

This is a single-request short direction screen, not an architecture
selection, target-length result, repetition sample, or production result. It
remains 415.976098 ms (17.737358%) slower in pure Prefill than the older
2,345.197566-ms legacy P1025 observation. The evidence therefore retains only
the complete sealed composition direction; it does not attribute the gain to
balanced segmentation, native GDN, or any other constituent in isolation. It
also confirms that the M512 compatibility internals must give way to the next
operator-panel dataflow boundary.

Source:
[`metadata/qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json`](metadata/qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json).

### 5.2 Latest retained cumulative short external result

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

### 5.3 Current cumulative internal Prefix attribution

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

### 5.4 Historical external baseline

The earlier 32-request EvalScope 1.9.1 run used 20--1,160-token prompts and 16
output tokens. Native mean TTFT was 3,168.79 ms versus 1,144.51 ms for matched
stock vLLM; total workload prompt throughput was 102.8141 versus 182.1476
tok/s. It set Prefill architecture priority, but its native binary provenance
was incomplete and it was a single-process directional protocol. It remains
historical evidence, not current release qualification.

### 5.5 The 1,224.7335 tok/s number

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

Prefill and Decode are logically identifiable. The default legacy route is
still primarily serial: it processes bounded prompt tiles through the shared
runner and synchronizes before each tile commit. Gate/Up uses a limited
layer-local auxiliary-stream fork/join, but the default route has no general
double/triple-buffered cross-tile or cross-layer pipeline.

At chunk 512, 40K tokens require about 79 tiles. Repeating a 64-layer weight
and synchronization traversal for every small tile is a first-class
architecture seam. The required response is a whole prompt-span execution
plan with explicit state semantics, residency, buffer ownership and overlap,
not a return to unrelated kernel parameter scans.

The explicit development route now implements the whole-request response:

- immutable 64-layer/logical-C8192 topology and request-owned progress;
- typed prompt-wide `RequestState` storage with fixed final-hidden handoff;
- a sealed 17-role Engine plan bound to exact model, arena, views, streams,
  tactics, sidecars, workspaces, and two completion events;
- true `layer -> logical panel -> physical segment` traversal with balanced
  final logical and physical panel pairs rather than a one-token tail;
- fail-closed native exact C64 GDN binding for every eligible M32--M512
  linear-attention segment, with exact fallback restricted to M1--M31;
- an uncommitted retained-hidden finalizer and one final sequence-length
  publication guarded by a move-only receipt and RAII rollback; and
- a two-slot layer-panel submission window with bounded API cancellation and
  v2 success evidence.

This integration remains a compatibility executor: every logical panel is
physically segmented at M512, and the exact Marlin inventories that authorize
it are still test-only. It is therefore an executable route for real API
evaluation, not yet the native C8192 large-M dataflow, a performance-selected
architecture, or a production route. Isolated true-C8192 NVFP4/FP8 surfaces
remain outside the executor. The clean-host P1025 screen confirms that the
cumulative sealed composition improves its older greedy/fallback witness, but
it also remains slower than the legacy M512 route. The immediate boundary is
therefore a private operator-panel executor: bind the existing M-at-most-8192
FP8/NVFP4 wrappers directly while keeping dependency-ordered GDN, Attention,
and BF16 work internally segmented until their native panel contracts are
ready. NVFP4 and large-N FP8 wrappers still lower to ordered kernel segments
of at most M1024; only N1024 FP8 K/V may issue one M8192 launch. Only after
that complete composition passes a short API sanity check does the
40K/60K/130K target-capacity witness become meaningful.

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
| Product API and long-context admission | Configured token-ID validation and host requirement plans exist; 40K/60K/130K still do not fit or execute through the default contract | P1 |
| Exact deliverable identity | No unique `BUILD_TESTING=OFF` release or authenticated DeploymentPlan | P2 |
| Target-length performance and physical Prefill plan | Explicit balanced/sealed layer-major compatibility route has a positive P1025 direction screen, but remains physically M512-segmented/test-admitted and has no valid target-length API witness; production remains legacy M512 | P3 |
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
  cumulative admission evidence and an explicit, unqualified balanced/sealed
  segmented layer-major route whose P1025 direction screen reaches about
  371 pure Prefill tok/s.
- **Not current:** production server, 40K--130K support, release-grade vLLM
  parity, 1,224.7335 tok/s lossless Prefill, or a fully qualified 10 token/s
  Decode release.
- **Target:** accuracy-preserving, non-MTP OpenAI-compatible runner reaching
  40K--60K first response within 2 s, about 130K within 4 s, and Decode at
  least 10 token/s, first matching and then exceeding useful vLLM behavior.
