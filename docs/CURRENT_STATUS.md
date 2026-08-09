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

Snapshot date: 2026-08-09. Implementation facts include committed revision
`21d5c28` and its default-off v5 exact-Marlin M8192 projection route.
Performance and qualification claims remain tied to their exact artifacts and
evidence authority. Clean P513, P8192, and P7712 state screens passed, but the
subsequent cold/no-cache P40K real-API gate reached 670.53071 s EvalScope TTFT
and 670.486890 s server pure Prefill. The composition produced no system
improvement, is closed without P60K, and does not revise production or
architecture-selection status. The faster grouped-Q64 observations remain
accuracy-unqualified and cannot be promoted.

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
architecture. The exact layer-major tactic still lowers Attention and the
admitted FP8/NVFP4 projections to bounded physical segments. The tree now also
contains an explicit `native-group-q64-panel` Attention tactic and an explicit
`segmented-marlin-operator-panel` projection wrapper. Both are default-off and
accuracy-unqualified. The projection name is intentional: it groups logical
panel work but still lowers P40K to 12,992 physical FP8/NVFP4 Marlin launches;
it is not a native large-M implementation.

A separate default-off
`native-quantized-large-m-operator-panel` v5 tactic is now connected to the
same real generation/API path. It binds the authenticated FP8/NVFP4 Marlin
sidecars plus typed reduction arenas and locks under a distinct arithmetic
contract. A complete M8192 panel uses exactly one physical Marlin launch per
logical projection. Every partial panel retains the complete exact span
ledger, legacy MLP workspace, and per-span Down-to-residual interleave; the
old balanced P40K topology therefore remains `3x8192 + 2x7712`. This is a
bounded architecture candidate, not a claim that arbitrary aligned M or the
complete true-large-M dataflow is solved. The evaluation server still selects
layer-major and this tactic only explicitly, its layer-major Attention and
projection defaults remain exact, and the server-wide legacy route remains
default. The retained and rejected development screens, including their dirty
binary limitation, are frozen in the
[M8192/partial-panel direction record](metadata/qwen36-27b-prefill-exact-marlin-m8192-development-screens-2026-08-09.json).

The newest clean-host P40K real-API screen used exact Attention and the v5
projection tactic. EvalScope reported 670.53071 s TTFT; the server reported
670.486890 s pure Prefill, or 59.658139 prompt tok/s. Pure Prefill occupied
99.994017% of server TTFT and external TTFT exceeded server TTFT by only
3.702638 ms, so the API is unequivocally not the active bottleneck. The route
consumed all 40,000 tokens as `3x8192 + 2x7712`, with 1,008 bulk launches, 672
partial-oracle hits, 13,104 physical projection launches, and zero Prefix,
MTP, cuBLASLt, external, approximate, or forbidden fallbacks. It was about
0.53% slower than the prior same-payload exact-Attention observation, so the
M8192-only composition is rejected and P60K was not run. Evidence is frozen in
the [v5 P40K API record](metadata/qwen36-27b-prefill-p40k-native-large-m-exact-api-2026-08-09.json).

The fastest P40K grouped-Q64 screen remains 179.51119 s EvalScope TTFT and
179.395679 s pure Prefill, but it generated a different first token and remains
accuracy-unqualified. It is an architectural upper-bound clue, not a
production candidate.

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
| 40K/60K/130K cold/no-cache service | Target; unqualified P40K development routes exercised | Token-ID ingress fails closed on capacity; exact and grouped-Q64 routes consumed all 40K tokens through the real API | Latest exact P40K is 670.53071 s TTFT and rejected; fastest 179.51119-s grouped-Q64 route changes output and is accuracy-unqualified; P60K/P130K, whole-process capacity, and qualification remain open |
| Prefill/Decode logical separation | Implemented in part | Separate phase APIs/metrics and an explicit state transition exist | Shared runner and synchronization-heavy physical plan prevent independent utilization and overlap |
| Layer-major C8192 candidate | Executable compatibility route plus default-off unqualified panel experiments | Sealed Engine transaction and role receipts remain; grouped Attention emits v3/v4 evidence, the segmented projection wrapper emits v4, and the M8192 exact-Marlin projection route emits v5 with separate bulk/partial counters | The v5 route failed its P40K system gate; all candidate tactics remain default-off, accuracy qualification is open, and the next prompt-wide architecture is not yet implemented |
| Large-M Prefill specializations | One exact-Marlin M8192 single-bulk route is implemented; complete large-M architecture remains incomplete | The clean M8192 and partial-panel state screens passed and the admitted route bypasses host segmentation at exact M8192 | P40K reached only 59.658139 tok/s: the underlying Marlin kernel still decomposes large M internally, M7712 retains the span ledger, and shape-specific Gate/Up, Down, FP8, exact Attention, BF16 A/B, and GDN dataflows remain required |
| Decode target | Directionally near target | Short API evidence reports about 104 ms TPOT | At least 10 token/s, long-output stability, and release repetition are not qualified |
| Production accuracy | Target with partial oracles | Exact deterministic outputs are available for selected prompts/routes | No complete public capability baseline and promotion gate has passed |
| AOT DeploymentPlan | Implemented internally for the development route; release artifact still designed | Engine-lifetime sealed plan binds model/state/resources/operator identities and one-shot request receipts | No authenticated installed plan artifact is loaded and attested by the default release |
| Unique `BUILD_TESTING=OFF` release | Not implemented | Installed targets can be built | No single build/route manifest reproduces the strongest evidence without admissions or environment composition |
| Automated release evidence lane | Designed | Local CTest and evidence policies exist | No checked-in remote CI/workflow enforces the Orin release gate |

## 4. Current API and capacity facts

At the current implementation snapshot, `EvaluationServerOptions` still
defaults to:

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

### 5.1 Rejected P40K segmented-Marlin projection screen

At `bbd8ac3`, one clean-host EvalScope 1.9.1 request sent the pinned 40,000
token-ID prompt through
`q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel.native-group-q64-attention.v1`.
The v4 witness attested all 40,000 tokens consumed, five logical panels, 1,680
logical segmented-projection hits, 12,992 physical Marlin projection launches,
and 80 grouped-Q64 Attention hits. It reported zero Prefix-cache, MTP,
cuBLASLt, external-reference, or generic approximate-route hits. The plan is
nevertheless explicitly `accuracy-unqualified-architecture-candidate` and
does not qualify numerical equivalence.

| Metric | Result |
| --- | ---: |
| EvalScope TTFT | 179,511.19 ms |
| EvalScope prompt throughput | 222.827281 tok/s |
| Server TTFT | 179,507.414119 ms |
| Server pure Prefill | 179,395.678907 ms |
| Server pure Prefill throughput | 222.970811 tok/s |
| External TTFT minus server TTFT | 3.775881 ms |

The wrapper saved only 618.639057 ms of server pure Prefill against the
preceding 180,014.317964-ms exact-projection/grouped-Q64 screen, a 1.003448x
direction. This is effectively neutral relative to the architectural gap.
More importantly, the 179.51119-s external TTFT crossed the predeclared
`>=165 s` rejection boundary. The segmented wrapper was therefore rejected
immediately; P60K, P130K, NSys, and NCU were not run for it.

This result distinguishes logical grouping from native large-M execution. The
wrapper does not remove the structural physical-launch/data-presentation
boundary and cannot be described as a large-M kernel. It motivated the later
default-off M8192 exact-Marlin route; that newer route still requires the same
clean-host P40K real-API gate before it has architecture authority. Exact
hashes, route counters, timings, and the historical stop decision are frozen in
[`metadata/qwen36-27b-prefill-p40k-segmented-marlin-q64-api-2026-08-09.json`](metadata/qwen36-27b-prefill-p40k-segmented-marlin-q64-api-2026-08-09.json).

### 5.2 P40K native-group-Q64 Attention direction and T4 profile

At `a9a065e`, one clean-host, one-request real-API/EvalScope plus Nsight
capture consumed all 40,000 tokens and generated one token through the
explicit deployment plan
`q3x.sm87.ac-prefill-layermajor-8k.native-group-q64-panel.v1`. The plan is
default-off and identifies itself as an
`accuracy-unqualified-architecture-candidate`; its numerical contract is
explicitly unqualified. EvalScope 1.9.1 is the declared invocation version,
but the retained raw artifacts do not self-attest that package version.

| Profiled metric | Result |
| --- | ---: |
| EvalScope TTFT | 180,864.404898 ms |
| Server TTFT | 180,860.665933 ms |
| Server pure Prefill | 180,345.412463 ms |
| Server pure Prefill throughput | 221.796604 tok/s |
| External TTFT minus server TTFT | 3.738965 ms |
| `q3x.prefill.whole_request` wall | 180,340.668128 ms |
| Whole-request kernel union | 179,988.524640 ms (99.804734%) |
| CUDA streams / kernel overlap | 1 / 0 ms |

The top grouped-Q64 Attention kernel consumes 84,318.548 ms (46.8% of
kernel time). The two dominant still-segmented Marlin signatures consume
51,960.015 and 26,067.5183 ms (43.4% together), while the BF16 M16 pair adds
5,432.6802 ms across 120,000 launches. Attention plus the two main Marlin
signatures occupy 90.021892% of the whole-request wall. The trace therefore
locates the next composed architecture boundary in both exact causal
Attention and panel-capable projection ownership; the loopback API is not the
dominant measured budget. The two-slot host submission window created no
device-kernel overlap in this trace.

A preceding unprofiled direction screen on the same corpus observed
180,014.317964 ms (222.204547 tok/s) for this tactic versus
666,946.668018 ms (59.974810 tok/s) for the exact segmented Attention
context. That 3.704965x performance direction does not qualify the candidate:
the exact context generated `The`, while the candidate and profiled runs
generated `Based`, and no full-state accuracy gate passed. The profiler's
request-total and whole-workload-throughput fields are also polluted by
capture-stop/report-generation delay and are not unprofiled baselines.

This is a T4 diagnostic record, not a completed target-length architecture
witness. P60K and P130K were deliberately not run; the default exact/legacy
routes, production status, and target remain unchanged. Exact route, hash,
NVTX, kernel-union, top-20, and limitation evidence is frozen in
[`metadata/qwen36-27b-prefill-p40k-native-group-q64-panel-nsys-2026-08-09.json`](metadata/qwen36-27b-prefill-p40k-native-group-q64-panel-nsys-2026-08-09.json).

### 5.3 Sealed layer-major P1025 direction screen

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

### 5.4 Latest retained cumulative short external result

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

### 5.5 Current cumulative internal Prefix attribution

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

### 5.6 Historical external baseline

The earlier 32-request EvalScope 1.9.1 run used 20--1,160-token prompts and 16
output tokens. Native mean TTFT was 3,168.79 ms versus 1,144.51 ms for matched
stock vLLM; total workload prompt throughput was 102.8141 versus 182.1476
tok/s. It set Prefill architecture priority, but its native binary provenance
was incomplete and it was a single-process directional protocol. It remains
historical evidence, not current release qualification.

### 5.7 The 1,224.7335 tok/s number

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

The exact integration remains a compatibility executor whose physical work is
segmented and test-admitted. The explicit grouped-Q64 experiment changes the
Attention arithmetic and the connected segmented-Marlin wrapper does not
remove projection segmentation: its P40K v4 witness records 12,992 physical
projection launches. The wrapper's 179.51119-s API result failed the 165-s
stop-loss and remains accuracy-unqualified. The v5 projection tactic removes
software-visible segmentation only for complete M8192 panels; the underlying
Marlin body still decomposes large M internally and every partial panel retains
the exact span ledger. Its 670.53071-s exact-Attention API result failed the
system gate, so this composition is also closed without P60K.

The active architecture response is prompt-wide and shape-specific: exact
online-softmax Attention must operate on one logical M8192/M7712 panel rather
than repeated C512 host spans; NVFP4 Gate/Up and Down and each FP8 projection
family need tactics matched to their different N/K geometry with real cross-row
weight reuse; BF16 A/B and GDN must eliminate recursive span dispatch while
preserving exact recurrent boundaries. This is a new composed dataflow, not an
extension of the rejected v5 local scan. P60K/P130K stay deferred until the
same P40K API witness is competitive and accuracy-admissible.

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
| Target-length performance and physical Prefill plan | The exact-Attention v5 P40K request reached 670.53071 s TTFT and only 59.658139 tok/s; API overhead was negligible, and the M8192-only composition is closed. The 179.51119-s grouped-Q64 upper-bound route is accuracy-unqualified. A prompt-wide exact Attention, true shape-specific large-M projection, BF16 A/B, and GDN architecture remains open | P3 |
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

- **Current:** real-model native evaluation runner with explicit, default-off,
  accuracy-unqualified grouped-Q64 and segmented-projection experiments plus a
  state-screened exact-Marlin M8192 projection route. The latter reached only
  59.658139 tok/s on the cold/no-cache P40K exact-Attention API witness and is
  closed as a system direction. The faster grouped-Q64 route changes output.
  Exact/default routes and production status are unchanged.
- **Not current:** production server, 40K--130K support, release-grade vLLM
  parity, 1,224.7335 tok/s lossless Prefill, or a fully qualified 10 token/s
  Decode release.
- **Target:** accuracy-preserving, non-MTP OpenAI-compatible runner reaching
  40K--60K first response within 2 s, about 130K within 4 s, and Decode at
  least 10 token/s, first matching and then exceeding useful vLLM behavior.
