---
q3x_document:
  id: q3x-current-status
  class: active
  status: active
  owner: project-maintainers
  authority: current implementation, qualification, production, metric, and blocker snapshot
  effective: 2026-08-12
  last_reviewed: 2026-08-14
  supersedes: []
  superseded_by: []
  ssot_for: current delivered state and open production gaps
  review_trigger: any default route, capability, qualification, metric, release, or blocker change
---

# Qwen3x-Orin current status

Snapshot date: 2026-08-14.

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

A target-first stock-vLLM reference witness was then attempted from clean
commit `fe626be` with the real checkpoint, one cold/no-cache P40 API request,
one output token, and a whole-prompt scheduler budget of `40000`. The API
transaction completed and the process-group, route, and cleanup receipts were
captured, but the measurement is invalid: the request interval crossed the
hard `70C` thermal gate, reached `81.812C`, and later showed real CPU
downclocking from 2201 MHz to approximately 1.1 GHz. Its manifest is therefore
`valid=false` with an empty result set. No timing from that run is retained,
compared, or used to alter either the native incumbent or the owner-established
4.3K tok/s starting line. The exact invalidation and artifact hashes are in the
[`P40 reference-witness invalidation record`](metadata/qwen36-27b-vllm-p40-target-witness-invalid-2026-08-12.json).
The current stock-auto route also selected Marlin FP8, FlashInfer Attention,
and Triton/FLA GDN without Humming, so it is not yet a matched reconstruction
of the known optimized vLLM observation.

The retained warmup from that invalid package now has a diagnostic-only
three-surface reconciliation. The three primary surfaces are the vLLM backend
`Avg prompt throughput` logger at 3999.7 token/s, the request-bound server
Prefill interval at 368.579880 token/s, and prompt tokens divided by EvalScope
TTFT at 368.238711 token/s. The server-TTFT-derived 368.358289 token/s and
EvalScope's printed 368.2478 input-plus-output token/s are auxiliary
observations, not additional pure-Prefill surfaces. The logger value is
10.851650x the request-bound server rate. vLLM's logger source semantics and the
fresh-process singleton join are consistent with completed prompt work
landing in a later local logger interval. Future harness executions now
explicitly fix `VLLM_LOG_STATS_INTERVAL=10.0` and bind `envs.py`, the server
logging task, and `LoggingStatLogger`: ten seconds is the configured trigger,
while the printed rate divides locally recorded computed prompt tokens by the
logger's actual monotonic time since reset. That actual denominator is not
printed and need not be exactly ten seconds; the value is service-window
telemetry, not per-request latency or pure Prefill. The evidence
lacks a stable cross-surface request ID and a formal warmup thermal/frequency
envelope, so it changes neither the invalid run's status nor the 4.3K tok/s
owner-established starting line. Exact formulas and raw hashes are frozen in
the [`warmup metric reconciliation`](metadata/qwen36-27b-vllm-p40-warmup-metric-reconciliation-2026-08-12.json).

A default-off, host-only descriptor now freezes the next whole-system AOT
candidate for P40/P60/approximately-P130: the exact 64-layer GDN/Attention
schedule, 14 physical groups, five projection roles, paired BF16 A/B producer,
Q128/KV32 online-softmax Attention, and per-token-BF16 GDN transaction are
represented as one typed DAG with 39 resource and 13 event edge classes. A
second host-only contract freezes actual NVFP4/FP8-Marlin projection arithmetic,
same-CTA Gate/Up lifetime, source-to-packed-payload authentication fields,
Attention preprocess/core finite-precision order, and the exact-C16 GDN
candidate's `[head,value,key]` axes and recurrence order. Exact-C16 does not
inherit qualification from the deployed Chunk64 family. A third default-off
admission now hashes caller-supplied host byte intervals, performs and replays
the bit-exact packed permutation, seals a host manifest plus transform receipt,
and compiles real NVFP4 Gate+Up and Down CUDA bodies. A fourth default-off,
test-only slice implements the exact 64-layer real-checkpoint NVFP4 source
audit, bounded host transformation, owned 9,625,927,680-byte device arena,
upload completion, independent device readback, and SHA-256-backed receipt
issuance for 128 artifacts and 192 sources. The Engine now has a default-off,
startup-only trigger that skips mutually exclusive legacy Prefill projection
sidecars and performs a private, owner-backed, transactional `ModelWeights`
attachment after the complete upload is authenticated. The owner is
non-movable, public release fails while attached, and no naked descriptor,
receipt, or device view is exposed.

A provenance-frozen Release/SM87 test-admission probe at `855a7cb` has now
executed that preparation path against the pinned real checkpoint. It audited
192 source tensors, generated and independently read back 128 authenticated
NVFP4 artifacts in one 9,625,927,680-byte device arena, transactionally
attached the nonzero owner/allocation lifetime to `ModelWeights`, and kept all
mutually exclusive legacy Prefill sidecars disabled. The verified payload
catalog SHA-256 is
`367572d8f5aab87c655695fc621562e0e88cb5d1a9656370353d55ab1c4ebdbe`.
Every CUDA, inventory, attachment, and Engine-destruction check passed. The
source probe remains formally `fail` because its immediate in-process
`cudaMemGetInfo` recovery check reported an 8,340,946,944-byte deficit. A
hash-frozen post-exit Jetson `nvmap` snapshot recovered from a completed Codex
rollout records 2,012,087 reusable pages, an empty IOVMM orphan table, and only
the three allowlisted desktop clients. It explains 8,241,508,352 bytes and
leaves a 99,438,592-byte residual, below the fixed 256 MiB diagnostic
tolerance. A later full process/IOVMM/FSI/handle snapshot independently found
no matching process, client, handle, or orphan. The derived classification is
therefore `no_owner_leak`; it preserves rather than rewrites the source
`fail`. Exact identities, raw-event hashes, limitations, and claim boundaries
are frozen in the
[`real-checkpoint preparation record`](metadata/qwen36-27b-sm87-target-aot-real-checkpoint-preparation-2026-08-12.json).

This is preparation and lifetime-attribution evidence only. The attachment
does not authorize a launcher or bind a runner/API route, and the public
launcher remains deliberately fail-closed. That preparation record by itself
has no target-AOT numerical, generation, API, performance, release, or
production authority.
The observed 699,705.551133 ms online prepare/attach time is a correctness-run
startup diagnostic, not a performance baseline; it makes offline-persisted,
authenticated AOT payload generation plus direct startup loading an explicit
implementation requirement. The incumbent remains 392.804397 tok/s and the
installed route is unchanged.

A clean Release/SM87 probe at `9d0613a` has now closed the next narrow gate on
the same pinned checkpoint. The private layer-0 M192 candidate covers one full
M128 region and one predicated M64 tail. Gate+Up matched the canonical route at
all 3,342,336 BF16 elements and Down-plus-residual matched at all 983,040
elements; baseline/candidate/replay digests are identical in both roles, with
zero mismatches, intact guards, complete writes, preserved inputs, and zero
CUDA errors. The same ELF records Gate+Up/Down at 246/210 registers per thread,
76,800 dynamic shared bytes, zero local bytes, 256 threads, one active CTA/SM,
and exactly 16 physical CTAs on the 16-SM device; both geometry and resource
gates pass.

The child evidence still preserves `status=fail` because only its immediate
in-process `cudaMemGetInfo` recovery check failed. Its parent bound the exact
child PID/start time/ELF and captured canonical `/proc` and Jetson `nvmap`
immediately after exit. The original parent report remains `inconclusive`
because its parser did not recognize the real 11-column allocation-detail
format. A strict parser at `5687871` re-derived the same immutable raw snapshot
without overwriting either source: all 20 criteria pass, the classification is
`no_owner_leak`, the 7,047,852,032-byte free-memory gap is fully covered by the
8,241,508,352-byte page pool, and no probe process, probe-named client,
unattributed handle, or orphan remains. The exact three-layer status and claim boundary are frozen in
the
[`layer-0 M192 Oracle record`](metadata/qwen36-27b-sm87-target-aot-layer0-m192-oracle-2026-08-12.json).

This grants only layer-0 M192 real-checkpoint numerical authority, same-ELF
SM87 resource/geometry authority, and a bounded lifecycle diagnosis. It grants
no complete-model, public-launcher, generation, API, performance, release, or
production authority. The 701,709.6913 ms prepare/attach and 728,418 ms probe
wall times remain diagnostics, not performance measurements. The default
runner and its 392.804397-token/s incumbent are unchanged.

A subsequent clean Release/SM87 create transaction and fresh-process direct
load at `27f5c71` have now passed the narrow persisted-asset lifecycle
admission. The create transaction published the fixed 9,626,456,064-byte
bundle for all 128 artifacts and 192 authenticated sources. The load
transaction performed two complete host-authentication passes, read exactly
19,252,912,128 bundle bytes, performed zero source-tensor D2H, reproduced the
same payload and record-header catalogs, attached the private owner, and
repassed the layer-0 M192 Oracle. Both schema-v4 child probes preserve
`status=fail` solely because their immediate in-process
`cudaMemGetInfo` recovery checks failed; both strict canonical Jetson parent
reports classified the post-exit state as `no_owner_leak` and set
`combined_lifecycle_accepted=true` without rewriting either child status.

This closes persistence and authenticated direct-load admission only. It
grants no complete-model execution, generation, public-launcher, API,
performance, release, or production authority, and it changes neither the
default runner nor the Prefill incumbent. Exact identities, catalogs,
byte-accounting, source statuses, and claim boundaries are frozen in the
[`persisted create/direct-load record`](metadata/qwen36-27b-sm87-target-aot-persisted-create-direct-load-2026-08-13.json).

Commit `fad42f7` now exposes the first complete target-AOT composition through
an explicit, default-off `sm87-target-aot-p40` API route. It authenticates and
attaches all 256 projection artifacts from 400 checkpoint sources, owns the
5,075,652,608-byte exact P40001 request arena plus Engine RoPE, rearms that
state without request-time allocation, executes all 64 layers, and validates
the final greedy handoff receipt. The default build cannot select this route;
MTP, cuBLASLt, JIT, request-time repack/autotune, approximation, and fallback
remain excluded.

The first clean-host real-model P40000-to-one API smoke did not return. The
client received zero bytes and no HTTP response before its 840.000399-second
timeout, while the GPU continued working after disconnect. It produced no
route receipt, accuracy result, or valid timing, so it changes neither the
production path nor the 392.804397-token/s incumbent. Static causal closure
found no software deadlock but did identify a structurally serial composition:
the principal kernels are one CTA/SM, GDN uses only 16 owners to traverse 40K
tokens and three value heads, all projection roles use 16 persistent CTAs, and
Attention repeatedly traverses the causal KV work from high-resource Q
owners. The route also lacks end-to-end target-path cancellation and
host-visible stage progress.

This closes `AC-PREFILL-SM87-AOT-SYSTEM-v1` as a performance candidate. The
exact chain remains a default-off correctness and diagnostic control; it will
not receive an unchanged rerun or a local parameter scan. The successor must
first add bounded progress/cancellation, then replace the whole dataflow with
48-value-head exact GDN ownership, role-specific bulk projection plans, and a
K/V-reusing effective-Q Attention plan before returning directly to the real
P40 API gate. Exact identity, failure semantics, work ledger, and decision are
frozen in the
[`failed target-AOT P40 API record`](analysis/prefill-target-aot-p40-failed-api-2026-08-14/README.md).

Commit `291305b` has since closed the diagnostic liveness prerequisite on that
default-off control. The request owner now waits at authenticated per-layer
events, emits structured elapsed and layer-delta progress, polls disconnect
cancellation before submission, after each completed layer, and before final
commit, and maps cancellation to a drained private transaction without an
HTTP 500 or partial state publication. At most one complete layer is queued.
This supplies bounded observability for development; its 64 host waits and
one-layer cancellation bound are deliberately not the v2 performance
schedule, and no second unchanged-v1 timing run was made.

Two real-checkpoint architecture screens now further constrain v2. Dense
INT8/INT4 limb or bit-plane re-expression is rejected before CUDA because an
impossible one-pass INT4 mapping of the complete listed projection work still
requires about 11.46 seconds at an optimistic 170-TOPS planning rate, beyond
the 5.0-second projection allocation. The actual source-pinned ModelOpt
NVFP4-Marlin path decodes to BF16 and uses BF16 Tensor Core MMA; it supplies no
hidden integer shortcut. Separately, sampled layers 0, 31, and 63 contain zero
repeated K16 E2M1 blocks across 50,135,040 authentic Gate/Up/Down instances,
even when block scales are ignored. Dictionary and cross-Gate-Up identical/
proportional-code reuse are therefore rejected as material successors. These
negative class screens do not lower the owner-established 4.3K target; they
made matched production work and route accounting the projection P0.

That matched source/receipt audit is now closed. The 64-layer topology and
the complete `1,948,044,492,800,000` conventional projection-operation ledger
contain no missing layer or role: the retained stock vLLM route does not
delete this work through MTP, Prefix cache, external KV, terminal liveness, or
activation quantization. On SM87 its FP8 projections are BF16-activation
W8A16 and its NVFP4 MLP is W4A16. The audit also corrects an earlier invalid
comparison: vLLM's 128 FP8 and 128 NVFP4 counts are fused outer operations,
not physical CUDA launches. The pinned P40 Marlin source splits every such
outer operation into 40 row chunks, giving a source-derived 5,120 physical
kernels per quantized family; Q3X v10's authenticated receipt instead records
1,040 FP8 and 128 NVFP4 physical launches. Neither count is a throughput or
utilization proxy. The exact formulas, source hashes, numerical-route gaps,
and vLLM logger semantics are frozen in the
[`P40 matched-work ledger`](metadata/qwen36-27b-prefill-p40-matched-work-ledger-2026-08-14.json).
The ledger now also has a default-off host checker that materializes all 496
logical roles, 304 fused outer operations, and 1,216 authenticated Q3X v10
physical launches with exact M/N/K, BF16 activation dtype, source-scale
partition, physical artifact, and P8000/P40000 ownership. Its logical and
physical arithmetic totals independently close to the same
`1,948,044,492,800,000` operations. This checker is diagnostic route
accounting, not a proposed v2 launch topology or a performance result.

The same audit confirms that vLLM's interval `Avg prompt throughput` can
receive a completed request's full logical prompt count in a later logger
window; it is not a per-request pure-Prefill timer. This metric correction
does not downgrade the owner's optimized-route observation, 1--2 second Agent
experience, or the 4.3K planning floor. The exact optimized route identity was
not retained and remains unknown. Q3X therefore does not spend another run on
the unchanged stock reference: every new native API receipt will bind logical
roles, fused operations, physical launches, shapes, arithmetic and cache
state while the target remains fixed.

The exact GDN v2 topology is now frozen for implementation: 48 independent
value-head recurrence owners replace the v1 16-owner/three-head serial chain;
C64 exact preparation, token-serial BF16 recurrence, and exact bulk
RMSNorm/SiLU form separate bounded stages. Per-token BF16 state rounding and
pre-round same-token output use remain unchanged. WY/KKT/SSD, FP32
authoritative chunk state, and delayed rounding remain excluded. The first
default-off single-C64 cell is now implemented as three kernels with no
selector: a token-parallel producer, 48-head recurrence, and rows8 epilogue.
Static SM87 compilation reports respectively 36/78/25 registers per thread,
zero local bytes, and 2,048/34,056/0 bytes of shared memory; the separate
state-trace recurrence reports 80 registers. A deterministic adversarial SM87
CUDA oracle now passes bitwise over raw/final output, all 50,331,648 per-token
BF16 state words, final recurrent state, and convolution history against an
independent ownership/reference implementation with nonzero incoming state
and history. The exact source, binary, surfaces, resources, and authority
boundary are frozen in the
[`GDN C64 oracle record`](metadata/qwen36-27b-sm87-bulk-v2-gdn-c64-oracle-2026-08-14.json).
This is synthetic correctness authority for the single C64 cell only. A
host-only P40 dependency plan now closes 625 continuation chunks, two bounded
prepared/raw/history slots, three logical streams, six reusable events, and
6,987,776 bytes of private state, but no multi-stream launcher exists yet.
The cell remains test-only and single-stream, and has no real-activation, API,
performance, cancellation, executable double-buffer, whole-layer,
qualification, or production authority until the complete v2 composition
closes.

The full-Attention v2 topology is also frozen for its first executable slice.
The closed target-AOT control already uses Q128/KV32, 128 KiB shared memory,
254 registers per thread, one CTA/SM, and 7,500 CTAs per P40 layer; it has no
valid P40 timing because its request never returned. Direct Q256 cannot retain
the exact FP32 D256 output state without exhausting the complete SM87 register
file before any other state is counted. The selected slice therefore keeps
the Q128 arithmetic body and schedules 16 persistent CTA lanes as an L2
temporal cohort over one KV head, using snake epochs and no cross-CTA barrier.
Its effective Q is 2,048 only at the whole-GPU L2 service boundary. The
default-off exact-P40 cell is now implemented as four stream-ordered
same-KV-head launches. Its host proof maps all 7,500 real query tiles to one
writer and retains 52 final-epoch repeat bodies with stores disabled. The
candidate and refactored target-AOT control both compile to 254 registers,
zero stack/local bytes, 128 KiB dynamic shared memory, and one-CTA/SM capacity;
the control contract tests pass. The exact mapping, build, resources, traffic
hypothesis, and authority limits are frozen in the
[`Attention cohort record`](metadata/qwen36-27b-sm87-bulk-v2-attention-l2-cohort-2026-08-14.json).
The shared device body narrows arithmetic drift, but no candidate-versus-
control bitwise CUDA oracle exists yet, so numerical qualification remains
false. Expected DRAM reuse is a falsifiable architecture hypothesis, not a
measured speedup; a positive real-API composition must precede any NCU
attribution.

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
| SM87 whole-system AOT Prefill candidate | Default-off v1 is executable through the real P40 API but rejected as a performance composition after a zero-byte 840.000399-second failed smoke; diagnostic per-layer progress and bounded cancellation are now implemented, with no production or timing authority | Replace serial v1 ownership with the complete bulk-dataflow v2 successor, preserve bounded cancellation without diagnostic host serialization, then require a successful clean-host real-P40 API return before EvalScope or qualification |
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
| Documentation-control propagation | The canonical main line now has one `AGENTS.md -> docs/README.md` Codex entry; pre-existing dirty worktrees do not receive it until explicitly integrated, because Codex reads the worktree in which a session starts | P0 |
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
