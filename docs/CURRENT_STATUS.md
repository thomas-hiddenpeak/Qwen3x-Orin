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

The complete `AC-PREFILL-SM87-BULK-DATAFLOW-v2` route has now also reached
the real cold/no-cache P40000 OpenAI API boundary. The EvalScope 1.9.1 request
received zero bytes and timed out after 680.73 seconds with 0/1 success, so it
has no TTFT, prompt-throughput, or generated-output result. The server drained
the cancelled private transaction without a CUDA or dependency error. Its
`retired_prefill_quanta=38`, `layer=37` line records host submission progress,
not 38 GPU-completed layers and not a basis for extrapolated throughput. One
subsequent 120.002145-second bounded NSys request window attributes 84.2735%
of aggregate kernel time to the whole-role Gate+Up and Down kernels and
87.3325% to all projection kernels. V2 is therefore performance-rejected,
default-off, accuracy-unqualified, and non-production; no P60/P130,
qualification, repetition, or local V2 parameter scan follows. The incumbent
above remains unchanged. Exact negative evidence is frozen in the
[`Bulk V2 P40 rejection record`](metadata/qwen36-27b-sm87-bulk-v2-p40-rejection-2026-08-14.json).

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
This is synthetic correctness authority for the single C64 cell only. The
default-off P40 continuation is now executable as a 625-chunk double-slot,
three-stream layer session with six reusable slot events. A request session
seals all 48 layer bindings and static CUDA facts once, enqueues successive
layers through a device-event bridge, and drains only at request termination,
cancellation, or failure; a partial submission permanently poisons the owner.
Its complete 17-range private layout is frozen at 6,988,032 aligned bytes,
including 16-byte-aligned cancellation snapshots at offsets 6,987,776 and
6,987,792. A clean-host P40 scheduling oracle has passed the exact continuation
and cancellation cases, but this remains synthetic scheduling/correctness
authority. It has no real-activation, API, performance, whole-composition,
qualification, or production authority until the complete v2 request path
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
The candidate and control now also pass a same-binary exact CUDA oracle over
all 7,500 output tiles / 245,760,000 BF16 elements, with all 52 repeated bodies
suppressed and guard/input immutability checked. This is synthetic arithmetic
authority only; the resource receipt deliberately retains
`numerical_contract_qualified=false` and `production_dispatch_eligible=false`
until the whole request is qualified. Expected DRAM reuse remains a
falsifiable architecture hypothesis, not a measured speedup; a positive
real-API composition must precede any NCU attribution.

The first v2 FP8 family is likewise implemented and exact-oracle qualified as
a default-off numerical control. It freezes 128 authenticated role bindings,
40 segments per role, 5,120 M64N256K64 four-stage physical launches, and
complete allocation-range validation at engine seal. The request hot seam
performs no device, pointer, function-resource, occupancy, allocation-range,
or stream query. Its clean-host exact CUDA oracle passes all GDN QKVZ, full
QKV, and O shapes; static compilation reports 137/138/136 registers, zero
local bytes, 98,304 bytes dynamic shared memory, and one CTA/SM. The forty
segment boundary is no longer the selected performance topology: this code is
retained for exact constituent control while the successor moves each FP8
role to one whole-P40000 persistent launch.

The first joint NVFP4 v2 constituent is now closed as a default-off exact
control stepping stone, not a performance candidate. Its genuine one-launch
M1024 body and M64 tail pass the same-ELF v1 comparison, independent expected
values, control counts, cancellation, guards, and input-immutability oracle.
The final static record is 127 registers with zero stack/spill for M64 and 128
registers with 8 bytes of stack, 8 bytes of spill stores, and 16 bytes of spill
loads for M1024. One M1024 launch still executes four serial M256 scheduler
epochs and rereads the packed payload at each epoch; launch coalescing did not
create cross-epoch weight residency.

The closing traffic audit makes that topology ineligible before any real API
timing. Gate+Up issues 222.8224 GB/layer of A requests and 62.6688 GB/layer of
weight/scale requests; even an ideal four-row Gate+Up reuse leaves
15.74240256 GB/layer of weight traffic. Down issues 27.8528 GB/layer of H
requests and 31.3344 GB/layer of weight/scale requests; its row-outer complete
50.13504-MB sweep cannot survive in the 4-MiB L2 across rows, so the Down
weight lower bound remains 31.3344 GB/layer. Across 64 layers the resulting
3.012915-TB NVFP4 weight floor alone is 14.7115 seconds at 204.8 GB/s. Even
granting impossible four-row reuse to every NVFP4 role and the most optimistic
current FP8 reuse still gives 1.799839 TB, or 8.7883 seconds, before A/H,
outputs, Attention, or GDN. Zero-spill work on this skeleton is therefore
stopped. The selected successor is separate whole-P40000 persistent Gate+Up,
K-heavy Down, and FP8 role kernels with shape-specific L2 cohorts; the exact
control remains only an arithmetic and scheduler witness. The hash-bound
oracle, final resources, byte formulas, authority limits, and topology decision
are frozen in the
[`NVFP4 exact-control closure`](metadata/qwen36-27b-sm87-bulk-v2-nvfp4-exact-control-closure-2026-08-14.json).

The three role-specific whole-P40000 projection constituents are present as
default-off CUDA admissions and were later connected to the rejected V2 real
generation/API route. Gate+Up uses one 32-CTA cooperative launch with an
M64/N64/K64, 4M-by-8N schedule; it shares each BF16 A stage while preserving
independent Gate and Up packed operands, scales, FP32 accumulations, BF16-RNE
publications, and the SiLU-times-Up boundary. K-heavy Down uses a separate
M64/N256/K64, 8M-by-4N schedule and preserves the Down-scale, BF16-RNE,
residual-add, BF16-RNE publication order. FP8 uses an independent
M64/N128/K64, 4M-by-8N schedule and freezes 48 GDN-QKVZ, 16 full-QKV, and 64
Attention-output outer roles, one whole-role launch each. All three use 32
persistent CTAs, three `cp.async.cg` stages, two register-feed stages, full-K
FP32 accumulation, and no split-K/global partial C.

Their precomposition T0 resource observations are Gate+Up at 107
registers/thread and 38,400 bytes dynamic shared memory, Down at 111
registers/thread and 52,224
bytes, and FP8 GDN-QKVZ/full-QKV/Attention-output at 90/93/89 registers and
49,152 bytes. All recorded variants have zero stack, spill, and local bytes
and a static two-CTA/SM capacity on the 16-SM target. Down additionally needs
startup configuration of the 52,224-byte dynamic-shared attribute on the
exact kernel symbol. These are compiler/resource observations, not NCU or
performance evidence. Same-kernel reduced-domain synthetic exact oracles pass
for all three families and cover only their declared M64 correctness cells,
cancellation, guards, and input/payload immutability; they are not a P40000
execution, real-weight/activation qualification, or proof that the intended
A/B service cohorts remain resident in L2.

A whole-P40 v2 host contract froze the actual composition boundary:
five streams, 12 reusable events, exact family-arena live phases, the separate
1,280-byte device control plane, one mapped cancellation word, request-epoch
progress, no request-time static CUDA queries, no whole 5.075-GB arena reset,
and exactly one terminal host wait/drain. The data-plane cold reset is limited
to 78,446,592 bytes of persistent GDN state/history; control state has a
separate explicit reset. Attention preprocess/core, GDN state publication,
embedding, final norm/LM-head/argmax/D2H, all 496 logical roles, 304 fused outer
operations, and the complete 1.948-Pop projection ledger are part of the
terminal receipt. Its ABI-major 3 projection contract replaces the segmented
projection counters with one versioned successor receipt: exactly 128 FP8
whole-role launches, 128 NVFP4 whole-role launches, and 48 BF16 A/B launches
cover the same 496 logical roles and 304 fused outer operations. A receipt
containing any of the old 5,120 FP8 or 2,560 NVFP4 exact-control launches
fails closed; the controls remain oracle-only.

The independent default-off `RequestState` lifecycle owns exactly one
5,075,652,608-byte device data-plane allocation, borrows the v2 owner's five
streams, keeps the 1,280-byte device control plane externally owned, and owns
only a private eight-byte pinned `{token, nonfinite}` handoff. Request rearm
clears 78,446,592 bytes of persistent GDN state/history rather than the whole
arena, and the only D2H seam has a fixed internal source and private
destination. The composite owner/executor, final GDN state/history
publication, final norm, LM-head, argmax, and fixed D2H handoff are now bound
to the distinct `sm87-bulk-v2-p40` route and a success-only ABI-major 4/V17
terminal receipt. No terminal receipt was emitted because the real P40 request
did not complete.

The bounded causal profile contains 15,080 kernel instances and
147.337288416 seconds of aggregate GPU-kernel time inside a 120.002145-second
client window; the sum may exceed wall time because streams overlap. Gate+Up
accounts for 86.781391584 seconds (58.8998%), Down 37.384968352 seconds
(25.3737%), Attention 11.114187328 seconds (7.5434%), FP8 projections
4.507008224 seconds (3.0590%), the four GDN families 6.833257472 seconds
(4.6378%), and all other kernels 0.716475456 seconds. CUDA API attribution
records 16,760 `cudaLaunchKernel` calls with 150.840949824 seconds of
aggregate API duration and a 39.174362368-second maximum; small GDN
prepare/cancellation launches block behind already queued work. This evidence
answers the single predeclared causal question: collapsing a role into a
32-CTA cooperative whole-prompt launch reduced nominal launch boundaries but
did not create a competitive feed. It does not prove an L2 hit rate or grant
NCU, numerical, or production authority.

The current development tree contains the complete default-off, test-only
`AC-PREFILL-SM87-MACROFEED-v3` P40000 execution composition. It preserves the
v10 five-panel API/control and whole-prompt
FlashInfer Attention substrate, but binds role-specific non-cooperative
M128/N256/K64-class Gate+Up, K-heavy Down, and FP8 macro feeds to one private
startup package. Exact GDN uses one prompt-wide convolution launch plus eight
M5000 recurrence/norm/gate macrochunks per linear layer, retaining 48
value-head owners, pre-round same-token output use, and the per-token BF16
state publication boundary.

Engine startup authenticates 256 physical projection artifacts from 400
checkpoint sources, binds their 256 immutable layer-role launch bindings to
the actual CUDA-device identity, and seals the resource facts before requests
are admitted. The runner executes all 64 layers in natural order, observes
their completion, and can issue a private ABI-major-1 physical transaction
only after recording 128 FP8, 64 Gate+Up, 64 Down, 48 BF16 A/B, 432 GDN, 16
whole-prompt Attention, and 80 Attention-preprocess physical operations. A
failed or cancelled request cold-resets the complete mutable request-arena
prefix while preserving the immutable RoPE suffix and publishes no partial
state.

The distinct API identity is
`q3x.sm87.ac-prefill-sm87-macrofeed-v3.native-p40-target-aot-whole-model.v1`
and its append-only external schema is `target-prefill-witness-v18`. The
physical receipt has `physical-execution-only` authority with numerical and
production qualification both false. No real-model P40 request has yet
measured this route, so it has no performance, full-model accuracy, release,
or production authority. The v10 observation at 392.804397 prompt tok/s
remains the incumbent.

The complete admission binary and the focused 20-test V3/authority/asset
suite now pass. One timing limitation remains explicit: the controller
commits V3 state after its first-token finalizer, so the continuous internal
prompt interval includes LM-head/argmax and is not qualified pure Prefill.
V18 keeps physical `package_complete` independent, reports the pure phase as
unavailable, and leaves the first P40 direction decision to external
EvalScope TTFT/New Prompt Throughput. No duration subtraction is permitted to
manufacture a non-contiguous pure phase.

The active implementation successor is
`AC-PREFILL-SM87-MACROFEED-v4`. Its present boundary is narrower and must not
be confused with an executable whole-model or API route. A default-off host
contract fixes exact P40000 as five contiguous C8000 panels with the panel
loop outside the natural
64-layer loop, two 81,920,000-byte hidden planes, one 278,528,000-byte
phase-aliased scratch plane, two explicit 78,446,592-byte private recurrent
epochs, private KV valid-end, and final-only canonical publication. The
contract requires in-place/streaming Attention and GDN workspace aliasing.
The independent admitted bodies exist, and one isolated execution package now
composes layer-0 input normalization with its BF16 A/B projection; their
complete layer/request executor does not.

The Attention lifetime is now frozen more tightly than that allocation-only
statement. Each scratch row retains the projection-native 24-head
`[Q256, Gate256]` interleave. Preprocessing overwrites only each raw Q slot,
the exact online Attention result overwrites that processed Q slot in place,
and every Gate slot stays at its original address. The FP8 Attention-output
projection must gather the logical 6,144-wide K axis directly from those Q
slots; compacting Q/G or materializing a separate Attention-output plane is
not a V4 implementation. This removes the corresponding C8000×6144 repack and
copy from the candidate dataflow before any kernel is selected.

A host-only V4 request-state admission now makes the private transaction
explicit without granting execution authority. It binds two non-overlapping
recurrent banks to one owner/allocation and a fresh request epoch, requires all
48 Conv/GDN writers plus all 16 KV writers before each panel swap, rejects
stale panel/event generations, and models cancellation, poisoning, exact
epoch-five B-to-A canonical copy, and the final logical sequence-length fence.
It deliberately forbids a whole 78,446,592-byte pre-panel copy: each GDN layer
copies only its 61,440-byte convolution history while the 1,572,864-byte GDN
state writes active-to-candidate directly. Its completion records are named
and marked test-only; it owns no CUDA event, produces no physical execution or
Decode receipt, and remains default-off/non-executable.

A separate default-off V4 execution-event owner now owns three nonblocking
CUDA streams and nine CUDA events without exposing any raw handle. It begins a
request only from the live request-state object and its sealed private access,
then enforces five consecutive panels and 48 `NormReady → AB → AbReady`
cycles per panel. `PanelDone` remains device ordered and is never converted
into a per-panel host synchronization; only final publication, safe discard,
or exceptional drain may mint a physical completion receipt. A CUDA failure
physically drains all three streams, preserves its original cause, and leaves
the owner permanently poisoned. A narrow owner-issued driver now permits only
the fixed state transitions and typed input-norm/BF16-A/B submissions; each
kernel enqueue and its ready-event record occur atomically under the owner
lock. The driver shares ownership of its event owner, so it cannot outlive the
streams it addresses; no caller receives a stream, event, or generic callback.
Any CUDA failure during tail record, join, physical observation, or discard is
immediately terminal-drained across all three streams. The isolated execution
package uses this capability for one layer-0 front-half admission, but the
owner remains scheduling/lifetime infrastructure rather than a whole-model
execution path.

A separate default-off, test-only V4 startup foundation package now enters
through the private `ModelWeights` attachment. It regenerates the canonical
P40 plan internally, authenticates the complete 256-artifact/400-source live
target-AOT catalog, preserves each payload, manifest, upload/readback receipt,
and tensor-scale identity, and mints V4-local Gate+Up, Down, and BF16 A/B
resource seals. It validates and binds the 96 live BF16 projection tensors as
48 canonical natural-layer A/B pairs and all 128 live BF16 outer-norm vectors
as 64 input/post-Attention pairs. Construction-only private seams seal both
catalogs exactly once; no request may query CUDA state or rescan the model or
catalog per layer. The startup package itself remains host-only and grants no
launcher or production-dispatch authority. Its non-owning asset access
requires the eventual Engine composition root to destroy the execution
package first, this startup package second, `ModelWeights` third, and the
complete target-AOT owner/resident root last.

A first default-off, test-only V4 execution package now composes those sealed
catalogs with the request state and event owner. It copies all 256 immutable
projection bindings by value rather than retaining startup-package addresses;
its launch operation is private to the CUDA fixture and future Engine
composition root, so an independently escaped package cannot launch against
expired weight ownership. It owns exactly
442,368,000 bytes of transient ping/pong/scratch storage and 156,893,184 bytes
of dual-epoch recurrent storage. Its only executable operation is a one-shot
GDN layer-0 front half: input normalization on Main records `NormReady`, AbAux
waits and submits the canonical layer-0 BF16 A/B pair, then records `AbReady`.
Because QKVZ is not yet bound, Main deliberately does not wait on `AbReady`;
the package instead joins both stream tails on Control, physically observes
`OwnerDrained`, and discards the unpublished request state. This proves one
real CUDA composition seam and teardown order, not a complete layer, panel,
model, selector, API route, real-checkpoint result, or performance result.

Two independent V4 NVFP4 constituents also exist. Gate+Up binds
M8000/K5120/N17408 to M64N128K64, 256 threads, two stages, and 32 persistent
CTAs; its production kernel compiles at 128 registers/thread with zero spill
and admits two CTAs/SM. Down binds M8000/K17408/N5120 to M64N128K64, 256
threads, two stages, and an ordinary 125×40 grid; its production kernel
compiles at 102 registers/thread with zero spill and admits two CTAs/SM. The
shared canonical NVFP4 decoder now has a direct asymmetric oracle covering all
65,536 packed four-nibble words at scale 1 plus every E4M3 scale code on
non-uniform words; the Gate/Up and Down tile oracles cover both canonical
N128 halves and M64/M37. These are synthetic correctness/resource admissions,
not real-model accuracy or performance results.

Three independent default-off V4 FP8 roles now share one C8000
M64/N128/K64, three-stage `cp.async.cg` admission body while retaining their
distinct authenticated payload partitions and tensor scales. GDN-QKVZ writes
QKV/Z directly into the phase-aliased scratch row. Full-QKV scatters the
projection-native 6,144 Q and 6,144 Gate values directly into 24
`[Q256, Gate256]` head slots and publishes K/V to private NHD caches.
Attention-output gathers its logical K=6,144 input directly from those
interleaved Q slots, without reading or rewriting Gate or the row gap. The
three instantiated kernels compile at 89/92/87 registers/thread, 49,152 bytes
of dynamic shared memory, zero local/spill, and two CTA/SM capacity. M64/M37,
both canonical N128 halves, all role partitions, interleaved-Q/G sentinel,
live-resource, device/allocation-range, undersized-range, and authority
negative tests pass independently. These are sparse synthetic T1 admissions;
no full-K real-checkpoint numerical qualification, private startup binding,
completion event, or production authority exists.

An independent default-off V4 BF16 A/B admission now reuses the established
two-stage `M64N96K64` tensor-core body for one fixed C8000 panel. One 125-CTA
grid reads the normalized C8000 hidden plane once for both 48-row projections
and writes directly to scratch columns A=`[16384,16432)` and
B=`[16432,16480)` with row stride 17,408; it has no compact bridge, tail,
selector, fallback, JIT, repack, or autotune path. The production body compiles
at 88 registers/thread with 46,080 bytes of dynamic shared memory, zero local
bytes/spill, and two CTA/SM capacity. A patterned M64 oracle is bitwise equal
between compact and direct-strided publication for all A/B elements while
every other scratch slot remains unchanged, and the complete C8000 entry has
executed once with its enqueue-only receipt. The old and V4 default-off ABIs
are also symbol-isolated in both build directions. The startup package binds
all 48 real layer pairs, and the isolated execution package now consumes
ordinal zero on the private AB stream between `NormReady` and `AbReady`. The
complete C8000 device test uses live CUDA allocations but zero-valued
synthetic weights; it is execution/lifetime evidence, not a real-checkpoint
numerical result or a runnable complete layer or request path.

A separate default-off V4 norm/residual admission now closes the fixed
two-hidden-plane arithmetic seam without adding a copy or third activation
plane. Input centered RMSNorm reuses the established exact body at
`grid=8000, block=256`; branch residual plus post-Attention centered RMSNorm
reuses the established fused body at `grid=8000, block=512`, publishing the
BF16 residual in place to the right plane and the normalized result in place
to the left plane. The bodies compile at 22/30 registers per thread with
1,024/11,264 bytes of static shared memory, zero local bytes, and 6/3 CTA/SM
capacity respectively. Independent C1 and special-value C65 executions are
bitwise equal to the established norm and residual-add-then-norm bodies, and
both complete C8000 entry points execute with enqueue-only receipts. All 64
real input/post-Attention norm pairs are now construction-sealed, and layer-0
input norm is connected to Main and `NormReady` in the isolated execution
package. Residual/post-norm and the remaining 63 layers are still unbound;
there is no full-layer numerical qualification or production dispatch.

A separate default-off V4 Full-Attention preprocess admission now changes Q
and K in place for one C8000 panel while leaving the projection-native Gate
slots, scratch-row gap, V cache, and every K row outside the active panel
bitwise unchanged. One 128-thread CTA owns one token/head and reproduces the
established centered RMSNorm addition tree, BF16 boundary, and partial-D64
NeoX RoPE order. Q remains in the 24 `[Q256, Gate256]` slots; K is addressed
from the complete private `[40000,8,128]` NHD allocation at panel positions
0/8000/16000/24000/32000. The admitted kernel compiles at 19
registers/thread, 516 bytes static shared memory, zero local/spill, and 12
CTA/SM capacity. C1/C65 bit oracles, nonzero-position, multi-head, sentinel,
allocation-range, resource, and authority negatives pass. This is still a
startup-unbound T1 admission: the fixed Attention body now exists separately,
but the per-layer readiness event, private composition, real-checkpoint
qualification, and production route are absent.

An independent default-off V4 exact-GDN admission constituent now covers one
contiguous C8000 panel without a whole recurrent-epoch copy. It copies only
the 61,440-byte convolution history into the candidate bank, runs one
in-place causal-convolution/SiLU kernel, then one exact ordered recurrence +
RMSNorm + SiLU-gate kernel. Active recurrent state remains const, every token
consumes the preceding BF16-rounded state, same-token output consumes the
pre-round FP32 update, and publication targets only the discardable candidate
bank. Independent C1/C65 bit oracles, pre-cancellation, resource tampering,
current-device/resource re-observation, and complete CUDA-allocation-range
negative tests pass. The kernels compile at 27 and 80 registers/thread with
4 and 34,316 bytes of static shared memory respectively and zero local/spill.
This public seam is caller-live-stream, startup-package-unbound T1 admission
only; it does not prove C8000 numerical qualification, bind a private stream,
commit a recurrent epoch, or authorize production dispatch.

V4 has no selector, whole-model launcher, complete startup/event-bound
BF16-A/B/FP8/GDN/Attention/norm/residual layer executor, real-checkpoint
oracle, API witness, performance result, numerical qualification, release
authority, or production eligibility. Its first authenticated device package
is intentionally limited to the discarded layer-0 norm-to-A/B slice. The v10
observation at 392.804397 prompt tok/s therefore remains the current
whole-product incumbent.

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
| SM87 whole-system AOT Prefill v1 | Default-off real-P40 API composition; performance-rejected after a zero-byte 840.000399-second timeout | Retain only as correctness/diagnostic control; it is not an active performance candidate |
| SM87 bulk-dataflow v2 Prefill | Complete default-off real-P40 API route; performance-rejected after a zero-byte 680.73-second EvalScope timeout and one bounded causal profile; accuracy remains unqualified | Retain exact constituents and evidence only; no V2 tuning, P60/P130, qualification, or production promotion |
| SM87 MacroFeed v3 Prefill | Complete default-off, test-only 64-layer source composition with startup-bound target-AOT assets, role-specific macro projections, nine-kernel-per-layer exact GDN, cold rollback, and a V18 physical transaction; integrated build and focused admission tests pass | Frozen executable diagnostic/control; no authoritative P40 timing, numerical, release, or production qualification exists |
| SM87 MacroFeed v4 Prefill | Active, default-off C8000×5 panel-major foundation with one isolated executable layer-0 norm→BF16-A/B slice; host workspace/request-state contracts, authenticated 256-artifact plus 48-real-BF16-pair and 64-real-norm-pair startup seals, private three-stream/nine-event owner, Gate+Up, Down, fixed direct-scratch BF16 A/B, four layout-specific FP8 admissions, exact two-plane norm/residual, exact in-place Q/K preprocess, fixed unsplit Q128/KV32 C8000 online Attention, and independent exact-GDN admission | Anchor the first slice under Engine lifetime, bind QKVZ and close the intended A/B overlap, then compose the remaining exact layer/request bodies, finalizer, rollback, and whole-model physical receipts before real-checkpoint/state/API qualification; no selector or performance authority exists |
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

The rejected V2 profile is a different, censored request window and must not
be numerically merged with the completed v10 request above:

| V2 family | Instances | Aggregate kernel time | Profile share |
| --- | ---: | ---: | ---: |
| Whole-role NVFP4 Gate+Up | 8 | 86.781391584 s | 58.8998% |
| Whole-role NVFP4 Down | 7 | 37.384968352 s | 25.3737% |
| Attention | 8 | 11.114187328 s | 7.5434% |
| FP8 projections | 16 | 4.507008224 s | 3.0590% |
| GDN prepare/recurrence/epilogue/cancellation sample | 15,000 | 6.833257472 s | 4.6378% |
| Other | 41 | 0.716475456 s | 0.4863% |

Gate+Down is 84.2735% of the captured aggregate kernel time; Gate+Down+FP8 is
87.3325%. These are attribution shares from overlapping streams, not elapsed
TTFT shares and not a throughput measurement.

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
| Bulk-dataflow V2 | No valid result | Rejected after zero-byte 680.73-second timeout; one bounded causal profile only |

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
  lossless Factorized-R1 Prefill, vLLM parity, a valid Bulk V2 Prefill timing,
  or a fully qualified 10-token/s Decode release.
- **Target:** the accuracy-preserving, non-MTP, OpenAI-compatible runner and
  performance region locked by the Constitution.

Before any performance run or profiler capture, the clean-host preflight must
pass using `tegrastats`, CPU/process inspection, and GPU-device-handle
ownership. Jetson `nvidia-smi` is not an idle or attribution authority.
