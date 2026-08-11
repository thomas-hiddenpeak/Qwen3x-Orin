---
q3x_document:
  id: q3x-prefill-architecture
  class: active
  status: active
  owner: prefill-maintainers
  authority: Prefill subsystem boundary, state contract, and architecture-candidate requirements
  effective: 2026-08-10
  last_reviewed: 2026-08-11
  supersedes: [docs/PREFILL_ARCHITECTURE_RESET_LEGACY.md, docs/PREFILL_REFERENCE_AUDIT.md]
  superseded_by: []
  ssot_for: Prefill inputs, outputs, ownership, synchronization, failure, and Decode handoff
  review_trigger: Prefill boundary, state ABI, execution-plan interface, or handoff change
---

# Prefill subsystem design

## 1. Authority and parent boundary

This subsystem SDD refines the Prefill boundary in
[`SDD.md`](SDD.md). It defines what Prefill must consume, produce, own,
synchronize, report, and commit so that the external runner can satisfy its
locked product contract. It does not define the project's delivery order,
record current implementation state, retain performance history, or prescribe
a kernel mechanism.

[`ROADMAP.md`](ROADMAP.md) is the only active delivery order.
[`CURRENT_STATUS.md`](CURRENT_STATUS.md) is the only current implementation
and qualification snapshot. Evidence and candidate selection follow
[`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md), while
the external witness protocol is owned by
[`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md).

The following product constraints leak into every Prefill design:

- cold, no-cache 40K and 60K real-Agent prompts must reach the first visible
  generated token within two seconds;
- the pinned approximately 130K prompt must do so within four seconds;
- the complete prompt must be consumed without silent truncation, Prefix/KV
  reuse, or an approximate numerical route;
- production accuracy and the declared Prefill-to-Decode state semantics must
  not regress;
- MTP is excluded from the current target path; and
- cuBLASLt is a reference only and has no production dependency, dispatch, or
  fallback eligibility.

These are system constraints, not local performance cells. A subsystem or
mechanism result has value only when it composes into a named architecture
candidate and returns through the real API path.

## 2. Prefill boundary

Prefill begins after protocol validation, exact tokenization or token-ID
validation, capacity admission, queue release, and request-resource binding.
It ends only when the complete prompt state and next-token inputs have been
atomically published at `PrefillStateCommitted` for Decode.

Queueing, tokenization, admission, first-token Decode, and response
publication are outside the pure Prefill interval even though they remain
part of user-visible TTFT. The runner must report those intervals separately;
it may not infer pure Prefill from TTFT by assumption.

### 2.1 Inputs

An admitted Prefill invocation receives:

- the exact prompt token IDs, their authenticated tokenizer/model identity,
  and their declared position range;
- the request's cache policy, context/output capacity, cancellation state,
  and preallocated memory-plan binding;
- the authenticated model weights, derived layouts, tactic identities, and
  supported context bucket from the release `DeploymentPlan`;
- an empty cold-request state for the locked witnesses, or an explicitly
  declared state only for a separately specified workload; and
- CUDA streams, events, arenas, and route identifiers allocated before the
  hot path begins.

The production Prefill path may not discover a new tactic, repack weights,
grow workspace, compile code, or silently change numerical mode during a
request.

### 2.2 Outputs

A successful Prefill invocation publishes one complete handoff containing:

- full-Attention KV state for every consumed prompt position in the declared
  production layout;
- GDN/SSM recurrent and convolution state at the exact prompt boundary;
- position and RoPE state needed by the first Decode step;
- next-token logits or the exact final representation from which the planned
  Decode entry computes them;
- completion events that make every handoff component visible to its declared
  Decode consumer; and
- route, interval, synchronization, fallback, resource, and prompt-consumption
  evidence for the request record.

No partially published state is a valid output. The externally visible first
token may be produced only after the complete handoff is committed.

## 3. State, lifetime, and ownership

Ownership is explicit at every lifetime:

| State or asset | Owner | Lifetime | Publication rule |
| --- | --- | --- | --- |
| Checkpoint weights and authenticated derived layouts | Engine / `DeploymentPlan` | Engine-wide | Read-only after readiness; no request-time replacement |
| Prompt token IDs and request metadata | Request controller | Request-wide | Immutable after admission |
| Full-Attention KV | Request state | Request-wide through Decode | Written by Prefill, transferred only at the handoff event |
| GDN/SSM recurrent and convolution state | Request state | Request-wide through Decode | Exact declared dtype, layout, and rounding at handoff |
| Position and RoPE state | Request state | Request-wide through Decode | Advances exactly by the admitted prompt length |
| Layer/span activations and scratch | Prefill execution plan | Bounded producer-consumer interval | Reused only after the declared consumer event |
| Streams and events | Runner execution plan | Engine- or request-wide as declared | Dependencies are plan entries, not implicit global barriers |
| Next-token representation/logits | Prefill/Decode boundary | Until first Decode consumption | Published atomically with the remaining handoff state |

Every buffer has one producer, one or more named consumers, a maximum size,
and a reuse event. A double-, triple-, or deeper buffer is allowed only when
the execution plan identifies a real producer/consumer overlap and the full
capacity profile retains required memory headroom. Buffer count is therefore
an architecture decision derived from dependencies, not a global mechanism
rule.

## 4. Synchronization and execution semantics

The Prefill plan must preserve model order and recurrent dependencies while
exposing independent work across prompt spans, operator families, layers, or
streams where the exact state contract permits it.

- Stream and event dependencies are declared in the `DeploymentPlan` and
  request execution plan.
- An unconditional device synchronization is not a normal layer, tile, or
  span boundary. If one remains, the architecture candidate must identify the
  correctness dependency it protects and its user-visible budget.
- Scratch may be recycled only after its last planned consumer completes.
- Cancellation is observed at bounded safe points without committing a
  partial Prefill state.
- The final state-commit event is the only transition that authorizes Decode
  to consume Prefill-owned state.

The plan may choose layer-major, span-major, pipelined, fused, or another
execution shape. This SDD does not select one in advance. The selected design
must explain prompt-wide reuse, state progression, memory capacity, and every
barrier for all locked context buckets.

## 5. Prefill-to-Decode handoff

`PrefillStateCommitted` is a versioned ABI, not an informal call boundary. Its
identity binds:

- the exact number and IDs of consumed prompt tokens;
- KV length, dtype, layout, addressing, and visibility;
- recurrent and convolution state dtype, layout, rounding, and position;
- position/RoPE state and the first Decode token index;
- next-token logits or the planned representation and owning producer;
- producing and consuming stream/event identities; and
- model, binary, release plan, numerical route, and state-schema versions.

Decode may not infer missing fields or depend on a test-only Prefill layout.
Prefill may not change this ABI to obtain a local speed result without an
atomic update to the parent SDD, Decode contract, implementation, oracles, and
release plan. Handoff equivalence is checked before performance selection.

## 6. Numerical route, failure, and cancellation contract

The active production contract is the accuracy-preserving route defined by
the parent SDD and pinned model/runtime contracts. An approximate activation,
recurrent-state, Attention, or handoff representation is not a Prefill
architecture candidate under this contract. Any opt-in numerical research
requires an explicit project-owner amendment and a separately identified work
package, route, evidence set, and non-production artifact; it cannot share an
implicit selector or promotion claim with the production route.

Unsupported context length, insufficient planned memory, unknown tactic,
artifact mismatch, route-coverage gap, forbidden cache mode, or incompatible
handoff schema fails closed before Prefill begins whenever the fact is known.

After Prefill begins:

- a device, numerical-invariant, cancellation, or execution-plan failure
  prevents `PrefillStateCommitted`;
- no partially written KV or recurrent state becomes visible to Decode;
- request-owned resources are released only after their producers are safely
  quiesced;
- the request state machine enters its declared error or cancellation state;
  and
- route identity, last completed safe point, resource state, and error cause
  are recorded without presenting the run as performance evidence.

The design must bound cancellation observation and resource cleanup even for
the longest supported prompt. A design that can only discover failure after
an uninterruptible full-prompt execution does not satisfy the product
boundary.

## 7. Architecture-candidate contract

A Prefill architecture candidate is a complete executable dataflow, not a
kernel list. Before activation it must declare:

1. the originating 40K, 60K, or approximately 130K API symptom and the
   downward budget assigned to Prefill;
2. one route from admitted token IDs through all model layers to the exact
   handoff, including fallback and numerical-mode exclusions;
3. tensor shapes, model order, state transitions, bytes, residency,
   synchronization, and producer/consumer ownership for every dominant
   operator family;
4. the mutually dependent local mechanisms required to realize that route,
   each inside a bounded Roadmap-activated work package;
5. a composition deadline and the target-length API return point;
6. output/state equivalence, resource, capacity, cancellation, and route
   attestation gates; and
7. the cumulative native incumbent and external vLLM starting-line evidence
   used at architecture and release selection.

When the Prefill gap is architectural, candidate design must inspect the
relevant proven vLLM, FlashInfer, Triton, FlashLinearAttention, and Mamba
dataflows before local parameter scanning. Their algorithms, scheduling, and
specialization mechanisms are references; they do not become production
dependencies by inspection.

The architecture candidate returns to the real OpenAI-compatible API and the
40K/60K/approximately-130K witness set as soon as its dependencies close or
its composition budget expires. Component timing and profiler evidence may
explain acceptance or rejection, but cannot select the whole architecture.

## 8. Designed candidate lineage: `AC-PREFILL-LAYERMAJOR-8K-v1`

`AC-PREFILL-LAYERMAJOR-8K-v1` is the first complete design response to the
repeated short-span layer traversal. The tree now contains an **executable
development compatibility route** for its transaction, memory, scheduling,
and API shape. That route still executes each logical C8192 panel as ordered
physical segments of at most C512, and its exact FP8/NVFP4 Marlin inventories
remain development/test admissions. Its final logical and physical panel pair
is balanced, and eligible segments bind sealed native exact C64 GDN, but it is
therefore still not the complete large-M architecture candidate, a selected
performance result, or production. The identifier freezes a lineage for
Roadmap-controlled integration; explicit evaluation of the compatibility
route does not change the default route.

### 8.1 Execution shape and progress

The candidate changes the scheduling skeleton from repeated public C512
Prefill tiles to one whole-request, layer-major model pass. Each model layer
consumes the complete admitted prompt before the next layer begins. Work
inside a layer is divided into operator panels of at most 8,192 prompt tokens
so kernels and scratch remain bounded.

`C8192` is an internal operator-panel capacity, not a public Prefill tile,
request boundary, partial handoff, or permission to truncate a final panel.
For 40K, 60K, and 130K prompts the design therefore replaces approximately
79, 118, and 254 complete 64-layer walks at C512 with one 64-layer walk whose
layers contain approximately 5, 8, and 16 ordered panels. These counts state
the scheduling hypothesis only; they are not a speedup or latency promise.

The execution plan owns two monotonic progress vectors:

- `kv_progress[layer]` records the largest prompt position whose exact KV is
  visible to later Attention panels in that layer; and
- `gdn_progress[layer]` records the largest position whose exact recurrent
  and convolution state has advanced in that layer.

Attention panels may consume only positions made visible by the applicable KV
event. GDN panels advance in token order and preserve the declared recurrent
state dtype, layout, rounding, and convolution history across every panel
boundary. Panel progress is internal, request-owned state: neither vector
authorizes Decode. After every layer has consumed every admitted token, the
plan publishes exactly one final `PrefillStateCommitted` handoff containing
the complete KV, GDN, position/RoPE, and next-token state. Cancellation and
failure before that event expose no partial handoff.

The current tree implements the immutable 64-layer topology (48
linear-Attention and 16 full-Attention layers), at most 32 logical C8192
panels, request-owned progress, and one final host-state commit transition.
For the explicit development route, `ReferenceEngine` provisions the
layer-major request profile, binds a sealed engine-lifetime plan to the exact
model, state, arena, typed views, main/auxiliary streams, completion events,
and operator receipts, then executes true outer-layer/inner-panel traversal.
The runner leaves all recurrent/KV work and final hidden state uncommitted;
the Engine performs logits finalization and the single sequence-length
publication through a move-only one-request receipt. Any failure is drained
and reset by the Engine transaction guard. The default selector remains the
legacy C512 route. The scheduler balances only the final full-plus-tail pair,
so P1025 becomes 512+257+256 rather than 512+512+1 while complete C512 blocks
remain unchanged. No production conclusion follows from executability.

### 8.2 Indivisible composition boundary

The design is complete only when one attested route composes all dominant
families below in natural model order:

| Family | Candidate requirement |
| --- | --- |
| NVFP4 Gate/Up and Down | Shape-specific large-M ownership for both asymmetric roles; neither may inherit a tactic merely because it won on the other shape |
| FP8 QKV, Z/O, and KV projections | Panel-capable, shape-specific exact routes with authenticated weights, scales, layout, and workspace |
| Attention | Exact causal Attention over the complete prompt, including ordered KV publication and consumption across panels |
| GDN/SSM | Exact recurrent and convolution update across panels and exact final boundary state; no approximate state or altered rounding contract |
| Residual, normalization, embedding, logits, and handoff | Consumer-native layouts or explicit planned conversions, with no undeclared synchronization, allocation, or fallback |

A build that implements only a larger projection, Attention, or GDN kernel is
a local mutation, not this candidate. Missing family coverage, a return to the
public C512 loop, approximate arithmetic, MTP, cuBLASLt dispatch, or an
undeclared fallback invalidates the route rather than producing a partial
candidate result. Buffering and overlap remain dependency-derived plan
choices; the `C8192` name does not imply double or triple buffering by itself.

The current tree implements and seals a typed binding contract for exactly 17
roles: NVFP4 Gate/Up and Down; linear-Attention FP8 QKV, Z, and O;
full-Attention FP8 Q, K, V, and O; linear BF16 A and B; exact GDN; exact
causal Attention; residual; normalization; embedding; and final handoff. At
engine creation, the compatibility route fails closed unless every role can
be associated with the exact authenticated weights/sidecars, launch tactic,
workspace, and resource identity. The installed execution inventory remains
capped at physical M512, however, so these receipts authorize the segmented
compatibility route; they do not establish the target C8192 operator-panel
contract. The exact GDN receipt is no longer nominal: the binary capability,
real workspace pointer and byte capacity, 48-layer producer/weight shapes,
and M32--M512 native tactic envelope are sealed and revalidated. Every
eligible segment must report the native disposition; exact fallback is
permitted only for M1--M31.

Candidate-only C8192 NVFP4 Gate/Up and Down surfaces and shape-specific FP8
projection surfaces now exist in the kernel tree. As architecture-candidate
assets they are completely unbound: they remain isolated from the binding
contract's physical identities and are not connected to `RequestState`, the
runner, or any production selector. They are distinct from the later v5 route,
which connects the existing frozen exact-Marlin sidecars and kernel body to
one complete M8192 launch while preserving the exact oracle sequence for every
partial panel. The bespoke candidate-only surfaces remain unbound; the v5
route does not claim to implement them, change the default route, or produce a
production performance result.

### 8.3 Implemented workspace requirements and capacity closure gate

The current tree implements a checked, host-only workspace requirements plan.
It neither reserves device memory nor authenticates model or sidecar
residency. A byte total exists only after the caller explicitly names the
C8192 GDN, legacy GDN, and MLP physical tactics. For C8192 exact C64-native GDN
with in-place convolution, current Release/default legacy exact C16 GDN, and
separate Gate+Up then SiLU, the target-bucket arithmetic is:

| Prompt tokens | Caller-selected conditional profile | Conservative disjoint profile |
| ---: | ---: | ---: |
| 40,000 | 3,975,374,848 bytes | 5,324,963,840 bytes |
| 60,000 | 5,496,014,848 bytes | 7,052,323,840 bytes |
| 130,000 | 10,818,254,848 bytes | 13,098,083,840 bytes |

Here `selected` means only the pair of strategies explicitly supplied to the
host planner: one prompt-wide hidden buffer under an unbound panelwise
in-place contract, plus sequential C8192 family-live-set overlay under
unbound completion and legacy-route-exclusion contracts. It does **not** mean
that the architecture candidate, a tactic, or a production route has been
selected. The conservative profile uses two disjoint prompt-wide hidden
buffers and separates three C8192 operator families plus the legacy C512
workspace. It still relies on each selected tactic's named phase-local layout
and therefore is not an alias-free executable reservation.

The corrected C8192 physical high-water ledger is:

- exact C64-native GDN: 446,365,696 bytes with in-place convolution, or
  456,851,456 bytes with one independent token-parallel C512 convolution
  output; the fixed 75,694,080-byte native workspace is reused serially and is
  never multiplied by physical segment count;
- Full Attention preprocess/core: 402,653,184 bytes for raw Q+gate, processed
  Q, and packed gate; K/V write directly to persistent cache;
- MLP: 855,638,016 bytes for separate Gate+Up then SiLU, or 940,572,928 bytes
  for the fused Gate+Up epilogue; and
- shared Marlin FP32 reduction plus locks: 1,048,832 bytes, placed only through
  an authenticated dead-span/phase layout.

The request total separately owns one stable 10,240-byte final-hidden handoff.
The native GDN layout and every CUDA partition offset now share one constexpr
host ABI, while the Marlin reduction and lock sizes come directly from the
public NVFP4/FP8 kernel ABIs. One 32,768-byte panel token-ID view may occupy the
operator-arena prefix only until embedding gather completes; its reuse is an
explicit event-gated alias rather than additional capacity.

Consequently separate-SiLU MLP still dominates the selected overlay. Choosing
token-parallel C64 convolution changes the conservative 40K/60K/130K rows to
5,335,449,600 / 7,062,809,600 / 13,108,569,600 bytes. Choosing a disjoint
test-only C64-native legacy route adds 75,694,080 bytes; the current Release
C16 legacy route has no such external workspace. C64-native and the large-M
projection admissions remain test-only today, so these are explicit candidate
requirements rather than deployed production facts.

The first `RequestState` allocation strategy is now separately expressible:
the three C8192 families share their sequential high-water, while the complete
legacy C512 workspace is physically disjoint. Its exact selected totals are
4,066,344,960 / 5,588,904,960 / 10,917,864,960 bytes at 40K/60K/130K. The
explicit create path allocates this exact profile and exposes typed GDN,
Attention, MLP, legacy, token-staging, residual, and final-hidden views. In
whole-request mode the Engine now selects it at startup, binds the compatibility
executor and two completion events, and fails creation rather than falling
back when the complete route is unavailable. It remains unreachable from the
legacy default request-state entry point.

Both profiles for all three buckets fit the planner's declared
17,437,720,576-byte request-arena limit. This is only a request-arena verdict.
The planner intentionally leaves resident-model bytes, derived-sidecar bytes,
and total whole-process required bytes absent, and reports whole-process
capacity as `kIndeterminate`. Runtime binding makes the segmented compatibility
path executable, but does not turn these request-only numbers into evidence
that the model plus target-length candidate fits the device or that a
production capacity profile exists; allocator fragmentation, runtime metadata,
and measured whole-process peak memory remain open.

The next capacity gate must exercise 40K/60K/130K reservations with measured
whole-process peaks. The release gate must then bind one authenticated AOT
layout/sidecar ownership model, native large-M panel tactics, all workspaces,
and the exact installed binary. Startup and admission fail closed when the
selected bucket cannot reserve the complete plan before Prefill begins; the
request path may not grow it.

### 8.4 Implemented integration seam and v1 closure

A default-off whole-request Engine transaction now supplies the complete
controller seam: it preallocates the transcript before touching GPU state,
executes the sealed layer-major plan, validates complete uncommitted progress,
invokes the retained-hidden logits finalizer, and calls one no-throw final
commit. A move-only receipt prevents reuse, and an RAII rollback guard resets
the runner on every failure or cancellation window. The API route adds two
engine-lifetime disable-timing events and treats one `layer x logical panel`
as a submission quantum. It submits at most two quanta, retires the oldest
before polling cancellation, checks again after final normalization, before
logits, and before commit, and never publishes a partial sequence length.

`qwen3x-eval-server` exposes this route only through explicit
`--prefill-execution-mode layer-major`. It propagates API disconnect/shutdown
state into the bounded probe and emits a v2 success witness containing the
actual mode, panel count, request profile, window/retirement facts, and sealed
plan identifier. The sealed balanced route identifies itself as
`q3x.sm87.exact.layer-major-c8192.balanced-segments.v2`. Legacy remains the
default and retains the v1 witness. Default, dual-Marlin, full-inventory, and
GDN-disabled control builds pass their applicable host/API/type tests.

A clean-host P1025, one-output-token OpenAI API/EvalScope direction screen on
commit `18363ad` is cumulatively positive against its older greedy/fallback
layer-major witness, returns the same exact token, and reports every required
operator role on its admitted production disposition with zero forbidden route
hits. It remains slower than the older legacy observation. The comparison is
not causal: the earlier witness used exact FP8, Attention, and GDN fallbacks,
so it retains the complete sealed composition direction only. The short screen
cannot select this architecture lineage or activate production; current
numbers belong to [`CURRENT_STATUS.md`](CURRENT_STATUS.md), and exact evidence
belongs to the
[machine-readable direction record](metadata/qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json).

The private `enqueue_prefill_layer_panel(...)` executor now binds three
engine-lifetime projection tactics. The exact tactic preserves the complete
C512 oracle sequence. The rejected segmented wrapper groups one logical panel
but retains bounded internal Marlin segmentation. The default-off v5 tactic
uses the same authenticated exact-Marlin sidecars, typed reduction arenas, and
locks to issue exactly one launch per logical FP8/NVFP4 projection only when
the panel is the complete M8192 shape. Every partial panel retains the entire
oracle span ledger and legacy MLP workspace, including per-span
Gate+Up-to-SiLU-to-Down-to-residual ordering. Projection tactics do not own or
change logical panel geometry; the existing balanced topology and plan
identity remain immutable.

Exact GDN/SSM, causal Attention, linear BF16 A/B, and the one final state
commit remain in the same complete linear/full-Attention API executable. At
`21d5c28`, clean P513, P8192, and P7712 real-model state screens passed for the
v5 tactic. The following cold/no-cache P40K real-API gate consumed all 40,000
tokens as `3x8192 + 2x7712`, but reached 670.53071 s EvalScope TTFT and
670.486890 s server pure Prefill, or 59.658139 prompt tok/s. Pure Prefill was
99.994017% of server TTFT and external overhead was only 3.702638 ms. The route
recorded 1,008 bulk launches, 672 partial-oracle hits, 13,104 physical
projection launches, and no Prefix, MTP, cuBLASLt, external, approximate, or
forbidden fallback. It was about 0.53% slower than the prior same-payload
exact-Attention observation.

This closes the M8192-only v1 composition as a negative system result. P60K
and P130K were not run, the route remains default-off, and production is
unchanged. The existing same-payload T4 profile is sufficient to justify
architecture redesign; duplicating its approximately eleven-minute trace
would not change the product decision. Exact artifacts and hashes are frozen
in the
[v5 P40K API record](metadata/qwen36-27b-prefill-p40k-native-large-m-exact-api-2026-08-09.json).

### 8.5 Designed successor: `AC-PREFILL-PROMPT-WIDE-v2`

The successor preserves the whole-request Engine transaction, exact model
weights, model-declared dtypes, FP32 accumulation where required, exact causal
masking, exact recurrent-state contract, cancellation semantics, and one final
state commit. It replaces the v1 physical lowering rather than adding more
parameters to it. A different parallel reduction order is a candidate
implementation detail, not permission to weaken accuracy: every such path
remains default-off and accuracy-unqualified until the complete no-regression
gate passes.

The required dataflow is:

| Family | v1 structural failure | v2 contract |
| --- | --- | --- |
| Exact full Attention | Each logical panel is lowered to repeated bounded spans; generic QT2 work repeatedly scans the causal K/V history | One M8192 or M7712 logical-panel launch graph using tiled Q, streamed K/V, FP32 online-softmax state, exact causal masking, and ordered KV publication; FlashInfer/FlashAttention is the reference dataflow, not an approximate grouped-Q64 substitute |
| NVFP4 projections | A single host launch still enters an M64-oriented Marlin body; partial M7712 panels retain 14x512+2x272 lowering | Separate merged Gate/Up (`K=5120,N=34816`) and Down (`K=17408,N=5120`) tactics, both covering M8192 and M7712, with cross-row weight/scale reuse and staged load/decode/MMA overlap |
| FP8 projections | The same segmented/panel-wrapper limitation applies across heterogeneous QKV/Z/O shapes | Shape-specific QKV, Z, and O tactics with authenticated sidecars and consumer-native layouts; no universal tile is presumed optimal |
| BF16 A/B | Recursive M16 pair dispatch creates 120,000 launches in the measured P40K route | One panel-wide exact tactic per role, retaining declared output/state boundaries |
| GDN/SSM | One logical panel is repeatedly submitted as C512 work and each span expands into multiple kernels | Submit the panel's C64 hierarchy as one work graph, parallelize chunk-local KKT/WY work, serialize only the mathematical boundary-state dependency, and write the final boundary state once; FLA and Mamba selective scan are design references |
| Synchronization | One CUDA stream and zero observed kernel overlap; the two host slots are cancellation windows, not a device pipeline | Events follow real producer/consumer dependencies; buffering is introduced only for a named overlap with measured critical-path effect |

Implementation began with exact logical-panel Attention because it owned the
largest observed interval and could return to the P40K product witness without
waiting for the remaining families. Prompt-wide GDN/BF16 and true
shape-specific projection packages follow under the composition deadline in
[`ROADMAP.md`](ROADMAP.md). Component timing and NSight evidence diagnose each
package; only the same cold/no-cache real API selects it. A competitive,
accuracy-admissible P40K result alone unlocks P60K and approximately 130K.

### 8.6 First v2 implementation slice and result

Revision `c45b7c5` implements `native-flashinfer-exact-panel` as a distinct,
default-off Attention tactic with a v6 witness and no compatibility fallback
for production-sized logical panels. It submits the complete M8192 or M7712
query panel against its exact causal KV extent through the authenticated
FlashInfer backend, uses FP32 online-softmax state, and preserves the model's
existing RoPE/KV publication boundaries. It is explicitly admitted only in a
development build; backend capability is queried without side effects and the
Engine plan fails closed if the route is unavailable.

The clean real-model P513 direction screen hit the new kernel 16 times and no
QT2/Q64/Q128 route. It preserved generated token `9419`/`Hello`, but its full
state hash differed from the oracle. The following clean-host cold/no-cache
P40K OpenAI API/EvalScope screen consumed all 40,000 tokens as
`3x8192 + 2x7712`, recorded 80 FlashInfer logical-panel hits and zero generic
Attention hits, and reached 109.02622 s external TTFT / 108.981855 s server
pure Prefill / 367.033577 prompt tok/s. This is 6.15x faster than the
preceding exact v5 route and 1.65x faster than the grouped-Q64 direction. API
boundary overhead remained 3.874726 ms.

The quantity change retains the logical-panel Attention architecture, not its
production arithmetic. P40K generated `The`, matching the preceding exact
P40K route, but that one-token match does not qualify a different reduction
order. Together with the P513 state mismatch and unexecuted accuracy gate,
this keeps the tactic accuracy-unqualified and default-off. P60K and P130K
were not run. Exact artifacts and hashes are frozen in the
[v6 P40K API record](metadata/qwen36-27b-prefill-p40k-flashinfer-exact-panel-api-2026-08-09.json).

### 8.7 Coupled NVFP4 C1 and G2/D2 v1 results

Revision `da2b9f6` implements WP-V2-C1-v1 as a distinct, default-off
`native-nvfp4-true-large-m-operator-panel` tactic. It reuses the authenticated
Marlin packed-weight and scale sidecars, decodes E2M1/E4M3 in registers, and
uses BF16 MMA with FP32 accumulation. The v1 launch surface is M128N256K64,
256 threads, three pipeline stages, and 82,944 bytes of dynamic shared memory.
Gate+Up uses 203 registers per thread and Down uses 220, so both reach only one
active CTA per SM. This describes implemented v1; it does not describe the
successor below.

The clean-host cold/no-cache P40K OpenAI API/EvalScope screen consumed all
40,000 tokens with the retained exact FlashInfer logical-panel Attention path.
It reached 136.97409 s external TTFT and 136.929918 s server pure Prefill, or
292.120 prompt tok/s. The retained `c45b7c5` path was 108.981855 s and
367.034 prompt tok/s, so v1 increased pure-Prefill latency by 25.64%. External
TTFT exceeded server TTFT by only 4.047 ms and pure Prefill was 99.97% of TTFT;
the API is not the active bottleneck.

The same-binary, same-checkpoint, same-prompt matched NSys pair attributes
99.30% of the whole-request interval increase to NVFP4. Gate+Up moved from
35.039284 s and 2,496 launches to 52.600776 s and 320 launches (1.501x). Down
moved from 17.051106 s and 2,496 launches to 27.094866 s and 320 launches
(1.589x). Launch reduction did not overcome the absence of persistent
cross-tile ownership, duplicate B/scale decode across M warp rows, two full-CTA
barriers per K64 step, and the accumulator plus decoded-B live range that
forces one-CTA/SM residency.

WP-V2-C1-v1 is rejected. It remains disabled, the default route is unchanged,
and the full accuracy harness, P60K, P130K, and NCU were not run. Exact route
counters, artifacts, timings, resource observations, and the decision are
frozen in the
[v1 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-true-large-m-rejection-2026-08-10.json).

The replacement WP-V2-C1-G2/DownD2 v1 pair is now implemented as the distinct,
default-off `native-nvfp4-g2-d2-large-m-operator-panel` route. G2 owns merged
Gate+Up (`K=5120,N=34816`) as an M128 paired Gate64+Up64 K64 raster and
publishes fused `SiLU(gate) * up` BF16 output. D2 owns Down
(`K=17408,N=5120`) as an M128N128K64 N-major/B-stationary raster and performs
the residual addition exactly once in its epilogue. Both reuse authenticated
Marlin sidecars, accumulate in FP32, use 256 threads and 41,984 bytes of
dynamic shared memory, and compile at 127/126 registers per thread under the
static two-active-CTA/SM resource contract. They admit M8192 and M7712 only.

The conservative `RequestState` MLP tactic continues to describe a three-span
workspace reservation, not the candidate's physical execution decomposition.
The v8 `DeploymentPlan` and request witness are the execution authority: they
identify fused Gate+Up+SiLU and fused Down+residual explicitly and validate
their 320+320 coupled hits. Because this route is rejected and default-off,
the planner is not extended with a new layout enum merely to rename dead-path
reservation storage.

Before the valid timing run, the original G2 output binding was found to alias
the plan's normalized input through `mlp.activated_bf16`. That would create a
cross-CTA in-place race and correctly failed launch validation. The repaired
route uses the exact-size dead `mlp.gate_bf16` span as G2 output/D2 input and
validates non-aliasing. Failed pre-repair attempts carry no performance
authority.

The following clean-host cold/no-cache P40K OpenAI API/EvalScope direction
consumed all 40,000 tokens as `3x8192 + 2x7712`. Its v8 witness recorded 320
GateUpG2 hits, 320 DownD2 hits, 80 exact FlashInfer Attention hits, and zero
Prefix-cache, MTP, cuBLASLt, external, approximate, exact-fallback, or
forbidden-route hits. EvalScope TTFT was 115.08576 s; server pure Prefill was
115.041751913 s, or 347.699851 tok/s. The retained `c45b7c5` route was
108.981854892 s and 367.033577 tok/s, so G2/D2 v1 increased latency by
5.560464% and reduced throughput by 5.267563%. The matching first token `The`
is an API/output smoke observation only, not numerical or capability
qualification.

A bounded diagnostic NSys capture measured G2 at 40.371860160 s and D2 at
20.532324128 s, 60.904184288 s combined. The retained native Marlin main
signature plus standalone SiLU consumed 53.864024544 s, leaving a
7.040159744-s role gap; all other kernels recovered 0.929761664 s, so complete
GPU kernel time still increased by 6.110398080 s. Resource admission, launch
fusion, and epilogue fusion did not provide persistent B/scale residency or
decoded-fragment reuse across CTA output tiles.

WP-V2-C1-G2/D2 v1 is therefore rejected and remains default-off. Full accuracy
qualification, repetition, NCU, P60K, and P130K were not run. P60's balanced
geometry is `6x8192 + 2x5424`; the candidate admits no M5424 path, so the
missing tail is an admission/geometry blocker rather than a P60 performance
conclusion. Exact evidence is frozen in the
[G2/D2 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-g2-d2-rejection-2026-08-10.json).

The Roadmap successor must replace one-raster-CTA ownership rather than scan
parameters on this closed version. Its complete Gate/Up and Down dataflow must
make B/scale residency and decoded-fragment reuse persist across multiple M
rows or output tiles and retain shape-specific ownership. Freeze an M5424 tail
extension in the dispatch contract, implement M8192/M7712 first, and return
directly to the same real P40K API witness. Only a competitive,
accuracy-admissible P40K result unlocks M5424 implementation and P60. This
section records the boundary exposed by v1; it does not claim that persistent
successor exists in code.

### 8.8 Exact-P40000 whole-core direction and replacement boundary

The working tree above `a4f95ba` implements the next complete schedule as five
equal M8000 fill panels, one whole-prompt core, five M8000 drain panels, and
one layer-wide MLP phase per layer. Its v10 witness fails closed unless one
64-layer pass retires exactly 768 bounded submissions and records the complete
FP8, BF16 A/B, GDN, Attention, Gate/Up, and Down role counts with zero
forbidden routes.

The clean-host OpenAI API/EvalScope P40K direction consumed all 40,000 tokens.
Server pure Prefill was 101.831853876 s / 392.804397 tok/s, compared with
108.981854892 s / 367.033577 tok/s for retained `c45b7c5`: a 7.02138%
throughput improvement. The exact transaction/memory/witness substrate is
therefore retained as positive architecture infrastructure. It remains
default-off and accuracy-unqualified because the inherited FlashInfer
arithmetic has a known P513 full-state mismatch.

A bounded NSys capture contains 102.113313600 s of kernels inside a
102.121306528-s whole-request interval. The 7.992928-ms gap is only 0.0078%,
so neither the OpenAI adapter nor launch gaps explain the remaining order-of-
magnitude deficit. Gate/Up, FP8, Down, and Attention contribute 37.273068224
s, 25.864646560 s, 17.559457280 s, and 13.634170272 s respectively.

The persistent NVFP4 implementation is not the required projection
architecture. It launches 16 CTAs and reuses the old M64N256K64 Marlin body,
changing task order without retaining decoded B or scales across CTA output
tiles. The first successor was required to start from the two-CTA G2/D2 feed
discipline, use a real multi-stage producer/consumer pipeline, and select L2
rastering separately for the asymmetric Gate/Up and Down shapes. Humming and
Triton were reference mechanisms for pipeline and raster design; their code
and any SM90+ mechanism were not production dependencies. Exact evidence for
the retained substrate is frozen in the
[whole-core direction record](metadata/qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json).

### 8.9 Shape-wide v3 closure and new reset boundary

WP-V2-C1-v3 implemented that first successor as separate Gate/Up and Down
rasters with three-stage `cp.async`, 126 registers/thread, 62,976 bytes of
dynamic shared memory, and two active CTAs/SM. Gate/Up used group-M=2 L2
ordering; Down used group-M=1 A-major ordering. This is L2 locality, not
cross-CTA shared-memory residency.

The clean-host OpenAI API/EvalScope P40K direction regressed pure Prefill from
101.831853876 s / 392.804397 tok/s to 106.374300578 s / 376.030675 tok/s.
The single permitted causal profile found 106.749233344 s of kernels inside a
106.752632896-s request. Gate/Up increased from 37.273068224 s to
41.163382144 s (+10.437332%); Down increased from 17.559457280 s to
17.788101568 s (+1.302115%). Their combined increase was 4.118958208 s /
7.511889%.

This disproves the v3 skeleton as a target architecture on real P40 model
weights. Its temporary runner overlay is removed; the independent default-off
kernel surface is retained only as experiment evidence. No stage, tile,
raster, NCU, accuracy, repetition, P60, or P130 expansion follows. The next
candidate must be derived from the complete dominant execution graph—NVFP4,
FP8, and Attention together—and must again return first to the same real API
P40 gate. Exact evidence is frozen in the
[v3 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-shape-wide-v3-rejection-2026-08-10.json).

### 8.10 Packed-operand projection contract

Large-M quantized projections in a production Prefill candidate must preserve
the packed operand as a first-class execution asset:

- authenticated real-checkpoint tensors are transformed only at build or
  load time into role- and shape-specific consumer layouts;
- NVFP4 or FP8 B operands and their scales remain packed through global and
  shared-memory movement, with decode adjacent to the exact BF16 MMA consumer
  and FP32 accumulation;
- A, packed B, scales, and epilogue state have one declared producer/consumer
  pipeline rather than independent mechanisms whose composition is assumed;
- merged Gate+Up, K-heavy Down, and FP8 QKV/Z/O families may select different
  layouts and schedules, while preserving each tensor's original scale,
  rounding, residual, and activation semantics;
- a request may not materialize a full BF16 copy of B, JIT compile or repack a
  weight, discover a tactic, grow an arena, or enter a fallback route;
- the physical launch and synchronization ledger must agree with the sealed
  deployment plan and runtime witness; and
- rejected implementation mechanisms may remain as default-off correctness
  evidence but cannot remain reachable from the runner.

This contract intentionally does not select an active candidate, work-package
identifier, tile, stage count, grid size, performance budget, or delivery
order. Those decisions belong only in [`ROADMAP.md`](ROADMAP.md); measured
history and current implementation state belong in the evidence records and
[`CURRENT_STATUS.md`](CURRENT_STATUS.md).

### 8.11 Mathematical-equivalence ledger

Prefill architecture starts from the following functions rather than from the
current kernel boundaries. Let `R_bf16` denote BF16 round-to-nearest-even,
`Acc_fp32^K` the declared K-fragment order and FP32 MMA accumulation, and
`D4(q,s) = R_bf16(E2M1(q) * E4M3FN(s))` the current NVFP4
per-value decode with one block scale for each 16-weight group. For projection
role `r` with authenticated tensor-global scale `a_r`,

```text
P_r(A, Q_r, S_r)[m,n]
  = R_bf16(a_r * Acc_fp32^K(A[m,k] * D4(Q_r[n,k], S_r[n,floor(k/16)])))
```

The exact MLP boundary is therefore:

```text
G = P_gate(A, Q_gate, S_gate)
U = P_up(A,   Q_up,   S_up)
H[m,n] = R_bf16(SiLU(float(G[m,n])) * float(U[m,n]))
D = P_down(H, Q_down, S_down)
R_next[m,n] = R_bf16(float(D[m,n]) + float(R[m,n]))
```

For the fixed P40 workload (`M=40000`, hidden `H=5120`, intermediate
`I=17408`, 64 MLP layers), Gate, Up, and Down each contain exactly
`M*H*I = 3,565,158,400,000` multiply-accumulate pairs. The three roles across
all layers therefore contain `684,510,412,800,000` MACs, or
`1,369,020,825,600,000` operations under the conventional two operations per
MAC accounting, before FP8 projection, Attention, GDN, normalization, or
epilogues. This is a workload invariant, not a ceiling claim. Without a proven
exact sparsity or model-structure equivalence, a candidate must execute this
dense function; its qualitative opportunity is to raise useful MMA issue and
remove redundant decode, movement, materialization, and synchronization around
that irreducible computation.

Gate and Up may be represented as one column-concatenated projection and may
share A movement because each output retains its original K order, independent
scale, BF16 publication, and SiLU/Up consumer. That equivalence removes an A
producer boundary; it does not authorize a joint scale or mixed accumulator.
Likewise Down may fuse its residual consumer only after the Down value has been
scaled and published to BF16. Moving the residual add, tensor-global scale, or
NVFP4 block scale across those boundaries is real-number algebra but not the
declared finite-precision function. In particular, factoring the E4M3FN scale
outside a K16 dot is ineligible because the current function rounds each
decoded weight to BF16 before MMA.

For exact causal Attention, a KV block can be summarized by online-softmax
state `(maximum, denominator, weighted_value)`. Block summaries compose in
real arithmetic by rescaling both denominators and weighted values to the
combined maximum. A production parallel or streamed implementation must also
preserve the causal domain, declared FP32 update order, final normalization,
KV publication, and final state; the real-number monoid alone does not prove
the finite-precision route. This is why an online-softmax implementation with
a different final state remains accuracy-unqualified even if its formula is
mathematically familiar.

For one GDN value row with normalized key `k_t`, value `v_t`, decay `alpha_t`,
and update gate `beta_t`, the unrounded real transition can be written

```text
s_t = s_(t-1) (alpha_t I - alpha_t beta_t k_t k_t^T)
      + beta_t v_t k_t^T
```

and affine transitions can be composed hierarchically. The production state,
however, applies `R_bf16` after every token while the token output consumes the
unrounded updated FP32 row. That rounded map is not the same associative affine
monoid. FLA WY/chunk composition and Mamba selective scan therefore identify
the lower-level mathematical opportunity, but a production implementation
must either reproduce every token's BF16 state boundary exactly or remain a
separate numerical-contract research line. Exact work may still parallelize
Q/K normalization, gate construction, value rows, chunk-local preparation,
and consumer fusion around the serial rounded boundary.

For every candidate, the work-package record must distinguish:

| Transformation | Equivalence obligation | Intended qualitative removal |
| --- | --- | --- |
| shared-input projection composition | independent K reductions, scales, and publication points remain unchanged | duplicate A movement or materialization |
| quantized-weight decode reuse | decoded BF16 word and special-value semantics remain identical | repeated decode and scale traffic within its legal reuse lifetime |
| online causal reduction | causal membership, update order, normalization, and final state pass the exact oracle | repeated history traversal and intermediate score materialization |
| recurrent block composition | every declared rounded state/output boundary is reproduced | token-serial work outside the irreducible state dependency |
| producer-consumer fusion | the pre-existing rounded publication value is still consumed | global round trip, launch, and synchronization boundary |

The ledger is mandatory before profiler-led parameter work. It can show that a
physical mechanism is useful, or that the numerical contract makes a tempting
algebraic shortcut illegal; neither conclusion may be replaced by a tile scan.

### 8.12 Packed NVFP4 v2 closure and mathematical successor boundary

The isolated packed-NVFP4-v2 experiment kept v10 FP8, Attention, GDN, API,
memory, and whole-request control fixed and changed only Gate+Up and Down. Its
valid v14 route completed all 40,000 prompt tokens with zero forbidden or
fallback hits, but pure Prefill regressed from 101.831853876 s /
392.804397 tok/s to 128.493372123 s / 311.300103 tok/s. It recovered
20.393636% latency relative to rejected packed v1, so removing packed-v1 FP8
and changing NVFP4 ownership was material; it did not make the candidate
competitive and the v2 skeleton is closed. Exact route and measurement
evidence is frozen in the
[v14 rejection record](metadata/qwen36-27b-prefill-p40k-packed-nvfp4-v2-rejection-2026-08-11.json).

The physical mapping explains why this is an architecture boundary rather
than another tile cell. At P40, Gate+Up creates 42,568 M128/N256 output-tile
CTAs per layer and Down creates 6,260. Across 64 layers v2 instantiates
3,124,992 CTAs, while the v10 two-role persistent Marlin route instantiates
2,048 physical CTAs that internally traverse the whole MN task space. Before
any cache reuse, v2's logical requests per layer are 55.795 GB of Gate A plus
31.385 GB of Gate/Up packed B and scales, and 27.897 GB of Down A plus
15.692 GB of Down packed B and scales. These are logical request volumes, not
measured DRAM bytes. They show that decoded-B reuse across eight M16 panels
inside one CTA does not establish reuse or residence across the full output
grid; a larger tile, another `cp.async` stage, or another cache operator does
not alter that ownership fact.

There is also a strict **work-package scope bound**. In v10, measured Gate+Up
and Down consume 37.273068224 s and 17.559457280 s. Holding every other v10
component fixed leaves 46.999328372 s even if NVFP4 MLP time is hypothetically
zero, so an NVFP4-only candidate can expose at most about 851.075992 prompt
tok/s on that frozen composition. This is not a hardware or project upper
bound. It proves only that a projection-only scope cannot reach the owner's
4.3K tok/s starting line; a quantity-changing successor must ultimately cover
FP8, Attention, GDN, and their composition as well.

Exact checkpoint sparsity does not provide the missing algebraic shortcut.
An exhaustive CPU inventory of all 192 Gate/Up/Down tensors finds
1,359,701,299 exact signed-or-unsigned E2M1 zero codes among
17,112,760,320 values (7.945540%) and only 151,470,811 consecutive K4 groups
with at least two zero codes among 4,278,190,080 groups (3.540535%). The
per-tensor ranges remain 7.819264%--8.537444% and
3.432756%--4.081497%, respectively. This cannot remove an order of magnitude
of dense work, and skipping zero products would still have to preserve
zero-times-NaN and signed-zero semantics. The sparse lineage is therefore
closed without a GPU implementation. Exact evidence is frozen in the
[real-weight zero-structure audit](metadata/qwen36-27b-nvfp4-mlp-zero-structure-audit-2026-08-11.json).

The next lowest-risk structural hypothesis is reference parity, not another
v2 mutation. The project must reconstruct the actual stock-vLLM W4A16 Marlin
packing, scale handling, K64 load/decode/MMA pipeline, LegacyStripe MN
scheduler, workspace/lock contract, epilogue, and Gate/Up versus Down dispatch
before specializing them. This is not v10: v10 uses project-added
GroupedM4NMajor/BStationaryNMajor full-K persistent schedules and custom fused
epilogues. Stock Marlin's small part-2 tail may split K and globally reduce
partial accumulators; that changes FP32 accumulation order and cannot silently
enter the production numerical contract. A production-eligible parity
candidate must retain stock full-tile behavior while assigning every tail
tile to one full-K CTA, or else keep the original split-K route explicitly
accuracy-unqualified until an independent numerical contract passes.

## 9. Global dataflow questions

Every candidate must answer these questions for the full Prefill route, not
only for a convenient operator or prompt tile:

| Question | Required architectural answer |
| --- | --- |
| Traversal | How are prompt spans and model layers ordered, and what work is repeated as context grows? |
| Residency | Where do weights, derived layouts, activations, KV, recurrent state, and scratch reside over their useful lifetimes? |
| Projection ownership | How do Gate/Up, Down, and FP8 projection shapes receive distinct ownership without forcing one universal tactic? |
| Attention | How are exact causal state, KV publication, and sequence-parallel work represented for the complete prompt? |
| GDN/SSM | How is exact recurrence advanced while parallel work is exposed, and where is the final boundary state rounded and committed? |
| Synchronization | Which dependency requires each event or barrier, and which independent stages can overlap? |
| Buffering | Which producer/consumer pair justifies each additional buffer, and what is its worst-case context-memory cost? |
| Capacity | How do 40K, 60K, and approximately 130K requests fit without request-time allocation growth or silent truncation? |
| Startup specialization | Which offline-selected layouts and tactics are authenticated in the AOT plan, with no request-time discovery? |
| Upward leakage | Which Prefill interval and external TTFT component must move, and how will route attribution prove the connection? |

These questions intentionally stop above tile size, pipeline stage count,
cache instruction, fusion recipe, profiler threshold, or kernel name. Such
choices belong to an explicitly active local optimization work package and
retain authority only inside its declared role, shape, numerical mode, and
composition budget.

## 10. Roadmap activation and historical boundary

This SDD owns no execution sequence. [`ROADMAP.md`](ROADMAP.md) alone selects
the active delivery slice, architecture candidate, and local work packages.
An activation must name a work-package ID, parent candidate, product symptom,
scope, incumbent, real payload, stop condition, composition deadline, and API
return witness before a mechanism document can guide implementation.

The following records are dormant and have no current planning authority:

- [`PREFILL_ARCHITECTURE_RESET_LEGACY.md`](PREFILL_ARCHITECTURE_RESET_LEGACY.md)
  preserves the former mixed architecture plan, historical measurements,
  physical-limit arguments, local budgets, and execution order;
- [`GDN_PREFILL_DATAFLOW.md`](GDN_PREFILL_DATAFLOW.md) preserves GDN mechanism
  designs and experiment lineage;
- [`LARGE_M_PROJECTION_DATAFLOW.md`](LARGE_M_PROJECTION_DATAFLOW.md) preserves
  projection mechanism designs and shape-specific experiment lineage; and
- [`PREFILL_REFERENCE_AUDIT.md`](PREFILL_REFERENCE_AUDIT.md) and
  [`FP8_MARLIN_W8A16_SOURCE_MAP.md`](FP8_MARLIN_W8A16_SOURCE_MAP.md) preserve
  reference analysis and provenance.

Roadmap activation may delegate a bounded section of these records to a local
work package. It does not make their old thresholds, priorities, feasibility
claims, or production wording current. Any reused mechanism must be restated
in the active package against the current product route, incumbent, numerical
contract, evidence protocol, and expiry point.
