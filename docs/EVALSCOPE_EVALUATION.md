---
q3x_document:
  id: q3x-evalscope-evaluation
  class: procedure
  status: active
  owner: evaluation-maintainers
  authority: external API evaluation protocol, metric semantics, and artifact requirements
  effective: 2026-08-09
  last_reviewed: 2026-08-11
  supersedes: []
  superseded_by: []
  ssot_for: EvalScope and target-length external evaluation procedure
  review_trigger: product API, corpus, EvalScope version, metric, or release protocol change
---

# OpenAI-compatible external evaluation

This document owns an evaluation procedure, not current capability or pending
work. The presence, defaults, qualification, and gaps of the loopback adapter
and final product API are reported only in
[`CURRENT_STATUS.md`](CURRENT_STATUS.md); delivery order belongs only to
[`ROADMAP.md`](ROADMAP.md).

Governing update, 2026-08-09: EvalScope is one product-facing observation
surface, not the sole definition of real Prefill. Under the
[engineering constitution](ENGINEERING_CONSTITUTION.md), confirmed cold/no-cache
Agent API behavior is an authoritative planning input. A short-prompt
EvalScope corpus, P513 timer, logger-window statistic, or unmatched endpoint
cannot invalidate the 40K--60K and 130K long-context targets. Conflicts trigger
same-workload protocol/configuration reconciliation; they do not lower the
goal.

The project distinguishes two API surfaces:

- the **final product API** is the deliverable runner boundary. Its admission,
  streaming, cancellation, observability, capacity, defaults, and failure
  behavior are part of the product and therefore part of architecture and
  release selection;
- an **evaluation adapter** is a bounded measurement instrument. It may expose
  compatible request semantics while the final product server is incomplete,
  but it has no authority to prove production serviceability, security,
  concurrency, cancellation, or packaging.

Architecture work is selected through the closest complete product path: a
pinned real checkpoint, the final product API semantics,
[EvalScope](https://github.com/modelscope/evalscope), real prompt tokens, and
user-visible latency. The adapter may temporarily supply that observation only
when its route and request contract are attested as equivalent for the metric
being measured. A `release_candidate` must be measured through the actual
deliverable API; adapter-only evidence cannot promote it.

Internal Prefix timing, component benchmarks, NSys, and NCU remain essential
for local work packages and attribution, but they do not substitute for an
architecture or release API witness. EvalScope's
[stress-test workflow](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html)
is the current external performance surface.

All generated evaluation state stays under the ignored workspace directory.
From the repository root, initialize it once per checkout:

```bash
Q3X_WORK="$PWD/.q3x-work"
mkdir -p "$Q3X_WORK/cache/uv" "$Q3X_WORK/tmp" \
  "$Q3X_WORK/evalscope/corpora" "$Q3X_WORK/evalscope/results"
```

The commands below assume that `Q3X_WORK` remains set. They redirect the tool
cache and temporary directory so neither `$HOME` nor `/tmp` accumulates
project-owned EvalScope state.

Before every timed request set or profiler capture, perform the mandatory
fail-closed clean-host preflight. On Jetson, retain `tegrastats` observations,
CPU/process state, and GPU device-handle ownership with the run. Do not use the
Jetson `nvidia-smi` implementation to decide idleness or attribute GPU users.
An unexpected CPU/GPU consumer invalidates the run; discard its timing rather
than averaging or reporting it, wait for exclusive ownership, and start a new
run. Store the preflight record below `.q3x-work/` beside the evaluation
artifacts.

## Evaluation-adapter procedure (not the product API)

When `CURRENT_STATUS.md` identifies `qwen3x-eval-server` as available, the
following procedure exercises its deliberately small OpenAI-compatible subset
on loopback:

- `POST /v1/chat/completions` for text-only system/user/assistant messages;
- `POST /v1/completions` for one raw string or one flat token-ID prompt;
- `GET /v1/models` and `GET /healthz`;
- non-streaming JSON and real committed-token SSE, including exact usage;
- one serialized batch-one GPU worker behind bounded ingress and inference
  queues.

Build and run it with:

```bash
Q3X_BUILD="$Q3X_WORK/build/eval-server"
cmake --build "$Q3X_BUILD" --target qwen3x-eval-server -j
"$Q3X_BUILD/qwen3x-eval-server" MODEL_DIR \
  --host 127.0.0.1 --port 18080 \
  --model qwen3.6-27b-nvfp4 \
  --max-sequence-length 4096 --max-output-tokens 4096 \
  --prefill-chunk-size 512 --prefill-execution-mode legacy \
  --projection-backend sm87
```

The default remains `legacy`. The implemented development layer-major route
is selected explicitly with `--prefill-execution-mode layer-major`; it fails
closed unless the engine is configured for the SM87 backend, the fixed C512
compatibility workspace, the layer-major request-memory profile, and both
exact FP8 and NVFP4 Marlin Prefill inventories. It additionally requires the
native exact C64 GDN binary capability, the real native workspace and exact
byte capacity, and the authenticated 48-layer producer/weight shape
inventory. Eligible M32--M512 segments must report the native GDN disposition;
M1--M31 alone may use the sealed exact fallback. These inventories are still
development/test admissions, so this route is an executable evaluation
candidate rather than the installed production default. The request arena is
reserved before GPU execution. Target-length runs must therefore also pass a
plan-derived `--request-max-arena-bytes` value; the 2 GiB default is not valid
for 40K, 60K, or 130K layer-major requests.

The API is greedy only. Requests must explicitly provide a positive
`max_tokens` (or `max_completion_tokens`) and `temperature=0`; this prevents
the gateway from silently replacing OpenAI's sampling default. `top_p=1`,
`n=1`, a signed 64-bit audit seed, and `stop=null` are accepted. Sampling,
tools, media, logprobs, multiple choices, custom stops, and unknown fields
fail closed. Chat formatting always uses `enable_thinking=false`.

The listener is restricted to `127.0.0.1` because it has no authentication,
TLS, tenant isolation, or production admission policy. It is not continuous
batching or a production serving API. It must not be described, packaged, or
promoted as the final product API. A connection carries one request.
Before the first committed token, an engine error retains its real HTTP
4xx/5xx status; after streaming begins, a runtime failure is an SSE error and
the stream closes. In legacy mode, disconnect and shutdown cancellation remain
committed-token-boundary observations. In layer-major mode, the API supplies a
lock-free cancellation probe to a two-slot bounded submission window whose
quantum is one `layer x logical-C8192-panel`. The runner retires the oldest
completion event before polling, never has more than two quanta submitted, and
polls again after final normalization, before logits finalization, and before
the single state commit. Cancellation drains already submitted work, rolls the
request state back, publishes no partial sequence length, and emits no success
witness. The final commit check is the linearization point: cancellation
observed after it loses to the completed commit. The first HTTP response header
is also delayed until a first token or early error exists; long-context clients
must set a read timeout above expected TTFT.

The matched stock-vLLM process used this retained configuration:

```bash
vllm serve MODEL_DIR \
  --host 127.0.0.1 --port 18081 \
  --served-model-name qwen3.6-27b-nvfp4 \
  --max-model-len 4096 --max-num-seqs 1 \
  --max-num-batched-tokens 4096 --gpu-memory-utilization 0.78 \
  --kv-cache-dtype bfloat16 --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype bfloat16 \
  --no-enable-prefix-caching --no-enable-chunked-prefill \
  --attention-backend FLASHINFER --mamba-backend TRITON \
  --generation-config vllm \
  --default-chat-template-kwargs '{"enable_thinking":false}'
```

This retained 4K-limited configuration and its 20--1,160-token corpus are
historical directional evidence only. They do not represent the current
long-context Agent target or bound vLLM's achievable Prefill performance.

## Target-length Prefill witness contract

Prefill architecture selection is made on three cold/no-cache real-Agent
witnesses: 40K tokens, 60K tokens, and approximately 130K tokens. Each corpus
and manifest must freeze the exact token IDs and count after the pinned
tokenizer, request body, output contract, cache controls, model/config
revision, and prompt hash. The server must attest that it admitted the entire
prompt without truncation, Prefix/KV reuse, MTP, or an approximate numerical
route.

Run the witnesses in fail-fast 40K, 60K, then 130K order. The first small safe
API request may precede them to catch protocol, route, allocation, or output
failures, but it is a sanity check. A short workload, P513, or component cell
cannot select a Prefill `architecture_candidate`, even when its speedup is
statistically clean. The architecture decision requires the complete
predeclared target-length set; a `release_candidate` additionally requires
independent-process repetition and the full performance/accuracy/capability
protocol.

### Architecture admission before a timed witness

Before an architecture route receives a witness identity, its admission
record follows the order owned by the
[`PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md`](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md):

1. derive the real-number computation and production-live graph;
2. bind decoded operands, scale placement, reduction tree, rounding,
   publication, and recurrent-state boundaries;
3. bind every production observable and the full lifetime, alias set, reuse
   event, and physical owner of live data, control state, and workspace; and
4. bind the resulting SM87 residency, CTA/warp ownership, pipeline, buffering,
   synchronization, and physical route receipts.

An `accuracy-unqualified` direction may record a deliberately different
finite-precision tree at step 2 instead of claiming strict equivalence, but it
may not omit or disguise that boundary. Minimum safe admission is still
proportional to the decision: this ordering does not require full release
qualification before an early direction screen, but it does require an honest
numerical identity and a memory/control-state lifetime that cannot corrupt the
run.

For cross-phase state such as split-K locks, pointer validity and one initial
clear are not sufficient witness facts. The route must attest that no alias
writer covers the bytes before the last consumer, or that the plan explicitly
re-establishes the required value at the correct dependency edge. A partial
kernel or reference surface that is not connected to the complete Engine,
runner, and API route receives no new target-witness version, EvalScope
command, or performance claim; current availability is reported only in
[`CURRENT_STATUS.md`](CURRENT_STATUS.md).

For every witness retain both:

- EvalScope's user-visible TTFT from POST start to the first non-empty token
  event, along with request success, exact input/output token counts, finish
  status, and complete raw event identity; and
- server-side route attestation and intervals separating queue/admission,
  Prefix, first-token Decode, response publication, and any fallback or
  synchronization cost.

Legacy/unsealed evidence retains the byte-stable
`target-prefill-witness-v1` schema. A successfully committed sealed
layer-major request emits `target-prefill-witness-v2`, adding the actual
execution mode, logical panel count, request-memory profile, bounded-window
fact, retirement count, and stable deployment-plan identifier. An empty plan
identifier can never upgrade a record to v2. The sealed balanced compatibility
route at `18363ad` must identify itself exactly as
`q3x.sm87.exact.layer-major-c8192.balanced-segments.v2`; another identifier is
a route mismatch, not equivalent evidence.

An explicitly selected, default-off architecture candidate may emit
`target-prefill-witness-v3`. Version 3 adds the engine-lifetime Attention
tactic and completed operator-panel, grouped-Q64, grouped-Q128-v4, and
generic-QT2 hit counts. The Q64 tactic identifies itself as
`native-group-q64-panel` with plan
`q3x.sm87.ac-prefill-layermajor-8k.native-group-q64-panel.v1`; the Q128-v4
tactic identifies itself as `native-group-q128-v4-panel` with plan
`q3x.sm87.ac-prefill-layermajor-8k.native-group-q128-v4-panel.v1`. Their hit
counters are separate: one tactic may not report work under the other's
counter. Both must also carry
`qualification=accuracy-unqualified-architecture-candidate`, a false
`numerical_contract.qualified` value, and an
`architecture_candidate_unqualified` disabled-boundary scope until the full
accuracy protocol passes. In that state `approximate_numerics` is not attested
as disabled, even when no generic forbidden-route counter was incremented.
A v3 direction record can select whether development continues; it cannot
qualify numerical correctness, change the exact default, or serve as release
or production evidence.

An explicitly selected segmented operator-panel projection screen emits
`target-prefill-witness-v4`. Version 4 retains every v3 safety and
qualification field, and additionally records:

- `projection_tactic=segmented-marlin-operator-panel`;
- `attention_tactic=exact-segmented` or
  `native-group-q64-panel` or `native-group-q128-v4-panel`;
- separate grouped-Q64 and grouped-Q128-v4 completed panel hit counters;
- completed logical segmented-wrapper projection hits; and
- shape-aware physical Marlin kernel launches submitted inside the FP8 and
  NVFP4 wrappers.

The exact-segmented-Attention combination identifies itself as
`q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel.exact-segmented-attention.v1`.
The grouped-Q64-Attention combination identifies itself as
`q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel.native-group-q64-attention.v1`.
The grouped-Q128-v4-Attention combination identifies itself as
`q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel.native-group-q128-v4-attention.v1`.
All three plan IDs remain default-off and
`accuracy-unqualified-architecture-candidate`; a v4 counter is execution
evidence, not proof of numerical equivalence or production eligibility. The
exact projection tactic continues to emit the existing v2 or v3 schema and
plan identity, so enabling v4 cannot silently relabel an incumbent result.

An explicitly selected native quantized large-M operator-panel projection
screen emits `target-prefill-witness-v5`. Version 5 preserves the v4 safety
and accuracy-qualification boundary while identifying a different physical
route:

- `projection_tactic=native-quantized-large-m-operator-panel`;
- `attention_tactic=exact-segmented`, `native-group-q64-panel`, or
  `native-group-q128-v4-panel`;
- separate grouped-Q64 and grouped-Q128-v4 completed panel hit counters;
- completed logical native large-M FP8/NVFP4 projection hits; and
- separate `native_large_m_projection_bulk_hits` and
  `native_large_m_projection_oracle_partial_hits` counters for projections
  that used the authenticated M8192 single-bulk route and projections that
  retained the exact partial-panel oracle ledger; their sum must equal
  `native_large_m_projection_hits`; and
- physical native large-M launches: one frozen Marlin launch for each
  authenticated M8192 logical projection, or the complete established Marlin
  span-ledger launch count for a partial panel.

The exact-segmented-Attention combination identifies itself as
`q3x.sm87.ac-prefill-layermajor-8k.native-quantized-large-m-operator-panel.exact-segmented-attention.v1`.
The grouped-Q64-Attention combination identifies itself as
`q3x.sm87.ac-prefill-layermajor-8k.native-quantized-large-m-operator-panel.native-group-q64-attention.v1`.
The grouped-Q128-v4-Attention combination identifies itself as
`q3x.sm87.ac-prefill-layermajor-8k.native-quantized-large-m-operator-panel.native-group-q128-v4-attention.v1`.
All three plans consume the prepared FP8/NVFP4 Marlin weight and scale
sidecars and bind their typed reduction arenas and locks. NVFP4 Gate+Up is the
single merged Marlin artifact; Down owns its independent sidecar and
workspace. A conforming v5 emitter must issue one physical bulk launch only
for a complete M8192 panel. Every partial panel must preserve the exact
arithmetic ledger, masked specialization positions, legacy MLP workspace, and
Down-to-residual interleave. The projection tactic does not own or alter the
immutable balanced logical-panel geometry. These are procedure and route
requirements, not release qualification:
every v5 plan remains default-off and
`accuracy-unqualified-architecture-candidate` until the full accuracy
protocol passes. Existing v1--v4 records and plan identities remain
byte-stable for their original routes.

An explicitly selected FlashInfer exact logical-panel Attention screen emits
`target-prefill-witness-v6`. Version 6 identifies the prompt-wide v2
Attention dataflow without relabeling the grouped-Q64/Q128 or segmented exact
routes:

- `attention_tactic=native-flashinfer-exact-panel`;
- `native_flashinfer_exact_panel_hits` counts completed full-Attention logical
  panels independently from grouped-Q64, grouped-Q128-v4, and generic-QT2;
- the count must equal `16 * logical_panel_count`, while all three incumbent
  Attention counters must be zero; and
- projection tactic and all v4/v5 projection counters remain explicit, so the
  Attention change cannot conceal which projection path executed.

The exact-segmented projection combination identifies itself as
`q3x.sm87.ac-prefill-prompt-wide-v2.exact-segmented-projection.native-flashinfer-exact-panel-attention.v1`.
The segmented-Marlin projection combination identifies itself as
`q3x.sm87.ac-prefill-prompt-wide-v2.segmented-marlin-operator-panel.native-flashinfer-exact-panel-attention.v1`.
The native-large-M projection combination identifies itself as
`q3x.sm87.ac-prefill-prompt-wide-v2.native-quantized-large-m-operator-panel.native-flashinfer-exact-panel-attention.v1`.

The tactic uses exact causal masking, the model's BF16 Q/K/V and output
boundaries, and FP32 online-softmax state; it is not the grouped-Q64
approximation. FlashInfer's parallel reduction order nevertheless differs
from the segmented incumbent, so every v6 route remains default-off and
`accuracy-unqualified-architecture-candidate` until the complete real-model
state and public no-regression protocol passes. No v6 record may be emitted
for a request whose panel geometry would route through the compatibility
executor. Existing v1--v5 serialization and plan identities remain
byte-stable.

An explicitly selected coupled GateUpG2/DownD2 projection screen emits
`target-prefill-witness-v8`. Version 8 identifies the default-off
`native-nvfp4-g2-d2-large-m-operator-panel` route without relabeling the
earlier native-large-M or retained native-Marlin paths:

- `nvfp4_package=gate-g2+down-d2`, `package_complete=true`, and the role
  identities `GateUpG2` and `DownD2` must all be present;
- `native_nvfp4_gate_up_g2_hits` and `native_nvfp4_down_d2_hits` must each
  equal `64 * logical_panel_count`, while their sum equals
  `native_nvfp4_g2_d2_projection_hits` and the physical-launch count;
- G2 must attest fused Gate+Up+SiLU publication to activated BF16, and D2 must
  attest its fused in-place residual publication;
- companion FP8 and the selected Attention-route counts remain explicit. For
  Q64, Q128-v4, or exact FlashInfer, the selected native count must equal
  `16 * logical_panel_count` and the other Attention counters must be zero;
  exact-segmented instead requires nonzero generic-QT2 evidence and zero native
  Attention counts. The coupled package therefore cannot conceal another
  projection or Attention route; and
- exact fallback, forbidden fallback, Prefix cache, MTP, cuBLASLt,
  external-reference, and approximate-route counters must remain zero.

The four admitted v8 Attention compositions have distinct plan identities:

- exact segmented:
  `q3x.sm87.ac-prefill-prompt-wide-v2.native-nvfp4-g2-d2-large-m-operator-panel.exact-segmented-attention.v2`;
- grouped Q64:
  `q3x.sm87.ac-prefill-prompt-wide-v2.native-nvfp4-g2-d2-large-m-operator-panel.native-group-q64-attention.v2`;
- grouped Q128-v4:
  `q3x.sm87.ac-prefill-prompt-wide-v2.native-nvfp4-g2-d2-large-m-operator-panel.native-group-q128-v4-attention.v2`; and
- exact FlashInfer:
  `q3x.sm87.ac-prefill-prompt-wide-v2.native-nvfp4-g2-d2-large-m-operator-panel.native-flashinfer-exact-panel-attention.v2`.

It admits only M8192 and M7712 logical panels and fails closed on another M.
The conservative `RequestState` three-span MLP tactic is a workspace
reservation identity, not a claim that physical Gate+Up, SiLU, and Down are
separate. The v8 DeploymentPlan and request witness are the execution
authority for the fused route. Because the measured route is rejected and
default-off, no new planner-layout enum is required merely to rename its dead
reservation spans. Existing v1--v7 serialization and plan identities remain
byte-stable.

Every v8 route remains `accuracy-unqualified-architecture-candidate` with a
false `numerical_contract.qualified` value until the complete state and public
no-regression protocol passes. Matching one generated token is a transport and
output smoke observation, never an accuracy gate.

The recorded clean-host P40K v8 direction selected the exact FlashInfer plan
identity above, consumed all 40,000 tokens, recorded 320 G2 plus 320 D2 hits,
and reported 115,085.76 ms EvalScope TTFT and 115,041.751913 ms server pure
Prefill (347.699851 tok/s). Retained `c45b7c5` was 108,981.854892 ms /
367.033577 tok/s, so the direction was rejected at +5.560464% latency /
-5.267563% throughput. Bounded NSys measured G2+D2 at
60.904184288 s versus 53.864024544 s for retained native Marlin main plus
SiLU, a +7.040159744-s role gap and +6.110398080 s in complete GPU kernel
time. This profile explains rejection; it cannot reverse it.

P60 was not run. Its balanced logical geometry is `6x8192 + 2x5424`, while
the v8 candidate has no M5424 path. That fact is an admission/geometry blocker,
not a P60 timing or performance conclusion. The complete P40 route, hashes,
limitations, and default-off decision are frozen in the
[G2/D2 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-g2-d2-rejection-2026-08-10.json).

The exact-P40000 layer-wide MLP experiment emits
`target-prefill-witness-v9`. It admits one 64-layer route pass, 64 persistent
Gate/Up calls, 64 persistent Down+residual calls, and 128 NVFP4 physical
launches while preserving the explicit panel-local FP8 and Attention counts.
Its real-API direction was neutral-negative and is closed; version 9 remains
necessary to distinguish the measured implementation from a planned
whole-core route. The exact route and rejection are frozen in the
[v9 rejection record](metadata/qwen36-27b-prefill-p40k-persistent-layerwide-mlp-rejection-2026-08-10.json).

The following exact-P40000 whole-core experiment emits
`target-prefill-witness-v10`. Version 10 is valid only when all of the
following are true:

- request memory profile is `layer-major-p40-whole-core`, projection tactic is
  `native-prompt-wide-p40-whole-core`, Attention tactic is
  `native-flashinfer-exact-whole-prompt`, and the sealed DeploymentPlan is
  `q3x.sm87.ac-prefill-prompt-wide-v2.native-p40-whole-core.v1`;
- the prompt is consumed exactly as five M8000 panels in one 64-layer route
  pass, with 320 fill, 64 prompt-core, 320 drain, and 64 MLP phases;
- the two-slot submission window is bounded and retires exactly 768 phases;
- FP8 projection hits and physical launches are both 1,040, BF16 A/B and GDN
  hits are both 48, whole-prompt FlashInfer hits are 16, and persistent NVFP4
  Gate/Up and Down hits are 64 each with 128 physical launches; and
- all exact-fallback, forbidden, Prefix-cache, MTP, cuBLASLt,
  external-reference, and approximate-route counters are zero.

The clean-host P40K v10 screen reached 101,870.53 ms EvalScope TTFT and
101,831.853876 ms server pure Prefill, or 392.804397 tok/s. This is a positive
7.02138% direction against retained v6, not a release baseline: the route is
default-off, the exact binary came from a pinned dirty working tree, and the
inherited FlashInfer arithmetic has a known P513 full-state mismatch. Bounded
Nsight then showed only 7.992928 ms of non-kernel space in a 102.121307-s
request, so the next work replaces dominant GPU dataflows rather than the API
adapter. Exact hashes, counts, and limitations are frozen in the
[v10 whole-core direction record](metadata/qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json).

The default-off exact-P40000 grouped projection-reset experiment emits
`target-prefill-witness-v11`; it may not reuse or be serialized as v10. A v11
record is complete only when all of the following are true:

- request memory profile is `layer-major-p40-whole-core`, projection tactic is
  `native-prompt-wide-p40-projection-reset`, MLP schedule is
  `prompt-wide-p40-projection-reset`, Attention tactic is
  `native-flashinfer-exact-whole-prompt`, and the sealed DeploymentPlan is
  `q3x.sm87.ac-prefill-prompt-wide-v2.native-p40-projection-reset.v1`;
- one 64-layer route pass retains five M8000 input-preparation and five M8000
  residual phases per layer, one whole-prompt core phase, one whole-prompt MLP
  phase, and the bounded two-slot window retires exactly 768 phases;
- FP8 reports 208 logical tensor-role hits but exactly 128 physical launches:
  one grouped P40000 input projection and one P40000 output projection per
  layer. Its bound artifact is the authenticated FP8 supermatrix sidecar, not
  the M8000 FP8 Marlin sidecar;
- BF16 A/B and GDN hits are both 48, whole-prompt FlashInfer hits are 16, and
  P40000 NVFP4 Gate/Up and Down hits are 64 each with 128 physical launches;
  and
- every exact-fallback, forbidden, Prefix-cache, MTP, cuBLASLt,
  external-reference, and approximate-route counter is zero.

Configure and start this default-off candidate with the complete fixed
geometry below. The exact arena is 8,640,542,976 bytes; using the generic
2 GiB adapter default is invalid for this route.

```bash
Q3X_BUILD="$Q3X_WORK/build/p40-projection-reset"
cmake -S . -B "$Q3X_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DQ3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION=ON \
  -DQ3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION=ON \
  -DQ3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION=ON \
  -DQ3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION=ON \
  -DQ3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION=ON \
  -DQ3X_BUILD_P40_PROJECTION_RESET_ADMISSION=ON
cmake --build "$Q3X_BUILD" --target qwen3x-eval-server -j

"$Q3X_BUILD/qwen3x-eval-server" MODEL_DIR \
  --host 127.0.0.1 --port 18089 \
  --model qwen3.6-27b-nvfp4 \
  --max-sequence-length 40001 --max-output-tokens 1 \
  --prefill-chunk-size 512 --prefill-execution-mode layer-major \
  --prefill-attention-tactic native-flashinfer-exact-whole-prompt \
  --prefill-projection-tactic native-prompt-wide-p40-projection-reset \
  --projection-backend sm87 \
  --request-max-arena-bytes 8640542976 \
  --min-free-bytes 4294967296 \
  --queue-capacity 1 --ingress-threads 3
```

The selection request must use the frozen flat token-ID P40000 corpus with
`max_tokens=1`, `temperature=0`, `stream=true`, no warmup, no Prefix cache,
and no MTP. A v11 package is incomplete unless all 40,000 prompt tokens are
reported consumed and at least one completion token is committed.

Version 11 remains `accuracy-unqualified-architecture-candidate` until the
real-checkpoint numerical protocol passes. Its first selection gate is the
cold/no-cache real-API P40 request defined here. P60 remains fail-closed until
a separate exact-P60000 geometry, capacity, plan, and witness exist; a P40
result cannot be relabeled as P60 support.

That P40 gate is now closed as a negative architecture result. The clean-host
request completed 40,000/40,000 prompt tokens with no cache hit and the same
one-token smoke output as v10, but reached 194,256.49 ms EvalScope TTFT and
194,220.222475 ms server pure Prefill, or 205.951777 tok/s. This is 90.7264%
more pure-Prefill latency than v10. A matched bounded Nsight capture assigns
92.264 seconds of added work to the substituted projection package: grouped
FP8 rises from 25.865 to 103.177 seconds, Gate/Up from 37.273 to 45.719
seconds, and Down from 17.559 to 24.065 seconds. Kernel time covers 99.997% of
the request, so API, host launch gaps, and repeated runtime validation do not
explain the regression. The route stays default-off and accuracy-unqualified;
P60 was not run. Exact hashes, route counts, kernel attribution, and
limitations are frozen in the
[v11 rejection record](metadata/qwen36-27b-prefill-p40k-projection-reset-rejection-2026-08-10.json).

The measured, now-removed default-off phase-local BF16 experiment emitted
`target-prefill-witness-v12`; that historical identity may not be reused by a
future route. Its experimental deployment plan was
`q3x.sm87.ac-prefill-p40.phase-local-canonical-bf16-dense.native-flashinfer-exact-whole-prompt.v1`,
projection tactic `native-prompt-wide-p40-phase-local-bf16`, the same complete
P40000 whole-request transaction, and 208 FP8 logical projection receipts.
Its 416 FP8 physical count was a runtime aggregate after successful
expansion+dense pairs. The 512 NVFP4 MLP count was inferred from the fixed
successful-enqueue composition of 192 expansion, 192 dense, 64 SiLU, and 64
residual launches; it was not an independent kernel counter. Post-attention
norm was separately expected 64 times and was not part of that total.

No raw v12 server witness was retained, and its immutable topology reused a
two-kernel projection-reset schedule that disagreed with the physical
phase-local lowering. The historical witness therefore has no sealed-route,
promotion, or replay authority. The experiment's external timing and bounded
profile remain negative-direction evidence under the limitations recorded in
the
[v12 phase-local record](metadata/qwen36-27b-prefill-p40k-phase-local-bf16-rejection-2026-08-10.json),
but results and current status do not belong in this procedure.

The default-off AOT packed-operand projection experiment emits the independent
`target-prefill-witness-v13` schema. It must never reuse, deserialize as, or
claim continuity from historical v12: v12 described a removed phase-local BF16
expansion/dense lowering, lacked a retained raw server witness, and had no
sealed-route authority. A conforming v13 request is complete only when all of
the following are true:

- projection tactic is `native-prompt-wide-p40-packed-projection`, MLP schedule
  is `prompt-wide-p40-packed-projection`, Attention tactic remains
  `native-flashinfer-exact-whole-prompt`, request memory profile remains
  `layer-major-p40-whole-core`, and the sealed DeploymentPlan is exactly
  `q3x.sm87.ac-prefill-p40-packed-dataflow.native-p40-packed-projection.v1`;
- the engine has prepared and authenticated exactly 256 physical packed
  artifacts from 400 original checkpoint tensor sources before readiness,
  reports 16,840,130,560 resident packed-asset bytes, and has not prepared or
  selected the old FP8 supermatrix sidecar for this route;
- one 64-layer pass consumes exactly five M8000 panels, completes 320 fill, 64
  prompt-core, 320 drain, and 64 MLP phases, and retires the bounded two-slot
  window exactly 768 times;
- FP8 records 208 logical tensor-role hits and exactly 128 physical launches;
  NVFP4 records 64 merged Gate+Up and 64 Down+residual hits in exactly 128
  physical launches; BF16 A/B and GDN each record 48 hits and whole-prompt
  FlashInfer records 16; and
- the package identity is `exact-p40000-packed-projection-dataflow-v1`, every
  role receipt is complete, all exact/forbidden fallback counts are zero, and
  Prefix cache, MTP, cuBLASLt, external-reference, and approximate-route hits
  are zero.

The v13 selection gate is the same frozen flat token-ID cold/no-cache P40000
OpenAI completions request: one successful EvalScope 1.9.1 request, exactly
40,000 prompt tokens fully consumed, one committed greedy token, no warmup,
Prefix cache, or MTP, plus an accepted clean-host preflight immediately before
timing. Route completeness and a matching first token are transport/admission
evidence only. Version 13 remains
`accuracy-unqualified-architecture-candidate` until full real-model state and
capability qualification passes, and one sample has early-stop rather than
architecture-selection or release authority. P60 remains fail-closed until a
competitive, accuracy-admissible P40 result and an independent exact-P60000
geometry, capacity plan, route, and witness exist.

The first valid clean-host v13 request proved that complete API, asset, plan, and
route integration, but rejected packed projection v1 on performance. Server
pure Prefill was 161,410.929373 ms / 247.814693561 prompt tok/s and EvalScope
reported 161,447.32 ms TTFT / 247.758768301 New Prompt tok/s. Against v10 at
101,831.853876 ms / 392.804397 tok/s, it added 59,579.075497 ms and reduced
throughput by 36.9114%. EvalScope TTFT exceeded server TTFT by only 3.807390
ms, so the API boundary is not the cause. The route remains default-off, full
accuracy/repetition did not run, and P60/P130 stay locked. Exact hashes,
counts, limitations, and the bounded post-rejection profiling boundary are
frozen in the
[v13 packed projection rejection record](metadata/qwen36-27b-prefill-p40k-packed-projection-rejection-2026-08-10.json).

The successor packed-NVFP4-v2 experiment emits the independent
`target-prefill-witness-v14` schema. Version 14 does not inherit v13's packed
FP8 execution path or its 208-logical/128-physical FP8 counts: it restores the
v10 whole-core FP8 dataflow and changes only the P40000 NVFP4 Gate+Up and Down
executors. Existing v1--v13 records, fields, counters, and plan identities
remain byte-stable. A conforming v14 request is complete only when all of the
following are true:

- projection tactic is `native-prompt-wide-p40-packed-nvfp4-v2`, MLP schedule
  is `prompt-wide-p40-packed-nvfp4-v2`, Attention tactic is
  `native-flashinfer-exact-whole-prompt`, request memory profile is
  `layer-major-p40-whole-core`, and the sealed DeploymentPlan is exactly
  `q3x.sm87.ac-prefill-p40-packed-dataflow-v2.native-p40-packed-nvfp4-shape-specific.v1`;
- the package identity is
  `exact-p40000-packed-nvfp4-v2-dataflow-v1`. Before readiness the engine has
  prepared and authenticated exactly 128 physical P40 packed NVFP4 artifacts
  from 192 logical checkpoint sources: 64 merged Gate+Up artifacts retain the
  independent Gate and Up source identities, and 64 Down artifacts retain
  their Down source identities. This NVFP4-only engine-lifetime asset arena is
  exactly 9,625,927,680 bytes. Version 14 must not prepare or select v13's
  packed FP8 artifacts, and it forbids request-time repacking or tactic
  selection. The startup readiness record must therefore report
  `p40_packed_projection_assets_enabled=1`,
  `p40_packed_projection_artifacts=128`,
  `p40_packed_projection_sources=192`, and
  `p40_packed_projection_asset_bytes=9625927680`. The corresponding engine
  load inventory must contain zero packed FP8 logical roles, zero packed FP8
  physical artifacts, and 128 packed NVFP4 physical artifacts;
- one 64-layer route pass consumes exactly five M8000 panels, completes 320
  fill, 64 whole-prompt core, 320 drain, and 64 whole-prompt MLP phases, and
  retires the bounded two-slot submission window exactly 768 times;
- the restored v10 FP8 path records exactly 1,040 logical projection hits and
  1,040 physical launches. BF16 A/B and GDN each record 48 completed hits and
  whole-prompt FlashInfer records 16;
- the common P40 MLP ledger records 64 completed layers, 64 Gate+Up hits, 64
  Down+residual hits, and 128 physical NVFP4 launches. The v2 identity ledger
  independently records `packed_nvfp4_v2_gate_up_hits=64`,
  `packed_nvfp4_v2_down_hits=64`, and
  `packed_nvfp4_v2_physical_launches=128`; these counters attest the same 128
  physical launches and must not be added to the common ledger; and
- every role receipt is production-only, every exact-fallback, forbidden, and
  incompatible-route counter is zero, and Prefix cache, MTP, cuBLASLt,
  external-reference, and approximate-route hits are zero.

Configure and start this default-off candidate with the complete fixed P40
geometry below. The whole-core request arena remains exactly 8,640,542,976
bytes and is independent from the 9,625,927,680-byte engine-lifetime packed
NVFP4 asset arena. The generic 2 GiB adapter default is invalid for this
route.

```bash
Q3X_BUILD="$Q3X_WORK/build/p40-packed-nvfp4-v2"
cmake -S . -B "$Q3X_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DQ3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION=ON \
  -DQ3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION=ON \
  -DQ3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION=ON \
  -DQ3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION=ON \
  -DQ3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION=ON \
  -DQ3X_BUILD_P40_PACKED_PROJECTION_ADMISSION=ON \
  -DQ3X_BUILD_P40_PACKED_NVFP4_V2_ADMISSION=ON
cmake --build "$Q3X_BUILD" --target qwen3x-eval-server -j

"$Q3X_BUILD/qwen3x-eval-server" MODEL_DIR \
  --host 127.0.0.1 --port 18091 \
  --model qwen3.6-27b-nvfp4 \
  --max-sequence-length 40001 --max-output-tokens 1 \
  --prefill-chunk-size 512 --prefill-execution-mode layer-major \
  --prefill-attention-tactic native-flashinfer-exact-whole-prompt \
  --prefill-projection-tactic native-prompt-wide-p40-packed-nvfp4-v2 \
  --projection-backend sm87 \
  --request-max-arena-bytes 8640542976 \
  --min-free-bytes 4294967296 \
  --queue-capacity 1 --ingress-threads 3
```

`Q3X_BUILD_P40_PACKED_PROJECTION_ADMISSION=ON` is a compile-time ABI and
packing-source prerequisite for v2. Its presence does not authorize preparing
or selecting the packed-v1 FP8 inventory; the selected v14 load transaction
must still publish the NVFP4-only counts above.

The v14 P40 first gate uses the frozen flat token-ID corpus
`.q3x-work/evalscope/corpora/q3x-repository-agent-context-p40000-one-token.jsonl`,
whose SHA-256 is
`8970ac50693f49d1b27d35a0610ecbe5072594330d69b301f4dab731789b6844`.
After an accepted clean-host preflight immediately before the request, run
exactly one cold/no-cache EvalScope 1.9.1 request with no warmup:

```bash
Q3X_P40_CORPUS="$Q3X_WORK/evalscope/corpora/q3x-repository-agent-context-p40000-one-token.jsonl"
Q3X_P40_RESULTS="$Q3X_WORK/evalscope/results/p40-packed-nvfp4-v2"

TMPDIR="$Q3X_WORK/tmp" XDG_CACHE_HOME="$Q3X_WORK/cache" \
UV_CACHE_DIR="$Q3X_WORK/cache/uv" \
uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
  --model qwen3.6-27b-nvfp4 --api openai \
  --url http://127.0.0.1:18091/v1/completions \
  --tokenizer-path MODEL_DIR \
  --dataset line_by_line --data-source local \
  --dataset-path "$Q3X_P40_CORPUS" \
  --number 1 --parallel 1 --warmup-num 0 --num-workers 1 \
  --max-prompt-length 131072 \
  --max-tokens 1 --temperature 0 --seed 42 \
  --stream --tokenize-prompt --no-test-connection \
  --total-timeout 680 \
  --outputs-dir "$Q3X_P40_RESULTS" \
  --name packed-nvfp4-v2 --no-timestamp
```

This single request has early-stop authority for the v2 architecture
direction only. It is valid only if EvalScope reports one successful request,
the server attests all 40,000 prompt tokens consumed without cache reuse, and
one greedy token is committed under the complete v14 route above. A negative
comparison with the frozen v10 P40 incumbent closes or redesigns v2 before
full accuracy, repetition, P60, or P130 work; a positive direction unlocks
the real-checkpoint numerical and capability gates but does not pass them.
Version 14 therefore remains default-off and
`accuracy-unqualified-architecture-candidate`, with
`numerical_contract.qualified=false`. P60 remains fail-closed until P40 is
competitive and accuracy-admissible and an independent exact-P60000
geometry, capacity plan, route, and witness exist.

The first valid clean-host v14 request completed on 2026-08-11 and exercised
the exact route above. EvalScope reported one successful request, 40,000
input tokens, one output token, zero cached-prompt throughput, 128,532.05 ms
TTFT, and 311.206330 New Prompt tok/s. The server independently reported
128,493.372123 ms pure Prefill, or 311.300103 prompt tok/s. Its single v14
witness matched the frozen body and token hashes, consumed the complete
prompt, recorded 1,040/1,040 restored-v10 FP8 hits/launches and both
64/64/128 NVFP4 ledgers, completed every per-operator production route, and
recorded zero fallback, forbidden, Prefix-cache, MTP, cuBLASLt, external, and
approximate-route hits. The external/server TTFT gap was only 3.933331 ms.

Against v10's 101,831.853876 ms / 392.804397 tok/s pure Prefill, v14 added
26,661.518247 ms, increased latency by 26.181904%, and reduced throughput by
20.749333%. This is a valid performance negative, not an invalid route or API
result. Under the predeclared early-stop rule it closes packed NVFP4 v2
without repetition, P60/P130, full accuracy, or a local parameter scan. The
default-off implementation and CUDA correctness gates remain development
evidence only. Exact hashes, preflights, counters, metrics, and cleanup facts
are frozen in the
[v14 packed NVFP4 rejection record](metadata/qwen36-27b-prefill-p40k-packed-nvfp4-v2-rejection-2026-08-11.json).

### Pending v15 stock-vLLM-Marlin projection-parity P40 gate

The Engine/runner/API host integration and stable lock-owner contract are now
implemented and pass both ordinary default-off and explicit admission host
builds/tests. The CUDA target compiles, but this exact integrated route has not
yet run a GPU numerical or performance gate. The default-off stock-vLLM-Marlin
projection-parity route must emit an independent
`target-prefill-witness-v15`. Version 15 may not deserialize as, relabel, or
borrow qualification from v10 or v14. Its serialization fields are frozen by
the host protocol tests, and the following facts are mandatory:

- request profile `layer-major-p40-whole-core`, projection tactic
  `native-prompt-wide-p40-vllm-marlin-parity`, Attention tactic
  `native-flashinfer-exact-whole-prompt`, and deployment plan
  `q3x.sm87.ac-prefill-p40-vllm-marlin-parity.native-p40-canonical-nvfp4-legacy-stripe.v1`;
- the exact independent 64-layer Gate+Up/Down parity artifact inventory,
  canonical per-token `[Gate, Up]` publication, standalone SiLU, and
  standalone Down residual boundary; its transformation digest binds source
  provenance and the deterministic repack recipe, not post-pack device bytes;
- for each of 64 layers and each of the two NVFP4 roles, exactly 39 full-K
  M1024 `LegacyStripe` launches followed by one M64 split-K tail, with the
  plan-declared FP32 Ctmp reduction and ordered-lock protocol;
- one request-level parity lock clear only after the lock view is bound to a
  physically disjoint stable owner whose complete lifetime excludes GDN or
  another family writer; surrounding ordered v10 FP8 users must leave that
  owner in the state required by the first and subsequent parity tails;
- the unchanged v10 FP8, BF16 A/B, GDN, whole-prompt FlashInfer, whole-request
  transaction, request arena, progress, and final state-commit route; and
- zero Prefix/cache reuse, MTP, cuBLASLt, external-reference, approximate,
  exact-fallback, forbidden-fallback, incompatible-route, request-time repack,
  or request-time tactic-discovery hits.

This route remains `accuracy-unqualified-architecture-candidate` and
default-off. Matching one token cannot qualify the stock split-K reduction
tree or the already known whole-prompt FlashInfer numerical difference.

After a clean-host Jetson resource preflight, configure and start the pending
v15 candidate with the fixed P40 geometry below. The command is a frozen
future measurement procedure, not evidence that it has already run.

```bash
Q3X_BUILD="$Q3X_WORK/build/p40-vllm-marlin-parity"
cmake -S . -B "$Q3X_BUILD" -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DQ3X_BUILD_FP8_MARLIN_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_NVFP4_MARLIN_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_BF16_AB_LARGE_M_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_FLASHINFER_PREFILL_ATTENTION_ADMISSION=ON \
  -DQ3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION=ON \
  -DQ3X_BUILD_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION=ON \
  -DQ3X_BUILD_LAYER_WIDE_P40_MLP_ADMISSION=ON \
  -DQ3X_BUILD_NVFP4_PERSISTENT_PREFILL_ADMISSION=ON \
  -DQ3X_BUILD_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION=ON \
  -DQ3X_BUILD_P40_VLLM_MARLIN_PARITY_ADMISSION=ON
cmake --build "$Q3X_BUILD" --target qwen3x-eval-server -j

"$Q3X_BUILD/qwen3x-eval-server" MODEL_DIR \
  --host 127.0.0.1 --port 18092 \
  --model qwen3.6-27b-nvfp4 \
  --max-sequence-length 40001 --max-output-tokens 1 \
  --prefill-chunk-size 512 --prefill-execution-mode layer-major \
  --prefill-attention-tactic native-flashinfer-exact-whole-prompt \
  --prefill-projection-tactic native-prompt-wide-p40-vllm-marlin-parity \
  --projection-backend sm87 \
  --request-max-arena-bytes 8640542976 \
  --min-free-bytes 4294967296 \
  --queue-capacity 1 --ingress-threads 3
```

The first v15 decision unit uses the same exact frozen token-ID corpus and
hash as v14. Immediately after an accepted clean-host Jetson preflight, run
one cold/no-cache EvalScope 1.9.1 request, one greedy output token, concurrency
one, no warmup:

```bash
Q3X_P40_CORPUS="$Q3X_WORK/evalscope/corpora/q3x-repository-agent-context-p40000-one-token.jsonl"
Q3X_P40_RESULTS="$Q3X_WORK/evalscope/results/p40-vllm-marlin-parity-v1"

TMPDIR="$Q3X_WORK/tmp" XDG_CACHE_HOME="$Q3X_WORK/cache" \
UV_CACHE_DIR="$Q3X_WORK/cache/uv" \
uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
  --model qwen3.6-27b-nvfp4 --api openai \
  --url http://127.0.0.1:18092/v1/completions \
  --tokenizer-path MODEL_DIR \
  --dataset line_by_line --data-source local \
  --dataset-path "$Q3X_P40_CORPUS" \
  --number 1 --parallel 1 --warmup-num 0 --num-workers 1 \
  --max-prompt-length 131072 \
  --max-tokens 1 --temperature 0 --seed 42 \
  --stream --tokenize-prompt --no-test-connection \
  --total-timeout 680 \
  --outputs-dir "$Q3X_P40_RESULTS" \
  --name vllm-marlin-parity-v1 --no-timestamp
```

The result is valid only if EvalScope reports one successful request, the
server attests the exact 40,000 input tokens and one committed output token,
the complete v15 route facts above hold, and both the preflight and raw route
witness are retained. Compare server pure Prefill and external TTFT against
the frozen v10 native incumbent; vLLM and cuBLASLt remain references, not
incremental rejection or production paths. A negative result closes or
redesigns this parity skeleton before repetition, P60, P130, or full accuracy;
a positive direction unlocks the independent numerical/SASS/state gates but
does not pass them.

**This v15 gate has not been run. There is currently no parity TTFT, pure-
Prefill latency, or prompt-throughput result.**

The first WP-V2-C1-v3 direction reused the exact v10 host schedule and route
counters through a binary-pinned, default-off overlay; the binary hash, not a
new witness name, distinguishes its substituted shape-wide NVFP4 body. It
returned to this same P40K API gate before accuracy or statistical work and
regressed server pure Prefill to 106,374.300578 ms / 376.030675 tok/s. One
bounded profile attributed the loss to the substituted NVFP4 package, so the
overlay was removed and P60/P130 stayed locked. This early-overlay exception
does not authorize ambiguous identity for a positive or promotable route.
Exact evidence is frozen in the
[v3 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-shape-wide-v3-rejection-2026-08-10.json).

The externally observed TTFT selects the whole architecture. Server-side pure
Prefill timing explains where that result came from; it never replaces the API
result. A witness with an incomplete stream, unowned host resources, a route
mismatch, allocation fallback, truncation, cache reuse, or missing interval is
invalid rather than slow or fast.

### Retained sealed P1025 direction screen

For the current sealed-route screen and its authority, see
[`CURRENT_STATUS.md`](CURRENT_STATUS.md). Exact identities, hashes, route
counts, comparator differences, and limitations are frozen in
[`metadata/qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json`](metadata/qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json).

## Historical pinned short workload

The workload below is retained to reproduce the first external direction
baseline and to provide a cheap protocol/regression proxy. It is not the
active Prefill architecture-selection workload.

Generate the 33-request corpus as described in
[`benchmarks/evalscope/README.md`](../benchmarks/evalscope/README.md). Its
SHA-256 is
`bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`.
The first request is a warmup. The 32 measured prompts have 20--1,160 tokens
(mean 493.8125; conventional even-sample median 509.5; EvalScope's
nearest-observed p50 511), request 16 output tokens, disable prefix caching
and MTP, and use concurrency one.

Run the pinned EvalScope 1.9.1 workload against either server endpoint:

```bash
TMPDIR="$Q3X_WORK/tmp" XDG_CACHE_HOME="$Q3X_WORK/cache" \
UV_CACHE_DIR="$Q3X_WORK/cache/uv" \
uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
  --model qwen3.6-27b-nvfp4 --api openai \
  --url http://127.0.0.1:18080/v1/completions \
  --tokenizer-path MODEL_DIR \
  --dataset line_by_line --data-source local \
  --dataset-path "$Q3X_WORK/evalscope/corpora/q3x-sharegpt-false-thinking-33.jsonl" \
  --number 32 --parallel 1 --warmup-num 1 --num-workers 1 \
  --max-tokens 16 --temperature 0 --seed 42 \
  --stream --tokenize-prompt --no-test-connection \
  --outputs-dir "$Q3X_WORK/evalscope/results/native" \
  --name native --no-timestamp
```

The `perf` extra is required by the pinned environment; the bare 1.9.1
package does not install all stress-test server dependencies.

The tokenized completions endpoint is the performance authority because it
sends the same exact false-thinking token IDs to both systems. EvalScope 1.9.1
counts TTFT from POST start to the first non-empty `choices` event. The native
gateway emits exactly one such event per committed token, merges finish into
the last token event, and emits usage separately with `choices: []`; no
role-only or finish-only event can pollute TTFT or ITL.

## Historical first external directional result

On the same MAXN Jetson AGX Orin, one warmup plus 32 measured requests gave:

The native measurement used an uncommitted functional precursor of gateway
commit `1d391a2`; its exact launch command and binary hash were not retained.
The committed gateway adds safety/protocol hardening without changing model
arithmetic or fixed request contents, but it is not byte-identical. This
provenance gap is another reason the result has directional authority only.

| Metric | Native | stock vLLM | Native / vLLM |
| --- | ---: | ---: | ---: |
| Success | 32/32 | 32/32 | — |
| Mean TTFT | 3,168.79 ms | 1,144.51 ms | **2.768687x** |
| p50 TTFT | 3,330.91 ms | 1,144.38 ms | **2.910668x** |
| p99 TTFT | 6,684.63 ms | 2,617.73 ms | **2.553598x** |
| Mean TPOT | 108.92 ms | 104.42 ms | **1.043095x** |
| Mean request latency | 4.8027 s | 2.7109 s | **1.771626x** |
| Total workload throughput | 106.1454 tok/s | 188.0494 tok/s | **56.4455%** |
| Workload prompt throughput | 102.8141 tok/s | 182.1476 tok/s | **56.4455%** |

`Workload prompt throughput` is total input tokens divided by complete run
wall time. `Total workload throughput` is total input plus output tokens
divided by that same wall time. Both include Prefill, Decode, HTTP, and request
transitions; neither is a Prefill-kernel rate. EvalScope 1.9.1 writes an
inconsistent zero `Input Throughput` into `benchmark_summary.json`, so the
table uses `workload_throughput.json`. At the recorded checkpoint, the much
larger TTFT gap than TPOT gap informed the then-current decision to investigate
Prefill first. This historical result owns no current priority or delivery
order; those belong to `CURRENT_STATUS.md` and `ROADMAP.md`.

Native and vLLM generated exactly identical text for 26/32 requests. The six
divergences are numerical/capability audit inputs, not evidence that either
runtime is the accuracy oracle. This single-process, single-round result
informed its dated roadmap decision. It may not set current priority, promote
a kernel, reset a release threshold, or serve as a publication-grade
performance baseline.

The independently repeated stock-vLLM measurement is frozen at 1,147.281 ms
mean TTFT and 182.0818 prompt tok/s. Its raw artifact identities and exact
configuration are retained in
[`qwen36-27b-evalscope-vllm-frozen-reference-2026-07-29.json`](metadata/qwen36-27b-evalscope-vllm-frozen-reference-2026-07-29.json).
Do not restart vLLM for each `local_mutation`. The historical short pair may
still detect a gross regression, and the frozen vLLM result remains a useful
reference for that exact workload. Architecture selection now uses the
target-length witness contract above. Rerun the external engine only when the
cumulative native runner approaches the applicable frozen floor or when the
model, workload, protocol, software stack, or hardware state changes.

Validate a native pair immediately after the runs with:

```bash
python3 tools/evaluation/validate_evalscope_triplet.py \
  --manifest benchmarks/evalscope/qwen36-sharegpt-false-thinking-v1.manifest.json \
  --corpus "$Q3X_WORK/evalscope/corpora/q3x-sharegpt-false-thinking-33.jsonl" \
  --baseline BASELINE_LEAF \
  --candidate CANDIDATE_LEAF \
  --output "$Q3X_WORK/evalscope/results/q3x-evalscope-direction.json"
```

For this historical validator, exit 0 means only that the candidate improves
both native mean TTFT and whole-workload prompt throughput while satisfying
the declared short-output policy. Exit 3 is a valid rejection for this short
proxy, not a Prefill architecture decision. The frozen vLLM result is not an
incremental retention threshold.

## API-first staged evolution funnel

The final runner boundary supplies the constraints; local mechanisms evolve
inside those constraints; the complete architecture returns to the API for
selection. The order is fixed:

1. Define the target request/API witness and its route, state, capacity,
   observability, accuracy, and user-visible latency contract before opening
   optimization work.
2. When a local bottleneck is identified, open a named local optimization work
   package under `REAL_MODEL_PERFORMANCE_POLICY.md`. Its real-payload component
   or short-route comparisons may retain `local_mutation` mechanisms for
   composition without requiring each one to pierce EvalScope's end-to-end
   noise. Those rules have local authority only.
3. At the package deadline, compose the mechanisms into one executable
   `architecture_candidate`. Uncomposed local wins expire into historical
   evidence rather than accumulating as a product claim.
4. Run the smallest real API sanity request to verify protocol, route, output,
   and allocation. It may reject a broken build, but its speed cannot select
   the Prefill architecture.
5. Run the cumulative native incumbent/candidate pair through the 40K, 60K,
   and 130K witness set. Those target-length API results select the
   architecture. NSys/NCU and pure Prefill intervals attribute the result.
6. Periodically compare cumulative progress with matched, frozen vLLM
   evidence; vLLM is never restarted for each local mutation and never becomes
   a production backend.
7. Freeze a selected architecture as a `release_candidate`, then run
   independent-process performance repetition, longer-output stability,
   concurrency/admission pressure, public capability, packaging, and final
   production-API conformance gates.

For a cheap historical protocol/regression proxy, use the same hash-locked
corpus and API contract as the 32-request run, but measure only the first eight
requests. The corpus
already pins 16 output tokens in every request body; EvalScope 1.9.1 preserves
that value instead of overriding it from the command line:

```bash
TMPDIR="$Q3X_WORK/tmp" XDG_CACHE_HOME="$Q3X_WORK/cache" \
UV_CACHE_DIR="$Q3X_WORK/cache/uv" \
uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
  --model qwen3.6-27b-nvfp4 --api openai \
  --url http://127.0.0.1:18080/v1/completions \
  --tokenizer-path MODEL_DIR \
  --dataset line_by_line --data-source local \
  --dataset-path "$Q3X_WORK/evalscope/corpora/q3x-sharegpt-false-thinking-33.jsonl" \
  --number 8 --parallel 1 --warmup-num 1 --num-workers 1 \
  --max-tokens 16 --temperature 0 --seed 42 \
  --stream --tokenize-prompt --no-test-connection \
  --outputs-dir OUTPUT_DIR --name RUN_NAME --no-timestamp
```

This eight-request, 16-token run has sanity/proxy authority only. It cannot
retain a `local_mutation`, select an `architecture_candidate`, promote a
`release_candidate`, or replace the target-length witness set. A true
one-token workload would require a separate hash-locked corpus and manifest;
changing only this command-line option is insufficient.

## Capability gate: first attempt invalid

A C-Eval 5-shot smoke used four public subsets and five validation examples
per subset. All 20 API calls succeeded, but every response reached the
configured 32-token output cap before emitting the required
`答案：[LETTER]` marker. EvalScope consequently reported `mean_acc=0.0`.
That number is an invalid measurement, not model accuracy: parseable answers
were 0/20 and all 20 stopped at `max_tokens`.

After target-length architecture selection admits a competitive cumulative
runner, the next capability step is to calibrate an output cap on a small
predeclared sample until the answer contract is observable, freeze it, rerun
the smoke, then remove `--limit` for the complete public suite. Capability
results do not become authoritative until answer extraction, request success,
truncation, and output-format coverage all pass.

## Remaining external gates

Before a release claim, run the actual product API and repeat native and
reference runs in independent processes with mirrored order, fixed clocks and
temperature capture; validate
every EvalScope DB for request hashes, token chunks, finish, and failures; add
a separate raw-SSE audit for the final usage event because EvalScope 1.9.1
does not retain its `choices: []` chunk in `benchmark_data.db`; add public
length buckets, long context, queue pressure, and a valid
capability suite. Each complete Prefill `architecture_candidate` must return
to the 40K/60K/130K external witness protocol before selection. The final
`release_candidate` must repeat that protocol through the deliverable API.
P513, historical short EvalScope workloads, and profiler cells remain sanity,
local, or explanatory tools rather than the project-level judge.
