---
q3x_document:
  id: q3x-evalscope-evaluation
  class: procedure
  status: active
  owner: evaluation-maintainers
  authority: external API evaluation protocol, metric semantics, and artifact requirements
  effective: 2026-08-09
  last_reviewed: 2026-08-09
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
