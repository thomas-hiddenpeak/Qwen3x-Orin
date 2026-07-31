# OpenAI-compatible external evaluation

Status: evaluation gateway implemented; first external performance direction
baseline complete; release-grade repetition and a valid public capability
score remain pending.

The project judges architecture work in the closest practical whole-product
path first: a pinned real checkpoint, public OpenAI request semantics,
[EvalScope](https://github.com/modelscope/evalscope), real prompt
distributions, and user-visible latency. Internal Prefix timing, component
benchmarks, NSys, and NCU remain essential for direction and attribution, but
they do not substitute for this external gate. EvalScope's
[stress-test workflow](https://evalscope.readthedocs.io/en/latest/user_guides/stress_test/quick_start.html)
is the current external performance surface.

## Evaluation-only gateway

`qwen3x-eval-server` loads one resident model and exposes a deliberately small
OpenAI-compatible subset on loopback:

- `POST /v1/chat/completions` for text-only system/user/assistant messages;
- `POST /v1/completions` for one raw string or one flat token-ID prompt;
- `GET /v1/models` and `GET /healthz`;
- non-streaming JSON and real committed-token SSE, including exact usage;
- one serialized batch-one GPU worker behind bounded ingress and inference
  queues.

Build and run it with:

```bash
cmake --build build --target qwen3x-eval-server -j
build/qwen3x-eval-server MODEL_DIR \
  --host 127.0.0.1 --port 18080 \
  --model qwen3.6-27b-nvfp4 \
  --max-sequence-length 4096 --max-output-tokens 4096 \
  --prefill-chunk-size 512 --projection-backend sm87 \
  --prefill-a4-payload A4_PAYLOAD \
  --prefill-a4-policy A4_POLICY \
  --prefill-a4-receipt A4_RECEIPT
```

The API is greedy only. Requests must explicitly provide a positive
`max_tokens` (or `max_completion_tokens`) and `temperature=0`; this prevents
the gateway from silently replacing OpenAI's sampling default. `top_p=1`,
`n=1`, a signed 64-bit audit seed, and `stop=null` are accepted. Sampling,
tools, media, logprobs, multiple choices, custom stops, and unknown fields
fail closed. Chat formatting always uses `enable_thinking=false`.

The listener is restricted to `127.0.0.1` because it has no authentication,
TLS, tenant isolation, or production admission policy. It is not continuous
batching or a production serving API. A connection carries one request.
Before the first committed token, an engine error retains its real HTTP
4xx/5xx status; after streaming begins, a runtime failure is an SSE error and
the stream closes. Disconnect and shutdown cancellation are observed at
committed-token boundaries, so a long Prefill cannot yet be interrupted in
the middle of a tile sequence. The first HTTP response header is also delayed
until a first token or early error exists; long-context clients must set a
read timeout above expected TTFT.

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

## Pinned performance workload

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
uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
  --model qwen3.6-27b-nvfp4 --api openai \
  --url http://127.0.0.1:18080/v1/completions \
  --tokenizer-path MODEL_DIR \
  --dataset line_by_line --data-source local \
  --dataset-path /tmp/q3x-sharegpt-false-thinking-33.jsonl \
  --number 32 --parallel 1 --warmup-num 1 --num-workers 1 \
  --max-tokens 16 --temperature 0 --seed 42 \
  --stream --tokenize-prompt --no-test-connection \
  --outputs-dir /tmp/q3x-evalscope-native --name native --no-timestamp
```

The `perf` extra is required by the pinned environment; the bare 1.9.1
package does not install all stress-test server dependencies.

The tokenized completions endpoint is the performance authority because it
sends the same exact false-thinking token IDs to both systems. EvalScope 1.9.1
counts TTFT from POST start to the first non-empty `choices` event. The native
gateway emits exactly one such event per committed token, merges finish into
the last token event, and emits usage separately with `choices: []`; no
role-only or finish-only event can pollute TTFT or ITL.

## Authenticated pure-Prefill matrix

The length-bucket harness refuses to start without all three authenticated A4
publication inputs. It passes them to the server explicitly and, after HTTP
readiness, requires the server log to prove all of the following before an
EvalScope request is issued:

- `prefill_a4_requested=1`, `prefill_a4_enabled=1`, and
  `prefill_a4_projections=400` (the engine sets `enabled` only after publication
  authentication and complete 400-projection attachment);
- `optimized_prefill_disabled=0`;
- the requested API capacity contract.

Run the exact-GDN baseline with:

```bash
tools/evaluation/run_native_pure_prefill_matrix.sh \
  --prefill-a4-payload A4_PAYLOAD \
  --prefill-a4-policy A4_POLICY \
  --prefill-a4-receipt A4_RECEIPT \
  --mode exact \
  ELF MODEL_DIR CORPUS_DIR OUTPUT_ROOT p2k
```

Run the structurally identical native-GDN candidate by changing only
`--mode native-gdn`. `exact` is the default. The harness removes inherited
experiment selectors from the server environment; candidate mode adds back
exactly `Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1` and
`Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1`, matching the measured
native-GDN bundle. This prevents an old shell session from silently
constructing a mixed experimental runtime. The A4,
native grouped-Q64 attention, and layer-major Prefill production defaults are
not re-enabled with legacy `Q3X_RUN_*` variables.

Use `--mode cumulative-prefill` to measure the current cumulative positive
Prefill direction. It adds exactly the two native-GDN selectors above plus
`Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION=1`, which selects the retained
whole-span BF16 A/B path. The harness first removes every inherited experiment
selector, so rejected M128 generic and Down candidates cannot leak into this
mode. `cumulative-prefill` is an evaluation bundle, not a claim that all three
mechanisms have been promoted to production defaults.

Use `--mode cumulative-prefill-down` for the next cumulative direction gate.
It starts from the same three selectors and adds exactly
`Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=1`.  The complete cell remains
eligible only for the authenticated K128, fixed Qwen3.6-27B Down shape and an
internally padded M divisible by 128.  Gate+Up complete-cell, old M128, and
short-context selectors stay cleared.  This mode records a retained
experiment; it does not make the Down cell a production default.

Use `--mode cumulative-prefill-short` with the pure-Prefill harness to measure
the retained short-context direction. It starts from the same three-selector
`cumulative-prefill` bundle and adds exactly
`Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION=1`. The harness sanitizes all
inherited experiment variables before adding those four selectors, so the
Gate+Up complete-cell selector and the generic/Down M128 selectors remain
cleared. The short selector itself is eligible only for authenticated K128
P480--P512 requests; it does not change longer prompts or imply production
promotion.

This short-only mode is intentionally absent from
`run_native_long_context_matrix.sh`. The 8K/16K/40K harness continues to
accept only `exact`, `native-gdn`, and `cumulative-prefill`.

`--dry-run` validates required paths and renders the fully quoted startup
command without starting the server or creating output. Corpus hashes are
reported as `dry-run-unverified` when a host-only fixture is used; a real run
still rejects every unpinned corpus. Both pure-Prefill and long-context
harnesses refuse an output root that already exists, so reruns require a new
root and cannot overwrite historical evidence.

The one-token workload is a TTFT/Prefill measurement: EvalScope 1.9.1 records
`TPOT=0`, `ITL=0`, and an empty per-request ITL sequence because there is no
pair of output tokens. `validate_evalscope_triplet.py` accepts that boundary
only when the completed request contains exactly one requested token. It still
rejects an incomplete request, a nonzero one-token TPOT, a nonempty one-token
ITL sequence, invalid latency, response-chunk mismatch, or summary mismatch.

For the authorized 8K/16K/40K matrix, use the same three A4 options and the
same `exact`, `native-gdn`, or `cumulative-prefill` mode with
`run_native_long_context_matrix.sh`. A live run no longer depends on a
FlashInfer marker or environment switch. Its readiness log must instead prove
the fields the current native runtime actually emits:

- `long_context_group_q64_compiled=1` and
  `long_context_group_q64_probe_selected=1`;
- `long_prefill_build_enabled=1`, the requested
  `long_prefill_hidden_capacity`, and a nonzero
  `long_prefill_projection_span_capacity`;
- authenticated A4 400/400 and `optimized_prefill_disabled=0`.

A long-context dry run says that these checks are deferred; it never reports a
BUILD/RUN admission that it did not observe from a real server readiness log.

## First external directional result

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
table uses `workload_throughput.json`. With TPOT only 4.31% apart but TTFT
2.77x apart, the external evidence makes Prefill architecture the P0
performance problem and keeps Decode frozen.

Native and vLLM generated exactly identical text for 26/32 requests. The six
divergences are numerical/capability audit inputs, not evidence that either
runtime is the accuracy oracle. This single-process, single-round result may
set roadmap priority. It may not promote a kernel, reset a release threshold,
or serve as a publication-grade performance baseline.

The independently repeated stock-vLLM measurement is frozen at 1,147.281 ms
mean TTFT and 182.0818 prompt tok/s. Its raw artifact identities and exact
configuration are retained in
[`qwen36-27b-evalscope-vllm-frozen-reference-2026-07-29.json`](metadata/qwen36-27b-evalscope-vllm-frozen-reference-2026-07-29.json).
Do not restart vLLM for each native candidate. Architecture direction screens
run only a native baseline/candidate pair; rerun the external engine when the
cumulative native runner approaches the frozen floor or when the model,
workload, protocol, software stack, or hardware state changes.

Validate a native pair immediately after the runs with:

```bash
python3 tools/evaluation/validate_evalscope_triplet.py \
  --manifest benchmarks/evalscope/qwen36-sharegpt-false-thinking-v1.manifest.json \
  --corpus /tmp/q3x-sharegpt-false-thinking-33.jsonl \
  --baseline BASELINE_LEAF \
  --candidate CANDIDATE_LEAF \
  --output /tmp/q3x-evalscope-direction.json
```

Exit 0 only means that the candidate improves both native mean TTFT and
whole-workload prompt throughput while satisfying the declared output policy.
Exit 3 is a valid external rejection. The frozen vLLM result is not an
incremental retention threshold.

## Performance-first architecture funnel

Prefill architecture work must pass the public serving path before broader
evaluation infrastructure is expanded. The order is fixed:

1. Run one real-checkpoint native baseline/candidate pair through
   `evalscope perf`, concurrency one, streaming enabled, and a short output.
   A candidate that does not improve both mean TTFT and whole-workload prompt
   throughput stops here. NSys or NCU may then explain the rejection, but a
   capability or length matrix is not built for it.
2. Retain positive native increments and periodically compare the cumulative
   runner with the frozen vLLM result above. vLLM is not restarted for each
   experiment and is never a production backend.
3. Only a cumulative native runner that approaches or beats the frozen vLLM
   short-output floor advances to longer-output `evalscope perf` stability
   runs.
4. Public capability suites, length buckets, concurrency, and release-grade
   repetition come last. They validate a competitive runner; they do not
   select an obviously noncompetitive Prefill architecture.

For a fast direction screen, use the same hash-locked corpus and API contract
as the 32-request run, but measure only the first eight requests. The corpus
already pins 16 output tokens in every request body; EvalScope 1.9.1 preserves
that value instead of overriding it from the command line:

```bash
uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
  --model qwen3.6-27b-nvfp4 --api openai \
  --url http://127.0.0.1:18080/v1/completions \
  --tokenizer-path MODEL_DIR \
  --dataset line_by_line --data-source local \
  --dataset-path /tmp/q3x-sharegpt-false-thinking-33.jsonl \
  --number 8 --parallel 1 --warmup-num 1 --num-workers 1 \
  --max-tokens 16 --temperature 0 --seed 42 \
  --stream --tokenize-prompt --no-test-connection \
  --outputs-dir OUTPUT_DIR --name RUN_NAME --no-timestamp
```

This eight-request, 16-token screen is directional only. It cannot promote a
kernel or replace the pinned 32-request short-output comparison. A true
one-token workload would require a separate hash-locked corpus and manifest;
changing only this command-line option is insufficient.

## Capability gate: first attempt invalid

A C-Eval 5-shot smoke used four public subsets and five validation examples
per subset. All 20 API calls succeeded, but every response reached the
configured 32-token output cap before emitting the required
`答案：[LETTER]` marker. EvalScope consequently reported `mean_acc=0.0`.
That number is an invalid measurement, not model accuracy: parseable answers
were 0/20 and all 20 stopped at `max_tokens`.

After the performance funnel admits a competitive cumulative runner, the next
capability step is to calibrate an output cap on a small predeclared sample
until the answer contract is observable, freeze it, rerun the smoke, then
remove `--limit` for the complete public suite. Capability results do not
become authoritative until answer extraction, request success, truncation,
and output-format coverage all pass.

## Remaining external gates

Before a release claim, repeat native and reference runs in independent
processes with mirrored order, fixed clocks and temperature capture; validate
every EvalScope DB for request hashes, token chunks, finish, and failures; add
a separate raw-SSE audit for the final usage event because EvalScope 1.9.1
does not retain its `choices: []` chunk in `benchmark_data.db`; add public
length buckets, long context, queue pressure, and a valid
capability suite. Each complete Prefill architecture milestone must return to
this same external protocol before retention or promotion. P513 and profiler
cells remain fast explanatory tools rather than the project-level judge.
