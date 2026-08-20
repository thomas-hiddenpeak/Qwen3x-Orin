# OpenAI-compatible external evaluation

Status: evaluation gateway implemented; first external performance direction
baseline complete; release-grade repetition and a valid public capability
score remain pending.

Governing update, 2026-08-09: EvalScope is one product-facing observation
surface, not the sole definition of real Prefill. Under the
[engineering constitution](ENGINEERING_CONSTITUTION.md), confirmed cold/no-cache
Agent API behavior is an authoritative planning input. A short-prompt
EvalScope corpus, P513 timer, logger-window statistic, or unmatched endpoint
cannot invalidate the 40K--60K and 130K long-context targets. Conflicts trigger
same-workload protocol/configuration reconciliation; they do not lower the
goal. Candidate direction is checked on the closest real API path first, then
qualified statistically and attributed internally.

The project judges architecture work in the closest practical whole-product
path first: a pinned real checkpoint, public OpenAI request semantics,
[EvalScope](https://github.com/modelscope/evalscope), real prompt
distributions, and user-visible latency. Internal Prefix timing, component
benchmarks, NSys, and NCU remain essential for direction and attribution, but
they do not substitute for this external gate. EvalScope's
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

## Mandatory Orin resource preflight

Every timing-bearing native or reference run must pass the fail-closed Orin
resource preflight immediately before EvalScope starts. `tegrastats` is the
authority for sampled GR3D activity; the preflight does not use
`nvidia-smi`. It also audits visible GPU device descriptors and samples
per-process CPU time, preventing an idle-at-one-instant GPU client or a heavy
non-GPU workload from silently contaminating the run.

When the server is already resident, explicitly allow only its known PID. The
allowance includes descendants so a server worker process need not be named
separately:

```bash
PREFLIGHT_JSON="$Q3X_WORK/evalscope/results/native-preflight.json"
python3 tools/evaluation/orin_perf_preflight.py \
  --output "$PREFLIGHT_JSON" \
  --allow-pid "$SERVER_PID"
```

Omit `--allow-pid` when checking exclusivity before the server starts. A PID
allowance does not excuse GR3D activity: the default contract still requires
all five 200-ms samples to report zero utilization. Exit `0` admits the run;
exit `3` is a valid busy-host rejection; exit `1` means required telemetry
could not be collected; and exit `2` means the invocation or evidence path is
invalid. All four cases fail closed except exit `0`.

The output must be a unique file below the repository `.q3x-work` tree;
replacement requires explicit `--force`. Retain the JSON beside the EvalScope
result and start the workload immediately after admission. Do not stop or
renice a discovered consumer merely to make the check pass—wait for the
resource owner or coordinate the shared host.

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
  --prefill-chunk-size 512 --projection-backend sm87
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

This retained 4K-limited configuration and its 20--1,160-token corpus are
historical directional evidence only. They do not represent the current
long-context Agent target or bound vLLM's achievable Prefill performance.

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
  --corpus "$Q3X_WORK/evalscope/corpora/q3x-sharegpt-false-thinking-33.jsonl" \
  --baseline BASELINE_LEAF \
  --candidate CANDIDATE_LEAF \
  --output "$Q3X_WORK/evalscope/results/q3x-evalscope-direction.json"
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
