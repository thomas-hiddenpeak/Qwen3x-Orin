# Reference benchmark and repeatability harness

The reference benchmark reuses one fully created `ReferenceEngine` across
warmup and measured rounds. It is a reproducibility and correctness harness
for the bounded batch-one generation path, not a throughput benchmark or a
claim about an optimized serving engine.

## Runtime API

Include `q3x/runtime/reference_benchmark.h` and call:

```cpp
q3x::runtime::ReferenceEngineCreateResult created =
    q3x::runtime::create_reference_engine(model_directory, engine_options);

q3x::runtime::ReferenceBenchmarkOptions options;
options.warmup_rounds = 1;
options.measured_rounds = 3;
options.max_new_tokens = 26;
options.prefill_chunk_size = 16;
options.logits_mode = q3x::runtime::ReferenceLogitsMode::kPredictedTokenOnly;

q3x::runtime::ReferenceBenchmarkResult result =
    q3x::runtime::benchmark_reference_engine(
        *created.value, {"first prompt", "second prompt"}, options);
```

Execution order is round-major: every prompt runs once in each warmup round,
then once in each measured round. `ReferenceEngine::generate` resets request
state before every invocation. The first result for each prompt becomes its
replay reference. Every later warmup and measured result must have identical:

- formatted prompt token IDs;
- generated token IDs and decoded text;
- stop reason;
- step count and each step's position, input token ID, result arm
  (`logits` or `prediction`), and predicted token ID.

Timing and floating-point logit values are deliberately excluded from the
replay identity. A mismatch fails with `repeatability_failure` and identifies
the prompt, phase, round, and first differing field. Generation failures retain
the nested `ReferenceEngineDiagnostic`.

Every measured invocation is retained as a `ReferenceBenchmarkSample` with
TTFT, total generation time, and all post-first-token latencies. Aggregate and
per-prompt summaries report count, minimum, median, p95, and maximum. Median is
the middle sorted value, or the mean of the two middle values for an even
count. P95 uses nearest rank:

```text
sorted[ceil(0.95 * count) - 1]
```

`ReferenceBenchmarkOptions::logits_mode` defaults to full statistics for API
compatibility and is retained in the report. The CLI benchmark explicitly
selects prediction-only because its replay signature and reports consume only
greedy token IDs.

The subsequent-token distribution flattens every post-first-token latency
from the applicable measured invocations. It is valid and has `count=0` when
all generations stop after their first token. Warmup timings never enter a
summary. The report also retains `max_new_tokens`, `stop_token_id`, and
`prefill_chunk_size`, so non-default termination and prefix-tiling policies
remain reproducible. Requested/effective chunk sizes are part of generation
replay identity; a replay that silently changes execution policy fails.

## Device-memory accounting

The harness calls `cudaMemGetInfo` immediately before the first invocation and
after every warmup and measured invocation. It reports starting, ending, and
minimum observed free bytes, device total bytes, maximum observed free-memory
drop, and the persistent start-to-end drop.

`persistent_drop_detected` is true when the persistent drop is strictly larger
than `device_memory_drop_tolerance_bytes`, which defaults to 64 MiB. This is a
warning signal in a successful report rather than an automatic benchmark
failure: CUDA libraries may retain legitimate lazy allocations. A memory-probe
failure or a changing total-memory value does fail the run with a structured
diagnostic.

`cudaMemGetInfo` is device-wide rather than process-scoped, and sampling
occurs between invocations. The fields are therefore a persistent-watermark
signal: another process can affect them, and they are not a measurement of the
peak inside one generation.

## CLI

```bash
qwen3x-orin benchmark MODEL_DIR \
  --prompt "first prompt" \
  --prompt "second prompt" \
  --max-tokens 26 \
  --warmup 1 \
  --iterations 3 \
  --max-sequence-length 512 \
  --projection-backend sm87 \
  --prefill-chunk-size 16
```

`--prompt` is repeatable. Defaults are 16 generated tokens, one warmup round,
three measured rounds, and a shared request capacity of 512 positions.
`--warmup 0` is allowed; measured iterations must be positive. Choose a
sequence capacity large enough for the longest formatted prompt plus its
decode input steps. Capacity is also bounded by the default 2 GiB request
arena; the CLI validates the complete host memory plan before loading model
weights. `--prefill-chunk-size` accepts 1 through 32 and defaults to 1. The
engine's request arena reserves activation workspace for the selected maximum
before weights are loaded. C2 through C32 tile only the prompt prefix; the final
prompt token and all decode steps remain M=1. Projection dispatch defaults to
`reference`; `sm87` explicitly selects the direct SM87 FP8/NVFP4-to-BF16
layer path and checks the active device capability before loading model
weights.

The command creates the engine once, then invokes the runtime harness. Stdout
is escaped line-oriented `key=value` data containing model directory, actual
projection backend, requested/effective prefill chunk sizes, stop token ID,
load information, replay references, every measured sample, all summaries,
and memory accounting.
Progress, diagnostics, and the persistent-memory-drop warning go to stderr.

The pure-host `reference_benchmark_control` test supplies fake generation and
memory callbacks. It covers round ordering, warmup exclusion, per-sample and
aggregate statistics, exact replay mismatches, nested failures, and memory-drop
classification without loading a model or executing a CUDA kernel.

The first matched reference/SM87 run and its machine-readable samples are in
the [Phase 3 performance evidence](PERFORMANCE_BASELINE.md). The same document
and its [C8 metadata record](metadata/qwen36-27b-c8-prefill-benchmark.json)
retain the first C1/C8 prompt-prefix comparison. The later
[C16 Tensor Core record](metadata/qwen36-27b-c16-tensor-core-prefill-benchmark.json)
retains the same-binary C8/C16/C16/C8 mirrored process order and eight measured
samples per chunk size: median TTFT 1,021.088 versus 761.037 ms and median total
generation 1,206.170 versus 946.217 ms, with strict replay preserved. These
short-prompt unlocked-clock results remain diagnostic rather than a
serving-throughput claim.
