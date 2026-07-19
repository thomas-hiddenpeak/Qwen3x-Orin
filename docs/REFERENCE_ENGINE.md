# Native reference engine

`q3x::engine` is the correctness-first, text-only generation surface for the
exact pinned `nvidia/Qwen3.6-27B-NVFP4` artifact. It is batch-one, decodes one
token at a time, and may execute prompt prefixes in bounded tiles of up to
16 tokens. The implementation remains a bring-up and oracle-alignment path,
not a large-prefill or serving engine.

## Ownership and creation

`ReferenceEngine` owns the complete native lifetime chain in one heap-stable
implementation object:

1. `ResidentWeights` authenticates the three official shards and loads the
   text tensors into one CUDA arena.
2. `ModelWeights` binds checked, non-owning views into that exact resident
   object.
3. `RequestState` owns one bounded state/cache/workspace CUDA arena.
4. `ReferenceRunner` retains pointers to the exact model and request objects.

Member destruction is the reverse order: runner, request state, model views,
then resident weights. Moving `ReferenceEngine` transfers only the owning
`unique_ptr`; it never relocates the objects retained by the runner.

`create_reference_engine` creates a reusable engine from explicit resident and
request options. `generate_reference` is the one-shot production/CLI path. It
loads `MODEL_DIR/tokenizer.json` first, formats the prompt to derive the exact
request capacity, then creates the native chain without parsing the tokenizer
a second time. Link installed consumers to `q3x::engine`.

For multi-round latency collection and strict replay checks on one reused
engine, see the [reference benchmark harness](REFERENCE_BENCHMARK.md).

## Prompt and greedy policy

Each call accepts exactly one user text message. It uses
`Tokenizer::format_qwen36_chat` with `add_generation_prompt=true` and
`enable_thinking=false`. The resulting token IDs are the runner inputs.

Prefill preserves token semantics while optionally batching projections:

- with `prefill_chunk_size=1` (the default), every prompt-prefix token executes
  all 64 layers with `compute_logits=false`, preserving the original order;
- with `prefill_chunk_size=2..16`, the prefix is split into bounded layer-major
  tiles. Quantized projections consume all tile rows together, while causal
  Conv/GDN updates, RoPE positions, K/V writes, and GQA lengths remain ordered
  per token;
- the final prompt token executes with `compute_logits=true` and produces the
  first greedy token;
- each later step consumes the previous prediction and produces the next one;
- decoding stops after token ID `248046` (`<|im_end|>`) or
  `max_new_tokens`, whichever comes first.

The required request capacity is therefore
`prompt_token_count + max_new_tokens - 1`: the first generated token is a
prediction of the final prompt step and is never fed back unless another token
is requested.

`ReferenceGenerateOptions::logits_mode` defaults to
`kFullStatistics`, preserving the complete `ReferenceStepLogits` result for
library callers. A caller that only needs greedy IDs may explicitly select
`kPredictedTokenOnly`; those compute steps expose `ReferenceStepPrediction`
instead of a partial or fabricated logits-statistics object. The production
CLI selects prediction-only because it prints token IDs/text but no logit
probabilities. Token choice, non-finite rejection, stop behavior, and trace
digests are identical between the two modes.

`generated_token_ids` retains an observed terminal `248046` so exact-token
fixtures can compare it. `generated_text` decodes only the preceding prefix,
so the stop marker is not exposed as user text. The stop ID is removed from
the text view only when the controller actually reports `im_end`; a coincidental
last ID under another stop reason is retained.

## Timing and allocation

Cold-load statistics separately report tokenizer parsing, authenticated
resident loading, model binding, request-state creation, runner creation, and
total wall time. Resident byte/chunk/copy counters, binding counters, and
request-arena size are also retained.

`prompt_prefill_milliseconds` is the sum of each whole prefix-tile timing and
the final prompt step.
`time_to_first_token_milliseconds` has the same value because the first token
becomes available after the final prompt step. Later decode latencies are kept
both individually and in `decode_after_first_milliseconds`.

All device allocations happen during resident, request, or runner creation.
The token loop performs no device allocation. Result vectors, decoded text,
timing records, and optional SHA-256 trace strings are host-side result
storage.

## Trace contract

An engine must be created with trace enabled before a generation call can
request capture. Immediately after every successfully committed step—and
before another step can invalidate the pinned trace view—the engine hashes:

- the embedding boundary;
- hidden and residual boundaries for each of all 64 layers;
- final norm;
- the complete 130-vector BF16 trace block.

`ReferenceTraceDigest` retains position and input token ID plus all boundary
digests. This permits locating the first divergent layer against the checked-in
BF16 oracle without retaining a model-sized activation history.

These hashes are exact replay gates only for the same backend and operation
order. Native-versus-vLLM comparison is diagnostic: use the recorded samples,
statistics, and explicit tolerances rather than requiring equal hashes.

## CLI

```bash
qwen3x-orin generate MODEL_DIR --prompt TEXT \
  [--max-tokens N] [--trace] \
  [--prefill-chunk-size 1..16] \
  [--projection-backend reference|sm87]
```

`N` defaults to 16 and the CLI admits 1 through 4096. Duplicate flags,
unknown arguments, empty prompts, malformed integers, and values outside the
range fail before model loading. Prefill chunk size defaults to 1. The request
arena is created for the selected maximum, and a later generation request may
not exceed that capacity. Trace capture forces an effective chunk size of 1 so
its per-token boundary ordering remains unchanged; stdout reports both the
requested and effective values. Projection dispatch defaults to `reference`;
`sm87` is an explicit, default-off selection for direct FP8/NVFP4-to-BF16
layer projections. `ReferenceEngineOptions` and `ReferenceOneShotOptions`
expose the same strongly typed policy. Engine creation verifies that `sm87`
is running on compute capability 8.7 before loading resident model weights.

On success stdout contains escaped, line-oriented `key=value` records for
cold-load statistics, the actual `projection.backend`, requested/effective
prefill chunk sizes, rendered prompt/IDs, complete generated IDs, decoded text,
stop reason, timings, and optional trace hashes. Loading progress and all
structured diagnostics go to stderr, so stdout can be redirected as one atomic
result stream. Exit code 2 denotes input/tokenization errors, 3 denotes
creation/load failures, 4 denotes execution/trace failures, and 5 denotes host
allocation failure.

## Target-device evidence

On 2026-07-18 the installed native CLI completed the full pinned checkpoint
path on the Jetson AGX Orin with:

```bash
qwen3x-orin generate MODEL_DIR \
  --prompt "用一句话解释 CUDA 是什么。" \
  --max-tokens 26 --trace
```

The formatted prompt produced the exact 19 IDs in the
[BF16 greedy fixture](../tests/fixtures/qwen36-27b-nvfp4-greedy-bf16.json).
The native runner then produced all 26 expected IDs and the exact UTF-8 text:

> CUDA 是 NVIDIA 开发的一种并行计算平台和编程模型，旨在利用 GPU
> 的强大算力来加速通用计算任务。

It stopped at ID `248046` with `stop_reason=im_end`; 19 prompt steps plus 25
feedback steps produced 44 total runner steps.

The complete structured record is checked in as
[`qwen36-27b-native-reference-run.json`](metadata/qwen36-27b-native-reference-run.json).

The later C8 gate runs the same 19-token prompt prefix as `8+8+2`, keeps the
final prompt/logits step scalar, and reproduces all 26 output IDs, decoded
text, stop reason, and 44 transcript steps exactly. Its machine-readable
aggregate performance record is
[`qwen36-27b-c8-prefill-benchmark.json`](metadata/qwen36-27b-c8-prefill-benchmark.json).

The C16 runtime (`dda4e3a`) and GDN (`c90f37e`) gates schedule that prefix as
`16+2`; the nineteenth prompt token and every decode step remain scalar. With
the SM87 backend, M9..M15 quantized projections split into M8 plus the
remainder. At exactly M16, the FP8 (`e7283d6`) and NVFP4 (`33948e3`) production
shapes use canonical-layout decode-to-BF16 Tensor Core kernels, with two M8
launches retained for other shapes or alignments. The complete C16 run
reproduces the same 19 prompt IDs, 26 output IDs, decoded text, stop reason,
and 44 steps exactly. Its implementation and diagnostic performance record is
[`qwen36-27b-c16-tensor-core-prefill-benchmark.json`](metadata/qwen36-27b-c16-tensor-core-prefill-benchmark.json).

The original C1 single-run timings are evidence, not performance targets:

| Measurement | Observed value |
| --- | ---: |
| Cold load total | 213,845.054 ms |
| Authenticated resident load | 212,683.392 ms |
| Request arena | 82,505,216 bytes |
| Prompt prefill / TTFT | 20,559.187 ms |
| Decode after first token | 28,652.488 ms |
| Total generation | 49,211.675 ms |
| Mean of 25 later token steps | about 1,146.100 ms/token |

The trace captured these native summaries at the two oracle localization
positions:

- position 18 full:
  `a780bede09bb74be107323444ec2be4b2165c89721624025308525c31c1dc226`;
  final norm:
  `1b4edfc36c9dea31414dc8a1668c8ad10cad9e574ff32871a18c8e0d28131b7b`;
- position 19 full:
  `76b767786fa92b117833e79062aab1e4ea8942e0c61ac8fd7b6840744fbc85f1`;
  final norm:
  `266c82ecdd4df8e7b3aad8e4de7a78b8c622e3f15129bbfda24bc0c85a23f680`.

The native boundary hashes are deliberately not claimed bitwise-equal to the
vLLM boundary fixture. The native projection path preserves each checkpoint
tensor's independent ModelOpt scale, while vLLM's fused projection path
requantizes fused operands; the native runner also updates BF16 GDN state one
token at a time, while vLLM prefill uses chunk execution. Those differences
change intermediate low bits and hashes even though the final greedy IDs,
decoded text, and stop decision match exactly.

## Verification scope

`reference_engine_control_test` exercises the pure host controller without
CUDA or model files. It gates scalar and `8+8+2` prefix scheduling, trace-to-C1
fallback, sequential inputs, prefix-logit suppression,
first-token timing, stop/max behavior, capacity checks, nested runner errors,
missing result fields, and the terminal-stop text rule. CUDA runner tests cover
factory and primitive behavior separately. An installed-package consumer also
configures against and links `q3x::engine`. The fixed-prompt 27B token/text gate
is now passed; repeatability across more prompts and tolerance-based
native-versus-vLLM boundary characterization remain release gates.

The same exact 19/26-ID, text, stop, and 44-step gate is available as the
conditional `reference_engine_e2e` CTest. Configure with
`-DQ3X_E2E_MODEL_DIR=MODEL_DIR` (or set that environment variable). Its
projection policy defaults to `reference`; configure
`-DQ3X_E2E_PROJECTION_BACKEND=sm87` to gate the optimized path. Without the
external pinned model directory it exits with the standard skip code 77. The
test executable also accepts an optional prefill chunk argument, so target
validation can run the same oracle at C1, C8, and C16.
