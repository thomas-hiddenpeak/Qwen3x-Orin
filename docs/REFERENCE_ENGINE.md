---
q3x_document:
  id: q3x-reference-engine-contract
  class: contract
  status: active
  owner: runtime-maintainers
  authority: correctness-first engine ownership, generation, timing, trace, and failure contract
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: ReferenceEngine lifecycle, generation semantics, tracing, timing, and error behavior
  review_trigger: any ReferenceEngine API, lifecycle, generation, trace, timing, or error-contract change
---

# Native reference engine contract

> **Authority boundary.** This component contract refines the generation
> boundary in the [system SDD](SDD.md) and is subordinate to it and the
> [engineering constitution](ENGINEERING_CONSTITUTION.md). Current
> implementation, capacity, default-route, qualification, performance, and
> Production lifecycle truth belongs in [`CURRENT_STATUS.md`](CURRENT_STATUS.md).
> “Reference” names the correctness-oriented API; it does not itself confer
> Production status. Mechanism-level tuning and benchmark thresholds are not
> part of this contract.

The public API is `include/q3x/runtime/reference_engine.h`. `ReferenceEngine`
is the batch-one, text-only, greedy-generation owner for the exact pinned
`nvidia/Qwen3.6-27B-NVFP4` artifact and tokenizer. It is the in-process runtime
used by the CLI and evaluation adapter.

## Ownership and creation

One heap-stable implementation object owns this lifetime chain:

1. `ResidentWeights` authenticates the checkpoint and owns its device arena.
2. Engine-owned sidecar arenas, when prepared, own derived device layouts.
3. `ModelWeights` binds checked, non-owning typed views into the resident arena
   and any attached sidecar arenas.
4. `RequestState` owns one bounded state/cache/workspace arena.
5. `ReferenceRunner` retains non-owning pointers to the exact model and
   request objects.

The sidecar owners cover FP8 output, NVFP4 Down scale6, optional Gate/Up
coupled-feed, FP8 Prefill QKV/supermatrix, build-selected Marlin admission, and
optional Down consumer-order layouts. Declaration order is a safety contract:
destruction is runner, request state, model bindings, sidecar owners in reverse
declaration order, resident arena, then tokenizer. Moving the engine transfers
its owning implementation pointer; it does not relocate objects retained by
the runner.

`create_reference_engine(model_directory, options)` creates a reusable engine.
`generate_reference(...)` is the one-shot CLI-oriented convenience path. The
one-shot path may overlap independent tokenizer parsing and resident loading,
but it joins all started work on every exit and never leaves a detached loader
or partial arena. This startup policy is an implementation choice, not a
global performance prescription.

Creation fails closed on tokenizer, resident-load, weight-binding,
request-state, runner-factory, capacity, arithmetic, or allocation errors. It
does not publish a partially usable engine.

## Accepted prompt surfaces

All public generation calls require non-empty input and converge on one reset,
capacity, Prefill, greedy Decode, stop, and result state machine:

- `generate(user_prompt)` formats exactly one user message with the pinned
  chat template, `add_generation_prompt=true`, and thinking disabled.
- `generate_chat(messages)` accepts an optional leading system message,
  alternating user/assistant messages, and a final user message. Unsupported
  roles, ordering, tool payloads, and multimodal content fail closed.
- `generate_prompt(prompt)` encodes a raw completion prompt without applying a
  chat template.
- `generate_prompt_token_ids(ids)` executes a non-empty flat list of IDs from
  the pinned 248,320-token vocabulary without a decode/re-encode round trip.

Completion requests must use the raw completion surfaces; emulating them
through the chat formatter changes token semantics and violates the contract.

## Generation state machine

The engine has no internal mutex. Its caller must serialize generation calls
and all mutation of the same engine; the current evaluation server provides
that external serialization. Each invocation begins with a successful
request-state reset. Prompt-prefix work executes without logits, the final
prompt boundary produces the first greedy token, and subsequent Decode steps
feed back the preceding prediction one token at a time. The logical control
plane distinguishes Prefill, finish-Prefill, and Decode. This contract does
not promise separate device executors, overlap, buffering depth, or a
particular tile schedule.

The requested workspace chunk is bounded by `1..512`. It is a capacity and
controller input, not a promise that every component launches that exact
width. The required sequence capacity is:

```text
prompt_token_count + max_new_tokens - 1
```

The subtraction reflects that the first generated token is predicted by the
final prompt boundary and is fed back only when another token is requested.

Generation stops after the configured stop token (default ID 248046,
`<|im_end|>`), after `max_new_tokens`, or after observer cancellation at a
committed-token boundary. The terminal stop ID remains in
`generated_token_ids` for exact replay but is omitted from `generated_text`
only when the reported stop reason is `kImEnd`.

## Logits and observer semantics

`ReferenceGenerateOptions::logits_mode` defaults to full statistics.
Prediction-only mode returns the same greedy token decision while omitting
probability statistics. Both modes reject non-finite logits and preserve stop
and trace semantics.

The optional noexcept observer runs synchronously after a token has completed,
been committed, and been decoded, but before a later Decode step begins. It
receives the token ID, zero-based generated index, incremental UTF-8 bytes,
accumulated text, elapsed step time, and stop-token flag. Returning false
records cancellation unless the same committed token already establishes the
semantic stop. String views are valid only during the callback. The observer
cannot cancel device work already executing inside Prefill.

## Result and timing semantics

`ReferenceGeneration` records rendered prompt text, prompt IDs, generated IDs,
decoded text, stop reason, requested/effective Prefill capacity, step results,
optional trace digests, timing fields, an explicit Prefill execution mode, and
the logical panel count against which completed route evidence is finalized.
A result field describing an experimental route reports observation only; it
does not promote that route or change lifecycle status.

`ReferencePrefillExecutionMode::kLegacyC512Tiled` is the compatibility
default. Its route count follows the existing controller Prefix executions
plus the separate final prompt execution when applicable. The default-off
`kWholeRequestLayerMajor` host seam instead records exactly one aggregate
whole-request Prefix duration while deriving route coverage from the immutable
C8192 topology. Thus P32, P513, P8193, and P40000 have respectively 1, 1, 2,
and 5 logical route panels even though every whole-request timing vector has
one entry; the P8193 M1 tail is not discarded. `ReferenceEngine` still leaves
the whole-request callback/finalizer/commit set unbound, so explicitly asking
it for this mode fails closed until a real executor is connected. No CLI,
line-oriented output, or OpenAI witness schema changes merely because this
host result identity exists.

Timing fields have these stable meanings:

- `prefix_execution_milliseconds`: one duration per legacy controller prefix
  execution, in order, or exactly one aggregate duration for an admitted
  whole-request callback; it is not the route-panel counter;
- `finish_prefill_milliseconds`: final retained-hidden/logits finalization,
  when used;
- `commit_prefill_milliseconds`: the successful whole-request commit callback
  interval that publishes staged request state; it is exactly zero on every
  legacy route;
- `prompt_prefill_milliseconds`: prefix execution plus finalization plus the
  whole-request commit interval when present; because it includes final-token
  readiness and publication into request state, it is not the SDD's
  server-side pure Prompt-Prefill interval;
- `time_to_first_token_milliseconds`: engine-local time through first-token
  readiness, including successful whole-request state commit; it excludes
  queueing, protocol, and response publication and is therefore not
  EvalScope's user-visible TTFT;
- `subsequent_token_milliseconds`: each later committed-token latency;
- `decode_after_first_milliseconds`: sum of those later latencies; and
- `total_generation_milliseconds`: complete generation wall time, excluding
  engine creation.

Accordingly, the exact host decomposition is:

`sum(prefix_execution) + finish_prefill + commit_prefill == prompt_prefill == TTFT`

Legacy results preserve their historical equation because `commit_prefill`
is zero. A whole-request finalizer's step timing covers only
`finish_prefill`, while readiness is not reported until the subsequent commit
callback succeeds.

Cold-start statistics separately report tokenizer, resident load, weight
binding, request-state creation, optional prepared-resource setup, runner
creation, and total wall time. Phase timings may overlap when independent
startup work overlaps, so their arithmetic sum is not required to equal wall
time. The product API must add separately attested queue/admission, pure
Prompt-Prefill, first-token, commit, and publication intervals as required by
the SDD; these legacy engine fields cannot substitute for them.

## Trace and replay contract

Trace capture must be enabled when the engine is created and requested by the
generation call. After each successfully committed captured step, the engine
hashes the embedding boundary, hidden and residual vectors for all 64 layers,
final norm, and the complete BF16 trace block. A digest records position and
input token ID.

These SHA-256 digests are exact replay gates only for the same backend and
operation order. Cross-engine or changed-operation-order comparisons require
the numerical oracle policy; digest inequality alone is not a quality verdict.

## Optional diagnostic surfaces

Fixed-position CUDA Graph screen APIs and prepared-cache options are
diagnostic/experimental ABI surfaces. Their preparation, replay, counters,
and fallback reasons are observable, but they have no authority to define the
default route or Production lifecycle state. Their detailed experiments and
historical measurements belong in evidence records rather than this active
contract.

## Installed `generate` CLI contract

```bash
qwen3x-orin generate MODEL_DIR --prompt TEXT \
  [--max-tokens N] [--trace] [--nvtx-phase-ranges] \
  [--prefill-chunk-size N] \
  [--projection-backend reference|sm87]
```

`--max-tokens` defaults to 16 and accepts `1..4096`;
`--prefill-chunk-size` defaults to 1 and accepts `1..512`; trace capture makes
the effective Prefill chunk 1. Projection backend defaults to `reference` and
`sm87` is explicit. Duplicate flags, unknown arguments, an empty prompt,
malformed integers, and out-of-range values fail before model loading.

Success writes escaped line-oriented `key=value` load, route, prompt, output,
timing, and optional trace records to stdout. Progress and structured errors
go to stderr, so redirected stdout remains one result stream. Exit code 2 is
an input/tokenization/capacity/arithmetic/decode error, 3 a
creation/load/bind/state/factory error, 4 an execution/reset/result/trace
error, and 5 a host-allocation or unclassified failure.

## Failure semantics

The engine reports a structured stage plus nested dependency/CUDA context.
Errors distinguish invalid arguments, capacity and arithmetic failures,
tokenizer/load/bind/state/factory failures, step/reset failures, missing
logits/prediction/timing, decode/trace failures, and allocation failure. A
failed invocation returns no successful generation value. Every started
owner is released through RAII; reusable-engine failure recovery follows the
runner reset/poison contract in [`REFERENCE_RUNNER.md`](REFERENCE_RUNNER.md).

The external evaluation procedure is documented in
[`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md). Current whole-product
behavior is recorded in [`CURRENT_STATUS.md`](CURRENT_STATUS.md). Immutable
engine lineage includes the
[`native reference run`](metadata/qwen36-27b-native-reference-run.json),
[`C8 Prefill record`](metadata/qwen36-27b-c8-prefill-benchmark.json),
[`C16 Prefill record`](metadata/qwen36-27b-c16-tensor-core-prefill-benchmark.json),
and [`M18 record`](metadata/qwen36-27b-nvfp4-m18-masked-m32-benchmark.json).
These historical records do not amend this contract.
