# 8K/16K/40K real-API runner preflight

Date: 2026-07-31

Status: host/source preflight only. **No 8K, 16K, or 40K GPU performance run
is claimed by this record.** The implementation base is `fd02798`; the
long-context harness added with this record still requires a separately
approved real-corpus manifest, a matching real corpus, a qualifying ELF, and
an explicit GPU execution.

## Locked capacity envelope

The request planner's absolute sequence capacity is 262,144 positions
([`include/q3x/runtime/request_state.h`](../../../include/q3x/runtime/request_state.h),
enforced by
[`src/runtime/request_state.cpp`](../../../src/runtime/request_state.cpp)).
For the retained C512 plan, host tests already lock these byte totals:

| server capacity | planned arena bytes | runner arena cap | phase |
| ---: | ---: | ---: | --- |
| 8,192 | 705,593,344 | 705,593,344 | max_tokens=1 |
| 16,384 | 1,245,085,696 | 1,245,085,696 | max_tokens=1 |
| 40,960 | 2,864,349,184 | 3,221,225,472 (3 GiB) | max_tokens=1 and 16 |

A 40,000-token prompt plus at most 16 generated tokens requires 40,015
resident positions (`prompt + max_new_tokens - 1`) and plans 2,802,085,376
bytes. The rounded 40,960 deployment capacity plans 2,864,349,184 bytes. The
evaluation server's default request cap is still 2 GiB in
[`include/q3x/server/evaluation_server.h`](../../../include/q3x/server/evaluation_server.h),
so P40K must pass `--request-max-arena-bytes 3221225472`; silent fallback is
not permitted.

## Existing boundaries the run must expose

- The retained short length-matrix script is intentionally fixed at
  `--max-sequence-length 4096`; it is preserved unchanged in
  [`tools/evaluation/run_native_pure_prefill_matrix.sh`](../../../tools/evaluation/run_native_pure_prefill_matrix.sh).
- The rejected layer-major experiment is limited to P<=4096 in
  [`include/q3x/runtime/reference_engine.h`](../../../include/q3x/runtime/reference_engine.h).
  The long runner explicitly removes its RUN gate, and the evaluation server
  now reserves its two hidden slabs only under BUILD+RUN. Readiness must report
  `long_prefill_run_requested=0 long_prefill_hidden_capacity=0`, so the exact
  arena totals above cannot silently include rejected architecture storage.
- The optimized Decode GQA split-KV selector ends at 4,096 in
  [`include/q3x/runtime/decode_ops.h`](../../../include/q3x/runtime/decode_ops.h).
  P40K `max_tokens=16` therefore audits the current long-position Decode
  behavior; it must not be presented as evidence that the <=4K optimized
  Decode route extends to 40K.
- The admitted long-context grouped-Q64 Prefill-attention selector covers
  eligible P1024+ tiles through sequence length 40,960 in the same header.
  [`src/server/evaluation_server.cpp`](../../../src/server/evaluation_server.cpp)
  now reports compiled, requested, and eligible-probe selection at readiness;
  the harness rejects anything other than `1/1/1`.

## Evidence entry and fail-closed policy

[`tools/evaluation/run_native_long_context_matrix.sh`](../../../tools/evaluation/run_native_long_context_matrix.sh)
parameterizes P8K/P16K/P40K server capacity, output ceiling, arena cap, and
EvalScope network timeouts. P40K supports a one-token Prefill authority phase
and a separate 16-token agent-cold-start phase. Every server launch prints the
actual arguments and the exact `/healthz` readiness route.

There is deliberately no long-corpus generator. The required external
manifest follows
[`benchmarks/evalscope/qwen36-long-context-prefill-v1.manifest.schema.json`](../../../benchmarks/evalscope/qwen36-long-context-prefill-v1.manifest.schema.json)
and must identify authorized complete natural conversation prefixes. The
runner requires an independently supplied manifest SHA-256 and validates the
tokenizer files, corpus bytes, source identities, prompt IDs, canonical
request hashes, lengths, request count, phase equality, and no duplicate
prompts before it takes the GPU lock. Missing authorized data is a rejection,
not an invitation to pad, repeat, truncate, concatenate, or synthesize text.

The checked host dry-run is only an interface/capacity validation. Its final
line is `performance_evidence=0`; it cannot close any performance gate.
