# P2048 prompt-wide Full Attention preprocess direction (2026-07-31)

## Result

The existing default-off selector
`Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION=1` was restored
to the complete current-best real-model bundle after the P2048 NSys audit
showed that the incumbent still launched 2,048 preprocess kernels totalling
31.093 ms.

One authenticated Qwen3.6-27B-NVFP4 OpenAI request, batch one, P2048,
temperature zero, and one generated token measured:

| Route | Server Prefill | Prompt rate |
|---|---:|---:|
| Previous retained bundle, three-run mean | 2971.597 ms | 689.192 tok/s |
| Plus prompt-wide preprocess, first direction run | 2954.730 ms | 693.128 tok/s |

The first direction run saves 16.867 ms and improves whole-request prompt
throughput by 0.571%. It returned the same `Hello` token/text and exact usage
2,048+1. HTTP completion time was 2.961539 s.

This is a positive real-generation-path result, so the selector is retained
in the cumulative optimization bundle. It is not treated as a primary
architecture breakthrough and does not justify additional local tuning. The
existing admission already has bitwise, CUDA Graph, resource, and real-P513
evidence recorded in
`docs/metadata/qwen36-27b-prefill-full-attention-preprocess-prompt-wide-128-admission.json`.
The next full EvalScope gate will measure it as part of the cumulative bundle
rather than as an isolated feature.

## Current target distance

The new single-run direction is approximately 693 tok/s. Reaching 2,000 tok/s
still requires P2048 latency at or below 1,024 ms, a further 2.886x reduction.
The projection exact-K128 v4 and the non-projection consumer-boundary/GDN
tracks remain the structural work; this recovered preprocess budget simply
removes one known launch-amplification omission.
