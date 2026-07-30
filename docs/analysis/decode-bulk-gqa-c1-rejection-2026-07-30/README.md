# Decode bulk-GQA C1 direct-reuse rejection — 2026-07-30

Status: **REJECTED**. The existing Prefill QT2 online-softmax kernel must not
be merged into the production Decode path at `token_count=1`.

## Scope

Commit `6b134703fdeb3461a75718d1c2645284cef2e429` admitted the existing bulk
causal GQA/Gate QT2 kernel to the real SM87 generation path behind
`Q3X_RUN_DECODE_BULK_GQA_C1_ADMISSION=1`. The candidate applied only after
the established sequence-length-64 fused Decode path became ineligible. It
preserved the online-softmax implementation and the attention BF16 rounding
boundary before sigmoid Gate. It did not change Prefill, fixed-position CUDA
Graphs, projections, or MTP.

At C1 the QT2 grid is `ceil(1 / 2) * 4 = 4` CTAs: one CTA for each KV head.
The otherwise valid second query row is masked during query load, causal
accumulation, and output writeback.

## Real serving-path direction gate

EvalScope 1.9.1 measured eight requests after one warmup with concurrency
one, streaming enabled, greedy decoding, and 16 output tokens. The candidate
used the real Qwen3.6-27B-NVFP4 checkpoint through the OpenAI-compatible
server.

- Candidate artifact:
  `/tmp/q3x-evalscope-decode-bulk-c1-6b13470-run1/decodebulkc1/parallel_1_number_8`
- Candidate-versus-`c92a2ef` validator:
  `/tmp/q3x-evalscope-decode-bulk-c1-6b13470-vs-c92a2ef.json`
- Corpus SHA-256:
  `bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`

| Metric | Native `c92a2ef` | Bulk C1 candidate | Candidate delta |
| --- | ---: | ---: | ---: |
| Mean TTFT | 1167.935631 ms | 1178.645483 ms | **+10.709853 ms** |
| P50 TTFT | 1294.804216 ms | 1306.132829 ms | +11.328613 ms |
| P99 TTFT | 2370.425952 ms | 2399.747448 ms | +29.321496 ms |
| Mean TPOT | 108.587628 ms | 117.668983 ms | **+9.081355 ms** |
| Prompt throughput | 177.913533 tok/s | 169.028894 tok/s | **-8.884639 tok/s** |
| Wall time | 22.376038 s | 23.552186 s | +1.176148 s |
| Exact generated outputs | 8/8 | 8/8 | unchanged |

All eight requests had slower TTFT. The prompt-length buckets report mean
candidate-minus-baseline TTFT of `+2.315401`, `+6.625705`, `+12.825038`, and
`+29.321496` ms for 1--128, 129--512, 513--1024, and 1025+ prompt tokens,
respectively. Output-set SHA-256 remained exactly
`262701c931d4f51828217d0e4c71b98ad1e3d7e6d30174f7b3127ea6cba3357e`.
The validator returned exit code 3 and `reject_direction` after passing
argument comparability, database integrity, manifest matching, summary
recomputation, completion completeness, and exact-output checks.

## Decision

The direct C1 reuse is correct but structurally under-parallel. Four total
CTAs cannot fill the 16-SM Orin GPU while each CTA serially traverses the
complete causal K/V span. Sharing K/V across the six query heads and removing
the probability scratch do not compensate for that grid-level starvation;
the increasingly negative TTFT with prompt length confirms the failure on the
actual runner rather than only in a microbenchmark.

This direction is closed and must not be merged into production. No further
GPU profiling is required for this candidate. A future long-context Decode
attention design must expose multiple CTAs per KV head, such as split-KV with
an online-state reduction, and must start with a new real serving-path
direction gate. This rejection applies to the four-CTA QT2 C1 mapping, not to
online GQA attention as a broader architecture.
