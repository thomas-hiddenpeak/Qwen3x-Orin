# Cumulative FlashInfer-direct plus Down EvalScope gate

Date: 2026-07-31 (Asia/Shanghai)

This is the first external performance matrix for the retained cumulative
Prefill bundle plus the Down complete-cell v2 and FlashInfer-direct Full
Attention.  It uses the real pinned Qwen3.6-27B-NVFP4 checkpoint, authenticated
K128 sidecar, natural ShareGPT prompts, the OpenAI-compatible streamed
`/v1/completions` endpoint, and EvalScope 1.9.1.  All 16 measured requests
succeeded.  Synthetic matrices have no performance authority here.

## Exact composition

The harness removed inherited experiment variables and enabled exactly:

```text
Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1
Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1
Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION=1
Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION=1
Q3X_FULL_ATTENTION_FLASHINFER_DIRECT=1
```

Gate+Up v3/v2, archived M128 routes, and the short-Prefill selector remained
off.  The exact ELF SHA-256 was
`8b9128b9700030ff787d15fe5f07c048a97764dd1e1daa5b00831445ef1507ec`.

Before the public matrix, a same-ELF direction pair isolated FlashInfer direct
on top of the cumulative Down bundle:

| Prompt | Native Group-Q64 | FlashInfer direct | Saved | Throughput gain |
|---|---:|---:|---:|---:|
| P2048 | 3448.06 ms | 3247.80 ms | 200.26 ms | 6.166% |
| P3840 | 6988.770 ms | 6297.835 ms | 690.935 ms | 10.971% |

The P3840 values are two-run means; the direct runs were `6298.29` and
`6297.38 ms`, while native runs were `7000.05` and `6977.49 ms`.

## EvalScope performance

Each bucket used one warmup and four measured requests, parallelism one, one
generated token, greedy decoding, and no prefix cache.

| Bucket | Avg input | TTFT | Input-only rate | EvalScope total rate | Frozen total rate | Total gain |
|---|---:|---:|---:|---:|---:|---:|
| P512 | 527.25 | 3262.65 ms | 161.602 tok/s | 161.895 tok/s | 151.412 tok/s | 6.923% |
| P1K | 1053.75 | 2227.00 ms | 473.170 tok/s | 473.554 tok/s | 429.877 tok/s | 10.160% |
| P2K | 1924.75 | 3313.18 ms | 580.937 tok/s | 581.173 tok/s | 520.911 tok/s | 11.568% |
| P4K | 3806.75 | 6516.91 ms | 584.134 tok/s | 584.245 tok/s | 522.795 tok/s | 11.754% |

The result passes the external performance-retention gate.  It does not close
the project target: the P2K/P4K input rates remain about `3.44x` below
2,000 tok/s, and a linear 40K extrapolation from P4K is still approximately
68.5 seconds.  Projection architecture remains the controlling priority.

## Capability observation

The candidate and frozen baseline databases contain the same 16 tokenized
prompts.  Ten generated first-token texts are equal and six differ:

- P512: 3/4 equal;
- P1K: 1/4 equal;
- P2K: 3/4 equal;
- P4K: 3/4 equal.

This is not a request failure, but it prevents an exact-output capability
claim.  At least one changed request cannot select Down v2 because its padded
M is not divisible by 128, so the difference is not attributable solely to
the new Down cell.  FlashInfer direct uses a numerically different attention
path, and token-level divergence near an argmax boundary is possible.

Production promotion therefore remains blocked on public task-level
capability evaluation.  Exact first-token equality is useful diagnostic
evidence but is not itself the final model-quality metric; the appropriate
gate is an externally recognized score matrix with the native cumulative path
as comparator.

## Infrastructure note

The first attempt stopped before any request because the readiness parser
required `prefill_chunk_size` and `readiness_route` to be adjacent.  The server
had added diagnostic fields between them.  Commit `53c44b8` makes both pure
and long-context harnesses accept intervening diagnostic fields, with host
regression coverage.  The successful run used a new output root and did not
overwrite the failed artifact.

## Decision

Retain Down v2 and FlashInfer direct as the next cumulative performance
candidate.  Do not enable either as a production default from this record.
Continue the three-family Attention projection supermatrix and Gate+Up v3
real-path gates; after the next material cumulative gain, rerun this exact
EvalScope matrix and add a public capability suite.
