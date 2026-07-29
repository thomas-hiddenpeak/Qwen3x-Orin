# EvalScope-8 C512 Down M128xN256 positive screen — 2026-07-29

Status: positive real-model serving-path candidate; test-only and not yet a
production default.

The exact C512 NVFP4 Down shape was moved from the incumbent M128xN128 route to
an M128xN256xK128, 256-thread cell. The long-K Down specialization traverses
the four M128 token tiles consecutively for each N256 packed-weight panel
(B-stationary) and loads activations with `.cg`. It has two shared pipeline
stages, an 80-CTA grid, no persistent sidecar, and no caller workspace.

Both performance screens used the authenticated
`nvidia/Qwen3.6-27B-NVFP4` checkpoint and the real runner/API path. Synthetic
matrices have no performance authority in this record. MTP was not used.
cuBLASLt remains reference/test-only and has no production, fallback, or
promotion authority.

## Fast real-model gate

Baseline and candidate were measured with the same ELF and with the native
C64/WY GDN admission enabled on both sides. The only route difference was
`Q3X_RUN_NVFP4_PREFILL_DOWN_M128N256_ADMISSION=1` for the candidate.

| Metric | Native baseline | Down M128xN256 | Change |
| --- | ---: | ---: | ---: |
| Prefix | 2,073.989 ms | 1,928.877 ms | -145.112 ms (6.997%) |
| TTFT | 2,201.500 ms | 2,055.147 ms | -146.353 ms (6.648%) |

Both P513 requests committed 513 steps and generated token `9419`, text
`Hello`.

## First whole-product gate

The first formal external gate was deliberately limited to the simplest
EvalScope performance workload: 8 measured requests, concurrency 1, 16 output
tokens, streaming, and the OpenAI-compatible completions endpoint. It is a
direction screen for rejecting or retaining an architecture before more
expensive validation.

| Metric | Native baseline | Down M128xN256 | Change |
| --- | ---: | ---: | ---: |
| Successful requests | 8/8 | 8/8 | unchanged |
| Mean TTFT | 2,934.745982 ms | 2,844.013579 ms | -90.732403 ms (3.092%) |
| Workload prompt throughput | 108.987387 token/s | 111.196077 token/s | +2.208689 token/s (2.027%) |
| Exact generated outputs | — | 8/8 versus baseline | pass |

The benefit is concentrated where the exact C512 Down route is exercised:

| Prompt-token bucket | Requests | Mean candidate minus baseline TTFT |
| --- | ---: | ---: |
| 1–128 | 2 | -0.073170 ms |
| 129–512 | 2 | -0.040216 ms |
| 513–1024 | 3 | -145.101628 ms |
| 1025+ | 1 | -290.327571 ms |

The validator at
`/tmp/q3x-evalscope-down-m128n256-v1-short-direction.json` returned
`advance_to_internal_validation`. It verified comparable arguments, database
integrity, manifest/request matching, summary recomputation, complete
completion chunks, and identical output-set hashes. The corpus SHA256 is
`bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`.

The raw EvalScope leaves are:

- baseline:
  `/tmp/q3x-evalscope-down-m128n256-v1-baseline-8/baseline/parallel_1_number_8`;
- candidate:
  `/tmp/q3x-evalscope-down-m128n256-v1-candidate-8/candidate/parallel_1_number_8`.

## Resource and correctness boundary

`cuobjdump` reports 247 registers per thread, 512 B static shared memory,
118,784 B dynamic shared memory, and zero local/stack bytes for the Down
specialization. Its expected residency is approximately one CTA per SM.

The reference-module bitwise/guard/immutability test, Gate specialization
regression, and real-checkpoint bulk-attention P257/P513 E2E test passed with
the candidate admission enabled. These checks and the 8/8 API output parity
support retention for internal validation; they do not yet grant production
promotion.

## Reproduction

The measured binary was built with the private test-only route:

```bash
cmake -S . -B build/prefill-down-m128n256 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DQ3X_BUILD_GDN_CHUNK64_NATIVE_ADMISSION=ON \
  -DQ3X_BUILD_NVFP4_PREFILL_DOWN_M128N256_ADMISSION=ON \
  -DQ3X_E2E_MODEL_DIR=/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4
cmake --build build/prefill-down-m128n256 -j2 --target \
  qwen3x-eval-server qwen3x-orin
```

The common EvalScope command family was:

```bash
uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
  --model qwen3.6-27b-nvfp4 --api openai \
  --url http://127.0.0.1:18080/v1/completions \
  --tokenizer-path /home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4 \
  --dataset line_by_line --data-source local \
  --dataset-path /tmp/q3x-sharegpt-false-thinking-33.jsonl \
  --number 8 --parallel 1 --warmup-num 1 --num-workers 1 \
  --max-tokens 16 --temperature 0 --seed 42 \
  --stream --tokenize-prompt --no-test-connection \
  --outputs-dir OUT --name NAME --no-timestamp
```

## Decision and funnel

This candidate improves both native mean TTFT and workload prompt throughput
on the real external serving path while preserving all eight outputs, so it
advances to internal validation and may become the next cumulative native
baseline after its admission tests close.

The 32-request confirmation, long-output runs, and capability/test-set matrix
were intentionally not run at this stage. The measurement funnel requires the
short external performance gate first; spending those larger validation
budgets before a positive direction would be wasteful.

This N=8 candidate result is not compared numerically with the separately
frozen N=32 vLLM reference because the measured request sets differ. The
cumulative native runner still has not cleared the project's vLLM parity
milestone, so no parity or market-floor claim is made.
