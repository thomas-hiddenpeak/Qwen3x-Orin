# EvalScope prefill checkpoint at `19e10f6`

Date: 2026-07-30

This is a whole-runner OpenAI-compatible API measurement with real model
weights. It uses the frozen eight-request corpus and does not rerun vLLM.

## Comparable result

| runner | prompt tok/s | mean TTFT ms | duration s | success |
|---|---:|---:|---:|---:|
| frozen vLLM, matched first 8 | 181.896870 | 1168.570642 | — | 8/8 |
| previous q3x `3fc24a4` | 169.666802 | 1300.800000 | 23.46 | 8/8 |
| q3x `19e10f6` | 171.688026 | 1267.810000 | 23.1874 | 8/8 |

Against the previous cumulative runner, prompt throughput improves by
2.021224 tok/s (1.1913%) and mean TTFT falls by 32.99 ms. Against the frozen
matched vLLM baseline, the remaining prompt-throughput gap is 10.208844 tok/s
(5.6124%) and mean TTFT remains 99.24 ms higher. This checkpoint therefore
does **not** meet the parity stop condition.

The fixed workload contains prompt lengths
`64,481,564,1025,695,32,713,407`; every request produces 16 tokens.

| prompt tokens | previous TTFT ms | `19e10f6` TTFT ms | saved ms | frozen vLLM TTFT ms |
|---:|---:|---:|---:|---:|
| 64 | 180.414932 | 177.841186 | 2.573746 | 210.847511 |
| 481 | 1277.450224 | 1249.816446 | 27.633778 | 1125.351516 |
| 564 | 1464.967725 | 1422.956987 | 42.010738 | 1281.485739 |
| 1025 | 2607.289215 | 2536.818654 | 70.470561 | 2343.723613 |
| 695 | 1787.946153 | 1739.890606 | 48.055547 | 1566.113445 |
| 32 | 156.698203 | 156.295359 | 0.402844 | 192.035844 |
| 713 | 1849.701420 | 1804.444708 | 45.256712 | 1659.355888 |
| 407 | 1081.903883 | 1054.385878 | 27.518005 | 969.651578 |

The length-dependent improvement and the continued advantage at P32/P64
confirm that the remaining deficit is still prefill compute slope rather than
API/server fixed overhead.

## Reproduction

Corpus:
`/tmp/q3x-sharegpt-false-thinking-33.jsonl`

Corpus SHA256:
`bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`

EvalScope result directory:
`/tmp/q3x-evalscope-cumulative-19e10f6-run1/cumulative19e10f6/parallel_1_number_8`

The server enabled the retained native admissions for large-M NVFP4, FP8,
BF16 A/B, GDN C64, FlashInfer direct attention, prompt-wide residual RMS,
token-parallel convolution, prompt-wide embedding, and prompt-wide
full-attention preprocessing. The compact/packless and group-owned GDN routes
are the production default in this commit.
