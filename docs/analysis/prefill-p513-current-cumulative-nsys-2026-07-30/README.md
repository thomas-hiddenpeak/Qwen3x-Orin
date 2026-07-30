# Current cumulative P513 Prefix profile

Date: 2026-07-30

This is one real-model Nsight Systems attribution capture of repository HEAD
`62cd808127d66737cc4b2b88692cf1876ed17fdc`, whose production code is
`19e10f6d74fa1b2d0d177bb1fb530238b624d59f`. It is diagnostic evidence, not a
latency retention sample, and it does not rerun vLLM.

## Frozen identity and protocol

- Binary SHA-256: `4e7a2e6db01526ce0b80260453b201931de0167495fb66b219772e37b375b3f9`
- ELF Build ID: `6b0925470c0c347938a5e2e0c7d2dfa491c69661`
- Model: authenticated `Qwen3.6-27B-NVFP4` checkpoint
- Prompt: canonical P513 from
  `benchmarks/qwen36-27b-sm87-prefill-prompts-v1.json`
- Generation: one token, C512, `sm87_weight_only`, all retained cumulative
  Prefill admissions enabled
- Profiler: Nsight Systems 2026.1.3, CUDA and NVTX trace, no CPU sampling
- Warmup: one separate non-profiler process before the single capture
- Oracle: token 9419, text `Hello`, 513 steps

The compact/packless GDN route was the production default. The capture used
the same native route set as the cumulative EvalScope checkpoint: NVFP4 and
FP8 Marlin, BF16 A/B, C64 GDN, token-parallel convolution, FlashInfer direct
attention, and prompt-wide residual RMS, embedding, and attention preprocess.

## Phase closure

| scope | wall ms | kernel calls | kernel sum ms | names |
|:---|---:|---:|---:|---:|
| Prefix M512 | 1217.934464 | 962 | 1209.422272 | 17 |
| Prefix M1 tail | 108.699616 | 434 | 104.812480 | 15 |
| Prefix total | 1326.634080 | 1396 | 1314.234752 | 32 |
| Finish Prefill | 5.371744 | 3 | 5.246208 | 3 |
| Prefix + finish | 1332.005824 | 1399 | 1319.480960 | 35 |

## Top kernels

The auditable top-20 aggregation is checked in as `kernel-top20.csv`; raw
demangled names remain in the hashed external artifact.

| rank | kernel | calls | total ms |
|---:|:---|---:|---:|
| 1 | Marlin FE2M1 NVFP4 Gate/Up M512 | 128 | 666.476800 |
| 2 | Marlin FE4M3 FP8 projections M512 | 208 | 332.032736 |
| 3 | `persistent_state_chunk64_resident_kernel<true>` | 48 | 49.340352 |
| 4 | `reconstruct_norm_gate_chunk64_kernel<true>` | 48 | 43.263936 |
| 5 | NVFP4 M1 Gate/Up composite | 64 | 38.662304 |
| 6 | `gqa_group_wy_chunk64_kernel<true>` | 48 | 35.901696 |
| 7 | `merged_gate_up_silu_kernel` | 64 | 26.593504 |
| 8 | FP8 QKV/Z plus BF16 A/B M1 tail | 48 | 23.391712 |
| 9 | prompt-wide residual RMS | 128 | 19.325376 |
| 10 | NVFP4 Down M1 scale6 | 53 | 16.767040 |
| 11 | FP8 row-quad M1 | 64 | 11.515264 |
| 12 | `qk_scaled_group_chunk64_kernel` | 48 | 9.236896 |
| 13 | token-parallel causal convolution | 48 | 8.219712 |
| 14 | FP8 Q/KV M1 reduction | 16 | 6.895104 |
| 15 | `normalize_qk_kernel<true>` | 48 | 5.032480 |
| 16 | BF16 A/B large-M | 48 | 3.760960 |
| 17 | prompt-wide attention preprocess | 16 | 3.723552 |
| 18 | NVFP4 Down M1 fallback | 11 | 3.545056 |
| 19 | FlashInfer single Prefill | 16 | 3.150144 |
| 20 | FlashInfer sigmoid gate | 16 | 2.718240 |

The top 20 account for 1309.552864 ms, or 99.643756% of Prefix kernel time.

## Parity decision

The two M512 Marlin rows total 998.509536 ms. With the native M1 projection
tails, the projection family is effectively level with the frozen vLLM
projection reference of 1098.201568 ms. Attention is also at reference scale:
the FlashInfer core is 3.150144 ms here versus 3.349504 ms in the frozen vLLM
capture.

The remaining whole-runner parity gap is therefore concentrated in GDN, not
large-M GEMM or attention. Current GDN kernel time is 153.211648 ms. Its three
largest stages are resident state at 49.340352 ms, reconstruction/norm/gate at
43.263936 ms, and group-WY at 35.901696 ms. The corresponding frozen vLLM
structural references are 11.600480 ms for state, 10.069408 ms for chunk-o
before its separate epilogue, and 19.286080 ms for KKT/recompute/merge.

This profile consequently locks the next work to those three structural GDN
dataflows. Further GEMM or attention tuning has no authority before that gap
is closed.

## Raw artifacts

The raw report remains outside Git at
`/tmp/q3x-current-cumulative-p513-19e10f6-nsys`. Its retained identities are
listed in `artifact-sha256.txt`.
