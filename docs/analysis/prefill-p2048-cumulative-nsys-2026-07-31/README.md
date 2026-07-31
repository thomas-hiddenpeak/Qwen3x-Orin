# Current cumulative P2048 NSys kernel audit

Date: 2026-07-31 (Asia/Shanghai)

This profile isolates one real 2,048-token OpenAI `/v1/completions` request
with one generated token. Model loading is outside the CUDA profiler range.
The server uses the authenticated K128 A4 publication plus native GDN/conv and
whole-span BF16 A/B; complete-cell-v2, both rejected M128 routes, short-Prefill
admission, MTP, and cuBLASLt production use are disabled.

The request reported `prompt_prefill_ms=3570.52`. The exact profiler ELF
SHA-256 is
`c7651088f72ee20011e83119e8969c9a7183036edf7c5905e78af3804412e7bf`.
The NSys report SHA-256 is
`195b9bae8ee4c19aa19bce4eccc1712881a342793e3a051f7d847377dc68bc0b`;
its exported SQLite SHA-256 is
`b8ccb25ecac6d371f4a74ec94af828c925a681e35ede3dc412c933e3fd83bab5`.

## Capture command

```text
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --output=/tmp/q3x-nsys-cumulative-k128-p2048-e595493 \
  .../qwen3x-eval-server ... --profile-request-index 1
```

`--profile-request-index 1` is compiled only in diagnostic/test builds. The
default and production build do not call the CUDA profiler API.

## Kernel list

These are the first 20 rows from `nsys stats --report cuda_gpu_kern_sum`, in
descending total GPU time. Names are shortened only where the namespace and
full C++ signature add no ambiguity.

| # | Kernel | Calls | Total ms | Share |
|---:|---|---:|---:|---:|
| 1 | `q3x_sm87_a4w4_prefill_gemm_m64n256_shared_k128_kernel` | 272 | 1471.572 | 41.5% |
| 2 | `q3x_sm87_a4w4_gateup_paired_k128_kernel` | 64 | 1316.821 | 37.1% |
| 3 | `bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<false>` | 48 | 229.054 | 6.5% |
| 4 | `residual_add_kernel` | 128 | 72.779 | 2.1% |
| 5 | `chunk_o_bv64_kernel` | 192 | 69.581 | 2.0% |
| 6 | `q3x_sm87_a4_quantize_bf16_k128_kernel` | 192 | 61.429 | 1.7% |
| 7 | `persistent_state_chunk64_vllm_faithful_kernel<false>` | 192 | 59.402 | 1.7% |
| 8 | `headwise_rms_norm_kernel<true,false>` | 129 | 57.268 | 1.6% |
| 9 | `causal_conv1d_silu_update_token_parallel_compact_qk_kernel` | 192 | 39.599 | 1.1% |
| 10 | `value_head_recompute_chunk64_kernel` | 192 | 37.804 | 1.1% |
| 11 | `value_head_solve_chunk64_kernel` | 192 | 32.444 | 0.9% |
| 12 | `full_attention_preprocess_24_4_256_64_kernel` | 2048 | 31.198 | 0.9% |
| 13 | `rms_norm_silu_rows8_kernel` | 192 | 26.613 | 0.8% |
| 14 | `bulk_causal_gqa_sigmoid_gate_24_4_256_c512_group_q64_kernel<true>` | 16 | 19.346 | 0.5% |
| 15 | `bf16_ab_prefill_m64_n96_k64_kernel` | 48 | 8.021 | 0.2% |
| 16 | `compact_lower_gram_chunk64_kernel` | 192 | 6.820 | 0.2% |
| 17 | `nvfp4_w4a16_gemv_bf16_scale_codebook_row_quad_k5120_activation_staged_kernel` | 1 | 4.395 | 0.1% |
| 18 | `prepare_gate_kernel` | 192 | 2.211 | 0.1% |
| 19 | `embedding_gather_prompt_kernel` | 4 | 0.253 | <0.1% |
| 20 | `bf16_greedy_argmax_kernel` | 1 | 0.015 | <0.1% |

The capture also contains 192 device-to-device copies totalling 13.307 ms, one
CUDA memset totalling 1.902 ms, and negligible host transfers.

## Exact projection-family attribution

The generic kernel uses a fixed 32-CTA persistent grid, so grid dimensions do
not identify the matrix. Its five disjoint duration/count clusters do, because
they exactly match the fixed model inventory:

| Projection family | Shape `N x K` | Calls | Total ms |
|---|---|---:|---:|
| Full Attention K and V | `1024 x 5120` | 32 | 19.906 |
| Linear Z/O plus Full O | `6144 x 5120` or `5120 x 6144` | 112 | 415.323 |
| Linear QKV | `10240 x 5120` | 48 | 304.653 |
| Full Attention Q | `12288 x 5120` | 16 | 124.334 |
| MLP Down | `5120 x 17408` | 64 | 607.355 |

Therefore the current projection plane is:

| Family | Work at P2048 | Time | Effective rate |
|---|---:|---:|---:|
| Gate+Up | 46.729 TOP | 1316.821 ms | 35.486 TOPS |
| Down | 23.365 TOP | 607.355 ms | 38.469 TOPS |
| Attention projections | 29.549 TOP | 864.216 ms | 34.192 TOPS |
| **All projections** | **99.643 TOP** | **2788.393 ms** | **35.735 TOPS** |

Projection kernels consume 78.6% of all kernel time and achieve only about
21% of the 170.459 TOPS planning S4 peak. Reaching the 2,000 tok/s system goal
requires roughly 0.70--0.78 s for the complete projection plane, or about
128--142 TOPS. This remains the decisive architecture problem.

## Other component budgets and corrected priority

- Full Attention main kernels plus preprocess: 279.597 ms. The source performs
  online max/denominator rescaling and output accumulation in one kernel; the
  profile contains no separate score/softmax/value kernel chain. It still
  launches four C512 attention kernels per Full Attention layer and 2,048
  preprocess kernels.
- Native GDN core stages: 274.473 ms. The prior 1.6-second GDN budget came from
  an older exact path and is no longer a valid optimization priority.
- The state plus BV64 output stages total only 128.983 ms; including all D2D
  copies gives a 142.290-ms absolute removable upper bound. Consequently the
  earlier prompt-span GDN requirement to save 400 ms is physically impossible
  on the current path and is withdrawn. A standalone fused mechanism may be
  retained only with zero local/stack, bitwise correctness, and a later real
  saving of at least 50 ms; projection work remains ahead of it.
- Whole-span BF16 A/B is now 8.021 ms. Further work there is not justified.

The executed projection kernels are the K128 `cp.async` implementations, so
asynchronous copies are genuinely on the real path. The current Full Attention
is likewise an online-softmax bulk kernel, not the old naive two-pass path.
