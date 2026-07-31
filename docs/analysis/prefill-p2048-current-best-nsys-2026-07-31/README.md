# Current-best real P2048 NSys audit (2026-07-31)

## Scope

This request-scoped capture profiles one real-weight OpenAI-compatible
`/v1/completions` request with 2,048 prompt tokens, batch one, temperature
zero, and one generated token. Model and authenticated A4 sidecar loading are
outside the profiler range. The request returned `Hello`, reported exact
usage 2,048+1, and measured `prompt_prefill_ms=2996.39`.

The complete default-off candidate composition was:

```text
native GDN + token-parallel Conv + BF16 AB
Down complete-cell v2 + Gate+Up v3
FlashInfer direct Prefill attention + Attention projection supermatrix
```

The server ELF SHA256 is
`933eeda4412d867baeca468ab098824f57484722b015584b7a03bb3e216ad69b`.
The NSys report SHA256 is
`658147ae5846540b6378bcc7c47fd728698724dc241b9eaf3c78a4cfa118c449`;
its exported SQLite SHA256 is
`c23c7b6295a799e27e9453a71ef4bc3805e3d8c0f9bb174b1c5ec55bb5fc7d4b`.

## Kernel list

These are the first 20 rows of `cuda_gpu_kern_sum`, ordered by total GPU
time. Template signatures are shortened without changing kernel identity.

| # | Kernel | Calls | Total ms | Share |
|---:|---|---:|---:|---:|
| 1 | `gateup_projection_v3_warp_specialized` | 64 | 1265.078 | 42.7% |
| 2 | `down_complete_cell_v2_m128n128k128` | 64 | 502.133 | 17.0% |
| 3 | `attention_pair_supermatrix<160,96>` | 48 | 358.382 | 12.1% |
| 4 | `attention_o_supermatrix<80>` | 64 | 174.287 | 5.9% |
| 5 | `full_q_k_v_supermatrix<192,16,16>` | 16 | 104.946 | 3.5% |
| 6 | `residual_add_kernel` | 128 | 72.341 | 2.4% |
| 7 | `chunk_o_bv64_kernel` | 192 | 68.798 | 2.3% |
| 8 | `a4_quantize_bf16_k128_kernel` | 192 | 61.185 | 2.1% |
| 9 | `persistent_state_chunk64_vllm_faithful` | 192 | 59.024 | 2.0% |
| 10 | `headwise_rms_norm` | 129 | 57.008 | 1.9% |
| 11 | `causal_conv1d_silu_update_token_parallel` | 192 | 38.812 | 1.3% |
| 12 | `FlashInfer SinglePrefillWithKVCacheKernel` | 64 | 38.427 | 1.3% |
| 13 | `value_head_recompute_chunk64_kernel` | 192 | 37.884 | 1.3% |
| 14 | `value_head_solve_chunk64_kernel` | 192 | 31.896 | 1.1% |
| 15 | `full_attention_preprocess_24_4_256_64_kernel` | 2048 | 31.093 | 1.1% |
| 16 | `rms_norm_silu_rows8_kernel` | 192 | 26.258 | 0.9% |
| 17 | `bulk_gqa_flashinfer_sigmoid_gate_kernel` | 64 | 10.941 | 0.4% |
| 18 | `bf16_ab_prefill_m64_n96_k64_kernel` | 48 | 8.003 | 0.3% |
| 19 | `compact_lower_gram_chunk64_kernel` | 192 | 6.790 | 0.2% |
| 20 | `nvfp4_w4a16_decode_GEMV` | 1 | 4.397 | 0.1% |

The complete kernel sum is 2,960.018 ms. The trace also contains 192 D2D
copies totalling 12.957 ms and one 1.893-ms memset.

## Projection ledger

| Family | P2048 work | Time | Effective rate |
|---|---:|---:|---:|
| Gate+Up v3 | 46.729 TOP | 1265.078 ms | 36.938 TOPS |
| Down v2 | 23.365 TOP | 502.133 ms | 46.531 TOPS |
| Attention supermatrices | 29.549 TOP | 637.615 ms | 46.343 TOPS |
| **All projections** | **99.643 TOP** | **2404.827 ms** | **41.435 TOPS** |

The projection plane is still 81.24% of kernel time. Relative to the frozen
cumulative trace, projection latency falls from 2,788.393 to 2,404.827 ms:
383.566 ms total. Attention alone falls from 864.216 to 637.615 ms, saving
226.601 ms and directly accounting for the whole-request Attention candidate
gain.

## Global target implication

P2048 at 2,000 tok/s requires at most 1,024 ms. The current trace has
555.191 ms of non-projection kernel time plus about 36.372 ms outside the
kernel sum. Even an unattainable-perfect projection execution at the
170.459-TOPS planning S4 peak would still take 584.564 ms, making the total
about 1,176 ms or only about 1,741 tok/s.

Therefore the target is not a one-kernel problem. The next architecture must
raise projection throughput by roughly 3--4x while also removing at least
about 150 ms from Full Attention/GDN/launch and synchronization overhead.
The next NCU gate profiles the retained projection kernels to separate tensor
issue/feed limits from occupancy, scoreboard, shared-bank, and memory limits;
that evidence selects the replacement projection dataflow. Non-projection
work remains a parallel system track, not a reason to resume small parameter
scans.
