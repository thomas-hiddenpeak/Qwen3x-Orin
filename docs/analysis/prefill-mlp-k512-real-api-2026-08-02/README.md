# Authenticated MLP K512 real-API direction gate (2026-08-02)

## Decision

The first complete MLP K512 route is **directionally retained but not
promoted**.  It is a real whole-runner improvement and a useful structural
baseline, but it is not the qualitative jump required by the 2,000 prompt
token/s target.

The external EvalScope/OpenAI result on the fixed natural P2K corpus is:

| Route | Success | Mean TTFT | Prompt throughput | Total throughput |
|---|---:|---:|---:|---:|
| K128 current-best comparator | 4/4 | 3,173.4 ms | 606.45 tok/s | 606.7634 tok/s |
| MLP K512 candidate | 4/4 | 2,782.6 ms | 691.62 tok/s | 691.9824 tok/s |
| MLP K512 + natural ceil128 | 4/4 | 2,720.5 ms | 707.40 tok/s | 707.7676 tok/s |
| Cumulative change | - | -452.9 ms (-14.27%) | +16.65% | +16.65% |

The candidate therefore passes the experiment gate against the incumbent,
but remains 2.83x below 2,000 prompt token/s.  The ceil128 route change is
itself +2.28% over the first K512 result.  No synthetic timing is used in this
decision.

## Real publication and production-shaped route

The independently requantized overlay contains all 192 MLP projections:
64 Gate, 64 Up, and 64 Down.  Gate and Up are `[17408,5120]`; Down is
`[5120,17408]`.  The authenticated payload is 8,623,226,880 bytes.

```text
manifest SHA256  6ab8818b34256646b4f1aca3d6cae1bb80425d30df1ab589a2c744a811351583
policy SHA256    5e4b31d0e93cd1ae13d1aebd66a9fd75151e6d3fb5ab448c1f7650ac127ef8e5
payload SHA256   541480dcad50227288530b22ed24e5984cef99f513d1df17ede8d4b702a2d5ec
layout           sm87_s4_n64_packed_k64_scale_k512_mlp_v1
```

The service startup contract proved 400/400 authenticated K128 base
projections, 192/192 authenticated MLP K512 overlays, the payload SHA, and an
enabled optimized Prefill dispatcher.  Every measured request used the real
checkpoint through `/v1/completions`; EvalScope 1.9.1 supplied one warm-up and
four serial measured requests with one output token.

The route is:

```text
post-attention norm
  -> K512 activation quantization
  -> fused Gate+Up K512 macrocell (two workspace windows)
  -> split-plane K512 product quantization
  -> Down K512 macrocell
  -> residual
```

Gate/Up share one activation quantization, so their activation clip ratios
must narrow to the same `float`.  Policy parsing and programmatic attachment
both now fail closed on a mismatch.

## Request-scoped NSys result

One natural 1,853-token request was captured through the same OpenAI server.
Model and sidecar loading are outside the `cudaProfilerStart/Stop` range.  The
request reported `prompt_prefill_ms=2840.46`.

```text
server ELF SHA256  f3f4b7d6927aa720c16a7f29d77f3535bb3eaa8d9190a60f6e8500a32add50c1
NSys report SHA256 6c5ad6bc861c7ee9d088dc39014b551449534b8dcd84617fbab3f2d08979d4cc
SQLite SHA256      22dbf23e16e2b90823102ea99b0857db2443d3ba36a886cf25fcdb3bb55c0a73
kernel CSV SHA256  8db849d77ff4438d012662900e49b5347c8bfb5af342264a587273b78a38db6d
```

The first 20 kernel families by total GPU time are:

| # | Kernel | Calls | Total ms | Share |
|---:|---|---:|---:|---:|
| 1 | `gateup_k512_m64n128_macrocell` | 128 | 888.499 | 31.5% |
| 2 | `prefill_gemm_m64n256_shared_k128` | 208 | 845.835 | 30.0% |
| 3 | `down_k512_m128n128_macrocell` | 64 | 514.770 | 18.2% |
| 4 | `a4_quantize_bf16_k512_split` | 64 | 67.187 | 2.4% |
| 5 | `residual_add` | 128 | 65.428 | 2.3% |
| 6 | `chunk_o_bv64` | 192 | 63.095 | 2.2% |
| 7 | `persistent_state_chunk64` | 192 | 54.298 | 1.9% |
| 8 | `headwise_rms_norm` | 129 | 51.870 | 1.8% |
| 9 | `a4_quantize_bf16_k128` | 128 | 38.159 | 1.4% |
| 10 | `causal_conv1d_token_parallel` | 192 | 35.978 | 1.3% |
| 11 | `value_head_recompute_chunk64` | 192 | 34.525 | 1.2% |
| 12 | `FlashInfer SinglePrefillWithKVCache` | 64 | 32.598 | 1.2% |
| 13 | `value_head_solve_chunk64` | 192 | 29.062 | 1.0% |
| 14 | `rms_norm_silu_rows8` | 192 | 24.223 | 0.9% |
| 15 | `a4_quantize_bf16_k512` | 64 | 20.179 | 0.7% |
| 16 | `full_attention_preprocess_prompt_wide_128` | 64 | 13.717 | 0.5% |
| 17 | `bulk_gqa_flashinfer_sigmoid_gate` | 64 | 10.001 | 0.4% |
| 18 | `bf16_ab_prefill_m64_n96_k64` | 48 | 7.149 | 0.3% |
| 19 | `bf16_gemv_pair_m16_projection_fused` | 144 | 6.764 | 0.2% |
| 20 | `compact_lower_gram_chunk64` | 192 | 6.291 | 0.2% |

The top three projection kernels consume 2,249.104 ms, or 79.7% of kernel
time.  Gate/Up plus Down alone consume 1,403.269 ms.  This proves that the
remaining gap is still projection architecture, not attention softmax or GDN
micro-tuning.

The 208 generic K128 launches are the complete non-MLP attention projection
inventory.  The natural 1,853 rows were padded only to M=1,856, while the
retained supermatrix requires M128.  Exact P2048 profiles therefore
overstated supermatrix coverage for natural traffic.

The follow-up changed whole-span K128 padding to ceil128 and repeated the
same external corpus.  The 1,853-token measured request fell from about
2,832 ms to 2,585 ms, saving about 247 ms.  Requests with 1,792, 1,906, and
2,148 tokens were already on M128 boundaries after the old padding and stayed
essentially unchanged.  This explains why a large per-affected-request route
repair becomes +2.28% over the four-request matrix rather than a universal
gain.

## Next structural replacement

The v1 K512 cells reduce scale application frequency but spend 83--128 KiB
of shared memory and run at one CTA/SM.  The retained K128 projection cells
run at two CTAs/SM.  The next implementation is not a stage-count scan:

1. Gate+Up becomes M64N64 with four Gate and four Up warps.  A K64/K128 copy
   ring remains below the two-CTA shared-memory budget; S32 partials accumulate
   through one K512 scale group before a single FP32 scale application.
2. Down becomes shape-specific M128N64 with per-warp M32N32 ownership.  The
   smaller ownership keeps the simultaneous S32 and FP32 accumulator sets
   below 128 registers/thread while preserving two CTAs/SM.
3. Natural prompt spans now use ceil128 internal projection rows so Attention
   supermatrices remain active on the affected EvalScope requests.
4. The next promotion decision is again OpenAI API plus external EvalScope.
   NCU is diagnostic only and must confirm two CTAs/SM, zero spill, tensor-pipe
   activity, issue activity, bank conflicts, and scoreboard stalls.

Evidence directories:

- comparator: `/home/rm01/q3x-k512-evalscope-p2048-baseline-4a90d1f`;
- candidate: `/home/rm01/q3x-mlp-k512-evalscope-p2048-systematic-pilot`;
- natural ceil128 candidate:
  `/home/rm01/q3x-mlp-k512-evalscope-p2048-natural-m128`;
- profiled EvalScope run:
  `/home/rm01/q3x-mlp-k512-evalscope-p2048-systematic-profile`;
- request trace:
  `/home/rm01/q3x-mlp-k512-nsys-p2048-systematic-request2.nsys-rep`.
