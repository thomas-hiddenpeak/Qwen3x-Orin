# Attention-O K512 real-API verdict (2026-08-02)

## Scope

This audit closes the first authenticated K512 projection pilot on the pinned
Qwen3.6-27B-NVFP4 checkpoint.  Performance authority is the external
EvalScope 1.9.1 OpenAI `/v1/completions` path with real weights.  Synthetic
fixtures are used only for correctness and resource admission.

The tested source is commit `4a90d1f`.  Baseline and candidate use the same
server ELF and the same eight-selector current-best bundle; the candidate adds
only `Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION=1` and the authenticated 64-matrix
Attention-O overlay.  MTP is disabled and cuBLASLt has no production role.

Artifact identity:

- base K128 payload SHA-256:
  `57bfe2c741f5a22052a48f1ce6a15967f2d1a526a9a32898eb130467879b3660`;
- K512 manifest SHA-256:
  `314528243d208f2692498f6d62ad756e69eccd8aa8c440be7aaf77bcc9e89c55`;
- K512 policy SHA-256:
  `eaf91c43e7b7dce866b91c176924d3684f336a6c05e63e2cef7a3742967752cf`;
- K512 payload SHA-256:
  `5cd0b4ea795bf279a0a21a7f9ab8809d62a4e9d711bb88864dc64f58ca8a9321`;
- K512 payload size: `1,014,497,280` bytes, 64/64 projections authenticated.

## External EvalScope P2K result

The pinned ShareGPT P2K corpus has SHA-256
`41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af`.
Each process runs one warmup followed by four measured requests at concurrency
one and `max_tokens=1`.  Mean input length is 1,924.75 tokens.

| Route | Success | Mean TTFT | Prompt throughput | Total throughput |
|---|---:|---:|---:|---:|
| K128 current-best | 4/4 | 3,173.43 ms | 606.45 tok/s | 606.7634 tok/s |
| Attention-O K512 | 4/4 | 3,167.95 ms | 607.50 tok/s | 607.8117 tok/s |

The aggregate delta is only `+0.17%`, below a meaningful whole-product
direction gate.  Paired TTFT is not consistently positive:

| Prompt tokens | K128 | K512 | Candidate delta |
|---:|---:|---:|---:|
| 1,792 | 2,740.273 ms | 2,752.830 ms | +12.557 ms |
| 1,853 | 3,291.613 ms | 3,229.735 ms | -61.878 ms |
| 1,906 | 2,863.926 ms | 2,869.524 ms | +5.598 ms |
| 2,148 | 3,797.912 ms | 3,819.719 ms | +21.807 ms |

Three of four requests regress.  The first generated token also differs for
P1792 (`#` versus `错误`); the other three match.  This is not a capability
evaluation, but it proves that the current K512 policy is not bitwise-equivalent
to the retained K128 numerical contract and therefore cannot be promoted on
this evidence.

## Same-prompt NSys execution proof

The diagnostic capture uses the same real P1804 API request in two independent
processes.  Profiler timing is used for attribution only, never as the formal
performance result.

| Attention-O chain | Calls | GEMM | Quantizer | Combined |
|---|---:|---:|---:|---:|
| K128 baseline | 64 | 231.531104 ms | 20.151136 ms | 251.682240 ms |
| K512 candidate | 64 | 174.461280 ms | 23.955264 ms | 198.416544 ms |

The candidate executes exactly 64
`q3x_sm87_a4w4_attention_o_k512_m128n64k512_kernel` launches, each immediately
preceded by `q3x_sm87_a4_quantize_bf16_k512_kernel`.  The local chain is
`1.268454x` faster and saves `53.265696 ms`.  Server-side profiled Prefill falls
from `3307.59` to `3263.14 ms`, consistent with a real local saving that is too
small relative to the whole request.

Top kernels from the same P1804 captures:

| Baseline kernel | Calls | Total |
|---|---:|---:|
| A4W4 generic K128 projection | 272 | 1,447.034848 ms |
| Gate+Up v3 | 64 | 1,355.154048 ms |
| GDN chunk-O BV64 | 192 | 63.350752 ms |
| residual add | 128 | 62.568096 ms |
| GDN persistent state C64 | 192 | 54.802560 ms |
| A4 K128 quantizer | 192 | 53.951424 ms |
| headwise RMSNorm | 129 | 49.149312 ms |
| GDN value recompute | 192 | 34.233664 ms |
| GDN token-parallel convolution | 192 | 33.750752 ms |
| FlashInfer Prefill attention | 64 | 31.846080 ms |

| Candidate kernel | Calls | Total |
|---|---:|---:|
| Gate+Up v3 | 64 | 1,354.896160 ms |
| A4W4 generic K128 projection | 208 | 1,216.585696 ms |
| Attention-O K512 | 64 | 174.461280 ms |
| residual add | 128 | 63.338112 ms |
| GDN chunk-O BV64 | 192 | 63.181728 ms |
| GDN persistent state C64 | 192 | 54.515968 ms |
| headwise RMSNorm | 129 | 50.661184 ms |
| GDN token-parallel convolution | 192 | 35.077696 ms |
| GDN value recompute | 192 | 34.598656 ms |
| A4 K128 quantizer | 128 | 33.944736 ms |
| FlashInfer Prefill attention | 64 | 32.006112 ms |
| A4 K512 quantizer | 64 | 23.955264 ms |

This proves that `cp.async`, the K512 quantizer, and the K512 Attention-O core
are in the real Prefill execution path.  It also shows why an O-only result
cannot be a qualitative system improvement: Gate+Up and the remaining 208
projection launches still dominate.

## Decision and architectural carry-forward

The Attention-O K512 route remains default-off and is rejected as a standalone
production promotion.  Its overlay must not be expanded family by family or
kept resident beside a second full sidecar.

One mechanism is retained as evidence: reducing four K128 floating-point scale
boundaries to one K512 boundary provides a real `1.268x` local speedup.  The
next architecture may reuse that mechanism only inside a replacement
shape-specific projection macro-core and a single mixed/full publication.  A
valid next vertical slice must cover the dominant MLP plane, use real model
weights, enter through the OpenAI API, and clear an external EvalScope
whole-request gate before statistical or NCU expansion.

Evidence roots:

- `/home/rm01/q3x-k512-evalscope-p2048-baseline-4a90d1f`;
- `/home/rm01/q3x-k512-evalscope-p2048-candidate-4a90d1f`;
- `/home/rm01/q3x-k512-nsys-real-api-4a90d1f`.
