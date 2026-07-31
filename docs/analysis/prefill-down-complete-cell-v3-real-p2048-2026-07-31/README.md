# Down complete-cell v3 real P2048 direction gate (2026-07-31)

## Decision

Retain the default-off Down complete-cell v3 candidate and replace v2 in the
explicit `cumulative-prefill-current-best` evaluation bundle.  This is an
incremental exact-K128 gain, not the architecture step expected to reach the
2,000-token/s target.

## Reproducible route

- Checkpoint: `nvidia/Qwen3.6-27B-NVFP4@0893e160`.
- Authenticated A4 publication: 400/400 K128 projections, payload SHA-256
  `57bfe2c741f5a22052a48f1ce6a15967f2d1a526a9a32898eb130467879b3660`.
- Runtime commit: `d410e73223481fb155c7bc47a2b38e4d58b791e8`.
- Same ELF SHA-256:
  `3657b934d3620e35b544915d28231ac55f2708e47cb27b8513752fba3c69cbe8`.
- API: OpenAI-compatible `/v1/completions`, batch one, 2,048 prompt tokens,
  one generated token, temperature zero, seed 42.
- Prompt: the first authenticated natural P4K corpus record truncated to
  2,048 token IDs; canonical truncated-array SHA-256
  `b3d327b76c45729f677b3f486c74f5bfaa00977aeb6d892caa3058506f98b9b0`.
- MTP disabled; cuBLASLt is not in the production route.

Both processes enabled native C64 GDN, token-parallel GDN convolution,
whole-span BF16 A/B, Down v2, FlashInfer-direct Prefill attention, Gate+Up v3,
the Attention supermatrix, and prompt-wide Full Attention preprocessing.  The
candidate process additionally enabled
`Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION=1`; selector precedence replaced
all 64 Down v2 launches with v3.  Model loading and publication authentication
were outside the request timing.

## Same-ELF result

| Route | Server Prefill runs (ms) | Mean (ms) | Prompt rate |
|---|---|---:|---:|
| Down v2 baseline | 2962.21, 2946.65, 2947.00 | 2951.953 | 693.778 tok/s |
| Down v3 candidate | 2935.68, 2932.86, 2932.55 | 2933.697 | 698.095 tok/s |

The candidate saves 18.257 ms, reduces Prefill latency by 0.618%, and improves
prompt throughput by 0.622%.  Its slowest retained run is faster than the
baseline's fastest run.  Mean end-to-end HTTP time moved from 2.955206 s to
2.937286 s.  Both routes returned the same first token/finish result and exact
usage 2,048+1.

The kernel had already passed K={128,256,384,512} bitwise tests, guard checks,
S4-MMA/`cp.async` PTX sentinels, zero-spill resource checks, and the hard SM87
resource gate: 127 registers/thread, 42,240 bytes shared/CTA, and two active
CTAs/SM.

## Target distance and next architecture step

The retained whole route is now approximately 698 tok/s at P2048.  Reaching
2,000 tok/s still requires a 2.865x throughput increase.  Single-variable
exact-K128 tuning cannot close that gap: the measured projection issue mix
requires one FP32 dequantization/FMA boundary for every logical K128 group.
The next projection core is therefore an independent, default-off K512
numerical contract, starting with Attention O, while exact-K128 remains the
fallback.  In parallel, the non-projection route must remove the
residual/RMS/A4 boundary and collapse the prompt-span GDN launch topology.
