# P513 GDN chunk-o BV64 parity candidate (2026-07-30)

## Scope and authority

This is one complete structural candidate for the authenticated Qwen3.6 GDN
shape on SM87. It replaces the frozen compact-path `qk_scaled` plus
`reconstruct_norm_gate` output stages only when
`Q3X_GDN_CHUNK64_CHUNK_O_BV64` (or its same-ELF test selector) is enabled.
Production remains unchanged until a real-checkpoint P513 direction gate and
the subsequent correctness suite admit it.

The dataflow was derived by inspecting both the vLLM/FLA `chunk_o.py` source
and the exact Triton artifact selected by vLLM for P513 on this Orin:

- `BT=BK=BV=64`
- grid `(2, 9, 48)` for the vLLM P513 padded span
- block size 128, four warps, two compiler stages
- 162 registers, 24 KiB shared memory, zero local memory

No Triton-generated source or binary is incorporated. The implementation is a
fixed-shape native CUDA/inline-PTX specialization.

## Exact stage boundary

Inputs already established by the compact native prefix are:

- `Q`: BF16 normalized and scaled by `1/sqrt(128)`, compact H16 layout
- `K`: BF16 normalized, compact H16 layout
- `H`: BF16 chunk-boundary state, H48 by value/key dimensions
- `Vnew`: BF16 corrected values, H48 token/value layout
- `g`: FP32 cumulative gate, H48 token layout

For each value head, chunk, and BV64 half, the kernel computes:

```text
S_fp32   = Q @ H^T
A_fp32   = Q @ K^T
A_bf16   = BF16(causal(A_fp32) * exp(g_query - g_source))
O_fp32   = S_fp32 * exp(g_query) + A_bf16 @ Vnew
O_bf16   = BF16(O_fp32)
```

`O_bf16` is written directly to the token-major `[T,48,128]` boundary. An
independent rows-8 kernel then performs the established exact D128 RMSNorm,
BF16 weight multiply, and SiLU gate in the same lane and shuffle order as the
frozen path.

## CTA ownership and on-chip lifetime

The chunk kernel uses grid `(2, chunks, 48)`, block 128, one CTA per
value-head/BV64 half:

1. Q, K, and H occupy three 8 KiB swizzled shared tiles. `cp.async.cg`
   (`LDGSTS.E.BYPASS.128` in SASS) loads both BK64 panels.
2. Initial warp ownership is query-M16 (`[4,1]`). Each warp accumulates the
   full QH and QK rows simultaneously in FP32.
3. After Q/K/H die, shared memory becomes a 16 KiB FP32 QH layout exchange
   plus the FP32 gate vector.
4. QK is gated, causally masked, rounded to BF16, and placed in an 8 KiB A
   tile. V is loaded coalesced and transposed only in shared memory into the
   adjacent 8 KiB tile.
5. Final warp ownership is `[2,2]` query/value. The A@V tensor-core phase
   accumulates directly into the reloaded FP32 QH fragments and publishes the
   BF16 output without a global score matrix or transpose boundary.

This deliberately rejects the archived V128/group-three-head skeleton: no CTA
serially owns four query panels, no value-head grouping is introduced, and the
two BV64 halves retain independent scheduling phase.

## Static build evidence

CUDA 13.3, release SM87 compilation completed for `q3x_kernels` and the native
engine E2E executable. `cuobjdump --dump-resource-usage` reports:

| kernel | block | registers/thread | shared | local | stack | static active CTA/SM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk-o BV64 | 128 | 164 | 24,576 B | 0 | 0 | 3 |
| RMSNorm+SiLU rows8 | 256 | 27 | 0 | 0 | 0 | at least 2 |

The chunk kernel is compiled with `__launch_bounds__(128,3)`; 164 registers
keeps three 128-thread CTAs inside the SM87 65,536-register file. SASS contains
24 static `LDGSTS.E.BYPASS.128`, 160 BF16 `HMMA.16816`, 16 row-major and 144
transposed `LDSM` instructions. There is no local-memory or stack spill.

## Gate order

The next action is intentionally a real-model gate, not a synthetic timing
screen:

1. P513, one real generation, same ELF, frozen compact baseline then BV64
   candidate; require valid generation semantics and positive prefix/TTFT.
2. If positive, run exact/characterization boundaries, graph capture/replay,
   then B-C-C-B timing and NSys attribution.
3. If negative or invalid, retain the profiler evidence for root-cause
   analysis and do not build a larger statistical harness around it.

Synthetic inputs remain suitable only for boundary correctness and smoke
coverage; they are not a performance authority.
