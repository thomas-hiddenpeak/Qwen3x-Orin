# P513 GDN chunk-o BV64 production admission (2026-07-30)

## Scope and authority

This is the promoted output stage for the authenticated Qwen3.6 GDN shape on
SM87. Inside the explicitly admitted native C64 architecture it replaces the
compact-path `qk_scaled` plus `reconstruct_norm_gate` stages by default. The
preceding path remains available in the same ELF with
`Q3X_GDN_CHUNK64_FORCE_LEGACY_QK_RECONSTRUCT_BASELINE=1`. This promotion does
not change the separate native-C64 admission policy.

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
| chunk-o BV64 | 128 | 168 | 24,576 B | 0 | 0 | 3 |
| RMSNorm+SiLU rows8 | 256 | 27 | 0 | 0 | 0 | at least 2 |

The chunk kernel is compiled with `__launch_bounds__(128,3)`; 168 registers
keeps three 128-thread CTAs inside the SM87 65,536-register file. The final
true-`[K,N]` shared-memory route contains 12 static
`LDGSTS.E.BYPASS.128`, 160 BF16 `HMMA.16816`, 16 four-matrix A loads and 144
transposed two-matrix B loads. There is no local-memory or stack spill.

## Real production-path direction

The same release ELF and real NVFP4 checkpoint ran the canonical P513 request
with the complete cumulative production environment. Both routes entered the
native C64 path; only the candidate run selected BV64. Both produced token
9419 (`Hello`).

| route | Prefix ms | TTFT ms |
|:---|---:|---:|
| legacy QK + reconstruct | 1,553.118 | 1,666.157 |
| BV64 chunk-o + rows8 | 1,523.803 | 1,635.637 |
| saved | **29.315** | **30.520** |

The first attempted profile set only the leaf selector and therefore never
entered native C64; its zero BV64 hits made that run invalid. The corrected
profile explicitly enabled the complete native production environment and
proved 48 BV64 calls. This route check is part of the retained evidence, not
an inferred attribution.

## NSys attribution

| kernel family | calls | total ms | average us |
|:---|---:|---:|---:|
| chunk-o BV64 | 48 | 17.488224 | 364.338 |
| rows8 RMSNorm + SiLU | 48 | 6.620640 | 137.930 |
| combined | 96 | **24.108864** | - |

The frozen QK plus reconstruct total was 52.500832 ms. The 28.391968-ms
kernel reduction explains the independently observed 29.315-ms complete
Prefix improvement.

## Exactness, Graph, and mirrored direction

The real-weight exact gate compares the first native layer and the complete
request. Transform (1,572,864 BF16), W and U (3,145,728 each), first-layer
state (786,432), first-layer output (3,145,728), and complete-request GDN
state (37,748,736) all have zero unequal words, zero nonfinite values, zero
NRMSE, and cosine 1. Both routes record 48 native hits and generate the same
token and text.

CUDA Graph capture contains six kernel nodes, no other node types, and two
successful replays with the BV64 route selected.

One engine then ran real P513 in `B-C-C-B` order:

| order | Prefix ms | TTFT ms |
|:---:|---:|---:|
| B1 | 1,656.035353 | 1,661.180379 |
| C1 | 1,624.486743 | 1,628.939857 |
| C2 | 1,625.009201 | 1,629.458122 |
| B2 | 1,653.133489 | 1,657.590154 |

Both mirrored pairs are positive. Mean Prefix improves from 1,654.584421 to
1,624.747972 ms (29.836449 ms, 1.018363740x); mean TTFT improves from
1,659.385267 to 1,629.198990 ms (30.186277 ms, 1.018528293x). Every sample
has 48 native hits and passes the structural and semantic oracles.

Synthetic inputs remain suitable only for boundary correctness and smoke
coverage; they are not a performance authority.
