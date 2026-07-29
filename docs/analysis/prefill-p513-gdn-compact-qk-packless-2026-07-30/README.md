# P513 compact-QK packless GDN admission

Date: 2026-07-30

This admission removes the expanded H48 Q/K materialization and the global
`pack_scaled_k_v` stage from the production chunk64 GDN path. It is an
incremental native-baseline admission, not a vLLM-parity claim.

## Production data flow

- Normalize Q/K once for each of the 16 QK heads instead of replicating them
  for all 48 value heads.
- In the group-owned WY kernel, stage compact K once per three-head group,
  generate the BF16 gated-K boundary from `gamma` in shared memory, and read
  raw token-major V directly from the convolution output.
- Compute the raw QK product once per QK head, then apply the three value-head
  gamma streams while preserving the established BF16 output boundary.
- In the resident-state kernel, generate the BF16 decayed-K boundary from
  compact K and the final gamma instead of reading a global packed copy.
- Reconstruct output with the compact normalized-Q layout.

The production graph changes from seven to six kernels per native GDN layer.
`Q3X_GDN_CHUNK64_FORCE_PACKED_QKV_BASELINE=1` retains the exact preceding d51
path in the same ELF for diagnostics. The workspace capacity remains large
enough for that diagnostic route; production does not populate its packed
K/V regions.

## Real-model direction gate

Model: `/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4`

Prompt: canonical P513 from
`benchmarks/qwen36-27b-sm87-prefill-prompts-v1.json` (manifest SHA-256
`18e719e9e9fac23bb409066c664c5bab961e94d67434045d709558d38dc3e966`).
Every request generated token 9419 (`Hello`) with usage 513 + 1 and HTTP 200.

The independent-process order was B-C-C-B. Every process performed one
warmup followed by one measured request. B and C used the same server ELF
(SHA-256 `8e24c2dc342b0e9d892efc2c9173ab88202c0363c47b7d9c8adfb9e68d4f4356`);
the only route difference was the packed-baseline selector.

| order | route | warmup wall ms | measured wall ms |
|---:|:---:|---:|---:|
| 1 | B | 1362.345 | 1359.611 |
| 2 | C | 1354.917 | 1351.410 |
| 3 | C | 1353.838 | 1352.826 |
| 4 | B | 1365.744 | 1361.608 |

Paired endpoint improvements:

- pair 1: 8.201 ms, 0.603187%, 1.006068477x
- pair 2: 8.782 ms, 0.644973%, 1.006491596x
- means: B 1360.6095 ms, C 1352.1180 ms; 8.4915 ms saved,
  0.624095%, 1.006280147x

These are curl request-wall/TTFT measurements. The API did not expose Prefix
as a separate field, so no Prefix number is inferred from them.

## Exact and structural gates

The same engine instance compared the packed d51 route with the packless
candidate on the real P513 request. All BF16 words were identical:

| boundary | elements | unequal | nonfinite |
|:---|---:|---:|---:|
| transform | 1,572,864 | 0 | 0 |
| W | 3,145,728 | 0 | 0 |
| U | 3,145,728 | 0 | 0 |
| final-layer state | 786,432 | 0 | 0 |
| final-layer output | 3,145,728 | 0 | 0 |
| complete request state | 37,748,736 | 0 | 0 |

Every boundary also had maximum absolute error 0, NRMSE 0, and cosine 1.
Both routes recorded 48 native hits; boundary contracts and generation
semantics passed.

CUDA Graph capture instantiated and replayed twice with six nodes, all six
kernel nodes and no other node types. The largest reported kernel uses 252
registers per thread, zero local bytes, and admits two active CTAs per SM.

## CUDA-event attribution

One real-weight final-layer event sample measured the common core window. The
window starts after the packed route's pack kernel, so it intentionally does
not claim the frontend, state, or reconstruction savings.

| route | group WY ms | QK ms | core window ms |
|:---|---:|---:|---:|
| packed | 0.737792 | 0.289632 | 1.027424 |
| packless | 0.748928 | 0.194368 | 0.943296 |

The group kernel costs 0.011136 ms more, while grouped QK saves 0.095264 ms;
the visible window saves 0.084128 ms per layer (1.089185x). In the same event
process, measured Prefix was 1749.874096 versus 1742.801170 ms (7.072926 ms
saved), and TTFT was 1858.830061 versus 1851.946480 ms (6.883581 ms saved).

## Scope

This result locks a smaller, bitwise-exact GDN data flow as the next native
baseline. The overall prefill gap to vLLM remains the governing objective;
the 0.624% request-wall improvement is not sufficient to satisfy that target.
