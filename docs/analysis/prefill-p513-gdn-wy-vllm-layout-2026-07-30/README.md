# P513 value-head-owned GDN WY admission

Date: 2026-07-30

This admission replaces the single group-owned WY kernel with a fixed-shape
three-kernel SM87 hierarchy modeled on the executed FLA/Triton C64 data flow.
It is a real-model production-baseline improvement, not a vLLM-parity claim.

## Production data flow

- A compact Gram kernel launches one CTA per `(chunk, qk_head)`, streams two
  K64 panels with `cp.async`, computes KKT once, and stores only the ten live
  lower 16x16 tiles.
- A solve kernel launches one CTA per `(chunk, value_head)` and keeps the
  triangular FP32/BF16 solve in a packed-tile shared layout.
- A recompute kernel launches one CTA per `(chunk, value_head)`, keeps the
  transform resident, reuses one operand bank for K then V, and publishes W/U
  with 128-bit stores.

At P513/C512 this is 128 Gram CTAs followed by 384 solve and 384 recompute
CTAs per GDN layer. The production Graph changes from six to eight kernels per
layer. `Q3X_GDN_CHUNK64_FORCE_GROUP_OWNED_WY_BASELINE=1` retains the preceding
same-ELF route for diagnosis.

The SM87 build reports no stack/local spill for any new kernel. The compiled
resource tuples `(threads, registers/thread, dynamic shared)` are
`(128,56,8 KiB)`, `(128,40,20.5 KiB)`, and `(256,48,32.25 KiB)`.

## Real-weight P513 gate

Model: `/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4`

Protocol: one engine, batch one, P513, C512, max one generated token, route
warmups B then C, followed by measured B-C-C-B. All cumulative production
prefill admissions were enabled. Every measured route hit all 48 GDN layers
and generated token 9419 (`Hello`).

| order | route | Prefix ms | TTFT ms | hits |
|---:|:---|---:|---:|---:|
| B1 | group-owned | 1312.985928 | 1317.450206 | 48 |
| C1 | value-head-owned | 1307.083351 | 1311.553997 | 48 |
| C2 | value-head-owned | 1308.295867 | 1312.744049 | 48 |
| B2 | group-owned | 1322.406617 | 1327.392629 | 48 |

Both mirrored pairs are positive: 5.902577 ms and 14.110750 ms. Mean Prefix
falls from 1317.6962725 to 1307.6896090 ms, saving 10.0066635 ms
(0.759406%, 1.007652170x). The corresponding 512-prefix-token rate rises
from 388.557 to 391.530 token/s. Mean TTFT falls by 10.2723945 ms.

## Exact and structural gates

The preceding group-owned route and the new route were compared in one engine
on the same real P513 request. Every BF16 word is identical:

| boundary | elements | unequal |
|:---|---:|---:|
| transform | 1,572,864 | 0 |
| W | 3,145,728 | 0 |
| U | 3,145,728 | 0 |
| final-layer state | 786,432 | 0 |
| final-layer output | 3,145,728 | 0 |
| complete request state | 37,748,736 | 0 |

All boundaries also have maximum absolute error 0, NRMSE 0, cosine 1, and no
nonfinite values. Both routes recorded 48 hits and passed generation
semantics. CUDA Graph capture contains eight kernel nodes, no other nodes,
and instantiated and replayed twice.

## Scope

This locks the value-head-owned hierarchy as the new native baseline. The
latest external EvalScope checkpoint remains the separately recorded
`19e10f6` result until the cumulative branch reruns it; this focused P513 gate
does not infer a new external score or claim that the vLLM stop condition has
been met.
