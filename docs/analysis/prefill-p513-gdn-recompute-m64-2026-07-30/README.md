# P513 GDN WY M64 recompute admission

Date: 2026-07-30

This candidate replaces only
`value_head_recompute_chunk64_kernel` in the admitted value-head-owned WY
path. It is based on cumulative baseline `85f5f01` and was screened with the
authenticated `Qwen3.6-27B-NVFP4` checkpoint.

## Data flow

- Two warps own one full `M64 x N64` product instead of eight warps owning
  independent `N16` slabs.
- The `64 x 64` transform stays resident while one swizzled 8 KiB operand
  bank is reused for `K0`, `K1`, `V0`, and `V1`.
- Each warp retains `M64 x N32` FP32 accumulators in registers and converts
  them directly to the public BF16 W/U boundaries. The previous per-warp
  FP32 shared scratch and round trip are removed.

The SM87 build reports 138 registers per thread, zero local/stack bytes, and
16.25 KiB dynamic shared memory for the 64-thread recompute CTA.

## Real-weight exact gate

The stale packed-QKV leg in the cumulative equivalence harness returns
`cudaErrorInvalidValue` on both the unmodified and candidate binaries. For
this diagnostic only, the same harness explicitly selected its existing
group-owned packless route as baseline and the value-head-owned route as
candidate. That temporary selector change was reverted before the production
build and is not part of this branch.

At real P513, both routes hit all 48 GDN layers and generated token 9419
(`Hello`). Every captured BF16 boundary was bitwise identical:

| boundary | elements | unequal BF16 | max abs | NRMSE | cosine |
|:---|---:|---:|---:|---:|---:|
| transform | 1,572,864 | 0 | 0 | 0 | 1 |
| W | 3,145,728 | 0 | 0 | 0 | 1 |
| U | 3,145,728 | 0 | 0 | 0 | 1 |
| final-layer state | 786,432 | 0 | 0 | 0 | 1 |
| final-layer output | 3,145,728 | 0 | 0 | 0 | 1 |
| complete request state | 37,748,736 | 0 | 0 | 0 | 1 |

## Locked real P513 direction

The unmodified cumulative binary and candidate binary ran consecutively
under `/tmp/q3x-gpu-bench.lock`, pinned to CPU 11, with all cumulative
production Prefill admissions enabled and without `CUDA_LAUNCH_BLOCKING`.
Each binary performed one production-default P513/C512 request and generated
token 9419 (`Hello`).

| route | Prefix ms | TTFT ms | prefix tok/s |
|:---|---:|---:|---:|
| `85f5f01` incumbent | 1173.840222 | 1300.510703 | 436.175205 |
| M64 recompute | 1160.694428 | 1287.415594 | 441.115239 |

Prefix falls by 13.145794 ms (1.119896%, 1.011325801x), and TTFT falls by
13.095109 ms. This is a positive real-path admission against the self-hosted
baseline; it is not yet an external EvalScope or vLLM parity claim.

## Scope

Gram, solve, persistent state, chunk output, decode, and public layouts are
unchanged. The next cumulative checkpoint should remeasure the whole runner;
kernel profiling remains diagnostic after this production-path direction
gate.
