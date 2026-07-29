# FP8 W8A16 Marlin direct-port source map

This admission path is traced against the installed, frozen vLLM revision
`ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb`. It is a reference implementation
for Qwen3.6-27B on the 16-SM Orin SM87 target. It is not a cuBLAS/Lt bridge and
is not connected to production dispatch.

## Frozen source-to-local mapping

| Frozen vLLM source | Behavior used here | Local destination |
| --- | --- | --- |
| `vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py:48-103` | BF16 activation, E4M3FN weight, BF16 output, no activation/global/zero-point/act-order inputs | `include/q3x/kernels/sm87_fp8_marlin_w8a16.h`, `src/kernels/sm87/fp8_marlin_w8a16.cu` |
| `marlin_utils_fp8.py:106-217,340-354` | `[N,K]` FP8 bytes -> `[K/4,N]` GPTQ words -> 8-bit Marlin repack; scalar F32 scale -> BF16 `[1,N]`; fuse `2^120` | `prepare_sm87_fp8_marlin_projection_cuda` |
| `vllm/model_executor/layers/quantization/utils/marlin_utils.py:401-421` | single-group scale permutation | constant per-tensor scale is expanded directly; permutation is identity on equal values |
| `csrc/libtorch_stable/quantization/marlin/marlin.cu:139-153,355-541` | auto tile priority, four SM87 stages, low-work N64 override, persistent `sms=16`, FP32 reduction | `sm87_fp8_marlin_tile_config`, `launch_sm87_fp8_marlin_projection_cuda` |
| `csrc/libtorch_stable/quantization/marlin/kernel_selector.h:904-932` | exact BF16/E4M3FN/BF16/BF16, `group_blocks=-1` template instances | direct `marlin::Marlin<...>` instantiations |
| `csrc/libtorch_stable/quantization/marlin/gptq_marlin_repack.cu` and `gptq_marlin_repack_kernel.cuh` | exact 8-bit `{0,2,1,3}` byte interleave | vendored `third_party/vllm_marlin/gptq_marlin_repack_kernel.cuh` |

The vendored Marlin core is the already-audited subset introduced for the
NVFP4 direct port. Its provenance and license are recorded in
`src/kernels/sm87/third_party/vllm_marlin/README.md`.

## Qwen projection inventory

| Projection | Layers | N | K | C32 | C64 | C256/C512 |
| --- | ---: | ---: | ---: | --- | --- | --- |
| linear QKV | 48 | 10240 | 5120 | M32N256K64/T256 | M64N256K64/T256 | M64N256K64/T256 |
| linear Z | 48 | 6144 | 5120 | M32N256K64/T256 | M64N256K64/T256 | M64N256K64/T256 |
| attention O | 64 | 5120 | 6144 | M32N256K64/T256 | M64N256K64/T256 | M64N256K64/T256 |
| full Q | 16 | 12288 | 5120 | M32N256K64/T256 | M64N256K64/T256 | M64N256K64/T256 |
| full K/V | 16 each | 1024 | 5120 | M32N64K128/T128 | M64N64K128/T128 | M64N256K64/T256 |

The K/V C32 and C64 difference is required by vLLM's low-work override:
`(N/thread_n) * ceil(M/thread_m) * 4 <= 16`. This is why the admission API
selects by both token count and output width.

## Layout and scale contract

The checkpoint owns canonical row-major E4M3FN bytes `[N,K]` and one positive
F32 `weight_scale` per projection. Activations stay BF16; `input_scale` is not
used by W8A16. Load-time preparation:

1. views every four adjacent K bytes as one GPTQ word and transposes
   `[N,K/4] -> [K/4,N]`;
2. invokes the exact vendored `gptq_marlin_repack_kernel<256,8,false,false>`;
3. rounds the F32 scalar scale to BF16, expands it across N, and shifts its
   exponent by 120, matching `fp8_fused_exponent_bias_into_scales`.

The weight sidecar remains exactly `N*K` bytes. The scale sidecar is `2*N`
bytes. The existing 7.214 GB FP8 supermatrix sidecar cannot be reused: it is a
different fragment-native permutation. A production admission decision must
therefore also account for sidecar memory ownership rather than merely kernel
latency.

## Admission boundary

The first compile cell covers C32/C64/C256/C512 and all five distinct Qwen
shapes. It deliberately performs no GPU timing and does not alter
`ReferenceRunner`, `ModelWeights`, or engine load. The next gate is a pinned
real layer-0 projection correctness/timing cell, followed by a same-ELF runner
measurement and only then external EvalScope performance.
