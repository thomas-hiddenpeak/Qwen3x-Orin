# P513 Prefix kernel audit — 2026-07-28

This directory freezes the exact kernel aggregation requested during the
external Prefill review. The workload is batch one, a rendered 513-token
prompt, one generated token, a 512-token Prefix tile, the production
`sm87_weight_only` projection backend, and fixed MAXN clocks on one Jetson AGX
Orin. Only kernels wholly contained in the
`q3x.prefill.prefix_tile` NVTX interval are included.

Both profiles use the same Nsight Systems 2026.1.3 command shape and
`--trace=cuda,nvtx --sample=none --cpuctxsw=none
--cuda-graph-trace=node`. The benchmark has no warmup and one measured round.
Formal end-to-end promotion numbers come from a separate mirrored B-C-C-B
protocol; the one-shot profiles are for attribution.

## Frozen identities

| Role | Source | Binary SHA-256 | ELF build ID |
| --- | --- | --- | --- |
| Baseline | `6e3b364` test-only Z admission; production Z still generic M128 | `2c40925214aeb0f683c38386375f2aaec8ee166a4a0095111c86e504ffcf8163` | `110de1b261db0c4c89e5763679f3ba7073778edc` |
| Production | `9967985` exact-C512 Z production promotion | `47bcddd46aecd8eb3472b3c745d2da1bdb9a995d69e14f6ef0a4ef96f88fbd11` | `4c815fa783428c1feaf60ced63e2abd72cb4e8c5` |

Raw `.nsys-rep` and SQLite files are not checked in. Their identities are:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| Baseline report | 854,631 | `9ab16e3f2cb6b92e80fcca33a238e5c132985916839dc3b3a2442ba7b343f4c0` |
| Baseline SQLite | 2,428,928 | `3bffa2f15baf092c07e981752e609f7669a07dbc86e32681a3fbc8fbab1c8296` |
| Production report | 854,655 | `9c99295ed413799d9fa785b0b8308d685cbafffca49a390fa2d45ca6488ae0f9` |
| Production SQLite | 2,424,832 | `75490fc7ce8ed356590aced9b3f17e28a7205f49fee8d9ce4c030db2d502c3b4` |

The checked-in exact summaries are
[baseline-6e3b364-prefix-kernel-top20.csv](baseline-6e3b364-prefix-kernel-top20.csv)
and
[production-9967985-prefix-kernel-top20.csv](production-9967985-prefix-kernel-top20.csv).
There are only 17 distinct demangled kernel names, so “top 20” contains all of
them. The extraction query is
[prefix-kernel-top20.sql](prefix-kernel-top20.sql).

## Sanity and result

| Prefix measure | Baseline | Production | Change |
| --- | ---: | ---: | ---: |
| NVTX wall time | 2,589.067872 ms | 2,568.257216 ms | -20.810656 ms |
| GPU kernel sum | 2,580.601152 ms | 2,559.205920 ms | -21.395232 ms |
| Kernel launches | 8,209 | 8,209 | 0 |
| Distinct demangled names | 17 | 17 | 0 |
| Z projection, 48 launches | 121.420864 ms | 99.290464 ms | -22.130400 ms |

The exact production substitution is visible in the CSV: the baseline
`fp8_w8a16_whole_chunk_m128_n128_k64_b_reuse_kernel<6144,5120,4>` row is
replaced 48-for-48 by
`fp8_w8a16_whole_chunk_z_m128_cp_async_canonical_xor_register_feed_kernel`.
The remaining small difference is ordinary one-shot noise in unchanged rows.

The production profile also answers the architecture questions directly:

- large whole-chunk projections are custom M128-by-N128/K64 kernels, not the
  historical M16 generic projection path;
- exact linear-attention QKV executes the production `cp_async_register_feed`
  kernel 48 times;
- full attention executes one
  `bulk_causal_gqa_sigmoid_gate_24_4_256_kernel` per full-attention layer, 16
  launches total; this is the native online-softmax path;
- exact GDN recurrence is still tiled at C16, producing 48 × 32 = 1,536
  launches; causal Conv1D is also 1,536 launches, while post-GDN
  RMSNorm/SiLU(Z) is a separate 48-launch kernel boundary. RMSNorm and
  SiLU(Z) are already fused with each other; the missing boundary is between
  that fused epilogue and GDN recurrence.

Nsight names show which function ran; SASS proves asynchronous-copy code
generation. `cuobjdump` of the frozen production binary shows
`LDGSTS.E.BYPASS.128`, `LDGDEPBAR`, and `DEPBAR` inside the selected Z
function. The extracted 312,725-byte function disassembly hashes to
`6ce6c3236edd1dcd52df1db6128fe6122275db3e658bfa9196c60d5885c9a064`.
The pre-promotion production QKV function independently contains the same
SM87 async-copy opcode family.

## Deployment history

- Bulk C256/C512 online-softmax attention entered the production runner in
  `1f7d6be` (`feat: route bulk causal GQA Prefill`), primarily in
  `src/kernels/reference/decode_ops.cu` and
  `src/runtime/reference_runner.cpp`.
- C512 QKV register feed entered production in `6ab59ea`
  (`feat: promote C512 FP8 QKV register feed`), primarily in
  `src/kernels/sm87/weight_only_gemv.cu` plus its bounded sidecar ownership and
  dispatch files.
- Z register feed was test-only at `6e3b364`. It entered the exact aligned
  C512 production route in `9967985`; C256 and valid 8-byte-but-not-16-byte
  activation pointers retain the generic M128 fallback. Z adds no sidecar and
  changes no model-weight ownership.

## Next bounded experiments

The installed CUDA 13.3/cuBLASLt 13.5 stack cannot execute FP8 Lt matmul on
SM87 (FP8 requires compute capability 8.9 or newer) or FP4 Lt matmul (compute
capability 10.0 or newer). A cuBLASLt comparison therefore requires BF16
unpacking. Full-model canonical-plus-BF16 dual residency would require
47.680664 GiB of BF16 sidecars and 66.447344 GiB before activation/workspace,
which exceeds this machine's 61.404 GiB MemTotal. It is not a production
candidate.

The bounded library comparison is two real-checkpoint cells only: FP8 Z C512
and NVFP4 Gate C512. Each first measures persistent-BF16 Lt GEMM with unpack
outside timing as an ideal ceiling, then inclusive JIT unpack plus Lt. The
persistent ceiling must reach at least 1.8x for FP8 Z or 1.7x for NVFP4 Gate
before the JIT path is attempted; inclusive JIT must reach 1.22x in every
mirrored round before any overlap or full-model work. These are prospective
stop-loss gates, not measured results.

In parallel, one test-only exact-C16 composite will preserve the current GDN
recurrence and BF16 boundaries, stage its 16-by-128 raw result per value-head
CTA, then execute the already-fused RMSNorm/SiLU(Z) locally. It removes only
48 standalone epilogue launches, not the 1,536 recurrent launches, and avoids
576 MiB of logical global write-plus-read traffic over 48 layers. The complete
`32×GDN + epilogue` chain must be bit-exact and at least 1.03x faster in every
mirrored round or the experiment stops test-only.
