# SM87 native A4W4 K64 primitive admission

Date: 2026-07-31

Base: `fd02798726098de90d2e6230d16e7b7a8406a366`

Evidence tier: T0 plus host-side synthetic ABI correctness; no performance
authority

## Result

This admission establishes the instruction and byte/fragment contracts used
by the separately gated persistent full-model A4W4 Prefill kernels. It does
not itself time or select a complete GEMM.

The compile sentinel emits compute-87 PTX and fails the build unless it finds:

```text
mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32
```

The ordinary CUDA object lowers the same primitive to:

```text
IMMA.16864.S4.S4
```

`cuobjdump --dump-resource-usage` reports 32 registers/thread, zero stack,
zero static shared memory, and zero local memory for the one-warp smoke
kernel. The runtime resource query additionally reports occupancy, but it was
not executed in this admission because the task explicitly excluded GPU use.

## Warp-level canonical ABI

- Signed S4 codes are two's-complement `[-8, 7]`.
- The even inner coordinate is stored in the low nibble; the odd coordinate
  is stored in the high nibble.
- A is canonical `[M, K/2]` bytes, row-major in logical `[M, K]`.
- B is canonical `[N, K/2]` bytes, row-major in logical `[N, K]`. At the MMA
  boundary this is the column-major `[K, N]` operand, so the warp helper needs
  no transpose or nibble shuffle.
- A scales are BF16 `[M, K/64]`; B scales are BF16 `[N, K/64]`.
- Each K64 integer partial is converted before scaling:

  ```text
  FP32 contribution(m,n,g) =
      S32 partial(m,n,g) * A_scale(m,g) * B_scale(n,g)
  ```

  Partials with different scale products cannot be summed in INT32 first.

The K64 group counts cover the model's projection K dimensions directly:
5120 -> 80, 6144 -> 96, and 17408 -> 272.

## Full-model consumer-prepacked ABI

The persistent consumers additionally use a physical N64/M64 block order:

```text
packed [ceil(outer/64), K/64, 64, 32]
scales [ceil(outer/64), K/64, 64]
```

The header owns overflow-checkable capacity/offset helpers for that layout.
The host ABI test proves those offsets are bijective across two outer blocks
and two K64 groups. N64/M64 are artifact-layout units; M32N128 CTAs may combine
or split them without changing the sidecar ABI.

Shared packed rows use `physical_half = logical_half ^ ((row >> 2) & 1)`.
The host test proves that the corresponding LDS.u32 instruction visits all 32
banks exactly once. Global sidecars remain contiguous consumer blocks; the XOR
mapping exists only in shared memory.

## Fragment and load contract

The host ABI test exhaustively proves that the lane/value mappings are
bijections over all 1024 A values, 512 B values, and 128 accumulator values.
It also proves that direct aligned 32-bit reads from canonical packed rows
produce the documented four A registers and two B registers for every lane.

The lane mapping was derived from the local CUTLASS/CuTe SM80
`SM80_16x8x64_S32S4S4S32_TN` traits and then encoded locally without taking a
CUTLASS runtime or header dependency. The persistent kernels stage consumer
blocks with `cp.async`, apply the shared-only XOR mapping, and reuse the same
MMA fragment ownership.

## Isolation and fail-closed behavior

`Q3X_BUILD_SM87_A4W4_PRIMITIVE_ADMISSION` defaults off, requires
`BUILD_TESTING=ON`, and builds a dedicated test library. The primitive is not
linked into `q3x_kernels`, the runner, or any production selector. Query and
launch reject every current device except compute capability 8.7; the device
helper traps as a second line of defense when compiled for another target.

Synthetic inputs are permitted only for later correctness/smoke coverage.
Any performance decision must start on pinned real checkpoint weights and the
real generation path under the repository's real-model evidence policy.
