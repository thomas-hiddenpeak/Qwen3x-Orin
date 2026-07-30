# SM87 A4W4 paired Gate+Up admission

Status: test admission only. This route is not linked into `q3x_kernels`, the
runner, or a production selector, and it has no synthetic performance claim.

## Dataflow

One persistent 256-thread CTA owns an M64 x N64 tile. A fixed 32-CTA grid is
launched on the 16-SM Orin target, with `__launch_bounds__(256, 2)` and a
runtime fail-closed occupancy/resource query.

For each K64 group, a three-stage pipeline contains:

- one 2,048-byte signed-A4 activation tile;
- one 2,048-byte signed-W4 Gate tile;
- one 2,048-byte signed-W4 Up tile;
- one shared copy of the 64 activation scales and both 64-value weight-scale
  vectors.

The activation bytes and activation scale are fetched once per CTA/K64 group
and feed both native
`mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` branches. Each integer K64
partial is separately dequantized to FP32 before K accumulation.

The CTA computes `SiLU(Gate) * Up` into one shared M64 x N64 FP32 tile. It
never writes Gate or Up as BF16. Eight warps then quantize complete N64 rows
directly to the canonical signed-A4 `[M,N/2]` payload and BF16 `[M,N/64]`
scales consumed by the planned Down projection. The final M64 tile is
zero-filled on input and masked on output.

Static shared memory is 35,968 bytes per CTA:

- 3 x 6,528-byte pipeline stages;
- 16,384-byte M64 x N64 FP32 product tile.

The runtime query rejects the binary unless the target is the 16-SM SM87
device, local-memory/stack usage is zero, static shared memory matches the
contract, and CUDA reports at least two resident CTAs per SM.

## Validation boundary

`Q3X_BUILD_SM87_A4W4_GATEUP_PAIR_ADMISSION=ON` builds:

- a host ABI/shape/fail-closed test;
- a compute_87 PTX sentinel requiring native S4 MMA plus `cp.async` commit and
  wait instructions;
- a GPU synthetic correctness test covering a three-group pipeline, two N64
  tiles, an M65 tail, padded output strides, guard rows, and resource
  admission.

The GPU test is intentionally not run by the implementation subtask. The root
validation owns the serialized Orin run. Synthetic data may admit correctness
and resource structure only. Any performance decision must use calibrated
real-model sidecars and the real generation/API path.
