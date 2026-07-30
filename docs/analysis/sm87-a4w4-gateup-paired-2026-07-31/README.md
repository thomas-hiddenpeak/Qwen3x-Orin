# SM87 A4W4 paired Gate+Up admission

Status: test admission only. This route is not linked into `q3x_kernels`, the
runner, or a production selector, and it has no synthetic performance claim.

## Dataflow

One persistent 256-thread CTA owns an M32 x N128 tile. A fixed 32-CTA grid is
launched on the 16-SM Orin target, with `__launch_bounds__(256, 2)` and a
runtime fail-closed occupancy/resource query.

For each K64 group, a three-stage pipeline contains:

- one 1,024-byte signed-A4 activation tile;
- one 4,096-byte signed-W4 Gate tile;
- one 4,096-byte signed-W4 Up tile;
- one shared copy of the 32 activation scales and both 128-value weight-scale
  vectors.

The activation bytes and activation scale are fetched once per CTA/K64 group
and feed both native
`mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32` branches. Each integer K64
partial is separately dequantized to FP32 before K accumulation.

Global packed weights, activations, and scales use the consumer-prepacked
`[outer/64,K/64,64,*]` layout. Each CTA combines two adjacent N64 physical
blocks; N64 remains an artifact-layout unit rather than a CTA restriction.
Packed rows are XOR half-swizzled only while resident in shared memory, making
the warp fragment LDS mapping bank-unique.

The CTA computes `SiLU(Gate) * Up` into one shared M32 x N128 FP32 tile. It
never writes Gate or Up as BF16. Eight warps then quantize complete N64 rows
directly to the same consumer-prepacked A4 payload and BF16 K64 scales consumed
by Down. The final M64 physical activation block remains addressable; invalid
M rows are zero-filled on input and never written on output.

Static shared memory is 45,760 bytes per CTA:

- 3 x 9,792-byte pipeline stages;
- 16,384-byte M32 x N128 FP32 product tile.

The runtime query rejects the binary unless the target is the 16-SM SM87
device, local-memory/stack usage is zero, static shared memory matches the
contract, and CUDA reports at least two resident CTAs per SM.

## Validation boundary

`Q3X_BUILD_SM87_A4W4_GATEUP_PAIR_ADMISSION=ON` builds:

- a host ABI/shape/fail-closed test;
- a compute_87 PTX sentinel requiring native S4 MMA plus `cp.async` commit and
  wait instructions;
- a GPU synthetic correctness test covering four K64 groups (including stage
  zero reuse after the three-stage ring), two N64 physical blocks, an M65
  tail, allocation guards, and resource
  admission.

The serialized Orin run reports zero packed-code and scale mismatches, 101
registers/thread, no local memory, and two active CTAs/SM. Synthetic data may
admit correctness and resource structure only. Any performance decision must
use calibrated real-model sidecars and the real generation/API path.
