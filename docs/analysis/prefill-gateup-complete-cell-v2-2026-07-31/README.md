# Gate+Up complete cell v2: residence and phase schedule

## Scope and evidence

This candidate is deliberately narrower than a general GEMM.  It targets the
Qwen3.6 Gate+Up shape `M=2048, N=17408, K=5120` on the 16-SM SM87 deployment,
consumes the existing signed-nibble/K128-scale publication, and directly emits
the same signed-nibble/K128-scale activation consumed by Down.  It does not
change packed order, quantization, clipping, SiLU, or the two-K64-to-one-K128
integer accumulation order.

The measured K128/M64 production baseline is 5.162--5.167 s at P2048.  The v1
M128 generic+paired route is negative at 5.179--5.186 s.  This is consistent
with its compiled resource shape rather than a missing tile constant: one warp
owns M16N128 for each projection, so Gate and Up keep 128 FP32 outputs per
thread live.  The paired image consequently uses 211 registers/thread and a
65,536-byte product tile, admits only one CTA/SM, and loses the latency hiding
that the larger M tile was meant to buy.

The local vLLM Marlin source supplies the scheduling model, not code for this
kernel.  `marlin.cuh` fixes 256 threads and a four-stage asynchronous pipeline;
`marlin_template.h` uses persistent stripes, circular shared stages, async
global-to-shared copies, and a two-deep shared-to-register feed.  The useful
lesson is that tile ownership, residence, and issue distance are one design.
Copying only Marlin's stage count into the v1 M128 ownership would retain the
same register/occupancy failure.

## Admitted residence for the SM87 NVFP4 cell

The external cell still owns M64N128, but it computes that cell as two
sequential N64 phases.  This is the smallest structural pivot that preserved
the K128 output ABI while closing both the register and stack gates:

| value | residence | lifetime |
|---|---|---|
| packed A4 A + BF16 A scale | global/L2, then a two-slot shared A ring | one logical K128 group inside one N64 phase; consumed by Gate and Up |
| packed W4 Gate/Up + BF16 B scale | global/L2, then a four-slot N64 B-phase ring | one projection phase; slots are `G(g), U(g), G(g+1), U(g+1)` |
| Gate and Up sums | registers | all 40 K128 groups of one N64 phase |
| phase-zero `SiLU(Gate)*Up` | independent shared M64N64 product tile | retained while phase one reuses the copy pipeline |
| phase-one `SiLU(Gate)*Up` | union of the now-dead phase-one pipeline and an M64N64 product tile | epilogue only |

All packed codes and scales use `cp.async.cg`: they are single-use inside one
CTA and must not displace the other operand in L1.  A is stationary in shared
across the Gate and Up phases of a K128 group.  At the persistent-scheduler
level, the target's 32 CTAs map one-to-one to its 32 M64 tiles and walk N128
cells, so a CTA retains one M phase while L2 supplies cross-cell activation
reuse.  Gate and Up weights remain independent phase streams.  A is reused
between Gate and Up within a N64 phase, but is deliberately reloaded for the
second N64 half; keeping it live across both halves reintroduced the rejected
resource lifetime.

The prologue submits two logical groups, six ordered async groups:
`A0,G0,U0,A1,G1,U1`.  `wait_group 4` publishes A0/G0 and starts Gate while U0
and group 1 remain in flight.  `wait_group 3` then publishes U0.  After both
phases consume A0, that A slot and its two B-phase slots are recycled for group
2.  The steady state repeats with six or fewer outstanding groups.  The final
logical group uses the corresponding `wait_group 1` / `wait_group 0` drain.

The schedule runs once for each N64 half.  After phase zero stores its product
to the independent tile, phase one is free to use the pipeline union.  Once
phase one's copies and MMAs are complete, its product overwrites that dead
pipeline.  A CTA barrier then publishes both products to the single epilogue,
which reduces all 128 values to one exact K128 scale and writes both physical
K64 code planes.

## Resource and traffic equations

For one logical K128 group:

* A slot: `64 * 128 / 2 + 64 * 2 = 4,224 B`.
* one N64 Gate-or-Up B phase: `64 * 128 / 2 + 64 * 2 = 4,224 B`.
* one live pipeline: `2*4,224 + 4*4,224 = 25,344 B`.
* one M64N64 product tile: `64*64*4 = 16,384 B`.
* total shared storage: phase-zero product plus the larger of phase-one
  pipeline/product: `16,384 + max(25,344, 16,384) = 41,728 B`.
* two resident CTAs consume `83,456 B`, below the repository's 96 KiB SM87
  admission budget.
* live sums: `2 * 64*64 / 256 = 32` FP32 accumulators/thread.

The occupancy recovery is not free.  B traffic is unchanged across the two
N64 halves, but A codes and scales are loaded twice.  Normalized per
M64N128/K128 cell, stage traffic rises from `4,224 + 2*8,448 = 21,120 B` to
`2*4,224 + 4*4,224 = 25,344 B`, exactly 20%.  There is still no global
Gate/Up intermediate or split-K scratch.  Real-weight timing must decide
whether two-CTA latency hiding repays the additional A traffic.

## Rejected precursors

The first complete-cell image computed the whole N128 width concurrently.
It compiled at 128 registers/thread but ptxas reported a 48-byte stack frame
and spill loads/stores, so it failed the zero-local hard gate.  A 512-thread
variant reduced register pressure to 64 registers/thread but still compiled
with a 24-byte stack frame plus spills.  Both retain the same incompatible
long-lived address/accumulator skeleton and were rejected rather than timed.

The admitted two-N64 image compiles to 118 registers/thread, 41,728 bytes of
static shared storage, zero stack, zero local memory, and two active CTAs/SM.
`cuobjdump` finds 36 `LDGSTS`, 26 dependency-barrier instructions, and 32
native `IMMA.16864.S4.S4` instructions in the kernel image.

## Admission gates

The image reports two CTAs/SM, 118 registers/thread, zero local/stack bytes,
and exactly 41,728 bytes of shared storage.  Packed output and BF16 scale bits
are byte-for-byte identical to the established K128 Gate+Up producer at the
small M64/N128/K512 shape (four logical groups, including ring recycling).
The real M2048/N17408/K5120 capacity, last-address, persistent-grid, and
output-plane layout contract also passes.

## Runtime vertical slice

The full-A4 runner contains a default-off, independent selector controlled
only by the exact value
`Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION=1`.  It is evaluated at the
shared Gate+Up launch boundary used by both C512 tiles and whole-M projection
spans.  Admission requires the immutable authenticated inventory consumer to
be K128, both branches to have the exact `[17408,5120]` model shape, the
runner's internal (possibly padded) M to be a non-zero multiple of 64, and all
eight packed/scale capacities for A, Gate, Up, and the Down publication to be
complete.  The logical natural-prompt M never reaches the kernel; the existing
ceil64 padding and explicit zero publication remain runner-owned.

When admitted, the complete cell writes the existing Down K128 packed/scale
ABI directly.  A selected launch failure is returned and is never converted
into a baseline retry.  K64 and every ineligible K128 shape retain the prior
M128/baseline order unchanged.  The route has its own worker-local enable bit
and hit counter; neither aliases nor enables the rejected M128 experiment.
Host selector, short-capacity, environment, and gate-orthogonality tests pass.
The route remains disabled by default.

## Real-path direction result

The complete cell was measured with the retained K128, native-GDN/conv, and
whole-span BF16 A/B directions in the same server. Both rejected M128
selectors remained disabled. The exact candidate ELF had SHA-256
`81b0ad05dbf01cef1be8262eaf1643733ebc8a965745d1f69d084ec9fea6697f`.

| Prompt | Server-side Prefill | HTTP total | Prompt rate |
|---|---:|---:|---:|
| P2048 run 1 | 3504.53 ms | 3.508089 s | 584.39 tok/s |
| P2048 run 2 | 3501.78 ms | 3.505383 s | 584.85 tok/s |
| P2048 mean | 3503.155 ms | 3.506736 s | 584.616 tok/s |
| P3840 direction | 7233.39 ms | 7.237619 s | 530.871 tok/s |

Against the prior cumulative P2048 server-side mean of 3554.425 ms, the cell
saves 51.270 ms and improves throughput by 1.464%. This is a repeat-positive
experiment result, so the mechanism remains available as the next opt-in
incumbent. It is not a production promotion: the gain is small enough to
require a same-ELF baseline/candidate replay and profiler hit proof before any
default change.

P3840 improves by only 3.18 ms against its prior single direction run. The
cell therefore does not satisfy the structural projection-plane objective.
Its recovered two-CTA residence is useful, but the extra A load and remaining
compressed-B decode/product traffic leave the dominant gap intact. Work must
continue on the Down and Attention projection shape families and on a more
direct compressed-weight consumer layout; this result is not a reason to
resume tile-constant scans.
