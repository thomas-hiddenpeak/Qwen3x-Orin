# Gate+Up projection v3: projection-specialized warp crews

## Why the v2 cell is not the final skeleton

The real P2048 NSys capture assigns `1,316.821 ms` to 64 Gate+Up launches,
or `35.486 TOPS` for `46.729 TOP` of work.  Complete-cell v2 preserves two
CTAs/SM but improves the full request by only `1.464%`.  Its M64N128 external
cell is internally two serial N64 phases.  Each phase reloads the same A
M64K128 plane and all eight warps compute Gate and then Up serially.

For one M64N128/K128 cell, v2 stages:

- two A copies: `2 * 4,224 = 8,448 B`;
- Gate and Up B: `4 * 4,224 = 16,896 B`;
- total: `25,344 B`.

`cp.async` can overlap these bytes, but it cannot remove the second A copy or
the six CTA synchronization points per logical K128 group.  A deeper version
of the same phase skeleton therefore does not address the structural loss.

## Global dataflow chosen for v3

The v3 test cell keeps M64N128, but divides the CTA by projection rather than
dividing N by time:

```text
                      one M64K128 A stage
                  global/L2 --cp.async.cg--> shared
                                  |
                         +--------+--------+
                         |                 |
                  Gate warps 0..3     Up warps 4..7
                  M16N128 each         M16N128 each
                         |                 |
               Gate accumulators     Up accumulators
                         +--------+--------+
                                  |
                 dead pipeline -> shared M64N128 exchange
                                  |
                     SiLU(Gate) * Up, exact K128 pack
```

Gate and Up weights remain independent streams.  Each is copied directly
from global/L2 into its own N128 shared plane with `cp.async.cg`; neither is
routed through L1 and neither is globally decoded.  A codes and A scales are
copied once and consumed by both warp crews during the same K iteration.
Every thread owns only one projection's 64 FP32 accumulators for the complete
K loop.  The two crews exchange only after K is finished, when the copy ring
is dead and can be reinterpreted as one M64N128 FP32 tile.

This changes the per-cell stage traffic to:

- A: `4,224 B`;
- Gate B: `8,448 B`;
- Up B: `8,448 B`;
- total: `21,120 B`, `16.67%` below v2 and `50%` less A traffic.

One complete logical stage is therefore `21,120 B`.  Two stages occupy
`42,240 B`; three would occupy `63,360 B` and make two CTA/SM residency
impossible under the pinned 96-KiB SM87 budget.  Two stages are consequently
part of the dataflow, not a scanned pipeline constant.  The steady K loop has
one wait/publication barrier and one slot-release barrier per group.  There is
no split-K scratch, BF16 Gate/Up materialization, global Gate intermediate, or
second A traversal.

## Static and runtime gates

The standalone kernel is compiled and linked only by an explicit test
admission.  The default-off runner slice described below uses that same
object; ordinary and production builds still contain no v3 dispatch.

Direct `ptxas` and `cuobjdump --dump-resource-usage` report:

| resource | result |
|---|---:|
| registers/thread | 128 |
| static shared memory | 42,240 B |
| stack frame | 0 B |
| local memory | 0 B |
| spill loads/stores | 0 B / 0 B |
| active CTAs/SM from runtime query | 2 |

The isolated object contains 32 static S4 IMMA instructions, 18 `LDGSTS`
instructions, and five dependency-barrier instructions.  The PTX sentinel
independently requires native `m16n8k64.s4.s4`, `cp.async.cg`, commit, wait,
and `sm_87` targeting.

The GPU correctness test compares the established K128 producer and v3
byte-for-byte for K=`128`, `256`, `384`, and `512`.  Packed codes and BF16
scales match in every case.  Packed and scale guards remain untouched, a
one-byte-short A capacity fails before enqueue, the real
M2048/N17408/K5120 last-address/capacity contract passes, and non-tile shapes
fail closed.

## Default-off runtime vertical slice

The kernel enters `q3x_kernels` only when the independent test build option
`Q3X_BUILD_SM87_A4W4_GATEUP_PROJECTION_V3_ADMISSION=ON` is present.  That
option requires both `BUILD_TESTING=ON` and the authenticated full-A4 Prefill
build.  Ordinary and production builds do not link the v3 object.

At runtime the exact value
`Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION=1` enables the worker-local
selector.  Missing values, values other than `1`, and the global optimized
Prefill disable all leave v3 off.  Admission additionally requires:

- the immutable full-inventory consumer to be authenticated K128;
- internal projection M to be non-zero and divisible by 64;
- both Gate and Up to have the exact model `[17408,5120]` shape;
- complete packed-code and K128-scale capacities for A, Gate, Up, and the
  Down-input publication.

The explicit Gate+Up priority order is v3, complete-cell v2, rejected M128,
then the established baseline.  All three experimental switches and hit
counters are independent.  Once v3 is selected its launch status is returned
directly; launch failure cannot trigger a v2 or baseline retry.  Long-tile and
projection-span accounting both exclude v2/M128 hits when v3 owns the launch.

Host tests cover the real and padded-M selectors, all eight individually
short capacities, K64/unavailable/non-M64/near-shape rejection, priority,
switch orthogonality, active/inert hit storage, exact environment value,
global disable, and ordinary-build inert behavior.  The runtime slice remains
disabled by default.

## Promotion boundary

Synthetic inputs are used only for bitwise and guard coverage and are not a
performance judge.  The runtime slice was therefore measured with the real
pinned checkpoint, authenticated K128 sidecar, a natural request truncated to
2,048 token IDs, and the OpenAI-compatible generation path.  The process also
enabled the retained native GDN/conv, whole-span BF16 A/B, Down v2, and
FlashInfer-direct Attention composition.  Gate+Up v3 was the only new switch.

The exact candidate ELF SHA-256 was
`8b9128b9700030ff787d15fe5f07c048a97764dd1e1daa5b00831445ef1507ec`.

| Prompt | Run 1 | Run 2 | Run 3 / direction | Mean / rate |
|---|---:|---:|---:|---:|
| P2048 | 3206.87 ms | 3193.74 ms | 3194.75 ms | 3198.453 ms / 640.309 tok/s |
| P3840 | - | - | 6262.02 ms | 613.221 tok/s |

The current cumulative FlashInfer-direct plus Down comparator reported
`3247.80 ms` at P2048 and a two-run mean of `6297.835 ms` at P3840.  V3
therefore saves `49.347 ms` and improves P2048 throughput by `1.543%`; the
single P3840 direction run improves throughput by only `0.572%`.

All three P2048 requests are positive, so v3 passes the real-path experiment
gate and remains available in the cumulative candidate.  The effect is still
far too small for production promotion or for the 2,000-token/s system goal.
Its structural A reuse and synchronization reduction are real, but the result
reinforces that compressed-weight scale/conversion issue and the other
projection families dominate.  A same-ELF off/on statistical replay and
public capability suite remain pending; the next higher-priority action is
the three-family Attention projection supermatrix real gate.
