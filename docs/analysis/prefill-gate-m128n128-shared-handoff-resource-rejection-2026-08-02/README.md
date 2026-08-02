# M128N128 paired Gate+Up shared-handoff resource rejection

This record closes the standalone SM87 `M128N128 / 256T / 8-warp` paired
Gate+Up candidate before runtime integration.  No synthetic timing or
microbenchmark was used as a performance verdict, and the candidate never
entered the real-model OpenAI API path.

## Intended structural change

The candidate doubled the incumbent Gate+Up token tile from M64 to M128 while
keeping N128.  Each warp owned M128N16.  Two complete K256 A+paired-B stages
used 99,840 bytes of dynamic shared memory.  To retain the exact K512 S32
accumulation boundary while overlapping the next even stage, one N8 phase of
Gate+Up S32 partials used a 65,536-byte CTA-local shared handoff.  Total
dynamic shared memory was therefore 165,376 bytes, below the Orin opt-in limit
of 166,912 bytes.  The design had no caller-visible or global scratch.

The mechanism was intended to halve compressed-B presentation by reusing one
N128 weight tile across M128 tokens.  Its hard admission contract required:

- at most 255 registers per thread;
- zero stack, local-memory traffic, and spills;
- exactly one active CTA per SM;
- bit-exact K512 S32-before-scale accumulation.

## Hard resource verdict

The first optimized SM87 compilation failed before correctness or performance
testing:

```text
REG                 255 registers/thread
STACK               216 bytes/thread
ptxas spill stores  356 bytes/thread
ptxas spill loads   356 bytes/thread
dynamic shared      165,376 bytes/CTA (design)
SASS LDL             89
SASS STL             89
SASS LDSM.x4        128
SASS IMMA           256
SASS LDGSTS          52
```

The compiled object was:

```text
/tmp/q3x-m128n128-resource-make-build/CMakeFiles/q3x_kernels.dir/
  src/kernels/sm87/a4w4_gateup_k512_m128n128_paired_ldmatrix.cu.o
SHA256 5bc4264de094ad826fd07cedee07377aa307c5820dd0dcfb2d925b8485614dd1
```

`cuobjdump --dump-resource-usage` independently reported
`REG:255 STACK:216 SHARED:0 LOCAL:0 CONSTANT[0]:424`.  The dynamic allocation
is a launch argument and therefore does not appear in the static `SHARED`
field.  The extracted kernel SASS independently contained 89 `LDL` and 89
`STL` instructions.

This is a structural rejection.  Moving one N8 S32 partial to shared memory
did not reduce the remaining live Gate+Up FP32 accumulators and compiler state
enough to avoid spills.  Correctness, CUDA Graph, real checkpoint API, and
EvalScope timing were deliberately not run because the binary had already
failed its resource gate.

## Closed and open successors

The direct `512T / 16-warp / warp-M128N8` spelling is also closed without an
implementation: its pure data lower bound is 64 persistent Gate+Up FP32
registers plus 64 Gate+Up S32 partial registers and at least eight operand
registers, or at least 136 registers/thread.  A 512-thread CTA on SM87 has a
physical ceiling of 128 registers/thread from the 65,536-register SM file.

A successor must lower simultaneous live state, not move the same state among
register, shared, and local memory.  The remaining eligible Gate direction is
to serialize Gate and Up S32 partial production while preserving their FP32
accumulators, paired-B staging, and M128 reuse.  It receives one compile-only
resource decision before any correctness or real-API work.  Reopening the
rejected shared handoff, global scratch, cache-policy, or stage-count variants
is out of scope.

## Projection-serial successor verdict

The one permitted successor was implemented as a distinct scheduling
skeleton rather than a resource-parameter scan:

```text
CTA / ownership        M128N128 / 512 threads / 16 warps
warp ownership         M128N8
persistent state       Gate FP32 32 + Up FP32 32 registers/lane
live S32 state         one projection at a time, 32 registers/lane
dynamic shared         two combined K256 stages, 99,840 bytes
global/shared handoff  none
```

Gate and Up consumed the same paired-B publication serially.  This reduced
the pure data lower bound to 102 registers/thread: 64 persistent FP32 values,
32 live S32 partials, four A-fragment registers and two B-fragment registers.
The physical 512-thread SM87 ceiling is 128 registers/thread, leaving only 26
registers for addressing, scale conversion, loop and control state.

The copy and synchronization schedule passed independent static review.  It
kept the exact `E(K64 0..3) -> O(K64 4..7) -> one scale FMA` order for each
projection and K512 group.  Three CTA-wide synchronization points protected
odd-stage publication, even-stage replacement, and next-group odd-stage
replacement.  Paired-B/A physical offsets, scale parity, last-group handling,
window-local output and padded-M zero stores were also consistent.

The capped first compilation nevertheless failed the resource gate:

```text
ptxas registers        128 registers/thread (--maxrregcount=128)
ptxas stack frame       96 bytes/thread
ptxas spill stores     148 bytes/thread
ptxas spill loads      148 bytes/thread
cuobjdump              REG:128 STACK:96 SHARED:0 LOCAL:0 CONSTANT[0]:416
SASS                    37 LDL, 37 STL, 128 LDSM, 128 IMMA, 28 LDGSTS
```

The artifacts were:

```text
source SHA256 37d2b98424cd658174338e0ed753b9392627503dc68cdd7db780a799d2fba815
header SHA256 c6e0d0beeacaace95a6f7eaf296140e643f2e572696c95865fa6e36c92850bca
object SHA256 ebdf0248ba5b9b5a03c49253e141d9018537499e11dacc9c0370005dc6eff93a
object /tmp/q3x-m128n128-projection-serial-resource-build/CMakeFiles/
       q3x_kernels.dir/src/kernels/sm87/
       a4w4_gateup_k512_m128n128_paired_projection_serial.cu.o
```

Correctness, CUDA Graph and performance tests were again not run after the
resource failure.  Projection serialization cut the first candidate's stack
216 -> 96 bytes and SASS local loads/stores 89/89 -> 37/37, proving that the
live-state diagnosis was correct, but it did not reach the zero-spill
production contract.  The M128N128 persistent-accumulator Gate family is now
closed.  Further panel splitting or compiler-scope tuning would trade the
same state among registers and shared/local memory and is below the project's
package-sized advancement threshold.  Work moves to the whole Attention
projection package and GDN/layer-boundary macro pipeline defined by the 2K
closure budget.
