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
