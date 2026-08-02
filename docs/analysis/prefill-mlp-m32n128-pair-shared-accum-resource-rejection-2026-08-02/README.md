# Prefill M32N128 paired-L1 shared-accumulator rejection

Date: 2026-08-02

Target: Qwen3.6-27B-NVFP4 on the pinned 16-SM SM87 Orin

Status: **REJECTED at the resource gate; never connected to runtime**

## Structural question

The preceding cooperative M32N128 direct-B skeleton failed because every
thread simultaneously retained one projection's 32 FP32 long-lived values
and 32 S32 K512 partial values.  This follow-up changed accumulator ownership
rather than scanning a tile constant: long-lived Gate and Up FP32 values were
moved to two component-major CTA-shared planes, leaving only the current K512
S32 delta in registers.

The fixed admission gate was:

- 256 threads and exactly two cooperative CTAs/SM;
- at most 128 registers/thread;
- zero stack, local memory, and spills;
- exactly 77,824 bytes dynamic shared memory/CTA;
- no runtime, correctness, or performance work before the resource gate.

## Implemented data flow

Each CTA owns M32N128.  Four warps compute Gate and four compute Up, while a
same-SM CTA pair owns the two M32 halves of one logical M64 cell.  A uses a
three-slot M32K256 `cp.async.cg` ring.  Gate and Up consume the authenticated
v1 B layout directly through `.ca` loads so the paired CTA can potentially
reuse the first copy through L1.

Shared storage is:

```text
three M32K256 A slots                 12,288 B
Gate and Up FP32 accumulator planes  32,768 B
M32N512 BF16 product edge            32,768 B
total                                77,824 B
two resident CTAs                   155,648 B
SM87 per-SM limit                   166,912 B
```

The accumulator layout is component-major so all 32 lanes access consecutive
banks for one scalar component.  K512 group zero initializes the shared
planes; intermediate groups load/FMA/store one Float4 at a time.  On the final
group, Gate publishes its final shared values while Up reuses each S32 partial
register slot for the final FP32 value, then consumes Gate after the seam
barrier.  This avoids allocating a second 32-register final-Up array.

The final cooperative scheduler uses the deployment-specific `%smid` ticket
path in the same kernel.  A one-time admission probe would have been required
to prove the physical SM-ID set is exactly 0 through 15.  There is no separate
setup launch and therefore no invalid cross-launch CTA-placement assumption.

## Two bounded resource images

The first image used a robust dense-SM mapping inside the cooperative kernel:

```text
registers/thread  128
stack             80 B
spill stores      84 B
spill loads       116 B
```

The one permitted structural rewrite removed the dense-map scans, restricted
the cooperative mapping lifetime, passed only the minimal compute parameters,
and used the pinned-Orin `%smid` contract.  It did not close the resource gap:

```text
registers/thread  128
stack             80 B
spill stores      84 B
spill loads       120 B
dynamic shared    77,824 B
active CTAs/SM    2
```

The runtime resource query fails closed with CUDA status 701 and reports
`local=80`.  The final SASS contains 28 `LDL` and 19 `STL` instructions.  Its
other static counts are 94 global loads, eight `LDGSTS`, 32 `IMMA`, 13
`BAR.SYNC`, 152 `LDS`, and 113 `STS`.

Because mapping simplification left stack/spill essentially unchanged, the
remaining peak belongs to the compute lifetime: eight four-value S32 partial
fragments, direct-B fragments, the A fragment, K512 scales/address state,
pipeline state, and the exact fused epilogue.  A third lifetime or tile rewrite
would violate the predeclared experiment bound and was not attempted.

## Decision

This candidate is a hard resource **REJECT**.  It did establish the intended
77,824-byte shared allocation and two-CTA residency, but a spill-backed kernel
cannot test the proposed L1/shared data flow honestly.  No GPU correctness,
OpenAI API, EvalScope, NSys, or NCU performance run was performed.  The
retained K512 production path and its 843.5446 token/s cumulative Prefill
baseline are unchanged.

The next Gate+Up skeleton must use one CTA and an A+B staged pipeline with
projection-specialized warp ownership.  It must not continue compressing this
two-CTA register lifetime or revive synchronous direct-B as an isolated
mechanism.

## Reproduction and hashes

Compilation used SM87 Release code generation, the full-A4 and K512 admission
definitions, `--maxrregcount=128`, and `-Xptxas=-v,-warn-spills`.

```text
first object   ef65f343392577bfeca0075f6815c750676055a5c0b20be0d1a43277565e5cd5
final object   fa1a5d3f9d45ef6838d0a2d405d647b7f99f3497516656f0b805d7d1ce814d82
resource dump  5b4245cb3250631065a7a6938a08ddf52c965ae3329da9226feb807fa40237e8
final SASS     8573bcdccdc306e8659b67557489ed9b65afd9bef9ddaefd80211d4fbb448047
header         17dc3f3a5b8a99bd495d7569afd4014605bbb7c92bf768d6a281825f898042b4
CUDA source    e87ce63d2e1aab4657dbcb2b9702daaf853db3e527812d3759e65f53e257968f
```
