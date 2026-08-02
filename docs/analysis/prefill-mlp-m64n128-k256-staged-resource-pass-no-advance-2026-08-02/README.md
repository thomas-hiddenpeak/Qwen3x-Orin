# Prefill M64N128 split-projection K256 staged resource result

Date: 2026-08-02

Target: Qwen3.6-27B-NVFP4 on the pinned 16-SM SM87 Orin

Status: **RESOURCE PASS; NOT ADVANCED TO RUNTIME**

## Question tested

This default-off skeleton asks whether one M64N128, 512-thread CTA can keep
Gate and Up on separate eight-warp crews while staging complete A, Gate-B,
and Up-B K256 operands.  Every warp owns M64N16.  The experiment preserves
the authenticated v1 K512 payload and the exact K512 S32-before-scale
boundary.  It exposes only a resource query and has no runtime launcher,
selector, correctness surface, or performance path.

The shared allocation is:

```text
two A+GateB+UpB K256 stages    81,920 B
two K512 scale slots            1,280 B
live pipeline                  83,200 B
M64K512 BF16 product edge      65,536 B
total                         148,736 B
```

After the K loop, a 32,768-byte component-major FP32 Gate exchange reuses
the dead pipeline allocation.  It therefore does not increase the dynamic
shared allocation.

## Resource result

The first complete compute image passed without a rewrite:

```text
registers/thread              128
stack/local/spill        0 / 0 / 0
dynamic shared             148,736 B
active CTAs/SM                    1
maximum threads/block           512
target                          SM87
```

The device resource query returned status zero and confirmed an opt-in
shared-memory limit of 166,912 bytes.  Static SASS contains 64 `IMMA`, 26
`LDGSTS`, 213 `LDS`, 48 `STS`, seven `BAR.SYNC`, and no `LDL` or `STL`.

The real K512 lifetime is present in this image: eight S32 fragments remain
live beside eight persistent FP32 fragments, both K256 halves are accumulated
before their BF16 scale is applied, and the final Gate/Up result reaches the
observable packed-output edge.  The resource result is therefore not an
empty-shell measurement.

## Whole-data-flow audit

The resource pass does not establish a performance direction.  A conservation
audit after compilation found that the split-projection mapping does not
reduce total shared operand reads.

For one warp and one K64 plane:

```text
retained paired mapping  1 A fragment + 8 B fragments = 20 LDS.u32 = 2,560 B
split projection         4 A fragments + 2 B fragments = 20 LDS.u32 = 2,560 B
```

Across 16 warps both request 40 KiB per K64 plane.  The split mapping moves
duplication from B to A; it does not remove it.  IMMA work is also unchanged.
It additionally writes and reads a 32-KiB Gate exchange for every N128 cell.

The genuine mechanism in the skeleton is instead its alternating whole-stage
pipeline.  It reduces the dynamic CTA barriers from about 165 to 93 per
M64K512 edge.  The same one-barrier-per-K256 schedule can be applied to the
retained paired M16N32 mapping without the Gate exchange, with an expected
about 85 barriers per edge.  The retained static image is also lighter:

| Static image | Registers | LDS | STS | BAR.SYNC | Spill |
|---|---:|---:|---:|---:|---:|
| Retained paired edge | 125 | 181 | 8 | 6 | 0 |
| Split-projection skeleton | 128 | 213 | 48 | 7 | 0 |

The split mapping is therefore structurally dominated before runtime work.
It is retained as reproducible resource and data-flow evidence but is not
eligible for a performance claim.  No synthetic timing, OpenAI API,
EvalScope, NSys, or NCU performance run was made.  The production baseline
remains 843.5446 token/s on the external P2K gate.

## Decision

Move the alternating one-barrier-per-K256 schedule into a default-off clone
of the retained paired Gate+Up edge.  That candidate must first preserve the
125-register, zero-spill, one-CTA envelope and bit-exact v1 seam.  Its first
performance verdict must then come from the real checkpoint through the
OpenAI API and external EvalScope 1.9.1: one warmup plus P1853, followed by
the four-request P2K gate only if the direction is positive.

## Reproduction hashes

```text
candidate object    81ad6cd45a57dc405f4739f0f85fee728b9adf9d82dd062b670181c559dabf4e
candidate resource  c55f281ae88fbd48e766b90a8656518a07190b192533dc08741200c70e0e08cb
candidate SASS      6564516d354bcec4d17901eacddea930be715a82900fb4e03a72078e7dc30233
candidate header    c0c920bf90b366d07a2b4fec30b391197ff1e55b0c5b73d65b2a8c6f25ba1ebd
candidate source    910e13a48e549f067bfc17157457b26389c2c59a757dcbc95ff6b488fd45bb7a
retained resources  a7dabcb56b749fbb0d17344398a4de7aee599c02399aaf6c4839364480a20a67
retained SASS       c03f4f933245bed2ff50483c175190543accd743c2eadc4972ce048ad8215986
```

Raw local artifacts:

```text
/tmp/q3x-m64n128-k256-staged-first.o
/tmp/q3x-m64n128-k256-staged-first.resources.txt
/tmp/q3x-m64n128-k256-staged-first.sass
/tmp/q3x-gateup-edge-current.resources.txt
/tmp/q3x-gateup-edge-current.sass
```
