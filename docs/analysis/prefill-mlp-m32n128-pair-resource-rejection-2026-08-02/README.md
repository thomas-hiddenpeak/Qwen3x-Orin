# Prefill M32N128 paired-L1 Gate+Up resource rejection

Date: 2026-08-02

Target: Qwen3.6-27B-NVFP4 on the pinned 16-SM SM87 Orin

Status: **REJECTED before runtime integration or performance measurement**

## Question

The retained K512 producer-owned Gate+Up edge uses one M64 CTA and reaches
only one CTA/SM.  This experiment asked whether two cooperative M32N128 CTAs
could reside on every SM, consume the same Gate/Up B cells in phase, and make
the second compressed-B read hit L1 while eliminating the decoded-B shared
round trip.

This is a structural experiment, not a tile scan.  Its hard admission gate was
fixed before implementation:

- exactly two CTAs/SM on the pinned 16-SM device;
- at most 128 registers/thread;
- zero stack, local memory, and spills;
- bit-exact K512 arithmetic and the existing Down-input publication ABI;
- no allocator or host synchronization in the production launch surface.

## Implemented skeleton

The default-off source implements an M32N128, 256-thread/eight-warp CTA.  Four
warps own Gate and four own Up.  Both crews share a three-slot M32K256
`cp.async.cg` A ring, while each projection consumes the authenticated v1 B
layout directly with two `ld.global.ca.u32` fragment loads.  A scales use
`.cg`; projection scales use `.ca`.  Each K512 group has exactly two CTA
barriers.

The shared allocation is 49,152 bytes:

```text
max(three M32K256 A slots, M32N128 FP32 Gate exchange)
+ M32N512 BF16 product edge
= max(12,288, 16,384) + 32,768
= 49,152 bytes
```

The final scheduler uses a real cooperative launch.  Request-local scratch is
initialized with grid barriers; `%smid` tickets verify two ranks per SM; both
ranks then follow the static work sequence `ordinal = smid + 16*n`.  The
production launch surface performs no allocation, free, stream
synchronization, or per-cell cross-CTA spin.  The allocation/synchronization
wrapper is explicitly test-only.

## Resource verdict

The first simpler rendezvous image already failed the zero-spill gate.  One
permitted structural lifetime rewrite hoisted/reduced scale state and replaced
the unsafe ordinary launch and dynamic work queue with the cooperative static
schedule.  The final image was then stopped immediately:

| image | registers/thread | stack | spill stores | spill loads |
|---|---:|---:|---:|---:|
| initial ordinary-launch skeleton | 128 | 80 B | 80 B | 144 B |
| final cooperative/static skeleton | 128 | 160 B | 156 B | 196 B |

Final `ptxas` evidence:

```text
ptxas warning : Registers are spilled to local memory in function
'q3x_sm87_a4w4_gateup_down_k512_edge_m32n128_pair_kernel',
156 bytes spill stores, 196 bytes spill loads

Function properties:
160 bytes stack frame
Used 128 registers, used 1 barriers
```

`cuobjdump --dump-resource-usage` independently reports:

```text
REG:128 STACK:160 SHARED:0 LOCAL:0 CONSTANT[0]:456
```

The dynamic 49,152-byte shared allocation is supplied at launch and therefore
does not appear in the static `SHARED` field.

## Decision

This ownership is structurally incompatible with the two-CTA/SM gate.  Each
thread must simultaneously retain 32 FP32 long-lived projection accumulators,
32 S32 K512 partial values, direct-B fragments, A fragments, scales, pointers,
and pipeline/scheduler state.  The simpler first image already spilled, and
the cooperative contract widened rather than closed the deficit.  Further
instruction rearrangement or tile-constant scanning is therefore closed.

No correctness performance claim is made: the resource gate failed first, so
the candidate was never connected to the runner and no GPU correctness,
OpenAI API, EvalScope, NSys, or NCU performance run was performed.  The
retained K512 production route is unchanged.  The next candidate must change
accumulator/epilogue ownership, not merely cache operators or pipeline depth.

## Reproduction

The final object was compiled with:

```bash
/usr/local/cuda/bin/nvcc \
  -forward-unknown-to-host-compiler \
  -DQ3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION=1 \
  -DQ3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION=1 \
  -DQ3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M32N128_PAIR_ADMISSION=1 \
  -DQ3X_ENABLE_A4W4_MLP_K512_ADMISSION=1 \
  -Iinclude -I/tmp/q3x-m32-pair-make/generated \
  -isystem /usr/local/cuda/targets/sbsa-linux/include \
  -O3 -DNDEBUG -std=c++17 \
  --generate-code=arch=compute_87,code=sm_87 \
  -Xcompiler=-fPIC --expt-relaxed-constexpr \
  --maxrregcount=128 -Xptxas=-v,-warn-spills \
  -x cu -c \
  src/kernels/sm87/a4w4_gateup_down_k512_edge_m32n128_pair.cu \
  -o /tmp/q3x-m32-pair-second.o
```

Evidence hashes at the rejection boundary:

```text
final object  e6981c9bf51143600531db87345a86e619f5cd05b9834980121e33e568773ae4
CUDA source   6df10b9262bc400bb6b8933826f06d302f01c507562e30b3bf491fd645f7ad38
header        e45abb593a601e1788b46469d8455c5b239bcb211d4cafb2d4c55c836db6e6c5
```
