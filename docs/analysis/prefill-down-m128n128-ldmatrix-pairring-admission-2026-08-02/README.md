# Down M128N128 LDSM whole-K512 pair-ring admission

Date: 2026-08-02

Status: default-off structural admission. It is not connected to the runtime
or to a performance harness, and no synthetic timing is reported.

## Scope

The candidate preserves the incumbent Down K512 contract exactly:

- authenticated packed signed-A4 v1 payload and BF16 K512 scales;
- M128N128, 256 threads, eight M32N64 warps;
- four K256 shared stages (128 KiB), one CTA/SM;
- eight K64 integer reductions before the single K512 scale application;
- production and correctness launch validation/alias rules.

It changes two coupled parts of the consumer dataflow:

1. K64 is the operand-reuse loop. Two A fragments are loaded once with
   `ldmatrix.x4`, eight B fragments are loaded with `ldmatrix.x2`, and those
   operands issue sixteen MMAs. One K512 group therefore contains exactly
   16 x4 loads, 64 x2 loads, and 128 IMMAs.
2. Shared slots 0/1 and 2/3 are complete K512 pairs. While group `g` consumes
   one pair, group `g+1` is copied into the other. One wait+CTA barrier both
   releases the old pair and publishes the new pair.

## P0 resource and SASS result

Both kernels were compiled independently for `sm_87` with CUDA 13.3. The
candidate passes the hard gate.

| Static property | Incumbent | Candidate |
|---|---:|---:|
| Registers/thread | 255 | 238 |
| Stack/local/spill | 0/0/0 | 0/0/0 |
| LDSM.x4 | 0 | 16 |
| LDSM.x2 | 0 | 64 |
| IMMA.16864.S4.S4 | 128 | 128 |
| Scalar LDS sites | 201 | 6 inert |
| LDGSTS | 32 | 32 |
| BAR.SYNC sites | 3 | 3 |

The six candidate scalar `LDS` instructions are all exactly
`@!PT LDS RZ, [RZ]`: false-predicated ptxas fillers adjacent to the two
cp.async issue regions. They can never execute and carry no operand data.
The binary contract rejects any other scalar LDS, plus LDL, STL, PRMT, SHFL,
stack, local memory, or spills.

For the real Down shape, K=17408 means 34 K512 groups. Dynamic barriers per
output tile are:

- incumbent: `2 * 34 + 1 = 69`;
- candidate: initial publish `1` + 33 pair transitions + tile release `1`
  = `35`.

The static BAR site count stays three because the transition is a loop site;
the dynamic schedule is the intended reduction.

## Correctness admission

The candidate is compared bit-for-bit with the incumbent Down launcher, not
with a relaxed numerical tolerance:

- M128/P128, N128, K512 (one odd K512 group);
- M129/P256, N256, K1024 (two even K512 groups and padded tail);
- M129/P256, N256, K17408 (the exact production K with 34 K512 groups,
  repeated pair-ring reuse, graph capture, and two replays).

All cases pass. Prefix/suffix guards and row-stride guards remain intact,
padded rows remain BF16 zero, and packed A/B plus both scale inputs are
unchanged. These are synthetic correctness checks only.

## Static 512-thread feasibility

The 128-KiB shared allocation fixes residency at one CTA/SM, so the current
256-thread CTA exposes only eight active warps. A 512-thread M128N128 CTA
could expose sixteen warps, but it must compile at no more than 128
registers/thread to fit the SM87 register file without spills.

| Warp ownership | Output fragments/warp | K512 partial fragments/warp | x4/x2 per warp | CTA operand bytes/K512 | Assessment |
|---|---:|---:|---:|---:|---|
| current M32N64, 8 warps | 16 | 16 | 16/64 | 192 KiB | 238 regs observed |
| M16N64, 16 warps | 8 | 8 | 8/64 | 320 KiB | too much duplicated B feed |
| M32N32, 16 warps | 8 | 8 | 16/32 | 256 KiB | preferred 512-thread probe |

Both 512-thread mappings have a 64-register accumulator floor: 32 persistent
FP32 output registers plus 32 K512 S32 partial registers. That leaves at most
64 registers for A/B fragments, scales, addresses, and loop state. Merely
halving the current arrays projects roughly `238 - 64 = 174` registers, so a
mechanical remap will not pass.

M32N32 remains statically plausible only with a new low-live-range schedule:

- constrain each K64 issue sequence so ptxas cannot hoist many B fragments;
- load scales after the integer K512 reduction, not across the MMA body;
- derive shared/global addresses from compact bases;
- compile with a 512-thread launch bound and reject any spill/local traffic.

It increases shared operand bytes by 33% versus the current warp ownership,
but doubles active warps and is materially better than M16N64's 67% increase.
The next probe should therefore be M32N32, with P0 admission fixed at
`<=128` registers, zero stack/local/spill, 16 active warps, and exact
LDSM/IMMA counts. No implementation is included in this admission.
