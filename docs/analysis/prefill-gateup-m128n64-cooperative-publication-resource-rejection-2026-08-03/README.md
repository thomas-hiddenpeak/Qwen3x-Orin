# Gate+Up M128N64 cooperative publication resource rejection

Date: 2026-08-03

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, exact-model resource mirror only

## Decision

Reject the producer-owned cooperative K512 publication skeleton before
correctness or performance timing.  Its best bounded lifetime rewrite still
compiled with local-memory traffic:

```text
registers/thread  128
stack frame        16 bytes
spill stores       12 bytes
spill loads        12 bytes
```

Zero spill is a hard admission requirement for this register-saturated
projection.  The candidate never entered the runner, OpenAI API or external
EvalScope path, and it has no production performance claim.

## Intended structural change

The experiment started from the bit-exact M128N64 same-CTA Gate-then-Up
projection.  Thirty-two cooperatively launched CTAs would form four teams of
eight; a team jointly owns one M128N512 output quantization group.  Each CTA
would retain its M128N64 BF16 product in the dead operand-pipeline plane,
publish one row maximum, receive the team-reduced K512 scale, and write its
N64 strip directly into the canonical Down A4/K512 layout.

This removes the rejected same-CTA candidate's global BF16 Product seam while
keeping product values CTA-local.  It does **not** remove a seam from the
current production pairfeed route: production already fuses Gate+Up product
and canonical K512 quantization without a standalone global BF16 quantizer.
Therefore even a resource-clean implementation would be a replacement for
the 722 ms Gate+Up family, not a system-wide seam deletion.

## Bounded resource attempts

The first full-lifetime resource mirror kept the team coordinate, descriptor
pointers and publication state live across the two projections.  It compiled
at 128 registers/thread with a 64-byte stack and 60 bytes of spill stores plus
60 bytes of spill loads.

One structural lifetime rewrite was then permitted.  It split projection and
publication phases, stored the durable descriptor in device state, and
re-derived team/generation/macrocell coordinates only after Up released its
accumulators.  This reduced the compiler result to:

```text
ptxas warning: 12 bytes spill stores, 12 bytes spill loads
Function properties:
  16 bytes stack frame
  12 bytes spill stores
  12 bytes spill loads
  128 registers
  1 barrier
```

The remaining three 32-bit values cross the Up accumulator lifetime.  Moving
them through another explicit workspace slot would merely rename the local
publication/reload seam and add team synchronization to a kernel whose upper
bound affects only the Gate+Up portion.  That is not the qualitative whole
Prefill transition required at the current distance from target, so no third
register-expression or workspace-location scan follows.

## Carry-forward

The production M64N128 K256 LDSM pairfeed remains unchanged.  The useful
architectural lesson is retained: producer-owned publication is viable only
when publication coordinates and quantization state fit outside the
projection's saturated register lifetime without local or global staging.
The next active candidate changes Attention residency from one CTA/SM to two
CTA/SM, targeting a larger 404.84 ms request budget and an independently
measurable concurrency transition.
