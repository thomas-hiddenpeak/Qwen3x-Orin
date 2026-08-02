# P1853 Prefill 2K token/s closure budget

This budget converts the user-visible single-request Prefill target into two
whole-system architecture packages.  It is based on the promoted real
checkpoint, OpenAI-compatible API, external EvalScope 1.9.1 P1853 result and
its request-scoped NSys capture.  Synthetic matrices have no performance
authority here.

## Current authority and target

```text
EvalScope P1853 TTFT                 2,103.35 ms
EvalScope total throughput             881.43 token/s
profiled server Prefill span         2,108.37 ms
profiled GPU kernel time             2,090.78 ms
2,000 token/s P1853 budget             926.50 ms
required profiled-span reduction     1,181.87 ms
```

The target is `1853 / 2000 = 0.9265 seconds`.  The profiled server span is
used for the conservative component budget; the non-profiled external API
result remains the delivery metric.

## Why isolated Gate or Attention wins cannot close the target

The current profiled span divides as follows:

| Plane | Current | Closure budget | Required reduction |
|---|---:|---:|---:|
| Gate+Up kernel | 748.269 ms | 338.238 ms | 410.031 ms |
| Down kernel | 399.490 ms | 169.122 ms | 230.368 ms |
| Attention projection chain | 475.863 ms | 213.884 ms | 261.979 ms |
| Projection plane subtotal | 1,623.622 ms | 721.244 ms | 902.378 ms |
| GDN, layer boundaries, attention core, and fixed work | 484.748 ms | 205.256 ms | 279.492 ms |
| Total | 2,108.370 ms | 926.500 ms | 1,181.870 ms |

The projection closure corresponds to roughly 125 blended TOPS for the
pinned Qwen3.6-27B shapes on Orin.  The values are component budgets, not
claims that any untested kernel already achieves them.

The already designed first Attention projection package has a continuation
gate of 150 ms.  Even if it passes, with Gate unchanged the system would still
be about 1,958.37 ms, or 946 token/s.  It would leave 1,031.87 ms to remove.
Likewise, a Gate kernel at the former 620-ms structural gate plus that first
Attention win would still be about 1,830.10 ms, or 1,013 token/s.  Those gates
therefore validate mechanisms; they are not the terminal performance bar.

## Package A: whole projection plane

Gate+Up, Down, and all Attention projections must advance as one 721.244-ms
budget.  Shape-specific kernels remain mandatory:

- Gate+Up: `M x 17408 x 5120`, paired Gate/Up weights, target 338.238 ms;
- Down: `M x 5120 x 17408`, independent K-heavy schedule, target 169.122 ms;
- Attention projections: all 128 physical calls plus their K256 activation
  boundary, target 213.884 ms.

The current M128N128/256-thread shared-handoff Gate candidate is excluded by
its compiled stack and spills.  M128N512 global scratch, M16N64 imbalance,
spilling M128N256 Down, cache toggles, and stage-count scans remain closed.
Each successor must remove operand presentation or simultaneous live state at
the dataflow level.  The first real performance decision is the real model,
OpenAI API, and external EvalScope P1853 run; NSys/NCU follows for attribution.

## Package B: GDN and layer-boundary macro pipeline

The non-projection span must fall from 484.748 to at most 205.256 ms.  Its
current diagnostic decomposition is:

```text
GDN/SSM core                           248.863 ms
GDN BF16 A/B helpers                    20.174 ms
residual and head RMS                  117.308 ms
Attention core plus pre/post            56.173 ms
K512 activation quantization            20.142 ms
memops                                  15.415 ms
fixed/host                               6.672 ms
```

The eligible successor is a value-head-owned prompt-span macro pipeline.  A
CTA keeps the value-head state across the 29 C64 chunks while preserving the
existing explicit BF16 round/reload boundary every eight chunks.  WY solve,
value recompute, state update, chunk output, RMSNorm and SiLU must share
phase-reused register/shared state instead of publishing every intermediate
to global memory.  Its output must be consumed directly by the next
projection/layer boundary, with residual finalization and next-layer
quantization combined where the exact numerical seam permits it.

This package is not a launch-count exercise.  The current request has only
about 4.15 ms of kernel-span idle time, while the GDN intermediates account
for approximately 13.93 GiB of logical read/write traffic per request.  A
candidate that merely combines launches without removing those global
boundaries cannot meet the roughly 279.5-ms reduction requirement.

The internal closure targets are:

```text
GDN plus BF16 helpers                  <= 110 ms
residual/RMS/packing boundary          <=  40 ms
Attention core and pre/post            <=  40 ms
memops                                 <=   8 ms
fixed                                  <=   7 ms
```

The historical C16+norm fusion, exact persistent-C512 state kernel,
state-lifetime-only prompt span, and C4096 workspace expansion are excluded:
their measured absolute pools or numerical contracts cannot provide this
package-sized improvement.

## Advancement rule

Resource and bit-exact gates protect safety, but performance advancement is
always decided on the production path.  A candidate first runs one warmup and
one measured real P1853 request through the OpenAI API and external EvalScope.
A positive package-sized result advances to the natural P2K four-request
matrix and profiler attribution.  A negative direction is recorded and
closed; it does not trigger a tile, cache, or pipeline-depth sweep.
