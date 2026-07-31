# Real P2048 projection single-pass NCU audit (2026-07-31)

## Scope and profiler constraint

This audit samples the first real-model launch of each retained projection
specialization inside one natural-token P2048 OpenAI request. The profiled
server ELF is the same SHA256
`933eeda4412d867baeca468ab098824f57484722b015584b7a03bb3e216ad69b`
used by the current-best NSys and whole-request measurements. The
authenticated A4 K128 payload contains the real Qwen3.6-27B-NVFP4 weights.

An initial conventional NCU attempt requested the full stall, instruction,
cache, bank, and launch set. It failed before producing a report because
kernel replay tried to back up the complete resident model on Orin unified
memory (`NvMapMemAllocInternalTagged failed: error 12`). No conclusion is
drawn from that failed attempt.

The retained audit first proved on the CUDA correctness fixture that this
minimal counter set schedules in exactly one pass:

```text
duration
Tensor-pipe active
SM issue active
active warps
DRAM-read throughput
launch/resource attributes
```

It then profiled five unique real-request symbols, each at invocation one,
with one pass and no kernel-replay backup. `--cache-control none` preserves
the request's natural cache state; GPU clocks are not modified. NCU terminates
the server immediately after the fifth match, so an empty HTTP reply is
expected and is not a generation result.

The NCU report SHA256 is
`be0e78b052d4adebc47055df45f887b72910c962ab6ffa9093172dd615ec99f7`.
Its raw CSV SHA256 is
`a553ff34bf28a3242b225496e8bef9da7bca049587af9ee5cfde03c5b7f77fbb`.

## Results

| Kernel | One launch | DRAM read | Tensor active | Active warps | Issue active |
|---|---:|---:|---:|---:|---:|
| Linear QKV+Z pair | 7.497920 ms | 35.25% | 27.00% | 30.18% | 63.63% |
| Attention O | 2.726144 ms | 38.73% | 27.83% | 30.43% | 65.05% |
| Gate+Up v3 | 19.853632 ms | 19.59% | 21.65% | 29.84% | 56.88% |
| Down v2 | 7.848320 ms | 44.74% | 27.37% | 30.36% | 63.93% |
| Full Q/K/V | 6.557504 ms | 35.47% | 26.98% | 30.13% | 63.61% |

All launches use a 32-CTA grid, 256 threads, 42,240 B static shared memory,
and two active CTAs/SM. Registers/thread are 123 for Linear and Full, 128 for
Attention O and Gate+Up, and 124 for Down. The theoretical active-warp limit
is 33.33%; the measured 29.84--30.43% therefore shows that residency is
already close to its resource ceiling.

## Interpretation

The projection kernels are neither DRAM-saturated nor occupancy-starved.
Gate+Up is the clearest case: only 19.59% DRAM-read throughput and 21.65%
Tensor activity despite 29.84% active warps and 56.88% issue activity. The
other shapes reach only about 27% Tensor activity while schedulers issue at
about 64%.

This matches the compiled exact-K128 loop structure. For every 32 S4 tensor
MMA instructions, Down and Attention execute 64 integer-to-FP conversions,
64 scale multiplies, and 64 FP32 accumulation FFMAs. The scalar feed is six
instructions per tensor instruction before address, shared-load, pipeline,
and loop instructions. More `cp.async` stages cannot remove those operations,
and the measured DRAM headroom says that memory prefetch is not the principal
missing factor.

The next projection architecture is therefore an exact-K128 v4 rather than a
stage-count scan:

1. consumer-native scale/code packets with FP32-expanded exact BF16 scales;
2. Marlin-style double-buffered scale fragments in registers;
3. M-oriented B/scale reuse and shape-specific Gate, Down, and Attention
   ownership;
4. two independent S32 partial groups so the next tensor work can overlap
   conversion/scale/FP32 accumulation of the previous group.

The hard resource gate remains at most 128 registers/thread, two CTAs/SM,
zero spill, and bitwise exact K128 order. A separately authenticated coarser
scale contract may have a higher ceiling, but it is a later accuracy/performance
experiment and cannot replace the exact path without real API and public
capability gates.
