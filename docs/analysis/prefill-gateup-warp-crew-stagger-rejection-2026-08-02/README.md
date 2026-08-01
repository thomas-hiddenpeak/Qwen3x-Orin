# Gate+Up warp-crew phase-stagger rejection (2026-08-02)

## Hypothesis

The exact-K128 Gate+Up v3 kernel makes each warp issue an S4 Tensor fragment
and then immediately converts, scales, and accumulates its S32 result on the
FP32 pipe.  The candidate offset the two independent Gate and Up warp crews by
one fragment so one crew could issue Tensor MMA while the other applied the
previous fragment.  It preserved every K128 scale boundary, instruction,
barrier, packed layout, and floating-point accumulation order.

This was a bounded causal experiment, not the projection architecture reset.
Performance authority was the real-weight OpenAI API through external
EvalScope; no synthetic performance result was used.

## Correctness and resources

The isolated contract and CUDA bitwise-correctness tests pass 2/2.  The clean
Release server kernel compiles to:

```text
REG:113 STACK:0 SHARED:42240 LOCAL:0
```

The incumbent used 128 registers/thread.  Both retain two CTAs/SM and zero
spill.  The lower register count therefore did not imply a performance win;
the 42,240-byte shared allocation already fixes occupancy.

Candidate server ELF SHA-256:
`cc55605dc83af354e7c655fee887895f77e641e61d1c62a5bf11f2ee74cc8821`.

## External EvalScope P2K result

The run uses the verified ShareGPT P2K corpus, EvalScope 1.9.1, one warmup,
four measured requests, concurrency one, and `max_tokens=1`.  It uses the
same current-best eight-selector composition and authenticated 400/400 K128
sidecar as the incumbent.

| Route | Success | Mean TTFT | Total throughput |
|---|---:|---:|---:|
| incumbent `4a90d1f` | 4/4 | 3,173.43 ms | 606.7634 tok/s |
| warp-crew stagger | 4/4 | 3,867.43 ms | 497.8929 tok/s |

Every paired request regresses:

| Prompt tokens | Incumbent | Candidate | Delta |
|---:|---:|---:|---:|
| 1,792 | 2,740.273 ms | 3,277.790 ms | +537.517 ms |
| 1,853 | 3,291.613 ms | 3,823.067 ms | +531.454 ms |
| 1,906 | 2,863.926 ms | 3,393.127 ms | +529.201 ms |
| 2,148 | 3,797.912 ms | 4,975.753 ms | +1,177.841 ms |

All four first tokens remain identical to the incumbent (`#`, `这段`, `这是`,
`我已`).  This isolates the rejection to performance rather than an observed
one-token trajectory change.

Candidate summary SHA-256:
`411d9945307415ed9925cc5654919dfc05655a95281dbe83822a9747532afc88`.

## Decision

The source change is fully reverted and is not retained as an experimental
baseline.  SM87 already schedules independent resident warps; source-level
crew divergence instead lengthened the critical path without removing the
six scalar feed instructions per exact-K128 MMA, any of the 40 scale
boundaries, or any CTA barrier.

The result closes phase-only scheduling as a route to a qualitative Prefill
gain.  Work moves to the K512 shape-specific projection macro-core, which
reduces the scalar boundary count fourfold and targets the whole 64-layer MLP
plane before any further tuning.

Evidence root:
`/home/rm01/q3x-gateup-stagger-evalscope-p2048-4a90d1f`.
