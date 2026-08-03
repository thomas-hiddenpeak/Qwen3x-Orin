# Current production Prefill MLP: real-API NCU baseline (2026-08-03)

## Verdict

The current production Gate+Up kernel is not DRAM-bandwidth saturated.  It
runs exactly one 512-thread CTA per SM, with 128 registers per thread and
149.76 KiB reported shared memory per CTA.  Its measured DRAM read throughput
is only 20.33% of peak.  This makes the next structural target a smaller
single-projection working set that can keep two CTAs resident, rather than
another local change to the one-CTA pairfeed loop.

The current Down kernel is a different regime: it already reaches 79.23% of
peak DRAM read throughput.  It remains the production incumbent while the new
Gate+Up architecture is evaluated.

## Provenance

- Git commit: `da74f47`
- Production server ELF SHA-256:
  `1365747c888d45c59eae0f3a675b9f37e43a518eb3ba998c897614bf6bbe94a2`
- Model: real `Qwen3.6-27B-NVFP4` checkpoint and authenticated K512 sidecars
- Request path: OpenAI-compatible `/v1/completions`
- External request driver: `evalscope[perf]==1.9.1`
- Dataset: the established line-by-line P2K corpus; request 1 was the P1804
  warm-up and request 2 was the P1853 profiled request
- NCU report:
  `/home/rm01/q3x-results/q3x-ncu-current-p1853-core.ncu-rep`
- NCU report SHA-256:
  `96f5e8cdeb7b5546b8b1bba94b65aa7f7503147dbd0e1f7f654cb9568f14e123`
- Replay: kernel replay, one pass, first invocation of each selected kernel
- The profiler intentionally killed the server after the two selected kernels.
  Consequently the second EvalScope request is expected to be reported as a
  disconnected failure; it is only the real external API trigger, not a
  latency result.

## Single-invocation metrics

| Metric | Gate+Up pairfeed | Down16 pairring |
|---|---:|---:|
| Kernel time | 11.19 ms | 4.46 ms |
| Grid / block | 16 / 512 | 16 / 512 |
| Registers per thread | 128 | 128 |
| Reported shared memory | 149.76 KiB | 133.12 KiB |
| Resident limit from registers | 1 CTA | 1 CTA |
| Resident limit from shared memory | 1 CTA | 1 CTA |
| Active warps | 33.33% | 33.33% |
| Tensor-pipe active | 36.12% | 45.30% |
| Issue active | 40.74% | 35.54% |
| DRAM read throughput | 20.33% | 79.23% |

Multiplying the single selected invocation by 64 layers gives approximately
716.2 ms for Gate+Up and 285.4 ms for Down.  These agree with the independent
production Nsys totals (722.002 ms and 288.797 ms) closely enough to establish
that the selected invocations are representative.

## Next structural gate

The selected candidate is a default-off M128N64 Gate+Up kernel with one
256-thread CTA executing two serialized single-projection phases:

1. Gate uses a two-stage K256 `cp.async`/LDSM pipeline and stores the exact
   FP32 Gate tile in shared memory.
2. The CTA reuses the pipeline and accumulator registers for Up, consumes the
   shared FP32 Gate tile, and performs the existing exact SiLU-product BF16
   epilogue.

The target footprint is 82,688 bytes per CTA: 49,920 bytes for two code/scale
slots plus 32,768 bytes for the M128N64 FP32 Gate plane.  On SM87, two such
CTAs fit inside the per-SM shared-memory budget.  The candidate does not add a
global Gate scratch allocation and leaves the current split K512 quantizer and
Down16 path unchanged.

The candidate may enter the real endpoint only if its compiled resource query
proves all of the following:

- 256 threads per CTA;
- no more than 128 registers per thread;
- exactly 82,688 bytes of dynamic shared memory and zero static shared memory;
- at least two active CTAs per SM;
- no local-memory allocation or spill evidence.

After correctness, the first performance verdict is one real P1853
OpenAI-compatible API request driven by external EvalScope.  Full statistical
validation and deeper stall/bank-conflict NCU collection are conditional on a
positive endpoint direction.
