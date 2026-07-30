# Decode FP8 QKV/Z dual-worker candidate

Status: rejected by the real-model serving direction gate. This branch is a
negative experiment archive and is not eligible for the cumulative path.

Base checkpoint: `fba8c7b`. Candidate implementation: `b51fe7c`.

## Structural question tested

The prior 64-CTA persistent candidate reduced available scheduler freedom and
was negative. This candidate kept the resident and queued parallelism while
changing CTA ownership in the style of independently synchronized CUTLASS /
Triton worker groups:

- 1,024 CTAs, 512 threads per CTA;
- two independent 256-thread row-quad workers per CTA, synchronized by SM87
  named barriers rather than CTA-wide barriers;
- two row quads per worker and four per CTA, exactly covering all 4,096 QKV/Z
  row quads without a short final wave;
- one decoded E4M3FN codebook setup shared by both workers;
- each worker retained five packed activation words across only two tasks;
- the first 48 early CTA ids computed one A row and one B row concurrently so
  their work could overlap the remaining queued grid;
- canonical `.cs` weight loads, FFMA/reduction order, scales, and BF16 RNE
  boundaries were preserved.

The default-disabled selector was
`Q3X_RUN_FP8_QKVZ_DUAL_WORKER_ADMISSION=1`.

## Static and host gates

The Release `sm_87` ELF was
`/tmp/q3x-decode-fp8-qkv-dual-worker-build/qwen3x-eval-server`, SHA-256
`b7cd7a4fc023fa5ff3eba47704df1c1c4869a92b41ee4f48f74b93f1a58ac0f5`.

`cuobjdump --dump-resource-usage` reported 64 registers/thread, 3,584 bytes
static shared memory, zero local memory, and zero stack. With 512 threads this
meets the intended two-CTA/SM limit, giving four resident 256-thread workers
and 32 warps/SM. `q3x_reference_runner_host_test` passed.

## Real serving result

The real Qwen3.6-27B-NVFP4 external direction gate used the frozen first-eight
ShareGPT manifest, serial OpenAI streaming, and 16 generated tokens per
request:

| Metric | split-KV baseline | candidate | Delta |
| --- | ---: | ---: | ---: |
| exact generated outputs | 8/8 | 8/8 | unchanged |
| wall time | 22.196906 s | 22.349290 s | +0.152383 s |
| mean TTFT | 1167.302413 ms | 1167.940622 ms | +0.638209 ms |
| mean TPOT | 107.136280 ms | 108.363273 ms | +1.226993 ms |
| prompt throughput | 179.349317 tok/s | 178.126467 tok/s | -1.222850 tok/s |

The result leaf is
`/tmp/q3x-evalscope-decode-fp8-qkv-dual-worker-run1/fp8qkvdualworker/parallel_1_number_8`;
the validator is
`/tmp/q3x-evalscope-decode-fp8-qkv-dual-worker-vs-split.json`.

## Stop decision

Both structural CTA experiments were bitwise exact and serving-negative:

1. 64 persistent CTAs: mean TPOT regressed by 1.600478 ms/token;
2. 512-thread dual workers with a deep queued grid: mean TPOT regressed by
   1.226993 ms/token.

The second result rules out reduced grid depth as the sole cause of the first
failure. Coarsened CTA ownership, extra activation lifetime, and named-barrier
worker grouping do not beat the selected 256-thread row-quad topology on this
real serving path. Per the experiment gate, FP8 QKV/Z CTA-topology work stops
here; no third variant should be built from either candidate. Any future FP8
work requires a different decomposition and fresh whole-runner evidence, not
another CTA-count or worker-group permutation.
