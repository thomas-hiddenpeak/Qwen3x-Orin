# Decode FP8 QKV/Z balanced-persistent candidate

Status: rejected by the first real-model serving direction gate; retained only
as negative evidence. It is not eligible for the cumulative production path.

Base checkpoint: `fba8c7b`. Candidate implementation: `e8e8d56`.

## Structural experiment

The candidate replaced the production 1,536-CTA, two-phase QKV/Z topology
with 64 persistent CTAs over one logical space of 4,096 row quads:

- every CTA owned exactly 40 QKV and 24 Z row quads;
- every thread loaded its five packed K-stage activation words once and kept
  them across all 64 row quads;
- the E4M3FN codebook remained CTA-resident in shared memory;
- canonical FP8 `.cs` loads, FFMA order, warp/block reduction trees, output
  scales, and BF16 RNE publication boundaries were unchanged;
- A/B work used 48 CTAs with one row per CTA instead of 24 CTAs with two rows.

This tested a global ownership change rather than another AoSoA layout or
grid-cap parameter. The admission selector was
`Q3X_RUN_FP8_QKVZ_BALANCED_PERSISTENT_ADMISSION=1`; it is disabled by default.

## Static and host gates

The measured Release `sm_87` ELF was
`/tmp/q3x-decode-fp8-qkv-openbook-build-make/qwen3x-eval-server`, SHA-256
`a8ede0cc8823b9f03d14b753ff3937d79b3044962c5f9be0142241251f50d4b9`.

`cuobjdump --dump-resource-usage` reported:

| kernel | registers/thread | static shared | local | stack |
| --- | ---: | ---: | ---: | ---: |
| production QKV/Z/A/B | 64 | 1,280 B | 0 B | 0 B |
| balanced persistent candidate | 64 | 1,280 B | 0 B | 0 B |

The candidate therefore met the four-CTA/SM register budget and introduced no
spill. `q3x_reference_runner_host_test` passed.

## Real serving result

The external direction gate used the real Qwen3.6-27B-NVFP4 checkpoint, the
frozen first-eight ShareGPT request manifest, serial OpenAI streaming, and 16
generated tokens per request. It compared against the retained split-KV
baseline:

| Metric | split-KV baseline | candidate | Delta |
| --- | ---: | ---: | ---: |
| exact generated outputs | 8/8 | 8/8 | unchanged |
| wall time | 22.196906 s | 22.406703 s | +0.209797 s |
| mean TTFT | 1167.302413 ms | 1169.504796 ms | +2.202382 ms |
| mean TPOT | 107.136280 ms | 108.736758 ms | +1.600478 ms |
| prompt throughput | 179.349317 tok/s | 177.670045 tok/s | -1.679272 tok/s |

The result leaf is
`/tmp/q3x-evalscope-decode-fp8-qkv-openbook-run1/fp8qkvopenbook/parallel_1_number_8`;
the independent validator record is
`/tmp/q3x-evalscope-decode-fp8-qkv-openbook-vs-split.json`.

## Conclusion and successor constraint

The exact result rules out correctness drift. With the same 64-register and
1,280-byte shared footprint, collapsing the launch to only the resident CTA
set lost more scheduling and memory-latency tolerance than activation
residency recovered. Cached activation traffic was not important enough to
pay for long-lived CTA ownership. This is a serving-level attribution; NCU
would be required to assign the loss among memory dependency, instruction
issue, and scheduler stalls.

The next structural candidate must not compress the available work grid. It
should instead use CUTLASS/Triton-style independent worker groups inside a CTA:
keep the full queued CTA population, give equally sized worker groups a fixed
number of row quads, share only codebook/setup state, and preserve the current
per-row FFMA and BF16 boundaries. The 64-CTA persistent topology and its
activation-register lifetime must not be promoted or combined with unrelated
changes.
