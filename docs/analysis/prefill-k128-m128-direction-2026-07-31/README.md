# K128 publication and M128-v1 real-path direction gate

Date: 2026-07-31 (Asia/Shanghai)

This is an early direction gate, not an EvalScope matrix or a production
promotion. Performance used the real pinned Qwen3.6-27B-NVFP4 checkpoint,
the OpenAI-compatible `/v1/completions` path, one generated token, and a
2,048-token prefix of a natural P4K corpus request. Synthetic matrices were
used only by the prior correctness tests.

## Authenticated inputs

- Source checkpoint: `nvidia-qwen3.6-27b-nvfp4@0893e160`
- Runner commit: `7c5a4bcf25e8c3d50de1e8f55158e3361d1e404b`
- Production ELF SHA-256:
  `bed52f5ae707401b55eb166f5fb2202f5687b0da48be4bd973589db16f361c8a`
- A4 publication: version 2.0, 400 projections, packed K64 codes with one
  shared BF16 scale per K128 group
- Payload bytes: `12543590400`
- Payload SHA-256:
  `57bfe2c741f5a22052a48f1ce6a15967f2d1a526a9a32898eb130467879b3660`
- Policy SHA-256:
  `d3695d84c77c7ed0e235bee2264660a03bccb5c68eb75e3d4d9a7a1bd461ae6c`

Both runs used the same ELF, model, K128 payload, prompt token array, server
arguments, and GPU lock. The only candidate difference was:

```text
Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION=1
```

## Direction result

| Route | HTTP total run 1 | HTTP total run 2 | Mean | Prompt rate |
|---|---:|---:|---:|---:|
| K128 + incumbent M64 | 5.167061 s | 5.161949 s | 5.164505 s | 396.553 tok/s |
| K128 + M128-v1 Generic/Paired | 5.186295 s | 5.179067 s | 5.182681 s | 395.162 tok/s |

M128-v1 regressed mean latency by 18.176 ms (`1.003519x`) and prompt
throughput by 0.351%. The candidate is therefore rejected at the real-path
direction gate. No B-B-B-B harness or capability matrix is justified for
this version.

The negative result is consistent with its compiled resource contract: the
M128 Generic and Paired kernels admit only one CTA per SM. Doubling the M
reuse did not compensate for the lost residency and register/shared-memory
pressure. The implementation remains opt-in as an attribution/reference
candidate; it is not enabled by default.

## Separate K128 observation

The incumbent M64 K128 run is materially below the frozen real P2048 K64
exact-GDN profile (`prompt_prefill_ms=6481.45`), but the recorded metrics are
not identical (`HTTP total` versus server-side Prefill timing). Treat this as
a positive K128 direction only. A same-metric statistical and capability
gate is still required before K128 can be promoted.

## Next action

1. Test the Down-specific M128 kernel behind an independent gate while
   Generic/Paired stay on the K128 incumbent.
2. Test the corrected whole-span BF16 A/B route independently.
3. Replace M128-v1 with a complete Gate/Up cell targeting at least two
   CTAs/SM, a 2--4 stage async pipeline, and a longer useful scale/accumulator
   lifetime; do not continue tile-constant scans on the rejected skeleton.
