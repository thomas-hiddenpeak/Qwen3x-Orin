# Rejected P513 FP8 projection-supermatrix experiment

Status: rejected by the first real full-model generation-path direction gate.
This branch is an archival experiment and is not production eligible. The
default build and runtime route remain unchanged.

## Question

The experiment asked whether all 208 FP8 Prefill projections could move as one
coupled unit from the existing M128xN128 W8A16 kernels to a Marlin-class
supermatrix dataflow:

- merge linear-attention QKV/Z and full-attention Q/K/V inputs;
- prepack every FP8 projection into a fragment-native engine-lifetime sidecar;
- use one persistent CTA per SM, M64xN256xK64 ownership, and a four-stage
  `cp.async` raw-operand ring; and
- decode E4M3FN in registers and publish each output tile once.

The implementation is compiled only with
`Q3X_BUILD_FP8_PREFILL_SUPERMATRIX_ADMISSION=ON` together with
`BUILD_TESTING=ON`, and is selected only by the private
`Q3X_RUN_FP8_PREFILL_SUPERMATRIX_ADMISSION=1` environment variable. It has no
production selector, fallback, or installed-library role.

## Real-model direction results

Both candidates ran the tokenizer-pinned P513 prompt against the authenticated
`nvidia/Qwen3.6-27B-NVFP4@0893e160` checkpoint on the target SM87 Orin. Both
generated token ID 9419, text `Hello`, and 513 ordered steps.

| Route | Prefix execution | Result |
| --- | ---: | --- |
| frozen native production | 2,260.7385 ms | incumbent |
| scalar exact E4M3FN decode | 3,157.035 ms | rejected |
| coupled exponent-bias/register decode | 2,377.535 ms | rejected |

The second route recovers about 779.5 ms from the first prototype, proving
that register decode is a coupled architectural mechanism rather than a small
parameter choice. It nevertheless remains about 116.8 ms, or 5.17%, slower
than the native full Prefix direction anchor. No exhaustive harness or
production promotion work is justified after this negative result.

The profile-wrapped second run measured 2,386.572 ms Prefix execution and
retained the same output oracle. Its three FP8 kernel families sum to
634.188 ms:

| Kernel family | Calls | Total | Mean |
| --- | ---: | ---: | ---: |
| K5120, pair QKV/Z | 48 | 357.484 ms | 7.448 ms |
| K5120, triple Q/K/V | 16 | 100.046 ms | 6.253 ms |
| K6144, output | 64 | 176.658 ms | 2.760 ms |

The frozen native FP8 family is about 496.921 ms. The 137.267 ms FP8-family
regression explains the whole-Prefix loss; Gate/Up, Down, GDN, and attention
did not conceal an offsetting benefit.

## Causal closure

The first scalar-decoder NCU cell showed only 12.47% Tensor-pipe activity and
fixed-latency/ALU starvation. The exponent-bias decoder raises Tensor-pipe
activity to 28.17%, but the matched real checkpoint cell still reports:

- 7.457 ms duration;
- 16.67% achieved occupancy with eight active warps/SM;
- 28.11% issue active and 71.89% no-eligible cycles;
- 7.12 warp cycles per issued instruction;
- 2.694 fixed-latency wait, 0.740 short-scoreboard, 0.615 math-pipe, and
  0.438 LG-throttle stall ratios per issue; and
- 103 registers/thread, 102.4 KiB dynamic shared memory, and one CTA/SM.

The previously captured stock-vLLM merged-QKVZ Marlin cell reaches 53.74%
Tensor-pipe activity with a shared-to-register ping-pong pipeline, direct
`ldmatrix`, and interleaved register decode/MMA. The archival candidate has a
global-to-shared four-stage ring but no equivalent second register pipeline;
continuing isolated stage/cache/tile scans on this skeleton is therefore not
admitted.

## Decision and methodology boundary

The code is archived off `main`. It is neither retained as a native
development incumbent nor used by the evaluation gateway. P513 established
the negative direction and enabled bounded profiler attribution; it is not a
business-performance or release claim.

Further Prefill work is blocked on the Phase 3.5 OpenAI-compatible evaluation
gateway and an external EvalScope workload/capability baseline. Future
architecture candidates may use P513 as a deterministic profiler sentinel,
but their project value must be decided by the external real-workload matrix
before local statistical and engineering qualification.

## Local profiler artifacts

These reports contain real-checkpoint payloads and are intentionally not
checked into Git:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `/tmp/q3x-fp8-supermatrix-p513-negative.nsys-rep` | 855,086 | `23a4767bfaf0a204337659d1b99a2161495f2b183d1a7e88c115f289a7c7e78a` |
| `/tmp/q3x-fp8-supermatrix-p513-negative-cell.ncu-rep` | 331,909 | `3a9418a812e6bde69e8dfeb16860edb9d76086f39dcb33e38035433ea754f9d4` |
| `/tmp/q3x-fp8-supermatrix-p513-bias-fold-negative.nsys-rep` | 857,833 | `7766fafdee30a5eaaa16d6aff9f3a2417e7af55bf52127f9aae83a06dc5aed41` |
| `/tmp/q3x-fp8-supermatrix-p513-bias-fold-negative-cell.ncu-rep` | 239,431 | `fe9a98d74bcc1d3f28bcb045ed9782ee203831d84b54e88461c8f9ae07a094a8` |

