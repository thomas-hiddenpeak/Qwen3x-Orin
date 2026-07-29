# P513 FP8 projection-supermatrix NCU audit

Status: architecture-selection evidence, not a production promotion. The
payload in both captures comes from the authenticated real Qwen3.6-27B-NVFP4
checkpoint. Synthetic matrices have no timing authority here.

## Question

The audit asks whether the native large-M FP8 path can close the stock-vLLM
gap by continuing to tune its current M128xN128 skeleton, or whether the
operand pipeline and call topology must be replaced as one unit.

The two kernels are deliberately not presented as equal-shape benchmark
competitors:

- native computes linear-attention QKV, M512xN10240xK5120;
- stock vLLM computes the merged QKVZ supermatrix,
  M512xN16384xK5120.

That asymmetry strengthens the structural conclusion: vLLM performs 1.6x the
output work in only 1.11x the profiled time. Absolute timings remain profiler
timings and are not substituted for the P513 direction gate.

## Reproducible report identities

| Report | Bytes | SHA-256 | Checked in |
| --- | ---: | --- | --- |
| native QKV register-feed | 12,066,218 | `c3e8e11486240d025665585881cbbf01268102c30810ca404bbf26cb1fb451df` | no |
| vLLM merged QKVZ Marlin | 2,752,736 | `1d59727737d809234d7cf28016d48d93dc8e6e44d6396c0b0d20d075b386f8ef` | no |

The native report is
`/tmp/q3x-prefill-fp8-qkv-c512-register-feed-candidate.ncu-rep`. The vLLM
report is `/tmp/q3x-vllm-fp8-large-p513-architecture.ncu-rep`. The latter
captured the first real P513 merged-FP8 large-M kernel after initialization;
the profiled process was terminated after the report completed, so this NCU
run is not claimed as a completed request timing.

## Matched architectural observations

| Metric | Native QKV | vLLM merged QKVZ | Consequence |
| --- | ---: | ---: | --- |
| NCU duration | 3.47 ms | 3.86 ms | vLLM covers 1.6x N in 1.11x time |
| grid / block | 320 / 256 | 16 / 256 | vLLM uses one persistent CTA per SM |
| logical tile | M128xN128xK64 | M64xN256xK64 | vLLM doubles N ownership and A reuse |
| asynchronous stages | 3 | 4 | the reference keeps a deeper raw A/B ring |
| registers / thread | 128 | 255 | vLLM spends registers on decoded B and long-lived accumulators |
| dynamic shared memory | 79,872 B | 166,912 B | the reference intentionally selects one CTA/SM |
| achieved occupancy | 32.96% | 16.67% | occupancy is not the governing promotion gate |
| active warps / SM | 15.82 | 8.00 | fewer warps achieve more tensor work |
| tensor-pipe active | 36.78% | 53.74% | the reference feeds MMA materially better |
| scheduler no-eligible | 70.40% | 73.30% | raw no-eligible is not causal by itself |
| warp cycles / issued instruction | 13.23 | 7.51 | vLLM retires useful work more densely |
| L2 hit rate | 81.60% | 32.37% | high native L2 hit rate did not compensate for its on-SM pipeline |
| shared-load bank conflicts | 16,442,656 | 0 observed | native lookup/shared traffic is structurally conflicting |
| shared-store bank conflicts | 503,616 | 0 observed | native publication/layout also conflicts |
| LG-throttle stall / issue | 3.210 | 0.00177 | reference nearly removes this native pressure |
| MIO-throttle stall / issue | 2.835 | 0.164 | register decode and swizzle remove the shared/MIO bottleneck |
| long-scoreboard stall / issue | 0.089 | 1.195 | vLLM trades local shared pressure for a deliberate global-memory pipeline |

The raw selected metrics are transcribed in
[`selected-metrics.csv`](selected-metrics.csv). NCU also warns that 25% of the
native global sectors are excessive. The native kernel initializes and reads
a 256-entry shared E4M3 lookup table, decodes B through shared/MIO, and drains
four output bands through repeated shared stores and full-CTA barriers. The
vLLM source stages packed B, expands it in registers with bitwise operations,
uses a bank-conflict-free A layout, and publishes after the persistent slice.

## Decision

Do not run another isolated stage-count, cache-policy, or occupancy scan on
the native QKV skeleton. The selected compatibility-mode cell is the coupled
set below:

1. merge same-input projections at model load and dispatch: linear QKVZ,
   full-attention QKV/gate, and MLP Gate+Up;
2. use shape-specific projection ownership, beginning with M64xN256xK64 for
   the large-N FP8 proof and a separate output-projection configuration;
3. launch a persistent SM-count grid and let each CTA traverse complete
   output stripes;
4. maintain a four-stage raw-A/raw-B asynchronous ring;
5. apply an XOR/bank-safe shared layout to A;
6. keep packed B in shared memory only, expand exact E4M3 or E2M1 values in
   registers, and feed MMA fragments directly;
7. retain accumulators over the complete K traversal and publish each output
   tile once; and
8. authenticate and build engine-lifetime sidecars from the real checkpoint.

The unit of admission is the complete FP8 projection family in the real P513
generation path, not one QKV microbenchmark. A QKVZ-only kernel may be used as
a correctness bring-up cell, but it is not a performance milestone and is not
eligible to restart local scanning.

## Whole-Prefill program

Compatibility mode first closes the measured 383.378464-ms common W4A16 and
W8A16 pipeline gap while the positive Chunk64 GDN route supplies a separate
347.539616-ms architecture result. These are the exact external-parity steps.
They cannot by themselves reach 2,000 token/s because the dense BF16 Tensor
Core floor is too high.

The same merged-projection interface and persistent scheduler then become the
host for an opt-in throughput mode:

- W8A8/INT8 first, using authenticated checkpoint `input_scale` values and
  one activation quantization per shared normalized input;
- W4A4/INT4 only after capability evidence, because uniform INT4 cannot
  represent every NVFP4 E2M1 level exactly with one scale; and
- Decode remains on the locked exact path; MTP is excluded.

OpenAI-compatible serving and EvalScope capability gates must land before an
activation-quantized route can become production-eligible. cuBLASLt remains a
measurement reference with no dispatch, fallback, retention, or promotion
authority.

## References inspected

No source is imported. The dataflow reconstruction used the local vLLM tree at
`ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb`, notably
`vllm/model_executor/models/qwen3_5.py` and
`csrc/libtorch_stable/quantization/marlin/marlin_template.h`. The Qwen3.5
loader packs `in_proj_qkvz`, `qkv_proj`, and `gate_up_proj`; the Marlin
template supplies the persistent stripe, four-stage raw-operand, register
decode, and bank-safe layout reference. FlashInfer, FLA/Triton, and
qwen35-thor remain algorithm/scheduler references only.
