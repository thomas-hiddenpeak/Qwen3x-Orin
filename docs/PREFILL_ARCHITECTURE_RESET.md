# Prefill architecture reset for SM87

Status: active architecture plan, 2026-07-29. This document supersedes
operator-local Prefill scanning as the primary development strategy. It does
not change the default numerical contract or production dispatch by itself.

## Decision

The current P513 Prefix is **2,260.7385 ms / 226.474667 token/s**. Closing the
gap to even 2,000 token/s requires an 8.83x whole-Prefix speedup. No remaining
single exact kernel edit has that authority. Development therefore moves from
isolated parameter scans to two coupled architecture tracks:

1. replace the token-serial GDN chain with a chunk/WY hierarchy designed for
   SM87 Tensor Cores; and
2. split large-M projections by numerical format and by Gate/Up, Down, and
   attention-projection shape instead of using one schedule family.

Full Attention, launch-count reduction, buffering, `cp.async`, L2 access-policy
windows, and epilogue fusion remain supporting mechanisms. None is a primary
architecture track because each has a measured whole-Prefix ceiling far below
the required gain.

MTP is excluded. cuBLASLt is an external timing and numerical reference only;
it has no production, fallback, retention, or promotion eligibility.

## Same-host evidence

One post-warmup stock-vLLM P513 request was captured on the same Orin and the
same real checkpoint. It produced token 9419 and measured **1,246.689081 ms /
411.489928 token/s**. Its 1,224.727008 ms of GPU kernel time divides as follows.

| Group | This project, ms | stock vLLM, ms | Consequence |
| --- | ---: | ---: | --- |
| NVFP4 MLP Gate/Up + Down | 984.659296 | 1,001.201344 | The native aggregate is already slightly faster; W4A16 kernel replacement alone cannot explain or close the gap. |
| FP8 QKV/Z/O projections | 496.920736 | 97.000224 | A Marlin-class large-M FP8 path is a first-order exact-path target. |
| GDN recurrence/core | 493.889408 | 41.549376 named chunk core | A token-serial recurrence is structurally wrong for Prefill. |
| Full Attention core | 76.471744 | 3.349504 | Worth fixing after the two dominant gaps, but it cannot lead the program. |
| Everything else | 208.797316 | at most 81.626560 | Fusion and scheduling are cleanup, not the architecture. |

The vLLM GDN trace contains 48 calls each to chunk-state propagation, output
reconstruction, inverse-layout merge, and WY recomputation, plus 48 KKT and 48
local-cumsum calls. The current native trace contains 1,536 exact C16 GDN
calls. This is the expected signature of an algorithm change rather than a
better serial kernel.

The complete captured top-20 list and reproduction command are recorded in
[the vLLM architecture profile](analysis/prefill-p513-vllm-architecture-2026-07-29/README.md).

## Physical boundary

At M=512, the 64 MLP blocks alone perform about 17.52 TFLOP, or 34.23 GFLOP
per prompt token. Including attention projections raises the dense projection
work to about 25.4 TFLOP, or 49.7 GFLOP/token. AGX Orin's published dense
rates are 43 FP16/BF16 TFLOP/s and 85 INT8 TOPS; SM87 also exposes INT4 Tensor
Cores but no native FP8 or FP4 Tensor Core instruction.

Consequently:

- the ideal BF16 projection-only floor is about 591 ms, before GDN,
  Attention, norms, convolution, or scheduling. An exact W4A16/W8A16 program
  cannot reach the 256 ms whole-Prefix budget required for 2,000 token/s;
- stock vLLM's 411.49 token/s is a useful first external milestone, not proof
  that the terminal target is reachable with weight-only arithmetic; and
- a 1,000-2,000 token/s single-request path requires activation-quantized
  integer Tensor Core work, sparsity, heterogeneous compute, or a comparable
  reduction in effective operation cost. The 8,000 token/s observation must
  be re-measured under this repository's batch-one/P513/one-output-token
  protocol before it is treated as a like-for-like target.

The checkpoint provides a concrete route rather than a speculative one. All
192 MLP projections and 209 non-MLP quantized projections retain an
`input_scale` tensor. The MLP range is 0.0016276042 to 0.20535715 and the
non-MLP range is 0.0098353801 to 0.44642860. The current SM87 weight-only path
does not consume these 401 calibration values.

## Numerical contracts

The project now distinguishes two explicit modes. They may share loading,
workspace, and scheduling infrastructure, but they may not silently share a
promotion decision.

### Compatibility mode

- Preserve the current per-token BF16 recurrent-state rounding and current
  quantized-weight interpretation.
- Remain the default and the fallback during development.
- Pursue Marlin-class FP8 projection dataflow and any exact shape-specific
  projection improvement.
- Do not claim that this mode can reach 2,000 token/s; its purpose is exact
  compatibility and a stable comparator.

### Throughput mode

- Opt in explicitly; never activate through an implicit shape fallback.
- Keep FP32 GDN state across a chunk and use the WY representation, then
  publish the declared Prefill-to-Decode boundary representation.
- Permit calibrated activation quantization for Prefill projections. The
  retained checkpoint `input_scale` values seed the first route; real prompt
  activation statistics may refine group scales without modifying the source
  checkpoint.
- Keep Decode on the locked exact production path. MTP remains disabled.
- Require deterministic P513 direction evidence first, followed by numerical
  state/output characterization and capability evaluation through an
  OpenAI-compatible API and EvalScope before production eligibility.

Changing the old recurrent-state NRMSE gate is not an implementation shortcut.
It is a named product contract change. The exact and throughput results must
always be reported separately.

## Selected GDN architecture

The SM87 path follows the hierarchy demonstrated by FLA/vLLM, not the
single-CTA scalar skeleton from qwen35-thor:

1. normalize Q/K and form chunk-local log-decay/beta data;
2. compute KKT and the lower-triangular WY solve for chunk 64;
3. propagate only eight C64 boundary states for C512, retaining FP32
   accumulation inside the throughput route;
4. reconstruct all token outputs in parallel; and
5. fuse or immediately consume plain-RMSNorm and SiLU(Z) only after the
   recurrence hierarchy is within budget.

The first implementation may use separate kernels. Kernel fusion is admitted
only after the algorithmic route wins in the real generation path. The SM87
design uses BF16 HMMA/WMMA for the KKT, state, and output matrix products; a
one-CTA-per-head scalar port is a correctness oracle, not a performance
candidate.

References inspected, without importing their source or binaries:

- vLLM `6f00a1ae3bd4b86168667bce673998218f461c0f`;
- FlashLinearAttention `9c8e42e762fce087c27b673af4922795d9edb85e`;
- FlashInfer `7adc546db113f00a42df07dd738f81299a839376`;
- qwen35-thor `57e29777c2aff8a97f42df6e3d9487b1327f014f`.

FlashInfer's current Prefill GDN backend targets SM90/SM100/SM120, so it is an
algebra and scheduler reference, not an SM87 implementation dependency.
qwen35-thor's scalar WY result usefully demonstrates the formulation but also
shows why matrixizing the phases is mandatory. vLLM's measured SM87 path is
the decisive reference.

## Selected projection architecture

Projection dispatch is a matrix, not one kernel:

| Family | Compatibility path | Throughput path | First structural requirement |
| --- | --- | --- | --- |
| Gate/Up, K=5120 N=17408 | retained native W4A16 large-M kernel | calibrated groupwise W4A4 INT4 MMA, fused activation quantization and SiLU where profitable | preserve high N-parallelism and share A without coupling independent weight-pipeline phases |
| Down, K=17408 N=5120 | independent W4A16 Down schedule | calibrated groupwise W4A4 INT4 MMA with a Down-specific K pipeline | choose stages and K ownership for only 40 N128 tiles; do not inherit Gate tuning |
| FP8 QKV/Z/O | Marlin-class W8A16 BF16-HMMA path | calibrated W8A8 INT8 MMA, optionally W4A4 only after quality evidence | eliminate scalar FP8 decode from the HMMA feed and specialize large-N versus O projection |

The uniform integer sidecar is generated from the authenticated real weights,
cached with a source-payload digest, and never replaces the checkpoint. Group
scales must make each integer MMA partial sum independently dequantizable; a
single global approximation is not assumed sufficient. The first candidate
uses the calibration already present in the checkpoint and measures actual
trajectory error before adding a calibration pipeline.

## Budgets and stop-loss gates

The terminal numerator is 512 Prefix tokens. The architectural milestones are:

| Milestone | Prefix budget | Throughput | Meaning |
| --- | ---: | ---: | --- |
| current native anchor | 2,260.739 ms | 226.475 token/s | frozen comparator |
| stock-vLLM parity | 1,247 ms | 411 token/s | first external architecture milestone |
| architecture proof | 768 ms | 667 token/s | both GDN and projection redesigns must be active |
| useful throughput | 512 ms | 1,000 token/s | activation-quantized path is contributing materially |
| stretch | 320 ms | 1,600 token/s | near hardware-efficient integer path |
| terminal | 256 ms | 2,000 token/s | current single-request goal |

Component stop-loss gates are deliberately large enough to prevent a return
to low-yield scanning:

- GDN C512 across 48 layers: named recurrence/core at or below 100 ms and at
  least 300 ms saved from whole Prefix in its first real P513 direction run;
- exact FP8 QKV/Z/O: at or below 150 ms aggregate, then at or below the
  stock-vLLM 97 ms reference after stabilization;
- throughput MLP projections: at or below 350 ms aggregate in the first full
  route and below 200 ms before the 512 ms Prefix milestone;
- any new experiment must name a credible path to at least 100 ms P513
  savings, remove a prerequisite for such a path, or answer a bounded causal
  question after a real-path failure.

An experiment still compares the current native incumbent and retains a
stable positive result above real noise. These architecture budgets are
stop-loss and milestone budgets, not a requirement that every intermediate
candidate beat stock vLLM or the terminal target by itself.

## Execution order

1. Land the profiler capture control and this frozen architecture evidence.
2. Add a throughput-only GDN workspace/selector with no default-route change.
3. Implement the chunk-64 KKT/WY, boundary-state, and output stages using SM87
   BF16 Tensor Cores.
4. After minimum safety/correctness admission, run one snapshot-free real P513
   baseline-versus-candidate direction cell. A negative direction closes the
   architecture version; a positive direction unlocks full numerical and
   resource work.
5. In parallel at the design level, specify authenticated W4A4/W8A8 sidecars;
   device implementation begins with Down and FP8 because Gate/Up is already
   competitive with stock vLLM.
6. Combine retained routes and re-profile only after each whole-path milestone.
7. Introduce the OpenAI-compatible API before any throughput-mode production
   promotion so EvalScope can gate capability, not merely token identity.

The unit of progress is now a Prefix budget transition, not the count of
individually positive micro-edits.
