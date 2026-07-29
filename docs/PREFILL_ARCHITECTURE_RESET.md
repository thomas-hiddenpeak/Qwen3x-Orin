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
2. replace the current decoded-shared/WMMA projection family with one
   Marlin-class raw-operand pipeline shared by W4A16 and W8A16, while keeping
   Gate/Up, Down, and attention-projection ownership shape-specific.

Full Attention, launch-count reduction, buffering, `cp.async`, L2 access-policy
windows, and epilogue fusion remain supporting mechanisms. None is a primary
architecture track because each has a measured whole-Prefix ceiling far below
the required gain.

MTP is excluded. cuBLASLt is an external timing and numerical reference only;
it has no production, fallback, retention, or promotion eligibility.

## External whole-product checkpoint

The OpenAI-compatible adapter and EvalScope 1.9.1 direction baseline are now
implemented. On one warmup plus 32 real ShareGPT requests at concurrency one,
native mean TTFT is **3,168.79 ms** versus **1,144.51 ms** for matched stock
vLLM, a **2.768687x** gap. Mean TPOT is much closer at **108.92 versus 104.42
ms (1.043095x)**. Total external workload throughput is **106.1454 versus
188.0494 token/s**. This separates the project-level priority cleanly:
Decode remains frozen and Prefill architecture is P0.

This is a single-process directional baseline, not release evidence. Every
complete GDN or projection architecture milestone must first show value in the
real generation path and then return to the same external workload after its
numerical and engineering gates pass. P513 remains the fast direction and
profiler-attribution cell; it no longer has sole authority over project-level
progress. The reproduction procedure, retained configuration, and limitations
are in [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md).

## Same-host evidence

One post-warmup stock-vLLM P513 request was captured on the same Orin and the
same real checkpoint. It produced token 9419 and measured **1,246.689081 ms /
411.489928 token/s**. Its 1,224.727008 ms of GPU kernel time divides as follows.

| Group | This project, ms | stock vLLM, ms | Consequence |
| --- | ---: | ---: | --- |
| NVFP4 MLP Gate/Up + Down | 984.659296 | 723.607136 | The 261.052160 ms gap is the largest exact projection-family opportunity. |
| FP8 QKV/Z/O projections | 496.920736 | 374.594432 | The 122.326304 ms gap is material, but not the previously reported 399.9 ms. |
| GDN recurrence/core | 493.889408 | 41.549376 named chunk core | A token-serial recurrence is structurally wrong for Prefill. |
| Full Attention core | 76.471744 | 3.349504 | Worth fixing after the two dominant gaps, but it cannot lead the program. |
| Everything else | 208.797316 | at most 81.626560 | Fusion and scheduling are cleanup, not the architecture. |

The vLLM GDN trace contains 48 calls each to chunk-state propagation, output
reconstruction, inverse-layout merge, and WY recomputation, plus 48 KKT and 48
local-cumsum calls. The current native trace contains 1,536 exact C16 GDN
calls. This is the expected signature of an algorithm change rather than a
better serial kernel.

The Marlin split above corrects an earlier classification error. The second
kernel template type ID is authoritative: `562949953487106` is FE2M1/NVFP4
and `2814749767172868` is FE4M3FN/FP8. Grouping by M64 versus M8 tile shape
incorrectly produced 1,001.201344 ms W4A16 and 97.000224 ms W8A16 even though
the four raw row times were correct. The corrected W4A16+W8A16 total remains
1,098.201568 ms, versus 1,481.580032 ms native, leaving a 383.378464 ms common
projection-pipeline opportunity.

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
- Pursue a Marlin-class W4A16/W8A16 projection dataflow and exact
  shape-specific ownership without changing numerical formats.
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
| Gate/Up, K=5120 N=17408 | Marlin-class W4A16 raw-operand/register-decode path | calibrated groupwise W4A4 INT4 MMA, fused activation quantization and SiLU where profitable | preserve high N-parallelism and share A without coupling independent weight-pipeline phases |
| Down, K=17408 N=5120 | the same operand-pipeline skeleton with independent Down ownership | calibrated groupwise W4A4 INT4 MMA with a Down-specific K pipeline | choose stages and K ownership for the smaller N grid; do not inherit Gate tuning |
| FP8 QKV/Z/O | Marlin-class W8A16 raw-operand/register-decode path | calibrated W8A8 INT8 MMA, optionally W4A4 only after quality evidence | remove the decoded-B shared tensor and specialize large-N versus O projection |

The exact common skeleton is one complete dataflow cell, not a tile sweep. The
measured vLLM large-M configuration uses 256 threads, M64xN256xK64 ownership,
four asynchronous stages, XOR-swizzled A, register-buffered packed B,
register dequantization, and direct MMA. The native implementation may change
shape where SM87 resource evidence requires it, but it must preserve the
coupled mechanism: engine-lifetime authenticated prepack, pipelined raw A/B,
fragment-oriented register feed, no decoded-B shared tensor, no per-K64
full-CTA producer/consumer bubble, and one final output publication. FP8 is
the first executable proof because its exact bitwise expansion is simpler;
the cell is admitted only with a concrete NVFP4 port and aggregate budget.

The uniform integer sidecar is generated from the authenticated real weights,
cached with a source-payload digest, and never replaces the checkpoint. Group
scales must make each integer MMA partial sum independently dequantizable; a
single global approximation is not assumed sufficient. The first candidate
uses the calibration already present in the checkpoint and measures actual
trajectory error before adding a calibration pipeline.

### Matched NCU verdict

The same-real-checkpoint NCU audit is now complete. The current native FP8
QKV kernel computes M512xN10240xK5120 in 3.47 ms under NCU; stock vLLM's
merged QKVZ kernel computes M512xN16384xK5120 in 3.86 ms. Native uses 320
M128xN128 CTA instances, 128 registers/thread, 79,872 bytes of dynamic shared
memory, and 32.96% achieved occupancy. vLLM uses 16 persistent M64xN256 CTA
instances, 255 registers/thread, 166,912 bytes of dynamic shared memory, and
16.67% achieved occupancy. Despite half the occupancy and 1.6x the output
work, vLLM reaches 53.74% tensor-pipe activity versus native 36.78%.

The causal difference is on-SM dataflow. Native records 16,442,656 shared-load
and 503,616 shared-store bank conflicts, 3.210 LG-throttle stall cycles per
issue, and 2.835 MIO-throttle cycles per issue. The vLLM capture records no
shared bank conflicts in these metrics, 0.00177 LG-throttle cycles, and 0.164
MIO-throttle cycles. Therefore neither two-CTA residency nor the native shared
E4M3 lookup/decode skeleton remains an architectural requirement. The full
evidence and report hashes are pinned in
[`analysis/prefill-p513-fp8-supermatrix-ncu-2026-07-29/README.md`](analysis/prefill-p513-fp8-supermatrix-ncu-2026-07-29/README.md).

### Projection-supermatrix call topology

The compatibility proof targets the whole FP8 family. At load time it forms
logical same-input supermatrices without modifying the checkpoint:

- linear attention: QKV + Z = N16384;
- full attention: Q/gate + K + V = N14336; and
- MLP: Gate + Up = N34816, with Down retaining its own K-heavy ownership.

Each partition retains its own exact scalar weight scale and output address;
merging is a scheduling and prepack contract, not a numerical rescaling. For
FP8 this reduces 208 native large projection calls to 128 family calls (one
merged input and one output per layer). For NVFP4 it reduces 192 calls to 128
(one merged Gate+Up and one Down per layer). A QKVZ-only bring-up is allowed
to establish correctness, but performance admission waits for the complete
family on real P513.

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
- exact W4A16+W8A16 projections: at or below 1,250 ms aggregate in the first
  complete engine route, then at or below the stock-vLLM 1,098.201568 ms
  reference after stabilization;
- exact W4A16: at or below 850 ms, then at or below 723.607136 ms; exact
  W8A16: at or below 425 ms, then at or below 374.594432 ms;
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
5. Profile one real-weight native and stock-vLLM large-M W8A16 cell under the
   same NCU metrics, then implement the complete four-stage raw-operand,
   register-decode, direct-MMA proof in FP8. Do not retain decoded-B shared
   storage merely to preserve the current kernel skeleton.
6. Port the proven operand pipeline to NVFP4 Gate/Up and Down as separate
   ownership configurations; judge the program on the aggregate
   383.378464-ms projection gap, not on the number of local edits.
7. In parallel at the design level, specify authenticated W4A4/W8A8 sidecars
   for the throughput contract. Combine retained routes and re-profile only
   after each whole-path milestone.
8. [done] Introduce the OpenAI-compatible API before any throughput-mode
   production promotion so EvalScope can gate capability, not merely token
   identity.
9. Return every complete architecture milestone to the pinned external
   workload; only then may it advance from internal explanation to a
   whole-product decision.

The unit of progress is now a Prefix budget transition, not the count of
individually positive micro-edits.

## First architecture checkpoint

The isolated Chunk64/WY route has now completed its first real-model
checkpoint. Three native SM87 stages (block-16 KKT/solve, persistent FP32
state, and reconstruction+norm+gate) plus reference-only W/U/QK GEMMs reduced
P513 Prefix from 2,260.333589 to 1,912.793973 ms, saving 347.539616 ms. It
therefore passed the 300 ms whole-Prefix stop-loss but remains above the
100 ms component budget at an attributed 150.035424 ms.

The P512 recurrent-state NRMSE is 0.117148528 across all 37,748,736 BF16
elements. This confirms throughput mode is a distinct numerical contract;
the route remains test-only and the default compatibility path is unchanged.
Full evidence is in
[`analysis/prefill-p513-gdn-chunk64-architecture-2026-07-29/README.md`](analysis/prefill-p513-gdn-chunk64-architecture-2026-07-29/README.md).

Because the remaining projection program is 1,481.580032 ms native versus
1,098.201568 ms in stock vLLM, the next implementation priority is the common
Marlin-class W4A16/W8A16 operand pipeline. FP8 is its first exact executable
proof, not a standalone optimization campaign. GDN resumes only for a design
capable of removing the W/U/QK global boundaries or most of the remaining
50.035424 ms budget.
