---
q3x_document:
  id: q3x-prefill-reference-translation-matrix
  class: local-work-package
  status: active
  owner: prefill-maintainers
  authority: source-to-SM87 translation and selection record for WP-PREFILL-REFERENCE-TRANSLATION-v1
  effective: 2026-08-12
  last_reviewed: 2026-08-12
  supersedes: []
  superseded_by: []
  ssot_for: the active Prefill reference-translation work package only
  review_trigger: source pin, P40 geometry witness, selected AOT plan, or work-package closure
---

# Prefill reference-to-SM87 translation matrix

This document is the bounded working record for
`WP-PREFILL-REFERENCE-TRANSLATION-v1` under
`AC-PREFILL-SM87-AOT-SYSTEM-v1`. It does not amend the product target,
stable SDD boundary, current implementation status, numerical contract, or
production route. Those authorities remain with the Constitution, system and
Prefill SDDs, Current Status, and numerical ledger.

The purpose of a separate work-package document is deliberate: source pins,
candidate hypotheses, and falsifiers change during development without
turning the SDD into an experiment notebook. When this package closes, this
document becomes frozen evidence or names its successor; only a genuinely
stable system boundary returns to the SDD.

## 1. Package boundary and return point

| Field | Binding |
| --- | --- |
| Product symptom | Cold/no-cache, single-request 40K--60K and approximately 130K Prefill remains below the owner-set API target and useful vLLM starting line |
| Parent candidate | `AC-PREFILL-SM87-AOT-SYSTEM-v1` |
| Active package | `WP-PREFILL-REFERENCE-TRANSLATION-v1` |
| Incumbent | The P40 development route identified by [Current Status](CURRENT_STATUS.md), not restated here |
| Numerical boundary | [Prefill mathematical-equivalence ledger](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md) |
| Production exclusions | No MTP, cuBLASLt production path, silent fallback, approximate mainline, request-time JIT/repack/autotune, or full-model BF16 weight copy |
| First return point | One clean-host, real-checkpoint, cold/no-cache P40 OpenAI API direction witness |
| Stop condition | Close or redesign the complete architecture after one negative composition and at most one predeclared causal profile; do not turn it into a tile scan |

## 2. Control model and current Q3X plant

The active engineering loop treats the complete runner as the controlled
system:

| Control-system role | Project binding |
| --- | --- |
| Reference command | Owner-set API capacity, accuracy, latency, Decode, non-MTP, and dependency constraints |
| Controller | Authenticated `DeploymentPlan`, `PrefillExecutionPlan`, role tactics, memory plan, and stream/event graph |
| Plant | The pinned Qwen3.6-27B-NVFP4 checkpoint executing on one Jetson AGX Orin SM87 |
| State | Resident weights and sidecars, residual, KV, GDN/convolution state, arenas, progress, and route identity |
| Actuators | Operand layout, physical work grain, CTA/warp ownership, pipeline, buffering, fusion, and AOT tactic selection |
| Observables | Exact route receipts, consumed-token count, state/oracle result, pure Prefill, TTFT, Decode, memory, and clean-host resource state |
| Disturbance rejection | Fail closed on foreign CPU/GPU consumers, cache/protocol mismatch, fallback, late compilation/allocation, or artifact drift |

The current P40 whole-core controller is structurally layer-major and
Prefill/Decode are logically separate. Its per-layer GPU submission graph is:

```text
five M8000 fill panels
  -> one P40000 Attention or GDN core
  -> five M8000 output/drain panels
  -> one P40000 Gate+Up/SiLU then Down/residual MLP
  -> next layer
```

All of these launches use the main CUDA stream in
[`reference_runner.cpp`](../src/runtime/reference_runner.cpp). The two-slot
whole-request submission window records and synchronizes events for bounded
cancellation observation; it does not submit work to two GPU streams and is
not double buffering. An auxiliary Prefill stream exists for an older
branch-parallel path, but the P40 whole-core functions do not use it.

Consequently the source study must not assume that buffering, operator
overlap, or a producer/consumer pipeline is already deployed. Every new buffer
must name the producer, consumer, dependency edge, lifetime, and measured
critical-path effect.

Current dominant implementation seams are:

- FP8 fill/drain projections dispatch M8000 Marlin panels from
  [`reference_runner.cpp`](../src/runtime/reference_runner.cpp);
- whole-layer NVFP4 Gate+Up and Down dispatch through
  [`nvfp4_prefill_persistent.cu`](../src/kernels/sm87/nvfp4_prefill_persistent.cu);
- full Attention calls the vendored FlashInfer single-prefill FA2 template
  from
  [`full_attention_c512_register_pipeline.cu`](../src/kernels/reference/full_attention_c512_register_pipeline.cu);
- the direct whole-prompt Attention wrapper passes no split-KV temporary; and
- the GDN route has a prompt-wide work graph, but the candidate remains
  accuracy-unqualified and cannot lend production authority to a new graph.

## 3. Inspected source pins

| Reference | Inspected identity | Scope |
| --- | --- | --- |
| Q3X plant snapshot | `a8c914e737a5d833e739fb0d42af5ccdc65efb2b` | Current plan, runner, projection, Attention, GDN, stream, and state seams |
| vLLM current source | [`1c1077c6cc4308bdf4c2bf207d7420757a9bfd87`](https://github.com/vllm-project/vllm/tree/1c1077c6cc4308bdf4c2bf207d7420757a9bfd87) | ModelOpt W4A16 selection, linear backends, scheduler/chunk geometry, startup specialization |
| vLLM comparison release | `v0.26.0`, local source commit `568afb3a13806beb53bb2e6bd518269357b237c0` | Installed comparison-path reconstruction |
| Humming | [`b18cfac980d2427c0b32a2c027974b3274d0413a`](https://github.com/inclusionAI/humming/tree/b18cfac980d2427c0b32a2c027974b3274d0413a) | SM8x W4A16 packed operand, pipeline, scheduler, JIT/AOT specialization |
| Triton paths in vLLM | Same vLLM `1c1077c6...` pin | GPTQ/ROCm W4A16 and scaled-MM ownership/fusion comparison only |
| FlashInfer source study | [`4b969c9363e0d32a33f8fcccd9ef5a5fec51cd9f`](https://github.com/flashinfer-ai/flashinfer/tree/4b969c9363e0d32a33f8fcccd9ef5a5fec51cd9f), 0.6.15 | FA2/SM8x Prefill Attention planning and dataflow |
| FlashInfer in Q3X | [vendored 0.6.12 closure](../third_party/flashinfer/README.q3x.md) | Existing AOT-compiled direct Attention control path |
| FlashLinearAttention | [`51a8c0a88b9e0e7a4409e3670db771fd58c389c7`](https://github.com/fla-org/flash-linear-attention/tree/51a8c0a88b9e0e7a4409e3670db771fd58c389c7) | Gated-delta chunk work graph, GVA sharing, fusion, and state passing |
| Mamba | [`e9594ce1c732d97440f0332fdc43170a2294dbfa`](https://github.com/state-spaces/mamba/tree/e9594ce1c732d97440f0332fdc43170a2294dbfa) | Selective scan, SSD chunk/state/output decomposition |
| qwen35-thor | [historical pinned audit](PREFILL_REFERENCE_AUDIT.md) | Separate Prefill entry, model-specific chunking, bulk Attention, persistent workspace |

Source review obeys provenance and license boundaries. A source mechanism is
an existence proof and design input, not permission to copy it or add a
runtime dependency.

The principal audited loci are:

- vLLM:
  `model_executor/layers/quantization/modelopt.py`,
  `model_executor/kernels/linear/__init__.py`,
  `layers/quantization/utils/humming_utils.py`,
  `config/scheduler.py`, and `v1/core/sched/scheduler.py`;
- Humming: `tune/sm8x.py`, `tune/base.py`, `tune/raster.py`,
  `kernel/humming.py`, `include/humming/kernel/humming.cuh`,
  `scheduler.cuh`, `memory/g2s_pipeline.cuh`,
  `memory/s2r_pipeline.cuh`, and `mma/wmma.cuh`;
- Triton/vLLM:
  `model_executor/kernels/linear/mixed_precision/triton_w4a16.py` and
  `compressed_tensors/triton_scaled_mm.py`;
- FlashInfer: `flashinfer/utils.py`, `include/flashinfer/utils.cuh`,
  `attention/prefill.cuh`, and the Q3X wrapper named in section 2;
- FLA: `ops/gated_delta_rule/chunk.py`, `chunk_fwd.py`,
  `wy_fast.py`, `ops/common/chunk_delta_h.py`, and
  `ops/common/chunk_o.py`; and
- Mamba: `csrc/selective_scan/selective_scan_fwd_kernel.cuh`,
  `modules/ssd_minimal.py`, `ops/triton/ssd_combined.py`,
  `ssd_state_passing.py`, and `ssd_chunk_state.py`.

## 4. Reference translation matrix

| Source mechanism | Invariant to retain | Architecture/runtime shell to replace | SM87 AOT translation | Main risk or falsifier |
| --- | --- | --- | --- | --- |
| vLLM scheduler and startup plan | Explicit scheduled-token budget, shape warmup, backend identity, route-ready barrier | Python scheduler, general batching policy, late JIT ecosystem | Offline enumerate only P40/P60/P130 and required tails; authenticate the selected physical plan before ready | A service-level chunk can improve concurrency while regressing batch-one Prefill |
| vLLM ModelOpt W4A16 selection | W4A16 is distinct from native W4A4; BF16 A, packed E2M1/E4M3 scales, exact role boundaries | General backend registry and automatic fallback order | Bind every NVFP4/FP8 role to one explicit SM87 launcher and fail closed | Accidentally attributing a W4A4/native-FP4 result to the W4A16 route |
| Humming SM8x W4A16 | Static N/K/layout, M buckets, packed B+scale through global/shared movement, register decode immediately before BF16 MMA, multistage load/decode/MMA, persistent scheduler, L2-aware raster | Runtime source generation, cubin cache/load, generic palette, NVML discovery | Generate fixed cubins offline; freeze role-specific layout, M range, tile, stages, CTA residency, Stream-K/raster, compiler and SASS identity | Reusing one Gate tactic for K-heavy Down or reproducing only the tile while omitting the pipeline |
| Triton/vLLM kernels | Packed load, local unpack/dequant, dot ownership, masked tails, fused consumer boundaries, and shape heuristics | Runtime Triton JIT plus the audited ROCm/GPTQ W4A16 or same-dtype scaled-MM contracts | Use only the transferable fusion and ownership ideas; express an independently derived NVFP4/FP8 result as native AOT C++/CUDA | Copying GPTQ/ROCm scales, W8A8 semantics, or a fixed tile into the ModelOpt W4A16 path |
| FlashInfer FA2 Prefill | Packed-Q GQA reuse, swizzled shared K/V, `cp.async`, BF16 HMMA QK/PV, FP32 online-softmax state, no score matrix | Template/JIT URI, paged-cache generality, dynamic plan buffers, FA3-only resources | Fixed Q24/KV4/D256/NHD/causal AOT plan; retain a Q64/KV32 control and design one larger effective-Q producer/consumer candidate | Rewriting the current Q64 path without reducing K/V reload, or changing the reduction/publication contract |
| FLA gated delta rule | Q/K work sharing, bounded C16/C32/C64 work graph, producer preparation, boundary-state ownership, adjacent output consumer | FP32 authoritative state across a chunk, WY/KKT associative state transform, global intermediate tensors | Keep the exact token-serial BF16 state core; fuse/shared-stage QK, gate, convolution, RMSNorm/SiLU, and O-projection boundaries around it | Deferring per-token BF16 state rounding or materializing multi-GB FLA intermediates |
| Mamba selective scan/SSD | Local work, explicit boundary state, state pass, and output reconstruction as separate schedulable phases | Associative FP32 scan state and model-specific recurrence | Use the work-graph decomposition and fixed resource planning without replacing the Qwen GDN transition | Calling a real-number associative scan bitwise-equivalent to the rounded recurrence |
| qwen35-thor | Separate Prefill/Decode entry points, model-specific chunk size, bulk Attention, persistent handles/workspace | SM110 TMA/UMMA/cluster mechanisms and its hardware/model thresholds | Preserve the separation and static workspace plan; translate physical work to SM87 primitives | Copying Thor thresholds, materialized score storage, or hardware-specific mechanisms |

## 5. Projection translation

vLLM distinguishes ModelOpt `W4A16_NVFP4` from native W4A4. On SM87 its
automatic W4A16 route selects Marlin; Humming is an explicit backend choice.
The project must therefore compare and translate W4A16 dataflows rather than
projecting native-FP4 behavior onto Orin. vLLM's compatibility handling may
collapse unequal fused-partition global scales by taking their maximum; that
behavior is not admissible here. Every Gate, Up, and FP8 partition keeps its
authenticated independent scale and publication boundary.

For BF16-A W4A16 on SM87, the inspected Humming heuristic first selects a
`M128N256K64` block, `M64N64K64` warp, eight-warps, three-stage,
one-CTA/SM seed. Gate+Up (`N=34816`) owns 136 N256 tiles and Down
(`N=5120`) owns 20. This is a source-derived starting structure, not a
performance result or immutable resource gate. Humming's dense-M table stops
at 8192 and extends that final configuration upward; P40/P60/P130 therefore
require explicit project AOT buckets rather than an unexamined extrapolation.

The distinction from the rejected Q3X packed-v2 skeleton is structural:
packed-v2 used K128 stages with about 153.6 KiB shared memory and did not
reproduce Humming's K64 feed, shared-to-register double buffering,
Stream-K tail ownership, or complete consumer layout. Its rejection does not
falsify the Humming dataflow, and the former two-CTA/SM local rule does not
automatically constrain this new architecture package.

The AOT projection plan must bind separately:

- merged Gate+Up: shared A lifetime, independent weight/scale/reduction and
  BF16 publication, then the declared SiLU/Up consumer;
- Down: K-heavy feed, short N grid, residual publication boundary, and its own
  raster/Stream-K policy; and
- linear-layer FP8 QKV/Z, full-layer FP8 Q/K/V, and FP8 Attention O: exact
  E4M3FN decode/scale/publication, with role grouping only when shared-input
  reuse survives the complete consumer path.

The first exact implementation disables Stream-K because it can change the
FP32 reduction tree. Stream-K can enter only under a separately proved
finite-precision identity. Humming's `--use_fast_math` compilation behavior
is likewise not inherited.

The first implementation is selected from the combined A/B/scale
load-decode-MMA-epilogue plan. It is not assembled by accepting individual
`.ca`, `cp.async`, raster, stage, or decode-sharing toggles in isolation.

## 6. Attention translation

The existing vendored path proves the owner's SM87 compatibility correction:
FlashInfer FA2 is usable on SM87 and already executes in this project. For the
fixed BF16 Q24/KV4/D256 shape, the current path uses packed Q, Q64, KV32, four
Q warps, and `cp.async`/HMMA online softmax. The whole-prompt wrapper passes
no split-KV workspace.

At P40 the Q grid already has about 15,000 CTAs. Split-KV adds no missing
occupancy for this case; paging and service-level chunking are not automatic
batch-one throughput wins. The current Attention result instead indicates
repeated K/V service across many Q64 owners. The next structural comparison
therefore has only two planned topologies:

1. the existing Q64/KV32 direct path as a control; and
2. a Q128 or Q256 effective-Q producer/consumer path in which one staged K/V
   tile serves more Q work, with V-output partitioning used to control
   register pressure.

The first real P40 API direction decides whether the complete composition
moves. Only after a positive direction may a bounded profile verify that
L2/DRAM traffic fell, HMMA and `cp.async` are active, and register/shared
pressure did not create a new critical bottleneck. A same-skeleton Q64 rewrite
or broad tile scan is outside this package.

The direct FlashInfer path does not quantize Q/K/V or intentionally truncate
the Attention formula, but it remains accuracy-unqualified because its
full-state oracle differs from the incumbent. A translated Attention route
must declare its FP32 online-softmax reduction, the `FP32 Attention -> BF16
output -> sigmoid gate -> BF16 output` publication sequence, and KV state
boundaries; matching one generated token is insufficient.

## 7. Exact GDN/SSM translation

FLA and Mamba make chunk recurrence parallel by keeping an authoritative FP32
state across token/chunk boundaries and composing associative transitions.
The current Q3X production contract rounds the authoritative state to BF16
after every token and the next token consumes those bits. That rounding makes
the transition non-associative. WY, KKT, pair scan, and SSD state passing
cannot enter the present mainline under a claim of bitwise equivalence.

The admissible translation is a complete bounded graph:

```text
FP8 QKV/Z and BF16 A/B producers
  -> causal convolution plus shared Q/K normalization and alpha/beta/gate prep
  -> exact per-value-head token-serial FP32 update
  -> per-token BF16 state publication and pre-round output use
  -> declared BF16 output boundary
  -> RMSNorm + SiLU(Z) consumer
  -> streaming FP8 O projection
```

Parallelism remains available across 48 value heads, 16 shared QK groups,
preparation, and consumers. Q/K normalization is computed once per QK head
and may be reused by its three value heads only after the original reduction
and FP32 operand bits are preserved. C16/C32/C64 bounded workspaces, four
warps, and a two-stage ping-pong pipeline are the source-derived starting
structure. The current 256-thread, roughly 139-KiB, one-CTA/SM monolith is not
inherited as a universal resource plan.

No FLA-style full-prompt A/W/U/state materialization is allowed. Such tensors
would consume multiple GiB at P40/P60 and reproduce the movement problem the
translation is meant to remove. Any fusion that keeps a formerly published
BF16 value only in FP32 is a numerical-contract change and must receive a
separate identity and qualification; this package does not silently authorize
it.

## 8. Physical work grain and buffering decision

There is no single project-wide `chunked prefill` boolean or chunk size.
The external request remains one admitted whole prompt, while the AOT plan
chooses a physical grain for each dependency:

| Boundary | Planned grain | Reason |
| --- | --- | --- |
| NVFP4/FP8 projection | Role-specific M bucket, potentially the complete layer span | Maximize packed-weight/scale reuse without forcing one Gate/Down tactic |
| Full Attention | Internal Q/KV tiles; whole-prompt logical call at P40 | Q parallelism is already ample; optimize K/V service rather than scheduler segmentation |
| Exact GDN | C16/C32/C64 bounded recurrence chunks | Respect per-token state order while bounding on-chip state and workspace |
| API/scheduler | Whole admitted request with explicit capacity | No public partial commit, truncation, or cache ambiguity |

The current P40 route is serial. The first candidate may use a two-buffer
producer/consumer boundary only where the source/dataflow matrix proves
independence, for example bounded GDN preparation versus exact state
consumption. A third buffer is not selected by convention; it requires a
third independently overlappable stage, a bounded memory cost, and a positive
complete-route result.

vLLM chunked Prefill is primarily a scheduler token budget: it improves
fairness, memory bounds, Decode interleaving, and hybrid-state management, and
may align a GDN/Mamba boundary to a cacheable block. It can regress a
single-request Prefill by shrinking M, repeating layer walks and weight
traffic, or publishing more boundary state. The reference witness below
measures that system trade rather than treating chunking as a kernel feature.

## 9. One bounded reference-geometry witness

After the source matrix is frozen, perform exactly one reference-engine
geometry study before native implementation:

- real checkpoint and tokenization;
- one cold/no-cache, batch-one P40 API request with one output token;
- MTP and prefix-cache reuse disabled;
- explicit vLLM scheduled-token/macrochunk budgets `2048`, `4096`,
  `8192`, and a near-whole-prompt admissible value;
- JIT, kernel warmup, and deterministic tuning completed before timing;
- exact backend, model, token count, route, cache, memory, and process
  identity retained; and
- EvalScope performance CLI plus vLLM pure-Prefill telemetry kept as distinct
  observables.

This witness chooses service/macrochunk geometry and reconciles the owner's
known vLLM behavior with the exact current configuration. It is not repeated
for every Q3X change and does not select a Q3X production dependency.

Before the run, the Jetson clean-host preflight must pass with `tegrastats`,
CPU/process inspection, and GPU-device-handle ownership. If another workload
owns a critical resource, the run does not start and no timing is retained.

## 10. Package closure

This package closes only when all of the following are true:

1. source pins and invariant/ISA/SM87 translations are complete;
2. the one P40 geometry witness either selects or rejects each macrochunk
   class under the matched reference protocol;
3. projection, Attention, GDN, buffer, state, and handoff plans form one
   authenticated AOT candidate with no undeclared fallback;
4. each added-compute-for-movement trade has an equivalence/resource ledger;
5. the candidate has a direct clean-host real P40 API return point and
   falsifier; and
6. Roadmap names the next implementation package or records architecture
   closure.

Until then, source review and local code inspection change no default route.
