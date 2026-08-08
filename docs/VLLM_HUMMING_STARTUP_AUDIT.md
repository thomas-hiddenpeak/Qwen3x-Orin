---
q3x_document:
  id: q3x-vllm-humming-startup-audit
  class: external
  status: frozen
  owner: runtime-maintainers
  authority: pinned external startup and deployment mechanism input only
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: external source pin, local startup design, or DeploymentPlan change
---

# vLLM and Humming startup-specialization audit

This is a frozen external-source and dated local-gap audit. It is not an active
local work package, current-status page, delivery plan, or production route.
Its mechanisms apply only through a named SDD/Roadmap architecture candidate.
Local observations below belong to the original audit line ending at
`eb39a3e`; they do not describe the current tree unless `CURRENT_STATUS.md`
explicitly adopts them.

The accepted native design is owned by the
[system SDD](SDD.md#3-deploymentplan-contract) and the sole delivery order by
the [Roadmap](ROADMAP.md). This audit neither changes a selector nor reports a
Humming performance result. The controlling performance, accuracy, non-MTP,
and cuBLASLt rules remain in the
[engineering constitution](ENGINEERING_CONSTITUTION.md) and the
[real-model evidence policy](REAL_MODEL_PERFORMANCE_POLICY.md).

## Source pins and scope

The audit inspected:

- vLLM official `main` at
  [`75231eff2f3873e2bce7cc9558bb5227ea70b808`](https://github.com/vllm-project/vllm/tree/75231eff2f3873e2bce7cc9558bb5227ea70b808),
  dated 2026-08-08;
- Humming official `main` at
  [`b18cfac980d2427c0b32a2c027974b3274d0413a`](https://github.com/inclusionAI/humming/tree/b18cfac980d2427c0b32a2c027974b3274d0413a),
  dated 2026-08-07; and
- the Q3X development tree captured by original audit commit `eb39a3e` on the
  target SM87 Orin and pinned `nvidia/Qwen3.6-27B-NVFP4` checkpoint.

The local comparison environment currently identifies itself as vLLM 0.26.0
with Humming 0.1.10. Results or behavior from that installed release must not
be attributed to the newer source pins without a matched run.

## Transferable conclusion

The audit supported the project-owner hypothesis: the most valuable lesson from JIT
for this fixed-model, fixed-hardware runner is not runtime compilation. It is
the protocol that turns observed model, device, shape, and memory facts into a
verified execution plan before the server reports ready.

The transferable model is a two-stage specialization system:

1. an offline target-Orin qualification/build stage may enumerate candidates,
   compile kernels, transform weights, measure the real API path, and select a
   plan; and
2. production startup validates and loads that immutable plan. The request
   path performs no compilation, autotuning, repacking, route discovery, or
   unexpected allocation.

JIT is therefore a discovery and confirmation mechanism. The production
artifact is AOT: authenticated weight layouts, fixed cubins, a fixed operation
graph, a small shape-bucket dispatch table, and a declared memory plan.

This work is not a substitute for the large-M GEMM, Attention, and GDN
dataflow reset. It removes repeated interpretation and makes the selected
dataflows enforceable. Only load-time fusion, final weight layout, tactic
selection, launch segmentation, and removal of hot-path decisions can affect
steady-state Prefill. Cache reuse and offline packing alone mainly reduce
process startup and first-request variance.

## What current vLLM establishes before ready

At the pinned source, the relevant startup chain is:

```text
API configuration
  -> worker/device and distributed initialization
  -> model load and quantized-weight post-processing
  -> real model profile at the maximum scheduled-token shape
  -> KV-memory plan and allocation
  -> compile-shape warmup and kernel-specific tuning
  -> CUDA Graph capture and workspace finalization
  -> scheduler-realistic prefill/decode/mixed warmup
  -> late-JIT monitoring
  -> engine READY
  -> renderer warmup
  -> HTTP serving
```

The important mechanisms are:

| Mechanism | Pinned implementation | Transfer to Q3X |
| --- | --- | --- |
| Startup fingerprint | `vllm/v1/worker/startup_plan.py` binds the reusable KV measurement to vLLM/config, device name/memory/CC, Torch/CUDA, rank and world size. It also requires current free memory to meet the recorded baseline. | Bind every deployment plan to the complete checkpoint, Q3X build and kernel ABI, SM87 device contract, driver/runtime, capacity and measured free-memory floor. |
| Compile cache | vLLM's compile cache includes configuration, compiler, environment, rank and traced-source content. Its AOT loader can fail instead of silently recompiling on a miss. | Store cubin/SASS and plan hashes as derived artifacts. A mismatch fails before ready; production never recompiles or silently changes route. |
| Loading as compilation | Qwen3.5 loading maps separate `in_proj_qkv` + `in_proj_z` into `in_proj_qkvz`, and `in_proj_b` + `in_proj_a` into `in_proj_ba`; quantized layers then repack for the selected kernels. | Generate fused, kernel-native QKVZ/BA, Gate/Up, Down and FP8 layouts offline from authenticated real weights. |
| Shape coverage | Compile ranges and explicit sizes are exercised before serving. Kernel-specific warmups include Qwen, FlashInfer, Triton, CuTeDSL and other registered paths. | Enumerate only the actual Qwen3.6 phase/role/M buckets and require a route hit for every declared bucket. |
| GDN warmup | Qwen GDN explicitly warms its chunk-64 Prefill kernels because profiling can otherwise omit them; the source notes that first-request autotuning after KV allocation can cause a latency spike or OOM. | Run real C512 plus declared tails through GDN/Attention before ready. First user traffic may not discover a GDN tactic. |
| Piecewise graphs | Static regions are captured around attention or other dynamic boundaries. Large shapes are captured first so smaller graphs can reuse memory; the workspace is locked after capture. | Generate static C++ launch segments around explicit Attention/GDN boundaries. Prefill Graph use remains conditional on a positive real-API result. |
| Late-JIT monitor | After warmup, unexpected Triton autotune/JIT, CuTeDSL, or TileLang compilation can warn or fail. | Add zero-tolerance counters for route miss, late allocation, sidecar build, plan mutation, and unplanned launcher initialization. |

The relevant official design contract says compilation is completed before
requests are served and documents the compile cache and piecewise graph model:
[vLLM `torch.compile` design](https://docs.vllm.ai/en/latest/design/torch_compile/).

Current vLLM also exposes Humming as an explicit linear/MoE backend, including
an NVFP4 adapter. It is not accurate to call Humming the automatic SM87
Qwen3.5 production path: in the inspected selector, Humming is late in the
automatic NVFP4 candidate order, and the A16 automatic route selects Marlin.
The useful integration lesson is capability qualification, loading-time
canonicalization, one-time configuration, and decision-free invocation.

## What Humming contributes

Humming specializes generated CUDA on exact N/K and data types, scale
semantics, block/warp shapes, pipeline stages, CTA residency, Stream-K,
`cp.async`/TMA mode, rasterization and epilogue choices. Its compiler hashes
the compiler version, flags, rendered source and included-header timestamps,
caches a cubin under a file lock, and loads the finished module for dispatch.
It can generate a palette of M intervals and compile those variants in
parallel.

The SM87 large-M W4A16 dataflow is the relevant design reference:

- packed B stays packed through global/shared movement and is decoded in
  registers immediately before BF16 tensor-core MMA;
- A, packed B and scales are pipelined with `cp.async` on SM8x;
- register buffers prefetch the next B fragment while the current fragment is
  consumed;
- producer traffic, decode and MMA are overlapped in a multistage pipeline;
- a persistent Stream-K scheduler and L2-aware M raster control ownership; and
- epilogue work is part of the planned kernel rather than an accidental
  runtime boundary.

SM87 has no native FP4 MMA in this implementation. Humming expands the packed
FP4 operand for BF16 MMA on SM87; native FP4 MMA is a later-architecture path.
That makes its operand pipeline relevant to Q3X's exact NVFP4 large-M work,
without making Humming a production dependency or granting copied code
provenance.

### Static M512 schedule extraction

The following is a T0 source/heuristic extraction, not a performance
benchmark. Humming's Orin auto-heuristic first calls NVML for memory-bus width
and maximum memory/SM clocks. All three calls return `NVMLError_NotSupported`
on this machine. The audit therefore supplied the analytical Orin
compute-bound threshold only to inspect the downstream M512 schedule. M512 is
well above that threshold and the reported tile is insensitive to this
substitution.

| Role | M/N/K | Block | Warp | Stages | Persistent grid | Stream-K | M-raster group |
| --- | --- | --- | --- | ---: | ---: | --- | ---: |
| Gate | 512/17408/5120 | 128x256x64 | 64x64x64 | 3 | 16 CTAs, 1 CTA/SM | yes | 2 |
| fused Gate+Up | 512/34816/5120 | 128x256x64 | 64x64x64 | 3 | 16 CTAs, 1 CTA/SM | yes | 2 |
| Down | 512/5120/17408 | 128x256x64 | 64x64x64 | 3 | 16 CTAs, 1 CTA/SM | yes | 1 |

This is useful in two ways. First, Humming couples several mechanisms rather
than judging `.cg`, B register decode, Stream-K, and A reuse as unrelated
single-variable experiments. Second, even where Gate/Up and Down share a base
tile at M512, their K/N geometry changes raster ownership. The plan must retain
separate role-specific tactics; one kernel template does not imply one tactic.

The current Humming automatic configuration is not directly production-ready
on this Orin because its device-discovery path does not handle the platform's
NVML omissions. Explicit tuning configuration can avoid that query. This is
additional evidence for offline selection and static deployment, not a reason
to discard its dataflow ideas.

## Frozen Q3X startup and hot-path gap snapshot

At the original audit revision, the runner contained useful specialization
assets: a strict
checkpoint manifest, resident weights, fixed-model bindings, FP8/NVFP4
sidecars, runtime Decode graph capture, and an OpenAI-compatible evaluation
server. The observed missing layer was one versioned artifact proving that
those assets formed the intended execution plan. Current implementation truth
belongs only to [`CURRENT_STATUS.md`](CURRENT_STATUS.md).

| Fixed fact rediscovered at the snapshot | Observed behavior | Reference form proposed by the audit |
| --- | --- | --- |
| FP8 Prefill layout | Engine startup inventories 208 projections, allocates exactly 7,214,202,880 bytes, launches one pack per projection, then synchronizes. Other M1/QKV/Down/Marlin sidecars are prepared in additional phases. | An atomic, authenticated `.q3x` offline pack with source-tensor and pack-ABI hashes; startup directly loads the final layout. |
| Tensor bindings and scalar scales | The loader reconstructs 64-layer name/variant/shape relations and synchronously reads quantization scalars while binding. | Generated tensor offset/type/shape/scale tables tied to the checkpoint hash. |
| Prefill route | Every tile and layer repeats variant, sidecar, shape, admission and backend checks; projection dispatch recomputes spans, aliases and route selection. | An immutable `PrefillExecutionPlan` with a typed launcher, tensor offsets, workspace, stream/event dependency and tactic for every operation. |
| Final prompt ownership | Commit `e7d6403` retained an exact, directionally positive all-prompt bulk candidate, but `Q3X_RUN_PREFILL_ALL_PROMPT_TOKENS_ADMISSION=1` still marks it as test-only and the default runs a scalar final-model pass. | Resolve positive and negative tail classes in the static scheduler, complete T3 qualification, and promote only the validated buckets. |
| Embedding | A C512 tile launches 512 individual embedding-gather kernels. | One batched gather in the planned C512/tail segment. |
| Prefill readiness | Decode graphs are warmed/captured and topology-checked; Prefill has no equivalent production preflight or route attestation. Several launchers still set function attributes per call. | Before ready, execute the declared C512 and tail paths, initialize launch attributes once, and publish route-hit proof. |
| Context capacity | The API defaults to 8,192 tokens and a 2 GiB request arena. | Versioned 64K and 131,072 capacity/memory plans; a deployment that cannot serve the locked prompt targets fails before ready. |
| Ready identity | Ready reports a small subset of sidecar statistics; health identifies the model but not the execution plan. | Expose deployment-plan digest, checkpoint/build/kernel identities, context capacity, active routes, workspace/KV plan, warmup result and zero-miss counters. |

The API construction path is also serialized: it completes model construction
before creating the listener. There is an existing tokenizer/resident-load
overlap in the one-shot CLI, but the API does not use it. That overlap is a
later process-startup improvement; it must not displace steady-state Prefill
architecture work.

## Reference deployment-plan checklist

The following was the audit's proposed checklist. The authoritative adopted
requirements now live in the
[`DeploymentPlan` contract](SDD.md#3-deploymentplan-contract) and release
attestation section of the SDD; changes belong there, not in this frozen
external audit.

`Q3xDeploymentPlan` should contain at least:

```text
schema version
Q3X source/build/ELF identity and kernel ABI
checkpoint repository, revision, shard and tensor hashes
SM87 device, driver/runtime and memory-capacity contract
precision contract and non-MTP declaration
kernel cubin/SASS hashes and static resource contracts
offline sidecar type, offset, length, alignment and source digest
KV, recurrent state, RoPE, scratch and workspace layouts
layer-role x phase x M-bucket tactic and typed launcher identity
stream/event dependencies and graph/segment capture set
supported context-capacity bucket
real-API qualification artifact and deterministic output oracle
```

The audit proposed fail-closed startup when this plan is absent or mismatched,
with no cuBLASLt, runtime JIT, generic unqualified-kernel, smaller-context,
altered-accuracy, or MTP fallback. It separated an explicit development
rebuild/qualification mode from a production-ready server.

After ready, these invariants hold:

- zero route misses and zero shape discovery;
- zero sidecar transforms and zero kernel compilation/autotuning;
- zero unplanned device allocations or workspace growth;
- zero per-call function-attribute initialization; and
- no silent plan or precision fallback.

## Adoption boundary

This frozen audit owns no mainline execution order. The SDD adopted the
authenticated AOT `DeploymentPlan` boundary, and the Roadmap decides when API
capacity, release identity, Prefill architecture, offline sidecars, warmup, and
qualification are implemented. Humming remains a schedule/dataflow reference;
vLLM remains the real external starting line. Neither a faster startup nor a
strong isolated Humming-shaped kernel constitutes whole-product progress until
the active architecture candidate returns through the target API witness.
