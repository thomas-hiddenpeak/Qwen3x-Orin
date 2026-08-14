---
q3x_document:
  id: q3x-prefill-reference-translation-matrix
  class: local-work-package
  status: active
  owner: prefill-maintainers
  authority: source-to-SM87 translation and selection record for WP-PREFILL-REFERENCE-TRANSLATION-v1
  effective: 2026-08-12
  last_reviewed: 2026-08-14
  supersedes: []
  superseded_by: []
  ssot_for: the active Prefill reference-translation work package only
  review_trigger: source pin, P40 geometry witness, selected AOT plan, or work-package closure
---

# Prefill reference-to-SM87 translation matrix

This document is the bounded working record for
`WP-PREFILL-REFERENCE-TRANSLATION-v1` under
`AC-PREFILL-SM87-MACROFEED-v3`. The V1 and V2 predecessors are retained only
as the closed diagnostic controls described by Current Status. This document
does not amend the product
target, stable SDD boundary, current implementation status, numerical
contract, or production route. Those authorities remain with the
Constitution, system and Prefill SDDs, Current Status, and numerical ledger.

The purpose of a separate work-package document is deliberate: source pins,
candidate hypotheses, and falsifiers change during development without
turning the SDD into an experiment notebook. When this package closes, this
document becomes frozen evidence or names its successor; only a genuinely
stable system boundary returns to the SDD.

## 1. Package boundary and return point

| Field | Binding |
| --- | --- |
| Product symptom | Cold/no-cache, single-request 40K--60K and approximately 130K Prefill remains below the owner-set API target and useful vLLM starting line |
| Parent candidate | `AC-PREFILL-SM87-MACROFEED-v3` |
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
| FlashInfer FA2 Prefill | Packed-Q GQA reuse, swizzled shared K/V, `cp.async`, BF16 HMMA QK/PV, FP32 online-softmax state, no score matrix | Template/JIT URI, paged-cache generality, dynamic plan buffers, FA3-only resources | Fixed Q24/KV4/D256/NHD/causal AOT plan; retain the Q64/KV32 incumbent control and the exact Q128/KV32 body, then schedule the latter as a 16-SM persistent L2 cohort | Rewriting Q64 in place, claiming Q256 despite the FP32-output-state resource closure, or changing the reduction/publication contract |
| FLA gated delta rule | Q/K work sharing, bounded C16/C32/C64 work graph, producer preparation, boundary-state ownership, adjacent output consumer | FP32 authoritative state across a chunk, WY/KKT associative state transform, global intermediate tensors | Keep the exact token-serial BF16 state core; fuse/shared-stage QK, gate, convolution, RMSNorm/SiLU, and O-projection boundaries around it | Deferring per-token BF16 state rounding or materializing multi-GB FLA intermediates |
| Mamba selective scan/SSD | Local work, explicit boundary state, state pass, and output reconstruction as separate schedulable phases | Associative FP32 scan state and model-specific recurrence | Use the work-graph decomposition and fixed resource planning without replacing the Qwen GDN transition | Calling a real-number associative scan bitwise-equivalent to the rounded recurrence |
| qwen35-thor | Separate Prefill/Decode entry points, model-specific chunk size, bulk Attention, persistent handles/workspace | SM110 TMA/UMMA/cluster mechanisms and its hardware/model thresholds | Preserve the separation and static workspace plan; translate physical work to SM87 primitives | Copying Thor thresholds, materialized score storage, or hardware-specific mechanisms |

## 5. Projection translation

vLLM distinguishes ModelOpt `W4A16_NVFP4` from native W4A4. On SM87 its
ModelOpt W4A16 method directly constructs Marlin; even an explicit
`--linear-backend humming` does not change Gate+Up or Down. An explicit
`--quantization humming` is also not a valid substitute for this checkpoint:
the installed adapter does not resolve its `MIXED_PRECISION` groups with
explicit per-module targets. The FP8 backend selector can nominally choose
Humming, but the installed ModelOpt handoff transposes the weight into a
non-contiguous last dimension and loses the layout attributes required by
the converter. Humming is therefore a source/dataflow reference here, not an
executable full-model comparison route. The project must compare and
translate W4A16 dataflows rather than projecting native-FP4 behavior onto
Orin. vLLM's compatibility handling may
collapse unequal fused-partition global scales by taking their maximum; that
behavior is not admissible here. Every Gate, Up, and FP8 partition keeps its
authenticated independent scale and publication boundary.

For BF16-A W4A16 on SM87, the inspected Humming heuristic first selects a
`M128N256K64` block, `M64N64K64` warp, eight-warps, three-stage,
one-CTA/SM seed. Gate+Up (`N=34816`) owns 136 N256 tiles and Down
(`N=5120`) owns 20. This is a source-derived starting structure, not a
performance result or immutable resource gate. Humming's dense-M table stops
at 8192 and extends that final configuration upward. Installed Humming 0.1.10
therefore handles `M=40000` as one call that selects the terminal catch-all
tactic: 16 persistent CTAs traverse all 313 M128 rows rather than issuing five
M8192 calls. The installed version has no L2 raster field; the newer frozen
`b18cfac` source adds role-specific Gate=2 and Down=1 M-raster groups. Those
version boundaries must not be conflated. P40/P60/P130 receive explicit Q3X
AOT range declarations rather than inheriting an unaudited `2^30` catch-all.

| Source-derived P40 role | M/N/K | M128 x N256 tile space | Installed 0.1.10 terminal tactic |
| --- | --- | ---: | --- |
| Gate | 40000/17408/5120 | 313 x 68 = 21,284 | 16 persistent CTAs, 3 stages, Stream-K tail |
| fused Gate+Up | 40000/34816/5120 | 313 x 136 = 42,568 | same base tactic; distinct fused role/scale contract |
| Down | 40000/5120/17408 | 313 x 20 = 6,260 | 16 persistent CTAs, 3 stages, K-heavy Stream-K tail |

This table is a source/heuristic extraction, not a benchmark. The first exact
Q3X implementation keeps the persistent DP body and defers Stream-K until its
finite-precision reduction identity is separately established.

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

Source inspection closes a misleading arithmetic branch. ModelOpt W4A16
NVFP4 Marlin accepts BF16/FP16 activation without activation quantization,
decodes FE2M1 plus E4M3 block scale to BF16, and executes the BF16
`mma.sync.m16n8k16` template. Its INT8 template specialization is not the
FE2M1 path. Re-expressing the complete listed P40 projection work as dense
integer limbs also fails the v2 projection allocation under an impossible
one-pass INT4 upper-bound assumption, so no IMMA sidecar or kernel is selected.
The authentic checkpoint provides no alternative K16 dictionary shortcut:
early, middle, and terminal layer searches find zero repeated packed-code
keys in more than fifty million Gate/Up/Down block instances, even with scale
ignored. The subsequent matched executed-work ledger confirms all 64 layers,
the complete 1.948-Pop projection total, SM87 W8A16/W4A16 arithmetic, and no
MTP/cache/activation-quantization deletion. It also corrects fused outer calls
versus physical launch counts. Projection work now resumes only inside the
complete role-specific v2 dataflow, not as another nominal-ISA skeleton.

## 6. Attention translation

The vendored FlashInfer path proves the owner's SM87 compatibility correction:
FA2 is usable on SM87 and already executes in the retained v10 incumbent. For
the fixed BF16 Q24/KV4/D256 shape its heuristic uses packed Q, Q64, KV32,
`cp.async`/HMMA online softmax, and no split-KV workspace. That path remains
an accuracy-unqualified direction, not the v2 numerical body.

The closed target-AOT v1 control is a separate exact Q128/KV32 body: 256
threads, 128 KiB shared memory, 254 registers per thread, zero local bytes,
one CTA/SM, and 7,500 CTAs per P40 layer. It performs 75,080,000 KV32
iterations over all 16 full-Attention layers. Its first P40 API request did
not return, so there is no valid target-AOT Attention time and the retained
13.634-second FlashInfer interval must not be assigned to it.

Direct Q256 is closed as the first v2 slice. Its exact FP32 output state alone
requires `256 * 256 = 65,536` FP32 accumulators, the complete SM87 register
file, before QK, softmax, addressing, or loop state. Moving that state to
shared memory requires another 256 KiB. A D128-times-two form reduces local
state but repeats QK/softmax and is diagnostic rather than terminal.

The selected executable slice therefore reuses the exact Q128/KV32 arithmetic
body and changes only whole-GPU ownership. Sixteen persistent CTA lanes work
on the same KV head in 118 snake-mapped epochs. Even epochs map lane `l` to
`16e+l`; odd epochs map it to `16e+15-l`. In the last epoch, 13 lanes perform
the same final-tile read/compute without storing, keeping the cohort aligned
for the next head at about 1.3852% extra KV iterations. No cooperative launch,
global lock, or cross-CTA barrier is needed because every real query tile has
one output owner and K/V are read-only. CTA-local Q remains 128; the claimed
effective Q of 2,048 exists only as an L2 temporal-service opportunity across
the 16 SMs.

The current logical K/V request is about 2.460 TB over P40. Perfect cohort
reuse would reduce distinct K/V payload to about 157.12 GB, but CUDA does not
guarantee block phase alignment and the existing grid may already receive
some natural L2 hits. These are design arithmetic, not a measured floor or
speed result. The complete real P40 API direction decides whether the
composition moves; only after a positive direction may bounded NCU verify
that DRAM/L2 service changed without changing the exact arithmetic body.

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
and FP32 operand bits are preserved. The selected v2 topology is a C64
three-stage graph: a token-parallel exact producer, 48 independent
value-head recurrence owners, and an exact rows8 RMSNorm/SiLU consumer. Two
producer slots and two raw-output slots bound the pipeline; authoritative
state remains one BF16 private span ordered by the recurrence stream. The
48-owner core removes the v1's artificial serialization of three value heads
behind each of 16 owners while preserving every token's BF16 state boundary.

The recurrence cell targets 48 CTAs of 256 threads, one value head and one
128x128 state per CTA, no more than 85 registers per thread, zero local bytes,
and at least three resident CTAs per SM. Falling to two CTAs/SM changes the
16-SM execution from one owner wave to two and reopens the architecture; it is
not accepted as a local tuning result. C64 state boundaries add about 94.30 GB
of logical state read/write over all P40 GDN layers. L2 persistence may reduce
its physical DRAM service, but the cost remains explicit. A later C256
super-epoch may be considered only if the complete direction proves this
boundary is causal. The old 256-thread, roughly 139-KiB, one-CTA/SM monolith
is retained only as the closed v1 control.

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
traffic, or publishing more boundary state. The target-first witness below
establishes the whole-prompt endpoint; only a later explanatory sweep may
measure that system trade, rather than treating chunking as a kernel feature.

### 8.1 Host-only whole-system design freeze

The first architecture translation is frozen as a non-executable host
descriptor for exactly P40, P60, and approximately P130. It describes one
64-layer request DAG with 14 physical execution groups: embedding, input
normalization, GDN QKVZ and A/B producers, GDN, GDN output/residual, full-QKV,
Q/K normalization and RoPE publication, full Attention, Attention
output/residual, post-Attention normalization, Gate+Up, Down/residual, and the
final handoff. Full Attention occurs at zero-based layers `3,7,...,63`; the
remaining 48 layers use the exact GDN constituent. Its canonical launch
ledger is `1 + 64 + 48 + 48 + 48 + 48 + 16 + 16 + 16 + 16 + 64 + 64 + 64 +
1 = 514`. Logical publication counts, projection tiles, Attention CTAs, and
GDN recurrence tasks remain separate domains and are never added to that
ledger.

The projection constituent declares five shape-specific roles: NVFP4
Gate+Up, NVFP4 Down, FP8 GDN QKVZ, FP8 full-QKV, and FP8 Attention output.
The earlier host freeze proposed one M128/N256/K64 family with only a small
M-raster distinction. Executable traffic accounting has superseded that
projection portion of the freeze: Gate+Up, K-heavy Down, and FP8 require three
different whole-P40000 persistent schedules. They may not inherit one tile,
raster, cache policy, or segment count merely because the arithmetic primitive
is shared. NVFP4 and FP8 operands keep independent packed-weight and scale
contracts. The GDN A/B producer remains a separate paired BF16 plan with two
independent weights, accumulators, and BF16-RNE publications; pairing does not
turn the two model tensors into one numerical operation.

The first executable NVFP4 control used 39 M1024 launches plus one M64 tail
per layer, but each M1024 body serialized four complete M256 epochs. Static
source accounting proves that boundary cannot fit the projection allocation:
the realistic NVFP4 weight floor is 3.012915 TB across the family, or 14.7115
seconds at 204.8 GB/s, and even impossible four-row reuse for every NVFP4 role
plus optimistic segmented FP8 reuse remains 8.7883 seconds before other
traffic. The exact body is retained as a same-ELF numerical control only.
The selected successor follows the proven whole-M principle in vLLM Marlin's
persistent stripe/Stream-K scheduler and Humming's L2-aware raster, translated
to fixed AOT SM87 kernels: one whole-P40000 launch per projection role, 32
persistent CTAs, and role-specific two-dimensional M/N cohorts. Gate+Up shares
each A stage while preserving two independent accumulations and publications;
Down balances H reuse against B-panel reuse under its much larger K and short
N grid; FP8 receives a separate schedule rather than retaining forty M1024
segments. The source hashes, exact oracle, final compiler resources, and
communication rejection are frozen in the
[`NVFP4 exact-control closure`](metadata/qwen36-27b-sm87-bulk-v2-nvfp4-exact-control-closure-2026-08-14.json).

That translation now has three separately compiled default-off CUDA
constituents. Gate+Up maps the shared-A/independent-B invariant to
M64/N64/K64 cells in a 4M-by-8N cohort. Down maps its K-heavy short-N shape to
M64/N256/K64 cells in an 8M-by-4N cohort. FP8 maps the authenticated
K16-major Marlin payload, including partition-private scales, to
M64/N128/K64 cells in a separate 4M-by-8N cohort. Each uses 32 persistent
CTAs, three `cp.async.cg` stages, two register-feed stages, ascending full-K
FP32 accumulation, no split-K/global partial C, and one cooperative outer
launch for each complete P40000 model role. Gate+Up retains two independent
NVFP4 scale and rounding authorities through the fused SiLU-times-Up
publication; Down retains its two BF16-RNE boundaries around residual add;
FP8 retains the original raw E4M3 bytes and the distinct QKV/Z/O output
partitions.

Their precomposition evidence stopped at T0 compiler/resource observations
and T1 reduced-domain synthetic exactness. Gate+Up/Down/FP8 compile at
respectively 107/111 and 89--93 registers/thread, zero stack/spill/local
bytes, and 38,400/52,224/49,152 bytes of dynamic shared memory with a static two-CTA/SM
capacity; Down's exact symbol requires startup dynamic-shared opt-in. The
oracles exercise M64 correctness cells, not the full P40000 dataflow. No NCU
counter, real-payload timing, scheduler co-residency, DRAM reduction, or L2
residency had been measured. The three same-kernel reduced-domain synthetic
exact oracles pass, but that evidence did not validate the cohort mapping.

The complete V2 composition has now falsified the service-order hypothesis at
the product boundary. Its cold/no-cache EvalScope 1.9.1 P40000-plus-one request
received zero bytes and timed out after 680.73 seconds with 0/1 success. It
therefore has no TTFT or throughput result. The server cancellation line's
layer 37 / quantum 38 values are host submission progress rather than
GPU-completed-layer timing. The only permitted causal NSys follow-up captured
120.002145 seconds of the same real route and 147.337288416 seconds of
overlapped aggregate kernel time: Gate+Up is 58.8998%, Down 25.3737%,
Attention 7.5434%, FP8 3.0590%, the four GDN families 4.6378%, and other work
0.4863%. Gate+Down is 84.2735%; all projections are 87.3325%. This closes V2
as a performance architecture without claiming a cache-hit rate, completed
generation, accuracy, or production authority. Exact evidence is frozen in
the
[`Bulk V2 P40 rejection record`](metadata/qwen36-27b-sm87-bulk-v2-p40-rejection-2026-08-14.json).

The canonical system contract contains 39 typed resource edges and 13 typed
event edges. Gate and Up are paired inside one CTA while retaining independent
reduction and BF16-RNE authority. Processed K/V state is staged for the private
request transaction but remains unpublished; GDN output is layer-local until
its O projection consumes it. Convolution history, recurrent state, position,
final KV state, and final hidden state become Decode-visible only through one
whole-request commit.

The full-Attention constituent fixes Q128/KV32, two K/V shared-memory stages,
FP32 online softmax, no score matrix, no split-KV, and the exact model
preprocess contract: centered Q/K RMSNorm, 64 rotary elements (32 NeoX
pairs), bit-exact per-head gate split/copy, and bit-exact V publication. The
first host-only GDN constituent was a layer-long 16-CTA v1 design. Each CTA
owned one Q/K group and three value-head state chains. The failed target API
smoke has now closed that ownership as a performance architecture. The v2
successor uses the 48-owner C64 producer/recurrence/consumer graph above. Cold
state and convolution history still begin from register zero; no cooperative
grid barrier or in-kernel global commit is permitted. Completion of a bounded
recurrence epoch may publish only layer-private readiness. It is not
`PrefillStateCommitted`: state remains invisible to Decode until the single
whole-request commit, and cancellation discards every unpublished span.

Internal buffering is therefore explicit but local: three projection stages,
two Attention K/V stages, and the v2 GDN's two C64 producer plus two raw-output
slots. No cross-operator double- or triple-buffered GPU pipeline is selected
yet. The DAG permits GDN
QKVZ and A/B to overlap after input normalization, but stream assignment,
simultaneous resource fit, and any performance benefit remain unqualified.

The precomposition projection contract advanced to ABI-major 3. Its
projection-successor manifest keeps the 496 logical
checkpoint roles and 304 fused outer operations distinct from physical work:
48 GDN-QKVZ, 16 full-QKV, and 64 Attention-output launches form the 128 FP8
whole-role receipts; 64 Gate+Up and 64 Down launches form the 128 NVFP4
receipts; 48 BF16 A/B launches complete the projection ledger. The old 5,120
FP8 and 2,560 NVFP4 control launch counts must both be zero in a complete v2
receipt. Caller-filled receipts are terminal observations, not launch
capabilities. The completed executor later binds those fields into a
success-only ABI-major 4 `target-prefill-witness-v17`; the timed-out request
emitted no V17 terminal receipt.

The separate v2 `RequestState` now realizes the data-plane lifetime side of
this host contract: one exact 5,075,652,608-byte device allocation, only
78,446,592 cold-reset bytes for GDN persistent state/history, the owner's five
borrowed streams, an externally owned 1,280-byte device control plane, and a
private eight-byte pinned greedy handoff. Its fixed D2H source prevents an
executor from redirecting the handoff. The private startup capability,
executable natural-layer chain, terminal state publication, and distinct
default-off `sm87-bulk-v2-p40` route are now composed. They remain development
infrastructure only because the request did not complete and the architecture
is performance-rejected.

The selected translation successor is
`AC-PREFILL-SM87-MACROFEED-v3`. Its complete default-off, test-only P40000
source composition is now present. It preserves the completed v10 five-panel
API/control and whole-prompt FlashInfer Attention substrate, while Gate+Up,
K-heavy Down, and FP8 use separate non-cooperative M128/N256/K64, three-stage,
16-persistent-CTA feeds. Each role keeps its own authenticated payload and
scale semantics; Gate and Up retain independent reductions/publications
through their same-CTA SiLU-times-Up consumer, and Down retains its two BF16
publication boundaries around residual addition.

Exact GDN retains 48 value-head owners, pre-round same-token output use, and
the per-token BF16 recurrence/publication semantics. One whole-P40 convolution
launch is followed by eight ordered M5000 recurrence/RMSNorm/SiLU-gate
macrochunks, reducing each of the 48 GDN layers to nine physical kernels
without using WY/KKT/SSD, associative scan, or FP32 authoritative chunk state.
Cancellation polling remains bounded and failure or cancellation cold-resets
the mutable request-arena prefix without altering the immutable RoPE suffix.

The private startup package authenticates 256 physical projection artifacts
from 400 checkpoint sources and seals 256 immutable layer-role launch bindings
against their real payload/scale ranges, CUDA resource facts, owner/allocation
catalog, device ordinal, and physical CUDA-device identity. The request hot
path borrows those bindings; it performs no CUDA resource/device query, JIT,
repack, autotune, or tactic discovery. The actual executor runs all 64 layers
in natural order and can seal its ABI-major-1 transaction only after observing
128 FP8, 64 Gate+Up, 64 Down, 48 BF16 A/B, 432 GDN, 16 whole-prompt Attention,
and 80 Attention-preprocess physical operations, complete fill/drain, and
request completion. A caller-filled forecast or partial layer receipt cannot
manufacture that transaction.

The complete route identifies itself as
`q3x.sm87.ac-prefill-sm87-macrofeed-v3.native-p40-target-aot-whole-model.v1`
and reserves `target-prefill-witness-v18`. Its receipt authority is only
`physical-execution-only`; numerical qualification and production dispatch
are both false. V3 returns to P40 first. No real P40 request has yet measured
this composition, so none of these implementation facts is a performance,
full-model accuracy, release, or production claim.

The current controller orders the first-token finalizer before the V3 state
commit. Consequently its continuous prompt interval includes LM-head/argmax
and cannot qualify as the pure-Prefill interval defined by the subsystem SDD.
V18 fails closed on pure-Prefill promotion while keeping the authenticated
64-layer transaction independently complete; the first P40 direction gate
uses external EvalScope TTFT/New Prompt Throughput. Subtracting the finalizer
duration would produce non-contiguous active time and is not an admissible
metric. A positive architecture must later move or independently instrument
the physical `PrefillStateCommitted` boundary before performance promotion.

The earlier default-off target-AOT admissions supplied the byte-authenticated
asset and lifetime prerequisites now consumed by V3. One admission contains a
real-byte host transformation and NVFP4 Gate+Up/Down CUDA bodies, while a
further test-only slice implements an engine-lifetime device owner and an
all-or-nothing real-checkpoint uploader. That uploader audits the exact
64-layer resident source inventory, transforms one bounded artifact at a time,
and issues a receipt only after an independent device readback matches the
host payload SHA-256. The Engine startup trigger performs the upload and then
attaches the retained owner to `ModelWeights` through one private all-or-
nothing lifetime capability; it skips mutually exclusive old Prefill
projection sidecars and exposes no naked view. Those prerequisite milestones
had no execution authority by themselves; the separately sealed V3 package,
runner transaction, and V18 API contract now supply the default-off execution
boundary without granting production dispatch.
A provenance-frozen Release/SM87 probe at `855a7cb` has run that online loader
against the pinned real checkpoint: 192 source tensors produced 128 artifacts
in one 9,625,927,680-byte device arena, independent readback closed the payload
catalog at
`367572d8f5aab87c655695fc621562e0e88cb5d1a9656370353d55ab1c4ebdbe`,
and private owner-backed attachment completed with all older Prefill sidecars
absent. The source probe remains `fail` solely at its immediate in-process
memory-recovery check. The first retained post-exit `nvmap` snapshot was
recovered from a completed, hash-frozen Codex rollout. Its 2,012,087 reusable
pages leave a 99,438,592-byte adjusted residual, below the fixed 256 MiB
tolerance, while the IOVMM orphan table is empty and only allowlisted desktop
clients remain. A later complete process/IOVMM/FSI/handle snapshot corroborates
that no matching owner survived. The derived `no_owner_leak` diagnosis
preserves rather than overwrites the source status. The tracked
[`real-checkpoint preparation record`](metadata/qwen36-27b-sm87-target-aot-real-checkpoint-preparation-2026-08-12.json)
freezes both layers of evidence and their narrow authority.

The following clean Release/SM87 execution at `9d0613a` passed the combined
layer-0 real-model M192 Gate+Up and Down-plus-residual bitwise Oracle, including
the full M128 region and predicated M64 tail, and passed the same-ELF resource
and exact-geometry gates. Its parent captured the child identity and canonical
Jetson owner tables immediately after exit. The original lifecycle report is
preserved as `inconclusive`; a strict parser at `5687871` re-derived the same
immutable raw snapshot as `no_owner_leak` with 20/20 criteria. The tracked
[`layer-0 M192 Oracle record`](metadata/qwen36-27b-sm87-target-aot-layer0-m192-oracle-2026-08-12.json)
freezes the numerical, resource, lifecycle, and source-status boundaries.

The current source slice adds persisted target-AOT creation and direct loading
to that same default-off lifetime admission. When asset preparation is
selected, its three legal modes are: online prepare only; online prepare plus
a create-only bundle and externally supplied expected payload catalog; or
direct load from a bundle plus that external trust root. The persistence ABI
is one fixed little-endian 9,626,456,064-byte file: a 4,096-byte superblock and
128 layer-major records, each with a canonical 4,096-byte header followed by
its payload. Headers retain source inventories, manifests, and transform
receipts but never runtime pointers or upload receipts.

Direct loading authenticates the live pinned checkpoint and complete source
inventory, superblock, every record header, exact layout/offset/scale identity,
every payload, and both complete catalogs before any CUDA operation or
target-arena allocation. It performs no source-tensor D2H and no online
transform/repack; each persisted payload is reread, rehashed and scale-domain
checked before H2D, independently read back, and included in the externally
authenticated catalog before the private owner is committed. Offline creation
writes those device-readback bytes during the
existing online transaction and publishes the completed file with create-only
atomic rename semantics. It never overwrites an existing asset.
Publication is a separate file transaction from later Engine attachment: a
durably published asset remains recoverable by authenticated direct load even
if attachment or runner creation later fails. Crash-left same-target temporary
files fail closed before another full reservation and require manual audit;
the runtime does not guess that an unknown file is safe to delete.

A clean Release/SM87 create transaction and fresh-process direct-load
transaction at `27f5c71` have now exercised the persisted path against the
pinned real checkpoint. The schema-v4 create record published the fixed
9,626,456,064-byte bundle for 128 artifacts and 192 authenticated sources.
The schema-v4 load record performed two complete host-authentication passes,
read exactly 19,252,912,128 bundle bytes, performed zero source-tensor D2H,
reproduced both authenticated catalogs, attached the private owner, and
repassed the layer-0 M192 Oracle. Both child probes preserve `status=fail`
solely at their immediate in-process memory-recovery check. Their strict
canonical Jetson parent reports independently classified the post-exit state
as `no_owner_leak` and set `combined_lifecycle_accepted=true` without
rewriting the child status.

This passes only the persisted create/direct-load lifecycle admission. No
admission contains a production selector, API wiring, public execution route,
or production dispatch authority, and the observations have no startup or
model-performance authority. The narrow M192 result still does not qualify
the complete model. Exact identities, catalogs, byte accounting, source
statuses, and claim boundaries are frozen in the
[`persisted create/direct-load record`](metadata/qwen36-27b-sm87-target-aot-persisted-create-direct-load-2026-08-13.json).
The next architecture gate is the complete 64-layer target-AOT composition of
FP8 and NVFP4 projections, Attention, exact GDN, buffers, state, and handoff,
not another isolated loader or local tile scan. Neither payload
authentication, lifetime attachment, persistence, nor this narrow Oracle can
promote the production path.

The pre-CUDA implementation contract is now frozen as a second host-only
milestone:

- NVFP4 and admitted FP8-Marlin projection paths have distinct decode, scale,
  MMA, full-K accumulation, exceptional-encoding, and BF16 publication
  identities. Gate and Up retain independent weights, scales, reduction trees,
  and BF16-RNE values inside one CTA; their private lifetime ends only after
  `SiLU(Gate) * Up`, with no global intermediate or cross-CTA handoff;
- full Attention freezes both preprocess and core instruction order: the
  production D256 Q/K RMS reduction tree, centered BF16 norm-weight `+1`,
  `rsqrtf`, BF16 publication before NeoX RoPE, RoPE FMA order and passthrough
  tail, QK/PV MMA traversal, online-softmax exponent and denominator tree,
  probability BF16 boundary, reciprocal, stable sigmoid, and final BF16
  publication;
- GDN names recurrent state as `[head,value,key]` with key contiguous and
  freezes the exact-C16 CUDA candidate's Q/K reduction tree, scalar transcendental
  operations, key-ascending recurrence, per-token BF16 state, pre-round output,
  and norm/gate epilogue. It explicitly does not borrow qualification from the
  deployed Chunk64 WY/WMMA family; and
- the packed transformation receipt binds actual source-weight/source-scale
  byte digests, exact permutation counts, payload byte digest, and a complete
  NVFP4 E4M3FN block-scale forbidden-code scan. FP8 weight bytes instead keep
  the admitted Marlin raw-code semantics. A metadata-derived pseudo-digest is
  not an authenticated payload.

These declarations are implementation constraints. The new asset builder
proves that supplied byte intervals can
be hashed, domain-scanned, permuted and replayed bit-exactly; its synthetic
host tests do not authenticate the deployed checkpoint. The unexecuted,
compiled NVFP4 bodies encode the intended three-stage K64 feed, decoded-B
register double buffer, full-K FP32 ownership, same-CTA Gate/Up BF16 lifetime
and Down residual boundary. Static compilation currently reports Gate+Up at
246 registers/thread and Down at 210, both with zero stack and zero spill,
using 76,800 bytes of dynamic shared memory; SASS contains `LDGSTS`, `LDSM`,
and BF16 `HMMA`. The later real-checkpoint M192 execution qualifies these
resource and geometry facts only for that private layer-0 Oracle and same ELF.
A typed
device-upload receipt is now issued only by the default-off loader after
binding the authenticated host digest to an owned allocation and exact payload
subrange, observing upload completion, reading the exact device range back,
and matching its SHA-256. The public schema and binder remain non-authoritative
and cannot manufacture those facts from a raw pointer. Real-checkpoint
preparation, readback, and private attachment have executed under the bounded
evidence above; they grant no execution authority. The CUDA bodies must pass
complete-composition and complete-model accuracy gates beyond the already
passed narrow layer-0 M192 Oracle, and a separately reviewed admission launcher
must reach the real API before any performance or production qualification bit
changes.

The earlier online-preparation probe reported
`target_prepare_attach_milliseconds=699705.551133`. This is diagnostic timing
from a correctness-only run, not a startup or model-performance baseline. It
does establish that production cannot repack and independently reread all
9.626 GB online. The persisted-bundle code and admitted real-checkpoint
create/direct-load pair above are the implementation response; neither their
wall times nor this correctness/lifecycle evidence is a startup result.

## 9. One bounded reference-geometry witness

After the source matrix is frozen, obtain exactly one valid target-first
reference-engine witness before native implementation. An invalid execution
does not consume that valid-witness slot:

- real checkpoint and tokenization;
- one cold/no-cache, batch-one P40 API request with one output token;
- MTP and prefix-cache reuse disabled;
- an explicit target-like whole-P40 vLLM scheduled-token budget of `40000`;
- JIT, kernel warmup, and deterministic tuning completed before timing;
- exact backend, model, token count, route, cache, memory, and process
  identity retained; and
- EvalScope TTFT plus vLLM's scheduled-to-first-token Prefill-phase telemetry
  kept as distinct observables. The latter is not mislabeled as pure GPU
  kernel time or as the logger's local-interval aggregate prompt throughput;
  that interval's exact duration is not established by the log line.

This first witness reconciles the owner's known vLLM behavior with the exact
current configuration at the production target before any explanatory
geometry sweep. Budgets `8192`, `4096`, and `2048` are deferred: they may be
used later to explain a target result, but cannot delay or invalidate the
whole-P40 gate. The witness is not repeated for every Q3X change and does not
select a Q3X production dependency.

The locked launcher is
[`run_vllm_p40_geometry.py`](../tools/evaluation/run_vllm_p40_geometry.py).
It runs a fresh stock-vLLM server with budget `40000`; uses distinct
hash-pinned P40 warmup and measured token-ID corpora;
pins EvalScope `1.9.1` in offline mode; and requires exact Prometheus deltas
for one no-cache request. The formal path also pins the Orin/Torch/CUDA
runtime, brackets its CUDA identity probe with clean-host admissions, audits
the actual vLLM process group and loaded runtime libraries, and requires a
measured-request `tegrastats` GPU/thermal envelope. Every generated cache,
log, command receipt, metric snapshot, cleanup receipt, and result stays
under `/.q3x-work/`; child processes receive a repository-local `HOME`. A dry
run validates non-GPU identities and prints the complete plan without
starting a server or probing the CUDA device. The formal run requires a clean
Git worktree and a previously unused output directory.

Executable deleted mappings remain forbidden. The only deleted mapping class
accepted from the vLLM process group is a non-executable `rw-s`
`/dev/shm/sem.*` POSIX semaphore: each such mapping is retained in the runtime
receipt and must remain stable across warmup and measurement.

The vLLM ZeroMQ base is the shortest safe repository-local path: its complete
UUID socket path is checked against the 107-byte Unix-domain limit. Each
run requires no stale UUID socket before startup, and after the owned
process group is empty the launcher removes only inode-matched socket entries
created under `/.q3x-work/`, records them, and proves that none remain.

vLLM imports its vendored DeepGEMM extension while probing backend
availability, so the witness also pins that wrapper, package entry point,
mapped ELF hash, and build ID. This is load-time identity evidence, not a
claim that DeepGEMM executes on SM87: the pinned vLLM platform gate enables it
only for SM90, SM100, or SM120 families.

Before the run, the Jetson clean-host preflight must pass with `tegrastats`,
CPU/process inspection, and GPU-device-handle ownership. If another workload
owns a critical resource, the run does not start and no timing is retained.
Thermal admission uses hysteresis rather than changing the hard limit: three
consecutive five-second samples of CPU/GPU/TJ at or below `65C` are required
before server startup, before the measured request, and before post-release
admission; every measured sample must still remain strictly below `70C`.

### 9.1 First target-first execution feedback

The first formal `40000`-budget execution from clean commit `fe626be`
completed its real API transaction and cleanup, but it did not produce a
performance result. The measured interval exceeded the fixed thermal gate,
reached `81.812C`, and later showed CPU clocks falling from 2201 MHz to about
1.1 GHz. The launcher consequently wrote `valid=false`, retained no result,
and left the native incumbent and the owner-established 4.3K tok/s reference
unchanged. Its exact classification and artifact hashes are frozen in the
[`invalid P40 reference-witness record`](metadata/qwen36-27b-vllm-p40-target-witness-invalid-2026-08-12.json).

Its backend log remains valid route evidence, with an important streaming
correction. Line 129 is the warmup POST's HTTP response-start access
observation, not request completion; line 130 records
`_compute_slot_mapping_kernel` JIT while that warmup inference/stream can still
be active. The later `3999.7 tok/s` line is the aggregate logger interval after
the warmup's prompt accounting settled. It is before the measured POST's
response-start observation and is not the measured P40 result. The vLLM
0.26.0 logger accumulates computed iteration prompt tokens into a local
interval. Future witnesses explicitly fix `VLLM_LOG_STATS_INTERVAL=10.0` and
bind the relevant `envs.py`, server logging task, and `LoggingStatLogger`
source identities. Ten seconds is the configured sleep/trigger interval; the
printed rate's denominator is the logger's actual monotonic time since reset,
which is not printed and need not equal ten seconds exactly. A long single
iteration can be accounted atomically in a later bucket and print a
service-window rate unrelated to that iteration's elapsed time. Future
witnesses retain this structured backend-log timeline even when their
performance result is invalid, but join it with request TTFT, explicit
request/engine boundaries, and Prometheus scheduler geometry before any phase
or performance interpretation. Access-line order alone never establishes
request completion.

The retained warmup is now reconciled across all three surfaces in the
[`warmup metric reconciliation`](metadata/qwen36-27b-vllm-p40-warmup-metric-reconciliation-2026-08-12.json).
The singleton-joined cold/no-cache 40K warmup recorded `40000 /
108.52464322606102 =
368.579880` token/s from the server's scheduled-to-first-token Prefill
interval, `40000 / 108.58992791175842 = 368.358289` token/s from server TTFT,
and `40000 / 108.62519 = 368.238711` token/s when EvalScope prompt tokens are
divided by external TTFT. EvalScope's separately printed `368.2478` token/s
is total input-plus-output throughput over test duration and is not relabeled
as pure prompt throughput. The external-to-server-Prefill gap is only
`100.546774 ms`, whereas the nearby logger sample is `3999.7` token/s, or
`10.851650x` the request-bound server rate. The vLLM logger source semantics
and fresh-process singleton join support this metric interpretation, although
the evidence lacks a stable cross-surface request ID and therefore has no
performance authority. It is consistent with the logger sample being a
window-accounting observation, not a pure-Prefill measurement.

These warmup figures have metric-semantics authority only. That warmup did not
have a formal thermal/frequency envelope, while the later measured request was
thermally invalid; neither becomes a vLLM performance baseline. The mismatch
instead requires auditing the stock r5 route, preparation, and configuration
against the owner's optimized vLLM route. It does not lower the owner-observed
4.3K reference or the locked 40K--60K product target.

The r5 thermal rejection also exposed a retained-evidence gap: the measured
telemetry validator raised before the launcher captured
`metrics-after.prom`, `cache-after-measured.json`, and the post-request runtime
snapshot. Their absence is not evidence that scheduler state, compilation
cache, or process identity remained stable. Before another reference GPU run,
the launcher should best-effort capture those diagnostic-only snapshots while
the owned server is still alive even when timing has already been invalidated;
doing so must not restore timing authority to the rejected sample. This
control-flow change is deferred here rather than mixed into the access-log
semantic correction.

The retained log currently gives the following causal comparison surface:

| Log observation | Captured value | What it can explain | What it cannot prove |
| --- | --- | --- | --- |
| Requested linear backend | `auto` | The runtime, rather than the launch command, selected the concrete projection path | That the selected path matches the owner's optimized vLLM configuration |
| Actual FP8 linear kernel | `MarlinFP8ScaledMMLinearKernel` | Projection timings belong to Marlin FP8, not Humming | Native Q3X projection parity or the owner-observed 4.3K result |
| Attention backend | `AttentionBackendEnum.FLASHINFER` | Attention executes through FlashInfer on SM87 | A request-level Attention duration without a matched phase interval |
| GDN Prefill backend | Triton/FLA | GDN executes through the Triton/FLA path | Exact equivalence to the Q3X per-token BF16-state contract |
| Humming selected | `false` | This run does not exercise Humming's packed W4A16 pipeline | That Humming is unavailable or unsuitable on SM87 |
| Compile and warmup | range `[1,40000]`; `torch.compile` 133.46 s; initial profile/warmup 99.82 s; AOT function saved | Startup specialization and cache readiness are material parts of the observed route | Timed-request Prefill or TTFT performance |
| FlashInfer autotune | enabled | The Attention route may select a shape-specific plan during preparation | That every relevant shape was tuned or reused during measurement |
| Inference-time JIT warning | `_compute_slot_mapping_kernel` at line 130, immediately after the warmup streaming response-start access at line 129 | Runtime compilation occurred during the continuing warmup inference; the response-start access did not close the request | Its cost in the later invalid measured request, or any request phase inferred from access order alone |
| Prompt-throughput logger | `3999.7`, then `0.0` tok/s; one HTTP response-start access-log observation previously emitted | Joined with the harness, this is a post-warmup accounting interval before the measured response start | The interval's exact duration, per-request elapsed throughput, pure Prefill throughput, measured-P40 throughput, or request completion from the access line |

Joined with the current Q3X request witness, those facts narrow the gap without
pretending that the invalid run is a reference score:

| Comparison surface | Current observable difference | Causal interpretation and next discriminator |
| --- | --- | --- |
| Request geometry | The matched audit closes vLLM's batch-one/no-cache P40 as one scheduler work unit; Q3X v10 executes five M8000 fill panels and five M8000 drain panels around each whole-P40 core | Scheduler grain and kernel grain remain distinct: pinned Marlin internally splits every relevant P40 outer call into 40 row chunks. V2 proved that replacing those boundaries with a 32-CTA cooperative monolith is not sufficient; V3 instead composes role-specific non-cooperative macro feeds. |
| Projection route | vLLM logged Marlin FP8 and source-pinned ModelOpt NVFP4 Marlin; Q3X spends about 79% of request time in Gate/Up, Down, and FP8 projection families | The matched ledger finds 128 fused outer calls but source-derives 5,120 Marlin kernels per quantized family, versus Q3X v10's 1,040 FP8 and 128 NVFP4 receipt counts. Launch count is not the gap; packed layout, scale partitions, CTA work, decode/MMA utilization, and useful data residence remain the discriminators. Same-shape NCU/NSys follows a real-API direction. |
| Attention route | Both observed paths name FlashInfer, while Q3X still spends 13.35% in whole-prompt Attention | Backend selection alone cannot explain parity. Preprocess layout, Q/K/V publication, cache format, selected tactic, and request-shape specialization must be compared on the joined request interval. |
| GDN route | vLLM selected Triton/FLA; Q3X retains its own exact per-token-BF16 recurrence path | This is a genuine architecture difference, but FLA's algebra cannot be assumed numerically equivalent. The native layer-long fused candidate must preserve Q3X's exact recurrence boundary and prove accuracy before performance promotion. |
| Shape preparation | vLLM completed `torch.compile`, warmup, AOT save, and FlashInfer autotune for the declared range before timing; Q3X's selected production route has no equivalent authenticated whole-system plan | The transferable mechanism is pre-request shape and tactic closure, not request-time JIT. Q3X therefore freezes P40/P60/P130 AOT plans and forbids request-time discovery. |
| Runtime compilation | A slot-mapping kernel JIT-compiled immediately after the first streaming response-start access | The line belongs to continuing warmup inference, not a completed-request boundary. It proves the warmup route exercised late JIT, while the current log cannot place such a cost inside the later invalid measured interval. A future valid witness must join JIT/cache events to explicit request boundaries and reject cache mutation during timing. |

The current evidence therefore points first to whole-request dataflow and
physical work geometry, not Python or HTTP overhead: Q3X's retained P40
witness differs from server TTFT by only 3.730501 ms while the GPU request is
about 101.8 seconds. It does not yet allocate the remaining gap to individual
vLLM kernels; that requires a valid, route-matched request and phase-aligned
profile.

Every future reference run therefore keeps three evidence classes separate:
the backend/startup log identifies the route and possible causes; the
request-bound EvalScope/API record measures product-visible TTFT; and an
explicit server phase interval, when available, measures scheduled-to-first-
token Prefill. A conclusion that one backend explains a performance gap must
join all three to the same request identity rather than infer latency from a
nearby log line.

This feedback closes an unchanged rerun, not the target. The captured route
used stock vLLM `0.26.0` with auto linear selection, Marlin FP8, FlashInfer
Attention, and Triton/FLA GDN; Humming was not selected. Before another GPU
run, the package must reconcile that exact route with the owner's known
optimized vLLM startup, JIT/AOT-cache, backend, Humming, and scheduler
configuration. Only a materially different, hash-bound route returns to the
same target-first P40 gate. The thermal limit is not relaxed, and smaller
budgets remain explanatory tools rather than substitutes for the product
workload.

## 10. Package closure

This package closes only when all of the following are true:

1. source pins and invariant/ISA/SM87 translations are complete;
2. the V3 composition reaches one valid clean-host native P40 API witness and
   is selected or rejected against the current native incumbent; explanatory
   smaller prompts or component cells cannot delay or replace that result;
3. projection, Attention, GDN, buffer, state, and handoff plans form one
   authenticated AOT candidate with no undeclared fallback;
4. each added-compute-for-movement trade has an equivalence/resource ledger;
5. the candidate has a direct clean-host real P40 API return point and
   falsifier; and
6. Roadmap names the next implementation package or records architecture
   closure.

The thermally invalid stock-vLLM P40 attempt remains route and metric-semantics
feedback only. Reconstructing the unknown optimized reference route does not
block the native V3 return point and cannot lower the owner-established 4.3K
starting line. Until the native P40 result is recorded, source review and the
present implementation change no default route.
