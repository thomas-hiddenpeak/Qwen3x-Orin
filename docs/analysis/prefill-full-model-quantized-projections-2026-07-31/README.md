# Qwen3.6-27B SM87 full-model quantized Prefill projection design

Date: 2026-07-31  
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill  
Target: at least 2,000 prompt token/s on the real API path; no MTP and no
cuBLASLt production fallback

## Decision

The next Prefill step is not another isolated Gate kernel. It is a gated,
layer-major execution plane in which **every projection family** consumes a
quantized activation slab and uses an SM87 integer Tensor Core kernel:

1. use A8 first to validate the complete dataflow, sidecar lifecycle, family
   schedulers, real-API timing, and accuracy measurement;
2. move Gate+Up and Down to calibrated A4W4, then move the FP8 attention
   projections to A4W4;
3. promote only after the cumulative full-model route exceeds 2,000 token/s
   and passes the external capability gate.

A8 is an integration milestone, not the terminal performance architecture.
Even an impossible 100%-of-peak A8 implementation cannot reach 2,000 token/s
at P3847 before attention, GDN, normalization, launch, and allocation costs.
The terminal route must therefore be A4-dominant, shape-specialized, and
layer-major.

No synthetic matrix result may admit or reject a performance change. Synthetic
inputs remain useful only for correctness enumeration and smoke tests. The
first performance observation for each vertical slice is one real-checkpoint,
real-API generation; profiling and statistical harness expansion follow only
after that observation is positive.

## Fixed model inventory

The checkpoint and `src/runtime/model_weights.cpp` fix the relevant model
geometry:

| Family | Layers | Projection shapes, `N x K` | Count |
|---|---:|---|---:|
| MLP Gate | 64 | `17408 x 5120` | 64 |
| MLP Up | 64 | `17408 x 5120` | 64 |
| MLP Down | 64 | `5120 x 17408` | 64 |
| Linear attention | 48 | QKV `10240 x 5120`, Z `6144 x 5120`, O `5120 x 6144` | 144 |
| Full attention | 16 | Q `12288 x 5120`, K `1024 x 5120`, V `1024 x 5120`, O `5120 x 6144` | 64 |

There are 192 NVFP4 MLP matrices and 208 FP8 attention projection matrices.
Their logical element counts are:

| Set | Logical weights |
|---|---:|
| Gate+Up | 11,408,506,880 |
| Down | 5,704,253,440 |
| All MLP | 17,112,760,320 |
| All FP8 attention projections | 7,214,202,880 |
| All projections | 24,326,963,200 |

At P3847 the projection work, counting one multiply-add as two operations, is:

| Set | Operations | Current NSys budget | Current effective rate |
|---|---:|---:|---:|
| Gate+Up | 87.777 TFLOP | 3.44 s | 25.52 TOPS-equivalent |
| Down | 43.889 TFLOP | 1.64 s | 26.76 TOPS-equivalent |
| FP8 QKV/Z/O | 55.506 TFLOP | 2.51 s | 22.11 TOPS-equivalent |
| **Projection total** | **187.172 TFLOP** | **7.59 s** | **24.66 TOPS-equivalent** |

Full-attention/linear-attention compute contributes another approximately
0.151 s in the same profile. The current roughly 450 token/s means a 40K-token
agent cold start is roughly 89 seconds. The immediate 2,000 token/s target
reduces that projection-scale user wait to about 20 seconds; it is only the
first acceptable checkpoint, not the eventual ceiling.

## Hard arithmetic bound

For 16 SMs, four Tensor Cores per SM, and the 1.3005 GHz planning clock, the
dense integer arithmetic ceilings are:

| MMA mode | Dense peak | Ideal projection floor at P3847 |
|---|---:|---:|
| S8 x S8, `m16n8k32` | 85.230 TOPS | 2.196 s |
| S4 x S4, `m16n8k64` | 170.459 TOPS | 1.098 s |

The P3847 whole-Prefill target is 1.9235 s. Even ideal A8 projection time alone
is 2.196 s, so A8 cannot meet it. If the redesigned non-projection path is held
to 0.30-0.40 s, projection time must be at most 1.52-1.62 s. That requires a
blended 116-123 TOPS, or 68-72% of the dense S4 peak.

This is aggressive but arithmetically possible. It also explains why the A4
work cannot be postponed until after exhaustive A8 tuning.

### Why Gate-only A8 is not a route to the target

Give a hypothetical A8 kernel the entire Gate+Up category and assume it runs at
the impossible 100% dense S8 peak. Gate+Up would still take 1.030 s. Keeping the
current Down and FP8 categories gives:

```text
1.030 s + 1.640 s + 2.510 s = 5.180 s
3847 / 5.180 = 743 token/s before every non-projection cost
```

A literal Gate-only conversion is worse. The existing Gate A8W4 prototype is a
valuable instruction-path and real-API direction probe, but it cannot be
mistaken for the target architecture. Gate-only A8 leaves more than half of the
projection wall time untouched and cannot expose the layer-major producer /
consumer fusions needed by Down and attention projections.

## Reference audit and what is actually reusable

This plan uses mechanisms from the audited local checkouts; it does not import
their kernels or claim unsupported hardware coverage.

### vLLM Marlin

The local vLLM Marlin template contains the useful SM87 mechanics:

- S8 Tensor Core MMA (`mma.sync...m16n8k32...s8.s8.s32`);
- packed U4B8 weights decoded into S8 fragments;
- multistage `cp.async` movement;
- persistent stripe scheduling, locks, and reductions;
- row activation scales and grouped weight scales.

However, current vLLM explicitly rejects NVFP4 weights with INT8/FP8
activations (`marlin_utils_fp4.py`) and explicitly reports W8A8 Marlin as
unsupported (`marlin_utils_fp8.py`). Therefore the project can reuse the data
movement and scheduling ideas only after owning an offline re-encode and native
sidecar format. There is no vLLM production switch that solves this route.

### Triton

`python/tutorials/09-persistent-matmul.py` supplies two directly relevant
ideas: a fixed `NUM_SMS` persistent grid and grouped M-tile ordering for cache
locality. `10-block-scaled-matmul.py` is useful for scale packing and block
semantics. Its native block-scaled FP4 path targets newer instructions and is
not an SM87 implementation to transplant.

### FlashInfer / CUTLASS

FlashInfer's native FP4 GEMMs in the audited checkout target SM120, not SM87.
Its bundled CUTLASS examples are still valuable architectural references:

- example 45 dual GEMM shares A across two B matrices and carries two
  accumulator sets, which is the natural Gate+Up dataflow;
- example 47 Stream-K provides work-centric K decomposition and residual
  epilogues, which are relevant to Down's large-K/narrow-N geometry.

The result is a native SM87 design informed by these mechanisms, not a
FlashInfer backend integration.

## One layer-major execution plane

The quantized projection route and layer-major scheduling are one architectural
change. Wiring quantized kernels into the current C512 tile-major traversal
would repeatedly cross quantization boundaries, reload layer weights for every
chunk, and prevent prompt-wide scheduling. Conversely, an exact W4A16
layer-major route alone does not remove enough arithmetic and has already shown
that scheduling order by itself is not the leap.

The gated plane processes one full layer before advancing to the next:

```text
BF16 residual slab
  -> RMSNorm + quantize once
  -> QKV/Z (or Q/K/V) integer projections
  -> GDN or full attention; quantize O input
  -> O integer projection + residual partials
  -> normalize + quantize once
  -> paired Gate+Up integer projection
  -> fused SiLU(Gate) * Up + quantize Down input
  -> Down integer projection + residual partials
  -> finalize residual / statistics
  -> swap BF16 hidden slabs and advance layer
```

Two prompt-wide BF16 hidden slabs are 819.2 MB at 40K tokens. Prompt-wide
quantized staging sizes are:

| Slab | A8 at 40K | A4 at 40K |
|---|---:|---:|
| hidden width 5120 | 204.8 MB | 102.4 MB |
| MLP intermediate width 17408 | 696.32 MB | 348.16 MB |

Gate and Up must never be materialized as two prompt-wide BF16 outputs; that
would consume 2.785 GB at 40K. Their paired epilogue owns aligned 64- or
128-element output groups, computes `SiLU(gate) * up`, derives that group's
activation scale, and writes the packed Down input directly.

O and Down projection epilogues can fuse residual addition and publish partial
row sum-of-squares, but they cannot honestly claim a complete RMSNorm while
separate CTAs own different N tiles. A small finalize/norm/quant pass performs
the cross-N reduction and writes the next quantized slab. This avoids a false
fusion contract and provides a single place to compute trustworthy activation
scales.

When the arena cannot hold a whole prompt intermediate, the fallback is a
2K/4K-token **microspan within the same layer**, not a return to traversing all
64 layers per C512 tile. The persistent scheduler orders identical B tiles
across adjacent microspans so their short-term L2 residence remains useful.

## A8 full-projection plan

A8 validates the execution plane and establishes a materially faster, safer
intermediate baseline. Two sidecar choices should share one activation ABI.

### A8-safe: A8W8 everywhere

- Reconstruct each authenticated checkpoint matrix once at load/offline-convert
  time and encode symmetric signed W8 with BF16 scales per `[N, K128]` group.
- RMSNorm/finalize kernels dynamically encode activations as signed A8. The
  accuracy-first layout uses scales per `[M, K128]`; a per-row fast layout may
  advance only if the real capability gate remains unchanged.
- Accumulate four `m16n8k32.s8.s8.s32` operations per K128 group in INT32,
  convert once, multiply the A and B scales, and accumulate the group result in
  FP32.
- Use a three/four-stage `cp.async` pipeline and a fixed 32-CTA grid (two
  CTAs/SM), with resource queries enforcing the occupancy contract.

The raw A8-safe sidecar payload is 24,707,072,000 bytes (23.0103 GiB):
24,326,963,200 W8 bytes plus 380,108,800 BF16 K128 scale bytes. It replaces,
rather than coexists with, the exact Prefill sidecars in a production process.

### A8-compact: A8W4 MLP plus A8W8 attention

- Re-encode the NVFP4 MLP matrices as signed linear W4 in K32 groups, pack them
  in the S8/U4B8 consumer layout, and store BF16 group scales.
- Decode W4 fragments into S8 registers and execute the same S8 MMA path; the
  win relative to A8-safe is weight traffic and sidecar capacity, not a doubled
  arithmetic ceiling.
- Keep the former FP8 attention matrices as W8 K128 because the real-weight
  re-encode error is much smaller.

The existing experimental Gate implementation already demonstrates the
mechanical K32 recode, dynamic A8, four-stage pipeline, and M64N256K64 route. It
has not established a full-model accuracy or performance result and is not a
production path.

The compact payload is:

| Payload | Bytes | GiB |
|---|---:|---:|
| MLP W4 K32, BF16 scales, per-matrix factors | 9,625,928,448 | 8.9648 |
| Attention W8 K128 plus BF16 scales | 7,326,924,800 | 6.8237 |
| **Total** | **16,952,853,248** | **15.7886** |

This is only about 0.105 GiB above the raw exact NVFP4+FP8 sidecar payload.
Allocator alignment, manifests, and prompt scratch are deliberately excluded
from all payload totals.

### A8 family kernels

**Gate+Up.** Quantize the post-attention normalized A slab once. Interleave
corresponding Gate and Up N tiles in the sidecar. An M64 x (N128 Gate + N128 Up)
CTA loads A once, maintains two accumulator sets, and fuses SiLU/multiply plus
Down-input quantization. Four-stage and three-stage variants are resource-gated;
both must preserve two CTAs/SM.

**Down.** Do not reuse the Gate scheduler. Down has K=17408, N=5120, only 40
N128 tiles, and a much longer K loop. Use an M128N128 family with a persistent
32-CTA work queue. Ordinary data-parallel work is preferred at P2K/P4K; a
Stream-K or split-K variant is selected only for short-M/tail occupancy and must
include the reduction cost in real-API timing.

**Linear-attention projections.** Pack QKV and Z into one grouped/supermatrix
sidecar because both consume K=5120. One activation quantization and one
persistent launch produce consumer-native QKV and Z layouts. O consumes K=6144
and uses a separate family; the GDN producer should emit its quantized input
directly when ownership permits.

**Full-attention projections.** Group Q, K, and V, which share K=5120, behind
one quantization and persistent work queue. Keep their independent output
layouts. O again uses the K=6144 family. Attention/GDN producer-side packing is
preferred to writing BF16 and rereading it, but it advances only with an
unchanged real-API result.

### A8 performance envelope

The following is a planning envelope, not a benchmark result:

| Category | Sustained planning rate | P3847 time |
|---|---:|---:|
| Gate+Up | 55-65 TOPS | 1.35-1.60 s |
| Down | 48-58 TOPS | 0.76-0.91 s |
| FP8-source projections | 50-62 TOPS | 0.90-1.11 s |
| **Projection total** | — | **3.01-3.62 s** |

With 0.30-0.50 s of redesigned non-projection work, A8 is expected in the
roughly 930-1,165 token/s region. Its ideal mathematical whole-Prefill ceiling
is only about 1,540 token/s even with 0.30 s of other work. Do not spend months
tuning A8 toward a target it cannot reach.

## A4 full-projection plan

A4 uses signed linear A4 and W4 with
`mma.sync.aligned.m16n8k64...s4.s4.s32`. The first correct layout uses K64
scales; K128 is the speed/capacity variant after accuracy admission. B is stored
in `ldmatrix`-consumer order, A is nibble-packed by the producer, and each CTA
uses a three-stage `cp.async` pipeline while retaining two CTAs/SM.

The raw all-projection payloads are:

| W4 scale group | Payload | GiB |
|---|---:|---:|
| K64 | 12,923,699,200 bytes | 12.0361 |
| K128 | 12,543,590,400 bytes | 11.6821 |

### A4 family policy

**Gate+Up.** Retain the paired dual-accumulator topology. An aligned N64 or
N128 epilogue group computes SiLU/multiply, clips with the calibrated Down-input
policy, and writes packed A4 plus its BF16 scale without a full BF16
intermediate. Gate and Up may have different weight rounding/calibration even
though they share A.

**Down.** Retain a separate M128N128 persistent/Stream-K family. Its deep K
dimension benefits from longer work ownership and amortized scheduling, while
Gate's much larger N benefits more from A reuse. Runtime dispatch selects by
the N/K geometry; one M128N128K64 skeleton is not assumed optimal for both.

**FP8-source QKV/Z/O.** Offline-convert to calibrated W4. Linear QKV+Z and full
Q/K/V remain grouped by shared input, and O remains its own K6144 family. Q/K
are the highest semantic-risk matrices, so rollout may keep them A8 while Z/O
and MLP are A4, but the terminal 2K design and budget include all 208 attention
projections in A4.

**Epilogues.** Perform scale application in FP32, clamp before narrowing, and
write BF16 only at semantic boundaries needed by attention/state or the
canonical residual slab. Quantized internal edges are producer-owned; generic
repack kernels are a bring-up fallback, not the intended steady state.

### Offline calibration is mandatory

Native NVFP4 E2M1 values are nonlinear and cannot be fed exactly to integer S4
MMA. Naive symmetric recoding is a second quantization. A real-checkpoint CPU
direction audit (512 output rows, not a performance test) found:

| Source -> candidate | Group | Weight NRMSE | Mean cosine |
|---|---:|---:|---:|
| layer-0 NVFP4 Gate -> W4 | K32 | 0.10340 | 0.99509 |
| layer-0 NVFP4 Up -> W4 | K32 | 0.10327 | 0.99510 |
| layer-0 NVFP4 Down -> W4 | K32 | 0.10384 | 0.99506 |
| layer-0/3 FP8 attention -> W8 | K128 | 0.0066-0.0069 | about 0.999978 |
| layer-0/3 FP8 attention -> W4 | K64 | 0.109-0.114 | about 0.994 |
| layer-0/3 FP8 attention -> W4 | K128 | 0.120-0.126 | about 0.993 |

The layer-0 NVFP4 code audit also found that magnitudes 4 and 6 account for
about 18.7% of sampled values. They are not sparse enough to assume a cheap
high-code correction stream.

These numbers are only weight-recode lower bounds; activation quantization and
64-layer error propagation add risk. Consequently naive round-to-nearest W4 is
not eligible for production. The offline converter needs real-prompt
calibration and family/layer-specific policies:

- activation clipping search on ShareGPT, code, and agent-style long prompts;
- GPTQ/Hessian-aware or AWQ-aware weight rounding;
- SmoothQuant-style channel equalization only where the adjacent algebra permits
  the inverse scale to be folded exactly;
- independent policies for Gate, Up, Down, linear QKV/Z/O, and full Q/K/V/O;
- explicit K64 versus K128 decisions rather than one global group size.

### A4 performance envelope

| Category | Conservative first-version rate | P3847 time |
|---|---:|---:|
| Gate+Up | 95-120 TOPS | 0.73-0.92 s |
| Down | 80-105 TOPS | 0.42-0.55 s |
| FP8-source projections | 90-115 TOPS | 0.48-0.62 s |
| **Projection total** | — | **1.63-2.09 s** |

Including 0.30-0.45 s elsewhere gives an initial 1.93-2.54 s envelope, or
roughly 1,515-1,990 token/s. This means merely turning on S4 is insufficient.
The promoted design needs about 125 TOPS blended projection rate:

```text
187.172 TOP / 125 TOPS = 1.497 s projection
1.497 s + 0.350 s non-projection = 1.847 s whole Prefill
3847 / 1.847 = 2,083 token/s
```

The target therefore depends on the paired Gate+Up edge, direct producer
quantization, shape-specific Down scheduling, grouped attention projections,
and layer-major weight/A reuse. They are not optional follow-on polish.

## Accuracy and promotion gates

Experimental retention and terminal promotion are deliberately different:

- compare every experiment with the current self-owned production baseline;
  retain a real-API-positive change and update that baseline;
- compare the cumulative route with 2,000 token/s and the external reference
  only at promotion points;
- cuBLASLt remains a profiler/reference oracle and is never eligible for the
  production dispatch table.

### Stage 0: authenticated sidecar and memory gate

- authenticate the exact real checkpoint and inventory all 400 projection
  matrices;
- encode a versioned manifest containing source hashes, scale policy, layout,
  and per-layer offsets;
- preflight peak arena/sidecar memory; production loads exactly one Prefill
  sidecar family rather than retaining exact+A8+A4 copies;
- record CPU recode diagnostics, but do not use them as a performance decision.

### Stage 1: earliest real generation observation

As soon as a vertical slice traverses all 64 layers, run one same-ELF,
feature-off versus feature-on real-checkpoint request through the
OpenAI-compatible endpoint, initially P513 and `max_tokens=1`, on a frozen
ShareGPT/code/agent prompt. Use the EvalScope performance client as the external
driver as soon as the endpoint exposes the required fields; do not substitute a
kernel microbenchmark for this observation. Record one whole TTFT number. If it
is negative, stop building a formal timing harness and use NSys/NCU only to
explain and archive the failure. If it is positive, retain it as the next
experimental baseline.

### Stage 2: short performance matrix

Run the EvalScope performance CLI against the real API at P512, P1K, P2K, and
P4K, then expand to 8K/16K/40K only after the arena is safe. Use at least four
real prompts. B-C-C-B/mirrored timing is required only after the first positive
observation.

- A8 integration milestone: full projection route, no whole-API regression at
  short prompts, at least 1.5x whole-Prefill improvement at P4K.
- A4 terminal milestone: at least 2,000 token/s at both P2K and P4K with no
  short-prompt regression; 8K/16K/40K must confirm scaling before promotion.

### Stage 3: numerical and external capability gate

- zero non-finite values at every layer boundary;
- aggregate hidden-state NRMSE no more than 0.01 and cosine at least 0.9999 on
  the frozen set; any single-layer boundary no worse than 0.03 / 0.999;
- greedy first-26-token agreement remains a sentinel, not the sole quality
  claim;
- use EvalScope on the frozen public capability set only after the simple
  performance gate passes; no score may lose more than 1 percentage point
  absolute or more than the baseline confidence interval, whichever is tighter.

If A4 cannot pass this gate after calibration-aware rounding, a mixed A4/A8
route is allowed as an intermediate product, but its attainable throughput must
be measured honestly. The target is not waived.

### Stage 4: profile the positive whole route

- NSys kernel top-20 must prove that W4A16/W8A16 projection kernels are absent
  from the candidate Prefill span and that launch counts match the layer plan;
- NCU compares occupancy, achieved MMA issue, stage behavior, bank conflicts,
  L2 traffic, and stall reasons between the retained baseline and candidate;
- profile data selects the next structural closure only after the real API has
  shown value. Failed changes may still be profiled and archived for learning.

### Stage 5: production promotion

- frozen agent-style 8K/16K/40K cold starts, followed by short- and long-output
  requests;
- memory high-water, state continuity, first-token correctness, decode rate,
  and capability all pass;
- Decode remains on its existing exact path; no MTP is enabled;
- the feature flag and exact fallback remain until the promoted sidecar and
  route have passed all gates.

## Implementation order

1. Land the versioned quantized-sidecar manifest and prompt-slab arena contract.
   Add no production dispatch yet.
2. Build the gated layer-major state machine and A8-safe vertical slice across
   all Gate/Up/Down and QKV/Z/O projections. Commit in small reviewable pieces,
   but do not promote exact layer-major alone as a performance result.
3. Take the Stage-1 real API observation immediately; only then expand the A8
   timing/accuracy harness and profile the positive path.
4. Introduce calibrated A4 Gate+Up and Down together, because their producer /
   consumer edge removes the 17408-wide BF16 intermediate. Keep attention A8
   temporarily for fault isolation.
5. Convert Z/O, then linear QKV, then full K/V/Q to A4 according to measured
   semantic risk. Every step runs on the whole real generation path first.
6. Close the remaining gap to 125 blended TOPS with the profile-guided,
   family-specific schedulers; then run EvalScope and the long-context
   production gate.

The first implementation checkpoint is therefore not “a faster Gate kernel.”
It is “all 64 layers complete one real API request through a single A8
projection plane.” The performance checkpoint after that is full A4 projection
at or above 2,000 token/s.

## Audited source map

Project:

- `src/runtime/model_weights.cpp`
- `include/q3x/runtime/model_weights.h`
- `src/kernels/sm87/nvfp4_marlin_w4a16.cu`
- `src/kernels/sm87/fp8_prefill_supermatrix.cu`
- `src/kernels/sm87/projection_route_registry.h`

Local reference checkouts:

- `/home/rm01/setup/build4all/vllm/csrc/libtorch_stable/quantization/marlin/`
- `/home/rm01/setup/build4all/vllm/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py`
- `/home/rm01/setup/build4all/vllm/vllm/model_executor/layers/quantization/utils/marlin_utils_fp8.py`
- `/home/rm01/setup/build4all/triton/python/tutorials/09-persistent-matmul.py`
- `/home/rm01/setup/build4all/triton/python/tutorials/10-block-scaled-matmul.py`
- `/home/rm01/setup/build4all/flashinfer/3rdparty/cutlass/examples/45_dual_gemm/`
- `/home/rm01/setup/build4all/flashinfer/3rdparty/cutlass/examples/47_ampere_gemm_universal_streamk/`

Experimental A8 direction probe (not production):

- branch `codex/a8w4-gate-admission`
- commits `59ed87d`, `0d613ed`, `e5c73bb`

This design task did not run the GPU and does not claim new performance data.
All time envelopes above are explicit arithmetic/planning bounds derived from
the recorded real P3847 kernel budget; only the staged real-API gates can turn
them into project results.
