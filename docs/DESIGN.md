# Qwen3x-Orin design

## 1. Purpose and current status

Qwen3x-Orin is a purpose-built inference engine for NVIDIA Jetson AGX Orin
(`sm_87`) running selected Qwen3.5 and Qwen3.6 text models. It uses pure
C++/CUDA in the inference process and targets the ModelOpt mixed-precision
27B dense and 35B-A3B MoE checkpoints.

This document describes the intended architecture. At project bootstrap, most
components below are contracts and milestones rather than completed features.
Support is claimed only after the relevant correctness and end-to-end tests are
checked into the repository.

## 2. Goals and non-goals

### Goals

- Run the selected models on a 64 GB Jetson AGX Orin without converting their
  weights to a different lossy quantization scheme.
- Support checkpoint-directed ModelOpt mixed precision: NVFP4 W4A16, FP8
  W8A16, and BF16 fallback in one model graph.
- Make single-user decode efficient while retaining a path to continuous
  batching and useful prefill performance.
- Keep memory peaks predictable. Model loading and packing must not require two
  persistent model-sized copies.
- Validate each optimized path against a simple CPU or CUDA reference and
  against trusted framework output.
- Make benchmark results reproducible on Jetson, including power and clock
  configuration.

### Non-goals for the first release

- General support for every Qwen size, revision, or quantization format.
- NVIDIA Blackwell, discrete Ampere GPUs, x86 hosts, or non-CUDA devices.
- Training, fine-tuning, or quantization of model weights.
- Tensor parallelism or multi-node inference.
- Vision inputs, MTP, speculative decoding, and maximum-context serving before
  text-only autoregressive inference is correct.
- Binary compatibility with qwen35-thor, vLLM, TensorRT-LLM, or llama.cpp.

## 3. Supported model contract

The loader identifies compatibility from checkpoint configuration, tensor
metadata, and quantization metadata. A directory or repository name is never
sufficient evidence of compatibility.

The initial support matrix is:

| Family | Parameters | Topology | Execution priority |
| --- | ---: | --- | --- |
| Qwen3.5 / Qwen3.6 | 27B | Dense hybrid text model | First |
| Qwen3.5 / Qwen3.6 | 35B-A3B | Routed MoE hybrid text model | Second |

Within a recognized revision, the loader validates at least:

- architecture and text-configuration identifiers;
- hidden, intermediate, head, layer, expert, and top-k dimensions;
- tensor names, shapes, dtypes, and shard ranges;
- per-module quantization method, group size, scale dtype, and global scale;
- tokenizer and special-token configuration needed by the runtime.

Unknown values fail closed with a diagnostic naming the unsupported field.
Model-revision-specific behavior belongs in an explicit model descriptor, not
in scattered kernel conditionals.

## 4. Quantization contract

The checkpoints are mixed precision. Dispatch is selected per linear module
from quantization metadata.

### 4.1 NVFP4 W4A16

The baseline NVFP4 representation packs two E2M1 values per byte. ModelOpt
metadata supplies group scales (normally E4M3 with group size 16) and a global
FP32 scale. Exact interpretation is validated from the checkpoint rather than
assumed globally.

The correctness reference consumes the checkpoint's canonical row-major
logical views, `[N, K/2]` packed weights and `[N, K/16]` block scales. Those
shapes are not a Marlin ABI. A Marlin-oriented cache may transpose and permute
both arrays and may convert scales to a backend-specific representation; its
layout version must describe that transformation explicitly.

`weight_scale_2` belongs to one original quantized tensor. Fused projections,
MoE experts, and checkpoint shards must retain the separate value associated
with every source tensor rather than applying one scalar to a combined buffer.

Orin has no native NVFP4 tensor-core instruction. The `sm_87` path therefore:

1. loads prepacked E2M1 nibbles and the corresponding scales;
2. decodes and scales values in registers;
3. presents BF16 fragments to Ampere tensor-core MMA or CUDA arithmetic;
4. accumulates according to the kernel's documented FP32/BF16 policy.

Repacking may permute bits, rows, columns, groups, and scale storage, but it
must be reversible and must not select a new codebook or fit new scales.

The current canonical M=1 path does not repack. When K is divisible by 256 and
the packed row is 4-byte aligned, each SM87 lane loads four adjacent packed
bytes, decodes eight E2M1 values, and shares the group scale with its adjacent
lane. Other shapes use the scalar decoder. This is a measured CUDA-arithmetic
path, not a claim that Orin has native NVFP4 tensor-core support. The bounded
M2..M8 path consumes the same canonical layout and reuses each decoded weight
across the tile's BF16 activation rows; unsupported alignment or K falls back
to checked M1 launches. The fixed-M16 path recognizes the production
`[17408,5120]` and `[5120,17408]` projections, expands their E2M1 values and
E4M3FN block scales into BF16 shared-memory tiles, and accumulates with Ampere
BF16 Tensor Core MMA. It requires 16-byte-aligned packed weights,
2-byte-aligned block scales, and 8-byte-aligned activations; other valid shapes
or alignments fall back to two ordered M8 launches. Neither route introduces a
checkpoint repack or claims native NVFP4 arithmetic.

The C512 large-M projection families are designed and admitted separately.
Their operand-residency model, Gate/Up-versus-Down specialization boundary,
FP8 scope, and current bridge-beating gates are recorded in
[LARGE_M_PROJECTION_DATAFLOW.md](LARGE_M_PROJECTION_DATAFLOW.md).

The NVIDIA artifact also stores an FP32 `input_scale` beside quantized linear
modules. That value belongs to an activation-quantized W4A4 calibration path.
Qwen3x-Orin's initial W4A16 path keeps BF16 activations, so it validates and
records `input_scale` for provenance but does not multiply activations by it.
Treating it as a weight scale would change the model.

### 4.2 FP8 W8A16

FP8 weight modules consume BF16 activations and apply the checkpoint's E4M3
weights and scale policy. The initial implementation may use a reference
dequantization path. The optimized path uses Marlin-style register conversion
or a shape-appropriate custom kernel. Per-tensor, block, and secondary/global
scale policies must remain distinct in metadata and dispatch.

The current canonical M=1 path does not repack. When K is divisible by 1,024,
the row-major weight pointer is 4-byte aligned, and the BF16 activation is
8-byte aligned, each SM87 lane loads four E4M3FN bytes and four BF16 values
with one 32-bit and one 64-bit global load. Four FP32 accumulators amortize the
loop dependency, and a branchless conversion preserves all finite E4M3FN
values, signed zero, and the two reserved encodings as signed canonical quiet
NaNs. Other K values or either unaligned pointer use the scalar decoder. This
is the measured M=1 CUDA-arithmetic route. The bounded M2..M8 route likewise
uses the canonical row-major checkpoint bytes, reusing one decoded FP8 weight
across the tile activation rows and falling back to M1 for unsupported
alignment/K. The fixed-M16 path uses the same checkpoint bytes directly: the
row-major `W[N,K]` storage is the column-major `B[K,N]` operand consumed after
E4M3FN-to-BF16 expansion, and Ampere BF16 Tensor Core MMA accumulates FP32.
It is enabled for `[10240,5120]`, `[5120,6144]`, `[6144,5120]`, and
`[12288,5120]` with 16-byte-aligned weights and 8-byte-aligned activations.
The measured `[1024,5120]` case and every other valid shape/alignment retain
two ordered M8 launches. A repacked or larger dense-prefill route remains a
separate layout and dispatch milestone; the completed canonical C16 route is
still bounded prefill.

Likewise, the pinned artifact's FP32 `input_scale` is not applied by the
W8A16 path: the stored E4M3 weight is multiplied by its `weight_scale`, while
the activation remains BF16. Any future W8A8 path must have a separate
interface and numerical contract.

### 4.3 BF16

Unquantized embeddings, normalization parameters, selected projections, and
other fallback tensors remain BF16 (or their declared metadata dtype). Dense
BF16 matrix products use cuBLASLt initially unless a measured custom path is
both faster and numerically acceptable.

### 4.4 Pinned NVFP4 evidence

The reference contract was audited on 2026-07-18 against NVIDIA ModelOpt
commit `9392dfeabfde8695e9f58421c551ea8004fc3a1a` and vLLM commit
`b5433b6f5079feb32f9f278cf4ae23bd87375148`. In particular:

- ModelOpt defines the [E2M1 values and tensor-level scale](https://github.com/NVIDIA/Model-Optimizer/blob/9392dfeabfde8695e9f58421c551ea8004fc3a1a/modelopt/torch/quantization/qtensor/nvfp4_tensor.py#L25-L27)
  and computes `weight_scale_2` as `amax / (6 * 448)` in its
  [NVFP4 quantizer](https://github.com/NVIDIA/Model-Optimizer/blob/9392dfeabfde8695e9f58421c551ea8004fc3a1a/modelopt/torch/quantization/qtensor/nvfp4_tensor.py#L88-L108).
- ModelOpt's [pack/dequantize implementation](https://github.com/NVIDIA/Model-Optimizer/blob/9392dfeabfde8695e9f58421c551ea8004fc3a1a/modelopt/torch/quantization/qtensor/nvfp4_tensor.py#L320-L407)
  confirms low-then-high nibble order and multiplication by, rather than the
  reciprocal of, `weight_scale_2`.
- vLLM records the [canonical dense shapes and scale direction](https://github.com/vllm-project/vllm/blob/b5433b6f5079feb32f9f278cf4ae23bd87375148/vllm/model_executor/layers/quantization/modelopt.py#L1312-L1384),
  while its [MoE loader](https://github.com/vllm-project/vllm/blob/b5433b6f5079feb32f9f278cf4ae23bd87375148/vllm/model_executor/layers/quantization/modelopt.py#L1455-L1532)
  demonstrates why fused experts need separate tensor-level scales.
- vLLM's [Marlin FP4 preprocessing](https://github.com/vllm-project/vllm/blob/b5433b6f5079feb32f9f278cf4ae23bd87375148/vllm/model_executor/layers/quantization/utils/marlin_utils_fp4.py#L238-L299)
  is evidence that backend-prepacked weight and scale layouts require their own
  versioned contract.
- vLLM commit `ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb` makes the Ampere
  weight-only boundary explicit: its
  [NVFP4 W4A16 adapter](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/layers/quantization/modelopt.py#L1243-L1387)
  discards the stored activation scale, and its
  [FP8 Marlin adapter](https://github.com/vllm-project/vllm/blob/ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb/vllm/model_executor/kernels/linear/scaled_mm/marlin.py#L29-L95)
  likewise runs weight-only with BF16/FP16 input.

## 5. System architecture

```text
official checkpoint + quantization metadata
                    |
            inspect / validate
                    |
          +---------+----------+
          |                    |
     direct loader       qwen3x-pack (offline)
          |                    |
          |          lossless .q3x prepacked cache
          +---------+----------+
                    |
           model execution graph
                    |
       +------------+-------------+
       |            |             |
    common CUDA   sm_87 GEMV    Marlin-style
       |            |          W4A16 / W8A16
       +------------+-------------+
                    |
          allocator / streams / cache
                    |
             tokens or server API
```

### 5.1 Core runtime

The core runtime owns:

- typed tensor views with explicit device, shape, stride, and lifetime;
- checked CUDA device/stream/event wrappers;
- workspace planning and allocation;
- error propagation with source context;
- profiling ranges and benchmark metadata.

Jetson's CPU and GPU share physical memory, but host, pinned, mapped, managed,
and device allocations have different access and synchronization behavior.
They remain explicit allocation kinds. The engine will choose among them from
measurements; it will not treat "unified memory" as permission to omit
lifetime or residency management.

The pinned 27B direct path implements this policy as one checked `cudaMalloc`
arena, two bounded page-locked staging buffers, and an 8 GiB default post-load
free-memory gate. Its exact byte budget and one-pass authentication contract are
recorded in [RESIDENT_WEIGHT_LOADER.md](RESIDENT_WEIGHT_LOADER.md).

After authentication, the pinned dense model is converted once into a fixed
non-owning 64-layer typed graph. Exact view checks, owner lifetime, ModelOpt
scale handling, and allocation-free projection dispatch are specified in
[MODEL_WEIGHT_BINDING.md](MODEL_WEIGHT_BINDING.md).

### 5.2 Model layer

The model layer converts validated checkpoint metadata into an immutable
execution plan. It describes hybrid sequence layers, attention, recurrent or
DeltaNet state, dense MLPs, routed experts, shared experts, normalization,
embeddings, and output projection without embedding a CUDA launch policy in
the model parser.

Weights reference an abstract storage object so direct safetensors loading and
prepacked `.q3x` loading produce the same logical model graph.

### 5.3 Kernel registry and dispatch

Kernel selection uses an explicit key including:

```text
operation, SM, weight format, activation dtype, output dtype,
M/N/K, group size, layout, batch/routing shape, workspace limit
```

The registry never silently falls back across incompatible number formats. A
slow but correct supported fallback is allowed and is reported in profiling;
an unsupported layout is an error.

The expected linear dispatch policy is initially:

| Work shape | Preferred path | Alternative |
| --- | --- | --- |
| Decode, `M = 1` | `sm_87` FP4/FP8 weight-only GEMV | Reference kernel |
| Bounded tile, `M = 2..8` | Canonical-layout weight-reuse CUDA kernels | GEMV or reference |
| Bounded tile, `M = 9..15` | Ordered M8 plus M1..M7 SM87 launches | M scalar reference launches |
| Exact production tile, `M = 16` | Shape-gated FP8/NVFP4 decode to BF16 plus Ampere Tensor Core MMA | Two ordered M8 launches or scalar reference |
| Larger dense prefill | Measured Marlin or cuBLASLt-based path | Reference tiles |
| Routed MoE | Sorted/grouped expert kernels | Per-expert correctness path |

Thresholds are device-and-shape benchmark data, not constants justified only
by intuition. A checked-in tuning table may provide defaults; users can
regenerate it for their JetPack and power mode.

### 5.4 MoE execution

The 35B-A3B path must avoid launching gate, up, and down kernels separately for
every selected expert and token. Its optimized flow is:

1. compute router logits and top-k assignments;
2. build stable token/expert assignments and offsets;
3. group work by expert without losing routing weights;
4. execute fused or grouped gate/up projections and activation;
5. execute grouped down projection;
6. scatter and reduce weighted outputs deterministically enough for the
   documented numerical tolerance;
7. combine the shared-expert path where the model revision requires it.

A simple per-expert implementation remains available for correctness tests.
Routing, capacity, normalization, and shared-expert semantics are model
behavior and must be verified separately from GEMM accuracy.

### 5.5 Sequence state and KV cache

The first end-to-end runner uses the simplest bounded BF16 cache/state layout
that can establish correctness. Paged allocation, FP8 KV storage, prefix
reuse, and continuous batching are later optimizations.

Cache sizing is computed from validated configuration and requested context.
Startup reports persistent weights, KV/state reservation, workspaces, and
remaining headroom. Requests that cannot fit fail before partial execution
where possible.

## 6. `.q3x` prepacked cache

The optional cache exists to reduce startup time, allocator pressure, and peak
memory. It is not a new quantization format.

The container must include:

- a magic value and independently versioned container/layout versions;
- source checkpoint identity and hashes for every consumed shard;
- normalized model and quantization metadata;
- tensor records with logical and physical layouts, offsets, lengths,
  alignments, dtypes, and checksums;
- the target architecture (`sm_87`) and packer version;
- fused/repacked tensors plus unchanged tensors needed by the model;
- an atomic-completion marker so interrupted packing is never accepted.

The packer writes a temporary output and publishes it atomically after full
validation. It streams tensors or layers and releases transient data promptly;
holding both entire source and destination models in resident memory is a bug.

A cache is rejected when its source hashes, layout version, architecture, or
relevant runtime ABI is incompatible. Rebuilding is always preferable to
guessing compatibility.

## 7. Correctness and testing

Validation proceeds from small formats to complete generation:

1. Exhaustive E2M1 decode tests and edge-case E4M3/scale tests.
2. Repack/unpack round trips with byte- or value-exact expectations.
3. W4A16, W8A16, and BF16 kernel comparisons on representative and awkward
   dimensions.
4. Layer tests for normalization, positional encoding, hybrid sequence state,
   attention, dense MLP, routing, experts, and logits.
5. Per-layer activation comparisons against a trusted reference runtime using
   fixed prompts and checkpoint revisions.
6. End-to-end greedy token comparisons, followed by tolerance-based logits and
   sampling tests.
7. Long-running memory, determinism, malformed-checkpoint, and out-of-memory
   tests.

Tolerances are defined per operation and accumulation policy. A single loose
global tolerance is not acceptable. Test fixtures record the generating model
revision and script so they can be audited.

## 8. Performance measurement

Microbenchmarks cover at least one representative dense projection, one small
MoE expert projection, and one FP8 projection before whole-model optimization.
They report:

- cold and warm latency distributions;
- effective weight bandwidth and achieved operation rate;
- tokens or rows per launch, dimensions, and selected backend;
- workspace and peak allocated memory;
- Jetson model, JetPack/CUDA versions, power mode, clocks, temperature, and
  throttling state.

End-to-end reports separate model load, first-token/prefill latency, decode
tokens per second, and memory at specified prompt/output lengths. Results from
different power modes or throttling states are not compared as if equivalent.

## 9. Failure and compatibility policy

- CUDA errors are checked at their ownership boundary and include operation
  context.
- File offsets, lengths, tensor products, and allocation sizes use checked
  arithmetic.
- Loader input is untrusted: truncated shards, duplicate tensors, invalid JSON,
  unsupported dtypes, and inconsistent metadata receive explicit errors.
- Container and kernel-layout versions are independent so either can evolve
  without pretending ABI compatibility.
- Public compatibility begins only when a versioned release declares it; prior
  bootstrap interfaces may change freely.

## 10. Provenance and license boundaries

Qwen3x-Orin is Apache-2.0 licensed original work. qwen35-thor is a useful MIT
licensed architectural reference; vLLM's ModelOpt/Marlin work is an
Apache-2.0 reference; upstream Marlin has its own license and notices. Merely
reading or benchmarking another project does not make its code part of this
repository. If source is copied or adapted, the change must:

- be compatible with Apache-2.0 distribution;
- retain the upstream copyright and license header;
- identify the source revision in a nearby record or commit;
- update `NOTICE` when the upstream license or notice requires it;
- keep vendored license texts with vendored material.

Model artifacts are separate works governed by their publishers' terms. The
runtime must not bundle or download weights by default in a way that obscures
those terms. Names such as Qwen, NVIDIA, ModelOpt, CUDA, and Jetson are used
only to identify compatibility or provenance; this project makes no claim of
affiliation or endorsement.
