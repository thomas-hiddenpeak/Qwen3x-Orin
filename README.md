# Qwen3x-Orin

Qwen3x-Orin is an experimental, pure C++/CUDA inference engine for selected
Qwen3.5 and Qwen3.6 models on NVIDIA Jetson AGX Orin (`sm_87`). It is designed
around Orin's unified memory and Ampere Tensor Cores instead of treating Orin
as a reduced version of a Blackwell system.

The project now contains a correctness-first native generation path for the
exact pinned NVIDIA Qwen3.6-27B-NVFP4 checkpoint. It authenticates and loads
the text weights into one resident CUDA arena, binds the fixed hybrid decoder,
creates one bounded batch-one request arena, formats/tokenizes one user prompt,
and performs sequential greedy generation. This reference path is intended for
oracle alignment and bring-up. On the target Orin it now reproduces the pinned
fixture's 19 prompt IDs, all 26 greedy output IDs, decoded text, and stop token
exactly. Broader prompt coverage, cross-backend boundary analysis, and
aggressive performance tuning remain in progress. An explicit, default-off
SM87 weight-only projection backend now passes the same full-model gate and
reduces median subsequent-token latency from 1,144.108 ms to 651.554 ms in the
first matched diagnostic benchmark.

> Qwen3x-Orin is an independent community project. It is not an official Qwen,
> Alibaba, NVIDIA, or Jetson project and is not endorsed by those organizations.

## Target support

| Model family | Topology | ModelOpt weights | Initial status |
| --- | --- | --- | --- |
| Qwen3.6 27B pinned NVIDIA revision | Dense | FP8 W8A16 + NVFP4 W4A16 + BF16 fallback | Native reference generation plus opt-in SM87 M=1 projections; both pass the fixed oracle gate on Orin |
| Qwen3.5 / Qwen3.6 35B-A3B | MoE | FP8 W8A16 + NVFP4 W4A16 + BF16 fallback | Planned, after the dense path |

The initial scope is text-only, batch-one correctness and decode performance.
Paged KV cache, continuous batching, an OpenAI-compatible server, MTP, and
vision support are staged follow-up work. See [the roadmap](docs/ROADMAP.md).

## Why a new engine?

The target checkpoints use ModelOpt mixed precision. Orin cannot execute
Blackwell-native NVFP4 tensor-core instructions, but it can use a Marlin-style
W4A16 path: unpack E2M1 FP4 weights and apply their scales in registers, then
feed BF16 values to Ampere tensor-core operations. The same runtime also needs
an FP8 W8A16 path and BF16 kernels because the checkpoints are not uniformly
FP4.

Qwen3x-Orin therefore uses multiple shape-dependent backends rather than one
universal kernel:

- an `sm_87` weight-only GEMV path for single-token decode;
- Marlin-style W4A16/W8A16 kernels for small token batches;
- cuBLASLt or tuned CUDA kernels for larger prefill operations;
- routed, grouped expert execution for the 35B-A3B MoE model.

Official safetensors remain the source of truth. A future offline packer will
create an optional `.q3x` cache containing losslessly reordered, fused, and
prepacked weights so startup does not need a second full model-sized copy.

## Repository layout

```text
include/q3x/       Public runtime interfaces
src/core/          Tensor, device, allocation, and runtime primitives
src/text/          Pinned tokenizer and text/chat preprocessing
src/model/         Qwen configuration, weight metadata, and execution graph
src/runtime/       Authenticated resident weights and runtime ownership
src/kernels/       Reference and architecture-specific CUDA kernels
tools/             Model inspection and offline packing tools
tests/             Unit, numerical, and integration tests
benchmarks/        Representative kernel and end-to-end benchmarks
docs/              Design and roadmap
third_party/       Vendored dependencies and their notices, when introduced
```

The detailed architecture and format rules live in
[docs/DESIGN.md](docs/DESIGN.md). The pinned upstream configurations and exact
engine-visible dimensions are recorded in
[docs/MODEL_SUPPORT.md](docs/MODEL_SUPPORT.md). The layer equations, tensor
layouts, state sizes, and decode order for the first model are specified in the
[Qwen3.6 27B runtime contract](docs/QWEN36_27B_RUNTIME_CONTRACT.md).
The exact resident-memory budget, I/O trust boundary, and full-model Orin run
are recorded in the [resident-weight loader report](docs/RESIDENT_WEIGHT_LOADER.md).
The per-request state layouts, workspace capacities, reset semantics, and
128/262144-token budgets are specified in
[the request-state contract](docs/REQUEST_STATE.md).
The strict non-owning 64-layer view graph, scale-copy boundary, and
allocation-free reference projection dispatch are documented in the
[resident model-weight binding contract](docs/MODEL_WEIGHT_BINDING.md).
The generation lifecycle, token/stop semantics, timing fields, trace hashes,
and CLI output contract are documented in the
[native reference engine contract](docs/REFERENCE_ENGINE.md).
Reusable single-load latency/replay measurement is specified in the
[benchmark harness](docs/REFERENCE_BENCHMARK.md), and the first Nsight plus
matched backend results are recorded in the
[Phase 3 performance evidence](docs/PERFORMANCE_BASELINE.md).

## Build and inspect

The current tree requires a CUDA-capable Jetson AGX Orin development
environment with CMake, a C++17 compiler, and ICU 74 (`uc` and `i18n`). ICU is
linked only by the separate `q3x::text` tokenizer library.

```bash
cmake -S . -B build
cmake --build build -j
ctest --test-dir build --output-on-failure
```

The equivalent checked-in Orin release preset is:

```bash
cmake --preset orin-release
cmake --build --preset orin-release -j
ctest --preset orin-release
```

Probe the target and list the catalogued model descriptors with:

```bash
build/orin-release/qwen3x-orin probe
build/orin-release/qwen3x-orin models
```

Run correctness-first batch-one greedy generation with:

```bash
build/orin-release/qwen3x-orin generate MODEL_DIR \
  --prompt "Explain unified memory in one sentence." \
  --max-tokens 32
```

The correctness reference remains the default. On an SM87 device, select the
validated direct FP8/NVFP4-to-BF16 layer path explicitly with
`--projection-backend sm87`. Reuse one loaded engine for repeatability and
latency distributions with:

```bash
build/orin-release/qwen3x-orin benchmark MODEL_DIR \
  --prompt "Explain unified memory in one sentence." \
  --max-tokens 2 --warmup 1 --iterations 3 \
  --max-sequence-length 64 --projection-backend sm87
```

Add `--trace` to emit embedding, every layer hidden/residual, final-norm, and
whole-step SHA-256 digests. Successful machine-readable `key=value` results go
to stdout; loading progress and structured errors go to stderr. `MODEL_DIR`
must contain the exact pinned `tokenizer.json` and all three official shards.

The target-device fixed-oracle run used the Chinese prompt recorded in the
fixture with `--max-tokens 26 --trace`. It produced the exact expected 26 IDs
and text, stopped at `<|im_end|>`, and completed 44 sequential runner steps.
Its non-normative cold load was 213.845 seconds and generation was 49.212
seconds; see the [reference engine evidence](docs/REFERENCE_ENGINE.md) for the
complete timing and boundary-hash record.
The same run is available as
[machine-readable metadata](docs/metadata/qwen36-27b-native-reference-run.json).

Inspect a local checkpoint without loading weight payloads:

```bash
build/orin-release/qwen3x-inspect index \
  MODEL_DIR/model.safetensors.index.json
build/orin-release/qwen3x-inspect header \
  MODEL_DIR/model-00001-of-00003.safetensors
build/orin-release/qwen3x-inspect checkpoint MODEL_DIR
build/orin-release/qwen3x-inspect checkpoint MODEL_DIR --require-shards
build/orin-release/qwen3x-inspect manifest MODEL_DIR
build/orin-release/qwen3x-inspect load-plan MODEL_DIR
```

`checkpoint` accepts only explicitly pinned Qwen3.6 ModelOpt revisions. It
validates the raw metadata hashes and their semantic model, quantization, and
index contracts; a shape-compatible unknown revision still fails closed.
Without `--require-shards`, missing weight shards are warnings so downloaded
metadata can be audited independently. See
[the checked-in evidence](docs/metadata/README.md) and the
[Orin Phase 0 record](docs/PHASE0_EVIDENCE.md).

`load-plan` performs the strict manifest and pinned-layout checks but does not
allocate GPU memory or read full payloads. The production resident loader is a
separate API and authenticates every full shard while populating only text
tensors in one 256-byte-aligned CUDA arena.

The pure C++ tokenizer API loads only the pinned raw `tokenizer.json`, applies
ICU NFC/Unicode pre-tokenization plus GPT-2 ByteLevel BPE, and exposes a
fail-closed text-only Qwen chat formatter. Its exact schema, resource limits,
and supported chat subset are documented in
[the tokenizer contract](docs/TOKENIZER.md).

Native inference is currently a deliberately sequential, batch-one surface
with a correctness reference and an opt-in M=1 projection optimization. It
does not yet provide optimized multi-token prefill, continuous batching, a
server, or a release-grade performance claim. The independent target-device oracle,
including exact prompt/output token IDs and chosen-token log probabilities, is
checked in as
[tests/fixtures/qwen36-27b-nvfp4-greedy.json](tests/fixtures/qwen36-27b-nvfp4-greedy.json);
the native BF16 cache/state policy is pinned separately in
[tests/fixtures/qwen36-27b-nvfp4-greedy-bf16.json](tests/fixtures/qwen36-27b-nvfp4-greedy-bf16.json)
and its
[64-layer boundary fixture](tests/fixtures/qwen36-27b-nvfp4-layers-bf16.json).
Both policies produce the same 26 greedy output token IDs and decoded text.
The native reference CLI now reproduces those IDs and text exactly as well.
The bounded C++ schema loader, cross-file checks, and exact-hash versus
cross-backend sample semantics are specified in the
[BF16 reference oracle contract](docs/REFERENCE_ORACLE.md).
The vLLM reproduction utility is
[tools/reference/qwen36_27b_vllm_greedy.py](tools/reference/qwen36_27b_vllm_greedy.py).

## Engineering rules

- Correctness precedes optimization. Every optimized kernel must have a simple
  reference implementation and a numerical comparison test.
- Quantization metadata is parsed from the checkpoint; shape or scale policy
  must not be inferred from a model name.
- Repacking is lossless. Qwen3x-Orin will not silently requantize official
  checkpoint weights into another 4-bit format.
- Unsupported model revisions and tensor layouts fail with actionable errors.
- Performance claims must include the model revision, context length, batch,
  power mode, clocks, JetPack/CUDA version, and measured memory use.

## Sources and licensing

Qwen3x-Orin is original work licensed under the
[Apache License 2.0](LICENSE). Its design is informed by, and implementation
may adapt properly attributed code from, projects including:

- [qwen35-thor](https://github.com/thomas-hiddenpeak/qwen35-thor), MIT licensed,
  as a reference for Qwen hybrid-model execution on Jetson-class hardware;
- [vLLM](https://github.com/vllm-project/vllm), Apache-2.0 licensed, including
  its ModelOpt and Marlin integration patterns;
- [Marlin](https://github.com/IST-DASLab/marlin), subject to its upstream
  license, as a reference for weight-only Ampere kernels.

Any source copied or adapted from another project must keep the applicable
copyright and license header. Required attributions will be recorded in
[NOTICE](NOTICE) and beside vendored code. Design inspiration alone does not
mean that upstream code is present in the repository.

Model weights, tokenizers, configuration files, and generated artifacts are
distributed separately and are **not** covered by this repository's Apache-2.0
license. Users must review and comply with each model publisher's license and
acceptable-use terms.

## Contributing

Contributions are welcome once the first interfaces settle. Until then, small,
testable changes are preferred: checkpoint metadata inspection, reference
dequantization, representative microbenchmarks, and numerical fixtures. By
submitting a contribution, you agree that it is provided under Apache-2.0 as
described in section 5 of the license.
