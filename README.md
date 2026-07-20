# Qwen3x-Orin

Qwen3x-Orin is an experimental, pure C++/CUDA inference engine for selected
Qwen3.5 and Qwen3.6 models on NVIDIA Jetson AGX Orin (`sm_87`). It is designed
around Orin's unified memory and Ampere Tensor Cores instead of treating Orin
as a reduced version of a Blackwell system.

The project now contains a correctness-first native generation path for the
exact pinned NVIDIA Qwen3.6-27B-NVFP4 checkpoint. It authenticates and loads
the text weights into one resident CUDA arena, binds the fixed hybrid decoder,
creates one bounded batch-one request arena, formats/tokenizes one user prompt,
and performs bounded greedy generation. This reference path is intended for
oracle alignment and bring-up. On the target Orin it now reproduces the pinned
fixture's 19 prompt IDs, all 26 greedy output IDs, decoded text, and stop token
exactly. Broader prompt coverage, cross-backend boundary analysis, and
aggressive performance tuning remain in progress. An explicit, default-off
SM87 weight-only projection backend now passes the same full-model gate and
supports bounded `C=1..16` prompt-prefix tiles. In the historical two-prompt,
two-output-token diagnostic, the first `C=8` path reduced median TTFT from
6,107.420 ms to 2,005.784 ms. The kernel-optimization chain through `5fe0ae0`
reaches 1,020.755 ms TTFT, 1,205.989 ms total generation, and 185.108 ms for
the subsequent token without increasing the 85,011,968-byte C8 request arena.
The subsequent C16 runtime/GDN and FP8/NVFP4 Tensor Core sequence through
`33948e3` preserves the same fixed oracle. In a mirrored same-binary comparison,
C16 reduced median TTFT from 1,021.088 to 761.037 ms and total
two-token generation from 1,206.170 to 946.217 ms; the subsequent-token median
remained effectively flat at 185.084 versus 185.186 ms. Each policy contributes
eight measured samples from the mirrored C8/C16/C16/C8 process order.
The defaults remain `C=1` and the `reference` backend, and the result is not a
serving-throughput claim. Comparisons with the earlier 1,144.108 ms reference
decode are historical rather than randomized same-binary trials. The
default authenticated loader now uses Linux AF_ALG when available, reducing
the measured resident-load phase for the same pinned model from about 203.7
seconds to 21.5 seconds without weakening the three full-file SHA-256 checks.

> Qwen3x-Orin is an independent community project. It is not an official Qwen,
> Alibaba, NVIDIA, or Jetson project and is not endorsed by those organizations.

## Target support

| Model family | Topology | ModelOpt weights | Initial status |
| --- | --- | --- | --- |
| Qwen3.6 27B pinned NVIDIA revision | Dense | FP8 W8A16 + NVFP4 W4A16 + BF16 fallback | Native reference generation plus opt-in SM87 M=1 decode and `C<=16` prompt-prefix projections; C1, C8, and C16 runs pass the fixed oracle gate on Orin |
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
linked only by the separate `q3x::text` tokenizer library. Linux AF_ALG is an
optional startup acceleration supplied by the kernel; no OpenSSL development
dependency is required, and the loader falls back to its portable SHA-256
implementation if AF_ALG cannot be initialized before loading begins.

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
  --max-tokens 32 --projection-backend sm87 \
  --prefill-chunk-size 16
```

The correctness reference remains the default. On an SM87 device, select the
validated direct FP8/NVFP4-to-BF16 layer path explicitly with
`--projection-backend sm87`. Aligned canonical NVFP4 projections whose K is a
multiple of 256 automatically use the packed-x8 path; other shapes retain the
checked scalar fallback. Within M=1, aligned exact NVFP4 `[5120,17408]` down,
`[17408,5120]` gate/up, and `[248320,5120]` lm-head use separately gated CTA
activation-staged XOR-dual instances. Down stages its 34-KiB activation, while
gate/up and lm-head each stage 10 KiB. All three use 8-byte cooperative global
copies, so the public alignment contract and checkpoint layout are unchanged.
Near-miss shapes, packed-weight or activation misalignment, M2 through M16,
and prefill retain their previous routes.
Canonical FP8 projections whose K is a multiple of
1,024 use packed-x4 when weights are 4-byte aligned and BF16 activations are
8-byte aligned; other FP8 shapes also retain their scalar fallback. At M=8,
the exact NVFP4 `[17408,5120]` and `[5120,17408]` production projections use
compile-time shape specializations; other aligned M=8 shapes retain the
generic row-pair path. The exact FP8 `[10240,5120]`, `[5120,6144]`,
`[6144,5120]`, `[12288,5120]`, and `[1024,5120]` M=8 projections likewise use
compile-time specializations, while other aligned FP8 M=8 shapes retain their
generic row-pair path. At M=16, the FP8 `[10240,5120]`, `[5120,6144]`,
`[6144,5120]`, and `[12288,5120]` shapes and the NVFP4 `[17408,5120]` and
`[5120,17408]` shapes decode the canonical checkpoint layout into BF16 tiles
for Ampere Tensor Core MMA. The smaller FP8 `[1024,5120]` projection, other
shapes, and insufficiently aligned inputs retain two ordered M=8 launches.
Reuse one loaded engine for repeatability and latency distributions with:

```bash
build/orin-release/qwen3x-orin benchmark MODEL_DIR \
  --prompt "Explain unified memory in one sentence." \
  --max-tokens 2 --warmup 1 --iterations 3 \
  --max-sequence-length 64 --projection-backend sm87 \
  --prefill-chunk-size 16
```

`--prefill-chunk-size` accepts 1 through 16 and defaults to 1. Values above 1
batch the prompt prefix into layer-major projection tiles while preserving
causal Conv/GDN/KV updates; the final prompt token and all decode steps remain
single-token operations. With the SM87 backend, M9 through M15 quantized tiles
are split into an M8 launch plus the remaining M1..M7 rows. M16 selects the
shape-gated Tensor Core path above or safely falls back to two M8 launches;
the reference backend and BF16 weights retain ordered M1 launches. Trace
capture deliberately reports and uses an effective chunk size of 1 so existing
per-token boundary hashes retain their ordering contract.
Exact aligned SM87 FP8 M1 linear-attention QKV/Z projections use one
two-phase launch; C2 through C16 prefix tiles, near-miss shapes, unaligned
operands, and other backends retain two ordered projections.

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
That original run is available as
[machine-readable metadata](docs/metadata/qwen36-27b-native-reference-run.json).
That is the historical portable-hash reference. With the current automatic
AF_ALG path and the SM87 projection backend, a diagnostic two-token run reported
21.485 seconds for resident loading and 22.638 seconds for total engine loading;
the packed-x8 and packed-x4 two-prompt milestones then reached 6.107 seconds
median TTFT and 385.181 ms median subsequent-token latency. With the same
two-prompt/two-token benchmark shape, the first C8 prefix-tiling milestone
reached 2.006 seconds median TTFT, 2.389 seconds total generation, and
383.320 ms median subsequent-token latency. The post-C8 kernel sequence at
`5fe0ae0` then reached 1.021 seconds TTFT, 1.206 seconds total generation, and
185.108 ms subsequent-token latency. Its 64-position request arena remains
85,011,968 bytes, 1,190,912 bytes above C1. The final same-binary fixed-shape
NVFP4 M=8 gate measured a 1.16079x weighted speedup while reducing normalized
SASS from 1,272 to 1,144 instructions. The corresponding FP8 M=8 gate measured
a 1.13694x call-weighted speedup across five production shapes and reduced
normalized SASS from 1,864 to 784 instructions. The complete optimized C8
fixed-oracle run retained all 19 prompt IDs, 26 output IDs, exact text,
`<|im_end|>`, and all 44 steps. The C16 path retains that same oracle while
reserving an 86,373,376-byte 64-position request arena versus 85,011,968 bytes
at C8. Its same-binary C16 medians reduce TTFT and total generation by 25.468%
and 21.552% relative to C8; the FP8 and NVFP4 fixed-M16 kernel gates provide
2.41756x and 1.56406x call-weighted speedups over two production M8 launches.
The C16 Nsight diagnostic records 929.615 ms across 9,210 kernel instances,
versus the historical C8 trace's 1,192.639 ms across 10,107 instances. These
are diagnostic, unlocked-clock measurements rather than a release claim.
The decode-only full-attention FP8 K/V projection-pair path then reduced the
matched max-26 profile from 832 independent K/V kernel instances taking
39.328 ms to 416 fused pair instances taking 31.317 ms, a 1.25583x speedup for
the replaced work. A mirrored baseline/candidate/candidate/baseline diagnostic
reduced the average total-generation median by 7.123 ms (0.196%) and the
subsequent-token median by 0.219%, while every run retained the exact 26-token
oracle. This measurement was also taken with unlocked clocks and is not a
release claim; see the
[FP8 K/V projection-pair record](docs/metadata/qwen36-27b-fp8-kv-pair-benchmark.json).
The subsequent aligned M1 NVFP4 `[5120,17408]` down-projection diagnostic
measured 1.02808x/1.02775x checkpoint-like and 1.03670x/1.03572x same-bank
speedups in two repeated same-binary gates. Its matched max-26 profile reduced
that down work from 586.411200 to 568.161568 ms and the mirrored single-load
benchmark reduced average total generation by 15.457 ms (0.427106169%) and
subsequent-token latency by 0.595 ms (0.488724429%). These unlocked-clock
measurements are diagnostic, not a release claim; see the
[NVFP4 down dual-iteration record](docs/metadata/qwen36-27b-nvfp4-down-dual-benchmark.json).
The subsequent aligned M1 NVFP4 `[17408,5120]` gate/up XOR-dual diagnostic
measured 1.04708x/1.04656x checkpoint-like and 1.06423x/1.06438x same-bank
speedups in two repeated same-binary gates. Its matched max-26 profile reduced
3,328 gate/up kernel instances from 1,123.488832 to 1,073.110560 ms and
reduced aggregate CUDA-kernel time by 49.607008 ms (1.377722691%). The
mirrored single-load benchmark reduced average total generation by 50.9235 ms
(1.412995418%) and subsequent-token latency by 1.9465 ms (1.606573208%).
These unlocked-clock measurements are diagnostic, not a release claim; see
the
[NVFP4 gate/up XOR-dual record](docs/metadata/qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json).
The following aligned M1 NVFP4 `[248320,5120]` lm-head XOR-dual diagnostic
measured 1.05649x checkpoint-like and 1.07370x same-bank speedups in its
same-binary production gate. Its matched max-26 profile reduced 26 lm-head
launches from 125.008736 to 117.825600 ms and reduced aggregate CUDA-kernel
time by 8.014432 ms (0.225692174%). The mirrored single-load benchmark reduced
the average-of-process-medians total generation by 7.4135 ms (0.208554724%)
and subsequent-token latency by 0.267 ms (0.223859949%). All runs retained the
exact oracle. These unlocked-clock measurements are diagnostic, not a release
claim; see the
[NVFP4 lm-head XOR-dual record](docs/metadata/qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json).
The next NVFP4 data-reuse follow-up replaced the down indexed dual-iteration
route with the adjacent-lane XOR-dual kernel and staged the lm-head activation
once per CTA. The final same-binary gates measured 1.05609x/1.05610x down and
1.02228x/1.02937x lm-head speedups on checkpoint-like/same-bank fixtures. In
matched max-26 profiles, the two target routes fell from 685.548800 to
653.925344 ms and aggregate CUDA-kernel time fell by 30.878208 ms (0.871519514%).
The B-C-C-B benchmark against an independently rebuilt base commit reduced
average total generation from 3,548.701 to 3,514.877 ms (0.953137500%) and
subsequent-token latency from 119.0265 to 117.7150 ms (1.101855469%), with the
exact oracle preserved.
These unlocked-clock measurements remain diagnostic; see the
[NVFP4 data-reuse record](docs/metadata/qwen36-27b-nvfp4-data-reuse-benchmark.json).
The next gate/up follow-up stages the same 10-KiB activation once per CTA for
aligned M1 `[17408,5120]`. Its final same-binary gate measured 1.01436x on the
checkpoint-like fixture and 1.02093x on the same-bank stress fixture. A
matched max-26 profile reduced 3,328 gate/up kernels from 1,074.533504 to
1,054.402944 ms and aggregate CUDA-kernel time from 3,512.152960 to
3,490.693120 ms. An independent-base B-C-C-B benchmark reduced average total
generation from 3,515.5365 to 3,498.1105 ms (0.495685%) and subsequent-token
latency from 117.7605 to 117.0925 ms (0.567253%), with the exact oracle
preserved. Clocks remained unlocked, so this is diagnostic evidence rather
than a release claim; see the
[NVFP4 gate/up activation-staging record](docs/metadata/qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json).
The following down-projection follow-up stages the 34-KiB activation once per
CTA for aligned M1 `[5120,17408]`. Its final same-binary gate measured 1.02862x
on the checkpoint-like fixture and 1.03026x on the same-bank stress fixture.
The matched max-26 profile reduced 1,664 down kernels from 536.467904 to
516.386464 ms and aggregate CUDA-kernel time from 3,490.693120 to
3,471.580320 ms. A saved-base B-C-C-B benchmark reduced average total
generation from 3,498.3615 to 3,478.2960 ms (0.573569%) and subsequent-token
latency from 117.1025 to 116.3305 ms (0.659252%), with the exact oracle
preserved. Clocks remained unlocked, so this is diagnostic evidence rather
than a release claim; see the
[NVFP4 down activation-staging record](docs/metadata/qwen36-27b-nvfp4-down-activation-staged-benchmark.json).
The subsequent exact aligned FP8 M1 QKV/Z fusion shares one decoded codebook
across topology-preserving QKV and Z phases. Five frozen same-binary processes
measure 1.05501x to 1.05868x on actual checkpoint bytes and 1.00832x to
1.01092x on the same-bank stress guard. A matched max-26 profile removes 1,248
launches and reduces target work from 634.147712 to 615.753920 ms. A B-C-C-B
diagnostic against a detached base reduces average total generation from
3,452.8470 to 3,434.3865 ms (0.534646%) and subsequent-token latency from
115.7860 to 115.0580 ms (0.628746%). C1, C8, and C16 retain the exact
19/26-token oracle.
Clocks remained unlocked and the result is diagnostic, not a release or
serving-throughput claim; see the
[FP8 QKV/Z fusion record](docs/metadata/qwen36-27b-fp8-qkv-z-fusion-benchmark.json).
At the earlier packed-x4 C1 milestone, the complete 26-token fixed-oracle CTest
had fallen from 234.35 to 40.60 seconds while retaining exact IDs, text, stop
semantics, and runner steps. See the
[updated performance evidence](docs/PERFORMANCE_BASELINE.md) and its
[machine-readable AF_ALG](docs/metadata/qwen36-27b-afalg-loader-benchmark.json)
and
[packed-x8 records](docs/metadata/qwen36-27b-nvfp4-packedx8-benchmark.json),
plus the
[packed-x4 records](docs/metadata/qwen36-27b-fp8-packedx4-benchmark.json) and
[C8 prefill record](docs/metadata/qwen36-27b-c8-prefill-benchmark.json), plus
the
[post-C8 kernel record](docs/metadata/qwen36-27b-c8-kernel-optimization-benchmark.json).
The C16 implementation, same-binary comparison, Tensor Core gates, and profile
are recorded in the
[C16 Tensor Core prefill record](docs/metadata/qwen36-27b-c16-tensor-core-prefill-benchmark.json).

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

Native inference is currently a bounded, batch-one surface with a correctness
reference, opt-in shape-gated projection optimization, and opt-in `C<=16`
prompt-prefix tiling. Recurrent state and causal attention still advance token
by token inside each tile. It does not yet provide large-prefill kernels,
continuous batching, a server, or a release-grade performance claim. The independent
target-device oracle,
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
