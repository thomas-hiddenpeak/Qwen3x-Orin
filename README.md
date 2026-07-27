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
supports bounded `C=1..512` prompt-prefix tiles. In the historical two-prompt,
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
The latest tokenizer-pinned maximum-one-token matrix on the current SM87 path
measures the same 19-token prompt at 2,031.901, 1,366.633, 831.525, and
554.386 ms TTFT for C1/C2/C8/C16; matched Nsight profiles reduce launch count
from 8,249 to 2,633 while projection still accounts for 91.348% of C16 kernel
time. These remain unlocked-clock batch-one diagnostics.

The fixed-M32 NVFP4 path now also factorizes its shared product lookup. Against
the preceding K256 scale-window binary, the tokenizer-pinned P33/C32 B-C-C-B
diagnostic moves TTFT from 499.0785 to 486.0620 ms (-2.6081%), while matched
Nsight Systems traces reduce gate/up plus down work from 217.672096 to
204.422208 ms (1.06482x) with the same 2,166 launches. C1/C8/C16/C32 continue
to pass the exact 27B model oracle. These measurements remain unlocked-clock,
batch-one evidence rather than a serving-throughput claim.

The current exact NVFP4 M18 prefill route replaces the usual ordered M16+M2
composition for the aligned `[17408,5120]` and `[5120,17408]` MLP projections
with one masked-M32 kernel. Its final production-call-weighted microbenchmark
measures 1.73817x over the preceding M16+M2 path. A detached-binary P19/C32
B-C-C-B comparison reduces median TTFT from 548.7825 to 439.5980 ms
(1.24837x), and matched Nsight traces reduce all CUDA kernels from 2,264 to
2,072 while the target MLP projections fall from 384 to 192 launches. Release
CTest reports 49 passed, 5 skipped, and 0 failed; the host C++ ASan/UBSan build
reports 48 passed, 5 skipped, and 0 failed. Exact C1/C8/C16/C32 model-oracle
runs pass. Device `compute-sanitizer` was attempted, but Orin reported that GPU
debugging features are disabled, so device memcheck is platform-blocked rather
than passed. These are unlocked-clock batch-one diagnostics; see the
[machine-readable M18 record](docs/metadata/qwen36-27b-nvfp4-m18-masked-m32-benchmark.json).

Commit `8b19d2aa27370f7daafb743bf024fbcfc8b25950` extends that exact-capacity
design to M17 and M19 through M31 for the same two aligned NVFP4 MLP shapes,
while retaining the fixed M18 and M32 production specializations. The
production-call-weighted microbenchmark ranges from 1.59909x at M17 to
4.19900x at M31. Across ten tokenizer-pinned prompts, the B-C-C-B comparison
has no reversal and improves the equally weighted TTFT aggregate by 25.914%
(1.349789x). Matched P18, P26, and P64 Nsight traces attribute the gain to
removing M16-plus-tail work; the P64 target tail falls from 576 launches and
663.078592 ms to 192 launches and 171.773312 ms (3.860196x). Release, host
ASan/UBSan, C1/C8/C16/C32 oracle, exact-span, graph, and unchanged fixed
M18/M32 SASS gates pass. See the
[runtime-masked M17/M19-M31 record](docs/metadata/qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json)
and its
[pinned prompt manifest](benchmarks/qwen36-27b-sm87-prefill-tail-prompts-v1.json).

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
| Qwen3.6 27B pinned NVIDIA revision | Dense | FP8 W8A16 + NVFP4 W4A16 + BF16 fallback | Native reference generation plus opt-in SM87 M=1 decode and `C<=512` prompt-prefix execution; matched P257/C256 and P513/C512 exact-output gates pass on Orin |
| Qwen3.5 / Qwen3.6 35B-A3B | MoE | FP8 W8A16 + NVFP4 W4A16 + BF16 fallback | Planned, after the dense path |

The current scope is text-only, batch-one correctness and performance. The
achieved non-MTP P19/C32/max26 Decode result of **105.870500 ms/token /
9.445501816 token/s** is frozen as the regression anchor while dedicated
Prefill optimization resumes. A minimal OpenAI-compatible evaluation gateway
is staged alongside Prefill work; Paged KV, continuous batching, production
serving, MTP, and vision remain later work. See [the roadmap](docs/ROADMAP.md).

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
  --prefill-chunk-size 32
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
Near-miss shapes and packed-weight or activation misalignment retain their
previous routes. M2 through M15 retain the small-M routes; exact M17 and M19
through M31, fixed M18, and fixed M32 follow the masked/fixed-tile dispatch
described below.
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
  --prefill-chunk-size 32
```

`--prefill-chunk-size` accepts 1 through 512 and defaults to 1. Values above 1
batch the prompt prefix into layer-major projection tiles while preserving
causal Conv/GDN/KV updates; the final prompt token and all decode steps remain
single-token operations. Extending `ReferencePrefillTileResult::steps` to 512
entries is a public C++ ABI change; the current project/package ABI is `0.4.0`,
and exact-version consumers must rebuild rather than mix older objects with the
new static library. At the default 128-token sequence capacity, C512 uses an
exact 88,123,392-byte workspace and 174,991,360-byte request arena. At the
absolute 262,144-token capacity those values are 112,295,936 and
17,437,720,576 bytes, respectively.

With the SM87 backend, M9 through M15 quantized tiles
are split into an M8 launch plus the remaining M1..M7 rows. M16 selects the
shape-gated Tensor Core path above or safely falls back to two M8 launches.
At M17 and M19 through M31, aligned NVFP4 `[17408,5120]` and
`[5120,17408]` projections use one runtime-valid-count masked-M32 Tensor Core
kernel when packed weights are 16-byte aligned, block scales are 2-byte
aligned, and the BF16 input is 8-byte aligned. The public API requires only
the exact M-row input/output spans: internal rows M through 31 are zero-filled,
never read from the caller, and never stored. M18 retains its separately fixed
masked-M32 API and specialization with the same exact-C18 contract. Every FP8
M17..M31 case, plus NVFP4 shape near-misses or calls that miss any alignment
gate, uses the ordered M16 plus at-most-M8 tail fallback. The pair dispatcher
validates both complete projections and their cross-ranges before enqueue,
then either issues two ordered single-kernel masked projections or preserves
the existing ordered recursive fallback. These M17..M31 MLP calls remain
serial on the main stream. At M32, the four exact
aligned FP8 production shapes and the exact aligned NVFP4
`[17408,5120]` gate/up and `[5120,17408]` down shapes use one fixed-M32 Tensor
Core kernel. The NVFP4 route is the factorized K64/LD72 kernel: it keeps two
16-token activation panels resident, coalesces scales in K256 windows, and
constructs BF16 products from compact E2M1-pair and E4M3 lookup tables before
reusing each decoded weight tile across independent WMMA accumulator chains.
Other valid FP8 and NVFP4 M32 cases use two ordered M16 launches. Conv/GDN and
Q/K+RoPE retain ordered subtiles of at most 16 rows; the reference backend and
BF16 weights retain ordered M1 launches. Trace capture deliberately reports
and uses an effective chunk size of 1 so existing
per-token boundary hashes retain their ordering contract.

For a partial-wide C33..C63 public tile, dispatch validates the complete tile
and all ranges before enqueue, then executes one ordered C32 prefix followed by
the exact C1..C31 tail. At C64, exact aligned NVFP4 `[5120,17408]` down uses one
M64 kernel; every other projection route preserves two ordered C32 schedules.
The runner keeps non-down projections on at-most-C32 subtiles, applies residual
add/RMSNorm as two exact M32 operations, and retains the ordered at-most-M16
causal Conv/GDN and Q/K+RoPE subtiles.

The request controller schedules larger prefixes with the explicit palette
`{C512,C256,C64,C32,tail<=31}`. On SM87, exact C256/C512 full-attention tiles
use one bulk causal GQA plus sigmoid-Gate kernel, and exact aligned NVFP4
`[5120,17408]` Down and FP8 linear-attention QKV `[10240,5120]`, Z
`[6144,5120]`, and attention output `[5120,6144]` each use one N-major
whole-chunk grid. Full-attention Q `[12288,5120]` also uses N-major at C256 and
C512; K/V `[1024,5120]` use M-major at C256's under-filled 32-CTA grid and
N-major at C512. Generic projection APIs remain capped at C64: NVFP4 Gate/Up,
residual/RMS, Conv/GDN, other shapes, and every near miss retain their
established ordered subtiles. The early synchronous M64 Gate/Up candidate was
rejected, but the later test-only whole-chunk main/aux pair clears its frozen
C512 gate at 1.12867x. It is not yet production-integrated: the fresh current
P513 trace still executes all 10,129 Prefix GPU operations on one stream, with
2,048 serial Gate/Up launches. Integrating that selected pair is the next
bounded step before any C1024 test-only expansion.

Exact aligned SM87 FP8 M1 full-attention Q/K/V projections use one launch for
the ordered `[12288,5120]`, `[1024,5120]`, and `[1024,5120]` weights. Near-miss
shapes, unaligned operands, C2 through C64 prefix tiles, and other backends
retain the prevalidated Q projection followed by the existing K/V paired or
independent fallback.
Exact aligned SM87 FP8 M1 linear-attention QKV/Z projections use one
two-phase launch; C2 through C64 prefix tiles, near-miss shapes, unaligned
operands, and other backends retain two ordered projections.
Exact aligned SM87 NVFP4 M1 dense-MLP gate/up projections and their SiLU
multiply likewise use one rolled two-phase projection plus CTA-parallel
epilogue launch. Gate and up are independently rounded to BF16 before the
epilogue. The generic low- and high-level APIs retain both observable rounded
outputs. The explicit Decode-runner-only exact route instead holds the pair in
two CTA-local `BF16[576]` arrays, publishes only the final
`SiLU(gate) * up`, and leaves its validated `up_workspace` untouched. At the
production post-attention boundary, that route also folds BF16 residual
addition and centered RMSNorm into the launch: CTA 0 writes the public
residual, while the normalized activation remains CTA-local. Its repeated
256-thread RMS reduction preserves the shared-tree strides 128/64/32 and the
exact remaining pairings through warp-zero shuffle-down strides 16/8/4/2/1.
The runner does not observe `up_workspace` before the following same-stream
down projection overwrites it. C2 through C64 prefix tiles, near-miss or
unaligned operands, BF16/FP8 weights, other backends, and every fallback retain
the validated ordered chain and may write the workspace.
The following exact aligned SM87 NVFP4 M1 `[5120,17408]` down projection can
likewise absorb its BF16 residual-add and centered-RMSNorm boundary into one
64-CTA cooperative launch. It retains three separately rounded public BF16
boundaries: raw down, residual, and normalized output. A grid synchronization
separates the raw/residual writes from the norm reduction. On the tested
16-SM AGX Orin, four active CTAs per SM provide exactly the 64 resident CTAs
required by the launch, with no cooperative-capacity margin. Decode keeps the
raw output in the now-dead up workspace for trace capture, the residual in
`hidden[0]`, and the normalized next-layer input in `hidden[1]`. Near-miss or
unaligned operands, C2 through C64 prefill, other weight types, and the
reference backend retain the fully prevalidated down-then-residual/norm chain.
The canonical M1 linear-attention GDN update and its following 48-head by
128-wide plain-RMSNorm/SiLU-gate boundary can also use one 48-CTA launch. Each
CTA owns one complete value head, publishes the raw GDN result to BF16, and
reads that exact rounded boundary back after a block barrier before the norm
and gate overwrite; the kernel needs neither a cooperative launch nor a grid
barrier. Other valid norm partitions and multi-token prefill retain the
prevalidated two-launch path. The final 40-register, 34,568-byte-shared kernel
permits four active blocks per SM; graph and bitwise gates cover its one-node
canonical route, ordered fallbacks, aliases, in-place and disjoint state, and
pre-enqueue failures. Five early same-binary probes remain catalogued as
excluded prototype inventory; the hardened production microbenchmark and
separate exact-model, full-trace, end-to-end, and Nsight diagnostics are final.
Production now also finishes that fused kernel's RMS sum with the exact
warp-zero shuffle-down tail after shared strides 128/64/32. The test-only full
shared tree remains available for direct same-binary comparison. Both use the
same 40-register/34,568-byte-shared launch resources, while SASS `BAR` count
drops from 28 to 24.

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
The subsequent exact aligned NVFP4 M1 dense-MLP fusion stages the activation
and both decode codebooks once, runs gate and up as rolled projection phases,
then computes SiLU times up across the CTA. Five frozen same-binary processes
measure 1.02593x to 1.02800x on actual checkpoint bytes and 1.02555x to
1.02764x on the same-bank stress guard. A matched max-26 profile replaces
4,992 gate/up/SiLU kernels with 1,664 fused kernels and reduces target time
from 1,063.999712 to 1,037.277440 ms. A detached-base B-C-C-B diagnostic
reduces average total generation from 3,434.2100 to 3,404.6840 ms (0.859761%)
and subsequent-token latency from 115.0690 to 113.9415 ms (0.979847%). C1,
C8, and C16 retain the exact 19/26-token oracle. Clocks remained unlocked, so
the result is diagnostic rather than a release or serving-throughput claim;
see the
[NVFP4 gate/up/SiLU fusion record](docs/metadata/qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json).
The next exact aligned NVFP4 M1 boundary fusion absorbs post-attention BF16
residual addition and centered RMSNorm into that existing gate/up/SiLU kernel.
Five final same-binary processes measure 1.02507x to 1.02593x on checkpoint
bytes and 1.02534x to 1.02558x on the same-bank guard. The matched max-26
profile removes another 1,664 launches and reduces all CUDA-kernel time from
3,400.911136 to 3,383.698144 ms. A detached-base B-C-C-B diagnostic reduces
average total generation from 3,405.9070 to 3,380.9000 ms (0.734224%) and
subsequent-token latency from 113.9935 to 113.0220 ms (0.852242%). C1, C8,
and C16 preserve the exact oracle. Clocks remained unlocked, so this is
diagnostic rather than release or serving-throughput evidence; see the
[NVFP4 residual/norm/gate/up/SiLU fusion record](docs/metadata/qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json).
The reduction-only follow-up keeps that kernel's exact 64-CTA/256-thread
topology and per-thread accumulation, but stops the shared-memory tree after
strides 128/64/32 and uses warp-zero shuffle-down strides 16/8/4/2/1 with the
same pairings. The preserved shared-tree predecessor and production warp-tail
kernel remain available in one test binary. Five direct B-C-C-B processes
measure 1.01483x to 1.01598x on pinned checkpoint bytes and 1.01356x to
1.02103x on the same-bank guard; every process clears the frozen 1.005x gate
for both fixtures without changing the 64-register, 11,328-byte shared-memory,
zero-local, four-active-block resource envelope. A matched max-26 profile
reduces the 1,664 target kernels from 1,054.426816 to 1,044.102176 ms; non-target
kernels rise by 7.352640 ms, leaving an all-CUDA reduction from 3,360.869920 to
3,357.897920 ms. An independent base/candidate B-C-C-B run reduces average
total generation from 3,367.8495 to 3,345.1965 ms
(0.672625%) and subsequent-token latency from 112.5065 to 111.7115 ms
(0.706626%), with identical prompt/generated IDs, text, stop, and 44 steps.
An independent base/candidate trace comparison also matches all 5,905
canonical prompt/generated/boundary lines. Release passes 52/52 tests, and
ASan/UBSan passes 51/51 with `package_consumer` excluded; both suites retain
the four model-dependent skips. Clocks remained unlocked, so the
results remain diagnostic rather than serving-throughput or release claims.
See the
[NVFP4 residual/norm warp-tail reduction record](docs/metadata/qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json).
The following exact aligned FP8 M1 full-attention fusion preserves the existing
Q row-quad and paired K/V reduction orders while executing all three ordered
projections in one kernel. Five independent processes running the synthetic
same-binary gate measure 1.02610x to 1.03165x for the old Q-plus-K/V chain.
Independent matched max-26 profiles replace 832 Q and K/V launches taking
188.508256 ms with 416 fused launches taking 184.886112 ms. A detached-base
B-C-C-B diagnostic reduces the
average total-generation median from 3,381.4900 to 3,376.3400 ms (0.152300%)
and subsequent-token latency from 113.0405 to 112.8370 ms (0.180024%). All
processes preserve the exact oracle. Clocks remained unlocked, so the three
separate measurements are diagnostic rather than release or
serving-throughput evidence; see the
[FP8 full-attention Q+K/V fusion record](docs/metadata/qwen36-27b-fp8-q-kv-fusion-benchmark.json).
The next exact aligned NVFP4 M1 down-boundary fusion preserves the independent
raw-down, BF16 residual, and centered-RMSNorm outputs while replacing the
down-plus-norm chain with one cooperative kernel. Five independent synthetic
same-binary processes measure 1.02862x to 1.03403x. Matched max-26 profiles
replace 3,328 kernels taking 560.404736 ms with 1,664 fused kernels taking
539.154752 ms, a 21.249984 ms directly attributed reduction. A detached
`d047007` B-C-C-B diagnostic reduces average total generation from 3,377.5860
to 3,359.6770 ms (0.530231%) and subsequent-token latency from 112.9050 to
112.1985 ms (0.625747%). An independent max-26 trace comparison also matches
the full 5,902-line contract across all 44 steps, and C1, C8, and C16 preserve
the exact oracle. Clocks remained unlocked, and the zero-margin cooperative
capacity makes this evidence specific to the tested Jetson AGX Orin; see the
[NVFP4 down/residual/norm fusion record](docs/metadata/qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json).
The subsequent canonical M1 GDN boundary fusion combines the production
eight-row update with its 48x128 headwise plain-RMSNorm/SiLU gate. Five
hardened same-binary B-C-C-B processes rotate 36 MiB of state per variant,
compare the complete bank bitwise after timing, and measure 1.11078x to
1.11610x against the ordered two-launch chain. The matched max-26 profile
replaces 2,496 decode target launches taking 44.600480 ms with 1,248 fused
launches taking 40.273216 ms, a directly attributed 4.327264 ms saving. A
detached-base B-C-C-B diagnostic reduces average total generation from
3,345.1885 to 3,339.2345 ms (0.177987%) and subsequent-token latency from
111.6535 to 111.4235 ms (0.205994%). C1, C8, and C16 preserve the exact
19/26-token oracle, and the independent base/candidate trace comparison
matches all 5,905 canonical lines. Release reports zero failures across 52
discovered tests and ASan/UBSan reports zero across 51; both retain four
model-dependent skips. Clocks remained unlocked, so these remain diagnostic
rather than release or serving-throughput claims; see the
[GDN/plain-RMSNorm/SiLU-gate fusion record](docs/metadata/qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json).
The reduction-only follow-up inside that fused GDN kernel preserves exact
floating-point pairings while replacing the shared 16/8/4/2/1 tail with warp
shuffle-down. Five final 36 MiB cold-state same-binary processes measure
1.00812x to 1.01022x against the test-only full shared tree; all compare the
complete state bank and output bitwise after timing. The matched max-26 profile
reduces the unchanged 1,248-launch target from 40.273216 to 39.977728 ms
(1.007391315x). C1/C8/C16 and all 5,905 trace lines remain exact, while the
detached B-C-C-B result is honestly recorded as no end-to-end gain because
average total generation regresses by 0.646 ms. Release and sanitizer suites
remain failure-free with four model-dependent skips. Clocks were unlocked and
the shared-tree launcher is excluded from production; see the
[GDN RMSNorm warp-tail record](docs/metadata/qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json).
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
The resolved route registry, 29-cell direct-kernel atlas, pinned long-prompt
fixtures, chunk matrix, and Nsight attribution are recorded in the
[SM87 shape/chunk/prompt matrix](docs/metadata/qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json).
The latest fixed-clock P19/C32/max26 single-request Decode gate records
**108.2645 ms/token (9.236638048 token/s)** after one exact M1 kernel combines
the established FP8 QKV/Z work with BF16 A/B in 24 light tail CTAs. Both
mirrored end-to-end pairs improve, all outputs remain exact, and fresh Nsys
closes 25 Decode steps with 1,200 composite launches and 1,200 fewer kernel
rows. See the
[production benchmark](docs/metadata/qwen36-27b-fp8-m1-qkv-z-bf16-ab-tail-composite-production-benchmark.json).

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
reference, opt-in shape-gated projection optimization, and opt-in `C<=512`
prompt-prefix tiling. Exact C256/C512 full-attention work is bulked, while
linear-attention recurrent state still advances in ordered C16 subtiles. It
does not yet provide a complete large-Prefill backend,
continuous batching, a server, or a release-grade performance claim. Prefill
and Decode have distinct internal host-control plans but still execute through
the same runner, without double/triple buffering or Prefill/Decode overlap.
Most work, including exact M17 through M31 gate/up, remains serialized on the
main stream.
The narrow exception is exact aligned NVFP4 C32/C64 MLP gate/up: the runner may
overlap gate on its main stream with up on one owned auxiliary stream and join
them with events. C256/C512 wide tiles still issue ordered C32 projections on
one stream; the test-only whole-chunk main/aux pair has passed its screen but
is not yet selected by production. Existing C32/C64 overlap is layer-local
branch overlap, not a general multi-stream scheduler. The
independent target-device oracle,
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
