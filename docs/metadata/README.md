# Pinned checkpoint metadata evidence

This directory records normalized facts used to design and test Qwen3x-Orin.
The reports are evidence artifacts, not model files and not end-to-end support
claims. A report reaches only `metadata-compatible` until all referenced shard
headers and tensor contracts pass the runtime inspector on the target Orin.

The current reports pin the official NVIDIA ModelOpt artifacts:

- `nvidia/Qwen3.6-27B-NVFP4` at
  `0893e1606ff3d5f97a441f405d5fc541a6bdf404`;
- `nvidia/Qwen3.6-35B-A3B-NVFP4` at
  `491c2f1ea524c639598bf8fa787a93fed5a6fbce`.

The machine-readable
[`qwen36-27b-native-reference-run.json`](qwen36-27b-native-reference-run.json)
records the first successful full native 27B generation: exact 19 prompt and
26 oracle output IDs, exact decoded text/stop semantics, cold-load and
generation timings, native position-18/19 trace hashes, and the canonical
reasons native and vLLM boundary hashes are not expected to be bitwise equal.

The diagnostic Phase 3 records are:

- [`qwen36-27b-reference-nsys-baseline.json`](qwen36-27b-reference-nsys-baseline.json),
  which pins the reference CUDA kernel and launch-shape profile, toolchain,
  report hash, and reproducible `nsys stats` commands;
- [`qwen36-27b-projection-backend-benchmark.json`](qwen36-27b-projection-backend-benchmark.json),
  which retains every paired reference/SM87 sample, strict replay results,
  exact full-model gate, memory watermarks, microbenchmarks, and limitations.
- [`qwen36-27b-afalg-loader-benchmark.json`](qwen36-27b-afalg-loader-benchmark.json),
  which records the post-SM87 startup diagnosis, observed kernel SHA-256
  provider, accelerated full-shard authentication/load timings, unchanged
  exact-generation gates, and historical-comparison limitations.
- [`qwen36-27b-parallel-loader-benchmark.json`](qwen36-27b-parallel-loader-benchmark.json),
  which records the same-binary warm-cache one-/two-/three-worker resident-load
  measurements, deterministic loader invariants, derived speedups, and
  measurement limitations.
- [`qwen36-27b-startup-overlap-benchmark.json`](qwen36-27b-startup-overlap-benchmark.json),
  which records the same-binary serial/overlapped tokenizer-and-resident
  startup comparison, exact full-model gates, wall-time contract, fallback
  behavior, and measurement limitations.
- [`qwen36-27b-nvfp4-packedx8-benchmark.json`](qwen36-27b-nvfp4-packedx8-benchmark.json),
  which records the canonical packed-x8 NVFP4 dispatch, same-binary scalar/
  vector gate, two-prompt replay benchmark, full-model exact gate, SASS
  resources, and post-change Nsight profile.
- [`qwen36-27b-fp8-packedx4-benchmark.json`](qwen36-27b-fp8-packedx4-benchmark.json),
  which records the canonical packed-x4 FP8 dispatch, exhaustive E4M3FN and
  fallback gates, same-binary scalar/vector measurements, exact 27B replay,
  SASS resources, and the post-change Nsight profile.
- [`qwen36-27b-c8-prefill-benchmark.json`](qwen36-27b-c8-prefill-benchmark.json),
  which records the first bounded C1/C8 prompt-prefix comparison, exact C8
  full-model oracle gate, request-arena increment, execution policy, and
  diagnostic limitations.
- [`qwen36-27b-c8-kernel-optimization-benchmark.json`](qwen36-27b-c8-kernel-optimization-benchmark.json),
  which records the subsequent C8 kernel-optimization chain, matched-shape
  end-to-end medians, GDN and NVFP4 row-pair microbenchmarks, unchanged exact
  full-model gate, dispatch scope, and diagnostic limitations.
- [`qwen36-27b-c16-tensor-core-prefill-benchmark.json`](qwen36-27b-c16-tensor-core-prefill-benchmark.json),
  which records the bounded C16 runtime and GDN extension, fixed-shape FP8 and
  NVFP4 Tensor Core dispatch, exhaustive correctness gates, mirrored same-binary
  C8/C16 samples, and the final C16 Nsight kernel profile.
- [`qwen36-27b-fp8-kv-pair-benchmark.json`](qwen36-27b-fp8-kv-pair-benchmark.json),
  which records the exact-shape M1 FP8 K/V projection pair, same-binary
  synthetic kernel gate, max-26-token Nsight comparison, mirrored single-load
  generation benchmark, exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-dual-benchmark.json`](qwen36-27b-nvfp4-down-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[5120,17408]` down-projection
  dual-iteration kernel, repeated same-binary gates, unchanged fallbacks,
  max-26-token Nsight comparison, mirrored single-load generation benchmark,
  exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json`](qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[17408,5120]` gate/up adjacent-lane
  XOR-dual kernel, repeated same-binary gates, unchanged fallbacks, max-26-token
  Nsight comparison, mirrored single-load generation benchmark, exact replay
  gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json`](qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[248320,5120]` lm-head adjacent-lane
  XOR-dual kernel, its same-binary production gate, unchanged fallbacks,
  max-26-token Nsight comparison, mirrored single-load generation benchmark,
  exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-data-reuse-benchmark.json`](qwen36-27b-nvfp4-data-reuse-benchmark.json),
  which records the follow-up aligned M1 `[5120,17408]` down adjacent-lane
  XOR-dual route and `[248320,5120]` lm-head CTA activation-staged route,
  same-binary identity/performance gates, B-C-C-B generation evidence against
  an independently rebuilt base commit, matched max-26-token profiles, exact
  replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json`](qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json),
  which records the follow-up aligned M1 `[17408,5120]` gate/up CTA
  activation-staged route, default capture-only exact/fallback dispatch
  identity gates, same-binary performance and bitwise gates, an independently
  rebuilt-base B-C-C-B generation benchmark, matched max-26-token profiles,
  exact replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-activation-staged-benchmark.json`](qwen36-27b-nvfp4-down-activation-staged-benchmark.json),
  which records the follow-up aligned M1 `[5120,17408]` down CTA
  activation-staged route, its preserved direct-XOR baseline and scalar
  fallbacks, same-binary performance and bitwise gates, a saved-base B-C-C-B
  generation benchmark, matched max-26-token profiles,
  exact replay, and diagnostic limitations.
- [`qwen36-27b-gdn-eight-row-benchmark.json`](qwen36-27b-gdn-eight-row-benchmark.json),
  which records the production eight-row lane-striped GDN update against its
  preserved four-row predecessor, M1/M2/M8/M16 bitwise state/output gates,
  mirrored same-binary measurements with current-profile call weights, matched
  max-26 profiles, compiler resources, exact replay, and diagnostic
  limitations.
- [`qwen36-27b-fp8-qkv-z-fusion-benchmark.json`](qwen36-27b-fp8-qkv-z-fusion-benchmark.json),
  which records the exact aligned FP8 M1 linear-attention QKV/Z two-phase
  fusion, five frozen actual-checkpoint/stress gates, compiler resources,
  matched max-26 profiles, detached-base B-C-C-B generation evidence, C1/C8/C16
  exact replay, ordered fallbacks, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json`](qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 dense-MLP gate/up/SiLU fusion,
  five frozen actual-checkpoint/same-bank gates, compiler resources, matched
  max-26 profiles, detached-base B-C-C-B generation evidence, C1/C8/C16 exact
  replay, ordered fallbacks, and diagnostic limitations.
- [`qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json`](qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 post-attention residual-add,
  centered-RMSNorm, gate/up, and SiLU fusion, its redundant per-CTA norm
  reduction, five frozen gates, dispatch/alias contracts, matched profiles,
  detached-base B-C-C-B evidence, exact replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json`](qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json),
  which records the reduction-only follow-up inside that post-attention
  fusion: shared-memory strides 128/64/32 followed by exact warp-zero shuffle
  strides 16/8/4/2/1, the preserved full-shared-tree same-binary predecessor,
  compiler resources, five direct actual-checkpoint/same-bank B-C-C-B gates,
  a matched max-26 profile, an independent B-C-C-B generation comparison,
  a bitwise-equal 5,905-line trace, full-shape finite/nonfinite closeout gates,
  and diagnostic limits.
- [`qwen36-27b-fp8-q-kv-fusion-benchmark.json`](qwen36-27b-fp8-q-kv-fusion-benchmark.json),
  which records the exact aligned FP8 M1 full-attention Q plus K/V fusion,
  five same-binary production-shaped gates, compiler resources, complete
  dispatch/alias contracts, matched profiles, detached-base B-C-C-B evidence,
  exact model gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json`](qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 dense-MLP down projection plus
  residual-add and centered-RMSNorm cooperative fusion, five production-shaped
  gates, finite and signed-nonfinite bitwise contracts, zero-node validation,
  matched profiles, detached-base B-C-C-B and full-trace equality, exact model
  gates, the zero-residency-margin AGX Orin constraint, and diagnostic limits.
- [`qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json`](qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json),
  which records the canonical M1 GDN plus 48x128 headwise plain-RMSNorm/SiLU
  gate implementation, its one-node/fallback/alias/resource contracts, the
  hardened 24-state same-binary gate, matched max-26 profiles, detached-base
  B-C-C-B and 5,905-line full-trace equality, C1/C8/C16 exact-model gates,
  release/sanitizer suites, and diagnostic limitations. Five earlier probes
  remain explicitly excluded as pre-production prototype inventory.
- [`qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json`](qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json),
  which records the exact shared-prefix/warp-zero reduction tail inside that
  production fused GDN kernel, the test-only full shared-tree predecessor,
  identical launch/resource and three-step bitwise contracts, the 28-to-24
  SASS `BAR` reduction, five hardened cold-state same-binary gates, the
  directly attributed max-26 target-kernel delta, exact model/trace suites,
  and the absence of an attributable detached-process end-to-end gain. Early
  prototypes and the test-only launcher are explicitly excluded from the
  production claim.
- [`qwen36-27b-prefill-decode-plan-split-benchmark.json`](qwen36-27b-prefill-decode-plan-split-benchmark.json),
  which records the host-control Prefill/Decode plan seam, explicit prefix,
  final-prompt, and feedback routing, the 44-case controller matrix, exact
  C1/C8/C16 model and arena gates, 5,905-line trace equality, identical ordered
  13,558-launch contracts, detached-base B-C-C-B regression evidence, and the
  explicit absence of new streams, buffers, kernels, or an attributed speedup.
- [`qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json`](qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json),
  which freezes the resolved SM87 projection route registry, its exact
  launch-contract and B-C-C-B equivalence gates, all 29 intended M1/M2/M8/M16
  direct-kernel cells, the P19 C1/C2/C8/C16 TTFT matrix, tokenizer-pinned
  P33/P65/P129/P513 prompts, and matched Nsight launch/time attribution. It
  records the unrelated historical M4 aggregate-threshold failure separately
  instead of treating that measurement gap as a production correctness error.
- [`qwen36-27b-c32-composite-prefill-benchmark.json`](qwen36-27b-c32-composite-prefill-benchmark.json),
  which records the ABI-0.2.0 C32 composite outer tile, exact real-model gates,
  P19 B-C-C-B and long-prompt C16/C32 timings, matched Nsight attribution, and
  the explicit single-stream M16-plus-tail projection and unlocked-clock limits.
- [`qwen36-27b-fp8-m32-production-benchmark.json`](qwen36-27b-fp8-m32-production-benchmark.json),
  which records the production fixed-M32 FP8 route, four-shape bitwise/guard/
  single-node gates, complete fallback validation, exact C1/C8/C16/C32 model
  replay, P33/C32 B-C-C-B TTFT evidence, and matched Nsight launch/time
  attribution against the frozen composite baseline.

The model-compatibility reports contain raw SHA-256 hashes for `config.json`,
`hf_quant_config.json`, and `model.safetensors.index.json`; normalized model and
quantization fields; index counts; representative shapes read from the first
safetensors shard header; and benchmark shapes derived from that evidence.
Tensor payloads were not downloaded to produce the header probes: the first
eight bytes were read to obtain the little-endian header length, followed by a
bounded HTTP byte-range request for the JSON header.

The pinned 27B artifact was subsequently materialized in a separate local
model directory. Git LFS and an independent `sha256sum` pass matched all three
published object IDs. `qwen3x-inspect checkpoint MODEL_DIR --require-shards`
then validated all 3 shard headers, all 2,194 index mappings, and exactly
21,921,428,072 payload bytes without reading payload contents. The 27B report
records all shard hashes/sizes and a text/vision/MTP storage split. No model
file is copied into this repository.

Once `qwen3x-inspect` is built, local evidence can be reproduced without loading
weight payloads into memory:

```bash
qwen3x-inspect index MODEL_DIR/model.safetensors.index.json
qwen3x-inspect header MODEL_DIR/model-00001-of-00003.safetensors
qwen3x-inspect checkpoint MODEL_DIR
qwen3x-inspect checkpoint MODEL_DIR --require-shards
qwen3x-inspect manifest MODEL_DIR
qwen3x-inspect load-plan MODEL_DIR
```

The strict form requires regular, non-symlink shard files and validates every
header, tensor membership/ownership, and aggregate payload size. The
non-strict form is intentionally metadata-only: it reports shard presence but
sets `shard_contract_validated=false`.

`load-plan` additionally assigns the exact deterministic text-only arena layout
against the three compiled full-file identities without allocating GPU memory.
The separate conditional resident-loader integration then read and SHA-256
authenticated all 21,921,697,184 file bytes in one sequential pass, copied
20,150,569,096 text bytes into one 20,150,786,560-byte CUDA arena, and skipped
1,771,128,088 bytes. Full details and the target-device reproduction command
are in [RESIDENT_WEIGHT_LOADER.md](../RESIDENT_WEIGHT_LOADER.md).

The inspector treats filenames and directory names as untrusted context. Model
series and quantization compatibility require exact pinned descriptors plus
semantic validation. A shape-compatible unknown revision remains unsupported.

## Known upstream metadata quirk

For the pinned 35B-A3B artifact, external `hf_quant_config.json` identifies
ModelOpt 0.44.0 and FP8 KV cache, while the embedded
`config.json.quantization_config` carries stale ModelOpt 0.37.0 metadata and no
KV declaration. Their 291 per-module `quantized_layers` entries agree exactly.
Qwen3x-Orin permits that discrepancy only for the pinned file hashes and treats
the external file as authoritative. The same disagreement on an unknown
revision fails closed.

Model weights, tokenizers, and source configuration files remain separately
licensed artifacts and are not copied into this repository.
