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
- [`qwen36-27b-nvfp4-packedx8-benchmark.json`](qwen36-27b-nvfp4-packedx8-benchmark.json),
  which records the canonical packed-x8 NVFP4 dispatch, same-binary scalar/
  vector gate, two-prompt replay benchmark, full-model exact gate, SASS
  resources, and post-change Nsight profile.
- [`qwen36-27b-fp8-packedx4-benchmark.json`](qwen36-27b-fp8-packedx4-benchmark.json),
  which records the canonical packed-x4 FP8 dispatch, exhaustive E4M3FN and
  fallback gates, same-binary scalar/vector measurements, exact 27B replay,
  SASS resources, and the post-change Nsight profile.

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
