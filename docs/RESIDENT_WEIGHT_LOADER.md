# Authenticated resident-weight loader

The first production weight path is implemented for the exact pinned
`nvidia/Qwen3.6-27B-NVFP4` revision. It authenticates every byte of all three
safetensors files while copying only the 1,846 text tensors into one CUDA
device arena. Vision and MTP bytes are read and hashed, but never become
resident GPU tensors.

This is a direct, lossless loader. It does not dequantize, requantize, repack,
use managed memory, or fall back to swap-backed oversubscription.

## Public contract

`include/q3x/runtime/resident_weights.h` exposes two layers:

- `build_resident_load_plan(...)` is CPU-only. It validates identities and
  locator ranges, rejects overlap and arithmetic overflow, sorts by
  `(shard, absolute source offset, tensor name)`, and assigns every text tensor
  a deterministic 256-byte-aligned arena offset.
- `load_pinned_qwen36_27b(...)` rebuilds the strict pinned tensor manifest,
  requires the compiled three-shard size/SHA identity, applies the device-memory
  gate, and executes the plan.

`ResidentWeights` is RAII-owned, move-only, and contains exactly one
`cudaMalloc` allocation. CUDA types do not appear in the public header.
`DeviceTensorView` reports the arena offset, opaque device pointer, byte size,
dtype, and logical shape for lookup by the original safetensors tensor name.

The lower-level `load_resident_weights(...)` accepts an explicit manifest and
identity table. It exists for synthetic tests and future pinned descriptors;
production 27B callers should use the pinned entry point.

## I/O and identity boundary

The pinned loader compiles in these full-file identities:

| Shard | File bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00003.safetensors` | 9,965,652,512 | `b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d` |
| `model-00002-of-00003.safetensors` | 9,985,757,032 | `06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d` |
| `model-00003-of-00003.safetensors` | 1,970,287,640 | `e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845` |

The checkpoint root is opened as a directory. Each relative path component is
then opened with `openat`, `O_NOFOLLOW`, and `O_CLOEXEC`; intermediate
components additionally require `O_DIRECTORY`. `fstat` requires a regular file
and the exact pinned size. The loader never performs a path check followed by a
plain pathname reopen.

Each open file descriptor is consumed once, sequentially from byte 0 through
EOF. The same chunk feeds the full-file SHA-256 and every intersecting H2D
scatter operation. Hash verification is therefore not a separate second pass.
A final EOF read and `fstat` detect growth or truncation during loading; any
I/O, hash, or CUDA failure destroys the partial arena.

Two page-locked staging buffers are used in ping-pong order. A per-slot CUDA
event must complete before the CPU reuses that slot, allowing the next host
read and SHA update to overlap the prior asynchronous H2D scatter. The default
chunk is 64 MiB per slot and the hard limit is 256 MiB. The final stream is
synchronized before ownership is returned.

## Exact Orin memory budget

The following values are derived from and verified against the complete pinned
artifact, rather than estimated from parameter count:

| Component | Bytes | GiB | Resident policy |
| --- | ---: | ---: | --- |
| All tensor payloads | 21,921,428,072 | 20.416 | Authenticated; not all copied |
| All three complete files | 21,921,697,184 | 20.416 | Read exactly once |
| Raw text tensors | 20,150,569,096 | 18.766 | Copied losslessly |
| 256-byte-aligned text arena | 20,150,786,560 | 18.767 | One `cudaMalloc` |
| Arena alignment padding | 217,464 | 0.0002 | Included in arena |
| Vision + MTP tensor payload | 1,770,858,976 | 1.649 | Hash-only, not resident |
| File bytes skipped by H2D | 1,771,128,088 | 1.649 | Headers plus vision/MTP |
| Two default staging buffers | 134,217,728 | 0.125 | Page-locked, temporary |
| Default post-arena free margin | 8,589,934,592 | 8.000 | Required, not allocated |

The staging buffers are allocated before `cudaMemGetInfo`; the loader then
requires `free >= arena + min_free_bytes_after_load` before calling
`cudaMalloc`. Thus the default gate accounts for staging pressure and fails
closed before creating a partial model arena. Callers may tighten the resource
limits; reducing the 8 GiB margin is an explicit policy decision.

## Reproduced official integration

On the target Jetson AGX Orin, using the pinned local model and the default
64 MiB chunk, the conditional integration test completed successfully:

```text
elapsed                  212.736 s
arena bytes       20,150,786,560
full-file bytes   21,921,697,184
H2D text bytes    20,150,569,096
skipped bytes      1,771,128,088
chunks                       328
async scatter operations    2,146
full-file SHA matches         3/3
resident tensor views      1,846
```

The test copied samples of the embedding, layer-0 FP8 QKV weight, and
`lm_head.weight_scale_2` back from the device and compared them byte-for-byte
with their authenticated source ranges. It never modifies the model directory.

Reproduce the CPU-only layout without allocating the 20.15 GB arena:

```bash
qwen3x-inspect load-plan MODEL_DIR
```

Run the full conditional integration only when the GPU memory is intentionally
available:

```bash
Q3X_OFFICIAL_27B_ROOT=MODEL_DIR \
  ctest --test-dir build -R resident_weights_cuda --output-on-failure
```

Synthetic tests cover chunk-split tensors, skipped regions, incorrect SHA,
truncation, shard symlinks, unsafe paths, wrapped offsets, stale CUDA last-error
state, statistics, device round-trips, and RAII moves. The CPU plan test is also
run under the host ASan/UBSan configuration.
