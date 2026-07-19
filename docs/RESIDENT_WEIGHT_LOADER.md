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

`ResidentLoadOptions::sha256_backend` defaults to `auto`. On Linux this prepares
one AF_ALG `sha256` operation for every shard before reading any checkpoint
bytes. If that initialization is unavailable, and only in `auto` mode, the
loader uses the portable in-process SHA-256 implementation for the entire load.
Callers can force `portable` or `linux_af_alg`; the successful concrete choice is
reported by `ResidentLoadStats::sha256_backend` and is never mixed across shards.

`ResidentLoadOptions::max_parallel_shards` defaults to 3. The effective worker
count is the smaller of that limit and the number of authenticated shards in the
plan. Each worker owns an independent nonblocking CUDA stream, two events, and
two page-locked staging buffers. Worker count is capped at 16, and the aggregate
staging allocation must not exceed 2 GiB.

## I/O and identity boundary

The pinned loader compiles in these full-file identities:

| Shard | File bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00003.safetensors` | 9,965,652,512 | `b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d` |
| `model-00002-of-00003.safetensors` | 9,985,757,032 | `06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d` |
| `model-00003-of-00003.safetensors` | 1,970,287,640 | `e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845` |

The checkpoint root is opened as a directory. Each relative path component is
then opened with `openat`, `O_NOFOLLOW`, and `O_CLOEXEC`; intermediate
components additionally require `O_DIRECTORY`, while the final component is
opened with `O_NONBLOCK`. `fstat` requires a regular file and the exact pinned
size, so a FIFO or other non-regular final component is rejected without a
blocking open. The loader never performs a path check followed by a plain
pathname reopen.

Each open file descriptor is consumed once, sequentially from byte 0 through
EOF. Independent shard workers perform read, full-file SHA-256, and H2D scatter
in parallel. Within a shard, the same chunk feeds the hash and every
intersecting scatter operation, so hash verification is not a separate second
pass. A final EOF read and `fstat` detect growth or truncation during loading;
any I/O, hash, or CUDA failure destroys the partial arena. Per-shard statistics
are aggregated in load-plan order, and concurrent failures return the first
failed shard in that order rather than scheduler-dependent output.

The Linux backend sends each complete chunk to the kernel's streaming AF_ALG
SHA-256 operation, then receives the standard 32-byte digest. On Jetson AGX
Orin the generic `sha256` name selects the kernel's ARMv8 SHA2 implementation.
Backend setup failure is distinct from a streaming/finalization failure: only
the former may fall back under `auto`, while an error after any bytes have been
consumed fails closed and releases the partial arena.

Each worker uses two page-locked staging buffers in ping-pong order. A per-slot
CUDA event must complete before the CPU reuses that slot, allowing the next host
read and SHA update to overlap the prior asynchronous H2D scatter. The default
chunk is 64 MiB per slot and the hard limit is 256 MiB. With the default three
workers this is six buffers, or 384 MiB (402,653,184 bytes), of temporary pinned
memory. Every worker stream is synchronized before ownership is returned.

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
| Six default staging buffers (3 workers x 2) | 402,653,184 | 0.375 | Page-locked, temporary |
| Default post-arena free margin | 8,589,934,592 | 8.000 | Required, not allocated |

The staging buffers are allocated before `cudaMemGetInfo`; the loader then
requires `free >= arena + min_free_bytes_after_load` before calling
`cudaMalloc`. Thus the default gate accounts for all active workers' staging
pressure and fails closed before creating a partial model arena. Callers may
tighten the resource limits; reducing the 8 GiB margin is an explicit policy
decision. Irrespective of the configured chunk and worker limits, aggregate
pinned staging above 2 GiB is rejected before allocation.

## Reproduced official integration

On the target Jetson AGX Orin, using the pinned local model and the default
64 MiB chunk, the original portable-SHA conditional integration baseline
completed successfully:

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

With the default `auto` backend at commit
`06f16c924c47c3d876f42fba872b0172b676161b`, the same fixture selected
`linux_af_alg` and reported:

```text
elapsed                   21.4772 s
full-file bytes    21,921,697,184
H2D text bytes     20,150,569,096
skipped bytes       1,771,128,088
chunks                        328
async scatter operations     2,146
full-file SHA matches          3/3
sha256 backend        linux_af_alg
```

The resident phase of a complete two-token SM87 CLI run was 21.485 seconds,
versus 203.677 seconds in the previous unprofiled SM87 benchmark artifact. This
is a 9.480x diagnostic historical speedup; clocks were not locked, and the two
load numbers were not collected as randomized same-binary trials.

### Parallel shard-loader benchmark

On 2026-07-19, the conditional official-checkpoint integration was run from one
Release binary with a warm filesystem cache. The mirrored order was one worker,
three workers, three workers, one worker:

| Order | Shard workers | Pinned staging | Elapsed seconds |
| ---: | ---: | ---: | ---: |
| 1 | 1 | 128 MiB | 21.7359 |
| 2 | 3 | 384 MiB | 10.9301 |
| 3 | 3 | 384 MiB | 10.9303 |
| 4 | 1 | 128 MiB | 21.6890 |

The one-worker average was 21.71245 seconds and the three-worker average was
10.93020 seconds: a 10.78225-second (49.6593%) reduction, or 1.98646x speedup.
A separate two-worker measurement took 12.6207 seconds, a 41.8734% reduction
and 1.72038x speedup relative to the mirrored one-worker average.

Every measured load selected `linux_af_alg` and reported exactly
21,921,697,184 bytes read, 20,150,569,096 bytes copied, 328 chunks, and 2,146
scatter operations. All three full-file digests and the embedding, layer-0 FP8
QKV, and `lm_head.weight_scale_2` device samples passed on every run. These are
same-binary, non-cold-cache diagnostic measurements; clocks were not locked.
The complete machine-readable record is
[`docs/metadata/qwen36-27b-parallel-loader-benchmark.json`](metadata/qwen36-27b-parallel-loader-benchmark.json).

A subsequent real CLI gate using the default three workers reported 12,227.436
ms total load time, including 11,036.720 ms in the resident phase and 1,159.766
ms in tokenizer setup. It reported 402,653,184 pinned staging bytes. All 19
prompt IDs, 26 generated IDs, decoded text, `im_end` stop, and 44 runner steps
matched the oracle exactly. Prompt, decode, and total generation took 597.976,
3,053.563, and 3,651.540 ms respectively.

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

Set `Q3X_RESIDENT_LOAD_WORKERS=1`, `2`, or `3` to reproduce a specific worker
limit with the same test binary; omitting it exercises the default of 3.

Synthetic tests cover chunk-split tensors, skipped regions, incorrect SHA,
truncation, shard symlinks, unsafe paths, wrapped offsets, stale CUDA last-error
state, statistics, device round-trips, and RAII moves. Parallel cases compare
one-, two-, three-, and excess-worker limits, reversed identity input,
plan-ordered failures and statistics, and the exact global scatter-operation
limit. When AF_ALG is available, the tiny CUDA fixture runs both forced
`portable` and forced `linux_af_alg` and compares their digest/statistics and
cross-chunk device bytes. It also checks that default `auto` selects AF_ALG when
available and otherwise falls back to the portable backend. The CPU plan test
is also run under the host ASan/UBSan configuration.
