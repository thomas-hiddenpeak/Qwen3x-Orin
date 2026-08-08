---
q3x_document:
  id: q3x-resident-weight-loader
  class: contract
  status: active
  owner: runtime-maintainers
  authority: authenticated resident-weight identity, I/O, allocation, ownership, and failure contract
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: resident checkpoint authentication, load planning, arena ownership, and loader failure behavior
  review_trigger: any checkpoint identity, tensor set, load plan, allocation, ownership, or loader-contract change
---

# Authenticated resident-weight loader contract

> **Authority boundary.** This contract refines the resident-asset boundary in
> the [system SDD](SDD.md) and is subordinate to it and the
> [engineering constitution](ENGINEERING_CONSTITUTION.md). Current capacity,
> default-route, qualification, performance, and Production lifecycle truth
> belongs in [`CURRENT_STATUS.md`](CURRENT_STATUS.md). Read overlap, staging
> topology, prepacking, residency experiments, and their thresholds are not
> global loader rules; they belong only to a named active local work package.

The public API is `include/q3x/runtime/resident_weights.h`. The pinned loader
authenticates every byte of the exact three-shard
`nvidia/Qwen3.6-27B-NVFP4` checkpoint while placing only its 1,846 text tensor
payloads into one device arena. Vision and MTP payloads are authenticated but
are not resident runtime tensors.

This boundary is lossless: it does not dequantize, requantize, mutate, or
silently repack checkpoint tensors, and it has no managed-memory, swap, or
oversubscription fallback.

## Immutable checkpoint identity

| Shard | Exact bytes | SHA-256 |
| --- | ---: | --- |
| `model-00001-of-00003.safetensors` | 9,965,652,512 | `b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d` |
| `model-00002-of-00003.safetensors` | 9,985,757,032 | `06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d` |
| `model-00003-of-00003.safetensors` | 1,970,287,640 | `e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845` |

`load_pinned_qwen36_27b` rebuilds the strict manifest and requires this full
identity before ownership is returned. Authentication and copying share the
same complete-file read; bytes excluded from the text arena are still hashed.

## Planning and resident layout

`build_resident_load_plan(manifest, identities)` is CPU-only. It validates
identities and locator ranges, rejects duplicate/overlapping ranges and
arithmetic overflow, sorts tensors by `(shard, source offset, tensor name)`,
and assigns each text tensor a deterministic 256-byte-aligned arena offset.
It performs no file I/O or CUDA call.

The pinned plan has these identity-derived quantities:

| Quantity | Bytes |
| --- | ---: |
| Complete shard files read and authenticated | 21,921,697,184 |
| Raw text tensor payload copied | 20,150,569,096 |
| Aligned resident arena | 20,150,786,560 |
| Vision/MTP plus non-payload bytes excluded from H2D | 1,771,128,088 |

`ResidentWeights` is move-only RAII ownership of exactly one `cudaMalloc`
arena plus its immutable name-to-view lookup table. The original tensor name
is the key supplied to `ResidentWeights::find(name)`; it is not a field inside
`DeviceTensorView`. The returned view exposes arena offset, byte count, dtype,
shape, and a non-owning device pointer. Destruction or move-assignment releases
the arena and invalidates all derived views; a move transfers the allocation
without relocating it.

The lower-level `load_resident_weights` accepts an explicit manifest and
identity table for synthetic tests and future pinned descriptors. It does not
weaken plan, identity, path, allocation, or failure checks.

## File-system and authentication safety

The checkpoint root is opened as a directory. Each relative component is
opened with `openat`, `O_NOFOLLOW`, and `O_CLOEXEC`; intermediate components
must be directories and the final component must be a regular file of the
exact pinned size. Unsafe paths, symlinks, FIFOs, truncation, growth, and
check-then-reopen races fail closed.

Each file descriptor is consumed sequentially from byte zero through EOF. A
final EOF read and `fstat` detect mutation during loading. Shards may be
processed concurrently within the caller's explicit worker/resource limits,
but each shard's digest and statistics remain deterministic and failures are
reported in load-plan order rather than scheduler order.

`ResidentSha256Backend::kAuto` prefers Linux AF_ALG and may fall back to the
portable implementation only if AF_ALG cannot be initialized before any shard
bytes are consumed. A streaming or finalization error never changes backend;
a successful load uses one concrete backend for every shard and reports it.

## Allocation and resource limits

Public options have stable defaults and defensive validation bounds:

| Option/resource | Default | Accepted bound/policy |
| --- | ---: | ---: |
| chunk size | 64 MiB | 256 MiB |
| parallel shard workers | 3 | 16 |
| arena bytes | 32 GiB | 64 GiB |
| tensor count | 10,000 | 1,000,000 |
| shard count | 16 | 1,024 |
| scatter/memcpy operations | 1,000,000 | 10,000,000 |
| aggregate page-locked staging | derived | 2 GiB |
| free bytes retained after load | 8 GiB | no independent fixed ceiling; must satisfy arithmetic and device-memory gates |

Each active shard worker owns two page-locked staging buffers of the configured
chunk size. The effective worker count is capped by both the option and the
authenticated shard count. All option bounds and aggregate staging are checked
before their corresponding allocation or load work. A caller may tighten a
margin upward or explicitly reduce it; either choice is invocation policy, not
a performance or qualification claim.

Before allocating the resident arena, the loader requires:

```text
device_free >= planned_arena + min_free_bytes_after_load
```

The default retained-free margin is 8 GiB and the pinned arena is
20,150,786,560 bytes. Failure creates no partially usable owner. Reducing a
resource margin is an explicit caller policy; it is not evidence that the
whole runner is admissible or qualified.

## Statistics and failure semantics

Successful load statistics report complete bytes read, bytes copied/skipped,
chunk and copy counts, concrete hash backend, worker/staging totals, device
memory observed before allocation, and deterministic per-shard digests and
counters. Statistics describe the invocation; they are not performance or
lifecycle claims.

Diagnostics distinguish invalid options/manifests/identities, unsafe paths,
duplicate or missing shards, arithmetic overflow, directory/shard open and
regular-file failures, size/I/O/hash failures, CUDA failure, insufficient
device memory, and allocation failure. Any failure closes descriptors,
releases staging and the partial device arena, joins all started workers, and
returns no `ResidentWeights` value.

## Verification boundary

CPU tests cover deterministic planning, identity/range/overflow rules, path
safety, and statistics ordering. CUDA tests cover authenticated scatter,
device-byte identity, backend consistency, resource gates, failure cleanup,
stale CUDA-error isolation, and move ownership. Tests never modify the model
directory.

Reproduction commands and historical loader timings are retained in
[`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md); exact parallel-loader
lineage is the
[`parallel shard-loader record`](metadata/qwen36-27b-parallel-loader-benchmark.json).
It does not define an active loader optimization threshold or current result.
