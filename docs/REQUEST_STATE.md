# Qwen3.6-27B batch-one request state

`RequestState` is the bounded per-request device memory contract for the first
Qwen3x-Orin decoder. It is separate from resident model weights and owns one
move-only CUDA arena. All state, KV cache, decode workspace, and RoPE lookup
tables are planned before allocation; token execution performs no allocation.

The public API is `include/q3x/runtime/request_state.h`. CUDA types remain out
of the header: device storage is represented by non-owning buffer views and an
optional stream is passed as `void*`.

## Fixed layer and persistent-state ABI

The planner encodes the exact 64-layer hybrid schedule. Layers 3, 7, ..., 63
are the 16 full-attention layers; the other 48 layers are linear attention.
Every layer-to-slot lookup requires the caller's expected layer type and rejects
out-of-range or mismatched access.

Persistent BF16 storage is canonical and fully zeroed for a new request:

| Region | Logical layout | Bytes |
| --- | --- | ---: |
| Conv history | `[48, 10240, 3]` | 2,949,120 |
| Gated DeltaNet state | `[48, 48, 128, 128]` | 75,497,472 |
| K cache per full layer | `[max_seq, 4, 256]` | `2048 * max_seq` |
| V cache per full layer | `[max_seq, 4, 256]` | `2048 * max_seq` |
| All K/V cache | 16 independent K/V pairs | `65,536 * max_seq` |

K and V are separate logical views per full layer. Per-position views expose
exactly `[4,256]` BF16 elements and reject `position >= max_seq`.

## Reusable decode workspace

The same request arena contains activation workspace for the configured
`prefill_chunk_size` `C` (1 through 32):

- three independent `[C,5120]` BF16 hidden/residual buffers;
- four independent `[C,17408]` BF16 projection buffers (`P3` can hold the
  full-attention tile's contiguous `q[C,6144] + gate[C,6144]` payload);
- independent BF16 `a[C,48]` and `b[C,48]` buffers, preventing the linear branch
  from overwriting values still consumed after `P0=qkv`, `P1=z`, and
  `P2=GDN output` are assigned;
- one FP32 GEMV/logits buffer with capacity
  `max(248320, 24 * max_seq)`;
- an explicit `[24,max_seq]` GQA probability view that aliases the same FP32
  allocation.

Aliasing is restricted to the documented FP32/GQA pair. Every owning region
starts at a deterministic 256-byte-aligned offset and otherwise does not
overlap.

## RoPE cache policy

The cache holds separate cosine and sine tables with logical shape
`[max_seq,32]`. It uses `theta=10,000,000` and `rotary_dim=64`:

```text
inv_freq[i] = theta ^ (-2*i/64), i in [0,31]
angle[p,i]  = p * inv_freq[i]
```

To match the independent vLLM oracle, cosine and sine are first generated in
FP32, rounded to BF16 with round-to-nearest-even, and then expanded back into
FP32 device storage. This preserves the current decode-op `float*` ABI while
matching the numerical values of vLLM's BF16 RoPE cache. Position zero is exact
`cos=1`, `sin=0`.

Creation copies both tables asynchronously on an internal stream and
synchronizes all initialization before returning.

## Exact memory plans

The default sequence capacity is 128 tokens and the default chunk is C1:

| Default region | Bytes |
| --- | ---: |
| Conv + GDN + all K/V persistent storage | 86,835,200 |
| Three hidden buffers | 30,720 |
| Four projection buffers | 139,264 |
| Aligned independent `a`/`b` allocations | 512 |
| FP32 GEMV/logits/GQA scratch | 993,280 |
| Total workspace | 1,163,776 |
| RoPE cosine + sine | 32,768 |
| **Single request arena** | **88,031,744** |

Selecting C8 changes only activation workspace; persistent state, FP32/GQA
scratch, and RoPE storage stay identical:

| Default-128 C8 region | Bytes |
| --- | ---: |
| Conv + GDN + all K/V persistent storage | 86,835,200 |
| C8 hidden/projection/`a`/`b` plus FP32 scratch | 2,354,688 |
| RoPE cosine + sine | 32,768 |
| **Single request arena** | **89,222,656** |

The exact C8-minus-C1 increment is 1,190,912 bytes. For the 64-position
end-to-end benchmark plan, the corresponding arenas are 83,821,056 bytes at
C1 and 85,011,968 bytes at C8.

Selecting C16 follows the same workspace-only growth rule:

| Default-128 C16 region | Bytes |
| --- | ---: |
| Conv + GDN + all K/V persistent storage | 86,835,200 |
| C16 hidden/projection/`a`/`b` plus FP32 scratch | 3,716,096 |
| RoPE cosine + sine | 32,768 |
| **Single request arena** | **90,584,064** |

The exact C16-minus-C1 increment is 2,552,320 bytes. At the 64-position
end-to-end capacity, C16 uses 86,373,376 bytes, or 1,361,408 bytes more than
C8. Persistent state, FP32/GQA scratch capacity, and RoPE capacity remain
unchanged at every chunk size.

Selecting C32 again changes only activation workspace:

| Default-128 C32 region | Bytes |
| --- | ---: |
| Conv + GDN + all K/V persistent storage | 86,835,200 |
| C32 hidden/projection/`a`/`b` plus FP32 scratch | 6,438,912 |
| RoPE cosine + sine | 32,768 |
| **Single request arena** | **93,306,880** |

The exact C32-minus-C16 increment is 2,722,816 bytes. At the fixed 44-position
full-model oracle capacity, C32 uses 87,780,352 bytes. KV/state and RoPE
capacity continue to depend only on sequence length, not on chunk size.

At the absolute supported capacity of 262,144 tokens, the caller must
explicitly raise `max_arena_bytes`:

| Maximum region | Bytes |
| --- | ---: |
| Persistent storage | 17,258,315,776 |
| C32 workspace (`24*max_seq` dominates FP32 scratch) | 30,611,456 |
| RoPE cosine + sine | 67,108,864 |
| **Single request arena** | **17,356,036,096** |

The planner computes these values with checked `uint64_t` arithmetic before
CUDA is touched. It rejects batch sizes other than one, chunk sizes outside
1 through 32, zero sequence capacity,
capacities over 262,144, malicious arithmetic overflow, invalid resource
limits, and plans larger than `max_arena_bytes`.

`create_request_state` calls `cudaMemGetInfo` and requires
`free >= arena + min_free_bytes_after_create` before `cudaMalloc`. The default
post-create free-memory margin is 8 GiB. It does not use managed memory, swap,
or a partially allocated fallback.

## Position and reset lifecycle

The host logical length starts at zero. `commit_token()` and
`set_sequence_length()` are capacity checked and allocate nothing. A current
RoPE lookup fails once the logical position reaches capacity.

`reset_async(stream)` performs one asynchronous memset over the contiguous
persistent span, including all Conv, GDN, K, and V storage. It sets host logical
length to zero immediately after a successful enqueue. Workspace and immutable
RoPE tables are deliberately untouched. Subsequent state use must be ordered
on the same stream, or the caller must synchronize explicitly.

The implementation clears unrelated stale CUDA last-error state at create and
reset boundaries. Any create failure releases the partial arena; move
construction and assignment transfer sole ownership and leave the source
empty.

## Verification

`request_state_plan` checks exact C1, C8, C16, and C32 byte totals, the C8/C16/C32
workspace-only deltas, default/minimum/maximum sequence totals, overflow and
bad options, schedule counts, slot mappings, alignment, and non-overlap.
`request_state_cuda` uses a small four-token capacity and verifies true device
allocation, zero samples, pointer ranges, view capacities, BF16-rounded RoPE
values, stale-error isolation, logical positions, asynchronous reset behavior,
workspace preservation, memory gating, and RAII moves. It returns skip code 77
when no CUDA device is present and never allocates the 262,144-token plan.
