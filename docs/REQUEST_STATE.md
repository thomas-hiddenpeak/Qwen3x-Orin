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

The same request arena contains:

- three independent `[5120]` BF16 hidden/residual buffers;
- four independent `[17408]` BF16 projection buffers (`P3` can hold contiguous
  full-attention `q[6144] + gate[6144]`);
- independent BF16 `a[48]` and `b[48]` buffers, preventing the linear branch
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

The default capacity is 128 tokens:

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

At the absolute supported capacity of 262,144 tokens, the caller must
explicitly raise `max_arena_bytes`:

| Maximum region | Bytes |
| --- | ---: |
| Persistent storage | 17,258,315,776 |
| Workspace (`24*max_seq` dominates FP32 scratch) | 25,336,320 |
| RoPE cosine + sine | 67,108,864 |
| **Single request arena** | **17,350,760,960** |

The planner computes these values with checked `uint64_t` arithmetic before
CUDA is touched. It rejects batch sizes other than one, zero capacity,
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

`request_state_plan` checks exact default/minimum/maximum byte totals, overflow
and bad options, schedule counts, slot mappings, alignment, and non-overlap.
`request_state_cuda` uses a small four-token capacity and verifies true device
allocation, zero samples, pointer ranges, view capacities, BF16-rounded RoPE
values, stale-error isolation, logical positions, asynchronous reset behavior,
workspace preservation, memory gating, and RAII moves. It returns skip code 77
when no CUDA device is present and never allocates the 262,144-token plan.
