---
q3x_document:
  id: q3x-request-state
  class: contract
  status: active
  owner: runtime-maintainers
  authority: per-request state, workspace, memory-plan, and lifecycle ownership contract
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: RequestState persistent state, workspace, RoPE, allocation, and lifecycle behavior
  review_trigger: any RequestState ABI, capacity, layout, allocation, ownership, or lifecycle change
---

# Qwen3.6-27B batch-one request state

> **Authority boundary.** This contract refines the request-state boundary in
> the [system SDD](SDD.md) and is subordinate to it and the
> [engineering constitution](ENGINEERING_CONSTITUTION.md). Current capacity,
> default-route, qualification, and release truth belongs in
> [`CURRENT_STATUS.md`](CURRENT_STATUS.md). Tile selection, buffering,
> stream topology, kernel routing, and performance gates are deliberately not
> part of this contract; they belong only to a named active local work package.

The public API is `include/q3x/runtime/request_state.h`. `RequestState` owns the
bounded, mutable device state of one batch-one request. It is separate from
resident model weights, is move-only, and owns exactly one CUDA arena. The
planner fixes every region before allocation; token execution does not grow or
reallocate the state.

## Model-state contract

The planner represents the pinned 64-layer hybrid schedule. Layers 3, 7, ...,
63 are the 16 full-attention layers; the remaining 48 layers are linear
attention. A layer lookup includes the caller's expected layer type and fails
on an out-of-range index or type mismatch.

Canonical persistent storage is BF16 and is zero-initialized when the state is
created:

| Region | Logical layout | Size rule |
| --- | --- | ---: |
| Conv history | `[48,10240,3]` | 2,949,120 bytes |
| Gated DeltaNet state | `[48,48,128,128]` | 75,497,472 bytes |
| K cache per full layer | `[max_seq,4,256]` | `2048 * max_seq` bytes |
| V cache per full layer | `[max_seq,4,256]` | `2048 * max_seq` bytes |
| All K/V caches | 16 separate K/V pairs | `65,536 * max_seq` bytes |

K and V remain distinct logical views. A per-position K or V view contains
exactly `[4,256]` BF16 elements and rejects `position >= max_seq`.

## Workspace and aliasing contract

For configured `prefill_chunk_size = C`, where `1 <= C <= 512`, the arena
contains:

- three independent BF16 `[C,5120]` hidden/residual buffers;
- four independent BF16 `[C,17408]` projection buffers;
- independent BF16 `[C,48]` linear-attention `a` and `b` buffers;
- FP32 scratch with capacity `max(248320, 24 * max_seq)` elements; and
- a logical `[24,max_seq]` GQA probability view that aliases that FP32 scratch.

The FP32/GQA pair is the only documented alias. Every owning region begins at
a deterministic 256-byte-aligned offset and otherwise does not overlap. These
sizes are capacity and ownership guarantees, not a required execution tile or
kernel schedule.

## RoPE numerical contract

Cosine and sine are separate tables with logical shape `[max_seq,32]`, using
`theta = 10,000,000` and `rotary_dim = 64`:

```text
inv_freq[i] = theta ^ (-2*i/64), i in [0,31]
angle[p,i]  = p * inv_freq[i]
```

Values are generated in FP32, rounded to BF16 round-to-nearest-even, then
expanded back to FP32 storage for the current `float*` consumer ABI. Position
zero is exactly `cos=1` and `sin=0`. Creation does not return until table
initialization and persistent-state zeroing are complete.

## Capacity and allocation

The stable option limits are:

- batch size exactly one;
- sequence capacity from 1 through 262,144;
- Prefill workspace capacity from 1 through 512 tokens;
- checked `uint64_t` planning and 256-byte region alignment; and
- caller-controlled arena and post-create free-memory limits.

The default plan is `max_sequence_length=128`, `prefill_chunk_size=1`, and
88,031,744 arena bytes. The largest public plan, sequence 262,144 with C512
workspace, is 17,437,720,576 bytes and therefore requires an explicit arena
limit above the default 2 GiB. These are deterministic planner outputs, not
claims that a current product configuration admits those capacities; see
[`CURRENT_STATUS.md`](CURRENT_STATUS.md) for that distinction.

Before allocation, `create_request_state` checks the planned arena against
`max_arena_bytes`, queries free device memory, and requires
`free >= arena + min_free_bytes_after_create`. The default retained-free
margin is 8 GiB. There is no managed-memory, swap, partial-allocation, or
smaller silent fallback. A failed creation releases any partial ownership.

## Position, reset, and ownership lifecycle

Logical sequence length starts at zero. `commit_token()` increments it only
within capacity; `set_sequence_length()` rejects an excessive value. Current
RoPE lookup fails when the logical position reaches capacity.

`reset_async(stream)` enqueues one reset of the complete persistent Conv,
GDN, K, and V span and sets host logical length to zero after successful
enqueue. Workspace and immutable RoPE tables are not reset. Subsequent use
must be ordered on the same stream or explicitly synchronized by the caller.

Move construction and assignment transfer sole arena ownership and leave the
source empty. Views are non-owning and remain valid only while the owning
state remains alive and is not moved or destroyed. CUDA types stay out of the
public header; an optional stream is represented as `void*`.

## Failure semantics

Planning fails closed for invalid options, arithmetic overflow, invalid layer
schedules, and arena-limit violations. Creation additionally distinguishes
insufficient device memory, CUDA failure, and allocation failure. View and
state operations distinguish layer, type, slot, position, capacity, buffer
index, and empty-state errors. Valid boundaries clear unrelated stale CUDA
last-error state before reporting their own CUDA result.

## Verification boundary

Host tests cover the exact schedule, region sizes, alignment, non-overlap,
capacity limits, arithmetic overflow, and bad options. CUDA tests cover real
allocation, initialization, view ranges, BF16-rounded RoPE values, lifecycle,
reset ordering, memory gating, and move ownership. Performance measurements
and historical plan examples are retained in
[`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md) and the
[`metadata/`](metadata/) evidence index; they do not amend this contract.
