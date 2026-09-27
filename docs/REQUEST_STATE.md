---
q3x_document:
  id: q3x-request-state
  class: contract
  status: active
  owner: runtime-maintainers
  authority: per-request state, workspace, memory-plan, and lifecycle ownership contract
  effective: 2026-08-09
  last_reviewed: 2026-09-28
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
reallocate the state. Package ABI 0.5.0 adds an explicit candidate-only
layer-major profile and changes the public C++ `RequestState`/plan object
layout; consumers must rebuild against the exact package version, as already
required by the 0.x package policy.

Package ABI 0.6.0 also extends the public Engine load receipt for the sealed
Decode layouts. It does not change the request-state geometry described here;
0.x consumers must still rebuild against the exact installed package.

Package ABI 0.7.0 adds the public reset mode, plan, receipt, diagnostic, and
planning surface used by lifecycle-derived ordinary-request cleanup. It also
changes related public runner, generation, and protocol witness object layouts.
This is an intentional 0.x ABI break; consumers must rebuild and request the
exact installed package version; the 0.8.0 extension below supersedes the
0.7.0 consumer build requirement.

Package ABI 0.8.0 exports the corrected service's compiled whole-core geometry
alongside its public planners. The full-range profile owns 44,095 sequence
positions and 44,096 rows of aligned family scratch. Installed consumers must
rebuild against the current exact 0.8.1 package; the exported compile definitions are part of this exact
package configuration. The legacy explicit planner and reset contracts remain
available under their own memory profiles. Version 0.8.1 preserves this
geometry and numerical state; its changes are at the service reliability and
packaging boundary.

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
- FP32 scratch with capacity `max(262144, 24 * max_seq)` elements; and
- a logical `[24,max_seq]` GQA probability view that aliases that FP32 scratch.

The FP32/GQA pair is the only documented alias. Every owning region begins at
a deterministic 256-byte-aligned offset and otherwise does not overlap. These
sizes are capacity and ownership guarantees, not a required execution tile or
kernel schedule.

### Isolated layer-major profile

`build_layer_major_request_memory_plan` and
`create_layer_major_request_state` are separate explicit entry points. They
have no implicit sequence default and cannot be selected through the legacy
planner or creator. The fixed first-integration profile owns:

- one prompt-wide dense BF16 `[max_seq,5120]` residual;
- one 855,638,016-byte C8192 phase arena shared sequentially by GDN,
  Attention, and MLP typed views;
- one physically disjoint complete legacy C512 workspace;
- one independent BF16 `[1,5120]` final-hidden handoff; and
- the same persistent Conv/GDN/KV and RoPE state as the legacy profile.

The C8192 arena has no public untyped device view. Its named matrices are
phase views, and overlapping storage is legal only across ordered lifetimes.
In particular, each dense `[8192,5120]` normalized input occupies the future
output span before its input projections consume it; it is not a row-prefix
alias of the later wider-stride matrix. Attention core output, final branch
output, and O-projection temporary are simultaneously disjoint. The MLP
Gate/Up temporary supports only serialized projections or one fused-pair
launch; the legacy dual-stream schedule is not bound to this profile.

Allocation does not make the profile executable. Prompt-residual in-place,
family completion, intra-family phase, token-ID consumption, projection
workspace, and operator-binding conditions all default false. Flat legacy
workspace accessors reject the layer-major profile; its retained disjoint C512
workspace is reachable only through the explicit typed bundle. Persistent,
KV, RoPE, position, reset, and ownership operations remain common.

The corrected production service and its separately compiled admission bound
actual prompt lengths to 1..44095 while reserving 44,095 sequence positions and a
9,508,218,624-byte arena. It supports the P40000/O4096 capacity boundary
without advancing state for the final predicted token. Family storage rounds to 44,096 rows while residual and persistent KV own
44,095 positions. Alias checks use each allocation's actual extent; the spare
family row is not a token and never advances state. The full maximum-length
Decode scratch occupies this arena; its FP32 payload is rounded to the
256-byte owning alignment. Cos/sin remain contiguous halves of one allocation.
The original fixed production/development geometry remains 40,016 positions
and 8,641,684,992 bytes. Neither geometry alone qualifies general prompt
lengths, successful long-output execution, or whole-process fit.

### Whole-core projection scratch lifetime

The corrected whole-core composition supplies FP8 and Gate/Up CUTLASS
projection scratch from the request arena. It does not use the historical
process-global growable dequantization buffers. In the full-range family layout,
the GDN workspace occupies bytes `[2356490240,5443915776)` relative to the
family arena and has capacity 3,087,425,536 bytes. This range is reused only
on the runner's ordered stream:

- fill projections finish before GDN starts consuming its workspace;
- drain projections begin after GDN's last workspace consumer;
- MLP Gate/Up begins after the Attention/GDN family completes and finishes
  before Down; its activated output is outside this scratch range.

The first aligned `prompt_capacity*4` bytes retain token IDs until all fill
panels have embedded them. FP8 scratch starts strictly after this live prefix
and uses at most `N*K*2` bytes. Gate/Up uses `2*N*K*2` bytes for decoded
weights followed by `8000*2*N*2` bytes for panel outputs. Both reject insufficient
or misaligned storage. Their selected CUTLASS configuration needs no separate
GEMM workspace; a nonzero requirement fails closed. Scratch never overlaps
live projection inputs, outputs, residual, or persistent state, and no CUDA
allocation or free occurs in these owned projection entries. The old entry
points remain for separately identified historical routes only. This lifetime
change does not extend prompt admission by itself. The bounded
[full-handoff comparison](metadata/qwen36-27b-prefill-owned-scratch-2026-09-28.json)
checks the unchanged arithmetic against the corrected predecessor.

### Actual-row MLP tail

Whole-core Down executes complete M64 tiles in place. For a final 1..63 rows,
it reuses dead GDN workspace for one zero-filled `[64,17408]` input tile and
one `[64,5120]` residual tile. It copies only actual rows into those buffers,
runs the same Down kernel at M64, and copies only actual result rows back.
Every operation uses the same ordered stream, after Gate/Up's last consumer.
No padded row is embedded or reaches Attention, recurrent state, KV, logits,
position, or usage. The scratch requirement is 2,883,584 bytes and no allocation
occurs. Complete input rows preserve the existing K reduction and BF16
projection/residual publication. Plan and witness projection-entry counts
include the second Down grid when both an aligned prefix and a tail exist;
they are not a count of all dequantization, copy, or CUDA kernel launches.

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
88,087,040 arena bytes. The largest public plan, sequence 262,144 with C512
workspace, is 17,437,720,576 bytes and therefore requires an explicit arena
limit above the default 2 GiB. These are deterministic planner outputs, not
claims that a current product configuration admits those capacities; see
[`CURRENT_STATUS.md`](CURRENT_STATUS.md) for that distinction.

The FP32 floor and default total above correct stale documentation to the
existing planner, whose unconditional floor became 262,144 elements in
`cac6b515`. This is not an allocation, ABI, or runtime change. Historical
evidence retains its original source-specific values.

For the fixed layer-major strategy, exact 40K/60K/130K request-arena totals
are 4,066,344,960, 5,588,904,960, and 10,917,864,960 bytes. These are isolated
allocation-profile totals, not whole-process fit or production-admission
claims. Resident model/sidecar bytes and measured whole-process peaks remain
separate requirements.

Before allocation, `create_request_state` checks the planned arena against
`max_arena_bytes`, queries free device memory, and requires
`free >= arena + min_free_bytes_after_create`. The default retained-free
margin is 8 GiB. There is no managed-memory, swap, partial-allocation, or
smaller silent fallback. A failed creation releases any partial ownership.

## Position, reset, and ownership lifecycle

Logical sequence length starts at zero. `commit_token()` increments it only
within capacity; `set_sequence_length()` rejects an excessive value. Current
RoPE lookup fails when the logical position reaches capacity.

Whole-request prefill publishes host-visible progress through
`publish_sequence_length(expected_current, desired)`. This is a host-only,
`noexcept` conditional update: an empty state, a stale `expected_current`, a
`desired` value beyond request capacity, or a regression below the matched
current length is rejected before mutation. On success, the logical length is
assigned exactly once; on every failure it remains unchanged. Equality is an
allowed idempotent publication. The executor must capture the initial length,
finish all layer/panel device work, and publish the final length once at the
whole-request commit boundary--never once per panel or layer.

This transactional API does not change the existing legacy semantics of
`commit_token()` or `set_sequence_length()`; in particular, the bounded setter
remains available to reset or restore logical length where its callers already
own that lifecycle. Conditional publication is supported identically by the
legacy C512 and layer-major C8192 memory profiles and never launches CUDA work.

The public `reset_async(stream)` remains the conservative recovery operation:
it enqueues one reset of the complete persistent Conv, GDN, K, and V span and
sets host logical length to zero after successful enqueue. Workspace and
immutable RoPE tables are not reset. Subsequent use must be ordered on the
same stream or explicitly synchronized by the caller.

The ordinary production runner may use the private request-reuse boundary to
select less work without weakening that zero-state contract. A state whose
creation or prior full reset is known complete enqueues no reset. After one
successfully accepted request, the next request clears the complete contiguous
Conv/GDN region plus exactly `committed_positions` rows from each of the 16
separate K and V caches. The byte receipt is therefore
`78,446,592 + 65,536 * committed_positions`, submitted as one fixed-state
memset plus 32 cache-prefix memsets. Zero or out-of-capacity positions, a
noncanonical persistent layout, a host-length mismatch, or any lifecycle
uncertainty rejects the prefix plan. The runner then uses the existing
one-span full reset. No caller, environment variable, CLI flag, or test
selector chooses a production reset mode.

Move construction and assignment transfer sole arena ownership and leave the
source empty. Views are non-owning and remain valid only while the owning
state remains alive and is not moved or destroyed. CUDA types stay out of the
public header; an optional stream is represented as `void*`.

## Failure semantics

Planning fails closed for invalid options, arithmetic overflow, invalid layer
schedules, and arena-limit violations. Creation additionally distinguishes
insufficient device memory, CUDA failure, and allocation failure. View and
state operations distinguish layer, type, slot, position, capacity, buffer
index, memory-profile mismatch, and empty-state errors. Valid boundaries clear
unrelated stale CUDA last-error state before reporting their own CUDA result.

## Verification boundary

Host tests cover the exact schedule, region sizes, alignment, non-overlap,
capacity limits, arithmetic overflow, bad options, all target-bucket totals,
typed temporal aliases, profile gates, and exact clean/prefix/full reset-plan
bytes. CUDA tests cover real allocation,
initialization, view ranges, BF16-rounded RoPE values, lifecycle, reset
ordering, memory gating, and move ownership. The approximately 1 GiB minimum
layer-major allocation case is opt-in and may skip explicitly when its
free-memory reservation cannot be met. Performance measurements
and historical plan examples are retained in
[`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md) and the
[`metadata/`](metadata/) evidence index; they do not amend this contract.
