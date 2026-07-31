# Prompt-span native GDN vertical-slice plan

Date: 2026-07-31 (Asia/Shanghai)

Design base: `d0352eb5a8df464b9fe3c4356a91e96997b5ef35`

Standalone implementation base: `6eccbbd`

Status: the bounded fused state/output mechanism is implemented and retained
as a standalone component, but runner admission and the 217.5-MiB workspace
expansion are intentionally frozen.  A newer cumulative P2048 NSys profile
invalidated the original GDN time budget: the complete GDN core is now only
274.473 ms, so the old 400-ms request-saving gate is impossible.  Projection
optimization has priority.  This note records the mechanism, its exact
resource/correctness evidence, and the revised stop-loss; it does not change
the production default or make a real-model speed claim.

## Result of the real runner audit

The long-prompt executor is layer-major at projection-span granularity, but
linear-attention recurrent work is still explicitly restarted at every C512
state tile.

`ReferenceRunner::execute_long_prefill_projection_span` first projects the
complete natural span into `long_prefill_projection_primary` and
`long_prefill_projection_secondary`.  It then enters the loop at
`src/runtime/reference_runner.cpp:5707` and calls
`long_prefill_projection_span_state_tile` once per C512/tail tile.  Inside the
native branch at lines 5723--5795, every state tile independently does all of
the following:

1. launch convolution plus compact Q/K preprocessing;
2. call `gdn_prefill_chunk64_native_detail::launch_qk_preprocessed`;
3. pass the persistent layer state as both `state_input` and `state_output`;
4. copy normalized output from the C512 scratch
   `views_.projection[2]` into the whole-span `secondary` buffer at lines
   5899--5907.

The native launcher itself already retains one state owner across the eight
C64 chunks *inside* a C512 tile.  In
`persistent_state_chunk64_vllm_faithful_kernel`, one `(value_head,
value_half)` CTA loads the initial state at lines 2432--2435, loops over all
C64 chunks at lines 2440--2672, and stores the final state in its extra
publication iteration.  It nevertheless publishes a BF16 pre-chunk
`boundary_state` for every C64 chunk at lines 2475--2484 because the separate
chunk-o kernel consumes that state.

Consequently, one exact P2048 request executes four independent native GDN
launchers per linear-attention layer; P3840/P4096 executes eight.  Qwen3.6-27B
has 48 linear-attention layers.

| natural prompt | C512 state tiles/layer | native launchers | native stage kernel instances |
|---:|---:|---:|---:|
| P2048 | 4 | 192 | 1,536 |
| P3840 or P4096 | 8 | 384 | 3,072 |

The last column counts the current default eight-stage sequence per launcher:
fused conv/compact-QK, gate preparation, Gram, solve, W/U recompute,
persistent state, chunk-o BV64, and rows-8 norm/gate.  It does not count the
separate D2D output publication.

The current cumulative real API checkpoint is therefore the correct first
comparison boundary:

- P2048 mean server Prefill: 3,554.425 ms, 576.18 token/s;
- P3840 direction: 7,236.57 ms, 530.64 token/s.

These values come from
`docs/analysis/prefill-k128-m128-direction-2026-07-31/README.md`; they are
direction evidence, not a publication-quality distribution.

### New cumulative P2048 kernel profile supersedes the old GDN budget

A later request-level P2048 NSys capture on the current cumulative route
measured the complete GDN core at **274.473 ms**.  Its top-level attribution
is:

| GDN stage | P2048 total |
|---|---:|
| chunk-o BV64 | 69.581 ms |
| persistent state | 59.402 ms |
| convolution | 39.599 ms |
| W/U recompute | 37.804 ms |
| solve | 32.444 ms |
| rows-8 norm/gate | 26.613 ms |
| compact Gram | 6.820 ms |
| gate preparation | 2.211 ms |
| **complete GDN core** | **274.473 ms** |

The two stages targeted by the vertical fusion total only **128.983 ms**.
That number is an absolute elimination ceiling, not an achievable saving:
the fused kernel must still execute both algebraic stages and now performs an
extra shared-state publication plus a shared Vnew copy to keep register
lifetimes spill-free.  Consequently the earlier `>=400 ms` request gate and
the claim that this slice could supply such a saving are withdrawn.  No
runner integration or large workspace allocation is justified by this
profile alone.

## Why a token-limit extension is rejected before implementation

The active implementation in
`gdn_prefill_chunk64_cublas_reference_sm87.cu` fixes workspace capacity at
C512.  Its production path still inherits a legacy diagnostic layout with
five full value-head token tensors, two C64 matrices, another three full
token tensors, all per-chunk boundary states, FP32 KKT scratch, gamma, and
beta.  The exact capacity from `required_workspace_bytes()` is:

| capacity | existing legacy layout | compact live regions if merely resized |
|---:|---:|---:|
| C512 | 72.1875 MiB | 45.1875 MiB |
| C4096 | 577.5 MiB | 361.5 MiB |

At C4096, `boundary_state` alone is 96 MiB.  Merely lifting these constants
would retain all per-C64 boundary publications and all `v_new` publications;
it would save only three or seven cross-C512 state reload/store pairs and
host launches.  The repository has already rejected a materially similar
state-lifetime-only direction: the archived P513 persistent-span experiment
was 0.999909x end to end and its directly aggregated candidate kernel was
4.314 ms slower per request.  See
`docs/metadata/qwen36-27b-prefill-gdn-persistent-span-direction-rejection-2026-07-29.json`.

Thus a 577.5-MiB launcher extension cannot be the first slice.  It has no
credible route to the required P2048 saving and would add a large resident
allocation before proving value.

## Selected vertical slice: C64/WY plus fused state-and-output owner

The selected slice keeps the validated C64/WY arithmetic and separates the
pipeline into bounded stages.  It does **not** build one giant GDN kernel.

```text
raw QKV + initial conv history
  -> prompt-span conv + compact Q/K + compact V       (kernel 1)
  -> gamma/beta preparation                           (kernel 2)
  -> compact Gram -> solve -> W/U recompute           (kernels 3..5)
  -> persistent state + exact BV64 chunk-o            (kernel 6)
  -> rows-8 RMSNorm + SiLU(Z), in place to secondary  (kernel 7)
  -> whole-span output projection
```

Kernel 6 is the structural change.  The current state kernel and chunk-o
already use the same 128-thread ownership: one CTA per
`(value_head, value_half)`.  The new kernel must:

1. keep the current state fragments live across all C64 chunks in one natural
   projection span;
2. stage the exact BF16 pre-chunk state boundary in shared memory, as the
   current state kernel already does;
3. compute the existing W@H correction and round `v_new` to the identical
   BF16 boundary, but retain it in shared memory rather than publishing it;
4. reuse the exact inline-PTX/lane mapping from `chunk_o_bv64_kernel` to
   compute Q@H, causal rounded QK, and QK@Vnew before updating H;
5. publish only the final BF16 raw output for the chunk and, after the final
   chunk, the persistent state.

This is an adjacent-stage fusion, not a whole-GDN kernel: convolution,
compact-QK, gate preparation, all three WY stages, and RMSNorm/SiLU remain
independent.  The state-and-output kernel must keep at least two CTAs/SM,
have zero local/stack bytes, and retain the existing 128-thread
`(head,half)` ownership.

### Scheduler-boundary numerical contract

Removing the host C512 boundary also removes its BF16 final-state
store/reload.  That would silently change the incumbent numerical contract.
The first slice must therefore round and reload the register state through
the existing shared BF16 representation after every eighth C64 chunk.  This
keeps the CTA owner resident while making one C2048 launch bitwise equivalent
to four ordered incumbent C512 launches.  Removing this artificial rounding
is a later capability experiment, not part of the first performance slice.

The selector must reject a natural span whose final C512 remainder is 1--31,
because the incumbent sends that tail to the exact scalar/C16 route rather
than native C64.  Initial admission is therefore:

```text
512 < M <= 4096
M % 512 == 0 || M % 512 >= 32
native C64 enabled
whole-M projection-span executor selected
SM87 weight-only backend
trace and Q3X_DISABLE_OPTIMIZED_PREFILL both off
prompt-span workspace capacity valid
```

All misses continue through the unchanged C512 loop.

## Compact prompt-span workspace

The prompt-span path needs a new compact partition; it must not resize the
legacy partition.  At the maximum C4096 projection span, the exact regions
are:

| region | layout | C4096 bytes |
|---|---|---:|
| compact Q | `[64,16,64,128]` BF16 | 16 MiB |
| compact K | `[64,16,64,128]` BF16 | 16 MiB |
| convolved V | `[4096,48,128]` BF16 | 48 MiB |
| transform | `[64,48,64,64]` BF16 | 24 MiB |
| W | `[64,48,64,128]` BF16 | 48 MiB |
| U | `[64,48,64,128]` BF16 | 48 MiB |
| compact raw Gram | `[64,16,64,64]` FP32 | 16 MiB |
| gamma + beta | two `[64,48,64]` FP32 tensors | 1.5 MiB |
| **total** | | **217.5 MiB** |

No prompt-span `boundary_state`, `v_new`, packed Q/K replicas, QK score
matrix, diagnostic K tensors, or raw-output tensor is allocated.  Kernel 6
writes raw BF16 output into the now-dead leading `[M,6144]` portion of
`long_prefill_projection_primary`.  The rows-8 kernel reads that raw tensor
and Z from `secondary`, then overwrites `secondary` in place with normalized
gated output.  The following output projection already consumes `secondary`.

The current 72.1875-MiB pointer can remain a union allocation: when the
prompt-span selector is present, allocate `max(legacy_bytes,
prompt_span_bytes(capacity))`; legacy misses partition only their established
prefix.  The default environment continues to allocate exactly the current
legacy size.  No new persistent request-state region is required.

## Concrete implementation map (frozen after the standalone mechanism)

1. `src/kernels/reference/gdn_prefill_whole_span_conv_sm87.{h,cu}`
   - add a prompt-span launcher that writes compact Q/K and contiguous
     convolved V without materializing convolved Q/K;
   - cap it at 4096 and preserve current C8 token-parallel arithmetic and
     history update.
2. `src/kernels/sm87/gdn_prefill_wy_vllm_layout_sm87.{h,cu}`
   - add a separate `launch_packless_compact_v` entry accepting up to 64
     chunks and contiguous V;
   - leave the current <=8-chunk ABI and default untouched.
3. `src/kernels/sm87/gdn_prefill_prompt_span_state_o_sm87.{h,cu}`
   - implement the fused state-and-BV64 owner described above;
   - expose resource and graph-test handles, not a public installed ABI.
4. `src/kernels/reference/gdn_prefill_chunk64_cublas_reference_sm87.cu`
   - add `prompt_span_workspace_bytes(max_tokens)`, a compact partition, and
     a seven-stage `launch_prompt_span_qk_preprocessed` wrapper;
   - do not alter `workspace_bytes()` or the existing launcher.
5. `src/runtime/reference_runner.cpp`
   - parse `Q3X_RUN_GDN_PROMPT_SPAN_NATIVE_ADMISSION=1` next to the native
     selector;
   - add a pure selector next to
     `use_prefill_gdn_chunk64_native_admission`;
   - insert the prompt-span branch immediately before the C512 loop at line
     5707; a hit launches once and skips the loop;
   - allocate the union workspace at factory lines 7006--7018 only when the
     explicit selector and a projection-span capacity above C512 are present;
   - retain distinct prompt-span and virtual-C512 counters so route
     accounting cannot confuse one span hit with one old tile hit.
6. `include/q3x/runtime/reference_runner.h`
   - no public method is added; only private workspace capacity/accounting is
     extended if required.

Production default, Decode, MTP, and cuBLASLt remain unchanged.

Only item 3's fused state/output ownership and the pure host workspace/shape
contract were completed in this bounded round.  Items 1, 2, and 4--6 are
deferred; in particular, the runtime selector and 217.5-MiB allocation were
not added.  Reopening them requires projection work to finish and the revised
real-path gate below to remain economically relevant.

## Correctness gate before any model timing

Synthetic data is permitted only here.

1. Compare candidate C1024 and C2048 with two/four ordered existing C512
   native calls.  Convolved V, compact Q/K, W/U, raw output, conv history,
   and final state must be bitwise equal.
2. Cover C1024, C2048, C3840, C3987, and C4096 plus rejection of C513 and a
   remainder in 1--31.  Padded C64 lanes must remain identity transitions.
3. Test in-place/disjoint state, immutable inputs, finite/nonfinite values,
   guards, two Graph replays, and invalid calls producing zero Graph nodes.
4. Resource gate for kernel 6: 128 threads, no local/stack, at least two
   CTAs/SM.  A miss rejects the implementation before runner integration.
5. Host selector/plan tests must prove default-off behavior and exact hit
   counts: P2048 changes 192 native tile hits into 48 prompt-span hits;
   P3840/P4096 changes 384 into 48.

Every GPU test must take `/tmp/q3x-gpu-bench.lock` with `flock`.  It must not
start a model server or run while an EvalScope/server process owns the lock.

### Standalone mechanism evidence

The first fused C64 fixture compares the incumbent faithful-state plus
separate BV64 output sequence against the fused 96-CTA mechanism.  It checks
that the fixture produces nonzero raw output and changes recurrent state, then
requires complete final-state, raw-output, and normalized-output tensors to
be bitwise equal.  Offline cubin/SASS audit after shortening the product and
QH/QK lifetimes reports:

```text
threads/CTA: 128
CTA count: 96 (48 value heads x 2 value halves)
registers/thread: 248
stack bytes/thread: 0
local bytes/thread: 0
LDL/STL instructions: 0
active blocks/SM: 2
```

The shared BF16 round/reload after every eighth C64 remains in the kernel.
Long-span C1024/C2048 and real-model timing are deliberately not claimed by
the standalone C64 fixture.

## Revised real-path stop-loss and priority

Projection optimization now precedes any prompt-span GDN runner work.  If the
standalone mechanism is revisited, first complete its C1024/C2048 boundary
fixture without allocating runner workspace.  Only then may one same-ELF
real-model OpenAI generation comparison add a temporary admission route.  The
first request remains the frozen natural P2048 token array, batch one,
temperature zero, one generated token.

The revised first direction gate is a request-level Prefill saving of **at
least 50 ms**, with identical generated token/text and usage, exactly 48
prompt-span hits, zero incumbent native-C512 hits for the linear layers, and
no allocation failure/fallback.  Fifty milliseconds is already 38.8% of the
128.983-ms absolute state-plus-chunk-o elimination ceiling.  A smaller saving
does not justify a 217.5-MiB opt-in workspace or further runtime complexity
and the standalone mechanism remains archived.  A negative first request
stops immediately; NSys/NCU is permitted only for attribution.

Only a passing P2048 direction may proceed to natural P4K and repeated
statistics.  No EvalScope matrix is justified before those two simple real
requests pass.
