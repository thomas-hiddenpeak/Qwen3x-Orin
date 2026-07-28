# Real-model performance evidence policy

This policy defines which measurements may select, reject, or promote a model
kernel.  It applies to microbenchmarks, component screens, end-to-end
benchmarks, and Nsight evidence.

## Authority rule

Any result used to continue or stop an optimization, change production
dispatch, set a performance threshold, or publish a model-performance claim
must use a pinned production payload.  Synthetic timing is non-admissible: it
may smoke-test a path or help form a hypothesis, but it may not pass or fail a
performance gate.

Synthetic inputs remain required for exhaustive correctness: all quantized
codes, NaNs and scale edge cases, same-bank stress, guards, aliases, invalid
zero-enqueue behavior, race checks, and CUDA Graph replay.  A synthetic timing
anomaly must be investigated with a broader real-payload panel before it can
affect a performance decision.

Pure device-health calibration, such as launch or memcpy smoke tests, may use
synthetic bytes when it is explicitly labelled `hardware_calibration` and no
model-performance conclusion is drawn from it.

## Test tiers

| Tier | Payload | Purpose | Performance authority |
|---|---|---|---|
| T0 | none | build, ABI, selector, SASS, and static resource contracts | none |
| T1 | synthetic | exhaustive correctness, robustness, Graph, and smoke tests | none |
| T2 | pinned model tensors | same-binary component timing and candidate selection | component authority |
| T3 | pinned model and prompts | natural layer order, route attribution, Prefix/TTFT, tokens, and state | production authority |
| T4 | the exact T2 or T3 payload | NCU/NSys mechanism attribution after timing passes | diagnostic only |

T2 is necessary but not sufficient for production promotion.  Production
selection also requires the applicable T3 gates.

## Required production payload

For weight-dependent kernels, T2--T4 must load the real weight, block scale,
and tensor scale directly from the checkpoint.  NVFP4 evidence includes the
packed weight, E4M3 block scale, and the exact raw bits of `weight_scale_2`;
FP8 evidence includes the weight and its tensor scale.  A generated tensor
with a checkpoint-like distribution is still synthetic.

When values affect control flow, addresses, sparsity, compression, or state
evolution, the activation, KV cache, or recurrent state must also come from a
pinned real-prompt capture at the relevant layer boundary.  Attention, SSM,
GDN, and any data-dependent fast path therefore cannot be promoted from real
weights combined only with arbitrary synthetic state.  A value-independent
dense projection may use a deterministic activation fixture at T2, but the
record must identify the evidence as `checkpoint_weight_only`, and T3 remains
mandatory.

The current model identity is:

- repository: `nvidia/Qwen3.6-27B-NVFP4`;
- revision: `0893e1606ff3d5f97a441f405d5fc541a6bdf404`.

Every evidence record must additionally pin the config, quantization config,
index and shard identities; tensor name, layer, role, dtype, shape, file range
and payload SHA256; and scalar scale bits.  Paths must be canonical, read-only,
non-symlink files inside the model root.  A real activation/state capture must
pin the prompt or token IDs, tokenizer/config revision, capture point, dtype,
shape, and payload SHA256.

The record must also identify the git tree, binary SHA256/build ID, CUDA and
driver versions, device, clocks, nvpmodel, CPU affinity, temperature envelope,
and cache protocol.  A hash or shape mismatch fails closed before enqueue.

## Modes and exit behavior

Performance harnesses must expose separate, unambiguous modes.  Names may be
adapted to an existing executable, but their behavior must match this contract:

- `validate-synthetic` (currently spelled `smoke` by the C512 harnesses): T1
  only; it contains no timing threshold.
- `performance-checkpoint`: T2 or T3; a checkpoint is mandatory and no
  synthetic fallback exists.
- `profile-checkpoint`: T4; it uses the same pinned payload and selected route
  as the preceding timing run.

Exit `0` means that the requested tier completed and its applicable gates
passed.  Exit `1` means invalid evidence, a correctness failure, or a runtime
failure.  Exit `2` means invalid or incomplete command-line configuration.
Exit `3` distinguishes a valid, completed measurement that rejected the
performance candidate.  Exit `77` means that optional hardware or checkpoint
data is unavailable in a non-admission environment.  When strict admission is
enabled, for example with `Q3X_REQUIRE_REAL_PERF=1`, missing model data,
incorrect hashes, unsupported hardware, or unlocked required clocks are
failures, not skips.  A missing checkpoint must never select a synthetic
performance path implicitly.

## Current C512 migration state

The Gate-only cuBLASLt ceiling is synthetic T1 only.  Its legacy absolute
reference and directional speedup are diagnostic fields and cannot retain or
reject a route.  The Gate+Up pair and Down ceiling expose separate `smoke` and
`performance-checkpoint` modes.  Only the latter may emit a performance
decision, requires a checkpoint explicitly, and is registered as a serialized
CTest when `Q3X_REAL_PERF_MODEL_DIR` (or the E2E model directory) is supplied.

These component harnesses currently combine pinned layer-0 weight, block
scale, and tensor scale with a deterministic activation fixture.  Their
evidence class is therefore `checkpoint_weight_only`, not `full_model`.
Their layer-0 results are useful migration controls but remain provisional
until the candidate-independent four-layer panel and T3 full-model gates are
implemented.  No historical synthetic threshold is thereby re-certified.

The production exact-C512 Down bridge no longer times synthetic BF16 operands
while constructing its cuBLASLt context.  It scans the runtime heuristic list
for the zero-workspace configuration locked by the current T2 checkpoint
evidence (`algorithm_id=6`, `tile_id=23`, `split_k=1`, no reduction,
`cta_swizzle=1`, `custom_option=0`, `stages_id=15`) and fails closed when that
configuration is unavailable.  Heuristic-list rank is recorded but is not an
ABI or selection key.

## CI lanes

Ordinary CI requires T0 and T1.  It may report T2--T4 as explicit skips when
the pinned model or SM87 device is unavailable; a skip is not a performance
pass.  Suggested labels are `correctness;synthetic`,
`correctness;checkpoint`, `performance;checkpoint;sm87;admission`, and
`profile;checkpoint`.

The Orin admission lane requires the pinned checkpoint, strict admission,
fixed GPU and EMC clocks, CPU affinity, and serialized ownership of the GPU.
Kernel, selector, pipeline, cache-policy, or threshold changes may not claim a
performance completion without evidence from this lane.  Release promotion
requires both T2 and T3 results.

## Layer panels and cache state

A single convenient layer, including layer 0, is not a general production
sample.  Before viewing a candidate result, each projection or stateful role
must freeze a layer panel chosen from all eligible layers using payload
features such as quantized-code and scale-exponent histograms, zero fraction,
entropy, and compressibility.  The development panel must contain at least
four layers spanning low, median, high, and extreme payloads.  Gate, Up, Down,
FP8 QKV/Z/O/KV, Attention, and SSM use independent panels.

Final promotion must either sweep every eligible layer or exercise all of them
in the natural T3 model path.  Component timing should rotate through real
layers or explicitly scrub cache so that repeated use of one tensor does not
create an accidental warm-cache gate.  If warm-cache behavior is relevant, it
is recorded as a separate steady-state result.

## Threshold recalibration

Synthetic thresholds and absolute timings are not transferred to real
payloads.  Recalibration proceeds as follows:

1. Freeze the live incumbent, payload manifest, layer panel, environment, and
   timing protocol before inspecting the candidate.
2. Run incumbent-versus-incumbent paired controls on the real panel to measure
   the local noise floor.
3. Predeclare a relative continuation or promotion threshold that exceeds
   both the engineering minimum benefit and the measured noise allowance.
   Absolute milliseconds are drift alarms, not the primary admission gate.
4. Use the same final binary and six B-C-C-B rounds for incumbent and
   candidate.  Preserve all raw rounds, including reversals and failures.
5. Require per-layer non-regression and the predeclared aggregate gain.  Final
   promotion repeats the test in independent processes and completes T3.
6. Re-run the gates after the production binary and selector are finalized.

Thresholds are role- and shape-specific.  Gate evidence does not set Down or
FP8 thresholds, and a microbenchmark gain is weighted by the number of actual
model calls before its end-to-end significance is claimed.

## Historical evidence

Historical artifacts remain immutable.  An evidence registry should classify
each result as `synthetic_only`, `checkpoint_weight_only`,
`checkpoint_full_payload`, or `full_model`, and mark its authority as
`non_admissible_legacy`, `provisional`, `confirmed`, or `superseded`.

- Synthetic microbenchmarks backed by later real full-model evidence may
  retain the production decision, but their mechanism claims are directional.
- A synthetic-only production selection is provisional and must be
  requalified in priority order; it is not reverted without replacement
  evidence.
- A synthetic-only rejection is no longer a permanent stop.  It is rerun only
  if that mechanism re-enters the active roadmap; old negative cells need not
  be migrated wholesale.

## Profiling order

NCU or NSys follows, never precedes, a completed and valid T2 timing run on the
same real payload.  A rejected candidate may be profiled to explain its gap,
but the profile is diagnostic and cannot reverse that rejection.  For the
C512 NVFP4 bridge comparison, profile the dequantization
kernel, the immediately following cuBLASLt BF16 GEMM, and the native NVFP4
kernel separately, while retaining `dequantization + Lt` versus `native` as
the authoritative timing comparison.  The Lt-only comparison is diagnostic
because it assumes an already materialized BF16 weight.

Collect occupancy and launch resources, pipeline/SASS structure, issue and
Tensor activity, HMMA count, stall reasons, global/L2/L1/shared traffic, and
bank conflicts.  Normalize counters by useful work and record counter
denominators.  Profiler percentages overlap and must not be added into a
latency explanation.  A gap is eligible for implementation only after it has
a quantified cycle or byte ceiling on the real payload; the resulting change
must return through T2 and T3.
