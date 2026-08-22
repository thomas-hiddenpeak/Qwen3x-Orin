---
q3x_document:
  id: q3x-reference-runner
  class: contract
  status: active
  owner: runtime-maintainers
  authority: batch-one CUDA runner state, numerical, ownership, and failure contract
  effective: 2026-08-09
  last_reviewed: 2026-08-23
  supersedes: []
  superseded_by: []
  ssot_for: ReferenceRunner public execution, commit, poison, reset, trace, and dependency behavior
  review_trigger: any ReferenceRunner ABI, state transition, numerical boundary, ownership, or failure change
---

# Qwen3.6-27B batch-one CUDA runner contract

> **Authority boundary.** This component contract refines the execution
> boundary in the [system SDD](SDD.md) and is subordinate to it and the
> [engineering constitution](ENGINEERING_CONSTITUTION.md). Current
> implementation, default-route, qualification, and release truth belongs in
> [`CURRENT_STATUS.md`](CURRENT_STATUS.md). Kernel names, tile palettes,
> stream topology, local fusion choices, and performance gates are excluded;
> they may govern only a named active local work package.

The public API is `include/q3x/runtime/reference_runner.h`.
`ReferenceRunner` is a serialized, correctness-first CUDA executor for one
request. It exposes token-step, prompt-prefix, retained-final-hidden, reset,
trace, and explicitly diagnostic graph surfaces without giving any one kernel
mechanism global scheduling authority.

## Dependencies and ownership

The runner is move-only and owns its internal CUDA resources. It retains
non-owning pointers to the exact `ModelWeights` and mutable `RequestState`
objects supplied at creation. Those objects, the resident arena behind all
weight views, and any attached sidecar arenas must outlive the runner and all
queued consumers. They must not be moved, externally reset, or concurrently
used from another stream while the runner is alive.

Factory validation checks the pinned 64-layer model graph, the 48/16 layer
schedule, request workspace/cache/RoPE capacities, backend selection, and all
required payloads before publishing a runner. Failure publishes no usable
runner.

### Layer-major candidate view seam

The ordinary `create_reference_runner()` factory remains bound to the legacy
C512 `RequestState` profile. It continues to collect the established flat
workspace views and does not admit, reinterpret, or execute a layer-major
state.

A separate explicit candidate-only seam prepares the next integration step
for `AC-PREFILL-LAYERMAJOR-8K-v1`. Its pure-host descriptor validates the
fixed profile, 48/16 schedule, physical tactic identities, request capacity,
persistent GDN/KV/RoPE regions, prompt residual, token-ID staging, exact
GDN/Attention/MLP phase topology, disjoint legacy C512 bundle, and fixed final
hidden handoff. All six execution/alias/event/operator binding flags must
remain false. A legacy profile fails with `kMemoryProfileMismatch` before an
empty-state or CUDA check.

For a real allocated layer-major state, one explicit collector obtains every
typed device view and proves that its metadata and pointer-offset identity
match the validated descriptor. Compact persistent views are indexed by the
fixed layer-slot schedule. The collector never returns an untyped view of the
owning C8192 family arena.

This seam owns no stream, launcher, model weight, completion event, traversal,
state publication, factory connection, engine callback, or selector. Its
descriptor is always `kUnboundCandidateOnly`; collecting it is not an
executable-plan attestation and changes no default or production route.

## Token-step semantics

`step(input_token_id, options)` operates at the current request position and
executes the pinned decoder in layer order. It rejects token IDs outside the
248,320-entry vocabulary and positions without capacity.

The stable numerical boundaries are:

- BF16 model activations and persistent state;
- FP32 accumulation inside projection and reduction operations unless a
  narrower public operation contract says otherwise;
- centered RMSNorm with the pinned model epsilon and BF16-rounded boundary
  values;
- canonical FP8/NVFP4 checkpoint decoding as defined by
  [`MODEL_WEIGHT_BINDING.md`](MODEL_WEIGHT_BINDING.md); and
- greedy selection after the complete logits vector is rounded to BF16.

`ReferenceLogitsMode::kFullStatistics` returns the predicted token, chosen
logit, logsumexp, and maximum log-probability. `kPredictedTokenOnly` validates
finiteness and returns only the greedy ID. Argmax ties select the smallest
index. NaN or infinity in logits fails the step.

When `compute_logits=false`, all decoder layers and persistent-state updates
still execute, but lm-head and logits analysis are omitted. Timing, when
requested, is host elapsed time through the required completion boundary.
Trace capture is valid only if trace storage was enabled at factory time.

A successful step commits exactly one logical position only after all required
work and host-visible validation succeed. A failed step does not report a
partial result.

## Prompt-prefix contract: 512 is the public capacity

`prefill_prefix_tile(input_token_ids, token_count, options)` accepts
`1 <= token_count <= 512`, provided the request plan reserves at least that
many workspace rows and remaining sequence capacity. It executes every token's
64-layer semantics, updates Conv/GDN/KV state in logical token order, produces
no logits or trace, and commits the complete tile only at its successful
completion boundary.

`ReferencePrefillTileResult::steps` therefore has 512 entries. That public
C512 boundary was introduced by package ABI 0.4.0 and remains unchanged
through 0.6.0; version 0.5.0 changed the separate request-state object ABI for an isolated
layer-major candidate. The 64-token limit on a generic projection dispatcher
is an internal component constraint; it does **not** narrow this runner API.
The runner may decompose an admitted 1..512 request internally, but no
particular decomposition, kernel, buffering pattern, or stream plan is part of
this contract.

If `retain_last_hidden_for_logits=true`, a successful tile retains the final
normalized hidden row for exactly one matching
`finish_prefill_from_retained_tile` call. The finish operation performs only
lm-head/logits analysis, does not rerun decoder layers, does not update
persistent state, and does not advance logical length. Any mismatch, reset,
intervening incompatible operation, or failed tile invalidates that retained
boundary.

## Poison, reset, and trace

Every failed ordinary execution call routed through the step or prefix failure
boundary poisons the runner. This includes host-visible invalid options,
tokens, capacity, or trace requests detected before enqueue as well as failures
after mutable device work may have begun. Retained-final-hidden completion and
graph execution follow their corresponding checked failure paths. A poisoned
runner rejects further execution. `reset()` is the only poison-recovery
operation: it synchronizes owned execution resources, resets the request's
persistent state, clears retained-final-hidden and trace validity, and returns
the position to zero. If reset itself fails, poison is retained.

Public `reset()` always retains that complete one-span behavior. The Engine's
private ordinary-request entry instead derives reset work from runner-owned
lifecycle facts. A newly created or successfully full-reset legacy C512 state
is `already_clean`; a prior successful non-cancelled request whose exact
logical length still matches is `committed_dirty_prefix`; every poison,
active/uncommitted whole-request stage, cancellation boundary, position
mismatch, unknown boundary, or nonlegacy candidate profile is
`conservative_full`. Request entry consumes the prior authority and marks the
new boundary uncertain before execution. Only the Engine's completed-request
acceptance republishes an exact committed position. Direct runner operations
cannot grant prefix-reset authority.

Each successful request entry returns a synchronized receipt containing reset
mode, cleared positions, zeroed bytes, and elapsed milliseconds. The
already-clean mode performs no CUDA reset; prefix and full modes synchronize
their submitted reset before execution. These modes are compiled lifecycle
policy, not an environment, CLI, or test-selected production tactic.

The last trace is a non-owning view of the most recently captured and
successfully committed step. Its fixed layout is embedding, then hidden and
residual vectors for layers 0..63, then final norm; each logical vector has
5,120 BF16 elements. It is invalidated by reset, runner destruction, or the
next captured step.

## Diagnostic graph surface

The fixed-position CUDA Graph APIs are explicitly diagnostic/test surfaces.
Their public slot bank covers positions `[0,64)` and is unrelated to the
512-token Prefill result capacity. Preparation is transactional; replay is
valid only for a prepared current position and preserves the ordinary
prediction-validation and commit boundary. Missing or incompatible slots are
caller-visible conditions, not authority to redefine the default runtime
route or Production lifecycle state.

## Failure semantics

Errors distinguish invalid dependencies, weights, request state, layer
schedule, runner/options, token/capacity, trace availability, non-finite
logits, commit failure, allocation failure, CUDA failure, and poison. Public
launch boundaries validate all host-visible arguments before enqueue whenever
the operation can do so. An unsupported optional route must either fall back
before enqueue through its documented caller boundary or return an explicit
unsupported result; malformed payloads are errors, not fallback signals.

Correctness and historical performance evidence lives in
[`PERFORMANCE_BASELINE.md`](PERFORMANCE_BASELINE.md) and
[`PHASE0_EVIDENCE.md`](PHASE0_EVIDENCE.md). Exact lineage for the widened
runtime-prefix boundary includes the
[`M17/M19..M31 runtime-masked record`](metadata/qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json).
Those records do not impose active runner mechanisms or promotion thresholds.
