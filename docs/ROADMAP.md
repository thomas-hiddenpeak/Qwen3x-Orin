# Qwen3x-Orin roadmap

This roadmap is ordered by technical dependency, not by calendar date. A phase
is complete only when its exit criteria are reproducible on a Jetson AGX Orin.
The project has a complete Phase 1 direct-load reference pipeline and is in
Phase 2 correctness bring-up for the pinned Qwen3.6 27B model. Qwen3.6 dense
and MoE metadata evidence is pinned and reproducible; Qwen3.5 ModelOpt
packaging remains an explicit gap rather than being inferred from shape
compatibility.

## Phase 0 — Bootstrap and evidence capture

Deliverables:

- CMake/C++/CUDA repository skeleton targeting `sm_87`.
- Checked CUDA device discovery and a minimal test executable.
- Model-support descriptors separated from kernels.
- Scripts or tools that inspect target checkpoint headers and quantization
  metadata without loading all weights.
- Recorded representative dimensions and formats for dense FP4, dense FP8,
  and MoE expert microbenchmarks.
- Apache-2.0 license, source-provenance policy, design, and roadmap.

Exit criteria:

- The bootstrap tree configures, builds, and tests on the target Orin.
- Each planned model revision has a checked-in metadata report or reproducible
  command, and unknown revisions fail closed.
- Benchmark shapes come from inspected checkpoint metadata rather than memory
  or model-name assumptions.

Current evidence: the Orin build/device gate, Qwen3.6 pinned reports,
metadata-only inspector, negative-format/revision cases, and checkpoint-derived
benchmark shapes are recorded in [PHASE0_EVIDENCE.md](PHASE0_EVIDENCE.md).
The two Qwen3.5 rows remain catalogued architecture targets until a concrete
ModelOpt checkpoint revision is available and pinned.

## Phase 1 — Quantization references and weight pipeline

Deliverables:

- [done] Safetensors index/header reader with checked offsets and bounded
  metadata-only I/O.
- [done] Strict ModelOpt mixed-precision metadata parser and pinned per-module
  dispatch records for the Qwen3.6 artifacts.
- [done] Exhaustive E2M1 decoder plus E4M3 and global-scale reference handling.
- [done] CPU or straightforward CUDA references for NVFP4 W4A16 and FP8
  W8A16.
- [done] Strict ownership, membership, range, and aggregate-payload validation
  across every indexed shard in the fully materialized 27B checkpoint.
- [done] Authenticated one-pass, double-buffered direct loader with one
  256-byte-aligned text-only CUDA arena and a pre-allocation memory gate.
- Lossless Marlin-oriented repack and scale preprocessing primitives.
- Versioned `.q3x` container prototype with source hashes and atomic output.

Exit criteria:

- Repacking round-trips on synthetic and real representative tensors.
- Reference W4A16/W8A16 results match an independent implementation within
  documented per-operation tolerances.
- Direct loading streams and authenticates data without retaining two complete
  model copies; the same property remains required of the future repacker.
- Corrupt, stale, truncated, or wrong-architecture caches are rejected.

## Phase 2 — 27B text-only correctness

Deliverables:

- [done, reference] BF16 common decode kernels, exact Q/Gate layout split, and
  BF16/FP8/NVFP4 allocation-free projection dispatch. cuBLASLt/optimized
  backends remain performance work.
- [done, native reference; fixed-oracle token/text gate passed] Hybrid sequence
  layers, attention, recurrent/DeltaNet state, normalization, positional
  encoding, embeddings, and output projection for the pinned 27B revision.
- [done, direct] Direct checkpoint loading; optional `.q3x` loading remains.
- [done, native reference; fixed-oracle token/text gate passed] Bounded BF16
  KV/state cache, pinned tokenizer/chat integration, sequential greedy
  generation, and a script-friendly command-line runner.
- [done, memory substrate] Single-arena batch-one Conv/GDN state, full-attention
  KV cache, allocation-free workspace, BF16-rounded RoPE cache, and async reset.
- [done, oracle] Pinned target-device greedy tokens and log probabilities from
  an independent runtime for the first fixed prompt.
- [done, oracle] Two-phase 64-layer boundary fixtures for both the original
  runtime cache policy and an explicitly all-BF16 cache/state policy.

Current native evidence: the fixed 19-token prompt produces all 26 oracle
output IDs, decoded UTF-8 text, and `<|im_end|>` stop exactly on the target
Orin. The one-run request arena was 82,505,216 bytes; cold load was 213.845
seconds and sequential generation was 49.212 seconds. Native boundary hashes
are not required to equal vLLM hashes because independent checkpoint scales
versus fused requantization and sequential versus chunk BF16 GDN updates have
different rounding/order. Tolerance-based boundary characterization and
broader prompt repeatability remain in progress.

Exit criteria:

- Fixed prompts produce reference-aligned intermediate activations and logits.
- Greedy generation is stable across repeated runs within the documented
  accumulation policy.
- Startup reports a complete memory budget and short-context generation fits
  on a 64 GB Orin without swap dependence.

## Phase 3 — Ampere weight-only performance

Deliverables:

- `sm_87` NVFP4 and FP8 single-token GEMV kernels.
- Marlin-style W4A16 and W8A16 kernels for small token batches.
- Shape-driven kernel registry and measured dispatch thresholds.
- Dense-prefill comparison among Marlin-style, cuBLASLt-assisted, and reference
  paths.
- Reproducible benchmark harness with Jetson power/clock metadata.

Exit criteria:

- All optimized kernels pass Phase 1 numerical tests and awkward-shape tests.
- Representative kernels show a material, repeatable improvement over the
  reference path without increasing model-load peak beyond its budget.
- End-to-end 27B decode and time-to-first-token measurements are published with
  enough environment detail to reproduce them.

No throughput target is fixed before Phase 0 measurements. This prevents a
speculative estimate from becoming an accidental compatibility promise.

## Phase 4 — 35B-A3B MoE

Deliverables:

- Router, top-k, routing-weight normalization, shared-expert, and scatter/reduce
  semantics for a supported 35B-A3B revision.
- Simple per-expert execution path for numerical reference.
- Token-to-expert sorting and grouped/fused FP4 expert execution.
- Decode, prefill, and multi-token dispatch policies that avoid per-expert
  launch explosions.
- MoE-specific workspace and memory planning.

Exit criteria:

- Router choices, routing weights, expert intermediates, and final layer
  outputs match trusted fixtures.
- Grouped execution matches the simple path within documented tolerances.
- Full text generation runs within the 64 GB budget.
- Profiling demonstrates that optimized execution no longer launches a naive
  gate/up/down sequence independently for every selected expert and token.

## Phase 5 — Serving and long-context foundations

Deliverables:

- Paged KV/state allocation and request lifecycle management.
- FP8 KV option after independent accuracy validation.
- Continuous batching and cancellation-safe scheduler.
- Streaming OpenAI-compatible text API with explicit supported fields.
- Memory admission control, structured metrics, and graceful error responses.

Exit criteria:

- Mixed prompt lengths and cancellations do not leak cache pages or corrupt
  another request.
- Server and CLI use the same model/runtime interfaces and produce equivalent
  deterministic output.
- Context-limit and out-of-memory decisions happen before unsafe partial work.
- Concurrency benchmarks publish latency percentiles as well as throughput.

## Phase 6 — MTP, vision, and broader optimization

These features begin only after both text models have stable correctness and
performance baselines.

Possible work:

- MTP loading, verification, acceptance logic, and tuned small-batch kernels.
- Vision encoder and multimodal preprocessing for explicitly supported model
  revisions.
- Prefix caching and cache persistence where correctness and privacy boundaries
  are clear.
- CUDA graph capture for stable launch shapes.
- Auto-tuning across Orin power modes and JetPack releases.
- Additional Qwen3.5/3.6 revisions when their architecture and formats fit the
  engine's explicit compatibility contract.

Exit criteria are feature-specific and must include independent correctness,
memory, and end-to-end tests. MTP speedup is reported together with acceptance
rate and any output-policy difference; vision support is not declared from
text-only model compatibility.

## Cross-cutting release gates

Every versioned release must include:

- exact supported checkpoint revisions or hashes;
- build and runtime requirements;
- clean build and test results on a named Jetson/JetPack configuration;
- model memory budgets and benchmark methodology;
- known numerical and functional limitations;
- an attribution audit covering copied, adapted, and vendored code;
- no model weights or other separately licensed artifacts in release archives.

Work may move between phases when measurements justify it, but correctness,
metadata-driven compatibility, bounded memory, and license provenance remain
release gates rather than optional cleanup.
