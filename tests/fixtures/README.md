---
q3x_document:
  id: q3x-reference-fixtures
  class: evidence
  status: frozen
  owner: verification-maintainers
  authority: normalized correctness-fixture meaning and provenance catalog
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: meaning and provenance of the checked-in fixtures at their recorded identities
  review_trigger: any fixture addition, replacement, schema, source identity, or provenance change
---

# Reference fixtures

> **Authority boundary.** This frozen fixture catalog is subordinate to the
> [engineering constitution](../../docs/ENGINEERING_CONSTITUTION.md) and
> [system SDD](../../docs/SDD.md). It records oracle meaning and provenance for
> the exact checked-in fixture identities; current implementation,
> qualification, accuracy, and default-route truth belongs in
> [`CURRENT_STATUS.md`](../../docs/CURRENT_STATUS.md). Fixture timing or local
> diagnostics cannot select architecture or production.

These small, normalized fixtures contain expected numerical or token results,
not model weights or upstream configuration files. Each fixture records the
exact source revision and file hash used to produce it so a test cannot silently
drift with a tokenizer or checkpoint update.

`qwen36-27b-tokenizer.json` was produced with the official tokenizer at the
pinned NVIDIA checkpoint revision and Hugging Face Transformers 5.12.1. It is
the independent oracle for the future pure-C++ tokenizer implementation.

`qwen36-27b-nvfp4-greedy.json` is a real, text-only end-to-end oracle captured
on the target Jetson AGX Orin. The pinned NVIDIA ModelOpt checkpoint was loaded
by the recorded vLLM revision in eager mode, with FP8 and NVFP4 weights routed
through the SM87-compatible Marlin W8A16/W4A16 paths. The same greedy request
was executed twice in one engine; all output token IDs, decoded text, and
reported chosen-token log probabilities matched. Token IDs are the normative
gate. Log probabilities are a diagnostic gate with the tolerances recorded in
the fixture; wall-clock timings are evidence only.

That primary capture follows the checkpoint/runtime defaults: FP8 E4M3
full-attention KV, BF16 causal-conv cache, and the model configuration's FP32
DeltaNet state. `qwen36-27b-nvfp4-greedy-bf16.json` is the matching policy
capture with all three caches explicitly forced to BF16, which is the native
correctness runner's bounded-memory baseline. All 26 greedy token IDs and the
decoded text remain identical; chosen-token log probabilities move slightly,
as expected from the cache policy.

`qwen36-27b-nvfp4-layers.json` is a compact locator for the same oracle. It
captures the final prompt position and the first recurrent decode position at
the embedding, every one of the 64 decoder-layer boundaries, final norm, and
logits. Each BF16 boundary records a raw-byte SHA-256, distribution statistics,
and values at the fixed indices declared once in the fixture. Hashes detect
exact same-runtime drift; samples/statistics are the portable diagnostic path
for locating the first native-runtime divergence. The capture is reproducible
with `tools/reference/qwen36_27b_vllm_layers.py`.

`qwen36-27b-nvfp4-layers-bf16.json` repeats those two layer-boundary phases
with KV, causal-conv, and DeltaNet state all forced to BF16. Native layer
traces should compare against this fixture first. Both reference utilities
now pass every cache dtype recorded by a fixture explicitly and also accept
command-line dtype overrides for controlled policy comparisons.

The schema-1 BF16 pair is consumed by the bounded, fail-closed C++ API
documented in `docs/REFERENCE_ORACLE.md`. Raw hashes are exact same-backend
gates; fixed samples and statistics are the cross-backend localization path.
