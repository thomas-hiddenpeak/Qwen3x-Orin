# BF16 reference oracle loader and comparison contract

`q3x::runtime::load_reference_oracle` turns the checked-in BF16-cache greedy
and layer-boundary JSON pair into a runner-independent host data model. It is
the strict handoff between captured vLLM evidence and future native trace
comparison; it does not execute or own a model.

## Input and trust boundary

Callers provide both paths explicitly:

```cpp
auto result = q3x::runtime::load_reference_oracle(
    "qwen36-27b-nvfp4-greedy-bf16.json",
    "qwen36-27b-nvfp4-layers-bf16.json");
```

Each path must resolve to a regular, non-symlink file. The loader opens with
`O_NOFOLLOW`, validates the open descriptor with `fstat`, reads exactly the
bounded size, and rejects a file whose identity or size changes during the
read. `ReferenceOracleLimits` independently bounds:

- bytes per file;
- total JSON DOM nodes and container members;
- nesting depth;
- bytes in every string or object key;
- elements in every individual array.

Caller limits cannot exceed compiled absolute ceilings. JSON duplicate keys,
invalid UTF-8, malformed numbers, trailing bytes, unexpected fields, missing
fields, and wrong JSON types fail with a structured diagnostic.

## Pinned schema

Both files must use schema version 1 and the exact NVIDIA checkpoint revision
`0893e1606ff3d5f97a441f405d5fc541a6bdf404`. Repository file sizes and hashes
are checked against the same pinned checkpoint identity as the resident
loader. KV, causal-conv, and DeltaNet state/cache policies must all be
`bfloat16`.

The greedy record requires exactly 19 prompt token IDs, 26 expected token IDs,
26 finite chosen-token log probabilities, exact stop semantics, and the
recorded deterministic sampling policy. The layer record requires exactly:

- prefill position 18, input token 271, prediction 77517;
- decode position 19, input token 77517, prediction 220;
- embedding and final-norm summaries of length 5120 in each phase;
- ordered layer indices 0 through 63, each with hidden and residual summaries;
- BF16 length-5120 boundaries with a lowercase SHA-256, finite statistics,
  and 16 samples at the canonical positions;
- BF16 length-248320 logits, finite logsumexp/chosen log probability, and 20
  unique top logits in descending order.

The layer source filename and revision are cross-checked against the greedy
path, and both phase predictions must equal the first two greedy output IDs.

## Boundary diagnostics

`summarize_bf16_span` accepts raw host BF16 words. It rejects null, empty, or
non-finite spans and invalid sample positions, then returns:

- SHA-256 over canonical little-endian BF16 bytes;
- FP64 mean, RMS, minimum, and maximum;
- decoded values at caller-supplied, strictly increasing sample indices.

The raw SHA is an exact regression gate only for the same backend, cache
policy, operation order, and BF16 rounding behavior. A different correct
backend can change low bits, reduction order, or intermediate rounding.

`compare_boundary_samples` is the cross-backend diagnostic path. It scans the
actual span for NaN/Inf, then compares the declared sample positions with
`abs(actual - expected) <= atol + rtol * abs(expected)`. A numerical mismatch
is a valid result and reports the first sample ordinal, tensor index, values,
and applied tolerance. Samples and distribution statistics help locate a
divergence; they are not a proof that every unsampled element matches.

The future runner should therefore use hashes for exact same-backend replay,
and samples/statistics plus a deliberately chosen tolerance for native versus
reference localization.
