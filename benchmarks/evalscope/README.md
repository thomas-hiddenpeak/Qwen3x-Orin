# Pinned EvalScope workloads

This directory stores reproducibility manifests, not prompt corpora or
EvalScope result databases. Prompt token IDs can be reversed into copyrighted
text, and the generated databases contain full requests and responses. Those
artifacts stay outside the repository; their SHA-256 identities are recorded
in the manifests and evidence records instead.

## ShareGPT false-thinking performance corpus

`qwen36-sharegpt-false-thinking-v1.manifest.json` describes 33 exact
`/v1/completions` request bodies. Ordinal 0 is the warmup and ordinals 1--32
are measured. Every request contains a flat token-ID prompt produced with the
pinned Qwen3.6 tokenizer, `add_generation_prompt=true`, and
`enable_thinking=false`. It requests 16 greedy streamed tokens with seed 42
and a final usage chunk.

Materialize the corpus from the hash-locked `swift/sharegpt` source:

```bash
python3 tools/evaluation/build_evalscope_sharegpt_requests.py \
  --source SHAREGPT_ROOT/common_zh_70k.jsonl \
  --tokenizer-dir MODEL_DIR \
  --output /tmp/q3x-sharegpt-false-thinking-33.jsonl \
  --manifest /tmp/qwen36-sharegpt-false-thinking-v1.manifest.json
```

The builder fails closed on the source, tokenizer, standalone chat template,
and final corpus hashes. Its default contract intentionally rejects altered
counts, offsets, tokenization, or request parameters. To define another
version, first declare a new manifest and pass its precomputed expected corpus
hash; do not silently mutate this workload.

`prompt_sha256` hashes the canonical JSON token-ID array.
`request_sha256` hashes the builder's canonical JSON request object; it is not
a claim about the exact HTTP bytes later serialized by EvalScope.

The full external evaluation procedure and evidence authority are documented
in [the EvalScope evaluation contract](../../docs/EVALSCOPE_EVALUATION.md).
