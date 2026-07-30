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

## Pure-Prefill length matrix

The `qwen36-sharegpt-prefill-*-v1.manifest.json` contracts define four
one-token, batch-one length buckets.  Each bucket contains one warmup followed
by four measured prompts.  P512/P1K/P2K use complete natural conversations
from the pinned `swift/sharegpt` source.  P4K uses five distinct complete
conversation prefixes from the pinned public
`anon8231489123/ShareGPT_Vicuna_unfiltered` revision.  Selection never pads,
repeats, truncates, or concatenates token sequences.

Materialize the first three buckets from the local hash-locked source:

```bash
uv run --no-project \
  --with 'transformers==5.14.1' --with 'jinja2==3.1.6' \
  python tools/evaluation/build_evalscope_prefill_length_matrix.py \
  --source SHAREGPT_ROOT/common_zh_70k.jsonl \
  --tokenizer-dir MODEL_DIR --output-dir /tmp/q3x-pure-prefill-corpora-v1
```

Materialize P4K by streaming the immutable public revision:

```bash
uv run --no-project \
  --with 'transformers==5.14.1' --with 'jinja2==3.1.6' \
  --with 'ijson==3.4.0.post0' --with 'requests==2.32.5' \
  python tools/evaluation/build_evalscope_prefill_p4k.py \
  --tokenizer-dir MODEL_DIR --output-dir /tmp/q3x-pure-prefill-corpora-v1
```

The builders fail closed on source, tokenizer, template, selected-row, prompt
window, and final corpus hashes.  Run all four buckets through the native API:

```bash
tools/evaluation/run_native_pure_prefill_matrix.sh \
  SERVER_ELF MODEL_DIR /tmp/q3x-pure-prefill-corpora-v1 OUTPUT_ROOT
```

This harness fixes `max_tokens=1`, `parallel=1`, one worker, one warmup, four
measured requests, streaming token-ID completions, the SM87 backend, a C512
Prefill chunk, no Prefix cache, and no MTP.  It deliberately removes every
Decode-only environment gate; there is no Decode step after the first token.

EvalScope TTFT is the external product metric: POST start through receipt of
the first non-empty token event.  `prompt_tokens / TTFT` is therefore an
effective end-to-end Prefill rate, not a kernel rate.  The runtime's
`prompt_prefill_milliseconds` is the internal GPU schedule metric (Prefix
tiles plus the logits-bearing finalizer).  It excludes HTTP and must always be
reported separately from external TTFT.  EvalScope's workload prompt
throughput includes serialized request transitions and is not a pure Prefill
rate.
