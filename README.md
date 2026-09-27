---
q3x_document:
  id: q3x-project-readme
  class: active
  status: active
  owner: project-maintainers
  authority: product introduction, bounded evaluation quick start, and high-level navigation
  effective: 2026-08-09
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: concise project introduction and bounded functional evaluation entry; dynamic state remains in docs/CURRENT_STATUS.md
  review_trigger: mission, target scope, user-facing evaluation entry, product boundary, or controlling-document change
---

# Qwen3x-Orin

A specialized C++17/CUDA inference service for **Qwen3.6-27B-NVFP4 on
NVIDIA Jetson AGX Orin (SM87)**. It provides OpenAI-compatible text and chat
endpoints with streaming, using native kernels and a fixed execution plan.

The default production service is `qwen3x-eval-server`: corrected whole-core
Prefill plus ordered v7 Decode, packaged as **0.8.0**. The bounded production
switch is complete. Longer-context support and the remaining performance
and full-release goals are tracked in [Current Status](docs/CURRENT_STATUS.md)
and the [Roadmap](docs/ROADMAP.md).

## Supported scope

| Item | Current service |
| --- | --- |
| Model | Exact pinned `nvidia/Qwen3.6-27B-NVFP4` checkpoint; text only |
| Hardware | Jetson AGX Orin, CUDA architecture `sm_87` |
| API | `/healthz`, `/v1/models`, `/v1/completions`, `/v1/chat/completions` |
| Input | Nonempty text, text chat messages, or token IDs for completions |
| Output | Greedy generation; streaming or nonstreaming; optional streaming usage |
| Capacity | `1 <= O <= 4096` and `P + O - 1 <= 44095` |
| Concurrency | One GPU request and one queued request; overload returns HTTP 429 |
| Deployment | Loopback by default; non-loopback requires a Bearer key file; external TLS termination |

`P` is the tokenized prompt length, including chat-template tokens. `O` is
the requested output limit. For example, 40,000 prompt tokens permit 4,096
output tokens; the maximum 44,095-token prompt permits one output token.
Requests exceeding the capacity are rejected, never silently truncated.

Sampling, tool calls, media, custom stop sequences, batching, MTP and Prefix/KV
reuse are outside the current service contract. OpenAI compatibility covers
the supported endpoints and parameters, not the entire OpenAI API. See the
[API procedure](docs/EVALSCOPE_EVALUATION.md) for exact request semantics.

## Measured performance

Selected results from the **2026-09-28 installed-artifact qualification**:

| Prompt / output tokens | Prefill tokens/s | First-token latency (s) | Decode tokens/s |
| --- | ---: | ---: | ---: |
| 1,089 / 16 | 403.23 | 2.719 | 9.588 |
| 8,192 / 256 | 486.84 | 16.849 | 9.218 |
| 40,000 / 256 | 434.92 | 92.007 | 7.864 |
| 40,000 / 4,096 | 434.35 | 92.126 | 7.792 |

The first three rows average two installed processes; the full-output row is
a separate capacity run. These are batch-one real API results with no
Prefix/KV reuse or MTP, an exclusive GPU at 1300.5 MHz, and recorded host/cache
preparation. Prefill is the engine's pure interval; first-token latency is
measured externally; Decode excludes the first token produced by Prefill.
Startup/model loading is outside these request intervals. The
[complete results and protocol](docs/CURRENT_STATUS.md#paired-prefill-and-decode-performance)
own the numbers and their scope; they are not a speed guarantee for another
host or workload.

**40K Decode has not reached the interim 8.55 tokens/s target.** The service
also does not yet support 60K/130K contexts or meet the long-term 2s/4s
first-token targets. Its bounded production eligibility is distinct from full
product release qualification (`release_qualified=false`).

## Functional evaluation quick start

Run these commands from the repository root. This smoke test starts the
production path; it does not reproduce the performance or accuracy panel.

### Requirements

- Jetson AGX Orin with enough available unified memory for the pinned weights,
  startup sidecars, a 9,508,218,624-byte request arena and an additional 8-GiB
  free-memory reserve. The arena alone is not the total memory requirement;
  startup rejects insufficient memory.
- CMake 3.24+, CUDA Toolkit 12.0+ with C++17 support, a system threading
  library, ICU 74+ (`uc` and `i18n`), and `curl` for the examples.
- The checkpoint pinned at revision
  `0893e1606ff3d5f97a441f405d5fc541a6bdf404`, including its tokenizer assets.
  Consult [Model Support](docs/MODEL_SUPPORT.md) for authenticated model facts.
  Other catalogued models are not supported-service claims.

Keep model files read-only. All generated files below stay in the ignored
`.q3x-work/` directory.

### Build and install

```bash
cmake --preset orin-release
cmake --build --preset orin-release --parallel 3
cmake --install .q3x-work/build/orin-release \
  --prefix "$PWD/.q3x-work/install/orin-release"
```

The preset explicitly selects `BUILD_TESTING=OFF` and
`Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=ON`. A fresh custom build with testing
OFF also defaults to the service, but an existing CMake cache can retain old
settings. Use the preset and a separate directory for experiments; do not mix
production with admission options. C++ consumers must rebuild against the
exact 0.8.0 installed package and its exported geometry definitions.

### Start the service

```bash
Q3X_MODEL_DIR="/absolute/path/to/nvidia/Qwen3.6-27B-NVFP4"
Q3X_SERVER="$PWD/.q3x-work/install/orin-release/bin/qwen3x-eval-server"

"$Q3X_SERVER" "$Q3X_MODEL_DIR" \
  --host 127.0.0.1 \
  --port 18080 \
  --model qwen3.6-27b-nvfp4
```

Startup loads and validates weights, prepares the fixed execution resources,
and opens the listener only after readiness. No candidate or tactic flag is
needed. The service identity is
`q3x.sm87.production.whole-core-service.v1`.

### Send a request

From another terminal:

```bash
curl -fsS http://127.0.0.1:18080/healthz
curl -fsS http://127.0.0.1:18080/v1/models

curl -N -fsS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"你好，请用一句话介绍你自己。"}],"max_tokens":64,"temperature":0,"stream":true,"stream_options":{"include_usage":true}}'
```

Use `"stream":false` and omit `stream_options` for a nonstream response.
Generation requests must explicitly specify a positive `max_tokens` or
`max_completion_tokens`, plus `temperature:0`. Stop the foreground server
with Ctrl-C.

For a non-loopback listener, supply `--api-key-file /path/to/key` pointing to
an owner-only regular file (0400 or 0600). Add `Authorization: Bearer …` to
models and generation requests; `/healthz` remains public. Put TLS termination
in a trusted reverse proxy when exposing the service beyond the host.

## Numerical validation and remaining boundaries

The selected Prefill fixes projection coordinates and tensor-scale placement.
Its reference was independently assessed with FP32 GDN recurrence and full
prompt-boundary state/logit comparisons. Legacy per-token BF16 Prefill is a
regression comparator, not model truth. Ordered v7 Decode also has same-input
FP64 Attention checks.

On the pinned four-subject, five-shot C-Eval panel, native and vLLM answers
match on all 98 questions, with both scoring 79/98. This is bounded capability
evidence, not the full 52-subject benchmark or a claim that all future output
is bitwise identical. See the [numerical decision](docs/CURRENT_STATUS.md#numerical-baseline-decision)
and [production record](docs/metadata/qwen36-27b-whole-core-service-production-2026-09-28.json).

`qwen3x-orin generate` is a diagnostic CLI with legacy execution options. It
is **not the default production service path**. Historical P40 development
and fixed-profile presets exist for reproduction; their old numerical
qualification does not transfer to the current service.

## Development and documentation

Start at [the documentation index](docs/README.md) for the required reading
order and [AGENTS.md](AGENTS.md) for workspace and execution rules.

| Task | Document |
| --- | --- |
| Current defaults, qualification, performance and gaps | [Current Status](docs/CURRENT_STATUS.md) |
| Remaining delivery work | [Roadmap](docs/ROADMAP.md) |
| API-to-kernel design and lifecycle | [System SDD](docs/SDD.md) |
| Reproducible API evaluation | [EvalScope procedure](docs/EVALSCOPE_EVALUATION.md) |
| Performance-run requirements | [Real-model performance policy](docs/REAL_MODEL_PERFORMANCE_POLICY.md) |
| Engineering mission and owner-set targets | [Constitution](docs/ENGINEERING_CONSTITUTION.md) |
| Current contracts and historical evidence | [Document registry](docs/DOCUMENT_REGISTRY.md) |

Code lives in `include/q3x/` and `src/`; tests in `tests/`; evaluation and
inspection tools in `tools/`; pinned workload definitions in `benchmarks/`.
Builds, raw results and temporary tools belong under `.q3x-work/`.

The project deliberately specializes model, numerical format, hardware and
serving together. It is an independent community project, not an official or
endorsed Qwen, Alibaba, NVIDIA or Jetson product.

## License and provenance

Project-authored code uses [Apache License 2.0](LICENSE). Upstream-derived
components retain their own notices and licenses; see [NOTICE](NOTICE) and
vendored source notices. No external inference engine or cuBLASLt runtime
fallback is part of the production route. Model weights and tokenizer assets
are distributed separately under their publishers' terms.
