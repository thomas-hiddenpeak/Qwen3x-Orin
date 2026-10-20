---
q3x_document:
  id: q3x-project-readme
  class: active
  status: active
  owner: project-maintainers
  authority: product introduction, bounded evaluation quick start, and high-level navigation
  effective: 2026-08-09
  last_reviewed: 2026-10-21
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
Prefill plus ordered v7 Decode, packaged as **0.8.2**. The bounded production
switch is complete. Longer-context support and the remaining performance
and full-release goals are tracked in [Current Status](docs/CURRENT_STATUS.md)
and the [Roadmap](docs/ROADMAP.md). The [0.8.2 corrective package](docs/metadata/qwen36-27b-softmax-package-2026-10-14.json)
adds the validated softmax synchronization repair; its bounded installed checks
do not renew the historical two-hour reliability qualification.

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
reuse are outside the ordinary non-MTP service contract. The independent MTP
package below adds target-verified speculative generation. OpenAI compatibility covers
the supported endpoints and parameters, not the entire OpenAI API. See the
[API procedure](docs/EVALSCOPE_EVALUATION.md) for exact request semantics.

## Measured performance

The **2026-10-20 exact installed comparison** uses two fresh processes per
configuration and the same requests. Values below are medians; the conservative
Decode gain uses the slower MTP observation divided by the faster non-MTP one.

| Prompt / output | Configuration | Prefill s / tokens/s | TTFT s | Decode tokens/s | Total s | Conservative gain |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 65 / 16 | Non-MTP | 0.888 / 73.17 | 0.907 | 9.582 | 2.472 | — |
| 65 / 16 | MTP cap 2 | 0.894 / 72.74 | 0.912 | 14.408 | 1.953 | +50.28% |
| 65 / 16 | MTP cap 3 | 0.899 / 72.30 | 0.918 | 15.264 | 1.900 | +59.13% |
| 8192 / 256 | Non-MTP | 16.781 / 488.17 | 16.803 | 9.215 | 44.480 | — |
| 8192 / 256 | MTP cap 2 | 17.325 / 472.84 | 17.347 | 15.946 | 33.339 | +73.02% |
| 8192 / 256 | MTP cap 3 | 17.348 / 472.22 | 17.370 | 15.930 | 33.378 | +72.79% |
| 40000 / 256 | Non-MTP | 90.064 / 444.13 | 90.099 | 7.863 | 122.533 | — |
| 40000 / 256 | MTP cap 2 | 92.667 / 431.65 | 92.702 | 12.018 | 113.921 | +52.57% |
| 40000 / 256 | MTP cap 3 | 92.698 / 431.51 | 92.733 | 12.009 | 113.968 | +52.53% |

These are batch-one API results with no Prefix/KV reuse, an exclusive GPU at
1300.5 MHz and recorded host/cache preparation. Prefill uses the engine interval
and includes MTP draft initialization; TTFT and total are external. Decode
excludes the first token produced by Prefill. Startup/model loading is outside
these intervals. MTP increases Prefill time but reduces total time here.
Cap 2's short-context gain has only a 0.28 percentage-point margin above 50%.
Neither 200% gain nor a universal prompt guarantee is established.
[Current Status](docs/CURRENT_STATUS.md#installed-mtp-delivery-complete)
owns the measurements, exact artifact identities and qualification limits.

**Non-MTP 40K Decode has not reached the interim 8.55 tokens/s target.** The service
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
  library, Python 3 for delivery checks, ICU 74+ (`uc` and `i18n`), and `curl` for the examples.
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
ctest --preset orin-release --output-on-failure
cmake --install .q3x-work/build/orin-release \
  --prefix "$PWD/.q3x-work/install/orin-release"
```

The preset explicitly selects `BUILD_TESTING=OFF` and
`Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=ON`. A fresh custom build with testing
OFF also defaults to the service, but an existing CMake cache can retain old
settings. Use the preset and a separate directory for experiments; do not mix
production with admission options. C++ consumers must rebuild against the
exact 0.8.2 installed package and its exported geometry definitions.

The release CTest lane checks the gateway, sealed identity, documentation,
installed package and validation-tool failure cases even though `BUILD_TESTING=OFF`; production test
hooks remain excluded. An empty test set fails. GPU/model qualification is an
explicit second stage using the [versioned service validation
procedure](docs/EVALSCOPE_EVALUATION.md#service-industrialization-validation).

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

## Independent MTP package

MTP is an opt-in executable-only package, separate from the ordinary non-MTP
service above. **Exact installed delivery checks are complete for caps 2 and 3**:
real API correctness, fixed-workload performance, existing capacity, 98-case
capability, two-hour reuse per cap and fresh-process recovery. The
[delivery audit](docs/metadata/qwen36-27b-mtp-delivery-complete-2026-10-21.json)
binds the qualified binaries built from clean source `65e13db`; rebuilding a
new artifact requires the affected checks. The ordinary default remains non-MTP;
no running service or systemd unit has been switched.

Build one fixed cap (replace `cap3` with `cap2` for the other package):

```bash
cmake --preset orin-mtp-package-cap3
cmake --build --preset orin-mtp-package-cap3 --parallel 3
ctest --preset orin-mtp-package-cap3
cmake --install .q3x-work/build/orin-mtp-package-cap3 \
  --prefix "$PWD/.q3x-work/install/mtp-cap3"
python3 -B .q3x-work/install/mtp-cap3/share/Qwen3xOrinMTP/tools/audit_mtp_install.py \
  --prefix .q3x-work/install/mtp-cap3 --cap 3
.q3x-work/install/mtp-cap3/bin/qwen3x-mtp-server-cap3 MODEL_DIR \
  --host 127.0.0.1 --port 8000 --api-key-file /absolute/path/to/owner-only-key
```

Use a separate install prefix; this package exports no C++ libraries or headers.
It preserves the same API capacity and authentication contract and requires the
8-GiB post-startup free-memory reserve. Run only one GPU service at a time.
The optional `share/Qwen3xOrinMTP/systemd/qwen3x-mtp@.service` template uses
instance `cap2` or `cap3`; configure its account, model, key, port and executable
path before deployment. Installation never enables or starts it. Fatal runtime
recovery restarts the process; recreating the large MTP owner in the same
process is not qualified.

## Service operation

The installed `share/Qwen3xOrin/systemd/` directory contains a service unit and
`server.env.example`. Before enabling it, create a dedicated `qwen3x` system
account, grant read/traverse access to the pinned model directory, and confirm
that the `video` and `render` groups grant access to the Jetson GPU devices.
Copy the environment template to `/etc/qwen3x-orin/server.env` and set its
absolute model path. Copy the unit to `/etc/systemd/system/qwen3x-orin.service`.
The template assumes `/usr/local/bin`; for a different installation, override
`ExecStart` with the absolute installed server path and the same arguments.
Then run `systemctl daemon-reload` and `systemctl enable --now qwen3x-orin`.
These are operator deployment steps, not actions performed by CMake install.

The unit binds loopback port 8000. To require authentication locally or expose
another interface, use an `ExecStart` override with `--api-key-file` and the
intended host/port. The key must be owned and readable only by the service
account; keep TLS termination at the trusted proxy. Model files remain
read-only inputs. Use `journalctl -u qwen3x-orin` for failure diagnostics.

Fatal CUDA/state failures withdraw readiness and exit nonzero. The template
restarts on failure, at most three starts per ten minutes. Correct the cause
before clearing a start-limit failure. Normal SIGINT shutdown does not restart.
Rollback restores a complete retained installation prefix and its matching
headers/libraries; do not mix package versions. Preserve the model and API
configuration when comparing recovery output.

Output limits may split a byte-level token sequence. The API renders any
terminal incomplete UTF-8 subsequence as U+FFFD while preserving model tokens
and usage. Slow incomplete HTTP requests have bounded staging capacity and
deadlines; the service remains a bounded single-request engine.

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

The **0.8.1 reliability batch is complete**: six production-build checks,
36 Python tests, 98 matched capability answers, two hours / 1,708 reuse
requests without RSS/descriptor/thread growth, and fresh-process recovery.
The soak uses one-second request gaps; continuous maximum-load saturation is
not qualified. The [delivery record](docs/metadata/qwen36-27b-service-industrialization-2026-09-28.json)
and [Current Status](docs/CURRENT_STATUS.md#non-performance-product-audit-2026-09-28)
retain exact scope, the earlier thermal stop, and paired Prefill/Decode observations.
