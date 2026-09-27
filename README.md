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

Qwen3x-Orin is a pure C++17/CUDA runner built for one deliberately narrow
proof vehicle: text-only execution of the exact pinned
`nvidia/Qwen3.6-27B-NVFP4` checkpoint on NVIDIA Jetson AGX Orin (`sm_87`). It
explores what becomes possible when the model, numerical format, hardware,
execution plan, and serving boundary are engineered as one system instead of
treated as interchangeable layers.

The ordinary installed service now selects corrected whole-core Prefill and
ordered v7 Decode under profile `q3x.sm87.production.whole-core-service.v1`.
It serves nonempty text, chat and token-ID prompts, streaming and nonstreaming,
with outputs up to 4,096 tokens and `prompt + output - 1 <= 44,095`. Package
version is 0.8.0. The bounded service switch is complete; future 60K/130K
capacity and the locked latency targets remain open, so full product
`release_qualified` remains false. [Current Status](docs/CURRENT_STATUS.md)
owns the exact qualification, paired Prefill/Decode results and remaining gaps.

Qwen3x-Orin is an independent community project. It is not an official Qwen,
Alibaba, NVIDIA, or Jetson project and is not endorsed by those organizations.

## Why a specialized runner?

This project is not trying to become a universal inference framework. It
intentionally trades unrelated model and hardware compatibility for the
ability to co-design weight layout, state ownership, scheduling, kernels,
admission, observability, and API behavior around one deployment target. The
repeatable output is the engineering method and evidence chain; a delivered
binary may correctly reject every model or device outside its contract.

The intended deliverable is an externally callable OpenAI-compatible runner,
not a loader or a collection of fast kernels:

```text
OpenAI-compatible request
  -> tokenize and admit the complete request
  -> exact Prefill and state commit
  -> first useful token
  -> exact Decode
  -> streaming response, usage, and terminal state
```

Constraints flow downward from that boundary. Local mechanisms evolve inside
explicit work packages, compose into an architecture candidate, and matter
only when their value returns to the real API. The controlling design is
[`docs/SDD.md`](docs/SDD.md); the full engineering philosophy is in the
[`engineering constitution`](docs/ENGINEERING_CONSTITUTION.md).

## Proof contract

These are locked **targets**, not claims about current performance:

| Scope | Required outcome |
| --- | --- |
| Cold/no-cache 40K–60K prompt | First visible committed generated token within 2 seconds |
| Cold/no-cache approximately 130K prompt | First visible committed generated token within 4 seconds |
| Single-request Decode | At least 10 token/s without MTP |
| Accuracy | No Production numerical or generated-behavior regression |
| Route integrity | No silent truncation, hidden cache reuse, or undeclared fallback |
| Production dependency | No cuBLASLt dispatch, fallback, or runtime dependency |
| Competitive floor | First match, then exceed matched same-workload vLLM behavior |

The [`engineering constitution`](docs/ENGINEERING_CONSTITUTION.md) owns these
targets. The
[`EvalScope procedure`](docs/EVALSCOPE_EVALUATION.md) owns the exact external
measurement protocol, and [`Current Status`](docs/CURRENT_STATUS.md) owns the
current implementation and qualification facts.

## Functional evaluation quick start

This functional smoke path exercises building the current service,
loading the pinned model, generating text, and answering through its evaluation
adapter. It is **not** an accuracy validation, performance result, long-context
qualification, or Production release attestation.

### Requirements

- Jetson AGX Orin with an SM87-capable Linux/CUDA development environment;
- CMake 3.24 or newer;
- CUDA Toolkit 12.0 or newer with a CUDA C++17 compiler;
- a CMake-detectable system threading library;
- ICU 74 with the `uc` and `i18n` components; and
- the exact pinned `nvidia/Qwen3.6-27B-NVFP4` artifact described by
  [`docs/MODEL_SUPPORT.md`](docs/MODEL_SUPPORT.md).

The API smoke commands below also use `curl` as a client.

Keep the user-owned model directory read-only. Put every project-generated
build or artifact below the ignored `.q3x-work/` tree:

```bash
Q3X_BUILD="$PWD/.q3x-work/build/quickstart"
Q3X_MODEL_DIR="/absolute/path/to/nvidia/Qwen3.6-27B-NVFP4"

cmake -S . -B "$Q3X_BUILD" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=OFF \
  -DQ3X_CUDA_ARCHITECTURES=87
cmake --build "$Q3X_BUILD" --parallel \
  --target qwen3x-orin qwen3x-eval-server qwen3x-inspect
```

`BUILD_TESTING=OFF` excludes test-only admission paths; it does not by itself
make this a fully qualified product release. Fresh OFF builds select the sealed
whole-core service; test builds do not. Explicit historical reproduction must
set `Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=OFF`. The default-OFF
`Q3X_BUILD_P40_WHOLE_CORE_DEVELOPMENT_ROUTE` bundle. That bundle builds the
separately named, accuracy-unqualified
`qwen3x-eval-server-p40-v10-dev` baseline, requires its typed
`--development-route p40-whole-core-v10` selector, rejects ambient `Q3X_*`
controls, and disables installation; it is not a release or production
configuration. Inspect the ordinary binary and target device:

```bash
"$Q3X_BUILD/qwen3x-orin" version
"$Q3X_BUILD/qwen3x-orin" probe
"$Q3X_BUILD/qwen3x-orin" models
"$Q3X_BUILD/qwen3x-inspect" manifest "$Q3X_MODEL_DIR"
```

`models` reports catalogued descriptors; a catalog entry is not a runtime
support or qualification claim.

The diagnostic `generate` CLI retains its explicit legacy execution options;
use the server below for the production route. For a diagnostic smoke, select the SM87
projection backend and request the maximum 512-token Prefill chunk
capacity:

```bash
"$Q3X_BUILD/qwen3x-orin" generate "$Q3X_MODEL_DIR" \
  --prompt "用一句话解释统一内存。" \
  --max-tokens 16 \
  --prefill-chunk-size 512 \
  --projection-backend sm87
```

Install and start the production service in one terminal:

```bash
Q3X_INSTALL="$PWD/.q3x-work/install/quickstart"
cmake --install "$Q3X_BUILD" --prefix "$Q3X_INSTALL"
"$Q3X_INSTALL/bin/qwen3x-eval-server" "$Q3X_MODEL_DIR" \
  --host 127.0.0.1 \
  --port 18080 \
  --model qwen3.6-27b-nvfp4
```

After it becomes ready, exercise health and committed-token streaming from
another terminal:

```bash
curl -fsS http://127.0.0.1:18080/healthz

curl -N -fsS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"你好，请用一句话介绍你自己。"}],"max_tokens":16,"temperature":0,"stream":true}'
```

The ordinary server fixes corrected whole-core Prefill, ordered v7 Decode and
the full-range SM87 inventory; public tactic and arena overrides are rejected. The loopback command
above omits authentication. `--api-key-file` enables Bearer authentication for
models and generation; health remains public, and a non-loopback listener
requires an owner-only key file. TLS termination is external. Execution is
greedy and serialized at the GPU worker. Generation requests must explicitly provide a positive
`max_tokens` or `max_completion_tokens` within the configured ceiling and use
`temperature=0`. This is the bounded production service; full long-term product qualification
remains separate. See the
[`evaluation procedure`](docs/EVALSCOPE_EVALUATION.md) for supported request
semantics and reproducible EvalScope commands.

## How performance is judged

Architecture selection begins with the pinned real model on the real API path:

- EvalScope/user-visible TTFT selects the whole system; server-side pure
  Prefill timing explains it but does not replace it.
- NSys, NCU, component timing, and short prompts are attribution tools inside
  a named work package, not product-performance substitutes.
- Synthetic payloads are for exhaustive correctness and smoke coverage, never
  performance selection.
- Every Jetson timing run records a decision-class `tegrastats`, process, and
  GPU-device-handle preflight; the incomplete Jetson `nvidia-smi` view is not
  an idle-resource authority. Unowned GPU use or confirmed material contention
  invalidates timing. Other environment telemetry annotates ordinary
  engineering work, while architecture selection and release qualification
  use their strict predeclared envelope.
- Production paths preserve the declared numerical/state contract, exclude
  MTP from the current target, and keep cuBLASLt reference-only.

The normative rules are in
[`docs/REAL_MODEL_PERFORMANCE_POLICY.md`](docs/REAL_MODEL_PERFORMANCE_POLICY.md).
Historical measurements retain only their recorded protocol and do not become
current truth by appearing in the repository.

## Start from the right document

| Need | Start here |
| --- | --- |
| What works now? | [`Current Status`](docs/CURRENT_STATUS.md) |
| What is the complete runner design? | [`System SDD`](docs/SDD.md) |
| What should happen next? | [`Active Roadmap`](docs/ROADMAP.md) |
| How is the pinned model identified? | [`Model Support`](docs/MODEL_SUPPORT.md) |
| How do I run external evaluation? | [`EvalScope Evaluation`](docs/EVALSCOPE_EVALUATION.md) |
| How do I contribute or optimize safely? | [`AGENTS.md`](AGENTS.md), which routes to the [`documentation index`](docs/README.md) |
| Where is every Markdown document classified? | [`Document Registry`](docs/DOCUMENT_REGISTRY.md) |

The documentation index owns the required reading order and routes active
designs, contracts, decisions, immutable evidence, historical records, and
external source studies. The root README does not duplicate those authorities.

## Repository map

```text
include/q3x/       C++ API plus kernel, model, runtime, and internal contracts
src/core/          Device inspection and SHA-256 primitives
src/io/            Bounded JSON and safetensors parsing
src/quantization/  FP8 and NVFP4 format primitives
src/text/          Pinned tokenizer and chat/text preprocessing
src/model/         Model descriptors and checkpoint metadata
src/runtime/       Weight binding, request state, reference engine, and runner
src/kernels/       Reference and SM87-specialized CUDA kernels
src/server/        Native OpenAI-compatible service
tools/             Inspection, evidence, reference, and evaluation tools
tests/             Unit, numerical, route, and integration tests
benchmarks/        Pinned benchmark and EvalScope inputs
third_party/       Pinned upstream source subsets and their licenses
docs/              Governance, SDDs, contracts, plans, and evidence
.q3x-work/         Ignored project-owned builds, profiles, and artifacts
```

Before performance or architecture work, start with [`AGENTS.md`](AGENTS.md).
It routes contributors and Codex to [`docs/README.md`](docs/README.md). Local
tile, cache, stream, fusion, profiler, and benchmark rules have authority only
inside an explicitly active named optimization work package. User-owned model
directories, virtual environments, and shared caches remain external read-only
inputs unless the project owner explicitly requests otherwise.

## License and provenance

Project-authored code is distributed under the
[`Apache License 2.0`](LICENSE). The repository also contains pinned
upstream-derived components and adaptations from vLLM Marlin, FlashInfer, and
FlashLinearAttention; they retain their applicable notices and licenses. See
[`NOTICE`](NOTICE) and the notices beside vendored source. This provenance does
not introduce an external runtime backend or fallback.

Model weights, tokenizers, configurations, and generated artifacts are
distributed separately under their publishers' terms and are not covered by
this repository's Apache-2.0 license.
