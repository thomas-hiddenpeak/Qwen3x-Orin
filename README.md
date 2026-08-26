---
q3x_document:
  id: q3x-project-readme
  class: active
  status: active
  owner: project-maintainers
  authority: product introduction, bounded evaluation quick start, and high-level navigation
  effective: 2026-08-09
  last_reviewed: 2026-08-27
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

> **Project status — evaluation stage.** The repository has an implemented
> batch-one native runner and loopback OpenAI-compatible evaluation adapter.
> It does **not** yet have a qualified Production release or the final serving
> API, and the current default capacity does not admit the locked long-context
> workload. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) is the sole
> source for current capability, routes, performance, capacity, and blockers.

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

This functional smoke path exercises building the current development runner,
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
make this an attested release. The sole explicit exception is the default-OFF
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

Run one functional greedy generation while explicitly selecting the SM87
projection backend and requesting the maximum 512-token Prefill chunk
capacity:

```bash
"$Q3X_BUILD/qwen3x-orin" generate "$Q3X_MODEL_DIR" \
  --prompt "用一句话解释统一内存。" \
  --max-tokens 16 \
  --prefill-chunk-size 512 \
  --projection-backend sm87
```

Start the loopback evaluation adapter in one terminal:

```bash
"$Q3X_BUILD/qwen3x-eval-server" "$Q3X_MODEL_DIR" \
  --host 127.0.0.1 \
  --port 18080 \
  --model qwen3.6-27b-nvfp4 \
  --max-sequence-length 4096 \
  --max-output-tokens 256 \
  --prefill-chunk-size 512 \
  --projection-backend sm87
```

After it becomes ready, exercise health and committed-token streaming from
another terminal:

```bash
curl -fsS http://127.0.0.1:18080/healthz

curl -N -fsS http://127.0.0.1:18080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-27b-nvfp4","messages":[{"role":"user","content":"你好，请用一句话介绍你自己。"}],"max_tokens":16,"temperature":0,"stream":true}'
```

The adapter is unauthenticated, loopback-only, greedy, and serialized at the
GPU worker. Generation requests must explicitly provide a positive
`max_tokens` or `max_completion_tokens` within the configured ceiling and use
`temperature=0`. It is an external-evaluation instrument, not the final
Production API. See the
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
src/server/        Loopback OpenAI-compatible evaluation adapter
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
