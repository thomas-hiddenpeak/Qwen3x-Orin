---
q3x_document:
  id: q3x-project-readme
  class: active
  status: active
  owner: project-maintainers
  authority: product introduction and repository documentation entry
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: concise project introduction; dynamic state remains in docs/CURRENT_STATUS.md
  review_trigger: mission, target scope, product boundary, or controlling-document change
---

# Qwen3x-Orin

Qwen3x-Orin is an experimental, pure C++/CUDA runner specialized for selected
Qwen3.5/Qwen3.6 ModelOpt checkpoints on NVIDIA Jetson AGX Orin (`sm_87`). Its
first proof vehicle is the pinned Qwen3.6-27B-NVFP4 dense text model.

The project is not trying to become a universal inference framework. It
explores a repeatable engineering paradigm in which one model family, one
numerical representation, one hardware target, and one runner are co-designed
as a complete product. The reusable result is the method and evidence chain;
the deployed binary may deliberately reject unrelated models and devices.

> Qwen3x-Orin is an independent community project. It is not an official Qwen,
> Alibaba, NVIDIA, or Jetson project and is not endorsed by those organizations.

## Delivery boundary

The product is an externally callable OpenAI-compatible runner, not a loader
or a collection of kernels. System design starts at the request and first
useful response, then propagates constraints down through admission, Prefill,
Decode, state, memory, scheduling, operators, weight layouts, and kernels.

The implementation evolves in the opposite direction: bounded local changes
compose into a runnable architecture candidate, and their value must propagate
back to the real API. Local component evidence remains essential, but only the
whole runner can become production.

The governing rule is:

> Constraints leak downward from the final runner; variation occurs locally;
> mechanisms compose in an architecture candidate; value must leak upward to
> the real API.

## Locked proof goals

For the pinned dense model on Orin, the active product contract requires:

- cold, no-cache first response within two seconds for 40K--60K prompt tokens;
- cold, no-cache first response within four seconds for the pinned
  approximately 130K prompt contract;
- at least 10 token/s single-request Decode without MTP;
- no production accuracy or generated-behavior regression;
- no silent truncation, hidden cache reuse, or unplanned fallback;
- no cuBLASLt production dependency; and
- useful performance that first matches and then exceeds matched vLLM.

Exact workload, capacity, route, release, and evidence definitions live in the
canonical documents below. Current numbers are intentionally not duplicated
in this README.

## Current maturity

The repository contains an authenticated resident-weight loader, tokenizer,
fixed 64-layer dense runner, exact reference and optimized SM87 paths,
deterministic generation fixtures, a bounded request arena, and a loopback
OpenAI-compatible evaluation adapter.

It does **not** yet have a qualified production release artifact or a final
production serving API. In particular, the evaluation adapter, test-admitted
kernel compositions, historical P513 measurements, and research-only
approximate paths must not be presented as the delivered runner.

See [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) for the only current
support, default-route, performance, capacity, and blocker snapshot.

## Documentation

Start at [`docs/README.md`](docs/README.md). Before performance or architecture
work, read the controlling documents in this order:

1. [`docs/ENGINEERING_CONSTITUTION.md`](docs/ENGINEERING_CONSTITUTION.md) —
   mission, hard constraints, and engineering philosophy;
2. [`docs/SDD.md`](docs/SDD.md) — API-first system design;
3. [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md) — current implementation
   and evidence truth;
4. [`docs/REAL_MODEL_PERFORMANCE_POLICY.md`](docs/REAL_MODEL_PERFORMANCE_POLICY.md)
   — evidence and candidate promotion rules;
5. [`docs/PREFILL_ARCHITECTURE_RESET.md`](docs/PREFILL_ARCHITECTURE_RESET.md) —
   active Prefill subsystem design; and
6. [`docs/EVALSCOPE_EVALUATION.md`](docs/EVALSCOPE_EVALUATION.md) — external
   performance and capability protocol.

The documentation index then routes governance, the sole active
[`Roadmap`](docs/ROADMAP.md), component contracts, decisions, immutable
evidence, and external references.

Every tracked Markdown file, including historical evidence and third-party
notes, is classified in
[`docs/DOCUMENT_REGISTRY.md`](docs/DOCUMENT_REGISTRY.md). Its authority and
maintenance rules are defined in
[`docs/DOCUMENT_GOVERNANCE.md`](docs/DOCUMENT_GOVERNANCE.md).

## Target scope

| Target | Role | Status owner |
| --- | --- | --- |
| Qwen3.6-27B-NVFP4 dense text | First specialized-runner proof | Current Status |
| Qwen3.5/Qwen3.6 35B-A3B MoE | Later proof after dense release gates | Roadmap |
| Jetson AGX Orin SM87 | Production hardware target | SDD |

Text-only, batch-one correctness and the locked single-request performance
contract remain the first scope. MTP, vision, speculative decoding, generic
multi-model serving, tensor parallelism, and other hardware are outside the
current proof unless the project owner amends the contract.

## Repository layout

```text
include/q3x/       Public runtime interfaces
src/core/          Tensor, allocation, device, and runtime primitives
src/text/          Pinned tokenizer and chat/text preprocessing
src/model/         Model descriptors, metadata, and graph binding
src/runtime/       Resident weights, execution plans, state, and runner
src/kernels/       Reference and SM87-specialized CUDA kernels
src/server/        OpenAI-compatible evaluation/server boundary
tools/             Inspection, packing, evidence, and evaluation tools
tests/             Unit, numerical, route, and integration tests
benchmarks/        Pinned benchmark and EvalScope inputs
docs/              Governance, SDD, contracts, plans, and evidence
.q3x-work/         Ignored project-owned builds, profiles, and artifacts
```

Project-generated artifacts belong under `.q3x-work/`. User-owned model
directories, virtual environments, and shared caches are external read-only
inputs unless the project owner explicitly requests otherwise.

## Build and use

Build, release, model, API, and evaluation commands are deliberately owned by
their applicable contracts rather than repeated here:

- model and checkpoint compatibility: [`docs/MODEL_SUPPORT.md`](docs/MODEL_SUPPORT.md);
- current system and release status: [`docs/CURRENT_STATUS.md`](docs/CURRENT_STATUS.md);
- external API evaluation: [`docs/EVALSCOPE_EVALUATION.md`](docs/EVALSCOPE_EVALUATION.md);
- request state and capacity: [`docs/REQUEST_STATE.md`](docs/REQUEST_STATE.md);
- deterministic reference engine: [`docs/REFERENCE_ENGINE.md`](docs/REFERENCE_ENGINE.md).

Do not infer production readiness from a successful component benchmark or
test-only CMake admission. The SDD release contract and Current Status are the
authority.

## Engineering boundaries

- Production changes preserve model accuracy and the declared numerical/state
  contract.
- MTP cannot satisfy the current Prefill or Decode targets.
- cuBLASLt is a numerical and performance reference only and is unreachable
  from production.
- Synthetic data supports correctness and smoke coverage, never performance
  selection.
- Performance work begins with a clean-host, real-model, real-API observation.
- Local mechanism rules apply only inside an explicitly active local
  optimization work package with a bounded composition point.
- vLLM, FlashInfer, Triton, FlashLinearAttention, Mamba, Humming, and
  qwen35-thor are architecture references and existence proofs, not required
  production dependencies.

## Source and license boundaries

Qwen3x-Orin is original work licensed under the
[Apache License 2.0](LICENSE). Design study does not imply that upstream code
is present. Source copied or adapted from another project must retain required
copyright/license notices, identify the upstream revision, and update
[`NOTICE`](NOTICE) where required.

Model checkpoints, tokenizers, configurations, and generated artifacts are
distributed separately under their publishers' terms and are not covered by
this repository's Apache-2.0 license.
