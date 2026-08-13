---
q3x_document:
  id: q3x-prefill-target-aot-p40-failed-api-2026-08-14
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: failed real-model P40 API smoke and architecture-closure evidence for the exact target-AOT v1 composition
  effective: 2026-08-14
  last_reviewed: 2026-08-14
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: erratum supported by retained raw evidence
---

# Exact target-AOT P40 failed API smoke

Date: 2026-08-14

This record closes the first executable composition of
`AC-PREFILL-SM87-AOT-SYSTEM-v1` as a performance candidate. The route is kept
default-off as an exact ownership, correctness, and diagnostic control. It is
not a production route and produced no performance result.

## Frozen identity and protocol

- implementation: `fad42f7` (`sm87-target-aot-p40`, explicit test admission);
- server ELF SHA-256:
  `c682fc683e0233086cc0d276385c731c4b612afd8c2d8ba8ee3fc136d60e1922`;
- server ELF build ID: `078c2c3fa6ca7af9687f457be750d996231ba250`;
- model: authenticated `Qwen3.6-27B-NVFP4` checkpoint;
- API: loopback OpenAI-compatible `/v1/completions`;
- request: exactly 40,000 prompt tokens and one output token;
- corpus SHA-256:
  `8970ac50693f49d1b27d35a0610ecbe5072594330d69b301f4dab731789b6844`;
- request arena: 5,075,652,608 bytes;
- excluded routes: MTP, cuBLASLt, JIT, request-time repack/autotune,
  approximate arithmetic, legacy fallback, and graph replay; and
- host gate: the retained request preflight was accepted with maximum GR3D
  utilization `0%` and a complete GPU-device-handle audit.

The local raw records are retained under
`.q3x-work/evidence/prefill-target-path/target-aot-p40-api-r2/`. They include
the host preflight and curl transaction record. The ignored artifact tree is
not itself authoritative; the immutable facts required for the decision are
reproduced here.

## Observation

The client timed out after `840.000399 s` with:

- HTTP code `0`;
- time to first byte `0`;
- response bytes `0`;
- curl exit code `28`; and
- no generation receipt, final-state result, logit/token oracle, or completed
  API transaction.

The GPU remained active after the client disconnected. A later unsupported
debugger attach attempt terminated the already-failed server process, so no
completion time can be inferred. The transaction is therefore a failed smoke
and a censored TTFT lower bound only. It must not be reported as token/s,
compared statistically, or used to replace the 392.804397-token/s incumbent.

## Causal architecture audit

Static source and compiled-resource inspection found no global lock,
cooperative-grid barrier, unbounded loop, or non-uniform CTA barrier that
would establish a CUDA software deadlock. The executor instead submits all
64 layers to one stream and performs its only host-visible completion wait at
the terminal token handoff.

The complete work ledger exposes three structural serialization classes:

| family | exact v1 physical ownership | whole-request work indicator |
| --- | --- | ---: |
| full Attention | 7,500 CTA per layer, one high-resource CTA/SM | 75,080,000 causal KV-tile iterations across 16 layers |
| exact GDN | 16 owner CTA per layer; each owner serializes 40K tokens and three value heads | 92,160,000 value-head/token updates across 48 layers |
| NVFP4/FP8 projections | 16 persistent CTA, one CTA/SM, one universal M128N256K64-style skeleton | about 1.951 PF of BF16 tensor-core work in the current exact mapping |

Compiled main-kernel footprints reinforce the ownership result: Attention
uses 254 registers/thread and about 128 KiB shared memory; NVFP4 Gate/Down use
246/210 registers and about 75 KiB; FP8 roles use 197--205 registers and about
96 KiB; GDN uses 198 registers and about 100 KiB. Each of those families is
limited to one resident CTA/SM in this composition.

The API liveness boundary is also incomplete. The target route rejects the
existing Prefill cancellation callback, the server binds that callback only
for the legacy whole-request mode, and only the GDN kernel observes the
request cancellation word. The continued GPU activity after disconnect is
therefore an implementation defect, not a client measurement artifact.

## Decision

`AC-PREFILL-SM87-AOT-SYSTEM-v1` is rejected as a performance composition. No
unchanged P40 rerun and no tile, cache, stage, or launch-parameter scan is
authorized on this skeleton. The exact chain remains useful for ownership,
state, artifact, and numerical-control development only.

Its successor must change the complete dataflow before returning to the API:

1. provide layer/operator progress plus request cancellation with bounded safe
   points before another target-length run;
2. split exact GDN ownership across the 48 independent value-head chains and
   move convolution, Q/K normalization, and alpha/beta/gate preparation into
   bounded token-parallel producers while retaining per-token BF16 state
   publication;
3. give Gate/Up, K-heavy Down, and FP8 roles separate AOT macro-tile,
   load/decode/MMA, and pipeline plans rather than a universal persistent
   skeleton; and
4. retain the current one-pass Attention as a control while translating the
   frozen FlashInfer/FlashAttention effective-Q producer/consumer plan that
   reuses staged K/V across more Q work.

The first complete successor returns directly to the clean-host, real-model
P40 API gate. Only a successful response can open external EvalScope timing,
correctness qualification, P60, or approximately-P130 work.
