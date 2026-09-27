---
q3x_document:
  id: q3x-product-readiness-audit-20260928
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: production-boundary source audit and host-only reproductions at 8c0eb23
  effective: 2026-09-28
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: dated non-performance product audit findings and their evidence limits
  review_trigger: successor fixes or new verification recorded separately
---

# Product readiness audit — 2026-09-28

Source: `8c0eb23f9f7027d46d41b57289f47f2438208c05`. The owner has paused
performance optimization. This audit examines the current production entry,
protocol/lifecycle boundaries, build/test entry points, release packaging,
reproducibility and maintenance. It is not an exhaustive CUDA-kernel audit,
a new numerical qualification or a security certification. No model execution,
performance measurement, hardware-fault injection or runtime modification was
performed.

The default service remains the corrected whole-core Prefill plus ordered v7
Decode described in [Current Status](../../CURRENT_STATUS.md). The earlier
bounded numerical and API qualification is not withdrawn by this audit.
However, that panel did not cover every product boundary below.

## Findings and order

P1 means address before treating the service as unattended production. P2
means a bounded follow-up for operational reliability or reproducible delivery.
These priorities do not activate performance optimization.

| ID | Priority | Finding | Evidence strength |
| --- | --- | --- | --- |
| A1 | P1 | A valid output token cap can become a UTF-8 encoding failure | Exact gateway callback reproduced on host; tokenizer token confirmed |
| A2 | P1 | Unauthenticated incomplete connections can occupy all HTTP workers and block health checks | Actual ingress workers reproduced with sockets, no model |
| A3 | P1 | The release test preset succeeds with zero tests | Executed CTest, exit 0 and no tests |
| A4 | P2 | Readiness tracks inventory, but not all fatal inference/reset failures | Source-confirmed missing error-to-health transition; no GPU fault injected |
| A5 | P2 | Complete qualification cannot be reproduced from a clean clone using tracked harnesses alone | Production manifest points to ignored executable scripts |
| A6 | P2 | The install tree lacks a license/notice bundle | Install rules and exact installed tree inspected |
| A7 | P2 | Sustained service and broader generation behavior remain outside the qualified panel | Coverage gap, not a newly demonstrated model defect |

### A1 — Token limits must not turn valid generation into encoding failure

[The streaming observer](../../../src/server/evaluation_server.cpp) accumulates
raw token bytes, but at the last permitted token it rejects any incomplete
UTF-8 suffix (`observe_gateway_token`, lines 1141–1148 at the audited commit).
`execute_job` converts this to `stream_encoding_error`; nonstream responses
have the analogous complete-UTF-8 check at lines 1622–1632.

A byte-level vocabulary does not promise one Unicode character per token.
The pinned tokenizer contains token **160**, whose decoded byte is `0xe4`, a
valid start of a three-byte character. With `max_tokens=1`, feeding that decoded byte
to the production callback returns false and marks a protocol failure. Feeding
the complete three-byte character instead passes. The failure therefore does
not require corrupt weights, invalid token IDs or malformed client input.
Depending on when streaming started, the client receives an HTTP error or a
failed SSE response rather than a normal `length` completion.

Fix boundary: define one shared replacement/flush policy for terminal partial
byte sequences, preserve actual generated token IDs and usage, and test both
stream/nonstream and a multibyte character split across successive tokens.
This is text serialization repair, not a change to model arithmetic. The audit
does not claim a measured incidence on free-running prompts.

### A2 — The control-plane worker is not reserved

The service fixes three HTTP workers and a 10,000 ms read timeout in
[EvaluationServerOptions](../../../include/q3x/server/evaluation_server.h).
[The ingress loop](../../../src/server/evaluation_server.cpp) assigns a worker
to each connection before the HTTP body is complete and before Bearer
validation. Health requests enter the same queue and pool. The intended
extra worker for control/overload responses is therefore only a capacity
assumption, not enforced isolation.

The host reproducer starts the actual three ingress workers, with a configured
key, and supplies three idle unauthenticated sockets followed by a valid health
request. The health socket has no response after 750 ms. Closing one idle
socket immediately allows HTTP 200. The production read timeout is ten
seconds; the reproduction releases the socket early rather than waiting for
that timeout. No GPU work is involved.

Impact: partial uploads or stalled clients can delay readiness/overload
responses even with an idle GPU. A reverse proxy may reduce exposure, but the
repository provides no tested proxy policy that establishes that protection.
Fix boundary: separate bounded HTTP parsing from inference waits, preserve
control-plane availability, and test slow/incomplete clients. Increasing the
thread count alone does not establish the intended guarantee.

### A3 — A successful release test command currently means no tests ran

`ctest --preset orin-release` returns **0**, printing `No tests were found!!!`.
The [preset](../../../CMakePresets.json) points to a `BUILD_TESTING=OFF` build;
[CMake](../../../CMakeLists.txt) also forbids enabling testing on the sealed
service. Most registered tests exercise a different build profile. No
repository-tracked GitHub workflow runs a release validation lane.

Keeping admission hooks out of production is appropriate. The missing piece
is an external test driver targeting the installed Release artifact, with
nonzero-test enforcement and a clear split between host checks and explicitly
requested real-model checks. The recently added production help check is useful
but remains a manually invoked script; it does not close this gap.

Fix boundary: a documented host-test preset and an installed-service black-box
lane that fails if it runs nothing. Do not enable test-only hooks in production
just to obtain a green CTest result.

### A4 — Fatal compute failures do not consistently close readiness

`EvaluationProductionRuntimeHealth::observe` accepts only load/inventory stats
and closes admission on inventory mismatch. Graph-cache retirement changes
those stats and is covered. However, `execute_job` publishes generic engine
errors without passing the failure diagnostic into the health latch.
`ReferenceEngine::generate_prompt_token_ids` can return immediately after
`begin_ordinary_request` or transaction rollback fails, without retiring the
load inventory.

A fatal CUDA/reset error that leaves inventory unchanged can therefore leave
`/healthz` reporting ready while generation keeps failing. This is a
source-confirmed gap in the failure contract; the audit did not induce a real
sticky CUDA error and does not claim one has occurred in qualified runs.

Fix boundary: distinguish request errors, recoverable cancellation and
unrecoverable device/reset errors; latch only the latter unhealthy and provide
an explicit supervisor/restart policy. Add host fault-injection coverage of
that classification before any destructive hardware test.

### A5 — Successful one-off qualification is not a reusable release gate

The [production record](../../metadata/qwen36-27b-whole-core-service-production-2026-09-28.json)
points to `run_service.py`, `run_service_pairs.py` and the environment protocol
under ignored `.q3x-work/prefill-convergence-20260928/`. The tracked
[validator](../../../tools/evaluation/validate_whole_core_service.py) audits
existing records; it does not create the complete capability, cancellation,
queue, long-output and mirrored-request panel.

The existing evidence is still available on this workstation. The problem is
that a clean clone cannot repeat the release procedure from versioned tools
alone. Fix boundary: version the reusable runner and protocol definitions,
parameterize model/artifact paths, and keep large raw data and outputs ignored.
Preserve original evidence rather than rerunning it merely to reorganize files.

### A6 — Source notices do not accompany the installed package

The complete install rules copy binaries, libraries, public headers and CMake
exports, but no root `LICENSE`, `NOTICE` or third-party license bundle. A scan
of the qualified install prefix finds no files named LICENSE/NOTICE. For
example, compiled CUTLASS headers retain their BSD notice in the source tree,
which is not present in the binary install tree.

This is a packaging defect to resolve before distributing a standalone install
archive. Add versioned notices and provenance to the install manifest and test
the installed contents. It does not require changing the inference backend.

### A7 — Coverage must match the intended deployment

The qualified capability panel is 98 questions across four subjects, and the
long-output run is one P40000/O4096 request. Those are meaningful bounded
checks, but do not establish repeated hours-long request reuse, repeated
cancel/recover cycles, slow consumers, deployment restart behavior, or broad
multilingual free generation. A1 is a concrete example of a missing boundary.

The next reliability panel should be finite: supported endpoint/stream modes,
Unicode output caps, repeated request/cancellation cycles and stable memory,
then one defined unattended-service duration. Broader public capability can be
added separately. This audit does not prescribe unlimited retesting or claim
that the numerical repair failed.

## Maintenance debt

The audited source has 11,711 lines in `reference_runner.cpp`, 9,449 in
`reference_engine.cpp`, and 3,626 in CMakeLists. Production and historical
admission branches share these files and many build macros. File size alone
is not a bug, but together with A3/A5 it makes route-specific regression hard
to detect. First make the production tests reproducible; then extract stable
boundaries in small behavior-preserving changes. Do not combine cleanup with
another kernel redesign or delete historical evidence indiscriminately.

## Evidence and reproduction

The [retained probe source](gateway_probe.cpp) includes the audited gateway
translation unit, uses its production compile definitions, and calls the real
callback and ingress-worker implementations. It uses local socket pairs and
no model. It deliberately asserts the observed bugs; it is frozen audit
evidence, not a future passing regression test after repairs.

Raw build command/logs, probe output and file hashes are in
`.q3x-work/product-audit-20260928/`. The compact
[check record](checks.json) retains outputs and hashes. Existing installed
libraries came from `.q3x-work/build/whole-core-service-release-20260928`.
The probe does not establish a hardware-fault result or replace real-model
qualification. No runtime code was changed by this audit.

Suggested closure order: **A1/A2 → A3/A5 → A4/A6 → bounded A7**, while
performance work stays paused and its owner-set targets remain recorded.
