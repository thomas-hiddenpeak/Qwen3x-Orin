---
q3x_document:
  id: q3x-decode-exact-performance-20260927
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: bounded exact Decode performance successor
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: freeze on work-package closeout
---

# Exact Decode performance successor

Source parent: `c0318dd`. The owner requests performance work after the
[ordered repair](../decode-ordered-repair-2026-09-27/README.md).
`WP-DECODE-EXACT-PERFORMANCE-20260927` belongs to the
[active Decode architecture](../../ROADMAP.md#2026-09-27-product-convergence--active).
The actual P40000/O256 incumbent is 135.888518 ms/token (7.358973 tok/s).
The interim goal remains approximately 8.55 tok/s; strict scalar numerical
identity, BF16 KV, non-MTP scope and existing Prefill arithmetic remain fixed.

## Actual Decode attribution

One fresh API P40000/O16 request captures the first Decode NVTX range at
S40001. All 437 kernels use one CUDA stream. Kernel time sums to 136.255616 ms
inside a 136.819552 ms kernel span, leaving 0.563936 ms between kernels.
This is diagnostic attribution, not an unprofiled throughput result.

| Family | Calls | Summed device ms |
| --- | ---: | ---: |
| Grouped-register QK | 16 | 13.202496 |
| Ordered PV | 16 | 17.057216 |
| Scalar softmax | 16 | 2.836864 |
| Other kernels | 389 | 103.159040 |

A privileged NCU retry passes the previous HWPM permission boundary, but
whole-model kernel-replay backup encounters NvMap allocation error 12 and
the request disconnects before a response. No counters are accepted. The
original failed run, cleanup and memory observations remain retained; no
driver configuration is changed. A later bounded replay of only the retained actual operands successfully
collects hardware counters. QK v6 has 49.19% achieved occupancy, 72 registers
and a long-scoreboard ratio of 3.97 cycles per 8.79 issue cycles (45.15%).
PV v5 has 24.96% achieved occupancy (11.98 active warps/SM), 68.01% L1
data-pipe wavefront throughput, and shared/issue dependencies. These are
local denominators, not additive percentages of API latency. Profiling does
not lock or modify clocks; observed SM frequency is 1.29--1.30 GHz.

The remedy is to retain actual operands
already copied by the numerical observer, so a bounded local comparison does
not need the entire resident model or a whole-model replay backup.

## Numerical and ownership ledger

QK v6 retains the exact BF16 products and original FP32 reduction tree. At
stride s, only lanes below s remain ancestors of lane zero. Updating dead
lanes unconditionally removes conditional merge work without changing any
live ancestor. Six independent heads are interleaved at each reduction stage.
Lane-zero results first enter a 6 by 128 FP32 shared tile, then a CTA barrier
permits coalesced publication of adjacent positions. Invalid tail cells are
never read. The new shared publication adds no rounding.

The QK compiler record uses 72 registers, 3,072 shared bytes, no stack or
spill, versus 72 registers and no shared memory for v5. Source-level score
store coalescing reduces a worst-case 32-byte sector per 4-byte result;
this is a logical transaction bound, not a measured DRAM-byte claim. Whole
static SASS instruction count is not a latency predictor because publication
and loop paths have different dynamic execution counts.

The prepared PV correction uses four physical buffers and keeps up to three
future copy groups in flight. The current consumer and new producer always
own different buffers. A CTA barrier protects reuse, and wait-group counts
shrink at the final tiles so the next consumer always observes completed
copies. The sequence interval starts at 512, so three initial 64-row tiles
always exist. Each output retains the exact increasing-position FMA chain
and final BF16 rounding. It uses 40 registers, 22,528 shared bytes, no stack
or spill. Real-payload comparison decides whether this correction is retained.

Raw protocols, binaries, source bridges and output evidence live under
`.q3x-work/decode-exact-performance-20260927/`. The final source, binary and raw-artifact hashes are recorded in the
[closeout metadata](../../metadata/qwen36-27b-decode-exact-performance-2026-09-27.json).
This evidence grants no production or release authority.

## Bounded real-payload composition

The numerical test exports the already host-copied Q/K/V of the first step at
layers 3, 23, 43 and 63. Before timing, all three complete scalar trajectory
comparisons pass and all raw input sizes/SHA-256 values are pinned. One helper
links v5, v6 and the proposed successor in separate namespaces. Each of four
layers runs six mirrored B-C-C-B rounds with five complete Attention launches
per event interval, comparing every output and FP32 probability bit first.
This fixed-depth panel is local diagnostic evidence; the complete API supplies
all-layer product selection.

QK v6 reduces full-Attention local time by approximately 9% across all four
layers. Merely deepening PV prefetch reduces that local time by a further
3.2--3.6%, positive in every round but insufficient as a complete response.
The counter-informed v7 composition therefore stages K with one 16-byte
asynchronous transfer per thread and retains four K buffers. Every warp still
uses its original live product/add tree, including masked tails. PV instead
loads two adjacent BF16 dimensions in a packed shared read and reuses each
probability for two independent ordered accumulators. QK uses 78 registers
and 19,456 shared bytes; PV uses 40 registers and 38,912 shared bytes. Neither
spills or uses a stack.

The composed local time is 1.5558--1.5563 ms per layer versus v6's
1.7710--1.7743 ms, positive in every paired round. Outputs and probabilities
remain exact. Compared with v5's approximately 1.96 ms in the first local
panel, this is approximately 20% less local Attention time, not a claimed
20% API speedup. The composition completes the whole-model numerical and
actual API
gates below; only the API comparison establishes product-path value.

## Completed API return and decision

The frozen v7 admission retains strict scalar equality and improves actual
P40000/O256 Decode from a fresh v5 control's 7.338505
to 7.877653 tok/s (7.35% higher throughput). An independent v7 process
returns 7.876981 tok/s with identical request,
text and usage. The approximately 8.55 tok/s interim goal is **not met**:
the first v7 run still needs approximately 9.98 ms/token removed.
Close the bounded work package and retain v7 only in the isolated admission.
Default production dispatch and whole-core Prefill composition are not selected.

| Prompt / output | Fresh v5 Decode tok/s | v7 Decode tok/s | v7 engine Prefill tok/s | v7 external TTFT s |
| --- | ---: | ---: | ---: | ---: |
| 1089 / 32 | 9.5894 | 9.6132 | 258.252 | 4.218471 |
| 8192 / 32 | 9.0837 | 9.2516 | 242.973 | 33.721587 |
| 40000 / 256 | 7.3385 | 7.8777 | 182.242 | 219.511265 |

These are one Legacy-C512 admission API route and the same frozen token-ID
requests, with LAZY module loading, no prefix reuse, greedy non-MTP generation
and concurrency one. Each process records idle ownership, sanitized telemetry,
successful cache preparation and owned shutdown. This is a development
direction comparison with an independent target repeat, not mirrored repeated
release qualification. Short O32 and long O256 rates retain their different
output lengths; they are not a controlled fixed-output-length scaling fit.

Engine Prefill is the server's `engine_prompt_prefill` interval, including the
final prompt token and first-token readiness. It is not pure kernel Prefill.
External Decode uses `(last content time - first content time)/(O-1)`;
every response has exactly one token per nonempty content event, exact usage,
`length` finish and `[DONE]`. The metadata retains p50/p95/p99/max inter-content
latency and separate engine intervals. No new Prefill arithmetic or speedup is
claimed, and this panel does not borrow the separate whole-core profile's TTFT.

## Numerical, protocol and isolation closure

- P576/P8192/P40000 O16 each preserve all 84 raw spans, complete live-state
  metadata, all logits and all 256 observed Attention outputs exactly against
  the frozen scalar trajectory. KL and relative-L2 error are zero. These are
  bounded deterministic checks, not a universal model-capability certification.
- T1 covers bounds, guards, mapping, non-finite propagation, Graph replay,
  and exact scalar probability/output bits at nine lengths, including 512,
  513, 575, 576, 577, 2047, 8192, 40000 and 44095.
- All three API requests preserve v5 text and exact usage. The independent
  40K repeat also preserves both. The pinned 20-case C-Eval-derived screen
  remains 15/20, all parseable, with scalar-identical answers, text, request
  hashes and usage. It is not the full public suite.
- Malformed input returns HTTP 400. Disconnect after three content events
  recovers with matching text/usage and a witnessed conservative full reset.
- Ordinary OFF preprocessing and public scalar source are unchanged. The
  rebuilt ordinary server is byte-identical to the previous default
  (`7ac5a7fa211a6d09c1087a1033d59df9d542346be2e4c11c7609ca996767fd0f`),
  with no admission or cuBLASLt symbols/dependency. This isolation proof does
  not newly qualify the previous default's performance.

The first OFF isolation check caught an overly broad schema-literal edit in
inactive non-admission branches. Those literals were restored; the final check
passes. The corrected admission ELF remains byte-identical to the v7 ELF used
for numerical and API tests. Initial manual host-test invocations had
incomplete link libraries, omitted
the fixture-directory argument, and then incorrectly applied testing-inventory
expectations to an OFF gateway. Those failed invocations are retained. The
corrected ordinary testing gateway (without fused admission) and the dedicated
admission protocol test pass separately; OFF isolation is established by the
unchanged binary/preprocessing proof above, not by the mismatched test.

## Lessons and remaining delivery gap

Local Attention savings do reach the API, but do not transfer one-for-one:
Attention is only part of the step. The first-step trace rules out a large
launch-gap explanation for this workload; it does not establish a universal
hardware limit. The successful small replay shows that a profiler's
whole-model replay allocation failure is a tooling-footprint issue, not
evidence that hardware counters are unavailable or performance is exhausted.

The next delivery constraint remains an exact P40 API step around 116.96 ms.
Further work must explain the composed v7 critical path and remove a measured
operand-feed/residency dependency under a newly bounded package. Do not reopen
approximate reductions, reuse the Prefill-only numerical waiver, add local
percentage gains into an API claim, or resume a tile/stage parameter sweep.
Release packaging and whole-core composition remain conditional downstream
work, not completed by this admission result.


The first completed API return, including its capability/recovery closeout,
arrives within approximately 30.06 minutes of conservatively accounted process
wall time, inside the 40-minute return boundary. Including the conditional
fresh control and independent target repeat gives approximately 40.17 minutes;
this upper accounting includes shutdown and monitor teardown. The raw budget
record keeps both quantities separate. A repeat preflight initially observes
systemd-journald at 6.00% of one CPU over 1.17 seconds, with no GPU holder;
no model starts on that attempt and no system service is stopped. The fresh
preflight and final postflight qualify only active Codex control CPU activity.
Final GPU ownership is empty; all model processes exit successfully except
the separately recorded failed whole-model NCU attempt.
