---
q3x_document:
  id: q3x-current-status
  class: active
  status: active
  owner: project-maintainers
  authority: current implementation, qualification, production, metric, and blocker snapshot
  effective: 2026-08-12
  last_reviewed: 2026-09-26
  supersedes: []
  superseded_by: []
  ssot_for: current delivered state and open production gaps
  review_trigger: any default route, capability, qualification, metric, release, or blocker change
---

# Qwen3x-Orin current status

Snapshot date: 2026-09-09.

This page is a replaceable state snapshot. It does not own architecture,
delivery order, or experiment history. The system design is in
[`SDD.md`](SDD.md), the only active dependency order is in
[`ROADMAP.md`](ROADMAP.md), and exact observations remain in their linked
metadata/evidence records.

## 1. Answer-first state

**Liveness is integrated and installed-main closeout passes; release remains unqualified. The whole-core route is now a second, accuracy-qualified production deployment profile, and both production profiles passed a comprehensive API e2e evaluation (2026-09-26).**
Main `8fe4e67` (2026-09-26) promotes the layer-major whole-core route to the
sealed production profile `q3x.sm87.production.p40.whole-core.v1`
(`kP40WholeCoreV1`), selected at process start with
`--production-profile p40-whole-core-v1` from the
`orin-p40-whole-core-prod` build. It is fixed P40000/O16 geometry
(40'016 max sequence, 8'641'684'992-byte request arena) and trades the
default profile's 4'096-token output ceiling for a ~2.4x faster
whole-prompt Prefill on the pinned workload. Its accuracy qualification is
the full-state committed-state comparison against the accepted BF16 class
(ADR-0002, 2026-09-26); real-model verification produced output bitwise
identical to the development route with the witness
`numerical_contract.qualified=true`. The default route, the Legacy-C512
production plan, and the development route are unchanged; this profile
grants no release qualification and no capability beyond the pinned
P40000/O16 streaming contract.

A comprehensive API e2e evaluation (2026-09-26, all traffic through the
OpenAI-compatible HTTP API, EvalScope 1.9.1 for performance) confirmed both
production profiles on the real model. Legacy-C512 (default, no selector):
short protocol 32/32 at 2,743.8 ms mean TTFT / 105.1 ms mean TPOT / 9.51 decode
tok/s, matching its 2026-09-09 baseline; P40000/O16 at 219.5 s TTFT / 3.94
decode tok/s (decode identical to the 09-09 baseline; the lower TTFT reflects
33 preceding short requests having warmed the cuBLASLt/prefill paths, whereas
09-09 was the first request after server start). Whole-core
(`--production-profile p40-whole-core-v1`): P40000/O16 at 91.4 s TTFT /
419.75 prompt tok/s / 3.87 decode tok/s, a ~2.3x prompt-throughput gain over
the warmed Legacy profile on the pinned workload, with short requests rejected
by its profile contract as designed. Accuracy: the two routes produce coherent
but token-divergent greedy output on the same P40000 request (different GEMM
backends), each matching its own qualified lineage. Capability: the whole-core
contract matrix passed 6/6 and the Legacy generic-protocol matrix passed 21/21.
This evaluation is an observation surface; it does not promote or demote any
production route. Evidence: `.q3x-work/evidence/comprehensive-eval-20260926/`.

A proposed [ADR-0003](decisions/0003-prefill-target-hardware-bound.md) records
the hardware-bound analysis of the locked Prefill targets against the measured
Orin SM87 ceiling (41.9 TFLOPS mma ceiling; 52.5 s FLOP floor for P40000). It
finds the 2 s / 4 s Prefill targets require 26x / 43x the measured ceiling and
requests owner adjudication; it changes no target, route, or priority. The
Decode 10 tok/s target remains locked, but the 2026-09-26 hardware-bound
analysis (see the Decode row and the unroll-128 section below) shows the
batch-one weight-read floor is 101.5 ms/step (9.85 tok/s), i.e. 10 tok/s is
below the hardware floor for this model on this device; the short-context
9.51 tok/s measurement is consistent with that floor. Owner adjudication is
requested (batched-decode contract change, lower quantization, or re-set
target); no route is selected and no target is changed here.

The original integration `edf4da2` combined main `13be53c` with liveness-only
`055fb245` after both matched P40000/O16 API pairs improved pure Prefill and
external TTFT.
The selected source's ordinary compiled profile is
`q3x.sm87.candidate.p40.legacy-c512-terminal-prefix.v1`, with
`production_eligible=false`, `release_qualified=false`, and v20 elision
receipts. It preserves incumbent QT2/GroupQ64 Attention, scalar final/Decode,
the complete startup repair and inventory, and all existing main metadata.
Score-feed kernels and selectors are absent. The candidate branch's installed
OFF artifact, not a new main artifact, supplied the matched API result.
That merge preserved runtime, headers, tests, and CMake exactly as `055fb245`.
The separately admitted `396ce977` repair changes only the runner's host-side
source-Graph lifetime after full executable upload/synchronization; it keeps
the original embedding node, complete executable, numerical kernels, arguments,
liveness dataflow, arena, and startup thresholds. Successor `0816bb20` changes
only the existing Graph test's stale arena constant and explanatory comment.
The resulting runtime and test source are therefore not wholly identical to
`055fb245`; the earlier numerical and matched-performance evidence remains
bound to its recorded artifacts.

The first `edf4da2` installed-main OFF build passed, but both fresh P40 startup
attempts failed before readiness with `device_memory_budget_exceeded` and no
request or timing result. The original failures remain recorded: observed
Graph free drops were 292,073,472 and 320,229,376 bytes, with 113.616 and
110.735 ms preparation. A separate instrumented diagnostic is not an API
result. The repaired source's existing C32/P45 real-model Graph test now passes
serial/replay/reset/trace/P44 boundary checks with 25 slots and the unchanged
resource gates. Its first attempt failed the stale fixture before model load,
not the ownership repair. This is bounded Graph correctness, not a full-state
Graph oracle, repeated startup reliability, or performance qualification.

The repair's separate installed OFF P40000/O16 request completed exact output,
v20 route/reset, complete startup inventory, and owned cleanup. Its raw parent
remains `accepted=false` with no timing authority: one post-readiness CPU
sample lost the command line of an already-exited process and produced two
duplicate identity rejection reasons. The process is not identified by guess.
Independent review accepts only the unaffected startup, output, route, and
resource-closure facts together with the Graph test as grounds for this bounded
source repair. No C1 speed, memory improvement, or repeated reliability is
claimed, and the original rejected record is unchanged.

The original attempt and source-bridge boundaries remain frozen in the
[Graph lifetime selection record](metadata/qwen36-27b-graph-template-lifetime-selection-2026-09-09.json).

Fresh installed main `230eac1` / tree `f3b76a8` now closes both required API
checks on Release/OFF ELF
`270a6bb46b6e9862d3c75f757e30a78a12867a049a0f02735ba8e44558bc9cbc`
(Build ID `78402841bace909f054fe0a1924189a99c9f46b6`). The actual cold/no-cache
P40000/O16 request reports 663,663.424828 ms pure Prefill /
**60.271514903 prompt tok/s**, 663,686.007987 ms external TTFT, and
253.827138333 ms/token / 3.939689060 Decode tok/s. Spawn-to-ready was
35,317.616175 ms. Exact output, complete SSE/usage, v20 native role/reset
receipts, 25-slot startup inventory, and owned resource closure pass.

A separate fresh server using the same installed artifact passes standard
EvalScope 1.9.1 one warmup plus 8/8 measured requests and an additional raw-SSE
request. Mean TTFT is 2,610.732378 ms, TPOT 104.590151 ms, and independently
recomputed workload prompt throughput 119.051974290 tok/s. Server phase sums
give 191.054002986 prompt tok/s and 9.553634715 Decode tok/s, with different
denominators. All ten dynamic v20 P/O and reset receipts pass; the standard
client does not retain the warmup finish reason, so that field remains
unavailable. Short spawn-to-ready was 35,487.076225 ms.
All eight measured outputs and P/O counts match the prior `d6565eb` short
main run by exact request-body SHA. This is bounded text/usage agreement,
not a full capability, state, or logit oracle.

These are completed current-main integration observations, not a new matched
speedup, Graph-memory improvement, complete capability result, or repeated
startup-reliability qualification. The original candidate CBBC improvement
remains limited to its 5.36%--5.87% panel. Exact new artifact/results and all
limitations are in the separate
[mainline closeout record](metadata/qwen36-27b-terminal-prefix-mainline-closeout-2026-09-09.json);
neither earlier selection record is overwritten.

The selected 0.7.0 source profile retains 40,000 prompt tokens, a 4,096-token
output ceiling, `prompt + output - 1 <= 44,095`, the 3,070,908,416-byte exact
request arena, Bearer authentication for non-health endpoints, and no public
execution-tactic or arena selector. It retains complete startup inventory and
the strict 25-slot/256-MiB/one-second/8-GiB Graph readiness contract. Ordinary
eligible generation uses private layer-63 terminal-prefix elision with v20
route/reset receipts; outside that scope the existing v1/v16 behavior is
unchanged. Source selection grants no complete capability, long-output,
repeated startup reliability, or release qualification.

The candidate-only CBBC panel preserves exact output and complete actual
route, startup, clock, cache, and owned-closure checks in all four processes:

| Matched pair | Baseline / candidate pure Prefill (tok/s) | Throughput change | External TTFT change |
| --- | ---: | ---: | ---: |
| B1 / C1 | 57.050820 / 60.401250 | +5.8727% | -5.5468% |
| B2 / C2 | 57.286530 / 60.357629 | +5.3609% | -5.0880% |

The separate same-ELF ON P40000/O16 baseline/liveness comparison matches
complete persistent/used-KV state, all 16 full-vocabulary BF16-logit digests,
live final rows, token IDs, and text. P576 distinct dual-poison/canary cases
also match. Its numerical-only schema-3 Graph rollback exception is not an
OFF startup waiver. The ON capture ELF and installed OFF candidate ELF are
different and remain separately identified. Decode movement is small and
mixed; startup is not improved. Planned allocations are unchanged, but P40
peak memory was not measured. Exact identities, raw hashes, pair effects,
source bridge, and limits are frozen in the
[liveness-only selection record](metadata/qwen36-27b-terminal-prefix-p40000-api-selection-2026-09-09.json).

### Pre-elision startup and earlier route evidence

The bounded startup repair is now integrated into main source. It performs
one exact scalar initialization and complete state reset before the unchanged
Graph increment checks. Its isolated installed Release/OFF artifact completed
the ordinary P40000/O16 API request at 57.1632374429 prompt tok/s and
699,773.335528 ms external TTFT, with exact output, valid v16 route, complete
SSE/usage, and clean resource closure. All 25 Graph slots passed admission;
the recorded free-memory drop was 113,025,024 bytes and preparation took
111.358 ms. This is a completed request and a bounded startup repair result,
not a Prefill or startup speedup, repeated startup reliability, or release
qualification. Its fresh installed-main integration closeout has now passed
below; the installed-main P40000/O16 profile has also completed with the
current-route attribution in section 4.
Neither Prefill candidate was absorbed at that startup-repair checkpoint. See the
[startup repair record](metadata/qwen36-27b-graph-startup-initialization-repair-2026-09-09.json).

The first combined terminal-prefix/score-feed candidate is now rejected on
the actual installed Release/OFF P40000/O16 API. Source `09de111` passed the
ordinary same-ELF numerical gate using ON capture ELF `4318252c...`, including
full live-state and full-vocabulary-logit digests. Its separate installed OFF
ELF `f9f08154...` then completed the API at
51.1657775463 prompt tok/s and 781,795.433639 ms external TTFT. This is
10.4918% less throughput than the retained same-workload ordinary request
above. Output, route, SSE, complete 25-slot startup inventory, fixed clocks,
and resource closure pass; it is a valid negative direction, not a numerical
or infrastructure failure. No repetition, profile, or mainline absorption is
assigned to this version. The result does not quantify score-feed alone;
the independent liveness selection above does not rehabilitate this combined
version, and the sole score-feed correction has no selection from this record.
Exact identities and limitations are in the
[combined-v1 rejection record](metadata/qwen36-27b-ordinary-exact-score-feed-v1-api-rejection-2026-09-09.json).

The pre-elision installed main `d6565eb` / ELF `b4ccef99...` reached readiness in
35,478.588 ms and passed EvalScope 1.9.1 on one warmup plus 8/8 measured short
requests and a separate raw-SSE request. Mean TTFT was 2,636.723 ms, TPOT
104.733 ms, and independently recomputed workload prompt throughput
118.255733 tok/s. Server phase sums gave 189.163535 prompt tok/s and
9.540915 Decode tok/s; these use different denominators from workload
throughput. All ten actual v16 P/O, reset, and route receipts pass, as do the
25-slot Graph inventory, cache drop, owned shutdown, and resource closure.
The standard client does not store the warmup finish response, so no warmup
finish-reason claim is made. The eight measured texts match the same request
bodies in the historical 0.6.0 short corpus, which is bounded output sanity,
not complete accuracy or a timing comparison. This closes integration health,
not a speedup, candidate qualification, or release gate; exact identities and
limits are in the
[installed-main profile and short closeout record](metadata/qwen36-27b-ordinary-main-p40-nsys-attribution-2026-09-09.json).

Startup stability remains unresolved. Separate original Release/ON captures at
`8992140` still rejected the 256-MiB Graph increment before generation: P576
poison-A attempts r1/r2 and P40000 baseline r1 reported approximately 111 ms
preparation, zero retained slots, and `device_memory_budget_exceeded`.
These are startup failures, not numerical mismatches. The successful OFF
integration above does not qualify repeated startup reliability; later
numerical-only capture success does not qualify ordinary OFF startup or
retroactively accept those failures. Their raw hashes and diagnostic boundary
are retained in the same closeout record.

The preceding ordinary installed Release/OFF P40000/O16 request on 2026-09-09
completed at 57.2572803117 prompt tok/s and 698,623.065496 ms external TTFT.
Output, SSE, usage, route, cache-drop, shutdown, and resource closure passed.
The raw monitor rejection is preserved; a separate request-window review
classifies benign CPU activity as supporting context and accepts this ordinary
request observation, not a speedup. A subsequent installed startup failed
before readiness because its Graph preparation free-memory drop exceeded the
existing 256-MiB check, despite complete sidecars and a 171.983-ms preparation.
The shared-memory attribution remains unresolved; the bounded initialization
repair above does not claim exclusive attribution. Identities and claim limits are in
the [ordinary request/startup record](metadata/qwen36-27b-ordinary-p40-startup-and-request-observation-2026-09-09.json).

The 2026-09-09 source/raw-record review does not change the installed runtime
or establish a new performance result. It corrects three interpretation
boundaries: the v10-to-Legacy restoration changed FP8, A/B, GDN, and MLP as
well as Attention; strict P40000/O16 captures contain exposed logit statistics,
not the complete vocabulary-logit vector; and numerical disagreement with
Legacy alone is not a demonstrated business-capability regression. The frozen
r12 strict comparator explicitly uses all-prompt tiles, whereas ordinary main
uses a P-1 tiled prefix plus the existing scalar final prompt step. A new
ordinary-route change therefore needs a matched production-policy comparator;
r12 cannot silently qualify that different boundary. Existing exact gates
remain in force. The newly authorized engineering window and its
bounded real-API implementation packages are owned by
[`ROADMAP.md`](ROADMAP.md#2026-09-09-bounded-engineering-window).

The pre-elision mainline built and installed a production-shaped 0.7.0 service from the
ordinary `orin-release` preset (`Release`, SM87, `BUILD_TESTING=OFF`). Its
sealed default profile was
`q3x.sm87.production.p40.legacy-c512-exact.v3`: 40,000 prompt tokens, a
4,096-token output ceiling, `prompt + output - 1 <= 44,095`, a
3,070,908,416-byte exact request arena, Bearer authentication for non-health
endpoints, and no public execution-tactic or arena selector. A non-loopback
listener requires an owner-only key file. The installed profile remains
`release_qualified=false` until the exact installed artifact completes the
current target-length, Decode, accuracy, stability, and capability matrix.

Version 0.7.0 is an intentional 0.x C++ ABI transition for ordinary request
reuse. The first request after state creation is receipted `already_clean`; a
prior exactly committed successful Legacy-C512 request clears complete
Conv/GDN state plus only its written K/V prefix; cancellation, poison,
uncommitted work, position mismatch, direct runner use, and every uncertain
boundary retain the conservative full reset. The historical ordinary witness
recorded reset mode, positions, bytes, and synchronized duration in schema v16;
the selected liveness profile preserves those fields in v20.
That earlier installed 0.7.0 reset-policy artifact is ELF
`d70ba913de3f5256b544b06d864dad5a7b5f4293d6b2accc4353761c872c0ae2`
(Build ID `9364ad0f825e755537da05dff97187e58053ab66`). It now has real-model/API
authority for retaining this reset policy in the ordinary default. All other
0.6.0 installed-artifact measurements below remain historical evidence for
their exact binary and do not qualify the new ABI tuple.

The formal installed Release/OFF P1024/P4096/P8192 BCCB accepted 12/12 cells.
All six matched pairs reduced the external-TTFT-minus-pure-Prefill bridge;
bridge medians moved from 23.325371 to 7.152718 ms, 22.188700 to 10.818110 ms,
and 33.941824 to 15.180447 ms respectively. All 24 warmup plus measured
requests preserved generated text, SSE termination, usage, finish, and route
receipts. External TTFT medians were lower at all three lengths, but pair
direction above P1024 was mixed; pure Prefill and Decode also moved only at
mixed or noise scale. This supports the targeted cleanup policy, not a
material TTFT, Prefill-throughput, or Decode improvement claim.

A separate current-main same-server P40000 lifecycle then measured
15.185065 ms for the exact 40,015-position/2,700,869,632-byte committed prefix
and 17.504564 ms for the 44,095-position/2,968,256,512-byte conservative full
reset: a 2.319499-ms or 13.2508% cleanup-duration reduction relative to full.
An intervening P1024 cancellation was observed after fresh 99% GPU activity,
emitted no success witness, and forced the full-reset receipt on the next P40
request. All three successful P40 requests had identical generated text, SSE,
usage, finish, and route receipts. The bundle did not retain the complete
external content-event arrival timeline, so its derived external TTFT and
Decode figures have no authority. Its single-sample approximately-695-second
server Prefill differences are not attributed to reset and do not establish a
P40 throughput gain. Exact identities, invalid attempts, hashes, and claim
limits are frozen in the
[`request-reset mainline closeout`](metadata/qwen36-27b-request-reset-mainline-closeout-2026-08-23.json).

The ordinary profile fails before listening unless its complete Prefill,
Decode, and short-position Graph inventory is present. This includes the
208-projection/7,214,202,880-byte FP8 Prefill supermatrix and
11,013,898,240 bytes of Decode layouts; total retained acceleration sidecars
are 18,228,101,120 bytes. Ordinary Release route composition is typed and
does not consult `Q3X_*` environment selectors. The preceding 0.6.0 batch passed
the Release/OFF build, install-tree consumer, help/authentication, host policy,
exact-span GDN CUDA, Decode numerical-rejection, and package tests. Commit
`c6c34ef` / tree `6a2df18` also passed a fresh installed-artifact
short EvalScope closeout and one exact-default P40 request. Target-length
repetition, complete accuracy, capability, and release qualification remain
open.

By explicit project-owner direction on 2026-08-22, performance work is
resumed with production conversion as the selection boundary. The first
current-main real-API comparisons establish the exact C256/C512 GDN whole-span
path as a positive ordinary-default candidate: pure Prefill is 3.5290% faster
at P1024, 4.5009% faster at P4096, and 3.6983% faster at P8192, with exact
generated output. The older prompt-wide Embedding and Attention preprocess
gains remain selected. Decode split-KV is not selected despite a favorable
API timing proxy because its direct S65 BF16 oracle differs from the exact
fallback; S>=65 therefore remains on the exact route. The measured Gate/Up
coupled-feed and Down consumer-order Decode layouts are retained together in
the ordinary installed default, with startup failing if their exact inventory
cannot fit.

The final same-ELF Decode BCCB panel accepted 8/8 controlled requests. At
P1024, both layouts moved server Decode from 111.134339 ms/token / 8.998120
tok/s to 107.020180 ms/token / 9.344032 tok/s, a 3.8443% speedup. At P4096,
they moved 122.797655 ms/token / 8.143479 tok/s to 118.573211 ms/token /
8.433608 tok/s, a 3.5627% speedup. Prefill was neutral in both panels.
Marginal BCCB then showed Gate/Up contributes 3.6921% at P1024 and 3.3208% at
P4096 given Down, while Down contributes a smaller but still positive 0.2504%
and 0.2992% given Gate/Up. All 24 requests across the combined and marginal
panels produced the same UTF-8/SSE text hash; that is not a hidden-state,
logit, or generated-token-ID oracle. Exact evidence and authority are frozen
in the
[`current-main production-default closeout`](metadata/qwen36-27b-production-default-mainline-closeout-2026-08-23.json).

The fresh installed `BUILD_TESTING=OFF` server is ELF
`ab3492a690622e63bcfeafab055081a635c4f65507fc48b9d1a1bca0c1cebe02`
(Build ID `d8ee66d1355e05dab71a9434c537419e08e4b121`). One warmup plus
eight measured EvalScope 1.9.1 short requests passed 8/8: mean TTFT was
2,652.203 ms, TPOT 104.931 ms, ITL 104.920 ms, and independently recomputed
prompt throughput 117.737386 tok/s. Server intervals reported 189.257892
prompt tok/s and 105.013349 ms/token / 9.522599 Decode tok/s. This is a
single-process short integration proxy, not a matched B/C speedup or a
target-length result. Its first predecessor (`r1`) remains invalid because of
two harness-validation failures and contributes no timing; only corrected
`r2` is accepted.

The preceding 0.6.0 exact-default controlled length observations are:

| Prompt | Server pure Prefill | External TTFT | Server Decode | External Decode |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 183.598703 tok/s | 5,605.552 ms | 9.340364 tok/s | 9.349471 tok/s |
| 4,096 | 192.530782 tok/s | 21,303.933 ms | 8.434443 tok/s | 8.441086 tok/s |
| 8,192 | 153.943196 tok/s | 53,245.766 ms | 7.460155 tok/s | 7.464576 tok/s |
| 16,384 | 107.513660 tok/s | 152,424.173 ms | 6.071000 tok/s | 6.072984 tok/s |
| 32,768 | 66.726052 tok/s | 491,127.837 ms | 4.413194 tok/s | 4.413428 tok/s |
| 40,000 | **57.230764 tok/s** | **698,967.822 ms** | **3.937236 tok/s** | **3.937134 tok/s** |

Each row is one accepted C-only process with 16 output tokens, so the table
reports that earlier artifact's behavior, not current-main performance,
repetition, or speedup qualification.
P1024--P32768 used the runtime-identical `0e89575` installed artifact;
P40000 is a fresh direct `c6c34ef` run. At P40 the ordinary exact default is
only 14.5698% of the historical v10 392.804397 tok/s result, a 6.8635x gap.

The subsequent selector-exact persistent-Attention v1 composition did not
close that gap. Candidate `e325c6ba` / tree `7423cd2f`, built as fresh
candidate-only ELF `f427c85b...` (Build ID `a2b5ca47...`), completed both
fresh P40000/O16 arms but returned `rc=3` / `valid_mismatch`. All 37 retained
Prefill-commit state-digest comparisons, all 37 generation-return state-digest
comparisons, and 80 generation comparisons differed; exposed step-0 logit
statistics, four generated token IDs, and final text had already diverged. The
comparison is
infrastructure-valid and has no timing authority. It rejects the complete
whole-core selector v1 composition without proving one kernel as the root
cause; the branch remains unmerged, default-off, and non-production, and its
candidate-only P40016 capacity fixes are not ordinary-mainline changes. Exact
identities, state boundaries, route receipts, hashes, and claim limits are
frozen in the
[`selector v1 rejection record`](metadata/qwen36-27b-selector-exact-persistent-attention-v1-rejection-2026-08-27.json).

An owner-directed fail-fast checkpoint-weight-only P40000 screen at clean
`e61cff9` has now rejected the current NVFP4 M128N256K64 persistent16
BF16-HMMA GateUp+SiLU and Down+residual kernel skeleton. Its two single-layer
CUDA-event pairs were 17,919.347656 and 17,919.783203 ms. The more favorable
sample, charged to only 63 complete P40 MLP layers while treating the terminal
M1 work and every other projection family as free, projects to 1,128.918902 s
against the complete 5.0-second P40 projection allocation: a 225.7838x
excess. The prerequisite real-weight M192 bitwise gate, full-output poison/
zero check, input preservation, guards, CUDA cleanup, and complete pre/post
device-handle audits pass. This valid negative result ends timing, profiling,
and tuning of that BF16-HMMA skeleton; exact identities and claim limits are
frozen in the
[`P40000 quick-kill record`](metadata/qwen36-27b-sm87-target-aot-p40000-quick-kill-2026-09-01.json).

The required successor arithmetic gate has also completed and rejected the
frozen `bmma-static-support-k16-parent-zero-fill-v2` mapping before CUDA. Its
authenticated real-P40 route reached 35,328 prompt rows and proved
419,640,115,200 mandatory zero-filled K256 warp instructions, already
3,480,115,200 instructions or 0.8362445% above the absolute five-second
capacity while every additional exactness pass, incomplete tail, terminal
scalar path, and unprocessed suffix was charged free. This closes mapping v2
without implementation, repetition, or profiling; it is not a hardware
ceiling and does not reject a materially different exact arithmetic/dataflow
class. Exact identities and limits are frozen in the
[`mapping-v2 rejection record`](metadata/qwen36-27b-sm87-target-aot-real-p40-arithmetic-mapping-v2-rejection-2026-09-02.json).

AOT payload authentication and persisted direct loading remain useful
architecture prerequisites, but AOT exploration is deferred rather than the
active next gate. `AC-PREFILL-SM87-AOT-SYSTEM-v1` receives no CUDA successor
by default; resumption requires a materially different exact mapping with a
new bounded proof or an explicit successor architecture. This scheduling
change does not alter the Constitution targets or grant AOT numerical,
generation, API, performance, release, or production authority.

The later exact-span/P39936 branch sequence is now closed on the real
P40000/O16 API path. The corrected exact selector measured
27.96118777397028 tok/s. Replacing its persistent Q8 Attention suffix with 80
exact spans per full-Attention layer recovered the test-only branch to
55.94887170163682 tok/s, still 2.239865% below the ordinary exact main
reference of 57.2307638066734 tok/s. Replacing 10,336 FP8 physical launches
with the P39936 M128N256 route then regressed to 51.17353679519425 tok/s with
5,056 Gate/Up dual-stream pairs and 51.17176853575732 tok/s with the serial
MLP schedule. All candidates passed the separate strict P40000/O16
state/logit/token/text oracle and completed the actual API request; they are
`BUILD_TESTING=ON`, test-only, uninstalled branch artifacts and are not main.
The P39936 route and its dual-stream rescue are rejected without repetition
or profiling; exact-span is retained only as a test baseline and supplies no
mainline absorption. The exact source, binary, route, output, metric, and
authority boundaries are frozen in the
[`exact-span/P39936 closeout`](metadata/qwen36-27b-exact-span-p39936-api-lineage-closeout-2026-09-05.json).

The exact-span branch's final same-skeleton regular-Q4 Attention child is also
closed on the strict P40000/O16 and real API path. Clean candidate commit
`dc138cbf` / tree `02ed6dd0` passed the complete state/logit/token/text oracle,
then completed one cold-cache API request at 1,043,416.453717 ms /
38.33560402225455 tok/s pure Prefill. That is 33.0157393% lower throughput and
49.2888015% higher latency than the separately pinned ordinary exact main
reference, and 31.4810060% lower throughput than the exact-span QT2 parent.
Its v19 route recorded 32 GroupQ64 plus 1,248 regular-Q4 Attention
submissions, with zero QT2, Q8-suffix, or Attention fallback submissions.
The artifact is `BUILD_TESTING=ON`, test-only, uninstalled, unmerged, and not
main. Regular Q4 receives no repetition or profile, the same-skeleton Q3--Q5
span scan is closed, and no code is absorbed. Exact identities and claim
limits are frozen in the
[`Q4 rejection record`](metadata/qwen36-27b-selector-exact-regular-q4-span-v3-rejection-2026-09-05.json).

This AOT deferral does not amend the Constitution targets. The
exact recovery anchor remains
`archive/v4-construction-ownership-20260820@f3545240075651eaa54a5bea6c0f15ee9dfd9a3e`;
it is an incomplete, default-off archive and is not part of this mainline. Its
schema-v5 closeout source remains `status=fail` because a 419,917,824-byte
post-destruction free-memory gap exceeded the fixed 33,554,432-byte tolerance.
The later holder-free post-exit preflight is not a Jetson `nvmap`
`no_owner_leak` classification. The archive grants no mainline numerical,
generation, API, timing, performance, release, or production authority.

The consolidated default-off mainline at `19a4b465` has also completed its
required real-model health closeout on the pinned Qwen3.6-27B checkpoint. One
fresh server reached `/healthz` in 25,680.088 ms, then EvalScope 1.9.1 completed
one warmup plus eight measured short requests with 8/8 success. The measured
proxy reported mean TTFT 2,749.124 ms, mean TPOT 108.912 ms, and 113.533327
prompt token/s; these figures are a valid eight-request historical short
proxy, not a performance improvement or a 32-request baseline. Peak `/proc`
VmRSS/VmHWM was 30,703,714,304 bytes, the system `MemAvailable` drop from the
first post-spawn sample was 23,852,707,840 bytes, and CPU/Tj and GPU remained
below the then-declared 70C evidence gate at 69.343C and 68.250C. That
conservative threshold is now superseded by the owner-specified normal range
through 85C. From above 85C through 90C validity is decided from actual clock,
over-current, and throttle evidence; above 90C is operational risk.
Temperature is a validity/diagnostic guardrail, not an optimization target.
All preflights passed,
SIGINT shutdown returned zero, and no server-owned `nvmap` residue was
observed. The exact record and its measurement limitations are in the
[`mainline real-model acceptance record`](metadata/qwen36-27b-mainline-real-model-acceptance-2026-08-21.json).
It grants no 40K/60K/approximately-130K, final-product API, release, or
production authority. That record did not itself resume Prefill work; the
later 2026-08-22 owner direction did.

Under the later explicit owner direction to absorb confirmed cross-branch
gains before returning completely to delivery work, the runtime stack ending
at `ff47f179` now selects two zero-allocation mechanisms on one narrow
development-default scope: whole-tile prompt-wide Embedding gather and the
128-thread exact full-Attention preprocessing map. Selection requires legacy
execution control, the SM87WeightOnly backend, the Legacy-C512 request-memory
profile, and a non-sealed exact-arithmetic route. Other backends, memory
profiles, controls, the sealed arithmetic route, and the M1 tail keep their
established behavior. The policy is source-local, has no environment
selector, and its mutable A/B/hit seams are absent from `BUILD_TESTING=OFF`.

The selected mechanisms passed real-model baseline/Embedding-only/
Attention-only/combined comparisons at P514, P4096, and P8192. The comparisons
are bit-exact for complete convolution and GDN state, every used K/V row in
all 16 full-Attention layers, sequence length, prompt and generated token IDs,
generated text, and public logits statistics. P4096 generated 16 tokens and
therefore exercised 15 Decode transitions after the first token. The related
exact-C512 all-prompt final-token candidate was deliberately not selected: it
changed state, used K/V, and public-logit results at both P512 and P4096.

Matched clean-host BCCB measurements used fresh `BUILD_TESTING=OFF` processes,
two baseline and two candidate samples per length, 16 output tokens, fixed
1.4976 GHz CPU / 1.020 GHz GPU / 3.2 GHz EMC clocks in MAXN, and controlled
repeated token-ID prompts. Candidate means were:

| Prompt | External TTFT | Server pure Prefill | Pure prompt rate | External Decode | Startup | Max candidate VmHWM | Prefill direction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 | 27,751.272 ms | 27,742.651 ms | 147.642708 tok/s | 6.832613 tok/s | 35,358.079 ms | 30,417,940,480 B | +0.2599% |
| 8,192 | 68,267.597 ms | 68,255.342 ms | 120.019915 tok/s | 6.103963 tok/s | 35,348.268 ms | 30,688,944,128 B | +0.2268% |
| 16,384 | 191,768.588 ms | 191,747.013 ms | 85.445936 tok/s | 5.031028 tok/s | 35,553.153 ms | 31,233,032,192 B | +0.1224% |
| 32,768 | 607,164.704 ms | 607,128.226 ms | 53.972166 tok/s | 3.716240 tok/s | 36,020.790 ms | 32,319,053,824 B | +0.1851% |

All 16 runs passed the clean-host, continuous ownership, then-declared sub-70C
thermal, route, output, SIGINT shutdown, and post-exit `nvmap` gates. The same small
positive sign at all four lengths supports development-default absorption,
but the 0.12%--0.26% range and two-sample protocol are not statistical or
release qualification. Decode, startup, and memory moved at noise scale in
both directions and receive no improvement claim. These prompts are
controlled regression witnesses rather than real Agent/EvalScope prompts;
16K/32K used explicit expanded capacity and an 8 GiB arena; for that historical
artifact the adapter's 8,192-token/2-GiB defaults remained unchanged. Across
the two candidate runs at
4K/8K/16K/32K, maximum PSS was respectively 344,807,424 / 344,801,280 /
344,803,328 / 345,393,152 bytes, and maximum system `MemAvailable` drop was
22,971,265,024 / 23,236,730,880 / 23,591,137,280 / 24,705,056,768 bytes.
After SIGINT, BCCB listener ports remained in TCP TIME_WAIT and became reusable
only after approximately 44--48 seconds.

The final `ff47f179` `BUILD_TESTING=OFF` integration run then reached owned
health readiness in 25,431.773 ms and passed EvalScope 1.9.1 on one warmup
plus 8/8 measured short requests. Mean TTFT was 2,741.090 ms, mean TPOT was
108.612 ms, and independently recomputed prompt throughput was 113.857651
tok/s; EvalScope's written zero `Input Throughput` field is invalid and is not
cited. The corresponding server witnesses averaged 2,733.324 ms pure Prefill
and 2,737.075 ms TTFT; 3,981 tokens divided by the sum of the eight per-request
pure-Prefill intervals was 182.058549 tok/s, while server Decode was 108.702
ms/token / 9.199479 tok/s. That aggregate server-interval rate and EvalScope's
wall-clock prompt throughput use different denominators and are not directly
interchangeable. Peak VmRSS/VmHWM was 30,688,911,360 bytes; maximum CPU/Tj and
GPU temperatures were 69.593C and 68.718C. Raw SSE usage and completion
ordering passed, SIGINT returned zero, the port was reusable, and the post-exit
clean-host/`nvmap` gate passed. This is candidate integration health, not a
baseline/candidate speedup or target-length result. Exact identities,
raw-bundle hashes, and all limits are frozen in the
[`prompt-wide mainline absorption record`](metadata/qwen36-27b-prompt-wide-mainline-absorption-2026-08-21.json).

Separate from that absorbed Legacy-C512 route, commit `0cf4048` now makes the
exact-P40000 whole-core v10 inventory a first-class, separately named
`BUILD_TESTING=OFF` development artifact. The single default-OFF
`orin-p40-whole-core-dev` preset builds
`qwen3x-eval-server-p40-v10-dev`; the binary requires the typed
`--development-route p40-whole-core-v10` acknowledgement, fixes the complete
P40000 profile atomically, rejects ambient `Q3X_*` controls, and has no install
rules. The ordinary default binary, capacity, route, and installation remain
unchanged. Review follow-up `289f6d0` safely ignores malformed non-assignment
environment entries, leaves the ordinary default ELF byte-identical, and has a
fresh final development build receipt. This is development-route retention,
not release promotion.

The strongest whole-product P40 observation remains the historical v10
measurement. It was recorded once from a `BUILD_TESTING=ON`, binary-pinned
dirty tree above `a4f95ba`; the implementation was committed later as
`a46d165`, and the new tracked development artifact makes that route
rebuildable without claiming that its historical timing transfers:

| Observable | Current incumbent observation |
| --- | ---: |
| Consumed prompt tokens | 40,000 |
| Server pure Prefill | 101,831.853876 ms |
| Pure prompt throughput | 392.804397 tok/s |
| EvalScope TTFT | 101,870.53 ms |
| EvalScope minus server TTFT | 3.730501 ms |

That route improved the preceding retained direction by 7.02138%, but it is
not a production result. It was measured from a pinned dirty-tree binary, is
default-off, and inherits a known P513 full-state mismatch from its
FlashInfer Attention arithmetic. It therefore has neither production-accuracy
nor release authority. Its transaction, memory, route-receipt, and
whole-prompt control substrate is retained as development infrastructure.
Its timing authority is one clean-host real-API direction sample plus one
bounded NSys capture, not a repetition-qualified performance baseline.
The route is now present in tracked source, but 392.804397 tok/s is still the
historical incumbent rather than a current-mainline reproduction or production
rate. It is unrelated to the two Legacy-C512 mechanisms absorbed above. The
one-output-token workload contains no Decode transition, so Decode latency and
token/s are unavailable rather than zero.
The witness consumed all 40,000 prompt tokens and reported zero Prefix-cache,
MTP, cuBLASLt, external-reference, approximate, exact-fallback, and forbidden
route hits.
Exact evidence is frozen in the
[`v10` whole-core record](metadata/qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json).

The historical main snapshot `b0c0c837` / tree `2963cb99`, built as ELF
`edb999f91683df85bfab3b757c6bfcb055e55110663566d177fcc7379cfed8f4`
(Build ID `e349cce6283731ea85fe3dd6e654b8ff29eb0ac2`), now has a strict
sustainable-profile BCCB integration-health result. All four cells pass the
continuous sysfs, cooling, over-current, thermal, ownership, route, and cleanup
gates at CPU 1,497,600 kHz, GPU 1.02 GHz, and EMC 3.2 GHz. Candidate mean pure
Prefill is 325.983493208 tok/s versus 326.111753524 baseline, C/B
0.9996066983. This passes the predeclared 0.99 matched-baseline floor but is
not a speedup and does not reproduce the historical 392.804397 max-clock
observation. Full max-GPU r4/r5 are wholly invalid because over-current event
counters increased, despite all temperatures remaining below 85C. At CPU
1,497,600 kHz / GPU 1.224 GHz / EMC 3.2 GHz, r6-r8 contain four fully valid C
cells (375.895393126, 376.039811195, 375.958913176, and 375.931597138 tok/s;
descriptive mean 375.956428659) and three fully valid B cells (descriptive mean
375.935680902; descriptive C/B 1.00005519). Independent external CPU samples
prevent each bundle from completing strict BCCB, so these cells are not
stitched into a performance decision. The one-output-token workload has no
Decode transition. Exact build identities, bundle hashes, and claim limits are
frozen in the
[`P40 v10 mainline absorption record`](metadata/qwen36-27b-p40-v10-mainline-absorption-2026-08-21.json).

The later independent stock-vLLM-Marlin projection reference completed the
same P40 API path at 101,857.500727 ms / 392.705493 tok/s. It supplied no
positive direction against the incumbent, so that skeleton is closed and
remains default-off and accuracy-unqualified. See the
[`v15` rejection record](metadata/qwen36-27b-prefill-p40k-vllm-marlin-parity-rejection-2026-08-11.json).

The incumbent is still 10.95x below the owner-established 4.3K tok/s useful
vLLM starting line. Pure Prefill alone consumes 50.92x the complete two-second
TTFT budget, while the measured TTFT is 50.94x the target. These are active
product constraints from the
[`ENGINEERING_CONSTITUTION.md`](ENGINEERING_CONSTITUTION.md), not limits
inferred from the current implementation. P60 and approximately P130 have not
been opened because no competitive, accuracy-admissible P40 composition
exists.

A target-first stock-vLLM reference witness was then attempted from clean
commit `fe626be` with the real checkpoint, one cold/no-cache P40 API request,
one output token, and a whole-prompt scheduler budget of `40000`. The API
transaction completed and the process-group, route, and cleanup receipts were
captured, but the measurement is invalid: the request interval crossed the
old `70C` protocol threshold and reached `81.812C`, which is within the
owner-specified normal range through 85C. The independent measured fact that
still invalidates it is real CPU downclocking from 2201 MHz to approximately
1.1 GHz. Its manifest remains `valid=false` with an empty result set. No timing
from that run is retained, compared, or used to alter either the native
incumbent or the owner-established 4.3K tok/s starting line. The exact
invalidation and artifact hashes are in the
[`P40 reference-witness invalidation record`](metadata/qwen36-27b-vllm-p40-target-witness-invalid-2026-08-12.json).
The current stock-auto route also selected Marlin FP8, FlashInfer Attention,
and Triton/FLA GDN without Humming, so it is not yet a matched reconstruction
of the known optimized vLLM observation.

The retained warmup from that invalid package now has a diagnostic-only
three-surface reconciliation. The three primary surfaces are the vLLM backend
`Avg prompt throughput` logger at 3999.7 token/s, the request-bound server
Prefill interval at 368.579880 token/s, and prompt tokens divided by EvalScope
TTFT at 368.238711 token/s. The server-TTFT-derived 368.358289 token/s and
EvalScope's printed 368.2478 input-plus-output token/s are auxiliary
observations, not additional pure-Prefill surfaces. The logger value is
10.851650x the request-bound server rate. vLLM's logger source semantics and the
fresh-process singleton join are consistent with completed prompt work
landing in a later local logger interval. Future harness executions now
explicitly fix `VLLM_LOG_STATS_INTERVAL=10.0` and bind `envs.py`, the server
logging task, and `LoggingStatLogger`: ten seconds is the configured trigger,
while the printed rate divides locally recorded computed prompt tokens by the
logger's actual monotonic time since reset. That actual denominator is not
printed and need not be exactly ten seconds; the value is service-window
telemetry, not per-request latency or pure Prefill. The evidence
lacks a stable cross-surface request ID and a formal warmup thermal/frequency
envelope, so it changes neither the invalid run's status nor the 4.3K tok/s
owner-established starting line. Exact formulas and raw hashes are frozen in
the [`warmup metric reconciliation`](metadata/qwen36-27b-vllm-p40-warmup-metric-reconciliation-2026-08-12.json).

A default-off, host-only descriptor retains the paused whole-system AOT
candidate for P40/P60/approximately-P130: the exact 64-layer GDN/Attention
schedule, 14 physical groups, five projection roles, paired BF16 A/B producer,
Q128/KV32 online-softmax Attention, and per-token-BF16 GDN transaction are
represented as one typed DAG with 39 resource and 13 event edge classes. A
second host-only contract freezes actual NVFP4/FP8-Marlin projection arithmetic,
same-CTA Gate/Up lifetime, source-to-packed-payload authentication fields,
Attention preprocess/core finite-precision order, and the exact-C16 GDN
candidate's `[head,value,key]` axes and recurrence order. Exact-C16 does not
inherit qualification from the deployed Chunk64 family. A third default-off
admission now hashes caller-supplied host byte intervals, performs and replays
the bit-exact packed permutation, seals a host manifest plus transform receipt,
and compiles real NVFP4 Gate+Up and Down CUDA bodies. A fourth default-off,
test-only slice implements the exact 64-layer real-checkpoint NVFP4 source
audit, bounded host transformation, owned 9,625,927,680-byte device arena,
upload completion, independent device readback, and SHA-256-backed receipt
issuance for 128 artifacts and 192 sources. The Engine now has a default-off,
startup-only trigger that skips mutually exclusive legacy Prefill projection
sidecars and performs a private, owner-backed, transactional `ModelWeights`
attachment after the complete upload is authenticated. The owner is
non-movable, public release fails while attached, and no naked descriptor,
receipt, or device view is exposed.

A provenance-frozen Release/SM87 test-admission probe at `855a7cb` has now
executed that preparation path against the pinned real checkpoint. It audited
192 source tensors, generated and independently read back 128 authenticated
NVFP4 artifacts in one 9,625,927,680-byte device arena, transactionally
attached the nonzero owner/allocation lifetime to `ModelWeights`, and kept all
mutually exclusive legacy Prefill sidecars disabled. The verified payload
catalog SHA-256 is
`367572d8f5aab87c655695fc621562e0e88cb5d1a9656370353d55ab1c4ebdbe`.
Every CUDA, inventory, attachment, and Engine-destruction check passed. The
source probe remains formally `fail` because its immediate in-process
`cudaMemGetInfo` recovery check reported an 8,340,946,944-byte deficit. A
hash-frozen post-exit Jetson `nvmap` snapshot recovered from a completed Codex
rollout records 2,012,087 reusable pages, an empty IOVMM orphan table, and only
the three allowlisted desktop clients. It explains 8,241,508,352 bytes and
leaves a 99,438,592-byte residual, below the fixed 256 MiB diagnostic
tolerance. A later full process/IOVMM/FSI/handle snapshot independently found
no matching process, client, handle, or orphan. The derived classification is
therefore `no_owner_leak`; it preserves rather than rewrites the source
`fail`. Exact identities, raw-event hashes, limitations, and claim boundaries
are frozen in the
[`real-checkpoint preparation record`](metadata/qwen36-27b-sm87-target-aot-real-checkpoint-preparation-2026-08-12.json).

This is preparation and lifetime-attribution evidence only. The attachment
does not authorize a launcher or bind a runner/API route, and the public
launcher remains deliberately fail-closed. There is still no target-AOT
numerical, generation, API, performance, release, or production authority.
The observed 699,705.551133 ms online prepare/attach time is a correctness-run
startup diagnostic, not a performance baseline. Offline-persisted,
authenticated payload generation plus direct startup loading remains a
retained prerequisite if an AOT successor is explicitly resumed; it is not
the active next implementation step. The historical max-clock incumbent
remains 392.804397 tok/s and the installed route is unchanged.

A clean Release/SM87 probe at `9d0613a` has now closed the next narrow gate on
the same pinned checkpoint. The private layer-0 M192 candidate covers one full
M128 region and one predicated M64 tail. Gate+Up matched the canonical route at
all 3,342,336 BF16 elements and Down-plus-residual matched at all 983,040
elements; baseline/candidate/replay digests are identical in both roles, with
zero mismatches, intact guards, complete writes, preserved inputs, and zero
CUDA errors. The same ELF records Gate+Up/Down at 246/210 registers per thread,
76,800 dynamic shared bytes, zero local bytes, 256 threads, one active CTA/SM,
and exactly 16 physical CTAs on the 16-SM device; both geometry and resource
gates pass.

The child evidence still preserves `status=fail` because only its immediate
in-process `cudaMemGetInfo` recovery check failed. Its parent bound the exact
child PID/start time/ELF and captured canonical `/proc` and Jetson `nvmap`
immediately after exit. The original parent report remains `inconclusive`
because its parser did not recognize the real 11-column allocation-detail
format. A strict parser at `5687871` re-derived the same immutable raw snapshot
without overwriting either source: all 20 criteria pass, the classification is
`no_owner_leak`, the 7,047,852,032-byte free-memory gap is fully covered by the
8,241,508,352-byte page pool, and no probe process, probe-named client,
unattributed handle, or orphan remains. The exact three-layer status and claim boundary are frozen in
the
[`layer-0 M192 Oracle record`](metadata/qwen36-27b-sm87-target-aot-layer0-m192-oracle-2026-08-12.json).

This grants only layer-0 M192 real-checkpoint numerical authority, same-ELF
SM87 resource/geometry authority, and a bounded lifecycle diagnosis. It grants
no complete-model, public-launcher, generation, API, performance, release, or
production authority. The 701,709.6913 ms prepare/attach and 728,418 ms probe
wall times remain diagnostics, not performance measurements. The default
runner and its historical 392.804397-token/s max-clock incumbent are unchanged.

## 2. Current capability matrix

| Capability | Current state | Missing production condition |
| --- | --- | --- |
| Pinned Qwen3.6-27B NVFP4 checkpoint | Implemented | Installed artifact must bind checkpoint, binary, layouts, and DeploymentPlan in one attestation |
| Resident loader and typed weights | Implemented | Whole-process memory and release identity remain unqualified |
| Pure C++ tokenizer and greedy generation | Implemented | Public capability and long-run qualification remain incomplete |
| OpenAI-compatible product API | Installed production-shaped 0.7.0 service with bounded queues, streaming, Bearer authentication, public health, and external TLS termination contract | Cancellation, multi-tenant policy, capability, and release stability remain incomplete |
| Installed default context | Sealed P40 profile admits `prompt + output - 1 <= 44,095` with a 4,096-token output ceiling | P60 and approximately-P130 profiles remain unopened |
| Whole-core production profile | Second sealed deployment `q3x.sm87.production.p40.whole-core.v1` (main `8fe4e67`): fixed P40000/O16, whole-prompt layer-major Prefill, full-state accuracy qualified inside the accepted BF16 class, real-model output bitwise identical to the dev route; comprehensive API e2e (2026-09-26) measured 91.4 s TTFT / 419.75 prompt tok/s on P40000/O16 and passed the 6/6 contract matrix | Not the default route; no release qualification; capacity is the pinned P40000/O16 streaming contract only |
| Terminal-prefix source integration | Integrated ordinary OFF liveness-only profile and v20 receipts, with the separately identified host Graph-template repair; fresh main `230eac1` / ELF `270a6bb4...` passes P40000/O16 and separate short integration | Complete capability, repeated startup reliability, and release qualification remain open; original candidate-panel gains are not a new main speedup |
| Ordinary request-state reuse | Preserved 0.7.0 lifecycle and original v1/v16 scopes; current-main v20 P40 and ten short P/O/reset receipts pass. Historical v3 BCCB/lifecycle evidence remains separately scoped | Complete accuracy, independent-process target-length repetition, capability, and release qualification remain open; no Prefill-throughput claim is attached to reset |
| Target-length Prefill | Current installed main executes terminal layer-63 prefix elision with incumbent QT2/GroupQ64, exact-span GDN, and prompt-wide preprocessing; its actual P40000/O16 request reports 60.271514903 prompt tok/s | Complete accuracy, P60/P130, the 2s/4s targets, and further accuracy-preserving whole-product optimization remain open; closed lineages and same-skeleton span scans remain excluded |
| SM87 whole-system AOT Prefill candidate | `AC-PREFILL-SM87-AOT-SYSTEM-v1` is default-off, non-executable, and paused. Real-checkpoint upload/readback/private attachment plus the layer-0 M192 Oracle remain retained prerequisites. The P40000 BF16-HMMA skeleton is rejected at 225.7838x over budget, and `bmma-static-support-k16-parent-zero-fill-v2` is separately rejected before CUDA after its authenticated mandatory-instruction lower bound exceeds the complete five-second projection allocation | No active AOT implementation gate. Resumption requires an explicitly named materially different exact arithmetic/dataflow class with a new bounded proof or a successor architecture; persisted direct loading remains prerequisite work only after such a resumption |
| Prefill/Decode phase identity | Logically separated | Physical scheduling and state ownership do not yet provide an independently optimized/overlapped production pipeline |
| Decode | Exact S>=65 fallback plus retained coupled-feed/consumer-order layouts and fixed short-position Graph cache; current-main short integration reports 9.553634715 tok/s and P40 reports 3.939689060 tok/s. The 2026-09-26 P40000 decode-step nsys attribution (T4) shows the 254 ms/token step is 58.6% exact full-attention (values kernel 111.7 ms/token, 6,144 threads, 6x GQA-redundant V loads; scores kernel 38.6 ms/token) and 38.1% weight-read GEMV at its prior ceilings; launch overhead is 0.28%. The values kernel is latency-bound, not bandwidth-bound: a matched access-pattern microbenchmark shows the serial FMA chain at unroll-4 is 6.4 ms/layer, and unroll-8 (bit-exact, same FMA order) is 3.4 ms/layer. The unroll-8 change is committed and validated on the real P40000/O16 API: decode 3,855.8 -> 3,111.4 ms for 15 steps (257.0 -> 207.4 ms/step, 3.89 -> 4.82 tok/s), prefill unchanged, 16 completion tokens byte-identical to the 44842df baseline. The follow-up unroll-8 -> 128 deepening (bit-exact, same FMA order) is committed and validated the same way: decode 3,111.4 -> 2,895.2 ms for 15 steps (207.4 -> 193.0 ms/step, 4.82 -> 5.18 tok/s), 16 completion tokens byte-identical to the unroll-8 baseline. The 2026-09-26 hardware-bound analysis (T4, no route selected) measures the pure-read ceiling at 182.5 GB/s, the exact per-step weight read at 18.52 GB, and therefore the batch-one weight-read floor at 101.5 ms/step (9.85 tok/s): the 10 tok/s locked target is below the hardware floor for this model on this device. A follow-up measurement (2026-09-26, corrected) shows the 6x GQA-redundant values V read is a REAL DRAM cost, not L2-absorbed: the values kernel's 40.3 ms/layer at S=40000 matches the pure-DRAM time for its 6x requested volume (7.86 GB / 182.5 GB/s = 43 ms), and the earlier '195 GB/s above the ceiling, therefore L2' reasoning is inside the measurement noise band. A split-chunk GQA-shared values kernel (dedicated TU, 512 threads/block, 32 KiB smem V tile, 6 query heads per group, atomicAdd combine; opt-in Q3X_ENABLE_SPLIT_VALUES, S>=512, capture-guarded) reads each V row once and is validated end-to-end on the real P40000/O16 API: decode 2,894.9 -> 2,478.1 ms for 15 steps (193.0 -> 165.2 ms/step, 5.18 -> 6.05 tok/s, 1.17x; microbenchmark 1.66x per layer). The split path is NOT bit-exact (non-deterministic cross-chunk atomicAdd order) and the two runs produce different 16-token completions, so it is an evaluation route only; the production default remains the bit-exact unroll-128 kernel. The realistic batch-one ceiling with the split values path is ~165 ms/step (~6.1 tok/s); a follow-up microbenchmark (2026-09-27) found the scores kernel is NOT bandwidth-bound (40.5 GB/s effective 1x K vs 182.7 GB/s ceiling), so the same split treatment buys only 1.14x there; it was still integrated (bit-exact, 0/24,576 diff, no atomicAdd, opt-in Q3X_ENABLE_SPLIT_SCORES) and measured end-to-end with both splits: decode 2,385.5 ms for 15 steps = 159.0 ms/step = 6.29 tok/s (vs 193.0 bit-exact, 165.2 values-only) -- the maximum reachable with local engineering only (GEMV floor 101.4 ms at the DRAM ceiling + split attention ~57 ms). Owner adjudication is requested: batched-decode contract change, lower quantization, re-set target, or accept the non-bit-exact split route for production. Evidence: [decode split-values e2e record](metadata/qwen36-27b-decode-split-values-e2e-2026-09-26.json) | Long-output stability is qualified (256-token short-context runs byte-identical, no divergence, 9.38 tok/s corroborating the floor); the 10 tok/s target and P40000 target-length behavior remain, and the hardware-bound record requests owner adjudication before further architecture work |
| Production accuracy | Partial deterministic oracles | No complete public capability, hidden/state/logit, and release-repeat bundle has passed |
| Canonical release artifact | Fresh main `230eac1` installs 0.7.0 Release/OFF terminal-prefix.v1 ELF `270a6bb4...`; P40 and short integration pass with `production_eligible=false` and `release_qualified=false` | Complete accuracy, capability, target-length repetition, and stability/release attestation remain incomplete |
| Automated release lane | Designed only | Local tests and policies exist, but no checked-in Orin release workflow enforces the complete gate |

Status terms are strict:

- **implemented** means executable code exists;
- **default-off development** means an explicitly selected research route
  exists but cannot describe the release;
- **qualified** means the applicable real-model accuracy, performance,
  repetition, route, and resource evidence has passed;
- **production** means the installed default artifact and API are attested;
- **target** means required capability, not current implementation.

## 3. API and capacity snapshot

The selected ordinary 0.7.0 source uses the terminal-prefix P40 profile
described above and exposes no public capacity or tactic selector. Its fresh
main installation and P40/short API closeout now pass as recorded above. The separate
development evaluation adapter defaults to:

- loopback `127.0.0.1` binding;
- one serialized inference worker behind bounded ingress/inference queues;
- `max_sequence_length=8192`;
- `maximum_output_tokens=4096`;
- Prefill chunk size 512; and
- a 2 GiB request-arena admission limit.

It supports health/model discovery, completions/chat, non-streaming responses,
and committed-token SSE. It remains an evaluation instrument, not the final
serving boundary. The product API requirements are owned by
[`SDD.md`](SDD.md); the executable external procedure and metric semantics are
owned by [`EVALSCOPE_EVALUATION.md`](EVALSCOPE_EVALUATION.md).

The current exact M512 request-state planner reports:

| Maximum sequence length | Planned request arena |
| ---: | ---: |
| 8,192 | 705,593,344 bytes |
| 40,000 | 2,801,096,704 bytes |
| 60,000 | 4,118,856,704 bytes |
| 130,000 | 8,731,016,704 bytes |

The P8192 row is a source-derived planning correction, not a new allocation or
measurement: 615,317,504 persistent bytes + 88,178,688 C512 workspace bytes +
2,097,152 RoPE bytes. FP32 capacity is `max(262144, 24 * max_seq)` elements.
The prior 705,331,200-byte row is 262,144 bytes short: the 196,608-element GQA
view at this length does not replace the larger FP32 capacity floor.
Historical evidence is unchanged. P40/P60/P130 totals are unaffected.

The 2 GiB default therefore rejects P40 before performance is considered.
Raising the command-line limit alone is not a capacity qualification: the
resident checkpoint, derived layouts, Prefill workspace, KV/recurrent state,
cancellation resources, thermal headroom, and request arena have not yet been
proven to fit simultaneously. Whole-process capacity remains indeterminate.

## 4. Current Prefill execution and attribution

On the eligible Legacy-C512/SM87WeightOnly development route, Embedding now
gathers each complete admitted Prefix tile and full-Attention preprocessing
uses the exact prompt-wide 128-thread mapping for M>=2. Both reuse existing
request storage, retain the M1 reference path, and are excluded from sealed
exact-arithmetic and non-Legacy scopes. Their measured upward effect is the
small controlled 4K--32K direction reported in section 1; it must not be
added to or confused with the unrelated P40 whole-core observation below.

### Selected ordinary terminal-prefix source route

Ordinary eligible P40000/O16 generation executes `78*C512+C32+C31` prefix
tiles and the unchanged scalar final prompt step: 80 elided prefixes and 81
logical passes. Layers 0--62 and every terminal Q/K/V/cache row are retained;
only production-dead terminal Attention, O, residual/post-norm, and MLP/final
norm suffix work is skipped. Scalar prefixes, scalar final, Decode, tracing,
and direct-runner/retained-hidden contracts are unchanged.

The observed candidate v20 native logical dispositions are Gate/Up and Down
`0/5104`, QKV `7488/288`, Z `3744/144`, O `4914/190`, Attention `1200/16`, and
GDN `3744/144` (production/exact fallback). These are logical role receipts,
not OFF kernel-launch counts. No score-feed code enters this integration.
The following earlier profile explains the pre-elision incumbent; its counts
and percentages are not relabelled as the selected default's attribution.

### Split-P tensor prefill route, 2026-09-21 (ADR-0002)

By explicit owner instruction, the production prefill full-attention numerical
class is aligned to the vLLM/FlashInfer BF16-probability deployment class. The
dispatch predicate now routes every legal C2..C512 prefill chunk to the
grouped-Q64 WMMA tensor kernel with split-P probability (P_hi + P_lo, two
mma, FP32 denominator). The pinned P40000/O16 real API request on this build
reports 219.8 s pure prefill (vs 663.7 s scalar, ~3.0x) with 16 coherent
generated tokens. The generated text differs from the historical scalar-oracle
text from token 0: the oracle's own BF16 logits are an exact top-2 tie at
token 0 (margin 0.0), which any BF16-input tensor route can flip. This is
recorded in [ADR-0002](decisions/0002-prefill-attention-vllm-numerical-alignment.md);
the scalar kernel remains the reference oracle. An nsys profile of this build
attributes the 219.5 s GPU prefill to 36.4% full attention (split-P tensor),
29.4% MLP Gate/Up (NVFP4), 14.4% QKV/O projection (FP8), 12.3% GDN linear
attention, and 7.5% small-M projections/norms/conv; projections are now the
largest cost (43.8% combined). Measured hardware floor: the Orin's true BF16
dense peak is 33.5 TFLOPS on the production MLP shape (C8000x17408x5120) and
26.1 TFLOPS on an 8192^3 square GEMM (MAXN power, 57 C, no throttle); P40000
prefill of the 27B dense model is 2.2e15 FLOPs, giving a 65.7 s FLOP floor at
the MLP-shape peak (84.6 s at the square rate, 8.0 s at the INT8-sparse
ceiling). The current 219.5 s Legacy-C512 route runs at 30% of the measured
MLP-shape peak. The 2 s prefill target would need 1100 TFLOPS (33x the measured
peak) and is physically unreachable on Orin for this model. The layer-major
whole-core architecture (5x M8000 panels, whole-prompt FlashInfer attention,
persistent NVFP4 large-M MLP) was reproduced from a clean
`orin-p40-whole-core-dev` build on 2026-09-21: 101.34 s and 101.56 s pure
prefill over two clean-host real-API runs (first token "Based", matching the
split-P production output), 2.17x the Legacy-C512 219.76 s. That is 21.7
TFLOPS effective, 65% of the measured MLP-shape peak, and matches the
historical v10 incumbent (101,831.85 ms) within 0.5%. It remains
accuracy-unqualified (inherited FlashInfer P513 full-state mismatch) and
default-off; the production route is still the Legacy-C512 split-P build.
The achievable headroom on the layer-major route is kernel efficiency toward
the floor (~66-80 s at 80-100% of peak).

### Pre-elision ordinary main attribution, 2026-09-09

The fresh installed Release/OFF main artifact at `d6565eb` (ELF
`b4ccef99...`, Build ID `99f5d3f0...`) completed one bounded real
P40000/O16 API profile. Its unique generation range contains 80 ordinary
prefix tiles, the existing scalar final prompt step, and 15 Decode steps.
QT2 Attention accounts for 557.798038208 s across 1,248 launches, or
79.84165% of the Prefill kernel-duration sum. The complete Prefill NVTX window
is 699.202508416 s; cross-stream kernel union is 698.508698848 s, leaving
0.693809568 s (0.09923%) outside kernels. The 0.121739808-s Gate/Up overlap
is counted once in that union. This establishes QT2 as the dominant pre-elision
ordinary Prefill cost, not a candidate speedup or a performance prediction.

All 332,976 generation kernels bind uniquely to the exact server process and
leaf phase, and the official Nsight reports reproduce the per-phase kernel
counts and nanosecond totals. The profiled engine-call wall also includes
9.417331081 s outside the generation NVTX range; that time is not assigned to
Decode or the in-range GPU gaps. Full roles, hashes, source/workload identity,
cross-checks, and diagnostic-only limits are frozen in the
[ordinary-main P40 attribution record](metadata/qwen36-27b-ordinary-main-p40-nsys-attribution-2026-09-09.json).
The same installed-main artifact also passed the short EvalScope integration
closeout in section 1; these earlier observations did not qualify or select
either Prefill candidate.

### P40000 decode-step attribution, 2026-09-26 (T4 diagnostic)

A fresh nsys capture of one real P40000/O16 API request on the installed
`orin-release` Legacy-C512 server (commit `56012ee`, `--nvtx-phase-ranges`)
attributes the 254 ms/token decode step (15 steps, 437 kernels each, witness
3,855.8 ms after first token). The step is 99.7% GPU kernels with a 0.28%
host-launch gap, so launch overhead and Graph-cache extension are not the
P40000 lever. Full attention over the 40K KV is 58.6% of the step:
`attention_values_exact_24_4_256` at 111.7 ms/token (grid 6x4x256 = 6,144
threads; its `value_index` is independent of the query-within-KV index, so the
6 query heads per KV head each issue identical V loads, a 6x redundant global
request pattern) and
`attention_scores_warp_positions_24_4_256` at 38.6 ms/token (well-parallelized
at 120,024 warps). The remaining 38.1% is the weight-read GEMV/MLP floor,
already at its measured ceilings per the retained decode sidecar records. The
exact DRAM traffic of the redundant loads (versus L2 absorption) is not
determined by this capture because ncu is unavailable on Orin; a matched
access-pattern microbenchmark is the required next diagnostic. Exact
identities, the per-kernel table, the corrected bandwidth note, and claim
limits are frozen in the
[P40000 decode-step attribution record](metadata/qwen36-27b-p40000-decode-step-attribution-2026-09-26.json).
This is a T4 diagnostic; it claims no speedup and selects no route.

### P40000 decode values-kernel unroll-8, 2026-09-26 (bit-exact local optimization)

The matched access-pattern microbenchmark required by the attribution above
shows the exact S>=65 values kernel is **latency-bound, not bandwidth-bound**:
the serial position-ordered FMA chain (one 40K-FMA dependency chain per output
dimension, 6,144 threads) runs at 6.4 ms/layer at unroll-4 and 3.4 ms/layer at
unroll-8, while a position-parallel variant (which would reorder the FMA and is
not bit-exact) reaches 0.4 ms/layer. The only bit-exact lever is deeper unroll,
which raises memory-level parallelism without changing the FMA order. The
`#pragma unroll 4 -> 8` change in
[`attention_values_exact_24_4_256_kernel`](../src/kernels/reference/decode_ops.cu)
is validated two ways: the component test
`q3x_decode_ops_cuda_test` reports `ATTENTION_VALUE_BINARY_IDENTITY ...
status=PASS` (29 boundary lengths, 16 special BF16, 14 special FP32, replay
enabled) with registers 40 -> 36 and static shared still 0, and the real
P40000/O16 `/v1/completions` API returns 16 completion tokens byte-identical to
the 44842df baseline. On the real API the decode step falls from 3,855.8 ms to
3,111.4 ms for 15 steps (257.0 -> 207.4 ms/step, 3.89 -> 4.82 tok/s, +23.9%);
pure prefill is unchanged at ~219.6 s, confirming the gain is decode-only. The
measured 49.6 ms/step saved matches the microbenchmark prediction (6.4 -> 3.4
ms x 16 layers = 48 ms) almost exactly. Decode is still below the 10 tok/s
locked target; the remaining gap needs a GQA-redundancy-free or bit-exact
position-parallel values dataflow, a larger architecture change. Evidence is
frozen in the
[unroll-8 API validation record](metadata/qwen36-27b-p40000-decode-unroll8-api-2026-09-26.json).

### P40000 decode values-kernel unroll-128 and hardware-bound analysis, 2026-09-26

Deepening the same bit-exact lever, the matched access-pattern microbenchmark
(S=40000, per layer) shows unroll 8 / 16 / 32 / 64 / 128 = 3.45 / 3.25 / 2.83 /
2.67 / 2.52 ms (131.7 GB/s effective at 128, against the 182.5 GB/s measured
pure-read ceiling). The `#pragma unroll 8 -> 128` change in
[`attention_values_exact_24_4_256_kernel`](../src/kernels/reference/decode_ops.cu)
is validated two ways: `q3x_decode_ops_cuda_test` reports
`ATTENTION_VALUE_BINARY_IDENTITY ... status=PASS` with resources unchanged
(registers 40, static shared 0, local 0, active blocks 6), and the real
P40000/O16 `/v1/completions` API returns 16 completion tokens byte-identical to
the unroll-8 baseline. On the real API the decode step falls from 3,111.4 ms to
2,895.2 ms for 15 steps (207.4 -> 193.0 ms/step, 4.82 -> 5.18 tok/s, +14.4
ms/step, matching the microbenchmark prediction of 14.8 ms/step); pure prefill
is unchanged at ~219.6 s. The 6-chain GQA-redundancy-free values merge (6,144
-> 1,024 threads) was measured and rejected: 12.70 ms/layer versus 3.44 ms/layer
for the production shape, a 3.7x regression from occupancy collapse; the
initial illegal-memory-access seen while building that benchmark was a benchmark
indexing bug (the probability index advances +1 per position, not +S), not a
production defect.

The companion hardware-bound analysis (T4 diagnostic; claims no speedup and
selects no route) corrects the bandwidth denominator to the measured 182.5
GB/s pure-read ceiling (the earlier 95.4 GB/s D2D figure is read+write
bidirectional and understates pure-read efficiency by ~2x), fixes the exact
per-step weight read at 18.52 GB (MLP NVFP4 packed 10.16 + GDN proj FP8 5.59 +
full-attn proj FP8 1.68 + lm_head 0.72 + norms/misc 0.38 GB, excluding the
2.552 GB embed full table and MTP), and therefore places the batch-one
weight-read floor at 101.5 ms/step (9.85 tok/s). The 10 tok/s locked target
(100 ms/step) is below that floor. A follow-up measurement (2026-09-26, **corrected the same day**) shows
the 6x GQA-redundant values V read is a **real DRAM cost, not L2-absorbed**:
the values kernel's 40.3 ms/layer at S=40000 matches, within noise, the
pure-DRAM time for its 6x requested volume (7.86 GB / 182.5 GB/s = 43 ms),
and the earlier '195 GB/s above the ceiling, therefore L2' reasoning sits
inside the measurement noise band. A split-chunk GQA-shared values kernel
(dedicated translation unit, 512 threads/block, 32 KiB smem V tile, 6 query
heads per group, atomicAdd combine; opt-in `Q3X_ENABLE_SPLIT_VALUES`, S>=512,
capture-guarded) reads each V row once and is validated end-to-end on the real
P40000/O16 API: decode 2,894.9 -> 2,478.1 ms for 15 steps (193.0 -> 165.2
ms/step, 5.18 -> 6.05 tok/s, 1.17x; microbenchmark 1.66x per layer). The
split path is NOT bit-exact (non-deterministic cross-chunk atomicAdd order)
and the two runs produce different 16-token completions, so it is an
evaluation route only; the production default remains the bit-exact
unroll-128 kernel. The realistic batch-one ceiling with the split values path
is weight floor 101.5 + split values ~24.0 + scores 38.6 + launch ~0.7 =
~165 ms/step (~6.1 tok/s); the measured 165.2 ms/step (6.05 tok/s) matches
it. A follow-up microbenchmark (2026-09-27) then tested the same treatment on
the scores kernel (38.55 ms/step, the largest single decode kernel): a
bit-exact split-chunk variant (K tile in smem, 1x read, 0/24,576 diff at
S=1024) buys only 1.14x per layer (2.023 -> 1.781 ms), because both the
production and split scores kernels run at ~40-46 GB/s of 1x K volume, far
below the 182.7 GB/s ceiling -- the scores kernel is not DRAM-bandwidth-bound
(its cost is the per-position shuffle tree and FMA issue), so the direction is
rejected as a production change. The short-context 9.51 tok/s measurement is consistent
with the floor. Evidence is frozen in the
[decode split-values e2e record](metadata/qwen36-27b-decode-split-values-e2e-2026-09-26.json). This record requests owner adjudication among a batched-decode
contract change (amortize the 18.52 GB weight read across concurrent
requests), lower weight quantization (accuracy risk, outside the current
production accuracy scope), or re-setting the P40000 target to a reachable
value; it changes no target and selects no route. Evidence is frozen in the
[decode hardware-bound analysis record](metadata/qwen36-27b-decode-hardware-bound-2026-09-26.json).

The scores kernel was examined for the same deep-unroll lever and confirmed
**not applicable**: `attention_scores_warp_positions_24_4_256_kernel` is
position-parallel (one warp per position) with the dimension reduction already
a fully unrolled shuffle tree, so it has no serial position-FMA chain to
unroll; its 6x GQA key read was measured (2026-09-27) with a bit-exact
split-chunk GQA-shared variant: only 1.14x per layer, because the scores
kernel runs at ~40-46 GB/s of 1x K volume, far below the 182.7 GB/s pure-read
ceiling -- it is not DRAM-bandwidth-bound, so unlike the values kernel the
redundancy is not its bottleneck and the direction is rejected.

The 6x GQA redundancy was first attacked with a **bit-exact**
shared-memory merge: one block per KV head (1,024 threads, the sm_87 maximum)
serves all six query heads from a single smem V tile, cutting the V read from
6x to 1x while preserving each (query, dimension) position-ordered FMA
sequence. It is bit-identical to the CPU oracle (0/6,144 diff at S=1024), but
measured **3.19x slower** (8.865 vs 2.782 ms/layer): the 1,024-thread block
collapses occupancy to 1 block/SM and adds per-tile `__syncthreads`. (The
original rejection reason -- 'the 6x redundancy was already free via L2' --
was wrong and is corrected above.) The same merge re-shaped as a 512-thread
split-chunk kernel (grid (chunks, 4), 32 KiB smem per block) removes the
occupancy collapse and measures **1.66x faster** per layer (1.689 vs 2.799
ms/layer at S=40000); its cost is bit-exactness (atomicAdd combine), which is
why it ships as the opt-in evaluation route rather than the production
default. The bit-exact local optimization route is exhausted at unroll-128;
the non-bit-exact split route is the validated next lever, and the scores
kernel's 6x GQA key read is the same treatment's next target.

To qualify the unroll-128 values kernel beyond the 16-token check, a long-output
stability run was performed at a short context (first 1089 tokens of the
P40000 body, still on the exact S>=65 values route; `max_tokens=256`,
`temperature=0.0`, `seed=42`): two independent 256-token runs are byte-identical,
contain no NaN/inf, and show no divergence over 255 decode steps at 106.6
ms/step (9.38 tok/s). That short-context rate independently corroborates the
9.85 tok/s weight-read floor. This is a bounded decode qualification; it does
not change the P40000 10 tok/s reachability conclusion. Evidence is frozen in
the [decode long-output stability record](metadata/qwen36-27b-decode-longout-stability-2026-09-26.json).

Per-kernel nsys attribution of the split values route (2026-09-27, same-ELF
back-to-back, no ceiling claim): in the full-model nsys environment the
unroll-128 values kernel runs 50.48 ms/step (3.16 ms/layer) and the split
kernel 22.79 ms/step (1.42 ms/layer); the entire -27.79 ms/step kernel-time
delta is the values-kernel swap plus a 0.07 ms/step finalize, with every other
kernel moving <0.05 ms/step (noise), and the delta matches the non-profiled
e2e delta (27.8 ms/step) to within 0.1 ms. The nsys environment inflates
memory-bound kernels relative to the isolated microbenchmark (2.799 / 1.689
ms/layer), so the two measurement environments are kept separate. The two
premise numbers behind the earlier ceiling arithmetic were re-measured the
same day and hold: pure-read bandwidth 182.5 -> 182.7 GB/s (4 GB
read-accumulate) and per-step weight read 18.52 -> 18.529 GB (safetensors
header sum); the same-nsys GEMV kernels run at ~184 GB/s effective, i.e. at
the pure-read ceiling. Evidence: the
[split-vs-baseline nsys attribution record](metadata/qwen36-27b-decode-split-vs-baseline-nsys-2026-09-27.json)
and the
[premise re-measure record](metadata/qwen36-27b-decode-premise-remeasure-2026-09-27.json).

Before extending the split treatment to the scores kernel (38.55 ms/step, the
largest single decode kernel), a microbenchmark was run first: a bit-exact
split-chunk GQA-shared scores variant (K tile in smem, 1x read; 0/24,576 diff
at S=1024, no atomicAdd so no token divergence) measures only 1.14x per layer
(2.023 -> 1.781 ms at S=40000). Both the production and split scores kernels
run at ~40-46 GB/s of 1x K volume, far below the 182.7 GB/s pure-read ceiling:
the scores kernel is not DRAM-bandwidth-bound (its cost is the per-position
shuffle tree and FMA issue), so the values-kernel result does not transfer and
the direction is rejected as a production change. This corrects the prior
'scores kernel is the next candidate' framing, which was a prediction rather
than a measurement. Evidence: the [scores split microbenchmark record]
(metadata/qwen36-27b-decode-scores-split-microbench-2026-09-27.json).

A further microbenchmark (2026-09-27) measured the integrated split values
kernel itself: at S=40000 it runs at 62 GB/s (34% of the 182.7 GB/s ceiling)
on its 1x V volume, and the same access pattern with no FMA reaches 105 GB/s
(58%) -- so the split values kernel is also not DRAM-bandwidth-bound, and
ILP-2/ILP-4 reordering (independent accumulators) does not help (1.320 ->
1.323 ms/layer). The 0.54 ms/layer gap is FMA/probability-load/store overhead
that ILP does not remove; this kernel design is near where it will land, and
the dominant remaining decode cost is the weight-read GEMV kernels (100.52
ms/step, 38% of the step, already at the DRAM ceiling). Evidence: the [split
values bandwidth record](metadata/qwen36-27b-decode-split-values-bandwidth-2026-09-27.json).

Per the owner's direction (2026-09-27) to explore all adjudication directions
except lower quantization, the three remaining directions were evaluated
against the measured data and the SDD/ROADMAP contract: (1) batched decode --
the engine is batch=1 by design (the oracle rejects `max_num_seqs != 1`; the
SDD Decode target is single-request; continuous batching is deferred to ROADMAP
P6), so it raises aggregate throughput (toward B x 9.86 tok/s) but does not
satisfy the single-request 10 tok/s target and needs a new product contract;
(4) non-bit-exact split for production -- a real 1.17x (6.05 tok/s) but the
16-token completion diverges from the bit-exact baseline from token 1 (both
coherent, non-deterministic), trading reproducibility, and still < 10 tok/s;
(3) target re-set -- the measured batch-one weight-read floor is 9.86 tok/s,
so 10 tok/s is below the hardware floor and the reachable single-request range
is 5.18 (bit-exact) to 6.05 (split) tok/s. With lower quantization excluded,
no direction reaches the single-request 10 tok/s target; the choice is a
product decision (single-request bit-exact vs non-deterministic vs
multi-request batching), not an engineering one. Evidence: the [directions
evaluation record](metadata/qwen36-27b-decode-directions-evaluation-2026-09-27.json).

The bit-exact scores split path (dedicated TU, opt-in
Q3X_ENABLE_SPLIT_SCORES, S>=512, capture-guarded; 0/24,576 diff at S=1024, no
atomicAdd) was integrated and measured end-to-end on the real P40000/O16 API
with both splits enabled: decode 2,385.5 ms for 15 steps = 159.0 ms/step =
6.29 tok/s (vs 193.0 bit-exact, 165.2 values-split-only). The scores split
contributes 6.17 ms/step, matching the microbenchmark band. This is the
maximum reachable with local engineering only: the 159.0 ms/step is the
weight-read GEMV floor (101.4 ms, 64%, at the DRAM ceiling) plus the split
attention (~57 ms), and neither component has further local headroom
(measured). The reachable single-request range is 5.18 tok/s (bit-exact) to
6.29 tok/s (both splits, non-deterministic due to the values atomicAdd). The
e2e completion is 'Based on the provided repository documents, here is the
analysis of the current state and' -- close to the bit-exact baseline but not
byte-identical. Evidence: the [scores split e2e record]
(metadata/qwen36-27b-decode-scores-split-e2e-2026-09-27.json).

### Historical v10 route only

The following topology and profile describe the historical typed P40 v10
development artifact, not the current ordinary exact main route above. Its
retained layer-major, single-stream route performs, for each of 64 layers:

```text
five M8000 fill panels
  -> one P40000 Attention or GDN core
  -> five M8000 drain panels
  -> one P40000 Gate+Up/SiLU and Down/residual MLP phase
```

That v10 runner's two-slot submission window bounds cancellation and completion
retirement. It is not GPU double buffering: all kernels are submitted to one
CUDA stream, and the P40 whole-core path does not use the older auxiliary
branch stream. That v10 path has no general double- or triple-buffered
cross-panel/cross-layer pipeline.

That historical v10 whole-request NSys capture reports 102.121307 s around
102.113314 s of kernels; only 7.992928 ms, or 0.0078%, lies outside kernels.
It establishes kernel dominance only for that recorded v10 route.

| Dominant role | Calls | Total | Request share |
| --- | ---: | ---: | ---: |
| NVFP4 Gate/Up persistent Marlin | 64 | 37,273.068224 ms | 36.50% |
| FP8 Marlin | 1,040 | 25,864.646560 ms | 25.33% |
| NVFP4 Down persistent Marlin | 64 | 17,559.457280 ms | 17.19% |
| Whole-prompt FlashInfer Attention | 16 | 13,634.170272 ms | 13.35% |

These four roles account for about 92.37% of that historical v10 request;
their shares must not be used to attribute current ordinary P40 behavior.
Architecture selection, composition scope, and the real-API return point are
owned only by [`ROADMAP.md`](ROADMAP.md).

## 5. Retained and rejected Prefill code

The following selected routes are the minimum set needed to interpret the
current incumbent, the closed v11--v15 projection lineage, and the closed
selector/exact-span/Q4/P39936 branches. This is not an experiment inventory;
other earlier screens remain only in frozen evidence. None below is a
production path or an active parameter scan.

| Route | P40 pure prompt throughput | Current disposition |
| --- | ---: | --- |
| v10 whole-core substrate | historical max-clock 392.804397 tok/s; historical main strict sustainable 325.983493208 tok/s | Retained default-off, accuracy-unqualified infrastructure; that earlier integration-health gate passed, historical max-clock not reproduced |
| selector-exact persistent-Attention v1 whole-core composition | not reported; formal P40000/O16 comparison has `timing_authority=false` | Rejected; fresh same-ELF full-model state/logit/token/text mismatch, unmerged and default-off |
| corrected selector-exact persistent-Attention v1 | 27.96118777397028 tok/s | Exact P40000/O16 test-only branch; gross negative API direction, closed and unmerged |
| selector exact-span Attention v2 | 55.94887170163682 tok/s | Exact test-only branch baseline; recovers the corrected selector but remains below ordinary exact main, so no mainline absorption |
| selector exact regular-Q4 Attention v3 | 38.33560402225455 tok/s | Strict-exact, cold-cache real-API test-only branch; rejected 33.0157393% below ordinary exact main, with no repeat, profile, or Q3--Q5 same-skeleton scan |
| FP8 P39936 M128N256 plus Gate/Up dual stream | 51.17353679519425 tok/s | Exact test-only branch; rejected on the real API, dual stream does not rescue P39936 |
| FP8 P39936 M128N256 plus serial MLP | 51.17176853575732 tok/s | Exact clean-commit test-only branch; rejected on the real API and closes P39936 without profiling or repetition |
| Shape-wide NVFP4 v3 replacement | 376.030675 tok/s | Rejected; temporary runner overlay removed |
| v11 grouped projection reset | 205.951777 tok/s | Rejected |
| v12 phase-local BF16 projection | 320.472999 tok/s | Rejected; unsealed historical direction |
| v13 AOT packed projection v1 | 247.814694 tok/s | Rejected |
| v14 packed NVFP4 v2 | 311.300103 tok/s | Rejected |
| v15 stock-Marlin parity reference | 392.705493 tok/s | Rejected; no positive direction |

Exact negative observations and route limitations are frozen in the
[`shape-wide v3`](metadata/qwen36-27b-prefill-p40k-nvfp4-shape-wide-v3-rejection-2026-08-10.json),
[`v11`](metadata/qwen36-27b-prefill-p40k-projection-reset-rejection-2026-08-10.json),
[`v12`](metadata/qwen36-27b-prefill-p40k-phase-local-bf16-rejection-2026-08-10.json),
[`v13`](metadata/qwen36-27b-prefill-p40k-packed-projection-rejection-2026-08-10.json),
[`v14`](metadata/qwen36-27b-prefill-p40k-packed-nvfp4-v2-rejection-2026-08-11.json),
and [`v15`](metadata/qwen36-27b-prefill-p40k-vllm-marlin-parity-rejection-2026-08-11.json)
records. They remain evidence for their exact protocols; they do not select
the next architecture. The later composed whole-core rejection is frozen in
the
[`selector v1 rejection record`](metadata/qwen36-27b-selector-exact-persistent-attention-v1-rejection-2026-08-27.json).
The later exact branch metrics, strict-oracle joins, route counts, artifact
hashes, and non-main boundary are frozen in the
[`exact-span/P39936 closeout`](metadata/qwen36-27b-exact-span-p39936-api-lineage-closeout-2026-09-05.json).
The regular-Q4 child and its no-scan boundary are frozen separately in the
[`Q4 rejection record`](metadata/qwen36-27b-selector-exact-regular-q4-span-v3-rejection-2026-09-05.json).

A previously discussed Factorized-R1 research direction changes the numerical
trajectory. No tracked qualification/evidence record currently gives it
current measurement authority. It is not mainline, lossless, or
production-eligible and must not be reported as current Prefill performance.

## 6. Accuracy and release state

Accuracy is a hard production constraint. Any changed precision, recurrent
state boundary, reduction tree, logits, or generated behavior must first pass
the declared exact/no-regression numerical and behavioral gates. An
approximate route or changed product numerical contract may not enter the
mainline without an explicit owner amendment.

Current evidence is incomplete:

- selected native routes have deterministic component, state, token, and
  output oracles;
- selected terminal-prefix source has the ordinary-policy same-ELF P40000/O16
  full live-state/used-KV and 16 full-vocabulary-logit-digest comparison plus
  P576 dual-poison/canary evidence; the separate OFF candidate also preserves
  exact API text across the matched panel. This is not complete capability
  qualification. Fresh installed-main P40 and short integration also pass, but
  do not expand those numerical or public-capability scopes;
- the absorbed Legacy-C512 Embedding and full-Attention preprocess mechanisms
  match complete persistent/used-KV state, generation, and public logits at
  P514/P4096/P8192, including 15 P4096 Decode transitions;
- the retained Gate/Up and Down Decode sidecars reproduce one identical
  generated UTF-8/SSE text hash across 24 controlled P1024/P4096 requests,
  but that comparison exposes no token-ID, hidden-state, or logit oracle;
- the short cumulative native route reproduced its comparator on 8/8 outputs;
- the first external native/vLLM comparison matched text on 26/32 requests,
  but vLLM is not the accuracy oracle;
- per [ADR-0002](decisions/0002-prefill-attention-vllm-numerical-alignment.md)
  (owner instruction 2026-09-21), the prefill full-attention numerical class
  is now the vLLM/FlashInfer BF16-probability class with split-P; token-
  identical scalar-oracle reproduction is waived for tie-prone prompts, the
  synthetic gate passes all legal C2..C512 shapes (nrmse <= 1.7e-4), and the
  pinned P40000/O16 real API run is retained as the accepted evidence;
- the retained FlashInfer P40 direction has a known P513 full-state mismatch;
- the later selector-exact persistent-Attention v1 whole-core composition is
  formally rejected at P40000/O16 because Prefill-commit state,
  generation-return state, exposed logits, token IDs, and text differ from
  the Legacy exact comparator;
- the corrected selector artifact and its later exact-span, regular-Q4, and
  P39936 descendants each pass the separate strict P40000/O16
  actual-generation state/logit/token/text comparison to the frozen exact
  oracle, but remain test-only, uninstalled, non-main artifacts; regular Q4
  and both P39936 routes are performance-rejected, and the server plan's
  release self-qualification remains pending; and
- the first public C-Eval attempt produced no parseable answer within its cap,
  so the zero score is a protocol failure rather than a model-capability
  measurement.

A release still needs a frozen public capability baseline, exact request and
output contract, full state/logit oracles for every changed numerical route,
independent-process repetition, and the installed-artifact attestation.

## 7. Open gaps

These rows report current facts only. They do not reorder the work; the active
sequence and successor identity live exclusively in
[`ROADMAP.md`](ROADMAP.md).

| Gap | Current fact | Roadmap owner |
| --- | --- | --- |
| Documentation-control propagation | The canonical main line now has one `AGENTS.md -> docs/README.md` Codex entry; pre-existing dirty worktrees do not receive it until explicitly integrated, because Codex reads the worktree in which a session starts | P0 |
| Product API and long-context admission | Installed sealed P40 profile admits 40K plus normal Decode output; P60/P130 profiles and full cancellation semantics remain | P1 |
| Exact deliverable identity | Fresh installed-main 0.7.0/terminal-prefix.v1 OFF identity and both P40/short closeouts pass; production/release flags remain false and earlier v3 tuples remain historical | P2/P4 |
| Prefill parity and physical plan | Current installed-main P40 reports 60.271514903 prompt tok/s; the original matched candidate panel remains separately scoped. The locked target remains open; rejected combined-v1, selector/Q4/P39936 and AOT lineages stay excluded, and isolated score-feed v2 stays paused | P3 (active after this window's completed integration handoff; any successor retains strict P40000/O16 and real-API return) |
| Accuracy, capability, stability, and release evidence | Partial oracles only; no complete qualification bundle | P4 |
| Packaging and operations | No attested install, startup, upgrade, or rollback lane | P5 |

## 8. Claim boundary

Use the following language until this snapshot changes:

- **Current:** selected 0.7.0 Release/OFF source uses
  `q3x.sm87.candidate.p40.legacy-c512-terminal-prefix.v1`, private terminal-prefix
  elision and v20 route/reset receipts, with incumbent QT2/GroupQ64 and retained
  exact-span GDN, prompt-wide preprocessing, Decode layouts, request reuse, and
  startup repair. Both matched candidate P40000/O16 API pairs are positive;
  fresh main `230eac1` / installed OFF ELF `270a6bb4...` now passes P40 API and
  separate short integration. `production_eligible=false` and
  `release_qualified=false` are unchanged. This is not a new matched speedup.
  No score-feed code, numerical relaxation, public tactic selector, or new
  capacity enters this source integration. Previous v3/v16 installed-artifact
  results qualify only their historical scopes, not the new main artifact.
  Complete capability, accuracy, repeated startup reliability, stability, and
  target-length qualification remain open.
- **Not current:** production-default P60/P130 support, the accuracy-unqualified
  FlashInfer v10 arithmetic as a default, the rejected selector-exact
  persistent-Attention v1 composition, the test-only exact-span result, the
  rejected regular-Q4 result or Q3--Q5 same-skeleton span scan, either rejected
  P39936 route, their candidate-only P40016 capacity fixes, an active AOT
  implementation gate, any archived V4 construction route, lossless
  Factorized-R1 Prefill, vLLM parity, or a fully qualified 10-token/s Decode
  release.
- **Target:** the accuracy-preserving, non-MTP, OpenAI-compatible runner and
  performance region locked by the Constitution.

Performance-evidence authority and host-preparation rules are owned by
[`REAL_MODEL_PERFORMANCE_POLICY.md`](REAL_MODEL_PERFORMANCE_POLICY.md); this
status snapshot does not duplicate or alter them.
