---
q3x_document:
  id: q3x-current-status
  class: active
  status: active
  owner: project-maintainers
  authority: current implementation, qualification, production, metric, and blocker snapshot
  effective: 2026-08-10
  last_reviewed: 2026-08-11
  supersedes: []
  superseded_by: []
  ssot_for: current delivered state and open production gaps
  review_trigger: any default route, capability, qualification, metric, release, or blocker change
---

# Qwen3x-Orin current status

Snapshot date: 2026-08-11. Baseline revision `a46d165` implements an
exact-P40000 whole-core route: each layer executes five M8000 fill panels,
one whole-prompt core, five M8000 drain panels, and one layer-wide NVFP4 MLP
phase. Its clean-host cold/no-cache OpenAI API/EvalScope direction consumed all
40,000 tokens and reached 101.831854 s pure Prefill, or 392.804397 prompt
tok/s. Against retained revision `c45b7c5` at 108.981855 s / 367.033577
prompt tok/s, latency fell by 7.150001 s and throughput improved by 7.02138%.
EvalScope TTFT was 101.87053 s, only 3.730501 ms beyond server TTFT, so the API
is not the bottleneck.

The v10 witness records one complete 64-layer route pass, 768 bounded
submission retirements, 320 fill and 320 drain panels, 1,040 FP8 launches, 48
BF16 A/B calls, 48 GDN calls, 16 whole-prompt FlashInfer calls, and 64+64
persistent NVFP4 MLP calls. Prefix-cache, MTP, cuBLASLt, external-reference,
approximate, exact-fallback, and forbidden-route hits are all zero. Bounded
NSys covers a 102.121307-s whole-request interval with 102.113314 s of kernels;
the 7.992928-ms gap is only 0.0078%. Gate/Up, FP8, Down, and Attention consume
37.273068 s, 25.864647 s, 17.559457 s, and 13.634170 s respectively.

This is the strongest whole-product P40K development direction, but it is not
a product-gate pass or a production route. The inherited FlashInfer arithmetic
has a known P513 full-state mismatch; the candidate is default-off,
accuracy-unqualified, and measured from a pinned dirty-tree binary. The
whole-core control/memory/witness infrastructure is retained. Its inherited
16-CTA persistent Marlin bodies are not accepted as the target large-M
architecture because they only reorder old raster work and provide no
cross-CTA decoded-B/scale sharing. P60K and P130K remain locked behind a
competitive, accuracy-admissible P40K result, and production status is
unchanged. Exact evidence is in the
[whole-core direction record](metadata/qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json).

The first shape-wide NVFP4 replacement then returned through the same real
API P40K gate. Its three-stage, two-CTA/SM Gate/Up and Down package reached
106.374301 s / 376.030675 prompt tok/s, a 4.460733% latency regression against
the whole-core baseline. One bounded profile attributes a 4.118958-s increase
to those two roles: Gate/Up regressed 10.437332% and Down regressed 1.302115%.
This closes the skeleton before accuracy/repetition work or P60/P130. The
temporary runner overlay has been removed; only its independent default-off
experiment surface and immutable
[v3 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-shape-wide-v3-rejection-2026-08-10.json)
remain. The 392.804397 tok/s whole-core direction is still the strongest P40K
result.

The complete P40 projection-reset candidate subsequently replaced all FP8
QKV/Z/O and both NVFP4 MLP projections with grouped P40000 kernels while
preserving the v10 whole-request transaction. Its v11 API witness consumed all
40,000 tokens, committed the same one-token smoke output as v10, and reported
128 FP8 plus 128 NVFP4 physical launches with no forbidden route. It still
regressed decisively: pure Prefill was 194.220222 s / 205.951777 prompt tok/s,
90.7264% more latency than v10. Bounded NSys attributes 103.177068 s to the
new FP8 kernels, 45.718603 s to Gate/Up, and 24.065402 s to Down. That package
adds 92.263901 s while all non-projection work is net 0.355825 s faster, and
kernel time covers 99.997% of the request. The 16/32-CTA persistent projection
skeleton is therefore rejected; API/host work is not the cause. The route is
test-only, default-off, accuracy-unqualified, and frozen in the
[v11 rejection record](metadata/qwen36-27b-prefill-p40k-projection-reset-rejection-2026-08-10.json).
P60 was not run. The v10 392.804397 tok/s result remains the incumbent.

The subsequent default-off phase-local candidate expanded each canonical
NVFP4/FP8 weight into short-lived BF16 scratch and consumed it with a native
M128N128K64 BF16 Tensor Core grid. Its live experimental v12 output reported
416 FP8 physical launches; the 512 NVFP4 MLP total was inferred from the
successful-enqueue composition, not independently observed. No raw witness
was retained, and its immutable topology disagreed with the physical lowering,
so it has no sealed-route authority. It nevertheless regressed to
124.815475 s / 320.472999 prompt
tok/s; EvalScope TTFT was 124.853852 s. NSys attributes 98.999425 s / 79.4070%
of kernel time to its dense consumers and only 0.363332 s / 0.2914% to all
weight expansion. The BF16-consumer architecture is therefore rejected; it
is not an active tuning surface, did not run P60 or full accuracy, and remains
default-off. The exact result is frozen in the
[phase-local rejection record](metadata/qwen36-27b-prefill-p40k-phase-local-bf16-rejection-2026-08-10.json).

The complete AOT packed-operand projection v1 package is now also implemented
and has crossed the same external P40 gate. Engine load prepared 256 physical
artifacts from all 400 real-checkpoint projection sources into a
16,840,130,560-byte resident arena. Its independent v13 witness retained the
v10 transaction, consumed all 40,000 tokens, and attested 208 logical FP8
roles in 128 physical launches plus 64 Gate+Up and 64 Down roles in 128 NVFP4
launches, with every forbidden-route count zero. Thus the API, asset, sealed
plan, and request route are complete for this exact development screen.

Performance rejected the package: pure Prefill was 161.410929 s /
247.814694 prompt tok/s and EvalScope TTFT was 161.44732 s with
247.758768 New Prompt tok/s. Against v10, latency increased by
59.579075 s / 58.5073% and throughput fell 36.9114%. EvalScope exceeded
server TTFT by only 3.807390 ms, so the API is not causal. Packed v1 remains
default-off and accuracy-unqualified; full accuracy/repetition and P60/P130
did not run. The v10 392.804397 tok/s direction remains the incumbent. Exact
evidence is frozen in the
[packed projection rejection record](metadata/qwen36-27b-prefill-p40k-packed-projection-rejection-2026-08-10.json).

The subsequent packed-NVFP4-v2 package isolated that question: it restored
the v10 FP8 path and replaced only Gate+Up and Down with shape-specific
M128N256 packed NVFP4 consumers. The real-model load authenticated 128
NVFP4-only artifacts from 192 checkpoint sources in a 9,625,927,680-byte
engine-lifetime arena. Its independent v14 witness consumed all 40,000 prompt
tokens, recorded the restored 1,040/1,040 FP8 ledger and 64/64/128 NVFP4
ledger, completed every operator on the intended route, and reported zero
cache, MTP, cuBLASLt, external, approximate, exact-fallback, and forbidden
route hits.

That valid whole-product result was negative. Server pure Prefill was
128.493372 s / 311.300103 prompt tok/s; EvalScope reported 128.53205 s TTFT
and 311.206330 New Prompt tok/s. Against v10, latency increased 26.181904%
and pure-Prefill throughput fell 20.749333%. It did recover 20.393636% latency
relative to rejected packed v1, proving that removing packed FP8 contamination
and changing NVFP4 ownership mattered, but it did not beat the incumbent.
Packed NVFP4 v2 is therefore closed without a parameter scan, remains
default-off and accuracy-unqualified, and did not unlock repetition, full
accuracy, P60, or P130. The v10 392.804397 tok/s direction remains the
incumbent. Exact evidence is frozen in the
[packed NVFP4 v2 rejection record](metadata/qwen36-27b-prefill-p40k-packed-nvfp4-v2-rejection-2026-08-11.json).

The latest mathematical scope audit also closes two misleading shortcuts.
With v10 FP8/Attention/GDN and all other work frozen, subtracting measured
Gate+Up and Down still leaves 46.999328 s, so even a hypothetical zero-time
NVFP4 MLP would expose only about 851.075992 prompt tok/s. This is an
NVFP4-only work-package bound, not a hardware or project upper bound; the
quantity-changing architecture must cover the other dominant roles. Separately,
an exhaustive real-checkpoint scan of all 192 MLP weight tensors found only
7.945540% exact E2M1 zero codes and 3.540535% K4 groups with at least two
zeros. Exact sparsity is therefore not an active successor; the inventory is
frozen in the
[real-weight zero-structure audit](metadata/qwen36-27b-nvfp4-mlp-zero-structure-audit-2026-08-11.json).
The stock-vLLM Marlin source-parity work package has now returned through the
same P40 product gate and supplied no positive direction. The active successor
is qualification-first: it must measure whether real P40 activations admit an
exact block-floating/integer-limb/bit-plane execution class within the system
budget, and whether an exact whole-live-graph rewrite can delete producer-
consumer or recurrent-state traffic, before any corresponding CUDA kernel is
authorized. This is neither lossy activation quantization nor another tile-
parameter scan.

Prefill architecture work now follows the normative order in the
[`PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md`](PREFILL_MATHEMATICAL_EQUIVALENCE_LEDGER.md):
real-number equivalence first; decoded operands, reduction tree, rounding and
state boundaries second; production observables plus complete buffer/control-
state lifetime and alias ownership third; and only then CUDA residency,
ownership, pipeline, fusion, buffering, and synchronization. This is a change
to the design discipline, not a new performance result.

Commit `1cb5ea1` implements the independent, default-off exact-P40000 stock-
vLLM-Marlin **projection reference**. The tree connects that reference through
an independent real-checkpoint sidecar owner, sealed Engine plan, runner, and
`target-prefill-witness-v15` OpenAI API path. Both the ordinary default-off
build and the explicit test-only admission build compile, and their host
weight/plan/view/runner/protocol contracts pass. Commit `986b9fd` then corrected
the P40 commit-profile binding exposed by the first API attempt. A subsequent
r2 timing was discarded because its request preflight observed a non-
allowlisted Codex control process above the stricter project resource limit;
it has no comparison or decision authority. The replacement r3 used a 5%-of-
one-core unexpected-consumer ceiling and passed the server, request, and post-
run `tegrastats` plus complete process/device-handle audits.

The valid r3 request consumed all 40,000 tokens, committed one `Based` token,
completed the exact 64-layer parity ledger with zero forbidden or fallback hits, and
reported 101,857.500727 ms pure Prefill / 392.705493 prompt tok/s. EvalScope
recorded 101,896.324979 ms TTFT and 392.555711 New Prompt tok/s; its
3.905603-ms gap beyond server TTFT excludes the loopback API boundary as the
material bottleneck. Against v10 at 101,831.853876 ms / 392.804397 tok/s, v15
was 25.646851 ms (+0.025185%) slower and 0.025179% lower in throughput. The
single decision sample therefore supplied no positive direction and closes or
redesigns this parity skeleton before repetition, numerical/SASS/state
qualification, P60, or P130. It does not establish a statistically stable
regression. The route remains default-off and accuracy-unqualified; v10 stays
the incumbent and production is unchanged. Exact evidence is frozen in the
[v15 rejection record](metadata/qwen36-27b-prefill-p40k-vllm-marlin-parity-rejection-2026-08-11.json).

Its `transformation_digest` binds the pinned checkpoint provenance, source
manifests, deterministic repack ABI, shape, and schedule. It is intentionally
not described as a byte hash of the generated 9.6-GB device allocation and
does not substitute for the pending real-weight numerical oracle.

That integration audit found a concrete cross-phase lifetime defect before
timing. The proposed parity lock view began at family offset
`4,178,968,576`; relative to the prompt-wide GDN workspace it began at
`2,041,368,576`, inside the GDN W/raw-output live interval
`[1,802,240,000, 2,293,760,000)`. A later linear-layer prompt-core GDN phase
could therefore overwrite locks that a one-request-level clear claimed would
remain zero. The host-integrated correction retains only the 1-MiB reduction
Ctmp in the family temporary and binds the locks to a physically disjoint,
stable legacy-C512 Marlin owner. A pure layout validator proves that prefix
disjoint from persistent conv/GDN and KV state, prompt residual, the complete
P40 family arena, final hidden, RoPE, every other legacy matrix, and FP8
reduction scratch; authority seals that proof and the runner revalidates it
before the one clear. The v15 receipt deliberately attests stable ownership,
alias exclusion, ordered protocol completion, and request-stream completion;
it does not pretend that every intermediate lock value was copied to the host.
The first request before `986b9fd` failed
`whole_request_prefill_commit_contract` before retiring any Prefill quantum and
has no performance authority. Real-weight numerical/state and SASS
qualification were intentionally not run after the valid negative direction,
so the lifetime correction does not make the route production-eligible.

This is the single point-in-time status page. It records what is target,
designed, implemented, qualified, and production. Architecture contracts
belong in [`SDD.md`](SDD.md), pending work in [`ROADMAP.md`](ROADMAP.md), and
immutable measurements in evidence records. A new result must update this
page explicitly before it may be described as the current project state.

## 1. Status vocabulary

| State | Meaning |
| --- | --- |
| **Target** | Required by the owner/constitution, but no claim of design or implementation follows |
| **Designed** | A reviewed contract or architecture exists; executable support does not follow |
| **Implemented** | Code exists on the named revision and can exercise the path; release qualification does not follow |
| **Qualified** | The exact implementation tuple passed its applicable correctness, API, performance, resource, repetition, and evidence gates |
| **Production** | The qualified route is the default installed `BUILD_TESTING=OFF` release, with an attested DeploymentPlan and no test-only composition |

These states are deliberately non-interchangeable. “Runs through the API”
means implemented. “Faster in a direction screen” means evidence about an
implemented candidate. Neither means production.

## 2. Answer-first state

Qwen3x-Orin currently has a real-model, batch-one native runner and a useful
loopback OpenAI-compatible evaluation gateway. It does **not** yet have a
uniquely defined, attested production release. Its strongest short-prompt
results were obtained from a cumulative production-like development build
whose optimized Prefill routes require `BUILD_TESTING=ON`, compile-time
admission options, and runtime route composition. Those results are not the
ordinary default release path.

The default evaluation server is limited to 8,192 sequence tokens and a 2 GiB
per-request arena. It therefore does not deliver the locked cold/no-cache
40K--60K or 130K Agent workloads. The current listener is explicitly an
evaluation adapter: loopback-only, unauthenticated, serialized batch one, and
without a production network, tenant, admission, or cancellation contract.

The development tree now contains an executable, explicit layer-major
compatibility route. `ReferenceEngine` provisions the typed layer-major
`RequestState`, seals the exact model/state/arena/views/streams/two completion
events and 17 operator-role receipts, executes outer-layer/inner-logical-panel
traversal, finalizes logits from uncommitted hidden state, and publishes one
final sequence length through a move-only request receipt. A two-slot bounded
submission window carries API disconnect/shutdown cancellation into long
Prefill without exposing partial state; failures drain and reset through the
Engine transaction guard. Logical and physical tails are balanced without
changing the legacy scheduler, and eligible M32--M512 linear-attention layers
must bind the exact native C64 GDN tactic and its real workspace; only M1--M31
may use the sealed exact fallback.

This closes the system execution and API-observation shape, not the performance
architecture. The exact layer-major tactic still lowers Attention and the
admitted FP8/NVFP4 projections to bounded physical segments. The tree now also
contains an explicit `native-group-q64-panel` Attention tactic and an explicit
`segmented-marlin-operator-panel` projection wrapper. Both are default-off and
accuracy-unqualified. The projection name is intentional: it groups logical
panel work but still lowers P40K to 12,992 physical FP8/NVFP4 Marlin launches;
it is not a native large-M implementation.

A separate default-off
`native-quantized-large-m-operator-panel` v5 tactic is now connected to the
same real generation/API path. It binds the authenticated FP8/NVFP4 Marlin
sidecars plus typed reduction arenas and locks under a distinct arithmetic
contract. A complete M8192 panel uses exactly one physical Marlin launch per
logical projection. Every partial panel retains the complete exact span
ledger, legacy MLP workspace, and per-span Down-to-residual interleave; the
old balanced P40K topology therefore remains `3x8192 + 2x7712`. This is a
bounded architecture candidate, not a claim that arbitrary aligned M or the
complete true-large-M dataflow is solved. The evaluation server still selects
layer-major and this tactic only explicitly, its layer-major Attention and
projection defaults remain exact, and the server-wide legacy route remains
default. The retained and rejected development screens, including their dirty
binary limitation, are frozen in the
[M8192/partial-panel direction record](metadata/qwen36-27b-prefill-exact-marlin-m8192-development-screens-2026-08-09.json).

The newest clean-host P40K real-API screen used exact Attention and the v5
projection tactic. EvalScope reported 670.53071 s TTFT; the server reported
670.486890 s pure Prefill, or 59.658139 prompt tok/s. Pure Prefill occupied
99.994017% of server TTFT and external TTFT exceeded server TTFT by only
3.702638 ms, so the API is unequivocally not the active bottleneck. The route
consumed all 40,000 tokens as `3x8192 + 2x7712`, with 1,008 bulk launches, 672
partial-oracle hits, 13,104 physical projection launches, and zero Prefix,
MTP, cuBLASLt, external, approximate, or forbidden fallbacks. It was about
0.53% slower than the prior same-payload exact-Attention observation, so the
M8192-only composition is rejected and P60K was not run. Evidence is frozen in
the [v5 P40K API record](metadata/qwen36-27b-prefill-p40k-native-large-m-exact-api-2026-08-09.json).

The fastest P40K grouped-Q64 screen remains 179.51119 s EvalScope TTFT and
179.395679 s pure Prefill, but it generated a different first token and remains
accuracy-unqualified. It is an architectural upper-bound clue, not a
production candidate.

Revision `c45b7c5` adds the first executable slice of
`AC-PREFILL-PROMPT-WIDE-v2`: `native-flashinfer-exact-panel` submits one
M8192 or M7712 logical causal-Attention panel through the authenticated
FlashInfer backend with FP32 online-softmax state and no QT2, Q64, or Q128
compatibility dispatch. The P40K witness recorded 80 such Attention hits and
zero generic Attention hits while consuming the unchanged `3x8192 + 2x7712`
topology. EvalScope TTFT fell to 109.02622 s and server pure Prefill to
108.981855 s, 367.033577 tok/s. Relative to the preceding exact v5 route this
is a 6.15x speedup and 83.75% latency reduction; it is also 1.65x faster than
the grouped-Q64 development direction. External TTFT exceeded server TTFT by
only 3.874726 ms, so the API is still not the bottleneck.

The result retains the logical-panel Attention dataflow but does not qualify
its arithmetic. The real-model P513 screen preserved token `9419`/`Hello`
while producing a different full-state hash. The P40K response `The` matches
the preceding exact P40K route, but a matching first token does not qualify the
new reduction order. The route is BUILD_TESTING-admitted, explicitly selected,
and reports an unqualified numerical contract. It cannot become the default or
production route until the complete no-regression gate closes. Exact evidence
is frozen in the
[v6 P40K API record](metadata/qwen36-27b-prefill-p40k-flashinfer-exact-panel-api-2026-08-09.json).

Revision `da2b9f6` then connected the default-off
`native-nvfp4-true-large-m-operator-panel` v1 tactic to that same exact
Attention and real generation/API path. The implementation replaces each
Gate+Up or Down operator panel with one M128N256K64, three-stage kernel launch
surface, reducing matched NVFP4 physical launches from 4,992 to 640. It is an
implemented architecture experiment, not a production path and not evidence
that the required shape-specific large-M dataflow has been solved.

The clean-host P40K screen reached 136.97409 s EvalScope TTFT and
136.929918 s server pure Prefill, or 292.120 prompt tok/s. Against the retained
`c45b7c5` result, pure-Prefill latency increased by 27.948063 s, or 25.64%.
The external/server boundary was only 4.047 ms. In the matched NSys pair,
99.30% of the whole-request interval increase was NVFP4 time: Gate+Up rose
from 35.039284 s to 52.600776 s (1.501x) and Down from 17.051106 s to
27.094866 s (1.589x). The large launch-count reduction did not offset decoded-B
duplication, two full-CTA barriers per K64 step, 203/220 registers per thread,
and one-CTA-per-SM residency.

WP-V2-C1-v1 is therefore rejected. The candidate stays disabled, the retained
route and all defaults remain unchanged, and the full accuracy harness, P60K,
P130K, and NCU were not run. At that stop point its successor was only a
design: separate Gate/Up G2 and Down D2 skeletons derived from the proven
Marlin consumer/feed topology, with a two-CTA-per-SM resource contract and
another real P40K API direction gate before deeper qualification. The
complete measurements, route counters, resource observations, and stop
decision are in the
[v1 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-true-large-m-rejection-2026-08-10.json).

Revision `8889de4` subsequently implements that GateUpG2/DownD2 pair as a
distinct, default-off `native-nvfp4-g2-d2-large-m-operator-panel` route. A
request-plan alias discovered before timing was repaired by publishing fused
G2 output into the exact-size dead `mlp.gate_bf16` span rather than the
intentionally aliased normalized/activated span; the failed pre-repair
launches have no performance
authority. The clean r4 P40K API request then consumed all 40,000 tokens,
returned one token, and attested the complete 320+320 coupled route with no
forbidden fallback.

That valid direction still lost: EvalScope TTFT was 115,085.76 ms and server
pure Prefill was 115,041.751913 ms, or 347.699851 tok/s. Against retained
`c45b7c5`, this is +6,059.897021 ms / +5.560464% latency and -5.267563%
throughput. The bounded T4 profile measured G2 at 40.371860160 s and D2 at
20.532324128 s. Their 60.904184288-s combined role time exceeded the retained
native Marlin main plus SiLU scope by 7.040159744 s, while complete GPU kernel
time increased by 6.110398080 s. Two-CTA resource admission and epilogue
fusion therefore did not overcome the one-raster-CTA ownership boundary or
create persistent B/scale reuse across output tiles.

G2/D2 v1 is rejected, stays disabled, and does not change the retained route
or any default. The matching first token `The` is not a state or capability
gate, so full accuracy qualification was not run. P60K was not run: its
balanced `6x8192 + 2x5424` geometry needs an exact M5424 path that this
M8192/M7712-only candidate lacks. That fail-closed geometry gap is not a P60
timing result. P130K and NCU were also not run. Exact evidence is frozen in the
[G2/D2 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-g2-d2-rejection-2026-08-10.json).

Accordingly, current product status is **implemented evaluation runner,
unqualified production runner**.

## 3. Capability state matrix

| Capability | State at the audited baseline | What exists | What prevents the next state |
| --- | --- | --- | --- |
| Pinned Qwen3.6-27B NVFP4 model identity and loader | Implemented | Exact revision/shard authentication, one resident arena and typed weight binding | Must be tied to the installed binary and DeploymentPlan in release attestation |
| Pure C++ tokenizer and greedy generation | Implemented | Pinned tokenizer, batch-one generation and deterministic test/oracle surfaces | Public capability qualification is incomplete |
| OpenAI-compatible evaluation API | Implemented | `/healthz`, `/v1/models`, completions/chat, non-streaming and committed-token SSE; explicit layer-major mode has bounded Prefill cancellation | Loopback/evaluation-only; no production exposure, security, admission, or multi-tenant contract |
| Production serving API | Designed | Product/API contract is defined in the SDD | No installed release profile or release attestation exists |
| Default context capacity | Implemented at 8,192 | Server default `max_sequence_length=8192`, maximum output 4,096, 2 GiB request-arena limit | Does not admit the locked long-context workloads |
| 40K/60K/130K cold/no-cache service | Target; unqualified P40K development routes exercised | Token-ID ingress fails closed on capacity; v10 consumed all 40K tokens through the real API at 101.87053 s TTFT / 392.804397 pure prompt tok/s. v11 projection-reset, experimental v12 phase-local BF16, sealed v13 packed-projection, sealed v14 packed-NVFP4-v2, and sealed v15 stock-Marlin-parity routes also consumed all 40K but reached only 205.951777, 320.472999, 247.814694, 311.300103, and 392.705493 pure prompt tok/s and were rejected; v12 alone lacks sealed-route authority | The incumbent is still 10.95x below the owner's 4.3K tok/s vLLM starting line and 50.92x above the 2-s P40K target; inherited FlashInfer full state differs. P60K has not been timed; P130K, whole-process capacity, and qualification remain open |
| Prefill/Decode logical separation | Implemented in part | Separate phase APIs/metrics and an explicit state transition exist | Shared runner and synchronization-heavy physical plan prevent independent utilization and overlap |
| Prompt-wide Prefill candidate | v10 whole-request substrate retained but accuracy-unqualified; v11--v15 projection packages rejected | Sealed Engine transactions and role receipts remain; v15 proves the complete stock-Marlin host-dispatch reference can execute through the real P40 API, but it did not improve v10 | Inherited Attention state differs. The next activation must change the mathematical/dataflow decomposition and return to P40 before exact Attention/GDN continuation under the 9.302326-s total budget |
| Stock-vLLM Marlin projection parity | Default-off route implemented; P40 direction rejected | Independent 9,625,928,192-byte real-checkpoint owner, 128 transformation manifests from 192 sources, sealed Engine/runner path, stable request-long lock owner, and strict v15 API witness completed 5,120 parity launches, 64 lifetime receipts, and one lock clear with no fallback | It did not beat v10. Numerical/SASS/full-state gates were intentionally not run; retain it only as finite-precision/schedule reference evidence, not a production or tuning lineage |
| Large-M Prefill specializations | C1, G2/D2, shape-wide v3, projection-reset v1, phase-local BF16 v1, packed projection v1, packed NVFP4 v2, and stock-Marlin parity v1 are rejected default-off experiments | v15 closes the stock host-dispatch skeleton at 392.705493 tok/s; retained default-off code, topology, and correctness gates are implementation/reference evidence, not a performance path | Do not scan the rejected skeletons. Derive a materially different whole-path architecture from mathematical equivalence, finite-precision boundaries, communication/reuse costs, and proven implementations; open P60 only after P40 is competitive and accuracy-admissible |
| Terminal-layer exact liveness deletion | Designed; not implemented | The ledger proves full layer-63 K/V remains live while ordinary next-token generation needs only the final row through Q/Attention/O/MLP | Its exact `2.1479599663%` arithmetic-scope deletion needs an independent tactic, receipts, progress and liveness oracle; it composes with a successor but does not lower the 4.3K tok/s target |
| Decode target | Directionally near target | Short API evidence reports about 104 ms TPOT | At least 10 token/s, long-output stability, and release repetition are not qualified |
| Production accuracy | Target with partial oracles | Exact deterministic outputs are available for selected prompts/routes | No complete public capability baseline and promotion gate has passed |
| AOT DeploymentPlan | Implemented internally for the development route; release artifact still designed | Engine-lifetime sealed plan binds model/state/resources/operator identities and one-shot request receipts | No authenticated installed plan artifact is loaded and attested by the default release |
| Unique `BUILD_TESTING=OFF` release | Not implemented | Installed targets can be built | No single build/route manifest reproduces the strongest evidence without admissions or environment composition |
| Automated release evidence lane | Designed | Local CTest and evidence policies exist | No checked-in remote CI/workflow enforces the Orin release gate |

## 4. Current API and capacity facts

At the current implementation snapshot, `EvaluationServerOptions` still
defaults to:

- bind address `127.0.0.1`;
- one serialized inference worker behind bounded ingress/inference queues;
- `max_sequence_length=8192`;
- `maximum_output_tokens=4096`;
- Prefill chunk size 512; and
- `request_max_arena_bytes=2 GiB`.

The adapter has no authentication or TLS. Legacy mode still observes
disconnect/shutdown only at committed-token boundaries. Explicit layer-major
mode uses a two-slot bounded window over `layer x logical-panel` quanta and
checks cancellation after retirement, final normalization, before logits, and
before the single commit; it drains and resets without publishing partial
position. The first response header is delayed until the first committed token
or an early error. These are honest evaluation-stage properties, not
production API guarantees.

Under the existing request-state planner at the maximum production-route
chunk M512, exact arena demand is:

| Maximum sequence length | Planned request arena bytes |
| ---: | ---: |
| 8,192 | 705,331,200 |
| 40,000 | 2,801,096,704 |
| 60,000 | 4,118,856,704 |
| 130,000 | 8,731,016,704 |

The 2 GiB default therefore fails the 40K target before performance is
considered. Merely increasing the command-line limit is not a production
solution: resident-weight/derived-layout footprint, transient Prefill
workspace, thermal headroom, cancellation, queue policy, and exact API
qualification must be planned together.

The unbound layer-major workspace planner separately reports the following
exact request-arena arithmetic for one explicit physical-tactic profile:
C8192 exact C64-native GDN with in-place convolution, current Release/default
legacy C16 GDN, and separate Gate+Up then SiLU. C64-native remains a
development admission and is not the default production route; at `18363ad`
it is nevertheless authenticated and mandatory for eligible M32--M512
segments in the explicit layer-major plan.

| Prompt tokens | Caller-selected conditional profile | Conservative disjoint profile |
| ---: | ---: | ---: |
| 40,000 | 3,975,374,848 bytes | 5,324,963,840 bytes |
| 60,000 | 5,496,014,848 bytes | 7,052,323,840 bytes |
| 130,000 | 10,818,254,848 bytes | 13,098,083,840 bytes |

The `selected` label in this table is a caller-selected host-planner strategy,
not selection of an architecture candidate or production route. It assumes
one prompt-wide hidden allocation and family-live-set overlay whose alias,
completion-event, and legacy-route-exclusion contracts are still unbound. The
conservative profile uses two prompt-wide hidden buffers and makes the three
C8192 operator families plus the legacy C512 workspace disjoint; it still
depends on the named phase-local layout contract inside each selected tactic.
Changing the tactic changes the exact total: token-parallel C64 convolution
raises the three conservative rows to 5,335,449,600, 7,062,809,600, and
13,108,569,600 bytes; fused Gate+Up epilogue raises the C8192 overlay from
855,638,016 to 940,572,928 bytes. A disjoint test-only native legacy GDN adds
another 75,694,080 bytes, whereas the current Release C16 route adds none.
Every profile also includes an independent 10,240-byte final-hidden handoff.
The initial `RequestState` shape keeps the C8192 family overlay but gives the
legacy C512 workspace disjoint storage; its exact 40K/60K/130K totals are
4,066,344,960, 5,588,904,960, and 10,917,864,960 bytes. A 32,768-byte C8192
token-ID staging view reuses the operator-arena prefix only after an explicit
embedding-consumed event, so it does not add another allocation.

All six request-arena values fit the planner's declared
17,437,720,576-byte limit, but no whole-process fit follows. Resident-model and
derived-sidecar byte requirements are absent, total whole-process bytes are
unknown, and the planner's whole-process capacity verdict is
`kIndeterminate`. The layer-major Engine create path can allocate the selected
request arena, expose typed phase views, and bind the segmented compatibility
executor and completion events. This is a fail-closed development admission,
not a production capacity verdict, and it does not replace the default M512
route or establish that model plus sidecars plus the largest request fit
concurrently.

## 5. Current performance evidence and its authority

### 5.1 Rejected P40K segmented-Marlin projection screen

At `bbd8ac3`, one clean-host EvalScope 1.9.1 request sent the pinned 40,000
token-ID prompt through
`q3x.sm87.ac-prefill-layermajor-8k.segmented-marlin-operator-panel.native-group-q64-attention.v1`.
The v4 witness attested all 40,000 tokens consumed, five logical panels, 1,680
logical segmented-projection hits, 12,992 physical Marlin projection launches,
and 80 grouped-Q64 Attention hits. It reported zero Prefix-cache, MTP,
cuBLASLt, external-reference, or generic approximate-route hits. The plan is
nevertheless explicitly `accuracy-unqualified-architecture-candidate` and
does not qualify numerical equivalence.

| Metric | Result |
| --- | ---: |
| EvalScope TTFT | 179,511.19 ms |
| EvalScope prompt throughput | 222.827281 tok/s |
| Server TTFT | 179,507.414119 ms |
| Server pure Prefill | 179,395.678907 ms |
| Server pure Prefill throughput | 222.970811 tok/s |
| External TTFT minus server TTFT | 3.775881 ms |

The wrapper saved only 618.639057 ms of server pure Prefill against the
preceding 180,014.317964-ms exact-projection/grouped-Q64 screen, a 1.003448x
direction. This is effectively neutral relative to the architectural gap.
More importantly, the 179.51119-s external TTFT crossed the predeclared
`>=165 s` rejection boundary. The segmented wrapper was therefore rejected
immediately; P60K, P130K, NSys, and NCU were not run for it.

This result distinguishes logical grouping from native large-M execution. The
wrapper does not remove the structural physical-launch/data-presentation
boundary and cannot be described as a large-M kernel. It motivated the later
default-off M8192 exact-Marlin route; that newer route still requires the same
clean-host P40K real-API gate before it has architecture authority. Exact
hashes, route counters, timings, and the historical stop decision are frozen in
[`metadata/qwen36-27b-prefill-p40k-segmented-marlin-q64-api-2026-08-09.json`](metadata/qwen36-27b-prefill-p40k-segmented-marlin-q64-api-2026-08-09.json).

### 5.2 P40K native-group-Q64 Attention direction and T4 profile

At `a9a065e`, one clean-host, one-request real-API/EvalScope plus Nsight
capture consumed all 40,000 tokens and generated one token through the
explicit deployment plan
`q3x.sm87.ac-prefill-layermajor-8k.native-group-q64-panel.v1`. The plan is
default-off and identifies itself as an
`accuracy-unqualified-architecture-candidate`; its numerical contract is
explicitly unqualified. EvalScope 1.9.1 is the declared invocation version,
but the retained raw artifacts do not self-attest that package version.

| Profiled metric | Result |
| --- | ---: |
| EvalScope TTFT | 180,864.404898 ms |
| Server TTFT | 180,860.665933 ms |
| Server pure Prefill | 180,345.412463 ms |
| Server pure Prefill throughput | 221.796604 tok/s |
| External TTFT minus server TTFT | 3.738965 ms |
| `q3x.prefill.whole_request` wall | 180,340.668128 ms |
| Whole-request kernel union | 179,988.524640 ms (99.804734%) |
| CUDA streams / kernel overlap | 1 / 0 ms |

The top grouped-Q64 Attention kernel consumes 84,318.548 ms (46.8% of
kernel time). The two dominant still-segmented Marlin signatures consume
51,960.015 and 26,067.5183 ms (43.4% together), while the BF16 M16 pair adds
5,432.6802 ms across 120,000 launches. Attention plus the two main Marlin
signatures occupy 90.021892% of the whole-request wall. The trace therefore
locates the next composed architecture boundary in both exact causal
Attention and panel-capable projection ownership; the loopback API is not the
dominant measured budget. The two-slot host submission window created no
device-kernel overlap in this trace.

A preceding unprofiled direction screen on the same corpus observed
180,014.317964 ms (222.204547 tok/s) for this tactic versus
666,946.668018 ms (59.974810 tok/s) for the exact segmented Attention
context. That 3.704965x performance direction does not qualify the candidate:
the exact context generated `The`, while the candidate and profiled runs
generated `Based`, and no full-state accuracy gate passed. The profiler's
request-total and whole-workload-throughput fields are also polluted by
capture-stop/report-generation delay and are not unprofiled baselines.

This is a T4 diagnostic record, not a completed target-length architecture
witness. P60K and P130K were deliberately not run; the default exact/legacy
routes, production status, and target remain unchanged. Exact route, hash,
NVTX, kernel-union, top-20, and limitation evidence is frozen in
[`metadata/qwen36-27b-prefill-p40k-native-group-q64-panel-nsys-2026-08-09.json`](metadata/qwen36-27b-prefill-p40k-native-group-q64-panel-nsys-2026-08-09.json).

### 5.3 Sealed layer-major P1025 direction screen

Commit `18363ad` was measured through the real OpenAI-compatible API with
EvalScope 1.9.1 on one hash-locked 1,025-token prompt and one greedy output
token after a clean-host Jetson preflight. The exact binary SHA-256 was
`de172fd1dca5d62241f6df5954bbc90efce4d8e3ba0d7cf1ccf8ca28f4aae8c0`.
The prior comparator is an older full-inventory build family using greedy
layer-major physical segmentation and exact fallback for FP8, Attention, and
GDN.

| Metric | Prior greedy/fallback layer-major | `18363ad` sealed composition |
| --- | ---: | ---: |
| EvalScope TTFT | 3,291.95 ms | **2,767.24 ms** |
| Server pure Prefill | 3,285.658588 ms | **2,761.173664 ms** |
| EvalScope prompt throughput | 311.36 tok/s | **370.40 tok/s** |
| Server pure Prefill throughput | 311.96 tok/s | **371.218954 tok/s** |

The cumulative sealed route saves 524.484924 ms of pure Prefill, a 1.189950x
speedup or 15.962855% latency reduction against that older witness. This is
not a single-mechanism comparison: the older witness reports production
dispositions only for NVFP4 Gate/Up and Down, with FP8 QKV/Z/O, Attention, and
GDN on exact fallback. The `18363ad` v2 witness records 64 NVFP4 Gate/Up hits,
64 NVFP4 Down hits, 96 FP8 QKV hits, 48 FP8 Z hits, 64 FP8 O hits, 16 Attention
hits, and 48 native GDN hits on their admitted production dispositions, with
zero exact fallback, forbidden fallback, Prefix cache, MTP, cuBLASLt,
external-reference, or approximate-route hits. The output token is exactly
`在` and finishes by the declared one-token length cap.

This is a single-request short direction screen, not an architecture
selection, target-length result, repetition sample, or production result. It
remains 415.976098 ms (17.737358%) slower in pure Prefill than the older
2,345.197566-ms legacy P1025 observation. The evidence therefore retains only
the complete sealed composition direction; it does not attribute the gain to
balanced segmentation, native GDN, or any other constituent in isolation. It
also confirms that the M512 compatibility internals must give way to the next
operator-panel dataflow boundary.

Source:
[`metadata/qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json`](metadata/qwen36-27b-prefill-layer-major-balanced-p1025-direction-2026-08-09.json).

### 5.4 Latest retained cumulative short external result

The latest retained cumulative comparison associated with main used eight
real requests, prompt lengths 32--1,025, 16 output tokens, concurrency one,
and a cumulative development build with all relevant Prefill admissions plus
runtime composition enabled.

| Metric | Native run 1 | Native run 2 | Frozen vLLM slice |
| --- | ---: | ---: | ---: |
| Workload prompt throughput | 183.341934 tok/s | 183.315553 tok/s | 181.896870 tok/s |
| Mean TTFT | 1,152.676220 ms | 1,153.721993 ms | 1,168.570642 ms |
| Mean TPOT | 104.083710 ms | 104.041991 ms | 104.466390 ms |
| Exact native outputs | 8/8 | 8/8 | not an accuracy oracle |

Source:
[`analysis/decode-gate-up-coupled-feed-vllm-parity-2026-07-30/README.md`](analysis/decode-gate-up-coupled-feed-vllm-parity-2026-07-30/README.md).

This is useful short-workload directional/parity evidence for the exact
admission composition. `Workload prompt throughput` includes the complete
request run and is not pure Prefill tok/s. The small panel, short contexts,
test-only build requirements, runtime selectors, and missing public capability
gate prevent a production or long-context claim.

The reported TPOT corresponds to roughly 9.61 token/s in the best of these
runs, which remains below the locked 10 token/s release target and is not a
long-output qualification.

### 5.5 Rejected G2/D2 P40K architecture direction and T4 attribution

The explicit, default-off
`native-nvfp4-g2-d2-large-m-operator-panel` route was run once through the
clean-host, cold/no-cache OpenAI API/EvalScope P40K witness after its request
buffer alias was repaired. The v8 witness consumed all 40,000 tokens as
`3x8192 + 2x7712`, recorded 320 GateUpG2 hits, 320 DownD2 hits, 80 exact
FlashInfer Attention hits, and zero Prefix, MTP, cuBLASLt, external,
approximate, exact-fallback, or forbidden-route hits.

| Metric | Retained `c45b7c5` | G2/D2 v1 | Change |
| --- | ---: | ---: | ---: |
| Server pure Prefill | 108,981.854892 ms | 115,041.751913 ms | +6,059.897021 ms / +5.560464% |
| Server pure Prefill throughput | 367.033577 tok/s | 347.699851 tok/s | -5.267563% |
| EvalScope TTFT | 109,026.22 ms | 115,085.76 ms | direction only |

The G2/D2 run returned `The`, the same first token as the retained route, but
one token cannot qualify the full state or model capability. The candidate
therefore remains accuracy-unqualified; no full accuracy or repetition work
was unlocked.

The bounded post-rejection NSys capture attributes the loss directly to the
new projection package. G2 consumed 40.371860160 s and D2 20.532324128 s,
60.904184288 s together. The retained native Marlin main signature consumed
51.866262272 s and standalone SiLU 1.997762272 s, 53.864024544 s together.
The resulting role gap is +7.040159744 s; other kernels recovered 0.929761664
s, leaving complete GPU kernel time +6.110398080 s. This profile is diagnostic
only and cannot reverse the negative T3 direction.

P60K was not run. Its balanced geometry is `6x8192 + 2x5424`, whereas this
candidate admits only M8192 and M7712; the missing M5424 route is a fail-closed
geometry blocker, not a P60 performance conclusion. P130K and NCU were also
not run. G2/D2 v1 stays default-off and closed. Exact hashes, route counts,
timings, profile scopes, and limitations are frozen in
[`metadata/qwen36-27b-prefill-p40k-nvfp4-g2-d2-rejection-2026-08-10.json`](metadata/qwen36-27b-prefill-p40k-nvfp4-g2-d2-rejection-2026-08-10.json).

### 5.6 Positive P40K whole-core direction and dominant-kernel attribution

The exact-P40000 whole-core route returned once through the clean-host OpenAI
API/EvalScope gate. It consumed all 40,000 tokens as `5x8000`, recorded a
complete v10 route with exactly 768 bounded retirements, and reached
101,831.853876 ms pure Prefill / 392.804397 prompt tok/s. This is 7.02138%
faster than retained `c45b7c5`, while EvalScope TTFT remained only 3.730501 ms
beyond server TTFT.

The following bounded NSys capture found one stream and no kernel overlap, but
only 7.992928 ms of non-kernel space in the 102,121.306528-ms whole-request
range. The existing serial schedule is therefore a property of the current
dependency/dataflow implementation, not an API or launch-gap explanation.

| Dominant role | Calls | Total | Request share |
| --- | ---: | ---: | ---: |
| NVFP4 Gate/Up persistent Marlin | 64 | 37,273.068224 ms | 36.50% |
| FP8 Marlin | 1,040 | 25,864.646560 ms | 25.33% |
| NVFP4 Down persistent Marlin | 64 | 17,559.457280 ms | 17.19% |
| Whole-prompt FlashInfer Attention | 16 | 13,634.170272 ms | 13.35% |

The whole-core infrastructure is retained because it moves the product metric
and closes transaction, memory, route, and evidence seams. The NVFP4 body is
not retained as the final architecture: it launches a fixed 16 CTAs and wraps
the old M64N256K64 Marlin raster, so it does not create the required decoded-B
or scale reuse. The first distinct shape-wide replacement also regressed, so
neither NVFP4 skeleton is an active implementation target. Full hashes and
limitations for this retained substrate are frozen in the
[whole-core direction record](metadata/qwen36-27b-prefill-p40k-whole-core-direction-2026-08-10.json).

### 5.6.1 Rejected shape-wide NVFP4 v3 direction

The v3 package used separate real-shape rasters, a three-stage `cp.async`
feed, 126 registers/thread, 62,976 bytes of dynamic shared memory, and measured
two-CTA/SM admission for both Gate/Up and Down. It was selected only by a
binary-pinned, default-off overlay inside the exact-P40 whole-core route. The
clean API run consumed all 40,000 tokens with no forbidden route, but pure
Prefill increased from 101,831.853876 ms to 106,374.300578 ms.

The permitted causal profile measured Gate/Up at 41,163.382144 ms versus
37,273.068224 ms for the whole-core baseline and Down at 17,788.101568 ms
versus 17,559.457280 ms. Their combined 7.511889% regression explains the
whole-product loss; the 3.399552-ms kernel gap is immaterial. The overlay was
removed, no NCU or accuracy promotion was run, and stage/tile/raster scans on
this skeleton are closed. Exact evidence is in the
[v3 rejection record](metadata/qwen36-27b-prefill-p40k-nvfp4-shape-wide-v3-rejection-2026-08-10.json).

### 5.6.2 Rejected phase-local BF16 projection direction

The experimental phase-local package used exact real-checkpoint NVFP4/FP8
expansion into short-lived BF16 scratch followed by a serialized native
M128N128K64 dense consumer. Its independent v12 deployment identity reached
the real OpenAI API and consumed all 40,000 tokens. Live output reported one
matching smoke token and 416 FP8 physical launches; 512 NVFP4 MLP launches were
inferred from the successful-enqueue composition. No raw witness was retained,
and the immutable topology disagreed with the physical lowering, so these
counters do not provide sealed-route or promotion authority. The route
remained default-off and accuracy-unqualified.

Pure Prefill increased to 124,815.475334 ms / 320.472999 prompt tok/s versus
101,831.853876 ms / 392.804397 tok/s for v10. EvalScope TTFT was
124,853.851547 ms, so the API boundary added only 38.376213 ms. The bounded
profile found 98,999.424800 ms in all BF16 dense projections but only
363.332320 ms in every canonical expansion kernel. Gate/Up dense plus the
separate SiLU was 49,661.799968 ms, FP8 dense was 29,642.775168 ms, and
Attention remained effectively unchanged. The architecture is rejected
because it abandons compressed-B movement and register-side decode; the
expansion kernel is not the problem. It unlocks no NCU/parameter scan,
accuracy expansion, P60, P130, or production promotion. Exact evidence is in
the
[phase-local rejection record](metadata/qwen36-27b-prefill-p40k-phase-local-bf16-rejection-2026-08-10.json).

### 5.6.3 Rejected AOT packed-operand projection v1 direction

The independent v13 route replaced all NVFP4 Gate/Up/Down and FP8 QKV/Z/O
projection operands with role-specific load-time packed artifacts. Startup
authenticated and prepared 256 physical artifacts from 400 original
real-checkpoint sources into 16,840,130,560 resident bytes; no request-time
repack, BF16 weight materialization, runtime JIT, FP8 supermatrix fallback,
MTP, or cuBLASLt production route was present. Synthetic CUDA arithmetic,
canary, invalid-input, and resource gates passed, but they do not qualify
real-model state or capability.

The clean-host cold/no-cache EvalScope 1.9.1 request consumed all 40,000 token
IDs as `5x8000`, committed one `Based` token, and retained the v10 whole-core
control schedule. Its v13 witness recorded exactly 208 FP8 logical roles in
128 physical launches, 64 Gate+Up and 64 Down roles in 128 NVFP4 launches,
48 BF16 A/B, 48 GDN, 16 whole-prompt FlashInfer hits, 768 retirements, and no
forbidden route. Historical v12 was not reused: the tactic, deployment plan,
package identity, topology, and retained raw witness are independent.

| Metric | v10 incumbent | Packed projection v1 | Change |
| --- | ---: | ---: | ---: |
| Server pure Prefill | 101,831.853876 ms | 161,410.929373 ms | +59,579.075497 ms / +58.5073% |
| Server pure Prefill throughput | 392.804397 tok/s | 247.814694 tok/s | -36.9114% |
| EvalScope TTFT | 101,870.53 ms | 161,447.32 ms | direction only |
| EvalScope New Prompt | — | 247.758768 tok/s | direction only |

EvalScope TTFT exceeded server TTFT by only 3.807390 ms and pure Prefill was
99.9798% of server TTFT, so API, admission, and first-response publication do
not explain the regression. The API/asset/plan/route integration is complete
for this exact P40 development screen, but the packed v1 performance dataflow
is rejected. It remains default-off and accuracy-unqualified. Full accuracy,
repetition, NCU, P60, and P130 were not run. At most one bounded same-payload
profile may answer a named causal question for a materially different
successor; it cannot reverse this rejection. Exact hashes and limitations are
frozen in the
[packed projection rejection record](metadata/qwen36-27b-prefill-p40k-packed-projection-rejection-2026-08-10.json).

### 5.7 Current cumulative internal Prefix attribution

One real-model P513 NSys capture reports:

| Scope | Wall time |
| --- | ---: |
| M512 Prefix | 1,217.934464 ms |
| M1 tail | 108.699616 ms |
| Prefix total | 1,326.634080 ms |
| Finish Prefill | 5.371744 ms |
| Prefix plus finish | 1,332.005824 ms |

Source:
[`analysis/prefill-p513-current-cumulative-nsys-2026-07-30/README.md`](analysis/prefill-p513-current-cumulative-nsys-2026-07-30/README.md).

This is one diagnostic capture, not a retention sample and not the target
workload. It attributes the admitted P513 path; it cannot establish 40K--130K
API fitness.

### 5.8 Historical external baseline

The earlier 32-request EvalScope 1.9.1 run used 20--1,160-token prompts and 16
output tokens. Native mean TTFT was 3,168.79 ms versus 1,144.51 ms for matched
stock vLLM; total workload prompt throughput was 102.8141 versus 182.1476
tok/s. It set Prefill architecture priority, but its native binary provenance
was incomplete and it was a single-process directional protocol. It remains
historical evidence, not current release qualification.

### 5.9 The 1,224.7335 tok/s number

The 1,224.7335 tok/s result belongs to an opt-in Factorized-R1 experimental
branch and a one-warmup/one-measurement P1853 `/v1/completions` direction
screen with `max_tokens=1`. It changes the numerical trajectory: its recorded
outputs do not match the native baseline, and its metadata explicitly marks
it `quality_production_eligible=false`,
`production_residency_eligible=false`, and
`performance_upper_bound_only=true`.

It is therefore a research upper-bound observation. It is not in main, not a
default path, not lossless, and not a production Prefill result.

## 6. Accuracy state

Production accuracy is a hard constraint; no lossy Attention, GDN state,
activation-quantization, or other changed numerical contract is eligible
without an explicit owner amendment.

Current evidence includes deterministic token/output and layer/component
oracles, but it does not close the product gate:

- the latest short cumulative route reproduced its native comparator on 8/8
  outputs;
- the first external native/vLLM comparison matched text on 26/32 requests,
  but neither runtime is the accuracy oracle; and
- the first public C-Eval attempt produced no parseable answer before its
  output cap, so the reported zero score is an invalid protocol result rather
  than a capability measurement.

A parseable public capability baseline, exact request/output contract,
deterministic production oracles, and post-integration repeat are still
required.

## 7. Current execution-architecture gap

Prefill and Decode are logically identifiable. The default legacy route is
still primarily serial: it processes bounded prompt tiles through the shared
runner and synchronizes before each tile commit. Gate/Up uses a limited
layer-local auxiliary-stream fork/join, but the default route has no general
double/triple-buffered cross-tile or cross-layer pipeline.

At chunk 512, 40K tokens require about 79 tiles. Repeating a 64-layer weight
and synchronization traversal for every small tile is a first-class
architecture seam. The required response is a whole prompt-span execution
plan with explicit state semantics, residency, buffer ownership and overlap,
not a return to unrelated kernel parameter scans.

The explicit development route now implements the whole-request response:

- immutable 64-layer/logical-C8192 topology and request-owned progress;
- typed prompt-wide `RequestState` storage with fixed final-hidden handoff;
- a sealed 17-role Engine plan bound to exact model, arena, views, streams,
  tactics, sidecars, workspaces, and two completion events;
- true `layer -> logical panel -> physical segment` traversal with balanced
  final logical and physical panel pairs rather than a one-token tail;
- fail-closed native exact C64 GDN binding for every eligible M32--M512
  linear-attention segment, with exact fallback restricted to M1--M31;
- an uncommitted retained-hidden finalizer and one final sequence-length
  publication guarded by a move-only receipt and RAII rollback; and
- a two-slot layer-panel submission window with bounded API cancellation and
  v2 success evidence.

The v1 exact integration remains a compatibility executor whose physical work
is segmented and test-admitted. The explicit grouped-Q64 experiment changes
the Attention arithmetic and the connected segmented-Marlin wrapper does not
remove projection segmentation: its P40K v4 witness records 12,992 physical
projection launches. The v5 projection tactic removes software-visible
segmentation only for complete M8192 panels; the underlying Marlin body still
decomposes large M internally and every partial panel retains the exact span
ledger. Its 670.53071-s exact-Attention API result closed that composition.

The first v2 slice now replaces repeated exact QT2 Attention with one
FlashInfer logical-panel graph per Attention layer and panel. On the same P40K
API workload, 80 v6 hits and zero QT2/Q64/Q128 hits reduced pure Prefill to
108.981855 s. This material product movement retains the prompt-wide
Attention dataflow. It does not relax the accuracy boundary: P513 state
diverges and the P40K single-token match is insufficient, so the candidate
remains unqualified and default-off.

The first coupled true-large-M NVFP4 implementation at `da2b9f6` did return to
the same P40K product witness. Its M128N256K64 v1 reduced projection launches,
but pure Prefill regressed from 108.981855 s to 136.929918 s. A matched NSys
pair attributes 99.30% of the interval increase to NVFP4; Gate+Up and Down are
1.501x and 1.589x slower than the retained Marlin route. The v1 skeleton is
therefore closed and must not be parameter-scanned further. The API boundary
remained only 4.047 ms, so API work cannot explain or repair this regression.

Gate/Up G2 and Down D2 are implemented and returned together through
the real P40K API path. They meet the static two-CTA/SM resource envelope,
fuse the SiLU/residual epilogues, and attest all 320+320 logical role hits, but
pure Prefill regressed from 108.981855 s to 115.041752 s. Bounded NSys shows
G2+D2 7.040160 s above the retained native Marlin main plus SiLU scope and
complete GPU kernel time 6.110398 s higher. The one-raster-CTA skeleton is
therefore closed; more parameter scans on this version are not the next P0.

The following exact-P40000 whole-core route then collapsed host ownership to
five M8000 fill/drain panels and one prompt-wide core per layer. It improved
the product result to 101.831854 s / 392.804397 tok/s. Its profile proves that
the API and inter-kernel gaps are negligible, while the four dominant roles
consume more than 92% of the request. It also proves that the current
"persistent" NVFP4 body is still a 16-CTA old-Marlin raster without useful
cross-CTA B/scale residency. The whole-core infrastructure is retained; that
MLP body remains insufficient. The first distinct Gate/Up and Down shape-wide
replacement subsequently regressed to 106.374301 s / 376.030675 tok/s; its
profile makes Gate/Up the principal cause. Both NVFP4 skeletons are therefore
closed against local parameter scanning.

The next three complete projection resets also failed the same product gate.
The v11 16/32-CTA grouped reset reduced software-visible launches but regressed
to 194.220222 s / 205.951777 tok/s. The v12 phase-local path proved exact
request-time expansion is inexpensive at 0.363332 s, but its expanded-B dense
consumers consumed 98.999425 s and regressed the request to 124.815475 s /
320.472999 tok/s. Therefore neither tiny persistent ownership nor BF16 B
materialization is an active tuning surface. Packed v1 then implemented the
complete AOT 256-artifact/400-source route, preserved packed operands through
the load-time asset and request interfaces, and proved the sealed API/count
contract. It still regressed to 161.410929 s / 247.814694 tok/s. Thus packed
representation alone does not validate this kernel ownership/pipeline; v1 is
closed against parameter scans.

Packed NVFP4 v2 then removed packed-v1 FP8 from the experiment, restored the
v10 FP8 path, and changed only Gate+Up and Down ownership. Its authenticated
128-artifact/192-source route was complete and improved pure-Prefill latency
by 20.393636% relative to v13, but still reached only 128.493372 s /
311.300103 tok/s. The valid v14 product result closes that M128N256/K128,
one-CTA/SM skeleton as well. Neither packed representation nor its current
shape-specific mapping is an active tuning surface.

The first system selection point remains the same clean real P40K API against
the 392.804397 tok/s whole-core direction and ultimately the owner's 4.3K
tok/s vLLM starting line. Only a competitive, accuracy-admissible P40K
composition unlocks M5424 implementation, P60, and then P130. No local packed
v1 or v2 scan is active. One bounded same-payload profile may answer a
predeclared successor-design question; the Roadmap must then activate a
materially different named ownership/dataflow hypothesis, derived first from
the exact mathematical function, finite-precision boundaries, and
communication/reuse lower bound, and return it directly to P40. The 4.3K tok/s
starting line permits only 9.302326 s at P40, while v10's non-projection
kernels already consume about 21.416142 s. Exact Attention and GDN therefore
remain mandatory system slices after a positive projection architecture, not
optional follow-ups; a projection win alone cannot complete the goal.

Unpinned or dirty experimental branches are intentionally excluded from this
status snapshot. A candidate affects current truth only after its exact commit,
route, numerical mode, and evidence authority are recorded; branch proximity
or a chat description is not implementation status.

## 8. Open gaps and roadmap ownership

The rows below are status facts, not an independent priority list. The sole
active dependency order and exit criteria are in
[`ROADMAP.md`](ROADMAP.md).

| Gap | Audited state | Controlling roadmap slice |
| --- | --- | --- |
| Product API and long-context admission | Configured token-ID validation and host requirement plans exist; 40K/60K/130K still do not fit or execute through the default contract | P1 |
| Exact deliverable identity | No unique `BUILD_TESTING=OFF` release or authenticated DeploymentPlan | P2 |
| Target-length performance and physical Prefill plan | The exact-P40000 whole-core direction reaches 392.804397 pure prompt tok/s, +7.02138% versus retained v6, with negligible API and kernel-gap overhead. Shape-wide NVFP4, grouped projection-reset, phase-local BF16, AOT packed projection v1, and packed NVFP4 v2 all regressed and are closed. v14 proves the complete NVFP4-only API/asset/route contract and improves on v13, but reaches only 311.300103 tok/s, -20.749333% versus v10. The retained incumbent remains 10.95x below the owner's vLLM floor and accuracy-unqualified. P60/P130 were not timed | P3 |
| Accuracy, capability, stability, and release evidence | Partial deterministic oracles; no complete qualification bundle | P4 |
| Packaging and operations | No attested install/startup/upgrade lane | P5 |

Subsystem and mechanism documents may explain these gaps but cannot reorder
them. When one changes, update this snapshot and let the Roadmap own the
resulting delivery sequence.

## 9. Measurement preflight

No performance run is valid while another unexpected host process owns
material CPU/GPU resources. Jetson resource preflight uses `tegrastats` and
process/device-handle inspection, never `nvidia-smi` as the idle authority.

## 10. Claim boundary

Until the gaps above close, use the following language:

- **Current:** real-model native evaluation runner whose strongest whole-product
  P40K development direction is the default-off exact whole-core candidate at
  392.804397 pure prompt tok/s, +7.02138% versus retained v6. Its complete v10
  witness has no forbidden route, but the inherited FlashInfer arithmetic has
  a known P513 state mismatch and the measured binary came from a dirty working
  tree, so it remains accuracy-unqualified and unpromoted. The whole-core
  control substrate is retained. Its old-Marlin body is insufficient, while
  the first shape-wide replacement regressed and has been removed from the
  runner. P60/P130 were not run. Exact/default routes and production status
  are unchanged. The rejected v13 route proves complete AOT packed assets and
  sealed API/route integration, but reached only 247.814694 tok/s, 36.9114%
  below v10, and remains default-off implementation/correctness evidence. The
  historical v12 route is also rejected and cannot be reused. The rejected
  v14 route restores v10 FP8 and proves a complete NVFP4-only sealed path, but
  reaches only 311.300103 tok/s, 20.749333% below v10. None is the active
  performance architecture; no packed-v1 or packed-v2 parameter scan is
  active.
- **Not current:** production server, 40K--130K support, release-grade vLLM
  parity, 1,224.7335 tok/s lossless Prefill, or a fully qualified 10 token/s
  Decode release.
- **Target:** accuracy-preserving, non-MTP OpenAI-compatible runner reaching
  40K--60K first response within 2 s, about 130K within 4 s, and Decode at
  least 10 token/s, first matching and then exceeding useful vLLM behavior.
