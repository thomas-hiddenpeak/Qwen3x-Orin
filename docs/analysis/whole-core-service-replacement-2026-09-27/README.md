---
q3x_document:
  id: q3x-whole-core-service-replacement-audit-20260927
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: source and history audit of whole-core service replacement eligibility
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: successor variable-length composition and actual API comparison
---

# Whole-core service replacement assessment

Source: `c1f32d0`. Owner direction: investigate why the faster Prefill remains
separate, and consider replacing the current path throughout the served context
range if the complete alternative is superior. This audit changes no runtime
route and performs no new real-model timing. A host-only contract probe and
source/history inspection distinguish unsupported shapes from performance
losses. Raw probe source/output are in
`.q3x-work/whole-core-replacement-audit-20260927/`.
Source pins, historical commit messages, probe output and artifact hashes are
retained in the [audit metadata](../../metadata/qwen36-27b-whole-core-service-replacement-audit-2026-09-27.json).

## Finding

The current whole-core implementation is a qualified P40000/O16 specialization,
not a measured full-range alternative. Its restrictions originated in bounded
architecture construction and qualification, then became hard-coded shape,
layout and deployment contracts. There is no evidence here that its architecture
must lose at shorter contexts, or that O16 is a hardware output limit. Conversely,
P40 success does not establish a win at every served length.

The correct next selection unit is one runnable service combining variable-length
whole-core Prefill, the exact Decode implementation and its complete acceleration
inventory. Reaching the standalone Legacy Decode speed target need not precede
that composition. Retire Legacy as the default if the composed service preserves
the existing API/capacity/accuracy contract and wins across the representative
served workload panel; retain a short-context branch only if matched evidence
requires it, not as a permanent assumption.

## Why the isolation exists

1. Commit `82d6665` initially described whole-core as Prefill-only after the
   first post-Prefill Attention failed. That diagnosis was superseded by
   `3c76516`: probability scratch had 24 x 40000 entries while the first Decode
   required 24 x 40001. The shared KV/Decode path did exist. Sizing scratch from
   request capacity fixed the failure, and O16 completed. Do not revive the
   earlier claim that the architecture cannot decode.
2. `8fe4e67` explicitly promoted a second profile, preserving the default.
   The accepted tradeoff was exact P40000/O16, an 8,641,684,992-byte request
   arena, and a distinct sidecar inventory. The
   [accepted decision](../../decisions/0002-prefill-attention-vllm-numerical-alignment.md)
   records this bounded promotion; it does not record an all-context comparison.
3. The newer fused Decode admission was isolated for numerical investigation.
   [CMake](../../../CMakeLists.txt) requires BUILD_TESTING and rejects composition
   with other Q3X_BUILD options. This is a build-isolation rule, not proof of
   mathematical incompatibility with whole-core.
4. The two Decode acceleration layouts for Gate/Up coupled feed and Down consumer
   order are selected only for the Legacy Prefill configuration in
   [engine creation](../../../src/runtime/reference_engine.cpp). The whole-core
   profile reports zero layers for both. Connecting only the new Attention kernel
   does not automatically bring these optimizations along. Their combined retained
   payload is 8,779,202,560 bytes; an integrated memory plan must account for that
   payload and avoid retaining an unnecessary Legacy Prefill supermatrix.

## Actual restrictions below the HTTP gate

| Boundary | Current restriction | Work needed before a full-range API comparison |
| --- | --- | --- |
| [API/deployment plan](../../../include/q3x/server/evaluation_server.h) | Exactly 40000 token IDs, O16, streaming completions with usage; rejects chat, text prompts and other lengths | Preserve the current service's prompt surfaces and output/capacity semantics in the new profile |
| [Workspace planner](../../../src/runtime/prefill_workspace_plan.cpp) | P40000, capacity 40016, five panels of 8000, fixed family layout | Derive bounded layouts from actual prompt and reserved sequence capacity; preserve aliases and live state |
| [Runner](../../../src/runtime/reference_runner.cpp) | Exact panel lengths, offsets, row capacities and typed views | Support partial panels and valid rows without fake prompt tokens or extra state commits |
| [Persistent MLP contract](../../../include/q3x/kernels/sm87_nvfp4_prefill_persistent.h) | Only M40000 is admitted; other aligned lengths also reject, and tails name missing companions | Implement and validate the required length/tail coverage; admission changes alone do not make kernels safe |
| [Engine and plan authority](../../../src/runtime/reference_engine.cpp) | Exact capacity and whole-prompt Attention binding | Extend creation, execution, handoff and route witnesses together |
| Decode and resources | Separate shape admission, sidecar policies and profile receipts | Compose the full Decode inventory, scratch, Graph boundary and memory reserve, then verify actual hits |

The host probe confirms rejection at P1/64/65/512/513/1089/7999/8000/8001/
8192/16000/32000/39999/40001/44095, with only P40000 admitted. It also confirms
that changing the reserved capacity to 40256 or 44096 is rejected while 40016
passes. These are contract checks, not model execution or timing. A short HTTP
request returning 400 cannot supply a whole-core performance comparison.

## Define the replacement envelope correctly

The current default advertises a 40K target prompt, but its actual token-ID
admission uses `P + O - 1 <= 44095`, with nonempty P and `1 <= O <= 4096`.
Thus P40000/O4096 and P44095/O1 are both boundary cases. Do not silently narrow
replacement eligibility to exactly 40K or only O256. Whole-core's existing
planner uses its own P+O capacity convention, which must be reconciled explicitly
instead of blindly copying the Legacy off-by-one convention.

The service also accepts raw-text completions and formatted chat, supports
streaming/non-streaming responses, request reuse, error handling and cancellation.
A faster token-ID-only P40 endpoint cannot fully replace that service.
60K/130K remain larger product targets; they are not currently served by this
default capacity and are not silently claimed by a 44,095-position replacement.

## Paired performance facts currently available

| Artifact / date | P / O | Engine Prefill seconds / token/s | TTFT | Decode token/s |
| --- | --- | --- | --- | ---: |
| Whole-core production-gate witness, 2026-09-26 | 40000 / 16 | 90.147 / 443.720 | 90.180 s, server first-token commit | 3.869, engine interval |
| Exact v7 Legacy API, 2026-09-27 | 1089 / 32 | 4.217 / 258.252 | 4.218 s, external content | 9.613, external |
| Exact v7 Legacy API, 2026-09-27 | 8192 / 32 | 33.716 / 242.973 | 33.722 s, external content | 9.252, external |
| Exact v7 Legacy API, 2026-09-27 | 40000 / 256 | 219.489 / 182.242 | 219.511 s, external content | 7.878, external |

Whole-core numbers above are recomputed from the retained raw
`.q3x-work/evidence/whole-core-prod-gate-20260926/witness.json`, rather than the
rounded commit narrative. Its engine Prefill includes finalization. This is a
different run from the later comprehensive EvalScope result (91.4 s TTFT,
419.75 reported prompt throughput); neither is relabelled as the other metric.
The [v7 record](../decode-exact-performance-2026-09-27/README.md) owns its paired
data. These rows differ in dates, artifacts, output lengths and timing surfaces;
they demonstrate the composition gap, not a matched replacement decision.
No combined 90-second Prefill plus 7.88-token/s Decode result exists yet.

## Recommended selection sequence

1. Generalize the existing whole-core execution and memory plan, reusing its
   proven P40 path. Preserve arbitrary valid prompt tails and reserve the full
   output capacity. Do not pad short requests into artificial P40000 workloads.
2. Bind the exact Decode route and full sidecar inventory independently of the
   Prefill selection. Check peak startup and execution memory with the existing
   default reserve; do not silently inherit a smaller reserve to make it fit.
3. Use host enumeration for plan/admission bounds, numerical checks for tails and
   state handoff, and a bounded real API panel for performance. Initial useful
   buckets are short chat, P1089, P8192 and P40000/O256. Then cover 64/512/8000/
   8192 boundaries, intermediate lengths, maximum input and long-output boundaries
   including P40000/O4096. Every timed row reports Prefill, TTFT and Decode from
   the same artifact/request; O1 has no subsequent-Decode rate.
4. Compare both arms with the same Decode inventory to isolate the Prefill effect,
   then select on complete API latency, capability, correctness, capacity and
   resources. Test repeated reuse, cancellation/recovery and both API surfaces.
   Representative performance coverage and shape/state proofs have different
   roles; testing every integer length on the real model is not required.
5. If the complete route is consistently better without lost functionality,
   replace the default and keep the old route only as a reference/rollback
   artifact. If some short buckets lose, measure a crossover and justify any
   remaining branch explicitly. Neither universal superiority nor an unavoidable
   short-context penalty is established by this audit.

Keep the numerical decisions separate: whole-core Prefill uses its accepted
and bounded numerical lineage; exact Decode must match the scalar Decode
comparator on the *same whole-core-produced inputs and state*. Existing exact
Decode checks after Legacy Prefill do not alone prove that composed path, and
greedy text divergence between two accepted Prefill backends is not by itself
a Decode failure. Generalizing shapes still requires the corresponding
Prefill/state/capability checks; it grants no new numerical waiver.

The short-context uncertainty is concrete: the P40 gain comes from more work
per weight traversal and larger projection matrices; small prompts offer less
of that reuse, while conversion and scheduling overhead may remain. This is a
hypothesis for matched measurement, not a reason to reject broad replacement.
