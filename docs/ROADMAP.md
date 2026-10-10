---
q3x_document:
  id: q3x-active-roadmap
  class: active
  status: active
  owner: project-maintainers
  authority: current delivery dependency order and exit criteria
  effective: 2026-08-10
  last_reviewed: 2026-10-10
  supersedes: [docs/ROADMAP_LEGACY.md]
  superseded_by: []
  ssot_for: active unfinished delivery slices and their ordering
  review_trigger: delivery dependency, active phase, or milestone-exit change
---

# Qwen3x-Orin active delivery roadmap

This page owns unfinished work and dependency order. [Current Status](CURRENT_STATUS.md)
owns delivered behavior and metrics; [SDD](SDD.md) owns architecture. The
[Constitution](ENGINEERING_CONSTITUTION.md) retains all owner-set targets.

## 2026-10-10 MTP QK bulk publication — closed without retention

`WP-MTP-QK-BULK-PUBLICATION-20261010` / `AC-MTP-GREEDY-v52` owns one
[immutable K-tile composition](metadata/qwen36-27b-mtp-qk-bulk-publication-design-2026-10-10.json).
Publish 32 K positions once and consume sixteen original two-position steps
before retiring the tile. This replaces the fine-grained copy ring, preserving
all scalar arithmetic, causal domains and query/output ownership. The historical
2.215-second QK family cannot alone close the remaining 2.27-second Decode gap;
this is a bounded composition prerequisite. Full P513/P8192 admission returns
immediately to d2 API; >3% slowdown stops, useful direction unlocks d3, otherwise
remove. No tile/pipeline scan or profile follows. Useful retention expires at
the next complete verifier API return or archival; production stays unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-qk-bulk-publication-rejection-2026-10-10.json) preserves all eighteen complete P513/P8192
prefix state/logit comparisons and P513 transactions/recovery. D2 8K observes
Prefill 17.297 seconds / 473.62 token/s, TTFT 17.319 seconds,
Decode 14.112 token/s and total 35.388 seconds.
The declared useful-direction screen is not met. All new QK publication and
profile paths are removed and v52 rebuilt. The driver stops in its post-8K
idle interval; 40K, d3 and remaining API lifecycle checks are not run. No bulk tile,
copy-stage or register scan follows; retained v52 observations remain current.
These single-process observations do not qualify a speedup or hardware cause.
The complete 1.5x–3x goal remains unmet.

## 2026-10-10 MTP NVFP4 native prefetch — closed without retention

`WP-MTP-NV-NATIVE-PREFETCH-20261010` / `AC-MTP-GREEDY-v51` owns one
[native operand-prefetch composition](metadata/qwen36-27b-mtp-nv-native-prefetch-design-2026-10-10.json)
against the remaining 2.27-second 40K Decode gap. Gate/Up and Down issue
cache hints for one valid next packed weight/scale and unstaged activation
record. Demand reads and arithmetic remain original. Native early hints and
no spills precede complete projection/P65 admission and immediate d2 API.
>3% slowdown stops; useful direction unlocks d3, otherwise remove. No profile,
distance/cache-level or register scan follows. Useful retention expires at the
next complete verifier API return or archival; production and the goal stay fixed.

The [completed rejection](metadata/qwen36-27b-mtp-nv-native-prefetch-rejection-2026-10-10.json) passes native instruction/no-spill gates, 45 complete projection cases,
all P65 prefix state/logit/cache comparisons and transactions/recovery. D2 8K observes
Prefill 17.299 seconds / 473.56 token/s, TTFT 17.321 seconds,
Decode 14.061 token/s and total 35.457 seconds.
The declared useful-direction screen is not met. All new NV prefetch and
profile paths are removed and v52 rebuilt. The driver stops in its post-8K
idle interval; 40K, d3 and remaining API lifecycle checks are not run. No prefetch-distance,
cache-level or register scan follows; retained v52 observations remain current.
These single-process observations do not qualify a speedup or hardware cause.
The complete 1.5x–3x goal remains unmet.

## 2026-10-10 MTP Attention rendezvous — closed without retention

`WP-MTP-ATTENTION-RENDEZVOUS-20261010` / `AC-MTP-GREEDY-v50` owns one
[QK/PV synchronization composition](metadata/qwen36-27b-mtp-attention-rendezvous-design-2026-10-10.json)
against the remaining 2.27-second 40K Decode gap. One retained post-wait CTA
barrier retires current readers and publishes next operands; the redundant
pre-wait barrier is removed. Ring ownership and scalar arithmetic are unchanged.
Full P513/P8192 state/logit admission returns immediately to d2 API; >3%
slowdown stops, neutral removes, useful direction unlocks d3. No profile or
parameter scan follows. Useful retention expires at the next full verifier API
return or archival. Production and the complete 1.5x–3x goal remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-attention-rendezvous-rejection-2026-10-10.json) preserves all eighteen complete P513/P8192
prefix state/logit comparisons and P513 transactions/recovery. D2 8K observes
Prefill 17.319 seconds / 473.01 token/s, TTFT 17.341 seconds,
Decode 14.139 token/s and total 35.376 seconds.
The declared useful-direction screen is not met. All new synchronization and
profile paths are removed and v52 rebuilt. The following 40K request is
interrupted before completion; d3 and remaining API lifecycle checks are not
run. No further barrier,
ring or tile scan follows; retained v52 observations remain current.
These single-process observations do not qualify a speedup or hardware cause.
The complete 1.5x–3x goal remains unmet.

## 2026-10-10 MTP native FP8 cache prefetch — API return complete

`WP-MTP-NATIVE-PREFETCH-20261010` / `AC-MTP-GREEDY-v49` owns one
[native cache-prefetch composition](metadata/qwen36-27b-mtp-native-prefetch-design-2026-10-10.json)
against the remaining 2.67-second 40K Decode gap. An explicit cache hint for
the next weight/activation records avoids v51's folded operand-register
lifetime. Original demand loads, arithmetic and state remain authoritative.
The exact compiled server must show early native prefetch and no spills before
full numerical admission and immediate d2 API. >3% slowdown stops, neutral
removes, useful direction unlocks d3. No distance/cache-level scan or profile
follows. Production and the 1.5x–3x target remain unchanged.

The [completed direction](metadata/qwen36-27b-mtp-native-prefetch-direction-2026-10-10.json)
passes exact compiled-instruction/no-spill gates, 36 complete output/guard cases,
all P65 numerical/cache/transaction checks and sixteen d2/d3 API checks.
Both policies reduce complete Decode and verification at 8K/40K with identical
output/work. D2 reaches 14.143/10.672 token/s, or
1.535x/1.357x non-MTP. Retain this composition as a bounded
next-verifier dependency, without a distance/cache-level scan or production
promotion. The full 1.5x–3x goal remains unmet; 40K still needs approximately
2.27 seconds removed from full Decode. These are single-process directions,
not noise-qualified speedups or measured cache-hit attribution.

## 2026-10-10 MTP complete FP8 operand feed — closed at static gate

`WP-MTP-COMPLETE-FEED-20261010` / `AC-MTP-GREEDY-v48` owns one
[complete operand pipeline](metadata/qwen36-27b-mtp-complete-feed-design-2026-10-10.json)
against the remaining 2.67-second 40K Decode gap. Both next weights and next
activations become register-resident before current ordered FMA consumption;
the rejected weight-only v29 left activation reads on the current dependency
path. Current epilogues, scalar arithmetic and storage ownership remain fixed.
Historical L2 miss bytes closely match one weight pass, narrowing the traffic
hypothesis without asserting a current DRAM or PC-stall cause. One no-spill/
full-numerical gate returns immediately to d2 API; >3% slowdown stops, neutral
removes, useful direction unlocks d3. No profile or parameter scan follows.
Production and the complete 1.5x–3x target remain unchanged.

The [completed static rejection](metadata/qwen36-27b-mtp-complete-feed-rejection-2026-10-10.json)
passes builds and no-spill resource checks but fails its mechanism gate: in
the actual server's M3/K5120 loop, all next-record global loads follow the
last current FFMA. The compiler removed the intended overlap. This construction
is removed before GPU numerical or API runs; no new performance result follows.
Both retained-v48 executables are rebuilt with identical program sections.
A successor must prove its native producer/consumer overlap before GPU testing;
source ordering alone is insufficient. No launch/feed parameter scan follows.
Production, retained metrics and the complete 1.5x–3x objective are unchanged.

## 2026-10-10 MTP FP8 lane-owned codebook — closed negative

`WP-MTP-FP8-LANE-TABLE-20261010` / `AC-MTP-GREEDY-v47` addresses the
remaining 2.67-second 40K Decode gap with one [packed FP8 consumer composition](metadata/qwen36-27b-mtp-fp8-lane-table-design-2026-10-10.json).
A lane-owned codebook removes possible lookup bank aliasing while preserving
all scalar arithmetic and current output ownership. Unlike the rejected v12
bundle, it adds neither full-K input staging nor a persistent output grid.
The 5.869-second family is an opportunity budget, not evidence of bank stalls.
One static/full-numerical gate returns directly to the d2 API; >3% slowdown
stops, neutral removes and useful direction unlocks d3. No parameter scan or
profile follows. Production and the complete 1.5x–3x target remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-fp8-lane-table-rejection-2026-10-10.json)
passes all 36 complete FP8 output/guard cases and full P65 prefix, state/logit,
cache and transaction checks. P8192/O256 d2 nevertheless falls to 13.398
Decode token/s versus v48 13.833, with identical output and work and
0.602 seconds more verification. Prefill is 17.299 seconds / 473.55 token/s,
external TTFT 17.321 seconds and total 36.354 seconds. The declared stop
interrupts 40K; d3 and the remaining API lifecycle panel are not run.
All lane-table and profile paths are removed and retained v48 rebuilt.
No lookup-layout scan or profiler follows. This closes the consumer-table
hypothesis without establishing a hardware cause or changing the unmet goal.

## 2026-10-10 retained-v48 budget reconciliation — complete

`WP-MTP-V48-BUDGET-20261010` uses one [bounded diagnostic](metadata/qwen36-27b-mtp-v48-budget-design-2026-10-10.json)
on the exact retained v48 API binary and P40000/O256 d2 request. It asks which
verifier families still dominate the remaining 2.67-second full Decode gap
after QK, paired weights, prefix publication and row-boundary changes. One
NSys capture reconciles kernel families, copies and uncovered intervals with
the old v31 ranking. No NCU, runtime mutation or parameter scan is included.
The historical analysis window is retained explicitly; profiler durations do
not replace unprofiled API metrics. Completion must select or exclude the next
architecture directions, not open a repetition campaign. The full 1.5x–3x
target and production remain unchanged.

The [completed diagnosis](metadata/qwen36-27b-mtp-v48-budget-2026-10-10.json)
preserves exact request/output/work and normal shutdown. Current FP8 costs
5.869 seconds, Gate/Up 5.802 and QK/PV 4.532 in the declared diagnostic window.
Copies plus uncovered intervals total 0.516 seconds; a launch/copy-only change
cannot supply the remaining 2.67-second budget in this opportunity comparison.
Prioritize a distinct packed FP8 consumer composition, preserving original
scalar chains, rather than repeating the closed Gate/Up/PV/storage variants.
No hardware-stall cause or new runtime implementation is selected by NSys.
No further capture, NCU or parameter scan follows this diagnostic package.
Unprofiled v48 metrics, production and the unmet goal remain unchanged.

## 2026-10-10 MTP Gate/Up input residency — closed without retention

`WP-MTP-GATE-RESIDENCY-20261010` / `AC-MTP-GREEDY-v46` preserves the
1.5x–3x objective and remaining 2.67-second 40K full Decode gap. The [design](metadata/qwen36-27b-mtp-gate-residency-design-2026-10-10.json)
keeps existing Gate/Up shared inputs and paired codebooks resident across
disjoint output groups. The rejected v9 bundle did not isolate this dependency;
its FP8 executor is not reintroduced. Current exact paired scaling, K chains
and publication remain unchanged. One fixed resident-wave grid passes native
resource, complete projection and full P65 state/cache gates before immediately
returning to P65/8K/40K d2 API. Greater-than-3% slowdown stops, neutral removes,
useful direction unlocks d3. No geometry scan or profile follows. Useful
retention expires at the next complete verifier API return or archival;
production and its reserve remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-gate-residency-rejection-2026-10-10.json)
passes native occupancy, 45 complete projection/guard cases, all P65 numerical/
transaction checks and all eight d2 API checks. Outputs and physical/logical
work match v48, but 8K/40K Decode is 13.744/10.439 token/s versus
13.833/10.498. These single-process differences establish no useful direction,
not a statistical regression. Lower register use and staging requests do not
select the composition. All new Gate/Up/resource-plan/profile paths are removed
and v48 rebuilt. No d3, profile or geometry scan follows; the full 1.5x–3x
objective remains active and unmet, and production remains unchanged.

## 2026-10-10 MTP row boundary composition — API return complete

`WP-MTP-ROW-BOUNDARY-20261010` / `AC-MTP-GREEDY-v45` preserves the
1.5x–3x objective and remaining 0.41/3.14-second full Decode gap. The [design](metadata/qwen36-27b-mtp-row-boundary-design-2026-10-10.json)
reuses the existing exact independent-row residual/norm operator in both
per-layer boundaries, removing repeated scalar cooperative-grid launches.
The historical family costs 0.605 seconds and cannot alone close 40K; this
is a bounded scheduling prerequisite, not a reduced product objective.
One composition passes full P65 state/logit/cache and recovery admission,
then immediately returns to P65/8K/40K d2 API. Greater-than-3% slowdown stops,
neutral removes, useful direction unlocks d3. No kernel/geometry scan or
profile follows. Useful retention expires at the next complete verifier API
return or archival; production remains unchanged.

The [completed direction](metadata/qwen36-27b-mtp-row-boundary-direction-2026-10-10.json)
preserves every P65 state/cache prefix and transaction/recovery check plus all
sixteen d2/d3 API outputs. Both policies reduce verification and complete Decode
at 8K/40K. D2 reaches 13.833/10.498 token/s, or
1.502x/1.335x non-MTP. The narrow single-process 8K crossing is not
repeated qualification. Retain this scheduling composition as a bounded next-
verifier dependency, without further geometry scans or production promotion.
The full 1.5x–3x goal remains active and unmet; 40K still needs approximately
2.67 seconds removed from full Decode.

## 2026-10-10 MTP register codebook — closed negative

`WP-MTP-REGISTER-CODEBOOK-20261010` / `AC-MTP-GREEDY-v44` preserves the
1.5x–3x objective and remaining 0.41/3.14-second full Decode gap. The [design](metadata/qwen36-27b-mtp-register-codebook-design-2026-10-10.json)
replaces repeated shared-memory E2M1 pair lookups with exact register byte
selection, retaining v46 state ownership and v45 paired scaling. The historical
NV projection family supplies a roughly 9.5-second opportunity budget, not
predicted savings. One composition passes exhaustive pair/product and complete
projection checks plus full P65 numerical admission, then returns immediately
to P65/8K/40K d2 API. Greater-than-3% slowdown stops, neutral removes, useful
direction unlocks d3. No decoder parameter scan or profile follows. Useful
retention expires at the next complete verifier API return or archival.
Production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-register-codebook-rejection-2026-10-10.json)
passes exhaustive raw pairs/scaled products, 45 complete projection/guard cases,
full P65 target/cache prefix and transaction/recovery checks, but P65 d2 API
Decode falls to 10.917 token/s versus v46 12.283.
Prefill is 0.887 seconds / 73.27 token/s, external TTFT
0.906 seconds and total 2.280 seconds. Output and physical/logical
work match. The stop prevents a completed long-context/d3 panel. All new
runtime paths are removed and v46 rebuilt; no codebook scan or profile follows.
The 1.5x–3x goal remains active and unmet, with production unchanged.

## 2026-10-10 MTP expanded QKV representation — closed host assessment

`WP-MTP-EXPANDED-QKV-ASSESSMENT-20261010` assesses exact BF16 expansion
of the 48 GDN input-QKV matrices in dead Prefill storage against the remaining
0.41/3.14-second 8K/40K full Decode budgets. The [completed assessment](metadata/qwen36-27b-mtp-expanded-qkv-assessment-2026-10-10.json)
proves the proposed 5,033,164,800-byte layout fits, but does not authorize a
storage borrow. The unchanged FP8 consumer's historical same-model 40K cell
costs 2.051 seconds over 5,040 calls. Expanded weight traffic alone takes
2.168 seconds in an optimistic model allowing 16 MiB of useful cached weights
per call and full documented peak bandwidth, before any preparation or other
traffic. This is a conditional host screen, not measured candidate performance
or a hardware impossibility claim. Do not implement this representation or
repeat a GPU screen. No runtime or current API metrics change; retained v46
and the 1.5x–3x objective remain active. Further verifier work must address
the complete remaining budget without assuming decoded storage is free.

## 2026-10-10 MTP prefix publication — API return complete

`WP-MTP-PREFIX-PUBLICATION-20261010` / `AC-MTP-GREEDY-v43` preserves the
1.5x–3x goal and remaining 0.75/3.51-second full Decode gap. The [design](metadata/qwen36-27b-mtp-prefix-publication-design-2026-10-10.json)
removes dead entry snapshots, publishes original GDN states directly into
immutable prefix slots, and reuses the exact first seed-conditioned draft KV.
This bounded transaction prerequisite cannot alone close the 40K gap. One
composition passes full P513 state/logit/cache and recovery checks, then
immediately returns to P65/8K/40K d2 API. Greater-than-3% slowdown stops,
neutral removes, useful direction unlocks d3. No profile or parameter scan
follows; useful retention expires at the next complete verifier API return or
archival. Production and observer-visible state semantics remain unchanged.

The [completed direction](metadata/qwen36-27b-mtp-prefix-publication-direction-2026-10-10.json)
preserves every P513 target/cache prefix and transaction/recovery check plus
all sixteen d2/d3 API outputs. Both policies reduce verification, reconciliation
and complete Decode at 8K/40K. D2 reaches 13.518/10.299 token/s,
only 1.467x/1.310x non-MTP. Retain this exact ownership composition as a
bounded next-verifier dependency, without further transaction scans or production
promotion. The 1.5x–3x goal remains active and unmet; full Decode still needs
approximately 0.41/3.14 seconds removed at 8K/40K.

## 2026-10-10 MTP paired weight decoding — API return complete

`WP-MTP-PAIRED-WEIGHT-DECODE-20261010` / `AC-MTP-GREEDY-v42` preserves the
1.5x–3x objective and remaining 1.16/3.94-second full Decode gap. The [design](metadata/qwen36-27b-mtp-paired-weight-decode-design-2026-10-10.json)
replaces repeated scalar NVFP4 decoding/scaling with exact paired products,
keeping all FP32 accumulation and state boundaries. One fixed composition
passes exhaustive product and full numerical checks, then immediately returns
to P65/8K/40K d2 API. Greater-than-3% slowdown stops, neutral removes, useful
direction unlocks d3. No parameter scan or profile follows. Any useful dependency
expires at the next complete verifier API return or archival. Production is
unchanged; family timing is an opportunity ceiling, not predicted savings.

The [completed direction](metadata/qwen36-27b-mtp-paired-weight-decode-direction-2026-10-10.json)
passes exhaustive native products, 45 complete output/guard cases, full P65
numerical/transaction checks and all sixteen d2/d3 API checks. Both policies
reduce verification and complete Decode at 8K/40K with unchanged output/work.
D2 reaches 13.275/10.147 token/s, only 1.441x/1.291x non-MTP.
Retain this exact decoder as a bounded next-composition dependency, with no
further decoder/resource scan or production promotion. The 1.5x–3x goal remains
active and unmet; the remaining full Decode gap is 0.75/3.51 seconds.

## 2026-10-10 MTP direct short-chain execution — closed negative

`WP-MTP-DIRECT-CHAIN-20261010` / `AC-MTP-GREEDY-v41` preserves the
1.5x–3x goal and remaining 1.16/3.94-second full Decode gap. The [design](metadata/qwen36-27b-mtp-direct-chain-design-2026-10-10.json)
reuses proved exact short-chain eligibility but replaces the rejected executor's
expanded shared operands and serial eight-group loop with direct packed register
fragments and independent original-warp producers. A separate original-tree
merge consumes bounded FP32 roots. Preparation and repairs remain charged.
One fixed composition passes full synthetic/P65 admission, then returns
immediately to 8K/40K d2 API; greater-than-3% slowdown stops, neutral removes,
useful direction unlocks d3. No guard, layout, resource or geometry scan follows.
Any useful dependency expires at the next complete verifier API return or
archival. Production and its numerical contract remain unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-direct-chain-rejection-2026-10-10.json)
passes all synthetic and full P65 numerical/transaction checks, but P8192/O256
d2 Decode falls to 3.061 token/s versus retained v41 13.001.
Prefill is 17.302 seconds / 473.46 token/s, external TTFT
17.325 seconds and total 100.624 seconds. Both completed API outputs
and acceptance/work match; verification adds 63.684 seconds at 8K.
The declared stop prevents 40K/d3 and the remaining API lifecycle panel.
All new executor, metadata ownership and profile paths are removed and v41
rebuilt. Direct fragments and independent warp producers do not make this
exact short-chain executor competitive; numerical eligibility alone cannot
select it. No profile or further guard/layout/grid scan follows. The 1.5x–3x
goal remains active and unmet, and production remains unchanged.

## 2026-10-10 MTP Gate/Up shared-input chain composition — closed negative

`WP-MTP-GATE-CHAIN-SHARE-20261010` / `AC-MTP-GREEDY-v40` preserves the
1.5x–3x goal and remaining 1.16/3.94-second complete Decode gap. The
[fixed design](metadata/qwen36-27b-mtp-gate-chain-share-design-2026-10-10.json)
composes independent Gate/Up chains with retained full-K input sharing and
original N16 output ownership. The closed v34 removed that producer reuse and
also changed Down; this composition retains Down and all other v41 execution.
One fixed implementation must pass static resource and full numerical gates,
then immediately return to P65/8K/40K d2 API. Greater-than-3% slowdown stops,
neutral removes, useful direction unlocks d3. No geometry or resource scan
follows. A useful dependency expires at the next full verifier API return or
archival. Production remains unchanged; family cost is not projected savings.

The [completed rejection](metadata/qwen36-27b-mtp-gate-chain-share-rejection-2026-10-10.json)
passes static resource, nine complete output/guard cases and all P65 numerical
and transaction checks, but P65/O16 d2 Decode falls to 10.748 token/s versus
retained v41 11.879. Output and acceptance/work are identical; verification
adds 133.3 ms. The declared stop interrupts the following 8K request before
completion; no 40K, d3 or full API lifecycle panel follows. All new runtime
paths are removed and v41 rebuilt. Retaining input reuse does not rescue this
chain-consumer composition. No further chain, geometry or resource scan opens.
The 1.5x–3x goal remains active and unmet; production is unchanged.

## 2026-10-10 MTP shared-row PV recomposition — closed without retention

`WP-MTP-PV-ROW-COMPOSE-20261010` / `AC-MTP-GREEDY-v39` preserves the1.5x–3x
goal and remaining1.16/3.94-second full Decode gap. The [fixed composition](metadata/qwen36-27b-mtp-pv-row-compose-design-2026-10-10.json)
adds the existing exact shared-row PV component to retained compact QK. The
historical v19 combined failure did not identify PV's separate contribution;
this one composition answers that question without a tile or staging scan.
Full P513/P8192 admission returns immediately to8K/40K d2 API; greater-than3%
slowdown stops, neutral removes, useful direction unlocks d3. A useful dependency
expires at the next complete verifier API return or archival. PV alone cannot
close the40K gap. Production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-pv-row-compose-rejection-2026-10-10.json)
preserves full P513/P8192 numerical checks and all eight d2 API checks, with
unchanged output and acceptance/work. Decode is12.989/9.957 token/s at8K/40K,
respectively0.09%/0.19% below retained v41 in one process. This provides no
useful gain or statistical regression claim. All new PV/profile/auditor paths
are removed and v41 rebuilt. No d3, profile or staging/geometry scan follows.
Retained v41 and the unmet1.5x–3x objective remain unchanged.

## 2026-10-10 MTP compact QK row sharing — API return complete

`WP-MTP-QK-ROW-SHARE-20261010` / `AC-MTP-GREEDY-v38` preserves the1.5x–3x
objective and remaining1.19/4.21-second complete Decode gap. The [design](metadata/qwen36-27b-mtp-qk-row-share-design-2026-10-10.json)
shares each K position across all speculative rows in a compact2*M-warp CTA,
keeping all six query heads and original numerical trees. This replaces the
rejected large8*M-warp/head-split ownership, not its parameters. One fixed
implementation passes P513/P8192 full state/logits and immediately returns to
8K/40K d2 API; useful direction unlocks d3, greater-than3% slowdown stops and
neutral removes. No geometry or staging scan follows. Any useful dependency
expires at the next complete verifier API return or archival. Production is
unchanged; this QK budget alone cannot close the40K target gap.

The [completed direction](metadata/qwen36-27b-mtp-qk-row-share-direction-2026-10-10.json)
preserves all complete numerical/transaction checks and all sixteen d2/d3 API
outputs and work counts. Verification falls by34/51ms at8K and269/306ms at40K.
D2 Decode is13.001/9.976 token/s; its non-MTP ratios remain only1.411x/1.269x.
This is a small bounded dependency through the next complete verifier API
return or archival, not noise-qualified architecture selection. Prefill changes
are outside the edited code; no whole-request attribution or production
promotion follows. No mapping or staging scan remains active. The1.5x–3x
goal remains unmet, with approximately1.16/3.94 seconds left in full Decode.

## 2026-10-10 MTP lossless FP8 traffic — closed at host gate

`WP-MTP-LOSSLESS-FP8-20261010` preserves the 1.5x–3x objective and remaining
1.19/4.21-second complete Decode gap. The [fixed packet design](metadata/qwen36-27b-mtp-lossless-fp8-design-2026-10-10.json)
assesses all208 real FP8 tensors using six-bit palette codes, literal exceptions
and bounded packet offsets. It preserves raw bytes and original arithmetic.
Less than20% complete payload saving closes before GPU, without a format scan.
A pass requires one separately bounded owner/decoder/executor composition and
immediate real API return; compression alone grants no speedup or retention.
Production and retained v39 remain unchanged.

The [completed assessment](metadata/qwen36-27b-mtp-lossless-fp8-assessment-2026-10-10.json)
authenticates all three shards and all208 FP8 tensors. Including literal
exceptions, packet offsets and palettes, 7,214,202,880 raw bytes become
6,702,302,636 bytes: only7.096% saving, below the declared20% floor. All3328
sampled packet roundtrips and12 controls preserve raw bytes. The format closes
before native implementation or GPU work; no alternate format scan follows.
No speedup, hardware bound or new API measurement is claimed. The unchanged
1.5x–3x goal still requires a materially larger complete-verifier improvement.

## 2026-10-10 MTP short-chain executor — closed negative

`WP-MTP-SHORT-CHAIN-EXECUTOR-20261010` / `AC-MTP-GREEDY-v37` implements the
[fixed complete executor](metadata/qwen36-27b-mtp-short-chain-executor-design-2026-10-10.json)
after native/current-payload admission. The unchanged 1.5x–3x goal and
1.19/4.21-second remaining complete Decode gap select compact operands,
nine-bit metadata, cheap runtime guards, exact short-chain repairs and on-chip
original reductions together. Preparation and repair remain charged to Decode.
One implementation plus at most one correctness repair returns through complete
synthetic/P65 state admission immediately to 8K/40K d2 API; useful direction
unlocks d3. Greater-than-3% long-context slowdown stops; neutral removes.
No further guard/layout/grid scan is active. Production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-short-chain-executor-rejection-2026-10-10.json)
preserves all synthetic and full P65 state/logit/transaction checks, plus both
completed API outputs and work counts. P8192/O256 d2 Decode falls to 4.291
token/s versus retained v39 12.978, so 40K/d3 are not run. One same-ELF P65
profile assigns 2.745 GPU seconds to repeated FP8 execution, versus 0.167 to
one-time metadata and 0.006 to input bounds. Preparation amortization cannot
rescue this version. The executor and borrowed ownership are removed and v39
rebuilt; no guard/layout/grid scan or additional certificate campaign follows.
The 1.5x–3x objective remains unmet. A successor must eliminate repeated
execution cost, not merely increase numerical eligibility or move preparation.

## 2026-10-10 MTP short-chain native admission — prerequisite complete

`WP-MTP-SHORT-CHAIN-DEVICE-20261010` preserves the 1.5x–3x objective and
remaining 1.19/4.21-second Decode gap. The [design](metadata/qwen36-27b-mtp-short-chain-device-design-2026-10-10.json)
checks one native BF16 matrix instruction that pairs two original short chains
in disjoint K/column blocks. The earlier 128-cell host grouping was a proxy,
not a realizable dense fragment; the explicit M3 mapping has 96 useful cells.
One numerical device process must pass before one current P65 adjacent-row
capture. Product, root, fragment and guard checks do not establish speedup.
Coverage below 50% or numerical failure closes this version without scanning.
After both gates, a bounded complete guard/operand/on-chip reduction/fallback
executor must return promptly to the real API. Production remains unchanged.

The [completed admission](metadata/qwen36-27b-mtp-short-chain-device-admission-2026-10-10.json)
passes 16,255,180 native valid product pairs and 581,632 matrix output checks,
with zero mismatches. Current P65 capture preserves all three complete prefix
states/full logits and all 2,688 saved projection outputs. Actual paired M3
complete-unit eligibility is 20,957/28,672 (73.09%), above the declared floor.
One fixed separable exponent guard admits 2,448,199/2,752,512 individual roots
(88.94%), with no false positives against the exact host predicate. This closes
numerical feasibility, not runtime guard economics or a speedup. Temporary
capture seams are removed and the exact retained v39 oracle is rebuilt.
The next work is one composed executor with cheap guards, compact operands,
on-chip original reductions and exact fallback, followed immediately by the
same real API. No further standalone admission campaign is pending. The
1.5x–3x target remains unmet and production is unchanged.

## 2026-10-10 MTP original short-chain exactness — host feasibility passed

`WP-MTP-SHORT-CHAIN-LATTICE-20261010` keeps the 1.5x–3x goal and remaining
1.19/4.21-second full Decode gap. The [design](metadata/qwen36-27b-mtp-short-chain-lattice-design-2026-10-10.json)
checks exact integer-lattice closure of the original five/six-product FP8
chains, preserving every later reduction and publication. This differs from
approximate full-K certificates and rounded-carry MMA substitution. One
hash-authenticated historical real-input host panel reproduces complete scalar
outputs and assesses 128-cell cooperative units. Below 50% complete-unit
coverage, close without GPU; a pass requires separate current-payload and
native-instruction proof before a bounded executor and immediate API return.
No numerical waiver, timing or parameter scan opens here. Production remains
unchanged; artifacts stay under `.q3x-work/mtp-short-chain-lattice-20261010/`.

The [completed host assessment](metadata/qwen36-27b-mtp-short-chain-lattice-feasibility-2026-10-10.json)
reproduces all 2,688 captured outputs and finds 2,734,028 of 2,752,512 chains
exact (99.33%). Complete 128-cell coverage is 13,616/21,504 (63.32%), above
the declared floor; every admitted root matches its independent integer sum.
This closes the host assessment positively, with no GPU or performance claim.
Next admission must bind current adjacent output groups and prove native
zero-start instruction behavior. Runtime guard cost, compact operand delivery,
on-chip original-tree merging and fallback remain unresolved; repeating the
old costly certificate executor is not selected. The 1.5x–3x goal is unmet.

## 2026-10-10 MTP QK live reductions — API return complete

`WP-MTP-QK-LIVE-REDUCTION-20261010` / `AC-MTP-GREEDY-v36` retains the
1.5x–3x objective and complete 18.45/21.62-second Decode budgets. The
[design](metadata/qwen36-27b-mtp-qk-live-reduction-design-2026-10-10.json)
transfers the retained exact live-ancestor projection reduction to six QK head
trees, eliminating dead-lane work without changing products, grids, causal
bounds or KV ownership. Current matched QK work is 3.077 GPU seconds; this
alone cannot close the remaining 40K gap. One fixed implementation plus at
most one correctness repair passes directed trees and complete P513/P8192
state/logits, then returns immediately to the same d2 API. Greater-than-3%
regression stops; neutral removes; useful direction unlocks d3. No mapping
or resource scan follows. Any useful prerequisite expires at the next full
verifier API return or archival. Production remains unchanged.

The [completed direction](metadata/qwen36-27b-mtp-qk-live-reduction-direction-2026-10-10.json) passes 24,576 raw FP32 roots,
complete P513/P8192 state/logits and all sixteen d2/d3 API/lifecycle checks.
Output and acceptance/work remain unchanged. Verification falls by about
0.10 seconds at 8K and 0.57/0.63 seconds at 40K; d2 Decode is 12.978/9.872
token/s. This exact reduction remains a bounded development dependency through
the next full verifier API return or archival. There is no further QK mapping
scan or qualification-only campaign. The single-process 40K total is mixed
because Prefill is longer; no production selection follows. The 1.5x–3x goal
remains unmet, with roughly 1.20/4.21 seconds still to remove from complete
8K/40K Decode. The next architecture must address dominant target computation.

## 2026-10-10 retained MTP v31 budget — diagnosis complete

`WP-MTP-V31-BUDGET-20261010` preserves the 1.5x–3x goal and complete
18.45/21.62-second Decode budgets. The [design](metadata/qwen36-27b-mtp-v31-budget-design-2026-10-10.json)
permits one same-frozen-v31-ELF P40000/O256 d2 real API capture after its
completed unprofiled direction. The old full-model attribution predates the
retained draft/head/projection changes. This capture updates family/copy/gap
attribution to select the next complete execution response; it changes no
performance baseline, runtime or production route. Close on completion or
resource failure, with no NCU replay or parameter scan. Artifacts remain under
`.q3x-work/mtp-v31-budget-20261010/`.

The [completed diagnosis](metadata/qwen36-27b-mtp-v31-budget-2026-10-10.json)
preserves output, acceptance and work counts at the same frozen v31 ELF.
From first verifier projection to last generation kernel, current verifier
projections take 14.706 GPU seconds, QK/PV 5.395, all copies 0.522 and the
uncovered activity interval 0.317. The latter two together are far below the
remaining 4.774-second unprofiled 40K Decode gap. Collection is closed; no
hardware-stall cause, ceiling or new performance baseline is claimed. The next
complete computation/executor response must address dominant verification;
copy/coordination work and repeated rejected layout/feed variants cannot
substitute for that budget. Production, retained v31 and the unmet goal remain
unchanged.

## 2026-10-10 MTP activation publication — closed negative

`WP-MTP-ACTIVATION-PUBLICATION-20261010` / `AC-MTP-GREEDY-v35` targets the
unchanged 1.5x–3x objective and 18.45/21.62-second complete Decode budgets.
The [design](metadata/qwen36-27b-mtp-activation-publication-design-2026-10-10.json)
publishes exact FP32 inputs once per transformer projection for all output
consumers. Original FMA/reduction/publication boundaries remain unchanged;
doubled input traffic and producer overhead must be repaid in the actual API.
One implementation plus at most one correctness repair passes resource and
full numerical admission, then immediately returns to P65/8K/40K d2 API.
Greater-than-3% slowdown stops; neutral removes; useful direction unlocks d3.
No parameter scan follows. Any retained dependency expires at the next full
verifier API return or archival. Production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-activation-publication-rejection-2026-10-10.json)
passes all 65,536 BF16 patterns, 72 complete projection cases and full P65
state/logit/transaction checks. P65/O16 d2 preserves output and work, but
Decode falls to 10.737 token/s versus v31 11.902, a 9.78% negative direction.
Verification adds 136.775 ms. The stop occurs before 8K; no 40K, d3 or full
API lifecycle claim follows. All new runtime/profile/auditor paths are removed
and v31 rebuilt. No activation representation/cache/grid scan follows. The
1.5x–3x goal remains active and unmet; removing unpack instructions without
improving the complete projection execution cost did not deliver the target.

## 2026-10-10 MTP batched cache reconciliation — closed without retention

`WP-MTP-BATCH-RECONCILE-20261010` / `AC-MTP-GREEDY-v34` preserves the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The [design](metadata/qwen36-27b-mtp-batch-reconcile-design-2026-10-10.json)
batches already available target-conditioned cache rows before publishing each
accepted logical prefix. It removes repeated FC/K/V weight traversals while
preserving per-token observer state and the original draft arithmetic. The
approximately 0.447-second reconciliation phase bounds this prerequisite;
it cannot alone close the remaining complete-verifier gap.

One implementation plus at most one correctness repair returns through full
P65 state/logit/cache/transaction admission immediately to the same d2 API.
Greater-than-3% slowdown stops; neutral removes; useful direction unlocks d3.
No batch-size or draft-length scan follows. Any useful dependency expires at
the next complete verifier API return or archival. Artifacts remain under
`.q3x-work/mtp-batch-reconcile-20261010/`; production is unchanged.

The [completed direction](metadata/qwen36-27b-mtp-batch-reconcile-rejection-2026-10-10.json)
passes all nine complete target and draft-cache prefix comparisons, tail and
prepared-authority guards, full transaction/recovery admission and all eight
d2 API/lifecycle checks. Work and acceptance match v31. Reconciliation falls
by 131/125 ms at 8K/40K, but complete Decode changes only +0.61%/+0.36% in
one process, and 40K external total is effectively unchanged. This does not
select useful whole-runner retention. All new runtime/profile/test/auditor paths
are removed and v31 rebuilt. No d3, repetition or batching scan follows. The
1.5x–3x objective remains unmet; dominant target verification, rather than this
small coordination phase, must supply the remaining complete Decode savings.

## 2026-10-10 MTP exact QK lattice — broad-cooperative proposal closed

`WP-MTP-QK-LATTICE-20261010` preserves the 1.5x–3x real API objective and
18.45/21.62-second complete Decode budgets. Its [design](metadata/qwen36-27b-mtp-qk-lattice-design-2026-10-10.json)
checks one exact integer-lattice predicate for the original eight-product QK
subtrees. All signed intermediate sums must fit FP32 exactly before any
order-independent executor could be considered. One authenticated historical
real-input panel checks coverage and the original scalar tree; its older
Legacy Prefill inputs cannot establish current MTP eligibility or performance.
Below 50% complete cooperative-unit coverage, close this broad-cooperative
proposal without a GPU or parameter scan. Above that floor, current MTP capture
and an independent native-instruction proof must precede a separately bounded
executor with immediate real API return. The old 3.077-second QK attribution
cannot alone close the 40K gap. Artifacts remain under
`.q3x-work/mtp-qk-lattice-20261010/`; runtime and production are unchanged.

The [completed assessment](metadata/qwen36-27b-mtp-qk-lattice-feasibility-2026-10-10.json)
authenticates the four-layer P40000 payload panel and checks 786,432 original
eight-product groups. Of these, 672,226 satisfy the exact predicate, with zero
admitted scalar-tree mismatches; independent integer checks cover every subset
of 64 selected admitted groups. Only 990 of 16,384 complete cooperative units
pass (6.04%), below the declared floor. This broad-cooperative proposal closes
before current-MTP capture or GPU implementation. No per-cell repair or mapping
scan follows. The historical-input scope cannot establish current MTP eligibility,
native instruction equivalence or a performance ceiling. Retained v31 and its
performance observations remain unchanged; the 1.5x–3x objective stays active.

## 2026-10-10 MTP Gate/Up producer-consumer fusion — closed without retention

`WP-MTP-GATE-PAIR-20261010` / `AC-MTP-GREEDY-v33` preserves the 1.5x–3x
goal and 18.45/21.62-second complete Decode budgets. The [design](metadata/qwen36-27b-mtp-gate-pair-design-2026-10-10.json)
selects one paired Gate/Up executor: separate original warp consumers share
input/table publication, then original BF16-rounded values feed exact SiLU
inside the CTA. It halves duplicate input production without doubling each
thread's accumulators and removes dead global intermediate publication. The
older 5.874-second family attribution is a ceiling, not projected savings.
State-copy removal and draft precision retuning are not selected by the bounded
source/evidence audit. No state, numerical, acceptance or inventory change occurs.

One fixed implementation plus at most one correctness repair passes no-spill
and resource-floor checks, complete producer/SiLU output guards and P65 full
prefix/state/logit transactions, then immediately returns to the same P65/8K/40K
d2 API. Greater-than-3% slowdown stops; neutral/negative removes; useful direction
unlocks d3. No launch/register/stage scan follows. A useful dependency expires
at the next complete verifier API return or archival. Artifacts stay under
`.q3x-work/mtp-gate-pair-20261010/`; production remains unchanged.

The [completed direction](metadata/qwen36-27b-mtp-gate-pair-rejection-2026-10-10.json)
passes all static, twelve full-output/guard, complete P65 state/logit/transaction
and eight d2 API/lifecycle checks. Work and acceptance match v31. At 8K/40K,
Decode is 12.915/9.656 token/s versus 12.914/9.660, only +0.014%/-0.045% in
one process; verification differs by -4.8/+9.9 ms. No useful whole-runner gain
is established. All new runtime/profile/auditor paths are removed and v31
rebuilt. No d3, profiler, producer-grid or register scan follows. The 1.5x–3x
goal remains active and unmet; fewer publications and launches did not address
the material verifier cost. A successor needs a different complete computation
or execution response with an immediate real API return.

## 2026-10-10 MTP direct PV consumption — closed negative

`WP-MTP-DIRECT-PV-20261010` / `AC-MTP-GREEDY-v32` keeps the 1.5x–3x goal
and complete 18.45/21.62-second Decode budgets. The [design](metadata/qwen36-27b-mtp-direct-pv-design-2026-10-10.json)
removes the shared PV producer/copy ring and CTA barriers, retaining the exact
increasing-position FMA chains in independent full-head consumers. Repeated
global V loads replace physical shared reuse, so only the complete API selects
the trade. This differs from the rejected shared-KV ownership and rounding
elision mechanisms. The older 2.318-second PV attribution cannot alone close
the 40K gap and is not a projected gain.

One fixed implementation plus at most one correctness repair passes static
resource review and complete P513/P8192 scalar state/logit admission, then
returns immediately to P65/8K/40K d2 API; useful direction unlocks d3. A
greater-than-3% long-context regression stops; neutral/negative removes the
new route. No prefetch/cache/grid scan follows. A useful dependency expires at
the next complete verifier API return or archival. Artifacts remain under
`.q3x-work/mtp-direct-pv-20261010/`; production is unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-direct-pv-rejection-2026-10-10.json)
passes the static no-spill gate, all eighteen P513/P8192 prefix comparisons
and full P513 transaction/recovery checks. P65 output/performance is unchanged,
but P8192/O256 d2 Decode falls to 11.750 token/s versus v31 12.914, with
identical acceptance/work. Verification adds 1.950 seconds; external total
adds 1.981 seconds. The declared stop occurs during the idle gap before 40K;
40K, d3 and the full API lifecycle panel are not run. All new runtime paths
are removed and v31 restored. No cache/prefetch/grid scan follows. This closes
direct global consumption without a hardware-stall claim; the 1.5x–3x goal
remains active and unmet, and the next architecture must address a materially
different complete verification budget rather than repeat PV ownership variants.

## 2026-10-10 MTP scalar-chain Tensor Core mapping — direct substitution rejected

`WP-MTP-SINGLE-PRODUCT-MMA-20261010` retains the 1.5x–3x goal and complete
18.45/21.62-second Decode budgets. The [bounded design](metadata/qwen36-27b-mtp-single-product-mma-design-2026-10-10.json)
tests a different executor prerequisite: one nonzero product per MMA output,
with the original FP32 accumulator carried in original chain order. NVIDIA's
ISA does not guarantee identical rounding to scalar FMA. One directed SM87
numerical process therefore checks finite rounding and final BF16 publication
before any runtime integration. A finite counterexample closes this direct
substitution; a sample pass alone is not equivalence proof or selection. No
timing, profiler or geometry scan follows this screen. Artifacts remain under
`.q3x-work/mtp-single-product-mma-20261010/`; production is unchanged.

The [completed numerical screen](metadata/qwen36-27b-mtp-single-product-mma-2026-10-10.json)
confirms ordinary finite counterexamples on SM87. Exact layout controls and
guards pass, and GPU scalar FMA agrees with independent host/directed bits.
The zero-start four-product chain yields scalar FP32 `0x3f818000` versus MMA
`0x3f817fff`, publishing different BF16 values `0x3f82` versus `0x3f81` on
all 128 outputs. Every operand is an exactly representable FP8 weight or BF16
activation. Thus eliminating multi-product reassociation does not make this
instruction a drop-in scalar FMA. The direct substitution is closed before
runtime integration, with no timing or hardware-performance conclusion. A
sample cannot quantify real-model mismatch frequency. Retained v31 and the
unmet 1.5x–3x objective remain unchanged; a successor must supply a different
proved computation or exact execution dataflow and a bounded real API return.

## 2026-10-10 MTP role-aware draft representation — rejected before GPU

`WP-MTP-MIXED-DRAFT-20261010` keeps the 1.5x–3x goal and exact target
verification. The [bounded design](metadata/qwen36-27b-mtp-mixed-draft-design-2026-10-10.json)
responds to the measured FC diagonal risk by preserving the complete FC matrix
in BF16 and using one fixed K128-group INT8 representation for the other seven
draft matrices. Its host screen includes BF16 operand reconstruction before
independent FP64 projections and compares complete hidden/K/V to the original
reference at the unchanged 0.02 bound. One fixed full-layer evaluation either
closes this representation or proceeds directly to a separately identified
native packed owner/consumer and the same real d2/d3 API. No group-size or
quantizer scan follows. Draft savings alone cannot close the 40K budget gap;
any retained dependency must compose with further verifier work. Artifacts
stay under `.q3x-work/mtp-mixed-draft-20261010/`; production is unchanged.

The [completed assessment](metadata/qwen36-27b-mtp-mixed-draft-2026-10-10.json)
authenticates all reused captures, original independent outputs and checkpoint
payloads. K/V pass at 0.008094/0.006036 maximum-row relative L2, but complete
hidden fails at 0.036985, with two of 64 rows above 0.02. Preserving FC and
using grouped scales substantially reduces this representation's error but
does not meet the unchanged full-layer contract. The conditional native
owner/consumer is therefore not implemented; no GPU, acceptance or performance
result follows. This fixed representation is closed without a quantizer or
group-size scan. Retained v31, production and performance observations are
unchanged. The 1.5x–3x objective remains active and unmet; a successor must
address the complete verifier budget with a new bounded architecture and real
API return, rather than treat lower weight error as a delivered improvement.

## 2026-10-10 MTP draft INT8 feasibility — fixed quantizer rejected

`WP-MTP-DRAFT-INT8-FEASIBILITY-20261010` preserves the 1.5x–3x target,
18.45/21.62-second Decode budgets and exact target verification. The
[design](metadata/qwen36-27b-mtp-draft-int8-feasibility-design-2026-10-10.json)
assesses one fixed per-channel INT8 draft representation against the original
independent FP64 full-layer reference on retained real P65 target hidden.
The unchanged 0.02 maximum-row hidden/K/V bound is a feasibility screen;
no runtime arithmetic, accuracy contract or numerical baseline changes here.
The measured 2.06/2.45-second draft cost is only a bounded opportunity and
cannot alone close the 40K gap. One original/quantized host pass either closes
the fixed quantizer or selects a separately bounded native owner/consumer
composition with immediate real API return and acceptance accounting. No
quantizer/group-size scan or GPU process opens. Artifacts stay under
`.q3x-work/mtp-draft-int8-feasibility-20261010/`; production is unchanged.

The [completed assessment](metadata/qwen36-27b-mtp-draft-int8-feasibility-2026-10-10.json)
reproduces the original BF16 draft's independent-reference pass but rejects
this fixed INT8 representation before GPU work. Maximum-row hidden/K/V
relative L2 is 0.19197/0.13003/0.11821 against the unchanged 0.02 bound.
FC has 5,113 of 5,120 row maxima on its hidden-half diagonal; its one-scale
quantization zeros 23.0% of codes and gives 12.77% weight relative L2 error.
Other matrices have 0.93–1.50% weight error. This is a specific representation
risk, not proof that every INT8 method fails or attribution of all output error
to FC. No runtime, numerical contract, GPU process or performance baseline
changes. The 1.5x–3x goal remains active; a future outlier-preserving proposal
requires a new bounded design and real API return, and is not selected by
this host-only closure. No quantizer/group-size scan follows here.

## 2026-10-10 MTP independent accumulation chains — closed negative

`WP-MTP-CHAIN-PARTITION-20261010` / `AC-MTP-GREEDY-v31` keeps the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The
[design](metadata/qwen36-27b-mtp-chain-partition-design-2026-10-10.json)
partitions four independent NVFP4 FMA chains across adjacent lanes while
preserving per-weight reuse across all speculative rows. Unlike closed token
ownership or lookahead variants, it changes ownership of independent scalar
chains and reconstructs the identical original tree after K completes. This
addresses the older 8.845-second NV family ceiling; it is not a projected gain.
FP8 and head stay v31. One fixed mapping, no spill and complete synthetic/P65
state admission precede immediate P65/8K/40K d2 API; greater-than-3% regression
stops, neutral/negative removes, useful direction unlocks d3. At most one
correctness repair is allowed. No chain, grid, decoder, layout or resource scan
follows. A useful dependency expires at the next complete verifier API return
or archival. Artifacts stay under `.q3x-work/mtp-chain-partition-20261010/`;
production remains unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-chain-partition-rejection-2026-10-10.json)
passes the static no-spill gate, 30 complete output/guard cases and full P65
prefix/state/logit/transaction checks. P65/O16 d2 preserves baseline output and
work counts but Decode falls to 9.691 token/s versus v31 11.902, triggering
the declared early stop. Verification adds 287.332 ms. The following 8K
request is interrupted; there is no 40K, d3 or full API lifecycle claim.
All new runtime/profile/auditor paths are removed and v31 rebuilt. Lower
register pressure and exact arithmetic did not select a faster runner. No
chain, ownership, grid or resource scan follows. The 1.5x–3x objective remains
unmet; the next package must remove material complete-verifier work or change
its producer/consumer boundaries, rather than repeat accumulator redistribution.

## 2026-10-10 MTP NV register feed — closed without retention

`WP-MTP-NV-REGISTER-FEED-20261010` / `AC-MTP-GREEDY-v30` retains the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The
[design](metadata/qwen36-27b-mtp-nv-register-feed-design-2026-10-10.json)
selects one raw-record lookahead in NVFP4 Gate/Up and Down, preserving scale
ownership and original arithmetic. This family has a much longer consumption
loop than the closed FP8 lookahead; its existing real-input counter evidence
supports a bounded test, not a transferred performance claim. The old 8.845-second
family attribution is a ceiling only. FP8 and vocabulary execution stay v31.

Fixed launch bounds preserve the incumbent per-role M2/3/4 CTA floors. Any
spill or resource-floor loss closes this implementation before GPU execution;
no resource, launch, decoder or layout scan is admitted. If admitted, complete
synthetic and P65 state/logit checks precede immediate P65/8K/40K d2 API.
Greater-than-3% regression stops; neutral/negative removes; useful direction
unlocks d3. One implementation plus at most one correctness repair is the bound.
A useful result expires at the next complete verifier API return or archival.
Artifacts stay under `.q3x-work/mtp-nv-register-feed-20261010/`; production
is unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-nv-register-feed-rejection-2026-10-10.json)
passes the static no-spill/resource floor, 30 complete synthetic outputs/guards,
full P65 state/logit/transaction checks and all eight d2 API/lifecycle requests.
Work and acceptance match v31. D2 Decode is 12.698/9.527 token/s at 8K/40K,
versus retained 12.914/9.660; external total adds 0.305/0.301 seconds. This is
a negative direction screen, not a statistically qualified regression. All new
runtime/profile/auditor changes are removed and v31 restored. D3 is not run;
no register, launch, feed or resource scan follows. Earlier weight issue and
no spills do not establish whole-runner benefit. The 1.5x–3x goal remains
active and unmet; the next package must address the complete verifier budget
with a materially different execution response and immediate real API return.

## 2026-10-10 MTP softmax lifetime composition — closed without retention

`WP-MTP-SOFTMAX-LIFETIME-20261010` / `AC-MTP-GREEDY-v29` keeps the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The
[design](metadata/qwen36-27b-mtp-softmax-lifetime-design-2026-10-10.json)
selects one complete normalization-lifetime change: batch all speculative
head rows in one grid, compute exponentials once in existing dead score cells,
and retain the exact original denominator tree with warp-local final ancestors.
The older matched softmax attribution is below one second and includes draft;
this is a bounded prerequisite, not a claim to close the full remaining gap.
QK/PV, projections, draft and all state/publication semantics stay v31.

One implementation plus at most one correctness repair passes complete FP32
probability/guard tests and P513/P8192 state/logit admission, then returns
immediately to P65/8K/40K d2 API. Greater-than-3% regression stops; neutral or
negative removes it. Useful direction unlocks d3 and retains only a bounded
dependency through the next complete verifier API return or archival. No local
parameter/timing or profiler campaign opens; artifacts stay under
`.q3x-work/mtp-softmax-lifetime-20261010/`. Production is unchanged.

The [completed rejection](metadata/qwen36-27b-mtp-softmax-lifetime-rejection-2026-10-10.json)
passes 60 full FP32 probability/guard cases, all eighteen P513/P8192 prefix
comparisons, full P513 transactions/recovery and eight d2 API/lifecycle checks.
Work and acceptance match v31. D2 observes 12.984/9.724 token/s at 8K/40K
versus 12.914/9.660: only +0.55%/+0.67% in one process, without noise
separation. The 40K external request is 0.403 seconds longer. This does not
establish a material whole-product gain; all new runtime/profile/auditor paths
are removed and v31 restored. D3 is not run. No further softmax mapping,
exponential-storage or launch scan follows. The 1.5x–3x goal remains active
and unmet; the next package must address a materially larger complete verifier
budget, rather than compound additional sub-percent normalization variants.

## 2026-10-10 MTP NV live reduction composition — API return complete

`WP-MTP-NV-LIVE-REDUCTION-20261010` / `AC-MTP-GREEDY-v28` retains the
1.5x–3x objective and 18.45/21.62-second complete Decode budgets. The
[design](metadata/qwen36-27b-mtp-nv-live-reduction-design-2026-10-10.json)
composes the proved v30 live-ancestor reduction with NVFP4 Gate/Up, Down and
target vocabulary finalization. Longer NV mainloops limit the source-level
opportunity; instruction ratios are not measured latency or a speedup claim.
One fixed reuse of the existing helper, without new grids/layouts/decoders,
returns through complete synthetic and P65 state/logit admission to the real
P65/8K/40K d2 API. Greater-than-3% regression stops; neutral/negative removes
the new mapping, and useful direction unlocks d3. At most one correctness
repair is allowed. A useful result is only a bounded prerequisite through the
next complete verifier API return or archival. Artifacts remain under
`.q3x-work/mtp-nv-live-reduction-20261010/`; production is unchanged.

The [completed direction](metadata/qwen36-27b-mtp-nv-live-reduction-direction-2026-10-10.json)
passes all numerical checks and both complete API panels. D2 observes
12.914/9.660 token/s at 8K/40K versus v30 12.708/9.533; d3 observes
11.310/8.665 versus 11.164/8.574, with identical work and acceptance counts.
The exact mapping remains a bounded development dependency through the next
complete verifier architecture's first API return or archival. No further
mapping/grid scan or qualification-only campaign follows. This is a small
single-process direction, not statistical architecture selection or production
promotion. The 1.5x–3x objective remains active and unmet: even d2 still needs
about 1.30/4.78 seconds less complete Decode at 8K/40K. A successor must name
a materially different complete verifier response and promptly return to the
same real API rather than repeat tail-reduction variants.

## 2026-10-10 MTP causal proposal reuse — assessment complete

`WP-MTP-CAUSAL-REUSE-20261010` keeps the 1.5x–3x objective and the
18.45/21.62-second complete Decode budgets. The [design](metadata/qwen36-27b-mtp-causal-reuse-design-2026-10-10.json)
selects one host-only feasibility audit of repeated committed-history proposals
at fixed draft lengths 2/3. The pinned vLLM default supplies one fixed five-token
match with earliest occurrence. Selection may read only prompt and committed
history; retained future target tokens are comparison-only after selection.
Every possible entry in the existing P65/8K/40K output is assessed. This cannot
simulate unknown MTP proposals at changed fallback positions or establish API
speedup. One audit either closes the direction or selects a separately bounded
hybrid-proposal composition with unchanged exact target verification and prompt
API return. No n-gram/draft-length scan, GPU run or runtime mutation is opened.
Artifacts stay under `.q3x-work/mtp-causal-reuse-20261010/`.

The [completed assessment](metadata/qwen36-27b-mtp-causal-reuse-assessment-2026-10-10.json)
finds zero matched entries at P65, about 54% at P8192 and only 7.1% at P40000.
All 1,035 entry decisions pass an independent earliest-match/causal-bound scan.
This is insufficient evidence to select proposal reuse as the response to the
full target-domain gap; no runtime composition, GPU test or matching-parameter
scan follows. The 8K opportunity remains workload-specific, without a hybrid
schedule or speedup claim. Retained v30 and the 1.5x–3x objective are unchanged;
a successor must address the complete verifier budget and return to the real
40K API, rather than substitute a repetition-rich subset for the goal.

## 2026-10-10 MTP live reduction ownership — API return complete

`WP-MTP-LIVE-REDUCTION-20261010` / `AC-MTP-GREEDY-v27` retains the
1.5x–3x objective and 18.45/21.62-second complete Decode budgets. Weight
expansion was assessed but is not selected: complete decoded FP8 inputs exceed
the borrowed arena and double repeated weight bytes. The [design](metadata/qwen36-27b-mtp-live-reduction-design-2026-10-10.json)
instead removes dead-lane reduction work while preserving every live ordered
add. Four original FP8 output trees migrate into four 8-lane groups; packed
final partial reduction retains both leading zero additions. Packed weights,
mainloops, launch grids and all other v28 mechanisms remain unchanged.

The [MTP ledger](MTP_ADMISSION.md#live-reduction-ownership-composition-v27)
defines exact mapping. One implementation plus at most one correctness repair
passes directed tree/full-output and P65 transaction checks, then returns
immediately to P65/8K/40K d2 API. Greater-than-3% regression stops; neutral
or negative direction removes it. Useful direction unlocks d3 and grants only
bounded prerequisite authority through the next complete verifier API return.
No kernel grid, layout or decoder scan opens. Artifacts remain under
`.q3x-work/mtp-live-reduction-20261010/`; production is unchanged.

The [completed direction](metadata/qwen36-27b-mtp-live-reduction-direction-2026-10-10.json)
passes all directed/full-output and real-model numerical gates plus both
complete API panels. D2 observes 12.708/9.533 token/s at 8K/40K versus v28
12.377/9.337, with identical work and acceptance. D3 improves to 11.164/8.574
but remains slower than d2. The exact reduction mapping remains a bounded
development dependency through the next complete verifier's first API return
or archival. No repeat qualification or reduction/launch sweep follows. The
1.5x–3x objective remains active and unmet; the next architecture still needs
to remove substantial complete verifier cost, especially at 40K. Static
instruction counts neither establish a hardware limit nor project further gains.

## 2026-10-10 MTP register lookahead — closed without retention

`WP-MTP-REGISTER-LOOKAHEAD-20261010` / `AC-MTP-GREEDY-v26` responds to
bounded real-input FP8 long-scoreboard evidence and the recorded current-block
load/consume SASS. The [design](metadata/qwen36-27b-mtp-register-lookahead-design-2026-10-10.json)
keeps the 1.5x–3x goal and 18.45/21.62-second complete Decode budgets. One
fixed K5120/6144 executor issues the next raw weight record into registers
before consuming the current one. It retains current activations, the original
decoder/grid/FMA ownership, and the incumbent register-limited CTA floors.
This changes producer/consumer overlap without the rejected shared packed
ring's stores and per-tile barriers. No new layout, allocation or parameter
scan opens; counter ratios are not projected API gains.

The [MTP ledger](MTP_ADMISSION.md#register-lookahead-composition-v26) binds
finite-precision and tail safety. One implementation plus at most one correctness
repair passes static no-spill, full synthetic and P65 transaction admission,
then returns immediately to P65/8K/40K d2 API versus frozen v28. Greater-than-3%
regression stops; neutral/negative direction removes this executor. Useful
direction unlocks d3 and retains only a bounded dependency through the next
full verifier API return. Artifacts remain under
`.q3x-work/mtp-register-lookahead-20261010/`; production stays non-MTP.

The [completed direction](metadata/qwen36-27b-mtp-register-lookahead-rejection-2026-10-10.json)
passes 42 complete synthetic cases, full P65 numerical/transaction admission
and all eight d2 API/lifecycle requests. Work and acceptance counts match v28.
Decode observes 12.434/9.365 token/s at 8K/40K versus v28 12.377/9.337:
only +0.46%/+0.30% in one process, without measured noise separation. The 40K
external total is 0.222 seconds longer. This provides no useful retention
signal or statistically qualified speedup. All new runtime/profile/auditor
paths are removed, restoring v28; d3 is not run. No register, pipeline, grid
or qualification-only scan follows. The complete 1.5x–3x objective remains
active and unmet. Retained v28 mechanisms have now reached their next API
composition point; they remain development-only. Any successor must identify
a materially different complete verifier response and its API return budget.

## 2026-10-10 MTP projection lifetime composition — API return complete

`WP-MTP-PROJECTION-LIFETIME-20261010` / `AC-MTP-GREEDY-v25` composes
v27 with shared Gate/Up input publication and distributed FP8 finalization.
The [design record](metadata/qwen36-27b-mtp-projection-lifetime-design-2026-10-10.json)
binds the newly available component counters and the unchanged 1.5x–3x goal.
Original grids, packed weights, decoders and FMA ownership stay fixed. Gate
input publication removes three of four repeated global activation traversals
for up to three rows; all eight FP8 warps perform independent final reductions.
This is one coupled execution-lifetime response, not another packed-weight,
register-decoder or launch sweep. Neither counter ratios nor byte reductions
are claimed as API gains. The [MTP ledger](MTP_ADMISSION.md#projection-lifetime-composition-v25)
owns exact arithmetic and lifetime.

One composition plus at most one correctness repair passes complete synthetic
output/guard and real P65 transaction checks, then immediately returns to
P65/8K/40K d2 API against frozen v27. Greater-than-3% regression stops; neutral
or negative direction removes the bundle. Useful direction unlocks d3 and
retains only a bounded dependency through the next full verifier API return.
No allocation, inventory, numerical, reserve or production change is admitted.
Artifacts stay under `.q3x-work/mtp-projection-lifetime-20261010/`.

The [completed direction](metadata/qwen36-27b-mtp-projection-lifetime-direction-2026-10-10.json)
passes all numerical checks and both complete API panels. D2 observes
12.377/9.337 token/s at 8K/40K versus v27 11.984/9.090, with identical work
and acceptance. The exact execution changes remain a bounded development
dependency through the next complete verifier API return. No qualification-only
campaign, input-row/carveout/grid sweep or production switch follows. The
1.5x–3x goal remains active and unmet; the next architecture must address the
remaining complete verifier budget using the now-available bounded counters,
without treating local stall ratios as whole-API savings.

## 2026-10-10 MTP projection counters — complete

`WP-MTP-PROJECTION-COUNTERS-20261010` preserves the 1.5x–3x objective and
18.45/21.62-second complete Decode budgets. The prior full-model NCU attempt
exhausted replay-backup memory. One separate, source-identical component owner
loads only authenticated complete layer-0 QKV/Gate weights and retained real M3
inputs, avoiding that full-model backup footprint. The
[diagnostic design](metadata/qwen36-27b-mtp-projection-counters-design-2026-10-10.json)
permits one unprofiled correctness/timing process and one bounded root NCU
process with unchanged preflight/safety rules. No runtime mutation, local
retention panel, performance baseline or production change is selected by these
two cells. Close collection on success or resource failure, then use its
bounded attribution to select a materially different complete verifier response
and return promptly to the same d2/d3 API. No full-model replay retry or
layout/grid/decoder scan opens. Artifacts remain under
`.q3x-work/mtp-projection-counters-20261010/`.

The [completed diagnostic](metadata/qwen36-27b-mtp-projection-counters-2026-10-10.json)
obtains counters without resource failure. Both real-payload cells reproduce
all 96 captured elements and guards. Long-scoreboard waits dominate the sampled
stall ratios; no spills are observed, and Gate MIO throttle is negligible.
FP8 achieved occupancy is 37.29% versus theoretical 50%, including its
single-warp finalization. These component facts do not establish full-model
stall percentages or a bank-conflict cause. Collection is closed.

## 2026-10-10 MTP batched vocabulary finalization — API return complete

`WP-MTP-BATCHED-HEAD-20261010` / `AC-MTP-GREEDY-v24` composes one
previously coupled dependency with retained v25: exact full-vocabulary target
finalization across all verified rows. The earlier persistent-operand rejection
changed several projection families and did not isolate this dependency. The
matched 40K trace assigns 2.300 GPU seconds to target plus draft heads; even
eliminating that entire budget cannot alone close the unchanged 1.5x–3x target.
The [design record](metadata/qwen36-27b-mtp-batched-head-design-2026-10-10.json)
binds one complete API composition, not a new weight-decoder or layout scan.

All transformer projections remain v25. Complete logits reuse dead projection-0
storage and reach each immutable prefix slot before finite/argmax validation.
Original scalar arithmetic, resource reserve and committed-token accounting
remain mandatory. Full-vocabulary synthetic checks and complete P65 transaction
admission precede immediate P65/8K/40K d2 API; useful direction unlocks d3.
A greater-than-3% regression stops; neutral direction removes this version.
At most one correctness repair is permitted. Any useful result has bounded
prerequisite authority only through the next full verifier API return, never
a production or 50% claim. Artifacts stay in `.q3x-work/mtp-batched-head-20261010/`.

The [completed direction](metadata/qwen36-27b-mtp-batched-head-direction-2026-10-10.json)
passes complete numerical admission and all eight API/lifecycle requests for
both policies. D2 Decode observes 11.984/9.090 token/s at 8K/40K versus v25
11.630/8.874; d3 observes 10.299/8.034 versus 10.026/7.859. Identical acceptance
and verification counts accompany about 0.65–0.71 seconds less verification.
The isolated code retains this as a bounded dependency, not a statistically
selected baseline or production route. No further qualification-only campaign
or head/grid/draft-length scan is pending. The next complete verifier architecture
must compose this dependency at its first API return or archive it. The dominant
projection and long-context Attention budgets still need a materially different
execution response; the 1.5x–3x objective remains active and unmet.

## 2026-10-10 MTP combined immutable decoder — closed negative

`WP-MTP-COMBINED-CODEBOOK-20261010` / `AC-MTP-GREEDY-v23` composes the
ordered draft dependency with a different exact NVFP4 computation boundary.
The matched 40K verifier spends 8.845 GPU seconds in this family. Instead of
repeating two shared lookups and block-scale multiplication per weight, one
construction-owned 4,096-entry table materializes their exact finite-domain
product. Every original FMA and publication remains unchanged. This also removes
per-CTA table construction/barriers, charging read-only cache traffic and index
work to the complete API. It adds only 16,384 immutable device bytes, with no
request-time allocation or model layout change.

The [design record](metadata/qwen36-27b-mtp-combined-codebook-design-2026-10-10.json)
and [equivalence ledger](MTP_ADMISSION.md#combined-immutable-decoder-composition-v23)
bind one composition plus at most one correctness repair. Exhaustive decoder/
full-output guards and complete P65 transaction admission precede immediate
P65/8K/40K d2 API; useful direction unlocks d3. A greater-than-3% regression
against v25 stops early; neutral direction removes this version without a
lookup/layout/grid scan. The unchanged 1.5x–3x goal and 18.45/21.62-second
complete Decode budgets remain controlling. No hardware-stall claim or target
attainment follows from instruction counts. Artifacts stay under
`.q3x-work/mtp-combined-codebook-20261010/`. This is the ordered-draft dependency's
next complete API composition point; negative new decoding returns to the
independently positive v25 direction, without production selection.

The [combined-decoder rejection](metadata/qwen36-27b-mtp-combined-codebook-rejection-2026-10-10.json)
closes this composition after exhaustive decoder/consumer checks and complete
P65 numerical/transaction admission. P65/O16 d2 output and accounting match,
but Decode falls to 9.252 token/s versus v25 10.704. The early-stop gate
interrupts 8K; no 40K/d3 or complete lifecycle claim follows. All new table,
owner and consumer paths are removed. The independently positive v25 wiring
remains in isolated development, and its next API composition point is now
fulfilled. This result closes another weight-decoding substitution; a successor
must address the complete verification execution budget rather than open a
table/cache/layout scan. The 1.5x–3x objective remains active and unmet.

## 2026-10-10 MTP ordered draft composition — API return complete

`WP-MTP-DRAFT-ORDERED-COMPOSITION-20261010` / `AC-MTP-GREEDY-v22`
returns an earlier coupled dependency through the retained v7 target verifier.
The prior losing projection/Attention packages did not select the independent
whole-path effect of exact production ordered draft GQA. This is a complete
service wiring change, without another target kernel variant. The
[design record](metadata/qwen36-27b-mtp-draft-ordered-composition-design-2026-10-10.json)
binds the 3.258-second 40K draft budget and unchanged 18.45/21.62-second
complete Decode budgets. Draft savings alone cannot establish the 1.5x–3x goal.

One composition uses the existing fixed production operator on eligible draft
steps, with explicit scratch/device validation and actual hit receipts. Full
P513 transaction admission precedes immediate P65/8K/40K d2 API/lifecycle;
useful long-context direction unlocks the same d3 panel. Neutral/negative
composition is removed. A positive result is only a bounded prerequisite for
the next complete verifier architecture's API return, never a production or
50% claim. No local kernel/layout scan opens. Artifacts stay in
`.q3x-work/mtp-draft-ordered-composition-20261010/`.

The [completed direction](metadata/qwen36-27b-mtp-draft-ordered-composition-direction-2026-10-10.json)
passes the new same-input original-draft comparison, complete P513 numerical/
transaction admission and all eight API/lifecycle requests for each policy.
D2 observes 11.630/8.874 token/s at 8K/40K versus v7 11.560/8.636; d3 observes
10.026/7.859 versus 9.954/7.624. Target verification stays effectively unchanged.
The private wiring remains as a bounded dependency for the next complete
verifier composition, not a statistically selected baseline or production
route. No further qualification-only campaign or draft-length scan is pending.
The unchanged 1.5x–3x goal still selects the dominant verifier budget; the next
architecture must compose this dependency at its first real API return or
archive it. The negative layout, shared-residency and certificate-executor
lineages remain closed.

## 2026-10-10 MTP phase-borrowed FP8 input layout — closed without retention

`WP-MTP-PHASE-FP8-LAYOUT-20261010` / `AC-MTP-GREEDY-v21` preserves the
1.5x–3x goal and 18.45/21.62-second complete Decode budgets. The earlier
startup-owned layout exchanged O/Down acceleration for input layouts; its
neutral result did not isolate input benefit. This composition instead borrows
the dead Prefill family arena after draft initialization and prompt-capture
expiry, retaining the complete original acceleration inventory. Its
[design record](metadata/qwen36-27b-mtp-phase-fp8-layout-design-2026-10-10.json)
and [lifetime ledger](MTP_ADMISSION.md#phase-borrowed-fp8-input-layout-composition-v21)
bind 5,200,936,960 packed bytes within 5,591,000,064 available bytes excluding
snapshots. Preparation and independent byte checking occur once per request,
are included in complete Decode, and receive a separate receipt. This changes
cross-phase ownership rather than a local layout or launch parameter.

One composition plus at most one correctness repair returns through packing,
range/lifetime, full P65 transaction and P8192 prefix admission immediately to
P65/8K/40K d2 API; d3 follows useful d2 direction. P65 reports the new fixed preparation cost but cannot alone reject its O256
amortization. A greater-than-3% 8K/40K regression stops early; neutral complete
long-context direction closes the version. No allocation, canonical-weight mutation, inventory removal,
reserve reduction or production switch is admitted. Artifacts remain under
`.q3x-work/mtp-phase-fp8-layout-20261010/`. The measured 5.454-second input
projection budget alone cannot guarantee the full target; only the complete
API can select this prerequisite.

The [phase-layout rejection](metadata/qwen36-27b-mtp-phase-fp8-layout-rejection-2026-10-10.json)
closes this composition after complete P65/P8192 numerical admission and all
eight d2 API/lifecycle requests. Preparation costs about 159 ms per request;
8K/40K verification is effectively unchanged from v7. Decode is 11.477/8.587
versus retained 11.560/8.636 token/s. The startup reserve passes with the full
inventory, but no useful API gain emerges. All new runtime/test/auditor paths
are removed. No d3 qualification, profile or layout/ownership scan follows.
The 1.5x–3x objective remains unmet; this closes the proposed cross-phase FP8
layout prerequisite rather than retaining it indefinitely. A successor needs
a different complete verifier computation/execution response, with the exact
publication boundary and prompt API return preserved.

## 2026-10-10 MTP compact full-K residency — closed negative

`WP-MTP-COMPACT-RESIDENCY-20261010` / `AC-MTP-GREEDY-v20` targets the
unchanged 1.5x–3x objective and 18.45/21.62-second complete Decode budgets.
The matched 40K profile assigns 13.321 GPU seconds to FP8 plus Gate/Up
verification. One complete composition stages compressed full-K records once
per CTA, then uses independent original consumer groups for each speculative
row. This changes both weight lifetime and accumulator ownership across both
families. It avoids the rejected full-K BF16 expansion and per-tile FP32
barriers, but charges repeated shared decode and large-block residency to the
API. Down, Attention and draft execution remain retained v7.

The [design record](metadata/qwen36-27b-mtp-compact-residency-design-2026-10-10.json)
and [MTP ledger](MTP_ADMISSION.md#compact-full-k-residency-composition-v20)
bind one fixed implementation plus at most one correctness repair. T1 full
output/guard checks, P65 full transactions and P8192 complete prefixes precede
immediate P65/8K/40K d2/d3 API return. Neutral/negative direction closes this
version without a shape/grid/stage scan; d3 follows only useful d2 direction.
No new allocation, inventory exchange or reserve reduction is allowed.
Artifacts remain under `.q3x-work/mtp-compact-residency-20261010/`.

The [compact-residency rejection](metadata/qwen36-27b-mtp-compact-residency-rejection-2026-10-10.json)
closes this version after all 42 synthetic output/guard cases and complete
P65/P8192 numerical admission. P65/O16 d2 output and accounting match, but
Decode falls to 7.459 token/s versus retained v7 10.717; verification rises
to 1.854 seconds. The early-stop gate interrupts the following 8K API request.
All new runtime paths are removed, with no 40K/d3 or full lifecycle claim.
Compact residency and lower registers do not select a faster runner. No further
shared-residency/row-ownership variant is active. The 1.5x–3x goal remains
unmet; a successor needs a different complete execution/computation response
and an immediate real-API return, preserving the exact publication contract.

## 2026-10-10 MTP startup-owned FP8 input layout — closed without retention

`WP-MTP-FP8-INPUT-LAYOUT-20261010` / `AC-MTP-GREEDY-v19` retains the
1.5x–3x objective and 18.45/21.62-second complete Decode budgets. The matched
40K trace assigns 5.454 GPU seconds to canonical FP8 input verification.
One startup-owned compact quad layout removes four scalar weight loads and
four index swizzles per K iteration in favor of the existing exact vector
consumer. It does not expand weight bytes. To retain the 8-GiB reserve, the
composition replaces FP8 O and NVFP4 Down auxiliary layouts with canonical
consumers, retaining NVFP4 Gate/Up. Only the complete API selects this trade.
The [design record](metadata/qwen36-27b-mtp-fp8-input-layout-design-2026-10-10.json)
and [MTP ledger](MTP_ADMISSION.md#startup-owned-fp8-input-layout-composition-v19)
bind its private ownership and exact arithmetic. The 5,200,936,960-byte owner
replaces 4,596,613,120 bytes, adding 604,323,840 bytes. No canonical payload or
Prefill operator is changed. A full BF16 expansion is not selected because it
doubles the repeated input-weight traffic.

One implementation plus at most one correctness repair passes exhaustive
pack/consumer T1, P65 full transactions and P8192 complete prefixes, then
returns directly to P65/8K/40K d2/d3 API. Negative direction closes this version
without a layout/grid scan. No local timing campaign or production switch is
opened. Artifacts stay under `.q3x-work/mtp-fp8-input-layout-20261010/`.

The [FP8 input-layout rejection](metadata/qwen36-27b-mtp-fp8-input-layout-rejection-2026-10-10.json)
closes this complete composition after exhaustive packing/consumer checks,
complete P65/P8192 numerical admission and all eight d2 API/lifecycle requests.
D2 Decode is 11.581 at 8K and 8.642 at 40K versus retained v7 11.560/8.636:
these single-process differences provide no useful gain or qualified speedup.
Startup retains 9,666,056,192 free bytes, but adds 604,323,840 bytes of inventory.
All new runtime/build/auditor paths are removed. No d3 qualification, layout
scan or additional profile follows. The 1.5x–3x objective remains active and
unmet; a successor must address the complete verifier execution budget rather
than repeat an isolated weight-layout substitution.

## 2026-10-10 MTP startup-owned pair layout — closed negative

`WP-MTP-PAIR-LAYOUT-20261010` / `AC-MTP-GREEDY-v18` addresses the unchanged
1.5x–3x objective and 18.45/21.62-second complete Decode budgets. The matched
40K profile assigns 8.845 GPU seconds to NVFP4 verification. The earlier pair
consumer still used quad-packed records: its warp's 256 useful weight bytes
span 16 aligned 32-byte regions. A pair-packed producer makes these eight
regions. This is an address fact, not measured DRAM traffic or a stall claim;
cache reuse may already serve the old consumers. The
[design audit](metadata/qwen36-27b-mtp-pair-layout-design-2026-10-10.json)
proves the lossless Gate/Up permutation and binds the source identities.

One composition replaces the old Gate/Up, Down consumer and scale6 startup
sidecars with one private MTP pair-packed NVFP4 owner, then binds exact M2..4
pair consumers. Canonical weights and scalar arithmetic remain intact; FP8
stays v7. The new owner is 9,625,927,680 bytes, replacing 9,000,632,320 bytes,
so actual composed startup must retain the unchanged 8-GiB reserve after the
625,295,360-byte increase. The private layout is never attached under an old
public layout identity. M1 and the scalar oracle consume canonical NVFP4.

The [MTP ledger](MTP_ADMISSION.md#startup-owned-pair-layout-composition-v18)
controls lifecycle and proof. One implementation plus at most one correctness
repair includes startup identity, full payload permutation checking and the
consumer before full P65 transaction/P8192 prefix checks and immediate
P65/8K/40K d2/d3 API. No local timing or launch/layout scan precedes that return.
A neutral/negative API composition closes the version. This is a bounded
attempt on the dominant projection budget, not a guaranteed target claim.
Artifacts stay under `.q3x-work/mtp-pair-layout-20261010/`.
The [pair-layout rejection](metadata/qwen36-27b-mtp-pair-layout-rejection-2026-10-10.json) closes the complete owner/inventory/consumer composition. Both full-size
T1 layouts/consumers, full P65 transactions and all P8192 prefixes pass. P65/O16
d2 API output and accounting match, but Decode is 10.361 token/s versus retained
v7 10.717, triggering the predeclared direction stop. P8192 API is interrupted;
no 40K/d3 or complete lifecycle claim follows. Startup retains 9,728,929,792
free bytes, so resource fit is not the rejection reason. All new runtime paths,
including the previously committed pack/check prerequisite, are removed.
The 1.5x–3x goal remains active and unmet. This closes the pair-layout version
without a layout/grid scan; the next package must change the complete verifier
execution architecture and return promptly to the same real API.

## 2026-10-10 MTP exact PV work elimination — closed without retention

`WP-MTP-EXACT-PV-ELISION-20261010` / `AC-MTP-GREEDY-v17` retains the
1.5x–3x objective and 18.45/21.62-second full Decode budgets. The matched
40K profile assigns 2.318 GPU seconds to ordered verifier PV. Shared KV
movement did not improve the runner. This materially different computation
eliminates an entire ordered block only when a runtime bound proves that every
FMA leaves its current FP32 accumulator unchanged. It changes no reduction,
probability, state or output rounding. The maximum measured PV budget alone
cannot close the full target; this is a bounded composed prerequisite, not a
redefined speedup goal. Projection, QK, draft and all other paths stay v7.

The [MTP ledger](MTP_ADMISSION.md#exact-pv-work-elimination-composition-v17)
controls the proof, exception handling and conservative cache-bound lifetime.
One fixed implementation plus at most one correctness repair passes directed
rounding/exception checks, full P513 transactions and P8192 complete prefixes,
then immediately returns to P65/8K/40K d2/d3 API. If the real-prefix checks
prove zero eliminated groups, close the mechanism without a timing campaign;
otherwise the normal negative API stop applies. No threshold, tile, numerical
bound or launch scan is opened. Persistent bounds borrow less than 177 KiB of
otherwise dead post-Prefill storage; the 8-GiB reserve stays fixed. Every API
receipt counts tested and eliminated PV groups. Artifacts remain under
`.q3x-work/mtp-exact-pv-elision-20261010/`.

The [exact-PV rejection](metadata/qwen36-27b-mtp-exact-pv-elision-rejection-2026-10-10.json) closes this composition after the one
initialization-cancellation repair, complete P513/P8192 numerical checks and
all eight d2 API/lifecycle requests. D2 Decode is 11.502 at 8K and 8.522 at
40K versus retained v7 11.560/8.636, with negligible actual elision. These
single-process differences establish no qualified regression or gain. All new
runtime paths are removed; no d3 qualification or bound/tile scan follows.
The 1.5x–3x goal remains active and unmet. A successor must address the dominant
complete verifier execution budget with a materially different architecture;
this closure selects no further local variant.

## 2026-10-10 MTP shared multi-query KV — closed negative

`WP-MTP-SHARED-KV-20261010` / `AC-MTP-GREEDY-v16` returns to retained
v7 projections after the certificate lineage's negative API results. The same
1.5x–3x goal and 18.45/21.62-second complete Decode budgets remain controlling.
The matched 40K profile assigns 3.077 GPU seconds to QK and 2.318 to PV.
Current interleaved CTAs enable cache reuse but load each speculative row's KV
separately. This composition physically stages K and V once for all rows.
QK partitions three heads per warp to bound query registers while all row/head
groups share a causal K tile. PV uses 32-dimension ownership and all-row
accumulators, keeping 32 CTAs rather than the old shared-V experiment's 16.
These choices follow the complete shared-KV resource/ownership contract;
there is no launch or dimension sweep. Ordered production draft GQA composes
at the same return. The original projection, recurrence and logit paths remain.

One fixed implementation plus at most one correctness repair passes full P513
prefix/transaction and P8192 complete-prefix admission, then returns immediately
to the same P65/8K/40K d2/d3 API panel. Short P65 cannot select a long-Attention
win; it only guards unaffected execution. Negative long-context direction
closes the composition. No new allocation, sidecar, mask workspace or reserve
change is introduced. The [MTP ledger](MTP_ADMISSION.md#shared-multi-query-kv-composition-v16)
controls exact arithmetic and causal tails. Artifacts remain under
`.q3x-work/mtp-shared-kv-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-shared-kv-rejection-2026-10-10.json) closes this version after full P513 and P8192
numerical admission. D2 completes all three context requests with identical
output/accounting, but Decode is 11.422 at 8K and 8.296 at 40K versus retained
v7 11.560/8.636. New shared-KV and ordered draft paths are removed. No d3,
qualification-only continuation or KV tile/ownership scan follows. The exact
causal transformation is valid but does not select a faster runner. The
1.5x–3x objective remains active and unmet; subsequent work must address a
materially different complete verifier execution architecture rather than
reopen the rejected packed-feed, certificate-executor or shared-KV variants.

## 2026-10-10 MTP partitioned execution and compact repair — closed negative

`WP-MTP-PARTITION-REPAIR-20261010` / `AC-MTP-GREEDY-v15` responds to
v17's matched 1.697-second fast execution plus 0.737-second repair costs. The
unchanged 1.5x–3x objective and 18.45/21.62-second full Decode budgets select
a complete two-stage scheduling change. Independent K128 fragment producers
replace each full-K serial CTA; a fixed balanced merge certifies each output
and compacts only uncertified elements into a bounded device worklist. Fixed
persistent repair workers consume that list, preparing codebooks once instead
of launching the entire possible output grid. The original scalar tree and
all certificate/exception rules remain unchanged. This composes the measured
fast and repair dependencies, rather than reopening a tile/grid scan.

One fixed composition plus at most one correctness repair passes exhaustive
pair/boundary and complete P65 state/logit checks, then returns immediately to
P65/8K/40K d2/d3 API in fail-fast order. Negative direction closes the version.
The partials, mask, activation preparation and worst-case worklist fit a 64-MiB
post-Prefill borrow; no new allocation, host count readback, reserve relaxation
or production change occurs. Ordered draft GQA composes at the same return.
The [MTP ledger](MTP_ADMISSION.md#partitioned-execution-and-compact-repair-composition-v15)
controls proof and lifetime. Artifacts stay in
`.q3x-work/mtp-partition-repair-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-partition-repair-rejection-2026-10-10.json)
closes this composition: exhaustive conversion, complete worklist/output and
real P65 state/logit/transaction checks pass, but Decode is 4.702 token/s
against retained v7 10.717. All new paths are removed, with no long-context/d3
qualification or further certificate/executor/worklist variant. The 1.5x–3x
objective remains unmet. Subsequent work returns to the retained projection
architecture and the measured multi-query Attention budget; rejected projection
compositions did not independently select or reject an Attention dependency.

## 2026-10-10 MTP direct fragment delivery — closed negative

`WP-MTP-REGISTER-CERTIFICATE-20261010` / `AC-MTP-GREEDY-v14` retains
1.5x–3x committed Decode and 18.45/21.62-second complete Decode budgets.
The rejected streamed executor still expands every weight into shared storage,
then reads it back into matrix fragments. This successor replaces that entire
operand path: one projection-wide guarded activation conversion publishes a
bounded FP16 array; warps load canonical packed weights directly into the
native MMA B-fragment coordinates and convert pairs in registers. No decoded
weight tile, codebook lookup or per-K producer/consumer barrier remains.
The pinned Marlin bit-field/pair conversion supplies the proven representation
principle, with full bias restoration before multiplication. The certificate,
zero-start K16 tree and exact sparse repair retain their numerical contract.
This source-derived movement/instruction hypothesis is not a hardware-stall claim.

One fixed implementation plus at most one correctness repair passes pair/code
and complete P65 state/logit admission, then returns immediately to the same
P65/8K/40K d2/d3 API in fail-fast order. Negative direction closes this version
without a launch, shape or bound scan. Ordered draft GQA composes at that return.
Activation/mask storage fits the existing 1-MiB post-Prefill borrow; no weight
arena, reserve reduction or production switch is introduced. The detailed
[MTP ledger](MTP_ADMISSION.md#direct-fragment-delivery-composition-v14) controls
ownership. Artifacts stay under `.q3x-work/mtp-register-certificate-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-register-certificate-rejection-2026-10-10.json)
closes this composition after complete numerical admission: P65 d2 Decode is
4.901 token/s versus retained v7 10.717. All new paths are removed. One bounded
same-ELF profile assigns 1.697 GPU seconds to direct fragments and 0.737 to
exact repair; input preparation is only 0.008 seconds. This is family-level
attribution, not a hardware-stall diagnosis. No long-context/d3 qualification or
feed/launch parameter scan follows. Any successor must address both the full-K
executor and repair scheduling, retaining the same exact publication proof
and prompt API return. The 1.5x–3x goal remains unmet.

## 2026-10-10 MTP streamed certified execution — closed negative

`WP-MTP-STREAMED-CERTIFICATE-20261010` / `AC-MTP-GREEDY-v13` responds
to the measured 4.875-second fast-generator cost in the rejected P65 composition.
The unchanged 1.5x–3x goal and 18.45/21.62-second full Decode budgets select
whole-K CTA ownership, vector packed-weight delivery and on-chip bounded-depth
reduction. The composition eliminates all global signed/absolute/error partials
and the separate certificate kernel. Native m16n8k16 fragments retain only the
observable first eight rows; a thread-owned shared binary stack limits the
arithmetic depth without excessive register retention. Vector repair consumes
four FP8 or eight NVFP4 codes per load, preserving the original scalar tree.
This replaces the complete producer/publication path, not its tile parameters.

The preceding certificate proof and exception rules remain unchanged; the
[MTP ledger](MTP_ADMISSION.md#streamed-certified-execution-composition-v13)
binds the new ownership. One fixed composition plus at most one correctness
repair returns through boundary and complete P65 state/logit checks directly to
P65/8K/40K d2/d3 API. Negative direction closes this version without a shape,
register, pipeline or certificate-bound scan. Ordered draft GQA composes at the
same return. Only a 1-MiB borrowed mask workspace remains. The 8-GiB reserve,
production default and numerical contract are unchanged. Artifacts stay in
`.q3x-work/mtp-streamed-certificate-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-streamed-certificate-rejection-2026-10-10.json)
closes this version: complete numerical/transaction gates pass, but P65/O16
Decode is 3.956 token/s against retained v7 10.717. All new runtime paths are
removed. There is no 8K/40K/d3 result or follow-on tile, pipeline or bound scan.
Eliminating global partials and reducing workspace did not select a faster
runner. The 1.5x–3x objective remains active; a successor must replace the
complete packed-to-fragment feed/compute architecture, account for repair,
and return promptly to the same API. No new implementation is selected here.

## 2026-10-10 MTP bounded reduction with sparse repair — closed negative

`WP-MTP-CERTIFIED-SPARSE-20261010` / `AC-MTP-GREEDY-v12` targets the
unchanged 1.5x–3x API objective and 18.45/21.62-second full Decode budgets.
The measured FP8 plus Gate/Up verification cost selects a new computation,
not another packed-feed/ownership variant. Native Tensor Core K16 reductions
start from zero, combine by explicit balanced FP32 trees within K128 partitions,
and publish signed, absolute-product and input-conversion-error partials.
An outward-rounded per-element certificate proves the original scalar BF16
publication; every uncertified element executes the original scalar tree.
Down retains its exact incumbent because its longer scalar chain makes this
certificate less useful. This differs from the closed coarse-norm feasibility:
it computes data-dependent absolute-product bounds, limits fast reduction depth,
and repairs individual elements rather than rejecting a whole M-row group.
No approximate target value is permitted to escape the certificate/repair pair.

The [MTP ledger](MTP_ADMISSION.md#bounded-reduction-and-sparse-repair-composition-v12)
owns the proof, exception handling and scratch lifetime. One implementation
plus at most one correctness repair must pass full P65 state/logit/transaction
admission and return immediately to the same P65/8K/40K d2/d3 API panel. Negative
API direction closes this version without a tile or certificate-parameter scan.
The ordered draft GQA dependency composes at that return. Scratch borrows at
most 64 MiB of dead post-Prefill GDN storage after snapshots and prompt capture;
no new allocation, smaller reserve or production switch is authorized.
Artifacts remain under `.q3x-work/mtp-certified-sparse-20261010/`.

The [API rejection](metadata/qwen36-27b-mtp-certified-sparse-rejection-2026-10-10.json)
closes this version: numerical gates pass, but P65 Decode is 2.283 token/s
versus retained v7 10.717. One same-ELF P65 profile attributes 4.875 GPU seconds
to fast partial generation, 0.178 to certificates and 0.927 to exact repairs.
The fast executor is the dominant measured cost; this is not proof that exact
certification itself prevents acceleration. All new runtime paths are removed.
No 8K/40K/d3 qualification or tile/bound scan follows. A successor must replace
the complete fast operand/execution path and account for sparse repair costs;
the 1.5x–3x objective and full-round budget remain unchanged.

## 2026-10-10 MTP asynchronous packed feed — closed negative

`WP-MTP-ASYNC-PACKED-FEED-20261010` / `AC-MTP-GREEDY-v11` addresses the
same 1.5x–3x objective and 18.45/21.62-second full Decode budgets. The measured
16.29-second verifier projection cost selects overlapping packed-weight movement
with the existing exact register-local M-row consumption. Unlike rejected
expanded shared-weight tiles or token-ownership variants, a three-buffer
cp.async pipeline retains FP8/NVFP4 bytes unchanged, prefetches two future K
blocks, and keeps the original four FMA chains and output ownership. It
transfers the proven Marlin/production Attention asynchronous producer-consumer
lifetime to this private SIMT verifier without importing MMA reassociation.
No hardware-stall cause is asserted from the unavailable counters.

One fixed composition, including the admitted ordered draft GQA dependency,
returns through full P65 state/logit/transaction checks to P65/8K/40K d2/d3 API.
At most one correctness repair is permitted; negative API direction closes
this version without buffer/grid/tile scans. Shared packed storage is 12,288
bytes per block, no new arena or runtime allocation. Artifacts stay under
`.q3x-work/mtp-async-packed-20261010/`. The 8-GiB reserve and production
non-MTP path remain unchanged.

The [API rejection](metadata/qwen36-27b-mtp-async-packed-rejection-2026-10-10.json)
closes the composition: complete P65 numerical checks pass, but P65 Decode is
10.308 token/s versus retained v7 10.717. New runtime paths are removed; no
8K/40K/d3 qualification or pipeline-parameter scan follows. Retained v7 remains
the development incumbent. The next implementation must address the complete
verification budget with a different execution architecture, preserving exact
publication and the 1.5x–3x target; this closed package authorizes no variant.

## 2026-10-10 MTP independent token CTAs — closed negative

`WP-MTP-INDEPENDENT-CTAS-20261010` / `AC-MTP-GREEDY-v10` retains the
1.5x–3x objective and 18.45/21.62-second full 8K/40K Decode budgets. The
16.29-second projection observation selects independent token CTAs with
M-fast output-tile scheduling: adjacent blocks consume the same packed weight
span through shared L2, while each thread retains only one token's original
four FP32 chains. No token-group barrier or shuffle broadcast is introduced.
This is distinct from the rejected v3 CTA-row implementation, which forces
M*256 FP8 or M*128 NVFP4 threads into each block with shared codebooks and
multi-token reduction storage. Here the unchanged 256/128-thread blocks can
reside and retire independently, trading duplicated cached reads/conversion
for fewer registers and independently schedulable consumers. L2 reuse is a
hypothesis, not a measured counter claim. Packed weight layout and decoder
remain unchanged; there is no grid/tile sweep. Ordered draft Attention composes
at this immediate API return with dependency-only authority.

One fixed composition plus at most one correctness repair must pass complete
P65 scalar prefix state/logits and transactions, then return to P65/8K/40K d2/d3
API in fail-fast order. Negative direction closes this version. No new arena,
resource-margin reduction, approximate output or production change is allowed.
Artifacts stay under `.q3x-work/mtp-independent-ctas-20261010/`.
The [API rejection](metadata/qwen36-27b-mtp-independent-ctas-rejection-2026-10-10.json)
records P65 Decode 7.509 token/s versus retained v7 10.717. All numerical and
recovery checks pass, but the composition is removed. No long-context or d3
qualification follows. Independent-CTA, cooperative-CTA and per-thread row
ownership have now all been compared; another ownership/cache/grid variation
is not the next package. The overall 1.5x–3x goal remains active and unmet.
A successor requires a different complete execution/data-representation
architecture and a traceable exact-publication proof before implementation.


## 2026-10-10 MTP certified publication — feasibility closed

`WP-MTP-CERTIFIED-PUBLICATION-20261010` addresses the same 1.5x–3x API goal
and measured 16.29-second projection cost. Repeated packed-feed variants are
closed. The alternative is a fast reduction followed by a rigorous rounding
certificate and exact scalar repair of uncertified outputs. A certificate must
prove the identical BF16 publication, including scale and reduction error;
statistical error or matching argmax is insufficient. Recurrent state and all
observable target outputs remain exact. This does not authorize approximate
production arithmetic or a weaker numerical baseline.

Before implementing a new executor, one whole-core real-prompt M3 capture at
P65 samples 32 output rows per projection in layers 0/3/20/23/40/43/60/63.
The numerical-only harness records canonical weights/scales, real activations,
exact BF16 outputs and identities. A host error-bound audit measures the
optimistic certification and required repair fractions, both element-wise and
for shared M-row output ownership. This is a bounded feasibility prerequisite,
not a local performance screen. It may reject an uneconomic certificate but
cannot select a runner or claim speedup. At most one capture and one audit
precede either closure or a fully specified executor/repair composition with
an immediate d2/d3 real API return. No quantized-decoder or launch sweep opens.
Artifacts remain in `.q3x-work/mtp-certified-publication-20261010/`.
The [bounded audit](metadata/qwen36-27b-mtp-certified-publication-feasibility-2026-10-10.json)
finds that a cheap norm certificate would repair at least one row in 46.4%
of FP8, 59.2% of Gate/Up and 97.7% of Down sampled output groups. Even an
optimistic exact center cannot remove most Down repairs with this bound.
The coarse-bound whole-verifier composition is closed before GPU executor
implementation; no tighter-bound or parameter sweep follows under this package.
The capture seam is frozen numerical-only evidence and was removed after this
completed gate; it was never part of serving.


## 2026-10-10 MTP persistent operands — closed negative

`WP-MTP-PERSISTENT-OPERANDS-20261010` / `AC-MTP-GREEDY-v9` uses the
[complete 40K attribution](metadata/qwen36-27b-mtp-bottleneck-reset-2026-10-10.json):
16.29 GPU seconds in verifier projections, 5.40 in QK/PV, and 2.30 in combined
target/draft lm-head after the first verifier. No valid NCU hardware-stall
counters were obtained, and that collection is closed. The selected composition
retains packed global weights and register-local M reuse, but makes activation
and lookup storage persist across output groups. It also batches target lm-head
weights across all verified rows and reuses ordered production draft Attention.
This follows the production lm-head's proven persistent activation lifetime;
it is not another expanded-weight tile or register-decoder variant. Main-model
FMA trees/state/logits remain exact. One fixed composition plus at most one
correctness repair returns to complete prefix admission and P65/8K/40K d2/d3.
The 1.5x–3x objective and 18.45/21.62-second full Decode budgets remain fixed;
negative API direction closes the version without tile, grid or draft scans.

The [API rejection](metadata/qwen36-27b-mtp-persistent-operands-rejection-2026-10-10.json)
closes this version: complete scalar-prefix state/logit checks pass, but P65
Decode falls from retained v7's 10.717 to 9.536 token/s. The new runtime paths
are removed; long-context/d3 qualification and grid/tile scans do not follow.
The overall MTP goal remains active and unmet. A successor must change the
complete verifier execution architecture rather than reopen another packed
weight-decoder, output-group or persistence variation. It must first reconcile
its numerical contract and expected whole-round budget against the measured
16.29-second projection cost, then implement one composition and return to the
same real API. No new kernel variant is selected by this closure.

## 2026-10-09 MTP bottleneck reset — complete

`WP-MTP-BOTTLENECK-RESET-20261009` restores the retained v7 composition after
full-K residency regressed the real P65 API. No losing projection or draft
route remains selected. One bounded same-ELF P40000/O256 d2 Nsight attempt
will distinguish projection execution, long-context Attention and controller
cost on the complete matched real workload. The existing fresh v7 API record
is its timing authority; profile timings cannot replace it. Startup reserve
and clean ownership stay mandatory; a resource failure stops that attempt.
The completed trace attributes 16.29 seconds to verifier projections and
5.40 seconds to verifier QK/PV (softmax separate). The bounded NCU collection produced no hardware counters: filtering and
permission failures were followed by replay-backup memory exhaustion under
root. It is closed without reducing the reserve or claiming a measured stall
cause. The subsequent persistent composition above also returned negative.
The 1.5x–3x goal remains active.

## 2026-10-09 MTP full-K operand residency — closed negative

`WP-MTP-FULLK-RESIDENCY-20261009` / `AC-MTP-GREEDY-v8` follows the
packed-conversion composition's negative P65/8K direction. Repeated scalar
weight conversion changes did not close the verifier budget. This fixed
architecture instead decodes a complete FP8 output group into CTA-owned BF16
once, publishes it once, then lets independent token groups consume the whole
K span without an inner producer barrier. Unlike rejected shared-FP32 tiles,
there is no per-tile synchronization or expanded FP32 operand traffic. BF16
holds every FP8 code exactly. Output ownership is four channels for K5120 and
two for K6144, solely to fit the static shared-memory limit. All original
four-chain, warp/eight-warp reductions and BF16 publication remain exact.
NVFP4 stays incumbent. Production ordered draft Attention remains dependency-only
until this composition's API result. One implementation and at most one
correctness repair return to full-prefix admission and P65/8K/40K d2/d3;
negative direction closes it without a launch/shape sweep. The same 1.5x–3x
goal and 15.7/17.9-second verifier budgets select this work.

## 2026-10-09 MTP packed FP8 conversion — negative direction

`WP-MTP-PACKED-FP8-20261009` / `AC-MTP-GREEDY-v7` restores the v7
vector/output-quad and interleaved-query implementation after v9 regressed all
three d2 API buckets. Shared-V and output-pair ownership are archived, with
no d3 qualification-only continuation. The next fixed composition transfers
the pinned Marlin pairwise FP8-to-FP16 conversion to exact FP32 verifier
operands and reuses production ordered GQA for draft S512..44095. It changes
neither target FMA trees nor draft outputs. Exhaustive 256-code/all-pair device
checks precede complete real-prefix state/logit admission, then immediate
P65/8K/40K d2/d3 API return. One composition plus one correctness repair is
the bound; no parameter scan or numerical waiver is opened. The unchanged
1.5x–3x goal and full-round budgets remain the selection constraint.

## 2026-10-09 MTP accumulator ownership — closed negative

`WP-MTP-ACCUMULATOR-OWNERSHIP-20261009` / `AC-MTP-GREEDY-v6` follows the
negative P65 API result of warp-broadcast verification. The v8 projection is
removed; its exact shared-V implementation is a dependency only, bounded to
this immediate composition. Retain register-local reuse across all M rows,
but partition the existing four-channel packed sidecar into two aligned
output pairs for M3/M4. This reduces per-thread accumulator liveness without
multiplying decode/broadcast instructions across token subgroups. M2 retains
the existing four-output route. This is one fixed ownership transformation,
not a tile/launch sweep. It composes with the exact shared-V kernel and admitted
draft MMA before the next P65/8K/40K d2/d3 API return. The 15.7/17.9-second
verifier budgets and 1.5x–3x target remain unchanged. One implementation plus
at most one correctness repair bounds this package; no third weight-decoder
variant or repeated qualification of a losing policy is opened.

## 2026-10-09 MTP cooperative verifier — projection rejected

`WP-MTP-COOPERATIVE-VERIFY-20261009` / `AC-MTP-GREEDY-v5` follows the
completed d2/d3 API return of the draft matrix path.
[MTP Admission](MTP_ADMISSION.md#cooperative-verifier-composition-v5) defines
warp-local packed-weight broadcast and actual shared-V query consumption.
Exactly these two dependencies compose before the next API direction screen;
no separate local-timing campaign or parameter scan is opened. The unchanged
1.5x–3x product constraint selects a 15.7/17.9-second 8K/40K verifier budget.
One composition plus at most one correctness repair is the bound; complete
scalar state/logit identity, causal masks and the reserve remain mandatory.

## 2026-10-09 MTP draft matrix execution — API return complete

`WP-MTP-DRAFT-MMA-20261009` / `AC-MTP-GREEDY-v4` addresses the same
1.5x–3x owner target. The v4 verifier is restored; numerically admitted causal
query scheduling and post-Prefill workspace borrowing compose with one native
BF16 draft matrix executor. The measured 2.83/13.80-second 8K/40K draft
initialization and 3.41/4.54-second proposal costs select this dependency.
The complete Decode budgets remain 18.45/21.62 seconds, so a draft-only win
cannot close the verifier gap. The bounded composition is one fixed CUTLASS
M1..32 executor, independent draft-oracle admission, then the existing
P65/8K/40K d2/d3 API panel. No kernel-parameter scan is opened. Target state
and logits remain bitwise scalar-equivalent; only the private draft reduction
gets a separate numerical identity, subject to the existing independent
FP64 0.02 relative-L2 bound and exact within-route cache replay.

## 2026-10-09 MTP shared verification — closed negative

`WP-MTP-SHARED-VERIFY-20261009` / `AC-MTP-GREEDY-v3` implements the next
bounded composition described in [MTP Admission](MTP_ADMISSION.md#shared-verification-composition-v3).
The originating owner constraint remains 1.5x–3x committed Decode versus the
same-request ordinary production service. The previous 8K/40K full-Decode
budgets of 18.43/21.61 seconds select shared decoded projection operands and
causal multi-query Attention, with unchanged exact state/logit semantics.
One initial composition and at most one causally justified repair must return
to P65/8K/40K d2/d3 API; no launch-parameter or draft-length sweep is opened.
Both weight-delivery implementations regressed API Decode and are removed.
No third projection repair is permitted by this package. This is isolated
development, not a production switch. Prefill initialization
remains an explicitly measured unresolved cost; this package first returns the
new verifier to the API before selecting its next prompt-side dependency.

## 2026-09-28 MTP — active, configured drafts 2 and 3

`WP-MTP-EFFICIENCY-20260928` / `AC-MTP-GREEDY-v2` has completed its bounded
implementation and API return, recorded in the
[efficiency direction](metadata/qwen36-27b-mtp-efficiency-direction-2026-09-28.json).
Vector-feed verification, M32 initialization and cache-only reconciliation
remain in the isolated v4 admission; the single CTA-row repair is rejected and
removed. Both drafts pass the declared numerical/API panel, but neither
provides the owner's 1.5x–3x Decode objective across contexts. No production
switch, draft-length sweep or qualification-only campaign follows this result.

The MTP efficiency goal remains active. The downward budget now requires
8K d2's entire Decode to fit about 18.43 seconds, versus 19.63 seconds in
verification alone; at 40K the full d2 budget is about 21.61 seconds versus
26.91 seconds in verification alone. This selects a materially new verifier
operand/Attention dataflow and prompt/draft work elimination, not another
launch-parameter variant of the rejected CTA design. Any successor must name
its exact arithmetic/state/lifetime ledger, bounded composition point and
return to the same P65/8K/40K d2/d3 API panel, reporting Prefill, TTFT, committed
Decode and external total time. The numerical contract and 8-GiB reserve remain
hard constraints; the production service remains the speedup denominator.

The earlier owner authorization opened `WP-MTP-20260928` and
`AC-MTP-GREEDY-v1`, governed by [MTP Admission](MTP_ADMISSION.md).
Non-MTP optimization packages remain closed and their targets remain intact.
The host foundation and the [native scalar device milestone](metadata/qwen36-27b-mtp-device-2026-09-28.json)
are delivered. Authenticated draft execution, shifted scalar hidden/KV
initialization, complete recurrent prefix restoration and target-conditioned
draft reconciliation pass the bounded real-model correctness harness. That scalar
milestone executed the main model serially and had no API or acceleration claim.

The whole-core prompt handoff now captures all final normalized hidden rows
and initializes shifted draft KV in the non-installable engine harness. Full
logits are included in prefix restoration. The
[handoff record](metadata/qwen36-27b-mtp-whole-core-2026-09-28.json) bounds these
checks. The [batched initialization milestone](metadata/qwen36-27b-mtp-batch-prefill-2026-09-28.json)
now closes cache-only FC/K/V batching and bounded initialization cancellation
on the real P65 state panel. The [multi-row verifier milestone](metadata/qwen36-27b-mtp-multirow-2026-09-28.json)
now closes exact FP8/NVFP4 weight reuse and complete immutable prefix snapshots.
P61/P65/P509 pass every M2/M3/M4 prefix plus rejection, cancellation and recovery
against full scalar state/logits, including the S64 and S512 Attention switches.
The generic small-M reduction mismatch is fixed; no additional local verifier
scan is pending.

The [service composition](metadata/qwen36-27b-mtp-service-direction-2026-09-28.json)
now completes that API return: P65/O16 and P8192/P40000 O256 run for both draft
lengths, with matched responses and bounded lifecycle/route checks. Both
policies worsen Prefill and Decode relative to the ordinary service. This
closes the performance version of `AC-MTP-GREEDY-v1` without selection; no
additional v1 qualification or draft-length sweep is pending. The isolated
controller, full-capacity admission and truthful physical witness are retained
as development infrastructure, not enabled in production.

The efficiency package above supersedes v1's next-step proposal. Its one
CTA-row operand repair is rejected after a valid P65/P8192 slowdown; it is not
an outstanding tuning branch. The final composition uses stage-one vector
projection feed with M32 initialization and cache-only reconciliation. Both
configured draft lengths have returned to the complete API direction panel,
closing this bounded package. Numerical admission does not authorize production or a
50% speedup claim. A remaining gap must select a materially new verifier or
Attention/draft dataflow with a downward whole-round budget; repeating these
projection variants, qualifying a losing policy or scanning draft lengths is
not the next task.

## 2026-09-28 publication fusion — closed without promotion

`WP-PUBLICATION-FUSION-20260928` implemented both selected epilogues in
`AC-PREFILL-PUBLICATION-FUSION-v1`: Up/SiLU and FP8 O/residual, preserving
the GEMM mainloops and intermediate BF16 boundaries. Resource admission and
32 real-weight numerical cases passed, but the composed API direction was
mixed: P8192 Prefill improved 0.56%, while P40000 took 0.95% longer; Decode
was effectively unchanged. This single-process comparison does not establish
a statistically qualified regression or a promotable gain. Under the
predeclared negative-composition stop, the version is closed. The
[direction record](analysis/publication-fusion-direction-2026-09-28/README.md)
retains the reproducible patch, outputs, hashes and scope. Candidate integration
was removed, production is unchanged, and no qualification-only campaign or
tile scan remains active. FP8 preparation reuse remains a separate unimplemented
opportunity; this closed package reopened neither Down nor MTP. The later
owner-authorized MTP package above is separate.

## 2026-09-28 alternative-path source assessment — complete

The owner's follow-up to examine other paths is recorded in the
[alternative assessment](analysis/alternative-paths-2026-09-28/README.md).
It recommends preserving current GEMM mainloops while fusing Up/SiLU and
FP8 O/residual publication, with exact intermediate BF16 rounding; FP8
per-phase decoded-weight reuse is the next separate opportunity. It does not
claim measured gains, activate implementation, reopen Down, or alter production.
The report defines the bounded composition proposal for a subsequent package.
Decode's approximately 10 ms/step interim gap remains open; existing KV sharing
and rejected PV variants must not be presented as new opportunities.

## 2026-09-28 bounded non-MTP architecture work — closed

The owner-authorized `WP-NON-MTP-ARCHITECTURE-20260928` assessment and its
`WP-DOWN-ORDERED-20260928` implementation are complete. The
[assessment](analysis/non-mtp-architecture-assessment-2026-09-28/README.md)
selected Down's 17.56-second profiled Prefill budget. The
[implementation record](analysis/ordered-down-direction-2026-09-28/README.md)
closes both permitted dataflows: first external FP32 partials, then a corrected
block-local two-partition merge. Both preserved the bounded numerical panel
and compared API outputs, but regressed real 8K and 40K Prefill. Neither is
retained for production or future composition.

Candidate runner/build/profile branches have been removed; frozen reproducible
patches and all failures remain in evidence. The default 0.8.1 artifact,
numerical contract, capacity and paired performance baseline are unchanged.
No third variant, local tuning, extra profile or qualification-only campaign
is authorized by this closed package. It grants no MTP execution authority. A materially new
architecture requires a new bounded product-connected package; the deferred
product targets below are neither lowered nor claimed achieved.

## 2026-09-28 product audit — performance optimization paused

The owner paused performance optimization for this now-completed product
audit. The subsequent bounded architecture package above has also closed;
the earlier milestone list does not authorize unrelated tuning. The 8.55 interim
and long-term targets remain recorded, with no waiver or achievement claim.

The [product audit](analysis/product-readiness-audit-2026-09-28/README.md)
identifies Unicode output-cap handling, HTTP control-plane availability,
zero-test release validation, error-to-readiness propagation, reproducible
qualification tools, install notices and bounded sustained-service coverage.
The owner has authorized one delivery batch, `WP-SERVICE-INDUSTRIALIZATION-20260928`.
Its originating constraints are correct text serialization, available control
plane, fail-closed runtime health, reproducible validation and deployable
packaging. These select bounded UTF-8 carry, staged incomplete connections,
fatal-error health latching, production host tests, a versioned API driver and
installed notices/supervision templates. Value returns at the same installed
0.8.1 service through the unified protocol in
[EVALSCOPE_EVALUATION](EVALSCOPE_EVALUATION.md#service-industrialization-validation).
The batch is complete: six host/package CTests, 36 Python tests, multilingual/
slow-client/cancel/capacity checks, the pinned 98-question panel, two hours /
1,708 bounded reuse requests, fresh-process recovery and complete route/cleanup
checks passed. The [delivery record](metadata/qwen36-27b-service-industrialization-2026-09-28.json)
closes this package; no further qualification-only loop is active. It does not
reopen kernel optimization or change model arithmetic. The one-second reuse
cadence and 60-second capacity spacing do not qualify continuous maximum-load
saturation; the earlier thermal stop remains recorded as a deployment limit.

## 2026-09-27 product convergence — delivered scope and retained gaps

The owner's later numerical-repair and independent-baseline direction closes
`WP-PREFILL-REFERENCE-REPAIR-20260928` and the bounded service integration of
`AC-WHOLE-CORE-SERVICE-20260927`. Corrected whole-core Prefill plus ordered v7
Decode is installed and selected by the ordinary release preset. No further
Prefill numerical variant or qualification-only loop is open. The
[production record](metadata/qwen36-27b-whole-core-service-production-2026-09-28.json)
is the exact completion boundary.

The reference decision is explicit: checkpoint-faithful independent FP32 GDN
and same-input FP64 Decode checks assess the reference itself. Legacy remains
a regression comparator rather than universal truth. Complete prompt-boundary
state/logits, public capability and real installed API behavior qualify the
corrected service. Do not revive the invalid error denominator or compare
free-running states after token histories diverge.

A retained performance gap is long-context Decode: current correct
40K service is below the owner's approximately 8.55 token/s interim target.
Retain that target and report Prefill and Decode together. The four-chain PV
arithmetic variant and 64-CTA spatial-ownership variant are both closed for
slower real-payload performance. They are not pending tests and must not be
reopened as another local parameter scan.

Any next Decode optimization needs a new whole-runner bottleneck observation,
a named accuracy-preserving architecture change, a downward latency budget,
and a bounded return to the actual P40000/O256 production API. Preserve the
selected Prefill contract, BF16 KV, no-MTP scope and independent reference
checks. A component improvement without a positive composed API result does
not select production. A negative bounded composition closes its own version.

## Deferred product milestones

1. Resolve the remaining 40K Decode gap without numerical regression; retain
   the long-term 10 token/s target separately from the interim 8.55 target.
2. Expand capacity to the declared 60K and approximately 130K workloads through
   a new state/memory plan and complete real API witness. Current 44,095-state
   service qualification does not extend by extrapolation.
3. Resolve the locked 40K–60K 2-second and approximately 130K 4-second TTFT
   targets, starting from matched request-bound vLLM evidence and a complete
   architecture/dataflow budget. Do not infer impossibility from the present
   implementation or a dense-BF16 proxy.
4. Complete full product release qualification across those capacities/SLOs,
   longer-running stability and the required broader capability coverage.
   Preserve exact installed artifact, numerical, route, resource, dependency,
   package and public API attestation.
5. Basic service-supervision templates are delivered. Add further startup/AOT
   or deployment improvements only when selected by an
   actual deployment constraint. Additional models, batching and media
   remain later separately authorized product scopes. MTP is now separately
   authorized in the active package above.

These milestones do not authorize an open-ended experiment campaign. Each
new package must identify the product constraint, finite-precision/state
contract, one bounded implementation/composition point and real API exit.

## 2026-09-09 bounded engineering window

This window is closed and has no current execution authority. Its accepted
terminal-prefix and Graph ownership work, rejected score-feed and other
lineages remain documented in immutable metadata and Git history. The
[legacy roadmap](ROADMAP_LEGACY.md) is historical only. Old inbound links to
this anchor do not reactivate a package or make its profile the default.

## Evidence and stop discipline

Use the smallest safe product-connected check for a reversible iteration.
Freeze independent-process mirrored comparisons for architecture selection
and installed-artifact promotion. Stop collecting evidence once the declared
engineering question is answered. All performance work uses real prompts and
weights, clean GPU ownership, recorded preflight/cache preparation and the
[performance policy](REAL_MODEL_PERFORMANCE_POLICY.md). Preserve raw rejected
runs and source/binary identity; do not rewrite history as a pass.
