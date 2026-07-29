# P513 NVFP4 persistent Marlin rejection (2026-07-29)

## Decision

Reject this test-only exact-C512 Gate+Up persistent Marlin architecture from the production path. The best revision remained slower on the first real-model P513 gate, so EvalScope 8 and all larger external matrices were deliberately not run.

The comparison used the authenticated Qwen3.6-27B-NVFP4 weights and the complete one-token generation path. Native GDN C64/WY was enabled on both sides; only the Gate+Up Marlin admission differed. The generated result remained token `9419`, text `Hello`, with all 64 dense-layer candidate route hits.

## Real-model progression

| Revision | Registers | Fixed grid | Baseline prefix | Candidate prefix | Result |
|---|---:|---:|---:|---:|---:|
| v1 | 176/thread | 16 CTAs | 2260.534531 ms | 2809.022615 ms | +548.488084 ms (+24.263645%) |
| v2 | 128/thread | 16 CTAs | 2072.841305 ms | 2281.591426 ms | +208.750121 ms (+10.070724%) |
| final | 128/thread | 32 CTAs | 2073.508802 ms | 2145.321072 ms | +71.812270 ms (+3.463321%) |

Final TTFT regressed from 2182.391545 ms to 2254.100070 ms (+71.708525 ms, +3.285777%). Moving from 16 to 32 CTAs recovered most of the loss, but did not cross the required positive real-run direction gate.

The v1 NSys run measured prefix time at 2081.186151 ms baseline and 2630.813251 ms candidate. The persistent pair kernel averaged 17.380041 ms per layer; the incumbent native branch kernel averaged 4.371922 ms per launch, or 8.743844 ms for two branches. Thus the pair cell was about 1.987689x the two incumbent branch launches before considering the unchanged standalone SiLU operation. The report is `/tmp/q3x-marlin-v1-p513-20260729.nsys-rep` (SHA-256 `544987fa0e87c46527aff650574474f666ff042276eb1e20b7e90eed92aa1f82`).

## Architecture and cost

The final kernel used a 256-thread M64N256K64 cell, four global-to-shared stages, two shared-to-register weight slots, a logical Gate+Up work queue, and a fixed 32-CTA launch. Each CTA processed only one branch tile at a time. Its measured resource contract was 128 registers/thread, 512 B static shared memory, 73,728 B dynamic shared memory, zero local/spill bytes, and two active CTAs/SM.

The packed sidecar retained the 4-bit payload and one-byte E4M3 scales. It occupied 50,135,040 B per projection and 6,417,285,120 B for Gate+Up across 64 layers. Sidecar preparation cost about 2263 ms in the direct load run; a separate profile observed 1387.91 ms. This startup and memory cost was accepted for screening but gains no production justification after the runtime rejection.

cuBLASLt remained a reference only and never entered the candidate or production route. MTP was not used.

## Stop rule

This experiment used the direct real-model P513 run as a development stop-loss and found a negative direction before consuming the external gate. Project-level admission is decided by the shortest EvalScope perf command against the frozen vLLM reference; only a passing candidate proceeds to repeated confirmation, longer outputs, concurrency, or capability matrices. Synthetic matrices retain correctness/smoke value but have no performance authority here.

The 16-to-32-CTA recovery is useful evidence, but register pressure and persistent-grid tuning alone did not make this skeleton competitive. Any revisit must be a structurally different dataflow, not another production admission of this candidate.
