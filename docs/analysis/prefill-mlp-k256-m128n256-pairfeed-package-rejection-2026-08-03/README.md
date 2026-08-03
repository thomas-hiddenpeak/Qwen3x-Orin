# Prefill MLP K256 M128N256 pair-feed package rejection

Date: 2026-08-03

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill

Performance authority: real checkpoint, OpenAI-compatible API, external
EvalScope

## Verdict

The K256 M128N256 pair-feed MLP package is rejected. The production route
remains unchanged. The candidate stays default-off and is retained only as a
reproducible negative architectural record; it is not a production-path
replacement.

The direction gate used one warm-up request followed by one measured P1853
request. The measured request succeeded (1/1), executed exactly 64 package
launches, and reported zero launches for every old MLP route. Neither MTP nor
cuBLASLt was used.

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| Locked production baseline | 1,916.24 ms | 967.4983 tok/s | 1,911.30 ms |
| K256 package candidate | 2,194.29 ms | 844.9046 tok/s | 2,189.47 ms |
| Candidate change | +278.05 ms (+14.51%) | -122.5937 tok/s (-12.67%) | +278.17 ms (+14.55%) |

This is a decisive endpoint regression, not a borderline statistical result.
The real API result therefore closes the candidate before any broader
performance matrix or local tuning sweep.

## Request-scoped NSys attribution

The candidate was profiled once after the endpoint rejection. Profiler timing
is diagnostic only; the external EvalScope request above remains the
performance verdict.

| GPU family | Locked production | K256 candidate | Change |
|---|---:|---:|---:|
| Gate+Up, 64 calls | 722.001952 ms | 1,004.869536 ms | +282.867584 ms (+39.18%) |
| Down, 64 calls | 288.796640 ms | 345.232704 ms | +56.436064 ms (+19.54%) |
| Activation quantization | 48.717568 ms | 41.580032 ms / 192 calls | -7.137536 ms (-14.65%) |
| Gate+Up + Down + quantization | 1,059.516160 ms | 1,391.682272 ms | +332.166112 ms (+31.35%) |
| All GPU kernels | 1,911.120448 ms | 2,242.465600 ms | +331.345152 ms (+17.34%) |

The production quantization total is the sum of 28.538464 ms of K256 work and
20.179104 ms of K512 work. The candidate's quantization saving is real, but it
is overwhelmed by the Gate and Down regressions. The combined MLP-plane
increase also accounts for essentially the entire request-scoped GPU increase.

## Resource gate outcome

Both kernels met their compile-time resource gates, so resource admission does
not rescue the dataflow:

| Kernel | Registers/thread | Spill | Dynamic shared memory | Residency |
|---|---:|---:|---:|---:|
| Gate+Up K256 M128N256 pair-feed | 128 | 0 | 132,096 B | 1 CTA/SM |
| Down K256 M128N128 pair-ring | 128 | 0 | 133,120 B | 1 CTA/SM |

## Architectural conclusion

The same-panel cooperative N-major schedule did not offset its structural
costs. Duplicated A delivery remained, 15 M tiles left one CTA permanently
idle, and 68 grid barriers forced slowest-CTA lockstep. Moving the full MLP
contract from K512 to K256 also doubled the scale cadence. Together these
costs made both Gate and Down materially slower despite the lower aggregate
quantization time.

Do not run cache-policy, pipeline-stage, tile-size, or CTA-order micro-scans on
this skeleton. A successor must change the cross-CTA ownership and
synchronization structure rather than locally tune this implementation.

## External evidence

The following paths are machine-local result artifacts under
`/home/rm01/q3x-results`; they are not repository files and are not implied to
be part of this documentation commit:

- [Measured EvalScope/API result root](/home/rm01/q3x-results/k256-m128n256-pairfeed-package-p1853-run3)
- [Profile trigger result root](/home/rm01/q3x-results/k256-m128n256-pairfeed-package-p1853-profile-run1)
- [Request-scoped NSys report](/home/rm01/q3x-results/k256-m128n256-pairfeed-package-p1853-nsys-run1.nsys-rep)
- [Exported NSys SQLite database](/home/rm01/q3x-results/k256-m128n256-pairfeed-package-p1853-nsys-run1.sqlite)
