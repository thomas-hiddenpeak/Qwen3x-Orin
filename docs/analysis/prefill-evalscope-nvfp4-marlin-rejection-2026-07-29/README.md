# EvalScope NVFP4 fused Gate+Up architecture rejection

Status: **rejected by the first external whole-product direction gate**;
test-only archive, never a native incumbent or production route.

This experiment tested one complete exact-C512 NVFP4 Gate+Up dataflow rather
than another isolated tile edit. The candidate compiled only with
`BUILD_TESTING=ON` and `Q3X_BUILD_NVFP4_PREFILL_MARLIN_ADMISSION=ON`, remained
disabled by default, and used the same `qwen3x-eval-server` ELF as its native
baseline. `Q3X_RUN_NVFP4_PREFILL_MARLIN_ADMISSION=1` was the only measured
route difference.

The decision was made first with EvalScope 1.9.1 through the public
`/v1/completions` path. The later NSys comparison only explains the rejection;
it does not override it. NCU was stopped before a valid report existed and no
NCU claim follows.

## External decision

One warmup plus 32 measured requests used the pinned false-thinking ShareGPT
corpus, exact token-ID prompts, 16 greedy output tokens, concurrency one, fixed
Orin clocks, and the real Qwen3.6-27B-NVFP4 checkpoint. A fresh stock-vLLM run
used the matched configuration as the market reference. It did not participate
in the incremental candidate decision.

| Metric | Native baseline | Native candidate | stock vLLM |
| --- | ---: | ---: | ---: |
| Success | 32/32 | 32/32 | 32/32 |
| Mean TTFT | 3,169.170 ms | 3,186.292 ms | 1,147.281 ms |
| p50 TTFT | 3,326.889 ms | 3,326.606 ms | 1,149.568 ms |
| p99 TTFT | 6,664.325 ms | 6,750.515 ms | 2,623.852 ms |
| Mean request latency | 4.801299 s | 4.814527 s | 2.711837 s |
| Mean TPOT | 108.809 ms | 108.549 ms | 104.304 ms |
| Prompt throughput | 102.8433 tok/s | 102.5610 tok/s | 182.0818 tok/s |
| Total throughput | 106.1755 tok/s | 105.8841 tok/s | 187.9814 tok/s |

Candidate versus native baseline:

- mean TTFT regressed **17.121883 ms / 0.540264%**;
- p99 TTFT regressed **86.189769 ms**;
- prompt and total throughput regressed **0.274436%**;
- generated text and token-chunk boundaries remained exact for **32/32**
  requests.

The length-paired result isolates the route itself:

| Prompt tokens | Requests | Candidate − baseline mean TTFT | Slower requests |
| --- | ---: | ---: | ---: |
| 1–128 | 4 | -1.212117 ms | 1/4 |
| 129–512 | 13 | -1.615645 ms | 2/13 |
| 513–1,024 | 13 | +32.109445 ms | 13/13 |
| 1,025+ | 2 | +78.164666 ms | 2/2 |

The route needs one complete C512 tile. The two shorter buckets therefore
measure ordinary cross-process noise. Every request in the first hit bucket
regressed, and the two prompts with two complete C512 tiles regressed more.
This is sufficient early-stop evidence without a multi-round microbenchmark.

The fresh native baseline remains **2.762332x** slower than stock vLLM in mean
TTFT and reaches only **56.481912%** of its prompt throughput. Native and vLLM
generated identical text for 25/32 requests; vLLM is not an accuracy oracle.

## Candidate dataflow

The engine prepacked real Gate and Up weights for all 64 MLP layers into
fragment-native sidecars:

- 55,705,600 bytes per projection;
- 128 projections;
- 7,130,316,800 bytes total (about 6.64 GiB).

The fused kernel used a fixed 32-CTA persistent grid, 256 threads, four Gate
warps plus four Up warps, M64xN128xK64 ownership, a two-stage global-to-shared
pipeline, two register slots, shared A, and an in-kernel BF16
`SiLU(Gate) * Up` epilogue. Static resource inspection reported 128 registers
per thread, no stack/local memory, and 71,680 bytes dynamic shared memory,
allowing two CTAs per SM on SM87.

The production build, installed runtime, selector, Decode path, and default
Prefill dispatch never expose this route. Engine construction fails closed if
an explicitly requested test sidecar cannot be completely prepared; a selected
kernel launch cannot silently fall back.

## Bounded mechanism explanation

An NSys whole-request comparison after the negative direction screen reported:

| Path | Calls represented | Mean kernel time |
| --- | ---: | ---: |
| Fused Gate+Up+SiLU candidate | 128 | 10.036363 ms/layer |
| Native Gate or Up branch | 256 | 4.365762 ms/branch/layer |
| Native standalone SiLU | 128 | 0.670960 ms/layer |

The native pair plus epilogue is about 9.402484 ms/layer. The candidate is
about 0.633879 ms/layer slower, or 40.568 ms over 64 layers. This accounts for
the observed roughly 38.25 ms mean regression on requests containing a single
complete C512 tile. The coupled branch fusion lost more branch-pipeline
efficiency than it recovered through A sharing and one fewer launch.

This result closes this exact 32-CTA, two-branch, M64xN128xK64 skeleton. It does
not claim that every raw-operand NVFP4 large-M kernel or every Gate/Up scheduling
topology is impossible.

## Evidence integrity

- Corpus SHA-256:
  `bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`
- Native baseline DB:
  `c3b227e90beea9eda1bca43784a3c34fa9e577db75177f54650614796af704d3`
- Native candidate DB:
  `baf9b9140d46b6f04b63d3cb52cd5802e6a226073e5ee6bf8d751f9f302393ef`
- stock-vLLM DB:
  `2db1ff4c2156432ad466dd1e00bea70482beb0f90225b2efbe95a5e18b93ad41`
- Measured native ELF:
  `53fd18c085e6df3cf6ead204709dc0773e1eeb78b4be25412e5cf1fe3fc25051`
- Validated triplet summary:
  `6f6e3086acec1f3ff154ae8760560d0e9cbc4b8e6a9b11e6b95fa3b04b0def90`
- NSys report:
  `4d2c2d1ff2c7647b04be54c025f26429ed43a01bb9a02fe8f34cc4d8aff8a23c`

All three SQLite databases passed `PRAGMA integrity_check`; all contained the
same ordered 32 unique manifest requests, 32 successful fixed-length
completions, 16 non-empty choice chunks and 15 ITL samples per request, and one
final `finish_reason=length`.

EvalScope 1.9.1 does not retain the final `choices: []` usage SSE event in
`benchmark_data.db`. Usage-event conformance must therefore remain a separate
raw-SSE protocol test; this database audit does not claim to observe it.

## Decision and next step

Archive this test-only experiment on a rejected branch and return development
to the clean native baseline. The next whole-system candidate is the existing
Chunk64/WY GDN architecture proof: first expose that already-measured route to
the evaluation worker and run a native baseline/candidate pair. The stock-vLLM
result above is frozen; it is rerun only when the cumulative native runner
approaches that floor or when the protocol, model, or hardware state changes.
Only a positive
native-baseline TTFT and prompt-throughput direction unlocks capability,
numerical, NSys, and NCU work. Classic cuBLAS in that proof remains an external
architecture oracle with no production eligibility; a positive result would
authorize the native SM87 closed-cell replacement, not production admission of
the reference path.

## Limitations

- This is one independent process per system, not mirrored multi-round release
  evidence.
- Temperature was not captured, so the result has directional authority only.
- EvalScope databases contain full reversible prompts/responses and remain
  outside the repository; only non-reversible hashes and aggregate metrics are
  retained here.
- No valid NCU report exists for this candidate.
- No public capability score follows from this performance run.
