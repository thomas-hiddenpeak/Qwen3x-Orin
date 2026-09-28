---
q3x_document:
  id: q3x-publication-fusion-direction-20260928
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: bounded publication fusion implementation and mixed API direction
  effective: 2026-09-28
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: materially new implementation requires a successor package
---

# Publication fusion: closed without promotion

The owner authorized trying the path in the
[alternative assessment](../alternative-paths-2026-09-28/README.md).
`WP-PUBLICATION-FUSION-20260928` implemented Up/SiLU and FP8 O/residual
epilogues as one isolated composition against source `99e10db`. It passed
bounded numerical and resource admission, but did not show a positive P40000
API direction. This version is closed; integration was removed and production
remains unchanged. No tile sweep, further variant or qualification-only loop
is pending. FP8 preparation reuse was not part of this experiment.

The [metadata](../../metadata/qwen36-27b-publication-fusion-direction-2026-09-28.json)
binds source/binary hashes, inputs, preflights, numerical checks and API
records. The [self-contained patch](candidate.patch) reproduces all nine
modified/new implementation files against `99e10db`; an isolated-index check
matched every reconstructed file to its original source hash. Raw evidence is
under `.q3x-work/publication-fusion-20260928/`.

## Mechanism and admission

The product constraint was lower Prefill/TTFT without numerical or Decode
regression. The selected downward contract removed intermediate publication
while retaining the existing CUTLASS M128/N256/K64 mainloop, M64/N64/K64 warps,
three stages, swizzle 2 and a single K partition:

- Gate retains its existing BF16 output. Up's custom output operator invokes
  the incumbent no-source scale/conversion operator first, then reads Gate
  through the C-source iterator and performs the same FP32 SiLU/multiply and
  final BF16 conversion. Up's intermediate output and standalone SiLU launch
  are omitted on the request-owned candidate path.
- FP8 O uses the same scaled BF16 branch conversion, then adds the BF16
  residual in FP32 and rounds again. C and D point to the same residual range;
  each element has one output owner. The raw branch and separate residual
  launch are omitted. Other FP8 roles and Decode are unchanged.
- Existing request scratch, protected token-ID prefix, input/output spans,
  stream order, panel scheduling, actual-row tails and arena remain unchanged.
  No request allocation, persistent decoded cache or new reduction is added.

The proposed deletions are 178.258 GB of logical Up write/read bytes and
52.429 GB of logical O branch bytes per P40000 request. These are not measured
DRAM savings. SiLU arithmetic and Gate loads move into the Up kernel; residual
work moves into O. Eliminating standalone kernels does not eliminate their
useful work or guarantee faster GEMMs.

CUDA 13.3/SM87 compilation reports 242 registers/thread for both incumbent
and fused GEMM specializations, zero stack and zero spill loads/stores. The
256-thread launch fits the register budget. This rules out the prior Down
launch-resource failure, but does not prove identical instruction scheduling
or epilogue cost. No NCU/NSys causal profile was collected for this version.

Before running any timing, Gate layers 0/59/8/23 and O layers 1/38/28/43 were
selected from sorted all-layer mean decoded block-scale magnitude (Gate) and
absolute FP8 weight magnitude (O), at ranks 0/21/42/63. Both Gate and Up
checkpoint weights and independent tensor scales were loaded for each MLP case.
At M1/65/129/8001, all 32 comparisons passed: **738,557,952 BF16 elements were
bitwise identical**, with no nonfinite outputs and all checked trailing guards
or untouched residual tails intact. Inputs were deterministic BF16 fixtures,
so this is `checkpoint_weight_only` numerical admission, not real activation
performance evidence or complete model-state qualification.

## Real API direction

B1 is installed 0.8.1, SHA-256
`5ee029cc8877ad0086e5e6ea5bcf8d7674626e55cd96098d518252d33c14aa33`.
C1 is the isolated non-installable admission, SHA-256
`75acacd1d454ede55cb996da9c7dcb7d54cef7bb083f4e1c296316c5ea35a0ef`,
profile/plan `q3x.sm87.admission.whole-core-publication-fusion.v1`, with its
own `target-prefill-witness-whole-core-publication-fusion-v1` identity.

Each fresh process ran the same real P65/P8192/P40000 prompt prefixes, O16/256/256,
greedy generation, no Prefix/KV reuse and no MTP. Engine Prefill excludes
finalization/HTTP; external TTFT is POST to first visible token; Decode is
`(O-1)/engine_decode_interval`. These are **single-process B1-C1 direction
observations**, not mirrored selection or a replacement published baseline.

| Prompt / output | Route | Prefill seconds | Prefill token/s | External TTFT seconds | Decode token/s |
| --- | --- | ---: | ---: | ---: | ---: |
| 65 / 16 | B1 main | 0.880078 | 73.857 | 0.904987 | 9.58448 |
| 65 / 16 | C1 fusion | 0.887923 | 73.205 | 0.906512 | 9.59070 |
| 8,192 / 256 | B1 main | 16.724625 | 489.817 | 16.746625 | 9.21841 |
| 8,192 / 256 | C1 fusion | 16.631790 | 492.551 | 16.656647 | 9.22243 |
| 40,000 / 256 | B1 main | 90.016316 | 444.364 | 90.051576 | 7.86672 |
| 40,000 / 256 | C1 fusion | 90.870475 | 440.187 | 90.906066 | 7.86673 |

Prefill duration changes are +0.89%, -0.56% and +0.95%, respectively. Complete
request bodies, generated text, usage, finish reasons and SSE termination match
between B1/C1 at all three lengths. All six request witnesses pass exact token
identity, complete route/physical projection-entry counts and zero fallback.
Those counts are projection entries, not a count of the now-fused CUDA kernels.
Cold-cache preparation, GPU ownership and fixed 1300.5-MHz clock checks pass;
both servers exit zero and remain below the 90C stop. Exact temperatures are in
metadata. Closeout confirms no owned server or GPU workload remains.

## Decision and limitations

The first direction decision used two API processes, within the declared
30-minute real-model budget. There was no correctness/resource repair or
parameter scan. The small 8K gain does not establish a long-context win;
P40000 is neutral/negative for this screening decision. This is enough to
decline further investment in this composition, **not** enough to claim a
statistically established 0.95% regression, a hardware ceiling, or that every
output-fusion implementation must fail.

No independent epilogue timing was used to select either member. The result
therefore does not isolate which fusion caused the mixed direction. Equal
register counts and logical byte deletion do not supply that missing causal
evidence. No member is retained as a proven optimization.

Under the predeclared stop rule, complete state/logit capture, all-capacity
service qualification, capability repetition and mirrored release comparisons
were not opened. Bounded numerical agreement and matching API text are not
represented as those unperformed gates. The current correct production route,
accuracy qualification, capacity, MTP exclusion and owner performance targets
remain unchanged.
