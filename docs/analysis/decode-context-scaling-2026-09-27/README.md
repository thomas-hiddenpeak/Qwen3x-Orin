---
q3x_document:
  id: q3x-decode-context-scaling-investigation-20260927
  class: evidence
  status: frozen
  owner: runtime-maintainers
  authority: bounded source audit and real-API research direction; no production selection
  effective: 2026-09-27
  last_reviewed: 2026-09-27
  supersedes: []
  superseded_by: []
  ssot_for: decode context-scaling investigation at d6d5afb
  review_trigger: immutable record; correct through a successor or explicit erratum
---

# Decode context scaling: source audit and isolated fused-GQA experiment

The owner provisionally accepts Prefill and asks why this hybrid model falls
from about 10 to about 5 Decode tokens/s as context grows, before discussing
product convergence. This record preserves that priority and the existing
accuracy, single-request, non-MTP and weight-format constraints. It does not
amend a target or promote a numerical route.

The investigation starts at clean `d6d5afb912c4de1cbf68ec36a811088b07e4e5f0`.
Recent commits improved the existing scalar implementation: `23e3b0a` and
`216df21` deepen the ordered value reduction's unroll; `5f625d8` and
`158d04e` add optional split values and scores; `d6d5afb` measures the split
route's output divergence. These experiments do not cover fused Tensor Core
attention. Their observed gains remain valid within their original scope.

## Observed API result

Replacing only the scalar full-attention call in an isolated native executable
raises the observed P40000 rate from 5.18 to 8.56 tokens/s. This is positive
research direction, not numerical acceptance, a fully environment-matched
performance claim, or a release. The 10 tokens/s target remains open.

| Prompt tokens / output 32 | Original ms/token | Fused ms/token | Original tokens/s | Fused tokens/s | UTF-8 text equal |
| --- | ---: | ---: | ---: | ---: | --- |
| 1,089 | 106.3054 | 103.7825 | 9.4069 | 9.6355 | yes |
| 8,192 | 121.5121 | 106.1812 | 8.2296 | 9.4179 | yes |
| 40,000 | 192.9516 | 116.8582 | 5.1826 | 8.5574 | no |

The long-minus-short excess falls from 86.6461 to 13.0757 ms/token in these
observations (84.91% smaller). The P40 rate ratio is 1.6512. All six requests
return 32 content events, exact prompt/completion usage, `finish_reason=length`
and DONE. External and server TPOT agree to below 0.01 ms/token. These are
one-pass observations, with original/default module loading for the baseline
and EAGER for the fused executable; two EAGER baseline starts failed before
requests. Their configuration mismatch remains a qualification limit.

The prototype's final receipt counts exactly 1,536 fused calls:
three requests times 32 scalar steps times 16 full-attention layers. The
native logical route witness does not know about linker interposition; its
"unchanged/exact" final-step labels do not attest candidate numerics. The
wrapper receipt, linked ELF hash and source identify the experimental route.
Tiled Prefill stays approximately 219.5 seconds at P40 in both arms; its
separately accepted faster whole-core profile was not selected here.

All 32 shadow comparisons pass the narrow admission: no non-finite output,
zero repeated-call BF16 differences, maximum relative L2 error 0.00320752053
(0.320752%), and maximum absolute error 0.25. Restored shadow generation is
byte-identical to baseline. The true fused P40 output starts "The user wants
me to act as an AI agent...", whereas baseline starts "Based on the provided
repository files...". This demonstrates a changed numerical trajectory; it
neither establishes model-quality regression nor supplies a quality pass.
Short prefix outputs are also not a representative capability benchmark.

## Why a hybrid model can still slow down

This checkpoint has 48 Gated DeltaNet layers and 16 full-attention layers.
GDN Decode updates fixed-size recurrent state. Full attention still traverses
the growing history. Reducing the number of history-reading layers reduces
its coefficient; it does not make that cost independent of context.

The full-attention shape is Q24/KV4/D256: six query heads share each KV head.
The native score kernel assigns one warp to each query-head/position pair,
performs a scalar FP32 dot-product tree, and publishes a full FP32 score
matrix. Separate Softmax publishes probabilities. The value kernel launches
24 blocks of 256 threads; each thread traverses all positions through one
ordered FP32 FMA chain, with unroll 128. Query heads sharing a KV head issue
repeated logical K/V reads. This is a source-level access pattern, not a
measurement of physical DRAM transactions.

The earlier real-model profile records approximately 38.56 ms/token for
scores, 50.48 for values, and 2.86 for Softmax at P40000. These are profiler
kernel sums from the separately pinned historical artifact; they explain a
hypothesis, not a new matched attribution or hardware lower bound.

## Reference dataflow and numerical boundary

The installed vLLM 0.27.1 FlashInfer backend explicitly requests
`use_tensor_cores=True` for Decode. FlashInfer maps grouped queries into
matrix rows, tiles history, computes QK and PV with Tensor Cores, maintains
online Softmax state, and combines partition outputs with a separate merge.
The installed Triton unified-attention implementation similarly uses `tl.dot`
for QK/PV and a separate segment reduction. FLA's recurrent Decode updates
fixed-size state. These references suggest changing full-attention dataflow
while leaving projections and GDN unchanged.

The isolated executable links a GNU `--wrap` shim against the exact original
native object/library set. It uses the repository's already-vendored
FlashInfer 0.6.12 headers, compiled for SM87, with one query token and the
existing contiguous BF16 KV layout. It is an adaptation of that architecture,
not the paged vLLM backend or a claim of vLLM numerical identity. Static SASS
contains HMMA instructions and no ATOMG instruction in the wrapper object.

This route packs the six queries per KV head into a 16-row tile. It retains
FP32 accumulation and BF16 output before the original sigmoid gate, but
changes reduction order, probability rounding, exponent implementation and
partition-output rounding. It is **not bit-exact to the incumbent**. A fixed
merge avoids the previous split-values route's unordered atomicAdd combine;
repeatability must still be measured and does not prove model quality.

The shim admits only Q24/KV4/D256, scale 1/16, lengths 512..44095 and
non-captured execution. Other calls use the original implementation. Its
worst-case partition-output plus LSE storage is bounded by
`ceil(S/256) * (24*256*2 + 24*4)` bytes, below the existing `24*S*4`-byte
scratch span for every admitted length. No request allocation or KV change is
introduced. The shim repurposes scratch: it does **not** preserve the public
GQA API's normalized-probability scratch postcondition. This is safe only at
the audited internal runner call site, whose next consumer reads the BF16
output. A product implementation needs its own explicit internal boundary.

## Protocol and claim limits

`WP-DECODE-FUSED-GQA-20260927`, under
`AC-DECODE-CONTEXT-SCALING-20260927`, freezes one implementation, at most one
correctness/ownership repair, no parameter scan, immediate API return after
admission, and a 30-minute native-prototype device-time budget. Raw artifacts
are retained in `.q3x-work/decode-context-investigation-20260927/`.

All native requests use the existing `/v1/completions` API, the same real P40
prompt's exact token-ID prefixes (1089, 8192, 40000), greedy output, seed 42,
and 32 output tokens. All requests consume the full prompt; Prefix reuse and
MTP are absent. TPOT is `(last content arrival - first content arrival)/31`;
Decode rate is `1000/TPOT_ms`. This one-pass direct HTTP diagnostic is not a
formal EvalScope release panel or statistically repeated performance claim.
The baseline uses the ordinary Legacy-C512 profile; it does not measure the
separate fixed P40000/O16 whole-core deployment. Tiled Prefill is unchanged;
the scalar final-prompt step also reaches the shim.

Each process records preflight, GPU ownership, cache-drop attempt, sanitized
thermal telemetry, exact executable/request identities and owned cleanup.
Benign active-Codex CPU observations are retained with ordinary-diagnostic
qualification; they are not represented as strict release acceptance.
Both executables retain the original unqualified profile flags. The inherited
link command lists cuBLAS/cuBLASLt, but `readelf -d` and dynamic-symbol
inspection show neither as a retained dependency/import in either final ELF;
a link-line entry alone is not a runtime-dependency finding. No cuBLASLt
implementation is introduced by this experiment.

The shadow process compares all 16 full-attention layers at S40000 and
S40001, runs the fused operation twice on the same inputs, and restores the
original output before model execution continues. Its predeclared 2% relative
L2 screen detects gross integration errors only; it is not an accuracy
acceptance threshold. Full state/logits and public capability qualification
remain required before any production decision.

Two initial performance-process starts also failed the existing native
short-Graph free-memory increment check (360,701,952 and 357,687,296 bytes,
versus the 256 MiB limit), before any fused call or request. The successful
shadow process using the same ELF had recorded 101,654,528 bytes. These
failures remain explicit startup evidence. A subsequent attempt sets
`CUDA_MODULE_LOADING=EAGER` to move module initialization earlier; the
attempted comparison baseline uses the same setting. No admission threshold is changed.
Both unchanged-baseline EAGER starts failed (336,334,848 and 319,684,608
bytes), so EAGER is not a startup repair. The completed comparison therefore
retains a module-loading-mode mismatch: it is a directional observation, not
a fully matched performance pair. Successful runs do not establish repeated
startup reliability. Repetition stops after these two control attempts. Initial lazy/default observations are labeled separately.

An attempted stock-vLLM eager reference produced no usable metric: the first
startup rejected inconsistent scheduler limits, and a corrected attempt was
stopped after more than nine minutes of maximum-size multimodal initialization
before API readiness. Both failures and cleanup are retained. There is no
new measured vLLM rate or parity claim in this record.

## Bounded profile and independent process

After the positive direction run, a fresh process of the same research ELF
captures only the first `q3x.decode.step` after P40000 (S40001). Exactly one
registered NVTX range contains all 421 kernels on one CUDA stream and one
identified research-server PID. Old scalar score/value kernels are absent.

| Captured work | Calls | GPU kernel sum, ms |
| --- | ---: | ---: |
| Fused Tensor Core GQA | 16 | 13.670240 |
| Fixed partition-state merge | 16 | 0.088960 |
| Attention total | 32 | 13.759200 |
| Other model kernels | 389 | 102.969696 |
| Complete step | 421 | 116.728896 |

The kernel span is 117.274720 ms. The actual attention launch is
`grid=(1,8,4), block=(32,1,4)`, Q tile 16, eight KV partitions, 254 registers,
73,760 dynamic shared bytes, and zero local bytes/thread. The separate merge
uses `grid=(1,24,1), block=(32,4,1)`, 40 registers and 8,704 dynamic shared
bytes. This confirms the intended SM87 dataflow in real model execution;
it is not a matched old/new profiler pair or a DRAM-counter measurement.

All 32 output tokens' UTF-8 text match the previous fused process exactly.
That is a bounded independent-process text reproduction, not a full-logit,
long-output or universal determinism proof. The profiled request's external
TPOT is 176.216167 ms: capture termination/report handling interrupts the
stream, so that number has no unprofiled performance authority and is not
used in the comparison table. The 116.858220 ms result comes from the
separate ordinary unprofiled experiment.

## Explicit corrections to earlier interpretation

These corrections preserve the original historical observations:

- At S40000, unique BF16 K+V payload for the 16 full-attention layers is
  `16 * 2 * 40000 * 4 * 256 * 2 = 2,621,440,000` bytes (2.621 GB, decimal),
  not 2.013 GB. This is unique payload, not proven physical DRAM traffic.
- The native score launch has `24 * 40000 = 960,000` useful warps/layer,
  packed into 120,000 eight-warp blocks. The older approximately-120,024
  "warps" description confuses blocks with warps.
- The measured 6.29 tokens/s split route is a result for one scalar-kernel
  architecture. Its sum of incumbent component times is not a hardware bound
  on a materially different fused architecture. Neither that result nor a
  pure-read bandwidth calibration justifies lowering the product target.
- Non-bit-exactness, nondeterminism, and model-quality regression are different
  questions. An unordered atomic merge is one implementation choice, not an
  unavoidable cost of every changed floating-point reduction.

## Primary source pointers

- Native implementation: `src/kernels/reference/decode_ops.cu`, score and
  value kernels, and its GQA launcher; runner consumer in
  `src/runtime/reference_runner.cpp`.
- Vendored implementation: `third_party/flashinfer/include/flashinfer/attention/prefill.cuh`,
  `SinglePrefillWithKVCacheDispatched`; `attention/default_prefill_params.cuh`;
  `attention/variants.cuh` numerical parameters; `cascade.cuh` partition merge. Manifest checksums are verified.
- Installed reference source hashes and versions are in the raw identity
  records, including vLLM's FlashInfer backend, Triton unified attention and
  FLA recurrent implementation. User-owned package files remain read-only.
- [FlashInfer architecture introduction](https://flashinfer.ai/2024/02/02/introduce-flashinfer.html)
  explains Tensor Core GQA decoding;
  [attention API documentation](https://docs.flashinfer.ai/api/attention.html)
  describes deterministic split merging. The web documentation is not the
  version pin for the compiled local headers.

## Product convergence boundary

The exploration is bounded to one fused architecture and returns its value to
the real API. No additional unroll/split/tile scan is opened. The positive
direction justifies discussing a Decode numerical class and qualifying one
internal fused-GQA route, while preserving current weights, BF16 KV, recurrent
state, normal output capacity and single-request semantics. It would not
justify accepting nondeterministic atomic reduction or lowering a target by
default.

Product qualification should compose that route with the accepted Prefill
profile, exercise meaningful long outputs and representative prompts, compare
full logits/state and public capability under an explicitly accepted
numerical contract, and verify startup/resource/cancellation behavior. The
separate whole-core deployment currently has a fixed P40000/O16 contract; this
investigation does not extend it to normal product output lengths.

## Retained experiment source

The adjacent `fused_wrapper.cu`, `build_probe.py`, `run_panel.py`,
`preflight.py`, `summarize.py` and `summarize_profile.py` are frozen research
source snapshots, outside CMake and install
rules. They retain this host's paths and explicit active-Codex PID
qualification; they are not a generic unattended performance harness. The
final panel script includes EAGER and bounded profiling support; earlier
script versions and every attempt's command/environment are retained in the
raw bundle. The build script preserves original archives, adds only the
wrapper object and link interposition, and emits a separately named binary
under `.q3x-work/`. No `src/`, `include/`, or production build rule changes.

The [machine-readable record](../../metadata/qwen36-27b-decode-context-scaling-investigation-2026-09-27.json)
contains exact requests, outputs, metrics, source/build identities, admission,
profile, failed attempts and raw-artifact hashes. Final resource audit finds
all recorded server parents exited and no unexpected GPU holder; the raw
preflight's active-Codex CPU-only rejection remains preserved and qualified
as ordinary host context. The original executable hash and all runtime
sources remain unchanged. Nsight's workspace temporary was removed by the
tool. No production selection or numerical-contract amendment is made.
