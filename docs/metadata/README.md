# Pinned checkpoint metadata evidence

This directory records normalized facts used to design and test Qwen3x-Orin.
The reports are evidence artifacts, not model files and not end-to-end support
claims. A report reaches only `metadata-compatible` until all referenced shard
headers and tensor contracts pass the runtime inspector on the target Orin.

The current reports pin the official NVIDIA ModelOpt artifacts:

- `nvidia/Qwen3.6-27B-NVFP4` at
  `0893e1606ff3d5f97a441f405d5fc541a6bdf404`;
- `nvidia/Qwen3.6-35B-A3B-NVFP4` at
  `491c2f1ea524c639598bf8fa787a93fed5a6fbce`.

The machine-readable
[`qwen36-27b-native-reference-run.json`](qwen36-27b-native-reference-run.json)
records the first successful full native 27B generation: exact 19 prompt and
26 oracle output IDs, exact decoded text/stop semantics, cold-load and
generation timings, native position-18/19 trace hashes, and the canonical
reasons native and vLLM boundary hashes are not expected to be bitwise equal.

The diagnostic Phase 3 records are:

- [`qwen36-27b-reference-nsys-baseline.json`](qwen36-27b-reference-nsys-baseline.json),
  which pins the reference CUDA kernel and launch-shape profile, toolchain,
  report hash, and reproducible `nsys stats` commands;
- [`qwen36-27b-projection-backend-benchmark.json`](qwen36-27b-projection-backend-benchmark.json),
  which retains every paired reference/SM87 sample, strict replay results,
  exact full-model gate, memory watermarks, microbenchmarks, and limitations.
- [`qwen36-27b-afalg-loader-benchmark.json`](qwen36-27b-afalg-loader-benchmark.json),
  which records the post-SM87 startup diagnosis, observed kernel SHA-256
  provider, accelerated full-shard authentication/load timings, unchanged
  exact-generation gates, and historical-comparison limitations.
- [`qwen36-27b-parallel-loader-benchmark.json`](qwen36-27b-parallel-loader-benchmark.json),
  which records the same-binary warm-cache one-/two-/three-worker resident-load
  measurements, deterministic loader invariants, derived speedups, and
  measurement limitations.
- [`qwen36-27b-startup-overlap-benchmark.json`](qwen36-27b-startup-overlap-benchmark.json),
  which records the same-binary serial/overlapped tokenizer-and-resident
  startup comparison, exact full-model gates, wall-time contract, fallback
  behavior, and measurement limitations.
- [`qwen36-27b-nvfp4-packedx8-benchmark.json`](qwen36-27b-nvfp4-packedx8-benchmark.json),
  which records the canonical packed-x8 NVFP4 dispatch, same-binary scalar/
  vector gate, two-prompt replay benchmark, full-model exact gate, SASS
  resources, and post-change Nsight profile.
- [`qwen36-27b-fp8-packedx4-benchmark.json`](qwen36-27b-fp8-packedx4-benchmark.json),
  which records the canonical packed-x4 FP8 dispatch, exhaustive E4M3FN and
  fallback gates, same-binary scalar/vector measurements, exact 27B replay,
  SASS resources, and the post-change Nsight profile.
- [`qwen36-27b-c8-prefill-benchmark.json`](qwen36-27b-c8-prefill-benchmark.json),
  which records the first bounded C1/C8 prompt-prefix comparison, exact C8
  full-model oracle gate, request-arena increment, execution policy, and
  diagnostic limitations.
- [`qwen36-27b-c8-kernel-optimization-benchmark.json`](qwen36-27b-c8-kernel-optimization-benchmark.json),
  which records the subsequent C8 kernel-optimization chain, matched-shape
  end-to-end medians, GDN and NVFP4 row-pair microbenchmarks, unchanged exact
  full-model gate, dispatch scope, and diagnostic limitations.
- [`qwen36-27b-c16-tensor-core-prefill-benchmark.json`](qwen36-27b-c16-tensor-core-prefill-benchmark.json),
  which records the bounded C16 runtime and GDN extension, fixed-shape FP8 and
  NVFP4 Tensor Core dispatch, exhaustive correctness gates, mirrored same-binary
  C8/C16 samples, and the final C16 Nsight kernel profile.
- [`qwen36-27b-fp8-kv-pair-benchmark.json`](qwen36-27b-fp8-kv-pair-benchmark.json),
  which records the exact-shape M1 FP8 K/V projection pair, same-binary
  synthetic kernel gate, max-26-token Nsight comparison, mirrored single-load
  generation benchmark, exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-dual-benchmark.json`](qwen36-27b-nvfp4-down-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[5120,17408]` down-projection
  dual-iteration kernel, repeated same-binary gates, unchanged fallbacks,
  max-26-token Nsight comparison, mirrored single-load generation benchmark,
  exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json`](qwen36-27b-nvfp4-gate-up-xor-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[17408,5120]` gate/up adjacent-lane
  XOR-dual kernel, repeated same-binary gates, unchanged fallbacks, max-26-token
  Nsight comparison, mirrored single-load generation benchmark, exact replay
  gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json`](qwen36-27b-nvfp4-lm-head-xor-dual-benchmark.json),
  which records the aligned M1 NVFP4 `[248320,5120]` lm-head adjacent-lane
  XOR-dual kernel, its same-binary production gate, unchanged fallbacks,
  max-26-token Nsight comparison, mirrored single-load generation benchmark,
  exact replay gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-data-reuse-benchmark.json`](qwen36-27b-nvfp4-data-reuse-benchmark.json),
  which records the follow-up aligned M1 `[5120,17408]` down adjacent-lane
  XOR-dual route and `[248320,5120]` lm-head CTA activation-staged route,
  same-binary identity/performance gates, B-C-C-B generation evidence against
  an independently rebuilt base commit, matched max-26-token profiles, exact
  replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json`](qwen36-27b-nvfp4-gate-up-activation-staged-benchmark.json),
  which records the follow-up aligned M1 `[17408,5120]` gate/up CTA
  activation-staged route, default capture-only exact/fallback dispatch
  identity gates, same-binary performance and bitwise gates, an independently
  rebuilt-base B-C-C-B generation benchmark, matched max-26-token profiles,
  exact replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-activation-staged-benchmark.json`](qwen36-27b-nvfp4-down-activation-staged-benchmark.json),
  which records the follow-up aligned M1 `[5120,17408]` down CTA
  activation-staged route, its preserved direct-XOR baseline and scalar
  fallbacks, same-binary performance and bitwise gates, a saved-base B-C-C-B
  generation benchmark, matched max-26-token profiles,
  exact replay, and diagnostic limitations.
- [`qwen36-27b-gdn-eight-row-benchmark.json`](qwen36-27b-gdn-eight-row-benchmark.json),
  which records the production eight-row lane-striped GDN update against its
  preserved four-row predecessor, M1/M2/M8/M16 bitwise state/output gates,
  mirrored same-binary measurements with current-profile call weights, matched
  max-26 profiles, compiler resources, exact replay, and diagnostic
  limitations.
- [`qwen36-27b-fp8-qkv-z-fusion-benchmark.json`](qwen36-27b-fp8-qkv-z-fusion-benchmark.json),
  which records the exact aligned FP8 M1 linear-attention QKV/Z two-phase
  fusion, five frozen actual-checkpoint/stress gates, compiler resources,
  matched max-26 profiles, detached-base B-C-C-B generation evidence, C1/C8/C16
  exact replay, ordered fallbacks, and diagnostic limitations.
- [`qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json`](qwen36-27b-nvfp4-gate-up-silu-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 dense-MLP gate/up/SiLU fusion,
  five frozen actual-checkpoint/same-bank gates, compiler resources, matched
  max-26 profiles, detached-base B-C-C-B generation evidence, C1/C8/C16 exact
  replay, ordered fallbacks, and diagnostic limitations.
- [`qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json`](qwen36-27b-nvfp4-residual-norm-gate-up-silu-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 post-attention residual-add,
  centered-RMSNorm, gate/up, and SiLU fusion, its redundant per-CTA norm
  reduction, five frozen gates, dispatch/alias contracts, matched profiles,
  detached-base B-C-C-B evidence, exact replay, and diagnostic limitations.
- [`qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json`](qwen36-27b-nvfp4-residual-norm-warp-tail-reduction-benchmark.json),
  which records the reduction-only follow-up inside that post-attention
  fusion: shared-memory strides 128/64/32 followed by exact warp-zero shuffle
  strides 16/8/4/2/1, the preserved full-shared-tree same-binary predecessor,
  compiler resources, five direct actual-checkpoint/same-bank B-C-C-B gates,
  a matched max-26 profile, an independent B-C-C-B generation comparison,
  a bitwise-equal 5,905-line trace, full-shape finite/nonfinite closeout gates,
  and diagnostic limits.
- [`qwen36-27b-fp8-q-kv-fusion-benchmark.json`](qwen36-27b-fp8-q-kv-fusion-benchmark.json),
  which records the exact aligned FP8 M1 full-attention Q plus K/V fusion,
  five same-binary production-shaped gates, compiler resources, complete
  dispatch/alias contracts, matched profiles, detached-base B-C-C-B evidence,
  exact model gates, and diagnostic limitations.
- [`qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json`](qwen36-27b-nvfp4-down-residual-norm-fusion-benchmark.json),
  which records the exact aligned NVFP4 M1 dense-MLP down projection plus
  residual-add and centered-RMSNorm cooperative fusion, five production-shaped
  gates, finite and signed-nonfinite bitwise contracts, zero-node validation,
  matched profiles, detached-base B-C-C-B and full-trace equality, exact model
  gates, the zero-residency-margin AGX Orin constraint, and diagnostic limits.
- [`qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json`](qwen36-27b-gdn-rmsnorm-silu-gate-fusion-benchmark.json),
  which records the canonical M1 GDN plus 48x128 headwise plain-RMSNorm/SiLU
  gate implementation, its one-node/fallback/alias/resource contracts, the
  hardened 24-state same-binary gate, matched max-26 profiles, detached-base
  B-C-C-B and 5,905-line full-trace equality, C1/C8/C16 exact-model gates,
  release/sanitizer suites, and diagnostic limitations. Five earlier probes
  remain explicitly excluded as pre-production prototype inventory.
- [`qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json`](qwen36-27b-gdn-rmsnorm-silu-gate-warp-tail-benchmark.json),
  which records the exact shared-prefix/warp-zero reduction tail inside that
  production fused GDN kernel, the test-only full shared-tree predecessor,
  identical launch/resource and three-step bitwise contracts, the 28-to-24
  SASS `BAR` reduction, five hardened cold-state same-binary gates, the
  directly attributed max-26 target-kernel delta, exact model/trace suites,
  and the absence of an attributable detached-process end-to-end gain. Early
  prototypes and the test-only launcher are explicitly excluded from the
  production claim.
- [`qwen36-27b-prefill-decode-plan-split-benchmark.json`](qwen36-27b-prefill-decode-plan-split-benchmark.json),
  which records the host-control Prefill/Decode plan seam, explicit prefix,
  final-prompt, and feedback routing, the 44-case controller matrix, exact
  C1/C8/C16 model and arena gates, 5,905-line trace equality, identical ordered
  13,558-launch contracts, detached-base B-C-C-B regression evidence, and the
  explicit absence of new streams, buffers, kernels, or an attributed speedup.
- [`qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json`](qwen36-27b-sm87-shape-chunk-prompt-matrix-benchmark.json),
  which freezes the resolved SM87 projection route registry, its exact
  launch-contract and B-C-C-B equivalence gates, all 29 intended M1/M2/M8/M16
  direct-kernel cells, the P19 C1/C2/C8/C16 TTFT matrix, tokenizer-pinned
  P33/P65/P129/P513 prompts, and matched Nsight launch/time attribution. It
  records the unrelated historical M4 aggregate-threshold failure separately
  instead of treating that measurement gap as a production correctness error.
- [`qwen36-27b-c32-composite-prefill-benchmark.json`](qwen36-27b-c32-composite-prefill-benchmark.json),
  which records the ABI-0.2.0 C32 composite outer tile, exact real-model gates,
  P19 B-C-C-B and long-prompt C16/C32 timings, matched Nsight attribution, and
  the explicit single-stream M16-plus-tail projection and unlocked-clock limits.
- [`qwen36-27b-fp8-m32-production-benchmark.json`](qwen36-27b-fp8-m32-production-benchmark.json),
  which records the production fixed-M32 FP8 route, four-shape bitwise/guard/
  single-node gates, complete fallback validation, exact C1/C8/C16/C32 model
  replay, P33/C32 B-C-C-B TTFT evidence, and matched Nsight launch/time
  attribution against the frozen composite baseline.
- [`qwen36-27b-nvfp4-m32-production-benchmark.json`](qwen36-27b-nvfp4-m32-production-benchmark.json),
  which records the production fixed-M32 NVFP4 route, the K64/LD72 versus
  K128/LD136 same-cubin selection gate, two-shape bitwise/resource/graph and
  fallback contracts, exact C1/C8/C16/C32 model replay, the tokenizer-pinned
  P33/C32 B-C-C-B result, and matched baseline/candidate Nsight attribution.
- [`qwen36-27b-nvfp4-m32-scale-window-benchmark.json`](qwen36-27b-nvfp4-m32-scale-window-benchmark.json),
  which records the K256 coalesced block-scale window, same-cubin and NCU
  evidence, exact model replay, P33/C32 attribution, rejected swizzle/tile
  probes, and the later correction that its historical `build/orin-asan`
  directory was not actually sanitizer-instrumented.
- [`qwen36-27b-nvfp4-m32-factorized-lookup-benchmark.json`](qwen36-27b-nvfp4-m32-factorized-lookup-benchmark.json),
  which records the production factorized E2M1/E4M3 lookup, exhaustive bitwise
  gate, resource/SASS changes, same-cubin timing, exact-template NCU tradeoff,
  verified ASan/UBSan and model suites, and matched P33/C32 TTFT/Nsight
  attribution.
- [`qwen36-27b-nvfp4-m32-vector-store-benchmark.json`](qwen36-27b-nvfp4-m32-vector-store-benchmark.json),
  which records the production vectorized decoded-tile stores, direct and
  same-cubin bitwise gates, unchanged resources, replay-scoped shared-wavefront
  reduction, verified Release/sanitizer/model suites, and matched P33/C32
  TTFT/Nsight attribution.
- [`qwen36-27b-nvfp4-m32-gate-up-dual-stream-benchmark.json`](qwen36-27b-nvfp4-m32-gate-up-dual-stream-benchmark.json),
  which records the exact C32 NVFP4 MLP gate/up auxiliary-stream schedule,
  best-effort serial fallback and lifecycle contracts, production-dispatch
  bitwise/performance gates, real cross-stream overlap, P33 promotion rounds,
  P33/P65/P129/P513 scaling, and Release/sanitizer/model validation.
- [`qwen36-27b-linear-ab-sidecar-rejection.json`](qwen36-27b-linear-ab-sidecar-rejection.json),
  which records the test-only exact-C32 production-dispatch A/B sidecar gate,
  three scheduling variants, four-output bitwise equality, marginal and
  regressing envelopes, and the decision to retain the serial runtime.
- [`qwen36-27b-nvfp4-m18-masked-m32-benchmark.json`](qwen36-27b-nvfp4-m18-masked-m32-benchmark.json),
  which records the exact aligned NVFP4 M18 masked-M32 Prefill route, exact-C18
  capacity and bitwise gates, four-cell synthetic kernel result, P19/C32
  B-C-C-B TTFT promotion, matched Nsight launch/time attribution, full model
  replay, host sanitizer coverage, and the platform-blocked device memcheck.
- [`qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json`](qwen36-27b-nvfp4-m17-m31-runtime-masked-m32-benchmark.json),
  which records the production exact-capacity runtime-masked M17 and M19-M31
  routes for both NVFP4 MLP shapes, all 14 weighted microbenchmark gates,
  direct/pair/fallback and zero-node validation, the ten-prompt no-reversal
  B-C-C-B result, matched P18/P26/P64 Nsight attribution, exact model and host
  sanitizer suites, and unchanged fixed-M18/M32 production SASS. Its ten prompt
  texts, token IDs, hashes, and C32 schedules are frozen separately in the
  [`qwen36-27b-sm87-prefill-tail-prompts-v1.json` manifest](../../benchmarks/qwen36-27b-sm87-prefill-tail-prompts-v1.json).
  Prefill and Decode remain logically separate plans; only exact M32 gate/up
  has the existing layer-local dual-stream path, with no general double/triple
  buffering or Prefill/Decode overlap.
- [`qwen36-27b-nvfp4-m17-m32-gate-up-dual-stream-rejection.json`](qwen36-27b-nvfp4-m17-m32-gate-up-dual-stream-rejection.json),
  which records the rejected attempt to generalize that M32 gate/up schedule
  across M17-M32: all 32 synthetic correctness/cell gates, all 16 per-M gates,
  and the 1.08146x micro aggregate pass, but the twelve-prompt whole-model
  result reaches only 1.007059x against the required 1.01x. It freezes the
  baseline/candidate/withdrawn-patch identities, representative prompt and
  output contracts, the deliberate absence of a post-failure Nsight Systems
  profile for that candidate, and the decision to retain serial M17-M31/M18
  plus the existing M32-only auxiliary stream. It also separates a later
  Nsight Compute diagnostic of the retained production M17 kernel from the
  rejected candidate and records the evidence basis for a test-only 4 KiB
  raw-weight prefetch experiment.
- [`qwen36-27b-nvfp4-m17-m31-gate-up-raw-weight-cp-async-rejection.json`](qwen36-27b-nvfp4-m17-m31-gate-up-raw-weight-cp-async-rejection.json),
  which closes that 4 KiB single-slot experiment. The exact gate/up probe
  passes all 28 synthetic cells and 112 mirrored rounds at 1.06036x aggregate,
  while matched M17 Nsight Compute measures 1.06315x and lower long-scoreboard
  pressure without reducing five-CTA occupancy. All six screened down cells
  regress, however, and the formal twelve-prompt B-C-C-B result reaches only
  1.007587x against the required 1.01x. The temporary selector was fully
  withdrawn; production retains serial M17/M19-M31 and fixed-M18 gate/up plus
  the existing exact-M32 auxiliary-stream route. The next priority is a
  trace-backed Prefill/Decode phase and scheduler-overlap ceiling, not wider
  local raw-weight prefetch.
- [`qwen36-27b-prefill-decode-phase-trace-baseline.json`](qwen36-27b-prefill-decode-phase-trace-baseline.json),
  which records the default-off phase-aware timing/NVTX contract, exact
  P19/P33/P64/P513 range-to-kernel closure, CUDA-event topology calibration,
  phase-local hotspot split, registered-string validation, and the matched
  B-C-C-B proof that disabled instrumentation has no material regression. It
  also freezes the current boundary: Prefill and Decode are logically
  measurable but dependency-serialized, with no general double/triple buffer
  or multi-request scheduler.
- [`qwen36-27b-prefill-attention-score-warp-positions-benchmark.json`](qwen36-27b-prefill-attention-score-warp-positions-benchmark.json),
  which records the promoted exact Q24/KV4/D256 S>=65 warp-position attention
  score kernel: bitwise and S64/S65 dispatch gates, five mirrored same-binary
  micro rounds, matched S513 NCU, P513 Prefill Nsight and B-C-C-B promotion,
  direct P64-max26 long-context Decode evidence, P19 short-context fallback,
  complete test suites, memory-watermark caveats, and the unchanged public
  ABI, streams, buffers, softmax, value path, and serialized phase scheduler.
- [`qwen36-27b-nvfp4-m1-down-residual-norm-warp-tail-rejection.json`](qwen36-27b-nvfp4-m1-down-residual-norm-warp-tail-rejection.json),
  which records the rejected first post-trace Decode M1 experiment: full
  finite/nonfinite, invalid-call, graph, and resource contracts pass, but all
  five same-binary processes regress to 0.972242x-0.974540x against the 1.005x
  gate. Matched NCU explains why fewer `BAR`/shared-memory instructions do not
  win in this kernel, and the record freezes complete candidate removal with
  no retained test probe, runtime change, end-to-end run, or dispatch change.
- [`qwen36-27b-fp8-m1-qkv-z-activation-staged-rejection.json`](qwen36-27b-fp8-m1-qkv-z-activation-staged-rejection.json),
  which records the rejected test-only Decode M1 FP8 QKV/Z activation-staging
  experiment: actual layer-0 checkpoint payload identity, exhaustive bitwise/
  replay/canary/input and resource gates, all seven actual-checkpoint and
  same-bank cap cells, and the frozen 0.913426x/0.905513x failure against the
  1.02x/1.00x gates. It records the fixed 10-KiB-per-CTA staging mechanism,
  exact log and restored-artifact identities, complete candidate removal, the
  production-only NCU locality bound, and the stop-loss decision not to run
  candidate NCU or end-to-end evaluation.
- [`qwen36-27b-nvfp4-m1-factorized-rejection.json`](qwen36-27b-nvfp4-m1-factorized-rejection.json),
  which records the rejected test-only Decode M1 BF16 pair/scale factorization:
  exhaustive and full-path bitwise gates, compiler resources, unchanged
  production SASS, two consistently regressing full-shape synthetic cells, the
  static-instruction interpretation, removal of all candidate code, and the
  explicit absence of a production-dispatch change.
- [`qwen36-27b-nvfp4-m1-full-product-table-rejection.json`](qwen36-27b-nvfp4-m1-full-product-table-rejection.json),
  which records the rejected test-only Decode M1 canonical full-product-table
  candidate for the fused residual/norm/gate/up/SiLU hotspot. Corrected
  synthetic checkpoint-like and same-bank screens reach only 0.656836x and
  0.615217x despite bitwise full-path, replay, canary, graph, and resource
  gates. The record freezes candidate/clean binary identities, static-SASS
  attribution and its no-NCU limit, host/SM87 NaN-canonicalization boundaries,
  complete candidate removal, and unchanged production dispatch.
- [`qwen36-27b-gdn-m16-shared-resident-bf16-state-rejection.json`](qwen36-27b-gdn-m16-shared-resident-bf16-state-rejection.json),
  which records the rejected test-only Prefill GDN M16 row4/row8
  shared-resident BF16 state candidates. Both pass exact M1/M2/M8/M16
  in-place/disjoint output and state, poison replay, C16 versus C8+C8, Graph,
  canary, input-preservation, and resource gates, but the hardened 24-bank
  screen reaches only 0.627300x and 0.752132x against production row8. The
  record freezes the initial same-TU dynamic-shared padding hazard, final
  test-only-TU isolation, complete candidate removal, restored production
  source/resource/SASS identity, exact artifact hashes, and the stop-loss
  decision not to run candidate NCU, Nsys, or end-to-end evaluation.
- [`qwen36-27b-fp8-m1-qkv-z-reduction-scratch-ping-pong-benchmark.json`](qwen36-27b-fp8-m1-qkv-z-reduction-scratch-ping-pong-benchmark.json),
  which records the promoted two-slot CTA-local FP8 M1 QKV/Z reduction-scratch
  pipeline. It freezes exhaustive 256-code/four-byte-position coverage,
  four-cap race-signature/replay and Graph contracts, exact checkpoint payload
  identity, resources and per-function SASS, the 142-non-target-function
  production-isolation proof, frozen 1.02407x/1.00697x micro gates, and exact
  P19/P64/P513 B-C-C-B evidence with all 12 log hashes. The record retains the
  external P19 memory-watermark warning and unavailable racecheck as explicit
  limitations, and makes clear that this intra-kernel scratch ping-pong is not
  general double/triple buffering or Prefill/Decode overlap.
- [`qwen36-27b-fp8-m1-q-kv-reduction-scratch-ping-pong-benchmark.json`](qwen36-27b-fp8-m1-q-kv-reduction-scratch-ping-pong-benchmark.json),
  which records the promoted two-slot CTA-local FP8 M1 full-attention Q+K/V
  reduction-scratch pipeline. It freezes exhaustive 256-code/four-byte-position
  coverage across Q, K, and V, ordered Q-row-quad and Q-to-K/V race/replay
  contracts, exact checkpoint payloads, resources, three explicitly distinct
  SASS canonicalizations, five independent actual/stress micro processes, and
  exact P19/P64/P513 B-C-C-B evidence with all 12 log hashes. It also records
  the symmetric-engine device-code isolation proof, zero persistent-drop flags,
  the 0.042171% worst whole-model stage regression against the 0.5% limit, and
  the boundary between this intra-kernel scratch mechanism and system buffering.
- [`qwen36-27b-fp8-m32-dual-resident-a-benchmark.json`](qwen36-27b-fp8-m32-dual-resident-a-benchmark.json),
  which records the promoted exact-shape FP8 M32 dual-resident-A WMMA route.
  It freezes full 256-code/four-byte-position bitwise and NaN coverage, resource
  and Graph role gates, the preserved single-resident predecessor SASS, five
  independent same-binary micro processes, and symmetric P33/P513 B-C-C-B
  evidence. The record also proves that both end-to-end runners contain the
  same 149 CUDA functions with identical encodings, while P33 and P513 TTFT
  improve by 2.94% and 3.76% respectively without changing the serialized
  Prefill/Decode scheduler or adding system double/triple buffering.

The model-compatibility reports contain raw SHA-256 hashes for `config.json`,
`hf_quant_config.json`, and `model.safetensors.index.json`; normalized model and
quantization fields; index counts; representative shapes read from the first
safetensors shard header; and benchmark shapes derived from that evidence.
Tensor payloads were not downloaded to produce the header probes: the first
eight bytes were read to obtain the little-endian header length, followed by a
bounded HTTP byte-range request for the JSON header.

The pinned 27B artifact was subsequently materialized in a separate local
model directory. Git LFS and an independent `sha256sum` pass matched all three
published object IDs. `qwen3x-inspect checkpoint MODEL_DIR --require-shards`
then validated all 3 shard headers, all 2,194 index mappings, and exactly
21,921,428,072 payload bytes without reading payload contents. The 27B report
records all shard hashes/sizes and a text/vision/MTP storage split. No model
file is copied into this repository.

Once `qwen3x-inspect` is built, local evidence can be reproduced without loading
weight payloads into memory:

```bash
qwen3x-inspect index MODEL_DIR/model.safetensors.index.json
qwen3x-inspect header MODEL_DIR/model-00001-of-00003.safetensors
qwen3x-inspect checkpoint MODEL_DIR
qwen3x-inspect checkpoint MODEL_DIR --require-shards
qwen3x-inspect manifest MODEL_DIR
qwen3x-inspect load-plan MODEL_DIR
```

The strict form requires regular, non-symlink shard files and validates every
header, tensor membership/ownership, and aggregate payload size. The
non-strict form is intentionally metadata-only: it reports shard presence but
sets `shard_contract_validated=false`.

`load-plan` additionally assigns the exact deterministic text-only arena layout
against the three compiled full-file identities without allocating GPU memory.
The separate conditional resident-loader integration then read and SHA-256
authenticated all 21,921,697,184 file bytes in one sequential pass, copied
20,150,569,096 text bytes into one 20,150,786,560-byte CUDA arena, and skipped
1,771,128,088 bytes. Full details and the target-device reproduction command
are in [RESIDENT_WEIGHT_LOADER.md](../RESIDENT_WEIGHT_LOADER.md).

The inspector treats filenames and directory names as untrusted context. Model
series and quantization compatibility require exact pinned descriptors plus
semantic validation. A shape-compatible unknown revision remains unsupported.

## Known upstream metadata quirk

For the pinned 35B-A3B artifact, external `hf_quant_config.json` identifies
ModelOpt 0.44.0 and FP8 KV cache, while the embedded
`config.json.quantization_config` carries stale ModelOpt 0.37.0 metadata and no
KV declaration. Their 291 per-module `quantized_layers` entries agree exactly.
Qwen3x-Orin permits that discrepancy only for the pinned file hashes and treats
the external file as authoritative. The same disagreement on an unknown
revision fails closed.

Model weights, tokenizers, and source configuration files remain separately
licensed artifacts and are not copied into this repository.
