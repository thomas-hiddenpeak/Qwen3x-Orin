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
- [`qwen36-27b-nvfp4-m32-table-free-e2m1-benchmark.json`](qwen36-27b-nvfp4-m32-table-free-e2m1-benchmark.json),
  which records the exact-M32 table-free E2M1 PRMT promotion, the retained
  512-byte E4M3 scale table, exhaustive signed-zero/NaN validation, compiler
  resources and canonical SASS, repeated same-cubin single-kernel and absolute
  dual-stream pair gates, detached-base P33/P513 max1 and P513 max8 B-C-C-B,
  matched Nsys critical-path attribution, exact model/Release/sanitizer suites,
  and the platform-blocked Nsight Compute counter attempt.
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
- [`qwen36-27b-fp8-m32-codebook-swizzle-rejection.json`](qwen36-27b-fp8-m32-codebook-swizzle-rejection.json),
  which records the rejected test-only U16 codebook swizzles for the four exact
  FP8 M32 dual-resident-A shapes. It freezes exhaustive byte-bijection,
  256-code/four-position, signed-NaN, replay, guard, resource, Graph, mode-0
  all-word SASS, four pinned real-payload, and alternating six-round evidence.
  The record separates the non-representative 1.55792x/1.52486x synthetic
  stress result from the failed 1.00308x/0.988108x actual weighted gates and
  includes the full 5,570,560-instruction static bank-wavefront audit, its
  non-NCU limits, and the complete source/test removal. Its then-next Decode
  FP8 M1 `[5120,6144]` single-body/no-tail-barrier screen is resolved by the
  separate rejection record below rather than retroactively claimed as part of
  the M32 evidence.
- [`qwen36-27b-attention-values-exact-benchmark.json`](qwen36-27b-attention-values-exact-benchmark.json),
  which records the production exact Q24/KV4/D256 attention-values route, its
  40-to-26 register and 376-to-112 `cuobjdump` function-block reductions,
  exact output/replay/guard/Graph/resource and full-suite gates, five-process
  hot-cell (1.30170x-1.50671x), S65..S513 chain (1.39585x-1.39948x), and rotating
  16-bank cold-S513 (1.79801x-1.80948x) results, neutral P33/max1 guard, and
  P513 reductions of 1.213% for max1 TTFT and 0.987% for max8
  subsequent-token latency. It introduces no runtime double/triple buffering:
  Prefill and Decode remain logically separate plans but batch-one execution
  is dependency-serialized; the next step is a refreshed profile of the next
  hotspot.
- [`qwen36-27b-bf16-m16-projection-fused-benchmark.json`](qwen36-27b-bf16-m16-projection-fused-benchmark.json),
  which records the production exact M16/N48/K5120 linear-attention BF16 A/B
  pair. It freezes the rejected row-resident comparison, selected 32-chain
  projection-fused implementation, bitwise/special-value/replay/canary and
  production Graph gates, final resources and SASS, five independent hot and
  rotating-cold processes, P33/P513 max1 and P513 max8 B-C-C-B evidence, and
  the before/after P513 profile. The exact target falls from 229.592608 to
  66.677600 ms (3.44332x); P33/P513 max1 TTFT falls by 2.304%/2.914%, while
  Decode remains neutral because M1 keeps the generic route. The record also
  retains one baseline-only global-memory warning and its clean audit, and
  makes clear that no system double/triple buffer or Prefill/Decode overlap
  was introduced.
- [`qwen36-27b-gdn-m16-register-resident-bf16-state-benchmark.json`](qwen36-27b-gdn-m16-register-resident-bf16-state-benchmark.json),
  which records the promoted exact-C16 GDN register-state route. It freezes the
  packed per-thread BF16 state and 45-MiB traffic-removal mechanism, exact
  output/state/replay/guard/Graph/selector gates, final SASS and resources,
  five fixed-frequency same-binary processes at 1.21660x-1.21796x, strict P33
  and P513/C32 B-C-C-B gains, and matched 96-launch Nsys attribution. It also
  records and excludes the stale-test-object exact-versus-exact runs, proves
  the orphan candidate object was not linked, and preserves the boundary
  between this intra-kernel optimization and system buffering or
  Prefill/Decode overlap.
- [`qwen36-27b-post-gdn-m16-phase-profile.json`](qwen36-27b-post-gdn-m16-phase-profile.json),
  which records the fixed-frequency production phase profile after that
  promotion. It freezes the P513/C32 Prefill and P19/C32/max26 Decode commands,
  binary/report identities, the retained Decode log, observed exact oracles,
  and NVTX-to-kernel closure; corrects the M32 gate/up raw sum to its two-stream
  marginal exposure; and ranks the resulting Prefill and Decode hotspots. It
  also records the current scheduling boundary: logical phase plans exist, but
  batch-one has no general double/triple buffer; Decode has zero overlap and
  only 0.733879% kernel-span idle, so phase-local work remains ahead of
  multi-request scheduler work.
- [`qwen36-27b-fp8-m1-attention-o-proj-no-tail-rejection.json`](qwen36-27b-fp8-m1-attention-o-proj-no-tail-rejection.json),
  which records the rejected first bounded follow-up to that Decode ranking.
  The exact `[5120,6144]` single-body candidate passes exhaustive E4M3FN,
  replay, NaN, guard, input, invalid Graph, topology, resource, and compiled
  mechanism gates; complete SASS removes 192 instructions and the third static
  barrier while preserving 64-register/four-CTA residency, 128 FFMA, and all
  key global/shared/output-store counts. All six fixed-clock actual layer-0
  payload rounds regress, however, producing a 0.994914x paired median and
  0.994295x minimum round against required 1.01x/1.00x gates. The record freezes
  the actual-first stop-loss, two older same-family failures, 142-function
  production-isolation manifest, retained candidate artifacts, complete
  source/test removal, forced clean rollback, and the deliberate absence of
  synthetic, five-process, P19, Nsys, or NCU work.
- [`qwen36-27b-nvfp4-m32-down-residual-epilogue-fusion-rejection.json`](qwen36-27b-nvfp4-m32-down-residual-epilogue-fusion-rejection.json),
  which records the rejected exact-M32 Prefill down/residual epilogue fusion.
  The test-only kernel preserves five-CTA residency and passes its finite,
  replay, guard, Graph, and resource checks, but reaches only 1.00050x across
  six fixed-clock actual-payload rounds, below the 1.005x technical and 1.02x
  promotion gates. It also records the independent audit finding that the
  candidate and test reverse the real residual-add operand order: finite
  performance evidence remains valid, while the test's 4/4 NaN bit match is
  explicitly not treated as a full runtime semantics proof. The candidate was
  removed before stress, independent-repeat, Nsys, NCU, or end-to-end work.
- [`qwen36-27b-m32-residual-centered-rmsnorm-fusion-benchmark.json`](qwen36-27b-m32-residual-centered-rmsnorm-fusion-benchmark.json),
  which records the promoted exact-M32 residual-add plus centered-RMSNorm
  boundary fusion. It freezes the rounded-BF16 bitwise and NaN-order contract,
  alias/guard/replay/Graph and selector gates, actual checkpoint norm fixtures,
  final resources and SASS, P33/P64/P513 model oracles, exact launch-topology
  reduction, and two independent P513 B-C-C-B groups. Hot repeated chains reach
  about 1.497x, production raw target kernels reach 1.0703x, and the measured
  whole P513 prefix improves by 0.278%; the record explicitly confines each
  claim and retains the cross-capture Nsight drift warning.
- [`qwen36-27b-m32-residual-rms-staged-register-cache-benchmark.json`](qwen36-27b-m32-residual-rms-staged-register-cache-benchmark.json),
  which records the follow-up C1/C2/C3/C4 screen and promotion of the
  512-thread shared-staging plus low-lane register-cache C4 kernel. It preserves
  the exact BF16 boundary and reduction tree while raising producer parallelism
  and removing the output-phase residual reread. The 2,048 production launches
  fall from a bracketed 62.403 to 37.396 ms (1.6687x), and two unprofiled P513
  B-C-C-B pairs reduce the whole prefix by 27.559 ms, or 0.579%, with neutral
  finish-prefill. The record separates hot-chain, production-kernel, and
  end-to-end claims and explains why the other three variants were not selected.
- [`qwen36-27b-post-c4-prefill-phase-profile.json`](qwen36-27b-post-c4-prefill-phase-profile.json),
  which refreshes fixed-frequency P513/C32 Prefill attribution after C4. It
  closes all 40,176 prefix kernels, separates the 391.939 ms dual-stream
  overlap from raw gate/up duration, places the promoted residual/RMSNorm
  chain at only 0.803% of kernel union, and shows that exact-M32 NVFP4 gate/up
  plus down retain 49.001% marginal exposure. It also freezes the next three
  test-only mechanism gates and keeps general scheduler buffering behind the
  much larger phase-local opportunities.
- [`qwen36-27b-nvfp4-m32-table-free-e4m3-rejection.json`](qwen36-27b-nvfp4-m32-table-free-e4m3-rejection.json),
  which records the first bounded post-C4 NVFP4 screen. The exact integer
  E4M3FN-to-BF16 constructor removes 512 B of shared state, two static LDS,
  and one barrier while preserving five-CTA residency and all bitwise/Graph
  contracts, but every isolated cell regresses to 0.94947x-0.95670x. Its
  dual-stream pair reaches only 1.01952x and includes a checkpoint-like round
  reversal, so the candidate and all test hooks are removed before wider
  profiling or end-to-end work.
- [`qwen36-27b-nvfp4-m32-gate-up-raw-weight-cp-async-benchmark.json`](qwen36-27b-nvfp4-m32-gate-up-raw-weight-cp-async-benchmark.json),
  which records the promoted exact-M32 gate/up single-slot raw-weight
  `cp.async` pipeline. It preserves five-CTA residency and bit-exact output,
  improves the isolated kernel by at least 1.0799x across five processes, and
  improves the production dual-stream pair by at least 1.0732x. Matched
  P513/C32 profiling reduces gate/up marginal exposure by 126.236 ms
  (1.1001x) and whole-prefix kernel union by 123.220 ms (1.0270x), while the
  source- and selector-unchanged down kernel differs by 0.0140%. The 4 KiB
  CTA-local staging slot is explicitly not a runtime double buffer or
  Prefill/Decode scheduler.
- [`qwen36-27b-nvfp4-m32-down-n160-rejection.json`](qwen36-27b-nvfp4-m32-down-n160-rejection.json),
  which records the rejected exact-M32 down N160/320-thread balanced-grid
  screen. It reduces the grid from 40 to 32 CTAs and passes bitwise, replay,
  guard, Graph, resource, and compiled-code gates, but all six fixed-clock
  actual-checkpoint rounds regress to 0.84983x-0.85050x. The stop-loss omits
  same-bank, Nsys, and end-to-end work, removes the candidate and test hooks,
  and leaves the promoted gate/up `cp.async` source unchanged.
- [`qwen36-27b-fp8-m32-half-tile-raw-weight-cp-async-rejection.json`](qwen36-27b-fp8-m32-half-tile-raw-weight-cp-async-rejection.json),
  which records the bounded exact-M32 FP8 `[10240,5120]` half-tile raw-weight
  `cp.async` screen. Its 4 KiB single slot preserves five-CTA residency,
  bit-exact output, final-stage draining, and every-round non-regression, but
  the retained actual-checkpoint paired median is only 1.02225x against the
  frozen 1.03x gate. Stop-loss therefore removes the candidate and skips
  stress, profiling, end-to-end work, and five-process replication while
  retaining a complete recovery patch for possible independent shape study.
- [`qwen36-27b-fp8-m32-k6144-half-tile-raw-weight-cp-async-rejection.json`](qwen36-27b-fp8-m32-k6144-half-tile-raw-weight-cp-async-rejection.json),
  which records that independent exact-M32 FP8 `[5120,6144]` K6144 screen.
  The test-only 4 KiB half-tile slot passes resource, SASS, exhaustive-code,
  full-K, replay, guard, input-preservation, Graph, and actual-weight/scalar
  plus deterministic-activation correctness gates, but all six fixed-clock rounds regress to
  0.947361x-0.948105x and the paired median is 0.947746x. The actual-first
  stop-loss removes the candidate and skips stress, NCU/Nsys, end-to-end, and
  five-process work, closing this longer-K shape without changing production.
- [`qwen36-27b-nvfp4-m32-asymmetric-table-free-e4m3-bgu-rejection.json`](qwen36-27b-nvfp4-m32-asymmetric-table-free-e4m3-bgu-rejection.json),
  which closes the exact-M32 NVFP4 one-sided table-free-E4M3 gate/up screen.
  The corrected packed-x2 candidate passes 65,536-pair exhaustive, static,
  full B/G/U, replay, guard, input-preservation, Graph, and actual-payload
  correctness gates, but gate/main and up/aux regress in all six fixed-clock
  rounds to 0.957097x and 0.956026x paired medians. The actual-first stop-loss
  removes the candidate and skips stress, profiling, end-to-end, and
  replication without changing production.
- [`qwen36-27b-gdn-m32-register-resident-bf16-state-rejection.json`](qwen36-27b-gdn-m32-register-resident-bf16-state-rejection.json),
  which closes the test-only exact-C32 extension of the production M16
  register-state lifetime. Output, final state, replay, input, Graph, and
  resource gates pass, and every fixed-clock round improves, but the 1.02474x
  paired median misses the frozen 1.03x early gate. Stop-loss removes the
  candidate before runner/model/profiler work and keeps production unchanged;
  its required fresh current-HEAD phase profile is recorded immediately below.
- [`qwen36-27b-current-head-phase-profile.json`](qwen36-27b-current-head-phase-profile.json),
  which refreshes fixed-clock P513/C32 Prefill and P19/C32/max26 Decode
  attribution at `9bddbda`. It freezes the binary, exact output oracles,
  reports, SQLite exports, analysis script, NVTX closure, and interval-union
  method. The current Prefill gate/up and down rows account for 48.238973% of
  prefix critical-path exposure; Decode remains single-stream with 0.584882%
  span idle and its top three rows account for 75.591051% of kernel time. The
  result keeps phase-local, NCU-guided kernel work ahead of general batch-one
  double/triple buffering and forbids restoring already closed mechanisms.
- [`qwen36-27b-fp8-m32-packed-decode-static-rejection.json`](qwen36-27b-fp8-m32-packed-decode-static-rejection.json),
  which closes the CPU/compile/SASS-only exact-M32 FP8 packed-decode screen.
  The exact four-code BF16-bit constructor removes all 32 codebook `LDS.U16`,
  512 B of shared state, and one barrier while retaining four `STS.128` decoded-B
  stores and five-CTA inferred residency, but grows the function from 632 to
  760 instructions and the K64 body from 200 to 379 instructions. Both exceed
  the frozen static stop, so no GPU program runs and production remains
  unchanged. This M32 WMMA shared-B constructor is distinct from ad22fdd's
  earlier M1 packed-load/FP32-decode GEMV. The next bounded priority at that
  snapshot was the phase-local exact-M32 FP8 QKV/Z dual-N128-tile fusion,
  now closed below.
- [`qwen36-27b-fp8-m32-qkv-z-dual-n128-fusion-rejection.json`](qwen36-27b-fp8-m32-qkv-z-dual-n128-fusion-rejection.json),
  which closes that phase-local exact-M32 QKV/Z screen. One 512-thread CTA
  maps two independent eight-warp N128 groups, halves the baseline 128 CTAs
  to 64, and shares the decoded codebook and `[32,5120]` activation traffic.
  Static, exhaustive-codebook, full-K, replay, guard, input-preservation,
  Graph, and actual-payload correctness gates all pass. The fixed-clock real
  QKV/Z payload result nevertheless regresses in all six `B-C-C-B` rounds:
  paired median 0.908776x, range 0.908670x-0.909103x. Stop-loss skips stress,
  NCU/Nsys, end-to-end work, and five-process replication, restores the two
  tracked source/test files exactly to `ce98625`, and retains the complete
  recovery patch.
- [`qwen36-27b-nvfp4-m1-gate-up-balanced-tail-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-balanced-tail-rejection.json),
  which closes the first Decode follow-up to the current-HEAD profile. The
  exact-M1 test-only candidate preserves the first 16,384 rows and redistributes
  only the final 1,024 rows from 32 row quads to 64 row pairs. Resources remain
  64 registers, 11,328 B shared, zero local, and four active CTAs/SM; actual,
  stress, and signed-nonfinite outputs are bit-exact with intact guards and
  preserved inputs. Nevertheless, all five fixed-clock `B-C-C-B` rounds regress:
  paired medians are 0.951273x actual and 0.955264x stress. Stop-loss removes
  the candidate, restores both tracked files exactly, and skips production,
  model-oracle, end-to-end, Nsys, NCU, and replication work.
- [`qwen36-27b-nvfp4-m1-gate-up-k256-packed-weight-pipeline-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-k256-packed-weight-pipeline-rejection.json),
  which closes the next exact-M1 Decode gate/up screen. A canonical-layout
  K256 cooperative loader adds one 4-KiB CTA raw-weight slot and forms a
  logical register/shared double buffer with `cp.async`; resources remain 64
  registers, zero local memory, and four active CTAs/SM. Static SASS,
  alignment, actual/stress bitwise, canary, and input-preservation gates all
  pass, but every fixed-clock round regresses. Paired medians are 0.932869x
  actual and 0.928905x stress. Stop-loss restores source/test exactly and
  retains the observed out-of-line `CALL/WARPSYNC/RET` synchronization as one
  narrower code-generation follow-up before an interleaved Decode sidecar.
- [`qwen36-27b-nvfp4-m1-gate-up-k256-inline-warp-barrier-static-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-k256-inline-warp-barrier-static-rejection.json),
  which closes that narrower code-generation follow-up. Volatile inline PTX
  `bar.warp.sync 0xffffffff` preserves the candidate's 64-register/15,424-byte
  resource envelope but compiles to a Function byte-identical to the measured
  source-`__syncwarp` version, including the same four `CALL.REL` sites and
  `WARPSYNC R30; RET` helper. Static stop-loss skips redundant correctness and
  timing, restores source/test exactly, and moves the next bounded screen to a
  single-layer AoSoA4 row-quad-interleaved Decode weight sidecar.
- [`qwen36-27b-nvfp4-m1-gate-up-aosoa4-sidecar-rejection.json`](qwen36-27b-nvfp4-m1-gate-up-aosoa4-sidecar-rejection.json),
  which closes that materially different layout screen. The single-layer
  85-MiB sidecar combines four row words into each `uint4`, generates two
  `LDG.E.128` weight loads and zero 32-bit weight loads in the unrolled phase,
  and retains the production 64-register/11,328-byte/four-CTA resource
  envelope. Actual and stress outputs are bit-exact, but actual regresses in
  all five rounds to a 0.994987x paired median; stress reaches only 1.00353x.
  Stop-loss skips the 5.3125-GiB full-model sidecar and restores source/test
  before re-ranking the remaining Decode hotspots.
- [`qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-selection.json`](qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-selection.json),
  which selects, at the test-only formal-microbenchmark stage, the exact FP8
  M1 `[5120,6144]` output-projection AoSoA4 plus byte-preswizzled sidecar. The
  pinned real layer-0 payload clears a 1.05155x paired-median gate with all
  five rounds non-regressing, same-bank stress clears 1.01465x, and bitwise,
  replay, guard, input/sidecar-preservation, and resource gates pass. The
  record also freezes the 30-MiB-per-layer/1.875-GiB-for-64-layers memory
  obligation and a projection-only 110.951-to-110.370 ms/token estimate. It
  does not claim production dispatch, loader integration, or an achieved
  end-to-end latency or throughput result.
- [`qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-production-benchmark.json`](qwen36-27b-fp8-m1-o-proj-aosoa4-preswizzled-production-benchmark.json),
  which records the subsequent persistent-sidecar production promotion. It
  separates the timed candidate from the final exception-path-fix binary,
  pins the GPU pack oracle, seven pack-invalid and eleven GEMV-invalid cases,
  the exact final-model replay, and the P19/C32/max26 `B1-C1-C2-B2` result.
  The mirrored hot Decode value is
  109.7585 ms/token, or 9.1109 token/s, while the record also exposes the
  non-amortized 1.875-GiB residency and roughly 467-ms cold pack cost.
- [`qwen36-27b-nvfp4-m1-down-norm-cta-prune-rejection.json`](qwen36-27b-nvfp4-m1-down-norm-cta-prune-rejection.json),
  which closes the synthetic exact-M1 fused-down post-grid-sync CTA-prune
  screen. Resources, distinct one-node `64x256` Graphs, both distributions,
  all three BF16 outputs, guards, and inputs pass, and every timing round
  non-regresses. Paired medians reach only 1.00135x checkpoint-like and
  1.00107x same-bank stress against the frozen 1.002x gate, so stop-loss skips
  actual checkpoint, NCU, model-oracle, and end-to-end work.
- [`qwen36-27b-post-fp8-output-sidecar-decode-phase-profile.json`](qwen36-27b-post-fp8-output-sidecar-decode-phase-profile.json),
  which records the required fixed-frequency Decode-only refresh at `aa7312b`
  using the final `77931b8` Release binary. All 25 Decode ranges close over
  10,925 unique kernels on one stream, raw equals union at 2,745.814816 ms,
  and the top three gate/up, QKV/Z, and down rows retain 75.878904% of kernel
  time. The production output-sidecar row averages 0.181293 ms per launch and
  contributes 11.602725 ms per Decode step. Its comparison with the earlier
  `9bddbda` trace is explicitly directional; the formal performance anchor
  remains the unprofiled 109.7585 ms/token and 9.1109 token/s `B1-C1-C2-B2`
  result.
- [`qwen36-27b-fp8-m1-qkv-z-aosoa4-preswizzled-sidecar-rejection.json`](qwen36-27b-fp8-m1-qkv-z-aosoa4-preswizzled-sidecar-rejection.json),
  which closes the test-only exact-M1 linear-attention QKV/Z
  AoSoA4/preswizzled sidecar screen. The hash-pinned layer-0 payload, stress,
  replay, guard, input-preservation, invalid-call, resource, Graph, and static
  production-identity gates pass. Actual payload improves in all five rounds
  but reaches only 1.00562x against 1.03x, while stress regresses in all five
  rounds to 0.984394x. Stop-loss skips the 3.75-GiB 48-layer sidecar and all
  production, model, profiler, and replication work, removes the candidate,
  and retains the restored default-test pass.
- [`qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-selection.json`](qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-selection.json),
  which selects the test-only exact-M1 residual/norm/gate/up/SiLU `32x512`
  CTA grouping for production integration. It preserves the same 512
  projection warps and 32 resident warps/SM as production's `64x256` route
  while halving repeated per-CTA residual/RMSNorm setup. Across three
  independent same-binary processes, the pinned actual-checkpoint and
  same-bank-stress paired medians reach 1.02410x and 1.02295x, with all 30
  rounds improving and finite/replay, signed Inf/NaN, guard, input, resource,
  Graph, invalid-call, and production-static-identity gates passing. The
  0.926016-ms/token 64-layer value (0.833856–0.927104-ms/token process range)
  is an arithmetic projection; production remains at 109.7585 ms/token and
  9.1109 token/s pending final production integration, model-oracle,
  end-to-end, and Nsys evidence.
- [`qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-production-benchmark.json`](qwen36-27b-nvfp4-m1-gate-up-cta-coarsen-production-benchmark.json),
  which records the subsequent zero-additional-sidecar production promotion.
  The final
  exact checkpoint/stress gates reach 1.02388x/1.02302x with every round
  improving; finite, replay, signed Inf/NaN, resource, Graph, invalid-call,
  SASS, Release, CTest, and full-model oracle checks pass. Fixed-frequency
  P19/C32/max26 `B1-C1-C2-B2` moves the mirrored hot Decode median from
  109.817 to 109.056 ms/token with both pairs improving, establishing
  9.1696 token/s and a remaining 9.056-ms/token stage gap.
- [`qwen36-27b-post-gate-up-cta-coarsen-decode-phase-profile.json`](qwen36-27b-post-gate-up-cta-coarsen-decode-phase-profile.json),
  which records the fresh production Decode closure. Across all 25 ranges,
  there are exactly 10,925 distinct kernel rows (437 per range) on one stream,
  with raw equal to union and zero overlap. The promoted `32x512` gate/up
  kernel appears 1,600 times and falls
  directionally from 39.675852 to 38.772791 ms/step; the old `64x256` kernel is
  absent. Generation/leaf closure and the retained FP8 output-sidecar route
  both pass, while the unprofiled 109.056-ms result remains authoritative.

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
