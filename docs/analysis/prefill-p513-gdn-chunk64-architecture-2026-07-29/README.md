# P513 Chunk64/WY GDN architecture screen

Status: positive architecture proof; test-only and not production eligible.

This screen replaced the token-serial Prefill GDN recurrence with a C64/WY
hierarchy on the authenticated Qwen3.6-27B-NVFP4 checkpoint. The default
runtime remains unchanged. The route is compiled only when both
`BUILD_TESTING=ON` and `Q3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION=ON`; it is
also gated at runtime by a private test switch and accepts only the exact
P513/C512/position-zero shape.

The route deliberately used classic cuBLAS batched GEMMs to establish the
W/U/QK architecture ceiling while its three dominant stages were replaced by
native SM87 WMMA kernels. Neither classic cuBLAS nor cuBLASLt has production,
fallback, retention, or promotion authority.

## Real-model result

The final snapshot-free P513 run used one baseline and one candidate warm-up,
then one measured baseline and candidate request. Both generated token 9419,
text `Hello`, and the candidate hit all 48 linear layers.

| Metric | Native baseline | Chunk64 candidate | Change |
| --- | ---: | ---: | ---: |
| Prefix | 2,260.333589 ms | 1,912.793973 ms | -347.539616 ms |
| Prefix throughput | 226.515238 token/s | 267.671274 token/s | 1.181692x |
| TTFT | 2,369.074987 ms | 2,021.535722 ms | -347.539265 ms |

The architecture therefore crossed the required 300 ms whole-Prefix saving
in one coherent route. It did not cross the second GDN budget: the final
profile attributes about 150.035424 ms to the remaining Chunk64/GDN stages,
versus the 100 ms milestone.

The progression was architectural rather than a tile scan:

| Route state | Candidate Prefix | Saved from paired baseline | Structural change |
| --- | ---: | ---: | --- |
| external stage reference | 1,991.864247 ms | 269.290752 ms | complete C64/WY hierarchy |
| native stage A | 1,957.784642 ms | 303.614568 ms | fused KKT plus 4x4 block-16 triangular solve |
| native stages A+B | 1,935.609063 ms | 324.560907 ms | FP32 state fragments persistent across eight chunks |
| native stages A+B+C | 1,912.793973 ms | 347.539616 ms | fused output reconstruction, BF16 boundary, RMSNorm, and SiLU(Z) |

The rows are separate paired real-model runs and are not additive microkernel
claims.

## Final kernel attribution

The CUDA Profiler API captured only the measured candidate request. The most
recent stage totals are checked into [`gdn-stage-summary.csv`](gdn-stage-summary.csv).
The three native stages account for 108.968512 ms:

- output reconstruction plus norm/gate: 43.388864 ms;
- persistent boundary-state propagation: 40.831648 ms; and
- fused KKT/block solve: 24.748000 ms.

Pack/normalize/gate, the QK scale, and the remaining external W/U/QK GEMMs
raise the attributed Chunk64 total to 150.035424 ms. This is now a bounded
producer/consumer and data-ownership problem, not justification for another
parameter sweep.

NCU on the first real-model invocation of each native stage found no local or
shared-memory spills. The two largest stages were L1/TEX limited rather than
Tensor-Core limited:

| Kernel | Memory throughput | Compute throughput | Achieved occupancy | Long-scoreboard share |
| --- | ---: | ---: | ---: | ---: |
| fused KKT/solve | 64.19% | 17.33% | 24.31% | 36.15% |
| persistent state | 91.12% | 16.30% | 24.41% | 54.78% |
| reconstruct/norm/gate | 97.24% | 13.77% | 64.63% | 74.39% |

A bounded attempt to copy shared W/K-decay and Q/QK into larger shared-memory
banks was rejected on the real path: persistent state regressed from
40.842944 to 49.409152 ms and reconstruction from 43.396896 to 45.030272 ms.
The extra copies and lower residency outweighed reuse. The failed route is not
retained.

## Numerical characterization

After the positive performance direction, a non-timed hook copied the full
P512 GDN state for all 48 linear layers from baseline and candidate runs:

| Metric | Result |
| --- | ---: |
| BF16 elements | 37,748,736 |
| unequal BF16 elements | 36,851,249 (97.6224714%) |
| global RMS-normalized error | 0.117148528 |
| maximum absolute error | 2.84375 |
| worst layer RMS-normalized error | 0.120440945 (layer 0) |

All values were finite and both runs retained the first-token semantics, but
this is not an exact-compatibility result. It confirms that Chunk64/FP32-state
Prefill must remain an explicit throughput-mode numerical contract and needs
capability evaluation through the planned API/EvalScope path before any
production decision.

## Reproduction

```bash
cmake -S . -B build/gdn-chunk64-reference \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DQ3X_BUILD_GDN_CHUNK64_REFERENCE_ADMISSION=ON \
  -DQ3X_E2E_MODEL_DIR=/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4
cmake --build build/gdn-chunk64-reference -j2 --target \
  q3x_reference_gdn_prefill_chunk64_reference_engine_e2e_test
Q3X_RUN_GDN_CHUNK64_REFERENCE_ADMISSION=1 \
Q3X_E2E_MODEL_DIR=/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4 \
  ./build/gdn-chunk64-reference/\
q3x_reference_gdn_prefill_chunk64_reference_engine_e2e_test
```

Add `Q3X_GDN_CHUNK64_CHARACTERIZE_STATE=1` for the non-timed state
characterization. Add `Q3X_GDN_CHUNK64_PROFILE_CANDIDATE=1` and wrap the
command with `nsys --capture-range=cudaProfilerApi` to isolate the measured
candidate.

The next whole-project priority is the 496.920736 ms FP8 projection family.
Returning to GDN is justified only by a producer/consumer fusion that removes
the W/U/QK global boundaries or by a new value/head ownership scheme capable
of closing most of the remaining 50.035424 ms budget.
