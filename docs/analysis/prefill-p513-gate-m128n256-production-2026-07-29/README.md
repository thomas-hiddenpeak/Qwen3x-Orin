# P513 C512 M128xN256 Gate/Up production profile — 2026-07-29

Commit `55da501` promotes the previously retained native M128xN256 NVFP4
Gate/Up cell for the exact C512 shape. C256 remains on M128xN128. No external
library, persistent sidecar, request workspace, Decode path, or MTP route is
added.

The fixed-clock, real-model P513 B-C-C-B result is:

| Measure | Baseline `b6741b0` | M128xN256 production | Change |
|---|---:|---:|---:|
| Prefix | 2559.131 ms | 2331.928 ms | 1.097431x |
| Prefix throughput | 200.067914 tok/s | 219.560810 tok/s | +9.743% |
| TTFT | 2685.244 ms | 2457.942 ms | 1.092477x |

Every run generates token `9419` (`Hello`) with 513 committed steps. Pinned
layer-0 Gate and Up each compare 8,912,896 BF16 values with zero mismatches
through eager and CUDA Graph replay.

The production NSys profile contains 128 calls to
`nvfp4_w4a16_gate_c512_m128_n256_bswizzle_scale512_3stage_256t_kernel`,
totalling 605.830496 ms. The prior checked-in production profile totals
839.867392 ms for Gate/Up, so the kernel substitution removes 234.036896 ms.
The complete Prefix NVTX wall falls by 226.859360 ms and the complete kernel
sum by 232.516160 ms, closing the attribution against the 227.203-ms mirrored
end-to-end saving.

The exact kernel aggregation is in
[prefix-kernel-top20.csv](prefix-kernel-top20.csv). It uses the same query as
[the previous P513 audit](../prefill-p513-nsys-2026-07-28/prefix-kernel-top20.sql).
Raw NSys report and SQLite files remain outside git; their paths, sizes, and
hashes are frozen in the accompanying metadata record.
