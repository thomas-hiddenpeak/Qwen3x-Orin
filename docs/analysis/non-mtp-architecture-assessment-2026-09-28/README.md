---
q3x_document:
  id: q3x-non-mtp-architecture-assessment-20260928
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: current installed API profile attribution and bounded next-direction assessment
  effective: 2026-09-28
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: new implementation or evidence requires a successor record
---

# Non-MTP architecture assessment

At `e3f90fd3134ba3fd097bd2af4d943c89e33228be`, the first assessment is complete.
The selected next implementation hypothesis is **separating Down weight decode
from its ordered tensor consumer**, retaining the incumbent finite-precision
operator. This is a bounded opportunity, not an implemented speedup or a claim
that it solves the locked Prefill SLO. MTP remains deferred. The installed
0.8.1 route, accuracy qualification and production defaults are unchanged.

[Machine-readable evidence](../../metadata/qwen36-27b-non-mtp-architecture-assessment-2026-09-28.json)
binds raw captures, source files, reference sources and the reproduction script.
[Roadmap](../../ROADMAP.md) owns subsequent execution, not this frozen record.

## Product anchor and current attribution

Reuse the completed 0.8.1 reliability run, rather than launch another timing
campaign. These are its **single-process observations**, not new mirrored
performance claims; [Current Status](../../CURRENT_STATUS.md) retains the
qualified 0.8.0 mirrored table and the 0.8.1 reliability scope.

| Real API prompt / output | Prefill seconds | Prefill token/s | External TTFT seconds | Decode token/s |
| --- | ---: | ---: | ---: | ---: |
| 8,192 / 256 | 16.753 | 488.99 | 16.773 | 9.220 |
| 40,000 / 256 | 90.964 | 439.73 | 90.999 | 7.867 |

One current installed P40000/O256 Nsight Systems capture answers the missing
attribution question. Binary SHA-256 is
`5ee029cc8877ad0086e5e6ea5bcf8d7674626e55cd96098d518252d33c14aa33`.
The request hash, output text, usage, finish reason and completed stream match
that prior run. Host/device ownership preflight and cache preparation passed;
maximum recorded temperature was 76.968C, clock errors were empty, and owned
server shutdown returned zero. This is T4 diagnostic evidence only. Profiling
changes timing; its numbers do not replace the API performance table.

Kernel attribution joins CUDA correlation IDs to CPU launch-thread NVTX
ranges. Every captured kernel is assigned exactly once, on one device/context/
stream, with no overlapping kernel intervals. The Prefill range is 90.1015 s,
of which kernel intervals sum to 90.0862 s. Kernel-launch count reduction alone
cannot explain away this cost. The finish/head interval is recorded separately.

| P40000 Prefill family | GPU seconds | Share of whole-Prefill NVTX interval |
| --- | ---: | ---: |
| Gate/Up CUTLASS GEMMs | 29.454 | 32.69% |
| FP8 projection CUTLASS GEMMs | 18.417 | 20.44% |
| Persistent packed Down, including residual publication | 17.562 | 19.49% |
| Full-attention Prefill | 13.334 | 14.80% |
| Explicit FP8 and Gate/Up weight decode | 2.163 | 2.40% |
| Remaining kernels and interval remainder | 9.172 | 10.18% |

The first two rows share a generic `Kernel` short name. Grid shape plus source
call topology distinguishes Gate/Up (`[126,34,1]`, 640 calls) from FP8 roles
(1,040 calls). Down has 64 calls, and full attention has 16. Historical profiles
from before the CUTLASS changes cannot supply these current percentages.

| Decode family, 255 steps | GPU ms / step | Share of summed Decode NVTX intervals |
| --- | ---: | ---: |
| Projections and their fused norm/residual/SiLU work | 96.846 | 75.85% |
| QK + Softmax + PV | 23.782 | 18.63% |
| LM head | 4.398 | 3.44% |
| GDN update + convolution | 1.756 | 1.38% |
| Remaining kernels and interval remainder | 0.897 | 0.70% |

Hybrid attention does not make full-attention cost constant. Its 16 layers
still traverse growing KV history; the current ordered PV path preserves the
sequence accumulation order. The observed 23.8 ms/step attention cost is large
enough to matter beside roughly 100 ms/step of other work. This explains why
material context scaling is plausible; **a single 40K profile does not prove
that every millisecond of the short-to-long difference belongs to attention**.
It also does not establish DRAM saturation: no new hardware traffic counters
were collected. Logical loads, unique bytes and measured DRAM traffic remain
separate quantities.

## Why this one implementation hypothesis

The 64-layer Down path is still a persistent packed operand consumer, using
16 CTAs, M64/N256/K64 tiles and four stages. It repeatedly transforms packed
weights and block scales inside the M-tile work. Its N-major raster is not a
proof that the entire decoded operand stays resident across M tiles.

The proposed composition is:

1. Decode one layer's Down operand into bounded request-owned BF16 scratch,
   preserving the incumbent FP4/block-scale operation exactly.
2. Feed an ordered tensor consumer from that representation. Preserve the
   incumbent K partition, MMA sequence, final FP32 merge, global scale and
   BF16 publication before the residual add.
3. Reuse the scratch only after stream-ordered consumption, without adding a
   persistent all-layer BF16 copy or changing request state. Keep actual-row
   padding, cancellation, short-prompt behavior and the arena reserve intact.

A full layer operand is `17408 * 5120 * 2 = 178,257,920` bytes. This is a
scratch requirement to prove against current lifetimes, not permission to
increase the arena. The producer writes approximately 11.41 GB across 64
layers; consumer reads add traffic. The exchange removes repeated decode/feed
instructions while increasing operand bytes, so a win is a hypothesis.

**The numerical trap is concrete.** In the selected Marlin specialization,
`b_sh_stride_threads=128`, `threads=256`, and `red_off=1`. Two warp groups
retain separate FP32 K partials and merge them at the end. A normal sequential
CUTLASS GEMM is not automatically the same floating-point operator. The
incumbent also rounds the globally scaled Down branch to BF16 before adding
the BF16 residual in FP32 and rounding again. Moving either boundary repeats
the kind of scale/rounding mistake that previously invalidated fast paths.
The candidate must prove exact operand and ordered-output equivalence first;
this assessment does **not** claim that proof is already complete.

The affected ceiling is 17.56 s, not the whole 90 s Prefill interval. A net
20% reduction of the complete Down producer/consumer would save about 3.51 s
(3.9% of this profiled phase); a 10% reduction saves only 1.76 s. These are
conditional budget calculations, not forecasts or a new acceptance threshold.
The candidate must include producer, reads, tails and residual costs. It
cannot establish the locked 2-second TTFT target, and that target remains open.

Other opportunities are not simultaneous work packages:

- FP8 weights are decoded five times per role at P40 because of 8K panels.
  Perfectly removing four of those five explicit decodes saves at most about
  1.60 s in this capture, before any added lifetime/layout cost. Useful later,
  but not the selected primary change.
- Paired Gate/Up publication could remove a logical 356.52 GB of intermediate
  writes/reads across P40 and 64 layers. This is not measured DRAM traffic;
  preserving CUTLASS reduction order and pairing the epilogue needs its own
  resource/dataflow design. The standalone SiLU kernel costs only 1.387 s.
- More PV CTAs or independent sequence accumulator chains repeat already
  rejected lineages; reassociating the ordered reduction also changes the
  numerical operator. Neither is reopened as a parameter scan.

## Reference translation

Source pins/hashes are in the machine-readable record. No external framework
was rerun and no current matched vLLM speedup or hardware bound is claimed.

| Studied source | Transferable mechanism | Boundary for this project |
| --- | --- | --- |
| vLLM Marlin and the selected local specialization | Packed operand feed, asynchronous stages, explicit K-partial merge and output rounding | Preserve actual numerical order; changing warp ownership can change the operator |
| Humming producer/consumer/S2R/MMA pipeline | Separate representation transform, transfer, consumption and epilogue lifetimes | Translate to SM87; TMA/PDL or newer-device facilities are not assumed available |
| Current CUTLASS multistage projection route | Decoded BF16 operand with tensor scale after FP32 accumulation already works in current Gate/Up and FP8 paths | Evidence of a viable representation trade, not proof of Down equivalence or speed |
| FlashInfer Prefill | Tiled causal attention and bounded shared-memory Q/K/V flow | Already represented in the current Prefill family; do not silently substitute its arithmetic for ordered Decode |
| vLLM's Triton/FLA chunk GDN | Separate cumulative gates, KKT/solve, W/U, state propagation and output phases | Current chunked GDN already follows this structure; its modest current budget does not select another rewrite |
| Mamba SSD | Separate intra-chunk work, state passing and inter-chunk composition with FP32 state | A state/dataflow reference, not an interchangeable GDN recurrence |

## Historical mistakes that constrain the next step

The [Decode retrospective](../decode-performance-lessons-2026-09-27/README.md)
and [numerical repair](../prefill-reference-repair-2026-09-28/README.md) retain
the detailed evidence. This audit adds the current-path comparison, rather
than rewriting old results:

| Earlier mistake or rejected direction | Consequence for this work |
| --- | --- |
| Calling a local 5–6 token/s plateau a hardware ceiling; unmatched loader/configuration comparisons | Keep owner targets; bind exact API, artifact and protocol before causal claims |
| Error divided by the wrong reference norm; diverged token histories; sampled logits/KV treated as full correctness | Same-input full-boundary checks and independently assessed references remain mandatory |
| Transposed A/B `ldmatrix` coordinates; tensor scale before the correct accumulation boundary | Verify physical operand mapping and every rounding boundary before timing |
| Rewriting all projections to packed kernels because fewer bytes looked better | The August projection-reset and packed-v2 attempts lost at the real API; register/feed/shared-memory costs matter |
| Higher occupancy or more CTAs assumed to imply faster execution | Packed projection and later 64-CTA PV failures remain closed; no grid sweep follows this assessment |
| Logical traffic presented as physical DRAM traffic; historical profile shares reused after kernel replacement | Label byte arithmetic and use the current installed capture for current attribution |
| Independent local gains added together; stale executable or test selector treated as production | Compose one complete producer/consumer and judge the installed API; no silent default change |

## Bounded return to the product

The assessment stops here: its one profile has answered the attribution
question. The next implementation has at most one initial dataflow and one
causally justified correction, not a tile search. Numerical admission precedes
real-payload timing. If exact arithmetic cannot be retained, reject this
version rather than widen the accuracy tolerance or reopen the reference.

An admissible complete candidate returns to short, 8K and 40K production-shaped
API requests immediately, reporting Prefill, TTFT and Decode together. A
negative or indistinguishable composed result closes it. A positive direction
unlocks the affected full-range/tail, complete numerical, capability,
resource and installed-artifact qualification under the existing policy; it
does not itself promote production or qualify future 60K/130K capacity.
