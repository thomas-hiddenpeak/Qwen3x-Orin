# REJECTED: GDN chunk-o direct-async structural candidate

Date: 2026-07-30

## Decision and authority

**REJECTED.  Do not merge or cherry-pick this kernel into any production
path.**  The real-checkpoint P513 generation gate was both much slower and
semantically wrong.  The synthetic component PASS was therefore insufficient
and has no admission authority.

This experiment is based on cumulative commit `85f5f01`.  Commits `7f7f735`
and `7f2c02a` are development/diagnostic skeletons only; neither is an
admissible implementation.  The real model and complete generation path take
priority over fragment tests, resource tables, and structural similarity to a
reference artifact.

## Real-weight production-path rejection

The authenticated Qwen3.6-27B-NVFP4 P513 request produced:

| route | token / text | Prefix ms | result |
|:---|:---|---:|:---|
| incumbent | 9419 / `Hello` | 1,255.367 | semantic PASS |
| direct-async | 71093 / <code>&#96;&#96;&#96;</code> | 1,579.420 | semantic **FAIL** |
| delta | — | **+324.054** | strong regression |

The candidate was about 25.8% slower by these rounded Prefix values and
changed the generated token/text.  This immediately terminates the direction:
there is no further performance profiling, parameter scan, external EvalScope
run, or attempt to repair it.  Its compiled 162 registers/thread, 24 KiB
shared memory and three-CTA/SM bound did not translate into either correct
execution or a real-path gain.

The comparison source is the installed vLLM/FLA implementation at:

```text
/home/rm01/setup/build4all/vllm/vllm/model_executor/layers/fla/ops/chunk_o.py
```

The exact P513 Triton artifact previously selected on this Orin is:

```text
/tmp/q3x-vllm-ncu-home.UD85AH/triton/
  V5WMRM4WIEZLV2ATT7BR2HK7WSITEDQQR53MRPDGLI7SDRC75NOQ/
  chunk_fwd_kernel_o.cubin
```

Its SHA-256 is
`af1a2e9556e74388d4de5983bb1adc351fafd239e9e1393d15ff7072821cbaa3`.
Metadata fixes the relevant configuration at `BT=BK=BV=64`, 128 threads,
four warps, two compiler stages, 162 registers/thread and 24 KiB shared
memory.  The frozen P513 NSys reference is 48 launches and 10.069408 ms.

## Incumbent data-flow gap

The incumbent already fuses the correct three mathematical terms in one CTA:

```text
O = exp(gq) * (Q @ H^T)
    + BF16(causal(exp(gq-gs) * (Q @ K^T))) @ Vnew
```

It does not materialize a global score or partial-output matrix.  The gap was
therefore below that high-level formula:

1. Q used `cp.async`, but K and H were loaded through registers and scattered
   element-by-element into a physically transposed shared tile.
2. The two BK64 panels were completely unrolled, duplicating the full
   load/MMA/control body in SASS.
3. Each adjacent pair of N8 B fragments used two separate `ldmatrix.x2`
   instructions and duplicate address arithmetic.
4. Each pair of BF16 results was rounded as two scalar conversions and only
   then packed, preventing the native packed BF16 conversion selected by
   Triton.

Those mechanisms are coupled: changing only the load cache operator cannot
remove the register transpose, and changing only occupancy cannot repair the
instruction/data lifetime.

## Rejected candidate data flow

Grid and ownership remain `(2, chunks, 48)` and one BV64 half per CTA, so the
candidate preserves the incumbent scheduling phase and exact external
boundaries.

For each BK64 panel:

1. Q `[M,K]`, K `[N,K]`, and H `[N,K]` are each copied in coalesced 16-byte
   transactions directly to the three swizzled 8 KiB shared tiles.
2. K/H stay in canonical `[N,K]` row-major storage.  That byte layout is also
   the logical `[K,N]` column-major MMA B operand, so the non-transposing
   `ldmatrix` form produces B fragments without a transpose boundary.  The
   `.trans` form is reserved for V's true `[K,N]` row-major backing.
3. One `ldmatrix.x4` supplies two adjacent N8 K/H fragments; one
   `ldmatrix.x4.trans` does the same for V.  Q/A retain the natural
   `ldmatrix.x4` M16K16 load.
4. The BK loop is retained in SASS and executes twice.  It has the same
   dynamic tensor-core work as the incumbent, but only one static body.
5. Gated scores and raw output are converted with one packed BF16x2 RNE
   instruction per pair.  For finite model values this is bit-identical to
   the former two scalar RNE conversions.

The existing 16 KiB FP32 QH layout exchange, causal BF16 score boundary, and
separate exact rows-8 RMSNorm+SiLU epilogue remain unchanged.  This isolates
the candidate to the on-chip load/fragment skeleton.

## Intended tail contract

The kernel remains fixed C64 internally.  For a final partial chunk, the
producer contract is the FLA identity-padding contract already implemented by
the arbitrary-length native candidate:

- padded Q/K/V rows are zero;
- padded beta/update rows are zero;
- padded log-decay is zero, so gamma lane 63 repeats the last real gamma;
- boundary state is a complete D128-by-D128 matrix for every chunk;
- raw output may be written through padded lane 63 in the fixed 512-token
  internal workspace, while the rows-8 epilogue publishes only
  `token_count * 48` real rows.

Consequently the direct-async loads consume producer-owned padded buffers as a
full C64 tile.  Launch uses `ceil(token_count/64)` and selects one of two
compile-time specializations: the full-C64 specialization has no validity
instructions, while the partial specialization explicitly zeros score entries
whose query or source is outside `token_count` and suppresses padded raw-output
stores.  Thus real query/source masking does not depend on the producer zeros,
while the common full-C64 production path pays no tail predicate cost.

## Static evidence (non-admission)

Release CUDA 13.3 SM87 compilation completed for `q3x_kernels` and
`q3x_gdn_prefill_chunk_o_bv64_component_test`, and the complete native-engine
E2E executable.

| property | incumbent `85f5f01` | vLLM selected artifact | candidate |
|:---|---:|---:|---:|
| threads | 128 | 128 | 128 |
| registers/thread | 168 | 162 | 162 |
| shared memory | 24,576 B | 24,576 B | 24,576 B |
| local/stack | 0 B | 0 B | 0 B |
| static CTA/SM register bound | 3 | 3 | 3 |
| kernel text end | `0x8780` | `0x55d0` | `0x5100` |
| static HMMA sites | 160 | 96 | 96 |
| static B-fragment `ldmatrix.x4` sites | 0 | 52 | 40 |
| static A-fragment `ldmatrix.x4` sites | 16 | included above | 12 |
| packed BF16x2 conversions | 64 | 31 | 32 |

The candidate's 96 static HMMA sites are not a reduction in dynamic math:
the 64-site QH/QK body executes for two BK panels and the final 32 sites
execute once, totaling the same 160 dynamic MMA operations per warp path as
the incumbent.  The reduction is instruction-cache/control duplication.

## Component correctness (insufficient)

The first x4 draft incorrectly used `.trans` for canonical `[N,K]`, which
double-transposed the logical B operand.  The expanded B4 sentinel rejected
all 128 B registers and the QH/QK component cases failed, so no real model was
run.  After selecting non-transposing x4 for canonical K/H while retaining
x4.trans for true-row-major V, the locked component run passed:

```text
GDN_CHUNK_O_BV64_FRAGMENT_A unequal=0 gate=PASS
GDN_CHUNK_O_BV64_FRAGMENT_B unequal=0 gate=PASS
GDN_CHUNK_O_BV64_FRAGMENT_MMA unequal=0 gate=PASS
GDN_CHUNK_O_BV64_KN_TRANSPOSED_QH_RAW unequal=0 gate=PASS
GDN_CHUNK_O_BV64_KN_TRANSPOSED_QK_V_RAW unequal=0 gate=PASS
GDN_CHUNK_O_BV64_PARTIAL_QUERY_MASK unequal=0 gate=PASS
GDN_CHUNK_O_BV64_ROWS8_NORM unequal=0 gate=PASS
GDN_CHUNK_O_BV64_COMPONENT_RESULT gate=PASS
```

The fixed SASS had exactly the selected Triton artifact's B/A fragment
site split: 44 non-transposing `LDSM.M88.4` and eight transposing
`LDSM.MT88.4` sites.  The subsequent real-model semantic failure demonstrates
why matching instruction shape and passing isolated fragments cannot establish
the actual model-layout/numerical contract.

No further work is authorized on this candidate.  It is retained only as a
negative result that rules out this implementation skeleton.
