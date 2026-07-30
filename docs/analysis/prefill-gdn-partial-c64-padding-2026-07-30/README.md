# Prefill GDN arbitrary-tail C64 padding candidate

Date: 2026-07-30

## Scope and decision

This candidate starts from cumulative commit `85f5f01`. It removes the
floor-C64 native prefix plus token-serial legacy tail inside an admitted GDN
Prefill tile. The native implementation now executes `ceil(T/64)` chunks for
arbitrary `T` through C512, masks invalid rows at every producer and consumer,
and commits state/output only for the true token count.

The low-level launch contract supports C1..C512 for correctness coverage. The
real production selector is deliberately C32..C512. A one-token public request
and the C1 tail of the ordinary P513 scheduler both remain on the existing
scalar runner path, so neither supplies route-hit evidence for bulk-C1
admission. This is a scheduler boundary, not a claimed C1 performance result.

No cuBLAS, cuBLASLt, Triton, FlashLinearAttention, or vLLM implementation is
introduced into the runtime. FLA's tail-mask semantics are used only as the
reference contract for neutral padding.

## Neutral-padding contract

For every padded row `t >= T`, the compact producer and native hierarchy use:

- Q/K/V/W/U and the raw gate input equal to zero;
- beta equal to zero;
- the cumulative gate extended from the final real row through lane 63;
- no normalized output store and no public state/output update for the padded
  row.

The final native state boundary is therefore the true last token even though
the fixed-shape state kernel reads chunk lane 63. WY recompute loads, compact
Q/K normalization, BV64 chunk output, rows-8 normalization, and the fused
convolution producer all receive the true token count and mask their padded
work. The runner no longer launches a serial legacy recurrence for the
remainder of an admitted C32..C512 tile.

## First real-weight direction screen

The screen uses the authenticated
`/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4` checkpoint, one warmup,
one measured request, one output token, and the complete cumulative production
environment. Each comparison is old commit `85f5f01` followed immediately by
the candidate under the same GPU lock. `SINGLE_ARBITRARY` is enabled for the
exact C32/C52/C481 cells so each request is one native tile.

| exact tile | old Prefix ms | candidate Prefix ms | saved ms | Prefix ratio | old/new native hits |
|:---|---:|---:|---:|---:|---:|
| C32 | 145.542382 | 123.664089 | 21.878293 | **1.17691x** | 0 / 48 |
| C52 | 204.426188 | 160.216862 | 44.209326 | **1.27593x** | 0 / 48 |
| C481 | 1173.695009 | 1145.756356 | 27.938653 | **1.02438x** | 48 / 48 |

TTFT savings are respectively 21.858100, 44.211534, and 27.888604 ms. Every
valid comparison passes the route structure and real-model semantic oracle.
The positive C481 result isolates removal of its 33-token serial tail; C32 and
C52 prove that a formerly all-legacy span benefits from the padded native
hierarchy.

## Ordinary P513 boundary audit

The ordinary full-prompt scheduler was also run without
`Q3X_RUN_PREFILL_SINGLE_ARBITRARY_TILE_ADMISSION`. It decomposes the request at
the C512+C1 boundary. The `85f5f01` native candidate measured:

- Prefix 1240.732976 ms;
- TTFT 1245.211641 ms;
- 48 native route hits.

The new binary's adjacent warmup measured Prefix 1242.073005 ms and TTFT
1246.516694 ms with the same 48 hits and exact token 9419 / text `Hello`.
It did not report the 96 hits that bulk-C1 admission would require, so the
harness rejected that attempted expectation as structurally invalid and did
not assign timing authority to it. The evidence says the actual scheduler
keeps C1 on the scalar route and that this patch neither improves nor regresses
that boundary. This is why the selector is frozen at C32 rather than C1.

## Authority and next gate

These numbers are a first real-runner direction screen, not final production
promotion evidence. They justify retaining the structural candidate and
advancing it to the external OpenAI-compatible EvalScope performance gate.
Synthetic matrices remain limited to exhaustive mask, redzone, invalid-call,
CUDA Graph, and numerical correctness work; they have no performance authority.
No vLLM rerun is needed: the frozen external vLLM baseline remains the parity
reference.
