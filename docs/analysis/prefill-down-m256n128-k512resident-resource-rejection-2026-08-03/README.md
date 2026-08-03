# Prefill Down M256N128 K512-resident resource rejection (2026-08-03)

## Decision

The direct dense-A4/K512 `M256N128` Down cell is rejected before runtime
admission. It is default-off, is not selected by `reference_runner`, and has
not entered the OpenAI API or EvalScope production path.

This is a resource rejection, not a performance result. The final kernel
still uses local memory after the only two structural ownership corrections,
so a real-model timing would measure register spill plus a single-buffered
pipeline rather than the intended B-resident architecture.

## Intended qualitative change

For the pinned Down shape at the real P1853 direction gate:

- logical/launch M: `1853/1920`;
- N/K: `5120/17408`;
- incumbent cell: `M128N128`, `15 * 40 = 600` work tiles;
- candidate cell: `M256N128`, `8 * 40 = 320` work tiles;
- one CTA owns two incumbent M128 panels and presents each N128 B panel once.

The logical B presentations therefore fall from 15 to 8, or 46.7%. A remains
unchanged because the zero-filled rows in the final M256 tile issue
`cp.async` with a zero source size.

Including packed codes and BF16 K512 scale planes, the ideal requested global
operand bytes per layer are:

| route | A bytes | B bytes | total bytes |
|---|---:|---:|---:|
| incumbent M128N128 | 673,689,600 | 673,689,600 | 1,347,379,200 |
| candidate M256N128 | 673,689,600 | 359,301,120 | 1,032,990,720 |

The structural traffic ceiling is therefore a 314,388,480-byte reduction
(23.33%, or at most 1.304x if nothing else changes).

## Resource evidence

The candidate retains the incumbent dense-A4 numerical order: eight physical
K64 MMA planes form one S32 K512 partial, then BF16 A/B scales update the
persistent FP32 output. That requires both persistent FP32 C and transient
S32 partial state.

The bounded implementation sequence was:

| ownership/lifetime | registers/thread | runtime local | spill stores | spill loads |
|---|---:|---:|---:|---:|
| 8 warps, M128N32/warp | 255 | 208 B | 400 B | 240 B |
| 8 warps, N16 partial phase | 255 | 56 B | 104 B | 72 B |
| 16 warps, M64N32/warp | 128 | 208 B | 408 B | 300 B |
| 16 warps, named Outputs/Partials | 128 | 144 B | 392 B | 268 B |

The final 16-warp cell has:

- 512 threads and 16 resident warps;
- 99,072 B dynamic shared memory;
- one active CTA per SM;
- 64 registers/thread of persistent FP32 outputs;
- 32 registers/thread of live N16 S32 partials before operand, address, and
  loop state;
- 144 B local memory/thread after explicit named scalarization.

The runtime resource query therefore returns
`cudaErrorLaunchOutOfResources`. The focused test stops before its synthetic
bitwise body, as required by the zero-local admission gate.

The P1920 launch also rounds the final M256 cell to 2048 computed rows. Loads
and stores are masked, but MMA is not, so the candidate performs 6.67% more
MMA work than the M128 incumbent. In addition, its 99,072-byte full-K512
shared cell has no space for a next-K512 buffer; it executes
`load -> wait_all -> barrier -> compute -> barrier`, while the incumbent uses
a pair-ring pipeline. N16 partial phasing also repeats A `ldmatrix` traffic.
Those costs make spill timing actively misleading.

## Verification

Focused Release build and resource execution:

```text
resources regs=128 static_shared=0 dynamic_shared=99072 local=144
          max_threads=512 active_blocks_per_sm=1
PASS: expected zero-local resource rejection before correctness
```

`git diff --check` passes. There is deliberately no OpenAI API/EvalScope
number for this rejected binary: it cannot pass the prerequisite resource and
correctness gates and has no runner route.

## Architecture consequence

The B-reuse objective is retained, but it cannot be implemented as an
isolated tile enlargement under the current K512 scale boundary. The current
contract forces persistent FP32 C and repeated S32 partials to coexist.

The next complete package must couple:

1. factorized scale lanes, so S32 partial lifetime crosses far fewer scale
   boundaries;
2. shape-specific wider-M ownership: M256 for R1, where only S32 output is
   live, and M192 for R4, where S32 lane partials and FP32 cross-lane output
   must coexist without local memory;
3. a zero-local output lifecycle and an overlapped load/compute pipeline;
4. real checkpoint conversion with authenticated inverse activation factors;
5. OpenAI-compatible serving followed immediately by external
   `evalscope[perf]==1.9.1` P1804 warmup/P1853 measurement.

R1 remains only a performance upper-bound route and uses a new M256 kernel
that accumulates the complete K dot in S32 before one scale application. R4 is
the quality candidate and uses M192N128/12 warps so its 64 S32 partials plus
64 FP32 outputs have a credible zero-local register budget. Neither may enter
production without real-model capability and end-to-end performance
admission. The direct exact-K512 M256 skeleton in this record must not be wired
to the runner. Its public launcher enforces the same resource query as the
focused test, so the archived spill binary fails closed if called directly.
