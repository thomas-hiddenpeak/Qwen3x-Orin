# P513 tiny-tail scheduler rebalance — rejected

Date: 2026-07-30

This experiment changed the production single-arbitrary scheduler from
`M512 + M1` to `M449 + M64` for a 513-token prompt.  Its purpose was to test
whether avoiding a second small-M projection traversal could remove the
observed 106-ms final tile without changing the number or order of tokens.

The first authority was one real-checkpoint generation on the complete native
runner, not a synthetic kernel benchmark.  Both processes used the cumulative
`85f5f01` production admissions, the authenticated
`Qwen3.6-27B-NVFP4` checkpoint, P513, greedy one-token generation, and emitted
token 9419 (`Hello`).

| route | Prefix tiles ms | TTFT ms |
|:---|:---|---:|
| cumulative baseline | 1138.687, 105.441 | 1249.333 |
| M449 + M64 candidate | 1118.653, 183.201 | 1307.131 |
| candidate delta | -20.034, +77.760 | **+57.798** |

The large first tile improved slightly, but the second M64 traversal cost far
more than the incumbent M1 path.  The candidate is rejected before EvalScope,
NCU, or broader validation.  The scheduler unit test passed; this establishes
that the negative result is the intended route rather than a malformed tile
sequence.

The result redirects the work to a masked partial-C64 GDN implementation that
keeps one projection tile and removes the serial GDN tail, instead of moving
the public layer-major boundary.
