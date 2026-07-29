# P513 prompt-wide embedding admission

Date: 2026-07-30

The incumbent prefill path launched one embedding gather per prompt token.
The admission copies the validated prompt token IDs to existing device
workspace once and launches one CTA per token in a single prompt-wide kernel.
The gathered BF16 rows remain bitwise copies of the scalar path.

## Real-model B-C-C-B gate

Model: `/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4`

`B` is the scalar-launch embedding path and `C` is the prompt-wide path. The
table records the measured native-candidate part of each real P513 generation
process; all four generated token 9419 (`Hello`).

| order | route | prefix ms | TTFT ms |
|---:|:---:|---:|---:|
| 1 | B | 1739.191647 | 1847.846220 |
| 2 | C | 1736.697884 | 1845.424780 |
| 3 | C | 1737.312256 | 1845.955295 |
| 4 | B | 1737.954605 | 1846.738773 |

Both endpoint pairings are positive:

- pair 1: prefix -2.493763 ms; TTFT -2.421440 ms
- pair 2: prefix -0.642349 ms; TTFT -0.783478 ms
- paired mean: prefix -1.568056 ms; TTFT -1.602459 ms

This is the incremental admission gate against the native incumbent, not a
claim of vLLM parity.

## Correctness and Graph gate

`decode_ops_cuda` passes for M=1,2,7,8,31,32,64,407,481,511,512. The test
compares every output bit with scalar gathers, checks boundary and repeated
token IDs, immutable inputs and guards, validates rejected range/alias cases,
and replays the single-kernel CUDA Graph twice. Device-side invalid IDs retain
their output sentinel; the runner validates all IDs on the host before launch.
