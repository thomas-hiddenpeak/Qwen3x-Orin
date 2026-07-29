# P513 GDN token-parallel convolution admission

Date: 2026-07-30

This admission replaces the serial one-thread-per-channel C512 convolution
loop with a C8 token tile per CTA while preserving the established BF16 and
FP32 arithmetic boundaries.  The raw QKV projection is kept immutable and the
convolution output is written to a disjoint existing projection workspace.

## Real-model direction gate

Model:
`/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4`

The gate used the real P513 generation path and generated token 9419 (`Hello`)
for both routes.  The values below are the measured native-candidate samples
from a B-C-C-B process order.  `B` is the incumbent whole-span convolution and
`C` is the token-parallel admission.

| order | route | prefix ms | TTFT ms |
|---:|:---:|---:|---:|
| 1 | B | 1756.299703 | 1865.023910 |
| 2 | C | 1742.104600 | 1851.145300 |
| 3 | C | 1743.858369 | 1853.137146 |
| 4 | B | 1758.119225 | 1867.448473 |

Paired endpoint improvements:

- pair 1: prefix -14.195103 ms; TTFT -13.878610 ms
- pair 2: prefix -14.260856 ms; TTFT -14.311327 ms
- mean: prefix -14.227980 ms; TTFT -14.094969 ms

Both pairings are positive.  These figures are an incremental admission gate
against the incumbent native path, not a claim of statistical confidence or
vLLM parity.

## Reproduction

Incumbent:

```bash
env Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1 \
  Q3X_GDN_CHUNK64_SKIP_STATE_CHARACTERIZATION=1 \
  build/orin-release/q3x_reference_gdn_prefill_chunk64_native_engine_e2e_test \
  /home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4
```
Candidate:

```bash
env Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1 \
  Q3X_GDN_CHUNK64_SKIP_STATE_CHARACTERIZATION=1 \
  Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1 \
  build/orin-release/q3x_reference_gdn_prefill_chunk64_native_engine_e2e_test \
  /home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4
```
