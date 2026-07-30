# FlashInfer single-Prefill split-KV audit for the 40K target

Date: 2026-07-31

This is a source and launch-contract audit. It does not change a production
selector, allocate GPU memory, or contain a GPU performance result. The code
base is `a6bb6eebb00acfd32ea1c6b9c318f6ea6584166e`; the directly compiled
FlashInfer header subset is `flashinfer-python==0.6.12`, as pinned by
`third_party/flashinfer/README.q3x.md`.

## Decision

Passing a non-null `tmp` pointer to the current FlashInfer wrapper is safe to
prototype, but it **does not split a C512 Prefill tile** on the target Orin.
C512 already launches 192 attention CTAs for one KV chunk, while the exact
compiled kernel can keep only 32 CTAs resident across the 16-SM device. The
vendored auto-policy consequently computes zero available extra KV chunks and
selects the unsplit path independently of whether `tmp` is null.

The immediate 40K path should therefore remain:

1. enable the already admitted arbitrary-position FlashInfer unsplit path;
2. validate its real P2K/P4K/P8K/P16K/40K scaling;
3. consider a forced two-way C512 split only if the unsplit long-context
   profile shows an attention-kernel utilization or tail-latency problem.

Stock auto split-KV is useful only for very small final query tails on this
fixed shape. It is not the missing structural change for the 40K body.

## Exact vendored interface

The Q3x wrapper constructs
`SinglePrefillParams<bf16,bf16,bf16>` with:

- Q: `[qo_len,24,256]` BF16, NHD stride `[6144,256]`;
- K/V: `[kv_len,4,256]` BF16, NHD stride `[1024,256]`;
- `kv_len = first_position + qo_len`;
- causal mask, position encoding `NONE`, FP32 QK reduction, scale `1/16`;
- final LSE pointer null.

It then calls:

```cpp
flashinfer::SinglePrefillWithKVCacheDispatched<
    256, 256, flashinfer::PosEncodingMode::kNone, false,
    flashinfer::MaskMode::kCausal,
    flashinfer::DefaultAttention<false, false, false, false>>(
        params, tmp_bf16, stream);
```

The current wrapper passes `tmp_bf16=nullptr`. The vendored function has no
workspace-size argument and performs no bounds check. A caller must reserve
enough storage for whatever `num_chunks` its internal occupancy query chooses.
The upstream Python single-Prefill wrapper handles this by allocating a fixed
32-MiB byte tensor (`SINGLE_KERNEL_TMP_SIZE`), not by asking the CUDA launcher
for an exact size.

When split-KV is selected, the contiguous scratch layout is:

```text
partial_v   BF16 [qo_len, num_chunks, 24, 256]
partial_lse FP32 [qo_len, num_chunks, 24]
```

`partial_lse` starts immediately after
`num_chunks * qo_len * 24 * 256` BF16 elements. For this fixed model the
required byte count is therefore:

```text
num_chunks * qo_len * 24 * (256 * sizeof(BF16) + sizeof(float))
= num_chunks * qo_len * 12,384 bytes.
```

The attention kernel writes one normalized BF16 partial output and one FP32
base-2 LSE per partition. `flashinfer::MergeStates` then merges those states
into the final BF16 output. Because every partial output has already crossed a
BF16 boundary, forced split-KV is not expected to be bitwise identical to the
unsplit path; it needs the established finite/error/token accuracy gates.

For the common `num_chunks < qo_len` case, `MergeStates` launches one CTA per
query token, with block `(32,24,1)` for D256 BF16. For forced C512/S2 this is a
512-CTA merge in addition to the attention and sigmoid-Gate kernels.

## Exact auto-policy on SM87

For D256, `FA2DetermineCtaTileQ` cannot select Q128. For C512 it selects:

| property | value |
|---|---:|
| grouped packed query length | `512 * (24/4) = 3072` |
| `CTA_TILE_Q` | 64 |
| `NUM_WARPS_Q / NUM_WARPS_KV` | 4 / 1 |
| `NUM_MMA_Q / NUM_MMA_KV` | 1 / 2 |
| threads | 128 |
| dynamic shared memory | 65,568 B |
| registers/thread in retained cubin | 239 |
| CTAs per KV chunk | `ceil(3072/64) * 4 = 192` |

The target evidence records 16 SMs and 166,912 B shared memory per SM. Two
blocks consume 131,136 B shared memory and 61,184 registers; a third block
exceeds both resource budgets. The device-wide resident capacity is therefore
`2 * 16 = 32` CTAs.

The vendored planner computes:

```text
max_num_kv_chunks = resident_CTA_capacity / CTAs_per_KV_chunk
                  = 32 / 192
                  = 0
num_chunks        = 0
```

and enters the `num_chunks <= 1` unsplit branch. This remains true for any
`kv_len`, including 40,960.

The same retained ELF was inspected without running the GPU:

- ELF SHA-256:
  `93d3e4d799c1f7ddc55cf0adeffb203f312a4cb79fed05108d0ab8d671d0e54b`;
- attention object SHA-256:
  `62cf2cbe2cb7074240bfcf2a28fe393b1d4505c849936856c59e6efea246fc25`;
- `cuobjdump --dump-resource-usage` reports 239 registers for the selected
  Q64/MMA-KV2 specialization;
- the existing CUDA graph contract independently fixes the dynamic shared
  size at 65,568 B and the unsplit grid at `(48,1,4)`.

## Small-tail auto split and exact global capacity

The same fixed kernel has two resident CTAs per SM for both Q16 and Q64
specializations. At `kv_len=40,960`, the stock planner produces the following
representative results:

| `qo_len` | Q tile | CTAs/chunk | max chunks | actual chunks | scratch B |
|---:|---:|---:|---:|---:|---:|
| 2 | 16 | 4 | 8 | 8 | 198,144 |
| 10 | 64 | 4 | 8 | 8 | 990,720 |
| 21 | 64 | 8 | 4 | 4 | 1,040,256 |
| 22 | 64 | 12 | 2 | 2 | 544,896 |
| 42 | 64 | 16 | 2 | 2 | 1,040,256 |
| 43 | 64 | 20 | 1 | 1 | 0 |
| 64 | 64 | 24 | 1 | 1 | 0 |
| 128 | 64 | 48 | 0 | 0 | 0 |
| 256 | 64 | 96 | 0 | 0 | 0 |
| 512 | 64 | 192 | 0 | 0 | 0 |

Thus stock auto split is reachable only through query tails of at most 42
tokens. A model- and device-specialized workspace covering every legal
`qo_len=2..512` and `kv_len<=40,960` needs 1,040,256 usable bytes. A request
plan that aligns the following region to 256 bytes advances its arena cursor
by 1,040,384 bytes.

The safer first plumbing implementation can simply reserve the upstream
32-MiB contract. It avoids coupling memory safety to a reproduced occupancy
formula and increases the current 40,960 request arena from 2,864,349,184 B
to 2,897,903,616 B, still below the proposed 3-GiB request cap. Only one
workspace is required globally because all 16 full-attention layers and all
query tiles execute in dependency order on the same stream. Per-layer
allocation would waste `16x` memory. Future concurrent attention streams
would require one workspace per concurrently live launch, not per model
layer.

## Forced C512 split sizes and screening policy

The stock dispatcher overwrites `params.partition_kv` and derives its own
chunk count, so setting that field before calling it cannot force a split. A
forced fixed-shape experiment must either add an explicit override upstream or
launch the known `KernelTraits<Causal,Q64,MMA-KV2,D256,...>` specialization
directly and then call `MergeStates`.

Exact C512 scratch sizes are:

| forced split count | scratch B | MiB | storage option |
|---:|---:|---:|---|
| 2 | 12,681,216 | 12.09375 | fits dead `projection[2]` (17,825,792 B) |
| 4 | 25,362,432 | 24.1875 | needs a dedicated region; fits 32 MiB |
| 5 | 31,703,040 | 30.234375 | fits upstream 32 MiB |
| 8 | 50,724,864 | 48.375 | exceeds upstream 32 MiB |
| 16 | 101,449,728 | 96.75 | reject before evidence |

Start with S2 only, and only after an unsplit long-context profile. C512/S1
already has 192 balanced CTAs; at long append positions the per-CTA causal
work differs by at most 511 keys over a roughly 8K--40K prefix. Splitting does
not reduce QK/PV arithmetic or aggregate KV traversal. It adds repeated Q
loads, BF16 partial-state writes, FP32 LSE writes, a full merge read/write,
one kernel launch, and an extra numerical boundary. Consequently S2 is a
latency/utilization probe, not an assumed optimization. S4 has no authority
unless S2 is positive and profiling shows remaining under-utilization.

The promotion order should be:

1. same-ELF real API, max-output-1, P8K then P16K; reject immediately if S2
   is negative;
2. if positive, P40K and a short P512/P2K non-regression check;
3. NSys confirms the expected `attention -> MergeStates -> Gate` topology and
   no generic-QT2 calls;
4. NCU compares S1/S2 achieved occupancy, tensor utilization, memory
   throughput, long-scoreboard stalls, and merge traffic;
5. full numerical tensor metrics, nonfinite checks, exact greedy token, then
   external capability evaluation.

## Integration map

Low-risk stock-auto plumbing:

1. Add an additive workspace-aware launch entry rather than changing the
   existing public function in place. The old entry forwards `nullptr,0` and
   retains ABI and graph behavior.
2. Thread `(void* workspace, size_t workspace_bytes)` from
   `ReferenceRunner::Views` into `launch_bulk_gqa_flashinfer_direct`.
3. Before passing a non-null pointer to the size-unaware FlashInfer API,
   require either the full 32-MiB upstream contract or a checked fixed-Orin
   planner result. Insufficient storage must fall back to `tmp=nullptr`; it
   must never launch split-KV with an undersized region.
4. Keep attention, MergeStates, and Gate on the same stream. The scratch may
   be reused immediately after Gate or after the next ordered consumer.
5. Extend cold/warm CUDA graph tests: unsplit C512 remains two nodes;
   auto-split small tails become three nodes and must verify MergeStates
   identity and guards.

For a forced S2 experiment, no arena change is initially necessary:
`projection[2]` is dead after full-attention Q/K/V preprocessing and remains
dead until the subsequent MLP. Its 17,825,792-byte contiguous capacity fits
C512/S2. This liveness reuse must be documented and guarded against aliasing
Q (`projection[3]`), Gate (the upper half of `projection[3]`), output
(`projection[1]`), and raw Q/Gate (`projection[0]`). A retained S4 route should
instead get one explicit 32-MiB mixed-type byte region in the request arena;
depending on an implicit cross-buffer concatenation is invalid because the
FlashInfer scratch must be contiguous.

## Complexity and risks

| item | complexity | main risk |
|---|---|---|
| pass 32-MiB scratch to stock dispatcher | low | only tiny tails change numerically/topologically |
| exact host planner for the fixed Orin | low-medium | safety becomes tied to SM/resource assumptions |
| forced C512/S2 direct specialization | medium | vendored internal-template coupling and merge rounding |
| dynamic S1/S2/S4 policy | medium-high | benchmark overfitting and 32-MiB arena/test churn |
| per-layer workspaces | unnecessary | memory multiplication with no concurrency benefit |

The dominant performance risk is a negative result: the existing C512 grid is
already much larger than device residency, so MergeStates overhead can exceed
any scheduling benefit. The dominant correctness risk is not causal masking;
the vendored suffix-aligned causal contract matches
`kv_len=first_position+qo_len`. It is the additional BF16 partial-state
rounding. The dominant maintenance risk is calling FlashInfer internal
templates directly; keep any forced specialization isolated behind a
BUILD+RUN admission until real long-context evidence justifies it.
