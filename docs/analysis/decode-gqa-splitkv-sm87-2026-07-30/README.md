# Decode GQA split-KV design (SM87)

Status: compiled candidate; no GPU correctness or performance claim yet.

Base checkpoint: `c65f938`.

## Why this changes the decomposition

The rejected C1 bulk attention candidate exposed only four CTAs, one per KV
head.  That shape cannot cover the 16 SMs on Orin for long decode sequences.
The candidate in this branch changes the ownership model instead of tuning the
four-CTA kernel:

- sequence length 65..512: `4 KV heads * 4 splits = 16` first-stage CTAs;
- sequence length 513..4096: `4 KV heads * 8 splits = 32` first-stage CTAs;
- one CTA has six warps, one warp per Q head in the KV head's fixed 6:1 GQA
  group;
- each K/V cache row is loaded and BF16-decoded once per CTA into shared
  memory, then consumed by all six Q warps;
- Q and eight FP32 value-accumulator lanes per thread stay resident across the
  CTA's whole sequence split.

This is a fixed Qwen3.6 `Q=24, KV=4, D=256` implementation.  It does not add a
generic attention abstraction.

## Online state and merge

Stage one never writes a score or probability matrix.  For each Q head and
split it writes the stable online-softmax state

```
(maximum, denominator, unnormalized FP32 value numerator[256])
```

The fixed state is 258 FP32 elements.  Workspace layout is
`[query_head=24, split=4|8, state=258]`, requiring:

- 24,768 FP32 elements (96.75 KiB) for 16 first-stage CTAs;
- 49,536 FP32 elements (193.5 KiB) for 32 first-stage CTAs.

The existing runner FP32 scratch allocation is larger than the maximum state
workspace, so the candidate adds no request allocation and remains graph
capture safe.

Stage two launches 24 CTAs, one per Q head.  It computes the global maximum,
rescales and merges every split denominator/numerator, normalizes the result,
then preserves the production numerical boundary:

```
attention FP32 -> BF16 RNE -> sigmoid(gate BF16) -> BF16 RNE
```

The state/merge formulation follows the online-softmax and split-state merge
ideas in FlashInfer's `attention/state.cuh`, `attention/decode.cuh`, and
`attention/cascade.cuh`; the fixed SM87 kernel and layout here are independently
implemented for this runner.

## Compiled resources

CUDA 13.3, `sm_87`, Release build; `cuobjdump --dump-resource-usage`:

| kernel | threads | registers/thread | static shared | local/stack |
| --- | ---: | ---: | ---: | ---: |
| split state | 192 | 44 | 2,048 B | 0 B |
| merge + BF16 gate | 256 | 40 | 36 B | 0 B |

The state kernel's 44 registers and 2 KiB shared allocation leave resource
headroom beyond the requested two CTAs/SM; the actual scheduler occupancy must
still be measured with NCU after the whole-runner direction is established.

## Production selector and stop/go protocol

The old score/softmax/value/gate route remains in the same executable.
`Q3X_RUN_DECODE_GQA_SPLITKV_ADMISSION=1` selects this candidate only for
dynamic sequence lengths 65..4096.  Lengths 1..64 retain the existing fused
kernel and lengths above 4096 retain the old long-sequence route.

Before promotion:

1. run the CUDA reference comparison at boundary and long lengths;
2. run the real-weight short-output EvalScope workload with the selector off
   and on against the same executable;
3. retain only if real TPOT/latency improve without changing exact generated
   outputs;
4. only after a positive whole-runner direction, add repeated timing, NSys,
   NCU, graph-topology, and full correctness coverage.

