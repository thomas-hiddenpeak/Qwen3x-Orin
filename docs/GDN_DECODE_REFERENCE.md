# Single-token and bounded-tile Gated DeltaNet reference

`q3x/runtime/gdn_decode.h` defines the allocation-free CPU and CUDA numerical
boundary for one Qwen3.6-27B linear-attention decode step and CUDA prompt tiles
of up to 16 tokens. The implementation
follows [the pinned runtime contract](QWEN36_27B_RUNTIME_CONTRACT.md#5-linear-attentiongated-deltanet-精确语义)
and was independently checked against vLLM commit
`ccd49f6821ee110cc5a2b1aba620a8a1d66c7cbb`. It does not copy third-party
kernel code.

## Fixed logical ABI

- Q/K heads: 16, value heads: 48, all with dimension 128.
- Projection/conv vector: BF16 `[10240]` laid out as Q `[16,128]`, then K
  `[16,128]`, then V `[48,128]`.
- Conv history: channel-major BF16 `[10240,3]`, oldest to newest.
- Conv weight: channel-major BF16 `[10240,4]` (checkpoint `[10240,1,4]`
  with the singleton dimension removed).
- Recurrent state: canonical BF16 `[48,V=128,K=128]`, row-major in K.
- Recurrent output: BF16 `[48,128]`.

`GdnDimensions` is checked before pointer use. Only the exact pinned shape is
accepted, while overflowing products receive a distinct structured status.

## Causal convolution

For every channel, `causal_conv1d_silu_update_*` accumulates the three raw
history values with weights 0..2 and the current raw projection with weight 3
in FP32, applies SiLU, and writes BF16 RNE output. It then shifts history and
stores the original current BF16 value. A convolved value is never fed back
into history.

Raw input and conv output may be exactly in-place. History, weights, and output
must otherwise be disjoint; exact invalid aliases are rejected and partial
overlap is outside the contract.

## Gated Delta recurrence

Value head `j` uses Q/K head `floor(j/3)`. Q and K are separately normalized
with `rsqrt(sum(x^2) + epsilon)`; Q alone is additionally scaled by
`1/sqrt(128)`. Scalar gates stay FP32:

```text
x     = a[j] + dt_bias[j]
g     = -exp(A_log[j]) * softplus(x, threshold=20)
alpha = exp(g)
beta  = sigmoid(b[j])
```

For each value row, the implementation decays the BF16 input state into FP32,
computes its K prediction, applies the beta-scaled prediction error, performs
the outer-product update, and computes output against Q. State is then stored
with BF16 RNE. Output is computed from the FP32 updated row before that state
rounding; a dedicated fixture distinguishes this from incorrectly rereading
the quantized state.

Exact `state_input == state_output` is supported and is the normal persistent
decode path. State and output use canonical `[head][value][key]` orientation.

## Bounded causal tiles

The CUDA tile entry points accept `token_count=1..16`. Convolution input/output
is token-major BF16 `[M,10240]`; GDN `a` and `b` are `[M,48]`, and recurrent
output is `[M,48,128]`. M1 delegates to the corresponding single-token entry
point. For M2..M16, causal convolution advances raw BF16 history in token order
and GDN recurrence reads the BF16 state persisted by the preceding row. Thus
tiling changes launch organization without changing the causal rounding
boundary. `A_log`, `dt_bias`, convolution weights, and the one persistent
history/state allocation remain shared across the tile.

Commit `c90f37e` raises the checked tile limit from eight to 16. The C16 CUDA
gate compares one 16-token launch with two ordered C8 launches and requires
bitwise-equal convolution output/history and GDN output/state. Invalid zero or
17-token requests, malformed dimensions, overflow, and forbidden aliasing fail
before a kernel is accepted.

## CUDA ownership and verification

CUDA calls accept caller-owned device pointers and an optional stream. They
perform no allocation, copy, or synchronization, and isolate their launch
status from an unrelated stale CUDA last-error. The recurrent kernel launches
one 128-thread block per value head; one thread owns one V row, shared memory
holds normalized Q/K, and the thread computes both its full state row and
output.

Host tests cover cold state, five-step conv history, weight/history
orientation, 48-to-16 head sharing, stable softplus and sigmoid extremes,
canonical state axes, in-place versus separate state, invalid aliases,
overflow, non-finite epsilon, and FP32-before-BF16 output. SM87 CUDA tests run
the exact 10240-channel conv and 48x128x128 recurrent shape for multiple
persistent BF16 steps against the CPU oracle on a non-default stream. They also
cover every tile size from M1 through M16 and the explicit C16-versus-C8+C8
causal equivalence gate.
