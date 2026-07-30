# Prefill GDN convolution + compact Q/K structural candidate

Date: 2026-07-30

This branch starts from cumulative checkpoint `fba08add`. It is a structural
candidate only: no real-model performance result is claimed here.

## Frozen evidence and selection

The frozen current P513 Nsight Systems capture attributes the remaining GDN
preprocessing window as follows:

| stage | calls | total ms | average us/layer |
|:---|---:|---:|---:|
| token-parallel causal convolution + SiLU | 48 | 8.219712 | 171.244 |
| standalone compact Q/K normalize | 48 | 5.032480 | 104.843 |
| gate preparation | 48 | 0.465856 | 9.705 |

The matching frozen vLLM trace uses a 6.764480 ms causal-convolution kernel
and a 5.961728 ms fused post-convolution kernel. That is evidence for a
post-convolution layout boundary, not a reason to reproduce its Python/Triton
glue. The project already owns a stronger fixed-shape fact: every C8 x C256
convolution CTA owns two complete 128-wide Q or K heads.

The selected replacement therefore folds compact Q/K normalization into the
existing convolution CTA. It removes the standalone 8192-CTA-per-layer
normalizer and its second global read of convolved Q/K. The strict P513
latency-saving upper bound is the entire standalone normalizer:

`5.032480 ms`, or `36.58%` of the three-stage preprocessing window.

Gate preparation is only 0.465856 ms and needs a C64 prefix across eight C8
CTAs. It is deliberately excluded rather than coupling a small extra target
to this first structural decision.

## Exact dataflow

```text
raw QKV BF16 + width-4 weights + history
                  |
                  v
      C8 x C256 token-parallel convolution
                  |
          FP32 FMA -> SiLU -> BF16
             /                 \
            /                   \
   all 10240 channels       Q/K channel CTAs only
   write conv_qkv           stage C8 x C256 BF16 in 4 KiB shared
                                   |
                          8 warps own 8 tokens
                                   |
                         two 128-d head waves
                                   |
                       exact baseline reduction tree
                     (64,32 pairs; 16..1 shuffle)
                                   |
                  compact [chunk, head, 64, 128] Q/K BF16
                                   |
               native GDN begins after normalize stage
```

The convolution output is rounded to BF16 before it participates in the
normalization square sum, matching the incumbent boundary. The reduction
association is also preserved: `(d0+d64) + (d32+d96)` precedes the
stride-16/8/4/2/1 tree. Only the execution topology changes.

## Static and component gates

The CUDA 13.3 SM87 build reports:

- 30 registers per thread;
- 4096 bytes static shared memory;
- zero local memory and zero stack;
- six active 256-thread CTAs per SM on the target;
- one CTA barrier in the fused kernel.

The component harness compares the incumbent two-launch sequence against the
candidate on a C512 tensor. It passed with zero unequal BF16 values for all
four observed boundaries: convolution output, convolution history, compact Q,
and compact K. Existing arbitrary-C1..C512 convolution and CUDA Graph cases
also remain passing.

Synthetic values are used only for this correctness component gate. They are
not a performance verdict.

## Real-path admission evidence

The same-ELF selector is:

`Q3X_RUN_GDN_CONV_COMPACT_QK_FUSED_CANDIDATE=1`

It has authority only when both production prerequisites are also active:

- `Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION=1`
- `Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION=1`

The runner additionally requires a C64-aligned tile with no legacy tail. A
dedicated runtime counter is reset/read through
`exchange_gdn_conv_compact_qk_fused_candidate_test_hits`; the native real-model
harness prints both observed and expected fused-preprocess hits. Nsight must
also show
`causal_conv1d_silu_update_token_parallel_compact_qk_kernel` and no standalone
`normalize_qk_kernel` for the aligned tile before any timing is accepted.

The next authorized gate is one real-model P513 baseline/candidate direction
sample with the complete production environment. A negative result is enough
to reject the structure; a positive result must then pass the full project
retention protocol.
