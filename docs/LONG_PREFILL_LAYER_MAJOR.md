# Long-Prefill layer-major admission

This is a test-build/runtime-gated real-runner admission. It is **not** the
default production route and does not change the existing tile-major fallback.
It adds no Attention, GDN, GEMM, or CUDA kernel and cannot be selected by an
ordinary build.

The bounded shape is batch-one `P<=40960`, `C=512`. The request planner
option `long_prefill_token_capacity` reserves two BF16 `[P,5120]` hidden slabs
after the existing C512 scratch. All three C512 hidden buffers, four projection
buffers, linear `a`/`b`, and FP32 scratch retain their original offsets and are
reused by every layer/tile. At P4096 this adds exactly 83,886,080 bytes:

| P4096/C512 region | Bytes |
| --- | ---: |
| Existing request arena | 436,109,312 |
| Two full hidden slabs | 83,886,080 |
| Layer-major admission arena | 519,995,392 |

The same checked layout scales to the external long-context matrix without
falling back to tile-major weight rescans:

| Capacity | Two hidden slabs | Exact layer-major arena |
| ---: | ---: | ---: |
| 8,192 | 167,772,160 | 873,365,504 |
| 16,384 | 335,544,320 | 1,580,630,016 |
| 40,960 | 838,860,800 | 3,703,209,984 |

The runner binding gathers all embeddings into hidden slab 0, then emits all
prompt tiles of layer 0 before any tile of layer 1. A layer reads slab
`layer % 2` and writes slab `(layer + 1) % 2`. The existing C512 workspace is
used as the per-tile staging area, so the initial binding adds one slab-to-C512
and one C512-to-slab device copy per work item instead of changing any kernel
ABI. Cross-layer residual/RMS fusion is disabled because its C512 normalized
scratch cannot survive intervening tiles; the exact unfused kernels are used.

Linear-attention tiles execute in increasing global position so Conv/GDN
recurrent state is updated in sequence. Full-attention tiles use the same
order, and their existing projection and Attention APIs receive the explicit
global `first_position` for K/V and RoPE. There is no successful per-tile
stream synchronization. The runner synchronizes and publishes the prompt
sequence length once, after all 64 layers, and retains the final normalized
row for the existing logits-only finalizer.

Build and run the independent host gate without using a GPU:

```bash
cmake -S . -B build/long-prefill-host \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DQ3X_BUILD_LONG_PREFILL_LAYER_MAJOR_ADMISSION=ON
cmake --build build/long-prefill-host \
  --target qwen3x-eval-server q3x_reference_engine_control_test \
           q3x_long_prefill_layer_major_plan_test q3x_request_state_plan_test
ctest --test-dir build/long-prefill-host --output-on-failure \
  -R 'reference_engine_control|long_prefill_layer_major_plan|request_state_plan|eval_server_help'
```

The build-enabled evaluation server always reserves the bounded slabs when its
configured C512/max-sequence shape can use them. The environment variable only
changes request dispatch, so separate invocations of the same binary provide a
clean route-off/on comparison:

```bash
# Existing tile-major fallback.
env -u Q3X_RUN_LONG_PREFILL_LAYER_MAJOR_ADMISSION \
  build/long-prefill-host/qwen3x-eval-server MODEL_DIR \
  --prefill-chunk-size 512 --max-sequence-length 4096

# P513..P40960 layer-major candidate.
Q3X_RUN_LONG_PREFILL_LAYER_MAJOR_ADMISSION=1 \
  build/long-prefill-host/qwen3x-eval-server MODEL_DIR \
  --prefill-chunk-size 512 --max-sequence-length 40960 \
  --request-max-arena-bytes 3703209984
```

Each successful request log reports `layer_major_prefill=0|1` and the native
`prompt_prefill_ms`. No GPU execution or real-weight correctness/performance
claim is part of the host admission commit. The first device-direction run
must use the authorized real tokenized 8K/16K/40K requests, compare
final prediction and persistent Conv/GDN/KV/position state, and only then judge
timing. Host plan coverage is not a GPU correctness or performance claim.
