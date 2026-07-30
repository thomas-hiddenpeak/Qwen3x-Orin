# Long-Prefill layer-major admission

This is a host-only, test-build admission scaffold. It is **not** a production
route and does not change the existing tile-major runner. It adds no Attention,
GDN, GEMM, or CUDA kernel and cannot be selected by an ordinary build.

The first bounded shape is batch-one `P<=4096`, `C=512`. The request planner
option `long_prefill_token_capacity` reserves two BF16 `[P,5120]` hidden slabs
after the existing C512 scratch. All three C512 hidden buffers, four projection
buffers, linear `a`/`b`, and FP32 scratch retain their original offsets and are
reused by every layer/tile. At P4096 this adds exactly 83,886,080 bytes:

| P4096/C512 region | Bytes |
| --- | ---: |
| Existing request arena | 436,109,312 |
| Two full hidden slabs | 83,886,080 |
| Layer-major admission arena | 519,995,392 |

The pure-host executor fills hidden slab 0 once, then emits all prompt tiles of
layer 0 before any tile of layer 1. A layer reads slab `layer % 2` and writes
slab `(layer + 1) % 2`. Linear-attention tiles are emitted in increasing global
position so Conv/GDN recurrent state is updated in sequence. Full-attention
tiles use the same order, making their K/V append interval identical to their
global causal positions. The executor itself introduces no per-tile device
synchronization. Its finish callback publishes the prompt sequence length only
after all 64 layers have completed, so a real binding must not advance the
request's global logical position once per layer.

Build and run the independent host gate without using a GPU:

```bash
cmake -S . -B build/long-prefill-host \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
  -DQ3X_BUILD_LONG_PREFILL_LAYER_MAJOR_ADMISSION=ON
cmake --build build/long-prefill-host \
  --target q3x_long_prefill_layer_major_plan_test q3x_request_state_plan_test
ctest --test-dir build/long-prefill-host --output-on-failure \
  -R 'long_prefill_layer_major_plan|request_state_plan'
```

The device-direction experiment must remain separate. Its first implementation
should bind the executor callbacks to one real-model runner, compare the
ordinary tile-major route against layer-major at the same tokenized P4096
prompt and `max_new_tokens=1`, and require identical final prediction plus
GDN/KV/position state before timing. Only after this directional run is
positive should the project add repeated timing, long-context Attention work,
or raise the 4096-token admission bound toward 40K.
