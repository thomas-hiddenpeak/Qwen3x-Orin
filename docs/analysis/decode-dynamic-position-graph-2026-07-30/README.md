# Long-position reusable Decode CUDA Graph admission

Status: **rejected; route closed**. The first 65-SetParams implementation and
its graph-stable device-parameter-block successor were both catastrophically
negative on the real-model serving path. The code remains behind admission
build/runtime gates only so the result is auditable; neither candidate is a
production route and no follow-on topology tuning is planned.

## Candidate contract

The candidate captures one long-attention Decode graph at current position
64 (the generated token sees sequence length 65). The same `cudaGraphExec_t`
covers current positions 64 through
`min(max_sequence_length - 1, 4095)`; it does not allocate or select a graph
per position.

The superseding topology has 407 nodes: 405 kernels, one 16-byte parameter
H2D memcpy, and one logits D2H memcpy. The H2D node is the single graph root
and copies from stable pinned-host storage to a stable device control block.
Before replay the host writes `{token_id, position, sequence_length,
split_count}` to that pinned block and calls `cudaGraphLaunch`; it performs
zero `cudaGraphExecKernelNodeSetParams` calls.

Embedding, full-attention Q/K/V, preprocess, split-state, and split-merge
kernels read the device block. Q/K/V and preprocess receive base KV-cache
pointers and derive the current row on device. Split-state always captures a
32-CTA grid and guards CTAs above `4 * split_count`, so positions 64..511 use
four splits and positions 512..4095 use eight without a graph mutation.
Positions above 4095 retain the serial long-attention path.

## Build and runtime gate

```bash
cmake -S . -B build/decode-dynamic-graph \
  -DBUILD_TESTING=ON \
  -DQ3X_BUILD_DECODE_DYNAMIC_GRAPH_ADMISSION=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/decode-dynamic-graph \
  --target qwen3x-eval-server -j2

Q3X_RUN_DECODE_GQA_SPLITKV_ADMISSION=1 \
Q3X_RUN_DECODE_DYNAMIC_GRAPH_ADMISSION=1 \
build/decode-dynamic-graph/qwen3x-eval-server MODEL_DIR \
  --host 127.0.0.1 --port 18080 \
  --model qwen3.6-27b-nvfp4 \
  --max-sequence-length 8192 --max-output-tokens 4096 \
  --prefill-chunk-size 512 --projection-backend sm87
```

The ready line must contain all of:

```text
decode_graph_dynamic_position_ready=1
decode_graph_dynamic_first_position=64
decode_graph_dynamic_nodes=407
decode_graph_dynamic_updated_nodes=0
```

Keep `Q3X_RUN_DECODE_GQA_SPLITKV_ADMISSION=1` on both sides of the comparison;
omit only `Q3X_RUN_DECODE_DYNAMIC_GRAPH_ADMISSION` for the same-ELF serial
split-KV baseline. A capture, topology, reset, or admission failure clears both graph banks
and leaves the engine on the serial path; it never silently publishes a
partial dynamic graph.

## Real-run rejection

Use the pinned real-model OpenAI/EvalScope perf8 workload from
`docs/EVALSCOPE_EVALUATION.md`. Run the same ELF and corpus once with the
dynamic-graph runtime variable absent and once with it set to `1`. The prompts in that
corpus exercise real generation positions above 64, so the reported ITL is a
whole-runner measurement of this path rather than an isolated kernel proxy.

The original 65-SetParams candidate completed seven of eight requests at
roughly 5--8 seconds each, versus roughly 3--4 seconds on the same-ELF serial
split-KV baseline. It was stopped before request eight.

The device-parameter-block candidate removed every dynamic replay
`cudaGraphExecKernelNodeSetParams` call, leaving one captured 16-byte H2D and
one `cudaGraphLaunch` per token. Its real-path screen still produced a
7.83-second warmup and approximately **7.83 / 5.35 / 4.0 / 5.5 seconds** for
the first four requests. The run was stopped at 4/8 with `SIGINT` because it
was in the same catastrophic range as the predecessor.

This A/B sequence falsifies the working hypothesis that host-side node
parameter replacement was the dominant regression. The remaining dynamic
kernel/topology route is not competitive with the retained serial split-KV
path, so it is rejected without B-C-C-B noise characterization or any further
micro-tuning.

## Static verification completed

- the admission build compiled successfully;
- the admission `qwen3x-eval-server` is an AArch64 ELF with build ID
  `50bb8202d0baad4eeacdef02f65bbf496741abce` and SHA-256
  `5ea3a51eb5710ecb8fe67040fdc16f5d25c84be17762e1fe721fa05fe86f6121`;
- the engine translation unit contains
  `Q3X_ENABLE_DECODE_DYNAMIC_GRAPH_ADMISSION=1`;
- the linked kernel archive exports dynamic prepare, availability, and replay
  methods; and
- `qwen3x-eval-server --help` succeeds without loading a model.

No model was loaded and no CUDA work was launched during this static cell.
