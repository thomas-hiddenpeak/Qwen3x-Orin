# Long-position reusable Decode CUDA Graph admission

Status: code-complete, admission ELF built, GPU execution intentionally left
to the root measurement cell. This document makes no performance or
correctness promotion claim.

## Candidate contract

The candidate captures one long-attention Decode graph at current position
64 (the generated token sees sequence length 65). The same `cudaGraphExec_t`
covers current positions 64 through `max_sequence_length - 1`; it does not
allocate or select a graph per position.

The captured topology has 438 nodes: 437 kernels and one logits D2H memcpy.
Before each replay the host replaces exactly 81 kernel-node parameter sets:

- one embedding node for the new token id;
- 16 Q/K/V projection nodes for the position-specific K/V write addresses;
- 16 full-attention preprocess nodes for the K address and RoPE position;
- 16 attention-score nodes for sequence length and `grid.y`;
- 16 softmax nodes for sequence length; and
- 16 attention-value nodes for sequence length.

All other graph nodes and parameters remain fixed. Capture records the tail
node immediately after each high-level launch and recovers score/softmax from
the value node's dependency chain. It does not depend on kernel names or the
unspecified order returned by `cudaGraphGetNodes`.

## Build and runtime gate

```bash
cmake -S . -B build/decode-dynamic-graph \
  -DBUILD_TESTING=ON \
  -DQ3X_BUILD_DECODE_DYNAMIC_GRAPH_ADMISSION=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/decode-dynamic-graph \
  --target qwen3x-eval-server -j2

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
decode_graph_dynamic_nodes=438
decode_graph_dynamic_updated_nodes=81
```

Omitting the runtime environment variable selects the baseline in the same
ELF. A capture, topology, reset, or admission failure clears both graph banks
and leaves the engine on the serial path; it never silently publishes a
partial dynamic graph.

## First real-run gate

Use the pinned real-model OpenAI/EvalScope perf8 workload from
`docs/EVALSCOPE_EVALUATION.md`. Run the same ELF and corpus once with the
runtime variable absent and once with it set to `1`. The prompts in that
corpus exercise real generation positions above 64, so the reported ITL is a
whole-runner measurement of this path rather than an isolated kernel proxy.

The first decision is directional: preserve the candidate only if all eight
outputs exactly match and mean ITL improves by at least 0.5 ms/token. If it
passes, run the normal B-C-C-B confirmation and longer-output stability gate.
If it is negative, retain the profile and rejection record; do not promote it
or compensate by expanding the fixed-position table.

## Static verification completed

- both the default build and admission build compile;
- the admission `qwen3x-eval-server` is an AArch64 ELF;
- the engine translation unit contains
  `Q3X_ENABLE_DECODE_DYNAMIC_GRAPH_ADMISSION=1`;
- the linked kernel archive exports dynamic prepare, availability, and replay
  methods; and
- `qwen3x-eval-server --help` succeeds without loading a model.

No model was loaded and no CUDA work was launched during this static cell.
