# Stock-vLLM P513 architecture profile

This is a same-host, same-checkpoint architecture reference. It has no native
production-selection authority and cuBLASLt has no production role.

The capture wraps exactly measurement 1, after one warm-up request, using
CUDA Profiler API capture range:

```bash
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop \
  --force-overwrite=true -o /tmp/q3x-vllm-p513-architecture \
  /home/rm01/setup/.venv/bin/python \
  tools/reference/qwen36_27b_vllm_prefill.py \
  --model /home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4 \
  --tokenizer same --profiles P513 --attention-backend FLASHINFER \
  --warmups 1 --measurements 1 --cuda-profiler-measurement 1 \
  --gpu-memory-utilization 0.8 \
  --output /tmp/q3x-vllm-p513-architecture.jsonl
```

The request reports 513 prompt tokens, trusted scheduled-to-first-token
latency 1,246.689081 ms, 411.489928 prompt token/s, zero cached tokens, and
first token 9419. The engine selected Marlin W4A16 NVFP4, Marlin W8A16 FP8,
Triton/FLA GDN, and FlashInfer Full Attention on SM87.

`nsys stats --report cuda_gpu_kern_sum --format csv` attributes 1,224.727008 ms
of kernel time. The four Marlin rows occupy 1,098.201568 ms (89.67%):
723.607136 ms W4A16 and 374.594432 ms W8A16. Named chunk-GDN core kernels
occupy 41.549376 ms and the FlashInfer Full Attention kernel occupies
3.349504 ms.

The format split is decoded from the second `Marlin` template type ID, not
from tile shape: `562949953487106` is FE2M1/NVFP4 and
`2814749767172868` is FE4M3FN/FP8. The earlier checked-in classification
incorrectly grouped the two M64 rows as W4A16 and the two M8 rows as W8A16;
the raw durations were correct, but that interpretation and the resulting
priority decision were not.

The raw report is not checked in. Its retained identity is:

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `/tmp/q3x-vllm-p513-architecture.nsys-rep` | 206,614 | `5d9125bc479d5356d0da11ae039710432b390beee8d81fec7ff1b87bff164a6d` |
| `/tmp/q3x-vllm-p513-architecture.jsonl` | 3,270 | `391d5d8d604947375b10ea10d656749e49231561f1d02814e7ea3ad944667284` |

The checked-in [top-20 summary](kernel-top20.csv) is sufficient to audit the
architecture claims without retaining the profiler database.
