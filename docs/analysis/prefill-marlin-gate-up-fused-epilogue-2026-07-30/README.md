# Prefill Marlin Gate/Up fused epilogue (SM87)

Status: retained after the cumulative real-serving direction gate.

Base checkpoint: `8051485`.

## Structural change

The candidate preserves the existing vLLM-Marlin NVFP4 GEMM scheduler, tiles,
FP32 cross-CTA reduction, and BF16 output boundary.  It changes only the
Prefill Gate/Up path when
`Q3X_RUN_PREFILL_MARLIN_GATE_UP_EPILOGUE_ADMISSION=1` is present:

- load-time Gate and Up rows and block scales are packed as
  `[gate0, up0, gate1, up1, ...]` before the unchanged Marlin repack;
- the vendored Marlin template has a compile-time `fused_gate_up` selector,
  defaulting to `false`, so FP8, Down, and the ordinary Gate/Up route retain
  their old specialization;
- non-final slices retain the existing FP32 reduction workspace and order;
- the last writer reads adjacent BF16 Gate/Up values from Marlin's shared
  writeback tile, performs the same
  `gate / (1 + expf(-gate)) * up` operation, rounds once to BF16, and writes
  the intermediate tensor directly;
- the runner skips the separate merged Gate/Up SiLU kernel only on this
  admitted Prefill Marlin path.

The existing unused `b_bias_ptr` kernel argument carries the fused output
pointer in this specialization.  Its base advances with Marlin's own `C` and
`par_id` row-stripe state.  This preserves the equality between a merged
`int4` index and the corresponding fused `int2` index for every M stripe.  An
initial real-serving run exposed a missing row-base advance at long prompts;
the final measured ELF contains the corrected pointer progression and passes
all eight request/output checks.

## Compiled resources

CUDA 13.3, `sm_87`, Release build; `cuobjdump --dump-resource-usage`:

| fused specialization | registers/thread | stack | local |
| --- | ---: | ---: | ---: |
| M64 (`M4N16K4`) | 255 | 0 B | 0 B |
| M32 (`M2N16K4`) | 200 | 0 B | 0 B |
| M16 (`M1N8K8`) | 126 | 0 B | 0 B |
| M8 (`M1N8K8`, block-size 8) | 123 | 0 B | 0 B |

The final measured server ELF SHA-256 is
`17a64518ab85e5d95356ae4937c67e088c70d7bce34e53b88f609f608c4953f7`.

## Real-model serving direction gate

The external EvalScope 1.9.1 first-eight run used the real
Qwen3.6-27B-NVFP4 checkpoint, frozen ShareGPT corpus, one warmup, serial OpenAI
streaming, and 16 generated tokens per request.  The comparison baseline is
the cumulative split-KV plus K512 Down result at `8051485`.

| Metric | cumulative baseline | fused epilogue | Delta |
| --- | ---: | ---: | ---: |
| exact generated outputs | 8/8 | 8/8 | unchanged |
| wall time | 22.169210 s | 22.053249 s | -0.115961 s |
| mean TTFT | 1167.688591 ms | 1153.096972 ms | -14.591619 ms |
| mean TPOT | 106.876786 ms | 106.884883 ms | +0.008096 ms |
| prompt throughput | 179.573380 tok/s | 180.517615 tok/s | +0.944235 tok/s |

Every prompt-length bucket was positive for TTFT: 2/2 requests at 1--128,
2/2 at 129--512, 3/3 at 513--1024, and 1/1 at 1025+.

Evidence:

- result leaf:
  `/tmp/q3x-evalscope-prefill-marlin-gateup-epilogue-run2/prefillmarlingateupepiloguefix/parallel_1_number_8`;
- validator:
  `/tmp/q3x-evalscope-prefill-marlin-gateup-epilogue-run2-validation.json`;
- corpus SHA-256:
  `bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`.

The existing real-weight layer-0 Marlin fixture was extended to compare the
ordinary Marlin plus independent SiLU output with the interleaved fused
epilogue bit-for-bit at arbitrary M, including its default M512 case.  The
fixture target compiles successfully; the real runner gate above is the
executed correctness and performance authority for this commit.
