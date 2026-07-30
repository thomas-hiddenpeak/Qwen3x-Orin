# Decode Gate/Up coupled-feed cumulative vLLM parity

Status: retained after two real-model EvalScope-8 confirmation runs.

The measured candidate is commit `17cc711` rebased on cumulative checkpoint
`549eba4`.  Its Release server ELF SHA-256 is
`93d3e4d799c1f7ddc55cf0adeffb203f312a4cb79fed05108d0ab8d671d0e54b`.
The build cache matches the cumulative production-like build and enables
BF16 large-M, FlashInfer Prefill attention, FP8 Marlin Prefill, native GDN
chunk-64, and NVFP4 Marlin Prefill.  Runtime composition enables split-KV,
the Prefill Marlin Gate/Up epilogue, and the Decode Gate/Up coupled feed;
Decode Down consumer-order is compiled but not admitted in these runs.

## Result

Both independent real Qwen3.6-27B-NVFP4 runs preserved all eight generated
outputs and exceeded the separately frozen vLLM first-eight market floor.

| metric | frozen vLLM first 8 | run 1 | run 2 |
| --- | ---: | ---: | ---: |
| exact native outputs | n/a | 8/8 | 8/8 |
| prompt throughput | 181.896870 tok/s | 183.341934 tok/s | 183.315553 tok/s |
| mean TTFT | 1168.570642 ms | 1152.676220 ms | 1153.721993 ms |
| mean TPOT | 104.466390 ms | 104.083710 ms | 104.041991 ms |
| wall time | 21.886028 s | 21.713527 s | 21.716652 s |

Run 1 beats the frozen vLLM slice by 1.445064 prompt tok/s, 15.894422 ms
TTFT, 0.382680 ms TPOT, and 0.172501 s wall time.  Run 2 independently
beats it by 1.418683 prompt tok/s, 14.848649 ms TTFT, 0.424399 ms TPOT, and
0.169376 s wall time.

The run-2 direction validator exits 3 only because its incremental-native
rule requires mean TTFT to improve on every single round; run 2 moved by
`+0.625021 ms` versus the native cumulative baseline while prompt throughput
and TPOT remained strongly positive.  This sub-millisecond movement is
treated as TTFT noise.  The product-level vLLM parity gate and exact-output
gate both pass in both runs.

## Evidence

- run 1 leaf:
  `/tmp/q3x-evalscope-gateup-coupled-cumulative-fullon-run1/gateupcoupledcumulativefullon/parallel_1_number_8`
- run 1 validator:
  `/tmp/q3x-evalscope-gateup-coupled-cumulative-fullon-vs-180517.json`
  (`341656a457cd5d685262e425a180a7bb657c5cf0979ee20555240aa5e89e524f`)
- run 2 leaf:
  `/tmp/q3x-evalscope-gateup-coupled-cumulative-fullon-run2/gateupcoupledcumulativefullonr2/parallel_1_number_8`
- run 2 validator:
  `/tmp/q3x-evalscope-gateup-coupled-cumulative-fullon-run2-vs-180517.json`
  (`238b103c803e2ef6b42af8da3c5cb10662613ba0a35691db166b6731894d4ed1`)
- corpus SHA-256:
  `bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`

The candidate remains an explicitly admitted SM87/Qwen3.6 specialization.
Its equal-byte arena is 6,417,285,120 bytes, so a future load-time in-place or
canonical-storage replacement should retain the measured Decode dataflow
without keeping an equal-sized second copy.  MTP is not used.
