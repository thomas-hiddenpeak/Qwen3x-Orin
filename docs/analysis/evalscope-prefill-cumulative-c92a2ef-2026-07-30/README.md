# EvalScope cumulative Prefill checkpoint at `c92a2ef`

Date: 2026-07-30

This checkpoint combines two retained native GDN changes on top of
`85f5f01`: neutral padded partial-C64 tiles and the M64 WY recompute kernel.
It is measured through the real OpenAI-compatible server with authenticated
Qwen3.6-27B-NVFP4 weights.  cuBLASLt is not present in the production path and
the frozen vLLM process was not restarted.

## Frozen eight-request external result

The hash-locked corpus contains prompt lengths
`64, 481, 564, 1025, 695, 32, 713, 407`, with 16 generated tokens per request.

| runner | prompt tok/s | mean TTFT ms | mean TPOT ms | duration s | success |
|---|---:|---:|---:|---:|---:|
| previous native `85f5f01` | 175.486615 | 1201.245599 | 108.942627 | 22.685491 | 8/8 |
| native `c92a2ef` | **177.913533** | **1167.935631** | **108.587628** | **22.376038** | 8/8 |
| frozen matched vLLM | **181.896870** | 1168.570642 | **104.466390** | 21.886028 inferred | 8/8 |

Against the previous cumulative native runner, mean TTFT falls by
33.309969 ms and whole-workload prompt throughput rises by
2.426918 tok/s.  Every one of the eight requests has lower TTFT and all eight
generated outputs are byte-for-byte identical to the previous native runner.

Native mean TTFT is now 0.635011 ms lower than the frozen matched vLLM mean,
but the parity stop condition is not met.  The native 16-token TPOT deficit
leaves complete workload duration about 0.49 seconds too high and prompt
throughput 3.983337 tok/s below vLLM.  Optimization therefore continues; this
checkpoint is retained but is not presented as vLLM parity.

## Per-request TTFT

| prompt tokens | native `c92a2ef` ms | frozen vLLM ms | native minus vLLM ms |
|---:|---:|---:|---:|
| 64 | 165.194753 | 210.847511 | -45.652758 |
| 481 | 1143.708630 | 1125.351516 | +18.357114 |
| 564 | 1294.804216 | 1281.485739 | +13.318477 |
| 1025 | 2370.425952 | 2343.723613 | +26.702339 |
| 695 | 1584.358149 | 1566.113445 | +18.244704 |
| 32 | 131.106443 | 192.035844 | -60.929401 |
| 713 | 1681.198263 | 1659.355888 | +21.842375 |
| 407 | 972.688639 | 969.651578 | +3.037061 |

The remaining Prefill gap is concentrated in the longer requests; short P32
and P64 requests are already materially faster than vLLM.  This supports the
next structural work on per-layer large-M/GDN throughput rather than fixed API
overhead.

## Reproduction and provenance

- corpus: `/tmp/q3x-sharegpt-false-thinking-33.jsonl`
- corpus SHA-256:
  `bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`
- result leaf:
  `/tmp/q3x-evalscope-cumulative-c92a2ef-run1/cumulativec92a2ef/parallel_1_number_8`
- direction validator:
  `/tmp/q3x-evalscope-cumulative-c92a2ef-vs-85f5f01.json`
- server ELF SHA-256:
  `988a908d047672e582919f98f7bf8c529318937c529e35f5a1890602f298a5e5`

The validator passed database integrity, request-manifest matching,
completion completeness, recomputed summaries, exact native outputs, lower
mean TTFT, and higher native prompt throughput.  This is an external
direction checkpoint; the frozen vLLM workload remains the final parity
reference.
