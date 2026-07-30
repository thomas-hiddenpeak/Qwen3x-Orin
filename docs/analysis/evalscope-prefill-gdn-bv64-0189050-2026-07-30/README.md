# EvalScope8 GDN BV64 external direction at `0189050`

Date: 2026-07-30

This is a one-round, whole-runner OpenAI-compatible direction check with real
model weights. EvalScope 1.9.1 measured eight requests after one warmup at
concurrency one, with streaming enabled and 16 generated tokens per request.
The baseline is commit `a23bf1d983b86569f25603a01572768e0dd4891f`; the
candidate is commit `0189050bcfa4a21ee5b5aa7d38e465ef637b49f5` with the
production BV64 GDN chunk-output path.

## Frozen identities

- Candidate binary: `/tmp/q3x-cumulative-eval-build-make/qwen3x-eval-server`
- Candidate binary SHA-256:
  `4e85c938cd344d42e84ea621bdb819486e93f781f227aa89eca8d16bbfd6a727`
- Candidate ELF Build ID: `cf74ba5ffcccc1b636fdd7e515c245eea05ef637`
- Corpus SHA-256:
  `bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`
- Baseline leaf:
  `/tmp/q3x-evalscope-cumulative-a23bf1d-run1/cumulativea23bf1d/parallel_1_number_8`
- Candidate leaf:
  `/tmp/q3x-evalscope-cumulative-0189050-run1/cumulative0189050/parallel_1_number_8`
- Validator report:
  `/tmp/q3x-evalscope-cumulative-0189050-vs-a23bf1d.json`
- Validator report SHA-256:
  `2e41ac137014967497c0c697bb22bea91e13fe05de9c473fd29645a8dd06a1c2`

The validator passed benchmark-argument comparability, database integrity,
ordered manifest matching, summary recomputation, and completion completeness.
All eight generated outputs are exact between baseline and candidate.

## Result

| metric | `a23bf1d` baseline | `0189050` candidate | candidate delta |
|:---|---:|---:|---:|
| mean TTFT ms | 1262.600676 | 1234.418541 | **-28.182136** |
| p50 TTFT ms | 1417.982516 | 1390.547019 | -27.435497 |
| p99 TTFT ms | 2527.719202 | 2466.830263 | -60.888939 |
| prompt throughput tok/s | 171.569392 | 173.764991 | **+2.195599** |
| mean TPOT ms | 109.168362 | 108.606864 | -0.561498 |
| wall time s | 23.203439 | 22.910254 | -0.293186 |
| exact outputs | 8/8 | 8/8 | unchanged |

Every request is faster in its prompt-length bucket: `2/2` for 1--128,
`2/2` for 129--512, `3/3` for 513--1024, and `1/1` for 1025+ tokens. Mean
candidate-minus-baseline TTFT changes are respectively `-2.706861`,
`-26.155564`, `-35.614432`, and `-60.888939` ms. The validator exits zero and
marks the candidate `advance_to_internal_validation`.

The separately frozen matched-first-eight vLLM reference remains
`181.896870119` prompt tok/s and `1168.570642` ms mean TTFT. The `0189050`
candidate remains `8.131878953` tok/s below that prompt-throughput reference
and `65.847898902` ms above its mean TTFT. vLLM was not rerun.
