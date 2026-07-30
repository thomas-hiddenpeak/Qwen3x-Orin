# EvalScope8 GDN WY same-ELF direction at `a23bf1d`

Date: 2026-07-30

This is a one-round, whole-runner OpenAI-compatible direction check with real
model weights. Both routes use commit `a23bf1d983b86569f25603a01572768e0dd4891f`
and the same `qwen3x-eval-server` ELF. The baseline sets
`Q3X_GDN_CHUNK64_FORCE_GROUP_OWNED_WY_BASELINE=1`; the production candidate
leaves that diagnostic selector absent and therefore uses the admitted
value-head-owned WY hierarchy.

## Frozen identities

- Binary SHA-256:
  `97fcbf213387100efdc51abe5cc1533d4f7b8e63b0472bef5722a8f9e8464e85`
- ELF Build ID: `33bd594efd4bd1a9e72bbe027f912d2b2cfd0cab`
- Corpus SHA-256:
  `bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`
- Group-owned baseline leaf:
  `/tmp/q3x-evalscope-wy-group-baseline-a23bf1d-run1/groupbaselinea23bf1d/parallel_1_number_8`
- Production-WY candidate leaf:
  `/tmp/q3x-evalscope-cumulative-a23bf1d-run1/cumulativea23bf1d/parallel_1_number_8`
- Validator report:
  `/tmp/q3x-evalscope-wy-a23bf1d-same-elf-direction.json`
- Validator report SHA-256:
  `63c37dd0c25814b5c0eef37c927e417aa09c445f57afa6b7c502f4126f781ae1`

EvalScope 1.9.1 measured eight requests after one warmup, concurrency one,
streaming enabled, and 16 generated tokens per request. The validator passed
database integrity, ordered manifest matching, summary recomputation, and
completion completeness. All eight generated outputs are exact between the
two routes.

## Result

| metric | group-owned baseline | production WY | candidate delta |
|:---|---:|---:|---:|
| mean TTFT ms | 1268.645781 | 1262.600676 | **-6.045105** |
| p50 TTFT ms | 1423.222938 | 1417.982516 | -5.240422 |
| p99 TTFT ms | 2541.821026 | 2527.719202 | -14.101824 |
| prompt throughput tok/s | 171.597063 | 171.569392 | -0.027671 |
| mean TPOT ms | 108.736196 | 109.168362 | +0.432166 |
| wall time s | 23.199698 | 23.203439 | +0.003742 |
| exact outputs | 8/8 | 8/8 | unchanged |

The Prefill direction is consistent with prompt length. For prompts of
129--512, 513--1024, and 1025+ tokens, the candidate is faster on all
`2/2`, `3/3`, and `1/1` requests; mean TTFT changes are respectively
`-5.543889`, `-7.344421`, and `-14.101824` ms. The two 1--128-token requests
split one positive and one negative while retaining a `-0.568986` ms mean.

The candidate saves 48.361 ms of aggregate TTFT across eight requests. The
observed TPOT increase contributes about 51.860 ms over the 120 measured
inter-token intervals, offsetting that Prefill gain in total wall time. This
leaves prompt throughput lower by only 0.027671 tok/s (0.0161%), so the strict
validator correctly returns exit 3 because its single-round rule requires
both TTFT and prompt throughput to improve. The result is therefore evidence
that the production WY route improves Prefill latency, but it does not
override the formal external gate or promote a parity claim from this noisy
short-output round.

The frozen matched-first-eight vLLM reference remains 181.896870 prompt tok/s
and 1168.570642 ms mean TTFT. The production candidate is 10.327478 tok/s
(5.6777%) below that prompt-throughput floor and 94.030035 ms above its mean
TTFT. vLLM was not rerun.

Prompt throughput above is recomputed and cross-checked against
`workload_throughput.json`'s `Total Prompt tok/s`; EvalScope's known-zero
`benchmark_summary.json` `Input Throughput` field is not used.
