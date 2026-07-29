# FP8 projection-supermatrix v2 external direction gate

Status: promoted to the exact-C512 production path after passing the real
OpenAI-compatible `evalscope perf` direction gate, focused correctness tests,
and real-checkpoint C256/C512 validation. It is not a release-performance or
vLLM-parity claim.

## Decision first

The exact-codebook v2 candidate improves both required whole-runner metrics
against the immediate native baseline and preserves every generated output:

| EvalScope perf8 metric | Native baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 2,845.220029 ms | 2,775.288518 ms | -69.931511 ms (-2.4579%) |
| Prompt throughput | 111.240248 tok/s | 113.030502 tok/s | +1.790254 tok/s (+1.6094%) |
| Total wall time | 35.787407 s | 35.220581 s | -0.566825 s |
| Successful requests | 8/8 | 8/8 | unchanged |
| Exact generated outputs | - | 8/8 | passed |

The validator decision is `advance_to_internal_validation`. The 32-request
confirmation, long-output run, capability suite, and test-set matrix were not
run: this stage uses the cheapest real external performance gate to reject or
retain an architecture.

## Production promotion

Before removing the gates, a second production-like same-ELF pair was built
with every GDN admission disabled. It reproduced the direction independently:

| EvalScope perf8 metric | Native baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Mean TTFT | 2,937.757735 ms | 2,865.285808 ms | -72.471927 ms (-2.4669%) |
| Prompt throughput | 108.955249 tok/s | 110.714190 tok/s | +1.758941 tok/s (+1.6144%) |
| Total wall time | 36.537937 s | 35.957450 s | -0.580486 s |
| Successful requests | 8/8 | 8/8 | unchanged |
| Exact generated outputs | - | 8/8 | passed |

The final no-gate production binary then measured 2,866.941191 ms mean TTFT
and 110.686236 prompt tok/s. Its 8/8 outputs matched the production-like
candidate, and its TTFT differed by only 1.655383 ms (0.0578%). This closes
the discrepancy seen in an isolated absolute run: the promotion decision is
the paired baseline-to-candidate delta, not an unpaired historical number.
The validator intentionally labels this second comparison `reject_direction`
because it is a candidate-to-candidate equivalence smoke, not another positive
direction gate.

Exact C512 now prepares the complete arena unconditionally for the SM87
backend. The ready log observed `enabled=1`, `projections=208`, and
`bytes=7214202880`; no CMake option or runtime environment selector is needed.
C256 and the reference backend retain zero supermatrix statistics. A failed
C512 inventory, allocation, memory-margin check, pack, or attach aborts engine
creation instead of falling back.

## Real P513 fast direction

Before external measurement, one same-ELF real-checkpoint P513/max1 pair used
the native generation path. Both sides generated token `9419` (`Hello`) and
committed 513 steps.

| Metric | Native baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Prefix execution | 1,929.970 ms | 1,817.521 ms | -112.449 ms (-5.8265%) |
| TTFT | 2,054.007 ms | 1,943.873 ms | -110.134 ms (-5.3619%) |

The real-checkpoint bulk-Prefill E2E test subsequently passed both P257/C256
and P513/C512 against the frozen C64 oracle, including token IDs, text,
scheduler steps, and prompt-token hashes.

## Architecture

The candidate replaces 208 independent C512 FP8 projection calls with 128
group launches:

- 48 merged linear-attention QKV/Z groups;
- 16 merged full-attention Q/K/V groups; and
- 64 attention-output projections.

Each projection retains its own weight scale and output address. The kernel
uses 16 persistent CTAs, 256 threads, M64xN256xK64 ownership, a four-stage
global-to-shared `cp.async` ring, and a two-slot shared-to-register pipeline.
A is stored in an LD64 row-XOR layout and loaded directly with
`ldmatrix.x4`; packed FP8 B remains in shared and is decoded into registers.
An exact 256-entry BF16 codebook is initialized once per persistent CTA.

The equal-byte engine-lifetime sidecar covers all 208 projections and occupies
7,214,202,880 bytes (6.71875 GiB). It is mutually exclusive with the older
2.34375 GiB QKV-only Prefill sidecars. Decode output-projection sidecars remain
unchanged. Production C512 creation hard-fails if the post-create device-memory
margin cannot be preserved; it never falls back to cuBLAS or cuBLASLt.

Resource audit of the exact measured ELF:

| Shape | Registers/thread | Static shared | Dynamic shared | Local/stack |
| --- | ---: | ---: | ---: | ---: |
| K5120, QKV/Z pair | 182 | 512 B | 98,304 B | 0/0 B |
| K5120, Q/K/V triple | 169 | 512 B | 98,304 B | 0/0 B |
| K6144, O | 186 | 512 B | 98,304 B | 0/0 B |

All forms satisfy the locked `<=255` register and zero-spill constraints and
run at one CTA/SM.

## Correctness closure and rejected variants

The first v2 direction cell used the old `2^-120` exponent-bias decode. It was
fast (P513 Prefix 1,867.817 ms), but only 7/8 external outputs matched. A
register-level probe proved the direct `ldmatrix`, B-fragment, MMA, and output
lane mappings bitwise identical to WMMA. The actual defect was numerical:
small FP8 operands entered the FP32 subnormal region after the artificial
shift, so Tensor Core arithmetic could not preserve production semantics.

Two repairs were rejected before the retained design:

- per-fragment scalar exact repair: P513 Prefix 2,321.896 ms;
- branch-heavy exact scalar conversion: P513 Prefix 2,139.393 ms.

The retained per-CTA exact codebook removes that underflow and the repeated
scalar conversion. It is both exact and faster than the approximate cell.

## External command

Both native runs used the same corpus and command contract:

```bash
uvx --from 'evalscope[perf]==1.9.1' evalscope perf \
  --model qwen3.6-27b-nvfp4 --api openai \
  --url http://127.0.0.1:18080/v1/completions \
  --tokenizer-path /home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4 \
  --dataset line_by_line --data-source local \
  --dataset-path /tmp/q3x-sharegpt-false-thinking-33.jsonl \
  --number 8 --parallel 1 --warmup-num 1 --num-workers 1 \
  --max-tokens 16 --temperature 0 --seed 42 \
  --stream --tokenize-prompt --no-test-connection \
  --outputs-dir OUT --name NAME --no-timestamp
```

Corpus SHA-256:
`bd2091cdc0599ac59ab881efc6d1307a03bb149f49e6fd728a99d926ccc67989`.
The final same-ELF native baseline is
`/tmp/q3x-evalscope-fp8-v2-final-baseline/baseline/parallel_1_number_8`.
The validated report is
`/tmp/q3x-evalscope-fp8-supermatrix-v2-final-short-direction.json` with
SHA-256
`c4a2c7614a289a7c92173db156e891bd06643c86467c2ee007a132566a363ccd`.

## Scope boundary

This result retains a cumulative native improvement. It does not claim vLLM
parity: the frozen vLLM reference remains the end-stage market floor and is
not restarted per candidate. The production-like result is still only 60.8%
of the frozen vLLM prompt throughput and has 2.50x its mean TTFT, so the next
work remains a larger Prefill architecture change rather than a broader test
matrix. cuBLASLt remains reference-only, MTP is disabled, and synthetic
matrices have no performance authority.
