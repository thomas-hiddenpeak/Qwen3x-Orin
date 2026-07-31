# Natural-length cumulative Prefill baseline

Date: 2026-07-31 (Asia/Shanghai)

This is the first four-bucket external performance baseline for the cumulative
K128 Prefill direction. It uses the real pinned Qwen3.6-27B-NVFP4 checkpoint,
the OpenAI-compatible `/v1/completions` endpoint, and EvalScope 1.9.1. It is a
performance direction baseline, not a capability or production-promotion
claim.

## Runtime contract

- Authenticated A4 publication: 400/400 projections, physical packed-K64
  codes with one shared BF16 scale per K128 consumer group.
- Candidate mode: `cumulative-prefill`.
- Enabled selectors: native C64 GDN, token-parallel GDN convolution, and
  whole-span BF16 A/B.
- Rejected M128 generic and Down selectors: disabled.
- MTP: disabled/not used.
- cuBLASLt: absent from the production path.
- Concurrency: 1; warmup: 1; measured requests: 4 per bucket; generated
  tokens: 1; all 16 measured requests succeeded.

The exact ELF SHA-256 was
`059a7e838674d05795b8a00b512873bf6071a7a2765b4e0f9f3acc3bad25c88f`.
The server log SHA-256 was
`746b3351756bad02e2e90557ffb821742a5017f970fc5be7e5ff9676a726dc81`.

## EvalScope result

| Bucket | Mean input tokens | TTFT | Measured input rate | EvalScope total rate | Success |
|---|---:|---:|---:|---:|---:|
| P512 | 527.25 | 3488.54 ms | 151.125 tok/s | 151.412 tok/s | 4/4 |
| P1K | 1053.75 | 2453.29 ms | 429.471 tok/s | 429.877 tok/s | 4/4 |
| P2K | 1924.75 | 3696.51 ms | 520.642 tok/s | 520.911 tok/s | 4/4 |
| P4K | 3806.75 | 7282.97 ms | 522.658 tok/s | 522.795 tok/s | 4/4 |

The input rate is recomputed as the exact measured prompt-token total divided
by EvalScope test duration. EvalScope total rate additionally includes the one
generated token per request.

The P2K/P4K plateau is about 521--523 input tok/s, leaving a 3.83x gap to the
2,000 tok/s capability target. A naive P4K-rate extrapolation puts a 40K cold
prompt at about 76.5 seconds; the target's unavoidable first milestone is 20
seconds before separately validating actual P8K/P40K scaling.

## Short-prompt discontinuity

The server log proves that natural prompts at 481, 482, and 508 tokens used
`layer_major_prefill=0` and took 5.06--5.55 seconds, while 556 and 564 tokens
used `layer_major_prefill=1` and took about 1.624 seconds. The P512 aggregate
therefore exposes a dispatch cliff at the strict `prompt_tokens > 512`
boundary. This is a system route problem, not kernel timing noise, and is now
tracked as an independent vertical slice.

## Decision

This cumulative route becomes the next candidate baseline for real-path
experiments. It is retained because the fixed P2048 direction run reached
576.18 tok/s and all natural EvalScope requests completed, but it is not yet a
production default: K128/native-GDN capability validation remains outstanding.
The next performance gate is the two-CTA Gate+Up complete cell, followed by
prompt-span GDN and prompt-wide Attention/consumer-boundary work.
