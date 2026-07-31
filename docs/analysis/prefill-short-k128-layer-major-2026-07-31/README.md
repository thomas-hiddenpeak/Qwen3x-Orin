# Short K128 layer-major real-path direction

Date: 2026-07-31 (Asia/Shanghai)

The natural P512 EvalScope baseline exposed a strict dispatch cliff: prompts
at or below 512 tokens used the legacy tile-major runner, while prompts above
512 used the optimized layer-major runner. This candidate admits only
P480--P512 with a complete authenticated K128 A4 inventory behind the
default-off selector:

```text
Q3X_RUN_SHORT_PREFILL_LAYER_MAJOR_ADMISSION=1
```

Projection M is internally padded to 512. State, KV, Attention, residual, and
the externally reported prompt length remain the exact logical M. P479 and
shorter prompts, K64 or unauthenticated inventories, trace mode, decode, and
the default route are unchanged.

## Real-path result

Both sides use the real Qwen3.6-27B-NVFP4 checkpoint, the authenticated K128
publication, the OpenAI `/v1/completions` path, one generated token, and the
same natural P512 corpus. The candidate also uses the retained native GDN/conv
and whole-span BF16 A/B selectors. The complete-cell-v2 selector is disabled.

| Prompt | Baseline route/time | Candidate route/time | Speedup | Candidate rate |
|---|---:|---:|---:|---:|
| P481 | tile-major / 5058.86 ms | layer-major / 1572.97 ms | 3.216x | 305.79 tok/s |
| P482 | tile-major / 5187.10 ms | layer-major / 1559.90 ms | 3.325x | 308.99 tok/s |
| P508 | tile-major / 5549.87 ms | layer-major / 1549.56 ms | 3.582x | 327.83 tok/s |
| P564 | layer-major / 1623.85 ms | layer-major / 1625.18 ms | 0.999x | 347.04 tok/s |
| P556 | layer-major / 1623.60 ms | layer-major / 1624.39 ms | 1.000x | 342.28 tok/s |

Using the same four measured prompts as the prior EvalScope P512 bucket
(P481/P564/P508/P556), candidate server-side Prefill totals 6372.10 ms for
2109 input tokens, or 330.974 input tok/s. The prior external EvalScope total
rate was 151.412 tok/s (151.125 input tok/s after removing four output
tokens), so the expected bucket-level improvement is about 2.19x. A new
EvalScope candidate run is still required for external statistical authority.

## Capability boundary

The P481 request completed successfully, but the first generated text changed
from the baseline route's `由于` to the candidate route's `以下是`. This does
not invalidate the real performance direction, because the optimized
layer-major path changes numerical execution order and the cumulative K128/GDN
bundle is already capability-pending. It does block production promotion.
The selector must remain opt-in until public EvalScope capability validation
passes the project's accuracy thresholds; exact first-token equality is not
claimed.

## Decision

Retain the route as a high-value system candidate. It removes the artificial
512-token cliff without affecting prompts already above the boundary. Add an
explicit EvalScope candidate mode, run the P512 external performance bucket,
then include this route in the later cumulative capability gate. Do not merge
it into the production default on timing evidence alone.
