---
q3x_document:
  id: q3x-ordered-down-direction-20260928
  class: evidence
  status: frozen
  owner: project-maintainers
  authority: bounded ordered Down implementation, numerical admission and negative API direction
  effective: 2026-09-28
  last_reviewed: 2026-09-28
  supersedes: []
  superseded_by: []
  ssot_for: none
  review_trigger: materially new dataflow requires a separate successor work package
---

# Ordered Down producer/consumer: closed without promotion

`WP-DOWN-ORDERED-20260928` is complete and rejected on whole-API performance.
Both implemented dataflows preserved the bounded numerical panel and every
compared API output, but made Prefill slower. Candidate source integration,
CMake selection and profile branches were removed after the decision. The
ordinary 0.8.1 source route and installed binary are unchanged. No third
variant, parameter scan, further profile or qualification-only loop is open.
MTP remains deferred; no product target is amended.

This executes the direction selected by the
[architecture assessment](../non-mtp-architecture-assessment-2026-09-28/README.md).
The [machine-readable record](../../metadata/qwen36-27b-ordered-down-direction-2026-09-28.json)
binds source patches, binaries, payloads, numerical results, preflight, telemetry,
request identities, route receipts and cleanup. All raw runs remain under
`.q3x-work/down-ordered-20260928/`.

## Real API result

Use the same pinned model and real token-ID prefixes through the actual HTTP
service, without MTP or prefix/KV reuse. B1 is the installed production binary
`5ee029cc8877ad0086e5e6ea5bcf8d7674626e55cd96098d518252d33c14aa33`.
C1 and C2 are separately identified, non-installable whole-core admissions
built from `50f7323` plus their archived patch, with production eligibility
false. Requests run in the same 65/8192/40000 order in each fresh process.

These are **single-process engineering direction observations**, in B1-C1-C2
order, not a mirrored release comparison or a replacement for
[Current Status](../../CURRENT_STATUS.md)'s qualified performance baseline.
Both candidates are sufficiently negative to close without further repetition.

| Route | Prompt / output | Prefill seconds | Prefill token/s | External TTFT seconds | Decode token/s |
| --- | --- | ---: | ---: | ---: | ---: |
| Current B1 | 65 / 16 | 0.880 | 73.89 | 0.904 | 9.591 |
| First C1 | 65 / 16 | 0.879 | 73.93 | 0.906 | 9.590 |
| Corrected C2 | 65 / 16 | 0.879 | 73.93 | 0.898 | 9.588 |
| Current B1 | 8,192 / 256 | 16.742 | 489.30 | 16.764 | 9.222 |
| First C1 | 8,192 / 256 | 17.653 | 464.06 | 17.675 | 9.225 |
| Corrected C2 | 8,192 / 256 | 20.213 | 405.27 | 20.236 | 9.220 |
| Current B1 | 40,000 / 256 | 90.274 | 443.10 | 90.309 | 7.868 |
| First C1 | 40,000 / 256 | 94.837 | 421.78 | 94.873 | 7.870 |
| Corrected C2 | 40,000 / 256 | 107.057 | 373.63 | 107.093 | 7.844 |

C1 increases 8K/40K Prefill time by 5.44%/5.05%; C2 by 20.73%/18.59%.
All nine requests completed. Corresponding request hashes, output text, usage,
finish reasons and completed streams match. Each has a complete route receipt
with no forbidden fallback. The unchanged short path and nearly unchanged
Decode do not rescue the negative long-Prefill result.

All three process preflights and cache preparation passed; GPU clocks remained
at 1300.5 MHz under the audit. Maximum temperatures were 79.406C, 83.718C and
81.437C respectively, with no monitor/clock errors and zero-exit owned cleanup.
The API process windows total about 650 seconds, including startup and shutdown,
within the predeclared 30-minute budget. No new Nsight capture was needed.

## Implemented arithmetic and ownership

The original function remains Down followed by its BF16 residual boundary:

```text
B = the exact BF16 product of the incumbent's transformed FP4 and scale words
c0 = ordered FP32 MMA over K positions 64*t + [0,31]
c1 = ordered FP32 MMA over K positions 64*t + [32,63]
branch = BF16_RNE(global_scale * FP32_RNE(c0 + c1))
output = BF16_RNE(FP32(branch) + FP32(residual))
```

The producer decodes the same transformed Marlin operands, including its scale
permutation and exponent representation. It does not replace them with a
nominally equivalent canonical decode and move global scaling into the dot.
Both candidates preserve the two K16 updates per K32 chain, final FP32 merge,
and BF16 branch rounding before residual addition. An ordinary one-chain GEMM
was never substituted for the incumbent.

C1 materializes K-permuted weights and activation panels, computes two
FP32-output CUTLASS GEMMs and performs an explicit ordered merge/residual
kernel. It needs 784,465,920 bytes of dead GDN scratch, plus the existing
2,883,584-byte padded-tail reservation. There is no arena growth or persistent
all-layer BF16 copy. The complete conversion/read/merge costs are in API time.

After C1's negative API result, the only dataflow correction was C2: two warp
K partitions inside one CTA, CUTLASS's explicit FP32 partition merge, and an
output operator that rounds the branch before adding the residual. It removes
the activation permutation and both global FP32 partial matrices. Scratch
falls to one 178,257,920-byte decoded layer operand, plus the tail reservation.

The first C2 M64/N256/K64 instantiation generated 142 registers per thread
across 512 threads, exceeding the SM's 65,536-register budget and failing
before candidate execution. The resource-fit correction uses M64/N128/K64,
M32/N64/K32 warps and 256 threads, still with 142 registers/thread and zero
reported local spill storage. This was a resource admission correction,
not a timed tile search. No other tile, stage or swizzle was benchmarked.

Both executable routes select the new dataflow only for aligned prefixes of
at least 8,000 rows, fixed before API timing. Smaller prefixes and padded tails
retain their explicitly declared incumbent path before enqueue. Thus the
65-token request exercises existing prefix/tail behavior, while 8K and 40K
exercise all 64 layers of the new Down path. Existing stream ordering,
actual-row publication, model state and request-arena geometry remain intact.

## Numerical admission and its limits

The real-weight panel was frozen before timing by sorting all 64 Down layers
by mean block-scale value and taking ranks 0, 21, 42 and 63: layers
**2, 11, 44 and 17**. The record retains each layer's scale histogram and tensor
name, dtype, shape, shard range, payload hash and global-scale bits.

Each executable dataflow passed one synthetic finite-code case and those four
real-weight cases at M64/N5120/K17408, using deterministic BF16 activations and
residuals. Every case compared all 327,680 outputs: **1,638,400 bit-identical
BF16 outputs per dataflow**, zero nonfinite outputs. The synthetic case spans
all FP4 codes and finite nonnegative canonical scale codes; it has correctness
authority only. Real-weight cases are `checkpoint_weight_only`, not captured
real activation performance evidence. No component timing selected either path.

The first harness attempt omitted the incumbent's required dynamic-shared-
memory setup and failed before comparison; it was fixed and retained as a
failed setup run. The first C2 resource failure is also retained. Neither is
rewritten as a pass or counted as a numerical discrepancy.

Full-model API output agreement is an additional bounded observation, not proof
of all internal tensor equality. Complete state/logit, maximum-capacity,
capability and release qualification were **not** rerun after the negative
performance decision. No accuracy promotion relies on these admission checks.

## What this result changes in the earlier reasoning

The earlier profile correctly located a 17.56-second Down family. It did not
prove how much of that time was removable decode cost. Gate/Up's successful
ordinary CUTLASS path was a useful representation example, but its throughput
could not be transferred to Down while ignoring Down's two accumulation chains.

C1 demonstrates that externalizing those chains adds enough whole-path cost
to lose. C2 demonstrates that removing the new global intermediates is still
insufficient in this concrete implementation: preserving two accumulator
partitions constrains the executable tile/resources, while BF16 operands
increase data movement. Resource counts and source topology support that
explanation; they do **not** isolate physical DRAM traffic or prove one stall
category caused the full regression. No hardware ceiling or impossibility
claim follows.

The reusable result is the exact transformed-operand and ordered-reduction
mapping, plus the closed negative composition. It is not a retained optimization.
Do not reopen this version through an occupancy, tile or cache-policy scan.
A future materially different architecture needs its own product-connected
package and must preserve the same numerical obligations.

## Reproduction and closure

The self-contained [C1 patch](v1.patch) and [C2 patch](v2.patch) apply to
`50f7323`. They retain the actual kernel, isolated build flag, runner binding,
profile identity and work-package contract. Apply each only to an isolated
checkout inside `.q3x-work/`; never to the ordinary production tree. The record
also hashes the original driver and numerical harness retained in the workspace.

The candidate build used Release, `BUILD_TESTING=ON`,
`Q3X_BUILD_WHOLE_CORE_SERVICE_PRODUCTION=OFF`,
`Q3X_BUILD_WHOLE_CORE_EXACT_DECODE_ADMISSION=ON`, and
`Q3X_BUILD_ORDERED_DOWN_ADMISSION=ON`, targeting `qwen3x-eval-server`.
The CLI acknowledgement is `--candidate-profile whole-core-exact-decode`;
the actual health/deployment/witness identities distinguish ordered Down v1/v2.
Reference libraries in the admission build grant no production dependency
eligibility, and request receipts show no cuBLASLt execution.

The closeout removes all candidate runtime/build/profile branches. Only frozen
patches, this evidence, metadata and current-status/roadmap updates remain
tracked. The installed production binary is unchanged, so its preceding
numerical and reliability qualification retains its original scope.
