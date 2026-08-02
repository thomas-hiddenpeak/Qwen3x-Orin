# GDN prompt-span macro real-API direction result

Date: 2026-08-03

Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill

Authority: real checkpoint, OpenAI `/v1/completions`, external EvalScope 1.9.1

## Decision

The full-prompt persistent-state GDN macro is **rejected as a cumulative
production candidate in its current form**. It executed the intended route
for every Linear-Attention layer, but made the real P1853 API request slower.
The promoted A-exchange/B4 cumulative route remains the performance baseline.

No P2K4 closure was run after the negative P2K1 direction result. Synthetic
data was used only for bit-exact correctness and resource checks and did not
contribute to this performance decision.

## Structural candidate

For each Linear-Attention layer and projection span, the candidate replaces
the incumbent per-C512 GDN state/output consumers with one C64 prompt-span
macro CTA per value head. The macro owns recurrence state across all chunks,
preserves the incumbent BF16 state round/reload every eight chunks, fuses
solve/transform, W/U, recurrence, Vnew, raw output, RMSNorm, SiLU gate and
final-state publication, and supports exact `state_input == state_output` and
`output == silu_gate` aliases.

The first production direction path intentionally reuses the authenticated
C512 token-parallel Conv+compact-Q/K producer in slices, then launches one
whole-span gate producer, one raw-Gram producer and one macro. It is a
structural direction probe, not the intended final four-kernel preprocess
layout. The selector is default-off and fail-closed. The runner validates all
span shapes before any GPU mutation, and engine preflight requires the A4
whole-prompt publication, native C64 selector and token-parallel Conv selector.

## Same-ELF real API P1853

Both routes used the same Release server ELF, authenticated real checkpoint,
K256 A4 and K512 MLP publications, natural P1804 warmup, natural P1853
measured prompt, one output token, and external EvalScope through the OpenAI
streaming API. The only selector delta was adding:

```text
Q3X_RUN_GDN_PREFILL_PROMPT_SPAN_MACRO_ADMISSION=1
```

Server ELF SHA-256:

```text
d6a09cbbbcb7f383c5d0ccf00043d52bbd073cdc524c0845c25e79fb97f3afc5
```

| Route | EvalScope TTFT | Server Prefill | Effective prompt throughput | Total throughput |
|---|---:|---:|---:|---:|
| Promoted native-C512 baseline | 1,939.87 ms | 1,934.91 ms | 955.22 tok/s | 955.7111 tok/s |
| Prompt-span macro candidate | 2,061.71 ms | 2,056.71 ms | 898.77 tok/s | 899.2352 tok/s |
| Change | **+121.84 ms (+6.281%)** | **+121.80 ms (+6.295%)** | **-56.45 tok/s (-5.910%)** | **-56.4759 tok/s (-5.909%)** |

Both requests succeeded and returned the same one-token text response. The
server's request-local accounting proves the intended mutually exclusive
route replacement:

```text
baseline P1853:
  gdn_chunk64_native_launch_hits=192
  gdn_chunk64_native_logical_token_hits=88944
  gdn_prompt_span_macro_launch_hits=0
  gdn_prompt_span_macro_logical_token_hits=0

candidate P1853:
  gdn_chunk64_native_launch_hits=0
  gdn_chunk64_native_logical_token_hits=0
  gdn_prompt_span_macro_launch_hits=48
  gdn_prompt_span_macro_logical_token_hits=88944
```

The P1804 warmup showed the same direction: server Prefill increased from
1,927.81 ms to 2,051.11 ms (+123.30 ms, +6.396%). This rules out a measured
request-only inversion.

Evidence:

```text
baseline root       /tmp/q3x-gdn-prompt-span-same-elf-baseline-p2k1
candidate root      /tmp/q3x-gdn-prompt-span-same-elf-candidate-p2k1
baseline summary    f9cff8df7e54e185267aabf5163854411806ec81c19a5bdda8c75b7db7a31411
candidate summary   ff70dff7fe0eedf480d836b6229695e3225e2b66f6a8136f9e5f35b6b544045e
baseline provenance 46606e530f626754bd2a6cf6d5ea58e8d354f110871a67258247d50507c8fb90
candidate provenance 729c516552835fc274a23e2f12dc49a20d67c0702a1b93f7f410e4e660f36ebd
```

## Correctness and integration evidence

The isolated P545 oracle is bit-exact at transform, W, U, C512 boundary,
Vnew, raw output, final state and gated output, including true in-place state
and output aliases. The production kernel reports 210 registers/thread,
115,200 bytes dynamic shared memory, zero local memory and one active CTA/SM.

The production build and targeted tests passed:

```text
gdn_prefill_prompt_span_macro_c64_component PASS
reference_runner_host                           PASS
reference_engine_control                        PASS
pure_prefill_evalscope_harness_test              63/63 PASS
```

## Next attribution step

The negative result is large enough that no statistical closure or local
parameter sweep is justified. Request-scoped NSys attribution should compare
the candidate against the existing promoted A-exchange/B4 P1853 profile and
separate:

1. sliced token-parallel Conv+compact-Q/K and D2D publication;
2. prompt-span gate and raw-Gram producers;
3. the persistent macro itself;
4. all unchanged GEMM and Attention families.

Only an attribution that points to a different global dataflow (for example,
a true full-span fused preprocess plus a lower-cost state consumer) can reopen
this direction. The current implementation must not be promoted or
incrementally micro-tuned.
