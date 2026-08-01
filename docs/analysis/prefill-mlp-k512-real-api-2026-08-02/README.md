# Authenticated MLP K512 real-API direction gate (2026-08-02)

## Decision

The first complete MLP K512 route is **directionally retained but not
promoted**.  It is a real whole-runner improvement and a useful structural
baseline, but it is not the qualitative jump required by the 2,000 prompt
token/s target.

The external EvalScope/OpenAI result on the fixed natural P2K corpus is:

| Route | Success | Mean TTFT | Prompt throughput | Total throughput |
|---|---:|---:|---:|---:|
| K128 current-best comparator | 4/4 | 3,173.4 ms | 606.45 tok/s | 606.7634 tok/s |
| MLP K512 candidate | 4/4 | 2,782.6 ms | 691.62 tok/s | 691.9824 tok/s |
| MLP K512 + natural ceil128 | 4/4 | 2,720.5 ms | 707.40 tok/s | 707.7676 tok/s |
| Shared global-flow experiment | 4/4 | 3,004.7 ms | 640.58 tok/s | 640.8404 tok/s |
| Fragment-native v2 production candidate | 4/4 | 3,304.3 ms | 582.43 tok/s | 582.7297 tok/s |
| Retained v1 + ceil128 cumulative change | - | -452.9 ms (-14.27%) | +16.65% | +16.65% |

The retained v1 plus ceil128 candidate therefore passes the experiment gate
against the incumbent, but remains 2.83x below 2,000 prompt token/s.  The
ceil128 route change is itself +2.28% over the first K512 result.  No synthetic
timing is used in this decision.  The fragment-native v2 route later reached
the complete real-model OpenAI path, but its first external EvalScope result
was negative.  It is therefore **rejected** and is not enabled in the
production default.

## Real publication and production-shaped route

The independently requantized overlay contains all 192 MLP projections:
64 Gate, 64 Up, and 64 Down.  Gate and Up are `[17408,5120]`; Down is
`[5120,17408]`.  The authenticated payload is 8,623,226,880 bytes.

```text
manifest SHA256  6ab8818b34256646b4f1aca3d6cae1bb80425d30df1ab589a2c744a811351583
policy SHA256    5e4b31d0e93cd1ae13d1aebd66a9fd75151e6d3fb5ab448c1f7650ac127ef8e5
payload SHA256   541480dcad50227288530b22ed24e5984cef99f513d1df17ede8d4b702a2d5ec
layout           sm87_s4_n64_packed_k64_scale_k512_mlp_v1
```

The service startup contract proved 400/400 authenticated K128 base
projections, 192/192 authenticated MLP K512 overlays, the payload SHA, and an
enabled optimized Prefill dispatcher.  Every measured request used the real
checkpoint through `/v1/completions`; EvalScope 1.9.1 supplied one warm-up and
four serial measured requests with one output token.

The route is:

```text
post-attention norm
  -> K512 activation quantization
  -> fused Gate+Up K512 macrocell (two workspace windows)
  -> split-plane K512 product quantization
  -> Down K512 macrocell
  -> residual
```

Gate/Up share one activation quantization, so their activation clip ratios
must narrow to the same `float`.  Policy parsing and programmatic attachment
both now fail closed on a mismatch.

## Request-scoped NSys result

One natural 1,853-token request was captured through the same OpenAI server.
Model and sidecar loading are outside the `cudaProfilerStart/Stop` range.  The
request reported `prompt_prefill_ms=2840.46`.

```text
server ELF SHA256  f3f4b7d6927aa720c16a7f29d77f3535bb3eaa8d9190a60f6e8500a32add50c1
NSys report SHA256 6c5ad6bc861c7ee9d088dc39014b551449534b8dcd84617fbab3f2d08979d4cc
SQLite SHA256      22dbf23e16e2b90823102ea99b0857db2443d3ba36a886cf25fcdb3bb55c0a73
kernel CSV SHA256  8db849d77ff4438d012662900e49b5347c8bfb5af342264a587273b78a38db6d
```

The first 20 kernel families by total GPU time are:

| # | Kernel | Calls | Total ms | Share |
|---:|---|---:|---:|---:|
| 1 | `gateup_k512_m64n128_macrocell` | 128 | 888.499 | 31.5% |
| 2 | `prefill_gemm_m64n256_shared_k128` | 208 | 845.835 | 30.0% |
| 3 | `down_k512_m128n128_macrocell` | 64 | 514.770 | 18.2% |
| 4 | `a4_quantize_bf16_k512_split` | 64 | 67.187 | 2.4% |
| 5 | `residual_add` | 128 | 65.428 | 2.3% |
| 6 | `chunk_o_bv64` | 192 | 63.095 | 2.2% |
| 7 | `persistent_state_chunk64` | 192 | 54.298 | 1.9% |
| 8 | `headwise_rms_norm` | 129 | 51.870 | 1.8% |
| 9 | `a4_quantize_bf16_k128` | 128 | 38.159 | 1.4% |
| 10 | `causal_conv1d_token_parallel` | 192 | 35.978 | 1.3% |
| 11 | `value_head_recompute_chunk64` | 192 | 34.525 | 1.2% |
| 12 | `FlashInfer SinglePrefillWithKVCache` | 64 | 32.598 | 1.2% |
| 13 | `value_head_solve_chunk64` | 192 | 29.062 | 1.0% |
| 14 | `rms_norm_silu_rows8` | 192 | 24.223 | 0.9% |
| 15 | `a4_quantize_bf16_k512` | 64 | 20.179 | 0.7% |
| 16 | `full_attention_preprocess_prompt_wide_128` | 64 | 13.717 | 0.5% |
| 17 | `bulk_gqa_flashinfer_sigmoid_gate` | 64 | 10.001 | 0.4% |
| 18 | `bf16_ab_prefill_m64_n96_k64` | 48 | 7.149 | 0.3% |
| 19 | `bf16_gemv_pair_m16_projection_fused` | 144 | 6.764 | 0.2% |
| 20 | `compact_lower_gram_chunk64` | 192 | 6.291 | 0.2% |

The top three projection kernels consume 2,249.104 ms, or 79.7% of kernel
time.  Gate/Up plus Down alone consume 1,403.269 ms.  This proves that the
remaining gap is still projection architecture, not attention softmax or GDN
micro-tuning.

The 208 generic K128 launches are the complete non-MLP attention projection
inventory.  The natural 1,853 rows were padded only to M=1,856, while the
retained supermatrix requires M128.  Exact P2048 profiles therefore
overstated supermatrix coverage for natural traffic.

The follow-up changed whole-span K128 padding to ceil128 and repeated the
same external corpus.  The 1,853-token measured request fell from about
2,832 ms to 2,585 ms, saving about 247 ms.  Requests with 1,792, 1,906, and
2,148 tokens were already on M128 boundaries after the old padding and stayed
essentially unchanged.  This explains why a large per-affected-request route
repair becomes +2.28% over the four-request matrix rather than a universal
gain.

## Natural-M128 request profile

The same 1,853-token OpenAI request was captured again after ceil128 became
the whole-span K128 contract.  The request reported 2,601.43 ms while being
profiled.  Profiler-perturbed EvalScope throughput is not a performance
decision; this capture is used only for kernel attribution.

```text
server ELF SHA256  60d2d6f78ec5b87c4423b121baea20472b943a214c915b1f42af0d2b44aea85d
NSys report SHA256 594864599ec8b7e919a30477acd4ee5fbfa59beefb4901e378bf415486e8b614
SQLite SHA256      fb81cec54655c7265d126951e0da7b9392a8fd95d1941765e6d91152a771d91c
kernel CSV SHA256  67b8f3cc4056a75b655bd1c19e43716a48b0fa618f9dd06acb1196201dd0ab7d
```

The repaired projection inventory is:

| Projection family | Calls | Total ms | Share |
|---|---:|---:|---:|
| Gate+Up K512 | 128 | 888.941 | 34.4% |
| Down K512 | 64 | 523.248 | 20.3% |
| GDN Attention pair supermatrix | 48 | 336.135 | 13.0% |
| Attention-O supermatrix | 64 | 165.631 | 6.4% |
| Full-Attention Q/K/V supermatrix | 16 | 98.187 | 3.8% |

The old 208-call generic K128 family is absent.  The five projection families
now consume 2,012.142 ms, about 77.9% of captured GPU time.  Attention
supermatrices therefore worked as intended, but Gate+Up, Down, and Attention
projection architecture still dominate the route.

## Next structural replacement

The v1 K512 cells reduce scale application frequency but keep B in a large
shared-memory round trip.  Gate+Up uses 83,200 bytes and 16 warps; Down uses
128 KiB, only eight warps, and a P2048-specific persistent phase relationship.
The next work is a structural package rather than a stage-count scan:

1. Gate+Up first keeps global-flow ownership at M128 by processing N64 phases,
   so B fragments are reused across two M16 rows without increasing the
   incumbent B presentation.  In parallel, a stronger paired publication puts
   Gate and Up directly in MMA lane order so B can bypass shared memory and the
   same warp can finish SwiGLU without a cross-warp exchange.
2. Down keeps M128N128 but moves from eight M32N64 warps to sixteen M16N64
   warps.  A rolling K256 pipeline targets 16 active warps, at most 128
   registers/thread, zero local spill, and an explicit B-stationary owner
   schedule for every natural M rather than only M=2048.
3. A candidate reaches the production selector only after bitwise and resource
   admission.  Its first performance verdict is one real-model OpenAI request
   through external EvalScope; a negative direction stops the architecture.
4. Even a large MLP gain cannot by itself close 707.4 to 2,000 prompt token/s.
   Once the MLP structural package is measured, the same fragment-native
   treatment must cover the remaining 599.95 ms of Attention projections.
   NCU remains diagnostic for candidates that first show real-API benefit.

## Rejected shared global-flow experiment

Before changing the publication, one complete structural package was tested
with the existing authenticated v1 weights.  Gate+Up used logical M128N128
ownership with two N64 phases; Down used M128N128, 512 threads, 16 warps, and
a three-slot K256 ring.  Both kernels were bitwise exact, had zero local spill,
and used tail-balanced N-major scheduling on natural uneven M.

The real OpenAI/EvalScope result was negative:

```text
server ELF SHA256       74cf9232a2b1aedf615c61be0224c5e99cf7ed33ea0e860bb761e5db78eca042
success                 4/4
mean TTFT               3,004.66 ms
prompt throughput       640.58 token/s
total throughput        640.8404 token/s
change vs current best  -9.46% throughput, +10.44% TTFT
summary SHA256          0afa1af40099cef4527d5b4b26fb4b8dc81506ee8fe935dda4213a7c69de9047
provenance SHA256       9d5aeba4cdc93dd860c3049c898b2e188f57a9ba9c3bfc20e19d1263fc2a19d1
```

Every measured natural length was slower, so no additional noise harness or
NCU promotion work is justified.  The runtime selector was removed.  The
result falsifies the idea that occupancy plus a different shared tile is
sufficient; the next candidate must remove B's shared-memory round trip via
an equal-byte fragment-native publication.

## Fragment-native v2 production-candidate verdict

The stronger replacement passed kernel-level admission and was then integrated
as a production-shaped candidate.  Both consumers keep exact K512 numerical
semantics and change only the offline byte permutation of B:

| Consumer | CTA / warp ownership | Registers | Dynamic shared | Active CTA/SM | Local spill |
|---|---|---:|---:|---:|---:|
| Gate+Up paired | M64N64, 8 warps each owning M64N8 for both projections | 124 | 12,288 B | 2 | 0 B |
| Down | M64N128, 8 warps each owning M64N16 | 128 | 24,960 B | 2 | 0 B |

Gate and Up are computed in the same warp and finish SwiGLU directly from
registers.  Down loads two adjacent N8 fragments per warp and reuses them
across four M16 rows.  In both kernels B uses an aligned 128-bit global load
in MMA lane order and never enters shared memory; SASS contains direct
`LDG.E.128` plus `IMMA.16864`, while LDS is confined to runtime A.  Contract,
equal-byte bijection, K512/K1024, and model-K correctness tests pass bitwise.

The offline converter produced and authenticated one same-size
8,623,226,880-byte composite publication.  Its receipt binds the source v1
receipt, manifest, policy, payload, checkpoint identity, and required K128 base
publication.  The candidate is attached transactionally and selected only by
the explicit fragment-native admission selector.

```text
physical layout       sm87_s4_gateup_n64_paired_down_n128_fragment_native_scale_k512_mlp_v2
payload bytes         8623226880
payload SHA256        e3946d020ff44b454f7fbf31aacfdab4af8e593b2eb25ba58ddecb461893bd36
receipt SHA256        559f18d48b3acbedd569692bc89a64495f96dc00e8ef42ddeb8a2796110a0b25
manifest SHA256       e5aaf9ab1897c2b05d3dcf1760b1daf1d14e86e92b34464513ab0e7ae4cfb536
source-v1 receipt     e39795f13d15f83a93bd8152d9ef581975893baf1c42884af6632e0272aa57de
source-v1 policy      5e4b31d0e93cd1ae13d1aebd66a9fd75151e6d3fb5ab448c1f7650ac127ef8e5
```

The first performance verdict used the real publication, real checkpoint,
OpenAI `/v1/completions`, and external EvalScope 1.9.1 on the same natural P2K
corpus.  Four of four serial requests succeeded:

```text
test duration          13.2188 s
average input          1924.75 tokens
mean TTFT              3304.33 ms
prompt throughput      582.4281 token/s
total throughput       582.7297 token/s
vs 707.7676 baseline   -17.6665% throughput
vs 2720.5 ms baseline  +21.4604% TTFT
```

This is an unambiguous **REJECT**.  The selector does not enter the production
default and the v1 K512 plus natural-ceil128 route remains the retained
baseline.  Synthetic inputs were used only for correctness and smoke testing;
none of their timings contributed to the verdict.

```text
server ELF SHA256      6f2c84240e6b84d925e386f1c0269398449cd612849513b27a5f0abf41c6966f
server log SHA256      5cf765826fa94a7b47f26880d4d261c47dda375babeaa48b16c2f8d078d11458
summary SHA256         65a3b48fca4d09ed8d384035a8cc9eaf4df04ad046f509a3acc22600ead19e71
provenance SHA256      3148cb21c94312e45e89f4c7132023ddca3a3169eec5f9b0a1d261462aba38e1
```

### Fragment-native real-API NSys evidence

A request-scoped NSys capture through the same real OpenAI API route isolates
the regression in the MLP path.  The comparison below uses the same natural
P2K request as the retained natural-M128 profile:

| Kernel family | Natural-M128 baseline | Fragment-native v2 | Change |
|---|---:|---:|---:|
| Gate+Up | 888.941 ms (128 calls) | 1,278.063 ms (128 calls) | +43.77% |
| Down | 523.248 ms (64 calls) | 540.490 ms (64 calls) | +3.30% |
| MLP total | 1,412.189 ms | 1,818.553 ms | +28.78% (+406.364 ms) |

The non-MLP families are essentially unchanged: the GDN Attention pair moves
from 336.135 to 336.719 ms, Attention-O from 165.631 to 165.897 ms,
Full-Attention Q/K/V from 98.187 to 98.339 ms, and split K512 activation
quantization from 67.154 to 67.275 ms.  This isolates the measured regression
to MLP, and overwhelmingly to Gate+Up, rather than the API, corpus, Attention,
or quantization route.

```text
server ELF SHA256      6f2c84240e6b84d925e386f1c0269398449cd612849513b27a5f0abf41c6966f
NSys report SHA256     1584de510f7d30419c7c05e228d8bc8cc967ee5621b1e9dfa2b1cbd1f9899d73
SQLite SHA256          82dd4306ea91558fc40cb1a1018e57c177de5b498f4a1b1ae2c7a1cb8cfeb1e2
kernel CSV SHA256      6a6fc2529512334cbc4a80eca6421ba2918a04dc00f27553e6e16d817793ce59
```

The M64 Gate ownership launches twice as many M-axis CTAs as M128 ownership;
therefore each N tile must fetch the fragment-native B weights twice across
the same span.  That doubled B-read explanation is a structural inference
consistent with the +43.77% Gate timing, not an NCU-proven traffic
measurement.  The next candidate keeps the authenticated v2 payload and its
direct-B lane order, but restores balanced M128 ownership so those B fragments
can be reused across the larger M tile without introducing a publication
variable.

Evidence directories:

- comparator: `/home/rm01/q3x-k512-evalscope-p2048-baseline-4a90d1f`;
- candidate: `/home/rm01/q3x-mlp-k512-evalscope-p2048-systematic-pilot`;
- natural ceil128 candidate:
  `/home/rm01/q3x-mlp-k512-evalscope-p2048-natural-m128`;
- profiled EvalScope run:
  `/home/rm01/q3x-mlp-k512-evalscope-p2048-systematic-profile`;
- request trace:
  `/home/rm01/q3x-mlp-k512-nsys-p2048-systematic-request2.nsys-rep`;
- natural-M128 request trace:
  `/home/rm01/q3x-mlp-k512-nsys-p2048-natural-m128-request2.nsys-rep`;
- rejected shared global-flow run:
  `/home/rm01/q3x-mlp-k512-evalscope-p2048-global-flow`;
- rejected fragment-native v2 run:
  `/home/rm01/q3x-mlp-k512-evalscope-p2048-fragment-native`;
- fragment-native v2 request trace:
  `/home/rm01/q3x-mlp-k512-fragment-native-nsys-p2048-request2.nsys-rep`;
- fragment-native v2 publication:
  `/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4-q3x-mlp-k512-fragment-native/weights-mlp-k512-fragment-native.bin`.
