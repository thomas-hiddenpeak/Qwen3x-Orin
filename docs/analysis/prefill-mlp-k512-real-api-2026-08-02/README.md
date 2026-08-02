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
| Fragment-native M128N32 L1 experiment | 4/4 | 3,527.4 ms | 545.60 tok/s | 545.8841 tok/s |
| Fragment-native direct M128N64 one-CTA | 4/4 | 2,736.2 ms | 703.35 tok/s | 703.7168 tok/s |
| Direct M128N64 + Down M128N256 one-CTA | 4/4 | 2,807.8 ms | 685.42 tok/s | 685.7787 tok/s |
| Fragment-native M64N128 16-warp one-CTA | 4/4 | 3,236.2 ms | 594.69 tok/s | 594.9980 tok/s |
| Fragment-native M128N64 staged paired-B | 4/4 | 2,744.8 ms | 701.15 tok/s | 701.5126 tok/s |
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

## Rejected M128N32 same-CTA L1 experiment

The first M128 follow-up deliberately answered one bounded structural
question: can two adjacent M64 warp owners make the second direct-B load cheap
enough through same-CTA `.ca` reuse while preserving two active CTAs per SM?
The kernel uses an M128N32 CTA, 256 threads, a three-slot K128 A-only ring,
119 registers/thread, 24,576 bytes of dynamic shared memory, two active
CTAs/SM, and no local stack or spill.  Its K512/K1024 and model-K5120
correctness cases are bitwise exact.

It then ran through the authenticated v2 publication, the real checkpoint,
OpenAI `/v1/completions`, and external EvalScope 1.9.1 on the fixed natural
P2K corpus.  Four of four requests succeeded:

```text
test duration          14.1111 s
average input          1924.75 tokens
mean TTFT              3527.39 ms
prompt throughput      545.5989 token/s
total throughput       545.8841 token/s
vs fragment-native v2  -6.3234% prompt throughput, +6.7505% TTFT
vs retained v1         -22.8727% prompt throughput, +29.6596% TTFT
```

This is a hard **REJECT** and is not production eligible.  Covering the same
M128N64 output region requires two N32 CTAs: B load/scale/feed instructions
are not removed, while A presentation doubles.  The real result therefore
rejects L1 timing plus two-CTA residency as the missing qualitative mechanism.
No stage-count or cache-hint scan follows this failure.

```text
server ELF SHA256      c2a44a47ce98d6aac21d33081640a9382ba36868566f6edab5175d21603ba665
server log SHA256      536df54215fb181ed8ba7c0c3defd7ec9e20a8d1f1e797e3ecd06df65095f5bc
summary SHA256         4d90c1b702ea08c96547539ffc6a4d720d83a617eb2ec7a1488df7a72a49f247
provenance SHA256      37c0da68e63e447b36e58c0618c358133da98ae5fa8e70427b5d5ff0020edfaa
```

The next qualitative Gate+Up candidate is direct M128N64 with one high-register
persistent CTA per SM.  Relative to M64N64 it can actually halve B loads,
scale handling, and fragment feed by reusing each B record across eight M16
panels.  The experiment admits lower occupancy only if compilation proves zero
stack/spill; its first performance authority remains the same real API and
external EvalScope P2K gate.

## Retained fragment-native M128N64 one-CTA development baseline

The direct follow-up implements the mechanism that M128N32 did not.  One
256-thread CTA owns M128N64, and warp `w` owns M128N8.  Each warp loads one
paired Gate/Up B record per K64 plane and feeds that record across eight M16
panels.  Relative to the v2 M64N64 consumer, B record loads, K512 B-scale
handling, and fragment feed are halved for the same logical output.  A uses a
two-stage K512 `cp.async` ring; B remains a direct register operand and never
enters shared memory.

The resource and instruction gates are:

```text
threads / ownership     256 / 8 warps x M128N8
registers               224 per thread
dynamic shared          66,560 bytes
active CTA/SM            1
local stack / spill      0 / 0 bytes
static B LDG.E.128       8 (16 in the rejected duplicate-M-owner draft)
static IMMA.16864        128
static LDGSTS            20
separate STS             0
```

Contract validation and four BF16 bit-exact cases pass, including K5120 and
M1920 persistent-stride coverage.  The first performance authority was again
the real checkpoint, authenticated v2 publication, OpenAI API, and external
EvalScope 1.9.1 fixed natural P2K corpus:

```text
success                 4/4
test duration           10.9462 s
average input           1924.75 tokens
mean TTFT               2736.17 ms
prompt throughput       703.3514 token/s
total throughput        703.7168 token/s
vs fragment-native v2   +20.7619% prompt throughput, -17.1944% TTFT
vs retained v1          -0.5723% prompt throughput, +0.5760% TTFT
```

This is **retained as the fragment-native development baseline** because it
is a large cumulative improvement over its v2 parent.  It is **not promoted
to production** because the retained v1 K512 plus natural-ceil128 route still
wins the whole-product comparison, albeit by only 0.57%.  This applies the
separate experiment and production gates: a strong intermediate mechanism is
preserved without displacing a faster production route.

```text
server ELF SHA256      3309dfb51ebbfcdd0d59e0ebd9c40ea20f659a18a36fbb365b94e45ad3483882
server log SHA256      1da417b4d5a56b621ac0359c2f175d2a9c16faa416f6c8c0b60014089844f115
summary SHA256         5e6e701c2eecf374dc32ca00359925bb61bbac4252935e3a5881708e9e0391bb
provenance SHA256      41b8c643cb83cdd8ccac175ca667516a0069c22a46a4b0c42f476346146e6f5c
```

### Request-level structural attribution

NSys captured only the second real EvalScope/OpenAI request.  Its timing is
diagnostic, not a throughput verdict:

| Kernel family | Retained v1 | Fragment-native v2 | Direct M128N64 | Direct vs v2 | Direct vs v1 |
|---|---:|---:|---:|---:|---:|
| Gate+Up | 888.941 ms | 1,278.063 ms | 969.944 ms | -308.119 ms (-24.11%) | +81.003 ms (+9.11%) |
| Down | 523.248 ms | 540.490 ms | 543.331 ms | +2.841 ms (+0.53%) | +20.083 ms (+3.84%) |
| MLP total | 1,412.189 ms | 1,818.553 ms | 1,513.275 ms | -305.278 ms (-16.79%) | +101.086 ms (+7.16%) |

The direct M128 Gate kernel appears exactly 128 times and accounts for
969.944 ms; the unchanged Down kernel appears 64 times and accounts for
543.331 ms.  The next three projection families remain Attention pair at
335.867 ms, Attention-O at 165.610 ms, and full-QKV at 98.069 ms.  This
confirms that real M reuse, rather than L1 timing, caused the recovered Gate
performance.  It also shows that a Gate-only cleanup cannot provide the next
100-ms whole-P2K transition: the next complete cell must reduce another
projection family's logical operand/feed work while preserving this Gate
mechanism.

```text
NSys report SHA256     10219abb8746426d888bcc924dca692c3c5bbf380d23107a87f53f771eb593d4
SQLite SHA256          38ff16d1f9637faf7146e0447a119534fb8af948be675814252922834ed1df5f
kernel CSV SHA256      321019a3d585d4f8e3c45fc5d48562c5fb3744c9cac60c4ebb4693a912b8ba55
profile log SHA256     ecb936f20eacdc77aed01026d74e0778fd798b85ed13a790b01a91e4184a6b66
```

## Rejected fragment-native Down M128N256 one-CTA cell

The next complete structural cell doubled both axes of the v2 Down consumer.
One 256-thread CTA owns M128N256; each warp owns M128N32 in two N16 phases,
and one 128-bit B record feeds eight M16 panels.  A uses a two-stage M128K512
`cp.async` ring.  The candidate retains the authenticated v2 payload byte ABI
and has no synthetic performance gate.

Resource and correctness admission passed before runtime integration:

```text
registers/thread       255
dynamic shared         67,072 bytes
active CTA/SM          1
stack/local/spill      0 / 0 / 0
SASS                    16 B LDG.E.128, 256 IMMA, 20 LDGSTS, 0 STS/LDL/STL
correctness             bit exact at K512 and K1024
model-shape differential M128 x N5120 x K17408, 655,360 BF16 outputs exact
```

The first timing verdict then used the real checkpoint, authenticated payload,
OpenAI `/v1/completions`, and external EvalScope 1.9.1.  Four of four natural
P2K requests succeeded:

```text
test duration          11.2325 s
average input          1924.75 tokens
mean TTFT              2807.75 ms
prompt throughput      685.4225 token/s
total throughput       685.7787 token/s
vs direct-M128 baseline -2.5491% throughput, +71.58 ms TTFT
```

This is a real-API **REJECT**.  The candidate remains default-off and does not
replace either the fragment-native M128 development baseline or the faster v1
production route.  Synthetic payloads were used only for bitwise admission.

The per-request log also exposed a structural natural-length failure.  M14 and
M15 spans were roughly flat, while the 2,148-token request pads to M2176, or
17 M128 tiles.  Sixteen resident owner CTAs leave one CTA responsible for two
tiles on every N stripe, so that CTA executes 40 cells while the other fifteen
execute 20.  The request regressed by about 300 ms.  A rotated tail-owner
schedule could remove this imbalance, but it is not pursued because the
aligned P1853 request profile proves that the cell itself is only a small
improvement:

| Kernel family | Direct M128 baseline | Down M128N256 | Change |
|---|---:|---:|---:|
| Gate+Up | 969.944 ms | 970.464 ms | +0.520 ms (+0.05%) |
| Down | 543.331 ms | 526.767 ms | -16.564 ms (-3.05%) |
| Attention pair | 335.867 ms | 336.005 ms | +0.138 ms (+0.04%) |
| Attention-O | 165.610 ms | 165.611 ms | +0.001 ms |
| Full-Attention Q/K/V | 98.069 ms | 98.075 ms | +0.005 ms |

Only 16.6 ms is recoverable on an already aligned natural request, far below
the 100-ms structural continuation gate.  Correcting its long-tail schedule
would turn a negative result into at most another local increment, not the
qualitative transition needed to reach 2,000 prompt token/s.  Work therefore
returns to the dominant 970-ms Gate+Up architecture rather than tuning this
Down cell.

```text
server ELF SHA256      0247e1c2b421d32ff0767e0c1d31025cafbfc95c725b729c247f7d9fe0eeee40
server log SHA256      587a0d175dc5f5e3717abb2de777bcb814c5292ee4389672b1934697b4a43940
summary SHA256         22e4114222299218c1dfddacb02b9986b34fce0741b540ef44c8111d0dba38cb
provenance SHA256      b5f9e4fed5610068a027a965c93328cb1c5e162127ac9384ca24406b67525a0d
NSys report SHA256     e16bfc16f6d90ea17ddf053feb79dfd68adbc2019a316cd8ea82e44d2f21bb1b
SQLite SHA256          35079ff7b116f1c86d9b8fa9ad4c2900788119bcbe05c0bc2ca129e21af6268a
kernel CSV SHA256      de30c91417f832a4c8906481124120fc2b7ae3ea7ad918777993e9b2c132a8e7
profile log SHA256     a597da35915677e6cc6f45f7f2178cb4fc71102adf964bd40f4204fc3217e1a1
```

## Rejected fragment-native M64N128 16-warp one-CTA skeleton

The first Gate+Up replacement after the Down stop was intentionally a whole
execution skeleton rather than another tile-variable scan.  A 512-thread CTA
owns M64N128.  Its 16 warps each retain the existing paired M64N8 Gate+Up
fragment and exact same-warp SwiGLU, while only the first 256 threads issue the
shared M64K128 `cp.async` ring.  Sixteen persistent CTAs stride a linear cell
space, so the implementation consumes two adjacent authenticated v2 N64
payload blocks without adding a weight format or workspace.

Resource and correctness admission passed:

```text
registers/thread       120
dynamic shared         12,288 bytes
active CTA/SM          1
local bytes            0
correctness             BF16 bit exact at K512, K1024, persistent M1088,
                        and model K5120
```

The authoritative first timing was the real checkpoint through the OpenAI
API and external EvalScope 1.9.1.  Four of four measured natural P2K requests
succeeded:

```text
test duration          12.9463 s
average input          1924.75 tokens
mean TTFT              3236.20 ms
prompt throughput      594.6890 token/s
total throughput       594.9980 token/s
vs production v1      -15.9331% throughput, +515.70 ms TTFT
P1853 prompt prefill    3223.74 ms vs 2584.91 ms (+638.83 ms)
```

This is a hard **REJECT**.  It missed the required 100-ms improvement by more
than 600 ms in the opposite direction, so no parameter sweep follows and the
production v1 route remains locked.

A request-scoped NSys capture explains why reducing A presentation was the
wrong structural objective:

| Kernel family | Production v1 | M64N64 v2 | Direct M128N64 | M64N128 16-warp |
|---|---:|---:|---:|---:|
| Gate+Up | 888.941 ms | 1,278.063 ms | 969.944 ms | 1,472.018 ms |
| Down | 523.248 ms | 540.490 ms | 543.331 ms | 541.856 ms |

The new Gate kernel alone regressed 583.077 ms (+65.59%) against production
and 193.955 ms (+15.18%) against the M64N64 direct-B consumer.  Down and the
sequence path stayed approximately unchanged.  Therefore the end-to-end loss
is localized to Gate+Up.  The comparison shows that 16 resident warps are not
an adequate design target by themselves.  Both this kernel and the faster
production macrocell are 512-thread, 16-warp, one-CTA/SM kernels and present
the same total code bytes per M64N128 cell.  The actual structural regression
is that direct B performs a synchronous global load immediately before use,
uses an 81-barrier K128 ring instead of the production kernel's 42-barrier
K256 A+B pipeline, and abandons fixed-M-owner scheduling.  It did not reduce
total operand presentation; it exposed B latency and doubled synchronization.
The next architecture must combine M128 B reuse with staged paired-B loads,
exact same-warp SwiGLU, a K256 ping-pong pipeline, and fixed-M ownership.  It
must not repeat direct-B or linear-cell scheduling.

The real run also exposed a harness fail-fast gap: the first launch was given
the base A4 policy instead of the K512 policy and the server discovered the
schema mismatch only after model loading.  The harness now validates the v2
receipt schema/layout/size plus its source-v1 policy SHA and payload SHA before
starting the server.  This is an evaluation-infrastructure correction, not a
performance result.

```text
server ELF SHA256      79c6de407d9de44b010e9196958f0f3ee55037b232d33fa7a9c4418d96d90be1
server log SHA256      75734addac598ed946b85ddc192d297da804c34febbb7db8d44d392731783960
summary SHA256         f1877016908b7b7f72f5167aee385b50ae39b21af32268740056247f17752fd6
provenance SHA256      16fa10750cd05ac20d75b125345e21528522b3f60a1293c547566a726c71f331
NSys report SHA256     b89903be3a57f034b8f750aaba93c066feb3b60d53c3b1a6ed26762bb0d4c5fc
SQLite SHA256          8b693e0945baba84eef580c73ae610636b5dc6380d38cd3ab154def55022053a
kernel CSV SHA256      d0b810d764897b37793895dc4f6e7ef6d0179bb437ce91c2bcf8a252c4b508e7
```

## Rejected fragment-native M128N64 staged paired-B skeleton

The next Gate+Up replacement combined the mechanisms that the M64N128
failure had separated incorrectly.  One 512-thread CTA owns M128N64.  The
upper and lower eight-warp crews reuse one paired Gate+Up B fragment, keep
same-warp SwiGLU, stage A and B with a two-stage K256 `cp.async` pipeline,
and use fixed M owners so all SMs traverse the same N64 weight stream.

Independent resource and correctness admission passed before any timing:

```text
registers/thread       128
dynamic shared         66,560 bytes
active CTA/SM          1
stack/local/spill      0 / 0 / 0
correctness             BF16 bit exact at K512, K1024, M1920/N128,
                        and model K5120
SASS                    20 LDGSTS, 154 LDS, 64 IMMA, no LDG.E.128
synchronization         41 CTA barriers per model-shape cell
```

The first performance verdict again used only the real checkpoint,
authenticated v2 publication, OpenAI `/v1/completions`, and external
EvalScope 1.9.1.  Four of four measured natural P2K requests succeeded:

```text
test duration          10.9806 s
average input          1924.75 tokens
mean TTFT              2744.78 ms
prompt throughput      701.1483 token/s
total throughput       701.5126 token/s
vs production v1      -0.8838% throughput, +24.27 ms TTFT
```

This is an end-to-end **REJECT**.  It did not meet the structural continuation
gate of at least 100 ms TTFT reduction, so it remains default-off and does not
replace the v1 production route.  No parameter sweep follows.

The result is nevertheless useful because the request-scoped P1853 NSys
capture proves that paired-B staging closed a material part of the Gate gap:

| Kernel family | Production v1 | M128N64 staged | Change |
|---|---:|---:|---:|
| Gate+Up | 888.941 ms | 802.911 ms | -86.030 ms (-9.68%) |
| Down | 523.248 ms | 541.481 ms | +18.233 ms (+3.48%) |
| Attention pair | 336.135 ms | 335.992 ms | -0.142 ms |
| Attention-O | 165.631 ms | 165.630 ms | -0.001 ms |
| Full-Attention Q/K/V | 98.187 ms | 98.134 ms | -0.053 ms |

The end-to-end reversal comes from the natural prompt-length schedule, not
from the aligned M128N64 cell.  Against the retained v1 route, prompt-prefill
time improved by 47--79 ms at 1,792, 1,853, and 1,906 tokens, but the
2,148-token request regressed from 3,235.39 to 3,528.26 ms.  That request
contains 17 M128 tiles.  With 16 fixed owner CTAs, one owner serially executes
two complete M tiles across every N64 stripe while the other 15 execute one.
The resulting long tail costs the four-request matrix roughly 73 ms on
average and masks the aligned-cell gain.

The evidence freezes two requirements for the next structural candidate:
distribute the 17th M tile without abandoning synchronized N traversal, and
retain the faster v1 Down consumer instead of paying the v2 Down regression.
Together these are an architectural composition problem, not a request to
scan tile constants.

```text
server log SHA256      3e4edc7373e3f78118a0b571a971d16d6eef296ad43f53bc1898766894351ecf
summary SHA256         68e7cb0b02984245f1f3b47417d6d787ee7cbdffc3bec6c7e67fc3792ced3931
provenance SHA256      00d620ecdb06926b0347fa3075a90ba32478813a69ce01deca83631039afa11c
NSys report SHA256     4cc50314110a5111797d5cee2e3c999681945dd0d62fc29836defad8bb8ccd0c
SQLite SHA256          bb1332942100a5fb9524c49aad6e8e03813f581034e9ab3d3273cb0158ee69bd
kernel CSV SHA256      dfee6b3c6579226091c6c52b2471e593c069fc71a533fbabadf9120631a5e8c0
profile log SHA256     6c2cd2c910b77d95479cf5adb24189a679b15a2e0e34c6f27c2efecf9ea5a801
```

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
- rejected fragment-native M128N32 run:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n32-evalscope-p2048`;
- retained fragment-native M128N64 one-CTA run:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n64-1cta-evalscope-p2048`;
- fragment-native M128N64 request trace:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n64-1cta-request2.nsys-rep`;
- rejected fragment-native Down M128N256 one-CTA run:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta-evalscope-p2048`;
- profiled fragment-native Down M128N256 EvalScope run:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta-nsys-p2048`;
- fragment-native Down M128N256 request trace:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n64-1cta-down-m128n256-1cta-request2.nsys-rep`;
- rejected fragment-native M64N128 16-warp run:
  `/home/rm01/q3x-mlp-k512-fragment-native-m64n128-1cta-evalscope-p2048-v2`;
- profiled fragment-native M64N128 EvalScope run:
  `/home/rm01/q3x-mlp-k512-fragment-native-m64n128-1cta-nsys-p2048`;
- fragment-native M64N128 request trace:
  `/home/rm01/q3x-mlp-k512-fragment-native-m64n128-1cta-request2.nsys-rep`;
- rejected fragment-native M128N64 staged run:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n64-staged-evalscope-p2048`;
- profiled fragment-native M128N64 staged EvalScope run:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n64-staged-nsys-p2048`;
- fragment-native M128N64 staged request trace:
  `/home/rm01/q3x-mlp-k512-fragment-native-m128n64-staged-request2.nsys-rep`;
- fragment-native v2 publication:
  `/home/rm01/models/dev/llm/nvidia/Qwen3.6-27B-NVFP4-q3x-mlp-k512-fragment-native/weights-mlp-k512-fragment-native.bin`.
