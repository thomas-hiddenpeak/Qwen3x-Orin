# Prefill Gate+Up M128N128 projection-serial rejection

Date: 2026-08-03
Scope: Qwen3.6-27B-NVFP4, AGX Orin SM87, single-request Prefill
Performance authority: real checkpoint, OpenAI `/v1/completions`, external
EvalScope 1.9.1

## Decision

The M128N128 projection-serial Gate+Up candidate is rejected at the first
real-path direction gate. It remains a default-off, fail-closed experiment
for correctness and architectural reference; it does not replace the
M64N128 K256 LDSM pair-feed production baseline and is not part of the
current-best mode.

No P2K four-request closure or synthetic performance sweep follows this
negative result. Synthetic inputs were used only for bit-exact correctness,
guards, tails, and resource validation.

## Structural hypothesis

One 512-thread CTA owns an M128N128 cell. It traverses the complete Gate
projection, places 64 KiB of FP32 Gate accumulators in a component-major
shared handoff, reuses the register file for the complete Up projection, and
writes the BF16 SiLU product for the existing exact K512 split quantizer and
independently selected Down kernel.

The candidate therefore tests a coherent skeleton change rather than a
cache-policy or stage-count adjustment:

- 16 persistent, N-major CTAs cover each M128 prompt tile;
- Gate and Up each use a three-stage K256 `cp.async.cg` pipeline;
- A uses `ldmatrix.x4`, B uses `ldmatrix.x2`;
- the primary N=12,288 and secondary N=5,120 windows share the same kernel;
- Gate, Up, product quantization, and Down retain their exact arithmetic and
  authenticated K512 publication contracts.

The route is selected only by
`Q3X_RUN_A4W4_GATEUP_K512_M128N128_PROJECTION_SERIAL_ADMISSION`. Request
telemetry proves 64 candidate launches and zero pair-feed/alternating
launches while leaving the Down selector independent.

## Correctness and resources

The focused K1536, logical-M117/nonzero-N-window test is bit exact and covers
tails and guards. The contract test, incumbent macrocell test, candidate
correctness test, reference-runner host test, and engine-control test pass
5/5; the EvalScope harness unit suite passes 67/67.

The release kernel compiles for SM87 with:

```text
threads/CTA       512
registers/thread  127
dynamic shared    165376 bytes
active CTA/SM     1
stack             0 bytes
spill loads       0
spill stores      0
```

The 1,536-byte shared-memory margin and one-CTA residency leave no credible
room to repair this skeleton by adding another pipeline stage or another
full output plane.

## External EvalScope P1853 direction result

The run used the verified natural P2K corpus, one P1804 warmup, one measured
P1853 request, concurrency one, and one generated token. Both routes used
the same production composition outside Gate+Up: native GDN, whole-span BF16
A/B, FlashInfer attention, Attention K256 A-exchange B4, MLP K512, and the
Down16 pair-ring kernel.

```text
candidate server ELF SHA256 2926bda1bef226b43dd43603c40ee25f084d21f1b5f4a2660d8155ddae3bf6c1
corpus SHA256               41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af
```

| Route | P1853 TTFT | Total throughput | Server Prefill |
|---|---:|---:|---:|
| LDSM pair-feed baseline | 1,916.24 ms | 967.4983 tok/s | 1,911.30 ms |
| M128N128 projection serial | 2,034.85 ms | 911.1064 tok/s | 2,029.84 ms |
| Candidate change | +118.61 ms (+6.19%) | -56.3919 tok/s (-5.83%) | +118.54 ms (+6.20%) |

The candidate warmup reports 2,043.94 ms server Prefill. Both candidate
requests prove `gateup_m128n128_projection_serial_launch_hits=64`,
`gateup_ldmatrix_pairfeed_launch_hits=0`, Down16=64, Attention A-exchange
B4=128, and prompt-length-derived native GDN accounting. The measured
regression is therefore on the intended production path, not a selector
miss.

```text
result root                 /tmp/q3x-projection-serial-real-p1853-candidate-root-20260803
provenance SHA256           01a6ee85e9e3883b72c06b6ab97aa380e38a6dc42d3da34968265b6a0d9b0340
server log SHA256           b99b5c2bdd66569fa2ef7f1293c37797426057134c810826e5695ff9a9a2dd13
performance summary SHA256  def8dc6d846698b5c76434a8b3c5bfb80ac8be2b87b30b240787b163f5289844
EvalScope stdout SHA256     95bc120a166b2811658618d27446af9a8a9e8221217b9a862bd83a1e0612a881
```

## Attribution boundary

A request-scoped NSys capture was attempted only after the API rejection.
The profiler-wrapped server exited before readiness while opening the
authenticated A4 receipt, so no request was issued and no GPU report was
generated. An isolated NSys smoke report succeeded, which narrows this to
the profiler/service sidecar-open path rather than general NSys
availability. Per the direction-first stop rule, the full model was not
started a third time solely to recover optional attribution.

```text
failed capture root          /home/rm01/q3x-evidence/projection-serial-20260803/run
failed provenance SHA256     e9c2ece9b49d3fe111ede226ced7221086e8da2bfa63cc5e9b2b547a7272bf32
failed server log SHA256     0f872a4153d668a4f9f566d2cb3706b23acc0063646bfcdb0cbd2199433f3908
NSys smoke report SHA256     56f3f2f74ecb184143077e771b5617c3efae7c25a20580626b493b638657c4be
```

Without a request profile, barrier cost and shared-handoff cost remain
hypotheses, not measured attribution. The source nevertheless exposes three
structural costs that the real-path result rejects in combination: A is
loaded once for Gate and again for Up, the full 64 KiB Gate plane is written
and read through shared memory, and the BF16 product is still materialized
for a separate K512 quantizer.

## Carry-forward

Do not tune cache operators, pipeline depth, or CTA phase order on this
M128N128 projection-serial skeleton. The next Gate+Up candidate must remove
at least two of the three costs above as one design:

1. load each A K-slice once per output sub-cell and consume independent Gate
   and Up B slices before releasing it;
2. widen producer ownership to a complete N512 quantization group without
   storing a full FP32 Gate plane;
3. quantize the SiLU product inside the producer lifetime so no global BF16
   product and split quantizer remain.

Its first performance verdict remains the real model through the OpenAI API
and external EvalScope. Only a positive direction result earns broader
stability, profile, and capability work.

The selected resource prototype is a 512-thread M128N512 hybrid made from
four M128N128 paired sub-cells. Its two pipeline stages require 99,840 bytes
for A, Gate-B, Up-B, and scales; two completed BF16 product sub-cells require
65,536 bytes, for 165,376 bytes total. A third product sub-cell is held as
packed BF16 registers and the fourth remains as live Gate/Up accumulators
until the CTA derives and applies the exact N512 scale. The prototype must
compile at no more than 128 registers/thread with zero stack, local memory,
or spill before it earns correctness work.

For P1920, its audited logical traffic is about 1.898 GiB per layer, 60.20%
of the 3.153-GiB pair-feed estimate, while removing the global product
scratch. This is the structural basis for implementation; it is not a
performance result. Loading A only once for the entire M128N512 macro-tile is
not feasible under the exactness and no-global-scratch constraints: the live
Gate/Up FP32 state alone would exceed both the SM87 shared-memory and
register limits. Here "A once" therefore means once for Gate+Up within each
M128N128 sub-cell.
