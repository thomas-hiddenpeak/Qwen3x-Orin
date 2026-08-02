# Alternating K256 Gate+Up real-API promotion (2026-08-02)

## Decision

The retained-paired M64N128 alternating-K256 Gate+Up edge is **retained as
the new cumulative Prefill baseline**.  It is a default-off, explicit
admission route over the authenticated K512 MLP v1 publication.  It changes
only the Gate+Up consumer schedule: the incumbent edge, old split Gate/Up
stages, split product quantizer, and Down-v2 experiment are excluded, while
the K512 input quantizer and Down v1 consumer remain unchanged.

This is a measured cumulative improvement, not the requested qualitative
Prefill breakthrough.  External EvalScope improves from 843.1066 to
850.9994 prompt token/s, only 42.55% of the 2,000 token/s floor.  At the
1,924.75-token corpus mean, that floor requires about 962.38 ms mean TTFT;
the promoted 2,261.38 ms result still needs another 1,299.00 ms reduction.
Work therefore moves directly to the LDSM-fed M128 Gate+Up replacement.  No
additional barrier or tile-parameter scan is justified on this skeleton.

## External OpenAI API and EvalScope result

All performance decisions use the pinned real checkpoint, authenticated real
sidecars, the OpenAI `/v1/completions` API, and external EvalScope 1.9.1.
The natural P2K corpus SHA-256 is
`41ab42aecfbf7157ece82df889df7a38a8f0ba2963b39958409e732a4681d4af`.
Synthetic inputs are used only for correctness.

The one-request direction gate used one P1804 warm-up followed by one P1853
measurement from the same server ELF:

| Route | Success | P1853 TTFT | Prompt throughput | Total throughput |
|---|---:|---:|---:|---:|
| K256 Attention + incumbent Gate+Up edge | 1/1 | 2,247.11 ms | 824.5983 token/s | 825.0433 token/s |
| K256 Attention + alternating Gate+Up edge | 1/1 | 2,218.77 ms | 835.1309 token/s | 835.5816 token/s |
| **Change** | - | **-28.34 ms (-1.26%)** | **+1.28%** | **+1.28%** |

The server-local P1853 Prefill time moves from 2,241.63 to 2,213.33 ms,
which independently places the same 28.30-ms saving inside Prefill.

The four-request stability gate used one P1804 warm-up followed by P1853,
P1792, P2148, and P1906 measurements:

| Route | Success | Mean TTFT | Prompt throughput | Total throughput |
|---|---:|---:|---:|---:|
| Promoted K256 Attention baseline | 4/4 | 2,282.55 ms | 843.1066 token/s | 843.5446 token/s |
| Alternating K256 Gate+Up | 4/4 | 2,261.38 ms | 850.9994 token/s | 851.4416 token/s |
| **Change** | - | **-21.17 ms (-0.93%)** | **+0.94%** | **+0.94%** |

Every natural length improves:

| Prompt tokens | Previous TTFT | Alternating TTFT | Saving | Change |
|---:|---:|---:|---:|---:|
| 1,792 | 2,087.304 ms | 2,066.110 ms | 21.194 ms | -1.02% |
| 1,853 | 2,240.520 ms | 2,219.639 ms | 20.881 ms | -0.93% |
| 1,906 | 2,250.921 ms | 2,229.977 ms | 20.944 ms | -0.93% |
| 2,148 | 2,551.468 ms | 2,529.808 ms | 21.660 ms | -0.85% |

The harness proved `gateup_alternating_launch_hits=0` for both baseline
requests, `64` for both direction-candidate requests, and `64` for the
warm-up plus all four stability requests.  A missing layer hit, extra hit,
selector conflict, or legacy-stage fallback fails the run.

## Isolated mechanism and validation

The candidate retains the incumbent paired M16N32 ownership, v1 weight ABI,
global/shared byte traffic, IMMA count, scale application, SwiGLU math, and
Down publication.  It changes the K256 stage schedule from two barriers per
half-stage to an alternating one-barrier hand-off.  Static code generation is
otherwise identical:

```text
                            incumbent    alternating
registers/thread            125          125
dynamic shared memory       148736 B     148736 B
active CTA/SM               1            1
IMMA sites                  64           64
LDGSTS sites                26           26
scalar LDS sites            181          181
STS sites                   8            8
BAR.SYNC sites              6            5
dynamic barriers/edge       165          85
local stack/spill           0 B          0 B
```

CUDA correctness is bit-exact against the incumbent packed output and K512
scales for base-wave and residual schedules, CUDA Graph capture and two
replays, immutable inputs, output guards, and a Down BF16 consumer.  An
independent review identified a parity gap in the first test revision; the
suite now covers both odd and even K512-group counts.  Focused contract,
correctness, runner, engine-control, and EvalScope harness tests pass.

## Evidence and provenance

The production integration commit is `c4876553e41427b54eb9866f2c583da4ca6a4d08`;
the even-K validation commit is `40e7d6d`.  The tested server ELF SHA-256 is
`8f98dd549f1dc1e669f32fc481d8b94a1fe4418f933169f2bb64ca0d941238d6`.

```text
direction baseline summary  a94d2bfd228dba8d5fce529c33d4f7a0f8903aa7f1b1f41efb74b220eced257d
direction baseline DB       9faba974717e5e2fd013c79b9c927f3bb38030987f8b007beef5915cd77e4a2a
direction candidate summary ec7036f511e94352cdeba6f71b18d8161c53caabed1951f249c278ad73dc2ee4
direction candidate DB      772d98faab93b16fd9965b9f5ba3d93545f92b530e1f1a8043591049152b8ccb
stability summary           9b6fa6f9a3b424f4d5e86526f51c5fa9cde8cd6622cdc9d671099e5702dce9db
stability workload          69fb176d4164f11fd14c8735a3df526f96f95e3b0469c5f3661a7327065205b6
stability DB                bae90d7f919ddbfdd9da9dc76d8e16e01088e4230a695b41b81c9936fda5617c
stability provenance        b5e6f7e0e6e372222298c04789540efc3e2eb65cdcbbfe609556381db3a35e61
```

Raw EvalScope artifacts remain outside Git under:

```text
/tmp/q3x-alternating-direction-baseline-40e7d6d
/tmp/q3x-alternating-direction-candidate-40e7d6d
/tmp/q3x-alternating-p2k4-candidate-40e7d6d
```

## Next structural replacement

The next architecture replaces the scalar shared-memory operand feed rather
than tuning the retained schedule:

1. Prove the SM87 S4 `ldmatrix.x4/x2` lane mapping against the scalar loader
   and a CPU oracle, with exact SASS and zero-spill contracts.
2. Build Gate+Up as M128N64, 256 threads, eight warps.  Reuse one paired
   Gate/Up B fragment directly from `.ca.v4.u32` across eight M16 panels and
   load A with `ldmatrix.x4`; combine the 66,560-byte A pipeline with a
   half-product plane and stable scratch-backed publication.
3. Run the real API P1853 direction gate immediately.  A qualitative Gate
   result targets at least 250 ms below the 773.3-ms retained Gate total.
4. Only after Gate is positive, apply `x4+x2` operand loading and a whole-pair
   alternating ring to the shape-distinct M128N128 Down kernel.

NCU and NSys remain attribution tools after a real-API positive direction;
they are not performance admission gates.
