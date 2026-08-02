# Prefill paired M64N128 alternating-K256 resource pass

Date: 2026-08-02

Target: Qwen3.6-27B-NVFP4 on the pinned 16-SM SM87 Orin

Status: **RESOURCE PASS; SELECTED FOR CORRECTNESS AND REAL-API ADMISSION**

## Isolated structural variable

This default-off kernel is a byte- and ownership-preserving clone of the
retained Gate+Up K512 edge.  It keeps:

- one 512-thread M64N128 CTA;
- sixteen paired M16N32 warps, each computing matching Gate and Up values;
- the authenticated v1 packed-code and K512-scale ABI;
- two complete A+GateB+UpB K256 shared stages and two scale slots;
- the M64K512 BF16 product edge, SwiGLU seam, and output quantizer.

Only the stage schedule changes.  The retained loop places a reader-release
barrier after each K256 computation and another ready barrier before the next
stage.  The candidate alternates the two whole stages so one wait/barrier is
both the next-stage publication point and the previous-stage reader release.

For model K=5120, one N128 cell uses:

```text
startup publication                         1 barrier
even-to-odd publication/release             10 barriers
non-final odd-to-next-even release           9 barriers
final product publication/stage release      1 barrier
total                                       21 barriers
```

Four N128 cells plus the final K512 quantizer barrier require 85 barriers per
M64K512 edge, versus 165 for the retained schedule.  K512 arithmetic order is
unchanged: both K256 halves, or eight physical K64 planes, accumulate into the
same S32 partial before one BF16 scale product is applied.

## Resource result

The first complete image met the preferred gate without a structural rewrite:

```text
registers/thread              125
stack/local/spill        0 / 0 / 0
dynamic shared             148,736 B
configured dynamic         148,736 B
device opt-in limit        166,912 B
active CTAs/SM                    1
maximum threads/block           512
target                          SM87
```

The static compute image is otherwise identical to the retained edge:

| Static SASS | Retained | Alternating |
|---|---:|---:|
| registers/thread | 125 | 125 |
| IMMA | 64 | 64 |
| LDGSTS | 26 | 26 |
| LDS | 181 | 181 |
| STS | 8 | 8 |
| BAR.SYNC sites | 6 | 5 |
| LDL / STL | 0 / 0 | 0 / 0 |

The candidate retains `STG.E.U8` and `STG.E.U16` sinks for packed output and
scale.  Its 32 persistent FP32 values overlap the complete 32-register S32
partial lifetime, so the resource result is not an empty or dead-code image.

## Decision

This is a resource **PASS**, not a performance claim.  Unlike the archived
split-projection skeleton, it removes no operand traffic and adds no new
exchange traffic; it isolates the only surviving mechanism, the 48.5%
reduction in edge barriers, against the exact retained data flow.

The next slice must expose a default-off launch surface, prove the packed
codes, K512 scales, and downstream BF16 result bit-for-bit against the retained
edge on real v1 payload semantics, and fail closed on runtime hit accounting.
The first performance verdict will use the real checkpoint through the OpenAI
API and external EvalScope 1.9.1: P1804 warmup plus one P1853 measurement.
Only a positive whole-request direction advances to the four-request P2K gate.
Synthetic matrices remain correctness-only and cannot admit performance.

The production baseline remains unchanged at 843.5446 token/s on the external
P2K workload.

## Reproduction hashes

```text
manual object      ac4c388228c1658b5f01eb118b87a68759acbd4b7bbe9693a46afc83d9a5e4fe
CMake object       2bf6904a0cba8c35b224b2ee7221e42faf2904d590af2982e64ba640995f6149
resource dump      9b520ffc766a583ecaaef5b700596ce5dc85a0cf1cfc9d6eb39f586cd2c4b357
SASS               b3c7a42e692e37eb75a87de0cbebabece6f012afafdda4ca7d987bf46f64f28a
header             2081dd1bf6dc875ac1aaf713af814b6a4a117619cede6e9c0e3d8582cc670303
CUDA source        1fd9ff39495ec706616314a672eb5d48fc6ec6fd888eb0ea0d33fafaaf46586a
```

Raw local artifacts:

```text
/tmp/q3x-m64n128-k256-alternating-final.o
/tmp/q3x-m64n128-k256-alternating-final.resources.txt
/tmp/q3x-m64n128-k256-alternating-final.sass
/tmp/q3x-m64n128-k256-alternating-cmake
```
