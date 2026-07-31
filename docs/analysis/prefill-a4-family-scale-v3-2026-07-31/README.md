# A4 family-scale v3 contract (design only)

Status: **intermediate design; no runtime implementation or production
admission**

Design base: `1f7427cb0feb472f0868fdbfd55f82f9b4801d58`

This design removes the current requirement that all 400 Prefill projections
use one A4 scale group. It keeps the physical signed-W4 code plane in the
existing N64/K64 consumer order, but makes scale ownership a projection
property:

- all attention projections use K64 or K128;
- Gate and Up use K256 in the first candidate;
- Down uses K256 first and may independently advance to K512;
- Gate and Up share one input contract, QKV projection groups share their
  respective input contracts, and the Gate+Up epilogue emits the scale group
  declared by Down.

The design borrows one dataflow idea from the in-tree frozen vLLM Marlin
reference: packed weight traversal and scale-group traversal are separate.
Marlin represents scales as `[K/group, N]` and fetches them only at group
boundaries. No Marlin source or general-model compatibility is copied into the
proposed production route.

cuBLASLt remains an external benchmark reference only. MTP is not part of this
plan. Decode continues to consume the authenticated checkpoint representation
and is not changed by the Prefill sidecar.

## Audit of the uniform-consumer assumption

The current v1/v2 implementation stores a scale group on every projection
entry and view, but the surrounding contract forces those values to be
uniform.

| Boundary | Current assumption | Consequence for v3 |
|---|---|---|
| Manifest | `PrefillSidecarKind` selects K64 or K128 for every projection; `expected_sidecar_encoding` derives one layout/group and one fixed model payload total from that kind. | A v3 manifest needs a family profile in its header and must copy the resolved group into every projection entry. The entry remains the launch authority. |
| Policy | `A4SidecarFormat::scale_group_size` is global; every calibration record must equal it. | Validate each calibration against its manifest entry instead. Keep the existing shared-activation-boundary equality check. |
| Converter | One global output scale stride and one global K64/K128 quantizer are selected for the complete 400-projection file. | Dispatch a bounded `G in {64,128,256,512}` quantizer per entry. Packed-code offsets remain K64-based; only scale offsets/strides use `G`. |
| Receipt/authentication | v1 implies K64; v2 explicitly authenticates one `scale_group_size=128`. The engine rebuilds the manifest from `sidecar_kind` alone. | A strict v3 receipt must authenticate the family profile, then the engine rebuilds the same deterministic manifest and verifies its digest before residency. |
| `ModelWeights` | Each linear view already carries packed group and scale group, but attachment derives an expected group/layout globally and writes the same sidecar kind to all 400 bindings. | Keep the per-projection fields, add v3 attachment validation, and publish all bindings transactionally only when every entry matches the family profile. |
| Runner inventory | `A4W4PrefillConsumer` is one immutable K64 or K128 enum; inventory validation rejects mixed groups. Quantize, GEMM, capacity, and paired-output dispatch all use it. | Replace it with a complete inventory/profile plus a per-boundary group. A launcher may receive a scale group only after the input activation and weight entry are proven equal. |
| Request workspace | Hidden and intermediate scale buffers are sized at the K64 maximum. | No allocation growth is needed. Retain K64 capacity for v3 bring-up; shrinking it is an independent later change. |
| Prompt alignment | The current all-K128 route fails closed for a prompt not divisible by 64 because its admitted kernel family has no tail consumer. | v3 must not silently use a K64 tail with K256/K512 scales. Either provide group-correct arbitrary-M tails or fail the entire v3 route before layer 0. |

The useful existing primitives are therefore the transactional 400-entry
attachment, manifest/source digests, per-projection clip ratios, activation
boundary keys, and per-view scale-group field. The global `A4W4PrefillConsumer`
and global receipt scale are the assumptions to retire.

## Minimal v3 disk and in-memory ABI

Add one new sidecar kind, provisionally `kA4FamilyV3`, without changing the
serialized bytes or parsers for K64-v1 and K128-v2.

The minimal public profile has only three parameters:

```text
packed_k_group_size = 64                         # fixed
attention_scale_group_size in {64, 128}
gate_up_scale_group_size = 256                   # fixed in v3.0
down_scale_group_size in {256, 512}
```

The manifest header stores this profile. Its deterministic projection
inventory expands it as follows:

| Projection family | Entry `scale_group_size` |
|---|---:|
| MLP Gate, MLP Up | `gate_up_scale_group_size` |
| MLP Down | `down_scale_group_size` |
| Linear QKV/Z/O, full Q/K/V/O | `attention_scale_group_size` |

This intentionally does not force Q/K and MLP to use the same quantization
granularity. A future minor contract may split attention input and O groups,
but v3.0 should not add that state before one of the four bounded profiles has
shown whole-run value.

Each entry retains the physical code layout

```text
weights [N/64][K/64][64][32] bytes
scales  [N/64][K/G][64] BF16
```

where `G` is the entry's declared scale group. Changing `G` changes the
quantized code values because one scale covers a larger source range; it does
not change the physical K64 code-plane addressing.

The v3 receipt must serialize the three profile values and
`packed_k_group_size=64` as exact required keys. It must not serialize a
misleading global `scale_group_size`. The receipt continues to bind the
manifest, policy, payload digest, payload byte count, source checkpoint, and
400-entry count. An optional family-profile field can coexist in the C++
receipt structure with the legacy scalar, but exactly one representation is
valid for a given receipt version.

The manifest digest must include the profile header as well as every resolved
entry group. The engine reconstructs the manifest from the authenticated v3
profile; it does not trust the receipt's payload size as a substitute for
entry arithmetic.

### Activation contracts

For the first v3 kernel family, activation and weight groups must match within
each GEMM. Existing activation-boundary validation remains mandatory:

- Gate and Up have identical input group, clip, rounding, and equalization;
- linear QKV and Z have identical shared-input policy;
- full Q, K, and V have identical shared-input policy;
- O is a separate activation boundary;
- the paired Gate+Up input uses the Gate/Up group, while its packed epilogue
  uses the Down projection's activation group and clip ratio.

That last rule is why the paired launcher must take distinct input and output
group arguments. Reusing one global consumer enum would encode the wrong
contract.

## Kernel dataflow implied by the contract

The code pipeline stays physically K64. For a logical group `G`, retain the
same S32 fragment across `G/64` ordered `m16n8k64` instructions, then convert
and apply one A-scale/B-scale product:

```text
G=64   : 1 K64 MMA -> scale -> FP32 accumulation
G=128  : 2 K64 MMAs -> scale -> FP32 accumulation
G=256  : 4 K64 MMAs -> scale -> FP32 accumulation
G=512  : 8 K64 MMAs -> scale -> FP32 accumulation
```

With signed codes in `[-7,7]`, the worst absolute S32 dot partial for K512 is
`512 * 7 * 7 = 25,088`, far below S32 overflow. The pipeline does not need to
stage an entire K256/K512 code group at once: K64/K128 code stages can continue
to rotate while the integer fragment remains live, and the scale stream only
advances on a logical group boundary. This keeps scale cadence independent of
the `cp.async` code cadence, which is the Marlin idea worth retaining.

Group enlargement does **not** reduce packed-code traffic or S4 MMA count. It
reduces scale loads, BF16 scale decodes, A/B scale products, S32-to-FP32
conversions, and FP32 group accumulations. It may also reduce shared scale
storage, but it can increase integer-fragment live ranges and restrict software
scheduling. NCU must determine whether scale work is large enough for this
trade to matter.

## P2048 arithmetic and payload envelope

The pinned 400 projections contain 24,326,963,200 logical weights, so every
profile has the same 12,163,481,600 packed-code bytes. The table below includes
all BF16 weight scales and excludes no projection. `Scale applications` is the
architecture-level count

```text
P * sum_over_projections(N * K / G), P=2048
```

It counts output/group rescale-and-accumulate opportunities, not emitted SASS
instructions; vectorization changes instruction count but not this ratio.

| Profile | Payload bytes | Weight-scale elements | P2048 scale applications | Versus all-K64 | Dynamic activation-scale elements |
|---|---:|---:|---:|---:|---:|
| all K64 | 12,923,699,200 | 380,108,800 | 778,462,822,400 | 100.00% | 69,206,016 |
| all K128 | 12,543,590,400 | 190,054,400 | 389,231,411,200 | 50.00% | 34,603,008 |
| attention K64 / Gate+Up K256 / Down K256 | 12,522,618,880 | 179,568,640 | 367,756,574,720 | 47.24% | 34,603,008 |
| attention K64 / Gate+Up K256 / Down K512 | 12,500,336,640 | 168,427,520 | 344,939,560,960 | 44.31% | 30,146,560 |
| attention K128 / Gate+Up K256 / Down K256 | 12,409,896,960 | 123,207,680 | 252,329,328,640 | 32.41% | 23,068,672 |
| attention K128 / Gate+Up K256 / Down K512 | 12,387,614,720 | 112,066,560 | 229,512,314,880 | 29.48% | 18,612,224 |

Against the current all-K128 contract, the first recommended profile
(attention K128, Gate+Up K256, Down K256) removes 136,902,082,560, or 35.18%,
of these scale-application opportunities and 133,693,440 resident scale bytes.
Advancing Down to K512 raises the reductions to 159,719,096,320 (41.03%) and
156,475,680 bytes. The attention-K64 fallbacks save only 5.52% and 11.38%
respectively versus all-K128; they are accuracy fallbacks, not convincing
performance architectures by themselves.

These ratios are not whole-GEMM speedup claims. If scale/conversion work is a
fraction `f` of current kernel time and the opportunity ratio is `r`, the
best Amdahl estimate is `1 / (1 - f + f*r)`. With the recommended K256 profile
(`r=0.6482` versus K128), even `f=0.5` implies only about 1.21x. Packed-code
traffic and MMA remain unchanged, so the real API direction test must decide
whether the contract is worth completing.

The current P2048 request A4 workspace is already a K64 upper bound:

| Region | Bytes |
|---|---:|
| hidden packed `[2048,5120/2]` | 5,242,880 |
| hidden K64-scale capacity | 327,680 |
| intermediate packed `[2048,17408/2]` | 17,825,792 |
| intermediate K64-scale capacity | 1,114,112 |
| **Total** | **24,510,464 (23.375 MiB)** |

v3 needs no additional request workspace. Keeping these upper bounds in the
first implementation avoids coupling an ABI change to a low-value allocation
shrink.

## Accuracy and capability risk

Larger groups expose more values to one maximum/clip threshold. Outliers lower
the effective resolution of the rest of the group. This is especially risky
because the MLP source is already nonlinear NVFP4 and the sidecar is a second,
linear W4 quantization. K512 Down also spans 32 original ModelOpt K16 scale
blocks per output row. Attention Q/K remain the highest semantic-risk
projections and therefore retain an independent K64/K128 choice.

No naive nearest-even K256/K512 artifact is production-qualified merely
because its receipt authenticates. Per-layer/per-family real-prompt
calibration remains required. The existing per-projection weight and
activation clip ratios are enough for the minimal v3 disk schema; adding
equalization, GPTQ, or AWQ-aware rounding is a later calibration change and
must not be smuggled into the layout commit.

The gates remain:

- exact bounded CPU/CUDA oracle for each admitted `G`, all signed codes, scale
  edge cases, guard/alias/error behavior, and group-correct arbitrary-M tails;
- compare against the authenticated all-K128 incumbent at explicit family
  capture points on all 64 natural-order layers: decoded Gate+Up/Down input,
  Down BF16 output, and post-MLP hidden. Report NRMSE/cosine per family and per
  layer; no aggregate may hide one bad family or layer;
- because attention remains K128 in the first profile, its projection outputs
  must remain bit-identical through the selector refactor. Any attention delta
  is a routing bug, not an allowed K256 numerical change;
- zero non-finite layer-boundary values;
- for each affected MLP family and the post-layer hidden boundary, aggregate
  NRMSE is at most 0.01 and cosine is at least 0.9999 against all-K128;
- every affected family/layer boundary is at most 0.03 NRMSE and at least
  0.999 cosine against all-K128;
- greedy output agreement is a sentinel, not the capability verdict;
- no frozen public EvalScope capability score may lose more than one
  percentage point absolute or more than its baseline confidence interval,
  whichever is tighter;
- Decode rate, state continuity, memory reserve, and exact Decode path remain
  non-regression gates.

## Executable commit sequence

Each numbered item is independently reviewable. Commits 1--5 do not select a
new runtime route.

1. **v3 manifest/profile contract, host only.** Add `kA4FamilyV3`, the bounded
   three-field profile, per-family expansion, payload arithmetic, digest
   coverage, and hostile validation tests. Assert that existing K64-v1 and
   K128-v2 manifest bytes/digests are unchanged.
2. **v3 policy contract, host only.** Add the version-3 exact-key policy parser
   and template serializer. Validate every calibration group against its
   manifest entry and retain the shared-activation-boundary equality checks.
   Preserve K64-v1/K128-v2 policy bytes and parsing behavior.
3. **v3 receipt/profile authentication, host only.** Add the version-3
   exact-key receipt parser/serializer and bounded profile authentication.
   Rebuild the deterministic manifest from the receipt profile and bind its
   payload size and digests using temporary host files only. No CUDA,
   `ModelWeights`, runner selector, or feature flag changes belong here.
4. **generic offline group quantizer.** Generalize the CPU converter to
   `G={64,128,256,512}` while preserving K64 packed offsets. Add exhaustive
   small-matrix correctness and no-replace/authentication tests. The CLI emits
   one of four named profiles; it does not accept arbitrary unsupported group
   combinations.
5. **transactional model binding.** Admit v3 views in `ModelWeights`, validate
   all 400 family/group/shape bindings, and keep runtime scheduling disabled.
   Existing exact, K64, and K128 attachment tests remain unchanged.
6. **runner selector refactor with no numerical change.** Replace the global
   consumer enum with a complete inventory/profile plus per-boundary groups.
   Route current all-K64 and all-K128 publications through the new selectors
   and require bit-identical projection/layer outputs before adding K256.
7. **first vertical kernel slice.** Implement attention-K128 reuse plus K256
   Gate/Up input, K256 Gate+Up packed epilogue, and K256 Down. Keep physical
   K64 code stages; apply scales after four MMAs. Add group-correct quantizer,
   paired, Down, and arbitrary-M tail or fail-before-layer-0 contracts.
8. **earliest real observation.** Generate an authenticated real-weight
   attention-K128/Gate+Up-K256/Down-K256 artifact. Run one P2048,
   `max_tokens=1`, OpenAI-compatible request against the live all-K128 native
   incumbent. This is the first performance decision; synthetic matrices have
   no authority.
9. **qualify only a positive route.** If step 8 is positive, capture all-K128
   and K256 family/layer boundaries from the same real prompts, apply the
   family-level NRMSE/cosine gates above, run the frozen P2K/P4K EvalScope
   performance matrix and final public capability set, then use NSys/NCU to
   explain the positive whole route. Retain only a stable improvement with all
   numerical and capability gates passing.
10. **Down K512 as a separate candidate.** Change only the Gate+Up output/Down
   boundary from K256 to K512 after K256 passes capability. Repeat the real
   P2048 direction test before building its formal harness.
11. **production promotion.** Require at least 2,000 prompt token/s at both P2K
    and P4K, confirm 8K/16K/40K scaling and arbitrary lengths, pass public
    capability and Decode non-regression, and keep the exact fallback until
    those gates are complete.

## Stop-loss rules

- If the first K256 whole-run request is non-positive, stop before formal
  timing/profile infrastructure. A bounded NCU run may explain the failure but
  cannot retain it.
- The 35.18% P2048 scale-application reduction is a mechanism estimate only.
  It cannot retain, qualify, or promote a candidate without the family/layer
  all-K128 comparisons and final public capability result.
- If the formal K256 route does not clear the measured noise allowance and at
  least 1.03x whole-Prefill at P2048, do not build K512; scale cadence is not a
  sufficiently large limiter.
- If attention-K128/Gate+Up-K256/Down-K256 fails capability after explicit
  calibration, try the attention-K64 profile once. If that profile also fails,
  abandon v3 group enlargement rather than accumulating exceptions.
- If attention-K64 recovers capability but yields less than 1.03x whole-Prefill
  versus all-K128, retain the contract work but stop the kernel branch: its
  theoretical scale reduction is too small to justify production complexity.
- K512 is rejected on any capability failure or any non-positive real-path
  direction, even if a microbenchmark improves.
- A missing group-specific tail, mixed boundary, incomplete 400-entry
  inventory, or receipt/profile mismatch fails before layer 0. There is no
  silent K64 scale fallback.

This contract is a bounded mechanism test, not the complete 2K-token/s answer.
Its value is that it can remove 35--41% of the current all-K128 scale cadence
without sacrificing Q/K granularity by construction. The real P2048 route
decides whether that reduction is material enough to keep.
