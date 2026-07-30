# Stage 0 implementation: sidecar and prompt-arena contract

This checkpoint implements only the host-side ownership and byte-layout ABI.
It adds no projection kernel, environment switch, or production dispatch.

## Versioned projection manifest

`PrefillSidecarManifest` version 1.0 inventories every one of the pinned
Qwen3.6-27B model's 400 Prefill projections. Each entry records:

- layer and projection family;
- logical `[N, K]` shape;
- weight quantization, scale group, and consumer layout;
- SHA-256 over all authenticated source components, including pinned shard
  digests, ranges, dtypes, and shapes;
- weight, scale, metadata, total-byte, and deterministic 256-byte-aligned
  offset fields.

The manifest body has its own SHA-256. Revalidation reconstructs the complete
inventory and byte arithmetic before residency is allowed.

| Sidecar kind | Payload bytes | 256-byte arena bytes |
|---|---:|---:|
| Exact NVFP4/FP8 | 16,840,132,160 | 16,840,232,960 |
| A8-safe | 24,707,072,000 | 24,707,072,000 |
| A8-compact | 16,952,853,248 | 16,952,901,632 |
| A4 K64 | 12,923,699,200 | 12,923,699,200 |
| A4 K128 | 12,543,590,400 | 12,543,590,400 |

Exact, A8, and A4 are mutually exclusive residency classes. The residency
preflight rejects zero or multiple selected classes, a manifest placed in the
wrong class, a mismatched activation arena, combined-byte overflow, and a peak
above the configured device budget.

## Prompt-slab arena

The prompt planner owns two full `[P, 5120]` BF16 hidden slabs plus one reusable
quantized hidden staging slab, one paired Gate+Up Down-input staging slab,
their BF16 group scales, and one FP32 row sum-of-squares vector. The `[S, 6144]`
O-projection input and scales explicitly alias prefixes of the larger
`[S, 17408]` Down-input regions because those phases do not overlap. The plan
never allocates two `[P, 17408]` BF16 Gate/Up outputs. Staging can cover the
whole prompt or a smaller layer-local microspan; both BF16 hidden slabs always
cover the full prompt.

| Plan | Arena bytes |
|---|---:|
| P512 A8 K128 | 22,202,368 |
| P4K A4 K64 | 132,923,392 |
| P40K A8 K128 | 1,734,560,000 |
| P40K A4 K64 | 1,298,080,000 |
| P40K A4 K128 | 1,284,000,000 |

All multiplication, addition, and alignment is checked before a region is
published. P40K admission fails closed if the byte cap is even one byte below
the planned arena.

The host contract test is `prefill_quantized_contract`; the existing
`weight_manifest`, `resident_load_plan`, and `request_state_plan` tests remain
unchanged and pass alongside it.
