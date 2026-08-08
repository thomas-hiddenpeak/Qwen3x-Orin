---
q3x_document:
  id: q3x-model-support-catalog
  class: contract
  status: active
  owner: runtime-maintainers
  authority: pinned model and architecture catalog contract
  effective: 2026-08-09
  last_reviewed: 2026-08-09
  supersedes: []
  superseded_by: []
  ssot_for: catalogued model architecture facts and catalog status vocabulary
  review_trigger: any supported model, pinned revision, architecture fact, or catalog-status change
---

# Model support and architecture catalog

> **Authority boundary.** This catalog contract refines the model boundary in
> the [system SDD](SDD.md) and is subordinate to it and the
> [engineering constitution](ENGINEERING_CONSTITUTION.md). Cataloguing an
> architecture is not a delivery or production claim; current implemented,
> qualified, and default-route support belongs in
> [`CURRENT_STATUS.md`](CURRENT_STATUS.md). Local mechanism conclusions apply
> only within their named active local optimization work package.

Checked on **2026-07-18 (Asia/Shanghai)**. The catalog records architecture
facts from pinned upstream Hugging Face revisions. It does not imply that an
end-to-end inference path has already been validated.

## Status vocabulary

- **Catalogued**: a built-in descriptor and validation coverage exist.
- **Target**: the capability is in project scope but is not yet an end-to-end
  support claim.
- **Verified**: reserved for a checkpoint that passes tensor-layout, layer,
  logits, and generation tests on Jetson AGX Orin.

## Target matrix

| Model | Topology | Built-in descriptor | Pinned ModelOpt metadata | Text inference | Vision | MTP |
|---|---|---:|---:|---:|---:|---:|
| Qwen3.5-27B | Dense | Catalogued | Not yet available/pinned | Target | Deferred | Deferred |
| Qwen3.5-35B-A3B | MoE, top-8/256 | Catalogued | Not yet available/pinned | Target | Deferred | Deferred |
| Qwen3.6-27B | Dense | Catalogued | Metadata-compatible at exact NVIDIA revision | Target | Deferred | Deferred |
| Qwen3.6-35B-A3B | MoE, top-8/256 | Catalogued | Metadata-compatible at exact NVIDIA revision | Target | Deferred | Deferred |

The first execution milestone is text-only. Vision and the single MTP layer are
described so that loaders can reject incompatible tensors cleanly, but their
presence in the catalog is not an execution support claim.

## Engine-visible dimensions

Qwen3.5 and Qwen3.6 have the same engine-visible tensor shapes within each
size/topology pair.

| Field | 27B Dense (3.5 and 3.6) | 35B-A3B MoE (3.5 and 3.6) |
|---|---:|---:|
| Text hidden size | 5120 | 2048 |
| Text layers | 64 | 40 |
| Vocabulary | 248320 | 248320 |
| Maximum positions | 262144 | 262144 |
| Dense intermediate size | 17408 | n/a |
| Attention heads / KV heads | 24 / 4 | 16 / 2 |
| Attention head dimension | 256 | 256 |
| Q dimension / gated Q projection | 6144 / 12288 | 4096 / 8192 |
| KV dimension | 1024 | 512 |
| Partial rotary factor / dimension | 1/4 / 64 | 1/4 / 64 |
| RoPE theta | 10000000 | 10000000 |
| Linear key heads x dimension | 16 x 128 | 16 x 128 |
| Linear value heads x dimension | 48 x 128 | 32 x 128 |
| Linear QKV projection dimension | 10240 | 8192 |
| Layer schedule | 3 linear + 1 full | 3 linear + 1 full |
| Linear / full-attention layers | 48 / 16 | 30 / 10 |
| Routed experts / selected per token | n/a | 256 / 8 |
| Routed / shared expert intermediate | n/a | 512 / 512 |
| MTP hidden layers | 1 | 1 |
| Vision depth / hidden / intermediate | 27 / 1152 / 4304 | 27 / 1152 / 4304 |
| Vision output hidden size | 5120 | 2048 |

Useful derived state sizes exposed by `ModelConfig` are:

- 27B: 786432 recurrent-state elements per linear-attention layer and 32768
  full-attention KV elements per token across all full-attention layers.
- 35B-A3B: 524288 recurrent-state elements per linear-attention layer and
  10240 full-attention KV elements per token across all full-attention layers.

Element counts do not include batch, sequence, allocator padding, or element
byte width.

## Qwen3.5 versus Qwen3.6

The 3.6 checkpoints retain the Hugging Face identifiers used by 3.5:

| Topology | `architectures[0]` | top-level `model_type` | text `model_type` |
|---|---|---|---|
| Dense | `Qwen3_5ForConditionalGeneration` | `qwen3_5` | `qwen3_5_text` |
| MoE | `Qwen3_5MoeForConditionalGeneration` | `qwen3_5_moe` | `qwen3_5_moe_text` |

Consequently, dispatching solely on `model_type` cannot distinguish 3.5 from
3.6. `ModelConfig` carries an explicit `ModelSeries`.

The official 3.5 and 3.6 MoE configs have no tensor-shape, routing, or layer
schedule differences: both use 40 layers, 256 routed experts, top-8 routing,
512-wide routed experts, and a 512-wide shared expert. The 3.6 MoE JSON adds
explicit/default metadata (`bos_token_id`, `pad_token_id`,
`partial_rotary_factor`, `tie_word_embeddings`, and
`output_router_logits=false`) and removes the empty `mlp_only_layers` entry.
Those changes do not alter the descriptor dimensions.

The dense 3.6 JSON similarly adds explicit/default metadata and
`output_gate_type="swish"`, plus `language_model_only=false` at the top level.
The weights and release behavior are still distinct even though the tensor
shapes match; shape compatibility is not weight interchangeability.

## Official ModelOpt artifacts

The official NVIDIA Qwen3.6 NVFP4 checkpoints are mixed-precision ModelOpt
artifacts, not all-FP4 files:

| Checkpoint | ModelOpt producer | W4A16 NVFP4 entries | FP8 entries | KV cache declaration |
|---|---:|---:|---:|---:|
| `nvidia/Qwen3.6-27B-NVFP4` | 0.45.0 | 193 | 208 | FP8 |
| `nvidia/Qwen3.6-35B-A3B-NVFP4` | 0.44.0 | 161 | 130 | FP8 |

Counts are the number of entries in `quantization.quantized_layers` at the
pinned revisions below. The architecture catalog intentionally does not bake
these per-tensor precision choices into `ModelConfig`; the checkpoint loader
must parse and validate them independently.

The checked-in [normalized reports](metadata/README.md) also pin raw SHA-256
digests for `config.json`, `hf_quant_config.json`, and
`model.safetensors.index.json`, plus shard-index totals and representative
safetensors header shapes. `qwen3x-inspect checkpoint` recognizes only those
exact evidence sets; matching dimensions alone are insufficient.

## Pinned sources

Official Qwen base configuration sources:

- [Qwen3.5-27B `config.json`](https://huggingface.co/Qwen/Qwen3.5-27B/blob/fc05daec18b0a78c049392ed2e771dde82bdf654/config.json), revision `fc05daec18b0a78c049392ed2e771dde82bdf654`.
- [Qwen3.5-35B-A3B `config.json`](https://huggingface.co/Qwen/Qwen3.5-35B-A3B/blob/59d61f3ce65a6d9863b86d2e96597125219dc754/config.json), revision `59d61f3ce65a6d9863b86d2e96597125219dc754`.
- [Qwen3.6-27B `config.json`](https://huggingface.co/Qwen/Qwen3.6-27B/blob/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/config.json), revision `6a9e13bd6fc8f0983b9b99948120bc37f49c13e9`.
- [Qwen3.6-35B-A3B `config.json`](https://huggingface.co/Qwen/Qwen3.6-35B-A3B/blob/995ad96eacd98c81ed38be0c5b274b04031597b0/config.json), revision `995ad96eacd98c81ed38be0c5b274b04031597b0`.

Official NVIDIA ModelOpt configuration sources:

- [Qwen3.6-27B-NVFP4 `hf_quant_config.json`](https://huggingface.co/nvidia/Qwen3.6-27B-NVFP4/blob/0893e1606ff3d5f97a441f405d5fc541a6bdf404/hf_quant_config.json), revision `0893e1606ff3d5f97a441f405d5fc541a6bdf404`.
- [Qwen3.6-35B-A3B-NVFP4 `hf_quant_config.json`](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4/blob/491c2f1ea524c639598bf8fa787a93fed5a6fbce/hf_quant_config.json), revision `491c2f1ea524c639598bf8fa787a93fed5a6fbce`.

All six files were downloaded from Hugging Face and checked against the stated
revision on 2026-07-18.
