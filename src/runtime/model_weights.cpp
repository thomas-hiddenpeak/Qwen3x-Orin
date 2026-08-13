#include "q3x/runtime/model_weights.h"

#include "q3x/io/safetensors.h"
#include "q3x/runtime/sm87_target_aot_projection_device_assets.h"
#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
#include "sm87_target_aot_projection_execution_access_internal.h"
#endif
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
#include "sm87_target_aot_projection_complete_execution_access_internal.h"
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

namespace st = q3x::io::safetensors;

[[nodiscard]] std::string shape_string(
    const std::initializer_list<std::uint64_t> shape) {
  std::ostringstream output;
  output << '[';
  bool first = true;
  for (const std::uint64_t dimension : shape) {
    if (!first) {
      output << ',';
    }
    first = false;
    output << dimension;
  }
  output << ']';
  return output.str();
}

[[nodiscard]] std::string shape_string(
    const std::vector<std::uint64_t>& shape) {
  std::ostringstream output;
  output << '[';
  for (std::size_t index = 0U; index < shape.size(); ++index) {
    if (index != 0U) {
      output << ',';
    }
    output << shape[index];
  }
  output << ']';
  return output.str();
}

[[nodiscard]] const DeviceTensorView* resident_lookup(
    const void* const context, const std::string_view name) noexcept {
  if (context == nullptr) {
    return nullptr;
  }
  return static_cast<const ResidentWeights*>(context)->find(name);
}

[[nodiscard]] int read_cuda_scalar(const float* const device_value,
                                   float* const host_value) noexcept {
  // CUDA last-error is host-thread state. A binding failure must describe its
  // own synchronous D2H copy rather than an unrelated earlier launch.
  (void)cudaGetLastError();
  return static_cast<int>(cudaMemcpy(host_value, device_value, sizeof(float),
                                     cudaMemcpyDeviceToHost));
}

template <typename Weight>
[[nodiscard]] std::size_t output_size_of(const Weight& weight) noexcept {
  return weight.output_size;
}

template <typename Weight>
[[nodiscard]] std::size_t input_size_of(const Weight& weight) noexcept {
  return weight.input_size;
}

[[nodiscard]] bool has_valid_fp8_payload(
    const Fp8LinearWeight* const weight) noexcept {
  return weight != nullptr && weight->weight != nullptr &&
         weight->weight_scale_device != nullptr &&
         weight->input_scale_device != nullptr &&
         std::isfinite(weight->weight_scale) && weight->weight_scale >= 0.0F &&
         std::isfinite(weight->input_scale) && weight->input_scale >= 0.0F;
}

[[nodiscard]] bool has_valid_nvfp4_payload(
    const NvFp4LinearWeight* const weight) noexcept {
  return weight != nullptr && weight->packed_weight != nullptr &&
         weight->block_scale != nullptr &&
         weight->weight_scale_2_device != nullptr &&
         weight->input_scale_device != nullptr &&
         std::isfinite(weight->weight_scale_2) &&
         weight->weight_scale_2 >= 0.0F &&
         std::isfinite(weight->input_scale) && weight->input_scale >= 0.0F;
}

[[nodiscard]] bool empty_p40_packed_artifact_view(
    const kernels::Sm87P40PackedProjectionDeviceView& view) noexcept {
  return view.payload == nullptr && view.payload_bytes == 0U &&
         view.artifact_identity == 0U &&
         view.role == kernels::Sm87P40PackedProjectionRole::kInvalid &&
         view.tactic == kernels::Sm87P40PackedTactic::kInvalid &&
         view.source_count == 0U &&
         std::all_of(view.scalar_scales.begin(), view.scalar_scales.end(),
                     [](const float scale) { return scale == 0.0F; });
}

[[nodiscard]] bool empty_nvfp4_marlin_p40_parity_view(
    const NvFp4MarlinP40ParityDeviceView& view) noexcept {
  const auto digest_empty = [](const NvFp4MarlinP40ParityDigest& digest) {
    return std::all_of(digest.begin(), digest.end(),
                       [](const std::uint8_t byte) { return byte == 0U; });
  };
  const auto source_empty = [&digest_empty](
                                const NvFp4MarlinP40ParitySourceManifest&
                                    source) {
    return source.role == NvFp4MarlinP40ParitySourceRole::kInvalid &&
           source.tensor_identity == 0U &&
           digest_empty(source.weight_digest) &&
           digest_empty(source.scale_digest) &&
           source.global_scale_bits == 0U;
  };
  const NvFp4MarlinP40ParityArtifactManifest& manifest = view.manifest;
  return view.weight == nullptr && view.scales == nullptr &&
         view.global_scale == nullptr &&
         manifest.version == kNvFp4MarlinP40ParityManifestVersion &&
         manifest.layer_index == 0U &&
         manifest.role == NvFp4MarlinP40ParityRole::kInvalid &&
         manifest.layout == NvFp4MarlinP40ParityLayout::kInvalid &&
         manifest.output_features == 0U && manifest.input_features == 0U &&
         manifest.weight_bytes == 0U && manifest.scale_bytes == 0U &&
         manifest.artifact_identity == 0U &&
         digest_empty(manifest.transformation_digest) &&
         manifest.source_count == 0U &&
         std::all_of(manifest.sources.begin(), manifest.sources.end(),
                     source_empty);
}

[[nodiscard]] LinearWeight* attention_output_projection(
    DecoderLayerWeights& layer) noexcept {
  if (auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer.attention)) {
    return &linear->out_proj;
  }
  if (auto* const full =
          std::get_if<FullAttentionWeights>(&layer.attention)) {
    return &full->o_proj;
  }
  return nullptr;
}

[[nodiscard]] NvFp4LinearWeight* nvfp4_down_projection(
    DecoderLayerWeights& layer) noexcept {
  return std::get_if<NvFp4LinearWeight>(&layer.mlp.down_proj);
}

[[nodiscard]] NvFp4LinearWeight* nvfp4_gate_projection(
    DecoderLayerWeights& layer) noexcept {
  return std::get_if<NvFp4LinearWeight>(&layer.mlp.gate_proj);
}

[[nodiscard]] NvFp4LinearWeight* nvfp4_up_projection(
    DecoderLayerWeights& layer) noexcept {
  return std::get_if<NvFp4LinearWeight>(&layer.mlp.up_proj);
}

}  // namespace

ModelWeights::ModelWeights(ModelWeights&& other) noexcept
    : embed_tokens_(std::move(other.embed_tokens_)),
      final_norm_(std::move(other.final_norm_)),
      lm_head_(std::move(other.lm_head_)),
      layers_(std::move(other.layers_)),
      stats_(std::move(other.stats_)),
      target_aot_projection_attachment_(
          other.target_aot_projection_attachment_),
      target_aot_complete_projection_attachment_(
          other.target_aot_complete_projection_attachment_) {
  auto* const owner = const_cast<Sm87TargetAotProjectionDeviceAssets*>(
      target_aot_projection_attachment_.owner);
  if (owner != nullptr) {
    if (owner->attached_model_weights_ == &other &&
        owner->prepared_model_weights_ == &other) {
      owner->attached_model_weights_ = this;
      owner->prepared_model_weights_ = this;
    } else {
      if (owner->attached_model_weights_ == &other) {
        owner->attached_model_weights_ = nullptr;
      }
      if (owner->prepared_model_weights_ == &other) {
        owner->prepared_model_weights_ = nullptr;
      }
      target_aot_projection_attachment_ = {};
    }
  }
  other.target_aot_projection_attachment_ = {};
  auto* const complete_owner =
      const_cast<Sm87TargetAotCompleteProjectionDeviceAssets*>(
          target_aot_complete_projection_attachment_.owner);
  if (complete_owner != nullptr) {
    if (complete_owner->prepared_model_weights_ == &other &&
        complete_owner->execution_bound_) {
      complete_owner->prepared_model_weights_ = this;
    } else {
      target_aot_complete_projection_attachment_ = {};
    }
  }
  other.target_aot_complete_projection_attachment_ = {};
}

ModelWeights& ModelWeights::operator=(ModelWeights&& other) noexcept {
  if (this != &other) {
    detach_sm87_target_aot_complete_projection_assets();
    detach_sm87_target_aot_projection_assets();
    embed_tokens_ = std::move(other.embed_tokens_);
    final_norm_ = std::move(other.final_norm_);
    lm_head_ = std::move(other.lm_head_);
    layers_ = std::move(other.layers_);
    stats_ = std::move(other.stats_);
    target_aot_projection_attachment_ =
        other.target_aot_projection_attachment_;
    auto* const owner = const_cast<Sm87TargetAotProjectionDeviceAssets*>(
        target_aot_projection_attachment_.owner);
    if (owner != nullptr) {
      if (owner->attached_model_weights_ == &other &&
          owner->prepared_model_weights_ == &other) {
        owner->attached_model_weights_ = this;
        owner->prepared_model_weights_ = this;
      } else {
        if (owner->attached_model_weights_ == &other) {
          owner->attached_model_weights_ = nullptr;
        }
        if (owner->prepared_model_weights_ == &other) {
          owner->prepared_model_weights_ = nullptr;
        }
        target_aot_projection_attachment_ = {};
      }
    }
    other.target_aot_projection_attachment_ = {};
    target_aot_complete_projection_attachment_ =
        other.target_aot_complete_projection_attachment_;
    auto* const complete_owner =
        const_cast<Sm87TargetAotCompleteProjectionDeviceAssets*>(
            target_aot_complete_projection_attachment_.owner);
    if (complete_owner != nullptr) {
      if (complete_owner->prepared_model_weights_ == &other &&
          complete_owner->execution_bound_) {
        complete_owner->prepared_model_weights_ = this;
      } else {
        target_aot_complete_projection_attachment_ = {};
      }
    }
    other.target_aot_complete_projection_attachment_ = {};
  }
  return *this;
}

ModelWeights::~ModelWeights() {
  detach_sm87_target_aot_complete_projection_assets();
  detach_sm87_target_aot_projection_assets();
}

void ModelWeights::detach_sm87_target_aot_projection_assets() noexcept {
  auto* const owner = const_cast<Sm87TargetAotProjectionDeviceAssets*>(
      target_aot_projection_attachment_.owner);
  if (owner != nullptr && owner->attached_model_weights_ == this) {
    owner->attached_model_weights_ = nullptr;
  }
  if (owner != nullptr && owner->prepared_model_weights_ == this) {
    owner->prepared_model_weights_ = nullptr;
  }
  target_aot_projection_attachment_ = {};
}

void ModelWeights::detach_sm87_target_aot_complete_projection_assets()
    noexcept {
  auto* const owner =
      const_cast<Sm87TargetAotCompleteProjectionDeviceAssets*>(
          target_aot_complete_projection_attachment_.owner);
  if (owner != nullptr && owner->prepared_model_weights_ == this) {
    owner->execution_bound_ = false;
  }
  target_aot_complete_projection_attachment_ = {};
}

bool ModelWeights::attach_sm87_target_aot_complete_projection_assets(
    Sm87TargetAotCompleteProjectionDeviceAssets& owner) noexcept {
  if (target_aot_projection_attachment_.owner != nullptr ||
      target_aot_complete_projection_attachment_.owner != nullptr ||
      owner.empty() || owner.prepared_model_weights_ != this ||
      owner.prepared_resident_ == nullptr || owner.execution_bound_ ||
      owner.arena_ == nullptr ||
      owner.bytes_ != kSm87TargetAotCompleteProjectionDeviceArenaBytes ||
      owner.owner_identity_ == 0U || owner.allocation_identity_ == 0U ||
      owner.device_ordinal_ < 0 ||
      owner.descriptor_count_ !=
          kSm87TargetAotCompleteProjectionDeviceArtifactCount) {
    return false;
  }
  const auto arena_begin = reinterpret_cast<std::uintptr_t>(owner.arena_);
  if (arena_begin == 0U ||
      owner.bytes_ > std::numeric_limits<std::uintptr_t>::max() -
                         arena_begin) {
    return false;
  }

  std::uint64_t expected_offset = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    DecoderLayerWeights& layer = layers_[layer_index];
    const auto* const gate =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.gate_proj);
    const auto* const up =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.up_proj);
    const auto* const down =
        std::get_if<NvFp4LinearWeight>(&layer.mlp.down_proj);
    if (!has_valid_nvfp4_payload(gate) || !has_valid_nvfp4_payload(up) ||
        !has_valid_nvfp4_payload(down) ||
        !empty_p40_packed_artifact_view(gate->prefill_p40_packed_artifact) ||
        !empty_p40_packed_artifact_view(up->prefill_p40_packed_artifact) ||
        !empty_p40_packed_artifact_view(down->prefill_p40_packed_artifact) ||
        gate->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        up->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        down->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        gate->prefill_marlin_weight != nullptr ||
        up->prefill_marlin_weight != nullptr ||
        down->prefill_marlin_weight != nullptr ||
        gate->prefill_marlin_scales != nullptr ||
        up->prefill_marlin_scales != nullptr ||
        down->prefill_marlin_scales != nullptr ||
        gate->prefill_marlin_global_scale != nullptr ||
        up->prefill_marlin_global_scale != nullptr ||
        down->prefill_marlin_global_scale != nullptr ||
        gate->down_scale6_sidecar != nullptr ||
        up->down_scale6_sidecar != nullptr ||
        down->down_scale6_sidecar != nullptr ||
        gate->down_scale6_base != 0U || up->down_scale6_base != 0U ||
        down->down_scale6_base != 0U ||
        gate->down_consumer_order_weight != nullptr ||
        up->down_consumer_order_weight != nullptr ||
        down->down_consumer_order_weight != nullptr ||
        gate->decode_gate_up_coupled_feed_sidecar != nullptr ||
        up->decode_gate_up_coupled_feed_sidecar != nullptr ||
        down->decode_gate_up_coupled_feed_sidecar != nullptr ||
        !empty_nvfp4_marlin_p40_parity_view(
            gate->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            up->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            down->prefill_p40_vllm_marlin_parity)) {
      return false;
    }

    std::array<const Fp8LinearWeight*, 4U> fp8{};
    std::size_t fp8_count = 0U;
    if (sm87_target_aot_complete_is_full_layer(layer_index)) {
      const auto* const full =
          std::get_if<FullAttentionWeights>(&layer.attention);
      if (full == nullptr) {
        return false;
      }
      fp8[fp8_count++] = std::get_if<Fp8LinearWeight>(&full->q_proj);
      fp8[fp8_count++] = std::get_if<Fp8LinearWeight>(&full->k_proj);
      fp8[fp8_count++] = std::get_if<Fp8LinearWeight>(&full->v_proj);
      fp8[fp8_count++] = std::get_if<Fp8LinearWeight>(&full->o_proj);
    } else {
      const auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer.attention);
      if (linear == nullptr) {
        return false;
      }
      fp8[fp8_count++] =
          std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
      fp8[fp8_count++] = std::get_if<Fp8LinearWeight>(&linear->in_proj_z);
      fp8[fp8_count++] = std::get_if<Fp8LinearWeight>(&linear->out_proj);
    }
    for (std::size_t index = 0U; index < fp8_count; ++index) {
      const Fp8LinearWeight* const projection = fp8[index];
      if (!has_valid_fp8_payload(projection) ||
          projection->m1_aosoa4_preswizzled_weight != nullptr ||
          projection->prefill_qkv_register_feed_sidecar != nullptr ||
          projection->prefill_supermatrix_sidecar != nullptr ||
          projection->prefill_marlin_weight != nullptr ||
          projection->prefill_marlin_scales != nullptr ||
          !empty_p40_packed_artifact_view(
              projection->prefill_p40_packed_artifact)) {
        return false;
      }
    }

    const std::array<kernels::Sm87TargetAotProjectionRole, 4U> roles{{
        kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp,
        kernels::Sm87TargetAotProjectionRole::kNvFp4Down,
        sm87_target_aot_complete_is_full_layer(layer_index)
            ? kernels::Sm87TargetAotProjectionRole::kFp8FullQkv
            : kernels::Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
        kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput}};
    for (const auto role : roles) {
      const auto* const descriptor = owner.find(layer_index, role);
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(role);
      if (descriptor == nullptr || !layout.valid() ||
          !sm87_target_aot_complete_device_descriptor_valid(
              *descriptor, layer_index, role, expected_offset, arena_begin,
              owner.bytes_, owner.owner_identity_, owner.allocation_identity_,
              owner.device_ordinal_)) {
        return false;
      }
      expected_offset += layout.payload_bytes;
    }
  }
  if (expected_offset != owner.bytes_) {
    return false;
  }
  Sm87TargetAotCompleteProjectionAttachment attachment;
  attachment.owner = &owner;
  attachment.owner_identity = owner.owner_identity_;
  attachment.allocation_identity = owner.allocation_identity_;
  attachment.arena_begin = arena_begin;
  attachment.arena_bytes = owner.bytes_;
  attachment.device_ordinal = owner.device_ordinal_;
  attachment.artifact_count = owner.descriptor_count_;
  owner.execution_bound_ = true;
  target_aot_complete_projection_attachment_ = attachment;
  return true;
}

bool ModelWeights::attach_sm87_target_aot_projection_assets(
    Sm87TargetAotProjectionDeviceAssets& owner) noexcept {
  using kernels::Sm87TargetAotLogicalRole;
  using kernels::Sm87TargetAotProjectionRole;

  if (target_aot_projection_attachment_.owner != nullptr ||
      target_aot_complete_projection_attachment_.owner != nullptr ||
      owner.empty() ||
      owner.attached_model_weights_ != nullptr ||
      owner.prepared_model_weights_ != this || owner.data() == nullptr ||
      owner.size_bytes() != kSm87TargetAotProjectionDeviceArenaBytes ||
      owner.owner_identity() == 0U || owner.allocation_identity() == 0U ||
      owner.device_ordinal() < 0 ||
      owner.descriptor_count() !=
          kSm87TargetAotProjectionDeviceArtifactCount) {
    return false;
  }
  const auto arena_begin =
      reinterpret_cast<std::uintptr_t>(owner.data());
  if (arena_begin == 0U ||
      owner.size_bytes() > std::numeric_limits<std::uintptr_t>::max() ||
      arena_begin > std::numeric_limits<std::uintptr_t>::max() -
                        static_cast<std::uintptr_t>(owner.size_bytes())) {
    return false;
  }
  const auto arena_end =
      arena_begin + static_cast<std::uintptr_t>(owner.size_bytes());
#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
  int active_device_ordinal = -1;
  cudaPointerAttributes arena_attributes{};
  if (cudaGetDevice(&active_device_ordinal) != cudaSuccess ||
      active_device_ordinal != owner.device_ordinal() ||
      cudaPointerGetAttributes(&arena_attributes, owner.data()) !=
          cudaSuccess ||
      arena_attributes.type != cudaMemoryTypeDevice ||
      arena_attributes.device != owner.device_ordinal()) {
    return false;
  }
#endif
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceArtifactCount>
      artifact_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceArtifactCount>
      source_inventory_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceSourceCount>
      tensor_identities{};
  std::size_t artifact_index = 0U;
  std::size_t source_inventory_index = 0U;
  std::size_t tensor_identity_index = 0U;
  std::uint64_t expected_offset = 0U;

  const auto float_bits = [](const float value) noexcept {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    if (!has_valid_nvfp4_payload(gate) || !has_valid_nvfp4_payload(up) ||
        !has_valid_nvfp4_payload(down) || gate->output_size != 17'408U ||
        gate->input_size != 5'120U || up->output_size != 17'408U ||
        up->input_size != 5'120U || down->output_size != 5'120U ||
        down->input_size != 17'408U ||
        gate->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        up->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        down->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        gate->prefill_marlin_weight != nullptr ||
        gate->prefill_marlin_scales != nullptr ||
        gate->prefill_marlin_global_scale != nullptr ||
        up->prefill_marlin_weight != nullptr ||
        up->prefill_marlin_scales != nullptr ||
        up->prefill_marlin_global_scale != nullptr ||
        down->prefill_marlin_weight != nullptr ||
        down->prefill_marlin_scales != nullptr ||
        down->prefill_marlin_global_scale != nullptr ||
        !empty_p40_packed_artifact_view(
            gate->prefill_p40_packed_artifact) ||
        !empty_p40_packed_artifact_view(up->prefill_p40_packed_artifact) ||
        !empty_p40_packed_artifact_view(
            down->prefill_p40_packed_artifact) ||
        !empty_nvfp4_marlin_p40_parity_view(
            gate->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            up->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            down->prefill_p40_vllm_marlin_parity)) {
      return false;
    }

    const auto validate = [&](const Sm87TargetAotProjectionRole role,
                              const std::uint32_t gate_bits,
                              const std::uint32_t up_bits,
                              const std::uint32_t source_count) noexcept {
      const auto* const descriptor = owner.find(layer_index, role);
      const auto layout =
          kernels::sm87_target_aot_projection_packed_layout(role);
      if (descriptor == nullptr || !layout.valid() ||
          expected_offset > owner.size_bytes() ||
          layout.payload_bytes > owner.size_bytes() - expected_offset ||
          descriptor->layer_index != layer_index ||
          descriptor->role != role ||
          descriptor->device_arena_offset != expected_offset ||
          descriptor->manifest.source_count != source_count ||
          descriptor->source_inventory.source_count != source_count ||
          !kernels::sm87_target_aot_projection_validate_transform_receipt(
              descriptor->manifest, descriptor->source_inventory,
              descriptor->transform_receipt) ||
          !kernels::
              sm87_target_aot_nvfp4_cuda_device_upload_receipt_structurally_valid(
                  descriptor->manifest, descriptor->upload_receipt) ||
          !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(
              descriptor->view) ||
          descriptor->view.artifact_identity !=
              descriptor->manifest.artifact_identity ||
          descriptor->view.source_inventory_identity !=
              descriptor->source_inventory.identity ||
          descriptor->view.transform_identity !=
              descriptor->transform_receipt.transform_identity ||
          descriptor->view.host_payload_digest !=
              descriptor->manifest.payload_digest ||
          descriptor->view.host_manifest_seal != descriptor->manifest.seal ||
          descriptor->view.device_upload_receipt.receipt_identity !=
              descriptor->upload_receipt.receipt_identity ||
          descriptor->view.payload.begin !=
              arena_begin + static_cast<std::uintptr_t>(expected_offset) ||
          descriptor->view.payload.end !=
              descriptor->view.payload.begin +
                  static_cast<std::uintptr_t>(layout.payload_bytes) ||
          descriptor->view.payload.end > arena_end ||
          descriptor->upload_receipt.device_allocation_identity !=
              owner.allocation_identity() ||
          descriptor->upload_receipt.device_allocation_owner_identity !=
              owner.owner_identity() ||
          descriptor->upload_receipt.device_ordinal !=
              owner.device_ordinal() ||
          descriptor->upload_receipt.device_allocation_begin != arena_begin ||
          descriptor->upload_receipt.device_allocation_end != arena_end ||
          descriptor->upload_receipt.device_allocation_bytes !=
              owner.size_bytes() ||
          descriptor->manifest.sources[0U].tensor_scale_bits != gate_bits ||
          (source_count == 2U &&
           descriptor->manifest.sources[1U].tensor_scale_bits != up_bits) ||
          artifact_index >= artifact_identities.size() ||
          source_inventory_index >= source_inventory_identities.size() ||
          descriptor->manifest.artifact_identity == 0U ||
          descriptor->source_inventory.identity == 0U ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifact_index,
                    descriptor->manifest.artifact_identity) !=
              artifact_identities.begin() + artifact_index ||
          std::find(source_inventory_identities.begin(),
                    source_inventory_identities.begin() +
                        source_inventory_index,
                    descriptor->source_inventory.identity) !=
              source_inventory_identities.begin() +
                  source_inventory_index) {
        return false;
      }
      if ((role == Sm87TargetAotProjectionRole::kNvFp4GateUp &&
           (descriptor->manifest.sources[0U].logical_role !=
                Sm87TargetAotLogicalRole::kNvFp4Gate ||
            descriptor->manifest.sources[1U].logical_role !=
                Sm87TargetAotLogicalRole::kNvFp4Up)) ||
          (role == Sm87TargetAotProjectionRole::kNvFp4Down &&
           descriptor->manifest.sources[0U].logical_role !=
               Sm87TargetAotLogicalRole::kNvFp4Down)) {
        return false;
      }
      for (std::size_t source_index = 0U; source_index < source_count;
           ++source_index) {
        const std::uint64_t tensor_identity =
            descriptor->source_inventory.sources[source_index]
                .tensor_identity;
        if (tensor_identity == 0U ||
            tensor_identity_index >= tensor_identities.size() ||
            std::find(tensor_identities.begin(),
                      tensor_identities.begin() + tensor_identity_index,
                      tensor_identity) !=
                tensor_identities.begin() + tensor_identity_index) {
          return false;
        }
        tensor_identities[tensor_identity_index++] = tensor_identity;
      }
      artifact_identities[artifact_index++] =
          descriptor->manifest.artifact_identity;
      source_inventory_identities[source_inventory_index++] =
          descriptor->source_inventory.identity;
      expected_offset += layout.payload_bytes;
      return true;
    };

    if (!validate(Sm87TargetAotProjectionRole::kNvFp4GateUp,
                  float_bits(gate->weight_scale_2),
                  float_bits(up->weight_scale_2), 2U) ||
        !validate(Sm87TargetAotProjectionRole::kNvFp4Down,
                  float_bits(down->weight_scale_2), 0U, 1U)) {
      return false;
    }
  }
  if (artifact_index != artifact_identities.size() ||
      source_inventory_index != source_inventory_identities.size() ||
      tensor_identity_index != tensor_identities.size() ||
      expected_offset != owner.size_bytes()) {
    return false;
  }

  Sm87TargetAotProjectionAttachment attachment;
  attachment.owner = &owner;
  attachment.owner_identity = owner.owner_identity();
  attachment.allocation_identity = owner.allocation_identity();
  attachment.arena_begin = arena_begin;
  attachment.arena_bytes = owner.size_bytes();
  attachment.device_ordinal = owner.device_ordinal();
  attachment.artifact_count = artifact_index;
  target_aot_projection_attachment_ = attachment;
  owner.attached_model_weights_ = this;
  return true;
}

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
namespace target_aot_complete_execution_detail {
namespace {

using Role = kernels::Sm87TargetAotProjectionRole;
using Encoding = kernels::Sm87TargetAotProjectionEncoding;
using Descriptor = Sm87TargetAotCompleteDeviceAssetDescriptor;
using Owner = Sm87TargetAotCompleteProjectionDeviceAssets;

[[nodiscard]] constexpr std::array<Role, 4U> layer_roles(
    const std::size_t layer_index) noexcept {
  return {{Role::kNvFp4GateUp, Role::kNvFp4Down,
           sm87_target_aot_complete_is_full_layer(layer_index)
               ? Role::kFp8FullQkv
               : Role::kFp8GdnQkvZ,
           Role::kFp8AttentionOutput}};
}

}  // namespace

bool Sm87TargetAotCompleteProjectionExecutionAccess::attachment_matches(
    const ModelWeights* const model_weights, const Owner* const owner,
    const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::uintptr_t arena_begin, const std::uint64_t arena_bytes,
    const std::int32_t device_ordinal) noexcept {
  if (model_weights == nullptr || owner == nullptr || owner_identity == 0U ||
      allocation_identity == 0U || arena_begin == 0U ||
      arena_bytes != kSm87TargetAotCompleteProjectionDeviceArenaBytes ||
      device_ordinal < 0 ||
      arena_bytes > std::numeric_limits<std::uintptr_t>::max() - arena_begin ||
      owner->arena_ == nullptr ||
      reinterpret_cast<std::uintptr_t>(owner->arena_) != arena_begin ||
      owner->bytes_ != arena_bytes || owner->owner_identity_ != owner_identity ||
      owner->allocation_identity_ != allocation_identity ||
      owner->device_ordinal_ != device_ordinal ||
      owner->descriptor_count_ !=
          kSm87TargetAotCompleteProjectionDeviceArtifactCount ||
      owner->prepared_resident_ == nullptr ||
      owner->prepared_model_weights_ != model_weights ||
      !owner->execution_bound_ ||
      model_weights->target_aot_projection_attachment_.owner != nullptr) {
    return false;
  }
  const auto& attachment =
      model_weights->target_aot_complete_projection_attachment_;
  return attachment.owner == owner &&
         attachment.owner_identity == owner_identity &&
         attachment.allocation_identity == allocation_identity &&
         attachment.arena_begin == arena_begin &&
         attachment.arena_bytes == arena_bytes &&
         attachment.device_ordinal == device_ordinal &&
         attachment.artifact_count ==
             kSm87TargetAotCompleteProjectionDeviceArtifactCount;
}

bool Sm87TargetAotCompleteProjectionExecutionAccess::descriptor_matches(
    const Owner& owner, const Descriptor& descriptor,
    const std::size_t layer_index, const Role role,
    const std::uint64_t expected_offset) noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(role);
  const std::uintptr_t arena_begin =
      reinterpret_cast<std::uintptr_t>(owner.arena_);
  if (index >= owner.descriptors_.size() ||
      index >= owner.descriptor_count_ ||
      &descriptor != &owner.descriptors_[index] || !layout.valid() ||
      expected_offset !=
          sm87_target_aot_complete_expected_arena_offset(layer_index, role) ||
      descriptor.encoding != layout.encoding ||
      descriptor.manifest.artifact_identity == 0U ||
      descriptor.source_inventory.identity == 0U ||
      descriptor.source_inventory.source_count != layout.partition_count ||
      !sm87_target_aot_complete_device_descriptor_valid(
          descriptor, layer_index, role, expected_offset, arena_begin,
          owner.bytes_, owner.owner_identity_, owner.allocation_identity_,
          owner.device_ordinal_)) {
    return false;
  }

  if (sm87_target_aot_complete_role_is_nvfp4(role)) {
    return descriptor.encoding ==
               Encoding::kNvFp4E2M1Block16E4M3FnScale &&
           descriptor.nvfp4_upload_receipt.receipt_identity != 0U &&
           descriptor.nvfp4_upload_receipt.receipt_identity ==
               descriptor.nvfp4_view.device_upload_receipt.receipt_identity &&
           descriptor.nvfp4_view.payload.begin ==
               arena_begin + static_cast<std::uintptr_t>(expected_offset);
  }
  if (sm87_target_aot_complete_role_is_fp8(role)) {
    return descriptor.encoding == Encoding::kFp8E4M3FnTensorScale &&
           descriptor.fp8_upload_receipt.receipt_identity != 0U &&
           descriptor.fp8_upload_receipt.receipt_identity ==
               descriptor.fp8_view.device_upload_receipt.receipt_identity &&
           descriptor.fp8_view.payload.begin ==
               arena_begin + static_cast<std::uintptr_t>(expected_offset);
  }
  return false;
}

std::optional<Sm87TargetAotCompleteProjectionExecutionAccess>
Sm87TargetAotCompleteProjectionExecutionAccess::bind(
    const ModelWeights& model_weights) noexcept {
  const auto& attachment =
      model_weights.target_aot_complete_projection_attachment_;
  const Owner* const owner = attachment.owner;
  if (!attachment_matches(
          &model_weights, owner, attachment.owner_identity,
          attachment.allocation_identity, attachment.arena_begin,
          attachment.arena_bytes, attachment.device_ordinal)) {
    return std::nullopt;
  }

  std::array<const Descriptor*,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      descriptors{};
  std::array<std::uint64_t,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      artifact_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotCompleteProjectionDeviceArtifactCount>
      inventory_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotCompleteProjectionDeviceSourceCount>
      source_identities{};
  std::size_t artifact_count = 0U;
  std::size_t source_count = 0U;
  std::uint64_t expected_offset = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87TargetAotCompleteProjectionDeviceLayerCount;
       ++layer_index) {
    for (const Role role : layer_roles(layer_index)) {
      const std::size_t index =
          sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
      if (index != artifact_count || index >= descriptors.size()) {
        return std::nullopt;
      }
      const Descriptor* const descriptor = &owner->descriptors_[index];
      if (!descriptor_matches(*owner, *descriptor, layer_index, role,
                              expected_offset) ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifact_count,
                    descriptor->manifest.artifact_identity) !=
              artifact_identities.begin() + artifact_count ||
          std::find(inventory_identities.begin(),
                    inventory_identities.begin() + artifact_count,
                    descriptor->source_inventory.identity) !=
              inventory_identities.begin() + artifact_count) {
        return std::nullopt;
      }
      for (std::size_t source_index = 0U;
           source_index < descriptor->source_inventory.source_count;
           ++source_index) {
        const std::uint64_t identity =
            descriptor->source_inventory.sources[source_index]
                .tensor_identity;
        if (identity == 0U || source_count >= source_identities.size() ||
            std::find(source_identities.begin(),
                      source_identities.begin() + source_count,
                      identity) != source_identities.begin() + source_count) {
          return std::nullopt;
        }
        source_identities[source_count++] = identity;
      }
      descriptors[index] = descriptor;
      artifact_identities[artifact_count] =
          descriptor->manifest.artifact_identity;
      inventory_identities[artifact_count] =
          descriptor->source_inventory.identity;
      ++artifact_count;
      expected_offset += descriptor->manifest.payload_bytes;
    }
  }
  if (artifact_count != descriptors.size() ||
      source_count != source_identities.size() ||
      expected_offset != attachment.arena_bytes) {
    return std::nullopt;
  }
  return Sm87TargetAotCompleteProjectionExecutionAccess(
      &model_weights, owner, attachment.owner_identity,
      attachment.allocation_identity, attachment.arena_begin,
      attachment.arena_bytes, attachment.device_ordinal, descriptors);
}

bool Sm87TargetAotCompleteProjectionExecutionAccess::attached()
    const noexcept {
  return attachment_matches(model_weights_, owner_, owner_identity_,
                            allocation_identity_, arena_begin_, arena_bytes_,
                            device_ordinal_);
}

std::optional<Sm87TargetAotCompleteProjectionExecutionAsset>
Sm87TargetAotCompleteProjectionExecutionAccess::resolve(
    const std::size_t layer_index, const Role role) const noexcept {
  const std::size_t index =
      sm87_target_aot_complete_descriptor_ordinal(layer_index, role);
  if (index >= descriptors_.size() || !attached()) {
    return std::nullopt;
  }
  const Descriptor* const descriptor = descriptors_[index];
  const std::uint64_t expected_offset =
      sm87_target_aot_complete_expected_arena_offset(layer_index, role);
  if (descriptor == nullptr ||
      !descriptor_matches(*owner_, *descriptor, layer_index, role,
                          expected_offset)) {
    return std::nullopt;
  }
  return Sm87TargetAotCompleteProjectionExecutionAsset(
      model_weights_, owner_, descriptor, owner_identity_,
      allocation_identity_, arena_begin_, arena_bytes_, device_ordinal_,
      layer_index, role, descriptor->encoding,
      descriptor->manifest.artifact_identity,
      descriptor->source_inventory.identity,
      descriptor->manifest.payload_bytes);
}

const kernels::Sm87TargetAotNvFp4CudaAssetView*
Sm87TargetAotCompleteProjectionExecutionAsset::borrow_nvfp4_cuda_asset()
    const noexcept {
  if (!sm87_target_aot_complete_role_is_nvfp4(role_) ||
      encoding_ != Encoding::kNvFp4E2M1Block16E4M3FnScale ||
      !Sm87TargetAotCompleteProjectionExecutionAccess::attachment_matches(
          model_weights_, owner_, owner_identity_, allocation_identity_,
          arena_begin_, arena_bytes_, device_ordinal_)) {
    return nullptr;
  }
  const std::uint64_t expected_offset =
      sm87_target_aot_complete_expected_arena_offset(layer_index_, role_);
  if (descriptor_ == nullptr ||
      !Sm87TargetAotCompleteProjectionExecutionAccess::descriptor_matches(
          *owner_, *descriptor_, layer_index_, role_, expected_offset) ||
      descriptor_->encoding != encoding_ ||
      descriptor_->manifest.artifact_identity != artifact_identity_ ||
      descriptor_->source_inventory.identity != source_inventory_identity_ ||
      descriptor_->manifest.payload_bytes != payload_bytes_) {
    return nullptr;
  }
  return &descriptor_->nvfp4_view;
}

const kernels::Sm87TargetAotFp8CudaAssetView*
Sm87TargetAotCompleteProjectionExecutionAsset::borrow_fp8_cuda_asset()
    const noexcept {
  if (!sm87_target_aot_complete_role_is_fp8(role_) ||
      encoding_ != Encoding::kFp8E4M3FnTensorScale ||
      !Sm87TargetAotCompleteProjectionExecutionAccess::attachment_matches(
          model_weights_, owner_, owner_identity_, allocation_identity_,
          arena_begin_, arena_bytes_, device_ordinal_)) {
    return nullptr;
  }
  const std::uint64_t expected_offset =
      sm87_target_aot_complete_expected_arena_offset(layer_index_, role_);
  if (descriptor_ == nullptr ||
      !Sm87TargetAotCompleteProjectionExecutionAccess::descriptor_matches(
          *owner_, *descriptor_, layer_index_, role_, expected_offset) ||
      descriptor_->encoding != encoding_ ||
      descriptor_->manifest.artifact_identity != artifact_identity_ ||
      descriptor_->source_inventory.identity != source_inventory_identity_ ||
      descriptor_->manifest.payload_bytes != payload_bytes_) {
    return nullptr;
  }
  return &descriptor_->fp8_view;
}

}  // namespace target_aot_complete_execution_detail
#endif

#if defined(Q3X_ENABLE_SM87_TARGET_AOT_DEVICE_ASSETS_V1_ADMISSION)
namespace target_aot_execution_detail {
namespace {

using Role = kernels::Sm87TargetAotProjectionRole;
using Descriptor = Sm87TargetAotProjectionDeviceAssetDescriptor;
using Owner = Sm87TargetAotProjectionDeviceAssets;

[[nodiscard]] constexpr std::size_t role_ordinal(
    const Role role) noexcept {
  return role == Role::kNvFp4GateUp
             ? 0U
             : (role == Role::kNvFp4Down ? 1U : 2U);
}

[[nodiscard]] constexpr std::uint64_t expected_artifact_offset(
    const std::size_t layer_index, const Role role) noexcept {
  const auto gate_up = kernels::sm87_target_aot_projection_packed_layout(
      Role::kNvFp4GateUp);
  const auto down = kernels::sm87_target_aot_projection_packed_layout(
      Role::kNvFp4Down);
  return static_cast<std::uint64_t>(layer_index) *
             (gate_up.payload_bytes + down.payload_bytes) +
         (role == Role::kNvFp4Down ? gate_up.payload_bytes : 0U);
}

}  // namespace

std::size_t Sm87TargetAotProjectionExecutionAccess::ordinal(
    const std::size_t layer_index, const Role role) noexcept {
  const std::size_t role_index = role_ordinal(role);
  if (layer_index >= kSm87TargetAotProjectionDeviceLayerCount ||
      role_index >= 2U) {
    return kSm87TargetAotProjectionDeviceArtifactCount;
  }
  return 2U * layer_index + role_index;
}

bool Sm87TargetAotProjectionExecutionAccess::attachment_matches(
    const ModelWeights* const model_weights, const Owner* const owner,
    const std::uint64_t owner_identity,
    const std::uint64_t allocation_identity,
    const std::uintptr_t arena_begin, const std::uint64_t arena_bytes,
    const std::int32_t device_ordinal) noexcept {
  if (model_weights == nullptr || owner == nullptr || owner_identity == 0U ||
      allocation_identity == 0U || arena_begin == 0U ||
      arena_bytes != kSm87TargetAotProjectionDeviceArenaBytes ||
      device_ordinal < 0 || owner->arena_ == nullptr ||
      reinterpret_cast<std::uintptr_t>(owner->arena_) != arena_begin ||
      owner->bytes_ != arena_bytes || owner->owner_identity_ != owner_identity ||
      owner->allocation_identity_ != allocation_identity ||
      owner->device_ordinal_ != device_ordinal ||
      owner->descriptor_count_ !=
          kSm87TargetAotProjectionDeviceArtifactCount ||
      owner->prepared_model_weights_ != model_weights ||
      owner->attached_model_weights_ != model_weights) {
    return false;
  }
  const auto& attachment = model_weights->target_aot_projection_attachment_;
  return attachment.owner == owner &&
         attachment.owner_identity == owner_identity &&
         attachment.allocation_identity == allocation_identity &&
         attachment.arena_begin == arena_begin &&
         attachment.arena_bytes == arena_bytes &&
         attachment.device_ordinal == device_ordinal &&
         attachment.artifact_count ==
             kSm87TargetAotProjectionDeviceArtifactCount;
}

bool Sm87TargetAotProjectionExecutionAccess::descriptor_matches(
    const Owner& owner, const Descriptor& descriptor,
    const std::size_t layer_index, const Role role,
    const std::uint64_t expected_offset) noexcept {
  const auto layout =
      kernels::sm87_target_aot_projection_packed_layout(role);
  if (!layout.valid() || descriptor.layer_index != layer_index ||
      descriptor.role != role ||
      descriptor.device_arena_offset != expected_offset ||
      expected_offset > owner.bytes_ ||
      layout.payload_bytes > owner.bytes_ - expected_offset ||
      !kernels::sm87_target_aot_projection_validate_transform_receipt(
          descriptor.manifest, descriptor.source_inventory,
          descriptor.transform_receipt) ||
      !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(descriptor.view) ||
      descriptor.view.artifact_identity !=
          descriptor.manifest.artifact_identity ||
      descriptor.view.source_inventory_identity !=
          descriptor.source_inventory.identity ||
      descriptor.view.transform_identity !=
          descriptor.transform_receipt.transform_identity ||
      descriptor.view.host_payload_digest !=
          descriptor.manifest.payload_digest ||
      descriptor.view.host_manifest_seal != descriptor.manifest.seal ||
      descriptor.upload_receipt.receipt_identity !=
          descriptor.view.device_upload_receipt.receipt_identity ||
      descriptor.upload_receipt.device_allocation_owner_identity !=
          owner.owner_identity_ ||
      descriptor.upload_receipt.device_allocation_identity !=
          owner.allocation_identity_ ||
      descriptor.upload_receipt.device_ordinal != owner.device_ordinal_ ||
      descriptor.upload_receipt.device_allocation_begin !=
          reinterpret_cast<std::uintptr_t>(owner.arena_) ||
      descriptor.upload_receipt.device_allocation_bytes != owner.bytes_ ||
      descriptor.upload_receipt.device_payload_begin !=
          reinterpret_cast<std::uintptr_t>(owner.arena_) +
              static_cast<std::uintptr_t>(expected_offset) ||
      descriptor.upload_receipt.device_payload_bytes != layout.payload_bytes ||
      descriptor.view.payload.begin !=
          descriptor.upload_receipt.device_payload_begin ||
      descriptor.view.payload.end !=
          descriptor.upload_receipt.device_payload_end ||
      descriptor.view.payload.bytes != layout.payload_bytes) {
    return false;
  }
  return true;
}

std::optional<Sm87TargetAotProjectionExecutionAccess>
Sm87TargetAotProjectionExecutionAccess::bind(
    const ModelWeights& model_weights) noexcept {
  const auto& attachment = model_weights.target_aot_projection_attachment_;
  const Owner* const owner = attachment.owner;
  if (!attachment_matches(
          &model_weights, owner, attachment.owner_identity,
          attachment.allocation_identity, attachment.arena_begin,
          attachment.arena_bytes, attachment.device_ordinal)) {
    return std::nullopt;
  }

  std::array<const Descriptor*, kSm87TargetAotProjectionDeviceArtifactCount>
      descriptors{};
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceArtifactCount>
      artifact_identities{};
  std::array<std::uint64_t,
             kSm87TargetAotProjectionDeviceArtifactCount>
      inventory_identities{};
  std::array<std::uint64_t, kSm87TargetAotProjectionDeviceSourceCount>
      source_identities{};
  std::size_t artifact_count = 0U;
  std::size_t source_count = 0U;
  std::uint64_t expected_offset = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kSm87TargetAotProjectionDeviceLayerCount;
       ++layer_index) {
    for (const Role role : {Role::kNvFp4GateUp, Role::kNvFp4Down}) {
      const std::size_t index = ordinal(layer_index, role);
      if (index >= descriptors.size()) {
        return std::nullopt;
      }
      const Descriptor* const descriptor = &owner->descriptors_[index];
      if (!descriptor_matches(*owner, *descriptor, layer_index, role,
                              expected_offset) ||
          descriptor->manifest.artifact_identity == 0U ||
          descriptor->source_inventory.identity == 0U ||
          std::find(artifact_identities.begin(),
                    artifact_identities.begin() + artifact_count,
                    descriptor->manifest.artifact_identity) !=
              artifact_identities.begin() + artifact_count ||
          std::find(inventory_identities.begin(),
                    inventory_identities.begin() + artifact_count,
                    descriptor->source_inventory.identity) !=
              inventory_identities.begin() + artifact_count) {
        return std::nullopt;
      }
      for (std::size_t source_index = 0U;
           source_index < descriptor->source_inventory.source_count;
           ++source_index) {
        const std::uint64_t identity =
            descriptor->source_inventory.sources[source_index].tensor_identity;
        if (identity == 0U || source_count >= source_identities.size() ||
            std::find(source_identities.begin(),
                      source_identities.begin() + source_count,
                      identity) != source_identities.begin() + source_count) {
          return std::nullopt;
        }
        source_identities[source_count++] = identity;
      }
      descriptors[index] = descriptor;
      artifact_identities[artifact_count] =
          descriptor->manifest.artifact_identity;
      inventory_identities[artifact_count] =
          descriptor->source_inventory.identity;
      ++artifact_count;
      expected_offset += descriptor->manifest.payload_bytes;
    }
  }
  if (artifact_count != descriptors.size() ||
      source_count != source_identities.size() ||
      expected_offset != attachment.arena_bytes) {
    return std::nullopt;
  }
  return Sm87TargetAotProjectionExecutionAccess(
      &model_weights, owner, attachment.owner_identity,
      attachment.allocation_identity, attachment.arena_begin,
      attachment.arena_bytes, attachment.device_ordinal, descriptors);
}

bool Sm87TargetAotProjectionExecutionAccess::attached() const noexcept {
  return attachment_matches(model_weights_, owner_, owner_identity_,
                            allocation_identity_, arena_begin_, arena_bytes_,
                            device_ordinal_);
}

std::optional<Sm87TargetAotProjectionExecutionAsset>
Sm87TargetAotProjectionExecutionAccess::resolve(
    const std::size_t layer_index, const Role role) const noexcept {
  const std::size_t index = ordinal(layer_index, role);
  if (index >= descriptors_.size() || !attached()) {
    return std::nullopt;
  }
  const Descriptor* const descriptor = descriptors_[index];
  const std::uint64_t expected_offset =
      expected_artifact_offset(layer_index, role);
  if (descriptor == nullptr || descriptor != &owner_->descriptors_[index] ||
      !descriptor_matches(*owner_, *descriptor, layer_index, role,
                          expected_offset)) {
    return std::nullopt;
  }
  return Sm87TargetAotProjectionExecutionAsset(
      model_weights_, owner_, descriptor, owner_identity_,
      allocation_identity_, arena_begin_, arena_bytes_, device_ordinal_,
      layer_index, role,
      descriptor->manifest.artifact_identity,
      descriptor->source_inventory.identity,
      descriptor->manifest.payload_bytes);
}

const kernels::Sm87TargetAotNvFp4CudaAssetView*
Sm87TargetAotProjectionExecutionAsset::borrow_cuda_asset() const noexcept {
  if (!Sm87TargetAotProjectionExecutionAccess::attachment_matches(
          model_weights_, owner_, owner_identity_, allocation_identity_,
          arena_begin_, arena_bytes_, device_ordinal_)) {
    return nullptr;
  }
  const std::uint64_t expected_offset =
      expected_artifact_offset(layer_index_, role_);
  if (descriptor_ == nullptr ||
      !Sm87TargetAotProjectionExecutionAccess::descriptor_matches(
          *owner_, *descriptor_, layer_index_, role_, expected_offset) ||
      descriptor_->manifest.artifact_identity != artifact_identity_ ||
      descriptor_->source_inventory.identity != source_inventory_identity_ ||
      descriptor_->manifest.payload_bytes != payload_bytes_) {
    return nullptr;
  }
  return &descriptor_->view;
}

}  // namespace target_aot_execution_detail
#endif

class ModelWeightBinder {
 public:
  explicit ModelWeightBinder(const WeightBindingSource& source)
      : source_(source) {}

  [[nodiscard]] WeightBindResult run() {
    WeightBindResult result;
    if (!validate_source() || !validate_config()) {
      result.diagnostic = std::move(diagnostic_);
      return result;
    }

    ModelWeights weights;
    weights.embed_tokens_ = bind_bf16_matrix(
        "model.language_model.embed_tokens.weight", config_->vocab_size,
        config_->hidden_size);
    weights.final_norm_ = bind_bf16_vector(
        "model.language_model.norm.weight", config_->hidden_size);
    if (!ok()) {
      result.diagnostic = std::move(diagnostic_);
      return result;
    }

    for (std::size_t layer_index = 0U;
         layer_index < kQwen36DenseLayerCount; ++layer_index) {
      const std::string prefix = "model.language_model.layers." +
                                 std::to_string(layer_index) + ".";
      DecoderLayerWeights layer;
      layer.input_layernorm = bind_bf16_vector(
          prefix + "input_layernorm.weight", config_->hidden_size);
      layer.post_attention_layernorm = bind_bf16_vector(
          prefix + "post_attention_layernorm.weight", config_->hidden_size);
      layer.mlp.gate_proj = bind_linear(
          prefix + "mlp.gate_proj", config_->intermediate_size,
          config_->hidden_size);
      layer.mlp.up_proj = bind_linear(
          prefix + "mlp.up_proj", config_->intermediate_size,
          config_->hidden_size);
      layer.mlp.down_proj = bind_linear(
          prefix + "mlp.down_proj", config_->hidden_size,
          config_->intermediate_size);
      if (!ok()) {
        result.diagnostic = std::move(diagnostic_);
        return result;
      }

      if (config_->layer_type(layer_index) ==
          model::LayerType::kLinearAttention) {
        LinearAttentionWeights attention;
        attention.in_proj_qkv = bind_linear(
            prefix + "linear_attn.in_proj_qkv", config_->linear_qkv_projection_dim(),
            config_->hidden_size);
        attention.in_proj_z = bind_linear(
            prefix + "linear_attn.in_proj_z", config_->linear_value_dim(),
            config_->hidden_size);
        attention.in_proj_a = bind_linear(
            prefix + "linear_attn.in_proj_a", config_->linear_num_value_heads,
            config_->hidden_size);
        attention.in_proj_b = bind_linear(
            prefix + "linear_attn.in_proj_b", config_->linear_num_value_heads,
            config_->hidden_size);
        attention.conv1d = bind_bf16_tensor3(
            prefix + "linear_attn.conv1d.weight",
            config_->linear_qkv_projection_dim(), 1U,
            config_->linear_conv_kernel_dim);
        attention.a_log = bind_bf16_vector(
            prefix + "linear_attn.A_log", config_->linear_num_value_heads);
        attention.dt_bias = bind_bf16_vector(
            prefix + "linear_attn.dt_bias", config_->linear_num_value_heads);
        attention.norm = bind_bf16_vector(
            prefix + "linear_attn.norm.weight",
            config_->linear_value_head_dim);
        attention.out_proj = bind_linear(
            prefix + "linear_attn.out_proj", config_->hidden_size,
            config_->linear_value_dim());
        layer.attention = std::move(attention);
        ++stats_.linear_attention_layers;
      } else if (config_->layer_type(layer_index) ==
                 model::LayerType::kFullAttention) {
        FullAttentionWeights attention;
        attention.q_proj = bind_linear(
            prefix + "self_attn.q_proj", config_->q_projection_dim(),
            config_->hidden_size);
        attention.k_proj = bind_linear(
            prefix + "self_attn.k_proj", config_->kv_dim(),
            config_->hidden_size);
        attention.v_proj = bind_linear(
            prefix + "self_attn.v_proj", config_->kv_dim(),
            config_->hidden_size);
        attention.o_proj = bind_linear(
            prefix + "self_attn.o_proj", config_->hidden_size,
            config_->q_dim());
        attention.q_norm = bind_bf16_vector(
            prefix + "self_attn.q_norm.weight", config_->head_dim);
        attention.k_norm = bind_bf16_vector(
            prefix + "self_attn.k_norm.weight", config_->head_dim);
        layer.attention = std::move(attention);
        ++stats_.full_attention_layers;
      } else {
        fail(WeightBindErrorCode::kInvalidLayerSchedule, prefix,
             "model catalog returned an invalid hybrid-attention layer type");
      }
      if (!ok()) {
        result.diagnostic = std::move(diagnostic_);
        return result;
      }
      weights.layers_[layer_index] = std::move(layer);
    }

    weights.lm_head_ = bind_linear("lm_head", config_->vocab_size,
                                   config_->hidden_size);
    if (!ok()) {
      result.diagnostic = std::move(diagnostic_);
      return result;
    }
    if (stats_.linear_attention_layers !=
            kQwen36LinearAttentionLayerCount ||
        stats_.full_attention_layers != kQwen36FullAttentionLayerCount) {
      fail(WeightBindErrorCode::kInvalidLayerSchedule, "decoder layers",
           "bound hybrid-attention layer counts do not match Qwen3.6-27B",
           "48 linear / 16 full",
           std::to_string(stats_.linear_attention_layers) + " linear / " +
               std::to_string(stats_.full_attention_layers) + " full");
      result.diagnostic = std::move(diagnostic_);
      return result;
    }
    weights.stats_ = stats_;
    result.value.emplace(std::move(weights));
    return result;
  }

 private:
  [[nodiscard]] bool ok() const noexcept {
    return diagnostic_.code == WeightBindErrorCode::kNone;
  }

  void fail(const WeightBindErrorCode code, std::string tensor,
            std::string message, std::string expected = {},
            std::string actual = {}, const int cuda_error = 0) {
    if (!ok()) {
      return;
    }
    diagnostic_.code = code;
    diagnostic_.tensor = std::move(tensor);
    diagnostic_.message = std::move(message);
    diagnostic_.expected = std::move(expected);
    diagnostic_.actual = std::move(actual);
    diagnostic_.cuda_error = cuda_error;
  }

  [[nodiscard]] bool validate_source() {
    if (source_.lookup == nullptr || source_.arena_data == nullptr ||
        source_.arena_bytes == 0U) {
      fail(WeightBindErrorCode::kInvalidSource, "binding source",
           "lookup, arena pointer, and nonzero arena size are required");
      return false;
    }
    const auto base = reinterpret_cast<std::uintptr_t>(source_.arena_data);
    if ((base % kResidentTensorAlignment) != 0U) {
      fail(WeightBindErrorCode::kMisalignedTensor, "binding arena",
           "arena base is not 256-byte aligned", "256-byte alignment");
      return false;
    }
    constexpr auto kPointerMaximum =
        std::numeric_limits<std::uintptr_t>::max();
    if (source_.arena_bytes > kPointerMaximum - base) {
      fail(WeightBindErrorCode::kArithmeticOverflow, "binding arena",
           "arena pointer range overflows uintptr_t");
      return false;
    }
    arena_base_ = base;
    arena_end_ = base + static_cast<std::uintptr_t>(source_.arena_bytes);
    return true;
  }

  [[nodiscard]] bool validate_config() {
    config_ = model::find_known_model(model::KnownModel::kQwen36_27B);
    if (config_ == nullptr || !model::validate(*config_) ||
        config_->num_hidden_layers != kQwen36DenseLayerCount ||
        config_->hidden_size != 5120U || config_->vocab_size != 248320U ||
        config_->intermediate_size != 17408U ||
        config_->linear_qkv_projection_dim() != 10240U ||
        config_->linear_value_dim() != 6144U ||
        config_->q_projection_dim() != 12288U || config_->q_dim() != 6144U ||
        config_->kv_dim() != 1024U || config_->head_dim != 256U ||
        config_->linear_value_head_dim != 128U) {
      fail(WeightBindErrorCode::kInvalidLayerSchedule, "model catalog",
           "exact Qwen3.6-27B architecture constants are unavailable");
      return false;
    }
    return true;
  }

  [[nodiscard]] const DeviceTensorView* find_view(
      const std::string& name) {
    const DeviceTensorView* const view =
        source_.lookup(source_.lookup_context, name);
    if (view == nullptr) {
      fail(WeightBindErrorCode::kMissingTensor, name,
           "required device tensor view is missing");
    }
    return view;
  }

  [[nodiscard]] bool shape_matches(
      const std::vector<std::uint64_t>& actual,
      const std::initializer_list<std::uint64_t> expected) const noexcept {
    if (actual.size() != expected.size()) {
      return false;
    }
    std::size_t index = 0U;
    for (const std::uint64_t dimension : expected) {
      if (actual[index] != dimension) {
        return false;
      }
      ++index;
    }
    return true;
  }

  [[nodiscard]] bool expected_bytes(
      const st::DType dtype,
      const std::initializer_list<std::uint64_t> shape,
      std::uint64_t& bytes) {
    std::uint64_t elements = 1U;
    for (const std::uint64_t dimension : shape) {
      if (dimension != 0U &&
          elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
        fail(WeightBindErrorCode::kArithmeticOverflow, "tensor ABI",
             "tensor element count overflows uint64_t");
        return false;
      }
      elements *= dimension;
    }
    const std::uint64_t bits = st::bit_width(dtype);
    if (bits == 0U ||
        elements > std::numeric_limits<std::uint64_t>::max() / bits) {
      fail(WeightBindErrorCode::kArithmeticOverflow, "tensor ABI",
           "tensor storage size overflows uint64_t");
      return false;
    }
    const std::uint64_t total_bits = elements * bits;
    if ((total_bits % 8U) != 0U) {
      fail(WeightBindErrorCode::kArithmeticOverflow, "tensor ABI",
           "tensor storage size is not byte-addressable");
      return false;
    }
    bytes = total_bits / 8U;
    return true;
  }

  [[nodiscard]] const DeviceTensorView* validate_view(
      const std::string& name, const DeviceTensorView* const view,
      const st::DType dtype,
      const std::initializer_list<std::uint64_t> shape) {
    if (view == nullptr || !ok()) {
      return nullptr;
    }
    if (view->dtype != dtype) {
      fail(WeightBindErrorCode::kDTypeMismatch, name,
           "device tensor dtype does not match the runtime ABI",
           std::string(st::to_string(dtype)),
           std::string(st::to_string(view->dtype)));
      return nullptr;
    }
    if (!shape_matches(view->shape, shape)) {
      fail(WeightBindErrorCode::kShapeMismatch, name,
           "device tensor shape does not match the runtime ABI",
           shape_string(shape), shape_string(view->shape));
      return nullptr;
    }
    std::uint64_t byte_size = 0U;
    if (!expected_bytes(dtype, shape, byte_size)) {
      return nullptr;
    }
    if (view->byte_size != byte_size) {
      fail(WeightBindErrorCode::kByteSizeMismatch, name,
           "device tensor byte size does not match dtype and shape",
           std::to_string(byte_size), std::to_string(view->byte_size));
      return nullptr;
    }
    if (view->device_data == nullptr) {
      fail(WeightBindErrorCode::kNullDevicePointer, name,
           "non-empty device tensor has a null pointer");
      return nullptr;
    }
    if ((view->arena_offset % kResidentTensorAlignment) != 0U ||
        (reinterpret_cast<std::uintptr_t>(view->device_data) %
         kResidentTensorAlignment) != 0U) {
      fail(WeightBindErrorCode::kMisalignedTensor, name,
           "device tensor is not 256-byte aligned", "256-byte alignment");
      return nullptr;
    }
    if (view->arena_offset > source_.arena_bytes ||
        view->byte_size > source_.arena_bytes - view->arena_offset) {
      fail(WeightBindErrorCode::kArenaRangeMismatch, name,
           "device tensor range escapes the resident arena");
      return nullptr;
    }
    const auto expected_pointer =
        arena_base_ + static_cast<std::uintptr_t>(view->arena_offset);
    const auto actual_pointer =
        reinterpret_cast<std::uintptr_t>(view->device_data);
    if (expected_pointer < arena_base_ || expected_pointer >= arena_end_ ||
        actual_pointer != expected_pointer) {
      fail(WeightBindErrorCode::kArenaRangeMismatch, name,
           "device pointer does not equal arena base plus arena_offset");
      return nullptr;
    }
    ++stats_.tensor_views;
    return view;
  }

  [[nodiscard]] const DeviceTensorView* bind_view(
      const std::string& name, const st::DType dtype,
      const std::initializer_list<std::uint64_t> shape) {
    return validate_view(name, find_view(name), dtype, shape);
  }

  [[nodiscard]] Bf16VectorWeight bind_bf16_vector(
      const std::string& name, const std::uint64_t element_count) {
    const DeviceTensorView* const view =
        bind_view(name, st::DType::kBf16, {element_count});
    return view == nullptr
               ? Bf16VectorWeight{}
               : Bf16VectorWeight{
                     static_cast<const std::uint16_t*>(view->device_data),
                     static_cast<std::size_t>(element_count)};
  }

  [[nodiscard]] Bf16Tensor3Weight bind_bf16_tensor3(
      const std::string& name, const std::uint64_t first,
      const std::uint64_t second, const std::uint64_t third) {
    const DeviceTensorView* const view =
        bind_view(name, st::DType::kBf16, {first, second, third});
    return view == nullptr
               ? Bf16Tensor3Weight{}
               : Bf16Tensor3Weight{
                     static_cast<const std::uint16_t*>(view->device_data),
                     {static_cast<std::size_t>(first),
                      static_cast<std::size_t>(second),
                      static_cast<std::size_t>(third)}};
  }

  [[nodiscard]] Bf16LinearWeight bind_bf16_matrix(
      const std::string& name, const std::uint64_t output_size,
      const std::uint64_t input_size) {
    const DeviceTensorView* const view =
        bind_view(name, st::DType::kBf16, {output_size, input_size});
    return view == nullptr
               ? Bf16LinearWeight{}
               : Bf16LinearWeight{
                     static_cast<const std::uint16_t*>(view->device_data),
                     static_cast<std::size_t>(output_size),
                     static_cast<std::size_t>(input_size)};
  }

  [[nodiscard]] bool read_scalar(const std::string& name,
                                 const DeviceTensorView* const view,
                                 float& value) {
    if (view == nullptr || !ok()) {
      return false;
    }
    const auto* const device_value =
        static_cast<const float*>(view->device_data);
    const int status =
        source_.scalar_read == nullptr
            ? read_cuda_scalar(device_value, &value)
            : source_.scalar_read(source_.scalar_read_context, device_value,
                                  &value);
    if (status != 0) {
      fail(WeightBindErrorCode::kCudaFailure, name,
           "failed to copy scalar from device during binding", {}, {},
           status);
      return false;
    }
    ++stats_.scalar_reads;
    if (!std::isfinite(value) || value < 0.0F) {
      fail(WeightBindErrorCode::kInvalidScalar, name,
           "quantization scale must be finite and non-negative",
           "finite value >= 0");
      return false;
    }
    return true;
  }

  [[nodiscard]] LinearWeight bind_linear(const std::string& module,
                                         const std::uint64_t output_size,
                                         const std::uint64_t input_size) {
    const std::string weight_name = module + ".weight";
    const DeviceTensorView* const unchecked = find_view(weight_name);
    if (unchecked == nullptr) {
      return Bf16LinearWeight{};
    }
    if (output_size > std::numeric_limits<std::size_t>::max() ||
        input_size > std::numeric_limits<std::size_t>::max()) {
      fail(WeightBindErrorCode::kArithmeticOverflow, module,
           "projection dimensions do not fit size_t");
      return Bf16LinearWeight{};
    }
    const auto rows = static_cast<std::size_t>(output_size);
    const auto columns = static_cast<std::size_t>(input_size);

    if (unchecked->dtype == st::DType::kBf16) {
      const DeviceTensorView* const weight = validate_view(
          weight_name, unchecked, st::DType::kBf16,
          {output_size, input_size});
      if (weight == nullptr) {
        return Bf16LinearWeight{};
      }
      ++stats_.bf16_projections;
      return Bf16LinearWeight{
          static_cast<const std::uint16_t*>(weight->device_data), rows,
          columns};
    }

    if (unchecked->dtype == st::DType::kF8E4M3) {
      const DeviceTensorView* const weight = validate_view(
          weight_name, unchecked, st::DType::kF8E4M3,
          {output_size, input_size});
      const std::string weight_scale_name = module + ".weight_scale";
      const std::string input_scale_name = module + ".input_scale";
      const DeviceTensorView* const weight_scale =
          bind_view(weight_scale_name, st::DType::kF32, {});
      const DeviceTensorView* const input_scale =
          bind_view(input_scale_name, st::DType::kF32, {});
      float host_weight_scale = 0.0F;
      float host_input_scale = 0.0F;
      if (weight == nullptr || weight_scale == nullptr ||
          input_scale == nullptr ||
          !read_scalar(weight_scale_name, weight_scale, host_weight_scale) ||
          !read_scalar(input_scale_name, input_scale, host_input_scale)) {
        return Bf16LinearWeight{};
      }
      ++stats_.fp8_projections;
      return Fp8LinearWeight{
          static_cast<const std::uint8_t*>(weight->device_data),
          static_cast<const float*>(weight_scale->device_data),
          static_cast<const float*>(input_scale->device_data),
          host_weight_scale,
          host_input_scale,
          rows,
          columns};
    }

    if (unchecked->dtype == st::DType::kU8) {
      if ((input_size % 16U) != 0U) {
        fail(WeightBindErrorCode::kShapeMismatch, module,
             "NVFP4 input size is not divisible by group size 16");
        return Bf16LinearWeight{};
      }
      const DeviceTensorView* const weight = validate_view(
          weight_name, unchecked, st::DType::kU8,
          {output_size, input_size / 2U});
      const std::string block_scale_name = module + ".weight_scale";
      const std::string weight_scale_2_name = module + ".weight_scale_2";
      const std::string input_scale_name = module + ".input_scale";
      const DeviceTensorView* const block_scale = bind_view(
          block_scale_name, st::DType::kF8E4M3,
          {output_size, input_size / 16U});
      const DeviceTensorView* const weight_scale_2 =
          bind_view(weight_scale_2_name, st::DType::kF32, {});
      const DeviceTensorView* const input_scale =
          bind_view(input_scale_name, st::DType::kF32, {});
      float host_weight_scale_2 = 0.0F;
      float host_input_scale = 0.0F;
      if (weight == nullptr || block_scale == nullptr ||
          weight_scale_2 == nullptr || input_scale == nullptr ||
          !read_scalar(weight_scale_2_name, weight_scale_2,
                       host_weight_scale_2) ||
          !read_scalar(input_scale_name, input_scale, host_input_scale)) {
        return Bf16LinearWeight{};
      }
      ++stats_.nvfp4_projections;
      return NvFp4LinearWeight{
          static_cast<const std::uint8_t*>(weight->device_data),
          static_cast<const std::uint8_t*>(block_scale->device_data),
          static_cast<const float*>(weight_scale_2->device_data),
          static_cast<const float*>(input_scale->device_data),
          host_weight_scale_2,
          host_input_scale,
          rows,
          columns};
    }

    fail(WeightBindErrorCode::kUnsupportedWeightDType, weight_name,
         "projection weight dtype is not BF16, F8_E4M3, or packed U8 NVFP4",
         "BF16 | F8_E4M3 | U8", std::string(st::to_string(unchecked->dtype)));
    return Bf16LinearWeight{};
  }

  const WeightBindingSource& source_;
  const model::ModelConfig* config_ = nullptr;
  std::uintptr_t arena_base_ = 0U;
  std::uintptr_t arena_end_ = 0U;
  WeightBindingStats stats_;
  WeightBindDiagnostic diagnostic_;
};

bool ModelWeights::attach_fp8_m1_output_projection_sidecars(
    const std::uint8_t* const arena, const std::size_t bytes) noexcept {
  constexpr std::uintptr_t kRequiredAlignment = 16U;
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena == nullptr ||
      bytes != kQwen36Fp8M1OutputProjectionAosoa4PreswizzledBytes ||
      (arena_address % kRequiredAlignment) != 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max() ||
      arena_address >
          std::numeric_limits<std::uintptr_t>::max() - bytes) {
    return false;
  }

  std::array<Fp8LinearWeight*, kQwen36DenseLayerCount> outputs{};
  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    LinearWeight* const output =
        attention_output_projection(layers_[layer_index]);
    Fp8LinearWeight* const fp8 =
        output == nullptr ? nullptr : std::get_if<Fp8LinearWeight>(output);
    if (!has_valid_fp8_payload(fp8) ||
        fp8->output_size != kFp8M1OutputProjectionRows ||
        fp8->input_size != kFp8M1OutputProjectionColumns) {
      return false;
    }
    outputs[layer_index] = fp8;
  }

  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    const std::uintptr_t layer_address =
        arena_address +
        layer_index *
            kFp8M1OutputProjectionAosoa4PreswizzledBytesPerLayer;
    outputs[layer_index]->m1_aosoa4_preswizzled_weight =
        reinterpret_cast<const std::uint8_t*>(layer_address);
  }
  return true;
}

bool ModelWeights::attach_fp8_prefill_qkv_register_feed_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const Fp8PrefillQkvRegisterFeedSidecarDescriptor* const descriptors,
    const std::size_t descriptor_count) noexcept {
  constexpr std::uintptr_t kRequiredAlignment = 16U;
  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();

  // The sole empty representation is an explicit detach. Do not make detach
  // depend on the current projection dtype/shape: clearing every extant FP8
  // QKV view is always safe and leaves a partially modified test fixture in a
  // canonical state too.
  if (descriptor_count == 0U) {
    if (arena != nullptr || arena_bytes != 0U || descriptors != nullptr) {
      return false;
    }
    for (DecoderLayerWeights& layer : layers_) {
      auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer.attention);
      if (linear == nullptr) {
        continue;
      }
      if (auto* const qkv =
              std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
          qkv != nullptr) {
        qkv->prefill_qkv_register_feed_sidecar = nullptr;
      }
    }
    return true;
  }

  if (descriptor_count != kQwen36LinearAttentionLayerCount ||
      arena == nullptr || descriptors == nullptr ||
      descriptor_count >
          std::numeric_limits<std::size_t>::max() /
              kFp8PrefillQkvRegisterFeedSidecarBytesPerLayer ||
      arena_bytes !=
          descriptor_count *
              kFp8PrefillQkvRegisterFeedSidecarBytesPerLayer) {
    return false;
  }

  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if ((arena_address % kRequiredAlignment) != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  // Enumerate the actual hybrid schedule first. This deliberately does not
  // derive layer indices from a 3:1 pattern or assume the 48 linear layers
  // occupy a contiguous prefix.
  std::array<Fp8LinearWeight*, kQwen36DenseLayerCount> qkv_by_layer{};
  std::size_t linear_layer_count = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    auto* const linear =
        std::get_if<LinearAttentionWeights>(&layers_[layer_index].attention);
    if (linear == nullptr) {
      continue;
    }
    ++linear_layer_count;
    auto* const qkv = std::get_if<Fp8LinearWeight>(&linear->in_proj_qkv);
    if (!has_valid_fp8_payload(qkv) ||
        qkv->output_size != kFp8PrefillQkvRegisterFeedRows ||
        qkv->input_size != kFp8PrefillQkvRegisterFeedColumns) {
      return false;
    }
    qkv_by_layer[layer_index] = qkv;
  }
  if (linear_layer_count != kQwen36LinearAttentionLayerCount) {
    return false;
  }

  std::array<bool, kQwen36DenseLayerCount> seen_layers{};
  std::array<Fp8LinearWeight*, kQwen36LinearAttentionLayerCount> targets{};
  std::array<const std::uint8_t*, kQwen36LinearAttentionLayerCount>
      validated_sidecars{};
  std::array<std::uintptr_t, kQwen36LinearAttentionLayerCount>
      range_begins{};
  std::array<std::uintptr_t, kQwen36LinearAttentionLayerCount> range_ends{};
  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    const Fp8PrefillQkvRegisterFeedSidecarDescriptor& descriptor =
        descriptors[index];
    if (descriptor.layer_index >= kQwen36DenseLayerCount ||
        seen_layers[descriptor.layer_index] || descriptor.sidecar == nullptr ||
        descriptor.bytes !=
            kFp8PrefillQkvRegisterFeedSidecarBytesPerLayer ||
        descriptor.output_size != kFp8PrefillQkvRegisterFeedRows ||
        descriptor.input_size != kFp8PrefillQkvRegisterFeedColumns ||
        qkv_by_layer[descriptor.layer_index] == nullptr) {
      return false;
    }

    const std::uintptr_t sidecar_address =
        reinterpret_cast<std::uintptr_t>(descriptor.sidecar);
    if ((sidecar_address % kRequiredAlignment) != 0U ||
        sidecar_address < arena_address || sidecar_address >= arena_end ||
        descriptor.bytes > kPointerMaximum ||
        sidecar_address > kPointerMaximum - descriptor.bytes) {
      return false;
    }
    const std::uintptr_t sidecar_end = sidecar_address + descriptor.bytes;
    if (sidecar_end > arena_end) {
      return false;
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (sidecar_address < range_ends[prior] &&
          range_begins[prior] < sidecar_end) {
        return false;
      }
    }

    seen_layers[descriptor.layer_index] = true;
    targets[index] = qkv_by_layer[descriptor.layer_index];
    validated_sidecars[index] = descriptor.sidecar;
    range_begins[index] = sidecar_address;
    range_ends[index] = sidecar_end;
  }

  // Validation is complete. Clear the full old set and install the complete
  // replacement without any remaining fallible operation.
  for (Fp8LinearWeight* const qkv : qkv_by_layer) {
    if (qkv != nullptr) {
      qkv->prefill_qkv_register_feed_sidecar = nullptr;
    }
  }
  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    targets[index]->prefill_qkv_register_feed_sidecar =
        validated_sidecars[index];
  }
  return true;
}

bool ModelWeights::attach_fp8_prefill_supermatrix_sidecars(
    const std::uint8_t* const arena,
    const std::size_t arena_bytes) noexcept {
  constexpr std::uintptr_t kRequiredAlignment = 16U;
  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();

  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      if (auto* const linear =
              std::get_if<LinearAttentionWeights>(&layer.attention)) {
        for (LinearWeight* const weight :
             {&linear->in_proj_qkv, &linear->in_proj_z,
              &linear->out_proj}) {
          if (auto* const fp8 = std::get_if<Fp8LinearWeight>(weight)) {
            fp8->prefill_supermatrix_sidecar = nullptr;
          }
        }
      } else if (auto* const full =
                     std::get_if<FullAttentionWeights>(&layer.attention)) {
        for (LinearWeight* const weight :
             {&full->q_proj, &full->k_proj, &full->v_proj,
              &full->o_proj}) {
          if (auto* const fp8 = std::get_if<Fp8LinearWeight>(weight)) {
            fp8->prefill_supermatrix_sidecar = nullptr;
          }
        }
      }
    }
  };

  if (arena == nullptr && arena_bytes == 0U) {
    clear_all();
    return true;
  }
  if (arena == nullptr ||
      arena_bytes != kQwen36Fp8PrefillSupermatrixSidecarBytes) {
    return false;
  }
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if ((arena_address % kRequiredAlignment) != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }

  std::array<Fp8LinearWeight*, kFp8PrefillSupermatrixProjectionCount>
      projections{};
  std::size_t projection_count = 0U;
  std::size_t validated_bytes = 0U;
  const auto append = [&](LinearWeight& binding, const std::size_t rows,
                          const std::size_t columns) noexcept {
    Fp8LinearWeight* const fp8 = std::get_if<Fp8LinearWeight>(&binding);
    if (projection_count >= projections.size() ||
        !has_valid_fp8_payload(fp8) || fp8->output_size != rows ||
        fp8->input_size != columns ||
        rows > std::numeric_limits<std::size_t>::max() / columns ||
        rows * columns > arena_bytes - validated_bytes) {
      return false;
    }
    projections[projection_count++] = fp8;
    validated_bytes += rows * columns;
    return true;
  };

  for (DecoderLayerWeights& layer : layers_) {
    if (auto* const linear =
            std::get_if<LinearAttentionWeights>(&layer.attention)) {
      if (!append(linear->in_proj_qkv, 10'240U, 5'120U) ||
          !append(linear->in_proj_z, 6'144U, 5'120U) ||
          !append(linear->out_proj, 5'120U, 6'144U)) {
        return false;
      }
    } else if (auto* const full =
                   std::get_if<FullAttentionWeights>(&layer.attention)) {
      if (!append(full->q_proj, 12'288U, 5'120U) ||
          !append(full->k_proj, 1'024U, 5'120U) ||
          !append(full->v_proj, 1'024U, 5'120U) ||
          !append(full->o_proj, 5'120U, 6'144U)) {
        return false;
      }
    } else {
      return false;
    }
  }
  if (projection_count != projections.size() ||
      validated_bytes != arena_bytes) {
    return false;
  }

  std::uintptr_t sidecar_address = arena_address;
  for (Fp8LinearWeight* const projection : projections) {
    projection->prefill_supermatrix_sidecar =
        reinterpret_cast<const std::uint8_t*>(sidecar_address);
    sidecar_address += projection->output_size * projection->input_size;
  }
  return sidecar_address == arena_address + arena_bytes;
}

bool ModelWeights::attach_fp8_marlin_prefill_sidecars(
    const std::uint8_t* const weight_arena,
    const std::size_t weight_arena_bytes,
    const std::uint16_t* const scale_arena,
    const std::size_t scale_arena_elements) noexcept {
  constexpr std::uintptr_t kRequiredAlignment = 16U;
  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      if (auto* const linear =
              std::get_if<LinearAttentionWeights>(&layer.attention)) {
        for (LinearWeight* const binding :
             {&linear->in_proj_qkv, &linear->in_proj_z,
              &linear->out_proj}) {
          if (auto* const fp8 = std::get_if<Fp8LinearWeight>(binding)) {
            fp8->prefill_marlin_weight = nullptr;
            fp8->prefill_marlin_scales = nullptr;
          }
        }
      } else if (auto* const full =
                     std::get_if<FullAttentionWeights>(&layer.attention)) {
        for (LinearWeight* const binding :
             {&full->q_proj, &full->k_proj, &full->v_proj,
              &full->o_proj}) {
          if (auto* const fp8 = std::get_if<Fp8LinearWeight>(binding)) {
            fp8->prefill_marlin_weight = nullptr;
            fp8->prefill_marlin_scales = nullptr;
          }
        }
      }
    }
  };

  if (weight_arena == nullptr && weight_arena_bytes == 0U &&
      scale_arena == nullptr && scale_arena_elements == 0U) {
    clear_all();
    return true;
  }
  if (weight_arena == nullptr || scale_arena == nullptr ||
      weight_arena_bytes != kQwen36Fp8PrefillSupermatrixSidecarBytes ||
      scale_arena_elements != kQwen36Fp8MarlinScaleElements) {
    return false;
  }
  const std::uintptr_t weight_address =
      reinterpret_cast<std::uintptr_t>(weight_arena);
  const std::uintptr_t scale_address =
      reinterpret_cast<std::uintptr_t>(scale_arena);
  constexpr std::size_t kScaleBytes = kQwen36Fp8MarlinScaleBytes;
  if ((weight_address % kRequiredAlignment) != 0U ||
      (scale_address % kRequiredAlignment) != 0U ||
      weight_arena_bytes > kPointerMaximum ||
      weight_address > kPointerMaximum - weight_arena_bytes ||
      kScaleBytes > kPointerMaximum ||
      scale_address > kPointerMaximum - kScaleBytes) {
    return false;
  }
  const std::uintptr_t weight_end = weight_address + weight_arena_bytes;
  const std::uintptr_t scale_end = scale_address + kScaleBytes;
  if (weight_address < scale_end && scale_address < weight_end) {
    return false;
  }

  std::array<Fp8LinearWeight*, kFp8PrefillSupermatrixProjectionCount>
      projections{};
  std::size_t projection_count = 0U;
  std::size_t validated_weight_bytes = 0U;
  std::size_t validated_scale_elements = 0U;
  const auto append = [&](LinearWeight& binding, const std::size_t rows,
                          const std::size_t columns) noexcept {
    Fp8LinearWeight* const fp8 = std::get_if<Fp8LinearWeight>(&binding);
    if (projection_count >= projections.size() ||
        !has_valid_fp8_payload(fp8) || fp8->output_size != rows ||
        fp8->input_size != columns ||
        rows > std::numeric_limits<std::size_t>::max() / columns ||
        rows * columns > weight_arena_bytes - validated_weight_bytes ||
        rows > scale_arena_elements - validated_scale_elements) {
      return false;
    }
    projections[projection_count++] = fp8;
    validated_weight_bytes += rows * columns;
    validated_scale_elements += rows;
    return true;
  };

  for (DecoderLayerWeights& layer : layers_) {
    if (auto* const linear =
            std::get_if<LinearAttentionWeights>(&layer.attention)) {
      if (!append(linear->in_proj_qkv, 10'240U, 5'120U) ||
          !append(linear->in_proj_z, 6'144U, 5'120U) ||
          !append(linear->out_proj, 5'120U, 6'144U)) {
        return false;
      }
    } else if (auto* const full =
                   std::get_if<FullAttentionWeights>(&layer.attention)) {
      if (!append(full->q_proj, 12'288U, 5'120U) ||
          !append(full->k_proj, 1'024U, 5'120U) ||
          !append(full->v_proj, 1'024U, 5'120U) ||
          !append(full->o_proj, 5'120U, 6'144U)) {
        return false;
      }
    } else {
      return false;
    }
  }
  if (projection_count != projections.size() ||
      validated_weight_bytes != weight_arena_bytes ||
      validated_scale_elements != scale_arena_elements) {
    return false;
  }

  clear_all();
  std::size_t weight_offset = 0U;
  std::size_t scale_offset = 0U;
  for (Fp8LinearWeight* const projection : projections) {
    projection->prefill_marlin_weight = weight_arena + weight_offset;
    projection->prefill_marlin_scales = scale_arena + scale_offset;
    weight_offset += projection->output_size * projection->input_size;
    scale_offset += projection->output_size;
  }
  return weight_offset == weight_arena_bytes &&
         scale_offset == scale_arena_elements;
}

bool ModelWeights::attach_nvfp4_down_scale6_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const NvFp4DownScale6SidecarDescriptor* const descriptors,
    const std::size_t descriptor_count) noexcept {
  constexpr std::uintptr_t kRequiredAlignment = 32U;
  constexpr unsigned int kMaximumScaleBase = 192U;
  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();

  // A canonical empty call is an explicit atomic detach. Nonempty calls use
  // an exact compact arena so every owned byte belongs to one descriptor.
  if (descriptor_count == 0U) {
    if (arena != nullptr || arena_bytes != 0U || descriptors != nullptr) {
      return false;
    }
    for (DecoderLayerWeights& layer : layers_) {
      if (NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
          down != nullptr) {
        down->down_scale6_sidecar = nullptr;
        down->down_scale6_base = 0U;
        down->down_consumer_order_weight = nullptr;
      }
    }
    return true;
  }

  if (arena == nullptr || descriptors == nullptr ||
      descriptor_count > kQwen36DenseLayerCount ||
      descriptor_count >
          std::numeric_limits<std::size_t>::max() /
              kNvFp4DownScale6SidecarBytesPerProjection ||
      arena_bytes !=
          descriptor_count * kNvFp4DownScale6SidecarBytesPerProjection) {
    return false;
  }

  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if ((arena_address % kRequiredAlignment) != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  std::array<bool, kQwen36DenseLayerCount> seen_layers{};
  std::array<NvFp4LinearWeight*, kQwen36DenseLayerCount> targets{};
  std::array<const std::uint8_t*, kQwen36DenseLayerCount>
      validated_sidecars{};
  std::array<unsigned int, kQwen36DenseLayerCount> validated_bases{};
  std::array<std::uintptr_t, kQwen36DenseLayerCount> range_begins{};
  std::array<std::uintptr_t, kQwen36DenseLayerCount> range_ends{};
  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    const NvFp4DownScale6SidecarDescriptor& descriptor = descriptors[index];
    if (descriptor.layer_index >= kQwen36DenseLayerCount ||
        seen_layers[descriptor.layer_index] || descriptor.sidecar == nullptr ||
        descriptor.bytes !=
            kNvFp4DownScale6SidecarBytesPerProjection ||
        descriptor.scale_base > kMaximumScaleBase ||
        descriptor.output_size != kNvFp4DownScale6Rows ||
        descriptor.input_size != kNvFp4DownScale6Columns) {
      return false;
    }

    NvFp4LinearWeight* const down =
        nvfp4_down_projection(layers_[descriptor.layer_index]);
    if (!has_valid_nvfp4_payload(down) ||
        down->output_size != kNvFp4DownScale6Rows ||
        down->input_size != kNvFp4DownScale6Columns) {
      return false;
    }

    const std::uintptr_t sidecar_address =
        reinterpret_cast<std::uintptr_t>(descriptor.sidecar);
    if ((sidecar_address % kRequiredAlignment) != 0U ||
        sidecar_address < arena_address || sidecar_address >= arena_end ||
        descriptor.bytes > kPointerMaximum ||
        sidecar_address > kPointerMaximum - descriptor.bytes) {
      return false;
    }
    const std::uintptr_t sidecar_end = sidecar_address + descriptor.bytes;
    if (sidecar_end > arena_end) {
      return false;
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (sidecar_address < range_ends[prior] &&
          range_begins[prior] < sidecar_end) {
        return false;
      }
    }

    seen_layers[descriptor.layer_index] = true;
    targets[index] = down;
    validated_sidecars[index] = descriptor.sidecar;
    validated_bases[index] = descriptor.scale_base;
    range_begins[index] = sidecar_address;
    range_ends[index] = sidecar_end;
  }

  // No operation below can fail. Clear the prior sparse set and install the
  // completely validated replacement as one observable state transition.
  for (DecoderLayerWeights& layer : layers_) {
    if (NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
        down != nullptr) {
      down->down_scale6_sidecar = nullptr;
      down->down_scale6_base = 0U;
      down->down_consumer_order_weight = nullptr;
    }
  }
  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    targets[index]->down_scale6_sidecar = validated_sidecars[index];
    targets[index]->down_scale6_base = validated_bases[index];
  }
  return true;
}

bool ModelWeights::attach_nvfp4_down_consumer_order_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const NvFp4DownConsumerOrderSidecarDescriptor* const descriptors,
    const std::size_t descriptor_count) noexcept {
  constexpr std::uintptr_t kRequiredAlignment = 16U;
  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();

  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      if (NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
          down != nullptr) {
        down->down_consumer_order_weight = nullptr;
      }
    }
  };

  if (descriptor_count == 0U) {
    if (arena != nullptr || arena_bytes != 0U || descriptors != nullptr) {
      return false;
    }
    clear_all();
    return true;
  }

  if (arena == nullptr || descriptors == nullptr ||
      descriptor_count > kQwen36DenseLayerCount ||
      descriptor_count >
          std::numeric_limits<std::size_t>::max() /
              kNvFp4DownConsumerOrderWeightBytesPerProjection ||
      arena_bytes !=
          descriptor_count *
              kNvFp4DownConsumerOrderWeightBytesPerProjection) {
    return false;
  }

  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if ((arena_address % kRequiredAlignment) != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  std::array<bool, kQwen36DenseLayerCount> seen_layers{};
  std::array<NvFp4LinearWeight*, kQwen36DenseLayerCount> targets{};
  std::array<const std::uint8_t*, kQwen36DenseLayerCount>
      validated_sidecars{};
  std::array<std::uintptr_t, kQwen36DenseLayerCount> range_begins{};
  std::array<std::uintptr_t, kQwen36DenseLayerCount> range_ends{};
  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    const NvFp4DownConsumerOrderSidecarDescriptor& descriptor =
        descriptors[index];
    if (descriptor.layer_index >= kQwen36DenseLayerCount ||
        seen_layers[descriptor.layer_index] || descriptor.sidecar == nullptr ||
        descriptor.bytes !=
            kNvFp4DownConsumerOrderWeightBytesPerProjection ||
        descriptor.output_size != kNvFp4DownScale6Rows ||
        descriptor.input_size != kNvFp4DownScale6Columns) {
      return false;
    }

    NvFp4LinearWeight* const down =
        nvfp4_down_projection(layers_[descriptor.layer_index]);
    if (!has_valid_nvfp4_payload(down) ||
        down->output_size != kNvFp4DownScale6Rows ||
        down->input_size != kNvFp4DownScale6Columns ||
        down->down_scale6_sidecar == nullptr) {
      return false;
    }

    const std::uintptr_t sidecar_address =
        reinterpret_cast<std::uintptr_t>(descriptor.sidecar);
    if ((sidecar_address % kRequiredAlignment) != 0U ||
        sidecar_address < arena_address || sidecar_address >= arena_end ||
        descriptor.bytes > kPointerMaximum ||
        sidecar_address > kPointerMaximum - descriptor.bytes) {
      return false;
    }
    const std::uintptr_t sidecar_end = sidecar_address + descriptor.bytes;
    if (sidecar_end > arena_end) {
      return false;
    }
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (sidecar_address < range_ends[prior] &&
          range_begins[prior] < sidecar_end) {
        return false;
      }
    }

    seen_layers[descriptor.layer_index] = true;
    targets[index] = down;
    validated_sidecars[index] = descriptor.sidecar;
    range_begins[index] = sidecar_address;
    range_ends[index] = sidecar_end;
  }

  clear_all();
  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    targets[index]->down_consumer_order_weight =
        validated_sidecars[index];
  }
  return true;
}

bool ModelWeights::attach_nvfp4_gate_up_coupled_feed_sidecars(
    const std::uint8_t* const arena,
    const std::size_t arena_bytes) noexcept {
  constexpr std::uintptr_t kRequiredAlignment = 16U;
  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();

  if (arena == nullptr || arena_bytes == 0U) {
    if (arena != nullptr || arena_bytes != 0U) {
      return false;
    }
    for (DecoderLayerWeights& layer : layers_) {
      if (NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
          gate != nullptr) {
        gate->decode_gate_up_coupled_feed_sidecar = nullptr;
      }
      if (NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
          up != nullptr) {
        up->decode_gate_up_coupled_feed_sidecar = nullptr;
      }
    }
    return true;
  }

  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_bytes != kQwen36NvFp4GateUpCoupledFeedBytes ||
      (arena_address % kRequiredAlignment) != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }

  std::array<NvFp4LinearWeight*, kQwen36DenseLayerCount> gates{};
  std::array<NvFp4LinearWeight*, kQwen36DenseLayerCount> ups{};
  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    NvFp4LinearWeight* const gate =
        nvfp4_gate_projection(layers_[layer_index]);
    NvFp4LinearWeight* const up =
        nvfp4_up_projection(layers_[layer_index]);
    if (!has_valid_nvfp4_payload(gate) || !has_valid_nvfp4_payload(up) ||
        gate->output_size != kNvFp4GateUpCoupledFeedRows ||
        gate->input_size != kNvFp4GateUpCoupledFeedColumns ||
        up->output_size != kNvFp4GateUpCoupledFeedRows ||
        up->input_size != kNvFp4GateUpCoupledFeedColumns) {
      return false;
    }
    gates[layer_index] = gate;
    ups[layer_index] = up;
  }

  // Validation above is complete; pointer publication cannot fail.
  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    const std::size_t layer_offset =
        layer_index * kNvFp4GateUpCoupledFeedBytesPerLayer;
    gates[layer_index]->decode_gate_up_coupled_feed_sidecar =
        arena + layer_offset;
    ups[layer_index]->decode_gate_up_coupled_feed_sidecar =
        arena + layer_offset + kNvFp4GateUpCoupledFeedBytesPerProjection;
  }
  return true;
}

bool ModelWeights::attach_nvfp4_marlin_prefill_sidecars(
    const NvFp4MarlinPrefillSidecarDescriptor* const descriptors,
    const std::size_t descriptor_count) noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      for (NvFp4LinearWeight* const projection :
           {nvfp4_gate_projection(layer), nvfp4_up_projection(layer),
            nvfp4_down_projection(layer)}) {
        if (projection != nullptr) {
          projection->prefill_marlin_gate_up_layout =
              NvFp4MarlinGateUpLayout::kUnbound;
          projection->prefill_marlin_weight = nullptr;
          projection->prefill_marlin_scales = nullptr;
          projection->prefill_marlin_global_scale = nullptr;
        }
      }
    }
  };

  if (descriptor_count == 0U) {
    if (descriptors != nullptr) {
      return false;
    }
    clear_all();
    return true;
  }
  if (descriptors == nullptr ||
      descriptor_count != kQwen36DenseLayerCount) {
    return false;
  }

  struct Validated {
    NvFp4LinearWeight* gate = nullptr;
    NvFp4LinearWeight* up = nullptr;
    NvFp4LinearWeight* down = nullptr;
    const NvFp4MarlinPrefillSidecarDescriptor* descriptor = nullptr;
  };
  std::array<bool, kQwen36DenseLayerCount> seen{};
  std::array<Validated, kQwen36DenseLayerCount> validated{};
  constexpr std::size_t kGateRows = 17'408U;
  constexpr std::size_t kHidden = 5'120U;
  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    const NvFp4MarlinPrefillSidecarDescriptor& descriptor = descriptors[index];
    if (descriptor.layer_index >= kQwen36DenseLayerCount ||
        seen[descriptor.layer_index] || descriptor.gate_up_weight == nullptr ||
        (descriptor.gate_up_layout !=
             NvFp4MarlinGateUpLayout::kCanonicalGateThenUp &&
         descriptor.gate_up_layout !=
             NvFp4MarlinGateUpLayout::kInterleavedGateUp) ||
        descriptor.gate_up_scales == nullptr ||
        descriptor.gate_up_global_scale == nullptr ||
        descriptor.down_weight == nullptr || descriptor.down_scales == nullptr ||
        descriptor.down_global_scale == nullptr ||
        reinterpret_cast<std::uintptr_t>(descriptor.gate_up_weight) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(descriptor.gate_up_scales) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(descriptor.gate_up_global_scale) %
                alignof(float) !=
            0U ||
        reinterpret_cast<std::uintptr_t>(descriptor.down_weight) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(descriptor.down_scales) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(descriptor.down_global_scale) %
                alignof(float) !=
            0U) {
      return false;
    }
    DecoderLayerWeights& layer = layers_[descriptor.layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    if (!has_valid_nvfp4_payload(gate) || !has_valid_nvfp4_payload(up) ||
        !has_valid_nvfp4_payload(down) || gate->output_size != kGateRows ||
        up->output_size != kGateRows || gate->input_size != kHidden ||
        up->input_size != kHidden || down->output_size != kHidden ||
        down->input_size != kGateRows ||
        !empty_nvfp4_marlin_p40_parity_view(
            gate->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            up->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            down->prefill_p40_vllm_marlin_parity)) {
      return false;
    }
    seen[descriptor.layer_index] = true;
    validated[index] = Validated{gate, up, down, &descriptor};
  }
  if (std::any_of(seen.begin(), seen.end(), [](const bool value) {
        return !value;
      })) {
    return false;
  }

  clear_all();
  for (const Validated& entry : validated) {
    const NvFp4MarlinPrefillSidecarDescriptor& descriptor = *entry.descriptor;
    for (NvFp4LinearWeight* const projection : {entry.gate, entry.up}) {
      projection->prefill_marlin_gate_up_layout =
          descriptor.gate_up_layout;
      projection->prefill_marlin_weight = descriptor.gate_up_weight;
      projection->prefill_marlin_scales = descriptor.gate_up_scales;
      projection->prefill_marlin_global_scale =
          descriptor.gate_up_global_scale;
    }
    entry.down->prefill_marlin_weight = descriptor.down_weight;
    entry.down->prefill_marlin_scales = descriptor.down_scales;
    entry.down->prefill_marlin_global_scale = descriptor.down_global_scale;
  }
  return true;
}

bool ModelWeights::attach_p40_packed_projection_sidecars(
    const P40PackedProjectionSidecarDescriptor* const descriptors,
    const std::size_t descriptor_count) noexcept {
  using kernels::Sm87P40PackedProjectionDeviceView;
  using kernels::Sm87P40PackedProjectionRole;
  constexpr std::size_t kArtifactKindsPerLayer = 4U;
  constexpr std::size_t kExpectedArtifacts =
      kernels::kSm87P40PackedProjectionArtifactCount;
  static_assert(kExpectedArtifacts ==
                kQwen36DenseLayerCount * kArtifactKindsPerLayer);

  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      for (NvFp4LinearWeight* const projection :
           {nvfp4_gate_projection(layer), nvfp4_up_projection(layer),
            nvfp4_down_projection(layer)}) {
        if (projection != nullptr) {
          projection->prefill_p40_packed_artifact = {};
        }
      }
      if (auto* const linear =
              std::get_if<LinearAttentionWeights>(&layer.attention)) {
        for (LinearWeight* const binding :
             {&linear->in_proj_qkv, &linear->in_proj_z,
              &linear->out_proj}) {
          if (auto* const fp8 = std::get_if<Fp8LinearWeight>(binding)) {
            fp8->prefill_p40_packed_artifact = {};
          }
        }
      } else if (auto* const full =
                     std::get_if<FullAttentionWeights>(&layer.attention)) {
        for (LinearWeight* const binding :
             {&full->q_proj, &full->k_proj, &full->v_proj,
              &full->o_proj}) {
          if (auto* const fp8 = std::get_if<Fp8LinearWeight>(binding)) {
            fp8->prefill_p40_packed_artifact = {};
          }
        }
      }
    }
  };

  if (descriptors == nullptr && descriptor_count == 0U) {
    clear_all();
    return true;
  }
  if (descriptors == nullptr || descriptor_count != kExpectedArtifacts) {
    return false;
  }

  using LayerViews =
      std::array<Sm87P40PackedProjectionDeviceView,
                 kArtifactKindsPerLayer>;
  std::array<LayerViews, kQwen36DenseLayerCount> validated{};
  std::array<std::array<bool, kArtifactKindsPerLayer>,
             kQwen36DenseLayerCount>
      seen{};
  std::array<std::uintptr_t, kExpectedArtifacts> range_begins{};
  std::array<std::uintptr_t, kExpectedArtifacts> range_ends{};
  std::array<std::uint64_t, kExpectedArtifacts> identities{};

  const auto role_slot = [](const Sm87P40PackedProjectionRole role)
      noexcept -> std::size_t {
    switch (role) {
      case Sm87P40PackedProjectionRole::kNvFp4GateUp:
        return 0U;
      case Sm87P40PackedProjectionRole::kNvFp4Down:
        return 1U;
      case Sm87P40PackedProjectionRole::kFp8LinearQkvZ:
      case Sm87P40PackedProjectionRole::kFp8FullQkv:
        return 2U;
      case Sm87P40PackedProjectionRole::kFp8AttentionOutput:
        return 3U;
      case Sm87P40PackedProjectionRole::kCount:
      case Sm87P40PackedProjectionRole::kInvalid:
        return kArtifactKindsPerLayer;
    }
    return kArtifactKindsPerLayer;
  };

  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    const P40PackedProjectionSidecarDescriptor& descriptor =
        descriptors[index];
    const Sm87P40PackedProjectionDeviceView& view = descriptor.view;
    const std::size_t slot = role_slot(view.role);
    const auto plan = kernels::sm87_p40_packed_projection_plan(view.role);
    if (descriptor.layer_index >= kQwen36DenseLayerCount ||
        slot >= kArtifactKindsPerLayer ||
        seen[descriptor.layer_index][slot] || !plan.valid() ||
        (slot == 2U &&
         ((kernels::sm87_p40_packed_is_full_layer(descriptor.layer_index) &&
           view.role != Sm87P40PackedProjectionRole::kFp8FullQkv) ||
          (!kernels::sm87_p40_packed_is_full_layer(descriptor.layer_index) &&
           view.role != Sm87P40PackedProjectionRole::kFp8LinearQkvZ))) ||
        view.payload == nullptr || view.payload_bytes != plan.payload_bytes ||
        view.artifact_identity == 0U || view.tactic != plan.tactic ||
        view.source_count != plan.source_count ||
        reinterpret_cast<std::uintptr_t>(view.payload) %
                kernels::kSm87P40PackedProjectionPayloadAlignment !=
            0U ||
        view.payload_bytes >
            std::numeric_limits<std::uintptr_t>::max() -
                reinterpret_cast<std::uintptr_t>(view.payload)) {
      return false;
    }
    for (std::size_t source = 0U; source < view.scalar_scales.size();
         ++source) {
      const float scale = view.scalar_scales[source];
      if ((source < view.source_count &&
           (!std::isfinite(scale) || scale < 0.0F)) ||
          (source >= view.source_count && scale != 0.0F)) {
        return false;
      }
    }
    const std::uintptr_t begin =
        reinterpret_cast<std::uintptr_t>(view.payload);
    range_begins[index] = begin;
    range_ends[index] = begin + view.payload_bytes;
    identities[index] = view.artifact_identity;
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (identities[prior] == identities[index] ||
          (range_begins[prior] < range_ends[index] &&
           range_begins[index] < range_ends[prior])) {
        return false;
      }
    }
    seen[descriptor.layer_index][slot] = true;
    validated[descriptor.layer_index][slot] = view;
  }

  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    if (std::any_of(seen[layer_index].begin(), seen[layer_index].end(),
                    [](const bool value) { return !value; })) {
      return false;
    }
    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    if (!has_valid_nvfp4_payload(gate) || !has_valid_nvfp4_payload(up) ||
        !has_valid_nvfp4_payload(down) ||
        gate->prefill_marlin_weight != nullptr ||
        up->prefill_marlin_weight != nullptr ||
        down->prefill_marlin_weight != nullptr ||
        !empty_nvfp4_marlin_p40_parity_view(
            gate->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            up->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            down->prefill_p40_vllm_marlin_parity)) {
      return false;
    }
    if (auto* const linear =
            std::get_if<LinearAttentionWeights>(&layer.attention)) {
      for (LinearWeight* const binding :
           {&linear->in_proj_qkv, &linear->in_proj_z,
            &linear->out_proj}) {
        const auto* const fp8 = std::get_if<Fp8LinearWeight>(binding);
        if (!has_valid_fp8_payload(fp8) ||
            fp8->prefill_supermatrix_sidecar != nullptr ||
            fp8->prefill_marlin_weight != nullptr) {
          return false;
        }
      }
    } else if (auto* const full =
                   std::get_if<FullAttentionWeights>(&layer.attention)) {
      for (LinearWeight* const binding :
           {&full->q_proj, &full->k_proj, &full->v_proj,
            &full->o_proj}) {
        const auto* const fp8 = std::get_if<Fp8LinearWeight>(binding);
        if (!has_valid_fp8_payload(fp8) ||
            fp8->prefill_supermatrix_sidecar != nullptr ||
            fp8->prefill_marlin_weight != nullptr) {
          return false;
        }
      }
    } else {
      return false;
    }
  }

  clear_all();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    gate->prefill_p40_packed_artifact = validated[layer_index][0U];
    up->prefill_p40_packed_artifact = validated[layer_index][0U];
    down->prefill_p40_packed_artifact = validated[layer_index][1U];
    if (auto* const linear =
            std::get_if<LinearAttentionWeights>(&layer.attention)) {
      std::get<Fp8LinearWeight>(linear->in_proj_qkv)
          .prefill_p40_packed_artifact = validated[layer_index][2U];
      std::get<Fp8LinearWeight>(linear->in_proj_z)
          .prefill_p40_packed_artifact = validated[layer_index][2U];
      std::get<Fp8LinearWeight>(linear->out_proj)
          .prefill_p40_packed_artifact = validated[layer_index][3U];
    } else {
      auto& full = std::get<FullAttentionWeights>(layer.attention);
      for (LinearWeight* const binding :
           {&full.q_proj, &full.k_proj, &full.v_proj}) {
        std::get<Fp8LinearWeight>(*binding).prefill_p40_packed_artifact =
            validated[layer_index][2U];
      }
      std::get<Fp8LinearWeight>(full.o_proj).prefill_p40_packed_artifact =
          validated[layer_index][3U];
    }
  }
  return true;
}

bool ModelWeights::attach_p40_packed_nvfp4_sidecars(
    const P40PackedProjectionSidecarDescriptor* const descriptors,
    const std::size_t descriptor_count) noexcept {
  using kernels::Sm87P40PackedProjectionDeviceView;
  using kernels::Sm87P40PackedProjectionRole;
  constexpr std::size_t kArtifactKindsPerLayer = 2U;
  constexpr std::size_t kExpectedArtifacts =
      2U * kernels::kSm87P40PackedProjectionLayerCount;
  static_assert(kExpectedArtifacts == 128U);

  const auto clear_nvfp4 = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      for (NvFp4LinearWeight* const projection :
           {nvfp4_gate_projection(layer), nvfp4_up_projection(layer),
            nvfp4_down_projection(layer)}) {
        if (projection != nullptr) {
          projection->prefill_p40_packed_artifact = {};
        }
      }
    }
  };

  if (descriptors == nullptr && descriptor_count == 0U) {
    clear_nvfp4();
    return true;
  }
  if (descriptors == nullptr || descriptor_count != kExpectedArtifacts) {
    return false;
  }

  using LayerViews =
      std::array<Sm87P40PackedProjectionDeviceView,
                 kArtifactKindsPerLayer>;
  std::array<LayerViews, kQwen36DenseLayerCount> validated{};
  std::array<std::array<bool, kArtifactKindsPerLayer>,
             kQwen36DenseLayerCount>
      seen{};
  std::array<std::uintptr_t, kExpectedArtifacts> range_begins{};
  std::array<std::uintptr_t, kExpectedArtifacts> range_ends{};
  std::array<std::uint64_t, kExpectedArtifacts> identities{};

  const auto role_slot = [](const Sm87P40PackedProjectionRole role)
      noexcept -> std::size_t {
    if (role == Sm87P40PackedProjectionRole::kNvFp4GateUp) {
      return 0U;
    }
    if (role == Sm87P40PackedProjectionRole::kNvFp4Down) {
      return 1U;
    }
    return kArtifactKindsPerLayer;
  };

  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    const P40PackedProjectionSidecarDescriptor& descriptor =
        descriptors[index];
    const Sm87P40PackedProjectionDeviceView& view = descriptor.view;
    const std::size_t slot = role_slot(view.role);
    const auto plan = kernels::sm87_p40_packed_projection_plan(view.role);
    if (descriptor.layer_index >= kQwen36DenseLayerCount ||
        slot >= kArtifactKindsPerLayer ||
        seen[descriptor.layer_index][slot] || !plan.valid() ||
        view.payload == nullptr || view.payload_bytes != plan.payload_bytes ||
        view.artifact_identity == 0U || view.tactic != plan.tactic ||
        view.source_count != plan.source_count ||
        reinterpret_cast<std::uintptr_t>(view.payload) %
                kernels::kSm87P40PackedProjectionPayloadAlignment !=
            0U ||
        view.payload_bytes >
            std::numeric_limits<std::uintptr_t>::max() -
                reinterpret_cast<std::uintptr_t>(view.payload)) {
      return false;
    }
    for (std::size_t source = 0U; source < view.scalar_scales.size();
         ++source) {
      const float scale = view.scalar_scales[source];
      if ((source < view.source_count &&
           (!std::isfinite(scale) || scale < 0.0F)) ||
          (source >= view.source_count && scale != 0.0F)) {
        return false;
      }
    }
    const std::uintptr_t begin =
        reinterpret_cast<std::uintptr_t>(view.payload);
    range_begins[index] = begin;
    range_ends[index] = begin + view.payload_bytes;
    identities[index] = view.artifact_identity;
    for (std::size_t prior = 0U; prior < index; ++prior) {
      if (identities[prior] == identities[index] ||
          (range_begins[prior] < range_ends[index] &&
           range_begins[index] < range_ends[prior])) {
        return false;
      }
    }
    seen[descriptor.layer_index][slot] = true;
    validated[descriptor.layer_index][slot] = view;
  }

  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    if (std::any_of(seen[layer_index].begin(), seen[layer_index].end(),
                    [](const bool value) { return !value; })) {
      return false;
    }
    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    if (!has_valid_nvfp4_payload(gate) || !has_valid_nvfp4_payload(up) ||
        !has_valid_nvfp4_payload(down) || gate->output_size != 17'408U ||
        gate->input_size != 5'120U || up->output_size != 17'408U ||
        up->input_size != 5'120U || down->output_size != 5'120U ||
        down->input_size != 17'408U ||
        gate->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        up->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        gate->prefill_marlin_weight != nullptr ||
        gate->prefill_marlin_scales != nullptr ||
        gate->prefill_marlin_global_scale != nullptr ||
        up->prefill_marlin_weight != nullptr ||
        up->prefill_marlin_scales != nullptr ||
        up->prefill_marlin_global_scale != nullptr ||
        down->prefill_marlin_weight != nullptr ||
        down->prefill_marlin_scales != nullptr ||
        down->prefill_marlin_global_scale != nullptr ||
        !empty_nvfp4_marlin_p40_parity_view(
            gate->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            up->prefill_p40_vllm_marlin_parity) ||
        !empty_nvfp4_marlin_p40_parity_view(
            down->prefill_p40_vllm_marlin_parity)) {
      return false;
    }

    const auto fp8_packed_views_empty = [](const LinearWeight& binding) {
      const auto* const fp8 = std::get_if<Fp8LinearWeight>(&binding);
      return fp8 != nullptr &&
             empty_p40_packed_artifact_view(
                 fp8->prefill_p40_packed_artifact);
    };
    if (auto* const linear =
            std::get_if<LinearAttentionWeights>(&layer.attention)) {
      if (!fp8_packed_views_empty(linear->in_proj_qkv) ||
          !fp8_packed_views_empty(linear->in_proj_z) ||
          !fp8_packed_views_empty(linear->out_proj)) {
        return false;
      }
    } else if (auto* const full =
                   std::get_if<FullAttentionWeights>(&layer.attention)) {
      if (!fp8_packed_views_empty(full->q_proj) ||
          !fp8_packed_views_empty(full->k_proj) ||
          !fp8_packed_views_empty(full->v_proj) ||
          !fp8_packed_views_empty(full->o_proj)) {
        return false;
      }
    } else {
      return false;
    }
  }

  clear_nvfp4();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    gate->prefill_p40_packed_artifact = validated[layer_index][0U];
    up->prefill_p40_packed_artifact = validated[layer_index][0U];
    down->prefill_p40_packed_artifact = validated[layer_index][1U];
  }
  return true;
}

bool ModelWeights::attach_nvfp4_marlin_p40_parity_sidecars(
    const NvFp4MarlinP40ParitySidecarDescriptor* const descriptors,
    const std::size_t descriptor_count) noexcept {
  constexpr std::size_t kRolesPerLayer = 2U;
  constexpr std::size_t kSourceRanges =
      kNvFp4MarlinP40ParitySourceCount * 3U;
  constexpr std::size_t kArtifactRanges =
      kNvFp4MarlinP40ParityArtifactCount * 3U;
  static_assert(kNvFp4MarlinP40ParityArtifactCount ==
                kQwen36DenseLayerCount * kRolesPerLayer);

  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      for (NvFp4LinearWeight* const projection :
           {nvfp4_gate_projection(layer), nvfp4_up_projection(layer),
            nvfp4_down_projection(layer)}) {
        if (projection != nullptr) {
          projection->prefill_p40_vllm_marlin_parity = {};
        }
      }
    }
  };

  if (descriptors == nullptr && descriptor_count == 0U) {
    clear_all();
    return true;
  }
  if (descriptors == nullptr ||
      descriptor_count != kNvFp4MarlinP40ParityArtifactCount) {
    return false;
  }

  struct PointerRange {
    std::uintptr_t begin = 0U;
    std::uintptr_t end = 0U;
  };
  const auto make_range = [](const void* const pointer,
                             const std::uint64_t bytes,
                             const std::uintptr_t alignment,
                             PointerRange* const range) noexcept {
    if (pointer == nullptr || range == nullptr || bytes == 0U ||
        bytes > std::numeric_limits<std::uintptr_t>::max()) {
      return false;
    }
    const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
    const std::uintptr_t size = static_cast<std::uintptr_t>(bytes);
    if (alignment == 0U || begin % alignment != 0U ||
        begin > std::numeric_limits<std::uintptr_t>::max() - size) {
      return false;
    }
    *range = PointerRange{begin, begin + size};
    return true;
  };
  const auto overlaps = [](const PointerRange& first,
                           const PointerRange& second) noexcept {
    return first.begin < second.end && second.begin < first.end;
  };
  const auto digest_present = [](const NvFp4MarlinP40ParityDigest& digest)
      noexcept {
    return std::any_of(digest.begin(), digest.end(),
                       [](const std::uint8_t byte) { return byte != 0U; });
  };
  const auto float_bits = [](const float value) noexcept {
    std::uint32_t bits = 0U;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
  };
  const auto source_manifest_empty =
      [&digest_present](const NvFp4MarlinP40ParitySourceManifest& source)
      noexcept {
        return source.role == NvFp4MarlinP40ParitySourceRole::kInvalid &&
               source.tensor_identity == 0U &&
               !digest_present(source.weight_digest) &&
               !digest_present(source.scale_digest) &&
               source.global_scale_bits == 0U;
      };

  struct LayerTargets {
    NvFp4LinearWeight* gate = nullptr;
    NvFp4LinearWeight* up = nullptr;
    NvFp4LinearWeight* down = nullptr;
  };
  std::array<LayerTargets, kQwen36DenseLayerCount> targets{};
  std::array<PointerRange, kSourceRanges> source_ranges{};
  std::size_t source_range_count = 0U;
  const auto append_source_ranges =
      [&make_range, &overlaps, &source_ranges,
       &source_range_count](const NvFp4LinearWeight& projection) noexcept {
        const std::uint64_t elements =
            static_cast<std::uint64_t>(projection.output_size) *
            projection.input_size;
        const std::array<std::pair<const void*, std::uint64_t>, 3U> payloads{{
            {projection.packed_weight, elements / 2U},
            {projection.block_scale, elements / 16U},
            {projection.weight_scale_2_device, sizeof(float)},
        }};
        for (std::size_t index = 0U; index < payloads.size(); ++index) {
          PointerRange range{};
          const std::uintptr_t alignment =
              index == 2U ? alignof(float) : 16U;
          if (source_range_count >= source_ranges.size() ||
              !make_range(payloads[index].first, payloads[index].second,
                          alignment, &range)) {
            return false;
          }
          for (std::size_t prior = 0U; prior < source_range_count; ++prior) {
            if (overlaps(source_ranges[prior], range)) {
              return false;
            }
          }
          source_ranges[source_range_count++] = range;
        }
        return true;
      };

  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    if (!has_valid_nvfp4_payload(gate) || !has_valid_nvfp4_payload(up) ||
        !has_valid_nvfp4_payload(down) ||
        gate->output_size != kNvFp4MarlinP40ParityIntermediate ||
        gate->input_size != kNvFp4MarlinP40ParityHidden ||
        up->output_size != kNvFp4MarlinP40ParityIntermediate ||
        up->input_size != kNvFp4MarlinP40ParityHidden ||
        down->output_size != kNvFp4MarlinP40ParityHidden ||
        down->input_size != kNvFp4MarlinP40ParityIntermediate ||
        float_bits(gate->weight_scale_2) !=
            float_bits(up->weight_scale_2) ||
        gate->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        up->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        down->prefill_marlin_gate_up_layout !=
            NvFp4MarlinGateUpLayout::kUnbound ||
        gate->prefill_marlin_weight != nullptr ||
        gate->prefill_marlin_scales != nullptr ||
        gate->prefill_marlin_global_scale != nullptr ||
        up->prefill_marlin_weight != nullptr ||
        up->prefill_marlin_scales != nullptr ||
        up->prefill_marlin_global_scale != nullptr ||
        down->prefill_marlin_weight != nullptr ||
        down->prefill_marlin_scales != nullptr ||
        down->prefill_marlin_global_scale != nullptr ||
        !empty_p40_packed_artifact_view(
            gate->prefill_p40_packed_artifact) ||
        !empty_p40_packed_artifact_view(up->prefill_p40_packed_artifact) ||
        !empty_p40_packed_artifact_view(
            down->prefill_p40_packed_artifact) ||
        !append_source_ranges(*gate) || !append_source_ranges(*up) ||
        !append_source_ranges(*down)) {
      return false;
    }
    targets[layer_index] = LayerTargets{gate, up, down};
  }
  if (source_range_count != source_ranges.size()) {
    return false;
  }

  using LayerViews =
      std::array<NvFp4MarlinP40ParityDeviceView, kRolesPerLayer>;
  std::array<LayerViews, kQwen36DenseLayerCount> validated{};
  std::array<std::array<bool, kRolesPerLayer>, kQwen36DenseLayerCount> seen{};
  std::array<std::uint64_t, kNvFp4MarlinP40ParityArtifactCount>
      artifact_identities{};
  std::size_t artifact_identity_count = 0U;
  std::array<std::uint64_t, kNvFp4MarlinP40ParitySourceCount>
      tensor_identities{};
  std::size_t tensor_identity_count = 0U;
  std::array<PointerRange, kArtifactRanges> artifact_ranges{};
  std::size_t artifact_range_count = 0U;

  for (std::size_t index = 0U; index < descriptor_count; ++index) {
    const NvFp4MarlinP40ParitySidecarDescriptor& descriptor =
        descriptors[index];
    const NvFp4MarlinP40ParityDeviceView& view = descriptor.view;
    const NvFp4MarlinP40ParityArtifactManifest& manifest = view.manifest;
    const bool gate_up =
        manifest.role == NvFp4MarlinP40ParityRole::kGateUp;
    const bool down = manifest.role == NvFp4MarlinP40ParityRole::kDown;
    const std::size_t slot = gate_up ? 0U : down ? 1U : kRolesPerLayer;
    if (descriptor.layer_index >= kQwen36DenseLayerCount ||
        manifest.layer_index != descriptor.layer_index ||
        slot >= kRolesPerLayer || seen[descriptor.layer_index][slot] ||
        manifest.version != kNvFp4MarlinP40ParityManifestVersion ||
        manifest.artifact_identity == 0U ||
        !digest_present(manifest.transformation_digest) ||
        (gate_up &&
         (manifest.layout !=
              NvFp4MarlinP40ParityLayout::kCanonicalGateThenUp ||
          manifest.output_features != kNvFp4MarlinP40ParityGateUpOutput ||
          manifest.input_features != kNvFp4MarlinP40ParityHidden ||
          manifest.weight_bytes !=
              kNvFp4MarlinP40ParityGateUpWeightBytes ||
          manifest.scale_bytes !=
              kNvFp4MarlinP40ParityGateUpScaleBytes ||
          manifest.source_count != 2U)) ||
        (down &&
         (manifest.layout != NvFp4MarlinP40ParityLayout::kCanonicalDown ||
          manifest.output_features != kNvFp4MarlinP40ParityHidden ||
          manifest.input_features != kNvFp4MarlinP40ParityIntermediate ||
          manifest.weight_bytes != kNvFp4MarlinP40ParityDownWeightBytes ||
          manifest.scale_bytes != kNvFp4MarlinP40ParityDownScaleBytes ||
          manifest.source_count != 1U ||
          !source_manifest_empty(manifest.sources[1U])))) {
      return false;
    }

    for (std::size_t prior = 0U; prior < artifact_identity_count; ++prior) {
      if (artifact_identities[prior] == manifest.artifact_identity) {
        return false;
      }
    }
    artifact_identities[artifact_identity_count++] =
        manifest.artifact_identity;

    const std::array<PointerRange, 3U> candidate_ranges = [&]() noexcept {
      std::array<PointerRange, 3U> ranges{};
      if (!make_range(view.weight, manifest.weight_bytes, 16U,
                      &ranges[0U]) ||
          !make_range(view.scales, manifest.scale_bytes, 16U,
                      &ranges[1U]) ||
          !make_range(view.global_scale, sizeof(float), alignof(float),
                      &ranges[2U])) {
        return std::array<PointerRange, 3U>{};
      }
      return ranges;
    }();
    if (candidate_ranges[0U].end == 0U ||
        candidate_ranges[1U].end == 0U ||
        candidate_ranges[2U].end == 0U) {
      return false;
    }
    for (const PointerRange& range : candidate_ranges) {
      for (std::size_t source = 0U; source < source_range_count; ++source) {
        if (overlaps(range, source_ranges[source])) {
          return false;
        }
      }
      for (std::size_t prior = 0U; prior < artifact_range_count; ++prior) {
        if (overlaps(range, artifact_ranges[prior])) {
          return false;
        }
      }
      artifact_ranges[artifact_range_count++] = range;
    }

    const LayerTargets& layer = targets[descriptor.layer_index];
    const std::array<const NvFp4LinearWeight*, 2U> expected_sources{
        gate_up ? layer.gate : layer.down, gate_up ? layer.up : nullptr};
    const std::array<NvFp4MarlinP40ParitySourceRole, 2U> expected_roles{
        gate_up ? NvFp4MarlinP40ParitySourceRole::kGate
                : NvFp4MarlinP40ParitySourceRole::kDown,
        gate_up ? NvFp4MarlinP40ParitySourceRole::kUp
                : NvFp4MarlinP40ParitySourceRole::kInvalid};
    for (std::size_t source_index = 0U;
         source_index < manifest.source_count; ++source_index) {
      const NvFp4MarlinP40ParitySourceManifest& source =
          manifest.sources[source_index];
      const NvFp4LinearWeight* const expected =
          expected_sources[source_index];
      if (expected == nullptr || source.role != expected_roles[source_index] ||
          source.tensor_identity == 0U ||
          !digest_present(source.weight_digest) ||
          !digest_present(source.scale_digest) ||
          source.global_scale_bits != float_bits(expected->weight_scale_2)) {
        return false;
      }
      for (std::size_t prior = 0U; prior < tensor_identity_count; ++prior) {
        if (tensor_identities[prior] == source.tensor_identity) {
          return false;
        }
      }
      tensor_identities[tensor_identity_count++] = source.tensor_identity;
    }

    seen[descriptor.layer_index][slot] = true;
    validated[descriptor.layer_index][slot] = view;
  }

  if (artifact_identity_count != artifact_identities.size() ||
      tensor_identity_count != tensor_identities.size() ||
      artifact_range_count != artifact_ranges.size()) {
    return false;
  }
  for (const auto& layer_seen : seen) {
    if (std::any_of(layer_seen.begin(), layer_seen.end(),
                    [](const bool value) { return !value; })) {
      return false;
    }
  }

  clear_all();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    targets[layer_index].gate->prefill_p40_vllm_marlin_parity =
        validated[layer_index][0U];
    targets[layer_index].up->prefill_p40_vllm_marlin_parity =
        validated[layer_index][0U];
    targets[layer_index].down->prefill_p40_vllm_marlin_parity =
        validated[layer_index][1U];
  }
  return true;
}

LinearWeightKind linear_weight_kind(const LinearWeight& weight) noexcept {
  if (std::holds_alternative<Bf16LinearWeight>(weight)) {
    return LinearWeightKind::kBf16;
  }
  if (std::holds_alternative<Fp8LinearWeight>(weight)) {
    return LinearWeightKind::kFp8;
  }
  return LinearWeightKind::kNvFp4;
}

std::size_t linear_output_size(const LinearWeight& weight) noexcept {
  return std::visit(
      [](const auto& selected) noexcept { return output_size_of(selected); },
      weight);
}

std::size_t linear_input_size(const LinearWeight& weight) noexcept {
  return std::visit(
      [](const auto& selected) noexcept { return input_size_of(selected); },
      weight);
}

bool supports_bf16_projection_pair(
    const ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight) noexcept {
  constexpr std::size_t kPairRows = 48U;
  constexpr std::size_t kPairColumns = 5120U;
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      first_weight.valueless_by_exception() ||
      second_weight.valueless_by_exception()) {
    return false;
  }
  const auto* const first =
      std::get_if<Bf16LinearWeight>(&first_weight);
  const auto* const second =
      std::get_if<Bf16LinearWeight>(&second_weight);
  return first != nullptr && second != nullptr && first->weight != nullptr &&
         second->weight != nullptr && first->output_size == kPairRows &&
         second->output_size == kPairRows &&
         first->input_size == kPairColumns &&
         second->input_size == kPairColumns;
}

bool supports_fp8_projection_pair(
    const ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight) noexcept {
  constexpr std::size_t kPairRows = 1'024U;
  constexpr std::size_t kPairColumns = 5'120U;
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      first_weight.valueless_by_exception() ||
      second_weight.valueless_by_exception()) {
    return false;
  }
  const auto* const first = std::get_if<Fp8LinearWeight>(&first_weight);
  const auto* const second = std::get_if<Fp8LinearWeight>(&second_weight);
  return has_valid_fp8_payload(first) && has_valid_fp8_payload(second) &&
         first->output_size == kPairRows &&
         second->output_size == kPairRows &&
         first->input_size == kPairColumns &&
         second->input_size == kPairColumns;
}

bool supports_fp8_qkv_z_projection_pair(
    const ProjectionBackend backend, const LinearWeight& qkv_weight,
    const LinearWeight& z_weight) noexcept {
  constexpr std::size_t kQkvRows = 10'240U;
  constexpr std::size_t kZRows = 6'144U;
  constexpr std::size_t kColumns = 5'120U;
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      qkv_weight.valueless_by_exception() ||
      z_weight.valueless_by_exception()) {
    return false;
  }
  const auto* const qkv = std::get_if<Fp8LinearWeight>(&qkv_weight);
  const auto* const z = std::get_if<Fp8LinearWeight>(&z_weight);
  return has_valid_fp8_payload(qkv) && has_valid_fp8_payload(z) &&
         qkv->output_size == kQkvRows && z->output_size == kZRows &&
         qkv->input_size == kColumns && z->input_size == kColumns;
}

bool supports_linear_attention_qkv_z_ab_projection_fusion(
    const ProjectionBackend backend, const LinearWeight& qkv_weight,
    const LinearWeight& z_weight, const LinearWeight& a_weight,
    const LinearWeight& b_weight) noexcept {
  return supports_fp8_qkv_z_projection_pair(backend, qkv_weight, z_weight) &&
         supports_bf16_projection_pair(backend, a_weight, b_weight);
}

bool supports_fp8_q_kv_projection_fusion(
    const ProjectionBackend backend, const LinearWeight& q_weight,
    const LinearWeight& key_weight,
    const LinearWeight& value_weight) noexcept {
  constexpr std::size_t kQRows = 12'288U;
  constexpr std::size_t kKvRows = 1'024U;
  constexpr std::size_t kColumns = 5'120U;
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      q_weight.valueless_by_exception() ||
      key_weight.valueless_by_exception() ||
      value_weight.valueless_by_exception()) {
    return false;
  }
  const auto* const q = std::get_if<Fp8LinearWeight>(&q_weight);
  const auto* const key = std::get_if<Fp8LinearWeight>(&key_weight);
  const auto* const value = std::get_if<Fp8LinearWeight>(&value_weight);
  return has_valid_fp8_payload(q) && has_valid_fp8_payload(key) &&
         has_valid_fp8_payload(value) && q->output_size == kQRows &&
         key->output_size == kKvRows && value->output_size == kKvRows &&
         q->input_size == kColumns && key->input_size == kColumns &&
         value->input_size == kColumns;
}

bool supports_nvfp4_gate_up_silu_fusion(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight) noexcept {
  constexpr std::size_t kRows = 17'408U;
  constexpr std::size_t kColumns = 5'120U;
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      gate_weight.valueless_by_exception() ||
      up_weight.valueless_by_exception()) {
    return false;
  }
  const auto* const gate = std::get_if<NvFp4LinearWeight>(&gate_weight);
  const auto* const up = std::get_if<NvFp4LinearWeight>(&up_weight);
  return has_valid_nvfp4_payload(gate) && has_valid_nvfp4_payload(up) &&
         gate->output_size == kRows && up->output_size == kRows &&
         gate->input_size == kColumns && up->input_size == kColumns;
}

bool supports_nvfp4_down_residual_norm_fusion(
    const ProjectionBackend backend,
    const LinearWeight& down_weight) noexcept {
  constexpr std::size_t kRows = 5'120U;
  constexpr std::size_t kColumns = 17'408U;
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      down_weight.valueless_by_exception()) {
    return false;
  }
  const auto* const down = std::get_if<NvFp4LinearWeight>(&down_weight);
  return has_valid_nvfp4_payload(down) && down->output_size == kRows &&
         down->input_size == kColumns;
}

WeightBindResult bind_qwen36_27b_weights(
    const WeightBindingSource& source) {
  try {
    return ModelWeightBinder(source).run();
  } catch (const std::bad_alloc&) {
    WeightBindResult result;
    result.diagnostic.code = WeightBindErrorCode::kAllocationFailure;
    result.diagnostic.message = "allocation failed while constructing bind diagnostics";
    return result;
  }
}

WeightBindResult bind_qwen36_27b_weights(const ResidentWeights& resident) {
  if (!resident || resident.arena_data() == nullptr ||
      resident.size_bytes() != kPinnedQwen36_27BArenaBytes ||
      resident.tensor_count() !=
          model::weights::kPinnedQwen36_27BTextTensorCount) {
    WeightBindResult result;
    result.diagnostic.code = WeightBindErrorCode::kInvalidPinnedArena;
    result.diagnostic.tensor = "ResidentWeights arena";
    result.diagnostic.message =
        "production binding requires the exact pinned Qwen3.6-27B arena and "
        "text-view count";
    result.diagnostic.expected =
        std::to_string(kPinnedQwen36_27BArenaBytes) + " bytes / " +
        std::to_string(model::weights::kPinnedQwen36_27BTextTensorCount) +
        " views";
    result.diagnostic.actual = std::to_string(resident.size_bytes()) +
                               " bytes / " +
                               std::to_string(resident.tensor_count()) +
                               " views";
    return result;
  }
  WeightBindingSource source;
  source.lookup_context = &resident;
  source.lookup = &resident_lookup;
  source.arena_data = resident.arena_data();
  source.arena_bytes = resident.size_bytes();
  WeightBindResult result = bind_qwen36_27b_weights(source);
  const std::size_t consumed_views =
      result ? result.value->stats().tensor_views : 0U;
  if (result && consumed_views != resident.tensor_count()) {
    result.value.reset();
    result.diagnostic.code = WeightBindErrorCode::kInvalidPinnedArena;
    result.diagnostic.tensor = "ResidentWeights view table";
    result.diagnostic.message =
        "typed binding did not consume every pinned text tensor view";
    result.diagnostic.expected = std::to_string(resident.tensor_count());
    result.diagnostic.actual = std::to_string(consumed_views);
  }
  return result;
}

std::string_view to_string(const WeightBindErrorCode code) noexcept {
  switch (code) {
    case WeightBindErrorCode::kNone:
      return "none";
    case WeightBindErrorCode::kInvalidSource:
      return "invalid source";
    case WeightBindErrorCode::kInvalidPinnedArena:
      return "invalid pinned arena";
    case WeightBindErrorCode::kMissingTensor:
      return "missing tensor";
    case WeightBindErrorCode::kUnsupportedWeightDType:
      return "unsupported weight dtype";
    case WeightBindErrorCode::kDTypeMismatch:
      return "dtype mismatch";
    case WeightBindErrorCode::kShapeMismatch:
      return "shape mismatch";
    case WeightBindErrorCode::kByteSizeMismatch:
      return "byte-size mismatch";
    case WeightBindErrorCode::kNullDevicePointer:
      return "null device pointer";
    case WeightBindErrorCode::kMisalignedTensor:
      return "misaligned tensor";
    case WeightBindErrorCode::kArenaRangeMismatch:
      return "arena range mismatch";
    case WeightBindErrorCode::kInvalidScalar:
      return "invalid scalar";
    case WeightBindErrorCode::kCudaFailure:
      return "CUDA failure";
    case WeightBindErrorCode::kInvalidLayerSchedule:
      return "invalid layer schedule";
    case WeightBindErrorCode::kArithmeticOverflow:
      return "arithmetic overflow";
    case WeightBindErrorCode::kAllocationFailure:
      return "allocation failure";
  }
  return "unknown";
}

}  // namespace q3x::runtime
