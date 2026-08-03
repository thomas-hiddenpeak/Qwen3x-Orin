#include "q3x/runtime/model_weights.h"

#include "q3x/core/sha256.h"
#include "q3x/io/safetensors.h"
#include "q3x/runtime/prefill_a4_sidecar_converter.h"
#include "q3x/runtime/prefill_attention_factorized_lane_overlay.h"
#include "q3x/runtime/prefill_attention_o_k512_overlay.h"
#include "q3x/runtime/prefill_mlp_k512_fragment_native_overlay.h"
#include "q3x/runtime/prefill_mlp_k512_overlay.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_converter.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_r4_publication.h"
#include "q3x/runtime/prefill_mlp_factorized_lane_r4_candidate_converter.h"
#include "q3x/runtime/prefill_quantized_contract.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

[[nodiscard]] bool attention_r1_lower_sha256(
    const std::string_view value) noexcept {
  return value.size() == 64U &&
         std::all_of(value.begin(), value.end(), [](const char character) {
           return (character >= '0' && character <= '9') ||
                  (character >= 'a' && character <= 'f');
         });
}

[[nodiscard]] std::string_view attention_r1_family_name(
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      return "linear_qkv";
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      return "linear_z";
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      return "linear_o";
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      return "full_q";
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      return "full_k";
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      return "full_v";
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      return "full_o";
  }
  return "invalid";
}

[[nodiscard]] std::string attention_r1_expected_source_module(
    const std::uint32_t layer,
    const PrefillAttentionFactorizedLaneProjectionFamily family) {
  std::string_view suffix;
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      suffix = "linear_attn.in_proj_qkv";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      suffix = "linear_attn.in_proj_z";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      suffix = "linear_attn.out_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      suffix = "self_attn.q_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      suffix = "self_attn.k_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      suffix = "self_attn.v_proj";
      break;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      suffix = "self_attn.o_proj";
      break;
    default:
      return {};
  }
  return "model.language_model.layers." + std::to_string(layer) + "." +
         std::string(suffix);
}

[[nodiscard]] const PrefillA4FactorizedLaneProjectionLayoutPlan*
attention_r1_projection_plan(
    const PrefillAttentionFactorizedLaneOverlayLayoutPlan& plan,
    const PrefillAttentionFactorizedLaneProjectionFamily family) noexcept {
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      return &plan.linear_qkv;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      return &plan.linear_z;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
      return &plan.linear_o;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      return &plan.full_q;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
      return &plan.full_k;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      return &plan.full_v;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      return &plan.full_o;
  }
  return nullptr;
}

[[nodiscard]] bool attention_r1_expected_shape(
    const PrefillAttentionFactorizedLaneProjectionFamily family,
    std::uint64_t& output_size, std::uint64_t& input_size) noexcept {
  input_size = kPrefillAttentionFactorizedLaneHiddenSize;
  switch (family) {
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv:
      output_size = kPrefillAttentionFactorizedLaneLinearQkvOutputSize;
      return true;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ:
      output_size = kPrefillAttentionFactorizedLaneLinearZOutputSize;
      return true;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ:
      output_size = kPrefillAttentionFactorizedLaneFullQOutputSize;
      return true;
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullK:
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullV:
      output_size = kPrefillAttentionFactorizedLaneFullKvOutputSize;
      return true;
    case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO:
    case PrefillAttentionFactorizedLaneProjectionFamily::kFullO:
      output_size = kPrefillAttentionFactorizedLaneHiddenSize;
      input_size = kPrefillAttentionFactorizedLaneAttentionOutputSize;
      return true;
  }
  return false;
}

[[nodiscard]] PrefillAttentionFactorizedLaneProjectionFamily
attention_r1_expected_family(const std::uint32_t layer,
                             const std::uint32_t position) noexcept {
  if (prefill_attention_factorized_lane_is_full_layer(layer)) {
    constexpr std::array<PrefillAttentionFactorizedLaneProjectionFamily, 4U>
        families = {
            PrefillAttentionFactorizedLaneProjectionFamily::kFullQ,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullK,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullV,
            PrefillAttentionFactorizedLaneProjectionFamily::kFullO};
    return position < families.size()
               ? families[position]
               : static_cast<
                     PrefillAttentionFactorizedLaneProjectionFamily>(0xffU);
  }
  constexpr std::array<PrefillAttentionFactorizedLaneProjectionFamily, 3U>
      families = {
          PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv,
          PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ,
          PrefillAttentionFactorizedLaneProjectionFamily::kLinearO};
  return position < families.size()
             ? families[position]
             : static_cast<PrefillAttentionFactorizedLaneProjectionFamily>(
                   0xffU);
}

[[nodiscard]] std::string attention_r1_manifest_sha256(
    const PrefillAttentionFactorizedLaneOverlayManifestBinding& manifest) {
  constexpr std::string_view kSchema =
      "q3x.prefill.attention-factorized-r1.overlay-manifest";
  std::ostringstream output;
  output << "schema=" << kSchema << "\nversion=" << manifest.version_major
         << '.' << manifest.version_minor << "\nlayout="
         << manifest.physical_layout << "\ncheckpoint="
         << manifest.source_checkpoint_id << "\nconfig="
         << manifest.source_config_sha256 << "\nindex="
         << manifest.source_index_sha256 << "\nlane="
         << manifest.lane_count << "\nbase="
         << manifest.required_base_k256.physical_layout << ':'
         << manifest.required_base_k256.packed_k_group_size << ':'
         << manifest.required_base_k256.scale_group_size << ':'
         << manifest.required_base_k256.manifest_sha256 << ':'
         << manifest.required_base_k256.policy_sha256 << ':'
         << manifest.required_base_k256.payload_sha256 << ':'
         << manifest.required_base_k256.receipt_sha256 << "\npayload="
         << manifest.payload_bytes << '\n';
  for (const auto& entry : manifest.projections) {
    output << entry.ordinal << ':' << entry.layer_index << ':'
           << attention_r1_family_name(entry.family) << ':'
           << entry.source_module << ':' << entry.source_sha256 << ':'
           << entry.output_size << ':' << entry.input_size << ':'
           << entry.payload_offset << ':' << entry.payload_bytes << '\n';
  }
  const std::string bytes = output.str();
  core::Sha256 hasher;
  return hasher.update(bytes.data(), bytes.size())
             ? hasher.finalize().hex()
             : std::string{};
}

}  // namespace

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
        down->input_size != kGateRows) {
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

bool ModelWeights::attach_prefill_a4_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillSidecarManifest* const manifest,
    const PrefillA4CalibrationPolicy* const policy,
    const std::string_view authenticated_payload_sha256) noexcept {
  const auto projection_for = [](DecoderLayerWeights& layer,
                                 const PrefillProjectionFamily family)
      noexcept -> LinearWeight* {
    switch (family) {
      case PrefillProjectionFamily::kMlpGate:
        return &layer.mlp.gate_proj;
      case PrefillProjectionFamily::kMlpUp:
        return &layer.mlp.up_proj;
      case PrefillProjectionFamily::kMlpDown:
        return &layer.mlp.down_proj;
      case PrefillProjectionFamily::kLinearQkv:
      case PrefillProjectionFamily::kLinearZ:
      case PrefillProjectionFamily::kLinearO: {
        auto* const attention =
            std::get_if<LinearAttentionWeights>(&layer.attention);
        if (attention == nullptr) {
          return nullptr;
        }
        if (family == PrefillProjectionFamily::kLinearQkv) {
          return &attention->in_proj_qkv;
        }
        if (family == PrefillProjectionFamily::kLinearZ) {
          return &attention->in_proj_z;
        }
        return &attention->out_proj;
      }
      case PrefillProjectionFamily::kFullQ:
      case PrefillProjectionFamily::kFullK:
      case PrefillProjectionFamily::kFullV:
      case PrefillProjectionFamily::kFullO: {
        auto* const attention =
            std::get_if<FullAttentionWeights>(&layer.attention);
        if (attention == nullptr) {
          return nullptr;
        }
        if (family == PrefillProjectionFamily::kFullQ) {
          return &attention->q_proj;
        }
        if (family == PrefillProjectionFamily::kFullK) {
          return &attention->k_proj;
        }
        if (family == PrefillProjectionFamily::kFullV) {
          return &attention->v_proj;
        }
        return &attention->o_proj;
      }
      case PrefillProjectionFamily::kCount:
        return nullptr;
    }
    return nullptr;
  };
  const auto clear_weight = [](LinearWeight& binding) noexcept {
    if (auto* const fp8 = std::get_if<Fp8LinearWeight>(&binding)) {
      fp8->prefill_a4_weight = nullptr;
      fp8->prefill_a4_scales = nullptr;
      fp8->prefill_a4_metadata = nullptr;
      fp8->prefill_a4_sidecar_kind = PrefillSidecarKind::kExact;
      fp8->prefill_a4_packed_k_group_size = 0U;
      fp8->prefill_a4_scale_group_size = 0U;
      fp8->prefill_a4_activation_clip_ratio = 0.0F;
      fp8->prefill_attention_o_k512_weight = nullptr;
      fp8->prefill_attention_o_k512_scales = nullptr;
      fp8->prefill_attention_o_k512_activation_clip_ratio = 0.0F;
    } else if (auto* const nvfp4 =
                   std::get_if<NvFp4LinearWeight>(&binding)) {
      nvfp4->prefill_a4_weight = nullptr;
      nvfp4->prefill_a4_scales = nullptr;
      nvfp4->prefill_a4_metadata = nullptr;
      nvfp4->prefill_a4_sidecar_kind = PrefillSidecarKind::kExact;
      nvfp4->prefill_a4_packed_k_group_size = 0U;
      nvfp4->prefill_a4_scale_group_size = 0U;
      nvfp4->prefill_a4_activation_clip_ratio = 0.0F;
      nvfp4->prefill_mlp_k512_weight = nullptr;
      nvfp4->prefill_mlp_k512_scales = nullptr;
      nvfp4->prefill_mlp_k512_activation_clip_ratio = 0.0F;
    }
  };
  const auto clear_all = [this, &clear_weight]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      layer.prefill_mlp_k512_fragment_native = {};
      layer.prefill_mlp_factorized_lane_r1 = {};
      layer.prefill_attention_factorized_lane_r1 = {};
      clear_weight(layer.mlp.gate_proj);
      clear_weight(layer.mlp.up_proj);
      clear_weight(layer.mlp.down_proj);
      if (auto* const linear =
              std::get_if<LinearAttentionWeights>(&layer.attention)) {
        clear_weight(linear->in_proj_qkv);
        clear_weight(linear->in_proj_z);
        clear_weight(linear->out_proj);
      } else if (auto* const full =
                     std::get_if<FullAttentionWeights>(&layer.attention)) {
        clear_weight(full->q_proj);
        clear_weight(full->k_proj);
        clear_weight(full->v_proj);
        clear_weight(full->o_proj);
      }
    }
    prefill_a4_attachment_complete_ = false;
    prefill_a4_attachment_manifest_sha256_.fill('\0');
    prefill_a4_attachment_policy_sha256_.fill('\0');
    prefill_a4_attachment_payload_sha256_.fill('\0');
  };

  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      policy == nullptr) {
    clear_all();
    return true;
  }
  bool contracts_valid = false;
  try {
    contracts_valid = manifest != nullptr && policy != nullptr &&
                      validate_prefill_sidecar_manifest(*manifest).ok() &&
                      validate_prefill_a4_calibration_policy(
                          *policy, *manifest)
                          .ok();
  } catch (...) {
    return false;
  }
  const auto lower_sha256 = [](const std::string_view value) noexcept {
    if (value.size() != 64U) {
      return false;
    }
    for (const char character : value) {
      if (!((character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f'))) {
        return false;
      }
    }
    return true;
  };
  if (!authenticated_payload_sha256.empty() &&
      !lower_sha256(authenticated_payload_sha256)) {
    return false;
  }
  const bool k64_v1 =
      manifest != nullptr && manifest->kind == PrefillSidecarKind::kA4K64;
  const bool k128_v2 =
      manifest != nullptr && manifest->kind == PrefillSidecarKind::kA4K128;
  const bool k256_v3 =
      manifest != nullptr && manifest->kind == PrefillSidecarKind::kA4K256;
  const std::uint32_t expected_scale_group_size =
      k256_v3 ? 256U : k128_v2 ? 128U : 64U;
  const PrefillSidecarLayout expected_layout =
      k256_v3 ? PrefillSidecarLayout::kSm87S4K256Consumer
              : k128_v2 ? PrefillSidecarLayout::kSm87S4K128Consumer
                        : PrefillSidecarLayout::kSm87S4K64Consumer;
  if (arena == nullptr || manifest == nullptr || policy == nullptr ||
      manifest->residency_class != PrefillSidecarResidencyClass::kA4 ||
      (!k64_v1 && !k128_v2 && !k256_v3) ||
      !contracts_valid ||
      manifest->projections.size() != kQwen36PrefillProjectionCount ||
      policy->projections.size() != manifest->projections.size() ||
      manifest->summary.arena_bytes != arena_bytes ||
      arena_bytes == 0U) {
    return false;
  }
  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % manifest->arena_alignment != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  struct ValidatedBinding {
    LinearWeight* binding = nullptr;
    const std::uint8_t* weight = nullptr;
    const std::uint16_t* scales = nullptr;
    const std::uint8_t* metadata = nullptr;
    PrefillSidecarKind sidecar_kind = PrefillSidecarKind::kExact;
    std::uint32_t packed_k_group_size = 0U;
    std::uint32_t scale_group_size = 0U;
    float activation_clip_ratio = 0.0F;
  };
  std::array<ValidatedBinding, kQwen36PrefillProjectionCount> validated{};
  for (std::size_t index = 0U; index < manifest->projections.size();
       ++index) {
    const PrefillProjectionSidecarEntry& entry =
        manifest->projections[index];
    const PrefillA4ProjectionCalibration& calibration =
        policy->projections[index];
    if (entry.ordinal != index || entry.layer_index >= layers_.size() ||
        entry.quantization != PrefillWeightQuantization::kSymmetricW4 ||
        entry.layout != expected_layout ||
        entry.scale_group_size != expected_scale_group_size ||
        calibration.activation_scale_group_size !=
            expected_scale_group_size ||
        calibration.channel_equalization.has_value() ||
        !std::isfinite(calibration.activation_clip_ratio) ||
        calibration.activation_clip_ratio <= 0.0 ||
        calibration.activation_clip_ratio > 1.0 ||
        entry.input_size % entry.scale_group_size != 0U ||
        entry.sidecar_offset > arena_bytes ||
        entry.sidecar_byte_size > arena_bytes - entry.sidecar_offset ||
        entry.weight_bytes > entry.sidecar_byte_size ||
        entry.scale_bytes > entry.sidecar_byte_size - entry.weight_bytes ||
        entry.metadata_bytes !=
            entry.sidecar_byte_size - entry.weight_bytes -
                entry.scale_bytes) {
      return false;
    }
    LinearWeight* const binding =
        projection_for(layers_[entry.layer_index], entry.family);
    if (binding == nullptr ||
        linear_output_size(*binding) != entry.output_size ||
        linear_input_size(*binding) != entry.input_size) {
      return false;
    }
    const bool mlp =
        entry.family == PrefillProjectionFamily::kMlpGate ||
        entry.family == PrefillProjectionFamily::kMlpUp ||
        entry.family == PrefillProjectionFamily::kMlpDown;
    if (mlp) {
      const auto* const nvfp4 =
          std::get_if<NvFp4LinearWeight>(binding);
      if (!has_valid_nvfp4_payload(nvfp4) ||
          nvfp4->prefill_marlin_weight != nullptr ||
          nvfp4->prefill_marlin_scales != nullptr ||
          nvfp4->prefill_marlin_global_scale != nullptr) {
        return false;
      }
    } else {
      const auto* const fp8 = std::get_if<Fp8LinearWeight>(binding);
      if (!has_valid_fp8_payload(fp8) ||
          fp8->prefill_qkv_register_feed_sidecar != nullptr ||
          fp8->prefill_supermatrix_sidecar != nullptr ||
          fp8->prefill_marlin_weight != nullptr ||
          fp8->prefill_marlin_scales != nullptr) {
        return false;
      }
    }

    const std::uintptr_t projection_address =
        arena_address + entry.sidecar_offset;
    if (projection_address >= arena_end ||
        projection_address % 16U != 0U ||
        entry.weight_bytes > arena_end - projection_address) {
      return false;
    }
    const std::uintptr_t scale_address =
        projection_address + entry.weight_bytes;
    if (scale_address % alignof(std::uint16_t) != 0U ||
        entry.scale_bytes > arena_end - scale_address) {
      return false;
    }
    const std::uintptr_t metadata_address =
        scale_address + entry.scale_bytes;
    if (entry.metadata_bytes > arena_end - metadata_address) {
      return false;
    }
    validated[index] = ValidatedBinding{
        binding,
        reinterpret_cast<const std::uint8_t*>(projection_address),
        reinterpret_cast<const std::uint16_t*>(scale_address),
        entry.metadata_bytes == 0U
            ? nullptr
            : reinterpret_cast<const std::uint8_t*>(metadata_address),
        manifest->kind,
        64U,
        entry.scale_group_size,
        static_cast<float>(calibration.activation_clip_ratio)};
  }

  clear_all();
  for (const ValidatedBinding& entry : validated) {
    if (auto* const fp8 = std::get_if<Fp8LinearWeight>(entry.binding)) {
      fp8->prefill_a4_weight = entry.weight;
      fp8->prefill_a4_scales = entry.scales;
      fp8->prefill_a4_metadata = entry.metadata;
      fp8->prefill_a4_sidecar_kind = entry.sidecar_kind;
      fp8->prefill_a4_packed_k_group_size = entry.packed_k_group_size;
      fp8->prefill_a4_scale_group_size = entry.scale_group_size;
      fp8->prefill_a4_activation_clip_ratio =
          entry.activation_clip_ratio;
    } else if (auto* const nvfp4 =
                   std::get_if<NvFp4LinearWeight>(entry.binding)) {
      nvfp4->prefill_a4_weight = entry.weight;
      nvfp4->prefill_a4_scales = entry.scales;
      nvfp4->prefill_a4_metadata = entry.metadata;
      nvfp4->prefill_a4_sidecar_kind = entry.sidecar_kind;
      nvfp4->prefill_a4_packed_k_group_size = entry.packed_k_group_size;
      nvfp4->prefill_a4_scale_group_size = entry.scale_group_size;
      nvfp4->prefill_a4_activation_clip_ratio =
          entry.activation_clip_ratio;
    } else {
      clear_all();
      return false;
    }
  }
  if (!authenticated_payload_sha256.empty()) {
    prefill_a4_attachment_complete_ = true;
    std::copy(manifest->manifest_sha256.begin(),
              manifest->manifest_sha256.end(),
              prefill_a4_attachment_manifest_sha256_.begin());
    std::copy(policy->policy_sha256.begin(), policy->policy_sha256.end(),
              prefill_a4_attachment_policy_sha256_.begin());
    std::copy(authenticated_payload_sha256.begin(),
              authenticated_payload_sha256.end(),
              prefill_a4_attachment_payload_sha256_.begin());
  }
  return true;
}

bool ModelWeights::attach_prefill_attention_o_k512_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillAttentionOK512OverlayManifest* const manifest,
    const PrefillAttentionOK512OverlayPolicy* const policy) noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      LinearWeight* output = nullptr;
      if (auto* const linear =
              std::get_if<LinearAttentionWeights>(&layer.attention)) {
        output = &linear->out_proj;
      } else if (auto* const full =
                     std::get_if<FullAttentionWeights>(&layer.attention)) {
        output = &full->o_proj;
      }
      if (output != nullptr) {
        if (auto* const fp8 = std::get_if<Fp8LinearWeight>(output)) {
          fp8->prefill_attention_o_k512_weight = nullptr;
          fp8->prefill_attention_o_k512_scales = nullptr;
          fp8->prefill_attention_o_k512_activation_clip_ratio = 0.0F;
        }
      }
    }
  };

  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      policy == nullptr) {
    clear_all();
    return true;
  }
  if (std::any_of(
          layers_.begin(), layers_.end(), [](const DecoderLayerWeights& layer) {
            return !layer.prefill_attention_factorized_lane_r1.empty();
          })) {
    return false;
  }
  if (arena == nullptr || manifest == nullptr || policy == nullptr ||
      arena_bytes != kPrefillAttentionOK512OverlayPayloadBytes ||
      manifest->payload_bytes != arena_bytes ||
      manifest->projections.size() !=
          kPrefillAttentionOK512OverlayProjectionCount ||
      policy->projections.size() != manifest->projections.size()) {
    return false;
  }
  const auto lower_sha256 = [](const std::string_view value) noexcept {
    if (value.size() != 64U) {
      return false;
    }
    for (const char character : value) {
      if (!((character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f'))) {
        return false;
      }
    }
    return true;
  };
  try {
    if (!validate_prefill_attention_o_k512_overlay_manifest(*manifest) ||
        policy->version_major != manifest->version_major ||
        policy->version_minor != manifest->version_minor ||
        policy->physical_layout != manifest->physical_layout ||
        policy->source_checkpoint_id != manifest->source_checkpoint_id ||
        policy->source_config_sha256 != manifest->source_config_sha256 ||
        policy->source_index_sha256 != manifest->source_index_sha256 ||
        policy->manifest_sha256 != manifest->manifest_sha256 ||
        policy->required_base.physical_layout !=
            manifest->required_base.physical_layout ||
        policy->required_base.manifest_sha256 !=
            manifest->required_base.manifest_sha256 ||
        policy->required_base.policy_sha256 !=
            manifest->required_base.policy_sha256 ||
        policy->required_base.payload_sha256 !=
            manifest->required_base.payload_sha256 ||
        !lower_sha256(policy->policy_sha256) || policy->policy_bytes == 0U ||
        !prefill_a4_attachment_complete_ ||
        std::string_view(prefill_a4_attachment_manifest_sha256_.data(),
                         prefill_a4_attachment_manifest_sha256_.size()) !=
            manifest->required_base.manifest_sha256 ||
        std::string_view(prefill_a4_attachment_policy_sha256_.data(),
                         prefill_a4_attachment_policy_sha256_.size()) !=
            manifest->required_base.policy_sha256 ||
        std::string_view(prefill_a4_attachment_payload_sha256_.data(),
                         prefill_a4_attachment_payload_sha256_.size()) !=
            manifest->required_base.payload_sha256) {
      return false;
    }
  } catch (...) {
    return false;
  }

  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % 256U != 0U || arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  struct ValidatedBinding final {
    Fp8LinearWeight* binding = nullptr;
    const std::uint8_t* weight = nullptr;
    const std::uint16_t* scales = nullptr;
    float activation_clip_ratio = 0.0F;
  };
  std::array<ValidatedBinding,
             kPrefillAttentionOK512OverlayProjectionCount>
      validated{};
  std::array<bool, kQwen36DenseLayerCount> seen{};
  for (std::size_t index = 0U; index < manifest->projections.size();
       ++index) {
    const PrefillAttentionOK512OverlayEntry& entry =
        manifest->projections[index];
    const PrefillAttentionOK512OverlayCalibration& calibration =
        policy->projections[index];
    if (entry.ordinal != index || entry.layer_index >= layers_.size() ||
        seen[entry.layer_index] ||
        entry.output_size != kFp8M1OutputProjectionRows ||
        entry.input_size != kFp8M1OutputProjectionColumns ||
        entry.weight_bytes !=
            kPrefillAttentionOK512OverlayProjectionWeightBytes ||
        entry.scale_bytes !=
            kPrefillAttentionOK512OverlayProjectionScaleBytes ||
        entry.sidecar_offset > arena_bytes ||
        entry.weight_bytes > arena_bytes - entry.sidecar_offset ||
        entry.scale_bytes >
            arena_bytes - entry.sidecar_offset - entry.weight_bytes ||
        calibration.ordinal != entry.ordinal ||
        calibration.source_module != entry.source_module ||
        calibration.source_sha256 != entry.source_sha256 ||
        calibration.activation_scale_group_size !=
            kPrefillAttentionOK512OverlayScaleK ||
        !std::isfinite(calibration.weight_clip_ratio) ||
        calibration.weight_clip_ratio < kPrefillA4MinimumClipRatio ||
        calibration.weight_clip_ratio > 1.0 ||
        !std::isfinite(calibration.activation_clip_ratio) ||
        calibration.activation_clip_ratio < kPrefillA4MinimumClipRatio ||
        calibration.activation_clip_ratio > 1.0 ||
        static_cast<float>(calibration.activation_clip_ratio) <
            static_cast<float>(kPrefillA4MinimumClipRatio) ||
        static_cast<float>(calibration.activation_clip_ratio) > 1.0F) {
      return false;
    }

    DecoderLayerWeights& layer = layers_[entry.layer_index];
    LinearWeight* output = nullptr;
    if (entry.family == "linear_o") {
      auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer.attention);
      if (linear != nullptr) {
        output = &linear->out_proj;
      }
    } else if (entry.family == "full_o") {
      auto* const full =
          std::get_if<FullAttentionWeights>(&layer.attention);
      if (full != nullptr) {
        output = &full->o_proj;
      }
    }
    auto* const fp8 =
        output == nullptr ? nullptr : std::get_if<Fp8LinearWeight>(output);
    if (!has_valid_fp8_payload(fp8) ||
        fp8->output_size != entry.output_size ||
        fp8->input_size != entry.input_size) {
      return false;
    }
    const PrefillA4LinearSidecarView base =
        prefill_a4_sidecar_view(*output);
    if (!base.attached() ||
        base.sidecar_kind != PrefillSidecarKind::kA4K128 ||
        base.packed_k_group_size != 64U || base.scale_group_size != 128U) {
      return false;
    }

    const std::uintptr_t weight_address =
        arena_address + entry.sidecar_offset;
    const std::uintptr_t scale_address =
        weight_address + entry.weight_bytes;
    if (weight_address >= arena_end || weight_address % 16U != 0U ||
        scale_address >= arena_end ||
        scale_address % alignof(std::uint16_t) != 0U ||
        entry.scale_bytes > arena_end - scale_address) {
      return false;
    }
    seen[entry.layer_index] = true;
    validated[index] = {
        fp8, reinterpret_cast<const std::uint8_t*>(weight_address),
        reinterpret_cast<const std::uint16_t*>(scale_address),
        static_cast<float>(calibration.activation_clip_ratio)};
  }
  if (std::any_of(seen.begin(), seen.end(), [](const bool value) {
        return !value;
      })) {
    return false;
  }

  clear_all();
  for (const ValidatedBinding& entry : validated) {
    entry.binding->prefill_attention_o_k512_weight = entry.weight;
    entry.binding->prefill_attention_o_k512_scales = entry.scales;
    entry.binding->prefill_attention_o_k512_activation_clip_ratio =
        entry.activation_clip_ratio;
  }
  return true;
}

bool ModelWeights::attach_prefill_mlp_k512_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillMLPK512OverlayManifest* const manifest,
    const PrefillMLPK512OverlayPolicy* const policy) noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      for (NvFp4LinearWeight* const projection :
           {nvfp4_gate_projection(layer), nvfp4_up_projection(layer),
            nvfp4_down_projection(layer)}) {
        if (projection != nullptr) {
          projection->prefill_mlp_k512_weight = nullptr;
          projection->prefill_mlp_k512_scales = nullptr;
          projection->prefill_mlp_k512_activation_clip_ratio = 0.0F;
        }
      }
    }
  };

  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      policy == nullptr) {
    clear_all();
    return true;
  }
  if (std::any_of(
          layers_.begin(), layers_.end(), [](const DecoderLayerWeights& layer) {
            return !layer.prefill_mlp_factorized_lane_r1.empty() ||
                   !layer.prefill_mlp_factorized_lane_r4.empty();
          })) {
    return false;
  }
  if (std::any_of(
          layers_.begin(), layers_.end(), [](const DecoderLayerWeights& layer) {
            return !layer.prefill_mlp_k512_fragment_native.empty();
          })) {
    return false;
  }
  if (arena == nullptr || manifest == nullptr || policy == nullptr ||
      arena_bytes != kPrefillMLPK512OverlayPayloadBytes ||
      manifest->payload_bytes != arena_bytes ||
      manifest->projections.size() !=
          kPrefillMLPK512OverlayProjectionCount ||
      policy->projections.size() != manifest->projections.size()) {
    return false;
  }
  const auto lower_sha256 = [](const std::string_view value) noexcept {
    if (value.size() != 64U) {
      return false;
    }
    for (const char character : value) {
      if (!((character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f'))) {
        return false;
      }
    }
    return true;
  };
  try {
    if (!validate_prefill_mlp_k512_overlay_manifest(*manifest) ||
        policy->version_major != manifest->version_major ||
        policy->version_minor != manifest->version_minor ||
        policy->physical_layout != manifest->physical_layout ||
        policy->source_checkpoint_id != manifest->source_checkpoint_id ||
        policy->source_config_sha256 != manifest->source_config_sha256 ||
        policy->source_index_sha256 != manifest->source_index_sha256 ||
        policy->manifest_sha256 != manifest->manifest_sha256 ||
        policy->required_base.physical_layout !=
            manifest->required_base.physical_layout ||
        policy->required_base.manifest_sha256 !=
            manifest->required_base.manifest_sha256 ||
        policy->required_base.policy_sha256 !=
            manifest->required_base.policy_sha256 ||
        policy->required_base.payload_sha256 !=
            manifest->required_base.payload_sha256 ||
        !lower_sha256(policy->policy_sha256) || policy->policy_bytes == 0U ||
        !prefill_a4_attachment_complete_ ||
        std::string_view(prefill_a4_attachment_manifest_sha256_.data(),
                         prefill_a4_attachment_manifest_sha256_.size()) !=
            manifest->required_base.manifest_sha256 ||
        std::string_view(prefill_a4_attachment_policy_sha256_.data(),
                         prefill_a4_attachment_policy_sha256_.size()) !=
            manifest->required_base.policy_sha256 ||
        std::string_view(prefill_a4_attachment_payload_sha256_.data(),
                         prefill_a4_attachment_payload_sha256_.size()) !=
            manifest->required_base.payload_sha256) {
      return false;
    }
  } catch (...) {
    return false;
  }

  for (std::size_t layer = 0U;
       layer < policy->projections.size() / 3U; ++layer) {
    const std::size_t gate_index = 3U * layer;
    const std::size_t up_index = gate_index + 1U;
    if (static_cast<float>(
            policy->projections[gate_index].activation_clip_ratio) !=
        static_cast<float>(
            policy->projections[up_index].activation_clip_ratio)) {
      return false;
    }
  }

  if (manifest->required_base.physical_layout !=
          kPrefillA4K128PhysicalLayout &&
      manifest->required_base.physical_layout !=
          kPrefillA4K256PhysicalLayout) {
    return false;
  }

  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % 256U != 0U || arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  struct ValidatedBinding final {
    NvFp4LinearWeight* binding = nullptr;
    const std::uint8_t* weight = nullptr;
    const std::uint16_t* scales = nullptr;
    float activation_clip_ratio = 0.0F;
  };
  std::array<ValidatedBinding, kPrefillMLPK512OverlayProjectionCount>
      validated{};
  for (std::size_t index = 0U; index < manifest->projections.size();
       ++index) {
    const PrefillMLPK512OverlayEntry& entry = manifest->projections[index];
    const PrefillMLPK512OverlayCalibration& calibration =
        policy->projections[index];
    const bool gate_or_up = entry.family == "gate" || entry.family == "up";
    const bool down = entry.family == "down";
    if (entry.ordinal != index || entry.layer_index >= layers_.size() ||
        (!gate_or_up && !down) ||
        entry.output_size !=
            (down ? kPrefillMLPK512OverlayDownOutputSize
                  : kPrefillMLPK512OverlayGateUpOutputSize) ||
        entry.input_size !=
            (down ? kPrefillMLPK512OverlayDownInputSize
                  : kPrefillMLPK512OverlayGateUpInputSize) ||
        entry.weight_bytes !=
            kPrefillMLPK512OverlayProjectionWeightBytes ||
        entry.scale_bytes !=
            kPrefillMLPK512OverlayProjectionScaleBytes ||
        entry.sidecar_offset > arena_bytes ||
        entry.weight_bytes > arena_bytes - entry.sidecar_offset ||
        entry.scale_bytes >
            arena_bytes - entry.sidecar_offset - entry.weight_bytes ||
        calibration.ordinal != entry.ordinal ||
        calibration.source_module != entry.source_module ||
        calibration.source_sha256 != entry.source_sha256 ||
        calibration.activation_scale_group_size !=
            kPrefillMLPK512OverlayScaleK ||
        !std::isfinite(calibration.weight_clip_ratio) ||
        calibration.weight_clip_ratio < kPrefillA4MinimumClipRatio ||
        calibration.weight_clip_ratio > 1.0 ||
        !std::isfinite(calibration.activation_clip_ratio) ||
        calibration.activation_clip_ratio < kPrefillA4MinimumClipRatio ||
        calibration.activation_clip_ratio > 1.0 ||
        static_cast<float>(calibration.activation_clip_ratio) <
            static_cast<float>(kPrefillA4MinimumClipRatio) ||
        static_cast<float>(calibration.activation_clip_ratio) > 1.0F) {
      return false;
    }

    DecoderLayerWeights& layer = layers_[entry.layer_index];
    LinearWeight* target = nullptr;
    if (entry.family == "gate") {
      target = &layer.mlp.gate_proj;
    } else if (entry.family == "up") {
      target = &layer.mlp.up_proj;
    } else {
      target = &layer.mlp.down_proj;
    }
    auto* const nvfp4 = std::get_if<NvFp4LinearWeight>(target);
    if (!has_valid_nvfp4_payload(nvfp4) ||
        nvfp4->output_size != entry.output_size ||
        nvfp4->input_size != entry.input_size ||
        nvfp4->prefill_marlin_weight != nullptr ||
        nvfp4->prefill_marlin_scales != nullptr ||
        nvfp4->prefill_marlin_global_scale != nullptr) {
      return false;
    }
    const PrefillA4LinearSidecarView base =
        prefill_a4_sidecar_view(*target);
    const bool supported_base =
        prefill_mlp_k512_base_layout_matches_contract(
            manifest->required_base.physical_layout, base.sidecar_kind,
            base.packed_k_group_size, base.scale_group_size);
    if (!base.attached() || !supported_base ||
        base.scale_group_size == 0U) {
      return false;
    }

    const std::uintptr_t weight_address =
        arena_address + entry.sidecar_offset;
    const std::uintptr_t scale_address =
        weight_address + entry.weight_bytes;
    if (weight_address >= arena_end || weight_address % 16U != 0U ||
        scale_address >= arena_end ||
        scale_address % alignof(std::uint16_t) != 0U ||
        entry.scale_bytes > arena_end - scale_address) {
      return false;
    }
    validated[index] = {
        nvfp4, reinterpret_cast<const std::uint8_t*>(weight_address),
        reinterpret_cast<const std::uint16_t*>(scale_address),
        static_cast<float>(calibration.activation_clip_ratio)};
  }

  clear_all();
  for (const ValidatedBinding& entry : validated) {
    entry.binding->prefill_mlp_k512_weight = entry.weight;
    entry.binding->prefill_mlp_k512_scales = entry.scales;
    entry.binding->prefill_mlp_k512_activation_clip_ratio =
        entry.activation_clip_ratio;
  }
  return true;
}

bool ModelWeights::attach_prefill_mlp_k512_fragment_native_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillMLPK512FragmentNativeManifest* const manifest,
    const PrefillMLPK512OverlayManifest* const source_v1_manifest,
    const PrefillMLPK512OverlayPolicy* const source_v1_policy) noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      layer.prefill_mlp_k512_fragment_native = {};
    }
  };
  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      source_v1_manifest == nullptr && source_v1_policy == nullptr) {
    clear_all();
    return true;
  }
  if (std::any_of(
          layers_.begin(), layers_.end(), [](const DecoderLayerWeights& layer) {
            return !layer.prefill_mlp_factorized_lane_r1.empty() ||
                   !layer.prefill_mlp_factorized_lane_r4.empty();
          })) {
    return false;
  }

  for (const DecoderLayerWeights& layer : layers_) {
    const auto& composite = layer.prefill_mlp_k512_fragment_native;
    if (!composite.empty() &&
        composite.physical_layout !=
            PrefillMLPK512CompositeLayout::
                kPairedGateUpFragmentNativeDown) {
      return false;
    }
  }

  const auto v1_projection_attached = [](const NvFp4LinearWeight* const value)
      noexcept {
    return value != nullptr &&
           (value->prefill_mlp_k512_weight != nullptr ||
            value->prefill_mlp_k512_scales != nullptr ||
            value->prefill_mlp_k512_activation_clip_ratio != 0.0F);
  };
  for (DecoderLayerWeights& layer : layers_) {
    if (v1_projection_attached(nvfp4_gate_projection(layer)) ||
        v1_projection_attached(nvfp4_up_projection(layer)) ||
        v1_projection_attached(nvfp4_down_projection(layer))) {
      return false;
    }
  }

  if (arena == nullptr || manifest == nullptr ||
      source_v1_manifest == nullptr || source_v1_policy == nullptr ||
      arena_bytes != kPrefillMLPK512FragmentNativePayloadBytes ||
      manifest->payload_bytes != arena_bytes ||
      manifest->layer_count != kPrefillMLPK512FragmentNativeLayerCount ||
      source_v1_manifest->payload_bytes !=
          kPrefillMLPK512OverlayPayloadBytes ||
      source_v1_manifest->projections.size() !=
          kPrefillMLPK512OverlayProjectionCount ||
      source_v1_policy->projections.size() !=
          source_v1_manifest->projections.size()) {
    return false;
  }

  const auto same_base = [](const PrefillMLPK512BaseBinding& first,
                            const PrefillMLPK512BaseBinding& second) noexcept {
    return first.physical_layout == second.physical_layout &&
           first.manifest_sha256 == second.manifest_sha256 &&
           first.policy_sha256 == second.policy_sha256 &&
           first.payload_sha256 == second.payload_sha256;
  };
  const auto lower_sha256 = [](const std::string_view value) noexcept {
    return value.size() == 64U &&
           std::all_of(value.begin(), value.end(), [](const char character) {
             return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
           });
  };
  try {
    if (!validate_prefill_mlp_k512_fragment_native_manifest(*manifest) ||
        !validate_prefill_mlp_k512_overlay_manifest(*source_v1_manifest) ||
        manifest->source_checkpoint_id !=
            source_v1_manifest->source_checkpoint_id ||
        manifest->source_config_sha256 !=
            source_v1_manifest->source_config_sha256 ||
        manifest->source_index_sha256 !=
            source_v1_manifest->source_index_sha256 ||
        !same_base(manifest->required_base,
                   source_v1_manifest->required_base) ||
        manifest->source_v1.physical_layout !=
            source_v1_manifest->physical_layout ||
        manifest->source_v1.manifest_sha256 !=
            source_v1_manifest->manifest_sha256 ||
        manifest->source_v1.payload_bytes !=
            source_v1_manifest->payload_bytes ||
        source_v1_policy->version_major !=
            source_v1_manifest->version_major ||
        source_v1_policy->version_minor !=
            source_v1_manifest->version_minor ||
        source_v1_policy->physical_layout !=
            source_v1_manifest->physical_layout ||
        source_v1_policy->source_checkpoint_id !=
            source_v1_manifest->source_checkpoint_id ||
        source_v1_policy->source_config_sha256 !=
            source_v1_manifest->source_config_sha256 ||
        source_v1_policy->source_index_sha256 !=
            source_v1_manifest->source_index_sha256 ||
        source_v1_policy->manifest_sha256 !=
            source_v1_manifest->manifest_sha256 ||
        !same_base(source_v1_policy->required_base,
                   source_v1_manifest->required_base) ||
        !lower_sha256(source_v1_policy->policy_sha256) ||
        source_v1_policy->policy_bytes == 0U ||
        manifest->source_v1.policy_sha256 !=
            source_v1_policy->policy_sha256 ||
        manifest->source_v1.policy_bytes != source_v1_policy->policy_bytes ||
        !prefill_a4_attachment_complete_ ||
        std::string_view(prefill_a4_attachment_manifest_sha256_.data(),
                         prefill_a4_attachment_manifest_sha256_.size()) !=
            manifest->required_base.manifest_sha256 ||
        std::string_view(prefill_a4_attachment_policy_sha256_.data(),
                         prefill_a4_attachment_policy_sha256_.size()) !=
            manifest->required_base.policy_sha256 ||
        std::string_view(prefill_a4_attachment_payload_sha256_.data(),
                         prefill_a4_attachment_payload_sha256_.size()) !=
            manifest->required_base.payload_sha256) {
      return false;
    }
  } catch (...) {
    return false;
  }

  constexpr std::uintptr_t kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % 256U != 0U || arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  std::array<PrefillMLPK512FragmentNativeCompositeView,
             kPrefillMLPK512FragmentNativeLayerCount>
      validated{};
  for (std::size_t layer_index = 0U;
       layer_index < kPrefillMLPK512FragmentNativeLayerCount;
       ++layer_index) {
    const std::size_t gate_index = 3U * layer_index;
    const std::size_t up_index = gate_index + 1U;
    const std::size_t down_index = gate_index + 2U;
    const PrefillMLPK512OverlayEntry& gate_entry =
        source_v1_manifest->projections[gate_index];
    const PrefillMLPK512OverlayEntry& up_entry =
        source_v1_manifest->projections[up_index];
    const PrefillMLPK512OverlayEntry& down_entry =
        source_v1_manifest->projections[down_index];
    const PrefillMLPK512OverlayCalibration& gate_calibration =
        source_v1_policy->projections[gate_index];
    const PrefillMLPK512OverlayCalibration& up_calibration =
        source_v1_policy->projections[up_index];
    const PrefillMLPK512OverlayCalibration& down_calibration =
        source_v1_policy->projections[down_index];

    const auto calibration_valid = [](const PrefillMLPK512OverlayEntry& entry,
                                      const PrefillMLPK512OverlayCalibration&
                                          calibration) noexcept {
      return calibration.ordinal == entry.ordinal &&
             calibration.source_module == entry.source_module &&
             calibration.source_sha256 == entry.source_sha256 &&
             calibration.activation_scale_group_size ==
                 kPrefillMLPK512OverlayScaleK &&
             std::isfinite(calibration.weight_clip_ratio) &&
             calibration.weight_clip_ratio >= kPrefillA4MinimumClipRatio &&
             calibration.weight_clip_ratio <= 1.0 &&
             std::isfinite(calibration.activation_clip_ratio) &&
             calibration.activation_clip_ratio >=
                 kPrefillA4MinimumClipRatio &&
             calibration.activation_clip_ratio <= 1.0 &&
             static_cast<float>(calibration.activation_clip_ratio) >=
                 static_cast<float>(kPrefillA4MinimumClipRatio) &&
             static_cast<float>(calibration.activation_clip_ratio) <= 1.0F;
    };
    if (!calibration_valid(gate_entry, gate_calibration) ||
        !calibration_valid(up_entry, up_calibration) ||
        !calibration_valid(down_entry, down_calibration) ||
        static_cast<float>(gate_calibration.activation_clip_ratio) !=
            static_cast<float>(up_calibration.activation_clip_ratio)) {
      return false;
    }

    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    const auto projection_valid = [](const NvFp4LinearWeight* const weight,
                                     const std::size_t output_size,
                                     const std::size_t input_size) noexcept {
      if (!has_valid_nvfp4_payload(weight) ||
          weight->output_size != output_size ||
          weight->input_size != input_size ||
          weight->prefill_marlin_weight != nullptr ||
          weight->prefill_marlin_scales != nullptr ||
          weight->prefill_marlin_global_scale != nullptr) {
        return false;
      }
      return weight->prefill_a4_weight != nullptr &&
             weight->prefill_a4_scales != nullptr &&
             weight->prefill_a4_sidecar_kind ==
                 PrefillSidecarKind::kA4K128 &&
             weight->prefill_a4_packed_k_group_size ==
                 kPrefillMLPK512OverlayPackedK &&
             weight->prefill_a4_scale_group_size == 128U &&
             weight->prefill_a4_activation_clip_ratio > 0.0F &&
             weight->prefill_a4_activation_clip_ratio <= 1.0F;
    };
    if (!projection_valid(gate, kPrefillMLPK512OverlayGateUpOutputSize,
                          kPrefillMLPK512OverlayGateUpInputSize) ||
        !projection_valid(up, kPrefillMLPK512OverlayGateUpOutputSize,
                          kPrefillMLPK512OverlayGateUpInputSize) ||
        !projection_valid(down, kPrefillMLPK512OverlayDownOutputSize,
                          kPrefillMLPK512OverlayDownInputSize)) {
      return false;
    }

    const PrefillMLPK512FragmentNativeLayerView layout =
        prefill_mlp_k512_fragment_native_layer_view(layer_index);
    const auto range_valid = [arena_bytes](const std::uint64_t offset,
                                           const std::uint64_t bytes) noexcept {
      return offset < arena_bytes && bytes <= arena_bytes - offset;
    };
    if (!layout.valid || layout.layer_index != layer_index ||
        layout.gateup_code_bytes !=
            kPrefillMLPK512FragmentNativeGateUpCodeBytes ||
        layout.gateup_scale_bytes !=
            kPrefillMLPK512FragmentNativeGateUpScaleBytes ||
        layout.down_code_bytes !=
            kPrefillMLPK512FragmentNativeDownCodeBytes ||
        layout.down_scale_bytes !=
            kPrefillMLPK512FragmentNativeDownScaleBytes ||
        !range_valid(layout.gateup_code_offset,
                     layout.gateup_code_bytes) ||
        !range_valid(layout.gateup_scale_offset,
                     layout.gateup_scale_bytes) ||
        !range_valid(layout.down_code_offset, layout.down_code_bytes) ||
        !range_valid(layout.down_scale_offset, layout.down_scale_bytes)) {
      return false;
    }
    const std::uintptr_t gateup_code_address =
        arena_address + layout.gateup_code_offset;
    const std::uintptr_t gateup_scale_address =
        arena_address + layout.gateup_scale_offset;
    const std::uintptr_t down_code_address =
        arena_address + layout.down_code_offset;
    const std::uintptr_t down_scale_address =
        arena_address + layout.down_scale_offset;
    if (gateup_code_address >= arena_end ||
        gateup_code_address % 16U != 0U ||
        gateup_scale_address >= arena_end ||
        gateup_scale_address % alignof(std::uint16_t) != 0U ||
        down_code_address >= arena_end || down_code_address % 16U != 0U ||
        down_scale_address >= arena_end ||
        down_scale_address % alignof(std::uint16_t) != 0U) {
      return false;
    }
    validated[layer_index] = {
        reinterpret_cast<const std::uint8_t*>(gateup_code_address),
        reinterpret_cast<const std::uint16_t*>(gateup_scale_address),
        reinterpret_cast<const std::uint8_t*>(down_code_address),
        reinterpret_cast<const std::uint16_t*>(down_scale_address),
        static_cast<std::size_t>(layout.gateup_code_bytes),
        static_cast<std::size_t>(layout.gateup_scale_bytes /
                                 sizeof(std::uint16_t)),
        static_cast<std::size_t>(layout.down_code_bytes),
        static_cast<std::size_t>(layout.down_scale_bytes /
                                 sizeof(std::uint16_t)),
        static_cast<float>(gate_calibration.activation_clip_ratio),
        static_cast<float>(down_calibration.activation_clip_ratio),
        PrefillMLPK512CompositeLayout::
            kPairedGateUpFragmentNativeDown};
    if (!validated[layer_index].attached()) {
      return false;
    }
  }

  clear_all();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    layers_[layer_index].prefill_mlp_k512_fragment_native =
        validated[layer_index];
  }
  return true;
}

bool ModelWeights::
attach_prefill_mlp_k512_paired_gateup_canonical_down_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillMLPK512PairedGateUpCanonicalDownManifest* const manifest,
    const PrefillMLPK512OverlayManifest* const source_v1_manifest,
    const PrefillMLPK512OverlayPolicy* const source_v1_policy) noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      layer.prefill_mlp_k512_fragment_native = {};
    }
  };
  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      source_v1_manifest == nullptr && source_v1_policy == nullptr) {
    clear_all();
    return true;
  }
  if (std::any_of(
          layers_.begin(), layers_.end(), [](const DecoderLayerWeights& layer) {
            return !layer.prefill_mlp_factorized_lane_r1.empty() ||
                   !layer.prefill_mlp_factorized_lane_r4.empty();
          })) {
    return false;
  }

  for (const DecoderLayerWeights& layer : layers_) {
    const auto& composite = layer.prefill_mlp_k512_fragment_native;
    if (!composite.empty() &&
        composite.physical_layout !=
            PrefillMLPK512CompositeLayout::
                kPairedGateUpCanonicalV1Down) {
      return false;
    }
  }
  const auto v1_projection_attached = [](const NvFp4LinearWeight* const value)
      noexcept {
    return value != nullptr &&
           (value->prefill_mlp_k512_weight != nullptr ||
            value->prefill_mlp_k512_scales != nullptr ||
            value->prefill_mlp_k512_activation_clip_ratio != 0.0F);
  };
  for (DecoderLayerWeights& layer : layers_) {
    if (v1_projection_attached(nvfp4_gate_projection(layer)) ||
        v1_projection_attached(nvfp4_up_projection(layer)) ||
        v1_projection_attached(nvfp4_down_projection(layer))) {
      return false;
    }
  }
  if (arena == nullptr || manifest == nullptr ||
      source_v1_manifest == nullptr || source_v1_policy == nullptr ||
      arena_bytes != kPrefillMLPK512FragmentNativePayloadBytes ||
      manifest->payload_bytes != arena_bytes ||
      manifest->layer_count != kPrefillMLPK512FragmentNativeLayerCount ||
      source_v1_manifest->payload_bytes !=
          kPrefillMLPK512OverlayPayloadBytes ||
      source_v1_manifest->projections.size() !=
          kPrefillMLPK512OverlayProjectionCount ||
      source_v1_policy->projections.size() !=
          source_v1_manifest->projections.size()) {
    return false;
  }

  const auto same_base_local = [](
                                   const PrefillMLPK512BaseBinding& first,
                                   const PrefillMLPK512BaseBinding& second)
      noexcept {
    return first.physical_layout == second.physical_layout &&
           first.manifest_sha256 == second.manifest_sha256 &&
           first.policy_sha256 == second.policy_sha256 &&
           first.payload_sha256 == second.payload_sha256;
  };
  const auto lower_sha256_local = [](const std::string_view value) noexcept {
    return value.size() == 64U &&
           std::all_of(value.begin(), value.end(), [](const char character) {
             return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
           });
  };
  try {
    if (!validate_prefill_mlp_k512_paired_gateup_canonical_down_manifest(
            *manifest) ||
        !validate_prefill_mlp_k512_overlay_manifest(*source_v1_manifest) ||
        manifest->source_checkpoint_id !=
            source_v1_manifest->source_checkpoint_id ||
        manifest->source_config_sha256 !=
            source_v1_manifest->source_config_sha256 ||
        manifest->source_index_sha256 !=
            source_v1_manifest->source_index_sha256 ||
        !same_base_local(manifest->required_base,
                         source_v1_manifest->required_base) ||
        manifest->source_v1.physical_layout !=
            source_v1_manifest->physical_layout ||
        manifest->source_v1.manifest_sha256 !=
            source_v1_manifest->manifest_sha256 ||
        manifest->source_v1.payload_bytes !=
            source_v1_manifest->payload_bytes ||
        source_v1_policy->version_major !=
            source_v1_manifest->version_major ||
        source_v1_policy->version_minor !=
            source_v1_manifest->version_minor ||
        source_v1_policy->physical_layout !=
            source_v1_manifest->physical_layout ||
        source_v1_policy->source_checkpoint_id !=
            source_v1_manifest->source_checkpoint_id ||
        source_v1_policy->source_config_sha256 !=
            source_v1_manifest->source_config_sha256 ||
        source_v1_policy->source_index_sha256 !=
            source_v1_manifest->source_index_sha256 ||
        source_v1_policy->manifest_sha256 !=
            source_v1_manifest->manifest_sha256 ||
        !same_base_local(source_v1_policy->required_base,
                         source_v1_manifest->required_base) ||
        !lower_sha256_local(source_v1_policy->policy_sha256) ||
        source_v1_policy->policy_bytes == 0U ||
        manifest->source_v1.policy_sha256 !=
            source_v1_policy->policy_sha256 ||
        manifest->source_v1.policy_bytes != source_v1_policy->policy_bytes ||
        !prefill_a4_attachment_complete_ ||
        std::string_view(prefill_a4_attachment_manifest_sha256_.data(),
                         prefill_a4_attachment_manifest_sha256_.size()) !=
            manifest->required_base.manifest_sha256 ||
        std::string_view(prefill_a4_attachment_policy_sha256_.data(),
                         prefill_a4_attachment_policy_sha256_.size()) !=
            manifest->required_base.policy_sha256 ||
        std::string_view(prefill_a4_attachment_payload_sha256_.data(),
                         prefill_a4_attachment_payload_sha256_.size()) !=
            manifest->required_base.payload_sha256) {
      return false;
    }
  } catch (...) {
    return false;
  }

  constexpr std::uintptr_t kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % 256U != 0U || arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;
  std::array<PrefillMLPK512FragmentNativeCompositeView,
             kPrefillMLPK512FragmentNativeLayerCount>
      validated{};
  for (std::size_t layer_index = 0U;
       layer_index < kPrefillMLPK512FragmentNativeLayerCount;
       ++layer_index) {
    const std::size_t gate_index = 3U * layer_index;
    const std::size_t up_index = gate_index + 1U;
    const std::size_t down_index = gate_index + 2U;
    const PrefillMLPK512OverlayEntry& gate_entry =
        source_v1_manifest->projections[gate_index];
    const PrefillMLPK512OverlayEntry& up_entry =
        source_v1_manifest->projections[up_index];
    const PrefillMLPK512OverlayEntry& down_entry =
        source_v1_manifest->projections[down_index];
    const PrefillMLPK512OverlayCalibration& gate_calibration =
        source_v1_policy->projections[gate_index];
    const PrefillMLPK512OverlayCalibration& up_calibration =
        source_v1_policy->projections[up_index];
    const PrefillMLPK512OverlayCalibration& down_calibration =
        source_v1_policy->projections[down_index];
    const auto calibration_valid = [](
                                       const PrefillMLPK512OverlayEntry& entry,
                                       const PrefillMLPK512OverlayCalibration&
                                           calibration) noexcept {
      return calibration.ordinal == entry.ordinal &&
             calibration.source_module == entry.source_module &&
             calibration.source_sha256 == entry.source_sha256 &&
             calibration.activation_scale_group_size ==
                 kPrefillMLPK512OverlayScaleK &&
             std::isfinite(calibration.weight_clip_ratio) &&
             calibration.weight_clip_ratio >= kPrefillA4MinimumClipRatio &&
             calibration.weight_clip_ratio <= 1.0 &&
             std::isfinite(calibration.activation_clip_ratio) &&
             calibration.activation_clip_ratio >=
                 kPrefillA4MinimumClipRatio &&
             calibration.activation_clip_ratio <= 1.0;
    };
    if (!calibration_valid(gate_entry, gate_calibration) ||
        !calibration_valid(up_entry, up_calibration) ||
        !calibration_valid(down_entry, down_calibration) ||
        static_cast<float>(gate_calibration.activation_clip_ratio) !=
            static_cast<float>(up_calibration.activation_clip_ratio)) {
      return false;
    }

    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    const auto projection_valid = [manifest](
                                      const NvFp4LinearWeight* const weight,
                                      const std::size_t output_size,
                                      const std::size_t input_size) noexcept {
      if (!has_valid_nvfp4_payload(weight) ||
          weight->output_size != output_size ||
          weight->input_size != input_size ||
          weight->prefill_marlin_weight != nullptr ||
          weight->prefill_marlin_scales != nullptr ||
          weight->prefill_marlin_global_scale != nullptr) {
        return false;
      }
      return weight->prefill_a4_weight != nullptr &&
             weight->prefill_a4_scales != nullptr &&
             prefill_mlp_k512_base_layout_matches_contract(
                 manifest->required_base.physical_layout,
                 weight->prefill_a4_sidecar_kind,
                 weight->prefill_a4_packed_k_group_size,
                 weight->prefill_a4_scale_group_size) &&
             weight->prefill_a4_activation_clip_ratio > 0.0F &&
             weight->prefill_a4_activation_clip_ratio <= 1.0F;
    };
    if (!projection_valid(gate, kPrefillMLPK512OverlayGateUpOutputSize,
                          kPrefillMLPK512OverlayGateUpInputSize) ||
        !projection_valid(up, kPrefillMLPK512OverlayGateUpOutputSize,
                          kPrefillMLPK512OverlayGateUpInputSize) ||
        !projection_valid(down, kPrefillMLPK512OverlayDownOutputSize,
                          kPrefillMLPK512OverlayDownInputSize)) {
      return false;
    }

    const PrefillMLPK512FragmentNativeLayerView layout =
        prefill_mlp_k512_fragment_native_layer_view(layer_index);
    const auto range_valid = [arena_bytes](const std::uint64_t offset,
                                           const std::uint64_t bytes) noexcept {
      return offset < arena_bytes && bytes <= arena_bytes - offset;
    };
    if (!layout.valid || layout.layer_index != layer_index ||
        !range_valid(layout.gateup_code_offset, layout.gateup_code_bytes) ||
        !range_valid(layout.gateup_scale_offset, layout.gateup_scale_bytes) ||
        !range_valid(layout.down_code_offset, layout.down_code_bytes) ||
        !range_valid(layout.down_scale_offset, layout.down_scale_bytes)) {
      return false;
    }
    const std::uintptr_t gateup_code_address =
        arena_address + layout.gateup_code_offset;
    const std::uintptr_t gateup_scale_address =
        arena_address + layout.gateup_scale_offset;
    const std::uintptr_t down_code_address =
        arena_address + layout.down_code_offset;
    const std::uintptr_t down_scale_address =
        arena_address + layout.down_scale_offset;
    if (gateup_code_address >= arena_end ||
        gateup_code_address % 16U != 0U ||
        gateup_scale_address >= arena_end ||
        gateup_scale_address % alignof(std::uint16_t) != 0U ||
        down_code_address >= arena_end || down_code_address % 16U != 0U ||
        down_scale_address >= arena_end ||
        down_scale_address % alignof(std::uint16_t) != 0U) {
      return false;
    }
    validated[layer_index] = {
        reinterpret_cast<const std::uint8_t*>(gateup_code_address),
        reinterpret_cast<const std::uint16_t*>(gateup_scale_address),
        reinterpret_cast<const std::uint8_t*>(down_code_address),
        reinterpret_cast<const std::uint16_t*>(down_scale_address),
        static_cast<std::size_t>(layout.gateup_code_bytes),
        static_cast<std::size_t>(layout.gateup_scale_bytes /
                                 sizeof(std::uint16_t)),
        static_cast<std::size_t>(layout.down_code_bytes),
        static_cast<std::size_t>(layout.down_scale_bytes /
                                 sizeof(std::uint16_t)),
        static_cast<float>(gate_calibration.activation_clip_ratio),
        static_cast<float>(down_calibration.activation_clip_ratio),
        PrefillMLPK512CompositeLayout::kPairedGateUpCanonicalV1Down};
    if (!validated[layer_index].attached()) {
      return false;
    }
  }

  clear_all();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    layers_[layer_index].prefill_mlp_k512_fragment_native =
        validated[layer_index];
  }
  return true;
}

bool ModelWeights::
attach_prefill_mlp_k512_projection_major_gateup_canonical_down_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillMLPK512ProjectionMajorGateUpCanonicalDownManifest* const
        manifest,
    const PrefillMLPK512OverlayManifest* const source_v1_manifest,
    const PrefillMLPK512OverlayPolicy* const source_v1_policy) noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      layer.prefill_mlp_k512_fragment_native = {};
    }
  };
  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      source_v1_manifest == nullptr && source_v1_policy == nullptr) {
    clear_all();
    return true;
  }
  if (std::any_of(
          layers_.begin(), layers_.end(), [](const DecoderLayerWeights& layer) {
            return !layer.prefill_mlp_factorized_lane_r1.empty() ||
                   !layer.prefill_mlp_factorized_lane_r4.empty();
          })) {
    return false;
  }

  for (const DecoderLayerWeights& layer : layers_) {
    const auto& composite = layer.prefill_mlp_k512_fragment_native;
    if (!composite.empty() &&
        composite.physical_layout !=
            PrefillMLPK512CompositeLayout::
                kProjectionMajorGateUpCanonicalV1Down) {
      return false;
    }
  }
  const auto v1_projection_attached = [](const NvFp4LinearWeight* const value)
      noexcept {
    return value != nullptr &&
           (value->prefill_mlp_k512_weight != nullptr ||
            value->prefill_mlp_k512_scales != nullptr ||
            value->prefill_mlp_k512_activation_clip_ratio != 0.0F);
  };
  for (DecoderLayerWeights& layer : layers_) {
    if (v1_projection_attached(nvfp4_gate_projection(layer)) ||
        v1_projection_attached(nvfp4_up_projection(layer)) ||
        v1_projection_attached(nvfp4_down_projection(layer))) {
      return false;
    }
  }
  if (arena == nullptr || manifest == nullptr ||
      source_v1_manifest == nullptr || source_v1_policy == nullptr ||
      arena_bytes != kPrefillMLPK512FragmentNativePayloadBytes ||
      manifest->payload_bytes != arena_bytes ||
      manifest->layer_count != kPrefillMLPK512FragmentNativeLayerCount ||
      source_v1_manifest->payload_bytes !=
          kPrefillMLPK512OverlayPayloadBytes ||
      source_v1_manifest->projections.size() !=
          kPrefillMLPK512OverlayProjectionCount ||
      source_v1_policy->projections.size() !=
          source_v1_manifest->projections.size()) {
    return false;
  }

  const auto same_base_local = [](
                                   const PrefillMLPK512BaseBinding& first,
                                   const PrefillMLPK512BaseBinding& second)
      noexcept {
    return first.physical_layout == second.physical_layout &&
           first.manifest_sha256 == second.manifest_sha256 &&
           first.policy_sha256 == second.policy_sha256 &&
           first.payload_sha256 == second.payload_sha256;
  };
  const auto lower_sha256_local = [](const std::string_view value) noexcept {
    return value.size() == 64U &&
           std::all_of(value.begin(), value.end(), [](const char character) {
             return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
           });
  };
  try {
    if (!validate_prefill_mlp_k512_projection_major_gateup_canonical_down_manifest(
            *manifest) ||
        !validate_prefill_mlp_k512_overlay_manifest(*source_v1_manifest) ||
        manifest->source_checkpoint_id !=
            source_v1_manifest->source_checkpoint_id ||
        manifest->source_config_sha256 !=
            source_v1_manifest->source_config_sha256 ||
        manifest->source_index_sha256 !=
            source_v1_manifest->source_index_sha256 ||
        !same_base_local(manifest->required_base,
                         source_v1_manifest->required_base) ||
        manifest->source_v1.physical_layout !=
            source_v1_manifest->physical_layout ||
        manifest->source_v1.manifest_sha256 !=
            source_v1_manifest->manifest_sha256 ||
        manifest->source_v1.payload_bytes !=
            source_v1_manifest->payload_bytes ||
        source_v1_policy->version_major !=
            source_v1_manifest->version_major ||
        source_v1_policy->version_minor !=
            source_v1_manifest->version_minor ||
        source_v1_policy->physical_layout !=
            source_v1_manifest->physical_layout ||
        source_v1_policy->source_checkpoint_id !=
            source_v1_manifest->source_checkpoint_id ||
        source_v1_policy->source_config_sha256 !=
            source_v1_manifest->source_config_sha256 ||
        source_v1_policy->source_index_sha256 !=
            source_v1_manifest->source_index_sha256 ||
        source_v1_policy->manifest_sha256 !=
            source_v1_manifest->manifest_sha256 ||
        !same_base_local(source_v1_policy->required_base,
                         source_v1_manifest->required_base) ||
        !lower_sha256_local(source_v1_policy->policy_sha256) ||
        source_v1_policy->policy_bytes == 0U ||
        manifest->source_v1.policy_sha256 !=
            source_v1_policy->policy_sha256 ||
        manifest->source_v1.policy_bytes != source_v1_policy->policy_bytes ||
        !prefill_a4_attachment_complete_ ||
        std::string_view(prefill_a4_attachment_manifest_sha256_.data(),
                         prefill_a4_attachment_manifest_sha256_.size()) !=
            manifest->required_base.manifest_sha256 ||
        std::string_view(prefill_a4_attachment_policy_sha256_.data(),
                         prefill_a4_attachment_policy_sha256_.size()) !=
            manifest->required_base.policy_sha256 ||
        std::string_view(prefill_a4_attachment_payload_sha256_.data(),
                         prefill_a4_attachment_payload_sha256_.size()) !=
            manifest->required_base.payload_sha256) {
      return false;
    }
  } catch (...) {
    return false;
  }

  constexpr std::uintptr_t kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % 256U != 0U || arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;
  std::array<PrefillMLPK512FragmentNativeCompositeView,
             kPrefillMLPK512FragmentNativeLayerCount>
      validated{};
  for (std::size_t layer_index = 0U;
       layer_index < kPrefillMLPK512FragmentNativeLayerCount;
       ++layer_index) {
    const std::size_t gate_index = 3U * layer_index;
    const std::size_t up_index = gate_index + 1U;
    const std::size_t down_index = gate_index + 2U;
    const PrefillMLPK512OverlayEntry& gate_entry =
        source_v1_manifest->projections[gate_index];
    const PrefillMLPK512OverlayEntry& up_entry =
        source_v1_manifest->projections[up_index];
    const PrefillMLPK512OverlayEntry& down_entry =
        source_v1_manifest->projections[down_index];
    const PrefillMLPK512OverlayCalibration& gate_calibration =
        source_v1_policy->projections[gate_index];
    const PrefillMLPK512OverlayCalibration& up_calibration =
        source_v1_policy->projections[up_index];
    const PrefillMLPK512OverlayCalibration& down_calibration =
        source_v1_policy->projections[down_index];
    const auto calibration_valid = [](
                                       const PrefillMLPK512OverlayEntry& entry,
                                       const PrefillMLPK512OverlayCalibration&
                                           calibration) noexcept {
      return calibration.ordinal == entry.ordinal &&
             calibration.source_module == entry.source_module &&
             calibration.source_sha256 == entry.source_sha256 &&
             calibration.activation_scale_group_size ==
                 kPrefillMLPK512OverlayScaleK &&
             std::isfinite(calibration.weight_clip_ratio) &&
             calibration.weight_clip_ratio >= kPrefillA4MinimumClipRatio &&
             calibration.weight_clip_ratio <= 1.0 &&
             std::isfinite(calibration.activation_clip_ratio) &&
             calibration.activation_clip_ratio >=
                 kPrefillA4MinimumClipRatio &&
             calibration.activation_clip_ratio <= 1.0;
    };
    if (!calibration_valid(gate_entry, gate_calibration) ||
        !calibration_valid(up_entry, up_calibration) ||
        !calibration_valid(down_entry, down_calibration) ||
        static_cast<float>(gate_calibration.activation_clip_ratio) !=
            static_cast<float>(up_calibration.activation_clip_ratio)) {
      return false;
    }

    DecoderLayerWeights& layer = layers_[layer_index];
    NvFp4LinearWeight* const gate = nvfp4_gate_projection(layer);
    NvFp4LinearWeight* const up = nvfp4_up_projection(layer);
    NvFp4LinearWeight* const down = nvfp4_down_projection(layer);
    const auto projection_valid = [manifest](
                                      const NvFp4LinearWeight* const weight,
                                      const std::size_t output_size,
                                      const std::size_t input_size) noexcept {
      if (!has_valid_nvfp4_payload(weight) ||
          weight->output_size != output_size ||
          weight->input_size != input_size ||
          weight->prefill_marlin_weight != nullptr ||
          weight->prefill_marlin_scales != nullptr ||
          weight->prefill_marlin_global_scale != nullptr) {
        return false;
      }
      return weight->prefill_a4_weight != nullptr &&
             weight->prefill_a4_scales != nullptr &&
             prefill_mlp_k512_base_layout_matches_contract(
                 manifest->required_base.physical_layout,
                 weight->prefill_a4_sidecar_kind,
                 weight->prefill_a4_packed_k_group_size,
                 weight->prefill_a4_scale_group_size) &&
             weight->prefill_a4_activation_clip_ratio > 0.0F &&
             weight->prefill_a4_activation_clip_ratio <= 1.0F;
    };
    if (!projection_valid(gate, kPrefillMLPK512OverlayGateUpOutputSize,
                          kPrefillMLPK512OverlayGateUpInputSize) ||
        !projection_valid(up, kPrefillMLPK512OverlayGateUpOutputSize,
                          kPrefillMLPK512OverlayGateUpInputSize) ||
        !projection_valid(down, kPrefillMLPK512OverlayDownOutputSize,
                          kPrefillMLPK512OverlayDownInputSize)) {
      return false;
    }

    const PrefillMLPK512FragmentNativeLayerView layout =
        prefill_mlp_k512_fragment_native_layer_view(layer_index);
    const auto range_valid = [arena_bytes](const std::uint64_t offset,
                                           const std::uint64_t bytes) noexcept {
      return offset < arena_bytes && bytes <= arena_bytes - offset;
    };
    if (!layout.valid || layout.layer_index != layer_index ||
        layout.gateup_code_bytes !=
            kPrefillMLPK512FragmentNativeGateUpCodeBytes ||
        layout.gateup_scale_bytes !=
            kPrefillMLPK512FragmentNativeGateUpScaleBytes ||
        layout.down_code_bytes !=
            kPrefillMLPK512FragmentNativeDownCodeBytes ||
        layout.down_scale_bytes !=
            kPrefillMLPK512FragmentNativeDownScaleBytes ||
        !range_valid(layout.gateup_code_offset, layout.gateup_code_bytes) ||
        !range_valid(layout.gateup_scale_offset,
                     layout.gateup_scale_bytes) ||
        !range_valid(layout.down_code_offset, layout.down_code_bytes) ||
        !range_valid(layout.down_scale_offset, layout.down_scale_bytes)) {
      return false;
    }
    const std::uintptr_t gateup_code_address =
        arena_address + layout.gateup_code_offset;
    const std::uintptr_t gateup_scale_address =
        arena_address + layout.gateup_scale_offset;
    const std::uintptr_t down_code_address =
        arena_address + layout.down_code_offset;
    const std::uintptr_t down_scale_address =
        arena_address + layout.down_scale_offset;
    if (gateup_code_address >= arena_end ||
        gateup_code_address % 16U != 0U ||
        gateup_scale_address >= arena_end ||
        gateup_scale_address % alignof(std::uint16_t) != 0U ||
        down_code_address >= arena_end || down_code_address % 16U != 0U ||
        down_scale_address >= arena_end ||
        down_scale_address % alignof(std::uint16_t) != 0U) {
      return false;
    }
    validated[layer_index] = {
        reinterpret_cast<const std::uint8_t*>(gateup_code_address),
        reinterpret_cast<const std::uint16_t*>(gateup_scale_address),
        reinterpret_cast<const std::uint8_t*>(down_code_address),
        reinterpret_cast<const std::uint16_t*>(down_scale_address),
        static_cast<std::size_t>(layout.gateup_code_bytes),
        static_cast<std::size_t>(layout.gateup_scale_bytes /
                                 sizeof(std::uint16_t)),
        static_cast<std::size_t>(layout.down_code_bytes),
        static_cast<std::size_t>(layout.down_scale_bytes /
                                 sizeof(std::uint16_t)),
        static_cast<float>(gate_calibration.activation_clip_ratio),
        static_cast<float>(down_calibration.activation_clip_ratio),
        PrefillMLPK512CompositeLayout::
            kProjectionMajorGateUpCanonicalV1Down};
    if (!validated[layer_index].attached()) {
      return false;
    }
  }

  clear_all();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    layers_[layer_index].prefill_mlp_k512_fragment_native =
        validated[layer_index];
  }
  return true;
}

bool ModelWeights::attach_prefill_mlp_factorized_lane_r1_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillMLPFactorizedLaneOverlayManifestBinding* const manifest,
    const PrefillMLPFactorizedLaneOverlayPolicyBinding* const policy) noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      layer.prefill_mlp_factorized_lane_r1 = {};
    }
  };
  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      policy == nullptr) {
    clear_all();
    return true;
  }

  if (std::any_of(
          layers_.begin(), layers_.end(), [](const DecoderLayerWeights& layer) {
            return !layer.prefill_mlp_factorized_lane_r4.empty();
          })) {
    return false;
  }

  // No factorized view may coexist with any canonical or composite MLP K512
  // publication.  Count malformed/partial legacy views as attached too: a
  // caller must explicitly detach the conflicting layout before R1 can be
  // published.
  for (const DecoderLayerWeights& layer : layers_) {
    if (!layer.prefill_mlp_k512_fragment_native.empty()) {
      return false;
    }
    for (const LinearWeight* const binding :
         {&layer.mlp.gate_proj, &layer.mlp.up_proj,
          &layer.mlp.down_proj}) {
      const auto* const projection =
          std::get_if<NvFp4LinearWeight>(binding);
      if (projection == nullptr ||
          projection->prefill_mlp_k512_weight != nullptr ||
          projection->prefill_mlp_k512_scales != nullptr ||
          projection->prefill_mlp_k512_activation_clip_ratio != 0.0F) {
        return false;
      }
    }
  }

  const auto same_base = [](
                             const PrefillMLPFactorizedLaneBaseK256Binding&
                                 left,
                             const PrefillMLPFactorizedLaneBaseK256Binding&
                                 right) noexcept {
    return left.physical_layout == right.physical_layout &&
           left.packed_k_group_size == right.packed_k_group_size &&
           left.scale_group_size == right.scale_group_size &&
           left.manifest_sha256 == right.manifest_sha256 &&
           left.policy_sha256 == right.policy_sha256 &&
           left.payload_sha256 == right.payload_sha256 &&
           left.receipt_sha256 == right.receipt_sha256;
  };
  const auto lower_sha256 = [](const std::string_view value) noexcept {
    return value.size() == 64U &&
           std::all_of(value.begin(), value.end(), [](const char character) {
             return (character >= '0' && character <= '9') ||
                    (character >= 'a' && character <= 'f');
           });
  };
  const auto same_factor = [](
                               const PrefillMLPFactorizedLaneFactorSourceBinding&
                                   left,
                               const PrefillMLPFactorizedLaneFactorSourceBinding&
                                   right) noexcept {
    return left.scheme == right.scheme && left.path == right.path &&
           left.sha256 == right.sha256 &&
           left.element_count == right.element_count;
  };
  const auto valid_clip = [](const double value) noexcept {
    return std::isfinite(value) && value >= kPrefillA4MinimumClipRatio &&
           value <= 1.0 &&
           static_cast<float>(value) >=
               static_cast<float>(kPrefillA4MinimumClipRatio) &&
           static_cast<float>(value) <= 1.0F;
  };

  const PrefillMLPFactorizedLaneOverlayLayoutPlan plan =
      prefill_mlp_factorized_lane_overlay_layout_plan(
          kPrefillMLPFactorizedLaneR1LaneCount);
  if (arena == nullptr || manifest == nullptr || policy == nullptr || !plan ||
      arena_bytes != kPrefillMLPFactorizedLaneR1PayloadBytes ||
      arena_bytes != plan.payload_bytes ||
      manifest->payload_bytes != arena_bytes ||
      manifest->projections.size() !=
          kPrefillMLPFactorizedLaneProjectionCount ||
      policy->projections.size() != manifest->projections.size() ||
      policy->version_major != manifest->version_major ||
      policy->version_minor != manifest->version_minor ||
      policy->physical_layout != manifest->physical_layout ||
      policy->source_checkpoint_id != manifest->source_checkpoint_id ||
      policy->source_config_sha256 != manifest->source_config_sha256 ||
      policy->source_index_sha256 != manifest->source_index_sha256 ||
      policy->manifest_sha256 != manifest->manifest_sha256 ||
      !same_base(policy->required_base_k256,
                 manifest->required_base_k256) ||
      policy->lane_count != kPrefillMLPFactorizedLaneR1LaneCount ||
      !lower_sha256(policy->policy_sha256) || policy->policy_bytes == 0U ||
      manifest->required_base_k256.physical_layout !=
          kPrefillMLPFactorizedLaneRequiredBaseK256Layout ||
      manifest->required_base_k256.packed_k_group_size !=
          kPrefillMLPFactorizedLaneRequiredBasePackedK ||
      manifest->required_base_k256.scale_group_size !=
          kPrefillMLPFactorizedLaneRequiredBaseScaleK ||
      !lower_sha256(manifest->required_base_k256.manifest_sha256) ||
      !lower_sha256(manifest->required_base_k256.policy_sha256) ||
      !lower_sha256(manifest->required_base_k256.payload_sha256) ||
      !lower_sha256(manifest->required_base_k256.receipt_sha256) ||
      !prefill_a4_attachment_complete_ ||
      std::string_view(prefill_a4_attachment_manifest_sha256_.data(),
                       prefill_a4_attachment_manifest_sha256_.size()) !=
          manifest->required_base_k256.manifest_sha256 ||
      std::string_view(prefill_a4_attachment_policy_sha256_.data(),
                       prefill_a4_attachment_policy_sha256_.size()) !=
          manifest->required_base_k256.policy_sha256 ||
      std::string_view(prefill_a4_attachment_payload_sha256_.data(),
                       prefill_a4_attachment_payload_sha256_.size()) !=
          manifest->required_base_k256.payload_sha256) {
    return false;
  }
  try {
    if (!validate_prefill_mlp_factorized_lane_r1_manifest(*manifest)) {
      return false;
    }
  } catch (...) {
    return false;
  }

  constexpr std::string_view kGateUpIdentityDigest =
      "42010c1c68b632e2ab15c82bca6edef2cac2026c889dd0202d609602b756f568";
  constexpr std::string_view kDownIdentityDigest =
      "15cd4df15b3bcb53816bb119e9d52efa3c0bbee237fa17c5a7c351dc9bfdcbcd";
  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % kPrefillA4FactorizedLaneMinimumAlignment != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  std::array<PrefillMLPFactorizedLaneLayerView,
             kPrefillMLPFactorizedLaneLayerCount>
      validated{};
  for (std::size_t index = 0U; index < manifest->projections.size();
       ++index) {
    const PrefillMLPFactorizedLaneManifestProjection& entry =
        manifest->projections[index];
    const PrefillMLPFactorizedLaneProjectionCalibrationBinding& calibration =
        policy->projections[index];
    const PrefillA4FactorizedLaneProjectionLayoutPlan& projection_plan =
        entry.family == PrefillMLPFactorizedLaneProjectionFamily::kDown
            ? plan.down
            : entry.family == PrefillMLPFactorizedLaneProjectionFamily::kUp
                  ? plan.up
                  : plan.gate;
    const std::string_view expected_factor_digest =
        entry.family == PrefillMLPFactorizedLaneProjectionFamily::kDown
            ? kDownIdentityDigest
            : kGateUpIdentityDigest;
    if (entry.ordinal != index || entry.layer_index >= layers_.size() ||
        entry.payload_offset > arena_bytes ||
        entry.payload_bytes > arena_bytes - entry.payload_offset ||
        entry.payload_bytes != projection_plan.projection_bytes ||
        calibration.ordinal != entry.ordinal ||
        calibration.source_module != entry.source_module ||
        calibration.source_sha256 != entry.source_sha256 ||
        !valid_clip(calibration.weight_clip_ratio) ||
        !valid_clip(calibration.activation_clip_ratio) ||
        calibration.factor_source.scheme !=
            kPrefillMLPFactorizedLaneR1FactorScheme ||
        !calibration.factor_source.path.empty() ||
        calibration.factor_source.sha256 != expected_factor_digest ||
        calibration.factor_source.element_count != entry.input_size) {
      return false;
    }

    DecoderLayerWeights& layer = layers_[entry.layer_index];
    LinearWeight* target = nullptr;
    PrefillMLPFactorizedLaneLinearSidecarView* destination = nullptr;
    switch (entry.family) {
      case PrefillMLPFactorizedLaneProjectionFamily::kGate:
        target = &layer.mlp.gate_proj;
        destination = &validated[entry.layer_index].gate;
        break;
      case PrefillMLPFactorizedLaneProjectionFamily::kUp:
        target = &layer.mlp.up_proj;
        destination = &validated[entry.layer_index].up;
        break;
      case PrefillMLPFactorizedLaneProjectionFamily::kDown:
        target = &layer.mlp.down_proj;
        destination = &validated[entry.layer_index].down;
        break;
      default:
        return false;
    }
    const auto* const nvfp4 = std::get_if<NvFp4LinearWeight>(target);
    const PrefillA4LinearSidecarView base =
        prefill_a4_sidecar_view(*target);
    if (destination == nullptr || !has_valid_nvfp4_payload(nvfp4) ||
        nvfp4->output_size != entry.output_size ||
        nvfp4->input_size != entry.input_size || !base.attached() ||
        base.sidecar_kind != PrefillSidecarKind::kA4K256 ||
        base.packed_k_group_size !=
            kPrefillMLPFactorizedLaneRequiredBasePackedK ||
        base.scale_group_size !=
            kPrefillMLPFactorizedLaneRequiredBaseScaleK) {
      return false;
    }

    const std::uintptr_t projection_address =
        arena_address + entry.payload_offset;
    const std::uintptr_t weight_address =
        projection_address + projection_plan.packed_weight_offset;
    const std::uintptr_t scale_address =
        projection_address + projection_plan.weight_scale_offset;
    const std::uintptr_t inverse_alpha_address =
        projection_address + projection_plan.inverse_alpha_offset;
    const auto range_valid = [arena_end](const std::uintptr_t address,
                                         const std::uint64_t bytes) noexcept {
      return address < arena_end && bytes <= arena_end - address;
    };
    if (weight_address % 16U != 0U || scale_address % 16U != 0U ||
        inverse_alpha_address % 16U != 0U ||
        !range_valid(weight_address, projection_plan.packed_weight_bytes) ||
        !range_valid(scale_address, projection_plan.weight_scale_bytes) ||
        !range_valid(inverse_alpha_address,
                     projection_plan.inverse_alpha_bytes) ||
        projection_plan.packed_weight_bytes >
            std::numeric_limits<std::size_t>::max() ||
        projection_plan.weight_scale_elements >
            std::numeric_limits<std::size_t>::max() ||
        projection_plan.inverse_alpha_elements >
            std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    *destination = {
        reinterpret_cast<const std::uint8_t*>(weight_address),
        static_cast<std::size_t>(projection_plan.packed_weight_bytes),
        reinterpret_cast<const std::uint16_t*>(scale_address),
        static_cast<std::size_t>(projection_plan.weight_scale_elements),
        reinterpret_cast<const float*>(inverse_alpha_address),
        static_cast<std::size_t>(projection_plan.inverse_alpha_elements),
        static_cast<std::size_t>(entry.output_size),
        static_cast<std::size_t>(entry.input_size),
        kPrefillMLPFactorizedLaneR1LaneCount,
        static_cast<float>(calibration.activation_clip_ratio)};
    if (!destination->attached()) {
      return false;
    }
  }

  for (std::size_t layer_index = 0U; layer_index < validated.size();
       ++layer_index) {
    const PrefillMLPFactorizedLaneProjectionCalibrationBinding& gate =
        policy->projections[layer_index * 3U];
    const PrefillMLPFactorizedLaneProjectionCalibrationBinding& up =
        policy->projections[layer_index * 3U + 1U];
    if (!same_factor(gate.factor_source, up.factor_source) ||
        static_cast<float>(gate.activation_clip_ratio) !=
            static_cast<float>(up.activation_clip_ratio) ||
        !validated[layer_index].attached()) {
      return false;
    }
  }

  clear_all();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    layers_[layer_index].prefill_mlp_factorized_lane_r1 =
        validated[layer_index];
  }
  return true;
}

bool ModelWeights::attach_prefill_mlp_factorized_lane_r4_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillMLPFactorizedLaneR4Manifest* const manifest,
    const PrefillMLPFactorizedLaneR4Policy* const policy) noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      layer.prefill_mlp_factorized_lane_r4 = {};
    }
  };
  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      policy == nullptr) {
    clear_all();
    return true;
  }

  // Partial legacy publications are conflicts too.  A caller must explicitly
  // detach R1/K512 before replacing the independent R4 view family.
  for (const DecoderLayerWeights& layer : layers_) {
    if (!layer.prefill_mlp_factorized_lane_r1.empty() ||
        !layer.prefill_mlp_k512_fragment_native.empty()) {
      return false;
    }
    for (const LinearWeight* const binding :
         {&layer.mlp.gate_proj, &layer.mlp.up_proj,
          &layer.mlp.down_proj}) {
      const auto* const projection =
          std::get_if<NvFp4LinearWeight>(binding);
      if (projection == nullptr ||
          projection->prefill_mlp_k512_weight != nullptr ||
          projection->prefill_mlp_k512_scales != nullptr ||
          projection->prefill_mlp_k512_activation_clip_ratio != 0.0F) {
        return false;
      }
    }
  }

  const auto plan = prefill_mlp_factorized_lane_overlay_layout_plan(
      kPrefillMLPFactorizedLaneR4PublicationLaneCount);
  if (arena == nullptr || manifest == nullptr || policy == nullptr || !plan ||
      arena_bytes !=
          kPrefillMLPFactorizedLaneR4PublicationPayloadBytes ||
      arena_bytes != plan.payload_bytes ||
      manifest->payload_bytes != arena_bytes ||
      manifest->lane_count !=
          kPrefillMLPFactorizedLaneR4PublicationLaneCount ||
      manifest->projections.size() !=
          kPrefillMLPFactorizedLaneProjectionCount ||
      policy->lane_count !=
          kPrefillMLPFactorizedLaneR4PublicationLaneCount ||
      policy->projections.size() != manifest->projections.size()) {
    return false;
  }
  try {
    if (!validate_prefill_mlp_factorized_lane_r4_policy_binding(
             *policy, *manifest)) {
      return false;
    }
  } catch (...) {
    return false;
  }

  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % kPrefillA4FactorizedLaneMinimumAlignment != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;
  const auto valid_clip = [](const double value) noexcept {
    return std::isfinite(value) &&
           value >=
               kPrefillMLPFactorizedLaneR4PublicationMinimumClipRatio &&
           value <= 1.0 &&
           static_cast<float>(value) >=
               static_cast<float>(
                   kPrefillMLPFactorizedLaneR4PublicationMinimumClipRatio) &&
           static_cast<float>(value) <= 1.0F;
  };
  const auto same_factor = [](
                               const PrefillMLPFactorizedLaneR4ProjectionPolicyBinding&
                                   left,
                               const PrefillMLPFactorizedLaneR4ProjectionPolicyBinding&
                                   right) noexcept {
    return left.factor_scheme == right.factor_scheme &&
           left.factor_path == right.factor_path &&
           left.factor_sha256 == right.factor_sha256 &&
           left.factor_element_count == right.factor_element_count;
  };

  std::array<PrefillMLPFactorizedLaneR4LayerView,
             kPrefillMLPFactorizedLaneLayerCount>
      validated{};
  for (std::size_t index = 0U; index < manifest->projections.size();
       ++index) {
    const PrefillMLPFactorizedLaneManifestProjection& entry =
        manifest->projections[index];
    const PrefillMLPFactorizedLaneR4ProjectionPolicyBinding& calibration =
        policy->projections[index];
    const auto expected_family =
        static_cast<PrefillMLPFactorizedLaneProjectionFamily>(index % 3U);
    const std::uint32_t expected_layer =
        static_cast<std::uint32_t>(index / 3U);
    const PrefillA4FactorizedLaneProjectionLayoutPlan& projection_plan =
        entry.family == PrefillMLPFactorizedLaneProjectionFamily::kDown
            ? plan.down
            : entry.family == PrefillMLPFactorizedLaneProjectionFamily::kUp
                  ? plan.up
                  : plan.gate;
    const std::uint64_t expected_offset =
        prefill_mlp_factorized_lane_projection_absolute_offset(
            plan, expected_layer, expected_family);
    const std::string_view factor_path(calibration.factor_path);
    const bool calibrated_factor =
        calibration.factor_scheme ==
        kPrefillMLPFactorizedLaneR4PublicationFactorScheme;
    const bool exact_identity_factor =
        calibration.factor_scheme ==
            kPrefillMLPFactorizedLaneR4IdentityCandidateFactorScheme &&
        ((entry.input_size == 5'120U &&
          factor_path ==
              kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha5120 &&
          calibration.factor_sha256 ==
              kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha5120Sha256) ||
         (entry.input_size == 17'408U &&
          factor_path ==
              kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha17408 &&
          calibration.factor_sha256 ==
              kPrefillMLPFactorizedLaneR4IdentityCandidateAlpha17408Sha256));
    if (entry.ordinal != index || entry.layer_index != expected_layer ||
        entry.family != expected_family ||
        entry.payload_offset != expected_offset ||
        entry.payload_offset > arena_bytes ||
        entry.payload_bytes > arena_bytes - entry.payload_offset ||
        entry.payload_bytes != projection_plan.projection_bytes ||
        calibration.ordinal != entry.ordinal ||
        calibration.source_module != entry.source_module ||
        calibration.source_sha256 != entry.source_sha256 ||
        !valid_clip(calibration.weight_clip_ratio) ||
        !valid_clip(calibration.activation_clip_ratio) ||
        (!calibrated_factor && !exact_identity_factor) ||
        calibration.factor_element_count != entry.input_size) {
      return false;
    }

    DecoderLayerWeights& layer = layers_[entry.layer_index];
    LinearWeight* target = nullptr;
    PrefillMLPFactorizedLaneR4LinearSidecarView* destination = nullptr;
    switch (entry.family) {
      case PrefillMLPFactorizedLaneProjectionFamily::kGate:
        target = &layer.mlp.gate_proj;
        destination = &validated[entry.layer_index].gate;
        break;
      case PrefillMLPFactorizedLaneProjectionFamily::kUp:
        target = &layer.mlp.up_proj;
        destination = &validated[entry.layer_index].up;
        break;
      case PrefillMLPFactorizedLaneProjectionFamily::kDown:
        target = &layer.mlp.down_proj;
        destination = &validated[entry.layer_index].down;
        break;
      default:
        return false;
    }
    const auto* const nvfp4 = std::get_if<NvFp4LinearWeight>(target);
    if (destination == nullptr || !has_valid_nvfp4_payload(nvfp4) ||
        nvfp4->output_size != entry.output_size ||
        nvfp4->input_size != entry.input_size) {
      return false;
    }

    const std::uintptr_t projection_address =
        arena_address + entry.payload_offset;
    const std::uintptr_t weight_address =
        projection_address + projection_plan.packed_weight_offset;
    const std::uintptr_t scale_address =
        projection_address + projection_plan.weight_scale_offset;
    const std::uintptr_t inverse_alpha_address =
        projection_address + projection_plan.inverse_alpha_offset;
    const auto range_valid = [arena_end](const std::uintptr_t address,
                                         const std::uint64_t bytes) noexcept {
      return address < arena_end && bytes <= arena_end - address;
    };
    if (weight_address % 16U != 0U || scale_address % 16U != 0U ||
        inverse_alpha_address % 16U != 0U ||
        !range_valid(weight_address, projection_plan.packed_weight_bytes) ||
        !range_valid(scale_address, projection_plan.weight_scale_bytes) ||
        !range_valid(inverse_alpha_address,
                     projection_plan.inverse_alpha_bytes) ||
        projection_plan.packed_weight_bytes >
            std::numeric_limits<std::size_t>::max() ||
        projection_plan.weight_scale_elements >
            std::numeric_limits<std::size_t>::max() ||
        projection_plan.inverse_alpha_elements >
            std::numeric_limits<std::size_t>::max()) {
      return false;
    }
    *destination = {
        reinterpret_cast<const std::uint8_t*>(weight_address),
        static_cast<std::size_t>(projection_plan.packed_weight_bytes),
        reinterpret_cast<const std::uint16_t*>(scale_address),
        static_cast<std::size_t>(projection_plan.weight_scale_elements),
        reinterpret_cast<const float*>(inverse_alpha_address),
        static_cast<std::size_t>(projection_plan.inverse_alpha_elements),
        static_cast<std::size_t>(entry.output_size),
        static_cast<std::size_t>(entry.input_size),
        kPrefillMLPFactorizedLaneR4PublicationLaneCount,
        static_cast<float>(calibration.activation_clip_ratio)};
    if (!destination->attached()) {
      return false;
    }
  }

  for (std::size_t layer_index = 0U; layer_index < validated.size();
       ++layer_index) {
    const auto& gate = policy->projections[layer_index * 3U];
    const auto& up = policy->projections[layer_index * 3U + 1U];
    if (!same_factor(gate, up) ||
        gate.activation_clip_ratio != up.activation_clip_ratio ||
        !validated[layer_index].attached()) {
      return false;
    }
  }

  clear_all();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    layers_[layer_index].prefill_mlp_factorized_lane_r4 =
        validated[layer_index];
  }
  return true;
}

bool ModelWeights::attach_prefill_attention_factorized_lane_r1_sidecars(
    const std::uint8_t* const arena, const std::size_t arena_bytes,
    const PrefillAttentionFactorizedLaneOverlayManifestBinding* const manifest,
    const PrefillAttentionFactorizedLaneOverlayPolicyBinding* const policy)
    noexcept {
  const auto clear_all = [this]() noexcept {
    for (DecoderLayerWeights& layer : layers_) {
      layer.prefill_attention_factorized_lane_r1 = {};
    }
  };
  if (arena == nullptr && arena_bytes == 0U && manifest == nullptr &&
      policy == nullptr) {
    clear_all();
    return true;
  }

  // The older Attention-O K512 plane and the all-projection R1 plane expose
  // different scale/factor ABIs.  Reject even malformed partial legacy state
  // so a caller must explicitly detach it before publishing R1.
  for (const DecoderLayerWeights& layer : layers_) {
    const LinearWeight* output = nullptr;
    if (const auto* const linear =
            std::get_if<LinearAttentionWeights>(&layer.attention)) {
      output = &linear->out_proj;
    } else if (const auto* const full =
                   std::get_if<FullAttentionWeights>(&layer.attention)) {
      output = &full->o_proj;
    }
    const auto* const fp8 =
        output == nullptr ? nullptr : std::get_if<Fp8LinearWeight>(output);
    if (fp8 == nullptr || fp8->prefill_attention_o_k512_weight != nullptr ||
        fp8->prefill_attention_o_k512_scales != nullptr ||
        fp8->prefill_attention_o_k512_activation_clip_ratio != 0.0F) {
      return false;
    }
  }

  const auto same_base = [](
                             const PrefillAttentionFactorizedLaneBaseK256Binding&
                                 left,
                             const PrefillAttentionFactorizedLaneBaseK256Binding&
                                 right) noexcept {
    return left.physical_layout == right.physical_layout &&
           left.packed_k_group_size == right.packed_k_group_size &&
           left.scale_group_size == right.scale_group_size &&
           left.manifest_sha256 == right.manifest_sha256 &&
           left.policy_sha256 == right.policy_sha256 &&
           left.payload_sha256 == right.payload_sha256 &&
           left.receipt_sha256 == right.receipt_sha256;
  };
  const auto same_factor = [](
                               const PrefillAttentionFactorizedLaneFactorSourceBinding&
                                   left,
                               const PrefillAttentionFactorizedLaneFactorSourceBinding&
                                   right) noexcept {
    return left.scheme == right.scheme && left.path == right.path &&
           left.sha256 == right.sha256 &&
           left.element_count == right.element_count;
  };
  const auto valid_clip = [](const double value) noexcept {
    const float narrowed = static_cast<float>(value);
    return std::isfinite(value) && value >= kPrefillA4MinimumClipRatio &&
           value <= 1.0 && std::isfinite(narrowed) &&
           narrowed >= static_cast<float>(kPrefillA4MinimumClipRatio) &&
           narrowed <= 1.0F;
  };

  constexpr std::uint32_t kR1LaneCount = 1U;
  constexpr std::string_view kR1FactorScheme = "identity_alpha_f32_v1";
  constexpr std::string_view kIdentity5120Sha256 =
      "42010c1c68b632e2ab15c82bca6edef2cac2026c889dd0202d609602b756f568";
  constexpr std::string_view kIdentity6144Sha256 =
      "08f46753296b40512f918614f5b0be6f15e4a95fb0aeeff6c9026be0a396c4a7";
  const PrefillAttentionFactorizedLaneOverlayLayoutPlan plan =
      prefill_attention_factorized_lane_overlay_layout_plan(kR1LaneCount);
  if (arena == nullptr || manifest == nullptr || policy == nullptr || !plan ||
      plan.payload_bytes > std::numeric_limits<std::size_t>::max() ||
      arena_bytes != static_cast<std::size_t>(plan.payload_bytes) ||
      manifest->version_major !=
          kPrefillAttentionFactorizedLaneOverlayVersionMajor ||
      manifest->version_minor !=
          kPrefillAttentionFactorizedLaneOverlayVersionMinor ||
      manifest->physical_layout !=
          kPrefillAttentionFactorizedLaneOverlayLayout ||
      manifest->source_checkpoint_id.empty() ||
      !attention_r1_lower_sha256(manifest->source_config_sha256) ||
      !attention_r1_lower_sha256(manifest->source_index_sha256) ||
      manifest->lane_count != kR1LaneCount ||
      manifest->payload_bytes != plan.payload_bytes ||
      manifest->projections.size() !=
          kPrefillAttentionFactorizedLaneProjectionCount ||
      !attention_r1_lower_sha256(manifest->manifest_sha256) ||
      manifest->required_base_k256.physical_layout !=
          kPrefillAttentionFactorizedLaneRequiredBaseK256Layout ||
      manifest->required_base_k256.packed_k_group_size !=
          kPrefillAttentionFactorizedLaneRequiredBasePackedK ||
      manifest->required_base_k256.scale_group_size !=
          kPrefillAttentionFactorizedLaneRequiredBaseScaleK ||
      !attention_r1_lower_sha256(
          manifest->required_base_k256.manifest_sha256) ||
      !attention_r1_lower_sha256(
          manifest->required_base_k256.policy_sha256) ||
      !attention_r1_lower_sha256(
          manifest->required_base_k256.payload_sha256) ||
      !attention_r1_lower_sha256(
          manifest->required_base_k256.receipt_sha256) ||
      policy->version_major != manifest->version_major ||
      policy->version_minor != manifest->version_minor ||
      policy->physical_layout != manifest->physical_layout ||
      policy->source_checkpoint_id != manifest->source_checkpoint_id ||
      policy->source_config_sha256 != manifest->source_config_sha256 ||
      policy->source_index_sha256 != manifest->source_index_sha256 ||
      policy->manifest_sha256 != manifest->manifest_sha256 ||
      !same_base(policy->required_base_k256,
                 manifest->required_base_k256) ||
      policy->lane_count != kR1LaneCount ||
      policy->projections.size() != manifest->projections.size() ||
      !attention_r1_lower_sha256(policy->policy_sha256) ||
      policy->policy_bytes == 0U || !prefill_a4_attachment_complete_ ||
      std::string_view(prefill_a4_attachment_manifest_sha256_.data(),
                       prefill_a4_attachment_manifest_sha256_.size()) !=
          manifest->required_base_k256.manifest_sha256 ||
      std::string_view(prefill_a4_attachment_policy_sha256_.data(),
                       prefill_a4_attachment_policy_sha256_.size()) !=
          manifest->required_base_k256.policy_sha256 ||
      std::string_view(prefill_a4_attachment_payload_sha256_.data(),
                       prefill_a4_attachment_payload_sha256_.size()) !=
          manifest->required_base_k256.payload_sha256) {
    return false;
  }
  try {
    if (attention_r1_manifest_sha256(*manifest) !=
        manifest->manifest_sha256) {
      return false;
    }
  } catch (...) {
    return false;
  }

  constexpr auto kPointerMaximum =
      std::numeric_limits<std::uintptr_t>::max();
  const std::uintptr_t arena_address =
      reinterpret_cast<std::uintptr_t>(arena);
  if (arena_address % kPrefillA4FactorizedLaneMinimumAlignment != 0U ||
      arena_bytes > kPointerMaximum ||
      arena_address > kPointerMaximum - arena_bytes) {
    return false;
  }
  const std::uintptr_t arena_end = arena_address + arena_bytes;

  std::array<PrefillAttentionFactorizedLaneR1LayerView,
             kPrefillAttentionFactorizedLaneLayerCount>
      validated{};
  std::size_t index = 0U;
  try {
    for (std::uint32_t layer_index = 0U;
         layer_index < kPrefillAttentionFactorizedLaneLayerCount;
         ++layer_index) {
      const std::uint32_t projection_count =
          prefill_attention_factorized_lane_is_full_layer(layer_index)
              ? kPrefillAttentionFactorizedLaneFullProjectionsPerLayer
              : kPrefillAttentionFactorizedLaneLinearProjectionsPerLayer;
      for (std::uint32_t position = 0U; position < projection_count;
           ++position, ++index) {
        const PrefillAttentionFactorizedLaneProjectionFamily expected_family =
            attention_r1_expected_family(layer_index, position);
        const PrefillA4FactorizedLaneProjectionLayoutPlan* const
            projection_plan = attention_r1_projection_plan(plan,
                                                           expected_family);
        std::uint64_t expected_output_size = 0U;
        std::uint64_t expected_input_size = 0U;
        const std::uint64_t expected_offset =
            prefill_attention_factorized_lane_projection_absolute_offset(
                plan, layer_index, expected_family);
        const std::uint32_t expected_ordinal =
            prefill_attention_factorized_lane_projection_ordinal(
                layer_index, expected_family);
        if (projection_plan == nullptr ||
            !attention_r1_expected_shape(expected_family,
                                         expected_output_size,
                                         expected_input_size) ||
            expected_ordinal != index) {
          return false;
        }

        const PrefillAttentionFactorizedLaneManifestProjection& entry =
            manifest->projections[index];
        const PrefillAttentionFactorizedLaneProjectionCalibrationBinding&
            calibration = policy->projections[index];
        const std::string expected_module =
            attention_r1_expected_source_module(layer_index,
                                                expected_family);
        const std::string_view expected_factor_digest =
            expected_input_size ==
                    kPrefillAttentionFactorizedLaneAttentionOutputSize
                ? kIdentity6144Sha256
                : kIdentity5120Sha256;
        if (entry.ordinal != expected_ordinal ||
            entry.layer_index != layer_index ||
            entry.family != expected_family ||
            entry.source_module != expected_module ||
            !attention_r1_lower_sha256(entry.source_sha256) ||
            entry.output_size != expected_output_size ||
            entry.input_size != expected_input_size ||
            entry.payload_offset != expected_offset ||
            entry.payload_offset > arena_bytes ||
            entry.payload_bytes > arena_bytes - entry.payload_offset ||
            entry.payload_bytes != projection_plan->projection_bytes ||
            calibration.ordinal != entry.ordinal ||
            calibration.source_module != entry.source_module ||
            calibration.source_sha256 != entry.source_sha256 ||
            !valid_clip(calibration.weight_clip_ratio) ||
            !valid_clip(calibration.activation_clip_ratio) ||
            calibration.factor_source.scheme != kR1FactorScheme ||
            !calibration.factor_source.path.empty() ||
            calibration.factor_source.sha256 != expected_factor_digest ||
            calibration.factor_source.element_count != entry.input_size) {
          return false;
        }

        DecoderLayerWeights& layer = layers_[layer_index];
        LinearWeight* target = nullptr;
        PrefillAttentionFactorizedLaneR1LinearSidecarView* destination =
            nullptr;
        switch (expected_family) {
          case PrefillAttentionFactorizedLaneProjectionFamily::kLinearQkv: {
            auto* const attention =
                std::get_if<LinearAttentionWeights>(&layer.attention);
            if (attention != nullptr) {
              target = &attention->in_proj_qkv;
              destination = &validated[layer_index].linear_qkv;
            }
            break;
          }
          case PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ: {
            auto* const attention =
                std::get_if<LinearAttentionWeights>(&layer.attention);
            if (attention != nullptr) {
              target = &attention->in_proj_z;
              destination = &validated[layer_index].linear_z;
            }
            break;
          }
          case PrefillAttentionFactorizedLaneProjectionFamily::kLinearO: {
            auto* const attention =
                std::get_if<LinearAttentionWeights>(&layer.attention);
            if (attention != nullptr) {
              target = &attention->out_proj;
              destination = &validated[layer_index].linear_o;
            }
            break;
          }
          case PrefillAttentionFactorizedLaneProjectionFamily::kFullQ: {
            auto* const attention =
                std::get_if<FullAttentionWeights>(&layer.attention);
            if (attention != nullptr) {
              target = &attention->q_proj;
              destination = &validated[layer_index].full_q;
            }
            break;
          }
          case PrefillAttentionFactorizedLaneProjectionFamily::kFullK: {
            auto* const attention =
                std::get_if<FullAttentionWeights>(&layer.attention);
            if (attention != nullptr) {
              target = &attention->k_proj;
              destination = &validated[layer_index].full_k;
            }
            break;
          }
          case PrefillAttentionFactorizedLaneProjectionFamily::kFullV: {
            auto* const attention =
                std::get_if<FullAttentionWeights>(&layer.attention);
            if (attention != nullptr) {
              target = &attention->v_proj;
              destination = &validated[layer_index].full_v;
            }
            break;
          }
          case PrefillAttentionFactorizedLaneProjectionFamily::kFullO: {
            auto* const attention =
                std::get_if<FullAttentionWeights>(&layer.attention);
            if (attention != nullptr) {
              target = &attention->o_proj;
              destination = &validated[layer_index].full_o;
            }
            break;
          }
          default:
            return false;
        }
        const auto* const fp8 =
            target == nullptr ? nullptr : std::get_if<Fp8LinearWeight>(target);
        const PrefillA4LinearSidecarView base =
            target == nullptr ? PrefillA4LinearSidecarView{}
                              : prefill_a4_sidecar_view(*target);
        if (destination == nullptr || !has_valid_fp8_payload(fp8) ||
            fp8->output_size != entry.output_size ||
            fp8->input_size != entry.input_size || !base.attached() ||
            base.sidecar_kind != PrefillSidecarKind::kA4K256 ||
            base.packed_k_group_size !=
                kPrefillAttentionFactorizedLaneRequiredBasePackedK ||
            base.scale_group_size !=
                kPrefillAttentionFactorizedLaneRequiredBaseScaleK) {
          return false;
        }

        const std::uintptr_t projection_address =
            arena_address + entry.payload_offset;
        const std::uintptr_t weight_address =
            projection_address + projection_plan->packed_weight_offset;
        const std::uintptr_t scale_address =
            projection_address + projection_plan->weight_scale_offset;
        const std::uintptr_t inverse_alpha_address =
            projection_address + projection_plan->inverse_alpha_offset;
        const auto range_valid =
            [arena_end](const std::uintptr_t address,
                        const std::uint64_t bytes) noexcept {
              return address < arena_end && bytes <= arena_end - address;
            };
        if (weight_address % 16U != 0U || scale_address % 16U != 0U ||
            inverse_alpha_address % 16U != 0U ||
            !range_valid(weight_address,
                         projection_plan->packed_weight_bytes) ||
            !range_valid(scale_address,
                         projection_plan->weight_scale_bytes) ||
            !range_valid(inverse_alpha_address,
                         projection_plan->inverse_alpha_bytes) ||
            projection_plan->packed_weight_bytes >
                std::numeric_limits<std::size_t>::max() ||
            projection_plan->weight_scale_elements >
                std::numeric_limits<std::size_t>::max() ||
            projection_plan->inverse_alpha_elements >
                std::numeric_limits<std::size_t>::max()) {
          return false;
        }
        *destination = {
            reinterpret_cast<const std::uint8_t*>(weight_address),
            static_cast<std::size_t>(projection_plan->packed_weight_bytes),
            reinterpret_cast<const std::uint16_t*>(scale_address),
            static_cast<std::size_t>(projection_plan->weight_scale_elements),
            reinterpret_cast<const float*>(inverse_alpha_address),
            static_cast<std::size_t>(projection_plan->inverse_alpha_elements),
            static_cast<std::size_t>(entry.output_size),
            static_cast<std::size_t>(entry.input_size),
            kR1LaneCount,
            static_cast<float>(calibration.activation_clip_ratio)};
        if (!destination->attached()) {
          return false;
        }
      }
    }
  } catch (...) {
    return false;
  }
  if (index != manifest->projections.size()) {
    return false;
  }

  for (std::uint32_t layer_index = 0U;
       layer_index < kPrefillAttentionFactorizedLaneLayerCount;
       ++layer_index) {
    const bool full_layer =
        prefill_attention_factorized_lane_is_full_layer(layer_index);
    const auto first_family =
        full_layer ? PrefillAttentionFactorizedLaneProjectionFamily::kFullQ
                   : PrefillAttentionFactorizedLaneProjectionFamily::
                         kLinearQkv;
    const auto second_family =
        full_layer ? PrefillAttentionFactorizedLaneProjectionFamily::kFullK
                   : PrefillAttentionFactorizedLaneProjectionFamily::kLinearZ;
    const std::uint32_t first_ordinal =
        prefill_attention_factorized_lane_projection_ordinal(layer_index,
                                                              first_family);
    const std::uint32_t second_ordinal =
        prefill_attention_factorized_lane_projection_ordinal(layer_index,
                                                              second_family);
    if (first_ordinal >= policy->projections.size() ||
        second_ordinal >= policy->projections.size()) {
      return false;
    }
    const auto& first = policy->projections[first_ordinal];
    const auto& second = policy->projections[second_ordinal];
    bool group_valid =
        same_factor(first.factor_source, second.factor_source) &&
        static_cast<float>(first.activation_clip_ratio) ==
            static_cast<float>(second.activation_clip_ratio);
    if (full_layer) {
      const std::uint32_t third_ordinal =
          prefill_attention_factorized_lane_projection_ordinal(
              layer_index,
              PrefillAttentionFactorizedLaneProjectionFamily::kFullV);
      if (third_ordinal >= policy->projections.size()) {
        return false;
      }
      const auto& third = policy->projections[third_ordinal];
      group_valid =
          group_valid && same_factor(first.factor_source,
                                     third.factor_source) &&
          static_cast<float>(first.activation_clip_ratio) ==
              static_cast<float>(third.activation_clip_ratio);
    }
    if (!group_valid ||
        (full_layer ? !validated[layer_index].full_attached()
                    : !validated[layer_index].linear_attached())) {
      return false;
    }
  }

  clear_all();
  for (std::size_t layer_index = 0U; layer_index < layers_.size();
       ++layer_index) {
    layers_[layer_index].prefill_attention_factorized_lane_r1 =
        validated[layer_index];
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

PrefillA4LinearSidecarView prefill_a4_sidecar_view(
    const LinearWeight& weight) noexcept {
  if (const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight)) {
    return {fp8->prefill_a4_weight, fp8->prefill_a4_scales,
            fp8->prefill_a4_metadata,
            fp8->prefill_a4_sidecar_kind,
            fp8->prefill_a4_packed_k_group_size,
            fp8->prefill_a4_scale_group_size,
            fp8->prefill_a4_activation_clip_ratio, fp8->output_size,
            fp8->input_size};
  }
  if (const auto* const nvfp4 =
          std::get_if<NvFp4LinearWeight>(&weight)) {
    return {nvfp4->prefill_a4_weight, nvfp4->prefill_a4_scales,
            nvfp4->prefill_a4_metadata,
            nvfp4->prefill_a4_sidecar_kind,
            nvfp4->prefill_a4_packed_k_group_size,
            nvfp4->prefill_a4_scale_group_size,
            nvfp4->prefill_a4_activation_clip_ratio, nvfp4->output_size,
            nvfp4->input_size};
  }
  return {};
}

PrefillAttentionOK512LinearSidecarView
prefill_attention_o_k512_sidecar_view(
    const LinearWeight& weight) noexcept {
  const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
  if (fp8 == nullptr) {
    return {};
  }
  return {fp8->prefill_attention_o_k512_weight,
          fp8->prefill_attention_o_k512_scales,
          fp8->prefill_attention_o_k512_activation_clip_ratio,
          fp8->output_size, fp8->input_size};
}

PrefillMLPK512LinearSidecarView prefill_mlp_k512_sidecar_view(
    const LinearWeight& weight) noexcept {
  const auto* const nvfp4 = std::get_if<NvFp4LinearWeight>(&weight);
  if (nvfp4 == nullptr) {
    return {};
  }
  return {nvfp4->prefill_mlp_k512_weight,
          nvfp4->prefill_mlp_k512_scales,
          nvfp4->prefill_mlp_k512_activation_clip_ratio,
          nvfp4->output_size, nvfp4->input_size};
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
