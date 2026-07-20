#include "q3x/runtime/model_weights.h"

#include "q3x/io/safetensors.h"

#include <cuda_runtime.h>

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
