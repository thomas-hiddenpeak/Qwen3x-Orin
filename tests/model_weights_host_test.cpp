#include "q3x/runtime/model_weights.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/model_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace st = q3x::io::safetensors;

class TestContext {
 public:
  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures_;
      std::cerr << "FAIL: " << message << '\n';
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

enum class SyntheticLinearKind : std::uint8_t {
  kBf16,
  kFp8,
  kNvFp4,
};

[[nodiscard]] std::uint64_t aligned(const std::uint64_t value) {
  constexpr std::uint64_t kAlignment = runtime::kResidentTensorAlignment;
  return (value + kAlignment - 1U) & ~(kAlignment - 1U);
}

[[nodiscard]] std::uint64_t tensor_bytes(
    const st::DType dtype, const std::vector<std::uint64_t>& shape) {
  std::uint64_t elements = 1U;
  for (const std::uint64_t dimension : shape) {
    elements *= dimension;
  }
  return elements * st::bit_width(dtype) / 8U;
}

class SyntheticArena {
 public:
  explicit SyntheticArena(
      const bool force_fp8_attention_outputs = false,
      const bool force_nvfp4_down_projections = false,
      const bool force_fp8_linear_qkv = false,
      const bool force_fp8_prefill_supermatrix = false,
      const bool force_p40_packed_projection = false)
      : force_fp8_attention_outputs_(force_fp8_attention_outputs),
        force_nvfp4_down_projections_(force_nvfp4_down_projections),
        force_fp8_linear_qkv_(force_fp8_linear_qkv),
        force_fp8_prefill_supermatrix_(force_fp8_prefill_supermatrix ||
                                       force_p40_packed_projection),
        force_p40_packed_projection_(force_p40_packed_projection) {
    build();
  }

  [[nodiscard]] runtime::WeightBindingSource source() const noexcept {
    runtime::WeightBindingSource source;
    source.lookup_context = this;
    source.lookup = &lookup;
    source.arena_data = reinterpret_cast<const void*>(kBaseAddress);
    source.arena_bytes = arena_bytes_;
    source.scalar_read_context = this;
    source.scalar_read = &read_scalar;
    return source;
  }

  [[nodiscard]] runtime::DeviceTensorView& at(const std::string& name) {
    return tensors_.at(name);
  }

  void erase(const std::string& name) { tensors_.erase(name); }

  [[nodiscard]] float& scalar_for(const std::string& name) {
    const auto pointer =
        reinterpret_cast<std::uintptr_t>(tensors_.at(name).device_data);
    return scalars_.at(pointer);
  }

  void set_scalar_status(const int status) noexcept {
    scalar_status_ = status;
  }

  [[nodiscard]] std::uint64_t arena_bytes() const noexcept {
    return arena_bytes_;
  }

  [[nodiscard]] std::size_t tensor_count() const noexcept {
    return tensors_.size();
  }

 private:
  static constexpr std::uintptr_t kBaseAddress = 0x0000010000000000ULL;

  [[nodiscard]] static const runtime::DeviceTensorView* lookup(
      const void* const context, const std::string_view name) noexcept {
    const auto* const arena = static_cast<const SyntheticArena*>(context);
    if (arena == nullptr) {
      return nullptr;
    }
    const auto selected = arena->tensors_.find(name);
    return selected == arena->tensors_.end() ? nullptr : &selected->second;
  }

  [[nodiscard]] static int read_scalar(const void* const context,
                                       const float* const device_value,
                                       float* const host_value) noexcept {
    const auto* const arena = static_cast<const SyntheticArena*>(context);
    if (arena == nullptr || host_value == nullptr) {
      return 1;
    }
    if (arena->scalar_status_ != 0) {
      return arena->scalar_status_;
    }
    const auto selected = arena->scalars_.find(
        reinterpret_cast<std::uintptr_t>(device_value));
    if (selected == arena->scalars_.end()) {
      return 2;
    }
    *host_value = selected->second;
    return 0;
  }

  void add(std::string name, const st::DType dtype,
           std::vector<std::uint64_t> shape, const float scalar = -1.0F) {
    cursor_ = aligned(cursor_);
    runtime::DeviceTensorView view;
    view.arena_offset = cursor_;
    view.byte_size = tensor_bytes(dtype, shape);
    view.device_data = reinterpret_cast<const void*>(
        kBaseAddress + static_cast<std::uintptr_t>(cursor_));
    view.dtype = dtype;
    view.shape = std::move(shape);
    const auto pointer = reinterpret_cast<std::uintptr_t>(view.device_data);
    if (scalar >= 0.0F) {
      scalars_.emplace(pointer, scalar);
    }
    cursor_ += view.byte_size;
    tensors_.emplace(std::move(name), std::move(view));
  }

  [[nodiscard]] SyntheticLinearKind next_kind() noexcept {
    const auto kind = static_cast<SyntheticLinearKind>(linear_index_ % 3U);
    ++linear_index_;
    return kind;
  }

  void add_linear(const std::string& module, const std::uint64_t output_size,
                  const std::uint64_t input_size,
                  const SyntheticLinearKind kind) {
    if (kind == SyntheticLinearKind::kBf16) {
      add(module + ".weight", st::DType::kBf16,
          {output_size, input_size});
      return;
    }
    if (kind == SyntheticLinearKind::kFp8) {
      add(module + ".weight", st::DType::kF8E4M3,
          {output_size, input_size});
      add(module + ".weight_scale", st::DType::kF32, {}, 0.125F);
      add(module + ".input_scale", st::DType::kF32, {}, 0.25F);
      return;
    }
    add(module + ".weight", st::DType::kU8,
        {output_size, input_size / 2U});
    add(module + ".weight_scale", st::DType::kF8E4M3,
        {output_size, input_size / 16U});
    add(module + ".weight_scale_2", st::DType::kF32, {}, 0.03125F);
    add(module + ".input_scale", st::DType::kF32, {}, 0.5F);
  }

  void build() {
    const q3x::model::ModelConfig* const config =
        q3x::model::find_known_model(q3x::model::KnownModel::kQwen36_27B);
    if (config == nullptr) {
      return;
    }
    add("model.language_model.embed_tokens.weight", st::DType::kBf16,
        {config->vocab_size, config->hidden_size});
    add("model.language_model.norm.weight", st::DType::kBf16,
        {config->hidden_size});

    for (std::size_t layer = 0U; layer < config->num_hidden_layers; ++layer) {
      const std::string prefix = "model.language_model.layers." +
                                 std::to_string(layer) + ".";
      add(prefix + "input_layernorm.weight", st::DType::kBf16,
          {config->hidden_size});
      add(prefix + "post_attention_layernorm.weight", st::DType::kBf16,
          {config->hidden_size});
      add_linear(prefix + "mlp.gate_proj", config->intermediate_size,
                 config->hidden_size,
                 force_p40_packed_projection_
                     ? SyntheticLinearKind::kNvFp4
                     : next_kind());
      add_linear(prefix + "mlp.up_proj", config->intermediate_size,
                 config->hidden_size,
                 force_p40_packed_projection_
                     ? SyntheticLinearKind::kNvFp4
                     : next_kind());
      const SyntheticLinearKind down_kind = next_kind();
      add_linear(prefix + "mlp.down_proj", config->hidden_size,
                 config->intermediate_size,
                 force_nvfp4_down_projections_
                         || force_p40_packed_projection_
                     ? SyntheticLinearKind::kNvFp4
                     : down_kind);

      if (config->layer_type(layer) ==
          q3x::model::LayerType::kLinearAttention) {
        add_linear(prefix + "linear_attn.in_proj_qkv",
                   config->linear_qkv_projection_dim(), config->hidden_size,
                   force_fp8_linear_qkv_ || force_fp8_prefill_supermatrix_
                       ? SyntheticLinearKind::kFp8
                       : next_kind());
        add_linear(prefix + "linear_attn.in_proj_z",
                   config->linear_value_dim(), config->hidden_size,
                   force_fp8_prefill_supermatrix_
                       ? SyntheticLinearKind::kFp8
                       : next_kind());
        add_linear(prefix + "linear_attn.in_proj_a",
                   config->linear_num_value_heads, config->hidden_size,
                   SyntheticLinearKind::kBf16);
        add_linear(prefix + "linear_attn.in_proj_b",
                   config->linear_num_value_heads, config->hidden_size,
                   SyntheticLinearKind::kBf16);
        add(prefix + "linear_attn.conv1d.weight", st::DType::kBf16,
            {config->linear_qkv_projection_dim(), 1U,
             config->linear_conv_kernel_dim});
        add(prefix + "linear_attn.A_log", st::DType::kBf16,
            {config->linear_num_value_heads});
        add(prefix + "linear_attn.dt_bias", st::DType::kBf16,
            {config->linear_num_value_heads});
        add(prefix + "linear_attn.norm.weight", st::DType::kBf16,
            {config->linear_value_head_dim});
        const SyntheticLinearKind output_kind = next_kind();
        add_linear(prefix + "linear_attn.out_proj", config->hidden_size,
                   config->linear_value_dim(),
                   force_fp8_attention_outputs_ ||
                           force_fp8_prefill_supermatrix_
                       ? SyntheticLinearKind::kFp8
                       : output_kind);
      } else {
        add_linear(prefix + "self_attn.q_proj", config->q_projection_dim(),
                   config->hidden_size,
                   force_fp8_prefill_supermatrix_
                       ? SyntheticLinearKind::kFp8
                       : next_kind());
        add_linear(prefix + "self_attn.k_proj", config->kv_dim(),
                   config->hidden_size,
                   force_fp8_prefill_supermatrix_
                       ? SyntheticLinearKind::kFp8
                       : next_kind());
        add_linear(prefix + "self_attn.v_proj", config->kv_dim(),
                   config->hidden_size,
                   force_fp8_prefill_supermatrix_
                       ? SyntheticLinearKind::kFp8
                       : next_kind());
        const SyntheticLinearKind output_kind = next_kind();
        add_linear(prefix + "self_attn.o_proj", config->hidden_size,
                   config->q_dim(),
                   force_fp8_attention_outputs_ ||
                           force_fp8_prefill_supermatrix_
                       ? SyntheticLinearKind::kFp8
                       : output_kind);
        add(prefix + "self_attn.q_norm.weight", st::DType::kBf16,
            {config->head_dim});
        add(prefix + "self_attn.k_norm.weight", st::DType::kBf16,
            {config->head_dim});
      }
    }
    add_linear("lm_head", config->vocab_size, config->hidden_size,
               next_kind());
    arena_bytes_ = aligned(cursor_) + runtime::kResidentTensorAlignment;
  }

  std::map<std::string, runtime::DeviceTensorView, std::less<>> tensors_;
  std::map<std::uintptr_t, float> scalars_;
  std::uint64_t cursor_ = 0U;
  std::uint64_t arena_bytes_ = 0U;
  std::size_t linear_index_ = 0U;
  int scalar_status_ = 0;
  bool force_fp8_attention_outputs_ = false;
  bool force_nvfp4_down_projections_ = false;
  bool force_fp8_linear_qkv_ = false;
  bool force_fp8_prefill_supermatrix_ = false;
  bool force_p40_packed_projection_ = false;
};

[[nodiscard]] runtime::Fp8LinearWeight* mutable_attention_output(
    runtime::ModelWeights& weights, const std::size_t layer_index) {
  auto& layer =
      const_cast<runtime::DecoderLayerWeights&>(weights.layer(layer_index));
  if (auto* const linear =
          std::get_if<runtime::LinearAttentionWeights>(&layer.attention)) {
    return std::get_if<runtime::Fp8LinearWeight>(&linear->out_proj);
  }
  if (auto* const full =
          std::get_if<runtime::FullAttentionWeights>(&layer.attention)) {
    return std::get_if<runtime::Fp8LinearWeight>(&full->o_proj);
  }
  return nullptr;
}

[[nodiscard]] bool sidecars_match(
    runtime::ModelWeights& weights, const std::uintptr_t arena_address) {
  for (std::size_t layer_index = 0U;
       layer_index < runtime::kQwen36DenseLayerCount; ++layer_index) {
    const runtime::Fp8LinearWeight* const output =
        mutable_attention_output(weights, layer_index);
    const std::uintptr_t expected =
        arena_address +
        layer_index *
            runtime::kFp8M1OutputProjectionAosoa4PreswizzledBytesPerLayer;
    if (output == nullptr ||
        reinterpret_cast<std::uintptr_t>(
            output->m1_aosoa4_preswizzled_weight) != expected) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] runtime::Fp8LinearWeight* mutable_linear_attention_qkv(
    runtime::ModelWeights& weights, const std::size_t layer_index) {
  auto& layer =
      const_cast<runtime::DecoderLayerWeights&>(weights.layer(layer_index));
  auto* const linear =
      std::get_if<runtime::LinearAttentionWeights>(&layer.attention);
  return linear == nullptr
             ? nullptr
             : std::get_if<runtime::Fp8LinearWeight>(&linear->in_proj_qkv);
}

[[nodiscard]] bool qkv_register_feed_attachments_match(
    runtime::ModelWeights& weights,
    const std::vector<
        runtime::Fp8PrefillQkvRegisterFeedSidecarDescriptor>& expected) {
  for (std::size_t layer_index = 0U;
       layer_index < runtime::kQwen36DenseLayerCount; ++layer_index) {
    if (!std::holds_alternative<runtime::LinearAttentionWeights>(
            weights.layer(layer_index).attention)) {
      continue;
    }
    const runtime::Fp8LinearWeight* const qkv =
        mutable_linear_attention_qkv(weights, layer_index);
    if (qkv == nullptr) {
      return false;
    }
    const auto selected = std::find_if(
        expected.begin(), expected.end(),
        [layer_index](const auto& descriptor) {
          return descriptor.layer_index == layer_index;
        });
    const std::uint8_t* const expected_sidecar =
        selected == expected.end() ? nullptr : selected->sidecar;
    if (qkv->prefill_qkv_register_feed_sidecar != expected_sidecar) {
      return false;
    }
  }
  return true;
}

struct Fp8PrefillSupermatrixProjection {
  runtime::Fp8LinearWeight* weight = nullptr;
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
};

[[nodiscard]] std::vector<Fp8PrefillSupermatrixProjection>
mutable_fp8_prefill_supermatrix_projections(runtime::ModelWeights& weights) {
  std::vector<Fp8PrefillSupermatrixProjection> projections;
  projections.reserve(runtime::kFp8PrefillSupermatrixProjectionCount);
  const auto append = [&](runtime::LinearWeight& binding,
                          const std::size_t output_size,
                          const std::size_t input_size) {
    projections.push_back(
        {std::get_if<runtime::Fp8LinearWeight>(&binding), output_size,
         input_size});
  };

  for (std::size_t layer_index = 0U;
       layer_index < runtime::kQwen36DenseLayerCount; ++layer_index) {
    auto& layer = const_cast<runtime::DecoderLayerWeights&>(
        weights.layer(layer_index));
    if (auto* const linear =
            std::get_if<runtime::LinearAttentionWeights>(
                &layer.attention)) {
      append(linear->in_proj_qkv, 10'240U, 5'120U);
      append(linear->in_proj_z, 6'144U, 5'120U);
      append(linear->out_proj, 5'120U, 6'144U);
    } else if (auto* const full =
                   std::get_if<runtime::FullAttentionWeights>(
                       &layer.attention)) {
      append(full->q_proj, 12'288U, 5'120U);
      append(full->k_proj, 1'024U, 5'120U);
      append(full->v_proj, 1'024U, 5'120U);
      append(full->o_proj, 5'120U, 6'144U);
    }
  }
  return projections;
}

[[nodiscard]] bool fp8_prefill_supermatrix_attachments_match_layout(
    const std::vector<Fp8PrefillSupermatrixProjection>& projections,
    const std::uintptr_t arena_address) {
  if (projections.size() !=
      runtime::kFp8PrefillSupermatrixProjectionCount) {
    return false;
  }
  std::size_t offset = 0U;
  for (const Fp8PrefillSupermatrixProjection& projection : projections) {
    if (projection.weight == nullptr ||
        projection.weight->output_size != projection.output_size ||
        projection.weight->input_size != projection.input_size ||
        reinterpret_cast<std::uintptr_t>(
            projection.weight->prefill_supermatrix_sidecar) !=
            arena_address + offset) {
      return false;
    }
    offset += projection.output_size * projection.input_size;
  }
  return offset == runtime::kQwen36Fp8PrefillSupermatrixSidecarBytes;
}

[[nodiscard]] std::vector<const std::uint8_t*>
fp8_prefill_supermatrix_attachment_snapshot(
    const std::vector<Fp8PrefillSupermatrixProjection>& projections) {
  std::vector<const std::uint8_t*> snapshot;
  snapshot.reserve(projections.size());
  for (const Fp8PrefillSupermatrixProjection& projection : projections) {
    snapshot.push_back(projection.weight == nullptr
                           ? nullptr
                           : projection.weight->prefill_supermatrix_sidecar);
  }
  return snapshot;
}

[[nodiscard]] bool fp8_prefill_supermatrix_attachments_match_snapshot(
    const std::vector<Fp8PrefillSupermatrixProjection>& projections,
    const std::vector<const std::uint8_t*>& snapshot) {
  if (projections.size() != snapshot.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < projections.size(); ++index) {
    if (projections[index].weight == nullptr ||
        projections[index].weight->prefill_supermatrix_sidecar !=
            snapshot[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] runtime::NvFp4LinearWeight* mutable_nvfp4_down(
    runtime::ModelWeights& weights, const std::size_t layer_index) {
  auto& layer =
      const_cast<runtime::DecoderLayerWeights&>(weights.layer(layer_index));
  return std::get_if<runtime::NvFp4LinearWeight>(&layer.mlp.down_proj);
}

[[nodiscard]] bool down_scale6_attachments_match(
    runtime::ModelWeights& weights,
    const std::vector<runtime::NvFp4DownScale6SidecarDescriptor>& expected) {
  for (std::size_t layer_index = 0U;
       layer_index < runtime::kQwen36DenseLayerCount; ++layer_index) {
    const runtime::NvFp4LinearWeight* const down =
        mutable_nvfp4_down(weights, layer_index);
    if (down == nullptr) {
      return false;
    }
    const auto selected = std::find_if(
        expected.begin(), expected.end(),
        [layer_index](
            const runtime::NvFp4DownScale6SidecarDescriptor& descriptor) {
          return descriptor.layer_index == layer_index;
        });
    const std::uint8_t* const expected_sidecar =
        selected == expected.end() ? nullptr : selected->sidecar;
    const unsigned int expected_base =
        selected == expected.end() ? 0U : selected->scale_base;
    if (down->down_scale6_sidecar != expected_sidecar ||
        down->down_scale6_base != expected_base) {
      return false;
    }
  }
  return true;
}

void test_successful_bind(TestContext& test) {
  static_assert(!std::is_copy_constructible_v<runtime::ModelWeights>);
  static_assert(!std::is_copy_assignable_v<runtime::ModelWeights>);
  static_assert(std::is_nothrow_move_constructible_v<runtime::ModelWeights>);

  SyntheticArena arena;
  runtime::WeightBindResult result =
      runtime::bind_qwen36_27b_weights(arena.source());
  test.expect(result.ok(), "complete synthetic 64-layer ABI binds");
  if (!result) {
    std::cerr << "bind diagnostic: " << result.diagnostic.message << " ("
              << result.diagnostic.tensor << ")\n";
    return;
  }

  const runtime::ModelWeights& weights = *result.value;
  test.expect(weights.embed_tokens().output_size == 248320U &&
                  weights.embed_tokens().input_size == 5120U,
              "embedding matrix has exact dimensions");
  test.expect(weights.final_norm().element_count == 5120U,
              "final norm has exact dimension");
  test.expect(weights.layers().size() == runtime::kQwen36DenseLayerCount,
              "binding exposes exactly 64 decoder layers");
  test.expect(weights.stats().linear_attention_layers == 48U &&
                  weights.stats().full_attention_layers == 16U,
              "binding records exact 48/16 hybrid schedule");
  test.expect(weights.stats().bf16_projections > 0U &&
                  weights.stats().fp8_projections > 0U &&
                  weights.stats().nvfp4_projections > 0U,
              "binding covers BF16, FP8, and NVFP4 projections");
  test.expect(weights.stats().bf16_projections +
                      weights.stats().fp8_projections +
                      weights.stats().nvfp4_projections ==
                  497U,
              "all 497 projection objects are typed");
  test.expect(weights.stats().tensor_views == arena.tensor_count(),
              "typed graph consumes every synthetic ABI tensor exactly once");

  for (std::size_t index = 0U; index < weights.layers().size(); ++index) {
    const bool expected_full = ((index + 1U) % 4U) == 0U;
    test.expect(expected_full ==
                    std::holds_alternative<runtime::FullAttentionWeights>(
                        weights.layer(index).attention),
                "each layer has the exact 3-linear/1-full alternative");
  }

  const auto& layer0 = std::get<runtime::LinearAttentionWeights>(
      weights.layer(0U).attention);
  test.expect(runtime::linear_output_size(layer0.in_proj_qkv) == 10240U &&
                  runtime::linear_input_size(layer0.in_proj_qkv) == 5120U,
              "linear-attention QKV dimensions are exact");
  test.expect(runtime::linear_output_size(layer0.in_proj_a) == 48U &&
                  runtime::linear_weight_kind(layer0.in_proj_a) ==
                      runtime::LinearWeightKind::kBf16,
              "linear-attention A projection is exact BF16 [48,5120]");
  test.expect(layer0.conv1d.shape ==
                  std::array<std::size_t, 3U>{10240U, 1U, 4U},
              "linear-attention conv shape is exact");

  const auto& layer3 = std::get<runtime::FullAttentionWeights>(
      weights.layer(3U).attention);
  test.expect(runtime::linear_output_size(layer3.q_proj) == 12288U &&
                  runtime::linear_output_size(layer3.k_proj) == 1024U &&
                  runtime::linear_output_size(layer3.o_proj) == 5120U,
              "full-attention projection dimensions are exact");
  test.expect(layer3.q_norm.element_count == 256U &&
                  layer3.k_norm.element_count == 256U,
              "full-attention head norms use head_dim=256");

  const auto& layer0_up =
      std::get<runtime::Fp8LinearWeight>(weights.layer(0U).mlp.up_proj);
  test.expect(layer0_up.weight_scale == 0.125F &&
                  layer0_up.input_scale == 0.25F &&
                  layer0_up.weight_scale_device != nullptr &&
                  layer0_up.input_scale_device != nullptr,
              "FP8 device companions and host scalar copies are retained");
  const auto& layer0_down =
      std::get<runtime::NvFp4LinearWeight>(weights.layer(0U).mlp.down_proj);
  test.expect(layer0_down.block_scale != nullptr &&
                  layer0_down.weight_scale_2 == 0.03125F &&
                  layer0_down.input_scale == 0.5F,
              "NVFP4 block/scale companions and host scalars are retained");

  runtime::ModelWeights moved = std::move(*result.value);
  test.expect(moved.embed_tokens().weight != nullptr,
              "non-owning model view is movable");
}

void test_fp8_m1_output_projection_sidecar_attachment(TestContext& test) {
  static_assert(runtime::kFp8M1OutputProjectionRows == 5'120U);
  static_assert(runtime::kFp8M1OutputProjectionColumns == 6'144U);
  static_assert(
      runtime::kFp8M1OutputProjectionAosoa4PreswizzledBytesPerLayer ==
      30U * 1024U * 1024U);
  static_assert(
      runtime::kQwen36Fp8M1OutputProjectionAosoa4PreswizzledBytes ==
      1'920U * 1024U * 1024U);

  SyntheticArena arena(/*force_fp8_attention_outputs=*/true);
  runtime::WeightBindResult result =
      runtime::bind_qwen36_27b_weights(arena.source());
  test.expect(result.ok(),
              "all-FP8 output-projection synthetic ABI binds");
  if (!result) {
    return;
  }
  runtime::ModelWeights& weights = *result.value;
  bool defaults_are_null = true;
  for (std::size_t layer_index = 0U;
       layer_index < runtime::kQwen36DenseLayerCount; ++layer_index) {
    const runtime::Fp8LinearWeight* const output =
        mutable_attention_output(weights, layer_index);
    defaults_are_null =
        defaults_are_null && output != nullptr &&
        output->m1_aosoa4_preswizzled_weight == nullptr;
  }
  test.expect(defaults_are_null,
              "FP8 sidecar views default to null after binding");

  constexpr std::uintptr_t kFirstArenaAddress = 0x0000040000000000ULL;
  constexpr std::uintptr_t kSecondArenaAddress = 0x0000050000000000ULL;
  constexpr std::size_t kSidecarBytes =
      runtime::kQwen36Fp8M1OutputProjectionAosoa4PreswizzledBytes;
  const auto* const first_arena =
      reinterpret_cast<const std::uint8_t*>(kFirstArenaAddress);
  const auto* const second_arena =
      reinterpret_cast<const std::uint8_t*>(kSecondArenaAddress);

  test.expect(weights.attach_fp8_m1_output_projection_sidecars(
                  first_arena, kSidecarBytes) &&
                  sidecars_match(weights, kFirstArenaAddress),
              "exact 64-layer AoSoA4 arena attaches in layer order");

  test.expect(!weights.attach_fp8_m1_output_projection_sidecars(
                  nullptr, kSidecarBytes) &&
                  sidecars_match(weights, kFirstArenaAddress),
              "null sidecar arena fails without changing attached views");
  test.expect(!weights.attach_fp8_m1_output_projection_sidecars(
                  second_arena, kSidecarBytes - 1U) &&
                  sidecars_match(weights, kFirstArenaAddress),
              "wrong sidecar byte count fails without changing views");
  test.expect(!weights.attach_fp8_m1_output_projection_sidecars(
                  reinterpret_cast<const std::uint8_t*>(
                      kSecondArenaAddress + 1U),
                  kSidecarBytes) &&
                  sidecars_match(weights, kFirstArenaAddress),
              "misaligned sidecar arena fails without changing views");

  runtime::Fp8LinearWeight* const last_output =
      mutable_attention_output(weights,
                               runtime::kQwen36DenseLayerCount - 1U);
  test.expect(last_output != nullptr,
              "last attention output is FP8 for atomicity test");
  if (last_output != nullptr) {
    const std::size_t original_input_size = last_output->input_size;
    last_output->input_size =
        runtime::kFp8M1OutputProjectionColumns - 1U;
    test.expect(!weights.attach_fp8_m1_output_projection_sidecars(
                    second_arena, kSidecarBytes) &&
                    sidecars_match(weights, kFirstArenaAddress),
                "late shape failure leaves all 64 prior views unchanged");
    last_output->input_size = original_input_size;
  }

  test.expect(weights.attach_fp8_m1_output_projection_sidecars(
                  second_arena, kSidecarBytes) &&
                  sidecars_match(weights, kSecondArenaAddress),
              "a valid replacement arena atomically updates all views");

  SyntheticArena mixed_arena;
  runtime::WeightBindResult mixed_result =
      runtime::bind_qwen36_27b_weights(mixed_arena.source());
  test.expect(mixed_result.ok(), "mixed projection synthetic ABI binds");
  if (mixed_result) {
    test.expect(!mixed_result.value->attach_fp8_m1_output_projection_sidecars(
                    first_arena, kSidecarBytes),
                "non-FP8 attention output rejects the sidecar contract");
  }
}

void test_fp8_prefill_qkv_register_feed_sidecar_attachment(
    TestContext& test) {
  static_assert(runtime::kFp8PrefillQkvRegisterFeedRows == 10'240U);
  static_assert(runtime::kFp8PrefillQkvRegisterFeedColumns == 5'120U);
  static_assert(
      runtime::kFp8PrefillQkvRegisterFeedSidecarBytesPerLayer ==
      52'428'800U);
  static_assert(runtime::kQwen36Fp8PrefillQkvRegisterFeedSidecarBytes ==
                2'516'582'400U);

  SyntheticArena arena(/*force_fp8_attention_outputs=*/false,
                       /*force_nvfp4_down_projections=*/false,
                       /*force_fp8_linear_qkv=*/true);
  runtime::WeightBindResult result =
      runtime::bind_qwen36_27b_weights(arena.source());
  test.expect(result.ok(), "all-FP8 linear QKV synthetic ABI binds");
  if (!result) {
    return;
  }
  runtime::ModelWeights& weights = *result.value;

  std::vector<std::size_t> linear_layers;
  std::size_t full_attention_layer = runtime::kQwen36DenseLayerCount;
  for (std::size_t layer_index = 0U;
       layer_index < runtime::kQwen36DenseLayerCount; ++layer_index) {
    if (std::holds_alternative<runtime::LinearAttentionWeights>(
            weights.layer(layer_index).attention)) {
      linear_layers.push_back(layer_index);
    } else if (full_attention_layer == runtime::kQwen36DenseLayerCount) {
      full_attention_layer = layer_index;
    }
  }
  test.expect(linear_layers.size() ==
                      runtime::kQwen36LinearAttentionLayerCount &&
                  full_attention_layer < runtime::kQwen36DenseLayerCount,
              "QKV sidecar test discovers the actual 48/16 schedule");

  using Descriptor =
      runtime::Fp8PrefillQkvRegisterFeedSidecarDescriptor;
  const std::vector<Descriptor> empty_state;
  test.expect(qkv_register_feed_attachments_match(weights, empty_state),
              "FP8 QKV register-feed views default to detached");

  constexpr std::size_t kSidecarBytes =
      runtime::kFp8PrefillQkvRegisterFeedSidecarBytesPerLayer;
  constexpr std::size_t kArenaBytes =
      runtime::kQwen36Fp8PrefillQkvRegisterFeedSidecarBytes;
  constexpr std::uintptr_t kFirstArenaAddress = 0x0000080000000000ULL;
  constexpr std::uintptr_t kSecondArenaAddress = 0x0000090000000000ULL;
  const auto pointer = [](const std::uintptr_t address) {
    return reinterpret_cast<const std::uint8_t*>(address);
  };
  const auto make_descriptors =
      [&](const std::uintptr_t arena_address) {
        std::vector<Descriptor> descriptors;
        descriptors.reserve(linear_layers.size());
        // Reverse descriptor order to prove binding is by explicit actual
        // layer index, not a contiguous or schedule-derived position.
        for (std::size_t index = 0U; index < linear_layers.size(); ++index) {
          const std::size_t layer_index =
              linear_layers[linear_layers.size() - 1U - index];
          descriptors.push_back(
              {layer_index, pointer(arena_address + index * kSidecarBytes),
               kSidecarBytes, runtime::kFp8PrefillQkvRegisterFeedRows,
               runtime::kFp8PrefillQkvRegisterFeedColumns});
        }
        return descriptors;
      };

  const std::vector<Descriptor> first_descriptors =
      make_descriptors(kFirstArenaAddress);
  test.expect(weights.attach_fp8_prefill_qkv_register_feed_sidecars(
                  pointer(kFirstArenaAddress), kArenaBytes,
                  first_descriptors.data(), first_descriptors.size()) &&
                  qkv_register_feed_attachments_match(weights,
                                                      first_descriptors),
              "exact unordered 48-layer QKV sidecars attach atomically");

  const auto expect_rejected_preserves =
      [&](const std::uint8_t* const candidate_arena,
          const std::size_t candidate_bytes,
          const Descriptor* const candidate_descriptors,
          const std::size_t candidate_count, const std::string_view label) {
        test.expect(
            !weights.attach_fp8_prefill_qkv_register_feed_sidecars(
                candidate_arena, candidate_bytes, candidate_descriptors,
                candidate_count) &&
                qkv_register_feed_attachments_match(weights,
                                                    first_descriptors),
            label);
      };

  expect_rejected_preserves(nullptr, kArenaBytes, first_descriptors.data(),
                            first_descriptors.size(),
                            "null QKV sidecar arena preserves prior set");
  expect_rejected_preserves(pointer(kSecondArenaAddress), kArenaBytes,
                            nullptr, first_descriptors.size(),
                            "null QKV descriptors preserve prior set");
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes - 16U,
                            first_descriptors.data(),
                            first_descriptors.size(),
                            "wrong QKV arena span is rejected atomically");
  expect_rejected_preserves(pointer(kSecondArenaAddress + 1U), kArenaBytes,
                            first_descriptors.data(),
                            first_descriptors.size(),
                            "misaligned QKV arena is rejected atomically");
  expect_rejected_preserves(
      pointer(std::numeric_limits<std::uintptr_t>::max() - 15U),
      kArenaBytes, first_descriptors.data(), first_descriptors.size(),
      "wrapping QKV arena span is rejected atomically");
  expect_rejected_preserves(
      pointer(kFirstArenaAddress),
      (first_descriptors.size() - 1U) * kSidecarBytes,
      first_descriptors.data(), first_descriptors.size() - 1U,
      "non-48 QKV descriptor count is rejected atomically");

  std::vector<Descriptor> invalid = first_descriptors;
  invalid.back().layer_index = invalid.front().layer_index;
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "duplicate QKV layer is rejected atomically");
  invalid = first_descriptors;
  invalid.back().layer_index = runtime::kQwen36DenseLayerCount;
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "out-of-range QKV layer is rejected atomically");
  invalid = first_descriptors;
  invalid.back().layer_index = full_attention_layer;
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "full-attention layer cannot claim a QKV sidecar");
  invalid = first_descriptors;
  invalid.back().bytes = kSidecarBytes - 16U;
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "wrong QKV descriptor byte span is atomic");
  invalid = first_descriptors;
  invalid.back().output_size =
      runtime::kFp8PrefillQkvRegisterFeedRows - 1U;
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "wrong QKV descriptor output shape is atomic");
  invalid = first_descriptors;
  invalid.back().input_size =
      runtime::kFp8PrefillQkvRegisterFeedColumns - 1U;
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "wrong QKV descriptor input shape is atomic");
  invalid = first_descriptors;
  invalid.back().sidecar = nullptr;
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "null QKV descriptor sidecar is atomic");
  invalid = first_descriptors;
  invalid.back().sidecar =
      pointer(reinterpret_cast<std::uintptr_t>(invalid.back().sidecar) + 1U);
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "misaligned QKV descriptor sidecar is atomic");
  invalid = first_descriptors;
  invalid.back().sidecar = pointer(kFirstArenaAddress - 16U);
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "QKV sidecar before arena is rejected atomically");
  invalid = first_descriptors;
  invalid.back().sidecar =
      pointer(kFirstArenaAddress + kArenaBytes - kSidecarBytes + 16U);
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "QKV sidecar past arena is rejected atomically");
  invalid = first_descriptors;
  invalid[1U].sidecar = invalid[0U].sidecar;
  expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                            invalid.data(), invalid.size(),
                            "overlapping QKV sidecars are rejected atomically");

  runtime::Fp8LinearWeight* const late_qkv =
      mutable_linear_attention_qkv(weights,
                                   first_descriptors.back().layer_index);
  test.expect(late_qkv != nullptr,
              "late QKV target exists for shape atomicity test");
  if (late_qkv != nullptr) {
    const std::size_t original_input_size = late_qkv->input_size;
    late_qkv->input_size =
        runtime::kFp8PrefillQkvRegisterFeedColumns - 1U;
    expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                              first_descriptors.data(),
                              first_descriptors.size(),
                              "late QKV target shape failure preserves set");
    late_qkv->input_size = original_input_size;

    const std::size_t original_output_size = late_qkv->output_size;
    late_qkv->output_size =
        runtime::kFp8PrefillQkvRegisterFeedRows - 1U;
    expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                              first_descriptors.data(),
                              first_descriptors.size(),
                              "late QKV output shape failure preserves set");
    late_qkv->output_size = original_output_size;

    const std::uint8_t* const original_weight = late_qkv->weight;
    late_qkv->weight = nullptr;
    expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                              first_descriptors.data(),
                              first_descriptors.size(),
                              "late invalid FP8 QKV payload preserves set");
    late_qkv->weight = original_weight;

    const float* const original_scale = late_qkv->weight_scale_device;
    late_qkv->weight_scale_device = nullptr;
    expect_rejected_preserves(pointer(kFirstArenaAddress), kArenaBytes,
                              first_descriptors.data(),
                              first_descriptors.size(),
                              "late QKV null scale payload preserves set");
    late_qkv->weight_scale_device = original_scale;
  }

  expect_rejected_preserves(nullptr, 0U, first_descriptors.data(), 0U,
                            "noncanonical QKV detach preserves prior set");

  const std::vector<Descriptor> second_descriptors =
      make_descriptors(kSecondArenaAddress);
  test.expect(weights.attach_fp8_prefill_qkv_register_feed_sidecars(
                  pointer(kSecondArenaAddress), kArenaBytes,
                  second_descriptors.data(), second_descriptors.size()) &&
                  qkv_register_feed_attachments_match(weights,
                                                      second_descriptors),
              "valid QKV replacement atomically changes all 48 views");
  test.expect(weights.attach_fp8_prefill_qkv_register_feed_sidecars(
                  nullptr, 0U, nullptr, 0U) &&
                  qkv_register_feed_attachments_match(weights, empty_state),
              "canonical empty QKV call detaches every sidecar");

  SyntheticArena mixed_arena;
  runtime::WeightBindResult mixed_result =
      runtime::bind_qwen36_27b_weights(mixed_arena.source());
  test.expect(mixed_result.ok(), "mixed QKV synthetic ABI binds");
  if (mixed_result) {
    test.expect(
        !mixed_result.value
             ->attach_fp8_prefill_qkv_register_feed_sidecars(
                 pointer(kFirstArenaAddress), kArenaBytes,
                 first_descriptors.data(), first_descriptors.size()),
        "non-FP8 linear QKV payload rejects the sidecar contract");
  }
}

void test_fp8_prefill_supermatrix_sidecar_attachment(TestContext& test) {
  static_assert(runtime::kFp8PrefillSupermatrixProjectionCount == 208U);
  static_assert(runtime::kQwen36Fp8PrefillSupermatrixSidecarBytes ==
                7'214'202'880ULL);

  SyntheticArena arena(/*force_fp8_attention_outputs=*/false,
                       /*force_nvfp4_down_projections=*/false,
                       /*force_fp8_linear_qkv=*/false,
                       /*force_fp8_prefill_supermatrix=*/true);
  runtime::WeightBindResult result =
      runtime::bind_qwen36_27b_weights(arena.source());
  test.expect(result.ok(),
              "all-FP8 Prefill supermatrix synthetic ABI binds");
  if (!result) {
    return;
  }
  runtime::ModelWeights& weights = *result.value;
  const std::vector<Fp8PrefillSupermatrixProjection> projections =
      mutable_fp8_prefill_supermatrix_projections(weights);
  test.expect(projections.size() ==
                  runtime::kFp8PrefillSupermatrixProjectionCount,
              "supermatrix attachment enumerates the fixed 208 projections");

  const std::vector<const std::uint8_t*> detached =
      fp8_prefill_supermatrix_attachment_snapshot(projections);
  test.expect(std::all_of(detached.begin(), detached.end(),
                          [](const std::uint8_t* const sidecar) {
                            return sidecar == nullptr;
                          }),
              "all 208 supermatrix views default to detached");

  constexpr std::size_t kArenaBytes =
      runtime::kQwen36Fp8PrefillSupermatrixSidecarBytes;
  constexpr std::uintptr_t kFirstArenaAddress =
      0x0000100000000000ULL;
  constexpr std::uintptr_t kSecondArenaAddress =
      0x0000200000000000ULL;
  const auto pointer = [](const std::uintptr_t address) {
    return reinterpret_cast<const std::uint8_t*>(address);
  };

  test.expect(weights.attach_fp8_prefill_supermatrix_sidecars(
                  pointer(kFirstArenaAddress), kArenaBytes) &&
                  fp8_prefill_supermatrix_attachments_match_layout(
                      projections, kFirstArenaAddress),
              "exact supermatrix arena attaches all 208 offsets in ABI order");
  const std::vector<const std::uint8_t*> first_snapshot =
      fp8_prefill_supermatrix_attachment_snapshot(projections);

  const auto expect_rejected_preserves =
      [&](const std::uint8_t* const candidate_arena,
          const std::size_t candidate_bytes, const std::string_view label) {
        test.expect(
            !weights.attach_fp8_prefill_supermatrix_sidecars(
                candidate_arena, candidate_bytes) &&
                fp8_prefill_supermatrix_attachments_match_snapshot(
                    projections, first_snapshot),
            label);
      };

  expect_rejected_preserves(nullptr, kArenaBytes,
                            "null supermatrix arena is rejected atomically");
  expect_rejected_preserves(pointer(kSecondArenaAddress), kArenaBytes - 1U,
                            "short supermatrix arena is rejected atomically");
  expect_rejected_preserves(pointer(kSecondArenaAddress), kArenaBytes + 1U,
                            "long supermatrix arena is rejected atomically");
  expect_rejected_preserves(pointer(kSecondArenaAddress + 1U), kArenaBytes,
                            "misaligned supermatrix arena is atomic");
  expect_rejected_preserves(
      pointer(std::numeric_limits<std::uintptr_t>::max() - 15U),
      kArenaBytes, "wrapping supermatrix arena is rejected atomically");

  test.expect(!projections.empty() && projections.back().weight != nullptr,
              "late supermatrix target exists for atomicity tests");
  if (!projections.empty() && projections.back().weight != nullptr) {
    runtime::Fp8LinearWeight& late = *projections.back().weight;
    const std::size_t original_input_size = late.input_size;
    late.input_size = original_input_size - 1U;
    expect_rejected_preserves(pointer(kSecondArenaAddress), kArenaBytes,
                              "late supermatrix shape failure is atomic");
    late.input_size = original_input_size;

    const std::uint8_t* const original_weight = late.weight;
    late.weight = nullptr;
    expect_rejected_preserves(pointer(kSecondArenaAddress), kArenaBytes,
                              "late supermatrix payload failure is atomic");
    late.weight = original_weight;
  }

  test.expect(weights.attach_fp8_prefill_supermatrix_sidecars(
                  pointer(kSecondArenaAddress), kArenaBytes) &&
                  fp8_prefill_supermatrix_attachments_match_layout(
                      projections, kSecondArenaAddress),
              "valid supermatrix replacement atomically updates 208 views");
  test.expect(weights.attach_fp8_prefill_supermatrix_sidecars(nullptr, 0U) &&
                  fp8_prefill_supermatrix_attachments_match_snapshot(
                      projections, detached),
              "canonical empty supermatrix call detaches all 208 views");
}

void test_p40_packed_projection_sidecar_attachment(TestContext& test) {
  namespace kernels = q3x::kernels;
  static_assert(kernels::kSm87P40PackedProjectionArtifactCount == 256U);
  static_assert(kernels::kSm87P40PackedProjectionFp8LogicalRoleCount ==
                208U);

  SyntheticArena arena(/*force_fp8_attention_outputs=*/false,
                       /*force_nvfp4_down_projections=*/false,
                       /*force_fp8_linear_qkv=*/false,
                       /*force_fp8_prefill_supermatrix=*/false,
                       /*force_p40_packed_projection=*/true);
  runtime::WeightBindResult result =
      runtime::bind_qwen36_27b_weights(arena.source());
  test.expect(result.ok(), "P40 packed synthetic model ABI binds");
  if (!result) {
    return;
  }
  runtime::ModelWeights& weights = *result.value;

  constexpr std::uintptr_t kBaseAddress = 0x0000300000000000ULL;
  std::vector<runtime::P40PackedProjectionSidecarDescriptor> descriptors;
  descriptors.reserve(kernels::kSm87P40PackedProjectionArtifactCount);
  std::uintptr_t next_address = kBaseAddress;
  std::uint64_t next_identity = 1U;
  const auto append = [&](const std::size_t layer_index,
                          const kernels::Sm87P40PackedProjectionRole role) {
    const kernels::Sm87P40PackedProjectionPlan plan =
        kernels::sm87_p40_packed_projection_plan(role);
    kernels::Sm87P40PackedProjectionDeviceView view;
    view.payload = reinterpret_cast<const std::uint8_t*>(next_address);
    view.payload_bytes = plan.payload_bytes;
    view.artifact_identity = next_identity++;
    view.role = role;
    view.tactic = plan.tactic;
    view.source_count = plan.source_count;
    for (std::size_t source = 0U; source < plan.source_count; ++source) {
      view.scalar_scales[source] =
          role == kernels::Sm87P40PackedProjectionRole::kNvFp4GateUp ||
                  role == kernels::Sm87P40PackedProjectionRole::kNvFp4Down
              ? 0.03125F
              : 0.125F;
    }
    descriptors.push_back({layer_index, view});
    next_address += plan.payload_bytes;
  };
  for (std::size_t layer = 0U;
       layer < kernels::kSm87P40PackedProjectionLayerCount; ++layer) {
    append(layer, kernels::Sm87P40PackedProjectionRole::kNvFp4GateUp);
    append(layer, kernels::Sm87P40PackedProjectionRole::kNvFp4Down);
    append(layer, kernels::sm87_p40_packed_is_full_layer(layer)
                      ? kernels::Sm87P40PackedProjectionRole::kFp8FullQkv
                      : kernels::Sm87P40PackedProjectionRole::kFp8LinearQkvZ);
    append(layer,
           kernels::Sm87P40PackedProjectionRole::kFp8AttentionOutput);
  }
  test.expect(descriptors.size() ==
                  kernels::kSm87P40PackedProjectionArtifactCount,
              "packed descriptor builder covers four artifacts per layer");

  test.expect(weights.attach_p40_packed_projection_sidecars(
                  descriptors.data(), descriptors.size()),
              "complete disjoint packed inventory attaches atomically");
  const auto layer_zero_gate =
      std::get<runtime::NvFp4LinearWeight>(weights.layer(0U).mlp.gate_proj)
          .prefill_p40_packed_artifact;
  const auto layer_zero_up =
      std::get<runtime::NvFp4LinearWeight>(weights.layer(0U).mlp.up_proj)
          .prefill_p40_packed_artifact;
  const auto& layer_zero_linear =
      std::get<runtime::LinearAttentionWeights>(
          weights.layer(0U).attention);
  const auto layer_zero_qkv =
      std::get<runtime::Fp8LinearWeight>(layer_zero_linear.in_proj_qkv)
          .prefill_p40_packed_artifact;
  const auto layer_zero_z =
      std::get<runtime::Fp8LinearWeight>(layer_zero_linear.in_proj_z)
          .prefill_p40_packed_artifact;
  test.expect(
      layer_zero_gate.artifact_identity == descriptors[0U].view.artifact_identity &&
          layer_zero_up.artifact_identity == layer_zero_gate.artifact_identity &&
          layer_zero_qkv.artifact_identity ==
              descriptors[2U].view.artifact_identity &&
          layer_zero_z.artifact_identity == layer_zero_qkv.artifact_identity,
      "Gate/Up and grouped FP8 logical sources share only their declared "
      "physical artifacts");

  const auto expect_rejected_preserves =
      [&](std::vector<runtime::P40PackedProjectionSidecarDescriptor> invalid,
          const std::string_view message) {
        test.expect(
            !weights.attach_p40_packed_projection_sidecars(
                invalid.data(), invalid.size()) &&
                std::get<runtime::NvFp4LinearWeight>(
                    weights.layer(0U).mlp.gate_proj)
                        .prefill_p40_packed_artifact.artifact_identity ==
                    layer_zero_gate.artifact_identity,
            message);
      };
  auto invalid = descriptors;
  invalid.back().view.artifact_identity = invalid.front().view.artifact_identity;
  expect_rejected_preserves(std::move(invalid),
                            "duplicate artifact identity is atomic");
  invalid = descriptors;
  invalid[1U].view.payload = invalid[0U].view.payload;
  expect_rejected_preserves(std::move(invalid),
                            "overlapping artifact payload is atomic");
  invalid = descriptors;
  invalid[2U].view.source_count = 1U;
  expect_rejected_preserves(std::move(invalid),
                            "grouped FP8 source loss is atomic");
  invalid = descriptors;
  invalid[0U].view.scalar_scales[2U] = 1.0F;
  expect_rejected_preserves(std::move(invalid),
                            "undeclared scalar scale is atomic");
  test.expect(!weights.attach_p40_packed_projection_sidecars(
                  descriptors.data(), descriptors.size() - 1U),
              "incomplete physical inventory is rejected");

  test.expect(weights.attach_p40_packed_projection_sidecars(nullptr, 0U) &&
                  std::get<runtime::NvFp4LinearWeight>(
                      weights.layer(0U).mlp.gate_proj)
                          .prefill_p40_packed_artifact.payload == nullptr &&
                  std::get<runtime::Fp8LinearWeight>(
                      layer_zero_linear.in_proj_qkv)
                          .prefill_p40_packed_artifact.payload == nullptr,
              "canonical empty call detaches the complete packed inventory");
}

void test_nvfp4_down_scale6_sidecar_attachment(TestContext& test) {
  static_assert(runtime::kNvFp4DownScale6Rows == 5'120U);
  static_assert(runtime::kNvFp4DownScale6Columns == 17'408U);
  static_assert(runtime::kNvFp4DownScale6SidecarBytesPerProjection ==
                4'177'920U);

  SyntheticArena arena(/*force_fp8_attention_outputs=*/false,
                       /*force_nvfp4_down_projections=*/true);
  runtime::WeightBindResult result =
      runtime::bind_qwen36_27b_weights(arena.source());
  test.expect(result.ok(), "all-NVFP4 down synthetic ABI binds");
  if (!result) {
    return;
  }
  runtime::ModelWeights& weights = *result.value;
  const std::vector<runtime::NvFp4DownScale6SidecarDescriptor> empty_state;
  test.expect(down_scale6_attachments_match(weights, empty_state),
              "NVFP4 down scale6 views default to detached");

  constexpr std::size_t kSidecarBytes =
      runtime::kNvFp4DownScale6SidecarBytesPerProjection;
  constexpr std::uintptr_t kFirstArenaAddress = 0x0000060000000000ULL;
  constexpr std::uintptr_t kSecondArenaAddress = 0x0000070000000000ULL;
  const auto pointer = [](const std::uintptr_t address) {
    return reinterpret_cast<const std::uint8_t*>(address);
  };
  const auto* const first_arena = pointer(kFirstArenaAddress);
  const auto* const second_arena = pointer(kSecondArenaAddress);
  const std::array<runtime::NvFp4DownScale6SidecarDescriptor, 2U>
      first_descriptors{{
          {5U, first_arena, kSidecarBytes, 66U,
           runtime::kNvFp4DownScale6Rows,
           runtime::kNvFp4DownScale6Columns},
          {61U, pointer(kFirstArenaAddress + kSidecarBytes), kSidecarBytes,
           73U, runtime::kNvFp4DownScale6Rows,
           runtime::kNvFp4DownScale6Columns},
      }};
  const std::vector<runtime::NvFp4DownScale6SidecarDescriptor> first_state(
      first_descriptors.begin(), first_descriptors.end());
  test.expect(weights.attach_nvfp4_down_scale6_sidecars(
                  first_arena, 2U * kSidecarBytes,
                  first_descriptors.data(), first_descriptors.size()) &&
                  down_scale6_attachments_match(weights, first_state),
              "sparse exact down scale6 descriptors attach atomically");

  const auto expect_rejected_preserves =
      [&](const std::uint8_t* const candidate_arena,
          const std::size_t candidate_bytes,
          const runtime::NvFp4DownScale6SidecarDescriptor*
              const candidate_descriptors,
          const std::size_t candidate_count, const std::string_view label) {
        test.expect(!weights.attach_nvfp4_down_scale6_sidecars(
                        candidate_arena, candidate_bytes,
                        candidate_descriptors, candidate_count) &&
                        down_scale6_attachments_match(weights, first_state),
                    label);
      };

  expect_rejected_preserves(nullptr, 2U * kSidecarBytes,
                            first_descriptors.data(),
                            first_descriptors.size(),
                            "null scale6 arena preserves prior sparse set");
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, nullptr,
                            first_descriptors.size(),
                            "null scale6 descriptors preserve prior set");
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes - 1U,
                            first_descriptors.data(),
                            first_descriptors.size(),
                            "wrong scale6 arena byte count is atomic");
  expect_rejected_preserves(pointer(kSecondArenaAddress + 1U),
                            2U * kSidecarBytes,
                            first_descriptors.data(),
                            first_descriptors.size(),
                            "misaligned scale6 arena is atomic");

  auto invalid = first_descriptors;
  invalid[1U].layer_index = invalid[0U].layer_index;
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "duplicate scale6 layer is rejected atomically");
  invalid = first_descriptors;
  invalid[1U].layer_index = runtime::kQwen36DenseLayerCount;
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "out-of-range scale6 layer is rejected atomically");
  invalid = first_descriptors;
  invalid[1U].bytes = kSidecarBytes - 1U;
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "wrong descriptor byte count is rejected atomically");
  invalid = first_descriptors;
  invalid[1U].scale_base = 193U;
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "out-of-range scale base is rejected atomically");
  invalid = first_descriptors;
  invalid[1U].output_size = runtime::kNvFp4DownScale6Rows - 1U;
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "wrong descriptor output shape is atomic");
  invalid = first_descriptors;
  invalid[1U].input_size = runtime::kNvFp4DownScale6Columns - 16U;
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "wrong descriptor input shape is atomic");
  invalid = first_descriptors;
  invalid[1U].sidecar = nullptr;
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "null descriptor sidecar is rejected atomically");
  invalid = first_descriptors;
  invalid[1U].sidecar =
      pointer(kFirstArenaAddress + kSidecarBytes + 1U);
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "misaligned descriptor sidecar is atomic");
  invalid = first_descriptors;
  invalid[0U].sidecar = pointer(kFirstArenaAddress - 32U);
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "sidecar before arena span is rejected atomically");
  invalid = first_descriptors;
  invalid[1U].sidecar =
      pointer(kFirstArenaAddress + kSidecarBytes + 32U);
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "sidecar past arena span is rejected atomically");
  invalid = first_descriptors;
  invalid[1U].sidecar =
      pointer(kFirstArenaAddress + kSidecarBytes - 32U);
  expect_rejected_preserves(first_arena, 2U * kSidecarBytes, invalid.data(),
                            invalid.size(),
                            "overlapping scale6 ranges are rejected atomically");

  runtime::NvFp4LinearWeight* const late_down =
      mutable_nvfp4_down(weights, first_descriptors.back().layer_index);
  test.expect(late_down != nullptr,
              "late down target exists for shape atomicity test");
  if (late_down != nullptr) {
    const std::size_t original_input_size = late_down->input_size;
    late_down->input_size = runtime::kNvFp4DownScale6Columns - 16U;
    expect_rejected_preserves(first_arena, 2U * kSidecarBytes,
                              first_descriptors.data(),
                              first_descriptors.size(),
                              "late target shape failure preserves prior set");
    late_down->input_size = original_input_size;
  }

  expect_rejected_preserves(nullptr, 0U, first_descriptors.data(), 0U,
                            "noncanonical empty detach preserves prior set");

  const std::array<runtime::NvFp4DownScale6SidecarDescriptor, 1U>
      replacement{{
          {11U, second_arena, kSidecarBytes, 74U,
           runtime::kNvFp4DownScale6Rows,
           runtime::kNvFp4DownScale6Columns},
      }};
  const std::vector<runtime::NvFp4DownScale6SidecarDescriptor>
      replacement_state(replacement.begin(), replacement.end());
  test.expect(weights.attach_nvfp4_down_scale6_sidecars(
                  second_arena, kSidecarBytes, replacement.data(),
                  replacement.size()) &&
                  down_scale6_attachments_match(weights, replacement_state),
              "valid sparse replacement clears the prior attachment set");
  test.expect(weights.attach_nvfp4_down_scale6_sidecars(
                  nullptr, 0U, nullptr, 0U) &&
                  down_scale6_attachments_match(weights, empty_state),
              "canonical empty call detaches every down scale6 sidecar");
}

void test_projection_pair_eligibility(TestContext& test) {
  std::uint16_t first_storage = 0U;
  std::uint16_t second_storage = 0U;
  const runtime::LinearWeight first = runtime::Bf16LinearWeight{
      &first_storage, 48U, 5120U};
  const runtime::LinearWeight second = runtime::Bf16LinearWeight{
      &second_storage, 48U, 5120U};
  test.expect(runtime::supports_bf16_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first,
                  second),
              "SM87 accepts only the exact BF16 A/B projection pair");
  test.expect(!runtime::supports_bf16_projection_pair(
                  runtime::ProjectionBackend::kReference, first, second),
              "reference backend preserves independent BF16 projections");

  const runtime::LinearWeight wrong_rows = runtime::Bf16LinearWeight{
      &second_storage, 47U, 5120U};
  const runtime::LinearWeight wrong_columns = runtime::Bf16LinearWeight{
      &second_storage, 48U, 5119U};
  const runtime::LinearWeight null_weight = runtime::Bf16LinearWeight{
      nullptr, 48U, 5120U};
  test.expect(!runtime::supports_bf16_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first,
                  wrong_rows) &&
                  !runtime::supports_bf16_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly, first,
                      wrong_columns) &&
                  !runtime::supports_bf16_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly, first,
                      null_weight),
              "pair eligibility rejects near-miss shapes and null payloads");

  std::uint8_t fp8_storage = 0U;
  float scale = 1.0F;
  const runtime::LinearWeight fp8 = runtime::Fp8LinearWeight{
      &fp8_storage, &scale, &scale, 1.0F, 1.0F, 48U, 5120U};
  test.expect(!runtime::supports_bf16_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first, fp8),
              "pair eligibility does not fuse a non-BF16 projection");

  std::uint8_t second_fp8_storage = 0U;
  float second_scale = 0.5F;
  const runtime::LinearWeight first_fp8 = runtime::Fp8LinearWeight{
      &fp8_storage, &scale, &scale, 1.0F, 1.0F, 1'024U, 5'120U};
  const runtime::LinearWeight second_fp8 = runtime::Fp8LinearWeight{
      &second_fp8_storage, &second_scale, &second_scale, 0.5F, 1.0F,
      1'024U, 5'120U};
  test.expect(runtime::supports_fp8_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first_fp8,
                  second_fp8),
              "SM87 accepts the exact FP8 K/V projection pair");
  test.expect(!runtime::supports_fp8_projection_pair(
                  runtime::ProjectionBackend::kReference, first_fp8,
                  second_fp8),
              "reference backend preserves independent FP8 projections");

  const runtime::LinearWeight wrong_fp8_rows = runtime::Fp8LinearWeight{
      &second_fp8_storage, &second_scale, &second_scale, 0.5F, 1.0F,
      1'023U, 5'120U};
  const runtime::LinearWeight wrong_fp8_columns = runtime::Fp8LinearWeight{
      &second_fp8_storage, &second_scale, &second_scale, 0.5F, 1.0F,
      1'024U, 5'119U};
  const runtime::LinearWeight missing_fp8_companion =
      runtime::Fp8LinearWeight{&second_fp8_storage, nullptr, &second_scale,
                               0.5F, 1.0F, 1'024U, 5'120U};
  const runtime::LinearWeight invalid_fp8_scale = runtime::Fp8LinearWeight{
      &second_fp8_storage, &second_scale, &second_scale,
      std::numeric_limits<float>::infinity(), 1.0F, 1'024U, 5'120U};
  test.expect(!runtime::supports_fp8_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, first_fp8,
                  wrong_fp8_rows) &&
                  !runtime::supports_fp8_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly,
                      first_fp8, wrong_fp8_columns) &&
                  !runtime::supports_fp8_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly,
                      first_fp8, missing_fp8_companion) &&
                  !runtime::supports_fp8_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly,
                      first_fp8, invalid_fp8_scale) &&
                  !runtime::supports_fp8_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly, first,
                      second_fp8),
              "FP8 pair eligibility rejects near-miss and malformed pairs");
}

void test_fp8_qkv_z_projection_pair_eligibility(TestContext& test) {
  std::uint8_t qkv_storage = 0U;
  std::uint8_t z_storage = 0U;
  float qkv_weight_scale_device = 1.0F;
  float qkv_input_scale_device = 1.0F;
  float z_weight_scale_device = 0.5F;
  float z_input_scale_device = 0.25F;
  const runtime::LinearWeight qkv = runtime::Fp8LinearWeight{
      &qkv_storage, &qkv_weight_scale_device, &qkv_input_scale_device,
      1.0F, 0.75F, 10'240U, 5'120U};
  const runtime::LinearWeight z = runtime::Fp8LinearWeight{
      &z_storage, &z_weight_scale_device, &z_input_scale_device,
      0.5F, 0.25F, 6'144U, 5'120U};

  test.expect(runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, qkv, z),
              "SM87 accepts the exact ordered FP8 QKV/Z projection pair");
  test.expect(!runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, z, qkv),
              "QKV/Z eligibility rejects the reversed projection order");
  test.expect(!runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kReference, qkv, z),
              "reference backend preserves independent QKV/Z projections");

  const runtime::LinearWeight wrong_qkv_rows = runtime::Fp8LinearWeight{
      &qkv_storage, &qkv_weight_scale_device, &qkv_input_scale_device,
      1.0F, 0.75F, 10'239U, 5'120U};
  const runtime::LinearWeight wrong_z_rows = runtime::Fp8LinearWeight{
      &z_storage, &z_weight_scale_device, &z_input_scale_device,
      0.5F, 0.25F, 6'143U, 5'120U};
  const runtime::LinearWeight wrong_qkv_columns = runtime::Fp8LinearWeight{
      &qkv_storage, &qkv_weight_scale_device, &qkv_input_scale_device,
      1.0F, 0.75F, 10'240U, 5'119U};
  const runtime::LinearWeight wrong_z_columns = runtime::Fp8LinearWeight{
      &z_storage, &z_weight_scale_device, &z_input_scale_device,
      0.5F, 0.25F, 6'144U, 5'119U};
  test.expect(!runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  wrong_qkv_rows, z) &&
                  !runtime::supports_fp8_qkv_z_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                      wrong_z_rows) &&
                  !runtime::supports_fp8_qkv_z_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly,
                      wrong_qkv_columns, z) &&
                  !runtime::supports_fp8_qkv_z_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                      wrong_z_columns),
              "QKV/Z eligibility rejects every near-miss matrix shape");

  std::uint16_t bf16_storage = 0U;
  const runtime::LinearWeight bf16_qkv = runtime::Bf16LinearWeight{
      &bf16_storage, 10'240U, 5'120U};
  test.expect(!runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, bf16_qkv, z),
              "QKV/Z eligibility rejects a non-FP8 projection");

  const runtime::LinearWeight null_qkv_payload = runtime::Fp8LinearWeight{
      nullptr, &qkv_weight_scale_device, &qkv_input_scale_device,
      1.0F, 0.75F, 10'240U, 5'120U};
  const runtime::LinearWeight missing_qkv_weight_scale =
      runtime::Fp8LinearWeight{&qkv_storage, nullptr, &qkv_input_scale_device,
                               1.0F, 0.75F, 10'240U, 5'120U};
  const runtime::LinearWeight missing_z_input_scale = runtime::Fp8LinearWeight{
      &z_storage, &z_weight_scale_device, nullptr,
      0.5F, 0.25F, 6'144U, 5'120U};
  test.expect(!runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  null_qkv_payload, z) &&
                  !runtime::supports_fp8_qkv_z_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly,
                      missing_qkv_weight_scale, z) &&
                  !runtime::supports_fp8_qkv_z_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                      missing_z_input_scale),
              "QKV/Z eligibility rejects null payload and scale companions");

  const runtime::LinearWeight negative_qkv_weight_scale =
      runtime::Fp8LinearWeight{
          &qkv_storage, &qkv_weight_scale_device, &qkv_input_scale_device,
          -0.01F, 0.75F, 10'240U, 5'120U};
  const runtime::LinearWeight nan_z_input_scale = runtime::Fp8LinearWeight{
      &z_storage, &z_weight_scale_device, &z_input_scale_device,
      0.5F, std::numeric_limits<float>::quiet_NaN(), 6'144U, 5'120U};
  const runtime::LinearWeight infinite_z_weight_scale =
      runtime::Fp8LinearWeight{
          &z_storage, &z_weight_scale_device, &z_input_scale_device,
          std::numeric_limits<float>::infinity(), 0.25F, 6'144U, 5'120U};
  test.expect(!runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  negative_qkv_weight_scale, z) &&
                  !runtime::supports_fp8_qkv_z_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                      nan_z_input_scale) &&
                  !runtime::supports_fp8_qkv_z_projection_pair(
                      runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                      infinite_z_weight_scale),
              "QKV/Z eligibility rejects negative and non-finite scales");

  const runtime::LinearWeight zero_scales = runtime::Fp8LinearWeight{
      &z_storage, &z_weight_scale_device, &z_input_scale_device,
      0.0F, 0.0F, 6'144U, 5'120U};
  test.expect(runtime::supports_fp8_qkv_z_projection_pair(
                  runtime::ProjectionBackend::kSm87WeightOnly, qkv,
                  zero_scales),
              "QKV/Z eligibility accepts finite zero scales");
}

void test_linear_attention_qkv_z_ab_projection_fusion_eligibility(
    TestContext& test) {
  std::uint8_t qkv_storage = 0U;
  std::uint8_t z_storage = 0U;
  std::uint16_t a_storage = 0U;
  std::uint16_t b_storage = 0U;
  float qkv_weight_scale_device = 1.0F;
  float qkv_input_scale_device = 1.0F;
  float z_weight_scale_device = 0.5F;
  float z_input_scale_device = 0.25F;
  const runtime::LinearWeight qkv = runtime::Fp8LinearWeight{
      &qkv_storage, &qkv_weight_scale_device, &qkv_input_scale_device,
      1.0F, 0.75F, 10'240U, 5'120U};
  const runtime::LinearWeight z = runtime::Fp8LinearWeight{
      &z_storage, &z_weight_scale_device, &z_input_scale_device,
      0.5F, 0.25F, 6'144U, 5'120U};
  const runtime::LinearWeight a = runtime::Bf16LinearWeight{
      &a_storage, 48U, 5'120U};
  const runtime::LinearWeight b = runtime::Bf16LinearWeight{
      &b_storage, 48U, 5'120U};

  test.expect(runtime::supports_linear_attention_qkv_z_ab_projection_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, a,
                  b),
              "SM87 accepts the exact linear-attention QKV/Z/A/B group");
  test.expect(!runtime::supports_linear_attention_qkv_z_ab_projection_fusion(
                  runtime::ProjectionBackend::kReference, qkv, z, a, b),
              "reference backend rejects linear-attention QKV/Z/A/B fusion");

  const runtime::LinearWeight near_miss_z = runtime::Fp8LinearWeight{
      &z_storage, &z_weight_scale_device, &z_input_scale_device,
      0.5F, 0.25F, 6'143U, 5'120U};
  const runtime::LinearWeight near_miss_a = runtime::Bf16LinearWeight{
      &a_storage, 48U, 5'119U};
  test.expect(
      !runtime::supports_linear_attention_qkv_z_ab_projection_fusion(
          runtime::ProjectionBackend::kSm87WeightOnly, qkv, near_miss_z, a,
          b) &&
          !runtime::supports_linear_attention_qkv_z_ab_projection_fusion(
              runtime::ProjectionBackend::kSm87WeightOnly, qkv, z,
              near_miss_a, b),
      "linear-attention QKV/Z/A/B fusion rejects near-miss shapes");

  const runtime::LinearWeight fp8_a = runtime::Fp8LinearWeight{
      &qkv_storage, &qkv_weight_scale_device, &qkv_input_scale_device,
      1.0F, 0.75F, 48U, 5'120U};
  const runtime::LinearWeight bf16_qkv = runtime::Bf16LinearWeight{
      &a_storage, 10'240U, 5'120U};
  test.expect(
      !runtime::supports_linear_attention_qkv_z_ab_projection_fusion(
          runtime::ProjectionBackend::kSm87WeightOnly, qkv, z, fp8_a, b) &&
          !runtime::supports_linear_attention_qkv_z_ab_projection_fusion(
              runtime::ProjectionBackend::kSm87WeightOnly, bf16_qkv, z, a,
              b),
      "linear-attention QKV/Z/A/B fusion enforces FP8/FP8/BF16/BF16 types");
}

void test_nvfp4_gate_up_silu_fusion_eligibility(TestContext& test) {
  std::uint8_t gate_packed = 0U;
  std::uint8_t gate_scales = 0U;
  std::uint8_t up_packed = 0U;
  std::uint8_t up_scales = 0U;
  float gate_scale_2_device = 1.0F;
  float gate_input_scale_device = 1.0F;
  float up_scale_2_device = 0.5F;
  float up_input_scale_device = 0.25F;
  const runtime::LinearWeight gate = runtime::NvFp4LinearWeight{
      &gate_packed, &gate_scales, &gate_scale_2_device,
      &gate_input_scale_device, 1.0F, 0.75F, 17'408U, 5'120U};
  const runtime::LinearWeight up = runtime::NvFp4LinearWeight{
      &up_packed, &up_scales, &up_scale_2_device, &up_input_scale_device,
      0.5F, 0.25F, 17'408U, 5'120U};

  test.expect(runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, gate, up),
              "SM87 accepts the exact NVFP4 gate/up SiLU fusion payload");
  test.expect(!runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kReference, gate, up),
              "reference backend preserves ordered gate/up/SiLU launches");

  const runtime::LinearWeight wrong_gate_rows = runtime::NvFp4LinearWeight{
      &gate_packed, &gate_scales, &gate_scale_2_device,
      &gate_input_scale_device, 1.0F, 0.75F, 17'407U, 5'120U};
  const runtime::LinearWeight wrong_up_columns = runtime::NvFp4LinearWeight{
      &up_packed, &up_scales, &up_scale_2_device, &up_input_scale_device,
      0.5F, 0.25F, 17'408U, 5'104U};
  const runtime::LinearWeight null_gate_packed = runtime::NvFp4LinearWeight{
      nullptr, &gate_scales, &gate_scale_2_device, &gate_input_scale_device,
      1.0F, 0.75F, 17'408U, 5'120U};
  const runtime::LinearWeight null_up_scales = runtime::NvFp4LinearWeight{
      &up_packed, nullptr, &up_scale_2_device, &up_input_scale_device, 0.5F,
      0.25F, 17'408U, 5'120U};
  const runtime::LinearWeight missing_gate_scale_2 =
      runtime::NvFp4LinearWeight{
          &gate_packed, &gate_scales, nullptr, &gate_input_scale_device,
          1.0F, 0.75F, 17'408U, 5'120U};
  const runtime::LinearWeight missing_up_input_scale =
      runtime::NvFp4LinearWeight{
          &up_packed, &up_scales, &up_scale_2_device, nullptr, 0.5F, 0.25F,
          17'408U, 5'120U};
  test.expect(
      !runtime::supports_nvfp4_gate_up_silu_fusion(
          runtime::ProjectionBackend::kSm87WeightOnly, wrong_gate_rows, up) &&
          !runtime::supports_nvfp4_gate_up_silu_fusion(
              runtime::ProjectionBackend::kSm87WeightOnly, gate,
              wrong_up_columns) &&
          !runtime::supports_nvfp4_gate_up_silu_fusion(
              runtime::ProjectionBackend::kSm87WeightOnly,
              null_gate_packed, up) &&
          !runtime::supports_nvfp4_gate_up_silu_fusion(
              runtime::ProjectionBackend::kSm87WeightOnly, gate,
              null_up_scales) &&
          !runtime::supports_nvfp4_gate_up_silu_fusion(
              runtime::ProjectionBackend::kSm87WeightOnly,
              missing_gate_scale_2, up) &&
          !runtime::supports_nvfp4_gate_up_silu_fusion(
              runtime::ProjectionBackend::kSm87WeightOnly, gate,
              missing_up_input_scale),
      "gate/up SiLU eligibility rejects near-miss and missing payloads");

  const runtime::LinearWeight negative_gate_scale =
      runtime::NvFp4LinearWeight{
          &gate_packed, &gate_scales, &gate_scale_2_device,
          &gate_input_scale_device, -0.01F, 0.75F, 17'408U, 5'120U};
  const runtime::LinearWeight nan_up_input_scale =
      runtime::NvFp4LinearWeight{
          &up_packed, &up_scales, &up_scale_2_device, &up_input_scale_device,
          0.5F, std::numeric_limits<float>::quiet_NaN(), 17'408U, 5'120U};
  test.expect(!runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly,
                  negative_gate_scale, up) &&
                  !runtime::supports_nvfp4_gate_up_silu_fusion(
                      runtime::ProjectionBackend::kSm87WeightOnly, gate,
                      nan_up_input_scale),
              "gate/up SiLU eligibility rejects invalid host scales");

  std::uint16_t bf16_storage = 0U;
  const runtime::LinearWeight bf16_gate = runtime::Bf16LinearWeight{
      &bf16_storage, 17'408U, 5'120U};
  test.expect(!runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, bf16_gate,
                  up),
              "gate/up SiLU eligibility rejects a non-NVFP4 projection");

  const runtime::LinearWeight zero_scales = runtime::NvFp4LinearWeight{
      &up_packed, &up_scales, &up_scale_2_device, &up_input_scale_device,
      0.0F, 0.0F, 17'408U, 5'120U};
  test.expect(runtime::supports_nvfp4_gate_up_silu_fusion(
                  runtime::ProjectionBackend::kSm87WeightOnly, gate,
                  zero_scales),
              "gate/up SiLU eligibility accepts finite zero scales");
}

void test_source_and_tensor_failures(TestContext& test) {
  runtime::WeightBindingSource invalid;
  test.expect(runtime::bind_qwen36_27b_weights(invalid).diagnostic.code ==
                  runtime::WeightBindErrorCode::kInvalidSource,
              "null source is rejected");
  test.expect(runtime::bind_qwen36_27b_weights(runtime::ResidentWeights{})
                      .diagnostic.code ==
                  runtime::WeightBindErrorCode::kInvalidPinnedArena,
              "production overload rejects a non-pinned arena");

  {
    SyntheticArena arena;
    runtime::WeightBindingSource source = arena.source();
    source.arena_data = reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(source.arena_data) + 1U);
    test.expect(runtime::bind_qwen36_27b_weights(source).diagnostic.code ==
                    runtime::WeightBindErrorCode::kMisalignedTensor,
                "misaligned arena base is rejected");
  }
  {
    SyntheticArena arena;
    arena.erase("model.language_model.norm.weight");
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kMissingTensor,
                "missing required tensor is rejected");
  }
  {
    SyntheticArena arena;
    arena.at("model.language_model.norm.weight").dtype = st::DType::kF32;
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kDTypeMismatch,
                "fixed BF16 tensor dtype mismatch is rejected");
  }
  {
    SyntheticArena arena;
    arena.at("model.language_model.norm.weight").shape = {5119U};
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kShapeMismatch,
                "tensor shape mismatch is rejected");
  }
  {
    SyntheticArena arena;
    ++arena.at("model.language_model.norm.weight").byte_size;
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kByteSizeMismatch,
                "tensor byte-size mismatch is rejected");
  }
  {
    SyntheticArena arena;
    arena.at("model.language_model.norm.weight").device_data = nullptr;
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kNullDevicePointer,
                "null tensor device pointer is rejected");
  }
  {
    SyntheticArena arena;
    auto& view = arena.at("model.language_model.norm.weight");
    ++view.arena_offset;
    view.device_data = reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(view.device_data) + 1U);
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kMisalignedTensor,
                "misaligned tensor offset and pointer are rejected");
  }
  {
    SyntheticArena arena;
    auto& view = arena.at("model.language_model.norm.weight");
    view.device_data = reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(view.device_data) +
        runtime::kResidentTensorAlignment);
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kArenaRangeMismatch,
                "pointer/offset disagreement is rejected");
  }
  {
    SyntheticArena arena;
    auto& view = arena.at("model.language_model.norm.weight");
    const runtime::WeightBindingSource source = arena.source();
    view.arena_offset = arena.arena_bytes();
    view.device_data = reinterpret_cast<const void*>(
        reinterpret_cast<std::uintptr_t>(source.arena_data) +
        static_cast<std::uintptr_t>(view.arena_offset));
    test.expect(runtime::bind_qwen36_27b_weights(source).diagnostic.code ==
                    runtime::WeightBindErrorCode::kArenaRangeMismatch,
                "tensor range beyond arena is rejected");
  }
  {
    SyntheticArena arena;
    arena.at("model.language_model.layers.0.mlp.gate_proj.weight").dtype =
        st::DType::kF32;
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kUnsupportedWeightDType,
                "unsupported projection dtype is rejected from actual dtype");
  }
  {
    SyntheticArena arena;
    arena.at("model.language_model.layers.0.mlp.up_proj.weight_scale").shape =
        {1U};
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kShapeMismatch,
                "wrong FP8 scalar companion shape is rejected");
  }
  {
    SyntheticArena arena;
    arena.at("model.language_model.layers.0.mlp.down_proj.weight_scale").shape =
        {17408U, 1087U};
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kShapeMismatch,
                "wrong NVFP4 block-scale shape is rejected");
  }
}

void test_scalar_failures(TestContext& test) {
  constexpr std::string_view kInputScale =
      "model.language_model.layers.0.mlp.up_proj.input_scale";
  {
    SyntheticArena arena;
    arena.scalar_for(std::string(kInputScale)) =
        std::numeric_limits<float>::quiet_NaN();
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kInvalidScalar,
                "NaN input scale is rejected");
  }
  {
    SyntheticArena arena;
    arena.scalar_for(std::string(kInputScale)) = -0.25F;
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kInvalidScalar,
                "negative input scale is rejected");
  }
  {
    SyntheticArena arena;
    arena.scalar_for(
        "model.language_model.layers.0.mlp.down_proj.weight_scale_2") =
        std::numeric_limits<float>::infinity();
    test.expect(runtime::bind_qwen36_27b_weights(arena.source())
                        .diagnostic.code ==
                    runtime::WeightBindErrorCode::kInvalidScalar,
                "infinite NVFP4 scale_2 is rejected");
  }
  {
    SyntheticArena arena;
    arena.set_scalar_status(1234);
    const runtime::WeightBindResult result =
        runtime::bind_qwen36_27b_weights(arena.source());
    test.expect(result.diagnostic.code ==
                        runtime::WeightBindErrorCode::kCudaFailure &&
                    result.diagnostic.cuda_error == 1234,
                "scalar read callback failure is diagnosed structurally");
  }
}

}  // namespace

int main() {
  TestContext test;
  test_successful_bind(test);
  test_fp8_m1_output_projection_sidecar_attachment(test);
  test_fp8_prefill_qkv_register_feed_sidecar_attachment(test);
  test_fp8_prefill_supermatrix_sidecar_attachment(test);
  test_p40_packed_projection_sidecar_attachment(test);
  test_nvfp4_down_scale6_sidecar_attachment(test);
  test_projection_pair_eligibility(test);
  test_fp8_qkv_z_projection_pair_eligibility(test);
  test_linear_attention_qkv_z_ab_projection_fusion_eligibility(test);
  test_nvfp4_gate_up_silu_fusion_eligibility(test);
  test_source_and_tensor_failures(test);
  test_scalar_failures(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " model-weight binding test(s) failed\n";
    return 1;
  }
  std::cout << "model-weight binding host tests passed\n";
  return 0;
}
