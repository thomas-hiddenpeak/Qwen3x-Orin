#include "q3x/runtime/model_weights.h"

#include "q3x/io/safetensors.h"
#include "q3x/model/model_config.h"

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
  SyntheticArena() { build(); }

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
                 config->hidden_size, next_kind());
      add_linear(prefix + "mlp.up_proj", config->intermediate_size,
                 config->hidden_size, next_kind());
      add_linear(prefix + "mlp.down_proj", config->hidden_size,
                 config->intermediate_size, next_kind());

      if (config->layer_type(layer) ==
          q3x::model::LayerType::kLinearAttention) {
        add_linear(prefix + "linear_attn.in_proj_qkv",
                   config->linear_qkv_projection_dim(), config->hidden_size,
                   next_kind());
        add_linear(prefix + "linear_attn.in_proj_z",
                   config->linear_value_dim(), config->hidden_size,
                   next_kind());
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
        add_linear(prefix + "linear_attn.out_proj", config->hidden_size,
                   config->linear_value_dim(), next_kind());
      } else {
        add_linear(prefix + "self_attn.q_proj", config->q_projection_dim(),
                   config->hidden_size, next_kind());
        add_linear(prefix + "self_attn.k_proj", config->kv_dim(),
                   config->hidden_size, next_kind());
        add_linear(prefix + "self_attn.v_proj", config->kv_dim(),
                   config->hidden_size, next_kind());
        add_linear(prefix + "self_attn.o_proj", config->hidden_size,
                   config->q_dim(), next_kind());
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
};

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

void test_bf16_projection_pair_eligibility(TestContext& test) {
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
  test_bf16_projection_pair_eligibility(test);
  test_source_and_tensor_failures(test);
  test_scalar_failures(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " model-weight binding test(s) failed\n";
    return 1;
  }
  std::cout << "model-weight binding host tests passed\n";
  return 0;
}
