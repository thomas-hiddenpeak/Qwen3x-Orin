#include "q3x/model/model_config.h"

#include <cstdint>
#include <iostream>
#include <limits>
#include <string_view>

namespace {

class TestContext {
public:
    void expect(bool condition, std::string_view message) {
        if (!condition) {
            ++failures_;
            std::cerr << "FAILED: " << message << '\n';
        }
    }

    [[nodiscard]] int failures() const noexcept { return failures_; }

private:
    int failures_ = 0;
};

void expect_same_engine_shape(const q3x::model::ModelConfig& left,
                              const q3x::model::ModelConfig& right,
                              TestContext& test) {
    test.expect(left.topology == right.topology, "topology differs across releases");
    test.expect(left.vocab_size == right.vocab_size, "vocab size differs across releases");
    test.expect(left.hidden_size == right.hidden_size, "hidden size differs across releases");
    test.expect(left.intermediate_size == right.intermediate_size,
                "dense intermediate size differs across releases");
    test.expect(left.num_hidden_layers == right.num_hidden_layers,
                "layer count differs across releases");
    test.expect(left.max_position_embeddings == right.max_position_embeddings,
                "context length differs across releases");
    test.expect(left.num_attention_heads == right.num_attention_heads,
                "attention head count differs across releases");
    test.expect(left.num_key_value_heads == right.num_key_value_heads,
                "KV head count differs across releases");
    test.expect(left.head_dim == right.head_dim, "attention head dimension differs across releases");
    test.expect(left.full_attention_interval == right.full_attention_interval,
                "attention schedule differs across releases");
    test.expect(left.rotary_dim() == right.rotary_dim(),
                "rotary dimension differs across releases");
    test.expect(left.linear_num_key_heads == right.linear_num_key_heads,
                "linear key head count differs across releases");
    test.expect(left.linear_key_head_dim == right.linear_key_head_dim,
                "linear key head dimension differs across releases");
    test.expect(left.linear_num_value_heads == right.linear_num_value_heads,
                "linear value head count differs across releases");
    test.expect(left.linear_value_head_dim == right.linear_value_head_dim,
                "linear value head dimension differs across releases");
    test.expect(left.num_experts == right.num_experts,
                "expert count differs across releases");
    test.expect(left.num_experts_per_token == right.num_experts_per_token,
                "experts-per-token differs across releases");
    test.expect(left.moe_intermediate_size == right.moe_intermediate_size,
                "expert intermediate size differs across releases");
    test.expect(left.shared_expert_intermediate_size ==
                    right.shared_expert_intermediate_size,
                "shared expert intermediate size differs across releases");
    test.expect(left.mtp_num_hidden_layers == right.mtp_num_hidden_layers,
                "MTP layer count differs across releases");
    test.expect(left.vision.depth == right.vision.depth,
                "vision depth differs across releases");
    test.expect(left.vision.out_hidden_size == right.vision.out_hidden_size,
                "vision output size differs across releases");
}

void test_catalog_and_lookup(TestContext& test) {
    using namespace q3x::model;

    const auto& catalog = known_model_catalog();
    test.expect(catalog.size() == 4, "catalog must contain exactly four target models");

    for (std::size_t index = 0; index < catalog.size(); ++index) {
        const auto& config = catalog[index];
        const auto expected_model = static_cast<KnownModel>(index);
        const auto validation = validate(config);
        test.expect(validation.ok(), validation.message());
        test.expect(config.known_model == expected_model, "catalog order and KnownModel differ");
        test.expect(find_known_model(expected_model) == &config, "KnownModel lookup failed");
        test.expect(find_known_model(config.name) == &config, "short-name lookup failed");
        test.expect(find_known_model(config.hf_repo) == &config, "HF repository lookup failed");
        test.expect(to_string(expected_model) == config.name, "KnownModel string differs from name");
    }

    test.expect(find_known_model(KnownModel::kCount) == nullptr,
                "sentinel KnownModel must not resolve");
    test.expect(find_known_model("Qwen3.6-unknown") == nullptr,
                "unknown name must not resolve");
}

void test_dense_27b_dimensions(TestContext& test) {
    using namespace q3x::model;
    const auto* config = find_known_model(KnownModel::kQwen36_27B);
    test.expect(config != nullptr, "Qwen3.6-27B is missing");
    if (config == nullptr) {
        return;
    }

    test.expect(config->topology == ModelTopology::kDense, "27B must be dense");
    test.expect(config->hidden_size == 5120, "27B hidden size");
    test.expect(config->intermediate_size == 17408, "27B intermediate size");
    test.expect(config->num_hidden_layers == 64, "27B layer count");
    test.expect(config->q_dim() == 6144, "27B Q dimension");
    test.expect(config->q_projection_dim() == 12288, "27B gated Q projection dimension");
    test.expect(config->kv_dim() == 1024, "27B KV dimension");
    test.expect(config->rotary_dim() == 64, "27B rotary dimension");
    test.expect(config->linear_qk_dim() == 2048, "27B linear Q/K dimension");
    test.expect(config->linear_value_dim() == 6144, "27B linear value dimension");
    test.expect(config->linear_value_dim_per_key_head() == 384,
                "27B value dimension per linear key head");
    test.expect(config->linear_qkv_projection_dim() == 10240,
                "27B linear QKV projection dimension");
    test.expect(config->linear_state_elements_per_layer() == 786432,
                "27B recurrent state elements per linear layer");
    test.expect(config->num_full_attention_layers() == 16,
                "27B full-attention layer count");
    test.expect(config->num_linear_attention_layers() == 48,
                "27B linear-attention layer count");
    test.expect(config->full_attention_kv_elements_per_token() == 32768,
                "27B full-attention KV elements per token");
    test.expect(config->layer_type(0) == LayerType::kLinearAttention, "layer 0 type");
    test.expect(config->layer_type(2) == LayerType::kLinearAttention, "layer 2 type");
    test.expect(config->layer_type(3) == LayerType::kFullAttention, "layer 3 type");
    test.expect(config->layer_type(63) == LayerType::kFullAttention, "layer 63 type");
    test.expect(config->layer_type(64) == LayerType::kInvalid, "out-of-range layer type");
}

void test_moe_35b_dimensions(TestContext& test) {
    using namespace q3x::model;
    const auto* config = find_known_model(KnownModel::kQwen36_35BA3B);
    test.expect(config != nullptr, "Qwen3.6-35B-A3B is missing");
    if (config == nullptr) {
        return;
    }

    test.expect(config->is_moe(), "35B-A3B must be MoE");
    test.expect(config->hidden_size == 2048, "35B-A3B hidden size");
    test.expect(config->intermediate_size == 0, "MoE must not expose a dense MLP size");
    test.expect(config->num_hidden_layers == 40, "35B-A3B layer count");
    test.expect(config->num_experts == 256, "35B-A3B expert count");
    test.expect(config->num_experts_per_token == 8, "35B-A3B top-k");
    test.expect(config->moe_intermediate_size == 512, "35B-A3B expert size");
    test.expect(config->shared_expert_intermediate_size == 512,
                "35B-A3B shared expert size");
    test.expect(config->q_dim() == 4096, "35B-A3B Q dimension");
    test.expect(config->q_projection_dim() == 8192,
                "35B-A3B gated Q projection dimension");
    test.expect(config->kv_dim() == 512, "35B-A3B KV dimension");
    test.expect(config->linear_qk_dim() == 2048, "35B-A3B linear Q/K dimension");
    test.expect(config->linear_value_dim() == 4096, "35B-A3B linear value dimension");
    test.expect(config->linear_value_dim_per_key_head() == 256,
                "35B-A3B value dimension per linear key head");
    test.expect(config->linear_qkv_projection_dim() == 8192,
                "35B-A3B linear QKV projection dimension");
    test.expect(config->linear_state_elements_per_layer() == 524288,
                "35B-A3B recurrent state elements per linear layer");
    test.expect(config->num_full_attention_layers() == 10,
                "35B-A3B full-attention layer count");
    test.expect(config->num_linear_attention_layers() == 30,
                "35B-A3B linear-attention layer count");
    test.expect(config->full_attention_kv_elements_per_token() == 10240,
                "35B-A3B full-attention KV elements per token");
}

void test_release_shape_compatibility(TestContext& test) {
    using namespace q3x::model;
    const auto* q35_dense = find_known_model(KnownModel::kQwen35_27B);
    const auto* q36_dense = find_known_model(KnownModel::kQwen36_27B);
    const auto* q35_moe = find_known_model(KnownModel::kQwen35_35BA3B);
    const auto* q36_moe = find_known_model(KnownModel::kQwen36_35BA3B);

    test.expect(q35_dense != nullptr && q36_dense != nullptr && q35_moe != nullptr &&
                    q36_moe != nullptr,
                "release compatibility inputs must exist");
    if (q35_dense == nullptr || q36_dense == nullptr || q35_moe == nullptr ||
        q36_moe == nullptr) {
        return;
    }

    expect_same_engine_shape(*q35_dense, *q36_dense, test);
    expect_same_engine_shape(*q35_moe, *q36_moe, test);
    test.expect(q35_dense->series != q36_dense->series,
                "dense releases must retain distinct series identity");
    test.expect(q35_moe->series != q36_moe->series,
                "MoE releases must retain distinct series identity");
    test.expect(q36_moe->hf_architecture == "Qwen3_5MoeForConditionalGeneration",
                "Qwen3.6 MoE must retain its upstream qwen3_5 architecture tag");
}

void test_validation_failures(TestContext& test) {
    using namespace q3x::model;
    const auto* dense_source = find_known_model(KnownModel::kQwen35_27B);
    const auto* moe_source = find_known_model(KnownModel::kQwen35_35BA3B);
    test.expect(dense_source != nullptr && moe_source != nullptr,
                "validation fixtures must exist");
    if (dense_source == nullptr || moe_source == nullptr) {
        return;
    }

    auto config = *dense_source;
    config.known_model = KnownModel::kQwen36_27B;
    test.expect(
        validate(config).error ==
            ConfigValidationError::kKnownModelIdentityMismatch,
        "known-model enum and release identity mismatch must fail validation");

    config = *dense_source;
    config.hidden_size = 0;
    test.expect(validate(config).error == ConfigValidationError::kInvalidCoreDimensions,
                "zero hidden size must fail validation");

    config = *dense_source;
    config.num_attention_heads = 25;
    test.expect(validate(config).error == ConfigValidationError::kInvalidAttentionDimensions,
                "invalid GQA head ratio must fail validation");

    config = *dense_source;
    config.partial_rotary_denominator = 3;
    test.expect(validate(config).error == ConfigValidationError::kInvalidRotaryDimensions,
                "fractional rotary dimension must fail validation");

    config = *dense_source;
    config.full_attention_interval = 0;
    test.expect(validate(config).error == ConfigValidationError::kInvalidLayerSchedule,
                "zero full-attention interval must fail validation");

    config = *dense_source;
    config.linear_num_value_heads = 47;
    test.expect(validate(config).error ==
                    ConfigValidationError::kInvalidLinearAttentionDimensions,
                "invalid linear head ratio must fail validation");

    config = *dense_source;
    config.num_experts = 1;
    test.expect(validate(config).error == ConfigValidationError::kInvalidDenseMlpDimensions,
                "MoE dimensions on a dense model must fail validation");

    config = *moe_source;
    config.num_experts_per_token = 257;
    test.expect(validate(config).error == ConfigValidationError::kInvalidMoeDimensions,
                "top-k greater than expert count must fail validation");

    config = *dense_source;
    config.mtp_num_hidden_layers = 0;
    config.mtp_uses_dedicated_embeddings = true;
    test.expect(validate(config).error == ConfigValidationError::kInvalidMtpDimensions,
                "dedicated MTP embeddings without an MTP layer must fail validation");

    config = *dense_source;
    config.vision.out_hidden_size = config.hidden_size - 1;
    test.expect(validate(config).error == ConfigValidationError::kInvalidVisionDimensions,
                "vision/text projection mismatch must fail validation");
}

void test_large_derived_dimensions(TestContext& test) {
    using namespace q3x::model;
    const auto* dense_source = find_known_model(KnownModel::kQwen35_27B);
    test.expect(dense_source != nullptr, "large-dimension fixture must exist");
    if (dense_source == nullptr) {
        return;
    }

    constexpr std::uint32_t kMaximum32 =
        std::numeric_limits<std::uint32_t>::max();
    constexpr std::uint64_t kMaximum64 =
        std::numeric_limits<std::uint64_t>::max();

    auto config = *dense_source;
    config.num_attention_heads = kMaximum32;
    config.num_key_value_heads = 1;
    config.head_dim = kMaximum32 - 1U;
    config.partial_rotary_numerator = 1;
    config.partial_rotary_denominator = 1;

    const auto expected_q_dim = static_cast<std::uint64_t>(kMaximum32) *
                                static_cast<std::uint64_t>(kMaximum32 - 1U);
    test.expect(config.q_dim() == expected_q_dim,
                "Q dimension must not overflow at 32 bits");
    test.expect(config.q_projection_dim() == kMaximum64,
                "overflowing gated Q dimension must saturate");
    test.expect(config.derived_dimensions_overflow(),
                "gated Q overflow must be detectable");
    test.expect(validate(config).error ==
                    ConfigValidationError::kDerivedDimensionOverflow,
                "derived Q overflow must fail validation");

    config = *dense_source;
    config.linear_num_key_heads = kMaximum32;
    config.linear_key_head_dim = kMaximum32;
    config.linear_num_value_heads = kMaximum32;
    config.linear_value_head_dim = kMaximum32;
    const auto expected_linear_dim =
        static_cast<std::uint64_t>(kMaximum32) * kMaximum32;
    test.expect(config.linear_qk_dim() == expected_linear_dim,
                "linear Q/K dimension must not overflow at 32 bits");
    test.expect(config.linear_qkv_projection_dim() == kMaximum64,
                "overflowing linear projection dimension must saturate");
    test.expect(config.linear_state_elements_per_layer() == kMaximum64,
                "overflowing linear state size must saturate");
    test.expect(config.derived_dimensions_overflow(),
                "linear derived overflow must be detectable");
    test.expect(validate(config).error ==
                    ConfigValidationError::kDerivedDimensionOverflow,
                "linear derived overflow must fail validation");
}

}  // namespace

int main() {
    TestContext test;
    test_catalog_and_lookup(test);
    test_dense_27b_dimensions(test);
    test_moe_35b_dimensions(test);
    test_release_shape_compatibility(test);
    test_validation_failures(test);
    test_large_derived_dimensions(test);

    if (test.failures() != 0) {
        std::cerr << test.failures() << " model config test(s) failed\n";
        return 1;
    }

    std::cout << "All model config tests passed\n";
    return 0;
}
