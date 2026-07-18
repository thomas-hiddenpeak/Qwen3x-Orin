#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string_view>

namespace q3x::model {

// Qwen3.6 checkpoints intentionally retain the qwen3_5 Hugging Face model
// types, so the release series must not be inferred from model_type alone.
enum class ModelSeries : std::uint8_t {
    kQwen35,
    kQwen36,
};

enum class ModelTopology : std::uint8_t {
    kDense,
    kMixtureOfExperts,
};

enum class LayerType : std::uint8_t {
    kInvalid,
    kLinearAttention,
    kFullAttention,
};

enum class KnownModel : std::uint8_t {
    kQwen35_27B,
    kQwen35_35BA3B,
    kQwen36_27B,
    kQwen36_35BA3B,
    kCount,
};

inline constexpr std::size_t kKnownModelCount =
    static_cast<std::size_t>(KnownModel::kCount);

namespace detail {

[[nodiscard]] constexpr std::uint64_t saturating_multiply(
    const std::uint64_t left,
    const std::uint64_t right) noexcept {
    constexpr auto kMaximum = std::numeric_limits<std::uint64_t>::max();
    return left != 0U && right > kMaximum / left ? kMaximum : left * right;
}

[[nodiscard]] constexpr std::uint64_t saturating_add(
    const std::uint64_t left,
    const std::uint64_t right) noexcept {
    constexpr auto kMaximum = std::numeric_limits<std::uint64_t>::max();
    return right > kMaximum - left ? kMaximum : left + right;
}

}  // namespace detail

struct VisionConfig {
    bool present = false;
    std::uint32_t depth = 0;
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;
    std::uint32_t num_heads = 0;
    std::uint32_t out_hidden_size = 0;
    std::uint32_t num_position_embeddings = 0;
    std::uint32_t patch_size = 0;
    std::uint32_t spatial_merge_size = 0;
    std::uint32_t temporal_patch_size = 0;
};

// Architecture-only description of a supported model. Checkpoint precision
// and per-tensor quantization are deliberately separate concerns: a single
// architecture may be packaged as BF16, FP8, or ModelOpt mixed NVFP4/FP8.
struct ModelConfig {
    KnownModel known_model = KnownModel::kQwen35_27B;
    ModelSeries series = ModelSeries::kQwen35;
    ModelTopology topology = ModelTopology::kDense;

    std::string_view name;
    std::string_view hf_repo;
    std::string_view hf_architecture;
    std::string_view hf_model_type;
    std::string_view hf_text_model_type;

    std::uint32_t vocab_size = 0;
    std::uint32_t hidden_size = 0;
    std::uint32_t intermediate_size = 0;  // Dense MLP only; zero for MoE.
    std::uint32_t num_hidden_layers = 0;
    std::uint32_t max_position_embeddings = 0;
    float rms_norm_eps = 0.0F;

    // Full-attention (GQA) dimensions.
    std::uint32_t num_attention_heads = 0;
    std::uint32_t num_key_value_heads = 0;
    std::uint32_t head_dim = 0;
    std::uint32_t full_attention_interval = 0;
    std::uint32_t partial_rotary_numerator = 0;
    std::uint32_t partial_rotary_denominator = 0;
    double rope_theta = 0.0;
    bool attention_output_gate = false;

    // Gated DeltaNet / linear-attention dimensions.
    std::uint32_t linear_num_key_heads = 0;
    std::uint32_t linear_key_head_dim = 0;
    std::uint32_t linear_num_value_heads = 0;
    std::uint32_t linear_value_head_dim = 0;
    std::uint32_t linear_conv_kernel_dim = 0;

    // MoE dimensions; all remain zero for dense models.
    std::uint32_t num_experts = 0;
    std::uint32_t num_experts_per_token = 0;
    std::uint32_t moe_intermediate_size = 0;
    std::uint32_t shared_expert_intermediate_size = 0;

    std::uint32_t mtp_num_hidden_layers = 0;
    bool mtp_uses_dedicated_embeddings = false;
    VisionConfig vision;

    [[nodiscard]] constexpr bool is_moe() const noexcept {
        return topology == ModelTopology::kMixtureOfExperts;
    }

    [[nodiscard]] constexpr LayerType layer_type(std::size_t layer_index) const noexcept {
        if (layer_index >= num_hidden_layers || full_attention_interval == 0) {
            return LayerType::kInvalid;
        }
        return ((layer_index + 1) % full_attention_interval) == 0
                   ? LayerType::kFullAttention
                   : LayerType::kLinearAttention;
    }

    [[nodiscard]] constexpr std::uint32_t num_full_attention_layers() const noexcept {
        return full_attention_interval == 0 ? 0 : num_hidden_layers / full_attention_interval;
    }

    [[nodiscard]] constexpr std::uint32_t num_linear_attention_layers() const noexcept {
        return num_hidden_layers - num_full_attention_layers();
    }

    [[nodiscard]] constexpr std::uint64_t q_dim() const noexcept {
        return detail::saturating_multiply(num_attention_heads, head_dim);
    }

    [[nodiscard]] constexpr std::uint64_t q_projection_dim() const noexcept {
        return detail::saturating_multiply(
            q_dim(), attention_output_gate ? 2U : 1U);
    }

    [[nodiscard]] constexpr std::uint64_t kv_dim() const noexcept {
        return detail::saturating_multiply(num_key_value_heads, head_dim);
    }

    [[nodiscard]] constexpr std::uint64_t rotary_dim() const noexcept {
        return partial_rotary_denominator == 0
                   ? 0
                   : detail::saturating_multiply(
                         head_dim, partial_rotary_numerator) /
                         partial_rotary_denominator;
    }

    [[nodiscard]] constexpr std::uint64_t linear_qk_dim() const noexcept {
        return detail::saturating_multiply(
            linear_num_key_heads, linear_key_head_dim);
    }

    [[nodiscard]] constexpr std::uint64_t linear_value_dim() const noexcept {
        return detail::saturating_multiply(
            linear_num_value_heads, linear_value_head_dim);
    }

    [[nodiscard]] constexpr std::uint64_t linear_value_dim_per_key_head() const noexcept {
        return linear_num_key_heads == 0
                   ? 0
                   : detail::saturating_multiply(
                         linear_num_value_heads / linear_num_key_heads,
                         linear_value_head_dim);
    }

    [[nodiscard]] constexpr std::uint64_t linear_qkv_projection_dim() const noexcept {
        return detail::saturating_add(
            detail::saturating_multiply(2U, linear_qk_dim()),
            linear_value_dim());
    }

    [[nodiscard]] constexpr std::uint64_t linear_state_elements_per_layer() const noexcept {
        return detail::saturating_multiply(
            linear_qk_dim(), linear_value_dim_per_key_head());
    }

    [[nodiscard]] constexpr std::uint64_t full_attention_kv_elements_per_token() const noexcept {
        return detail::saturating_multiply(
            detail::saturating_multiply(num_full_attention_layers(), 2U),
            kv_dim());
    }

    [[nodiscard]] constexpr bool derived_dimensions_overflow() const noexcept {
        constexpr auto kMaximum = std::numeric_limits<std::uint64_t>::max();
        const std::uint64_t query = q_dim();
        if (attention_output_gate && query > kMaximum / 2U) {
            return true;
        }

        const std::uint64_t linear_qk = linear_qk_dim();
        const std::uint64_t linear_value = linear_value_dim();
        if (linear_qk > (kMaximum - linear_value) / 2U) {
            return true;
        }

        const std::uint64_t value_per_key_head =
            linear_value_dim_per_key_head();
        if (value_per_key_head != 0U &&
            linear_qk > kMaximum / value_per_key_head) {
            return true;
        }

        const std::uint64_t full_attention_layers =
            num_full_attention_layers();
        return full_attention_layers != 0U &&
               kv_dim() > (kMaximum / 2U) / full_attention_layers;
    }
};

enum class ConfigValidationError : std::uint8_t {
    kNone,
    kMissingIdentity,
    kKnownModelIdentityMismatch,
    kInvalidCoreDimensions,
    kInvalidAttentionDimensions,
    kInvalidRotaryDimensions,
    kInvalidLayerSchedule,
    kInvalidLinearAttentionDimensions,
    kDerivedDimensionOverflow,
    kInvalidDenseMlpDimensions,
    kInvalidMoeDimensions,
    kInvalidMtpDimensions,
    kInvalidVisionDimensions,
};

struct ConfigValidationResult {
    ConfigValidationError error = ConfigValidationError::kNone;

    [[nodiscard]] constexpr bool ok() const noexcept {
        return error == ConfigValidationError::kNone;
    }

    [[nodiscard]] constexpr explicit operator bool() const noexcept {
        return ok();
    }

    [[nodiscard]] std::string_view message() const noexcept;
};

[[nodiscard]] ConfigValidationResult validate(const ModelConfig& config) noexcept;

[[nodiscard]] const std::array<ModelConfig, kKnownModelCount>& known_model_catalog() noexcept;

[[nodiscard]] const ModelConfig* find_known_model(KnownModel model) noexcept;

// Matches either the short model name (for example, "Qwen3.6-27B") or the
// official upstream repository name ("Qwen/Qwen3.6-27B"). Matching is exact.
[[nodiscard]] const ModelConfig* find_known_model(std::string_view name_or_repo) noexcept;

[[nodiscard]] std::string_view to_string(ModelSeries series) noexcept;
[[nodiscard]] std::string_view to_string(ModelTopology topology) noexcept;
[[nodiscard]] std::string_view to_string(LayerType layer_type) noexcept;
[[nodiscard]] std::string_view to_string(KnownModel model) noexcept;

}  // namespace q3x::model
