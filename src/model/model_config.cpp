#include "q3x/model/model_config.h"

#include <cmath>

namespace q3x::model {
namespace {

VisionConfig make_vision_config(std::uint32_t out_hidden_size) {
    VisionConfig config;
    config.present = true;
    config.depth = 27;
    config.hidden_size = 1152;
    config.intermediate_size = 4304;
    config.num_heads = 16;
    config.out_hidden_size = out_hidden_size;
    config.num_position_embeddings = 2304;
    config.patch_size = 16;
    config.spatial_merge_size = 2;
    config.temporal_patch_size = 2;
    return config;
}

ModelConfig make_dense_27b(KnownModel known_model,
                           ModelSeries series,
                           std::string_view name,
                           std::string_view hf_repo) {
    ModelConfig config;
    config.known_model = known_model;
    config.series = series;
    config.topology = ModelTopology::kDense;
    config.name = name;
    config.hf_repo = hf_repo;
    config.hf_architecture = "Qwen3_5ForConditionalGeneration";
    config.hf_model_type = "qwen3_5";
    config.hf_text_model_type = "qwen3_5_text";

    config.vocab_size = 248320;
    config.hidden_size = 5120;
    config.intermediate_size = 17408;
    config.num_hidden_layers = 64;
    config.max_position_embeddings = 262144;
    config.rms_norm_eps = 1.0e-6F;

    config.num_attention_heads = 24;
    config.num_key_value_heads = 4;
    config.head_dim = 256;
    config.full_attention_interval = 4;
    config.partial_rotary_numerator = 1;
    config.partial_rotary_denominator = 4;
    config.rope_theta = 10000000.0;
    config.attention_output_gate = true;

    config.linear_num_key_heads = 16;
    config.linear_key_head_dim = 128;
    config.linear_num_value_heads = 48;
    config.linear_value_head_dim = 128;
    config.linear_conv_kernel_dim = 4;

    config.mtp_num_hidden_layers = 1;
    config.mtp_uses_dedicated_embeddings = false;
    config.vision = make_vision_config(config.hidden_size);
    return config;
}

ModelConfig make_moe_35b_a3b(KnownModel known_model,
                             ModelSeries series,
                             std::string_view name,
                             std::string_view hf_repo) {
    ModelConfig config;
    config.known_model = known_model;
    config.series = series;
    config.topology = ModelTopology::kMixtureOfExperts;
    config.name = name;
    config.hf_repo = hf_repo;
    config.hf_architecture = "Qwen3_5MoeForConditionalGeneration";
    config.hf_model_type = "qwen3_5_moe";
    config.hf_text_model_type = "qwen3_5_moe_text";

    config.vocab_size = 248320;
    config.hidden_size = 2048;
    config.intermediate_size = 0;
    config.num_hidden_layers = 40;
    config.max_position_embeddings = 262144;
    config.rms_norm_eps = 1.0e-6F;

    config.num_attention_heads = 16;
    config.num_key_value_heads = 2;
    config.head_dim = 256;
    config.full_attention_interval = 4;
    config.partial_rotary_numerator = 1;
    config.partial_rotary_denominator = 4;
    config.rope_theta = 10000000.0;
    config.attention_output_gate = true;

    config.linear_num_key_heads = 16;
    config.linear_key_head_dim = 128;
    config.linear_num_value_heads = 32;
    config.linear_value_head_dim = 128;
    config.linear_conv_kernel_dim = 4;

    config.num_experts = 256;
    config.num_experts_per_token = 8;
    config.moe_intermediate_size = 512;
    config.shared_expert_intermediate_size = 512;

    config.mtp_num_hidden_layers = 1;
    config.mtp_uses_dedicated_embeddings = false;
    config.vision = make_vision_config(config.hidden_size);
    return config;
}

const std::array<ModelConfig, kKnownModelCount> kCatalog = {
    make_dense_27b(KnownModel::kQwen35_27B,
                   ModelSeries::kQwen35,
                   "Qwen3.5-27B",
                   "Qwen/Qwen3.5-27B"),
    make_moe_35b_a3b(KnownModel::kQwen35_35BA3B,
                     ModelSeries::kQwen35,
                     "Qwen3.5-35B-A3B",
                     "Qwen/Qwen3.5-35B-A3B"),
    make_dense_27b(KnownModel::kQwen36_27B,
                   ModelSeries::kQwen36,
                   "Qwen3.6-27B",
                   "Qwen/Qwen3.6-27B"),
    make_moe_35b_a3b(KnownModel::kQwen36_35BA3B,
                     ModelSeries::kQwen36,
                     "Qwen3.6-35B-A3B",
                     "Qwen/Qwen3.6-35B-A3B"),
};

bool valid_series(ModelSeries series) noexcept {
    return series == ModelSeries::kQwen35 || series == ModelSeries::kQwen36;
}

bool valid_topology(ModelTopology topology) noexcept {
    return topology == ModelTopology::kDense ||
           topology == ModelTopology::kMixtureOfExperts;
}

bool matches_known_model_identity(const ModelConfig& config,
                                  const std::size_t known_index) noexcept {
    const ModelConfig& expected = kCatalog[known_index];
    return config.known_model == expected.known_model &&
           config.series == expected.series &&
           config.topology == expected.topology &&
           config.name == expected.name && config.hf_repo == expected.hf_repo &&
           config.hf_architecture == expected.hf_architecture &&
           config.hf_model_type == expected.hf_model_type &&
           config.hf_text_model_type == expected.hf_text_model_type;
}

}  // namespace

std::string_view ConfigValidationResult::message() const noexcept {
    switch (error) {
        case ConfigValidationError::kNone:
            return "valid";
        case ConfigValidationError::kMissingIdentity:
            return "model identity or Hugging Face metadata is missing";
        case ConfigValidationError::kKnownModelIdentityMismatch:
            return "model identity does not match the known-model catalog";
        case ConfigValidationError::kInvalidCoreDimensions:
            return "core model dimensions are invalid";
        case ConfigValidationError::kInvalidAttentionDimensions:
            return "full-attention dimensions are invalid";
        case ConfigValidationError::kInvalidRotaryDimensions:
            return "rotary embedding dimensions are invalid";
        case ConfigValidationError::kInvalidLayerSchedule:
            return "hybrid attention layer schedule is invalid";
        case ConfigValidationError::kInvalidLinearAttentionDimensions:
            return "linear-attention dimensions are invalid";
        case ConfigValidationError::kDerivedDimensionOverflow:
            return "derived model dimensions overflow uint64_t";
        case ConfigValidationError::kInvalidDenseMlpDimensions:
            return "dense MLP dimensions are invalid";
        case ConfigValidationError::kInvalidMoeDimensions:
            return "mixture-of-experts dimensions are invalid";
        case ConfigValidationError::kInvalidMtpDimensions:
            return "MTP dimensions are invalid";
        case ConfigValidationError::kInvalidVisionDimensions:
            return "vision tower dimensions are invalid";
    }
    return "unknown validation error";
}

ConfigValidationResult validate(const ModelConfig& config) noexcept {
    const auto known_index = static_cast<std::size_t>(config.known_model);
    if (known_index >= kKnownModelCount || !valid_series(config.series) ||
        !valid_topology(config.topology) || config.name.empty() ||
        config.hf_repo.empty() || config.hf_architecture.empty() ||
        config.hf_model_type.empty() || config.hf_text_model_type.empty()) {
        return {ConfigValidationError::kMissingIdentity};
    }

    if (!matches_known_model_identity(config, known_index)) {
        return {ConfigValidationError::kKnownModelIdentityMismatch};
    }

    if (config.vocab_size == 0 || config.hidden_size == 0 ||
        config.num_hidden_layers == 0 || config.max_position_embeddings == 0 ||
        !(config.rms_norm_eps > 0.0F) || !std::isfinite(config.rms_norm_eps)) {
        return {ConfigValidationError::kInvalidCoreDimensions};
    }

    if (config.num_attention_heads == 0 || config.num_key_value_heads == 0 ||
        config.head_dim == 0 ||
        (config.num_attention_heads % config.num_key_value_heads) != 0) {
        return {ConfigValidationError::kInvalidAttentionDimensions};
    }

    const auto rotary_numerator =
        static_cast<std::uint64_t>(config.head_dim) * config.partial_rotary_numerator;
    if (config.partial_rotary_numerator == 0 ||
        config.partial_rotary_denominator == 0 ||
        config.partial_rotary_numerator > config.partial_rotary_denominator ||
        (rotary_numerator % config.partial_rotary_denominator) != 0 ||
        config.rotary_dim() == 0 || config.rotary_dim() > config.head_dim ||
        (config.rotary_dim() % 2U) != 0 || !(config.rope_theta > 0.0) ||
        !std::isfinite(config.rope_theta)) {
        return {ConfigValidationError::kInvalidRotaryDimensions};
    }

    if (config.full_attention_interval == 0 ||
        config.full_attention_interval > config.num_hidden_layers ||
        config.num_full_attention_layers() == 0) {
        return {ConfigValidationError::kInvalidLayerSchedule};
    }

    if (config.linear_num_key_heads == 0 || config.linear_key_head_dim == 0 ||
        config.linear_num_value_heads == 0 || config.linear_value_head_dim == 0 ||
        config.linear_conv_kernel_dim == 0 ||
        (config.linear_num_value_heads % config.linear_num_key_heads) != 0) {
        return {ConfigValidationError::kInvalidLinearAttentionDimensions};
    }

    if (config.derived_dimensions_overflow()) {
        return {ConfigValidationError::kDerivedDimensionOverflow};
    }

    if (config.topology == ModelTopology::kDense) {
        if (config.intermediate_size == 0 || config.num_experts != 0 ||
            config.num_experts_per_token != 0 || config.moe_intermediate_size != 0 ||
            config.shared_expert_intermediate_size != 0) {
            return {ConfigValidationError::kInvalidDenseMlpDimensions};
        }
    } else {
        if (config.intermediate_size != 0 || config.num_experts == 0 ||
            config.num_experts_per_token == 0 ||
            config.num_experts_per_token > config.num_experts ||
            config.moe_intermediate_size == 0 ||
            config.shared_expert_intermediate_size == 0) {
            return {ConfigValidationError::kInvalidMoeDimensions};
        }
    }

    if (config.mtp_uses_dedicated_embeddings && config.mtp_num_hidden_layers == 0) {
        return {ConfigValidationError::kInvalidMtpDimensions};
    }

    const auto& vision = config.vision;
    if (vision.present) {
        if (vision.depth == 0 || vision.hidden_size == 0 ||
            vision.intermediate_size == 0 || vision.num_heads == 0 ||
            (vision.hidden_size % vision.num_heads) != 0 ||
            vision.out_hidden_size != config.hidden_size ||
            vision.num_position_embeddings == 0 || vision.patch_size == 0 ||
            vision.spatial_merge_size == 0 || vision.temporal_patch_size == 0) {
            return {ConfigValidationError::kInvalidVisionDimensions};
        }
    } else if (vision.depth != 0 || vision.hidden_size != 0 ||
               vision.intermediate_size != 0 || vision.num_heads != 0 ||
               vision.out_hidden_size != 0 || vision.num_position_embeddings != 0 ||
               vision.patch_size != 0 || vision.spatial_merge_size != 0 ||
               vision.temporal_patch_size != 0) {
        return {ConfigValidationError::kInvalidVisionDimensions};
    }

    return {};
}

const std::array<ModelConfig, kKnownModelCount>& known_model_catalog() noexcept {
    return kCatalog;
}

const ModelConfig* find_known_model(KnownModel model) noexcept {
    for (const auto& config : kCatalog) {
        if (config.known_model == model) {
            return &config;
        }
    }
    return nullptr;
}

const ModelConfig* find_known_model(std::string_view name_or_repo) noexcept {
    for (const auto& config : kCatalog) {
        if (config.name == name_or_repo || config.hf_repo == name_or_repo) {
            return &config;
        }
    }
    return nullptr;
}

std::string_view to_string(ModelSeries series) noexcept {
    switch (series) {
        case ModelSeries::kQwen35:
            return "Qwen3.5";
        case ModelSeries::kQwen36:
            return "Qwen3.6";
    }
    return "unknown";
}

std::string_view to_string(ModelTopology topology) noexcept {
    switch (topology) {
        case ModelTopology::kDense:
            return "dense";
        case ModelTopology::kMixtureOfExperts:
            return "mixture-of-experts";
    }
    return "unknown";
}

std::string_view to_string(LayerType layer_type) noexcept {
    switch (layer_type) {
        case LayerType::kInvalid:
            return "invalid";
        case LayerType::kLinearAttention:
            return "linear-attention";
        case LayerType::kFullAttention:
            return "full-attention";
    }
    return "unknown";
}

std::string_view to_string(KnownModel model) noexcept {
    switch (model) {
        case KnownModel::kQwen35_27B:
            return "Qwen3.5-27B";
        case KnownModel::kQwen35_35BA3B:
            return "Qwen3.5-35B-A3B";
        case KnownModel::kQwen36_27B:
            return "Qwen3.6-27B";
        case KnownModel::kQwen36_35BA3B:
            return "Qwen3.6-35B-A3B";
        case KnownModel::kCount:
            break;
    }
    return "unknown";
}

}  // namespace q3x::model
