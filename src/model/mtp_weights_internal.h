#pragma once

#include "q3x/model/weight_manifest.h"

#include <array>
#include <cstdint>
#include <optional>
#include <string_view>

namespace q3x::model::mtp_detail {

struct TensorSpec {
  std::string_view name;
  std::uint64_t rows;
  std::uint64_t columns;  // Zero denotes a rank-one normalization weight.
};

inline constexpr std::array<TensorSpec, 15> kTensorSpecs{{
    {"mtp.fc.weight", 5120, 10240},
    {"mtp.layers.0.input_layernorm.weight", 5120, 0},
    {"mtp.layers.0.mlp.down_proj.weight", 5120, 17408},
    {"mtp.layers.0.mlp.gate_proj.weight", 17408, 5120},
    {"mtp.layers.0.mlp.up_proj.weight", 17408, 5120},
    {"mtp.layers.0.post_attention_layernorm.weight", 5120, 0},
    {"mtp.layers.0.self_attn.k_norm.weight", 256, 0},
    {"mtp.layers.0.self_attn.k_proj.weight", 1024, 5120},
    {"mtp.layers.0.self_attn.o_proj.weight", 5120, 6144},
    {"mtp.layers.0.self_attn.q_norm.weight", 256, 0},
    {"mtp.layers.0.self_attn.q_proj.weight", 12288, 5120},
    {"mtp.layers.0.self_attn.v_proj.weight", 1024, 5120},
    {"mtp.norm.weight", 5120, 0},
    {"mtp.pre_fc_norm_embedding.weight", 5120, 0},
    {"mtp.pre_fc_norm_hidden.weight", 5120, 0},
}};

struct TensorPlan {
  const weights::TensorLocator* locator = nullptr;
  std::uint64_t arena_offset = 0;
};

struct WeightPlan {
  std::array<TensorPlan, kTensorSpecs.size()> tensors{};
  std::uint64_t arena_bytes = 0;
};

enum class PlanError : std::uint8_t {
  kNone, kTensorSet, kCategory, kDtype, kShape, kRange, kBytes,
};

struct PlanResult {
  std::optional<WeightPlan> value;
  PlanError error = PlanError::kNone;
  std::string_view tensor;
};

// Header/ABI validation ONLY, not payload authentication or a resident loader.
// The source manifest must outlive the returned non-owning plan. Startup must
// still authenticate shard bytes through the resident loader before binding.
[[nodiscard]] PlanResult plan_weights(const weights::WeightManifest&) noexcept;

}  // namespace q3x::model::mtp_detail
