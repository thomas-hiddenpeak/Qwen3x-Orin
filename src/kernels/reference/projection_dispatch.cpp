#include "q3x/runtime/model_weights.h"

#include "q3x/kernels/reference_gemv.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <variant>

namespace q3x::runtime {
namespace {

constexpr std::size_t kMaximumSm87SmallMTokens = 8U;
constexpr std::size_t kSm87DirectBf16Rows = 48U;
constexpr std::size_t kSm87DirectBf16Columns = 5120U;

[[nodiscard]] bool valid_scale(const float value) noexcept {
  return std::isfinite(value) && value >= 0.0F;
}

[[nodiscard]] int invalid_value() noexcept {
  return static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] bool multiply_overflows(const std::size_t left,
                                      const std::size_t right) noexcept {
  return right != 0U &&
         left > std::numeric_limits<std::size_t>::max() / right;
}

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

[[nodiscard]] bool ranges_overlap(const void* const first,
                                  const std::size_t first_bytes,
                                  const void* const second,
                                  const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return true;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  return first_begin < second_begin + second_bytes &&
         second_begin < first_begin + first_bytes;
}

[[nodiscard]] bool pointer_is_aligned(
    const void* const pointer, const std::size_t alignment) noexcept {
  return pointer != nullptr &&
         (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
}

struct ProjectionTileSpans {
  std::size_t rows = 0U;
  std::size_t columns = 0U;
  std::size_t input_bytes = 0U;
  std::size_t output_bytes = 0U;
  const void* weight = nullptr;
  std::size_t weight_bytes = 0U;
  const void* auxiliary_weight = nullptr;
  std::size_t auxiliary_weight_bytes = 0U;
  std::array<const void*, 2U> scalar_weights{};
};

[[nodiscard]] int validate_projection_tile(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, const std::size_t token_count,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const output, ProjectionTileSpans* const spans,
    const bool direct_output = false,
    const std::size_t maximum_token_count =
        kMaximumProjectionTileTokenCount) noexcept {
  if (!is_valid_projection_backend(backend) ||
      weight.valueless_by_exception() || token_count == 0U ||
      token_count > maximum_token_count || input == nullptr ||
      output == nullptr || spans == nullptr ||
      !pointer_is_aligned(input, alignof(std::uint16_t)) ||
      !pointer_is_aligned(output, alignof(std::uint16_t))) {
    return invalid_value();
  }

  const int weight_status = std::visit(
      [spans](const auto& selected) noexcept -> int {
        using Selected = std::decay_t<decltype(selected)>;
        if (selected.output_size == 0U || selected.input_size == 0U ||
            multiply_overflows(selected.output_size,
                               selected.input_size)) {
          return invalid_value();
        }
        spans->rows = selected.output_size;
        spans->columns = selected.input_size;
        const std::size_t weight_elements =
            selected.output_size * selected.input_size;

        if constexpr (std::is_same_v<Selected, Bf16LinearWeight>) {
          if (selected.weight == nullptr ||
              multiply_overflows(weight_elements,
                                 sizeof(std::uint16_t))) {
            return invalid_value();
          }
          spans->weight = selected.weight;
          spans->weight_bytes = weight_elements * sizeof(std::uint16_t);
        } else if constexpr (std::is_same_v<Selected, Fp8LinearWeight>) {
          if (selected.weight == nullptr ||
              selected.weight_scale_device == nullptr ||
              selected.input_scale_device == nullptr ||
              !valid_scale(selected.weight_scale) ||
              !valid_scale(selected.input_scale)) {
            return invalid_value();
          }
          spans->weight = selected.weight;
          spans->weight_bytes = weight_elements;
          spans->scalar_weights = {selected.weight_scale_device,
                                   selected.input_scale_device};
        } else if constexpr (std::is_same_v<Selected,
                                            NvFp4LinearWeight>) {
          if (selected.packed_weight == nullptr ||
              selected.block_scale == nullptr ||
              selected.weight_scale_2_device == nullptr ||
              selected.input_scale_device == nullptr ||
              !valid_scale(selected.weight_scale_2) ||
              !valid_scale(selected.input_scale) ||
              (selected.input_size % 16U) != 0U) {
            return invalid_value();
          }
          spans->weight = selected.packed_weight;
          spans->weight_bytes = weight_elements / 2U;
          spans->auxiliary_weight = selected.block_scale;
          spans->auxiliary_weight_bytes = weight_elements / 16U;
          spans->scalar_weights = {selected.weight_scale_2_device,
                                   selected.input_scale_device};
        } else {
          return invalid_value();
        }
        return static_cast<int>(cudaSuccess);
      },
      weight);
  if (weight_status != static_cast<int>(cudaSuccess) ||
      multiply_overflows(token_count, spans->columns) ||
      multiply_overflows(token_count, spans->rows)) {
    return invalid_value();
  }

  const std::size_t input_elements = token_count * spans->columns;
  const std::size_t output_elements = token_count * spans->rows;
  if (multiply_overflows(input_elements, sizeof(std::uint16_t)) ||
      multiply_overflows(output_elements, sizeof(std::uint16_t))) {
    return invalid_value();
  }
  spans->input_bytes = input_elements * sizeof(std::uint16_t);
  spans->output_bytes = output_elements * sizeof(std::uint16_t);

  if (byte_range_overflows(input, spans->input_bytes) ||
      byte_range_overflows(output, spans->output_bytes) ||
      byte_range_overflows(spans->weight, spans->weight_bytes) ||
      (spans->auxiliary_weight != nullptr &&
       byte_range_overflows(spans->auxiliary_weight,
                            spans->auxiliary_weight_bytes)) ||
      ranges_overlap(output, spans->output_bytes, input,
                     spans->input_bytes) ||
      ranges_overlap(output, spans->output_bytes, spans->weight,
                     spans->weight_bytes) ||
      (spans->auxiliary_weight != nullptr &&
       ranges_overlap(output, spans->output_bytes,
                      spans->auxiliary_weight,
                      spans->auxiliary_weight_bytes))) {
    return invalid_value();
  }
  for (const void* const scalar_weight : spans->scalar_weights) {
    if (scalar_weight != nullptr &&
        (byte_range_overflows(scalar_weight, sizeof(float)) ||
         ranges_overlap(output, spans->output_bytes, scalar_weight,
                        sizeof(float)))) {
      return invalid_value();
    }
  }

  const bool requires_reference_scratch =
      !direct_output &&
      (backend == ProjectionBackend::kReference || weight.index() == 0U);
  if (!requires_reference_scratch) {
    return static_cast<int>(cudaSuccess);
  }
  if (fp32_scratch == nullptr || scratch_elements < spans->rows ||
      multiply_overflows(spans->rows, sizeof(float))) {
    return invalid_value();
  }
  const std::size_t scratch_bytes = spans->rows * sizeof(float);
  if (byte_range_overflows(fp32_scratch, scratch_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, input,
                     spans->input_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, output,
                     spans->output_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, spans->weight,
                     spans->weight_bytes) ||
      (spans->auxiliary_weight != nullptr &&
       ranges_overlap(fp32_scratch, scratch_bytes,
                      spans->auxiliary_weight,
                      spans->auxiliary_weight_bytes))) {
    return invalid_value();
  }
  for (const void* const scalar_weight : spans->scalar_weights) {
    if (scalar_weight != nullptr &&
        ranges_overlap(fp32_scratch, scratch_bytes, scalar_weight,
                       sizeof(float))) {
      return invalid_value();
    }
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] bool overlaps_projection_weights(
    const void* const pointer, const std::size_t bytes,
    const ProjectionTileSpans& spans) noexcept {
  if (ranges_overlap(pointer, bytes, spans.weight, spans.weight_bytes) ||
      (spans.auxiliary_weight != nullptr &&
       ranges_overlap(pointer, bytes, spans.auxiliary_weight,
                      spans.auxiliary_weight_bytes))) {
    return true;
  }
  for (const void* const scalar_weight : spans.scalar_weights) {
    if (scalar_weight != nullptr &&
        ranges_overlap(pointer, bytes, scalar_weight, sizeof(float))) {
      return true;
    }
  }
  return false;
}

[[nodiscard]] bool requires_projection_scratch(
    const ProjectionBackend backend, const LinearWeight& weight) noexcept {
  return backend == ProjectionBackend::kReference || weight.index() == 0U;
}

[[nodiscard]] int validate_projection_pair_cross_ranges(
    const ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight, const std::uint16_t* const input,
    float* const fp32_scratch, std::uint16_t* const first_output,
    std::uint16_t* const second_output,
    const ProjectionTileSpans& first,
    const ProjectionTileSpans& second,
    const bool direct_output) noexcept {
  if (first.columns != second.columns ||
      ranges_overlap(first_output, first.output_bytes, second_output,
                     second.output_bytes) ||
      overlaps_projection_weights(first_output, first.output_bytes, second) ||
      overlaps_projection_weights(second_output, second.output_bytes, first)) {
    return invalid_value();
  }

  std::size_t scratch_rows = 0U;
  if (!direct_output &&
      requires_projection_scratch(backend, first_weight)) {
    scratch_rows = first.rows;
  }
  if (!direct_output &&
      requires_projection_scratch(backend, second_weight) &&
      second.rows > scratch_rows) {
    scratch_rows = second.rows;
  }
  if (scratch_rows == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (fp32_scratch == nullptr ||
      multiply_overflows(scratch_rows, sizeof(float))) {
    return invalid_value();
  }
  const std::size_t scratch_bytes = scratch_rows * sizeof(float);
  if (byte_range_overflows(fp32_scratch, scratch_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, input,
                     first.input_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, first_output,
                     first.output_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, second_output,
                     second.output_bytes) ||
      overlaps_projection_weights(fp32_scratch, scratch_bytes, first) ||
      overlaps_projection_weights(fp32_scratch, scratch_bytes, second)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

struct MlpGateUpSiluLaunchPlan {
  bool aligned_nvfp4_fusion = false;
  bool direct_output = false;
  ProjectionTileSpans gate;
  ProjectionTileSpans up;
};

struct LinearAttentionQkvZAbLaunchPlan {
  ProjectionTileSpans qkv;
  ProjectionTileSpans z;
  ProjectionTileSpans a;
  ProjectionTileSpans b;
};

[[nodiscard]] int validate_linear_attention_qkv_z_ab_direct_launch(
    const ProjectionBackend backend, const LinearWeight& qkv_weight,
    const LinearWeight& z_weight, const LinearWeight& a_weight,
    const LinearWeight& b_weight, const std::uint16_t* const input,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output,
    LinearAttentionQkvZAbLaunchPlan* const plan) noexcept {
  if (plan == nullptr) {
    return invalid_value();
  }
  *plan = LinearAttentionQkvZAbLaunchPlan{};

  const std::array<const LinearWeight*, 4U> weights{
      &qkv_weight, &z_weight, &a_weight, &b_weight};
  const std::array<std::uint16_t*, 4U> outputs{
      qkv_output, z_output, a_output, b_output};
  const std::array<ProjectionTileSpans*, 4U> spans{
      &plan->qkv, &plan->z, &plan->a, &plan->b};
  for (std::size_t index = 0U; index < weights.size(); ++index) {
    const int validation = validate_projection_tile(
        backend, *weights[index], input, 1U, nullptr, 0U, outputs[index],
        spans[index], true);
    if (validation != static_cast<int>(cudaSuccess)) {
      return validation;
    }
  }

  for (std::size_t first = 0U; first < weights.size(); ++first) {
    for (std::size_t second = first + 1U; second < weights.size(); ++second) {
      const int validation = validate_projection_pair_cross_ranges(
          backend, *weights[first], *weights[second], input, nullptr,
          outputs[first], outputs[second], *spans[first], *spans[second],
          true);
      if (validation != static_cast<int>(cudaSuccess)) {
        return validation;
      }
    }
  }
  return static_cast<int>(cudaSuccess);
}

struct FullAttentionQKvLaunchPlan {
  bool aligned_fp8_fusion = false;
  ProjectionTileSpans q;
  ProjectionTileSpans key;
  ProjectionTileSpans value;
};

struct DownResidualNormLaunchPlan {
  bool aligned_nvfp4_fusion = false;
  ProjectionTileSpans down;
};

[[nodiscard]] int validate_full_attention_q_kv_launch(
    const ProjectionBackend backend, const LinearWeight& q_weight,
    const LinearWeight& key_weight, const LinearWeight& value_weight,
    const std::uint16_t* const input, float* const fp32_scratch,
    const std::size_t scratch_elements, std::uint16_t* const q_output,
    std::uint16_t* const key_output, std::uint16_t* const value_output,
    FullAttentionQKvLaunchPlan* const plan) noexcept {
  if (plan == nullptr) {
    return invalid_value();
  }
  *plan = FullAttentionQKvLaunchPlan{};
  if (supports_fp8_q_kv_projection_fusion(
          backend, q_weight, key_weight, value_weight)) {
    const auto& q = std::get<Fp8LinearWeight>(q_weight);
    const auto& key = std::get<Fp8LinearWeight>(key_weight);
    const auto& value = std::get<Fp8LinearWeight>(value_weight);
    plan->aligned_fp8_fusion =
        pointer_is_aligned(q.weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(key.weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(value.weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(input, alignof(std::uint64_t)) &&
        pointer_is_aligned(q_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(key_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(value_output, alignof(std::uint16_t));
  }

  const int q_validation = validate_projection_tile(
      backend, q_weight, input, 1U, fp32_scratch, scratch_elements, q_output,
      &plan->q);
  if (q_validation != static_cast<int>(cudaSuccess)) {
    return q_validation;
  }
  const int key_validation = validate_projection_tile(
      backend, key_weight, input, 1U, fp32_scratch, scratch_elements,
      key_output, &plan->key);
  if (key_validation != static_cast<int>(cudaSuccess)) {
    return key_validation;
  }
  const int value_validation = validate_projection_tile(
      backend, value_weight, input, 1U, fp32_scratch, scratch_elements,
      value_output, &plan->value);
  if (value_validation != static_cast<int>(cudaSuccess)) {
    return value_validation;
  }
  if (plan->q.columns != plan->key.columns ||
      plan->q.columns != plan->value.columns ||
      ranges_overlap(q_output, plan->q.output_bytes, key_output,
                     plan->key.output_bytes) ||
      ranges_overlap(q_output, plan->q.output_bytes, value_output,
                     plan->value.output_bytes) ||
      ranges_overlap(key_output, plan->key.output_bytes, value_output,
                     plan->value.output_bytes) ||
      overlaps_projection_weights(q_output, plan->q.output_bytes,
                                  plan->key) ||
      overlaps_projection_weights(q_output, plan->q.output_bytes,
                                  plan->value) ||
      overlaps_projection_weights(key_output, plan->key.output_bytes,
                                  plan->q) ||
      overlaps_projection_weights(key_output, plan->key.output_bytes,
                                  plan->value) ||
      overlaps_projection_weights(value_output, plan->value.output_bytes,
                                  plan->q) ||
      overlaps_projection_weights(value_output, plan->value.output_bytes,
                                  plan->key)) {
    return invalid_value();
  }

  std::size_t scratch_rows = 0U;
  if (requires_projection_scratch(backend, q_weight)) {
    scratch_rows = plan->q.rows;
  }
  if (requires_projection_scratch(backend, key_weight) &&
      plan->key.rows > scratch_rows) {
    scratch_rows = plan->key.rows;
  }
  if (requires_projection_scratch(backend, value_weight) &&
      plan->value.rows > scratch_rows) {
    scratch_rows = plan->value.rows;
  }
  if (scratch_rows == 0U) {
    return static_cast<int>(cudaSuccess);
  }
  if (fp32_scratch == nullptr ||
      multiply_overflows(scratch_rows, sizeof(float))) {
    return invalid_value();
  }
  const std::size_t scratch_bytes = scratch_rows * sizeof(float);
  if (byte_range_overflows(fp32_scratch, scratch_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, input,
                     plan->q.input_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, q_output,
                     plan->q.output_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, key_output,
                     plan->key.output_bytes) ||
      ranges_overlap(fp32_scratch, scratch_bytes, value_output,
                     plan->value.output_bytes) ||
      overlaps_projection_weights(fp32_scratch, scratch_bytes, plan->q) ||
      overlaps_projection_weights(fp32_scratch, scratch_bytes, plan->key) ||
      overlaps_projection_weights(fp32_scratch, scratch_bytes, plan->value)) {
    return invalid_value();
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int validate_mlp_gate_up_silu_launch(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* const input,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_output,
    MlpGateUpSiluLaunchPlan* const plan) noexcept {
  if (plan == nullptr) {
    return invalid_value();
  }
  *plan = MlpGateUpSiluLaunchPlan{};
  const bool eligible_fusion = supports_nvfp4_gate_up_silu_fusion(
      backend, gate_weight, up_weight);
  if (eligible_fusion) {
    const auto& gate = std::get<NvFp4LinearWeight>(gate_weight);
    const auto& up = std::get<NvFp4LinearWeight>(up_weight);
    plan->aligned_nvfp4_fusion =
        pointer_is_aligned(gate.packed_weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(up.packed_weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(input, alignof(std::uint64_t)) &&
        pointer_is_aligned(gate_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(up_output, alignof(std::uint16_t));
  }
  // Preserve the generic pair fallback's scratch contract. In particular,
  // the exact SM87 BF16 A/B pair is already direct-to-BF16 and accepts null
  // scratch even though it is not eligible for this MLP-specific fusion.
  plan->direct_output =
      plan->aligned_nvfp4_fusion ||
      supports_bf16_projection_pair(backend, gate_weight, up_weight);

  const int gate_validation = validate_projection_tile(
      backend, gate_weight, input, 1U, fp32_scratch, scratch_elements,
      gate_output, &plan->gate, plan->direct_output);
  if (gate_validation != static_cast<int>(cudaSuccess)) {
    return gate_validation;
  }
  const int up_validation = validate_projection_tile(
      backend, up_weight, input, 1U, fp32_scratch, scratch_elements,
      up_output, &plan->up, plan->direct_output);
  if (up_validation != static_cast<int>(cudaSuccess)) {
    return up_validation;
  }
  const int cross_validation = validate_projection_pair_cross_ranges(
      backend, gate_weight, up_weight, input, fp32_scratch, gate_output,
      up_output, plan->gate, plan->up, plan->direct_output);
  if (cross_validation != static_cast<int>(cudaSuccess)) {
    return cross_validation;
  }
  return plan->gate.rows == plan->up.rows
             ? static_cast<int>(cudaSuccess)
             : invalid_value();
}

[[nodiscard]] int validate_mlp_down_residual_norm_launch(
    const ProjectionBackend backend, const LinearWeight& down_weight,
    const std::uint16_t* const activation,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const raw_down_output,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    DownResidualNormLaunchPlan* const plan) noexcept {
  constexpr std::size_t kHiddenElements = 5'120U;
  constexpr std::size_t kHiddenBytes =
      kHiddenElements * sizeof(std::uint16_t);
  if (plan == nullptr) {
    return invalid_value();
  }
  *plan = DownResidualNormLaunchPlan{};

  if (supports_nvfp4_down_residual_norm_fusion(backend, down_weight)) {
    const auto& down = std::get<NvFp4LinearWeight>(down_weight);
    plan->aligned_nvfp4_fusion =
        pointer_is_aligned(down.packed_weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(activation, alignof(std::uint64_t)) &&
        pointer_is_aligned(residual_left, alignof(std::uint16_t)) &&
        pointer_is_aligned(norm_weight, alignof(std::uint16_t)) &&
        pointer_is_aligned(raw_down_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(residual_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(normalized_output, alignof(std::uint16_t));
  }

  const int projection_validation = validate_projection_tile(
      backend, down_weight, activation, 1U, fp32_scratch, scratch_elements,
      raw_down_output, &plan->down, plan->aligned_nvfp4_fusion);
  if (projection_validation != static_cast<int>(cudaSuccess)) {
    return projection_validation;
  }
  if (plan->down.rows != kHiddenElements || !std::isfinite(epsilon) ||
      epsilon <= 0.0F || residual_left == nullptr || norm_weight == nullptr ||
      residual_output == nullptr || normalized_output == nullptr ||
      byte_range_overflows(residual_left, kHiddenBytes) ||
      byte_range_overflows(norm_weight, kHiddenBytes) ||
      byte_range_overflows(residual_output, kHiddenBytes) ||
      byte_range_overflows(normalized_output, kHiddenBytes) ||
      ranges_overlap(raw_down_output, plan->down.output_bytes, residual_left,
                     kHiddenBytes) ||
      ranges_overlap(raw_down_output, plan->down.output_bytes, norm_weight,
                     kHiddenBytes) ||
      ranges_overlap(raw_down_output, plan->down.output_bytes,
                     residual_output, kHiddenBytes) ||
      ranges_overlap(raw_down_output, plan->down.output_bytes,
                     normalized_output, kHiddenBytes) ||
      ranges_overlap(residual_output, kHiddenBytes, activation,
                     plan->down.input_bytes) ||
      ranges_overlap(residual_output, kHiddenBytes, residual_left,
                     kHiddenBytes) ||
      ranges_overlap(residual_output, kHiddenBytes, norm_weight,
                     kHiddenBytes) ||
      ranges_overlap(residual_output, kHiddenBytes, normalized_output,
                     kHiddenBytes) ||
      ranges_overlap(normalized_output, kHiddenBytes, activation,
                     plan->down.input_bytes) ||
      ranges_overlap(normalized_output, kHiddenBytes, residual_left,
                     kHiddenBytes) ||
      ranges_overlap(normalized_output, kHiddenBytes, norm_weight,
                     kHiddenBytes) ||
      overlaps_projection_weights(residual_output, kHiddenBytes,
                                  plan->down) ||
      overlaps_projection_weights(normalized_output, kHiddenBytes,
                                  plan->down)) {
    return invalid_value();
  }

  if (requires_projection_scratch(backend, down_weight)) {
    const std::size_t scratch_bytes = plan->down.rows * sizeof(float);
    if (ranges_overlap(fp32_scratch, scratch_bytes, residual_left,
                       kHiddenBytes) ||
        ranges_overlap(fp32_scratch, scratch_bytes, norm_weight,
                       kHiddenBytes) ||
        ranges_overlap(fp32_scratch, scratch_bytes, residual_output,
                       kHiddenBytes) ||
        ranges_overlap(fp32_scratch, scratch_bytes, normalized_output,
                       kHiddenBytes)) {
      return invalid_value();
    }
  }
  return static_cast<int>(cudaSuccess);
}

}  // namespace

bool is_valid_projection_backend(const ProjectionBackend backend) noexcept {
  switch (backend) {
    case ProjectionBackend::kReference:
    case ProjectionBackend::kSm87WeightOnly:
      return true;
  }
  return false;
}

std::string_view to_string(const ProjectionBackend backend) noexcept {
  switch (backend) {
    case ProjectionBackend::kReference:
      return "reference";
    case ProjectionBackend::kSm87WeightOnly:
      return "sm87_weight_only";
  }
  return "unknown";
}

int launch_projection_reference_cuda(const LinearWeight& weight,
                                     const std::uint16_t* const input,
                                     float* const output,
                                     void* const cuda_stream) noexcept {
  return std::visit(
      [input, output, cuda_stream](const auto& selected) noexcept -> int {
        using Selected = std::decay_t<decltype(selected)>;
        if (selected.output_size == 0U || selected.input_size == 0U ||
            input == nullptr || output == nullptr) {
          return invalid_value();
        }
        if constexpr (std::is_same_v<Selected, Bf16LinearWeight>) {
          if (selected.weight == nullptr) {
            return invalid_value();
          }
          return kernels::launch_bf16_gemv_reference_cuda(
              selected.weight, input, selected.output_size,
              selected.input_size, output, cuda_stream);
        } else if constexpr (std::is_same_v<Selected, Fp8LinearWeight>) {
          if (selected.weight == nullptr ||
              selected.weight_scale_device == nullptr ||
              selected.input_scale_device == nullptr ||
              !valid_scale(selected.weight_scale) ||
              !valid_scale(selected.input_scale)) {
            return invalid_value();
          }
          return kernels::launch_fp8_gemv_reference_cuda(
              selected.weight, selected.weight_scale, input,
              selected.output_size, selected.input_size, output,
              cuda_stream);
        } else if constexpr (std::is_same_v<Selected,
                                            NvFp4LinearWeight>) {
          if (selected.packed_weight == nullptr ||
              selected.block_scale == nullptr ||
              selected.weight_scale_2_device == nullptr ||
              selected.input_scale_device == nullptr ||
              !valid_scale(selected.weight_scale_2) ||
              !valid_scale(selected.input_scale) ||
              (selected.input_size % 16U) != 0U) {
            return invalid_value();
          }
          return kernels::launch_nvfp4_gemv_reference_cuda(
              selected.packed_weight, selected.block_scale,
              selected.weight_scale_2, input, selected.output_size,
              selected.input_size, output, cuda_stream);
        } else {
          return invalid_value();
        }
      },
      weight);
}

int launch_projection_to_bf16_reference_cuda(
    const LinearWeight& weight, const std::uint16_t* const input,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  const std::size_t output_size = std::visit(
      [](const auto& selected) noexcept { return selected.output_size; },
      weight);
  if (output_size == 0U || scratch_elements < output_size ||
      fp32_scratch == nullptr || output == nullptr) {
    return invalid_value();
  }
  const int projection_status = launch_projection_reference_cuda(
      weight, input, fp32_scratch, cuda_stream);
  if (projection_status != static_cast<int>(cudaSuccess)) {
    return projection_status;
  }
  return launch_fp32_to_bf16_reference_cuda(
      fp32_scratch, output_size, output, cuda_stream);
}

int launch_projection_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, float* const fp32_scratch,
    const std::size_t scratch_elements, std::uint16_t* const output,
    void* const cuda_stream) noexcept {
  if (!is_valid_projection_backend(backend) ||
      weight.valueless_by_exception()) {
    return invalid_value();
  }
  if (backend == ProjectionBackend::kReference) {
    return launch_projection_to_bf16_reference_cuda(
        weight, input, fp32_scratch, scratch_elements, output, cuda_stream);
  }

  switch (weight.index()) {
    case 0U: {
      const auto* const selected = std::get_if<Bf16LinearWeight>(&weight);
      if (selected != nullptr &&
          selected->output_size == kSm87DirectBf16Rows &&
          selected->input_size == kSm87DirectBf16Columns) {
        return kernels::launch_bf16_gemv_bf16_cuda(
            selected->weight, input, selected->output_size,
            selected->input_size, output, cuda_stream);
      }
      // Preserve the FP32-scratch reference path for every BF16 shape that
      // is not the production single-token linear-attention projection.
      return launch_projection_to_bf16_reference_cuda(
          weight, input, fp32_scratch, scratch_elements, output, cuda_stream);
    }
    case 1U: {
      const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
      if (selected == nullptr || selected->output_size == 0U ||
          selected->input_size == 0U || selected->weight == nullptr ||
          selected->weight_scale_device == nullptr ||
          selected->input_scale_device == nullptr ||
          !valid_scale(selected->weight_scale) ||
          !valid_scale(selected->input_scale) || input == nullptr ||
          output == nullptr) {
        return invalid_value();
      }
      if (selected->m1_aosoa4_preswizzled_weight != nullptr &&
          selected->output_size == kFp8M1OutputProjectionRows &&
          selected->input_size == kFp8M1OutputProjectionColumns) {
        return kernels::
            launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_bf16_cuda(
                selected->m1_aosoa4_preswizzled_weight,
                selected->weight_scale, input, selected->output_size,
                selected->input_size, output, cuda_stream);
      }
      return kernels::launch_sm87_fp8_w8a16_gemv_bf16_cuda(
          selected->weight, selected->weight_scale, input,
          selected->output_size, selected->input_size, output, cuda_stream);
    }
    case 2U: {
      const auto* const selected = std::get_if<NvFp4LinearWeight>(&weight);
      if (selected == nullptr || selected->output_size == 0U ||
          selected->input_size == 0U || selected->packed_weight == nullptr ||
          selected->block_scale == nullptr ||
          selected->weight_scale_2_device == nullptr ||
          selected->input_scale_device == nullptr ||
          !valid_scale(selected->weight_scale_2) ||
          !valid_scale(selected->input_scale) ||
          (selected->input_size % 16U) != 0U || input == nullptr ||
          output == nullptr) {
        return invalid_value();
      }
      return kernels::launch_sm87_nvfp4_w4a16_gemv_bf16_cuda(
          selected->packed_weight, selected->block_scale,
          selected->weight_scale_2, input, selected->output_size,
          selected->input_size, output, cuda_stream);
    }
    default:
      return invalid_value();
  }
}

int launch_projection_tile_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, const std::size_t token_count,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  if (token_count == 1U) {
    return launch_projection_to_bf16_cuda(
        backend, weight, input, fp32_scratch, scratch_elements, output,
        cuda_stream);
  }

  ProjectionTileSpans spans;
  const int validation = validate_projection_tile(
      backend, weight, input, token_count, fp32_scratch, scratch_elements,
      output, &spans);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  if (token_count > 32U) {
    // Validate the complete supertile above before enqueueing any prefix or
    // tail work.
    // The exact aligned FP8 attention-output and NVFP4 down projections reuse
    // each decoded weight tile across all 64 rows. Every other backend/shape
    // preserves the established C32 schedule followed by one ordered tail
    // operation.
    if (token_count == 64U &&
        backend == ProjectionBackend::kSm87WeightOnly) {
      if (const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
          selected != nullptr && spans.rows == 5'120U &&
          spans.columns == 6'144U &&
          pointer_is_aligned(selected->weight, 16U) &&
          pointer_is_aligned(input, alignof(std::uint64_t)) &&
          pointer_is_aligned(output, alignof(std::uint16_t))) {
        return kernels::
            launch_sm87_fp8_w8a16_m64_attention_output_gemm_bf16_cuda(
                selected->weight, selected->weight_scale, input, spans.rows,
                spans.columns, output, cuda_stream);
      }
      if (const auto* const selected =
              std::get_if<NvFp4LinearWeight>(&weight);
          selected != nullptr && spans.rows == 5'120U &&
          spans.columns == 17'408U &&
          pointer_is_aligned(selected->packed_weight, 16U) &&
          pointer_is_aligned(selected->block_scale,
                             alignof(std::uint16_t)) &&
          pointer_is_aligned(input, alignof(std::uint64_t)) &&
          pointer_is_aligned(output, alignof(std::uint16_t))) {
        return kernels::launch_sm87_nvfp4_w4a16_m64_down_gemm_bf16_cuda(
            selected->packed_weight, selected->block_scale,
            selected->weight_scale_2, input, spans.rows, spans.columns,
            output, cuda_stream);
      }
    }

    constexpr std::size_t kPrefixTokens = 32U;
    int status = launch_projection_tile_to_bf16_cuda(
        backend, weight, input, kPrefixTokens, fp32_scratch, scratch_elements,
        output, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    return launch_projection_tile_to_bf16_cuda(
        backend, weight, input + kPrefixTokens * spans.columns,
        token_count - kPrefixTokens, fp32_scratch, scratch_elements,
        output + kPrefixTokens * spans.rows, cuda_stream);
  }

  if (backend == ProjectionBackend::kSm87WeightOnly) {
    if (const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
        selected != nullptr) {
      if (token_count == 32U) {
        return kernels::launch_sm87_fp8_w8a16_m32_gemm_bf16_cuda(
            selected->weight, selected->weight_scale, input, spans.rows,
            spans.columns, output, cuda_stream);
      }
      std::size_t token_offset = 0U;
      if (token_count >= 16U) {
        const int status =
            kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
                selected->weight, selected->weight_scale, input,
                spans.rows, spans.columns, output, cuda_stream);
        if (status != static_cast<int>(cudaSuccess)) {
          return status;
        }
        token_offset = 16U;
      }
      while (token_offset < token_count) {
        const std::size_t remaining = token_count - token_offset;
        if (remaining == 16U) {
          return kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
              selected->weight, selected->weight_scale,
              input + token_offset * spans.columns, spans.rows,
              spans.columns, output + token_offset * spans.rows,
              cuda_stream);
        }
        const std::size_t launch_tokens =
            remaining < kMaximumSm87SmallMTokens
                ? remaining
                : kMaximumSm87SmallMTokens;
        const int status =
            kernels::launch_sm87_fp8_w8a16_small_m_gemm_bf16_cuda(
                selected->weight, selected->weight_scale,
                input + token_offset * spans.columns, launch_tokens,
                spans.rows, spans.columns, output + token_offset * spans.rows,
                cuda_stream);
        if (status != static_cast<int>(cudaSuccess)) {
          return status;
        }
        token_offset += launch_tokens;
      }
      return static_cast<int>(cudaSuccess);
    }
    if (const auto* const selected = std::get_if<NvFp4LinearWeight>(&weight);
        selected != nullptr) {
      if (token_count == 17U ||
          (token_count >= 19U && token_count <= 31U)) {
        return kernels::launch_sm87_nvfp4_w4a16_m17_m31_gemm_bf16_cuda(
            selected->packed_weight, selected->block_scale,
            selected->weight_scale_2, input, token_count, spans.rows,
            spans.columns, output, cuda_stream);
      }
      if (token_count == 18U) {
        return kernels::launch_sm87_nvfp4_w4a16_m18_gemm_bf16_cuda(
            selected->packed_weight, selected->block_scale,
            selected->weight_scale_2, input, spans.rows, spans.columns,
            output, cuda_stream);
      }
      if (token_count == 32U) {
        return kernels::launch_sm87_nvfp4_w4a16_m32_gemm_bf16_cuda(
            selected->packed_weight, selected->block_scale,
            selected->weight_scale_2, input, spans.rows, spans.columns, output,
            cuda_stream);
      }
      std::size_t token_offset = 0U;
      if (token_count >= 16U) {
        const int status =
            kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
                selected->packed_weight, selected->block_scale,
                selected->weight_scale_2, input, spans.rows, spans.columns,
                output, cuda_stream);
        if (status != static_cast<int>(cudaSuccess)) {
          return status;
        }
        token_offset = 16U;
      }
      while (token_offset < token_count) {
        const std::size_t remaining = token_count - token_offset;
        if (remaining == 16U) {
          return kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
              selected->packed_weight, selected->block_scale,
              selected->weight_scale_2,
              input + token_offset * spans.columns, spans.rows,
              spans.columns, output + token_offset * spans.rows,
              cuda_stream);
        }
        const std::size_t launch_tokens =
            remaining < kMaximumSm87SmallMTokens
                ? remaining
                : kMaximumSm87SmallMTokens;
        const int status =
            kernels::launch_sm87_nvfp4_w4a16_small_m_gemm_bf16_cuda(
                selected->packed_weight, selected->block_scale,
                selected->weight_scale_2,
                input + token_offset * spans.columns, launch_tokens,
                spans.rows, spans.columns, output + token_offset * spans.rows,
                cuda_stream);
        if (status != static_cast<int>(cudaSuccess)) {
          return status;
        }
        token_offset += launch_tokens;
      }
      return static_cast<int>(cudaSuccess);
    }
  }

  for (std::size_t token = 0U; token < token_count; ++token) {
    // Keep the existing multi-token BF16 fallback contract. The dedicated
    // direct-output BF16 route is intentionally limited to the M=1 entry
    // point above; eligible projection pairs use their separate fused path.
    const int status =
        backend == ProjectionBackend::kSm87WeightOnly && weight.index() == 0U
            ? launch_projection_to_bf16_reference_cuda(
                  weight, input + token * spans.columns, fp32_scratch,
                  scratch_elements, output + token * spans.rows, cuda_stream)
            : launch_projection_to_bf16_cuda(
                  backend, weight, input + token * spans.columns,
                  fp32_scratch, scratch_elements,
                  output + token * spans.rows, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return static_cast<int>(cudaSuccess);
}

int launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, const std::size_t token_count,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  constexpr std::size_t kMaximumWholeChunkTokens = 512U;
  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr std::size_t kQkvSize = 10'240U;
  constexpr std::size_t kZSize = 6'144U;

  if (backend != ProjectionBackend::kSm87WeightOnly ||
      weight.valueless_by_exception() ||
      (token_count != 256U && token_count != 512U)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
  if (selected == nullptr) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const bool qkv_shape = selected->output_size == kQkvSize &&
                         selected->input_size == kHiddenSize;
  const bool z_shape = selected->output_size == kZSize &&
                       selected->input_size == kHiddenSize;
  const bool attention_output_shape =
      selected->output_size == kHiddenSize &&
      selected->input_size == kZSize;
  if (!qkv_shape && !z_shape && !attention_output_shape) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  ProjectionTileSpans spans;
  const int validation = validate_projection_tile(
      backend, weight, input, token_count, nullptr, 0U, output, &spans, true,
      kMaximumWholeChunkTokens);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool whole_chunk_aligned =
      pointer_is_aligned(selected->weight, 16U) &&
      pointer_is_aligned(input, alignof(std::uint64_t)) &&
      pointer_is_aligned(output, alignof(std::uint16_t));
  if (!whole_chunk_aligned) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  return kernels::launch_sm87_fp8_w8a16_whole_chunk_gemm_bf16_cuda(
      selected->weight, selected->weight_scale, input, token_count,
      spans.rows, spans.columns, output, cuda_stream);
}

int launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, const std::size_t token_count,
    std::uint16_t* const output, void* const cuda_stream) noexcept {
  constexpr std::size_t kMaximumWholeChunkTokens = 512U;
  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr std::size_t kIntermediateSize = 17'408U;

  if (backend != ProjectionBackend::kSm87WeightOnly ||
      weight.valueless_by_exception() ||
      (token_count != 256U && token_count != 512U)) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const auto* const selected = std::get_if<NvFp4LinearWeight>(&weight);
  if (selected == nullptr) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const bool gate_up_shape =
      selected->output_size == kIntermediateSize &&
      selected->input_size == kHiddenSize;
  const bool down_shape = selected->output_size == kHiddenSize &&
                          selected->input_size == kIntermediateSize;
  if (!gate_up_shape && !down_shape) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  ProjectionTileSpans spans;
  const int validation = validate_projection_tile(
      backend, weight, input, token_count, nullptr, 0U, output, &spans, true,
      kMaximumWholeChunkTokens);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  const bool whole_chunk_aligned =
      pointer_is_aligned(selected->packed_weight, 16U) &&
      pointer_is_aligned(selected->block_scale, alignof(std::uint16_t)) &&
      pointer_is_aligned(input, alignof(std::uint64_t)) &&
      pointer_is_aligned(output, alignof(std::uint16_t));
  if (!whole_chunk_aligned) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  if (gate_up_shape) {
    return kernels::
        launch_sm87_nvfp4_w4a16_whole_chunk_gate_up_branch_gemm_bf16_cuda(
            selected->packed_weight, selected->block_scale,
            selected->weight_scale_2, input, token_count, spans.rows,
            spans.columns, output, cuda_stream);
  }
  return kernels::launch_sm87_nvfp4_w4a16_whole_chunk_down_gemm_bf16_cuda(
      selected->packed_weight, selected->block_scale,
      selected->weight_scale_2, input, token_count, spans.rows,
      spans.columns, output, cuda_stream);
}

int launch_projection_pair_tile_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight, const std::uint16_t* const input,
    const std::size_t token_count, float* const fp32_scratch,
    const std::size_t scratch_elements,
    std::uint16_t* const first_output,
    std::uint16_t* const second_output,
    void* const cuda_stream) noexcept {
  const bool fused_bf16_pair = supports_bf16_projection_pair(
      backend, first_weight, second_weight);
  const bool eligible_fp8_kv_pair =
      token_count == 1U && supports_fp8_projection_pair(
                               backend, first_weight, second_weight);
  const bool eligible_fp8_qkv_z_pair =
      token_count == 1U && supports_fp8_qkv_z_projection_pair(
                               backend, first_weight, second_weight);
  const auto* const first_nvfp4 =
      std::get_if<NvFp4LinearWeight>(&first_weight);
  const auto* const second_nvfp4 =
      std::get_if<NvFp4LinearWeight>(&second_weight);
  const auto is_exact_nvfp4_mlp_shape = [](const NvFp4LinearWeight& weight) {
    return (weight.output_size == 17'408U && weight.input_size == 5'120U) ||
           (weight.output_size == 5'120U && weight.input_size == 17'408U);
  };
  const bool nvfp4_masked_token_count =
      token_count == 18U || token_count == 17U ||
      (token_count >= 19U && token_count <= 31U);
  const bool eligible_nvfp4_masked_pair =
      backend == ProjectionBackend::kSm87WeightOnly &&
      nvfp4_masked_token_count &&
      first_nvfp4 != nullptr && second_nvfp4 != nullptr &&
      is_exact_nvfp4_mlp_shape(*first_nvfp4) &&
      is_exact_nvfp4_mlp_shape(*second_nvfp4) &&
      pointer_is_aligned(first_nvfp4->packed_weight, 16U) &&
      pointer_is_aligned(second_nvfp4->packed_weight, 16U) &&
      pointer_is_aligned(first_nvfp4->block_scale, 2U) &&
      pointer_is_aligned(second_nvfp4->block_scale, 2U) &&
      pointer_is_aligned(input, 8U) &&
      pointer_is_aligned(first_output, alignof(std::uint16_t)) &&
      pointer_is_aligned(second_output, alignof(std::uint16_t));
  bool aligned_fp8_pair = false;
  if (eligible_fp8_kv_pair || eligible_fp8_qkv_z_pair) {
    const auto& first = std::get<Fp8LinearWeight>(first_weight);
    const auto& second = std::get<Fp8LinearWeight>(second_weight);
    aligned_fp8_pair =
        pointer_is_aligned(first.weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(second.weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(input, alignof(std::uint64_t)) &&
        pointer_is_aligned(first_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(second_output, alignof(std::uint16_t));
  }
  const bool direct_output = fused_bf16_pair || aligned_fp8_pair;
  ProjectionTileSpans first_spans;
  ProjectionTileSpans second_spans;
  const int first_validation = validate_projection_tile(
      backend, first_weight, input, token_count, fp32_scratch,
      scratch_elements, first_output, &first_spans, direct_output);
  if (first_validation != static_cast<int>(cudaSuccess)) {
    return first_validation;
  }
  const int second_validation = validate_projection_tile(
      backend, second_weight, input, token_count, fp32_scratch,
      scratch_elements, second_output, &second_spans, direct_output);
  if (second_validation != static_cast<int>(cudaSuccess)) {
    return second_validation;
  }
  const int cross_validation = validate_projection_pair_cross_ranges(
      backend, first_weight, second_weight, input, fp32_scratch,
      first_output, second_output, first_spans, second_spans,
      direct_output);
  if (cross_validation != static_cast<int>(cudaSuccess)) {
    return cross_validation;
  }

  // Preserve full-operation validation above, then keep the established
  // first-projection-before-second launch order while allowing each exact
  // masked-M32 NVFP4 projection to reach one kernel. Eligibility includes
  // every alignment used by either public launcher; therefore, after both
  // tiles and all cross-ranges pass above, the second call cannot fail a
  // deterministic host-side validation after the first call is enqueued.
  if (eligible_nvfp4_masked_pair) {
    const auto launch = [&](const NvFp4LinearWeight& selected,
                            const ProjectionTileSpans& spans,
                            std::uint16_t* const selected_output) noexcept {
      if (token_count == 18U) {
        return kernels::launch_sm87_nvfp4_w4a16_m18_gemm_bf16_cuda(
            selected.packed_weight, selected.block_scale,
            selected.weight_scale_2, input, spans.rows, spans.columns,
            selected_output, cuda_stream);
      }
      return kernels::launch_sm87_nvfp4_w4a16_m17_m31_gemm_bf16_cuda(
          selected.packed_weight, selected.block_scale,
          selected.weight_scale_2, input, token_count, spans.rows,
          spans.columns, selected_output, cuda_stream);
    };
    int status = launch(*first_nvfp4, first_spans, first_output);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    return launch(*second_nvfp4, second_spans, second_output);
  }

  if (token_count > 16U) {
    for (std::size_t token_offset = 0U; token_offset < token_count;
         token_offset += 16U) {
      const std::size_t remaining = token_count - token_offset;
      const std::size_t launch_tokens = remaining < 16U ? remaining : 16U;
      const int status = launch_projection_pair_tile_to_bf16_cuda(
          backend, first_weight, second_weight,
          input + token_offset * first_spans.columns, launch_tokens,
          fp32_scratch, scratch_elements,
          first_output + token_offset * first_spans.rows,
          second_output + token_offset * second_spans.rows, cuda_stream);
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
    return static_cast<int>(cudaSuccess);
  }

  if (fused_bf16_pair) {
    const auto& first = std::get<Bf16LinearWeight>(first_weight);
    const auto& second = std::get<Bf16LinearWeight>(second_weight);
    if (token_count == 16U) {
      return kernels::launch_bf16_gemv_pair_m16_projection_fused_cuda(
          first.weight, second.weight, input, first_output, second_output,
          cuda_stream);
    }
    return kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
        first.weight, second.weight, input, token_count, first.output_size,
        first.input_size, first_output, second_output, cuda_stream);
  }
  if (aligned_fp8_pair && eligible_fp8_qkv_z_pair) {
    const auto& qkv = std::get<Fp8LinearWeight>(first_weight);
    const auto& z = std::get<Fp8LinearWeight>(second_weight);
    return kernels::launch_sm87_fp8_w8a16_gemv_qkv_z_bf16_cuda(
        qkv.weight, qkv.weight_scale, z.weight, z.weight_scale, input,
        qkv.output_size, z.output_size, qkv.input_size, first_output,
        second_output, cuda_stream);
  }
  if (aligned_fp8_pair && eligible_fp8_kv_pair) {
    const auto& first = std::get<Fp8LinearWeight>(first_weight);
    const auto& second = std::get<Fp8LinearWeight>(second_weight);
    return kernels::launch_sm87_fp8_w8a16_gemv_pair_bf16_cuda(
        first.weight, first.weight_scale, second.weight,
        second.weight_scale, input, first.output_size, first.input_size,
        first_output, second_output, cuda_stream);
  }

  const int first_status = launch_projection_tile_to_bf16_cuda(
      backend, first_weight, input, token_count, fp32_scratch,
      scratch_elements, first_output, cuda_stream);
  if (first_status != static_cast<int>(cudaSuccess)) {
    return first_status;
  }
  return launch_projection_tile_to_bf16_cuda(
      backend, second_weight, input, token_count, fp32_scratch,
      scratch_elements, second_output, cuda_stream);
}

int launch_linear_attention_qkv_z_ab_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& qkv_weight,
    const LinearWeight& z_weight, const LinearWeight& a_weight,
    const LinearWeight& b_weight, const std::uint16_t* const input,
    std::uint16_t* const qkv_output, std::uint16_t* const z_output,
    std::uint16_t* const a_output, std::uint16_t* const b_output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kQkvRows = 10'240U;
  constexpr std::size_t kZRows = 6'144U;
  constexpr std::size_t kAbRows = 48U;
  constexpr std::size_t kColumns = 5'120U;
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      qkv_weight.valueless_by_exception() ||
      z_weight.valueless_by_exception() ||
      a_weight.valueless_by_exception() ||
      b_weight.valueless_by_exception()) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  const auto* const qkv = std::get_if<Fp8LinearWeight>(&qkv_weight);
  const auto* const z = std::get_if<Fp8LinearWeight>(&z_weight);
  const auto* const a = std::get_if<Bf16LinearWeight>(&a_weight);
  const auto* const b = std::get_if<Bf16LinearWeight>(&b_weight);
  if (qkv == nullptr || z == nullptr || a == nullptr || b == nullptr ||
      qkv->output_size != kQkvRows || z->output_size != kZRows ||
      a->output_size != kAbRows || b->output_size != kAbRows ||
      qkv->input_size != kColumns || z->input_size != kColumns ||
      a->input_size != kColumns || b->input_size != kColumns) {
    return static_cast<int>(cudaErrorNotSupported);
  }
  LinearAttentionQkvZAbLaunchPlan plan;
  const int validation = validate_linear_attention_qkv_z_ab_direct_launch(
      backend, qkv_weight, z_weight, a_weight, b_weight, input, qkv_output,
      z_output, a_output, b_output, &plan);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }
  if (!pointer_is_aligned(a->weight, alignof(std::uint16_t)) ||
      !pointer_is_aligned(b->weight, alignof(std::uint16_t))) {
    return invalid_value();
  }
  const bool aligned =
      pointer_is_aligned(qkv->weight, alignof(std::uint32_t)) &&
      pointer_is_aligned(z->weight, alignof(std::uint32_t)) &&
      pointer_is_aligned(input, alignof(std::uint64_t)) &&
      pointer_is_aligned(qkv_output, alignof(std::uint16_t)) &&
      pointer_is_aligned(z_output, alignof(std::uint16_t)) &&
      pointer_is_aligned(a_output, alignof(std::uint16_t)) &&
      pointer_is_aligned(b_output, alignof(std::uint16_t));
  if (!aligned) {
    return static_cast<int>(cudaErrorNotSupported);
  }

  return kernels::launch_sm87_fp8_w8a16_gemv_qkv_z_bf16_ab_pair_cuda(
      qkv->weight, qkv->weight_scale, z->weight, z->weight_scale, a->weight,
      b->weight, input, qkv->output_size, z->output_size, a->output_size,
      qkv->input_size, qkv_output, z_output, a_output, b_output, cuda_stream);
}

int launch_full_attention_q_kv_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& q_weight,
    const LinearWeight& key_weight, const LinearWeight& value_weight,
    const std::uint16_t* const input, float* const fp32_scratch,
    const std::size_t scratch_elements, std::uint16_t* const q_output,
    std::uint16_t* const key_output, std::uint16_t* const value_output,
    void* const cuda_stream) noexcept {
  FullAttentionQKvLaunchPlan plan;
  const int validation = validate_full_attention_q_kv_launch(
      backend, q_weight, key_weight, value_weight, input, fp32_scratch,
      scratch_elements, q_output, key_output, value_output, &plan);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  if (plan.aligned_fp8_fusion) {
    const auto& q = std::get<Fp8LinearWeight>(q_weight);
    const auto& key = std::get<Fp8LinearWeight>(key_weight);
    const auto& value = std::get<Fp8LinearWeight>(value_weight);
    return kernels::launch_sm87_fp8_w8a16_gemv_q_kv_bf16_cuda(
        q.weight, q.weight_scale, key.weight, key.weight_scale, value.weight,
        value.weight_scale, input, q.output_size, key.output_size,
        q.input_size, q_output, key_output, value_output, cuda_stream);
  }

  const int q_status = launch_projection_to_bf16_cuda(
      backend, q_weight, input, fp32_scratch, scratch_elements, q_output,
      cuda_stream);
  if (q_status != static_cast<int>(cudaSuccess)) {
    return q_status;
  }
  return launch_projection_pair_tile_to_bf16_cuda(
      backend, key_weight, value_weight, input, 1U, fp32_scratch,
      scratch_elements, key_output, value_output, cuda_stream);
}

int launch_mlp_gate_up_silu_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* const input,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_output,
    void* const cuda_stream) noexcept {
  MlpGateUpSiluLaunchPlan plan;
  const int validation = validate_mlp_gate_up_silu_launch(
      backend, gate_weight, up_weight, input, fp32_scratch, scratch_elements,
      gate_output, up_output, &plan);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  if (plan.aligned_nvfp4_fusion) {
    const auto& gate = std::get<NvFp4LinearWeight>(gate_weight);
    const auto& up = std::get<NvFp4LinearWeight>(up_weight);
    return kernels::launch_sm87_nvfp4_w4a16_gemv_gate_up_silu_bf16_cuda(
        gate.packed_weight, gate.block_scale, gate.weight_scale_2,
        up.packed_weight, up.block_scale, up.weight_scale_2, input,
        gate.output_size, gate.input_size, gate_output, up_output,
        cuda_stream);
  }

  const int projection_status = launch_projection_pair_tile_to_bf16_cuda(
      backend, gate_weight, up_weight, input, 1U, fp32_scratch,
      scratch_elements, gate_output, up_output, cuda_stream);
  if (projection_status != static_cast<int>(cudaSuccess)) {
    return projection_status;
  }
  return launch_silu_mul_reference_cuda(
      gate_output, up_output, plan.gate.rows, gate_output, cuda_stream);
}

namespace {

enum class MlpUpPublication {
  kRequired,
  kDecodeRunnerDead,
};

int launch_post_attention_residual_norm_mlp_gate_up_silu_impl(
    const MlpUpPublication up_publication,
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight,
    const std::uint16_t* const residual_left,
    std::uint16_t* const residual_right_and_normalized,
    const std::uint16_t* const norm_weight, const float epsilon,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_output,
    void* const cuda_stream) noexcept {
  constexpr std::size_t kHiddenElements = 5'120U;
  constexpr std::size_t kHiddenBytes =
      kHiddenElements * sizeof(std::uint16_t);

  // Validate the complete operation before selecting either route. This also
  // protects the persistent device scalar weights, which the low-level fused
  // ABI does not receive because their already-loaded host values are passed
  // to the kernel by value.
  MlpGateUpSiluLaunchPlan plan;
  const int mlp_validation = validate_mlp_gate_up_silu_launch(
      backend, gate_weight, up_weight, residual_right_and_normalized,
      fp32_scratch, scratch_elements, gate_output, up_output, &plan);
  if (mlp_validation != static_cast<int>(cudaSuccess)) {
    return mlp_validation;
  }
  if (plan.gate.columns != kHiddenElements ||
      !std::isfinite(epsilon) || epsilon <= 0.0F ||
      residual_left == nullptr || residual_right_and_normalized == nullptr ||
      norm_weight == nullptr || residual_output == nullptr ||
      byte_range_overflows(residual_left, kHiddenBytes) ||
      byte_range_overflows(residual_right_and_normalized, kHiddenBytes) ||
      byte_range_overflows(norm_weight, kHiddenBytes) ||
      byte_range_overflows(residual_output, kHiddenBytes) ||
      ranges_overlap(residual_output, kHiddenBytes, residual_left,
                     kHiddenBytes) ||
      ranges_overlap(residual_output, kHiddenBytes,
                     residual_right_and_normalized, kHiddenBytes) ||
      ranges_overlap(residual_output, kHiddenBytes, norm_weight,
                     kHiddenBytes) ||
      ranges_overlap(residual_right_and_normalized, kHiddenBytes,
                     residual_left, kHiddenBytes) ||
      ranges_overlap(residual_right_and_normalized, kHiddenBytes,
                     norm_weight, kHiddenBytes) ||
      ranges_overlap(residual_output, kHiddenBytes, gate_output,
                     plan.gate.output_bytes) ||
      ranges_overlap(residual_output, kHiddenBytes, up_output,
                     plan.up.output_bytes) ||
      ranges_overlap(gate_output, plan.gate.output_bytes, residual_left,
                     kHiddenBytes) ||
      ranges_overlap(gate_output, plan.gate.output_bytes, norm_weight,
                     kHiddenBytes) ||
      ranges_overlap(up_output, plan.up.output_bytes, residual_left,
                     kHiddenBytes) ||
      ranges_overlap(up_output, plan.up.output_bytes, norm_weight,
                     kHiddenBytes) ||
      overlaps_projection_weights(residual_output, kHiddenBytes,
                                  plan.gate) ||
      overlaps_projection_weights(residual_output, kHiddenBytes, plan.up) ||
      overlaps_projection_weights(residual_right_and_normalized,
                                  kHiddenBytes, plan.gate) ||
      overlaps_projection_weights(residual_right_and_normalized,
                                  kHiddenBytes, plan.up)) {
    return invalid_value();
  }

  if (!plan.direct_output) {
    std::size_t scratch_rows = 0U;
    if (requires_projection_scratch(backend, gate_weight)) {
      scratch_rows = plan.gate.rows;
    }
    if (requires_projection_scratch(backend, up_weight) &&
        plan.up.rows > scratch_rows) {
      scratch_rows = plan.up.rows;
    }
    if (scratch_rows != 0U) {
      const std::size_t scratch_bytes = scratch_rows * sizeof(float);
      if (ranges_overlap(fp32_scratch, scratch_bytes, residual_left,
                         kHiddenBytes) ||
          ranges_overlap(fp32_scratch, scratch_bytes, norm_weight,
                         kHiddenBytes) ||
          ranges_overlap(fp32_scratch, scratch_bytes, residual_output,
                         kHiddenBytes)) {
        return invalid_value();
      }
    }
  }

  const bool eligible_fusion = supports_nvfp4_gate_up_silu_fusion(
      backend, gate_weight, up_weight);
  if (eligible_fusion) {
    const auto& gate = std::get<NvFp4LinearWeight>(gate_weight);
    const auto& up = std::get<NvFp4LinearWeight>(up_weight);
    const bool aligned_fusion =
        pointer_is_aligned(gate.packed_weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(up.packed_weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(residual_left, alignof(std::uint16_t)) &&
        pointer_is_aligned(residual_right_and_normalized,
                           alignof(std::uint16_t)) &&
        pointer_is_aligned(norm_weight, alignof(std::uint16_t)) &&
        pointer_is_aligned(residual_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(gate_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(up_output, alignof(std::uint16_t));
    if (aligned_fusion) {
      if (up_publication == MlpUpPublication::kDecodeRunnerDead) {
        return kernels::
            launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_dead_up_bf16_cuda(
                gate.packed_weight, gate.block_scale, gate.weight_scale_2,
                up.packed_weight, up.block_scale, up.weight_scale_2,
                residual_left, residual_right_and_normalized, norm_weight,
                epsilon, gate.output_size, gate.input_size, residual_output,
                gate_output, up_output, cuda_stream);
      }
      return kernels::
          launch_sm87_nvfp4_w4a16_residual_norm_gate_up_silu_bf16_cuda(
              gate.packed_weight, gate.block_scale, gate.weight_scale_2,
              up.packed_weight, up.block_scale, up.weight_scale_2,
              residual_left, residual_right_and_normalized, norm_weight,
              epsilon, gate.output_size, gate.input_size, residual_output,
              gate_output, up_output, cuda_stream);
    }
  }

  const int norm_status = launch_residual_add_centered_rms_norm_5120_cuda(
      residual_left, residual_right_and_normalized, norm_weight, epsilon,
      residual_output, residual_right_and_normalized, cuda_stream);
  if (norm_status != static_cast<int>(cudaSuccess)) {
    return norm_status;
  }
  return launch_mlp_gate_up_silu_to_bf16_cuda(
      backend, gate_weight, up_weight, residual_right_and_normalized,
      fp32_scratch, scratch_elements, gate_output, up_output, cuda_stream);
}

}  // namespace

int launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight,
    const std::uint16_t* const residual_left,
    std::uint16_t* const residual_right_and_normalized,
    const std::uint16_t* const norm_weight, const float epsilon,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_output,
    void* const cuda_stream) noexcept {
  return launch_post_attention_residual_norm_mlp_gate_up_silu_impl(
      MlpUpPublication::kRequired, backend, gate_weight, up_weight,
      residual_left, residual_right_and_normalized, norm_weight, epsilon,
      fp32_scratch, scratch_elements, residual_output, gate_output, up_output,
      cuda_stream);
}

int launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight,
    const std::uint16_t* const residual_left,
    std::uint16_t* const residual_right_and_normalized,
    const std::uint16_t* const norm_weight, const float epsilon,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const residual_output,
    std::uint16_t* const gate_output,
    std::uint16_t* const up_workspace,
    void* const cuda_stream) noexcept {
  return launch_post_attention_residual_norm_mlp_gate_up_silu_impl(
      MlpUpPublication::kDecodeRunnerDead, backend, gate_weight, up_weight,
      residual_left, residual_right_and_normalized, norm_weight, epsilon,
      fp32_scratch, scratch_elements, residual_output, gate_output,
      up_workspace, cuda_stream);
}

int launch_mlp_down_residual_norm_to_bf16_cuda(
    const ProjectionBackend backend, const LinearWeight& down_weight,
    const std::uint16_t* const activation,
    const std::uint16_t* const residual_left,
    const std::uint16_t* const norm_weight, const float epsilon,
    float* const fp32_scratch, const std::size_t scratch_elements,
    std::uint16_t* const raw_down_output,
    std::uint16_t* const residual_output,
    std::uint16_t* const normalized_output,
    void* const cuda_stream) noexcept {
  DownResidualNormLaunchPlan plan;
  const int validation = validate_mlp_down_residual_norm_launch(
      backend, down_weight, activation, residual_left, norm_weight, epsilon,
      fp32_scratch, scratch_elements, raw_down_output, residual_output,
      normalized_output, &plan);
  if (validation != static_cast<int>(cudaSuccess)) {
    return validation;
  }

  if (plan.aligned_nvfp4_fusion) {
    const auto& down = std::get<NvFp4LinearWeight>(down_weight);
    if (down.down_scale6_sidecar != nullptr) {
      return kernels::
          launch_sm87_nvfp4_w4a16_down_residual_norm_scale6_bf16_cuda(
              down.packed_weight, down.down_scale6_sidecar,
              down.down_scale6_base, down.weight_scale_2, activation,
              residual_left, norm_weight, epsilon, down.output_size,
              down.input_size, raw_down_output, residual_output,
              normalized_output, cuda_stream);
    }
    return kernels::
        launch_sm87_nvfp4_w4a16_down_residual_norm_bf16_cuda(
            down.packed_weight, down.block_scale, down.weight_scale_2,
            activation, residual_left, norm_weight, epsilon,
            down.output_size, down.input_size, raw_down_output,
            residual_output, normalized_output, cuda_stream);
  }

  const int projection_status = launch_projection_to_bf16_cuda(
      backend, down_weight, activation, fp32_scratch, scratch_elements,
      raw_down_output, cuda_stream);
  if (projection_status != static_cast<int>(cudaSuccess)) {
    return projection_status;
  }
  return launch_residual_add_centered_rms_norm_5120_cuda(
      residual_left, raw_down_output, norm_weight, epsilon, residual_output,
      normalized_output, cuda_stream);
}

}  // namespace q3x::runtime
