#include "q3x/runtime/model_weights.h"

#include "q3x/kernels/reference_gemv.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/runtime/decode_ops.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <variant>

namespace q3x::runtime {
namespace {

[[nodiscard]] bool valid_scale(const float value) noexcept {
  return std::isfinite(value) && value >= 0.0F;
}

[[nodiscard]] int invalid_value() noexcept {
  return static_cast<int>(cudaErrorInvalidValue);
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
    case 0U:
      // The optimized policy is weight-only. Preserve the BF16 reference
      // contract rather than silently changing its accumulation behavior.
      return launch_projection_to_bf16_reference_cuda(
          weight, input, fp32_scratch, scratch_elements, output, cuda_stream);
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

}  // namespace q3x::runtime
