#include "q3x/runtime/model_weights.h"

#include "q3x/kernels/reference_gemv.h"
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
        } else {
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

}  // namespace q3x::runtime
