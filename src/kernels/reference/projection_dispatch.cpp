#include "q3x/runtime/model_weights.h"

#include "q3x/kernels/reference_gemv.h"
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/request_state.h"

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
    const bool direct_output = false) noexcept {
  if (!is_valid_projection_backend(backend) ||
      weight.valueless_by_exception() || token_count == 0U ||
      token_count > kMaximumRequestPrefillChunkSize || input == nullptr ||
      output == nullptr || spans == nullptr) {
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

  if (backend == ProjectionBackend::kSm87WeightOnly) {
    if (const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
        selected != nullptr) {
      if (token_count == 16U) {
        return kernels::launch_sm87_fp8_w8a16_m16_gemm_bf16_cuda(
            selected->weight, selected->weight_scale, input, spans.rows,
            spans.columns, output, cuda_stream);
      }
      for (std::size_t token_offset = 0U; token_offset < token_count;) {
        const std::size_t remaining = token_count - token_offset;
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
      if (token_count == 16U) {
        return kernels::launch_sm87_nvfp4_w4a16_m16_gemm_bf16_cuda(
            selected->packed_weight, selected->block_scale,
            selected->weight_scale_2, input, spans.rows, spans.columns,
            output, cuda_stream);
      }
      for (std::size_t token_offset = 0U; token_offset < token_count;) {
        const std::size_t remaining = token_count - token_offset;
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
  const bool eligible_fp8_pair =
      token_count == 1U && supports_fp8_projection_pair(
                               backend, first_weight, second_weight);
  bool fused_fp8_pair = false;
  if (eligible_fp8_pair) {
    const auto& first = std::get<Fp8LinearWeight>(first_weight);
    const auto& second = std::get<Fp8LinearWeight>(second_weight);
    fused_fp8_pair =
        pointer_is_aligned(first.weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(second.weight, alignof(std::uint32_t)) &&
        pointer_is_aligned(input, alignof(std::uint64_t)) &&
        pointer_is_aligned(first_output, alignof(std::uint16_t)) &&
        pointer_is_aligned(second_output, alignof(std::uint16_t));
  }
  const bool direct_output = fused_bf16_pair || fused_fp8_pair;
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

  if (fused_bf16_pair) {
    const auto& first = std::get<Bf16LinearWeight>(first_weight);
    const auto& second = std::get<Bf16LinearWeight>(second_weight);
    return kernels::launch_bf16_gemv_pair_tile_bf16_cuda(
        first.weight, second.weight, input, token_count, first.output_size,
        first.input_size, first_output, second_output, cuda_stream);
  }
  if (fused_fp8_pair) {
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

}  // namespace q3x::runtime
