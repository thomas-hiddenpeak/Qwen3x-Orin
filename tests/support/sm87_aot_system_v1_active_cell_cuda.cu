#include "sm87_aot_system_v1_active_cell_cuda.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::test::sm87_aot_active_cell {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr std::size_t kMaximumBlocks = 65'535U;

[[nodiscard]] bool checked_multiply(const std::size_t left,
                                    const std::size_t right,
                                    std::size_t& result) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::size_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] bool checked_add(const std::size_t left,
                               const std::size_t right,
                               std::size_t& result) noexcept {
  if (right > std::numeric_limits<std::size_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] constexpr std::size_t ceil_div_nonzero(
    const std::size_t value, const std::size_t divisor) noexcept {
  return 1U + (value - 1U) / divisor;
}

[[nodiscard]] bool positive_finite_scale(const float scale) noexcept {
  return std::isfinite(scale) && scale > 0.0F;
}

[[nodiscard]] bool pointer_aligned(const void* const pointer,
                                   const std::size_t alignment) noexcept {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] bool validate_bf16_span(
    const std::size_t rows, const std::size_t columns,
    const std::size_t row_stride_elements, MaskLayout& layout) noexcept {
  if (rows == 0U || columns == 0U ||
      columns % kColumnsPerMask != 0U ||
      row_stride_elements < columns) {
    return false;
  }

  std::size_t last_row_offset = 0U;
  std::size_t last_element = 0U;
  std::size_t addressed_elements = 0U;
  std::size_t addressed_bytes = 0U;
  if (!checked_multiply(rows - 1U, row_stride_elements, last_row_offset) ||
      !checked_add(last_row_offset, columns - 1U, last_element) ||
      !checked_add(last_element, 1U, addressed_elements) ||
      !checked_multiply(addressed_elements, sizeof(std::uint16_t),
                        addressed_bytes)) {
    return false;
  }
  static_cast<void>(addressed_bytes);

  layout.outer_tiles = ceil_div_nonzero(rows, kActivationRowsPerMask);
  layout.k16_groups = columns / kColumnsPerMask;
  std::size_t mask_bytes = 0U;
  return checked_multiply(layout.outer_tiles, layout.k16_groups,
                          layout.mask_count) &&
         layout.mask_count != 0U &&
         checked_multiply(layout.mask_count, sizeof(std::uint16_t),
                          mask_bytes);
}

[[nodiscard]] bool validate_b_span(const std::size_t output_features,
                                   const std::size_t input_features,
                                   MaskLayout& layout) noexcept {
  if (output_features == 0U || input_features == 0U ||
      input_features % kColumnsPerMask != 0U) {
    return false;
  }
  std::size_t logical_values = 0U;
  if (!checked_multiply(output_features, input_features, logical_values) ||
      logical_values == 0U) {
    return false;
  }
  layout.outer_tiles = ceil_div_nonzero(output_features, kWeightRowsPerMask);
  layout.k16_groups = input_features / kColumnsPerMask;
  std::size_t mask_bytes = 0U;
  return checked_multiply(layout.outer_tiles, layout.k16_groups,
                          layout.mask_count) &&
         layout.mask_count != 0U &&
         checked_multiply(layout.mask_count, sizeof(std::uint16_t),
                          mask_bytes);
}

[[nodiscard]] dim3 launch_grid(const std::size_t mask_count) noexcept {
  const std::size_t requested = ceil_div_nonzero(mask_count, kThreads);
  return dim3(static_cast<unsigned int>(
      std::min(requested, kMaximumBlocks)));
}

__device__ __forceinline__ bool raw_signed_value_is_nonzero(
    const std::uint8_t bits, const std::uint8_t magnitude_mask) noexcept {
  return (bits & magnitude_mask) != 0U;
}

__global__ void bf16_a_k_support_masks_kernel(
    const std::uint16_t* const bf16_bits, const std::size_t rows,
    const std::size_t row_stride_elements, const std::size_t k16_groups,
    const std::size_t mask_count, std::uint16_t* const masks,
    std::uint32_t* const exceptional_flag) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = thread; linear < mask_count; linear += stride) {
    const std::size_t m16 = linear / k16_groups;
    const std::size_t k16 = linear - m16 * k16_groups;
    const std::size_t first_row = m16 * kActivationRowsPerMask;
    const std::size_t first_column = k16 * kColumnsPerMask;
    std::uint16_t mask = 0U;
    for (std::size_t local_k = 0U; local_k < kColumnsPerMask; ++local_k) {
      bool active = false;
      for (std::size_t local_row = 0U;
           local_row < kActivationRowsPerMask; ++local_row) {
        const std::size_t row = first_row + local_row;
        if (row >= rows) {
          break;
        }
        const std::uint16_t bits =
            bf16_bits[row * row_stride_elements + first_column + local_k];
        if ((bits & 0x7f80U) == 0x7f80U) {
          atomicExch(exceptional_flag, 1U);
        }
        // BF16 sign is bit 15. Masking it makes both signed zeros inactive
        // while keeping every finite nonzero, infinity, and NaN active.
        active = active || (bits & 0x7fffU) != 0U;
      }
      if (active) {
        mask = static_cast<std::uint16_t>(mask | (1U << local_k));
      }
    }
    masks[linear] = mask;
  }
}

__global__ void fp8_b_k_support_masks_kernel(
    const std::uint8_t* const fp8_bytes,
    const std::size_t output_features, const std::size_t input_features,
    const std::size_t k16_groups, const std::size_t mask_count,
    std::uint16_t* const masks, std::uint32_t* const exceptional_flag) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = thread; linear < mask_count; linear += stride) {
    const std::size_t n8 = linear / k16_groups;
    const std::size_t k16 = linear - n8 * k16_groups;
    const std::size_t first_row = n8 * kWeightRowsPerMask;
    const std::size_t first_column = k16 * kColumnsPerMask;
    std::uint16_t mask = 0U;
    for (std::size_t local_k = 0U; local_k < kColumnsPerMask; ++local_k) {
      bool active = false;
      for (std::size_t local_row = 0U; local_row < kWeightRowsPerMask;
           ++local_row) {
        const std::size_t row = first_row + local_row;
        if (row >= output_features) {
          break;
        }
        const std::uint8_t bits =
            fp8_bytes[row * input_features + first_column + local_k];
        if ((bits & 0x7fU) == 0x7fU) {
          atomicExch(exceptional_flag, 1U);
        }
        active = active || raw_signed_value_is_nonzero(bits, 0x7fU);
      }
      if (active) {
        mask = static_cast<std::uint16_t>(mask | (1U << local_k));
      }
    }
    masks[linear] = mask;
  }
}

enum class BlockScaleClass : std::uint8_t {
  kZero = 0U,
  kPositiveFinite,
  kInvalid,
};

__device__ __forceinline__ BlockScaleClass classify_e4m3fn_scale(
    const std::uint8_t bits) noexcept {
  const std::uint8_t magnitude = static_cast<std::uint8_t>(bits & 0x7fU);
  if (magnitude == 0U) {
    return BlockScaleClass::kZero;
  }
  // E4M3FN has only the terminal magnitude 0x7f as NaN. Scales are
  // non-negative, so every negative nonzero code is invalid as well.
  if ((bits & 0x80U) != 0U || magnitude == 0x7fU) {
    return BlockScaleClass::kInvalid;
  }
  return BlockScaleClass::kPositiveFinite;
}

__global__ void nvfp4_b_k_support_masks_kernel(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales_e4m3fn,
    const std::size_t output_features, const std::size_t input_features,
    const std::size_t k16_groups, const std::size_t mask_count,
    std::uint16_t* const masks, std::uint32_t* const exceptional_flag) {
  const std::size_t thread =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const std::size_t stride =
      static_cast<std::size_t>(gridDim.x) * blockDim.x;
  const std::size_t packed_row_bytes = input_features / 2U;
  for (std::size_t linear = thread; linear < mask_count; linear += stride) {
    const std::size_t n8 = linear / k16_groups;
    const std::size_t k16 = linear - n8 * k16_groups;
    const std::size_t first_row = n8 * kWeightRowsPerMask;
    const std::size_t first_column = k16 * kColumnsPerMask;
    std::uint16_t mask = 0U;
    for (std::size_t local_row = 0U; local_row < kWeightRowsPerMask;
         ++local_row) {
      const std::size_t row = first_row + local_row;
      if (row >= output_features) {
        break;
      }
      const BlockScaleClass scale_class = classify_e4m3fn_scale(
          block_scales_e4m3fn[row * k16_groups + k16]);
      if (scale_class == BlockScaleClass::kInvalid) {
        atomicExch(exceptional_flag, 1U);
        // Invalid device data cannot be surfaced synchronously by this API.
        // All-active is the fail-closed result for downstream pruning.
        mask = 0xffffU;
        break;
      }
      if (scale_class == BlockScaleClass::kZero) {
        continue;
      }
      const std::size_t row_offset = row * packed_row_bytes;
      for (std::size_t local_k = 0U; local_k < kColumnsPerMask; ++local_k) {
        const std::size_t column = first_column + local_k;
        const std::uint8_t packed =
            packed_weights[row_offset + column / 2U];
        const unsigned int shift = 4U * static_cast<unsigned int>(column & 1U);
        const std::uint8_t nibble =
            static_cast<std::uint8_t>((packed >> shift) & 0x0fU);
        if (raw_signed_value_is_nonzero(nibble, 0x07U)) {
          mask = static_cast<std::uint16_t>(mask | (1U << local_k));
        }
      }
    }
    masks[linear] = mask;
  }
}

}  // namespace

bool bf16_a_mask_layout(const std::size_t rows, const std::size_t columns,
                        const std::size_t row_stride_elements,
                        MaskLayout* const layout) noexcept {
  if (layout == nullptr) {
    return false;
  }
  *layout = {};
  MaskLayout candidate;
  if (!validate_bf16_span(rows, columns, row_stride_elements, candidate)) {
    return false;
  }
  *layout = candidate;
  return true;
}

bool canonical_b_mask_layout(const std::size_t output_features,
                             const std::size_t input_features,
                             MaskLayout* const layout) noexcept {
  if (layout == nullptr) {
    return false;
  }
  *layout = {};
  MaskLayout candidate;
  if (!validate_b_span(output_features, input_features, candidate)) {
    return false;
  }
  *layout = candidate;
  return true;
}

cudaError_t launch_bf16_a_k_support_masks(
    const std::uint16_t* const bf16_bits, const std::size_t rows,
    const std::size_t columns, const std::size_t row_stride_elements,
    std::uint16_t* const masks, const std::size_t mask_element_count,
    std::uint32_t* const exceptional_flag,
    const cudaStream_t stream) noexcept {
  MaskLayout layout;
  if (bf16_bits == nullptr || masks == nullptr || exceptional_flag == nullptr ||
      !pointer_aligned(bf16_bits, alignof(std::uint16_t)) ||
      !pointer_aligned(masks, alignof(std::uint16_t)) ||
      !pointer_aligned(exceptional_flag, alignof(std::uint32_t)) ||
      !validate_bf16_span(rows, columns, row_stride_elements, layout) ||
      mask_element_count != layout.mask_count) {
    return cudaErrorInvalidValue;
  }
  bf16_a_k_support_masks_kernel<<<launch_grid(layout.mask_count), kThreads, 0U,
                                  stream>>>(
      bf16_bits, rows, row_stride_elements, layout.k16_groups,
      layout.mask_count, masks, exceptional_flag);
  return cudaPeekAtLastError();
}

cudaError_t launch_fp8_b_k_support_masks(
    const std::uint8_t* const fp8_bytes,
    const std::size_t output_features, const std::size_t input_features,
    const float tensor_scale, std::uint16_t* const masks,
    const std::size_t mask_element_count,
    std::uint32_t* const exceptional_flag,
    const cudaStream_t stream) noexcept {
  MaskLayout layout;
  std::size_t source_bytes = 0U;
  if (fp8_bytes == nullptr || masks == nullptr || exceptional_flag == nullptr ||
      !pointer_aligned(masks, alignof(std::uint16_t)) ||
      !pointer_aligned(exceptional_flag, alignof(std::uint32_t)) ||
      !positive_finite_scale(tensor_scale) ||
      !validate_b_span(output_features, input_features, layout) ||
      !checked_multiply(output_features, input_features, source_bytes) ||
      source_bytes == 0U || mask_element_count != layout.mask_count) {
    return cudaErrorInvalidValue;
  }
  fp8_b_k_support_masks_kernel<<<launch_grid(layout.mask_count), kThreads, 0U,
                                 stream>>>(
      fp8_bytes, output_features, input_features, layout.k16_groups,
      layout.mask_count, masks, exceptional_flag);
  return cudaPeekAtLastError();
}

cudaError_t launch_nvfp4_b_k_support_masks(
    const std::uint8_t* const packed_weights,
    const std::uint8_t* const block_scales_e4m3fn,
    const std::size_t output_features, const std::size_t input_features,
    const float global_scale, std::uint16_t* const masks,
    const std::size_t mask_element_count,
    std::uint32_t* const exceptional_flag,
    const cudaStream_t stream) noexcept {
  MaskLayout layout;
  std::size_t packed_bytes = 0U;
  std::size_t scale_bytes = 0U;
  if (packed_weights == nullptr || block_scales_e4m3fn == nullptr ||
      masks == nullptr ||
      !pointer_aligned(masks, alignof(std::uint16_t)) ||
      exceptional_flag == nullptr ||
      !pointer_aligned(exceptional_flag, alignof(std::uint32_t)) ||
      !positive_finite_scale(global_scale) ||
      !validate_b_span(output_features, input_features, layout) ||
      !checked_multiply(output_features, input_features / 2U,
                        packed_bytes) ||
      !checked_multiply(output_features, input_features / kColumnsPerMask,
                        scale_bytes) ||
      packed_bytes == 0U || scale_bytes == 0U ||
      mask_element_count != layout.mask_count) {
    return cudaErrorInvalidValue;
  }
  nvfp4_b_k_support_masks_kernel<<<launch_grid(layout.mask_count), kThreads,
                                   0U, stream>>>(
      packed_weights, block_scales_e4m3fn, output_features, input_features,
      layout.k16_groups, layout.mask_count, masks, exceptional_flag);
  return cudaPeekAtLastError();
}

}  // namespace q3x::test::sm87_aot_active_cell
