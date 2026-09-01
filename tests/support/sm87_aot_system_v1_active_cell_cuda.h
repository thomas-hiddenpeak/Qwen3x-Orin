#pragma once

#if !defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS)
#error "The SM87 AOT active-cell CUDA support is private and default-off"
#endif

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace q3x::test::sm87_aot_active_cell {

inline constexpr std::size_t kActivationRowsPerMask = 16U;
inline constexpr std::size_t kWeightRowsPerMask = 8U;
inline constexpr std::size_t kColumnsPerMask = 16U;

struct MaskLayout {
  // BF16 A uses ceil(rows / 16) outer tiles. FP8/NVFP4 B uses
  // ceil(output_features / 8) outer tiles.
  std::size_t outer_tiles = 0U;
  std::size_t k16_groups = 0U;
  std::size_t mask_count = 0U;
};

// Validates the complete logical A span, including row-stride and output-size
// arithmetic. `columns` must be a positive multiple of 16 and
// `row_stride_elements >= columns`. No pointer is inspected by this helper.
[[nodiscard]] bool bf16_a_mask_layout(std::size_t rows,
                                      std::size_t columns,
                                      std::size_t row_stride_elements,
                                      MaskLayout* layout) noexcept;

// Validates the common canonical B geometry. `input_features` must be a
// positive multiple of 16. The returned row-major layout is
// [ceil(output_features / 8)][input_features / 16].
[[nodiscard]] bool canonical_b_mask_layout(std::size_t output_features,
                                           std::size_t input_features,
                                           MaskLayout* layout) noexcept;

// `bf16_bits` is a device-resident row-major BF16-bit matrix with the supplied
// element stride. `masks` is device storage with exactly
// `mask_element_count == ceil(rows / 16) * (columns / 16)` elements. In each
// uint16_t result, bit k is one iff at least one valid row in that M16 tile has
// a nonzero BF16 bit pattern at the corresponding K position. Both +0 and -0
// are zero. NaN and infinity remain active so this support mask cannot hide an
// exceptional value. Tail-row lanes and all out-of-range bits are zero.
// `exceptional_flag` is caller-owned, 4-byte-aligned device storage. The
// caller must clear it before launch. Any BF16 NaN or infinity atomically sets
// it to one; the kernel never clears it.
//
// Invalid host-visible arguments return cudaErrorInvalidValue before any work
// is enqueued. A successful return means only that the asynchronous kernel was
// accepted by CUDA; this function performs no device synchronization. Input
// and output device ranges must be valid for the documented shapes and must
// not overlap.
[[nodiscard]] cudaError_t launch_bf16_a_k_support_masks(
    const std::uint16_t* bf16_bits, std::size_t rows, std::size_t columns,
    std::size_t row_stride_elements, std::uint16_t* masks,
    std::size_t mask_element_count, std::uint32_t* exceptional_flag,
    cudaStream_t stream) noexcept;

// `fp8_bytes` is canonical device-resident row-major [N][K] E4M3FN storage.
// `masks` has exactly ceil(N / 8) * (K / 16) elements. Bit k is one iff at
// least one valid N row has a nonzero raw FP8 value at that K position. Signed
// zero is zero; every other raw code, including a terminal exceptional code,
// remains active and therefore cannot be pruned by this low-level mask.
// `tensor_scale` must be positive and finite or the call fails before launch.
// `exceptional_flag` follows the same caller-clear contract and is atomically
// set to one for either signed terminal E4M3FN code; it is never cleared.
[[nodiscard]] cudaError_t launch_fp8_b_k_support_masks(
    const std::uint8_t* fp8_bytes, std::size_t output_features,
    std::size_t input_features, float tensor_scale, std::uint16_t* masks,
    std::size_t mask_element_count, std::uint32_t* exceptional_flag,
    cudaStream_t stream) noexcept;

// `packed_weights` is canonical device-resident row-major [N][K/2] NVFP4
// storage (low nibble is even K), and `block_scales_e4m3fn` is canonical
// [N][K/16] E4M3FN storage. The output layout matches FP8 B. Signed E2M1 zero
// is zero. A positive finite nonzero block scale enables its K16 group; a
// signed-zero block scale makes the mathematical block zero. Since this
// asynchronous API has no device-to-host error channel, a negative nonzero or
// terminal-NaN block-scale code conservatively publishes all 16 support bits
// for that row/group instead of falsely proving it inactive. `global_scale`
// must be positive and finite or the call fails before launch. The caller-
// cleared `exceptional_flag` is atomically set to one for either invalid block
// scale class and is never cleared by the kernel.
[[nodiscard]] cudaError_t launch_nvfp4_b_k_support_masks(
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales_e4m3fn,
    std::size_t output_features, std::size_t input_features,
    float global_scale, std::uint16_t* masks,
    std::size_t mask_element_count, std::uint32_t* exceptional_flag,
    cudaStream_t stream) noexcept;

}  // namespace q3x::test::sm87_aot_active_cell
