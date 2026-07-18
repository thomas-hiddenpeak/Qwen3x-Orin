#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::quantization {

inline constexpr std::size_t kNvFp4GroupSize = 16;
inline constexpr std::size_t kNvFp4ValuesPerByte = 2;

// ModelOpt serializes NVFP4 weights as two E2M1 values per byte. Along K,
// even elements occupy the low nibble and odd elements the high nibble.
[[nodiscard]] std::uint8_t unpack_e2m1_nibble(
    std::uint8_t packed,
    bool high_nibble) noexcept;

// Decode one raw E2M1 nibble. All 16 encodings are finite. The 0x8 encoding
// is preserved as negative zero.
[[nodiscard]] float decode_e2m1(std::uint8_t nibble) noexcept;

// Decode one raw NVIDIA/PyTorch float8 E4M3FN value. FP8 is not an IEEE 754
// interchange format. E4M3FN has no infinities; encodings 0x7f and 0xff are
// NaNs and the largest finite magnitude is 448. ModelOpt block scales are
// non-negative, but the decoder intentionally handles every bit pattern for
// use in checkpoint validation.
[[nodiscard]] float decode_e4m3fn(std::uint8_t bits) noexcept;

// Decode one packed weight using ModelOpt's native scale convention:
//
//   real_weight = E2M1(weight) * E4M3FN(weight_scale) * weight_scale_2
//
// weight_scale_2 is the value stored by ModelOpt for this original tensor
// (amax / 2688), not its reciprocal. A fused tensor or MoE pack must retain
// the distinct value belonging to every source tensor/expert.
[[nodiscard]] float dequantize_nvfp4_value(
    std::uint8_t packed,
    bool high_nibble,
    std::uint8_t weight_scale,
    float weight_scale_2) noexcept;

enum class NvFp4Status : std::uint8_t {
    kSuccess = 0,
    kInvalidArgument,
    kInvalidColumnCount,
    kSizeOverflow,
};

[[nodiscard]] const char* nvfp4_status_string(NvFp4Status status) noexcept;

// CPU correctness reference for the canonical row-major checkpoint view of a
// [rows, columns] matrix, before any Marlin/backend repack. packed_weights has
// logical shape [rows, columns / 2] and weight_scales has logical shape
// [rows, columns / 16]. All rows must belong to one original tensor sharing
// the supplied weight_scale_2; expert-batched inputs need a future per-expert
// interface. columns must be a multiple of 16.
[[nodiscard]] NvFp4Status dequantize_nvfp4_reference(
    const std::uint8_t* packed_weights,
    const std::uint8_t* weight_scales,
    float weight_scale_2,
    std::size_t rows,
    std::size_t columns,
    float* output) noexcept;

// Minimal CUDA correctness reference. All pointer arguments are device
// pointers. The launch is asynchronous on the default stream and the return
// value is a cudaError_t represented as int, keeping this public header free
// of a CUDA header dependency. Shape/argument failures return
// cudaErrorInvalidValue.
[[nodiscard]] int launch_nvfp4_reference_cuda(
    const std::uint8_t* packed_weights,
    const std::uint8_t* weight_scales,
    float weight_scale_2,
    std::size_t rows,
    std::size_t columns,
    float* output) noexcept;

}  // namespace q3x::quantization
