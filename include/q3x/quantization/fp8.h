#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::quantization {

enum class Fp8Status : std::uint8_t {
  kSuccess = 0,
  kInvalidArgument,
  kSizeOverflow,
};

[[nodiscard]] const char* fp8_status_string(Fp8Status status) noexcept;

// CPU correctness reference for a canonical row-major ModelOpt FP8 weight
// tensor. The stored values use NVIDIA/PyTorch E4M3FN and weight_scale is the
// non-negative FP32 multiplier stored for the original tensor:
//
//   real_weight = E4M3FN(weight) * weight_scale
//
// This interface intentionally models the per-tensor scale policy observed in
// the pinned Qwen3.6 ModelOpt checkpoints. Future block/per-channel policies
// must use a distinct interface rather than being inferred from a tensor name.
[[nodiscard]] Fp8Status dequantize_fp8_reference(
    const std::uint8_t* weights, float weight_scale, std::size_t rows,
    std::size_t columns, float* output) noexcept;

// Minimal CUDA correctness reference. All pointer arguments are device
// pointers. The launch is asynchronous on the default stream and returns a
// cudaError_t represented as int so this public header does not depend on CUDA
// headers. Shape/argument failures return cudaErrorInvalidValue.
[[nodiscard]] int launch_fp8_reference_cuda(
    const std::uint8_t* weights, float weight_scale, std::size_t rows,
    std::size_t columns, float* output) noexcept;

}  // namespace q3x::quantization
