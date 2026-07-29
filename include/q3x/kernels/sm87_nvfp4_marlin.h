#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Fixed Qwen3.6-27B/Orin contract. This API intentionally does not expose a
// generic Marlin surface: production callers may use only M512 BF16 x NVFP4
// Gate+Up and Down projections on an exact 16-SM SM87 device.
inline constexpr std::size_t kSm87NvFp4MarlinTokens = 512U;
inline constexpr std::size_t kSm87NvFp4MarlinHidden = 5'120U;
inline constexpr std::size_t kSm87NvFp4MarlinIntermediate = 17'408U;
inline constexpr std::size_t kSm87NvFp4MarlinGateUpOutput = 34'816U;
inline constexpr std::size_t kSm87NvFp4MarlinSmCount = 16U;
inline constexpr std::size_t kSm87NvFp4MarlinThreadM = 64U;
inline constexpr std::size_t kSm87NvFp4MarlinThreadN = 256U;
inline constexpr std::size_t kSm87NvFp4MarlinThreadK = 64U;
inline constexpr std::size_t kSm87NvFp4MarlinPipelineStages = 4U;
inline constexpr std::size_t kSm87NvFp4MarlinDynamicSharedBytes = 166'912U;
inline constexpr std::size_t kSm87NvFp4MarlinLockBytes =
    kSm87NvFp4MarlinSmCount * sizeof(std::int32_t);
inline constexpr std::size_t kSm87NvFp4MarlinReductionElements =
    kSm87NvFp4MarlinSmCount * kSm87NvFp4MarlinThreadM *
    kSm87NvFp4MarlinThreadN;
inline constexpr std::size_t kSm87NvFp4MarlinReductionBytes =
    kSm87NvFp4MarlinReductionElements * sizeof(float);

[[nodiscard]] constexpr std::size_t sm87_nvfp4_marlin_weight_bytes(
    const std::size_t output_size, const std::size_t input_size) noexcept {
  return output_size * input_size / 2U;
}

[[nodiscard]] constexpr std::size_t sm87_nvfp4_marlin_scale_bytes(
    const std::size_t output_size, const std::size_t input_size) noexcept {
  return output_size * input_size / 16U;
}

// Derives the same power-of-two scale factor as vLLM's
// nvfp4_marlin_process_scales. Inputs are authenticated canonical ModelOpt
// E4M3FN scale payloads in host memory. Gate+Up passes both tensors so the
// merged projection uses one common factor. Negative/NaN encodings fail.
[[nodiscard]] bool derive_sm87_nvfp4_marlin_scale_factor(
    const std::uint8_t* first_scales, std::size_t first_scale_bytes,
    const std::uint8_t* second_scales, std::size_t second_scale_bytes,
    float* scale_factor) noexcept;

// Load-time-only transformation from two canonical ModelOpt matrices
// [17408,5120/2] into the single vLLM Marlin N=34816 layout. transpose_scratch
// is a disposable device buffer exactly as large as marlin_weight. The output
// block-scale tensor uses vLLM's permuted S0E5M3 byte representation. The
// supplied scale factor must be derived jointly from both scale payloads.
[[nodiscard]] int prepare_sm87_nvfp4_marlin_gate_up_cuda(
    const std::uint8_t* canonical_gate_weight,
    const std::uint8_t* canonical_up_weight,
    const std::uint8_t* canonical_gate_scales,
    const std::uint8_t* canonical_up_scales,
    const float* canonical_shared_weight_scale_2_device,
    float scale_factor, std::uint8_t* marlin_weight,
    std::uint8_t* marlin_scales, float* marlin_global_scale,
    void* transpose_scratch, std::size_t transpose_scratch_bytes,
    void* cuda_stream = nullptr) noexcept;

// Load-time-only transformation for canonical Down [5120,17408/2].
[[nodiscard]] int prepare_sm87_nvfp4_marlin_down_cuda(
    const std::uint8_t* canonical_weight,
    const std::uint8_t* canonical_scales,
    const float* canonical_weight_scale_2_device, float scale_factor,
    std::uint8_t* marlin_weight, std::uint8_t* marlin_scales,
    float* marlin_global_scale, void* transpose_scratch,
    std::size_t transpose_scratch_bytes,
    void* cuda_stream = nullptr) noexcept;

// Locks must be zero-initialized before their first launch. The vendored
// stripe scheduler returns them to zero, so one workspace is reusable on an
// ordered stream. C_tmp is the fixed FP32 cross-CTA reduction workspace.
[[nodiscard]] int launch_sm87_nvfp4_marlin_gate_up_m512_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::uint16_t* merged_gate_up_output, float* c_tmp,
    std::int32_t* locks, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int launch_sm87_nvfp4_marlin_down_m512_cuda(
    const std::uint16_t* input, const std::uint8_t* marlin_weight,
    const std::uint8_t* marlin_scales, const float* marlin_global_scale,
    std::uint16_t* output, float* c_tmp, std::int32_t* locks,
    void* cuda_stream = nullptr) noexcept;

// Converts the row-major merged [512,34816] Marlin result into the canonical
// BF16 SiLU(gate)*up [512,17408] Down input without splitting Gate/Up into two
// additional matrices.
[[nodiscard]] int launch_sm87_nvfp4_marlin_gate_up_silu_m512_cuda(
    const std::uint16_t* merged_gate_up, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

struct Sm87NvFp4MarlinKernelResources {
  int registers_per_thread = 0;
  int static_shared_bytes = 0;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

[[nodiscard]] int query_sm87_nvfp4_marlin_m512_resources_cuda(
    Sm87NvFp4MarlinKernelResources* resources) noexcept;

}  // namespace q3x::kernels
