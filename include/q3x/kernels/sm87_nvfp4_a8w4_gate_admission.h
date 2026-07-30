#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-only, approximate large-M direction probe for the exact Qwen3.6-27B
// layer-0 Gate shape on a 16-SM SM87 device.  This is deliberately not a
// generic quantization API and is not compiled unless the independent CMake
// admission switch is enabled.
//
// The candidate reconstructs the canonical ModelOpt tensor at load time, then
// requantizes every K32 group to symmetric signed W4:
//
//   beta[n,g] = max(abs(W[n, g*32:(g+1)*32])) / 7
//   q_w        = clamp(round_to_nearest_even(W / beta), -7, 7)
//   rho        = max(beta) / 4096
//   s16[n,g]   = round_to_nearest_even(beta / rho)
//   W_hat      = q_w * s16 * rho
//
// Each invocation dynamically quantizes one BF16 activation row:
//
//   alpha[m] = max(abs(A[m,:])) / 127
//   q_a      = clamp(round_to_nearest_even(A / alpha), -127, 127)
//   C_hat    = BF16((q_a @ (q_w*s16)) * (alpha*rho))
//
// This is not bit-exact NVFP4.  The production/canonical BF16 x NVFP4 route
// remains the mandatory fallback until real-runner accuracy and performance
// admission are both passed. The direction ELF intentionally retains the
// locked exact Marlin sidecars beside this Gate-only view: RUN-off measures
// exact Gate+Up+Down, while RUN-on substitutes only Gate and reuses Marlin
// Down after the fork/join.
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionTokens = 512U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionRows = 17'408U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionColumns = 5'120U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionGroupSize = 32U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionGroups = 160U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionSmCount = 16U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionThreads = 256U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionThreadM = 64U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionThreadN = 256U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionThreadK = 64U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionStages = 4U;

// Marlin's A8 specialization reserves 256 bytes for row scales, 33,792 for
// the B/reduction/bias overlay, 4,096 for four scale stages, and 16,384 for A.
inline constexpr std::size_t
    kSm87NvFp4A8W4GateAdmissionDynamicSharedBytes = 54'528U;
inline constexpr std::size_t
    kSm87NvFp4A8W4GateAdmissionRepackSharedBytes = 4'096U;

inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionWeightBytes =
    kSm87NvFp4A8W4GateAdmissionRows *
    kSm87NvFp4A8W4GateAdmissionColumns / 2U;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionScaleElements =
    kSm87NvFp4A8W4GateAdmissionRows *
    kSm87NvFp4A8W4GateAdmissionGroups;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionScaleBytes =
    kSm87NvFp4A8W4GateAdmissionScaleElements * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionSidecarBytes =
    kSm87NvFp4A8W4GateAdmissionWeightBytes +
    kSm87NvFp4A8W4GateAdmissionScaleBytes + sizeof(float);

// Load-time scratch contains a canonical [K/8,N] uint32 staging tensor, one
// FP32 beta per K32/N group, and one FP32 maximum.  It may be released after
// prepare returns and the stream is synchronized.
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionScratchBytes =
    kSm87NvFp4A8W4GateAdmissionWeightBytes +
    kSm87NvFp4A8W4GateAdmissionScaleElements * sizeof(float) + sizeof(float);

inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionActivationBytes =
    kSm87NvFp4A8W4GateAdmissionTokens *
    kSm87NvFp4A8W4GateAdmissionColumns;
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionRowScaleBytes =
    kSm87NvFp4A8W4GateAdmissionTokens * sizeof(float);
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionOutputBytes =
    kSm87NvFp4A8W4GateAdmissionTokens *
    kSm87NvFp4A8W4GateAdmissionRows * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionReductionBytes =
    kSm87NvFp4A8W4GateAdmissionSmCount *
    kSm87NvFp4A8W4GateAdmissionThreadM *
    kSm87NvFp4A8W4GateAdmissionThreadN * sizeof(float);
inline constexpr std::size_t kSm87NvFp4A8W4GateAdmissionLockBytes =
    kSm87NvFp4A8W4GateAdmissionSmCount * sizeof(std::int32_t);

static_assert(kSm87NvFp4A8W4GateAdmissionWeightBytes == 44'564'480U);
static_assert(kSm87NvFp4A8W4GateAdmissionScaleBytes == 5'570'560U);
static_assert(kSm87NvFp4A8W4GateAdmissionSidecarBytes == 50'135'044U);
static_assert(kSm87NvFp4A8W4GateAdmissionScratchBytes == 55'705'604U);

// All pointers are device pointers.  canonical_weight is [N,K/2],
// canonical_block_scales is [N,K/16], and canonical_weight_scale_2 is the
// authenticated scalar stored by ModelOpt.  The two sidecar outputs are the
// packed Marlin weight and permuted raw-int16-in-BF16 scale tensors.  rho is a
// single FP32 device scalar.  enable_approximate_admission must be true; false
// fails before device probing or work enqueueing and forms the independent
// runtime admission gate.
[[nodiscard]] int prepare_sm87_nvfp4_a8w4_gate_m512_admission_cuda(
    const std::uint8_t* canonical_weight,
    const std::uint8_t* canonical_block_scales,
    const float* canonical_weight_scale_2,
    std::uint8_t* marlin_weight,
    std::uint16_t* marlin_integer_scales,
    float* rho,
    void* scratch,
    std::size_t scratch_bytes,
    bool enable_approximate_admission,
    void* cuda_stream = nullptr) noexcept;

// Dynamic per-row BF16 -> S8 quantization.  row_scales stores alpha*rho, the
// exact scale expected by the A8 Marlin epilogue.  This launch is part of the
// timed candidate path; it is never treated as load-time work.
[[nodiscard]] int launch_sm87_dynamic_a8_gate_m512_admission_cuda(
    const std::uint16_t* activation,
    const float* rho,
    std::int8_t* quantized_activation,
    float* row_scales,
    bool enable_approximate_admission,
    void* cuda_stream = nullptr) noexcept;

// A8xW4 Gate GEMM after dynamic quantization.  locks must be zero initialized
// before first use and are restored by the ordered Marlin scheduler.
[[nodiscard]] int launch_sm87_nvfp4_a8w4_gate_m512_admission_cuda(
    const std::int8_t* quantized_activation,
    const float* row_scales,
    const std::uint8_t* marlin_weight,
    const std::uint16_t* marlin_integer_scales,
    std::uint16_t* output,
    float* reduction_workspace,
    std::int32_t* locks,
    bool enable_approximate_admission,
    void* cuda_stream = nullptr) noexcept;

struct Sm87NvFp4A8W4GateAdmissionResources {
  int registers_per_thread = 0;
  int static_shared_bytes = 0;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
};

[[nodiscard]] int query_sm87_nvfp4_a8w4_gate_m512_admission_resources_cuda(
    Sm87NvFp4A8W4GateAdmissionResources* resources,
    bool enable_approximate_admission) noexcept;

}  // namespace q3x::kernels
