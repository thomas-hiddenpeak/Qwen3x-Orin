#pragma once

#include "q3x/kernels/sm87_target_aot_projection_cuda.h"

#include <cstddef>
#include <cstdint>

// Source-private seam for the real-checkpoint layer-0 numerical/resource
// oracle. This is deliberately outside the installed include tree. It admits
// exactly M192, plus an explicitly requested exact-P40000 quick-kill screen,
// and cannot be reached by a runner, selector, or public launch surface.
namespace q3x::kernels::sm87_target_aot_layer0_m192_oracle_detail {

inline constexpr std::size_t kTokenCount = 192U;
inline constexpr std::size_t kKillTestTokenCount = 40'000U;
inline constexpr std::size_t kFullTokenCount = 128U;
inline constexpr std::size_t kTailTokenCount = 64U;
inline constexpr int kThreads = 256;
inline constexpr std::size_t kDynamicSharedBytes = 76'800U;
inline constexpr int kPhysicalCtas = 16;

struct KernelResources final {
  Sm87TargetAotProjectionRole role =
      Sm87TargetAotProjectionRole::kInvalid;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int physical_ctas = 0;
  int device_ordinal = -1;
  int device_major = 0;
  int device_minor = 0;
  int device_sm_count = 0;
};

[[nodiscard]] int query_resources(Sm87TargetAotProjectionRole role,
                                  KernelResources* resources) noexcept;

// GateUp publishes exact BF16 SiLU(Gate_bf16)*Up_bf16. Down reads a separate
// immutable residual fixture and publishes the exact BF16
// Down_bf16+residual_bf16 value. Separate output ownership lets the oracle
// poison and independently retain baseline/candidate/replay publications.
[[nodiscard]] int launch(
    Sm87TargetAotProjectionRole role, const std::uint16_t* input,
    const Sm87TargetAotNvFp4CudaAssetView& authenticated_asset,
    const std::uint16_t* residual, std::uint16_t* output,
    void* cuda_stream) noexcept;

// Optional exact-P40000 quick-kill launch. It is source-private, accepts the
// same authenticated real-checkpoint assets as launch(), and never changes
// the fail-closed public launcher.
[[nodiscard]] int launch_kill_test(
    Sm87TargetAotProjectionRole role, const std::uint16_t* input,
    const Sm87TargetAotNvFp4CudaAssetView& authenticated_asset,
    const std::uint16_t* residual, std::uint16_t* output,
    void* cuda_stream) noexcept;

// Sets nonzero_flag atomically if any BF16 value is numerically non-zero.
// Both +0 and -0 are accepted as zero. The caller owns initialization and
// readback of the device flag.
[[nodiscard]] int launch_all_zero_check(
    const std::uint16_t* values, std::size_t element_count,
    std::uint32_t* nonzero_flag, void* cuda_stream) noexcept;

}  // namespace q3x::kernels::sm87_target_aot_layer0_m192_oracle_detail
