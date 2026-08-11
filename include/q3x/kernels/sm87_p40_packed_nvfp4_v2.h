#pragma once

#include "q3x/kernels/sm87_p40_packed_projection.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Execution-only ABI for the default-off packed NVFP4 v2 candidate.  The
// authenticated v1 payload remains unchanged; this ABI gives that payload a
// materially different M128, shape-specific consumer without relabelling the
// historical v13 route.
enum class Sm87P40PackedNvFp4V2Tactic : std::uint8_t {
  kGateUpM128N256K128GroupedM2 = 0U,
  kDownM128N256K128AMajor,
  kInvalid = 0xffU,
};

inline constexpr std::uint32_t kSm87P40PackedNvFp4V2TileM = 128U;
inline constexpr std::uint32_t kSm87P40PackedNvFp4V2TileK = 128U;
inline constexpr std::uint32_t kSm87P40PackedNvFp4V2Threads = 256U;
inline constexpr std::uint32_t kSm87P40PackedNvFp4V2Warps = 8U;
inline constexpr std::uint32_t kSm87P40PackedNvFp4V2PipelineStages = 3U;
inline constexpr std::uint32_t kSm87P40PackedNvFp4V2GridM = 313U;
inline constexpr std::uint32_t kSm87P40PackedNvFp4V2GateGridN = 136U;
inline constexpr std::uint32_t kSm87P40PackedNvFp4V2DownGridN = 20U;
inline constexpr std::size_t kSm87P40PackedNvFp4V2DynamicSharedBytes =
    153'600U;

int query_sm87_p40_packed_nvfp4_v2_resources_cuda(
    Sm87P40PackedProjectionRole role,
    Sm87P40PackedProjectionResources* resources) noexcept;

// The resource query is the binding-time admission point for the production
// default-off runtime candidate launchers below. A launcher fails closed until
// its role has passed that
// query; request-time launches do not repeat function-attribute or occupancy
// discovery.
int launch_sm87_p40_packed_nvfp4_v2_gate_up_cuda(
    const std::uint16_t* input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    std::size_t token_count, std::uint16_t* activated_output,
    void* cuda_stream = nullptr) noexcept;

int launch_sm87_p40_packed_nvfp4_v2_down_cuda(
    const std::uint16_t* input,
    const Sm87P40PackedProjectionDeviceView& artifact,
    std::size_t token_count, std::uint16_t* residual_in_out,
    void* cuda_stream = nullptr) noexcept;

// T1-only launchers.  They instantiate the production M128/K128 mainloop on
// two K64 cells (and two Down N128 cells) without allocating a P40 matrix.
// They have correctness/resource authority only and are never called by the
// runtime route.
int launch_sm87_p40_packed_nvfp4_v2_gate_up_tile_test_cuda(
    const std::uint16_t* input, const std::uint8_t* compact_payload,
    float gate_scale, float up_scale, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

int launch_sm87_p40_packed_nvfp4_v2_down_tile_test_cuda(
    const std::uint16_t* input, const std::uint8_t* compact_payload,
    float global_scale, std::uint16_t* residual_in_out,
    void* cuda_stream = nullptr) noexcept;

// T1-only full-K launchers.  They use one M128 output tile but instantiate the
// exact production Gate K=5120 and Down K=17408 mainloops. token_limit is
// deliberately restricted to 128 (full tile) or 64 (the P40000 tail shape),
// so the latter also exercises predicated cp.async zero-fill and epilogue
// masking without allocating a P40000 activation matrix.
int launch_sm87_p40_packed_nvfp4_v2_gate_up_full_k_test_cuda(
    const std::uint16_t* input, const std::uint8_t* compact_payload,
    std::size_t token_limit, float gate_scale, float up_scale,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

int launch_sm87_p40_packed_nvfp4_v2_down_full_k_test_cuda(
    const std::uint16_t* input, const std::uint8_t* compact_payload,
    std::size_t token_limit, float global_scale,
    std::uint16_t* residual_in_out,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
