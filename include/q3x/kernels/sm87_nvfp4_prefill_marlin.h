#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-only exact-C512 architecture admission. The sidecar is a fragment-
// native permutation of one canonical [17408,5120] NVFP4 projection. Packed
// E2M1 values remain byte-exact; E4M3 block scales are expanded once to exact
// BF16 at engine creation. Neither entry participates in production dispatch.
[[nodiscard]] std::size_t
sm87_nvfp4_prefill_marlin_sidecar_bytes_per_projection() noexcept;

[[nodiscard]] int launch_sm87_nvfp4_prefill_marlin_pack_cuda(
    const std::uint8_t* canonical_packed_weights,
    const std::uint8_t* canonical_block_scales,
    std::uint8_t* sidecar, std::size_t rows, std::size_t columns,
    void* cuda_stream = nullptr) noexcept;

// Computes Gate and Up together over the same C512 BF16 activation matrix and
// publishes only BF16 SiLU(Gate)*Up. The kernel uses a fixed 32-CTA persistent
// grid.  Four warps own Gate and four own Up, so shared A survives both
// branches without doubling each thread's accumulator footprint.  A two-stage
// global-to-shared ring and two-slot shared-to-register pipeline keep the
// exact checkpoint cell eligible for two resident CTAs per SM.
[[nodiscard]] int launch_sm87_nvfp4_prefill_marlin_gate_up_silu_cuda(
    const std::uint8_t* gate_sidecar, float gate_weight_scale_2,
    const std::uint8_t* up_sidecar, float up_weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
