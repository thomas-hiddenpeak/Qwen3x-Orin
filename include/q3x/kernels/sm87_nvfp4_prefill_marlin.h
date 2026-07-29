#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-only exact-C512 prefill architecture admission for one
// [N=17408,K=5120] NVFP4 projection.  Packing keeps the E2M1 payload and
// E4M3 block scales byte-exact; each execution CTA runtime-decodes the 256
// E4M3 codes once and fetches the selected BF16 scale into registers.  The
// sidecar is one contiguous, 16-byte-aligned allocation.
[[nodiscard]] std::size_t
sm87_nvfp4_prefill_marlin_sidecar_bytes_per_projection() noexcept;

[[nodiscard]] int launch_sm87_nvfp4_prefill_marlin_pack_cuda(
    const std::uint8_t* canonical_packed_weights,
    const std::uint8_t* canonical_block_scales,
    std::uint8_t* sidecar, std::size_t rows, std::size_t columns,
    void* cuda_stream = nullptr) noexcept;

// Computes Gate and Up in one logical N=34816 persistent work queue and
// publishes two independent row-major [512,17408] BF16 matrices.  A CTA owns
// exactly one branch/N256 work item at a time: the two B pipelines are never
// resident or coupled, and this kernel does not apply SiLU.  The exact
// admission cell is M64N256K64 with a four-stage global-to-shared ring, a
// two-slot shared-to-register pipeline, 256 threads, and a fixed 32-CTA grid.
[[nodiscard]] int launch_sm87_nvfp4_prefill_marlin_pair_cuda(
    const std::uint8_t* gate_sidecar, float gate_weight_scale_2,
    const std::uint8_t* up_sidecar, float up_weight_scale_2,
    const std::uint16_t* activations, std::size_t token_count,
    std::size_t rows, std::size_t columns, std::uint16_t* gate_output,
    std::uint16_t* up_output,
    void* cuda_stream = nullptr) noexcept;

// Reports the pair kernel after installing its required dynamic-shared
// memory attribute.  No kernel is launched.
[[nodiscard]] int query_sm87_nvfp4_prefill_marlin_pair_resources(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* maximum_threads_per_block, int* active_blocks_per_sm) noexcept;

}  // namespace q3x::kernels
