#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-admission-only Qwen3.6 linear-attention A/B projection. The two
// canonical BF16 [48, 5120] row-major matrices are treated as one logical
// N96 matrix, while the public outputs retain their independent token-major
// [M, 48] layouts. M is accepted in [2, 512]. Complete M64 spans use the
// SM87 BF16 Tensor Core pipeline; a final M1..M63 span preserves the existing
// M16/generic BF16 projection pair boundary. No persistent weight transform,
// scratch allocation, cuBLAS, or cuBLASLt dependency is introduced.
[[nodiscard]] int launch_sm87_bf16_ab_large_m_prefill_cuda(
    const std::uint16_t* first_weights,
    const std::uint16_t* second_weights,
    const std::uint16_t* input, std::size_t token_count,
    std::uint16_t* first_output, std::uint16_t* second_output,
    void* cuda_stream = nullptr) noexcept;

// Resource probe for the M64xN96xK64 Tensor Core kernel. The occupancy result
// includes the launcher's two-stage dynamic shared-memory footprint.
[[nodiscard]] int query_sm87_bf16_ab_large_m_prefill_resources_cuda(
    int* registers_per_thread, std::size_t* static_shared_bytes,
    std::size_t* dynamic_shared_bytes, std::size_t* local_bytes,
    int* active_blocks_per_sm) noexcept;

}  // namespace q3x::kernels
