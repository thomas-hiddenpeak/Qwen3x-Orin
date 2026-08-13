#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::sm87_target_aot_attention_preprocess_execution_detail {

// Source-private exact-M8000 preprocessing seam for the target-AOT executor.
// It reaches the established prompt-wide CUDA body without opening the older
// public whole-core admission or consulting an environment selector.
[[nodiscard]] int launch_p8000(
    const std::uint16_t* interleaved_q_gate, std::uint16_t* key,
    const std::uint16_t* q_weight, const std::uint16_t* k_weight,
    float epsilon, std::uint16_t* query_output,
    std::uint16_t* gate_output, const float* cosines, const float* sines,
    std::size_t first_position, std::size_t token_count,
    void* cuda_stream) noexcept;

}  // namespace q3x::runtime::sm87_target_aot_attention_preprocess_execution_detail
