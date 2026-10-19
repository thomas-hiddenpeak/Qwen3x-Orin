#pragma once
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::mtp_detail {
// Private draft arithmetic only. Never use for target verification.
inline constexpr std::size_t kIntegerDraftHeadScratchBytes = 6400;
int launch_mtp_integer_draft_head(const std::uint8_t* weights,
    const std::uint8_t* scales, float tensor_scale, const std::uint16_t* input,
    unsigned outputs, unsigned inputs, void* scratch, std::size_t scratch_bytes,
    std::uint16_t* output, void* stream) noexcept;
}
