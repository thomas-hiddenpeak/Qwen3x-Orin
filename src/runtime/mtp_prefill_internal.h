#pragma once
#include <cstdint>
namespace q3x::runtime::mtp_detail {
int launch_mtp_draft_confidence(const std::uint16_t*, unsigned, float*, void*) noexcept;
inline constexpr unsigned kDraftPrefillBatch = 32;
int launch_mtp_prefill_projection(const std::uint16_t*, const std::uint16_t*,
    unsigned, unsigned, unsigned, std::uint16_t*, void*) noexcept;
}  // namespace q3x::runtime::mtp_detail
