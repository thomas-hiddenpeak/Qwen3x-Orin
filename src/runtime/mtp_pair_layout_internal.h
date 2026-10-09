#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime::mtp_detail {

// Private immutable representation; never attach it as a public quad sidecar.
struct NvPairLayout {
  static constexpr std::size_t projection_bytes = 50'135'040;
  static constexpr std::size_t arena_bytes = 64 * 3 * projection_bytes;
  static constexpr const char* identity = "nvfp4-output-pair-k256-v1";
};

// Stream ordered, allocation free. The caller owns disjoint complete spans and
// initializes mismatch to zero before checking. Nonzero mismatch forbids use.
int launch_mtp_nv_pair_pack(const std::uint8_t* weights,
    const std::uint8_t* scales, unsigned n, unsigned k,
    std::uint8_t* packed, void* stream) noexcept;
int launch_mtp_nv_pair_check(const std::uint8_t* weights,
    const std::uint8_t* scales, unsigned n, unsigned k,
    const std::uint8_t* packed, unsigned* mismatch, void* stream) noexcept;

}  // namespace q3x::runtime::mtp_detail
