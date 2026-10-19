#pragma once
#include <cstddef>
#include <cstdint>
namespace q3x::runtime::mtp_detail {
// Private draft only. Two INT8 components plus two FP32 scales per group128.
constexpr std::size_t mtp_compensated_bytes(unsigned k) noexcept {
  return 2ULL * k + 8ULL * (k / 128);
}
int launch_mtp_compensated_quantize(const std::uint16_t* input, unsigned k,
    void* scratch, std::size_t bytes, void* stream) noexcept;
int launch_mtp_compensated_project(const std::uint8_t* weights,
    const std::uint16_t* scales, unsigned n, unsigned k,
    const void* scratch, std::size_t bytes, std::uint16_t* output,
    void* stream) noexcept;
}
