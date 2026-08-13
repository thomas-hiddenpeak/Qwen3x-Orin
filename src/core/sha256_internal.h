#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::core::internal {

inline constexpr std::array<std::uint32_t, 8> kSha256InitialState = {
    0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU,
    0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U,
};

inline constexpr std::array<std::uint32_t, 64> kSha256RoundConstants = {
    0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
    0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U,
    0x243185beU, 0x550c7dc3U, 0x72be5d74U, 0x80deb1feU, 0x9bdc06a7U,
    0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU,
    0x2de92c6fU, 0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U,
    0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U, 0xd5a79147U,
    0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU,
    0x53380d13U, 0x650a7354U, 0x766a0abbU, 0x81c2c92eU, 0x92722c85U,
    0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
    0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U,
    0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U, 0x4ed8aa4aU, 0x5b9cca4fU,
    0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U,
    0x90befffaU, 0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U,
};

// Test/benchmark-visible implementation seams. They are deliberately kept in
// the source tree rather than the installed public API.
void sha256_transform_blocks_portable(
    std::uint32_t* state, const std::uint8_t* blocks,
    std::size_t block_count) noexcept;

[[nodiscard]] bool sha256_arm_sha2_compiled() noexcept;
[[nodiscard]] bool sha256_arm_sha2_available() noexcept;

// Returns false without touching state when the accelerated implementation is
// not compiled or the running CPU does not advertise HWCAP_SHA2. state and
// blocks must be non-null when block_count is nonzero.
[[nodiscard]] bool sha256_transform_blocks_arm_sha2(
    std::uint32_t* state, const std::uint8_t* blocks,
    std::size_t block_count) noexcept;

#if defined(Q3X_SHA256_ARM_SHA2_COMPILED)
// This entry point contains ARMv8 SHA2 instructions and must only be called
// after sha256_arm_sha2_available() succeeds.
void sha256_transform_blocks_arm_sha2_unchecked(
    std::uint32_t* state, const std::uint8_t* blocks,
    std::size_t block_count) noexcept;
#endif

}  // namespace q3x::core::internal
