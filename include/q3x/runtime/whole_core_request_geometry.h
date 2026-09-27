#pragma once
#include <cstdint>

namespace q3x::runtime {
// Separate compiled composition geometry. Fixed production/dev profiles keep
// their original capacity; no existing qualification transfers to admission.
#if defined(Q3X_ENABLE_WHOLE_CORE_SERVICE)
inline constexpr std::uint32_t kWholeCoreCompiledSequenceCapacity = 44'095U;
#else
inline constexpr std::uint32_t kWholeCoreCompiledSequenceCapacity = 40'016U;
#endif
// Storage rounds the served 44095 positions to C64 for aligned typed regions.
// The spare row is capacity only: execution always uses actual prompt length.
inline constexpr std::uint32_t kWholeCoreCompiledPromptStorageTokens =
    kWholeCoreCompiledSequenceCapacity == 44'095U ? 44'096U : 40'000U;
inline constexpr std::uint32_t kWholeCoreCompiledPanelCapacity =
    (kWholeCoreCompiledPromptStorageTokens + 7999U) / 8000U;
inline constexpr std::uint64_t kWholeCoreCompiledConvOffset =
    20480ULL * kWholeCoreCompiledPromptStorageTokens;
inline constexpr std::uint64_t kWholeCoreCompiledZOffset =
    2ULL * kWholeCoreCompiledConvOffset;
inline constexpr std::uint64_t kWholeCoreCompiledAOffset =
    kWholeCoreCompiledZOffset + 12288ULL * kWholeCoreCompiledPromptStorageTokens;
inline constexpr std::uint64_t kWholeCoreCompiledBOffset =
    kWholeCoreCompiledAOffset + 96ULL * kWholeCoreCompiledPromptStorageTokens;
inline constexpr std::uint64_t kWholeCoreCompiledGdnOffset =
    kWholeCoreCompiledBOffset + 96ULL * kWholeCoreCompiledPromptStorageTokens;
inline constexpr std::uint64_t kWholeCoreCompiledGdnBytes =
    70016ULL * kWholeCoreCompiledPromptStorageTokens;
inline constexpr std::uint64_t kWholeCoreCompiledOutputOffset =
    kWholeCoreCompiledGdnOffset + kWholeCoreCompiledGdnBytes;
inline constexpr std::uint64_t kWholeCoreCompiledFamilyBytes =
    kWholeCoreCompiledOutputOffset + 12288ULL * kWholeCoreCompiledPromptStorageTokens;
inline constexpr std::uint64_t kWholeCoreCompiledProcessedQOffset =
    24576ULL * kWholeCoreCompiledPromptStorageTokens;
inline constexpr std::uint64_t kWholeCoreCompiledGateOffset =
    36864ULL * kWholeCoreCompiledPromptStorageTokens;
inline constexpr std::uint64_t kWholeCoreCompiledAttentionBranchOffset =
    12288ULL * kWholeCoreCompiledPromptStorageTokens;
// Actual prompt rows fit the current family arena. Down isolates a partial
// final tile in request scratch; padding never advances model state.
[[nodiscard]] constexpr bool whole_core_prompt_tokens_admitted(std::uint64_t tokens) noexcept {
#if defined(Q3X_ENABLE_WHOLE_CORE_SERVICE)
  return tokens >= 1U && tokens <= kWholeCoreCompiledSequenceCapacity;
#else
  return tokens == 40'000U;
#endif
}
// Projection-entry count: one Gate/Up entry plus one or two Down grids.
[[nodiscard]] constexpr std::uint64_t whole_core_mlp_launch_count(std::uint64_t rows) noexcept {
#if defined(Q3X_ENABLE_WHOLE_CORE_SERVICE)
  return 1U + (rows >= 64U ? 1U : 0U) + (rows % 64U ? 1U : 0U);
#else
  (void)rows;
  return 2U;
#endif
}
inline constexpr std::uint64_t kWholeCoreCompiledPersistentBytes =
    78'446'592ULL + 65'536ULL * kWholeCoreCompiledSequenceCapacity;
inline constexpr std::uint64_t kWholeCoreCompiledResidualBytes =
    10'240ULL * kWholeCoreCompiledSequenceCapacity;
// The C512 vectors occupy 87,130,112 bytes. FP32 attention scratch is rounded
// to the owning arena's 256-byte alignment; its logical capacity stays 24*S.
inline constexpr std::uint64_t kWholeCoreCompiledLegacyBytes =
    87'130'112ULL +
    ((96ULL * kWholeCoreCompiledSequenceCapacity + 255ULL) / 256ULL) * 256ULL;
// Cos/sin are contiguous halves of one aligned allocation.
inline constexpr std::uint64_t kWholeCoreCompiledRopeBytes =
    256ULL * kWholeCoreCompiledSequenceCapacity;
inline constexpr std::uint64_t kWholeCoreCompiledArenaBytes =
    kWholeCoreCompiledPersistentBytes + kWholeCoreCompiledResidualBytes +
    kWholeCoreCompiledFamilyBytes + kWholeCoreCompiledLegacyBytes + 10'240ULL +
    kWholeCoreCompiledRopeBytes;
static_assert(kWholeCoreCompiledArenaBytes ==
    (kWholeCoreCompiledSequenceCapacity == 40'016U
         ? 8'641'684'992ULL : 9'508'218'624ULL));
}  // namespace q3x::runtime
