#pragma once
#include <cstdint>

namespace q3x::runtime {
// Separate compiled composition geometry. Fixed production/dev profiles keep
// their original capacity; no existing qualification transfers to admission.
#if defined(Q3X_ENABLE_WHOLE_CORE_EXACT_DECODE_ADMISSION)
inline constexpr std::uint32_t kWholeCoreCompiledSequenceCapacity = 44'095U;
#else
inline constexpr std::uint32_t kWholeCoreCompiledSequenceCapacity = 40'016U;
#endif
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
    5'429'760'000ULL + kWholeCoreCompiledLegacyBytes + 10'240ULL +
    kWholeCoreCompiledRopeBytes;
static_assert(kWholeCoreCompiledArenaBytes ==
    (kWholeCoreCompiledSequenceCapacity == 40'016U
         ? 8'641'684'992ULL : 8'952'211'200ULL));
}  // namespace q3x::runtime
