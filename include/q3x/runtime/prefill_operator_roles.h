#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::runtime {

// Candidate-neutral logical Prefill boundary. These identities describe the
// publications that a complete Prefill system must provide; they do not name
// an execution plan, kernel, launcher, device resource, or historical route.
enum class PrefillBindingRole : std::uint8_t {
  kNvfp4GateUp = 0,
  kNvfp4Down,
  kLinearFp8Qkv,
  kLinearFp8Z,
  kLinearFp8O,
  kFullFp8Q,
  kFullFp8K,
  kFullFp8V,
  kFullFp8O,
  kLinearBf16A,
  kLinearBf16B,
  kExactGdn,
  kExactCausalAttention,
  kResidual,
  kNormalization,
  kEmbedding,
  kFinalHandoff,
  kCount,
  kInvalid = 0xffU,
};

inline constexpr std::size_t kPrefillRequiredOperatorRoleCount =
    static_cast<std::size_t>(PrefillBindingRole::kCount);
static_assert(kPrefillRequiredOperatorRoleCount == 17U);

// kExact declares the intended numerical contract. It is not evidence that
// a candidate has passed finite-precision or production-accuracy qualification.
enum class PrefillNumericalMode : std::uint8_t {
  kUnbound = 0,
  kExact,
  kApproximate,
};

enum class PrefillOperatorProvider : std::uint8_t {
  kUnbound = 0,
  kNative,
  kCuBlasLtReference,
  kExternalRuntime,
};

enum class PrefillTacticMode : std::uint8_t {
  kUnbound = 0,
  kAheadOfTime,
  kJit,
  kRequestTimeSelected,
};

}  // namespace q3x::runtime
