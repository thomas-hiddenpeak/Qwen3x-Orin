#pragma once

#include "sm87_target_aot_p40_executor_internal.h"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace q3x::runtime::sm87_target_aot_p40_executor_detail {

inline constexpr std::size_t kSm87TargetAotEngineRopePositions = 262'144U;
inline constexpr std::size_t kSm87TargetAotEngineRopePairs = 32U;
inline constexpr std::uint64_t kSm87TargetAotEngineRopeTableBytes =
    static_cast<std::uint64_t>(kSm87TargetAotEngineRopePositions) *
    kSm87TargetAotEngineRopePairs * sizeof(float);
inline constexpr std::uint64_t kSm87TargetAotEngineRopeAllocationBytes =
    2U * kSm87TargetAotEngineRopeTableBytes;

enum class Sm87TargetAotEngineRopeError : std::uint8_t {
  kNone = 0U,
  kAdmissionDisabled,
  kInvalidArgument,
  kInsufficientDeviceMemory,
  kHostAllocationFailure,
  kCudaFailure,
  kIdentityFailure,
};

struct Sm87TargetAotEngineRopeStatus final {
  Sm87TargetAotEngineRopeError code = Sm87TargetAotEngineRopeError::kNone;
  const char* context = "none";
  int cuda_error = 0;

  [[nodiscard]] explicit operator bool() const noexcept {
    return code == Sm87TargetAotEngineRopeError::kNone;
  }
};

struct Sm87TargetAotEngineRopeCreateResult;

// Engine-lifetime, source-private owner for the complete pinned Qwen3.6-model
// RoPE table.
// It materializes the same theta=1e7, rotary_dim=64, BF16-rounded-in-FP32
// values as RequestState, but owns an independent contiguous device
// allocation so the exact-P40000 route never borrows a legacy arena.
class Sm87TargetAotEngineRopeOwner final {
 public:
  Sm87TargetAotEngineRopeOwner(
      const Sm87TargetAotEngineRopeOwner&) = delete;
  Sm87TargetAotEngineRopeOwner& operator=(
      const Sm87TargetAotEngineRopeOwner&) = delete;
  Sm87TargetAotEngineRopeOwner(
      Sm87TargetAotEngineRopeOwner&&) = delete;
  Sm87TargetAotEngineRopeOwner& operator=(
      Sm87TargetAotEngineRopeOwner&&) = delete;
  ~Sm87TargetAotEngineRopeOwner();

  [[nodiscard]] Sm87TargetAotP40EngineRope view() const noexcept;
  [[nodiscard]] std::uint64_t bytes() const noexcept {
    return allocation_ == nullptr ? 0U
                                  : kSm87TargetAotEngineRopeAllocationBytes;
  }
  [[nodiscard]] std::int32_t device_ordinal() const noexcept {
    return device_ordinal_;
  }

 private:
  friend struct Sm87TargetAotEngineRopeCreateResult;
  friend Sm87TargetAotEngineRopeCreateResult
  create_sm87_target_aot_engine_rope(
      std::uint64_t minimum_free_bytes_after_create) noexcept;
  Sm87TargetAotEngineRopeOwner(float* allocation,
                              const std::uint64_t identity,
                              const std::int32_t device_ordinal) noexcept
      : allocation_(allocation),
        identity_(identity),
        device_ordinal_(device_ordinal) {}

  float* allocation_ = nullptr;
  std::uint64_t identity_ = 0U;
  std::int32_t device_ordinal_ = -1;
};

struct Sm87TargetAotEngineRopeCreateResult final {
  std::unique_ptr<Sm87TargetAotEngineRopeOwner> owner;
  Sm87TargetAotEngineRopeStatus status{};

  [[nodiscard]] explicit operator bool() const noexcept {
    return owner != nullptr && static_cast<bool>(status);
  }
};

[[nodiscard]] Sm87TargetAotEngineRopeCreateResult
create_sm87_target_aot_engine_rope(
    std::uint64_t minimum_free_bytes_after_create) noexcept;

[[nodiscard]] const char* to_string(
    Sm87TargetAotEngineRopeError error) noexcept;

static_assert(kSm87TargetAotEngineRopeAllocationBytes == 67'108'864ULL);
static_assert(kSm87TargetAotEngineRopePositions >=
              kSm87TargetAotP40PromptTokens);

}  // namespace q3x::runtime::sm87_target_aot_p40_executor_detail
