#pragma once

#include <array>
#include <cstdint>

// Private, host-only admission contract. Not part of the installed engine ABI.
namespace q3x::runtime::mtp_detail {

inline constexpr std::uint32_t kMaximumDraftLength = 3;
inline constexpr std::uint32_t kMaximumVerifyRows = 4;

struct RoundOptions {
  std::uint32_t draft_length = 2;  // Configured length: exactly 2 or 3.
  std::uint32_t seed_token = 0;    // Already emitted, not yet in target state.
  std::uint32_t remaining_output = 0;  // Excludes seed_token.
  std::uint32_t available_target_rows = 0;
  std::uint32_t vocabulary_size = 0;
  std::uint32_t stop_token = 0;
};

enum class RoundStatus : std::uint8_t {
  kContinue,
  kLength,
  kStop,
  kCancelled,
  kInvalidOptions,
  kBackendFailure,
  kInvalidPrediction,
};

struct RoundResult {
  RoundStatus status = RoundStatus::kInvalidOptions;
  std::array<std::uint32_t, kMaximumVerifyRows> output{};
  std::uint32_t output_count = 0;
  std::uint32_t proposed_tokens = 0;
  std::uint32_t accepted_tokens = 0;
  std::uint32_t verified_rows = 0;
  std::uint32_t committed_target_rows = 0;
  bool abort_succeeded = true;

  [[nodiscard]] bool ok() const noexcept {
    return status == RoundStatus::kContinue || status == RoundStatus::kLength ||
           status == RoundStatus::kStop || status == RoundStatus::kCancelled;
  }
};

// begin owns a transaction even if it fails. Every failure calls abort, which
// drains pending work and poisons uncertain state; it must not authorize reuse.
// verify stages target transitions for [seed, draft[0], ..., draft[n-1]],
// returning the target argmax after each row (all logits must be finite).
// No speculative token may reach a client. commit_prefix synchronously selects
// the exact prefix's KV, Conv, GDN and hidden state, reconciles draft state with
// target hidden rows, and invalidates rejected tails before returning true.
// The last emitted token is pending, NOT yet consumed by the target.
// finish retires the transaction; all device resources were planned at startup.
class RoundBackend {
 public:
  virtual ~RoundBackend() = default;
  virtual bool begin(std::uint32_t verify_rows) noexcept = 0;
  virtual bool propose(std::uint32_t seed, std::uint32_t count,
                       std::uint32_t* draft) noexcept = 0;
  virtual bool verify(std::uint32_t seed, const std::uint32_t* draft,
                      std::uint32_t count,
                      std::uint32_t* predictions) noexcept = 0;
  virtual bool commit_prefix(std::uint32_t target_rows,
                             std::uint32_t pending_token) noexcept = 0;
  virtual bool finish() noexcept = 0;
  virtual bool abort() noexcept = 0;
};

// Called only AFTER the matching state commit. False cancels at that token.
using TokenObserver = bool (*)(void*, std::uint32_t token) noexcept;

[[nodiscard]] RoundResult run_round(
    const RoundOptions&, RoundBackend&, TokenObserver = nullptr,
    void* observer_context = nullptr) noexcept;

}  // namespace q3x::runtime::mtp_detail
