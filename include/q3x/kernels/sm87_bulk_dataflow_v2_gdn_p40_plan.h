#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_c64.h"

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Host-only execution contract for composing the exact C64 cell across P40.
// It grants no launcher or selector authority. CUDA stream order advances the
// convolution history on the producer stream and the authoritative BF16 state
// on the recurrence stream. Events exist only on cross-stream or slot-reuse
// edges; no event is presented as a numerical state boundary.
inline constexpr std::size_t kSm87BulkV2GdnP40Tokens = 40'000U;
inline constexpr std::size_t kSm87BulkV2GdnP40Chunks =
    kSm87BulkV2GdnP40Tokens / kSm87BulkV2GdnC64Tokens;
inline constexpr std::size_t kSm87BulkV2GdnP40SlotCount = 2U;
inline constexpr std::size_t kSm87BulkV2GdnP40StreamCount = 3U;
inline constexpr std::size_t kSm87BulkV2GdnP40EventsPerSlot = 3U;
inline constexpr std::size_t kSm87BulkV2GdnP40EventCount =
    kSm87BulkV2GdnP40SlotCount * kSm87BulkV2GdnP40EventsPerSlot;

inline constexpr std::uint64_t kSm87BulkV2GdnProducerWorkspaceBytes =
    kSm87BulkV2GdnNormalizedQBytes + kSm87BulkV2GdnNormalizedKBytes +
    kSm87BulkV2GdnPreparedVBytes + kSm87BulkV2GdnAlphaBytes +
    kSm87BulkV2GdnBetaBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnP40PreparedSlotsBytes =
    kSm87BulkV2GdnP40SlotCount * kSm87BulkV2GdnProducerWorkspaceBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnP40RawOutputSlotsBytes =
    kSm87BulkV2GdnP40SlotCount * kSm87BulkV2GdnRawOutputBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnP40HistorySlotsBytes =
    kSm87BulkV2GdnP40SlotCount * kSm87BulkV2GdnConvHistoryBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnP40PrivateBytes =
    kSm87BulkV2GdnP40PreparedSlotsBytes +
    kSm87BulkV2GdnP40RawOutputSlotsBytes +
    kSm87BulkV2GdnP40HistorySlotsBytes + kSm87BulkV2GdnStateBytes;

enum class Sm87BulkV2GdnStream : std::uint8_t {
  kInvalid = 0U,
  kProducer,
  kRecurrence,
  kEpilogue,
};

enum Sm87BulkV2GdnP40Policy : std::uint64_t {
  kSm87BulkV2GdnP40TwoPreparedSlots = 1ULL << 0U,
  kSm87BulkV2GdnP40TwoRawOutputSlots = 1ULL << 1U,
  kSm87BulkV2GdnP40TwoHistorySlots = 1ULL << 2U,
  kSm87BulkV2GdnP40ThreeLogicalStreams = 1ULL << 3U,
  kSm87BulkV2GdnP40StateOrderedOnRecurrenceStream = 1ULL << 4U,
  kSm87BulkV2GdnP40HistoryOrderedOnProducerStream = 1ULL << 5U,
  kSm87BulkV2GdnP40NoCrossCtaBarrier = 1ULL << 6U,
  kSm87BulkV2GdnP40NoPartialStateCommit = 1ULL << 7U,
  kSm87BulkV2GdnP40ChunkCancellationSafePoint = 1ULL << 8U,
  kSm87BulkV2GdnP40NoProductionSelector = 1ULL << 9U,
};

inline constexpr std::uint64_t kSm87BulkV2GdnP40RequiredPolicy =
    kSm87BulkV2GdnP40TwoPreparedSlots |
    kSm87BulkV2GdnP40TwoRawOutputSlots |
    kSm87BulkV2GdnP40TwoHistorySlots |
    kSm87BulkV2GdnP40ThreeLogicalStreams |
    kSm87BulkV2GdnP40StateOrderedOnRecurrenceStream |
    kSm87BulkV2GdnP40HistoryOrderedOnProducerStream |
    kSm87BulkV2GdnP40NoCrossCtaBarrier |
    kSm87BulkV2GdnP40NoPartialStateCommit |
    kSm87BulkV2GdnP40ChunkCancellationSafePoint |
    kSm87BulkV2GdnP40NoProductionSelector;

struct Sm87BulkV2GdnP40Plan final {
  std::size_t prompt_tokens = 0U;
  std::size_t chunk_tokens = 0U;
  std::size_t chunk_count = 0U;
  std::size_t tail_tokens = 0U;
  std::size_t prepared_slots = 0U;
  std::size_t raw_output_slots = 0U;
  std::size_t history_slots = 0U;
  std::size_t logical_streams = 0U;
  std::size_t reusable_events = 0U;
  std::uint64_t private_bytes = 0U;
  std::uint64_t policy = 0U;
  bool default_off = false;
  bool production_dispatch_eligible = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return prompt_tokens == kSm87BulkV2GdnP40Tokens &&
           chunk_tokens == kSm87BulkV2GdnC64Tokens &&
           chunk_count == kSm87BulkV2GdnP40Chunks && tail_tokens == 0U &&
           prepared_slots == kSm87BulkV2GdnP40SlotCount &&
           raw_output_slots == kSm87BulkV2GdnP40SlotCount &&
           history_slots == kSm87BulkV2GdnP40SlotCount &&
           logical_streams == kSm87BulkV2GdnP40StreamCount &&
           reusable_events == kSm87BulkV2GdnP40EventCount &&
           private_bytes == kSm87BulkV2GdnP40PrivateBytes &&
           policy == kSm87BulkV2GdnP40RequiredPolicy && default_off &&
           !production_dispatch_eligible;
  }
};

[[nodiscard]] constexpr Sm87BulkV2GdnP40Plan
sm87_bulk_v2_gdn_p40_plan() noexcept {
  return {kSm87BulkV2GdnP40Tokens,
          kSm87BulkV2GdnC64Tokens,
          kSm87BulkV2GdnP40Chunks,
          0U,
          kSm87BulkV2GdnP40SlotCount,
          kSm87BulkV2GdnP40SlotCount,
          kSm87BulkV2GdnP40SlotCount,
          kSm87BulkV2GdnP40StreamCount,
          kSm87BulkV2GdnP40EventCount,
          kSm87BulkV2GdnP40PrivateBytes,
          kSm87BulkV2GdnP40RequiredPolicy,
          true,
          false};
}

struct Sm87BulkV2GdnP40Chunk final {
  std::size_t index = 0U;
  std::size_t first_position = 0U;
  std::size_t token_count = 0U;
  std::size_t prepared_slot = 0U;
  std::size_t raw_output_slot = 0U;
  std::size_t output_history_slot = 0U;
  std::size_t input_history_slot = 0U;
  bool input_history_is_request_initial = false;
  bool producer_waits_for_recurrence_slot = false;
  bool recurrence_waits_for_prepared = false;
  bool recurrence_waits_for_epilogue_slot = false;
  bool epilogue_waits_for_recurrence = false;
  bool cancellation_safe_after_epilogue = false;

  [[nodiscard]] constexpr bool valid(
      const Sm87BulkV2GdnP40Plan& plan) const noexcept {
    return plan.valid() && index < plan.chunk_count &&
           first_position == index * plan.chunk_tokens &&
           token_count == plan.chunk_tokens &&
           prepared_slot == index % plan.prepared_slots &&
           raw_output_slot == index % plan.raw_output_slots &&
           output_history_slot == index % plan.history_slots &&
           input_history_slot ==
               (index == 0U ? 0U : (index - 1U) % plan.history_slots) &&
           input_history_is_request_initial == (index == 0U) &&
           producer_waits_for_recurrence_slot == (index >= 2U) &&
           recurrence_waits_for_prepared &&
           recurrence_waits_for_epilogue_slot == (index >= 2U) &&
           epilogue_waits_for_recurrence &&
           cancellation_safe_after_epilogue;
  }
};

[[nodiscard]] constexpr Sm87BulkV2GdnP40Chunk
sm87_bulk_v2_gdn_p40_chunk(const Sm87BulkV2GdnP40Plan& plan,
                           const std::size_t index) noexcept {
  if (!plan.valid() || index >= plan.chunk_count) {
    return {};
  }
  const std::size_t slot = index % plan.prepared_slots;
  return {index,
          index * plan.chunk_tokens,
          plan.chunk_tokens,
          slot,
          slot,
          slot,
          index == 0U ? 0U : (index - 1U) % plan.history_slots,
          index == 0U,
          index >= 2U,
          true,
          index >= 2U,
          true,
          true};
}

static_assert(kSm87BulkV2GdnP40Chunks == 625U);
static_assert(kSm87BulkV2GdnProducerWorkspaceBytes == 1'859'584U);
static_assert(kSm87BulkV2GdnP40PreparedSlotsBytes == 3'719'168U);
static_assert(kSm87BulkV2GdnP40RawOutputSlotsBytes == 1'572'864U);
static_assert(kSm87BulkV2GdnP40HistorySlotsBytes == 122'880U);
static_assert(kSm87BulkV2GdnP40PrivateBytes == 6'987'776U);
static_assert(kSm87BulkV2GdnP40RequiredPolicy == 0x3ffULL);
static_assert(sm87_bulk_v2_gdn_p40_plan().valid());

}  // namespace q3x::kernels
