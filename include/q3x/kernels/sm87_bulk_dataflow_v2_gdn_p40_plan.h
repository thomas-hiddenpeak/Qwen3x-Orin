#pragma once

#include "q3x/kernels/sm87_bulk_dataflow_v2_gdn_c64.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

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
inline constexpr std::uint64_t kSm87BulkV2GdnP40CancellationSnapshotBytes =
    kSm87BulkV2GdnP40SlotCount * sizeof(std::uint32_t);
inline constexpr std::uint64_t kSm87BulkV2GdnP40PrivateRangeAlignment =
    kSm87BulkV2GdnPointerAlignment;
inline constexpr std::uint64_t kSm87BulkV2GdnP40PrivateExtentAlignment = 256U;

[[nodiscard]] constexpr std::uint64_t sm87_bulk_v2_gdn_p40_align_up(
    const std::uint64_t value, const std::uint64_t alignment) noexcept {
  return alignment == 0U ? value
                         : ((value + alignment - 1U) / alignment) * alignment;
}

// Frozen request-private suballocation. Producer workspaces are slot-major so
// each slot's Q/K/V/alpha/beta publications form one contiguous owner. The
// two four-byte cancellation snapshots receive independent 16-byte-aligned
// addresses; the complete allocation extent is rounded to 256 bytes.
inline constexpr std::array<std::uint64_t, kSm87BulkV2GdnP40SlotCount>
    kSm87BulkV2GdnP40NormalizedQOffsets{{
        0U,
        kSm87BulkV2GdnProducerWorkspaceBytes,
    }};
inline constexpr std::array<std::uint64_t, kSm87BulkV2GdnP40SlotCount>
    kSm87BulkV2GdnP40NormalizedKOffsets{{
        kSm87BulkV2GdnP40NormalizedQOffsets[0U] +
            kSm87BulkV2GdnNormalizedQBytes,
        kSm87BulkV2GdnP40NormalizedQOffsets[1U] +
            kSm87BulkV2GdnNormalizedQBytes,
    }};
inline constexpr std::array<std::uint64_t, kSm87BulkV2GdnP40SlotCount>
    kSm87BulkV2GdnP40PreparedVOffsets{{
        kSm87BulkV2GdnP40NormalizedKOffsets[0U] +
            kSm87BulkV2GdnNormalizedKBytes,
        kSm87BulkV2GdnP40NormalizedKOffsets[1U] +
            kSm87BulkV2GdnNormalizedKBytes,
    }};
inline constexpr std::array<std::uint64_t, kSm87BulkV2GdnP40SlotCount>
    kSm87BulkV2GdnP40AlphaOffsets{{
        kSm87BulkV2GdnP40PreparedVOffsets[0U] +
            kSm87BulkV2GdnPreparedVBytes,
        kSm87BulkV2GdnP40PreparedVOffsets[1U] +
            kSm87BulkV2GdnPreparedVBytes,
    }};
inline constexpr std::array<std::uint64_t, kSm87BulkV2GdnP40SlotCount>
    kSm87BulkV2GdnP40BetaOffsets{{
        kSm87BulkV2GdnP40AlphaOffsets[0U] + kSm87BulkV2GdnAlphaBytes,
        kSm87BulkV2GdnP40AlphaOffsets[1U] + kSm87BulkV2GdnAlphaBytes,
    }};
inline constexpr std::array<std::uint64_t, kSm87BulkV2GdnP40SlotCount>
    kSm87BulkV2GdnP40RawOutputOffsets{{
        kSm87BulkV2GdnP40PreparedSlotsBytes,
        kSm87BulkV2GdnP40PreparedSlotsBytes +
            kSm87BulkV2GdnRawOutputBytes,
    }};
inline constexpr std::array<std::uint64_t, kSm87BulkV2GdnP40SlotCount>
    kSm87BulkV2GdnP40ConvHistoryOffsets{{
        kSm87BulkV2GdnP40PreparedSlotsBytes +
            kSm87BulkV2GdnP40RawOutputSlotsBytes,
        kSm87BulkV2GdnP40PreparedSlotsBytes +
            kSm87BulkV2GdnP40RawOutputSlotsBytes +
            kSm87BulkV2GdnConvHistoryBytes,
    }};
inline constexpr std::uint64_t
    kSm87BulkV2GdnP40TransactionalRecurrentStateOffset =
        kSm87BulkV2GdnP40PreparedSlotsBytes +
        kSm87BulkV2GdnP40RawOutputSlotsBytes +
        kSm87BulkV2GdnP40HistorySlotsBytes;
inline constexpr std::array<std::uint64_t, kSm87BulkV2GdnP40SlotCount>
    kSm87BulkV2GdnP40CancellationSnapshotOffsets{{
        kSm87BulkV2GdnP40TransactionalRecurrentStateOffset +
            kSm87BulkV2GdnStateBytes,
        sm87_bulk_v2_gdn_p40_align_up(
            kSm87BulkV2GdnP40TransactionalRecurrentStateOffset +
                kSm87BulkV2GdnStateBytes + sizeof(std::uint32_t),
            kSm87BulkV2GdnP40PrivateRangeAlignment),
    }};
inline constexpr std::uint64_t kSm87BulkV2GdnP40PrivatePayloadBytes =
    kSm87BulkV2GdnP40CancellationSnapshotOffsets[1U] +
    sizeof(std::uint32_t);
inline constexpr std::uint64_t kSm87BulkV2GdnP40PrivateBytes =
    sm87_bulk_v2_gdn_p40_align_up(
        kSm87BulkV2GdnP40PrivatePayloadBytes,
        kSm87BulkV2GdnP40PrivateExtentAlignment);
inline constexpr std::size_t kSm87BulkV2GdnP40PrivateRangeCount = 17U;

struct Sm87BulkV2GdnP40PrivateRange final {
  std::uint64_t offset = 0U;
  std::uint64_t bytes = 0U;

  [[nodiscard]] constexpr std::uint64_t end() const noexcept {
    return offset + bytes;
  }

  [[nodiscard]] constexpr bool valid() const noexcept {
    return bytes != 0U &&
           offset % kSm87BulkV2GdnP40PrivateRangeAlignment == 0U &&
           offset <= std::numeric_limits<std::uint64_t>::max() - bytes &&
           end() <= kSm87BulkV2GdnP40PrivateBytes;
  }
};

// Physical order: slot0 Q/K/V/alpha/beta, slot1 Q/K/V/alpha/beta,
// raw-output slots, convolution-history slots, transactional state, then the
// two device-local cancellation snapshots.
[[nodiscard]] constexpr auto sm87_bulk_v2_gdn_p40_private_ranges() noexcept {
  return std::array<Sm87BulkV2GdnP40PrivateRange,
                    kSm87BulkV2GdnP40PrivateRangeCount>{{
      {kSm87BulkV2GdnP40NormalizedQOffsets[0U],
       kSm87BulkV2GdnNormalizedQBytes},
      {kSm87BulkV2GdnP40NormalizedKOffsets[0U],
       kSm87BulkV2GdnNormalizedKBytes},
      {kSm87BulkV2GdnP40PreparedVOffsets[0U],
       kSm87BulkV2GdnPreparedVBytes},
      {kSm87BulkV2GdnP40AlphaOffsets[0U], kSm87BulkV2GdnAlphaBytes},
      {kSm87BulkV2GdnP40BetaOffsets[0U], kSm87BulkV2GdnBetaBytes},
      {kSm87BulkV2GdnP40NormalizedQOffsets[1U],
       kSm87BulkV2GdnNormalizedQBytes},
      {kSm87BulkV2GdnP40NormalizedKOffsets[1U],
       kSm87BulkV2GdnNormalizedKBytes},
      {kSm87BulkV2GdnP40PreparedVOffsets[1U],
       kSm87BulkV2GdnPreparedVBytes},
      {kSm87BulkV2GdnP40AlphaOffsets[1U], kSm87BulkV2GdnAlphaBytes},
      {kSm87BulkV2GdnP40BetaOffsets[1U], kSm87BulkV2GdnBetaBytes},
      {kSm87BulkV2GdnP40RawOutputOffsets[0U],
       kSm87BulkV2GdnRawOutputBytes},
      {kSm87BulkV2GdnP40RawOutputOffsets[1U],
       kSm87BulkV2GdnRawOutputBytes},
      {kSm87BulkV2GdnP40ConvHistoryOffsets[0U],
       kSm87BulkV2GdnConvHistoryBytes},
      {kSm87BulkV2GdnP40ConvHistoryOffsets[1U],
       kSm87BulkV2GdnConvHistoryBytes},
      {kSm87BulkV2GdnP40TransactionalRecurrentStateOffset,
       kSm87BulkV2GdnStateBytes},
      {kSm87BulkV2GdnP40CancellationSnapshotOffsets[0U],
       sizeof(std::uint32_t)},
      {kSm87BulkV2GdnP40CancellationSnapshotOffsets[1U],
       sizeof(std::uint32_t)},
  }};
}

[[nodiscard]] constexpr bool
sm87_bulk_v2_gdn_p40_private_layout_valid() noexcept {
  if (kSm87BulkV2GdnP40PrivateBytes %
          kSm87BulkV2GdnP40PrivateExtentAlignment !=
      0U) {
    return false;
  }
  constexpr auto ranges = sm87_bulk_v2_gdn_p40_private_ranges();
  for (std::size_t first = 0U; first < ranges.size(); ++first) {
    if (!ranges[first].valid()) {
      return false;
    }
    for (std::size_t second = first + 1U; second < ranges.size(); ++second) {
      if (ranges[first].offset < ranges[second].end() &&
          ranges[second].offset < ranges[first].end()) {
        return false;
      }
    }
  }
  return true;
}
inline constexpr std::uint64_t kSm87BulkV2GdnP40RawQkvzBytes =
    kSm87BulkV2GdnP40Tokens * kSm87TargetAotGdnRawQkvZChannels *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnP40InterleavedAbBytes =
    kSm87BulkV2GdnP40Tokens * kSm87TargetAotGdnAbChannels *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnP40OutputBytes =
    kSm87BulkV2GdnP40Tokens * kSm87TargetAotGdnOutputChannels *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::size_t kSm87BulkV2GdnP40ArgumentRangeCount = 27U;
inline constexpr std::size_t kSm87BulkV2GdnP40DeviceRangeCount = 26U;
inline constexpr std::size_t kSm87BulkV2GdnP40SessionLayerCount = 48U;
inline constexpr std::size_t kSm87BulkV2GdnP40SessionEventCount =
    kSm87BulkV2GdnP40EventCount + 1U;
inline constexpr std::uint64_t kSm87BulkV2GdnP40SubmissionReceiptMagic =
    0x5147'3344'4e50'3430ULL;
inline constexpr std::uint32_t kSm87BulkV2GdnP40SubmissionReceiptVersion = 1U;

enum class Sm87BulkV2GdnP40OwnerLifecycle : std::uint32_t {
  kReady = 0x5245'4144U,
  kSubmitted = 0x5355'424dU,
  kPoisoned = 0x504f'4953U,
};

// One receipt belongs to the same three-stream/six-event owner for its whole
// lifetime. A post-enqueue host API error drains every stream, poisons this
// receipt, and permanently rejects it. Destroying and recreating the CUDA
// owner is the only valid recovery from kPoisoned. A successful explicit
// drain returns the same owner to kReady and preserves the completed audit
// fields until the next generation starts.
struct Sm87BulkV2GdnP40SubmissionReceipt final {
  std::uint64_t magic = kSm87BulkV2GdnP40SubmissionReceiptMagic;
  std::uint32_t version = kSm87BulkV2GdnP40SubmissionReceiptVersion;
  Sm87BulkV2GdnP40OwnerLifecycle lifecycle =
      Sm87BulkV2GdnP40OwnerLifecycle::kReady;
  std::uint64_t generation = 0U;
  std::uint64_t successful_submission_calls = 0U;
  std::int32_t first_error = 0;
  bool submission_started = false;
  bool drain_attempted = false;
  bool drain_completed = false;
  bool reusable = true;
};

[[nodiscard]] constexpr Sm87BulkV2GdnP40SubmissionReceipt
sm87_bulk_v2_gdn_p40_submission_receipt() noexcept {
  return {};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_p40_submission_receipt_valid(
    const Sm87BulkV2GdnP40SubmissionReceipt& receipt) noexcept {
  if (receipt.magic != kSm87BulkV2GdnP40SubmissionReceiptMagic ||
      receipt.version != kSm87BulkV2GdnP40SubmissionReceiptVersion) {
    return false;
  }
  switch (receipt.lifecycle) {
    case Sm87BulkV2GdnP40OwnerLifecycle::kReady:
      return receipt.reusable;
    case Sm87BulkV2GdnP40OwnerLifecycle::kSubmitted:
      return receipt.submission_started && !receipt.reusable;
    case Sm87BulkV2GdnP40OwnerLifecycle::kPoisoned:
      return receipt.submission_started && !receipt.reusable;
  }
  return false;
}

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
  kSm87BulkV2GdnP40CancellationSnapshotOwnedThroughEpilogue = 1ULL << 10U,
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
    kSm87BulkV2GdnP40NoProductionSelector |
    kSm87BulkV2GdnP40CancellationSnapshotOwnedThroughEpilogue;

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
           sm87_bulk_v2_gdn_p40_private_layout_valid() &&
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
  bool producer_waits_for_epilogue_snapshot = false;
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
           producer_waits_for_epilogue_snapshot == (index >= 2U) &&
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
          index >= 2U,
          true,
          index >= 2U,
          true,
          true};
}

// The complete P40 launcher is request-asynchronous and allocation-free.  Its
// owner creates the three non-default streams, six disable-timing events, and
// all transaction-private buffers before request admission.  The launcher
// never exposes a partially updated recurrent state: all 625 chunks update
// transactional_recurrent_state, which becomes observable only when the
// final epilogue event completes and the request transaction is committed.
struct Sm87BulkV2GdnP40Arguments final {
  const std::uint16_t* raw_qkvz = nullptr;             // [40000,16384]
  const std::uint16_t* interleaved_ab = nullptr;       // [40000,96]
  const std::uint16_t* conv_weight = nullptr;          // [10240,4]
  const std::uint16_t* initial_conv_history = nullptr; // [10240,3]
  const std::uint16_t* a_log = nullptr;                // [48]
  const std::uint16_t* dt_bias = nullptr;              // [48]
  const std::uint16_t* norm_weight = nullptr;          // [128]
  const std::uint16_t* initial_recurrent_state = nullptr;

  std::uint32_t l2_epsilon_fp32_bits = 0U;
  std::uint32_t norm_epsilon_fp32_bits = 0U;

  std::array<float*, kSm87BulkV2GdnP40SlotCount> normalized_q{};
  std::array<float*, kSm87BulkV2GdnP40SlotCount> normalized_k{};
  std::array<std::uint16_t*, kSm87BulkV2GdnP40SlotCount> prepared_v{};
  std::array<float*, kSm87BulkV2GdnP40SlotCount> alpha{};
  std::array<float*, kSm87BulkV2GdnP40SlotCount> beta{};
  std::array<std::uint16_t*, kSm87BulkV2GdnP40SlotCount> raw_output{};
  std::uint16_t* output = nullptr; // request-private [40000,6144]
  std::array<std::uint16_t*, kSm87BulkV2GdnP40SlotCount> conv_history{};
  std::uint16_t* transactional_recurrent_state = nullptr;
  // One device-local snapshot per reusable slot. Exactly one single-thread
  // sampler reads the mapped owner word at each C64 boundary; all producer,
  // recurrence and epilogue CTAs consume this stable device snapshot. This
  // bounds cancellation at C64 without polling mapped host memory per CTA.
  std::array<std::uint32_t*, kSm87BulkV2GdnP40SlotCount>
      cancellation_snapshot{};

  std::array<void*, kSm87BulkV2GdnP40StreamCount> streams{};
  std::array<void*, kSm87BulkV2GdnP40SlotCount> prepared_ready_events{};
  std::array<void*, kSm87BulkV2GdnP40SlotCount> recurrence_done_events{};
  std::array<void*, kSm87BulkV2GdnP40SlotCount> epilogue_done_events{};

  // Strict owner-controlled mapped pair. Runtime validation requires a
  // cudaHostAllocMapped host word and the exact alias returned by
  // cudaHostGetDevicePointer for that word; managed memory and ordinary
  // device allocations are rejected. A nonzero sampled value suppresses the
  // entire next C64 cell. The owner then drains all three streams and discards
  // the unpublished request transaction.
  std::uint32_t* cancellation_host_word = nullptr;
  const std::uint32_t* cancellation_device_alias = nullptr;

  // Stable host receipt owned beside streams/events. It is never a device
  // range. The launcher records whether submission started and whether a
  // failure path drained and poisoned the owner.
  Sm87BulkV2GdnP40SubmissionReceipt* submission_receipt = nullptr;

  // Correctness-only host failure injection. SIZE_MAX disables it. A finite
  // value N injects cudaErrorUnknown immediately after N successful mutating
  // CUDA submission calls. It exists only in this default-off admission cell.
  std::size_t test_only_fail_after_successful_submissions =
      std::numeric_limits<std::size_t>::max();
  // Correctness-only deterministic mid-flight cancellation injection. At the
  // selected C64 boundary a producer-stream host callback sets the mapped
  // owner word before the boundary sampler. SIZE_MAX disables it.
  std::size_t test_only_cancel_before_chunk =
      std::numeric_limits<std::size_t>::max();
};

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_p40_arguments_valid(
    const Sm87BulkV2GdnP40Arguments& arguments) noexcept;

template <typename T, std::size_t Size>
[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_p40_same_identities(
    const std::array<T, Size>& first,
    const std::array<T, Size>& second) noexcept {
  for (std::size_t index = 0U; index < Size; ++index) {
    if (first[index] != second[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_p40_same_static_owner(
    const Sm87BulkV2GdnP40Arguments& first,
    const Sm87BulkV2GdnP40Arguments& second) noexcept {
  return sm87_bulk_v2_gdn_p40_same_identities(first.normalized_q,
                                               second.normalized_q) &&
         sm87_bulk_v2_gdn_p40_same_identities(first.normalized_k,
                                               second.normalized_k) &&
         sm87_bulk_v2_gdn_p40_same_identities(first.prepared_v,
                                               second.prepared_v) &&
         sm87_bulk_v2_gdn_p40_same_identities(first.alpha, second.alpha) &&
         sm87_bulk_v2_gdn_p40_same_identities(first.beta, second.beta) &&
         sm87_bulk_v2_gdn_p40_same_identities(first.raw_output,
                                               second.raw_output) &&
         sm87_bulk_v2_gdn_p40_same_identities(first.conv_history,
                                               second.conv_history) &&
         sm87_bulk_v2_gdn_p40_same_identities(
             first.cancellation_snapshot, second.cancellation_snapshot) &&
         sm87_bulk_v2_gdn_p40_same_identities(first.streams,
                                               second.streams) &&
         sm87_bulk_v2_gdn_p40_same_identities(
             first.prepared_ready_events, second.prepared_ready_events) &&
         sm87_bulk_v2_gdn_p40_same_identities(
             first.recurrence_done_events, second.recurrence_done_events) &&
         sm87_bulk_v2_gdn_p40_same_identities(
             first.epilogue_done_events, second.epilogue_done_events) &&
         first.cancellation_host_word == second.cancellation_host_word &&
         first.cancellation_device_alias ==
             second.cancellation_device_alias &&
         first.submission_receipt == second.submission_receipt &&
         first.l2_epsilon_fp32_bits == second.l2_epsilon_fp32_bits &&
         first.norm_epsilon_fp32_bits == second.norm_epsilon_fp32_bits;
}

// The session plan seals all 48 natural-order GDN layer bindings before the
// request submits work. Layer-dependent input/weight/state/output pointers may
// differ. Double-slot workspaces, streams, events, cancellation ownership and
// the submission receipt are one shared physical owner for the complete
// request. The ingress event transfers main-stream input readiness to the
// producer stream; the final per-epoch epilogue event transfers output/state
// readiness back to the same main stream without a host wait.
struct Sm87BulkV2GdnP40SessionPlan final {
  std::array<Sm87BulkV2GdnP40Arguments,
             kSm87BulkV2GdnP40SessionLayerCount>
      layers{};
  void* main_stream = nullptr;
  void* ingress_ready_event = nullptr;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_p40_session_plan_valid(
    const Sm87BulkV2GdnP40SessionPlan& plan) noexcept {
  if (plan.main_stream == nullptr || plan.ingress_ready_event == nullptr ||
      !sm87_bulk_v2_gdn_p40_arguments_valid(plan.layers[0U])) {
    return false;
  }
  const auto& owner = plan.layers[0U];
  for (void* const stream : owner.streams) {
    if (stream == plan.main_stream) {
      return false;
    }
  }
  const std::array<void*, kSm87BulkV2GdnP40EventCount> owner_events{{
      owner.prepared_ready_events[0U], owner.prepared_ready_events[1U],
      owner.recurrence_done_events[0U], owner.recurrence_done_events[1U],
      owner.epilogue_done_events[0U], owner.epilogue_done_events[1U],
  }};
  for (void* const event : owner_events) {
    if (event == plan.ingress_ready_event) {
      return false;
    }
  }
  for (std::size_t layer = 1U; layer < plan.layers.size(); ++layer) {
    if (!sm87_bulk_v2_gdn_p40_arguments_valid(plan.layers[layer]) ||
        !sm87_bulk_v2_gdn_p40_same_static_owner(owner,
                                                plan.layers[layer])) {
      return false;
    }
  }
  return true;
}

enum class Sm87BulkV2GdnP40SessionLifecycle : std::uint32_t {
  kEmpty = 0U,
  kReady = 0x5245'4144U,
  kActive = 0x4143'5449U,
  kAwaitingDrain = 0x4452'4149U,
  kDrained = 0x444f'4e45U,
  kPoisoned = 0x504f'4953U,
};

// Mutable request-session state. Callers initialize this object exactly once,
// enqueue epochs 0..47 in order, bridge each completed epoch back to the
// sealed main stream, then drain once at request completion/cancellation.
// Allocation and every pointer/stream/event identity must remain live and
// unchanged until kDrained or kPoisoned.
struct Sm87BulkV2GdnP40Session final {
  Sm87BulkV2GdnP40SessionPlan sealed_plan{};
  Sm87BulkV2GdnC64Resources sealed_resources{};
  int sealed_device = -1;
  Sm87BulkV2GdnP40SessionLifecycle lifecycle =
      Sm87BulkV2GdnP40SessionLifecycle::kEmpty;
  std::size_t next_epoch = 0U;
  std::size_t bridged_epochs = 0U;
  bool bridge_pending = false;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_p40_session_state_valid(
    const Sm87BulkV2GdnP40Session& session) noexcept {
  if (session.lifecycle == Sm87BulkV2GdnP40SessionLifecycle::kEmpty) {
    return session.sealed_device == -1 && session.next_epoch == 0U &&
           session.bridged_epochs == 0U && !session.bridge_pending;
  }
  if (!sm87_bulk_v2_gdn_p40_session_plan_valid(session.sealed_plan) ||
      session.sealed_device < 0 ||
      !sm87_bulk_v2_gdn_c64_resources_valid(session.sealed_resources) ||
      session.next_epoch > kSm87BulkV2GdnP40SessionLayerCount ||
      session.bridged_epochs > session.next_epoch) {
    return false;
  }
  switch (session.lifecycle) {
    case Sm87BulkV2GdnP40SessionLifecycle::kReady:
      return session.next_epoch == 0U && session.bridged_epochs == 0U &&
             !session.bridge_pending;
    case Sm87BulkV2GdnP40SessionLifecycle::kActive:
      return session.next_epoch < kSm87BulkV2GdnP40SessionLayerCount &&
             (session.bridge_pending
                  ? session.next_epoch == session.bridged_epochs + 1U
                  : session.next_epoch == session.bridged_epochs);
    case Sm87BulkV2GdnP40SessionLifecycle::kAwaitingDrain:
      return session.next_epoch == kSm87BulkV2GdnP40SessionLayerCount &&
             (session.bridge_pending
                  ? session.next_epoch == session.bridged_epochs + 1U
                  : session.next_epoch == session.bridged_epochs);
    case Sm87BulkV2GdnP40SessionLifecycle::kDrained:
    case Sm87BulkV2GdnP40SessionLifecycle::kPoisoned:
      return !session.bridge_pending;
    case Sm87BulkV2GdnP40SessionLifecycle::kEmpty:
      return false;
  }
  return false;
}

[[nodiscard]] constexpr auto sm87_bulk_v2_gdn_p40_argument_ranges(
    const Sm87BulkV2GdnP40Arguments& arguments) noexcept {
  return std::array<Sm87BulkV2GdnByteRange,
                    kSm87BulkV2GdnP40ArgumentRangeCount>{{
      sm87_bulk_v2_gdn_byte_range(arguments.raw_qkvz,
                                  kSm87BulkV2GdnP40RawQkvzBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.interleaved_ab,
                                  kSm87BulkV2GdnP40InterleavedAbBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.conv_weight,
                                  kSm87BulkV2GdnConvWeightBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.initial_conv_history,
                                  kSm87BulkV2GdnConvHistoryBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.a_log,
                                  kSm87BulkV2GdnHeadVectorBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.dt_bias,
                                  kSm87BulkV2GdnHeadVectorBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.norm_weight,
                                  kSm87BulkV2GdnNormWeightBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.initial_recurrent_state,
                                  kSm87BulkV2GdnStateBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.normalized_q[0U],
                                  kSm87BulkV2GdnNormalizedQBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.normalized_q[1U],
                                  kSm87BulkV2GdnNormalizedQBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.normalized_k[0U],
                                  kSm87BulkV2GdnNormalizedKBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.normalized_k[1U],
                                  kSm87BulkV2GdnNormalizedKBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.prepared_v[0U],
                                  kSm87BulkV2GdnPreparedVBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.prepared_v[1U],
                                  kSm87BulkV2GdnPreparedVBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.alpha[0U],
                                  kSm87BulkV2GdnAlphaBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.alpha[1U],
                                  kSm87BulkV2GdnAlphaBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.beta[0U],
                                  kSm87BulkV2GdnBetaBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.beta[1U],
                                  kSm87BulkV2GdnBetaBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.raw_output[0U],
                                  kSm87BulkV2GdnRawOutputBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.raw_output[1U],
                                  kSm87BulkV2GdnRawOutputBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.output,
                                  kSm87BulkV2GdnP40OutputBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.conv_history[0U],
                                  kSm87BulkV2GdnConvHistoryBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.conv_history[1U],
                                  kSm87BulkV2GdnConvHistoryBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.transactional_recurrent_state,
                                  kSm87BulkV2GdnStateBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.cancellation_snapshot[0U],
                                  sizeof(std::uint32_t)),
      sm87_bulk_v2_gdn_byte_range(arguments.cancellation_snapshot[1U],
                                  sizeof(std::uint32_t)),
      sm87_bulk_v2_gdn_byte_range(arguments.cancellation_device_alias,
                                  sizeof(std::uint32_t)),
  }};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_p40_arguments_valid(
    const Sm87BulkV2GdnP40Arguments& arguments) noexcept {
  if (arguments.l2_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.norm_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits) {
    return false;
  }
  const auto ranges = sm87_bulk_v2_gdn_p40_argument_ranges(arguments);
  for (std::size_t first = 0U; first < ranges.size(); ++first) {
    if (!ranges[first].valid ||
        ranges[first].begin % kSm87BulkV2GdnPointerAlignment != 0U) {
      return false;
    }
    for (std::size_t second = first + 1U; second < ranges.size(); ++second) {
      if (sm87_bulk_v2_gdn_ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  if (arguments.cancellation_host_word == nullptr ||
      reinterpret_cast<std::uintptr_t>(arguments.cancellation_host_word) %
              alignof(std::uint32_t) !=
          0U ||
      reinterpret_cast<std::uintptr_t>(arguments.cancellation_device_alias) %
              alignof(std::uint32_t) !=
          0U ||
      arguments.submission_receipt == nullptr ||
      reinterpret_cast<std::uintptr_t>(arguments.submission_receipt) %
              alignof(Sm87BulkV2GdnP40SubmissionReceipt) !=
          0U) {
    return false;
  }
  for (std::size_t first = 0U; first < arguments.streams.size(); ++first) {
    if (arguments.streams[first] == nullptr) {
      return false;
    }
    for (std::size_t second = first + 1U; second < arguments.streams.size();
         ++second) {
      if (arguments.streams[first] == arguments.streams[second]) {
        return false;
      }
    }
  }
  const std::array<void*, kSm87BulkV2GdnP40EventCount> events{{
      arguments.prepared_ready_events[0U],
      arguments.prepared_ready_events[1U],
      arguments.recurrence_done_events[0U],
      arguments.recurrence_done_events[1U],
      arguments.epilogue_done_events[0U],
      arguments.epilogue_done_events[1U],
  }};
  for (std::size_t first = 0U; first < events.size(); ++first) {
    if (events[first] == nullptr) {
      return false;
    }
    for (std::size_t second = first + 1U; second < events.size(); ++second) {
      if (events[first] == events[second]) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] inline Sm87BulkV2GdnC64Arguments
sm87_bulk_v2_gdn_p40_chunk_arguments(
    const Sm87BulkV2GdnP40Arguments& arguments,
    const Sm87BulkV2GdnP40Chunk& chunk, void* const stream) noexcept {
  if (!chunk.valid(sm87_bulk_v2_gdn_p40_plan()) || stream == nullptr) {
    return {};
  }
  Sm87BulkV2GdnC64Arguments cell;
  cell.raw_qkvz =
      arguments.raw_qkvz + chunk.first_position *
                                kSm87TargetAotGdnRawQkvZChannels;
  cell.interleaved_ab =
      arguments.interleaved_ab +
      chunk.first_position * kSm87TargetAotGdnAbChannels;
  cell.conv_weight = arguments.conv_weight;
  cell.initial_conv_history =
      chunk.input_history_is_request_initial
          ? arguments.initial_conv_history
          : arguments.conv_history[chunk.input_history_slot];
  cell.a_log = arguments.a_log;
  cell.dt_bias = arguments.dt_bias;
  cell.norm_weight = arguments.norm_weight;
  cell.initial_recurrent_state =
      chunk.index == 0U ? arguments.initial_recurrent_state
                        : arguments.transactional_recurrent_state;
  cell.l2_epsilon_fp32_bits = arguments.l2_epsilon_fp32_bits;
  cell.norm_epsilon_fp32_bits = arguments.norm_epsilon_fp32_bits;
  cell.first_position = chunk.first_position;
  cell.token_count = chunk.token_count;
  cell.normalized_q = arguments.normalized_q[chunk.prepared_slot];
  cell.normalized_k = arguments.normalized_k[chunk.prepared_slot];
  cell.prepared_v = arguments.prepared_v[chunk.prepared_slot];
  cell.alpha = arguments.alpha[chunk.prepared_slot];
  cell.beta = arguments.beta[chunk.prepared_slot];
  cell.raw_output = arguments.raw_output[chunk.raw_output_slot];
  cell.output = arguments.output +
                chunk.first_position * kSm87TargetAotGdnOutputChannels;
  cell.final_conv_history =
      arguments.conv_history[chunk.output_history_slot];
  cell.final_recurrent_state = arguments.transactional_recurrent_state;
  cell.cuda_stream = stream;
  return cell;
}

// Enqueues all 625 chunks and returns after recording the final epilogue
// event.  It never synchronizes the device.  The final completion identity is
// epilogue_done_events[624 % 2]; because the epilogue stream is ordered, that
// event also covers every earlier output chunk.
[[nodiscard]] int launch_sm87_bulk_dataflow_v2_gdn_p40_cuda(
    const Sm87BulkV2GdnP40Arguments& arguments) noexcept;

// Blocks only for explicit request retirement. It drains all three streams,
// marks a successful submitted owner reusable, and poisons it on any CUDA
// completion failure. Call this after normal completion or cancellation;
// synchronizing only the epilogue stream is not the owner-retirement contract.
[[nodiscard]] int drain_sm87_bulk_dataflow_v2_gdn_p40_cuda(
    const Sm87BulkV2GdnP40Arguments& arguments) noexcept;

// Session startup is the sole expensive admission point. It validates and
// seals the fixed SM87 resource record, all 48 allocation extents and derived
// C64 views, the mapped cancellation pair, and every stream/event identity.
// No work is enqueued on success.
[[nodiscard]] int initialize_sm87_bulk_dataflow_v2_gdn_p40_session_cuda(
    Sm87BulkV2GdnP40Session* session,
    const Sm87BulkV2GdnP40SessionPlan& plan) noexcept;

// Enqueues exactly the next natural-order layer epoch. This hot call consumes
// only the sealed binding and device event DAG. It performs no pointer,
// allocation, event-idle, device-property, or occupancy query and never waits
// on the host. A partial submission failure drains the three internal streams
// and permanently poisons the request session.
[[nodiscard]] int enqueue_sm87_bulk_dataflow_v2_gdn_p40_session_epoch_cuda(
    Sm87BulkV2GdnP40Session* session, std::size_t epoch) noexcept;

// Makes the just-enqueued epoch's final epilogue event a dependency of the
// sealed main stream. No host synchronization occurs. The next epoch is not
// admissible until this bridge has been submitted.
[[nodiscard]] int
bridge_sm87_bulk_dataflow_v2_gdn_p40_session_epoch_to_main_cuda(
    Sm87BulkV2GdnP40Session* session) noexcept;

// The single request-retirement path for normal completion or cancellation.
// It drains producer, recurrence and epilogue once. The session is terminal
// afterwards; a poisoned session remains poisoned and cannot be rearmed.
[[nodiscard]] int drain_sm87_bulk_dataflow_v2_gdn_p40_session_cuda(
    Sm87BulkV2GdnP40Session* session) noexcept;

static_assert(kSm87BulkV2GdnP40Chunks == 625U);
static_assert(kSm87BulkV2GdnP40RawQkvzBytes == 1'310'720'000U);
static_assert(kSm87BulkV2GdnP40InterleavedAbBytes == 7'680'000U);
static_assert(kSm87BulkV2GdnP40OutputBytes == 491'520'000U);
static_assert(kSm87BulkV2GdnProducerWorkspaceBytes == 1'859'584U);
static_assert(kSm87BulkV2GdnP40PreparedSlotsBytes == 3'719'168U);
static_assert(kSm87BulkV2GdnP40RawOutputSlotsBytes == 1'572'864U);
static_assert(kSm87BulkV2GdnP40HistorySlotsBytes == 122'880U);
static_assert(kSm87BulkV2GdnP40CancellationSnapshotBytes == 8U);
static_assert(kSm87BulkV2GdnP40CancellationSnapshotOffsets[0U] ==
              6'987'776U);
static_assert(kSm87BulkV2GdnP40CancellationSnapshotOffsets[1U] ==
              6'987'792U);
static_assert(kSm87BulkV2GdnP40PrivatePayloadBytes == 6'987'796U);
static_assert(kSm87BulkV2GdnP40PrivateBytes == 6'988'032U);
static_assert(sm87_bulk_v2_gdn_p40_private_layout_valid());
static_assert(kSm87BulkV2GdnP40RequiredPolicy == 0x7ffULL);
static_assert(sm87_bulk_v2_gdn_p40_plan().valid());

}  // namespace q3x::kernels
