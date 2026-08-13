#pragma once

#include "q3x/kernels/sm87_target_aot_gdn_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// First executable cell for AC-PREFILL-SM87-BULK-DATAFLOW-v2.  This is a
// default-off, single-layer, single-C64 component.  It deliberately has no
// selector, dispatcher, or production-eligibility bit.  Kernel completion is
// the only cross-CTA synchronization boundary in this first cell.
inline constexpr std::size_t kSm87BulkV2GdnC64Tokens = 64U;
inline constexpr std::size_t kSm87BulkV2GdnMaximumPromptTokens =
    kSm87TargetAotCapacityContracts[2U].maximum_prompt_tokens;
inline constexpr std::size_t kSm87BulkV2GdnProducerRoles = 64U;
inline constexpr std::size_t kSm87BulkV2GdnQkProducerRoles = 16U;
inline constexpr std::size_t kSm87BulkV2GdnValueProducerRoles = 48U;
inline constexpr std::size_t kSm87BulkV2GdnProducerRowsPerCta = 4U;
inline constexpr std::size_t kSm87BulkV2GdnProducerThreads = 128U;
inline constexpr std::size_t kSm87BulkV2GdnProducerCtas =
    (kSm87BulkV2GdnC64Tokens / kSm87BulkV2GdnProducerRowsPerCta) *
    kSm87BulkV2GdnProducerRoles;
inline constexpr std::size_t kSm87BulkV2GdnRecurrenceThreads = 256U;
inline constexpr std::size_t kSm87BulkV2GdnRecurrenceCtas =
    kSm87TargetAotGdnValueHeads;
inline constexpr std::size_t kSm87BulkV2GdnEpilogueRowsPerCta = 8U;
inline constexpr std::size_t kSm87BulkV2GdnEpilogueThreads = 256U;
inline constexpr std::size_t kSm87BulkV2GdnEpilogueCtas =
    (kSm87BulkV2GdnC64Tokens * kSm87TargetAotGdnValueHeads) /
    kSm87BulkV2GdnEpilogueRowsPerCta;

inline constexpr std::size_t kSm87BulkV2GdnPointerAlignment = 16U;
inline constexpr std::size_t kSm87BulkV2GdnArgumentRangeCount = 17U;
inline constexpr int kSm87BulkV2GdnMaximumRegistersPerThread = 85;
inline constexpr int kSm87BulkV2GdnMinimumActiveCtasPerSm = 3;

inline constexpr std::uint64_t kSm87BulkV2GdnRawQkvzBytes =
    kSm87BulkV2GdnC64Tokens * kSm87TargetAotGdnRawQkvZChannels *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnInterleavedAbBytes =
    kSm87BulkV2GdnC64Tokens * kSm87TargetAotGdnAbChannels *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnConvWeightBytes =
    kSm87TargetAotGdnConvWeightElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnConvHistoryBytes =
    kSm87TargetAotGdnTotalConvHistoryBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnHeadVectorBytes =
    kSm87TargetAotGdnScalarHeadElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnNormWeightBytes =
    kSm87TargetAotGdnNormWeightElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnStateBytes =
    kSm87TargetAotGdnTotalStateBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnNormalizedQBytes =
    kSm87BulkV2GdnC64Tokens * kSm87TargetAotGdnQkGroups *
    kSm87TargetAotGdnStateKeyDimension * sizeof(float);
inline constexpr std::uint64_t kSm87BulkV2GdnNormalizedKBytes =
    kSm87BulkV2GdnNormalizedQBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnPreparedVBytes =
    kSm87BulkV2GdnC64Tokens * kSm87TargetAotGdnValueHeads *
    kSm87TargetAotGdnStateValueDimension * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnAlphaBytes =
    kSm87BulkV2GdnC64Tokens * kSm87TargetAotGdnValueHeads * sizeof(float);
inline constexpr std::uint64_t kSm87BulkV2GdnBetaBytes =
    kSm87BulkV2GdnAlphaBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnRawOutputBytes =
    kSm87BulkV2GdnPreparedVBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnOutputBytes =
    kSm87BulkV2GdnC64Tokens * kSm87TargetAotGdnOutputChannels *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87BulkV2GdnStateTraceBytes =
    kSm87BulkV2GdnC64Tokens * kSm87BulkV2GdnStateBytes;
inline constexpr std::uint64_t kSm87BulkV2GdnWorkspaceBytes =
    kSm87BulkV2GdnNormalizedQBytes + kSm87BulkV2GdnNormalizedKBytes +
    kSm87BulkV2GdnPreparedVBytes + kSm87BulkV2GdnAlphaBytes +
    kSm87BulkV2GdnBetaBytes + kSm87BulkV2GdnRawOutputBytes;

enum Sm87BulkV2GdnC64Policy : std::uint64_t {
  kSm87BulkV2GdnExactPerTokenBf16State = 1ULL << 0U,
  kSm87BulkV2GdnPreRoundFp32TokenOutput = 1ULL << 1U,
  kSm87BulkV2GdnRawBf16BeforeNormGate = 1ULL << 2U,
  kSm87BulkV2GdnQkOncePerThreeValueHeads = 1ULL << 3U,
  kSm87BulkV2GdnNoWyKktSsdScan = 1ULL << 4U,
  kSm87BulkV2GdnNoFp32AuthoritativeState = 1ULL << 5U,
  kSm87BulkV2GdnNoCooperativeLaunch = 1ULL << 6U,
  kSm87BulkV2GdnNoProductionSelector = 1ULL << 7U,
  kSm87BulkV2GdnSingleStreamFirstCell = 1ULL << 8U,
  kSm87BulkV2GdnResourceGateFailClosed = 1ULL << 9U,
};

inline constexpr std::uint64_t kSm87BulkV2GdnC64RequiredPolicy =
    kSm87BulkV2GdnExactPerTokenBf16State |
    kSm87BulkV2GdnPreRoundFp32TokenOutput |
    kSm87BulkV2GdnRawBf16BeforeNormGate |
    kSm87BulkV2GdnQkOncePerThreeValueHeads |
    kSm87BulkV2GdnNoWyKktSsdScan |
    kSm87BulkV2GdnNoFp32AuthoritativeState |
    kSm87BulkV2GdnNoCooperativeLaunch |
    kSm87BulkV2GdnNoProductionSelector |
    kSm87BulkV2GdnSingleStreamFirstCell |
    kSm87BulkV2GdnResourceGateFailClosed;

// The cell does not define a numerically similar v2 contract.  It reuses the
// target-AOT identities verbatim so a future oracle cannot replace bitwise
// comparison with a tolerance merely because producer/consumer ownership
// changed.
inline constexpr Sm87TargetAotGdnConvNumericalContract
    kSm87BulkV2GdnConvContract =
        Sm87TargetAotGdnConvNumericalContract::
            kFp32FmaOldestToCurrentThenSiluThenBf16Rne;
inline constexpr Sm87TargetAotGdnQkNormalizationContract
    kSm87BulkV2GdnQkContract =
        Sm87TargetAotGdnQkNormalizationContract::
            kFp32Pair0_64Pair32_96Shuffle16To1RsqrtfQInvSqrt128;
inline constexpr Sm87TargetAotGdnGateScalarContract
    kSm87BulkV2GdnGateContract =
        Sm87TargetAotGdnGateScalarContract::
            kCudaExpfLog1pfThreshold20StableSigmoidFp32;
inline constexpr Sm87TargetAotGdnRecurrenceExecutionContract
    kSm87BulkV2GdnRecurrenceContract =
        Sm87TargetAotGdnRecurrenceExecutionContract::
            kAlphaScalePredictionUpdateOutputKeyAscendingFmafPerTokenBf16;
inline constexpr Sm87TargetAotGdnNormGateContract
    kSm87BulkV2GdnNormGateContract =
        Sm87TargetAotGdnNormGateContract::
            kRawBf16PairShuffleRmsRsqrtfPlainWeightSiluBf16Rne;

struct Sm87BulkV2GdnC64Arguments final {
  const std::uint16_t* raw_qkvz = nullptr;             // [64,16384]
  const std::uint16_t* interleaved_ab = nullptr;       // [64,96]
  const std::uint16_t* conv_weight = nullptr;          // [10240,4]
  const std::uint16_t* initial_conv_history = nullptr; // [10240,3]
  const std::uint16_t* a_log = nullptr;                // [48]
  const std::uint16_t* dt_bias = nullptr;              // [48]
  const std::uint16_t* norm_weight = nullptr;          // [128]
  const std::uint16_t* initial_recurrent_state = nullptr;

  std::uint32_t l2_epsilon_fp32_bits = 0U;
  std::uint32_t norm_epsilon_fp32_bits = 0U;
  std::size_t first_position = 0U;
  std::size_t token_count = 0U;

  // Single-slot first-cell workspace.  The complete v2 plan will own two
  // instances of each producer/consumer edge; this cell intentionally does
  // not claim that overlap yet.
  float* normalized_q = nullptr;          // [16,64,128]
  float* normalized_k = nullptr;          // [16,64,128]
  std::uint16_t* prepared_v = nullptr;    // [48,64,128]
  float* alpha = nullptr;                 // [48,64]
  float* beta = nullptr;                  // [48,64]
  std::uint16_t* raw_output = nullptr;    // [64,48,128]

  std::uint16_t* output = nullptr;             // [64,6144]
  std::uint16_t* final_conv_history = nullptr; // [10240,3]
  std::uint16_t* final_recurrent_state = nullptr;
  void* cuda_stream = nullptr;
};

struct Sm87BulkV2GdnByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87BulkV2GdnByteRange
sm87_bulk_v2_gdn_byte_range(const void* const pointer,
                            const std::uint64_t bytes) noexcept {
  if (pointer == nullptr || bytes == 0U ||
      bytes > std::numeric_limits<std::uintptr_t>::max()) {
    return {};
  }
  const auto begin = reinterpret_cast<std::uintptr_t>(pointer);
  if (begin > std::numeric_limits<std::uintptr_t>::max() -
                  static_cast<std::uintptr_t>(bytes)) {
    return {};
  }
  return {begin, begin + static_cast<std::uintptr_t>(bytes), true};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_ranges_overlap(
    const Sm87BulkV2GdnByteRange& left,
    const Sm87BulkV2GdnByteRange& right) noexcept {
  return !left.valid || !right.valid ||
         (left.begin < right.end && right.begin < left.end);
}

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_pointer_aligned(
    const void* const pointer) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) %
                 kSm87BulkV2GdnPointerAlignment ==
             0U;
}

[[nodiscard]] constexpr auto sm87_bulk_v2_gdn_c64_argument_ranges(
    const Sm87BulkV2GdnC64Arguments& arguments) noexcept {
  return std::array<Sm87BulkV2GdnByteRange,
                    kSm87BulkV2GdnArgumentRangeCount>{{
      sm87_bulk_v2_gdn_byte_range(arguments.raw_qkvz,
                                  kSm87BulkV2GdnRawQkvzBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.interleaved_ab,
                                  kSm87BulkV2GdnInterleavedAbBytes),
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
      sm87_bulk_v2_gdn_byte_range(arguments.normalized_q,
                                  kSm87BulkV2GdnNormalizedQBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.normalized_k,
                                  kSm87BulkV2GdnNormalizedKBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.prepared_v,
                                  kSm87BulkV2GdnPreparedVBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.alpha,
                                  kSm87BulkV2GdnAlphaBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.beta,
                                  kSm87BulkV2GdnBetaBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.raw_output,
                                  kSm87BulkV2GdnRawOutputBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.output,
                                  kSm87BulkV2GdnOutputBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.final_conv_history,
                                  kSm87BulkV2GdnConvHistoryBytes),
      sm87_bulk_v2_gdn_byte_range(arguments.final_recurrent_state,
                                  kSm87BulkV2GdnStateBytes),
  }};
}

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_c64_arguments_valid(
    const Sm87BulkV2GdnC64Arguments& arguments) noexcept {
  if (arguments.token_count != kSm87BulkV2GdnC64Tokens ||
      arguments.first_position % kSm87BulkV2GdnC64Tokens != 0U ||
      arguments.first_position >
          kSm87BulkV2GdnMaximumPromptTokens -
              kSm87BulkV2GdnC64Tokens ||
      arguments.l2_epsilon_fp32_bits != kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.norm_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.cuda_stream == nullptr) {
    return false;
  }

  const std::array<const void*, kSm87BulkV2GdnArgumentRangeCount> pointers{{
      arguments.raw_qkvz,
      arguments.interleaved_ab,
      arguments.conv_weight,
      arguments.initial_conv_history,
      arguments.a_log,
      arguments.dt_bias,
      arguments.norm_weight,
      arguments.initial_recurrent_state,
      arguments.normalized_q,
      arguments.normalized_k,
      arguments.prepared_v,
      arguments.alpha,
      arguments.beta,
      arguments.raw_output,
      arguments.output,
      arguments.final_conv_history,
      arguments.final_recurrent_state,
  }};
  for (const void* const pointer : pointers) {
    if (!sm87_bulk_v2_gdn_pointer_aligned(pointer)) {
      return false;
    }
  }

  const auto ranges = sm87_bulk_v2_gdn_c64_argument_ranges(arguments);

  constexpr std::size_t kInitialStateRange = 7U;
  constexpr std::size_t kFinalStateRange = 16U;
  for (std::size_t first = 0U; first < ranges.size(); ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U; second < ranges.size(); ++second) {
      const bool exact_in_place_state =
          first == kInitialStateRange && second == kFinalStateRange &&
          ranges[first].begin == ranges[second].begin &&
          ranges[first].end == ranges[second].end;
      if (!exact_in_place_state &&
          sm87_bulk_v2_gdn_ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_c64_state_trace_valid(
    const Sm87BulkV2GdnC64Arguments& arguments,
    const std::uint16_t* const state_after_token) noexcept {
  if (!sm87_bulk_v2_gdn_c64_arguments_valid(arguments) ||
      !sm87_bulk_v2_gdn_pointer_aligned(state_after_token)) {
    return false;
  }
  const auto trace_range = sm87_bulk_v2_gdn_byte_range(
      state_after_token, kSm87BulkV2GdnStateTraceBytes);
  if (!trace_range.valid) {
    return false;
  }
  for (const auto& argument_range :
       sm87_bulk_v2_gdn_c64_argument_ranges(arguments)) {
    if (sm87_bulk_v2_gdn_ranges_overlap(trace_range, argument_range)) {
      return false;
    }
  }
  return true;
}

struct Sm87BulkV2GdnKernelResources final {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int threads_per_block = 0;
  int physical_grid_ctas = 0;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_kernel_resources_pass(
    const Sm87BulkV2GdnKernelResources& resources,
    const int expected_threads, const int expected_grid_ctas) noexcept {
  return resources.registers_per_thread > 0 &&
         resources.registers_per_thread <=
             kSm87BulkV2GdnMaximumRegistersPerThread &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >= expected_threads &&
         resources.active_blocks_per_sm >=
             kSm87BulkV2GdnMinimumActiveCtasPerSm &&
         resources.threads_per_block == expected_threads &&
         resources.physical_grid_ctas == expected_grid_ctas;
}

struct Sm87BulkV2GdnC64Resources final {
  int binary_version = 0;
  Sm87BulkV2GdnKernelResources producer{};
  Sm87BulkV2GdnKernelResources recurrence{};
  Sm87BulkV2GdnKernelResources epilogue{};
  bool kernels_compiled = false;
  bool exact_geometry = false;
  bool resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool sm87_bulk_v2_gdn_c64_resources_valid(
    const Sm87BulkV2GdnC64Resources& resources) noexcept {
  if (resources.numerical_contract_qualified ||
      resources.production_dispatch_eligible) {
    return false;
  }
  if (!resources.kernels_compiled) {
    return false;
  }
  const bool kernels_pass =
      resources.binary_version == 87 && resources.exact_geometry &&
      sm87_bulk_v2_gdn_kernel_resources_pass(
          resources.producer,
          static_cast<int>(kSm87BulkV2GdnProducerThreads),
          static_cast<int>(kSm87BulkV2GdnProducerCtas)) &&
      sm87_bulk_v2_gdn_kernel_resources_pass(
          resources.recurrence,
          static_cast<int>(kSm87BulkV2GdnRecurrenceThreads),
          static_cast<int>(kSm87BulkV2GdnRecurrenceCtas)) &&
      sm87_bulk_v2_gdn_kernel_resources_pass(
          resources.epilogue,
          static_cast<int>(kSm87BulkV2GdnEpilogueThreads),
          static_cast<int>(kSm87BulkV2GdnEpilogueCtas));
  return kernels_pass && resources.resource_gate_passed;
}

// Querying and launching are present only when the explicit test-only CMake
// admission switch compiles this cell.  launch() re-runs the resource query
// and refuses to enqueue any kernel when one of the three hard gates fails.
[[nodiscard]] int query_sm87_bulk_dataflow_v2_gdn_c64_resources_cuda(
    Sm87BulkV2GdnC64Resources* resources) noexcept;

[[nodiscard]] int launch_sm87_bulk_dataflow_v2_gdn_c64_cuda(
    const Sm87BulkV2GdnC64Arguments& arguments) noexcept;

// Correctness-only scaffold.  It executes the same three-stage cell but uses
// a distinct recurrence specialization that publishes the packed BF16 state
// after every token into [64,48,128,128].  The trace has no performance or
// production authority and is never used by launch().
[[nodiscard]] int launch_sm87_bulk_dataflow_v2_gdn_c64_state_trace_cuda(
    const Sm87BulkV2GdnC64Arguments& arguments,
    std::uint16_t* state_after_token) noexcept;

static_assert(kSm87BulkV2GdnProducerCtas == 1'024U);
static_assert(kSm87BulkV2GdnMaximumPromptTokens == 130'000U);
static_assert(kSm87BulkV2GdnRecurrenceCtas == 48U);
static_assert(kSm87BulkV2GdnEpilogueCtas == 384U);
static_assert(kSm87BulkV2GdnNormalizedQBytes == 524'288U);
static_assert(kSm87BulkV2GdnPreparedVBytes == 786'432U);
static_assert(kSm87BulkV2GdnAlphaBytes == 12'288U);
static_assert(kSm87BulkV2GdnRawOutputBytes == 786'432U);
static_assert(kSm87BulkV2GdnWorkspaceBytes == 2'646'016U);
static_assert(kSm87BulkV2GdnStateTraceBytes == 100'663'296U);
static_assert(kSm87BulkV2GdnC64RequiredPolicy == 0x3ffULL);

}  // namespace q3x::kernels
