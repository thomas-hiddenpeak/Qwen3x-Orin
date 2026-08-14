#pragma once

#include "q3x/kernels/sm87_target_aot_gdn_plan.h"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// Default-off executable slice for AC-PREFILL-SM87-MACROFEED-v3.  One
// convolution launch advances the complete P40 Q/K/V span.  Eight exact
// recurrence+RMSNorm+SiLU-gate macrochunks then retain one value-head state
// per CTA, reducing the former C64 graph from 1,875 launches per linear layer
// to nine.  This surface is an admission candidate, not a production selector.
inline constexpr std::size_t kSm87MacrofeedV3GdnP40Tokens = 40'000U;
inline constexpr std::size_t kSm87MacrofeedV3GdnMacrochunkTokens = 5'000U;
inline constexpr std::size_t kSm87MacrofeedV3GdnMacrochunks =
    kSm87MacrofeedV3GdnP40Tokens / kSm87MacrofeedV3GdnMacrochunkTokens;
inline constexpr std::size_t kSm87MacrofeedV3GdnConvThreads = 256U;
inline constexpr std::size_t kSm87MacrofeedV3GdnConvCtas =
    kSm87TargetAotGdnTotalConvChannels / kSm87MacrofeedV3GdnConvThreads;
inline constexpr std::size_t kSm87MacrofeedV3GdnRecurrenceThreads = 256U;
inline constexpr std::size_t kSm87MacrofeedV3GdnRecurrenceCtas =
    kSm87TargetAotGdnValueHeads;
inline constexpr std::size_t kSm87MacrofeedV3GdnPhysicalKernels =
    1U + kSm87MacrofeedV3GdnMacrochunks;
inline constexpr std::size_t kSm87MacrofeedV3GdnCancellationQuantum = 256U;
inline constexpr std::size_t kSm87MacrofeedV3GdnPointerAlignment = 16U;

inline constexpr std::uint64_t kSm87MacrofeedV3GdnRawQkvBytes =
    kSm87MacrofeedV3GdnP40Tokens * kSm87TargetAotGdnTotalConvChannels *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacrofeedV3GdnScalarPlaneBytes =
    kSm87MacrofeedV3GdnP40Tokens * kSm87TargetAotGdnValueHeads *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacrofeedV3GdnZBytes =
    kSm87MacrofeedV3GdnP40Tokens * kSm87TargetAotGdnOutputChannels *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacrofeedV3GdnOutputBytes =
    kSm87MacrofeedV3GdnZBytes;
inline constexpr std::uint64_t kSm87MacrofeedV3GdnConvWeightBytes =
    kSm87TargetAotGdnConvWeightElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacrofeedV3GdnConvHistoryBytes =
    kSm87TargetAotGdnTotalConvHistoryBytes;
inline constexpr std::uint64_t kSm87MacrofeedV3GdnHeadVectorBytes =
    kSm87TargetAotGdnScalarHeadElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacrofeedV3GdnNormWeightBytes =
    kSm87TargetAotGdnNormWeightElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87MacrofeedV3GdnStateBytes =
    kSm87TargetAotGdnTotalStateBytes;

enum Sm87MacrofeedV3GdnPolicy : std::uint64_t {
  kSm87MacrofeedV3GdnExactPerTokenBf16State = 1ULL << 0U,
  kSm87MacrofeedV3GdnPreRoundFp32TokenOutput = 1ULL << 1U,
  kSm87MacrofeedV3GdnRawBf16BeforeNormGate = 1ULL << 2U,
  kSm87MacrofeedV3GdnExactReductionTrees = 1ULL << 3U,
  kSm87MacrofeedV3GdnNoAssociativeScan = 1ULL << 4U,
  kSm87MacrofeedV3GdnNoFp32AuthoritativeState = 1ULL << 5U,
  kSm87MacrofeedV3GdnNoCooperativeLaunch = 1ULL << 6U,
  kSm87MacrofeedV3GdnBoundedCancellationPolling = 1ULL << 7U,
  kSm87MacrofeedV3GdnNoProductionSelector = 1ULL << 8U,
};

inline constexpr std::uint64_t kSm87MacrofeedV3GdnRequiredPolicy =
    kSm87MacrofeedV3GdnExactPerTokenBf16State |
    kSm87MacrofeedV3GdnPreRoundFp32TokenOutput |
    kSm87MacrofeedV3GdnRawBf16BeforeNormGate |
    kSm87MacrofeedV3GdnExactReductionTrees |
    kSm87MacrofeedV3GdnNoAssociativeScan |
    kSm87MacrofeedV3GdnNoFp32AuthoritativeState |
    kSm87MacrofeedV3GdnNoCooperativeLaunch |
    kSm87MacrofeedV3GdnBoundedCancellationPolling |
    kSm87MacrofeedV3GdnNoProductionSelector;

struct Sm87MacrofeedV3GdnP40Arguments final {
  const std::uint16_t* raw_qkv = nullptr;       // [40000,10240]
  const std::uint16_t* a = nullptr;             // [40000,48]
  const std::uint16_t* b = nullptr;             // [40000,48]
  const std::uint16_t* conv_weight = nullptr;   // [10240,4]
  const std::uint16_t* a_log = nullptr;         // [48]
  const std::uint16_t* dt_bias = nullptr;       // [48]
  const std::uint16_t* norm_weight = nullptr;   // [128]
  const std::uint16_t* z = nullptr;             // [40000,6144]
  std::uint16_t* conv_history = nullptr;        // [10240,3], in/out
  std::uint16_t* recurrent_state = nullptr;     // [48,128,128], in/out
  std::uint16_t* conv_qkv = nullptr;            // [40000,10240]
  std::uint16_t* output = nullptr;              // [40000,6144]
  const std::uint32_t* cancellation_signal = nullptr;
  std::uint32_t l2_epsilon_fp32_bits = 0U;
  std::uint32_t norm_epsilon_fp32_bits = 0U;
  void* cuda_stream = nullptr;
};

struct Sm87MacrofeedV3GdnByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87MacrofeedV3GdnByteRange
sm87_macrofeed_v3_gdn_byte_range(const void* const pointer,
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

[[nodiscard]] constexpr bool sm87_macrofeed_v3_gdn_ranges_overlap(
    const Sm87MacrofeedV3GdnByteRange& left,
    const Sm87MacrofeedV3GdnByteRange& right) noexcept {
  return !left.valid || !right.valid ||
         (left.begin < right.end && right.begin < left.end);
}

[[nodiscard]] constexpr bool sm87_macrofeed_v3_gdn_pointer_aligned(
    const void* const pointer) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) %
                 kSm87MacrofeedV3GdnPointerAlignment ==
             0U;
}

[[nodiscard]] constexpr bool sm87_macrofeed_v3_gdn_p40_arguments_valid(
    const Sm87MacrofeedV3GdnP40Arguments& arguments) noexcept {
  if (arguments.l2_epsilon_fp32_bits != kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.norm_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.cuda_stream == nullptr) {
    return false;
  }
  const void* const pointers[] = {
      arguments.raw_qkv,       arguments.a,
      arguments.b,             arguments.conv_weight,
      arguments.a_log,         arguments.dt_bias,
      arguments.norm_weight,   arguments.z,
      arguments.conv_history,  arguments.recurrent_state,
      arguments.conv_qkv,      arguments.output,
  };
  for (const void* const pointer : pointers) {
    if (!sm87_macrofeed_v3_gdn_pointer_aligned(pointer)) {
      return false;
    }
  }
  if (arguments.cancellation_signal != nullptr &&
      reinterpret_cast<std::uintptr_t>(arguments.cancellation_signal) %
              alignof(std::uint32_t) !=
          0U) {
    return false;
  }

  const Sm87MacrofeedV3GdnByteRange ranges[] = {
      sm87_macrofeed_v3_gdn_byte_range(arguments.raw_qkv,
                                       kSm87MacrofeedV3GdnRawQkvBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.a,
                                       kSm87MacrofeedV3GdnScalarPlaneBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.b,
                                       kSm87MacrofeedV3GdnScalarPlaneBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.conv_weight,
                                       kSm87MacrofeedV3GdnConvWeightBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.a_log,
                                       kSm87MacrofeedV3GdnHeadVectorBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.dt_bias,
                                       kSm87MacrofeedV3GdnHeadVectorBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.norm_weight,
                                       kSm87MacrofeedV3GdnNormWeightBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.z,
                                       kSm87MacrofeedV3GdnZBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.conv_history,
                                       kSm87MacrofeedV3GdnConvHistoryBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.recurrent_state,
                                       kSm87MacrofeedV3GdnStateBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.conv_qkv,
                                       kSm87MacrofeedV3GdnRawQkvBytes),
      sm87_macrofeed_v3_gdn_byte_range(arguments.output,
                                       kSm87MacrofeedV3GdnOutputBytes),
  };
  for (std::size_t left = 0U; left < 12U; ++left) {
    if (!ranges[left].valid) {
      return false;
    }
    for (std::size_t right = left + 1U; right < 12U; ++right) {
      if (sm87_macrofeed_v3_gdn_ranges_overlap(ranges[left],
                                               ranges[right])) {
        return false;
      }
    }
  }
  return true;
}

struct Sm87MacrofeedV3GdnKernelResources final {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int threads_per_block = 0;
  int physical_grid_ctas = 0;
};

struct Sm87MacrofeedV3GdnResources final {
  int binary_version = 0;
  Sm87MacrofeedV3GdnKernelResources convolution{};
  Sm87MacrofeedV3GdnKernelResources recurrence_epilogue{};
  bool kernels_compiled = false;
  bool exact_geometry = false;
  bool resource_gate_passed = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool sm87_macrofeed_v3_gdn_resources_valid(
    const Sm87MacrofeedV3GdnResources& resources) noexcept {
  const auto kernel_valid = [](const Sm87MacrofeedV3GdnKernelResources& item,
                               const int threads, const int ctas,
                               const int minimum_active_ctas) noexcept {
    return item.registers_per_thread > 0 && item.local_bytes == 0U &&
           item.maximum_threads_per_block >= threads &&
           item.active_blocks_per_sm >= minimum_active_ctas &&
           item.threads_per_block == threads &&
           item.physical_grid_ctas == ctas;
  };
  return resources.binary_version == 87 && resources.kernels_compiled &&
         resources.exact_geometry && resources.resource_gate_passed &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible &&
         kernel_valid(resources.convolution,
                      static_cast<int>(kSm87MacrofeedV3GdnConvThreads),
                      static_cast<int>(kSm87MacrofeedV3GdnConvCtas), 2) &&
         kernel_valid(
             resources.recurrence_epilogue,
             static_cast<int>(kSm87MacrofeedV3GdnRecurrenceThreads),
             static_cast<int>(kSm87MacrofeedV3GdnRecurrenceCtas), 3);
}

[[nodiscard]] int query_sm87_macrofeed_v3_gdn_p40_resources_cuda(
    Sm87MacrofeedV3GdnResources* resources) noexcept;

// Enqueues exactly one convolution kernel followed by eight ordered
// recurrence+epilogue macrochunks. It allocates, copies, and synchronizes
// nothing. A non-null cancellation signal is sampled at bounded C256 points;
// an observed non-zero value suppresses the next persistent-state/history
// publication and leaves partial request-local output uncommitted.
[[nodiscard]] int launch_sm87_macrofeed_v3_gdn_p40_cuda(
    const Sm87MacrofeedV3GdnP40Arguments& arguments) noexcept;

// Correctness-only C64 seam. It launches the identical convolution and
// recurrence+epilogue kernels with a shorter loop extent so the complete
// finite-precision surface can be compared against the established exact
// CUDA operators without allocating synthetic P40 tensors. It has no timing,
// route-selection, receipt, or production authority.
struct Sm87MacrofeedV3GdnC64OracleArguments final {
  const std::uint16_t* raw_qkv = nullptr;       // [64,10240]
  const std::uint16_t* a = nullptr;             // [64,48]
  const std::uint16_t* b = nullptr;             // [64,48]
  const std::uint16_t* conv_weight = nullptr;   // [10240,4]
  const std::uint16_t* a_log = nullptr;         // [48]
  const std::uint16_t* dt_bias = nullptr;       // [48]
  const std::uint16_t* norm_weight = nullptr;   // [128]
  const std::uint16_t* z = nullptr;             // [64,6144]
  std::uint16_t* conv_history = nullptr;        // [10240,3], in/out
  std::uint16_t* recurrent_state = nullptr;     // [48,128,128], in/out
  std::uint16_t* conv_qkv = nullptr;            // [64,10240]
  std::uint16_t* output = nullptr;              // [64,6144]
  std::uint32_t l2_epsilon_fp32_bits = 0U;
  std::uint32_t norm_epsilon_fp32_bits = 0U;
  void* cuda_stream = nullptr;
};

[[nodiscard]] int launch_sm87_macrofeed_v3_gdn_c64_oracle_cuda(
    const Sm87MacrofeedV3GdnC64OracleArguments& arguments) noexcept;

static_assert(kSm87MacrofeedV3GdnP40Tokens %
                      kSm87MacrofeedV3GdnMacrochunkTokens ==
                  0U);
static_assert(kSm87MacrofeedV3GdnMacrochunks == 8U);
static_assert(kSm87MacrofeedV3GdnConvCtas == 40U);
static_assert(kSm87MacrofeedV3GdnRecurrenceCtas == 48U);
static_assert(kSm87MacrofeedV3GdnPhysicalKernels == 9U);
static_assert(kSm87MacrofeedV3GdnRequiredPolicy == 0x1ffULL);

}  // namespace q3x::kernels
