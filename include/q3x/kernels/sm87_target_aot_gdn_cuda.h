#pragma once

#include "q3x/kernels/sm87_target_aot_gdn_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::kernels {

// CUDA realization boundary for the frozen exact-C16 target-AOT GDN plan.
// The kernel is intentionally specialized to the cold P40000 witness.  It
// accepts the current physical [T,96] producer allocation.  Every row is
// planar -- A[0:48] followed by B[48:96] -- and is never interpreted as
// element-wise AoS pairs.
inline constexpr std::size_t kSm87TargetAotGdnCudaTokenCount = 40'000U;
inline constexpr std::size_t kSm87TargetAotGdnCudaDynamicSharedBytes =
    100'252U;
inline constexpr std::size_t kSm87TargetAotGdnCudaPointerAlignment = 16U;
inline constexpr std::size_t kSm87TargetAotGdnCudaCancellationAlignment =
    alignof(std::uint32_t);
inline constexpr std::size_t kSm87TargetAotGdnCudaArgumentRangeCount = 10U;

inline constexpr std::uint64_t kSm87TargetAotGdnCudaRawQkvZBytes =
    static_cast<std::uint64_t>(kSm87TargetAotGdnCudaTokenCount) *
    kSm87TargetAotGdnRawQkvZChannels * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87TargetAotGdnCudaInterleavedAbBytes =
    static_cast<std::uint64_t>(kSm87TargetAotGdnCudaTokenCount) *
    kSm87TargetAotGdnAbChannels * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87TargetAotGdnCudaConvWeightBytes =
    kSm87TargetAotGdnConvWeightElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87TargetAotGdnCudaHeadVectorBytes =
    kSm87TargetAotGdnScalarHeadElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87TargetAotGdnCudaNormWeightBytes =
    kSm87TargetAotGdnNormWeightElements * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87TargetAotGdnCudaOutputBytes =
    static_cast<std::uint64_t>(kSm87TargetAotGdnCudaTokenCount) *
    kSm87TargetAotGdnOutputChannels * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::uint64_t kSm87TargetAotGdnCudaFinalConvHistoryBytes =
    kSm87TargetAotGdnTotalConvHistoryBytes;
inline constexpr std::uint64_t kSm87TargetAotGdnCudaFinalStateBytes =
    kSm87TargetAotGdnTotalStateBytes;

struct Sm87TargetAotGdnCudaArguments final {
  // Prompt-major [40000,16384] raw Q/K/V/Z projection.  Q/K/V pass through
  // the width-four convolution; Z bypasses it bit-exactly.
  const std::uint16_t* raw_qkvz = nullptr;

  // Prompt-major [40000,96] BF16.  The inner row is [A48,B48], not
  // [A0,B0,A1,B1,...].
  const std::uint16_t* interleaved_ab = nullptr;

  // Engine-lifetime exact inputs.
  const std::uint16_t* conv_weight = nullptr;  // [10240,4]
  const std::uint16_t* a_log = nullptr;        // [48]
  const std::uint16_t* dt_bias = nullptr;      // [48]
  const std::uint16_t* norm_weight = nullptr;  // [128], plain BF16

  // The authenticated plan fixes both scalars to 1e-6 exactly.  Raw bits
  // avoid a host arithmetic/conversion boundary in the launch contract.
  std::uint32_t l2_epsilon_fp32_bits = 0U;
  std::uint32_t norm_epsilon_fp32_bits = 0U;

  std::size_t first_position = 0U;
  std::size_t token_count = 0U;

  // Layer-local publication and the two prebound, still-unpublished request
  // transaction spans.  Cold execution does not read either final span.
  std::uint16_t* output = nullptr;                  // [40000,6144]
  std::uint16_t* final_conv_history = nullptr;      // [10240,3]
  std::uint16_t* final_recurrent_state = nullptr;   // [48,128,128]

  // Owner-owned mapped host/device control word.  The kernel only observes
  // it at the uniform beginning and end of each ordered C16.  A nonzero word
  // discards the entire unpublished transaction and suppresses final state
  // and history publication.
  const std::uint32_t* cancellation_signal = nullptr;

  // The authenticated executor supplies an owner-bound non-default stream.
  void* cuda_stream = nullptr;
};

struct Sm87TargetAotGdnCudaByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] constexpr Sm87TargetAotGdnCudaByteRange
sm87_target_aot_gdn_cuda_byte_range(const void* const pointer,
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

[[nodiscard]] constexpr bool sm87_target_aot_gdn_cuda_ranges_overlap(
    const Sm87TargetAotGdnCudaByteRange& left,
    const Sm87TargetAotGdnCudaByteRange& right) noexcept {
  return !left.valid || !right.valid ||
         (left.begin < right.end && right.begin < left.end);
}

[[nodiscard]] constexpr bool sm87_target_aot_gdn_cuda_pointer_aligned(
    const void* const pointer) noexcept {
  return pointer != nullptr &&
         reinterpret_cast<std::uintptr_t>(pointer) %
                 kSm87TargetAotGdnCudaPointerAlignment ==
             0U;
}

[[nodiscard]] constexpr bool sm87_target_aot_gdn_cuda_arguments_valid(
    const Sm87TargetAotGdnCudaArguments& arguments) noexcept {
  constexpr auto plan =
      sm87_target_aot_gdn_plan(kSm87TargetAotGdnCudaTokenCount);
  if (!plan.valid() || arguments.first_position != plan.first_position ||
      arguments.token_count != plan.token_count ||
      arguments.l2_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.norm_epsilon_fp32_bits !=
          kSm87TargetAotGdnEpsilonFp32Bits ||
      arguments.cuda_stream == nullptr) {
    return false;
  }

  const std::array<const void*, kSm87TargetAotGdnCudaArgumentRangeCount>
      pointers{{arguments.raw_qkvz,
                arguments.interleaved_ab,
                arguments.conv_weight,
                arguments.a_log,
                arguments.dt_bias,
                arguments.norm_weight,
                arguments.output,
                arguments.final_conv_history,
                arguments.final_recurrent_state,
                arguments.cancellation_signal}};
  for (std::size_t index = 0U; index + 1U < pointers.size(); ++index) {
    if (!sm87_target_aot_gdn_cuda_pointer_aligned(pointers[index])) {
      return false;
    }
  }
  if (arguments.cancellation_signal == nullptr ||
      reinterpret_cast<std::uintptr_t>(arguments.cancellation_signal) %
              kSm87TargetAotGdnCudaCancellationAlignment !=
          0U) {
    return false;
  }

  const std::array<Sm87TargetAotGdnCudaByteRange,
                   kSm87TargetAotGdnCudaArgumentRangeCount>
      ranges{{sm87_target_aot_gdn_cuda_byte_range(
                  arguments.raw_qkvz, kSm87TargetAotGdnCudaRawQkvZBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.interleaved_ab,
                  kSm87TargetAotGdnCudaInterleavedAbBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.conv_weight,
                  kSm87TargetAotGdnCudaConvWeightBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.a_log, kSm87TargetAotGdnCudaHeadVectorBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.dt_bias, kSm87TargetAotGdnCudaHeadVectorBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.norm_weight,
                  kSm87TargetAotGdnCudaNormWeightBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.output, kSm87TargetAotGdnCudaOutputBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.final_conv_history,
                  kSm87TargetAotGdnCudaFinalConvHistoryBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.final_recurrent_state,
                  kSm87TargetAotGdnCudaFinalStateBytes),
              sm87_target_aot_gdn_cuda_byte_range(
                  arguments.cancellation_signal,
                  sizeof(std::uint32_t))}};
  for (std::size_t first = 0U; first < ranges.size(); ++first) {
    if (!ranges[first].valid) {
      return false;
    }
    for (std::size_t second = first + 1U; second < ranges.size(); ++second) {
      if (sm87_target_aot_gdn_cuda_ranges_overlap(ranges[first],
                                                   ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

struct Sm87TargetAotGdnCudaResources final {
  std::size_t token_count = 0U;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  int physical_grid_ctas = 0;
  bool kernel_compiled = false;
  bool exact_owner_geometry = false;
  bool static_resources_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool
sm87_target_aot_gdn_cuda_resources_structurally_valid(
    const Sm87TargetAotGdnCudaResources& resources) noexcept {
  constexpr auto plan =
      sm87_target_aot_gdn_plan(kSm87TargetAotGdnCudaTokenCount);
  if (!plan.valid() ||
      resources.token_count != kSm87TargetAotGdnCudaTokenCount ||
      resources.static_resources_qualified ||
      resources.numerical_contract_qualified ||
      resources.production_dispatch_eligible) {
    return false;
  }
  if (!resources.kernel_compiled) {
    return resources.binary_version == 0 &&
           resources.registers_per_thread == 0 &&
           resources.static_shared_bytes == 0U &&
           resources.dynamic_shared_bytes == 0U &&
           resources.local_bytes == 0U &&
           resources.maximum_threads_per_block == 0 &&
           resources.active_blocks_per_sm == 0 &&
           resources.physical_grid_ctas == 0 &&
           !resources.exact_owner_geometry;
  }
  return resources.binary_version > 0 &&
         resources.registers_per_thread > 0 &&
         resources.dynamic_shared_bytes ==
             kSm87TargetAotGdnCudaDynamicSharedBytes &&
         resources.maximum_threads_per_block >=
             static_cast<int>(kSm87TargetAotGdnThreadsPerCta) &&
         resources.active_blocks_per_sm == 1 &&
         resources.physical_grid_ctas ==
             static_cast<int>(kSm87TargetAotGdnOwnerCtas) &&
         resources.exact_owner_geometry;
}

// Resource inspection is T0 evidence only.  The public launch sentinel never
// executes the body, including in an admission build; only the source-private
// complete-owner executor can reach the authenticated launch seam.
[[nodiscard]] int query_sm87_target_aot_gdn_cuda_resources(
    std::size_t token_count,
    Sm87TargetAotGdnCudaResources* resources) noexcept;

[[nodiscard]] int launch_sm87_target_aot_gdn_cuda(
    const Sm87TargetAotGdnCudaArguments& arguments) noexcept;

static_assert(kSm87TargetAotGdnCudaDynamicSharedBytes <= 102'400U);
static_assert(kSm87TargetAotGdnCudaRawQkvZBytes == 1'310'720'000ULL);
static_assert(kSm87TargetAotGdnCudaInterleavedAbBytes == 7'680'000ULL);
static_assert(kSm87TargetAotGdnCudaOutputBytes == 491'520'000ULL);
static_assert(kSm87TargetAotGdnCudaFinalConvHistoryBytes == 61'440ULL);
static_assert(kSm87TargetAotGdnCudaFinalStateBytes == 1'572'864ULL);

}  // namespace q3x::kernels
