#pragma once

#include "q3x/kernels/sm87_target_aot_attention_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime::sm87_target_aot_attention_execution_detail {

inline constexpr std::size_t kTargetP40TokenCount = 40'000U;
inline constexpr std::size_t kTargetP40QueryElements =
    kTargetP40TokenCount *
    kernels::kSm87TargetAotAttentionQueryHeads *
    kernels::kSm87TargetAotAttentionHeadDimension;
inline constexpr std::size_t kTargetP40KvElements =
    kTargetP40TokenCount * kernels::kSm87TargetAotAttentionKvHeads *
    kernels::kSm87TargetAotAttentionHeadDimension;
inline constexpr std::size_t kTargetP40QueryBytes =
    kTargetP40QueryElements * sizeof(std::uint16_t);
inline constexpr std::size_t kTargetP40KvBytes =
    kTargetP40KvElements * sizeof(std::uint16_t);

struct TargetP40ByteRange final {
  std::uintptr_t begin = 0U;
  std::uintptr_t end = 0U;
  bool valid = false;
};

[[nodiscard]] inline TargetP40ByteRange target_p40_byte_range(
    const void* const pointer, const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return pointer != nullptr && bytes != 0U &&
                 bytes <= std::numeric_limits<std::uintptr_t>::max() - begin
             ? TargetP40ByteRange{begin, begin + bytes, true}
             : TargetP40ByteRange{};
}

[[nodiscard]] inline bool target_p40_ranges_overlap(
    const TargetP40ByteRange& left,
    const TargetP40ByteRange& right) noexcept {
  return !left.valid || !right.valid ||
         (left.begin < right.end && right.begin < left.end);
}

// Processed Q/Gate and ordered NHD K/V are produced by the executor's five
// P8000 preprocess spans before this core is admitted.  This source-private
// contract accepts only the exact P40000 geometry and five disjoint,
// vector-aligned device ranges; it offers no generic panel fallback.
struct TargetP40Arguments final {
  const std::uint16_t* processed_query = nullptr;
  const std::uint16_t* processed_key = nullptr;
  const std::uint16_t* processed_value = nullptr;
  const std::uint16_t* processed_gate = nullptr;
  std::uint16_t* gated_output = nullptr;
  std::size_t token_count = 0U;
  std::int32_t device_ordinal = -1;
  void* cuda_stream = nullptr;
};

[[nodiscard]] inline bool target_p40_arguments_structurally_valid(
    const TargetP40Arguments& arguments) noexcept {
  if (arguments.token_count != kTargetP40TokenCount ||
      arguments.device_ordinal < 0) {
    return false;
  }
  const std::array<TargetP40ByteRange, 5U> ranges{{
      target_p40_byte_range(arguments.processed_query,
                            kTargetP40QueryBytes),
      target_p40_byte_range(arguments.processed_key, kTargetP40KvBytes),
      target_p40_byte_range(arguments.processed_value, kTargetP40KvBytes),
      target_p40_byte_range(arguments.processed_gate,
                            kTargetP40QueryBytes),
      target_p40_byte_range(arguments.gated_output,
                            kTargetP40QueryBytes)}};
  for (std::size_t first = 0U; first < ranges.size(); ++first) {
    if (!ranges[first].valid || (ranges[first].begin % 16U) != 0U) {
      return false;
    }
    for (std::size_t second = first + 1U; second < ranges.size(); ++second) {
      if (target_p40_ranges_overlap(ranges[first], ranges[second])) {
        return false;
      }
    }
  }
  return true;
}

struct TargetP40Resources final {
  std::size_t token_count = 0U;
  std::size_t query_rows = 0U;
  std::size_t kv_tokens = 0U;
  std::size_t pipeline_stages = 0U;
  std::size_t threads = 0U;
  std::size_t warps = 0U;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads_per_block = 0;
  int active_blocks_per_sm = 0;
  std::int32_t device_ordinal = -1;
  int device_sm_count = 0;
  std::size_t device_optin_shared_bytes = 0U;
  bool kernel_compiled = false;
  bool exact_p40000_only = false;
  bool cp_async_kv = false;
  bool kv_ping_pong = false;
  bool static_resources_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;
};

[[nodiscard]] constexpr bool target_p40_resources_structurally_valid(
    const TargetP40Resources& resources) noexcept {
  return resources.token_count == kTargetP40TokenCount &&
         resources.query_rows ==
             kernels::kSm87TargetAotAttentionQueryRows &&
         resources.kv_tokens ==
             kernels::kSm87TargetAotAttentionKvTokens &&
         resources.pipeline_stages ==
             kernels::kSm87TargetAotAttentionPipelineStages &&
         resources.threads == kernels::kSm87TargetAotAttentionThreads &&
         resources.warps == kernels::kSm87TargetAotAttentionWarps &&
         resources.binary_version == 87 &&
         resources.registers_per_thread > 0 &&
         resources.registers_per_thread <= 255 &&
         resources.static_shared_bytes == 0U &&
         resources.dynamic_shared_bytes ==
             kernels::kSm87TargetAotAttentionSharedBytes &&
         resources.local_bytes == 0U &&
         resources.maximum_threads_per_block >=
             static_cast<int>(kernels::kSm87TargetAotAttentionThreads) &&
         resources.active_blocks_per_sm == 1 &&
         resources.device_ordinal >= 0 && resources.device_sm_count == 16 &&
         resources.device_optin_shared_bytes >=
             kernels::kSm87TargetAotAttentionSharedBytes &&
         resources.kernel_compiled && resources.exact_p40000_only &&
         resources.cp_async_kv && resources.kv_ping_pong &&
         !resources.static_resources_qualified &&
         !resources.numerical_contract_qualified &&
         !resources.production_dispatch_eligible;
}

// Final Attention constituent body for the target executor.  Query/Gate/K/V
// preprocessing remains an executor responsibility; this entry is a pure
// exact-P40000 Q128/KV32 two-stage core and has no fallback, environment
// selector, request-time JIT, or autotune path.  The kernel and its acquiring
// implementation exist only under the complete target-AOT admission or the
// default-off Attention-v2 admission that needs this exact body as its CUDA
// differential control. The default build retains these declarations solely
// as fail-closed NotSupported sentinels.
[[nodiscard]] int query_q128_kv32_p40_two_stage_resources(
    std::int32_t device_ordinal,
    TargetP40Resources* resources) noexcept;

[[nodiscard]] int launch_q128_kv32_p40_two_stage(
    const TargetP40Arguments& arguments) noexcept;

static_assert(kernels::kSm87TargetAotAttentionSharedBytes == 128U * 1024U);
static_assert(kernels::kSm87TargetAotAttentionThreads == 256U);
static_assert(kernels::kSm87TargetAotAttentionWarps == 8U);
static_assert(kTargetP40QueryBytes == 491'520'000U);
static_assert(kTargetP40KvBytes == 81'920'000U);

}  // namespace q3x::runtime::sm87_target_aot_attention_execution_detail
