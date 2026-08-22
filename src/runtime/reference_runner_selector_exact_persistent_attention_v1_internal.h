#pragma once

#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/prefill_execution_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace q3x::runtime::reference_runner_detail {

inline constexpr std::size_t
    kSelectorExactPersistentAttentionV1QueryTokens = 8U;
inline constexpr std::size_t
    kSelectorExactPersistentAttentionV1PersistentBlocksPerKvHead = 4U;
inline constexpr std::size_t
    kSelectorExactPersistentAttentionV1PersistentBlockCount =
        kSelectorExactPersistentAttentionV1PersistentBlocksPerKvHead * 4U;
inline constexpr std::size_t
    kSelectorExactPersistentAttentionV1P40PromptTokens = 40'000U;
inline constexpr std::size_t
    kSelectorExactPersistentAttentionV1P40PanelTokens = 8'000U;
inline constexpr std::size_t
    kSelectorExactPersistentAttentionV1P40PanelCount = 5U;
inline constexpr std::string_view
    kSelectorExactPersistentAttentionV1RouteIdentity =
        "selector-exact-persistent-attention-v1";

struct SelectorExactPersistentAttentionV1GroupSpan {
  std::uint32_t token_offset = 0U;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
};

// Candidate-only compact plan.  It proves that the incumbent arithmetic
// ledger contains no GroupQ64 span after the first GenericQT2 span, allowing
// the complete GenericQT2 suffix to be submitted once without changing any
// per-query arithmetic.  P40000 uses the retained v10 five-M8000 topology.
struct SelectorExactPersistentAttentionV1Plan {
  bool valid = false;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
  std::uint32_t arithmetic_span_count = 0U;
  std::uint32_t group_q64_span_count = 0U;
  std::uint32_t generic_qt2_span_count = 0U;
  std::array<SelectorExactPersistentAttentionV1GroupSpan, 2U>
      group_q64_spans{};
  std::uint32_t generic_suffix_token_offset = 0U;
  std::uint32_t generic_suffix_first_position = 0U;
  std::uint32_t generic_suffix_token_count = 0U;
  std::uint32_t physical_submission_count = 0U;
  std::uint32_t minimum_physical_submission_tokens = 0U;
  std::uint32_t maximum_physical_submission_tokens = 0U;
  std::uint32_t logical_panel_tokens = 0U;
};

[[nodiscard]] inline bool
is_selector_exact_persistent_attention_v1_plan_prompt_tokens(
    const std::size_t token_count) noexcept {
  return token_count == 513U || token_count == 4'096U ||
         token_count == 8'192U ||
         token_count == kSelectorExactPersistentAttentionV1P40PromptTokens;
}

[[nodiscard]] inline bool
is_selector_exact_persistent_attention_v1_runner_prompt_tokens(
    const std::size_t token_count) noexcept {
  return token_count == 513U || token_count == 4'096U ||
         token_count == 8'192U;
}

[[nodiscard]] inline SelectorExactPersistentAttentionV1Plan
make_selector_exact_persistent_attention_v1_plan(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  SelectorExactPersistentAttentionV1Plan plan;
  if (first_position != 0U ||
      !is_selector_exact_persistent_attention_v1_plan_prompt_tokens(
          token_count) ||
      token_count > kBulkCausalGqaMaximumSequenceLength) {
    return plan;
  }
  plan.first_position = static_cast<std::uint32_t>(first_position);
  plan.token_count = static_cast<std::uint32_t>(token_count);

  bool generic_suffix_started = false;
  std::size_t covered_tokens = 0U;
  const auto consume_span = [&](const std::size_t token_offset,
                                const std::size_t span_tokens) noexcept {
    if (span_tokens == 0U || token_offset != covered_tokens ||
        token_offset > token_count || span_tokens > token_count - token_offset) {
      return false;
    }
    const std::size_t absolute_position = first_position + token_offset;
    const FixedBulkCausalGqaPrefillTactic tactic =
        select_fixed_bulk_causal_gqa_prefill_tactic(absolute_position,
                                                    span_tokens);
    if (tactic == FixedBulkCausalGqaPrefillTactic::kGroupQ64V3) {
      if (generic_suffix_started ||
          plan.group_q64_span_count >= plan.group_q64_spans.size()) {
        return false;
      }
      plan.group_q64_spans[plan.group_q64_span_count++] =
          SelectorExactPersistentAttentionV1GroupSpan{
              static_cast<std::uint32_t>(token_offset),
              static_cast<std::uint32_t>(absolute_position),
              static_cast<std::uint32_t>(span_tokens)};
    } else if (tactic == FixedBulkCausalGqaPrefillTactic::kGenericQt2) {
      if (!generic_suffix_started) {
        generic_suffix_started = true;
        plan.generic_suffix_token_offset =
            static_cast<std::uint32_t>(token_offset);
        plan.generic_suffix_first_position =
            static_cast<std::uint32_t>(absolute_position);
      }
      if (token_offset !=
          static_cast<std::size_t>(plan.generic_suffix_token_offset) +
              plan.generic_suffix_token_count) {
        return false;
      }
      plan.generic_suffix_token_count +=
          static_cast<std::uint32_t>(span_tokens);
      ++plan.generic_qt2_span_count;
    } else {
      return false;
    }
    ++plan.arithmetic_span_count;
    covered_tokens += span_tokens;
    return true;
  };

  if (token_count == kSelectorExactPersistentAttentionV1P40PromptTokens) {
    for (std::size_t panel = 0U;
         panel < kSelectorExactPersistentAttentionV1P40PanelCount; ++panel) {
      const std::size_t panel_offset =
          panel * kSelectorExactPersistentAttentionV1P40PanelTokens;
      const LayerMajorPrefillArithmeticSpanLedger ledger =
          make_layer_major_prefill_arithmetic_span_ledger(
              kSelectorExactPersistentAttentionV1P40PanelTokens);
      if (!is_valid_layer_major_prefill_arithmetic_span_ledger(ledger)) {
        return {};
      }
      for (std::size_t index = 0U; index < ledger.span_count; ++index) {
        if (!consume_span(panel_offset + ledger.spans[index].token_offset,
                          ledger.spans[index].token_count)) {
          return {};
        }
      }
    }
  } else {
    const LayerMajorPrefillArithmeticSpanLedger ledger =
        make_layer_major_prefill_arithmetic_span_ledger(token_count);
    if (!is_valid_layer_major_prefill_arithmetic_span_ledger(ledger)) {
      return {};
    }
    for (std::size_t index = 0U; index < ledger.span_count; ++index) {
      if (!consume_span(ledger.spans[index].token_offset,
                        ledger.spans[index].token_count)) {
        return {};
      }
    }
  }

  plan.physical_submission_count =
      plan.group_q64_span_count +
      (plan.generic_suffix_token_count != 0U ? 1U : 0U);
  plan.logical_panel_tokens = static_cast<std::uint32_t>(token_count);
  for (std::size_t index = 0U; index < plan.group_q64_span_count; ++index) {
    const std::uint32_t physical_tokens =
        plan.group_q64_spans[index].token_count;
    if (plan.minimum_physical_submission_tokens == 0U ||
        physical_tokens < plan.minimum_physical_submission_tokens) {
      plan.minimum_physical_submission_tokens = physical_tokens;
    }
    if (physical_tokens > plan.maximum_physical_submission_tokens) {
      plan.maximum_physical_submission_tokens = physical_tokens;
    }
  }
  if (plan.minimum_physical_submission_tokens == 0U ||
      plan.generic_suffix_token_count <
          plan.minimum_physical_submission_tokens) {
    plan.minimum_physical_submission_tokens =
        plan.generic_suffix_token_count;
  }
  if (plan.generic_suffix_token_count >
      plan.maximum_physical_submission_tokens) {
    plan.maximum_physical_submission_tokens =
        plan.generic_suffix_token_count;
  }
  plan.valid = covered_tokens == token_count &&
               plan.group_q64_span_count != 0U &&
               plan.generic_qt2_span_count != 0U &&
               plan.generic_suffix_token_count != 0U &&
               static_cast<std::size_t>(plan.generic_suffix_token_offset) +
                       plan.generic_suffix_token_count ==
                   token_count &&
               plan.physical_submission_count ==
                   plan.group_q64_span_count + 1U &&
               plan.minimum_physical_submission_tokens != 0U &&
               plan.maximum_physical_submission_tokens >=
                   plan.minimum_physical_submission_tokens;
  return plan;
}

struct SelectorExactPersistentAttentionV1LaunchReceipt {
  SelectorExactPersistentAttentionV1Plan plan{};
  std::uint32_t group_q64_submissions = 0U;
  std::uint32_t generic_q8_suffix_submissions = 0U;
  std::uint32_t fallback_submissions = 0U;
  std::uint32_t persistent_ctas = 0U;
};

struct SelectorExactPersistentAttentionV1RouteReceipt {
  std::uint64_t panel_calls = 0U;
  std::uint64_t arithmetic_spans = 0U;
  std::uint64_t group_q64_submissions = 0U;
  std::uint64_t generic_qt2_spans = 0U;
  std::uint64_t generic_q8_suffix_submissions = 0U;
  std::uint64_t fallback_submissions = 0U;
  std::uint64_t persistent_ctas = 0U;
  std::uint32_t minimum_physical_submission_tokens = 0U;
  std::uint32_t maximum_physical_submission_tokens = 0U;
  std::uint32_t maximum_logical_panel_tokens = 0U;
};

struct SelectorExactPersistentAttentionV1Resources {
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int maximum_threads = 0;
  int active_blocks_per_multiprocessor = 0;
  int threads_per_block = 0;
};

[[nodiscard]] bool
exchange_selector_exact_persistent_attention_v1_for_test(bool enabled) noexcept;

[[nodiscard]] SelectorExactPersistentAttentionV1RouteReceipt
exchange_selector_exact_persistent_attention_v1_route_receipt_for_test(
    SelectorExactPersistentAttentionV1RouteReceipt receipt) noexcept;

[[nodiscard]] int
launch_selector_exact_persistent_attention_v1_cuda(
    const std::uint16_t* query, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate,
    std::size_t first_position, std::size_t token_count,
    std::uint16_t* output, SelectorExactPersistentAttentionV1LaunchReceipt* receipt,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_selector_exact_persistent_attention_v1_q8_generic_suffix_cuda(
    const std::uint16_t* query_suffix, const std::uint16_t* key_cache,
    const std::uint16_t* value_cache, const std::uint16_t* gate_suffix,
    std::size_t first_position, std::size_t token_count,
    std::uint16_t* output_suffix, void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
query_selector_exact_persistent_attention_v1_q8_resources_cuda(
    SelectorExactPersistentAttentionV1Resources* resources) noexcept;

}  // namespace q3x::runtime::reference_runner_detail
