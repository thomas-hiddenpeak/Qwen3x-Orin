#pragma once

#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)

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

enum class SelectorExactPersistentAttentionV1PhysicalSubmissionTactic
    : std::uint8_t {
  kNone = 0U,
  kGroupQ64 = 1U,
  // Retained only so historical candidate receipts keep their stable byte.
  kPersistentGenericQt2Q8 = 2U,
  kGenericQt2 = 3U,
};
static_assert(static_cast<std::uint8_t>(
                  SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
                      kGroupQ64) ==
              kSelectorExactSpanAttentionV2GroupQ64PhysicalTactic);
static_assert(static_cast<std::uint8_t>(
                  SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
                      kPersistentGenericQt2Q8) ==
              kSelectorExactSpanAttentionV2PersistentGenericQt2Q8PhysicalTactic);
static_assert(static_cast<std::uint8_t>(
                  SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
                      kGenericQt2) ==
              kSelectorExactSpanAttentionV2GenericQt2PhysicalTactic);

struct SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt {
  SelectorExactPersistentAttentionV1PhysicalSubmissionTactic tactic =
      SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::kNone;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
};

[[nodiscard]] constexpr bool
selector_exact_persistent_attention_v1_physical_submission_equal(
    const SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt& left,
    const SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt& right)
    noexcept {
  return left.tactic == right.tactic &&
         left.first_position == right.first_position &&
         left.token_count == right.token_count;
}

// Candidate-only exact-span plan. P40000 retains the v10 five-M8000
// arithmetic ledgers and records every one of their physical spans in issue
// order. The first two spans select GroupQ64; all remaining spans select the
// incumbent GenericQT2 tactic independently, preserving its per-span
// finite-precision boundary instead of replacing it with one Q8 suffix.
struct SelectorExactPersistentAttentionV1Plan {
  bool valid = false;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
  std::uint32_t arithmetic_span_count = 0U;
  std::uint32_t group_q64_span_count = 0U;
  std::uint32_t generic_qt2_span_count = 0U;
  std::array<SelectorExactPersistentAttentionV1GroupSpan, 2U>
      group_q64_spans{};
  std::array<SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt,
             kSelectorExactSpanAttentionV2PhysicalSubmissionsPerLayer>
      physical_submissions{};
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
#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
  return token_count == 513U || token_count == 4'096U ||
         token_count == 8'192U ||
         token_count == kSelectorExactPersistentAttentionV1P40PromptTokens;
#else
  return token_count == 513U || token_count == 4'096U ||
         token_count == 8'192U;
#endif
}

#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
[[nodiscard]] inline bool
is_selector_exact_persistent_attention_v1_p40_runner_profile(
    const std::size_t request_capacity_tokens) noexcept {
  return request_capacity_tokens ==
         kSelectorExactPersistentAttentionV1P40RequestCapacityTokens;
}
#endif

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
        token_offset > token_count || span_tokens > token_count - token_offset ||
        plan.physical_submission_count >= plan.physical_submissions.size()) {
      return false;
    }
    const std::size_t absolute_position = first_position + token_offset;
    const FixedBulkCausalGqaPrefillTactic tactic =
        select_fixed_bulk_causal_gqa_prefill_tactic(absolute_position,
                                                    span_tokens);
    SelectorExactPersistentAttentionV1PhysicalSubmissionTactic
        physical_tactic =
            SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
                kNone;
    if (tactic == FixedBulkCausalGqaPrefillTactic::kGroupQ64V3) {
      if (generic_suffix_started ||
          plan.group_q64_span_count >= plan.group_q64_spans.size()) {
        return false;
      }
      physical_tactic =
          SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
              kGroupQ64;
      plan.group_q64_spans[plan.group_q64_span_count++] =
          SelectorExactPersistentAttentionV1GroupSpan{
              static_cast<std::uint32_t>(token_offset),
              static_cast<std::uint32_t>(absolute_position),
              static_cast<std::uint32_t>(span_tokens)};
    } else if (tactic == FixedBulkCausalGqaPrefillTactic::kGenericQt2) {
      physical_tactic =
          SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
              kGenericQt2;
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
    plan.physical_submissions[plan.physical_submission_count++] = {
        physical_tactic, static_cast<std::uint32_t>(absolute_position),
        static_cast<std::uint32_t>(span_tokens)};
    ++plan.arithmetic_span_count;
    covered_tokens += span_tokens;
    return true;
  };

  if (token_count == kSelectorExactPersistentAttentionV1P40PromptTokens) {
    for (const auto& expected :
         kSelectorExactSpanAttentionV2ExpectedPhysicalSpans) {
      if (!consume_span(expected.first_position, expected.token_count)) {
        return {};
      }
      const auto& actual =
          plan.physical_submissions[plan.physical_submission_count - 1U];
      if (static_cast<std::uint8_t>(actual.tactic) != expected.tactic ||
          actual.first_position != expected.first_position ||
          actual.token_count != expected.token_count) {
        return {};
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

  plan.logical_panel_tokens = static_cast<std::uint32_t>(token_count);
  for (std::size_t index = 0U; index < plan.physical_submission_count;
       ++index) {
    const std::uint32_t physical_tokens =
        plan.physical_submissions[index].token_count;
    if (plan.minimum_physical_submission_tokens == 0U ||
        physical_tokens < plan.minimum_physical_submission_tokens) {
      plan.minimum_physical_submission_tokens = physical_tokens;
    }
    if (physical_tokens > plan.maximum_physical_submission_tokens) {
      plan.maximum_physical_submission_tokens = physical_tokens;
    }
  }
  plan.valid = covered_tokens == token_count &&
               plan.group_q64_span_count != 0U &&
               plan.generic_qt2_span_count != 0U &&
               plan.generic_suffix_token_count != 0U &&
               static_cast<std::size_t>(plan.generic_suffix_token_offset) +
                       plan.generic_suffix_token_count ==
                   token_count &&
               plan.physical_submission_count ==
                   plan.arithmetic_span_count &&
               plan.physical_submission_count ==
                   plan.group_q64_span_count + plan.generic_qt2_span_count &&
               (token_count !=
                    kSelectorExactPersistentAttentionV1P40PromptTokens ||
                (plan.physical_submission_count ==
                     kSelectorExactSpanAttentionV2PhysicalSubmissionsPerLayer &&
                 plan.minimum_physical_submission_tokens ==
                     kSelectorExactSpanAttentionV2MinimumPhysicalTokens &&
                 plan.maximum_physical_submission_tokens ==
                     kSelectorExactSpanAttentionV2MaximumPhysicalTokens)) &&
               plan.minimum_physical_submission_tokens != 0U &&
               plan.maximum_physical_submission_tokens >=
                   plan.minimum_physical_submission_tokens;
  return plan;
}

[[nodiscard]] inline bool
selector_exact_persistent_attention_v1_physical_receipt_matches_plan(
    const SelectorExactPersistentAttentionV1Plan& plan,
    const std::uint32_t physical_submission_count,
    const std::array<
        SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt,
        kSelectorExactSpanAttentionV2PhysicalSubmissionsPerLayer>&
        physical_submissions) noexcept {
  if (!plan.valid ||
      physical_submission_count != plan.physical_submission_count ||
      physical_submission_count > physical_submissions.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < physical_submission_count; ++index) {
    if (!selector_exact_persistent_attention_v1_physical_submission_equal(
            physical_submissions[index], plan.physical_submissions[index])) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] inline bool
selector_exact_persistent_attention_v1_physical_receipts_equal(
    const std::array<
        SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt,
        kSelectorExactSpanAttentionV2PhysicalSubmissionsPerLayer>& left,
    const std::array<
        SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt,
        kSelectorExactSpanAttentionV2PhysicalSubmissionsPerLayer>& right,
    const std::uint32_t physical_submission_count) noexcept {
  if (physical_submission_count > left.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < physical_submission_count; ++index) {
    if (!selector_exact_persistent_attention_v1_physical_submission_equal(
            left[index], right[index])) {
      return false;
    }
  }
  return true;
}

#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
struct SelectorExactPersistentAttentionV1LayerSubmissionReceipt {
  std::uint32_t layer = 0U;
  std::uint32_t physical_submission_count = 0U;
  std::array<SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt,
             kSelectorExactSpanAttentionV2PhysicalSubmissionsPerLayer>
      physical_submissions{};
};
#endif

struct SelectorExactPersistentAttentionV1LaunchReceipt {
  SelectorExactPersistentAttentionV1Plan plan{};
  std::uint32_t group_q64_submissions = 0U;
  std::uint32_t generic_q8_suffix_submissions = 0U;
  std::uint32_t fallback_submissions = 0U;
  std::uint32_t persistent_ctas = 0U;
#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
  bool completed_physical_receipt = false;
  std::uint32_t physical_submission_count = 0U;
  std::array<SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt,
             kSelectorExactSpanAttentionV2PhysicalSubmissionsPerLayer>
      physical_submissions{};
#endif
};

struct SelectorExactPersistentAttentionV1RouteReceipt {
  // Candidate-only correctness repair receipts. Linear QKV/Z and full Q/K/V
  // each attest their Legacy global C512 geometry separately from Attention
  // submissions, so a completed real request cannot hide a projection
  // fallback behind the whole-prompt Attention receipt.
  std::uint64_t linear_qkvz_grouped_c512_submissions = 0U;
  std::uint64_t linear_qkvz_generic_c32_submissions = 0U;
  std::uint32_t linear_qkvz_completed_layers = 0U;
  std::uint64_t linear_qkvz_completed_layer_mask = 0U;
  std::uint64_t full_qkv_grouped_c512_submissions = 0U;
  std::uint64_t full_qkv_generic_c32_submissions = 0U;
  std::uint32_t full_qkv_completed_layers = 0U;
  std::uint64_t full_qkv_completed_layer_mask = 0U;
  // All 48 linear layers use the one-grid M16 arithmetic body that is
  // bitwise-equivalent to the Legacy A/B producer.
  std::uint32_t legacy_exact_bf16_ab_layers = 0U;
  // All 48 linear layers advance the recurrent core through the same
  // 78xC512 exact-span plus C64 warp-exact tail schedule as Legacy.
  std::uint32_t legacy_exact_gdn_layers = 0U;
  std::uint64_t legacy_exact_gdn_spans = 0U;
  // All 64 decoder layers restore the incumbent O projection's 78 canonical
  // whole-chunk C512 launches and one exact C64 tail.
  std::uint64_t legacy_exact_o_whole_chunk_c512_submissions = 0U;
  std::uint64_t legacy_exact_o_canonical_m64_submissions = 0U;
  std::uint32_t legacy_exact_o_completed_layers = 0U;
  // Canonical Legacy MLP schedule over all 64 layers. Gate and Up each own
  // 78 whole-chunk C512 launches plus two M32 tail launches. Down owns 78
  // C512 launches plus one M64 tail; SiLU and residual publish once per
  // logical C512/C64 span.
  std::uint64_t legacy_exact_mlp_gate_c512_submissions = 0U;
  std::uint64_t legacy_exact_mlp_gate_m32_tail_submissions = 0U;
  std::uint64_t legacy_exact_mlp_up_c512_submissions = 0U;
  std::uint64_t legacy_exact_mlp_up_m32_tail_submissions = 0U;
  std::uint64_t legacy_exact_mlp_silu_submissions = 0U;
  std::uint64_t legacy_exact_mlp_down_c512_submissions = 0U;
  std::uint64_t legacy_exact_mlp_down_m64_tail_submissions = 0U;
  std::uint64_t legacy_exact_mlp_residual_submissions = 0U;
  std::uint32_t legacy_exact_mlp_completed_layers = 0U;
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
#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
  bool completed_physical_receipt = false;
  std::uint32_t physical_submission_count_per_panel = 0U;
  std::array<SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt,
             kSelectorExactSpanAttentionV2PhysicalSubmissionsPerLayer>
      physical_submissions{};
  std::uint32_t issued_layer_count = 0U;
  std::uint32_t completed_layer_count = 0U;
  std::array<SelectorExactPersistentAttentionV1LayerSubmissionReceipt, 16U>
      completed_layers{};
#endif
};

#if defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
// One synchronized Legacy-C512 residual chunk at its absolute prompt rows.
// The callback must consume the device pointer synchronously; it is reused by
// the next tile. Ordinary OFF binaries contain neither this type nor TLS hook.
using ReferenceLegacyPrefillResidualChunkCallback = bool (*)(
    const std::uint16_t* residual_bf16, std::uint32_t first_position,
    std::uint32_t token_count, std::size_t elements, void* context) noexcept;

struct ReferenceLegacyPrefillResidualChunkHook {
  ReferenceLegacyPrefillResidualChunkCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] ReferenceLegacyPrefillResidualChunkHook
exchange_reference_legacy_prefill_residual_chunk_hook(
    ReferenceLegacyPrefillResidualChunkHook hook) noexcept;
#endif

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

#endif
