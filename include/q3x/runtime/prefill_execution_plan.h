#pragma once

#include "q3x/model/model_config.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string_view>

namespace q3x::runtime {

// AC-PREFILL-LAYERMAJOR-8K-v1 keeps the existing public C512 runner tile
// contract separate from its internal operator-panel capacity. This header is
// currently a pure-host, unbound topology contract: it contains no launcher,
// device pointer, stream, event, allocation, or production route selector.
inline constexpr std::uint32_t kPrefillPhysicalSegmentMaximumTokens = 512U;
inline constexpr std::uint32_t kPrefillPhysicalSegmentM256Tokens = 256U;
inline constexpr std::uint32_t kPrefillPhysicalSegmentM64Tokens = 64U;
inline constexpr std::uint32_t kPrefillPhysicalSegmentM32Tokens = 32U;
inline constexpr std::uint32_t kPrefillPhysicalSegmentTailMaximumTokens = 31U;
inline constexpr std::uint32_t kLayerMajorPrefillLegacyPublicTileTokens =
    kPrefillPhysicalSegmentMaximumTokens;
inline constexpr std::uint32_t kLayerMajorPrefillOperatorPanelTokens =
    8'192U;
inline constexpr std::uint32_t
    kLayerMajorPrefillTrueLargeMPartialPanelTokens = 7'712U;
// Test-only P40 architecture target. The candidate deliberately admits one
// exact cold-prompt shape so a host plan cannot silently generalize an
// unprofiled full-M scheduling contract to other prompt lengths.
inline constexpr std::uint32_t kLayerMajorPrefillPromptWideP40Tokens =
    40'000U;
// The complete whole-core route deliberately uses five equal M8000 panels.
// M8000 is M64-aligned and remains below the C8192 operator-panel ceiling,
// allowing every aligned FP8 fill/drain projection to retain one physical
// launch without changing any incumbent panel geometry.
inline constexpr std::uint32_t kLayerMajorPrefillPromptWideP40PanelTokens =
    8'000U;
inline constexpr std::size_t kLayerMajorPrefillPromptWideP40PanelCount = 5U;
inline constexpr std::uint32_t kLayerMajorPrefillLayerWideMlpP40Tokens =
    kLayerMajorPrefillPromptWideP40Tokens;
// The API must retain one position beyond the exact P40000 prompt so
// max_tokens=1 is a valid delivered request. This capacity is not an MLP M;
// full-M typed views remain exactly 40000 rows.
inline constexpr std::uint32_t
    kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens = 40'001U;
inline constexpr std::uint32_t
    kLayerMajorPrefillPromptWideP40RequestCapacityTokens =
        kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens;
// The projection-reset candidate preserves the same exact P40000 request
// boundary, but replaces the old five-panel FP8 physical inventory with one
// grouped full-prompt input launch and one full-prompt O launch per layer.
// Tensor-role hits remain logical route evidence: linear layers own QKV, Z,
// and O; full-Attention layers own Q, K, V, and O.
inline constexpr std::size_t
    kLayerMajorPrefillProjectionResetFp8GroupedInputLaunchesPerLayer = 1U;
inline constexpr std::size_t
    kLayerMajorPrefillProjectionResetFp8OutputLaunchesPerLayer = 1U;
inline constexpr std::size_t
    kLayerMajorPrefillProjectionResetFp8PhysicalLaunchesPerRequest = 128U;
inline constexpr std::size_t
    kLayerMajorPrefillProjectionResetFp8TensorRoleHitsPerRequest = 208U;
inline constexpr std::size_t
    kLayerMajorPrefillProjectionResetNvFp4GateUpLaunchesPerLayer = 1U;
inline constexpr std::size_t
    kLayerMajorPrefillProjectionResetNvFp4DownLaunchesPerLayer = 1U;
inline constexpr std::size_t
    kLayerMajorPrefillProjectionResetNvFp4PhysicalLaunchesPerRequest = 128U;
inline constexpr std::uint32_t kLayerMajorPrefillLayerWideMlpAlignmentTokens =
    64U;
inline constexpr std::uint32_t kLayerMajorPrefillMaximumSequenceTokens =
    262'144U;
inline constexpr std::size_t kLayerMajorPrefillLayerCount = 64U;
inline constexpr std::size_t kLayerMajorPrefillLinearLayerCount = 48U;
inline constexpr std::size_t kLayerMajorPrefillFullLayerCount = 16U;
inline constexpr std::size_t kLayerMajorPrefillMaximumPanelCount =
    (kLayerMajorPrefillMaximumSequenceTokens +
     kLayerMajorPrefillOperatorPanelTokens - 1U) /
    kLayerMajorPrefillOperatorPanelTokens;

static_assert(kLayerMajorPrefillLegacyPublicTileTokens == 512U);
static_assert(kLayerMajorPrefillOperatorPanelTokens == 8'192U);
static_assert(kLayerMajorPrefillTrueLargeMPartialPanelTokens == 7'712U);
static_assert(kLayerMajorPrefillLayerWideMlpP40Tokens %
                  kLayerMajorPrefillLayerWideMlpAlignmentTokens ==
              0U);
static_assert(kLayerMajorPrefillPromptWideP40PanelTokens %
                  kLayerMajorPrefillLayerWideMlpAlignmentTokens ==
              0U);
static_assert(kLayerMajorPrefillPromptWideP40PanelTokens <=
              kLayerMajorPrefillOperatorPanelTokens);
static_assert(kLayerMajorPrefillPromptWideP40PanelTokens *
                      kLayerMajorPrefillPromptWideP40PanelCount ==
                  kLayerMajorPrefillPromptWideP40Tokens);
static_assert(
    kLayerMajorPrefillProjectionResetFp8PhysicalLaunchesPerRequest ==
    kLayerMajorPrefillLayerCount *
        (kLayerMajorPrefillProjectionResetFp8GroupedInputLaunchesPerLayer +
         kLayerMajorPrefillProjectionResetFp8OutputLaunchesPerLayer));
static_assert(
    kLayerMajorPrefillProjectionResetFp8TensorRoleHitsPerRequest ==
    kLayerMajorPrefillLinearLayerCount * 3U +
        kLayerMajorPrefillFullLayerCount * 4U);
static_assert(
    kLayerMajorPrefillProjectionResetNvFp4PhysicalLaunchesPerRequest ==
    kLayerMajorPrefillLayerCount *
        (kLayerMajorPrefillProjectionResetNvFp4GateUpLaunchesPerLayer +
         kLayerMajorPrefillProjectionResetNvFp4DownLaunchesPerLayer));
static_assert(kLayerMajorPrefillMaximumPanelCount == 32U);

[[nodiscard]] constexpr bool is_nvfp4_true_large_m_prefill_panel_tokens(
    const std::size_t token_count) noexcept {
  return token_count == kLayerMajorPrefillOperatorPanelTokens ||
         token_count == kLayerMajorPrefillTrueLargeMPartialPanelTokens;
}

// Engine-lifetime full-Attention ownership for the layer-major route.  The
// incumbent keeps the established C512 arithmetic spans and their exact
// fixed selector.  The default-off architecture tactics give one native
// grouped online-softmax launch ownership of the complete logical operator
// panel.  Q64 and Q128-v4 remain distinct accuracy-unqualified tactics so
// route evidence can never relabel one as the other.  This is part of the
// sealed plan, never a request-time or environment selector.
enum class LayerMajorPrefillFullAttentionTactic : std::uint8_t {
  kExactSegmentedC512 = 0,
  kNativeGroupQ64Panel,
  kNativeGroupQ128V4Panel,
  kNativeFlashInferExactPanel,
  // Complete cold P40000 Attention ownership. This is distinct from the
  // logical-panel FlashInfer tactic and cannot be selected by it implicitly.
  kNativeFlashInferExactWholePrompt,
};

[[nodiscard]] constexpr bool
is_valid_layer_major_prefill_full_attention_tactic(
    const LayerMajorPrefillFullAttentionTactic tactic) noexcept {
  return tactic ==
             LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512 ||
         tactic ==
             LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel ||
         tactic == LayerMajorPrefillFullAttentionTactic::
                       kNativeGroupQ128V4Panel ||
         tactic == LayerMajorPrefillFullAttentionTactic::
                       kNativeFlashInferExactPanel ||
         tactic == LayerMajorPrefillFullAttentionTactic::
                       kNativeFlashInferExactWholePrompt;
}

[[nodiscard]] constexpr std::string_view to_string(
    const LayerMajorPrefillFullAttentionTactic tactic) noexcept {
  switch (tactic) {
    case LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512:
      return "exact-segmented";
    case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel:
      return "native-group-q64-panel";
    case LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel:
      return "native-group-q128-v4-panel";
    case LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactPanel:
      return "native-flashinfer-exact-panel";
    case LayerMajorPrefillFullAttentionTactic::
        kNativeFlashInferExactWholePrompt:
      return "native-flashinfer-exact-whole-prompt";
  }
  return "unknown";
}

// Engine-lifetime projection ownership for the layer-major route. The exact
// incumbent retains the authenticated C512 arithmetic ledger. The segmented
// Marlin operator-panel value is an explicit, default-off dependency screen:
// it submits each logical FP8/NVFP4 projection through the existing wrapper,
// whose large-N kernels still segment at no more than M1024, while leaving
// BF16 A/B on the exact contract. The native quantized large-M value reuses
// the frozen exact Marlin sidecars and their bound reduction/lock workspaces:
// an authenticated M8192 panel is one physical Marlin launch per logical
// projection, while every partial panel retains the complete oracle span
// ledger and workspace/interleave sequence. Both candidates remain
// explicitly selected, default-off, and accuracy-unqualified.
// The true-large-M NVFP4 value is a third, independently witnessed candidate:
// Gate+Up and Down are one coupled binding package and both must own the
// complete M8192 or M7712 logical panel. It never inherits the old NVFP4
// Marlin partial-panel fallback. FP8 keeps the authenticated native-large-M
// Marlin policy until its own shape-specific successor is admitted.
// Selection is sealed into the bound plan and never changes per request.
enum class LayerMajorPrefillProjectionTactic : std::uint8_t {
  kExactSegmentedC512 = 0,
  kSegmentedMarlinOperatorPanel,
  kNativeQuantizedLargeMOperatorPanel,
  kNativeNvfp4TrueLargeMOperatorPanel,
  kNativeNvfp4G2D2LargeMOperatorPanel,
  // Exact-P40000 BUILD_TESTING-only route. FP8 Attention/GDN projections
  // remain panel-owned; post-attention norm and the persistent fused NVFP4
  // GateUp+SiLU / Down+residual pair execute once over the complete layer.
  kNativeNvfp4PersistentP40LayerWideMlp,
  // Exact-P40000 whole-core architecture tactic. It owns the complete
  // 5xM8000 fill -> prompt-wide core -> 5xM8000 drain/O -> persistent MLP
  // schedule and must never inherit the layer-wide-MLP-only identity.
  kNativePromptWideP40WholeCore,
  // Exact-P40000 projection reset. FP8 owns one grouped full-prompt input
  // launch plus one full-prompt O launch per layer, while NVFP4 owns one
  // unsplit Gate+Up and one unsplit Down projection. This is a distinct,
  // default-off route and may not be reported as the earlier whole-core
  // tactic.
  kNativePromptWideP40ProjectionReset,
};

[[nodiscard]] constexpr bool is_valid_layer_major_prefill_projection_tactic(
    const LayerMajorPrefillProjectionTactic tactic) noexcept {
  return tactic ==
             LayerMajorPrefillProjectionTactic::kExactSegmentedC512 ||
         tactic ==
             LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel ||
         tactic == LayerMajorPrefillProjectionTactic::
                       kNativeQuantizedLargeMOperatorPanel ||
         tactic == LayerMajorPrefillProjectionTactic::
                       kNativeNvfp4TrueLargeMOperatorPanel ||
         tactic == LayerMajorPrefillProjectionTactic::
                       kNativeNvfp4G2D2LargeMOperatorPanel ||
         tactic == LayerMajorPrefillProjectionTactic::
                       kNativeNvfp4PersistentP40LayerWideMlp ||
         tactic == LayerMajorPrefillProjectionTactic::
                       kNativePromptWideP40WholeCore ||
         tactic == LayerMajorPrefillProjectionTactic::
                       kNativePromptWideP40ProjectionReset;
}

[[nodiscard]] constexpr std::string_view to_string(
    const LayerMajorPrefillProjectionTactic tactic) noexcept {
  switch (tactic) {
    case LayerMajorPrefillProjectionTactic::kExactSegmentedC512:
      return "exact-segmented";
    case LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel:
      return "segmented-marlin-operator-panel";
    case LayerMajorPrefillProjectionTactic::
        kNativeQuantizedLargeMOperatorPanel:
      return "native-quantized-large-m-operator-panel";
    case LayerMajorPrefillProjectionTactic::
        kNativeNvfp4TrueLargeMOperatorPanel:
      return "native-nvfp4-true-large-m-operator-panel";
    case LayerMajorPrefillProjectionTactic::
        kNativeNvfp4G2D2LargeMOperatorPanel:
      return "native-nvfp4-g2-d2-large-m-operator-panel";
    case LayerMajorPrefillProjectionTactic::
        kNativeNvfp4PersistentP40LayerWideMlp:
      return "native-nvfp4-persistent-p40-layer-wide-mlp";
    case LayerMajorPrefillProjectionTactic::kNativePromptWideP40WholeCore:
      return "native-prompt-wide-p40-whole-core";
    case LayerMajorPrefillProjectionTactic::
        kNativePromptWideP40ProjectionReset:
      return "native-prompt-wide-p40-projection-reset";
  }
  return "unknown";
}

// Preserve full-capacity work while preventing a final one-token panel or
// physical segment.  Once only the final full-capacity unit plus its tail
// remain, split that suffix into ceil/floor halves.  This keeps the minimum
// number of units, retains every earlier full unit, and prevents the exact
// optimized routes from being defeated by a pathological scalar tail.
[[nodiscard]] constexpr std::size_t
next_layer_major_prefill_operator_panel_token_count(
    const std::size_t remaining_prompt_tokens) noexcept {
  if (remaining_prompt_tokens <= kLayerMajorPrefillOperatorPanelTokens) {
    return remaining_prompt_tokens;
  }
  if (remaining_prompt_tokens <
      2U * kLayerMajorPrefillOperatorPanelTokens) {
    return (remaining_prompt_tokens + 1U) / 2U;
  }
  return kLayerMajorPrefillOperatorPanelTokens;
}

// Canonical exact-route subdivision retained for the legacy scheduling
// contract and its historical evidence.
[[nodiscard]] constexpr std::size_t
next_prefill_physical_segment_token_count(
    const std::size_t remaining_panel_tokens) noexcept {
  if (remaining_panel_tokens >= kPrefillPhysicalSegmentMaximumTokens) {
    return kPrefillPhysicalSegmentMaximumTokens;
  }
  if (remaining_panel_tokens >= kPrefillPhysicalSegmentM256Tokens) {
    return kPrefillPhysicalSegmentM256Tokens;
  }
  if (remaining_panel_tokens >= kPrefillPhysicalSegmentM64Tokens) {
    return kPrefillPhysicalSegmentM64Tokens;
  }
  if (remaining_panel_tokens >= kPrefillPhysicalSegmentM32Tokens) {
    return kPrefillPhysicalSegmentM32Tokens;
  }
  return remaining_panel_tokens;
}

[[nodiscard]] constexpr bool is_prefill_physical_segment_token_count(
    const std::size_t token_count) noexcept {
  return token_count == kPrefillPhysicalSegmentMaximumTokens ||
         token_count == kPrefillPhysicalSegmentM256Tokens ||
         token_count == kPrefillPhysicalSegmentM64Tokens ||
         token_count == kPrefillPhysicalSegmentM32Tokens ||
         (token_count != 0U &&
          token_count <= kPrefillPhysicalSegmentTailMaximumTokens);
}

// The layer-major compatibility executor accepts every C1..C512 geometry.
// Keep earlier C512 work full and split only the final C512-plus-tail suffix
// into ceil/floor halves. This avoids a scalar layer tail without nearly
// doubling the masked Marlin launch count over the whole panel.
[[nodiscard]] constexpr std::size_t
next_layer_major_prefill_physical_segment_token_count(
    const std::size_t remaining_panel_tokens) noexcept {
  if (remaining_panel_tokens <= kPrefillPhysicalSegmentMaximumTokens) {
    return remaining_panel_tokens;
  }
  if (remaining_panel_tokens <
      2U * kPrefillPhysicalSegmentMaximumTokens) {
    return (remaining_panel_tokens + 1U) / 2U;
  }
  return kPrefillPhysicalSegmentMaximumTokens;
}

[[nodiscard]] constexpr bool
is_layer_major_prefill_physical_segment_token_count(
    const std::size_t token_count) noexcept {
  return token_count != 0U &&
         token_count <= kPrefillPhysicalSegmentMaximumTokens;
}

// Exact arithmetic is owned by the historical C512-balanced compatibility
// route, even when a layer is submitted as one larger operator panel.  The
// ledger makes those physical ownership boundaries explicit so every
// projection and stateful operator can retain the oracle's masked-tail
// specialization sequence without giving up panel-wide storage and
// scheduling.
inline constexpr std::size_t kLayerMajorPrefillMaximumArithmeticSpanCount =
    (kLayerMajorPrefillOperatorPanelTokens +
     kPrefillPhysicalSegmentMaximumTokens - 1U) /
    kPrefillPhysicalSegmentMaximumTokens;

struct LayerMajorPrefillArithmeticSpan {
  std::uint32_t token_offset = 0U;
  std::uint32_t token_count = 0U;
};

struct LayerMajorPrefillArithmeticSpanLedger {
  std::array<LayerMajorPrefillArithmeticSpan,
             kLayerMajorPrefillMaximumArithmeticSpanCount>
      spans{};
  std::size_t span_count = 0U;
  std::uint32_t token_count = 0U;
};

[[nodiscard]] constexpr LayerMajorPrefillArithmeticSpanLedger
make_layer_major_prefill_arithmetic_span_ledger(
    const std::size_t panel_token_count) noexcept {
  LayerMajorPrefillArithmeticSpanLedger ledger;
  if (panel_token_count == 0U ||
      panel_token_count > kLayerMajorPrefillOperatorPanelTokens) {
    return ledger;
  }

  std::size_t offset = 0U;
  std::size_t remaining = panel_token_count;
  while (remaining != 0U &&
         ledger.span_count < ledger.spans.size()) {
    const std::size_t span_token_count =
        next_layer_major_prefill_physical_segment_token_count(remaining);
    if (!is_layer_major_prefill_physical_segment_token_count(
            span_token_count) ||
        span_token_count > remaining) {
      return {};
    }
    ledger.spans[ledger.span_count++] = LayerMajorPrefillArithmeticSpan{
        static_cast<std::uint32_t>(offset),
        static_cast<std::uint32_t>(span_token_count)};
    offset += span_token_count;
    remaining -= span_token_count;
  }
  if (remaining != 0U || offset != panel_token_count) {
    return {};
  }
  ledger.token_count = static_cast<std::uint32_t>(panel_token_count);
  return ledger;
}

[[nodiscard]] constexpr bool
is_valid_layer_major_prefill_arithmetic_span_ledger(
    const LayerMajorPrefillArithmeticSpanLedger& ledger) noexcept {
  if (ledger.token_count == 0U ||
      ledger.token_count > kLayerMajorPrefillOperatorPanelTokens ||
      ledger.span_count == 0U ||
      ledger.span_count > ledger.spans.size()) {
    return false;
  }
  std::size_t expected_offset = 0U;
  for (std::size_t index = 0U; index < ledger.span_count; ++index) {
    const LayerMajorPrefillArithmeticSpan& span = ledger.spans[index];
    if (expected_offset >= ledger.token_count) {
      return false;
    }
    const std::size_t remaining = ledger.token_count - expected_offset;
    if (span.token_offset != expected_offset ||
        !is_layer_major_prefill_physical_segment_token_count(
            span.token_count) ||
        span.token_count !=
            next_layer_major_prefill_physical_segment_token_count(
                remaining)) {
      return false;
    }
    expected_offset += span.token_count;
  }
  return expected_offset == ledger.token_count;
}

enum class PrefillBf16AbArithmeticTactic : std::uint8_t {
  kEstablishedM32ProjectionPair = 0,
  kPromptWideP40SingleGrid,
};

enum class PrefillFp8ArithmeticTactic : std::uint8_t {
  kOracleSpanMarlin = 0,
  kM8192SingleBulkOtherwiseOracleSpanMarlin,
  kOperatorPanelSegmentedMarlin,
  kP8000FillDrainSingleBulk,
  kP40000GroupedInputAndOutputSingleBulk,
};

enum class PrefillNvFp4ArithmeticTactic : std::uint8_t {
  kOracleSpanGateSiluDownSequence = 0,
  kM8192SingleBulkOtherwiseOracleSpanGateSiluDown,
  kOperatorPanelGateSiluDownSequence,
  kM8192OrM7712TrueLargeMPanelGateSiluDown,
  kP40000PersistentGateUpSiluDownResidual,
};

enum class PrefillGdnArithmeticTactic : std::uint8_t {
  kOracleSpanWholeRawQkv = 0,
  kPromptWideP40ChunkGraph,
};

enum class PrefillAttentionPreprocessArithmeticTactic : std::uint8_t {
  kOracleSpanC16FixedReference256 = 0,
  kP8000FillWholePromptFlashInferDrain,
};

struct LayerMajorPrefillArithmeticContract {
  std::uint32_t version = 1U;
  PrefillBf16AbArithmeticTactic bf16_ab =
      PrefillBf16AbArithmeticTactic::kEstablishedM32ProjectionPair;
  PrefillFp8ArithmeticTactic fp8 =
      PrefillFp8ArithmeticTactic::kOracleSpanMarlin;
  PrefillNvFp4ArithmeticTactic nvfp4 =
      PrefillNvFp4ArithmeticTactic::kOracleSpanGateSiluDownSequence;
  PrefillGdnArithmeticTactic gdn =
      PrefillGdnArithmeticTactic::kOracleSpanWholeRawQkv;
  PrefillAttentionPreprocessArithmeticTactic attention_preprocess =
      PrefillAttentionPreprocessArithmeticTactic::
          kOracleSpanC16FixedReference256;
  bool reset_fp8_locks_per_projection_span = true;
  bool nvfp4_interleaves_gate_silu_down_per_span = true;
  bool nvfp4_down_reuses_gate_up_locks = true;
  bool nvfp4_residual_follows_down_per_span = true;
  // The four fields above define the partial-panel oracle sequence. The
  // fields below define the conditional M8192 override and therefore may be
  // true at the same time without weakening the partial-panel contract.
  bool m8192_single_bulk_projection = false;
  bool m8192_fp8_resets_locks_once = false;
  bool m8192_nvfp4_uses_independent_down_workspace = false;
  bool m8192_nvfp4_residual_once_after_bulk = false;
  bool nvfp4_true_large_m_m8192 = false;
  bool nvfp4_true_large_m_m7712 = false;
  bool nvfp4_gate_up_down_coupled = false;
  bool p40000_post_attention_norm_prompt_wide = false;
  bool p40000_persistent_gate_up_silu = false;
  bool p40000_persistent_down_residual = false;
  bool environment_independent = true;
  bool p8000_fp8_fill_drain_single_bulk = false;
  bool p40000_bf16_ab_prompt_wide = false;
  bool p40000_gdn_prompt_wide = false;
  bool p40000_flashinfer_whole_prompt = false;
};

inline constexpr LayerMajorPrefillArithmeticContract
    kLayerMajorPrefillExactArithmeticContract{};

inline constexpr LayerMajorPrefillArithmeticContract
    kLayerMajorPrefillExactMarlinM8192ArithmeticContract{
        2U,
        PrefillBf16AbArithmeticTactic::kEstablishedM32ProjectionPair,
        PrefillFp8ArithmeticTactic::
            kM8192SingleBulkOtherwiseOracleSpanMarlin,
        PrefillNvFp4ArithmeticTactic::
            kM8192SingleBulkOtherwiseOracleSpanGateSiluDown,
        PrefillGdnArithmeticTactic::kOracleSpanWholeRawQkv,
        PrefillAttentionPreprocessArithmeticTactic::
            kOracleSpanC16FixedReference256,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        false,
        false,
        false,
        true};

inline constexpr LayerMajorPrefillArithmeticContract
    kLayerMajorPrefillSegmentedMarlinArithmeticContract{
        3U,
        PrefillBf16AbArithmeticTactic::kEstablishedM32ProjectionPair,
        PrefillFp8ArithmeticTactic::kOperatorPanelSegmentedMarlin,
        PrefillNvFp4ArithmeticTactic::kOperatorPanelGateSiluDownSequence,
        PrefillGdnArithmeticTactic::kOracleSpanWholeRawQkv,
        PrefillAttentionPreprocessArithmeticTactic::
            kOracleSpanC16FixedReference256,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true};

inline constexpr LayerMajorPrefillArithmeticContract
    kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract{
        4U,
        PrefillBf16AbArithmeticTactic::kEstablishedM32ProjectionPair,
        PrefillFp8ArithmeticTactic::
            kM8192SingleBulkOtherwiseOracleSpanMarlin,
        PrefillNvFp4ArithmeticTactic::
            kM8192OrM7712TrueLargeMPanelGateSiluDown,
        PrefillGdnArithmeticTactic::kOracleSpanWholeRawQkv,
        PrefillAttentionPreprocessArithmeticTactic::
            kOracleSpanC16FixedReference256,
        true,
        false,
        false,
        false,
        true,
        true,
        false,
        true,
        true,
        true,
        true,
        false,
        false,
        false,
        true};

inline constexpr LayerMajorPrefillArithmeticContract
    kLayerMajorPrefillPersistentP40NvFp4ArithmeticContract{
        5U,
        PrefillBf16AbArithmeticTactic::kEstablishedM32ProjectionPair,
        PrefillFp8ArithmeticTactic::
            kM8192SingleBulkOtherwiseOracleSpanMarlin,
        PrefillNvFp4ArithmeticTactic::
            kP40000PersistentGateUpSiluDownResidual,
        PrefillGdnArithmeticTactic::kOracleSpanWholeRawQkv,
        PrefillAttentionPreprocessArithmeticTactic::
            kOracleSpanC16FixedReference256,
        true,
        false,
        false,
        false,
        true,
        true,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true};

inline constexpr LayerMajorPrefillArithmeticContract
    kLayerMajorPrefillPromptWideP40WholeCoreArithmeticContract{
        6U,
        PrefillBf16AbArithmeticTactic::kPromptWideP40SingleGrid,
        PrefillFp8ArithmeticTactic::kP8000FillDrainSingleBulk,
        PrefillNvFp4ArithmeticTactic::
            kP40000PersistentGateUpSiluDownResidual,
        PrefillGdnArithmeticTactic::kPromptWideP40ChunkGraph,
        PrefillAttentionPreprocessArithmeticTactic::
            kP8000FillWholePromptFlashInferDrain,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true,
        true};

// The projection-reset route changes the physical FP8 arithmetic boundary:
// every layer owns one grouped M40000 input launch and one M40000 O launch.
// Keep that identity separate from v6, whose FP8 projections remain M8000
// fill/drain bulks, even though the two routes share the surrounding BF16,
// GDN, Attention, and persistent-NVFP4 arithmetic tactics.
inline constexpr LayerMajorPrefillArithmeticContract
    kLayerMajorPrefillPromptWideP40ProjectionResetArithmeticContract{
        7U,
        PrefillBf16AbArithmeticTactic::kPromptWideP40SingleGrid,
        PrefillFp8ArithmeticTactic::
            kP40000GroupedInputAndOutputSingleBulk,
        PrefillNvFp4ArithmeticTactic::
            kP40000PersistentGateUpSiluDownResidual,
        PrefillGdnArithmeticTactic::kPromptWideP40ChunkGraph,
        PrefillAttentionPreprocessArithmeticTactic::
            kP8000FillWholePromptFlashInferDrain,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        false,
        true,
        true,
        true,
        true,
        true,
        false,
        true,
        true,
        true};

[[nodiscard]] constexpr bool is_valid_layer_major_prefill_arithmetic_contract(
    const LayerMajorPrefillArithmeticContract& contract) noexcept {
  const bool common =
      contract.environment_independent;
  const bool legacy_common =
      contract.bf16_ab == PrefillBf16AbArithmeticTactic::
                                kEstablishedM32ProjectionPair &&
      contract.gdn == PrefillGdnArithmeticTactic::kOracleSpanWholeRawQkv &&
      contract.attention_preprocess ==
          PrefillAttentionPreprocessArithmeticTactic::
              kOracleSpanC16FixedReference256 &&
      !contract.p8000_fp8_fill_drain_single_bulk &&
      !contract.p40000_bf16_ab_prompt_wide &&
      !contract.p40000_gdn_prompt_wide &&
      !contract.p40000_flashinfer_whole_prompt;
  const bool exact =
      contract.version == 1U &&
      contract.fp8 == PrefillFp8ArithmeticTactic::kOracleSpanMarlin &&
      contract.nvfp4 ==
          PrefillNvFp4ArithmeticTactic::kOracleSpanGateSiluDownSequence &&
      contract.reset_fp8_locks_per_projection_span &&
      contract.nvfp4_interleaves_gate_silu_down_per_span &&
      contract.nvfp4_down_reuses_gate_up_locks &&
      contract.nvfp4_residual_follows_down_per_span &&
      !contract.m8192_single_bulk_projection &&
      !contract.m8192_fp8_resets_locks_once &&
      !contract.m8192_nvfp4_uses_independent_down_workspace &&
      !contract.m8192_nvfp4_residual_once_after_bulk &&
      !contract.nvfp4_true_large_m_m8192 &&
      !contract.nvfp4_true_large_m_m7712 &&
      !contract.nvfp4_gate_up_down_coupled &&
      !contract.p40000_post_attention_norm_prompt_wide &&
      !contract.p40000_persistent_gate_up_silu &&
      !contract.p40000_persistent_down_residual;
  const bool exact_marlin_m8192 =
      contract.version == 2U &&
      contract.fp8 == PrefillFp8ArithmeticTactic::
                          kM8192SingleBulkOtherwiseOracleSpanMarlin &&
      contract.nvfp4 == PrefillNvFp4ArithmeticTactic::
                            kM8192SingleBulkOtherwiseOracleSpanGateSiluDown &&
      contract.reset_fp8_locks_per_projection_span &&
      contract.nvfp4_interleaves_gate_silu_down_per_span &&
      contract.nvfp4_down_reuses_gate_up_locks &&
      contract.nvfp4_residual_follows_down_per_span &&
      contract.m8192_single_bulk_projection &&
      contract.m8192_fp8_resets_locks_once &&
      contract.m8192_nvfp4_uses_independent_down_workspace &&
      contract.m8192_nvfp4_residual_once_after_bulk &&
      !contract.nvfp4_true_large_m_m8192 &&
      !contract.nvfp4_true_large_m_m7712 &&
      !contract.nvfp4_gate_up_down_coupled &&
      !contract.p40000_post_attention_norm_prompt_wide &&
      !contract.p40000_persistent_gate_up_silu &&
      !contract.p40000_persistent_down_residual;
  const bool segmented_marlin =
      contract.version == 3U &&
      contract.fp8 ==
          PrefillFp8ArithmeticTactic::kOperatorPanelSegmentedMarlin &&
      contract.nvfp4 == PrefillNvFp4ArithmeticTactic::
                            kOperatorPanelGateSiluDownSequence &&
      !contract.reset_fp8_locks_per_projection_span &&
      !contract.nvfp4_interleaves_gate_silu_down_per_span &&
      !contract.nvfp4_down_reuses_gate_up_locks &&
      !contract.nvfp4_residual_follows_down_per_span &&
      !contract.m8192_single_bulk_projection &&
      !contract.m8192_fp8_resets_locks_once &&
      !contract.m8192_nvfp4_uses_independent_down_workspace &&
      !contract.m8192_nvfp4_residual_once_after_bulk &&
      !contract.nvfp4_true_large_m_m8192 &&
      !contract.nvfp4_true_large_m_m7712 &&
      !contract.nvfp4_gate_up_down_coupled &&
      !contract.p40000_post_attention_norm_prompt_wide &&
      !contract.p40000_persistent_gate_up_silu &&
      !contract.p40000_persistent_down_residual;
  const bool true_large_m_nvfp4 =
      contract.version == 4U &&
      contract.fp8 == PrefillFp8ArithmeticTactic::
                          kM8192SingleBulkOtherwiseOracleSpanMarlin &&
      contract.nvfp4 == PrefillNvFp4ArithmeticTactic::
                            kM8192OrM7712TrueLargeMPanelGateSiluDown &&
      contract.reset_fp8_locks_per_projection_span &&
      !contract.nvfp4_interleaves_gate_silu_down_per_span &&
      !contract.nvfp4_down_reuses_gate_up_locks &&
      !contract.nvfp4_residual_follows_down_per_span &&
      contract.m8192_single_bulk_projection &&
      contract.m8192_fp8_resets_locks_once &&
      !contract.m8192_nvfp4_uses_independent_down_workspace &&
      contract.m8192_nvfp4_residual_once_after_bulk &&
      contract.nvfp4_true_large_m_m8192 &&
      contract.nvfp4_true_large_m_m7712 &&
      contract.nvfp4_gate_up_down_coupled &&
      !contract.p40000_post_attention_norm_prompt_wide &&
      !contract.p40000_persistent_gate_up_silu &&
      !contract.p40000_persistent_down_residual;
  const bool persistent_p40_nvfp4 =
      contract.version == 5U &&
      contract.fp8 == PrefillFp8ArithmeticTactic::
                          kM8192SingleBulkOtherwiseOracleSpanMarlin &&
      contract.nvfp4 == PrefillNvFp4ArithmeticTactic::
                            kP40000PersistentGateUpSiluDownResidual &&
      contract.reset_fp8_locks_per_projection_span &&
      !contract.nvfp4_interleaves_gate_silu_down_per_span &&
      !contract.nvfp4_down_reuses_gate_up_locks &&
      !contract.nvfp4_residual_follows_down_per_span &&
      contract.m8192_single_bulk_projection &&
      contract.m8192_fp8_resets_locks_once &&
      !contract.m8192_nvfp4_uses_independent_down_workspace &&
      !contract.m8192_nvfp4_residual_once_after_bulk &&
      !contract.nvfp4_true_large_m_m8192 &&
      !contract.nvfp4_true_large_m_m7712 &&
      contract.nvfp4_gate_up_down_coupled &&
      contract.p40000_post_attention_norm_prompt_wide &&
      contract.p40000_persistent_gate_up_silu &&
      contract.p40000_persistent_down_residual && legacy_common;
  const bool prompt_wide_p40_whole_core =
      contract.version == 6U &&
      contract.bf16_ab ==
          PrefillBf16AbArithmeticTactic::kPromptWideP40SingleGrid &&
      contract.fp8 == PrefillFp8ArithmeticTactic::kP8000FillDrainSingleBulk &&
      contract.nvfp4 == PrefillNvFp4ArithmeticTactic::
                            kP40000PersistentGateUpSiluDownResidual &&
      contract.gdn == PrefillGdnArithmeticTactic::kPromptWideP40ChunkGraph &&
      contract.attention_preprocess ==
          PrefillAttentionPreprocessArithmeticTactic::
              kP8000FillWholePromptFlashInferDrain &&
      !contract.reset_fp8_locks_per_projection_span &&
      !contract.nvfp4_interleaves_gate_silu_down_per_span &&
      !contract.nvfp4_down_reuses_gate_up_locks &&
      !contract.nvfp4_residual_follows_down_per_span &&
      !contract.m8192_single_bulk_projection &&
      !contract.m8192_fp8_resets_locks_once &&
      !contract.m8192_nvfp4_uses_independent_down_workspace &&
      !contract.m8192_nvfp4_residual_once_after_bulk &&
      !contract.nvfp4_true_large_m_m8192 &&
      !contract.nvfp4_true_large_m_m7712 &&
      contract.nvfp4_gate_up_down_coupled &&
      contract.p40000_post_attention_norm_prompt_wide &&
      contract.p40000_persistent_gate_up_silu &&
      contract.p40000_persistent_down_residual &&
      contract.p8000_fp8_fill_drain_single_bulk &&
      contract.p40000_bf16_ab_prompt_wide &&
      contract.p40000_gdn_prompt_wide &&
      contract.p40000_flashinfer_whole_prompt;
  const bool prompt_wide_p40_projection_reset =
      contract.version == 7U &&
      contract.bf16_ab ==
          PrefillBf16AbArithmeticTactic::kPromptWideP40SingleGrid &&
      contract.fp8 == PrefillFp8ArithmeticTactic::
                          kP40000GroupedInputAndOutputSingleBulk &&
      contract.nvfp4 == PrefillNvFp4ArithmeticTactic::
                            kP40000PersistentGateUpSiluDownResidual &&
      contract.gdn == PrefillGdnArithmeticTactic::kPromptWideP40ChunkGraph &&
      contract.attention_preprocess ==
          PrefillAttentionPreprocessArithmeticTactic::
              kP8000FillWholePromptFlashInferDrain &&
      !contract.reset_fp8_locks_per_projection_span &&
      !contract.nvfp4_interleaves_gate_silu_down_per_span &&
      !contract.nvfp4_down_reuses_gate_up_locks &&
      !contract.nvfp4_residual_follows_down_per_span &&
      !contract.m8192_single_bulk_projection &&
      !contract.m8192_fp8_resets_locks_once &&
      !contract.m8192_nvfp4_uses_independent_down_workspace &&
      !contract.m8192_nvfp4_residual_once_after_bulk &&
      !contract.nvfp4_true_large_m_m8192 &&
      !contract.nvfp4_true_large_m_m7712 &&
      contract.nvfp4_gate_up_down_coupled &&
      contract.p40000_post_attention_norm_prompt_wide &&
      contract.p40000_persistent_gate_up_silu &&
      contract.p40000_persistent_down_residual &&
      !contract.p8000_fp8_fill_drain_single_bulk &&
      contract.p40000_bf16_ab_prompt_wide &&
      contract.p40000_gdn_prompt_wide &&
      contract.p40000_flashinfer_whole_prompt;
  return common &&
         ((legacy_common &&
           (exact || exact_marlin_m8192 || segmented_marlin ||
            true_large_m_nvfp4 || persistent_p40_nvfp4)) ||
          prompt_wide_p40_whole_core ||
          prompt_wide_p40_projection_reset);
}

static_assert(kLayerMajorPrefillMaximumArithmeticSpanCount == 16U);
static_assert(is_valid_layer_major_prefill_arithmetic_contract(
    kLayerMajorPrefillExactArithmeticContract));
static_assert(is_valid_layer_major_prefill_arithmetic_contract(
    kLayerMajorPrefillExactMarlinM8192ArithmeticContract));
static_assert(is_valid_layer_major_prefill_arithmetic_contract(
    kLayerMajorPrefillSegmentedMarlinArithmeticContract));
static_assert(is_valid_layer_major_prefill_arithmetic_contract(
    kLayerMajorPrefillTrueLargeMNvFp4ArithmeticContract));
static_assert(is_valid_layer_major_prefill_arithmetic_contract(
    kLayerMajorPrefillPersistentP40NvFp4ArithmeticContract));
static_assert(is_valid_layer_major_prefill_arithmetic_contract(
    kLayerMajorPrefillPromptWideP40WholeCoreArithmeticContract));
static_assert(is_valid_layer_major_prefill_arithmetic_contract(
    kLayerMajorPrefillPromptWideP40ProjectionResetArithmeticContract));
static_assert(is_valid_layer_major_prefill_arithmetic_span_ledger(
    make_layer_major_prefill_arithmetic_span_ledger(513U)));

enum class PrefillTraversalOrder : std::uint8_t {
  kLayerMajor = 0,
};

enum class PrefillProgressDomain : std::uint8_t {
  kGdnState = 0,
  kKvCache,
};

// Logical MLP ownership inside one layer. The incumbent completes MLP inside
// every operator-panel submission. The test-only P40 candidate first
// completes Attention/GDN and its post-attention residual for all five
// panels, then submits post-attention norm and the exact full-M MLP once.
// This enum is topology only: a plan remains unbound and non-executable.
enum class LayerMajorPrefillMlpScheduleTactic : std::uint8_t {
  kPerOperatorPanel = 0,
  kLayerWideP40ExactFullM,
  kPromptWideP40WholeCore,
  kPromptWideP40ProjectionReset,
};

[[nodiscard]] constexpr bool is_valid_layer_major_prefill_mlp_schedule_tactic(
    const LayerMajorPrefillMlpScheduleTactic tactic) noexcept {
  return tactic ==
             LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel ||
         tactic == LayerMajorPrefillMlpScheduleTactic::
                       kLayerWideP40ExactFullM ||
         tactic == LayerMajorPrefillMlpScheduleTactic::
                       kPromptWideP40WholeCore ||
         tactic == LayerMajorPrefillMlpScheduleTactic::
                       kPromptWideP40ProjectionReset;
}

[[nodiscard]] constexpr std::string_view to_string(
    const LayerMajorPrefillMlpScheduleTactic tactic) noexcept {
  switch (tactic) {
    case LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel:
      return "per-operator-panel";
    case LayerMajorPrefillMlpScheduleTactic::kLayerWideP40ExactFullM:
      return "layer-wide-p40-exact-full-m";
    case LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore:
      return "prompt-wide-p40-whole-core";
    case LayerMajorPrefillMlpScheduleTactic::kPromptWideP40ProjectionReset:
      return "prompt-wide-p40-projection-reset";
  }
  return "unknown";
}

// Route evidence counts complete 64-layer logical passes, not storage panels.
// The incumbent publishes one pass per panel; the exact-P40000 layer-wide
// schedule collapses all five panels plus their full-M MLP phases into one
// request-wide pass.
[[nodiscard]] constexpr std::uint64_t prefill_route_layer_pass_count(
    const std::size_t logical_panel_count,
    const LayerMajorPrefillMlpScheduleTactic tactic) noexcept {
  return tactic != LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel
             ? 1U
             : static_cast<std::uint64_t>(logical_panel_count);
}

// Reports the compile-time admission inventory owned by q3x_core. Production
// builds return false. Enabling the candidate requires the default-OFF,
// BUILD_TESTING-only CMake admission switch.
[[nodiscard]] bool layer_wide_p40_mlp_prefill_plan_enabled() noexcept;

// Independent architecture admission. Enabling the layer-wide MLP-only
// experiment never enables the whole-core schedule by implication.
[[nodiscard]] bool prompt_wide_p40_whole_core_prefill_plan_enabled() noexcept;

// Independent default-OFF admission for the projection reset. The reset may
// not become selectable merely because the earlier whole-core experiment is
// compiled into the same binary.
[[nodiscard]] bool prompt_wide_p40_projection_reset_prefill_plan_enabled()
    noexcept;

enum class PrefillExecutionPlanError : std::uint8_t {
  kNone = 0,
  kInvalidArgument,
  kArithmeticOverflow,
  kCapacityExceeded,
  kInvalidTopology,
};

struct PrefillExecutionPlanOptions {
  std::uint64_t first_position = 0U;
  std::uint64_t prompt_token_count = 0U;
  std::uint64_t max_sequence_length =
      kLayerMajorPrefillMaximumSequenceTokens;
  LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic =
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
};

struct PrefillOperatorPanel {
  std::uint32_t ordinal = 0U;
  std::uint32_t first_position = 0U;
  std::uint32_t token_count = 0U;
  std::uint32_t end_position = 0U;
};

struct PrefillLayerExecution {
  std::size_t layer_index = 0U;
  model::LayerType layer_type = model::LayerType::kInvalid;
  PrefillProgressDomain progress_domain = PrefillProgressDomain::kGdnState;
  std::size_t panel_count = 0U;
};

struct PrefillFinalCommitPlan {
  std::uint32_t expected_initial_sequence_length = 0U;
  std::uint32_t committed_sequence_length = 0U;
  std::uint32_t commit_count = 0U;
};

struct PrefillMlpSchedulePlan {
  LayerMajorPrefillMlpScheduleTactic tactic =
      LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel;
  // Number of Attention/GDN panel phases that precede completion of a layer.
  std::size_t operator_panel_phase_count_per_layer = 0U;
  // The incumbent owns one MLP phase per panel; P40 owns one full-M phase.
  std::size_t mlp_phase_submission_count_per_layer = 0U;
  std::uint32_t maximum_m_per_mlp_submission = 0U;

  // Physical-launch envelope for a future candidate binding. Exact separate
  // GateUp/SiLU/Down owns three launches; an exact persistent GateUp+SiLU
  // epilogue owns two. Both must keep each full-M projection unsplit.
  std::size_t required_gate_up_projection_launches_per_layer = 0U;
  std::size_t maximum_standalone_silu_launches_per_layer = 0U;
  std::size_t required_down_projection_launches_per_layer = 0U;
  std::size_t minimum_total_kernel_launches_per_layer = 0U;
  std::size_t maximum_total_kernel_launches_per_layer = 0U;
  bool waits_for_all_operator_panels = false;
  bool post_attention_residual_completed_panelwise = true;
  bool post_attention_norm_is_prompt_wide = false;
  bool exact_full_m_binding_required = false;
  bool internal_m_segmentation_forbidden = false;
};

struct PrefillWholeCoreSchedulePlan {
  bool enabled = false;
  std::size_t fill_panel_phase_count_per_layer = 0U;
  std::size_t prompt_core_phase_count_per_layer = 0U;
  std::size_t drain_panel_phase_count_per_layer = 0U;
  std::size_t persistent_mlp_phase_count_per_layer = 0U;
  std::uint32_t panel_token_count = 0U;
  std::uint32_t prompt_core_token_count = 0U;
  std::uint32_t request_capacity_tokens = 0U;
  std::uint64_t route_pass_count = 0U;
  bool fp8_single_launch_per_projection_required = false;
  bool bf16_ab_prompt_wide_required = false;
  bool gdn_prompt_wide_required = false;
  bool flashinfer_whole_prompt_required = false;
};

// Exact-P40000 projection ownership for the reset architecture. The five
// M8000 panels remain bounded storage/progress geometry; they are not FP8
// physical projection boundaries. A conforming binding first makes all input
// rows ready, then submits one grouped P40000 input projection and one P40000
// O projection per layer. Logical tensor-role hits remain separately visible
// so grouping cannot erase QKV/Z or Q/K/V route attestation.
struct PrefillP40ProjectionResetSchedulePlan {
  bool enabled = false;
  std::size_t input_preparation_panel_count_per_layer = 0U;
  std::size_t prompt_core_phase_count_per_layer = 0U;
  std::size_t persistent_mlp_phase_count_per_layer = 0U;
  std::uint32_t panel_token_count = 0U;
  std::uint32_t projection_m_tokens = 0U;
  std::uint32_t request_capacity_tokens = 0U;
  std::uint64_t route_pass_count = 0U;
  std::size_t fp8_grouped_input_launches_per_layer = 0U;
  std::size_t fp8_output_launches_per_layer = 0U;
  std::size_t fp8_physical_launches_per_request = 0U;
  std::size_t fp8_tensor_role_hits_per_request = 0U;
  std::size_t nvfp4_gate_up_launches_per_layer = 0U;
  std::size_t nvfp4_down_launches_per_layer = 0U;
  std::size_t nvfp4_physical_launches_per_request = 0U;
  bool fp8_grouped_full_prompt_input_required = false;
  bool fp8_full_prompt_output_required = false;
  bool nvfp4_full_prompt_required = false;
  bool internal_m_segmentation_forbidden = false;
  bool production_accuracy_required = false;
  bool approximate_numerics_forbidden = false;
  bool mtp_forbidden = false;
  bool cublaslt_forbidden = false;
};

struct PrefillExecutionPlan {
  PrefillTraversalOrder traversal = PrefillTraversalOrder::kLayerMajor;
  // Descriptive compatibility metadata only. It never determines panels.
  std::uint32_t legacy_public_tile_limit =
      kLayerMajorPrefillLegacyPublicTileTokens;
  std::uint32_t operator_panel_capacity =
      kLayerMajorPrefillOperatorPanelTokens;
  std::uint32_t first_position = 0U;
  std::uint32_t prompt_token_count = 0U;
  std::uint32_t final_position = 0U;
  std::array<PrefillOperatorPanel,
             kLayerMajorPrefillMaximumPanelCount>
      panels{};
  std::size_t panel_count = 0U;
  std::array<PrefillLayerExecution, kLayerMajorPrefillLayerCount> layers{};
  PrefillMlpSchedulePlan mlp_schedule;
  PrefillWholeCoreSchedulePlan whole_core_schedule;
  PrefillP40ProjectionResetSchedulePlan projection_reset_schedule;
  PrefillFinalCommitPlan final_commit;

  // The scaffold deliberately has no mutation or binder that can make this
  // true. A later, separately reviewed BoundPrefillExecutionPlan must own
  // typed launchers before an execution route can become selectable.
  bool operator_bindings_complete = false;

  [[nodiscard]] constexpr bool executable() const noexcept {
    return operator_bindings_complete;
  }
};

struct PrefillExecutionPlanResult {
  std::optional<PrefillExecutionPlan> value;
  PrefillExecutionPlanError error = PrefillExecutionPlanError::kNone;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && error == PrefillExecutionPlanError::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Builds only the immutable layer/panel topology. It does not inspect model
// weights, select a kernel, reserve memory, or alter the current C512 route.
[[nodiscard]] PrefillExecutionPlanResult
build_unbound_layer_major_prefill_execution_plan(
    const PrefillExecutionPlanOptions& options) noexcept;

// Allocation-free host validation for an externally supplied immutable
// layer-major topology. This accepts only the unbound design contract:
// operator_bindings_complete must remain false. It does not authenticate or
// convert a bound execution/deployment plan.
[[nodiscard]] bool is_valid_unbound_layer_major_prefill_execution_plan(
    const PrefillExecutionPlan& plan) noexcept;

enum class PrefillExecutionProgressError : std::uint8_t {
  kNone = 0,
  kInvalidPlan,
  kLayerOutOfRange,
  kPanelOutOfRange,
  kOutOfOrder,
  kExecutionIncomplete,
  kCommitNotReady,
  kAlreadyCommitted,
};

// Mutable request-owned progress is intentionally not part of the immutable
// execution plan. Positions are exclusive ends. An executor may call the
// transition helper only after the corresponding device completion/event has
// established visibility; enqueue alone is not completion.
struct PrefillExecutionProgress {
  std::array<std::uint32_t, kLayerMajorPrefillLayerCount> kv_visible_end{};
  std::array<std::uint32_t, kLayerMajorPrefillLayerCount> gdn_advanced_end{};
  std::array<std::size_t, kLayerMajorPrefillLayerCount> completed_panels{};
  // Per-panel schedules advance this with every completed panel. The P40
  // layer-wide schedule advances it once, only after all panel Attention/GDN
  // and residual work for that layer has completed.
  std::array<std::size_t, kLayerMajorPrefillLayerCount>
      completed_mlp_phases{};
  std::array<std::size_t, kLayerMajorPrefillLayerCount>
      completed_fill_panels{};
  std::array<std::size_t, kLayerMajorPrefillLayerCount>
      completed_prompt_core_phases{};
  std::array<std::size_t, kLayerMajorPrefillLayerCount>
      completed_drain_panels{};
  std::size_t next_layer = 0U;
  std::size_t next_panel = 0U;
  bool final_hidden_ready = false;
  bool prefill_state_committed = false;
};

[[nodiscard]] PrefillExecutionProgressError
advance_prompt_wide_p40_fill_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    std::size_t layer_index, std::size_t panel_index) noexcept;

[[nodiscard]] PrefillExecutionProgressError
advance_prompt_wide_p40_prompt_core_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    std::size_t layer_index) noexcept;

[[nodiscard]] PrefillExecutionProgressError
advance_prompt_wide_p40_drain_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    std::size_t layer_index, std::size_t panel_index) noexcept;

[[nodiscard]] PrefillExecutionProgressError
advance_prompt_wide_p40_persistent_mlp_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    std::size_t layer_index) noexcept;

// The projection reset publishes request progress only after the complete
// grouped FP8 input, mathematical core, FP8 O, and full-Prompt NVFP4 MLP for
// one layer have completed. The host transition is intentionally atomic so a
// partially executed grouped projection can never look like a completed
// M8000 panel.
[[nodiscard]] PrefillExecutionProgressError
advance_prompt_wide_p40_projection_reset_layer_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    std::size_t layer_index) noexcept;

[[nodiscard]] PrefillExecutionProgress make_prefill_execution_progress(
    const PrefillExecutionPlan& plan) noexcept;

[[nodiscard]] PrefillExecutionProgressError
advance_prefill_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    std::size_t layer_index, std::size_t panel_index) noexcept;

// Publishes completion of the single full-M MLP phase for one P40 layer. It
// is invalid for the incumbent per-panel schedule or before all operator
// panels in the layer are complete.
[[nodiscard]] PrefillExecutionProgressError
advance_layer_wide_p40_mlp_progress_after_completion(
    const PrefillExecutionPlan& plan, PrefillExecutionProgress& progress,
    std::size_t layer_index) noexcept;

[[nodiscard]] PrefillExecutionProgressError mark_prefill_final_hidden_ready(
    const PrefillExecutionPlan& plan,
    PrefillExecutionProgress& progress) noexcept;

[[nodiscard]] bool prefill_final_commit_ready(
    const PrefillExecutionPlan& plan,
    const PrefillExecutionProgress& progress) noexcept;

// This is a host-state transition only. It deliberately does not call
// RequestState::set_sequence_length() and therefore cannot alter production
// request state while the plan remains unbound.
[[nodiscard]] PrefillExecutionProgressError publish_prefill_state_committed(
    const PrefillExecutionPlan& plan,
    PrefillExecutionProgress& progress) noexcept;

}  // namespace q3x::runtime
