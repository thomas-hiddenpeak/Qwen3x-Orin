#pragma once

#include "q3x/kernels/sm87_target_aot_context.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Host-only Attention constituent for AC-PREFILL-SM87-AOT-SYSTEM-v1. This
// freezes the real Qwen3.6 tensor ABI and the Q128/KV32 producer-consumer
// contract. It exposes no launcher, selector, or callable GPU surface.
inline constexpr std::size_t kSm87TargetAotAttentionQueryHeads = 24U;
inline constexpr std::size_t kSm87TargetAotAttentionKvHeads = 4U;
inline constexpr std::size_t kSm87TargetAotAttentionQueriesPerKv = 6U;
inline constexpr std::size_t kSm87TargetAotAttentionHeadDimension = 256U;
inline constexpr std::size_t kSm87TargetAotAttentionQueryRows = 128U;
inline constexpr std::size_t kSm87TargetAotAttentionKvTokens = 32U;
inline constexpr std::size_t kSm87TargetAotAttentionThreads = 256U;
inline constexpr std::size_t kSm87TargetAotAttentionWarps = 8U;
inline constexpr std::size_t kSm87TargetAotAttentionPipelineStages = 2U;
inline constexpr std::size_t kSm87TargetAotAttentionQueryRowsPerWarp = 16U;
inline constexpr std::size_t kSm87TargetAotAttentionLayerCount = 64U;
inline constexpr std::size_t kSm87TargetAotAttentionLayerInterval = 4U;
inline constexpr std::size_t kSm87TargetAotAttentionBf16Bytes = 2U;
inline constexpr std::size_t kSm87TargetAotAttentionFp32Bytes = 4U;

// The model has head_dim=256 and partial_rotary_factor=0.25. Therefore 64
// elements are rotary, split into two NeoX halves of 32 elements: exactly 32
// pairs, not 64 pairs/128 elements.
inline constexpr std::size_t kSm87TargetAotAttentionRotaryElements = 64U;
inline constexpr std::size_t kSm87TargetAotAttentionRotaryPairs = 32U;
inline constexpr std::size_t kSm87TargetAotAttentionRopePassthroughElements =
    kSm87TargetAotAttentionHeadDimension -
    kSm87TargetAotAttentionRotaryElements;
inline constexpr std::uint32_t kSm87TargetAotAttentionRmsEpsilonFp32Bits =
    0x3586'37bdU;  // 1.0e-6F
inline constexpr std::uint32_t kSm87TargetAotAttentionScaleFp32Bits =
    0x3d80'0000U;  // 1/sqrt(256) = 1/16 exactly
inline constexpr std::uint32_t kSm87TargetAotAttentionLog2EFp32Bits =
    0x3fb8'aa3bU;  // float(1.4426950408889634)
inline constexpr std::uint64_t kSm87TargetAotAttentionRopeTheta =
    10'000'000ULL;

inline constexpr std::size_t kSm87TargetAotAttentionRawQGateTokenStride =
    kSm87TargetAotAttentionQueryHeads * 2U *
    kSm87TargetAotAttentionHeadDimension;
inline constexpr std::size_t kSm87TargetAotAttentionQueryTokenStride =
    kSm87TargetAotAttentionQueryHeads *
    kSm87TargetAotAttentionHeadDimension;
inline constexpr std::size_t kSm87TargetAotAttentionKvTokenStride =
    kSm87TargetAotAttentionKvHeads *
    kSm87TargetAotAttentionHeadDimension;
inline constexpr std::size_t kSm87TargetAotAttentionQuerySharedBytes =
    kSm87TargetAotAttentionQueryRows *
    kSm87TargetAotAttentionHeadDimension *
    kSm87TargetAotAttentionBf16Bytes;
inline constexpr std::size_t kSm87TargetAotAttentionKvStageSharedBytes =
    2U * kSm87TargetAotAttentionKvTokens *
    kSm87TargetAotAttentionHeadDimension *
    kSm87TargetAotAttentionBf16Bytes;
inline constexpr std::size_t kSm87TargetAotAttentionSharedBytes =
    kSm87TargetAotAttentionQuerySharedBytes +
    kSm87TargetAotAttentionPipelineStages *
        kSm87TargetAotAttentionKvStageSharedBytes;

enum class Sm87TargetAotAttentionTopology : std::uint8_t {
  kInvalid = 0U,
  kQ128Kv32TwoStage,
  // A placeholder, not a selectable tactic. A direct Q256 copy of the old
  // kernel needs a new Q feed or V/output partition and a resource audit.
  kQ256RequiresNewProducerConsumer,
};

enum class Sm87TargetAotAttentionScalarType : std::uint8_t {
  kInvalid = 0U,
  kBf16,
  kFp32,
};

enum class Sm87TargetAotAttentionBufferRole : std::uint8_t {
  kInvalid = 0U,
  kRawQGate,
  kRawK,
  kRawV,
  kProcessedQ,
  kProcessedGate,
  kProcessedK,
  kProcessedV,
  kQNormWeight,
  kKNormWeight,
  kRopeCos,
  kRopeSin,
  // The online-softmax result is rounded to BF16 before the sigmoid gate.
  kBf16AttentionOutput,
  // This [T,24,256] span is the only legal input to the full-Attention O
  // projection. It is distinct from the pre-gate BF16 publication.
  kSigmoidGatedOutput,
  kCount,
};

inline constexpr std::size_t kSm87TargetAotAttentionBufferCount =
    static_cast<std::size_t>(Sm87TargetAotAttentionBufferRole::kCount) - 1U;

inline constexpr std::array<Sm87TargetAotAttentionBufferRole,
                            kSm87TargetAotAttentionBufferCount>
    kSm87TargetAotAttentionBufferRoles{{
        Sm87TargetAotAttentionBufferRole::kRawQGate,
        Sm87TargetAotAttentionBufferRole::kRawK,
        Sm87TargetAotAttentionBufferRole::kRawV,
        Sm87TargetAotAttentionBufferRole::kProcessedQ,
        Sm87TargetAotAttentionBufferRole::kProcessedGate,
        Sm87TargetAotAttentionBufferRole::kProcessedK,
        Sm87TargetAotAttentionBufferRole::kProcessedV,
        Sm87TargetAotAttentionBufferRole::kQNormWeight,
        Sm87TargetAotAttentionBufferRole::kKNormWeight,
        Sm87TargetAotAttentionBufferRole::kRopeCos,
        Sm87TargetAotAttentionBufferRole::kRopeSin,
        Sm87TargetAotAttentionBufferRole::kBf16AttentionOutput,
        Sm87TargetAotAttentionBufferRole::kSigmoidGatedOutput,
    }};

[[nodiscard]] constexpr std::size_t sm87_target_aot_attention_buffer_index(
    const Sm87TargetAotAttentionBufferRole role) noexcept {
  for (std::size_t index = 0U;
       index < kSm87TargetAotAttentionBufferRoles.size(); ++index) {
    if (kSm87TargetAotAttentionBufferRoles[index] == role) {
      return index;
    }
  }
  return kSm87TargetAotAttentionBufferCount;
}

enum class Sm87TargetAotAttentionBufferLayout : std::uint8_t {
  kInvalid = 0U,
  // Canonical projection ABI: [T, 24, 2, 256], where each head stores its Q
  // 256 immediately followed by its gate 256.
  kRawQGateT24x2xD256PerHeadInterleaved,
  kRawKvT4xD256,
  kProcessedQT24xD256,
  kProcessedGateT24xD256,
  // NHD means token/sequence, head, dimension. These processed K/V buffers
  // are staged in the private request transaction. They become Decode cache
  // only after the system-wide PrefillStateCommitted event.
  kProcessedKeyNhdT4xD256,
  kProcessedValueNhdT4xD256,
  kBf16AttentionT24xD256,
  kSigmoidGatedT24xD256,
  kCenteredRmsNormWeightD256,
  kRopeTableT32Pairs,
};

enum class Sm87TargetAotAttentionBufferLifetime : std::uint8_t {
  kInvalid = 0U,
  kLayerPreparation,
  kLayerCore,
  kLayerProjectionConsumer,
  kRequestTransactionUnpublishedUntilCommit,
  kEngine,
};

enum class Sm87TargetAotAttentionBufferPublication : std::uint8_t {
  kInvalid = 0U,
  kProjectionOutput,
  kPreprocessOutput,
  kOrderedKvTransactionStage,
  kAttentionCoreOutput,
  kSigmoidGateOutput,
  kEngineReady,
};

enum class Sm87TargetAotAttentionNormContract : std::uint8_t {
  kInvalid = 0U,
  kCenteredRmsNormBf16WeightFp32Epsilon,
};

enum class Sm87TargetAotAttentionRopeContract : std::uint8_t {
  kInvalid = 0U,
  kPartialNeox64Elements32PairsFp32Table,
};

enum class Sm87TargetAotAttentionGateContract : std::uint8_t {
  kInvalid = 0U,
  kBitExactPerHeadSplitCopyNoNormNoRope,
};

enum class Sm87TargetAotAttentionValueContract : std::uint8_t {
  kInvalid = 0U,
  kBitExactPassthroughToOrderedKvTransactionStage,
};

enum class Sm87TargetAotAttentionKvCacheLayout : std::uint8_t {
  kInvalid = 0U,
  kNhdTokenHeadDimension,
};

enum class Sm87TargetAotAttentionKvResetContract : std::uint8_t {
  kInvalid = 0U,
  kColdEpochZeroLengthCurrentLayer,
};

enum class Sm87TargetAotAttentionTaskRole : std::uint8_t {
  kInvalid = 0U,
  kCausalQ128AgainstOrderedNhdKv,
};

// These contracts transcribe the finite-precision instruction order of the
// production SM87 full-Attention path.  They are deliberately more specific
// than real-number online-softmax equivalence: changing an exp backend, a
// BF16 publication point, or an MMA/reduction traversal creates a different
// numerical candidate.
enum class Sm87TargetAotAttentionExpContract : std::uint8_t {
  kInvalid = 0U,
  kEx2ApproxF32AfterFp32Log2eMultiply,
};

enum class Sm87TargetAotAttentionQkTraversalContract : std::uint8_t {
  kInvalid = 0U,
  kMmaSyncM16N16K16DimensionAscendingScoreSubtilesAscending,
};

enum class Sm87TargetAotAttentionProbabilityContract : std::uint8_t {
  kInvalid = 0U,
  kExpF32ToBf16RneBeforeDenominatorAndPv,
};

enum class Sm87TargetAotAttentionDenominatorContract : std::uint8_t {
  kInvalid = 0U,
  kRescalePriorThenRegister0145ThenXor1Xor2,
};

enum class Sm87TargetAotAttentionPvTraversalContract : std::uint8_t {
  kInvalid = 0U,
  kMmaSyncBf16ProbabilityBf16VToFp32OutputDimensionAscending,
};

enum class Sm87TargetAotAttentionOutputContract : std::uint8_t {
  kInvalid = 0U,
  kFp32ReciprocalThenNumeratorMultiplyThenBf16Rne,
};

enum class Sm87TargetAotAttentionSigmoidContract : std::uint8_t {
  kInvalid = 0U,
  kStableSignBranchEx2ApproxF32TimesBf16AttentionThenBf16Rne,
};

enum class Sm87TargetAotAttentionRmsReductionContract : std::uint8_t {
  kInvalid = 0U,
  kPromptWide128FmafDAndD128ThenShared64Pair32Pair96Shuffle16To1,
};

enum class Sm87TargetAotAttentionRsqrtContract : std::uint8_t {
  kInvalid = 0U,
  kRsqrtfFp32SumDiv256PlusFp32Epsilon,
};

enum class Sm87TargetAotAttentionNormPublicationContract : std::uint8_t {
  kInvalid = 0U,
  kBf16WeightPlusOneValueTimesInverseThenGammaBf16Rne,
};

enum class Sm87TargetAotAttentionRopeFmaContract : std::uint8_t {
  kInvalid = 0U,
  kNeoxSeparateSinProductThenCosFmaBf16Rne,
};

enum class Sm87TargetAotAttentionRopeMappingContract : std::uint8_t {
  kInvalid = 0U,
  kRotateD0To31WithDPlus32Tail64To255Passthrough,
};

struct Sm87TargetAotAttentionNumericalContract {
  Sm87TargetAotAttentionExpContract exp_contract =
      Sm87TargetAotAttentionExpContract::kInvalid;
  Sm87TargetAotAttentionQkTraversalContract qk_traversal =
      Sm87TargetAotAttentionQkTraversalContract::kInvalid;
  Sm87TargetAotAttentionProbabilityContract probability_publication =
      Sm87TargetAotAttentionProbabilityContract::kInvalid;
  Sm87TargetAotAttentionDenominatorContract denominator_update =
      Sm87TargetAotAttentionDenominatorContract::kInvalid;
  Sm87TargetAotAttentionPvTraversalContract pv_traversal =
      Sm87TargetAotAttentionPvTraversalContract::kInvalid;
  Sm87TargetAotAttentionOutputContract output_publication =
      Sm87TargetAotAttentionOutputContract::kInvalid;
  Sm87TargetAotAttentionSigmoidContract sigmoid_publication =
      Sm87TargetAotAttentionSigmoidContract::kInvalid;
  std::uint32_t log2e_fp32_bits = 0U;
  std::size_t mma_m = 0U;
  std::size_t mma_n = 0U;
  std::size_t mma_k = 0U;
  std::size_t qk_head_dimension_tiles = 0U;
  std::size_t kv_stage_score_subtiles = 0U;
  std::size_t pv_output_dimension_tiles = 0U;
  std::array<std::size_t, 4U> denominator_register_offsets{};
  std::array<std::size_t, 2U> denominator_shuffle_xor_masks{};
  bool attention_scale_after_qk_accumulation = false;
  bool prior_denominator_rescaled_before_probability_add = false;
  bool prior_output_rescaled_before_pv_mma = false;
  bool probability_bf16_reused_by_denominator = false;
  bool probability_bf16_reused_by_pv_mma = false;
  bool bf16_attention_reused_by_sigmoid_epilogue = false;
};

struct Sm87TargetAotAttentionPreprocessNumericalContract {
  Sm87TargetAotAttentionRmsReductionContract rms_reduction =
      Sm87TargetAotAttentionRmsReductionContract::kInvalid;
  Sm87TargetAotAttentionRsqrtContract inverse_rms =
      Sm87TargetAotAttentionRsqrtContract::kInvalid;
  Sm87TargetAotAttentionNormPublicationContract norm_publication =
      Sm87TargetAotAttentionNormPublicationContract::kInvalid;
  Sm87TargetAotAttentionRopeFmaContract rope_fma =
      Sm87TargetAotAttentionRopeFmaContract::kInvalid;
  Sm87TargetAotAttentionRopeMappingContract rope_mapping =
      Sm87TargetAotAttentionRopeMappingContract::kInvalid;
  std::size_t threads = 0U;
  std::size_t head_dimension = 0U;
  std::array<std::size_t, 2U> thread_dimension_offsets{};
  std::array<std::size_t, 4U> shared_tree_offsets{};
  std::array<std::size_t, 5U> shuffle_down_strides{};
  std::array<std::size_t, 2U> rope_pair_offsets{};
  std::size_t rope_pair_count = 0U;
  std::size_t rope_tail_begin = 0U;
  std::size_t rope_tail_end = 0U;
  bool square_uses_fmaf_with_positive_zero = false;
  bool pair_add_low_square_before_high_square = false;
  bool epsilon_added_after_divide_by_head_dimension = false;
  bool norm_weight_decoded_from_bf16_then_fp32_plus_one = false;
  bool norm_multiplies_value_by_inverse_before_gamma = false;
  bool normalized_qk_published_bf16_rne_before_rope = false;
  bool rope_consumes_published_bf16_qk = false;
  bool rope_sine_product_rounded_before_fma = false;
  bool rope_output_published_bf16_rne = false;
  bool rope_tail_is_bit_exact_normalized_bf16 = false;
};

[[nodiscard]] constexpr bool
sm87_target_aot_same_attention_numerical_contract(
    const Sm87TargetAotAttentionNumericalContract& left,
    const Sm87TargetAotAttentionNumericalContract& right) noexcept {
  if (left.exp_contract != right.exp_contract ||
      left.qk_traversal != right.qk_traversal ||
      left.probability_publication != right.probability_publication ||
      left.denominator_update != right.denominator_update ||
      left.pv_traversal != right.pv_traversal ||
      left.output_publication != right.output_publication ||
      left.sigmoid_publication != right.sigmoid_publication ||
      left.log2e_fp32_bits != right.log2e_fp32_bits ||
      left.mma_m != right.mma_m || left.mma_n != right.mma_n ||
      left.mma_k != right.mma_k ||
      left.qk_head_dimension_tiles != right.qk_head_dimension_tiles ||
      left.kv_stage_score_subtiles != right.kv_stage_score_subtiles ||
      left.pv_output_dimension_tiles != right.pv_output_dimension_tiles ||
      left.attention_scale_after_qk_accumulation !=
          right.attention_scale_after_qk_accumulation ||
      left.prior_denominator_rescaled_before_probability_add !=
          right.prior_denominator_rescaled_before_probability_add ||
      left.prior_output_rescaled_before_pv_mma !=
          right.prior_output_rescaled_before_pv_mma ||
      left.probability_bf16_reused_by_denominator !=
          right.probability_bf16_reused_by_denominator ||
      left.probability_bf16_reused_by_pv_mma !=
          right.probability_bf16_reused_by_pv_mma ||
      left.bf16_attention_reused_by_sigmoid_epilogue !=
          right.bf16_attention_reused_by_sigmoid_epilogue) {
    return false;
  }
  for (std::size_t index = 0U;
       index < left.denominator_register_offsets.size(); ++index) {
    if (left.denominator_register_offsets[index] !=
        right.denominator_register_offsets[index]) {
      return false;
    }
  }
  for (std::size_t index = 0U;
       index < left.denominator_shuffle_xor_masks.size(); ++index) {
    if (left.denominator_shuffle_xor_masks[index] !=
        right.denominator_shuffle_xor_masks[index]) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool
sm87_target_aot_same_attention_preprocess_numerical_contract(
    const Sm87TargetAotAttentionPreprocessNumericalContract& left,
    const Sm87TargetAotAttentionPreprocessNumericalContract& right) noexcept {
  if (left.rms_reduction != right.rms_reduction ||
      left.inverse_rms != right.inverse_rms ||
      left.norm_publication != right.norm_publication ||
      left.rope_fma != right.rope_fma ||
      left.rope_mapping != right.rope_mapping ||
      left.threads != right.threads ||
      left.head_dimension != right.head_dimension ||
      left.rope_pair_count != right.rope_pair_count ||
      left.rope_tail_begin != right.rope_tail_begin ||
      left.rope_tail_end != right.rope_tail_end ||
      left.square_uses_fmaf_with_positive_zero !=
          right.square_uses_fmaf_with_positive_zero ||
      left.pair_add_low_square_before_high_square !=
          right.pair_add_low_square_before_high_square ||
      left.epsilon_added_after_divide_by_head_dimension !=
          right.epsilon_added_after_divide_by_head_dimension ||
      left.norm_weight_decoded_from_bf16_then_fp32_plus_one !=
          right.norm_weight_decoded_from_bf16_then_fp32_plus_one ||
      left.norm_multiplies_value_by_inverse_before_gamma !=
          right.norm_multiplies_value_by_inverse_before_gamma ||
      left.normalized_qk_published_bf16_rne_before_rope !=
          right.normalized_qk_published_bf16_rne_before_rope ||
      left.rope_consumes_published_bf16_qk !=
          right.rope_consumes_published_bf16_qk ||
      left.rope_sine_product_rounded_before_fma !=
          right.rope_sine_product_rounded_before_fma ||
      left.rope_output_published_bf16_rne !=
          right.rope_output_published_bf16_rne ||
      left.rope_tail_is_bit_exact_normalized_bf16 !=
          right.rope_tail_is_bit_exact_normalized_bf16) {
    return false;
  }
  for (std::size_t index = 0U;
       index < left.thread_dimension_offsets.size(); ++index) {
    if (left.thread_dimension_offsets[index] !=
        right.thread_dimension_offsets[index]) {
      return false;
    }
  }
  for (std::size_t index = 0U;
       index < left.shared_tree_offsets.size(); ++index) {
    if (left.shared_tree_offsets[index] !=
        right.shared_tree_offsets[index]) {
      return false;
    }
  }
  for (std::size_t index = 0U;
       index < left.shuffle_down_strides.size(); ++index) {
    if (left.shuffle_down_strides[index] !=
        right.shuffle_down_strides[index]) {
      return false;
    }
  }
  for (std::size_t index = 0U;
       index < left.rope_pair_offsets.size(); ++index) {
    if (left.rope_pair_offsets[index] !=
        right.rope_pair_offsets[index]) {
      return false;
    }
  }
  return true;
}

enum Sm87TargetAotAttentionPolicy : std::uint32_t {
  kSm87TargetAotAttentionCausal = 1U << 0U,
  kSm87TargetAotAttentionOnlineSoftmaxFp32 = 1U << 1U,
  kSm87TargetAotAttentionNoScoreMatrix = 1U << 2U,
  kSm87TargetAotAttentionNoSplitKv = 1U << 3U,
  kSm87TargetAotAttentionNoCrossCtaReduction = 1U << 4U,
  kSm87TargetAotAttentionBf16PublicationBeforeGate = 1U << 5U,
  kSm87TargetAotAttentionNoRequestJit = 1U << 6U,
  kSm87TargetAotAttentionNoFallback = 1U << 7U,
  kSm87TargetAotAttentionAccuracyUnqualified = 1U << 8U,
  kSm87TargetAotAttentionPackedQueryGqaReuse = 1U << 9U,
  kSm87TargetAotAttentionPerQueryCausalMask = 1U << 10U,
  kSm87TargetAotAttentionOrderedKvTransactionStage = 1U << 11U,
  kSm87TargetAotAttentionQkRmsNormBeforeRope = 1U << 12U,
  kSm87TargetAotAttentionPositionBoundRope = 1U << 13U,
  kSm87TargetAotAttentionProcessedQGatePublication = 1U << 14U,
  kSm87TargetAotAttentionProcessedKvBeforeCore = 1U << 15U,
  kSm87TargetAotAttentionTypedRawProducerEvents = 1U << 16U,
  kSm87TargetAotAttentionCanonicalPerHeadQGate = 1U << 17U,
  kSm87TargetAotAttentionGateBitExactSplitCopy = 1U << 18U,
  kSm87TargetAotAttentionVBitExactPassthrough = 1U << 19U,
  kSm87TargetAotAttentionCenteredQkNorm = 1U << 20U,
  kSm87TargetAotAttentionPartialNeox64Elements = 1U << 21U,
  kSm87TargetAotAttentionExactOneOver16Scale = 1U << 22U,
  kSm87TargetAotAttentionNhdColdKvEpoch = 1U << 23U,
  kSm87TargetAotAttentionTypedBufferSpans = 1U << 24U,
  kSm87TargetAotAttentionNoMtp = 1U << 25U,
  kSm87TargetAotAttentionNoCuBlasLt = 1U << 26U,
  kSm87TargetAotAttentionColdFirstPositionZero = 1U << 27U,
  kSm87TargetAotAttentionTypedCoreOutputSpans = 1U << 28U,
  kSm87TargetAotAttentionGatedOutputFeedsOProjection = 1U << 29U,
  kSm87TargetAotAttentionFinitePrecisionExecutionFrozen = 1U << 30U,
  kSm87TargetAotAttentionStableSigmoidEx2Bf16Rne = 1U << 31U,
};

inline constexpr std::uint32_t kSm87TargetAotAttentionRequiredPolicy =
    kSm87TargetAotAttentionCausal |
    kSm87TargetAotAttentionOnlineSoftmaxFp32 |
    kSm87TargetAotAttentionNoScoreMatrix |
    kSm87TargetAotAttentionNoSplitKv |
    kSm87TargetAotAttentionNoCrossCtaReduction |
    kSm87TargetAotAttentionBf16PublicationBeforeGate |
    kSm87TargetAotAttentionNoRequestJit |
    kSm87TargetAotAttentionNoFallback |
    kSm87TargetAotAttentionAccuracyUnqualified |
    kSm87TargetAotAttentionPackedQueryGqaReuse |
    kSm87TargetAotAttentionPerQueryCausalMask |
    kSm87TargetAotAttentionOrderedKvTransactionStage |
    kSm87TargetAotAttentionQkRmsNormBeforeRope |
    kSm87TargetAotAttentionPositionBoundRope |
    kSm87TargetAotAttentionProcessedQGatePublication |
    kSm87TargetAotAttentionProcessedKvBeforeCore |
    kSm87TargetAotAttentionTypedRawProducerEvents |
    kSm87TargetAotAttentionCanonicalPerHeadQGate |
    kSm87TargetAotAttentionGateBitExactSplitCopy |
    kSm87TargetAotAttentionVBitExactPassthrough |
    kSm87TargetAotAttentionCenteredQkNorm |
    kSm87TargetAotAttentionPartialNeox64Elements |
    kSm87TargetAotAttentionExactOneOver16Scale |
    kSm87TargetAotAttentionNhdColdKvEpoch |
    kSm87TargetAotAttentionTypedBufferSpans |
    kSm87TargetAotAttentionNoMtp |
    kSm87TargetAotAttentionNoCuBlasLt |
    kSm87TargetAotAttentionColdFirstPositionZero |
    kSm87TargetAotAttentionTypedCoreOutputSpans |
    kSm87TargetAotAttentionGatedOutputFeedsOProjection |
    kSm87TargetAotAttentionFinitePrecisionExecutionFrozen |
    kSm87TargetAotAttentionStableSigmoidEx2Bf16Rne;

struct Sm87TargetAotAttentionBufferContract {
  Sm87TargetAotAttentionBufferRole role =
      Sm87TargetAotAttentionBufferRole::kInvalid;
  Sm87TargetAotAttentionScalarType scalar_type =
      Sm87TargetAotAttentionScalarType::kInvalid;
  Sm87TargetAotAttentionBufferLayout layout =
      Sm87TargetAotAttentionBufferLayout::kInvalid;
  Sm87TargetAotAttentionBufferLifetime lifetime =
      Sm87TargetAotAttentionBufferLifetime::kInvalid;
  Sm87TargetAotAttentionBufferPublication publication =
      Sm87TargetAotAttentionBufferPublication::kInvalid;
  std::size_t elements = 0U;
  std::size_t bytes = 0U;
  std::size_t span_begin_elements = 0U;
  std::size_t span_end_elements = 0U;
};

[[nodiscard]] constexpr bool sm87_target_aot_same_attention_buffer_contract(
    const Sm87TargetAotAttentionBufferContract& left,
    const Sm87TargetAotAttentionBufferContract& right) noexcept {
  return left.role == right.role &&
         left.scalar_type == right.scalar_type &&
         left.layout == right.layout && left.lifetime == right.lifetime &&
         left.publication == right.publication &&
         left.elements == right.elements && left.bytes == right.bytes &&
         left.span_begin_elements == right.span_begin_elements &&
         left.span_end_elements == right.span_end_elements;
}

[[nodiscard]] constexpr Sm87TargetAotAttentionBufferContract
sm87_target_aot_attention_buffer_contract(
    const Sm87TargetAotAttentionBufferRole role,
    const std::size_t token_count) noexcept {
  if (!sm87_target_aot_exact_witness_tokens(token_count)) {
    return {};
  }
  Sm87TargetAotAttentionBufferContract result;
  result.role = role;
  result.span_begin_elements = 0U;
  if (role == Sm87TargetAotAttentionBufferRole::kRawQGate) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kBf16;
    result.layout = Sm87TargetAotAttentionBufferLayout::
        kRawQGateT24x2xD256PerHeadInterleaved;
    result.lifetime = Sm87TargetAotAttentionBufferLifetime::kLayerPreparation;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kProjectionOutput;
    result.elements =
        token_count * kSm87TargetAotAttentionRawQGateTokenStride;
  } else if (role == Sm87TargetAotAttentionBufferRole::kRawK ||
             role == Sm87TargetAotAttentionBufferRole::kRawV) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kBf16;
    result.layout = Sm87TargetAotAttentionBufferLayout::kRawKvT4xD256;
    result.lifetime = Sm87TargetAotAttentionBufferLifetime::kLayerPreparation;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kProjectionOutput;
    result.elements = token_count * kSm87TargetAotAttentionKvTokenStride;
  } else if (role == Sm87TargetAotAttentionBufferRole::kProcessedQ) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kBf16;
    result.layout = Sm87TargetAotAttentionBufferLayout::kProcessedQT24xD256;
    result.lifetime = Sm87TargetAotAttentionBufferLifetime::kLayerCore;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kPreprocessOutput;
    result.elements = token_count * kSm87TargetAotAttentionQueryTokenStride;
  } else if (role == Sm87TargetAotAttentionBufferRole::kProcessedGate) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kBf16;
    result.layout =
        Sm87TargetAotAttentionBufferLayout::kProcessedGateT24xD256;
    result.lifetime = Sm87TargetAotAttentionBufferLifetime::kLayerCore;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kPreprocessOutput;
    result.elements = token_count * kSm87TargetAotAttentionQueryTokenStride;
  } else if (role == Sm87TargetAotAttentionBufferRole::kProcessedK ||
             role == Sm87TargetAotAttentionBufferRole::kProcessedV) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kBf16;
    result.layout = role == Sm87TargetAotAttentionBufferRole::kProcessedK
                        ? Sm87TargetAotAttentionBufferLayout::
                              kProcessedKeyNhdT4xD256
                        : Sm87TargetAotAttentionBufferLayout::
                              kProcessedValueNhdT4xD256;
    result.lifetime =
        Sm87TargetAotAttentionBufferLifetime::
            kRequestTransactionUnpublishedUntilCommit;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kOrderedKvTransactionStage;
    result.elements = token_count * kSm87TargetAotAttentionKvTokenStride;
  } else if (role == Sm87TargetAotAttentionBufferRole::kQNormWeight ||
             role == Sm87TargetAotAttentionBufferRole::kKNormWeight) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kBf16;
    result.layout =
        Sm87TargetAotAttentionBufferLayout::kCenteredRmsNormWeightD256;
    result.lifetime = Sm87TargetAotAttentionBufferLifetime::kEngine;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kEngineReady;
    result.elements = kSm87TargetAotAttentionHeadDimension;
  } else if (role == Sm87TargetAotAttentionBufferRole::kRopeCos ||
             role == Sm87TargetAotAttentionBufferRole::kRopeSin) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kFp32;
    result.layout = Sm87TargetAotAttentionBufferLayout::kRopeTableT32Pairs;
    result.lifetime = Sm87TargetAotAttentionBufferLifetime::kEngine;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kEngineReady;
    result.elements = token_count * kSm87TargetAotAttentionRotaryPairs;
  } else if (role ==
             Sm87TargetAotAttentionBufferRole::kBf16AttentionOutput) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kBf16;
    result.layout =
        Sm87TargetAotAttentionBufferLayout::kBf16AttentionT24xD256;
    result.lifetime = Sm87TargetAotAttentionBufferLifetime::kLayerCore;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kAttentionCoreOutput;
    result.elements = token_count * kSm87TargetAotAttentionQueryTokenStride;
  } else if (role ==
             Sm87TargetAotAttentionBufferRole::kSigmoidGatedOutput) {
    result.scalar_type = Sm87TargetAotAttentionScalarType::kBf16;
    result.layout =
        Sm87TargetAotAttentionBufferLayout::kSigmoidGatedT24xD256;
    result.lifetime =
        Sm87TargetAotAttentionBufferLifetime::kLayerProjectionConsumer;
    result.publication =
        Sm87TargetAotAttentionBufferPublication::kSigmoidGateOutput;
    result.elements = token_count * kSm87TargetAotAttentionQueryTokenStride;
  } else {
    return {};
  }
  const std::size_t scalar_bytes =
      result.scalar_type == Sm87TargetAotAttentionScalarType::kBf16
          ? kSm87TargetAotAttentionBf16Bytes
          : kSm87TargetAotAttentionFp32Bytes;
  result.bytes = result.elements * scalar_bytes;
  result.span_end_elements = result.elements;
  return result;
}

struct Sm87TargetAotAttentionBufferSpec {
  Sm87TargetAotAttentionBufferContract contract{};
  std::uint64_t publication_identity = 0U;
  std::uint64_t ready_event_identity = 0U;

  [[nodiscard]] constexpr bool valid(
      const Sm87TargetAotAttentionBufferContract& expected) const noexcept {
    return sm87_target_aot_same_attention_buffer_contract(contract, expected) &&
           publication_identity != 0U && ready_event_identity != 0U;
  }
};

struct Sm87TargetAotAttentionPlan {
  Sm87TargetAotCapacityBucket capacity_bucket =
      Sm87TargetAotCapacityBucket::kInvalid;
  Sm87TargetAotAttentionTopology topology =
      Sm87TargetAotAttentionTopology::kInvalid;
  Sm87TargetAotAttentionNormContract qk_norm_contract =
      Sm87TargetAotAttentionNormContract::kInvalid;
  Sm87TargetAotAttentionRopeContract rope_contract =
      Sm87TargetAotAttentionRopeContract::kInvalid;
  Sm87TargetAotAttentionGateContract gate_contract =
      Sm87TargetAotAttentionGateContract::kInvalid;
  Sm87TargetAotAttentionValueContract value_contract =
      Sm87TargetAotAttentionValueContract::kInvalid;
  Sm87TargetAotAttentionKvCacheLayout kv_cache_layout =
      Sm87TargetAotAttentionKvCacheLayout::kInvalid;
  Sm87TargetAotAttentionKvResetContract kv_reset_contract =
      Sm87TargetAotAttentionKvResetContract::kInvalid;
  Sm87TargetAotAttentionNumericalContract numerical_execution{};
  Sm87TargetAotAttentionPreprocessNumericalContract preprocess_numerical{};
  std::size_t token_count = 0U;
  std::size_t flattened_query_rows = 0U;
  std::size_t query_tiles_per_kv_head = 0U;
  std::size_t total_ctas = 0U;
  std::size_t query_tail_rows = 0U;
  std::size_t kv_tiles = 0U;
  std::size_t kv_tail_tokens = 0U;
  std::size_t raw_q_gate_token_stride_elements = 0U;
  std::size_t query_token_stride_elements = 0U;
  std::size_t kv_token_stride_elements = 0U;
  std::size_t q_rmsnorm_head_rows = 0U;
  std::size_t k_rmsnorm_head_rows = 0U;
  std::size_t q_rope_head_rows = 0U;
  std::size_t k_rope_head_rows = 0U;
  std::size_t processed_q_gate_head_rows = 0U;
  std::size_t processed_k_head_rows = 0U;
  std::size_t published_v_head_rows = 0U;
  std::size_t bf16_attention_output_head_rows = 0U;
  std::size_t sigmoid_gated_output_head_rows = 0U;
  std::size_t position_rows = 0U;
  std::size_t q_norm_weight_elements = 0U;
  std::size_t k_norm_weight_elements = 0U;
  std::size_t rotary_elements = 0U;
  std::size_t rotary_pairs = 0U;
  std::size_t rope_passthrough_elements = 0U;
  std::size_t rope_table_elements_per_position = 0U;
  std::uint32_t rms_epsilon_fp32_bits = 0U;
  std::uint32_t attention_scale_fp32_bits = 0U;
  std::uint32_t attention_scale_numerator = 0U;
  std::uint32_t attention_scale_denominator = 0U;
  std::uint64_t rope_theta = 0U;
  std::array<Sm87TargetAotAttentionBufferContract,
             kSm87TargetAotAttentionBufferCount>
      buffers{};
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t split_kv_workspace_bytes = 0U;
  std::uint32_t policy = 0U;
  bool qk_fp32_accumulation = false;
  bool pv_fp32_accumulation = false;
  bool cp_async_kv = false;
  bool kv_ping_pong = false;
  bool bf16_attention_round_before_sigmoid_gate = false;
  bool qk_rmsnorm_before_rope = false;
  bool rope_uses_explicit_position_range = false;
  bool gate_bit_exact_split_copy = false;
  bool gate_bypasses_qk_norm = false;
  bool gate_bypasses_rope = false;
  bool v_bit_exact_passthrough = false;
  bool processed_q_gate_published_before_core = false;
  bool processed_kv_staged_core_visible_before_core = false;
  bool ordered_kv_transaction_stage = false;
  bool cold_first_position_zero_only = false;
  bool cuda_implementation_present = false;
  bool static_resources_qualified = false;
  bool numerical_reduction_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;

  [[nodiscard]] constexpr bool valid() const noexcept;
};

struct Sm87TargetAotAttentionTask {
  Sm87TargetAotAttentionTaskRole role =
      Sm87TargetAotAttentionTaskRole::kInvalid;
  Sm87TargetAotAttentionBufferLayout raw_q_gate_layout =
      Sm87TargetAotAttentionBufferLayout::kInvalid;
  Sm87TargetAotAttentionBufferLayout processed_q_layout =
      Sm87TargetAotAttentionBufferLayout::kInvalid;
  Sm87TargetAotAttentionBufferLayout processed_gate_layout =
      Sm87TargetAotAttentionBufferLayout::kInvalid;
  Sm87TargetAotAttentionBufferLayout processed_k_layout =
      Sm87TargetAotAttentionBufferLayout::kInvalid;
  Sm87TargetAotAttentionBufferLayout processed_v_layout =
      Sm87TargetAotAttentionBufferLayout::kInvalid;
  std::size_t kv_head = 0U;
  std::size_t query_tile = 0U;
  std::size_t flattened_query_begin = 0U;
  std::size_t flattened_query_end = 0U;
  std::size_t flattened_query_rows = 0U;
  std::size_t kv_position_span_begin = 0U;
  std::size_t kv_position_span_end = 0U;
  bool valid = false;
};

struct Sm87TargetAotAttentionQueryRow {
  std::size_t token = 0U;
  std::size_t kv_head = 0U;
  std::size_t local_query_head = 0U;
  std::size_t query_head = 0U;
  std::size_t raw_query_element_offset = 0U;
  std::size_t raw_gate_element_offset = 0U;
  std::size_t raw_k_element_offset = 0U;
  std::size_t raw_v_element_offset = 0U;
  std::size_t query_element_offset = 0U;
  std::size_t gate_element_offset = 0U;
  std::size_t kv_head_element_offset = 0U;
  std::size_t processed_k_element_offset = 0U;
  std::size_t processed_v_element_offset = 0U;
  std::size_t absolute_position = 0U;
  std::size_t visible_kv_token_end = 0U;
  bool gate_is_bit_exact_split_copy = false;
  bool valid = false;
};

inline constexpr std::size_t kSm87TargetAotAttentionBaseIdentityCount = 14U;
inline constexpr std::size_t kSm87TargetAotAttentionBindingIdentityCount =
    kSm87TargetAotAttentionBaseIdentityCount +
    2U * kSm87TargetAotAttentionBufferCount;

struct Sm87TargetAotAttentionBinding {
  Sm87TargetAotCapacityBucket capacity_bucket =
      Sm87TargetAotCapacityBucket::kInvalid;
  Sm87TargetAotAttentionTopology topology =
      Sm87TargetAotAttentionTopology::kInvalid;
  std::size_t layer_index = kSm87TargetAotAttentionLayerCount;
  std::size_t first_position = 0U;
  std::size_t initial_kv_length = 0U;
  std::uint64_t cold_reset_epoch = 0U;
  std::uint64_t staged_kv_epoch = 0U;
  std::uint64_t plan_identity = 0U;
  std::uint64_t tactic_identity = 0U;
  std::uint64_t reduction_identity = 0U;
  std::uint64_t launcher_identity = 0U;
  std::uint64_t qk_preprocess_identity = 0U;
  std::uint64_t gate_split_copy_identity = 0U;
  std::uint64_t v_passthrough_identity = 0U;
  std::uint64_t attention_scale_identity = 0U;
  std::uint64_t position_reset_identity = 0U;
  std::uint64_t kv_cache_reset_epoch_identity = 0U;
  std::uint64_t ordered_kv_transaction_stage_identity = 0U;
  std::uint64_t core_publication_identity = 0U;
  std::uint64_t sigmoid_gate_identity = 0U;
  std::uint64_t core_completion_event_identity = 0U;
  std::array<Sm87TargetAotAttentionBufferSpec,
             kSm87TargetAotAttentionBufferCount>
      buffers{};

  [[nodiscard]] constexpr bool valid(
      const Sm87TargetAotAttentionPlan& plan) const noexcept {
    if (!plan.valid() || capacity_bucket != plan.capacity_bucket ||
        topology != plan.topology ||
        layer_index >= kSm87TargetAotAttentionLayerCount ||
        (layer_index + 1U) % kSm87TargetAotAttentionLayerInterval != 0U ||
        first_position != 0U || initial_kv_length != 0U ||
        cold_reset_epoch == 0U || staged_kv_epoch != cold_reset_epoch) {
      return false;
    }
    std::array<std::uint64_t, kSm87TargetAotAttentionBindingIdentityCount>
        identities{};
    const std::array<std::uint64_t,
                     kSm87TargetAotAttentionBaseIdentityCount>
        base{{plan_identity,
              tactic_identity,
              reduction_identity,
              launcher_identity,
              qk_preprocess_identity,
              gate_split_copy_identity,
              v_passthrough_identity,
              attention_scale_identity,
              position_reset_identity,
              kv_cache_reset_epoch_identity,
              ordered_kv_transaction_stage_identity,
              core_publication_identity,
              sigmoid_gate_identity,
              core_completion_event_identity}};
    for (std::size_t index = 0U; index < base.size(); ++index) {
      identities[index] = base[index];
    }
    std::size_t identity_index = base.size();
    for (std::size_t index = 0U; index < buffers.size(); ++index) {
      if (!buffers[index].valid(plan.buffers[index])) {
        return false;
      }
      identities[identity_index++] = buffers[index].publication_identity;
      identities[identity_index++] = buffers[index].ready_event_identity;
    }
    return identity_index == identities.size() &&
           sm87_target_aot_nonzero_unique_identities(identities);
  }
};

[[nodiscard]] constexpr bool sm87_target_aot_same_attention_buffer_spec(
    const Sm87TargetAotAttentionBufferSpec& left,
    const Sm87TargetAotAttentionBufferSpec& right) noexcept {
  return sm87_target_aot_same_attention_buffer_contract(left.contract,
                                                         right.contract) &&
         left.publication_identity == right.publication_identity &&
         left.ready_event_identity == right.ready_event_identity;
}

[[nodiscard]] constexpr bool sm87_target_aot_same_attention_binding(
    const Sm87TargetAotAttentionBinding& left,
    const Sm87TargetAotAttentionBinding& right) noexcept {
  if (left.capacity_bucket != right.capacity_bucket ||
      left.topology != right.topology ||
      left.layer_index != right.layer_index ||
      left.first_position != right.first_position ||
      left.initial_kv_length != right.initial_kv_length ||
      left.cold_reset_epoch != right.cold_reset_epoch ||
      left.staged_kv_epoch != right.staged_kv_epoch ||
      left.plan_identity != right.plan_identity ||
      left.tactic_identity != right.tactic_identity ||
      left.reduction_identity != right.reduction_identity ||
      left.launcher_identity != right.launcher_identity ||
      left.qk_preprocess_identity != right.qk_preprocess_identity ||
      left.gate_split_copy_identity != right.gate_split_copy_identity ||
      left.v_passthrough_identity != right.v_passthrough_identity ||
      left.attention_scale_identity != right.attention_scale_identity ||
      left.position_reset_identity != right.position_reset_identity ||
      left.kv_cache_reset_epoch_identity !=
          right.kv_cache_reset_epoch_identity ||
      left.ordered_kv_transaction_stage_identity !=
          right.ordered_kv_transaction_stage_identity ||
      left.core_publication_identity != right.core_publication_identity ||
      left.sigmoid_gate_identity != right.sigmoid_gate_identity ||
      left.core_completion_event_identity !=
          right.core_completion_event_identity) {
    return false;
  }
  for (std::size_t index = 0U; index < left.buffers.size(); ++index) {
    if (!sm87_target_aot_same_attention_buffer_spec(left.buffers[index],
                                                     right.buffers[index])) {
      return false;
    }
  }
  return true;
}

struct Sm87TargetAotAttentionReceipt {
  Sm87TargetAotAttentionBinding binding{};
  std::size_t token_count = 0U;
  std::array<std::size_t, kSm87TargetAotAttentionBufferCount>
      completed_buffer_elements{};
  std::size_t completed_q_rmsnorm_head_rows = 0U;
  std::size_t completed_k_rmsnorm_head_rows = 0U;
  std::size_t completed_q_rope_head_rows = 0U;
  std::size_t completed_k_rope_head_rows = 0U;
  std::size_t completed_processed_q_gate_head_rows = 0U;
  std::size_t completed_processed_k_head_rows = 0U;
  std::size_t completed_published_v_head_rows = 0U;
  std::size_t completed_bf16_attention_output_head_rows = 0U;
  std::size_t completed_sigmoid_gated_output_head_rows = 0U;
  std::size_t completed_position_rows = 0U;
  std::size_t processed_kv_core_visible_begin = 0U;
  std::size_t processed_kv_core_visible_end = 0U;
  std::uint64_t staged_kv_epoch = 0U;
  std::size_t completed_ctas = 0U;
  std::size_t completed_query_head_rows = 0U;
  std::size_t completed_token_rows = 0U;
  std::size_t core_kv_visible_end = 0U;
  std::uint64_t core_publication_identity = 0U;
  std::uint64_t sigmoid_gate_identity = 0U;
  std::uint64_t bf16_attention_output_publication_identity = 0U;
  std::uint64_t bf16_attention_output_ready_event_identity = 0U;
  std::uint64_t sigmoid_gated_output_publication_identity = 0U;
  std::uint64_t sigmoid_gated_output_ready_event_identity = 0U;
  std::uint64_t core_completion_event_identity = 0U;
  std::size_t fallback_hits = 0U;
  bool position_reset_complete = false;
  bool qk_rmsnorm_complete = false;
  bool rope_position_complete = false;
  bool gate_bit_exact_split_copy_complete = false;
  bool gate_bypassed_qk_norm_and_rope = false;
  bool v_bit_exact_passthrough_complete = false;
  bool processed_q_gate_published_before_core = false;
  bool processed_kv_staged_core_visible_before_core = false;
  bool ordered_kv_transaction_stage_complete = false;
  bool bf16_attention_published = false;
  bool sigmoid_gate_after_bf16_ready = false;
  bool sigmoid_gate_applied = false;

  [[nodiscard]] constexpr bool complete(
      const Sm87TargetAotAttentionPlan& plan,
      const Sm87TargetAotAttentionBinding& expected_binding) const noexcept {
    const std::size_t bf16_output_index =
        sm87_target_aot_attention_buffer_index(
            Sm87TargetAotAttentionBufferRole::kBf16AttentionOutput);
    const std::size_t gated_output_index =
        sm87_target_aot_attention_buffer_index(
            Sm87TargetAotAttentionBufferRole::kSigmoidGatedOutput);
    if (!expected_binding.valid(plan) || !binding.valid(plan) ||
        !sm87_target_aot_same_attention_binding(binding, expected_binding) ||
        token_count != plan.token_count ||
        bf16_output_index >= expected_binding.buffers.size() ||
        gated_output_index >= expected_binding.buffers.size() ||
        core_publication_identity !=
            expected_binding.core_publication_identity ||
        sigmoid_gate_identity != expected_binding.sigmoid_gate_identity ||
        bf16_attention_output_publication_identity !=
            expected_binding.buffers[bf16_output_index]
                .publication_identity ||
        bf16_attention_output_ready_event_identity !=
            expected_binding.buffers[bf16_output_index]
                .ready_event_identity ||
        sigmoid_gated_output_publication_identity !=
            expected_binding.buffers[gated_output_index]
                .publication_identity ||
        sigmoid_gated_output_ready_event_identity !=
            expected_binding.buffers[gated_output_index]
                .ready_event_identity ||
        core_completion_event_identity !=
            expected_binding.core_completion_event_identity ||
        staged_kv_epoch != expected_binding.staged_kv_epoch) {
      return false;
    }
    for (std::size_t index = 0U; index < binding.buffers.size(); ++index) {
      if (completed_buffer_elements[index] !=
          binding.buffers[index].contract.elements) {
        return false;
      }
    }
    return completed_q_rmsnorm_head_rows == plan.q_rmsnorm_head_rows &&
           completed_k_rmsnorm_head_rows == plan.k_rmsnorm_head_rows &&
           completed_q_rope_head_rows == plan.q_rope_head_rows &&
           completed_k_rope_head_rows == plan.k_rope_head_rows &&
           completed_processed_q_gate_head_rows ==
               plan.processed_q_gate_head_rows &&
           completed_processed_k_head_rows == plan.processed_k_head_rows &&
           completed_published_v_head_rows == plan.published_v_head_rows &&
           completed_bf16_attention_output_head_rows ==
               plan.bf16_attention_output_head_rows &&
           completed_sigmoid_gated_output_head_rows ==
               plan.sigmoid_gated_output_head_rows &&
           completed_position_rows == plan.position_rows &&
           processed_kv_core_visible_begin == expected_binding.first_position &&
           processed_kv_core_visible_end ==
               expected_binding.first_position + plan.token_count &&
           completed_ctas == plan.total_ctas &&
           completed_query_head_rows ==
               plan.token_count * kSm87TargetAotAttentionQueryHeads &&
           completed_token_rows == plan.token_count &&
           core_kv_visible_end == plan.token_count && fallback_hits == 0U &&
           position_reset_complete && qk_rmsnorm_complete &&
           rope_position_complete && gate_bit_exact_split_copy_complete &&
           gate_bypassed_qk_norm_and_rope &&
           v_bit_exact_passthrough_complete &&
           processed_q_gate_published_before_core &&
           processed_kv_staged_core_visible_before_core &&
           ordered_kv_transaction_stage_complete &&
           bf16_attention_published &&
           sigmoid_gate_after_bf16_ready && sigmoid_gate_applied;
  }
};

[[nodiscard]] constexpr Sm87TargetAotAttentionPlan
sm87_target_aot_attention_plan(const std::size_t token_count) noexcept {
  if (!sm87_target_aot_exact_witness_tokens(token_count)) {
    return {};
  }
  Sm87TargetAotAttentionPlan plan;
  plan.capacity_bucket =
      sm87_target_aot_capacity_for_witness(token_count).bucket;
  plan.topology = Sm87TargetAotAttentionTopology::kQ128Kv32TwoStage;
  plan.qk_norm_contract = Sm87TargetAotAttentionNormContract::
      kCenteredRmsNormBf16WeightFp32Epsilon;
  plan.rope_contract = Sm87TargetAotAttentionRopeContract::
      kPartialNeox64Elements32PairsFp32Table;
  plan.gate_contract = Sm87TargetAotAttentionGateContract::
      kBitExactPerHeadSplitCopyNoNormNoRope;
  plan.value_contract = Sm87TargetAotAttentionValueContract::
      kBitExactPassthroughToOrderedKvTransactionStage;
  plan.kv_cache_layout =
      Sm87TargetAotAttentionKvCacheLayout::kNhdTokenHeadDimension;
  plan.kv_reset_contract = Sm87TargetAotAttentionKvResetContract::
      kColdEpochZeroLengthCurrentLayer;
  plan.numerical_execution.exp_contract = Sm87TargetAotAttentionExpContract::
      kEx2ApproxF32AfterFp32Log2eMultiply;
  plan.numerical_execution.qk_traversal =
      Sm87TargetAotAttentionQkTraversalContract::
          kMmaSyncM16N16K16DimensionAscendingScoreSubtilesAscending;
  plan.numerical_execution.probability_publication =
      Sm87TargetAotAttentionProbabilityContract::
          kExpF32ToBf16RneBeforeDenominatorAndPv;
  plan.numerical_execution.denominator_update =
      Sm87TargetAotAttentionDenominatorContract::
          kRescalePriorThenRegister0145ThenXor1Xor2;
  plan.numerical_execution.pv_traversal =
      Sm87TargetAotAttentionPvTraversalContract::
          kMmaSyncBf16ProbabilityBf16VToFp32OutputDimensionAscending;
  plan.numerical_execution.output_publication =
      Sm87TargetAotAttentionOutputContract::
          kFp32ReciprocalThenNumeratorMultiplyThenBf16Rne;
  plan.numerical_execution.sigmoid_publication =
      Sm87TargetAotAttentionSigmoidContract::
          kStableSignBranchEx2ApproxF32TimesBf16AttentionThenBf16Rne;
  plan.numerical_execution.log2e_fp32_bits =
      kSm87TargetAotAttentionLog2EFp32Bits;
  plan.numerical_execution.mma_m = 16U;
  plan.numerical_execution.mma_n = 16U;
  plan.numerical_execution.mma_k = 16U;
  plan.numerical_execution.qk_head_dimension_tiles = 16U;
  plan.numerical_execution.kv_stage_score_subtiles = 2U;
  plan.numerical_execution.pv_output_dimension_tiles = 16U;
  plan.numerical_execution.denominator_register_offsets = {{0U, 1U, 4U,
                                                              5U}};
  plan.numerical_execution.denominator_shuffle_xor_masks = {{1U, 2U}};
  plan.numerical_execution.attention_scale_after_qk_accumulation = true;
  plan.numerical_execution
      .prior_denominator_rescaled_before_probability_add = true;
  plan.numerical_execution.prior_output_rescaled_before_pv_mma = true;
  plan.numerical_execution.probability_bf16_reused_by_denominator = true;
  plan.numerical_execution.probability_bf16_reused_by_pv_mma = true;
  plan.numerical_execution.bf16_attention_reused_by_sigmoid_epilogue = true;
  plan.preprocess_numerical.rms_reduction =
      Sm87TargetAotAttentionRmsReductionContract::
          kPromptWide128FmafDAndD128ThenShared64Pair32Pair96Shuffle16To1;
  plan.preprocess_numerical.inverse_rms =
      Sm87TargetAotAttentionRsqrtContract::
          kRsqrtfFp32SumDiv256PlusFp32Epsilon;
  plan.preprocess_numerical.norm_publication =
      Sm87TargetAotAttentionNormPublicationContract::
          kBf16WeightPlusOneValueTimesInverseThenGammaBf16Rne;
  plan.preprocess_numerical.rope_fma =
      Sm87TargetAotAttentionRopeFmaContract::
          kNeoxSeparateSinProductThenCosFmaBf16Rne;
  plan.preprocess_numerical.rope_mapping =
      Sm87TargetAotAttentionRopeMappingContract::
          kRotateD0To31WithDPlus32Tail64To255Passthrough;
  plan.preprocess_numerical.threads = 128U;
  plan.preprocess_numerical.head_dimension =
      kSm87TargetAotAttentionHeadDimension;
  plan.preprocess_numerical.thread_dimension_offsets = {{0U, 128U}};
  plan.preprocess_numerical.shared_tree_offsets = {{0U, 64U, 32U, 96U}};
  plan.preprocess_numerical.shuffle_down_strides = {{16U, 8U, 4U, 2U,
                                                     1U}};
  plan.preprocess_numerical.rope_pair_offsets = {{0U, 32U}};
  plan.preprocess_numerical.rope_pair_count =
      kSm87TargetAotAttentionRotaryPairs;
  plan.preprocess_numerical.rope_tail_begin =
      kSm87TargetAotAttentionRotaryElements;
  plan.preprocess_numerical.rope_tail_end =
      kSm87TargetAotAttentionHeadDimension;
  plan.preprocess_numerical.square_uses_fmaf_with_positive_zero = true;
  plan.preprocess_numerical.pair_add_low_square_before_high_square = true;
  plan.preprocess_numerical
      .epsilon_added_after_divide_by_head_dimension = true;
  plan.preprocess_numerical
      .norm_weight_decoded_from_bf16_then_fp32_plus_one = true;
  plan.preprocess_numerical
      .norm_multiplies_value_by_inverse_before_gamma = true;
  plan.preprocess_numerical
      .normalized_qk_published_bf16_rne_before_rope = true;
  plan.preprocess_numerical.rope_consumes_published_bf16_qk = true;
  plan.preprocess_numerical.rope_sine_product_rounded_before_fma = true;
  plan.preprocess_numerical.rope_output_published_bf16_rne = true;
  plan.preprocess_numerical.rope_tail_is_bit_exact_normalized_bf16 = true;
  plan.token_count = token_count;
  plan.flattened_query_rows =
      token_count * kSm87TargetAotAttentionQueriesPerKv;
  plan.query_tiles_per_kv_head =
      (plan.flattened_query_rows + kSm87TargetAotAttentionQueryRows - 1U) /
      kSm87TargetAotAttentionQueryRows;
  plan.total_ctas =
      plan.query_tiles_per_kv_head * kSm87TargetAotAttentionKvHeads;
  plan.query_tail_rows =
      plan.flattened_query_rows % kSm87TargetAotAttentionQueryRows;
  if (plan.query_tail_rows == 0U) {
    plan.query_tail_rows = kSm87TargetAotAttentionQueryRows;
  }
  plan.kv_tiles =
      (token_count + kSm87TargetAotAttentionKvTokens - 1U) /
      kSm87TargetAotAttentionKvTokens;
  plan.kv_tail_tokens = token_count % kSm87TargetAotAttentionKvTokens;
  if (plan.kv_tail_tokens == 0U) {
    plan.kv_tail_tokens = kSm87TargetAotAttentionKvTokens;
  }
  plan.raw_q_gate_token_stride_elements =
      kSm87TargetAotAttentionRawQGateTokenStride;
  plan.query_token_stride_elements =
      kSm87TargetAotAttentionQueryTokenStride;
  plan.kv_token_stride_elements = kSm87TargetAotAttentionKvTokenStride;
  plan.q_rmsnorm_head_rows =
      token_count * kSm87TargetAotAttentionQueryHeads;
  plan.k_rmsnorm_head_rows = token_count * kSm87TargetAotAttentionKvHeads;
  plan.q_rope_head_rows = plan.q_rmsnorm_head_rows;
  plan.k_rope_head_rows = plan.k_rmsnorm_head_rows;
  plan.processed_q_gate_head_rows = plan.q_rmsnorm_head_rows;
  plan.processed_k_head_rows = plan.k_rmsnorm_head_rows;
  plan.published_v_head_rows = plan.k_rmsnorm_head_rows;
  plan.bf16_attention_output_head_rows = plan.q_rmsnorm_head_rows;
  plan.sigmoid_gated_output_head_rows = plan.q_rmsnorm_head_rows;
  plan.position_rows = token_count;
  plan.q_norm_weight_elements = kSm87TargetAotAttentionHeadDimension;
  plan.k_norm_weight_elements = kSm87TargetAotAttentionHeadDimension;
  plan.rotary_elements = kSm87TargetAotAttentionRotaryElements;
  plan.rotary_pairs = kSm87TargetAotAttentionRotaryPairs;
  plan.rope_passthrough_elements =
      kSm87TargetAotAttentionRopePassthroughElements;
  plan.rope_table_elements_per_position =
      kSm87TargetAotAttentionRotaryPairs;
  plan.rms_epsilon_fp32_bits =
      kSm87TargetAotAttentionRmsEpsilonFp32Bits;
  plan.attention_scale_fp32_bits =
      kSm87TargetAotAttentionScaleFp32Bits;
  plan.attention_scale_numerator = 1U;
  plan.attention_scale_denominator = 16U;
  plan.rope_theta = kSm87TargetAotAttentionRopeTheta;
  for (std::size_t index = 0U; index < plan.buffers.size(); ++index) {
    plan.buffers[index] = sm87_target_aot_attention_buffer_contract(
        kSm87TargetAotAttentionBufferRoles[index], token_count);
  }
  plan.dynamic_shared_bytes = kSm87TargetAotAttentionSharedBytes;
  plan.split_kv_workspace_bytes = 0U;
  plan.policy = kSm87TargetAotAttentionRequiredPolicy;
  plan.qk_fp32_accumulation = true;
  plan.pv_fp32_accumulation = true;
  plan.cp_async_kv = true;
  plan.kv_ping_pong = true;
  plan.bf16_attention_round_before_sigmoid_gate = true;
  plan.qk_rmsnorm_before_rope = true;
  plan.rope_uses_explicit_position_range = true;
  plan.gate_bit_exact_split_copy = true;
  plan.gate_bypasses_qk_norm = true;
  plan.gate_bypasses_rope = true;
  plan.v_bit_exact_passthrough = true;
  plan.processed_q_gate_published_before_core = true;
  plan.processed_kv_staged_core_visible_before_core = true;
  plan.ordered_kv_transaction_stage = true;
  plan.cold_first_position_zero_only = true;
  return plan;
}

[[nodiscard]] constexpr bool sm87_target_aot_same_attention_plan(
    const Sm87TargetAotAttentionPlan& left,
    const Sm87TargetAotAttentionPlan& right) noexcept {
  if (left.topology != right.topology ||
      left.capacity_bucket != right.capacity_bucket ||
      left.qk_norm_contract != right.qk_norm_contract ||
      left.rope_contract != right.rope_contract ||
      left.gate_contract != right.gate_contract ||
      left.value_contract != right.value_contract ||
      left.kv_cache_layout != right.kv_cache_layout ||
      left.kv_reset_contract != right.kv_reset_contract ||
      !sm87_target_aot_same_attention_numerical_contract(
          left.numerical_execution, right.numerical_execution) ||
      !sm87_target_aot_same_attention_preprocess_numerical_contract(
          left.preprocess_numerical, right.preprocess_numerical) ||
      left.token_count != right.token_count ||
      left.flattened_query_rows != right.flattened_query_rows ||
      left.query_tiles_per_kv_head != right.query_tiles_per_kv_head ||
      left.total_ctas != right.total_ctas ||
      left.query_tail_rows != right.query_tail_rows ||
      left.kv_tiles != right.kv_tiles ||
      left.kv_tail_tokens != right.kv_tail_tokens ||
      left.raw_q_gate_token_stride_elements !=
          right.raw_q_gate_token_stride_elements ||
      left.query_token_stride_elements !=
          right.query_token_stride_elements ||
      left.kv_token_stride_elements != right.kv_token_stride_elements ||
      left.q_rmsnorm_head_rows != right.q_rmsnorm_head_rows ||
      left.k_rmsnorm_head_rows != right.k_rmsnorm_head_rows ||
      left.q_rope_head_rows != right.q_rope_head_rows ||
      left.k_rope_head_rows != right.k_rope_head_rows ||
      left.processed_q_gate_head_rows !=
          right.processed_q_gate_head_rows ||
      left.processed_k_head_rows != right.processed_k_head_rows ||
      left.published_v_head_rows != right.published_v_head_rows ||
      left.bf16_attention_output_head_rows !=
          right.bf16_attention_output_head_rows ||
      left.sigmoid_gated_output_head_rows !=
          right.sigmoid_gated_output_head_rows ||
      left.position_rows != right.position_rows ||
      left.q_norm_weight_elements != right.q_norm_weight_elements ||
      left.k_norm_weight_elements != right.k_norm_weight_elements ||
      left.rotary_elements != right.rotary_elements ||
      left.rotary_pairs != right.rotary_pairs ||
      left.rope_passthrough_elements != right.rope_passthrough_elements ||
      left.rope_table_elements_per_position !=
          right.rope_table_elements_per_position ||
      left.rms_epsilon_fp32_bits != right.rms_epsilon_fp32_bits ||
      left.attention_scale_fp32_bits != right.attention_scale_fp32_bits ||
      left.attention_scale_numerator != right.attention_scale_numerator ||
      left.attention_scale_denominator != right.attention_scale_denominator ||
      left.rope_theta != right.rope_theta ||
      left.dynamic_shared_bytes != right.dynamic_shared_bytes ||
      left.split_kv_workspace_bytes != right.split_kv_workspace_bytes ||
      left.policy != right.policy ||
      left.qk_fp32_accumulation != right.qk_fp32_accumulation ||
      left.pv_fp32_accumulation != right.pv_fp32_accumulation ||
      left.cp_async_kv != right.cp_async_kv ||
      left.kv_ping_pong != right.kv_ping_pong ||
      left.bf16_attention_round_before_sigmoid_gate !=
          right.bf16_attention_round_before_sigmoid_gate ||
      left.qk_rmsnorm_before_rope != right.qk_rmsnorm_before_rope ||
      left.rope_uses_explicit_position_range !=
          right.rope_uses_explicit_position_range ||
      left.gate_bit_exact_split_copy != right.gate_bit_exact_split_copy ||
      left.gate_bypasses_qk_norm != right.gate_bypasses_qk_norm ||
      left.gate_bypasses_rope != right.gate_bypasses_rope ||
      left.v_bit_exact_passthrough != right.v_bit_exact_passthrough ||
      left.processed_q_gate_published_before_core !=
          right.processed_q_gate_published_before_core ||
      left.processed_kv_staged_core_visible_before_core !=
          right.processed_kv_staged_core_visible_before_core ||
      left.ordered_kv_transaction_stage !=
          right.ordered_kv_transaction_stage ||
      left.cold_first_position_zero_only !=
          right.cold_first_position_zero_only ||
      left.cuda_implementation_present != right.cuda_implementation_present ||
      left.static_resources_qualified != right.static_resources_qualified ||
      left.numerical_reduction_qualified !=
          right.numerical_reduction_qualified ||
      left.numerical_contract_qualified !=
          right.numerical_contract_qualified ||
      left.production_dispatch_eligible !=
          right.production_dispatch_eligible) {
    return false;
  }
  for (std::size_t index = 0U; index < left.buffers.size(); ++index) {
    if (!sm87_target_aot_same_attention_buffer_contract(
            left.buffers[index], right.buffers[index])) {
      return false;
    }
  }
  return true;
}

constexpr bool Sm87TargetAotAttentionPlan::valid() const noexcept {
  return topology == Sm87TargetAotAttentionTopology::kQ128Kv32TwoStage &&
         sm87_target_aot_same_attention_plan(
             *this, sm87_target_aot_attention_plan(token_count));
}

[[nodiscard]] constexpr Sm87TargetAotAttentionTask
sm87_target_aot_attention_task(const Sm87TargetAotAttentionPlan& plan,
                               const std::size_t linear_cta) noexcept {
  if (!plan.valid() || linear_cta >= plan.total_ctas) {
    return {};
  }
  const std::size_t kv_head =
      linear_cta / plan.query_tiles_per_kv_head;
  const std::size_t query_tile =
      linear_cta % plan.query_tiles_per_kv_head;
  const std::size_t query_begin =
      query_tile * kSm87TargetAotAttentionQueryRows;
  const std::size_t remaining = plan.flattened_query_rows - query_begin;
  const std::size_t rows =
      remaining < kSm87TargetAotAttentionQueryRows
          ? remaining
          : kSm87TargetAotAttentionQueryRows;
  return {Sm87TargetAotAttentionTaskRole::kCausalQ128AgainstOrderedNhdKv,
          Sm87TargetAotAttentionBufferLayout::
              kRawQGateT24x2xD256PerHeadInterleaved,
          Sm87TargetAotAttentionBufferLayout::kProcessedQT24xD256,
          Sm87TargetAotAttentionBufferLayout::kProcessedGateT24xD256,
          Sm87TargetAotAttentionBufferLayout::kProcessedKeyNhdT4xD256,
          Sm87TargetAotAttentionBufferLayout::kProcessedValueNhdT4xD256,
          kv_head,
          query_tile,
          query_begin,
          query_begin + rows,
          rows,
          0U,
          plan.token_count,
          true};
}

[[nodiscard]] constexpr bool sm87_target_aot_same_attention_task(
    const Sm87TargetAotAttentionTask& left,
    const Sm87TargetAotAttentionTask& right) noexcept {
  return left.role == right.role &&
         left.raw_q_gate_layout == right.raw_q_gate_layout &&
         left.processed_q_layout == right.processed_q_layout &&
         left.processed_gate_layout == right.processed_gate_layout &&
         left.processed_k_layout == right.processed_k_layout &&
         left.processed_v_layout == right.processed_v_layout &&
         left.kv_head == right.kv_head &&
         left.query_tile == right.query_tile &&
         left.flattened_query_begin == right.flattened_query_begin &&
         left.flattened_query_end == right.flattened_query_end &&
         left.flattened_query_rows == right.flattened_query_rows &&
         left.kv_position_span_begin == right.kv_position_span_begin &&
         left.kv_position_span_end == right.kv_position_span_end &&
         left.valid == right.valid;
}

[[nodiscard]] constexpr Sm87TargetAotAttentionQueryRow
sm87_target_aot_attention_query_row(
    const Sm87TargetAotAttentionPlan& plan,
    const Sm87TargetAotAttentionTask& task,
    const std::size_t row_in_task) noexcept {
  if (!plan.valid() || !task.valid ||
      row_in_task >= task.flattened_query_rows ||
      task.kv_head >= kSm87TargetAotAttentionKvHeads ||
      task.query_tile >= plan.query_tiles_per_kv_head) {
    return {};
  }
  const std::size_t canonical_linear_cta =
      task.kv_head * plan.query_tiles_per_kv_head + task.query_tile;
  const Sm87TargetAotAttentionTask canonical_task =
      sm87_target_aot_attention_task(plan, canonical_linear_cta);
  if (!canonical_task.valid ||
      !sm87_target_aot_same_attention_task(task, canonical_task)) {
    return {};
  }
  const std::size_t flattened = task.flattened_query_begin + row_in_task;
  if (flattened >= plan.flattened_query_rows) {
    return {};
  }
  const std::size_t token =
      flattened / kSm87TargetAotAttentionQueriesPerKv;
  const std::size_t local_query_head =
      flattened % kSm87TargetAotAttentionQueriesPerKv;
  const std::size_t query_head =
      task.kv_head * kSm87TargetAotAttentionQueriesPerKv +
      local_query_head;
  const std::size_t raw_q_gate_head_offset =
      token * plan.raw_q_gate_token_stride_elements +
      query_head * 2U * kSm87TargetAotAttentionHeadDimension;
  const std::size_t raw_kv_offset =
      token * plan.kv_token_stride_elements +
      task.kv_head * kSm87TargetAotAttentionHeadDimension;
  const std::size_t processed_q_offset =
      token * plan.query_token_stride_elements +
      query_head * kSm87TargetAotAttentionHeadDimension;
  return {token,
          task.kv_head,
          local_query_head,
          query_head,
          raw_q_gate_head_offset,
          raw_q_gate_head_offset + kSm87TargetAotAttentionHeadDimension,
          raw_kv_offset,
          raw_kv_offset,
          processed_q_offset,
          processed_q_offset,
          task.kv_head * kSm87TargetAotAttentionHeadDimension,
          raw_kv_offset,
          raw_kv_offset,
          token,
          token + 1U,
          true,
          true};
}

static_assert(kSm87TargetAotAttentionQueryHeads ==
              kSm87TargetAotAttentionKvHeads *
                  kSm87TargetAotAttentionQueriesPerKv);
static_assert(kSm87TargetAotAttentionThreads ==
              32U * kSm87TargetAotAttentionWarps);
static_assert(kSm87TargetAotAttentionQueryRows ==
              kSm87TargetAotAttentionWarps *
                  kSm87TargetAotAttentionQueryRowsPerWarp);
static_assert(kSm87TargetAotAttentionRotaryElements ==
              2U * kSm87TargetAotAttentionRotaryPairs);
static_assert(kSm87TargetAotAttentionRawQGateTokenStride == 12'288U);
static_assert(kSm87TargetAotAttentionQueryTokenStride == 6'144U);
static_assert(kSm87TargetAotAttentionKvTokenStride == 1'024U);
static_assert(kSm87TargetAotAttentionQuerySharedBytes == 65'536U);
static_assert(kSm87TargetAotAttentionKvStageSharedBytes == 32'768U);
static_assert(kSm87TargetAotAttentionSharedBytes == 131'072U);

}  // namespace q3x::kernels
