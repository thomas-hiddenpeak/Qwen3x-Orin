#pragma once

#include "q3x/kernels/sm87_target_aot_context.h"

#include <array>
#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Host-only GDN constituent for AC-PREFILL-SM87-AOT-SYSTEM-v1. It freezes
// ownership, scheduling, and publication boundaries without exposing a
// launcher or selector. C64 is only a preparation window: each owner still
// advances four ordered C16 recurrence blocks and rounds state after every
// token. The route is accuracy-unqualified until its exact finite-precision
// identity and full real-model state are qualified.
inline constexpr std::size_t kSm87TargetAotGdnLayers = 48U;
inline constexpr std::size_t kSm87TargetAotGdnValueHeads = 48U;
inline constexpr std::size_t kSm87TargetAotGdnQkGroups = 16U;
inline constexpr std::size_t kSm87TargetAotGdnValueHeadsPerQkGroup = 3U;
inline constexpr std::size_t kSm87TargetAotGdnStateValueDimension = 128U;
inline constexpr std::size_t kSm87TargetAotGdnStateKeyDimension = 128U;
inline constexpr std::size_t kSm87TargetAotGdnStateKeyStrideElements = 1U;
inline constexpr std::size_t kSm87TargetAotGdnStateValueStrideElements =
    kSm87TargetAotGdnStateKeyDimension;
inline constexpr std::size_t kSm87TargetAotGdnStateHeadStrideElements =
    kSm87TargetAotGdnStateValueDimension *
    kSm87TargetAotGdnStateValueStrideElements;
inline constexpr std::size_t kSm87TargetAotGdnOwnerCtas = 16U;
inline constexpr std::size_t kSm87TargetAotGdnThreadsPerCta = 256U;
inline constexpr std::size_t kSm87TargetAotGdnWarpsPerCta = 8U;
inline constexpr std::size_t kSm87TargetAotGdnExactRecurrenceTokens = 16U;
inline constexpr std::size_t kSm87TargetAotGdnPreparationTokens = 64U;
inline constexpr std::size_t kSm87TargetAotGdnC16PerPreparation = 4U;
inline constexpr std::size_t kSm87TargetAotGdnPreparationSlots = 2U;
inline constexpr std::size_t kSm87TargetAotGdnBf16Bytes = 2U;
inline constexpr std::size_t kSm87TargetAotGdnPackedWordBytes = 4U;
inline constexpr std::size_t kSm87TargetAotGdnConvWidth = 4U;
inline constexpr std::size_t kSm87TargetAotGdnConvHistory = 3U;
inline constexpr std::size_t kSm87TargetAotGdnQConvChannels =
    kSm87TargetAotGdnQkGroups * kSm87TargetAotGdnStateKeyDimension;
inline constexpr std::size_t kSm87TargetAotGdnKConvChannels =
    kSm87TargetAotGdnQConvChannels;
inline constexpr std::size_t kSm87TargetAotGdnVConvChannels =
    kSm87TargetAotGdnValueHeads * kSm87TargetAotGdnStateValueDimension;
inline constexpr std::size_t kSm87TargetAotGdnQConvChannelOffset = 0U;
inline constexpr std::size_t kSm87TargetAotGdnKConvChannelOffset =
    kSm87TargetAotGdnQConvChannels;
inline constexpr std::size_t kSm87TargetAotGdnVConvChannelOffset =
    kSm87TargetAotGdnQConvChannels + kSm87TargetAotGdnKConvChannels;
inline constexpr std::size_t kSm87TargetAotGdnConvChannelsPerOwner = 640U;
inline constexpr std::size_t kSm87TargetAotGdnTotalConvChannels =
    kSm87TargetAotGdnQConvChannels + kSm87TargetAotGdnKConvChannels +
    kSm87TargetAotGdnVConvChannels;
inline constexpr std::size_t kSm87TargetAotGdnStateValuesPerHead =
    kSm87TargetAotGdnStateValueDimension *
    kSm87TargetAotGdnStateKeyDimension;
inline constexpr std::size_t kSm87TargetAotGdnStateBytesPerHead =
    kSm87TargetAotGdnStateValuesPerHead * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::size_t kSm87TargetAotGdnStateBytesPerOwner =
    kSm87TargetAotGdnStateBytesPerHead *
    kSm87TargetAotGdnValueHeadsPerQkGroup;
inline constexpr std::size_t kSm87TargetAotGdnTotalStateBytes =
    kSm87TargetAotGdnStateBytesPerHead * kSm87TargetAotGdnValueHeads;
inline constexpr std::size_t kSm87TargetAotGdnPackedStateWordsPerOwner =
    kSm87TargetAotGdnStateBytesPerOwner /
    kSm87TargetAotGdnPackedWordBytes;
inline constexpr std::size_t kSm87TargetAotGdnPackedStateWordsPerThread =
    kSm87TargetAotGdnPackedStateWordsPerOwner /
    kSm87TargetAotGdnThreadsPerCta;
inline constexpr std::size_t kSm87TargetAotGdnConvHistoryBytesPerOwner =
    kSm87TargetAotGdnConvChannelsPerOwner * kSm87TargetAotGdnConvHistory *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::size_t kSm87TargetAotGdnTotalConvHistoryBytes =
    kSm87TargetAotGdnConvHistoryBytesPerOwner *
    kSm87TargetAotGdnOwnerCtas;

inline constexpr std::size_t kSm87TargetAotGdnQBytesPerPayloadSlot =
    kSm87TargetAotGdnExactRecurrenceTokens *
    kSm87TargetAotGdnStateKeyDimension * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::size_t kSm87TargetAotGdnKBytesPerPayloadSlot =
    kSm87TargetAotGdnQBytesPerPayloadSlot;
inline constexpr std::size_t kSm87TargetAotGdnVBytesPerPayloadSlot =
    kSm87TargetAotGdnExactRecurrenceTokens *
    kSm87TargetAotGdnValueHeadsPerQkGroup *
    kSm87TargetAotGdnStateValueDimension * kSm87TargetAotGdnBf16Bytes;
inline constexpr std::size_t kSm87TargetAotGdnZBytesPerPayloadSlot =
    kSm87TargetAotGdnVBytesPerPayloadSlot;
inline constexpr std::size_t kSm87TargetAotGdnABytesPerPayloadSlot =
    kSm87TargetAotGdnExactRecurrenceTokens *
    kSm87TargetAotGdnValueHeadsPerQkGroup *
    kSm87TargetAotGdnBf16Bytes;
inline constexpr std::size_t kSm87TargetAotGdnBBytesPerPayloadSlot =
    kSm87TargetAotGdnABytesPerPayloadSlot;
inline constexpr std::size_t kSm87TargetAotGdnPayloadBytesPerSlot =
    kSm87TargetAotGdnQBytesPerPayloadSlot +
    kSm87TargetAotGdnKBytesPerPayloadSlot +
    kSm87TargetAotGdnVBytesPerPayloadSlot +
    kSm87TargetAotGdnZBytesPerPayloadSlot +
    kSm87TargetAotGdnABytesPerPayloadSlot +
    kSm87TargetAotGdnBBytesPerPayloadSlot;
inline constexpr std::size_t kSm87TargetAotGdnPrivateSharedPayloadBytes =
    kSm87TargetAotGdnPreparationSlots *
    kSm87TargetAotGdnPayloadBytesPerSlot;
inline constexpr std::size_t kSm87TargetAotGdnConvWeightElements =
    kSm87TargetAotGdnTotalConvChannels * kSm87TargetAotGdnConvWidth;
inline constexpr std::size_t kSm87TargetAotGdnConvHistoryElements =
    kSm87TargetAotGdnTotalConvChannels * kSm87TargetAotGdnConvHistory;
inline constexpr std::size_t kSm87TargetAotGdnScalarHeadElements =
    kSm87TargetAotGdnValueHeads;
inline constexpr std::size_t kSm87TargetAotGdnRecurrentStateElements =
    kSm87TargetAotGdnValueHeads * kSm87TargetAotGdnStateValuesPerHead;
inline constexpr std::size_t kSm87TargetAotGdnNormWeightElements =
    kSm87TargetAotGdnStateValueDimension;
inline constexpr std::size_t kSm87TargetAotGdnRawQkvZChannels = 16'384U;
inline constexpr std::size_t kSm87TargetAotGdnRawQChannels = 2'048U;
inline constexpr std::size_t kSm87TargetAotGdnRawKChannels = 2'048U;
inline constexpr std::size_t kSm87TargetAotGdnRawVChannels = 6'144U;
inline constexpr std::size_t kSm87TargetAotGdnRawZChannels = 6'144U;
inline constexpr std::size_t kSm87TargetAotGdnRawQOffset = 0U;
inline constexpr std::size_t kSm87TargetAotGdnRawKOffset =
    kSm87TargetAotGdnRawQOffset + kSm87TargetAotGdnRawQChannels;
inline constexpr std::size_t kSm87TargetAotGdnRawVOffset =
    kSm87TargetAotGdnRawKOffset + kSm87TargetAotGdnRawKChannels;
inline constexpr std::size_t kSm87TargetAotGdnRawZOffset =
    kSm87TargetAotGdnRawVOffset + kSm87TargetAotGdnRawVChannels;
inline constexpr std::size_t kSm87TargetAotGdnAbChannels =
    2U * kSm87TargetAotGdnValueHeads;
inline constexpr std::size_t kSm87TargetAotGdnOutputChannels =
    kSm87TargetAotGdnValueHeads * kSm87TargetAotGdnStateValueDimension;
inline constexpr std::uint32_t kSm87TargetAotGdnEpsilonFp32Bits =
    0x3586'37bdU;
inline constexpr std::size_t kSm87TargetAotGdnInputBindingCount = 6U;
inline constexpr std::size_t kSm87TargetAotGdnBindingIdentityCount =
    31U + kSm87TargetAotGdnInputBindingCount * 4U +
    kSm87TargetAotGdnOwnerCtas;

enum class Sm87TargetAotGdnTopology : std::uint8_t {
  kInvalid = 0U,
  kQkGroup16OwnersThreeValueHeads,
  // Resource-audit fallback only. It is deliberately not constructible or
  // valid in v1 because changing ownership requires a new plan identity.
  kValueHead48OwnersRequiresResourceAudit,
};

enum class Sm87TargetAotGdnRecurrence : std::uint8_t {
  kInvalid = 0U,
  kC16TokenOrderedPerTokenBf16,
};

enum class Sm87TargetAotGdnTactic : std::uint8_t {
  kInvalid = 0U,
  kLayerLongC16Persistent16CtaRegisterState,
};

enum class Sm87TargetAotGdnPublication : std::uint8_t {
  kInvalid = 0U,
  kPreRoundFp32OutputThenBf16State,
  kRawBf16ThenRmsNormSiluZ,
};

enum class Sm87TargetAotGdnConvHistoryLayout : std::uint8_t {
  kInvalid = 0U,
  kCanonicalQ2048K2048V6144History3,
};

enum class Sm87TargetAotGdnScalarType : std::uint8_t {
  kInvalid = 0U,
  kBf16,
  kFp32,
};

enum class Sm87TargetAotGdnStateAxisOrder : std::uint8_t {
  kInvalid = 0U,
  // The recurrent update is v_t k_t^T. Key is the contiguous inner axis:
  // linear(head, value, key) = head * (V * K) + value * K + key.
  kHeadValueKey,
  // Audit-only negative sentinel. Equal V/K extents make a transposition
  // size-compatible, so v1 names and rejects it explicitly.
  kHeadKeyValueTransposedForbidden,
};

struct Sm87TargetAotGdnStateLayout {
  Sm87TargetAotGdnStateAxisOrder axis_order =
      Sm87TargetAotGdnStateAxisOrder::kInvalid;
  std::size_t head_count = 0U;
  std::size_t value_dimension = 0U;
  std::size_t key_dimension = 0U;
  std::size_t head_stride_elements = 0U;
  std::size_t value_stride_elements = 0U;
  std::size_t key_stride_elements = 0U;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return axis_order == Sm87TargetAotGdnStateAxisOrder::kHeadValueKey &&
           head_count == kSm87TargetAotGdnValueHeads &&
           value_dimension == kSm87TargetAotGdnStateValueDimension &&
           key_dimension == kSm87TargetAotGdnStateKeyDimension &&
           head_stride_elements ==
               kSm87TargetAotGdnStateHeadStrideElements &&
           value_stride_elements ==
               kSm87TargetAotGdnStateValueStrideElements &&
           key_stride_elements == kSm87TargetAotGdnStateKeyStrideElements;
  }
};

inline constexpr Sm87TargetAotGdnStateLayout
    kSm87TargetAotGdnRecurrentStateLayout{
        Sm87TargetAotGdnStateAxisOrder::kHeadValueKey,
        kSm87TargetAotGdnValueHeads,
        kSm87TargetAotGdnStateValueDimension,
        kSm87TargetAotGdnStateKeyDimension,
        kSm87TargetAotGdnStateHeadStrideElements,
        kSm87TargetAotGdnStateValueStrideElements,
        kSm87TargetAotGdnStateKeyStrideElements};

[[nodiscard]] constexpr bool sm87_target_aot_same_gdn_state_layout(
    const Sm87TargetAotGdnStateLayout& left,
    const Sm87TargetAotGdnStateLayout& right) noexcept {
  return left.axis_order == right.axis_order &&
         left.head_count == right.head_count &&
         left.value_dimension == right.value_dimension &&
         left.key_dimension == right.key_dimension &&
         left.head_stride_elements == right.head_stride_elements &&
         left.value_stride_elements == right.value_stride_elements &&
         left.key_stride_elements == right.key_stride_elements;
}

enum class Sm87TargetAotGdnTensorLayout : std::uint8_t {
  kInvalid = 0U,
  kPromptMajorRawQkvZ_T16384,
  kPromptMajorAb_T2x48,
  kPromptMajorOutput_T6144,
  kConvWeight_C10240W4,
  kHeadVector_H48,
  kNormWeight_D128,
  kScalar,
  kConvHistory_C10240H3,
  kRecurrentState_H48V128K128,
};

// This names the intended incumbent operation order. It is not an accuracy
// attestation: the route remains unqualified until a CUDA implementation is
// compared bitwise against the production oracle on real model state.
enum class Sm87TargetAotGdnNumericalContract : std::uint8_t {
  kInvalid = 0U,
  // Candidate instruction graph inherited from the exact C16 CUDA path. It
  // does not borrow qualification from the deployed Chunk64/WY path.
  kExactC16CudaCandidatePerTokenBf16,
};

enum class Sm87TargetAotGdnConvNumericalContract : std::uint8_t {
  kInvalid = 0U,
  // For each Q/K/V channel: accumulate width-4 convolution in FP32 from
  // oldest history to current token, apply SiLU in FP32, then publish the
  // convolution result with BF16 RNE before any Q/K/V recurrence use.
  kFp32FmaOldestToCurrentThenSiluThenBf16Rne,
};

enum class Sm87TargetAotGdnOracleFamily : std::uint8_t {
  kInvalid = 0U,
  kExactC16CudaCandidate,
  kDeployedChunk64WyWmma,
};

enum class Sm87TargetAotGdnQkNormalizationContract : std::uint8_t {
  kInvalid = 0U,
  // Four BF16-decoded values per lane: square in FP32, form (0+64) and
  // (32+96), add those pairs, then shuffle-add at 16,8,4,2,1. Q multiplies
  // rsqrtf(sum+eps) by rsqrtf(128); K uses rsqrtf(sum+eps). Normalized values
  // remain private FP32 operands for the recurrence.
  kFp32Pair0_64Pair32_96Shuffle16To1RsqrtfQInvSqrt128,
};

enum class Sm87TargetAotGdnGateScalarContract : std::uint8_t {
  kInvalid = 0U,
  // gate=a+dt_bias in FP32; softplus is x>20 ? x : log1pf(expf(x));
  // alpha=expf(-expf(A_log)*softplus); beta uses the stable sign branch and
  // remains FP32 without an intervening BF16 publication.
  kCudaExpfLog1pfThreshold20StableSigmoidFp32,
};

enum class Sm87TargetAotGdnRecurrenceExecutionContract : std::uint8_t {
  kInvalid = 0U,
  // Scale BF16 state by alpha in FP32, prediction key 0..127 with fmaf,
  // delta=(BF16(V)-prediction)*beta, update with fmaf, publish state BF16 RNE
  // for the next token, but compute this token's output from the pre-round
  // update with a second key-ascending fmaf chain.
  kAlphaScalePredictionUpdateOutputKeyAscendingFmafPerTokenBf16,
};

enum class Sm87TargetAotGdnNormGateContract : std::uint8_t {
  kInvalid = 0U,
  // Consume raw BF16 recurrence output, square with fmaf(x,x,0), form the
  // same (0+64)/(32+96) pairs and shuffle 16..1, apply
  // rsqrtf(sum/128+eps), plain BF16 norm weight, then
  // z/(1+expf(-z)), and finally BF16 RNE.
  kRawBf16PairShuffleRmsRsqrtfPlainWeightSiluBf16Rne,
};

struct Sm87TargetAotGdnFinitePrecisionContract {
  Sm87TargetAotGdnOracleFamily oracle_family =
      Sm87TargetAotGdnOracleFamily::kInvalid;
  Sm87TargetAotGdnQkNormalizationContract qk_normalization =
      Sm87TargetAotGdnQkNormalizationContract::kInvalid;
  Sm87TargetAotGdnGateScalarContract gate_scalars =
      Sm87TargetAotGdnGateScalarContract::kInvalid;
  Sm87TargetAotGdnRecurrenceExecutionContract recurrence =
      Sm87TargetAotGdnRecurrenceExecutionContract::kInvalid;
  Sm87TargetAotGdnNormGateContract norm_gate =
      Sm87TargetAotGdnNormGateContract::kInvalid;
  std::uint32_t softplus_threshold_fp32_bits = 0U;
  std::array<std::size_t, 4U> qk_lane_dimension_offsets{};
  std::array<std::size_t, 5U> qk_shuffle_down_strides{};
  std::array<std::size_t, 2U> rms_pair_offsets{};
  bool conv_silu_uses_x_over_one_plus_expf_negative_x = false;
  bool normalized_qk_remain_fp32_private = false;
  bool beta_remains_fp32 = false;
  bool per_token_state_bf16_rne = false;
  bool same_token_output_uses_pre_round_state = false;
  bool raw_output_bf16_rne_before_norm_gate = false;
  bool no_fast_math_reassociation = false;
};

[[nodiscard]] constexpr bool sm87_target_aot_same_gdn_finite_precision(
    const Sm87TargetAotGdnFinitePrecisionContract& left,
    const Sm87TargetAotGdnFinitePrecisionContract& right) noexcept {
  if (left.oracle_family != right.oracle_family ||
      left.qk_normalization != right.qk_normalization ||
      left.gate_scalars != right.gate_scalars ||
      left.recurrence != right.recurrence ||
      left.norm_gate != right.norm_gate ||
      left.softplus_threshold_fp32_bits !=
          right.softplus_threshold_fp32_bits ||
      left.conv_silu_uses_x_over_one_plus_expf_negative_x !=
          right.conv_silu_uses_x_over_one_plus_expf_negative_x ||
      left.normalized_qk_remain_fp32_private !=
          right.normalized_qk_remain_fp32_private ||
      left.beta_remains_fp32 != right.beta_remains_fp32 ||
      left.per_token_state_bf16_rne != right.per_token_state_bf16_rne ||
      left.same_token_output_uses_pre_round_state !=
          right.same_token_output_uses_pre_round_state ||
      left.raw_output_bf16_rne_before_norm_gate !=
          right.raw_output_bf16_rne_before_norm_gate ||
      left.no_fast_math_reassociation != right.no_fast_math_reassociation) {
    return false;
  }
  for (std::size_t index = 0U;
       index < left.qk_lane_dimension_offsets.size(); ++index) {
    if (left.qk_lane_dimension_offsets[index] !=
        right.qk_lane_dimension_offsets[index]) {
      return false;
    }
  }
  for (std::size_t index = 0U;
       index < left.qk_shuffle_down_strides.size(); ++index) {
    if (left.qk_shuffle_down_strides[index] !=
        right.qk_shuffle_down_strides[index]) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < left.rms_pair_offsets.size(); ++index) {
    if (left.rms_pair_offsets[index] != right.rms_pair_offsets[index]) {
      return false;
    }
  }
  return true;
}

enum class Sm87TargetAotGdnRawPartitionRole : std::uint8_t {
  kInvalid = 0U,
  kQ,
  kK,
  kV,
  kZ,
};

struct Sm87TargetAotGdnRawPartition {
  Sm87TargetAotGdnRawPartitionRole role =
      Sm87TargetAotGdnRawPartitionRole::kInvalid;
  std::size_t channel_offset = 0U;
  std::size_t channel_count = 0U;
  bool passes_causal_conv_silu_bf16 = false;
  bool bit_exact_direct_gate_input = false;
};

inline constexpr std::array<Sm87TargetAotGdnRawPartition, 4U>
    kSm87TargetAotGdnRawPartitions{{
        {Sm87TargetAotGdnRawPartitionRole::kQ,
         kSm87TargetAotGdnRawQOffset, kSm87TargetAotGdnRawQChannels, true,
         false},
        {Sm87TargetAotGdnRawPartitionRole::kK,
         kSm87TargetAotGdnRawKOffset, kSm87TargetAotGdnRawKChannels, true,
         false},
        {Sm87TargetAotGdnRawPartitionRole::kV,
         kSm87TargetAotGdnRawVOffset, kSm87TargetAotGdnRawVChannels, true,
         false},
        {Sm87TargetAotGdnRawPartitionRole::kZ,
         kSm87TargetAotGdnRawZOffset, kSm87TargetAotGdnRawZChannels, false,
         true},
    }};

enum class Sm87TargetAotGdnInputRole : std::uint8_t {
  kInvalid = 0U,
  kConvWeight,
  kALog,
  kDtBias,
  kL2Epsilon,
  kNormWeight,
  kNormEpsilon,
};

inline constexpr std::array<Sm87TargetAotGdnInputRole,
                            kSm87TargetAotGdnInputBindingCount>
    kSm87TargetAotGdnInputRoles{{
        Sm87TargetAotGdnInputRole::kConvWeight,
        Sm87TargetAotGdnInputRole::kALog,
        Sm87TargetAotGdnInputRole::kDtBias,
        Sm87TargetAotGdnInputRole::kL2Epsilon,
        Sm87TargetAotGdnInputRole::kNormWeight,
        Sm87TargetAotGdnInputRole::kNormEpsilon,
    }};

[[nodiscard]] constexpr std::size_t sm87_target_aot_gdn_input_index(
    const Sm87TargetAotGdnInputRole role) noexcept {
  for (std::size_t index = 0U;
       index < kSm87TargetAotGdnInputRoles.size(); ++index) {
    if (kSm87TargetAotGdnInputRoles[index] == role) {
      return index;
    }
  }
  return kSm87TargetAotGdnInputBindingCount;
}

enum class Sm87TargetAotGdnCompletionContract : std::uint8_t {
  kInvalid = 0U,
  // The 16 owners never synchronize with one another. Ordinary kernel
  // completion establishes that every owner has returned. Only after that
  // event makes the layer-local output readable by its O projection and
  // records one stream-ordered ready receipt for history/state spans already
  // bound to the private request transaction at admission. No CPU callback or
  // host synchronization performs the append. This is not
  // PrefillStateCommitted and grants no Decode visibility; only the
  // system-wide final commit may do so.
  kIndependentOwnersKernelEventThenRequestTransactionAppend,
};

enum Sm87TargetAotGdnPolicy : std::uint64_t {
  kSm87TargetAotGdnExactC16Recurrence = 1ULL << 0U,
  kSm87TargetAotGdnC64PreparationOnly = 1ULL << 1U,
  kSm87TargetAotGdnTwoSlotPreparationPingPong = 1ULL << 2U,
  kSm87TargetAotGdnPerTokenBf16StateRound = 1ULL << 3U,
  kSm87TargetAotGdnPreRoundFp32Output = 1ULL << 4U,
  kSm87TargetAotGdnRawBf16BeforeRmsNormSiluZ = 1ULL << 5U,
  kSm87TargetAotGdnNoWy = 1ULL << 6U,
  kSm87TargetAotGdnNoKkt = 1ULL << 7U,
  kSm87TargetAotGdnNoSsd = 1ULL << 8U,
  kSm87TargetAotGdnNoFp32ChunkAuthoritativeState = 1ULL << 9U,
  kSm87TargetAotGdnNoFullPromptAwu = 1ULL << 10U,
  kSm87TargetAotGdnNoRequestJit = 1ULL << 11U,
  kSm87TargetAotGdnNoFallback = 1ULL << 12U,
  kSm87TargetAotGdnNoMtp = 1ULL << 13U,
  kSm87TargetAotGdnNoCuBlasLt = 1ULL << 14U,
  kSm87TargetAotGdnAccuracyUnqualified = 1ULL << 15U,
  kSm87TargetAotGdnSingleLayerLongKernel = 1ULL << 16U,
  kSm87TargetAotGdnPersistent16Cta = 1ULL << 17U,
  kSm87TargetAotGdnPackedRegisterState = 1ULL << 18U,
  kSm87TargetAotGdnSameCtaCollectivePayload = 1ULL << 19U,
  kSm87TargetAotGdnNoIndependentPreparationKernel = 1ULL << 20U,
  kSm87TargetAotGdnPrivateSharedC16Payload = 1ULL << 21U,
  kSm87TargetAotGdnConsumerBarrierReuse = 1ULL << 22U,
  kSm87TargetAotGdnCausalConvWidth4 = 1ULL << 23U,
  kSm87TargetAotGdnTypedProducerReadyEvents = 1ULL << 24U,
  kSm87TargetAotGdnC16CancelSafePoint = 1ULL << 25U,
  kSm87TargetAotGdnIndependentOwnerStateChains = 1ULL << 26U,
  kSm87TargetAotGdnResourceQualificationPending = 1ULL << 27U,
  kSm87TargetAotGdnOrdinaryKernelCompletionEvent = 1ULL << 28U,
  kSm87TargetAotGdnRequestTransactionUnpublishedSpans = 1ULL << 29U,
  kSm87TargetAotGdnPostEventRequestTransactionAppend = 1ULL << 30U,
  kSm87TargetAotGdnNoCopyPointerPublication = 1ULL << 31U,
  kSm87TargetAotGdnColdFirstPositionZeroOnly = 1ULL << 32U,
  kSm87TargetAotGdnRegisterZeroInitialState = 1ULL << 33U,
  kSm87TargetAotGdnNoInitialStateHistoryDramRead = 1ULL << 34U,
  kSm87TargetAotGdnNoCooperativeLaunch = 1ULL << 35U,
  kSm87TargetAotGdnNoCrossCtaBarrier = 1ULL << 36U,
  kSm87TargetAotGdnOutputProjectionReadsAfterKernelCompletion = 1ULL << 37U,
  kSm87TargetAotGdnCancelDiscardsWholeTransaction = 1ULL << 38U,
  kSm87TargetAotGdnTransactionSpansPreboundAtAdmission = 1ULL << 39U,
  kSm87TargetAotGdnStreamOrderedReadyReceipt = 1ULL << 40U,
  kSm87TargetAotGdnNoHostCallback = 1ULL << 41U,
  kSm87TargetAotGdnNoHostSynchronization = 1ULL << 42U,
};

inline constexpr std::uint64_t kSm87TargetAotGdnRequiredPolicy =
    kSm87TargetAotGdnExactC16Recurrence |
    kSm87TargetAotGdnC64PreparationOnly |
    kSm87TargetAotGdnTwoSlotPreparationPingPong |
    kSm87TargetAotGdnPerTokenBf16StateRound |
    kSm87TargetAotGdnPreRoundFp32Output |
    kSm87TargetAotGdnRawBf16BeforeRmsNormSiluZ |
    kSm87TargetAotGdnNoWy | kSm87TargetAotGdnNoKkt |
    kSm87TargetAotGdnNoSsd |
    kSm87TargetAotGdnNoFp32ChunkAuthoritativeState |
    kSm87TargetAotGdnNoFullPromptAwu |
    kSm87TargetAotGdnNoRequestJit | kSm87TargetAotGdnNoFallback |
    kSm87TargetAotGdnNoMtp | kSm87TargetAotGdnNoCuBlasLt |
    kSm87TargetAotGdnAccuracyUnqualified |
    kSm87TargetAotGdnSingleLayerLongKernel |
    kSm87TargetAotGdnPersistent16Cta |
    kSm87TargetAotGdnPackedRegisterState |
    kSm87TargetAotGdnSameCtaCollectivePayload |
    kSm87TargetAotGdnNoIndependentPreparationKernel |
    kSm87TargetAotGdnPrivateSharedC16Payload |
    kSm87TargetAotGdnConsumerBarrierReuse |
    kSm87TargetAotGdnCausalConvWidth4 |
    kSm87TargetAotGdnTypedProducerReadyEvents |
    kSm87TargetAotGdnC16CancelSafePoint |
    kSm87TargetAotGdnIndependentOwnerStateChains |
    kSm87TargetAotGdnResourceQualificationPending |
    kSm87TargetAotGdnOrdinaryKernelCompletionEvent |
    kSm87TargetAotGdnRequestTransactionUnpublishedSpans |
    kSm87TargetAotGdnPostEventRequestTransactionAppend |
    kSm87TargetAotGdnNoCopyPointerPublication |
    kSm87TargetAotGdnColdFirstPositionZeroOnly |
    kSm87TargetAotGdnRegisterZeroInitialState |
    kSm87TargetAotGdnNoInitialStateHistoryDramRead |
    kSm87TargetAotGdnNoCooperativeLaunch |
    kSm87TargetAotGdnNoCrossCtaBarrier |
    kSm87TargetAotGdnOutputProjectionReadsAfterKernelCompletion |
    kSm87TargetAotGdnCancelDiscardsWholeTransaction |
    kSm87TargetAotGdnTransactionSpansPreboundAtAdmission |
    kSm87TargetAotGdnStreamOrderedReadyReceipt |
    kSm87TargetAotGdnNoHostCallback |
    kSm87TargetAotGdnNoHostSynchronization;

struct Sm87TargetAotGdnPlan {
  Sm87TargetAotCapacityBucket capacity_bucket =
      Sm87TargetAotCapacityBucket::kInvalid;
  Sm87TargetAotGdnTopology topology = Sm87TargetAotGdnTopology::kInvalid;
  Sm87TargetAotGdnRecurrence recurrence =
      Sm87TargetAotGdnRecurrence::kInvalid;
  Sm87TargetAotGdnTactic tactic = Sm87TargetAotGdnTactic::kInvalid;
  Sm87TargetAotGdnPublication recurrence_publication =
      Sm87TargetAotGdnPublication::kInvalid;
  Sm87TargetAotGdnPublication output_publication =
      Sm87TargetAotGdnPublication::kInvalid;
  Sm87TargetAotGdnConvHistoryLayout conv_history_layout =
      Sm87TargetAotGdnConvHistoryLayout::kInvalid;
  Sm87TargetAotGdnNumericalContract numerical_contract =
      Sm87TargetAotGdnNumericalContract::kInvalid;
  Sm87TargetAotGdnConvNumericalContract conv_numerical_contract =
      Sm87TargetAotGdnConvNumericalContract::kInvalid;
  Sm87TargetAotGdnFinitePrecisionContract finite_precision{};
  Sm87TargetAotGdnCompletionContract completion_contract =
      Sm87TargetAotGdnCompletionContract::kInvalid;
  Sm87TargetAotGdnStateLayout recurrent_state_layout{};
  std::array<Sm87TargetAotGdnRawPartition, 4U> raw_partitions{};
  std::size_t first_position = 0U;
  std::size_t token_count = 0U;
  std::size_t owner_ctas = 0U;
  std::size_t value_head_state_chains = 0U;
  std::size_t qk_groups = 0U;
  std::size_t value_heads_per_owner = 0U;
  std::size_t state_value_dimension = 0U;
  std::size_t state_key_dimension = 0U;
  std::size_t state_bytes_per_owner = 0U;
  std::size_t total_state_bytes = 0U;
  std::size_t packed_state_words_per_owner = 0U;
  std::size_t packed_state_words_per_thread = 0U;
  std::size_t kernel_launches_per_layer = 0U;
  std::size_t persistent_ctas = 0U;
  std::size_t threads_per_cta = 0U;
  std::size_t warps_per_cta = 0U;
  std::size_t conv_width = 0U;
  std::size_t conv_history = 0U;
  std::size_t conv_channels_per_owner = 0U;
  std::size_t total_conv_channels = 0U;
  std::size_t conv_history_bytes_per_owner = 0U;
  std::size_t total_conv_history_bytes = 0U;
  std::size_t exact_c16_blocks = 0U;
  std::size_t preparation_c64_macros = 0U;
  std::size_t terminal_macro_c16_blocks = 0U;
  std::size_t terminal_macro_tokens = 0U;
  std::size_t preparation_slots = 0U;
  std::size_t q_bytes_per_payload_slot = 0U;
  std::size_t k_bytes_per_payload_slot = 0U;
  std::size_t v_bytes_per_payload_slot = 0U;
  std::size_t z_bytes_per_payload_slot = 0U;
  std::size_t a_bytes_per_payload_slot = 0U;
  std::size_t b_bytes_per_payload_slot = 0U;
  std::size_t payload_bytes_per_slot = 0U;
  std::size_t private_shared_payload_bytes = 0U;
  std::size_t producer_ready_event_count = 0U;
  std::size_t cross_cta_barriers_per_layer = 0U;
  std::size_t kernel_completion_events_per_layer = 0U;
  std::size_t post_event_request_transaction_appends_per_layer = 0U;
  std::size_t required_owner_receipts = 0U;
  std::uint64_t policy = 0U;
  bool layer_long_kernel = false;
  bool packed_register_state_prompt_resident = false;
  bool same_cta_collective_payload = false;
  bool independent_preparation_kernel = false;
  bool private_shared_c16_payload = false;
  bool payload_reuse_after_same_cta_consumer_barrier = false;
  bool causal_conv_token_order = false;
  bool preparation_ping_pong = false;
  bool c64_state_composition = false;
  bool per_token_bf16_state_rounding = false;
  bool output_uses_pre_round_fp32_update = false;
  bool raw_bf16_before_rmsnorm_silu_z = false;
  bool chunk_fp32_state_authoritative = false;
  bool full_prompt_awu_materialized = false;
  bool c16_cancel_safe_point = false;
  bool independent_owner_state_chains = false;
  bool independent_c16_cancel_observation = false;
  bool cooperative_launch_required = false;
  bool cross_cta_grid_barrier_required = false;
  bool in_kernel_commit = false;
  bool request_transaction_unpublished_spans = false;
  bool kernel_completion_event_waits_all_owners = false;
  bool request_transaction_append_after_kernel_event = false;
  bool transaction_spans_prebound_at_request_admission = false;
  bool stream_ordered_ready_receipt = false;
  bool host_callback_required = false;
  bool host_synchronization_required = false;
  bool cancel_discards_unpublished_spans = false;
  bool pointer_publication_without_copy = false;
  bool output_projection_read_requires_kernel_completion = false;
  bool cancelled_span_reclamation_without_publication = false;
  bool cold_start_only = false;
  bool continuation_supported = false;
  bool initial_conv_history_device_read = false;
  bool initial_recurrent_state_device_read = false;
  bool register_zero_initialization = false;
  bool conv_fp32_fma_oldest_history_to_current = false;
  bool conv_silu_fp32_before_publication = false;
  bool conv_output_bf16_rne_before_qkv_use = false;
  bool z_bit_exact_bypasses_conv = false;
  bool conv_history_stores_raw_current_bf16 = false;
  bool resource_qualified = false;
  bool numerical_contract_qualified = false;
  bool production_dispatch_eligible = false;

  [[nodiscard]] constexpr bool valid() const noexcept;
};

struct Sm87TargetAotGdnConvSpan {
  std::size_t channel_begin = 0U;
  std::size_t channel_count = 0U;
  std::size_t history_byte_offset = 0U;
  std::size_t history_bytes = 0U;
};

struct Sm87TargetAotGdnOwnerTask {
  std::size_t owner_cta = 0U;
  std::size_t qk_group = 0U;
  std::size_t first_value_head = 0U;
  std::size_t value_head_count = 0U;
  std::size_t state_byte_offset = 0U;
  std::size_t state_bytes = 0U;
  std::size_t packed_state_words = 0U;
  Sm87TargetAotGdnConvSpan q_conv{};
  Sm87TargetAotGdnConvSpan k_conv{};
  Sm87TargetAotGdnConvSpan v_conv{};
  std::size_t conv_channel_count = 0U;
  std::size_t conv_history_bytes = 0U;
  bool valid = false;
};

struct Sm87TargetAotGdnPreparationTask {
  // A host scheduling group only. It never represents a physical kernel or
  // a 64-token shared-memory allocation; its four C16 children fill and
  // consume the two private shared slots from the same persistent CTA.
  std::size_t macro_index = 0U;
  std::size_t token_begin = 0U;
  std::size_t token_count = 0U;
  std::size_t ordered_c16_blocks = 0U;
  std::size_t first_c16_payload_slot = 0U;
  bool independent_kernel = false;
  bool valid = false;
};

struct Sm87TargetAotGdnC16Task {
  std::size_t macro_index = 0U;
  std::size_t ordinal_in_macro = 0U;
  std::size_t global_c16_index = 0U;
  std::size_t token_begin = 0U;
  std::size_t token_count = 0U;
  std::size_t preparation_slot = 0U;
  std::size_t payload_epoch = 0U;
  std::size_t prior_same_slot_c16_index = 0U;
  std::size_t predecessor_c16_index = 0U;
  bool has_prior_same_slot = false;
  bool has_predecessor = false;
  bool same_cta_collective_producer = false;
  bool same_cta_collective_consumer = false;
  bool reuse_requires_same_cta_consumer_barrier = false;
  bool cancel_safe_point_after = false;
  bool independent_kernel = false;
  bool valid = false;
};

struct Sm87TargetAotGdnOwnerReceipt {
  std::size_t completed_preparation_macros = 0U;
  std::size_t completed_c16_blocks = 0U;
  std::size_t completed_tokens = 0U;
  std::size_t completed_cancel_safe_points = 0U;
  std::size_t completed_conv_channels = 0U;
  std::size_t staged_conv_history_values = 0U;
  std::size_t staged_gdn_state_chains = 0U;
  std::uint64_t staged_bf16_state_identity = 0U;
};

enum class Sm87TargetAotGdnInputLifetime : std::uint8_t {
  kInvalid = 0U,
  kEngineReadOnly,
};

struct Sm87TargetAotGdnInputSpec {
  Sm87TargetAotGdnInputRole role = Sm87TargetAotGdnInputRole::kInvalid;
  Sm87TargetAotGdnInputLifetime lifetime =
      Sm87TargetAotGdnInputLifetime::kInvalid;
  Sm87TargetAotGdnScalarType scalar_type =
      Sm87TargetAotGdnScalarType::kInvalid;
  Sm87TargetAotGdnTensorLayout layout =
      Sm87TargetAotGdnTensorLayout::kInvalid;
  std::size_t element_count = 0U;
  std::size_t byte_count = 0U;
  std::uint32_t required_scalar_bits = 0U;
  bool mutable_in_place = false;
};

[[nodiscard]] constexpr Sm87TargetAotGdnInputSpec
sm87_target_aot_gdn_input_spec(
    const Sm87TargetAotGdnInputRole role) noexcept {
  switch (role) {
    case Sm87TargetAotGdnInputRole::kConvWeight:
      return {role, Sm87TargetAotGdnInputLifetime::kEngineReadOnly,
              Sm87TargetAotGdnScalarType::kBf16,
              Sm87TargetAotGdnTensorLayout::kConvWeight_C10240W4,
              kSm87TargetAotGdnConvWeightElements,
              kSm87TargetAotGdnConvWeightElements *
                  kSm87TargetAotGdnBf16Bytes,
              0U,
              false};
    case Sm87TargetAotGdnInputRole::kALog:
    case Sm87TargetAotGdnInputRole::kDtBias:
      return {role, Sm87TargetAotGdnInputLifetime::kEngineReadOnly,
              Sm87TargetAotGdnScalarType::kBf16,
              Sm87TargetAotGdnTensorLayout::kHeadVector_H48,
              kSm87TargetAotGdnScalarHeadElements,
              kSm87TargetAotGdnScalarHeadElements *
                  kSm87TargetAotGdnBf16Bytes,
              0U,
              false};
    case Sm87TargetAotGdnInputRole::kL2Epsilon:
    case Sm87TargetAotGdnInputRole::kNormEpsilon:
      return {role, Sm87TargetAotGdnInputLifetime::kEngineReadOnly,
              Sm87TargetAotGdnScalarType::kFp32,
              Sm87TargetAotGdnTensorLayout::kScalar, 1U,
              sizeof(std::uint32_t), kSm87TargetAotGdnEpsilonFp32Bits,
              false};
    case Sm87TargetAotGdnInputRole::kNormWeight:
      return {role, Sm87TargetAotGdnInputLifetime::kEngineReadOnly,
              Sm87TargetAotGdnScalarType::kBf16,
              Sm87TargetAotGdnTensorLayout::kNormWeight_D128,
              kSm87TargetAotGdnNormWeightElements,
              kSm87TargetAotGdnNormWeightElements *
                  kSm87TargetAotGdnBf16Bytes,
              0U,
              false};
    case Sm87TargetAotGdnInputRole::kInvalid:
      return {};
  }
  return {};
}

struct Sm87TargetAotGdnInputBinding {
  Sm87TargetAotGdnInputRole role = Sm87TargetAotGdnInputRole::kInvalid;
  Sm87TargetAotGdnInputLifetime lifetime =
      Sm87TargetAotGdnInputLifetime::kInvalid;
  Sm87TargetAotGdnScalarType scalar_type =
      Sm87TargetAotGdnScalarType::kInvalid;
  Sm87TargetAotGdnTensorLayout layout =
      Sm87TargetAotGdnTensorLayout::kInvalid;
  std::size_t element_count = 0U;
  std::size_t byte_count = 0U;
  std::uint32_t scalar_bits = 0U;
  std::uint64_t value_identity = 0U;
  std::uint64_t physical_span_identity = 0U;
  std::uint64_t lifetime_identity = 0U;
  std::uint64_t ready_event_identity = 0U;
  bool mutable_in_place = false;

  [[nodiscard]] constexpr bool valid(
      const Sm87TargetAotGdnInputRole expected_role) const noexcept {
    const auto spec = sm87_target_aot_gdn_input_spec(expected_role);
    return spec.role != Sm87TargetAotGdnInputRole::kInvalid &&
           role == spec.role && lifetime == spec.lifetime &&
           scalar_type == spec.scalar_type && layout == spec.layout &&
           element_count == spec.element_count &&
           byte_count == spec.byte_count &&
           scalar_bits == spec.required_scalar_bits &&
           value_identity != 0U && physical_span_identity != 0U &&
           lifetime_identity != 0U && ready_event_identity != 0U &&
           mutable_in_place == spec.mutable_in_place;
  }
};

[[nodiscard]] constexpr bool sm87_target_aot_same_gdn_input_binding(
    const Sm87TargetAotGdnInputBinding& left,
    const Sm87TargetAotGdnInputBinding& right) noexcept {
  return left.role == right.role && left.lifetime == right.lifetime &&
         left.scalar_type == right.scalar_type && left.layout == right.layout &&
         left.element_count == right.element_count &&
         left.byte_count == right.byte_count &&
         left.scalar_bits == right.scalar_bits &&
         left.value_identity == right.value_identity &&
         left.physical_span_identity == right.physical_span_identity &&
         left.lifetime_identity == right.lifetime_identity &&
         left.ready_event_identity == right.ready_event_identity &&
         left.mutable_in_place == right.mutable_in_place;
}

enum class Sm87TargetAotGdnProducerRole : std::uint8_t {
  kInvalid = 0U,
  kRawQkvZ,
  kBf16Ab,
};

enum class Sm87TargetAotGdnSpanLifetime : std::uint8_t {
  kInvalid = 0U,
  kProducerReadOnlyUntilKernelCompletion,
  kLayerIntermediateUntilOutputProjection,
  kRequestTransactionUnpublishedUntilCommit,
};

struct Sm87TargetAotGdnTensorSpec {
  Sm87TargetAotGdnScalarType scalar_type =
      Sm87TargetAotGdnScalarType::kInvalid;
  Sm87TargetAotGdnTensorLayout layout =
      Sm87TargetAotGdnTensorLayout::kInvalid;
  std::size_t element_count = 0U;
  std::size_t byte_count = 0U;
};

[[nodiscard]] constexpr Sm87TargetAotGdnTensorSpec
sm87_target_aot_gdn_producer_spec(
    const Sm87TargetAotGdnProducerRole role,
    const std::size_t token_count) noexcept {
  switch (role) {
    case Sm87TargetAotGdnProducerRole::kRawQkvZ:
      return {Sm87TargetAotGdnScalarType::kBf16,
              Sm87TargetAotGdnTensorLayout::kPromptMajorRawQkvZ_T16384,
              token_count * kSm87TargetAotGdnRawQkvZChannels,
              token_count * kSm87TargetAotGdnRawQkvZChannels *
                  kSm87TargetAotGdnBf16Bytes};
    case Sm87TargetAotGdnProducerRole::kBf16Ab:
      return {Sm87TargetAotGdnScalarType::kBf16,
              Sm87TargetAotGdnTensorLayout::kPromptMajorAb_T2x48,
              token_count * kSm87TargetAotGdnAbChannels,
              token_count * kSm87TargetAotGdnAbChannels *
                  kSm87TargetAotGdnBf16Bytes};
    case Sm87TargetAotGdnProducerRole::kInvalid:
      return {};
  }
  return {};
}

struct Sm87TargetAotGdnTensorSpanBinding {
  Sm87TargetAotGdnScalarType scalar_type =
      Sm87TargetAotGdnScalarType::kInvalid;
  Sm87TargetAotGdnTensorLayout layout =
      Sm87TargetAotGdnTensorLayout::kInvalid;
  Sm87TargetAotGdnSpanLifetime lifetime =
      Sm87TargetAotGdnSpanLifetime::kInvalid;
  std::size_t element_count = 0U;
  std::size_t byte_count = 0U;
  std::uint64_t physical_span_identity = 0U;
  std::uint64_t lifetime_identity = 0U;

  [[nodiscard]] constexpr bool valid(
      const Sm87TargetAotGdnTensorSpec& spec,
      const Sm87TargetAotGdnSpanLifetime expected_lifetime) const noexcept {
    return spec.scalar_type != Sm87TargetAotGdnScalarType::kInvalid &&
           scalar_type == spec.scalar_type && layout == spec.layout &&
           lifetime == expected_lifetime &&
           element_count == spec.element_count && byte_count == spec.byte_count &&
           physical_span_identity != 0U && lifetime_identity != 0U;
  }
};

[[nodiscard]] constexpr bool sm87_target_aot_same_gdn_tensor_span_binding(
    const Sm87TargetAotGdnTensorSpanBinding& left,
    const Sm87TargetAotGdnTensorSpanBinding& right) noexcept {
  return left.scalar_type == right.scalar_type &&
         left.layout == right.layout && left.lifetime == right.lifetime &&
         left.element_count == right.element_count &&
         left.byte_count == right.byte_count &&
         left.physical_span_identity == right.physical_span_identity &&
         left.lifetime_identity == right.lifetime_identity;
}

struct Sm87TargetAotGdnProducerBinding {
  Sm87TargetAotGdnProducerRole role =
      Sm87TargetAotGdnProducerRole::kInvalid;
  std::uint64_t producer_identity = 0U;
  std::uint64_t ready_event_identity = 0U;
  Sm87TargetAotGdnTensorSpanBinding tensor{};

  [[nodiscard]] constexpr bool valid(
      const Sm87TargetAotGdnProducerRole expected_role,
      const std::size_t token_count) const noexcept {
    return role == expected_role && producer_identity != 0U &&
           ready_event_identity != 0U &&
           tensor.valid(sm87_target_aot_gdn_producer_spec(expected_role,
                                                          token_count),
                        Sm87TargetAotGdnSpanLifetime::
                            kProducerReadOnlyUntilKernelCompletion);
  }
};

[[nodiscard]] constexpr bool sm87_target_aot_same_gdn_producer_binding(
    const Sm87TargetAotGdnProducerBinding& left,
    const Sm87TargetAotGdnProducerBinding& right) noexcept {
  return left.role == right.role &&
         left.producer_identity == right.producer_identity &&
         left.ready_event_identity == right.ready_event_identity &&
         sm87_target_aot_same_gdn_tensor_span_binding(left.tensor,
                                                      right.tensor);
}

struct Sm87TargetAotGdnBinding {
  Sm87TargetAotCapacityBucket capacity_bucket =
      Sm87TargetAotCapacityBucket::kInvalid;
  Sm87TargetAotGdnTopology topology = Sm87TargetAotGdnTopology::kInvalid;
  Sm87TargetAotGdnRecurrence recurrence =
      Sm87TargetAotGdnRecurrence::kInvalid;
  Sm87TargetAotGdnTactic tactic = Sm87TargetAotGdnTactic::kInvalid;
  Sm87TargetAotGdnConvHistoryLayout conv_history_layout =
      Sm87TargetAotGdnConvHistoryLayout::kInvalid;
  Sm87TargetAotGdnNumericalContract numerical_contract =
      Sm87TargetAotGdnNumericalContract::kInvalid;
  Sm87TargetAotGdnConvNumericalContract conv_numerical_contract =
      Sm87TargetAotGdnConvNumericalContract::kInvalid;
  Sm87TargetAotGdnCompletionContract completion_contract =
      Sm87TargetAotGdnCompletionContract::kInvalid;
  Sm87TargetAotGdnStateLayout recurrent_state_layout{};
  std::size_t gdn_layer_ordinal = kSm87TargetAotGdnLayers;
  std::size_t first_position = 1U;
  Sm87TargetAotGdnProducerBinding raw_qkvz_producer{};
  Sm87TargetAotGdnProducerBinding ab_producer{};
  std::array<Sm87TargetAotGdnInputBinding,
             kSm87TargetAotGdnInputBindingCount>
      inputs{};
  std::uint64_t plan_identity = 0U;
  std::uint64_t tactic_identity = 0U;
  std::uint64_t recurrence_identity = 0U;
  std::uint64_t numerical_contract_identity = 0U;
  std::uint64_t conv_numerical_contract_identity = 0U;
  std::uint64_t raw_publication_identity = 0U;
  std::uint64_t norm_gate_publication_identity = 0U;
  std::uint64_t final_conv_state_publication_identity = 0U;
  std::uint64_t c16_cancel_safe_point_identity = 0U;
  std::uint64_t kernel_completion_event_identity = 0U;
  std::uint64_t post_event_request_transaction_append_identity = 0U;
  std::uint64_t launcher_identity = 0U;
  std::uint64_t completion_contract_identity = 0U;
  std::uint64_t request_transaction_identity = 0U;
  std::uint64_t canonical_cold_zero_identity = 0U;
  std::uint64_t cold_state_reset_epoch_identity = 0U;
  std::uint64_t recurrent_state_layout_identity = 0U;
  Sm87TargetAotGdnTensorSpanBinding output{};
  Sm87TargetAotGdnTensorSpanBinding final_conv_history{};
  Sm87TargetAotGdnTensorSpanBinding final_recurrent_state{};
  std::array<std::uint64_t, kSm87TargetAotGdnOwnerCtas>
      final_bf16_state_identities{};

  [[nodiscard]] constexpr bool identity_namespace_valid() const noexcept {
    std::array<std::uint64_t, kSm87TargetAotGdnBindingIdentityCount>
        identities{};
    std::size_t index = 0U;
    identities[index++] = raw_qkvz_producer.producer_identity;
    identities[index++] = raw_qkvz_producer.ready_event_identity;
    identities[index++] = raw_qkvz_producer.tensor.physical_span_identity;
    identities[index++] = raw_qkvz_producer.tensor.lifetime_identity;
    identities[index++] = ab_producer.producer_identity;
    identities[index++] = ab_producer.ready_event_identity;
    identities[index++] = ab_producer.tensor.physical_span_identity;
    identities[index++] = ab_producer.tensor.lifetime_identity;
    identities[index++] = plan_identity;
    identities[index++] = tactic_identity;
    identities[index++] = recurrence_identity;
    identities[index++] = numerical_contract_identity;
    identities[index++] = conv_numerical_contract_identity;
    identities[index++] = raw_publication_identity;
    identities[index++] = norm_gate_publication_identity;
    identities[index++] = final_conv_state_publication_identity;
    identities[index++] = c16_cancel_safe_point_identity;
    identities[index++] = kernel_completion_event_identity;
    identities[index++] = post_event_request_transaction_append_identity;
    identities[index++] = launcher_identity;
    for (const auto& input : inputs) {
      identities[index++] = input.value_identity;
      identities[index++] = input.physical_span_identity;
      identities[index++] = input.lifetime_identity;
      identities[index++] = input.ready_event_identity;
    }
    identities[index++] = completion_contract_identity;
    identities[index++] = request_transaction_identity;
    identities[index++] = canonical_cold_zero_identity;
    identities[index++] = cold_state_reset_epoch_identity;
    identities[index++] = recurrent_state_layout_identity;
    identities[index++] = output.physical_span_identity;
    identities[index++] = output.lifetime_identity;
    identities[index++] = final_conv_history.physical_span_identity;
    identities[index++] = final_conv_history.lifetime_identity;
    identities[index++] = final_recurrent_state.physical_span_identity;
    identities[index++] = final_recurrent_state.lifetime_identity;
    for (const std::uint64_t identity : final_bf16_state_identities) {
      identities[index++] = identity;
    }
    if (index != identities.size()) {
      return false;
    }
    for (std::size_t identity = 0U; identity < identities.size();
         ++identity) {
      if (identities[identity] == 0U) {
        return false;
      }
      for (std::size_t other = identity + 1U;
           other < identities.size(); ++other) {
        if (identities[identity] == identities[other]) {
          return false;
        }
      }
    }
    return true;
  }

  [[nodiscard]] constexpr bool valid(
      const Sm87TargetAotGdnPlan& plan) const noexcept {
    if (!plan.valid() || capacity_bucket != plan.capacity_bucket ||
        topology != plan.topology || recurrence != plan.recurrence ||
        tactic != plan.tactic ||
        conv_history_layout != plan.conv_history_layout ||
        numerical_contract != plan.numerical_contract ||
        conv_numerical_contract != plan.conv_numerical_contract ||
        completion_contract != plan.completion_contract ||
        !sm87_target_aot_same_gdn_state_layout(recurrent_state_layout,
                                               plan.recurrent_state_layout) ||
        !recurrent_state_layout.valid() ||
        gdn_layer_ordinal >= kSm87TargetAotGdnLayers ||
        first_position != plan.first_position ||
        !raw_qkvz_producer.valid(
            Sm87TargetAotGdnProducerRole::kRawQkvZ, plan.token_count) ||
        !ab_producer.valid(Sm87TargetAotGdnProducerRole::kBf16Ab,
                           plan.token_count) ||
        !output.valid(
            {Sm87TargetAotGdnScalarType::kBf16,
             Sm87TargetAotGdnTensorLayout::kPromptMajorOutput_T6144,
             plan.token_count * kSm87TargetAotGdnOutputChannels,
             plan.token_count * kSm87TargetAotGdnOutputChannels *
                 kSm87TargetAotGdnBf16Bytes},
            Sm87TargetAotGdnSpanLifetime::
                kLayerIntermediateUntilOutputProjection) ||
        !final_conv_history.valid(
            {Sm87TargetAotGdnScalarType::kBf16,
             Sm87TargetAotGdnTensorLayout::kConvHistory_C10240H3,
             kSm87TargetAotGdnConvHistoryElements,
             kSm87TargetAotGdnTotalConvHistoryBytes},
            Sm87TargetAotGdnSpanLifetime::
                kRequestTransactionUnpublishedUntilCommit) ||
        !final_recurrent_state.valid(
            {Sm87TargetAotGdnScalarType::kBf16,
             Sm87TargetAotGdnTensorLayout::kRecurrentState_H48V128K128,
             kSm87TargetAotGdnRecurrentStateElements,
             kSm87TargetAotGdnTotalStateBytes},
            Sm87TargetAotGdnSpanLifetime::
                kRequestTransactionUnpublishedUntilCommit) ||
        !identity_namespace_valid()) {
      return false;
    }
    for (std::size_t input = 0U; input < inputs.size(); ++input) {
      if (!inputs[input].valid(kSm87TargetAotGdnInputRoles[input])) {
        return false;
      }
    }
    return true;
  }
};

struct Sm87TargetAotGdnReceipt {
  Sm87TargetAotCapacityBucket capacity_bucket =
      Sm87TargetAotCapacityBucket::kInvalid;
  Sm87TargetAotGdnTopology topology = Sm87TargetAotGdnTopology::kInvalid;
  Sm87TargetAotGdnRecurrence recurrence =
      Sm87TargetAotGdnRecurrence::kInvalid;
  Sm87TargetAotGdnTactic tactic = Sm87TargetAotGdnTactic::kInvalid;
  Sm87TargetAotGdnConvHistoryLayout conv_history_layout =
      Sm87TargetAotGdnConvHistoryLayout::kInvalid;
  Sm87TargetAotGdnNumericalContract numerical_contract =
      Sm87TargetAotGdnNumericalContract::kInvalid;
  Sm87TargetAotGdnConvNumericalContract conv_numerical_contract =
      Sm87TargetAotGdnConvNumericalContract::kInvalid;
  Sm87TargetAotGdnCompletionContract completion_contract =
      Sm87TargetAotGdnCompletionContract::kInvalid;
  Sm87TargetAotGdnStateLayout recurrent_state_layout{};
  std::size_t gdn_layer_ordinal = kSm87TargetAotGdnLayers;
  std::size_t first_position = 1U;
  std::size_t token_count = 0U;
  Sm87TargetAotGdnProducerBinding raw_qkvz_producer{};
  Sm87TargetAotGdnProducerBinding ab_producer{};
  std::array<Sm87TargetAotGdnInputBinding,
             kSm87TargetAotGdnInputBindingCount>
      inputs{};
  std::uint64_t plan_identity = 0U;
  std::uint64_t tactic_identity = 0U;
  std::uint64_t recurrence_identity = 0U;
  std::uint64_t numerical_contract_identity = 0U;
  std::uint64_t conv_numerical_contract_identity = 0U;
  std::uint64_t launcher_identity = 0U;
  std::array<Sm87TargetAotGdnOwnerReceipt, kSm87TargetAotGdnOwnerCtas>
      owners{};
  std::size_t completed_output_rows = 0U;
  std::size_t completed_value_heads = 0U;
  std::size_t completed_conv_channels = 0U;
  std::size_t staged_conv_history_values = 0U;
  std::size_t staged_final_gdn_state_chains = 0U;
  std::size_t completed_owner_count = 0U;
  std::size_t request_transaction_append_count = 0U;
  std::size_t appended_owner_state_count = 0U;
  std::uint64_t raw_bf16_output_identity = 0U;
  std::uint64_t rmsnorm_silu_z_output_identity = 0U;
  std::uint64_t final_conv_state_publication_identity = 0U;
  std::uint64_t c16_cancel_safe_point_identity = 0U;
  std::uint64_t kernel_completion_event_identity = 0U;
  std::uint64_t post_event_request_transaction_append_identity = 0U;
  std::uint64_t completion_contract_identity = 0U;
  std::uint64_t request_transaction_identity = 0U;
  std::uint64_t canonical_cold_zero_identity = 0U;
  std::uint64_t cold_state_reset_epoch_identity = 0U;
  std::uint64_t recurrent_state_layout_identity = 0U;
  Sm87TargetAotGdnTensorSpanBinding output{};
  Sm87TargetAotGdnTensorSpanBinding final_conv_history{};
  Sm87TargetAotGdnTensorSpanBinding final_recurrent_state{};
  std::size_t fallback_hits = 0U;
  std::size_t forbidden_transform_hits = 0U;
  bool cold_zero_register_initialization_applied = false;
  bool initial_state_history_device_read_observed = false;
  bool numerical_contract_applied = false;
  bool raw_qkvz_partition_contract_applied = false;
  bool conv_silu_fp32_applied = false;
  bool conv_output_bf16_rne_before_qkv_use = false;
  bool z_bit_exact_conv_bypass_applied = false;
  bool per_token_bf16_state_rounding = false;
  bool output_used_pre_round_fp32_update = false;
  bool raw_bf16_published = false;
  bool rmsnorm_complete = false;
  bool silu_z_gate_complete = false;
  bool independent_owner_completion = false;
  bool cooperative_launch_observed = false;
  bool cross_cta_barrier_observed = false;
  bool in_kernel_commit_observed = false;
  bool staged_spans_unpublished_at_kernel_completion = false;
  bool kernel_completion_event_recorded = false;
  bool kernel_event_after_all_owner_receipts = false;
  bool request_transaction_append_after_kernel_event = false;
  bool transaction_spans_were_prebound_at_request_admission = false;
  bool stream_ordered_ready_receipt_recorded = false;
  bool host_callback_observed = false;
  bool host_synchronization_observed = false;
  bool transaction_spans_appended_unpublished = false;
  bool pointer_publication_without_copy = false;
  bool output_projection_visibility_after_kernel_event = false;
  bool partial_span_publication_observed = false;
  bool output_reclaimable_after_output_projection = false;
  bool final_conv_history_appended_unpublished = false;
  bool final_recurrent_state_appended_unpublished = false;
  bool cancellation_observed = false;

  [[nodiscard]] constexpr bool complete(
      const Sm87TargetAotGdnPlan& plan,
      const Sm87TargetAotGdnBinding& binding) const noexcept {
    if (!binding.valid(plan) ||
        capacity_bucket != binding.capacity_bucket ||
        topology != binding.topology || recurrence != binding.recurrence ||
        tactic != binding.tactic ||
        conv_history_layout != binding.conv_history_layout ||
        numerical_contract != binding.numerical_contract ||
        conv_numerical_contract != binding.conv_numerical_contract ||
        completion_contract != binding.completion_contract ||
        !sm87_target_aot_same_gdn_state_layout(recurrent_state_layout,
                                               binding.recurrent_state_layout) ||
        !recurrent_state_layout.valid() ||
        gdn_layer_ordinal != binding.gdn_layer_ordinal ||
        first_position != binding.first_position ||
        token_count != plan.token_count ||
        !sm87_target_aot_same_gdn_producer_binding(
            raw_qkvz_producer, binding.raw_qkvz_producer) ||
        !sm87_target_aot_same_gdn_producer_binding(
            ab_producer, binding.ab_producer) ||
        plan_identity != binding.plan_identity ||
        tactic_identity != binding.tactic_identity ||
        recurrence_identity != binding.recurrence_identity ||
        numerical_contract_identity !=
            binding.numerical_contract_identity ||
        conv_numerical_contract_identity !=
            binding.conv_numerical_contract_identity ||
        launcher_identity != binding.launcher_identity ||
        completed_output_rows != plan.token_count ||
        completed_value_heads != plan.value_head_state_chains ||
        completed_conv_channels != plan.total_conv_channels ||
        staged_conv_history_values !=
            plan.total_conv_channels * plan.conv_history ||
        staged_final_gdn_state_chains !=
            plan.value_head_state_chains ||
        completed_owner_count != plan.required_owner_receipts ||
        request_transaction_append_count != 1U ||
        appended_owner_state_count != plan.required_owner_receipts ||
        raw_bf16_output_identity != binding.raw_publication_identity ||
        rmsnorm_silu_z_output_identity !=
            binding.norm_gate_publication_identity ||
        final_conv_state_publication_identity !=
            binding.final_conv_state_publication_identity ||
        c16_cancel_safe_point_identity !=
            binding.c16_cancel_safe_point_identity ||
        kernel_completion_event_identity !=
            binding.kernel_completion_event_identity ||
        post_event_request_transaction_append_identity !=
            binding.post_event_request_transaction_append_identity ||
        completion_contract_identity !=
            binding.completion_contract_identity ||
        request_transaction_identity !=
            binding.request_transaction_identity ||
        canonical_cold_zero_identity !=
            binding.canonical_cold_zero_identity ||
        cold_state_reset_epoch_identity !=
            binding.cold_state_reset_epoch_identity ||
        recurrent_state_layout_identity !=
            binding.recurrent_state_layout_identity ||
        !sm87_target_aot_same_gdn_tensor_span_binding(output,
                                                      binding.output) ||
        !sm87_target_aot_same_gdn_tensor_span_binding(
            final_conv_history, binding.final_conv_history) ||
        !sm87_target_aot_same_gdn_tensor_span_binding(
            final_recurrent_state, binding.final_recurrent_state) ||
        fallback_hits != 0U ||
        forbidden_transform_hits != 0U ||
        !cold_zero_register_initialization_applied ||
        initial_state_history_device_read_observed ||
        !numerical_contract_applied ||
        !raw_qkvz_partition_contract_applied ||
        !conv_silu_fp32_applied ||
        !conv_output_bf16_rne_before_qkv_use ||
        !z_bit_exact_conv_bypass_applied ||
        !per_token_bf16_state_rounding ||
        !output_used_pre_round_fp32_update || !raw_bf16_published ||
        !rmsnorm_complete || !silu_z_gate_complete ||
        !independent_owner_completion || cooperative_launch_observed ||
        cross_cta_barrier_observed || in_kernel_commit_observed ||
        !staged_spans_unpublished_at_kernel_completion ||
        !kernel_completion_event_recorded ||
        !kernel_event_after_all_owner_receipts ||
        !request_transaction_append_after_kernel_event ||
        !transaction_spans_were_prebound_at_request_admission ||
        !stream_ordered_ready_receipt_recorded || host_callback_observed ||
        host_synchronization_observed ||
        !transaction_spans_appended_unpublished ||
        !pointer_publication_without_copy ||
        !output_projection_visibility_after_kernel_event ||
        partial_span_publication_observed ||
        !output_reclaimable_after_output_projection ||
        !final_conv_history_appended_unpublished ||
        !final_recurrent_state_appended_unpublished ||
        cancellation_observed) {
      return false;
    }
    for (std::size_t input = 0U; input < inputs.size(); ++input) {
      if (!sm87_target_aot_same_gdn_input_binding(inputs[input],
                                                   binding.inputs[input])) {
        return false;
      }
    }
    for (std::size_t owner_index = 0U; owner_index < owners.size();
         ++owner_index) {
      const auto& owner = owners[owner_index];
      if (owner.completed_preparation_macros !=
              plan.preparation_c64_macros ||
          owner.completed_c16_blocks != plan.exact_c16_blocks ||
          owner.completed_tokens != plan.token_count ||
          owner.completed_cancel_safe_points != plan.exact_c16_blocks ||
          owner.completed_conv_channels != plan.conv_channels_per_owner ||
          owner.staged_conv_history_values !=
              plan.conv_channels_per_owner * plan.conv_history ||
          owner.staged_gdn_state_chains !=
              plan.value_heads_per_owner ||
          owner.staged_bf16_state_identity !=
              binding.final_bf16_state_identities[owner_index]) {
        return false;
      }
    }
    return true;
  }
};

[[nodiscard]] constexpr Sm87TargetAotGdnPlan sm87_target_aot_gdn_plan(
    const std::size_t token_count) noexcept {
  const auto capacity = sm87_target_aot_capacity_for_witness(token_count);
  if (!capacity.valid() ||
      token_count % kSm87TargetAotGdnExactRecurrenceTokens != 0U) {
    return {};
  }

  Sm87TargetAotGdnPlan plan;
  plan.capacity_bucket = capacity.bucket;
  plan.topology =
      Sm87TargetAotGdnTopology::kQkGroup16OwnersThreeValueHeads;
  plan.recurrence =
      Sm87TargetAotGdnRecurrence::kC16TokenOrderedPerTokenBf16;
  plan.tactic = Sm87TargetAotGdnTactic::
      kLayerLongC16Persistent16CtaRegisterState;
  plan.recurrence_publication =
      Sm87TargetAotGdnPublication::kPreRoundFp32OutputThenBf16State;
  plan.output_publication =
      Sm87TargetAotGdnPublication::kRawBf16ThenRmsNormSiluZ;
  plan.conv_history_layout = Sm87TargetAotGdnConvHistoryLayout::
      kCanonicalQ2048K2048V6144History3;
  plan.numerical_contract = Sm87TargetAotGdnNumericalContract::
      kExactC16CudaCandidatePerTokenBf16;
  plan.conv_numerical_contract = Sm87TargetAotGdnConvNumericalContract::
      kFp32FmaOldestToCurrentThenSiluThenBf16Rne;
  plan.finite_precision.oracle_family =
      Sm87TargetAotGdnOracleFamily::kExactC16CudaCandidate;
  plan.finite_precision.qk_normalization =
      Sm87TargetAotGdnQkNormalizationContract::
          kFp32Pair0_64Pair32_96Shuffle16To1RsqrtfQInvSqrt128;
  plan.finite_precision.gate_scalars = Sm87TargetAotGdnGateScalarContract::
      kCudaExpfLog1pfThreshold20StableSigmoidFp32;
  plan.finite_precision.recurrence =
      Sm87TargetAotGdnRecurrenceExecutionContract::
          kAlphaScalePredictionUpdateOutputKeyAscendingFmafPerTokenBf16;
  plan.finite_precision.norm_gate = Sm87TargetAotGdnNormGateContract::
      kRawBf16PairShuffleRmsRsqrtfPlainWeightSiluBf16Rne;
  plan.finite_precision.softplus_threshold_fp32_bits = 0x41a0'0000U;
  plan.finite_precision.qk_lane_dimension_offsets = {{0U, 32U, 64U, 96U}};
  plan.finite_precision.qk_shuffle_down_strides = {{16U, 8U, 4U, 2U, 1U}};
  plan.finite_precision.rms_pair_offsets = {{64U, 32U}};
  plan.finite_precision.conv_silu_uses_x_over_one_plus_expf_negative_x =
      true;
  plan.finite_precision.normalized_qk_remain_fp32_private = true;
  plan.finite_precision.beta_remains_fp32 = true;
  plan.finite_precision.per_token_state_bf16_rne = true;
  plan.finite_precision.same_token_output_uses_pre_round_state = true;
  plan.finite_precision.raw_output_bf16_rne_before_norm_gate = true;
  plan.finite_precision.no_fast_math_reassociation = true;
  plan.completion_contract = Sm87TargetAotGdnCompletionContract::
      kIndependentOwnersKernelEventThenRequestTransactionAppend;
  plan.recurrent_state_layout = kSm87TargetAotGdnRecurrentStateLayout;
  plan.raw_partitions = kSm87TargetAotGdnRawPartitions;
  plan.first_position = 0U;
  plan.token_count = token_count;
  plan.owner_ctas = kSm87TargetAotGdnOwnerCtas;
  plan.value_head_state_chains = kSm87TargetAotGdnValueHeads;
  plan.qk_groups = kSm87TargetAotGdnQkGroups;
  plan.value_heads_per_owner = kSm87TargetAotGdnValueHeadsPerQkGroup;
  plan.state_value_dimension = kSm87TargetAotGdnStateValueDimension;
  plan.state_key_dimension = kSm87TargetAotGdnStateKeyDimension;
  plan.state_bytes_per_owner = kSm87TargetAotGdnStateBytesPerOwner;
  plan.total_state_bytes = kSm87TargetAotGdnTotalStateBytes;
  plan.packed_state_words_per_owner =
      kSm87TargetAotGdnPackedStateWordsPerOwner;
  plan.packed_state_words_per_thread =
      kSm87TargetAotGdnPackedStateWordsPerThread;
  plan.kernel_launches_per_layer = 1U;
  plan.persistent_ctas = kSm87TargetAotGdnOwnerCtas;
  plan.threads_per_cta = kSm87TargetAotGdnThreadsPerCta;
  plan.warps_per_cta = kSm87TargetAotGdnWarpsPerCta;
  plan.conv_width = kSm87TargetAotGdnConvWidth;
  plan.conv_history = kSm87TargetAotGdnConvHistory;
  plan.conv_channels_per_owner = kSm87TargetAotGdnConvChannelsPerOwner;
  plan.total_conv_channels = kSm87TargetAotGdnTotalConvChannels;
  plan.conv_history_bytes_per_owner =
      kSm87TargetAotGdnConvHistoryBytesPerOwner;
  plan.total_conv_history_bytes = kSm87TargetAotGdnTotalConvHistoryBytes;
  plan.exact_c16_blocks =
      token_count / kSm87TargetAotGdnExactRecurrenceTokens;
  plan.preparation_c64_macros =
      (plan.exact_c16_blocks + kSm87TargetAotGdnC16PerPreparation - 1U) /
      kSm87TargetAotGdnC16PerPreparation;
  plan.terminal_macro_c16_blocks =
      plan.exact_c16_blocks % kSm87TargetAotGdnC16PerPreparation;
  if (plan.terminal_macro_c16_blocks == 0U) {
    plan.terminal_macro_c16_blocks = kSm87TargetAotGdnC16PerPreparation;
  }
  plan.terminal_macro_tokens =
      plan.terminal_macro_c16_blocks *
      kSm87TargetAotGdnExactRecurrenceTokens;
  plan.preparation_slots = kSm87TargetAotGdnPreparationSlots;
  plan.q_bytes_per_payload_slot = kSm87TargetAotGdnQBytesPerPayloadSlot;
  plan.k_bytes_per_payload_slot = kSm87TargetAotGdnKBytesPerPayloadSlot;
  plan.v_bytes_per_payload_slot = kSm87TargetAotGdnVBytesPerPayloadSlot;
  plan.z_bytes_per_payload_slot = kSm87TargetAotGdnZBytesPerPayloadSlot;
  plan.a_bytes_per_payload_slot = kSm87TargetAotGdnABytesPerPayloadSlot;
  plan.b_bytes_per_payload_slot = kSm87TargetAotGdnBBytesPerPayloadSlot;
  plan.payload_bytes_per_slot = kSm87TargetAotGdnPayloadBytesPerSlot;
  plan.private_shared_payload_bytes =
      kSm87TargetAotGdnPrivateSharedPayloadBytes;
  plan.producer_ready_event_count = 2U;
  plan.cross_cta_barriers_per_layer = 0U;
  plan.kernel_completion_events_per_layer = 1U;
  plan.post_event_request_transaction_appends_per_layer = 1U;
  plan.required_owner_receipts = kSm87TargetAotGdnOwnerCtas;
  plan.policy = kSm87TargetAotGdnRequiredPolicy;
  plan.layer_long_kernel = true;
  plan.packed_register_state_prompt_resident = true;
  plan.same_cta_collective_payload = true;
  plan.independent_preparation_kernel = false;
  plan.private_shared_c16_payload = true;
  plan.payload_reuse_after_same_cta_consumer_barrier = true;
  plan.causal_conv_token_order = true;
  plan.preparation_ping_pong = true;
  plan.c64_state_composition = false;
  plan.per_token_bf16_state_rounding = true;
  plan.output_uses_pre_round_fp32_update = true;
  plan.raw_bf16_before_rmsnorm_silu_z = true;
  plan.chunk_fp32_state_authoritative = false;
  plan.full_prompt_awu_materialized = false;
  plan.c16_cancel_safe_point = true;
  plan.independent_owner_state_chains = true;
  plan.independent_c16_cancel_observation = true;
  plan.cooperative_launch_required = false;
  plan.cross_cta_grid_barrier_required = false;
  plan.in_kernel_commit = false;
  plan.request_transaction_unpublished_spans = true;
  plan.kernel_completion_event_waits_all_owners = true;
  plan.request_transaction_append_after_kernel_event = true;
  plan.transaction_spans_prebound_at_request_admission = true;
  plan.stream_ordered_ready_receipt = true;
  plan.host_callback_required = false;
  plan.host_synchronization_required = false;
  plan.cancel_discards_unpublished_spans = true;
  plan.pointer_publication_without_copy = true;
  plan.output_projection_read_requires_kernel_completion = true;
  plan.cancelled_span_reclamation_without_publication = true;
  plan.cold_start_only = true;
  plan.continuation_supported = false;
  plan.initial_conv_history_device_read = false;
  plan.initial_recurrent_state_device_read = false;
  plan.register_zero_initialization = true;
  plan.conv_fp32_fma_oldest_history_to_current = true;
  plan.conv_silu_fp32_before_publication = true;
  plan.conv_output_bf16_rne_before_qkv_use = true;
  plan.z_bit_exact_bypasses_conv = true;
  plan.conv_history_stores_raw_current_bf16 = true;
  plan.resource_qualified = false;
  plan.numerical_contract_qualified = false;
  plan.production_dispatch_eligible = false;
  return plan;
}

[[nodiscard]] constexpr bool sm87_target_aot_same_gdn_raw_partitions(
    const std::array<Sm87TargetAotGdnRawPartition, 4U>& left,
    const std::array<Sm87TargetAotGdnRawPartition, 4U>& right) noexcept {
  for (std::size_t index = 0U; index < left.size(); ++index) {
    if (left[index].role != right[index].role ||
        left[index].channel_offset != right[index].channel_offset ||
        left[index].channel_count != right[index].channel_count ||
        left[index].passes_causal_conv_silu_bf16 !=
            right[index].passes_causal_conv_silu_bf16 ||
        left[index].bit_exact_direct_gate_input !=
            right[index].bit_exact_direct_gate_input) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] constexpr bool sm87_target_aot_same_gdn_plan(
    const Sm87TargetAotGdnPlan& left,
    const Sm87TargetAotGdnPlan& right) noexcept {
  return left.capacity_bucket == right.capacity_bucket &&
         left.topology == right.topology &&
         left.recurrence == right.recurrence &&
         left.tactic == right.tactic &&
         left.recurrence_publication == right.recurrence_publication &&
         left.output_publication == right.output_publication &&
         left.conv_history_layout == right.conv_history_layout &&
         left.numerical_contract == right.numerical_contract &&
         left.conv_numerical_contract == right.conv_numerical_contract &&
         sm87_target_aot_same_gdn_finite_precision(
             left.finite_precision, right.finite_precision) &&
         left.completion_contract == right.completion_contract &&
         sm87_target_aot_same_gdn_state_layout(
             left.recurrent_state_layout, right.recurrent_state_layout) &&
         sm87_target_aot_same_gdn_raw_partitions(left.raw_partitions,
                                                 right.raw_partitions) &&
         left.first_position == right.first_position &&
         left.token_count == right.token_count &&
         left.owner_ctas == right.owner_ctas &&
         left.value_head_state_chains == right.value_head_state_chains &&
         left.qk_groups == right.qk_groups &&
         left.value_heads_per_owner == right.value_heads_per_owner &&
         left.state_value_dimension == right.state_value_dimension &&
         left.state_key_dimension == right.state_key_dimension &&
         left.state_bytes_per_owner == right.state_bytes_per_owner &&
         left.total_state_bytes == right.total_state_bytes &&
         left.packed_state_words_per_owner ==
             right.packed_state_words_per_owner &&
         left.packed_state_words_per_thread ==
             right.packed_state_words_per_thread &&
         left.kernel_launches_per_layer == right.kernel_launches_per_layer &&
         left.persistent_ctas == right.persistent_ctas &&
         left.threads_per_cta == right.threads_per_cta &&
         left.warps_per_cta == right.warps_per_cta &&
         left.conv_width == right.conv_width &&
         left.conv_history == right.conv_history &&
         left.conv_channels_per_owner == right.conv_channels_per_owner &&
         left.total_conv_channels == right.total_conv_channels &&
         left.conv_history_bytes_per_owner ==
             right.conv_history_bytes_per_owner &&
         left.total_conv_history_bytes == right.total_conv_history_bytes &&
         left.exact_c16_blocks == right.exact_c16_blocks &&
         left.preparation_c64_macros == right.preparation_c64_macros &&
         left.terminal_macro_c16_blocks ==
             right.terminal_macro_c16_blocks &&
         left.terminal_macro_tokens == right.terminal_macro_tokens &&
         left.preparation_slots == right.preparation_slots &&
         left.q_bytes_per_payload_slot == right.q_bytes_per_payload_slot &&
         left.k_bytes_per_payload_slot == right.k_bytes_per_payload_slot &&
         left.v_bytes_per_payload_slot == right.v_bytes_per_payload_slot &&
         left.z_bytes_per_payload_slot == right.z_bytes_per_payload_slot &&
         left.a_bytes_per_payload_slot == right.a_bytes_per_payload_slot &&
         left.b_bytes_per_payload_slot == right.b_bytes_per_payload_slot &&
         left.payload_bytes_per_slot == right.payload_bytes_per_slot &&
         left.private_shared_payload_bytes ==
             right.private_shared_payload_bytes &&
         left.producer_ready_event_count ==
             right.producer_ready_event_count &&
         left.cross_cta_barriers_per_layer ==
             right.cross_cta_barriers_per_layer &&
         left.kernel_completion_events_per_layer ==
             right.kernel_completion_events_per_layer &&
         left.post_event_request_transaction_appends_per_layer ==
             right.post_event_request_transaction_appends_per_layer &&
         left.required_owner_receipts == right.required_owner_receipts &&
         left.policy == right.policy &&
         left.layer_long_kernel == right.layer_long_kernel &&
         left.packed_register_state_prompt_resident ==
             right.packed_register_state_prompt_resident &&
         left.same_cta_collective_payload ==
             right.same_cta_collective_payload &&
         left.independent_preparation_kernel ==
             right.independent_preparation_kernel &&
         left.private_shared_c16_payload ==
             right.private_shared_c16_payload &&
         left.payload_reuse_after_same_cta_consumer_barrier ==
             right.payload_reuse_after_same_cta_consumer_barrier &&
         left.causal_conv_token_order == right.causal_conv_token_order &&
         left.preparation_ping_pong == right.preparation_ping_pong &&
         left.c64_state_composition == right.c64_state_composition &&
         left.per_token_bf16_state_rounding ==
             right.per_token_bf16_state_rounding &&
         left.output_uses_pre_round_fp32_update ==
             right.output_uses_pre_round_fp32_update &&
         left.raw_bf16_before_rmsnorm_silu_z ==
             right.raw_bf16_before_rmsnorm_silu_z &&
         left.chunk_fp32_state_authoritative ==
             right.chunk_fp32_state_authoritative &&
         left.full_prompt_awu_materialized ==
             right.full_prompt_awu_materialized &&
         left.c16_cancel_safe_point == right.c16_cancel_safe_point &&
         left.independent_owner_state_chains ==
             right.independent_owner_state_chains &&
         left.independent_c16_cancel_observation ==
             right.independent_c16_cancel_observation &&
         left.cooperative_launch_required ==
             right.cooperative_launch_required &&
         left.cross_cta_grid_barrier_required ==
             right.cross_cta_grid_barrier_required &&
         left.in_kernel_commit == right.in_kernel_commit &&
         left.request_transaction_unpublished_spans ==
             right.request_transaction_unpublished_spans &&
         left.kernel_completion_event_waits_all_owners ==
             right.kernel_completion_event_waits_all_owners &&
         left.request_transaction_append_after_kernel_event ==
             right.request_transaction_append_after_kernel_event &&
         left.transaction_spans_prebound_at_request_admission ==
             right.transaction_spans_prebound_at_request_admission &&
         left.stream_ordered_ready_receipt ==
             right.stream_ordered_ready_receipt &&
         left.host_callback_required == right.host_callback_required &&
         left.host_synchronization_required ==
             right.host_synchronization_required &&
         left.cancel_discards_unpublished_spans ==
             right.cancel_discards_unpublished_spans &&
         left.pointer_publication_without_copy ==
             right.pointer_publication_without_copy &&
         left.output_projection_read_requires_kernel_completion ==
             right.output_projection_read_requires_kernel_completion &&
         left.cancelled_span_reclamation_without_publication ==
             right.cancelled_span_reclamation_without_publication &&
         left.cold_start_only == right.cold_start_only &&
         left.continuation_supported == right.continuation_supported &&
         left.initial_conv_history_device_read ==
             right.initial_conv_history_device_read &&
         left.initial_recurrent_state_device_read ==
             right.initial_recurrent_state_device_read &&
         left.register_zero_initialization ==
             right.register_zero_initialization &&
         left.conv_fp32_fma_oldest_history_to_current ==
             right.conv_fp32_fma_oldest_history_to_current &&
         left.conv_silu_fp32_before_publication ==
             right.conv_silu_fp32_before_publication &&
         left.conv_output_bf16_rne_before_qkv_use ==
             right.conv_output_bf16_rne_before_qkv_use &&
         left.z_bit_exact_bypasses_conv ==
             right.z_bit_exact_bypasses_conv &&
         left.conv_history_stores_raw_current_bf16 ==
             right.conv_history_stores_raw_current_bf16 &&
         left.resource_qualified == right.resource_qualified &&
         left.numerical_contract_qualified ==
             right.numerical_contract_qualified &&
         left.production_dispatch_eligible ==
             right.production_dispatch_eligible;
}

constexpr bool Sm87TargetAotGdnPlan::valid() const noexcept {
  if (capacity_bucket == Sm87TargetAotCapacityBucket::kInvalid ||
      topology !=
          Sm87TargetAotGdnTopology::kQkGroup16OwnersThreeValueHeads ||
      recurrence !=
          Sm87TargetAotGdnRecurrence::kC16TokenOrderedPerTokenBf16 ||
      tactic != Sm87TargetAotGdnTactic::
                    kLayerLongC16Persistent16CtaRegisterState ||
      conv_history_layout != Sm87TargetAotGdnConvHistoryLayout::
                                 kCanonicalQ2048K2048V6144History3 ||
      numerical_contract != Sm87TargetAotGdnNumericalContract::
                                kExactC16CudaCandidatePerTokenBf16 ||
      completion_contract != Sm87TargetAotGdnCompletionContract::
          kIndependentOwnersKernelEventThenRequestTransactionAppend ||
      first_position != 0U ||
      !sm87_target_aot_exact_witness_tokens(token_count)) {
    return false;
  }
  return sm87_target_aot_same_gdn_plan(*this,
                                       sm87_target_aot_gdn_plan(token_count));
}

[[nodiscard]] constexpr Sm87TargetAotGdnOwnerTask
sm87_target_aot_gdn_owner_task(const Sm87TargetAotGdnPlan& plan,
                               const std::size_t owner_cta) noexcept {
  if (!plan.valid() || owner_cta >= plan.owner_ctas) {
    return {};
  }
  return {owner_cta,
          owner_cta,
          owner_cta * plan.value_heads_per_owner,
          plan.value_heads_per_owner,
          owner_cta * plan.state_bytes_per_owner,
          plan.state_bytes_per_owner,
          plan.packed_state_words_per_owner,
          {kSm87TargetAotGdnQConvChannelOffset +
               owner_cta * kSm87TargetAotGdnStateKeyDimension,
           kSm87TargetAotGdnStateKeyDimension,
           (kSm87TargetAotGdnQConvChannelOffset +
            owner_cta * kSm87TargetAotGdnStateKeyDimension) *
               plan.conv_history * kSm87TargetAotGdnBf16Bytes,
           kSm87TargetAotGdnStateKeyDimension * plan.conv_history *
               kSm87TargetAotGdnBf16Bytes},
          {kSm87TargetAotGdnKConvChannelOffset +
               owner_cta * kSm87TargetAotGdnStateKeyDimension,
           kSm87TargetAotGdnStateKeyDimension,
           (kSm87TargetAotGdnKConvChannelOffset +
            owner_cta * kSm87TargetAotGdnStateKeyDimension) *
               plan.conv_history * kSm87TargetAotGdnBf16Bytes,
           kSm87TargetAotGdnStateKeyDimension * plan.conv_history *
               kSm87TargetAotGdnBf16Bytes},
          {kSm87TargetAotGdnVConvChannelOffset +
               owner_cta * plan.value_heads_per_owner *
                   kSm87TargetAotGdnStateValueDimension,
           plan.value_heads_per_owner *
               kSm87TargetAotGdnStateValueDimension,
           (kSm87TargetAotGdnVConvChannelOffset +
            owner_cta * plan.value_heads_per_owner *
                kSm87TargetAotGdnStateValueDimension) *
               plan.conv_history * kSm87TargetAotGdnBf16Bytes,
           plan.value_heads_per_owner *
               kSm87TargetAotGdnStateValueDimension *
               plan.conv_history * kSm87TargetAotGdnBf16Bytes},
          plan.conv_channels_per_owner,
          plan.conv_history_bytes_per_owner,
          true};
}

[[nodiscard]] constexpr Sm87TargetAotGdnPreparationTask
sm87_target_aot_gdn_preparation_task(
    const Sm87TargetAotGdnPlan& plan,
    const std::size_t macro_index) noexcept {
  if (!plan.valid() || macro_index >= plan.preparation_c64_macros) {
    return {};
  }
  const std::size_t token_begin =
      macro_index * kSm87TargetAotGdnPreparationTokens;
  const std::size_t remaining = plan.token_count - token_begin;
  const std::size_t tokens =
      remaining < kSm87TargetAotGdnPreparationTokens
          ? remaining
          : kSm87TargetAotGdnPreparationTokens;
  return {macro_index,
          token_begin,
          tokens,
          tokens / kSm87TargetAotGdnExactRecurrenceTokens,
          (macro_index * kSm87TargetAotGdnC16PerPreparation) %
              plan.preparation_slots,
          false,
          true};
}

[[nodiscard]] constexpr Sm87TargetAotGdnC16Task
sm87_target_aot_gdn_c16_task(const Sm87TargetAotGdnPlan& plan,
                             const std::size_t macro_index,
                             const std::size_t ordinal_in_macro) noexcept {
  const auto preparation =
      sm87_target_aot_gdn_preparation_task(plan, macro_index);
  if (!preparation.valid ||
      ordinal_in_macro >= preparation.ordered_c16_blocks) {
    return {};
  }
  const std::size_t global_index =
      macro_index * kSm87TargetAotGdnC16PerPreparation + ordinal_in_macro;
  return {macro_index,
          ordinal_in_macro,
          global_index,
          preparation.token_begin +
              ordinal_in_macro * kSm87TargetAotGdnExactRecurrenceTokens,
          kSm87TargetAotGdnExactRecurrenceTokens,
          global_index % plan.preparation_slots,
          global_index,
          global_index < plan.preparation_slots
              ? 0U
              : global_index - plan.preparation_slots,
          global_index == 0U ? 0U : global_index - 1U,
          global_index >= plan.preparation_slots,
          global_index != 0U,
          true,
          true,
          true,
          true,
          false,
          true};
}

static_assert(kSm87TargetAotGdnValueHeads ==
              kSm87TargetAotGdnQkGroups *
                  kSm87TargetAotGdnValueHeadsPerQkGroup);
static_assert(kSm87TargetAotGdnOwnerCtas == kSm87TargetAotGdnQkGroups);
static_assert(kSm87TargetAotGdnThreadsPerCta ==
              32U * kSm87TargetAotGdnWarpsPerCta);
static_assert(kSm87TargetAotGdnPreparationTokens ==
              kSm87TargetAotGdnExactRecurrenceTokens *
                  kSm87TargetAotGdnC16PerPreparation);
static_assert(kSm87TargetAotGdnStateKeyStrideElements == 1U);
static_assert(kSm87TargetAotGdnStateValueStrideElements == 128U);
static_assert(kSm87TargetAotGdnStateHeadStrideElements == 16'384U);
static_assert(kSm87TargetAotGdnRecurrentStateLayout.valid());
static_assert(kSm87TargetAotGdnStateBytesPerHead == 32'768U);
static_assert(kSm87TargetAotGdnStateBytesPerOwner == 98'304U);
static_assert(kSm87TargetAotGdnTotalStateBytes == 1'572'864U);
static_assert(kSm87TargetAotGdnPackedStateWordsPerOwner == 24'576U);
static_assert(kSm87TargetAotGdnPackedStateWordsPerThread == 96U);
static_assert(kSm87TargetAotGdnConvChannelsPerOwner == 640U);
static_assert(kSm87TargetAotGdnTotalConvChannels == 10'240U);
static_assert(kSm87TargetAotGdnConvHistoryBytesPerOwner == 3'840U);
static_assert(kSm87TargetAotGdnTotalConvHistoryBytes == 61'440U);
static_assert(kSm87TargetAotGdnQBytesPerPayloadSlot == 4'096U);
static_assert(kSm87TargetAotGdnKBytesPerPayloadSlot == 4'096U);
static_assert(kSm87TargetAotGdnVBytesPerPayloadSlot == 12'288U);
static_assert(kSm87TargetAotGdnZBytesPerPayloadSlot == 12'288U);
static_assert(kSm87TargetAotGdnABytesPerPayloadSlot == 96U);
static_assert(kSm87TargetAotGdnBBytesPerPayloadSlot == 96U);
static_assert(kSm87TargetAotGdnPayloadBytesPerSlot == 32'960U);
static_assert(kSm87TargetAotGdnPrivateSharedPayloadBytes == 65'920U);

}  // namespace q3x::kernels
