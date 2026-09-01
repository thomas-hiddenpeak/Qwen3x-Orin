#pragma once

#if !defined(Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS)
#error "The SM87 AOT arithmetic witness hook is private and default-off"
#endif

#include "q3x/runtime/model_weights.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime::reference_runner_detail {

// A fresh P40000 Legacy-C512 request reaches this seam as 78 full C512 tiles
// plus one C63 tail: 79 prefix calls covering positions [0, 39999).  Position
// 39999 is delegated to the scalar step path and is intentionally outside the
// witness, so a monotone lower-bound calculation may charge that scalar as
// free without claiming it was observed.
inline constexpr std::size_t kAotArithmeticWitnessP40PrefixRows = 39'999U;
inline constexpr std::size_t kAotArithmeticWitnessP40PrefixTileRows = 512U;
inline constexpr std::size_t kAotArithmeticWitnessP40PrefixTileCount = 79U;
inline constexpr std::uint32_t kAotArithmeticWitnessP40FinalScalarPosition =
    39'999U;
static_assert((kAotArithmeticWitnessP40PrefixRows +
               kAotArithmeticWitnessP40PrefixTileRows - 1U) /
                      kAotArithmeticWitnessP40PrefixTileRows ==
                  kAotArithmeticWitnessP40PrefixTileCount);

// Source-local, default-off observation roles for the real Legacy-C512 Prefix
// projection operands.  These roles intentionally match the five grouped
// projection boundaries of AC-PREFILL-SM87-AOT-SYSTEM-v1 without granting that
// unfinished architecture any dispatch authority.
enum class AotArithmeticWitnessProjectionRole : std::uint8_t {
  kInvalid = 0U,
  kNvFp4GateUp,
  kNvFp4Down,
  kFp8GdnQkvZ,
  kFp8FullQkv,
  kFp8AttentionOutput,
};

enum class AotArithmeticWitnessCaptureScope : std::uint8_t {
  kInvalid = 0U,
  // Covers only calls made by ReferenceRunner::enqueue_prefill_layer_segment
  // under ordinary legacy execution control.  The production final prompt
  // token is executed by ReferenceRunner::step and is deliberately excluded.
  kLegacyPrefixLayerSegmentFinalScalarExcluded,
};

struct AotArithmeticWitnessAOperandView {
  AotArithmeticWitnessProjectionRole role =
      AotArithmeticWitnessProjectionRole::kInvalid;
  AotArithmeticWitnessCaptureScope scope =
      AotArithmeticWitnessCaptureScope::kInvalid;
  std::size_t layer = 0U;
  std::uint32_t first_position = 0U;
  std::size_t token_count = 0U;
  std::size_t k = 0U;
  std::size_t n = 0U;
  // Contiguous token-major BF16 A on cuda_stream.  The callback may enqueue an
  // ordered, read-only observation on that stream; it must not mutate A.
  const std::uint16_t* activation_bf16 = nullptr;
  std::size_t activation_row_stride_elements = 0U;
  std::array<const LinearWeight*, 3U> partitions{};
  std::size_t partition_count = 0U;
  void* cuda_stream = nullptr;
  // Set only after the runner has proved legacy control, the SM87
  // weight-only backend, and the Legacy-C512 request-memory profile through
  // the ordinary production prompt-wide scope predicate.
  bool ordinary_sm87_legacy_c512_scope = false;
  // This is a coverage declaration, not a per-call selector.  It is always
  // true for this hook because the scalar-final step has no callsite here.
  bool production_final_scalar_excluded = true;
};

// Zero accepts the observation.  Any nonzero callback result aborts the
// enclosing enqueue before the observed projection and is surfaced as a
// fail-closed CUDA failure.
using AotArithmeticWitnessAOperandCallback = int (*)(
    const AotArithmeticWitnessAOperandView& view,
    void* context) noexcept;

struct AotArithmeticWitnessAOperandHook {
  AotArithmeticWitnessAOperandCallback callback = nullptr;
  void* context = nullptr;
};

[[nodiscard]] inline bool is_valid_aot_arithmetic_witness_a_operand_view(
    const AotArithmeticWitnessAOperandView& view) noexcept {
  if (view.scope != AotArithmeticWitnessCaptureScope::
                        kLegacyPrefixLayerSegmentFinalScalarExcluded ||
      !view.ordinary_sm87_legacy_c512_scope ||
      !view.production_final_scalar_excluded ||
      view.layer >= kQwen36DenseLayerCount || view.token_count == 0U ||
      view.token_count > kAotArithmeticWitnessP40PrefixTileRows ||
      view.activation_bf16 == nullptr || view.k == 0U || view.n == 0U ||
      view.activation_row_stride_elements != view.k ||
      view.token_count >
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max() -
              view.first_position) ||
      view.partition_count == 0U ||
      view.partition_count > view.partitions.size()) {
    return false;
  }
  for (std::size_t index = view.partition_count;
       index < view.partitions.size(); ++index) {
    if (view.partitions[index] != nullptr) {
      return false;
    }
  }

  const auto valid_partition = [&view](
                                   const std::size_t index,
                                   const LinearWeightKind kind,
                                   const std::size_t output_features,
                                   const std::size_t input_features) noexcept {
    const LinearWeight* const weight = view.partitions[index];
    return weight != nullptr && linear_weight_kind(*weight) == kind &&
           linear_output_size(*weight) == output_features &&
           linear_input_size(*weight) == input_features;
  };

  switch (view.role) {
    case AotArithmeticWitnessProjectionRole::kNvFp4GateUp:
      return view.k == 5'120U && view.n == 34'816U &&
             view.partition_count == 2U &&
             valid_partition(0U, LinearWeightKind::kNvFp4, 17'408U,
                             5'120U) &&
             valid_partition(1U, LinearWeightKind::kNvFp4, 17'408U,
                             5'120U);
    case AotArithmeticWitnessProjectionRole::kNvFp4Down:
      return view.k == 17'408U && view.n == 5'120U &&
             view.partition_count == 1U &&
             valid_partition(0U, LinearWeightKind::kNvFp4, 5'120U,
                             17'408U);
    case AotArithmeticWitnessProjectionRole::kFp8GdnQkvZ:
      return view.layer % 4U != 3U && view.k == 5'120U &&
             view.n == 16'384U && view.partition_count == 2U &&
             valid_partition(0U, LinearWeightKind::kFp8, 10'240U, 5'120U) &&
             valid_partition(1U, LinearWeightKind::kFp8, 6'144U, 5'120U);
    case AotArithmeticWitnessProjectionRole::kFp8FullQkv:
      return view.layer % 4U == 3U && view.k == 5'120U &&
             view.n == 14'336U && view.partition_count == 3U &&
             valid_partition(0U, LinearWeightKind::kFp8, 12'288U, 5'120U) &&
             valid_partition(1U, LinearWeightKind::kFp8, 1'024U, 5'120U) &&
             valid_partition(2U, LinearWeightKind::kFp8, 1'024U, 5'120U);
    case AotArithmeticWitnessProjectionRole::kFp8AttentionOutput:
      return view.k == 6'144U && view.n == 5'120U &&
             view.partition_count == 1U &&
             valid_partition(0U, LinearWeightKind::kFp8, 5'120U, 6'144U);
    case AotArithmeticWitnessProjectionRole::kInvalid:
      return false;
  }
  return false;
}

// Present only in binaries compiled with
// Q3X_ENABLE_SM87_AOT_SYSTEM_V1_ARITHMETIC_WITNESS.  The source-local header
// and exchange keep the hook out of the installed ReferenceRunner ABI.
[[nodiscard]] AotArithmeticWitnessAOperandHook
exchange_aot_arithmetic_witness_a_operand_hook(
    AotArithmeticWitnessAOperandHook hook) noexcept;

}  // namespace q3x::runtime::reference_runner_detail
