#include "q3x/runtime/reference_runner.h"

#include "q3x/kernels/sm87_fp8_prefill_supermatrix.h"
#if defined(Q3X_ENABLE_P40_PROJECTION_RESET_ADMISSION)
#include "q3x/kernels/sm87_p40_projection_reset.h"
#endif
#include "q3x/kernels/sm87_p40_packed_projection.h"
#if defined(Q3X_ENABLE_P40_PACKED_NVFP4_V2_ADMISSION)
#include "q3x/kernels/sm87_p40_packed_nvfp4_v2.h"
#endif
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_bf16_ab_prefill.h"
#endif
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_fp8_marlin_w8a16.h"
#endif

#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_nvfp4_marlin.h"
#endif
#if defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_nvfp4_prefill_large_m.h"
#endif
#if defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_nvfp4_prefill_g2_d2.h"
#endif
#if defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
#include "q3x/kernels/sm87_nvfp4_prefill_persistent.h"
#endif

#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
#include "../kernels/reference/gdn_prefill_b8_sequential_sm87.h"
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
#include "../kernels/reference/gdn_prefill_chunk64_cublas_reference_sm87.h"
#include "reference_runner_gdn_chunk64_reference_admission.h"
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
#include "../kernels/reference/gdn_prefill_whole_span_conv_sm87.h"
#include "../kernels/sm87/gdn_prefill_chunk64_native_sm87.h"
#include "reference_runner_gdn_chunk64_native_admission.h"
#endif
#include "../kernels/reference/gdn_prefill_c16_norm_gate_sm87.h"
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
#include "reference_runner_gdn_c16_norm_gate_admission.h"
#endif
#include "q3x/runtime/decode_ops.h"
#include "q3x/runtime/gdn_decode.h"
#include "q3x/runtime/layout_ops.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <type_traits>
#include <utility>
#include <variant>

namespace q3x::runtime {
namespace {

constexpr std::size_t kLinearQkvElements = 10'240U;
constexpr std::size_t kLinearValueElements = 6'144U;
constexpr std::size_t kLinearScalarElements = 48U;
constexpr std::size_t kFullQueryHeads = 24U;
constexpr std::size_t kFullKvHeads = 4U;
constexpr std::size_t kFullHeadDimension = 256U;
constexpr std::size_t kFullQueryElements =
    kFullQueryHeads * kFullHeadDimension;
constexpr std::size_t kFullQGateElements = 2U * kFullQueryElements;
constexpr std::size_t kFullKvElements =
    kFullKvHeads * kFullHeadDimension;
constexpr std::size_t kRopePairs = 32U;
constexpr std::size_t kPrefillKernelTileMaximumTokens = 16U;
constexpr std::size_t kFullAttentionPreprocessTileMaximumTokens =
    kFullAttentionPreprocessMaximumTokens;
constexpr std::size_t kProductionProjectionSubtileTokens = 32U;
constexpr std::size_t kPromptWideP40WholeCorePromptTokens = 40'000U;
constexpr std::size_t kPromptWideP40WholeCoreRequestCapacityTokens = 40'001U;
constexpr std::size_t kPromptWideP40WholeCorePanelTokens = 8'000U;
constexpr std::size_t kPromptWideP40WholeCorePanelCount = 5U;
constexpr std::size_t kPromptWideP40WholeCoreFillPhases = 5U;
constexpr std::size_t kPromptWideP40WholeCorePromptCorePhase = 5U;
constexpr std::size_t kPromptWideP40WholeCoreDrainPhaseBegin = 6U;
constexpr std::size_t kPromptWideP40WholeCorePersistentMlpPhase = 11U;
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kAttentionScale = 1.0F / 16.0F;

enum class PrefillProjectionExecution : std::uint8_t {
  kUnknown = 0,
  kGenericExact,
  kFp8Marlin,
  kFp8WholeChunk,
  kFp8Supermatrix,
  kFp8M64Output,
};

enum class PrefillGdnExecution : std::uint8_t {
  kUnknown = 0,
  kChunk64Native,
  kC16Exact,
  kWarpExact,
  kExternalReference,
  kApproximateB8,
};

[[nodiscard]] PrefillRouteDisposition projection_disposition(
    const PrefillProjectionExecution execution) noexcept {
  switch (execution) {
    case PrefillProjectionExecution::kFp8Marlin:
    case PrefillProjectionExecution::kFp8WholeChunk:
    case PrefillProjectionExecution::kFp8Supermatrix:
    case PrefillProjectionExecution::kFp8M64Output:
      return PrefillRouteDisposition::kProduction;
    case PrefillProjectionExecution::kGenericExact:
      return PrefillRouteDisposition::kExactFallback;
    case PrefillProjectionExecution::kUnknown:
      return PrefillRouteDisposition::kForbidden;
  }
  return PrefillRouteDisposition::kForbidden;
}

[[nodiscard]] bool
full_attention_preprocess_prompt_wide_environment_enabled() noexcept {
  const char* const value = std::getenv(
      "Q3X_RUN_FULL_ATTENTION_PREPROCESS_PROMPT_WIDE_128_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_full_attention_preprocess_prompt_wide_admission =
    full_attention_preprocess_prompt_wide_environment_enabled();

[[nodiscard]] bool decode_gqa_splitkv_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_DECODE_GQA_SPLITKV_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_decode_gqa_splitkv_admission =
    decode_gqa_splitkv_environment_enabled();

[[nodiscard]] bool
prefill_residual_rms_prompt_wide_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_PREFILL_RESIDUAL_RMS_PROMPT_WIDE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_prefill_residual_rms_prompt_wide_admission =
    prefill_residual_rms_prompt_wide_environment_enabled();

[[nodiscard]] bool
prefill_embedding_prompt_wide_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_PREFILL_EMBEDDING_PROMPT_WIDE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_prefill_embedding_prompt_wide_admission =
    prefill_embedding_prompt_wide_environment_enabled();

[[nodiscard]] bool
gdn_conv_token_parallel_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_GDN_CONV_TOKEN_PARALLEL_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_gdn_conv_token_parallel_admission =
    gdn_conv_token_parallel_environment_enabled();

[[nodiscard]] bool
gdn_conv_compact_qk_fused_candidate_environment_enabled() noexcept {
  const char* const baseline = std::getenv(
      "Q3X_RUN_GDN_CONV_COMPACT_QK_STANDALONE_BASELINE");
  if (baseline != nullptr && std::strcmp(baseline, "1") == 0) {
    return false;
  }
  // The exact real-weight, pure-Graph, and same-engine B-C-C-B gates admit
  // this route as the native token-parallel production default.  Preserve
  // the original candidate selector as a compatibility override: an
  // explicit value other than "1" selects the old two-kernel path.
  const char* const compatibility_selector = std::getenv(
      "Q3X_RUN_GDN_CONV_COMPACT_QK_FUSED_CANDIDATE");
  return compatibility_selector == nullptr ||
         std::strcmp(compatibility_selector, "1") == 0;
}

thread_local bool g_enable_gdn_conv_compact_qk_fused_candidate =
    gdn_conv_compact_qk_fused_candidate_environment_enabled();
thread_local std::size_t g_gdn_conv_compact_qk_fused_candidate_hits = 0U;

#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
[[nodiscard]] bool nvfp4_marlin_prefill_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_NVFP4_MARLIN_PREFILL_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_nvfp4_marlin_prefill_admission =
    nvfp4_marlin_prefill_environment_enabled();
thread_local std::size_t g_nvfp4_marlin_prefill_admission_hits = 0U;

[[nodiscard]] bool
prefill_marlin_gate_up_epilogue_environment_enabled() noexcept {
  const char* const value = std::getenv(
      "Q3X_RUN_PREFILL_MARLIN_GATE_UP_EPILOGUE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_prefill_marlin_gate_up_epilogue_admission =
    prefill_marlin_gate_up_epilogue_environment_enabled();
#endif

#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
[[nodiscard]] bool fp8_marlin_prefill_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_FP8_MARLIN_PREFILL_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_fp8_marlin_prefill_admission =
    fp8_marlin_prefill_environment_enabled();
thread_local std::size_t g_fp8_marlin_prefill_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
[[nodiscard]] bool bf16_ab_large_m_prefill_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_BF16_AB_LARGE_M_PREFILL_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_bf16_ab_large_m_prefill_admission =
    bf16_ab_large_m_prefill_environment_enabled();
thread_local std::size_t g_bf16_ab_large_m_prefill_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
// Admission-only switch. It has internal storage and defaults off in every
// thread, so the public runner and CLI retain the production M16 route. The
// matching test-only setter is deliberately absent from the public header.
thread_local bool g_enable_prefill_gdn_b8_admission = false;
thread_local std::size_t g_prefill_gdn_b8_admission_hits = 0U;
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
// Native C64/WY architecture admission. This build remains test-only and the
// worker-local route is enabled only by the exact value "1". It has no
// external-library context, workspace, fallback, or default-route authority.
[[nodiscard]] bool gdn_chunk64_native_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_GDN_CHUNK64_NATIVE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_prefill_gdn_chunk64_native_admission =
    gdn_chunk64_native_environment_enabled();
thread_local std::size_t g_prefill_gdn_chunk64_native_admission_hits = 0U;
thread_local reference_runner_detail::PrefillGdnChunk64NativeSnapshotHook
    g_prefill_gdn_chunk64_native_snapshot_hook{};
thread_local reference_runner_detail::
    PrefillGdnChunk64NativeFinalSnapshotHook
        g_prefill_gdn_chunk64_native_final_snapshot_hook{};
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
// Compile-time-isolated architecture switch. It stays false unless the
// admission-only environment value is exactly "1"; the external reference
// context has no fallback or production authority.
// The evaluation server runs generation on a dedicated worker. Initialize
// that worker's private switch from the same explicit environment gate used
// by the test harness; synchronous tests may still override it through the
// private exchange accessor between runner calls.
[[nodiscard]] bool gdn_chunk64_reference_environment_enabled() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_GDN_CHUNK64_REFERENCE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_prefill_gdn_chunk64_reference_admission =
    gdn_chunk64_reference_environment_enabled();
thread_local std::size_t g_prefill_gdn_chunk64_reference_admission_hits = 0U;
thread_local reference_runner_detail::
    PrefillGdnChunk64ReferenceSnapshotHook
        g_prefill_gdn_chunk64_reference_snapshot_hook{};
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
// The opt-in test build can still select the legacy route (false) or the
// production fused route (true) inside one ELF. Release builds do not expose
// this switch and always use the production selector below.
thread_local bool g_enable_prefill_gdn_c16_norm_gate_admission = false;
thread_local std::size_t g_prefill_gdn_c16_norm_gate_admission_hits = 0U;
thread_local reference_runner_detail::
    PrefillGdnC16NormGateAdmissionSnapshotHook
        g_prefill_gdn_c16_norm_gate_admission_snapshot_hook{};
#endif

static_assert(kLinearQkvElements <= kReferenceIntermediateSize);
static_assert(kLinearValueElements <= kReferenceIntermediateSize);
static_assert(kFullQGateElements <= kReferenceIntermediateSize);
static_assert(kFullQueryElements <= kReferenceIntermediateSize);
static_assert(kFullKvElements <= kReferenceIntermediateSize);
static_assert(kPrefillKernelTileMaximumTokens ==
              kQkRopeTileMaximumTokens);
static_assert(kFullAttentionPreprocessTileMaximumTokens ==
              kMaximumRequestPrefillChunkSize);
static_assert(kMaximumRequestPrefillChunkSize == 512U);
static_assert(kReferenceHiddenSize == 5'120U);
static_assert(kProductionProjectionSubtileTokens ==
              reference_runner_detail::kPrefillResidualRmsM32Tokens);
static_assert(kProductionProjectionSubtileTokens <=
              kMaximumProjectionTileTokenCount);
static_assert(kMaximumProjectionTileTokenCount <=
              kMaximumRequestPrefillChunkSize);

[[nodiscard]] bool byte_range_overflows(const void* const pointer,
                                        const std::size_t bytes) noexcept {
  const std::uintptr_t begin = reinterpret_cast<std::uintptr_t>(pointer);
  return pointer == nullptr ||
         bytes > std::numeric_limits<std::uintptr_t>::max() - begin;
}

// Treat address overflow as overlap so a malformed span can never select a
// concurrent writer. Exactly adjacent half-open ranges remain disjoint.
[[nodiscard]] bool byte_ranges_are_disjoint(
    const void* const first, const std::size_t first_bytes,
    const void* const second, const std::size_t second_bytes) noexcept {
  if (byte_range_overflows(first, first_bytes) ||
      byte_range_overflows(second, second_bytes)) {
    return false;
  }
  const std::uintptr_t first_begin =
      reinterpret_cast<std::uintptr_t>(first);
  const std::uintptr_t second_begin =
      reinterpret_cast<std::uintptr_t>(second);
  const std::uintptr_t first_end = first_begin + first_bytes;
  const std::uintptr_t second_end = second_begin + second_bytes;
  return first_end <= second_begin || second_end <= first_begin;
}

struct ByteSpan {
  const void* data = nullptr;
  std::size_t bytes = 0U;
};

template <std::size_t SpanCount>
[[nodiscard]] bool byte_ranges_are_pairwise_disjoint(
    const std::array<ByteSpan, SpanCount>& spans) noexcept {
  for (std::size_t left = 0U; left < spans.size(); ++left) {
    for (std::size_t right = left + 1U; right < spans.size(); ++right) {
      if (!byte_ranges_are_disjoint(
              spans[left].data, spans[left].bytes,
              spans[right].data, spans[right].bytes)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] ReferenceRunnerStatus runner_status(
    const ReferenceRunnerError error, const char* const operation,
    const std::size_t layer = kReferenceNoLayer,
    const int cuda_error = 0,
    const std::uint64_t retired_prefill_quanta = 0U) noexcept {
  return {error, cuda_error, layer, operation,
          retired_prefill_quanta};
}

[[nodiscard]] bool valid_vector(const Bf16VectorWeight& weight,
                                const std::size_t elements) noexcept {
  return weight.data != nullptr && weight.element_count == elements;
}

[[nodiscard]] bool valid_linear_payload(const LinearWeight& weight,
                                        const std::size_t output_size,
                                        const std::size_t input_size) noexcept {
  if (linear_output_size(weight) != output_size ||
      linear_input_size(weight) != input_size) {
    return false;
  }
  return std::visit(
      [](const auto& selected) noexcept {
        using Selected = std::decay_t<decltype(selected)>;
        if constexpr (std::is_same_v<Selected, Bf16LinearWeight>) {
          return selected.weight != nullptr;
        } else if constexpr (std::is_same_v<Selected, Fp8LinearWeight>) {
          return selected.weight != nullptr &&
                 selected.weight_scale_device != nullptr &&
                 selected.input_scale_device != nullptr &&
                 std::isfinite(selected.weight_scale) &&
                 selected.weight_scale >= 0.0F &&
                 std::isfinite(selected.input_scale) &&
                 selected.input_scale > 0.0F;
        } else {
          return selected.packed_weight != nullptr &&
                 selected.block_scale != nullptr &&
                 selected.weight_scale_2_device != nullptr &&
                 selected.input_scale_device != nullptr &&
                 std::isfinite(selected.weight_scale_2) &&
                 selected.weight_scale_2 >= 0.0F &&
                 std::isfinite(selected.input_scale) &&
                 selected.input_scale >= 0.0F &&
                 selected.input_size % 16U == 0U;
        }
      },
      weight);
}

[[nodiscard]] ReferenceRunnerStatus validate_model_weights(
    const ModelWeights* const weights) noexcept {
  if (weights == nullptr) {
    return runner_status(ReferenceRunnerError::kInvalidDependency,
                         "model_weights");
  }
  const Bf16LinearWeight& embedding = weights->embed_tokens();
  if (embedding.weight == nullptr ||
      embedding.output_size != kReferenceVocabularySize ||
      embedding.input_size != kReferenceHiddenSize) {
    return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                         "embed_tokens");
  }
  if (!valid_vector(weights->final_norm(), kReferenceHiddenSize) ||
      !valid_linear_payload(weights->lm_head(), kReferenceVocabularySize,
                            kReferenceHiddenSize)) {
    return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                         "final_norm_or_lm_head");
  }

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& selected = weights->layer(layer);
    if (!valid_vector(selected.input_layernorm, kReferenceHiddenSize) ||
        !valid_vector(selected.post_attention_layernorm,
                      kReferenceHiddenSize) ||
        !valid_linear_payload(selected.mlp.gate_proj,
                              kReferenceIntermediateSize,
                              kReferenceHiddenSize) ||
        !valid_linear_payload(selected.mlp.up_proj,
                              kReferenceIntermediateSize,
                              kReferenceHiddenSize) ||
        !valid_linear_payload(selected.mlp.down_proj,
                              kReferenceHiddenSize,
                              kReferenceIntermediateSize)) {
      return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                           "decoder_common_weights", layer);
    }

    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&selected.attention);
      if (attention == nullptr) {
        return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                             "linear_attention_variant", layer);
      }
      const bool conv_shape = attention->conv1d.data != nullptr &&
                              attention->conv1d.shape[0] ==
                                  kLinearQkvElements &&
                              attention->conv1d.shape[1] == 1U &&
                              attention->conv1d.shape[2] == 4U;
      if (!valid_linear_payload(attention->in_proj_qkv,
                                kLinearQkvElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->in_proj_z,
                                kLinearValueElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->in_proj_a,
                                kLinearScalarElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->in_proj_b,
                                kLinearScalarElements,
                                kReferenceHiddenSize) ||
          !conv_shape ||
          !valid_vector(attention->a_log, kLinearScalarElements) ||
          !valid_vector(attention->dt_bias, kLinearScalarElements) ||
          !valid_vector(attention->norm, kGdnHeadDimension) ||
          !valid_linear_payload(attention->out_proj,
                                kReferenceHiddenSize,
                                kLinearValueElements)) {
        return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                             "linear_attention_weights", layer);
      }
    } else if (expected == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&selected.attention);
      if (attention == nullptr) {
        return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                             "full_attention_variant", layer);
      }
      if (!valid_linear_payload(attention->q_proj, kFullQGateElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->k_proj, kFullKvElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->v_proj, kFullKvElements,
                                kReferenceHiddenSize) ||
          !valid_linear_payload(attention->o_proj, kReferenceHiddenSize,
                                kFullQueryElements) ||
          !valid_vector(attention->q_norm, kFullHeadDimension) ||
          !valid_vector(attention->k_norm, kFullHeadDimension)) {
        return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                             "full_attention_weights", layer);
      }
    } else {
      return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                           "layer_index", layer);
    }
  }
  return {};
}

[[nodiscard]] bool valid_view(const RequestViewResult& view,
                              const std::uint64_t minimum_elements,
                              const std::uint32_t element_size) noexcept {
  return view && view.value->device_data != nullptr &&
         view.value->element_capacity >= minimum_elements &&
         view.value->element_size_bytes == element_size;
}

[[nodiscard]] bool valid_const_view(const RequestConstViewResult& view,
                                    const std::uint64_t minimum_elements,
                                    const std::uint32_t element_size) noexcept {
  return view && view.value->device_data != nullptr &&
         view.value->element_capacity >= minimum_elements &&
         view.value->element_size_bytes == element_size;
}

#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
[[nodiscard]] bool use_prefill_gdn_b8_admission(
    const bool enabled, const ProjectionBackend backend,
    const std::uint32_t first_position,
    const std::size_t token_count) noexcept {
  return enabled && backend == ProjectionBackend::kSm87WeightOnly &&
         (token_count == 256U || token_count == 512U) &&
         first_position % 8U == 0U;
}
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
[[nodiscard]] bool use_prefill_gdn_chunk64_native_admission(
    const bool enabled, const ProjectionBackend backend,
    const std::uint32_t /*first_position*/,
    const std::size_t token_count, const void* const workspace,
    const std::size_t workspace_bytes) noexcept {
  // The fixed C64 hierarchy can pad any C1..C512 tile. Production admission
  // starts at C32: the real scheduler keeps an isolated C1 seed/tail on its
  // scalar path, while exact C32/C52/C481 runner screens prove the bulk route.
  return enabled && backend == ProjectionBackend::kSm87WeightOnly &&
         token_count >= 32U && token_count <= 512U &&
         workspace != nullptr &&
         workspace_bytes >=
             gdn_prefill_chunk64_native_detail::workspace_bytes();
}
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
[[nodiscard]] bool use_prefill_gdn_chunk64_reference_admission(
    const bool enabled, const ProjectionBackend backend,
    const std::uint32_t first_position, const std::size_t token_count,
    const void* const context, const void* const workspace,
    const std::size_t workspace_bytes) noexcept {
  return enabled && backend == ProjectionBackend::kSm87WeightOnly &&
         first_position == 0U && token_count == 512U && context != nullptr &&
         workspace != nullptr &&
         workspace_bytes >=
             gdn_prefill_chunk64_reference_detail::workspace_bytes();
}
#endif
[[nodiscard]] bool should_use_prefill_gdn_c16_norm_gate(
    const bool enabled, const ProjectionBackend backend,
    const std::uint32_t first_position,
    const std::size_t token_count) noexcept {
  return enabled && backend == ProjectionBackend::kSm87WeightOnly &&
         (token_count == 256U || token_count == 512U) &&
         first_position % kPrefillKernelTileMaximumTokens == 0U;
}

}  // namespace

ReferenceRunnerStatus ReferenceRunner::collect_request_views(
    RequestState* const state, Views& views) noexcept {
  if (state == nullptr) {
    return runner_status(ReferenceRunnerError::kInvalidDependency,
                         "request_state");
  }
  if (!static_cast<bool>(*state)) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "empty_request_state");
  }
  const ReferenceRunnerError plan_error =
      reference_runner_detail::validate_reference_workspace_plan(state->plan());
  if (plan_error != ReferenceRunnerError::kNone) {
    return runner_status(plan_error, "request_memory_plan");
  }
  if (state->sequence_length() > state->max_sequence_length()) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "request_sequence_length");
  }
  const std::uint64_t workspace_tokens = state->plan().prefill_chunk_size;

  for (std::size_t index = 0U; index < 3U; ++index) {
    RequestViewResult view = state->hidden_buffer(index);
    if (!valid_view(view, workspace_tokens * kReferenceHiddenSize,
                    sizeof(std::uint16_t))) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "hidden_workspace");
    }
    views.hidden[index] =
        static_cast<std::uint16_t*>(view.value->device_data);
  }
  for (std::size_t index = 0U; index < 4U; ++index) {
    RequestViewResult view = state->projection_buffer(index);
    if (!valid_view(view, workspace_tokens * kReferenceIntermediateSize,
                    sizeof(std::uint16_t))) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "projection_workspace");
    }
    views.projection[index] =
        static_cast<std::uint16_t*>(view.value->device_data);
  }

  RequestViewResult linear_a = state->linear_a_buffer();
  RequestViewResult linear_b = state->linear_b_buffer();
  RequestViewResult scratch = state->fp32_scratch();
  if (!valid_view(linear_a, workspace_tokens * kLinearScalarElements,
                  sizeof(std::uint16_t)) ||
      !valid_view(linear_b, workspace_tokens * kLinearScalarElements,
                  sizeof(std::uint16_t)) ||
      !valid_view(scratch, kReferenceVocabularySize, sizeof(float))) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "scalar_or_fp32_workspace");
  }
  views.linear_a = static_cast<std::uint16_t*>(linear_a.value->device_data);
  views.linear_b = static_cast<std::uint16_t*>(linear_b.value->device_data);
  views.fp32_scratch = static_cast<float*>(scratch.value->device_data);
  views.fp32_scratch_elements =
      static_cast<std::size_t>(scratch.value->element_capacity);
  const RequestConstViewResult rope_cos = state->rope_cos(0U);
  const RequestConstViewResult rope_sin = state->rope_sin(0U);
  if (!valid_const_view(rope_cos, kRopePairs, sizeof(float)) ||
      !valid_const_view(rope_sin, kRopePairs, sizeof(float))) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "rope_workspace");
  }
  views.rope_cos = static_cast<const float*>(rope_cos.value->device_data);
  views.rope_sin = static_cast<const float*>(rope_sin.value->device_data);

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      RequestViewResult conv = state->conv_state(layer);
      RequestViewResult gdn = state->gdn_state(layer);
      if (!valid_view(conv, kGdnQkvChannels * kGdnConvHistoryWidth,
                      sizeof(std::uint16_t)) ||
          !valid_view(gdn, kGdnStateElements, sizeof(std::uint16_t))) {
        return runner_status(ReferenceRunnerError::kInvalidRequestState,
                             "linear_persistent_state", layer);
      }
      views.conv_state[layer] =
          static_cast<std::uint16_t*>(conv.value->device_data);
      views.gdn_state[layer] =
          static_cast<std::uint16_t*>(gdn.value->device_data);
    } else if (expected == model::LayerType::kFullAttention) {
      RequestViewResult key = state->key_cache(layer);
      RequestViewResult value = state->value_cache(layer);
      const std::uint64_t required =
          static_cast<std::uint64_t>(state->max_sequence_length()) *
          kFullKvElements;
      if (!valid_view(key, required, sizeof(std::uint16_t)) ||
          !valid_view(value, required, sizeof(std::uint16_t))) {
        return runner_status(ReferenceRunnerError::kInvalidRequestState,
                             "full_attention_cache", layer);
      }
      views.key_cache[layer] =
          static_cast<std::uint16_t*>(key.value->device_data);
      views.value_cache[layer] =
          static_cast<std::uint16_t*>(value.value->device_data);
    } else {
      return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                           "request_layer_schedule", layer);
    }
  }
  return {};
}

ReferenceRunnerStatus ReferenceRunner::map_layer_major_candidate_views(
    const ReferenceLayerMajorRequestViews& candidate,
    Views& views) noexcept {
  const ReferenceLayerMajorRequestBindingDescriptor& descriptor =
      candidate.descriptor;
  const bool layer_major_profile =
      descriptor.profile == RequestMemoryProfile::kLayerMajorC8192 ||
      descriptor.profile == RequestMemoryProfile::kLayerMajorP40WholeCore;
  if (!layer_major_profile ||
      descriptor.legacy_prefill_chunk_size !=
          kMaximumRequestPrefillChunkSize) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_runner_profile");
  }

  Views mapped;
  for (std::size_t index = 0U; index < 3U; ++index) {
    if (candidate.legacy_c512.hidden_bf16[index].storage.device_data ==
        nullptr) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_runner_hidden_workspace");
    }
    mapped.hidden[index] = static_cast<std::uint16_t*>(
        candidate.legacy_c512.hidden_bf16[index].storage.device_data);
  }
  for (std::size_t index = 0U; index < 4U; ++index) {
    if (candidate.legacy_c512.projection_bf16[index].storage.device_data ==
        nullptr) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_runner_projection_workspace");
    }
    mapped.projection[index] = static_cast<std::uint16_t*>(
        candidate.legacy_c512.projection_bf16[index].storage.device_data);
  }
  if (candidate.legacy_c512.linear_a_bf16.storage.device_data == nullptr ||
      candidate.legacy_c512.linear_b_bf16.storage.device_data == nullptr ||
      candidate.legacy_c512.fp32_scratch.device_data == nullptr ||
      candidate.legacy_c512.fp32_scratch.element_capacity <
          kReferenceVocabularySize) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_runner_scalar_or_fp32_workspace");
  }
  mapped.linear_a = static_cast<std::uint16_t*>(
      candidate.legacy_c512.linear_a_bf16.storage.device_data);
  mapped.linear_b = static_cast<std::uint16_t*>(
      candidate.legacy_c512.linear_b_bf16.storage.device_data);
  mapped.fp32_scratch =
      static_cast<float*>(candidate.legacy_c512.fp32_scratch.device_data);
  mapped.fp32_scratch_elements = static_cast<std::size_t>(
      candidate.legacy_c512.fp32_scratch.element_capacity);

  if (candidate.persistent.rope_cos_fp32.device_data == nullptr ||
      candidate.persistent.rope_sin_fp32.device_data == nullptr) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_runner_rope_workspace");
  }
  mapped.rope_cos = static_cast<const float*>(
      candidate.persistent.rope_cos_fp32.device_data);
  mapped.rope_sin = static_cast<const float*>(
      candidate.persistent.rope_sin_fp32.device_data);

  std::size_t linear_slot = 0U;
  std::size_t full_slot = 0U;
  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    const RequestLayerSlot& slot = descriptor.layers[layer];
    const std::size_t expected_slot =
        expected == model::LayerType::kFullAttention ? full_slot++
                                                     : linear_slot++;
    if (slot.type != expected || slot.slot != expected_slot) {
      return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                           "layer_major_runner_layer_schedule", layer);
    }
    if (expected == model::LayerType::kLinearAttention) {
      if (slot.slot >= candidate.persistent.conv_state_bf16.size() ||
          candidate.persistent.conv_state_bf16[slot.slot].device_data ==
              nullptr ||
          candidate.persistent.gdn_state_bf16[slot.slot].device_data ==
              nullptr) {
        return runner_status(ReferenceRunnerError::kInvalidRequestState,
                             "layer_major_runner_linear_state", layer);
      }
      mapped.conv_state[layer] = static_cast<std::uint16_t*>(
          candidate.persistent.conv_state_bf16[slot.slot].device_data);
      mapped.gdn_state[layer] = static_cast<std::uint16_t*>(
          candidate.persistent.gdn_state_bf16[slot.slot].device_data);
    } else if (expected == model::LayerType::kFullAttention) {
      if (slot.slot >= candidate.persistent.key_cache_bf16.size() ||
          candidate.persistent.key_cache_bf16[slot.slot].device_data ==
              nullptr ||
          candidate.persistent.value_cache_bf16[slot.slot].device_data ==
              nullptr) {
        return runner_status(ReferenceRunnerError::kInvalidRequestState,
                             "layer_major_runner_full_attention_state",
                             layer);
      }
      mapped.key_cache[layer] = static_cast<std::uint16_t*>(
          candidate.persistent.key_cache_bf16[slot.slot].device_data);
      mapped.value_cache[layer] = static_cast<std::uint16_t*>(
          candidate.persistent.value_cache_bf16[slot.slot].device_data);
    } else {
      return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                           "layer_major_runner_layer_schedule", layer);
    }
  }
  if (linear_slot != kRequestLinearLayerCount ||
      full_slot != kRequestFullLayerCount) {
    return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                         "layer_major_runner_layer_schedule");
  }

  views = mapped;
  return {};
}

ReferenceRunnerStatus ReferenceRunner::bind_layer_major_candidate_views(
    ReferenceLayerMajorRequestViews&& candidate) noexcept {
  Views mapped;
  const ReferenceRunnerStatus status =
      map_layer_major_candidate_views(candidate, mapped);
  if (!status) {
    return status;
  }
  layer_major_request_views_.emplace(std::move(candidate));
  views_ = mapped;
  return {};
}

namespace {

[[nodiscard]] ConstBf16Span trace_span(
    const ReferenceTraceView& trace, const std::size_t offset) noexcept {
  if (trace.data == nullptr || trace.element_count < kReferenceTraceElements ||
      offset > kReferenceTraceElements - kReferenceHiddenSize) {
    return {};
  }
  return {trace.data + offset, kReferenceHiddenSize};
}

}  // namespace

const char* reference_runner_error_string(
    const ReferenceRunnerError error) noexcept {
  switch (error) {
    case ReferenceRunnerError::kNone:
      return "none";
    case ReferenceRunnerError::kInvalidDependency:
      return "invalid_dependency";
    case ReferenceRunnerError::kInvalidModelWeights:
      return "invalid_model_weights";
    case ReferenceRunnerError::kInvalidRequestState:
      return "invalid_request_state";
    case ReferenceRunnerError::kInvalidLayerSchedule:
      return "invalid_layer_schedule";
    case ReferenceRunnerError::kCudaFailure:
      return "cuda_failure";
    case ReferenceRunnerError::kAllocationFailure:
      return "allocation_failure";
    case ReferenceRunnerError::kInvalidRunner:
      return "invalid_runner";
    case ReferenceRunnerError::kPoisoned:
      return "poisoned";
    case ReferenceRunnerError::kTokenOutOfRange:
      return "token_out_of_range";
    case ReferenceRunnerError::kCapacityExceeded:
      return "capacity_exceeded";
    case ReferenceRunnerError::kTraceUnavailable:
      return "trace_unavailable";
    case ReferenceRunnerError::kNonFiniteLogits:
      return "nonfinite_logits";
    case ReferenceRunnerError::kStateCommitFailure:
      return "state_commit_failure";
    case ReferenceRunnerError::kInvalidStepOptions:
      return "invalid_step_options";
    case ReferenceRunnerError::kRouteEvidenceFailure:
      return "route_evidence_failure";
    case ReferenceRunnerError::kCancelled:
      return "cancelled";
  }
  return "unknown";
}

ConstBf16Span ReferenceTraceView::raw() const noexcept {
  if (data == nullptr || element_count < kReferenceTraceElements) {
    return {};
  }
  return {data, kReferenceTraceElements};
}

ConstBf16Span ReferenceTraceView::embedding() const noexcept {
  return trace_span(*this, 0U);
}

ConstBf16Span ReferenceTraceView::layer_hidden(
    const std::size_t layer) const noexcept {
  if (layer >= kReferenceDecoderLayerCount) {
    return {};
  }
  const std::size_t offset =
      kReferenceHiddenSize + 2U * layer * kReferenceHiddenSize;
  return trace_span(*this, offset);
}

ConstBf16Span ReferenceTraceView::layer_residual(
    const std::size_t layer) const noexcept {
  if (layer >= kReferenceDecoderLayerCount) {
    return {};
  }
  const std::size_t offset =
      kReferenceHiddenSize + (2U * layer + 1U) * kReferenceHiddenSize;
  return trace_span(*this, offset);
}

ConstBf16Span ReferenceTraceView::final_norm() const noexcept {
  const std::size_t offset =
      (1U + 2U * kReferenceDecoderLayerCount) * kReferenceHiddenSize;
  return trace_span(*this, offset);
}

namespace reference_runner_detail {

bool exchange_nvfp4_marlin_prefill_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
  return std::exchange(g_enable_nvfp4_marlin_prefill_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

std::size_t exchange_nvfp4_marlin_prefill_admission_test_hits(
    const std::size_t hits) noexcept {
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
  return std::exchange(g_nvfp4_marlin_prefill_admission_hits, hits);
#else
  (void)hits;
  return 0U;
#endif
}

bool exchange_fp8_marlin_prefill_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  return std::exchange(g_enable_fp8_marlin_prefill_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

std::size_t exchange_fp8_marlin_prefill_admission_test_hits(
    const std::size_t hits) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  return std::exchange(g_fp8_marlin_prefill_admission_hits, hits);
#else
  (void)hits;
  return 0U;
#endif
}

#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
// Private test hook for real-checkpoint admission. Keeping this declaration
// out of the installed header prevents the numerically distinct B8 recurrence
// from becoming a supported runtime option before its state/Decode gates pass.
bool exchange_prefill_gdn_b8_admission_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(g_enable_prefill_gdn_b8_admission, enabled);
}

std::size_t exchange_prefill_gdn_b8_admission_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(g_prefill_gdn_b8_admission_hits, hits);
}
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
bool exchange_prefill_gdn_chunk64_native_admission_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(g_enable_prefill_gdn_chunk64_native_admission,
                       enabled);
}

std::size_t exchange_prefill_gdn_chunk64_native_admission_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(g_prefill_gdn_chunk64_native_admission_hits, hits);
}

std::size_t exchange_gdn_conv_compact_qk_fused_candidate_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(g_gdn_conv_compact_qk_fused_candidate_hits, hits);
}

bool exchange_gdn_conv_compact_qk_fused_candidate_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(g_enable_gdn_conv_compact_qk_fused_candidate,
                       enabled);
}

PrefillGdnChunk64NativeSnapshotHook
exchange_prefill_gdn_chunk64_native_snapshot_hook(
    const PrefillGdnChunk64NativeSnapshotHook hook) noexcept {
  return std::exchange(g_prefill_gdn_chunk64_native_snapshot_hook, hook);
}

PrefillGdnChunk64NativeFinalSnapshotHook
exchange_prefill_gdn_chunk64_native_final_snapshot_hook(
    const PrefillGdnChunk64NativeFinalSnapshotHook hook) noexcept {
  return std::exchange(g_prefill_gdn_chunk64_native_final_snapshot_hook,
                       hook);
}
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
bool exchange_prefill_gdn_chunk64_reference_admission_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(
      g_enable_prefill_gdn_chunk64_reference_admission, enabled);
}

std::size_t exchange_prefill_gdn_chunk64_reference_admission_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(
      g_prefill_gdn_chunk64_reference_admission_hits, hits);
}

PrefillGdnChunk64ReferenceSnapshotHook
exchange_prefill_gdn_chunk64_reference_snapshot_hook(
    const PrefillGdnChunk64ReferenceSnapshotHook hook) noexcept {
  return std::exchange(g_prefill_gdn_chunk64_reference_snapshot_hook, hook);
}
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
bool exchange_prefill_gdn_c16_norm_gate_admission_test_enabled(
    const bool enabled) noexcept {
  return std::exchange(g_enable_prefill_gdn_c16_norm_gate_admission,
                       enabled);
}

std::size_t exchange_prefill_gdn_c16_norm_gate_admission_test_hits(
    const std::size_t hits) noexcept {
  return std::exchange(g_prefill_gdn_c16_norm_gate_admission_hits, hits);
}

PrefillGdnC16NormGateAdmissionSnapshotHook
exchange_prefill_gdn_c16_norm_gate_admission_snapshot_hook(
    const PrefillGdnC16NormGateAdmissionSnapshotHook hook) noexcept {
  return std::exchange(
      g_prefill_gdn_c16_norm_gate_admission_snapshot_hook, hook);
}
#endif

std::uint16_t float_to_bf16_rne(const float value) noexcept {
  std::uint32_t bits = 0U;
  std::memcpy(&bits, &value, sizeof(bits));
  const std::uint32_t exponent = bits & 0x7F80'0000U;
  const std::uint32_t mantissa = bits & 0x007F'FFFFU;
  if (exponent == 0x7F80'0000U && mantissa != 0U) {
    // Preserve sign/payload high bits and force a quiet BF16 NaN.
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  const std::uint32_t rounding_bias =
      0x7FFFU + ((bits >> 16U) & 1U);
  return static_cast<std::uint16_t>((bits + rounding_bias) >> 16U);
}

float bf16_to_float(const std::uint16_t bits) noexcept {
  const std::uint32_t expanded = static_cast<std::uint32_t>(bits) << 16U;
  float value = 0.0F;
  std::memcpy(&value, &expanded, sizeof(value));
  return value;
}

float round_float_to_bf16(const float value) noexcept {
  return bf16_to_float(float_to_bf16_rne(value));
}

LogitsAnalysis analyze_bf16_argmax_in_place(
    float* const logits, const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = 0.0F;
  bool all_finite = true;
  for (std::size_t index = 0U; index < element_count; ++index) {
    const float value = round_float_to_bf16(logits[index]);
    logits[index] = value;
    all_finite = all_finite && std::isfinite(value);
    if (index == 0U || value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  return result;
}

LogitsAnalysis analyze_bf16_argmax_bits(
    const std::uint16_t* const logits,
    const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = bf16_to_float(logits[0]);
  bool all_finite = std::isfinite(maximum);
  for (std::size_t index = 1U; index < element_count; ++index) {
    const float value = bf16_to_float(logits[index]);
    all_finite = all_finite && std::isfinite(value);
    if (value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  return result;
}

namespace {

constexpr std::size_t kBf16CodeCount = 1U << 16U;

struct Bf16ExpMemoCache {
  std::array<double, kBf16CodeCount> values{};
  std::array<std::uint8_t, kBf16CodeCount> seen_stamps{};
  std::uint8_t generation = 0U;
  bool in_use = false;
};

thread_local Bf16ExpMemoCache bf16_exp_memo_cache{};

class ScopedBf16ExpMemoUse {
 public:
  explicit ScopedBf16ExpMemoUse(Bf16ExpMemoCache& cache) noexcept
      : cache_(cache) {
    cache_.in_use = true;
  }

  ScopedBf16ExpMemoUse(const ScopedBf16ExpMemoUse&) = delete;
  ScopedBf16ExpMemoUse& operator=(const ScopedBf16ExpMemoUse&) = delete;

  ~ScopedBf16ExpMemoUse() noexcept { cache_.in_use = false; }

 private:
  Bf16ExpMemoCache& cache_;
};

[[nodiscard]] LogitsAnalysis analyze_bf16_logits_bits_scalar(
    const std::uint16_t* const logits,
    const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = bf16_to_float(logits[0]);
  bool all_finite = std::isfinite(maximum);
  for (std::size_t index = 1U; index < element_count; ++index) {
    const float value = bf16_to_float(logits[index]);
    all_finite = all_finite && std::isfinite(value);
    if (value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }

  double exponential_sum = 0.0;
  for (std::size_t index = 0U; index < element_count; ++index) {
    exponential_sum +=
        std::exp(static_cast<double>(bf16_to_float(logits[index])) -
                 static_cast<double>(maximum));
  }
  const double logsumexp =
      static_cast<double>(maximum) + std::log(exponential_sum);
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  result.logsumexp = logsumexp;
  result.max_log_probability = static_cast<double>(maximum) - logsumexp;
  return result;
}

}  // namespace

LogitsAnalysis analyze_bf16_logits_in_place(
    float* const logits, const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  bool all_finite = true;
  for (std::size_t index = 0U; index < element_count; ++index) {
    logits[index] = round_float_to_bf16(logits[index]);
    all_finite = all_finite && std::isfinite(logits[index]);
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }

  std::size_t maximum_index = 0U;
  float maximum = logits[0];
  for (std::size_t index = 1U; index < element_count; ++index) {
    // Strict comparison preserves the earliest (smallest-id) tie.
    if (logits[index] > maximum) {
      maximum = logits[index];
      maximum_index = index;
    }
  }
  double exponential_sum = 0.0;
  for (std::size_t index = 0U; index < element_count; ++index) {
    exponential_sum +=
        std::exp(static_cast<double>(logits[index]) -
                 static_cast<double>(maximum));
  }
  const double logsumexp =
      static_cast<double>(maximum) + std::log(exponential_sum);
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  result.logsumexp = logsumexp;
  result.max_log_probability = static_cast<double>(maximum) - logsumexp;
  return result;
}

LogitsAnalysis analyze_bf16_logits_bits(
    const std::uint16_t* const logits,
    const std::size_t element_count) noexcept {
  LogitsAnalysis result;
  if (logits == nullptr || element_count == 0U) {
    result.status = LogitsAnalysisStatus::kInvalidArgument;
    return result;
  }

  Bf16ExpMemoCache& cache = bf16_exp_memo_cache;
  if (cache.in_use) {
    // A same-thread reentrant call must not overwrite the outer invocation's
    // stamps or memoized values.
    return analyze_bf16_logits_bits_scalar(logits, element_count);
  }
  const ScopedBf16ExpMemoUse cache_use(cache);
  std::size_t maximum_index = 0U;
  float maximum = bf16_to_float(logits[0]);
  bool all_finite = std::isfinite(maximum);
  for (std::size_t index = 1U; index < element_count; ++index) {
    const float value = bf16_to_float(logits[index]);
    all_finite = all_finite && std::isfinite(value);
    if (value > maximum) {
      maximum = value;
      maximum_index = index;
    }
  }
  if (!all_finite) {
    result.status = LogitsAnalysisStatus::kNonFinite;
    return result;
  }

  if (cache.generation == std::numeric_limits<std::uint8_t>::max()) {
    cache.seen_stamps.fill(0U);
    cache.generation = 1U;
  } else {
    cache.generation =
        static_cast<std::uint8_t>(cache.generation + 1U);
  }
  const double maximum_double = static_cast<double>(maximum);
  double exponential_sum = 0.0;
  for (std::size_t index = 0U; index < element_count; ++index) {
    const std::uint16_t code = logits[index];
    if (cache.seen_stamps[code] != cache.generation) {
      cache.seen_stamps[code] = cache.generation;
      cache.values[code] =
          std::exp(static_cast<double>(bf16_to_float(code)) -
                   maximum_double);
    }
    // Preserve the scalar oracle's original index order and double-addition
    // order. Only the repeated, deterministic exp evaluation is memoized.
    exponential_sum += cache.values[code];
  }
  const double logsumexp = maximum_double + std::log(exponential_sum);
  result.status = LogitsAnalysisStatus::kSuccess;
  result.predicted_index = maximum_index;
  result.maximum = maximum;
  result.logsumexp = logsumexp;
  result.max_log_probability = static_cast<double>(maximum) - logsumexp;
  return result;
}

bool valid_reference_linear_weight_contract(
    const LinearWeight& weight, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  return valid_linear_payload(weight, output_size, input_size);
}

model::LayerType expected_reference_layer_type(
    const std::size_t layer) noexcept {
  if (layer >= kReferenceDecoderLayerCount) {
    return model::LayerType::kInvalid;
  }
  return ((layer + 1U) % 4U) == 0U
             ? model::LayerType::kFullAttention
             : model::LayerType::kLinearAttention;
}

bool use_fused_gqa_sigmoid_gate_tile(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  return token_count != 0U &&
         fused_gqa_sigmoid_gate_prefix_token_count(first_position,
                                                   token_count) == token_count;
}

bool use_decode_gqa_splitkv(const std::size_t sequence_length) noexcept {
  return g_enable_decode_gqa_splitkv_admission &&
         sequence_length > kFusedGqaMaximumSequenceLength &&
         sequence_length <= kDecodeGqaSplitKvMaximumSequenceLength;
}

std::size_t fused_gqa_sigmoid_gate_prefix_token_count(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  if (token_count == 0U ||
      first_position >= kFusedGqaMaximumSequenceLength) {
    return 0U;
  }
  return std::min(token_count,
                  kFusedGqaMaximumSequenceLength - first_position);
}

bool use_bulk_causal_gqa_sigmoid_gate_prefill(
    const ProjectionBackend backend, const model::LayerType layer_type,
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  return backend == ProjectionBackend::kSm87WeightOnly &&
         layer_type == model::LayerType::kFullAttention &&
         token_count >= 2U &&
         token_count <= kMaximumRequestPrefillChunkSize &&
         first_position <=
             kBulkCausalGqaMaximumSequenceLength - token_count;
}

bool use_qk_rope_tile(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  if (token_count == 0U || token_count > kQkRopeTileMaximumTokens ||
      first_position >
          std::numeric_limits<std::size_t>::max() - token_count) {
    return false;
  }
  return first_position + token_count <=
         std::numeric_limits<std::size_t>::max() /
             (kRopePairs * sizeof(float));
}

bool use_full_attention_preprocess_tile(
    const std::size_t first_position,
    const std::size_t token_count) noexcept {
  if (token_count == 0U ||
      token_count > kFullAttentionPreprocessMaximumTokens ||
      first_position >
          std::numeric_limits<std::size_t>::max() - token_count) {
    return false;
  }
  return first_position + token_count <=
         std::numeric_limits<std::size_t>::max() /
             (kRopePairs * sizeof(float));
}

bool use_m32_prefill_residual_rms_fusion(
    const std::size_t token_count,
    const std::size_t hidden_size) noexcept {
  return prefill_residual_rms_m32_schedule(token_count, hidden_size).valid();
}

bool use_fp8_marlin_prefill_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count) noexcept {
  const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  if (backend != ProjectionBackend::kSm87WeightOnly || token_count < 2U ||
      token_count > kMaximumRequestPrefillChunkSize || selected == nullptr) {
    return false;
  }
  const bool supported_shape =
      (selected->output_size == 10'240U && selected->input_size == 5'120U) ||
      (selected->output_size == 6'144U && selected->input_size == 5'120U) ||
      (selected->output_size == 5'120U && selected->input_size == 6'144U) ||
      (selected->output_size == 12'288U && selected->input_size == 5'120U) ||
      (selected->output_size == 1'024U && selected->input_size == 5'120U);
  return supported_shape && aligned(selected->prefill_marlin_weight, 16U) &&
         aligned(selected->prefill_marlin_scales, 16U) &&
         aligned(input, 16U) && aligned(output, 16U);
}

bool use_fp8_whole_chunk_prefill_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kQkvRows = 10'240U;
  constexpr std::size_t kZRows = 6'144U;
  constexpr std::size_t kHiddenSize = 5'120U;
  constexpr std::size_t kFullQueryGateRows = 12'288U;
  constexpr std::size_t kFullKvRows = 1'024U;
  const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      (token_count != 256U && token_count != 512U) ||
      selected == nullptr) {
    return false;
  }
  const bool qkv_shape = selected->output_size == kQkvRows &&
                         selected->input_size == kHiddenSize;
  const bool z_shape = selected->output_size == kZRows &&
                       selected->input_size == kHiddenSize;
  const bool attention_output_shape =
      selected->output_size == kHiddenSize &&
      selected->input_size == kZRows;
  const bool full_query_shape =
      selected->output_size == kFullQueryGateRows &&
      selected->input_size == kHiddenSize;
  const bool full_kv_shape = selected->output_size == kFullKvRows &&
                             selected->input_size == kHiddenSize;
  return (qkv_shape || z_shape || attention_output_shape ||
          full_query_shape || full_kv_shape) &&
         aligned(selected->weight, 16U) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(output, alignof(std::uint16_t));
}

bool use_fp8_m64_prefill_attention_output_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kRows = 5'120U;
  constexpr std::size_t kColumns = 6'144U;
  const auto* const selected = std::get_if<Fp8LinearWeight>(&weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  return backend == ProjectionBackend::kSm87WeightOnly &&
         token_count == kMaximumProjectionTileTokenCount &&
         selected != nullptr && selected->output_size == kRows &&
         selected->input_size == kColumns &&
         aligned(selected->weight, 16U) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(output, alignof(std::uint16_t));
}

bool use_nvfp4_whole_chunk_prefill_down_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kRows = 5'120U;
  constexpr std::size_t kColumns = 17'408U;
  const auto* const selected = std::get_if<NvFp4LinearWeight>(&weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  return backend == ProjectionBackend::kSm87WeightOnly &&
         (token_count == 256U || token_count == 512U) &&
         selected != nullptr && selected->output_size == kRows &&
         selected->input_size == kColumns &&
         aligned(selected->packed_weight, 16U) &&
         aligned(selected->block_scale, alignof(std::uint16_t)) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(output, alignof(std::uint16_t));
}

bool use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* const input,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kRows = 17'408U;
  constexpr std::size_t kColumns = 5'120U;
  const auto* const gate = std::get_if<NvFp4LinearWeight>(&gate_weight);
  const auto* const up = std::get_if<NvFp4LinearWeight>(&up_weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  if (token_count != 256U && token_count != 512U) {
    return false;
  }
  const std::size_t output_bytes =
      token_count * kRows * sizeof(std::uint16_t);
  return backend == ProjectionBackend::kSm87WeightOnly &&
         gate != nullptr && up != nullptr &&
         gate->output_size == kRows && gate->input_size == kColumns &&
         up->output_size == kRows && up->input_size == kColumns &&
         aligned(gate->packed_weight, 16U) &&
         aligned(up->packed_weight, 16U) &&
         aligned(gate->block_scale, alignof(std::uint16_t)) &&
         aligned(up->block_scale, alignof(std::uint16_t)) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(gate_output, alignof(std::uint16_t)) &&
         aligned(up_output, alignof(std::uint16_t)) &&
         byte_ranges_are_disjoint(gate_output, output_bytes, up_output,
                                  output_bytes);
}

bool use_nvfp4_m32_prefill_gate_up_dual_stream(
    const ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* const input,
    std::uint16_t* const gate_output, std::uint16_t* const up_output,
    const std::size_t token_count) noexcept {
  constexpr std::size_t kRows = 17'408U;
  constexpr std::size_t kColumns = 5'120U;
  const auto* const gate = std::get_if<NvFp4LinearWeight>(&gate_weight);
  const auto* const up = std::get_if<NvFp4LinearWeight>(&up_weight);
  const auto aligned = [](const void* const pointer,
                          const std::size_t alignment) noexcept {
    return pointer != nullptr &&
           (reinterpret_cast<std::uintptr_t>(pointer) % alignment) == 0U;
  };
  if (token_count != kProductionProjectionSubtileTokens &&
      token_count != 2U * kProductionProjectionSubtileTokens) {
    return false;
  }
  const std::size_t output_bytes =
      token_count * kRows * sizeof(std::uint16_t);
  return backend == ProjectionBackend::kSm87WeightOnly &&
         gate != nullptr && up != nullptr &&
         gate->output_size == kRows && gate->input_size == kColumns &&
         up->output_size == kRows && up->input_size == kColumns &&
         aligned(gate->packed_weight, 16U) &&
         aligned(up->packed_weight, 16U) &&
         aligned(gate->block_scale, alignof(std::uint16_t)) &&
         aligned(up->block_scale, alignof(std::uint16_t)) &&
         aligned(input, alignof(std::uint64_t)) &&
         aligned(gate_output, alignof(std::uint16_t)) &&
         aligned(up_output, alignof(std::uint16_t)) &&
         byte_ranges_are_disjoint(gate_output, output_bytes, up_output,
                                  output_bytes);
}

ReferenceRunnerError validate_reference_workspace_plan(
    const RequestMemoryPlan& plan) noexcept {
  if (plan.batch_size != 1U || plan.prefill_chunk_size == 0U ||
      plan.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      plan.max_sequence_length == 0U ||
      plan.max_sequence_length > kAbsoluteRequestMaxSequenceLength ||
      plan.hidden_bf16.size() != 3U ||
      plan.projection_bf16.size() != 4U) {
    return ReferenceRunnerError::kInvalidRequestState;
  }
  const auto bf16_region_at_least = [](const RequestRegion& region,
                                       const std::uint64_t elements) noexcept {
    return region.element_size_bytes == sizeof(std::uint16_t) &&
           region.element_capacity >= elements &&
           region.byte_size >= elements * sizeof(std::uint16_t);
  };
  const auto fp32_region_at_least = [](const RequestRegion& region,
                                       const std::uint64_t elements) noexcept {
    return region.element_size_bytes == sizeof(float) &&
           region.element_capacity >= elements &&
           region.byte_size >= elements * sizeof(float);
  };

  for (const RequestRegion& region : plan.hidden_bf16) {
    if (!bf16_region_at_least(
            region, static_cast<std::uint64_t>(plan.prefill_chunk_size) *
                        kReferenceHiddenSize)) {
      return ReferenceRunnerError::kInvalidRequestState;
    }
  }
  for (const RequestRegion& region : plan.projection_bf16) {
    if (!bf16_region_at_least(
            region, static_cast<std::uint64_t>(plan.prefill_chunk_size) *
                        kReferenceIntermediateSize)) {
      return ReferenceRunnerError::kInvalidRequestState;
    }
  }
  if (!bf16_region_at_least(
          plan.linear_a_bf16,
          static_cast<std::uint64_t>(plan.prefill_chunk_size) *
              kLinearScalarElements) ||
      !bf16_region_at_least(
          plan.linear_b_bf16,
          static_cast<std::uint64_t>(plan.prefill_chunk_size) *
              kLinearScalarElements) ||
      !fp32_region_at_least(plan.fp32_scratch,
                            kReferenceVocabularySize) ||
      plan.gqa_probability_scratch.arena_offset !=
          plan.fp32_scratch.arena_offset ||
      !fp32_region_at_least(
          plan.gqa_probability_scratch,
          static_cast<std::uint64_t>(kFullQueryHeads) *
              plan.max_sequence_length) ||
      !fp32_region_at_least(
          plan.rope_cos_fp32,
          static_cast<std::uint64_t>(kRopePairs) *
              plan.max_sequence_length) ||
      !fp32_region_at_least(
          plan.rope_sin_fp32,
          static_cast<std::uint64_t>(kRopePairs) *
              plan.max_sequence_length) ||
      !bf16_region_at_least(
          plan.conv_state,
          static_cast<std::uint64_t>(kRequestLinearLayerCount) *
              kGdnQkvChannels * kGdnConvHistoryWidth) ||
      !bf16_region_at_least(
          plan.gdn_state,
          static_cast<std::uint64_t>(kRequestLinearLayerCount) *
              kGdnStateElements)) {
    return ReferenceRunnerError::kInvalidRequestState;
  }

  for (std::size_t slot = 0U; slot < kRequestFullLayerCount; ++slot) {
    const std::uint64_t required =
        static_cast<std::uint64_t>(plan.max_sequence_length) *
        kFullKvElements;
    if (!bf16_region_at_least(plan.key_cache[slot], required) ||
        !bf16_region_at_least(plan.value_cache[slot], required)) {
      return ReferenceRunnerError::kInvalidRequestState;
    }
  }

  std::size_t linear_slot = 0U;
  std::size_t full_slot = 0U;
  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const model::LayerType expected = expected_reference_layer_type(layer);
    const RequestLayerSlot& actual = plan.layers[layer];
    const std::size_t expected_slot =
        expected == model::LayerType::kFullAttention ? full_slot++
                                                     : linear_slot++;
    if (actual.type != expected || actual.slot != expected_slot) {
      return ReferenceRunnerError::kInvalidLayerSchedule;
    }
  }
  if (linear_slot != kRequestLinearLayerCount ||
      full_slot != kRequestFullLayerCount) {
    return ReferenceRunnerError::kInvalidLayerSchedule;
  }
  return ReferenceRunnerError::kNone;
}

}  // namespace reference_runner_detail

namespace {

struct ExactPrefillProjectionWorkspace {
  float* reduction = nullptr;
  std::size_t reduction_elements = 0U;
  std::int32_t* locks = nullptr;
  std::size_t lock_bytes = 0U;
};

[[nodiscard]] bool valid_exact_prefill_projection_workspace(
    const ExactPrefillProjectionWorkspace& workspace) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  return workspace.reduction != nullptr && workspace.locks != nullptr &&
         workspace.reduction_elements >=
             kernels::kSm87Fp8MarlinReductionElements &&
         workspace.lock_bytes >= kernels::kSm87Fp8MarlinLockBytes;
#else
  (void)workspace;
  return false;
#endif
}

[[nodiscard]] int clear_exact_prefill_projection_locks(
    const ExactPrefillProjectionWorkspace& workspace,
    void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  if (!valid_exact_prefill_projection_workspace(workspace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(cudaMemsetAsync(
      workspace.locks, 0, kernels::kSm87Fp8MarlinLockBytes,
      reinterpret_cast<cudaStream_t>(cuda_stream)));
#else
  (void)workspace;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

[[nodiscard]] int launch_exact_contract_fp8_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const LayerMajorPrefillArithmeticSpanLedger& ledger,
    const ExactPrefillProjectionWorkspace& workspace,
    void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
  if (!is_valid_layer_major_prefill_arithmetic_contract(
          kLayerMajorPrefillExactArithmeticContract) ||
      !is_valid_layer_major_prefill_arithmetic_span_ledger(ledger) ||
      !valid_exact_prefill_projection_workspace(workspace) || fp8 == nullptr ||
      !kernels::sm87_fp8_marlin_supports_shape(fp8->output_size,
                                               fp8->input_size)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  for (std::size_t index = 0U; index < ledger.span_count; ++index) {
    const LayerMajorPrefillArithmeticSpan& span = ledger.spans[index];
    const std::size_t offset = span.token_offset;
    if (!reference_runner_detail::use_fp8_marlin_prefill_projection(
            backend, weight, input + offset * fp8->input_size,
            output + offset * fp8->output_size, span.token_count)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    int status = clear_exact_prefill_projection_locks(workspace, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = kernels::launch_sm87_fp8_marlin_projection_cuda(
        input + offset * fp8->input_size, fp8->prefill_marlin_weight,
        fp8->prefill_marlin_scales, span.token_count, fp8->output_size,
        fp8->input_size, output + offset * fp8->output_size,
        workspace.reduction, workspace.locks, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    ++g_fp8_marlin_prefill_admission_hits;
  }
  return static_cast<int>(cudaSuccess);
#else
  (void)backend;
  (void)weight;
  (void)input;
  (void)output;
  (void)ledger;
  (void)workspace;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

[[nodiscard]] int launch_segmented_panel_fp8_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count,
    const ExactPrefillProjectionWorkspace& workspace,
    std::size_t& physical_launches, void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
  if (backend != ProjectionBackend::kSm87WeightOnly || fp8 == nullptr ||
      !kernels::sm87_fp8_marlin_supports_operator_panel_token_count(
          token_count) ||
      !kernels::sm87_fp8_marlin_supports_shape(fp8->output_size,
                                               fp8->input_size) ||
      !valid_exact_prefill_projection_workspace(workspace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const kernels::Sm87Fp8MarlinExecutionPlan plan =
      kernels::sm87_fp8_marlin_execution_plan(token_count, fp8->output_size);
  if (plan.launch_count == 0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int status = clear_exact_prefill_projection_locks(workspace, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = kernels::launch_sm87_fp8_marlin_projection_cuda(
      input, fp8->prefill_marlin_weight, fp8->prefill_marlin_scales,
      token_count, fp8->output_size, fp8->input_size, output,
      workspace.reduction, workspace.locks, cuda_stream);
  if (status == static_cast<int>(cudaSuccess)) {
    physical_launches += plan.launch_count;
    ++g_fp8_marlin_prefill_admission_hits;
  }
  return status;
#else
  (void)backend;
  (void)weight;
  (void)input;
  (void)output;
  (void)token_count;
  (void)workspace;
  (void)physical_launches;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

struct ExactMarlinOperatorPanelPlan {
  std::size_t aligned_prefix_tokens = 0U;
  std::size_t aligned_prefix_span_count = 0U;

  [[nodiscard]] constexpr bool valid(
      const LayerMajorPrefillArithmeticSpanLedger& ledger) const noexcept {
    return ledger.token_count == kLayerMajorPrefillOperatorPanelTokens &&
           aligned_prefix_tokens == kLayerMajorPrefillOperatorPanelTokens &&
           aligned_prefix_span_count == ledger.span_count &&
           aligned_prefix_span_count != 0U;
  }
};

// The admitted M8192 direction-screen shape consists entirely of consecutive
// oracle C512 spans. Partial panels retain every exact wrapper call and masked
// specialization position; the caller rejects this bulk plan for those shapes.
[[nodiscard]] constexpr ExactMarlinOperatorPanelPlan
exact_marlin_operator_panel_plan(
    const LayerMajorPrefillArithmeticSpanLedger& ledger) noexcept {
  if (!is_valid_layer_major_prefill_arithmetic_span_ledger(ledger)) {
    return {};
  }
  ExactMarlinOperatorPanelPlan plan;
  while (plan.aligned_prefix_span_count < ledger.span_count &&
         ledger.spans[plan.aligned_prefix_span_count].token_count ==
             kPrefillPhysicalSegmentMaximumTokens) {
    plan.aligned_prefix_tokens += kPrefillPhysicalSegmentMaximumTokens;
    ++plan.aligned_prefix_span_count;
  }
  return plan;
}

[[maybe_unused, nodiscard]] int launch_native_large_m_fp8_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count,
    const LayerMajorPrefillArithmeticSpanLedger& arithmetic_ledger,
    const ExactPrefillProjectionWorkspace& workspace,
    std::size_t& physical_launches, void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
  const ExactMarlinOperatorPanelPlan plan =
      exact_marlin_operator_panel_plan(arithmetic_ledger);
  if (backend != ProjectionBackend::kSm87WeightOnly || fp8 == nullptr ||
      fp8->prefill_marlin_weight == nullptr ||
      fp8->prefill_marlin_scales == nullptr || input == nullptr ||
      output == nullptr || arithmetic_ledger.token_count != token_count ||
      !valid_exact_prefill_projection_workspace(workspace) ||
      !kernels::sm87_fp8_marlin_supports_shape(fp8->output_size,
                                               fp8->input_size)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  // Only real-model M8192 has passed whole-state bitwise admission. Preserve
  // the exact authenticated ledger for every partial panel instead of
  // inferring correctness from alignment alone.
  if (token_count != kLayerMajorPrefillOperatorPanelTokens) {
    std::size_t launches = 0U;
    for (std::size_t index = 0U; index < arithmetic_ledger.span_count;
         ++index) {
      const auto plan = kernels::sm87_fp8_marlin_execution_plan(
          arithmetic_ledger.spans[index].token_count, fp8->output_size);
      if (plan.launch_count == 0U) {
        return static_cast<int>(cudaErrorInvalidValue);
      }
      launches += plan.launch_count;
    }
    const int status = launch_exact_contract_fp8_projection(
        backend, weight, input, output, arithmetic_ledger, workspace,
        cuda_stream);
    if (status == static_cast<int>(cudaSuccess)) {
      physical_launches += launches;
    }
    return status;
  }
  if (!plan.valid(arithmetic_ledger)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int status = clear_exact_prefill_projection_locks(workspace, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status =
      kernels::launch_sm87_fp8_marlin_projection_aligned_operator_panel_cuda(
          input, fp8->prefill_marlin_weight, fp8->prefill_marlin_scales,
          kLayerMajorPrefillOperatorPanelTokens, fp8->output_size,
          fp8->input_size, output, workspace.reduction, workspace.locks,
          cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  ++physical_launches;
  ++g_fp8_marlin_prefill_admission_hits;
  return static_cast<int>(cudaSuccess);
#else
  (void)backend;
  (void)weight;
  (void)input;
  (void)output;
  (void)token_count;
  (void)arithmetic_ledger;
  (void)workspace;
  (void)physical_launches;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

// Exact-P40000 whole-core projection primitive. Each logical fill/drain
// projection owns exactly one M8000, M64-aligned Marlin grid. This deliberately does
// not call the historical shape-aware host segmenter and has no partial-panel
// or exact-oracle fallback: any geometry or authenticated-sidecar mismatch
// rejects the whole architecture route before a different kernel can run.
[[maybe_unused, nodiscard]] int
launch_prompt_wide_p40_fp8_projection(
    const ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* const input, std::uint16_t* const output,
    const std::size_t token_count,
    const ExactPrefillProjectionWorkspace& workspace,
    std::size_t& physical_launches, void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION) && \
    defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
  if (backend != ProjectionBackend::kSm87WeightOnly || fp8 == nullptr ||
      input == nullptr || output == nullptr ||
      token_count != kPromptWideP40WholeCorePanelTokens ||
      fp8->prefill_marlin_weight == nullptr ||
      fp8->prefill_marlin_scales == nullptr ||
      !kernels::sm87_fp8_marlin_supports_shape(fp8->output_size,
                                               fp8->input_size) ||
      !kernels::sm87_fp8_marlin_supports_operator_panel_token_count(
          token_count) ||
      !valid_exact_prefill_projection_workspace(workspace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int status = clear_exact_prefill_projection_locks(workspace, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status =
      kernels::launch_sm87_fp8_marlin_projection_aligned_operator_panel_cuda(
          input, fp8->prefill_marlin_weight, fp8->prefill_marlin_scales,
          token_count, fp8->output_size, fp8->input_size, output,
          workspace.reduction, workspace.locks, cuda_stream);
  if (status == static_cast<int>(cudaSuccess)) {
    ++physical_launches;
    ++g_fp8_marlin_prefill_admission_hits;
  }
  return status;
#else
  (void)backend;
  (void)weight;
  (void)input;
  (void)output;
  (void)token_count;
  (void)workspace;
  (void)physical_launches;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

// Exact-P40000 grouped FP8 projection primitive for the projection-reset
// route.  One physical persistent launch owns the complete M dimension and
// all same-input partitions; it never delegates to the M8000 Marlin path.
[[maybe_unused, nodiscard]] int
launch_p40_projection_reset_fp8_group(
    const ProjectionBackend backend,
    const LinearWeight* const* const group_weights,
    std::uint16_t* const* const group_outputs,
    const std::size_t group_size,
    const std::uint16_t* const input,
    std::size_t& physical_launches, void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_P40_PROJECTION_RESET_ADMISSION)
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      group_weights == nullptr || group_outputs == nullptr ||
      group_size == 0U || group_size > 3U || input == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  std::array<kernels::Sm87Fp8PrefillSupermatrixPartition, 3U> partitions{};
  std::size_t columns = 0U;
  for (std::size_t index = 0U; index < group_size; ++index) {
    if (group_weights[index] == nullptr || group_outputs[index] == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    const auto* const fp8 =
        std::get_if<Fp8LinearWeight>(group_weights[index]);
    if (fp8 == nullptr || fp8->prefill_supermatrix_sidecar == nullptr ||
        (index != 0U && fp8->input_size != columns)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    columns = fp8->input_size;
    partitions[index] = {fp8->prefill_supermatrix_sidecar,
                         fp8->weight_scale, fp8->output_size,
                         group_outputs[index]};
  }
  const int status =
      kernels::launch_sm87_p40_projection_reset_fp8_supermatrix_cuda(
          partitions.data(), group_size, input,
          kPromptWideP40WholeCorePromptTokens, columns, cuda_stream);
  if (status == static_cast<int>(cudaSuccess)) {
    ++physical_launches;
  }
  return status;
#else
  (void)backend;
  (void)group_weights;
  (void)group_outputs;
  (void)group_size;
  (void)input;
  (void)physical_launches;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

// Exact-P40000 grouped FP8 primitive for the AOT packed route. All logical
// projections in one group must name the same authenticated physical
// artifact; partition order is fixed by the public packed-layout ABI.
[[maybe_unused, nodiscard]] int launch_p40_packed_fp8_group(
    const ProjectionBackend backend,
    const LinearWeight* const* const group_weights,
    std::uint16_t* const* const group_outputs,
    const std::size_t group_size, const std::uint16_t* const input,
    std::size_t& physical_launches, void* const cuda_stream) noexcept {
#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
  if (backend != ProjectionBackend::kSm87WeightOnly ||
      group_weights == nullptr || group_outputs == nullptr ||
      group_size == 0U || group_size >
          kernels::kSm87P40PackedProjectionMaximumSources ||
      input == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto* const first =
      std::get_if<Fp8LinearWeight>(group_weights[0U]);
  if (first == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const kernels::Sm87P40PackedProjectionDeviceView& artifact =
      first->prefill_p40_packed_artifact;
  kernels::Sm87P40PackedProjectionRole expected_role =
      kernels::Sm87P40PackedProjectionRole::kInvalid;
  if (group_size == 2U && first->output_size == 10'240U &&
      first->input_size == 5'120U) {
    expected_role =
        kernels::Sm87P40PackedProjectionRole::kFp8LinearQkvZ;
  } else if (group_size == 3U && first->output_size == 12'288U &&
             first->input_size == 5'120U) {
    expected_role = kernels::Sm87P40PackedProjectionRole::kFp8FullQkv;
  } else if (group_size == 1U && first->output_size == 5'120U &&
             first->input_size == 6'144U) {
    expected_role =
        kernels::Sm87P40PackedProjectionRole::kFp8AttentionOutput;
  }
  const kernels::Sm87P40PackedProjectionPlan expected_plan =
      kernels::sm87_p40_packed_projection_plan(expected_role);
  if (expected_role ==
          kernels::Sm87P40PackedProjectionRole::kInvalid ||
      artifact.payload == nullptr || artifact.artifact_identity == 0U ||
      artifact.role != expected_role ||
      artifact.tactic != expected_plan.tactic ||
      artifact.payload_bytes != expected_plan.payload_bytes ||
      artifact.source_count != group_size) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  std::array<std::uint16_t*,
             kernels::kSm87P40PackedProjectionMaximumSources>
      outputs{};
  for (std::size_t index = 0U; index < group_size; ++index) {
    const auto* const fp8 =
        group_weights[index] == nullptr
            ? nullptr
            : std::get_if<Fp8LinearWeight>(group_weights[index]);
    const auto& partition = expected_plan.partitions[index];
    if (fp8 == nullptr || group_outputs[index] == nullptr ||
        fp8->output_size != partition.output_features ||
        fp8->input_size != partition.input_features ||
        fp8->prefill_p40_packed_artifact.payload != artifact.payload ||
        fp8->prefill_p40_packed_artifact.payload_bytes !=
            artifact.payload_bytes ||
        fp8->prefill_p40_packed_artifact.artifact_identity !=
            artifact.artifact_identity ||
        fp8->prefill_p40_packed_artifact.role != artifact.role ||
        fp8->prefill_p40_packed_artifact.tactic != artifact.tactic ||
        fp8->prefill_p40_packed_artifact.source_count !=
            artifact.source_count ||
        fp8->prefill_p40_packed_artifact.scalar_scales !=
            artifact.scalar_scales ||
        fp8->weight_scale != artifact.scalar_scales[index]) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    outputs[index] = group_outputs[index];
  }
  const int status = kernels::launch_sm87_p40_packed_fp8_cuda(
      input, artifact, kernels::kSm87P40PackedProjectionTokens, outputs,
      cuda_stream);
  if (status == static_cast<int>(cudaSuccess)) {
    ++physical_launches;
  }
  return status;
#else
  (void)backend;
  (void)group_weights;
  (void)group_outputs;
  (void)group_size;
  (void)input;
  (void)physical_launches;
  (void)cuda_stream;
  return static_cast<int>(cudaErrorNotSupported);
#endif
}

[[nodiscard]] int launch_exact_contract_bf16_projection_pair(
    const ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight, const std::uint16_t* const input,
    std::uint16_t* const first_output, std::uint16_t* const second_output,
    const LayerMajorPrefillArithmeticSpanLedger& ledger, float* const scratch,
    const std::size_t scratch_elements, void* const cuda_stream) noexcept {
  if (!is_valid_layer_major_prefill_arithmetic_contract(
          kLayerMajorPrefillExactArithmeticContract) ||
      !is_valid_layer_major_prefill_arithmetic_span_ledger(ledger) ||
      input == nullptr || first_output == nullptr || second_output == nullptr ||
      scratch == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const std::size_t columns = linear_input_size(first_weight);
  const std::size_t first_rows = linear_output_size(first_weight);
  const std::size_t second_rows = linear_output_size(second_weight);
  if (columns == 0U || columns != linear_input_size(second_weight) ||
      first_rows == 0U || second_rows == 0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  for (std::size_t span_index = 0U; span_index < ledger.span_count;
       ++span_index) {
    const LayerMajorPrefillArithmeticSpan& span = ledger.spans[span_index];
    for (std::size_t local_offset = 0U; local_offset < span.token_count;
         local_offset += kProductionProjectionSubtileTokens) {
      const std::size_t subtile_tokens =
          std::min<std::size_t>(span.token_count - local_offset,
                                kProductionProjectionSubtileTokens);
      const std::size_t token_offset = span.token_offset + local_offset;
      const int status = launch_projection_pair_tile_to_bf16_cuda(
          backend, first_weight, second_weight,
          input + token_offset * columns, subtile_tokens, scratch,
          scratch_elements, first_output + token_offset * first_rows,
          second_output + token_offset * second_rows, cuda_stream);
      if (status != static_cast<int>(cudaSuccess)) {
        return status;
      }
    }
  }
  return static_cast<int>(cudaSuccess);
}

#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
[[nodiscard]] bool valid_exact_contract_nvfp4_mlp_weights(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down) noexcept {
  const auto aligned_16 = [](const void* const pointer) noexcept {
    return pointer != nullptr &&
           reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
  };
  return gate.output_size == kReferenceIntermediateSize &&
         gate.input_size == kReferenceHiddenSize &&
         up.output_size == kReferenceIntermediateSize &&
         up.input_size == kReferenceHiddenSize &&
         down.output_size == kReferenceHiddenSize &&
         down.input_size == kReferenceIntermediateSize &&
         gate.prefill_marlin_gate_up_layout ==
             NvFp4MarlinGateUpLayout::kCanonicalGateThenUp &&
         up.prefill_marlin_gate_up_layout ==
             gate.prefill_marlin_gate_up_layout &&
         aligned_16(gate.prefill_marlin_weight) &&
         aligned_16(gate.prefill_marlin_scales) &&
         gate.prefill_marlin_global_scale != nullptr &&
         gate.prefill_marlin_weight == up.prefill_marlin_weight &&
         gate.prefill_marlin_scales == up.prefill_marlin_scales &&
         gate.prefill_marlin_global_scale ==
             up.prefill_marlin_global_scale &&
         aligned_16(down.prefill_marlin_weight) &&
         aligned_16(down.prefill_marlin_scales) &&
         down.prefill_marlin_global_scale != nullptr;
}

[[nodiscard]] bool valid_native_large_m_nvfp4_mlp_weights(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down) noexcept {
  return valid_exact_contract_nvfp4_mlp_weights(gate, up, down);
}

#if defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
[[nodiscard]] bool valid_persistent_p40_nvfp4_mlp_weights(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down) noexcept {
  const auto aligned_16 = [](const void* const pointer) noexcept {
    return pointer != nullptr &&
           reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
  };
  return gate.output_size == kReferenceIntermediateSize &&
         gate.input_size == kReferenceHiddenSize &&
         up.output_size == kReferenceIntermediateSize &&
         up.input_size == kReferenceHiddenSize &&
         down.output_size == kReferenceHiddenSize &&
         down.input_size == kReferenceIntermediateSize &&
         gate.prefill_marlin_gate_up_layout ==
             NvFp4MarlinGateUpLayout::kInterleavedGateUp &&
         up.prefill_marlin_gate_up_layout ==
             gate.prefill_marlin_gate_up_layout &&
         aligned_16(gate.prefill_marlin_weight) &&
         aligned_16(gate.prefill_marlin_scales) &&
         gate.prefill_marlin_global_scale != nullptr &&
         gate.prefill_marlin_weight == up.prefill_marlin_weight &&
         gate.prefill_marlin_scales == up.prefill_marlin_scales &&
         gate.prefill_marlin_global_scale ==
             up.prefill_marlin_global_scale &&
         aligned_16(down.prefill_marlin_weight) &&
         aligned_16(down.prefill_marlin_scales) &&
         down.prefill_marlin_global_scale != nullptr;
}
#endif

#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
[[nodiscard]] bool valid_p40_packed_nvfp4_mlp_weights(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down) noexcept {
  const auto& gate_artifact = gate.prefill_p40_packed_artifact;
  const auto& up_artifact = up.prefill_p40_packed_artifact;
  const auto& down_artifact = down.prefill_p40_packed_artifact;
  const auto gate_plan = kernels::sm87_p40_packed_projection_plan(
      kernels::Sm87P40PackedProjectionRole::kNvFp4GateUp);
  const auto down_plan = kernels::sm87_p40_packed_projection_plan(
      kernels::Sm87P40PackedProjectionRole::kNvFp4Down);
  const bool shared_gate_up =
      gate_artifact.payload != nullptr &&
      gate_artifact.payload == up_artifact.payload &&
      gate_artifact.payload_bytes == gate_plan.payload_bytes &&
      gate_artifact.payload_bytes == up_artifact.payload_bytes &&
      gate_artifact.artifact_identity != 0U &&
      gate_artifact.artifact_identity == up_artifact.artifact_identity &&
      gate_artifact.role ==
          kernels::Sm87P40PackedProjectionRole::kNvFp4GateUp &&
      gate_artifact.role == up_artifact.role &&
      gate_artifact.tactic == gate_plan.tactic &&
      gate_artifact.tactic == up_artifact.tactic &&
      gate_artifact.source_count == 2U &&
      gate_artifact.source_count == up_artifact.source_count &&
      gate_artifact.scalar_scales == up_artifact.scalar_scales &&
      gate_artifact.scalar_scales[0U] == gate.weight_scale_2 &&
      gate_artifact.scalar_scales[1U] == up.weight_scale_2;
  const bool valid_down =
      down_artifact.payload != nullptr &&
      down_artifact.payload_bytes == down_plan.payload_bytes &&
      down_artifact.artifact_identity != 0U &&
      down_artifact.role ==
          kernels::Sm87P40PackedProjectionRole::kNvFp4Down &&
      down_artifact.tactic == down_plan.tactic &&
      down_artifact.source_count == 1U &&
      down_artifact.scalar_scales[0U] == down.weight_scale_2;
  return gate.output_size == kReferenceIntermediateSize &&
         gate.input_size == kReferenceHiddenSize &&
         up.output_size == kReferenceIntermediateSize &&
         up.input_size == kReferenceHiddenSize &&
         down.output_size == kReferenceHiddenSize &&
         down.input_size == kReferenceIntermediateSize && shared_gate_up &&
         valid_down;
}
#endif

[[nodiscard]] int launch_exact_contract_nvfp4_mlp(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down, const std::uint16_t* const input,
    std::uint16_t* const merged_gate_up, std::uint16_t* const activated,
    std::uint16_t* const output,
    const LayerMajorPrefillArithmeticSpanLedger& ledger,
    const ExactPrefillProjectionWorkspace& gate_up_workspace,
    const ExactPrefillProjectionWorkspace& down_workspace,
    void* const cuda_stream) noexcept {
  if (!is_valid_layer_major_prefill_arithmetic_contract(
          kLayerMajorPrefillExactArithmeticContract) ||
      !is_valid_layer_major_prefill_arithmetic_span_ledger(ledger) ||
      !valid_exact_contract_nvfp4_mlp_weights(gate, up, down) ||
      input == nullptr || merged_gate_up == nullptr || activated == nullptr ||
      output == nullptr ||
      !valid_exact_prefill_projection_workspace(gate_up_workspace) ||
      !valid_exact_prefill_projection_workspace(down_workspace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  constexpr std::size_t kMergedGateUpElements =
      2U * kReferenceIntermediateSize;
  for (std::size_t index = 0U; index < ledger.span_count; ++index) {
    const LayerMajorPrefillArithmeticSpan& span = ledger.spans[index];
    if (!kernels::sm87_nvfp4_marlin_supports_token_count(span.token_count)) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
    const std::size_t token_offset = span.token_offset;
    int status =
        clear_exact_prefill_projection_locks(gate_up_workspace, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = kernels::launch_sm87_nvfp4_marlin_gate_up_cuda(
        input + token_offset * kReferenceHiddenSize,
        gate.prefill_marlin_weight, gate.prefill_marlin_scales,
        gate.prefill_marlin_global_scale, span.token_count,
        merged_gate_up + token_offset * kMergedGateUpElements,
        gate_up_workspace.reduction, gate_up_workspace.locks, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    status = kernels::launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
        merged_gate_up + token_offset * kMergedGateUpElements,
        span.token_count,
        activated + token_offset * kReferenceIntermediateSize, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
    // The Gate+Up Marlin scheduler returns its locks to zero on this ordered
    // stream. Preserve the oracle sequence exactly: Down reuses those locks
    // without an intervening host memset. Its typed reduction arena remains
    // separate because the panel layout already owns that phase workspace.
    status = kernels::launch_sm87_nvfp4_marlin_down_cuda(
        activated + token_offset * kReferenceIntermediateSize,
        down.prefill_marlin_weight, down.prefill_marlin_scales,
        down.prefill_marlin_global_scale, span.token_count,
        output + token_offset * kReferenceHiddenSize,
        down_workspace.reduction, gate_up_workspace.locks, cuda_stream);
    if (status != static_cast<int>(cudaSuccess)) {
      return status;
    }
  }
  return static_cast<int>(cudaSuccess);
}

[[nodiscard]] int launch_segmented_panel_nvfp4_mlp(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down, const std::uint16_t* const input,
    std::uint16_t* const merged_gate_up, std::uint16_t* const activated,
    std::uint16_t* const output, const std::size_t token_count,
    const ExactPrefillProjectionWorkspace& gate_up_workspace,
    const ExactPrefillProjectionWorkspace& down_workspace,
    std::size_t& logical_projection_hits, std::size_t& physical_launches,
    void* const cuda_stream) noexcept {
  if (!valid_exact_contract_nvfp4_mlp_weights(gate, up, down) ||
      !kernels::sm87_nvfp4_marlin_supports_operator_panel_token_count(
          token_count) ||
      input == nullptr || merged_gate_up == nullptr || activated == nullptr ||
      output == nullptr ||
      !valid_exact_prefill_projection_workspace(gate_up_workspace) ||
      !valid_exact_prefill_projection_workspace(down_workspace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const kernels::Sm87NvFp4MarlinExecutionPlan plan =
      kernels::sm87_nvfp4_marlin_execution_plan(token_count);
  if (plan.launch_count == 0U) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int status =
      clear_exact_prefill_projection_locks(gate_up_workspace, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = kernels::launch_sm87_nvfp4_marlin_gate_up_cuda(
      input, gate.prefill_marlin_weight, gate.prefill_marlin_scales,
      gate.prefill_marlin_global_scale, token_count, merged_gate_up,
      gate_up_workspace.reduction, gate_up_workspace.locks, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  ++logical_projection_hits;
  physical_launches += plan.launch_count;
  status = kernels::launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
      merged_gate_up, token_count, activated, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = clear_exact_prefill_projection_locks(down_workspace, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = kernels::launch_sm87_nvfp4_marlin_down_cuda(
      activated, down.prefill_marlin_weight, down.prefill_marlin_scales,
      down.prefill_marlin_global_scale, token_count, output,
      down_workspace.reduction, down_workspace.locks, cuda_stream);
  if (status == static_cast<int>(cudaSuccess)) {
    ++logical_projection_hits;
    physical_launches += plan.launch_count;
  }
  return status;
}

[[nodiscard]] int launch_native_large_m_nvfp4_mlp(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down, const std::uint16_t* const input,
    std::uint16_t* const merged_gate_up,
    std::uint16_t* const activated, std::uint16_t* const output,
    const std::size_t token_count,
    const ExactPrefillProjectionWorkspace& gate_up_workspace,
    const ExactPrefillProjectionWorkspace& down_workspace,
    std::size_t& logical_projection_hits, std::size_t& physical_launches,
    void* const cuda_stream) noexcept {
  if (!valid_native_large_m_nvfp4_mlp_weights(gate, up, down) ||
      input == nullptr || merged_gate_up == nullptr || activated == nullptr ||
      output == nullptr ||
      token_count != kLayerMajorPrefillOperatorPanelTokens ||
      !valid_exact_prefill_projection_workspace(gate_up_workspace) ||
      !valid_exact_prefill_projection_workspace(down_workspace)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int status =
      clear_exact_prefill_projection_locks(gate_up_workspace, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status =
      kernels::launch_sm87_nvfp4_marlin_gate_up_aligned_operator_panel_cuda(
          input, gate.prefill_marlin_weight, gate.prefill_marlin_scales,
          gate.prefill_marlin_global_scale, token_count, merged_gate_up,
          gate_up_workspace.reduction, gate_up_workspace.locks, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = kernels::launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
      merged_gate_up, token_count, activated, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = clear_exact_prefill_projection_locks(down_workspace, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = kernels::launch_sm87_nvfp4_marlin_down_aligned_operator_panel_cuda(
      activated, down.prefill_marlin_weight, down.prefill_marlin_scales,
      down.prefill_marlin_global_scale, token_count, output,
      down_workspace.reduction, down_workspace.locks, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  logical_projection_hits += 2U;
  physical_launches += 2U;
  return static_cast<int>(cudaSuccess);
}

#if defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
[[nodiscard]] int launch_nvfp4_true_large_m_mlp(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down, const std::uint16_t* const input,
    std::uint16_t* const merged_gate_up,
    std::uint16_t* const activated, std::uint16_t* const output,
    const std::size_t token_count, std::size_t& logical_projection_hits,
    std::size_t& gate_up_hits, std::size_t& down_hits,
    std::size_t& physical_launches, void* const cuda_stream) noexcept {
  if (!valid_exact_contract_nvfp4_mlp_weights(gate, up, down) ||
      input == nullptr || merged_gate_up == nullptr || activated == nullptr ||
      output == nullptr ||
      !is_nvfp4_true_large_m_prefill_panel_tokens(token_count) ||
      !kernels::sm87_nvfp4_prefill_large_m_supports(
          kernels::Sm87NvFp4PrefillLargeMRole::kGateUp, token_count) ||
      !kernels::sm87_nvfp4_prefill_large_m_supports(
          kernels::Sm87NvFp4PrefillLargeMRole::kDown, token_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  int status = kernels::launch_sm87_nvfp4_prefill_large_m_gate_up_cuda(
      input, gate.prefill_marlin_weight, gate.prefill_marlin_scales,
      gate.prefill_marlin_global_scale, token_count, merged_gate_up,
      cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = kernels::launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
      merged_gate_up, token_count, activated, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  status = kernels::launch_sm87_nvfp4_prefill_large_m_down_cuda(
      activated, down.prefill_marlin_weight, down.prefill_marlin_scales,
      down.prefill_marlin_global_scale, token_count, output, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  logical_projection_hits += 2U;
  ++gate_up_hits;
  ++down_hits;
  physical_launches += 2U;
  return static_cast<int>(cudaSuccess);
}
#endif

#if defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
[[nodiscard]] int launch_nvfp4_g2_d2_large_m_mlp(
    const NvFp4LinearWeight& gate, const NvFp4LinearWeight& up,
    const NvFp4LinearWeight& down, const std::uint16_t* const input,
    std::uint16_t* const activated, std::uint16_t* const residual_output,
    const std::size_t token_count, std::size_t& logical_projection_hits,
    std::size_t& gate_up_hits, std::size_t& down_hits,
    std::size_t& physical_launches, const char** const failure_operation,
    void* const cuda_stream) noexcept {
  if (failure_operation == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *failure_operation = "prefill_operator_panel_nvfp4_g2_d2_contract";
  if (!valid_exact_contract_nvfp4_mlp_weights(gate, up, down) ||
      gate.prefill_marlin_gate_up_layout !=
          NvFp4MarlinGateUpLayout::kCanonicalGateThenUp ||
      input == nullptr || activated == nullptr || residual_output == nullptr ||
      !is_nvfp4_true_large_m_prefill_panel_tokens(token_count) ||
      !kernels::sm87_nvfp4_prefill_g2_d2_supports(
          kernels::Sm87NvFp4PrefillG2D2Role::kGateUpG2, token_count) ||
      !kernels::sm87_nvfp4_prefill_g2_d2_supports(
          kernels::Sm87NvFp4PrefillG2D2Role::kDownD2, token_count)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  *failure_operation = "prefill_operator_panel_nvfp4_gate_up_g2";
  int status = kernels::launch_sm87_nvfp4_prefill_gate_up_g2_cuda(
      input, gate.prefill_marlin_weight, gate.prefill_marlin_scales,
      gate.prefill_marlin_global_scale, token_count, activated, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  *failure_operation = "prefill_operator_panel_nvfp4_down_d2";
  status = kernels::launch_sm87_nvfp4_prefill_down_d2_cuda(
      activated, down.prefill_marlin_weight, down.prefill_marlin_scales,
      down.prefill_marlin_global_scale, residual_output, token_count,
      residual_output, cuda_stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  logical_projection_hits += 2U;
  ++gate_up_hits;
  ++down_hits;
  physical_launches += 2U;
  *failure_operation = "prefill_operator_panel_nvfp4_g2_d2_mlp";
  return static_cast<int>(cudaSuccess);
}
#endif
#endif

}  // namespace

ReferenceRunner::~ReferenceRunner() { release(); }

ReferenceRunner::ReferenceRunner(ReferenceRunner&& other) noexcept
    : ReferenceRunner() {
  *this = std::move(other);
}

ReferenceRunner& ReferenceRunner::operator=(ReferenceRunner&& other) noexcept {
  if (this == &other) {
    return *this;
  }
  release();
  weights_ = std::exchange(other.weights_, nullptr);
  state_ = std::exchange(other.state_, nullptr);
  stream_ = std::exchange(other.stream_, nullptr);
  prefill_auxiliary_stream_ =
      std::exchange(other.prefill_auxiliary_stream_, nullptr);
  prefill_branch_ready_event_ =
      std::exchange(other.prefill_branch_ready_event_, nullptr);
  prefill_branch_done_event_ =
      std::exchange(other.prefill_branch_done_event_, nullptr);
  for (std::size_t slot = 0U;
       slot < whole_request_submission_events_.size(); ++slot) {
    whole_request_submission_events_[slot] =
        std::exchange(other.whole_request_submission_events_[slot], nullptr);
  }
  prefill_gdn_chunk64_reference_context_ =
      std::exchange(other.prefill_gdn_chunk64_reference_context_, nullptr);
  prefill_gdn_chunk64_reference_workspace_ =
      std::exchange(other.prefill_gdn_chunk64_reference_workspace_, nullptr);
  prefill_gdn_chunk64_reference_workspace_bytes_ = std::exchange(
      other.prefill_gdn_chunk64_reference_workspace_bytes_, 0U);
  prefill_gdn_chunk64_native_workspace_ =
      std::exchange(other.prefill_gdn_chunk64_native_workspace_, nullptr);
  prefill_gdn_chunk64_native_workspace_bytes_ = std::exchange(
      other.prefill_gdn_chunk64_native_workspace_bytes_, 0U);
  pinned_logits_ = std::exchange(other.pinned_logits_, nullptr);
  pinned_trace_ = std::exchange(other.pinned_trace_, nullptr);
  decode_graph_p1_slots_ = other.decode_graph_p1_slots_;
  other.decode_graph_p1_slots_ = {};
  decode_graph_capture_active_ =
      std::exchange(other.decode_graph_capture_active_, false);
  views_ = other.views_;
  other.views_ = {};
  layer_major_request_views_ =
      std::move(other.layer_major_request_views_);
  other.layer_major_request_views_.reset();
  whole_request_prefill_stage_ = other.whole_request_prefill_stage_;
  other.whole_request_prefill_stage_ = {};
  projection_backend_ = std::exchange(
      other.projection_backend_, ProjectionBackend::kReference);
  trace_enabled_ = std::exchange(other.trace_enabled_, false);
  trace_valid_ = std::exchange(other.trace_valid_, false);
  poisoned_ = std::exchange(other.poisoned_, false);
  retained_prefill_hidden_valid_ =
      std::exchange(other.retained_prefill_hidden_valid_, false);
  retained_prefill_position_ =
      std::exchange(other.retained_prefill_position_, 0U);
  retained_prefill_input_token_ =
      std::exchange(other.retained_prefill_input_token_, 0U);
  retained_prefill_hidden_row_ =
      std::exchange(other.retained_prefill_hidden_row_, 0U);
  trace_position_ = std::exchange(other.trace_position_, 0U);
  trace_input_token_ = std::exchange(other.trace_input_token_, 0U);
  prefill_route_evidence_ =
      std::exchange(other.prefill_route_evidence_, PrefillRouteEvidence{});
  return *this;
}

ReferenceRunner::operator bool() const noexcept {
  bool ready = weights_ != nullptr && state_ != nullptr &&
               stream_ != nullptr && pinned_logits_ != nullptr;
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  ready = ready && prefill_gdn_chunk64_reference_context_ != nullptr &&
          prefill_gdn_chunk64_reference_workspace_ != nullptr &&
          prefill_gdn_chunk64_reference_workspace_bytes_ >=
              gdn_prefill_chunk64_reference_detail::workspace_bytes();
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  ready = ready && prefill_gdn_chunk64_native_workspace_ != nullptr &&
          prefill_gdn_chunk64_native_workspace_bytes_ >=
              gdn_prefill_chunk64_native_detail::workspace_bytes();
#endif
  if (layer_major_request_views_.has_value()) {
    for (const void* const event : whole_request_submission_events_) {
      ready = ready && event != nullptr;
    }
  }
  return ready;
}

std::uint32_t ReferenceRunner::current_position() const noexcept {
  return state_ == nullptr ? 0U : state_->current_position();
}

void ReferenceRunner::release() noexcept {
  // A whole-request core may have advanced recurrent/KV device state while
  // leaving the host sequence length unpublished. The Engine owns the normal
  // rollback guard; this is the final best-effort backstop for an active or
  // poisoned runner that is destroyed or overwritten by move-assignment.
  if ((whole_request_prefill_active() || poisoned_) &&
      static_cast<bool>(*this)) {
    (void)reset();
  }
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  if (prefill_gdn_chunk64_reference_context_ != nullptr) {
    (void)gdn_prefill_chunk64_reference_detail::destroy_context(
        prefill_gdn_chunk64_reference_context_);
  }
  if (prefill_gdn_chunk64_reference_workspace_ != nullptr) {
    (void)cudaFree(prefill_gdn_chunk64_reference_workspace_);
  }
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  if (prefill_gdn_chunk64_native_workspace_ != nullptr) {
    (void)cudaFree(prefill_gdn_chunk64_native_workspace_);
  }
#endif
  destroy_decode_graph_p1();
  if (prefill_branch_ready_event_ != nullptr) {
    (void)cudaEventDestroy(
        reinterpret_cast<cudaEvent_t>(prefill_branch_ready_event_));
  }
  if (prefill_branch_done_event_ != nullptr) {
    (void)cudaEventDestroy(
        reinterpret_cast<cudaEvent_t>(prefill_branch_done_event_));
  }
  for (void* const event : whole_request_submission_events_) {
    if (event != nullptr) {
      (void)cudaEventDestroy(reinterpret_cast<cudaEvent_t>(event));
    }
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamDestroy(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
  if (stream_ != nullptr) {
    (void)cudaStreamDestroy(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (pinned_trace_ != nullptr) {
    (void)cudaFreeHost(pinned_trace_);
  }
  if (pinned_logits_ != nullptr) {
    (void)cudaFreeHost(pinned_logits_);
  }
  weights_ = nullptr;
  state_ = nullptr;
  stream_ = nullptr;
  prefill_auxiliary_stream_ = nullptr;
  prefill_branch_ready_event_ = nullptr;
  prefill_branch_done_event_ = nullptr;
  whole_request_submission_events_ = {};
  prefill_gdn_chunk64_reference_context_ = nullptr;
  prefill_gdn_chunk64_reference_workspace_ = nullptr;
  prefill_gdn_chunk64_reference_workspace_bytes_ = 0U;
  prefill_gdn_chunk64_native_workspace_ = nullptr;
  prefill_gdn_chunk64_native_workspace_bytes_ = 0U;
  pinned_logits_ = nullptr;
  pinned_trace_ = nullptr;
  decode_graph_capture_active_ = false;
  views_ = {};
  layer_major_request_views_.reset();
  whole_request_prefill_stage_ = {};
  projection_backend_ = ProjectionBackend::kReference;
  trace_enabled_ = false;
  trace_valid_ = false;
  poisoned_ = false;
  retained_prefill_hidden_valid_ = false;
  retained_prefill_position_ = 0U;
  retained_prefill_input_token_ = 0U;
  retained_prefill_hidden_row_ = 0U;
  trace_position_ = 0U;
  trace_input_token_ = 0U;
  prefill_route_evidence_ = {};
}

int ReferenceRunner::destroy_decode_graph_p1_slot(
    DecodeGraphP1Slot& slot) noexcept {
  int first_error = static_cast<int>(cudaSuccess);
  if (slot.exec != nullptr) {
    const cudaError_t status = cudaGraphExecDestroy(
        reinterpret_cast<cudaGraphExec_t>(slot.exec));
    if (status != cudaSuccess) {
      first_error = static_cast<int>(status);
    }
  }
  if (slot.graph != nullptr) {
    const cudaError_t status =
        cudaGraphDestroy(reinterpret_cast<cudaGraph_t>(slot.graph));
    if (status != cudaSuccess && first_error == static_cast<int>(cudaSuccess)) {
      first_error = static_cast<int>(status);
    }
  }
  slot = {};
  return first_error;
}

void ReferenceRunner::destroy_decode_graph_p1_slot(
    const std::size_t position) noexcept {
  if (position >= decode_graph_p1_slots_.size()) {
    return;
  }
  (void)destroy_decode_graph_p1_slot(decode_graph_p1_slots_[position]);
}

void ReferenceRunner::destroy_decode_graph_p1() noexcept {
  for (std::size_t position = 0U;
       position < decode_graph_p1_slots_.size(); ++position) {
    destroy_decode_graph_p1_slot(position);
  }
}

ReferenceStepOutcome ReferenceRunner::fail_step(
    const ReferenceRunnerStatus status) noexcept {
  // A failed launch may follow earlier successful launches in this token.
  // Drain the owned stream before returning so every step has a synchronous
  // completion boundary even though its mutated device state is not committed
  // and cannot be reused until reset.
  if (decode_graph_capture_active_ && stream_ != nullptr) {
    cudaGraph_t discarded_graph = nullptr;
    (void)cudaStreamEndCapture(reinterpret_cast<cudaStream_t>(stream_),
                               &discarded_graph);
    decode_graph_capture_active_ = false;
    if (discarded_graph != nullptr) {
      (void)cudaGraphDestroy(discarded_graph);
    }
    (void)cudaGetLastError();
  }
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
  poisoned_ = true;
  trace_valid_ = false;
  retained_prefill_hidden_valid_ = false;
  whole_request_prefill_stage_ = {};
  ReferenceStepOutcome outcome;
  outcome.status = status;
  return outcome;
}

ReferencePrefillTileOutcome ReferenceRunner::fail_prefill_tile(
    const ReferenceRunnerStatus status) noexcept {
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
  poisoned_ = true;
  trace_valid_ = false;
  retained_prefill_hidden_valid_ = false;
  whole_request_prefill_stage_ = {};
  ReferencePrefillTileOutcome outcome;
  outcome.status = status;
  return outcome;
}

bool ReferenceRunner::whole_request_prefill_active() const noexcept {
  return whole_request_prefill_stage_.phase !=
         WholeRequestPrefillStagePhase::kIdle;
}

ReferenceWholeRequestPrefillOutcome
ReferenceRunner::fail_whole_request_prefill(
    const ReferenceRunnerStatus status) noexcept {
  ReferenceRunnerStatus terminal_status = status;
  const auto capture_drain_failure =
      [&terminal_status](const cudaError_t drain_status,
                         const char* const operation) noexcept {
        if (drain_status != cudaSuccess &&
            terminal_status.error != ReferenceRunnerError::kCudaFailure) {
          terminal_status = runner_status(
              ReferenceRunnerError::kCudaFailure, operation,
              terminal_status.layer, static_cast<int>(drain_status),
              terminal_status.retired_prefill_quanta);
        }
      };
  if (stream_ != nullptr) {
    capture_drain_failure(
        cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_)),
        "whole_request_prefill_failure_drain_main");
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    capture_drain_failure(
        cudaStreamSynchronize(
            reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_)),
        "whole_request_prefill_failure_drain_auxiliary");
  }
  poisoned_ = true;
  trace_valid_ = false;
  retained_prefill_hidden_valid_ = false;
  whole_request_prefill_stage_ = {};
  ReferenceWholeRequestPrefillOutcome outcome;
  outcome.status = terminal_status;
  return outcome;
}

ReferenceRunnerStatus ReferenceRunner::fail_whole_request_status(
    const ReferenceRunnerStatus status) noexcept {
  if (stream_ != nullptr) {
    (void)cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    (void)cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
  }
  poisoned_ = true;
  trace_valid_ = false;
  retained_prefill_hidden_valid_ = false;
  whole_request_prefill_stage_ = {};
  return status;
}

std::optional<ReferenceTraceView> ReferenceRunner::last_trace() const noexcept {
  if (!trace_valid_ || pinned_trace_ == nullptr) {
    return std::nullopt;
  }
  return ReferenceTraceView{trace_position_, trace_input_token_, pinned_trace_,
                            kReferenceTraceElements};
}

ReferenceRunnerStatus ReferenceRunner::reset() noexcept {
  // Reset is the sole recovery path from an uncommitted whole-request stage.
  // Preserve the hand-off as an abort-required latch until every producer is
  // drained and RequestState reset has completed successfully.
  if (!static_cast<bool>(*this)) {
    return runner_status(ReferenceRunnerError::kInvalidRunner, "reset");
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    const cudaError_t auxiliary_sync_status = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
    if (auxiliary_sync_status != cudaSuccess) {
      poisoned_ = true;
      trace_valid_ = false;
      return runner_status(ReferenceRunnerError::kCudaFailure,
                           "reset_auxiliary_synchronize",
                           kReferenceNoLayer,
                           static_cast<int>(auxiliary_sync_status));
    }
  }
  const RequestOperationStatus reset_status = state_->reset_async(stream_);
  if (!reset_status) {
    poisoned_ = true;
    trace_valid_ = false;
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "request_state_reset", kReferenceNoLayer,
                         reset_status.cuda_error);
  }
  const cudaError_t sync_status = cudaStreamSynchronize(
      reinterpret_cast<cudaStream_t>(stream_));
  if (sync_status != cudaSuccess) {
    poisoned_ = true;
    trace_valid_ = false;
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "reset_synchronize", kReferenceNoLayer,
                         static_cast<int>(sync_status));
  }
  poisoned_ = false;
  trace_valid_ = false;
  retained_prefill_hidden_valid_ = false;
  retained_prefill_position_ = 0U;
  retained_prefill_input_token_ = 0U;
  retained_prefill_hidden_row_ = 0U;
  trace_position_ = 0U;
  trace_input_token_ = 0U;
  whole_request_prefill_stage_ = {};
  reset_prefill_route_request(prefill_route_evidence_);
  return {};
}

ReferenceRunnerStatus ReferenceRunner::record_scalar_prefill_route_fallback()
    noexcept {
  if (whole_request_prefill_active()) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_scalar_route_active"));
  }
  PrefillRouteEvidence tile;
  for (std::size_t index = 0U;
       index < kPrefillOperatorRoleCount; ++index) {
    if (!record_prefill_operator_route(
            tile, static_cast<PrefillOperatorRole>(index),
            PrefillRouteDisposition::kExactFallback,
            kExpectedPrefillLogicalOperatorsPerTile[index])) {
      return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                           "prefill_scalar_route_record");
    }
  }
  if (!commit_prefill_route_layer_pass(prefill_route_evidence_, tile)) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_scalar_route_commit");
  }
  return {};
}

PrefillRouteEvidence ReferenceRunner::finalize_prefill_route_evidence(
    const std::uint64_t expected_layer_passes) noexcept {
  if (whole_request_prefill_active()) {
    PrefillRouteEvidence rejected = prefill_route_evidence_;
    // Finalization is a terminal request operation. Calling it before the
    // staged logits/commit hand-off finishes must abort the transaction just
    // like every other competing mutator; otherwise a caller could observe a
    // failed witness while later publishing the staged device state.
    (void)fail_whole_request_status(runner_status(
        ReferenceRunnerError::kRouteEvidenceFailure,
        "whole_request_prefill_route_finalize_active"));
    rejected.error = PrefillRouteEvidenceError::kIncompleteTile;
    rejected.request_active = false;
    rejected.complete = true;
    rejected.valid = false;
    rejected.expected_layer_passes = expected_layer_passes;
    return rejected;
  }
  return finalize_prefill_route_request(prefill_route_evidence_,
                                        expected_layer_passes);
}

ReferenceStepOutcome ReferenceRunner::step(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  if (whole_request_prefill_active()) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_step_active"));
  }
  return step_impl(input_token_id, options,
                   DecodeGraphP1Action::kDisabled);
}

ReferenceDecodeGraphP1PrepareOutcome
ReferenceRunner::prepare_fixed_position_decode_graph_p1(
    const std::uint32_t input_token_id) noexcept {
  if (whole_request_prefill_active()) {
    ReferenceDecodeGraphP1PrepareOutcome outcome;
    outcome.status = fail_whole_request_status(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_graph_prepare_active"));
    return outcome;
  }
  const std::uint32_t position = current_position();
  ReferenceStepOptions options;
  options.compute_logits = true;
  options.capture_trace = false;
  options.measure_timing = false;
  options.logits_mode = ReferenceLogitsMode::kPredictedTokenOnly;
  ReferenceStepOutcome captured =
      step_impl(input_token_id, options,
                DecodeGraphP1Action::kCaptureOnly);
  ReferenceDecodeGraphP1PrepareOutcome outcome;
  outcome.status = captured.status;
  const std::optional<ReferenceDecodeGraphP1Stats> stats =
      fixed_position_decode_graph_p1_stats(position);
  if (captured && stats.has_value()) {
    outcome.value.emplace(*stats);
  }
  return outcome;
}

ReferenceDecodeGraphCachePrepareOutcome
ReferenceRunner::prepare_fixed_position_decode_graph_cache(
    const std::uint32_t first_position,
    const std::uint32_t last_position,
    const std::uint32_t input_token_id) noexcept {
  ReferenceDecodeGraphCachePrepareOutcome outcome;
  if (whole_request_prefill_active()) {
    outcome.status = fail_whole_request_status(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_graph_cache_prepare_active"));
    return outcome;
  }
  if (!static_cast<bool>(*this)) {
    outcome.status = runner_status(ReferenceRunnerError::kInvalidRunner,
                                   "decode_graph_cache_prepare");
    return outcome;
  }
  if (poisoned_) {
    outcome.status = runner_status(ReferenceRunnerError::kPoisoned,
                                   "decode_graph_cache_prepare");
    return outcome;
  }
  if (first_position > last_position ||
      last_position >= decode_graph_p1_slots_.size()) {
    outcome.status = runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "decode_graph_cache_range");
    return outcome;
  }
  if (last_position >= state_->max_sequence_length()) {
    outcome.status = runner_status(
        ReferenceRunnerError::kCapacityExceeded,
        "decode_graph_cache_capacity");
    return outcome;
  }
  if (input_token_id >= kReferenceVocabularySize) {
    outcome.status = runner_status(ReferenceRunnerError::kTokenOutOfRange,
                                   "decode_graph_cache_input_token");
    return outcome;
  }
  if (projection_backend_ != ProjectionBackend::kSm87WeightOnly ||
      linear_weight_kind(weights_->lm_head()) == LinearWeightKind::kBf16) {
    outcome.status = runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "decode_graph_cache_contract");
    return outcome;
  }
  for (std::uint32_t position = first_position;
       position <= last_position; ++position) {
    if (has_fixed_position_decode_graph_p1(position)) {
      outcome.status = runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_cache_range_not_empty");
      return outcome;
    }
  }

  const std::uint32_t entry_position = state_->current_position();
  std::array<DecodeGraphP1Slot, kReferenceDecodeGraphP2MaximumSlots>
      staged_slots{};
  ReferenceDecodeGraphCachePrepareResult prepared;
  ReferenceStepOptions options;
  options.compute_logits = true;
  options.capture_trace = false;
  options.measure_timing = false;
  options.logits_mode = ReferenceLogitsMode::kPredictedTokenOnly;

  const auto destroy_staged = [&staged_slots]() noexcept {
    int first_error = static_cast<int>(cudaSuccess);
    for (DecodeGraphP1Slot& slot : staged_slots) {
      const int status = ReferenceRunner::destroy_decode_graph_p1_slot(slot);
      if (status != static_cast<int>(cudaSuccess) &&
          first_error == static_cast<int>(cudaSuccess)) {
        first_error = status;
      }
    }
    return first_error;
  };
  const auto restore_entry_position = [this, entry_position]() noexcept {
    return state_->set_sequence_length(entry_position);
  };

  for (std::uint32_t position = first_position;
       position <= last_position; ++position) {
    const RequestOperationStatus position_status =
        state_->set_sequence_length(position);
    if (!position_status) {
      const RequestOperationStatus restore_status = restore_entry_position();
      const int cleanup_status = destroy_staged();
      poisoned_ = true;
      trace_valid_ = false;
      if (!restore_status) {
        outcome.status = runner_status(
            ReferenceRunnerError::kStateCommitFailure,
            "decode_graph_cache_restore_position", kReferenceNoLayer,
            restore_status.cuda_error);
      } else if (cleanup_status != static_cast<int>(cudaSuccess)) {
        outcome.status = runner_status(
            ReferenceRunnerError::kCudaFailure,
            "decode_graph_cache_rollback_destroy", kReferenceNoLayer,
            cleanup_status);
      } else {
        outcome.status = runner_status(
            ReferenceRunnerError::kStateCommitFailure,
            "decode_graph_cache_set_position", kReferenceNoLayer,
            position_status.cuda_error);
      }
      return outcome;
    }
    ReferenceStepOutcome captured = step_impl(
        input_token_id, options, DecodeGraphP1Action::kCaptureOnly,
        &staged_slots[position]);
    if (!captured) {
      const RequestOperationStatus restore_status = restore_entry_position();
      const int cleanup_status = destroy_staged();
      if (!restore_status) {
        poisoned_ = true;
        trace_valid_ = false;
        outcome.status = runner_status(
            ReferenceRunnerError::kStateCommitFailure,
            "decode_graph_cache_restore_position", kReferenceNoLayer,
            restore_status.cuda_error);
      } else if (cleanup_status != static_cast<int>(cudaSuccess)) {
        poisoned_ = true;
        trace_valid_ = false;
        outcome.status = runner_status(
            ReferenceRunnerError::kCudaFailure,
            "decode_graph_cache_rollback_destroy", kReferenceNoLayer,
            cleanup_status);
      } else {
        outcome.status = captured.status;
      }
      return outcome;
    }
    prepared.graphs[prepared.graph_count++] =
        staged_slots[position].stats;
    prepared.prepared_mask |= std::uint64_t{1U} << position;
  }

  const RequestOperationStatus restore_status = restore_entry_position();
  if (!restore_status) {
    const int cleanup_status = destroy_staged();
    poisoned_ = true;
    trace_valid_ = false;
    outcome.status = runner_status(
        cleanup_status == static_cast<int>(cudaSuccess)
            ? ReferenceRunnerError::kStateCommitFailure
            : ReferenceRunnerError::kCudaFailure,
        cleanup_status == static_cast<int>(cudaSuccess)
            ? "decode_graph_cache_restore_position"
            : "decode_graph_cache_rollback_destroy",
        kReferenceNoLayer,
        cleanup_status == static_cast<int>(cudaSuccess)
            ? restore_status.cuda_error
            : cleanup_status);
    return outcome;
  }

  for (std::uint32_t position = first_position;
       position <= last_position; ++position) {
    decode_graph_p1_slots_[position] = staged_slots[position];
    staged_slots[position] = {};
  }
  outcome.value.emplace(std::move(prepared));
  return outcome;
}

std::uint64_t
ReferenceRunner::fixed_position_decode_graph_cache_mask() const noexcept {
  std::uint64_t mask = 0U;
  for (std::uint32_t position = 0U;
       position < decode_graph_p1_slots_.size(); ++position) {
    if (has_fixed_position_decode_graph_p1(position)) {
      mask |= std::uint64_t{1U} << position;
    }
  }
  return mask;
}

ReferenceRunnerStatus
ReferenceRunner::clear_fixed_position_decode_graph_cache() noexcept {
  if (whole_request_prefill_active()) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_graph_cache_clear_active"));
  }
  if (!static_cast<bool>(*this)) {
    return runner_status(ReferenceRunnerError::kInvalidRunner,
                         "decode_graph_cache_clear");
  }
  const cudaError_t main_sync_status =
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream_));
  if (main_sync_status != cudaSuccess) {
    poisoned_ = true;
    trace_valid_ = false;
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "decode_graph_cache_clear_main_synchronize",
                         kReferenceNoLayer,
                         static_cast<int>(main_sync_status));
  }
  if (prefill_auxiliary_stream_ != nullptr) {
    const cudaError_t auxiliary_sync_status = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_));
    if (auxiliary_sync_status != cudaSuccess) {
      poisoned_ = true;
      trace_valid_ = false;
      return runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_cache_clear_auxiliary_synchronize",
          kReferenceNoLayer, static_cast<int>(auxiliary_sync_status));
    }
  }

  std::array<DecodeGraphP1Slot, kReferenceDecodeGraphP2MaximumSlots>
      detached_slots = decode_graph_p1_slots_;
  decode_graph_p1_slots_ = {};
  int first_error = static_cast<int>(cudaSuccess);
  for (DecodeGraphP1Slot& slot : detached_slots) {
    const int status = destroy_decode_graph_p1_slot(slot);
    if (status != static_cast<int>(cudaSuccess) &&
        first_error == static_cast<int>(cudaSuccess)) {
      first_error = status;
    }
  }
  if (first_error != static_cast<int>(cudaSuccess)) {
    poisoned_ = true;
    trace_valid_ = false;
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "decode_graph_cache_clear_destroy",
                         kReferenceNoLayer, first_error);
  }
  return {};
}

bool ReferenceRunner::has_fixed_position_decode_graph_p1(
    const std::uint32_t position) const noexcept {
  if (position >= decode_graph_p1_slots_.size()) {
    return false;
  }
  const DecodeGraphP1Slot& slot = decode_graph_p1_slots_[position];
  return slot.graph != nullptr && slot.exec != nullptr &&
         slot.embedding_node != nullptr &&
         slot.embedding_launch.function != nullptr &&
         slot.stats.position == position;
}

std::optional<ReferenceDecodeGraphP1Stats>
ReferenceRunner::fixed_position_decode_graph_p1_stats(
    const std::uint32_t position) const noexcept {
  if (!has_fixed_position_decode_graph_p1(position)) {
    return std::nullopt;
  }
  return decode_graph_p1_slots_[position].stats;
}

ReferenceStepOutcome
ReferenceRunner::replay_fixed_position_decode_graph_p1(
    const std::uint32_t input_token_id,
    const bool measure_timing) noexcept {
  if (whole_request_prefill_active()) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_graph_replay_active"));
  }
  ReferenceStepOptions options;
  options.compute_logits = true;
  options.capture_trace = false;
  options.measure_timing = measure_timing;
  options.logits_mode = ReferenceLogitsMode::kPredictedTokenOnly;
  return step_impl(input_token_id, options,
                   DecodeGraphP1Action::kReplay);
}

ReferenceStepOutcome ReferenceRunner::step_impl(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options,
    const DecodeGraphP1Action graph_action,
    DecodeGraphP1Slot* const capture_destination) noexcept {
  if (whole_request_prefill_active()) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_step_impl_active"));
  }
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }
  // Any ordinary step may overwrite the shared hidden workspace. Retention is
  // a one-call hand-off from a marked prefill tile to its dedicated finalizer.
  retained_prefill_hidden_valid_ = false;
  if (!static_cast<bool>(*this)) {
    return fail_step(
        runner_status(ReferenceRunnerError::kInvalidRunner, "step"));
  }
  if (poisoned_) {
    ReferenceStepOutcome outcome;
    outcome.status =
        runner_status(ReferenceRunnerError::kPoisoned, "step");
    return outcome;
  }
  if (!is_valid_reference_logits_mode(options.logits_mode)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions, "logits_mode"));
  }
  if (input_token_id >= kReferenceVocabularySize) {
    return fail_step(runner_status(ReferenceRunnerError::kTokenOutOfRange,
                                   "input_token"));
  }
  if (state_->remaining_capacity() == 0U) {
    return fail_step(runner_status(ReferenceRunnerError::kCapacityExceeded,
                                   "request_capacity"));
  }
  if (options.capture_trace &&
      (!trace_enabled_ || pinned_trace_ == nullptr)) {
    return fail_step(runner_status(ReferenceRunnerError::kTraceUnavailable,
                                   "trace_not_reserved"));
  }

  const bool use_decode_graph_p1 =
      graph_action != DecodeGraphP1Action::kDisabled;
  if (use_decode_graph_p1 &&
      (!options.compute_logits || options.capture_trace ||
       options.logits_mode != ReferenceLogitsMode::kPredictedTokenOnly ||
       projection_backend_ != ProjectionBackend::kSm87WeightOnly ||
       linear_weight_kind(weights_->lm_head()) == LinearWeightKind::kBf16)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "decode_graph_p1_contract"));
  }
  if (graph_action == DecodeGraphP1Action::kCaptureOnly) {
    int device = -1;
    cudaDeviceProp properties{};
    cudaError_t cuda_status = cudaGetDevice(&device);
    if (cuda_status == cudaSuccess) {
      cuda_status = cudaGetDeviceProperties(&properties, device);
    }
    if (cuda_status != cudaSuccess) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_device", kReferenceNoLayer,
          static_cast<int>(cuda_status)));
    }
    if (properties.major != 8 || properties.minor != 7) {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_requires_sm87"));
    }
  }

  const std::uint32_t position = state_->current_position();
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  DecodeGraphP1Slot* decode_graph_slot = nullptr;
  Clock::time_point decode_graph_prepare_started{};
  if (graph_action == DecodeGraphP1Action::kCaptureOnly) {
    if (position >= decode_graph_p1_slots_.size()) {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_position_not_cacheable"));
    }
    decode_graph_slot = capture_destination == nullptr
                            ? &decode_graph_p1_slots_[position]
                            : capture_destination;
    decode_graph_prepare_started = Clock::now();
    const cudaError_t begin_status = cudaStreamBeginCapture(
        stream, cudaStreamCaptureModeThreadLocal);
    if (begin_status != cudaSuccess) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_begin_capture", kReferenceNoLayer,
          static_cast<int>(begin_status)));
    }
    decode_graph_capture_active_ = true;
  } else if (graph_action == DecodeGraphP1Action::kReplay) {
    if (!has_fixed_position_decode_graph_p1(position)) {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_not_prepared"));
    }
    decode_graph_slot = &decode_graph_p1_slots_[position];
  }
  ReferenceRunnerStatus launch_failure{};
  const auto check_cuda = [&launch_failure](
                              const int status, const char* const operation,
                              const std::size_t layer) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    launch_failure = runner_status(ReferenceRunnerError::kCudaFailure,
                                   operation, layer, status);
    return false;
  };
  const auto project = [this, &check_cuda](
                           const LinearWeight& weight,
                           const std::uint16_t* const input,
                           std::uint16_t* const output,
                           const char* const operation,
                           const std::size_t layer) noexcept {
    return check_cuda(launch_projection_to_bf16_cuda(
                          projection_backend_, weight, input,
                          views_.fp32_scratch,
                          views_.fp32_scratch_elements, output, stream_),
                      operation, layer);
  };
  const auto copy_trace = [this, stream, &check_cuda](
                              const std::uint16_t* const source,
                              const std::size_t offset,
                              const char* const operation,
                              const std::size_t layer) noexcept {
    if (source == nullptr || offset >
                                 kReferenceTraceElements -
                                     kReferenceHiddenSize) {
      return check_cuda(static_cast<int>(cudaErrorInvalidValue), operation,
                        layer);
    }
    return check_cuda(
        static_cast<int>(cudaMemcpyAsync(
            pinned_trace_ + offset, source,
            kReferenceHiddenSize * sizeof(std::uint16_t),
            cudaMemcpyDeviceToHost, stream)),
        operation, layer);
  };
  const bool prediction_only =
      options.compute_logits &&
      options.logits_mode == ReferenceLogitsMode::kPredictedTokenOnly;
  const bool use_sm87_bf16_logits =
      options.compute_logits &&
      projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
      linear_weight_kind(weights_->lm_head()) != LinearWeightKind::kBf16;

  if (graph_action != DecodeGraphP1Action::kReplay) {
  if (!check_cuda(launch_embedding_gather_reference_cuda(
                      weights_->embed_tokens().weight,
                      kReferenceVocabularySize, kReferenceHiddenSize,
                      input_token_id, views_.hidden[0], stream_),
                  "embedding_gather", kReferenceNoLayer)) {
    return fail_step(launch_failure);
  }
  if (options.capture_trace &&
      !copy_trace(views_.hidden[0], 0U, "trace_embedding",
                  kReferenceNoLayer)) {
    return fail_step(launch_failure);
  }
  if (!check_cuda(launch_centered_rms_norm_reference_cuda(
                      views_.hidden[0],
                      weights_->layer(0U).input_layernorm.data,
                      kReferenceHiddenSize, kRmsEpsilon, views_.hidden[1],
                      stream_),
                  "input_layernorm", 0U)) {
    return fail_step(launch_failure);
  }

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& layer_weights = weights_->layer(layer);
    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_step(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "linear_attention_variant", layer));
      }
      const int composite_status =
          launch_linear_attention_qkv_z_ab_to_bf16_cuda(
              projection_backend_, attention->in_proj_qkv,
              attention->in_proj_z, attention->in_proj_a,
              attention->in_proj_b, views_.hidden[1], views_.projection[0],
              views_.projection[1], views_.linear_a, views_.linear_b,
              stream_);
      const bool used_projection_composite =
          composite_status == static_cast<int>(cudaSuccess);
      if (!used_projection_composite &&
          composite_status != static_cast<int>(cudaErrorNotSupported)) {
        if (!check_cuda(composite_status,
                        "linear_qkv_z_a_b_projection", layer)) {
          return fail_step(launch_failure);
        }
      }
      if (!used_projection_composite) {
        if (!check_cuda(launch_projection_pair_tile_to_bf16_cuda(
                            projection_backend_, attention->in_proj_qkv,
                            attention->in_proj_z, views_.hidden[1], 1U,
                            views_.fp32_scratch,
                            views_.fp32_scratch_elements,
                            views_.projection[0], views_.projection[1],
                            stream_),
                        "linear_qkv_z_projection", layer)) {
          return fail_step(launch_failure);
        }
        if (supports_bf16_projection_pair(
                projection_backend_, attention->in_proj_a,
                attention->in_proj_b)) {
          if (!check_cuda(launch_projection_pair_tile_to_bf16_cuda(
                              projection_backend_, attention->in_proj_a,
                              attention->in_proj_b, views_.hidden[1], 1U,
                              views_.fp32_scratch,
                              views_.fp32_scratch_elements, views_.linear_a,
                              views_.linear_b, stream_),
                          "linear_a_b_projection", layer)) {
            return fail_step(launch_failure);
          }
        } else if (!project(attention->in_proj_a, views_.hidden[1],
                            views_.linear_a, "linear_a_projection", layer) ||
                   !project(attention->in_proj_b, views_.hidden[1],
                            views_.linear_b, "linear_b_projection", layer)) {
          return fail_step(launch_failure);
        }
      }
      if (!check_cuda(launch_causal_conv1d_silu_update_reference_cuda(
                          views_.projection[0], attention->conv1d.data,
                          views_.conv_state[layer], views_.projection[0], {},
                          stream_),
                      "linear_causal_conv", layer)) {
        return fail_step(launch_failure);
      }
      const bool use_gdn_norm_gate_composite =
          supports_gated_delta_net_update_plain_rms_norm_silu_gate_fusion(
              1U, {}, kGdnValueHeadCount, kGdnHeadDimension);
      if (use_gdn_norm_gate_composite) {
        if (!check_cuda(
                launch_gated_delta_net_update_plain_rms_norm_silu_gate_cuda(
                    views_.projection[0], views_.linear_a,
                    views_.linear_b, attention->a_log.data,
                    attention->dt_bias.data, views_.gdn_state[layer],
                    views_.gdn_state[layer], kRmsEpsilon,
                    attention->norm.data, views_.projection[1],
                    kGdnValueHeadCount, kGdnHeadDimension, kRmsEpsilon,
                    views_.projection[2], {}, stream_),
                "linear_gdn_output_norm_gate", layer)) {
          return fail_step(launch_failure);
        }
      } else if (!check_cuda(
                     launch_gated_delta_net_update_warp_parallel_cuda(
                         views_.projection[0], views_.linear_a,
                         views_.linear_b, attention->a_log.data,
                         attention->dt_bias.data, views_.gdn_state[layer],
                         views_.gdn_state[layer], kRmsEpsilon,
                         views_.projection[2], {}, stream_),
                     "linear_gdn", layer) ||
                 !check_cuda(
                     launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                         views_.projection[2], attention->norm.data,
                         views_.projection[1], kGdnValueHeadCount,
                         kGdnHeadDimension, kRmsEpsilon,
                         views_.projection[2], stream_),
                     "linear_output_norm_gate", layer)) {
        return fail_step(launch_failure);
      }
      if (!project(attention->out_proj, views_.projection[2],
                   views_.hidden[1], "linear_output_projection", layer)) {
        return fail_step(launch_failure);
      }
    } else if (expected == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_step(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "full_attention_variant", layer));
      }
      std::uint16_t* const current_key =
          views_.key_cache[layer] +
          static_cast<std::size_t>(position) * kFullKvElements;
      std::uint16_t* const current_value =
          views_.value_cache[layer] +
          static_cast<std::size_t>(position) * kFullKvElements;
      std::uint16_t* const packed_gates =
          views_.projection[3] + kFullQueryElements;
      std::uint16_t* full_query = views_.projection[0];
      const std::size_t rope_first_position =
          static_cast<std::size_t>(position);
      if (!check_cuda(launch_full_attention_q_kv_to_bf16_cuda(
                          projection_backend_, attention->q_proj,
                          attention->k_proj, attention->v_proj,
                          views_.hidden[1], views_.fp32_scratch,
                          views_.fp32_scratch_elements, views_.projection[0],
                          current_key, current_value, stream_),
                      "full_q_k_v_projection", layer)) {
        return fail_step(launch_failure);
      }

      if (reference_runner_detail::use_qk_rope_tile(rope_first_position,
                                                     1U)) {
        full_query = views_.projection[3];
        if (!check_cuda(launch_full_attention_preprocess_24_4_256_64_cuda(
                            views_.projection[0], current_key,
                            attention->q_norm.data, attention->k_norm.data,
                            kRmsEpsilon, full_query, packed_gates,
                            views_.rope_cos, views_.rope_sin,
                            rope_first_position, 1U, stream_),
                        "full_preprocess", layer)) {
          return fail_step(launch_failure);
        }
      } else {
        if (!check_cuda(launch_split_interleaved_q_gate_reference_cuda(
                            views_.projection[0], kFullQueryHeads,
                            kFullHeadDimension, views_.projection[3],
                            packed_gates, stream_),
                        "full_split_q_gate_fallback", layer) ||
            !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                            views_.projection[3], attention->q_norm.data,
                            kFullQueryHeads, kFullHeadDimension, kRmsEpsilon,
                            full_query, stream_),
                        "full_q_norm_fallback", layer) ||
            !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                            current_key, attention->k_norm.data,
                            kFullKvHeads, kFullHeadDimension, kRmsEpsilon,
                            current_key, stream_),
                        "full_k_norm_fallback", layer)) {
          return fail_step(launch_failure);
        }
        const float* const cosines =
            views_.rope_cos + rope_first_position * kRopePairs;
        const float* const sines =
            views_.rope_sin + rope_first_position * kRopePairs;
        if (!check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                            full_query, cosines, sines, kFullQueryHeads,
                            full_query, stream_),
                        "full_q_rope_fallback", layer) ||
            !check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                            current_key, cosines, sines, kFullKvHeads,
                            current_key, stream_),
                        "full_k_rope_fallback", layer)) {
          return fail_step(launch_failure);
        }
      }

      const std::size_t sequence_length =
          static_cast<std::size_t>(position) + 1U;
      if (reference_runner_detail::use_fused_gqa_sigmoid_gate_tile(
              position, 1U)) {
        if (!check_cuda(
                launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                    full_query, views_.key_cache[layer],
                    views_.value_cache[layer], sequence_length,
                    kAttentionScale, views_.fp32_scratch,
                    views_.fp32_scratch_elements, packed_gates,
                    views_.projection[1], stream_),
                "full_gqa_output_gate", layer)) {
          return fail_step(launch_failure);
        }
      } else if (reference_runner_detail::use_decode_gqa_splitkv(
                     sequence_length)) {
        if (!check_cuda(
                launch_gqa_attention_splitkv_sigmoid_gate_24_4_256_cuda(
                    full_query, views_.key_cache[layer],
                    views_.value_cache[layer], sequence_length,
                    kAttentionScale, views_.fp32_scratch,
                    views_.fp32_scratch_elements, packed_gates,
                    views_.projection[1], stream_),
                "full_gqa_splitkv_output_gate", layer)) {
          return fail_step(launch_failure);
        }
      } else if (!check_cuda(launch_gqa_attention_reference_cuda(
                                 full_query, views_.key_cache[layer],
                                 views_.value_cache[layer], kFullQueryHeads,
                                 kFullKvHeads, sequence_length,
                                 kFullHeadDimension, kAttentionScale,
                                 views_.fp32_scratch,
                                 views_.fp32_scratch_elements,
                                 views_.projection[1], stream_),
                             "full_gqa", layer) ||
                 !check_cuda(launch_sigmoid_gate_reference_cuda(
                                 views_.projection[1], packed_gates,
                                 kFullQueryElements, views_.projection[1],
                                 stream_),
                             "full_output_gate", layer)) {
        return fail_step(launch_failure);
      }
      if (!project(attention->o_proj, views_.projection[1], views_.hidden[1],
                   "full_output_projection", layer)) {
        return fail_step(launch_failure);
      }
    } else {
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidLayerSchedule, "layer_schedule",
          layer));
    }

    if (!check_cuda(
            launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
                projection_backend_, layer_weights.mlp.gate_proj,
                layer_weights.mlp.up_proj, views_.hidden[0],
                views_.hidden[1],
                layer_weights.post_attention_layernorm.data, kRmsEpsilon,
                views_.fp32_scratch, views_.fp32_scratch_elements,
                views_.hidden[2], views_.projection[0], views_.projection[1],
                stream_),
            "attention_residual_norm_mlp_gate_up_silu", layer)) {
      return fail_step(launch_failure);
    }
    const std::size_t trace_base =
        kReferenceHiddenSize + 2U * layer * kReferenceHiddenSize;
    if (options.capture_trace &&
        !copy_trace(views_.hidden[2], trace_base + kReferenceHiddenSize,
                    "trace_layer_residual", layer)) {
      return fail_step(launch_failure);
    }

    const bool is_final_layer =
        layer + 1U == kReferenceDecoderLayerCount;
    const std::uint16_t* const next_norm_weight =
        is_final_layer
            ? weights_->final_norm().data
            : weights_->layer(layer + 1U).input_layernorm.data;
    const char* const residual_norm_operation =
        is_final_layer ? "mlp_down_residual_final_norm"
                       : "mlp_down_residual_input_layernorm";
    if (!check_cuda(launch_mlp_down_residual_norm_to_bf16_cuda(
                        projection_backend_, layer_weights.mlp.down_proj,
                        views_.projection[0], views_.hidden[2],
                        next_norm_weight, kRmsEpsilon, views_.fp32_scratch,
                        views_.fp32_scratch_elements, views_.projection[1],
                        views_.hidden[0], views_.hidden[1], stream_),
                    residual_norm_operation, layer)) {
      return fail_step(launch_failure);
    }
    if (options.capture_trace &&
        !copy_trace(views_.projection[1], trace_base, "trace_layer_hidden",
                    layer)) {
      return fail_step(launch_failure);
    }
  }

  if (options.capture_trace &&
      !copy_trace(views_.hidden[1],
                  (1U + 2U * kReferenceDecoderLayerCount) *
                      kReferenceHiddenSize,
                  "trace_final_norm", kReferenceNoLayer)) {
    return fail_step(launch_failure);
  }

  if (options.compute_logits) {
    if (use_sm87_bf16_logits) {
      auto* const device_bf16_logits =
          reinterpret_cast<std::uint16_t*>(views_.fp32_scratch);
      if (!check_cuda(launch_projection_to_bf16_cuda(
                          projection_backend_, weights_->lm_head(),
                          views_.hidden[1], nullptr, 0U,
                          device_bf16_logits, stream_),
                      "lm_head_sm87_bf16", kReferenceNoLayer)) {
        return fail_step(launch_failure);
      }
      if (prediction_only) {
        constexpr std::size_t kGreedyWorkspaceBytes =
            kReferenceVocabularySize * sizeof(std::uint16_t) +
            kBf16GreedyArgmaxWorkspaceResults *
                sizeof(Bf16GreedyArgmaxResult);
        static_assert((kReferenceVocabularySize * sizeof(std::uint16_t)) %
                              alignof(Bf16GreedyArgmaxResult) ==
                          0U);
        if (views_.fp32_scratch_elements <
            (kGreedyWorkspaceBytes + sizeof(float) - 1U) / sizeof(float)) {
          return fail_step(runner_status(
              ReferenceRunnerError::kInvalidRequestState,
              "bf16_greedy_argmax_workspace"));
        }
        auto* const greedy_workspace =
            reinterpret_cast<Bf16GreedyArgmaxResult*>(
                device_bf16_logits + kReferenceVocabularySize);
        if (!check_cuda(launch_bf16_greedy_argmax_cuda(
                            device_bf16_logits, kReferenceVocabularySize,
                            greedy_workspace, stream_),
                        "bf16_greedy_argmax", kReferenceNoLayer) ||
            !check_cuda(
                static_cast<int>(cudaMemcpyAsync(
                    pinned_logits_, greedy_workspace,
                    sizeof(Bf16GreedyArgmaxResult),
                    cudaMemcpyDeviceToHost, stream)),
                "logits_prediction_d2h", kReferenceNoLayer)) {
          return fail_step(launch_failure);
        }
      } else if (!check_cuda(
                     static_cast<int>(cudaMemcpyAsync(
                         pinned_logits_, device_bf16_logits,
                         kReferenceVocabularySize * sizeof(std::uint16_t),
                         cudaMemcpyDeviceToHost, stream)),
                     "logits_bf16_d2h", kReferenceNoLayer)) {
        return fail_step(launch_failure);
      }
    } else if (!check_cuda(launch_projection_reference_cuda(
                               weights_->lm_head(), views_.hidden[1],
                               views_.fp32_scratch, stream_),
                           "lm_head", kReferenceNoLayer) ||
               !check_cuda(
                   static_cast<int>(cudaMemcpyAsync(
                       pinned_logits_, views_.fp32_scratch,
                       kReferenceVocabularySize * sizeof(float),
                       cudaMemcpyDeviceToHost, stream)),
                   "logits_d2h", kReferenceNoLayer)) {
      return fail_step(launch_failure);
    }
  }
  }

  if (graph_action == DecodeGraphP1Action::kCaptureOnly) {
    cudaGraph_t captured_graph = nullptr;
    const cudaError_t end_status =
        cudaStreamEndCapture(stream, &captured_graph);
    decode_graph_capture_active_ = false;
    if (end_status != cudaSuccess || captured_graph == nullptr) {
      if (captured_graph != nullptr) {
        (void)cudaGraphDestroy(captured_graph);
      }
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_end_capture", kReferenceNoLayer,
          static_cast<int>(end_status)));
    }
    const Clock::time_point capture_finished = Clock::now();

    constexpr std::size_t kMaximumDecodeGraphP1Nodes = 1'024U;
    std::size_t node_count = 0U;
    cudaError_t graph_status =
        cudaGraphGetNodes(captured_graph, nullptr, &node_count);
    if (graph_status != cudaSuccess || node_count == 0U ||
        node_count > kMaximumDecodeGraphP1Nodes) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          graph_status == cudaSuccess
              ? ReferenceRunnerError::kInvalidStepOptions
              : ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_node_capacity", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    std::array<cudaGraphNode_t, kMaximumDecodeGraphP1Nodes> nodes{};
    std::size_t returned_node_count = node_count;
    graph_status = cudaGraphGetNodes(captured_graph, nodes.data(),
                                     &returned_node_count);
    if (graph_status != cudaSuccess || returned_node_count != node_count) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_get_nodes", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }

    ReferenceDecodeGraphP1Stats stats;
    stats.position = position;
    stats.input_token_id = input_token_id;
    stats.node_count = node_count;
    stats.capture_enqueue_milliseconds =
        std::chrono::duration<double, std::milli>(
            capture_finished - decode_graph_prepare_started)
            .count();
    for (std::size_t index = 0U; index < node_count; ++index) {
      cudaGraphNodeType type{};
      graph_status = cudaGraphNodeGetType(nodes[index], &type);
      if (graph_status != cudaSuccess) {
        break;
      }
      if (type == cudaGraphNodeTypeKernel) {
        ++stats.kernel_node_count;
      } else if (type == cudaGraphNodeTypeMemcpy) {
        ++stats.memcpy_node_count;
      } else {
        ++stats.other_node_count;
      }
    }
    if (graph_status != cudaSuccess) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_classify_nodes", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }

    std::size_t root_count = 0U;
    graph_status =
        cudaGraphGetRootNodes(captured_graph, nullptr, &root_count);
    if (graph_status != cudaSuccess || root_count != 1U) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          graph_status == cudaSuccess
              ? ReferenceRunnerError::kInvalidStepOptions
              : ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_root_count", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    cudaGraphNode_t embedding_node = nullptr;
    std::size_t returned_root_count = 1U;
    graph_status = cudaGraphGetRootNodes(
        captured_graph, &embedding_node, &returned_root_count);
    cudaGraphNodeType root_type{};
    if (graph_status == cudaSuccess && returned_root_count == 1U &&
        embedding_node != nullptr) {
      graph_status = cudaGraphNodeGetType(embedding_node, &root_type);
    }
    if (graph_status != cudaSuccess || returned_root_count != 1U ||
        embedding_node == nullptr || root_type != cudaGraphNodeTypeKernel) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          graph_status == cudaSuccess
              ? ReferenceRunnerError::kInvalidStepOptions
              : ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_embedding_root", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }

    cudaKernelNodeParams embedding_launch{};
    graph_status =
        cudaGraphKernelNodeGetParams(embedding_node, &embedding_launch);
    if (graph_status != cudaSuccess) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_embedding_params", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    const bool valid_embedding_shape =
        embedding_launch.func != nullptr &&
        embedding_launch.gridDim.x == 20U &&
        embedding_launch.gridDim.y == 1U &&
        embedding_launch.gridDim.z == 1U &&
        embedding_launch.blockDim.x == 256U &&
        embedding_launch.blockDim.y == 1U &&
        embedding_launch.blockDim.z == 1U &&
        embedding_launch.sharedMemBytes == 0U &&
        embedding_launch.kernelParams != nullptr &&
        embedding_launch.extra == nullptr &&
        embedding_launch.kernelParams[0] != nullptr &&
        embedding_launch.kernelParams[1] != nullptr &&
        embedding_launch.kernelParams[2] != nullptr &&
        embedding_launch.kernelParams[3] != nullptr;
    if (!valid_embedding_shape) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_embedding_shape"));
    }
    const auto captured_table =
        *static_cast<const std::uint16_t* const*>(
            embedding_launch.kernelParams[0]);
    const std::size_t captured_offset =
        *static_cast<const std::size_t*>(
            embedding_launch.kernelParams[1]);
    const std::size_t captured_hidden_size =
        *static_cast<const std::size_t*>(
            embedding_launch.kernelParams[2]);
    const auto captured_output =
        *static_cast<std::uint16_t* const*>(
            embedding_launch.kernelParams[3]);
    if (captured_table != weights_->embed_tokens().weight ||
        captured_offset !=
            static_cast<std::size_t>(input_token_id) *
                kReferenceHiddenSize ||
        captured_hidden_size != kReferenceHiddenSize ||
        captured_output != views_.hidden[0]) {
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "decode_graph_p1_embedding_arguments"));
    }
    const Clock::time_point topology_finished = Clock::now();
    stats.topology_inspection_milliseconds =
        std::chrono::duration<double, std::milli>(
            topology_finished - capture_finished)
            .count();

    cudaGraphExec_t captured_exec = nullptr;
    const Clock::time_point instantiate_started = Clock::now();
    graph_status = cudaGraphInstantiate(
        &captured_exec, captured_graph, nullptr, nullptr, 0U);
    if (graph_status != cudaSuccess || captured_exec == nullptr) {
      if (captured_exec != nullptr) {
        (void)cudaGraphExecDestroy(captured_exec);
      }
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_instantiate", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    const Clock::time_point instantiate_finished = Clock::now();
    stats.instantiate_milliseconds =
        std::chrono::duration<double, std::milli>(
            instantiate_finished - instantiate_started)
            .count();

    const Clock::time_point upload_ready_started = Clock::now();
    graph_status = cudaGraphUpload(captured_exec, stream);
    if (graph_status == cudaSuccess) {
      graph_status = cudaStreamSynchronize(stream);
    }
    if (graph_status != cudaSuccess || decode_graph_slot == nullptr) {
      (void)cudaGraphExecDestroy(captured_exec);
      (void)cudaGraphDestroy(captured_graph);
      return fail_step(runner_status(
          graph_status == cudaSuccess
              ? ReferenceRunnerError::kInvalidRunner
              : ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_upload", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
    const Clock::time_point upload_ready_finished = Clock::now();
    stats.upload_ready_milliseconds =
        std::chrono::duration<double, std::milli>(
            upload_ready_finished - upload_ready_started)
            .count();

    DecodeGraphP1Slot prepared_slot;
    prepared_slot.graph = captured_graph;
    prepared_slot.exec = captured_exec;
    prepared_slot.embedding_node = embedding_node;
    prepared_slot.embedding_launch.function = embedding_launch.func;
    prepared_slot.embedding_launch.grid = {
        embedding_launch.gridDim.x, embedding_launch.gridDim.y,
        embedding_launch.gridDim.z};
    prepared_slot.embedding_launch.block = {
        embedding_launch.blockDim.x, embedding_launch.blockDim.y,
        embedding_launch.blockDim.z};
    prepared_slot.embedding_launch.shared_memory_bytes =
        embedding_launch.sharedMemBytes;
    stats.total_prepare_milliseconds =
        std::chrono::duration<double, std::milli>(
            Clock::now() - decode_graph_prepare_started)
            .count();
    prepared_slot.stats = stats;
    // Publish only after the replacement is completely uploaded and ready.
    // A failed recapture retains the prior slot handles; the runner's normal
    // failure contract still requires reset before any later reuse.
    (void)destroy_decode_graph_p1_slot(*decode_graph_slot);
    *decode_graph_slot = prepared_slot;
    ReferenceStepOutcome outcome;
    ReferenceStepResult result;
    result.position = position;
    result.input_token_id = input_token_id;
    outcome.value.emplace(std::move(result));
    return outcome;
  }

  if (graph_action == DecodeGraphP1Action::kReplay) {
    cudaKernelNodeParams embedding_params{};
    embedding_params.func = decode_graph_slot->embedding_launch.function;
    embedding_params.gridDim =
        dim3(decode_graph_slot->embedding_launch.grid[0],
             decode_graph_slot->embedding_launch.grid[1],
             decode_graph_slot->embedding_launch.grid[2]);
    embedding_params.blockDim =
        dim3(decode_graph_slot->embedding_launch.block[0],
             decode_graph_slot->embedding_launch.block[1],
             decode_graph_slot->embedding_launch.block[2]);
    embedding_params.sharedMemBytes =
        decode_graph_slot->embedding_launch.shared_memory_bytes;
    const std::uint16_t* embedding_table =
        weights_->embed_tokens().weight;
    std::size_t embedding_offset =
        static_cast<std::size_t>(input_token_id) * kReferenceHiddenSize;
    std::size_t embedding_hidden_size = kReferenceHiddenSize;
    std::uint16_t* embedding_output = views_.hidden[0];
    void* embedding_arguments[] = {
        &embedding_table, &embedding_offset, &embedding_hidden_size,
        &embedding_output};
    embedding_params.kernelParams = embedding_arguments;
    embedding_params.extra = nullptr;
    cudaError_t graph_status = cudaGraphExecKernelNodeSetParams(
        reinterpret_cast<cudaGraphExec_t>(decode_graph_slot->exec),
        reinterpret_cast<cudaGraphNode_t>(decode_graph_slot->embedding_node),
        &embedding_params);
    if (graph_status == cudaSuccess) {
      graph_status = cudaGraphLaunch(
          reinterpret_cast<cudaGraphExec_t>(decode_graph_slot->exec), stream);
    }
    if (graph_status != cudaSuccess) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "decode_graph_p1_update_or_launch", kReferenceNoLayer,
          static_cast<int>(graph_status)));
    }
  }

  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return fail_step(runner_status(ReferenceRunnerError::kCudaFailure,
                                   "step_synchronize", kReferenceNoLayer,
                                   static_cast<int>(sync_status)));
  }

  ReferenceStepResult result;
  result.position = position;
  result.input_token_id = input_token_id;
  if (options.compute_logits) {
    if (prediction_only && use_sm87_bf16_logits) {
      const auto& greedy =
          *static_cast<const Bf16GreedyArgmaxResult*>(pinned_logits_);
      if (greedy.has_nonfinite != 0U) {
        return fail_step(runner_status(
            ReferenceRunnerError::kNonFiniteLogits,
            "bf16_greedy_argmax"));
      }
      if (greedy.index >= kReferenceVocabularySize) {
        return fail_step(runner_status(
            ReferenceRunnerError::kCudaFailure,
            "bf16_greedy_argmax_result"));
      }
      result.prediction.emplace(
          ReferenceStepPrediction{greedy.index});
    } else {
      const reference_runner_detail::LogitsAnalysis analysis =
          use_sm87_bf16_logits
              ? reference_runner_detail::analyze_bf16_logits_bits(
                    static_cast<const std::uint16_t*>(pinned_logits_),
                    kReferenceVocabularySize)
              : (prediction_only
                     ? reference_runner_detail::analyze_bf16_argmax_in_place(
                           static_cast<float*>(pinned_logits_),
                           kReferenceVocabularySize)
                     : reference_runner_detail::
                           analyze_bf16_logits_in_place(
                               static_cast<float*>(pinned_logits_),
                               kReferenceVocabularySize));
      if (!analysis.ok()) {
        return fail_step(runner_status(
            ReferenceRunnerError::kNonFiniteLogits,
            "bf16_logits_analysis"));
      }
      if (prediction_only) {
        result.prediction.emplace(ReferenceStepPrediction{
            static_cast<std::uint32_t>(analysis.predicted_index)});
      } else {
        ReferenceStepLogits logits;
        logits.predicted_token_id =
            static_cast<std::uint32_t>(analysis.predicted_index);
        logits.chosen_logit = analysis.maximum;
        logits.max_log_probability = analysis.max_log_probability;
        logits.logsumexp = analysis.logsumexp;
        result.logits.emplace(logits);
      }
    }
  }

  const RequestOperationStatus commit_status = state_->commit_token();
  if (!commit_status) {
    return fail_step(runner_status(
        ReferenceRunnerError::kStateCommitFailure, "commit_token",
        kReferenceNoLayer, commit_status.cuda_error));
  }
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  const auto native_chunk64_final_snapshot_hook =
      g_prefill_gdn_chunk64_native_final_snapshot_hook;
  if (native_chunk64_final_snapshot_hook.callback != nullptr) {
    native_chunk64_final_snapshot_hook.callback(
        *state_, native_chunk64_final_snapshot_hook.context);
  }
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
  // step_synchronize above makes every recurrent-state write visible before
  // this admission-only observer runs. The callback has no return channel so
  // test instrumentation can never change the runner's error contract.
  const auto snapshot_hook =
      g_prefill_gdn_c16_norm_gate_admission_snapshot_hook;
  if (snapshot_hook.callback != nullptr) {
    snapshot_hook.callback(
        *state_, reference_runner_detail::
                     PrefillGdnC16NormGateAdmissionSnapshotStage::kStep,
        snapshot_hook.context);
  }
#endif
  if (options.capture_trace) {
    trace_valid_ = true;
    trace_position_ = position;
    trace_input_token_ = input_token_id;
  }
  if (options.measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    result.timing.emplace(
        ReferenceStepTiming{elapsed.count()});
  }

  ReferenceStepOutcome outcome;
  outcome.value.emplace(std::move(result));
  return outcome;
}

ReferenceRunner::PrefillTileExecutionControl
ReferenceRunner::legacy_prefill_tile_execution_control() noexcept {
  PrefillTileExecutionControl control;
  control.allow_experimental_gdn_b8_admission = true;
  control.allow_experimental_gdn_chunk64_native_admission = true;
  control.allow_experimental_gdn_chunk64_reference_admission = true;
  return control;
}

bool ReferenceRunner::is_legacy_prefill_tile_execution_control(
    const PrefillTileExecutionControl& control) noexcept {
  return !control.first_position_override.has_value() &&
         control.layer_begin == 0U &&
         control.layer_end == kReferenceDecoderLayerCount &&
         control.gather_embedding && control.apply_final_norm &&
         control.synchronize && control.commit_state && control.commit_route &&
         control.allow_scalar_m1_delegate &&
         control.allow_cross_layer_m32_fusion && control.emit_commit_hooks &&
         control.allow_experimental_gdn_b8_admission &&
         control.allow_experimental_gdn_chunk64_native_admission &&
         control.allow_experimental_gdn_chunk64_reference_admission &&
         !control.force_bound_nvfp4_marlin_prefill &&
         !control.force_bound_fp8_marlin_prefill &&
         !control.force_bound_gdn_chunk64_native_prefill;
}

ReferenceRunnerStatus ReferenceRunner::select_prefill_tile_execution(
    const PrefillTileExecutionControl& control,
    const std::uint32_t current_position,
    const std::uint32_t max_sequence_length,
    const std::uint32_t workspace_token_capacity,
    const std::size_t token_count,
    const ReferencePrefillTileOptions& options,
    PrefillTileExecutionSelection& selection) noexcept {
  selection = {};
  if (token_count == 0U ||
      token_count > kMaximumRequestPrefillChunkSize) {
    return runner_status(ReferenceRunnerError::kTokenOutOfRange,
                         "prefill_tile_tokens");
  }
  if (control.layer_begin >= control.layer_end ||
      control.layer_end > kReferenceDecoderLayerCount) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_tile_layer_range");
  }
  if (workspace_token_capacity == 0U ||
      workspace_token_capacity > kMaximumRequestPrefillChunkSize) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "prefill_tile_workspace_capacity");
  }

  const bool legacy =
      is_legacy_prefill_tile_execution_control(control);
  if (!legacy &&
      (control.allow_experimental_gdn_b8_admission ||
       control.allow_experimental_gdn_chunk64_native_admission ||
       control.allow_experimental_gdn_chunk64_reference_admission)) {
    return runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "prefill_tile_candidate_experimental_gdn_admission");
  }
  if (!legacy && !control.first_position_override.has_value()) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_tile_first_position_override");
  }
  // Enqueue-only candidate calls cannot publish retained rows or a host timer:
  // neither value represents completed device work until the outer executor's
  // single synchronization boundary.
  if (!legacy &&
      (options.retain_last_hidden_for_logits || options.measure_timing)) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_tile_candidate_completion_options");
  }
  if (!legacy && control.apply_final_norm) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_tile_partial_final_norm");
  }
  if (!legacy &&
      (control.synchronize || control.commit_state || control.commit_route ||
       control.allow_scalar_m1_delegate ||
       control.allow_cross_layer_m32_fusion || control.emit_commit_hooks)) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_tile_candidate_not_enqueue_only");
  }
  if (!legacy && control.layer_end != control.layer_begin + 1U) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_tile_candidate_layer_span");
  }
  if (control.gather_embedding && control.layer_begin != 0U) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_tile_embedding_layer_range");
  }

  const std::uint32_t first_position =
      control.first_position_override.value_or(current_position);
  if (first_position > max_sequence_length ||
      token_count >
          static_cast<std::size_t>(max_sequence_length - first_position) ||
      token_count > workspace_token_capacity) {
    return runner_status(ReferenceRunnerError::kCapacityExceeded,
                         "prefill_tile_capacity");
  }
  selection.first_position = first_position;
  selection.completed_position =
      first_position + static_cast<std::uint32_t>(token_count);
  selection.delegate_scalar_m1 =
      legacy && control.allow_scalar_m1_delegate && token_count == 1U;
  return {};
}

std::uint16_t ReferenceRunner::expected_prefill_layer_route_slots(
    const std::size_t layer) noexcept {
  const auto bit = [](const PrefillLayerRouteSlot slot) noexcept {
    return static_cast<std::uint16_t>(
        1U << static_cast<std::uint8_t>(slot));
  };
  const std::uint16_t common =
      bit(PrefillLayerRouteSlot::kNvFp4GateUp) |
      bit(PrefillLayerRouteSlot::kNvFp4Down) |
      bit(PrefillLayerRouteSlot::kQOrLinearQkv) |
      bit(PrefillLayerRouteSlot::kO);
  const model::LayerType layer_type =
      reference_runner_detail::expected_reference_layer_type(layer);
  if (layer_type == model::LayerType::kLinearAttention) {
    return common | bit(PrefillLayerRouteSlot::kLinearZ) |
           bit(PrefillLayerRouteSlot::kGdn);
  }
  if (layer_type == model::LayerType::kFullAttention) {
    return common | bit(PrefillLayerRouteSlot::kFullK) |
           bit(PrefillLayerRouteSlot::kFullV) |
           bit(PrefillLayerRouteSlot::kAttention);
  }
  return 0U;
}

ReferenceRunnerStatus ReferenceRunner::validate_prefill_layer_route_fragment(
    const PrefillLayerSegmentRouteFragment& fragment) noexcept {
  if (fragment.layer >= kReferenceDecoderLayerCount) {
    return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                         "prefill_layer_route_layer", fragment.layer);
  }
  if (fragment.token_count == 0U ||
      fragment.token_count > kLayerMajorRequestOperatorPanelCapacity ||
      fragment.first_position >
          std::numeric_limits<std::uint32_t>::max() -
              fragment.token_count) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_route_fragment_geometry",
                         fragment.layer);
  }
  const std::uint16_t expected =
      expected_prefill_layer_route_slots(fragment.layer);
  if (expected == 0U || fragment.recorded_slots != expected) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_route_fragment_slots",
                         fragment.layer);
  }
  constexpr std::uint8_t kKnownBoundaryMask = static_cast<std::uint8_t>(
      (1U << kPrefillForbiddenBoundaryCount) - 1U);
  if ((fragment.forbidden_boundaries &
       static_cast<std::uint8_t>(~kKnownBoundaryMask)) != 0U) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_route_fragment_boundaries",
                         fragment.layer);
  }
  for (std::size_t slot = 0U; slot < kPrefillLayerRouteSlotCount; ++slot) {
    if ((expected & static_cast<std::uint16_t>(1U << slot)) == 0U) {
      continue;
    }
    const PrefillRouteDisposition disposition =
        fragment.dispositions[slot];
    if (disposition != PrefillRouteDisposition::kProduction &&
        disposition != PrefillRouteDisposition::kExactFallback &&
        disposition != PrefillRouteDisposition::kForbidden) {
      return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                           "prefill_layer_route_fragment_disposition",
                           fragment.layer);
    }
  }
  return {};
}

ReferenceRunnerStatus
ReferenceRunner::validate_layer_wide_p40_prefill_layer_route_fragment(
    const PrefillLayerSegmentRouteFragment& fragment) noexcept {
  if (!layer_wide_p40_mlp_prefill_plan_enabled() ||
      fragment.layer >= kReferenceDecoderLayerCount ||
      fragment.first_position != 0U ||
      fragment.token_count != kLayerMajorPrefillLayerWideMlpP40Tokens) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_wide_p40_route_geometry",
                         fragment.layer);
  }
  PrefillLayerSegmentRouteFragment panel_geometry = fragment;
  panel_geometry.token_count = kLayerMajorPrefillOperatorPanelTokens;
  const ReferenceRunnerStatus slots =
      validate_prefill_layer_route_fragment(panel_geometry);
  if (!slots) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_wide_p40_route_slots",
                         fragment.layer);
  }
  return {};
}

ReferenceRunnerStatus ReferenceRunner::reduce_prefill_layer_route_fragment(
    const PrefillLayerSegmentRouteFragment& segment_fragment,
    PrefillLayerRouteReducer& reducer) noexcept {
  const ReferenceRunnerStatus segment_status =
      validate_prefill_layer_route_fragment(segment_fragment);
  if (!segment_status) {
    return segment_status;
  }
  if (segment_fragment.token_count > kMaximumRequestPrefillChunkSize) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_route_physical_segment",
                         segment_fragment.layer);
  }
  if (reducer.initialized) {
    const ReferenceRunnerStatus reducer_status =
        validate_prefill_layer_route_fragment(reducer.route_fragment);
    if (!reducer_status) {
      return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                           "prefill_layer_route_reducer_state",
                           segment_fragment.layer);
    }
    if (reducer.route_fragment.layer != segment_fragment.layer) {
      return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                           "prefill_layer_route_mixed_layer",
                           segment_fragment.layer);
    }
  } else {
    reducer.route_fragment = segment_fragment;
    reducer.initialized = true;
    return {};
  }

  PrefillLayerSegmentRouteFragment reduced = reducer.route_fragment;
  if (segment_fragment.first_position !=
      reduced.first_position + reduced.token_count) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_route_segment_order",
                         segment_fragment.layer);
  }
  if (reduced.token_count > kLayerMajorRequestOperatorPanelCapacity ||
      segment_fragment.token_count >
          kLayerMajorRequestOperatorPanelCapacity - reduced.token_count) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_route_segment_span",
                         segment_fragment.layer);
  }
  const auto weakest_disposition = [](
      const PrefillRouteDisposition left,
      const PrefillRouteDisposition right) noexcept {
    if (left == PrefillRouteDisposition::kForbidden ||
        right == PrefillRouteDisposition::kForbidden) {
      return PrefillRouteDisposition::kForbidden;
    }
    if (left == PrefillRouteDisposition::kExactFallback ||
        right == PrefillRouteDisposition::kExactFallback) {
      return PrefillRouteDisposition::kExactFallback;
    }
    return PrefillRouteDisposition::kProduction;
  };
  for (std::size_t slot = 0U; slot < kPrefillLayerRouteSlotCount; ++slot) {
    if ((reduced.recorded_slots &
         static_cast<std::uint16_t>(1U << slot)) == 0U) {
      continue;
    }
    reduced.dispositions[slot] = weakest_disposition(
        reduced.dispositions[slot],
        segment_fragment.dispositions[slot]);
  }
  reduced.forbidden_boundaries = static_cast<std::uint8_t>(
      reduced.forbidden_boundaries |
      segment_fragment.forbidden_boundaries);
  reduced.token_count += segment_fragment.token_count;
  reducer.route_fragment = reduced;
  return {};
}

ReferenceRunnerStatus ReferenceRunner::collapse_prefill_layer_route_fragment(
    const PrefillLayerSegmentRouteFragment& layer_fragment,
    PrefillRouteEvidence& layer_pass) noexcept {
  const ReferenceRunnerStatus fragment_status =
      validate_prefill_layer_route_fragment(layer_fragment);
  if (!fragment_status) {
    return fragment_status;
  }
  return collapse_validated_prefill_layer_route_fragment(layer_fragment,
                                                         layer_pass);
}

ReferenceRunnerStatus
ReferenceRunner::collapse_layer_wide_p40_prefill_layer_route_fragment(
    const PrefillLayerSegmentRouteFragment& layer_fragment,
    PrefillRouteEvidence& request_pass) noexcept {
  const ReferenceRunnerStatus fragment_status =
      validate_layer_wide_p40_prefill_layer_route_fragment(layer_fragment);
  if (!fragment_status) {
    return fragment_status;
  }
  return collapse_validated_prefill_layer_route_fragment(layer_fragment,
                                                         request_pass);
}

ReferenceRunnerStatus
ReferenceRunner::collapse_validated_prefill_layer_route_fragment(
    const PrefillLayerSegmentRouteFragment& layer_fragment,
    PrefillRouteEvidence& layer_pass) noexcept {
  PrefillRouteEvidence collapsed = layer_pass;
  const auto disposition = [&layer_fragment](
      const PrefillLayerRouteSlot slot) noexcept {
    return layer_fragment.dispositions[static_cast<std::size_t>(slot)];
  };
  const auto record = [&collapsed, &disposition](
      const PrefillLayerRouteSlot slot,
      const PrefillOperatorRole role) noexcept {
    return record_prefill_operator_route(collapsed, role,
                                         disposition(slot));
  };
  if (!record(PrefillLayerRouteSlot::kNvFp4GateUp,
              PrefillOperatorRole::kNvFp4GateUp) ||
      !record(PrefillLayerRouteSlot::kNvFp4Down,
              PrefillOperatorRole::kNvFp4Down) ||
      !record(PrefillLayerRouteSlot::kQOrLinearQkv,
              PrefillOperatorRole::kFp8Qkv) ||
      !record(PrefillLayerRouteSlot::kO, PrefillOperatorRole::kFp8O)) {
    return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                         "prefill_layer_route_collapse",
                         layer_fragment.layer);
  }
  const model::LayerType layer_type =
      reference_runner_detail::expected_reference_layer_type(
          layer_fragment.layer);
  if (layer_type == model::LayerType::kLinearAttention) {
    if (!record(PrefillLayerRouteSlot::kLinearZ,
                PrefillOperatorRole::kFp8Z) ||
        !record(PrefillLayerRouteSlot::kGdn,
                PrefillOperatorRole::kGdn)) {
      return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                           "prefill_layer_route_collapse_linear",
                           layer_fragment.layer);
    }
  } else if (layer_type == model::LayerType::kFullAttention) {
    if (!record(PrefillLayerRouteSlot::kFullK,
                PrefillOperatorRole::kFp8Qkv) ||
        !record(PrefillLayerRouteSlot::kFullV,
                PrefillOperatorRole::kFp8Qkv) ||
        !record(PrefillLayerRouteSlot::kAttention,
                PrefillOperatorRole::kAttention)) {
      return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                           "prefill_layer_route_collapse_full",
                           layer_fragment.layer);
    }
  } else {
    return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                         "prefill_layer_route_schedule",
                         layer_fragment.layer);
  }
  // Physical segments OR a boundary inside one logical layer. Collapse then
  // contributes one hit for each affected layer, preserving the legacy
  // request-level counter meaning without multiplying by C512 segment count.
  for (std::size_t boundary = 0U;
       boundary < kPrefillForbiddenBoundaryCount; ++boundary) {
    if ((layer_fragment.forbidden_boundaries &
        static_cast<std::uint8_t>(1U << boundary)) != 0U &&
        !record_prefill_forbidden_boundary(
            collapsed, static_cast<PrefillForbiddenBoundary>(boundary))) {
      return runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                           "prefill_layer_route_collapse_boundary",
                           layer_fragment.layer);
    }
  }
  layer_pass = collapsed;
  return {};
}

bool ReferenceRunner::same_prefill_execution_progress(
    const PrefillExecutionProgress& left,
    const PrefillExecutionProgress& right) noexcept {
  return left.kv_visible_end == right.kv_visible_end &&
         left.gdn_advanced_end == right.gdn_advanced_end &&
         left.completed_panels == right.completed_panels &&
         left.completed_mlp_phases == right.completed_mlp_phases &&
         left.completed_fill_panels == right.completed_fill_panels &&
         left.completed_prompt_core_phases ==
             right.completed_prompt_core_phases &&
         left.completed_drain_panels == right.completed_drain_panels &&
         left.next_layer == right.next_layer &&
         left.next_panel == right.next_panel &&
         left.final_hidden_ready == right.final_hidden_ready &&
         left.prefill_state_committed == right.prefill_state_committed;
}

bool ReferenceRunner::valid_prompt_wide_p40_whole_core_runner_contract(
    const PrefillExecutionPlan& immutable_topology,
    const RequestMemoryProfile memory_profile,
    const std::uint32_t max_sequence_length,
    const LayerMajorLayerExecutor executor,
    const LayerMajorPrefillProjectionTactic projection_tactic,
    const LayerMajorPrefillFullAttentionTactic full_attention_tactic) noexcept {
  const bool use_projection_reset =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40ProjectionReset;
  const bool use_packed_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40PackedProjection;
  const bool use_packed_nvfp4_v2 =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40PackedNvfp4V2;
  const bool use_any_packed_projection =
      use_packed_projection || use_packed_nvfp4_v2;
  const bool use_legacy_whole_core =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40WholeCore;
  if ((!use_projection_reset && !use_any_packed_projection &&
       !use_legacy_whole_core) ||
      (use_projection_reset
           ? !prompt_wide_p40_projection_reset_prefill_plan_enabled()
       : use_packed_projection
           ? !prompt_wide_p40_packed_projection_prefill_plan_enabled()
       : use_packed_nvfp4_v2
           ? !prompt_wide_p40_packed_nvfp4_v2_prefill_plan_enabled()
           : !prompt_wide_p40_whole_core_prefill_plan_enabled()) ||
      memory_profile != RequestMemoryProfile::kLayerMajorP40WholeCore ||
      executor != LayerMajorLayerExecutor::kOperatorPanel ||
      full_attention_tactic != LayerMajorPrefillFullAttentionTactic::
                                   kNativeFlashInferExactWholePrompt ||
      max_sequence_length !=
          kLayerMajorPrefillPromptWideP40RequestCapacityTokens ||
      !is_valid_unbound_layer_major_prefill_execution_plan(
          immutable_topology) ||
      immutable_topology.first_position != 0U ||
      immutable_topology.prompt_token_count !=
          kLayerMajorPrefillPromptWideP40Tokens ||
      immutable_topology.final_position !=
          kLayerMajorPrefillPromptWideP40Tokens ||
      immutable_topology.panel_count !=
          kLayerMajorPrefillPromptWideP40PanelCount ||
      immutable_topology.mlp_schedule.tactic !=
          (use_projection_reset
               ? LayerMajorPrefillMlpScheduleTactic::
                     kPromptWideP40ProjectionReset
           : use_packed_projection
               ? LayerMajorPrefillMlpScheduleTactic::
                     kPromptWideP40PackedProjection
           : use_packed_nvfp4_v2
               ? LayerMajorPrefillMlpScheduleTactic::
                     kPromptWideP40PackedNvfp4V2
               : LayerMajorPrefillMlpScheduleTactic::
                     kPromptWideP40WholeCore)) {
    return false;
  }
  if (use_projection_reset) {
    const PrefillP40ProjectionResetSchedulePlan& schedule =
        immutable_topology.projection_reset_schedule;
    if (!schedule.enabled ||
        schedule.input_preparation_panel_count_per_layer !=
            kLayerMajorPrefillPromptWideP40PanelCount ||
        schedule.prompt_core_phase_count_per_layer != 1U ||
        schedule.persistent_mlp_phase_count_per_layer != 1U ||
        schedule.panel_token_count !=
            kLayerMajorPrefillPromptWideP40PanelTokens ||
        schedule.projection_m_tokens !=
            kLayerMajorPrefillPromptWideP40Tokens ||
        schedule.request_capacity_tokens !=
            kLayerMajorPrefillPromptWideP40RequestCapacityTokens ||
        schedule.route_pass_count != 1U ||
        schedule.fp8_grouped_input_launches_per_layer != 1U ||
        schedule.fp8_output_launches_per_layer != 1U ||
        schedule.fp8_physical_launches_per_request != 128U ||
        schedule.fp8_tensor_role_hits_per_request != 208U ||
        schedule.nvfp4_gate_up_launches_per_layer != 1U ||
        schedule.nvfp4_down_launches_per_layer != 1U ||
        schedule.nvfp4_physical_launches_per_request != 128U ||
        !schedule.fp8_grouped_full_prompt_input_required ||
        !schedule.fp8_full_prompt_output_required ||
        !schedule.nvfp4_full_prompt_required ||
        !schedule.internal_m_segmentation_forbidden ||
        !schedule.production_accuracy_required ||
        !schedule.approximate_numerics_forbidden || !schedule.mtp_forbidden ||
        !schedule.cublaslt_forbidden) {
      return false;
    }
  }
  if (use_packed_projection) {
    const PrefillP40PackedProjectionSchedulePlan& schedule =
        immutable_topology.packed_projection_schedule;
    if (!schedule.enabled ||
        schedule.input_preparation_panel_count_per_layer !=
            kLayerMajorPrefillPromptWideP40PanelCount ||
        schedule.prompt_core_phase_count_per_layer != 1U ||
        schedule.packed_mlp_phase_count_per_layer != 1U ||
        schedule.panel_token_count !=
            kLayerMajorPrefillPromptWideP40PanelTokens ||
        schedule.projection_m_tokens !=
            kLayerMajorPrefillPromptWideP40Tokens ||
        schedule.request_capacity_tokens !=
            kLayerMajorPrefillPromptWideP40RequestCapacityTokens ||
        schedule.route_pass_count != 1U ||
        schedule.fp8_physical_launches_per_request != 128U ||
        schedule.fp8_tensor_role_hits_per_request != 208U ||
        schedule.nvfp4_physical_launches_per_request != 128U ||
        schedule.authenticated_artifact_count !=
            kernels::kSm87P40PackedProjectionArtifactCount ||
        schedule.authenticated_source_count !=
            kernels::kSm87P40PackedProjectionSourceIdentityCount ||
        schedule.stream_k_slice_count != 1U ||
        !schedule.packed_operands_retained_to_register_decode ||
        !schedule.role_specific_tactics_required ||
        !schedule.request_time_repack_forbidden ||
        !schedule.request_time_tactic_selection_forbidden ||
        !schedule.internal_m_segmentation_forbidden ||
        !schedule.production_accuracy_required ||
        !schedule.approximate_numerics_forbidden || !schedule.mtp_forbidden ||
        !schedule.cublaslt_forbidden) {
      return false;
    }
  }
  if (use_packed_nvfp4_v2) {
    const PrefillP40PackedNvfp4V2SchedulePlan& schedule =
        immutable_topology.packed_nvfp4_v2_schedule;
    if (!schedule.enabled ||
        schedule.input_preparation_panel_count_per_layer !=
            kLayerMajorPrefillPromptWideP40PanelCount ||
        schedule.prompt_core_phase_count_per_layer != 1U ||
        schedule.packed_mlp_phase_count_per_layer != 1U ||
        schedule.panel_token_count !=
            kLayerMajorPrefillPromptWideP40PanelTokens ||
        schedule.projection_m_tokens !=
            kLayerMajorPrefillPromptWideP40Tokens ||
        schedule.request_capacity_tokens !=
            kLayerMajorPrefillPromptWideP40RequestCapacityTokens ||
        schedule.route_pass_count != 1U ||
        schedule.fp8_physical_launches_per_request !=
            kLayerMajorPrefillPackedNvfp4V2Fp8PhysicalLaunchesPerRequest ||
        schedule.fp8_tensor_role_hits_per_request !=
            kLayerMajorPrefillPackedNvfp4V2Fp8TensorRoleHitsPerRequest ||
        schedule.nvfp4_physical_launches_per_request != 128U ||
        schedule.authenticated_artifact_count !=
            kLayerMajorPrefillPackedNvfp4V2ArtifactCount ||
        schedule.authenticated_source_count !=
            kLayerMajorPrefillPackedNvfp4V2AuthenticatedSourceCount ||
        schedule.stream_k_slice_count != 1U ||
        !schedule.packed_operands_retained_to_register_decode ||
        !schedule.role_specific_tactics_required ||
        !schedule.shape_specific_gate_up_required ||
        !schedule.shape_specific_down_required ||
        !schedule.request_time_repack_forbidden ||
        !schedule.request_time_tactic_selection_forbidden ||
        !schedule.internal_m_segmentation_forbidden ||
        !schedule.production_accuracy_required ||
        !schedule.approximate_numerics_forbidden || !schedule.mtp_forbidden ||
        !schedule.cublaslt_forbidden) {
      return false;
    }
  }
  const PrefillWholeCoreSchedulePlan& schedule =
      immutable_topology.whole_core_schedule;
  if (!use_projection_reset && !use_any_packed_projection &&
      (!schedule.enabled ||
      schedule.fill_panel_phase_count_per_layer !=
          kLayerMajorPrefillPromptWideP40PanelCount ||
      schedule.prompt_core_phase_count_per_layer != 1U ||
      schedule.drain_panel_phase_count_per_layer !=
          kLayerMajorPrefillPromptWideP40PanelCount ||
      schedule.persistent_mlp_phase_count_per_layer != 1U ||
      schedule.panel_token_count !=
          kLayerMajorPrefillPromptWideP40PanelTokens ||
      schedule.prompt_core_token_count !=
          kLayerMajorPrefillPromptWideP40Tokens ||
      schedule.route_pass_count != 1U ||
      !schedule.fp8_single_launch_per_projection_required ||
      !schedule.bf16_ab_prompt_wide_required ||
      !schedule.gdn_prompt_wide_required ||
       !schedule.flashinfer_whole_prompt_required)) {
    return false;
  }
  for (std::size_t panel_index = 0U;
       panel_index < immutable_topology.panel_count; ++panel_index) {
    const PrefillOperatorPanel& panel =
        immutable_topology.panels[panel_index];
    const std::uint32_t expected_first = static_cast<std::uint32_t>(
        panel_index * kLayerMajorPrefillPromptWideP40PanelTokens);
    if (panel.ordinal != panel_index ||
        panel.first_position != expected_first ||
        panel.token_count != kLayerMajorPrefillPromptWideP40PanelTokens ||
        panel.end_position !=
            expected_first + kLayerMajorPrefillPromptWideP40PanelTokens) {
      return false;
    }
  }
  return true;
}

ReferenceRunnerStatus
ReferenceRunner::enqueue_prompt_wide_p40_whole_core_fill_panel(
    const std::size_t layer, const PrefillOperatorPanel& panel,
    const ReferenceLayerMajorRequestViews& request_views,
    const PromptWideP40ProjectionPackage projection_package,
    std::size_t& fp8_projection_hits,
    std::size_t& fp8_physical_launches) noexcept {
#if !defined(Q3X_ENABLE_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION) || \
    !defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  (void)layer;
  (void)panel;
  (void)request_views;
  (void)projection_package;
  (void)fp8_projection_hits;
  (void)fp8_physical_launches;
  return runner_status(ReferenceRunnerError::kInvalidDependency,
                       "prefill_whole_core_fill_build_inventory");
#else
  const bool use_projection_reset =
      projection_package == PromptWideP40ProjectionPackage::kProjectionResetV11;
  const bool use_packed_projection =
      projection_package == PromptWideP40ProjectionPackage::kPackedV1;
  const auto exact_bf16_matrix = [](const DeviceMatrixView& view,
                                    const std::size_t rows,
                                    const std::size_t columns) noexcept {
    return view.storage.device_data != nullptr &&
           view.storage.element_size_bytes == sizeof(std::uint16_t) &&
           view.row_capacity == rows && view.columns == columns &&
           view.row_stride_elements == columns &&
           view.storage.element_capacity == rows * columns;
  };
  if (layer >= kReferenceDecoderLayerCount ||
      panel.token_count != kPromptWideP40WholeCorePanelTokens ||
      panel.first_position !=
          panel.ordinal * kPromptWideP40WholeCorePanelTokens ||
      panel.ordinal >= kPromptWideP40WholeCorePanelCount ||
      request_views.descriptor.profile !=
          RequestMemoryProfile::kLayerMajorP40WholeCore ||
      request_views.descriptor.layout !=
          LayerMajorRequestLayout::kP40WholeCorePromptWide ||
      !exact_bf16_matrix(request_views.prompt_residual_bf16,
                         kPromptWideP40WholeCoreRequestCapacityTokens,
                         kReferenceHiddenSize)) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "prefill_whole_core_fill_contract", layer);
  }

  ExactPrefillProjectionWorkspace workspace{};
  if (!use_projection_reset && !use_packed_projection) {
    const DeviceBufferView& reduction =
        request_views.legacy_c512.fp32_scratch;
    const DeviceMatrixView& lock_storage =
        request_views.legacy_c512.projection_bf16[3U];
    workspace.reduction = static_cast<float*>(reduction.device_data);
    workspace.reduction_elements = reduction.element_capacity;
    workspace.locks = reinterpret_cast<std::int32_t*>(
        lock_storage.storage.device_data);
    workspace.lock_bytes = static_cast<std::size_t>(
        lock_storage.storage.byte_size);
    if (!valid_exact_prefill_projection_workspace(workspace)) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_whole_core_fill_workspace", layer);
    }
  }

  const DecoderLayerWeights& layer_weights = weights_->layer(layer);
  const std::size_t offset = panel.first_position;
  auto* const residual = static_cast<std::uint16_t*>(
      request_views.prompt_residual_bf16.storage.device_data) +
      offset * kReferenceHiddenSize;
  const auto check = [layer](const int status,
                             const char* const operation) noexcept {
    return status == static_cast<int>(cudaSuccess)
               ? ReferenceRunnerStatus{}
               : runner_status(ReferenceRunnerError::kCudaFailure,
                               operation, layer, status);
  };
  if (layer == 0U) {
    const DeviceMatrixView& token_ids =
        request_views.p40_whole_core.prompt_token_ids_u32;
    if (token_ids.storage.device_data == nullptr ||
        token_ids.storage.element_size_bytes != sizeof(std::uint32_t) ||
        token_ids.row_capacity != kPromptWideP40WholeCorePromptTokens ||
        token_ids.columns != 1U || token_ids.row_stride_elements != 1U) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_whole_core_token_ids", layer);
    }
    const auto* const panel_token_ids = static_cast<const std::uint32_t*>(
        token_ids.storage.device_data) + offset;
    const ReferenceRunnerStatus embedding = check(
        launch_embedding_gather_prompt_reference_cuda(
            weights_->embed_tokens().weight, kReferenceVocabularySize,
            kReferenceHiddenSize, panel_token_ids, panel.token_count,
            residual, stream_),
        "prefill_whole_core_embedding");
    if (!embedding) {
      return embedding;
    }
  }

  const model::LayerType expected =
      reference_runner_detail::expected_reference_layer_type(layer);
  if (expected == model::LayerType::kLinearAttention) {
    const auto* const attention =
        std::get_if<LinearAttentionWeights>(&layer_weights.attention);
    const LayerMajorP40WholeCoreLinearPhaseViews& phase =
        request_views.p40_whole_core.linear;
    if (attention == nullptr ||
        !exact_bf16_matrix(phase.normalized_input_bf16,
                           kPromptWideP40WholeCorePromptTokens,
                           kReferenceHiddenSize) ||
        !exact_bf16_matrix(phase.raw_qkv_bf16,
                           kPromptWideP40WholeCorePromptTokens,
                           kLinearQkvElements) ||
        !exact_bf16_matrix(phase.z_bf16,
                           kPromptWideP40WholeCorePromptTokens,
                           kLinearValueElements)) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_whole_core_linear_fill_views", layer);
    }
    auto* const normalized = static_cast<std::uint16_t*>(
        phase.normalized_input_bf16.storage.device_data) +
        offset * kReferenceHiddenSize;
    auto* const raw_qkv = static_cast<std::uint16_t*>(
        phase.raw_qkv_bf16.storage.device_data) +
        offset * kLinearQkvElements;
    auto* const z = static_cast<std::uint16_t*>(
        phase.z_bf16.storage.device_data) + offset * kLinearValueElements;
    ReferenceRunnerStatus status = check(
        launch_headwise_centered_rms_norm_reference_cuda(
            residual, layer_weights.input_layernorm.data, panel.token_count,
            kReferenceHiddenSize, kRmsEpsilon, normalized, stream_),
        "prefill_whole_core_linear_fill_norm");
    if (!status) {
      return status;
    }
    if (use_projection_reset || use_packed_projection) {
      return {};
    }
    status = check(launch_prompt_wide_p40_fp8_projection(
                       projection_backend_, attention->in_proj_qkv,
                       normalized, raw_qkv, panel.token_count, workspace,
                       fp8_physical_launches, stream_),
                   "prefill_whole_core_linear_fill_qkv");
    if (!status) {
      return status;
    }
    ++fp8_projection_hits;
    status = check(launch_prompt_wide_p40_fp8_projection(
                       projection_backend_, attention->in_proj_z, normalized,
                       z, panel.token_count, workspace,
                       fp8_physical_launches, stream_),
                   "prefill_whole_core_linear_fill_z");
    if (!status) {
      return status;
    }
    ++fp8_projection_hits;
    return {};
  }
  if (expected == model::LayerType::kFullAttention) {
    const auto* const attention =
        std::get_if<FullAttentionWeights>(&layer_weights.attention);
    const LayerMajorP40WholeCoreFullAttentionPhaseViews& phase =
        request_views.p40_whole_core.full_attention;
    const RequestLayerSlot& slot = request_views.descriptor.layers[layer];
    if (attention == nullptr || slot.slot >= kRequestFullLayerCount ||
        !exact_bf16_matrix(phase.normalized_input_bf16,
                           kPromptWideP40WholeCorePromptTokens,
                           kReferenceHiddenSize) ||
        !exact_bf16_matrix(phase.raw_q_gate_bf16,
                           kPromptWideP40WholeCorePromptTokens,
                           kFullQGateElements) ||
        !exact_bf16_matrix(phase.processed_q_bf16,
                           kPromptWideP40WholeCorePromptTokens,
                           kFullQueryElements) ||
        !exact_bf16_matrix(phase.packed_gate_bf16,
                           kPromptWideP40WholeCorePromptTokens,
                           kFullQueryElements) ||
        request_views.persistent.key_cache_bf16[slot.slot].device_data ==
            nullptr ||
        request_views.persistent.value_cache_bf16[slot.slot].device_data ==
            nullptr ||
        !can_launch_full_attention_preprocess_prompt_wide_p8000(
            panel.first_position, panel.token_count)) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_whole_core_attention_fill_views", layer);
    }
    auto* const normalized = static_cast<std::uint16_t*>(
        phase.normalized_input_bf16.storage.device_data) +
        offset * kReferenceHiddenSize;
    auto* const raw_q_gate = static_cast<std::uint16_t*>(
        phase.raw_q_gate_bf16.storage.device_data) +
        offset * kFullQGateElements;
    auto* const processed_q = static_cast<std::uint16_t*>(
        phase.processed_q_bf16.storage.device_data) +
        offset * kFullQueryElements;
    auto* const packed_gate = static_cast<std::uint16_t*>(
        phase.packed_gate_bf16.storage.device_data) +
        offset * kFullQueryElements;
    auto* const key = static_cast<std::uint16_t*>(
        request_views.persistent.key_cache_bf16[slot.slot].device_data) +
        offset * kFullKvElements;
    auto* const value = static_cast<std::uint16_t*>(
        request_views.persistent.value_cache_bf16[slot.slot].device_data) +
        offset * kFullKvElements;
    ReferenceRunnerStatus status = check(
        launch_headwise_centered_rms_norm_reference_cuda(
            residual, layer_weights.input_layernorm.data, panel.token_count,
            kReferenceHiddenSize, kRmsEpsilon, normalized, stream_),
        "prefill_whole_core_attention_fill_norm");
    if (!status) {
      return status;
    }
    if (use_projection_reset || use_packed_projection) {
      return {};
    }
    const auto project = [&](const LinearWeight& weight,
                             std::uint16_t* const output,
                             const char* const operation) noexcept {
      ReferenceRunnerStatus projection = check(
          launch_prompt_wide_p40_fp8_projection(
              projection_backend_, weight, normalized, output,
              panel.token_count, workspace, fp8_physical_launches, stream_),
          operation);
      if (projection) {
        ++fp8_projection_hits;
      }
      return projection;
    };
    status = project(attention->q_proj, raw_q_gate,
                     "prefill_whole_core_attention_fill_q");
    if (!status) {
      return status;
    }
    status = project(attention->k_proj, key,
                     "prefill_whole_core_attention_fill_k");
    if (!status) {
      return status;
    }
    status = project(attention->v_proj, value,
                     "prefill_whole_core_attention_fill_v");
    if (!status) {
      return status;
    }
    return check(
        launch_full_attention_preprocess_24_4_256_64_prompt_wide_p8000_cuda(
            raw_q_gate, key, attention->q_norm.data,
            attention->k_norm.data, kRmsEpsilon, processed_q, packed_gate,
            static_cast<const float*>(
                request_views.persistent.rope_cos_fp32.device_data),
            static_cast<const float*>(
                request_views.persistent.rope_sin_fp32.device_data),
            panel.first_position, panel.token_count, stream_),
        "prefill_whole_core_attention_fill_preprocess");
  }
  return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                       "prefill_whole_core_fill_layer_schedule", layer);
#endif
}

ReferenceRunnerStatus
ReferenceRunner::enqueue_prompt_wide_p40_whole_core_prompt_core(
    const std::size_t layer,
    const ReferenceLayerMajorRequestViews& request_views,
    const PromptWideP40ProjectionPackage projection_package,
    std::size_t& fp8_projection_hits,
    std::size_t& fp8_physical_launches) noexcept {
#if !defined(Q3X_ENABLE_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION) || \
    !defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION) || \
    !defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION) || \
    !defined(Q3X_ENABLE_GDN_PROMPT_WIDE_CHUNK_GRAPH_ADMISSION)
  (void)layer;
  (void)request_views;
  (void)projection_package;
  (void)fp8_projection_hits;
  (void)fp8_physical_launches;
  return runner_status(ReferenceRunnerError::kInvalidDependency,
                       "prefill_whole_core_prompt_core_build_inventory");
#else
  const bool use_projection_reset =
      projection_package == PromptWideP40ProjectionPackage::kProjectionResetV11;
  const bool use_packed_projection =
      projection_package == PromptWideP40ProjectionPackage::kPackedV1;
  if (layer >= kReferenceDecoderLayerCount) {
    return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                         "prefill_whole_core_prompt_core_layer", layer);
  }
  const auto check = [layer](const int status,
                             const char* const operation) noexcept {
    return status == static_cast<int>(cudaSuccess)
               ? ReferenceRunnerStatus{}
               : runner_status(ReferenceRunnerError::kCudaFailure,
                               operation, layer, status);
  };
  const DecoderLayerWeights& layer_weights = weights_->layer(layer);
  const model::LayerType expected =
      reference_runner_detail::expected_reference_layer_type(layer);
  if (expected == model::LayerType::kLinearAttention) {
    const auto* const attention =
        std::get_if<LinearAttentionWeights>(&layer_weights.attention);
    const auto* const first = attention == nullptr
                                  ? nullptr
                                  : std::get_if<Bf16LinearWeight>(
                                        &attention->in_proj_a);
    const auto* const second = attention == nullptr
                                   ? nullptr
                                   : std::get_if<Bf16LinearWeight>(
                                         &attention->in_proj_b);
    const RequestLayerSlot& slot = request_views.descriptor.layers[layer];
    const LayerMajorP40WholeCoreLinearPhaseViews& phase =
        request_views.p40_whole_core.linear;
    if (attention == nullptr || first == nullptr || second == nullptr ||
        first->weight == nullptr || second->weight == nullptr ||
        first->input_size != kReferenceHiddenSize ||
        second->input_size != kReferenceHiddenSize ||
        first->output_size != kLinearScalarElements ||
        second->output_size != kLinearScalarElements ||
        slot.slot >= kRequestLinearLayerCount ||
        phase.prompt_wide_workspace.device_data == nullptr ||
        phase.prompt_wide_workspace.byte_size <
            gdn_prefill_prompt_wide_chunk_graph_detail::workspace_bytes() ||
        request_views.persistent.conv_state_bf16[slot.slot].device_data ==
            nullptr ||
        request_views.persistent.gdn_state_bf16[slot.slot].device_data ==
            nullptr ||
        !gdn_prefill_prompt_wide_chunk_graph_detail::supports(
            kPromptWideP40WholeCorePromptTokens)) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_whole_core_linear_core_views", layer);
    }
    const auto* const normalized = static_cast<const std::uint16_t*>(
        phase.normalized_input_bf16.storage.device_data);
    auto* const a = static_cast<std::uint16_t*>(
        phase.a_bf16.storage.device_data);
    auto* const b = static_cast<std::uint16_t*>(
        phase.b_bf16.storage.device_data);
    if (use_projection_reset || use_packed_projection) {
      const LinearWeight* const group_weights[2U] = {
          &attention->in_proj_qkv, &attention->in_proj_z};
      std::uint16_t* const group_outputs[2U] = {
          static_cast<std::uint16_t*>(phase.raw_qkv_bf16.storage.device_data),
          static_cast<std::uint16_t*>(phase.z_bf16.storage.device_data)};
      const int projection_status =
          use_packed_projection
              ? launch_p40_packed_fp8_group(
                    projection_backend_, group_weights, group_outputs, 2U,
                    normalized, fp8_physical_launches, stream_)
              : launch_p40_projection_reset_fp8_group(
                    projection_backend_, group_weights, group_outputs, 2U,
                    normalized, fp8_physical_launches, stream_);
      if (projection_status != static_cast<int>(cudaSuccess)) {
        return runner_status(ReferenceRunnerError::kCudaFailure,
                             use_packed_projection
                                 ? "prefill_p40_packed_linear_qkvz"
                                 : "prefill_projection_reset_linear_qkvz",
                             layer,
                             projection_status);
      }
      fp8_projection_hits += 2U;
    }
    ReferenceRunnerStatus status = check(
        kernels::launch_sm87_bf16_ab_prompt_wide_p40_cuda(
            first->weight, second->weight, normalized,
            kPromptWideP40WholeCorePromptTokens, a, b, stream_),
        "prefill_whole_core_linear_ab");
    if (!status) {
      return status;
    }
    return check(
        gdn_prefill_prompt_wide_chunk_graph_detail::launch(
            phase.prompt_wide_workspace.device_data,
            static_cast<std::size_t>(phase.prompt_wide_workspace.byte_size),
            static_cast<const std::uint16_t*>(
                phase.raw_qkv_bf16.storage.device_data),
            kPromptWideP40WholeCorePromptTokens, attention->conv1d.data,
            static_cast<std::uint16_t*>(
                request_views.persistent.conv_state_bf16[slot.slot]
                    .device_data),
            static_cast<std::uint16_t*>(
                phase.conv_qkv_bf16.storage.device_data),
            a, b, attention->a_log.data, attention->dt_bias.data,
            static_cast<const std::uint16_t*>(
                request_views.persistent.gdn_state_bf16[slot.slot]
                    .device_data),
            static_cast<std::uint16_t*>(
                request_views.persistent.gdn_state_bf16[slot.slot]
                    .device_data),
            kRmsEpsilon, attention->norm.data,
            static_cast<const std::uint16_t*>(
                phase.z_bf16.storage.device_data),
            kRmsEpsilon,
            static_cast<std::uint16_t*>(
                phase.output_bf16.storage.device_data),
            stream_),
        "prefill_whole_core_linear_gdn");
  }
  if (expected == model::LayerType::kFullAttention) {
    const auto* const attention =
        std::get_if<FullAttentionWeights>(&layer_weights.attention);
    const RequestLayerSlot& slot = request_views.descriptor.layers[layer];
    const LayerMajorP40WholeCoreFullAttentionPhaseViews& phase =
        request_views.p40_whole_core.full_attention;
    if (attention == nullptr || slot.slot >= kRequestFullLayerCount ||
        !can_launch_bulk_causal_gqa_flashinfer_exact_whole_prompt(
            0U, kPromptWideP40WholeCorePromptTokens)) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_whole_core_attention_core_views", layer);
    }
    if (use_projection_reset || use_packed_projection) {
      const LinearWeight* const group_weights[3U] = {
          &attention->q_proj, &attention->k_proj, &attention->v_proj};
      std::uint16_t* const group_outputs[3U] = {
          static_cast<std::uint16_t*>(
              phase.raw_q_gate_bf16.storage.device_data),
          static_cast<std::uint16_t*>(
              request_views.persistent.key_cache_bf16[slot.slot].device_data),
          static_cast<std::uint16_t*>(
              request_views.persistent.value_cache_bf16[slot.slot].device_data)};
      const auto* const normalized = static_cast<const std::uint16_t*>(
          phase.normalized_input_bf16.storage.device_data);
      const int projection_status =
          use_packed_projection
              ? launch_p40_packed_fp8_group(
                    projection_backend_, group_weights, group_outputs, 3U,
                    normalized, fp8_physical_launches, stream_)
              : launch_p40_projection_reset_fp8_group(
                    projection_backend_, group_weights, group_outputs, 3U,
                    normalized, fp8_physical_launches, stream_);
      if (projection_status != static_cast<int>(cudaSuccess)) {
        return runner_status(ReferenceRunnerError::kCudaFailure,
                             use_packed_projection
                                 ? "prefill_p40_packed_full_qkv"
                                 : "prefill_projection_reset_full_qkv",
                             layer,
                             projection_status);
      }
      fp8_projection_hits += 3U;
      for (std::size_t panel_index = 0U;
           panel_index < kPromptWideP40WholeCorePanelCount; ++panel_index) {
        const std::size_t offset =
            panel_index * kPromptWideP40WholeCorePanelTokens;
        const int preprocess_status =
            launch_full_attention_preprocess_24_4_256_64_prompt_wide_p8000_cuda(
                static_cast<const std::uint16_t*>(
                    phase.raw_q_gate_bf16.storage.device_data) +
                    offset * kFullQGateElements,
                static_cast<std::uint16_t*>(
                    request_views.persistent.key_cache_bf16[slot.slot]
                        .device_data) +
                    offset * kFullKvElements,
                attention->q_norm.data, attention->k_norm.data, kRmsEpsilon,
                static_cast<std::uint16_t*>(
                    phase.processed_q_bf16.storage.device_data) +
                    offset * kFullQueryElements,
                static_cast<std::uint16_t*>(
                    phase.packed_gate_bf16.storage.device_data) +
                    offset * kFullQueryElements,
                static_cast<const float*>(
                    request_views.persistent.rope_cos_fp32.device_data),
                static_cast<const float*>(
                    request_views.persistent.rope_sin_fp32.device_data),
                offset, kPromptWideP40WholeCorePanelTokens, stream_);
        if (preprocess_status != static_cast<int>(cudaSuccess)) {
          return runner_status(ReferenceRunnerError::kCudaFailure,
                               use_packed_projection
                                   ? "prefill_p40_packed_full_preprocess"
                                   : "prefill_projection_reset_full_preprocess",
                               layer, preprocess_status);
        }
      }
    }
    return check(
        launch_bulk_causal_gqa_sigmoid_gate_24_4_256_flashinfer_exact_whole_prompt_fixed_cuda(
            static_cast<const std::uint16_t*>(
                phase.processed_q_bf16.storage.device_data),
            static_cast<const std::uint16_t*>(
                request_views.persistent.key_cache_bf16[slot.slot]
                    .device_data),
            static_cast<const std::uint16_t*>(
                request_views.persistent.value_cache_bf16[slot.slot]
                    .device_data),
            static_cast<const std::uint16_t*>(
                phase.packed_gate_bf16.storage.device_data),
            0U, kPromptWideP40WholeCorePromptTokens,
            static_cast<std::uint16_t*>(
                phase.core_output_bf16.storage.device_data),
            stream_),
        "prefill_whole_core_attention_core");
  }
  return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                       "prefill_whole_core_prompt_core_schedule", layer);
#endif
}

ReferenceRunnerStatus
ReferenceRunner::enqueue_prompt_wide_p40_whole_core_drain_panel(
    const std::size_t layer, const PrefillOperatorPanel& panel,
    const ReferenceLayerMajorRequestViews& request_views,
    const PromptWideP40ProjectionPackage projection_package,
    std::size_t& fp8_projection_hits,
    std::size_t& fp8_physical_launches) noexcept {
#if !defined(Q3X_ENABLE_PROMPT_WIDE_P40_WHOLE_CORE_ADMISSION) || \
    !defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  (void)layer;
  (void)panel;
  (void)request_views;
  (void)projection_package;
  (void)fp8_projection_hits;
  (void)fp8_physical_launches;
  return runner_status(ReferenceRunnerError::kInvalidDependency,
                       "prefill_whole_core_drain_build_inventory");
#else
  const bool use_projection_reset =
      projection_package == PromptWideP40ProjectionPackage::kProjectionResetV11;
  const bool use_packed_projection =
      projection_package == PromptWideP40ProjectionPackage::kPackedV1;
  if (layer >= kReferenceDecoderLayerCount ||
      panel.ordinal >= kPromptWideP40WholeCorePanelCount ||
      panel.first_position !=
          panel.ordinal * kPromptWideP40WholeCorePanelTokens ||
      panel.token_count != kPromptWideP40WholeCorePanelTokens) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_whole_core_drain_geometry", layer);
  }
  ExactPrefillProjectionWorkspace workspace{};
  if (!use_projection_reset && !use_packed_projection) {
    const DeviceBufferView& reduction =
        request_views.legacy_c512.fp32_scratch;
    const DeviceMatrixView& lock_storage =
        request_views.legacy_c512.projection_bf16[3U];
    workspace.reduction = static_cast<float*>(reduction.device_data);
    workspace.reduction_elements = reduction.element_capacity;
    workspace.locks = reinterpret_cast<std::int32_t*>(
        lock_storage.storage.device_data);
    workspace.lock_bytes = static_cast<std::size_t>(
        lock_storage.storage.byte_size);
    if (!valid_exact_prefill_projection_workspace(workspace)) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_whole_core_drain_workspace", layer);
    }
  }
  const DecoderLayerWeights& layer_weights = weights_->layer(layer);
  const std::size_t offset = panel.first_position;
  const LinearWeight* output_weight = nullptr;
  const std::uint16_t* core_output_base = nullptr;
  std::uint16_t* branch_output_base = nullptr;
  if (reference_runner_detail::expected_reference_layer_type(layer) ==
      model::LayerType::kLinearAttention) {
    const auto* const attention =
        std::get_if<LinearAttentionWeights>(&layer_weights.attention);
    if (attention != nullptr) {
      output_weight = &attention->out_proj;
      core_output_base = static_cast<const std::uint16_t*>(
          request_views.p40_whole_core.linear.output_bf16.storage.device_data);
      branch_output_base = static_cast<std::uint16_t*>(
          request_views.p40_whole_core.linear.branch_output_bf16.storage
              .device_data);
    }
  } else {
    const auto* const attention =
        std::get_if<FullAttentionWeights>(&layer_weights.attention);
    if (attention != nullptr) {
      output_weight = &attention->o_proj;
      core_output_base = static_cast<const std::uint16_t*>(
          request_views.p40_whole_core.full_attention.core_output_bf16.storage
              .device_data);
      branch_output_base = static_cast<std::uint16_t*>(
          request_views.p40_whole_core.full_attention.branch_output_bf16
              .storage.device_data);
    }
  }
  auto* const residual = static_cast<std::uint16_t*>(
      request_views.prompt_residual_bf16.storage.device_data) +
      offset * kReferenceHiddenSize;
  if (output_weight == nullptr || core_output_base == nullptr ||
      branch_output_base == nullptr || residual == nullptr) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "prefill_whole_core_drain_views", layer);
  }
  const std::size_t core_columns =
      reference_runner_detail::expected_reference_layer_type(layer) ==
              model::LayerType::kLinearAttention
          ? kLinearValueElements
          : kFullQueryElements;
  const std::uint16_t* const core_output =
      core_output_base + offset * core_columns;
  std::uint16_t* const branch_output =
      branch_output_base + offset * kReferenceHiddenSize;
  int status = static_cast<int>(cudaSuccess);
  if (use_projection_reset || use_packed_projection) {
    if (panel.ordinal == 0U) {
      const LinearWeight* const group_weights[1U] = {output_weight};
      std::uint16_t* const group_outputs[1U] = {branch_output_base};
      status = use_packed_projection
                   ? launch_p40_packed_fp8_group(
                         projection_backend_, group_weights, group_outputs,
                         1U, core_output_base, fp8_physical_launches, stream_)
                   : launch_p40_projection_reset_fp8_group(
                         projection_backend_, group_weights, group_outputs,
                         1U, core_output_base, fp8_physical_launches, stream_);
      if (status == static_cast<int>(cudaSuccess)) {
        ++fp8_projection_hits;
      }
    }
  } else {
    status = launch_prompt_wide_p40_fp8_projection(
        projection_backend_, *output_weight, core_output, branch_output,
        panel.token_count, workspace, fp8_physical_launches, stream_);
    if (status == static_cast<int>(cudaSuccess)) {
      ++fp8_projection_hits;
    }
  }
  if (status != static_cast<int>(cudaSuccess)) {
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "prefill_whole_core_drain_o", layer, status);
  }
  status = launch_residual_add_reference_cuda(
      residual, branch_output,
      panel.token_count * kReferenceHiddenSize, residual, stream_);
  if (status != static_cast<int>(cudaSuccess)) {
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "prefill_whole_core_drain_residual", layer,
                         status);
  }
  return {};
#endif
}

ReferencePrefillTileOutcome ReferenceRunner::prefill_prefix_tile(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const ReferencePrefillTileOptions& options) noexcept {
  if (whole_request_prefill_active()) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_legacy_tile_active"));
  }
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }
  // Preserve the public contract: any attempted new tile invalidates a prior
  // retained row before runner, token, capacity, or control validation.
  retained_prefill_hidden_valid_ = false;
  if (!static_cast<bool>(*this)) {
    return fail_prefill_tile(
        runner_status(ReferenceRunnerError::kInvalidRunner,
                      "prefill_prefix_tile"));
  }
  if (poisoned_) {
    ReferencePrefillTileOutcome outcome;
    outcome.status = runner_status(ReferenceRunnerError::kPoisoned,
                                   "prefill_prefix_tile");
    return outcome;
  }
  if (input_token_ids == nullptr || token_count == 0U ||
      token_count > kMaximumRequestPrefillChunkSize) {
    return fail_prefill_tile(
        runner_status(ReferenceRunnerError::kTokenOutOfRange,
                      "prefill_tile_tokens"));
  }
  for (std::size_t token = 0U; token < token_count; ++token) {
    if (input_token_ids[token] >= kReferenceVocabularySize) {
      return fail_prefill_tile(
          runner_status(ReferenceRunnerError::kTokenOutOfRange,
                        "prefill_tile_token"));
    }
  }
  const PrefillTileExecutionControl control =
      legacy_prefill_tile_execution_control();
  PrefillTileExecutionSelection execution;
  const ReferenceRunnerStatus execution_status =
      select_prefill_tile_execution(
          control, state_->current_position(), state_->max_sequence_length(),
          state_->plan().prefill_chunk_size, token_count, options, execution);
  if (!execution_status) {
    return fail_prefill_tile(execution_status);
  }
  const std::uint32_t first_position = execution.first_position;

  if (execution.delegate_scalar_m1) {
    ReferenceStepOptions step_options;
    step_options.compute_logits = false;
    step_options.capture_trace = false;
    step_options.measure_timing = options.measure_timing;
    ReferenceStepOutcome step_outcome = step(input_token_ids[0], step_options);
    if (!step_outcome) {
      ReferencePrefillTileOutcome outcome;
      outcome.status = step_outcome.status;
      return outcome;
    }
    ReferencePrefillTileResult tile;
    tile.step_count = 1U;
    tile.steps[0] = std::move(*step_outcome.value);
    tile.timing = tile.steps[0].timing;
    const ReferenceRunnerStatus route_status =
        record_scalar_prefill_route_fallback();
    if (!route_status) {
      return fail_prefill_tile(route_status);
    }
    if (options.retain_last_hidden_for_logits) {
      retained_prefill_hidden_valid_ = true;
      retained_prefill_position_ = tile.steps[0].position;
      retained_prefill_input_token_ = tile.steps[0].input_token_id;
      retained_prefill_hidden_row_ = 0U;
    }
    ReferencePrefillTileOutcome outcome;
    outcome.value.emplace(std::move(tile));
    return outcome;
  }

  // The public tile validates its token IDs exactly once. The enqueue core is
  // also the future layer-major physical-segment seam and therefore assumes
  // its enclosing whole-request admission has already performed that check.
  const PrefillLayerSegmentEnqueueResult enqueued =
      enqueue_prefill_layer_segment(input_token_ids, token_count,
                                    first_position, control, views_);
  if (!enqueued) {
    return fail_prefill_tile(enqueued.status);
  }

  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kCudaFailure, "prefill_tile_synchronize",
        kReferenceNoLayer, static_cast<int>(sync_status)));
  }
  const std::uint32_t committed_length = execution.completed_position;
  const RequestOperationStatus commit_status =
      state_->set_sequence_length(committed_length);
  if (!commit_status) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kStateCommitFailure,
        "prefill_tile_commit", kReferenceNoLayer,
        commit_status.cuda_error));
  }
  if (!commit_prefill_route_layer_pass(
          prefill_route_evidence_,
          enqueued.route_fragment.legacy_layer_pass)) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kRouteEvidenceFailure,
        "prefill_tile_route_commit", kReferenceNoLayer));
  }
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  const auto native_chunk64_snapshot_hook =
      g_prefill_gdn_chunk64_native_snapshot_hook;
  if (native_chunk64_snapshot_hook.callback != nullptr) {
    native_chunk64_snapshot_hook.callback(
        *state_, native_chunk64_snapshot_hook.context);
  }
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  const auto chunk64_snapshot_hook =
      g_prefill_gdn_chunk64_reference_snapshot_hook;
  if (chunk64_snapshot_hook.callback != nullptr) {
    chunk64_snapshot_hook.callback(*state_, chunk64_snapshot_hook.context);
  }
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
  // The synchronize above completed the full C512 state transition; observe
  // it only after the logical sequence length was committed.
  const auto snapshot_hook =
      g_prefill_gdn_c16_norm_gate_admission_snapshot_hook;
  if (snapshot_hook.callback != nullptr) {
    snapshot_hook.callback(
        *state_, reference_runner_detail::
                     PrefillGdnC16NormGateAdmissionSnapshotStage::kPrefixTile,
        snapshot_hook.context);
  }
#endif

  ReferencePrefillTileResult tile;
  tile.step_count = token_count;
  for (std::size_t token = 0U; token < token_count; ++token) {
    tile.steps[token].position =
        first_position + static_cast<std::uint32_t>(token);
    tile.steps[token].input_token_id = input_token_ids[token];
  }
  if (options.measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    tile.timing.emplace(ReferenceStepTiming{elapsed.count()});
  }
  if (options.retain_last_hidden_for_logits) {
    retained_prefill_hidden_valid_ = true;
    retained_prefill_position_ = committed_length - 1U;
    retained_prefill_input_token_ = input_token_ids[token_count - 1U];
    retained_prefill_hidden_row_ = token_count - 1U;
  }
  ReferencePrefillTileOutcome outcome;
  outcome.value.emplace(std::move(tile));
  return outcome;
}

ReferenceWholeRequestPrefillOutcome
ReferenceRunner::prefill_whole_request_layer_major_compatibility_core(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const PrefillExecutionPlan& immutable_topology,
    const ReferenceWholeRequestPrefillOptions& options) noexcept {
  return prefill_whole_request_layer_major_core(
      input_token_ids, token_count, immutable_topology,
      LayerMajorLayerExecutor::kCompatibilitySegments,
      LayerMajorPrefillProjectionTactic::kExactSegmentedC512,
      LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512, options);
}

ReferenceWholeRequestPrefillOutcome
ReferenceRunner::prefill_whole_request_layer_major_panel_core(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const PrefillExecutionPlan& immutable_topology,
    const LayerMajorPrefillProjectionTactic projection_tactic,
    const LayerMajorPrefillFullAttentionTactic full_attention_tactic,
    const ReferenceWholeRequestPrefillOptions& options) noexcept {
  return prefill_whole_request_layer_major_core(
      input_token_ids, token_count, immutable_topology,
      LayerMajorLayerExecutor::kOperatorPanel, projection_tactic,
      full_attention_tactic, options);
}

ReferenceWholeRequestPrefillOutcome
ReferenceRunner::prefill_whole_request_layer_major_core(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const PrefillExecutionPlan& immutable_topology,
    const LayerMajorLayerExecutor executor,
    const LayerMajorPrefillProjectionTactic projection_tactic,
    const LayerMajorPrefillFullAttentionTactic full_attention_tactic,
    const ReferenceWholeRequestPrefillOptions& options) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }

  if (whole_request_prefill_active()) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_stage_active"));
  }
  retained_prefill_hidden_valid_ = false;
  trace_valid_ = false;
  if (!static_cast<bool>(*this)) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "whole_request_prefill"));
  }
  if (poisoned_) {
    ReferenceWholeRequestPrefillOutcome outcome;
    outcome.status = runner_status(ReferenceRunnerError::kPoisoned,
                                   "whole_request_prefill");
    return outcome;
  }
  const bool any_prompt_wide_p40_whole_core_identity =
      state_->memory_profile() == RequestMemoryProfile::kLayerMajorP40WholeCore ||
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40WholeCore ||
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40ProjectionReset ||
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40PackedProjection ||
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40PackedNvfp4V2 ||
      full_attention_tactic == LayerMajorPrefillFullAttentionTactic::
                                   kNativeFlashInferExactWholePrompt ||
      immutable_topology.mlp_schedule.tactic ==
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore ||
      immutable_topology.mlp_schedule.tactic ==
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40ProjectionReset ||
      immutable_topology.mlp_schedule.tactic ==
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40PackedProjection ||
      immutable_topology.mlp_schedule.tactic ==
          LayerMajorPrefillMlpScheduleTactic::kPromptWideP40PackedNvfp4V2;
  const bool prompt_wide_p40_whole_core =
      valid_prompt_wide_p40_whole_core_runner_contract(
          immutable_topology, state_->memory_profile(),
          state_->max_sequence_length(), executor, projection_tactic,
          full_attention_tactic);
  const bool use_projection_reset =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40ProjectionReset;
  const bool use_packed_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40PackedProjection;
  const bool use_packed_nvfp4_v2 =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativePromptWideP40PackedNvfp4V2;
  const PromptWideP40ProjectionPackage projection_package =
      use_projection_reset
          ? PromptWideP40ProjectionPackage::kProjectionResetV11
      : use_packed_projection
          ? PromptWideP40ProjectionPackage::kPackedV1
      : use_packed_nvfp4_v2
          ? PromptWideP40ProjectionPackage::kPackedNvfp4V2
          : PromptWideP40ProjectionPackage::kWholeCoreV10;
  if (!layer_major_request_views_.has_value() ||
      any_prompt_wide_p40_whole_core_identity !=
          prompt_wide_p40_whole_core ||
      (!prompt_wide_p40_whole_core &&
       state_->memory_profile() != RequestMemoryProfile::kLayerMajorC8192) ||
      !is_valid_layer_major_prefill_projection_tactic(projection_tactic) ||
      !is_valid_layer_major_prefill_full_attention_tactic(
          full_attention_tactic) ||
      (executor == LayerMajorLayerExecutor::kCompatibilitySegments &&
       (projection_tactic !=
            LayerMajorPrefillProjectionTactic::kExactSegmentedC512 ||
        full_attention_tactic !=
            LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512))) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kInvalidRequestState,
        "whole_request_prefill_memory_profile"));
  }
  const bool bounded_submission_window =
      options.cancellation_probe != nullptr;
  if ((!bounded_submission_window &&
       options.cancellation_context != nullptr) ||
      (bounded_submission_window &&
       (whole_request_submission_events_[0U] == nullptr ||
        whole_request_submission_events_[1U] == nullptr))) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_cancellation_options"));
  }
  if (input_token_ids == nullptr || token_count == 0U ||
      token_count > kLayerMajorPrefillMaximumSequenceTokens) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kTokenOutOfRange,
        "whole_request_prefill_tokens"));
  }
  for (std::size_t token = 0U; token < token_count; ++token) {
    if (input_token_ids[token] >= kReferenceVocabularySize) {
      return fail_whole_request_prefill(runner_status(
          ReferenceRunnerError::kTokenOutOfRange,
          "whole_request_prefill_token"));
    }
  }
  if (!is_valid_unbound_layer_major_prefill_execution_plan(
          immutable_topology) ||
      immutable_topology.first_position != state_->current_position() ||
      immutable_topology.prompt_token_count != token_count ||
      immutable_topology.final_position > state_->max_sequence_length() ||
      immutable_topology.final_commit.expected_initial_sequence_length !=
          state_->current_position() ||
      immutable_topology.final_commit.committed_sequence_length !=
          immutable_topology.final_position) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_topology"));
  }
  const bool layer_wide_p40_mlp =
      immutable_topology.mlp_schedule.tactic ==
      LayerMajorPrefillMlpScheduleTactic::kLayerWideP40ExactFullM;
  if (layer_wide_p40_mlp !=
          (projection_tactic == LayerMajorPrefillProjectionTactic::
                                    kNativeNvfp4PersistentP40LayerWideMlp) ||
      (layer_wide_p40_mlp &&
       (executor != LayerMajorLayerExecutor::kOperatorPanel ||
        immutable_topology.first_position != 0U ||
        immutable_topology.prompt_token_count !=
            kLayerMajorPrefillLayerWideMlpP40Tokens ||
        immutable_topology.panel_count != 5U ||
        state_->max_sequence_length() !=
            kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens))) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_mlp_schedule"));
  }
  if (projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeNvfp4TrueLargeMOperatorPanel ||
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeNvfp4G2D2LargeMOperatorPanel ||
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeNvfp4PersistentP40LayerWideMlp) {
    for (std::size_t panel_index = 0U;
         panel_index < immutable_topology.panel_count; ++panel_index) {
      const std::uint32_t panel_m =
          immutable_topology.panels[panel_index].token_count;
      if (!is_nvfp4_true_large_m_prefill_panel_tokens(panel_m)) {
        return fail_whole_request_prefill(runner_status(
            ReferenceRunnerError::kInvalidStepOptions,
            "whole_request_prefill_nvfp4_true_large_m_panel_geometry"));
      }
    }
  }
  if (full_attention_tactic == LayerMajorPrefillFullAttentionTactic::
                                   kNativeFlashInferExactPanel) {
    for (std::size_t panel_index = 0U;
         panel_index < immutable_topology.panel_count; ++panel_index) {
      const PrefillOperatorPanel& panel =
          immutable_topology.panels[panel_index];
      if (panel.token_count < kPrefillPhysicalSegmentM32Tokens ||
          !can_launch_bulk_causal_gqa_flashinfer_exact_panel(
              panel.first_position, panel.token_count)) {
        return fail_whole_request_prefill(runner_status(
            ReferenceRunnerError::kInvalidStepOptions,
            "whole_request_prefill_flashinfer_panel_geometry"));
      }
    }
  }
  if (!prefill_route_evidence_.request_active ||
      prefill_route_evidence_.complete ||
      prefill_route_evidence_.error != PrefillRouteEvidenceError::kNone) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kRouteEvidenceFailure,
        "whole_request_prefill_route_state"));
  }

  ReferenceLayerMajorRequestViews& layer_major =
      *layer_major_request_views_;
  DeviceMatrixView& prompt_residual =
      layer_major.prompt_residual_bf16;
  DeviceMatrixView& fixed_final_hidden =
      layer_major.final_hidden_bf16;
  if (prompt_residual.storage.device_data == nullptr ||
      prompt_residual.columns != kReferenceHiddenSize ||
      prompt_residual.row_stride_elements != kReferenceHiddenSize ||
      prompt_residual.row_capacity < immutable_topology.final_position ||
      fixed_final_hidden.storage.device_data == nullptr ||
      fixed_final_hidden.row_capacity != 1U ||
      fixed_final_hidden.columns != kReferenceHiddenSize ||
      fixed_final_hidden.row_stride_elements != kReferenceHiddenSize) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kInvalidRequestState,
        "whole_request_prefill_typed_views"));
  }
  auto* const prompt_residual_base = static_cast<std::uint16_t*>(
      prompt_residual.storage.device_data);
  auto* const final_hidden = static_cast<std::uint16_t*>(
      fixed_final_hidden.storage.device_data);

  const auto cancellation_requested = [&options]() noexcept {
    return options.cancellation_probe != nullptr &&
           options.cancellation_probe(options.cancellation_context);
  };
  if (cancellation_requested()) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kCancelled,
        "whole_request_prefill_cancelled"));
  }

  whole_request_prefill_stage_.phase =
      WholeRequestPrefillStagePhase::kExecuting;
  std::array<PrefillRouteEvidence,
             kLayerMajorPrefillMaximumPanelCount>
      panel_layer_passes{};
  PrefillRouteEvidence layer_wide_p40_request_pass;
  ReferencePrefillTileOptions segment_options;
  std::size_t submission_window_in_flight = 0U;
  std::size_t submission_window_oldest_slot = 0U;
  std::size_t submission_window_next_slot = 0U;
  std::size_t submission_window_retirements = 0U;
  std::array<std::size_t, kWholeRequestSubmissionWindowSlots>
      submission_window_layers{};
  std::array<std::size_t, kWholeRequestSubmissionWindowSlots>
      submission_window_phases{};
  std::size_t operator_panel_executor_hits = 0U;
  std::size_t native_group_q64_panel_hits = 0U;
  std::size_t native_group_q128_v4_panel_hits = 0U;
  std::size_t native_flashinfer_exact_panel_hits = 0U;
  std::size_t generic_qt2_hits = 0U;
  std::size_t segmented_panel_projection_hits = 0U;
  std::size_t segmented_panel_projection_physical_launches = 0U;
  std::size_t native_large_m_projection_hits = 0U;
  std::size_t native_large_m_projection_bulk_hits = 0U;
  std::size_t native_large_m_projection_oracle_partial_hits = 0U;
  std::size_t native_large_m_projection_physical_launches = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_hits = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_bulk_hits = 0U;
  std::size_t
      nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_physical_launches = 0U;
  std::size_t native_nvfp4_true_large_m_projection_hits = 0U;
  std::size_t native_nvfp4_true_large_m_gate_up_hits = 0U;
  std::size_t native_nvfp4_true_large_m_down_hits = 0U;
  std::size_t native_nvfp4_true_large_m_physical_launches = 0U;
  std::size_t layer_wide_p40_mlp_layer_hits = 0U;
  std::size_t persistent_p40_nvfp4_gate_up_hits = 0U;
  std::size_t persistent_p40_nvfp4_down_residual_hits = 0U;
  std::size_t persistent_p40_nvfp4_physical_launches = 0U;
  std::size_t persistent_p40_fp8_projection_hits = 0U;
  std::size_t persistent_p40_fp8_projection_bulk_hits = 0U;
  std::size_t persistent_p40_fp8_projection_oracle_partial_hits = 0U;
  std::size_t persistent_p40_fp8_projection_physical_launches = 0U;
  std::size_t prompt_wide_p40_whole_core_layer_hits = 0U;
  std::size_t prompt_wide_p40_fill_panel_hits = 0U;
  std::size_t prompt_wide_p40_prompt_core_hits = 0U;
  std::size_t prompt_wide_p40_drain_panel_hits = 0U;
  std::size_t prompt_wide_p40_fp8_projection_hits = 0U;
  std::size_t prompt_wide_p40_fp8_projection_physical_launches = 0U;
  std::size_t prompt_wide_p40_bf16_ab_hits = 0U;
  std::size_t prompt_wide_p40_gdn_hits = 0U;
  std::size_t native_flashinfer_exact_whole_prompt_hits = 0U;
  std::size_t packed_nvfp4_v2_gate_up_hits = 0U;
  std::size_t packed_nvfp4_v2_down_hits = 0U;
  std::size_t packed_nvfp4_v2_physical_launches = 0U;
  const auto retire_oldest_submission = [&]() noexcept {
    if (!bounded_submission_window || submission_window_in_flight == 0U) {
      return ReferenceRunnerStatus{};
    }
    const std::size_t retired_slot = submission_window_oldest_slot;
    const std::size_t retired_layer = submission_window_layers[retired_slot];
    const std::size_t retired_phase = submission_window_phases[retired_slot];
    const cudaError_t status = cudaEventSynchronize(
        reinterpret_cast<cudaEvent_t>(whole_request_submission_events_[
            retired_slot]));
    if (status != cudaSuccess) {
      return runner_status(
          ReferenceRunnerError::kCudaFailure,
          "whole_request_prefill_submission_wait", kReferenceNoLayer,
          static_cast<int>(status));
    }
    submission_window_oldest_slot =
        (submission_window_oldest_slot + 1U) %
        kWholeRequestSubmissionWindowSlots;
    --submission_window_in_flight;
    ++submission_window_retirements;
    if (cancellation_requested()) {
      const char* cancellation_operation =
          "whole_request_prefill_panel_phase_cancelled";
      if (prompt_wide_p40_whole_core) {
        if (retired_phase < kPromptWideP40WholeCoreFillPhases) {
          cancellation_operation =
              "whole_request_prefill_whole_core_fill_cancelled";
        } else if (retired_phase ==
                   kPromptWideP40WholeCorePromptCorePhase) {
          cancellation_operation =
              "whole_request_prefill_whole_core_prompt_core_cancelled";
        } else if (retired_phase <
                   kPromptWideP40WholeCorePersistentMlpPhase) {
          cancellation_operation =
              "whole_request_prefill_whole_core_drain_cancelled";
        } else {
          cancellation_operation =
              "whole_request_prefill_whole_core_mlp_cancelled";
        }
      } else if (retired_phase == immutable_topology.panel_count) {
        cancellation_operation =
            "whole_request_prefill_mlp_phase_cancelled";
      }
      return runner_status(
          ReferenceRunnerError::kCancelled, cancellation_operation,
          retired_layer, 0,
          static_cast<std::uint64_t>(submission_window_retirements));
    }
    return ReferenceRunnerStatus{};
  };
  const auto record_submission = [&](const std::size_t layer,
                                     const std::size_t phase) noexcept {
    if (!bounded_submission_window) {
      return ReferenceRunnerStatus{};
    }
    const std::size_t slot = submission_window_next_slot;
    const cudaError_t record_status = cudaEventRecord(
        reinterpret_cast<cudaEvent_t>(whole_request_submission_events_[slot]),
        reinterpret_cast<cudaStream_t>(stream_));
    if (record_status != cudaSuccess) {
      return runner_status(ReferenceRunnerError::kCudaFailure,
                           "whole_request_prefill_submission_record", layer,
                           static_cast<int>(record_status));
    }
    submission_window_layers[slot] = layer;
    submission_window_phases[slot] = phase;
    if (submission_window_in_flight == 0U) {
      submission_window_oldest_slot = slot;
    }
    submission_window_next_slot =
        (submission_window_next_slot + 1U) %
        kWholeRequestSubmissionWindowSlots;
    ++submission_window_in_flight;
    return ReferenceRunnerStatus{};
  };

  if (prompt_wide_p40_whole_core) {
    const DeviceMatrixView& token_ids =
        layer_major.p40_whole_core.prompt_token_ids_u32;
    if (token_ids.storage.device_data == nullptr ||
        token_ids.storage.element_size_bytes != sizeof(std::uint32_t) ||
        token_ids.row_capacity != kPromptWideP40WholeCorePromptTokens ||
        token_ids.columns != 1U || token_ids.row_stride_elements != 1U) {
      return fail_whole_request_prefill(runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "whole_request_prefill_whole_core_token_ids"));
    }
    const cudaError_t token_copy_status = cudaMemcpyAsync(
        token_ids.storage.device_data, input_token_ids,
        kPromptWideP40WholeCorePromptTokens * sizeof(std::uint32_t),
        cudaMemcpyHostToDevice, reinterpret_cast<cudaStream_t>(stream_));
    if (token_copy_status != cudaSuccess) {
      return fail_whole_request_prefill(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "whole_request_prefill_whole_core_token_copy", kReferenceNoLayer,
          static_cast<int>(token_copy_status)));
    }
  }

  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount;
       ++layer) {
    if (prompt_wide_p40_whole_core) {
      const auto make_submission_room = [&]() noexcept {
        if (bounded_submission_window &&
            submission_window_in_flight ==
                kWholeRequestSubmissionWindowSlots) {
          return retire_oldest_submission();
        }
        return ReferenceRunnerStatus{};
      };
      for (std::size_t panel_index = 0U;
           panel_index < kPromptWideP40WholeCorePanelCount; ++panel_index) {
        ReferenceRunnerStatus status = make_submission_room();
        if (!status) {
          return fail_whole_request_prefill(status);
        }
        status = enqueue_prompt_wide_p40_whole_core_fill_panel(
            layer, immutable_topology.panels[panel_index], layer_major,
            projection_package,
            prompt_wide_p40_fp8_projection_hits,
            prompt_wide_p40_fp8_projection_physical_launches);
        if (!status) {
          return fail_whole_request_prefill(status);
        }
        ++prompt_wide_p40_fill_panel_hits;
        status = record_submission(layer, panel_index);
        if (!status) {
          return fail_whole_request_prefill(status);
        }
      }

      ReferenceRunnerStatus status = make_submission_room();
      if (!status) {
        return fail_whole_request_prefill(status);
      }
      status = enqueue_prompt_wide_p40_whole_core_prompt_core(
          layer, layer_major, projection_package,
          prompt_wide_p40_fp8_projection_hits,
          prompt_wide_p40_fp8_projection_physical_launches);
      if (!status) {
        return fail_whole_request_prefill(status);
      }
      ++prompt_wide_p40_prompt_core_hits;
      if (reference_runner_detail::expected_reference_layer_type(layer) ==
          model::LayerType::kLinearAttention) {
        ++prompt_wide_p40_bf16_ab_hits;
        ++prompt_wide_p40_gdn_hits;
      } else {
        ++native_flashinfer_exact_whole_prompt_hits;
      }
      status = record_submission(layer,
                                 kPromptWideP40WholeCorePromptCorePhase);
      if (!status) {
        return fail_whole_request_prefill(status);
      }

      for (std::size_t panel_index = 0U;
           panel_index < kPromptWideP40WholeCorePanelCount; ++panel_index) {
        status = make_submission_room();
        if (!status) {
          return fail_whole_request_prefill(status);
        }
        status = enqueue_prompt_wide_p40_whole_core_drain_panel(
            layer, immutable_topology.panels[panel_index], layer_major,
            projection_package,
            prompt_wide_p40_fp8_projection_hits,
            prompt_wide_p40_fp8_projection_physical_launches);
        if (!status) {
          return fail_whole_request_prefill(status);
        }
        ++prompt_wide_p40_drain_panel_hits;
        status = record_submission(
            layer, kPromptWideP40WholeCoreDrainPhaseBegin + panel_index);
        if (!status) {
          return fail_whole_request_prefill(status);
        }
      }

      status = make_submission_room();
      if (!status) {
        return fail_whole_request_prefill(status);
      }
      status = enqueue_layer_wide_p40_mlp(layer, layer_major,
                                          projection_package);
      if (!status) {
        return fail_whole_request_prefill(status);
      }
      ++layer_wide_p40_mlp_layer_hits;
      ++persistent_p40_nvfp4_gate_up_hits;
      ++persistent_p40_nvfp4_down_residual_hits;
      persistent_p40_nvfp4_physical_launches += 2U;
      if (use_packed_nvfp4_v2) {
        ++packed_nvfp4_v2_gate_up_hits;
        ++packed_nvfp4_v2_down_hits;
        packed_nvfp4_v2_physical_launches += 2U;
      }
      ++prompt_wide_p40_whole_core_layer_hits;

      PrefillLayerSegmentRouteFragment full_layer_fragment;
      full_layer_fragment.layer = layer;
      full_layer_fragment.first_position = 0U;
      full_layer_fragment.token_count =
          kPromptWideP40WholeCorePromptTokens;
      full_layer_fragment.recorded_slots =
          expected_prefill_layer_route_slots(layer);
      for (std::size_t slot = 0U; slot < kPrefillLayerRouteSlotCount;
           ++slot) {
        if ((full_layer_fragment.recorded_slots &
             static_cast<std::uint16_t>(1U << slot)) != 0U) {
          full_layer_fragment.dispositions[slot] =
              PrefillRouteDisposition::kProduction;
        }
      }
      status = collapse_layer_wide_p40_prefill_layer_route_fragment(
          full_layer_fragment, layer_wide_p40_request_pass);
      if (!status) {
        return fail_whole_request_prefill(status);
      }
      status = record_submission(
          layer, kPromptWideP40WholeCorePersistentMlpPhase);
      if (!status) {
        return fail_whole_request_prefill(status);
      }
      continue;
    }

    std::array<PrefillLayerSegmentRouteFragment,
               kLayerMajorPrefillMaximumPanelCount>
        layer_wide_panel_fragments{};
    for (std::size_t panel_index = 0U;
         panel_index < immutable_topology.panel_count; ++panel_index) {
      if (bounded_submission_window &&
          submission_window_in_flight ==
              kWholeRequestSubmissionWindowSlots) {
        const ReferenceRunnerStatus retirement =
            retire_oldest_submission();
        if (!retirement) {
          return fail_whole_request_prefill(retirement);
        }
      }
      const PrefillOperatorPanel& panel =
          immutable_topology.panels[panel_index];
      const std::size_t prompt_offset =
          static_cast<std::size_t>(panel.first_position -
                                   immutable_topology.first_position);
      if (executor == LayerMajorLayerExecutor::kOperatorPanel &&
          panel.token_count >= kPrefillPhysicalSegmentM32Tokens) {
        const PrefillLayerSegmentEnqueueResult enqueued =
            enqueue_prefill_layer_panel(
                input_token_ids + prompt_offset, panel.token_count,
                panel.first_position, layer, projection_tactic,
                full_attention_tactic, layer_wide_p40_mlp,
                layer_major);
        if (!enqueued) {
          return fail_whole_request_prefill(enqueued.status);
        }
        if (!enqueued.route_fragment.has_single_layer_segment ||
            enqueued.mlp_deferred != layer_wide_p40_mlp) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kRouteEvidenceFailure,
              "whole_request_prefill_panel_operator_route", layer));
        }
        ++operator_panel_executor_hits;
        segmented_panel_projection_hits +=
            enqueued.segmented_panel_projection_hits;
        segmented_panel_projection_physical_launches +=
            enqueued.segmented_panel_projection_physical_launches;
        native_large_m_projection_hits +=
            enqueued.native_large_m_projection_hits;
        native_large_m_projection_bulk_hits +=
            enqueued.native_large_m_projection_bulk_hits;
        native_large_m_projection_oracle_partial_hits +=
            enqueued.native_large_m_projection_oracle_partial_hits;
        native_large_m_projection_physical_launches +=
            enqueued.native_large_m_projection_physical_launches;
        nvfp4_true_large_m_route_fp8_projection_hits +=
            enqueued.nvfp4_true_large_m_route_fp8_projection_hits;
        nvfp4_true_large_m_route_fp8_projection_bulk_hits +=
            enqueued.nvfp4_true_large_m_route_fp8_projection_bulk_hits;
        nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits +=
            enqueued
                .nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits;
        nvfp4_true_large_m_route_fp8_projection_physical_launches +=
            enqueued.nvfp4_true_large_m_route_fp8_projection_physical_launches;
        native_nvfp4_true_large_m_projection_hits +=
            enqueued.native_nvfp4_true_large_m_projection_hits;
        native_nvfp4_true_large_m_gate_up_hits +=
            enqueued.native_nvfp4_true_large_m_gate_up_hits;
        native_nvfp4_true_large_m_down_hits +=
            enqueued.native_nvfp4_true_large_m_down_hits;
        native_nvfp4_true_large_m_physical_launches +=
            enqueued.native_nvfp4_true_large_m_physical_launches;
        persistent_p40_fp8_projection_hits +=
            enqueued.persistent_p40_fp8_projection_hits;
        persistent_p40_fp8_projection_bulk_hits +=
            enqueued.persistent_p40_fp8_projection_bulk_hits;
        persistent_p40_fp8_projection_oracle_partial_hits +=
            enqueued.persistent_p40_fp8_projection_oracle_partial_hits;
        persistent_p40_fp8_projection_physical_launches +=
            enqueued.persistent_p40_fp8_projection_physical_launches;
        if (reference_runner_detail::expected_reference_layer_type(layer) ==
            model::LayerType::kFullAttention) {
          if (full_attention_tactic ==
              LayerMajorPrefillFullAttentionTactic::
                  kNativeGroupQ64Panel) {
            ++native_group_q64_panel_hits;
          } else if (full_attention_tactic ==
                     LayerMajorPrefillFullAttentionTactic::
                         kNativeGroupQ128V4Panel) {
            ++native_group_q128_v4_panel_hits;
          } else if (full_attention_tactic ==
                     LayerMajorPrefillFullAttentionTactic::
                         kNativeFlashInferExactPanel) {
            ++native_flashinfer_exact_panel_hits;
          } else {
            const LayerMajorPrefillArithmeticSpanLedger ledger =
                make_layer_major_prefill_arithmetic_span_ledger(
                    panel.token_count);
            for (std::size_t span_index = 0U;
                 span_index < ledger.span_count; ++span_index) {
              const LayerMajorPrefillArithmeticSpan& span =
                  ledger.spans[span_index];
              if (select_fixed_bulk_causal_gqa_prefill_tactic(
                      static_cast<std::size_t>(panel.first_position) +
                          span.token_offset,
                      span.token_count) ==
                  FixedBulkCausalGqaPrefillTactic::kGenericQt2) {
                ++generic_qt2_hits;
              }
            }
          }
        }
        if (layer_wide_p40_mlp) {
          layer_wide_panel_fragments[panel_index] =
              enqueued.route_fragment.layer_segment;
        } else {
          const ReferenceRunnerStatus collapse_status =
              collapse_prefill_layer_route_fragment(
                  enqueued.route_fragment.layer_segment,
                  panel_layer_passes[panel_index]);
          if (!collapse_status) {
            return fail_whole_request_prefill(collapse_status);
          }
        }
      } else {
        if (layer_wide_p40_mlp) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kInvalidStepOptions,
              "whole_request_prefill_layer_wide_executor", layer));
        }
        PrefillLayerRouteReducer reducer;
        std::uint32_t segment_position = panel.first_position;
        std::size_t remaining = panel.token_count;
        while (remaining != 0U) {
          const std::size_t segment_token_count =
              next_layer_major_prefill_physical_segment_token_count(remaining);
          if (!is_layer_major_prefill_physical_segment_token_count(
                  segment_token_count) ||
              segment_token_count > remaining ||
              segment_position > panel.end_position ||
              segment_token_count >
                  static_cast<std::size_t>(panel.end_position -
                                           segment_position)) {
            return fail_whole_request_prefill(runner_status(
                ReferenceRunnerError::kInvalidStepOptions,
                "whole_request_prefill_physical_segment", layer));
          }

          PrefillTileExecutionControl control;
          control.first_position_override = segment_position;
          control.layer_begin = layer;
          control.layer_end = layer + 1U;
          control.gather_embedding = layer == 0U;
          control.apply_final_norm = false;
          control.synchronize = false;
          control.commit_state = false;
          control.commit_route = false;
          control.allow_scalar_m1_delegate = false;
          control.allow_cross_layer_m32_fusion = false;
          control.emit_commit_hooks = false;
          control.force_bound_nvfp4_marlin_prefill = true;
          control.force_bound_fp8_marlin_prefill = true;
          control.force_bound_gdn_chunk64_native_prefill = true;
          PrefillTileExecutionSelection selection;
          const ReferenceRunnerStatus selection_status =
              select_prefill_tile_execution(
                  control, state_->current_position(),
                  state_->max_sequence_length(),
                  layer_major.descriptor.legacy_prefill_chunk_size,
                  segment_token_count, segment_options, selection);
          if (!selection_status ||
              selection.first_position != segment_position ||
              selection.completed_position !=
                  segment_position + segment_token_count ||
              selection.delegate_scalar_m1) {
            return fail_whole_request_prefill(
                selection_status
                    ? runner_status(
                          ReferenceRunnerError::kInvalidStepOptions,
                          "whole_request_prefill_segment_selection", layer)
                    : selection_status);
          }

          Views segment_views = views_;
          segment_views.hidden[0] =
              prompt_residual_base +
              static_cast<std::size_t>(segment_position) *
                  prompt_residual.row_stride_elements;
          const std::size_t segment_prompt_offset =
              static_cast<std::size_t>(segment_position -
                                       immutable_topology.first_position);
          const PrefillLayerSegmentEnqueueResult enqueued =
              enqueue_prefill_layer_segment(
                  input_token_ids + segment_prompt_offset,
                  segment_token_count, selection.first_position, control,
                  segment_views);
          if (!enqueued) {
            return fail_whole_request_prefill(enqueued.status);
          }
          if (!enqueued.route_fragment.has_single_layer_segment) {
            return fail_whole_request_prefill(runner_status(
                ReferenceRunnerError::kRouteEvidenceFailure,
                "whole_request_prefill_segment_route", layer));
          }
          if (reference_runner_detail::expected_reference_layer_type(layer) ==
              model::LayerType::kLinearAttention) {
            const PrefillRouteDisposition gdn_disposition =
                enqueued.route_fragment.layer_segment.dispositions[
                    static_cast<std::size_t>(PrefillLayerRouteSlot::kGdn)];
            const PrefillRouteDisposition expected_gdn_disposition =
                segment_token_count >= 32U
                    ? PrefillRouteDisposition::kProduction
                    : PrefillRouteDisposition::kExactFallback;
            if (gdn_disposition != expected_gdn_disposition) {
              return fail_whole_request_prefill(runner_status(
                  ReferenceRunnerError::kRouteEvidenceFailure,
                  "whole_request_prefill_bound_gdn_route", layer));
            }
          }
          const ReferenceRunnerStatus reduce_status =
              reduce_prefill_layer_route_fragment(
                  enqueued.route_fragment.layer_segment, reducer);
          if (!reduce_status) {
            return fail_whole_request_prefill(reduce_status);
          }
          segment_position +=
              static_cast<std::uint32_t>(segment_token_count);
          remaining -= segment_token_count;
        }
        if (!reducer.initialized ||
            reducer.route_fragment.first_position != panel.first_position ||
            reducer.route_fragment.token_count != panel.token_count ||
            segment_position != panel.end_position) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kRouteEvidenceFailure,
              "whole_request_prefill_panel_route", layer));
        }
        const ReferenceRunnerStatus collapse_status =
            collapse_prefill_layer_route_fragment(
                reducer.route_fragment,
                panel_layer_passes[panel_index]);
        if (!collapse_status) {
          return fail_whole_request_prefill(collapse_status);
        }
      }
      const ReferenceRunnerStatus submission =
          record_submission(layer, panel_index);
      if (!submission) {
        return fail_whole_request_prefill(submission);
      }
    }
    if (layer_wide_p40_mlp) {
      if (bounded_submission_window &&
          submission_window_in_flight ==
              kWholeRequestSubmissionWindowSlots) {
        const ReferenceRunnerStatus retirement = retire_oldest_submission();
        if (!retirement) {
          return fail_whole_request_prefill(retirement);
        }
      }
      const ReferenceRunnerStatus mlp_status =
          enqueue_layer_wide_p40_mlp(layer, layer_major);
      if (!mlp_status) {
        return fail_whole_request_prefill(mlp_status);
      }
      ++layer_wide_p40_mlp_layer_hits;
      ++persistent_p40_nvfp4_gate_up_hits;
      ++persistent_p40_nvfp4_down_residual_hits;
      persistent_p40_nvfp4_physical_launches += 2U;

      const auto route_bit = [](const PrefillLayerRouteSlot slot) noexcept {
        return static_cast<std::uint16_t>(
            1U << static_cast<std::uint8_t>(slot));
      };
      const std::uint16_t mlp_route_bits = static_cast<std::uint16_t>(
          route_bit(PrefillLayerRouteSlot::kNvFp4GateUp) |
          route_bit(PrefillLayerRouteSlot::kNvFp4Down));
      const std::uint16_t expected_route_slots =
          expected_prefill_layer_route_slots(layer);
      const std::uint16_t expected_panel_route_slots =
          static_cast<std::uint16_t>(expected_route_slots & ~mlp_route_bits);
      const auto valid_disposition = [](const PrefillRouteDisposition value)
          noexcept {
            return value == PrefillRouteDisposition::kProduction ||
                   value == PrefillRouteDisposition::kExactFallback ||
                   value == PrefillRouteDisposition::kForbidden;
          };
      const auto weakest_disposition = [](
          const PrefillRouteDisposition left,
          const PrefillRouteDisposition right) noexcept {
            if (left == PrefillRouteDisposition::kForbidden ||
                right == PrefillRouteDisposition::kForbidden) {
              return PrefillRouteDisposition::kForbidden;
            }
            if (left == PrefillRouteDisposition::kExactFallback ||
                right == PrefillRouteDisposition::kExactFallback) {
              return PrefillRouteDisposition::kExactFallback;
            }
            return PrefillRouteDisposition::kProduction;
          };
      PrefillLayerSegmentRouteFragment full_layer_fragment =
          layer_wide_panel_fragments[0U];
      std::uint32_t expected_first_position =
          immutable_topology.first_position;
      for (std::size_t panel_index = 0U;
           panel_index < immutable_topology.panel_count; ++panel_index) {
        const PrefillOperatorPanel& panel =
            immutable_topology.panels[panel_index];
        const PrefillLayerSegmentRouteFragment& fragment =
            layer_wide_panel_fragments[panel_index];
        bool dispositions_valid = true;
        for (std::size_t slot = 0U; slot < kPrefillLayerRouteSlotCount;
             ++slot) {
          if ((expected_panel_route_slots &
               static_cast<std::uint16_t>(1U << slot)) != 0U &&
              !valid_disposition(fragment.dispositions[slot])) {
            dispositions_valid = false;
            break;
          }
        }
        if (fragment.layer != layer ||
            fragment.first_position != expected_first_position ||
            fragment.first_position != panel.first_position ||
            fragment.token_count != panel.token_count ||
            fragment.recorded_slots != expected_panel_route_slots ||
            !dispositions_valid) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kRouteEvidenceFailure,
              "whole_request_prefill_layer_wide_panel_route", layer));
        }
        if (panel_index != 0U) {
          for (std::size_t slot = 0U; slot < kPrefillLayerRouteSlotCount;
               ++slot) {
            if ((expected_panel_route_slots &
                 static_cast<std::uint16_t>(1U << slot)) != 0U) {
              full_layer_fragment.dispositions[slot] = weakest_disposition(
                  full_layer_fragment.dispositions[slot],
                  fragment.dispositions[slot]);
            }
          }
          full_layer_fragment.forbidden_boundaries =
              static_cast<std::uint8_t>(
                  full_layer_fragment.forbidden_boundaries |
                  fragment.forbidden_boundaries);
          full_layer_fragment.token_count += fragment.token_count;
        }
        expected_first_position += fragment.token_count;
      }
      full_layer_fragment.recorded_slots = expected_route_slots;
      full_layer_fragment.dispositions[static_cast<std::size_t>(
          PrefillLayerRouteSlot::kNvFp4GateUp)] =
          PrefillRouteDisposition::kProduction;
      full_layer_fragment.dispositions[static_cast<std::size_t>(
          PrefillLayerRouteSlot::kNvFp4Down)] =
          PrefillRouteDisposition::kProduction;
      if (expected_first_position != immutable_topology.final_position ||
          full_layer_fragment.first_position != 0U ||
          full_layer_fragment.token_count !=
              kLayerMajorPrefillLayerWideMlpP40Tokens) {
        return fail_whole_request_prefill(runner_status(
            ReferenceRunnerError::kRouteEvidenceFailure,
            "whole_request_prefill_layer_wide_route_span", layer));
      }
      const ReferenceRunnerStatus collapse_status =
          collapse_layer_wide_p40_prefill_layer_route_fragment(
              full_layer_fragment, layer_wide_p40_request_pass);
      if (!collapse_status) {
        return fail_whole_request_prefill(collapse_status);
      }
      const ReferenceRunnerStatus submission = record_submission(
          layer, immutable_topology.panel_count);
      if (!submission) {
        return fail_whole_request_prefill(submission);
      }
    }
  }

  while (submission_window_in_flight != 0U) {
    const ReferenceRunnerStatus retirement = retire_oldest_submission();
    if (!retirement) {
      return fail_whole_request_prefill(retirement);
    }
  }

  const std::uint32_t final_position =
      immutable_topology.final_position - 1U;
  const std::uint16_t* const final_residual =
      prompt_residual_base +
      static_cast<std::size_t>(final_position) *
          prompt_residual.row_stride_elements;
  const int final_norm_status =
      launch_headwise_centered_rms_norm_reference_cuda(
          final_residual, weights_->final_norm().data, 1U,
          kReferenceHiddenSize, kRmsEpsilon, final_hidden, stream_);
  if (final_norm_status != static_cast<int>(cudaSuccess)) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kCudaFailure,
        "whole_request_prefill_final_norm", kReferenceNoLayer,
        final_norm_status));
  }

  const cudaError_t sync_status = cudaStreamSynchronize(
      reinterpret_cast<cudaStream_t>(stream_));
  if (sync_status != cudaSuccess) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kCudaFailure,
        "whole_request_prefill_synchronize", kReferenceNoLayer,
        static_cast<int>(sync_status)));
  }
  if (cancellation_requested()) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kCancelled,
        "whole_request_prefill_cancelled",
        kReferenceDecoderLayerCount - 1U, 0,
        static_cast<std::uint64_t>(submission_window_retirements)));
  }

  PrefillExecutionProgress progress =
      make_prefill_execution_progress(immutable_topology);
  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount;
       ++layer) {
    if (prompt_wide_p40_whole_core) {
      if (use_packed_nvfp4_v2) {
        if (advance_prompt_wide_p40_packed_nvfp4_v2_layer_progress_after_completion(
                immutable_topology, progress, layer) !=
            PrefillExecutionProgressError::kNone) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kInvalidStepOptions,
              "whole_request_prefill_packed_nvfp4_v2_progress", layer));
        }
        continue;
      }
      if (use_packed_projection) {
        if (advance_prompt_wide_p40_packed_projection_layer_progress_after_completion(
                immutable_topology, progress, layer) !=
            PrefillExecutionProgressError::kNone) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kInvalidStepOptions,
              "whole_request_prefill_packed_projection_progress", layer));
        }
        continue;
      }
      if (use_projection_reset) {
        if (advance_prompt_wide_p40_projection_reset_layer_progress_after_completion(
                immutable_topology, progress, layer) !=
            PrefillExecutionProgressError::kNone) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kInvalidStepOptions,
              "whole_request_prefill_projection_reset_progress", layer));
        }
        continue;
      }
      for (std::size_t panel_index = 0U;
           panel_index < immutable_topology.panel_count; ++panel_index) {
        if (advance_prompt_wide_p40_fill_progress_after_completion(
                immutable_topology, progress, layer, panel_index) !=
            PrefillExecutionProgressError::kNone) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kInvalidStepOptions,
              "whole_request_prefill_whole_core_fill_progress", layer));
        }
      }
      if (advance_prompt_wide_p40_prompt_core_progress_after_completion(
              immutable_topology, progress, layer) !=
          PrefillExecutionProgressError::kNone) {
        return fail_whole_request_prefill(runner_status(
            ReferenceRunnerError::kInvalidStepOptions,
            "whole_request_prefill_whole_core_prompt_core_progress", layer));
      }
      for (std::size_t panel_index = 0U;
           panel_index < immutable_topology.panel_count; ++panel_index) {
        if (advance_prompt_wide_p40_drain_progress_after_completion(
                immutable_topology, progress, layer, panel_index) !=
            PrefillExecutionProgressError::kNone) {
          return fail_whole_request_prefill(runner_status(
              ReferenceRunnerError::kInvalidStepOptions,
              "whole_request_prefill_whole_core_drain_progress", layer));
        }
      }
      if (advance_prompt_wide_p40_persistent_mlp_progress_after_completion(
              immutable_topology, progress, layer) !=
          PrefillExecutionProgressError::kNone) {
        return fail_whole_request_prefill(runner_status(
            ReferenceRunnerError::kInvalidStepOptions,
            "whole_request_prefill_whole_core_mlp_progress", layer));
      }
      continue;
    }
    for (std::size_t panel_index = 0U;
         panel_index < immutable_topology.panel_count; ++panel_index) {
      if (advance_prefill_progress_after_completion(
              immutable_topology, progress, layer, panel_index) !=
          PrefillExecutionProgressError::kNone) {
        return fail_whole_request_prefill(runner_status(
            ReferenceRunnerError::kInvalidStepOptions,
            "whole_request_prefill_progress", layer));
      }
    }
    if (layer_wide_p40_mlp &&
        advance_layer_wide_p40_mlp_progress_after_completion(
            immutable_topology, progress, layer) !=
            PrefillExecutionProgressError::kNone) {
      return fail_whole_request_prefill(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "whole_request_prefill_layer_wide_mlp_progress", layer));
    }
  }
  if (mark_prefill_final_hidden_ready(immutable_topology, progress) !=
          PrefillExecutionProgressError::kNone ||
      !prefill_final_commit_ready(immutable_topology, progress)) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_final_progress"));
  }

  PrefillRouteEvidence staged_route = prefill_route_evidence_;
  if (layer_wide_p40_mlp || prompt_wide_p40_whole_core) {
    if (!commit_prefill_route_layer_pass(staged_route,
                                         layer_wide_p40_request_pass)) {
      return fail_whole_request_prefill(runner_status(
          ReferenceRunnerError::kRouteEvidenceFailure,
          "whole_request_prefill_route_merge"));
    }
  } else {
    for (std::size_t panel_index = 0U;
         panel_index < immutable_topology.panel_count; ++panel_index) {
      if (!commit_prefill_route_layer_pass(
              staged_route, panel_layer_passes[panel_index])) {
        return fail_whole_request_prefill(runner_status(
            ReferenceRunnerError::kRouteEvidenceFailure,
            "whole_request_prefill_route_merge"));
      }
    }
  }
  if (staged_route.error != PrefillRouteEvidenceError::kNone ||
      staged_route.complete || !staged_route.request_active) {
    return fail_whole_request_prefill(runner_status(
        ReferenceRunnerError::kRouteEvidenceFailure,
        "whole_request_prefill_route_forbidden"));
  }

  WholeRequestPrefillStage stage;
  stage.phase = WholeRequestPrefillStagePhase::kAwaitingLogits;
  stage.expected_initial_sequence_length = immutable_topology.first_position;
  stage.committed_sequence_length = immutable_topology.final_position;
  stage.final_position = final_position;
  stage.final_input_token_id = input_token_ids[token_count - 1U];
  stage.prompt_token_count = token_count;
  stage.logical_panel_count = immutable_topology.panel_count;
  stage.mlp_phase_count_per_layer =
      immutable_topology.mlp_schedule.mlp_phase_submission_count_per_layer;
  stage.final_normalized_hidden = final_hidden;
  stage.completed_uncommitted_progress = progress;
  stage.route_evidence_after_commit = staged_route;
  whole_request_prefill_stage_ = stage;

  ReferenceWholeRequestPrefillResult result;
  result.first_position = immutable_topology.first_position;
  result.final_position = immutable_topology.final_position;
  result.prompt_token_count = token_count;
  result.logical_panel_count = immutable_topology.panel_count;
  result.bounded_submission_window = bounded_submission_window;
  result.submission_window_retirements =
      submission_window_retirements;
  result.operator_panel_executor_hits = operator_panel_executor_hits;
  result.native_group_q64_panel_hits = native_group_q64_panel_hits;
  result.native_group_q128_v4_panel_hits = native_group_q128_v4_panel_hits;
  result.native_flashinfer_exact_panel_hits =
      native_flashinfer_exact_panel_hits;
  result.generic_qt2_hits = generic_qt2_hits;
  result.segmented_panel_projection_hits = segmented_panel_projection_hits;
  result.segmented_panel_projection_physical_launches =
      segmented_panel_projection_physical_launches;
  result.native_large_m_projection_hits =
      native_large_m_projection_hits;
  result.native_large_m_projection_bulk_hits =
      native_large_m_projection_bulk_hits;
  result.native_large_m_projection_oracle_partial_hits =
      native_large_m_projection_oracle_partial_hits;
  result.native_large_m_projection_physical_launches =
      native_large_m_projection_physical_launches;
  result.nvfp4_true_large_m_route_fp8_projection_hits =
      nvfp4_true_large_m_route_fp8_projection_hits;
  result.nvfp4_true_large_m_route_fp8_projection_bulk_hits =
      nvfp4_true_large_m_route_fp8_projection_bulk_hits;
  result.nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits =
      nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits;
  result.nvfp4_true_large_m_route_fp8_projection_physical_launches =
      nvfp4_true_large_m_route_fp8_projection_physical_launches;
  result.native_nvfp4_true_large_m_projection_hits =
      native_nvfp4_true_large_m_projection_hits;
  result.native_nvfp4_true_large_m_gate_up_hits =
      native_nvfp4_true_large_m_gate_up_hits;
  result.native_nvfp4_true_large_m_down_hits =
      native_nvfp4_true_large_m_down_hits;
  result.native_nvfp4_true_large_m_physical_launches =
      native_nvfp4_true_large_m_physical_launches;
  result.mlp_schedule_tactic = immutable_topology.mlp_schedule.tactic;
  result.layer_wide_p40_mlp_layer_hits = layer_wide_p40_mlp_layer_hits;
  result.persistent_p40_nvfp4_gate_up_hits =
      persistent_p40_nvfp4_gate_up_hits;
  result.persistent_p40_nvfp4_down_residual_hits =
      persistent_p40_nvfp4_down_residual_hits;
  result.persistent_p40_nvfp4_physical_launches =
      persistent_p40_nvfp4_physical_launches;
  result.persistent_p40_fp8_projection_hits =
      persistent_p40_fp8_projection_hits;
  result.persistent_p40_fp8_projection_bulk_hits =
      persistent_p40_fp8_projection_bulk_hits;
  result.persistent_p40_fp8_projection_oracle_partial_hits =
      persistent_p40_fp8_projection_oracle_partial_hits;
  result.persistent_p40_fp8_projection_physical_launches =
      persistent_p40_fp8_projection_physical_launches;
  result.prompt_wide_p40_whole_core_layer_hits =
      prompt_wide_p40_whole_core_layer_hits;
  result.prompt_wide_p40_fill_panel_hits =
      prompt_wide_p40_fill_panel_hits;
  result.prompt_wide_p40_prompt_core_hits =
      prompt_wide_p40_prompt_core_hits;
  result.prompt_wide_p40_drain_panel_hits =
      prompt_wide_p40_drain_panel_hits;
  result.prompt_wide_p40_fp8_projection_hits =
      prompt_wide_p40_fp8_projection_hits;
  result.prompt_wide_p40_fp8_projection_physical_launches =
      prompt_wide_p40_fp8_projection_physical_launches;
  result.prompt_wide_p40_bf16_ab_hits = prompt_wide_p40_bf16_ab_hits;
  result.prompt_wide_p40_gdn_hits = prompt_wide_p40_gdn_hits;
  result.native_flashinfer_exact_whole_prompt_hits =
      native_flashinfer_exact_whole_prompt_hits;
  result.packed_nvfp4_v2_gate_up_hits = packed_nvfp4_v2_gate_up_hits;
  result.packed_nvfp4_v2_down_hits = packed_nvfp4_v2_down_hits;
  result.packed_nvfp4_v2_physical_launches =
      packed_nvfp4_v2_physical_launches;
  result.progress = progress;
  if (options.measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    result.timing.emplace(ReferenceStepTiming{elapsed.count()});
  }
  ReferenceWholeRequestPrefillOutcome outcome;
  outcome.value.emplace(std::move(result));
  return outcome;
}

ReferenceRunner::PrefillLayerSegmentEnqueueResult
ReferenceRunner::enqueue_prefill_layer_panel(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const std::uint32_t first_position,
    const std::size_t layer,
    const LayerMajorPrefillProjectionTactic projection_tactic,
    const LayerMajorPrefillFullAttentionTactic full_attention_tactic,
    const bool defer_mlp,
    const ReferenceLayerMajorRequestViews& request_views) noexcept {
  const auto fail_enqueue = [](const ReferenceRunnerStatus status) noexcept {
    PrefillLayerSegmentEnqueueResult result;
    result.status = status;
    return result;
  };
#if !defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION) || \
    !defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION) || \
    !defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  (void)input_token_ids;
  (void)token_count;
  (void)first_position;
  (void)layer;
  (void)projection_tactic;
  (void)full_attention_tactic;
  (void)defer_mlp;
  (void)request_views;
  return fail_enqueue(runner_status(
      ReferenceRunnerError::kInvalidDependency,
      "prefill_operator_panel_build_inventory"));
#else
  static_assert(kernels::kSm87Fp8MarlinReductionBytes ==
                kernels::kSm87NvFp4MarlinReductionBytes);
  static_assert(kernels::kSm87Fp8MarlinLockBytes ==
                kernels::kSm87NvFp4MarlinLockBytes);
  constexpr std::size_t kFp8MarlinWorkspaceBytes =
      kernels::kSm87Fp8MarlinReductionBytes +
      kernels::kSm87Fp8MarlinLockBytes;
  constexpr std::size_t kNvFp4MarlinWorkspaceBytes =
      kernels::kSm87NvFp4MarlinReductionBytes +
      kernels::kSm87NvFp4MarlinLockBytes;
  const LayerMajorPrefillArithmeticSpanLedger arithmetic_ledger =
      make_layer_major_prefill_arithmetic_span_ledger(token_count);
  if (input_token_ids == nullptr || layer >= kReferenceDecoderLayerCount ||
      token_count < kPrefillPhysicalSegmentM32Tokens ||
      token_count > kLayerMajorRequestOperatorPanelCapacity ||
      first_position > request_views.descriptor.max_sequence_length ||
      token_count > static_cast<std::size_t>(
                        request_views.descriptor.max_sequence_length -
                        first_position) ||
      !is_valid_layer_major_prefill_arithmetic_span_ledger(
          arithmetic_ledger) ||
      !is_valid_layer_major_prefill_projection_tactic(projection_tactic) ||
      !is_valid_layer_major_prefill_full_attention_tactic(
          full_attention_tactic) ||
      ((projection_tactic == LayerMajorPrefillProjectionTactic::
                                 kNativeNvfp4TrueLargeMOperatorPanel ||
        projection_tactic == LayerMajorPrefillProjectionTactic::
                                 kNativeNvfp4G2D2LargeMOperatorPanel ||
        projection_tactic == LayerMajorPrefillProjectionTactic::
                                 kNativeNvfp4PersistentP40LayerWideMlp) &&
       !is_nvfp4_true_large_m_prefill_panel_tokens(token_count)) ||
      projection_backend_ != ProjectionBackend::kSm87WeightOnly) {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "prefill_operator_panel_geometry", layer));
  }

  const auto valid_matrix = [token_count](const DeviceMatrixView& view,
                                           const std::size_t columns,
                                           const std::uint32_t element_bytes)
      noexcept {
        return view.storage.device_data != nullptr &&
               view.storage.element_size_bytes == element_bytes &&
               view.row_capacity >= token_count && view.columns == columns &&
               view.row_stride_elements == columns &&
               view.storage.element_capacity >= token_count * columns;
      };
  const auto valid_fp8_temporary = [](const DeviceBufferView& view) noexcept {
    return view.device_data != nullptr &&
           view.byte_size >= kFp8MarlinWorkspaceBytes;
  };
  const auto valid_nvfp4_temporary = [](const DeviceBufferView& view) noexcept {
    return view.device_data != nullptr &&
           view.byte_size >= kNvFp4MarlinWorkspaceBytes;
  };
  const auto fp8_workspace = [](const DeviceBufferView& view) noexcept {
    ExactPrefillProjectionWorkspace workspace;
    if (view.device_data == nullptr ||
        view.byte_size < kFp8MarlinWorkspaceBytes) {
      return workspace;
    }
    workspace.reduction = static_cast<float*>(view.device_data);
    workspace.reduction_elements =
        kernels::kSm87Fp8MarlinReductionElements;
    workspace.locks = reinterpret_cast<std::int32_t*>(
        static_cast<std::uint8_t*>(view.device_data) +
        kernels::kSm87Fp8MarlinReductionBytes);
    workspace.lock_bytes = kernels::kSm87Fp8MarlinLockBytes;
    return workspace;
  };
  const auto nvfp4_workspace = [](const DeviceBufferView& view) noexcept {
    ExactPrefillProjectionWorkspace workspace;
    if (view.device_data == nullptr ||
        view.byte_size < kNvFp4MarlinWorkspaceBytes) {
      return workspace;
    }
    workspace.reduction = static_cast<float*>(view.device_data);
    workspace.reduction_elements =
        kernels::kSm87NvFp4MarlinReductionElements;
    workspace.locks = reinterpret_cast<std::int32_t*>(
        static_cast<std::uint8_t*>(view.device_data) +
        kernels::kSm87NvFp4MarlinReductionBytes);
    workspace.lock_bytes = kernels::kSm87NvFp4MarlinLockBytes;
    return workspace;
  };
  const auto aligned_16 = [](const void* const pointer) noexcept {
    return pointer != nullptr &&
           reinterpret_cast<std::uintptr_t>(pointer) % 16U == 0U;
  };

  const DeviceMatrixView& prompt_view = request_views.prompt_residual_bf16;
  if (prompt_view.storage.device_data == nullptr ||
      prompt_view.storage.element_size_bytes != sizeof(std::uint16_t) ||
      prompt_view.columns != kReferenceHiddenSize ||
      prompt_view.row_stride_elements != kReferenceHiddenSize ||
      prompt_view.row_capacity <
          static_cast<std::size_t>(first_position) + token_count ||
      request_views.panel_token_ids_u32.storage.device_data == nullptr ||
      request_views.panel_token_ids_u32.storage.element_size_bytes !=
          sizeof(std::uint32_t) ||
      request_views.panel_token_ids_u32.row_capacity < token_count ||
      request_views.panel_token_ids_u32.columns != 1U ||
      request_views.panel_token_ids_u32.row_stride_elements != 1U) {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidRequestState,
        "prefill_operator_panel_common_views", layer));
  }

  ReferenceRunnerStatus launch_failure{};
  const auto check_cuda = [&launch_failure](
                              const int status,
                              const char* const operation,
                              const std::size_t failed_layer) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    launch_failure = runner_status(ReferenceRunnerError::kCudaFailure,
                                   operation, failed_layer, status);
    return false;
  };
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  std::size_t segmented_panel_projection_hits = 0U;
  std::size_t segmented_panel_projection_physical_launches = 0U;
  std::size_t native_large_m_projection_hits = 0U;
  std::size_t native_large_m_projection_bulk_hits = 0U;
  std::size_t native_large_m_projection_oracle_partial_hits = 0U;
  std::size_t native_large_m_projection_physical_launches = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_hits = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_bulk_hits = 0U;
  std::size_t
      nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits = 0U;
  std::size_t nvfp4_true_large_m_route_fp8_projection_physical_launches = 0U;
  std::size_t native_nvfp4_true_large_m_projection_hits = 0U;
  std::size_t native_nvfp4_true_large_m_gate_up_hits = 0U;
  std::size_t native_nvfp4_true_large_m_down_hits = 0U;
  std::size_t native_nvfp4_true_large_m_physical_launches = 0U;
  std::size_t persistent_p40_fp8_projection_hits = 0U;
  std::size_t persistent_p40_fp8_projection_bulk_hits = 0U;
  std::size_t persistent_p40_fp8_projection_oracle_partial_hits = 0U;
  std::size_t persistent_p40_fp8_projection_physical_launches = 0U;
  const auto fp8_project =
      [this, projection_tactic, token_count, &arithmetic_ledger, &check_cuda,
       &fp8_workspace, &segmented_panel_projection_hits,
       &segmented_panel_projection_physical_launches,
       &native_large_m_projection_hits,
       &native_large_m_projection_bulk_hits,
       &native_large_m_projection_oracle_partial_hits,
       &native_large_m_projection_physical_launches,
       &nvfp4_true_large_m_route_fp8_projection_hits,
       &nvfp4_true_large_m_route_fp8_projection_bulk_hits,
       &nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits,
       &nvfp4_true_large_m_route_fp8_projection_physical_launches,
       &persistent_p40_fp8_projection_hits,
       &persistent_p40_fp8_projection_bulk_hits,
       &persistent_p40_fp8_projection_oracle_partial_hits,
       &persistent_p40_fp8_projection_physical_launches](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const DeviceBufferView& temporary,
          const char* const operation, const std::size_t failed_layer)
      noexcept {
        int status = static_cast<int>(cudaErrorInvalidValue);
        switch (projection_tactic) {
          case LayerMajorPrefillProjectionTactic::
              kSegmentedMarlinOperatorPanel:
            status = launch_segmented_panel_fp8_projection(
                projection_backend_, weight, input, output, token_count,
                fp8_workspace(temporary),
                segmented_panel_projection_physical_launches, stream_);
            break;
          case LayerMajorPrefillProjectionTactic::
              kNativeQuantizedLargeMOperatorPanel:
            status = launch_native_large_m_fp8_projection(
                projection_backend_, weight, input, output, token_count,
                arithmetic_ledger, fp8_workspace(temporary),
                native_large_m_projection_physical_launches, stream_);
            break;
          case LayerMajorPrefillProjectionTactic::
              kNativeNvfp4TrueLargeMOperatorPanel:
          case LayerMajorPrefillProjectionTactic::
              kNativeNvfp4G2D2LargeMOperatorPanel:
            status = launch_native_large_m_fp8_projection(
                projection_backend_, weight, input, output, token_count,
                arithmetic_ledger, fp8_workspace(temporary),
                nvfp4_true_large_m_route_fp8_projection_physical_launches,
                stream_);
            break;
          case LayerMajorPrefillProjectionTactic::
              kNativeNvfp4PersistentP40LayerWideMlp:
            status = launch_native_large_m_fp8_projection(
                projection_backend_, weight, input, output, token_count,
                arithmetic_ledger, fp8_workspace(temporary),
                persistent_p40_fp8_projection_physical_launches, stream_);
            break;
          case LayerMajorPrefillProjectionTactic::kExactSegmentedC512:
            status = launch_exact_contract_fp8_projection(
                projection_backend_, weight, input, output,
                arithmetic_ledger, fp8_workspace(temporary), stream_);
            break;
          case LayerMajorPrefillProjectionTactic::
              kNativePromptWideP40WholeCore:
          case LayerMajorPrefillProjectionTactic::
              kNativePromptWideP40ProjectionReset:
          case LayerMajorPrefillProjectionTactic::
              kNativePromptWideP40PackedProjection:
          case LayerMajorPrefillProjectionTactic::
              kNativePromptWideP40PackedNvfp4V2:
          case LayerMajorPrefillProjectionTactic::
              kNativePromptWideP40VllmMarlinParity:
            // This helper is never part of the whole-core route. The latter
            // binds its explicit M8000 primitive before entering this legacy
            // panel executor.
            status = static_cast<int>(cudaErrorInvalidValue);
            break;
        }
        if (!check_cuda(status, operation, failed_layer)) {
          return false;
        }
        if (projection_tactic == LayerMajorPrefillProjectionTactic::
                                     kSegmentedMarlinOperatorPanel) {
          ++segmented_panel_projection_hits;
        } else if (projection_tactic == LayerMajorPrefillProjectionTactic::
                                            kNativeQuantizedLargeMOperatorPanel) {
          ++native_large_m_projection_hits;
          if (token_count == kLayerMajorPrefillOperatorPanelTokens) {
            ++native_large_m_projection_bulk_hits;
          } else {
            ++native_large_m_projection_oracle_partial_hits;
          }
        } else if (projection_tactic == LayerMajorPrefillProjectionTactic::
                                               kNativeNvfp4TrueLargeMOperatorPanel ||
                   projection_tactic == LayerMajorPrefillProjectionTactic::
                                               kNativeNvfp4G2D2LargeMOperatorPanel) {
          ++nvfp4_true_large_m_route_fp8_projection_hits;
          if (token_count == kLayerMajorPrefillOperatorPanelTokens) {
            ++nvfp4_true_large_m_route_fp8_projection_bulk_hits;
          } else {
            ++nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits;
          }
        } else if (projection_tactic ==
                   LayerMajorPrefillProjectionTactic::
                       kNativeNvfp4PersistentP40LayerWideMlp) {
          ++persistent_p40_fp8_projection_hits;
          if (token_count == kLayerMajorPrefillOperatorPanelTokens) {
            ++persistent_p40_fp8_projection_bulk_hits;
          } else {
            ++persistent_p40_fp8_projection_oracle_partial_hits;
          }
        }
        return true;
      };

  PrefillLayerSegmentRouteFragment layer_route_fragment;
  layer_route_fragment.layer = layer;
  layer_route_fragment.first_position = first_position;
  layer_route_fragment.token_count =
      static_cast<std::uint32_t>(token_count);
  const auto record_layer_route = [&layer_route_fragment](
                                      const PrefillLayerRouteSlot slot)
      noexcept {
        const std::size_t index = static_cast<std::size_t>(slot);
        if (index >= kPrefillLayerRouteSlotCount) {
          return false;
        }
        const std::uint16_t bit =
            static_cast<std::uint16_t>(1U << index);
        if ((layer_route_fragment.recorded_slots & bit) != 0U) {
          return false;
        }
        layer_route_fragment.dispositions[index] =
            PrefillRouteDisposition::kProduction;
        layer_route_fragment.recorded_slots = static_cast<std::uint16_t>(
            layer_route_fragment.recorded_slots | bit);
        return true;
      };

  auto* const prompt_residual =
      static_cast<std::uint16_t*>(prompt_view.storage.device_data) +
      static_cast<std::size_t>(first_position) *
          prompt_view.row_stride_elements;
  if (layer == 0U) {
    auto* const device_token_ids = static_cast<std::uint32_t*>(
        request_views.panel_token_ids_u32.storage.device_data);
    if (!check_cuda(
            static_cast<int>(cudaMemcpyAsync(
                device_token_ids, input_token_ids,
                token_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                stream)),
            "prefill_operator_panel_token_ids", layer) ||
        !check_cuda(
            launch_embedding_gather_prompt_reference_cuda(
                weights_->embed_tokens().weight, kReferenceVocabularySize,
                kReferenceHiddenSize, device_token_ids, token_count,
                prompt_residual, stream_),
            "prefill_operator_panel_embedding", layer)) {
      return fail_enqueue(launch_failure);
    }
  }

  const DecoderLayerWeights& layer_weights = weights_->layer(layer);
  const model::LayerType expected =
      reference_runner_detail::expected_reference_layer_type(layer);
  std::uint16_t* attention_branch_output = nullptr;
  if (expected == model::LayerType::kLinearAttention) {
    const auto* const attention =
        std::get_if<LinearAttentionWeights>(&layer_weights.attention);
    const LayerMajorGdnPhaseViews& gdn = request_views.gdn;
    if (attention == nullptr ||
        !valid_matrix(gdn.normalized_input_bf16, kReferenceHiddenSize,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(gdn.qkv_bf16, kLinearQkvElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(gdn.z_bf16, kLinearValueElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(gdn.a_bf16, kLinearScalarElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(gdn.b_bf16, kLinearScalarElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(gdn.recurrent_core_bf16, kLinearValueElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(gdn.branch_output_bf16, kReferenceHiddenSize,
                      sizeof(std::uint16_t)) ||
        !valid_fp8_temporary(gdn.input_projection_temporary) ||
        !valid_fp8_temporary(gdn.output_projection_temporary) ||
        gdn.native_c64_workspace.device_data == nullptr ||
        gdn.native_c64_workspace.byte_size <
            gdn_prefill_chunk64_native_detail::workspace_bytes() ||
        views_.conv_state[layer] == nullptr ||
        views_.gdn_state[layer] == nullptr ||
        !supports_bf16_projection_pair(
            projection_backend_, attention->in_proj_a,
            attention->in_proj_b)) {
      return fail_enqueue(runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_operator_panel_gdn_views", layer));
    }
    auto* const normalized = static_cast<std::uint16_t*>(
        gdn.normalized_input_bf16.storage.device_data);
    auto* const qkv =
        static_cast<std::uint16_t*>(gdn.qkv_bf16.storage.device_data);
    auto* const z =
        static_cast<std::uint16_t*>(gdn.z_bf16.storage.device_data);
    auto* const a =
        static_cast<std::uint16_t*>(gdn.a_bf16.storage.device_data);
    auto* const b =
        static_cast<std::uint16_t*>(gdn.b_bf16.storage.device_data);
    auto* const recurrent = static_cast<std::uint16_t*>(
        gdn.recurrent_core_bf16.storage.device_data);
    attention_branch_output = static_cast<std::uint16_t*>(
        gdn.branch_output_bf16.storage.device_data);
    if (!check_cuda(
            launch_headwise_centered_rms_norm_reference_cuda(
                prompt_residual, layer_weights.input_layernorm.data,
                token_count, kReferenceHiddenSize, kRmsEpsilon, normalized,
                stream_),
            "prefill_operator_panel_input_norm", layer) ||
        !fp8_project(attention->in_proj_qkv, normalized, qkv,
                     gdn.input_projection_temporary,
                     "prefill_operator_panel_linear_qkv", layer) ||
        !fp8_project(attention->in_proj_z, normalized, z,
                     gdn.input_projection_temporary,
                     "prefill_operator_panel_linear_z", layer)) {
      return fail_enqueue(launch_failure);
    }

    const ExactPrefillProjectionWorkspace gdn_input_workspace =
        fp8_workspace(gdn.input_projection_temporary);
    if (!check_cuda(launch_exact_contract_bf16_projection_pair(
                        projection_backend_, attention->in_proj_a,
                        attention->in_proj_b, normalized, a, b,
                        arithmetic_ledger, gdn_input_workspace.reduction,
                        gdn_input_workspace.reduction_elements, stream_),
                    "prefill_operator_panel_linear_ab", layer)) {
      return fail_enqueue(launch_failure);
    }

    for (std::size_t span_index = 0U;
         span_index < arithmetic_ledger.span_count; ++span_index) {
      const LayerMajorPrefillArithmeticSpan& span =
          arithmetic_ledger.spans[span_index];
      const std::size_t segment_offset = span.token_offset;
      const std::size_t segment_token_count = span.token_count;
      std::uint16_t* const qkv_segment =
          qkv + segment_offset * kLinearQkvElements;
      if (!check_cuda(
              gdn_prefill_whole_span_conv_detail::
                  launch_causal_conv1d_silu_update_whole_span_exact_cuda(
                      qkv_segment, segment_token_count,
                      attention->conv1d.data, views_.conv_state[layer],
                      qkv_segment, stream_),
              "prefill_operator_panel_linear_conv", layer) ||
          !check_cuda(
              gdn_prefill_chunk64_native_detail::launch_fixed_production(
                  gdn.native_c64_workspace.device_data,
                  static_cast<std::size_t>(
                      gdn.native_c64_workspace.byte_size),
                  qkv_segment, segment_token_count,
                  a + segment_offset * kLinearScalarElements,
                  b + segment_offset * kLinearScalarElements,
                  attention->a_log.data, attention->dt_bias.data,
                  views_.gdn_state[layer], views_.gdn_state[layer],
                  kRmsEpsilon, attention->norm.data,
                  z + segment_offset * kLinearValueElements, kRmsEpsilon,
                  recurrent + segment_offset * kLinearValueElements,
                  stream_),
              "prefill_operator_panel_linear_gdn", layer)) {
        return fail_enqueue(launch_failure);
      }
      ++g_prefill_gdn_chunk64_native_admission_hits;
    }
    if (!fp8_project(attention->out_proj, recurrent,
                     attention_branch_output,
                     gdn.output_projection_temporary,
                     "prefill_operator_panel_linear_o", layer) ||
        !record_layer_route(PrefillLayerRouteSlot::kQOrLinearQkv) ||
        !record_layer_route(PrefillLayerRouteSlot::kLinearZ) ||
        !record_layer_route(PrefillLayerRouteSlot::kO) ||
        !record_layer_route(PrefillLayerRouteSlot::kGdn)) {
      return fail_enqueue(
          launch_failure.ok()
              ? runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                              "prefill_operator_panel_linear_route", layer)
              : launch_failure);
    }
  } else if (expected == model::LayerType::kFullAttention) {
    const auto* const attention =
        std::get_if<FullAttentionWeights>(&layer_weights.attention);
    const LayerMajorAttentionPhaseViews& phase = request_views.attention;
    if (attention == nullptr ||
        !valid_matrix(phase.normalized_input_bf16, kReferenceHiddenSize,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(phase.raw_q_gate_bf16, kFullQGateElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(phase.processed_q_bf16, kFullQueryElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(phase.packed_gate_bf16, kFullQueryElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(phase.core_output_bf16, kFullQueryElements,
                      sizeof(std::uint16_t)) ||
        !valid_matrix(phase.branch_output_bf16, kReferenceHiddenSize,
                      sizeof(std::uint16_t)) ||
        !valid_fp8_temporary(phase.input_projection_temporary) ||
        !valid_fp8_temporary(phase.output_projection_temporary) ||
        views_.key_cache[layer] == nullptr ||
        views_.value_cache[layer] == nullptr || views_.rope_cos == nullptr ||
        views_.rope_sin == nullptr) {
      return fail_enqueue(runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_operator_panel_attention_views", layer));
    }
    auto* const normalized = static_cast<std::uint16_t*>(
        phase.normalized_input_bf16.storage.device_data);
    auto* const raw_q_gate = static_cast<std::uint16_t*>(
        phase.raw_q_gate_bf16.storage.device_data);
    auto* const processed_q = static_cast<std::uint16_t*>(
        phase.processed_q_bf16.storage.device_data);
    auto* const packed_gate = static_cast<std::uint16_t*>(
        phase.packed_gate_bf16.storage.device_data);
    auto* const core_output = static_cast<std::uint16_t*>(
        phase.core_output_bf16.storage.device_data);
    attention_branch_output = static_cast<std::uint16_t*>(
        phase.branch_output_bf16.storage.device_data);
    auto* const tile_key =
        views_.key_cache[layer] +
        static_cast<std::size_t>(first_position) * kFullKvElements;
    auto* const tile_value =
        views_.value_cache[layer] +
        static_cast<std::size_t>(first_position) * kFullKvElements;
    if (!check_cuda(
            launch_headwise_centered_rms_norm_reference_cuda(
                prompt_residual, layer_weights.input_layernorm.data,
                token_count, kReferenceHiddenSize, kRmsEpsilon, normalized,
                stream_),
            "prefill_operator_panel_input_norm", layer) ||
        !fp8_project(attention->q_proj, normalized, raw_q_gate,
                     phase.input_projection_temporary,
                     "prefill_operator_panel_full_q", layer) ||
        !fp8_project(attention->k_proj, normalized, tile_key,
                     phase.input_projection_temporary,
                     "prefill_operator_panel_full_k", layer) ||
        !fp8_project(attention->v_proj, normalized, tile_value,
                     phase.input_projection_temporary,
                     "prefill_operator_panel_full_v", layer)) {
      return fail_enqueue(launch_failure);
    }

    for (std::size_t span_index = 0U;
         span_index < arithmetic_ledger.span_count; ++span_index) {
      const LayerMajorPrefillArithmeticSpan& span =
          arithmetic_ledger.spans[span_index];
      for (std::size_t local_offset = 0U; local_offset < span.token_count;
           local_offset += kPrefillKernelTileMaximumTokens) {
        const std::size_t subtile_tokens = std::min<std::size_t>(
            span.token_count - local_offset,
            kPrefillKernelTileMaximumTokens);
        const std::size_t token_offset = span.token_offset + local_offset;
        if (!reference_runner_detail::use_full_attention_preprocess_tile(
                static_cast<std::size_t>(first_position) + token_offset,
                subtile_tokens) ||
            !check_cuda(
                launch_full_attention_preprocess_24_4_256_64_reference_256_cuda(
                    raw_q_gate + token_offset * kFullQGateElements,
                    tile_key + token_offset * kFullKvElements,
                    attention->q_norm.data, attention->k_norm.data,
                    kRmsEpsilon,
                    processed_q + token_offset * kFullQueryElements,
                    packed_gate + token_offset * kFullQueryElements,
                    views_.rope_cos, views_.rope_sin,
                    static_cast<std::size_t>(first_position) + token_offset,
                    subtile_tokens, stream_),
                "prefill_operator_panel_full_preprocess", layer)) {
          return fail_enqueue(
              launch_failure.ok()
                  ? runner_status(
                        ReferenceRunnerError::kInvalidStepOptions,
                        "prefill_operator_panel_preprocess_geometry", layer)
                  : launch_failure);
        }
      }
    }

    if (full_attention_tactic == LayerMajorPrefillFullAttentionTactic::
                                     kNativeFlashInferExactPanel) {
      if (!can_launch_bulk_causal_gqa_flashinfer_exact_panel(
              first_position, token_count) ||
          !check_cuda(
              launch_bulk_causal_gqa_sigmoid_gate_24_4_256_flashinfer_exact_panel_fixed_cuda(
                  processed_q, views_.key_cache[layer],
                  views_.value_cache[layer], packed_gate, first_position,
                  token_count, core_output, stream_),
              "prefill_operator_panel_full_attention_flashinfer_exact",
              layer)) {
        return fail_enqueue(
            launch_failure.ok()
                ? runner_status(ReferenceRunnerError::kInvalidStepOptions,
                                "prefill_operator_panel_attention_geometry",
                                layer)
                : launch_failure);
      }
    } else if (full_attention_tactic == LayerMajorPrefillFullAttentionTactic::
                                            kNativeGroupQ128V4Panel) {
      if (!can_launch_bulk_causal_gqa_group_q128_v4_panel(first_position,
                                                           token_count) ||
          !check_cuda(
              launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q128_v4_panel_fixed_cuda(
                  processed_q, views_.key_cache[layer],
                  views_.value_cache[layer], packed_gate, first_position,
                  token_count, core_output, stream_),
              "prefill_operator_panel_full_attention_group_q128_v4",
              layer)) {
        return fail_enqueue(
            launch_failure.ok()
                ? runner_status(ReferenceRunnerError::kInvalidStepOptions,
                                "prefill_operator_panel_attention_geometry",
                                layer)
                : launch_failure);
      }
    } else if (full_attention_tactic ==
               LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel) {
      if (!can_launch_bulk_causal_gqa_group_q64_panel(first_position,
                                                       token_count) ||
          !check_cuda(
              launch_bulk_causal_gqa_sigmoid_gate_24_4_256_group_q64_panel_fixed_cuda(
                  processed_q, views_.key_cache[layer],
                  views_.value_cache[layer], packed_gate, first_position,
                  token_count, core_output, stream_),
              "prefill_operator_panel_full_attention_group_q64", layer)) {
        return fail_enqueue(
            launch_failure.ok()
                ? runner_status(ReferenceRunnerError::kInvalidStepOptions,
                                "prefill_operator_panel_attention_geometry",
                                layer)
                : launch_failure);
      }
    } else {
      for (std::size_t span_index = 0U;
           span_index < arithmetic_ledger.span_count; ++span_index) {
        const LayerMajorPrefillArithmeticSpan& span =
            arithmetic_ledger.spans[span_index];
        const std::size_t segment_offset = span.token_offset;
        const std::size_t segment_token_count = span.token_count;
        const std::size_t segment_first_position =
            static_cast<std::size_t>(first_position) + segment_offset;
        if (!reference_runner_detail::
                use_bulk_causal_gqa_sigmoid_gate_prefill(
                    projection_backend_, expected, segment_first_position,
                    segment_token_count) ||
            !check_cuda(
                launch_bulk_causal_gqa_sigmoid_gate_24_4_256_fixed_cuda(
                    processed_q + segment_offset * kFullQueryElements,
                    views_.key_cache[layer], views_.value_cache[layer],
                    packed_gate + segment_offset * kFullQueryElements,
                    segment_first_position, segment_token_count,
                    core_output + segment_offset * kFullQueryElements,
                    stream_),
                "prefill_operator_panel_full_attention", layer)) {
          return fail_enqueue(
              launch_failure.ok()
                  ? runner_status(
                        ReferenceRunnerError::kInvalidStepOptions,
                        "prefill_operator_panel_attention_geometry", layer)
                  : launch_failure);
        }
      }
    }
    if (!fp8_project(attention->o_proj, core_output,
                     attention_branch_output,
                     phase.output_projection_temporary,
                     "prefill_operator_panel_full_o", layer) ||
        !record_layer_route(PrefillLayerRouteSlot::kQOrLinearQkv) ||
        !record_layer_route(PrefillLayerRouteSlot::kFullK) ||
        !record_layer_route(PrefillLayerRouteSlot::kFullV) ||
        !record_layer_route(PrefillLayerRouteSlot::kO) ||
        !record_layer_route(PrefillLayerRouteSlot::kAttention)) {
      return fail_enqueue(
          launch_failure.ok()
              ? runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                              "prefill_operator_panel_attention_route", layer)
              : launch_failure);
    }
  } else {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidLayerSchedule,
        "prefill_operator_panel_layer_schedule", layer));
  }

  if (defer_mlp) {
#if !defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_operator_panel_persistent_p40_binary", layer));
#else
    const auto bit = [](const PrefillLayerRouteSlot slot) noexcept {
      return static_cast<std::uint16_t>(
          1U << static_cast<std::uint8_t>(slot));
    };
    const std::uint16_t expected_without_mlp =
        static_cast<std::uint16_t>(
            expected_prefill_layer_route_slots(layer) &
            static_cast<std::uint16_t>(
                ~(bit(PrefillLayerRouteSlot::kNvFp4GateUp) |
                  bit(PrefillLayerRouteSlot::kNvFp4Down))));
    if (attention_branch_output == nullptr ||
        layer_route_fragment.recorded_slots != expected_without_mlp ||
        !check_cuda(
            launch_residual_add_reference_cuda(
                prompt_residual, attention_branch_output,
                token_count * kReferenceHiddenSize, prompt_residual, stream_),
            "prefill_operator_panel_attention_residual_deferred_mlp",
            layer)) {
      return fail_enqueue(
          launch_failure.ok()
              ? runner_status(ReferenceRunnerError::kRouteEvidenceFailure,
                              "prefill_operator_panel_deferred_mlp_route",
                              layer)
              : launch_failure);
    }
    PrefillLayerSegmentEnqueueResult result;
    result.route_fragment.layer_segment = layer_route_fragment;
    result.route_fragment.has_single_layer_segment = true;
    result.persistent_p40_fp8_projection_hits =
        persistent_p40_fp8_projection_hits;
    result.persistent_p40_fp8_projection_bulk_hits =
        persistent_p40_fp8_projection_bulk_hits;
    result.persistent_p40_fp8_projection_oracle_partial_hits =
        persistent_p40_fp8_projection_oracle_partial_hits;
    result.persistent_p40_fp8_projection_physical_launches =
        persistent_p40_fp8_projection_physical_launches;
    result.mlp_deferred = true;
    return result;
#endif
  }

  const LayerMajorMlpPhaseViews& mlp = request_views.mlp;
  const LayerMajorLegacyC512Views& oracle = request_views.legacy_c512;
  const auto valid_oracle_matrix = [](const DeviceMatrixView& view,
                                      const std::size_t columns) noexcept {
    return view.storage.device_data != nullptr &&
           view.storage.element_size_bytes == sizeof(std::uint16_t) &&
           view.row_capacity == kMaximumRequestPrefillChunkSize &&
           view.columns == columns && view.row_stride_elements == columns &&
           view.storage.element_capacity >=
               kMaximumRequestPrefillChunkSize * columns;
  };
  bool oracle_views_valid =
      valid_oracle_matrix(oracle.hidden_bf16[1], kReferenceHiddenSize) &&
      valid_oracle_matrix(oracle.projection_bf16[0],
                          kReferenceIntermediateSize) &&
      valid_oracle_matrix(oracle.projection_bf16[1],
                          kReferenceIntermediateSize) &&
      valid_oracle_matrix(oracle.projection_bf16[2],
                          kReferenceIntermediateSize) &&
      valid_oracle_matrix(oracle.projection_bf16[3],
                          kReferenceIntermediateSize) &&
      oracle.fp32_scratch.device_data != nullptr &&
      oracle.fp32_scratch.element_size_bytes == sizeof(float) &&
      oracle.fp32_scratch.element_capacity >=
          kernels::kSm87NvFp4MarlinReductionElements;
  auto* const oracle_merged_gate_up = static_cast<std::uint16_t*>(
      oracle.projection_bf16[0].storage.device_data);
  auto* const oracle_up = static_cast<std::uint16_t*>(
      oracle.projection_bf16[1].storage.device_data);
  auto* const oracle_activated = static_cast<std::uint16_t*>(
      oracle.projection_bf16[2].storage.device_data);
  auto* const oracle_locks = reinterpret_cast<std::int32_t*>(
      oracle.projection_bf16[3].storage.device_data);
  auto* const oracle_branch = static_cast<std::uint16_t*>(
      oracle.hidden_bf16[1].storage.device_data);
  oracle_views_valid =
      oracle_views_valid &&
      reinterpret_cast<std::uintptr_t>(oracle_up) ==
          reinterpret_cast<std::uintptr_t>(oracle_merged_gate_up) +
              oracle.projection_bf16[0].storage.byte_size;
  auto* const segmented_merged_gate_up = static_cast<std::uint16_t*>(
      mlp.gate_bf16.storage.device_data);
  auto* const native_up = static_cast<std::uint16_t*>(
      mlp.up_bf16.storage.device_data);
  auto* const segmented_activated = static_cast<std::uint16_t*>(
      mlp.activated_bf16.storage.device_data);
  // G2 fuses Gate, Up, BF16 branch rounding, and SiLU.  Its output must be
  // disjoint from normalized_input_bf16 for the full lifetime of the launch.
  // The legacy gate span is dead in this route and has the exact activated
  // matrix capacity, whereas segmented_activated aliases the input span.
  auto* const g2_d2_activated = segmented_merged_gate_up;
  auto* const segmented_branch = static_cast<std::uint16_t*>(
      mlp.branch_output_bf16.storage.device_data);
  const bool native_panel_mlp_views_valid =
      valid_matrix(mlp.gate_bf16, kReferenceIntermediateSize,
                   sizeof(std::uint16_t)) &&
      valid_matrix(mlp.up_bf16, kReferenceIntermediateSize,
                   sizeof(std::uint16_t)) &&
      valid_matrix(mlp.activated_bf16, kReferenceIntermediateSize,
                   sizeof(std::uint16_t)) &&
      valid_matrix(mlp.branch_output_bf16, kReferenceHiddenSize,
                   sizeof(std::uint16_t)) &&
      reinterpret_cast<std::uintptr_t>(native_up) ==
          reinterpret_cast<std::uintptr_t>(segmented_merged_gate_up) +
              mlp.gate_bf16.storage.byte_size;
  const bool marlin_panel_mlp_views_valid =
      native_panel_mlp_views_valid &&
      valid_nvfp4_temporary(mlp.gate_up_projection_temporary) &&
      valid_nvfp4_temporary(mlp.down_projection_temporary);
  const bool segmented_projection =
      projection_tactic ==
      LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel;
  const bool native_large_m_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeQuantizedLargeMOperatorPanel;
  const bool true_large_m_nvfp4_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeNvfp4TrueLargeMOperatorPanel;
  const bool g2_d2_nvfp4_projection =
      projection_tactic == LayerMajorPrefillProjectionTactic::
                               kNativeNvfp4G2D2LargeMOperatorPanel;
  const bool native_large_m_partial_panel =
      native_large_m_projection &&
      token_count != kLayerMajorPrefillOperatorPanelTokens;
  const bool operator_panel_projection =
      segmented_projection || native_large_m_projection ||
      true_large_m_nvfp4_projection || g2_d2_nvfp4_projection;
  const bool g2_d2_mlp_views_valid =
      valid_matrix(mlp.gate_bf16, kReferenceIntermediateSize,
                   sizeof(std::uint16_t)) &&
      mlp.gate_bf16.storage.device_data !=
          mlp.normalized_input_bf16.storage.device_data;
  const bool selected_mlp_views_valid =
      g2_d2_nvfp4_projection
          ? g2_d2_mlp_views_valid
      : true_large_m_nvfp4_projection
          ? native_panel_mlp_views_valid
          : (segmented_projection || native_large_m_projection)
                ? marlin_panel_mlp_views_valid
                : oracle_views_valid;
  if (attention_branch_output == nullptr ||
      !valid_matrix(mlp.normalized_input_bf16, kReferenceHiddenSize,
                    sizeof(std::uint16_t)) ||
      !selected_mlp_views_valid ||
      (native_large_m_partial_panel && !oracle_views_valid)) {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidRequestState,
        "prefill_operator_panel_mlp_views", layer));
  }
  auto* const mlp_normalized = static_cast<std::uint16_t*>(
      mlp.normalized_input_bf16.storage.device_data);
  constexpr std::size_t kOracleMergedGateUpCapacityBytes =
      2U * kMaximumRequestPrefillChunkSize *
      kReferenceIntermediateSize * sizeof(std::uint16_t);
  if (!operator_panel_projection &&
      kOracleMergedGateUpCapacityBytes >
          oracle.projection_bf16[0].storage.byte_size +
              oracle.projection_bf16[1].storage.byte_size) {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidRequestState,
        "prefill_operator_panel_exact_mlp_capacity", layer));
  }
  const auto* const marlin_gate =
      std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.gate_proj);
  const auto* const marlin_up =
      std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.up_proj);
  const auto* const marlin_down =
      std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.down_proj);
  const bool projection_buffers_aligned =
      segmented_projection
          ? aligned_16(segmented_merged_gate_up) &&
                aligned_16(segmented_activated) &&
                aligned_16(segmented_branch) &&
                aligned_16(mlp.gate_up_projection_temporary.device_data) &&
                aligned_16(mlp.down_projection_temporary.device_data)
      : native_large_m_projection
          ? aligned_16(segmented_merged_gate_up) && aligned_16(native_up) &&
                aligned_16(segmented_activated) &&
                aligned_16(segmented_branch) &&
                aligned_16(mlp.gate_up_projection_temporary.device_data) &&
                aligned_16(mlp.down_projection_temporary.device_data) &&
                (!native_large_m_partial_panel ||
                 (aligned_16(oracle_merged_gate_up) &&
                  aligned_16(oracle_activated) &&
                  aligned_16(oracle_branch) &&
                  aligned_16(oracle.fp32_scratch.device_data) &&
                  aligned_16(oracle_locks)))
      : g2_d2_nvfp4_projection
          ? aligned_16(g2_d2_activated) && aligned_16(prompt_residual)
      : true_large_m_nvfp4_projection
          ? aligned_16(segmented_merged_gate_up) && aligned_16(native_up) &&
                aligned_16(segmented_activated) && aligned_16(segmented_branch)
          : aligned_16(oracle_merged_gate_up) &&
                aligned_16(oracle_activated) && aligned_16(oracle_branch) &&
                aligned_16(oracle.fp32_scratch.device_data) &&
                aligned_16(oracle_locks);
  if (marlin_gate == nullptr || marlin_up == nullptr ||
      marlin_down == nullptr ||
      !valid_exact_contract_nvfp4_mlp_weights(
          *marlin_gate, *marlin_up, *marlin_down) ||
      !aligned_16(mlp_normalized) || !projection_buffers_aligned) {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidModelWeights,
        "prefill_operator_panel_mlp_inventory", layer));
  }
  if (!check_cuda(
          launch_residual_add_reference_cuda(
              prompt_residual, attention_branch_output,
              token_count * kReferenceHiddenSize, prompt_residual, stream_),
          "prefill_operator_panel_attention_residual", layer) ||
      !check_cuda(
          launch_headwise_centered_rms_norm_reference_cuda(
              prompt_residual,
              layer_weights.post_attention_layernorm.data, token_count,
              kReferenceHiddenSize, kRmsEpsilon, mlp_normalized, stream_),
          "prefill_operator_panel_post_attention_norm", layer)) {
    return fail_enqueue(launch_failure);
  }
  if (segmented_projection) {
    const ExactPrefillProjectionWorkspace gate_up_workspace =
        nvfp4_workspace(mlp.gate_up_projection_temporary);
    const ExactPrefillProjectionWorkspace down_workspace =
        nvfp4_workspace(mlp.down_projection_temporary);
    if (!check_cuda(
            launch_segmented_panel_nvfp4_mlp(
                *marlin_gate, *marlin_up, *marlin_down, mlp_normalized,
                segmented_merged_gate_up, segmented_activated, segmented_branch,
                token_count, gate_up_workspace, down_workspace,
                segmented_panel_projection_hits,
                segmented_panel_projection_physical_launches, stream_),
            "prefill_operator_panel_native_marlin_mlp", layer) ||
        !check_cuda(
            launch_residual_add_reference_cuda(
                prompt_residual, segmented_branch,
                token_count * kReferenceHiddenSize, prompt_residual, stream_),
            "prefill_operator_panel_native_marlin_residual", layer)) {
      return fail_enqueue(launch_failure);
    }
  } else if (g2_d2_nvfp4_projection) {
#if defined(Q3X_ENABLE_NVFP4_G2_D2_PREFILL_ADMISSION)
    const char* g2_d2_failure_operation =
        "prefill_operator_panel_nvfp4_g2_d2_mlp";
    if (!check_cuda(
            launch_nvfp4_g2_d2_large_m_mlp(
                *marlin_gate, *marlin_up, *marlin_down, mlp_normalized,
                g2_d2_activated, prompt_residual, token_count,
                native_nvfp4_true_large_m_projection_hits,
                native_nvfp4_true_large_m_gate_up_hits,
                native_nvfp4_true_large_m_down_hits,
                native_nvfp4_true_large_m_physical_launches,
                &g2_d2_failure_operation, stream_),
            g2_d2_failure_operation, layer)) {
      return fail_enqueue(launch_failure);
    }
#else
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_operator_panel_nvfp4_g2_d2_binary", layer));
#endif
  } else if (true_large_m_nvfp4_projection) {
#if defined(Q3X_ENABLE_NVFP4_TRUE_LARGE_M_PREFILL_ADMISSION)
    if (!check_cuda(
            launch_nvfp4_true_large_m_mlp(
                *marlin_gate, *marlin_up, *marlin_down, mlp_normalized,
                segmented_merged_gate_up, segmented_activated,
                segmented_branch, token_count,
                native_nvfp4_true_large_m_projection_hits,
                native_nvfp4_true_large_m_gate_up_hits,
                native_nvfp4_true_large_m_down_hits,
                native_nvfp4_true_large_m_physical_launches, stream_),
            "prefill_operator_panel_nvfp4_true_large_m_mlp", layer) ||
        !check_cuda(
            launch_residual_add_reference_cuda(
                prompt_residual, segmented_branch,
                token_count * kReferenceHiddenSize, prompt_residual, stream_),
            "prefill_operator_panel_nvfp4_true_large_m_residual", layer)) {
      return fail_enqueue(launch_failure);
    }
#else
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_operator_panel_nvfp4_true_large_m_binary", layer));
#endif
  } else if (native_large_m_projection) {
    if (token_count != kLayerMajorPrefillOperatorPanelTokens) {
      // Only real-model M8192 has passed whole-state bitwise admission.
      // Preserve the complete established sequence for every partial panel:
      // the legacy C512 output/reduction arena and per-span Down->residual
      // interleave are part of the bitwise oracle.
      ExactPrefillProjectionWorkspace oracle_workspace;
      oracle_workspace.reduction =
          static_cast<float*>(oracle.fp32_scratch.device_data);
      oracle_workspace.reduction_elements =
          static_cast<std::size_t>(oracle.fp32_scratch.element_capacity);
      oracle_workspace.locks = oracle_locks;
      oracle_workspace.lock_bytes = kernels::kSm87NvFp4MarlinLockBytes;
      std::size_t fallback_physical_launches = 0U;
      for (std::size_t span_index = 0U;
           span_index < arithmetic_ledger.span_count; ++span_index) {
        const LayerMajorPrefillArithmeticSpan& span =
            arithmetic_ledger.spans[span_index];
        const LayerMajorPrefillArithmeticSpanLedger local_ledger =
            make_layer_major_prefill_arithmetic_span_ledger(span.token_count);
        const auto plan =
            kernels::sm87_nvfp4_marlin_execution_plan(span.token_count);
        if (local_ledger.span_count != 1U ||
            local_ledger.spans[0].token_offset != 0U ||
            local_ledger.spans[0].token_count != span.token_count ||
            plan.launch_count == 0U ||
            !check_cuda(
                launch_exact_contract_nvfp4_mlp(
                    *marlin_gate, *marlin_up, *marlin_down,
                    mlp_normalized +
                        static_cast<std::size_t>(span.token_offset) *
                            kReferenceHiddenSize,
                    oracle_merged_gate_up, oracle_activated, oracle_branch,
                    local_ledger, oracle_workspace, oracle_workspace, stream_),
                "prefill_operator_panel_native_large_m_exact_span_mlp",
                layer) ||
            !check_cuda(
                launch_residual_add_reference_cuda(
                    prompt_residual +
                        static_cast<std::size_t>(span.token_offset) *
                            kReferenceHiddenSize,
                    oracle_branch,
                    static_cast<std::size_t>(span.token_count) *
                        kReferenceHiddenSize,
                    prompt_residual +
                        static_cast<std::size_t>(span.token_offset) *
                            kReferenceHiddenSize,
                    stream_),
                "prefill_operator_panel_native_large_m_exact_span_residual",
                layer)) {
          return fail_enqueue(
              launch_failure.ok()
                  ? runner_status(
                        ReferenceRunnerError::kInvalidStepOptions,
                        "prefill_operator_panel_native_large_m_exact_span_ledger",
                        layer)
                  : launch_failure);
        }
        fallback_physical_launches += 2U * plan.launch_count;
      }
      native_large_m_projection_hits += 2U;
      native_large_m_projection_oracle_partial_hits += 2U;
      native_large_m_projection_physical_launches +=
          fallback_physical_launches;
    } else {
      const ExactMarlinOperatorPanelPlan panel_plan =
          exact_marlin_operator_panel_plan(arithmetic_ledger);
      if (!panel_plan.valid(arithmetic_ledger)) {
        return fail_enqueue(runner_status(
            ReferenceRunnerError::kInvalidStepOptions,
            "prefill_operator_panel_native_large_m_prefix_plan", layer));
      }
      const ExactPrefillProjectionWorkspace gate_up_workspace =
          nvfp4_workspace(mlp.gate_up_projection_temporary);
      const ExactPrefillProjectionWorkspace down_workspace =
          nvfp4_workspace(mlp.down_projection_temporary);
      if (!check_cuda(
              launch_native_large_m_nvfp4_mlp(
                  *marlin_gate, *marlin_up, *marlin_down, mlp_normalized,
                  segmented_merged_gate_up, segmented_activated,
                  segmented_branch, panel_plan.aligned_prefix_tokens,
                  gate_up_workspace,
                  down_workspace, native_large_m_projection_hits,
                  native_large_m_projection_physical_launches, stream_),
              "prefill_operator_panel_native_large_m_mlp", layer) ||
          !check_cuda(
              launch_residual_add_reference_cuda(
                  prompt_residual, segmented_branch,
                  panel_plan.aligned_prefix_tokens * kReferenceHiddenSize,
                  prompt_residual, stream_),
              "prefill_operator_panel_native_large_m_residual", layer)) {
        return fail_enqueue(launch_failure);
      }
      native_large_m_projection_bulk_hits += 2U;
    }
  } else {
    ExactPrefillProjectionWorkspace oracle_workspace;
    oracle_workspace.reduction =
        static_cast<float*>(oracle.fp32_scratch.device_data);
    oracle_workspace.reduction_elements =
        static_cast<std::size_t>(oracle.fp32_scratch.element_capacity);
    oracle_workspace.locks = oracle_locks;
    oracle_workspace.lock_bytes = kernels::kSm87NvFp4MarlinLockBytes;
    for (std::size_t span_index = 0U;
         span_index < arithmetic_ledger.span_count; ++span_index) {
      const LayerMajorPrefillArithmeticSpan& span =
          arithmetic_ledger.spans[span_index];
      const LayerMajorPrefillArithmeticSpanLedger local_ledger =
          make_layer_major_prefill_arithmetic_span_ledger(span.token_count);
      if (local_ledger.span_count != 1U ||
          local_ledger.spans[0].token_offset != 0U ||
          local_ledger.spans[0].token_count != span.token_count ||
          !check_cuda(
              launch_exact_contract_nvfp4_mlp(
                  *marlin_gate, *marlin_up, *marlin_down,
                  mlp_normalized +
                      static_cast<std::size_t>(span.token_offset) *
                          kReferenceHiddenSize,
                  oracle_merged_gate_up, oracle_activated, oracle_branch,
                  local_ledger, oracle_workspace, oracle_workspace, stream_),
              "prefill_operator_panel_exact_span_mlp", layer) ||
          !check_cuda(
              launch_residual_add_reference_cuda(
                  prompt_residual +
                      static_cast<std::size_t>(span.token_offset) *
                          kReferenceHiddenSize,
                  oracle_branch,
                  static_cast<std::size_t>(span.token_count) *
                      kReferenceHiddenSize,
                  prompt_residual +
                      static_cast<std::size_t>(span.token_offset) *
                          kReferenceHiddenSize,
                  stream_),
              "prefill_operator_panel_exact_span_residual", layer)) {
        return fail_enqueue(
            launch_failure.ok()
                ? runner_status(ReferenceRunnerError::kInvalidStepOptions,
                                "prefill_operator_panel_exact_span_ledger",
                                layer)
                : launch_failure);
      }
    }
  }
  if (!true_large_m_nvfp4_projection && !g2_d2_nvfp4_projection) {
    ++g_nvfp4_marlin_prefill_admission_hits;
  }
  if (!record_layer_route(PrefillLayerRouteSlot::kNvFp4GateUp) ||
      !record_layer_route(PrefillLayerRouteSlot::kNvFp4Down)) {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kRouteEvidenceFailure,
        "prefill_operator_panel_mlp_route", layer));
  }
  const ReferenceRunnerStatus route_status =
      validate_prefill_layer_route_fragment(layer_route_fragment);
  if (!route_status) {
    return fail_enqueue(route_status);
  }
  PrefillLayerSegmentEnqueueResult result;
  result.route_fragment.layer_segment = layer_route_fragment;
  result.route_fragment.has_single_layer_segment = true;
  result.segmented_panel_projection_hits = segmented_panel_projection_hits;
  result.segmented_panel_projection_physical_launches =
      segmented_panel_projection_physical_launches;
  result.native_large_m_projection_hits = native_large_m_projection_hits;
  result.native_large_m_projection_bulk_hits =
      native_large_m_projection_bulk_hits;
  result.native_large_m_projection_oracle_partial_hits =
      native_large_m_projection_oracle_partial_hits;
  result.native_large_m_projection_physical_launches =
      native_large_m_projection_physical_launches;
  result.nvfp4_true_large_m_route_fp8_projection_hits =
      nvfp4_true_large_m_route_fp8_projection_hits;
  result.nvfp4_true_large_m_route_fp8_projection_bulk_hits =
      nvfp4_true_large_m_route_fp8_projection_bulk_hits;
  result.nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits =
      nvfp4_true_large_m_route_fp8_projection_oracle_partial_hits;
  result.nvfp4_true_large_m_route_fp8_projection_physical_launches =
      nvfp4_true_large_m_route_fp8_projection_physical_launches;
  result.native_nvfp4_true_large_m_projection_hits =
      native_nvfp4_true_large_m_projection_hits;
  result.native_nvfp4_true_large_m_gate_up_hits =
      native_nvfp4_true_large_m_gate_up_hits;
  result.native_nvfp4_true_large_m_down_hits =
      native_nvfp4_true_large_m_down_hits;
  result.native_nvfp4_true_large_m_physical_launches =
      native_nvfp4_true_large_m_physical_launches;
  return result;
#endif
}

ReferenceRunnerStatus ReferenceRunner::enqueue_layer_wide_p40_mlp(
    const std::size_t layer,
    const ReferenceLayerMajorRequestViews& request_views,
    const PromptWideP40ProjectionPackage projection_package) noexcept {
#if !defined(Q3X_ENABLE_NVFP4_PERSISTENT_PREFILL_ADMISSION)
  (void)layer;
  (void)request_views;
  (void)projection_package;
  return runner_status(ReferenceRunnerError::kInvalidDependency,
                       "prefill_layer_wide_p40_mlp_binary");
#else
  const bool use_projection_reset =
      projection_package == PromptWideP40ProjectionPackage::kProjectionResetV11;
  const bool use_packed_projection =
      projection_package == PromptWideP40ProjectionPackage::kPackedV1;
  const bool use_packed_nvfp4_v2 =
      projection_package == PromptWideP40ProjectionPackage::kPackedNvfp4V2;
  const bool use_any_packed_projection =
      use_packed_projection || use_packed_nvfp4_v2;
  if (layer >= kReferenceDecoderLayerCount ||
      request_views.descriptor.mlp_layout !=
          LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan ||
      request_views.descriptor.mlp_capacity_tokens !=
          kLayerMajorPrefillLayerWideMlpP40Tokens ||
      request_views.descriptor.max_sequence_length !=
          kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "prefill_layer_wide_p40_mlp_descriptor", layer);
  }
  const DeviceMatrixView& residual_view =
      request_views.prompt_residual_bf16;
  const LayerMajorMlpPhaseViews& mlp = request_views.mlp;
  const auto exact_bf16_matrix = [](const DeviceMatrixView& view,
                                    const std::uint32_t rows,
                                    const std::uint32_t columns) noexcept {
    return view.storage.device_data != nullptr &&
           view.storage.element_size_bytes == sizeof(std::uint16_t) &&
           view.row_capacity == rows && view.columns == columns &&
           view.row_stride_elements == columns &&
           view.storage.element_capacity ==
               static_cast<std::uint64_t>(rows) * columns;
  };
  if (residual_view.storage.device_data == nullptr ||
      residual_view.storage.element_size_bytes != sizeof(std::uint16_t) ||
      residual_view.row_capacity !=
          kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens ||
      residual_view.columns != kReferenceHiddenSize ||
      residual_view.row_stride_elements != kReferenceHiddenSize ||
      !exact_bf16_matrix(mlp.normalized_input_bf16,
                         kLayerMajorPrefillLayerWideMlpP40Tokens,
                         kReferenceHiddenSize) ||
      !exact_bf16_matrix(mlp.gate_bf16,
                         kLayerMajorPrefillLayerWideMlpP40Tokens,
                         kReferenceIntermediateSize)) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "prefill_layer_wide_p40_mlp_views", layer);
  }
  const DecoderLayerWeights& layer_weights = weights_->layer(layer);
  const auto* const gate =
      std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.gate_proj);
  const auto* const up =
      std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.up_proj);
  const auto* const down =
      std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.down_proj);
  if (gate == nullptr || up == nullptr || down == nullptr ||
      !valid_vector(layer_weights.post_attention_layernorm,
                    kReferenceHiddenSize) ||
      (use_any_packed_projection
#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
           ? !valid_p40_packed_nvfp4_mlp_weights(*gate, *up, *down)
#else
           ? true
#endif
           : !valid_persistent_p40_nvfp4_mlp_weights(*gate, *up, *down))) {
    return runner_status(ReferenceRunnerError::kInvalidModelWeights,
                         "prefill_layer_wide_p40_mlp_inventory", layer);
  }
  auto* const residual = static_cast<std::uint16_t*>(
      residual_view.storage.device_data);
  auto* const normalized = static_cast<std::uint16_t*>(
      mlp.normalized_input_bf16.storage.device_data);
  auto* const activated = static_cast<std::uint16_t*>(
      mlp.gate_bf16.storage.device_data);
  constexpr std::size_t kResidualBytes =
      static_cast<std::size_t>(kLayerMajorPrefillLayerWideMlpP40Tokens) *
      kReferenceHiddenSize * sizeof(std::uint16_t);
  constexpr std::size_t kNormalizedBytes = kResidualBytes;
  constexpr std::size_t kActivatedBytes =
      static_cast<std::size_t>(kLayerMajorPrefillLayerWideMlpP40Tokens) *
      kReferenceIntermediateSize * sizeof(std::uint16_t);
  if (!byte_ranges_are_pairwise_disjoint(
          std::array<ByteSpan, 3U>{{{residual, kResidualBytes},
                                   {normalized, kNormalizedBytes},
                                   {activated, kActivatedBytes}}})) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "prefill_layer_wide_p40_mlp_alias", layer);
  }
  const int norm_status =
      launch_headwise_centered_rms_norm_reference_cuda(
          residual, layer_weights.post_attention_layernorm.data,
          kLayerMajorPrefillLayerWideMlpP40Tokens, kReferenceHiddenSize,
          kRmsEpsilon, normalized, stream_);
  if (norm_status != static_cast<int>(cudaSuccess)) {
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         "prefill_layer_wide_p40_post_attention_norm", layer,
                         norm_status);
  }
  int gate_up_status = static_cast<int>(cudaErrorNotSupported);
  if (use_packed_nvfp4_v2) {
#if defined(Q3X_ENABLE_P40_PACKED_NVFP4_V2_ADMISSION)
    gate_up_status = kernels::launch_sm87_p40_packed_nvfp4_v2_gate_up_cuda(
        normalized, gate->prefill_p40_packed_artifact,
        kLayerMajorPrefillLayerWideMlpP40Tokens, activated, stream_);
#endif
  } else if (use_packed_projection) {
#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
    gate_up_status = kernels::launch_sm87_p40_packed_nvfp4_gate_up_cuda(
        normalized, gate->prefill_p40_packed_artifact,
        kLayerMajorPrefillLayerWideMlpP40Tokens, activated, stream_);
#endif
  } else {
    gate_up_status = use_projection_reset
#if defined(Q3X_ENABLE_P40_PROJECTION_RESET_ADMISSION)
      ? kernels::launch_sm87_p40_projection_reset_nvfp4_gate_up_silu_cuda(
            normalized, gate->prefill_marlin_weight,
            gate->prefill_marlin_scales,
            gate->prefill_marlin_global_scale,
            kLayerMajorPrefillLayerWideMlpP40Tokens, activated, stream_)
#else
      ? static_cast<int>(cudaErrorNotSupported)
#endif
      : kernels::launch_sm87_nvfp4_persistent_prefill_gate_up_cuda(
            normalized, gate->prefill_marlin_weight,
            gate->prefill_marlin_scales,
            gate->prefill_marlin_global_scale,
            kLayerMajorPrefillLayerWideMlpP40Tokens, activated, stream_);
  }
  if (gate_up_status != static_cast<int>(cudaSuccess)) {
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         use_packed_nvfp4_v2
                             ? "prefill_p40_packed_nvfp4_v2_gate_up"
                         : use_packed_projection
                             ? "prefill_p40_packed_nvfp4_v1_gate_up"
                             : "prefill_layer_wide_p40_persistent_gate_up",
                         layer,
                         gate_up_status);
  }
  int down_status = static_cast<int>(cudaErrorNotSupported);
  if (use_packed_nvfp4_v2) {
#if defined(Q3X_ENABLE_P40_PACKED_NVFP4_V2_ADMISSION)
    down_status = kernels::launch_sm87_p40_packed_nvfp4_v2_down_cuda(
        activated, down->prefill_p40_packed_artifact,
        kLayerMajorPrefillLayerWideMlpP40Tokens, residual, stream_);
#endif
  } else if (use_packed_projection) {
#if defined(Q3X_ENABLE_P40_PACKED_PROJECTION_ADMISSION)
    down_status = kernels::launch_sm87_p40_packed_nvfp4_down_cuda(
        activated, down->prefill_p40_packed_artifact,
        kLayerMajorPrefillLayerWideMlpP40Tokens, residual, stream_);
#endif
  } else {
    down_status = use_projection_reset
#if defined(Q3X_ENABLE_P40_PROJECTION_RESET_ADMISSION)
      ? kernels::launch_sm87_p40_projection_reset_nvfp4_down_residual_cuda(
            activated, down->prefill_marlin_weight,
            down->prefill_marlin_scales,
            down->prefill_marlin_global_scale,
            kLayerMajorPrefillLayerWideMlpP40Tokens, residual, stream_)
#else
      ? static_cast<int>(cudaErrorNotSupported)
#endif
      : kernels::launch_sm87_nvfp4_persistent_prefill_down_residual_cuda(
            activated, down->prefill_marlin_weight,
            down->prefill_marlin_scales,
            down->prefill_marlin_global_scale,
            kLayerMajorPrefillLayerWideMlpP40Tokens, residual, stream_);
  }
  if (down_status != static_cast<int>(cudaSuccess)) {
    return runner_status(ReferenceRunnerError::kCudaFailure,
                         use_packed_nvfp4_v2
                             ? "prefill_p40_packed_nvfp4_v2_down_residual"
                         : use_packed_projection
                             ? "prefill_p40_packed_nvfp4_v1_down_residual"
                             : "prefill_layer_wide_p40_persistent_down_residual",
                         layer, down_status);
  }
  return {};
#endif
}

ReferenceRunner::PrefillLayerSegmentEnqueueResult
ReferenceRunner::enqueue_prefill_layer_segment(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const std::uint32_t first_position,
    const PrefillTileExecutionControl& control,
    const Views& execution_views) noexcept {
  const auto fail_enqueue = [](const ReferenceRunnerStatus status) noexcept {
    PrefillLayerSegmentEnqueueResult result;
    result.status = status;
    return result;
  };
  const bool legacy_control =
      is_legacy_prefill_tile_execution_control(control);
  const bool any_bound_arithmetic_role =
      control.force_bound_nvfp4_marlin_prefill ||
      control.force_bound_fp8_marlin_prefill ||
      control.force_bound_gdn_chunk64_native_prefill;
  const bool sealed_exact_arithmetic =
      control.force_bound_nvfp4_marlin_prefill &&
      control.force_bound_fp8_marlin_prefill &&
      control.force_bound_gdn_chunk64_native_prefill;
  const LayerMajorPrefillArithmeticSpanLedger arithmetic_ledger =
      make_layer_major_prefill_arithmetic_span_ledger(token_count);
  if (!legacy_control &&
      (control.allow_experimental_gdn_b8_admission ||
       control.allow_experimental_gdn_chunk64_native_admission ||
       control.allow_experimental_gdn_chunk64_reference_admission)) {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "prefill_tile_candidate_experimental_gdn_admission"));
  }
  if (any_bound_arithmetic_role &&
      (!sealed_exact_arithmetic ||
       !is_valid_layer_major_prefill_arithmetic_contract(
           kLayerMajorPrefillExactArithmeticContract) ||
       !is_valid_layer_major_prefill_arithmetic_span_ledger(
           arithmetic_ledger))) {
    return fail_enqueue(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "prefill_tile_bound_arithmetic_contract"));
  }
  PrefillRouteEvidence legacy_layer_pass;
  PrefillLayerSegmentRouteFragment layer_route_fragment;
  const auto record_layer_route = [&layer_route_fragment](
                                      const PrefillLayerRouteSlot slot,
                                      const PrefillRouteDisposition disposition)
      noexcept {
        const std::size_t index = static_cast<std::size_t>(slot);
        if (index >= kPrefillLayerRouteSlotCount ||
            (disposition != PrefillRouteDisposition::kProduction &&
             disposition != PrefillRouteDisposition::kExactFallback &&
             disposition != PrefillRouteDisposition::kForbidden)) {
          return false;
        }
        const std::uint16_t bit =
            static_cast<std::uint16_t>(1U << index);
        if ((layer_route_fragment.recorded_slots & bit) != 0U) {
          return false;
        }
        layer_route_fragment.dispositions[index] = disposition;
        layer_route_fragment.recorded_slots = static_cast<std::uint16_t>(
            layer_route_fragment.recorded_slots | bit);
        return true;
      };
  const auto record_layer_forbidden_boundary =
      [&layer_route_fragment](
          const PrefillForbiddenBoundary boundary) noexcept {
        const std::size_t index = static_cast<std::size_t>(boundary);
        if (index >= kPrefillForbiddenBoundaryCount) {
          return false;
        }
        layer_route_fragment.forbidden_boundaries =
            static_cast<std::uint8_t>(
                layer_route_fragment.forbidden_boundaries |
                static_cast<std::uint8_t>(1U << index));
        return true;
      };
#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
  // Snapshot the admission switch once at the tile boundary. Tests may only
  // change it between synchronous public runner calls.
  const bool enable_gdn_b8_admission =
      control.allow_experimental_gdn_b8_admission &&
      g_enable_prefill_gdn_b8_admission;
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  const bool enable_gdn_chunk64_native_admission =
      (control.allow_experimental_gdn_chunk64_native_admission &&
       g_enable_prefill_gdn_chunk64_native_admission) ||
      control.force_bound_gdn_chunk64_native_prefill;
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  const bool enable_gdn_chunk64_reference_admission =
      control.allow_experimental_gdn_chunk64_reference_admission &&
      g_enable_prefill_gdn_chunk64_reference_admission;
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
  // Snapshot the private switch at the synchronous public-call boundary.
  const bool enable_gdn_c16_norm_gate_admission =
      !sealed_exact_arithmetic &&
      g_enable_prefill_gdn_c16_norm_gate_admission;
#else
  constexpr bool enable_gdn_c16_norm_gate_admission = true;
#endif
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  ReferenceRunnerStatus launch_failure{};
  const auto check_cuda = [&launch_failure](
                              const int status,
                              const char* const operation,
                              const std::size_t layer) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    launch_failure = runner_status(ReferenceRunnerError::kCudaFailure,
                                   operation, layer, status);
    return false;
  };
  bool fp8_marlin_tile_enabled = false;
  std::int32_t* fp8_marlin_locks = nullptr;
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  fp8_marlin_tile_enabled =
      (control.force_bound_fp8_marlin_prefill ||
       g_enable_fp8_marlin_prefill_admission) &&
      kernels::sm87_fp8_marlin_supports_token_count(token_count);
  fp8_marlin_locks =
      reinterpret_cast<std::int32_t*>(execution_views.projection[3]);
#endif
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
  const bool enable_bf16_ab_large_m_prefill_admission =
      !sealed_exact_arithmetic &&
      g_enable_bf16_ab_large_m_prefill_admission;
#else
  constexpr bool enable_bf16_ab_large_m_prefill_admission = false;
#endif
  const auto is_fp8_marlin_projection =
      [fp8_marlin_tile_enabled](const LinearWeight& weight) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
        const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
        return fp8_marlin_tile_enabled && fp8 != nullptr &&
               kernels::sm87_fp8_marlin_supports_shape(
                   fp8->output_size, fp8->input_size);
#else
        (void)weight;
        return false;
#endif
      };
  const auto project_on_stream =
      [this, token_count, &check_cuda, &is_fp8_marlin_projection,
       fp8_marlin_locks, &execution_views, sealed_exact_arithmetic,
       &arithmetic_ledger](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const char* const operation,
          const std::size_t layer, void* const launch_stream,
          PrefillProjectionExecution* const executed_route = nullptr) noexcept {
        if (executed_route != nullptr) {
          *executed_route = PrefillProjectionExecution::kUnknown;
        }
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
        if (is_fp8_marlin_projection(weight)) {
          if (sealed_exact_arithmetic) {
            ExactPrefillProjectionWorkspace workspace;
            workspace.reduction = execution_views.fp32_scratch;
            workspace.reduction_elements =
                execution_views.fp32_scratch_elements;
            workspace.locks = fp8_marlin_locks;
            workspace.lock_bytes = kernels::kSm87Fp8MarlinLockBytes;
            const int status = launch_exact_contract_fp8_projection(
                projection_backend_, weight, input, output,
                arithmetic_ledger, workspace, launch_stream);
            if (status == static_cast<int>(cudaSuccess)) {
              if (executed_route != nullptr) {
                *executed_route = PrefillProjectionExecution::kFp8Marlin;
              }
              return true;
            }
            return check_cuda(status, operation, layer);
          }
          const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
          if (fp8 == nullptr ||
              !reference_runner_detail::use_fp8_marlin_prefill_projection(
                  projection_backend_, weight, input, output, token_count) ||
              fp8_marlin_locks == nullptr ||
              execution_views.fp32_scratch_elements <
                  kernels::kSm87Fp8MarlinReductionElements) {
            return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                              operation, layer);
          }
          const auto stream =
              reinterpret_cast<cudaStream_t>(launch_stream);
          const int clear_status = static_cast<int>(cudaMemsetAsync(
              fp8_marlin_locks, 0, kernels::kSm87Fp8MarlinLockBytes,
              stream));
          if (clear_status != static_cast<int>(cudaSuccess)) {
            return check_cuda(clear_status, operation, layer);
          }
          const int status = kernels::launch_sm87_fp8_marlin_projection_cuda(
              input, fp8->prefill_marlin_weight,
              fp8->prefill_marlin_scales, token_count, fp8->output_size,
              fp8->input_size, output, execution_views.fp32_scratch,
              fp8_marlin_locks, launch_stream);
          if (status == static_cast<int>(cudaSuccess)) {
            ++g_fp8_marlin_prefill_admission_hits;
            if (executed_route != nullptr) {
              *executed_route = PrefillProjectionExecution::kFp8Marlin;
            }
            return true;
          }
          return check_cuda(status, operation, layer);
        }
#endif
        if (reference_runner_detail::use_fp8_whole_chunk_prefill_projection(
                projection_backend_, weight, input, output, token_count)) {
          const int status =
              launch_exact_fp8_whole_chunk_projection_to_bf16_cuda(
                  projection_backend_, weight, input, token_count, output,
                  launch_stream);
          if (status == static_cast<int>(cudaSuccess)) {
            if (executed_route != nullptr) {
              *executed_route = PrefillProjectionExecution::kFp8WholeChunk;
            }
            return true;
          }
          if (status != static_cast<int>(cudaErrorNotSupported)) {
            return check_cuda(status, operation, layer);
          }
        }
        const std::size_t columns = linear_input_size(weight);
        const std::size_t rows = linear_output_size(weight);
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kProductionProjectionSubtileTokens) {
          const std::size_t remaining = token_count - token_offset;
          const std::size_t subtile_tokens =
              remaining < kProductionProjectionSubtileTokens
                  ? remaining
                  : kProductionProjectionSubtileTokens;
          if (!check_cuda(launch_projection_tile_to_bf16_cuda(
                              projection_backend_, weight,
                              input + token_offset * columns, subtile_tokens,
                              execution_views.fp32_scratch,
                              execution_views.fp32_scratch_elements,
                              output + token_offset * rows, launch_stream),
                          operation, layer)) {
            return false;
          }
        }
        if (executed_route != nullptr) {
          *executed_route = PrefillProjectionExecution::kGenericExact;
        }
        return true;
      };
  const auto project = [this, &project_on_stream](
                           const LinearWeight& weight,
                           const std::uint16_t* const input,
                           std::uint16_t* const output,
                           const char* const operation,
                           const std::size_t layer,
                           PrefillProjectionExecution* const executed_route =
                               nullptr) noexcept {
    return project_on_stream(weight, input, output, operation, layer, stream_,
                             executed_route);
  };
  const auto has_fp8_prefill_supermatrix_sidecar =
      [this, token_count,
       fp8_marlin_tile_enabled](const LinearWeight& weight) noexcept {
        const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
        return !fp8_marlin_tile_enabled &&
               projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
               token_count == kMaximumRequestPrefillChunkSize &&
               fp8 != nullptr &&
               fp8->prefill_supermatrix_sidecar != nullptr;
      };
  const auto project_fp8_prefill_supermatrix =
      [this, token_count, &check_cuda](
          const LinearWeight* const* const group_weights,
          std::uint16_t* const* const group_outputs,
          const std::size_t group_size,
          const std::uint16_t* const input,
          const char* const operation,
          const std::size_t layer) noexcept {
        if (group_weights == nullptr || group_outputs == nullptr ||
            group_size == 0U || group_size > 3U) {
          return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                            operation, layer);
        }
        std::array<kernels::Sm87Fp8PrefillSupermatrixPartition, 3U>
            partitions{};
        std::size_t columns = 0U;
        for (std::size_t index = 0U; index < group_size; ++index) {
          if (group_weights[index] == nullptr ||
              group_outputs[index] == nullptr) {
            return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                              operation, layer);
          }
          const auto* const fp8 =
              std::get_if<Fp8LinearWeight>(group_weights[index]);
          if (fp8 == nullptr ||
              fp8->prefill_supermatrix_sidecar == nullptr ||
              (index != 0U && fp8->input_size != columns)) {
            return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                              operation, layer);
          }
          columns = fp8->input_size;
          partitions[index] =
              kernels::Sm87Fp8PrefillSupermatrixPartition{
                  fp8->prefill_supermatrix_sidecar, fp8->weight_scale,
                  fp8->output_size, group_outputs[index]};
        }
        return check_cuda(
            kernels::launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
                partitions.data(), group_size, input, token_count, columns,
                stream_),
            operation, layer);
      };
  const auto project_attention_output =
      [this, token_count, &check_cuda, &project, &execution_views,
       &has_fp8_prefill_supermatrix_sidecar,
       &project_fp8_prefill_supermatrix, &is_fp8_marlin_projection](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const char* const operation,
          const std::size_t layer,
          PrefillProjectionExecution* const executed_route = nullptr) noexcept {
        if (executed_route != nullptr) {
          *executed_route = PrefillProjectionExecution::kUnknown;
        }
        if (is_fp8_marlin_projection(weight)) {
          return project(weight, input, output, operation, layer,
                         executed_route);
        }
        if (has_fp8_prefill_supermatrix_sidecar(weight)) {
          const LinearWeight* const group_weights[1U] = {&weight};
          std::uint16_t* const group_outputs[1U] = {output};
          const bool succeeded = project_fp8_prefill_supermatrix(
              group_weights, group_outputs, 1U, input, operation, layer);
          if (succeeded && executed_route != nullptr) {
            *executed_route = PrefillProjectionExecution::kFp8Supermatrix;
          }
          return succeeded;
        }
        if (!reference_runner_detail::
                use_fp8_m64_prefill_attention_output_projection(
                    projection_backend_, weight, input, output,
                    token_count)) {
          return project(weight, input, output, operation, layer,
                         executed_route);
        }
        const bool succeeded =
            check_cuda(launch_projection_tile_to_bf16_cuda(
                           projection_backend_, weight, input, token_count,
                           execution_views.fp32_scratch, execution_views.fp32_scratch_elements,
                           output, stream_),
                       operation, layer);
        if (succeeded && executed_route != nullptr) {
          *executed_route = PrefillProjectionExecution::kFp8M64Output;
        }
        return succeeded;
      };
  const auto record_projection_route =
      [&record_layer_route](const PrefillLayerRouteSlot slot,
                            const PrefillProjectionExecution execution)
      noexcept {
        return record_layer_route(slot, projection_disposition(execution));
      };
  const auto project_pair =
      [this, token_count, &check_cuda, &execution_views,
       enable_bf16_ab_large_m_prefill_admission, sealed_exact_arithmetic,
       &arithmetic_ledger](
                                const LinearWeight& first_weight,
                                const LinearWeight& second_weight,
                                const std::uint16_t* const input,
                                std::uint16_t* const first_output,
                                std::uint16_t* const second_output,
                                const char* const operation,
                                const std::size_t layer) noexcept {
    if (sealed_exact_arithmetic) {
      return check_cuda(launch_exact_contract_bf16_projection_pair(
                            projection_backend_, first_weight, second_weight,
                            input, first_output, second_output,
                            arithmetic_ledger, execution_views.fp32_scratch,
                            execution_views.fp32_scratch_elements, stream_),
                        operation, layer);
    }
    const std::size_t columns = linear_input_size(first_weight);
    const std::size_t first_rows = linear_output_size(first_weight);
    const std::size_t second_rows = linear_output_size(second_weight);
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
    if (enable_bf16_ab_large_m_prefill_admission && token_count >= 2U &&
        supports_bf16_projection_pair(
            projection_backend_, first_weight, second_weight)) {
      const auto* const first = std::get_if<Bf16LinearWeight>(&first_weight);
      const auto* const second =
          std::get_if<Bf16LinearWeight>(&second_weight);
      if (first == nullptr || second == nullptr) {
        return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                          operation, layer);
      }
      const int status = kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
          first->weight, second->weight, input, token_count,
          first_output, second_output, stream_);
      if (status == static_cast<int>(cudaSuccess)) {
        ++g_bf16_ab_large_m_prefill_admission_hits;
      }
      return check_cuda(status, operation, layer);
    }
#else
    (void)enable_bf16_ab_large_m_prefill_admission;
#endif
    for (std::size_t token_offset = 0U; token_offset < token_count;
         token_offset += kProductionProjectionSubtileTokens) {
      const std::size_t remaining = token_count - token_offset;
      const std::size_t subtile_tokens =
          remaining < kProductionProjectionSubtileTokens
              ? remaining
              : kProductionProjectionSubtileTokens;
      if (!check_cuda(launch_projection_pair_tile_to_bf16_cuda(
                          projection_backend_, first_weight, second_weight,
                          input + token_offset * columns, subtile_tokens,
                          execution_views.fp32_scratch, execution_views.fp32_scratch_elements,
                          first_output + token_offset * first_rows,
                          second_output + token_offset * second_rows, stream_),
                      operation, layer)) {
        return false;
      }
    }
    return true;
  };
  const auto project_down = [this, token_count, &check_cuda,
                             &project, &execution_views](const LinearWeight& weight,
                                       const std::uint16_t* const input,
                                       std::uint16_t* const output,
                                       const char* const operation,
                                       const std::size_t layer) noexcept {
    if (reference_runner_detail::
            use_nvfp4_whole_chunk_prefill_down_projection(
                projection_backend_, weight, input, output, token_count)) {
      const int status = launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
          projection_backend_, weight, input, token_count, output, stream_);
      if (status == static_cast<int>(cudaErrorNotSupported)) {
        return project(weight, input, output, operation, layer);
      }
      return check_cuda(status, operation, layer);
    }
    if (token_count != kMaximumProjectionTileTokenCount) {
      return project(weight, input, output, operation, layer);
    }
    return check_cuda(launch_projection_tile_to_bf16_cuda(
                          projection_backend_, weight, input, token_count,
                          execution_views.fp32_scratch,
                          execution_views.fp32_scratch_elements, output, stream_),
                      operation, layer);
  };
  const auto project_nvfp4_whole_chunk_on_stream =
      [this, token_count, &check_cuda, &project_on_stream](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const char* const operation,
          const std::size_t layer, void* const launch_stream) noexcept {
        const int status =
            launch_exact_nvfp4_whole_chunk_branch_to_bf16_cuda(
                projection_backend_, weight, input, token_count, output,
                launch_stream);
        if (status == static_cast<int>(cudaSuccess)) {
          return true;
        }
        if (status == static_cast<int>(cudaErrorNotSupported)) {
          return project_on_stream(weight, input, output, operation, layer,
                                   launch_stream);
        }
        return check_cuda(status, operation, layer);
      };
  const auto residual_rms_m32_schedule =
      reference_runner_detail::prefill_residual_rms_m32_schedule(
          token_count, kReferenceHiddenSize);
  const bool use_residual_rms_prompt_wide =
      residual_rms_m32_schedule.valid() &&
      !sealed_exact_arithmetic &&
      g_enable_prefill_residual_rms_prompt_wide_admission;
  const auto residual_norm_m32_tiles =
      [this, token_count, residual_rms_m32_schedule,
       use_residual_rms_prompt_wide, &check_cuda](
          const std::uint16_t* const left,
          const std::uint16_t* const right,
          const std::uint16_t* const norm_weight,
          std::uint16_t* const residual_output,
          std::uint16_t* const normalized_output,
          const char* const operation, const std::size_t layer) noexcept {
        if (use_residual_rms_prompt_wide) {
          return check_cuda(
              launch_residual_add_headwise_centered_rms_norm_prefill_5120_cuda(
                  left, right, norm_weight, token_count,
                  kReferenceHiddenSize, kRmsEpsilon, residual_output,
                  normalized_output, stream_),
              operation, layer);
        }
        for (std::size_t token_offset = 0U;
             token_offset < residual_rms_m32_schedule.fused_prefix_tokens;
             token_offset += kProductionProjectionSubtileTokens) {
          const std::size_t element_offset =
              token_offset * kReferenceHiddenSize;
          if (!check_cuda(
                  launch_residual_add_headwise_centered_rms_norm_m32_5120_cuda(
                      left + element_offset, right + element_offset,
                      norm_weight, kProductionProjectionSubtileTokens,
                      kReferenceHiddenSize, kRmsEpsilon,
                      residual_output + element_offset,
                      normalized_output + element_offset, stream_),
                  operation, layer)) {
            return false;
          }
        }
        if (residual_rms_m32_schedule.fallback_tail_tokens != 0U) {
          const std::size_t element_offset =
              residual_rms_m32_schedule.fused_prefix_tokens *
              kReferenceHiddenSize;
          const std::size_t tail_elements =
              residual_rms_m32_schedule.fallback_tail_tokens *
              kReferenceHiddenSize;
          if (!check_cuda(
                  launch_residual_add_reference_cuda(
                      left + element_offset, right + element_offset,
                      tail_elements, residual_output + element_offset,
                      stream_),
                  operation, layer) ||
              !check_cuda(
                  launch_headwise_centered_rms_norm_reference_cuda(
                      residual_output + element_offset, norm_weight,
                      residual_rms_m32_schedule.fallback_tail_tokens,
                      kReferenceHiddenSize, kRmsEpsilon,
                      normalized_output + element_offset, stream_),
                  operation, layer)) {
            return false;
          }
        }
        return true;
      };
  const bool use_m32_residual_rms_fusion =
      control.allow_cross_layer_m32_fusion &&
      residual_rms_m32_schedule.valid();

  if (control.gather_embedding) {
    if (!sealed_exact_arithmetic &&
        g_enable_prefill_embedding_prompt_wide_admission) {
      auto* const device_token_ids =
          reinterpret_cast<std::uint32_t*>(execution_views.projection[3]);
      if (!check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  device_token_ids, input_token_ids,
                  token_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                  static_cast<cudaStream_t>(stream_))),
              "prefill_embedding_token_ids", kReferenceNoLayer) ||
          !check_cuda(launch_embedding_gather_prompt_reference_cuda(
                          weights_->embed_tokens().weight,
                          kReferenceVocabularySize, kReferenceHiddenSize,
                          device_token_ids, token_count, execution_views.hidden[0],
                          stream_),
                      "prefill_embedding_gather_prompt_wide",
                      kReferenceNoLayer)) {
        return fail_enqueue(launch_failure);
      }
    } else {
      for (std::size_t token = 0U; token < token_count; ++token) {
        if (!check_cuda(launch_embedding_gather_reference_cuda(
                            weights_->embed_tokens().weight,
                            kReferenceVocabularySize, kReferenceHiddenSize,
                            input_token_ids[token],
                            execution_views.hidden[0] + token * kReferenceHiddenSize,
                            stream_),
                        "prefill_embedding_gather", kReferenceNoLayer)) {
          return fail_enqueue(launch_failure);
        }
      }
    }
  }

  for (std::size_t layer = control.layer_begin; layer < control.layer_end;
       ++layer) {
    layer_route_fragment = {};
    layer_route_fragment.layer = layer;
    layer_route_fragment.first_position = first_position;
    layer_route_fragment.token_count =
        static_cast<std::uint32_t>(token_count);
    const DecoderLayerWeights& layer_weights = weights_->layer(layer);
    // The M32-prefix plus reference-tail layer-(N-1) MLP boundary already
    // produced layer N's normalized input. Layer 0 still normalizes the
    // embedding explicitly.
    if ((!use_m32_residual_rms_fusion || layer == 0U) &&
        !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                        execution_views.hidden[0],
                        layer_weights.input_layernorm.data, token_count,
                        kReferenceHiddenSize, kRmsEpsilon, execution_views.hidden[1],
                        stream_),
                    "prefill_input_layernorm", layer)) {
      return fail_enqueue(launch_failure);
    }

    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_enqueue(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "prefill_linear_attention_variant", layer));
      }
      bool linear_qkvz_projected = false;
      PrefillProjectionExecution linear_qkv_route =
          PrefillProjectionExecution::kUnknown;
      PrefillProjectionExecution linear_z_route =
          PrefillProjectionExecution::kUnknown;
      if (has_fp8_prefill_supermatrix_sidecar(attention->in_proj_qkv) &&
          has_fp8_prefill_supermatrix_sidecar(attention->in_proj_z)) {
        const LinearWeight* const group_weights[2U] = {
            &attention->in_proj_qkv, &attention->in_proj_z};
        std::uint16_t* const group_outputs[2U] = {
            execution_views.projection[0], execution_views.projection[1]};
        if (!project_fp8_prefill_supermatrix(
                group_weights, group_outputs, 2U, execution_views.hidden[1],
                "prefill_linear_qkvz_supermatrix_projection", layer)) {
          return fail_enqueue(launch_failure);
        }
        linear_qkvz_projected = true;
        linear_qkv_route = PrefillProjectionExecution::kFp8Supermatrix;
        linear_z_route = PrefillProjectionExecution::kFp8Supermatrix;
      }
      if (!linear_qkvz_projected &&
          (!project(attention->in_proj_qkv, execution_views.hidden[1],
                    execution_views.projection[0], "prefill_linear_qkv_projection",
                    layer, &linear_qkv_route) ||
           !project(attention->in_proj_z, execution_views.hidden[1],
                    execution_views.projection[1], "prefill_linear_z_projection",
                    layer, &linear_z_route))) {
        return fail_enqueue(launch_failure);
      }
      if (!record_projection_route(PrefillLayerRouteSlot::kQOrLinearQkv,
                                   linear_qkv_route) ||
          !record_projection_route(PrefillLayerRouteSlot::kLinearZ,
                                   linear_z_route)) {
        return fail_enqueue(runner_status(
            ReferenceRunnerError::kRouteEvidenceFailure,
            "prefill_linear_input_route", layer));
      }
      if (supports_bf16_projection_pair(
              projection_backend_, attention->in_proj_a,
              attention->in_proj_b)) {
        if (!project_pair(attention->in_proj_a, attention->in_proj_b,
                          execution_views.hidden[1], execution_views.linear_a, execution_views.linear_b,
                          "prefill_linear_a_b_projection", layer)) {
          return fail_enqueue(launch_failure);
        }
      } else if (!project(attention->in_proj_a, execution_views.hidden[1],
                          execution_views.linear_a, "prefill_linear_a_projection",
                          layer) ||
                 !project(attention->in_proj_b, execution_views.hidden[1],
                          execution_views.linear_b, "prefill_linear_b_projection",
                          layer)) {
        return fail_enqueue(launch_failure);
      }
      const bool use_gdn_c16_norm_gate =
          should_use_prefill_gdn_c16_norm_gate(
              enable_gdn_c16_norm_gate_admission, projection_backend_,
              first_position, token_count);
      bool gdn_output_is_normalized = false;
      PrefillGdnExecution gdn_execution = PrefillGdnExecution::kUnknown;
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
      const bool use_gdn_chunk64_native =
          use_prefill_gdn_chunk64_native_admission(
              enable_gdn_chunk64_native_admission, projection_backend_,
              first_position, token_count,
              prefill_gdn_chunk64_native_workspace_,
              prefill_gdn_chunk64_native_workspace_bytes_);
      if (use_gdn_chunk64_native) {
        const bool use_token_parallel_conv =
            !sealed_exact_arithmetic &&
            g_enable_gdn_conv_token_parallel_admission;
        // The fused producer writes compact Q/K for valid tokens and exact
        // zeros for the padded tail slots in the final C64 chunk.
        const bool use_conv_compact_qk_fused_candidate =
            use_token_parallel_conv &&
            g_enable_gdn_conv_compact_qk_fused_candidate;
        std::uint16_t* const conv_qkv =
            use_token_parallel_conv ? execution_views.projection[3]
                                    : execution_views.projection[0];
        // Convolution is exact for arbitrary C32..C512 admitted tiles and owns
        // a separate recurrent state. Execute it once over the complete
        // runner tile; the native hierarchy pads only its final C64 chunk.
        int conv_status = static_cast<int>(cudaErrorInvalidValue);
        if (use_conv_compact_qk_fused_candidate) {
          conv_status = gdn_prefill_chunk64_native_detail::
              launch_fused_conv_compact_qk_preprocess(
                  prefill_gdn_chunk64_native_workspace_,
                  prefill_gdn_chunk64_native_workspace_bytes_,
                  execution_views.projection[0], token_count,
                  attention->conv1d.data, execution_views.conv_state[layer],
                  conv_qkv, kRmsEpsilon, stream_);
        } else if (use_token_parallel_conv) {
          conv_status = gdn_prefill_whole_span_conv_detail::
              launch_causal_conv1d_silu_update_token_parallel_exact_cuda(
                  execution_views.projection[0], token_count,
                  attention->conv1d.data, execution_views.conv_state[layer],
                  conv_qkv, stream_);
        } else {
          conv_status = gdn_prefill_whole_span_conv_detail::
              launch_causal_conv1d_silu_update_whole_span_exact_cuda(
                  execution_views.projection[0], token_count,
                  attention->conv1d.data, execution_views.conv_state[layer],
                  conv_qkv, stream_);
        }
        if (!check_cuda(conv_status,
                        "prefill_linear_causal_conv_whole_span", layer)) {
          return fail_enqueue(launch_failure);
        }
        if (use_conv_compact_qk_fused_candidate) {
          ++g_gdn_conv_compact_qk_fused_candidate_hits;
        }
        ++g_prefill_gdn_chunk64_native_admission_hits;
        const int native_status =
            use_conv_compact_qk_fused_candidate
                ? gdn_prefill_chunk64_native_detail::launch_qk_preprocessed(
                    prefill_gdn_chunk64_native_workspace_,
                    prefill_gdn_chunk64_native_workspace_bytes_,
                    conv_qkv, token_count,
                    execution_views.linear_a, execution_views.linear_b, attention->a_log.data,
                    attention->dt_bias.data, execution_views.gdn_state[layer],
                    execution_views.gdn_state[layer], kRmsEpsilon,
                    attention->norm.data, execution_views.projection[1],
                    kRmsEpsilon, execution_views.projection[2], stream_)
                : (sealed_exact_arithmetic
                       ? gdn_prefill_chunk64_native_detail::
                             launch_fixed_production(
                                 prefill_gdn_chunk64_native_workspace_,
                                 prefill_gdn_chunk64_native_workspace_bytes_,
                                 conv_qkv, token_count,
                                 execution_views.linear_a,
                                 execution_views.linear_b,
                                 attention->a_log.data,
                                 attention->dt_bias.data,
                                 execution_views.gdn_state[layer],
                                 execution_views.gdn_state[layer],
                                 kRmsEpsilon, attention->norm.data,
                                 execution_views.projection[1], kRmsEpsilon,
                                 execution_views.projection[2], stream_)
                       : gdn_prefill_chunk64_native_detail::launch(
                             prefill_gdn_chunk64_native_workspace_,
                             prefill_gdn_chunk64_native_workspace_bytes_,
                             conv_qkv, token_count,
                             execution_views.linear_a,
                             execution_views.linear_b,
                             attention->a_log.data,
                             attention->dt_bias.data,
                             execution_views.gdn_state[layer],
                             execution_views.gdn_state[layer], kRmsEpsilon,
                             attention->norm.data,
                             execution_views.projection[1], kRmsEpsilon,
                             execution_views.projection[2], stream_));
        if (!check_cuda(native_status, "prefill_linear_gdn_chunk64_native",
                        layer)) {
          return fail_enqueue(launch_failure);
        }
        gdn_execution = PrefillGdnExecution::kChunk64Native;
        gdn_output_is_normalized = true;
      } else
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
      const bool use_gdn_chunk64_reference =
          use_prefill_gdn_chunk64_reference_admission(
              enable_gdn_chunk64_reference_admission, projection_backend_,
              first_position, token_count,
              prefill_gdn_chunk64_reference_context_,
              prefill_gdn_chunk64_reference_workspace_,
              prefill_gdn_chunk64_reference_workspace_bytes_);
      if (use_gdn_chunk64_reference) {
        // Keep the established causal-convolution transition, then execute
        // the complete C64x8 WY hierarchy as one test-only architecture
        // reference. No external-library failure may fall back after state
        // mutation; the public call is poisoned through the normal failure
        // path instead.
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kPrefillKernelTileMaximumTokens) {
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, execution_views.conv_state[layer],
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer)) {
            return fail_enqueue(launch_failure);
          }
        }
        ++g_prefill_gdn_chunk64_reference_admission_hits;
        if (!check_cuda(
                gdn_prefill_chunk64_reference_detail::launch(
                    prefill_gdn_chunk64_reference_context_,
                    prefill_gdn_chunk64_reference_workspace_,
                    prefill_gdn_chunk64_reference_workspace_bytes_,
                    token_count, execution_views.projection[0], execution_views.linear_a,
                    execution_views.linear_b,
                    attention->a_log.data, attention->dt_bias.data,
                    execution_views.gdn_state[layer], execution_views.gdn_state[layer],
                    kRmsEpsilon, attention->norm.data,
                    execution_views.projection[1], kRmsEpsilon,
                    execution_views.projection[2], stream_),
                "prefill_linear_gdn_chunk64_reference", layer)) {
          return fail_enqueue(launch_failure);
        }
        gdn_execution = PrefillGdnExecution::kExternalReference;
        gdn_output_is_normalized = true;
      } else
#endif
#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
      const bool use_gdn_b8 = use_prefill_gdn_b8_admission(
          enable_gdn_b8_admission, projection_backend_, first_position,
          token_count);
      if (use_gdn_b8) {
        // The B8 recurrence consumes the complete convolved QKV chunk. Keep
        // the causal-convolution state transition ordered in existing M16
        // subtiles, then replace only the GDN chain with one exact C256/C512
        // admission kernel. No workspace, synchronization, or Decode path is
        // added.
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kPrefillKernelTileMaximumTokens) {
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, execution_views.conv_state[layer],
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer)) {
            return fail_enqueue(launch_failure);
          }
        }
        ++g_prefill_gdn_b8_admission_hits;
        if (!check_cuda(
                gdn_prefill_b8_detail::
                    launch_gated_delta_net_update_sequential_fp32_b8_exact_cuda(
                        execution_views.projection[0], token_count, execution_views.linear_a,
                        execution_views.linear_b, attention->a_log.data,
                        attention->dt_bias.data, execution_views.gdn_state[layer],
                        execution_views.gdn_state[layer], kRmsEpsilon,
                        execution_views.projection[2], stream_),
                "prefill_linear_gdn_b8_admission", layer)) {
          return fail_enqueue(launch_failure);
        }
        gdn_execution = PrefillGdnExecution::kApproximateB8;
      } else
#endif
      if (use_gdn_c16_norm_gate) {
        // Preserve the established causal-convolution state order, then
        // consume every exact C16 slice in the C256/C512 composite
        // GDN/norm/gate route. Any launch error terminates the tile; the
        // production route never falls back after partially updating state.
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kPrefillKernelTileMaximumTokens) {
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, execution_views.conv_state[layer],
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer) ||
              !check_cuda(
                  gdn_prefill_c16_norm_gate_detail::
                      launch_shared_boundary(
                          execution_views.projection[0] +
                              token_offset * kLinearQkvElements,
                          kPrefillKernelTileMaximumTokens,
                          execution_views.linear_a +
                              token_offset * kLinearScalarElements,
                          execution_views.linear_b +
                              token_offset * kLinearScalarElements,
                          attention->a_log.data, attention->dt_bias.data,
                          execution_views.gdn_state[layer], execution_views.gdn_state[layer],
                          kRmsEpsilon, attention->norm.data,
                          execution_views.projection[1] +
                              token_offset * kLinearValueElements,
                          kRmsEpsilon,
                          execution_views.projection[2] +
                              token_offset * kLinearValueElements,
                          stream_),
                  "prefill_linear_gdn_c16_norm_gate", layer)) {
            return fail_enqueue(launch_failure);
          }
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
          ++g_prefill_gdn_c16_norm_gate_admission_hits;
#endif
        }
        gdn_execution = PrefillGdnExecution::kC16Exact;
        gdn_output_is_normalized = true;
      } else {
        for (std::size_t token_offset = 0U; token_offset < token_count;
             token_offset += kPrefillKernelTileMaximumTokens) {
          const std::size_t remaining = token_count - token_offset;
          const std::size_t subtile_tokens =
              remaining < kPrefillKernelTileMaximumTokens
                  ? remaining
                  : kPrefillKernelTileMaximumTokens;
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      subtile_tokens, attention->conv1d.data,
                      execution_views.conv_state[layer],
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer) ||
              !check_cuda(
                  launch_gated_delta_net_update_tile_warp_parallel_cuda(
                      execution_views.projection[0] +
                          token_offset * kLinearQkvElements,
                      subtile_tokens,
                      execution_views.linear_a +
                          token_offset * kLinearScalarElements,
                      execution_views.linear_b +
                          token_offset * kLinearScalarElements,
                      attention->a_log.data, attention->dt_bias.data,
                      execution_views.gdn_state[layer], execution_views.gdn_state[layer],
                      kRmsEpsilon,
                      execution_views.projection[2] +
                          token_offset * kLinearValueElements,
                      {}, stream_),
                  "prefill_linear_gdn", layer)) {
            return fail_enqueue(launch_failure);
          }
        }
        gdn_execution = PrefillGdnExecution::kWarpExact;
      }
      if (!gdn_output_is_normalized &&
          !check_cuda(
              launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                  execution_views.projection[2], attention->norm.data,
                  execution_views.projection[1], token_count * kGdnValueHeadCount,
                  kGdnHeadDimension, kRmsEpsilon, execution_views.projection[2],
                  stream_),
              "prefill_linear_output_norm_gate", layer)) {
        return fail_enqueue(launch_failure);
      }
      PrefillProjectionExecution linear_o_route =
          PrefillProjectionExecution::kUnknown;
      if (!project_attention_output(
              attention->out_proj, execution_views.projection[2], execution_views.hidden[1],
              "prefill_linear_output_projection", layer, &linear_o_route)) {
        return fail_enqueue(launch_failure);
      }
      if (!record_projection_route(PrefillLayerRouteSlot::kO,
                                   linear_o_route)) {
        return fail_enqueue(runner_status(
            ReferenceRunnerError::kRouteEvidenceFailure,
            "prefill_linear_output_route", layer));
      }
      PrefillRouteDisposition gdn_disposition =
          PrefillRouteDisposition::kForbidden;
      if (gdn_execution == PrefillGdnExecution::kChunk64Native) {
        gdn_disposition = PrefillRouteDisposition::kProduction;
      } else if (gdn_execution == PrefillGdnExecution::kC16Exact ||
                 gdn_execution == PrefillGdnExecution::kWarpExact) {
        gdn_disposition = PrefillRouteDisposition::kExactFallback;
      } else if (gdn_execution ==
                 PrefillGdnExecution::kExternalReference) {
        (void)record_layer_forbidden_boundary(
            PrefillForbiddenBoundary::kExternalReference);
      } else if (gdn_execution == PrefillGdnExecution::kApproximateB8) {
        (void)record_layer_forbidden_boundary(
            PrefillForbiddenBoundary::kApproximateNumerics);
      }
      if (!record_layer_route(PrefillLayerRouteSlot::kGdn,
                              gdn_disposition)) {
        return fail_enqueue(runner_status(
            ReferenceRunnerError::kRouteEvidenceFailure,
            "prefill_linear_gdn_route", layer));
      }
    } else if (expected == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_enqueue(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "prefill_full_attention_variant", layer));
      }
      std::uint16_t* const packed_gates =
          execution_views.projection[3] + token_count * kFullQueryElements;
      std::uint16_t* const tile_key =
          execution_views.key_cache[layer] +
          static_cast<std::size_t>(first_position) * kFullKvElements;
      std::uint16_t* const tile_value =
          execution_views.value_cache[layer] +
          static_cast<std::size_t>(first_position) * kFullKvElements;
      const std::size_t rope_first_position =
          static_cast<std::size_t>(first_position);
      bool full_qkv_projected = false;
      PrefillProjectionExecution full_q_route =
          PrefillProjectionExecution::kUnknown;
      PrefillProjectionExecution full_k_route =
          PrefillProjectionExecution::kUnknown;
      PrefillProjectionExecution full_v_route =
          PrefillProjectionExecution::kUnknown;
      if (has_fp8_prefill_supermatrix_sidecar(attention->q_proj) &&
          has_fp8_prefill_supermatrix_sidecar(attention->k_proj) &&
          has_fp8_prefill_supermatrix_sidecar(attention->v_proj)) {
        const LinearWeight* const group_weights[3U] = {
            &attention->q_proj, &attention->k_proj, &attention->v_proj};
        std::uint16_t* const group_outputs[3U] = {
            execution_views.projection[0], tile_key, tile_value};
        if (!project_fp8_prefill_supermatrix(
                group_weights, group_outputs, 3U, execution_views.hidden[1],
                "prefill_full_qkv_supermatrix_projection", layer)) {
          return fail_enqueue(launch_failure);
        }
        full_qkv_projected = true;
        full_q_route = PrefillProjectionExecution::kFp8Supermatrix;
        full_k_route = PrefillProjectionExecution::kFp8Supermatrix;
        full_v_route = PrefillProjectionExecution::kFp8Supermatrix;
      }
      if (!full_qkv_projected &&
          (!project(attention->q_proj, execution_views.hidden[1],
                    execution_views.projection[0], "prefill_full_q_gate_projection",
                    layer, &full_q_route) ||
           !project(attention->k_proj, execution_views.hidden[1], tile_key,
                    "prefill_full_k_projection", layer, &full_k_route) ||
           !project(attention->v_proj, execution_views.hidden[1], tile_value,
                    "prefill_full_v_projection", layer, &full_v_route))) {
        return fail_enqueue(launch_failure);
      }
      if (!record_projection_route(PrefillLayerRouteSlot::kQOrLinearQkv,
                                   full_q_route) ||
          !record_projection_route(PrefillLayerRouteSlot::kFullK,
                                   full_k_route) ||
          !record_projection_route(PrefillLayerRouteSlot::kFullV,
                                   full_v_route)) {
        return fail_enqueue(runner_status(
            ReferenceRunnerError::kRouteEvidenceFailure,
            "prefill_full_qkv_route", layer));
      }

      const std::size_t preprocess_tile_maximum =
          !sealed_exact_arithmetic &&
                  g_enable_full_attention_preprocess_prompt_wide_admission
              ? kFullAttentionPreprocessTileMaximumTokens
              : kPrefillKernelTileMaximumTokens;
      bool use_fused_preprocess = true;
      for (std::size_t token_offset = 0U; token_offset < token_count;
           token_offset += preprocess_tile_maximum) {
        const std::size_t remaining = token_count - token_offset;
        const std::size_t subtile_tokens =
            remaining < preprocess_tile_maximum
                ? remaining
                : preprocess_tile_maximum;
        const bool valid_tile =
            sealed_exact_arithmetic
                ? reference_runner_detail::use_full_attention_preprocess_tile(
                      rope_first_position + token_offset, subtile_tokens)
                : g_enable_full_attention_preprocess_prompt_wide_admission
                ? reference_runner_detail::use_full_attention_preprocess_tile(
                      rope_first_position + token_offset, subtile_tokens)
                : reference_runner_detail::use_qk_rope_tile(
                      rope_first_position + token_offset, subtile_tokens);
        if (!valid_tile) {
          use_fused_preprocess = false;
          break;
        }
      }
      std::uint16_t* const tile_query =
          use_fused_preprocess ? execution_views.projection[3]
                               : execution_views.projection[0];
      for (std::size_t token_offset = 0U; token_offset < token_count;
           token_offset += preprocess_tile_maximum) {
        const std::size_t remaining = token_count - token_offset;
        const std::size_t subtile_tokens =
            remaining < preprocess_tile_maximum
                ? remaining
                : preprocess_tile_maximum;
        std::uint16_t* const raw_query_gate =
            execution_views.projection[0] + token_offset * kFullQGateElements;
        std::uint16_t* const subtile_query =
            tile_query + token_offset * kFullQueryElements;
        std::uint16_t* const split_query =
            execution_views.projection[3] + token_offset * kFullQueryElements;
        std::uint16_t* const subtile_gates =
            packed_gates + token_offset * kFullQueryElements;
        std::uint16_t* const subtile_key =
            tile_key + token_offset * kFullKvElements;
        const std::size_t subtile_first_position =
            rope_first_position + token_offset;
        if (use_fused_preprocess) {
          const int preprocess_status =
              sealed_exact_arithmetic
                  ? launch_full_attention_preprocess_24_4_256_64_reference_256_cuda(
                        raw_query_gate, subtile_key, attention->q_norm.data,
                        attention->k_norm.data, kRmsEpsilon, subtile_query,
                        subtile_gates, execution_views.rope_cos,
                        execution_views.rope_sin, subtile_first_position,
                        subtile_tokens, stream_)
                  : launch_full_attention_preprocess_24_4_256_64_cuda(
                        raw_query_gate, subtile_key, attention->q_norm.data,
                        attention->k_norm.data, kRmsEpsilon, subtile_query,
                        subtile_gates, execution_views.rope_cos,
                        execution_views.rope_sin, subtile_first_position,
                        subtile_tokens, stream_);
          if (!check_cuda(preprocess_status, "prefill_full_preprocess",
                          layer)) {
            return fail_enqueue(launch_failure);
          }
          continue;
        }
        if (!check_cuda(launch_split_interleaved_q_gate_reference_cuda(
                            raw_query_gate,
                            subtile_tokens * kFullQueryHeads,
                            kFullHeadDimension, split_query, subtile_gates,
                            stream_),
                        "prefill_full_split_q_gate_fallback", layer) ||
            !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                            split_query, attention->q_norm.data,
                            subtile_tokens * kFullQueryHeads,
                            kFullHeadDimension, kRmsEpsilon, subtile_query,
                            stream_),
                        "prefill_full_q_norm_fallback", layer) ||
            !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                            subtile_key, attention->k_norm.data,
                            subtile_tokens * kFullKvHeads,
                            kFullHeadDimension, kRmsEpsilon, subtile_key,
                            stream_),
                        "prefill_full_k_norm_fallback", layer)) {
          return fail_enqueue(launch_failure);
        }
        for (std::size_t token = 0U; token < subtile_tokens; ++token) {
          const std::size_t position = subtile_first_position + token;
          const float* const cosines =
              execution_views.rope_cos + position * kRopePairs;
          const float* const sines =
              execution_views.rope_sin + position * kRopePairs;
          if (!check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                              subtile_query + token * kFullQueryElements,
                              cosines, sines, kFullQueryHeads,
                              subtile_query + token * kFullQueryElements,
                              stream_),
                          "prefill_full_q_rope_fallback", layer) ||
              !check_cuda(launch_partial_neox_rope_256_64_reference_cuda(
                              subtile_key + token * kFullKvElements,
                              cosines, sines, kFullKvHeads,
                              subtile_key + token * kFullKvElements,
                              stream_),
                          "prefill_full_k_rope_fallback", layer)) {
            return fail_enqueue(launch_failure);
          }
        }
      }
      const bool use_bulk_gqa_gate =
          reference_runner_detail::
              use_bulk_causal_gqa_sigmoid_gate_prefill(
                  projection_backend_, expected, first_position,
                  token_count);
      if (use_bulk_gqa_gate) {
        const int bulk_gqa_status =
            sealed_exact_arithmetic
                ? launch_bulk_causal_gqa_sigmoid_gate_24_4_256_fixed_cuda(
                      tile_query, execution_views.key_cache[layer],
                      execution_views.value_cache[layer], packed_gates,
                      first_position, token_count,
                      execution_views.projection[1], stream_)
                : launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
                      tile_query, execution_views.key_cache[layer],
                      execution_views.value_cache[layer], packed_gates,
                      first_position, token_count,
                      execution_views.projection[1], stream_);
        if (!check_cuda(
                bulk_gqa_status,
                "prefill_full_bulk_gqa_output_gate", layer)) {
          return fail_enqueue(launch_failure);
        }
      } else {
        const std::size_t fused_gqa_gate_prefix_tokens =
            reference_runner_detail::
                fused_gqa_sigmoid_gate_prefix_token_count(first_position,
                                                          token_count);
        for (std::size_t token = 0U; token < token_count; ++token) {
          const std::size_t sequence_length =
              static_cast<std::size_t>(first_position) + token + 1U;
          const std::uint16_t* const token_query =
              tile_query + token * kFullQueryElements;
          std::uint16_t* const token_output =
              execution_views.projection[1] + token * kFullQueryElements;
          const bool fuse_gqa_gate_token =
              token < fused_gqa_gate_prefix_tokens;
          const int gqa_status =
              fuse_gqa_gate_token
                  ? launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                        token_query, execution_views.key_cache[layer],
                        execution_views.value_cache[layer], sequence_length,
                        kAttentionScale, execution_views.fp32_scratch,
                        execution_views.fp32_scratch_elements,
                        packed_gates + token * kFullQueryElements,
                        token_output, stream_)
                  : launch_gqa_attention_reference_cuda(
                        token_query, execution_views.key_cache[layer],
                        execution_views.value_cache[layer], kFullQueryHeads,
                        kFullKvHeads, sequence_length, kFullHeadDimension,
                        kAttentionScale, execution_views.fp32_scratch,
                        execution_views.fp32_scratch_elements, token_output, stream_);
          if (!check_cuda(
                  gqa_status,
                  fuse_gqa_gate_token ? "prefill_full_gqa_output_gate"
                                      : "prefill_full_gqa",
                  layer)) {
            return fail_enqueue(launch_failure);
          }
        }
        if (fused_gqa_gate_prefix_tokens < token_count &&
            !check_cuda(launch_sigmoid_gate_reference_cuda(
                            execution_views.projection[1] +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            packed_gates +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            (token_count - fused_gqa_gate_prefix_tokens) *
                                kFullQueryElements,
                            execution_views.projection[1] +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            stream_),
                        "prefill_full_output_gate", layer)) {
          return fail_enqueue(launch_failure);
        }
      }
      PrefillProjectionExecution full_o_route =
          PrefillProjectionExecution::kUnknown;
      if (!project_attention_output(
              attention->o_proj, execution_views.projection[1], execution_views.hidden[1],
              "prefill_full_output_projection", layer, &full_o_route)) {
        return fail_enqueue(launch_failure);
      }
      if (!record_projection_route(PrefillLayerRouteSlot::kO,
                                   full_o_route) ||
          !record_layer_route(
              PrefillLayerRouteSlot::kAttention,
              use_bulk_gqa_gate ? PrefillRouteDisposition::kProduction
                                : PrefillRouteDisposition::kExactFallback)) {
        return fail_enqueue(runner_status(
            ReferenceRunnerError::kRouteEvidenceFailure,
            "prefill_full_attention_route", layer));
      }
    } else {
      return fail_enqueue(runner_status(
          ReferenceRunnerError::kInvalidLayerSchedule,
          "prefill_layer_schedule", layer));
    }

    if (use_m32_residual_rms_fusion) {
      if (!residual_norm_m32_tiles(
              execution_views.hidden[0], execution_views.hidden[1],
              layer_weights.post_attention_layernorm.data, execution_views.hidden[2],
              execution_views.hidden[1],
              "prefill_attention_residual_post_attention_layernorm", layer)) {
        return fail_enqueue(launch_failure);
      }
    } else {
      if (!check_cuda(launch_residual_add_reference_cuda(
                          execution_views.hidden[0], execution_views.hidden[1],
                          token_count * kReferenceHiddenSize,
                          execution_views.hidden[2], stream_),
                      "prefill_attention_residual", layer)) {
        return fail_enqueue(launch_failure);
      }
      if (!check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                          execution_views.hidden[2],
                          layer_weights.post_attention_layernorm.data,
                          token_count, kReferenceHiddenSize, kRmsEpsilon,
                          execution_views.hidden[1], stream_),
                      "prefill_post_attention_layernorm", layer)) {
        return fail_enqueue(launch_failure);
      }
    }
    bool marlin_mlp_completed = false;
#if defined(Q3X_ENABLE_NVFP4_MARLIN_PREFILL_ADMISSION)
    const auto* const marlin_gate =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.gate_proj);
    const auto* const marlin_up =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.up_proj);
    const auto* const marlin_down =
        std::get_if<NvFp4LinearWeight>(&layer_weights.mlp.down_proj);
    const std::size_t marlin_branch_elements =
        token_count * kReferenceIntermediateSize;
    const std::size_t marlin_workspace_branch_elements =
        static_cast<std::size_t>(state_->plan().prefill_chunk_size) *
        kReferenceIntermediateSize;
    const bool use_marlin_mlp =
        (control.force_bound_nvfp4_marlin_prefill ||
         g_enable_nvfp4_marlin_prefill_admission) &&
        kernels::sm87_nvfp4_marlin_supports_token_count(token_count) &&
        projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
        marlin_gate != nullptr && marlin_up != nullptr &&
        marlin_down != nullptr &&
        marlin_gate->prefill_marlin_weight != nullptr &&
        marlin_gate->prefill_marlin_scales != nullptr &&
        marlin_gate->prefill_marlin_global_scale != nullptr &&
        marlin_gate->prefill_marlin_weight ==
            marlin_up->prefill_marlin_weight &&
        marlin_gate->prefill_marlin_scales ==
            marlin_up->prefill_marlin_scales &&
        marlin_gate->prefill_marlin_global_scale ==
            marlin_up->prefill_marlin_global_scale &&
        marlin_down->prefill_marlin_weight != nullptr &&
        marlin_down->prefill_marlin_scales != nullptr &&
        marlin_down->prefill_marlin_global_scale != nullptr &&
        execution_views.projection[1] ==
            execution_views.projection[0] + marlin_workspace_branch_elements &&
        marlin_branch_elements <= marlin_workspace_branch_elements &&
        execution_views.fp32_scratch_elements >=
            kernels::kSm87NvFp4MarlinReductionElements;
    if (use_marlin_mlp) {
      auto* const locks =
          reinterpret_cast<std::int32_t*>(execution_views.projection[3]);
      const auto stream = reinterpret_cast<cudaStream_t>(stream_);
      if (sealed_exact_arithmetic) {
        ExactPrefillProjectionWorkspace workspace;
        workspace.reduction = execution_views.fp32_scratch;
        workspace.reduction_elements =
            execution_views.fp32_scratch_elements;
        workspace.locks = locks;
        workspace.lock_bytes = kernels::kSm87NvFp4MarlinLockBytes;
        if (!check_cuda(launch_exact_contract_nvfp4_mlp(
                            *marlin_gate, *marlin_up, *marlin_down,
                            execution_views.hidden[1],
                            execution_views.projection[0],
                            execution_views.projection[2],
                            execution_views.hidden[1], arithmetic_ledger,
                            workspace, workspace, stream_),
                        "prefill_marlin_exact_contract_mlp", layer)) {
          return fail_enqueue(launch_failure);
        }
      } else {
        const bool use_fused_gate_up_epilogue =
            g_enable_prefill_marlin_gate_up_epilogue_admission;
        if (!check_cuda(
                static_cast<int>(cudaMemsetAsync(
                    locks, 0, kernels::kSm87NvFp4MarlinLockBytes, stream)),
                "prefill_marlin_clear_locks", layer) ||
            !(use_fused_gate_up_epilogue
                  ? check_cuda(
                        kernels::
                            launch_sm87_nvfp4_marlin_gate_up_epilogue_cuda(
                                execution_views.hidden[1],
                                marlin_gate->prefill_marlin_weight,
                                marlin_gate->prefill_marlin_scales,
                                marlin_gate->prefill_marlin_global_scale,
                                token_count, execution_views.projection[0],
                                execution_views.projection[2],
                                execution_views.fp32_scratch, locks, stream_),
                        "prefill_marlin_gate_up_epilogue", layer)
                  : (check_cuda(
                         kernels::launch_sm87_nvfp4_marlin_gate_up_cuda(
                             execution_views.hidden[1],
                             marlin_gate->prefill_marlin_weight,
                             marlin_gate->prefill_marlin_scales,
                             marlin_gate->prefill_marlin_global_scale,
                             token_count, execution_views.projection[0],
                             execution_views.fp32_scratch, locks, stream_),
                         "prefill_marlin_gate_up", layer) &&
                     check_cuda(
                         kernels::launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
                             execution_views.projection[0], token_count,
                             execution_views.projection[2], stream_),
                         "prefill_marlin_gate_up_silu", layer))) ||
            !check_cuda(
                kernels::launch_sm87_nvfp4_marlin_down_cuda(
                    execution_views.projection[2],
                    marlin_down->prefill_marlin_weight,
                    marlin_down->prefill_marlin_scales,
                    marlin_down->prefill_marlin_global_scale, token_count,
                    execution_views.hidden[1], execution_views.fp32_scratch,
                    locks, stream_),
                "prefill_marlin_down", layer)) {
          return fail_enqueue(launch_failure);
        }
      }
      ++g_nvfp4_marlin_prefill_admission_hits;
      marlin_mlp_completed = true;
    }
#endif

    if (!marlin_mlp_completed) {
      // Production Prefill is self-hosted only. External-library comparison
      // modules are compiled exclusively into benchmark targets.
      {
      const bool gate_up_fork_join_available =
          prefill_auxiliary_stream_ != nullptr &&
          prefill_branch_ready_event_ != nullptr &&
          prefill_branch_done_event_ != nullptr;
      const bool use_gate_up_whole_chunk_dual_stream =
          gate_up_fork_join_available &&
          reference_runner_detail::
              use_nvfp4_whole_chunk_prefill_gate_up_dual_stream(
                  projection_backend_, layer_weights.mlp.gate_proj,
                  layer_weights.mlp.up_proj, execution_views.hidden[1],
                  execution_views.projection[0], execution_views.projection[1], token_count);
      const bool use_gate_up_m32_dual_stream =
          gate_up_fork_join_available &&
          reference_runner_detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              projection_backend_, layer_weights.mlp.gate_proj,
              layer_weights.mlp.up_proj, execution_views.hidden[1],
              execution_views.projection[0], execution_views.projection[1], token_count);
      const bool use_gate_up_dual_stream =
          use_gate_up_whole_chunk_dual_stream ||
          use_gate_up_m32_dual_stream;
      if (use_gate_up_dual_stream) {
        const auto auxiliary_stream =
            reinterpret_cast<cudaStream_t>(prefill_auxiliary_stream_);
        const auto branch_ready =
            reinterpret_cast<cudaEvent_t>(prefill_branch_ready_event_);
        const auto branch_done =
            reinterpret_cast<cudaEvent_t>(prefill_branch_done_event_);
        const auto project_gate = [&]() noexcept {
          if (use_gate_up_whole_chunk_dual_stream) {
            return project_nvfp4_whole_chunk_on_stream(
                layer_weights.mlp.gate_proj, execution_views.hidden[1],
                execution_views.projection[0], "prefill_mlp_gate_projection", layer,
                stream_);
          }
          return project(layer_weights.mlp.gate_proj, execution_views.hidden[1],
                         execution_views.projection[0], "prefill_mlp_gate_projection",
                         layer);
        };
        const auto project_up = [&]() noexcept {
          if (use_gate_up_whole_chunk_dual_stream) {
            return project_nvfp4_whole_chunk_on_stream(
                layer_weights.mlp.up_proj, execution_views.hidden[1],
                execution_views.projection[1], "prefill_mlp_up_projection_auxiliary",
                layer, prefill_auxiliary_stream_);
          }
          return project_on_stream(
              layer_weights.mlp.up_proj, execution_views.hidden[1],
              execution_views.projection[1], "prefill_mlp_up_projection_auxiliary",
              layer, prefill_auxiliary_stream_);
        };
        if (!check_cuda(
                static_cast<int>(cudaEventRecord(branch_ready, stream)),
                "prefill_mlp_branch_ready_record", layer) ||
            !check_cuda(static_cast<int>(cudaStreamWaitEvent(
                            auxiliary_stream, branch_ready, 0U)),
                        "prefill_mlp_branch_ready_wait", layer) ||
            !project_gate() || !project_up() ||
            !check_cuda(
                static_cast<int>(cudaEventRecord(branch_done,
                                                 auxiliary_stream)),
                "prefill_mlp_branch_done_record", layer) ||
            !check_cuda(static_cast<int>(
                            cudaStreamWaitEvent(stream, branch_done, 0U)),
                        "prefill_mlp_branch_done_wait", layer)) {
          return fail_enqueue(launch_failure);
        }
      } else if (!project(layer_weights.mlp.gate_proj, execution_views.hidden[1],
                          execution_views.projection[0],
                          "prefill_mlp_gate_projection", layer) ||
                 !project(layer_weights.mlp.up_proj, execution_views.hidden[1],
                          execution_views.projection[1],
                          "prefill_mlp_up_projection", layer)) {
        return fail_enqueue(launch_failure);
      }
      }
      if (!check_cuda(launch_silu_mul_reference_cuda(
                        execution_views.projection[0], execution_views.projection[1],
                        token_count * kReferenceIntermediateSize,
                        execution_views.projection[0], stream_),
                    "prefill_mlp_silu_mul", layer)) {
        return fail_enqueue(launch_failure);
      }
      if (!project_down(layer_weights.mlp.down_proj, execution_views.projection[0],
                        execution_views.hidden[1], "prefill_mlp_down_projection",
                        layer)) {
        return fail_enqueue(launch_failure);
      }
    }
    const bool nvfp4_mlp_weights =
        std::holds_alternative<NvFp4LinearWeight>(
            layer_weights.mlp.gate_proj) &&
        std::holds_alternative<NvFp4LinearWeight>(
            layer_weights.mlp.up_proj) &&
        std::holds_alternative<NvFp4LinearWeight>(
            layer_weights.mlp.down_proj);
    const PrefillRouteDisposition mlp_disposition =
        !nvfp4_mlp_weights
            ? PrefillRouteDisposition::kForbidden
            : (marlin_mlp_completed
                   ? PrefillRouteDisposition::kProduction
                   : PrefillRouteDisposition::kExactFallback);
    if (!record_layer_route(PrefillLayerRouteSlot::kNvFp4GateUp,
                            mlp_disposition) ||
        !record_layer_route(PrefillLayerRouteSlot::kNvFp4Down,
                            mlp_disposition)) {
      return fail_enqueue(runner_status(
          ReferenceRunnerError::kRouteEvidenceFailure,
          "prefill_mlp_route", layer));
    }
    if (use_m32_residual_rms_fusion) {
      const bool is_final_layer =
          layer + 1U == kReferenceDecoderLayerCount;
      const std::uint16_t* const next_norm_weight =
          is_final_layer
              ? weights_->final_norm().data
              : weights_->layer(layer + 1U).input_layernorm.data;
      const char* const operation =
          is_final_layer ? "prefill_layer_residual_final_norm"
                         : "prefill_layer_residual_next_input_layernorm";
      if (!residual_norm_m32_tiles(
              execution_views.hidden[2], execution_views.hidden[1], next_norm_weight,
              execution_views.hidden[0], execution_views.hidden[1], operation, layer)) {
        return fail_enqueue(launch_failure);
      }
    } else if (!check_cuda(launch_residual_add_reference_cuda(
                               execution_views.hidden[2], execution_views.hidden[1],
                               token_count * kReferenceHiddenSize,
                               execution_views.hidden[0], stream_),
                           "prefill_layer_residual", layer)) {
      return fail_enqueue(launch_failure);
    }
    const ReferenceRunnerStatus route_status =
        collapse_prefill_layer_route_fragment(layer_route_fragment,
                                              legacy_layer_pass);
    if (!route_status) {
      return fail_enqueue(route_status);
    }
  }

  // Match the non-logit step boundary even though this output is not consumed
  // by the following layer-major tile or by persistent state.
  // The M32-prefix plus reference-tail path folded this norm into the final
  // layer's MLP residual boundary.
  if (control.apply_final_norm && !use_m32_residual_rms_fusion &&
      !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                      execution_views.hidden[0], weights_->final_norm().data,
                      token_count, kReferenceHiddenSize, kRmsEpsilon,
                      execution_views.hidden[1], stream_),
                  "prefill_final_norm", kReferenceNoLayer)) {
    return fail_enqueue(launch_failure);
  }

  PrefillLayerSegmentEnqueueResult result;
  result.route_fragment.legacy_layer_pass = legacy_layer_pass;
  if (control.layer_end == control.layer_begin + 1U) {
    result.route_fragment.layer_segment = layer_route_fragment;
    result.route_fragment.has_single_layer_segment = true;
  }
  return result;
}

ReferenceStepOutcome ReferenceRunner::finish_prefill_from_retained_tile(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }

  if (whole_request_prefill_active()) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_legacy_finalizer_active"));
  }

  // Consume the hand-off before validation or launch. A failed finalization
  // poisons the request exactly like a failed ordinary step and can never
  // accidentally reuse a workspace row after another operation.
  const bool retained_valid = retained_prefill_hidden_valid_;
  const std::uint32_t retained_position = retained_prefill_position_;
  const std::uint32_t retained_input_token =
      retained_prefill_input_token_;
  const std::size_t retained_hidden_row = retained_prefill_hidden_row_;
  retained_prefill_hidden_valid_ = false;

  if (!static_cast<bool>(*this)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "finish_prefill_from_retained_tile"));
  }
  if (poisoned_) {
    ReferenceStepOutcome outcome;
    outcome.status = runner_status(
        ReferenceRunnerError::kPoisoned,
        "finish_prefill_from_retained_tile");
    return outcome;
  }
  if (!options.compute_logits || options.capture_trace ||
      !is_valid_reference_logits_mode(options.logits_mode)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "retained_prefill_logits_options"));
  }
  if (!retained_valid || state_->current_position() == 0U ||
      retained_position + 1U != state_->current_position() ||
      retained_input_token != input_token_id ||
      retained_hidden_row >= kMaximumRequestPrefillChunkSize) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "retained_prefill_hidden_contract"));
  }

  ReferenceRunnerStatus launch_failure{};
  const auto check_cuda = [&launch_failure](
                              const int status,
                              const char* const operation) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    launch_failure = runner_status(ReferenceRunnerError::kCudaFailure,
                                   operation, kReferenceNoLayer, status);
    return false;
  };
  const bool prediction_only =
      options.logits_mode == ReferenceLogitsMode::kPredictedTokenOnly;
  const bool use_sm87_bf16_logits =
      projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
      linear_weight_kind(weights_->lm_head()) != LinearWeightKind::kBf16;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  const std::uint16_t* const final_normalized_hidden =
      views_.hidden[1] + retained_hidden_row * kReferenceHiddenSize;

  if (use_sm87_bf16_logits) {
    auto* const device_bf16_logits =
        reinterpret_cast<std::uint16_t*>(views_.fp32_scratch);
    if (!check_cuda(
            launch_projection_to_bf16_cuda(
                projection_backend_, weights_->lm_head(),
                final_normalized_hidden, nullptr, 0U, device_bf16_logits,
                stream_),
            "retained_prefill_lm_head_sm87_bf16")) {
      return fail_step(launch_failure);
    }
    if (prediction_only) {
      constexpr std::size_t kGreedyWorkspaceBytes =
          kReferenceVocabularySize * sizeof(std::uint16_t) +
          kBf16GreedyArgmaxWorkspaceResults *
              sizeof(Bf16GreedyArgmaxResult);
      static_assert((kReferenceVocabularySize * sizeof(std::uint16_t)) %
                            alignof(Bf16GreedyArgmaxResult) ==
                        0U);
      if (views_.fp32_scratch_elements <
          (kGreedyWorkspaceBytes + sizeof(float) - 1U) / sizeof(float)) {
        return fail_step(runner_status(
            ReferenceRunnerError::kInvalidRequestState,
            "retained_prefill_bf16_greedy_argmax_workspace"));
      }
      auto* const greedy_workspace =
          reinterpret_cast<Bf16GreedyArgmaxResult*>(
              device_bf16_logits + kReferenceVocabularySize);
      if (!check_cuda(
              launch_bf16_greedy_argmax_cuda(
                  device_bf16_logits, kReferenceVocabularySize,
                  greedy_workspace, stream_),
              "retained_prefill_bf16_greedy_argmax") ||
          !check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  pinned_logits_, greedy_workspace,
                  sizeof(Bf16GreedyArgmaxResult), cudaMemcpyDeviceToHost,
                  stream)),
              "retained_prefill_logits_prediction_d2h")) {
        return fail_step(launch_failure);
      }
    } else if (!check_cuda(
                   static_cast<int>(cudaMemcpyAsync(
                       pinned_logits_, device_bf16_logits,
                       kReferenceVocabularySize * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost, stream)),
                   "retained_prefill_logits_bf16_d2h")) {
      return fail_step(launch_failure);
    }
  } else if (!check_cuda(
                 launch_projection_reference_cuda(
                     weights_->lm_head(), final_normalized_hidden,
                     views_.fp32_scratch, stream_),
                 "retained_prefill_lm_head") ||
             !check_cuda(
                 static_cast<int>(cudaMemcpyAsync(
                     pinned_logits_, views_.fp32_scratch,
                     kReferenceVocabularySize * sizeof(float),
                     cudaMemcpyDeviceToHost, stream)),
                 "retained_prefill_logits_d2h")) {
    return fail_step(launch_failure);
  }

  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return fail_step(runner_status(
        ReferenceRunnerError::kCudaFailure,
        "retained_prefill_logits_synchronize", kReferenceNoLayer,
        static_cast<int>(sync_status)));
  }

  ReferenceStepResult result;
  result.position = retained_position;
  result.input_token_id = input_token_id;
  if (prediction_only && use_sm87_bf16_logits) {
    const auto& greedy =
        *static_cast<const Bf16GreedyArgmaxResult*>(pinned_logits_);
    if (greedy.has_nonfinite != 0U) {
      return fail_step(runner_status(
          ReferenceRunnerError::kNonFiniteLogits,
          "retained_prefill_bf16_greedy_argmax"));
    }
    if (greedy.index >= kReferenceVocabularySize) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "retained_prefill_bf16_greedy_argmax_result"));
    }
    result.prediction.emplace(ReferenceStepPrediction{greedy.index});
  } else {
    const reference_runner_detail::LogitsAnalysis analysis =
        use_sm87_bf16_logits
            ? reference_runner_detail::analyze_bf16_logits_bits(
                  static_cast<const std::uint16_t*>(pinned_logits_),
                  kReferenceVocabularySize)
            : (prediction_only
                   ? reference_runner_detail::analyze_bf16_argmax_in_place(
                         static_cast<float*>(pinned_logits_),
                         kReferenceVocabularySize)
                   : reference_runner_detail::analyze_bf16_logits_in_place(
                         static_cast<float*>(pinned_logits_),
                         kReferenceVocabularySize));
    if (!analysis.ok()) {
      return fail_step(runner_status(
          ReferenceRunnerError::kNonFiniteLogits,
          "retained_prefill_bf16_logits_analysis"));
    }
    if (prediction_only) {
      result.prediction.emplace(ReferenceStepPrediction{
          static_cast<std::uint32_t>(analysis.predicted_index)});
    } else {
      ReferenceStepLogits logits;
      logits.predicted_token_id =
          static_cast<std::uint32_t>(analysis.predicted_index);
      logits.chosen_logit = analysis.maximum;
      logits.max_log_probability = analysis.max_log_probability;
      logits.logsumexp = analysis.logsumexp;
      result.logits.emplace(logits);
    }
  }
  if (options.measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    result.timing.emplace(ReferenceStepTiming{elapsed.count()});
  }

  ReferenceStepOutcome outcome;
  outcome.value.emplace(std::move(result));
  return outcome;
}

ReferenceStepOutcome
ReferenceRunner::finish_whole_request_compatibility_core(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }

  if (whole_request_prefill_stage_.phase !=
      WholeRequestPrefillStagePhase::kAwaitingLogits) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_finalizer_stage"));
  }
  WholeRequestPrefillStage stage = whole_request_prefill_stage_;
  whole_request_prefill_stage_.phase =
      WholeRequestPrefillStagePhase::kExecuting;
  if (!static_cast<bool>(*this)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "finish_whole_request_from_uncommitted_final_hidden"));
  }
  if (poisoned_) {
    return fail_step(runner_status(
        ReferenceRunnerError::kPoisoned,
        "finish_whole_request_from_uncommitted_final_hidden"));
  }
  if (!options.compute_logits || options.capture_trace ||
      !is_valid_reference_logits_mode(options.logits_mode)) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_logits_options"));
  }
  const std::uint16_t* const expected_final_hidden =
      layer_major_request_views_.has_value()
          ? static_cast<const std::uint16_t*>(
                layer_major_request_views_->final_hidden_bf16.storage
                    .device_data)
          : nullptr;
  bool complete_progress =
      stage.completed_uncommitted_progress.next_layer ==
          kReferenceDecoderLayerCount &&
      stage.completed_uncommitted_progress.next_panel == 0U &&
      stage.completed_uncommitted_progress.final_hidden_ready &&
      !stage.completed_uncommitted_progress.prefill_state_committed;
  for (std::size_t layer = 0U;
       complete_progress && layer < kReferenceDecoderLayerCount; ++layer) {
    const model::LayerType layer_type =
        reference_runner_detail::expected_reference_layer_type(layer);
    complete_progress =
        stage.completed_uncommitted_progress.completed_panels[layer] ==
            stage.logical_panel_count &&
        stage.completed_uncommitted_progress.completed_mlp_phases[layer] ==
            stage.mlp_phase_count_per_layer &&
        (layer_type == model::LayerType::kFullAttention
             ? stage.completed_uncommitted_progress.kv_visible_end[layer] ==
                       stage.committed_sequence_length &&
                   stage.completed_uncommitted_progress
                           .gdn_advanced_end[layer] ==
                       stage.expected_initial_sequence_length
             : stage.completed_uncommitted_progress
                           .gdn_advanced_end[layer] ==
                       stage.committed_sequence_length &&
                   stage.completed_uncommitted_progress
                           .kv_visible_end[layer] ==
                       stage.expected_initial_sequence_length);
  }
  if (stage.expected_initial_sequence_length !=
          state_->current_position() ||
      stage.committed_sequence_length == 0U ||
      stage.final_position + 1U != stage.committed_sequence_length ||
      stage.final_input_token_id != input_token_id ||
      stage.final_normalized_hidden == nullptr ||
      stage.final_normalized_hidden != expected_final_hidden ||
      !complete_progress) {
    return fail_step(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_final_hidden_contract"));
  }

  ReferenceRunnerStatus launch_failure{};
  const auto check_cuda = [&launch_failure](
                              const int status,
                              const char* const operation) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    launch_failure = runner_status(ReferenceRunnerError::kCudaFailure,
                                   operation, kReferenceNoLayer, status);
    return false;
  };
  const bool prediction_only =
      options.logits_mode == ReferenceLogitsMode::kPredictedTokenOnly;
  const bool use_sm87_bf16_logits =
      projection_backend_ == ProjectionBackend::kSm87WeightOnly &&
      linear_weight_kind(weights_->lm_head()) != LinearWeightKind::kBf16;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);

  if (use_sm87_bf16_logits) {
    auto* const device_bf16_logits =
        reinterpret_cast<std::uint16_t*>(views_.fp32_scratch);
    if (!check_cuda(
            launch_projection_to_bf16_cuda(
                projection_backend_, weights_->lm_head(),
                stage.final_normalized_hidden, nullptr, 0U,
                device_bf16_logits, stream_),
            "whole_request_prefill_lm_head_sm87_bf16")) {
      return fail_step(launch_failure);
    }
    if (prediction_only) {
      constexpr std::size_t kGreedyWorkspaceBytes =
          kReferenceVocabularySize * sizeof(std::uint16_t) +
          kBf16GreedyArgmaxWorkspaceResults *
              sizeof(Bf16GreedyArgmaxResult);
      static_assert((kReferenceVocabularySize * sizeof(std::uint16_t)) %
                            alignof(Bf16GreedyArgmaxResult) ==
                        0U);
      if (views_.fp32_scratch_elements <
          (kGreedyWorkspaceBytes + sizeof(float) - 1U) /
              sizeof(float)) {
        return fail_step(runner_status(
            ReferenceRunnerError::kInvalidRequestState,
            "whole_request_prefill_bf16_greedy_argmax_workspace"));
      }
      auto* const greedy_workspace =
          reinterpret_cast<Bf16GreedyArgmaxResult*>(
              device_bf16_logits + kReferenceVocabularySize);
      if (!check_cuda(
              launch_bf16_greedy_argmax_cuda(
                  device_bf16_logits, kReferenceVocabularySize,
                  greedy_workspace, stream_),
              "whole_request_prefill_bf16_greedy_argmax") ||
          !check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  pinned_logits_, greedy_workspace,
                  sizeof(Bf16GreedyArgmaxResult), cudaMemcpyDeviceToHost,
                  stream)),
              "whole_request_prefill_logits_prediction_d2h")) {
        return fail_step(launch_failure);
      }
    } else if (!check_cuda(
                   static_cast<int>(cudaMemcpyAsync(
                       pinned_logits_, device_bf16_logits,
                       kReferenceVocabularySize * sizeof(std::uint16_t),
                       cudaMemcpyDeviceToHost, stream)),
                   "whole_request_prefill_logits_bf16_d2h")) {
      return fail_step(launch_failure);
    }
  } else if (!check_cuda(
                 launch_projection_reference_cuda(
                     weights_->lm_head(), stage.final_normalized_hidden,
                     views_.fp32_scratch, stream_),
                 "whole_request_prefill_lm_head") ||
             !check_cuda(
                 static_cast<int>(cudaMemcpyAsync(
                     pinned_logits_, views_.fp32_scratch,
                     kReferenceVocabularySize * sizeof(float),
                     cudaMemcpyDeviceToHost, stream)),
                 "whole_request_prefill_logits_d2h")) {
    return fail_step(launch_failure);
  }

  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return fail_step(runner_status(
        ReferenceRunnerError::kCudaFailure,
        "whole_request_prefill_logits_synchronize", kReferenceNoLayer,
        static_cast<int>(sync_status)));
  }

  ReferenceStepResult result;
  result.position = stage.final_position;
  result.input_token_id = input_token_id;
  if (prediction_only && use_sm87_bf16_logits) {
    const auto& greedy =
        *static_cast<const Bf16GreedyArgmaxResult*>(pinned_logits_);
    if (greedy.has_nonfinite != 0U) {
      return fail_step(runner_status(
          ReferenceRunnerError::kNonFiniteLogits,
          "whole_request_prefill_bf16_greedy_argmax"));
    }
    if (greedy.index >= kReferenceVocabularySize) {
      return fail_step(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "whole_request_prefill_bf16_greedy_argmax_result"));
    }
    result.prediction.emplace(ReferenceStepPrediction{greedy.index});
  } else {
    const reference_runner_detail::LogitsAnalysis analysis =
        use_sm87_bf16_logits
            ? reference_runner_detail::analyze_bf16_logits_bits(
                  static_cast<const std::uint16_t*>(pinned_logits_),
                  kReferenceVocabularySize)
            : (prediction_only
                   ? reference_runner_detail::analyze_bf16_argmax_in_place(
                         static_cast<float*>(pinned_logits_),
                         kReferenceVocabularySize)
                   : reference_runner_detail::analyze_bf16_logits_in_place(
                         static_cast<float*>(pinned_logits_),
                         kReferenceVocabularySize));
    if (!analysis.ok()) {
      return fail_step(runner_status(
          ReferenceRunnerError::kNonFiniteLogits,
          "whole_request_prefill_bf16_logits_analysis"));
    }
    if (prediction_only) {
      result.prediction.emplace(ReferenceStepPrediction{
          static_cast<std::uint32_t>(analysis.predicted_index)});
    } else {
      ReferenceStepLogits logits;
      logits.predicted_token_id =
          static_cast<std::uint32_t>(analysis.predicted_index);
      logits.chosen_logit = analysis.maximum;
      logits.max_log_probability = analysis.max_log_probability;
      logits.logsumexp = analysis.logsumexp;
      result.logits.emplace(logits);
    }
  }
  if (options.measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    result.timing.emplace(ReferenceStepTiming{elapsed.count()});
  }

  stage.phase = WholeRequestPrefillStagePhase::kAwaitingCommit;
  whole_request_prefill_stage_ = stage;
  ReferenceStepOutcome outcome;
  outcome.value.emplace(std::move(result));
  return outcome;
}

RequestMemoryProfile
ReferenceRunner::whole_request_commit_memory_profile(
    const LayerMajorPrefillMlpScheduleTactic mlp_schedule_tactic) noexcept {
  return mlp_schedule_tactic ==
                 LayerMajorPrefillMlpScheduleTactic::kPromptWideP40WholeCore ||
             mlp_schedule_tactic == LayerMajorPrefillMlpScheduleTactic::
                                        kPromptWideP40ProjectionReset ||
             mlp_schedule_tactic == LayerMajorPrefillMlpScheduleTactic::
                                        kPromptWideP40PackedProjection ||
             mlp_schedule_tactic == LayerMajorPrefillMlpScheduleTactic::
                                        kPromptWideP40PackedNvfp4V2
         ? RequestMemoryProfile::kLayerMajorP40WholeCore
         : RequestMemoryProfile::kLayerMajorC8192;
}

ReferenceRunnerStatus
ReferenceRunner::commit_whole_request_layer_major_compatibility_core(
    const PrefillExecutionPlan& immutable_topology,
    const PrefillExecutionProgress&
        completed_uncommitted_progress) noexcept {
  if (whole_request_prefill_stage_.phase !=
      WholeRequestPrefillStagePhase::kAwaitingCommit) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_commit_stage"));
  }
  const WholeRequestPrefillStage stage =
      whole_request_prefill_stage_;
  if (!static_cast<bool>(*this)) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "whole_request_prefill_commit"));
  }
  if (poisoned_) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kPoisoned,
        "whole_request_prefill_commit"));
  }
  const std::uint16_t* const expected_final_hidden =
      layer_major_request_views_.has_value()
          ? static_cast<const std::uint16_t*>(
                layer_major_request_views_->final_hidden_bf16.storage
                    .device_data)
          : nullptr;
  const RequestMemoryProfile expected_memory_profile =
      whole_request_commit_memory_profile(
          immutable_topology.mlp_schedule.tactic);
  if (state_->memory_profile() != expected_memory_profile ||
      !is_valid_unbound_layer_major_prefill_execution_plan(
          immutable_topology) ||
      immutable_topology.first_position !=
          stage.expected_initial_sequence_length ||
      immutable_topology.final_position !=
          stage.committed_sequence_length ||
      immutable_topology.prompt_token_count != stage.prompt_token_count ||
      immutable_topology.panel_count != stage.logical_panel_count ||
      immutable_topology.mlp_schedule.mlp_phase_submission_count_per_layer !=
          stage.mlp_phase_count_per_layer ||
      immutable_topology.final_position == 0U ||
      immutable_topology.final_position - 1U != stage.final_position ||
      state_->current_position() !=
          stage.expected_initial_sequence_length ||
      stage.final_normalized_hidden == nullptr ||
      stage.final_normalized_hidden != expected_final_hidden ||
      !same_prefill_execution_progress(
          completed_uncommitted_progress,
          stage.completed_uncommitted_progress) ||
      !prefill_final_commit_ready(
          immutable_topology, completed_uncommitted_progress)) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_commit_contract"));
  }

  const PrefillRouteEvidence& staged_route =
      stage.route_evidence_after_commit;
  const std::uint64_t expected_route_layer_passes =
      prefill_route_layer_pass_count(
          stage.logical_panel_count,
          immutable_topology.mlp_schedule.tactic);
  if (!prefill_route_evidence_.request_active ||
      prefill_route_evidence_.complete ||
      prefill_route_evidence_.error != PrefillRouteEvidenceError::kNone ||
      !staged_route.request_active || staged_route.complete ||
      staged_route.error != PrefillRouteEvidenceError::kNone ||
      prefill_route_evidence_.completed_layer_passes >
          std::numeric_limits<std::uint64_t>::max() -
              expected_route_layer_passes ||
      staged_route.completed_layer_passes !=
          prefill_route_evidence_.completed_layer_passes +
              expected_route_layer_passes) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kRouteEvidenceFailure,
        "whole_request_prefill_commit_route"));
  }
  for (const PrefillOperatorRouteCounts& counts :
       staged_route.operators) {
    if (counts.forbidden_hits != 0U) {
      return fail_whole_request_status(runner_status(
          ReferenceRunnerError::kRouteEvidenceFailure,
          "whole_request_prefill_commit_forbidden_route"));
    }
  }
  for (const std::uint64_t hits :
       staged_route.forbidden_boundary_hits) {
    if (hits != 0U) {
      return fail_whole_request_status(runner_status(
          ReferenceRunnerError::kRouteEvidenceFailure,
          "whole_request_prefill_commit_forbidden_boundary"));
    }
  }

  PrefillExecutionProgress publication_probe =
      completed_uncommitted_progress;
  if (publish_prefill_state_committed(
          immutable_topology, publication_probe) !=
          PrefillExecutionProgressError::kNone ||
      !publication_probe.prefill_state_committed) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "whole_request_prefill_commit_progress"));
  }

  const RequestOperationStatus publication =
      state_->publish_sequence_length(
          stage.expected_initial_sequence_length,
          stage.committed_sequence_length);
  if (!publication) {
    return fail_whole_request_status(runner_status(
        ReferenceRunnerError::kStateCommitFailure,
        "whole_request_prefill_commit_publish", kReferenceNoLayer,
        publication.cuda_error));
  }

  // No fallible operation may follow the sole host-state publication.
  prefill_route_evidence_ = staged_route;
  whole_request_prefill_stage_ = {};
  return {};
}

ReferenceRunnerFactoryResult create_reference_runner(
    const ModelWeights* const weights, RequestState* const state,
    const ReferenceRunnerOptions& options) noexcept {
  ReferenceRunnerFactoryResult result;
  if (!is_valid_projection_backend(options.projection_backend)) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency, "projection_backend");
    return result;
  }
  const ReferenceRunnerStatus weights_status = validate_model_weights(weights);
  if (!weights_status) {
    result.diagnostic = weights_status;
    return result;
  }

  ReferenceRunner runner;
  runner.weights_ = weights;
  runner.state_ = state;
  runner.projection_backend_ = options.projection_backend;
  ReferenceRunnerStatus state_status;
  if (state == nullptr) {
    // Preserve the legacy null-dependency diagnostic and avoid reading a
    // profile from an absent RequestState.
    state_status =
        ReferenceRunner::collect_request_views(state, runner.views_);
  } else {
    switch (state->memory_profile()) {
      case RequestMemoryProfile::kLegacyC512:
        state_status =
            ReferenceRunner::collect_request_views(state, runner.views_);
        break;
      case RequestMemoryProfile::kLayerMajorC8192:
      case RequestMemoryProfile::kLayerMajorP40WholeCore: {
        ReferenceLayerMajorRequestViewsOutcome collected =
            collect_reference_layer_major_candidate_views(state);
        if (!collected) {
          state_status = collected.status;
          break;
        }
        state_status = runner.bind_layer_major_candidate_views(
            std::move(*collected.value));
        break;
      }
      default:
        state_status = runner_status(
            ReferenceRunnerError::kInvalidRequestState,
            "request_memory_profile");
        break;
    }
  }
  if (!state_status) {
    result.diagnostic = state_status;
    return result;
  }

  cudaStream_t stream = nullptr;
  cudaError_t status =
      cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kCudaFailure, "cudaStreamCreateWithFlags",
        kReferenceNoLayer, static_cast<int>(status));
    return result;
  }
  runner.stream_ = reinterpret_cast<void*>(stream);

  if (state->memory_profile() == RequestMemoryProfile::kLayerMajorC8192 ||
      state->memory_profile() ==
          RequestMemoryProfile::kLayerMajorP40WholeCore) {
    for (std::size_t slot = 0U;
         slot < runner.whole_request_submission_events_.size(); ++slot) {
      cudaEvent_t event = nullptr;
      status = cudaEventCreateWithFlags(&event, cudaEventDisableTiming);
      if (status != cudaSuccess) {
        result.diagnostic = runner_status(
            ReferenceRunnerError::kAllocationFailure,
            "cudaEventCreateWithFlags(whole_request_submission)",
            kReferenceNoLayer, static_cast<int>(status));
        return result;
      }
      runner.whole_request_submission_events_[slot] =
          reinterpret_cast<void*>(event);
    }
  }

#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  runner.prefill_gdn_chunk64_native_workspace_bytes_ =
      gdn_prefill_chunk64_native_detail::workspace_bytes();
  status = cudaMalloc(
      &runner.prefill_gdn_chunk64_native_workspace_,
      runner.prefill_gdn_chunk64_native_workspace_bytes_);
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kAllocationFailure,
        "cudaMalloc(gdn_chunk64_native_workspace)", kReferenceNoLayer,
        static_cast<int>(status));
    return result;
  }
#endif

#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  status = static_cast<cudaError_t>(
      gdn_prefill_chunk64_reference_detail::create_context(
          &runner.prefill_gdn_chunk64_reference_context_));
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kAllocationFailure,
        "create_gdn_chunk64_reference_context", kReferenceNoLayer,
        static_cast<int>(status));
    return result;
  }
  runner.prefill_gdn_chunk64_reference_workspace_bytes_ =
      gdn_prefill_chunk64_reference_detail::workspace_bytes();
  status = cudaMalloc(
      &runner.prefill_gdn_chunk64_reference_workspace_,
      runner.prefill_gdn_chunk64_reference_workspace_bytes_);
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kAllocationFailure,
        "cudaMalloc(gdn_chunk64_reference_workspace)", kReferenceNoLayer,
        static_cast<int>(status));
    return result;
  }
#endif

  if (options.projection_backend == ProjectionBackend::kSm87WeightOnly) {
    cudaStream_t auxiliary_stream = nullptr;
    cudaEvent_t branch_ready = nullptr;
    cudaEvent_t branch_done = nullptr;
    const bool auxiliary_ready =
        cudaStreamCreateWithFlags(&auxiliary_stream,
                                  cudaStreamNonBlocking) == cudaSuccess &&
        cudaEventCreateWithFlags(&branch_ready,
                                 cudaEventDisableTiming) == cudaSuccess &&
        cudaEventCreateWithFlags(&branch_done,
                                 cudaEventDisableTiming) == cudaSuccess;
    if (auxiliary_ready) {
      runner.prefill_auxiliary_stream_ =
          reinterpret_cast<void*>(auxiliary_stream);
      runner.prefill_branch_ready_event_ =
          reinterpret_cast<void*>(branch_ready);
      runner.prefill_branch_done_event_ =
          reinterpret_cast<void*>(branch_done);
    } else {
      // The auxiliary branch is a latency optimization, not a runner
      // dependency. Preserve the serial SM87 path if any control resource is
      // unavailable, including after a partially successful allocation.
      if (branch_ready != nullptr) {
        (void)cudaEventDestroy(branch_ready);
      }
      if (branch_done != nullptr) {
        (void)cudaEventDestroy(branch_done);
      }
      if (auxiliary_stream != nullptr) {
        (void)cudaStreamDestroy(auxiliary_stream);
      }
      // This failure is deliberately downgraded to a serial fallback. Do not
      // leak its thread-local CUDA last-error state to the returned runner.
      (void)cudaGetLastError();
    }
  }
  status = cudaHostAlloc(&runner.pinned_logits_,
                         kReferenceVocabularySize * sizeof(float),
                         cudaHostAllocDefault);
  if (status != cudaSuccess) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kAllocationFailure, "cudaHostAlloc(logits)",
        kReferenceNoLayer, static_cast<int>(status));
    return result;
  }

  runner.trace_enabled_ = options.enable_trace;
  if (options.enable_trace) {
    status = cudaHostAlloc(reinterpret_cast<void**>(&runner.pinned_trace_),
                           kReferenceTraceElements * sizeof(std::uint16_t),
                           cudaHostAllocDefault);
    if (status != cudaSuccess) {
      result.diagnostic = runner_status(
          ReferenceRunnerError::kAllocationFailure,
          "cudaHostAlloc(trace)", kReferenceNoLayer,
          static_cast<int>(status));
      return result;
    }
  }

  result.value.emplace(std::move(runner));
  return result;
}

}  // namespace q3x::runtime
