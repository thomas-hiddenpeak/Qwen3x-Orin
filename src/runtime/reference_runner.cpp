#include "q3x/runtime/reference_runner.h"

#include "q3x/kernels/sm87_fp8_prefill_supermatrix.h"
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
#include "q3x/kernels/sm87_a4w4_attention_supermatrix_cell.h"
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
#include "q3x/kernels/sm87_a4w4_attention_o_k512_cell.h"
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
#include "q3x/kernels/sm87_a4w4_down_k512_macrocell.h"
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge.h"
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating.h"
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n64.h"
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix.h"
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
#include "q3x/kernels/sm87_a4w4_down_k512_m128n128_ldmatrix_pairring.h"
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
#include "q3x/kernels/sm87_a4w4_down_k512_m16n64_v2.h"
#endif
#include "q3x/kernels/sm87_a4w4_gateup_k512_macrocell.h"
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native.h"
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_ADMISSION)
#include "q3x/kernels/sm87_a4w4_down_k512_fragment_native_m128n256_1cta.h"
#endif
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native.h"
#if defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m128.h"
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta.h"
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta.h"
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_k512_fragment_native_m128n64_staged.h"
#endif
#include "q3x/runtime/prefill_mlp_k512_fragment_native_overlay.h"
#endif
#include "q3x/kernels/sm87_a4w4_down_k128_stage_major.h"
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
#include "q3x/kernels/sm87_a4w4_down_complete_cell_v2.h"
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
#include "q3x/kernels/sm87_a4w4_down_complete_cell_v3.h"
#endif
#include "q3x/kernels/sm87_a4w4_gateup_complete_cell_v2.h"
#if defined(Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
#include "q3x/kernels/sm87_a4w4_gateup_projection_v3.h"
#endif
#include "q3x/kernels/sm87_a4w4_gateup_paired.h"
#include "q3x/kernels/sm87_a4w4_prefill_gemm.h"
#include "q3x/kernels/sm87_a4w4_prefill_m128_stage_major.h"
#include "q3x/kernels/sm87_a4w4_prefill_primitive.h"
#include "../kernels/reference/gdn_prefill_whole_span_conv_sm87.h"
#include "../kernels/sm87/gdn_prefill_exact_span_sm87.h"
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
constexpr float kRmsEpsilon = 1.0e-6F;
constexpr float kAttentionScale = 1.0F / 16.0F;

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

#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
[[nodiscard]] bool a4w4_full_prefill_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value =
      std::getenv("Q3X_RUN_A4W4_FULL_PREFILL_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_full_prefill_admission =
    a4w4_full_prefill_environment_enabled();
thread_local reference_runner_detail::A4W4FullPrefillAdmissionHits
    g_a4w4_full_prefill_admission_hits{};

#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
[[nodiscard]] bool
a4w4_attention_o_k512_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_ATTENTION_O_K512_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_attention_o_k512_admission =
    a4w4_attention_o_k512_environment_enabled();
thread_local std::size_t g_a4w4_attention_o_k512_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
[[nodiscard]] bool a4w4_mlp_k512_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value =
      std::getenv("Q3X_RUN_A4W4_MLP_K512_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_mlp_k512_admission =
    a4w4_mlp_k512_environment_enabled();
thread_local std::size_t g_a4w4_mlp_k512_admission_hits = 0U;

#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
[[nodiscard]] bool
a4w4_gateup_down_k512_edge_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_gateup_down_k512_edge_admission =
    a4w4_gateup_down_k512_edge_environment_enabled();
thread_local std::size_t
    g_a4w4_gateup_down_k512_edge_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
[[nodiscard]] bool
a4w4_gateup_down_k512_edge_m64n128_k256_alternating_environment_enabled()
    noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool
    g_enable_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission =
        a4w4_gateup_down_k512_edge_m64n128_k256_alternating_environment_enabled();
thread_local std::size_t
    g_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission_hits =
        0U;
#endif

[[nodiscard]] bool
a4w4_mlp_k512_paired_gateup_canonical_down_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_MLP_K512_PAIRED_GATEUP_CANONICAL_DOWN_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

[[nodiscard]] bool
a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_environment_enabled()
    noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

[[nodiscard]] bool
a4w4_down_k512_m128n128_ldmatrix_pairring_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool
    g_enable_a4w4_mlp_k512_paired_gateup_canonical_down_admission =
        a4w4_mlp_k512_paired_gateup_canonical_down_environment_enabled();
thread_local bool
    g_enable_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_admission =
        a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_environment_enabled();
thread_local bool
    g_enable_a4w4_down_k512_m128n128_ldmatrix_pairring_admission =
        a4w4_down_k512_m128n128_ldmatrix_pairring_environment_enabled();
thread_local std::size_t
    g_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_admission_hits =
        0U;
thread_local std::size_t
    g_a4w4_down_k512_m128n128_ldmatrix_pairring_admission_hits = 0U;

#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
[[nodiscard]] bool
a4w4_gateup_down_k512_edge_m128n64_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_gateup_down_k512_edge_m128n64_admission =
    a4w4_gateup_down_k512_edge_m128n64_environment_enabled();
thread_local std::size_t
    g_a4w4_gateup_down_k512_edge_m128n64_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
[[nodiscard]] bool
a4w4_down_k512_m16n64_v2_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_DOWN_K512_M16N64_V2_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_down_k512_m16n64_v2_admission =
    a4w4_down_k512_m16n64_v2_environment_enabled();
thread_local std::size_t
    g_a4w4_down_k512_m16n64_v2_admission_hits = 0U;
#endif
#endif

#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
[[nodiscard]] bool
a4w4_mlp_k512_fragment_native_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_mlp_k512_fragment_native_admission =
    a4w4_mlp_k512_fragment_native_environment_enabled();
thread_local std::size_t
    g_a4w4_mlp_k512_fragment_native_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
[[nodiscard]] bool
a4w4_attention_supermatrix_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_ATTENTION_SUPERMATRIX_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_attention_supermatrix_admission =
    a4w4_attention_supermatrix_environment_enabled();
thread_local reference_runner_detail::
    A4W4AttentionSupermatrixAdmissionHits
        g_a4w4_attention_supermatrix_admission_hits{};
#endif

#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
[[nodiscard]] bool
a4w4_attention_k256_m128n256_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_ATTENTION_K256_M128N256_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_attention_k256_m128n256_admission =
    a4w4_attention_k256_m128n256_environment_enabled();
thread_local reference_runner_detail::
    A4W4AttentionSupermatrixAdmissionHits
        g_a4w4_attention_k256_m128n256_admission_hits{};
#endif

[[nodiscard]] bool
a4w4_gateup_complete_cell_v2_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_COMPLETE_CELL_V2_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_gateup_complete_cell_v2_admission =
    a4w4_gateup_complete_cell_v2_environment_enabled();
thread_local std::size_t
    g_a4w4_gateup_complete_cell_v2_admission_hits = 0U;

#if defined(Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
[[nodiscard]] bool
a4w4_gateup_projection_v3_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_GATEUP_PROJECTION_V3_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_gateup_projection_v3_admission =
    a4w4_gateup_projection_v3_environment_enabled();
thread_local std::size_t
    g_a4w4_gateup_projection_v3_admission_hits = 0U;
#endif

[[nodiscard]] bool
a4w4_m128_stage_major_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value =
      std::getenv("Q3X_RUN_A4W4_M128_STAGE_MAJOR_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_m128_stage_major_admission =
    a4w4_m128_stage_major_environment_enabled();

[[nodiscard]] bool
a4w4_down_m128_stage_major_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value =
      std::getenv("Q3X_RUN_A4W4_DOWN_M128_STAGE_MAJOR_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_down_m128_stage_major_admission =
    a4w4_down_m128_stage_major_environment_enabled();

#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
[[nodiscard]] bool
a4w4_down_complete_cell_v2_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_down_complete_cell_v2_admission =
    a4w4_down_complete_cell_v2_environment_enabled();
thread_local std::size_t
    g_a4w4_down_complete_cell_v2_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
[[nodiscard]] bool
a4w4_down_complete_cell_v3_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value = std::getenv(
      "Q3X_RUN_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_a4w4_down_complete_cell_v3_admission =
    a4w4_down_complete_cell_v3_environment_enabled();
thread_local std::size_t
    g_a4w4_down_complete_cell_v3_admission_hits = 0U;
#endif

#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
[[nodiscard]] reference_runner_detail::
    A4W4PairedGateUpCanonicalDownSelectorQuery
a4w4_paired_gateup_canonical_down_selector_query(
    const bool projection_span) noexcept {
  reference_runner_detail::A4W4PairedGateUpCanonicalDownSelectorQuery query;
  query.master_requested =
      g_enable_a4w4_mlp_k512_paired_gateup_canonical_down_admission;
  query.gate_requested =
      g_enable_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_admission;
  query.down_requested =
      g_enable_a4w4_mlp_k512_paired_gateup_canonical_down_admission &&
      g_enable_a4w4_down_k512_m128n128_ldmatrix_pairring_admission;
  query.legacy_mlp_requested = g_enable_a4w4_mlp_k512_admission;
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
  query.legacy_mlp_requested =
      query.legacy_mlp_requested ||
      g_enable_a4w4_mlp_k512_fragment_native_admission;
#endif
  query.legacy_gate_requested =
      g_enable_a4w4_gateup_complete_cell_v2_admission ||
      g_enable_a4w4_m128_stage_major_admission;
#if defined(Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
  query.legacy_gate_requested =
      query.legacy_gate_requested ||
      g_enable_a4w4_gateup_projection_v3_admission;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
  query.legacy_gate_requested =
      query.legacy_gate_requested ||
      g_enable_a4w4_gateup_down_k512_edge_admission;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
  query.legacy_gate_requested =
      query.legacy_gate_requested ||
      g_enable_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
  query.legacy_gate_requested =
      query.legacy_gate_requested ||
      g_enable_a4w4_gateup_down_k512_edge_m128n64_admission;
#endif
  query.legacy_down_requested =
      g_enable_a4w4_down_m128_stage_major_admission;
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
  query.legacy_down_requested =
      query.legacy_down_requested ||
      g_enable_a4w4_down_complete_cell_v2_admission;
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
  query.legacy_down_requested =
      query.legacy_down_requested ||
      g_enable_a4w4_down_complete_cell_v3_admission;
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
  query.legacy_down_requested =
      query.legacy_down_requested ||
      g_enable_a4w4_down_k512_m16n64_v2_admission;
#endif
  query.projection_span = projection_span;
  return query;
}

[[nodiscard]] reference_runner_detail::
    A4W4DownK512M128N128LdmatrixPairringV1SelectorQuery
a4w4_down_k512_m128n128_ldmatrix_pairring_v1_selector_query(
    const bool projection_span) noexcept {
  reference_runner_detail::
      A4W4DownK512M128N128LdmatrixPairringV1SelectorQuery query;
  // The same leaf selector remains available to the authenticated hybrid
  // publication.  Its master owns that case; without the hybrid master this
  // query makes the selector a strict v1-only Down replacement.
  query.requested =
      g_enable_a4w4_down_k512_m128n128_ldmatrix_pairring_admission &&
      !g_enable_a4w4_mlp_k512_paired_gateup_canonical_down_admission;
  query.mlp_k512_v1_requested = g_enable_a4w4_mlp_k512_admission;
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
  query.fragment_native_requested =
      g_enable_a4w4_mlp_k512_fragment_native_admission;
#endif
  query.hybrid_requested =
      g_enable_a4w4_mlp_k512_paired_gateup_canonical_down_admission;
  query.conflicting_down_requested =
      g_enable_a4w4_down_m128_stage_major_admission;
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
  query.conflicting_down_requested =
      query.conflicting_down_requested ||
      g_enable_a4w4_down_complete_cell_v2_admission;
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
  query.conflicting_down_requested =
      query.conflicting_down_requested ||
      g_enable_a4w4_down_complete_cell_v3_admission;
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
  query.conflicting_down_requested =
      query.conflicting_down_requested ||
      g_enable_a4w4_down_k512_m16n64_v2_admission;
#endif
  query.projection_span = projection_span;
  return query;
}
#endif

// Explicit exact-numerics experiment for the whole-M executor.  This is not
// the archived C64/WY throughput contract: every token persists BF16 state
// and every K dot retains the incumbent left-to-right FMA order.
[[nodiscard]] bool gdn_exact_span_environment_enabled() noexcept {
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
  const char* const value =
      std::getenv("Q3X_RUN_GDN_EXACT_SPAN_ADMISSION");
  return value == nullptr || std::strcmp(value, "1") == 0;
}

thread_local bool g_enable_gdn_exact_span_admission =
    gdn_exact_span_environment_enabled();
#endif

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
  if (optimized_prefill_dispatch_disabled()) {
    return false;
  }
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
// Native C64/WY architecture admission. This build remains opt-in and the
// worker-local route is enabled only by the exact value "1". It has no
// external-library context, fallback, or default-route authority.
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
static_assert(kLinearQkvElements + 2U * kLinearScalarElements <=
              kRequestLongPrefillPrimaryWidth);
static_assert(2U * kFullKvElements <=
              kRequestLongPrefillSecondaryWidth);
static_assert(kLinearValueElements ==
              kRequestLongPrefillSecondaryWidth);
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
    const int cuda_error = 0) noexcept {
  return {error, cuda_error, layer, operation};
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

#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
[[nodiscard]] bool valid_a4w4_prefill_projection(
    const LinearWeight& weight, const std::size_t output_size,
    const std::size_t input_size,
    reference_runner_detail::A4W4PrefillConsumer& inventory_consumer)
    noexcept {
  const PrefillA4LinearSidecarView sidecar =
      prefill_a4_sidecar_view(weight);
  const reference_runner_detail::A4W4PrefillConsumer candidate =
      reference_runner_detail::a4w4_prefill_consumer_from_contract(
          sidecar.sidecar_kind, sidecar.packed_k_group_size,
          sidecar.scale_group_size);
  if (!sidecar.attached() ||
      candidate ==
          reference_runner_detail::A4W4PrefillConsumer::kUnavailable ||
      sidecar.output_size != output_size ||
      sidecar.input_size != input_size ||
      input_size % sidecar.scale_group_size != 0U ||
      output_size % kernels::kSm87A4W4ConsumerOuterBlock != 0U ||
      input_size % kernels::kSm87A4W4ConsumerKBlock != 0U ||
      reinterpret_cast<std::uintptr_t>(sidecar.weight) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(sidecar.scales) %
              alignof(std::uint16_t) !=
          0U) {
    return false;
  }
  if (inventory_consumer ==
      reference_runner_detail::A4W4PrefillConsumer::kUnavailable) {
    inventory_consumer = candidate;
    return true;
  }
  return reference_runner_detail::a4w4_prefill_inventory_consumers_match(
      inventory_consumer, candidate);
}

[[nodiscard]] bool same_a4w4_activation_policy(
    const LinearWeight& first, const LinearWeight& second) noexcept {
  const PrefillA4LinearSidecarView first_sidecar =
      prefill_a4_sidecar_view(first);
  const PrefillA4LinearSidecarView second_sidecar =
      prefill_a4_sidecar_view(second);
  return first_sidecar.attached() && second_sidecar.attached() &&
         first_sidecar.activation_clip_ratio ==
             second_sidecar.activation_clip_ratio;
}

[[nodiscard]] reference_runner_detail::A4W4PrefillConsumer
a4w4_full_prefill_inventory_consumer(
    const ModelWeights& weights) noexcept {
  reference_runner_detail::A4W4PrefillConsumer inventory_consumer =
      reference_runner_detail::A4W4PrefillConsumer::kUnavailable;
  std::size_t projection_count = 0U;
  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& layer_weights = weights.layer(layer);
    if (!valid_a4w4_prefill_projection(
            layer_weights.mlp.gate_proj, kReferenceIntermediateSize,
            kReferenceHiddenSize, inventory_consumer) ||
        !valid_a4w4_prefill_projection(
            layer_weights.mlp.up_proj, kReferenceIntermediateSize,
            kReferenceHiddenSize, inventory_consumer) ||
        !valid_a4w4_prefill_projection(
            layer_weights.mlp.down_proj, kReferenceHiddenSize,
            kReferenceIntermediateSize, inventory_consumer) ||
        !same_a4w4_activation_policy(layer_weights.mlp.gate_proj,
                                     layer_weights.mlp.up_proj)) {
      return reference_runner_detail::A4W4PrefillConsumer::kUnavailable;
    }
    projection_count += 3U;

    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr ||
          !valid_a4w4_prefill_projection(
              attention->in_proj_qkv, kLinearQkvElements,
              kReferenceHiddenSize, inventory_consumer) ||
          !valid_a4w4_prefill_projection(
              attention->in_proj_z, kLinearValueElements,
              kReferenceHiddenSize, inventory_consumer) ||
          !valid_a4w4_prefill_projection(
              attention->out_proj, kReferenceHiddenSize,
              kLinearValueElements, inventory_consumer) ||
          !same_a4w4_activation_policy(attention->in_proj_qkv,
                                       attention->in_proj_z)) {
        return reference_runner_detail::A4W4PrefillConsumer::kUnavailable;
      }
      projection_count += 3U;
      continue;
    }
    if (expected == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr ||
          !valid_a4w4_prefill_projection(
              attention->q_proj, kFullQGateElements,
              kReferenceHiddenSize, inventory_consumer) ||
          !valid_a4w4_prefill_projection(
              attention->k_proj, kFullKvElements,
              kReferenceHiddenSize, inventory_consumer) ||
          !valid_a4w4_prefill_projection(
              attention->v_proj, kFullKvElements,
              kReferenceHiddenSize, inventory_consumer) ||
          !valid_a4w4_prefill_projection(
              attention->o_proj, kReferenceHiddenSize,
              kFullQueryElements, inventory_consumer) ||
          !same_a4w4_activation_policy(attention->q_proj,
                                       attention->k_proj) ||
          !same_a4w4_activation_policy(attention->q_proj,
                                       attention->v_proj)) {
        return reference_runner_detail::A4W4PrefillConsumer::kUnavailable;
      }
      projection_count += 4U;
      continue;
    }
    return reference_runner_detail::A4W4PrefillConsumer::kUnavailable;
  }
  return projection_count == 400U
             ? inventory_consumer
             : reference_runner_detail::A4W4PrefillConsumer::kUnavailable;
}

[[nodiscard]] bool complete_mlp_k512_overlay_attached(
    const ModelWeights& weights) noexcept {
  std::size_t projection_count = 0U;
  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const DecoderLayerWeights& layer_weights = weights.layer(layer);
    const PrefillMLPK512LinearSidecarView gate =
        prefill_mlp_k512_sidecar_view(layer_weights.mlp.gate_proj);
    const PrefillMLPK512LinearSidecarView up =
        prefill_mlp_k512_sidecar_view(layer_weights.mlp.up_proj);
    const PrefillMLPK512LinearSidecarView down =
        prefill_mlp_k512_sidecar_view(layer_weights.mlp.down_proj);
    if (!gate.attached() || !up.attached() || !down.attached() ||
        gate.activation_clip_ratio != up.activation_clip_ratio) {
      return false;
    }
    projection_count += 3U;
  }
  return projection_count == kReferenceDecoderLayerCount * 3U;
}

#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
[[nodiscard]] bool
complete_mlp_k512_paired_gateup_canonical_down_attached(
    const ModelWeights& weights) noexcept {
  std::size_t layer_count = 0U;
  for (std::size_t layer = 0U; layer < kReferenceDecoderLayerCount; ++layer) {
    const PrefillMLPK512FragmentNativeCompositeView& composite =
        weights.layer(layer).prefill_mlp_k512_fragment_native;
    if (!composite.attached() ||
        composite.physical_layout !=
            PrefillMLPK512CompositeLayout::kPairedGateUpCanonicalV1Down ||
        composite.gateup_code_capacity_bytes !=
            kPrefillMLPK512FragmentNativeGateUpCodeBytes ||
        composite.gateup_scale_capacity_elements !=
            kPrefillMLPK512FragmentNativeGateUpScaleBytes /
                sizeof(std::uint16_t) ||
        composite.down_code_capacity_bytes !=
            kPrefillMLPK512FragmentNativeDownCodeBytes ||
        composite.down_scale_capacity_elements !=
            kPrefillMLPK512FragmentNativeDownScaleBytes /
                sizeof(std::uint16_t) ||
        !std::isfinite(composite.gateup_activation_clip_ratio) ||
        !std::isfinite(composite.down_activation_clip_ratio) ||
        reinterpret_cast<std::uintptr_t>(composite.gateup_codes) % 16U !=
            0U ||
        reinterpret_cast<std::uintptr_t>(composite.gateup_scales) % 16U !=
            0U ||
        reinterpret_cast<std::uintptr_t>(composite.down_codes) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(composite.down_scales) % 16U != 0U) {
      return false;
    }
    ++layer_count;
  }
  return layer_count == kReferenceDecoderLayerCount;
}
#endif

[[nodiscard]] std::size_t a4w4_scale_capacity_elements(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const std::size_t outer, const std::size_t input_size) noexcept {
  return consumer == reference_runner_detail::A4W4PrefillConsumer::kK128
             ? kernels::sm87_a4w4_consumer_k128_scale_capacity_elements(
                   outer, input_size)
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
             : consumer ==
                       reference_runner_detail::A4W4PrefillConsumer::kK256
                   ? kernels::
                         sm87_a4w4_attention_k256_scale_capacity_elements(
                             outer, input_size)
#endif
             : consumer ==
                       reference_runner_detail::A4W4PrefillConsumer::kK64
                   ? kernels::sm87_a4w4_consumer_scale_capacity_elements(
                         outer, input_size)
                   : 0U;
}

#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION) || \
    defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
[[nodiscard]] reference_runner_detail::
    A4W4AttentionSupermatrixProjectionPlane
make_a4w4_attention_supermatrix_plane(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const PrefillA4LinearSidecarView& sidecar,
    const std::size_t output_capacity_elements) noexcept {
  reference_runner_detail::A4W4AttentionSupermatrixProjectionPlane plane;
  plane.output_size = sidecar.output_size;
  plane.input_size = sidecar.input_size;
  plane.weight_capacity_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          sidecar.output_size, sidecar.input_size);
  plane.weight_scale_capacity_elements = a4w4_scale_capacity_elements(
      consumer, sidecar.output_size, sidecar.input_size);
  plane.output_row_stride_elements = sidecar.output_size;
  plane.output_capacity_elements = output_capacity_elements;
  return plane;
}

void record_a4w4_attention_supermatrix_launch(
    const reference_runner_detail::A4W4AttentionSupermatrixFamily family,
    reference_runner_detail::A4W4FullPrefillAdmissionHits& local_full_hits,
    reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits&
        local_attention_hits,
    reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits&
        global_attention_hits) noexcept {
  std::size_t logical = 0U;
  switch (family) {
    case reference_runner_detail::A4W4AttentionSupermatrixFamily::
        kLinearInput:
      ++local_attention_hits.linear_input_launch_hits;
      ++global_attention_hits.linear_input_launch_hits;
      logical = 2U;
      break;
    case reference_runner_detail::A4W4AttentionSupermatrixFamily::kFullInput:
      ++local_attention_hits.full_input_launch_hits;
      ++global_attention_hits.full_input_launch_hits;
      logical = 3U;
      break;
    case reference_runner_detail::A4W4AttentionSupermatrixFamily::kOutput:
      ++local_attention_hits.output_launch_hits;
      ++global_attention_hits.output_launch_hits;
      logical = 1U;
      break;
  }
  local_attention_hits.logical_projection_hits += logical;
  global_attention_hits.logical_projection_hits += logical;
  local_full_hits.logical_projection_hits += logical;
  g_a4w4_full_prefill_admission_hits.logical_projection_hits += logical;
}

[[nodiscard]] bool a4w4_attention_supermatrix_aggregate_delta(
    const reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits&
        before,
    const reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits&
        after,
    const std::size_t complete_model_units,
    std::size_t& logical_projection_delta,
    std::size_t& output_projection_delta) noexcept {
  logical_projection_delta = 0U;
  output_projection_delta = 0U;
  if (after.linear_input_launch_hits < before.linear_input_launch_hits ||
      after.full_input_launch_hits < before.full_input_launch_hits ||
      after.output_launch_hits < before.output_launch_hits ||
      after.logical_projection_hits < before.logical_projection_hits) {
    return false;
  }
  const std::size_t linear =
      after.linear_input_launch_hits - before.linear_input_launch_hits;
  const std::size_t full =
      after.full_input_launch_hits - before.full_input_launch_hits;
  const std::size_t output =
      after.output_launch_hits - before.output_launch_hits;
  const std::size_t logical =
      after.logical_projection_hits - before.logical_projection_hits;
  const bool inputs_all =
      linear == complete_model_units * 48U &&
      full == complete_model_units * 16U;
  const bool inputs_none = linear == 0U && full == 0U;
  const bool outputs_all = output == complete_model_units * 64U;
  const bool outputs_none = output == 0U;
  if ((!inputs_all && !inputs_none) ||
      (!outputs_all && !outputs_none) ||
      logical != 2U * linear + 3U * full + output) {
    return false;
  }
  logical_projection_delta = logical;
  output_projection_delta = output;
  return true;
}

#endif

#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)

[[nodiscard]] int launch_selected_a4w4_linear_attention_supermatrix(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const std::uint8_t* const packed_input,
    const std::size_t packed_input_capacity,
    const std::uint16_t* const input_scales,
    const std::size_t input_scale_capacity,
    const PrefillA4LinearSidecarView& qkv_sidecar,
    const PrefillA4LinearSidecarView& z_sidecar,
    const std::size_t token_count, std::uint16_t* const qkv_output,
    const std::size_t qkv_output_capacity,
    std::uint16_t* const z_output,
    const std::size_t z_output_capacity, void* const stream,
    reference_runner_detail::A4W4FullPrefillAdmissionHits& local_full_hits,
    reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits&
        local_attention_hits,
    bool* const selected) noexcept {
  if (selected != nullptr) {
    *selected = false;
  }
  reference_runner_detail::A4W4AttentionSupermatrixRouteQuery query;
  query.admission_enabled =
      g_enable_a4w4_attention_supermatrix_admission;
  query.inventory_consumer = consumer;
  query.family = reference_runner_detail::A4W4AttentionSupermatrixFamily::
      kLinearInput;
  query.projection_token_count = token_count;
  query.packed_input_capacity_bytes = packed_input_capacity;
  query.input_scale_capacity_elements = input_scale_capacity;
  query.projections[0U] = make_a4w4_attention_supermatrix_plane(
      consumer, qkv_sidecar, qkv_output_capacity);
  query.projections[1U] = make_a4w4_attention_supermatrix_plane(
      consumer, z_sidecar, z_output_capacity);
  if (!reference_runner_detail::use_a4w4_attention_supermatrix_route(query)) {
    return static_cast<int>(cudaSuccess);
  }
  if (selected != nullptr) {
    *selected = true;
  }
  const auto& qkv = query.projections[0U];
  const auto& z = query.projections[1U];
  const int status =
      kernels::launch_sm87_a4w4_linear_qkv_z_supermatrix_bf16_cuda(
          packed_input, packed_input_capacity, input_scales,
          input_scale_capacity, qkv_sidecar.weight,
          qkv.weight_capacity_bytes, qkv_sidecar.scales,
          qkv.weight_scale_capacity_elements, z_sidecar.weight,
          z.weight_capacity_bytes, z_sidecar.scales,
          z.weight_scale_capacity_elements, token_count, qkv_output,
          qkv.output_row_stride_elements,
          token_count * qkv.output_row_stride_elements, z_output,
          z.output_row_stride_elements,
          token_count * z.output_row_stride_elements, stream);
  if (status == static_cast<int>(cudaSuccess)) {
    record_a4w4_attention_supermatrix_launch(
        query.family, local_full_hits, local_attention_hits,
        g_a4w4_attention_supermatrix_admission_hits);
  }
  return status;
}

[[nodiscard]] int launch_selected_a4w4_full_attention_supermatrix(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const std::uint8_t* const packed_input,
    const std::size_t packed_input_capacity,
    const std::uint16_t* const input_scales,
    const std::size_t input_scale_capacity,
    const PrefillA4LinearSidecarView& q_sidecar,
    const PrefillA4LinearSidecarView& k_sidecar,
    const PrefillA4LinearSidecarView& v_sidecar,
    const std::size_t token_count, std::uint16_t* const q_output,
    const std::size_t q_output_capacity,
    std::uint16_t* const k_output,
    const std::size_t k_output_capacity,
    std::uint16_t* const v_output,
    const std::size_t v_output_capacity, void* const stream,
    reference_runner_detail::A4W4FullPrefillAdmissionHits& local_full_hits,
    reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits&
        local_attention_hits,
    bool* const selected) noexcept {
  if (selected != nullptr) {
    *selected = false;
  }
  reference_runner_detail::A4W4AttentionSupermatrixRouteQuery query;
  query.admission_enabled =
      g_enable_a4w4_attention_supermatrix_admission;
  query.inventory_consumer = consumer;
  query.family = reference_runner_detail::A4W4AttentionSupermatrixFamily::
      kFullInput;
  query.projection_token_count = token_count;
  query.packed_input_capacity_bytes = packed_input_capacity;
  query.input_scale_capacity_elements = input_scale_capacity;
  query.projections[0U] = make_a4w4_attention_supermatrix_plane(
      consumer, q_sidecar, q_output_capacity);
  query.projections[1U] = make_a4w4_attention_supermatrix_plane(
      consumer, k_sidecar, k_output_capacity);
  query.projections[2U] = make_a4w4_attention_supermatrix_plane(
      consumer, v_sidecar, v_output_capacity);
  if (!reference_runner_detail::use_a4w4_attention_supermatrix_route(query)) {
    return static_cast<int>(cudaSuccess);
  }
  if (selected != nullptr) {
    *selected = true;
  }
  const auto& q = query.projections[0U];
  const auto& k = query.projections[1U];
  const auto& v = query.projections[2U];
  const int status =
      kernels::launch_sm87_a4w4_full_q_k_v_supermatrix_bf16_cuda(
          packed_input, packed_input_capacity, input_scales,
          input_scale_capacity, q_sidecar.weight, q.weight_capacity_bytes,
          q_sidecar.scales, q.weight_scale_capacity_elements,
          k_sidecar.weight, k.weight_capacity_bytes, k_sidecar.scales,
          k.weight_scale_capacity_elements, v_sidecar.weight,
          v.weight_capacity_bytes, v_sidecar.scales,
          v.weight_scale_capacity_elements, token_count, q_output,
          q.output_row_stride_elements,
          token_count * q.output_row_stride_elements, k_output,
          k.output_row_stride_elements,
          token_count * k.output_row_stride_elements, v_output,
          v.output_row_stride_elements,
          token_count * v.output_row_stride_elements, stream);
  if (status == static_cast<int>(cudaSuccess)) {
    record_a4w4_attention_supermatrix_launch(
        query.family, local_full_hits, local_attention_hits,
        g_a4w4_attention_supermatrix_admission_hits);
  }
  return status;
}

[[nodiscard]] int launch_selected_a4w4_attention_o_supermatrix(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const std::uint8_t* const packed_input,
    const std::size_t packed_input_capacity,
    const std::uint16_t* const input_scales,
    const std::size_t input_scale_capacity,
    const PrefillA4LinearSidecarView& output_sidecar,
    const std::size_t token_count, std::uint16_t* const output,
    const std::size_t output_capacity, void* const stream,
    reference_runner_detail::A4W4FullPrefillAdmissionHits& local_full_hits,
    reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits&
        local_attention_hits,
    bool* const selected) noexcept {
  if (selected != nullptr) {
    *selected = false;
  }
  reference_runner_detail::A4W4AttentionSupermatrixRouteQuery query;
  query.admission_enabled =
      g_enable_a4w4_attention_supermatrix_admission;
  query.inventory_consumer = consumer;
  query.family =
      reference_runner_detail::A4W4AttentionSupermatrixFamily::kOutput;
  query.projection_token_count = token_count;
  query.packed_input_capacity_bytes = packed_input_capacity;
  query.input_scale_capacity_elements = input_scale_capacity;
  query.projections[0U] = make_a4w4_attention_supermatrix_plane(
      consumer, output_sidecar, output_capacity);
  if (!reference_runner_detail::use_a4w4_attention_supermatrix_route(query)) {
    return static_cast<int>(cudaSuccess);
  }
  if (selected != nullptr) {
    *selected = true;
  }
  const auto& plane = query.projections[0U];
  const int status =
      kernels::launch_sm87_a4w4_attention_o_supermatrix_bf16_cuda(
          packed_input, packed_input_capacity, input_scales,
          input_scale_capacity, output_sidecar.weight,
          plane.weight_capacity_bytes, output_sidecar.scales,
          plane.weight_scale_capacity_elements, token_count, output,
          plane.output_row_stride_elements,
          token_count * plane.output_row_stride_elements, stream);
  if (status == static_cast<int>(cudaSuccess)) {
    record_a4w4_attention_supermatrix_launch(
        query.family, local_full_hits, local_attention_hits,
        g_a4w4_attention_supermatrix_admission_hits);
  }
  return status;
}
#endif

#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
[[nodiscard]] int launch_selected_a4w4_attention_k256_m128n256(
    const kernels::Sm87A4W4AttentionK256Topology topology,
    const reference_runner_detail::A4W4AttentionSupermatrixFamily family,
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const std::uint8_t* const packed_input,
    const std::size_t packed_input_capacity,
    const std::uint16_t* const input_scales,
    const std::size_t input_scale_capacity,
    const PrefillA4LinearSidecarView* const sidecars,
    std::uint16_t* const* const outputs,
    const std::size_t* const output_capacities,
    const std::size_t projection_count, const std::size_t token_count,
    void* const stream,
    reference_runner_detail::A4W4FullPrefillAdmissionHits& local_full_hits,
    reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits&
        local_attention_hits,
    bool* const selected) noexcept {
  if (selected != nullptr) {
    *selected = false;
  }
  const bool topology_matches =
      (topology == kernels::Sm87A4W4AttentionK256Topology::kLinearQkvZ &&
       family == reference_runner_detail::A4W4AttentionSupermatrixFamily::
                     kLinearInput &&
       projection_count == 2U) ||
      (topology == kernels::Sm87A4W4AttentionK256Topology::kFullQkv &&
       family == reference_runner_detail::A4W4AttentionSupermatrixFamily::
                     kFullInput &&
       projection_count == 3U) ||
      (topology == kernels::Sm87A4W4AttentionK256Topology::kAttentionO &&
       family == reference_runner_detail::A4W4AttentionSupermatrixFamily::
                     kOutput &&
       projection_count == 1U);
  if (!topology_matches || sidecars == nullptr || outputs == nullptr ||
      output_capacities == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }

  reference_runner_detail::A4W4AttentionSupermatrixRouteQuery query;
  query.admission_enabled =
      g_enable_a4w4_attention_k256_m128n256_admission;
  query.inventory_consumer = consumer;
  query.family = family;
  query.projection_token_count = token_count;
  query.packed_input_capacity_bytes = packed_input_capacity;
  query.input_scale_capacity_elements = input_scale_capacity;
  for (std::size_t index = 0U; index < projection_count; ++index) {
    query.projections[index] = make_a4w4_attention_supermatrix_plane(
        consumer, sidecars[index], output_capacities[index]);
  }
  if (!reference_runner_detail::
          use_a4w4_attention_k256_m128n256_route(query)) {
    return static_cast<int>(cudaSuccess);
  }
  if (selected != nullptr) {
    *selected = true;
  }

  std::array<kernels::Sm87A4W4AttentionK256ProjectionView, 3U>
      projection_views{};
  for (std::size_t index = 0U; index < projection_count; ++index) {
    const auto& plane = query.projections[index];
    projection_views[index] = {
        sidecars[index].weight,
        plane.weight_capacity_bytes,
        sidecars[index].scales,
        plane.weight_scale_capacity_elements,
        sidecars[index].output_size,
        outputs[index],
        plane.output_row_stride_elements,
        plane.output_capacity_elements};
  }
  const int status =
      kernels::launch_sm87_a4w4_attention_k256_m128n256_bf16_cuda(
          topology, packed_input, packed_input_capacity, input_scales,
          input_scale_capacity, token_count, projection_views.data(),
          projection_count, stream);
  if (status == static_cast<int>(cudaSuccess)) {
    record_a4w4_attention_supermatrix_launch(
        family, local_full_hits, local_attention_hits,
        g_a4w4_attention_k256_m128n256_admission_hits);
  }
  return status;
}
#endif

#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
[[nodiscard]] int launch_selected_a4w4_attention_o_k512(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const LinearWeight& weight, const std::uint16_t* const input_bf16,
    const std::size_t input_row_stride_elements,
    const std::size_t logical_token_count,
    const std::size_t launch_token_count,
    std::uint8_t* const packed_input,
    const std::size_t packed_input_capacity_bytes,
    std::uint16_t* const input_scales,
    const std::size_t input_scale_capacity_elements,
    std::uint16_t* const output_bf16,
    const std::size_t output_row_stride_elements,
    const std::size_t output_capacity_elements, void* const stream,
    bool* const selected) noexcept {
  if (selected != nullptr) {
    *selected = false;
  }
  if (!g_enable_a4w4_attention_o_k512_admission) {
    return static_cast<int>(cudaSuccess);
  }
  const PrefillAttentionOK512LinearSidecarView sidecar =
      prefill_attention_o_k512_sidecar_view(weight);
  const std::size_t expected_launch_tokens =
      kernels::sm87_a4w4_prefill_k512_launch_token_count(
          logical_token_count);
  if (consumer !=
          reference_runner_detail::A4W4PrefillConsumer::kK128 ||
      !sidecar.attached() || logical_token_count == 0U ||
      launch_token_count != expected_launch_tokens ||
      output_row_stride_elements < sidecar.output_size) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if (selected != nullptr) {
    *selected = true;
  }
  int status = kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
      input_bf16, input_row_stride_elements, logical_token_count,
      launch_token_count, sidecar.input_size,
      sidecar.activation_clip_ratio, packed_input,
      packed_input_capacity_bytes, input_scales,
      input_scale_capacity_elements, stream);
  if (status != static_cast<int>(cudaSuccess)) {
    return status;
  }
  const std::size_t weight_capacity =
      kernels::sm87_a4w4_attention_o_k512_packed_capacity_bytes(
          sidecar.output_size, sidecar.input_size);
  const std::size_t weight_scale_capacity =
      kernels::sm87_a4w4_attention_o_k512_scale_capacity_elements(
          sidecar.output_size, sidecar.input_size);
  status = kernels::launch_sm87_a4w4_attention_o_k512_bf16_cuda(
      packed_input, packed_input_capacity_bytes, input_scales,
      input_scale_capacity_elements, sidecar.weight, weight_capacity,
      sidecar.scales, weight_scale_capacity, launch_token_count,
      sidecar.output_size, sidecar.input_size, output_bf16,
      output_row_stride_elements, output_capacity_elements, stream);
  if (status == static_cast<int>(cudaSuccess)) {
    ++g_a4w4_attention_o_k512_admission_hits;
  }
  return status;
}
#endif

[[nodiscard]] int launch_selected_a4w4_quantize(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const std::uint16_t* const input, const std::size_t input_row_stride,
    const std::size_t token_count, const std::size_t input_size,
    const float clip_ratio, std::uint8_t* const packed,
    const std::size_t packed_capacity, std::uint16_t* const scales,
    const std::size_t scale_capacity, void* const stream,
    const std::size_t launch_token_count = 0U) noexcept {
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
  if (consumer == reference_runner_detail::A4W4PrefillConsumer::kK256) {
    return kernels::launch_sm87_a4_quantize_bf16_k256_cuda(
        input, input_row_stride, token_count, launch_token_count, input_size,
        clip_ratio, packed, packed_capacity, scales, scale_capacity, stream);
  }
#endif
  if (consumer == reference_runner_detail::A4W4PrefillConsumer::kK128) {
    return kernels::launch_sm87_a4_quantize_bf16_k128_cuda(
        input, input_row_stride, token_count, input_size, clip_ratio, packed,
        packed_capacity, scales, scale_capacity, stream);
  }
  if (consumer == reference_runner_detail::A4W4PrefillConsumer::kK64) {
    return kernels::launch_sm87_a4_quantize_bf16_cuda(
        input, input_row_stride, token_count, input_size, clip_ratio, packed,
        packed_capacity, scales, scale_capacity, stream);
  }
  return static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] int launch_selected_a4w4_gemm(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const std::uint8_t* const packed_input,
    const std::size_t packed_input_capacity,
    const std::uint16_t* const input_scales,
    const std::size_t input_scale_capacity,
    const PrefillA4LinearSidecarView& sidecar,
    const std::size_t weight_capacity,
    const std::size_t weight_scale_capacity,
    const std::size_t token_count, std::uint16_t* const output,
    const std::size_t output_capacity_elements, void* const stream,
    // Reports either selected complete cell. v3/v2 retain independent global
    // success counters; this aggregate bit drives common route accounting.
    bool* const down_complete_cell_v2_selected) noexcept {
  if (down_complete_cell_v2_selected != nullptr) {
    *down_complete_cell_v2_selected = false;
  }
  if (consumer == reference_runner_detail::A4W4PrefillConsumer::kK128) {
    reference_runner_detail::A4W4DownCompleteCellV2RouteQuery cell_query;
    cell_query.inventory_consumer = consumer;
    cell_query.projection_token_count = token_count;
    cell_query.output_size = sidecar.output_size;
    cell_query.input_size = sidecar.input_size;
    cell_query.packed_input_capacity_bytes = packed_input_capacity;
    cell_query.input_scale_capacity_elements = input_scale_capacity;
    cell_query.weight_capacity_bytes = weight_capacity;
    cell_query.weight_scale_capacity_elements = weight_scale_capacity;
    cell_query.output_capacity_elements = output_capacity_elements;
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
    cell_query.admission_enabled =
        g_enable_a4w4_down_complete_cell_v2_admission;
#endif
    reference_runner_detail::A4W4DownCompleteCellV3RouteQuery
        cell_v3_query = cell_query;
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
    cell_v3_query.admission_enabled =
        g_enable_a4w4_down_complete_cell_v3_admission;
#else
    cell_v3_query.admission_enabled = false;
#endif
    const auto down_route =
        reference_runner_detail::select_a4w4_k128_down_prefill_route(
            cell_v3_query, cell_query,
            g_enable_a4w4_down_m128_stage_major_admission);
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
    if (down_route ==
        reference_runner_detail::A4W4K128DownPrefillRoute::kCompleteCellV3) {
      if (down_complete_cell_v2_selected != nullptr) {
        *down_complete_cell_v2_selected = true;
      }
      const int status =
          kernels::launch_sm87_a4w4_down_complete_cell_v3_bf16_cuda(
              packed_input, packed_input_capacity, input_scales,
              input_scale_capacity, sidecar.weight, weight_capacity,
              sidecar.scales, weight_scale_capacity, token_count,
              sidecar.output_size, sidecar.input_size, output,
              sidecar.output_size, stream);
      if (status == static_cast<int>(cudaSuccess)) {
        ++g_a4w4_down_complete_cell_v3_admission_hits;
      }
      return status;
    }
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
    if (down_route ==
        reference_runner_detail::A4W4K128DownPrefillRoute::kCompleteCellV2) {
      if (down_complete_cell_v2_selected != nullptr) {
        *down_complete_cell_v2_selected = true;
      }
      const int status =
          kernels::launch_sm87_a4w4_down_complete_cell_v2_bf16_cuda(
              packed_input, packed_input_capacity, input_scales,
              input_scale_capacity, sidecar.weight, weight_capacity,
              sidecar.scales, weight_scale_capacity, token_count,
              sidecar.output_size, sidecar.input_size, output,
              sidecar.output_size, stream);
      if (status == static_cast<int>(cudaSuccess)) {
        ++g_a4w4_down_complete_cell_v2_admission_hits;
      }
      return status;
    }
#endif
    if (down_route == reference_runner_detail::A4W4K128DownPrefillRoute::
                          kRejectedM128StageMajor) {
      return kernels::launch_sm87_a4w4_down_k128_stage_major_bf16_cuda(
          packed_input, packed_input_capacity, input_scales,
          input_scale_capacity, sidecar.weight, weight_capacity,
          sidecar.scales, weight_scale_capacity, token_count,
          sidecar.output_size, sidecar.input_size, output,
          sidecar.output_size, stream);
    }
    const auto route =
        reference_runner_detail::select_a4w4_k128_generic_prefill_route(
            g_enable_a4w4_m128_stage_major_admission, false, consumer,
            token_count, sidecar.output_size, sidecar.input_size);
    if (route == reference_runner_detail::A4W4K128GenericPrefillRoute::
                     kM128StageMajor) {
      return kernels::launch_sm87_a4w4_m128_stage_major_bf16_cuda(
          packed_input, packed_input_capacity, input_scales,
          input_scale_capacity, sidecar.weight, weight_capacity,
          sidecar.scales, weight_scale_capacity, token_count,
          sidecar.output_size, sidecar.input_size, output,
          sidecar.output_size, stream);
    }
    return kernels::launch_sm87_a4w4_prefill_gemm_k128_bf16_cuda(
        packed_input, packed_input_capacity, input_scales,
        input_scale_capacity, sidecar.weight, weight_capacity,
        sidecar.scales, weight_scale_capacity, token_count,
        sidecar.output_size, sidecar.input_size, output, sidecar.output_size,
        stream);
  }
  if (consumer == reference_runner_detail::A4W4PrefillConsumer::kK64) {
    (void)output_capacity_elements;
    return kernels::launch_sm87_a4w4_prefill_gemm_bf16_cuda(
        packed_input, packed_input_capacity, input_scales,
        input_scale_capacity, sidecar.weight, weight_capacity,
        sidecar.scales, weight_scale_capacity, token_count,
        sidecar.output_size, sidecar.input_size, output, sidecar.output_size,
        stream);
  }
  return static_cast<int>(cudaErrorInvalidValue);
}

[[nodiscard]] int launch_selected_a4w4_gateup_paired(
    const reference_runner_detail::A4W4PrefillConsumer consumer,
    const std::uint8_t* const packed_input,
    const std::size_t packed_input_capacity,
    const std::uint16_t* const input_scales,
    const std::size_t input_scale_capacity,
    const PrefillA4LinearSidecarView& gate_sidecar,
    const std::size_t gate_weight_capacity,
    const std::size_t gate_scale_capacity,
    const PrefillA4LinearSidecarView& up_sidecar,
    const std::size_t up_weight_capacity,
    const std::size_t up_scale_capacity, const std::size_t token_count,
    const float output_clip_ratio, std::uint8_t* const packed_output,
    const std::size_t packed_output_capacity,
    std::uint16_t* const output_scales,
    const std::size_t output_scale_capacity, void* const stream,
    bool* const projection_v3_selected,
    bool* const complete_cell_v2_selected) noexcept {
  if (projection_v3_selected != nullptr) {
    *projection_v3_selected = false;
  }
  if (complete_cell_v2_selected != nullptr) {
    *complete_cell_v2_selected = false;
  }
  if (consumer == reference_runner_detail::A4W4PrefillConsumer::kK128) {
    reference_runner_detail::A4W4GateUpCompleteCellV2RouteQuery
        complete_cell_v2_query;
    complete_cell_v2_query.admission_enabled =
        g_enable_a4w4_gateup_complete_cell_v2_admission;
    reference_runner_detail::A4W4GateUpProjectionV3RouteQuery
        projection_v3_query;
#if defined(Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
    projection_v3_query.admission_enabled =
        g_enable_a4w4_gateup_projection_v3_admission;
#endif
    const auto fill_query = [&](auto& query) {
      query.inventory_consumer = consumer;
      query.projection_token_count = token_count;
      query.gate_output_size = gate_sidecar.output_size;
      query.gate_input_size = gate_sidecar.input_size;
      query.up_output_size = up_sidecar.output_size;
      query.up_input_size = up_sidecar.input_size;
      query.packed_input_capacity_bytes = packed_input_capacity;
      query.input_scale_capacity_elements = input_scale_capacity;
      query.gate_weight_capacity_bytes = gate_weight_capacity;
      query.gate_scale_capacity_elements = gate_scale_capacity;
      query.up_weight_capacity_bytes = up_weight_capacity;
      query.up_scale_capacity_elements = up_scale_capacity;
      query.packed_output_capacity_bytes = packed_output_capacity;
      query.output_scale_capacity_elements = output_scale_capacity;
    };
    fill_query(complete_cell_v2_query);
    fill_query(projection_v3_query);
    const reference_runner_detail::A4W4K128GateUpPrefillRoute route =
        reference_runner_detail::select_a4w4_k128_gateup_prefill_route(
            projection_v3_query, complete_cell_v2_query,
            g_enable_a4w4_m128_stage_major_admission);
    if (route == reference_runner_detail::A4W4K128GateUpPrefillRoute::
                     kProjectionV3) {
#if defined(Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
      if (projection_v3_selected != nullptr) {
        *projection_v3_selected = true;
      }
      const int status =
          kernels::launch_sm87_a4w4_gateup_projection_v3_cuda(
              packed_input, packed_input_capacity, input_scales,
              input_scale_capacity, gate_sidecar.weight,
              gate_weight_capacity, gate_sidecar.scales,
              gate_scale_capacity, up_sidecar.weight, up_weight_capacity,
              up_sidecar.scales, up_scale_capacity, token_count,
              gate_sidecar.output_size, gate_sidecar.input_size,
              output_clip_ratio, packed_output, packed_output_capacity,
              output_scales, output_scale_capacity, stream);
      if (status == static_cast<int>(cudaSuccess)) {
        ++g_a4w4_gateup_projection_v3_admission_hits;
      }
      return status;
#else
      return static_cast<int>(cudaErrorInvalidValue);
#endif
    }
    if (route == reference_runner_detail::A4W4K128GateUpPrefillRoute::
                     kCompleteCellV2) {
      if (complete_cell_v2_selected != nullptr) {
        *complete_cell_v2_selected = true;
      }
      const int status = kernels::launch_sm87_a4w4_gateup_cell_v2_cuda(
          packed_input, packed_input_capacity, input_scales,
          input_scale_capacity, gate_sidecar.weight, gate_weight_capacity,
          gate_sidecar.scales, gate_scale_capacity, up_sidecar.weight,
          up_weight_capacity, up_sidecar.scales, up_scale_capacity,
          token_count, gate_sidecar.output_size, gate_sidecar.input_size,
          output_clip_ratio, packed_output, packed_output_capacity,
          output_scales, output_scale_capacity, stream);
      if (status == static_cast<int>(cudaSuccess)) {
        ++g_a4w4_gateup_complete_cell_v2_admission_hits;
      }
      return status;
    }
    if (route == reference_runner_detail::A4W4K128GateUpPrefillRoute::
                     kRejectedM128StageMajor) {
      return kernels::launch_sm87_a4w4_m128_stage_major_paired_cuda(
          packed_input, packed_input_capacity, input_scales,
          input_scale_capacity, gate_sidecar.weight, gate_weight_capacity,
          gate_sidecar.scales, gate_scale_capacity, up_sidecar.weight,
          up_weight_capacity, up_sidecar.scales, up_scale_capacity,
          token_count, kReferenceIntermediateSize, kReferenceHiddenSize,
          output_clip_ratio, packed_output, packed_output_capacity,
          output_scales, output_scale_capacity, stream);
    }
    return kernels::launch_sm87_a4w4_gateup_paired_k128_cuda(
        packed_input, packed_input_capacity, input_scales,
        input_scale_capacity, gate_sidecar.weight, gate_weight_capacity,
        gate_sidecar.scales, gate_scale_capacity, up_sidecar.weight,
        up_weight_capacity, up_sidecar.scales, up_scale_capacity, token_count,
        kReferenceIntermediateSize, kReferenceHiddenSize, output_clip_ratio,
        packed_output, packed_output_capacity, output_scales,
        output_scale_capacity, stream);
  }
  if (consumer == reference_runner_detail::A4W4PrefillConsumer::kK64) {
    return kernels::launch_sm87_a4w4_gateup_paired_cuda(
        packed_input, packed_input_capacity, input_scales,
        input_scale_capacity, gate_sidecar.weight, gate_weight_capacity,
        gate_sidecar.scales, gate_scale_capacity, up_sidecar.weight,
        up_weight_capacity, up_sidecar.scales, up_scale_capacity, token_count,
        kReferenceIntermediateSize, kReferenceHiddenSize, output_clip_ratio,
        packed_output, packed_output_capacity, output_scales,
        output_scale_capacity, stream);
  }
  return static_cast<int>(cudaErrorInvalidValue);
}
#endif

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
  if (state->plan().long_prefill_token_capacity != 0U) {
    const std::uint64_t long_hidden_elements =
        static_cast<std::uint64_t>(
            state->plan().long_prefill_token_capacity) *
        kReferenceHiddenSize;
    for (std::size_t index = 0U;
         index < kRequestLongPrefillHiddenBufferCount; ++index) {
      RequestViewResult view = state->long_prefill_hidden_buffer(index);
      if (!valid_view(view, long_hidden_elements, sizeof(std::uint16_t))) {
        return runner_status(ReferenceRunnerError::kInvalidRequestState,
                             "long_prefill_hidden_workspace");
      }
      views.long_prefill_hidden[index] =
          static_cast<std::uint16_t*>(view.value->device_data);
    }
  }
  if (state->plan().long_prefill_projection_span_capacity != 0U) {
    const std::uint64_t span_tokens =
        state->plan().long_prefill_projection_span_capacity;
    RequestViewResult primary =
        state->long_prefill_projection_primary_buffer();
    RequestViewResult secondary =
        state->long_prefill_projection_secondary_buffer();
    if (!valid_view(primary,
                    span_tokens * kRequestLongPrefillPrimaryWidth,
                    sizeof(std::uint16_t)) ||
        !valid_view(secondary,
                    span_tokens * kRequestLongPrefillSecondaryWidth,
                    sizeof(std::uint16_t))) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "long_prefill_projection_span_workspace");
    }
    views.long_prefill_projection_primary =
        static_cast<std::uint16_t*>(primary.value->device_data);
    views.long_prefill_projection_secondary =
        static_cast<std::uint16_t*>(secondary.value->device_data);
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
  if (state->plan().prefill_a4_hidden_packed.byte_size != 0U) {
    RequestViewResult hidden_packed = state->prefill_a4_hidden_packed();
    RequestViewResult hidden_scales = state->prefill_a4_hidden_scales();
    RequestViewResult intermediate_packed =
        state->prefill_a4_intermediate_packed();
    RequestViewResult intermediate_scales =
        state->prefill_a4_intermediate_scales();
    RequestViewResult gateup_cta_scratch =
        state->prefill_a4_gateup_cta_scratch();
    if (!valid_view(hidden_packed,
                    workspace_tokens * kReferenceHiddenSize / 2U, 1U) ||
        !valid_view(hidden_scales,
                    workspace_tokens * kReferenceHiddenSize /
                        kRequestA4PrefillScaleGroupSize,
                    sizeof(std::uint16_t)) ||
        !valid_view(intermediate_packed,
                    workspace_tokens * kReferenceIntermediateSize / 2U,
                    1U) ||
        !valid_view(intermediate_scales,
                    workspace_tokens * kReferenceIntermediateSize /
                        kRequestA4PrefillScaleGroupSize,
                    sizeof(std::uint16_t)) ||
        !valid_view(gateup_cta_scratch,
                    kRequestA4GateUpCtaScratchBytes, 1U)) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_a4_workspace");
    }
    views.prefill_a4_hidden_packed =
        static_cast<std::uint8_t*>(hidden_packed.value->device_data);
    views.prefill_a4_hidden_scales =
        static_cast<std::uint16_t*>(hidden_scales.value->device_data);
    views.prefill_a4_intermediate_packed =
        static_cast<std::uint8_t*>(intermediate_packed.value->device_data);
    views.prefill_a4_intermediate_scales =
        static_cast<std::uint16_t*>(intermediate_scales.value->device_data);
    views.prefill_a4_gateup_cta_scratch = static_cast<std::uint8_t*>(
        gateup_cta_scratch.value->device_data);
    views.prefill_a4_gateup_cta_scratch_bytes =
        static_cast<std::size_t>(gateup_cta_scratch.value->byte_size);
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

bool exchange_bf16_ab_large_m_prefill_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
  return std::exchange(g_enable_bf16_ab_large_m_prefill_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

std::size_t exchange_bf16_ab_large_m_prefill_admission_test_hits(
    const std::size_t hits) noexcept {
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
  return std::exchange(g_bf16_ab_large_m_prefill_admission_hits, hits);
#else
  (void)hits;
  return 0U;
#endif
}

bool exchange_a4w4_full_prefill_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  return std::exchange(g_enable_a4w4_full_prefill_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

bool exchange_a4w4_m128_stage_major_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  return std::exchange(g_enable_a4w4_m128_stage_major_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

bool exchange_a4w4_down_m128_stage_major_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  return std::exchange(g_enable_a4w4_down_m128_stage_major_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

bool exchange_a4w4_down_complete_cell_v2_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
  return std::exchange(
      g_enable_a4w4_down_complete_cell_v2_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

std::size_t exchange_a4w4_down_complete_cell_v2_admission_test_hits(
    const std::size_t hits) noexcept {
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
  return std::exchange(
      g_a4w4_down_complete_cell_v2_admission_hits, hits);
#else
  (void)hits;
  return 0U;
#endif
}

bool exchange_a4w4_down_complete_cell_v3_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
  return std::exchange(
      g_enable_a4w4_down_complete_cell_v3_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

std::size_t exchange_a4w4_down_complete_cell_v3_admission_test_hits(
    const std::size_t hits) noexcept {
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
  return std::exchange(
      g_a4w4_down_complete_cell_v3_admission_hits, hits);
#else
  (void)hits;
  return 0U;
#endif
}

bool exchange_a4w4_gateup_complete_cell_v2_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  return std::exchange(
      g_enable_a4w4_gateup_complete_cell_v2_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

std::size_t
exchange_a4w4_gateup_complete_cell_v2_admission_test_hits(
    const std::size_t hits) noexcept {
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  return std::exchange(
      g_a4w4_gateup_complete_cell_v2_admission_hits, hits);
#else
  (void)hits;
  return 0U;
#endif
}

bool exchange_a4w4_gateup_projection_v3_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
  return std::exchange(
      g_enable_a4w4_gateup_projection_v3_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

std::size_t exchange_a4w4_gateup_projection_v3_admission_test_hits(
    const std::size_t hits) noexcept {
#if defined(Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
  return std::exchange(
      g_a4w4_gateup_projection_v3_admission_hits, hits);
#else
  (void)hits;
  return 0U;
#endif
}

bool exchange_a4w4_attention_supermatrix_admission_test_enabled(
    const bool enabled) noexcept {
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  return std::exchange(
      g_enable_a4w4_attention_supermatrix_admission, enabled);
#else
  (void)enabled;
  return false;
#endif
}

A4W4AttentionSupermatrixAdmissionHits
exchange_a4w4_attention_supermatrix_admission_test_hits(
    const A4W4AttentionSupermatrixAdmissionHits hits) noexcept {
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  return std::exchange(
      g_a4w4_attention_supermatrix_admission_hits, hits);
#else
  (void)hits;
  return {};
#endif
}

A4W4FullPrefillAdmissionHits
exchange_a4w4_full_prefill_admission_test_hits(
    const A4W4FullPrefillAdmissionHits hits) noexcept {
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  return std::exchange(g_a4w4_full_prefill_admission_hits, hits);
#else
  (void)hits;
  return {};
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
  const auto byte_region_at_least = [](const RequestRegion& region,
                                       const std::uint64_t bytes) noexcept {
    return region.element_size_bytes == 1U &&
           region.element_capacity >= bytes && region.byte_size >= bytes;
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
  if (plan.long_prefill_projection_span_capacity != 0U) {
    const std::uint64_t span_tokens =
        plan.long_prefill_projection_span_capacity;
    if (plan.long_prefill_token_capacity == 0U ||
        span_tokens > plan.long_prefill_token_capacity ||
        span_tokens % kRequestLongPrefillProjectionSpanAlignment != 0U ||
        !bf16_region_at_least(
            plan.long_prefill_projection_primary_bf16,
            span_tokens * kRequestLongPrefillPrimaryWidth) ||
        !bf16_region_at_least(
            plan.long_prefill_projection_secondary_bf16,
            span_tokens * kRequestLongPrefillSecondaryWidth)) {
      return ReferenceRunnerError::kInvalidRequestState;
    }
  }
  const bool has_a4_workspace =
      plan.prefill_a4_hidden_packed.byte_size != 0U ||
      plan.prefill_a4_hidden_scales_bf16.byte_size != 0U ||
      plan.prefill_a4_intermediate_packed.byte_size != 0U ||
      plan.prefill_a4_intermediate_scales_bf16.byte_size != 0U ||
      plan.prefill_a4_gateup_cta_scratch.byte_size != 0U;
  const std::uint64_t a4_workspace_tokens =
      plan.long_prefill_projection_span_capacity == 0U
          ? plan.prefill_chunk_size
          : plan.long_prefill_projection_span_capacity;
  if (has_a4_workspace &&
      (!byte_region_at_least(
           plan.prefill_a4_hidden_packed,
           a4_workspace_tokens *
               kReferenceHiddenSize / 2U) ||
       !bf16_region_at_least(
           plan.prefill_a4_hidden_scales_bf16,
           a4_workspace_tokens *
               kReferenceHiddenSize / kRequestA4PrefillScaleGroupSize) ||
       !byte_region_at_least(
           plan.prefill_a4_intermediate_packed,
           a4_workspace_tokens *
               kReferenceIntermediateSize / 2U) ||
       !bf16_region_at_least(
           plan.prefill_a4_intermediate_scales_bf16,
           a4_workspace_tokens *
               kReferenceIntermediateSize /
               kRequestA4PrefillScaleGroupSize) ||
       !byte_region_at_least(plan.prefill_a4_gateup_cta_scratch,
                             kRequestA4GateUpCtaScratchBytes))) {
    return ReferenceRunnerError::kInvalidRequestState;
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
  projection_backend_ = std::exchange(
      other.projection_backend_, ProjectionBackend::kReference);
  a4w4_prefill_consumer_ = std::exchange(
      other.a4w4_prefill_consumer_,
      reference_runner_detail::A4W4PrefillConsumer::kUnavailable);
  a4w4_full_prefill_admission_enabled_ = std::exchange(
      other.a4w4_full_prefill_admission_enabled_, false);
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
  return ready;
}

std::uint32_t ReferenceRunner::current_position() const noexcept {
  return state_ == nullptr ? 0U : state_->current_position();
}

void ReferenceRunner::release() noexcept {
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
  prefill_gdn_chunk64_reference_context_ = nullptr;
  prefill_gdn_chunk64_reference_workspace_ = nullptr;
  prefill_gdn_chunk64_reference_workspace_bytes_ = 0U;
  prefill_gdn_chunk64_native_workspace_ = nullptr;
  prefill_gdn_chunk64_native_workspace_bytes_ = 0U;
  pinned_logits_ = nullptr;
  pinned_trace_ = nullptr;
  decode_graph_capture_active_ = false;
  views_ = {};
  projection_backend_ = ProjectionBackend::kReference;
  a4w4_prefill_consumer_ =
      reference_runner_detail::A4W4PrefillConsumer::kUnavailable;
  a4w4_full_prefill_admission_enabled_ = false;
  trace_enabled_ = false;
  trace_valid_ = false;
  poisoned_ = false;
  retained_prefill_hidden_valid_ = false;
  retained_prefill_position_ = 0U;
  retained_prefill_input_token_ = 0U;
  retained_prefill_hidden_row_ = 0U;
  trace_position_ = 0U;
  trace_input_token_ = 0U;
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
  ReferencePrefillTileOutcome outcome;
  outcome.status = status;
  return outcome;
}

ReferenceLongPrefillOutcome ReferenceRunner::fail_long_prefill(
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
  ReferenceLongPrefillOutcome outcome;
  outcome.status = status;
  return outcome;
}

std::optional<ReferenceTraceView> ReferenceRunner::last_trace() const noexcept {
  if (!trace_valid_ || pinned_trace_ == nullptr) {
    return std::nullopt;
  }
  return ReferenceTraceView{trace_position_, trace_input_token_, pinned_trace_,
                            kReferenceTraceElements};
}

ReferenceRunnerStatus ReferenceRunner::reset() noexcept {
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
  return {};
}

ReferenceStepOutcome ReferenceRunner::step(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  return step_impl(input_token_id, options,
                   DecodeGraphP1Action::kDisabled);
}

ReferenceDecodeGraphP1PrepareOutcome
ReferenceRunner::prepare_fixed_position_decode_graph_p1(
    const std::uint32_t input_token_id) noexcept {
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

ReferencePrefillTileOutcome ReferenceRunner::prefill_prefix_tile(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const ReferencePrefillTileOptions& options) noexcept {
  return prefill_prefix_tile_impl(input_token_ids, token_count, options,
                                  nullptr);
}

ReferencePrefillTileOutcome ReferenceRunner::prefill_prefix_tile_impl(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const ReferencePrefillTileOptions& options,
    const LongPrefillLayerTileInvocation* const layer_tile) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
  }
  // A new tile overwrites the prior layer-major workspace even if validation
  // later rejects the call, so stale retained rows must fail closed.
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
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  if (reference_runner_detail::select_a4w4_paired_gateup_canonical_down_route(
          a4w4_paired_gateup_canonical_down_selector_query(false)) !=
      reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
          kDisabled) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_paired_gateup_canonical_down_requires_projection_span"));
  }
  if (reference_runner_detail::
          select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
              a4w4_down_k512_m128n128_ldmatrix_pairring_v1_selector_query(
                  false)) !=
      reference_runner_detail::
          A4W4DownK512M128N128LdmatrixPairringV1Route::kDisabled) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_down_m128n128_ldmatrix_pairring_v1_requires_projection_span"));
  }
#endif
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
  if (token_count > state_->remaining_capacity() ||
      token_count > state_->plan().prefill_chunk_size) {
    return fail_prefill_tile(
        runner_status(ReferenceRunnerError::kCapacityExceeded,
                      "prefill_tile_capacity"));
  }

  if (layer_tile != nullptr) {
    const LongPrefillLayerMajorWorkItem& item = layer_tile->item;
    const std::uint64_t tile_end =
        static_cast<std::uint64_t>(item.first_position) + item.token_count;
    if (item.layer_index >= kReferenceDecoderLayerCount ||
        item.token_count != token_count || item.token_count == 0U ||
        item.input_hidden_buffer >=
            kRequestLongPrefillHiddenBufferCount ||
        item.output_hidden_buffer >=
            kRequestLongPrefillHiddenBufferCount ||
        layer_tile->input_hidden == nullptr ||
        layer_tile->output_hidden == nullptr ||
        tile_end > state_->plan().long_prefill_token_capacity ||
        item.layer_type !=
            reference_runner_detail::expected_reference_layer_type(
                item.layer_index)) {
      return fail_prefill_tile(
          runner_status(ReferenceRunnerError::kInvalidStepOptions,
                        "prefill_layer_major_tile_contract",
                        item.layer_index));
    }
  }

  bool a4w4_full_prefill_tile_enabled = false;
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  reference_runner_detail::A4W4FullPrefillAdmissionHits
      a4w4_full_prefill_tile_hits{};
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits
      a4w4_attention_supermatrix_tile_hits{};
#endif
  std::size_t a4w4_attention_o_k512_tile_hits = 0U;
  std::size_t a4w4_gateup_projection_v3_tile_hits = 0U;
  std::size_t a4w4_gateup_complete_cell_v2_tile_hits = 0U;
  std::size_t a4w4_down_complete_cell_v2_tile_hits = 0U;
  const bool a4w4_route_requested =
      reference_runner_detail::use_a4w4_full_prefill_tile_route(
          a4w4_full_prefill_admission_enabled_ ||
              g_enable_a4w4_full_prefill_admission,
          trace_enabled_, optimized_prefill_dispatch_disabled());
  if (a4w4_route_requested &&
      reference_runner_detail::a4w4_prefill_consumer_supports_token_count(
          a4w4_prefill_consumer_, token_count)) {
    const RequestMemoryPlan& plan = state_->plan();
    const std::size_t hidden_packed_capacity =
        kernels::sm87_a4w4_consumer_packed_capacity_bytes(
            kMaximumRequestPrefillChunkSize, kReferenceHiddenSize);
    const std::size_t hidden_scale_capacity =
        kernels::sm87_a4w4_consumer_scale_capacity_elements(
            kMaximumRequestPrefillChunkSize, kReferenceHiddenSize);
    const std::size_t intermediate_packed_capacity =
        kernels::sm87_a4w4_consumer_packed_capacity_bytes(
            kMaximumRequestPrefillChunkSize, kReferenceIntermediateSize);
    const std::size_t intermediate_scale_capacity =
        kernels::sm87_a4w4_consumer_scale_capacity_elements(
            kMaximumRequestPrefillChunkSize, kReferenceIntermediateSize);
    if (projection_backend_ != ProjectionBackend::kSm87WeightOnly) {
      return fail_prefill_tile(runner_status(
          ReferenceRunnerError::kInvalidDependency,
          "prefill_a4w4_projection_backend"));
    }
    if (plan.prefill_chunk_size != kMaximumRequestPrefillChunkSize ||
        views_.prefill_a4_hidden_packed == nullptr ||
        views_.prefill_a4_hidden_scales == nullptr ||
        views_.prefill_a4_intermediate_packed == nullptr ||
        views_.prefill_a4_intermediate_scales == nullptr ||
        plan.prefill_a4_hidden_packed.byte_size < hidden_packed_capacity ||
        plan.prefill_a4_hidden_scales_bf16.element_capacity <
            hidden_scale_capacity ||
        plan.prefill_a4_intermediate_packed.byte_size <
            intermediate_packed_capacity ||
        plan.prefill_a4_intermediate_scales_bf16.element_capacity <
            intermediate_scale_capacity ||
        reinterpret_cast<std::uintptr_t>(
            views_.prefill_a4_hidden_packed) % 16U != 0U ||
        reinterpret_cast<std::uintptr_t>(
            views_.prefill_a4_intermediate_packed) % 16U != 0U) {
      return fail_prefill_tile(runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_a4w4_workspace"));
    }
    if (a4w4_full_prefill_inventory_consumer(*weights_) !=
        a4w4_prefill_consumer_) {
      return fail_prefill_tile(runner_status(
          ReferenceRunnerError::kInvalidModelWeights,
          "prefill_a4w4_inventory"));
    }
    a4w4_full_prefill_tile_enabled = true;
  }
#endif

  if (token_count == 1U && layer_tile == nullptr &&
      !a4w4_full_prefill_tile_enabled) {
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

  const std::uint32_t first_position =
      layer_tile == nullptr ? state_->current_position()
                            : layer_tile->item.first_position;
#if defined(Q3X_ENABLE_GDN_B8_ADMISSION)
  // Snapshot the admission switch once at the tile boundary. Tests may only
  // change it between synchronous public runner calls.
  const bool enable_gdn_b8_admission =
      g_enable_prefill_gdn_b8_admission;
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  const bool enable_gdn_chunk64_native_admission =
      g_enable_prefill_gdn_chunk64_native_admission;
#endif
#if defined(Q3X_ENABLE_GDN_CHUNK64_REFERENCE_ADMISSION)
  const bool enable_gdn_chunk64_reference_admission =
      g_enable_prefill_gdn_chunk64_reference_admission;
#endif
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
  // Snapshot the private switch at the synchronous public-call boundary.
  const bool enable_gdn_c16_norm_gate_admission =
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
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  const auto a4w4_output_capacity_elements =
      [this](const std::uint16_t* const output) noexcept {
        const RequestMemoryPlan& plan = state_->plan();
        for (std::size_t index = 0U; index < 3U; ++index) {
          if (output == views_.hidden[index]) {
            return static_cast<std::size_t>(
                plan.hidden_bf16[index].element_capacity);
          }
        }
        for (std::size_t index = 0U; index < 4U; ++index) {
          if (output == views_.projection[index]) {
            return static_cast<std::size_t>(
                plan.projection_bf16[index].element_capacity);
          }
        }
        return std::size_t{0U};
      };
  const auto quantize_a4w4_activation =
      [this, token_count, &check_cuda,
       &a4w4_full_prefill_tile_hits](
          const std::uint16_t* const input,
          const std::size_t input_size, const float clip_ratio,
          std::uint8_t* const packed,
          const std::size_t packed_capacity_bytes,
          std::uint16_t* const scales,
          const std::size_t scale_capacity_elements,
          const char* const operation,
          const std::size_t layer) noexcept {
        const int status = launch_selected_a4w4_quantize(
            a4w4_prefill_consumer_, input, input_size, token_count,
            input_size, clip_ratio, packed, packed_capacity_bytes, scales,
            scale_capacity_elements, stream_);
        if (!check_cuda(status, operation, layer)) {
          return false;
        }
        ++a4w4_full_prefill_tile_hits.activation_quantize_hits;
        ++g_a4w4_full_prefill_admission_hits.activation_quantize_hits;
        return true;
      };
  const auto project_a4w4_from_packed =
      [this, token_count, &check_cuda, &a4w4_output_capacity_elements,
       &a4w4_full_prefill_tile_hits,
       &a4w4_down_complete_cell_v2_tile_hits](
          const LinearWeight& weight,
          const std::uint8_t* const packed_input,
          const std::size_t packed_input_capacity_bytes,
          const std::uint16_t* const input_scales,
          const std::size_t input_scale_capacity_elements,
          std::uint16_t* const output,
          const char* const operation,
          const std::size_t layer) noexcept {
        const PrefillA4LinearSidecarView sidecar =
            prefill_a4_sidecar_view(weight);
        const std::size_t weight_capacity_bytes =
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                sidecar.output_size, sidecar.input_size);
        const std::size_t weight_scale_capacity_elements =
            a4w4_scale_capacity_elements(
                a4w4_prefill_consumer_, sidecar.output_size,
                sidecar.input_size);
        bool down_complete_cell_v2_selected = false;
        const int status = launch_selected_a4w4_gemm(
            a4w4_prefill_consumer_, packed_input,
            packed_input_capacity_bytes, input_scales,
            input_scale_capacity_elements, sidecar, weight_capacity_bytes,
            weight_scale_capacity_elements, token_count, output,
            a4w4_output_capacity_elements(output), stream_,
            &down_complete_cell_v2_selected);
        if (!check_cuda(status, operation, layer)) {
          return false;
        }
        ++a4w4_full_prefill_tile_hits.generic_projection_hits;
        ++a4w4_full_prefill_tile_hits.logical_projection_hits;
        ++g_a4w4_full_prefill_admission_hits.generic_projection_hits;
        ++g_a4w4_full_prefill_admission_hits.logical_projection_hits;
        a4w4_down_complete_cell_v2_tile_hits +=
            down_complete_cell_v2_selected ? 1U : 0U;
        if (down_complete_cell_v2_selected) {
          return true;
        }
        const auto m128_route =
            reference_runner_detail::select_a4w4_k128_generic_prefill_route(
                g_enable_a4w4_m128_stage_major_admission,
                g_enable_a4w4_down_m128_stage_major_admission,
                a4w4_prefill_consumer_, token_count, sidecar.output_size,
                sidecar.input_size);
        if (m128_route ==
            reference_runner_detail::A4W4K128GenericPrefillRoute::
                kM128StageMajor) {
          ++a4w4_full_prefill_tile_hits
                .m128_stage_major_generic_projection_hits;
          ++g_a4w4_full_prefill_admission_hits
                .m128_stage_major_generic_projection_hits;
        } else if (m128_route ==
                   reference_runner_detail::A4W4K128GenericPrefillRoute::
                       kDownM128StageMajor) {
          ++a4w4_full_prefill_tile_hits
                .m128_stage_major_down_projection_hits;
          ++g_a4w4_full_prefill_admission_hits
                .m128_stage_major_down_projection_hits;
        }
        return true;
      };
#endif
  bool fp8_marlin_tile_enabled = false;
  std::int32_t* fp8_marlin_locks = nullptr;
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
  fp8_marlin_tile_enabled =
      g_enable_fp8_marlin_prefill_admission &&
      kernels::sm87_fp8_marlin_supports_token_count(token_count);
  fp8_marlin_locks =
      reinterpret_cast<std::int32_t*>(views_.projection[3]);
#endif
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
  const bool enable_bf16_ab_large_m_prefill_admission =
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
       fp8_marlin_locks](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const char* const operation,
          const std::size_t layer, void* const launch_stream) noexcept {
#if defined(Q3X_ENABLE_FP8_MARLIN_PREFILL_ADMISSION)
        if (is_fp8_marlin_projection(weight)) {
          const auto* const fp8 = std::get_if<Fp8LinearWeight>(&weight);
          if (fp8 == nullptr ||
              !reference_runner_detail::use_fp8_marlin_prefill_projection(
                  projection_backend_, weight, input, output, token_count) ||
              fp8_marlin_locks == nullptr ||
              views_.fp32_scratch_elements <
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
              fp8->input_size, output, views_.fp32_scratch,
              fp8_marlin_locks, launch_stream);
          if (status == static_cast<int>(cudaSuccess)) {
            ++g_fp8_marlin_prefill_admission_hits;
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
                              views_.fp32_scratch,
                              views_.fp32_scratch_elements,
                              output + token_offset * rows, launch_stream),
                          operation, layer)) {
            return false;
          }
        }
        return true;
      };
  const auto project = [this, &project_on_stream](
                           const LinearWeight& weight,
                           const std::uint16_t* const input,
                           std::uint16_t* const output,
                           const char* const operation,
                           const std::size_t layer) noexcept {
    return project_on_stream(weight, input, output, operation, layer, stream_);
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
      [this, token_count, &check_cuda, &project,
       &has_fp8_prefill_supermatrix_sidecar,
       &project_fp8_prefill_supermatrix, &is_fp8_marlin_projection](
          const LinearWeight& weight, const std::uint16_t* const input,
          std::uint16_t* const output, const char* const operation,
          const std::size_t layer) noexcept {
        if (is_fp8_marlin_projection(weight)) {
          return project(weight, input, output, operation, layer);
        }
        if (has_fp8_prefill_supermatrix_sidecar(weight)) {
          const LinearWeight* const group_weights[1U] = {&weight};
          std::uint16_t* const group_outputs[1U] = {output};
          return project_fp8_prefill_supermatrix(
              group_weights, group_outputs, 1U, input, operation, layer);
        }
        if (!reference_runner_detail::
                use_fp8_m64_prefill_attention_output_projection(
                    projection_backend_, weight, input, output,
                    token_count)) {
          return project(weight, input, output, operation, layer);
        }
        return check_cuda(launch_projection_tile_to_bf16_cuda(
                              projection_backend_, weight, input,
                              token_count, views_.fp32_scratch,
                              views_.fp32_scratch_elements, output, stream_),
                          operation, layer);
      };
  const auto project_pair =
      [this, token_count, &check_cuda,
       enable_bf16_ab_large_m_prefill_admission](
                                const LinearWeight& first_weight,
                                const LinearWeight& second_weight,
                                const std::uint16_t* const input,
                                std::uint16_t* const first_output,
                                std::uint16_t* const second_output,
                                const char* const operation,
                                const std::size_t layer) noexcept {
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
                          views_.fp32_scratch, views_.fp32_scratch_elements,
                          first_output + token_offset * first_rows,
                          second_output + token_offset * second_rows, stream_),
                      operation, layer)) {
        return false;
      }
    }
    return true;
  };
  const auto project_down = [this, token_count, &check_cuda,
                             &project](const LinearWeight& weight,
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
                          views_.fp32_scratch,
                          views_.fp32_scratch_elements, output, stream_),
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
      layer_tile == nullptr && residual_rms_m32_schedule.valid();

  if (layer_tile != nullptr) {
    if (!check_cuda(
            static_cast<int>(cudaMemcpyAsync(
                views_.hidden[0], layer_tile->input_hidden,
                token_count * kReferenceHiddenSize * sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice,
                static_cast<cudaStream_t>(stream_))),
            "prefill_layer_major_stage_input",
            layer_tile->item.layer_index)) {
      return fail_prefill_tile(launch_failure);
    }
  } else if (g_enable_prefill_embedding_prompt_wide_admission) {
    auto* const device_token_ids =
        reinterpret_cast<std::uint32_t*>(views_.projection[3]);
    if (!check_cuda(
            static_cast<int>(cudaMemcpyAsync(
                device_token_ids, input_token_ids,
                token_count * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
                static_cast<cudaStream_t>(stream_))),
            "prefill_embedding_token_ids", kReferenceNoLayer) ||
        !check_cuda(launch_embedding_gather_prompt_reference_cuda(
                        weights_->embed_tokens().weight,
                        kReferenceVocabularySize, kReferenceHiddenSize,
                        device_token_ids, token_count, views_.hidden[0],
                        stream_),
                    "prefill_embedding_gather_prompt_wide",
                    kReferenceNoLayer)) {
      return fail_prefill_tile(launch_failure);
    }
  } else {
    for (std::size_t token = 0U; token < token_count; ++token) {
      if (!check_cuda(launch_embedding_gather_reference_cuda(
                          weights_->embed_tokens().weight,
                          kReferenceVocabularySize, kReferenceHiddenSize,
                          input_token_ids[token],
                          views_.hidden[0] + token * kReferenceHiddenSize,
                          stream_),
                      "prefill_embedding_gather", kReferenceNoLayer)) {
        return fail_prefill_tile(launch_failure);
      }
    }
  }

  const std::size_t first_layer =
      layer_tile == nullptr ? 0U : layer_tile->item.layer_index;
  const std::size_t last_layer =
      layer_tile == nullptr ? kReferenceDecoderLayerCount : first_layer + 1U;
  for (std::size_t layer = first_layer; layer < last_layer; ++layer) {
    const DecoderLayerWeights& layer_weights = weights_->layer(layer);
    // The M32-prefix plus reference-tail layer-(N-1) MLP boundary already
    // produced layer N's normalized input. Layer 0 still normalizes the
    // embedding explicitly.
    if ((!use_m32_residual_rms_fusion || layer == 0U) &&
        !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                        views_.hidden[0],
                        layer_weights.input_layernorm.data, token_count,
                        kReferenceHiddenSize, kRmsEpsilon, views_.hidden[1],
                        stream_),
                    "prefill_input_layernorm", layer)) {
      return fail_prefill_tile(launch_failure);
    }

    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    if (expected == model::LayerType::kLinearAttention) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_prefill_tile(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "prefill_linear_attention_variant", layer));
      }
      bool linear_qkvz_projected = false;
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
      if (a4w4_full_prefill_tile_enabled) {
        const PrefillA4LinearSidecarView qkv_sidecar =
            prefill_a4_sidecar_view(attention->in_proj_qkv);
        [[maybe_unused]] const PrefillA4LinearSidecarView z_sidecar =
            prefill_a4_sidecar_view(attention->in_proj_z);
        const std::size_t hidden_packed_capacity =
            static_cast<std::size_t>(
                state_->plan().prefill_a4_hidden_packed.byte_size);
        const std::size_t hidden_scale_capacity =
            static_cast<std::size_t>(
                state_->plan()
                    .prefill_a4_hidden_scales_bf16.element_capacity);
        if (!quantize_a4w4_activation(
                views_.hidden[1], kReferenceHiddenSize,
                qkv_sidecar.activation_clip_ratio,
                views_.prefill_a4_hidden_packed,
                hidden_packed_capacity,
                views_.prefill_a4_hidden_scales,
                hidden_scale_capacity,
                "prefill_a4w4_linear_qkvz_quantize", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        bool attention_supermatrix_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
        if (!check_cuda(
                launch_selected_a4w4_linear_attention_supermatrix(
                    a4w4_prefill_consumer_,
                    views_.prefill_a4_hidden_packed,
                    hidden_packed_capacity,
                    views_.prefill_a4_hidden_scales,
                    hidden_scale_capacity, qkv_sidecar, z_sidecar,
                    token_count, views_.projection[0],
                    static_cast<std::size_t>(
                        state_->plan().projection_bf16[0U].element_capacity),
                    views_.projection[1],
                    static_cast<std::size_t>(
                        state_->plan().projection_bf16[1U].element_capacity),
                    stream_, a4w4_full_prefill_tile_hits,
                    a4w4_attention_supermatrix_tile_hits,
                    &attention_supermatrix_selected),
                "prefill_a4w4_linear_qkv_z_supermatrix", layer)) {
          return fail_prefill_tile(launch_failure);
        }
#endif
        if (!attention_supermatrix_selected &&
            (!project_a4w4_from_packed(
                 attention->in_proj_qkv,
                 views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                 views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                 views_.projection[0],
                 "prefill_a4w4_linear_qkv_projection", layer) ||
             !project_a4w4_from_packed(
                 attention->in_proj_z,
                 views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                 views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                 views_.projection[1],
                 "prefill_a4w4_linear_z_projection", layer))) {
          return fail_prefill_tile(launch_failure);
        }
        linear_qkvz_projected = true;
      }
#endif
      if (has_fp8_prefill_supermatrix_sidecar(attention->in_proj_qkv) &&
          has_fp8_prefill_supermatrix_sidecar(attention->in_proj_z) &&
          !linear_qkvz_projected) {
        const LinearWeight* const group_weights[2U] = {
            &attention->in_proj_qkv, &attention->in_proj_z};
        std::uint16_t* const group_outputs[2U] = {
            views_.projection[0], views_.projection[1]};
        if (!project_fp8_prefill_supermatrix(
                group_weights, group_outputs, 2U, views_.hidden[1],
                "prefill_linear_qkvz_supermatrix_projection", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        linear_qkvz_projected = true;
      }
      if (!linear_qkvz_projected &&
          (!project(attention->in_proj_qkv, views_.hidden[1],
                    views_.projection[0], "prefill_linear_qkv_projection",
                    layer) ||
           !project(attention->in_proj_z, views_.hidden[1],
                    views_.projection[1], "prefill_linear_z_projection",
                    layer))) {
        return fail_prefill_tile(launch_failure);
      }
      if (supports_bf16_projection_pair(
              projection_backend_, attention->in_proj_a,
              attention->in_proj_b)) {
        if (!project_pair(attention->in_proj_a, attention->in_proj_b,
                          views_.hidden[1], views_.linear_a, views_.linear_b,
                          "prefill_linear_a_b_projection", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      } else if (!project(attention->in_proj_a, views_.hidden[1],
                          views_.linear_a, "prefill_linear_a_projection",
                          layer) ||
                 !project(attention->in_proj_b, views_.hidden[1],
                          views_.linear_b, "prefill_linear_b_projection",
                          layer)) {
        return fail_prefill_tile(launch_failure);
      }
      const bool use_gdn_c16_norm_gate =
          should_use_prefill_gdn_c16_norm_gate(
              enable_gdn_c16_norm_gate_admission, projection_backend_,
              first_position, token_count);
      bool gdn_output_is_normalized = false;
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
      const bool use_gdn_chunk64_native =
          use_prefill_gdn_chunk64_native_admission(
              enable_gdn_chunk64_native_admission, projection_backend_,
              first_position, token_count,
              prefill_gdn_chunk64_native_workspace_,
              prefill_gdn_chunk64_native_workspace_bytes_);
      if (use_gdn_chunk64_native) {
        const bool use_token_parallel_conv =
            g_enable_gdn_conv_token_parallel_admission;
        // The fused producer writes compact Q/K for valid tokens and exact
        // zeros for the padded tail slots in the final C64 chunk.
        const bool use_conv_compact_qk_fused_candidate =
            g_enable_gdn_conv_compact_qk_fused_candidate &&
            use_token_parallel_conv;
        std::uint16_t* const conv_qkv =
            use_token_parallel_conv ? views_.projection[3]
                                    : views_.projection[0];
        // Convolution is exact for arbitrary C32..C512 admitted tiles and owns
        // a separate recurrent state. Execute it once over the complete
        // runner tile; the native hierarchy pads only its final C64 chunk.
        int conv_status = static_cast<int>(cudaErrorInvalidValue);
        if (use_conv_compact_qk_fused_candidate) {
          conv_status = gdn_prefill_chunk64_native_detail::
              launch_fused_conv_compact_qk_preprocess(
                  prefill_gdn_chunk64_native_workspace_,
                  prefill_gdn_chunk64_native_workspace_bytes_,
                  views_.projection[0], token_count,
                  attention->conv1d.data, views_.conv_state[layer],
                  conv_qkv, kRmsEpsilon, stream_);
        } else if (use_token_parallel_conv) {
          conv_status = gdn_prefill_whole_span_conv_detail::
              launch_causal_conv1d_silu_update_token_parallel_exact_cuda(
                  views_.projection[0], token_count,
                  attention->conv1d.data, views_.conv_state[layer],
                  conv_qkv, stream_);
        } else {
          conv_status = gdn_prefill_whole_span_conv_detail::
              launch_causal_conv1d_silu_update_whole_span_exact_cuda(
                  views_.projection[0], token_count,
                  attention->conv1d.data, views_.conv_state[layer],
                  conv_qkv, stream_);
        }
        if (!check_cuda(conv_status,
                        "prefill_linear_causal_conv_whole_span", layer)) {
          return fail_prefill_tile(launch_failure);
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
                    views_.linear_a, views_.linear_b, attention->a_log.data,
                    attention->dt_bias.data, views_.gdn_state[layer],
                    views_.gdn_state[layer], kRmsEpsilon,
                    attention->norm.data, views_.projection[1],
                    kRmsEpsilon, views_.projection[2], stream_)
                : gdn_prefill_chunk64_native_detail::launch(
                    prefill_gdn_chunk64_native_workspace_,
                    prefill_gdn_chunk64_native_workspace_bytes_,
                    conv_qkv, token_count,
                    views_.linear_a, views_.linear_b, attention->a_log.data,
                    attention->dt_bias.data, views_.gdn_state[layer],
                    views_.gdn_state[layer], kRmsEpsilon,
                    attention->norm.data, views_.projection[1],
                    kRmsEpsilon, views_.projection[2], stream_);
        if (!check_cuda(native_status, "prefill_linear_gdn_chunk64_native",
                        layer)) {
          return fail_prefill_tile(launch_failure);
        }
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
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, views_.conv_state[layer],
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
        ++g_prefill_gdn_chunk64_reference_admission_hits;
        if (!check_cuda(
                gdn_prefill_chunk64_reference_detail::launch(
                    prefill_gdn_chunk64_reference_context_,
                    prefill_gdn_chunk64_reference_workspace_,
                    prefill_gdn_chunk64_reference_workspace_bytes_,
                    token_count, views_.projection[0], views_.linear_a,
                    views_.linear_b,
                    attention->a_log.data, attention->dt_bias.data,
                    views_.gdn_state[layer], views_.gdn_state[layer],
                    kRmsEpsilon, attention->norm.data,
                    views_.projection[1], kRmsEpsilon,
                    views_.projection[2], stream_),
                "prefill_linear_gdn_chunk64_reference", layer)) {
          return fail_prefill_tile(launch_failure);
        }
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
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, views_.conv_state[layer],
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
        ++g_prefill_gdn_b8_admission_hits;
        if (!check_cuda(
                gdn_prefill_b8_detail::
                    launch_gated_delta_net_update_sequential_fp32_b8_exact_cuda(
                        views_.projection[0], token_count, views_.linear_a,
                        views_.linear_b, attention->a_log.data,
                        attention->dt_bias.data, views_.gdn_state[layer],
                        views_.gdn_state[layer], kRmsEpsilon,
                        views_.projection[2], stream_),
                "prefill_linear_gdn_b8_admission", layer)) {
          return fail_prefill_tile(launch_failure);
        }
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
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data, views_.conv_state[layer],
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer) ||
              !check_cuda(
                  gdn_prefill_c16_norm_gate_detail::
                      launch_shared_boundary(
                          views_.projection[0] +
                              token_offset * kLinearQkvElements,
                          kPrefillKernelTileMaximumTokens,
                          views_.linear_a +
                              token_offset * kLinearScalarElements,
                          views_.linear_b +
                              token_offset * kLinearScalarElements,
                          attention->a_log.data, attention->dt_bias.data,
                          views_.gdn_state[layer], views_.gdn_state[layer],
                          kRmsEpsilon, attention->norm.data,
                          views_.projection[1] +
                              token_offset * kLinearValueElements,
                          kRmsEpsilon,
                          views_.projection[2] +
                              token_offset * kLinearValueElements,
                          stream_),
                  "prefill_linear_gdn_c16_norm_gate", layer)) {
            return fail_prefill_tile(launch_failure);
          }
#if defined(Q3X_ENABLE_GDN_C16_NORM_GATE_ADMISSION)
          ++g_prefill_gdn_c16_norm_gate_admission_hits;
#endif
        }
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
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      subtile_tokens, attention->conv1d.data,
                      views_.conv_state[layer],
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      {}, stream_),
                  "prefill_linear_causal_conv", layer) ||
              !check_cuda(
                  launch_gated_delta_net_update_tile_warp_parallel_cuda(
                      views_.projection[0] +
                          token_offset * kLinearQkvElements,
                      subtile_tokens,
                      views_.linear_a +
                          token_offset * kLinearScalarElements,
                      views_.linear_b +
                          token_offset * kLinearScalarElements,
                      attention->a_log.data, attention->dt_bias.data,
                      views_.gdn_state[layer], views_.gdn_state[layer],
                      kRmsEpsilon,
                      views_.projection[2] +
                          token_offset * kLinearValueElements,
                      {}, stream_),
                  "prefill_linear_gdn", layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
      }
      if (!gdn_output_is_normalized &&
          !check_cuda(
              launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                  views_.projection[2], attention->norm.data,
                  views_.projection[1], token_count * kGdnValueHeadCount,
                  kGdnHeadDimension, kRmsEpsilon, views_.projection[2],
                  stream_),
              "prefill_linear_output_norm_gate", layer)) {
        return fail_prefill_tile(launch_failure);
      }
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
      if (a4w4_full_prefill_tile_enabled) {
        const PrefillA4LinearSidecarView output_sidecar =
            prefill_a4_sidecar_view(attention->out_proj);
        const std::size_t intermediate_packed_capacity =
            static_cast<std::size_t>(
                state_->plan().prefill_a4_intermediate_packed.byte_size);
        const std::size_t intermediate_scale_capacity =
            static_cast<std::size_t>(
                state_->plan()
                    .prefill_a4_intermediate_scales_bf16.element_capacity);
        bool attention_o_k512_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
        if (!check_cuda(
                launch_selected_a4w4_attention_o_k512(
                    a4w4_prefill_consumer_, attention->out_proj,
                    views_.projection[2], kLinearValueElements, token_count,
                    kernels::sm87_a4w4_prefill_k512_launch_token_count(
                        token_count),
                    views_.prefill_a4_intermediate_packed,
                    intermediate_packed_capacity,
                    views_.prefill_a4_intermediate_scales,
                    intermediate_scale_capacity, views_.hidden[1],
                    kReferenceHiddenSize,
                    static_cast<std::size_t>(
                        state_->plan().hidden_bf16[1U].element_capacity),
                    stream_, &attention_o_k512_selected),
                "prefill_a4w4_linear_output_k512", layer)) {
          return fail_prefill_tile(launch_failure);
        }
#endif
        if (attention_o_k512_selected) {
          ++a4w4_attention_o_k512_tile_hits;
          ++a4w4_full_prefill_tile_hits.activation_quantize_hits;
          ++a4w4_full_prefill_tile_hits.logical_projection_hits;
          ++g_a4w4_full_prefill_admission_hits.activation_quantize_hits;
          ++g_a4w4_full_prefill_admission_hits.logical_projection_hits;
        }
        if (!attention_o_k512_selected &&
            !quantize_a4w4_activation(
                views_.projection[2], kLinearValueElements,
                output_sidecar.activation_clip_ratio,
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity,
                "prefill_a4w4_linear_output_quantize", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        bool attention_supermatrix_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
        if (!attention_o_k512_selected && !check_cuda(
                launch_selected_a4w4_attention_o_supermatrix(
                    a4w4_prefill_consumer_,
                    views_.prefill_a4_intermediate_packed,
                    intermediate_packed_capacity,
                    views_.prefill_a4_intermediate_scales,
                    intermediate_scale_capacity, output_sidecar, token_count,
                    views_.hidden[1],
                    static_cast<std::size_t>(
                        state_->plan().hidden_bf16[1U].element_capacity),
                    stream_, a4w4_full_prefill_tile_hits,
                    a4w4_attention_supermatrix_tile_hits,
                    &attention_supermatrix_selected),
                "prefill_a4w4_linear_output_supermatrix", layer)) {
          return fail_prefill_tile(launch_failure);
        }
#endif
        if (!attention_o_k512_selected &&
            !attention_supermatrix_selected &&
            !project_a4w4_from_packed(
                attention->out_proj,
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity, views_.hidden[1],
                "prefill_a4w4_linear_output_projection", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      } else
#endif
      if (!project_attention_output(
              attention->out_proj, views_.projection[2], views_.hidden[1],
              "prefill_linear_output_projection", layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else if (expected == model::LayerType::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      if (attention == nullptr) {
        return fail_prefill_tile(runner_status(
            ReferenceRunnerError::kInvalidLayerSchedule,
            "prefill_full_attention_variant", layer));
      }
      std::uint16_t* const packed_gates =
          views_.projection[3] + token_count * kFullQueryElements;
      std::uint16_t* const tile_key =
          views_.key_cache[layer] +
          static_cast<std::size_t>(first_position) * kFullKvElements;
      std::uint16_t* const tile_value =
          views_.value_cache[layer] +
          static_cast<std::size_t>(first_position) * kFullKvElements;
      const std::size_t rope_first_position =
          static_cast<std::size_t>(first_position);
      bool full_qkv_projected = false;
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
      if (a4w4_full_prefill_tile_enabled) {
        const PrefillA4LinearSidecarView q_sidecar =
            prefill_a4_sidecar_view(attention->q_proj);
        [[maybe_unused]] const PrefillA4LinearSidecarView k_sidecar =
            prefill_a4_sidecar_view(attention->k_proj);
        [[maybe_unused]] const PrefillA4LinearSidecarView v_sidecar =
            prefill_a4_sidecar_view(attention->v_proj);
        const std::size_t hidden_packed_capacity =
            static_cast<std::size_t>(
                state_->plan().prefill_a4_hidden_packed.byte_size);
        const std::size_t hidden_scale_capacity =
            static_cast<std::size_t>(
                state_->plan()
                    .prefill_a4_hidden_scales_bf16.element_capacity);
        if (!quantize_a4w4_activation(
                views_.hidden[1], kReferenceHiddenSize,
                q_sidecar.activation_clip_ratio,
                views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                "prefill_a4w4_full_qkv_quantize", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        bool attention_supermatrix_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
        if (!check_cuda(
                launch_selected_a4w4_full_attention_supermatrix(
                    a4w4_prefill_consumer_,
                    views_.prefill_a4_hidden_packed,
                    hidden_packed_capacity,
                    views_.prefill_a4_hidden_scales,
                    hidden_scale_capacity, q_sidecar, k_sidecar, v_sidecar,
                    token_count, views_.projection[0],
                    static_cast<std::size_t>(
                        state_->plan().projection_bf16[0U].element_capacity),
                    tile_key, token_count * kFullKvElements, tile_value,
                    token_count * kFullKvElements, stream_,
                    a4w4_full_prefill_tile_hits,
                    a4w4_attention_supermatrix_tile_hits,
                    &attention_supermatrix_selected),
                "prefill_a4w4_full_q_k_v_supermatrix", layer)) {
          return fail_prefill_tile(launch_failure);
        }
#endif
        if (!attention_supermatrix_selected &&
            (!project_a4w4_from_packed(
                 attention->q_proj, views_.prefill_a4_hidden_packed,
                 hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                 hidden_scale_capacity, views_.projection[0],
                 "prefill_a4w4_full_q_gate_projection", layer) ||
             !project_a4w4_from_packed(
                 attention->k_proj, views_.prefill_a4_hidden_packed,
                 hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                 hidden_scale_capacity, tile_key,
                 "prefill_a4w4_full_k_projection", layer) ||
             !project_a4w4_from_packed(
                 attention->v_proj, views_.prefill_a4_hidden_packed,
                 hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                 hidden_scale_capacity, tile_value,
                 "prefill_a4w4_full_v_projection", layer))) {
          return fail_prefill_tile(launch_failure);
        }
        full_qkv_projected = true;
      }
#endif
      if (has_fp8_prefill_supermatrix_sidecar(attention->q_proj) &&
          has_fp8_prefill_supermatrix_sidecar(attention->k_proj) &&
          has_fp8_prefill_supermatrix_sidecar(attention->v_proj) &&
          !full_qkv_projected) {
        const LinearWeight* const group_weights[3U] = {
            &attention->q_proj, &attention->k_proj, &attention->v_proj};
        std::uint16_t* const group_outputs[3U] = {
            views_.projection[0], tile_key, tile_value};
        if (!project_fp8_prefill_supermatrix(
                group_weights, group_outputs, 3U, views_.hidden[1],
                "prefill_full_qkv_supermatrix_projection", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        full_qkv_projected = true;
      }
      if (!full_qkv_projected &&
          (!project(attention->q_proj, views_.hidden[1],
                    views_.projection[0], "prefill_full_q_gate_projection",
                    layer) ||
           !project(attention->k_proj, views_.hidden[1], tile_key,
                    "prefill_full_k_projection", layer) ||
           !project(attention->v_proj, views_.hidden[1], tile_value,
                    "prefill_full_v_projection", layer))) {
        return fail_prefill_tile(launch_failure);
      }

      const std::size_t preprocess_tile_maximum =
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
            g_enable_full_attention_preprocess_prompt_wide_admission
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
          use_fused_preprocess ? views_.projection[3]
                               : views_.projection[0];
      for (std::size_t token_offset = 0U; token_offset < token_count;
           token_offset += preprocess_tile_maximum) {
        const std::size_t remaining = token_count - token_offset;
        const std::size_t subtile_tokens =
            remaining < preprocess_tile_maximum
                ? remaining
                : preprocess_tile_maximum;
        std::uint16_t* const raw_query_gate =
            views_.projection[0] + token_offset * kFullQGateElements;
        std::uint16_t* const subtile_query =
            tile_query + token_offset * kFullQueryElements;
        std::uint16_t* const split_query =
            views_.projection[3] + token_offset * kFullQueryElements;
        std::uint16_t* const subtile_gates =
            packed_gates + token_offset * kFullQueryElements;
        std::uint16_t* const subtile_key =
            tile_key + token_offset * kFullKvElements;
        const std::size_t subtile_first_position =
            rope_first_position + token_offset;
        if (use_fused_preprocess) {
          if (!check_cuda(
                  launch_full_attention_preprocess_24_4_256_64_cuda(
                      raw_query_gate, subtile_key, attention->q_norm.data,
                      attention->k_norm.data, kRmsEpsilon, subtile_query,
                      subtile_gates, views_.rope_cos, views_.rope_sin,
                      subtile_first_position, subtile_tokens, stream_),
                  "prefill_full_preprocess", layer)) {
            return fail_prefill_tile(launch_failure);
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
          return fail_prefill_tile(launch_failure);
        }
        for (std::size_t token = 0U; token < subtile_tokens; ++token) {
          const std::size_t position = subtile_first_position + token;
          const float* const cosines =
              views_.rope_cos + position * kRopePairs;
          const float* const sines =
              views_.rope_sin + position * kRopePairs;
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
            return fail_prefill_tile(launch_failure);
          }
        }
      }
      const bool use_bulk_gqa_gate =
          reference_runner_detail::
              use_bulk_causal_gqa_sigmoid_gate_prefill(
                  projection_backend_, expected, first_position,
                  token_count);
      if (use_bulk_gqa_gate) {
        if (!check_cuda(
                launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
                    tile_query, views_.key_cache[layer],
                    views_.value_cache[layer], packed_gates,
                    first_position, token_count, views_.projection[1],
                    stream_),
                "prefill_full_bulk_gqa_output_gate", layer)) {
          return fail_prefill_tile(launch_failure);
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
              views_.projection[1] + token * kFullQueryElements;
          const bool fuse_gqa_gate_token =
              token < fused_gqa_gate_prefix_tokens;
          const int gqa_status =
              fuse_gqa_gate_token
                  ? launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                        token_query, views_.key_cache[layer],
                        views_.value_cache[layer], sequence_length,
                        kAttentionScale, views_.fp32_scratch,
                        views_.fp32_scratch_elements,
                        packed_gates + token * kFullQueryElements,
                        token_output, stream_)
                  : launch_gqa_attention_reference_cuda(
                        token_query, views_.key_cache[layer],
                        views_.value_cache[layer], kFullQueryHeads,
                        kFullKvHeads, sequence_length, kFullHeadDimension,
                        kAttentionScale, views_.fp32_scratch,
                        views_.fp32_scratch_elements, token_output, stream_);
          if (!check_cuda(
                  gqa_status,
                  fuse_gqa_gate_token ? "prefill_full_gqa_output_gate"
                                      : "prefill_full_gqa",
                  layer)) {
            return fail_prefill_tile(launch_failure);
          }
        }
        if (fused_gqa_gate_prefix_tokens < token_count &&
            !check_cuda(launch_sigmoid_gate_reference_cuda(
                            views_.projection[1] +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            packed_gates +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            (token_count - fused_gqa_gate_prefix_tokens) *
                                kFullQueryElements,
                            views_.projection[1] +
                                fused_gqa_gate_prefix_tokens *
                                    kFullQueryElements,
                            stream_),
                        "prefill_full_output_gate", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      }
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
      if (a4w4_full_prefill_tile_enabled) {
        const PrefillA4LinearSidecarView output_sidecar =
            prefill_a4_sidecar_view(attention->o_proj);
        const std::size_t intermediate_packed_capacity =
            static_cast<std::size_t>(
                state_->plan().prefill_a4_intermediate_packed.byte_size);
        const std::size_t intermediate_scale_capacity =
            static_cast<std::size_t>(
                state_->plan()
                    .prefill_a4_intermediate_scales_bf16.element_capacity);
        bool attention_o_k512_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
        if (!check_cuda(
                launch_selected_a4w4_attention_o_k512(
                    a4w4_prefill_consumer_, attention->o_proj,
                    views_.projection[1], kFullQueryElements, token_count,
                    kernels::sm87_a4w4_prefill_k512_launch_token_count(
                        token_count),
                    views_.prefill_a4_intermediate_packed,
                    intermediate_packed_capacity,
                    views_.prefill_a4_intermediate_scales,
                    intermediate_scale_capacity, views_.hidden[1],
                    kReferenceHiddenSize,
                    static_cast<std::size_t>(
                        state_->plan().hidden_bf16[1U].element_capacity),
                    stream_, &attention_o_k512_selected),
                "prefill_a4w4_full_output_k512", layer)) {
          return fail_prefill_tile(launch_failure);
        }
#endif
        if (attention_o_k512_selected) {
          ++a4w4_attention_o_k512_tile_hits;
          ++a4w4_full_prefill_tile_hits.activation_quantize_hits;
          ++a4w4_full_prefill_tile_hits.logical_projection_hits;
          ++g_a4w4_full_prefill_admission_hits.activation_quantize_hits;
          ++g_a4w4_full_prefill_admission_hits.logical_projection_hits;
        }
        if (!attention_o_k512_selected &&
            !quantize_a4w4_activation(
                views_.projection[1], kFullQueryElements,
                output_sidecar.activation_clip_ratio,
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity,
                "prefill_a4w4_full_output_quantize", layer)) {
          return fail_prefill_tile(launch_failure);
        }
        bool attention_supermatrix_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
        if (!attention_o_k512_selected && !check_cuda(
                launch_selected_a4w4_attention_o_supermatrix(
                    a4w4_prefill_consumer_,
                    views_.prefill_a4_intermediate_packed,
                    intermediate_packed_capacity,
                    views_.prefill_a4_intermediate_scales,
                    intermediate_scale_capacity, output_sidecar, token_count,
                    views_.hidden[1],
                    static_cast<std::size_t>(
                        state_->plan().hidden_bf16[1U].element_capacity),
                    stream_, a4w4_full_prefill_tile_hits,
                    a4w4_attention_supermatrix_tile_hits,
                    &attention_supermatrix_selected),
                "prefill_a4w4_full_output_supermatrix", layer)) {
          return fail_prefill_tile(launch_failure);
        }
#endif
        if (!attention_o_k512_selected &&
            !attention_supermatrix_selected &&
            !project_a4w4_from_packed(
                attention->o_proj,
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity, views_.hidden[1],
                "prefill_a4w4_full_output_projection", layer)) {
          return fail_prefill_tile(launch_failure);
        }
      } else
#endif
      if (!project_attention_output(
              attention->o_proj, views_.projection[1], views_.hidden[1],
              "prefill_full_output_projection", layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else {
      return fail_prefill_tile(runner_status(
          ReferenceRunnerError::kInvalidLayerSchedule,
          "prefill_layer_schedule", layer));
    }

    if (use_m32_residual_rms_fusion) {
      if (!residual_norm_m32_tiles(
              views_.hidden[0], views_.hidden[1],
              layer_weights.post_attention_layernorm.data, views_.hidden[2],
              views_.hidden[1],
              "prefill_attention_residual_post_attention_layernorm", layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else {
      if (!check_cuda(launch_residual_add_reference_cuda(
                          views_.hidden[0], views_.hidden[1],
                          token_count * kReferenceHiddenSize,
                          views_.hidden[2], stream_),
                      "prefill_attention_residual", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      if (!check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                          views_.hidden[2],
                          layer_weights.post_attention_layernorm.data,
                          token_count, kReferenceHiddenSize, kRmsEpsilon,
                          views_.hidden[1], stream_),
                      "prefill_post_attention_layernorm", layer)) {
        return fail_prefill_tile(launch_failure);
      }
    }
    bool a4w4_mlp_completed = false;
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
    if (a4w4_full_prefill_tile_enabled) {
      const PrefillA4LinearSidecarView gate_sidecar =
          prefill_a4_sidecar_view(layer_weights.mlp.gate_proj);
      const PrefillA4LinearSidecarView up_sidecar =
          prefill_a4_sidecar_view(layer_weights.mlp.up_proj);
      const PrefillA4LinearSidecarView down_sidecar =
          prefill_a4_sidecar_view(layer_weights.mlp.down_proj);
      const std::size_t hidden_packed_capacity =
          static_cast<std::size_t>(
              state_->plan().prefill_a4_hidden_packed.byte_size);
      const std::size_t hidden_scale_capacity =
          static_cast<std::size_t>(
              state_->plan()
                  .prefill_a4_hidden_scales_bf16.element_capacity);
      const std::size_t intermediate_packed_capacity =
          static_cast<std::size_t>(
              state_->plan().prefill_a4_intermediate_packed.byte_size);
      const std::size_t intermediate_scale_capacity =
          static_cast<std::size_t>(
              state_->plan()
                  .prefill_a4_intermediate_scales_bf16.element_capacity);
      const std::size_t gate_weight_capacity =
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(
              gate_sidecar.output_size, gate_sidecar.input_size);
      const std::size_t gate_scale_capacity =
          a4w4_scale_capacity_elements(
              a4w4_prefill_consumer_, gate_sidecar.output_size,
              gate_sidecar.input_size);
      const std::size_t up_weight_capacity =
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(
              up_sidecar.output_size, up_sidecar.input_size);
      const std::size_t up_scale_capacity =
          a4w4_scale_capacity_elements(
              a4w4_prefill_consumer_, up_sidecar.output_size,
              up_sidecar.input_size);
      if (!quantize_a4w4_activation(
              views_.hidden[1], kReferenceHiddenSize,
              gate_sidecar.activation_clip_ratio,
              views_.prefill_a4_hidden_packed, hidden_packed_capacity,
              views_.prefill_a4_hidden_scales, hidden_scale_capacity,
              "prefill_a4w4_mlp_gate_up_quantize", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      bool projection_v3_selected = false;
      bool complete_cell_v2_selected = false;
      const int paired_status = launch_selected_a4w4_gateup_paired(
          a4w4_prefill_consumer_, views_.prefill_a4_hidden_packed,
          hidden_packed_capacity, views_.prefill_a4_hidden_scales,
          hidden_scale_capacity, gate_sidecar, gate_weight_capacity,
          gate_scale_capacity, up_sidecar, up_weight_capacity,
          up_scale_capacity, token_count,
          down_sidecar.activation_clip_ratio,
          views_.prefill_a4_intermediate_packed,
          intermediate_packed_capacity,
          views_.prefill_a4_intermediate_scales,
          intermediate_scale_capacity, stream_,
          &projection_v3_selected, &complete_cell_v2_selected);
      if (!check_cuda(paired_status, "prefill_a4w4_mlp_gate_up_paired",
                      layer)) {
        return fail_prefill_tile(launch_failure);
      }
      ++a4w4_full_prefill_tile_hits.paired_gate_up_hits;
      a4w4_full_prefill_tile_hits.logical_projection_hits += 2U;
      ++g_a4w4_full_prefill_admission_hits.paired_gate_up_hits;
      g_a4w4_full_prefill_admission_hits.logical_projection_hits += 2U;
      a4w4_gateup_projection_v3_tile_hits +=
          projection_v3_selected ? 1U : 0U;
      a4w4_gateup_complete_cell_v2_tile_hits +=
          complete_cell_v2_selected ? 1U : 0U;
      if (!projection_v3_selected && !complete_cell_v2_selected &&
          reference_runner_detail::a4w4_m128_stage_major_common_route(
              g_enable_a4w4_m128_stage_major_admission,
              a4w4_prefill_consumer_, token_count)) {
        ++a4w4_full_prefill_tile_hits
              .m128_stage_major_paired_gate_up_hits;
        ++g_a4w4_full_prefill_admission_hits
              .m128_stage_major_paired_gate_up_hits;
      }
      if (!project_a4w4_from_packed(
              layer_weights.mlp.down_proj,
              views_.prefill_a4_intermediate_packed,
              intermediate_packed_capacity,
              views_.prefill_a4_intermediate_scales,
              intermediate_scale_capacity, views_.hidden[1],
              "prefill_a4w4_mlp_down_projection", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      a4w4_mlp_completed = true;
    }
#endif
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
        !a4w4_mlp_completed &&
        g_enable_nvfp4_marlin_prefill_admission &&
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
        views_.projection[1] ==
            views_.projection[0] + marlin_workspace_branch_elements &&
        marlin_branch_elements <= marlin_workspace_branch_elements &&
        views_.fp32_scratch_elements >=
            kernels::kSm87NvFp4MarlinReductionElements;
    if (use_marlin_mlp) {
      auto* const locks =
          reinterpret_cast<std::int32_t*>(views_.projection[3]);
      const auto stream = reinterpret_cast<cudaStream_t>(stream_);
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
                              views_.hidden[1],
                              marlin_gate->prefill_marlin_weight,
                              marlin_gate->prefill_marlin_scales,
                              marlin_gate->prefill_marlin_global_scale,
                              token_count, views_.projection[0],
                              views_.projection[2], views_.fp32_scratch, locks,
                              stream_),
                      "prefill_marlin_gate_up_epilogue", layer)
                : (check_cuda(
                       kernels::launch_sm87_nvfp4_marlin_gate_up_cuda(
                           views_.hidden[1],
                           marlin_gate->prefill_marlin_weight,
                           marlin_gate->prefill_marlin_scales,
                           marlin_gate->prefill_marlin_global_scale,
                           token_count, views_.projection[0],
                           views_.fp32_scratch, locks, stream_),
                       "prefill_marlin_gate_up", layer) &&
                   check_cuda(
                       kernels::launch_sm87_nvfp4_marlin_gate_up_silu_cuda(
                           views_.projection[0], token_count,
                           views_.projection[2], stream_),
                       "prefill_marlin_gate_up_silu", layer))) ||
          !check_cuda(
              kernels::launch_sm87_nvfp4_marlin_down_cuda(
                  views_.projection[2], marlin_down->prefill_marlin_weight,
                  marlin_down->prefill_marlin_scales,
                  marlin_down->prefill_marlin_global_scale, token_count,
                  views_.hidden[1], views_.fp32_scratch, locks, stream_),
              "prefill_marlin_down", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      ++g_nvfp4_marlin_prefill_admission_hits;
      marlin_mlp_completed = true;
    }
#endif

    if (!a4w4_mlp_completed && !marlin_mlp_completed) {
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
                  layer_weights.mlp.up_proj, views_.hidden[1],
                  views_.projection[0], views_.projection[1], token_count);
      const bool use_gate_up_m32_dual_stream =
          gate_up_fork_join_available &&
          reference_runner_detail::use_nvfp4_m32_prefill_gate_up_dual_stream(
              projection_backend_, layer_weights.mlp.gate_proj,
              layer_weights.mlp.up_proj, views_.hidden[1],
              views_.projection[0], views_.projection[1], token_count);
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
                layer_weights.mlp.gate_proj, views_.hidden[1],
                views_.projection[0], "prefill_mlp_gate_projection", layer,
                stream_);
          }
          return project(layer_weights.mlp.gate_proj, views_.hidden[1],
                         views_.projection[0], "prefill_mlp_gate_projection",
                         layer);
        };
        const auto project_up = [&]() noexcept {
          if (use_gate_up_whole_chunk_dual_stream) {
            return project_nvfp4_whole_chunk_on_stream(
                layer_weights.mlp.up_proj, views_.hidden[1],
                views_.projection[1], "prefill_mlp_up_projection_auxiliary",
                layer, prefill_auxiliary_stream_);
          }
          return project_on_stream(
              layer_weights.mlp.up_proj, views_.hidden[1],
              views_.projection[1], "prefill_mlp_up_projection_auxiliary",
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
          return fail_prefill_tile(launch_failure);
        }
      } else if (!project(layer_weights.mlp.gate_proj, views_.hidden[1],
                          views_.projection[0],
                          "prefill_mlp_gate_projection", layer) ||
                 !project(layer_weights.mlp.up_proj, views_.hidden[1],
                          views_.projection[1],
                          "prefill_mlp_up_projection", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      }
      if (!check_cuda(launch_silu_mul_reference_cuda(
                        views_.projection[0], views_.projection[1],
                        token_count * kReferenceIntermediateSize,
                        views_.projection[0], stream_),
                    "prefill_mlp_silu_mul", layer)) {
        return fail_prefill_tile(launch_failure);
      }
      if (!project_down(layer_weights.mlp.down_proj, views_.projection[0],
                        views_.hidden[1], "prefill_mlp_down_projection",
                        layer)) {
        return fail_prefill_tile(launch_failure);
      }
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
              views_.hidden[2], views_.hidden[1], next_norm_weight,
              views_.hidden[0], views_.hidden[1], operation, layer)) {
        return fail_prefill_tile(launch_failure);
      }
    } else if (!check_cuda(launch_residual_add_reference_cuda(
                               views_.hidden[2], views_.hidden[1],
                               token_count * kReferenceHiddenSize,
                               views_.hidden[0], stream_),
                           "prefill_layer_residual", layer)) {
      return fail_prefill_tile(launch_failure);
    }
  }

#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  if (a4w4_full_prefill_tile_enabled) {
    std::size_t expected_quantize_hits = 192U;
    std::size_t expected_generic_hits = 272U;
    std::size_t expected_paired_hits = 64U;
    std::size_t expected_logical_hits = 400U;
    if (layer_tile != nullptr) {
      const bool linear =
          layer_tile->item.layer_type == model::LayerType::kLinearAttention;
      expected_quantize_hits = 3U;
      expected_generic_hits = linear ? 4U : 5U;
      expected_paired_hits = 1U;
      expected_logical_hits = linear ? 6U : 7U;
    }
    const bool linear_layer =
        layer_tile != nullptr &&
        layer_tile->item.layer_type == model::LayerType::kLinearAttention;
    const std::size_t expected_attention_linear_hits =
        layer_tile == nullptr ? 48U : (linear_layer ? 1U : 0U);
    const std::size_t expected_attention_full_hits =
        layer_tile == nullptr ? 16U : (linear_layer ? 0U : 1U);
    const std::size_t expected_attention_output_hits =
        layer_tile == nullptr ? 64U : 1U;
    const std::size_t expected_attention_input_logical_hits =
        2U * expected_attention_linear_hits +
        3U * expected_attention_full_hits;
    std::size_t observed_attention_linear_hits = 0U;
    std::size_t observed_attention_full_hits = 0U;
    std::size_t observed_attention_output_hits = 0U;
    std::size_t observed_attention_logical_hits = 0U;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    observed_attention_linear_hits =
        a4w4_attention_supermatrix_tile_hits.linear_input_launch_hits;
    observed_attention_full_hits =
        a4w4_attention_supermatrix_tile_hits.full_input_launch_hits;
    observed_attention_output_hits =
        a4w4_attention_supermatrix_tile_hits.output_launch_hits;
    observed_attention_logical_hits =
        a4w4_attention_supermatrix_tile_hits.logical_projection_hits;
#endif
    const bool attention_supermatrix_inputs_all =
        observed_attention_linear_hits == expected_attention_linear_hits &&
        observed_attention_full_hits == expected_attention_full_hits &&
        observed_attention_logical_hits ==
            expected_attention_input_logical_hits +
                observed_attention_output_hits;
    const bool attention_supermatrix_inputs_none =
        observed_attention_linear_hits == 0U &&
        observed_attention_full_hits == 0U &&
        observed_attention_logical_hits == observed_attention_output_hits;
    const bool attention_supermatrix_output_all =
        observed_attention_output_hits == expected_attention_output_hits;
    const bool attention_supermatrix_output_none =
        observed_attention_output_hits == 0U;
    const bool attention_o_k512_all =
        a4w4_attention_o_k512_tile_hits == expected_attention_output_hits;
    const bool attention_o_k512_none =
        a4w4_attention_o_k512_tile_hits == 0U;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    const bool requested_attention_supermatrix_selected =
        !g_enable_a4w4_attention_supermatrix_admission ||
        (attention_supermatrix_inputs_all &&
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
         (g_enable_a4w4_attention_o_k512_admission
              ? attention_o_k512_all
              : attention_supermatrix_output_all));
#else
         attention_supermatrix_output_all);
#endif
#else
    constexpr bool requested_attention_supermatrix_selected = true;
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
    const bool requested_attention_o_k512_selected =
        !g_enable_a4w4_attention_o_k512_admission ||
        attention_o_k512_all;
#else
    constexpr bool requested_attention_o_k512_selected = true;
#endif
    if (attention_supermatrix_inputs_all) {
      expected_generic_hits -= expected_attention_input_logical_hits;
    }
    if (attention_supermatrix_output_all || attention_o_k512_all) {
      expected_generic_hits -= expected_attention_output_hits;
    }
    const bool expect_m128_stage_major =
        reference_runner_detail::a4w4_m128_stage_major_common_route(
            g_enable_a4w4_m128_stage_major_admission,
            a4w4_prefill_consumer_, token_count);
    const bool expect_down_m128_stage_major =
        reference_runner_detail::a4w4_m128_stage_major_common_route(
            g_enable_a4w4_down_m128_stage_major_admission,
            a4w4_prefill_consumer_, token_count);
    bool expect_down_complete_cell_v2 = false;
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION) || \
    defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
    reference_runner_detail::A4W4DownCompleteCellV2RouteQuery down_cell_query;
    down_cell_query.inventory_consumer = a4w4_prefill_consumer_;
    down_cell_query.projection_token_count = token_count;
    down_cell_query.output_size = kReferenceHiddenSize;
    down_cell_query.input_size = kReferenceIntermediateSize;
    down_cell_query.packed_input_capacity_bytes =
        static_cast<std::size_t>(
            state_->plan().prefill_a4_intermediate_packed.byte_size);
    down_cell_query.input_scale_capacity_elements =
        static_cast<std::size_t>(state_->plan()
                                     .prefill_a4_intermediate_scales_bf16
                                     .element_capacity);
    down_cell_query.weight_capacity_bytes =
        kernels::sm87_a4w4_consumer_packed_capacity_bytes(
            kReferenceHiddenSize, kReferenceIntermediateSize);
    down_cell_query.weight_scale_capacity_elements =
        a4w4_scale_capacity_elements(
            a4w4_prefill_consumer_, kReferenceHiddenSize,
            kReferenceIntermediateSize);
    down_cell_query.output_capacity_elements =
        static_cast<std::size_t>(
            state_->plan().hidden_bf16[1U].element_capacity);
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
    down_cell_query.admission_enabled =
        g_enable_a4w4_down_complete_cell_v3_admission;
    expect_down_complete_cell_v2 =
        reference_runner_detail::use_a4w4_down_complete_cell_v3_route(
            down_cell_query);
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
    down_cell_query.admission_enabled =
        g_enable_a4w4_down_complete_cell_v2_admission;
    expect_down_complete_cell_v2 =
        expect_down_complete_cell_v2 ||
        reference_runner_detail::use_a4w4_down_complete_cell_v2_route(
            down_cell_query);
#endif
#endif
    const std::size_t expected_m128_generic_hits =
        expected_generic_hits -
        (layer_tile == nullptr ? kReferenceDecoderLayerCount : 1U);
    const std::size_t expected_m128_down_hits =
        layer_tile == nullptr ? kReferenceDecoderLayerCount : 1U;
    const bool projection_v3_all =
        a4w4_gateup_projection_v3_tile_hits == expected_paired_hits;
    const bool projection_v3_none =
        a4w4_gateup_projection_v3_tile_hits == 0U;
    const bool complete_cell_v2_all =
        a4w4_gateup_complete_cell_v2_tile_hits == expected_paired_hits;
    const bool complete_cell_v2_none =
        a4w4_gateup_complete_cell_v2_tile_hits == 0U;
    if (!requested_attention_supermatrix_selected ||
        !requested_attention_o_k512_selected ||
        (!attention_supermatrix_inputs_all &&
         !attention_supermatrix_inputs_none) ||
        (!attention_supermatrix_output_all &&
         !attention_supermatrix_output_none) ||
        (!attention_o_k512_all && !attention_o_k512_none) ||
        (attention_supermatrix_output_all && attention_o_k512_all) ||
        (!projection_v3_all && !projection_v3_none) ||
        (!complete_cell_v2_all && !complete_cell_v2_none) ||
        (!projection_v3_none && !complete_cell_v2_none) ||
        a4w4_down_complete_cell_v2_tile_hits !=
            (expect_down_complete_cell_v2 ? expected_m128_down_hits : 0U) ||
        a4w4_full_prefill_tile_hits.activation_quantize_hits !=
            expected_quantize_hits ||
        a4w4_full_prefill_tile_hits.generic_projection_hits !=
            expected_generic_hits ||
        a4w4_full_prefill_tile_hits.paired_gate_up_hits !=
            expected_paired_hits ||
        a4w4_full_prefill_tile_hits.logical_projection_hits !=
            expected_logical_hits ||
        a4w4_full_prefill_tile_hits
                .m128_stage_major_generic_projection_hits !=
            (expect_m128_stage_major ? expected_m128_generic_hits : 0U) ||
        a4w4_full_prefill_tile_hits
                .m128_stage_major_down_projection_hits !=
            (expect_down_m128_stage_major &&
                     !expect_down_complete_cell_v2
                 ? expected_m128_down_hits
                 : 0U) ||
        a4w4_full_prefill_tile_hits
                .m128_stage_major_paired_gate_up_hits !=
            (expect_m128_stage_major && projection_v3_none &&
                     complete_cell_v2_none
                 ? expected_paired_hits
                 : 0U)) {
      return fail_prefill_tile(runner_status(
          ReferenceRunnerError::kInvalidRunner,
          "prefill_a4w4_route_accounting",
          layer_tile == nullptr ? kReferenceNoLayer
                                : layer_tile->item.layer_index));
    }
  }
#endif

  // Match the non-logit step boundary even though this output is not consumed
  // by the following layer-major tile or by persistent state.
  // The M32-prefix plus reference-tail path folded this norm into the final
  // layer's MLP residual boundary.
  const bool completes_final_layer =
      layer_tile == nullptr ||
      layer_tile->item.layer_index + 1U == kReferenceDecoderLayerCount;
  if (!use_m32_residual_rms_fusion && completes_final_layer &&
      !check_cuda(launch_headwise_centered_rms_norm_reference_cuda(
                      views_.hidden[0], weights_->final_norm().data,
                      token_count, kReferenceHiddenSize, kRmsEpsilon,
                      views_.hidden[1], stream_),
                  "prefill_final_norm", kReferenceNoLayer)) {
    return fail_prefill_tile(launch_failure);
  }

  if (layer_tile != nullptr) {
    if (!check_cuda(
            static_cast<int>(cudaMemcpyAsync(
                layer_tile->output_hidden, views_.hidden[0],
                token_count * kReferenceHiddenSize * sizeof(std::uint16_t),
                cudaMemcpyDeviceToDevice,
                static_cast<cudaStream_t>(stream_))),
            "prefill_layer_major_publish_output",
            layer_tile->item.layer_index)) {
      return fail_prefill_tile(launch_failure);
    }
    ReferencePrefillTileOutcome outcome;
    outcome.value.emplace();
    return outcome;
  }

  const cudaError_t sync_status = cudaStreamSynchronize(stream);
  if (sync_status != cudaSuccess) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kCudaFailure, "prefill_tile_synchronize",
        kReferenceNoLayer, static_cast<int>(sync_status)));
  }
  const std::uint32_t committed_length =
      first_position + static_cast<std::uint32_t>(token_count);
  const RequestOperationStatus commit_status =
      state_->set_sequence_length(committed_length);
  if (!commit_status) {
    return fail_prefill_tile(runner_status(
        ReferenceRunnerError::kStateCommitFailure,
        "prefill_tile_commit", kReferenceNoLayer,
        commit_status.cuda_error));
  }
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  if (a4w4_full_prefill_tile_enabled) {
    ++a4w4_full_prefill_tile_hits.complete_model_tile_hits;
    ++g_a4w4_full_prefill_admission_hits.complete_model_tile_hits;
  }
#endif
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
  // prefill_tile_synchronize above completed the full C512 state transition;
  // observe it only after the logical sequence length was committed.
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
    retained_prefill_position_ =
        committed_length - 1U;
    retained_prefill_input_token_ = input_token_ids[token_count - 1U];
    retained_prefill_hidden_row_ = token_count - 1U;
  }
  ReferencePrefillTileOutcome outcome;
  outcome.value.emplace(std::move(tile));
  return outcome;
}

ReferenceRunnerStatus ReferenceRunner::execute_long_prefill_projection_span(
    const LongPrefillProjectionSpanPlan& plan,
    const LongPrefillProjectionSpanWorkItem& item) noexcept {
#if !defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  (void)plan;
  (void)item;
  return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                       "prefill_projection_span_not_built");
#else
  LongPrefillProjectionSpanWorkItem expected_item;
  const std::uint64_t item_end =
      static_cast<std::uint64_t>(item.first_position) + item.token_count;
  if (!long_prefill_projection_span_work_item(
          plan, item.ordinal, expected_item) ||
      expected_item.layer_index != item.layer_index ||
      expected_item.projection_span_index != item.projection_span_index ||
      expected_item.first_position != item.first_position ||
      expected_item.token_count != item.token_count ||
      expected_item.input_hidden_buffer != item.input_hidden_buffer ||
      expected_item.output_hidden_buffer != item.output_hidden_buffer ||
      item.layer_index >= kReferenceDecoderLayerCount ||
      item.token_count == 0U ||
      item.token_count > plan.projection_span_token_count ||
      item_end > plan.prompt_token_count ||
      item.input_hidden_buffer >= kRequestLongPrefillHiddenBufferCount ||
      item.output_hidden_buffer >= kRequestLongPrefillHiddenBufferCount ||
      item.layer_type !=
          reference_runner_detail::expected_reference_layer_type(
              item.layer_index)) {
    return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                         "prefill_projection_span_item",
                         item.layer_index);
  }

  const RequestMemoryPlan& request_plan = state_->plan();
  const std::size_t span_capacity =
      request_plan.long_prefill_projection_span_capacity;
  const reference_runner_detail::A4W4ProjectionSpanPaddingPlan
      projection_padding =
          reference_runner_detail::a4w4_projection_span_padding_plan(
              a4w4_prefill_consumer_, item.token_count, span_capacity);
  const bool selected =
      reference_runner_detail::use_a4w4_full_prefill_tile_route(
          a4w4_full_prefill_admission_enabled_ ||
              g_enable_a4w4_full_prefill_admission,
          trace_enabled_, optimized_prefill_dispatch_disabled());
  if (!selected ||
      !projection_padding.valid() ||
      projection_backend_ != ProjectionBackend::kSm87WeightOnly ||
      span_capacity != plan.projection_span_token_count ||
      span_capacity < item.token_count ||
      views_.long_prefill_hidden[item.input_hidden_buffer] == nullptr ||
      views_.long_prefill_hidden[item.output_hidden_buffer] == nullptr ||
      views_.long_prefill_projection_primary == nullptr ||
      views_.long_prefill_projection_secondary == nullptr ||
      views_.prefill_a4_hidden_packed == nullptr ||
      views_.prefill_a4_hidden_scales == nullptr ||
      views_.prefill_a4_intermediate_packed == nullptr ||
      views_.prefill_a4_intermediate_scales == nullptr ||
      a4w4_full_prefill_inventory_consumer(*weights_) !=
          a4w4_prefill_consumer_) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "prefill_projection_span_workspace",
                         item.layer_index);
  }

  const std::size_t hidden_packed_capacity =
      static_cast<std::size_t>(request_plan.prefill_a4_hidden_packed.byte_size);
  const std::size_t hidden_scale_capacity = static_cast<std::size_t>(
      request_plan.prefill_a4_hidden_scales_bf16.element_capacity);
  const std::size_t intermediate_packed_capacity = static_cast<std::size_t>(
      request_plan.prefill_a4_intermediate_packed.byte_size);
  const std::size_t intermediate_scale_capacity = static_cast<std::size_t>(
      request_plan.prefill_a4_intermediate_scales_bf16.element_capacity);
  if (hidden_packed_capacity <
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(
              span_capacity, kReferenceHiddenSize) ||
      hidden_scale_capacity <
          kernels::sm87_a4w4_consumer_scale_capacity_elements(
              span_capacity, kReferenceHiddenSize) ||
      intermediate_packed_capacity <
          kernels::sm87_a4w4_consumer_packed_capacity_bytes(
              span_capacity, kReferenceIntermediateSize) ||
      intermediate_scale_capacity <
          kernels::sm87_a4w4_consumer_scale_capacity_elements(
              span_capacity, kReferenceIntermediateSize) ||
      reinterpret_cast<std::uintptr_t>(
          views_.prefill_a4_hidden_packed) % 16U != 0U ||
      reinterpret_cast<std::uintptr_t>(
          views_.prefill_a4_intermediate_packed) % 16U != 0U) {
    return runner_status(ReferenceRunnerError::kInvalidRequestState,
                         "prefill_projection_span_a4_capacity",
                         item.layer_index);
  }

  const std::size_t projection_token_count =
      projection_padding.projection_token_count;
  const auto stream = reinterpret_cast<cudaStream_t>(stream_);
  ReferenceRunnerStatus failure{};
  const auto check_cuda = [&failure, &item](
                              const int status,
                              const char* const operation) noexcept {
    if (status == static_cast<int>(cudaSuccess)) {
      return true;
    }
    failure = runner_status(ReferenceRunnerError::kCudaFailure, operation,
                            item.layer_index, status);
    return false;
  };
  const auto projection_output_capacity_elements =
      [this, &request_plan](const std::uint16_t* const output) noexcept {
        if (output == views_.long_prefill_projection_primary) {
          return static_cast<std::size_t>(
              request_plan.long_prefill_projection_primary_bf16
                  .element_capacity);
        }
        if (output == views_.long_prefill_projection_secondary) {
          return static_cast<std::size_t>(
              request_plan.long_prefill_projection_secondary_bf16
                  .element_capacity);
        }
        return std::size_t{0U};
      };
  const auto zero_projection_padding =
      [this, projection_token_count, &check_cuda, stream](
          const std::size_t logical_token_count,
          const std::size_t input_size,
          std::uint8_t* const packed,
          const std::size_t packed_capacity,
          std::uint16_t* const scales,
          const std::size_t scale_capacity,
          const char* const operation) noexcept {
        if (a4w4_prefill_consumer_ !=
                reference_runner_detail::A4W4PrefillConsumer::kK128 ||
            logical_token_count == projection_token_count) {
          return true;
        }
        if (logical_token_count == 0U ||
            logical_token_count > projection_token_count ||
            input_size == 0U ||
            (a4w4_prefill_consumer_ ==
                     reference_runner_detail::A4W4PrefillConsumer::kK256
                 ? input_size % 256U != 0U
                 : input_size % 128U != 0U)) {
          return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                            operation);
        }
        const std::size_t logical_packed_bytes =
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                logical_token_count, input_size);
        const std::size_t padded_packed_bytes =
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                projection_token_count, input_size);
        const std::size_t logical_scale_elements =
            a4w4_scale_capacity_elements(
                a4w4_prefill_consumer_, logical_token_count, input_size);
        const std::size_t padded_scale_elements =
            a4w4_scale_capacity_elements(
                a4w4_prefill_consumer_, projection_token_count, input_size);
        if (logical_packed_bytes == 0U ||
            logical_packed_bytes > padded_packed_bytes ||
            padded_packed_bytes > packed_capacity ||
            logical_scale_elements == 0U ||
            logical_scale_elements > padded_scale_elements ||
            padded_scale_elements > scale_capacity) {
          return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                            operation);
        }
        return check_cuda(
                   static_cast<int>(cudaMemsetAsync(
                       packed + logical_packed_bytes, 0,
                       padded_packed_bytes - logical_packed_bytes, stream)),
                   operation) &&
               check_cuda(
                   static_cast<int>(cudaMemsetAsync(
                       scales + logical_scale_elements, 0,
                       (padded_scale_elements - logical_scale_elements) *
                           sizeof(std::uint16_t),
                       stream)),
                   operation);
      };
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
  // Snapshot admission once for this synchronous projection-span call. The
  // default remains the exact recurrence; an explicitly admitted native
  // route owns every complete C512 state tile in this work item.
  const bool enable_gdn_chunk64_native_admission =
      g_enable_prefill_gdn_chunk64_native_admission;
#endif
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
  // Snapshot once for this synchronous layer/span. The ordinary build and
  // default admission build retain the established exact C16 schedule.
  const bool enable_bf16_ab_large_m_prefill_admission =
      g_enable_bf16_ab_large_m_prefill_admission;
#else
  constexpr bool enable_bf16_ab_large_m_prefill_admission = false;
#endif

  reference_runner_detail::A4W4FullPrefillAdmissionHits local_hits{};
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits
      local_attention_supermatrix_hits{};
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
  reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits
      local_attention_k256_hits{};
#endif
  std::size_t attention_o_k512_hits = 0U;
  std::size_t down_complete_cell_v2_hits = 0U;
  const auto quantize =
      [this, &check_cuda, &local_hits, &zero_projection_padding,
       projection_token_count](
          const std::uint16_t* const input,
          const std::size_t input_row_stride,
          const std::size_t token_count,
          const std::size_t input_size,
          const float clip_ratio,
          std::uint8_t* const packed,
          const std::size_t packed_capacity,
          std::uint16_t* const scales,
          const std::size_t scale_capacity,
          const char* const operation) noexcept {
        const int status = launch_selected_a4w4_quantize(
            a4w4_prefill_consumer_, input, input_row_stride, token_count,
            input_size, clip_ratio, packed, packed_capacity, scales,
            scale_capacity, stream_, projection_token_count);
        if (!check_cuda(status, operation) ||
            !zero_projection_padding(token_count, input_size, packed,
                                     packed_capacity, scales,
                                     scale_capacity, operation)) {
          return false;
        }
        ++local_hits.activation_quantize_hits;
        ++g_a4w4_full_prefill_admission_hits.activation_quantize_hits;
        return true;
      };
  const auto project =
      [this, &check_cuda, &projection_output_capacity_elements, &local_hits,
       &down_complete_cell_v2_hits](
          const LinearWeight& weight,
          const std::uint8_t* const packed_input,
          const std::size_t packed_input_capacity,
          const std::uint16_t* const input_scales,
          const std::size_t input_scale_capacity,
          const std::size_t token_count,
          std::uint16_t* const output,
          const char* const operation) noexcept {
        const PrefillA4LinearSidecarView sidecar =
            prefill_a4_sidecar_view(weight);
        const std::size_t weight_capacity =
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                sidecar.output_size, sidecar.input_size);
        const std::size_t weight_scale_capacity =
            a4w4_scale_capacity_elements(
                a4w4_prefill_consumer_, sidecar.output_size,
                sidecar.input_size);
        bool down_complete_cell_v2_selected = false;
        const int status = launch_selected_a4w4_gemm(
            a4w4_prefill_consumer_, packed_input, packed_input_capacity,
            input_scales, input_scale_capacity, sidecar, weight_capacity,
            weight_scale_capacity, token_count, output,
            projection_output_capacity_elements(output), stream_,
            &down_complete_cell_v2_selected);
        if (!check_cuda(status, operation)) {
          return false;
        }
        ++local_hits.generic_projection_hits;
        ++local_hits.logical_projection_hits;
        ++g_a4w4_full_prefill_admission_hits.generic_projection_hits;
        ++g_a4w4_full_prefill_admission_hits.logical_projection_hits;
        down_complete_cell_v2_hits +=
            down_complete_cell_v2_selected ? 1U : 0U;
        if (down_complete_cell_v2_selected) {
          return true;
        }
        const auto m128_route =
            reference_runner_detail::select_a4w4_k128_generic_prefill_route(
                g_enable_a4w4_m128_stage_major_admission,
                g_enable_a4w4_down_m128_stage_major_admission,
                a4w4_prefill_consumer_, token_count, sidecar.output_size,
                sidecar.input_size);
        if (m128_route ==
            reference_runner_detail::A4W4K128GenericPrefillRoute::
                kM128StageMajor) {
          ++local_hits.m128_stage_major_generic_projection_hits;
          ++g_a4w4_full_prefill_admission_hits
                .m128_stage_major_generic_projection_hits;
        } else if (m128_route ==
                   reference_runner_detail::A4W4K128GenericPrefillRoute::
                       kDownM128StageMajor) {
          ++local_hits.m128_stage_major_down_projection_hits;
          ++g_a4w4_full_prefill_admission_hits
                .m128_stage_major_down_projection_hits;
        }
        return true;
      };
  const auto project_linear_pair =
      [this, &check_cuda, enable_bf16_ab_large_m_prefill_admission](
          const LinearWeight& first_weight,
          const LinearWeight& second_weight,
          const std::uint16_t* const input,
          const std::size_t token_count,
          std::uint16_t* const first_output,
          std::uint16_t* const second_output,
          const char* const operation) noexcept {
        const std::size_t columns = linear_input_size(first_weight);
        const std::size_t first_rows = linear_output_size(first_weight);
        const std::size_t second_rows = linear_output_size(second_weight);
#if defined(Q3X_ENABLE_BF16_AB_LARGE_M_PREFILL_ADMISSION)
        if (enable_bf16_ab_large_m_prefill_admission &&
            token_count >= 2U &&
            token_count <=
                kernels::kSm87Bf16AbLargeMPrefillMaximumTokens &&
            supports_bf16_projection_pair(
                projection_backend_, first_weight, second_weight)) {
          const auto* const first =
              std::get_if<Bf16LinearWeight>(&first_weight);
          const auto* const second =
              std::get_if<Bf16LinearWeight>(&second_weight);
          if (first == nullptr || second == nullptr) {
            return check_cuda(static_cast<int>(cudaErrorInvalidValue),
                              operation);
          }
          const int status =
              kernels::launch_sm87_bf16_ab_large_m_prefill_cuda(
                  first->weight, second->weight, input, token_count,
                  first_output, second_output, stream_);
          if (status == static_cast<int>(cudaSuccess)) {
            // Canonical BF16 A/B are outside the authenticated 400-projection
            // A4 inventory. Keep this dedicated counter orthogonal to both
            // generic/paired A4 totals and their M128 implementation subset.
            ++g_bf16_ab_large_m_prefill_admission_hits;
          }
          return check_cuda(status, operation);
        }
#else
        (void)enable_bf16_ab_large_m_prefill_admission;
#endif
        for (std::size_t offset = 0U; offset < token_count;
             offset += kProductionProjectionSubtileTokens) {
          const std::size_t remaining = token_count - offset;
          const std::size_t count =
              remaining < kProductionProjectionSubtileTokens
                  ? remaining
                  : kProductionProjectionSubtileTokens;
          if (!check_cuda(
                  launch_projection_pair_tile_to_bf16_cuda(
                      projection_backend_, first_weight, second_weight,
                      input + offset * columns, count, views_.fp32_scratch,
                      views_.fp32_scratch_elements,
                      first_output + offset * first_rows,
                      second_output + offset * second_rows, stream_),
                  operation)) {
            return false;
          }
        }
        return true;
      };

  const std::size_t token_count = item.token_count;
  const std::size_t hidden_offset =
      static_cast<std::size_t>(item.first_position) * kReferenceHiddenSize;
  const std::uint16_t* const input_hidden =
      views_.long_prefill_hidden[item.input_hidden_buffer] + hidden_offset;
  std::uint16_t* const output_hidden =
      views_.long_prefill_hidden[item.output_hidden_buffer] + hidden_offset;
  std::uint16_t* const primary = views_.long_prefill_projection_primary;
  std::uint16_t* const secondary = views_.long_prefill_projection_secondary;
  const DecoderLayerWeights& layer_weights =
      weights_->layer(item.layer_index);

  if (!check_cuda(
          launch_headwise_centered_rms_norm_reference_cuda(
              input_hidden, layer_weights.input_layernorm.data, token_count,
              kReferenceHiddenSize, kRmsEpsilon, primary, stream_),
          "prefill_projection_span_input_norm")) {
    return failure;
  }

  const model::LayerType layer_type = item.layer_type;
  if (layer_type == model::LayerType::kLinearAttention) {
    const auto* const attention =
        std::get_if<LinearAttentionWeights>(&layer_weights.attention);
    if (attention == nullptr) {
      return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                           "prefill_projection_span_linear_variant",
                           item.layer_index);
    }
    const PrefillA4LinearSidecarView qkv_sidecar =
        prefill_a4_sidecar_view(attention->in_proj_qkv);
    [[maybe_unused]] const PrefillA4LinearSidecarView z_sidecar =
        prefill_a4_sidecar_view(attention->in_proj_z);
    std::uint16_t* const linear_a =
        primary + projection_token_count * kLinearQkvElements;
    std::uint16_t* const linear_b =
        linear_a + token_count * kLinearScalarElements;
    if (!project_linear_pair(
            attention->in_proj_a, attention->in_proj_b, primary,
            token_count, linear_a, linear_b,
            "prefill_projection_span_linear_ab") ||
        !quantize(primary, kReferenceHiddenSize, token_count,
                  kReferenceHiddenSize, qkv_sidecar.activation_clip_ratio,
                  views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                  "prefill_projection_span_linear_input_quantize")) {
      return failure;
    }
    bool attention_supermatrix_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    if (!check_cuda(
            launch_selected_a4w4_linear_attention_supermatrix(
                a4w4_prefill_consumer_, views_.prefill_a4_hidden_packed,
                hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                hidden_scale_capacity, qkv_sidecar, z_sidecar,
                projection_token_count, primary,
                static_cast<std::size_t>(
                    request_plan.long_prefill_projection_primary_bf16
                        .element_capacity),
                secondary,
                static_cast<std::size_t>(
                    request_plan.long_prefill_projection_secondary_bf16
                        .element_capacity),
                stream_, local_hits, local_attention_supermatrix_hits,
                &attention_supermatrix_selected),
            "prefill_projection_span_linear_qkv_z_supermatrix")) {
      return failure;
    }
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
    if (!attention_supermatrix_selected) {
      const std::array<PrefillA4LinearSidecarView, 2U> sidecars{
          qkv_sidecar, z_sidecar};
      std::uint16_t* const outputs[2U] = {primary, secondary};
      const std::size_t output_capacities[2U] = {
          static_cast<std::size_t>(
              request_plan.long_prefill_projection_primary_bf16
                  .element_capacity),
          static_cast<std::size_t>(
              request_plan.long_prefill_projection_secondary_bf16
                  .element_capacity)};
      if (!check_cuda(
              launch_selected_a4w4_attention_k256_m128n256(
                  kernels::Sm87A4W4AttentionK256Topology::kLinearQkvZ,
                  reference_runner_detail::
                      A4W4AttentionSupermatrixFamily::kLinearInput,
                  a4w4_prefill_consumer_,
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, sidecars.data(), outputs,
                  output_capacities, sidecars.size(), projection_token_count,
                  stream_, local_hits, local_attention_k256_hits,
                  &attention_supermatrix_selected),
              "prefill_projection_span_linear_qkv_z_k256_m128n256")) {
        return failure;
      }
    }
    if (g_enable_a4w4_attention_k256_m128n256_admission &&
        a4w4_prefill_consumer_ ==
            reference_runner_detail::A4W4PrefillConsumer::kK256 &&
        !attention_supermatrix_selected) {
      return runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_projection_span_linear_qkv_z_k256_contract",
          item.layer_index);
    }
#endif
    if (!attention_supermatrix_selected &&
        (!project(attention->in_proj_qkv,
                  views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                  projection_token_count, primary,
                  "prefill_projection_span_linear_qkv") ||
         !project(attention->in_proj_z,
                  views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                  projection_token_count, secondary,
                  "prefill_projection_span_linear_z"))) {
      return failure;
    }

    for (std::uint32_t tile_index = 0U;
         tile_index < item.state_tile_count; ++tile_index) {
      LongPrefillProjectionSpanStateTile tile;
      if (!long_prefill_projection_span_state_tile(
              plan, item.ordinal, tile_index, tile)) {
        return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                             "prefill_projection_span_linear_state_tile",
                             item.layer_index);
      }
      const std::size_t local_first =
          static_cast<std::size_t>(tile.first_position -
                                   item.first_position);
      std::uint16_t* const qkv =
          primary + local_first * kLinearQkvElements;
      const std::uint16_t* const z =
          secondary + local_first * kLinearValueElements;
#if defined(Q3X_ENABLE_GDN_CHUNK64_NATIVE_ADMISSION)
      const bool use_gdn_chunk64_native =
          use_prefill_gdn_chunk64_native_admission(
              enable_gdn_chunk64_native_admission, projection_backend_,
              tile.first_position, tile.token_count,
              prefill_gdn_chunk64_native_workspace_,
              prefill_gdn_chunk64_native_workspace_bytes_);
      if (use_gdn_chunk64_native) {
        const bool use_token_parallel_conv =
            g_enable_gdn_conv_token_parallel_admission;
        const bool use_conv_compact_qk_fused_candidate =
            g_enable_gdn_conv_compact_qk_fused_candidate &&
            use_token_parallel_conv;
        std::uint16_t* const conv_qkv =
            use_token_parallel_conv ? views_.projection[3] : qkv;
        int conv_status = static_cast<int>(cudaErrorInvalidValue);
        if (use_conv_compact_qk_fused_candidate) {
          conv_status = gdn_prefill_chunk64_native_detail::
              launch_fused_conv_compact_qk_preprocess(
                  prefill_gdn_chunk64_native_workspace_,
                  prefill_gdn_chunk64_native_workspace_bytes_, qkv,
                  tile.token_count, attention->conv1d.data,
                  views_.conv_state[item.layer_index], conv_qkv,
                  kRmsEpsilon, stream_);
        } else if (use_token_parallel_conv) {
          conv_status = gdn_prefill_whole_span_conv_detail::
              launch_causal_conv1d_silu_update_token_parallel_exact_cuda(
                  qkv, tile.token_count, attention->conv1d.data,
                  views_.conv_state[item.layer_index], conv_qkv, stream_);
        } else {
          conv_status = gdn_prefill_whole_span_conv_detail::
              launch_causal_conv1d_silu_update_whole_span_exact_cuda(
                  qkv, tile.token_count, attention->conv1d.data,
                  views_.conv_state[item.layer_index], conv_qkv, stream_);
        }
        if (!check_cuda(
                conv_status,
                "prefill_projection_span_linear_conv_chunk64_native")) {
          return failure;
        }
        if (use_conv_compact_qk_fused_candidate) {
          ++g_gdn_conv_compact_qk_fused_candidate_hits;
        }
        ++g_prefill_gdn_chunk64_native_admission_hits;
        const int native_status =
            use_conv_compact_qk_fused_candidate
                ? gdn_prefill_chunk64_native_detail::launch_qk_preprocessed(
                      prefill_gdn_chunk64_native_workspace_,
                      prefill_gdn_chunk64_native_workspace_bytes_, conv_qkv,
                      tile.token_count,
                      linear_a + local_first * kLinearScalarElements,
                      linear_b + local_first * kLinearScalarElements,
                      attention->a_log.data, attention->dt_bias.data,
                      views_.gdn_state[item.layer_index],
                      views_.gdn_state[item.layer_index], kRmsEpsilon,
                      attention->norm.data, z, kRmsEpsilon,
                      views_.projection[2], stream_)
                : gdn_prefill_chunk64_native_detail::launch(
                      prefill_gdn_chunk64_native_workspace_,
                      prefill_gdn_chunk64_native_workspace_bytes_, conv_qkv,
                      tile.token_count,
                      linear_a + local_first * kLinearScalarElements,
                      linear_b + local_first * kLinearScalarElements,
                      attention->a_log.data, attention->dt_bias.data,
                      views_.gdn_state[item.layer_index],
                      views_.gdn_state[item.layer_index], kRmsEpsilon,
                      attention->norm.data, z, kRmsEpsilon,
                      views_.projection[2], stream_);
        if (!check_cuda(
                native_status,
                "prefill_projection_span_linear_gdn_chunk64_native")) {
          return failure;
        }
      } else
#endif
      if (tile.token_count < kLongPrefillLayerMajorTileTokens) {
        // A final short state tile stays on the established exact recurrence
        // by default.  Never round the launch count up to C16: every pointer
        // and state transition is bounded by the true remaining token count.
        for (std::size_t offset = 0U; offset < tile.token_count;
             offset += kPrefillKernelTileMaximumTokens) {
          const std::size_t remaining = tile.token_count - offset;
          const std::size_t count =
              std::min(remaining, kPrefillKernelTileMaximumTokens);
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      qkv + offset * kLinearQkvElements, count,
                      attention->conv1d.data,
                      views_.conv_state[item.layer_index],
                      qkv + offset * kLinearQkvElements, {}, stream_),
                  "prefill_projection_span_linear_conv_tail_exact") ||
              !check_cuda(
                  launch_gated_delta_net_update_tile_warp_parallel_cuda(
                      qkv + offset * kLinearQkvElements, count,
                      linear_a + (local_first + offset) *
                                     kLinearScalarElements,
                      linear_b + (local_first + offset) *
                                     kLinearScalarElements,
                      attention->a_log.data, attention->dt_bias.data,
                      views_.gdn_state[item.layer_index],
                      views_.gdn_state[item.layer_index], kRmsEpsilon,
                      views_.projection[2] +
                          offset * kLinearValueElements,
                      {}, stream_),
                  "prefill_projection_span_linear_gdn_tail_exact")) {
            return failure;
          }
        }
        if (!check_cuda(
                launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                    views_.projection[2], attention->norm.data, z,
                    tile.token_count * kGdnValueHeadCount,
                    kGdnHeadDimension, kRmsEpsilon,
                    views_.projection[2], stream_),
                "prefill_projection_span_linear_tail_exact_norm_gate")) {
          return failure;
        }
      } else if (g_enable_gdn_exact_span_admission) {
        if (!check_cuda(
                gdn_prefill_whole_span_conv_detail::
                    launch_causal_conv1d_silu_update_whole_span_exact_cuda(
                        qkv, tile.token_count, attention->conv1d.data,
                        views_.conv_state[item.layer_index], qkv, stream_),
                "prefill_projection_span_linear_conv_exact_span") ||
            !check_cuda(
                gdn_prefill_exact_span_detail::
                    launch_row16_register_baton(
                        qkv, tile.token_count,
                        linear_a + local_first * kLinearScalarElements,
                        linear_b + local_first * kLinearScalarElements,
                        attention->a_log.data, attention->dt_bias.data,
                        views_.gdn_state[item.layer_index],
                        views_.gdn_state[item.layer_index], kRmsEpsilon,
                        views_.projection[2], stream_),
                "prefill_projection_span_linear_gdn_exact_span") ||
            !check_cuda(
                launch_headwise_plain_rms_norm_silu_gate_reference_cuda(
                    views_.projection[2], attention->norm.data, z,
                    tile.token_count * kGdnValueHeadCount,
                    kGdnHeadDimension, kRmsEpsilon,
                    views_.projection[2], stream_),
                "prefill_projection_span_linear_gdn_exact_span_norm_gate")) {
          return failure;
        }
      } else {
        for (std::size_t offset = 0U; offset < tile.token_count;
             offset += kPrefillKernelTileMaximumTokens) {
          if (!check_cuda(
                  launch_causal_conv1d_silu_update_tile_reference_cuda(
                      qkv + offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      attention->conv1d.data,
                      views_.conv_state[item.layer_index],
                      qkv + offset * kLinearQkvElements, {}, stream_),
                  "prefill_projection_span_linear_conv") ||
              !check_cuda(
                  gdn_prefill_c16_norm_gate_detail::launch_shared_boundary(
                      qkv + offset * kLinearQkvElements,
                      kPrefillKernelTileMaximumTokens,
                      linear_a + (local_first + offset) *
                                     kLinearScalarElements,
                      linear_b + (local_first + offset) *
                                     kLinearScalarElements,
                      attention->a_log.data, attention->dt_bias.data,
                      views_.gdn_state[item.layer_index],
                      views_.gdn_state[item.layer_index], kRmsEpsilon,
                      attention->norm.data,
                      z + offset * kLinearValueElements, kRmsEpsilon,
                      views_.projection[2] +
                          offset * kLinearValueElements,
                      stream_),
                  "prefill_projection_span_linear_gdn_norm_gate")) {
            return failure;
          }
        }
      }
      if (!check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  secondary + local_first * kLinearValueElements,
                  views_.projection[2],
                  static_cast<std::size_t>(tile.token_count) *
                      kLinearValueElements * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToDevice, stream)),
              "prefill_projection_span_linear_publish_output")) {
        return failure;
      }
    }

    const PrefillA4LinearSidecarView output_sidecar =
        prefill_a4_sidecar_view(attention->out_proj);
    bool attention_o_k512_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
    if (!check_cuda(
            launch_selected_a4w4_attention_o_k512(
                a4w4_prefill_consumer_, attention->out_proj, secondary,
                kLinearValueElements, token_count,
                kernels::sm87_a4w4_prefill_k512_launch_token_count(
                    token_count),
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity, secondary,
                kReferenceHiddenSize,
                static_cast<std::size_t>(
                    request_plan.long_prefill_projection_secondary_bf16
                        .element_capacity),
                stream_, &attention_o_k512_selected),
            "prefill_projection_span_linear_output_k512")) {
      return failure;
    }
#endif
    if (attention_o_k512_selected) {
      ++attention_o_k512_hits;
      ++local_hits.activation_quantize_hits;
      ++local_hits.logical_projection_hits;
      ++g_a4w4_full_prefill_admission_hits.activation_quantize_hits;
      ++g_a4w4_full_prefill_admission_hits.logical_projection_hits;
    }
    if (!attention_o_k512_selected &&
        !quantize(secondary, kLinearValueElements, token_count,
                  kLinearValueElements,
                  output_sidecar.activation_clip_ratio,
                  views_.prefill_a4_intermediate_packed,
                  intermediate_packed_capacity,
                  views_.prefill_a4_intermediate_scales,
                  intermediate_scale_capacity,
                  "prefill_projection_span_linear_output_quantize")) {
      return failure;
    }
    bool attention_output_supermatrix_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    if (!attention_o_k512_selected && !check_cuda(
            launch_selected_a4w4_attention_o_supermatrix(
                a4w4_prefill_consumer_,
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity, output_sidecar,
                projection_token_count, secondary,
                static_cast<std::size_t>(
                    request_plan.long_prefill_projection_secondary_bf16
                        .element_capacity),
                stream_, local_hits, local_attention_supermatrix_hits,
                &attention_output_supermatrix_selected),
            "prefill_projection_span_linear_output_supermatrix")) {
      return failure;
    }
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
    if (!attention_o_k512_selected &&
        !attention_output_supermatrix_selected) {
      const std::array<PrefillA4LinearSidecarView, 1U> sidecars{
          output_sidecar};
      std::uint16_t* const outputs[1U] = {secondary};
      const std::size_t output_capacities[1U] = {
          static_cast<std::size_t>(
              request_plan.long_prefill_projection_secondary_bf16
                  .element_capacity)};
      if (!check_cuda(
              launch_selected_a4w4_attention_k256_m128n256(
                  kernels::Sm87A4W4AttentionK256Topology::kAttentionO,
                  reference_runner_detail::
                      A4W4AttentionSupermatrixFamily::kOutput,
                  a4w4_prefill_consumer_,
                  views_.prefill_a4_intermediate_packed,
                  intermediate_packed_capacity,
                  views_.prefill_a4_intermediate_scales,
                  intermediate_scale_capacity, sidecars.data(), outputs,
                  output_capacities, sidecars.size(), projection_token_count,
                  stream_, local_hits, local_attention_k256_hits,
                  &attention_output_supermatrix_selected),
              "prefill_projection_span_linear_output_k256_m128n256")) {
        return failure;
      }
    }
    if (g_enable_a4w4_attention_k256_m128n256_admission &&
        a4w4_prefill_consumer_ ==
            reference_runner_detail::A4W4PrefillConsumer::kK256 &&
        !attention_output_supermatrix_selected) {
      return runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_projection_span_linear_output_k256_contract",
          item.layer_index);
    }
#endif
    if (!attention_o_k512_selected &&
        !attention_output_supermatrix_selected &&
        !project(attention->out_proj,
                 views_.prefill_a4_intermediate_packed,
                 intermediate_packed_capacity,
                 views_.prefill_a4_intermediate_scales,
                 intermediate_scale_capacity, projection_token_count,
                 secondary,
                 "prefill_projection_span_linear_output")) {
      return failure;
    }
  } else if (layer_type == model::LayerType::kFullAttention) {
    const auto* const attention =
        std::get_if<FullAttentionWeights>(&layer_weights.attention);
    if (attention == nullptr) {
      return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                           "prefill_projection_span_full_variant",
                           item.layer_index);
    }
    const PrefillA4LinearSidecarView q_sidecar =
        prefill_a4_sidecar_view(attention->q_proj);
    [[maybe_unused]] const PrefillA4LinearSidecarView k_sidecar =
        prefill_a4_sidecar_view(attention->k_proj);
    [[maybe_unused]] const PrefillA4LinearSidecarView v_sidecar =
        prefill_a4_sidecar_view(attention->v_proj);
    std::uint16_t* const span_key =
        views_.key_cache[item.layer_index] +
        static_cast<std::size_t>(item.first_position) * kFullKvElements;
    std::uint16_t* const span_value =
        views_.value_cache[item.layer_index] +
        static_cast<std::size_t>(item.first_position) * kFullKvElements;
    // A natural final span may end at the exact KV-cache capacity.  Padded
    // K128 projection rows must therefore land in span-local storage, never in
    // the cache.  Only the logical rows are copied into their persistent slots
    // before attention observes them.
    const bool pad_full_kv = projection_padding.padding_token_count != 0U;
    std::uint16_t* const projected_span_key =
        pad_full_kv ? secondary : span_key;
    std::uint16_t* const projected_span_value =
        pad_full_kv
            ? secondary + projection_token_count * kFullKvElements
            : span_value;
    const std::size_t secondary_output_capacity =
        static_cast<std::size_t>(
            request_plan.long_prefill_projection_secondary_bf16
                .element_capacity);
    [[maybe_unused]] const std::size_t projected_span_key_capacity =
        pad_full_kv ? secondary_output_capacity
                    : projection_token_count * kFullKvElements;
    const std::size_t value_offset_elements =
        projection_token_count * kFullKvElements;
    if (pad_full_kv && secondary_output_capacity < 2U * value_offset_elements) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_projection_span_full_kv_capacity",
                           item.layer_index);
    }
    [[maybe_unused]] const std::size_t projected_span_value_capacity =
        pad_full_kv ? secondary_output_capacity - value_offset_elements
                    : projection_token_count * kFullKvElements;
    if (!quantize(primary, kReferenceHiddenSize, token_count,
                  kReferenceHiddenSize, q_sidecar.activation_clip_ratio,
                  views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                  "prefill_projection_span_full_input_quantize")) {
      return failure;
    }
    bool attention_supermatrix_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    if (!check_cuda(
            launch_selected_a4w4_full_attention_supermatrix(
                a4w4_prefill_consumer_, views_.prefill_a4_hidden_packed,
                hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                hidden_scale_capacity, q_sidecar, k_sidecar, v_sidecar,
                projection_token_count, primary,
                static_cast<std::size_t>(
                    request_plan.long_prefill_projection_primary_bf16
                        .element_capacity),
                projected_span_key, projected_span_key_capacity,
                projected_span_value, projected_span_value_capacity, stream_,
                local_hits, local_attention_supermatrix_hits,
                &attention_supermatrix_selected),
            "prefill_projection_span_full_q_k_v_supermatrix")) {
      return failure;
    }
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
    if (!attention_supermatrix_selected) {
      const std::array<PrefillA4LinearSidecarView, 3U> sidecars{
          q_sidecar, k_sidecar, v_sidecar};
      std::uint16_t* const outputs[3U] = {
          primary, projected_span_key, projected_span_value};
      const std::size_t output_capacities[3U] = {
          static_cast<std::size_t>(
              request_plan.long_prefill_projection_primary_bf16
                  .element_capacity),
          projected_span_key_capacity, projected_span_value_capacity};
      if (!check_cuda(
              launch_selected_a4w4_attention_k256_m128n256(
                  kernels::Sm87A4W4AttentionK256Topology::kFullQkv,
                  reference_runner_detail::
                      A4W4AttentionSupermatrixFamily::kFullInput,
                  a4w4_prefill_consumer_,
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, sidecars.data(), outputs,
                  output_capacities, sidecars.size(), projection_token_count,
                  stream_, local_hits, local_attention_k256_hits,
                  &attention_supermatrix_selected),
              "prefill_projection_span_full_q_k_v_k256_m128n256")) {
        return failure;
      }
    }
    if (g_enable_a4w4_attention_k256_m128n256_admission &&
        a4w4_prefill_consumer_ ==
            reference_runner_detail::A4W4PrefillConsumer::kK256 &&
        !attention_supermatrix_selected) {
      return runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_projection_span_full_q_k_v_k256_contract",
          item.layer_index);
    }
#endif
    if (!attention_supermatrix_selected &&
        (!project(attention->q_proj, views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, projection_token_count, primary,
                  "prefill_projection_span_full_q") ||
         !project(attention->k_proj, views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, projection_token_count,
                  projected_span_key,
                  "prefill_projection_span_full_k") ||
         !project(attention->v_proj, views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity, views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, projection_token_count,
                  projected_span_value,
                  "prefill_projection_span_full_v"))) {
      return failure;
    }
    if (pad_full_kv &&
         (!check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  span_key, projected_span_key,
                  token_count * kFullKvElements * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToDevice, stream)),
              "prefill_projection_span_full_publish_k") ||
          !check_cuda(
              static_cast<int>(cudaMemcpyAsync(
                  span_value, projected_span_value,
                  token_count * kFullKvElements * sizeof(std::uint16_t),
                  cudaMemcpyDeviceToDevice, stream)),
              "prefill_projection_span_full_publish_v"))) {
      return failure;
    }

    for (std::uint32_t tile_index = 0U;
         tile_index < item.state_tile_count; ++tile_index) {
      LongPrefillProjectionSpanStateTile tile;
      if (!long_prefill_projection_span_state_tile(
              plan, item.ordinal, tile_index, tile)) {
        return runner_status(ReferenceRunnerError::kInvalidStepOptions,
                             "prefill_projection_span_full_state_tile",
                             item.layer_index);
      }
      const std::size_t local_first =
          static_cast<std::size_t>(tile.first_position -
                                   item.first_position);
      std::uint16_t* const raw_query =
          primary + local_first * kFullQGateElements;
      std::uint16_t* const tile_key =
          span_key + local_first * kFullKvElements;
      std::uint16_t* const packed_gates =
          views_.projection[3] +
          static_cast<std::size_t>(tile.token_count) * kFullQueryElements;
      const std::size_t preprocess_tile_maximum =
          g_enable_full_attention_preprocess_prompt_wide_admission
              ? kFullAttentionPreprocessTileMaximumTokens
              : kPrefillKernelTileMaximumTokens;
      bool use_fused_preprocess = true;
      for (std::size_t offset = 0U; offset < tile.token_count;
           offset += preprocess_tile_maximum) {
        const std::size_t remaining = tile.token_count - offset;
        const std::size_t count =
            remaining < preprocess_tile_maximum
                ? remaining
                : preprocess_tile_maximum;
        const bool valid =
            g_enable_full_attention_preprocess_prompt_wide_admission
                ? reference_runner_detail::use_full_attention_preprocess_tile(
                      static_cast<std::size_t>(tile.first_position) + offset,
                      count)
                : reference_runner_detail::use_qk_rope_tile(
                      static_cast<std::size_t>(tile.first_position) + offset,
                      count);
        if (!valid) {
          use_fused_preprocess = false;
          break;
        }
      }
      std::uint16_t* const tile_query =
          use_fused_preprocess ? views_.projection[3]
                               : views_.projection[0];
      for (std::size_t offset = 0U; offset < tile.token_count;
           offset += preprocess_tile_maximum) {
        const std::size_t remaining = tile.token_count - offset;
        const std::size_t count =
            remaining < preprocess_tile_maximum
                ? remaining
                : preprocess_tile_maximum;
        std::uint16_t* const raw_query_gate =
            raw_query + offset * kFullQGateElements;
        std::uint16_t* const query =
            tile_query + offset * kFullQueryElements;
        std::uint16_t* const split_query =
            views_.projection[3] + offset * kFullQueryElements;
        std::uint16_t* const gates =
            packed_gates + offset * kFullQueryElements;
        std::uint16_t* const key =
            tile_key + offset * kFullKvElements;
        const std::size_t first_position =
            static_cast<std::size_t>(tile.first_position) + offset;
        if (use_fused_preprocess) {
          if (!check_cuda(
                  launch_full_attention_preprocess_24_4_256_64_cuda(
                      raw_query_gate, key, attention->q_norm.data,
                      attention->k_norm.data, kRmsEpsilon, query, gates,
                      views_.rope_cos, views_.rope_sin, first_position,
                      count, stream_),
                  "prefill_projection_span_full_preprocess")) {
            return failure;
          }
          continue;
        }
        if (!check_cuda(
                launch_split_interleaved_q_gate_reference_cuda(
                    raw_query_gate, count * kFullQueryHeads,
                    kFullHeadDimension, split_query, gates, stream_),
                "prefill_projection_span_full_split_q_gate") ||
            !check_cuda(
                launch_headwise_centered_rms_norm_reference_cuda(
                    split_query, attention->q_norm.data,
                    count * kFullQueryHeads, kFullHeadDimension,
                    kRmsEpsilon, query, stream_),
                "prefill_projection_span_full_q_norm") ||
            !check_cuda(
                launch_headwise_centered_rms_norm_reference_cuda(
                    key, attention->k_norm.data, count * kFullKvHeads,
                    kFullHeadDimension, kRmsEpsilon, key, stream_),
                "prefill_projection_span_full_k_norm")) {
          return failure;
        }
        for (std::size_t token = 0U; token < count; ++token) {
          const std::size_t position = first_position + token;
          const float* const cosines =
              views_.rope_cos + position * kRopePairs;
          const float* const sines =
              views_.rope_sin + position * kRopePairs;
          if (!check_cuda(
                  launch_partial_neox_rope_256_64_reference_cuda(
                      query + token * kFullQueryElements, cosines, sines,
                      kFullQueryHeads,
                      query + token * kFullQueryElements, stream_),
                  "prefill_projection_span_full_q_rope") ||
              !check_cuda(
                  launch_partial_neox_rope_256_64_reference_cuda(
                      key + token * kFullKvElements, cosines, sines,
                      kFullKvHeads, key + token * kFullKvElements, stream_),
                  "prefill_projection_span_full_k_rope")) {
            return failure;
          }
        }
      }

      std::uint16_t* const attention_output =
          secondary + local_first * kFullQueryElements;
      if (reference_runner_detail::
              use_bulk_causal_gqa_sigmoid_gate_prefill(
                  projection_backend_, layer_type, tile.first_position,
                  tile.token_count)) {
        if (!check_cuda(
                launch_bulk_causal_gqa_sigmoid_gate_24_4_256_cuda(
                    tile_query, views_.key_cache[item.layer_index],
                    views_.value_cache[item.layer_index], packed_gates,
                    tile.first_position, tile.token_count,
                    attention_output, stream_),
                "prefill_projection_span_full_attention")) {
          return failure;
        }
      } else {
        const std::size_t fused_prefix =
            reference_runner_detail::
                fused_gqa_sigmoid_gate_prefix_token_count(
                    tile.first_position, tile.token_count);
        for (std::size_t token = 0U; token < tile.token_count; ++token) {
          const std::size_t sequence_length =
              static_cast<std::size_t>(tile.first_position) + token + 1U;
          const bool fused = token < fused_prefix;
          const int status =
              fused
                  ? launch_gqa_attention_sigmoid_gate_24_4_256_cuda(
                        tile_query + token * kFullQueryElements,
                        views_.key_cache[item.layer_index],
                        views_.value_cache[item.layer_index], sequence_length,
                        kAttentionScale, views_.fp32_scratch,
                        views_.fp32_scratch_elements,
                        packed_gates + token * kFullQueryElements,
                        attention_output + token * kFullQueryElements,
                        stream_)
                  : launch_gqa_attention_reference_cuda(
                        tile_query + token * kFullQueryElements,
                        views_.key_cache[item.layer_index],
                        views_.value_cache[item.layer_index],
                        kFullQueryHeads, kFullKvHeads, sequence_length,
                        kFullHeadDimension, kAttentionScale,
                        views_.fp32_scratch, views_.fp32_scratch_elements,
                        attention_output + token * kFullQueryElements,
                        stream_);
          if (!check_cuda(status,
                          "prefill_projection_span_full_attention_fallback")) {
            return failure;
          }
        }
        if (fused_prefix < tile.token_count &&
            !check_cuda(
                launch_sigmoid_gate_reference_cuda(
                    attention_output + fused_prefix * kFullQueryElements,
                    packed_gates + fused_prefix * kFullQueryElements,
                    (tile.token_count - fused_prefix) * kFullQueryElements,
                    attention_output + fused_prefix * kFullQueryElements,
                    stream_),
                "prefill_projection_span_full_gate_fallback")) {
          return failure;
        }
      }
    }

    const PrefillA4LinearSidecarView output_sidecar =
        prefill_a4_sidecar_view(attention->o_proj);
    bool attention_o_k512_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
    if (!check_cuda(
            launch_selected_a4w4_attention_o_k512(
                a4w4_prefill_consumer_, attention->o_proj, secondary,
                kFullQueryElements, token_count,
                kernels::sm87_a4w4_prefill_k512_launch_token_count(
                    token_count),
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity, secondary,
                kReferenceHiddenSize,
                static_cast<std::size_t>(
                    request_plan.long_prefill_projection_secondary_bf16
                        .element_capacity),
                stream_, &attention_o_k512_selected),
            "prefill_projection_span_full_output_k512")) {
      return failure;
    }
#endif
    if (attention_o_k512_selected) {
      ++attention_o_k512_hits;
      ++local_hits.activation_quantize_hits;
      ++local_hits.logical_projection_hits;
      ++g_a4w4_full_prefill_admission_hits.activation_quantize_hits;
      ++g_a4w4_full_prefill_admission_hits.logical_projection_hits;
    }
    if (!attention_o_k512_selected &&
        !quantize(secondary, kFullQueryElements, token_count,
                  kFullQueryElements,
                  output_sidecar.activation_clip_ratio,
                  views_.prefill_a4_intermediate_packed,
                  intermediate_packed_capacity,
                  views_.prefill_a4_intermediate_scales,
                  intermediate_scale_capacity,
                  "prefill_projection_span_full_output_quantize")) {
      return failure;
    }
    bool attention_output_supermatrix_selected = false;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    if (!attention_o_k512_selected && !check_cuda(
            launch_selected_a4w4_attention_o_supermatrix(
                a4w4_prefill_consumer_,
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity, output_sidecar,
                projection_token_count, secondary,
                static_cast<std::size_t>(
                    request_plan.long_prefill_projection_secondary_bf16
                        .element_capacity),
                stream_, local_hits, local_attention_supermatrix_hits,
                &attention_output_supermatrix_selected),
            "prefill_projection_span_full_output_supermatrix")) {
      return failure;
    }
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
    if (!attention_o_k512_selected &&
        !attention_output_supermatrix_selected) {
      const std::array<PrefillA4LinearSidecarView, 1U> sidecars{
          output_sidecar};
      std::uint16_t* const outputs[1U] = {secondary};
      const std::size_t output_capacities[1U] = {
          static_cast<std::size_t>(
              request_plan.long_prefill_projection_secondary_bf16
                  .element_capacity)};
      if (!check_cuda(
              launch_selected_a4w4_attention_k256_m128n256(
                  kernels::Sm87A4W4AttentionK256Topology::kAttentionO,
                  reference_runner_detail::
                      A4W4AttentionSupermatrixFamily::kOutput,
                  a4w4_prefill_consumer_,
                  views_.prefill_a4_intermediate_packed,
                  intermediate_packed_capacity,
                  views_.prefill_a4_intermediate_scales,
                  intermediate_scale_capacity, sidecars.data(), outputs,
                  output_capacities, sidecars.size(), projection_token_count,
                  stream_, local_hits, local_attention_k256_hits,
                  &attention_output_supermatrix_selected),
              "prefill_projection_span_full_output_k256_m128n256")) {
        return failure;
      }
    }
    if (g_enable_a4w4_attention_k256_m128n256_admission &&
        a4w4_prefill_consumer_ ==
            reference_runner_detail::A4W4PrefillConsumer::kK256 &&
        !attention_output_supermatrix_selected) {
      return runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_projection_span_full_output_k256_contract",
          item.layer_index);
    }
#endif
    if (!attention_o_k512_selected &&
        !attention_output_supermatrix_selected &&
        !project(attention->o_proj,
                 views_.prefill_a4_intermediate_packed,
                 intermediate_packed_capacity,
                 views_.prefill_a4_intermediate_scales,
                 intermediate_scale_capacity, projection_token_count,
                 secondary,
                 "prefill_projection_span_full_output")) {
      return failure;
    }
  } else {
    return runner_status(ReferenceRunnerError::kInvalidLayerSchedule,
                         "prefill_projection_span_layer_type",
                         item.layer_index);
  }

  if (!check_cuda(
          launch_residual_add_reference_cuda(
              input_hidden, secondary, token_count * kReferenceHiddenSize,
              output_hidden, stream_),
          "prefill_projection_span_attention_residual") ||
      !check_cuda(
          launch_headwise_centered_rms_norm_reference_cuda(
              output_hidden, layer_weights.post_attention_layernorm.data,
              token_count, kReferenceHiddenSize, kRmsEpsilon, primary,
              stream_),
          "prefill_projection_span_post_attention_norm")) {
    return failure;
  }

  const PrefillA4LinearSidecarView gate_sidecar =
      prefill_a4_sidecar_view(layer_weights.mlp.gate_proj);
  const PrefillA4LinearSidecarView up_sidecar =
      prefill_a4_sidecar_view(layer_weights.mlp.up_proj);
  const PrefillA4LinearSidecarView down_sidecar =
      prefill_a4_sidecar_view(layer_weights.mlp.down_proj);
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  const bool mlp_k512_requested = g_enable_a4w4_mlp_k512_admission;
#else
  constexpr bool mlp_k512_requested = false;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
  const bool gateup_down_k512_edge_requested =
      g_enable_a4w4_gateup_down_k512_edge_admission;
#else
  constexpr bool gateup_down_k512_edge_requested = false;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
  const bool gateup_down_k512_edge_m64n128_k256_alternating_requested =
      g_enable_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission;
#else
  constexpr bool gateup_down_k512_edge_m64n128_k256_alternating_requested =
      false;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
  const bool gateup_down_k512_edge_m128n64_requested =
      g_enable_a4w4_gateup_down_k512_edge_m128n64_admission;
#else
  constexpr bool gateup_down_k512_edge_m128n64_requested = false;
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
  const bool down_k512_m16n64_v2_requested =
      g_enable_a4w4_down_k512_m16n64_v2_admission;
#else
  constexpr bool down_k512_m16n64_v2_requested = false;
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
  const bool mlp_k512_fragment_native_requested =
      g_enable_a4w4_mlp_k512_fragment_native_admission;
#else
  constexpr bool mlp_k512_fragment_native_requested = false;
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  const reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute
      paired_gateup_canonical_down_route =
          reference_runner_detail::
              select_a4w4_paired_gateup_canonical_down_route(
                  a4w4_paired_gateup_canonical_down_selector_query(true));
  if (paired_gateup_canonical_down_route ==
      reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::kInvalid) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_paired_gateup_canonical_down_selector_contract",
        item.layer_index);
  }
#else
  constexpr reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute
      paired_gateup_canonical_down_route =
          reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
              kDisabled;
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  const reference_runner_detail::
      A4W4DownK512M128N128LdmatrixPairringV1Route
          down_m128n128_ldmatrix_pairring_v1_route =
              reference_runner_detail::
                  select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
                      a4w4_down_k512_m128n128_ldmatrix_pairring_v1_selector_query(
                          true));
  if (down_m128n128_ldmatrix_pairring_v1_route ==
      reference_runner_detail::
          A4W4DownK512M128N128LdmatrixPairringV1Route::kInvalid) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring_v1_selector_contract",
        item.layer_index);
  }
#else
  constexpr reference_runner_detail::
      A4W4DownK512M128N128LdmatrixPairringV1Route
          down_m128n128_ldmatrix_pairring_v1_route =
              reference_runner_detail::
                  A4W4DownK512M128N128LdmatrixPairringV1Route::kDisabled;
#endif
#if !defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
  if (paired_gateup_canonical_down_route !=
      reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
          kDisabled) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_paired_gateup_canonical_down_not_built",
        item.layer_index);
  }
#endif
#if !defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
  if (paired_gateup_canonical_down_route ==
      reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
          kGateAndDown) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring_not_built",
        item.layer_index);
  }
#endif
#if !defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
  if (down_m128n128_ldmatrix_pairring_v1_route ==
      reference_runner_detail::
          A4W4DownK512M128N128LdmatrixPairringV1Route::kEnabled) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring_v1_not_built",
        item.layer_index);
  }
#endif
  if (mlp_k512_requested && mlp_k512_fragment_native_requested) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_selector_conflict",
        item.layer_index);
  }
  if (gateup_down_k512_edge_requested && !mlp_k512_requested) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_gateup_down_edge_requires_v1",
        item.layer_index);
  }
  if (gateup_down_k512_edge_m64n128_k256_alternating_requested &&
      !mlp_k512_requested) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating_requires_v1",
        item.layer_index);
  }
  if (gateup_down_k512_edge_m128n64_requested && !mlp_k512_requested) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_gateup_down_edge_m128n64_requires_v1",
        item.layer_index);
  }
  if (gateup_down_k512_edge_requested &&
      gateup_down_k512_edge_m128n64_requested) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_gateup_down_edge_selector_conflict",
        item.layer_index);
  }
  if (gateup_down_k512_edge_m64n128_k256_alternating_requested &&
      (gateup_down_k512_edge_requested ||
       gateup_down_k512_edge_m128n64_requested)) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating_selector_conflict",
        item.layer_index);
  }
  if (gateup_down_k512_edge_m64n128_k256_alternating_requested &&
      (g_enable_a4w4_gateup_complete_cell_v2_admission ||
       g_enable_a4w4_m128_stage_major_admission
#if defined(Q3X_ENABLE_A4W4_GATEUP_PROJECTION_V3_ADMISSION)
       || g_enable_a4w4_gateup_projection_v3_admission
#endif
       )) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating_gateup_selector_conflict",
        item.layer_index);
  }
  if (gateup_down_k512_edge_m64n128_k256_alternating_requested &&
      down_k512_m16n64_v2_requested) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating_down_v2_selector_conflict",
        item.layer_index);
  }
  if (gateup_down_k512_edge_m128n64_requested &&
      down_k512_m16n64_v2_requested) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_mlp_k512_edge_m128n64_down_v2_selector_conflict",
        item.layer_index);
  }
  if (down_k512_m16n64_v2_requested && !mlp_k512_requested) {
    return runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_projection_span_down_k512_m16n64_v2_requires_v1",
        item.layer_index);
  }
  const std::size_t gate_weight_capacity =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          gate_sidecar.output_size, gate_sidecar.input_size);
  const std::size_t gate_scale_capacity =
      a4w4_scale_capacity_elements(
          a4w4_prefill_consumer_, gate_sidecar.output_size,
          gate_sidecar.input_size);
  const std::size_t up_weight_capacity =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          up_sidecar.output_size, up_sidecar.input_size);
  const std::size_t up_scale_capacity =
      a4w4_scale_capacity_elements(
          a4w4_prefill_consumer_, up_sidecar.output_size,
          up_sidecar.input_size);
  bool projection_v3_selected = false;
  bool complete_cell_v2_selected = false;
  bool mlp_k512_selected = false;
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
  if (paired_gateup_canonical_down_route ==
          reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
              kGateOnly ||
      paired_gateup_canonical_down_route ==
          reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
              kGateAndDown) {
    const PrefillMLPK512FragmentNativeCompositeView& composite =
        layer_weights.prefill_mlp_k512_fragment_native;
    const std::size_t secondary_capacity = static_cast<std::size_t>(
        request_plan.long_prefill_projection_secondary_bf16
            .element_capacity);
    const std::size_t mlp_launch_token_count =
        kernels::sm87_a4w4_prefill_k512_launch_token_count(token_count);
    const auto gate_plan = kernels::
        sm87_a4w4_gateup_down_edge_m128n512_paired_ldmatrix_plan(
            token_count, mlp_launch_token_count,
            kReferenceIntermediateSize, kReferenceHiddenSize);
    const auto down_plan = kernels::sm87_a4w4_down_k512_plan(
        mlp_launch_token_count, kReferenceHiddenSize,
        kReferenceIntermediateSize);
    if (!composite.attached() ||
        composite.physical_layout !=
            PrefillMLPK512CompositeLayout::kPairedGateUpCanonicalV1Down ||
        composite.gateup_code_capacity_bytes !=
            kPrefillMLPK512FragmentNativeGateUpCodeBytes ||
        composite.gateup_scale_capacity_elements !=
            kPrefillMLPK512FragmentNativeGateUpScaleBytes /
                sizeof(std::uint16_t) ||
        composite.down_code_capacity_bytes !=
            kPrefillMLPK512FragmentNativeDownCodeBytes ||
        composite.down_scale_capacity_elements !=
            kPrefillMLPK512FragmentNativeDownScaleBytes /
                sizeof(std::uint16_t) ||
        !std::isfinite(composite.gateup_activation_clip_ratio) ||
        !std::isfinite(composite.down_activation_clip_ratio) ||
        mlp_launch_token_count == 0U ||
        mlp_launch_token_count != projection_token_count ||
        mlp_launch_token_count > span_capacity ||
        gate_plan.launch_ctas == 0U || down_plan.launch_ctas == 0U ||
        gate_plan.required_scratch_bytes !=
            kRequestA4GateUpCtaScratchBytes ||
        views_.prefill_a4_gateup_cta_scratch == nullptr ||
        views_.prefill_a4_gateup_cta_scratch_bytes !=
            kRequestA4GateUpCtaScratchBytes ||
        reinterpret_cast<std::uintptr_t>(
            views_.prefill_a4_gateup_cta_scratch) % 16U != 0U ||
        hidden_packed_capacity <
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                mlp_launch_token_count, kReferenceHiddenSize) ||
        hidden_scale_capacity <
            kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(
                mlp_launch_token_count, kReferenceHiddenSize) ||
        intermediate_packed_capacity <
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                mlp_launch_token_count, kReferenceIntermediateSize) ||
        intermediate_scale_capacity <
            kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(
                mlp_launch_token_count, kReferenceIntermediateSize) ||
        secondary_capacity <
            mlp_launch_token_count * kReferenceHiddenSize) {
      return runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_projection_span_mlp_k512_paired_gateup_canonical_down_contract",
          item.layer_index);
    }
    if (!check_cuda(
            kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                primary, kReferenceHiddenSize, token_count,
                mlp_launch_token_count, kReferenceHiddenSize,
                composite.gateup_activation_clip_ratio,
                views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                stream_),
            "prefill_projection_span_mlp_k512_paired_gateup_canonical_down_input_quantize") ||
        !check_cuda(
            kernels::
                launch_sm87_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_cuda(
                    views_.prefill_a4_hidden_packed,
                    hidden_packed_capacity,
                    views_.prefill_a4_hidden_scales,
                    hidden_scale_capacity, composite.gateup_codes,
                    composite.gateup_code_capacity_bytes,
                    composite.gateup_scales,
                    composite.gateup_scale_capacity_elements, token_count,
                    mlp_launch_token_count, kReferenceIntermediateSize,
                    kReferenceHiddenSize,
                    composite.down_activation_clip_ratio,
                    views_.prefill_a4_gateup_cta_scratch,
                    views_.prefill_a4_gateup_cta_scratch_bytes,
                    views_.prefill_a4_intermediate_packed,
                    intermediate_packed_capacity,
                    views_.prefill_a4_intermediate_scales,
                    intermediate_scale_capacity, stream_),
            "prefill_projection_span_mlp_k512_gateup_down_edge_m128n512_paired_ldmatrix")) {
      return failure;
    }
    ++g_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_admission_hits;

    int down_status = static_cast<int>(cudaErrorInvalidValue);
    const char* down_stage = nullptr;
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
    if (paired_gateup_canonical_down_route ==
        reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
            kGateAndDown) {
      down_status = kernels::
          launch_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_bf16_cuda(
              views_.prefill_a4_intermediate_packed,
              intermediate_packed_capacity,
              views_.prefill_a4_intermediate_scales,
              intermediate_scale_capacity, composite.down_codes,
              composite.down_code_capacity_bytes, composite.down_scales,
              composite.down_scale_capacity_elements,
              mlp_launch_token_count, kReferenceHiddenSize,
              kReferenceIntermediateSize, secondary,
              kReferenceHiddenSize, secondary_capacity, stream_);
      down_stage =
          "prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring";
    } else
#endif
    {
      down_status = kernels::launch_sm87_a4w4_down_k512_macrocell_bf16_cuda(
          views_.prefill_a4_intermediate_packed,
          intermediate_packed_capacity,
          views_.prefill_a4_intermediate_scales,
          intermediate_scale_capacity, composite.down_codes,
          composite.down_code_capacity_bytes, composite.down_scales,
          composite.down_scale_capacity_elements, mlp_launch_token_count,
          kReferenceHiddenSize, kReferenceIntermediateSize, secondary,
          kReferenceHiddenSize, secondary_capacity, stream_);
      down_stage =
          "prefill_projection_span_mlp_k512_paired_gateup_canonical_down_down";
    }
    if (!check_cuda(down_status, down_stage)) {
      return failure;
    }
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
    if (paired_gateup_canonical_down_route ==
        reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
            kGateAndDown) {
      ++g_a4w4_down_k512_m128n128_ldmatrix_pairring_admission_hits;
    }
#endif
    local_hits.activation_quantize_hits += 2U;
    g_a4w4_full_prefill_admission_hits.activation_quantize_hits += 2U;
    ++local_hits.paired_gate_up_hits;
    ++local_hits.generic_projection_hits;
    local_hits.logical_projection_hits += 3U;
    ++g_a4w4_full_prefill_admission_hits.paired_gate_up_hits;
    ++g_a4w4_full_prefill_admission_hits.generic_projection_hits;
    g_a4w4_full_prefill_admission_hits.logical_projection_hits += 3U;
    mlp_k512_selected = true;
  }
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
  if (!mlp_k512_selected && mlp_k512_fragment_native_requested) {
    const PrefillMLPK512FragmentNativeCompositeView& fragment =
        layer_weights.prefill_mlp_k512_fragment_native;
    constexpr std::size_t kPrimaryProductColumns =
        kRequestLongPrefillPrimaryWidth;
    constexpr std::size_t kSecondaryProductColumns =
        kReferenceIntermediateSize - kPrimaryProductColumns;
    const std::size_t primary_capacity =
        static_cast<std::size_t>(
            request_plan.long_prefill_projection_primary_bf16
                .element_capacity);
    const std::size_t secondary_capacity =
        static_cast<std::size_t>(
            request_plan.long_prefill_projection_secondary_bf16
                .element_capacity);
    const std::size_t mlp_launch_token_count =
        kernels::sm87_a4w4_prefill_k512_launch_token_count(token_count);
#if defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_ADMISSION)
    const auto gateup_primary =
        kernels::
            sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
                mlp_launch_token_count, kReferenceIntermediateSize,
                kReferenceHiddenSize, 0U, kPrimaryProductColumns);
    const auto gateup_secondary =
        kernels::
            sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_plan(
                mlp_launch_token_count, kReferenceIntermediateSize,
                kReferenceHiddenSize, kPrimaryProductColumns,
                kSecondaryProductColumns);
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_ADMISSION)
    const auto gateup_primary =
        kernels::
            sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
                mlp_launch_token_count, kReferenceIntermediateSize,
                kReferenceHiddenSize, 0U, kPrimaryProductColumns);
    const auto gateup_secondary =
        kernels::
            sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_plan(
                mlp_launch_token_count, kReferenceIntermediateSize,
                kReferenceHiddenSize, kPrimaryProductColumns,
                kSecondaryProductColumns);
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_ADMISSION)
    const auto gateup_primary =
        kernels::
            sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
                mlp_launch_token_count, kReferenceIntermediateSize,
                kReferenceHiddenSize, 0U, kPrimaryProductColumns);
    const auto gateup_secondary =
        kernels::
            sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_plan(
                mlp_launch_token_count, kReferenceIntermediateSize,
                kReferenceHiddenSize, kPrimaryProductColumns,
                kSecondaryProductColumns);
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_ADMISSION)
    const auto gateup_primary =
        kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
            mlp_launch_token_count, kReferenceIntermediateSize,
            kReferenceHiddenSize, 0U, kPrimaryProductColumns);
    const auto gateup_secondary =
        kernels::sm87_a4w4_gateup_k512_fragment_native_m128_plan(
            mlp_launch_token_count, kReferenceIntermediateSize,
            kReferenceHiddenSize, kPrimaryProductColumns,
            kSecondaryProductColumns);
#else
    const auto gateup_primary =
        kernels::sm87_a4w4_gateup_k512_fragment_native_plan(
            mlp_launch_token_count, kReferenceIntermediateSize,
            kReferenceHiddenSize, 0U, kPrimaryProductColumns);
    const auto gateup_secondary =
        kernels::sm87_a4w4_gateup_k512_fragment_native_plan(
            mlp_launch_token_count, kReferenceIntermediateSize,
            kReferenceHiddenSize, kPrimaryProductColumns,
            kSecondaryProductColumns);
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_ADMISSION)
    const auto down =
        kernels::sm87_a4w4_down_k512_fragment_native_m128n256_1cta_plan(
            mlp_launch_token_count, kReferenceHiddenSize,
            kReferenceIntermediateSize);
#else
    const auto down = kernels::sm87_a4w4_down_k512_fragment_native_plan(
        mlp_launch_token_count, kReferenceHiddenSize,
        kReferenceIntermediateSize);
#endif
    if (!fragment.attached() ||
        fragment.physical_layout !=
            PrefillMLPK512CompositeLayout::
                kPairedGateUpFragmentNativeDown ||
        fragment.gateup_code_capacity_bytes !=
            kPrefillMLPK512FragmentNativeGateUpCodeBytes ||
        fragment.gateup_scale_capacity_elements !=
            kPrefillMLPK512FragmentNativeGateUpScaleBytes /
                sizeof(std::uint16_t) ||
        fragment.down_code_capacity_bytes !=
            kPrefillMLPK512FragmentNativeDownCodeBytes ||
        fragment.down_scale_capacity_elements !=
            kPrefillMLPK512FragmentNativeDownScaleBytes /
                sizeof(std::uint16_t) ||
        mlp_launch_token_count == 0U ||
        mlp_launch_token_count != projection_token_count ||
        mlp_launch_token_count > span_capacity ||
        gateup_primary.launch_ctas == 0U ||
        gateup_secondary.launch_ctas == 0U || down.launch_ctas == 0U ||
        primary_capacity <
            mlp_launch_token_count * kPrimaryProductColumns ||
        secondary_capacity <
            mlp_launch_token_count * kRequestLongPrefillSecondaryWidth ||
        hidden_packed_capacity <
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                mlp_launch_token_count, kReferenceHiddenSize) ||
        hidden_scale_capacity <
            kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(
                mlp_launch_token_count, kReferenceHiddenSize) ||
        intermediate_packed_capacity <
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                mlp_launch_token_count, kReferenceIntermediateSize) ||
        intermediate_scale_capacity <
            kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(
                mlp_launch_token_count, kReferenceIntermediateSize)) {
      return runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_projection_span_mlp_k512_fragment_native_contract",
          item.layer_index);
    }
    const auto launch_fragment_gateup =
        [&](const std::size_t n_start, const std::size_t n_count,
            std::uint16_t* const output,
            const std::size_t output_row_stride_elements,
            const std::size_t output_capacity_elements) noexcept {
#if defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_ADMISSION)
          return kernels::
              launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_staged_bf16_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, fragment.gateup_codes,
                  fragment.gateup_code_capacity_bytes,
                  fragment.gateup_scales,
                  fragment.gateup_scale_capacity_elements,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize, n_start, n_count, output,
                  output_row_stride_elements, output_capacity_elements,
                  stream_);
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_ADMISSION)
          return kernels::
              launch_sm87_a4w4_gateup_k512_fragment_native_m64n128_1cta_bf16_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, fragment.gateup_codes,
                  fragment.gateup_code_capacity_bytes,
                  fragment.gateup_scales,
                  fragment.gateup_scale_capacity_elements,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize, n_start, n_count, output,
                  output_row_stride_elements, output_capacity_elements,
                  stream_);
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_ADMISSION)
          return kernels::
              launch_sm87_a4w4_gateup_k512_fragment_native_m128n64_1cta_bf16_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, fragment.gateup_codes,
                  fragment.gateup_code_capacity_bytes,
                  fragment.gateup_scales,
                  fragment.gateup_scale_capacity_elements,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize, n_start, n_count, output,
                  output_row_stride_elements, output_capacity_elements,
                  stream_);
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_ADMISSION)
          return kernels::
              launch_sm87_a4w4_gateup_k512_fragment_native_m128_bf16_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, fragment.gateup_codes,
                  fragment.gateup_code_capacity_bytes,
                  fragment.gateup_scales,
                  fragment.gateup_scale_capacity_elements,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize, n_start, n_count, output,
                  output_row_stride_elements, output_capacity_elements,
                  stream_);
#else
          return kernels::
              launch_sm87_a4w4_gateup_k512_fragment_native_bf16_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, fragment.gateup_codes,
                  fragment.gateup_code_capacity_bytes,
                  fragment.gateup_scales,
                  fragment.gateup_scale_capacity_elements,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize, n_start, n_count, output,
                  output_row_stride_elements, output_capacity_elements,
                  stream_);
#endif
        };
#if defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_STAGED_ADMISSION)
    constexpr const char* kGateupPrimaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_primary";
    constexpr const char* kGateupSecondaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_m128n64_staged_gateup_secondary";
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M64N128_1CTA_ADMISSION)
    constexpr const char* kGateupPrimaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_primary";
    constexpr const char* kGateupSecondaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_m64n128_1cta_gateup_secondary";
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128N64_1CTA_ADMISSION)
    constexpr const char* kGateupPrimaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_primary";
    constexpr const char* kGateupSecondaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_m128n64_1cta_gateup_secondary";
#elif defined(Q3X_ENABLE_A4W4_GATEUP_K512_FRAGMENT_NATIVE_M128_ADMISSION)
    constexpr const char* kGateupPrimaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_m128_gateup_primary";
    constexpr const char* kGateupSecondaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_m128_gateup_secondary";
#else
    constexpr const char* kGateupPrimaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_gateup_primary";
    constexpr const char* kGateupSecondaryStage =
        "prefill_projection_span_mlp_k512_fragment_native_gateup_secondary";
#endif
    const auto launch_fragment_down = [&]() noexcept {
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_ADMISSION)
      return kernels::
          launch_sm87_a4w4_down_k512_fragment_native_m128n256_1cta_bf16_cuda(
              views_.prefill_a4_intermediate_packed,
              intermediate_packed_capacity,
              views_.prefill_a4_intermediate_scales,
              intermediate_scale_capacity, fragment.down_codes,
              fragment.down_code_capacity_bytes, fragment.down_scales,
              fragment.down_scale_capacity_elements,
              mlp_launch_token_count, kReferenceHiddenSize,
              kReferenceIntermediateSize, secondary,
              kReferenceHiddenSize, secondary_capacity, stream_);
#else
      return kernels::launch_sm87_a4w4_down_k512_fragment_native_bf16_cuda(
          views_.prefill_a4_intermediate_packed,
          intermediate_packed_capacity,
          views_.prefill_a4_intermediate_scales,
          intermediate_scale_capacity, fragment.down_codes,
          fragment.down_code_capacity_bytes, fragment.down_scales,
          fragment.down_scale_capacity_elements, mlp_launch_token_count,
          kReferenceHiddenSize, kReferenceIntermediateSize, secondary,
          kReferenceHiddenSize, secondary_capacity, stream_);
#endif
    };
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_FRAGMENT_NATIVE_M128N256_1CTA_ADMISSION)
    constexpr const char* kDownStage =
        "prefill_projection_span_mlp_k512_fragment_native_m128n256_1cta_down";
#else
    constexpr const char* kDownStage =
        "prefill_projection_span_mlp_k512_fragment_native_down";
#endif
    if (!check_cuda(
            kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                primary, kReferenceHiddenSize, token_count,
                mlp_launch_token_count, kReferenceHiddenSize,
                fragment.gateup_activation_clip_ratio,
                views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                stream_),
            "prefill_projection_span_mlp_k512_fragment_native_input_quantize") ||
        !check_cuda(
            launch_fragment_gateup(
                0U, kPrimaryProductColumns, primary,
                kPrimaryProductColumns, primary_capacity),
            kGateupPrimaryStage) ||
        !check_cuda(
            launch_fragment_gateup(
                kPrimaryProductColumns, kSecondaryProductColumns,
                secondary, kRequestLongPrefillSecondaryWidth,
                secondary_capacity),
            kGateupSecondaryStage) ||
        !check_cuda(
            kernels::launch_sm87_a4_quantize_bf16_k512_split_cuda(
                primary, kPrimaryProductColumns, kPrimaryProductColumns,
                secondary, kRequestLongPrefillSecondaryWidth,
                kSecondaryProductColumns, token_count,
                mlp_launch_token_count,
                fragment.down_activation_clip_ratio,
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity, stream_),
            "prefill_projection_span_mlp_k512_fragment_native_product_quantize") ||
        !check_cuda(launch_fragment_down(), kDownStage)) {
      return failure;
    }
    local_hits.activation_quantize_hits += 2U;
    g_a4w4_full_prefill_admission_hits.activation_quantize_hits += 2U;
    ++local_hits.paired_gate_up_hits;
    ++local_hits.generic_projection_hits;
    local_hits.logical_projection_hits += 3U;
    ++g_a4w4_full_prefill_admission_hits.paired_gate_up_hits;
    ++g_a4w4_full_prefill_admission_hits.generic_projection_hits;
    g_a4w4_full_prefill_admission_hits.logical_projection_hits += 3U;
    ++g_a4w4_mlp_k512_fragment_native_admission_hits;
    mlp_k512_selected = true;
  }
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  if (!mlp_k512_selected && mlp_k512_requested) {
    const PrefillMLPK512LinearSidecarView gate_k512 =
        prefill_mlp_k512_sidecar_view(layer_weights.mlp.gate_proj);
    const PrefillMLPK512LinearSidecarView up_k512 =
        prefill_mlp_k512_sidecar_view(layer_weights.mlp.up_proj);
    const PrefillMLPK512LinearSidecarView down_k512 =
        prefill_mlp_k512_sidecar_view(layer_weights.mlp.down_proj);
    constexpr std::size_t kPrimaryProductColumns =
        kRequestLongPrefillPrimaryWidth;
    constexpr std::size_t kSecondaryProductColumns =
        kReferenceIntermediateSize - kPrimaryProductColumns;
    const std::size_t primary_capacity =
        static_cast<std::size_t>(
            request_plan.long_prefill_projection_primary_bf16
                .element_capacity);
    const std::size_t secondary_capacity =
        static_cast<std::size_t>(
            request_plan.long_prefill_projection_secondary_bf16
                .element_capacity);
    const std::size_t mlp_launch_token_count =
        kernels::sm87_a4w4_prefill_k512_launch_token_count(token_count);
    const std::size_t gate_k512_weight_capacity =
        kernels::sm87_a4w4_consumer_packed_capacity_bytes(
            gate_k512.output_size, gate_k512.input_size);
    const std::size_t gate_k512_scale_capacity =
        kernels::sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
            gate_k512.output_size, gate_k512.input_size);
    const std::size_t up_k512_weight_capacity =
        kernels::sm87_a4w4_consumer_packed_capacity_bytes(
            up_k512.output_size, up_k512.input_size);
    const std::size_t up_k512_scale_capacity =
        kernels::sm87_a4w4_gateup_k512_macro_scale_capacity_elements(
            up_k512.output_size, up_k512.input_size);
    const std::size_t down_k512_weight_capacity =
        kernels::sm87_a4w4_down_k512_packed_capacity_bytes(
            down_k512.output_size, down_k512.input_size);
    const std::size_t down_k512_scale_capacity =
        kernels::sm87_a4w4_down_k512_scale_capacity_elements(
            down_k512.output_size, down_k512.input_size);
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
    const auto gateup_down_edge_plan =
        kernels::sm87_a4w4_gateup_down_edge_plan(
            token_count, mlp_launch_token_count,
            kReferenceIntermediateSize, kReferenceHiddenSize);
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
    const auto gateup_down_edge_m64n128_k256_alternating_plan =
        kernels::sm87_a4w4_gateup_down_edge_plan(
            token_count, mlp_launch_token_count,
            kReferenceIntermediateSize, kReferenceHiddenSize);
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
    const auto gateup_down_edge_m128n64_plan =
        kernels::sm87_a4w4_gateup_down_edge_m128n64_plan(
            token_count, mlp_launch_token_count,
            kReferenceIntermediateSize, kReferenceHiddenSize);
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
    const auto down_m16n64_v2_plan =
        kernels::sm87_a4w4_down_k512_m16n64_v2_plan(
            mlp_launch_token_count, kReferenceHiddenSize,
            kReferenceIntermediateSize);
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
    const auto down_m128n128_ldmatrix_pairring_plan =
        kernels::sm87_a4w4_down_k512_plan(
            mlp_launch_token_count, kReferenceHiddenSize,
            kReferenceIntermediateSize);
#endif
    if (!gate_k512.attached() || !up_k512.attached() ||
        !down_k512.attached() ||
        gate_k512.activation_clip_ratio !=
            up_k512.activation_clip_ratio ||
        mlp_launch_token_count == 0U ||
        mlp_launch_token_count > span_capacity ||
        kernels::sm87_a4w4_down_k512_plan(
            mlp_launch_token_count, kReferenceHiddenSize,
            kReferenceIntermediateSize).launch_ctas == 0U ||
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
        (gateup_down_k512_edge_requested &&
         gateup_down_edge_plan.launch_ctas == 0U) ||
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
        (gateup_down_k512_edge_m64n128_k256_alternating_requested &&
         gateup_down_edge_m64n128_k256_alternating_plan.launch_ctas == 0U) ||
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
        (gateup_down_k512_edge_m128n64_requested &&
         gateup_down_edge_m128n64_plan.launch_ctas == 0U) ||
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
        (down_k512_m16n64_v2_requested &&
         down_m16n64_v2_plan.launch_ctas == 0U) ||
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
        (down_m128n128_ldmatrix_pairring_v1_route ==
             reference_runner_detail::
                 A4W4DownK512M128N128LdmatrixPairringV1Route::kEnabled &&
         down_m128n128_ldmatrix_pairring_plan.launch_ctas == 0U) ||
#endif
        primary_capacity <
            mlp_launch_token_count * kPrimaryProductColumns ||
        secondary_capacity <
            mlp_launch_token_count * kRequestLongPrefillSecondaryWidth ||
        hidden_packed_capacity <
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                mlp_launch_token_count, kReferenceHiddenSize) ||
        hidden_scale_capacity <
            kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(
                mlp_launch_token_count, kReferenceHiddenSize) ||
        intermediate_packed_capacity <
            kernels::sm87_a4w4_consumer_packed_capacity_bytes(
                mlp_launch_token_count, kReferenceIntermediateSize) ||
        intermediate_scale_capacity <
            kernels::sm87_a4w4_prefill_k512_scale_capacity_elements(
                mlp_launch_token_count, kReferenceIntermediateSize)) {
      return runner_status(ReferenceRunnerError::kInvalidRequestState,
                           "prefill_projection_span_mlp_k512_contract",
                           item.layer_index);
    }
    if (!check_cuda(
            kernels::launch_sm87_a4_quantize_bf16_k512_cuda(
                primary, kReferenceHiddenSize, token_count,
                mlp_launch_token_count, kReferenceHiddenSize,
                gate_k512.activation_clip_ratio,
                views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                stream_),
            "prefill_projection_span_mlp_k512_input_quantize")) {
      return failure;
    }
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
    if (gateup_down_k512_edge_m64n128_k256_alternating_requested) {
      if (!check_cuda(
              kernels::launch_sm87_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, gate_k512.weight,
                  gate_k512_weight_capacity, gate_k512.scales,
                  gate_k512_scale_capacity, up_k512.weight,
                  up_k512_weight_capacity, up_k512.scales,
                  up_k512_scale_capacity, token_count,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize,
                  down_k512.activation_clip_ratio,
                  views_.prefill_a4_intermediate_packed,
                  intermediate_packed_capacity,
                  views_.prefill_a4_intermediate_scales,
                  intermediate_scale_capacity, stream_),
              "prefill_projection_span_mlp_k512_gateup_down_edge_m64n128_k256_alternating")) {
        return failure;
      }
      ++g_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission_hits;
    } else
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
    if (gateup_down_k512_edge_m128n64_requested) {
      if (!check_cuda(
              kernels::launch_sm87_a4w4_gateup_down_k512_edge_m128n64_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, gate_k512.weight,
                  gate_k512_weight_capacity, gate_k512.scales,
                  gate_k512_scale_capacity, up_k512.weight,
                  up_k512_weight_capacity, up_k512.scales,
                  up_k512_scale_capacity, token_count,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize,
                  down_k512.activation_clip_ratio,
                  views_.prefill_a4_intermediate_packed,
                  intermediate_packed_capacity,
                  views_.prefill_a4_intermediate_scales,
                  intermediate_scale_capacity, stream_),
              "prefill_projection_span_mlp_k512_gateup_down_edge_m128n64")) {
        return failure;
      }
      ++g_a4w4_gateup_down_k512_edge_m128n64_admission_hits;
    } else
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
    if (gateup_down_k512_edge_requested) {
      if (!check_cuda(
              kernels::launch_sm87_a4w4_gateup_down_k512_edge_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales,
                  hidden_scale_capacity, gate_k512.weight,
                  gate_k512_weight_capacity, gate_k512.scales,
                  gate_k512_scale_capacity, up_k512.weight,
                  up_k512_weight_capacity, up_k512.scales,
                  up_k512_scale_capacity, token_count,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize,
                  down_k512.activation_clip_ratio,
                  views_.prefill_a4_intermediate_packed,
                  intermediate_packed_capacity,
                  views_.prefill_a4_intermediate_scales,
                  intermediate_scale_capacity, stream_),
              "prefill_projection_span_mlp_k512_gateup_down_edge")) {
        return failure;
      }
      ++g_a4w4_gateup_down_k512_edge_admission_hits;
    } else
#endif
    {
      if (!check_cuda(
              kernels::launch_sm87_a4w4_gateup_k512_macrocell_bf16_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                  gate_k512.weight, gate_k512_weight_capacity,
                  gate_k512.scales, gate_k512_scale_capacity,
                  up_k512.weight, up_k512_weight_capacity,
                  up_k512.scales, up_k512_scale_capacity,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize, 0U, kPrimaryProductColumns,
                  primary, kPrimaryProductColumns, primary_capacity,
                  stream_),
              "prefill_projection_span_mlp_k512_gate_up_primary") ||
          !check_cuda(
              kernels::launch_sm87_a4w4_gateup_k512_macrocell_bf16_cuda(
                  views_.prefill_a4_hidden_packed,
                  hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                  gate_k512.weight, gate_k512_weight_capacity,
                  gate_k512.scales, gate_k512_scale_capacity,
                  up_k512.weight, up_k512_weight_capacity,
                  up_k512.scales, up_k512_scale_capacity,
                  mlp_launch_token_count, kReferenceIntermediateSize,
                  kReferenceHiddenSize, kPrimaryProductColumns,
                  kSecondaryProductColumns, secondary,
                  kRequestLongPrefillSecondaryWidth, secondary_capacity,
                  stream_),
              "prefill_projection_span_mlp_k512_gate_up_secondary") ||
          !check_cuda(
              kernels::launch_sm87_a4_quantize_bf16_k512_split_cuda(
                  primary, kPrimaryProductColumns,
                  kPrimaryProductColumns, secondary,
                  kRequestLongPrefillSecondaryWidth,
                  kSecondaryProductColumns, token_count,
                  mlp_launch_token_count,
                  down_k512.activation_clip_ratio,
                  views_.prefill_a4_intermediate_packed,
                  intermediate_packed_capacity,
                  views_.prefill_a4_intermediate_scales,
                  intermediate_scale_capacity, stream_),
              "prefill_projection_span_mlp_k512_product_quantize")) {
        return failure;
      }
    }
    int down_status = static_cast<int>(cudaErrorInvalidValue);
    const char* down_stage = nullptr;
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
    if (down_m128n128_ldmatrix_pairring_v1_route ==
        reference_runner_detail::
            A4W4DownK512M128N128LdmatrixPairringV1Route::kEnabled) {
      down_status = kernels::
          launch_sm87_a4w4_down_k512_m128n128_ldmatrix_pairring_bf16_cuda(
              views_.prefill_a4_intermediate_packed,
              intermediate_packed_capacity,
              views_.prefill_a4_intermediate_scales,
              intermediate_scale_capacity, down_k512.weight,
              down_k512_weight_capacity, down_k512.scales,
              down_k512_scale_capacity, mlp_launch_token_count,
              kReferenceHiddenSize, kReferenceIntermediateSize,
              secondary, kReferenceHiddenSize, secondary_capacity,
              stream_);
      down_stage =
          "prefill_projection_span_mlp_k512_down_m128n128_ldmatrix_pairring";
    } else
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
    if (down_k512_m16n64_v2_requested) {
      down_status =
          kernels::launch_sm87_a4w4_down_k512_m16n64_v2_bf16_cuda(
              views_.prefill_a4_intermediate_packed,
              intermediate_packed_capacity,
              views_.prefill_a4_intermediate_scales,
              intermediate_scale_capacity, down_k512.weight,
              down_k512_weight_capacity, down_k512.scales,
              down_k512_scale_capacity, mlp_launch_token_count,
              kReferenceHiddenSize, kReferenceIntermediateSize,
              secondary, kReferenceHiddenSize, secondary_capacity,
              stream_);
      down_stage =
          "prefill_projection_span_mlp_k512_down_m16n64_v2";
    } else
#endif
    {
      down_status =
          kernels::launch_sm87_a4w4_down_k512_macrocell_bf16_cuda(
                views_.prefill_a4_intermediate_packed,
                intermediate_packed_capacity,
                views_.prefill_a4_intermediate_scales,
                intermediate_scale_capacity, down_k512.weight,
                down_k512_weight_capacity, down_k512.scales,
                down_k512_scale_capacity, mlp_launch_token_count,
                kReferenceHiddenSize, kReferenceIntermediateSize,
                secondary, kReferenceHiddenSize, secondary_capacity,
                stream_);
      down_stage = "prefill_projection_span_mlp_k512_down";
    }
    if (!check_cuda(down_status, down_stage)) {
      return failure;
    }
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
    if (down_k512_m16n64_v2_requested) {
      ++g_a4w4_down_k512_m16n64_v2_admission_hits;
    }
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
    if (down_m128n128_ldmatrix_pairring_v1_route ==
        reference_runner_detail::
            A4W4DownK512M128N128LdmatrixPairringV1Route::kEnabled) {
      ++g_a4w4_down_k512_m128n128_ldmatrix_pairring_admission_hits;
    }
#endif
    local_hits.activation_quantize_hits += 2U;
    g_a4w4_full_prefill_admission_hits.activation_quantize_hits += 2U;
    ++local_hits.paired_gate_up_hits;
    ++local_hits.generic_projection_hits;
    local_hits.logical_projection_hits += 3U;
    ++g_a4w4_full_prefill_admission_hits.paired_gate_up_hits;
    ++g_a4w4_full_prefill_admission_hits.generic_projection_hits;
    g_a4w4_full_prefill_admission_hits.logical_projection_hits += 3U;
    ++g_a4w4_mlp_k512_admission_hits;
    mlp_k512_selected = true;
  }
#endif
  if (!mlp_k512_selected) {
    if (!quantize(primary, kReferenceHiddenSize, token_count,
                  kReferenceHiddenSize, gate_sidecar.activation_clip_ratio,
                  views_.prefill_a4_hidden_packed, hidden_packed_capacity,
                  views_.prefill_a4_hidden_scales, hidden_scale_capacity,
                  "prefill_projection_span_mlp_input_quantize")) {
      return failure;
    }
    const int paired_status = launch_selected_a4w4_gateup_paired(
        a4w4_prefill_consumer_, views_.prefill_a4_hidden_packed,
        hidden_packed_capacity, views_.prefill_a4_hidden_scales,
        hidden_scale_capacity, gate_sidecar, gate_weight_capacity,
        gate_scale_capacity, up_sidecar, up_weight_capacity,
        up_scale_capacity, projection_token_count,
        down_sidecar.activation_clip_ratio,
        views_.prefill_a4_intermediate_packed,
        intermediate_packed_capacity,
        views_.prefill_a4_intermediate_scales, intermediate_scale_capacity,
        stream_, &projection_v3_selected, &complete_cell_v2_selected);
    if (!check_cuda(paired_status,
                    "prefill_projection_span_mlp_gate_up_paired") ||
        !zero_projection_padding(
            token_count, kReferenceIntermediateSize,
            views_.prefill_a4_intermediate_packed,
            intermediate_packed_capacity,
            views_.prefill_a4_intermediate_scales,
            intermediate_scale_capacity,
            "prefill_projection_span_mlp_gate_up_padding")) {
      return failure;
    }
    ++local_hits.paired_gate_up_hits;
    local_hits.logical_projection_hits += 2U;
    ++g_a4w4_full_prefill_admission_hits.paired_gate_up_hits;
    g_a4w4_full_prefill_admission_hits.logical_projection_hits += 2U;
    if (!projection_v3_selected && !complete_cell_v2_selected &&
        reference_runner_detail::a4w4_m128_stage_major_common_route(
            g_enable_a4w4_m128_stage_major_admission,
            a4w4_prefill_consumer_, projection_token_count)) {
      ++local_hits.m128_stage_major_paired_gate_up_hits;
      ++g_a4w4_full_prefill_admission_hits
            .m128_stage_major_paired_gate_up_hits;
    }
    if (!project(layer_weights.mlp.down_proj,
                 views_.prefill_a4_intermediate_packed,
                 intermediate_packed_capacity,
                 views_.prefill_a4_intermediate_scales,
                 intermediate_scale_capacity, projection_token_count,
                 secondary,
                 "prefill_projection_span_mlp_down")) {
      return failure;
    }
  }
  if (!check_cuda(
          launch_residual_add_reference_cuda(
              output_hidden, secondary,
              token_count * kReferenceHiddenSize, output_hidden, stream_),
          "prefill_projection_span_mlp_residual")) {
    return failure;
  }

  const bool linear = layer_type == model::LayerType::kLinearAttention;
  std::size_t expected_generic = linear ? 4U : 5U;
  const std::size_t expected_logical = linear ? 6U : 7U;
  const std::size_t expected_attention_linear = linear ? 1U : 0U;
  const std::size_t expected_attention_full = linear ? 0U : 1U;
  const std::size_t expected_attention_output = 1U;
  const std::size_t expected_attention_input_logical = linear ? 2U : 3U;
  std::size_t observed_attention_linear = 0U;
  std::size_t observed_attention_full = 0U;
  std::size_t observed_attention_output = 0U;
  std::size_t observed_attention_logical = 0U;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  observed_attention_linear +=
      local_attention_supermatrix_hits.linear_input_launch_hits;
  observed_attention_full +=
      local_attention_supermatrix_hits.full_input_launch_hits;
  observed_attention_output +=
      local_attention_supermatrix_hits.output_launch_hits;
  observed_attention_logical +=
      local_attention_supermatrix_hits.logical_projection_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
  observed_attention_linear +=
      local_attention_k256_hits.linear_input_launch_hits;
  observed_attention_full +=
      local_attention_k256_hits.full_input_launch_hits;
  observed_attention_output +=
      local_attention_k256_hits.output_launch_hits;
  observed_attention_logical +=
      local_attention_k256_hits.logical_projection_hits;
#endif
  const bool attention_supermatrix_inputs_all =
      observed_attention_linear == expected_attention_linear &&
      observed_attention_full == expected_attention_full &&
      observed_attention_logical ==
          expected_attention_input_logical + observed_attention_output;
  const bool attention_supermatrix_inputs_none =
      observed_attention_linear == 0U && observed_attention_full == 0U &&
      observed_attention_logical == observed_attention_output;
  const bool attention_supermatrix_output_all =
      observed_attention_output == expected_attention_output;
  const bool attention_supermatrix_output_none =
      observed_attention_output == 0U;
  const bool attention_o_k512_all = attention_o_k512_hits == 1U;
  const bool attention_o_k512_none = attention_o_k512_hits == 0U;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  const bool incumbent_attention_inputs_all =
      local_attention_supermatrix_hits.linear_input_launch_hits ==
          expected_attention_linear &&
      local_attention_supermatrix_hits.full_input_launch_hits ==
          expected_attention_full &&
      local_attention_supermatrix_hits.logical_projection_hits ==
          expected_attention_input_logical +
              local_attention_supermatrix_hits.output_launch_hits;
  const bool incumbent_attention_output_all =
      local_attention_supermatrix_hits.output_launch_hits ==
      expected_attention_output;
  const bool requested_attention_supermatrix_selected =
      !g_enable_a4w4_attention_supermatrix_admission ||
      (incumbent_attention_inputs_all &&
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
       (g_enable_a4w4_attention_o_k512_admission
            ? attention_o_k512_all
            : incumbent_attention_output_all));
#else
       incumbent_attention_output_all);
#endif
#else
  constexpr bool requested_attention_supermatrix_selected = true;
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
  const bool k256_attention_inputs_all =
      local_attention_k256_hits.linear_input_launch_hits ==
          expected_attention_linear &&
      local_attention_k256_hits.full_input_launch_hits ==
          expected_attention_full &&
      local_attention_k256_hits.logical_projection_hits ==
          expected_attention_input_logical +
              local_attention_k256_hits.output_launch_hits;
  const bool k256_attention_output_all =
      local_attention_k256_hits.output_launch_hits ==
      expected_attention_output;
  const bool requested_attention_k256_selected =
      !g_enable_a4w4_attention_k256_m128n256_admission ||
      (k256_attention_inputs_all && k256_attention_output_all);
#else
  constexpr bool requested_attention_k256_selected = true;
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
  const bool requested_attention_o_k512_selected =
      !g_enable_a4w4_attention_o_k512_admission || attention_o_k512_all;
#else
  constexpr bool requested_attention_o_k512_selected = true;
#endif
  if (attention_supermatrix_inputs_all) {
    expected_generic -= expected_attention_input_logical;
  }
  if (attention_supermatrix_output_all || attention_o_k512_all) {
    expected_generic -= expected_attention_output;
  }
  const bool expect_m128_stage_major =
      reference_runner_detail::a4w4_m128_stage_major_common_route(
          g_enable_a4w4_m128_stage_major_admission,
          a4w4_prefill_consumer_, projection_token_count);
  const bool expect_down_m128_stage_major =
      reference_runner_detail::a4w4_m128_stage_major_common_route(
          g_enable_a4w4_down_m128_stage_major_admission,
          a4w4_prefill_consumer_, projection_token_count);
  bool expect_down_complete_cell_v2 = false;
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION) || \
    defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
  reference_runner_detail::A4W4DownCompleteCellV2RouteQuery down_cell_query;
  down_cell_query.inventory_consumer = a4w4_prefill_consumer_;
  down_cell_query.projection_token_count = projection_token_count;
  down_cell_query.output_size = kReferenceHiddenSize;
  down_cell_query.input_size = kReferenceIntermediateSize;
  down_cell_query.packed_input_capacity_bytes = intermediate_packed_capacity;
  down_cell_query.input_scale_capacity_elements = intermediate_scale_capacity;
  down_cell_query.weight_capacity_bytes =
      kernels::sm87_a4w4_consumer_packed_capacity_bytes(
          kReferenceHiddenSize, kReferenceIntermediateSize);
  down_cell_query.weight_scale_capacity_elements =
      a4w4_scale_capacity_elements(
          a4w4_prefill_consumer_, kReferenceHiddenSize,
          kReferenceIntermediateSize);
  down_cell_query.output_capacity_elements =
      static_cast<std::size_t>(
          request_plan.long_prefill_projection_secondary_bf16
              .element_capacity);
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V3_ADMISSION)
  down_cell_query.admission_enabled =
      g_enable_a4w4_down_complete_cell_v3_admission &&
      !mlp_k512_selected;
  expect_down_complete_cell_v2 =
      reference_runner_detail::use_a4w4_down_complete_cell_v3_route(
          down_cell_query);
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_COMPLETE_CELL_V2_ADMISSION)
  down_cell_query.admission_enabled =
      g_enable_a4w4_down_complete_cell_v2_admission &&
      !mlp_k512_selected;
  expect_down_complete_cell_v2 =
      expect_down_complete_cell_v2 ||
      reference_runner_detail::use_a4w4_down_complete_cell_v2_route(
          down_cell_query);
#endif
#endif
  if (!requested_attention_supermatrix_selected ||
      !requested_attention_k256_selected ||
      !requested_attention_o_k512_selected ||
      (!attention_supermatrix_inputs_all &&
       !attention_supermatrix_inputs_none) ||
      (!attention_supermatrix_output_all &&
       !attention_supermatrix_output_none) ||
      (!attention_o_k512_all && !attention_o_k512_none) ||
      (attention_supermatrix_output_all && attention_o_k512_all) ||
      local_hits.activation_quantize_hits !=
          (mlp_k512_selected ? 4U : 3U) ||
      local_hits.generic_projection_hits != expected_generic ||
      local_hits.paired_gate_up_hits != 1U ||
      local_hits.logical_projection_hits != expected_logical ||
      down_complete_cell_v2_hits !=
          (expect_down_complete_cell_v2 ? 1U : 0U) ||
      local_hits.m128_stage_major_generic_projection_hits !=
          (expect_m128_stage_major ? expected_generic - 1U : 0U) ||
      local_hits.m128_stage_major_down_projection_hits !=
          (expect_down_m128_stage_major && !mlp_k512_selected &&
                   !expect_down_complete_cell_v2
               ? 1U
               : 0U) ||
      local_hits.m128_stage_major_paired_gate_up_hits !=
          (expect_m128_stage_major && !mlp_k512_selected &&
                   !projection_v3_selected &&
                   !complete_cell_v2_selected
               ? 1U
               : 0U)) {
    return runner_status(ReferenceRunnerError::kInvalidRunner,
                         "prefill_projection_span_a4_accounting",
                         item.layer_index);
  }

  if (item.layer_index + 1U == kReferenceDecoderLayerCount &&
      item.last_projection_span_for_layer) {
    const std::size_t retained_row =
        (plan.prompt_token_count - 1U) %
        kLongPrefillLayerMajorTileTokens;
    if (!check_cuda(
            launch_headwise_centered_rms_norm_reference_cuda(
                output_hidden + (token_count - 1U) * kReferenceHiddenSize,
                weights_->final_norm().data, 1U, kReferenceHiddenSize,
                kRmsEpsilon,
                views_.hidden[1] + retained_row * kReferenceHiddenSize,
                stream_),
            "prefill_projection_span_final_norm")) {
      return failure;
    }
  }
  return {};
#endif
}

ReferenceLongPrefillOutcome ReferenceRunner::prefill_layer_major_prompt(
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const bool measure_timing) noexcept {
  using Clock = std::chrono::steady_clock;
  const Clock::time_point started = Clock::now();
  retained_prefill_hidden_valid_ = false;

  if (!static_cast<bool>(*this)) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_layer_major_prompt"));
  }
  if (poisoned_) {
    ReferenceLongPrefillOutcome outcome;
    outcome.status = runner_status(ReferenceRunnerError::kPoisoned,
                                   "prefill_layer_major_prompt");
    return outcome;
  }
  if (input_token_ids == nullptr || token_count == 0U ||
      token_count > kLongPrefillLayerMajorMaximumTokens ||
      token_count > std::numeric_limits<std::uint32_t>::max()) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kTokenOutOfRange,
        "prefill_layer_major_tokens"));
  }
  for (std::size_t token = 0U; token < token_count; ++token) {
    if (input_token_ids[token] >= kReferenceVocabularySize) {
      return fail_long_prefill(runner_status(
          ReferenceRunnerError::kTokenOutOfRange,
          "prefill_layer_major_token"));
    }
  }

  const std::uint32_t prompt_token_count =
      static_cast<std::uint32_t>(token_count);
  const std::size_t hidden_buffer_count =
      views_.long_prefill_hidden[0] != nullptr &&
              views_.long_prefill_hidden[1] != nullptr
          ? kRequestLongPrefillHiddenBufferCount
          : 0U;
  LongPrefillLayerMajorRouteQuery route_query;
  route_query.runtime_enabled = true;
  route_query.short_prompt_admission_enabled =
      short_prefill_layer_major_environment_enabled();
  route_query.authenticated_a4_prefill =
      authenticated_a4w4_prefill_enabled();
  route_query.projection_backend = projection_backend_;
  route_query.capture_trace = trace_enabled_;
  route_query.prompt_token_count = prompt_token_count;
  route_query.prefill_chunk_size = state_->plan().prefill_chunk_size;
  route_query.hidden_token_capacity =
      state_->plan().long_prefill_token_capacity;
  route_query.hidden_buffer_count = hidden_buffer_count;
  if (state_->current_position() != 0U ||
      token_count > state_->remaining_capacity() ||
      select_long_prefill_layer_major_route(route_query) !=
          LongPrefillLayerMajorRoute::kLayerMajorAdmission) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "prefill_layer_major_route"));
  }

#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  const std::uint32_t projection_span_capacity =
      state_->plan().long_prefill_projection_span_capacity;
  const bool a4_route_requested =
      reference_runner_detail::use_a4w4_full_prefill_tile_route(
          a4w4_full_prefill_admission_enabled_ ||
              g_enable_a4w4_full_prefill_admission,
          trace_enabled_, optimized_prefill_dispatch_disabled());
  if (a4_route_requested &&
      (a4w4_prefill_consumer_ ==
           reference_runner_detail::A4W4PrefillConsumer::kK128 ||
       a4w4_prefill_consumer_ ==
           reference_runner_detail::A4W4PrefillConsumer::kK256) &&
      !reference_runner_detail::
          a4w4_prefill_consumer_supports_projection_span_prompt(
              a4w4_prefill_consumer_, prompt_token_count,
              projection_span_capacity)) {
    // One runner owns one authenticated A4 publication.  Reject before any
    // layer launch unless every full/final projection span can be represented
    // by the shared-scale padded-M contract; never mix consumer ABIs.
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "prefill_a4w4_shared_scale_token_alignment"));
  }
  const bool a4_inventory_enabled =
      a4_route_requested &&
      reference_runner_detail::
          a4w4_prefill_consumer_supports_projection_span_prompt(
              a4w4_prefill_consumer_, prompt_token_count,
              projection_span_capacity);
  const bool projection_span_selected =
      use_long_prefill_projection_span_route(
          prompt_token_count, projection_span_capacity,
          a4_inventory_enabled,
          optimized_prefill_dispatch_disabled(),
          route_query.short_prompt_admission_enabled,
          route_query.authenticated_a4_prefill);
  if (projection_span_selected) {
    LongPrefillProjectionSpanOptions span_options;
    span_options.prompt_token_count = prompt_token_count;
    span_options.hidden_token_capacity =
        state_->plan().long_prefill_token_capacity;
    span_options.projection_span_token_count = projection_span_capacity;
    span_options.state_tile_token_count =
        kLongPrefillLayerMajorTileTokens;
    const LongPrefillProjectionSpanPlanResult span_plan =
        build_long_prefill_projection_span_plan(span_options);
    if (!span_plan) {
      return fail_long_prefill(runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "prefill_projection_span_plan"));
    }

    const auto hits_before = g_a4w4_full_prefill_admission_hits;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
    const std::size_t attention_o_k512_hits_before =
        g_a4w4_attention_o_k512_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
    const std::size_t mlp_k512_hits_before =
        g_a4w4_mlp_k512_admission_hits;
    const std::size_t
        gateup_m128n512_paired_ldmatrix_hits_before =
            g_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_admission_hits;
    const std::size_t down_m128n128_ldmatrix_pairring_hits_before =
        g_a4w4_down_k512_m128n128_ldmatrix_pairring_admission_hits;
    const reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute
        request_paired_gateup_canonical_down_route =
            reference_runner_detail::
                select_a4w4_paired_gateup_canonical_down_route(
                    a4w4_paired_gateup_canonical_down_selector_query(true));
    const reference_runner_detail::
        A4W4DownK512M128N128LdmatrixPairringV1Route
            request_down_m128n128_ldmatrix_pairring_v1_route =
                reference_runner_detail::
                    select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
                        a4w4_down_k512_m128n128_ldmatrix_pairring_v1_selector_query(
                            true));
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
    const std::size_t gateup_down_k512_edge_hits_before =
        g_a4w4_gateup_down_k512_edge_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
    const std::size_t
        gateup_down_k512_edge_m64n128_k256_alternating_hits_before =
            g_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
    const std::size_t gateup_down_k512_edge_m128n64_hits_before =
        g_a4w4_gateup_down_k512_edge_m128n64_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
    const std::size_t down_k512_m16n64_v2_hits_before =
        g_a4w4_down_k512_m16n64_v2_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
    const std::size_t mlp_k512_fragment_native_hits_before =
        g_a4w4_mlp_k512_fragment_native_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    const auto attention_hits_before =
        g_a4w4_attention_supermatrix_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
    const auto attention_k256_hits_before =
        g_a4w4_attention_k256_m128n256_admission_hits;
#endif
    auto* const device_token_ids =
        reinterpret_cast<std::uint32_t*>(views_.projection[3]);
    const auto stream = reinterpret_cast<cudaStream_t>(stream_);
    for (std::uint32_t first = 0U; first < prompt_token_count;
         first += kLongPrefillLayerMajorTileTokens) {
      const std::uint32_t count =
          std::min(kLongPrefillLayerMajorTileTokens,
                   prompt_token_count - first);
      cudaError_t status = cudaMemcpyAsync(
          device_token_ids, input_token_ids + first,
          static_cast<std::size_t>(count) * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream);
      if (status == cudaSuccess) {
        status = static_cast<cudaError_t>(
            launch_embedding_gather_prompt_reference_cuda(
                weights_->embed_tokens().weight,
                kReferenceVocabularySize, kReferenceHiddenSize,
                device_token_ids, count,
                views_.long_prefill_hidden[0] +
                    static_cast<std::size_t>(first) *
                        kReferenceHiddenSize,
                stream_));
      }
      if (status != cudaSuccess) {
        return fail_long_prefill(runner_status(
            ReferenceRunnerError::kCudaFailure,
            "prefill_projection_span_embedding", kReferenceNoLayer,
            static_cast<int>(status)));
      }
    }

    for (std::size_t ordinal = 0U;
         ordinal < span_plan.value->work_item_count; ++ordinal) {
      LongPrefillProjectionSpanWorkItem item;
      if (!long_prefill_projection_span_work_item(
              *span_plan.value, ordinal, item)) {
        return fail_long_prefill(runner_status(
            ReferenceRunnerError::kInvalidStepOptions,
            "prefill_projection_span_work_item"));
      }
      const ReferenceRunnerStatus executed =
          execute_long_prefill_projection_span(*span_plan.value, item);
      if (!executed) {
        return fail_long_prefill(executed);
      }
    }

    const cudaError_t sync_status = cudaStreamSynchronize(stream);
    if (sync_status != cudaSuccess) {
      return fail_long_prefill(runner_status(
          ReferenceRunnerError::kCudaFailure,
          "prefill_projection_span_synchronize", kReferenceNoLayer,
          static_cast<int>(sync_status)));
    }
    const auto& hits_after = g_a4w4_full_prefill_admission_hits;
    const std::size_t spans = span_plan.value->projection_span_count;
    const auto exact_delta = [spans](
                                 const std::size_t current,
                                 const std::size_t initial,
                                 const std::size_t per_span) noexcept {
      return current >= initial &&
             current - initial == spans * per_span;
    };
    std::size_t attention_logical_delta = 0U;
    std::size_t attention_supermatrix_output_delta = 0U;
    bool attention_accounting_valid = true;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
    attention_accounting_valid =
        a4w4_attention_supermatrix_aggregate_delta(
            attention_hits_before,
            g_a4w4_attention_supermatrix_admission_hits, spans,
            attention_logical_delta,
            attention_supermatrix_output_delta);
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
    std::size_t attention_k256_logical_delta = 0U;
    std::size_t attention_k256_output_delta = 0U;
    attention_accounting_valid =
        attention_accounting_valid &&
        a4w4_attention_supermatrix_aggregate_delta(
            attention_k256_hits_before,
            g_a4w4_attention_k256_m128n256_admission_hits, spans,
            attention_k256_logical_delta, attention_k256_output_delta);
    attention_logical_delta += attention_k256_logical_delta;
    attention_supermatrix_output_delta += attention_k256_output_delta;
#endif
    std::size_t attention_o_k512_delta = 0U;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
    if (g_a4w4_attention_o_k512_admission_hits <
        attention_o_k512_hits_before) {
      attention_accounting_valid = false;
    } else {
      attention_o_k512_delta =
          g_a4w4_attention_o_k512_admission_hits -
          attention_o_k512_hits_before;
      attention_accounting_valid =
          attention_accounting_valid &&
          (attention_o_k512_delta == 0U ||
           attention_o_k512_delta == spans * 64U);
    }
#endif
    attention_accounting_valid =
        attention_accounting_valid &&
        (attention_supermatrix_output_delta == 0U ||
         attention_o_k512_delta == 0U);
    attention_logical_delta += attention_o_k512_delta;
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
    const std::size_t mlp_k512_v1_delta =
        g_a4w4_mlp_k512_admission_hits >= mlp_k512_hits_before
            ? g_a4w4_mlp_k512_admission_hits - mlp_k512_hits_before
            : 0U;
    const bool mlp_k512_v1_accounting_valid =
        g_a4w4_mlp_k512_admission_hits >= mlp_k512_hits_before &&
        (mlp_k512_v1_delta == 0U ||
         mlp_k512_v1_delta == spans * 64U) &&
        (!g_enable_a4w4_mlp_k512_admission ||
         mlp_k512_v1_delta == spans * 64U);
#else
    constexpr std::size_t mlp_k512_v1_delta = 0U;
    constexpr bool mlp_k512_v1_accounting_valid = true;
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
    const std::size_t gateup_m128n512_paired_ldmatrix_delta =
        g_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_admission_hits >=
                gateup_m128n512_paired_ldmatrix_hits_before
            ? g_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_admission_hits -
                  gateup_m128n512_paired_ldmatrix_hits_before
            : 0U;
    const std::size_t down_m128n128_ldmatrix_pairring_delta =
        g_a4w4_down_k512_m128n128_ldmatrix_pairring_admission_hits >=
                down_m128n128_ldmatrix_pairring_hits_before
            ? g_a4w4_down_k512_m128n128_ldmatrix_pairring_admission_hits -
                  down_m128n128_ldmatrix_pairring_hits_before
            : 0U;
    const bool down_m128n128_ldmatrix_pairring_v1_selected =
        request_down_m128n128_ldmatrix_pairring_v1_route ==
        reference_runner_detail::
            A4W4DownK512M128N128LdmatrixPairringV1Route::kEnabled;
    const std::size_t down_m128n128_ldmatrix_pairring_hybrid_hits_after =
        down_m128n128_ldmatrix_pairring_v1_selected
            ? down_m128n128_ldmatrix_pairring_hits_before
            : g_a4w4_down_k512_m128n128_ldmatrix_pairring_admission_hits;
    const bool paired_gateup_canonical_down_accounting_valid =
        reference_runner_detail::
            a4w4_paired_gateup_canonical_down_accounting_valid(
                request_paired_gateup_canonical_down_route, spans,
                gateup_m128n512_paired_ldmatrix_hits_before,
                g_a4w4_gateup_down_k512_edge_m128n512_paired_ldmatrix_admission_hits,
                down_m128n128_ldmatrix_pairring_hits_before,
                down_m128n128_ldmatrix_pairring_hybrid_hits_after);
    const bool down_m128n128_ldmatrix_pairring_v1_accounting_valid =
        reference_runner_detail::
            a4w4_down_k512_m128n128_ldmatrix_pairring_v1_accounting_valid(
                request_down_m128n128_ldmatrix_pairring_v1_route, spans,
                mlp_k512_v1_delta,
                down_m128n128_ldmatrix_pairring_hits_before,
                request_paired_gateup_canonical_down_route ==
                        reference_runner_detail::
                            A4W4PairedGateUpCanonicalDownRoute::kGateAndDown
                    ? down_m128n128_ldmatrix_pairring_hits_before
                    : g_a4w4_down_k512_m128n128_ldmatrix_pairring_admission_hits);
#else
    constexpr std::size_t gateup_m128n512_paired_ldmatrix_delta = 0U;
    constexpr std::size_t down_m128n128_ldmatrix_pairring_delta = 0U;
    constexpr bool paired_gateup_canonical_down_accounting_valid = true;
    constexpr bool
        down_m128n128_ldmatrix_pairring_v1_accounting_valid = true;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
    const std::size_t gateup_down_k512_edge_delta =
        g_a4w4_gateup_down_k512_edge_admission_hits >=
                gateup_down_k512_edge_hits_before
            ? g_a4w4_gateup_down_k512_edge_admission_hits -
                  gateup_down_k512_edge_hits_before
            : 0U;
    const bool gateup_down_k512_edge_accounting_valid =
        g_a4w4_gateup_down_k512_edge_admission_hits >=
            gateup_down_k512_edge_hits_before &&
        (gateup_down_k512_edge_delta == 0U ||
         gateup_down_k512_edge_delta == spans * 64U) &&
        (!g_enable_a4w4_gateup_down_k512_edge_admission ||
         gateup_down_k512_edge_delta == spans * 64U) &&
        (gateup_down_k512_edge_delta == 0U ||
         gateup_down_k512_edge_delta == mlp_k512_v1_delta);
#else
    constexpr std::size_t gateup_down_k512_edge_delta = 0U;
    constexpr bool gateup_down_k512_edge_accounting_valid = true;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
    const std::size_t
        gateup_down_k512_edge_m64n128_k256_alternating_delta =
            g_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission_hits >=
                    gateup_down_k512_edge_m64n128_k256_alternating_hits_before
                ? g_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission_hits -
                      gateup_down_k512_edge_m64n128_k256_alternating_hits_before
                : 0U;
    const bool
        gateup_down_k512_edge_m64n128_k256_alternating_accounting_valid =
            g_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission_hits >=
                gateup_down_k512_edge_m64n128_k256_alternating_hits_before &&
            (gateup_down_k512_edge_m64n128_k256_alternating_delta == 0U ||
             gateup_down_k512_edge_m64n128_k256_alternating_delta ==
                 spans * 64U) &&
            (!g_enable_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission ||
             gateup_down_k512_edge_m64n128_k256_alternating_delta ==
                 spans * 64U) &&
            (gateup_down_k512_edge_m64n128_k256_alternating_delta == 0U ||
             gateup_down_k512_edge_m64n128_k256_alternating_delta ==
                 mlp_k512_v1_delta);
#else
    constexpr std::size_t
        gateup_down_k512_edge_m64n128_k256_alternating_delta = 0U;
    constexpr bool
        gateup_down_k512_edge_m64n128_k256_alternating_accounting_valid =
            true;
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
    const std::size_t gateup_down_k512_edge_m128n64_delta =
        g_a4w4_gateup_down_k512_edge_m128n64_admission_hits >=
                gateup_down_k512_edge_m128n64_hits_before
            ? g_a4w4_gateup_down_k512_edge_m128n64_admission_hits -
                  gateup_down_k512_edge_m128n64_hits_before
            : 0U;
    const bool gateup_down_k512_edge_m128n64_accounting_valid =
        g_a4w4_gateup_down_k512_edge_m128n64_admission_hits >=
            gateup_down_k512_edge_m128n64_hits_before &&
        (gateup_down_k512_edge_m128n64_delta == 0U ||
         gateup_down_k512_edge_m128n64_delta == spans * 64U) &&
        (!g_enable_a4w4_gateup_down_k512_edge_m128n64_admission ||
         gateup_down_k512_edge_m128n64_delta == spans * 64U) &&
        (gateup_down_k512_edge_m128n64_delta == 0U ||
         gateup_down_k512_edge_m128n64_delta == mlp_k512_v1_delta);
#else
    constexpr std::size_t gateup_down_k512_edge_m128n64_delta = 0U;
    constexpr bool gateup_down_k512_edge_m128n64_accounting_valid = true;
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
    const std::size_t down_k512_m16n64_v2_delta =
        g_a4w4_down_k512_m16n64_v2_admission_hits >=
                down_k512_m16n64_v2_hits_before
            ? g_a4w4_down_k512_m16n64_v2_admission_hits -
                  down_k512_m16n64_v2_hits_before
            : 0U;
    const bool down_k512_m16n64_v2_accounting_valid =
        g_a4w4_down_k512_m16n64_v2_admission_hits >=
            down_k512_m16n64_v2_hits_before &&
        (down_k512_m16n64_v2_delta == 0U ||
         down_k512_m16n64_v2_delta == spans * 64U) &&
        (!g_enable_a4w4_down_k512_m16n64_v2_admission ||
         down_k512_m16n64_v2_delta == spans * 64U) &&
        (down_k512_m16n64_v2_delta == 0U ||
         down_k512_m16n64_v2_delta == mlp_k512_v1_delta);
#else
    constexpr bool down_k512_m16n64_v2_accounting_valid = true;
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
    const std::size_t mlp_k512_fragment_native_delta =
        g_a4w4_mlp_k512_fragment_native_admission_hits >=
                mlp_k512_fragment_native_hits_before
            ? g_a4w4_mlp_k512_fragment_native_admission_hits -
                  mlp_k512_fragment_native_hits_before
            : 0U;
    const bool mlp_k512_fragment_native_accounting_valid =
        g_a4w4_mlp_k512_fragment_native_admission_hits >=
            mlp_k512_fragment_native_hits_before &&
        (mlp_k512_fragment_native_delta == 0U ||
         mlp_k512_fragment_native_delta == spans * 64U) &&
        (!g_enable_a4w4_mlp_k512_fragment_native_admission ||
         mlp_k512_fragment_native_delta == spans * 64U);
#else
    constexpr std::size_t mlp_k512_fragment_native_delta = 0U;
    constexpr bool mlp_k512_fragment_native_accounting_valid = true;
#endif
    const std::size_t mlp_k512_delta =
        mlp_k512_v1_delta + mlp_k512_fragment_native_delta +
        gateup_m128n512_paired_ldmatrix_delta;
    const bool mlp_k512_accounting_valid =
        mlp_k512_v1_accounting_valid &&
        paired_gateup_canonical_down_accounting_valid &&
        down_m128n128_ldmatrix_pairring_v1_accounting_valid &&
        gateup_down_k512_edge_accounting_valid &&
        gateup_down_k512_edge_m64n128_k256_alternating_accounting_valid &&
        gateup_down_k512_edge_m128n64_accounting_valid &&
        down_k512_m16n64_v2_accounting_valid &&
        mlp_k512_fragment_native_accounting_valid &&
        (gateup_down_k512_edge_delta == 0U ||
         gateup_down_k512_edge_m128n64_delta == 0U) &&
        (gateup_down_k512_edge_m64n128_k256_alternating_delta == 0U ||
         gateup_down_k512_edge_delta == 0U) &&
        (gateup_down_k512_edge_m64n128_k256_alternating_delta == 0U ||
         gateup_down_k512_edge_m128n64_delta == 0U) &&
        (mlp_k512_v1_delta == 0U ||
         mlp_k512_fragment_native_delta == 0U) &&
        (gateup_m128n512_paired_ldmatrix_delta == 0U ||
         (mlp_k512_v1_delta == 0U &&
          mlp_k512_fragment_native_delta == 0U &&
          gateup_down_k512_edge_delta == 0U &&
          gateup_down_k512_edge_m64n128_k256_alternating_delta == 0U &&
          gateup_down_k512_edge_m128n64_delta == 0U));
    const bool activation_accounting_valid =
        hits_after.activation_quantize_hits >=
            hits_before.activation_quantize_hits &&
        hits_after.activation_quantize_hits -
                hits_before.activation_quantize_hits ==
            spans * 192U + mlp_k512_delta;
    const bool generic_accounting_valid =
        hits_after.generic_projection_hits >=
            hits_before.generic_projection_hits &&
        hits_after.generic_projection_hits -
                    hits_before.generic_projection_hits +
                attention_logical_delta ==
            spans * 272U;
    if (!activation_accounting_valid ||
        !attention_accounting_valid || !mlp_k512_accounting_valid ||
        !generic_accounting_valid ||
        !exact_delta(hits_after.paired_gate_up_hits,
                     hits_before.paired_gate_up_hits, 64U) ||
        !exact_delta(hits_after.logical_projection_hits,
                     hits_before.logical_projection_hits, 400U) ||
        hits_after.complete_model_tile_hits !=
            hits_before.complete_model_tile_hits) {
      return fail_long_prefill(runner_status(
          ReferenceRunnerError::kInvalidRunner,
          "prefill_projection_span_aggregate_accounting"));
    }
    const RequestOperationStatus commit =
        state_->set_sequence_length(prompt_token_count);
    if (!commit) {
      return fail_long_prefill(runner_status(
          ReferenceRunnerError::kStateCommitFailure,
          "prefill_projection_span_commit", kReferenceNoLayer,
          commit.cuda_error));
    }
    g_a4w4_full_prefill_admission_hits.complete_model_tile_hits += spans;
    retained_prefill_hidden_valid_ = true;
    retained_prefill_position_ = prompt_token_count - 1U;
    retained_prefill_input_token_ =
        input_token_ids[prompt_token_count - 1U];
    retained_prefill_hidden_row_ =
        (prompt_token_count - 1U) % kLongPrefillLayerMajorTileTokens;

    ReferenceLongPrefillResult value;
    value.first_position = 0U;
    value.token_count = token_count;
    value.gateup_alternating_launch_hits =
        gateup_down_k512_edge_m64n128_k256_alternating_delta;
    value.gateup_m128n512_paired_ldmatrix_launch_hits =
        gateup_m128n512_paired_ldmatrix_delta;
    value.down_m128n128_ldmatrix_pairring_launch_hits =
        down_m128n128_ldmatrix_pairring_delta;
    if (measure_timing) {
      const std::chrono::duration<double, std::milli> elapsed =
          Clock::now() - started;
      value.timing.emplace(ReferenceStepTiming{elapsed.count()});
    }
    ReferenceLongPrefillOutcome outcome;
    outcome.value.emplace(std::move(value));
    return outcome;
  }
#endif

#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  if (reference_runner_detail::select_a4w4_paired_gateup_canonical_down_route(
          a4w4_paired_gateup_canonical_down_selector_query(false)) !=
      reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
          kDisabled) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_paired_gateup_canonical_down_requires_projection_span"));
  }
  if (reference_runner_detail::
          select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
              a4w4_down_k512_m128n128_ldmatrix_pairring_v1_selector_query(
                  false)) !=
      reference_runner_detail::
          A4W4DownK512M128N128LdmatrixPairringV1Route::kDisabled) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_down_m128n128_ldmatrix_pairring_v1_requires_projection_span"));
  }
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M64N128_K256_ALTERNATING_ADMISSION)
  // The alternating edge candidate owns only the authenticated whole-M K512
  // projection-span executor.  Never accept its selector and then fall back
  // to a tiled or K128 route.
  if (g_enable_a4w4_gateup_down_k512_edge_m64n128_k256_alternating_admission) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_gateup_down_edge_m64n128_k256_alternating_requires_projection_span"));
  }
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N64_ADMISSION)
  // The structural edge candidate is admitted only by the authenticated
  // whole-M projection-span executor.  Never silently fall back to K128.
  if (g_enable_a4w4_gateup_down_k512_edge_m128n64_admission) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_gateup_down_edge_m128n64_requires_projection_span"));
  }
#endif
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_ADMISSION)
  // The edge candidate is a strict sub-route of authenticated v1 K512 and
  // owns only the whole-M projection-span executor.  Never accept its
  // selector and then fall through to a tiled or K128 implementation.
  if (g_enable_a4w4_gateup_down_k512_edge_admission) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_gateup_down_edge_requires_projection_span"));
  }
#endif
#if defined(Q3X_ENABLE_A4W4_DOWN_K512_M16N64_V2_ADMISSION)
  if (g_enable_a4w4_down_k512_m16n64_v2_admission) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_down_k512_m16n64_v2_requires_projection_span"));
  }
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  // The authenticated K512 MLP route currently owns only the whole-M
  // projection-span executor.  An enabled selector must never fall through
  // silently to the legacy K128 callback/tile executor.
  if (g_enable_a4w4_mlp_k512_admission) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_requires_projection_span"));
  }
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
  // The fragment-native route has the same whole-M ownership contract.  A
  // requested selector may not silently enter the legacy tiled executor.
  if (g_enable_a4w4_mlp_k512_fragment_native_admission) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidRunner,
        "prefill_mlp_k512_fragment_native_requires_projection_span"));
  }
#endif

  LongPrefillLayerMajorOptions plan_options;
  plan_options.prompt_token_count = prompt_token_count;
  plan_options.hidden_token_capacity =
      state_->plan().long_prefill_token_capacity;
  plan_options.tile_token_count = kLongPrefillLayerMajorTileTokens;
  const LongPrefillLayerMajorPlanResult built =
      build_long_prefill_layer_major_plan(plan_options);
  if (!built) {
    return fail_long_prefill(runner_status(
        ReferenceRunnerError::kInvalidStepOptions,
        "prefill_layer_major_plan"));
  }

  struct ExecutionContext {
    ReferenceRunner* runner = nullptr;
    const std::uint32_t* input_token_ids = nullptr;
    ReferenceRunnerStatus failure;
    bool a4w4_full_prefill_selected = false;
    std::size_t a4w4_complete_model_tiles = 0U;
    reference_runner_detail::A4W4FullPrefillAdmissionHits
        a4w4_hits_before{};
    std::size_t a4w4_attention_o_k512_hits_before = 0U;
    reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits
        a4w4_attention_hits_before{};
    reference_runner_detail::A4W4AttentionSupermatrixAdmissionHits
        a4w4_attention_k256_hits_before{};
  } context{this, input_token_ids, {}};
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  context.a4w4_full_prefill_selected = a4_inventory_enabled;
  context.a4w4_complete_model_tiles = built.value->tile_count;
  context.a4w4_hits_before = g_a4w4_full_prefill_admission_hits;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
  context.a4w4_attention_o_k512_hits_before =
      g_a4w4_attention_o_k512_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  context.a4w4_attention_hits_before =
      g_a4w4_attention_supermatrix_admission_hits;
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
  context.a4w4_attention_k256_hits_before =
      g_a4w4_attention_k256_m128n256_admission_hits;
#endif
#endif

  LongPrefillLayerMajorCallbacks callbacks;
  callbacks.context = &context;
  callbacks.prepare_hidden = +[](void* const opaque,
                                  const std::size_t output_hidden_buffer,
                                  const std::uint32_t tokens) noexcept {
    auto& execution = *static_cast<ExecutionContext*>(opaque);
    ReferenceRunner& runner = *execution.runner;
    if (output_hidden_buffer >= kRequestLongPrefillHiddenBufferCount ||
        runner.views_.long_prefill_hidden[output_hidden_buffer] == nullptr ||
        tokens == 0U) {
      execution.failure = runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_layer_major_prepare_hidden");
      return false;
    }
    auto* const device_token_ids =
        reinterpret_cast<std::uint32_t*>(runner.views_.projection[3]);
    const auto stream = reinterpret_cast<cudaStream_t>(runner.stream_);
    for (std::uint32_t first = 0U; first < tokens;
         first += kLongPrefillLayerMajorTileTokens) {
      const std::uint32_t remaining = tokens - first;
      const std::uint32_t tile_tokens =
          remaining < kLongPrefillLayerMajorTileTokens
              ? remaining
              : kLongPrefillLayerMajorTileTokens;
      cudaError_t status = cudaMemcpyAsync(
          device_token_ids, execution.input_token_ids + first,
          static_cast<std::size_t>(tile_tokens) * sizeof(std::uint32_t),
          cudaMemcpyHostToDevice, stream);
      if (status == cudaSuccess) {
        status = static_cast<cudaError_t>(
            launch_embedding_gather_prompt_reference_cuda(
                runner.weights_->embed_tokens().weight,
                kReferenceVocabularySize, kReferenceHiddenSize,
                device_token_ids, tile_tokens,
                runner.views_.long_prefill_hidden[output_hidden_buffer] +
                    static_cast<std::size_t>(first) * kReferenceHiddenSize,
                runner.stream_));
      }
      if (status != cudaSuccess) {
        execution.failure = runner_status(
            ReferenceRunnerError::kCudaFailure,
            "prefill_layer_major_embedding", kReferenceNoLayer,
            static_cast<int>(status));
        return false;
      }
    }
    return true;
  };
  callbacks.execute_tile = +[](
                               void* const opaque,
                               const LongPrefillLayerMajorWorkItem& item)
      noexcept {
    auto& execution = *static_cast<ExecutionContext*>(opaque);
    ReferenceRunner& runner = *execution.runner;
    const std::size_t element_offset =
        static_cast<std::size_t>(item.first_position) *
        kReferenceHiddenSize;
    LongPrefillLayerTileInvocation invocation;
    invocation.item = item;
    invocation.input_hidden =
        runner.views_.long_prefill_hidden[item.input_hidden_buffer] +
        element_offset;
    invocation.output_hidden =
        runner.views_.long_prefill_hidden[item.output_hidden_buffer] +
        element_offset;
    ReferencePrefillTileOutcome outcome =
        runner.prefill_prefix_tile_impl(
            execution.input_token_ids + item.first_position,
            item.token_count, {}, &invocation);
    if (!outcome) {
      execution.failure = outcome.status;
      return false;
    }
    return true;
  };
  callbacks.finish_prompt = +[](void* const opaque,
                                 const std::uint32_t sequence_length,
                                 const std::size_t final_hidden_buffer)
      noexcept {
    auto& execution = *static_cast<ExecutionContext*>(opaque);
    ReferenceRunner& runner = *execution.runner;
    if (sequence_length == 0U ||
        final_hidden_buffer >= kRequestLongPrefillHiddenBufferCount ||
        runner.views_.long_prefill_hidden[final_hidden_buffer] == nullptr) {
      execution.failure = runner_status(
          ReferenceRunnerError::kInvalidRequestState,
          "prefill_layer_major_finish_contract");
      return false;
    }
    const cudaError_t sync_status = cudaStreamSynchronize(
        reinterpret_cast<cudaStream_t>(runner.stream_));
    if (sync_status != cudaSuccess) {
      execution.failure = runner_status(
          ReferenceRunnerError::kCudaFailure,
          "prefill_layer_major_synchronize", kReferenceNoLayer,
          static_cast<int>(sync_status));
      return false;
    }
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
    if (execution.a4w4_full_prefill_selected) {
      const auto& before = execution.a4w4_hits_before;
      const auto& after = g_a4w4_full_prefill_admission_hits;
      const std::size_t tiles = execution.a4w4_complete_model_tiles;
      const auto exact_delta = [tiles](const std::size_t current,
                                       const std::size_t initial,
                                       const std::size_t per_tile) noexcept {
        return current >= initial &&
               current - initial == tiles * per_tile;
      };
      std::size_t attention_logical_delta = 0U;
      std::size_t attention_supermatrix_output_delta = 0U;
      bool attention_accounting_valid = true;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
      attention_accounting_valid =
          a4w4_attention_supermatrix_aggregate_delta(
              execution.a4w4_attention_hits_before,
              g_a4w4_attention_supermatrix_admission_hits, tiles,
              attention_logical_delta,
              attention_supermatrix_output_delta);
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
      std::size_t attention_k256_logical_delta = 0U;
      std::size_t attention_k256_output_delta = 0U;
      attention_accounting_valid =
          attention_accounting_valid &&
          a4w4_attention_supermatrix_aggregate_delta(
              execution.a4w4_attention_k256_hits_before,
              g_a4w4_attention_k256_m128n256_admission_hits, tiles,
              attention_k256_logical_delta, attention_k256_output_delta);
      attention_logical_delta += attention_k256_logical_delta;
      attention_supermatrix_output_delta += attention_k256_output_delta;
#endif
      std::size_t attention_o_k512_delta = 0U;
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
      if (g_a4w4_attention_o_k512_admission_hits <
          execution.a4w4_attention_o_k512_hits_before) {
        attention_accounting_valid = false;
      } else {
        attention_o_k512_delta =
            g_a4w4_attention_o_k512_admission_hits -
            execution.a4w4_attention_o_k512_hits_before;
        attention_accounting_valid =
            attention_accounting_valid &&
            (attention_o_k512_delta == 0U ||
             attention_o_k512_delta == tiles * 64U);
      }
#endif
      attention_accounting_valid =
          attention_accounting_valid &&
          (attention_supermatrix_output_delta == 0U ||
           attention_o_k512_delta == 0U);
      attention_logical_delta += attention_o_k512_delta;
      const bool generic_accounting_valid =
          after.generic_projection_hits >= before.generic_projection_hits &&
          after.generic_projection_hits - before.generic_projection_hits +
                  attention_logical_delta ==
              tiles * 272U;
      if (!exact_delta(after.activation_quantize_hits,
                       before.activation_quantize_hits, 192U) ||
          !attention_accounting_valid || !generic_accounting_valid ||
          !exact_delta(after.paired_gate_up_hits,
                       before.paired_gate_up_hits, 64U) ||
          !exact_delta(after.logical_projection_hits,
                       before.logical_projection_hits, 400U) ||
          after.complete_model_tile_hits !=
              before.complete_model_tile_hits) {
        execution.failure = runner_status(
            ReferenceRunnerError::kInvalidRunner,
            "prefill_a4w4_layer_major_aggregate_accounting");
        return false;
      }
    }
#endif
    const RequestOperationStatus commit =
        runner.state_->set_sequence_length(sequence_length);
    if (!commit) {
      execution.failure = runner_status(
          ReferenceRunnerError::kStateCommitFailure,
          "prefill_layer_major_commit", kReferenceNoLayer,
          commit.cuda_error);
      return false;
    }
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
    if (execution.a4w4_full_prefill_selected) {
      g_a4w4_full_prefill_admission_hits.complete_model_tile_hits +=
          execution.a4w4_complete_model_tiles;
    }
#endif
    runner.retained_prefill_hidden_valid_ = true;
    runner.retained_prefill_position_ = sequence_length - 1U;
    runner.retained_prefill_input_token_ =
        execution.input_token_ids[sequence_length - 1U];
    runner.retained_prefill_hidden_row_ =
        (sequence_length - 1U) % kLongPrefillLayerMajorTileTokens;
    return true;
  };

  const LongPrefillLayerMajorExecutionResult executed =
      run_long_prefill_layer_major(*built.value, callbacks);
  if (!executed) {
    if (!context.failure) {
      context.failure = runner_status(
          ReferenceRunnerError::kInvalidStepOptions,
          "prefill_layer_major_execute");
    }
    return fail_long_prefill(context.failure);
  }

  ReferenceLongPrefillResult value;
  value.first_position = 0U;
  value.token_count = token_count;
  if (measure_timing) {
    const std::chrono::duration<double, std::milli> elapsed =
        Clock::now() - started;
    value.timing.emplace(ReferenceStepTiming{elapsed.count()});
  }
  ReferenceLongPrefillOutcome outcome;
  outcome.value.emplace(std::move(value));
  return outcome;
}

ReferenceStepOutcome ReferenceRunner::finish_prefill_from_retained_tile(
    const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) noexcept {
  using Clock = std::chrono::steady_clock;
  Clock::time_point started{};
  if (options.measure_timing) {
    started = Clock::now();
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

ReferenceRunnerFactoryResult create_reference_runner(
    const ModelWeights* const weights, RequestState* const state,
    const ReferenceRunnerOptions& options) noexcept {
  ReferenceRunnerFactoryResult result;
  if (!is_valid_projection_backend(options.projection_backend)) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency, "projection_backend");
    return result;
  }
#if !defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  if (options.enable_a4w4_full_prefill_admission) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "a4w4_full_prefill_admission_build");
    return result;
  }
#endif
  const ReferenceRunnerStatus weights_status = validate_model_weights(weights);
  if (!weights_status) {
    result.diagnostic = weights_status;
    return result;
  }

  ReferenceRunner runner;
  runner.weights_ = weights;
  runner.state_ = state;
  runner.projection_backend_ = options.projection_backend;
#if defined(Q3X_ENABLE_A4W4_FULL_PREFILL_ADMISSION)
  runner.a4w4_prefill_consumer_ =
      a4w4_full_prefill_inventory_consumer(*weights);
  if (options.enable_a4w4_full_prefill_admission &&
      runner.a4w4_prefill_consumer_ ==
          reference_runner_detail::A4W4PrefillConsumer::kUnavailable) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidModelWeights,
        "prefill_a4w4_inventory");
    return result;
  }
#if defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  const reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute
      paired_gateup_canonical_down_route =
          reference_runner_detail::
              select_a4w4_paired_gateup_canonical_down_route(
                  a4w4_paired_gateup_canonical_down_selector_query(true));
  if (paired_gateup_canonical_down_route ==
      reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::kInvalid) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_mlp_k512_paired_gateup_canonical_down_selector_contract");
    return result;
  }
#if !defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
  if (paired_gateup_canonical_down_route !=
      reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
          kDisabled) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_mlp_k512_paired_gateup_canonical_down_build");
    return result;
  }
#else
  if (paired_gateup_canonical_down_route !=
          reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
              kDisabled &&
      (!options.enable_a4w4_full_prefill_admission ||
       options.projection_backend != ProjectionBackend::kSm87WeightOnly ||
       (runner.a4w4_prefill_consumer_ !=
            reference_runner_detail::A4W4PrefillConsumer::kK128 &&
        runner.a4w4_prefill_consumer_ !=
            reference_runner_detail::A4W4PrefillConsumer::kK256) ||
       !complete_mlp_k512_paired_gateup_canonical_down_attached(*weights))) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidModelWeights,
        "prefill_mlp_k512_paired_gateup_canonical_down_contract");
    return result;
  }
#endif
#if !defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
  if (paired_gateup_canonical_down_route ==
      reference_runner_detail::A4W4PairedGateUpCanonicalDownRoute::
          kGateAndDown) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_mlp_k512_down_m128n128_ldmatrix_pairring_build");
    return result;
  }
#endif
  const reference_runner_detail::
      A4W4DownK512M128N128LdmatrixPairringV1Route
          down_m128n128_ldmatrix_pairring_v1_route =
              reference_runner_detail::
                  select_a4w4_down_k512_m128n128_ldmatrix_pairring_v1_route(
                      a4w4_down_k512_m128n128_ldmatrix_pairring_v1_selector_query(
                          true));
  if (down_m128n128_ldmatrix_pairring_v1_route ==
      reference_runner_detail::
          A4W4DownK512M128N128LdmatrixPairringV1Route::kInvalid) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_mlp_k512_down_m128n128_ldmatrix_pairring_v1_selector_contract");
    return result;
  }
#if !defined(Q3X_ENABLE_A4W4_DOWN_K512_M128N128_LDMATRIX_PAIRRING_ADMISSION)
  if (down_m128n128_ldmatrix_pairring_v1_route ==
      reference_runner_detail::
          A4W4DownK512M128N128LdmatrixPairringV1Route::kEnabled) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_mlp_k512_down_m128n128_ldmatrix_pairring_v1_build");
    return result;
  }
#else
  if (down_m128n128_ldmatrix_pairring_v1_route ==
          reference_runner_detail::
              A4W4DownK512M128N128LdmatrixPairringV1Route::kEnabled &&
      (!options.enable_a4w4_full_prefill_admission ||
       options.projection_backend != ProjectionBackend::kSm87WeightOnly ||
       (runner.a4w4_prefill_consumer_ !=
            reference_runner_detail::A4W4PrefillConsumer::kK128 &&
        runner.a4w4_prefill_consumer_ !=
            reference_runner_detail::A4W4PrefillConsumer::kK256) ||
       !complete_mlp_k512_overlay_attached(*weights))) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidModelWeights,
        "prefill_mlp_k512_down_m128n128_ldmatrix_pairring_v1_contract");
    return result;
  }
#endif
#endif
#if !defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
  if (options.enable_a4w4_full_prefill_admission &&
      runner.a4w4_prefill_consumer_ ==
          reference_runner_detail::A4W4PrefillConsumer::kK256) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_a4w4_attention_k256_build");
    return result;
  }
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_K256_M128N256_ADMISSION)
#if defined(Q3X_ENABLE_A4W4_ATTENTION_SUPERMATRIX_ADMISSION)
  if (options.enable_a4w4_full_prefill_admission &&
      g_enable_a4w4_attention_supermatrix_admission &&
      g_enable_a4w4_attention_k256_m128n256_admission) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_a4w4_attention_selector_conflict");
    return result;
  }
#endif
#if defined(Q3X_ENABLE_A4W4_ATTENTION_O_K512_ADMISSION)
  if (options.enable_a4w4_full_prefill_admission &&
      runner.a4w4_prefill_consumer_ ==
          reference_runner_detail::A4W4PrefillConsumer::kK256 &&
      g_enable_a4w4_attention_o_k512_admission) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_a4w4_k256_attention_o_k512_conflict");
    return result;
  }
#endif
#if defined(Q3X_ENABLE_A4W4_MLP_K512_FRAGMENT_NATIVE_ADMISSION)
  if (options.enable_a4w4_full_prefill_admission &&
      runner.a4w4_prefill_consumer_ ==
          reference_runner_detail::A4W4PrefillConsumer::kK256 &&
      g_enable_a4w4_mlp_k512_fragment_native_admission) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_a4w4_k256_mlp_fragment_native_conflict");
    return result;
  }
#endif
  if (options.enable_a4w4_full_prefill_admission &&
      runner.a4w4_prefill_consumer_ ==
          reference_runner_detail::A4W4PrefillConsumer::kK256 &&
      !g_enable_a4w4_attention_k256_m128n256_admission) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_a4w4_attention_k256_selector");
    return result;
  }
#if !defined(Q3X_ENABLE_A4W4_MLP_K512_ADMISSION)
  if (options.enable_a4w4_full_prefill_admission &&
      runner.a4w4_prefill_consumer_ ==
          reference_runner_detail::A4W4PrefillConsumer::kK256) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidDependency,
        "prefill_a4w4_k256_mlp_k512_build");
    return result;
  }
#else
  if (options.enable_a4w4_full_prefill_admission &&
      runner.a4w4_prefill_consumer_ ==
          reference_runner_detail::A4W4PrefillConsumer::kK256 &&
      !((g_enable_a4w4_mlp_k512_admission &&
         complete_mlp_k512_overlay_attached(*weights)) ||
#if defined(Q3X_ENABLE_A4W4_GATEUP_DOWN_K512_EDGE_M128N512_PAIRED_LDMATRIX_ADMISSION)
        (paired_gateup_canonical_down_route !=
             reference_runner_detail::
                 A4W4PairedGateUpCanonicalDownRoute::kDisabled &&
         complete_mlp_k512_paired_gateup_canonical_down_attached(*weights))
#else
        false
#endif
        )) {
    result.diagnostic = runner_status(
        ReferenceRunnerError::kInvalidModelWeights,
        "prefill_a4w4_k256_mlp_k512_contract");
    return result;
  }
#endif
#endif
#endif
  runner.a4w4_full_prefill_admission_enabled_ =
      options.enable_a4w4_full_prefill_admission;
  const ReferenceRunnerStatus state_status =
      ReferenceRunner::collect_request_views(state, runner.views_);
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
