#include "q3x/runtime/reference_runner.h"

#include "q3x/runtime/prefill_workspace_plan.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

namespace q3x::runtime {
namespace {

constexpr std::uint64_t kBf16Bytes = sizeof(std::uint16_t);
constexpr std::uint64_t kFp32Bytes = sizeof(float);
constexpr std::uint64_t kRopePairs = 32U;
constexpr std::uint64_t kFullKvElementsPerToken = 4U * 256U;
constexpr std::uint64_t kProjectionTemporaryBytes = 1'048'832U;

[[nodiscard]] ReferenceRunnerStatus binding_status(
    const ReferenceRunnerError error, const char* const operation,
    const std::size_t layer = kReferenceNoLayer) noexcept {
  return {error, 0, layer, operation};
}

[[nodiscard]] ReferenceLayerMajorRequestDescriptorOutcome descriptor_failure(
    const ReferenceRunnerError error, const char* const operation,
    const RequestAccessError access_error = RequestAccessError::kNone,
    const std::size_t layer = kReferenceNoLayer) noexcept {
  ReferenceLayerMajorRequestDescriptorOutcome result;
  result.status = binding_status(error, operation, layer);
  result.access_error = access_error;
  return result;
}

[[nodiscard]] ReferenceLayerMajorRequestViewsOutcome views_failure(
    const ReferenceRunnerError error, const char* const operation,
    const RequestAccessError access_error = RequestAccessError::kNone,
    const std::size_t layer = kReferenceNoLayer) noexcept {
  ReferenceLayerMajorRequestViewsOutcome result;
  result.status = binding_status(error, operation, layer);
  result.access_error = access_error;
  return result;
}

[[nodiscard]] bool checked_multiply(const std::uint64_t left,
                                    const std::uint64_t right,
                                    std::uint64_t& result) noexcept {
  if (left != 0U &&
      right > std::numeric_limits<std::uint64_t>::max() / left) {
    return false;
  }
  result = left * right;
  return true;
}

[[nodiscard]] bool checked_add(const std::uint64_t left,
                               const std::uint64_t right,
                               std::uint64_t& result) noexcept {
  if (right > std::numeric_limits<std::uint64_t>::max() - left) {
    return false;
  }
  result = left + right;
  return true;
}

[[nodiscard]] bool same_region(const RequestRegion& left,
                               const RequestRegion& right) noexcept {
  return left.arena_offset == right.arena_offset &&
         left.byte_size == right.byte_size &&
         left.element_capacity == right.element_capacity &&
         left.element_size_bytes == right.element_size_bytes;
}

[[nodiscard]] bool empty_region(const RequestRegion& region) noexcept {
  return region.arena_offset == 0U && region.byte_size == 0U &&
         region.element_capacity == 0U && region.element_size_bytes == 0U;
}

[[nodiscard]] bool empty_matrix(const RequestMatrixRegion& matrix) noexcept {
  return empty_region(matrix.storage) && matrix.row_capacity == 0U &&
         matrix.columns == 0U && matrix.row_stride_elements == 0U;
}

[[nodiscard]] bool region_end(const RequestRegion& region,
                              std::uint64_t& end) noexcept {
  return checked_add(region.arena_offset, region.byte_size, end);
}

[[nodiscard]] bool valid_exact_region(const RequestRegion& region,
                                      const std::uint64_t elements,
                                      const std::uint32_t element_bytes,
                                      const std::uint64_t arena_bytes) noexcept {
  std::uint64_t expected_bytes = 0U;
  std::uint64_t end = 0U;
  return element_bytes != 0U &&
         checked_multiply(elements, element_bytes, expected_bytes) &&
         region.element_capacity == elements &&
         region.element_size_bytes == element_bytes &&
         region.byte_size == expected_bytes &&
         region.arena_offset % kRequestArenaAlignment == 0U &&
         region_end(region, end) && end <= arena_bytes;
}

[[nodiscard]] bool valid_byte_region(const RequestRegion& region,
                                     const std::uint64_t expected_bytes,
                                     const std::uint64_t arena_bytes) noexcept {
  return valid_exact_region(region, expected_bytes, 1U, arena_bytes);
}

[[nodiscard]] bool valid_exact_matrix(
    const RequestMatrixRegion& matrix, const std::uint32_t rows,
    const std::uint32_t columns, const std::uint64_t row_stride,
    const std::uint32_t element_bytes,
    const std::uint64_t arena_bytes) noexcept {
  std::uint64_t elements = 0U;
  return columns != 0U && row_stride >= columns &&
         matrix.row_capacity == rows && matrix.columns == columns &&
         matrix.row_stride_elements == row_stride &&
         checked_multiply(rows, row_stride, elements) &&
         valid_exact_region(matrix.storage, elements, element_bytes,
                            arena_bytes);
}

[[nodiscard]] bool contains(const RequestRegion& owner,
                            const RequestRegion& child) noexcept {
  std::uint64_t owner_end = 0U;
  std::uint64_t child_end = 0U;
  return region_end(owner, owner_end) && region_end(child, child_end) &&
         child.arena_offset >= owner.arena_offset && child_end <= owner_end;
}

[[nodiscard]] bool contiguous(const RequestRegion& first,
                              const RequestRegion& second) noexcept {
  std::uint64_t first_end = 0U;
  return region_end(first, first_end) && first_end == second.arena_offset;
}

[[nodiscard]] bool aligned_next(const RequestRegion& first,
                                const RequestRegion& second) noexcept {
  std::uint64_t first_end = 0U;
  if (!region_end(first, first_end) ||
      first_end > std::numeric_limits<std::uint64_t>::max() -
                      (kRequestArenaAlignment - 1U)) {
    return false;
  }
  const std::uint64_t aligned =
      (first_end + (kRequestArenaAlignment - 1U)) &
      ~(kRequestArenaAlignment - 1U);
  return aligned == second.arena_offset;
}

[[nodiscard]] bool same_matrix_storage(
    const RequestMatrixRegion& matrix, const RequestRegion& region) noexcept {
  return same_region(matrix.storage, region);
}

[[nodiscard]] bool validate_common_identity(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const RequestMemoryPlan& common = plan.common;
  const RequestMemoryProfile expected_profile =
      plan.layout == LayerMajorRequestLayout::kP40WholeCorePromptWide
          ? RequestMemoryProfile::kLayerMajorP40WholeCore
          : RequestMemoryProfile::kLayerMajorC8192;
  if (common.profile != expected_profile ||
      common.batch_size != 1U || common.prefill_chunk_size !=
                                      kMaximumRequestPrefillChunkSize ||
      common.max_sequence_length == 0U ||
      common.max_sequence_length > kAbsoluteRequestMaxSequenceLength ||
      common.arena_bytes == 0U ||
      common.persistent_offset != 0U ||
      common.workspace_offset != common.persistent_bytes ||
      common.rope_offset != common.workspace_offset + common.workspace_bytes ||
      common.arena_bytes != common.rope_offset + common.rope_bytes) {
    return false;
  }
  if (plan.layout == LayerMajorRequestLayout::kC8192FamilyOverlay &&
      reference_runner_detail::validate_reference_workspace_plan(common) !=
          ReferenceRunnerError::kNone) {
    return false;
  }
  if (!valid_exact_region(common.conv_state,
                          kRequestConvStateBytes / kBf16Bytes,
                          kBf16Bytes, common.arena_bytes) ||
      !valid_exact_region(common.gdn_state,
                          kRequestGdnStateBytes / kBf16Bytes,
                          kBf16Bytes, common.arena_bytes) ||
      common.conv_state.arena_offset != common.persistent_offset ||
      !contiguous(common.conv_state, common.gdn_state)) {
    return false;
  }

  RequestRegion previous = common.gdn_state;
  std::uint64_t kv_elements = 0U;
  if (!checked_multiply(common.max_sequence_length,
                        kFullKvElementsPerToken, kv_elements)) {
    return false;
  }
  for (std::size_t slot = 0U; slot < kRequestFullLayerCount; ++slot) {
    if (!valid_exact_region(common.key_cache[slot], kv_elements,
                            kBf16Bytes, common.arena_bytes) ||
        !valid_exact_region(common.value_cache[slot], kv_elements,
                            kBf16Bytes, common.arena_bytes) ||
        !contiguous(previous, common.key_cache[slot]) ||
        !contiguous(common.key_cache[slot], common.value_cache[slot])) {
      return false;
    }
    previous = common.value_cache[slot];
  }
  std::uint64_t persistent_end = 0U;
  if (!region_end(previous, persistent_end) ||
      persistent_end != common.persistent_offset + common.persistent_bytes) {
    return false;
  }

  std::uint64_t rope_elements = 0U;
  if (!checked_multiply(common.max_sequence_length, kRopePairs,
                        rope_elements) ||
      !valid_exact_region(common.rope_cos_fp32, rope_elements, kFp32Bytes,
                          common.arena_bytes) ||
      common.rope_cos_fp32.arena_offset != common.rope_offset) {
    return false;
  }
  const bool whole_core_p40 =
      plan.layout == LayerMajorRequestLayout::kP40WholeCorePromptWide;
  std::uint64_t sine_end = 0U;
  if (whole_core_p40) {
    std::uint64_t expected_sine_offset = 0U;
    std::uint64_t expected_rope_bytes = 0U;
    if (!checked_add(common.rope_cos_fp32.arena_offset,
                     common.rope_cos_fp32.byte_size,
                     expected_sine_offset) ||
        !checked_multiply(rope_elements, kFp32Bytes,
                          expected_rope_bytes) ||
        !checked_multiply(expected_rope_bytes, 2U,
                          expected_rope_bytes) ||
        common.rope_sin_fp32.element_capacity != rope_elements ||
        common.rope_sin_fp32.element_size_bytes != kFp32Bytes ||
        common.rope_sin_fp32.byte_size !=
            common.rope_cos_fp32.byte_size ||
        common.rope_sin_fp32.arena_offset != expected_sine_offset ||
        common.rope_bytes != expected_rope_bytes ||
        !region_end(common.rope_sin_fp32, sine_end)) {
      return false;
    }
    return sine_end == common.arena_bytes;
  }
  return valid_exact_region(common.rope_sin_fp32, rope_elements,
                            kFp32Bytes, common.arena_bytes) &&
         aligned_next(common.rope_cos_fp32, common.rope_sin_fp32) &&
         [&common]() noexcept {
           std::uint64_t sine_end = 0U;
           if (!region_end(common.rope_sin_fp32, sine_end) ||
               sine_end > std::numeric_limits<std::uint64_t>::max() -
                              (kRequestArenaAlignment - 1U)) {
             return false;
           }
           return ((sine_end + (kRequestArenaAlignment - 1U)) &
                   ~(kRequestArenaAlignment - 1U)) == common.arena_bytes;
         }();
}

[[nodiscard]] bool validate_layer_schedule(
    const RequestMemoryPlan& common) noexcept {
  std::size_t linear_slot = 0U;
  std::size_t full_slot = 0U;
  for (std::size_t layer = 0U; layer < kRequestLayerCount; ++layer) {
    const model::LayerType expected =
        reference_runner_detail::expected_reference_layer_type(layer);
    const std::size_t expected_slot =
        expected == model::LayerType::kFullAttention ? full_slot++
                                                     : linear_slot++;
    if (common.layers[layer].type != expected ||
        common.layers[layer].slot != expected_slot) {
      return false;
    }
  }
  return linear_slot == kRequestLinearLayerCount &&
         full_slot == kRequestFullLayerCount;
}

[[nodiscard]] bool validate_gdn_regions(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const std::uint32_t panel = plan.operator_panel_capacity_tokens;
  const std::uint64_t arena_bytes = plan.common.arena_bytes;
  const RequestRegion& family = plan.c8192_family_phase_arena;
  const LayerMajorGdnPhaseRegions& gdn = plan.gdn;
  if (!valid_exact_matrix(gdn.qkv_bf16, panel, 10'240U, 10'240U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(gdn.z_bf16, panel, 6'144U, 6'144U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(gdn.a_bf16, panel, 48U, 48U, kBf16Bytes,
                          arena_bytes) ||
      !valid_exact_matrix(gdn.b_bf16, panel, 48U, 48U, kBf16Bytes,
                          arena_bytes) ||
      !valid_exact_matrix(gdn.recurrent_core_bf16, panel, 6'144U, 6'144U,
                          kBf16Bytes, arena_bytes) ||
      !valid_byte_region(gdn.native_c64_workspace,
                         kLayerMajorPrefillGdnC64NativeWorkspaceBytes,
                         arena_bytes) ||
      !valid_exact_matrix(gdn.normalized_input_bf16, panel, 5'120U, 5'120U,
                          kBf16Bytes, arena_bytes) ||
      !valid_byte_region(gdn.input_projection_temporary,
                         kProjectionTemporaryBytes, arena_bytes) ||
      !valid_exact_matrix(gdn.branch_output_bf16, panel, 5'120U, 5'120U,
                          kBf16Bytes, arena_bytes) ||
      !valid_byte_region(gdn.output_projection_temporary,
                         kProjectionTemporaryBytes, arena_bytes)) {
    return false;
  }
  const std::array<RequestRegion, 10U> regions{
      gdn.qkv_bf16.storage,
      gdn.z_bf16.storage,
      gdn.a_bf16.storage,
      gdn.b_bf16.storage,
      gdn.recurrent_core_bf16.storage,
      gdn.native_c64_workspace,
      gdn.normalized_input_bf16.storage,
      gdn.input_projection_temporary,
      gdn.branch_output_bf16.storage,
      gdn.output_projection_temporary};
  for (const RequestRegion& region : regions) {
    if (!contains(family, region)) {
      return false;
    }
  }
  return gdn.qkv_bf16.storage.arena_offset == family.arena_offset &&
         contiguous(gdn.qkv_bf16.storage, gdn.z_bf16.storage) &&
         contiguous(gdn.z_bf16.storage, gdn.a_bf16.storage) &&
         contiguous(gdn.a_bf16.storage, gdn.b_bf16.storage) &&
         contiguous(gdn.b_bf16.storage,
                    gdn.recurrent_core_bf16.storage) &&
         contiguous(gdn.recurrent_core_bf16.storage,
                    gdn.native_c64_workspace) &&
         gdn.normalized_input_bf16.storage.arena_offset ==
             gdn.recurrent_core_bf16.storage.arena_offset &&
         gdn.branch_output_bf16.storage.arena_offset ==
             gdn.qkv_bf16.storage.arena_offset &&
         contiguous(gdn.normalized_input_bf16.storage,
                    gdn.input_projection_temporary) &&
         contiguous(gdn.branch_output_bf16.storage,
                    gdn.output_projection_temporary);
}

[[nodiscard]] bool validate_attention_regions(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const std::uint32_t panel = plan.operator_panel_capacity_tokens;
  const std::uint64_t arena_bytes = plan.common.arena_bytes;
  const RequestRegion& family = plan.c8192_family_phase_arena;
  const LayerMajorAttentionPhaseRegions& attention = plan.attention;
  if (!valid_exact_matrix(attention.raw_q_gate_bf16, panel, 12'288U,
                          12'288U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(attention.processed_q_bf16, panel, 6'144U,
                          6'144U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(attention.packed_gate_bf16, panel, 6'144U,
                          6'144U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(attention.normalized_input_bf16, panel, 5'120U,
                          5'120U, kBf16Bytes, arena_bytes) ||
      !valid_byte_region(attention.input_projection_temporary,
                         kProjectionTemporaryBytes, arena_bytes) ||
      !valid_exact_matrix(attention.core_output_bf16, panel, 6'144U,
                          6'144U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(attention.branch_output_bf16, panel, 5'120U,
                          5'120U, kBf16Bytes, arena_bytes) ||
      !valid_byte_region(attention.output_projection_temporary,
                         kProjectionTemporaryBytes, arena_bytes)) {
    return false;
  }
  const std::array<RequestRegion, 8U> regions{
      attention.raw_q_gate_bf16.storage,
      attention.processed_q_bf16.storage,
      attention.packed_gate_bf16.storage,
      attention.normalized_input_bf16.storage,
      attention.input_projection_temporary,
      attention.core_output_bf16.storage,
      attention.branch_output_bf16.storage,
      attention.output_projection_temporary};
  for (const RequestRegion& region : regions) {
    if (!contains(family, region)) {
      return false;
    }
  }
  return attention.raw_q_gate_bf16.storage.arena_offset ==
             family.arena_offset &&
         contiguous(attention.raw_q_gate_bf16.storage,
                    attention.processed_q_bf16.storage) &&
         contiguous(attention.processed_q_bf16.storage,
                    attention.packed_gate_bf16.storage) &&
         attention.normalized_input_bf16.storage.arena_offset ==
             attention.processed_q_bf16.storage.arena_offset &&
         attention.core_output_bf16.storage.arena_offset ==
             attention.raw_q_gate_bf16.storage.arena_offset &&
         contiguous(attention.normalized_input_bf16.storage,
                    attention.input_projection_temporary) &&
         contiguous(attention.core_output_bf16.storage,
                    attention.branch_output_bf16.storage) &&
         contiguous(attention.branch_output_bf16.storage,
                    attention.output_projection_temporary);
}

[[nodiscard]] bool validate_p40_whole_core_regions(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const bool valid_p40_mlp_layout =
      plan.mlp_layout ==
          LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan ||
      plan.mlp_layout == LayerMajorRequestMlpLayout::
                             kLayerWideP40MarlinParityMergedGateUp;
  if (plan.layout != LayerMajorRequestLayout::kP40WholeCorePromptWide ||
      plan.common.profile != RequestMemoryProfile::kLayerMajorP40WholeCore ||
      plan.common.max_sequence_length !=
          kLayerMajorP40WholeCoreRequestCapacityTokens ||
      plan.operator_panel_capacity_tokens !=
          kLayerMajorP40WholeCorePanelTokens ||
      plan.mlp_capacity_tokens != kLayerMajorP40WholeCorePromptTokens ||
      !valid_p40_mlp_layout ||
      plan.common.arena_bytes != 8'640'542'976U) {
    return false;
  }

  const std::uint64_t arena_bytes = plan.common.arena_bytes;
  const LayerMajorP40WholeCoreRegions& whole = plan.p40_whole_core;
  const RequestRegion& family = whole.family_phase_arena;
  const auto relative_offset = [&family](const RequestRegion& region,
                                         const std::uint64_t expected) noexcept {
    return region.arena_offset >= family.arena_offset &&
           region.arena_offset - family.arena_offset == expected;
  };
  if (whole.prompt_token_count != kLayerMajorP40WholeCorePromptTokens ||
      whole.request_capacity_tokens !=
          kLayerMajorP40WholeCoreRequestCapacityTokens ||
      whole.logical_panel_capacity_tokens !=
          kLayerMajorP40WholeCorePanelTokens ||
      whole.logical_panel_count != kLayerMajorP40WholeCorePanelCount ||
      !valid_byte_region(family, kLayerMajorP40WholeCoreFamilyArenaBytes,
                         arena_bytes) ||
      !contiguous(plan.prompt_residual_bf16.storage, family)) {
    return false;
  }

  const LayerMajorP40WholeCoreLinearPhaseRegions& linear = whole.linear;
  if (!valid_exact_matrix(linear.raw_qkv_bf16, 40'000U, 10'240U,
                          10'240U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(linear.conv_qkv_bf16, 40'000U, 10'240U,
                          10'240U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(linear.z_bf16, 40'000U, 6'144U, 6'144U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(linear.a_bf16, 40'000U, 48U, 48U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(linear.b_bf16, 40'000U, 48U, 48U,
                          kBf16Bytes, arena_bytes) ||
      !valid_byte_region(linear.prompt_wide_workspace, 2'800'640'000U,
                         arena_bytes) ||
      !valid_exact_matrix(linear.output_bf16, 40'000U, 6'144U, 6'144U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(linear.normalized_input_bf16, 40'000U, 5'120U,
                          5'120U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(linear.branch_output_bf16, 40'000U, 5'120U,
                          5'120U, kBf16Bytes, arena_bytes) ||
      !relative_offset(linear.raw_qkv_bf16.storage, 0U) ||
      !relative_offset(linear.conv_qkv_bf16.storage, 819'200'000U) ||
      !relative_offset(linear.z_bf16.storage, 1'638'400'000U) ||
      !relative_offset(linear.a_bf16.storage, 2'129'920'000U) ||
      !relative_offset(linear.b_bf16.storage, 2'133'760'000U) ||
      !relative_offset(linear.prompt_wide_workspace, 2'137'600'000U) ||
      !relative_offset(linear.output_bf16.storage, 4'938'240'000U) ||
      !relative_offset(linear.normalized_input_bf16.storage,
                       4'938'240'000U) ||
      !relative_offset(linear.branch_output_bf16.storage, 0U) ||
      !contiguous(linear.raw_qkv_bf16.storage,
                  linear.conv_qkv_bf16.storage) ||
      !contiguous(linear.conv_qkv_bf16.storage, linear.z_bf16.storage) ||
      !contiguous(linear.z_bf16.storage, linear.a_bf16.storage) ||
      !contiguous(linear.a_bf16.storage, linear.b_bf16.storage) ||
      !contiguous(linear.b_bf16.storage,
                  linear.prompt_wide_workspace) ||
      !contiguous(linear.prompt_wide_workspace,
                  linear.output_bf16.storage)) {
    return false;
  }

  if (!valid_exact_matrix(whole.prompt_token_ids_u32, 40'000U, 1U, 1U,
                          sizeof(std::uint32_t), arena_bytes) ||
      !contains(linear.prompt_wide_workspace,
                whole.prompt_token_ids_u32.storage) ||
      whole.prompt_token_ids_u32.storage.arena_offset !=
          linear.prompt_wide_workspace.arena_offset ||
      whole.prompt_token_ids_u32.storage.arena_offset ==
          family.arena_offset) {
    return false;
  }

  const LayerMajorP40WholeCoreFullAttentionPhaseRegions& full =
      whole.full_attention;
  const std::array<RequestRegion, 6U> full_regions{
      full.raw_q_gate_bf16.storage,
      full.processed_q_bf16.storage,
      full.packed_gate_bf16.storage,
      full.normalized_input_bf16.storage,
      full.core_output_bf16.storage,
      full.branch_output_bf16.storage};
  if (!valid_exact_matrix(full.raw_q_gate_bf16, 40'000U, 12'288U,
                          12'288U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(full.processed_q_bf16, 40'000U, 6'144U, 6'144U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(full.packed_gate_bf16, 40'000U, 6'144U, 6'144U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(full.normalized_input_bf16, 40'000U, 5'120U,
                          5'120U, kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(full.core_output_bf16, 40'000U, 6'144U, 6'144U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(full.branch_output_bf16, 40'000U, 5'120U,
                          5'120U, kBf16Bytes, arena_bytes)) {
    return false;
  }
  for (const RequestRegion& region : full_regions) {
    if (!contains(family, region)) {
      return false;
    }
  }
  return relative_offset(full.raw_q_gate_bf16.storage, 0U) &&
         relative_offset(full.processed_q_bf16.storage, 983'040'000U) &&
         relative_offset(full.packed_gate_bf16.storage, 1'474'560'000U) &&
         relative_offset(full.normalized_input_bf16.storage,
                         4'938'240'000U) &&
         relative_offset(full.core_output_bf16.storage, 0U) &&
         relative_offset(full.branch_output_bf16.storage, 491'520'000U) &&
         contiguous(full.raw_q_gate_bf16.storage,
                    full.processed_q_bf16.storage) &&
         contiguous(full.processed_q_bf16.storage,
                    full.packed_gate_bf16.storage);
}

[[nodiscard]] bool validate_mlp_regions(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const std::uint32_t rows = plan.mlp_capacity_tokens;
  const std::uint64_t arena_bytes = plan.common.arena_bytes;
  const bool whole_core_p40 =
      plan.layout == LayerMajorRequestLayout::kP40WholeCorePromptWide;
  const bool marlin_parity =
      whole_core_p40 &&
      plan.mlp_layout == LayerMajorRequestMlpLayout::
                             kLayerWideP40MarlinParityMergedGateUp;
  const RequestRegion& family =
      whole_core_p40
          ? plan.p40_whole_core.family_phase_arena
          : plan.c8192_family_phase_arena;
  const LayerMajorMlpPhaseRegions& mlp = plan.mlp;
  if (!valid_exact_matrix(mlp.activated_bf16, rows, 17'408U, 17'408U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(mlp.normalized_input_bf16, rows, 5'120U, 5'120U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(mlp.branch_output_bf16, rows, 5'120U, 5'120U,
                          kBf16Bytes, arena_bytes)) {
    return false;
  }

  if (marlin_parity) {
    if (!valid_exact_matrix(
            mlp.merged_gate_up_bf16, rows,
            kLayerMajorP40MarlinParityMergedGateUpFeatures,
            kLayerMajorP40MarlinParityMergedGateUpRowStrideElements,
            kBf16Bytes, arena_bytes) ||
        !empty_matrix(mlp.gate_bf16) || !empty_matrix(mlp.up_bf16) ||
        !valid_byte_region(mlp.gate_up_projection_temporary,
                           kProjectionTemporaryBytes, arena_bytes) ||
        !valid_byte_region(mlp.down_projection_temporary,
                           kProjectionTemporaryBytes, arena_bytes)) {
      return false;
    }
    const std::array<RequestRegion, 6U> regions{
        mlp.merged_gate_up_bf16.storage,
        mlp.activated_bf16.storage,
        mlp.gate_up_projection_temporary,
        mlp.down_projection_temporary,
        mlp.normalized_input_bf16.storage,
        mlp.branch_output_bf16.storage};
    for (const RequestRegion& region : regions) {
      if (!contains(family, region)) {
        return false;
      }
    }
    std::uint64_t activated_end = 0U;
    return rows == kLayerMajorP40WholeCorePromptTokens &&
           mlp.merged_gate_up_bf16.storage.arena_offset ==
               family.arena_offset +
                   kLayerMajorP40MarlinParityMergedGateUpOffset &&
           mlp.activated_bf16.storage.arena_offset ==
               family.arena_offset +
                   kLayerMajorP40MarlinParityActivatedOffset &&
           mlp.gate_up_projection_temporary.arena_offset ==
               family.arena_offset +
                   kLayerMajorP40MarlinParityTemporaryOffset &&
           mlp.down_projection_temporary.arena_offset ==
               mlp.gate_up_projection_temporary.arena_offset &&
           mlp.normalized_input_bf16.storage.arena_offset ==
               family.arena_offset +
                   kLayerMajorP40MarlinParityNormalizedOffset &&
           mlp.branch_output_bf16.storage.arena_offset ==
               mlp.normalized_input_bf16.storage.arena_offset &&
           contiguous(mlp.merged_gate_up_bf16.storage,
                      mlp.activated_bf16.storage) &&
           region_end(mlp.activated_bf16.storage, activated_end) &&
           activated_end ==
               mlp.gate_up_projection_temporary.arena_offset &&
           mlp.gate_up_projection_temporary.arena_offset +
                   mlp.gate_up_projection_temporary.byte_size <=
               mlp.normalized_input_bf16.storage.arena_offset;
  }

  if (!empty_matrix(mlp.merged_gate_up_bf16) ||
      !valid_exact_matrix(mlp.gate_bf16, rows, 17'408U, 17'408U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(mlp.up_bf16, rows, 17'408U, 17'408U,
                          kBf16Bytes, arena_bytes) ||
      !valid_byte_region(mlp.gate_up_projection_temporary,
                         kProjectionTemporaryBytes, arena_bytes) ||
      !valid_byte_region(mlp.down_projection_temporary,
                         kProjectionTemporaryBytes, arena_bytes)) {
    return false;
  }
  const std::array<RequestRegion, 7U> regions{
      mlp.gate_bf16.storage,
      mlp.up_bf16.storage,
      mlp.activated_bf16.storage,
      mlp.normalized_input_bf16.storage,
      mlp.gate_up_projection_temporary,
      mlp.branch_output_bf16.storage,
      mlp.down_projection_temporary};
  for (const RequestRegion& region : regions) {
    if (!contains(family, region)) {
      return false;
    }
  }
  if (whole_core_p40) {
    if (rows != kLayerMajorP40WholeCorePromptTokens ||
        mlp.normalized_input_bf16.storage.arena_offset !=
            plan.p40_whole_core.linear.output_bf16.storage.arena_offset ||
        mlp.branch_output_bf16.storage.arena_offset !=
            mlp.normalized_input_bf16.storage.arena_offset) {
      return false;
    }
    if (plan.mlp_layout ==
        LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan) {
      return mlp.gate_bf16.storage.arena_offset == family.arena_offset &&
             mlp.up_bf16.storage.arena_offset == family.arena_offset &&
             mlp.activated_bf16.storage.arena_offset == family.arena_offset &&
             mlp.gate_up_projection_temporary.arena_offset ==
                 plan.p40_whole_core.linear.prompt_wide_workspace.arena_offset &&
             mlp.down_projection_temporary.arena_offset == family.arena_offset;
    }
    return false;
  }
  if (plan.mlp_layout ==
      LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan) {
    std::uint64_t family_end = 0U;
    std::uint64_t normalized_end = 0U;
    return mlp.gate_bf16.storage.arena_offset == family.arena_offset &&
           mlp.up_bf16.storage.arena_offset == family.arena_offset &&
           mlp.activated_bf16.storage.arena_offset == family.arena_offset &&
           contiguous(mlp.activated_bf16.storage,
                      mlp.normalized_input_bf16.storage) &&
           mlp.branch_output_bf16.storage.arena_offset ==
               mlp.normalized_input_bf16.storage.arena_offset &&
           mlp.gate_up_projection_temporary.arena_offset ==
               family.arena_offset &&
           mlp.down_projection_temporary.arena_offset ==
               family.arena_offset &&
           region_end(family, family_end) &&
           region_end(mlp.normalized_input_bf16.storage, normalized_end) &&
           normalized_end == family_end;
  }
  return mlp.gate_bf16.storage.arena_offset == family.arena_offset &&
         contiguous(mlp.gate_bf16.storage, mlp.up_bf16.storage) &&
         contiguous(mlp.up_bf16.storage, mlp.activated_bf16.storage) &&
         [&family, &mlp]() noexcept {
           std::uint64_t family_end = 0U;
           std::uint64_t activated_end = 0U;
           return region_end(family, family_end) &&
                  region_end(mlp.activated_bf16.storage, activated_end) &&
                  activated_end == family_end;
         }() &&
         mlp.normalized_input_bf16.storage.arena_offset ==
             mlp.activated_bf16.storage.arena_offset &&
         mlp.branch_output_bf16.storage.arena_offset ==
             mlp.up_bf16.storage.arena_offset &&
         mlp.down_projection_temporary.arena_offset ==
             mlp.gate_bf16.storage.arena_offset &&
         contiguous(mlp.normalized_input_bf16.storage,
                    mlp.gate_up_projection_temporary);
}

[[nodiscard]] bool validate_legacy_regions(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const RequestMemoryPlan& common = plan.common;
  const LayerMajorLegacyC512Regions& legacy = plan.legacy_c512;
  const std::uint32_t rows = plan.legacy_prefill_chunk_size;
  RequestRegion previous =
      plan.layout == LayerMajorRequestLayout::kP40WholeCorePromptWide
          ? plan.p40_whole_core.family_phase_arena
          : plan.c8192_family_phase_arena;
  for (std::size_t index = 0U; index < legacy.hidden_bf16.size(); ++index) {
    if (!valid_exact_matrix(legacy.hidden_bf16[index], rows, 5'120U, 5'120U,
                            kBf16Bytes, common.arena_bytes) ||
        !same_matrix_storage(legacy.hidden_bf16[index],
                             common.hidden_bf16[index]) ||
        !contiguous(previous, legacy.hidden_bf16[index].storage)) {
      return false;
    }
    previous = legacy.hidden_bf16[index].storage;
  }
  for (std::size_t index = 0U; index < legacy.projection_bf16.size(); ++index) {
    if (!valid_exact_matrix(legacy.projection_bf16[index], rows, 17'408U,
                            17'408U, kBf16Bytes, common.arena_bytes) ||
        !same_matrix_storage(legacy.projection_bf16[index],
                             common.projection_bf16[index]) ||
        !contiguous(previous, legacy.projection_bf16[index].storage)) {
      return false;
    }
    previous = legacy.projection_bf16[index].storage;
  }
  if (!valid_exact_matrix(legacy.linear_a_bf16, rows, 48U, 48U,
                          kBf16Bytes, common.arena_bytes) ||
      !valid_exact_matrix(legacy.linear_b_bf16, rows, 48U, 48U,
                          kBf16Bytes, common.arena_bytes) ||
      !same_matrix_storage(legacy.linear_a_bf16, common.linear_a_bf16) ||
      !same_matrix_storage(legacy.linear_b_bf16, common.linear_b_bf16) ||
      !contiguous(previous, legacy.linear_a_bf16.storage) ||
      !contiguous(legacy.linear_a_bf16.storage,
                  legacy.linear_b_bf16.storage)) {
    return false;
  }
  previous = legacy.linear_b_bf16.storage;

  std::uint64_t probability_elements = 0U;
  std::uint64_t expected_scratch_bytes = 0U;
  const std::uint64_t probability_tokens =
      plan.layout == LayerMajorRequestLayout::kP40WholeCorePromptWide
          ? kLayerMajorP40WholeCorePromptTokens
          : common.max_sequence_length;
  if (!checked_multiply(probability_tokens, 24U,
                        probability_elements)) {
    return false;
  }
  const std::uint64_t expected_scratch_elements =
      probability_elements > 262'144U ? probability_elements : 262'144U;
  if (!checked_multiply(expected_scratch_elements, kFp32Bytes,
                        expected_scratch_bytes) ||
      legacy.fp32_scratch.element_size_bytes != kFp32Bytes ||
      legacy.fp32_scratch.element_capacity != expected_scratch_elements ||
      legacy.fp32_scratch.byte_size != expected_scratch_bytes ||
      !same_region(legacy.fp32_scratch, common.fp32_scratch) ||
      !contiguous(previous, legacy.fp32_scratch) ||
      legacy.gqa_probability_scratch.arena_offset !=
          legacy.fp32_scratch.arena_offset ||
      legacy.gqa_probability_scratch.element_size_bytes != kFp32Bytes ||
      legacy.gqa_probability_scratch.element_capacity != probability_elements ||
      legacy.gqa_probability_scratch.byte_size !=
          probability_elements * kFp32Bytes ||
      !same_region(legacy.gqa_probability_scratch,
                   common.gqa_probability_scratch)) {
    return false;
  }
  std::uint64_t scratch_end = 0U;
  return region_end(legacy.fp32_scratch, scratch_end) &&
         scratch_end <= common.rope_offset &&
         aligned_next(legacy.fp32_scratch,
                      plan.final_hidden_bf16.storage);
}

[[nodiscard]] bool matrix_view_matches(
    const DeviceMatrixView& view, const RequestMatrixRegion& region,
    const std::uintptr_t arena_identity) noexcept;

[[nodiscard]] bool buffer_view_matches(
    const DeviceBufferView& view, const RequestRegion& region,
    const std::uintptr_t arena_identity) noexcept {
  if (empty_region(region)) {
    return view.device_data == nullptr && view.arena_offset == 0U &&
           view.byte_size == 0U && view.element_capacity == 0U &&
           view.element_size_bytes == 0U;
  }
  if (view.device_data == nullptr ||
      view.arena_offset != region.arena_offset ||
      view.byte_size != region.byte_size ||
      view.element_capacity != region.element_capacity ||
      view.element_size_bytes != region.element_size_bytes ||
      region.arena_offset >
          std::numeric_limits<std::uintptr_t>::max() - arena_identity) {
    return false;
  }
  return reinterpret_cast<std::uintptr_t>(view.device_data) ==
         arena_identity + region.arena_offset;
}

[[nodiscard]] bool const_buffer_view_matches(
    const ConstDeviceBufferView& view, const RequestRegion& region,
    const std::uintptr_t arena_identity) noexcept {
  if (empty_region(region)) {
    return view.device_data == nullptr && view.arena_offset == 0U &&
           view.byte_size == 0U && view.element_capacity == 0U &&
           view.element_size_bytes == 0U;
  }
  if (view.device_data == nullptr ||
      view.arena_offset != region.arena_offset ||
      view.byte_size != region.byte_size ||
      view.element_capacity != region.element_capacity ||
      view.element_size_bytes != region.element_size_bytes ||
      region.arena_offset >
          std::numeric_limits<std::uintptr_t>::max() - arena_identity) {
    return false;
  }
  return reinterpret_cast<std::uintptr_t>(view.device_data) ==
         arena_identity + region.arena_offset;
}

[[nodiscard]] bool matrix_view_matches(
    const DeviceMatrixView& view, const RequestMatrixRegion& region,
    const std::uintptr_t arena_identity) noexcept {
  return view.row_capacity == region.row_capacity &&
         view.columns == region.columns &&
         view.row_stride_elements == region.row_stride_elements &&
         buffer_view_matches(view.storage, region.storage, arena_identity);
}

[[nodiscard]] bool derive_arena_identity(
    const DeviceBufferView& view, std::uintptr_t& identity) noexcept {
  const std::uintptr_t address =
      reinterpret_cast<std::uintptr_t>(view.device_data);
  if (view.device_data == nullptr || address < view.arena_offset) {
    return false;
  }
  identity = address - static_cast<std::uintptr_t>(view.arena_offset);
  return identity != 0U;
}

[[nodiscard]] RequestRegion compact_linear_subregion(
    const RequestRegion& aggregate, const std::size_t slot,
    const std::uint64_t elements_per_layer) noexcept {
  RequestRegion region;
  region.arena_offset = aggregate.arena_offset +
                        slot * elements_per_layer * kBf16Bytes;
  region.byte_size = elements_per_layer * kBf16Bytes;
  region.element_capacity = elements_per_layer;
  region.element_size_bytes = kBf16Bytes;
  return region;
}

[[nodiscard]] bool validate_gdn_views(
    const LayerMajorGdnPhaseViews& views,
    const LayerMajorGdnPhaseRegions& regions,
    const std::uintptr_t arena_identity) noexcept {
  return matrix_view_matches(views.qkv_bf16, regions.qkv_bf16,
                             arena_identity) &&
         matrix_view_matches(views.z_bf16, regions.z_bf16, arena_identity) &&
         matrix_view_matches(views.a_bf16, regions.a_bf16, arena_identity) &&
         matrix_view_matches(views.b_bf16, regions.b_bf16, arena_identity) &&
         matrix_view_matches(views.recurrent_core_bf16,
                             regions.recurrent_core_bf16, arena_identity) &&
         buffer_view_matches(views.native_c64_workspace,
                             regions.native_c64_workspace, arena_identity) &&
         matrix_view_matches(views.normalized_input_bf16,
                             regions.normalized_input_bf16, arena_identity) &&
         buffer_view_matches(views.input_projection_temporary,
                             regions.input_projection_temporary,
                             arena_identity) &&
         matrix_view_matches(views.branch_output_bf16,
                             regions.branch_output_bf16, arena_identity) &&
         buffer_view_matches(views.output_projection_temporary,
                             regions.output_projection_temporary,
                             arena_identity);
}

[[nodiscard]] bool validate_attention_views(
    const LayerMajorAttentionPhaseViews& views,
    const LayerMajorAttentionPhaseRegions& regions,
    const std::uintptr_t arena_identity) noexcept {
  return matrix_view_matches(views.raw_q_gate_bf16,
                             regions.raw_q_gate_bf16, arena_identity) &&
         matrix_view_matches(views.processed_q_bf16,
                             regions.processed_q_bf16, arena_identity) &&
         matrix_view_matches(views.packed_gate_bf16,
                             regions.packed_gate_bf16, arena_identity) &&
         matrix_view_matches(views.normalized_input_bf16,
                             regions.normalized_input_bf16, arena_identity) &&
         buffer_view_matches(views.input_projection_temporary,
                             regions.input_projection_temporary,
                             arena_identity) &&
         matrix_view_matches(views.core_output_bf16,
                             regions.core_output_bf16, arena_identity) &&
         matrix_view_matches(views.branch_output_bf16,
                             regions.branch_output_bf16, arena_identity) &&
         buffer_view_matches(views.output_projection_temporary,
                             regions.output_projection_temporary,
                             arena_identity);
}

[[nodiscard]] bool validate_mlp_views(
    const LayerMajorMlpPhaseViews& views,
    const LayerMajorMlpPhaseRegions& regions,
    const std::uintptr_t arena_identity) noexcept {
  return matrix_view_matches(views.merged_gate_up_bf16,
                             regions.merged_gate_up_bf16,
                             arena_identity) &&
         matrix_view_matches(views.gate_bf16, regions.gate_bf16,
                             arena_identity) &&
         matrix_view_matches(views.up_bf16, regions.up_bf16,
                             arena_identity) &&
         matrix_view_matches(views.activated_bf16,
                             regions.activated_bf16, arena_identity) &&
         matrix_view_matches(views.normalized_input_bf16,
                             regions.normalized_input_bf16, arena_identity) &&
         buffer_view_matches(views.gate_up_projection_temporary,
                             regions.gate_up_projection_temporary,
                             arena_identity) &&
         matrix_view_matches(views.branch_output_bf16,
                             regions.branch_output_bf16, arena_identity) &&
         buffer_view_matches(views.down_projection_temporary,
                             regions.down_projection_temporary,
                             arena_identity);
}

[[nodiscard]] bool validate_legacy_views(
    const LayerMajorLegacyC512Views& views,
    const LayerMajorLegacyC512Regions& regions,
    const std::uintptr_t arena_identity) noexcept {
  for (std::size_t index = 0U; index < views.hidden_bf16.size(); ++index) {
    if (!matrix_view_matches(views.hidden_bf16[index],
                             regions.hidden_bf16[index], arena_identity)) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < views.projection_bf16.size(); ++index) {
    if (!matrix_view_matches(views.projection_bf16[index],
                             regions.projection_bf16[index], arena_identity)) {
      return false;
    }
  }
  return matrix_view_matches(views.linear_a_bf16, regions.linear_a_bf16,
                             arena_identity) &&
         matrix_view_matches(views.linear_b_bf16, regions.linear_b_bf16,
                             arena_identity) &&
         buffer_view_matches(views.fp32_scratch, regions.fp32_scratch,
                             arena_identity) &&
         buffer_view_matches(views.gqa_probability_scratch,
                             regions.gqa_probability_scratch,
                             arena_identity);
}

[[nodiscard]] bool validate_p40_whole_core_views(
    const LayerMajorP40WholeCoreViews& views,
    const LayerMajorP40WholeCoreRegions& regions,
    const std::uintptr_t arena_identity) noexcept {
  const auto& linear_views = views.linear;
  const auto& linear_regions = regions.linear;
  const auto& full_views = views.full_attention;
  const auto& full_regions = regions.full_attention;
  return matrix_view_matches(views.prompt_token_ids_u32,
                             regions.prompt_token_ids_u32,
                             arena_identity) &&
         matrix_view_matches(linear_views.raw_qkv_bf16,
                             linear_regions.raw_qkv_bf16,
                             arena_identity) &&
         matrix_view_matches(linear_views.conv_qkv_bf16,
                             linear_regions.conv_qkv_bf16,
                             arena_identity) &&
         matrix_view_matches(linear_views.z_bf16,
                             linear_regions.z_bf16, arena_identity) &&
         matrix_view_matches(linear_views.a_bf16,
                             linear_regions.a_bf16, arena_identity) &&
         matrix_view_matches(linear_views.b_bf16,
                             linear_regions.b_bf16, arena_identity) &&
         buffer_view_matches(linear_views.prompt_wide_workspace,
                             linear_regions.prompt_wide_workspace,
                             arena_identity) &&
         matrix_view_matches(linear_views.output_bf16,
                             linear_regions.output_bf16,
                             arena_identity) &&
         matrix_view_matches(linear_views.normalized_input_bf16,
                             linear_regions.normalized_input_bf16,
                             arena_identity) &&
         matrix_view_matches(linear_views.branch_output_bf16,
                             linear_regions.branch_output_bf16,
                             arena_identity) &&
         matrix_view_matches(full_views.raw_q_gate_bf16,
                             full_regions.raw_q_gate_bf16,
                             arena_identity) &&
         matrix_view_matches(full_views.processed_q_bf16,
                             full_regions.processed_q_bf16,
                             arena_identity) &&
         matrix_view_matches(full_views.packed_gate_bf16,
                             full_regions.packed_gate_bf16,
                             arena_identity) &&
         matrix_view_matches(full_views.normalized_input_bf16,
                             full_regions.normalized_input_bf16,
                             arena_identity) &&
         matrix_view_matches(full_views.core_output_bf16,
                             full_regions.core_output_bf16,
                             arena_identity) &&
         matrix_view_matches(full_views.branch_output_bf16,
                             full_regions.branch_output_bf16,
                             arena_identity);
}

}  // namespace

ReferenceLayerMajorRequestDescriptorOutcome
build_reference_layer_major_candidate_binding_descriptor(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const RequestMemoryPlan& common = plan.common;
  const bool whole_core_p40 =
      plan.layout ==
      LayerMajorRequestLayout::kP40WholeCorePromptWide;
  const RequestMemoryProfile expected_profile =
      whole_core_p40 ? RequestMemoryProfile::kLayerMajorP40WholeCore
                     : RequestMemoryProfile::kLayerMajorC8192;
  if (validate_request_memory_profile(common.profile, expected_profile) !=
      RequestAccessError::kNone) {
    return descriptor_failure(
        ReferenceRunnerError::kInvalidRequestState,
        "layer_major_memory_profile",
        RequestAccessError::kMemoryProfileMismatch);
  }
  if (!validate_layer_schedule(common)) {
    return descriptor_failure(ReferenceRunnerError::kInvalidLayerSchedule,
                              "layer_major_layer_schedule");
  }
  if (!validate_common_identity(plan)) {
    return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                              "layer_major_common_identity");
  }
  const bool valid_layout_identity =
      whole_core_p40
          ? plan.operator_panel_capacity_tokens ==
                    kLayerMajorP40WholeCorePanelTokens &&
                plan.common.max_sequence_length ==
                    kLayerMajorP40WholeCoreRequestCapacityTokens &&
                plan.mlp_capacity_tokens ==
                    kLayerMajorP40WholeCorePromptTokens &&
                ((plan.mlp_layout == LayerMajorRequestMlpLayout::
                                         kLayerWideP40PersistentTwoSpan &&
                  plan.mlp_tactic == PrefillMlpPhysicalTactic::
                                         kLayerWideP40PersistentFusedGateUp) ||
                 (plan.mlp_layout == LayerMajorRequestMlpLayout::
                                         kLayerWideP40MarlinParityMergedGateUp &&
                  plan.mlp_tactic == PrefillMlpPhysicalTactic::
                                         kLayerWideP40MarlinParityMergedGateUp)) &&
                plan.scratch_strategy == PrefillOperatorScratchStrategy::
                                             kP40WholeCorePromptWideWithDisjointLegacyC512 &&
                plan.gdn_tactic == PrefillGdnPhysicalTactic::
                                       kP40PromptWideChunkGraph
          : plan.layout == LayerMajorRequestLayout::kC8192FamilyOverlay &&
                plan.operator_panel_capacity_tokens ==
                    kLayerMajorRequestOperatorPanelCapacity &&
                ((plan.mlp_layout ==
                      LayerMajorRequestMlpLayout::kPanelLocalThreeSpan &&
                  plan.mlp_capacity_tokens ==
                      kLayerMajorRequestOperatorPanelCapacity &&
                  plan.mlp_tactic ==
                      PrefillMlpPhysicalTactic::kSeparateGateUpAndSilu) ||
                 (plan.mlp_layout == LayerMajorRequestMlpLayout::
                                         kLayerWideP40PersistentTwoSpan &&
                  plan.common.max_sequence_length ==
                      kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens &&
                  plan.mlp_capacity_tokens ==
                      kLayerMajorPrefillLayerWideMlpP40Tokens &&
                  plan.mlp_tactic == PrefillMlpPhysicalTactic::
                                         kLayerWideP40PersistentFusedGateUp));
  if (!valid_layout_identity ||
      plan.legacy_prefill_chunk_size !=
          kMaximumRequestPrefillChunkSize ||
      plan.hidden_strategy !=
          PrefillHiddenStrategy::kSinglePromptWideConditional ||
      (!whole_core_p40 &&
       (plan.scratch_strategy != PrefillOperatorScratchStrategy::
                                    kC8192FamilyOverlayWithDisjointLegacyC512 ||
        plan.gdn_tactic !=
            PrefillGdnPhysicalTactic::kC64NativeInPlaceConv)) ||
      plan.legacy_gdn_tactic !=
          PrefillLegacyGdnPhysicalTactic::kC16Composite) {
    return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                              "layer_major_physical_tactic_identity");
  }
  if (plan.prompt_residual_in_place_contract_bound ||
      plan.family_completion_events_bound ||
      plan.intra_family_phase_contract_bound ||
      plan.prompt_token_ids_consumed_event_bound ||
      plan.projection_workspace_subrange_binding_bound ||
      plan.operator_bindings_complete || plan.executable()) {
    return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                              "layer_major_unbound_contract_identity");
  }
  if (!valid_exact_matrix(
          plan.prompt_residual_bf16, common.max_sequence_length, 5'120U,
          5'120U, kBf16Bytes, common.arena_bytes) ||
      plan.prompt_residual_bf16.storage.arena_offset !=
          common.workspace_offset) {
    return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                              "layer_major_prompt_residual");
  }
  if (whole_core_p40) {
    if (!validate_p40_whole_core_regions(plan)) {
      return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                                "layer_major_p40_whole_core_views");
    }
  } else {
    const std::uint64_t expected_family_bytes =
        plan.mlp_layout ==
                LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan
            ? static_cast<std::uint64_t>(plan.mlp_capacity_tokens) *
                  (17'408U + 5'120U) * 2U
            : static_cast<std::uint64_t>(plan.mlp_capacity_tokens) *
                  17'408U * 2U * 3U;
    if (!valid_byte_region(plan.c8192_family_phase_arena,
                           expected_family_bytes,
                           common.arena_bytes) ||
        !contiguous(plan.prompt_residual_bf16.storage,
                    plan.c8192_family_phase_arena)) {
      return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                                "layer_major_family_arena_identity");
    }
    if (!valid_exact_matrix(
            plan.panel_token_ids_u32,
            kLayerMajorRequestOperatorPanelCapacity, 1U, 1U,
            sizeof(std::uint32_t), common.arena_bytes) ||
        !contains(plan.c8192_family_phase_arena,
                  plan.panel_token_ids_u32.storage) ||
        plan.panel_token_ids_u32.storage.arena_offset !=
            plan.c8192_family_phase_arena.arena_offset) {
      return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                                "layer_major_panel_token_ids");
    }
    if (!validate_gdn_regions(plan)) {
      return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                                "layer_major_gdn_views");
    }
    if (!validate_attention_regions(plan)) {
      return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                                "layer_major_attention_views");
    }
  }
  if (!validate_mlp_regions(plan)) {
    return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                              "layer_major_mlp_views");
  }
  if (!validate_legacy_regions(plan)) {
    return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                              "layer_major_legacy_c512_views");
  }
  if (!valid_exact_matrix(plan.final_hidden_bf16, 1U, 5'120U, 5'120U,
                          kBf16Bytes, common.arena_bytes) ||
      !contiguous(plan.final_hidden_bf16.storage, common.rope_cos_fp32)) {
    return descriptor_failure(ReferenceRunnerError::kInvalidRequestState,
                              "layer_major_final_hidden");
  }

  ReferenceLayerMajorRequestBindingDescriptor descriptor;
  descriptor.profile = common.profile;
  descriptor.max_sequence_length = common.max_sequence_length;
  descriptor.operator_panel_capacity_tokens =
      plan.operator_panel_capacity_tokens;
  descriptor.legacy_prefill_chunk_size = plan.legacy_prefill_chunk_size;
  descriptor.mlp_capacity_tokens = plan.mlp_capacity_tokens;
  descriptor.layout = plan.layout;
  descriptor.mlp_layout = plan.mlp_layout;
  descriptor.arena_bytes = common.arena_bytes;
  descriptor.layers = common.layers;
  descriptor.conv_state_bf16 = common.conv_state;
  descriptor.gdn_state_bf16 = common.gdn_state;
  descriptor.key_cache_bf16 = common.key_cache;
  descriptor.value_cache_bf16 = common.value_cache;
  descriptor.rope_cos_fp32 = common.rope_cos_fp32;
  descriptor.rope_sin_fp32 = common.rope_sin_fp32;
  descriptor.prompt_residual_bf16 = plan.prompt_residual_bf16;
  descriptor.panel_token_ids_u32 = plan.panel_token_ids_u32;
  descriptor.gdn = plan.gdn;
  descriptor.attention = plan.attention;
  descriptor.mlp = plan.mlp;
  descriptor.legacy_c512 = plan.legacy_c512;
  descriptor.p40_whole_core = plan.p40_whole_core;
  descriptor.final_hidden_bf16 = plan.final_hidden_bf16;
  descriptor.hidden_strategy = plan.hidden_strategy;
  descriptor.scratch_strategy = plan.scratch_strategy;
  descriptor.gdn_tactic = plan.gdn_tactic;
  descriptor.legacy_gdn_tactic = plan.legacy_gdn_tactic;
  descriptor.mlp_tactic = plan.mlp_tactic;

  ReferenceLayerMajorRequestDescriptorOutcome result;
  result.value.emplace(std::move(descriptor));
  return result;
}

ReferenceLayerMajorRequestViewsOutcome
collect_reference_layer_major_candidate_views(RequestState* const state) noexcept {
  if (state == nullptr) {
    return views_failure(ReferenceRunnerError::kInvalidDependency,
                         "layer_major_request_state");
  }
  const RequestMemoryProfile profile = state->memory_profile();
  if (profile != RequestMemoryProfile::kLayerMajorC8192 &&
      profile != RequestMemoryProfile::kLayerMajorP40WholeCore) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_memory_profile",
                         RequestAccessError::kMemoryProfileMismatch);
  }
  if (!static_cast<bool>(*state)) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "empty_layer_major_request_state",
                         RequestAccessError::kEmptyState);
  }
  const LayerMajorRequestMemoryPlan* const plan = state->layer_major_plan();
  if (plan == nullptr) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "missing_layer_major_request_plan");
  }
  ReferenceLayerMajorRequestDescriptorOutcome described =
      build_reference_layer_major_candidate_binding_descriptor(*plan);
  if (!described) {
    ReferenceLayerMajorRequestViewsOutcome result;
    result.status = described.status;
    result.access_error = described.access_error;
    return result;
  }
  if (state->sequence_length() > state->max_sequence_length() ||
      state->max_sequence_length() != described.value->max_sequence_length) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_sequence_identity");
  }

  RequestMatrixViewResult prompt = state->layer_major_prompt_residual();
  if (!prompt) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_prompt_residual_accessor",
                         prompt.error);
  }
  std::uintptr_t arena_identity = 0U;
  if (!derive_arena_identity(prompt.value->storage, arena_identity) ||
      described.value->arena_bytes >
          static_cast<std::uint64_t>(
              std::numeric_limits<std::uintptr_t>::max() - arena_identity) ||
      !matrix_view_matches(*prompt.value,
                           described.value->prompt_residual_bf16,
                           arena_identity)) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_prompt_residual_identity");
  }

  RequestMatrixViewResult token_ids;
  LayerMajorGdnPhaseViewResult gdn;
  LayerMajorAttentionPhaseViewResult attention;
  LayerMajorP40WholeCoreViewResult p40_whole_core;
  const bool whole_core_p40 =
      described.value->layout ==
      LayerMajorRequestLayout::kP40WholeCorePromptWide;
  if (whole_core_p40) {
    p40_whole_core = state->layer_major_p40_whole_core_views();
    if (!p40_whole_core) {
      return views_failure(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_p40_whole_core_accessor",
                           p40_whole_core.error);
    }
    if (!validate_p40_whole_core_views(
            *p40_whole_core.value, described.value->p40_whole_core,
            arena_identity)) {
      return views_failure(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_p40_whole_core_identity");
    }
  } else {
    token_ids = state->layer_major_panel_token_ids();
    if (!token_ids) {
      return views_failure(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_panel_token_ids_accessor",
                           token_ids.error);
    }
    if (!matrix_view_matches(*token_ids.value,
                             described.value->panel_token_ids_u32,
                             arena_identity)) {
      return views_failure(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_panel_token_ids_identity");
    }

    gdn = state->layer_major_gdn_phase_views();
    if (!gdn) {
      return views_failure(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_gdn_accessor", gdn.error);
    }
    if (!validate_gdn_views(*gdn.value, described.value->gdn,
                            arena_identity)) {
      return views_failure(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_gdn_identity");
    }

    attention = state->layer_major_attention_phase_views();
    if (!attention) {
      return views_failure(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_attention_accessor",
                           attention.error);
    }
    if (!validate_attention_views(*attention.value,
                                  described.value->attention,
                                  arena_identity)) {
      return views_failure(ReferenceRunnerError::kInvalidRequestState,
                           "layer_major_attention_identity");
    }
  }

  LayerMajorMlpPhaseViewResult mlp = state->layer_major_mlp_phase_views();
  if (!mlp) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_mlp_accessor", mlp.error);
  }
  if (!validate_mlp_views(*mlp.value, described.value->mlp,
                          arena_identity)) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_mlp_identity");
  }

  LayerMajorLegacyC512ViewResult legacy =
      state->layer_major_legacy_c512_views();
  if (!legacy) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_legacy_c512_accessor", legacy.error);
  }
  if (!validate_legacy_views(*legacy.value, described.value->legacy_c512,
                             arena_identity)) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_legacy_c512_identity");
  }

  RequestMatrixViewResult final_hidden =
      state->layer_major_final_hidden();
  if (!final_hidden) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_final_hidden_accessor",
                         final_hidden.error);
  }
  if (!matrix_view_matches(*final_hidden.value,
                           described.value->final_hidden_bf16,
                           arena_identity)) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_final_hidden_identity");
  }

  ReferenceLayerMajorPersistentViews persistent;
  std::array<bool, kRequestLinearLayerCount> linear_seen{};
  std::array<bool, kRequestFullLayerCount> full_seen{};
  constexpr std::uint64_t kConvElementsPerLayer = 10'240U * 3U;
  constexpr std::uint64_t kGdnElementsPerLayer = 48U * 128U * 128U;
  for (std::size_t layer = 0U; layer < kRequestLayerCount; ++layer) {
    const RequestLayerSlot& slot = described.value->layers[layer];
    if (slot.type == model::LayerType::kLinearAttention) {
      if (slot.slot >= linear_seen.size() || linear_seen[slot.slot]) {
        return views_failure(ReferenceRunnerError::kInvalidLayerSchedule,
                             "layer_major_linear_slot_identity",
                             RequestAccessError::kNone, layer);
      }
      RequestViewResult conv = state->conv_state(layer);
      RequestViewResult recurrent = state->gdn_state(layer);
      if (!conv || !recurrent) {
        return views_failure(
            ReferenceRunnerError::kInvalidRequestState,
            !conv ? "layer_major_conv_state_accessor"
                  : "layer_major_gdn_state_accessor",
            !conv ? conv.error : recurrent.error, layer);
      }
      const RequestRegion conv_region = compact_linear_subregion(
          described.value->conv_state_bf16, slot.slot,
          kConvElementsPerLayer);
      const RequestRegion gdn_region = compact_linear_subregion(
          described.value->gdn_state_bf16, slot.slot,
          kGdnElementsPerLayer);
      if (!buffer_view_matches(*conv.value, conv_region, arena_identity) ||
          !buffer_view_matches(*recurrent.value, gdn_region,
                               arena_identity)) {
        return views_failure(ReferenceRunnerError::kInvalidRequestState,
                             "layer_major_linear_state_identity",
                             RequestAccessError::kNone, layer);
      }
      persistent.conv_state_bf16[slot.slot] = *conv.value;
      persistent.gdn_state_bf16[slot.slot] = *recurrent.value;
      linear_seen[slot.slot] = true;
    } else if (slot.type == model::LayerType::kFullAttention) {
      if (slot.slot >= full_seen.size() || full_seen[slot.slot]) {
        return views_failure(ReferenceRunnerError::kInvalidLayerSchedule,
                             "layer_major_full_slot_identity",
                             RequestAccessError::kNone, layer);
      }
      RequestViewResult key = state->key_cache(layer);
      RequestViewResult value = state->value_cache(layer);
      if (!key || !value) {
        return views_failure(
            ReferenceRunnerError::kInvalidRequestState,
            !key ? "layer_major_key_cache_accessor"
                 : "layer_major_value_cache_accessor",
            !key ? key.error : value.error, layer);
      }
      if (!buffer_view_matches(
              *key.value, described.value->key_cache_bf16[slot.slot],
              arena_identity) ||
          !buffer_view_matches(
              *value.value, described.value->value_cache_bf16[slot.slot],
              arena_identity)) {
        return views_failure(ReferenceRunnerError::kInvalidRequestState,
                             "layer_major_kv_identity",
                             RequestAccessError::kNone, layer);
      }
      persistent.key_cache_bf16[slot.slot] = *key.value;
      persistent.value_cache_bf16[slot.slot] = *value.value;
      full_seen[slot.slot] = true;
    } else {
      return views_failure(ReferenceRunnerError::kInvalidLayerSchedule,
                           "layer_major_layer_type",
                           RequestAccessError::kNone, layer);
    }
  }
  for (const bool seen : linear_seen) {
    if (!seen) {
      return views_failure(ReferenceRunnerError::kInvalidLayerSchedule,
                           "layer_major_missing_linear_slot");
    }
  }
  for (const bool seen : full_seen) {
    if (!seen) {
      return views_failure(ReferenceRunnerError::kInvalidLayerSchedule,
                           "layer_major_missing_full_slot");
    }
  }

  RequestConstViewResult rope_cos = state->rope_cos(0U);
  RequestConstViewResult rope_sin = state->rope_sin(0U);
  if (!rope_cos || !rope_sin) {
    return views_failure(
        ReferenceRunnerError::kInvalidRequestState,
        !rope_cos ? "layer_major_rope_cos_accessor"
                  : "layer_major_rope_sin_accessor",
        !rope_cos ? rope_cos.error : rope_sin.error);
  }
  RequestRegion rope_cos_row = described.value->rope_cos_fp32;
  rope_cos_row.byte_size = kRopePairs * kFp32Bytes;
  rope_cos_row.element_capacity = kRopePairs;
  RequestRegion rope_sin_row = described.value->rope_sin_fp32;
  rope_sin_row.byte_size = kRopePairs * kFp32Bytes;
  rope_sin_row.element_capacity = kRopePairs;
  if (!const_buffer_view_matches(*rope_cos.value, rope_cos_row,
                                 arena_identity) ||
      !const_buffer_view_matches(*rope_sin.value, rope_sin_row,
                                 arena_identity)) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_rope_identity");
  }
  persistent.rope_cos_fp32 = {
      rope_cos.value->device_data,
      described.value->rope_cos_fp32.arena_offset,
      described.value->rope_cos_fp32.byte_size,
      described.value->rope_cos_fp32.element_capacity,
      described.value->rope_cos_fp32.element_size_bytes};
  persistent.rope_sin_fp32 = {
      rope_sin.value->device_data,
      described.value->rope_sin_fp32.arena_offset,
      described.value->rope_sin_fp32.byte_size,
      described.value->rope_sin_fp32.element_capacity,
      described.value->rope_sin_fp32.element_size_bytes};

  ReferenceLayerMajorRequestViews views;
  views.descriptor = std::move(*described.value);
  views.prompt_residual_bf16 = *prompt.value;
  if (whole_core_p40) {
    views.p40_whole_core = std::move(*p40_whole_core.value);
  } else {
    views.panel_token_ids_u32 = *token_ids.value;
    views.gdn = std::move(*gdn.value);
    views.attention = std::move(*attention.value);
  }
  views.mlp = std::move(*mlp.value);
  views.legacy_c512 = std::move(*legacy.value);
  views.final_hidden_bf16 = *final_hidden.value;
  views.persistent = std::move(persistent);

  ReferenceLayerMajorRequestViewsOutcome result;
  result.value.emplace(std::move(views));
  return result;
}

}  // namespace q3x::runtime
