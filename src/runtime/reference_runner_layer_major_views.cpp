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
  if (common.profile != RequestMemoryProfile::kLayerMajorC8192 ||
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
  if (reference_runner_detail::validate_reference_workspace_plan(common) !=
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
  return checked_multiply(common.max_sequence_length, kRopePairs,
                          rope_elements) &&
         valid_exact_region(common.rope_cos_fp32, rope_elements, kFp32Bytes,
                            common.arena_bytes) &&
         valid_exact_region(common.rope_sin_fp32, rope_elements, kFp32Bytes,
                            common.arena_bytes) &&
         common.rope_cos_fp32.arena_offset == common.rope_offset &&
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

[[nodiscard]] bool validate_mlp_regions(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const std::uint32_t panel = plan.operator_panel_capacity_tokens;
  const std::uint64_t arena_bytes = plan.common.arena_bytes;
  const RequestRegion& family = plan.c8192_family_phase_arena;
  const LayerMajorMlpPhaseRegions& mlp = plan.mlp;
  if (!valid_exact_matrix(mlp.gate_bf16, panel, 17'408U, 17'408U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(mlp.up_bf16, panel, 17'408U, 17'408U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(mlp.activated_bf16, panel, 17'408U, 17'408U,
                          kBf16Bytes, arena_bytes) ||
      !valid_exact_matrix(mlp.normalized_input_bf16, panel, 5'120U, 5'120U,
                          kBf16Bytes, arena_bytes) ||
      !valid_byte_region(mlp.gate_up_projection_temporary,
                         kProjectionTemporaryBytes, arena_bytes) ||
      !valid_exact_matrix(mlp.branch_output_bf16, panel, 5'120U, 5'120U,
                          kBf16Bytes, arena_bytes) ||
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
  RequestRegion previous = plan.c8192_family_phase_arena;
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
  if (!checked_multiply(common.max_sequence_length, 24U,
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
  return matrix_view_matches(views.gate_bf16, regions.gate_bf16,
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

}  // namespace

ReferenceLayerMajorRequestDescriptorOutcome
build_reference_layer_major_candidate_binding_descriptor(
    const LayerMajorRequestMemoryPlan& plan) noexcept {
  const RequestMemoryPlan& common = plan.common;
  if (validate_request_memory_profile(
          common.profile, RequestMemoryProfile::kLayerMajorC8192) !=
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
  if (plan.operator_panel_capacity_tokens !=
          kLayerMajorRequestOperatorPanelCapacity ||
      plan.legacy_prefill_chunk_size !=
          kMaximumRequestPrefillChunkSize ||
      plan.hidden_strategy !=
          PrefillHiddenStrategy::kSinglePromptWideConditional ||
      plan.scratch_strategy != PrefillOperatorScratchStrategy::
                                   kC8192FamilyOverlayWithDisjointLegacyC512 ||
      plan.gdn_tactic != PrefillGdnPhysicalTactic::kC64NativeInPlaceConv ||
      plan.legacy_gdn_tactic !=
          PrefillLegacyGdnPhysicalTactic::kC16Composite ||
      plan.mlp_tactic !=
          PrefillMlpPhysicalTactic::kSeparateGateUpAndSilu) {
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
  if (!valid_byte_region(plan.c8192_family_phase_arena, 855'638'016U,
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
  const RequestAccessError profile_error = validate_request_memory_profile(
      state->memory_profile(), RequestMemoryProfile::kLayerMajorC8192);
  if (profile_error != RequestAccessError::kNone) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_memory_profile", profile_error);
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

  RequestMatrixViewResult token_ids = state->layer_major_panel_token_ids();
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

  LayerMajorGdnPhaseViewResult gdn = state->layer_major_gdn_phase_views();
  if (!gdn) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_gdn_accessor", gdn.error);
  }
  if (!validate_gdn_views(*gdn.value, described.value->gdn,
                          arena_identity)) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_gdn_identity");
  }

  LayerMajorAttentionPhaseViewResult attention =
      state->layer_major_attention_phase_views();
  if (!attention) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_attention_accessor", attention.error);
  }
  if (!validate_attention_views(*attention.value,
                                described.value->attention,
                                arena_identity)) {
    return views_failure(ReferenceRunnerError::kInvalidRequestState,
                         "layer_major_attention_identity");
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
  views.panel_token_ids_u32 = *token_ids.value;
  views.gdn = std::move(*gdn.value);
  views.attention = std::move(*attention.value);
  views.mlp = std::move(*mlp.value);
  views.legacy_c512 = std::move(*legacy.value);
  views.final_hidden_bf16 = *final_hidden.value;
  views.persistent = std::move(persistent);

  ReferenceLayerMajorRequestViewsOutcome result;
  result.value.emplace(std::move(views));
  return result;
}

}  // namespace q3x::runtime
