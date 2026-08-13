#include "sm87_target_aot_p40_executor_internal.h"

#include "q3x/kernels/sm87_bf16_ab_prefill.h"
#include "q3x/kernels/sm87_target_aot_gdn_cuda.h"
#include "q3x/runtime/decode_ops.h"
#include "../kernels/sm87/sm87_target_aot_attention_launch_internal.h"
#include "../kernels/sm87/sm87_target_aot_attention_preprocess_launch_internal.h"
#include "../kernels/sm87/sm87_target_aot_bf16_ab_launch_internal.h"
#include "../kernels/sm87/sm87_target_aot_gdn_launch_internal.h"
#include "../kernels/sm87/sm87_target_aot_projection_launch_internal.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <optional>
#include <utility>
#include <variant>

namespace q3x::runtime::sm87_target_aot_p40_executor_detail {
namespace {

using kernels::Sm87TargetAotProjectionEncoding;
using kernels::Sm87TargetAotProjectionRole;
using RequestAccess =
    sm87_target_aot_request_detail::Sm87TargetAotRequestStateAccess;
using Transaction =
    sm87_target_aot_request_detail::OwnerBoundExecutionTransaction;
using LayerPoint =
    sm87_target_aot_request_detail::LayerCompletionPoint;
using GlobalPoint =
    sm87_target_aot_request_detail::GlobalCompletionPoint;
using ProjectionAccess =
    target_aot_complete_execution_detail::
        Sm87TargetAotCompleteProjectionExecutionAccess;

inline constexpr std::size_t kVocabulary =
    kSm87TargetAotP40Vocabulary;
inline constexpr std::size_t kGdnQkvRows = 10'240U;
inline constexpr std::size_t kGdnZRows = 6'144U;
inline constexpr std::size_t kGdnAbRows = 48U;
inline constexpr std::size_t kGdnNormElements = 128U;
inline constexpr std::size_t kGdnConvWidth = 4U;
inline constexpr std::size_t kFullQRows = 12'288U;
inline constexpr std::size_t kFullKvRows = 1'024U;
inline constexpr std::size_t kFullNormElements = 256U;
inline constexpr float kRmsEpsilon = 1.0e-6F;

static_assert(kSm87TargetAotP40GreedyWorkspaceResults ==
              kBf16GreedyArgmaxWorkspaceResults);
static_assert(kSm87TargetAotP40GreedyResultBytes ==
              sizeof(Bf16GreedyArgmaxResult));
static_assert(kSm87TargetAotP40Bf16LogitsBytes %
                      alignof(Bf16GreedyArgmaxResult) ==
                  0U);

[[nodiscard]] constexpr bool aligned(const void* const pointer,
                                     const std::size_t alignment) noexcept {
  return pointer != nullptr && alignment != 0U &&
         reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0U;
}

[[nodiscard]] bool valid_bf16_vector(const Bf16VectorWeight& weight,
                                     const std::size_t elements,
                                     const std::size_t alignment = 16U) noexcept {
  return weight.element_count == elements &&
         aligned(weight.data, alignment);
}

[[nodiscard]] bool valid_bf16_tensor3(
    const Bf16Tensor3Weight& weight,
    const std::array<std::size_t, 3U>& shape) noexcept {
  return weight.shape == shape && aligned(weight.data, 16U);
}

[[nodiscard]] bool valid_bf16_linear(const LinearWeight& linear,
                                     const std::size_t output,
                                     const std::size_t input) noexcept {
  const auto* const weight = std::get_if<Bf16LinearWeight>(&linear);
  return weight != nullptr && weight->output_size == output &&
         weight->input_size == input && aligned(weight->weight, 16U);
}

[[nodiscard]] bool valid_fp8_linear(const LinearWeight& linear,
                                    const std::size_t output,
                                    const std::size_t input) noexcept {
  const auto* const weight = std::get_if<Fp8LinearWeight>(&linear);
  return weight != nullptr && weight->output_size == output &&
         weight->input_size == input && aligned(weight->weight, 16U) &&
         aligned(weight->weight_scale_device, alignof(float)) &&
         aligned(weight->input_scale_device, alignof(float)) &&
         std::isfinite(weight->weight_scale) &&
         weight->weight_scale >= 0.0F &&
         std::isfinite(weight->input_scale) && weight->input_scale >= 0.0F;
}

[[nodiscard]] bool valid_nvfp4_linear(const LinearWeight& linear,
                                      const std::size_t output,
                                      const std::size_t input) noexcept {
  const auto* const weight = std::get_if<NvFp4LinearWeight>(&linear);
  return weight != nullptr && weight->output_size == output &&
         weight->input_size == input &&
         aligned(weight->packed_weight, 16U) &&
         aligned(weight->block_scale, 16U) &&
         aligned(weight->weight_scale_2_device, alignof(float)) &&
         aligned(weight->input_scale_device, alignof(float)) &&
         std::isfinite(weight->weight_scale_2) &&
         weight->weight_scale_2 >= 0.0F &&
         std::isfinite(weight->input_scale) && weight->input_scale >= 0.0F;
}

[[nodiscard]] bool valid_model_weights(const ModelWeights& weights) noexcept {
  const auto& embedding = weights.embed_tokens();
  if (embedding.output_size != kVocabulary ||
      embedding.input_size != kSm87TargetAotP40Hidden ||
      !aligned(embedding.weight, 16U) ||
      !valid_bf16_vector(weights.final_norm(),
                         kSm87TargetAotP40Hidden) ||
      !valid_nvfp4_linear(weights.lm_head(), kVocabulary,
                          kSm87TargetAotP40Hidden) ||
      weights.stats().linear_attention_layers !=
          kSm87TargetAotP40GdnLayerCount ||
      weights.stats().full_attention_layers !=
          kSm87TargetAotP40FullLayerCount) {
    return false;
  }

  std::size_t gdn_layers = 0U;
  std::size_t full_layers = 0U;
  for (std::size_t layer_index = 0U;
       layer_index < weights.layers().size(); ++layer_index) {
    const auto& layer = weights.layer(layer_index);
    if (!valid_bf16_vector(layer.input_layernorm,
                           kSm87TargetAotP40Hidden) ||
        !valid_bf16_vector(layer.post_attention_layernorm,
                           kSm87TargetAotP40Hidden) ||
        !valid_nvfp4_linear(layer.mlp.gate_proj,
                            kSm87TargetAotP40Intermediate,
                            kSm87TargetAotP40Hidden) ||
        !valid_nvfp4_linear(layer.mlp.up_proj,
                            kSm87TargetAotP40Intermediate,
                            kSm87TargetAotP40Hidden) ||
        !valid_nvfp4_linear(layer.mlp.down_proj,
                            kSm87TargetAotP40Hidden,
                            kSm87TargetAotP40Intermediate)) {
      return false;
    }

    const bool expected_full = (layer_index + 1U) % 4U == 0U;
    if (expected_full) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer.attention);
      if (attention == nullptr ||
          !valid_fp8_linear(attention->q_proj, kFullQRows,
                            kSm87TargetAotP40Hidden) ||
          !valid_fp8_linear(attention->k_proj, kFullKvRows,
                            kSm87TargetAotP40Hidden) ||
          !valid_fp8_linear(attention->v_proj, kFullKvRows,
                            kSm87TargetAotP40Hidden) ||
          !valid_fp8_linear(attention->o_proj,
                            kSm87TargetAotP40Hidden,
                            kSm87TargetAotP40AttentionWidth) ||
          !valid_bf16_vector(attention->q_norm, kFullNormElements) ||
          !valid_bf16_vector(attention->k_norm, kFullNormElements)) {
        return false;
      }
      ++full_layers;
      continue;
    }

    const auto* const attention =
        std::get_if<LinearAttentionWeights>(&layer.attention);
    if (attention == nullptr ||
        !valid_fp8_linear(attention->in_proj_qkv, kGdnQkvRows,
                          kSm87TargetAotP40Hidden) ||
        !valid_fp8_linear(attention->in_proj_z, kGdnZRows,
                          kSm87TargetAotP40Hidden) ||
        !valid_bf16_linear(attention->in_proj_a, kGdnAbRows,
                           kSm87TargetAotP40Hidden) ||
        !valid_bf16_linear(attention->in_proj_b, kGdnAbRows,
                           kSm87TargetAotP40Hidden) ||
        !valid_bf16_tensor3(attention->conv1d,
                            {kGdnQkvRows, 1U, kGdnConvWidth}) ||
        !valid_bf16_vector(attention->a_log, kGdnAbRows) ||
        !valid_bf16_vector(attention->dt_bias, kGdnAbRows) ||
        !valid_bf16_vector(attention->norm, kGdnNormElements) ||
        !valid_fp8_linear(attention->out_proj,
                          kSm87TargetAotP40Hidden,
                          kSm87TargetAotP40AttentionWidth)) {
      return false;
    }
    ++gdn_layers;
  }
  return gdn_layers == kSm87TargetAotP40GdnLayerCount &&
         full_layers == kSm87TargetAotP40FullLayerCount;
}

[[nodiscard]] bool pointer_device_ordinal(const void* const pointer,
                                          std::int32_t* const device) noexcept {
  if (pointer == nullptr || device == nullptr) {
    return false;
  }
  cudaPointerAttributes attributes{};
  (void)cudaGetLastError();
  const cudaError_t status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    return false;
  }
#if CUDART_VERSION >= 10000
  if (attributes.type != cudaMemoryTypeDevice &&
      attributes.type != cudaMemoryTypeManaged) {
    return false;
  }
#else
  if (attributes.memoryType != cudaMemoryTypeDevice) {
    return false;
  }
#endif
  *device = attributes.device;
  return *device >= 0;
}

[[nodiscard]] bool pointer_on_device(const void* const pointer,
                                     const std::int32_t expected) noexcept {
  std::int32_t actual = -1;
  return pointer_device_ordinal(pointer, &actual) && actual == expected;
}

[[nodiscard]] bool linear_on_device(const LinearWeight& linear,
                                    const std::int32_t device) noexcept {
  if (const auto* const weight = std::get_if<Bf16LinearWeight>(&linear)) {
    return pointer_on_device(weight->weight, device);
  }
  if (const auto* const weight = std::get_if<Fp8LinearWeight>(&linear)) {
    return pointer_on_device(weight->weight, device) &&
           pointer_on_device(weight->weight_scale_device, device) &&
           pointer_on_device(weight->input_scale_device, device);
  }
  const auto* const weight = std::get_if<NvFp4LinearWeight>(&linear);
  return weight != nullptr &&
         pointer_on_device(weight->packed_weight, device) &&
         pointer_on_device(weight->block_scale, device) &&
         pointer_on_device(weight->weight_scale_2_device, device) &&
         pointer_on_device(weight->input_scale_device, device);
}

[[nodiscard]] bool complete_model_residency_preflight(
    const ModelWeights& weights, const std::int32_t device) noexcept {
  if (!pointer_on_device(weights.embed_tokens().weight, device) ||
      !pointer_on_device(weights.final_norm().data, device) ||
      !linear_on_device(weights.lm_head(), device)) {
    return false;
  }
  for (const auto& layer : weights.layers()) {
    if (!pointer_on_device(layer.input_layernorm.data, device) ||
        !pointer_on_device(layer.post_attention_layernorm.data, device) ||
        !linear_on_device(layer.mlp.gate_proj, device) ||
        !linear_on_device(layer.mlp.up_proj, device) ||
        !linear_on_device(layer.mlp.down_proj, device)) {
      return false;
    }
    if (const auto* const attention =
            std::get_if<LinearAttentionWeights>(&layer.attention)) {
      if (!linear_on_device(attention->in_proj_qkv, device) ||
          !linear_on_device(attention->in_proj_z, device) ||
          !linear_on_device(attention->in_proj_a, device) ||
          !linear_on_device(attention->in_proj_b, device) ||
          !pointer_on_device(attention->conv1d.data, device) ||
          !pointer_on_device(attention->a_log.data, device) ||
          !pointer_on_device(attention->dt_bias.data, device) ||
          !pointer_on_device(attention->norm.data, device) ||
          !linear_on_device(attention->out_proj, device)) {
        return false;
      }
    } else if (const auto* const attention =
                   std::get_if<FullAttentionWeights>(&layer.attention)) {
      if (!linear_on_device(attention->q_proj, device) ||
          !linear_on_device(attention->k_proj, device) ||
          !linear_on_device(attention->v_proj, device) ||
          !linear_on_device(attention->o_proj, device) ||
          !pointer_on_device(attention->q_norm.data, device) ||
          !pointer_on_device(attention->k_norm.data, device)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool valid_engine_rope(
    const Sm87TargetAotP40EngineRope& rope,
    const std::int32_t expected_device) noexcept {
  if (rope.identity == 0U ||
      rope.position_count < 262'144U || rope.rotary_pairs != 32U ||
      !aligned(rope.cosines, 16U) || !aligned(rope.sines, 16U)) {
    return false;
  }
  constexpr std::uint64_t kOneTableBytes =
      262'144ULL * 32ULL * sizeof(float);
  const std::uintptr_t cosine_begin =
      reinterpret_cast<std::uintptr_t>(rope.cosines);
  const std::uintptr_t sine_begin =
      reinterpret_cast<std::uintptr_t>(rope.sines);
  if (cosine_begin > std::numeric_limits<std::uintptr_t>::max() -
                         kOneTableBytes ||
      sine_begin > std::numeric_limits<std::uintptr_t>::max() -
                       kOneTableBytes) {
    return false;
  }
  const std::uintptr_t cosine_end = cosine_begin + kOneTableBytes;
  const std::uintptr_t sine_end = sine_begin + kOneTableBytes;
  if (cosine_begin < sine_end && sine_begin < cosine_end) {
    return false;
  }
  std::int32_t cosine_device = -1;
  std::int32_t sine_device = -1;
  return pointer_device_ordinal(rope.cosines, &cosine_device) &&
         pointer_device_ordinal(rope.sines, &sine_device) &&
         cosine_device == expected_device && sine_device == expected_device;
}

[[nodiscard]] Sm87TargetAotProjectionRole input_projection_role(
    const std::size_t layer) noexcept {
  return (layer + 1U) % 4U == 0U
             ? Sm87TargetAotProjectionRole::kFp8FullQkv
             : Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
}

[[nodiscard]] bool complete_asset_preflight(
    const ProjectionAccess& access,
    std::int32_t* const device_ordinal) noexcept {
  if (device_ordinal == nullptr || !access.attached() ||
      access.artifact_count() != kSm87TargetAotP40ProjectionAssets) {
    return false;
  }
  std::size_t count = 0U;
  std::int32_t common_device = -1;
  for (std::size_t layer = 0U; layer < kSm87TargetAotP40LayerCount;
       ++layer) {
    const std::array<Sm87TargetAotProjectionRole, 4U> roles{{
        Sm87TargetAotProjectionRole::kNvFp4GateUp,
        Sm87TargetAotProjectionRole::kNvFp4Down,
        input_projection_role(layer),
        Sm87TargetAotProjectionRole::kFp8AttentionOutput,
    }};
    for (const auto role : roles) {
      auto asset = access.resolve(layer, role);
      if (!asset.has_value() || asset->layer_index() != layer ||
          asset->role() != role || asset->artifact_identity() == 0U ||
          asset->source_inventory_identity() == 0U ||
          asset->payload_bytes() == 0U) {
        return false;
      }
      std::int32_t asset_device = -1;
      if (kernels::sm87_target_aot_nvfp4_cuda_role(role)) {
        const auto* const view = asset->borrow_nvfp4_cuda_asset();
        if (view == nullptr || asset->borrow_fp8_cuda_asset() != nullptr ||
            asset->encoding() !=
                Sm87TargetAotProjectionEncoding::
                    kNvFp4E2M1Block16E4M3FnScale ||
            !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*view)) {
          return false;
        }
        asset_device = view->device_upload_receipt.device_ordinal;
      } else {
        const auto* const view = asset->borrow_fp8_cuda_asset();
        if (view == nullptr || asset->borrow_nvfp4_cuda_asset() != nullptr ||
            asset->encoding() !=
                Sm87TargetAotProjectionEncoding::kFp8E4M3FnTensorScale ||
            !kernels::sm87_target_aot_fp8_cuda_asset_valid(*view)) {
          return false;
        }
        asset_device = view->device_upload_receipt.device_ordinal;
      }
      if (asset_device < 0 ||
          (common_device >= 0 && asset_device != common_device)) {
        return false;
      }
      common_device = asset_device;
      ++count;
    }
  }
  if (count != kSm87TargetAotP40ProjectionAssets || common_device < 0) {
    return false;
  }
  *device_ordinal = common_device;
  return true;
}

[[nodiscard]] bool projection_resources_preflight() noexcept {
  for (const auto role : {Sm87TargetAotProjectionRole::kNvFp4GateUp,
                          Sm87TargetAotProjectionRole::kNvFp4Down}) {
    kernels::Sm87TargetAotNvFp4CudaResources resources{};
    const auto plan = kernels::sm87_target_aot_projection_plan(
        role, kSm87TargetAotP40PromptTokens);
    if (!plan.valid() ||
        kernels::query_sm87_target_aot_nvfp4_cuda_resources(
            role, kSm87TargetAotP40PromptTokens, &resources) !=
            static_cast<int>(cudaSuccess) ||
        resources.role != role || !resources.kernel_compiled ||
        resources.binary_version != 87 ||
        resources.registers_per_thread <= 0 ||
        resources.dynamic_shared_bytes != plan.dynamic_shared_bytes ||
        resources.maximum_threads_per_block <
            static_cast<int>(kernels::kSm87TargetAotProjectionThreads) ||
        resources.static_resources_qualified ||
        resources.numerical_contract_qualified ||
        resources.production_dispatch_eligible) {
      return false;
    }
  }
  for (const auto role : {Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
                          Sm87TargetAotProjectionRole::kFp8FullQkv,
                          Sm87TargetAotProjectionRole::kFp8AttentionOutput}) {
    kernels::Sm87TargetAotFp8CudaResources resources{};
    if (kernels::query_sm87_target_aot_fp8_cuda_resources(
            role, kSm87TargetAotP40PromptTokens, &resources) !=
            static_cast<int>(cudaSuccess) ||
        !kernels::sm87_target_aot_fp8_cuda_resources_structurally_valid(
            resources) ||
        !resources.kernel_compiled || resources.binary_version != 87) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool constituent_resources_preflight(
    const std::int32_t device_ordinal) noexcept {
  int current_device = -1;
  if (cudaGetDevice(&current_device) != cudaSuccess ||
      current_device != device_ordinal || !projection_resources_preflight()) {
    return false;
  }

  kernels::Sm87Bf16AbPromptWideP40Resources ab{};
  if (kernels::query_sm87_bf16_ab_prompt_wide_p40_resources_cuda(&ab) !=
          static_cast<int>(cudaSuccess) ||
      !ab.valid()) {
    return false;
  }

  kernels::Sm87TargetAotGdnCudaResources gdn{};
  if (kernels::query_sm87_target_aot_gdn_cuda_resources(
          kSm87TargetAotP40PromptTokens, &gdn) !=
          static_cast<int>(cudaSuccess) ||
      !kernels::sm87_target_aot_gdn_cuda_resources_structurally_valid(gdn) ||
      !gdn.kernel_compiled || gdn.binary_version != 87) {
    return false;
  }

  sm87_target_aot_attention_execution_detail::TargetP40Resources attention{};
  if (sm87_target_aot_attention_execution_detail::
          query_q128_kv32_p40_two_stage_resources(device_ordinal,
                                                  &attention) !=
          static_cast<int>(cudaSuccess) ||
      !sm87_target_aot_attention_execution_detail::
          target_p40_resources_structurally_valid(attention)) {
    return false;
  }
  return true;
}

[[nodiscard]] bool snapshot_ready_for_bind(
    const sm87_target_aot_request_detail::OwnerSnapshot* const snapshot,
    const std::int32_t expected_device) noexcept {
  if (snapshot == nullptr ||
      !sm87_target_aot_request_detail::validate_owner_snapshot(*snapshot) ||
      snapshot->transaction_phase !=
          Sm87TargetAotRequestTransactionPhase::kAdmittedUnpublished ||
      !snapshot->engine_rope_is_external ||
      !snapshot->all_kv_in_place_aliases_verified ||
      snapshot->plan == nullptr ||
      snapshot->plan->prompt_tokens != kSm87TargetAotP40PromptTokens ||
      snapshot->plan->request_capacity_tokens !=
          kSm87TargetAotP40RequestCapacityTokens ||
      !snapshot->plan->single_allocation ||
      !snapshot->plan->cold_request_only ||
      !snapshot->plan->one_request_wide_commit ||
      !snapshot->plan->cancellation_discards_unpublished ||
      !snapshot->plan->kv_preprocess_in_place_alias_required ||
      snapshot->plan->exposes_raw_arena ||
      snapshot->plan->permits_legacy_fallback) {
    return false;
  }
  std::int32_t request_device = -1;
  return pointer_device_ordinal(snapshot->allocation_base, &request_device) &&
         request_device == expected_device;
}

struct FinalHandoffScratchView final {
  std::uint16_t* logits = nullptr;
  Bf16GreedyArgmaxResult* greedy_workspace = nullptr;
};

// Reuse is admitted only after the executor has recorded every layer's
// terminal Down/LayerComplete edge.  The physical source is the terminal
// layer's owner-bound activated span: it is the family-arena prefix, and its
// declared lifetime ends at Down completion.  final_hidden sits in the next
// non-overlapping owner region and remains live until request commit.
[[nodiscard]] bool borrow_final_handoff_scratch(
    const sm87_target_aot_request_detail::OwnerSnapshot& snapshot,
    FinalHandoffScratchView* const view) noexcept {
  if (view == nullptr || snapshot.plan == nullptr ||
      !sm87_target_aot_p40_final_handoff_scratch_contract().valid() ||
      !sm87_target_aot_request_detail::validate_owner_snapshot(snapshot)) {
    return false;
  }
  const auto& plan = *snapshot.plan;
  const auto& storage = plan.mlp.activated.storage;
  const auto& dead_span = snapshot.layers.back().mlp.activated;
  const auto& final_span = snapshot.final_hidden;
  if (storage.arena_offset != plan.family_arena.arena_offset ||
      storage.lifetime !=
          Sm87TargetAotRequestLifetime::kMlpActivatedUntilDownCompletion ||
      plan.family_arena.lifetime !=
          Sm87TargetAotRequestLifetime::kRequestOwner ||
      plan.final_hidden.storage.lifetime !=
          Sm87TargetAotRequestLifetime::kFinalHiddenUntilRequestCommit ||
      storage.byte_size < kSm87TargetAotP40FinalHandoffScratchBytes ||
      dead_span.byte_size != storage.byte_size ||
      dead_span.physical_span_identity ==
          final_span.physical_span_identity ||
      plan.residual.storage.arena_offset +
              plan.residual.storage.byte_size !=
          plan.family_arena.arena_offset ||
      plan.family_arena.arena_offset + plan.family_arena.byte_size !=
          plan.final_hidden.storage.arena_offset) {
    return false;
  }

  const auto allocation_begin =
      reinterpret_cast<std::uintptr_t>(snapshot.allocation_base);
  const auto family_begin = allocation_begin + plan.family_arena.arena_offset;
  const auto family_end = family_begin + plan.family_arena.byte_size;
  const auto scratch_end =
      family_begin + kSm87TargetAotP40FinalHandoffScratchBytes;
  const auto final_begin =
      allocation_begin + plan.final_hidden.storage.arena_offset;
  if (reinterpret_cast<std::uintptr_t>(dead_span.device_data) != family_begin ||
      reinterpret_cast<std::uintptr_t>(final_span.device_data) != final_begin ||
      scratch_end > family_end || scratch_end > final_begin ||
      family_end != final_begin) {
    return false;
  }

  auto* const logits = reinterpret_cast<std::uint16_t*>(family_begin);
  auto* const greedy = reinterpret_cast<Bf16GreedyArgmaxResult*>(
      family_begin + kSm87TargetAotP40Bf16LogitsBytes);
  if (!aligned(logits, alignof(std::uint16_t)) ||
      !aligned(greedy, alignof(Bf16GreedyArgmaxResult))) {
    return false;
  }
  *view = {logits, greedy};
  return true;
}

[[nodiscard]] Sm87TargetAotP40ExecutorStatus status(
    const Sm87TargetAotP40ExecutorError code, const char* const context,
    const std::size_t layer = kSm87TargetAotP40LayerCount,
    const int cuda_error = 0,
    const sm87_target_aot_request_detail::ExecutionTransactionError
        transaction_error =
            sm87_target_aot_request_detail::ExecutionTransactionError::kNone)
    noexcept {
  return {code, context, layer, cuda_error, transaction_error};
}

[[nodiscard]] bool record_layer(
    const Transaction& transaction, const std::size_t layer,
    const LayerPoint point, Sm87TargetAotP40ExecutionResult* const result,
    const char* const context) noexcept {
  const auto recorded =
      RequestAccess::record_layer_completion(transaction, layer, point);
  if (!recorded) {
    result->status = status(
        Sm87TargetAotP40ExecutorError::kTransactionRecordFailure, context,
        layer, recorded.cuda_error, recorded.code);
    return false;
  }
  ++result->receipt.recorded_layer_events;
  return true;
}

[[nodiscard]] bool record_global(
    const Transaction& transaction, const GlobalPoint point,
    Sm87TargetAotP40ExecutionResult* const result,
    const char* const context) noexcept {
  const auto recorded = RequestAccess::record_global_completion(transaction,
                                                                 point);
  if (!recorded) {
    result->status = status(
        Sm87TargetAotP40ExecutorError::kTransactionRecordFailure, context,
        kSm87TargetAotP40LayerCount, recorded.cuda_error, recorded.code);
    return false;
  }
  ++result->receipt.recorded_global_events;
  return true;
}

[[nodiscard]] Sm87TargetAotP40ExecutionResult cancel_after_failure(
    const Transaction& transaction,
    Sm87TargetAotP40ExecutionResult result) noexcept {
  const auto cancelled = RequestAccess::cancel(transaction);
  if (!cancelled) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kTransactionCancelFailure,
        "cancel_and_drain_failed", result.status.layer,
        cancelled.cuda_error, cancelled.code);
    return result;
  }
  result.receipt.transaction_cancelled = true;
  return result;
}

[[nodiscard]] bool borrow_fp8_view(
    const ProjectionAccess& access, const std::size_t layer,
    const Sm87TargetAotProjectionRole role,
    kernels::Sm87TargetAotFp8CudaAssetView* const view) noexcept {
  if (view == nullptr) {
    return false;
  }
  auto asset = access.resolve(layer, role);
  if (!asset.has_value()) {
    return false;
  }
  const auto* const borrowed = asset->borrow_fp8_cuda_asset();
  if (borrowed == nullptr ||
      !kernels::sm87_target_aot_fp8_cuda_asset_valid(*borrowed)) {
    return false;
  }
  *view = *borrowed;
  return true;
}

[[nodiscard]] bool borrow_nvfp4_view(
    const ProjectionAccess& access, const std::size_t layer,
    const Sm87TargetAotProjectionRole role,
    kernels::Sm87TargetAotNvFp4CudaAssetView* const view) noexcept {
  if (view == nullptr) {
    return false;
  }
  auto asset = access.resolve(layer, role);
  if (!asset.has_value()) {
    return false;
  }
  const auto* const borrowed = asset->borrow_nvfp4_cuda_asset();
  if (borrowed == nullptr ||
      !kernels::sm87_target_aot_nvfp4_cuda_asset_valid(*borrowed)) {
    return false;
  }
  *view = *borrowed;
  return true;
}

}  // namespace

Sm87TargetAotP40Executor::~Sm87TargetAotP40Executor() {
  if (pinned_handoff_result_ != nullptr) {
    (void)cudaFreeHost(pinned_handoff_result_);
    pinned_handoff_result_ = nullptr;
    (void)cudaGetLastError();
  }
}

Sm87TargetAotP40ExecutorBindResult Sm87TargetAotP40Executor::bind(
    const ModelWeights& model_weights,
    Sm87TargetAotP40RequestState& request_owner,
    const Sm87TargetAotP40EngineRope& engine_rope) noexcept {
  Sm87TargetAotP40ExecutorBindResult result;
#if !defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION) ||       \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_REQUEST_STATE_V1_ADMISSION)
  (void)model_weights;
  (void)request_owner;
  (void)engine_rope;
  result.status = status(Sm87TargetAotP40ExecutorError::kAdmissionDisabled,
                         "executor_admission_not_compiled");
  return result;
#else
  if (!sm87_target_aot_p40_execution_contract().valid() ||
      !valid_model_weights(model_weights)) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kInvalidModelWeights,
        "complete_64_layer_weight_preflight");
    return result;
  }

  auto projection_access = ProjectionAccess::bind(model_weights);
  std::int32_t device_ordinal = -1;
  if (!projection_access.has_value() ||
      !complete_asset_preflight(*projection_access, &device_ordinal)) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kIncompleteProjectionAssets,
        "complete_256_asset_preflight");
    return result;
  }
  if (!complete_model_residency_preflight(model_weights, device_ordinal)) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kInvalidModelWeights,
        "complete_64_layer_device_residency_preflight");
    return result;
  }
  if (!snapshot_ready_for_bind(RequestAccess::snapshot(request_owner),
                               device_ordinal)) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kInvalidRequestOwner,
        "request_owner_static_preflight");
    return result;
  }
  if (!valid_engine_rope(engine_rope, device_ordinal)) {
    result.status = status(Sm87TargetAotP40ExecutorError::kInvalidEngineRope,
                           "engine_rope_static_preflight");
    return result;
  }
  if (!constituent_resources_preflight(device_ordinal)) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kStaticResourcePreflightFailure,
        "all_constituent_static_resources");
    return result;
  }

  void* pinned_handoff_result = nullptr;
  const cudaError_t host_status = cudaHostAlloc(
      &pinned_handoff_result, sizeof(Bf16GreedyArgmaxResult),
      cudaHostAllocPortable);
  if (host_status != cudaSuccess || pinned_handoff_result == nullptr) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kHostAllocationFailure,
        "executor_pinned_handoff_result", kSm87TargetAotP40LayerCount,
        static_cast<int>(host_status));
    return result;
  }
  std::memset(pinned_handoff_result, 0, sizeof(Bf16GreedyArgmaxResult));

  auto* const executor = new (std::nothrow) Sm87TargetAotP40Executor(
      model_weights, request_owner, engine_rope,
      std::move(*projection_access), device_ordinal,
      pinned_handoff_result);
  if (executor == nullptr) {
    (void)cudaFreeHost(pinned_handoff_result);
    (void)cudaGetLastError();
    result.status = status(
        Sm87TargetAotP40ExecutorError::kHostAllocationFailure,
        "executor_host_allocation");
    return result;
  }
  result.executor.reset(executor);
  return result;
#endif
}

Sm87TargetAotP40ExecutionResult Sm87TargetAotP40Executor::execute(
    const std::uint32_t* const host_prompt_token_ids,
    const std::size_t prompt_tokens,
    const std::size_t requested_handoff_tokens) noexcept {
  Sm87TargetAotP40ExecutionResult result;
  result.receipt.prompt_tokens = prompt_tokens;
  result.receipt.requested_handoff_tokens = requested_handoff_tokens;
#if !defined(Q3X_ENABLE_SM87_TARGET_AOT_P40_EXECUTOR_V1_ADMISSION) ||       \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION) || \
    !defined(Q3X_ENABLE_SM87_TARGET_AOT_REQUEST_STATE_V1_ADMISSION)
  (void)host_prompt_token_ids;
  result.status = status(Sm87TargetAotP40ExecutorError::kAdmissionDisabled,
                         "executor_admission_not_compiled");
  return result;
#else
  if (model_weights_ == nullptr || request_owner_ == nullptr ||
      host_prompt_token_ids == nullptr ||
      !sm87_target_aot_p40_exact_request_shape(prompt_tokens,
                                               requested_handoff_tokens)) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kInvalidExecuteInput,
        "exact_p40000_plus_one_handoff_required");
    return result;
  }
  for (std::size_t token = 0U; token < prompt_tokens; ++token) {
    if (host_prompt_token_ids[token] >= kVocabulary) {
      result.status = status(
          Sm87TargetAotP40ExecutorError::kInvalidExecuteInput,
          "prompt_token_out_of_vocabulary");
      return result;
    }
  }

  const auto* snapshot = RequestAccess::snapshot(*request_owner_);
  if (!projection_access_.attached() ||
      !snapshot_ready_for_bind(snapshot, device_ordinal_) ||
      !valid_engine_rope(engine_rope_, device_ordinal_) ||
      !valid_nvfp4_linear(model_weights_->lm_head(), kVocabulary,
                          kSm87TargetAotP40Hidden) ||
      pinned_handoff_result_ == nullptr ||
      !aligned(pinned_handoff_result_,
               alignof(Bf16GreedyArgmaxResult)) ||
      !complete_model_residency_preflight(*model_weights_, device_ordinal_)) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kLiveOwnerValidationFailure,
        "live_owner_asset_or_rope_revalidation");
    return result;
  }

  auto begun = RequestAccess::begin(*request_owner_);
  if (!begun || !begun.transaction.has_value()) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kTransactionBeginFailure,
        "begin_owner_bound_transaction", kSm87TargetAotP40LayerCount,
        begun.status.cuda_error, begun.status.code);
    return result;
  }
  const Transaction transaction = *begun.transaction;
  result.receipt.transaction_started = true;
  result.receipt.admission_epoch = transaction.admission_epoch();
  result.receipt.transaction_epoch = transaction.transaction_epoch();
  void* const stream = transaction.cuda_stream();
  if (stream == nullptr || transaction.device_cancellation_signal() == nullptr) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kLiveOwnerValidationFailure,
        "owner_stream_or_cancellation_control_missing");
    return cancel_after_failure(transaction, std::move(result));
  }

  auto cuda_failure = [&](const int code, const char* const context,
                          const std::size_t layer =
                              kSm87TargetAotP40LayerCount) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kCudaOperationFailure, context, layer,
        code);
    return cancel_after_failure(transaction, std::move(result));
  };
  auto record_failure = [&]() {
    return cancel_after_failure(transaction, std::move(result));
  };

  const cudaError_t copy_status = cudaMemcpyAsync(
      snapshot->token_ids.device_data, host_prompt_token_ids,
      prompt_tokens * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
      reinterpret_cast<cudaStream_t>(stream));
  if (copy_status != cudaSuccess) {
    return cuda_failure(static_cast<int>(copy_status), "prompt_token_h2d");
  }
  if (!record_global(transaction, GlobalPoint::kTokenIdsReady, &result,
                     "token_ids_ready")) {
    return record_failure();
  }

  const auto& embedding = model_weights_->embed_tokens();
  int launch_status = launch_embedding_gather_prompt_reference_cuda(
      embedding.weight, embedding.output_size, embedding.input_size,
      static_cast<const std::uint32_t*>(snapshot->token_ids.device_data),
      prompt_tokens,
      static_cast<std::uint16_t*>(snapshot->layers[0U].residual.device_data),
      stream);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return cuda_failure(launch_status, "embedding_prompt_gather");
  }
  if (!record_global(transaction, GlobalPoint::kEmbeddingComplete, &result,
                     "embedding_complete")) {
    return record_failure();
  }

  for (std::size_t layer_index = 0U;
       layer_index < kSm87TargetAotP40LayerCount; ++layer_index) {
    const auto& layer_weights = model_weights_->layer(layer_index);
    const auto& layer = snapshot->layers[layer_index];
    auto* const residual =
        static_cast<std::uint16_t*>(layer.residual.device_data);
    auto* const normalized =
        static_cast<std::uint16_t*>(layer.input_normalized.device_data);

    launch_status = launch_headwise_centered_rms_norm_reference_cuda(
        residual, layer_weights.input_layernorm.data, prompt_tokens,
        kSm87TargetAotP40Hidden, kRmsEpsilon, normalized, stream);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      return cuda_failure(launch_status, "layer_input_centered_rms",
                          layer_index);
    }

    if (layer.kind == Sm87TargetAotRequestLayerKind::kGdn) {
      const auto* const attention =
          std::get_if<LinearAttentionWeights>(&layer_weights.attention);
      const auto* const a_weight =
          attention == nullptr
              ? nullptr
              : std::get_if<Bf16LinearWeight>(&attention->in_proj_a);
      const auto* const b_weight =
          attention == nullptr
              ? nullptr
              : std::get_if<Bf16LinearWeight>(&attention->in_proj_b);
      kernels::Sm87TargetAotFp8CudaAssetView input_asset{};
      kernels::Sm87TargetAotFp8CudaAssetView output_asset{};
      if (attention == nullptr || a_weight == nullptr || b_weight == nullptr ||
          !borrow_fp8_view(projection_access_, layer_index,
                           Sm87TargetAotProjectionRole::kFp8GdnQkvZ,
                           &input_asset) ||
          !borrow_fp8_view(projection_access_, layer_index,
                           Sm87TargetAotProjectionRole::kFp8AttentionOutput,
                           &output_asset)) {
        result.status = status(
            Sm87TargetAotP40ExecutorError::kLiveAssetValidationFailure,
            "gdn_projection_asset_reborrow", layer_index);
        return record_failure();
      }

      kernels::Sm87TargetAotFp8CudaArguments input_arguments{};
      input_arguments.role =
          Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
      input_arguments.input = normalized;
      input_arguments.asset = input_asset;
      input_arguments.token_count = prompt_tokens;
      input_arguments.output =
          static_cast<std::uint16_t*>(layer.gdn.raw_qkvz.device_data);
      input_arguments.cuda_stream = stream;
      launch_status = kernels::sm87_target_aot_projection_execution_detail::
          launch_authenticated_fp8(input_arguments);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "gdn_fp8_qkvz", layer_index);
      }
      if (!record_layer(transaction, layer_index,
                        LayerPoint::kInputProjections, &result,
                        "gdn_qkvz_projection_complete")) {
        return record_failure();
      }

      // The owner stream is deliberately serial in v1.  The contract reserves
      // the QKVZ/AB producer-lane and join-event shape, but no event is recorded
      // as a claim of parallel readiness before both real launches complete.
      launch_status =
          kernels::sm87_target_aot_bf16_ab_execution_detail::
              launch_interleaved_p40(
                  a_weight->weight, b_weight->weight, normalized,
                  static_cast<std::uint16_t*>(layer.gdn.bf16_ab.device_data),
                  stream);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "gdn_bf16_ab_interleaved",
                            layer_index);
      }
      if (!record_layer(transaction, layer_index,
                        LayerPoint::kStateOrAttentionPreparation, &result,
                        "gdn_ab_producer_complete")) {
        return record_failure();
      }

      kernels::Sm87TargetAotGdnCudaArguments gdn{};
      gdn.raw_qkvz = static_cast<const std::uint16_t*>(
          layer.gdn.raw_qkvz.device_data);
      gdn.interleaved_ab = static_cast<const std::uint16_t*>(
          layer.gdn.bf16_ab.device_data);
      gdn.conv_weight = attention->conv1d.data;
      gdn.a_log = attention->a_log.data;
      gdn.dt_bias = attention->dt_bias.data;
      gdn.norm_weight = attention->norm.data;
      gdn.l2_epsilon_fp32_bits = kernels::kSm87TargetAotGdnEpsilonFp32Bits;
      gdn.norm_epsilon_fp32_bits =
          kernels::kSm87TargetAotGdnEpsilonFp32Bits;
      gdn.first_position = 0U;
      gdn.token_count = prompt_tokens;
      gdn.output = static_cast<std::uint16_t*>(layer.gdn.output.device_data);
      gdn.final_conv_history = static_cast<std::uint16_t*>(
          layer.gdn.final_conv_history.device_data);
      gdn.final_recurrent_state = static_cast<std::uint16_t*>(
          layer.gdn.final_recurrent_state.device_data);
      gdn.cancellation_signal = transaction.device_cancellation_signal();
      gdn.cuda_stream = stream;
      launch_status = kernels::sm87_target_aot_gdn_execution_detail::
          launch_authenticated(gdn);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "target_gdn_core", layer_index);
      }
      if (!record_layer(transaction, layer_index, LayerPoint::kCore, &result,
                        "gdn_core_complete")) {
        return record_failure();
      }

      kernels::Sm87TargetAotFp8CudaArguments output_arguments{};
      output_arguments.role =
          Sm87TargetAotProjectionRole::kFp8AttentionOutput;
      output_arguments.input = gdn.output;
      output_arguments.asset = output_asset;
      output_arguments.token_count = prompt_tokens;
      output_arguments.output =
          static_cast<std::uint16_t*>(layer.gdn.o_branch.device_data);
      output_arguments.cuda_stream = stream;
      launch_status = kernels::sm87_target_aot_projection_execution_detail::
          launch_authenticated_fp8(output_arguments);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "gdn_fp8_output_projection",
                            layer_index);
      }
      if (!record_layer(transaction, layer_index,
                        LayerPoint::kOutputProjection, &result,
                        "gdn_output_projection_complete")) {
        return record_failure();
      }

      launch_status = launch_residual_add_reference_cuda(
          residual,
          static_cast<const std::uint16_t*>(layer.gdn.o_branch.device_data),
          prompt_tokens * kSm87TargetAotP40Hidden, residual, stream);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "gdn_post_core_residual",
                            layer_index);
      }
      ++result.receipt.completed_gdn_layers;
    } else if (layer.kind ==
               Sm87TargetAotRequestLayerKind::kFullAttention) {
      const auto* const attention =
          std::get_if<FullAttentionWeights>(&layer_weights.attention);
      kernels::Sm87TargetAotFp8CudaAssetView input_asset{};
      kernels::Sm87TargetAotFp8CudaAssetView output_asset{};
      if (attention == nullptr ||
          !borrow_fp8_view(projection_access_, layer_index,
                           Sm87TargetAotProjectionRole::kFp8FullQkv,
                           &input_asset) ||
          !borrow_fp8_view(projection_access_, layer_index,
                           Sm87TargetAotProjectionRole::kFp8AttentionOutput,
                           &output_asset)) {
        result.status = status(
            Sm87TargetAotP40ExecutorError::kLiveAssetValidationFailure,
            "full_attention_projection_asset_reborrow", layer_index);
        return record_failure();
      }

      kernels::sm87_target_aot_projection_execution_detail::
          Sm87TargetAotFp8FullQkvScatterArguments scatter{};
      scatter.input = normalized;
      scatter.asset = input_asset;
      scatter.token_count = prompt_tokens;
      scatter.q_gate_output = static_cast<std::uint16_t*>(
          layer.attention.raw_q_gate.device_data);
      scatter.key_output =
          static_cast<std::uint16_t*>(layer.attention.raw_k.device_data);
      scatter.value_output =
          static_cast<std::uint16_t*>(layer.attention.raw_v.device_data);
      scatter.cuda_stream = stream;
      launch_status = kernels::sm87_target_aot_projection_execution_detail::
          launch_authenticated_fp8_full_qkv_scatter(scatter);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "full_fp8_qkv_scatter",
                            layer_index);
      }
      if (!record_layer(transaction, layer_index,
                        LayerPoint::kInputProjections, &result,
                        "full_qkv_projection_complete")) {
        return record_failure();
      }

      for (std::size_t panel = 0U;
           panel < kSm87TargetAotP40FullAttentionPanels; ++panel) {
        const std::size_t first_position =
            panel * kSm87TargetAotP40FullAttentionPanelTokens;
        if (!can_launch_full_attention_preprocess_prompt_wide_p8000(
                first_position,
                kSm87TargetAotP40FullAttentionPanelTokens)) {
          result.status = status(
              Sm87TargetAotP40ExecutorError::kLiveOwnerValidationFailure,
              "full_attention_p8000_panel_contract", layer_index);
          return record_failure();
        }
        auto* const raw_q_gate = static_cast<std::uint16_t*>(
            layer.attention.raw_q_gate.device_data);
        auto* const key = static_cast<std::uint16_t*>(
            layer.attention.raw_k.device_data);
        auto* const processed_q = static_cast<std::uint16_t*>(
            layer.attention.processed_q.device_data);
        auto* const processed_gate = static_cast<std::uint16_t*>(
            layer.attention.processed_gate.device_data);
        launch_status =
            sm87_target_aot_attention_preprocess_execution_detail::
                launch_p8000(
                raw_q_gate + first_position *
                                 kSm87TargetAotP40AttentionQGateWidth,
                key + first_position * kSm87TargetAotP40KvWidth,
                attention->q_norm.data, attention->k_norm.data, kRmsEpsilon,
                processed_q + first_position *
                                  kSm87TargetAotP40AttentionWidth,
                processed_gate + first_position *
                                     kSm87TargetAotP40AttentionWidth,
                engine_rope_.cosines, engine_rope_.sines, first_position,
                kSm87TargetAotP40FullAttentionPanelTokens, stream);
        if (launch_status != static_cast<int>(cudaSuccess)) {
          return cuda_failure(launch_status, "full_attention_p8000_preprocess",
                              layer_index);
        }
        ++result.receipt.completed_attention_panels;
      }
      if (!record_layer(transaction, layer_index,
                        LayerPoint::kStateOrAttentionPreparation, &result,
                        "full_attention_preprocess_complete")) {
        return record_failure();
      }

      sm87_target_aot_attention_execution_detail::TargetP40Arguments
          attention_arguments{};
      attention_arguments.processed_query =
          static_cast<const std::uint16_t*>(
              layer.attention.processed_q.device_data);
      attention_arguments.processed_key =
          static_cast<const std::uint16_t*>(
              layer.attention.processed_k.device_data);
      attention_arguments.processed_value =
          static_cast<const std::uint16_t*>(
              layer.attention.processed_v.device_data);
      attention_arguments.processed_gate =
          static_cast<const std::uint16_t*>(
              layer.attention.processed_gate.device_data);
      attention_arguments.gated_output = static_cast<std::uint16_t*>(
          layer.attention.gated_output.device_data);
      attention_arguments.token_count = prompt_tokens;
      attention_arguments.device_ordinal = device_ordinal_;
      attention_arguments.cuda_stream = stream;
      launch_status = sm87_target_aot_attention_execution_detail::
          launch_q128_kv32_p40_two_stage(attention_arguments);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "target_full_attention_core",
                            layer_index);
      }
      if (!record_layer(transaction, layer_index, LayerPoint::kCore, &result,
                        "full_attention_core_complete")) {
        return record_failure();
      }

      kernels::Sm87TargetAotFp8CudaArguments output_arguments{};
      output_arguments.role =
          Sm87TargetAotProjectionRole::kFp8AttentionOutput;
      output_arguments.input = attention_arguments.gated_output;
      output_arguments.asset = output_asset;
      output_arguments.token_count = prompt_tokens;
      output_arguments.output = static_cast<std::uint16_t*>(
          layer.attention.o_branch.device_data);
      output_arguments.cuda_stream = stream;
      launch_status = kernels::sm87_target_aot_projection_execution_detail::
          launch_authenticated_fp8(output_arguments);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "full_fp8_output_projection",
                            layer_index);
      }
      if (!record_layer(transaction, layer_index,
                        LayerPoint::kOutputProjection, &result,
                        "full_output_projection_complete")) {
        return record_failure();
      }

      launch_status = launch_residual_add_reference_cuda(
          residual,
          static_cast<const std::uint16_t*>(
              layer.attention.o_branch.device_data),
          prompt_tokens * kSm87TargetAotP40Hidden, residual, stream);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return cuda_failure(launch_status, "full_post_core_residual",
                            layer_index);
      }
      ++result.receipt.completed_full_attention_layers;
    } else {
      result.status = status(
          Sm87TargetAotP40ExecutorError::kLiveOwnerValidationFailure,
          "invalid_layer_kind", layer_index);
      return record_failure();
    }

    if (!record_layer(transaction, layer_index,
                      LayerPoint::kPostCoreResidual, &result,
                      "post_core_residual_complete")) {
      return record_failure();
    }

    auto* const mlp_normalized =
        static_cast<std::uint16_t*>(layer.mlp.normalized_input.device_data);
    launch_status = launch_headwise_centered_rms_norm_reference_cuda(
        residual, layer_weights.post_attention_layernorm.data, prompt_tokens,
        kSm87TargetAotP40Hidden, kRmsEpsilon, mlp_normalized, stream);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      return cuda_failure(launch_status, "post_attention_centered_rms",
                          layer_index);
    }

    kernels::Sm87TargetAotNvFp4CudaAssetView gate_up_asset{};
    kernels::Sm87TargetAotNvFp4CudaAssetView down_asset{};
    if (!borrow_nvfp4_view(projection_access_, layer_index,
                           Sm87TargetAotProjectionRole::kNvFp4GateUp,
                           &gate_up_asset) ||
        !borrow_nvfp4_view(projection_access_, layer_index,
                           Sm87TargetAotProjectionRole::kNvFp4Down,
                           &down_asset)) {
      result.status = status(
          Sm87TargetAotP40ExecutorError::kLiveAssetValidationFailure,
          "mlp_projection_asset_reborrow", layer_index);
      return record_failure();
    }

    kernels::Sm87TargetAotNvFp4CudaArguments gate_up{};
    gate_up.role = Sm87TargetAotProjectionRole::kNvFp4GateUp;
    gate_up.input = mlp_normalized;
    gate_up.asset = gate_up_asset;
    gate_up.token_count = prompt_tokens;
    gate_up.output_or_residual =
        static_cast<std::uint16_t*>(layer.mlp.activated.device_data);
    gate_up.cuda_stream = stream;
    launch_status = kernels::sm87_target_aot_projection_execution_detail::
        launch_authenticated_nvfp4(gate_up);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      return cuda_failure(launch_status, "nvfp4_gate_up", layer_index);
    }
    if (!record_layer(transaction, layer_index, LayerPoint::kGateUp, &result,
                      "gate_up_complete")) {
      return record_failure();
    }

    kernels::Sm87TargetAotNvFp4CudaArguments down{};
    down.role = Sm87TargetAotProjectionRole::kNvFp4Down;
    down.input = static_cast<const std::uint16_t*>(
        layer.mlp.activated.device_data);
    down.asset = down_asset;
    down.token_count = prompt_tokens;
    down.output_or_residual = residual;
    down.cuda_stream = stream;
    launch_status = kernels::sm87_target_aot_projection_execution_detail::
        launch_authenticated_nvfp4(down);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      return cuda_failure(launch_status, "nvfp4_down_residual", layer_index);
    }
    if (!record_layer(transaction, layer_index, LayerPoint::kDown, &result,
                      "down_complete") ||
        !record_layer(transaction, layer_index, LayerPoint::kLayerComplete,
                      &result, "layer_complete")) {
      return record_failure();
    }
    ++result.receipt.completed_layers;
  }

  if (!record_global(transaction, GlobalPoint::kAllLayersComplete, &result,
                     "all_layers_complete")) {
    return record_failure();
  }

  FinalHandoffScratchView handoff_scratch{};
  if (!borrow_final_handoff_scratch(*snapshot, &handoff_scratch)) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kLiveOwnerValidationFailure,
        "terminal_mlp_family_scratch_lifetime");
    return record_failure();
  }

  const auto* const last_residual =
      static_cast<const std::uint16_t*>(
          snapshot->layers[kSm87TargetAotP40LayerCount - 1U]
              .residual.device_data) +
      (kSm87TargetAotP40PromptTokens - 1U) * kSm87TargetAotP40Hidden;
  launch_status = launch_centered_rms_norm_reference_cuda(
      last_residual, model_weights_->final_norm().data,
      kSm87TargetAotP40Hidden, kRmsEpsilon,
      static_cast<std::uint16_t*>(snapshot->final_hidden.device_data), stream);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return cuda_failure(launch_status, "final_last_row_centered_rms");
  }
  result.receipt.finalization =
      Sm87TargetAotP40Finalization::kFinalHiddenReady;
  if (!record_global(transaction, GlobalPoint::kFinalNormComplete, &result,
                     "final_norm_complete") ||
      !record_global(transaction, GlobalPoint::kFinalHiddenComplete, &result,
                     "final_hidden_complete")) {
    return record_failure();
  }

  // LM head is not a request-time target-AOT sidecar.  It is the authenticated
  // resident checkpoint's exact NVFP4 [248320,5120] weight, and this private
  // source fixes the already-qualified production M1 dispatcher to the SM87
  // weight-only backend.  No generic backend, cuBLASLt, fallback, or runtime
  // tactic selection is reachable from this call site.
  const auto* const lm_head =
      std::get_if<NvFp4LinearWeight>(&model_weights_->lm_head());
  if (lm_head == nullptr || lm_head->output_size != kVocabulary ||
      lm_head->input_size != kSm87TargetAotP40Hidden) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kLiveAssetValidationFailure,
        "exact_nvfp4_lm_head_revalidation");
    return record_failure();
  }
  launch_status = launch_projection_to_bf16_cuda(
      ProjectionBackend::kSm87WeightOnly, model_weights_->lm_head(),
      static_cast<const std::uint16_t*>(
          snapshot->final_hidden.device_data),
      nullptr, 0U, handoff_scratch.logits, stream);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return cuda_failure(launch_status, "nvfp4_lm_head_m1_bf16");
  }
  result.receipt.finalization =
      Sm87TargetAotP40Finalization::kLogitsReady;

  launch_status = launch_bf16_greedy_argmax_cuda(
      handoff_scratch.logits, kVocabulary,
      handoff_scratch.greedy_workspace, stream);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return cuda_failure(launch_status, "bf16_greedy_argmax");
  }
  std::memset(pinned_handoff_result_, 0,
              sizeof(Bf16GreedyArgmaxResult));
  cudaError_t handoff_status = cudaMemcpyAsync(
      pinned_handoff_result_, handoff_scratch.greedy_workspace,
      sizeof(Bf16GreedyArgmaxResult), cudaMemcpyDeviceToHost,
      reinterpret_cast<cudaStream_t>(stream));
  if (handoff_status != cudaSuccess) {
    return cuda_failure(static_cast<int>(handoff_status),
                        "greedy_handoff_d2h");
  }

  // The host must validate non-finite logits and the selected vocabulary
  // index before the request-wide publication event.  This one terminal
  // synchronization also proves the pinned D2H has completed before either
  // the receipt or the executor-owned 8-byte host buffer can be observed.
  handoff_status =
      cudaStreamSynchronize(reinterpret_cast<cudaStream_t>(stream));
  if (handoff_status != cudaSuccess) {
    return cuda_failure(static_cast<int>(handoff_status),
                        "greedy_handoff_synchronize");
  }
  const auto greedy =
      *static_cast<const Bf16GreedyArgmaxResult*>(pinned_handoff_result_);
  result.receipt.handoff_token_id = greedy.index;
  result.receipt.handoff_value_bits = greedy.value_bits;
  result.receipt.handoff_has_nonfinite = greedy.has_nonfinite;
  result.receipt.handoff_result_observed = true;
  if (greedy.has_nonfinite != 0U) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kInvalidHandoffResult,
        "nonfinite_bf16_logits");
    return record_failure();
  }
  if (greedy.index >= kVocabulary) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kInvalidHandoffResult,
        "greedy_token_out_of_vocabulary");
    return record_failure();
  }
  result.receipt.finalization =
      Sm87TargetAotP40Finalization::kHandoffReady;

  // Persistent KV/GDN state was written transaction-privately throughout the
  // layer loop.  This terminal event now covers those writes together with
  // final_hidden, logits reduction, and the validated one-token handoff.
  if (!record_global(transaction, GlobalPoint::kPersistentStateStaged,
                     &result, "persistent_state_and_handoff_staged")) {
    return record_failure();
  }
  const auto committed = RequestAccess::commit(transaction);
  if (!committed) {
    result.status = status(
        Sm87TargetAotP40ExecutorError::kTransactionRecordFailure,
        "request_wide_prefill_commit", kSm87TargetAotP40LayerCount,
        committed.cuda_error, committed.code);
    return record_failure();
  }
  ++result.receipt.recorded_global_events;
  result.receipt.transaction_committed = true;
  result.receipt.handoff_complete = true;
  result.receipt.finalization =
      Sm87TargetAotP40Finalization::kCommitted;
  return result;
#endif
}

}  // namespace q3x::runtime::sm87_target_aot_p40_executor_detail
