#include "sm87_bulk_dataflow_v2_p40_executor_internal.h"

#include "q3x/kernels/sm87_bulk_dataflow_v2_fp8_whole_p40.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_down_whole_p40.h"
#include "q3x/kernels/sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40.h"
#include "../kernels/sm87/sm87_target_aot_attention_preprocess_launch_internal.h"

#include <cuda_runtime_api.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <optional>
#include <utility>
#include <variant>

namespace q3x::runtime::sm87_bulk_v2_p40_executor_detail {
namespace {

namespace composition = sm87_bulk_v2_p40_composition_detail;
namespace owner = sm87_bulk_v2_p40_owner_detail;
namespace bf16_ab =
    q3x::kernels::sm87_target_aot_bf16_ab_execution_detail;
namespace attention =
    q3x::kernels::sm87_bulk_v2_attention_execution_detail;

using q3x::kernels::Sm87TargetAotProjectionRole;
using Stream = Sm87BulkV2P40Stream;
using Event = Sm87BulkV2P40ReusableEvent;
using Counter = owner::Sm87BulkV2P40SubmissionCounter;
using Family = Sm87BulkV2P40FamilyPhase;

inline constexpr float kRmsEpsilon = 1.0e-6F;
inline constexpr std::size_t kAttentionPanelTokens = 8'000U;
inline constexpr std::size_t kAttentionPanels = 5U;

[[nodiscard]] Sm87BulkV2P40ExecutorStatus failure(
    const Sm87BulkV2P40ExecutorError error, const char* const context,
    const int cuda_error = 0,
    const std::size_t layer = kSm87BulkV2P40Layers,
    const std::size_t segment = 0U,
    const std::size_t constituent = 0U) noexcept {
  return {error, context, cuda_error, layer, segment, constituent};
}

template <typename T>
[[nodiscard]] T* byte_offset(void* const base,
                             const std::uint64_t offset) noexcept {
  if (base == nullptr ||
      offset > static_cast<std::uint64_t>(
                   std::numeric_limits<std::uintptr_t>::max())) {
    return nullptr;
  }
  const auto address = reinterpret_cast<std::uintptr_t>(base);
  if (address > std::numeric_limits<std::uintptr_t>::max() -
                    static_cast<std::uintptr_t>(offset)) {
    return nullptr;
  }
  return reinterpret_cast<T*>(address + static_cast<std::uintptr_t>(offset));
}

template <typename T>
[[nodiscard]] const T* byte_offset(const void* const base,
                                   const std::uint64_t offset) noexcept {
  return byte_offset<const T>(const_cast<void*>(base), offset);
}

[[nodiscard]] bool cancellation_requested(
    const Sm87BulkV2P40ExecutionControl& control) noexcept {
  return control.cancellation_probe != nullptr &&
         control.cancellation_probe(control.cancellation_context);
}

[[nodiscard]] bool fp8_asset_valid(
    const kernels::Sm87TargetAotFp8CudaAssetView& asset,
    const Sm87TargetAotProjectionRole role) noexcept {
  return kernels::sm87_target_aot_fp8_cuda_asset_valid(asset) &&
         asset.payload.role == role;
}

[[nodiscard]] bool nvfp4_asset_valid(
    const kernels::Sm87TargetAotNvFp4CudaAssetView& asset,
    const Sm87TargetAotProjectionRole role) noexcept {
  return kernels::sm87_target_aot_nvfp4_cuda_asset_valid(asset) &&
         asset.payload.role == role;
}

}  // namespace

Sm87BulkV2P40Executor::Sm87BulkV2P40Executor(
    std::unique_ptr<CompositionRoot> composition,
    const sm87_target_aot_p40_executor_detail::Sm87TargetAotP40EngineRope rope,
    const std::array<LayerAssets, kSm87BulkV2P40Layers>& assets) noexcept
    : composition_(std::move(composition)), rope_(rope), assets_(assets) {}

Sm87BulkV2P40Executor::~Sm87BulkV2P40Executor() = default;

bool Sm87BulkV2P40Executor::ready() const noexcept {
  if (terminal_failure_ || composition_ == nullptr || !composition_->valid() ||
      composition_->owner() == nullptr ||
      composition_->request_state() == nullptr ||
      composition_->request_access() == nullptr ||
      composition_->execution_access() == nullptr ||
      composition_->projection_access() == nullptr ||
      composition_->whole_projection_access() == nullptr ||
      composition_->bf16_ab_access() == nullptr ||
      composition_->attention_access() == nullptr ||
      composition_->gdn_session() == nullptr ||
      composition_->gdn_receipt() == nullptr || rope_.cosines == nullptr ||
      rope_.sines == nullptr || rope_.identity == 0U ||
      rope_.position_count < kSm87BulkV2P40Tokens ||
      rope_.rotary_pairs != 32U || next_request_epoch_ == 0U) {
    return false;
  }
  for (std::size_t layer = 0U; layer < assets_.size(); ++layer) {
    const auto input_role = sm87_bulk_v2_p40_is_full_layer(layer)
                                ? Sm87TargetAotProjectionRole::kFp8FullQkv
                                : Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    if (!fp8_asset_valid(assets_[layer].input_fp8, input_role) ||
        !fp8_asset_valid(
            assets_[layer].output_fp8,
            Sm87TargetAotProjectionRole::kFp8AttentionOutput) ||
        !nvfp4_asset_valid(assets_[layer].gate_up,
                           Sm87TargetAotProjectionRole::kNvFp4GateUp) ||
        !nvfp4_asset_valid(assets_[layer].down,
                           Sm87TargetAotProjectionRole::kNvFp4Down)) {
      return false;
    }
  }
  return true;
}

Sm87BulkV2P40ExecutorCreateResult
create_sm87_bulk_dataflow_v2_p40_executor(
    std::unique_ptr<composition::Sm87BulkV2P40CompositionRoot> composition,
    const sm87_target_aot_p40_executor_detail::
        Sm87TargetAotEngineRopeOwner& rope_owner) noexcept {
  Sm87BulkV2P40ExecutorCreateResult result;
#if !defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_EXECUTOR_ADMISSION)
  (void)composition;
  (void)rope_owner;
  result.status = failure(Sm87BulkV2P40ExecutorError::kAdmissionDisabled,
                          "bulk_v2_p40_executor_admission");
  return result;
#else
  if (composition == nullptr || !composition->valid() ||
      composition->projection_access() == nullptr ||
      composition->owner() == nullptr) {
    result.status = failure(Sm87BulkV2P40ExecutorError::kInvalidComposition,
                            "complete_composition_root_required");
    return result;
  }
  const auto rope = rope_owner.view();
  if (rope.cosines == nullptr || rope.sines == nullptr ||
      rope.identity == 0U || rope.position_count < kSm87BulkV2P40Tokens ||
      rope.rotary_pairs != 32U ||
      rope_owner.device_ordinal() != composition->owner()->device_ordinal()) {
    result.status = failure(Sm87BulkV2P40ExecutorError::kInvalidEngineRope,
                            "engine_owned_qwen36_rope_required");
    return result;
  }

  std::array<Sm87BulkV2P40Executor::LayerAssets,
             kSm87BulkV2P40Layers>
      assets{};
  const auto* const projection = composition->projection_access();
  for (std::size_t layer = 0U; layer < assets.size(); ++layer) {
    const auto input_role = sm87_bulk_v2_p40_is_full_layer(layer)
                                ? Sm87TargetAotProjectionRole::kFp8FullQkv
                                : Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    const auto input = projection->resolve(layer, input_role);
    const auto output = projection->resolve(
        layer, Sm87TargetAotProjectionRole::kFp8AttentionOutput);
    const auto gate = projection->resolve(
        layer, Sm87TargetAotProjectionRole::kNvFp4GateUp);
    const auto down = projection->resolve(
        layer, Sm87TargetAotProjectionRole::kNvFp4Down);
    if (!input || !output || !gate || !down ||
        input->borrow_fp8_cuda_asset() == nullptr ||
        output->borrow_fp8_cuda_asset() == nullptr ||
        gate->borrow_nvfp4_cuda_asset() == nullptr ||
        down->borrow_nvfp4_cuda_asset() == nullptr) {
      result.status = failure(
          Sm87BulkV2P40ExecutorError::kIncompleteProjectionAssets,
          "complete_256_asset_catalog_reborrow", 0, layer);
      return result;
    }
    assets[layer].input_fp8 = *input->borrow_fp8_cuda_asset();
    assets[layer].output_fp8 = *output->borrow_fp8_cuda_asset();
    assets[layer].gate_up = *gate->borrow_nvfp4_cuda_asset();
    assets[layer].down = *down->borrow_nvfp4_cuda_asset();
  }

  auto* const executor = new (std::nothrow) Sm87BulkV2P40Executor(
      std::move(composition), rope, assets);
  if (executor == nullptr) {
    result.status = failure(Sm87BulkV2P40ExecutorError::kAllocation,
                            "bulk_v2_p40_executor_allocation");
    return result;
  }
  result.executor.reset(executor);
  if (!result.executor->ready()) {
    result.executor.reset();
    result.status = failure(Sm87BulkV2P40ExecutorError::kInvalidComposition,
                            "bulk_v2_p40_executor_postcondition");
    return result;
  }
  result.status = {};
  return result;
#endif
}

Sm87BulkV2P40ExecutionResult Sm87BulkV2P40Executor::execute(
    const std::uint32_t* const host_prompt_token_ids,
    const std::size_t prompt_tokens,
    const Sm87BulkV2P40ExecutionControl& control) noexcept {
  Sm87BulkV2P40ExecutionResult result;
#if !defined(Q3X_ENABLE_SM87_BULK_DATAFLOW_V2_P40_EXECUTOR_ADMISSION)
  (void)host_prompt_token_ids;
  (void)prompt_tokens;
  (void)control;
  result.status = failure(Sm87BulkV2P40ExecutorError::kAdmissionDisabled,
                          "bulk_v2_p40_executor_admission");
  return result;
#else
  if (!ready() || host_prompt_token_ids == nullptr ||
      prompt_tokens != kSm87BulkV2P40Tokens ||
      (control.cancellation_probe == nullptr &&
       control.cancellation_context != nullptr)) {
    result.status = failure(Sm87BulkV2P40ExecutorError::kInvalidInput,
                            "exact_p40000_cold_request_required");
    return result;
  }
  for (std::size_t token = 0U; token < prompt_tokens; ++token) {
    if (host_prompt_token_ids[token] >= kSm87BulkV2P40Vocabulary) {
      result.status = failure(Sm87BulkV2P40ExecutorError::kInvalidInput,
                              "prompt_token_out_of_vocabulary", 0,
                              kSm87BulkV2P40Layers, token);
      return result;
    }
  }
  if (cancellation_requested(control)) {
    result.status = failure(Sm87BulkV2P40ExecutorError::kCancelled,
                            "cancelled_before_bulk_v2_submission");
    return result;
  }

  auto* const owner_state = composition_->owner();
  auto* const request_state = composition_->request_state();
  const auto* const request_access = composition_->request_access();
  const auto* const access = composition_->execution_access();
  const auto* const model = composition_->model_weights();
  auto* const gdn_session = composition_->gdn_session();
  if (owner_state == nullptr || request_state == nullptr ||
      request_access == nullptr || access == nullptr || model == nullptr ||
      gdn_session == nullptr) {
    terminal_failure_ = true;
    result.status = failure(Sm87BulkV2P40ExecutorError::kInvalidComposition,
                            "live_composition_binding_missing");
    return result;
  }

  if (request_state->lifecycle() ==
      owner::Sm87BulkV2P40RequestStateLifecycle::kCompleted) {
    const auto rearmed = request_state->rearm_for_cold_request(*request_access);
    if (!rearmed) {
      terminal_failure_ = true;
      result.status = failure(Sm87BulkV2P40ExecutorError::kRequestRearm,
                              rearmed.status.context,
                              rearmed.status.cuda_error);
      return result;
    }
    const auto gdn_rearmed =
        composition_->hot_rearm_gdn_session_after_completed_request();
    if (!gdn_rearmed) {
      terminal_failure_ = true;
      result.status = failure(Sm87BulkV2P40ExecutorError::kGdnSessionRearm,
                              gdn_rearmed.context,
                              gdn_rearmed.cuda_error);
      return result;
    }
  }
  if (gdn_session->lifecycle !=
      kernels::Sm87BulkV2GdnP40SessionLifecycle::kReady) {
    terminal_failure_ = true;
    result.status = failure(Sm87BulkV2P40ExecutorError::kGdnSessionRearm,
                            "owner_bound_gdn_session_rearm_required");
    return result;
  }

  const std::uint64_t request_epoch = next_request_epoch_;
  if (request_epoch == 0U) {
    terminal_failure_ = true;
    result.status = failure(Sm87BulkV2P40ExecutorError::kTransactionBegin,
                            "request_epoch_exhausted");
    return result;
  }
  const auto begun = owner_state->begin_request(
      *access, *request_state, *request_access, request_epoch);
  if (!begun) {
    terminal_failure_ = true;
    result.status = failure(Sm87BulkV2P40ExecutorError::kTransactionBegin,
                            begun.context, begun.cuda_error);
    return result;
  }
  ++next_request_epoch_;
  result.request_epoch = request_epoch;

  const auto poison = [&](const Sm87BulkV2P40ExecutorError code,
                          const char* const context,
                          const int cuda_error,
                          const std::size_t layer,
                          const Family family,
                          const std::size_t segment,
                          const std::size_t constituent) noexcept {
    terminal_failure_ = true;
    const int effective_error =
        cuda_error != 0 ? cuda_error : static_cast<int>(cudaErrorInvalidValue);
    if (owner_state->state() == owner::Sm87BulkV2P40OwnerState::kActive) {
      (void)owner_state->poison_after_submission_failure(
          *access, effective_error,
          layer < kSm87BulkV2P40Layers ? layer
                                       : kSm87BulkV2P40Layers - 1U,
          family == Family::kCount ? Family::kFinalHandoff : family,
          segment, constituent);
    }
    result.receipt = owner_state->receipt();
    result.status = failure(code, context, effective_error, layer, segment,
                            constituent);
    return result;
  };
  const auto cancelled = [&](const char* const context,
                             const std::size_t layer) noexcept {
    terminal_failure_ = true;
    if (owner_state->state() == owner::Sm87BulkV2P40OwnerState::kActive) {
      (void)owner_state->cancel_request(*access);
    }
    result.receipt = owner_state->receipt();
    result.status = failure(Sm87BulkV2P40ExecutorError::kCancelled, context,
                            0, layer);
    return result;
  };
  const auto note = [&](const Stream stream, const Counter counter,
                        const std::size_t count, const std::size_t layer,
                        const Family family, const std::size_t segment,
                        const std::size_t constituent) noexcept {
    return owner_state->note_submission(*access, stream, counter, count, layer,
                                        family, segment, constituent);
  };
  const auto record = [&](const Stream stream, const Event event) noexcept {
    return owner_state->record_event(*access, stream, event);
  };
  const auto wait = [&](const Stream stream, const Event event) noexcept {
    return owner_state->wait_event(*access, stream, event);
  };

  auto* const persistent = request_access->arena_span(
      owner::Sm87BulkV2P40RequestArenaRole::kPersistent);
  auto* const residual_void = request_access->arena_span(
      owner::Sm87BulkV2P40RequestArenaRole::kResidual);
  auto* const family_void = request_access->arena_span(
      owner::Sm87BulkV2P40RequestArenaRole::kFamily);
  auto* const final_hidden = static_cast<std::uint16_t*>(
      request_access->arena_span(
          owner::Sm87BulkV2P40RequestArenaRole::kFinalHidden));
  if (persistent == nullptr || residual_void == nullptr ||
      family_void == nullptr || final_hidden == nullptr) {
    return poison(Sm87BulkV2P40ExecutorError::kInvalidComposition,
                  "request_arena_span_missing",
                  static_cast<int>(cudaErrorInvalidDevicePointer), 0U,
                  Family::kEmbedding, 0U, 0U);
  }
  auto* const residual = static_cast<std::uint16_t*>(residual_void);
  const auto family_pointer = [&](const Sm87BulkV2P40BufferRole role) noexcept {
    const auto range = sm87_bulk_v2_p40_family_range(role);
    return byte_offset<std::uint16_t>(family_void, range.offset);
  };

  void* const main_stream = access->cuda_stream(Stream::kMain);
  void* const projection_stream =
      access->cuda_stream(Stream::kProjectionAndGdnProducer);
  void* const bf16_stream = access->cuda_stream(Stream::kBf16Ab);
  if (main_stream == nullptr || projection_stream == nullptr ||
      bf16_stream == nullptr) {
    return poison(Sm87BulkV2P40ExecutorError::kInvalidComposition,
                  "owner_stream_binding_missing",
                  static_cast<int>(cudaErrorInvalidResourceHandle), 0U,
                  Family::kEmbedding, 0U, 0U);
  }

  auto* const token_ids = reinterpret_cast<std::uint32_t*>(
      family_pointer(Sm87BulkV2P40BufferRole::kTokenIds));
  cudaError_t cuda_status = cudaMemcpyAsync(
      token_ids, host_prompt_token_ids,
      prompt_tokens * sizeof(std::uint32_t), cudaMemcpyHostToDevice,
      reinterpret_cast<cudaStream_t>(main_stream));
  if (cuda_status != cudaSuccess) {
    return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                  "prompt_token_h2d", static_cast<int>(cuda_status), 0U,
                  Family::kEmbedding, 0U, 0U);
  }
  int launch_status = launch_embedding_gather_prompt_reference_cuda(
      model->embed_tokens().weight, model->embed_tokens().output_size,
      model->embed_tokens().input_size, token_ids, prompt_tokens, residual,
      main_stream);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                  "embedding_prompt_gather", launch_status, 0U,
                  Family::kEmbedding, 0U, 1U);
  }

  constexpr auto persistent_plan = sm87_bulk_v2_p40_persistent_plan();
  auto* const device_control = access->device_control_arena();
  const auto* const cancellation = access->device_cancellation_alias();

  for (std::size_t layer = 0U; layer < kSm87BulkV2P40Layers; ++layer) {
    const auto& weights = model->layer(layer);
    auto* const normalized =
        family_pointer(Sm87BulkV2P40BufferRole::kNormalized);
    launch_status = launch_headwise_centered_rms_norm_reference_cuda(
        residual, weights.input_layernorm.data, prompt_tokens,
        kSm87BulkV2P40Hidden, kRmsEpsilon, normalized, main_stream);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                    "layer_input_centered_rms", launch_status, layer,
                    sm87_bulk_v2_p40_is_full_layer(layer)
                        ? Family::kFullInput
                        : Family::kGdnInput,
                    0U, 0U);
    }
    auto owner_status = record(Stream::kMain, Event::kNormalizedReady);
    if (!owner_status) {
      return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                    owner_status.context, owner_status.cuda_error, layer,
                    sm87_bulk_v2_p40_is_full_layer(layer)
                        ? Family::kFullInput
                        : Family::kGdnInput,
                    0U, 1U);
    }
    owner_status = wait(Stream::kProjectionAndGdnProducer,
                        Event::kNormalizedReady);
    if (!owner_status) {
      return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                    owner_status.context, owner_status.cuda_error, layer,
                    sm87_bulk_v2_p40_is_full_layer(layer)
                        ? Family::kFullInput
                        : Family::kGdnInput,
                    0U, 2U);
    }

    if (!sm87_bulk_v2_p40_is_full_layer(layer)) {
      const std::size_t gdn_ordinal = sm87_bulk_v2_p40_gdn_ordinal(layer);
      auto* const raw_qkvz =
          family_pointer(Sm87BulkV2P40BufferRole::kGdnRawQkvz);
      auto* const gdn_output =
          family_pointer(Sm87BulkV2P40BufferRole::kGdnOutput);
      auto* const o_branch =
          family_pointer(Sm87BulkV2P40BufferRole::kGdnOBranch);
      auto* const gdn_private = reinterpret_cast<std::uint8_t*>(
          family_pointer(Sm87BulkV2P40BufferRole::kGdnPrivate));

      kernels::Sm87BulkV2Fp8WholeP40Arguments fp8_input{};
      fp8_input.transaction_epoch = request_epoch;
      fp8_input.role = Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
      fp8_input.input = normalized;
      fp8_input.authenticated_asset = assets_[layer].input_fp8;
      fp8_input.primary_output = raw_qkvz;
      fp8_input.device_control =
          static_cast<kernels::Sm87BulkV2Fp8WholeP40DeviceControl*>(
              device_control);
      fp8_input.cancellation_signal = cancellation;
      fp8_input.cuda_stream = projection_stream;
      launch_status =
          kernels::launch_sm87_bulk_dataflow_v2_fp8_whole_p40_cuda(
              fp8_input);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "gdn_fp8_whole_input", launch_status, layer,
                      Family::kGdnInput, 0U, 3U);
      }
      owner_status = note(Stream::kProjectionAndGdnProducer,
                          Counter::kFp8GdnInputWholeRoleLaunch, 1U, layer,
                          Family::kGdnInput, 0U, 3U);
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnInput, 0U, 3U);
      }

      owner_status = wait(Stream::kBf16Ab, Event::kNormalizedReady);
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnInput, 0U, 4U);
      }
      std::size_t bf16_launches = 0U;
      launch_status = bf16_ab::enqueue_interleaved_p40_prevalidated(
          *composition_->bf16_ab_access(), gdn_ordinal, &bf16_launches);
      if (launch_status != static_cast<int>(cudaSuccess) ||
          bf16_launches != 1U) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "gdn_bf16_ab", launch_status, layer,
                      Family::kGdnInput, 0U, 4U);
      }
      owner_status = note(Stream::kBf16Ab, Counter::kBf16AbLaunch, 1U,
                          layer, Family::kGdnInput, 0U, 4U);
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnInput, 0U, 4U);
      }
      owner_status = record(Stream::kBf16Ab, Event::kBf16AbReady);
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnInput, 0U, 5U);
      }
      owner_status = wait(Stream::kProjectionAndGdnProducer,
                          Event::kBf16AbReady);
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnInput, 0U, 5U);
      }

      launch_status =
          kernels::enqueue_sm87_bulk_dataflow_v2_gdn_p40_session_epoch_cuda(
              gdn_session, gdn_ordinal);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return poison(Sm87BulkV2P40ExecutorError::kGdnSubmission,
                      "gdn_session_epoch", launch_status, layer,
                      Family::kGdnCore, 0U, 6U);
      }
      owner_status = note(Stream::kProjectionAndGdnProducer,
                          Counter::kGdnProducerChunk,
                          kernels::kSm87BulkV2GdnP40Chunks, layer,
                          Family::kGdnCore, 0U, 6U);
      if (owner_status) {
        owner_status = note(Stream::kGdnRecurrence,
                            Counter::kGdnRecurrenceChunk,
                            kernels::kSm87BulkV2GdnP40Chunks, layer,
                            Family::kGdnCore, 0U, 7U);
      }
      if (owner_status) {
        owner_status = note(Stream::kGdnEpilogue,
                            Counter::kGdnEpilogueChunk,
                            kernels::kSm87BulkV2GdnP40Chunks, layer,
                            Family::kGdnCore, 0U, 8U);
      }
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnCore, 0U, 8U);
      }
      launch_status = kernels::
          bridge_sm87_bulk_dataflow_v2_gdn_p40_session_epoch_to_main_cuda(
              gdn_session);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return poison(Sm87BulkV2P40ExecutorError::kGdnSubmission,
                      "gdn_session_bridge_to_main", launch_status, layer,
                      Family::kGdnCore, 0U, 9U);
      }

      const auto& history_range = persistent_plan.conv_history[gdn_ordinal];
      const auto& state_range = persistent_plan.recurrent_state[gdn_ordinal];
      cuda_status = cudaMemcpyAsync(
          byte_offset<void>(persistent, history_range.offset),
          gdn_private + kernels::kSm87BulkV2GdnP40ConvHistoryOffsets[0U],
          static_cast<std::size_t>(history_range.bytes),
          cudaMemcpyDeviceToDevice,
          reinterpret_cast<cudaStream_t>(main_stream));
      if (cuda_status == cudaSuccess) {
        cuda_status = cudaMemcpyAsync(
            byte_offset<void>(persistent, state_range.offset),
            gdn_private +
                kernels::kSm87BulkV2GdnP40TransactionalRecurrentStateOffset,
            static_cast<std::size_t>(state_range.bytes),
            cudaMemcpyDeviceToDevice,
            reinterpret_cast<cudaStream_t>(main_stream));
      }
      if (cuda_status != cudaSuccess) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "gdn_persistent_state_publish",
                      static_cast<int>(cuda_status), layer,
                      Family::kGdnStatePublish, 0U, 10U);
      }
      owner_status = note(Stream::kMain, Counter::kGdnPersistentCopy, 2U,
                          layer, Family::kGdnStatePublish, 0U, 10U);
      if (owner_status) {
        owner_status = record(Stream::kMain, Event::kCoreReady);
      }
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnStatePublish, 0U, 10U);
      }
      owner_status = wait(Stream::kProjectionAndGdnProducer,
                          Event::kCoreReady);
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnOutputProjection, 0U, 11U);
      }

      kernels::Sm87BulkV2Fp8WholeP40Arguments fp8_output{};
      fp8_output.transaction_epoch = request_epoch;
      fp8_output.role = Sm87TargetAotProjectionRole::kFp8AttentionOutput;
      fp8_output.input = gdn_output;
      fp8_output.authenticated_asset = assets_[layer].output_fp8;
      fp8_output.primary_output = o_branch;
      fp8_output.device_control =
          static_cast<kernels::Sm87BulkV2Fp8WholeP40DeviceControl*>(
              device_control);
      fp8_output.cancellation_signal = cancellation;
      fp8_output.cuda_stream = projection_stream;
      launch_status =
          kernels::launch_sm87_bulk_dataflow_v2_fp8_whole_p40_cuda(
              fp8_output);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "gdn_fp8_whole_output", launch_status, layer,
                      Family::kGdnOutputProjection, 0U, 12U);
      }
      owner_status = note(Stream::kProjectionAndGdnProducer,
                          Counter::kFp8OutputWholeRoleLaunch, 1U, layer,
                          Family::kGdnOutputProjection, 0U, 12U);
      if (owner_status) {
        owner_status = record(Stream::kProjectionAndGdnProducer,
                              Event::kProjectionOutputReady);
      }
      if (owner_status) {
        owner_status = wait(Stream::kMain, Event::kProjectionOutputReady);
      }
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kGdnOutputProjection, 0U, 12U);
      }
      launch_status = launch_residual_add_reference_cuda(
          residual, o_branch, prompt_tokens * kSm87BulkV2P40Hidden,
          residual, main_stream);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "gdn_residual_add", launch_status, layer,
                      Family::kGdnOutputProjection, 0U, 13U);
      }
    } else {
      const std::size_t full_ordinal = sm87_bulk_v2_p40_full_ordinal(layer);
      const auto* const full =
          std::get_if<FullAttentionWeights>(&weights.attention);
      if (full == nullptr) {
        return poison(Sm87BulkV2P40ExecutorError::kInvalidComposition,
                      "full_attention_weight_binding", static_cast<int>(
                          cudaErrorInvalidValue), layer, Family::kFullInput,
                      0U, 3U);
      }
      auto* const raw_q_gate =
          family_pointer(Sm87BulkV2P40BufferRole::kAttentionRawQGate);
      auto* const processed_q =
          family_pointer(Sm87BulkV2P40BufferRole::kAttentionProcessedQ);
      auto* const processed_gate =
          family_pointer(Sm87BulkV2P40BufferRole::kAttentionProcessedGate);
      auto* const gated_output =
          family_pointer(Sm87BulkV2P40BufferRole::kAttentionGatedOutput);
      auto* const o_branch =
          family_pointer(Sm87BulkV2P40BufferRole::kAttentionOBranch);
      auto* const key = byte_offset<std::uint16_t>(
          persistent, persistent_plan.key[full_ordinal].offset);
      auto* const value = byte_offset<std::uint16_t>(
          persistent, persistent_plan.value[full_ordinal].offset);

      kernels::Sm87BulkV2Fp8WholeP40Arguments fp8_input{};
      fp8_input.transaction_epoch = request_epoch;
      fp8_input.role = Sm87TargetAotProjectionRole::kFp8FullQkv;
      fp8_input.input = normalized;
      fp8_input.authenticated_asset = assets_[layer].input_fp8;
      fp8_input.primary_output = raw_q_gate;
      fp8_input.secondary_output = key;
      fp8_input.tertiary_output = value;
      fp8_input.device_control =
          static_cast<kernels::Sm87BulkV2Fp8WholeP40DeviceControl*>(
              device_control);
      fp8_input.cancellation_signal = cancellation;
      fp8_input.cuda_stream = projection_stream;
      launch_status =
          kernels::launch_sm87_bulk_dataflow_v2_fp8_whole_p40_cuda(
              fp8_input);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "full_fp8_whole_qkv", launch_status, layer,
                      Family::kFullInput, 0U, 3U);
      }
      owner_status = note(Stream::kProjectionAndGdnProducer,
                          Counter::kFp8FullInputWholeRoleLaunch, 1U, layer,
                          Family::kFullInput, 0U, 3U);
      if (owner_status) {
        owner_status = record(Stream::kProjectionAndGdnProducer,
                              Event::kProjectionInputReady);
      }
      if (owner_status) {
        owner_status = wait(Stream::kMain, Event::kProjectionInputReady);
      }
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kFullInput, 0U, 3U);
      }

      for (std::size_t panel = 0U; panel < kAttentionPanels; ++panel) {
        const std::size_t first_position = panel * kAttentionPanelTokens;
        launch_status =
            sm87_target_aot_attention_preprocess_execution_detail::
                launch_p8000(
                    raw_q_gate + first_position *
                                     kSm87BulkV2P40AttentionQGateWidth,
                    key + first_position * kSm87TargetAotP40KvWidth,
                    full->q_norm.data, full->k_norm.data, kRmsEpsilon,
                    processed_q + first_position *
                                      kSm87BulkV2P40AttentionWidth,
                    processed_gate + first_position *
                                         kSm87BulkV2P40AttentionWidth,
                    rope_.cosines, rope_.sines, first_position,
                    kAttentionPanelTokens, main_stream);
        if (launch_status != static_cast<int>(cudaSuccess)) {
          return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                        "full_attention_p8000_preprocess", launch_status,
                        layer, Family::kFullPreprocess, panel, 4U);
        }
      }
      owner_status = note(Stream::kMain,
                          Counter::kAttentionPreprocessPanel,
                          kAttentionPanels, layer, Family::kFullPreprocess,
                          0U, 4U);
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kFullPreprocess, 0U, 4U);
      }

      attention::Sm87BulkV2AttentionSubmissionReceipt attention_receipt{};
      launch_status =
          attention::enqueue_sm87_bulk_v2_attention_p40_prevalidated_cuda(
              *composition_->attention_access(), request_epoch, layer,
              &attention_receipt);
      if (launch_status != static_cast<int>(cudaSuccess) ||
          !attention::sm87_bulk_v2_attention_submission_receipt_valid(
              attention_receipt)) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "full_attention_v2_core", launch_status, layer,
                      Family::kFullAttentionCore, 0U, 5U);
      }
      owner_status = note(Stream::kMain, Counter::kAttentionLaunch,
                          attention_receipt.submitted_launches, layer,
                          Family::kFullAttentionCore, 0U, 5U);
      if (owner_status) {
        owner_status = record(Stream::kMain, Event::kCoreReady);
      }
      if (owner_status) {
        owner_status = wait(Stream::kProjectionAndGdnProducer,
                            Event::kCoreReady);
      }
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kFullAttentionCore, 0U, 5U);
      }

      kernels::Sm87BulkV2Fp8WholeP40Arguments fp8_output{};
      fp8_output.transaction_epoch = request_epoch;
      fp8_output.role = Sm87TargetAotProjectionRole::kFp8AttentionOutput;
      fp8_output.input = gated_output;
      fp8_output.authenticated_asset = assets_[layer].output_fp8;
      fp8_output.primary_output = o_branch;
      fp8_output.device_control =
          static_cast<kernels::Sm87BulkV2Fp8WholeP40DeviceControl*>(
              device_control);
      fp8_output.cancellation_signal = cancellation;
      fp8_output.cuda_stream = projection_stream;
      launch_status =
          kernels::launch_sm87_bulk_dataflow_v2_fp8_whole_p40_cuda(
              fp8_output);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "full_fp8_whole_output", launch_status, layer,
                      Family::kFullOutputProjection, 0U, 6U);
      }
      owner_status = note(Stream::kProjectionAndGdnProducer,
                          Counter::kFp8OutputWholeRoleLaunch, 1U, layer,
                          Family::kFullOutputProjection, 0U, 6U);
      if (owner_status) {
        owner_status = record(Stream::kProjectionAndGdnProducer,
                              Event::kProjectionOutputReady);
      }
      if (owner_status) {
        owner_status = wait(Stream::kMain, Event::kProjectionOutputReady);
      }
      if (!owner_status) {
        return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                      owner_status.context, owner_status.cuda_error, layer,
                      Family::kFullOutputProjection, 0U, 6U);
      }
      launch_status = launch_residual_add_reference_cuda(
          residual, o_branch, prompt_tokens * kSm87BulkV2P40Hidden,
          residual, main_stream);
      if (launch_status != static_cast<int>(cudaSuccess)) {
        return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                      "full_residual_add", launch_status, layer,
                      Family::kFullOutputProjection, 0U, 7U);
      }
    }

    auto* const mlp_normalized =
        family_pointer(Sm87BulkV2P40BufferRole::kMlpNormalized);
    auto* const h = family_pointer(Sm87BulkV2P40BufferRole::kNvFp4H);
    launch_status = launch_headwise_centered_rms_norm_reference_cuda(
        residual, weights.post_attention_layernorm.data, prompt_tokens,
        kSm87BulkV2P40Hidden, kRmsEpsilon, mlp_normalized, main_stream);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                    "post_attention_centered_rms", launch_status, layer,
                    Family::kMlp, 0U, 20U);
    }
    owner_status = record(Stream::kMain, Event::kNormalizedReady);
    if (owner_status) {
      owner_status = wait(Stream::kProjectionAndGdnProducer,
                          Event::kNormalizedReady);
    }
    if (!owner_status) {
      return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                    owner_status.context, owner_status.cuda_error, layer,
                    Family::kMlp, 0U, 20U);
    }

    kernels::Sm87BulkV2NvFp4GateUpWholeP40Arguments gate_up{};
    gate_up.transaction_epoch = request_epoch;
    gate_up.normalized_input = mlp_normalized;
    gate_up.h = h;
    gate_up.device_control = static_cast<
        kernels::Sm87BulkV2NvFp4GateUpWholeP40DeviceControl*>(
        device_control);
    gate_up.cancellation_signal = cancellation;
    gate_up.gate_up_asset = assets_[layer].gate_up;
    gate_up.cuda_stream = projection_stream;
    launch_status =
        kernels::launch_sm87_bulk_dataflow_v2_nvfp4_gate_up_whole_p40_cuda(
            gate_up);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                    "nvfp4_gate_up_whole", launch_status, layer,
                    Family::kMlp, 0U, 21U);
    }
    owner_status = note(Stream::kProjectionAndGdnProducer,
                        Counter::kNvFp4GateUpWholeRoleLaunch, 1U, layer,
                        Family::kMlp, 0U, 21U);
    if (!owner_status) {
      return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                    owner_status.context, owner_status.cuda_error, layer,
                    Family::kMlp, 0U, 21U);
    }

    kernels::Sm87BulkV2NvFp4DownWholeP40Arguments down{};
    down.transaction_epoch = request_epoch;
    down.h = h;
    down.residual = residual;
    down.device_control = static_cast<
        kernels::Sm87BulkV2NvFp4DownWholeP40DeviceControl*>(device_control);
    down.cancellation_signal = cancellation;
    down.down_asset = assets_[layer].down;
    down.cuda_stream = projection_stream;
    launch_status =
        kernels::launch_sm87_bulk_dataflow_v2_nvfp4_down_whole_p40_cuda(
            down);
    if (launch_status != static_cast<int>(cudaSuccess)) {
      return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                    "nvfp4_down_whole", launch_status, layer, Family::kMlp,
                    0U, 22U);
    }
    owner_status = note(Stream::kProjectionAndGdnProducer,
                        Counter::kNvFp4DownWholeRoleLaunch, 1U, layer,
                        Family::kMlp, 0U, 22U);
    if (owner_status) {
      owner_status = record(Stream::kProjectionAndGdnProducer,
                            Event::kProjectionOutputReady);
    }
    if (owner_status) {
      owner_status = wait(Stream::kMain, Event::kProjectionOutputReady);
    }
    if (!owner_status) {
      return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                    owner_status.context, owner_status.cuda_error, layer,
                    Family::kMlp, 0U, 22U);
    }

    owner_status = owner_state->close_layer(
        *access, layer,
        sm87_bulk_v2_p40_is_full_layer(layer)
            ? owner::Sm87BulkV2P40LayerKind::kFull
            : owner::Sm87BulkV2P40LayerKind::kGdn);
    if (!owner_status) {
      return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                    owner_status.context, owner_status.cuda_error, layer,
                    Family::kMlp, 0U, 23U);
    }
    if (cancellation_requested(control)) {
      return cancelled("cancelled_after_bulk_v2_layer", layer);
    }
  }

  const auto* const last_residual =
      residual + (kSm87BulkV2P40Tokens - 1U) * kSm87BulkV2P40Hidden;
  launch_status = launch_centered_rms_norm_reference_cuda(
      last_residual, model->final_norm().data, kSm87BulkV2P40Hidden,
      kRmsEpsilon, final_hidden, main_stream);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                  "final_last_row_centered_rms", launch_status,
                  kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U, 0U);
  }
  auto owner_status = note(Stream::kMain, Counter::kFinalNorm, 1U,
                           kSm87BulkV2P40Layers - 1U,
                           Family::kFinalHandoff, 0U, 0U);
  if (!owner_status) {
    return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                  owner_status.context, owner_status.cuda_error,
                  kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U, 0U);
  }

  auto* const logits =
      family_pointer(Sm87BulkV2P40BufferRole::kFinalLogits);
  auto* const greedy = reinterpret_cast<Bf16GreedyArgmaxResult*>(
      family_pointer(Sm87BulkV2P40BufferRole::kFinalGreedyWorkspace));
  launch_status = launch_projection_to_bf16_cuda(
      ProjectionBackend::kSm87WeightOnly, model->lm_head(), final_hidden,
      nullptr, 0U, logits, main_stream);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                  "nvfp4_lm_head_m1_bf16", launch_status,
                  kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U, 1U);
  }
  owner_status = note(Stream::kMain, Counter::kLmHead, 1U,
                      kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U,
                      1U);
  if (!owner_status) {
    return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                  owner_status.context, owner_status.cuda_error,
                  kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U, 1U);
  }

  launch_status = launch_bf16_greedy_argmax_cuda(
      logits, kSm87BulkV2P40Vocabulary, greedy, main_stream);
  if (launch_status != static_cast<int>(cudaSuccess)) {
    return poison(Sm87BulkV2P40ExecutorError::kCudaSubmission,
                  "bf16_greedy_argmax", launch_status,
                  kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U, 2U);
  }
  owner_status = note(Stream::kMain, Counter::kArgmax, 1U,
                      kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U,
                      2U);
  if (!owner_status) {
    return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                  owner_status.context, owner_status.cuda_error,
                  kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U, 2U);
  }

  constexpr std::array<std::pair<Stream, Event>, 4U> joins{{
      {Stream::kProjectionAndGdnProducer, Event::kProjectionOutputReady},
      {Stream::kBf16Ab, Event::kBf16AbReady},
      {Stream::kGdnRecurrence, Event::kGdnRecurrence0},
      {Stream::kGdnEpilogue, Event::kGdnEpilogue0},
  }};
  for (std::size_t join = 0U; join < joins.size(); ++join) {
    owner_status = record(joins[join].first, joins[join].second);
    if (owner_status) {
      owner_status = wait(Stream::kMain, joins[join].second);
    }
    if (!owner_status) {
      return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                    owner_status.context, owner_status.cuda_error,
                    kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U,
                    30U + join);
    }
  }

  owner_status = owner_state->enqueue_owner_bound_handoff_d2h(*access);
  if (!owner_status) {
    return poison(Sm87BulkV2P40ExecutorError::kOwnerTransaction,
                  owner_status.context, owner_status.cuda_error,
                  kSm87BulkV2P40Layers - 1U, Family::kFinalHandoff, 0U, 3U);
  }
  owner_status = owner_state->complete_request(
      *access, *request_state, *request_access);
  result.receipt = owner_state->receipt();
  if (!owner_status) {
    terminal_failure_ = true;
    result.status = failure(Sm87BulkV2P40ExecutorError::kInvalidHandoff,
                            owner_status.context, owner_status.cuda_error,
                            kSm87BulkV2P40Layers - 1U, 0U, 3U);
    return result;
  }
  result.token_id = result.receipt.aggregate.handoff_token_id;
  result.value_bits = result.receipt.handoff_value_bits;
  result.handoff_complete = result.receipt.aggregate.handoff_nonfinite == 0U;
  result.status = {};
  return result;
#endif
}

const char* to_string(const Sm87BulkV2P40ExecutorError error) noexcept {
  switch (error) {
    case Sm87BulkV2P40ExecutorError::kNone:
      return "none";
    case Sm87BulkV2P40ExecutorError::kAdmissionDisabled:
      return "admission_disabled";
    case Sm87BulkV2P40ExecutorError::kInvalidComposition:
      return "invalid_composition";
    case Sm87BulkV2P40ExecutorError::kInvalidEngineRope:
      return "invalid_engine_rope";
    case Sm87BulkV2P40ExecutorError::kIncompleteProjectionAssets:
      return "incomplete_projection_assets";
    case Sm87BulkV2P40ExecutorError::kInvalidInput:
      return "invalid_input";
    case Sm87BulkV2P40ExecutorError::kRequestRearm:
      return "request_rearm";
    case Sm87BulkV2P40ExecutorError::kGdnSessionRearm:
      return "gdn_session_rearm";
    case Sm87BulkV2P40ExecutorError::kTransactionBegin:
      return "transaction_begin";
    case Sm87BulkV2P40ExecutorError::kCudaSubmission:
      return "cuda_submission";
    case Sm87BulkV2P40ExecutorError::kGdnSubmission:
      return "gdn_submission";
    case Sm87BulkV2P40ExecutorError::kOwnerTransaction:
      return "owner_transaction";
    case Sm87BulkV2P40ExecutorError::kInvalidHandoff:
      return "invalid_handoff";
    case Sm87BulkV2P40ExecutorError::kCancelled:
      return "cancelled";
    case Sm87BulkV2P40ExecutorError::kAllocation:
      return "allocation";
  }
  return "unknown";
}

}  // namespace q3x::runtime::sm87_bulk_v2_p40_executor_detail
