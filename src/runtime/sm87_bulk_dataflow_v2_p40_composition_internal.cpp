#include "sm87_bulk_dataflow_v2_p40_composition_internal.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <new>
#include <utility>
#include <variant>

namespace q3x::runtime::sm87_bulk_v2_p40_composition_detail {
namespace {

[[nodiscard]] constexpr Sm87BulkV2P40CompositionStatus ok() noexcept {
  return {};
}

[[nodiscard]] constexpr Sm87BulkV2P40CompositionStatus error(
    const Sm87BulkV2P40CompositionError code, const char* const context,
    const int cuda_error = 0,
    const std::size_t constituent_index = 0U) noexcept {
  return {code, context, cuda_error, constituent_index};
}

// FNV-1a is used only as a compact in-process identity combiner.  None of the
// resulting values is a cryptographic digest or a numerical qualification.
[[nodiscard]] constexpr std::uint64_t mix_identity(
    std::uint64_t identity, const std::uint64_t value) noexcept {
  constexpr std::uint64_t kPrime = 1'099'511'628'211ULL;
  for (std::size_t byte = 0U; byte < sizeof(value); ++byte) {
    identity ^= (value >> (byte * 8U)) & 0xffU;
    identity *= kPrime;
  }
  return identity;
}

template <typename T, std::size_t Size>
[[nodiscard]] std::uint64_t mix_bytes(
    std::uint64_t identity, const std::array<T, Size>& values) noexcept {
  static_assert(sizeof(T) == 1U);
  constexpr std::uint64_t kPrime = 1'099'511'628'211ULL;
  for (const T value : values) {
    identity ^= static_cast<std::uint8_t>(value);
    identity *= kPrime;
  }
  return identity;
}

template <typename... Values>
[[nodiscard]] constexpr std::uint64_t domain_identity(
    const std::uint64_t domain, const Values... values) noexcept {
  std::uint64_t identity = mix_identity(14'695'981'039'346'656'037ULL,
                                        domain);
  ((identity = mix_identity(identity, static_cast<std::uint64_t>(values))),
   ...);
  return identity == 0U ? domain : identity;
}

[[nodiscard]] constexpr std::size_t stream_index(
    const Sm87BulkV2P40Stream stream) noexcept {
  return static_cast<std::size_t>(stream);
}

[[nodiscard]] constexpr std::size_t event_index(
    const Sm87BulkV2P40ReusableEvent event) noexcept {
  return static_cast<std::size_t>(event);
}

template <typename T>
[[nodiscard]] T* byte_offset(void* const base,
                             const std::uint64_t offset) noexcept {
  if (base == nullptr ||
      offset > static_cast<std::uint64_t>(
                   std::numeric_limits<std::ptrdiff_t>::max())) {
    return nullptr;
  }
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) +
                              static_cast<std::ptrdiff_t>(offset));
}

template <typename T>
[[nodiscard]] const T* byte_offset(const void* const base,
                                   const std::uint64_t offset) noexcept {
  if (base == nullptr ||
      offset > static_cast<std::uint64_t>(
                   std::numeric_limits<std::ptrdiff_t>::max())) {
    return nullptr;
  }
  return reinterpret_cast<const T*>(
      static_cast<const std::byte*>(base) +
      static_cast<std::ptrdiff_t>(offset));
}

[[nodiscard]] void* family_pointer(
    const owner_detail::Sm87BulkV2P40RequestStateSealedAccess& access,
    const Sm87BulkV2P40BufferRole role) noexcept {
  const auto range = sm87_bulk_v2_p40_family_range(role);
  if (!range.valid(kSm87BulkV2P40FamilyArenaBytes)) {
    return nullptr;
  }
  return byte_offset<std::byte>(
      access.arena_span(
          owner_detail::Sm87BulkV2P40RequestArenaRole::kFamily),
      range.offset);
}

[[nodiscard]] std::uint64_t retained_catalog_identity(
    const owner_detail::Sm87BulkV2P40WholeProjectionStartupAccess& access)
    noexcept {
  const auto& catalog = access.retained_evidence_catalog();
  if (!catalog.valid()) {
    return 0U;
  }
  std::uint64_t identity =
      domain_identity(0x5133'5842'494e'4341ULL, catalog.abi_major,
                      catalog.abi_minor, access.owner_identity(),
                      static_cast<std::uint64_t>(access.device_ordinal()));
  identity = mix_bytes(identity, catalog.magic);
  identity = mix_bytes(identity, catalog.fp8_retained_sass_record_sha256);
  identity =
      mix_bytes(identity, catalog.gate_up_retained_cubin_record_sha256);
  identity =
      mix_bytes(identity, catalog.gate_up_retained_sass_record_sha256);
  identity = mix_bytes(identity, catalog.down_retained_sass_record_sha256);
  return identity == 0U ? 0x5133'5842'494e'4341ULL : identity;
}

[[nodiscard]] std::uint64_t base_deployment_identity(
    const Sm87BulkV2P40CompositionRoot& root) noexcept {
  const auto* const projection = root.projection_access();
  const auto* const request = root.request_access();
  const auto* const owner = root.owner();
  const auto* const whole = root.whole_projection_access();
  if (projection == nullptr || request == nullptr || owner == nullptr ||
      whole == nullptr) {
    return 0U;
  }
  return domain_identity(
      0x5133'5844'4550'4c59ULL, owner->owner_identity(),
      request->identity().allocation_identity,
      request->identity().stream_event_owner_identity,
      projection->owner_identity(), projection->allocation_identity(),
      projection->catalog_identity(), retained_catalog_identity(*whole),
      static_cast<std::uint64_t>(owner->device_ordinal()));
}

[[nodiscard]] bool model_layer_is_exact_gdn(
    const DecoderLayerWeights& layer,
    const LinearAttentionWeights** const attention_out,
    const Bf16LinearWeight** const a_out,
    const Bf16LinearWeight** const b_out) noexcept {
  if (attention_out == nullptr || a_out == nullptr || b_out == nullptr) {
    return false;
  }
  const auto* const attention =
      std::get_if<LinearAttentionWeights>(&layer.attention);
  if (attention == nullptr ||
      linear_output_size(attention->in_proj_qkv) != 10'240U ||
      linear_input_size(attention->in_proj_qkv) !=
          kSm87BulkV2P40Hidden ||
      linear_output_size(attention->in_proj_a) != 48U ||
      linear_input_size(attention->in_proj_a) != kSm87BulkV2P40Hidden ||
      linear_output_size(attention->in_proj_b) != 48U ||
      linear_input_size(attention->in_proj_b) != kSm87BulkV2P40Hidden ||
      attention->conv1d.data == nullptr ||
      attention->conv1d.shape !=
          std::array<std::size_t, 3U>{10'240U, 1U, 4U} ||
      attention->a_log.data == nullptr ||
      attention->a_log.element_count != 48U ||
      attention->dt_bias.data == nullptr ||
      attention->dt_bias.element_count != 48U ||
      attention->norm.data == nullptr ||
      attention->norm.element_count != 128U) {
    return false;
  }
  const auto* const a = std::get_if<Bf16LinearWeight>(&attention->in_proj_a);
  const auto* const b = std::get_if<Bf16LinearWeight>(&attention->in_proj_b);
  if (a == nullptr || b == nullptr || a->weight == nullptr ||
      b->weight == nullptr || a->output_size != 48U ||
      a->input_size != kSm87BulkV2P40Hidden || b->output_size != 48U ||
      b->input_size != kSm87BulkV2P40Hidden) {
    return false;
  }
  *attention_out = attention;
  *a_out = a;
  *b_out = b;
  return true;
}

}  // namespace

bool Sm87BulkV2P40CompositionAudit::valid() const noexcept {
  return identity.direction_witness_valid() && identity.valid() &&
         projection_artifacts ==
             kSm87TargetAotCompleteProjectionDeviceArtifactCount &&
         bf16_ab_bindings == kSm87BulkV2P40GdnLayers &&
         attention_bindings == kSm87BulkV2P40FullLayers &&
         gdn_bindings == kSm87BulkV2P40GdnLayers &&
         complete_projection_attachment_live &&
         whole_projection_source_resource_root_live &&
         request_state_owner_bound && bf16_ab_access_sealed &&
         attention_access_sealed && gdn_session_sealed &&
         direction_witness_constituent_seal_root_minted &&
         owner_execution_access_issued && !caller_identity_claims_accepted &&
         !caller_raw_capability_set_accepted &&
         !exact_numerical_contract_qualified &&
         !production_dispatch_eligible;
}

Sm87BulkV2P40CompositionCreateResult::operator bool() const noexcept {
  return root != nullptr && root->valid() && audit.valid() &&
         static_cast<bool>(status);
}

Sm87BulkV2P40CompositionRoot::Sm87BulkV2P40CompositionRoot(
    const ModelWeights& model_weights,
    std::unique_ptr<owner_detail::Sm87BulkV2P40Owner> owner,
    std::unique_ptr<owner_detail::Sm87BulkV2P40RequestState> request_state,
    std::unique_ptr<owner_detail::Sm87BulkV2P40WholeProjectionStartupRoot>
        whole_projection_root,
    ProjectionAccess projection_access) noexcept
    : model_weights_(&model_weights),
      owner_(std::move(owner)),
      request_state_(std::move(request_state)),
      whole_projection_root_(std::move(whole_projection_root)),
      projection_access_(std::move(projection_access)) {}

Sm87BulkV2P40CompositionRoot::~Sm87BulkV2P40CompositionRoot() {
  using q3x::kernels::Sm87BulkV2GdnP40SessionLifecycle;
  if (gdn_session_.lifecycle == Sm87BulkV2GdnP40SessionLifecycle::kActive ||
      gdn_session_.lifecycle ==
          Sm87BulkV2GdnP40SessionLifecycle::kAwaitingDrain) {
    (void)q3x::kernels::
        drain_sm87_bulk_dataflow_v2_gdn_p40_session_cuda(&gdn_session_);
  }
}

const owner_detail::Sm87BulkV2P40RequestStateSealedAccess*
Sm87BulkV2P40CompositionRoot::request_access() const noexcept {
  return request_state_ == nullptr ? nullptr : request_state_->sealed_access();
}

const owner_detail::Sm87BulkV2P40ExecutionAccess*
Sm87BulkV2P40CompositionRoot::execution_access() const noexcept {
  return owner_ == nullptr ? nullptr : owner_->execution_access();
}

const Sm87BulkV2P40CompositionRoot::WholeProjectionAccess*
Sm87BulkV2P40CompositionRoot::whole_projection_access() const noexcept {
  return whole_projection_root_ == nullptr ? nullptr
                                           : whole_projection_root_->access();
}

bool Sm87BulkV2P40CompositionRoot::valid() const noexcept {
  const auto* const request = request_access();
  const auto* const whole = whole_projection_access();
  const auto* const execution = execution_access();
  const bool owner_capability_live =
      owner_ != nullptr &&
      owner_->state() != owner_detail::Sm87BulkV2P40OwnerState::kEmpty &&
      owner_->state() !=
          owner_detail::Sm87BulkV2P40OwnerState::kResourcesReady &&
      owner_->state() != owner_detail::Sm87BulkV2P40OwnerState::kDestroyed;
  return model_weights_ != nullptr && owner_ != nullptr &&
         owner_capability_live &&
         request_state_ != nullptr && request != nullptr && request->valid() &&
         request->default_off_development_resource_valid() &&
         request->identity().owner_identity == owner_->owner_identity() &&
         request->identity().device_ordinal == owner_->device_ordinal() &&
         whole_projection_root_ != nullptr && whole != nullptr &&
         whole->default_off_fixed_aot_resource_valid() &&
         whole->bound_to(*owner_) && projection_access_.has_value() &&
         projection_access_->attached() &&
         projection_access_->device_ordinal() == owner_->device_ordinal() &&
         projection_access_->catalog_identity() != 0U &&
         bf16_ab_access_.has_value() && bf16_ab_access_->valid() &&
         bf16_ab_access_->device_ordinal() == owner_->device_ordinal() &&
         attention_access_ != nullptr && attention_access_->valid() &&
         q3x::kernels::sm87_bulk_v2_gdn_p40_session_state_valid(
             gdn_session_) &&
         gdn_session_.lifecycle !=
             q3x::kernels::Sm87BulkV2GdnP40SessionLifecycle::kEmpty &&
         q3x::kernels::sm87_bulk_v2_gdn_p40_submission_receipt_valid(
             gdn_receipt_) &&
         gdn_session_.sealed_plan.layers[0U].submission_receipt ==
             &gdn_receipt_ &&
         constituent_seal_ != nullptr && identity_.valid() &&
         identity_.direction_witness_valid() && execution != nullptr &&
         execution->identity().valid() &&
         execution->identity().seal_nonce == identity_.seal_nonce &&
         audit_.valid();
}

owner_detail::Sm87BulkV2P40OwnerStatus
Sm87BulkV2P40CompositionRoot::
    hot_rearm_gdn_session_after_completed_request() noexcept {
  const auto* const access = execution_access();
  if (owner_ == nullptr || access == nullptr ||
      !identity_.direction_witness_valid()) {
    return {owner_detail::Sm87BulkV2P40OwnerError::kInvalidOwnerState,
            "composition_gdn_hot_rearm_requires_live_owner_access", 0, 0U};
  }
  return owner_->hot_rearm_gdn_session_after_completed_request(
      *access, gdn_session_);
}

Sm87BulkV2P40CompositionStatus Sm87BulkV2P40CompositionRoot::initialize()
    noexcept {
  if (model_weights_ == nullptr || owner_ == nullptr ||
      request_state_ == nullptr || whole_projection_root_ == nullptr ||
      !projection_access_.has_value()) {
    return error(Sm87BulkV2P40CompositionError::kModelBinding,
                 "composition_root_incomplete");
  }
  auto status = seal_bf16_ab();
  if (!status) {
    return status;
  }
  status = seal_attention();
  if (!status) {
    return status;
  }
  status = seal_gdn_session();
  if (!status) {
    return status;
  }
  status = derive_identity();
  if (!status) {
    return status;
  }
  return mint_and_install_direction_witness();
}

Sm87BulkV2P40CompositionStatus
Sm87BulkV2P40CompositionRoot::seal_bf16_ab() noexcept {
  using namespace q3x::kernels::sm87_target_aot_bf16_ab_execution_detail;
  const auto* const request = request_access();
  if (request == nullptr || model_weights_ == nullptr) {
    return error(Sm87BulkV2P40CompositionError::kBf16AbSeal,
                 "bf16_ab_missing_request_or_model");
  }
  auto* const normalized = static_cast<const std::uint16_t*>(
      family_pointer(*request, Sm87BulkV2P40BufferRole::kNormalized));
  auto* const interleaved = static_cast<std::uint16_t*>(
      family_pointer(*request, Sm87BulkV2P40BufferRole::kGdnAb));
  void* const stream = request->cuda_stream(Sm87BulkV2P40Stream::kBf16Ab);
  if (normalized == nullptr || interleaved == nullptr || stream == nullptr) {
    return error(Sm87BulkV2P40CompositionError::kBf16AbSeal,
                 "bf16_ab_request_binding");
  }

  std::array<InterleavedP40Binding, kInterleavedP40LayerBindings> bindings{};
  std::size_t binding_count = 0U;
  for (std::size_t model_layer = 0U;
       model_layer < kSm87BulkV2P40Layers; ++model_layer) {
    if (sm87_bulk_v2_p40_is_full_layer(model_layer)) {
      continue;
    }
    const LinearAttentionWeights* attention = nullptr;
    const Bf16LinearWeight* a = nullptr;
    const Bf16LinearWeight* b = nullptr;
    if (binding_count >= bindings.size() ||
        !model_layer_is_exact_gdn(model_weights_->layer(model_layer),
                                  &attention, &a, &b) ||
        attention == nullptr) {
      return error(Sm87BulkV2P40CompositionError::kModelBinding,
                   "bf16_ab_model_layer_binding", 0, model_layer);
    }
    bindings[binding_count] = InterleavedP40Binding{
        static_cast<std::uint32_t>(model_layer), a->weight, b->weight,
        normalized, interleaved};
    ++binding_count;
  }
  if (binding_count != bindings.size()) {
    return error(Sm87BulkV2P40CompositionError::kBf16AbSeal,
                 "bf16_ab_incomplete_layer_set", 0, binding_count);
  }
  auto sealed = seal_interleaved_p40(bindings.data(), bindings.size(), stream);
  if (!sealed) {
    return error(Sm87BulkV2P40CompositionError::kBf16AbSeal,
                 "bf16_ab_cuda_seal", sealed.cuda_error);
  }
  bf16_ab_access_ = std::move(*sealed.access);
  audit_.bf16_ab_bindings = bindings.size();
  audit_.bf16_ab_access_sealed = bf16_ab_access_->valid();
  return audit_.bf16_ab_access_sealed
             ? ok()
             : error(Sm87BulkV2P40CompositionError::kBf16AbSeal,
                     "bf16_ab_invalid_sealed_access");
}

Sm87BulkV2P40CompositionStatus
Sm87BulkV2P40CompositionRoot::seal_attention() noexcept {
  using namespace q3x::kernels::sm87_bulk_v2_attention_execution_detail;
  const auto* const request = request_access();
  if (request == nullptr || owner_ == nullptr) {
    return error(Sm87BulkV2P40CompositionError::kAttentionSeal,
                 "attention_missing_request_or_owner");
  }
  auto* const query = static_cast<const std::uint16_t*>(family_pointer(
      *request, Sm87BulkV2P40BufferRole::kAttentionProcessedQ));
  auto* const gate = static_cast<const std::uint16_t*>(family_pointer(
      *request, Sm87BulkV2P40BufferRole::kAttentionProcessedGate));
  auto* const output = static_cast<std::uint16_t*>(family_pointer(
      *request, Sm87BulkV2P40BufferRole::kAttentionGatedOutput));
  void* const persistent = request->arena_span(
      owner_detail::Sm87BulkV2P40RequestArenaRole::kPersistent);
  void* const stream = request->cuda_stream(Sm87BulkV2P40Stream::kMain);
  constexpr auto persistent_plan = sm87_bulk_v2_p40_persistent_plan();
  if (query == nullptr || gate == nullptr || output == nullptr ||
      persistent == nullptr || stream == nullptr ||
      !sm87_bulk_v2_p40_persistent_plan_valid(persistent_plan)) {
    return error(Sm87BulkV2P40CompositionError::kAttentionSeal,
                 "attention_request_binding");
  }

  std::array<Sm87BulkV2AttentionBinding,
             kSm87BulkV2AttentionFullLayerBindings>
      bindings{};
  for (std::size_t ordinal = 0U; ordinal < bindings.size(); ++ordinal) {
    const std::size_t model_layer = ordinal * 4U + 3U;
    if (model_weights_ == nullptr ||
        std::get_if<FullAttentionWeights>(
            &model_weights_->layer(model_layer).attention) == nullptr) {
      return error(Sm87BulkV2P40CompositionError::kModelBinding,
                   "attention_model_layer_binding", 0, model_layer);
    }
    bindings[ordinal].model_layer =
        static_cast<std::uint32_t>(model_layer);
    bindings[ordinal].arguments = q3x::kernels::Sm87BulkV2AttentionArguments{
        query,
        byte_offset<const std::uint16_t>(
            persistent, persistent_plan.key[ordinal].offset),
        byte_offset<const std::uint16_t>(
            persistent, persistent_plan.value[ordinal].offset),
        gate,
        output,
        kSm87BulkV2P40Tokens,
        owner_->device_ordinal(),
        stream};
  }

  const std::uint64_t deployment = base_deployment_identity(*this);
  const auto* const projection = projection_access();
  if (deployment == 0U || projection == nullptr ||
      projection->catalog_identity() == 0U) {
    return error(Sm87BulkV2P40CompositionError::kIdentityDerivation,
                 "attention_base_identity");
  }
  Sm87BulkV2AttentionSealRequest seal_request{};
  seal_request.layer_bindings = bindings.data();
  seal_request.layer_binding_count = bindings.size();
  seal_request.device_ordinal = owner_->device_ordinal();
  seal_request.cuda_stream = stream;
  seal_request.deployment_identity = deployment;
  seal_request.binding_catalog_identity = domain_identity(
      0x5133'5841'5443'4154ULL, projection->catalog_identity(),
      request->identity().allocation_identity, bindings.size());
  seal_request.binding_lifetime_owner_identity =
      request->identity().allocation_identity;
  seal_request.cuda_stream_owner_identity =
      request->identity().stream_event_owner_identity;
  auto sealed = seal_sm87_bulk_v2_attention_p40_cuda(seal_request);
  if (!sealed) {
    return error(Sm87BulkV2P40CompositionError::kAttentionSeal,
                 "attention_cuda_seal", sealed.cuda_error,
                 static_cast<std::size_t>(sealed.failure));
  }
  attention_access_ = std::move(sealed.access);
  audit_.attention_bindings = bindings.size();
  audit_.attention_access_sealed =
      attention_access_ != nullptr && attention_access_->valid();
  return audit_.attention_access_sealed
             ? ok()
             : error(Sm87BulkV2P40CompositionError::kAttentionSeal,
                     "attention_invalid_sealed_access");
}

Sm87BulkV2P40CompositionStatus
Sm87BulkV2P40CompositionRoot::seal_gdn_session() noexcept {
  using namespace q3x::kernels;
  const auto* const request = request_access();
  if (request == nullptr || owner_ == nullptr || model_weights_ == nullptr) {
    return error(Sm87BulkV2P40CompositionError::kGdnSessionSeal,
                 "gdn_missing_composition_resources");
  }
  void* const persistent = request->arena_span(
      owner_detail::Sm87BulkV2P40RequestArenaRole::kPersistent);
  auto* const raw_qkvz = static_cast<const std::uint16_t*>(
      family_pointer(*request, Sm87BulkV2P40BufferRole::kGdnRawQkvz));
  auto* const interleaved_ab = static_cast<const std::uint16_t*>(
      family_pointer(*request, Sm87BulkV2P40BufferRole::kGdnAb));
  auto* const output = static_cast<std::uint16_t*>(
      family_pointer(*request, Sm87BulkV2P40BufferRole::kGdnOutput));
  void* const private_base =
      family_pointer(*request, Sm87BulkV2P40BufferRole::kGdnPrivate);
  constexpr auto persistent_plan = sm87_bulk_v2_p40_persistent_plan();
  if (persistent == nullptr || raw_qkvz == nullptr ||
      interleaved_ab == nullptr || output == nullptr ||
      private_base == nullptr ||
      !sm87_bulk_v2_p40_persistent_plan_valid(persistent_plan)) {
    return error(Sm87BulkV2P40CompositionError::kGdnSessionSeal,
                 "gdn_request_binding");
  }

  std::array<void*, kSm87BulkV2GdnP40StreamCount> streams{{
      owner_->streams_[stream_index(
          Sm87BulkV2P40Stream::kProjectionAndGdnProducer)],
      owner_->streams_[stream_index(Sm87BulkV2P40Stream::kGdnRecurrence)],
      owner_->streams_[stream_index(Sm87BulkV2P40Stream::kGdnEpilogue)],
  }};
  std::array<void*, kSm87BulkV2GdnP40SlotCount> prepared_events{{
      owner_->events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnPrepared0)],
      owner_->events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnPrepared1)],
  }};
  std::array<void*, kSm87BulkV2GdnP40SlotCount> recurrence_events{{
      owner_->events_[
          event_index(Sm87BulkV2P40ReusableEvent::kGdnRecurrence0)],
      owner_->events_[
          event_index(Sm87BulkV2P40ReusableEvent::kGdnRecurrence1)],
  }};
  std::array<void*, kSm87BulkV2GdnP40SlotCount> epilogue_events{{
      owner_->events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnEpilogue0)],
      owner_->events_[event_index(Sm87BulkV2P40ReusableEvent::kGdnEpilogue1)],
  }};

  Sm87BulkV2GdnP40SessionPlan plan{};
  plan.main_stream =
      owner_->streams_[stream_index(Sm87BulkV2P40Stream::kMain)];
  plan.ingress_ready_event = owner_->events_[event_index(
      Sm87BulkV2P40ReusableEvent::kProjectionInputReady)];
  gdn_receipt_ = sm87_bulk_v2_gdn_p40_submission_receipt();
  std::size_t binding_count = 0U;
  for (std::size_t model_layer = 0U;
       model_layer < kSm87BulkV2P40Layers; ++model_layer) {
    if (sm87_bulk_v2_p40_is_full_layer(model_layer)) {
      continue;
    }
    const std::size_t ordinal = sm87_bulk_v2_p40_gdn_ordinal(model_layer);
    const LinearAttentionWeights* attention = nullptr;
    const Bf16LinearWeight* a = nullptr;
    const Bf16LinearWeight* b = nullptr;
    if (ordinal >= plan.layers.size() || ordinal != binding_count ||
        !model_layer_is_exact_gdn(model_weights_->layer(model_layer),
                                  &attention, &a, &b) ||
        attention == nullptr || a == nullptr || b == nullptr) {
      return error(Sm87BulkV2P40CompositionError::kModelBinding,
                   "gdn_model_layer_binding", 0, model_layer);
    }
    auto& arguments = plan.layers[ordinal];
    arguments.raw_qkvz = raw_qkvz;
    arguments.interleaved_ab = interleaved_ab;
    arguments.conv_weight = attention->conv1d.data;
    arguments.initial_conv_history = byte_offset<const std::uint16_t>(
        persistent, persistent_plan.conv_history[ordinal].offset);
    arguments.a_log = attention->a_log.data;
    arguments.dt_bias = attention->dt_bias.data;
    arguments.norm_weight = attention->norm.data;
    arguments.initial_recurrent_state = byte_offset<const std::uint16_t>(
        persistent, persistent_plan.recurrent_state[ordinal].offset);
    arguments.l2_epsilon_fp32_bits = kSm87TargetAotGdnEpsilonFp32Bits;
    arguments.norm_epsilon_fp32_bits = kSm87TargetAotGdnEpsilonFp32Bits;
    for (std::size_t slot = 0U; slot < kSm87BulkV2GdnP40SlotCount;
         ++slot) {
      arguments.normalized_q[slot] = byte_offset<float>(
          private_base, kSm87BulkV2GdnP40NormalizedQOffsets[slot]);
      arguments.normalized_k[slot] = byte_offset<float>(
          private_base, kSm87BulkV2GdnP40NormalizedKOffsets[slot]);
      arguments.prepared_v[slot] = byte_offset<std::uint16_t>(
          private_base, kSm87BulkV2GdnP40PreparedVOffsets[slot]);
      arguments.alpha[slot] = byte_offset<float>(
          private_base, kSm87BulkV2GdnP40AlphaOffsets[slot]);
      arguments.beta[slot] = byte_offset<float>(
          private_base, kSm87BulkV2GdnP40BetaOffsets[slot]);
      arguments.raw_output[slot] = byte_offset<std::uint16_t>(
          private_base, kSm87BulkV2GdnP40RawOutputOffsets[slot]);
      arguments.conv_history[slot] = byte_offset<std::uint16_t>(
          private_base, kSm87BulkV2GdnP40ConvHistoryOffsets[slot]);
      arguments.cancellation_snapshot[slot] = byte_offset<std::uint32_t>(
          private_base, kSm87BulkV2GdnP40CancellationSnapshotOffsets[slot]);
    }
    arguments.output = output;
    arguments.transactional_recurrent_state = byte_offset<std::uint16_t>(
        private_base, kSm87BulkV2GdnP40TransactionalRecurrentStateOffset);
    arguments.streams = streams;
    arguments.prepared_ready_events = prepared_events;
    arguments.recurrence_done_events = recurrence_events;
    arguments.epilogue_done_events = epilogue_events;
    arguments.cancellation_host_word = owner_->cancellation_host_word_;
    arguments.cancellation_device_alias = owner_->cancellation_device_alias_;
    arguments.submission_receipt = &gdn_receipt_;
    ++binding_count;
  }
  if (binding_count != plan.layers.size() ||
      !sm87_bulk_v2_gdn_p40_session_plan_valid(plan)) {
    return error(Sm87BulkV2P40CompositionError::kGdnSessionSeal,
                 "gdn_incomplete_session_plan", 0, binding_count);
  }

  const int cuda_status =
      initialize_sm87_bulk_dataflow_v2_gdn_p40_session_cuda(&gdn_session_,
                                                             plan);
  if (cuda_status != 0 ||
      gdn_session_.lifecycle != Sm87BulkV2GdnP40SessionLifecycle::kReady ||
      !sm87_bulk_v2_gdn_p40_session_state_valid(gdn_session_)) {
    return error(Sm87BulkV2P40CompositionError::kGdnSessionSeal,
                 "gdn_cuda_session_seal", cuda_status);
  }
  audit_.gdn_bindings = binding_count;
  audit_.gdn_session_sealed = true;
  return ok();
}

Sm87BulkV2P40CompositionStatus
Sm87BulkV2P40CompositionRoot::derive_identity() noexcept {
  const auto* const request = request_access();
  const auto* const whole = whole_projection_access();
  const auto* const projection = projection_access();
  if (model_weights_ == nullptr || owner_ == nullptr || request == nullptr ||
      whole == nullptr || projection == nullptr ||
      attention_access_ == nullptr || !attention_access_->valid() ||
      !q3x::kernels::sm87_bulk_v2_gdn_p40_session_state_valid(
          gdn_session_)) {
    return error(Sm87BulkV2P40CompositionError::kIdentityDerivation,
                 "identity_missing_real_object");
  }
  const std::uint64_t catalog = projection->catalog_identity();
  const std::uint64_t retained = retained_catalog_identity(*whole);
  const std::uint64_t deployment = base_deployment_identity(*this);
  const auto& request_identity = request->identity();
  const auto& attention_receipt = attention_access_->receipt();
  const auto& resources = gdn_session_.sealed_resources;
  if (catalog == 0U || retained == 0U || deployment == 0U ||
      !q3x::kernels::sm87_bulk_v2_attention_execution_detail::
          sm87_bulk_v2_attention_seal_receipt_valid(attention_receipt)) {
    return error(Sm87BulkV2P40CompositionError::kIdentityDerivation,
                 "identity_zero_source");
  }

  owner_detail::Sm87BulkV2P40OwnerIdentity identity{};
  identity.plan_magic = kSm87BulkV2P40PlanMagic;
  identity.abi_major = kSm87BulkV2P40PlanAbiMajor;
  identity.abi_minor = kSm87BulkV2P40PlanAbiMinor;
  identity.owner_identity = owner_->owner_identity();
  identity.seal_nonce = domain_identity(
      0x5133'5853'4541'4c32ULL, owner_->owner_identity(),
      request_identity.seal_nonce, projection->allocation_identity(),
      attention_receipt.seal_nonce);
  identity.deployment_identity = deployment;
  identity.model_identity = domain_identity(
      0x5133'584d'4f44'454cULL, catalog, projection->owner_identity(),
      projection->allocation_identity(),
      reinterpret_cast<std::uintptr_t>(model_weights_),
      model_weights_->stats().tensor_views,
      model_weights_->stats().linear_attention_layers,
      model_weights_->stats().full_attention_layers);
  identity.request_allocation_identity = request_identity.allocation_identity;
  identity.stream_event_owner_identity =
      request_identity.stream_event_owner_identity;
  identity.asset_catalog_identity = catalog;
  // These legacy evidence slots are domain-separated provenance identities.
  // They deliberately do not claim that any constituent is an exact oracle.
  identity.binary_evidence_identity = retained;
  identity.fp8_oracle_evidence_identity = domain_identity(
      0x5133'5846'5038'5052ULL, catalog, retained,
      projection->owner_identity());
  identity.attention_oracle_evidence_identity = domain_identity(
      0x5133'5841'5454'5052ULL, attention_receipt.seal_nonce,
      attention_receipt.binding_catalog_identity,
      attention_receipt.binding_lifetime_owner_identity,
      attention_receipt.cuda_stream_owner_identity);
  identity.gdn_oracle_evidence_identity = domain_identity(
      0x5133'5847'444e'5052ULL,
      static_cast<std::uint64_t>(gdn_session_.sealed_device),
      static_cast<std::uint64_t>(resources.binary_version),
      static_cast<std::uint64_t>(resources.producer.registers_per_thread),
      static_cast<std::uint64_t>(resources.recurrence.registers_per_thread),
      static_cast<std::uint64_t>(resources.epilogue.registers_per_thread),
      request_identity.allocation_identity);
  identity.nvfp4_oracle_evidence_identity = domain_identity(
      0x5133'584e'5634'5052ULL, catalog, retained,
      projection->allocation_identity());
  identity.device_ordinal = owner_->device_ordinal();
  identity.execution_class =
      owner_detail::Sm87BulkV2P40ExecutionClass::
          kDefaultOffDirectionWitness;
  identity.authenticated_real_constituents = true;
  identity.exact_numerical_contract_qualified = false;
  identity.development_execution_eligible = true;
  identity.production_dispatch_eligible = false;
  if (!identity.valid() || !identity.direction_witness_valid()) {
    return error(Sm87BulkV2P40CompositionError::kIdentityDerivation,
                 "derived_direction_witness_identity_invalid");
  }
  identity_ = identity;
  return ok();
}

Sm87BulkV2P40CompositionStatus
Sm87BulkV2P40CompositionRoot::mint_and_install_direction_witness() noexcept {
  if (owner_ == nullptr || !identity_.direction_witness_valid() ||
      owner_->state_ != owner_detail::Sm87BulkV2P40OwnerState::kResourcesReady ||
      owner_->execution_access() != nullptr) {
    return error(
        Sm87BulkV2P40CompositionError::kDirectionWitnessOwnerSeal,
        "direction_witness_owner_not_ready");
  }
  std::unique_ptr<owner_detail::Sm87BulkV2P40ConstituentSealAccess> seal(
      new (std::nothrow)
          owner_detail::Sm87BulkV2P40ConstituentSealAccess());
  if (seal == nullptr) {
    return error(Sm87BulkV2P40CompositionError::kConstituentSealAllocation,
                 "allocate_direction_witness_constituent_seal");
  }
  seal->identity_ = identity_;
  seal->bound_owner_identity_ = owner_->owner_identity_;
  seal->streams_ = owner_->streams_;
  seal->events_ = owner_->events_;
  seal->device_control_arena_ = owner_->device_control_arena_;
  seal->cancellation_host_word_ = owner_->cancellation_host_word_;
  seal->cancellation_device_alias_ = owner_->cancellation_device_alias_;
  seal->real_fp8_binding_seal = true;
  seal->real_attention_binding_seal = true;
  seal->real_bf16_ab_binding_seal = true;
  seal->real_gdn_session_seal = true;
  seal->real_nvfp4_binding_seal = true;
  seal->real_request_arena_seal = true;
  seal->real_pinned_handoff_seal = true;
  seal->all_static_resource_checks_complete = true;
  seal->authenticated_real_constituents = true;
  seal->exact_numerical_contract_qualified = false;
  seal->default_off_direction_witness_eligible = true;
  seal->default_off_candidate_eligible = false;
  seal->production_dispatch_eligible = false;
  seal->synthetic_host_contract_only = false;

  const auto owner_status =
      owner_->seal_for_default_off_direction_witness(*seal);
  if (!owner_status || owner_->execution_access() == nullptr ||
      !owner_->execution_access()->identity().direction_witness_valid() ||
      owner_->execution_access()->identity().seal_nonce !=
          identity_.seal_nonce) {
    return error(
        Sm87BulkV2P40CompositionError::kDirectionWitnessOwnerSeal,
        owner_status.context, owner_status.cuda_error,
        owner_status.resource_index);
  }
  constituent_seal_ = std::move(seal);
  audit_.identity = identity_;
  audit_.projection_artifacts = projection_access_->artifact_count();
  audit_.complete_projection_attachment_live = projection_access_->attached();
  audit_.whole_projection_source_resource_root_live =
      whole_projection_access() != nullptr &&
      whole_projection_access()->default_off_fixed_aot_resource_valid();
  audit_.request_state_owner_bound =
      request_access() != nullptr && request_access()->valid() &&
      request_access()->identity().owner_identity == owner_->owner_identity();
  audit_.direction_witness_constituent_seal_root_minted = true;
  audit_.owner_execution_access_issued = true;
  audit_.caller_identity_claims_accepted = false;
  audit_.caller_raw_capability_set_accepted = false;
  audit_.exact_numerical_contract_qualified = false;
  audit_.production_dispatch_eligible = false;
  if (!audit_.valid()) {
    return error(Sm87BulkV2P40CompositionError::kDirectionWitnessOwnerSeal,
                 "composition_audit_invalid_after_owner_seal");
  }
  return ok();
}

Sm87BulkV2P40CompositionCreateResult
create_sm87_bulk_dataflow_v2_p40_composition_root(
    const ModelWeights& model_weights) noexcept {
  Sm87BulkV2P40CompositionCreateResult result;
  auto projection = Sm87BulkV2P40CompositionRoot::ProjectionAccess::bind(
      model_weights);
  if (!projection.has_value() || !projection->attached()) {
    result.status = error(Sm87BulkV2P40CompositionError::kProjectionAttachment,
                          "complete_projection_model_attachment");
    return result;
  }
  if (projection->artifact_count() !=
          kSm87TargetAotCompleteProjectionDeviceArtifactCount ||
      projection->catalog_identity() == 0U) {
    result.status = error(Sm87BulkV2P40CompositionError::kProjectionCatalog,
                          "complete_projection_catalog_identity");
    return result;
  }

  auto owner_result =
      owner_detail::create_sm87_bulk_dataflow_v2_p40_owner_resources();
  if (!owner_result) {
    result.status = error(Sm87BulkV2P40CompositionError::kOwnerResources,
                          owner_result.status.context,
                          owner_result.status.cuda_error,
                          owner_result.status.resource_index);
    return result;
  }
  if (owner_result.owner->device_ordinal() != projection->device_ordinal()) {
    result.status = error(
        Sm87BulkV2P40CompositionError::kOwnerDeviceMismatch,
        "owner_projection_device_mismatch");
    return result;
  }

  auto request_result =
      owner_detail::create_sm87_bulk_dataflow_v2_p40_request_state(
          *owner_result.owner);
  if (!request_result) {
    result.status = error(Sm87BulkV2P40CompositionError::kRequestState,
                          request_result.status.context,
                          request_result.status.cuda_error,
                          request_result.status.resource_index);
    return result;
  }

  auto whole_result = owner_detail::
      create_sm87_bulk_dataflow_v2_p40_whole_projection_startup_root(
          *owner_result.owner);
  if (!whole_result) {
    result.status = error(
        Sm87BulkV2P40CompositionError::kWholeProjectionStartup,
        whole_result.status.context, whole_result.status.cuda_error,
        whole_result.status.resource_index);
    return result;
  }

  std::unique_ptr<Sm87BulkV2P40CompositionRoot> root(
      new (std::nothrow) Sm87BulkV2P40CompositionRoot(
          model_weights, std::move(owner_result.owner),
          std::move(request_result.state), std::move(whole_result.root),
          std::move(*projection)));
  if (root == nullptr) {
    result.status = error(Sm87BulkV2P40CompositionError::kRootAllocation,
                          "allocate_composition_root");
    return result;
  }
  result.status = root->initialize();
  if (!result.status) {
    result.audit = root->audit();
    return result;
  }
  if (!root->valid()) {
    result.status = error(Sm87BulkV2P40CompositionError::kIdentityDerivation,
                          "composition_root_invalid_after_initialize");
    result.audit = root->audit();
    return result;
  }
  result.audit = root->audit();
  result.root = std::move(root);
  return result;
}

const char* to_string(const Sm87BulkV2P40CompositionError error_code)
    noexcept {
  switch (error_code) {
    case Sm87BulkV2P40CompositionError::kNone:
      return "none";
    case Sm87BulkV2P40CompositionError::kProjectionAttachment:
      return "projection_attachment";
    case Sm87BulkV2P40CompositionError::kProjectionCatalog:
      return "projection_catalog";
    case Sm87BulkV2P40CompositionError::kOwnerResources:
      return "owner_resources";
    case Sm87BulkV2P40CompositionError::kOwnerDeviceMismatch:
      return "owner_device_mismatch";
    case Sm87BulkV2P40CompositionError::kRequestState:
      return "request_state";
    case Sm87BulkV2P40CompositionError::kWholeProjectionStartup:
      return "whole_projection_startup";
    case Sm87BulkV2P40CompositionError::kModelBinding:
      return "model_binding";
    case Sm87BulkV2P40CompositionError::kBf16AbSeal:
      return "bf16_ab_seal";
    case Sm87BulkV2P40CompositionError::kAttentionSeal:
      return "attention_seal";
    case Sm87BulkV2P40CompositionError::kGdnSessionSeal:
      return "gdn_session_seal";
    case Sm87BulkV2P40CompositionError::kIdentityDerivation:
      return "identity_derivation";
    case Sm87BulkV2P40CompositionError::kConstituentSealAllocation:
      return "constituent_seal_allocation";
    case Sm87BulkV2P40CompositionError::kDirectionWitnessOwnerSeal:
      return "direction_witness_owner_seal";
    case Sm87BulkV2P40CompositionError::kRootAllocation:
      return "root_allocation";
  }
  return "unknown";
}

}  // namespace q3x::runtime::sm87_bulk_v2_p40_composition_detail
