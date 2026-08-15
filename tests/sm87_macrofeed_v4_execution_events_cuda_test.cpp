#include "../src/runtime/sm87_macrofeed_v4_execution_events_internal.h"
#include "q3x/kernels/sm87_target_aot_projection_cuda.h"
#include "support/sm87_macrofeed_v4_live_fp8_asset_fixture.h"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>

namespace events =
    q3x::runtime::sm87_macrofeed_v4_execution_events_detail;
namespace runtime = q3x::runtime;
namespace kernels = q3x::kernels;
namespace support = q3x::tests::support;
using Fixture = events::Sm87MacroFeedV4ExecutionEventsCudaTestFixture;

static_assert(!std::is_default_constructible_v<
              events::Sm87MacroFeedV4ExecutionEventsAccess>);
static_assert(!std::is_copy_constructible_v<
              events::Sm87MacroFeedV4ExecutionEventsAccess>);
static_assert(!std::is_move_constructible_v<
              events::Sm87MacroFeedV4ExecutionEventsAccess>);
static_assert(!std::is_default_constructible_v<
              events::Sm87MacroFeedV4ExecutionPanelAccess>);
static_assert(!std::is_copy_constructible_v<
              events::Sm87MacroFeedV4ExecutionPanelAccess>);
static_assert(!std::is_convertible_v<
              events::Sm87MacroFeedV4EventEnqueueReceipt,
              events::Sm87MacroFeedV4PhysicalCompletionReceipt>);
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
static_assert(!std::is_aggregate_v<
              events::
                  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt>);
static_assert(std::is_default_constructible_v<
              events::
                  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt>);
static_assert(std::is_copy_constructible_v<
              events::
                  Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt>);
#endif

template <typename T, typename = void>
struct HasPublicAccess : std::false_type {};

template <typename T>
struct HasPublicAccess<
    T, std::void_t<decltype(std::declval<const T&>().access())>>
    : std::true_type {};

template <typename T, typename = void>
struct HasPublicBeginPanel : std::false_type {};

template <typename T>
struct HasPublicBeginPanel<
    T, std::void_t<decltype(std::declval<T&>().begin_panel(
           std::declval<const events::Sm87MacroFeedV4ExecutionEventsAccess&>(),
           std::size_t{}))>> : std::true_type {};

template <typename T, typename = void>
struct HasPublicRecordEvent : std::false_type {};

template <typename T>
struct HasPublicRecordEvent<
    T, std::void_t<decltype(std::declval<T&>().record_event(
           std::declval<const events::Sm87MacroFeedV4ExecutionEventsAccess&>(),
           std::declval<const events::Sm87MacroFeedV4ExecutionPanelAccess&>(),
           events::Sm87MacroFeedV4ExecutionStream::kMain,
           events::Sm87MacroFeedV4ExecutionEvent::kPanelDone))>>
    : std::true_type {};

static_assert(!HasPublicAccess<
              events::Sm87MacroFeedV4ExecutionEventsOwner>::value);
static_assert(!HasPublicBeginPanel<
              events::Sm87MacroFeedV4ExecutionEventsOwner>::value);
static_assert(!HasPublicRecordEvent<
              events::Sm87MacroFeedV4ExecutionEventsOwner>::value);

namespace {

struct Test final {
  int failures = 0;

  void expect(const bool condition, const std::string_view message) {
    if (!condition) {
      ++failures;
      std::cerr << "FAIL: " << message << '\n';
    }
  }
};

struct BoundOwner final {
  events::Sm87MacroFeedV4ExecutionEventsCreateResult execution{};
  runtime::Sm87MacroFeedV4RequestStateCreateResult request{};
  runtime::Sm87MacroFeedV4RequestStateSealedAccess request_access;
  void* recurrent_allocation = nullptr;

  BoundOwner(events::Sm87MacroFeedV4ExecutionEventsCreateResult execution_in,
             runtime::Sm87MacroFeedV4RequestStateCreateResult request_in,
             runtime::Sm87MacroFeedV4RequestStateSealedAccess access_in,
             void* const recurrent_allocation_in)
      : execution(std::move(execution_in)),
        request(std::move(request_in)),
        request_access(access_in),
        recurrent_allocation(recurrent_allocation_in) {}

  ~BoundOwner() {
    execution.owner.reset();
    request.state.reset();
    if (recurrent_allocation != nullptr) {
      (void)cudaFree(recurrent_allocation);
      recurrent_allocation = nullptr;
    }
  }
};

[[nodiscard]] std::unique_ptr<BoundOwner> make_bound_owner(
    Test& test, const std::uint64_t allocation_identity,
    const std::uint64_t kv_allocation_identity = 0U) {
  auto execution =
      events::create_sm87_macrofeed_v4_execution_events_owner();
  test.expect(static_cast<bool>(execution),
              "physical execution-event owner creates on exact SM87/16SM");
  if (!execution) {
    return nullptr;
  }
  const std::uint64_t owner_identity =
      Fixture::owner_identity(*execution.owner);
  test.expect(owner_identity != 0U,
              "test fixture observes a nonzero private owner identity");
  if (owner_identity == 0U) {
    return nullptr;
  }

  void* recurrent_allocation = nullptr;
  const cudaError_t allocation_status = cudaMalloc(
      &recurrent_allocation, runtime::kSm87MacroFeedV4RecurrentStorageBytes);
  test.expect(allocation_status == cudaSuccess &&
                  recurrent_allocation != nullptr,
              "test owner allocates exact dual-epoch recurrent storage");
  if (allocation_status != cudaSuccess || recurrent_allocation == nullptr) {
    return nullptr;
  }
  const auto cold_initialized = Fixture::initialize_cold_recurrent_storage(
      *execution.owner, recurrent_allocation,
      runtime::kSm87MacroFeedV4RecurrentStorageBytes, allocation_identity);
  test.expect(static_cast<bool>(cold_initialized),
              "test owner cold-initializes exact recurrent allocation once");
  if (!cold_initialized) {
    execution.owner.reset();
    (void)cudaFree(recurrent_allocation);
    return nullptr;
  }

  const auto admission = kv_allocation_identity == 0U
                             ? runtime::make_sm87_macrofeed_v4_request_state_admission(
                                   owner_identity, allocation_identity,
                                   allocation_identity + 1U,
                                   allocation_identity + 2U)
                             : runtime::make_sm87_macrofeed_v4_request_state_admission(
                                   owner_identity, allocation_identity,
                                   allocation_identity + 1U,
                                   allocation_identity + 2U,
                                   kv_allocation_identity);
  auto request = runtime::Sm87MacroFeedV4RequestState::create(admission);
  test.expect(static_cast<bool>(request),
              "host RequestState binds the same Engine identity");
  if (!request) {
    execution.owner.reset();
    (void)cudaFree(recurrent_allocation);
    return nullptr;
  }
  auto request_access = request.state->issue_sealed_access();
  return std::make_unique<BoundOwner>(std::move(execution),
                                      std::move(request), request_access,
                                      recurrent_allocation);
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
[[nodiscard]] float fp32_from_bits(const std::uint32_t bits) noexcept {
  float value = 0.0F;
  static_assert(sizeof(value) == sizeof(bits));
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

[[nodiscard]] kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt
make_gate_up_payload_receipt(
    const support::Sm87MacroFeedV4LiveNvFp4AssetFixture& fixture) noexcept {
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(
      kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp);
  kernels::Sm87MacroFeedV3NvFp4GateUpPayloadReceipt receipt;
  receipt.plan_identity = kernels::kSm87MacroFeedV3NvFp4GateUpIdentity;
  receipt.payload_identity = fixture.asset.artifact_identity;
  receipt.gate_source_identity = fixture.inventory.sources[0U].tensor_identity;
  receipt.up_source_identity = fixture.inventory.sources[1U].tensor_identity;
  receipt.device_ordinal = fixture.upload.device_ordinal;
  receipt.payload_begin = fixture.asset.payload.begin;
  receipt.payload_end = fixture.asset.payload.end;
  receipt.payload_bytes = fixture.asset.payload.bytes;
  receipt.gate_partition_bytes = layout.partitions[0U].payload_bytes;
  receipt.up_partition_bytes = layout.partitions[1U].payload_bytes;
  receipt.canonical_consumer_n64_k16_lane_component_v1 =
      fixture.asset.transform_identity ==
      kernels::Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.canonical_gate_then_up_partition_order =
      layout.partitions[0U].logical_role ==
          kernels::Sm87TargetAotLogicalRole::kNvFp4Gate &&
      layout.partitions[1U].logical_role ==
          kernels::Sm87TargetAotLogicalRole::kNvFp4Up;
  receipt.independent_tensor_scales =
      layout.partitions[0U].independent_tensor_scale &&
      layout.partitions[1U].independent_tensor_scale;
  receipt.host_bytes_authenticated_before_copy =
      fixture.upload.host_payload_digest_verified_before_copy &&
      fixture.upload.host_payload_immutable_until_completion;
  receipt.device_readback_authenticated =
      fixture.upload.verification_event_observed &&
      fixture.upload.verification_completed &&
      fixture.upload.device_payload_matches_host_payload;
  receipt.allocation_retained_for_launch =
      fixture.upload.allocation_retained_for_asset_lifetime;
  receipt.receipt_identity = kernels::
      sm87_macrofeed_v3_nvfp4_gate_up_compute_payload_receipt_identity(
          receipt);
  return receipt;
}

[[nodiscard]] kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt
make_down_payload_receipt(
    const support::Sm87MacroFeedV4LiveNvFp4AssetFixture& fixture) noexcept {
  kernels::Sm87MacroFeedV3NvFp4DownPayloadReceipt receipt;
  receipt.plan_identity = kernels::kSm87MacroFeedV3NvFp4DownIdentity;
  receipt.payload_identity = fixture.asset.artifact_identity;
  receipt.device_ordinal = fixture.upload.device_ordinal;
  receipt.payload_begin = fixture.asset.payload.begin;
  receipt.payload_end = fixture.asset.payload.end;
  receipt.payload_bytes = fixture.asset.payload.bytes;
  receipt.canonical_consumer_n64_k16_lane_component_v1 =
      fixture.asset.transform_identity ==
      kernels::Sm87TargetAotProjectionPackedTransformIdentity::
          kCanonicalNkToConsumerN64K16LaneComponentV1;
  receipt.host_bytes_authenticated_before_copy =
      fixture.upload.host_payload_digest_verified_before_copy &&
      fixture.upload.host_payload_immutable_until_completion;
  receipt.device_readback_authenticated =
      fixture.upload.verification_event_observed &&
      fixture.upload.verification_completed &&
      fixture.upload.device_payload_matches_host_payload;
  receipt.allocation_retained_for_launch =
      fixture.upload.allocation_retained_for_asset_lifetime;
  receipt.receipt_identity =
      kernels::sm87_macrofeed_v3_nvfp4_down_compute_payload_receipt_identity(
          receipt);
  return receipt;
}

struct FullAttentionTransactionFixture final {
  static constexpr std::uint64_t kKvAllocationIdentity =
      0x5133'4655'4c4c'4b56ULL;
  support::Sm87MacroFeedV4LiveFp8AssetFixture full_qkv_asset;
  support::Sm87MacroFeedV4LiveFp8AssetFixture full_output_asset;
  support::Sm87MacroFeedV4LiveNvFp4AssetFixture gate_up_asset;
  support::Sm87MacroFeedV4LiveNvFp4AssetFixture down_asset;
  support::Sm87MacroFeedV4LiveFp8Allocation hidden_a;
  support::Sm87MacroFeedV4LiveFp8Allocation hidden_b;
  support::Sm87MacroFeedV4LiveFp8Allocation scratch;
  support::Sm87MacroFeedV4LiveFp8Allocation kv_layer;
  support::Sm87MacroFeedV4LiveFp8Allocation input_norm_weight;
  support::Sm87MacroFeedV4LiveFp8Allocation post_norm_weight;
  support::Sm87MacroFeedV4LiveFp8Allocation q_norm_weight;
  support::Sm87MacroFeedV4LiveFp8Allocation k_norm_weight;
  support::Sm87MacroFeedV4LiveFp8Allocation cosines;
  support::Sm87MacroFeedV4LiveFp8Allocation sines;
  events::Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission
      submission{};

  template <typename T>
  [[nodiscard]] static T* as(
      support::Sm87MacroFeedV4LiveFp8Allocation& allocation) noexcept {
    return static_cast<T*>(allocation.data());
  }

  template <typename T>
  [[nodiscard]] static T* at(
      support::Sm87MacroFeedV4LiveFp8Allocation& allocation,
      const std::uint64_t byte_offset) noexcept {
    return reinterpret_cast<T*>(
        static_cast<std::uint8_t*>(allocation.data()) + byte_offset);
  }

  [[nodiscard]] bool initialize(const int device_ordinal) noexcept {
    if (!full_qkv_asset.initialize(
            kernels::Sm87TargetAotProjectionRole::kFp8FullQkv,
            device_ordinal) ||
        !full_output_asset.initialize(
            kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
            device_ordinal) ||
        !gate_up_asset.initialize(
            kernels::Sm87TargetAotProjectionRole::kNvFp4GateUp,
            device_ordinal) ||
        !down_asset.initialize(
            kernels::Sm87TargetAotProjectionRole::kNvFp4Down,
            device_ordinal) ||
        !hidden_a.allocate_zeroed(
            kernels::kSm87MacroFeedV4NormResidualHiddenBytes) ||
        !hidden_b.allocate_zeroed(
            kernels::kSm87MacroFeedV4NormResidualHiddenBytes) ||
        !scratch.allocate_zeroed(
            kernels::kSm87MacroFeedV4AttentionC8000ScratchBytes) ||
        !kv_layer.allocate_zeroed(
            runtime::kSm87MacroFeedV4AttentionKvLayerBytes) ||
        !input_norm_weight.allocate_zeroed(
            kernels::kSm87MacroFeedV4NormResidualWeightBytes) ||
        !post_norm_weight.allocate_zeroed(
            kernels::kSm87MacroFeedV4NormResidualWeightBytes) ||
        !q_norm_weight.allocate_zeroed(
            kernels::
                kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes) ||
        !k_norm_weight.allocate_zeroed(
            kernels::
                kSm87MacroFeedV4FullAttentionPreprocessNormWeightBytes) ||
        !cosines.allocate_zeroed(
            kernels::kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes) ||
        !sines.allocate_zeroed(
            kernels::kSm87MacroFeedV4FullAttentionPreprocessRopeTableBytes)) {
      return false;
    }

    submission.execution_package_identity = 0x5100U;
    submission.full_attention_catalog_identity = 0x5101U;
    submission.full_attention_binding_identity = 0x5102U;
    submission.mlp_binding_identity = 0x5103U;
    submission.input_norm_binding_identity = 0x5104U;
    submission.post_norm_binding_identity = 0x5105U;
    submission.rope_binding_identity = 0x5106U;
    submission.resource_bundle_identity = 0x5107U;
    submission.full_attention_ordinal = 0U;
    submission.model_layer = 3U;
    submission.input_norm = {
        as<std::uint16_t>(hidden_a),
        as<std::uint16_t>(input_norm_weight),
        as<std::uint16_t>(hidden_b),
        kernels::kSm87MacroFeedV4NormResidualTokens,
        kernels::kSm87MacroFeedV4NormResidualHidden,
        kernels::kSm87MacroFeedV4NormResidualEpsilonFp32Bits,
        nullptr};
    submission.full_qkv = {
        as<std::uint16_t>(hidden_b),
        full_qkv_asset.asset,
        as<std::uint16_t>(scratch),
        as<std::uint16_t>(kv_layer),
        at<std::uint16_t>(
            kv_layer, runtime::kSm87MacroFeedV4AttentionKvPlaneBytes)};
    submission.preprocess = {
        as<std::uint16_t>(scratch),
        as<std::uint16_t>(kv_layer),
        as<std::uint16_t>(q_norm_weight),
        as<std::uint16_t>(k_norm_weight),
        as<float>(cosines),
        as<float>(sines),
        0U};
    submission.attention = {
        as<std::uint16_t>(scratch),
        as<std::uint16_t>(kv_layer),
        at<std::uint16_t>(
            kv_layer, runtime::kSm87MacroFeedV4AttentionKvPlaneBytes),
        0U};
    submission.full_output = {
        as<std::uint16_t>(scratch), full_output_asset.asset,
        as<std::uint16_t>(hidden_b)};
    submission.residual_post_norm = {
        as<std::uint16_t>(hidden_a), as<std::uint16_t>(hidden_b),
        as<std::uint16_t>(post_norm_weight)};
    submission.gate_up.normalized_input = as<std::uint16_t>(hidden_a);
    submission.gate_up.payload = reinterpret_cast<const std::uint8_t*>(
        gate_up_asset.asset.payload.begin);
    submission.gate_up.payload_bytes = gate_up_asset.asset.payload.bytes;
    submission.gate_up.gate_tensor_scale =
        fp32_from_bits(gate_up_asset.asset.tensor_scale_bits[0U]);
    submission.gate_up.up_tensor_scale =
        fp32_from_bits(gate_up_asset.asset.tensor_scale_bits[1U]);
    submission.gate_up.intermediate_output = as<std::uint16_t>(scratch);
    submission.gate_up.canonical_v3_payload_receipt =
        make_gate_up_payload_receipt(gate_up_asset);
    submission.down.intermediate_input = as<std::uint16_t>(scratch);
    submission.down.payload = reinterpret_cast<const std::uint8_t*>(
        down_asset.asset.payload.begin);
    submission.down.payload_bytes = down_asset.asset.payload.bytes;
    submission.down.tensor_scale =
        fp32_from_bits(down_asset.asset.tensor_scale_bits[0U]);
    submission.down.residual_output = as<std::uint16_t>(hidden_b);
    submission.down.payload_receipt = make_down_payload_receipt(down_asset);

    if (kernels::query_sm87_macrofeed_v4_norm_residual_admission_resources_cuda(
            &submission.norm_resources) != static_cast<int>(cudaSuccess) ||
        kernels::query_sm87_macrofeed_v4_fp8_cuda_resources(
            kernels::Sm87TargetAotProjectionRole::kFp8FullQkv,
            kernels::Sm87MacroFeedV4Fp8InputLayout::
                kHiddenContiguousH5120V1,
            &submission.full_qkv_resources) !=
            static_cast<int>(cudaSuccess) ||
        kernels::
                query_sm87_macrofeed_v4_full_attention_preprocess_admission_resources_cuda(
                    &submission.preprocess_resources) !=
            static_cast<int>(cudaSuccess) ||
        kernels::query_sm87_macrofeed_v4_attention_c8000_admission_resources_cuda(
            &submission.attention_resources) !=
            static_cast<int>(cudaSuccess) ||
        kernels::query_sm87_macrofeed_v4_fp8_cuda_resources(
            kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput,
            kernels::Sm87MacroFeedV4Fp8InputLayout::
                kFullAttentionInterleavedQScratchV1,
            &submission.full_output_resources) !=
            static_cast<int>(cudaSuccess) ||
        kernels::query_sm87_macrofeed_v4_nvfp4_gate_up_cuda_resources(
            &submission.gate_up_resources) !=
            static_cast<int>(cudaSuccess) ||
        kernels::query_sm87_macrofeed_v4_nvfp4_down_cuda_resources(
            &submission.down_resources) != static_cast<int>(cudaSuccess)) {
      return false;
    }
    submission.full_qkv_resources.static_resource_gate_passed =
        kernels::sm87_macrofeed_v4_fp8_resource_gate(
            submission.full_qkv_resources);
    submission.full_output_resources.static_resource_gate_passed =
        kernels::sm87_macrofeed_v4_fp8_resource_gate(
            submission.full_output_resources);
    if (!kernels::sm87_macrofeed_v3_nvfp4_gate_up_payload_receipt_valid(
            submission.gate_up.canonical_v3_payload_receipt) ||
        !kernels::sm87_macrofeed_v3_nvfp4_down_payload_receipt_valid(
            submission.down.payload_receipt) ||
        cudaDeviceSynchronize() != cudaSuccess) {
      return false;
    }
    // Resource observation may intentionally probe unsupported attributes.
    // Clear only this construction-time test residue before the owner starts;
    // production transaction code never swallows a prior CUDA error.
    return cudaGetLastError() == cudaSuccess;
  }
};

[[nodiscard]] runtime::Sm87MacroFeedV4FullAttentionKvAuthorizationResult
prepare_first_full_attention_grant(Test& test, BoundOwner& bound) {
  auto& state = *bound.request.state;
  test.expect(static_cast<bool>(state.begin_panel(bound.request_access, 0U)),
              "Full fixture begins the matching RequestState panel");
  for (std::size_t model_layer = 0U; model_layer < 3U; ++model_layer) {
    auto authorization = state.authorize_gdn_layer_state(
        bound.request_access, 0U, model_layer);
    if (!authorization) {
      test.expect(false,
                  "Full fixture authorizes each preceding natural GDN layer");
      return {};
    }
    const auto committed = state.commit_gdn_layer_candidate_enqueued(
        bound.request_access, std::move(*authorization.grant));
    if (!committed) {
      test.expect(false,
                  "Full fixture commits each preceding natural GDN layer");
      return {};
    }
  }
  auto authorization =
      state.authorize_full_attention_kv(bound.request_access, 0U, 3U);
  test.expect(static_cast<bool>(authorization),
              "Full fixture mints the natural layer-3 move-only KV grant");
  return authorization;
}
#endif

void expect_physical_observation_forbidden(
    Test& test, events::Sm87MacroFeedV4ExecutionEventsOwner& owner,
    const events::Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const events::Sm87MacroFeedV4ExecutionEvent event) {
  const auto query = Fixture::observe_event_query(owner, panel_access, event);
  test.expect(
      !query &&
          query.status.error ==
              events::Sm87MacroFeedV4ExecutionError::
                  kPhysicalObservationForbidden &&
          query.status.event == event,
      "device-order-only event rejects host query observation");
  const auto synchronize =
      Fixture::observe_event_synchronize(owner, panel_access, event);
  test.expect(
      !synchronize &&
          synchronize.status.error ==
              events::Sm87MacroFeedV4ExecutionError::
                  kPhysicalObservationForbidden &&
          synchronize.status.event == event,
      "device-order-only event rejects host synchronize observation");
}

[[nodiscard]] bool run_ab_cycle(
    Test& test, events::Sm87MacroFeedV4ExecutionEventsOwner& owner,
    const events::Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const bool verify_observation_boundary) {
  const auto norm_record = Fixture::record_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kNormReady);
  test.expect(static_cast<bool>(norm_record), "Main records NormReady");
  if (!norm_record) {
    return false;
  }
  test.expect(!norm_record.receipt.physical_device_completion_attested &&
                  !norm_record.receipt.production_receipt_eligible,
              "record enqueue receipt has no completion authority");
  if (verify_observation_boundary) {
    expect_physical_observation_forbidden(
        test, owner, panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kNormReady);
  }

  const auto norm_wait = Fixture::wait_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kNormReady);
  const auto ab_record = Fixture::record_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kAbReady);
  if (verify_observation_boundary && ab_record) {
    expect_physical_observation_forbidden(
        test, owner, panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kAbReady);
  }
  const auto ab_wait = Fixture::wait_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kAbReady);
  test.expect(static_cast<bool>(norm_wait) && static_cast<bool>(ab_record) &&
                  static_cast<bool>(ab_wait),
              "one fixed NormReady/AbReady device-order cycle enqueues");
  return static_cast<bool>(norm_wait) && static_cast<bool>(ab_record) &&
         static_cast<bool>(ab_wait);
}

[[nodiscard]] bool enqueue_and_close_panel(
    Test& test, events::Sm87MacroFeedV4ExecutionEventsOwner& owner,
    const events::Sm87MacroFeedV4ExecutionPanelAccess& panel_access,
    const std::size_t expected_completed_panels) {
  for (std::size_t cycle = 0U;
       cycle < events::kSm87MacroFeedV4Bf16AbCyclesPerPanel; ++cycle) {
    if (!run_ab_cycle(test, owner, panel_access,
                      expected_completed_panels == 1U && cycle == 0U)) {
      return false;
    }
  }
  const auto panel_done = Fixture::record_event(
      owner, panel_access, events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kPanelDone);
  test.expect(static_cast<bool>(panel_done) &&
                  !panel_done.receipt.physical_device_completion_attested,
              "PanelDone is a device enqueue marker, not a host receipt");
  if (!panel_done) {
    return false;
  }
  if (expected_completed_panels == 1U) {
    expect_physical_observation_forbidden(
        test, owner, panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kPanelDone);
  }

  const auto closed = Fixture::close_panel(owner, panel_access);
  test.expect(static_cast<bool>(closed),
              "PanelDone enqueue advances host ledger without CUDA wait");
  const auto snapshot = owner.snapshot();
  test.expect(snapshot.completed_panels == expected_completed_panels &&
                  snapshot.completed_panels <=
                      runtime::kSm87MacroFeedV4PanelCount &&
                  snapshot.physical_completion_receipts_issued == 0U,
              "panel close is bounded and issues no physical receipt");

  const auto duplicate = Fixture::close_panel(owner, panel_access);
  test.expect(!duplicate &&
                  owner.snapshot().completed_panels ==
                      expected_completed_panels,
              "one panel generation is consumed exactly once");
  return static_cast<bool>(closed);
}

void test_five_panel_enqueue_without_host_barriers(Test& test) {
  auto bound = make_bound_owner(test, 0x2000U);
  if (bound == nullptr) {
    return;
  }
  auto& owner = *bound->execution.owner;
  test.expect(static_cast<bool>(Fixture::begin_request(
                  owner, *bound->request.state, bound->request_access)),
              "live admitted RequestState begins physical owner request");

  std::unique_ptr<events::Sm87MacroFeedV4ExecutionPanelAccess>
      final_panel_access;
  for (std::size_t panel = 0U;
       panel < runtime::kSm87MacroFeedV4PanelCount; ++panel) {
    auto panel_begin = Fixture::begin_panel(owner, panel);
    test.expect(static_cast<bool>(panel_begin) &&
                    panel_begin.panel_access->panel() == panel,
                "panels begin in fixed zero-to-four order");
    if (!panel_begin) {
      return;
    }
    if (!enqueue_and_close_panel(test, owner, *panel_begin.panel_access,
                                 panel + 1U)) {
      return;
    }
    if (panel + 1U == runtime::kSm87MacroFeedV4PanelCount) {
      final_panel_access = std::move(panel_begin.panel_access);
    }
  }

  test.expect(final_panel_access != nullptr &&
                  owner.snapshot().physical_completion_receipts_issued == 0U,
              "all five panels enqueue and close with no query/synchronize");
  if (final_panel_access == nullptr) {
    return;
  }

  const auto representation = Fixture::record_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady);
  const auto representation_join = Fixture::wait_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady);
  const auto copy_done = Fixture::record_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone);
  const auto copy_join = Fixture::wait_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone);
  const auto publish = Fixture::record_event(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kFinalPublish);
  test.expect(static_cast<bool>(representation) &&
                  static_cast<bool>(representation_join) &&
                  static_cast<bool>(copy_done) &&
                  static_cast<bool>(copy_join) &&
                  static_cast<bool>(publish),
              "fixed final representation/copy/publication chain enqueues");
  if (representation) {
    expect_physical_observation_forbidden(
        test, owner, *final_panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kFinalRepresentationReady);
  }
  if (copy_done) {
    expect_physical_observation_forbidden(
        test, owner, *final_panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kCanonicalCopyDone);
  }

  const auto final_observation = Fixture::observe_event_synchronize(
      owner, *final_panel_access,
      events::Sm87MacroFeedV4ExecutionEvent::kFinalPublish);
  test.expect(static_cast<bool>(final_observation) &&
                  final_observation.receipt.observed_by_synchronize() &&
                  final_observation.receipt
                      .physical_device_completion_attested() &&
                  !final_observation.receipt.production_receipt_eligible(),
              "FinalPublish alone crosses the physical completion boundary");
  test.expect(Fixture::completion_receipt_matches(
                  owner, *final_panel_access,
                  events::Sm87MacroFeedV4ExecutionEvent::kFinalPublish,
                  final_observation.receipt),
              "owner authenticates exact physical FinalPublish receipt");
  test.expect(static_cast<bool>(Fixture::complete_request(
                  owner, *final_panel_access, final_observation.receipt)),
              "physical FinalPublish receipt completes request");

  const auto snapshot = owner.snapshot();
  test.expect(snapshot.state ==
                  events::Sm87MacroFeedV4ExecutionOwnerState::
                      kRequestCompleted &&
                  snapshot.completed_panels ==
                      runtime::kSm87MacroFeedV4PanelCount &&
                  snapshot.physical_completion_receipts_issued == 1U,
              "complete request retains five panels and one physical receipt");
}

void test_dual_stream_discard_drain(Test& test) {
  auto bound = make_bound_owner(test, 0x4000U);
  if (bound == nullptr) {
    return;
  }
  auto& owner = *bound->execution.owner;
  test.expect(static_cast<bool>(Fixture::begin_request(
                  owner, *bound->request.state, bound->request_access)),
              "discard test begins request");
  auto panel = Fixture::begin_panel(owner, 0U);
  if (!panel) {
    test.expect(false, "discard test begins panel");
    return;
  }

  const auto main_tail = Fixture::record_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kMain,
      events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  const auto ab_tail = Fixture::record_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kAbAux,
      events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  if (main_tail) {
    expect_physical_observation_forbidden(
        test, owner, *panel.panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  }
  if (ab_tail) {
    expect_physical_observation_forbidden(
        test, owner, *panel.panel_access,
        events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  }
  const auto main_join = Fixture::wait_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kMainTail);
  const auto ab_join = Fixture::wait_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kAbTail);
  const auto drained_record = Fixture::record_event(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionStream::kControl,
      events::Sm87MacroFeedV4ExecutionEvent::kOwnerDrained);
  test.expect(static_cast<bool>(main_tail) && static_cast<bool>(ab_tail) &&
                  static_cast<bool>(main_join) &&
                  static_cast<bool>(ab_join) &&
                  static_cast<bool>(drained_record),
              "dual-stream tails join on Control before OwnerDrained");

  const events::Sm87MacroFeedV4PhysicalCompletionReceipt forged;
  test.expect(!Fixture::discard_after_drain(
                  owner, *panel.panel_access, forged),
              "forged completion cannot discard request");
  const auto drained = Fixture::observe_event_synchronize(
      owner, *panel.panel_access,
      events::Sm87MacroFeedV4ExecutionEvent::kOwnerDrained);
  test.expect(static_cast<bool>(drained) &&
                  drained.receipt.main_tail_generation() != 0U &&
                  drained.receipt.ab_tail_generation() != 0U,
              "physical OwnerDrained receipt binds both producer tails");
  test.expect(static_cast<bool>(Fixture::discard_after_drain(
                  owner, *panel.panel_access, drained.receipt)) &&
                  owner.snapshot().state ==
                      events::Sm87MacroFeedV4ExecutionOwnerState::
                          kRequestDiscarded,
              "physically drained request discards safely");
}

void test_request_owner_phase_and_identity_binding(Test& test) {
  auto first = make_bound_owner(test, 0x5000U);
  auto second = make_bound_owner(test, 0x6000U);
  if (first == nullptr || second == nullptr) {
    return;
  }

  const auto foreign_request = Fixture::begin_request(
      *second->execution.owner, *first->request.state, first->request_access);
  test.expect(!foreign_request &&
                  foreign_request.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kForeignRequestAccess,
              "request owner identity must match execution owner");

  const auto copied_access = first->request_access;
  const auto detached_copy = Fixture::begin_request(
      *second->execution.owner, *second->request.state, copied_access);
  test.expect(!detached_copy &&
                  detached_copy.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kForeignRequestAccess,
              "copied sealed access cannot detach from its RequestState");

  test.expect(static_cast<bool>(second->request.state->begin_panel(
                  second->request_access, 0U)),
              "host fixture moves RequestState out of admitted phase");
  const auto wrong_phase = Fixture::begin_request(
      *second->execution.owner, *second->request.state,
      second->request_access);
  test.expect(!wrong_phase &&
                  wrong_phase.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kForeignRequestAccess,
              "execution begin rejects non-admitted RequestState phase");
}

void test_poison_terminal_drain(Test& test) {
  auto bound = make_bound_owner(test, 0x7000U);
  if (bound == nullptr) {
    return;
  }
  auto& owner = *bound->execution.owner;
  test.expect(static_cast<bool>(Fixture::begin_request(
                  owner, *bound->request.state, bound->request_access)),
              "poison-drain test begins request");
  const auto drained = Fixture::inject_poison_and_drain(
      owner, events::Sm87MacroFeedV4ExecutionError::kCudaSubmission);
  const auto snapshot = owner.snapshot();
  test.expect(static_cast<bool>(drained) &&
                  drained.poison_cause.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kCudaSubmission &&
                  snapshot.state ==
                      events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned &&
                  snapshot.poisoned_terminal_quiescence_attested &&
                  snapshot.poison_cause.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kCudaSubmission &&
                  drained.all_stream_synchronizations_attempted &&
                  drained.discard_required &&
                  snapshot
                      .poison_drain_all_stream_synchronizations_attempted &&
                  snapshot.poison_drain_stream_cuda_status ==
                      drained.stream_cuda_status,
              "terminal three-stream drain attests quiescence and preserves "
              "original CUDA failure");
  const auto forbidden_reentry = Fixture::begin_request(
      owner, *bound->request.state, bound->request_access);
  test.expect(!forbidden_reentry &&
                  forbidden_reentry.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kInvalidOwnerState,
              "physically drained poisoned owner can never admit a new "
              "request");
  const auto repeated_drain = Fixture::inject_poison_and_drain(
      owner, events::Sm87MacroFeedV4ExecutionError::kCudaSubmission);
  test.expect(!repeated_drain &&
                  repeated_drain.drain_status.error ==
                      events::Sm87MacroFeedV4ExecutionError::
                          kInvalidOwnerState,
              "terminal poison drain is one-shot");
}

#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
void expect_full_attention_prevalidation_rejection(
    Test& test, FullAttentionTransactionFixture& full,
    const events::Sm87MacroFeedV4CompleteFullAttentionLayerC8000Submission&
        submission,
    const std::uint64_t allocation_identity,
    const std::string_view case_name) {
  auto bound = make_bound_owner(test, allocation_identity,
                                full.kKvAllocationIdentity);
  if (bound == nullptr) {
    return;
  }
  auto& owner = *bound->execution.owner;
  test.expect(static_cast<bool>(Fixture::begin_request(
                  owner, *bound->request.state, bound->request_access)),
              "full negative begins request");
  auto panel = Fixture::begin_panel(owner, 0U);
  if (!panel) {
    test.expect(false, "full negative begins panel");
    return;
  }
  auto authorization = prepare_first_full_attention_grant(test, *bound);
  if (!authorization) {
    return;
  }
  const auto before = owner.snapshot();
  const auto rejected =
      Fixture::submit_complete_full_attention_layer_c8000_prevalidated(
          owner, *panel.panel_access, *authorization.grant, submission);
  const auto after = owner.snapshot();
  test.expect(
      !rejected &&
          rejected.status.error ==
              events::Sm87MacroFeedV4ExecutionError::kKernelSubmitContract &&
          !rejected.receipt.valid_shape() &&
          after.state ==
              events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned &&
          after.bound_kernel_submissions == before.bound_kernel_submissions &&
          after.last_full_attention_accepted_prefix.valid_prefix() &&
          after.last_full_attention_accepted_prefix
                  .accepted_kernel_launches == 0U &&
          !after.last_full_attention_accepted_prefix.complete &&
          after.complete_full_attention_layers_submitted == 0U &&
          after.event_generations == before.event_generations &&
          after.bf16_ab_cycles_completed ==
              before.bf16_ab_cycles_completed &&
          after.bf16_ab_cycle_at_norm_boundary ==
              before.bf16_ab_cycle_at_norm_boundary,
      case_name);
  const auto drained = Fixture::drain_poisoned_request_and_discard(
      owner, *bound->request.state, bound->request_access);
  const auto request_after_drain = bound->request.state->snapshot();
  test.expect(static_cast<bool>(drained) &&
                  drained.all_stream_synchronizations_attempted &&
                  drained.request_state_discarded &&
                  static_cast<bool>(drained.request_state_status) &&
                  request_after_drain.phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::kFailed &&
                  request_after_drain.candidate_discard_count == 1U &&
                  request_after_drain
                          .pending_full_attention_kv_grant_identity == 0U,
              "full negative performs terminal three-stream drain and "
              "physically-authorized RequestState discard");
}

void test_full_attention_foreign_grant_rejection(
    Test& test, FullAttentionTransactionFixture& full) {
  auto issuer = make_bound_owner(test, 0x8200U,
                                 full.kKvAllocationIdentity);
  auto foreign = make_bound_owner(test, 0x8300U,
                                  full.kKvAllocationIdentity + 1U);
  if (issuer == nullptr || foreign == nullptr) {
    return;
  }
  auto& issuer_owner = *issuer->execution.owner;
  auto& foreign_owner = *foreign->execution.owner;
  if (!Fixture::begin_request(issuer_owner, *issuer->request.state,
                              issuer->request_access) ||
      !Fixture::begin_request(foreign_owner, *foreign->request.state,
                              foreign->request_access)) {
    test.expect(false, "foreign-grant fixture begins both requests");
    return;
  }
  auto issuer_panel = Fixture::begin_panel(issuer_owner, 0U);
  auto foreign_panel = Fixture::begin_panel(foreign_owner, 0U);
  if (!issuer_panel || !foreign_panel) {
    test.expect(false, "foreign-grant fixture begins both panels");
    return;
  }
  auto issuer_authorization =
      prepare_first_full_attention_grant(test, *issuer);
  auto foreign_authorization =
      prepare_first_full_attention_grant(test, *foreign);
  if (!issuer_authorization || !foreign_authorization) {
    return;
  }
  const auto before = issuer_owner.snapshot();
  const auto rejected =
      Fixture::submit_complete_full_attention_layer_c8000_prevalidated(
          issuer_owner, *issuer_panel.panel_access,
          *foreign_authorization.grant, full.submission);
  const auto after = issuer_owner.snapshot();
  test.expect(
      !rejected &&
          rejected.status.error ==
              events::Sm87MacroFeedV4ExecutionError::kKernelSubmitContract &&
          !rejected.receipt.valid_shape() &&
          after.state ==
              events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned &&
          after.bound_kernel_submissions == before.bound_kernel_submissions &&
          after.last_full_attention_accepted_prefix.valid_prefix() &&
          after.last_full_attention_accepted_prefix.grant_identity ==
              foreign_authorization.grant->grant_identity() &&
          after.last_full_attention_accepted_prefix
                  .kv_allocation_identity ==
              foreign_authorization.grant->kv_allocation_identity() &&
          after.last_full_attention_accepted_prefix
                  .accepted_kernel_launches == 0U,
      "same-panel transaction rejects foreign grant/KV-allocation authority "
      "before its first enqueue");
  const auto drained = Fixture::drain_poisoned_request_and_discard(
      issuer_owner, *issuer->request.state, issuer->request_access);
  test.expect(static_cast<bool>(drained) && drained.request_state_discarded &&
                  issuer->request.state->snapshot().phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::kFailed,
              "foreign-grant rejection drains and discards only the issuing "
              "request generation");
}

void test_full_attention_grant_at_most_once(
    Test& test, FullAttentionTransactionFixture& full) {
  auto bound = make_bound_owner(test, 0x8400U,
                                full.kKvAllocationIdentity);
  if (bound == nullptr) {
    return;
  }
  auto& owner = *bound->execution.owner;
  if (!Fixture::begin_request(owner, *bound->request.state,
                              bound->request_access)) {
    test.expect(false, "replay fixture begins request");
    return;
  }
  auto panel = Fixture::begin_panel(owner, 0U);
  if (!panel) {
    test.expect(false, "replay fixture begins panel");
    return;
  }
  auto authorization = prepare_first_full_attention_grant(test, *bound);
  if (!authorization) {
    return;
  }
  const auto first =
      Fixture::submit_complete_full_attention_layer_c8000_prevalidated(
          owner, *panel.panel_access, *authorization.grant,
          full.submission);
  test.expect(static_cast<bool>(first) && first.receipt.valid_shape(),
              "replay fixture accepts the grant exactly once");
  if (!first) {
    return;
  }
  const auto once = owner.snapshot();
  const auto replayed =
      Fixture::submit_complete_full_attention_layer_c8000_prevalidated(
          owner, *panel.panel_access, *authorization.grant,
          full.submission);
  const auto twice = owner.snapshot();
  test.expect(
      !replayed &&
          replayed.status.error ==
              events::Sm87MacroFeedV4ExecutionError::kKernelSubmitContract &&
          !replayed.receipt.valid_shape() &&
          twice.state ==
              events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned &&
          once.bound_kernel_submissions == 8U &&
          twice.bound_kernel_submissions == once.bound_kernel_submissions &&
          once.accepted_full_attention_grants == 1U &&
          twice.accepted_full_attention_grants == 1U &&
          twice.complete_full_attention_layers_submitted == 1U &&
          twice.last_full_attention_accepted_prefix.valid_prefix() &&
          twice.last_full_attention_accepted_prefix.grant_identity ==
              authorization.grant->grant_identity() &&
          twice.last_full_attention_accepted_prefix
                  .accepted_kernel_launches == 0U &&
          !twice.last_full_attention_accepted_prefix.complete,
      "a successful Full grant is at-most-once: replay poisons with zero "
      "additional kernels and no receipt");
  const auto drained = Fixture::drain_poisoned_request_and_discard(
      owner, *bound->request.state, bound->request_access);
  const auto request_after_drain = bound->request.state->snapshot();
  test.expect(static_cast<bool>(drained) && drained.request_state_discarded &&
                  request_after_drain.phase ==
                      runtime::Sm87MacroFeedV4RequestStatePhase::kFailed &&
                  request_after_drain
                          .pending_full_attention_kv_grant_identity == 0U,
              "replay poison drains all streams and physically discards the "
              "RequestState grant");
}

void test_complete_full_attention_transaction(Test& test) {
  int device = -1;
  if (cudaGetDevice(&device) != cudaSuccess) {
    test.expect(false, "full transaction observes current CUDA device");
    return;
  }
  FullAttentionTransactionFixture full;
  test.expect(full.initialize(device),
              "full transaction constructs seven honest resource seals and "
              "eight typed bindings");
  if (!kernels::sm87_macrofeed_v4_norm_residual_resource_gate(
          full.submission.norm_resources) ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(
          full.submission.full_qkv_resources) ||
      !kernels::
          sm87_macrofeed_v4_full_attention_preprocess_admission_resource_gate(
              full.submission.preprocess_resources) ||
      !kernels::sm87_macrofeed_v4_attention_c8000_admission_resource_gate(
          full.submission.attention_resources) ||
      !kernels::sm87_macrofeed_v4_fp8_resource_gate(
          full.submission.full_output_resources) ||
      !kernels::sm87_macrofeed_v4_nvfp4_gate_up_resource_gate(
          full.submission.gate_up_resources) ||
      !kernels::sm87_macrofeed_v4_nvfp4_down_resource_gate(
          full.submission.down_resources)) {
    return;
  }

  events::Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt
      issued_receipt;
  {
    auto bound = make_bound_owner(test, 0x8000U,
                                  full.kKvAllocationIdentity);
    if (bound == nullptr) {
      return;
    }
    auto& owner = *bound->execution.owner;
    test.expect(static_cast<bool>(Fixture::begin_request(
                    owner, *bound->request.state, bound->request_access)),
                "full transaction begins request");
    auto panel = Fixture::begin_panel(owner, 0U);
    if (!panel) {
      test.expect(false, "full transaction begins panel zero");
      return;
    }
    auto authorization = prepare_first_full_attention_grant(test, *bound);
    if (!authorization) {
      return;
    }
    const auto before = owner.snapshot();
    const events::Sm87MacroFeedV4CompleteFullAttentionLayerEnqueueReceipt
        forged;
    test.expect(!Fixture::full_attention_receipt_matches(
                    owner, *panel.panel_access, *authorization.grant,
                    full.submission, forged),
                "default Full receipt has no owner authority");
    const auto enqueued =
        Fixture::submit_complete_full_attention_layer_c8000_prevalidated(
            owner, *panel.panel_access, *authorization.grant,
            full.submission);
    const auto after = owner.snapshot();
    issued_receipt = enqueued.receipt;
    if (!enqueued) {
      std::cerr << "full positive error="
                << static_cast<unsigned>(enqueued.status.error)
                << " context=" << enqueued.status.context
                << " cuda=" << enqueued.status.cuda_error
                << " accepted="
                << after.last_full_attention_accepted_prefix
                       .accepted_kernel_launches
                << " ledger_valid="
                << after.last_full_attention_accepted_prefix.valid_prefix()
                << " grant="
                << after.last_full_attention_accepted_prefix.grant_identity
                << " state_epoch="
                << after.last_full_attention_accepted_prefix.grant_state_epoch
                << " kv="
                << after.last_full_attention_accepted_prefix
                       .kv_allocation_identity
                << " ordinal="
                << after.last_full_attention_accepted_prefix
                       .full_attention_ordinal
                << " layer="
                << after.last_full_attention_accepted_prefix.model_layer
                << " grant_owner=" << authorization.grant->owner_identity()
                << " request=" << authorization.grant->request_epoch()
                << " panel=" << authorization.grant->panel()
                << " key_full="
                << authorization.grant->key_full_allocation_origin()
                << " value_full="
                << authorization.grant->value_full_allocation_origin()
                << " key_panel="
                << authorization.grant->key_panel_allocation_offset()
                << " value_panel="
                << authorization.grant->value_panel_allocation_offset()
                << " panel_bytes=" << authorization.grant->panel_bytes()
                << " first=" << authorization.grant->first_position()
                << " prev=" << authorization.grant->previous_valid_end()
                << " candidate=" << authorization.grant->candidate_end()
                << '\n';
    }
    test.expect(
        static_cast<bool>(enqueued) && enqueued.receipt.valid_shape() &&
            enqueued.receipt.first_position() == 0U &&
            enqueued.receipt.grant_identity() ==
                authorization.grant->grant_identity() &&
            enqueued.receipt.kv_allocation_identity() ==
                full.kKvAllocationIdentity &&
            enqueued.receipt.full_attention_ordinal() == 0U &&
            enqueued.receipt.model_layer() == 3U &&
            enqueued.receipt.full_attention_catalog_identity() ==
                full.submission.full_attention_catalog_identity &&
            enqueued.receipt.resource_bundle_identity() ==
                full.submission.resource_bundle_identity &&
            enqueued.receipt.bound_kernel_submissions() == 8U &&
            enqueued.receipt.asynchronous_d2d_copies() == 0U &&
            enqueued.receipt.asynchronous_d2d_copy_bytes() == 0U &&
            enqueued.receipt.complete_layer_enqueued() &&
            !enqueued.receipt.physical_device_completion_attested() &&
            !enqueued.receipt.panel_complete() &&
            !enqueued.receipt.production_receipt_eligible(),
        "one Main token accepts exact Full DAG of eight kernels and zero "
        "copies");
    test.expect(
        Fixture::full_attention_receipt_matches(
            owner, *panel.panel_access, *authorization.grant,
            full.submission, enqueued.receipt),
        "issuing owner authenticates private Full enqueue receipt");
    auto substituted_submission = full.submission;
    ++substituted_submission.resource_bundle_identity;
    test.expect(
        !Fixture::full_attention_receipt_matches(
            owner, *panel.panel_access, *authorization.grant,
            substituted_submission, enqueued.receipt),
        "receipt rejects a nonzero sealed resource-bundle substitution");
    substituted_submission = full.submission;
    ++substituted_submission.full_attention_catalog_identity;
    test.expect(
        !Fixture::full_attention_receipt_matches(
            owner, *panel.panel_access, *authorization.grant,
            substituted_submission, enqueued.receipt),
        "receipt rejects a nonzero sealed Full-catalog substitution");
    substituted_submission = full.submission;
    ++substituted_submission.mlp_binding_identity;
    test.expect(
        !Fixture::full_attention_receipt_matches(
            owner, *panel.panel_access, *authorization.grant,
            substituted_submission, enqueued.receipt),
        "receipt rejects a nonzero sealed MLP-binding substitution");
    substituted_submission = full.submission;
    ++substituted_submission.rope_binding_identity;
    test.expect(
        !Fixture::full_attention_receipt_matches(
            owner, *panel.panel_access, *authorization.grant,
            substituted_submission, enqueued.receipt),
        "receipt rejects a nonzero sealed RoPE-binding substitution");
    substituted_submission = full.submission;
    ++substituted_submission.attention.value_cache_origin;
    test.expect(
        !Fixture::full_attention_receipt_matches(
            owner, *panel.panel_access, *authorization.grant,
            substituted_submission, enqueued.receipt),
        "submission digest rejects an argument-pointer substitution while "
        "all sealed identities remain unchanged");
    substituted_submission = full.submission;
    ++substituted_submission.full_qkv.asset.artifact_identity;
    test.expect(
        !Fixture::full_attention_receipt_matches(
            owner, *panel.panel_access, *authorization.grant,
            substituted_submission, enqueued.receipt),
        "submission digest rejects an asset substitution while all sealed "
        "identities remain unchanged");
    substituted_submission = full.submission;
    ++substituted_submission.attention_resources.kernel.grid_x;
    test.expect(
        !Fixture::full_attention_receipt_matches(
            owner, *panel.panel_access, *authorization.grant,
            substituted_submission, enqueued.receipt),
        "submission digest rejects a resource-geometry substitution while "
        "all sealed identities remain unchanged");
    test.expect(
        after.bound_kernel_submissions ==
                before.bound_kernel_submissions + 8U &&
            after.input_norm_submissions ==
                before.input_norm_submissions + 1U &&
            after.full_qkv_c8000_submissions == 1U &&
            after.full_attention_preprocess_c8000_submissions == 1U &&
            after.attention_c8000_submissions == 1U &&
            after.full_attention_output_c8000_submissions == 1U &&
            after.residual_post_norm_submissions == 1U &&
            after.gate_up_c8000_submissions == 1U &&
            after.down_c8000_submissions == 1U &&
            after.complete_full_attention_layers_submitted == 1U &&
            after.accepted_full_attention_grants == 1U &&
            after.last_full_attention_accepted_prefix.valid_prefix() &&
            after.last_full_attention_accepted_prefix.complete &&
            after.last_full_attention_accepted_prefix
                    .accepted_kernel_launches == 8U &&
            after.last_full_attention_accepted_prefix
                    .asynchronous_d2d_copies == 0U &&
            after.last_full_attention_accepted_prefix
                    .asynchronous_d2d_copy_bytes == 0U,
        "accepted-prefix and per-seam ledgers close at exact 8/0");
    test.expect(
        after.event_generations == before.event_generations &&
            after.enqueue_receipts_issued == before.enqueue_receipts_issued &&
            after.bf16_ab_cycles_completed ==
                before.bf16_ab_cycles_completed &&
            after.bf16_ab_cycle_at_norm_boundary &&
            before.bf16_ab_cycle_at_norm_boundary,
        "Full InputNorm records no NormReady and leaves A/B phase, events, "
        "and BF16 cycle count unchanged");

    auto foreign = make_bound_owner(test, 0x8100U,
                                    full.kKvAllocationIdentity);
    if (foreign != nullptr) {
      auto& foreign_owner = *foreign->execution.owner;
      test.expect(static_cast<bool>(Fixture::begin_request(
                      foreign_owner, *foreign->request.state,
                      foreign->request_access)),
                  "foreign receipt test begins second request");
      auto foreign_panel = Fixture::begin_panel(foreign_owner, 0U);
      if (foreign_panel) {
        auto foreign_authorization =
            prepare_first_full_attention_grant(test, *foreign);
        if (!foreign_authorization) {
          return;
        }
        test.expect(
            !Fixture::full_attention_receipt_matches(
                foreign_owner, *foreign_panel.panel_access,
                *foreign_authorization.grant, full.submission,
                issued_receipt),
            "foreign owner cannot authenticate copied Full receipt");
        test.expect(
            !Fixture::full_attention_receipt_matches(
                owner, *panel.panel_access, *foreign_authorization.grant,
                full.submission, issued_receipt),
            "issuing owner rejects a foreign move-only KV grant substitution");
        const auto stale_on_issuer =
            Fixture::submit_complete_full_attention_layer_c8000_prevalidated(
                owner, *foreign_panel.panel_access, *authorization.grant,
                full.submission);
        const auto stale_on_foreign =
            Fixture::submit_complete_full_attention_layer_c8000_prevalidated(
                foreign_owner, *panel.panel_access,
                *foreign_authorization.grant, full.submission);
        test.expect(
            !stale_on_issuer && !stale_on_foreign &&
                stale_on_issuer.status.error ==
                    events::Sm87MacroFeedV4ExecutionError::
                        kStalePanelGeneration &&
                stale_on_foreign.status.error ==
                    events::Sm87MacroFeedV4ExecutionError::
                        kStalePanelGeneration &&
                owner.snapshot().state ==
                    events::Sm87MacroFeedV4ExecutionOwnerState::
                        kRequestActive &&
                foreign_owner.snapshot().state ==
                    events::Sm87MacroFeedV4ExecutionOwnerState::
                        kRequestActive,
            "foreign/stale panel grants reject before ledger creation or "
            "poison");
      }
    }
    const auto request_commit =
        bound->request.state->commit_full_attention_layer_enqueued(
            bound->request_access, std::move(*authorization.grant));
    const auto request_snapshot = bound->request.state->snapshot();
    test.expect(static_cast<bool>(request_commit) &&
                    request_snapshot.next_model_layer == 4U &&
                    request_snapshot.panel_kv_layers_staged == 1U &&
                    request_snapshot
                            .pending_full_attention_kv_grant_identity == 0U &&
                    request_snapshot.private_kv_valid_end == 0U &&
                    request_snapshot.candidate_kv_valid_end == 0U,
                "authenticated Full receipt closes natural layers 0-3 and "
                "commits the same grant without publishing a partial KV "
                "epoch");
  }

  test_full_attention_foreign_grant_rejection(test, full);
  test_full_attention_grant_at_most_once(test, full);

  std::uint64_t negative_identity = 0x9000U;
  auto negative = full.submission;
  negative.full_attention_catalog_identity = 0U;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "zero Full catalog identity fails before first enqueue");
  negative = full.submission;
  negative.resource_bundle_identity = 0U;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "zero resource-bundle identity fails before first enqueue");
  negative = full.submission;
  negative.model_layer = 7U;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "natural layer substitution fails before first enqueue");
  negative = full.submission;
  negative.norm_resources.device_ordinal = device + 1;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "foreign Norm device fails before first enqueue");
  negative = full.submission;
  negative.full_qkv_resources.identity =
      kernels::Sm87MacroFeedV4Fp8Identity::
          kGdnQkvZM64N128K64OrdinaryGridV1;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "Full-QKV rejects every tactic except exact 4602");
  negative = full.submission;
  negative.preprocess_resources.identity =
      kernels::Sm87MacroFeedV4FullAttentionPreprocessIdentity::kInvalid;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "foreign preprocess resource fails before first enqueue");
  negative = full.submission;
  negative.attention_resources.identity =
      kernels::Sm87MacroFeedV4AttentionC8000Identity::kInvalid;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "foreign Attention resource fails before first enqueue");
  negative = full.submission;
  negative.full_output_resources.identity =
      kernels::Sm87MacroFeedV4Fp8Identity::
          kGdnAttentionOutputM64N128K64OrdinaryGridV1;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "Full-O rejects every tactic except exact 4603");
  negative = full.submission;
  negative.gate_up_resources.device_ordinal = device + 1;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "foreign Gate/Up device fails before first enqueue");
  negative = full.submission;
  negative.down_resources.device_ordinal = device + 1;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "foreign Down device fails before first enqueue");
  negative = full.submission;
  ++negative.attention.q_gate_scratch;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "cross-step Q scratch alias mismatch fails before first enqueue");
  negative = full.submission;
  ++negative.full_qkv.key_panel_output;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "K panel slice must equal origin plus grant-derived offset");
  negative = full.submission;
  negative.attention.first_position =
      kernels::kSm87MacroFeedV4AttentionC8000Tokens;
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "Attention position must equal active panel and preprocess position");
  negative = full.submission;
  negative.input_norm.cuda_stream = reinterpret_cast<void*>(1U);
  expect_full_attention_prevalidation_rejection(
      test, full, negative, negative_identity++,
      "caller-supplied stream fails before first enqueue");

  for (std::size_t prefix = 0U; prefix <= 8U; ++prefix) {
    auto bound = make_bound_owner(test, negative_identity++,
                                  full.kKvAllocationIdentity);
    if (bound == nullptr) {
      return;
    }
    auto& owner = *bound->execution.owner;
    test.expect(static_cast<bool>(Fixture::begin_request(
                    owner, *bound->request.state, bound->request_access)),
                "partial-prefix test begins request");
    auto panel = Fixture::begin_panel(owner, 0U);
    if (!panel) {
      test.expect(false, "partial-prefix test begins panel");
      return;
    }
    auto authorization = prepare_first_full_attention_grant(test, *bound);
    if (!authorization) {
      return;
    }
    const auto before = owner.snapshot();
    test.expect(Fixture::fail_full_after_accepted_prefix(owner, prefix),
                "fixture arms one exact accepted-prefix failure");
    const auto failed =
        Fixture::submit_complete_full_attention_layer_c8000_prevalidated(
            owner, *panel.panel_access, *authorization.grant,
            full.submission);
    const auto after = owner.snapshot();
    const auto& ledger = after.last_full_attention_accepted_prefix;
    if (ledger.accepted_kernel_launches != prefix ||
        failed.status.error !=
            events::Sm87MacroFeedV4ExecutionError::kCudaSubmission) {
      std::cerr << "full prefix=" << prefix << " error="
                << static_cast<unsigned>(failed.status.error)
                << " context=" << failed.status.context
                << " cuda=" << failed.status.cuda_error
                << " accepted=" << ledger.accepted_kernel_launches
                << " valid=" << ledger.valid_prefix() << '\n';
    }
    test.expect(
        !failed &&
            failed.status.error ==
                events::Sm87MacroFeedV4ExecutionError::kCudaSubmission &&
            !failed.receipt.valid_shape() && ledger.valid_prefix() &&
            ledger.accepted_kernel_launches == prefix && !ledger.complete &&
            ledger.input_norm_launches == (prefix >= 1U ? 1U : 0U) &&
            ledger.full_qkv_launches == (prefix >= 2U ? 1U : 0U) &&
            ledger.preprocess_launches == (prefix >= 3U ? 1U : 0U) &&
            ledger.attention_launches == (prefix >= 4U ? 1U : 0U) &&
            ledger.full_output_launches == (prefix >= 5U ? 1U : 0U) &&
            ledger.residual_post_norm_launches ==
                (prefix >= 6U ? 1U : 0U) &&
            ledger.gate_up_launches == (prefix >= 7U ? 1U : 0U) &&
            ledger.down_launches == (prefix >= 8U ? 1U : 0U) &&
            ledger.asynchronous_d2d_copies == 0U &&
            ledger.asynchronous_d2d_copy_bytes == 0U &&
            after.bound_kernel_submissions ==
                before.bound_kernel_submissions + prefix &&
            after.complete_full_attention_layers_submitted == 0U &&
            after.event_generations == before.event_generations &&
            after.bf16_ab_cycles_completed ==
                before.bf16_ab_cycles_completed &&
            after.bf16_ab_cycle_at_norm_boundary ==
                before.bf16_ab_cycle_at_norm_boundary &&
            after.state ==
                events::Sm87MacroFeedV4ExecutionOwnerState::kPoisoned,
        "every accepted prefix is retained before failure and never mints a "
        "receipt");
    const auto drained = Fixture::drain_poisoned_request_and_discard(
        owner, *bound->request.state, bound->request_access);
    const auto request_after_drain = bound->request.state->snapshot();
    test.expect(static_cast<bool>(drained) &&
                    drained.all_stream_synchronizations_attempted &&
                    drained.request_state_discarded &&
                    static_cast<bool>(drained.request_state_status) &&
                    request_after_drain.phase ==
                        runtime::Sm87MacroFeedV4RequestStatePhase::kFailed &&
                    request_after_drain.candidate_discard_count == 1U &&
                    request_after_drain
                            .pending_full_attention_kv_grant_identity == 0U,
                "partial Full failure drains Main, AbAux, and Control and "
                "discards the exact RequestState candidate");
  }
}
#endif

[[nodiscard]] bool exact_target_device_available() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess) {
    return false;
  }
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    return false;
  }
  return properties.major == 8 && properties.minor == 7 &&
         properties.multiProcessorCount == 16;
}

}  // namespace

int main() {
  if (!exact_target_device_available()) {
    std::cout << "SKIP: exact SM87/16SM device unavailable\n";
    return 77;
  }

  Test test;
  test_five_panel_enqueue_without_host_barriers(test);
  test_dual_stream_discard_drain(test);
  test_request_owner_phase_and_identity_binding(test);
  test_poison_terminal_drain(test);
#if defined(Q3X_ENABLE_SM87_MACROFEED_V4_P40_EXECUTION_PACKAGE_ADMISSION)
  test_complete_full_attention_transaction(test);
#endif
  if (test.failures != 0) {
    std::cerr << "sm87_macrofeed_v4_execution_events_cuda_test: "
              << test.failures << " failure(s)\n";
    return 1;
  }
  std::cout << "sm87_macrofeed_v4_execution_events_cuda_test: PASS\n";
  return 0;
}
