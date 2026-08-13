#include "q3x/runtime/sm87_target_aot_projection_complete_device_assets.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string_view>

namespace {

using q3x::kernels::Sm87TargetAotProjectionRole;
using namespace q3x::runtime;

void require(const bool condition, const std::string_view message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

void test_exact_catalog_geometry() {
  require(kSm87TargetAotCompleteProjectionDeviceArtifactCount == 256U,
          "complete artifact count drifted");
  require(kSm87TargetAotCompleteProjectionDeviceSourceCount == 400U,
          "complete source count drifted");
  require(kSm87TargetAotCompleteProjectionDeviceArenaBytes ==
              16'840'130'560ULL,
          "complete arena size drifted");
  require(kSm87TargetAotCompleteProjectionCanonicalSourceD2hBytes ==
              16'840'132'160ULL,
          "complete canonical D2H accounting drifted");

  std::size_t artifacts = 0U;
  std::size_t sources = 0U;
  std::size_t linear_layers = 0U;
  std::size_t full_layers = 0U;
  std::uint64_t offset = 0U;
  for (std::size_t layer = 0U;
       layer < kSm87TargetAotCompleteProjectionDeviceLayerCount; ++layer) {
    const bool full = sm87_target_aot_complete_is_full_layer(layer);
    full ? ++full_layers : ++linear_layers;
    const Sm87TargetAotProjectionRole input_role =
        full ? Sm87TargetAotProjectionRole::kFp8FullQkv
             : Sm87TargetAotProjectionRole::kFp8GdnQkvZ;
    const std::array<Sm87TargetAotProjectionRole, 4U> roles{{
        Sm87TargetAotProjectionRole::kNvFp4GateUp,
        Sm87TargetAotProjectionRole::kNvFp4Down, input_role,
        Sm87TargetAotProjectionRole::kFp8AttentionOutput}};
    for (const Sm87TargetAotProjectionRole role : roles) {
      const auto ordinal =
          sm87_target_aot_complete_descriptor_ordinal(layer, role);
      const auto layout =
          q3x::kernels::sm87_target_aot_projection_packed_layout(role);
      require(ordinal == artifacts, "layer-major O(1) ordinal drifted");
      require(sm87_target_aot_complete_expected_arena_offset(layer, role) ==
                  offset,
              "layer/role O(1) arena offset drifted");
      require(layout.valid(), "complete role layout is invalid");
      require(offset % layout.payload_alignment == 0U,
              "complete artifact offset lost alignment");
      offset += layout.payload_bytes;
      sources += layout.partition_count;
      ++artifacts;
    }
    require(sm87_target_aot_complete_descriptor_ordinal(
                layer, full ? Sm87TargetAotProjectionRole::kFp8GdnQkvZ
                            : Sm87TargetAotProjectionRole::kFp8FullQkv) ==
                kSm87TargetAotCompleteProjectionDeviceArtifactCount,
            "wrong attention-input role did not fail closed");
  }
  require(linear_layers == 48U && full_layers == 16U,
          "linear/full layer schedule drifted");
  require(artifacts == kSm87TargetAotCompleteProjectionDeviceArtifactCount &&
              sources == kSm87TargetAotCompleteProjectionDeviceSourceCount &&
              offset == kSm87TargetAotCompleteProjectionDeviceArenaBytes,
          "complete catalog did not close exact counts and bytes");
}

void test_online_only_request_contract() {
  Sm87TargetAotCompleteOnlinePreparationRequest request;
  require(!sm87_target_aot_complete_online_request_valid(request),
          "invalid preparation phase was accepted");
  request.phase = Sm87TargetAotCompletePreparationPhase::kEngineStartup;
  require(sm87_target_aot_complete_online_request_valid(request),
          "engine-startup online preparation was rejected");
  request.create_bundle_path = std::filesystem::path("bundle-v2.bin");
  require(!sm87_target_aot_complete_online_request_valid(request),
          "bundle creation was accepted by online-only v2");
  request.create_bundle_path.clear();
  request.load_bundle_path = std::filesystem::path("bundle-v1.bin");
  require(!sm87_target_aot_complete_online_request_valid(request),
          "bundle direct loading was accepted by online-only v2");
}

void test_typed_zero_domains_are_complete() {
  q3x::kernels::Sm87TargetAotNvFp4CudaDeviceUploadReceipt nvfp4;
  q3x::kernels::Sm87TargetAotFp8CudaDeviceUploadReceipt fp8;
  q3x::kernels::Sm87TargetAotNvFp4CudaAssetView nvfp4_view;
  q3x::kernels::Sm87TargetAotFp8CudaAssetView fp8_view;
  require(sm87_target_aot_complete_nvfp4_upload_receipt_zero(nvfp4) &&
              sm87_target_aot_complete_fp8_upload_receipt_zero(fp8) &&
              sm87_target_aot_complete_nvfp4_view_zero(nvfp4_view) &&
              sm87_target_aot_complete_fp8_view_zero(fp8_view),
          "canonical typed zero domains were rejected");

  const auto nvfp4_copy = nvfp4;
  const auto fp8_copy = fp8;
  require(sm87_target_aot_complete_same_nvfp4_upload_receipt(
              nvfp4, nvfp4_copy) &&
              sm87_target_aot_complete_same_fp8_upload_receipt(fp8,
                                                               fp8_copy),
          "identical typed receipts were not equal");

  nvfp4.verification_event_observed = true;
  require(!sm87_target_aot_complete_nvfp4_upload_receipt_zero(nvfp4),
          "deep NVFP4 receipt mutation survived zero validation");
  fp8.compensated_tensor_scale_bf16_bits[2U] = 1U;
  require(!sm87_target_aot_complete_fp8_upload_receipt_zero(fp8),
          "compensated FP8 scale mutation survived zero validation");
  require(!sm87_target_aot_complete_same_fp8_upload_receipt(fp8, fp8_copy),
          "FP8 receipt/view equality ignored compensated scales");
  nvfp4_view.tensor_scale_bits[1U] = 1U;
  require(!sm87_target_aot_complete_nvfp4_view_zero(nvfp4_view),
          "deep NVFP4 view mutation survived zero validation");
  fp8_view.no_request_time_scale_conversion = true;
  require(!sm87_target_aot_complete_fp8_view_zero(fp8_view),
          "FP8 conversion flag mutation survived zero validation");
}

void test_owner_default_state() {
  Sm87TargetAotCompleteProjectionDeviceAssets owner;
  require(owner.empty(), "default complete owner is not empty");
  require(!owner.has_asset(0U, Sm87TargetAotProjectionRole::kNvFp4GateUp),
          "empty complete owner exposed an asset");
  require(!owner.has_asset(3U, Sm87TargetAotProjectionRole::kFp8FullQkv),
          "empty complete owner exposed a full-attention asset");
  require(owner.release() && owner.empty(),
          "empty complete owner did not release idempotently");
#if defined(Q3X_EXPECT_SM87_TARGET_AOT_COMPLETE_DEVICE_ASSETS_V2_ADMISSION)
  require(sm87_target_aot_complete_projection_device_assets_compiled(),
          "complete admission build reported unavailable");
#else
  require(!sm87_target_aot_complete_projection_device_assets_compiled(),
          "default-off build reported complete admission available");
#endif
}

}  // namespace

int main() {
  test_exact_catalog_geometry();
  test_online_only_request_contract();
  test_typed_zero_domains_are_complete();
  test_owner_default_state();
  return 0;
}
