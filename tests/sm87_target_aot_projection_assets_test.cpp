#include "q3x/runtime/sm87_target_aot_projection_assets.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

namespace kernels = q3x::kernels;
namespace runtime = q3x::runtime;

using Error = runtime::Sm87TargetAotProjectionAssetError;

class TestContext {
 public:
  void expect(const bool condition, const char* const message) {
    if (!condition) {
      std::cerr << "FAILED: " << message << '\n';
      ++failures_;
    }
  }

  [[nodiscard]] int failures() const noexcept { return failures_; }

 private:
  int failures_ = 0;
};

[[nodiscard]] int hex_value(const char value) noexcept {
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  return value - 'a' + 10;
}

[[nodiscard]] kernels::Sm87TargetAotProjectionSha256Digest digest_from_hex(
    const char* const text) noexcept {
  kernels::Sm87TargetAotProjectionSha256Digest digest;
  for (std::size_t index = 0U; index < digest.bytes.size(); ++index) {
    digest.bytes[index] = static_cast<std::uint8_t>(
        (hex_value(text[index * 2U]) << 4U) |
        hex_value(text[index * 2U + 1U]));
  }
  return digest;
}

[[nodiscard]] std::size_t checked_size(const std::uint64_t value) {
  return static_cast<std::size_t>(value);
}

void test_sha256(TestContext& test) {
  kernels::Sm87TargetAotProjectionSha256Digest digest{};
  test.expect(runtime::sm87_target_aot_projection_sha256({}, &digest) &&
                  digest == digest_from_hex(
                                "e3b0c44298fc1c149afbf4c8996fb924"
                                "27ae41e4649b934ca495991b7852b855"),
              "SHA-256 empty-vector authority");
  constexpr std::array<std::uint8_t, 3U> kAbc{{'a', 'b', 'c'}};
  test.expect(
      runtime::sm87_target_aot_projection_sha256(
          {kAbc.data(), kAbc.size()}, &digest) &&
          digest == digest_from_hex(
                        "ba7816bf8f01cfea414140de5dae2223"
                        "b00361a396177a9cb410ff61f20015ad"),
      "SHA-256 abc-vector authority");
  test.expect(!runtime::sm87_target_aot_projection_sha256(
                  {nullptr, 1U}, &digest),
              "non-empty null SHA span fails closed");
  const auto* const wrapping = reinterpret_cast<const std::uint8_t*>(
      std::numeric_limits<std::uintptr_t>::max() - 3U);
  test.expect(!runtime::sm87_target_aot_projection_sha256(
                  {wrapping, 8U}, &digest),
              "wrapping SHA span fails before dereference");
  test.expect(!runtime::sm87_target_aot_projection_sha256({}, nullptr),
              "null SHA destination fails closed");
}

[[nodiscard]] runtime::Sm87TargetAotProjectionSourceSet make_fp8_sources(
    const std::vector<std::uint8_t>& weight) {
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(
      kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput);
  runtime::Sm87TargetAotProjectionSourceSet sources;
  sources.role = layout.role;
  sources.inventory_identity = 0x4650'3853'5243'3031ULL;
  sources.source_count = 1U;
  sources.sources[0U].logical_role =
      kernels::Sm87TargetAotLogicalRole::kFp8AttentionOutput;
  sources.sources[0U].tensor_identity = 0x4650'3854'454e'3031ULL;
  sources.sources[0U].output_features =
      layout.partitions[0U].output_features;
  sources.sources[0U].input_features =
      layout.partitions[0U].input_features;
  sources.sources[0U].packed_weight = {weight.data(), weight.size()};
  sources.sources[0U].tensor_scale_bits = 0x3f80'0000U;
  return sources;
}

void test_fp8_asset(TestContext& test) {
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(
      kernels::Sm87TargetAotProjectionRole::kFp8AttentionOutput);
  const auto& partition = layout.partitions[0U];
  const std::size_t weight_bytes = checked_size(
      static_cast<std::uint64_t>(partition.output_features) *
      partition.input_features);
  std::vector<std::uint8_t> weight(weight_bytes);
  for (std::size_t index = 0U; index < weight.size(); ++index) {
    weight[index] = static_cast<std::uint8_t>(index * 131U + 17U);
  }
  constexpr std::size_t kPositiveNanN = 7U;
  constexpr std::size_t kPositiveNanK = 13U;
  constexpr std::size_t kNegativeNanN = 11U;
  constexpr std::size_t kNegativeNanK = 29U;
  weight[kPositiveNanN * partition.input_features + kPositiveNanK] = 0x7fU;
  weight[kNegativeNanN * partition.input_features + kNegativeNanK] = 0xffU;

  auto sources = make_fp8_sources(weight);
  const auto inspection =
      runtime::sm87_target_aot_projection_inspect_sources(sources);
  test.expect(static_cast<bool>(inspection),
              "FP8 raw source bytes produce a real inventory");
  kernels::Sm87TargetAotProjectionSha256Digest direct_weight_digest{};
  test.expect(runtime::sm87_target_aot_projection_sha256(
                  {weight.data(), weight.size()}, &direct_weight_digest) &&
                  direct_weight_digest ==
                      inspection.inventory.sources[0U].weight_digest,
              "FP8 inventory weight digest hashes complete source bytes");
  constexpr std::array<std::uint8_t, 4U> kScaleOneLittleEndian{{
      0x00U, 0x00U, 0x80U, 0x3fU}};
  kernels::Sm87TargetAotProjectionSha256Digest direct_scale_digest{};
  test.expect(runtime::sm87_target_aot_projection_sha256(
                  {kScaleOneLittleEndian.data(),
                   kScaleOneLittleEndian.size()},
                  &direct_scale_digest) &&
                  direct_scale_digest ==
                      inspection.inventory.sources[0U].scale_digest,
              "FP8 scale digest hashes the little-endian FP32 scale bits");

  std::vector<std::uint8_t> payload(checked_size(layout.payload_bytes), 0xa5U);
  const auto build = runtime::sm87_target_aot_projection_build_asset(
      0x4650'3841'5353'3031ULL, sources, inspection.inventory,
      {payload.data(), payload.size()});
  test.expect(static_cast<bool>(build),
              "FP8 source bytes build a sealed packed asset");
  test.expect(kernels::sm87_target_aot_projection_validate_transform_receipt(
                  build.manifest, inspection.inventory,
                  build.transform_receipt),
              "FP8 transform receipt passes the frozen schema validator");
  test.expect(runtime::sm87_target_aot_projection_validate_asset(
                  build.manifest, build.transform_receipt, sources,
                  inspection.inventory, {payload.data(), payload.size()}) ==
                  Error::kSuccess,
              "FP8 packed bytes pass full source/payload validation");

  const auto positive_address =
      kernels::sm87_target_aot_projection_packed_weight_address(
          layout, 0U, kPositiveNanN, kPositiveNanK);
  const auto negative_address =
      kernels::sm87_target_aot_projection_packed_weight_address(
          layout, 0U, kNegativeNanN, kNegativeNanK);
  test.expect(positive_address.valid && negative_address.valid &&
                  payload[positive_address.byte_offset] == 0x7fU &&
                  payload[negative_address.byte_offset] == 0xffU,
              "FP8 terminal codes remain raw Marlin bytes, not NaNs rewritten");
  test.expect(build.transform_receipt.partitions[0U]
                      .source_block_scale_e4m3fn_bytes_scanned == 0U &&
                  build.transform_receipt.partitions[0U]
                          .repacked_weight_values ==
                      static_cast<std::uint64_t>(partition.output_features) *
                          partition.input_features,
              "FP8 receipt distinguishes raw weights from NVFP4 scale domain");

  auto bad_inventory = inspection.inventory;
  bad_inventory.sources[0U].weight_digest.bytes[0U] ^= 1U;
  test.expect(runtime::sm87_target_aot_projection_build_asset(
                  0x4650'3841'5353'3032ULL, sources, bad_inventory,
                  {payload.data(), payload.size()})
                  .error == Error::kSourceDigestMismatch,
              "metadata weight-digest corruption fails before packing");

  weight[101U] ^= 0x20U;
  test.expect(runtime::sm87_target_aot_projection_build_asset(
                  0x4650'3841'5353'3033ULL, sources, inspection.inventory,
                  {payload.data(), payload.size()})
                  .error == Error::kSourceDigestMismatch,
              "source-byte corruption fails the expected inventory digest");
  weight[101U] ^= 0x20U;

  auto wrong_size = sources;
  --wrong_size.sources[0U].packed_weight.bytes;
  test.expect(runtime::sm87_target_aot_projection_inspect_sources(wrong_size)
                      .error == Error::kSourceSizeMismatch,
              "truncated FP8 source span fails closed");

  auto wrapping_source = sources;
  wrapping_source.sources[0U].packed_weight.data =
      reinterpret_cast<const std::uint8_t*>(
          std::numeric_limits<std::uintptr_t>::max() - 3U);
  test.expect(
      runtime::sm87_target_aot_projection_inspect_sources(wrapping_source)
              .error == Error::kSourceRangeOverflow,
      "wrapping source range fails before hashing");

  test.expect(runtime::sm87_target_aot_projection_build_asset(
                  0x4650'3841'5353'3034ULL, sources, inspection.inventory,
                  {payload.data(), payload.size() - 1U})
                  .error == Error::kPayloadSizeMismatch,
              "truncated output payload fails closed");
  test.expect(runtime::sm87_target_aot_projection_build_asset(
                  0x4650'3841'5353'3035ULL, sources, inspection.inventory,
                  {weight.data(), weight.size()})
                  .error == Error::kPayloadAliasing,
              "in-place FP8 source/payload alias is forbidden");

  const std::size_t corrupt_offset = positive_address.byte_offset + 1U;
  payload[corrupt_offset] ^= 1U;
  test.expect(runtime::sm87_target_aot_projection_validate_asset(
                  build.manifest, build.transform_receipt, sources,
                  inspection.inventory, {payload.data(), payload.size()}) ==
                  Error::kPayloadDigestMismatch,
              "ordinary FP8 payload corruption fails its real digest");

  auto forged_manifest = build.manifest;
  auto forged_receipt = build.transform_receipt;
  kernels::Sm87TargetAotProjectionSha256Digest forged_digest{};
  test.expect(runtime::sm87_target_aot_projection_sha256(
                  {payload.data(), payload.size()}, &forged_digest),
              "forged payload digest can be independently recomputed");
  forged_manifest.payload_digest = forged_digest;
  test.expect(kernels::sm87_target_aot_projection_seal_packed_manifest(
                  &forged_manifest),
              "forged manifest is structurally resealable for negative test");
  forged_receipt.payload.observed_payload_digest = forged_digest;
  test.expect(runtime::sm87_target_aot_projection_validate_asset(
                  forged_manifest, forged_receipt, sources,
                  inspection.inventory, {payload.data(), payload.size()}) ==
                  Error::kPayloadBijectionMismatch,
              "rehashed FP8 corruption still fails source bijection replay");
  payload[corrupt_offset] ^= 1U;

  auto bad_receipt = build.transform_receipt;
  bad_receipt.no_arithmetic_conversion = false;
  test.expect(runtime::sm87_target_aot_projection_validate_asset(
                  build.manifest, bad_receipt, sources, inspection.inventory,
                  {payload.data(), payload.size()}) ==
                  Error::kTransformReceiptMismatch,
              "mutated transform receipt fails closed");

  auto bad_manifest = build.manifest;
  bad_manifest.sources[0U].scale_digest.bytes[1U] ^= 1U;
  test.expect(kernels::sm87_target_aot_projection_seal_packed_manifest(
                  &bad_manifest) &&
                  runtime::sm87_target_aot_projection_validate_asset(
                      bad_manifest, build.transform_receipt, sources,
                      inspection.inventory,
                      {payload.data(), payload.size()}) ==
                      Error::kManifestMismatch,
              "resealed source-metadata corruption fails expected inventory");
}

[[nodiscard]] runtime::Sm87TargetAotProjectionSourceSet make_nvfp4_sources(
    const std::vector<std::uint8_t>& weight,
    const std::vector<std::uint8_t>& scales) {
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(
      kernels::Sm87TargetAotProjectionRole::kNvFp4Down);
  runtime::Sm87TargetAotProjectionSourceSet sources;
  sources.role = layout.role;
  sources.inventory_identity = 0x4e56'3453'5243'3031ULL;
  sources.source_count = 1U;
  sources.sources[0U].logical_role =
      kernels::Sm87TargetAotLogicalRole::kNvFp4Down;
  sources.sources[0U].tensor_identity = 0x4e56'3454'454e'3031ULL;
  sources.sources[0U].output_features =
      layout.partitions[0U].output_features;
  sources.sources[0U].input_features =
      layout.partitions[0U].input_features;
  sources.sources[0U].packed_weight = {weight.data(), weight.size()};
  sources.sources[0U].block_scale = {scales.data(), scales.size()};
  sources.sources[0U].tensor_scale_bits = 0x3f80'0000U;
  return sources;
}

void test_nvfp4_asset(TestContext& test) {
  const auto layout = kernels::sm87_target_aot_projection_packed_layout(
      kernels::Sm87TargetAotProjectionRole::kNvFp4Down);
  const auto& partition = layout.partitions[0U];
  const std::uint64_t values =
      static_cast<std::uint64_t>(partition.output_features) *
      partition.input_features;
  std::vector<std::uint8_t> weight(checked_size(values / 2U));
  std::vector<std::uint8_t> scales(checked_size(values / 16U));
  for (std::size_t index = 0U; index < weight.size(); ++index) {
    weight[index] = static_cast<std::uint8_t>(index * 73U + 29U);
  }
  constexpr std::array<std::uint8_t, 5U> kAdmissibleScales{{
      0x00U, 0x01U, 0x38U, 0x7eU, 0x80U}};
  for (std::size_t index = 0U; index < scales.size(); ++index) {
    scales[index] = kAdmissibleScales[index % kAdmissibleScales.size()];
  }
  auto sources = make_nvfp4_sources(weight, scales);
  const auto inspection =
      runtime::sm87_target_aot_projection_inspect_sources(sources);
  test.expect(static_cast<bool>(inspection),
              "finite non-negative NVFP4 scales, including -0, are admitted");

  std::vector<std::uint8_t> payload(checked_size(layout.payload_bytes), 0x5aU);
  const auto build = runtime::sm87_target_aot_projection_build_asset(
      0x4e56'3441'5353'3031ULL, sources, inspection.inventory,
      {payload.data(), payload.size()});
  test.expect(static_cast<bool>(build),
              "NVFP4 canonical nibbles and scales build a packed asset");
  test.expect(runtime::sm87_target_aot_projection_validate_asset(
                  build.manifest, build.transform_receipt, sources,
                  inspection.inventory, {payload.data(), payload.size()}) ==
                  Error::kSuccess,
              "NVFP4 packed bytes pass digest, domain, and bijection replay");
  test.expect(build.transform_receipt.partitions[0U]
                      .source_block_scale_e4m3fn_bytes_scanned ==
                  scales.size() &&
              build.transform_receipt.partitions[0U]
                      .payload_block_scale_e4m3fn_bytes_scanned ==
                  scales.size(),
              "NVFP4 receipt covers every source and payload scale byte");

  constexpr std::size_t kSampleN = 19U;
  constexpr std::size_t kSampleEvenK = 66U;
  constexpr std::size_t kSampleOddK = 67U;
  for (const std::size_t k : {kSampleEvenK, kSampleOddK}) {
    const auto address =
        kernels::sm87_target_aot_projection_packed_weight_address(
            layout, 0U, kSampleN, k);
    const std::uint8_t canonical = static_cast<std::uint8_t>(
        (weight[kSampleN * (partition.input_features / 2U) + k / 2U] >>
         (4U * (k & 1U))) &
        0x0fU);
    const std::uint8_t packed = static_cast<std::uint8_t>(
        (payload[address.byte_offset] >> (4U * address.nibble)) & 0x0fU);
    test.expect(address.valid && canonical == packed,
                "NVFP4 low/high canonical nibble follows packed bijection");
  }
  constexpr std::size_t kScaleN = 23U;
  constexpr std::size_t kScaleGroup = 37U;
  const auto scale_address =
      kernels::sm87_target_aot_projection_packed_scale_address(
          layout, 0U, kScaleN, kScaleGroup);
  test.expect(
      scale_address.valid &&
          payload[scale_address.byte_offset] ==
              scales[kScaleN * (partition.input_features / 16U) +
                     kScaleGroup],
      "NVFP4 E4M3FN block scale follows the same bit-exact layout map");

  const std::uint8_t original_scale = scales[0U];
  for (const std::uint8_t forbidden :
       std::array<std::uint8_t, 3U>{{0x7fU, 0xffU, 0x81U}}) {
    scales[0U] = forbidden;
    test.expect(runtime::sm87_target_aot_projection_inspect_sources(sources)
                        .error == Error::kForbiddenNvFp4BlockScale,
                "NVFP4 NaN and negative-nonzero scale codes fail closed");
  }
  scales[0U] = 0x80U;
  test.expect(static_cast<bool>(
                  runtime::sm87_target_aot_projection_inspect_sources(sources)),
              "NVFP4 signed zero is explicitly admissible");
  scales[0U] = original_scale;

  auto aliased = sources;
  aliased.sources[0U].block_scale = {weight.data(), scales.size()};
  test.expect(runtime::sm87_target_aot_projection_inspect_sources(aliased)
                      .error == Error::kSourceAliasing,
              "NVFP4 weight/scale source alias fails closed");

  scales[1U] = scales[1U] == 0x38U ? 0x39U : 0x38U;
  test.expect(runtime::sm87_target_aot_projection_build_asset(
                  0x4e56'3441'5353'3032ULL, sources, inspection.inventory,
                  {payload.data(), payload.size()})
                  .error == Error::kSourceDigestMismatch,
              "NVFP4 scale-byte mutation fails source scale_digest");
  scales[1U] = kAdmissibleScales[1U % kAdmissibleScales.size()];

  const std::uint8_t original_payload_scale =
      payload[scale_address.byte_offset];
  payload[scale_address.byte_offset] = 0x81U;
  auto forged_manifest = build.manifest;
  auto forged_receipt = build.transform_receipt;
  kernels::Sm87TargetAotProjectionSha256Digest forged_digest{};
  test.expect(runtime::sm87_target_aot_projection_sha256(
                  {payload.data(), payload.size()}, &forged_digest),
              "corrupt NVFP4 payload receives a real negative-test digest");
  forged_manifest.payload_digest = forged_digest;
  test.expect(kernels::sm87_target_aot_projection_seal_packed_manifest(
                  &forged_manifest),
              "corrupt NVFP4 manifest can be resealed only as test input");
  forged_receipt.payload.observed_payload_digest = forged_digest;
  test.expect(runtime::sm87_target_aot_projection_validate_asset(
                  forged_manifest, forged_receipt, sources,
                  inspection.inventory, {payload.data(), payload.size()}) ==
                  Error::kForbiddenNvFp4BlockScale,
              "rehashed forbidden payload scale still fails domain scan");
  payload[scale_address.byte_offset] = original_payload_scale;

  const auto ordinary_scale_address =
      kernels::sm87_target_aot_projection_packed_scale_address(
          layout, 0U, 5U, 5U);
  const std::uint8_t ordinary_original =
      payload[ordinary_scale_address.byte_offset];
  payload[ordinary_scale_address.byte_offset] =
      ordinary_original == 0x38U ? 0x39U : 0x38U;
  forged_manifest = build.manifest;
  forged_receipt = build.transform_receipt;
  test.expect(runtime::sm87_target_aot_projection_sha256(
                  {payload.data(), payload.size()}, &forged_digest),
              "admissible NVFP4 corruption can be rehashed for replay test");
  forged_manifest.payload_digest = forged_digest;
  test.expect(kernels::sm87_target_aot_projection_seal_packed_manifest(
                  &forged_manifest),
              "admissible corrupt manifest reseals for bijection negative test");
  forged_receipt.payload.observed_payload_digest = forged_digest;
  test.expect(runtime::sm87_target_aot_projection_validate_asset(
                  forged_manifest, forged_receipt, sources,
                  inspection.inventory, {payload.data(), payload.size()}) ==
                  Error::kPayloadBijectionMismatch,
              "rehashed admissible scale corruption fails source replay");
  payload[ordinary_scale_address.byte_offset] = ordinary_original;
}

}  // namespace

int main() {
  TestContext test;
  test_sha256(test);
  test_fp8_asset(test);
  test_nvfp4_asset(test);
  if (test.failures() != 0) {
    std::cerr << test.failures() << " assertion(s) failed\n";
    return 1;
  }
  std::cout << "SM87 target-AOT projection real-byte asset checks passed\n";
  return 0;
}
