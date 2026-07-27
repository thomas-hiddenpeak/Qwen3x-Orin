#include "q3x/core/sha256.h"
#include "q3x/runtime/resident_weights.h"

#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace weights = q3x::model::weights;
namespace st = q3x::io::safetensors;

class TestContext {
  public:
    void expect(bool condition, const char* message) {
        if (!condition) {
            std::cerr << "FAILED: " << message << '\n';
            ++failures_;
        }
    }
    [[nodiscard]] int failures() const noexcept { return failures_; }

  private:
    int failures_ = 0;
};

weights::TensorLocator locator(std::string shard,
                               std::uint64_t begin,
                               std::uint64_t bytes,
                               weights::TensorCategory category) {
    weights::TensorLocator result;
    result.category = category;
    result.shard = std::move(shard);
    result.file_begin = begin;
    result.file_end = begin + bytes;
    result.byte_size = bytes;
    result.dtype = st::DType::kU8;
    result.shape = {bytes};
    return result;
}

weights::WeightManifest make_manifest(const std::string& shard) {
    weights::WeightManifest manifest;
    manifest.tensors.emplace(
        "model.language_model.test_a.weight",
        locator(shard, 5U, 12U, weights::TensorCategory::kText));
    manifest.tensors.emplace(
        "model.visual.skip.weight",
        locator(shard, 17U, 8U, weights::TensorCategory::kVision));
    manifest.tensors.emplace(
        "model.language_model.test_b.weight",
        locator(shard, 25U, 24U, weights::TensorCategory::kText));
    manifest.tensors.emplace(
        "mtp.skip.weight",
        locator(shard, 49U, 6U, weights::TensorCategory::kMtp));
    manifest.tensors.emplace(
        "lm_head.test_c",
        locator(shard, 60U, 4U, weights::TensorCategory::kText));
    manifest.summary.shard_count = 1U;
    manifest.summary.tensor_count = 5U;
    manifest.summary.text_tensor_count = 3U;
    manifest.summary.vision_tensor_count = 1U;
    manifest.summary.mtp_tensor_count = 1U;
    manifest.summary.raw_text_bytes = 40U;
    manifest.summary.vision_bytes = 8U;
    manifest.summary.mtp_bytes = 6U;
    manifest.summary.skipped_bytes = 14U;
    manifest.summary.arena_alignment = runtime::kResidentTensorAlignment;
    manifest.summary.estimated_text_arena_bytes = 768U;
    return manifest;
}

runtime::ShardIdentity identity(std::string filename,
                                std::string_view data) {
    runtime::ShardIdentity result;
    result.filename = std::move(filename);
    result.file_size = data.size();
    result.sha256 = q3x::core::sha256(data).hex();
    return result;
}

void test_valid_plan(TestContext& test) {
    constexpr std::string_view kData =
        "01234567890123456789012345678901234567890123456789012345678901234567890123456789";
    const std::string shard = "tiny.safetensors";
    const weights::WeightManifest manifest = make_manifest(shard);
    const runtime::PlanResult result = runtime::build_resident_load_plan(
        manifest, {identity(shard, kData)});
    test.expect(result.ok(), "valid synthetic resident plan succeeds");
    if (!result) {
        return;
    }
    const runtime::ResidentLoadPlan& plan = *result.value;
    test.expect(plan.arena_bytes == 768U && plan.copied_bytes == 40U &&
                    plan.source_bytes == 80U,
                "plan reports exact arena, copied, and source bytes");
    test.expect(plan.tensors.size() == 3U && plan.shards.size() == 1U &&
                    plan.shards[0].tensor_indices.size() == 3U &&
                    plan.shards[0].copied_bytes == 40U,
                "plan contains only the three text tensors");
    if (plan.tensors.size() == 3U) {
        test.expect(plan.tensors[0].name ==
                        "model.language_model.test_a.weight" &&
                        plan.tensors[0].arena_offset == 0U &&
                        plan.tensors[1].name ==
                            "model.language_model.test_b.weight" &&
                        plan.tensors[1].arena_offset == 256U &&
                        plan.tensors[2].name == "lm_head.test_c" &&
                        plan.tensors[2].arena_offset == 512U,
                    "text tensors are source-ordered and 256-byte aligned");
    }
}

void test_valid_mtp_plan(TestContext& test) {
    constexpr std::string_view kData =
        "01234567890123456789012345678901234567890123456789012345678901234567890123456789";
    const std::string shard = "tiny.safetensors";
    const weights::WeightManifest manifest = make_manifest(shard);
    const runtime::PlanResult result = runtime::build_resident_load_plan(
        manifest, {identity(shard, kData)}, runtime::ResidentPayload::kMtp);
    test.expect(result.ok(), "valid synthetic MTP resident plan succeeds");
    if (!result) {
        return;
    }
    const runtime::ResidentLoadPlan& plan = *result.value;
    test.expect(plan.arena_bytes == 256U && plan.copied_bytes == 6U &&
                    plan.source_bytes == 80U,
                "MTP plan reports exact arena, copied, and source bytes");
    test.expect(plan.tensors.size() == 1U && plan.shards.size() == 1U &&
                    plan.shards[0].tensor_indices.size() == 1U &&
                    plan.shards[0].copied_bytes == 6U,
                "MTP plan contains only the selected tensor");
    if (plan.tensors.size() == 1U) {
        test.expect(plan.tensors[0].name == "mtp.skip.weight" &&
                        plan.tensors[0].arena_offset == 0U,
                    "MTP tensor retains source order and alignment");
    }

    weights::WeightManifest wrong_summary = manifest;
    wrong_summary.summary.mtp_bytes = 7U;
    const runtime::PlanResult failed = runtime::build_resident_load_plan(
        wrong_summary, {identity(shard, kData)},
        runtime::ResidentPayload::kMtp);
    test.expect(!failed &&
                    failed.diagnostic.code ==
                        runtime::ResidentLoadErrorCode::kInvalidManifest,
                "incorrect MTP summary is rejected");
}

void test_identity_and_path_failures(TestContext& test) {
    constexpr std::string_view kData =
        "01234567890123456789012345678901234567890123456789012345678901234567890123456789";
    const runtime::ShardIdentity good = identity("tiny.safetensors", kData);
    const weights::WeightManifest source = make_manifest(good.filename);

    auto result = runtime::build_resident_load_plan(source, {good, good});
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kDuplicateShard,
                "duplicate shard identity is rejected");

    runtime::ShardIdentity bad_path = good;
    bad_path.filename = "../tiny.safetensors";
    result = runtime::build_resident_load_plan(source, {bad_path});
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kUnsafeShardPath,
                "unsafe identity path is rejected");

    result = runtime::build_resident_load_plan(
        source, {identity("other.safetensors", kData)});
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kMissingShardIdentity,
                "manifest shard without an identity is rejected");

    std::vector<runtime::ShardIdentity> extra = {
        good, identity("unused.safetensors", kData)};
    result = runtime::build_resident_load_plan(source, extra);
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kUnexpectedShardIdentity,
                "identity without a manifest shard is rejected");

    weights::WeightManifest unsafe = source;
    unsafe.tensors.begin()->second.shard = "../escape.safetensors";
    result = runtime::build_resident_load_plan(unsafe, {good});
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kUnsafeShardPath,
                "unsafe locator path is rejected before I/O");
}

void test_range_summary_and_overflow_failures(TestContext& test) {
    constexpr std::string_view kData =
        "01234567890123456789012345678901234567890123456789012345678901234567890123456789";
    const runtime::ShardIdentity good = identity("tiny.safetensors", kData);
    const weights::WeightManifest source = make_manifest(good.filename);

    weights::WeightManifest overlap = source;
    auto& skipped = overlap.tensors.at("model.visual.skip.weight");
    skipped.file_begin = 10U;
    skipped.file_end = 18U;
    auto result = runtime::build_resident_load_plan(overlap, {good});
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kInvalidManifest,
                "overlap between skipped and text ranges is rejected");

    weights::WeightManifest wrong_shape = source;
    wrong_shape.tensors.begin()->second.shape = {11U};
    result = runtime::build_resident_load_plan(wrong_shape, {good});
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kInvalidManifest,
                "dtype/shape byte mismatch is rejected");

    weights::WeightManifest wrong_summary = source;
    wrong_summary.summary.estimated_text_arena_bytes = 769U;
    result = runtime::build_resident_load_plan(wrong_summary, {good});
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kInvalidManifest,
                "incorrect manifest arena estimate is rejected");

    weights::WeightManifest overflow;
    weights::TensorLocator wrapped = locator(
        "huge.safetensors", 0U, 8U, weights::TensorCategory::kText);
    wrapped.file_begin = std::numeric_limits<std::uint64_t>::max() - 3U;
    wrapped.file_end = 4U;
    overflow.tensors.emplace("model.language_model.huge.weight",
                             std::move(wrapped));
    runtime::ShardIdentity huge_identity;
    huge_identity.filename = "huge.safetensors";
    huge_identity.file_size = std::numeric_limits<std::uint64_t>::max();
    huge_identity.sha256 = std::string(64U, '0');
    result = runtime::build_resident_load_plan(overflow, {huge_identity});
    test.expect(!result && result.diagnostic.code ==
                               runtime::ResidentLoadErrorCode::kArithmeticOverflow,
                "wrapped source offset is rejected as arithmetic overflow");
}

void test_pinned_identity(TestContext& test) {
    const auto& pinned = runtime::pinned_qwen36_27b_shards();
    test.expect(pinned.size() == 3U &&
                    pinned[0].file_size == 9'965'652'512ULL &&
                    pinned[1].file_size == 9'985'757'032ULL &&
                    pinned[2].file_size == 1'970'287'640ULL,
                "pinned loader exposes exact official shard sizes");
    test.expect(pinned.size() == 3U &&
                    pinned[0].sha256 ==
                        "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d" &&
                    pinned[1].sha256 ==
                        "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d" &&
                    pinned[2].sha256 ==
                        "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845",
                "pinned loader exposes exact official full-file SHA-256");
    test.expect(runtime::kPinnedQwen36_27BArenaBytes == 20'150'786'560ULL &&
                    runtime::kPinnedQwen36_27BMtpArenaBytes ==
                        849'398'784ULL &&
                    runtime::to_string(runtime::ResidentPayload::kText) ==
                        "text" &&
                    runtime::to_string(runtime::ResidentPayload::kMtp) ==
                        "mtp" &&
                    runtime::to_string(
                        runtime::ResidentLoadErrorCode::kSha256Mismatch) ==
                        "sha256_mismatch",
                "pinned arenas, payloads, and diagnostic names are stable");
}

}  // namespace

int main() {
    TestContext test;
    test_valid_plan(test);
    test_valid_mtp_plan(test);
    test_identity_and_path_failures(test);
    test_range_summary_and_overflow_failures(test);
    test_pinned_identity(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " resident load plan test(s) failed\n";
        return 1;
    }
    std::cout << "All resident load plan tests passed\n";
    return 0;
}
