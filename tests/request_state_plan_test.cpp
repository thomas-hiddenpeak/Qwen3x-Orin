#include "q3x/runtime/request_state.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace {

namespace runtime = q3x::runtime;
namespace model = q3x::model;

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

void add_region(std::vector<std::pair<std::uint64_t, std::uint64_t>>& ranges,
                const runtime::RequestRegion& region) {
    ranges.emplace_back(region.arena_offset,
                        region.arena_offset + region.byte_size);
}

void test_default_exact_plan(TestContext& test) {
    const runtime::RequestPlanResult result =
        runtime::build_request_memory_plan();
    test.expect(result.ok(), "default request memory plan succeeds");
    if (!result) {
        return;
    }
    const runtime::RequestMemoryPlan& plan = *result.value;
    test.expect(plan.batch_size == 1U && plan.max_sequence_length == 128U,
                "default plan is batch-one with 128 positions");
    test.expect(plan.conv_state.byte_size == 2'949'120U &&
                    plan.conv_state.element_capacity == 1'474'560U &&
                    plan.gdn_state.byte_size == 75'497'472U &&
                    plan.gdn_state.element_capacity == 37'748'736U,
                "canonical conv and GDN BF16 state totals are exact");
    test.expect(plan.persistent_offset == 0U &&
                    plan.persistent_bytes == 86'835'200U &&
                    plan.workspace_bytes == 1'163'776U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == runtime::kDefaultRequestArenaBytes &&
                    plan.arena_bytes == 88'031'744U,
                "default persistent/workspace/RoPE/arena bytes are exact");

    std::uint64_t kv_bytes = 0U;
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount; ++slot) {
        test.expect(plan.key_cache[slot].byte_size == 262'144U &&
                        plan.value_cache[slot].byte_size == 262'144U &&
                        plan.key_cache[slot].element_capacity == 131'072U,
                    "each default K/V slot has [128,4,256] BF16 capacity");
        kv_bytes += plan.key_cache[slot].byte_size;
        kv_bytes += plan.value_cache[slot].byte_size;
    }
    test.expect(kv_bytes == 8'388'608U &&
                    kv_bytes == runtime::kRequestKvBytesPerToken * 128U,
                "all 16 K/V pairs total exactly 65536 bytes per token");

    for (const runtime::RequestRegion& hidden : plan.hidden_bf16) {
        test.expect(hidden.element_capacity == 5'120U &&
                        hidden.byte_size == 10'240U,
                    "each hidden workspace has 5120 BF16 elements");
    }
    for (const runtime::RequestRegion& projection : plan.projection_bf16) {
        test.expect(projection.element_capacity == 17'408U &&
                        projection.byte_size == 34'816U,
                    "each projection workspace has 17408 BF16 elements");
    }
    test.expect(plan.linear_a_bf16.element_capacity == 48U &&
                    plan.linear_b_bf16.element_capacity == 48U &&
                    plan.linear_a_bf16.arena_offset !=
                        plan.linear_b_bf16.arena_offset,
                "linear a/b have independent 48-element BF16 buffers");
    test.expect(plan.fp32_scratch.element_capacity == 248'320U &&
                    plan.fp32_scratch.byte_size == 993'280U &&
                    plan.gqa_probability_scratch.element_capacity == 3'072U &&
                    plan.gqa_probability_scratch.byte_size == 12'288U &&
                    plan.gqa_probability_scratch.arena_offset ==
                        plan.fp32_scratch.arena_offset,
                "FP32 logits/GEMV scratch explicitly aliases sufficient GQA scratch");
    test.expect(plan.rope_cos_fp32.element_capacity == 4'096U &&
                    plan.rope_sin_fp32.element_capacity == 4'096U &&
                    plan.rope_cos_fp32.byte_size == 16'384U &&
                    plan.rope_sin_fp32.byte_size == 16'384U,
                "default RoPE cache has [128,32] FP32 storage per table");
}

void test_alignment_non_overlap_and_schedule(TestContext& test) {
    const runtime::RequestPlanResult result =
        runtime::build_request_memory_plan();
    if (!result) {
        test.expect(false, "plan exists for alignment test");
        return;
    }
    const runtime::RequestMemoryPlan& plan = *result.value;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges;
    add_region(ranges, plan.conv_state);
    add_region(ranges, plan.gdn_state);
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount; ++slot) {
        add_region(ranges, plan.key_cache[slot]);
        add_region(ranges, plan.value_cache[slot]);
    }
    for (const auto& region : plan.hidden_bf16) {
        add_region(ranges, region);
    }
    for (const auto& region : plan.projection_bf16) {
        add_region(ranges, region);
    }
    add_region(ranges, plan.linear_a_bf16);
    add_region(ranges, plan.linear_b_bf16);
    add_region(ranges, plan.fp32_scratch);
    add_region(ranges, plan.rope_cos_fp32);
    add_region(ranges, plan.rope_sin_fp32);
    for (const auto& range : ranges) {
        test.expect((range.first % runtime::kRequestArenaAlignment) == 0U &&
                        range.second > range.first &&
                        range.second <= plan.arena_bytes,
                    "every owning region is aligned and inside the arena");
    }
    std::sort(ranges.begin(), ranges.end());
    for (std::size_t index = 1U; index < ranges.size(); ++index) {
        test.expect(ranges[index - 1U].second <= ranges[index].first,
                    "owning regions do not overlap");
    }

    std::size_t linear_count = 0U;
    std::size_t full_count = 0U;
    for (std::size_t layer = 0U; layer < plan.layers.size(); ++layer) {
        const runtime::RequestLayerSlot& slot = plan.layers[layer];
        if (slot.type == model::LayerType::kFullAttention) {
            ++full_count;
            test.expect(((layer + 1U) % 4U) == 0U && slot.slot == layer / 4U,
                        "full-attention layer maps to its checked slot");
        } else {
            ++linear_count;
            test.expect(((layer + 1U) % 4U) != 0U &&
                            slot.slot == layer - layer / 4U,
                        "linear-attention layer maps to its checked slot");
        }
    }
    test.expect(linear_count == 48U && full_count == 16U &&
                    plan.layers[62].slot == 47U &&
                    plan.layers[63].slot == 15U,
                "fixed schedule contains exactly 48 linear and 16 full slots");

    const auto full = runtime::map_request_layer(
        3U, model::LayerType::kFullAttention);
    const auto linear = runtime::map_request_layer(
        62U, model::LayerType::kLinearAttention);
    const auto wrong = runtime::map_request_layer(
        3U, model::LayerType::kLinearAttention);
    const auto out = runtime::map_request_layer(
        64U, model::LayerType::kFullAttention);
    const auto invalid = runtime::map_request_layer(
        0U, model::LayerType::kInvalid);
    test.expect(full && full.value->slot == 0U &&
                    linear && linear.value->slot == 47U,
                "representative valid layer mappings succeed");
    test.expect(!wrong && wrong.error ==
                              runtime::RequestAccessError::kLayerTypeMismatch &&
                    !out && out.error ==
                                runtime::RequestAccessError::kLayerOutOfRange &&
                    !invalid && invalid.error ==
                                    runtime::RequestAccessError::kLayerTypeMismatch,
                "wrong type, out-of-range, and invalid mapping requests fail");
}

void test_minimum_maximum_and_bad_options(TestContext& test) {
    runtime::RequestMemoryOptions options;
    options.max_sequence_length = 1U;
    auto result = runtime::build_request_memory_plan(options);
    test.expect(result && result.value->arena_bytes == 79'676'416U &&
                    result.value->persistent_bytes == 78'512'128U &&
                    result.value->workspace_bytes == 1'163'776U &&
                    result.value->rope_bytes == 512U,
                "single-position plan includes checked RoPE alignment padding");

    options.max_sequence_length = runtime::kAbsoluteRequestMaxSequenceLength;
    options.max_arena_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    result = runtime::build_request_memory_plan(options);
    test.expect(result &&
                    result.value->arena_bytes ==
                        runtime::kMaximumRequestArenaBytes &&
                    result.value->persistent_bytes == 17'258'315'776ULL &&
                    result.value->workspace_bytes == 25'336'320U &&
                    result.value->rope_bytes == 67'108'864U &&
                    result.value->fp32_scratch.element_capacity == 6'291'456U &&
                    result.value->gqa_probability_scratch.element_capacity ==
                        6'291'456U,
                "absolute max plan has exact bounded 17.35GB totals");

    options = {};
    options.batch_size = 2U;
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "batch greater than one is rejected");

    options = {};
    options.max_sequence_length = 0U;
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "zero sequence capacity is rejected");

    options = {};
    options.max_sequence_length =
        runtime::kAbsoluteRequestMaxSequenceLength + 1U;
    options.max_arena_bytes = 32ULL * 1024ULL * 1024ULL * 1024ULL;
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "sequence capacity above 262144 is rejected");

    options.max_sequence_length = std::numeric_limits<std::uint64_t>::max();
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kArithmeticOverflow,
                "malicious sequence arithmetic overflow is detected");

    options = {};
    options.max_arena_bytes = 64U * 1024U * 1024U;
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kArenaLimitExceeded,
                "configured arena cap is enforced by the pure plan");

    options.max_arena_bytes = std::numeric_limits<std::uint64_t>::max();
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "unbounded arena option is rejected");
    test.expect(runtime::to_string(runtime::RequestErrorCode::kCudaFailure) ==
                        "cuda_failure" &&
                    runtime::to_string(
                        runtime::RequestAccessError::kCapacityExceeded) ==
                        "capacity_exceeded",
                "request diagnostic enum names are stable");
}

}  // namespace

int main() {
    TestContext test;
    test_default_exact_plan(test);
    test_alignment_non_overlap_and_schedule(test);
    test_minimum_maximum_and_bad_options(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " request-state plan test(s) failed\n";
        return 1;
    }
    std::cout << "All request-state plan tests passed\n";
    return 0;
}
