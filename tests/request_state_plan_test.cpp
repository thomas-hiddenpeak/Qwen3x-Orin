#include "q3x/runtime/request_state.h"

#include "q3x/runtime/prefill_workspace_plan.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
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

bool same_region(const runtime::RequestRegion& left,
                 const runtime::RequestRegion& right) {
    return left.arena_offset == right.arena_offset &&
           left.byte_size == right.byte_size &&
           left.element_capacity == right.element_capacity &&
           left.element_size_bytes == right.element_size_bytes;
}

bool matrix_is(const runtime::RequestMatrixRegion& matrix,
               const std::uint64_t offset,
               const std::uint64_t byte_size,
               const std::uint32_t rows,
               const std::uint32_t columns,
               const std::uint64_t stride,
               const std::uint32_t element_size) {
    return matrix.storage.arena_offset == offset &&
           matrix.storage.byte_size == byte_size &&
           matrix.storage.element_capacity ==
               static_cast<std::uint64_t>(rows) * stride &&
           matrix.storage.element_size_bytes == element_size &&
           matrix.row_capacity == rows && matrix.columns == columns &&
           matrix.row_stride_elements == stride;
}

bool region_is_empty(const runtime::RequestRegion& region) {
    return region.arena_offset == 0U && region.byte_size == 0U &&
           region.element_capacity == 0U &&
           region.element_size_bytes == 0U;
}

bool region_is(const runtime::RequestRegion& region,
               const std::uint64_t offset, const std::uint64_t byte_size,
               const std::uint32_t element_size) {
    return region.arena_offset == offset && region.byte_size == byte_size &&
           region.element_capacity == byte_size / element_size &&
           region.element_size_bytes == element_size;
}

bool matrix_is_empty(const runtime::RequestMatrixRegion& matrix) {
    return region_is_empty(matrix.storage) && matrix.row_capacity == 0U &&
           matrix.columns == 0U && matrix.row_stride_elements == 0U;
}

void test_default_exact_plan(TestContext& test) {
    const runtime::RequestMemoryOptions default_options;
    test.expect(default_options.max_arena_bytes ==
                        2ULL * 1024ULL * 1024ULL * 1024ULL &&
                    default_options.min_free_bytes_after_create ==
                        8ULL * 1024ULL * 1024ULL * 1024ULL,
                "default arena cap and post-create free margin remain fixed");
    const runtime::RequestPlanResult result =
        runtime::build_request_memory_plan();
    test.expect(result.ok(), "default request memory plan succeeds");
    if (!result) {
        return;
    }
    const runtime::RequestMemoryPlan& plan = *result.value;
    test.expect(plan.profile == runtime::RequestMemoryProfile::kLegacyC512 &&
                    plan.batch_size == 1U && plan.prefill_chunk_size == 1U &&
                    plan.max_sequence_length == 128U,
                "default plan remains the batch-one legacy profile");
    test.expect(plan.conv_state.byte_size == 2'949'120U &&
                    plan.conv_state.element_capacity == 1'474'560U &&
                    plan.gdn_state.byte_size == 75'497'472U &&
                    plan.gdn_state.element_capacity == 37'748'736U,
                "canonical conv and GDN BF16 state totals are exact");
    test.expect(plan.persistent_offset == 0U &&
                    plan.persistent_bytes == 86'835'200U &&
                    plan.workspace_offset == 86'835'200U &&
                    plan.workspace_bytes == 1'219'072U &&
                    plan.rope_offset == 88'054'272U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == runtime::kDefaultRequestArenaBytes &&
                    plan.arena_bytes == 88'087'040U,
                "default persistent/workspace/RoPE/arena bytes are exact");
    test.expect(plan.conv_state.arena_offset == 0U &&
                    plan.gdn_state.arena_offset == 2'949'120U,
                "default persistent aggregate offsets remain byte-exact");

    std::uint64_t kv_bytes = 0U;
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount; ++slot) {
        test.expect(plan.key_cache[slot].byte_size == 262'144U &&
                        plan.value_cache[slot].byte_size == 262'144U &&
                        plan.key_cache[slot].element_capacity == 131'072U &&
                        plan.key_cache[slot].arena_offset ==
                            78'446'592U + slot * 524'288U &&
                        plan.value_cache[slot].arena_offset ==
                            78'708'736U + slot * 524'288U,
                    "each default K/V slot has [128,4,256] BF16 capacity");
        kv_bytes += plan.key_cache[slot].byte_size;
        kv_bytes += plan.value_cache[slot].byte_size;
    }
    test.expect(kv_bytes == 8'388'608U &&
                    kv_bytes == runtime::kRequestKvBytesPerToken * 128U,
                "all 16 K/V pairs total exactly 65536 bytes per token");

    for (std::size_t index = 0U; index < plan.hidden_bf16.size(); ++index) {
        const runtime::RequestRegion& hidden = plan.hidden_bf16[index];
        test.expect(hidden.element_capacity == 5'120U &&
                        hidden.byte_size == 10'240U &&
                        hidden.arena_offset == 86'835'200U + index * 10'240U,
                    "each hidden workspace has 5120 BF16 elements");
    }
    for (std::size_t index = 0U; index < plan.projection_bf16.size(); ++index) {
        const runtime::RequestRegion& projection = plan.projection_bf16[index];
        test.expect(projection.element_capacity == 17'408U &&
                        projection.byte_size == 34'816U &&
                        projection.arena_offset ==
                            86'865'920U + index * 34'816U,
                    "each projection workspace has 17408 BF16 elements");
    }
    test.expect(plan.linear_a_bf16.element_capacity == 48U &&
                    plan.linear_b_bf16.element_capacity == 48U &&
                    plan.linear_a_bf16.arena_offset == 87'005'184U &&
                    plan.linear_b_bf16.arena_offset == 87'005'440U,
                "linear a/b have independent 48-element BF16 buffers");
    test.expect(plan.fp32_scratch.element_capacity == 262'144U &&
                    plan.fp32_scratch.byte_size == 1'048'576U &&
                    plan.fp32_scratch.arena_offset == 87'005'696U &&
                    plan.gqa_probability_scratch.element_capacity == 3'072U &&
                    plan.gqa_probability_scratch.byte_size == 12'288U &&
                    plan.gqa_probability_scratch.arena_offset ==
                        plan.fp32_scratch.arena_offset,
                "FP32 logits/GEMV scratch explicitly aliases sufficient GQA scratch");
    test.expect(plan.rope_cos_fp32.element_capacity == 4'096U &&
                    plan.rope_sin_fp32.element_capacity == 4'096U &&
                    plan.rope_cos_fp32.byte_size == 16'384U &&
                    plan.rope_sin_fp32.byte_size == 16'384U &&
                    plan.rope_cos_fp32.arena_offset == 88'054'272U &&
                    plan.rope_sin_fp32.arena_offset == 88'070'656U,
                "default RoPE cache has [128,32] FP32 storage per table");
}

void test_prefill_chunk_layout(TestContext& test) {
    const runtime::RequestPlanResult chunk_one =
        runtime::build_request_memory_plan();
    runtime::RequestMemoryOptions options;
    options.prefill_chunk_size = 8U;
    const runtime::RequestPlanResult chunk_eight =
        runtime::build_request_memory_plan(options);
    test.expect(chunk_one && chunk_eight,
                "chunk-one and chunk-eight plans both succeed");
    if (!chunk_one || !chunk_eight) {
        return;
    }

    const runtime::RequestMemoryPlan& baseline = *chunk_one.value;
    const runtime::RequestMemoryPlan& plan = *chunk_eight.value;
    test.expect(plan.prefill_chunk_size == 8U &&
                    plan.persistent_offset == 0U &&
                    plan.persistent_bytes == 86'835'200U &&
                    plan.workspace_offset == 86'835'200U &&
                    plan.workspace_bytes == 2'409'984U &&
                    plan.rope_offset == 89'245'184U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == 89'277'952U &&
                    plan.arena_bytes - baseline.arena_bytes == 1'190'912U,
                "chunk-eight workspace and arena totals are byte-exact");

    test.expect(same_region(plan.conv_state, baseline.conv_state) &&
                    same_region(plan.gdn_state, baseline.gdn_state),
                "chunk-eight leaves aggregate persistent state unchanged");
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount; ++slot) {
        test.expect(same_region(plan.key_cache[slot],
                                baseline.key_cache[slot]) &&
                        same_region(plan.value_cache[slot],
                                    baseline.value_cache[slot]),
                    "chunk-eight leaves every K/V cache region unchanged");
    }

    constexpr std::uint64_t hidden_offsets[] = {
        86'835'200U, 86'917'120U, 86'999'040U};
    for (std::size_t index = 0U; index < plan.hidden_bf16.size(); ++index) {
        const runtime::RequestRegion& region = plan.hidden_bf16[index];
        test.expect(region.arena_offset == hidden_offsets[index] &&
                        region.element_capacity == 40'960U &&
                        region.byte_size == 81'920U,
                    "chunk-eight hidden region layout is byte-exact");
    }
    constexpr std::uint64_t projection_offsets[] = {
        87'080'960U, 87'359'488U, 87'638'016U, 87'916'544U};
    for (std::size_t index = 0U; index < plan.projection_bf16.size(); ++index) {
        const runtime::RequestRegion& region = plan.projection_bf16[index];
        test.expect(region.arena_offset == projection_offsets[index] &&
                        region.element_capacity == 139'264U &&
                        region.byte_size == 278'528U,
                    "chunk-eight projection region layout is byte-exact");
    }
    test.expect(plan.linear_a_bf16.arena_offset == 88'195'072U &&
                    plan.linear_b_bf16.arena_offset == 88'195'840U &&
                    plan.linear_a_bf16.element_capacity == 384U &&
                    plan.linear_b_bf16.element_capacity == 384U &&
                    plan.linear_a_bf16.byte_size == 768U &&
                    plan.linear_b_bf16.byte_size == 768U,
                "chunk-eight linear a/b region layout is byte-exact");

    test.expect(plan.fp32_scratch.arena_offset == 88'196'608U &&
                    plan.fp32_scratch.byte_size ==
                        baseline.fp32_scratch.byte_size &&
                    plan.fp32_scratch.element_capacity ==
                        baseline.fp32_scratch.element_capacity &&
                    plan.gqa_probability_scratch.arena_offset ==
                        plan.fp32_scratch.arena_offset &&
                    plan.gqa_probability_scratch.byte_size ==
                        baseline.gqa_probability_scratch.byte_size &&
                    plan.gqa_probability_scratch.element_capacity ==
                        baseline.gqa_probability_scratch.element_capacity,
                "chunk-eight leaves FP32 and GQA scratch capacities unchanged");
    test.expect(plan.rope_cos_fp32.arena_offset == 89'245'184U &&
                    plan.rope_sin_fp32.arena_offset == 89'261'568U &&
                    plan.rope_cos_fp32.byte_size ==
                        baseline.rope_cos_fp32.byte_size &&
                    plan.rope_sin_fp32.byte_size ==
                        baseline.rope_sin_fp32.byte_size &&
                    plan.rope_cos_fp32.element_capacity ==
                        baseline.rope_cos_fp32.element_capacity &&
                    plan.rope_sin_fp32.element_capacity ==
                        baseline.rope_sin_fp32.element_capacity,
                "chunk-eight leaves RoPE capacities unchanged at exact offsets");
}

void test_prefill_chunk_sixteen_layout(TestContext& test) {
    const runtime::RequestPlanResult chunk_one =
        runtime::build_request_memory_plan();
    runtime::RequestMemoryOptions options;
    options.prefill_chunk_size = 16U;
    const runtime::RequestPlanResult chunk_sixteen =
        runtime::build_request_memory_plan(options);
    test.expect(chunk_one && chunk_sixteen,
                "chunk-one and chunk-sixteen plans both succeed");
    if (!chunk_one || !chunk_sixteen) {
        return;
    }

    const runtime::RequestMemoryPlan& baseline = *chunk_one.value;
    const runtime::RequestMemoryPlan& plan = *chunk_sixteen.value;
    test.expect(plan.prefill_chunk_size == 16U &&
                    plan.persistent_offset == 0U &&
                    plan.persistent_bytes == 86'835'200U &&
                    plan.workspace_offset == 86'835'200U &&
                    plan.workspace_bytes == 3'771'392U &&
                    plan.rope_offset == 90'606'592U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == 90'639'360U &&
                    plan.arena_bytes - baseline.arena_bytes == 2'552'320U,
                "chunk-sixteen workspace and arena totals are byte-exact");

    test.expect(same_region(plan.conv_state, baseline.conv_state) &&
                    same_region(plan.gdn_state, baseline.gdn_state),
                "chunk-sixteen leaves aggregate persistent state unchanged");
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
         ++slot) {
        test.expect(same_region(plan.key_cache[slot],
                                baseline.key_cache[slot]) &&
                        same_region(plan.value_cache[slot],
                                    baseline.value_cache[slot]),
                    "chunk-sixteen leaves every K/V cache region unchanged");
    }

    constexpr std::uint64_t hidden_offsets[] = {
        86'835'200U, 86'999'040U, 87'162'880U};
    for (std::size_t index = 0U; index < plan.hidden_bf16.size(); ++index) {
        const runtime::RequestRegion& region = plan.hidden_bf16[index];
        test.expect(region.arena_offset == hidden_offsets[index] &&
                        region.element_capacity == 81'920U &&
                        region.byte_size == 163'840U,
                    "chunk-sixteen hidden region layout is byte-exact");
    }
    constexpr std::uint64_t projection_offsets[] = {
        87'326'720U, 87'883'776U, 88'440'832U, 88'997'888U};
    for (std::size_t index = 0U; index < plan.projection_bf16.size();
         ++index) {
        const runtime::RequestRegion& region = plan.projection_bf16[index];
        test.expect(region.arena_offset == projection_offsets[index] &&
                        region.element_capacity == 278'528U &&
                        region.byte_size == 557'056U,
                    "chunk-sixteen projection region layout is byte-exact");
    }
    test.expect(plan.linear_a_bf16.arena_offset == 89'554'944U &&
                    plan.linear_b_bf16.arena_offset == 89'556'480U &&
                    plan.linear_a_bf16.element_capacity == 768U &&
                    plan.linear_b_bf16.element_capacity == 768U &&
                    plan.linear_a_bf16.byte_size == 1'536U &&
                    plan.linear_b_bf16.byte_size == 1'536U,
                "chunk-sixteen linear a/b region layout is byte-exact");

    test.expect(plan.fp32_scratch.arena_offset == 89'558'016U &&
                    plan.fp32_scratch.byte_size ==
                        baseline.fp32_scratch.byte_size &&
                    plan.fp32_scratch.element_capacity ==
                        baseline.fp32_scratch.element_capacity &&
                    plan.gqa_probability_scratch.arena_offset ==
                        plan.fp32_scratch.arena_offset &&
                    plan.gqa_probability_scratch.byte_size ==
                        baseline.gqa_probability_scratch.byte_size &&
                    plan.gqa_probability_scratch.element_capacity ==
                        baseline.gqa_probability_scratch.element_capacity,
                "chunk-sixteen leaves FP32 and GQA scratch capacities unchanged");
    test.expect(plan.rope_cos_fp32.arena_offset == 90'606'592U &&
                    plan.rope_sin_fp32.arena_offset == 90'622'976U &&
                    plan.rope_cos_fp32.byte_size ==
                        baseline.rope_cos_fp32.byte_size &&
                    plan.rope_sin_fp32.byte_size ==
                        baseline.rope_sin_fp32.byte_size &&
                    plan.rope_cos_fp32.element_capacity ==
                        baseline.rope_cos_fp32.element_capacity &&
                    plan.rope_sin_fp32.element_capacity ==
                        baseline.rope_sin_fp32.element_capacity,
                "chunk-sixteen leaves RoPE capacities unchanged at exact offsets");
}

void test_prefill_chunk_thirty_two_layout(TestContext& test) {
    const runtime::RequestPlanResult chunk_one =
        runtime::build_request_memory_plan();
    runtime::RequestMemoryOptions options;
    options.prefill_chunk_size = 32U;
    const runtime::RequestPlanResult chunk_thirty_two =
        runtime::build_request_memory_plan(options);
    test.expect(chunk_one && chunk_thirty_two,
                "chunk-one and chunk-thirty-two plans both succeed");
    if (!chunk_one || !chunk_thirty_two) {
        return;
    }

    const runtime::RequestMemoryPlan& baseline = *chunk_one.value;
    const runtime::RequestMemoryPlan& plan = *chunk_thirty_two.value;
    test.expect(plan.prefill_chunk_size == 32U &&
                    plan.persistent_offset == 0U &&
                    plan.persistent_bytes == 86'835'200U &&
                    plan.workspace_offset == 86'835'200U &&
                    plan.workspace_bytes == 6'494'208U &&
                    plan.rope_offset == 93'329'408U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == 93'362'176U &&
                    plan.arena_bytes - baseline.arena_bytes == 5'275'136U,
                "chunk-thirty-two workspace and arena totals are byte-exact");

    test.expect(same_region(plan.conv_state, baseline.conv_state) &&
                    same_region(plan.gdn_state, baseline.gdn_state),
                "chunk-thirty-two leaves aggregate persistent state unchanged");
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
         ++slot) {
        test.expect(same_region(plan.key_cache[slot],
                                baseline.key_cache[slot]) &&
                        same_region(plan.value_cache[slot],
                                    baseline.value_cache[slot]),
                    "chunk-thirty-two leaves every K/V cache region unchanged");
    }

    constexpr std::uint64_t hidden_offsets[] = {
        86'835'200U, 87'162'880U, 87'490'560U};
    for (std::size_t index = 0U; index < plan.hidden_bf16.size(); ++index) {
        const runtime::RequestRegion& region = plan.hidden_bf16[index];
        test.expect(region.arena_offset == hidden_offsets[index] &&
                        region.element_capacity == 163'840U &&
                        region.byte_size == 327'680U,
                    "chunk-thirty-two hidden region layout is byte-exact");
    }
    constexpr std::uint64_t projection_offsets[] = {
        87'818'240U, 88'932'352U, 90'046'464U, 91'160'576U};
    for (std::size_t index = 0U; index < plan.projection_bf16.size();
         ++index) {
        const runtime::RequestRegion& region = plan.projection_bf16[index];
        test.expect(region.arena_offset == projection_offsets[index] &&
                        region.element_capacity == 557'056U &&
                        region.byte_size == 1'114'112U,
                    "chunk-thirty-two projection region layout is byte-exact");
    }
    test.expect(plan.linear_a_bf16.arena_offset == 92'274'688U &&
                    plan.linear_b_bf16.arena_offset == 92'277'760U &&
                    plan.linear_a_bf16.element_capacity == 1'536U &&
                    plan.linear_b_bf16.element_capacity == 1'536U &&
                    plan.linear_a_bf16.byte_size == 3'072U &&
                    plan.linear_b_bf16.byte_size == 3'072U,
                "chunk-thirty-two linear a/b region layout is byte-exact");

    test.expect(plan.fp32_scratch.arena_offset == 92'280'832U &&
                    plan.fp32_scratch.byte_size ==
                        baseline.fp32_scratch.byte_size &&
                    plan.fp32_scratch.element_capacity ==
                        baseline.fp32_scratch.element_capacity &&
                    plan.gqa_probability_scratch.arena_offset ==
                        plan.fp32_scratch.arena_offset &&
                    plan.gqa_probability_scratch.byte_size ==
                        baseline.gqa_probability_scratch.byte_size &&
                    plan.gqa_probability_scratch.element_capacity ==
                        baseline.gqa_probability_scratch.element_capacity,
                "chunk-thirty-two leaves FP32 and GQA scratch capacities unchanged");
    test.expect(plan.rope_cos_fp32.arena_offset == 93'329'408U &&
                    plan.rope_sin_fp32.arena_offset == 93'345'792U &&
                    plan.rope_cos_fp32.byte_size ==
                        baseline.rope_cos_fp32.byte_size &&
                    plan.rope_sin_fp32.byte_size ==
                        baseline.rope_sin_fp32.byte_size &&
                    plan.rope_cos_fp32.element_capacity ==
                        baseline.rope_cos_fp32.element_capacity &&
                    plan.rope_sin_fp32.element_capacity ==
                        baseline.rope_sin_fp32.element_capacity,
                "chunk-thirty-two leaves RoPE capacities unchanged at exact offsets");
}

void test_prefill_chunk_sixty_four_layout(TestContext& test) {
    const runtime::RequestPlanResult chunk_one =
        runtime::build_request_memory_plan();
    runtime::RequestMemoryOptions options;
    options.prefill_chunk_size = 64U;
    const runtime::RequestPlanResult chunk_sixty_four =
        runtime::build_request_memory_plan(options);
    test.expect(chunk_one && chunk_sixty_four,
                "chunk-one and chunk-sixty-four plans both succeed");
    if (!chunk_one || !chunk_sixty_four) {
        return;
    }

    const runtime::RequestMemoryPlan& baseline = *chunk_one.value;
    const runtime::RequestMemoryPlan& plan = *chunk_sixty_four.value;
    test.expect(plan.prefill_chunk_size == 64U &&
                    plan.persistent_offset == 0U &&
                    plan.persistent_bytes == 86'835'200U &&
                    plan.workspace_offset == 86'835'200U &&
                    plan.workspace_bytes == 11'939'840U &&
                    plan.rope_offset == 98'775'040U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == 98'807'808U &&
                    plan.arena_bytes - baseline.arena_bytes == 10'720'768U,
                "chunk-sixty-four workspace and arena totals are byte-exact");

    constexpr std::uint64_t hidden_offsets[] = {
        86'835'200U, 87'490'560U, 88'145'920U};
    for (std::size_t index = 0U; index < plan.hidden_bf16.size(); ++index) {
        const runtime::RequestRegion& region = plan.hidden_bf16[index];
        test.expect(region.arena_offset == hidden_offsets[index] &&
                        region.element_capacity == 327'680U &&
                        region.byte_size == 655'360U,
                    "chunk-sixty-four hidden region layout is byte-exact");
    }

    constexpr std::uint64_t projection_offsets[] = {
        88'801'280U, 91'029'504U, 93'257'728U, 95'485'952U};
    for (std::size_t index = 0U; index < plan.projection_bf16.size();
         ++index) {
        const runtime::RequestRegion& region = plan.projection_bf16[index];
        test.expect(region.arena_offset == projection_offsets[index] &&
                        region.element_capacity == 1'114'112U &&
                        region.byte_size == 2'228'224U,
                    "chunk-sixty-four projection region layout is byte-exact");
    }
    test.expect(plan.linear_a_bf16.arena_offset == 97'714'176U &&
                    plan.linear_b_bf16.arena_offset == 97'720'320U &&
                    plan.linear_a_bf16.element_capacity == 3'072U &&
                    plan.linear_b_bf16.element_capacity == 3'072U &&
                    plan.linear_a_bf16.byte_size == 6'144U &&
                    plan.linear_b_bf16.byte_size == 6'144U,
                "chunk-sixty-four linear a/b layout is byte-exact");
    test.expect(plan.fp32_scratch.arena_offset == 97'726'464U &&
                    plan.fp32_scratch.byte_size ==
                        baseline.fp32_scratch.byte_size &&
                    plan.fp32_scratch.element_capacity ==
                        baseline.fp32_scratch.element_capacity &&
                    plan.rope_cos_fp32.arena_offset == 98'775'040U &&
                    plan.rope_sin_fp32.arena_offset == 98'791'424U,
                "chunk-sixty-four leaves FP32 capacity fixed and advances RoPE exactly");
}

void test_prefill_chunk_large_layout(TestContext& test) {
    runtime::RequestMemoryOptions options;
    options.prefill_chunk_size = 256U;
    const runtime::RequestPlanResult c256 =
        runtime::build_request_memory_plan(options);
    options.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
    const runtime::RequestPlanResult c512 =
        runtime::build_request_memory_plan(options);
    test.expect(c256 && c512 &&
                    runtime::kMaximumRequestPrefillChunkSize == 512U,
                "C256 canary and public C512 request plans both build");
    if (!c256 || !c512) {
        return;
    }

    const runtime::RequestMemoryPlan& plan256 = *c256.value;
    const runtime::RequestMemoryPlan& plan512 = *c512.value;
    test.expect(plan256.prefill_chunk_size == 256U &&
                    plan256.persistent_bytes == 86'835'200U &&
                    plan256.workspace_offset == 86'835'200U &&
                    plan256.workspace_bytes == 44'613'632U &&
                    plan256.rope_offset == 131'448'832U &&
                    plan256.rope_bytes == 32'768U &&
                    plan256.arena_bytes == 131'481'600U,
                "C256 canary workspace and arena totals are byte-exact");
    test.expect(plan512.prefill_chunk_size == 512U &&
                    plan512.persistent_bytes == 86'835'200U &&
                    plan512.workspace_offset == 86'835'200U &&
                    plan512.workspace_bytes == 88'178'688U &&
                    plan512.rope_offset == 175'013'888U &&
                    plan512.rope_bytes == 32'768U &&
                    plan512.arena_bytes == 175'046'656U,
                "public C512 workspace and arena totals are byte-exact");

    test.expect(plan256.hidden_bf16[0U].element_capacity == 1'310'720U &&
                    plan256.projection_bf16[0U].element_capacity ==
                        4'456'448U &&
                    plan512.hidden_bf16[0U].element_capacity == 2'621'440U &&
                    plan512.projection_bf16[0U].element_capacity ==
                        8'912'896U &&
                    plan512.linear_a_bf16.element_capacity == 24'576U &&
                    plan512.linear_b_bf16.element_capacity == 24'576U,
                "C256/C512 row capacities scale exactly with the ABI tile");
}

void test_native_only_c512_layout(TestContext& test) {
    runtime::RequestMemoryOptions options;
    options.prefill_chunk_size = 512U;
    options.max_sequence_length = 512U;
    options.max_arena_bytes = 200'310'784U;
    const runtime::RequestPlanResult c512 =
        runtime::build_request_memory_plan(options);

    options.max_sequence_length = 1'024U;
    options.max_arena_bytes = 233'996'288U;
    const runtime::RequestPlanResult c512_1024 =
        runtime::build_request_memory_plan(options);

    test.expect(c512 && c512_1024,
                "native-only C512 plans build without external-library scratch");
    if (!c512 || !c512_1024) {
        return;
    }

    const runtime::RequestMemoryPlan& plan = *c512.value;
    const runtime::RequestMemoryPlan& plan1024 = *c512_1024.value;
    test.expect(plan.persistent_bytes == 112'001'024U &&
                    plan.workspace_offset == 112'001'024U &&
                    plan.workspace_bytes == 88'178'688U &&
                    plan.rope_offset == 200'179'712U &&
                    plan.rope_bytes == 131'072U &&
                    plan.arena_bytes == 200'310'784U,
                "C512 maxseq-512 has the exact self-hosted arena layout");
    test.expect(plan1024.persistent_bytes == 145'555'456U &&
                    plan1024.workspace_offset == 145'555'456U &&
                    plan1024.workspace_bytes == 88'178'688U &&
                    plan1024.rope_offset == 233'734'144U &&
                    plan1024.rope_bytes == 262'144U &&
                    plan1024.arena_bytes == 233'996'288U,
                "C512 maxseq-1024 keeps no canonical BF16 weight scratch");

    options.max_sequence_length = 512U;
    options.max_arena_bytes = 200'310'783U;
    const runtime::RequestPlanResult one_byte_short =
        runtime::build_request_memory_plan(options);
    test.expect(!one_byte_short &&
                    one_byte_short.diagnostic.code ==
                        runtime::RequestErrorCode::kArenaLimitExceeded,
                "native-only C512 arena bound remains fail-closed");
}

void test_target_length_capacity_plans(TestContext& test) {
    struct TargetCapacityCase {
        std::uint32_t sequence_length;
        std::uint64_t persistent_bytes;
        std::uint64_t workspace_bytes;
        std::uint64_t rope_bytes;
        std::uint64_t arena_bytes;
    };
    constexpr TargetCapacityCase cases[] = {
        {40'000U, 2'699'886'592ULL, 90'970'112U, 10'240'000U,
         2'801'096'704ULL},
        {60'000U, 4'010'606'592ULL, 92'890'112U, 15'360'000U,
         4'118'856'704ULL},
        {130'000U, 8'598'126'592ULL, 99'610'112U, 33'280'000U,
         8'731'016'704ULL},
        {131'072U, 8'668'381'184ULL, 99'713'024U, 33'554'432U,
         8'801'648'640ULL},
    };

    runtime::RequestMemoryOptions default_cap;
    default_cap.prefill_chunk_size =
        runtime::kMaximumRequestPrefillChunkSize;
    default_cap.max_sequence_length = cases[0U].sequence_length;
    const runtime::RequestPlanResult default_cap_result =
        runtime::build_request_memory_plan(default_cap);
    test.expect(!default_cap_result &&
                    default_cap_result.diagnostic.code ==
                        runtime::RequestErrorCode::kArenaLimitExceeded,
                "the default 2-GiB arena cap fails closed for 40K C512");

    for (const TargetCapacityCase& target : cases) {
        runtime::RequestMemoryOptions options;
        options.prefill_chunk_size =
            runtime::kMaximumRequestPrefillChunkSize;
        options.max_sequence_length = target.sequence_length;
        options.max_arena_bytes = target.arena_bytes;
        const runtime::RequestPlanResult exact =
            runtime::build_request_memory_plan(options);
        test.expect(static_cast<bool>(exact),
                    "target-length C512 plan fits its byte-exact arena cap");
        if (!exact) {
            continue;
        }

        const runtime::RequestMemoryPlan& plan = *exact.value;
        const std::uint64_t kv_elements =
            static_cast<std::uint64_t>(target.sequence_length) * 4U * 256U;
        const std::uint64_t probability_elements =
            static_cast<std::uint64_t>(target.sequence_length) * 24U;
        const std::uint64_t rope_elements =
            static_cast<std::uint64_t>(target.sequence_length) * 32U;
        test.expect(plan.max_sequence_length == target.sequence_length &&
                        plan.prefill_chunk_size ==
                            runtime::kMaximumRequestPrefillChunkSize &&
                        plan.persistent_bytes == target.persistent_bytes &&
                        plan.workspace_bytes == target.workspace_bytes &&
                        plan.rope_bytes == target.rope_bytes &&
                        plan.arena_bytes == target.arena_bytes,
                    "target-length C512 aggregate capacities are byte-exact");
        test.expect(plan.key_cache.front().element_capacity == kv_elements &&
                        plan.key_cache.back().element_capacity == kv_elements &&
                        plan.value_cache.front().element_capacity ==
                            kv_elements &&
                        plan.value_cache.back().element_capacity ==
                            kv_elements &&
                        plan.fp32_scratch.element_capacity ==
                            probability_elements &&
                        plan.gqa_probability_scratch.element_capacity ==
                            probability_elements &&
                        plan.rope_cos_fp32.element_capacity == rope_elements &&
                        plan.rope_sin_fp32.element_capacity == rope_elements,
                    "target-length KV, attention scratch, and RoPE capacities "
                    "consume the full prompt");

        options.max_arena_bytes = target.arena_bytes - 1U;
        const runtime::RequestPlanResult one_byte_short =
            runtime::build_request_memory_plan(options);
        test.expect(!one_byte_short &&
                        one_byte_short.diagnostic.code ==
                            runtime::RequestErrorCode::kArenaLimitExceeded &&
                        one_byte_short.diagnostic.actual ==
                            std::to_string(target.arena_bytes),
                    "target-length planner rejects an arena cap one byte short");
    }
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
    test.expect(result && result.value->arena_bytes == 79'731'712U &&
                    result.value->persistent_bytes == 78'512'128U &&
                    result.value->workspace_bytes == 1'219'072U &&
                    result.value->rope_bytes == 512U,
                "single-position plan includes checked RoPE alignment padding");

    options.max_sequence_length = runtime::kAbsoluteRequestMaxSequenceLength;
    options.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
    options.max_arena_bytes = runtime::kMaximumRequestArenaBytes;
    result = runtime::build_request_memory_plan(options);
    test.expect(result &&
                    result.value->arena_bytes ==
                        runtime::kMaximumRequestArenaBytes &&
                    result.value->persistent_bytes == 17'258'315'776ULL &&
                    result.value->workspace_bytes == 112'295'936U &&
                    result.value->rope_bytes == 67'108'864U &&
                    result.value->fp32_scratch.element_capacity == 6'291'456U &&
                    result.value->gqa_probability_scratch.element_capacity ==
                        6'291'456U,
                "absolute sequence and chunk max plan has exact bounded totals");
    options.max_arena_bytes = runtime::kMaximumRequestArenaBytes - 1U;
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kArenaLimitExceeded,
                "one byte below the exact absolute arena bound is rejected");

    options = {};
    options.batch_size = 2U;
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "batch greater than one is rejected");

    options = {};
    options.prefill_chunk_size = 0U;
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "zero prefill chunk size is rejected");

    options = {};
    options.prefill_chunk_size =
        runtime::kMaximumRequestPrefillChunkSize + 1U;
    result = runtime::build_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "prefill chunk size above five hundred twelve is rejected");

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

void test_layer_major_target_profiles(TestContext& test) {
    struct TargetCase {
        std::uint32_t sequence_length;
        std::uint64_t persistent_offset_end;
        std::uint64_t family_arena_offset;
        std::uint64_t legacy_offset;
        std::uint64_t final_hidden_offset;
        std::uint64_t rope_offset;
        std::uint64_t arena_bytes;
    };
    constexpr TargetCase cases[] = {
        {40'000U, 2'699'886'592ULL, 3'109'486'592ULL,
         3'965'124'608ULL, 4'056'094'720ULL, 4'056'104'960ULL,
         4'066'344'960ULL},
        {60'000U, 4'010'606'592ULL, 4'625'006'592ULL,
         5'480'644'608ULL, 5'573'534'720ULL, 5'573'544'960ULL,
         5'588'904'960ULL},
        {130'000U, 8'598'126'592ULL, 9'929'326'592ULL,
         10'784'964'608ULL, 10'884'574'720ULL, 10'884'584'960ULL,
         10'917'864'960ULL},
    };

    for (const TargetCase& target : cases) {
        runtime::LayerMajorRequestMemoryOptions options;
        options.max_sequence_length = target.sequence_length;
        options.max_arena_bytes = target.arena_bytes;
        const runtime::LayerMajorRequestPlanResult result =
            runtime::build_layer_major_request_memory_plan(options);
        test.expect(static_cast<bool>(result),
                    "layer-major target bucket fits its exact arena cap");
        if (!result) {
            continue;
        }
        const runtime::LayerMajorRequestMemoryPlan& plan = *result.value;
        const runtime::RequestMemoryPlan& common = plan.common;
        test.expect(
            common.profile == runtime::RequestMemoryProfile::kLayerMajorC8192 &&
                common.batch_size == 1U &&
                common.prefill_chunk_size == 512U &&
                common.max_sequence_length == target.sequence_length &&
                plan.operator_panel_capacity_tokens == 8'192U &&
                plan.legacy_prefill_chunk_size == 512U,
            "layer-major plan identity cannot alter the public legacy C512 ABI");
        test.expect(
            common.persistent_offset == 0U &&
                common.persistent_bytes == target.persistent_offset_end &&
                common.workspace_offset == target.persistent_offset_end &&
                plan.prompt_residual_bf16.storage.arena_offset ==
                    target.persistent_offset_end &&
                plan.c8192_family_phase_arena.arena_offset ==
                    target.family_arena_offset &&
                plan.c8192_family_phase_arena.byte_size == 855'638'016U &&
                plan.legacy_c512.hidden_bf16[0U].storage.arena_offset ==
                    target.legacy_offset &&
                plan.final_hidden_bf16.storage.arena_offset ==
                    target.final_hidden_offset &&
                common.rope_offset == target.rope_offset &&
                common.arena_bytes == target.arena_bytes &&
                common.workspace_bytes ==
                    target.rope_offset - target.persistent_offset_end,
            "layer-major target bucket owns the byte-exact physical layout");
        test.expect(
            matrix_is(plan.prompt_residual_bf16,
                      target.persistent_offset_end,
                      static_cast<std::uint64_t>(target.sequence_length) *
                          5'120U * 2U,
                      target.sequence_length, 5'120U, 5'120U, 2U) &&
                matrix_is(plan.final_hidden_bf16,
                          target.final_hidden_offset, 10'240U, 1U, 5'120U,
                          5'120U, 2U) &&
                common.rope_bytes ==
                    static_cast<std::uint64_t>(target.sequence_length) *
                        32U * 4U * 2U,
            "prompt residual, final hidden, and RoPE retain exact shapes");

        runtime::LayerMajorPrefillWorkspaceOptions workspace_options;
        workspace_options.sequence_capacity_tokens = target.sequence_length;
        workspace_options.request_arena_limit_bytes = target.arena_bytes;
        workspace_options.hidden_strategy =
            runtime::PrefillHiddenStrategy::kSinglePromptWideConditional;
        workspace_options.scratch_strategy =
            runtime::PrefillOperatorScratchStrategy::
                kC8192FamilyOverlayWithDisjointLegacyC512;
        workspace_options.gdn_tactic =
            runtime::PrefillGdnPhysicalTactic::kC64NativeInPlaceConv;
        workspace_options.legacy_gdn_tactic =
            runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite;
        workspace_options.mlp_tactic =
            runtime::PrefillMlpPhysicalTactic::kSeparateGateUpAndSilu;
        const auto workspace =
            runtime::build_unbound_layer_major_prefill_workspace_plan(
                workspace_options);
        test.expect(
            workspace &&
                common.arena_bytes == workspace.value->selected.required_bytes &&
                plan.c8192_family_phase_arena.byte_size ==
                    workspace.value->operator_scratch
                        .c8192_family_overlay_conditional.total_required_bytes &&
                plan.gdn.native_c64_workspace.byte_size ==
                    runtime::kLayerMajorPrefillGdnC64NativeWorkspaceBytes,
            "constructed allocation equals the fixed planner-selected total");

        test.expect(
            plan.hidden_strategy ==
                    runtime::PrefillHiddenStrategy::
                        kSinglePromptWideConditional &&
                plan.scratch_strategy ==
                    runtime::PrefillOperatorScratchStrategy::
                        kC8192FamilyOverlayWithDisjointLegacyC512 &&
                plan.gdn_tactic ==
                    runtime::PrefillGdnPhysicalTactic::
                        kC64NativeInPlaceConv &&
                plan.legacy_gdn_tactic ==
                    runtime::PrefillLegacyGdnPhysicalTactic::kC16Composite &&
                plan.mlp_tactic ==
                    runtime::PrefillMlpPhysicalTactic::
                        kSeparateGateUpAndSilu,
            "layer-major RequestState fixes every physical tactic identity");
        test.expect(
            !plan.prompt_residual_in_place_contract_bound &&
                !plan.family_completion_events_bound &&
                !plan.intra_family_phase_contract_bound &&
                !plan.prompt_token_ids_consumed_event_bound &&
                !plan.projection_workspace_subrange_binding_bound &&
                !plan.operator_bindings_complete && !plan.executable(),
            "allocation does not claim any execution, alias, event, or binding");

        runtime::LayerMajorRequestMemoryOptions short_cap = options;
        short_cap.max_arena_bytes = target.arena_bytes - 1U;
        const runtime::LayerMajorRequestPlanResult rejected =
            runtime::build_layer_major_request_memory_plan(short_cap);
        test.expect(
            !rejected &&
                rejected.diagnostic.code ==
                    runtime::RequestErrorCode::kArenaLimitExceeded &&
                rejected.diagnostic.actual ==
                    std::to_string(target.arena_bytes),
            "layer-major target bucket rejects an arena cap one byte short");
    }
}

void test_layer_major_typed_phase_layout(TestContext& test) {
    runtime::LayerMajorRequestMemoryOptions options;
    options.max_sequence_length = 40'000U;
    options.max_arena_bytes = 4'066'344'960ULL;
    const auto result =
        runtime::build_layer_major_request_memory_plan(options);
    test.expect(static_cast<bool>(result),
                "40K layer-major plan exists for typed phase audit");
    if (!result) {
        return;
    }
    const runtime::LayerMajorRequestMemoryPlan& plan = *result.value;
    const std::uint64_t base = plan.c8192_family_phase_arena.arena_offset;

    test.expect(
        matrix_is(plan.panel_token_ids_u32, base, 32'768U, 8'192U, 1U,
                  1U, 4U),
        "panel token IDs own exactly one event-gated U32 arena prefix");
    test.expect(
        matrix_is(plan.gdn.qkv_bf16, base, 167'772'160U, 8'192U,
                  10'240U, 10'240U, 2U) &&
            matrix_is(plan.gdn.z_bf16, base + 167'772'160U,
                      100'663'296U, 8'192U, 6'144U, 6'144U, 2U) &&
            matrix_is(plan.gdn.a_bf16, base + 268'435'456U, 786'432U,
                      8'192U, 48U, 48U, 2U) &&
            matrix_is(plan.gdn.b_bf16, base + 269'221'888U, 786'432U,
                      8'192U, 48U, 48U, 2U) &&
            matrix_is(plan.gdn.recurrent_core_bf16,
                      base + 270'008'320U, 100'663'296U, 8'192U, 6'144U,
                      6'144U, 2U) &&
            plan.gdn.native_c64_workspace.arena_offset ==
                base + 370'671'616U &&
            plan.gdn.native_c64_workspace.byte_size == 75'694'080U &&
            matrix_is(plan.gdn.normalized_input_bf16,
                      base + 270'008'320U, 83'886'080U, 8'192U, 5'120U,
                      5'120U, 2U) &&
            plan.gdn.input_projection_temporary.arena_offset ==
                base + 353'894'400U &&
            plan.gdn.input_projection_temporary.byte_size == 1'048'832U &&
            matrix_is(plan.gdn.branch_output_bf16, base, 83'886'080U,
                      8'192U, 5'120U, 5'120U, 2U) &&
            plan.gdn.output_projection_temporary.arena_offset ==
                base + 83'886'080U &&
            plan.gdn.output_projection_temporary.byte_size == 1'048'832U,
        "GDN typed views bind the exact projection/recurrent alias ledger");

    test.expect(
        matrix_is(plan.attention.raw_q_gate_bf16, base, 201'326'592U,
                  8'192U, 12'288U, 12'288U, 2U) &&
            matrix_is(plan.attention.processed_q_bf16,
                      base + 201'326'592U, 100'663'296U, 8'192U, 6'144U,
                      6'144U, 2U) &&
            matrix_is(plan.attention.packed_gate_bf16,
                      base + 301'989'888U, 100'663'296U, 8'192U, 6'144U,
                      6'144U, 2U) &&
            matrix_is(plan.attention.normalized_input_bf16,
                      base + 201'326'592U, 83'886'080U, 8'192U, 5'120U,
                      5'120U, 2U) &&
            plan.attention.input_projection_temporary.arena_offset ==
                base + 285'212'672U &&
            plan.attention.input_projection_temporary.byte_size ==
                1'048'832U &&
            matrix_is(plan.attention.core_output_bf16, base,
                      100'663'296U, 8'192U, 6'144U, 6'144U, 2U) &&
            matrix_is(plan.attention.branch_output_bf16,
                      base + 100'663'296U, 83'886'080U, 8'192U, 5'120U,
                      5'120U, 2U) &&
            plan.attention.output_projection_temporary.arena_offset ==
                base + 184'549'376U &&
            plan.attention.output_projection_temporary.byte_size ==
                1'048'832U,
        "Attention typed views bind the exact preprocess/core alias ledger");
    const auto disjoint = [](const runtime::RequestRegion& left,
                             const runtime::RequestRegion& right) {
        return left.arena_offset + left.byte_size <= right.arena_offset ||
               right.arena_offset + right.byte_size <= left.arena_offset;
    };
    test.expect(
        disjoint(plan.attention.processed_q_bf16.storage,
                 plan.attention.packed_gate_bf16.storage) &&
            disjoint(plan.attention.processed_q_bf16.storage,
                     plan.attention.core_output_bf16.storage) &&
            disjoint(plan.attention.packed_gate_bf16.storage,
                     plan.attention.core_output_bf16.storage) &&
            disjoint(plan.attention.core_output_bf16.storage,
                     plan.attention.branch_output_bf16.storage) &&
            disjoint(plan.attention.core_output_bf16.storage,
                     plan.attention.output_projection_temporary) &&
            disjoint(plan.attention.branch_output_bf16.storage,
                     plan.attention.output_projection_temporary),
        "Attention live query/gate/core/branch/O-temp views are pairwise disjoint");

    test.expect(
        matrix_is_empty(plan.mlp.merged_gate_up_bf16) &&
            matrix_is(plan.mlp.gate_bf16, base, 285'212'672U, 8'192U,
                  17'408U, 17'408U, 2U) &&
            matrix_is(plan.mlp.up_bf16, base + 285'212'672U,
                      285'212'672U, 8'192U, 17'408U, 17'408U, 2U) &&
            matrix_is(plan.mlp.activated_bf16, base + 570'425'344U,
                      285'212'672U, 8'192U, 17'408U, 17'408U, 2U) &&
            matrix_is(plan.mlp.normalized_input_bf16,
                      base + 570'425'344U, 83'886'080U, 8'192U, 5'120U,
                      5'120U, 2U) &&
            plan.mlp.gate_up_projection_temporary.arena_offset ==
                base + 654'311'424U &&
            plan.mlp.gate_up_projection_temporary.byte_size == 1'048'832U &&
            matrix_is(plan.mlp.branch_output_bf16,
                      base + 285'212'672U, 83'886'080U, 8'192U, 5'120U,
                      5'120U, 2U) &&
            plan.mlp.down_projection_temporary.arena_offset == base &&
            plan.mlp.down_projection_temporary.byte_size == 1'048'832U,
        "MLP typed views keep full activated GateUp and a disjoint Down output");
    const std::uint64_t activated_begin =
        plan.mlp.activated_bf16.storage.arena_offset;
    const std::uint64_t activated_end =
        activated_begin + plan.mlp.activated_bf16.storage.byte_size;
    const std::uint64_t gate_begin =
        plan.mlp.gate_bf16.storage.arena_offset;
    const std::uint64_t gate_end =
        gate_begin + plan.mlp.gate_bf16.storage.byte_size;
    const std::uint64_t normalized_begin =
        plan.mlp.normalized_input_bf16.storage.arena_offset;
    const std::uint64_t normalized_end =
        normalized_begin + plan.mlp.normalized_input_bf16.storage.byte_size;
    const std::uint64_t down_output_begin =
        plan.mlp.branch_output_bf16.storage.arena_offset;
    const std::uint64_t down_output_end =
        down_output_begin + plan.mlp.branch_output_bf16.storage.byte_size;
    test.expect(
        normalized_begin == activated_begin &&
            plan.mlp.gate_bf16.storage.byte_size == 285'212'672U &&
            (gate_end <= normalized_begin || normalized_end <= gate_begin),
        "fused G2 may reuse the full Gate span only because it is disjoint "
        "from the intentionally aliased normalized/activated span");
    test.expect(down_output_end <= activated_begin ||
                    activated_end <= down_output_begin,
                "Down output never aliases the live activated MLP input");
}

void test_layer_major_disjoint_legacy_and_ownership(TestContext& test) {
    runtime::LayerMajorRequestMemoryOptions options;
    options.max_sequence_length = 40'000U;
    options.max_arena_bytes = 4'066'344'960ULL;
    const auto result =
        runtime::build_layer_major_request_memory_plan(options);
    if (!result) {
        test.expect(false, "40K plan exists for ownership audit");
        return;
    }
    const runtime::LayerMajorRequestMemoryPlan& plan = *result.value;
    const auto& legacy = plan.legacy_c512;
    const std::uint64_t legacy_base =
        legacy.hidden_bf16.front().storage.arena_offset;
    constexpr std::uint64_t hidden_size = 5'242'880U;
    constexpr std::uint64_t projection_size = 17'825'792U;
    test.expect(
        matrix_is(legacy.hidden_bf16[0U], legacy_base, hidden_size, 512U,
                  5'120U, 5'120U, 2U) &&
            matrix_is(legacy.hidden_bf16[1U], legacy_base + hidden_size,
                      hidden_size, 512U, 5'120U, 5'120U, 2U) &&
            matrix_is(legacy.hidden_bf16[2U],
                      legacy_base + 2U * hidden_size, hidden_size, 512U,
                      5'120U, 5'120U, 2U),
        "disjoint legacy bundle retains three exact C512 hidden matrices");
    for (std::size_t index = 0U; index < legacy.projection_bf16.size();
         ++index) {
        test.expect(
            matrix_is(legacy.projection_bf16[index],
                      legacy_base + 15'728'640U +
                          index * projection_size,
                      projection_size, 512U, 17'408U, 17'408U, 2U),
            "disjoint legacy bundle retains each exact projection matrix");
    }
    test.expect(
        matrix_is(legacy.linear_a_bf16, legacy_base + 87'031'808U,
                  49'152U, 512U, 48U, 48U, 2U) &&
            matrix_is(legacy.linear_b_bf16, legacy_base + 87'080'960U,
                      49'152U, 512U, 48U, 48U, 2U) &&
            legacy.fp32_scratch.arena_offset ==
                legacy_base + 87'130'112U &&
            legacy.fp32_scratch.byte_size == 3'840'000U &&
            legacy.gqa_probability_scratch.arena_offset ==
                legacy.fp32_scratch.arena_offset &&
            legacy.gqa_probability_scratch.byte_size == 3'840'000U,
        "disjoint legacy linear and long-context FP32 views are byte-exact");
    for (std::size_t index = 0U; index < legacy.hidden_bf16.size(); ++index) {
        test.expect(same_region(plan.common.hidden_bf16[index],
                                legacy.hidden_bf16[index].storage),
                    "common metadata names the same explicit legacy hidden view");
    }
    for (std::size_t index = 0U; index < legacy.projection_bf16.size();
         ++index) {
        test.expect(same_region(plan.common.projection_bf16[index],
                                legacy.projection_bf16[index].storage),
                    "common metadata names the same explicit legacy projection view");
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> owning;
    add_region(owning, plan.common.conv_state);
    add_region(owning, plan.common.gdn_state);
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
         ++slot) {
        add_region(owning, plan.common.key_cache[slot]);
        add_region(owning, plan.common.value_cache[slot]);
    }
    add_region(owning, plan.prompt_residual_bf16.storage);
    add_region(owning, plan.c8192_family_phase_arena);
    for (const auto& matrix : legacy.hidden_bf16) {
        add_region(owning, matrix.storage);
    }
    for (const auto& matrix : legacy.projection_bf16) {
        add_region(owning, matrix.storage);
    }
    add_region(owning, legacy.linear_a_bf16.storage);
    add_region(owning, legacy.linear_b_bf16.storage);
    add_region(owning, legacy.fp32_scratch);
    add_region(owning, plan.final_hidden_bf16.storage);
    add_region(owning, plan.common.rope_cos_fp32);
    add_region(owning, plan.common.rope_sin_fp32);
    for (const auto& range : owning) {
        test.expect(range.first % runtime::kRequestArenaAlignment == 0U &&
                        range.second > range.first &&
                        range.second <= plan.common.arena_bytes,
                    "every layer-major owning range is aligned and bounded");
    }
    std::sort(owning.begin(), owning.end());
    for (std::size_t index = 1U; index < owning.size(); ++index) {
        test.expect(owning[index - 1U].second <= owning[index].first,
                    "layer-major owning ranges are physically non-overlapping");
    }
    test.expect(
        plan.c8192_family_phase_arena.arena_offset +
                plan.c8192_family_phase_arena.byte_size <= legacy_base &&
            legacy.fp32_scratch.arena_offset +
                    legacy.fp32_scratch.byte_size <=
                plan.final_hidden_bf16.storage.arena_offset &&
            plan.final_hidden_bf16.storage.arena_offset +
                    plan.final_hidden_bf16.storage.byte_size <=
                plan.common.rope_offset,
        "family arena, legacy workspace, final hidden, and RoPE are disjoint");
}

void test_layer_wide_p40_mlp_request_layout(TestContext& test) {
    runtime::LayerMajorRequestMemoryOptions options;
    options.max_sequence_length =
        runtime::kLayerMajorPrefillLayerWideMlpP40RequestCapacityTokens;
    options.mlp_layout =
        runtime::LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan;
    const auto result =
        runtime::build_layer_major_request_memory_plan(options);
    if (!runtime::layer_wide_p40_mlp_prefill_plan_enabled()) {
        test.expect(
            !result &&
                result.diagnostic.code ==
                    runtime::RequestErrorCode::kInvalidOption,
            "default builds reject the test-only P40 RequestState layout");
        return;
    }
    test.expect(result.ok(),
                "test admission builds the conservative P40 RequestState");
    if (!result) {
        return;
    }

    const runtime::LayerMajorRequestMemoryPlan& plan = *result.value;
    const std::uint64_t base =
        plan.c8192_family_phase_arena.arena_offset;
    test.expect(
        plan.mlp_layout ==
                runtime::LayerMajorRequestMlpLayout::
                    kLayerWideP40PersistentTwoSpan &&
            plan.mlp_capacity_tokens == 40'000U &&
            plan.mlp_tactic ==
                runtime::PrefillMlpPhysicalTactic::
                    kLayerWideP40PersistentFusedGateUp &&
            plan.c8192_family_phase_arena.byte_size == 1'802'240'000U &&
            plan.common.arena_bytes == 5'013'023'488U &&
            !plan.executable(),
        "P40 RequestState owns the planner-exact persistent two-span arena "
        "and remains unbound");
    test.expect(
        matrix_is_empty(plan.mlp.merged_gate_up_bf16) &&
            matrix_is(plan.mlp.gate_bf16, base, 1'392'640'000U, 40'000U,
                  17'408U, 17'408U, 2U) &&
            matrix_is(plan.mlp.up_bf16, base,
                      1'392'640'000U, 40'000U, 17'408U, 17'408U, 2U) &&
            matrix_is(plan.mlp.activated_bf16, base,
                      1'392'640'000U, 40'000U, 17'408U, 17'408U, 2U) &&
            matrix_is(plan.mlp.normalized_input_bf16,
                      base + 1'392'640'000U, 409'600'000U, 40'000U,
                      5'120U, 5'120U, 2U) &&
            matrix_is(plan.mlp.branch_output_bf16,
                      base + 1'392'640'000U, 409'600'000U, 40'000U,
                      5'120U, 5'120U, 2U) &&
            plan.mlp.gate_up_projection_temporary.arena_offset ==
                base &&
            plan.mlp.down_projection_temporary.arena_offset == base,
        "full-M typed views expose one activated and one normalized span; "
        "unused compatibility views are honest aliases of those spans");

    runtime::LayerMajorRequestMemoryOptions wrong_shape = options;
    wrong_shape.max_sequence_length = 39'968U;
    const auto rejected =
        runtime::build_layer_major_request_memory_plan(wrong_shape);
    test.expect(
        !rejected &&
            rejected.diagnostic.code == runtime::RequestErrorCode::kInvalidOption,
        "P40 RequestState layout fails closed for every other capacity");
}

void test_p40_whole_core_request_layout(TestContext& test) {
    runtime::LayerMajorRequestMemoryOptions options;
    options.max_sequence_length =
        runtime::kLayerMajorP40WholeCoreRequestCapacityTokens;
    options.mlp_layout =
        runtime::LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan;
    options.layout =
        runtime::LayerMajorRequestLayout::kP40WholeCorePromptWide;
    const auto result =
        runtime::build_layer_major_request_memory_plan(options);
    if (!runtime::prompt_wide_p40_whole_core_prefill_plan_enabled()) {
        test.expect(
            !result &&
                result.diagnostic.code ==
                    runtime::RequestErrorCode::kInvalidOption,
            "default builds reject the exact P40 whole-core layout");
        return;
    }
    test.expect(result.ok(),
                "admission build constructs the exact P40 whole-core plan");
    if (!result) {
        return;
    }

    const runtime::LayerMajorRequestMemoryPlan& plan = *result.value;
    const runtime::RequestMemoryPlan& common = plan.common;
    const runtime::LayerMajorP40WholeCoreRegions& whole =
        plan.p40_whole_core;
    constexpr std::uint64_t family_base = 3'109'562'368U;
    test.expect(
        common.profile ==
                runtime::RequestMemoryProfile::kLayerMajorP40WholeCore &&
            plan.layout == runtime::LayerMajorRequestLayout::
                               kP40WholeCorePromptWide &&
            plan.operator_panel_capacity_tokens == 8'000U &&
            plan.mlp_capacity_tokens == 40'000U &&
            whole.prompt_token_count == 40'000U &&
            whole.request_capacity_tokens == 40'001U &&
            whole.logical_panel_capacity_tokens == 8'000U &&
            whole.logical_panel_count == 5U &&
            common.persistent_bytes == 2'699'952'128U &&
            plan.prompt_residual_bf16.storage.arena_offset ==
                2'699'952'128U &&
            plan.prompt_residual_bf16.storage.byte_size == 409'610'240U &&
            whole.family_phase_arena.arena_offset == family_base &&
            whole.family_phase_arena.byte_size == 5'429'760'000U &&
            common.workspace_bytes == 5'930'350'592U &&
            common.rope_offset == 8'630'302'720U &&
            common.rope_bytes == 10'240'256U &&
            common.arena_bytes == 8'640'542'976U && !plan.executable(),
        "whole-core plan fixes exact P40000/P40001 geometry and arena ledger");

    test.expect(
        matrix_is(whole.linear.raw_qkv_bf16, family_base, 819'200'000U,
                  40'000U, 10'240U, 10'240U, 2U) &&
            matrix_is(whole.linear.conv_qkv_bf16,
                      family_base + 819'200'000U, 819'200'000U, 40'000U,
                      10'240U, 10'240U, 2U) &&
            matrix_is(whole.linear.z_bf16,
                      family_base + 1'638'400'000U, 491'520'000U, 40'000U,
                      6'144U, 6'144U, 2U) &&
            matrix_is(whole.linear.a_bf16,
                      family_base + 2'129'920'000U, 3'840'000U, 40'000U,
                      48U, 48U, 2U) &&
            matrix_is(whole.linear.b_bf16,
                      family_base + 2'133'760'000U, 3'840'000U, 40'000U,
                      48U, 48U, 2U) &&
            whole.linear.prompt_wide_workspace.arena_offset ==
                family_base + 2'137'600'000U &&
            whole.linear.prompt_wide_workspace.byte_size ==
                2'800'640'000U &&
            matrix_is(whole.linear.output_bf16,
                      family_base + 4'938'240'000U, 491'520'000U, 40'000U,
                      6'144U, 6'144U, 2U),
        "whole-core linear/GDN family regions use the exact seven-span ledger");

    test.expect(
        matrix_is(whole.prompt_token_ids_u32,
                  family_base + 2'137'600'000U, 160'000U, 40'000U, 1U, 1U,
                  4U) &&
            whole.prompt_token_ids_u32.storage.arena_offset != family_base &&
            matrix_is(whole.linear.normalized_input_bf16,
                      family_base + 4'938'240'000U, 409'600'000U, 40'000U,
                      5'120U, 5'120U, 2U) &&
            matrix_is(whole.linear.branch_output_bf16, family_base,
                      409'600'000U, 40'000U, 5'120U, 5'120U, 2U),
        "token staging stays off raw-QKV and ordered aliases retain exact shapes");

    test.expect(
        matrix_is(whole.full_attention.raw_q_gate_bf16, family_base,
                  983'040'000U, 40'000U, 12'288U, 12'288U, 2U) &&
            matrix_is(whole.full_attention.processed_q_bf16,
                      family_base + 983'040'000U, 491'520'000U, 40'000U,
                      6'144U, 6'144U, 2U) &&
            matrix_is(whole.full_attention.packed_gate_bf16,
                      family_base + 1'474'560'000U, 491'520'000U, 40'000U,
                      6'144U, 6'144U, 2U) &&
            matrix_is(whole.full_attention.core_output_bf16, family_base,
                      491'520'000U, 40'000U, 6'144U, 6'144U, 2U) &&
            matrix_is(whole.full_attention.branch_output_bf16,
                      family_base + 491'520'000U, 409'600'000U, 40'000U,
                      5'120U, 5'120U, 2U),
        "full-Attention prompt-wide Q/gate/output aliases are byte-exact");

    test.expect(
        plan.legacy_c512.hidden_bf16.front().storage.arena_offset ==
                8'539'322'368U &&
            plan.legacy_c512.fp32_scratch.byte_size == 3'840'000U &&
            plan.final_hidden_bf16.storage.arena_offset ==
                8'630'292'480U &&
            common.rope_cos_fp32.arena_offset == 8'630'302'720U &&
            common.rope_sin_fp32.arena_offset == 8'635'422'848U,
        "legacy, final handoff, and RoPE follow the whole-core arena exactly");

    runtime::LayerMajorRequestMemoryOptions one_byte_short = options;
    one_byte_short.max_arena_bytes = 8'640'542'975U;
    const auto rejected =
        runtime::build_layer_major_request_memory_plan(one_byte_short);
    test.expect(
        !rejected && rejected.diagnostic.code ==
                         runtime::RequestErrorCode::kArenaLimitExceeded,
        "whole-core request layout rejects an arena one byte short");

    runtime::LayerMajorRequestMemoryOptions wrong_mlp = options;
    wrong_mlp.mlp_layout =
        runtime::LayerMajorRequestMlpLayout::kPanelLocalThreeSpan;
    test.expect(
        !runtime::build_layer_major_request_memory_plan(wrong_mlp),
        "whole-core request layout rejects the panel-local MLP identity");
}

void test_macrofeed_v3_p40_request_admission(TestContext& test) {
    runtime::LayerMajorRequestMemoryOptions options;
    options.max_sequence_length =
        runtime::kLayerMajorP40WholeCoreRequestCapacityTokens;
    options.max_arena_bytes =
        runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes;
    options.mlp_layout =
        runtime::LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan;
    options.layout =
        runtime::LayerMajorRequestLayout::kP40WholeCorePromptWide;
    options.admission =
        runtime::LayerMajorRequestPlanAdmission::kSm87MacroFeedV3;

    const auto result =
        runtime::build_layer_major_request_memory_plan(options);
    if (!runtime::prompt_wide_p40_macrofeed_v3_prefill_plan_enabled()) {
        test.expect(
            !result &&
                result.diagnostic.code ==
                    runtime::RequestErrorCode::kInvalidOption,
            "default builds reject the isolated MacroFeed-v3 P40 admission");
        return;
    }

    test.expect(
        result &&
            result.value->common.profile ==
                runtime::RequestMemoryProfile::kLayerMajorP40WholeCore &&
            result.value->layout ==
                runtime::LayerMajorRequestLayout::kP40WholeCorePromptWide &&
            result.value->mlp_layout == runtime::LayerMajorRequestMlpLayout::
                                            kLayerWideP40PersistentTwoSpan &&
            result.value->common.arena_bytes ==
                runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes,
        "MacroFeed-v3 independently admits the exact audited P40 byte plan");

    runtime::LayerMajorRequestMemoryOptions unlabeled = options;
    unlabeled.admission =
        runtime::LayerMajorRequestPlanAdmission::kDefault;
    if (!runtime::prompt_wide_p40_whole_core_prefill_plan_enabled()) {
        test.expect(
            !runtime::build_layer_major_request_memory_plan(unlabeled),
            "MacroFeed-only builds do not globally reopen old whole-core "
            "admission");
    }

    runtime::LayerMajorRequestMemoryOptions wrong_layout = options;
    wrong_layout.layout =
        runtime::LayerMajorRequestLayout::kC8192FamilyOverlay;
    runtime::LayerMajorRequestMemoryOptions wrong_mlp = options;
    wrong_mlp.mlp_layout =
        runtime::LayerMajorRequestMlpLayout::kPanelLocalThreeSpan;
    runtime::LayerMajorRequestMemoryOptions parity_mlp = options;
    parity_mlp.mlp_layout = runtime::LayerMajorRequestMlpLayout::
        kLayerWideP40MarlinParityMergedGateUp;
    runtime::LayerMajorRequestMemoryOptions wrong_capacity = options;
    wrong_capacity.max_sequence_length--;
    runtime::LayerMajorRequestMemoryOptions unknown_admission = options;
    unknown_admission.admission =
        static_cast<runtime::LayerMajorRequestPlanAdmission>(0xffU);
    test.expect(
        !runtime::build_layer_major_request_memory_plan(wrong_layout) &&
            !runtime::build_layer_major_request_memory_plan(wrong_mlp) &&
            !runtime::build_layer_major_request_memory_plan(parity_mlp) &&
            !runtime::build_layer_major_request_memory_plan(wrong_capacity) &&
            !runtime::build_layer_major_request_memory_plan(
                unknown_admission),
        "MacroFeed-v3 admission cannot authorize another layout, MLP ABI, "
        "capacity, or unknown tag");

    runtime::LayerMajorRequestMemoryOptions one_byte_short = options;
    one_byte_short.max_arena_bytes =
        runtime::kLayerMajorPrefillPromptWideP40RequestArenaBytes - 1U;
    const auto short_result =
        runtime::build_layer_major_request_memory_plan(one_byte_short);
    test.expect(
        !short_result &&
            short_result.diagnostic.code ==
                runtime::RequestErrorCode::kArenaLimitExceeded,
        "MacroFeed-v3 preserves the exact P40 arena high-water boundary");
}

void test_p40_marlin_parity_request_layout(TestContext& test) {
    runtime::LayerMajorRequestMemoryOptions options;
    options.max_sequence_length =
        runtime::kLayerMajorP40WholeCoreRequestCapacityTokens;
    options.mlp_layout = runtime::LayerMajorRequestMlpLayout::
                             kLayerWideP40MarlinParityMergedGateUp;
    options.layout =
        runtime::LayerMajorRequestLayout::kP40WholeCorePromptWide;
    const auto result =
        runtime::build_layer_major_request_memory_plan(options);
    if (!runtime::prompt_wide_p40_whole_core_prefill_plan_enabled() ||
        !runtime::
            prompt_wide_p40_vllm_marlin_parity_prefill_plan_enabled()) {
        test.expect(
            !result &&
                result.diagnostic.code ==
                    runtime::RequestErrorCode::kInvalidOption,
            "old P40 admissions do not enable the parity RequestState layout");
        return;
    }
    test.expect(result.ok(),
                "admission build constructs the P40 Marlin-parity layout");
    if (!result) {
        return;
    }

    const runtime::LayerMajorRequestMemoryPlan& plan = *result.value;
    const std::uint64_t base =
        plan.p40_whole_core.family_phase_arena.arena_offset;
    test.expect(
        plan.common.profile ==
                runtime::RequestMemoryProfile::kLayerMajorP40WholeCore &&
            plan.layout == runtime::LayerMajorRequestLayout::
                               kP40WholeCorePromptWide &&
            plan.mlp_layout == runtime::LayerMajorRequestMlpLayout::
                                   kLayerWideP40MarlinParityMergedGateUp &&
            plan.mlp_tactic == runtime::PrefillMlpPhysicalTactic::
                                   kLayerWideP40MarlinParityMergedGateUp &&
            plan.mlp_capacity_tokens == 40'000U &&
            plan.p40_whole_core.family_phase_arena.byte_size ==
                runtime::kLayerMajorP40WholeCoreFamilyArenaBytes &&
            plan.common.arena_bytes == 8'640'542'976U &&
            !plan.executable(),
        "Marlin parity has a distinct identity without changing the whole-core "
        "request high-water");
    test.expect(
        matrix_is(
            plan.mlp.merged_gate_up_bf16,
            base + runtime::kLayerMajorP40MarlinParityMergedGateUpOffset,
            2'785'280'000U, 40'000U, 34'816U, 34'816U, 2U) &&
            matrix_is_empty(plan.mlp.gate_bf16) &&
            matrix_is_empty(plan.mlp.up_bf16) &&
            matrix_is(
                plan.mlp.activated_bf16,
                base + runtime::kLayerMajorP40MarlinParityActivatedOffset,
                1'392'640'000U, 40'000U, 17'408U, 17'408U, 2U) &&
            region_is(plan.mlp.gate_up_projection_temporary,
                      base + runtime::kLayerMajorP40MarlinParityTemporaryOffset,
                      runtime::kLayerMajorP40MarlinParityTemporaryBytes, 1U) &&
            region_is(plan.mlp.down_projection_temporary,
                      base + runtime::kLayerMajorP40MarlinParityTemporaryOffset,
                      runtime::kLayerMajorP40MarlinParityTemporaryBytes, 1U) &&
            matrix_is(
                plan.mlp.normalized_input_bf16,
                base + runtime::kLayerMajorP40MarlinParityNormalizedOffset,
                409'600'000U, 40'000U, 5'120U, 5'120U, 2U) &&
            matrix_is(
                plan.mlp.branch_output_bf16,
                base + runtime::kLayerMajorP40MarlinParityNormalizedOffset,
                409'600'000U, 40'000U, 5'120U, 5'120U, 2U),
        "P40 typed views expose canonical GateThenUp, activation, shared "
        "stock tail temporary, empty legacy halves, and the Down alias");
    constexpr std::uint64_t kTokenOneGateByteOffset =
        runtime::kLayerMajorP40MarlinParityMergedGateUpRowStrideBytes;
    constexpr std::uint64_t kTokenOneUpByteOffset =
        kTokenOneGateByteOffset +
        runtime::kLayerMajorP40MarlinParityUpColumnOffsetBytes;
    static_assert(kTokenOneGateByteOffset == 69'632U);
    static_assert(kTokenOneUpByteOffset == 104'448U);
    test.expect(
        plan.mlp.merged_gate_up_bf16.storage.arena_offset +
                plan.mlp.merged_gate_up_bf16.storage.byte_size ==
            plan.mlp.activated_bf16.storage.arena_offset &&
            plan.mlp.activated_bf16.storage.arena_offset +
                    plan.mlp.activated_bf16.storage.byte_size ==
                plan.mlp.gate_up_projection_temporary.arena_offset &&
            plan.mlp.gate_up_projection_temporary.arena_offset ==
                plan.mlp.down_projection_temporary.arena_offset &&
            plan.mlp.gate_up_projection_temporary.arena_offset +
                    plan.mlp.gate_up_projection_temporary.byte_size <=
            plan.mlp.normalized_input_bf16.storage.arena_offset &&
            plan.mlp.normalized_input_bf16.storage.arena_offset +
                    plan.mlp.normalized_input_bf16.storage.byte_size <=
                base + plan.p40_whole_core.family_phase_arena.byte_size,
        "token-one Gate/Up columns share one row and the sequential tail "
        "temporary alias is bounded outside live tensors");

    runtime::LayerMajorRequestMemoryOptions wrong_profile = options;
    wrong_profile.layout =
        runtime::LayerMajorRequestLayout::kC8192FamilyOverlay;
    runtime::LayerMajorRequestMemoryOptions wrong_capacity = options;
    wrong_capacity.max_sequence_length = 40'000U;
    test.expect(
        !runtime::build_layer_major_request_memory_plan(wrong_profile) &&
            !runtime::build_layer_major_request_memory_plan(wrong_capacity),
        "Marlin parity fails closed outside exact whole-core P40000/P40001");
}

void test_layer_major_bad_options(TestContext& test) {
    test.expect(
        runtime::validate_request_memory_profile(
            runtime::RequestMemoryProfile::kLegacyC512,
            runtime::RequestMemoryProfile::kLegacyC512) ==
                runtime::RequestAccessError::kNone &&
            runtime::validate_request_memory_profile(
                runtime::RequestMemoryProfile::kLayerMajorC8192,
                runtime::RequestMemoryProfile::kLayerMajorC8192) ==
                runtime::RequestAccessError::kNone &&
            runtime::validate_request_memory_profile(
                runtime::RequestMemoryProfile::kLayerMajorC8192,
                runtime::RequestMemoryProfile::kLegacyC512) ==
                runtime::RequestAccessError::kMemoryProfileMismatch &&
            runtime::validate_request_memory_profile(
                runtime::RequestMemoryProfile::kLegacyC512,
                runtime::RequestMemoryProfile::kLayerMajorC8192) ==
                runtime::RequestAccessError::kMemoryProfileMismatch &&
            runtime::to_string(
                runtime::RequestAccessError::kMemoryProfileMismatch) ==
                "memory_profile_mismatch",
        "workspace profile access is an exact symmetric fail-closed gate");
    runtime::LayerMajorRequestMemoryOptions options;
    auto result = runtime::build_layer_major_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "layer-major planner has no implicit sequence default");
    options.max_sequence_length = 1U;
    options.batch_size = 2U;
    result = runtime::build_layer_major_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "layer-major planner rejects batch greater than one");
    options.batch_size = 1U;
    options.max_sequence_length =
        runtime::kAbsoluteRequestMaxSequenceLength + 1U;
    result = runtime::build_layer_major_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "layer-major planner rejects sequence capacity above maximum");
    options.max_sequence_length = 1U;
    options.max_arena_bytes = 0U;
    result = runtime::build_layer_major_request_memory_plan(options);
    test.expect(!result && result.diagnostic.code ==
                               runtime::RequestErrorCode::kInvalidOption,
                "layer-major planner rejects a zero arena bound");
}

void test_sequence_length_publication_validation(TestContext& test) {
    static_assert(noexcept(runtime::validate_sequence_length_publication(
        0U, 0U, 0U, 0U)));
    static_assert(noexcept(std::declval<runtime::RequestState&>()
                               .publish_sequence_length(0U, 0U)));
    static_assert(noexcept(runtime::is_valid_request_reset_scope(
        runtime::RequestResetScope::kPersistentState)));
    static_assert(noexcept(std::declval<runtime::RequestState&>().reset_async(
        nullptr, runtime::RequestResetScope::kColdMutableArena)));

    test.expect(
        runtime::is_valid_request_reset_scope(
            runtime::RequestResetScope::kPersistentState) &&
            runtime::is_valid_request_reset_scope(
                runtime::RequestResetScope::kColdMutableArena) &&
            !runtime::is_valid_request_reset_scope(
                static_cast<runtime::RequestResetScope>(0xFFU)) &&
            runtime::to_string(
                runtime::RequestAccessError::kInvalidResetScope) ==
                "invalid_reset_scope",
        "request reset scopes and diagnostic string are stable and fail closed");

    test.expect(
        runtime::validate_sequence_length_publication(7U, 7U, 8U, 8U) ==
                runtime::RequestAccessError::kNone &&
            runtime::validate_sequence_length_publication(7U, 7U, 7U, 8U) ==
                runtime::RequestAccessError::kNone,
        "conditional publication accepts bounded growth and idempotence");
    test.expect(
        runtime::validate_sequence_length_publication(7U, 6U, 9U, 8U) ==
            runtime::RequestAccessError::kSequenceLengthMismatch,
        "stale expected length wins before desired-value validation");
    test.expect(
        runtime::validate_sequence_length_publication(7U, 7U, 9U, 8U) ==
                runtime::RequestAccessError::kCapacityExceeded &&
            runtime::validate_sequence_length_publication(10U, 10U, 9U,
                                                          8U) ==
                runtime::RequestAccessError::kCapacityExceeded,
        "capacity rejection precedes regression after expected length matches");
    test.expect(
        runtime::validate_sequence_length_publication(7U, 7U, 6U, 8U) ==
            runtime::RequestAccessError::kSequenceLengthRegression,
        "conditional publication rejects a matching-current regression");
    test.expect(
        runtime::to_string(
            runtime::RequestAccessError::kSequenceLengthMismatch) ==
                "sequence_length_mismatch" &&
            runtime::to_string(
                runtime::RequestAccessError::kSequenceLengthRegression) ==
                "sequence_length_regression",
        "conditional publication errors have stable diagnostic strings");

    runtime::RequestState empty;
    const runtime::RequestOperationStatus empty_publish =
        empty.publish_sequence_length(0U, 1U);
    test.expect(
        !empty_publish &&
            empty_publish.error == runtime::RequestAccessError::kEmptyState &&
            empty.sequence_length() == 0U,
        "empty RequestState rejects publication without changing host length");
}

}  // namespace

int main() {
    TestContext test;
    test_default_exact_plan(test);
    test_prefill_chunk_layout(test);
    test_prefill_chunk_sixteen_layout(test);
    test_prefill_chunk_thirty_two_layout(test);
    test_prefill_chunk_sixty_four_layout(test);
    test_prefill_chunk_large_layout(test);
    test_native_only_c512_layout(test);
    test_target_length_capacity_plans(test);
    test_alignment_non_overlap_and_schedule(test);
    test_minimum_maximum_and_bad_options(test);
    test_layer_major_target_profiles(test);
    test_layer_major_typed_phase_layout(test);
    test_layer_major_disjoint_legacy_and_ownership(test);
    test_layer_wide_p40_mlp_request_layout(test);
    test_p40_whole_core_request_layout(test);
    test_macrofeed_v3_p40_request_admission(test);
    test_p40_marlin_parity_request_layout(test);
    test_layer_major_bad_options(test);
    test_sequence_length_publication_validation(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " request-state plan test(s) failed\n";
        return 1;
    }
    std::cout << "All request-state plan tests passed\n";
    return 0;
}
