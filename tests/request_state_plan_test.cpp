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

bool same_region(const runtime::RequestRegion& left,
                 const runtime::RequestRegion& right) {
    return left.arena_offset == right.arena_offset &&
           left.byte_size == right.byte_size &&
           left.element_capacity == right.element_capacity &&
           left.element_size_bytes == right.element_size_bytes;
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
    test.expect(plan.batch_size == 1U && plan.prefill_chunk_size == 1U &&
                    plan.max_sequence_length == 128U,
                "default plan is batch-one, chunk-one, with 128 positions");
    test.expect(plan.conv_state.byte_size == 2'949'120U &&
                    plan.conv_state.element_capacity == 1'474'560U &&
                    plan.gdn_state.byte_size == 75'497'472U &&
                    plan.gdn_state.element_capacity == 37'748'736U,
                "canonical conv and GDN BF16 state totals are exact");
    test.expect(plan.persistent_offset == 0U &&
                    plan.persistent_bytes == 86'835'200U &&
                    plan.workspace_offset == 86'835'200U &&
                    plan.workspace_bytes == 1'163'776U &&
                    plan.rope_offset == 87'998'976U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == runtime::kDefaultRequestArenaBytes &&
                    plan.arena_bytes == 88'031'744U,
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
    test.expect(plan.fp32_scratch.element_capacity == 248'320U &&
                    plan.fp32_scratch.byte_size == 993'280U &&
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
                    plan.rope_cos_fp32.arena_offset == 87'998'976U &&
                    plan.rope_sin_fp32.arena_offset == 88'015'360U,
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
                    plan.workspace_bytes == 2'354'688U &&
                    plan.rope_offset == 89'189'888U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == 89'222'656U &&
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
    test.expect(plan.rope_cos_fp32.arena_offset == 89'189'888U &&
                    plan.rope_sin_fp32.arena_offset == 89'206'272U &&
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
                    plan.workspace_bytes == 3'716'096U &&
                    plan.rope_offset == 90'551'296U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == 90'584'064U &&
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
    test.expect(plan.rope_cos_fp32.arena_offset == 90'551'296U &&
                    plan.rope_sin_fp32.arena_offset == 90'567'680U &&
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
                    plan.workspace_bytes == 6'438'912U &&
                    plan.rope_offset == 93'274'112U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == 93'306'880U &&
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
    test.expect(plan.rope_cos_fp32.arena_offset == 93'274'112U &&
                    plan.rope_sin_fp32.arena_offset == 93'290'496U &&
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
                    plan.workspace_bytes == 11'884'544U &&
                    plan.rope_offset == 98'719'744U &&
                    plan.rope_bytes == 32'768U &&
                    plan.arena_bytes == 98'752'512U &&
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
                    plan.rope_cos_fp32.arena_offset == 98'719'744U &&
                    plan.rope_sin_fp32.arena_offset == 98'736'128U,
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
                    plan256.workspace_bytes == 44'558'336U &&
                    plan256.rope_offset == 131'393'536U &&
                    plan256.rope_bytes == 32'768U &&
                    plan256.arena_bytes == 131'426'304U,
                "C256 canary workspace and arena totals are byte-exact");
    test.expect(plan512.prefill_chunk_size == 512U &&
                    plan512.persistent_bytes == 86'835'200U &&
                    plan512.workspace_offset == 86'835'200U &&
                    plan512.workspace_bytes == 88'123'392U &&
                    plan512.rope_offset == 174'958'592U &&
                    plan512.rope_bytes == 32'768U &&
                    plan512.arena_bytes == 174'991'360U,
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
    options.max_arena_bytes = 200'255'488U;
    const runtime::RequestPlanResult c512 =
        runtime::build_request_memory_plan(options);

    options.max_sequence_length = 1'024U;
    options.max_arena_bytes = 233'940'992U;
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
                    plan.workspace_bytes == 88'123'392U &&
                    plan.rope_offset == 200'124'416U &&
                    plan.rope_bytes == 131'072U &&
                    plan.arena_bytes == 200'255'488U,
                "C512 maxseq-512 has the exact self-hosted arena layout");
    test.expect(plan1024.persistent_bytes == 145'555'456U &&
                    plan1024.workspace_offset == 145'555'456U &&
                    plan1024.workspace_bytes == 88'123'392U &&
                    plan1024.rope_offset == 233'678'848U &&
                    plan1024.rope_bytes == 262'144U &&
                    plan1024.arena_bytes == 233'940'992U,
                "C512 maxseq-1024 keeps no canonical BF16 weight scratch");

    options.max_sequence_length = 512U;
    options.max_arena_bytes = 200'255'487U;
    const runtime::RequestPlanResult one_byte_short =
        runtime::build_request_memory_plan(options);
    test.expect(!one_byte_short &&
                    one_byte_short.diagnostic.code ==
                        runtime::RequestErrorCode::kArenaLimitExceeded,
                "native-only C512 arena bound remains fail-closed");
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
    test_alignment_non_overlap_and_schedule(test);
    test_minimum_maximum_and_bad_options(test);
    if (test.failures() != 0) {
        std::cerr << test.failures() << " request-state plan test(s) failed\n";
        return 1;
    }
    std::cout << "All request-state plan tests passed\n";
    return 0;
}
