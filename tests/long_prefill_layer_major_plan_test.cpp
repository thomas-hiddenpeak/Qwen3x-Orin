#include "q3x/runtime/long_prefill_layer_major.h"
#include "q3x/runtime/request_state.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

namespace {

namespace model = q3x::model;
namespace runtime = q3x::runtime;

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

[[nodiscard]] bool same_region(const runtime::RequestRegion& left,
                               const runtime::RequestRegion& right) {
  return left.arena_offset == right.arena_offset &&
         left.byte_size == right.byte_size &&
         left.element_capacity == right.element_capacity &&
         left.element_size_bytes == right.element_size_bytes;
}

void test_p4096_memory_plan(TestContext& test) {
  runtime::RequestMemoryOptions base_options;
  base_options.prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  base_options.max_sequence_length = 4'096U;
  base_options.max_arena_bytes = 1ULL * 1024ULL * 1024ULL * 1024ULL;
  const runtime::RequestPlanResult base =
      runtime::build_request_memory_plan(base_options);

  runtime::RequestMemoryOptions long_options = base_options;
  long_options.long_prefill_token_capacity = 4'096U;
  const runtime::RequestPlanResult long_plan =
      runtime::build_request_memory_plan(long_options);
  test.expect(base && long_plan,
              "base and layer-major P4096 C512 request plans build");
  if (!base || !long_plan) {
    return;
  }

  const runtime::RequestMemoryPlan& baseline = *base.value;
  const runtime::RequestMemoryPlan& plan = *long_plan.value;
  test.expect(baseline.long_prefill_token_capacity == 0U &&
                  baseline.long_prefill_hidden_bf16[0].byte_size == 0U &&
                  baseline.long_prefill_hidden_bf16[1].byte_size == 0U &&
                  baseline.persistent_bytes == 346'882'048U &&
                  baseline.workspace_bytes == 88'178'688U &&
                  baseline.rope_bytes == 1'048'576U &&
                  baseline.arena_bytes == 436'109'312U,
              "ordinary P4096 C512 arena remains byte-exact");
  test.expect(plan.long_prefill_token_capacity == 4'096U &&
                  plan.persistent_bytes == 346'882'048U &&
                  plan.workspace_offset == 346'882'048U &&
                  plan.workspace_bytes == 172'064'768U &&
                  plan.rope_offset == 518'946'816U &&
                  plan.rope_bytes == 1'048'576U &&
                  plan.arena_bytes == 519'995'392U &&
                  plan.arena_bytes - baseline.arena_bytes == 83'886'080U,
              "layer-major P4096 adds exactly two full hidden slabs");

  constexpr std::uint64_t kSlabElements = 4'096ULL * 5'120ULL;
  constexpr std::uint64_t kSlabBytes = kSlabElements * 2ULL;
  test.expect(plan.long_prefill_hidden_bf16[0].arena_offset ==
                      435'060'736U &&
                  plan.long_prefill_hidden_bf16[1].arena_offset ==
                      477'003'776U &&
                  plan.long_prefill_hidden_bf16[0].element_capacity ==
                      kSlabElements &&
                  plan.long_prefill_hidden_bf16[1].element_capacity ==
                      kSlabElements &&
                  plan.long_prefill_hidden_bf16[0].byte_size == kSlabBytes &&
                  plan.long_prefill_hidden_bf16[1].byte_size == kSlabBytes &&
                  plan.long_prefill_hidden_bf16[0].element_size_bytes == 2U &&
                  plan.long_prefill_hidden_bf16[1].element_size_bytes == 2U,
              "both P4096 BF16 hidden slabs have exact disjoint spans");
  test.expect(plan.long_prefill_hidden_bf16[0].arena_offset + kSlabBytes <=
                      plan.long_prefill_hidden_bf16[1].arena_offset &&
                  plan.long_prefill_hidden_bf16[1].arena_offset + kSlabBytes <=
                      plan.rope_offset,
              "long-Prefill hidden slabs do not overlap each other or RoPE");

  bool scratch_unchanged =
      same_region(plan.fp32_scratch, baseline.fp32_scratch) &&
      same_region(plan.linear_a_bf16, baseline.linear_a_bf16) &&
      same_region(plan.linear_b_bf16, baseline.linear_b_bf16);
  for (std::size_t index = 0U;
       scratch_unchanged && index < plan.hidden_bf16.size(); ++index) {
    scratch_unchanged =
        same_region(plan.hidden_bf16[index], baseline.hidden_bf16[index]);
  }
  for (std::size_t index = 0U;
       scratch_unchanged && index < plan.projection_bf16.size(); ++index) {
    scratch_unchanged = same_region(plan.projection_bf16[index],
                                    baseline.projection_bf16[index]);
  }
  test.expect(scratch_unchanged,
              "layer-major plan reuses every existing C512 scratch region");
}

void test_long_context_memory_plans(TestContext& test) {
  struct Case final {
    std::uint32_t tokens;
    std::uint64_t persistent_bytes;
    std::uint64_t workspace_bytes;
    std::uint64_t rope_bytes;
    std::uint64_t arena_bytes;
  };
  constexpr std::array<Case, 3U> cases = {{
      {8'192U, 615'317'504U, 255'950'848U, 2'097'152U, 873'365'504U},
      {16'384U, 1'152'188'416U, 424'247'296U, 4'194'304U,
       1'580'630'016U},
      {40'960U, 2'762'801'152U, 929'923'072U, 10'485'760U,
       3'703'209'984U},
  }};
  for (const Case& expected : cases) {
    runtime::RequestMemoryOptions options;
    options.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
    options.max_sequence_length = expected.tokens;
    options.long_prefill_token_capacity = expected.tokens;
    options.max_arena_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
    const runtime::RequestPlanResult built =
        runtime::build_request_memory_plan(options);
    test.expect(built.ok(), "long-context layer-major request plan builds");
    if (!built) {
      continue;
    }
    const runtime::RequestMemoryPlan& plan = *built.value;
    const std::uint64_t slab_elements =
        static_cast<std::uint64_t>(expected.tokens) * 5'120ULL;
    const std::uint64_t slab_bytes = slab_elements * 2ULL;
    test.expect(plan.long_prefill_token_capacity == expected.tokens &&
                    plan.persistent_bytes == expected.persistent_bytes &&
                    plan.workspace_bytes == expected.workspace_bytes &&
                    plan.rope_bytes == expected.rope_bytes &&
                    plan.arena_bytes == expected.arena_bytes &&
                    plan.long_prefill_hidden_bf16[0].element_capacity ==
                        slab_elements &&
                    plan.long_prefill_hidden_bf16[1].element_capacity ==
                        slab_elements &&
                    plan.long_prefill_hidden_bf16[0].byte_size == slab_bytes &&
                    plan.long_prefill_hidden_bf16[1].byte_size == slab_bytes,
                "8K/16K/40K layer-major arena and hidden slabs are exact");
    options.max_arena_bytes = expected.arena_bytes - 1U;
    const runtime::RequestPlanResult one_byte_short =
        runtime::build_request_memory_plan(options);
    test.expect(!one_byte_short &&
                    one_byte_short.diagnostic.code ==
                        runtime::RequestErrorCode::kArenaLimitExceeded,
                "long-context layer-major arena rejects one byte below plan");
  }
}

void test_memory_plan_fail_closed(TestContext& test) {
  runtime::RequestMemoryOptions options;
  options.prefill_chunk_size = runtime::kMaximumRequestPrefillChunkSize;
  options.max_sequence_length = 4'096U;
  options.long_prefill_token_capacity = 4'096U;
  options.max_arena_bytes = 519'995'391U;
  auto result = runtime::build_request_memory_plan(options);
  test.expect(!result && result.diagnostic.code ==
                             runtime::RequestErrorCode::kArenaLimitExceeded,
              "P4096 layer-major arena fails closed by one byte");

  options.max_arena_bytes = 4ULL * 1024ULL * 1024ULL * 1024ULL;
  options.max_sequence_length = 40'961U;
  options.long_prefill_token_capacity = 40'961U;
  result = runtime::build_request_memory_plan(options);
  test.expect(!result && result.diagnostic.code ==
                             runtime::RequestErrorCode::kInvalidOption,
              "P40961 hidden admission capacity is rejected");

  options.long_prefill_token_capacity = 40'960U;
  options.max_sequence_length = 40'959U;
  result = runtime::build_request_memory_plan(options);
  test.expect(!result && result.diagnostic.code ==
                             runtime::RequestErrorCode::kInvalidOption,
              "hidden capacity above request sequence capacity is rejected");

  options.max_sequence_length = 40'960U;
  options.prefill_chunk_size = 256U;
  result = runtime::build_request_memory_plan(options);
  test.expect(!result && result.diagnostic.code ==
                             runtime::RequestErrorCode::kInvalidOption,
              "layer-major slab reservation requires reusable C512 scratch");
}

void test_route_gate(TestContext& test) {
  runtime::LongPrefillLayerMajorRouteQuery query;
  query.runtime_enabled = true;
  query.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  query.capture_trace = false;
  query.prompt_token_count = 4'096U;
  query.prefill_chunk_size = runtime::kLongPrefillLayerMajorTileTokens;
  query.hidden_token_capacity = 4'096U;
  query.hidden_buffer_count =
      runtime::kRequestLongPrefillHiddenBufferCount;
  const auto selected =
      runtime::select_long_prefill_layer_major_route(query);
  test.expect(runtime::long_prefill_layer_major_build_enabled() &&
                  selected == runtime::LongPrefillLayerMajorRoute::
                                  kLayerMajorAdmission,
              "explicit SM87 P4096 C512 admission selects layer-major");

  const auto falls_back = [](runtime::LongPrefillLayerMajorRouteQuery value) {
    return runtime::select_long_prefill_layer_major_route(value) ==
           runtime::LongPrefillLayerMajorRoute::kTileMajorFallback;
  };
  auto candidate = query;
  candidate.runtime_enabled = false;
  test.expect(falls_back(candidate), "runtime-disabled admission falls back");
  candidate = query;
  candidate.projection_backend = runtime::ProjectionBackend::kReference;
  test.expect(falls_back(candidate), "reference backend falls back");
  candidate = query;
  candidate.capture_trace = true;
  test.expect(falls_back(candidate), "trace capture falls back");
  candidate = query;
  candidate.prompt_token_count = 512U;
  test.expect(falls_back(candidate), "short C512 prompt falls back");
  candidate.short_prompt_admission_enabled = true;
  candidate.authenticated_a4_k128 = true;
  test.expect(!falls_back(candidate),
              "explicit authenticated K128 short admission selects P512");
  candidate.prompt_token_count =
      runtime::kShortPrefillLayerMajorMinimumTokens;
  test.expect(!falls_back(candidate),
              "authenticated K128 short admission includes P480");
  candidate.prompt_token_count =
      runtime::kShortPrefillLayerMajorMinimumTokens - 1U;
  test.expect(falls_back(candidate),
              "authenticated K128 short admission excludes P479");
  candidate.prompt_token_count = 512U;
  candidate.authenticated_a4_k128 = false;
  test.expect(falls_back(candidate),
              "short admission never selects a non-K128 inventory");
  candidate.authenticated_a4_k128 = true;
  candidate.short_prompt_admission_enabled = false;
  test.expect(falls_back(candidate),
              "authenticated K128 remains on default short fallback");
  candidate = query;
  candidate.prompt_token_count = 40'961U;
  candidate.hidden_token_capacity = 40'961U;
  test.expect(falls_back(candidate), "P40961 falls back above admission cap");
  candidate = query;
  candidate.prefill_chunk_size = 256U;
  test.expect(falls_back(candidate), "non-C512 scratch policy falls back");
  candidate = query;
  candidate.hidden_token_capacity = 4'095U;
  test.expect(falls_back(candidate), "undersized hidden slabs fall back");
  candidate = query;
  candidate.hidden_buffer_count = 1U;
  test.expect(falls_back(candidate), "single hidden slab falls back");
}

void test_layer_major_schedule(TestContext& test) {
  runtime::LongPrefillLayerMajorOptions options;
  options.prompt_token_count = 4'096U;
  options.hidden_token_capacity = 4'096U;
  const auto result = runtime::build_long_prefill_layer_major_plan(options);
  test.expect(result && result.value->tile_count == 8U &&
                  result.value->full_tile_count == 8U &&
                  result.value->tail_token_count == 0U &&
                  result.value->work_item_count == 512U &&
                  result.value->embedding_output_hidden_buffer == 0U &&
                  result.value->final_hidden_buffer == 0U,
              "P4096 plan is 64 layers by eight C512 tiles");
  if (!result) {
    return;
  }

  const runtime::LongPrefillLayerMajorPlan& plan = *result.value;
  std::size_t linear_items = 0U;
  std::size_t full_items = 0U;
  std::array<std::uint32_t, runtime::kRequestLayerCount> covered_positions{};
  bool ordered = true;
  for (std::size_t ordinal = 0U; ordinal < plan.work_item_count; ++ordinal) {
    runtime::LongPrefillLayerMajorWorkItem item;
    if (!runtime::long_prefill_layer_major_work_item(plan, ordinal, item)) {
      ordered = false;
      break;
    }
    const std::size_t expected_layer = ordinal / plan.tile_count;
    const std::uint32_t expected_tile =
        static_cast<std::uint32_t>(ordinal % plan.tile_count);
    const model::LayerType expected_type =
        ((expected_layer + 1U) % 4U) == 0U
            ? model::LayerType::kFullAttention
            : model::LayerType::kLinearAttention;
    ordered = ordered && item.ordinal == ordinal &&
              item.layer_index == expected_layer &&
              item.layer_type == expected_type &&
              item.tile_index == expected_tile &&
              item.first_position == expected_tile * 512U &&
              item.token_count == 512U &&
              item.input_hidden_buffer == expected_layer % 2U &&
              item.output_hidden_buffer == (expected_layer + 1U) % 2U &&
              item.first_tile_for_layer == (expected_tile == 0U) &&
              item.last_tile_for_layer == (expected_tile == 7U) &&
              item.first_position == covered_positions[expected_layer];
    covered_positions[expected_layer] += item.token_count;
    if (expected_type == model::LayerType::kLinearAttention) {
      ordered = ordered && item.updates_recurrent_state && !item.appends_kv;
      ++linear_items;
    } else {
      ordered = ordered && !item.updates_recurrent_state && item.appends_kv;
      ++full_items;
    }
  }
  for (const std::uint32_t covered : covered_positions) {
    ordered = ordered && covered == 4'096U;
  }
  test.expect(ordered && linear_items == 384U && full_items == 128U,
              "every layer consumes ascending positions before slab swap");

  runtime::LongPrefillLayerMajorWorkItem out_of_range;
  test.expect(!runtime::long_prefill_layer_major_work_item(
                  plan, plan.work_item_count, out_of_range),
              "work-item lookup rejects the first out-of-range ordinal");
}

void test_p40960_layer_major_schedule(TestContext& test) {
  runtime::LongPrefillLayerMajorOptions options;
  options.prompt_token_count = 40'960U;
  options.hidden_token_capacity = 40'960U;
  const auto result = runtime::build_long_prefill_layer_major_plan(options);
  test.expect(result && result.value->tile_count == 80U &&
                  result.value->full_tile_count == 80U &&
                  result.value->tail_token_count == 0U &&
                  result.value->work_item_count == 5'120U,
              "P40960 plan is 64 layers by eighty C512 tiles");
  if (!result) {
    return;
  }
  std::array<std::uint32_t, runtime::kRequestLayerCount> covered{};
  bool ordered = true;
  for (std::size_t ordinal = 0U; ordinal < result.value->work_item_count;
       ++ordinal) {
    runtime::LongPrefillLayerMajorWorkItem item;
    if (!runtime::long_prefill_layer_major_work_item(*result.value, ordinal,
                                                     item)) {
      ordered = false;
      break;
    }
    const std::size_t layer = ordinal / 80U;
    const std::uint32_t tile = static_cast<std::uint32_t>(ordinal % 80U);
    ordered = ordered && item.layer_index == layer &&
              item.tile_index == tile &&
              item.first_position == tile * 512U &&
              item.token_count == 512U &&
              item.first_position == covered[layer] &&
              item.last_tile_for_layer == (tile == 79U);
    covered[layer] += item.token_count;
  }
  for (const std::uint32_t tokens : covered) {
    ordered = ordered && tokens == 40'960U;
  }
  test.expect(ordered,
              "P40960 preserves ascending recurrent/KV order in every layer");
}

void test_tail_and_invalid_plans(TestContext& test) {
  runtime::LongPrefillLayerMajorOptions options;
  options.prompt_token_count = 4'001U;
  options.hidden_token_capacity = 4'096U;
  auto result = runtime::build_long_prefill_layer_major_plan(options);
  runtime::LongPrefillLayerMajorWorkItem tail;
  const bool has_tail = result &&
                        runtime::long_prefill_layer_major_work_item(
                            *result.value, 7U, tail);
  test.expect(has_tail && result.value->tile_count == 8U &&
                  result.value->full_tile_count == 7U &&
                  result.value->tail_token_count == 417U &&
                  tail.first_position == 3'584U && tail.token_count == 417U,
              "P4001 keeps one exact ascending tail per layer");

  options.prompt_token_count = 513U;
  options.hidden_token_capacity = 513U;
  result = runtime::build_long_prefill_layer_major_plan(options);
  test.expect(result && result.value->tile_count == 2U &&
                  result.value->work_item_count == 128U &&
                  result.value->tail_token_count == 1U,
              "P513 plan keeps C512 plus an exact one-token tail");

  options.prompt_token_count = 480U;
  options.hidden_token_capacity = 512U;
  result = runtime::build_long_prefill_layer_major_plan(options);
  runtime::LongPrefillLayerMajorWorkItem short_item;
  const bool has_short_item =
      result && runtime::long_prefill_layer_major_work_item(
                    *result.value, 0U, short_item);
  test.expect(has_short_item && result.value->tile_count == 1U &&
                  result.value->full_tile_count == 0U &&
                  result.value->tail_token_count == 480U &&
                  result.value->work_item_count == 64U &&
                  short_item.first_position == 0U &&
                  short_item.token_count == 480U &&
                  short_item.first_tile_for_layer &&
                  short_item.last_tile_for_layer,
              "P480 plan is one exact layer-major state tile");

  options.prompt_token_count = 512U;
  result = runtime::build_long_prefill_layer_major_plan(options);
  test.expect(result && result.value->tile_count == 1U &&
                  result.value->full_tile_count == 1U &&
                  result.value->tail_token_count == 0U &&
                  result.value->work_item_count == 64U,
              "P512 plan is one complete layer-major state tile");

  options.prompt_token_count = 0U;
  result = runtime::build_long_prefill_layer_major_plan(options);
  test.expect(!result && result.error ==
                             runtime::LongPrefillLayerMajorPlanError::
                                 kInvalidOption,
              "zero-token macro plan is rejected");
  options.prompt_token_count = 40'961U;
  options.hidden_token_capacity = 40'961U;
  result = runtime::build_long_prefill_layer_major_plan(options);
  test.expect(!result && result.error ==
                             runtime::LongPrefillLayerMajorPlanError::
                                 kCapacityExceeded,
              "admission rejects P40961");
  options.prompt_token_count = 4'096U;
  options.hidden_token_capacity = 4'095U;
  result = runtime::build_long_prefill_layer_major_plan(options);
  test.expect(!result && result.error ==
                             runtime::LongPrefillLayerMajorPlanError::
                                 kCapacityExceeded,
              "macro plan rejects undersized hidden slabs");
  options.hidden_token_capacity = 4'096U;
  options.tile_token_count = 256U;
  result = runtime::build_long_prefill_layer_major_plan(options);
  test.expect(!result && result.error ==
                             runtime::LongPrefillLayerMajorPlanError::
                                 kInvalidOption,
              "first macro admission accepts only C512 scratch");
}

void test_projection_span_schedule_case(
    TestContext& test, const std::uint32_t prompt_tokens,
    const std::uint32_t projection_span_tokens,
    const std::uint32_t expected_projection_spans,
    const std::uint32_t expected_full_projection_spans,
    const std::uint32_t expected_projection_tail,
    const std::uint32_t expected_state_tiles,
    const std::uint32_t expected_full_state_tiles,
    const std::uint32_t expected_state_tail,
    const char* const plan_message, const char* const coverage_message) {
  runtime::LongPrefillProjectionSpanOptions options;
  options.prompt_token_count = prompt_tokens;
  options.hidden_token_capacity = prompt_tokens;
  options.projection_span_token_count = projection_span_tokens;
  const auto built =
      runtime::build_long_prefill_projection_span_plan(options);
  test.expect(built &&
                  built.value->projection_span_token_count ==
                      projection_span_tokens &&
                  built.value->projection_span_count ==
                      expected_projection_spans &&
                  built.value->full_projection_span_count ==
                      expected_full_projection_spans &&
                  built.value->projection_tail_token_count ==
                      expected_projection_tail &&
                  built.value->state_tile_token_count == 512U &&
                  built.value->state_tile_count == expected_state_tiles &&
                  built.value->full_state_tile_count ==
                      expected_full_state_tiles &&
                  built.value->state_tail_token_count == expected_state_tail &&
                  built.value->work_item_count ==
                      static_cast<std::size_t>(expected_projection_spans) *
                          runtime::kRequestLayerCount,
              plan_message);
  if (!built) {
    return;
  }

  const runtime::LongPrefillProjectionSpanPlan& plan = *built.value;
  std::array<std::uint32_t, runtime::kRequestLayerCount> span_covered{};
  std::array<std::uint32_t, runtime::kRequestLayerCount> state_covered{};
  std::array<std::uint32_t, runtime::kRequestLayerCount> state_tile_counts{};
  bool exact = true;
  for (std::size_t ordinal = 0U; ordinal < plan.work_item_count; ++ordinal) {
    runtime::LongPrefillProjectionSpanWorkItem span;
    if (!runtime::long_prefill_projection_span_work_item(plan, ordinal,
                                                         span)) {
      exact = false;
      break;
    }
    const std::size_t layer = ordinal / plan.projection_span_count;
    const std::uint32_t span_index =
        static_cast<std::uint32_t>(ordinal % plan.projection_span_count);
    const std::uint32_t expected_first =
        span_index * projection_span_tokens;
    const std::uint32_t remaining = prompt_tokens - expected_first;
    const std::uint32_t expected_tokens =
        remaining < projection_span_tokens ? remaining
                                           : projection_span_tokens;
    const std::uint32_t expected_tiles =
        1U + (expected_tokens - 1U) / 512U;
    const model::LayerType expected_type =
        ((layer + 1U) % 4U) == 0U
            ? model::LayerType::kFullAttention
            : model::LayerType::kLinearAttention;
    exact = exact && span.ordinal == ordinal && span.layer_index == layer &&
            span.layer_type == expected_type &&
            span.projection_span_index == span_index &&
            span.first_position == expected_first &&
            span.first_position == span_covered[layer] &&
            span.token_count == expected_tokens &&
            span.first_state_tile_index == expected_first / 512U &&
            span.state_tile_count == expected_tiles &&
            span.full_state_tile_count == expected_tokens / 512U &&
            span.state_tail_token_count == expected_tokens % 512U &&
            span.input_hidden_buffer == layer % 2U &&
            span.output_hidden_buffer == (layer + 1U) % 2U &&
            span.first_projection_span_for_layer == (span_index == 0U) &&
            span.last_projection_span_for_layer ==
                (span_index + 1U == plan.projection_span_count) &&
            span.updates_recurrent_state ==
                (expected_type == model::LayerType::kLinearAttention) &&
            span.appends_kv ==
                (expected_type == model::LayerType::kFullAttention);

    std::uint32_t span_state_covered = span.first_position;
    for (std::uint32_t tile_index = 0U;
         tile_index < span.state_tile_count; ++tile_index) {
      runtime::LongPrefillProjectionSpanStateTile tile;
      if (!runtime::long_prefill_projection_span_state_tile(
              plan, ordinal, tile_index, tile)) {
        exact = false;
        break;
      }
      const std::uint32_t tile_remaining =
          span.first_position + span.token_count - span_state_covered;
      const std::uint32_t expected_tile_tokens =
          tile_remaining < 512U ? tile_remaining : 512U;
      exact = exact && tile.projection_span_ordinal == ordinal &&
              tile.layer_index == layer && tile.layer_type == expected_type &&
              tile.projection_span_index == span_index &&
              tile.state_tile_index == state_tile_counts[layer] &&
              tile.state_tile_index_in_span == tile_index &&
              tile.first_position == span_state_covered &&
              tile.first_position == state_covered[layer] &&
              tile.token_count == expected_tile_tokens &&
              tile.token_count > 0U && tile.token_count <= 512U &&
              tile.first_state_tile_for_span == (tile_index == 0U) &&
              tile.last_state_tile_for_span ==
                  (tile_index + 1U == span.state_tile_count) &&
              tile.first_state_tile_for_layer ==
                  (state_tile_counts[layer] == 0U) &&
              tile.last_state_tile_for_layer ==
                  (tile.first_position + tile.token_count == prompt_tokens) &&
              tile.updates_recurrent_state == span.updates_recurrent_state &&
              tile.appends_kv == span.appends_kv;
      span_state_covered += tile.token_count;
      state_covered[layer] += tile.token_count;
      ++state_tile_counts[layer];
    }
    runtime::LongPrefillProjectionSpanStateTile out_of_span;
    exact = exact &&
            !runtime::long_prefill_projection_span_state_tile(
                plan, ordinal, span.state_tile_count, out_of_span) &&
            span_state_covered == span.first_position + span.token_count;
    span_covered[layer] += span.token_count;
  }
  for (std::size_t layer = 0U; layer < runtime::kRequestLayerCount; ++layer) {
    exact = exact && span_covered[layer] == prompt_tokens &&
            state_covered[layer] == prompt_tokens &&
            state_tile_counts[layer] == expected_state_tiles;
  }
  runtime::LongPrefillProjectionSpanWorkItem out_of_range;
  exact = exact && !runtime::long_prefill_projection_span_work_item(
                       plan, plan.work_item_count, out_of_range);
  test.expect(exact, coverage_message);
}

void test_projection_span_schedules(TestContext& test) {
  test_projection_span_schedule_case(
      test, 480U, 512U, 1U, 0U, 480U, 1U, 0U, 480U,
      "P480 short whole-M plan is one natural S480 projection span",
      "P480 keeps one exact C480 state tile despite padded K128 projection");
  test_projection_span_schedule_case(
      test, 512U, 512U, 1U, 1U, 0U, 1U, 1U, 0U,
      "P512 short whole-M plan is one full projection span",
      "P512 keeps one exact complete C512 state tile");
  test_projection_span_schedule_case(
      test, 513U, runtime::kLongPrefillProjectionSpanDefaultTokens, 1U, 0U,
      513U, 2U, 1U, 1U,
      "P513 whole-M plan is one short projection span with C512+C1 state",
      "P513 projection/state intervals are ordered, complete, and disjoint");
  test_projection_span_schedule_case(
      test, 1'804U, 1'536U, 2U, 1U, 268U, 4U, 3U, 268U,
      "P1804 uses one S1536 span plus an S268 projection tail",
      "P1804 preserves three C512 states plus one exact C268 tail");
  test_projection_span_schedule_case(
      test, 1'804U, runtime::kLongPrefillProjectionSpanDefaultTokens, 1U, 0U,
      1'804U, 4U, 3U, 268U,
      "P1804 uses one arbitrary-M span with an S4096 workspace",
      "P1804 production-span coverage preserves its exact C268 state tail");
  test_projection_span_schedule_case(
      test, 1'853U, runtime::kLongPrefillProjectionSpanDefaultTokens, 1U, 0U,
      1'853U, 4U, 3U, 317U,
      "P1853 uses one natural projection span inside S4096",
      "P1853 preserves three C512 states plus one exact C317 tail");
  test_projection_span_schedule_case(
      test, 3'987U, 3'584U, 2U, 1U, 403U, 8U, 7U, 403U,
      "P3987 uses one S3584 span plus an S403 projection tail",
      "P3987 preserves seven C512 states plus one exact C403 tail");
  test_projection_span_schedule_case(
      test, 3'987U, runtime::kLongPrefillProjectionSpanDefaultTokens, 1U, 0U,
      3'987U, 8U, 7U, 403U,
      "P3987 uses one arbitrary-M span with an S4096 workspace",
      "P3987 production-span coverage preserves its exact C403 state tail");
  test_projection_span_schedule_case(
      test, 4'096U, runtime::kLongPrefillProjectionSpanDefaultTokens, 1U, 1U,
      0U, 8U, 8U, 0U,
      "P4096 whole-M plan is one full S4096 span with eight C512 states",
      "P4096 projection/state intervals are ordered, complete, and disjoint");
  test_projection_span_schedule_case(
      test, 40'960U, runtime::kLongPrefillProjectionSpanDefaultTokens, 10U,
      10U, 0U, 80U, 80U, 0U,
      "P40960 whole-M plan is ten S4096 spans and 640 layer/span items",
      "P40960 projection/state intervals are ordered, complete, and disjoint");
  test_projection_span_schedule_case(
      test, 40'959U, runtime::kLongPrefillProjectionSpanDefaultTokens, 10U,
      9U, 4'095U, 80U, 79U, 511U,
      "P40959 ends in one natural S4095 projection span",
      "P40959 preserves 79 C512 states plus one exact C511 tail");
  test_projection_span_schedule_case(
      test, 40'960U, 2'048U, 20U, 20U, 0U, 80U, 80U, 0U,
      "P40960 accepts configured S2048 projection spans",
      "configured S2048 preserves the same exact C512 state sequence");
  test_projection_span_schedule_case(
      test, 4'096U, 3'072U, 2U, 1U, 1'024U, 8U, 8U, 0U,
      "projection span is configurable to another C512 multiple",
      "configured S3072 ends with an ordered non-overlapping S1024 span");
}

void test_projection_span_invalid_options(TestContext& test) {
  runtime::LongPrefillProjectionSpanOptions options;
  options.prompt_token_count = 4'096U;
  options.hidden_token_capacity = 4'096U;
  options.projection_span_token_count = 513U;
  auto built = runtime::build_long_prefill_projection_span_plan(options);
  test.expect(!built &&
                  built.error == runtime::LongPrefillLayerMajorPlanError::
                                     kInvalidOption,
              "projection span rejects a non-C512 multiple");

  options.projection_span_token_count = 256U;
  built = runtime::build_long_prefill_projection_span_plan(options);
  test.expect(!built &&
                  built.error == runtime::LongPrefillLayerMajorPlanError::
                                     kInvalidOption,
              "projection span rejects a value below one state tile");

  options.projection_span_token_count = 4'096U;
  options.state_tile_token_count = 256U;
  built = runtime::build_long_prefill_projection_span_plan(options);
  test.expect(!built &&
                  built.error == runtime::LongPrefillLayerMajorPlanError::
                                     kInvalidOption,
              "whole-M plan keeps the recurrent state tile fixed at C512");
}

struct RecordingCallbacks {
  std::size_t prepare_calls = 0U;
  std::size_t prepared_buffer = std::numeric_limits<std::size_t>::max();
  std::uint32_t prepared_tokens = 0U;
  std::size_t fail_ordinal = std::numeric_limits<std::size_t>::max();
  bool fail_prepare = false;
  bool fail_finish = false;
  std::size_t finish_calls = 0U;
  std::uint32_t finished_sequence_length = 0U;
  std::size_t finished_hidden_buffer =
      std::numeric_limits<std::size_t>::max();
  std::vector<runtime::LongPrefillLayerMajorWorkItem> items;
};

bool record_prepare(void* const opaque, const std::size_t buffer,
                    const std::uint32_t tokens) noexcept {
  auto& recording = *static_cast<RecordingCallbacks*>(opaque);
  ++recording.prepare_calls;
  recording.prepared_buffer = buffer;
  recording.prepared_tokens = tokens;
  return !recording.fail_prepare;
}

bool record_item(
    void* const opaque,
    const runtime::LongPrefillLayerMajorWorkItem& item) noexcept {
  auto& recording = *static_cast<RecordingCallbacks*>(opaque);
  if (item.ordinal == recording.fail_ordinal) {
    return false;
  }
  try {
    recording.items.push_back(item);
    return true;
  } catch (...) {
    return false;
  }
}

bool record_finish(void* const opaque, const std::uint32_t sequence_length,
                   const std::size_t hidden_buffer) noexcept {
  auto& recording = *static_cast<RecordingCallbacks*>(opaque);
  ++recording.finish_calls;
  recording.finished_sequence_length = sequence_length;
  recording.finished_hidden_buffer = hidden_buffer;
  return !recording.fail_finish;
}

void test_executor_callbacks(TestContext& test) {
  runtime::LongPrefillLayerMajorOptions options;
  options.prompt_token_count = 4'096U;
  options.hidden_token_capacity = 4'096U;
  const auto built = runtime::build_long_prefill_layer_major_plan(options);
  if (!built) {
    test.expect(false, "executor test plan builds");
    return;
  }

  RecordingCallbacks recording;
  runtime::LongPrefillLayerMajorCallbacks callbacks;
  callbacks.context = &recording;
  callbacks.prepare_hidden = record_prepare;
  callbacks.execute_tile = record_item;
  callbacks.finish_prompt = record_finish;
  auto executed =
      runtime::run_long_prefill_layer_major(*built.value, callbacks);
  test.expect(executed && recording.prepare_calls == 1U &&
                  recording.prepared_buffer == 0U &&
                  recording.prepared_tokens == 4'096U &&
                  recording.items.size() == 512U &&
                  executed.completed_work_items == 512U &&
                  recording.finish_calls == 1U &&
                  recording.finished_sequence_length == 4'096U &&
                  recording.finished_hidden_buffer == 0U &&
                  recording.items.front().layer_index == 0U &&
                  recording.items.back().layer_index == 63U &&
                  recording.items.back().first_position == 3'584U,
              "executor invokes one embedding fill then 512 layer-major tiles");

  RecordingCallbacks failure;
  failure.fail_ordinal = 27U;
  callbacks.context = &failure;
  executed = runtime::run_long_prefill_layer_major(*built.value, callbacks);
  test.expect(!executed &&
                  executed.error == runtime::LongPrefillLayerMajorExecutionError::
                                        kExecuteTileFailed &&
                  executed.completed_work_items == 27U &&
                  executed.failed_work_item.has_value() &&
                  executed.failed_work_item->ordinal == 27U &&
                  failure.items.size() == 27U && failure.finish_calls == 0U,
              "executor stops exactly at a failed layer/tile callback");

  RecordingCallbacks prepare_failure;
  prepare_failure.fail_prepare = true;
  callbacks.context = &prepare_failure;
  executed = runtime::run_long_prefill_layer_major(*built.value, callbacks);
  test.expect(!executed &&
                  executed.error == runtime::LongPrefillLayerMajorExecutionError::
                                        kPrepareHiddenFailed &&
                  executed.completed_work_items == 0U &&
                  prepare_failure.items.empty() &&
                  prepare_failure.finish_calls == 0U,
              "executor does not launch layers after embedding failure");

  RecordingCallbacks finish_failure;
  finish_failure.fail_finish = true;
  callbacks.context = &finish_failure;
  executed = runtime::run_long_prefill_layer_major(*built.value, callbacks);
  test.expect(!executed &&
                  executed.error == runtime::LongPrefillLayerMajorExecutionError::
                                        kFinishPromptFailed &&
                  executed.completed_work_items == 512U &&
                  finish_failure.items.size() == 512U &&
                  finish_failure.finish_calls == 1U,
              "executor reports failure to publish the final prompt state");

  callbacks.prepare_hidden = nullptr;
  executed = runtime::run_long_prefill_layer_major(*built.value, callbacks);
  test.expect(!executed &&
                  executed.error == runtime::LongPrefillLayerMajorExecutionError::
                                        kMissingCallback,
              "executor fails closed on an incomplete callback binding");
}

}  // namespace

int main() {
  TestContext test;
  test_p4096_memory_plan(test);
  test_long_context_memory_plans(test);
  test_memory_plan_fail_closed(test);
  test_route_gate(test);
  test_layer_major_schedule(test);
  test_p40960_layer_major_schedule(test);
  test_tail_and_invalid_plans(test);
  test_projection_span_schedules(test);
  test_projection_span_invalid_options(test);
  test_executor_callbacks(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " long-Prefill layer-major host test(s) failed\n";
    return 1;
  }
  std::cout << "All long-Prefill layer-major host tests passed\n";
  return 0;
}
