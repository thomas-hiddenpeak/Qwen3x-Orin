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
  test_executor_callbacks(test);
  if (test.failures() != 0) {
    std::cerr << test.failures()
              << " long-Prefill layer-major host test(s) failed\n";
    return 1;
  }
  std::cout << "All long-Prefill layer-major host tests passed\n";
  return 0;
}
