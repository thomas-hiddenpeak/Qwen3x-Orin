#include "runtime/mtp_device_internal.h"
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
#include "runtime/mtp_engine_internal.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/whole_core_request_geometry.h"
#include "q3x/runtime/decode_ops.h"
#endif
#include "q3x/runtime/resident_weights.h"
#include "q3x/core/sha256.h"
#include <cuda_runtime_api.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <chrono>

namespace rt = q3x::runtime;
namespace mtp = rt::mtp_detail;
void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }
std::vector<char> read_device(const void* source, std::size_t bytes) {
  std::vector<char> result(bytes);
  require(cudaMemcpy(result.data(), source, bytes, cudaMemcpyDeviceToHost) == cudaSuccess, "D2H capture");
  return result;
}
void write(const std::filesystem::path& path, const std::vector<char>& data) {
  std::ofstream out(path, std::ios::binary);
  out.write(data.data(), data.size());
  require(bool(out), "capture write");
}
std::vector<char> live_state(const rt::RequestState& state, const rt::ReferenceRunner& runner) {
  const auto& plan = state.plan();
  const auto* arena = static_cast<const char*>(state.arena_data());
  auto bytes = read_device(arena + plan.conv_state.arena_offset,
                          plan.conv_state.byte_size + plan.gdn_state.byte_size);
  for (std::size_t i = 0; i < 16; ++i) {
    for (const auto* region : {&plan.key_cache[i], &plan.value_cache[i]}) {
      auto row = read_device(arena + region->arena_offset, state.current_position() * 2048ULL);
      bytes.insert(bytes.end(), row.begin(), row.end());
    }
  }
  auto hidden = read_device(mtp::TargetTransaction::hidden(runner), 10240);
  bytes.insert(bytes.end(), hidden.begin(), hidden.end());
  auto logits = read_device(mtp::TargetTransaction::logits(runner), 248320 * 2);
  bytes.insert(bytes.end(), logits.begin(), logits.end());
  return bytes;
}
std::uint32_t step(rt::ReferenceRunner& runner, std::uint32_t token, bool logits) {
  rt::ReferenceStepOptions options;
  options.compute_logits = logits;
  options.logits_mode = rt::ReferenceLogitsMode::kPredictedTokenOnly;
  auto result = runner.step(token, options);
  if (!result) {
    std::cerr << "target step error=" << static_cast<int>(result.status.error) << '\n';
    throw std::runtime_error("target step failed");
  }
  if (!logits) return 0;
  require(result.value->prediction.has_value(), "target prediction missing");
  return result.value->prediction->predicted_token_id;
}

// Scripted proposals exercise every device restore prefix, independently of
// the checkpoint's natural acceptance. Native drafting still executes first.
struct Scripted final : mtp::RoundBackend {
  mtp::TargetTransaction& inner;
  const std::vector<std::uint32_t>& oracle;
  std::size_t offset = 0;
  int mismatch = -1;
  bool fail_after_verify = false;
  Scripted(mtp::TargetTransaction& backend, const std::vector<std::uint32_t>& ids)
      : inner(backend), oracle(ids) {}
  bool begin(std::uint32_t rows) noexcept override { return inner.begin(rows); }
  bool propose(std::uint32_t seed, std::uint32_t n, std::uint32_t* ids) noexcept override {
    if (!inner.propose(seed, n, ids)) return false;
    if (mismatch >= 0) {
      for (std::uint32_t i = 0; i < n; ++i) ids[i] = oracle[offset + i];
      if (std::uint32_t(mismatch) < n) ids[mismatch] = (ids[mismatch] + 1) % 248320;
    }
    return true;
  }
  bool verify(std::uint32_t seed, const std::uint32_t* ids, std::uint32_t n,
              std::uint32_t* predictions) noexcept override {
    return inner.verify(seed, ids, n, predictions) && !fail_after_verify;
  }
  bool commit_prefix(std::uint32_t rows, std::uint32_t pending) noexcept override {
    return inner.commit_prefix(rows, pending);
  }
  bool finish() noexcept override { return inner.finish(); }
  bool abort() noexcept override { return inner.abort(); }
};
struct Cancel {
  unsigned seen = 0;
  static bool observe(void* context, std::uint32_t) noexcept {
    return ++static_cast<Cancel*>(context)->seen < 2;
  }
};
struct PrefillCancel {
  unsigned polls = 0, stop_at = 1;
  static bool poll(void* context) noexcept {
    auto& c = *static_cast<PrefillCancel*>(context);
    return ++c.polls >= c.stop_at;
  }
};

int main(int argc, char** argv) try {
  if (argc != 4) { std::cerr << "MODEL PROMPT_U32 OUTPUT_DIRECTORY\n"; return 2; }
  const std::filesystem::path output(argv[3]);
  require(std::filesystem::is_directory(output), "output directory must exist");
  std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
  require(bool(input), "prompt input");
  const auto size = input.tellg();
  require(size >= 4 && size <= 513 * 4 && size % 4 == 0, "prompt must have 1..513 tokens");
  std::vector<std::uint32_t> prompt(static_cast<std::size_t>(size) / 4);
  input.seekg(0);
  input.read(reinterpret_cast<char*>(prompt.data()), size);
  require(bool(input), "prompt read incomplete");
  for (auto token : prompt) require(token < 248320, "prompt token out of range");
  std::cout << "loading_base\n" << std::flush;
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
  rt::ReferenceEngineOptions engine_options;
  engine_options.projection_backend = rt::ProjectionBackend::kSm87WeightOnly;
  engine_options.request_options.max_sequence_length = rt::kWholeCoreCompiledSequenceCapacity;
  engine_options.request_options.max_arena_bytes = rt::kWholeCoreCompiledArenaBytes;
  engine_options.request_options.prefill_chunk_size = 512;
  engine_options.prefill_execution_mode = rt::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
  engine_options.prefill_full_attention_tactic = rt::LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactWholePrompt;
  engine_options.prefill_projection_tactic = rt::LayerMajorPrefillProjectionTactic::kNativePromptWideP40WholeCore;
  engine_options.decode_graph_cache_policy = rt::ReferenceDecodeGraphCachePolicy::kDisabled;
  auto engine = rt::create_reference_engine(argv[1], engine_options);
  if (!engine) throw std::runtime_error(engine.diagnostic.stage + ": " + engine.diagnostic.message);
  auto* target = mtp::EngineAccess::runner(*engine.value);
  auto* request = mtp::EngineAccess::state(*engine.value);
  const auto* model_weights = mtp::EngineAccess::model(*engine.value);
#else
  auto resident = rt::load_pinned_qwen36_27b(argv[1]);
  require(bool(resident), "base resident load");
  auto model = rt::bind_qwen36_27b_weights(*resident.value);
  require(bool(model), "base binding");
  rt::RequestMemoryOptions state_options;
  state_options.max_sequence_length = prompt.size() + 32;
  auto state = rt::create_request_state(state_options);
  require(bool(state), "state allocation");
  rt::ReferenceRunnerOptions runner_options;
  runner_options.projection_backend = rt::ProjectionBackend::kSm87WeightOnly;
  auto runner = rt::create_reference_runner(*model.value, *state.value, runner_options);
  require(bool(runner), "runner creation");
  auto* target = &*runner.value;
  auto* request = &*state.value;
  const auto* model_weights = &*model.value;
#endif
  std::cout << "loading_mtp\n" << std::flush;
  mtp::Weights weights(argv[1]);
  mtp::Draft draft(weights, *model_weights, request->max_sequence_length(),
                   mtp::TargetTransaction::cosines(*target),
                   mtp::TargetTransaction::sines(*target));
  mtp::TargetTransaction transaction(*target, *request, draft,
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
                                   true
#else
                                   false
#endif
                                   );
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
  const auto initialize = [&]() {
    rt::ReferenceGenerateOptions options;
    options.max_new_tokens = 1;
    options.prefill_chunk_size = 512;
    options.logits_mode = rt::ReferenceLogitsMode::kPredictedTokenOnly;
    options.prefill_execution_mode = rt::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
    const auto result = engine.value->generate_prompt_token_ids(prompt, options);
    if (!result) throw std::runtime_error(result.diagnostic.stage + ": " + result.diagnostic.operation);
    const auto final_view = request->layer_major_final_hidden();
    require(bool(final_view), "whole-core final view");
    const auto final_hidden = read_device(final_view.value->storage.device_data, 10240);
    const auto before = live_state(*request, *target);
    require(transaction.initialize_whole_core_prefill(prompt.data(), prompt.size()), "whole-core MTP initialization");
    const auto after = live_state(*request, *target);
    // Only the previously unused scalar final-hidden workspace may change.
    const auto hidden_begin = before.size() - 248320 * 2 - 10240;
    require(std::equal(before.begin(), before.begin() + hidden_begin, after.begin()) &&
            std::equal(before.end() - 248320 * 2, before.end(), after.end() - 248320 * 2),
            "MTP initialization altered target persistent state/logits");
    require(read_device(mtp::TargetTransaction::hidden(*target), 10240) == final_hidden,
            "prompt capture final row differs from whole-core handoff");
    return result.value->generated_token_ids.at(0);
  };
#endif
  std::vector<std::uint32_t> expected;
  std::uint32_t seed = 0;
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
  seed = initialize();
#else
  for (std::size_t i = 0; i < prompt.size(); ++i) {
    seed = step(*target, prompt[i], i + 1 == prompt.size());
  }
#endif
  expected.push_back(seed);
  for (unsigned i = 1; i < 16 && seed != 248046; ++i) {
    seed = step(*target, seed, true);
    expected.push_back(seed);
  }
  require(expected.size() >= 5, "fixture must exercise a complete MTP round");
  std::cout << "baseline_ready\n" << std::flush;
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
  // Select every staged prefix before publishing a new round, then compare
  // complete recurrent/KV/hidden/logit state with independent scalar replay.
  for (unsigned count : {2U, 3U, 4U}) {
    require(bool(target->reset()) && draft.reset(), "prefix oracle reset");
    seed = initialize();
    std::array<std::uint32_t, 3> proposals{};
    std::array<std::uint32_t, 4> predictions{};
    for (unsigned i = 0; i + 1 < count; ++i) proposals[i] = expected[i + 1];
    require(transaction.begin(count) &&
            transaction.verify(seed, proposals.data(), count - 1, predictions.data()),
            "multi-row staged prefix verification");
    std::vector<std::vector<char>> prefix_states;
    for (unsigned i = 0; i < count; ++i) {
      require(predictions[i] == expected[i + 1] &&
              transaction.commit_prefix(i + 1, predictions[i]), "multi-row prefix selection");
      prefix_states.push_back(live_state(*request, *target));
    }
    require(transaction.finish(), "multi-row prefix finish");
    require(bool(target->reset()) && draft.reset(), "scalar prefix reset");
    require(initialize() == expected[0], "scalar prefix seed");
    for (unsigned i = 0; i < count; ++i) {
      require(step(*target, expected[i], true) == predictions[i], "scalar prefix prediction");
      const auto scalar_state = live_state(*request, *target);
      if (scalar_state != prefix_states[i]) {
        const auto diff = std::mismatch(scalar_state.begin(), scalar_state.end(), prefix_states[i].begin());
        std::cerr << "verify_rows=" << count << " prefix=" << i + 1
                  << " first_different_byte=" << diff.first - scalar_state.begin() << '\n';
        write(output / "failed-scalar-state.bin", scalar_state);
        write(output / "failed-multirow-state.bin", prefix_states[i]);
      }
      require(scalar_state == prefix_states[i], "multi-row full prefix differs");
    }
    std::cout << "verify_rows=" << count << " all_prefix_state_and_logits=bitwise_equal\n" << std::flush;
  }
#endif
  struct Case { unsigned length; int mismatch; bool cancel; bool fail; };
  std::vector<Case> cases;
  for (unsigned length : {2U, 3U}) {
    cases.push_back({length, -1, false, false});
    for (unsigned mismatch = 0; mismatch <= length; ++mismatch)
      cases.push_back({length, int(mismatch), false, false});
    cases.push_back({length, int(length), true, false});
    cases.push_back({length, int(length), false, true});
  }
  for (const auto& test : cases) {
    const auto length = test.length;
    require(bool(target->reset()) && draft.reset(), "request reset");
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
    seed = initialize();
    if (length == 2 && test.mismatch == -1) {
      const auto residual = request->layer_major_prompt_residual();
      require(bool(residual), "prompt residual view");
      write(output / "target-residual.bf16", read_device(residual.value->storage.device_data, prompt.size() * 10240));
      write(output / "target-hidden.bf16", read_device(transaction.prompt_hidden(), prompt.size() * 10240));
      write(output / "draft-prefill-k.bf16", read_device(draft.keys(), draft.position() * 2048ULL));
      write(output / "draft-prefill-v.bf16", read_device(draft.values(), draft.position() * 2048ULL));
      const auto expected_k = read_device(draft.keys(), draft.position() * 2048ULL);
      const auto expected_v = read_device(draft.values(), draft.position() * 2048ULL);
      require(draft.reset(), "draft capture reset");
      const auto scalar_started = std::chrono::steady_clock::now();
      std::vector<char> hidden_rows;
      for (std::size_t row = 0; row + 1 < prompt.size(); ++row) {
        std::uint32_t ignored = 0;
        require(draft.step(prompt[row + 1], transaction.prompt_hidden() + row * 5120,
                           false, ignored), "draft capture replay");
        auto hidden = read_device(draft.hidden(), 10240);
        hidden_rows.insert(hidden_rows.end(), hidden.begin(), hidden.end());
      }
      require(read_device(draft.keys(), expected_k.size()) == expected_k &&
              read_device(draft.values(), expected_v.size()) == expected_v,
              "draft initialization capture replay differs");
      write(output / "draft-prefill-hidden.bf16", hidden_rows);
      const auto scalar_ms = std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - scalar_started).count();
      // Exercise short/masked batches and prove no cache row beyond the
      // requested live prefix is written. Full-step replay above is the oracle.
      for (unsigned count : {0U, 1U, 7U, 8U, 9U, unsigned(prompt.size() - 1)}) {
        if (count >= prompt.size()) continue;
        require(draft.reset(), "batch test reset");
        const auto guard_k = read_device(draft.keys() + count * 1024ULL, 2048);
        const auto guard_v = read_device(draft.values() + count * 1024ULL, 2048);
        const auto started = std::chrono::steady_clock::now();
        require(draft.initialize_kv(prompt.data() + 1, transaction.prompt_hidden(), count),
                "batched K/V initialization");
        const auto batch_ms = std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - started).count();
        require(draft.position() == count &&
                read_device(draft.keys(), count * 2048ULL) ==
                    std::vector<char>(expected_k.begin(), expected_k.begin() + count * 2048ULL) &&
                read_device(draft.values(), count * 2048ULL) ==
                    std::vector<char>(expected_v.begin(), expected_v.begin() + count * 2048ULL),
                "batched prefix differs from scalar K/V");
        require(read_device(draft.keys() + count * 1024ULL, 2048) == guard_k &&
                read_device(draft.values() + count * 1024ULL, 2048) == guard_v,
                "batched K/V crossed live prefix");
        std::cout << "draft_prefill_rows=" << count << " batch_ms=" << batch_ms
                  << " scalar_prefix_with_capture_ms=" << scalar_ms
                  << " kv=bitwise_equal guard=pass\n" << std::flush;
      }
      for (unsigned stop_at : {1U, 2U}) {
        if (stop_at == 2 && prompt.size() == 1) continue;
        PrefillCancel cancellation{0, stop_at};
        require(!transaction.initialize_whole_core_prefill(prompt.data(), prompt.size(),
                     PrefillCancel::poll, &cancellation) && target->poisoned(),
                "Prefill cancellation must poison both participants");
        require(draft.position() == (stop_at == 1 ? 0U : std::min<unsigned>(8, prompt.size() - 1)),
                "Prefill cancellation crossed one batch");
        std::uint32_t ignored = 0;
        require(!draft.step(prompt[0], transaction.prompt_hidden(), false, ignored),
                "cancelled draft must reject reuse");
        require(bool(target->reset()) && draft.reset(), "Prefill cancellation reset");
        require(initialize() == expected[0], "Prefill cancellation recovery seed");
        std::cout << "draft_prefill_cancel_poll=" << stop_at << " recovery=pass\n" << std::flush;
      }
    }
#else
    std::vector<char> target_hidden, draft_hidden;
    for (std::size_t i = 0; i < prompt.size(); ++i) {
      seed = step(*target, prompt[i], i + 1 == prompt.size());
      auto row = read_device(mtp::TargetTransaction::hidden(*target), 10240);
      target_hidden.insert(target_hidden.end(), row.begin(), row.end());
      if (i + 1 < prompt.size()) {
        std::uint32_t ignored = 0;
        require(draft.step(prompt[i + 1], mtp::TargetTransaction::hidden(*target), false, ignored), "draft prefill");
        row = read_device(draft.hidden(), 10240);
        draft_hidden.insert(draft_hidden.end(), row.begin(), row.end());
      }
    }
    if (length == 2 && test.mismatch == -1) {
      write(output / "target-hidden.bf16", target_hidden);
      write(output / "draft-prefill-hidden.bf16", draft_hidden);
      write(output / "draft-prefill-k.bf16", read_device(draft.keys(), draft.position() * 2048ULL));
      write(output / "draft-prefill-v.bf16", read_device(draft.values(), draft.position() * 2048ULL));
    }
#endif
    std::vector<std::uint32_t> actual{seed};
    unsigned proposed = 0, accepted = 0, rows = 0, rounds = 0;
    Scripted backend(transaction, expected);
    backend.mismatch = test.mismatch;
    backend.fail_after_verify = test.fail;
    Cancel cancellation;
    while (actual.size() < expected.size()) {
      mtp::RoundOptions options;
      options.draft_length = length;
      options.seed_token = actual.back();
      options.remaining_output = expected.size() - actual.size();
      options.available_target_rows = request->max_sequence_length() - request->current_position();
      options.vocabulary_size = 248320;
      options.stop_token = 248046;
      backend.offset = actual.size();
      const auto result = mtp::run_round(options, backend,
          test.cancel ? Cancel::observe : nullptr, test.cancel ? &cancellation : nullptr);
      if (test.fail) {
        require(!result.ok() && result.output_count == 0 && result.abort_succeeded &&
                target->poisoned(), "device failure must poison without publication");
        break;
      }
      if (!result.ok()) std::cerr << "round=" << rounds << " status=" << int(result.status) << " draft_error=" << draft.error() << '\n';
      require(result.ok(), "MTP round");
      actual.insert(actual.end(), result.output.begin(), result.output.begin() + result.output_count);
      proposed += result.proposed_tokens;
      accepted += result.accepted_tokens;
      rows += result.verified_rows;
      ++rounds;
      if (test.cancel) require(result.status == mtp::RoundStatus::kCancelled &&
                               actual.size() == 3, "cancel publication boundary");
      if (result.status == mtp::RoundStatus::kStop || result.status == mtp::RoundStatus::kCancelled) break;
    }
    if (test.fail) {
      require(bool(target->reset()) && draft.reset(), "failed transaction recovery");
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
      seed = initialize();
#else
      for (std::size_t i = 0; i < prompt.size(); ++i)
        seed = step(*target, prompt[i], i + 1 == prompt.size());
#endif
      require(seed == expected[0], "post-failure request output");
      std::cout << "draft_length=" << length << " injected_verify_failure=poisoned recovery=pass\n" << std::flush;
      continue;
    }
    require(actual == std::vector<std::uint32_t>(expected.begin(), expected.begin() + actual.size()),
            "MTP tokens differ from scalar oracle");
    require(test.cancel || actual.size() == expected.size(), "incomplete generation");
    const auto actual_state = live_state(*request, *target);
    // Reconstruct the complete committed draft prefix independently using
    // scalar target hidden, and compare BOTH live KV arrays after rejection.
    const auto live_k = read_device(draft.keys(), draft.position() * 2048ULL);
    const auto live_v = read_device(draft.values(), draft.position() * 2048ULL);
    const auto live_position = draft.position();
    require(bool(target->reset()) && draft.reset(), "reconciliation oracle reset");
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
    require(initialize() == expected[0], "whole-core replay seed");
    for (std::size_t i = 0; i + 1 < actual.size(); ++i) {
      std::uint32_t ignored = 0;
      require(draft.step(actual[i], mtp::TargetTransaction::hidden(*target), false, ignored), "draft target replay");
      step(*target, actual[i], true);
    }
#else
    auto tokens = prompt;
    tokens.insert(tokens.end(), actual.begin(), actual.end() - 1);
    for (std::size_t i = 0; i + 1 < tokens.size(); ++i) {
      step(*target, tokens[i], false);
      std::uint32_t ignored = 0;
      require(draft.step(tokens[i + 1], mtp::TargetTransaction::hidden(*target), false, ignored), "reconciliation oracle draft");
    }
    step(*target, tokens.back(), true);
#endif
    require(live_state(*request, *target) == actual_state, "MTP full live target state differs");
    require(draft.position() == live_position, "draft position differs");
    require(read_device(draft.keys(), live_k.size()) == live_k, "draft K differs from target-conditioned replay");
    require(read_device(draft.values(), live_v.size()) == live_v, "draft V differs from target-conditioned replay");
    std::cout << "draft_length=" << length << " scripted_mismatch=" << test.mismatch
              << " cancelled=" << test.cancel << " output_tokens=" << actual.size()
              << " rounds=" << rounds << " proposed=" << proposed << " accepted=" << accepted
              << " verified_rows=" << rows << " target_state_and_full_logits=bitwise_equal draft_kv=bitwise_equal\n" << std::flush;
  }
#if defined(Q3X_MTP_WHOLE_CORE_TEST)
  std::cout << "prefill_route=corrected_whole_core draft_initialization=batched_live_kv\n"
               "verifier=multirow_four_chain scope=multirow_correctness_only\n";
#else
  std::cout << "verifier=scalar scope=scalar_correctness_only\n";
#endif
  std::cout << "production_api=false speedup_claim=false\n";
  return 0;
} catch (const std::exception& error) {
  std::cerr << error.what() << '\n';
  return 1;
}
