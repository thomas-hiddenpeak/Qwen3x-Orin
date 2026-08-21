#include "q3x/core/sha256.h"
#include "q3x/runtime/reference_engine.h"

#include "reference_engine_final_token_policy_internal.h"
#include "reference_runner_gdn_chunk64_native_admission.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

namespace core = q3x::core;
namespace runtime = q3x::runtime;
namespace detail = q3x::runtime::reference_engine_detail;
namespace runner_detail = q3x::runtime::reference_runner_detail;

constexpr char kSelectedPromptTokensEnvironment[] =
    "Q3X_E2E_PREFILL_PROMPT_TOKENS";
constexpr char kCandidateFirstEnvironment[] =
    "Q3X_E2E_PREFILL_CANDIDATE_FIRST";
constexpr std::uint32_t kPrefillChunkTokens =
    runtime::kMaximumRequestPrefillChunkSize;
constexpr std::size_t kMinimumPromptTokens = 12U;
constexpr std::size_t kStateValidationPromptTokens = 4'096U;
constexpr std::uint32_t kStateValidationDecodeTokens = 16U;
// Qwen's image-pad token cannot be emitted by this text-only request.  Using
// it as the stop id forces the P4096 state oracle to exercise all 15 Decode
// state transitions after the first generated token.
constexpr std::uint32_t kNonTextStopTokenId = 248'056U;
constexpr std::size_t kHashCopyChunkBytes = 4U * 1024U * 1024U;
constexpr std::size_t kKvHeadCount = 4U;
constexpr std::size_t kKvHeadDimension = 256U;
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
constexpr std::array<std::size_t, 11U> kPromptTokenCounts{
    32U, 64U, 407U, 481U, 512U, 564U, 695U, 713U, 1'025U,
    4'096U, 8'192U};

enum class SnapshotError : std::uint8_t {
  kNone = 0,
  kDuplicateHook,
  kInvalidState,
  kInvalidSequenceLength,
  kInvalidPersistentRegion,
  kInvalidKvRegion,
  kHashFailure,
};

[[nodiscard]] const char* to_string(const SnapshotError error) noexcept {
  switch (error) {
    case SnapshotError::kNone:
      return "none";
    case SnapshotError::kDuplicateHook:
      return "duplicate_hook";
    case SnapshotError::kInvalidState:
      return "invalid_state";
    case SnapshotError::kInvalidSequenceLength:
      return "invalid_sequence_length";
    case SnapshotError::kInvalidPersistentRegion:
      return "invalid_persistent_region";
    case SnapshotError::kInvalidKvRegion:
      return "invalid_kv_region";
    case SnapshotError::kHashFailure:
      return "hash_failure";
  }
  return "unknown";
}

struct StateSnapshot {
  StateSnapshot() : copy_scratch(kHashCopyChunkBytes) {}

  std::size_t hook_calls = 0U;
  std::uint32_t sequence_length = 0U;
  SnapshotError error = SnapshotError::kNone;
  int cuda_error = static_cast<int>(cudaSuccess);
  core::Sha256Digest conv_state;
  core::Sha256Digest gdn_state;
  std::array<core::Sha256Digest, runtime::kRequestFullLayerCount> key_cache;
  std::array<core::Sha256Digest, runtime::kRequestFullLayerCount> value_cache;
  core::Sha256Digest aggregate;
  std::vector<std::uint8_t> copy_scratch;
};

[[nodiscard]] bool region_fits(const runtime::RequestRegion& region,
                               const std::uint64_t arena_bytes) noexcept {
  return region.byte_size != 0U && region.element_size_bytes != 0U &&
         region.arena_offset <= arena_bytes &&
         region.byte_size <= arena_bytes - region.arena_offset;
}

[[nodiscard]] bool hash_device_bytes(
    const std::uint8_t* const source, const std::size_t bytes,
    std::vector<std::uint8_t>& scratch, core::Sha256Digest& digest,
    int& cuda_error) noexcept {
  if (source == nullptr || bytes == 0U || scratch.empty()) {
    return false;
  }
  core::Sha256 hash;
  std::size_t offset = 0U;
  while (offset < bytes) {
    const std::size_t chunk = std::min(scratch.size(), bytes - offset);
    const cudaError_t status = cudaMemcpy(
        scratch.data(), source + offset, chunk, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      return false;
    }
    if (!hash.update(scratch.data(), chunk)) {
      return false;
    }
    offset += chunk;
  }
  digest = hash.finalize();
  return true;
}

[[nodiscard]] bool hash_arena_region(
    const runtime::RequestState& state,
    const runtime::RequestRegion& region, const std::size_t bytes,
    StateSnapshot& snapshot, core::Sha256Digest& digest) noexcept {
  if (!region_fits(region, state.arena_bytes()) || bytes == 0U ||
      bytes > region.byte_size ||
      region.arena_offset >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      bytes > std::numeric_limits<std::size_t>::max() -
                  static_cast<std::size_t>(region.arena_offset)) {
    return false;
  }
  const auto* const arena =
      static_cast<const std::uint8_t*>(state.arena_data());
  return hash_device_bytes(
      arena + static_cast<std::size_t>(region.arena_offset), bytes,
      snapshot.copy_scratch, digest, snapshot.cuda_error);
}

void collect_generate_return_snapshot(
    const runtime::RequestState& state, void* const context) noexcept {
  auto* const snapshot = static_cast<StateSnapshot*>(context);
  if (snapshot == nullptr) {
    return;
  }
  ++snapshot->hook_calls;
  if (snapshot->hook_calls != 1U) {
    snapshot->error = SnapshotError::kDuplicateHook;
    return;
  }
  if (!state ||
      state.memory_profile() != runtime::RequestMemoryProfile::kLegacyC512 ||
      state.arena_data() == nullptr || snapshot->copy_scratch.empty()) {
    snapshot->error = SnapshotError::kInvalidState;
    return;
  }
  const cudaError_t synchronize_status = cudaDeviceSynchronize();
  if (synchronize_status != cudaSuccess) {
    snapshot->cuda_error = static_cast<int>(synchronize_status);
    snapshot->error = SnapshotError::kHashFailure;
    return;
  }

  snapshot->sequence_length = state.sequence_length();
  if (snapshot->sequence_length == 0U ||
      snapshot->sequence_length > state.max_sequence_length()) {
    snapshot->error = SnapshotError::kInvalidSequenceLength;
    return;
  }

  const runtime::RequestMemoryPlan& plan = state.plan();
  if (plan.conv_state.byte_size != runtime::kRequestConvStateBytes ||
      plan.gdn_state.byte_size != runtime::kRequestGdnStateBytes ||
      !hash_arena_region(state, plan.conv_state,
                         static_cast<std::size_t>(plan.conv_state.byte_size),
                         *snapshot, snapshot->conv_state) ||
      !hash_arena_region(state, plan.gdn_state,
                         static_cast<std::size_t>(plan.gdn_state.byte_size),
                         *snapshot, snapshot->gdn_state)) {
    snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                          ? SnapshotError::kInvalidPersistentRegion
                          : SnapshotError::kHashFailure;
    return;
  }

  constexpr std::size_t kKvBytesPerLayerPosition =
      kKvHeadCount * kKvHeadDimension * kBf16Bytes;
  if (snapshot->sequence_length >
      std::numeric_limits<std::size_t>::max() /
          kKvBytesPerLayerPosition) {
    snapshot->error = SnapshotError::kInvalidKvRegion;
    return;
  }
  const std::size_t used_kv_bytes =
      static_cast<std::size_t>(snapshot->sequence_length) *
      kKvBytesPerLayerPosition;
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    const runtime::RequestRegion& key = plan.key_cache[slot];
    const runtime::RequestRegion& value = plan.value_cache[slot];
    const bool valid_shape =
        key.element_size_bytes == kBf16Bytes &&
        value.element_size_bytes == kBf16Bytes &&
        key.byte_size == value.byte_size && used_kv_bytes <= key.byte_size;
    if (!valid_shape ||
        !hash_arena_region(state, key, used_kv_bytes, *snapshot,
                           snapshot->key_cache[slot]) ||
        !hash_arena_region(state, value, used_kv_bytes, *snapshot,
                           snapshot->value_cache[slot])) {
      snapshot->error = snapshot->cuda_error == static_cast<int>(cudaSuccess)
                            ? SnapshotError::kInvalidKvRegion
                            : SnapshotError::kHashFailure;
      return;
    }
  }

  core::Sha256 aggregate;
  const auto add_digest = [&aggregate](
                              const core::Sha256Digest& digest) noexcept {
    return aggregate.update(digest.bytes.data(), digest.bytes.size());
  };
  bool complete = aggregate.update(&snapshot->sequence_length,
                                   sizeof(snapshot->sequence_length)) &&
                  add_digest(snapshot->conv_state) &&
                  add_digest(snapshot->gdn_state);
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    complete = complete && add_digest(snapshot->key_cache[slot]) &&
               add_digest(snapshot->value_cache[slot]);
  }
  if (!complete) {
    snapshot->error = SnapshotError::kHashFailure;
    return;
  }
  snapshot->aggregate = aggregate.finalize();
}

class ScopedGenerateReturnSnapshot final {
 public:
  explicit ScopedGenerateReturnSnapshot(StateSnapshot& snapshot) noexcept
      : previous_(runner_detail::
                      exchange_reference_engine_generate_return_snapshot_hook(
                          {collect_generate_return_snapshot, &snapshot})) {}

  ~ScopedGenerateReturnSnapshot() {
    (void)runner_detail::
        exchange_reference_engine_generate_return_snapshot_hook(previous_);
  }

  ScopedGenerateReturnSnapshot(const ScopedGenerateReturnSnapshot&) = delete;
  ScopedGenerateReturnSnapshot& operator=(
      const ScopedGenerateReturnSnapshot&) = delete;

 private:
  runner_detail::ReferenceEngineGenerateReturnSnapshotHook previous_{};
};

class ScopedFinalTokenPolicy final {
 public:
  using Policy = detail::ReferenceEnginePrefillFinalTokenPolicyForTest;

  explicit ScopedFinalTokenPolicy(const Policy policy) noexcept
      : previous_(
            detail::exchange_reference_engine_prefill_final_token_policy_for_test(
                policy)) {}

  ~ScopedFinalTokenPolicy() {
    (void)detail::
        exchange_reference_engine_prefill_final_token_policy_for_test(
            previous_);
  }

  ScopedFinalTokenPolicy(const ScopedFinalTokenPolicy&) = delete;
  ScopedFinalTokenPolicy& operator=(const ScopedFinalTokenPolicy&) = delete;

 private:
  Policy previous_ = Policy::kProductionDefault;
};

[[nodiscard]] std::string model_directory_from(
    const int argc, char** const argv) {
  if (argc >= 2 && argv[1] != nullptr && argv[1][0] != '\0' &&
      std::string_view(argv[1]) != "-") {
    return argv[1];
  }
  const char* const environment = std::getenv("Q3X_E2E_MODEL_DIR");
  return environment == nullptr ? std::string{} : std::string(environment);
}

[[nodiscard]] std::string repeated_hello_prompt(
    const std::size_t prompt_token_count) {
  // The pinned non-thinking chat template maps "hello" followed by P-13
  // copies of " hello" to exactly P prompt tokens.
  const std::size_t repetitions = prompt_token_count - 12U;
  std::string prompt;
  prompt.reserve(6U * repetitions);
  for (std::size_t index = 0U; index < repetitions; ++index) {
    if (index != 0U) {
      prompt.push_back(' ');
    }
    prompt.append("hello");
  }
  return prompt;
}

void print_diagnostic(
    const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << "code=" << runtime::to_string(diagnostic.code)
            << " stage=" << diagnostic.stage
            << " message=" << diagnostic.message;
  if (!diagnostic.context.empty()) {
    std::cerr << " context=" << diagnostic.context;
  }
  if (diagnostic.cuda_error != 0) {
    std::cerr << " cuda_error=" << diagnostic.cuda_error;
  }
  if (diagnostic.layer != runtime::kReferenceNoLayer) {
    std::cerr << " layer=" << diagnostic.layer;
  }
  if (!diagnostic.operation.empty()) {
    std::cerr << " operation=" << diagnostic.operation;
  }
  std::cerr << '\n';
}

[[nodiscard]] bool valid_snapshot(
    const StateSnapshot& snapshot, const std::size_t expected_sequence_length,
    const std::string_view route, const std::size_t prompt_token_count) {
  const bool valid =
      expected_sequence_length <=
          static_cast<std::size_t>(
              std::numeric_limits<std::uint32_t>::max()) &&
      snapshot.hook_calls == 1U &&
      snapshot.sequence_length == expected_sequence_length &&
      snapshot.error == SnapshotError::kNone &&
      snapshot.cuda_error == static_cast<int>(cudaSuccess);
  if (!valid) {
    std::cerr << "P" << prompt_token_count << ' ' << route
              << " snapshot failed: calls=" << snapshot.hook_calls
              << " sequence_length=" << snapshot.sequence_length
              << " expected_sequence_length=" << expected_sequence_length
              << " snapshot_error=" << to_string(snapshot.error)
              << " cuda_error=" << snapshot.cuda_error << '\n';
  }
  return valid;
}

[[nodiscard]] bool same_snapshots(
    const StateSnapshot& baseline, const StateSnapshot& candidate,
    const std::size_t prompt_token_count) {
  bool exact = baseline.sequence_length == candidate.sequence_length &&
               baseline.conv_state == candidate.conv_state &&
               baseline.gdn_state == candidate.gdn_state &&
               baseline.aggregate == candidate.aggregate;
  for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
       ++slot) {
    exact = exact &&
            baseline.key_cache[slot] == candidate.key_cache[slot] &&
            baseline.value_cache[slot] == candidate.value_cache[slot];
  }
  if (!exact) {
    const auto report = [prompt_token_count](
                            const std::string_view region,
                            const core::Sha256Digest& expected,
                            const core::Sha256Digest& actual) {
      if (!(expected == actual)) {
        std::cerr << "P" << prompt_token_count << " mismatch region="
                  << region << " baseline_sha256=" << expected.hex()
                  << " candidate_sha256=" << actual.hex() << '\n';
      }
    };
    if (baseline.sequence_length != candidate.sequence_length) {
      std::cerr << "P" << prompt_token_count
                << " mismatch region=sequence_length baseline="
                << baseline.sequence_length
                << " candidate=" << candidate.sequence_length << '\n';
    }
    report("conv_state", baseline.conv_state, candidate.conv_state);
    report("gdn_state", baseline.gdn_state, candidate.gdn_state);
    report("aggregate", baseline.aggregate, candidate.aggregate);
    for (std::size_t slot = 0U; slot < runtime::kRequestFullLayerCount;
         ++slot) {
      const std::size_t layer = 3U + 4U * slot;
      report("key_cache_layer_" + std::to_string(layer),
             baseline.key_cache[slot], candidate.key_cache[slot]);
      report("value_cache_layer_" + std::to_string(layer),
             baseline.value_cache[slot], candidate.value_cache[slot]);
    }
  }
  return exact;
}

[[nodiscard]] bool same_float_bits(const float& left,
                                   const float& right) noexcept {
  return std::memcmp(&left, &right, sizeof(left)) == 0;
}

[[nodiscard]] bool same_double_bits(const double& left,
                                    const double& right) noexcept {
  return std::memcmp(&left, &right, sizeof(left)) == 0;
}

[[nodiscard]] bool same_logits_statistics(
    const runtime::ReferenceStepLogits& left,
    const runtime::ReferenceStepLogits& right) noexcept {
  return left.predicted_token_id == right.predicted_token_id &&
         same_float_bits(left.chosen_logit, right.chosen_logit) &&
         same_double_bits(left.max_log_probability,
                          right.max_log_probability) &&
         same_double_bits(left.logsumexp, right.logsumexp);
}

[[nodiscard]] bool same_step_structure_and_argmax(
    const runtime::ReferenceGeneration& baseline,
    const runtime::ReferenceGeneration& candidate) {
  if (baseline.steps.size() != candidate.steps.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < baseline.steps.size(); ++index) {
    const runtime::ReferenceStepResult& left = baseline.steps[index];
    const runtime::ReferenceStepResult& right = candidate.steps[index];
    if (left.position != right.position ||
        left.input_token_id != right.input_token_id ||
        left.logits.has_value() != right.logits.has_value() ||
        left.prediction.has_value() != right.prediction.has_value()) {
      return false;
    }
    if (left.logits.has_value() &&
        left.logits->predicted_token_id !=
            right.logits->predicted_token_id) {
      return false;
    }
    if (left.prediction.has_value() &&
        left.prediction->predicted_token_id !=
            right.prediction->predicted_token_id) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool same_logits_statistics_bits(
    const runtime::ReferenceGeneration& baseline,
    const runtime::ReferenceGeneration& candidate) noexcept {
  if (baseline.steps.size() != candidate.steps.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < baseline.steps.size(); ++index) {
    const auto& left = baseline.steps[index].logits;
    const auto& right = candidate.steps[index].logits;
    if (left.has_value() != right.has_value() ||
        (left.has_value() &&
         !same_logits_statistics(*left, *right))) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool has_complete_logits_statistics(
    const runtime::ReferenceGeneration& generation) noexcept {
  std::size_t logits_steps = 0U;
  for (const runtime::ReferenceStepResult& step : generation.steps) {
    if (step.prediction.has_value()) {
      return false;
    }
    logits_steps += step.logits.has_value() ? 1U : 0U;
  }
  return logits_steps == generation.generated_token_ids.size();
}

[[nodiscard]] bool run_case(runtime::ReferenceEngine& engine,
                            const std::size_t prompt_token_count) {
  const std::string prompt = repeated_hello_prompt(prompt_token_count);
  const bool exact_full_c512_candidate =
      (prompt_token_count % kPrefillChunkTokens) == 0U;
  const bool multi_token_decode =
      prompt_token_count == kStateValidationPromptTokens;
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens =
      multi_token_decode ? kStateValidationDecodeTokens : 1U;
  if (multi_token_decode) {
    options.stop_token_id = kNonTextStopTokenId;
  }
  options.prefill_chunk_size = kPrefillChunkTokens;
  options.logits_mode =
      exact_full_c512_candidate
          ? runtime::ReferenceLogitsMode::kFullStatistics
          : runtime::ReferenceLogitsMode::kPredictedTokenOnly;

  using Policy = detail::ReferenceEnginePrefillFinalTokenPolicyForTest;
  StateSnapshot baseline_snapshot;
  StateSnapshot candidate_snapshot;

  const auto generate = [&](const bool candidate_route,
                            StateSnapshot& snapshot) {
    const Policy policy =
        !candidate_route
            ? Policy::kLegacyScalarFinalStep
            : exact_full_c512_candidate ? Policy::kExactFullC512Tile
                                    : Policy::kAllPromptTiles;
    const ScopedFinalTokenPolicy policy_scope(policy);
    if (exact_full_c512_candidate) {
      const ScopedGenerateReturnSnapshot snapshot_scope(snapshot);
      return engine.generate(prompt, options);
    }
    return engine.generate(prompt, options);
  };
  const char* const candidate_first_value =
      std::getenv(kCandidateFirstEnvironment);
  const bool candidate_first =
      candidate_first_value != nullptr &&
      std::string_view(candidate_first_value) == "1";
  runtime::ReferenceGenerateResult baseline;
  runtime::ReferenceGenerateResult candidate;
  if (candidate_first) {
    candidate = generate(true, candidate_snapshot);
    baseline = generate(false, baseline_snapshot);
  } else {
    baseline = generate(false, baseline_snapshot);
    candidate = generate(true, candidate_snapshot);
  }
  if (!baseline) {
    std::cerr << "P" << prompt_token_count
              << " baseline generation failed: ";
    print_diagnostic(baseline.diagnostic);
    return false;
  }
  if (!candidate) {
    std::cerr << "P" << prompt_token_count
              << " candidate generation failed: ";
    print_diagnostic(candidate.diagnostic);
    return false;
  }

  const runtime::ReferenceGeneration& left = *baseline.value;
  const runtime::ReferenceGeneration& right = *candidate.value;
  const std::size_t expected_baseline_executions =
      detail::prefix_execution_count(prompt_token_count - 1U,
                                     kPrefillChunkTokens);
  const std::size_t expected_candidate_executions =
      detail::prefix_execution_count(prompt_token_count,
                                     kPrefillChunkTokens);
  const bool transcript_structure_and_argmax_exact =
      same_step_structure_and_argmax(left, right);
  const bool complete_logits_statistics_present =
      !exact_full_c512_candidate ||
      (has_complete_logits_statistics(left) &&
       has_complete_logits_statistics(right));
  const bool logits_statistics_exact =
      !exact_full_c512_candidate ||
      (complete_logits_statistics_present &&
       same_logits_statistics_bits(left, right));
  const bool generation_length_exact =
      !multi_token_decode ||
      (left.generated_token_ids.size() == kStateValidationDecodeTokens &&
       right.generated_token_ids.size() == kStateValidationDecodeTokens &&
       left.stop_reason == runtime::ReferenceStopReason::kMaxNewTokens &&
       right.stop_reason == runtime::ReferenceStopReason::kMaxNewTokens);
  const std::size_t expected_sequence_length =
      left.generated_token_ids.empty()
          ? 0U
          : prompt_token_count + left.generated_token_ids.size() - 1U;
  const bool baseline_snapshot_valid =
      !exact_full_c512_candidate ||
      valid_snapshot(baseline_snapshot, expected_sequence_length,
                     "baseline", prompt_token_count);
  const bool candidate_snapshot_valid =
      !exact_full_c512_candidate ||
      valid_snapshot(candidate_snapshot, expected_sequence_length,
                     "candidate", prompt_token_count);
  const bool snapshots_equal =
      !exact_full_c512_candidate ||
      (baseline_snapshot_valid && candidate_snapshot_valid &&
       same_snapshots(baseline_snapshot, candidate_snapshot,
                      prompt_token_count));
  const bool state_exact =
      baseline_snapshot_valid && candidate_snapshot_valid && snapshots_equal;
  // The exact-full-C512 scheduling candidate is deliberately retained as a
  // negative state oracle.  Matching argmax/text is insufficient: until the
  // complete public logits statistics and persistent request state both
  // match, production must remain on the legacy scalar final step.
  const bool candidate_rejection_confirmed =
      exact_full_c512_candidate && baseline_snapshot_valid &&
      candidate_snapshot_valid && complete_logits_statistics_present &&
      (!snapshots_equal || !logits_statistics_exact);
  const bool qualification_boundary_satisfied =
      exact_full_c512_candidate
          ? candidate_rejection_confirmed
          : logits_statistics_exact && state_exact;
  const bool exact =
      left.prompt_token_ids.size() == prompt_token_count &&
      right.prompt_token_ids.size() == prompt_token_count &&
      left.prompt_token_ids == right.prompt_token_ids &&
      left.generated_token_ids == right.generated_token_ids &&
      left.generated_text == right.generated_text &&
      left.stop_reason == right.stop_reason &&
      transcript_structure_and_argmax_exact && generation_length_exact &&
      qualification_boundary_satisfied &&
      !left.all_prompt_tokens_prefilled_by_tiles &&
      right.all_prompt_tokens_prefilled_by_tiles &&
      left.timing.prefix_execution_milliseconds.size() ==
          expected_baseline_executions &&
      right.timing.prefix_execution_milliseconds.size() ==
          expected_candidate_executions;

  std::cout << "P" << prompt_token_count
            << " baseline_prefix_ms="
            << left.timing.prompt_prefill_milliseconds -
                   left.timing.finish_prefill_milliseconds
            << " baseline_finish_ms="
            << left.timing.finish_prefill_milliseconds
            << " baseline_ttft_ms="
            << left.timing.time_to_first_token_milliseconds
            << " candidate_prefix_ms="
            << right.timing.prompt_prefill_milliseconds -
                   right.timing.finish_prefill_milliseconds
            << " candidate_finish_ms="
            << right.timing.finish_prefill_milliseconds
            << " candidate_ttft_ms="
            << right.timing.time_to_first_token_milliseconds
            << " baseline_active="
            << (left.all_prompt_tokens_prefilled_by_tiles ? 1 : 0)
            << " candidate_active="
            << (right.all_prompt_tokens_prefilled_by_tiles ? 1 : 0)
            << " exact_full_c512_candidate="
            << (exact_full_c512_candidate ? 1 : 0)
            << " order=" << (candidate_first ? "candidate-baseline"
                                               : "baseline-candidate")
            << " baseline_steps=" << left.steps.size()
            << " candidate_steps=" << right.steps.size()
            << " requested_new_tokens=" << options.max_new_tokens
            << " generated_tokens=" << right.generated_token_ids.size()
            << " final_position="
            << (right.steps.empty()
                    ? runtime::kReferenceNoLayer
                    : right.steps.back().position)
            << " generated_id="
            << (right.generated_token_ids.empty()
                    ? runtime::kReferenceVocabularySize
                    : right.generated_token_ids.front())
            << " logits_statistics_present="
            << (complete_logits_statistics_present ? "true" : "false")
            << " logits_statistics_exact="
            << (logits_statistics_exact ? "true" : "false")
            << " state_snapshot="
            << (exact_full_c512_candidate
                    ? "full_persistent_and_used_kv"
                    : "not_required")
            << " baseline_sequence_length="
            << (exact_full_c512_candidate
                    ? baseline_snapshot.sequence_length
                    : 0U)
            << " candidate_sequence_length="
            << (exact_full_c512_candidate
                    ? candidate_snapshot.sequence_length
                    : 0U)
            << " state_sha256="
            << (exact_full_c512_candidate
                    ? candidate_snapshot.aggregate.hex()
                    : std::string("not_required"))
            << " state_exact=" << (state_exact ? "true" : "false")
            << " candidate_rejection_confirmed="
            << (candidate_rejection_confirmed ? "true" : "false")
            << " exact=" << (exact ? "true" : "false") << '\n';
  return exact;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr
        << "usage: q3x_reference_prefill_all_prompt_tokens_admission_e2e_test "
           "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }

  runtime::ReferenceEngineOptions options;
  options.request_options.prefill_chunk_size = kPrefillChunkTokens;
  options.request_options.max_sequence_length =
      static_cast<std::uint32_t>(kPromptTokenCounts.back());
  options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(
          std::filesystem::path(model_directory), options);
  if (!created) {
    std::cerr << "engine creation failed: ";
    print_diagnostic(created.diagnostic);
    return 1;
  }

  bool exact = true;
  const char* const selected_value =
      std::getenv(kSelectedPromptTokensEnvironment);
  if (selected_value != nullptr && selected_value[0] != '\0') {
    std::size_t selected = 0U;
    const char* const selected_end =
        selected_value + std::string_view(selected_value).size();
    const auto parsed =
        std::from_chars(selected_value, selected_end, selected);
    if (parsed.ec != std::errc{} || parsed.ptr != selected_end ||
        selected < kMinimumPromptTokens ||
        selected > kPromptTokenCounts.back()) {
      std::cerr << "invalid " << kSelectedPromptTokensEnvironment << '\n';
      return 2;
    }
    return run_case(*created.value, selected) ? 0 : 1;
  }
  for (const std::size_t prompt_token_count : kPromptTokenCounts) {
    exact = run_case(*created.value, prompt_token_count) && exact;
  }
  return exact ? 0 : 1;
}
