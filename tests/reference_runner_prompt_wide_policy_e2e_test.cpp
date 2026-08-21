#include "q3x/core/sha256.h"
#include "q3x/runtime/reference_engine.h"

#include "reference_engine_final_token_policy_internal.h"
#include "reference_runner_gdn_chunk64_native_admission.h"
#include "reference_runner_prompt_wide_policy_internal.h"

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
#include <utility>
#include <vector>

namespace {

namespace core = q3x::core;
namespace runtime = q3x::runtime;
namespace engine_detail = q3x::runtime::reference_engine_detail;
namespace runner_detail = q3x::runtime::reference_runner_detail;

constexpr char kSelectedPromptTokensEnvironment[] =
    "Q3X_E2E_PREFILL_PROMPT_TOKENS";
constexpr std::uint32_t kPrefillChunkTokens =
    runtime::kMaximumRequestPrefillChunkSize;
constexpr std::uint32_t kP4096DecodeTokens = 16U;
constexpr std::uint32_t kP8192DecodeTokens = 1U;
// The image-pad token cannot be emitted by the text-only request. This keeps
// P4096 alive for all 16 requested tokens and therefore exercises 15 Decode
// state transitions after the first generated token.
constexpr std::uint32_t kNonTextStopTokenId = 248'056U;
constexpr std::size_t kHashCopyChunkBytes = 4U * 1024U * 1024U;
constexpr std::size_t kKvHeadCount = 4U;
constexpr std::size_t kKvHeadDimension = 256U;
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
static_assert(kPrefillChunkTokens == 512U);
static_assert(runtime::kRequestFullLayerCount == 16U);

struct PromptCase {
  std::size_t prompt_tokens = 0U;
  std::size_t generated_tokens = 0U;
  std::size_t embedding_hits = 0U;
  std::size_t attention_hits = 0U;
};

// Legacy final-token policy submits P-1 tokens through Prefix. P514 is the
// runner-level C512+C1 fallback witness: C512 enters the layer-segment route,
// while C1 delegates to scalar step() and contributes no prompt-wide hit.
// Prompt-wide Attention therefore launches only for the 16 full-Attention
// layers in C512.  Longer legacy Prefix requests use the canonical
// C512/C256/C64/C32/tail decomposition rather than arbitrary C511 tails:
// P4096 -> 7*C512+C256+3*C64+C32+C31 (13 executions), and P8192 adds
// another 8*C512 (21 executions total).
constexpr std::array<PromptCase, 3U> kPromptCases{{
    {514U, 1U, 1U, 16U},
    {4'096U, kP4096DecodeTokens, 13U, 208U},
    {8'192U, kP8192DecodeTokens, 21U, 336U},
}};
static_assert(runtime::reference_engine_detail::prefix_execution_count(
                  4'095U, kPrefillChunkTokens) == 13U);
static_assert(runtime::reference_engine_detail::prefix_execution_count(
                  8'191U, kPrefillChunkTokens) == 21U);

using PromptWidePolicy =
    runner_detail::ReferenceRunnerPromptWidePolicyForTest;

struct RouteCase {
  std::string_view name;
  PromptWidePolicy policy = PromptWidePolicy::kLegacy;
  bool expect_embedding_hits = false;
  bool expect_attention_hits = false;
};

constexpr std::array<RouteCase, 4U> kRoutes{{
    {"Baseline", PromptWidePolicy::kLegacy, false, false},
    {"EmbeddingOnly", PromptWidePolicy::kEmbeddingOnly, true, false},
    {"AttentionOnly", PromptWidePolicy::kFullAttentionPreprocessOnly, false,
     true},
    {"Both", PromptWidePolicy::kEmbeddingAndFullAttentionPreprocess, true,
     true},
}};

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

struct RouteSample {
  runtime::ReferenceGenerateResult generation;
  StateSnapshot snapshot;
  std::size_t stale_embedding_hits = 0U;
  std::size_t stale_attention_hits = 0U;
  std::size_t embedding_hits = 0U;
  std::size_t attention_hits = 0U;
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

class ScopedPromptWidePolicy final {
 public:
  explicit ScopedPromptWidePolicy(const PromptWidePolicy policy) noexcept
      : previous_(
            runner_detail::exchange_reference_runner_prompt_wide_policy_for_test(
                policy)) {}

  ~ScopedPromptWidePolicy() {
    (void)runner_detail::
        exchange_reference_runner_prompt_wide_policy_for_test(previous_);
  }

  ScopedPromptWidePolicy(const ScopedPromptWidePolicy&) = delete;
  ScopedPromptWidePolicy& operator=(const ScopedPromptWidePolicy&) = delete;

 private:
  PromptWidePolicy previous_ = PromptWidePolicy::kProductionDefault;
};

class ScopedFinalTokenPolicy final {
 public:
  using Policy =
      engine_detail::ReferenceEnginePrefillFinalTokenPolicyForTest;

  explicit ScopedFinalTokenPolicy(const Policy policy) noexcept
      : previous_(
            engine_detail::
                exchange_reference_engine_prefill_final_token_policy_for_test(
                    policy)) {}

  ~ScopedFinalTokenPolicy() {
    (void)engine_detail::
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

[[nodiscard]] bool valid_transcript_and_logits(
    const runtime::ReferenceGeneration& generation,
    const std::size_t prompt_token_count,
    const std::size_t expected_generated_tokens,
    const std::string_view route) {
  const std::size_t expected_step_count =
      prompt_token_count + expected_generated_tokens - 1U;
  const bool cardinality_valid =
      generation.prompt_token_ids.size() == prompt_token_count &&
      generation.generated_token_ids.size() == expected_generated_tokens &&
      generation.steps.size() == expected_step_count;
  bool valid = cardinality_valid &&
      generation.stop_reason == runtime::ReferenceStopReason::kMaxNewTokens &&
      !generation.all_prompt_tokens_prefilled_by_tiles &&
      !generation.single_arbitrary_prefill_tiles;
  if (!cardinality_valid) {
    std::cerr << "P" << prompt_token_count << ' ' << route
              << " transcript/logits cardinality failed: prompt_tokens="
              << generation.prompt_token_ids.size()
              << " generated_tokens=" << generation.generated_token_ids.size()
              << " steps=" << generation.steps.size() << '\n';
    return false;
  }
  for (std::size_t index = 0U; index < generation.steps.size(); ++index) {
    const runtime::ReferenceStepResult& step = generation.steps[index];
    const std::uint32_t expected_position =
        static_cast<std::uint32_t>(index);
    const std::uint32_t expected_input_token =
        index < prompt_token_count
            ? generation.prompt_token_ids[index]
            : generation.generated_token_ids[index - prompt_token_count];
    const bool logits_step = index + 1U >= prompt_token_count;
    valid = valid && step.position == expected_position &&
            step.input_token_id == expected_input_token &&
            step.logits.has_value() == logits_step &&
            !step.prediction.has_value();
    if (logits_step && step.logits.has_value()) {
      const std::size_t generated_index =
          index - (prompt_token_count - 1U);
      valid = valid &&
              step.logits->predicted_token_id ==
                  generation.generated_token_ids[generated_index];
    }
  }
  if (!valid) {
    std::cerr << "P" << prompt_token_count << ' ' << route
              << " transcript/logits failed: prompt_tokens="
              << generation.prompt_token_ids.size()
              << " generated_tokens=" << generation.generated_token_ids.size()
              << " steps=" << generation.steps.size()
              << " stop_reason="
              << static_cast<unsigned int>(generation.stop_reason)
              << " all_prompt_tiles="
              << (generation.all_prompt_tokens_prefilled_by_tiles ? 1 : 0)
              << " single_arbitrary_tiles="
              << (generation.single_arbitrary_prefill_tiles ? 1 : 0) << '\n';
  }
  return valid;
}

[[nodiscard]] bool valid_hits(const RouteSample& sample,
                              const RouteCase& route,
                              const PromptCase& prompt_case) {
  const std::size_t expected_embedding_hits =
      route.expect_embedding_hits ? prompt_case.embedding_hits : 0U;
  const std::size_t expected_attention_hits =
      route.expect_attention_hits ? prompt_case.attention_hits : 0U;
  const bool valid =
      sample.stale_embedding_hits == 0U &&
      sample.stale_attention_hits == 0U &&
      sample.embedding_hits == expected_embedding_hits &&
      sample.attention_hits == expected_attention_hits;
  if (!valid) {
    const std::size_t prefix_executions =
        sample.generation && sample.generation.value.has_value()
            ? sample.generation.value->timing.prefix_execution_milliseconds
                  .size()
            : 0U;
    std::cerr << "P" << prompt_case.prompt_tokens << ' ' << route.name
              << " route-hit failure: stale_embedding="
              << sample.stale_embedding_hits
              << " stale_attention=" << sample.stale_attention_hits
              << " embedding=" << sample.embedding_hits
              << " attention=" << sample.attention_hits
              << " expected_embedding=" << expected_embedding_hits
              << " expected_attention=" << expected_attention_hits
              << " prefix_executions=" << prefix_executions << '\n';
  }
  return valid;
}

[[nodiscard]] bool same_snapshots(
    const StateSnapshot& baseline, const StateSnapshot& candidate,
    const std::string_view candidate_route,
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
    const auto report = [prompt_token_count, candidate_route](
                            const std::string_view region,
                            const core::Sha256Digest& expected,
                            const core::Sha256Digest& actual) {
      if (!(expected == actual)) {
        std::cerr << "P" << prompt_token_count << ' ' << candidate_route
                  << " state mismatch region=" << region
                  << " baseline_sha256=" << expected.hex()
                  << " candidate_sha256=" << actual.hex() << '\n';
      }
    };
    if (baseline.sequence_length != candidate.sequence_length) {
      std::cerr << "P" << prompt_token_count << ' ' << candidate_route
                << " state mismatch region=sequence_length baseline="
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

[[nodiscard]] bool same_generation(
    const runtime::ReferenceGeneration& baseline,
    const runtime::ReferenceGeneration& candidate,
    const std::string_view candidate_route,
    const std::size_t prompt_token_count) {
  bool exact = baseline.rendered_prompt == candidate.rendered_prompt &&
               baseline.prompt_token_ids == candidate.prompt_token_ids &&
               baseline.generated_token_ids == candidate.generated_token_ids &&
               baseline.generated_text == candidate.generated_text &&
               baseline.stop_reason == candidate.stop_reason &&
               baseline.steps.size() == candidate.steps.size();
  const std::size_t shared_steps =
      std::min(baseline.steps.size(), candidate.steps.size());
  for (std::size_t index = 0U; index < shared_steps; ++index) {
    const runtime::ReferenceStepResult& left = baseline.steps[index];
    const runtime::ReferenceStepResult& right = candidate.steps[index];
    const bool structure_exact =
        left.position == right.position &&
        left.input_token_id == right.input_token_id &&
        left.logits.has_value() == right.logits.has_value() &&
        left.prediction.has_value() == right.prediction.has_value() &&
        left.timing.has_value() == right.timing.has_value();
    const bool logits_exact =
        (!left.logits.has_value() && !right.logits.has_value()) ||
        (left.logits.has_value() && right.logits.has_value() &&
         same_logits_statistics(*left.logits, *right.logits));
    const bool prediction_exact =
        !left.prediction.has_value() ||
        (right.prediction.has_value() &&
         left.prediction->predicted_token_id ==
             right.prediction->predicted_token_id);
    if (!structure_exact || !logits_exact || !prediction_exact) {
      std::cerr << "P" << prompt_token_count << ' ' << candidate_route
                << " transcript/logits mismatch step=" << index
                << " structure_exact=" << (structure_exact ? 1 : 0)
                << " logits_exact=" << (logits_exact ? 1 : 0)
                << " prediction_exact=" << (prediction_exact ? 1 : 0)
                << '\n';
      exact = false;
    }
  }
  if (!exact) {
    std::cerr << "P" << prompt_token_count << ' ' << candidate_route
              << " generation mismatch: prompt_ids="
              << (baseline.prompt_token_ids == candidate.prompt_token_ids)
              << " generated_ids="
              << (baseline.generated_token_ids ==
                  candidate.generated_token_ids)
              << " rendered_prompt="
              << (baseline.rendered_prompt == candidate.rendered_prompt)
              << " generated_text="
              << (baseline.generated_text == candidate.generated_text)
              << " baseline_steps=" << baseline.steps.size()
              << " candidate_steps=" << candidate.steps.size() << '\n';
  }
  return exact;
}

[[nodiscard]] RouteSample run_route(
    runtime::ReferenceEngine& engine, const std::string& prompt,
    const runtime::ReferenceGenerateOptions& options,
    const RouteCase& route) {
  RouteSample sample;
  sample.stale_embedding_hits =
      runner_detail::
          exchange_reference_runner_prompt_wide_embedding_launch_hits_for_test(
              0U);
  sample.stale_attention_hits =
      runner_detail::
          exchange_reference_runner_prompt_wide_attention_launch_hits_for_test(
              0U);
  {
    const ScopedPromptWidePolicy policy_scope(route.policy);
    const ScopedGenerateReturnSnapshot snapshot_scope(sample.snapshot);
    sample.generation = engine.generate(prompt, options);
  }
  sample.embedding_hits =
      runner_detail::
          exchange_reference_runner_prompt_wide_embedding_launch_hits_for_test(
              0U);
  sample.attention_hits =
      runner_detail::
          exchange_reference_runner_prompt_wide_attention_launch_hits_for_test(
              0U);
  return sample;
}

[[nodiscard]] bool valid_sample(
    const RouteSample& sample, const RouteCase& route,
    const PromptCase& prompt_case) {
  if (!sample.generation) {
    std::cerr << "P" << prompt_case.prompt_tokens << ' ' << route.name
              << " generation failed: ";
    print_diagnostic(sample.generation.diagnostic);
    return false;
  }
  const std::size_t expected_sequence_length =
      prompt_case.prompt_tokens + prompt_case.generated_tokens - 1U;
  const bool snapshot_valid =
      valid_snapshot(sample.snapshot, expected_sequence_length, route.name,
                     prompt_case.prompt_tokens);
  const bool hits_valid = valid_hits(sample, route, prompt_case);
  const bool transcript_valid =
      valid_transcript_and_logits(*sample.generation.value,
                                  prompt_case.prompt_tokens,
                                  prompt_case.generated_tokens, route.name);
  return snapshot_valid && hits_valid && transcript_valid;
}

[[nodiscard]] bool run_case(runtime::ReferenceEngine& engine,
                            const PromptCase& prompt_case) {
  const std::size_t prompt_token_count = prompt_case.prompt_tokens;
  const std::string prompt = repeated_hello_prompt(prompt_token_count);
  runtime::ReferenceGenerateOptions options;
  options.max_new_tokens =
      static_cast<std::uint32_t>(prompt_case.generated_tokens);
  options.stop_token_id = kNonTextStopTokenId;
  options.prefill_chunk_size = kPrefillChunkTokens;
  options.logits_mode = runtime::ReferenceLogitsMode::kFullStatistics;

  RouteSample baseline = run_route(engine, prompt, options, kRoutes.front());
  if (!valid_sample(baseline, kRoutes.front(), prompt_case)) {
    return false;
  }
  const runtime::ReferenceGeneration& baseline_generation =
      *baseline.generation.value;
  std::cout << "P" << prompt_token_count << " route=" << kRoutes.front().name
            << " embedding_hits=" << baseline.embedding_hits
            << " attention_hits=" << baseline.attention_hits
            << " sequence_length=" << baseline.snapshot.sequence_length
            << " generated_tokens="
            << baseline_generation.generated_token_ids.size()
            << " state_sha256=" << baseline.snapshot.aggregate.hex()
            << " exact=true\n";

  bool exact = true;
  for (std::size_t route_index = 1U; route_index < kRoutes.size();
       ++route_index) {
    const RouteCase& route = kRoutes[route_index];
    RouteSample candidate = run_route(engine, prompt, options, route);
    if (!valid_sample(candidate, route, prompt_case)) {
      return false;
    }
    const bool state_exact =
        same_snapshots(baseline.snapshot, candidate.snapshot, route.name,
                       prompt_token_count);
    const bool generation_exact =
        same_generation(baseline_generation, *candidate.generation.value,
                        route.name, prompt_token_count);
    const bool route_exact = state_exact && generation_exact;
    std::cout << "P" << prompt_token_count << " route=" << route.name
              << " embedding_hits=" << candidate.embedding_hits
              << " attention_hits=" << candidate.attention_hits
              << " sequence_length=" << candidate.snapshot.sequence_length
              << " generated_tokens="
              << candidate.generation.value->generated_token_ids.size()
              << " state_sha256=" << candidate.snapshot.aggregate.hex()
              << " state_exact=" << (state_exact ? "true" : "false")
              << " generation_exact="
              << (generation_exact ? "true" : "false")
              << " exact=" << (route_exact ? "true" : "false") << '\n';
    exact = route_exact && exact;
  }
  return exact;
}

[[nodiscard]] bool selected_prompt_token_count(
    std::size_t& selected) noexcept {
  const char* const selected_value =
      std::getenv(kSelectedPromptTokensEnvironment);
  if (selected_value == nullptr || selected_value[0] == '\0') {
    return false;
  }
  const char* const selected_end =
      selected_value + std::string_view(selected_value).size();
  const auto parsed =
      std::from_chars(selected_value, selected_end, selected);
  if (parsed.ec != std::errc{} || parsed.ptr != selected_end ||
      std::find_if(kPromptCases.begin(), kPromptCases.end(),
                   [selected](const PromptCase& prompt_case) {
                     return prompt_case.prompt_tokens == selected;
                   }) == kPromptCases.end()) {
    selected = 0U;
  }
  return true;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc > 2) {
    std::cerr << "usage: q3x_reference_runner_prompt_wide_policy_e2e_test "
                 "[MODEL_DIR|-]\n";
    return 2;
  }
  const std::string model_directory = model_directory_from(argc, argv);
  if (model_directory.empty()) {
    std::cout << "SKIP: set Q3X_E2E_MODEL_DIR to the pinned model directory\n";
    return 77;
  }

  std::size_t selected = 0U;
  const bool has_selection = selected_prompt_token_count(selected);
  if (has_selection && selected == 0U) {
    std::cerr << "invalid " << kSelectedPromptTokensEnvironment
              << "; expected 514, 4096, or 8192\n";
    return 2;
  }

  runtime::ReferenceEngineOptions options;
  options.request_options.prefill_chunk_size = kPrefillChunkTokens;
  options.request_options.max_sequence_length =
      kPromptCases.back().prompt_tokens;
  options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(
          std::filesystem::path(model_directory), options);
  if (!created) {
    std::cerr << "engine creation failed: ";
    print_diagnostic(created.diagnostic);
    return 1;
  }

  using FinalPolicy =
      engine_detail::ReferenceEnginePrefillFinalTokenPolicyForTest;
  const ScopedFinalTokenPolicy final_token_policy(
      FinalPolicy::kLegacyScalarFinalStep);

  if (has_selection) {
    const auto selected_case =
        std::find_if(kPromptCases.begin(), kPromptCases.end(),
                     [selected](const PromptCase& prompt_case) {
                       return prompt_case.prompt_tokens == selected;
                     });
    return run_case(*created.value, *selected_case) ? 0 : 1;
  }
  bool exact = true;
  for (const PromptCase& prompt_case : kPromptCases) {
    exact = run_case(*created.value, prompt_case) && exact;
  }
  return exact ? 0 : 1;
}
