#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/runtime/reference_engine.h"
#include "reference_engine_final_token_policy_internal.h"
#include "reference_runner_gdn_chunk64_native_admission.h"
#if __has_include("reference_runner_terminal_prefix_internal.h")
#include "reference_runner_terminal_prefix_internal.h"
#define Q3X_CAPTURE_HAS_LIVENESS 1
#endif
#if __has_include("reference_runner_exact_attention_score_feed_internal.h")
#include "reference_runner_exact_attention_score_feed_internal.h"
#define Q3X_CAPTURE_HAS_SCORE_FEED 1
#endif

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {
namespace core = q3x::core;
namespace rt = q3x::runtime;
namespace detail = rt::reference_runner_detail;
constexpr std::size_t kOutputs = 16U;
constexpr std::size_t kCopyBytes = 4U * 1024U * 1024U;
constexpr std::size_t kHiddenBytes = rt::kReferenceHiddenSize * 2U;
constexpr std::size_t kLogitsBytes = rt::kReferenceVocabularySize * 2U;

struct ScalarSnapshot {
  std::uint32_t sequence = 0U;
  core::Sha256Digest hidden, residual, logits;
  std::uint32_t argmax = 0U;
  float chosen_logit = 0.0F;
  double logsumexp = 0.0;
};
struct FullSnapshot {
  ScalarSnapshot scalar;
  core::Sha256Digest conv, gdn;
  std::array<core::Sha256Digest, rt::kRequestFullLayerCount> key, value;
};
struct Capture {
  std::vector<std::uint8_t> scratch = std::vector<std::uint8_t>(kCopyBytes);
  std::vector<std::uint16_t> logits =
      std::vector<std::uint16_t>(rt::kReferenceVocabularySize);
  std::array<ScalarSnapshot, kOutputs> steps{};
  FullSnapshot prefill, returned;
  std::size_t step_calls = 0U, return_calls = 0U;
  std::uint32_t prompt_tokens = 0U;
  const char* error = nullptr;
  int cuda_error = 0;
};

// Same bounded streaming SHA-256 as the all-prompt state oracle. Only live
// rows are copied; no complete arena or dead terminal-prefix scratch is read.
bool hash_device_bytes(const std::uint8_t* source, std::size_t bytes,
                       Capture& capture, core::Sha256Digest& digest) noexcept {
  core::Sha256 hash;
  for (std::size_t offset = 0U; offset < bytes;) {
    const std::size_t count = std::min(capture.scratch.size(), bytes - offset);
    const auto status = cudaMemcpy(capture.scratch.data(), source + offset,
                                   count, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      capture.cuda_error = static_cast<int>(status);
      return false;
    }
    if (!hash.update(capture.scratch.data(), count)) return false;
    offset += count;
  }
  digest = hash.finalize();
  return true;
}
const std::uint8_t* region(const rt::RequestState& state,
                           const rt::RequestRegion& value,
                           std::size_t bytes) noexcept {
  if (!state || state.arena_data() == nullptr || bytes == 0U ||
      bytes > value.byte_size || value.arena_offset > state.arena_bytes() ||
      value.byte_size > state.arena_bytes() - value.arena_offset) return nullptr;
  return static_cast<const std::uint8_t*>(state.arena_data()) + value.arena_offset;
}
bool hash_region(const rt::RequestState& state, const rt::RequestRegion& value,
                 std::size_t bytes, Capture& capture,
                 core::Sha256Digest& digest) noexcept {
  const auto* source = region(state, value, bytes);
  return source != nullptr && hash_device_bytes(source, bytes, capture, digest);
}
float bf16(std::uint16_t bits) noexcept {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
  float result;
  std::memcpy(&result, &word, sizeof(result));
  return result;
}
bool capture_scalar(const rt::RequestState& state, Capture& capture,
                    ScalarSnapshot& out) noexcept {
  const auto& plan = state.plan();
  out.sequence = state.sequence_length();
  if (state.memory_profile() != rt::RequestMemoryProfile::kLegacyC512 ||
      out.sequence == 0U || out.sequence > state.max_sequence_length() ||
      !hash_region(state, plan.hidden_bf16[1], kHiddenBytes, capture, out.hidden) ||
      !hash_region(state, plan.hidden_bf16[0], kHiddenBytes, capture, out.residual))
    return false;
  // Scalar SM87 predicted-only logits start at fp32_scratch; the argmax
  // workspace is strictly after the complete BF16 vocabulary vector.
  const auto* source = region(state, plan.fp32_scratch, kLogitsBytes);
  if (source == nullptr) return false;
  const auto status = cudaMemcpy(capture.logits.data(), source, kLogitsBytes,
                                 cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) {
    capture.cuda_error = static_cast<int>(status);
    return false;
  }
  core::Sha256 hash;
  if (!hash.update(capture.logits.data(), kLogitsBytes)) return false;
  out.logits = hash.finalize();
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0U; index < capture.logits.size(); ++index) {
    const float value = bf16(capture.logits[index]);
    if (!std::isfinite(value)) return false;
    if (value > maximum) {
      maximum = value;
      out.argmax = static_cast<std::uint32_t>(index);
    }
  }
  double denominator = 0.0;
  for (const auto bits : capture.logits)
    denominator += std::exp(static_cast<double>(bf16(bits)) - maximum);
  out.chosen_logit = maximum;
  out.logsumexp = static_cast<double>(maximum) + std::log(denominator);
  return true;
}
bool capture_full(const rt::RequestState& state, Capture& capture,
                  FullSnapshot& out) noexcept {
  const auto& plan = state.plan();
  if (!capture_scalar(state, capture, out.scalar) ||
      plan.conv_state.byte_size != rt::kRequestConvStateBytes ||
      plan.gdn_state.byte_size != rt::kRequestGdnStateBytes ||
      !hash_region(state, plan.conv_state, rt::kRequestConvStateBytes,
                   capture, out.conv) ||
      !hash_region(state, plan.gdn_state, rt::kRequestGdnStateBytes,
                   capture, out.gdn)) return false;
  const std::size_t used = static_cast<std::size_t>(out.scalar.sequence) *
                           4U * 256U * sizeof(std::uint16_t);
  for (std::size_t slot = 0U; slot < rt::kRequestFullLayerCount; ++slot) {
    if (plan.key_cache[slot].element_size_bytes != sizeof(std::uint16_t) ||
        plan.value_cache[slot].element_size_bytes != sizeof(std::uint16_t) ||
        !hash_region(state, plan.key_cache[slot], used, capture, out.key[slot]) ||
        !hash_region(state, plan.value_cache[slot], used, capture, out.value[slot]))
      return false;
  }
  return true;
}
void step_hook(const rt::RequestState& state, void* context) noexcept {
  auto& capture = *static_cast<Capture*>(context);
  const std::size_t index = capture.step_calls++;
  if (capture.error != nullptr) return;
  if (index >= kOutputs || state.sequence_length() != capture.prompt_tokens + index) {
    capture.error = "step_sequence_or_count";
    return;
  }
  if (!capture_scalar(state, capture, capture.steps[index]) ||
      (index == 0U && !capture_full(state, capture, capture.prefill)))
    capture.error = "step_snapshot";
}
void return_hook(const rt::RequestState& state, void* context) noexcept {
  auto& capture = *static_cast<Capture*>(context);
  if (++capture.return_calls != 1U || capture.step_calls != kOutputs ||
      state.sequence_length() != capture.prompt_tokens + kOutputs - 1U) {
    capture.error = "return_sequence_or_count";
    return;
  }
  if (capture.error == nullptr && !capture_full(state, capture, capture.returned))
    capture.error = "return_snapshot";
}
std::string quoted(std::string_view input) {
  static constexpr char digits[] = "0123456789abcdef";
  std::string out = "\"";
  for (const unsigned char c : input) {
    if (c == '"' || c == '\\') { out += '\\'; out += static_cast<char>(c); }
    else if (c < 0x20U) {
      out += "\\u00"; out += digits[c >> 4U]; out += digits[c & 15U];
    } else out += static_cast<char>(c);
  }
  return out + '"';
}
void write_scalar(std::ostream& out, const ScalarSnapshot& value) {
  out << "{\"sequence_length\":" << value.sequence
      << ",\"normalized_hidden_sha256\":" << quoted(value.hidden.hex())
      << ",\"residual_sha256\":" << quoted(value.residual.hex())
      << ",\"full_bf16_logits_sha256\":" << quoted(value.logits.hex())
      << ",\"host_derived_argmax\":" << value.argmax
      << ",\"host_derived_chosen_logit\":" << value.chosen_logit
      << ",\"host_derived_logsumexp\":" << value.logsumexp << '}';
}
void write_full(std::ostream& out, const FullSnapshot& value) {
  out << "{\"scalar\":"; write_scalar(out, value.scalar);
  out << ",\"conv_sha256\":" << quoted(value.conv.hex())
      << ",\"gdn_sha256\":" << quoted(value.gdn.hex()) << ",\"kv\":[";
  for (std::size_t slot = 0U; slot < value.key.size(); ++slot) {
    if (slot != 0U) out << ',';
    out << "{\"layer\":" << 4U * slot + 3U << ",\"key_sha256\":"
        << quoted(value.key[slot].hex()) << ",\"value_sha256\":"
        << quoted(value.value[slot].hex()) << '}';
  }
  out << "]}";
}
void write_ids(std::ostream& out, const std::vector<std::uint32_t>& ids) {
  out << '[';
  for (std::size_t i = 0U; i < ids.size(); ++i) { if (i != 0U) out << ','; out << ids[i]; }
  out << ']';
}
std::size_t parse_count(std::string_view value) {
  std::size_t result = 0U;
  const auto parsed = std::from_chars(value.data(), value.data() + value.size(), result);
  if (parsed.ec != std::errc{} || parsed.ptr != value.data() + value.size())
    throw std::runtime_error("invalid --prompt-tokens");
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cerr << "usage: MODEL REQUEST_JSON OUTPUT_JSON [--prompt-tokens N] "
                 "[--variant baseline|liveness|score-feed|combined]\n";
    return 2;
  }
  try {
    std::size_t selected = 0U;
    std::string variant = "baseline";
    for (int i = 4; i < argc; i += 2) {
      if (i + 1 >= argc) throw std::runtime_error("missing option value");
      if (std::string_view(argv[i]) == "--prompt-tokens") selected = parse_count(argv[i + 1]);
      else if (std::string_view(argv[i]) == "--variant") variant = argv[i + 1];
      else throw std::runtime_error("unknown option");
    }
    const bool liveness = variant == "liveness" || variant == "combined";
    const bool score_feed = variant == "score-feed" || variant == "combined";
    if (variant != "baseline" && !liveness && !score_feed)
      throw std::runtime_error("unknown variant");
#if !defined(Q3X_CAPTURE_HAS_LIVENESS)
    if (liveness) throw std::runtime_error("liveness unavailable in this source");
#endif
#if !defined(Q3X_CAPTURE_HAS_SCORE_FEED)
    if (score_feed) throw std::runtime_error("score-feed unavailable in this source");
#endif
    if (std::filesystem::file_size(argv[2]) > 8U * 1024U * 1024U)
      throw std::runtime_error("request too large");
    std::ifstream input(argv[2], std::ios::binary);
    if (!input) throw std::runtime_error("request open failed");
    const std::string bytes((std::istreambuf_iterator<char>(input)), {});
    const auto parsed = q3x::io::json::parse(bytes);
    if (!parsed) throw std::runtime_error("request JSON invalid");
    const auto* prompt = parsed.value->find("prompt");
    if (prompt == nullptr) prompt = parsed.value->find("prompt_token_ids");
    if (prompt == nullptr || prompt->as_array() == nullptr)
      throw std::runtime_error("request requires exact token-ID prompt array");
    const auto* maximum = parsed.value->find("max_tokens");
    std::uint64_t output_count = 0U;
    if (maximum == nullptr || maximum->as_number() == nullptr ||
        !maximum->as_number()->to_uint64(output_count) || output_count != kOutputs)
      throw std::runtime_error("request must have max_tokens=16");
    std::vector<std::uint32_t> ids;
    for (const auto& item : *prompt->as_array()) {
      std::uint64_t token = 0U;
      if (item.as_number() == nullptr || !item.as_number()->to_uint64(token) ||
          token >= rt::kReferenceVocabularySize) throw std::runtime_error("invalid prompt ID");
      ids.push_back(static_cast<std::uint32_t>(token));
    }
    if (selected != 0U) {
      if (selected > ids.size()) throw std::runtime_error("prefix exceeds supplied prompt");
      ids.resize(selected);
    }
    // The existing Graph cache covers short positions and bypasses this
    // scalar observer. P>=65 keeps the ordinary policy but captures every step.
    if (ids.size() < 65U || ids.size() > 40000U)
      throw std::runtime_error("capture prompt must be in [65,40000]");
    core::Sha256 prompt_hash;
    if (!prompt_hash.update(ids.data(), ids.size() * sizeof(ids[0])))
      throw std::runtime_error("prompt hash failed");
    if (setenv("Q3X_RUN_DECODE_DOWN_K512_CONSUMER_ORDER_ADMISSION", "1", 1) != 0 ||
        setenv("Q3X_RUN_DECODE_GATE_UP_COUPLED_FEED_ADMISSION", "1", 1) != 0)
      throw std::runtime_error("Decode production-layout admission failed");
    (void)rt::reference_engine_detail::exchange_reference_engine_prefill_final_token_policy_for_test(
        rt::reference_engine_detail::ReferenceEnginePrefillFinalTokenPolicyForTest::kProductionDefault);
#if defined(Q3X_CAPTURE_HAS_LIVENESS)
    detail::set_terminal_prefix_elision_enabled_for_test(liveness);
#endif
#if defined(Q3X_CAPTURE_HAS_SCORE_FEED)
    (void)detail::exchange_exact_attention_score_feed_for_test(score_feed
        ? detail::ExactAttentionScoreFeedForTest::kScoreFeed
        : detail::ExactAttentionScoreFeedForTest::kIncumbentQt2);
    (void)detail::exchange_exact_attention_score_feed_launch_hits_for_test(0U);
#endif
    rt::ReferenceEngineOptions options;
    options.projection_backend = rt::ProjectionBackend::kSm87WeightOnly;
    options.request_options.max_sequence_length = 44095U;
    options.request_options.max_arena_bytes = 3070908416ULL;
    options.request_options.prefill_chunk_size = 512U;
    options.decode_graph_cache_policy = rt::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions;
    auto created = rt::create_reference_engine(argv[1], options);
    if (!created) throw std::runtime_error("engine creation failed: " +
        created.diagnostic.stage + ": " + created.diagnostic.message);
    const auto& load = created.value->load_stats();
    if (load.request_arena_bytes != 3070908416ULL ||
        !load.fp8_output_sidecars_enabled || load.fp8_output_sidecar_layers != 64U ||
        !load.fp8_prefill_supermatrix_sidecars_enabled ||
        load.fp8_prefill_supermatrix_sidecar_projections != 208U ||
        !load.nvfp4_down_consumer_order_sidecars_enabled ||
        load.nvfp4_down_consumer_order_sidecar_layers != 53U ||
        !load.nvfp4_gate_up_coupled_feed_enabled ||
        load.nvfp4_gate_up_coupled_feed_layers != 64U ||
        load.decode_graph_cache_slot_count != 25U ||
        load.decode_graph_cache_effective_policy != rt::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions ||
        load.fp8_marlin_prefill_sidecars_enabled || load.nvfp4_marlin_prefill_sidecars_enabled)
      throw std::runtime_error("ordinary production inventory mismatch");
    Capture capture;
    capture.prompt_tokens = static_cast<std::uint32_t>(ids.size());
#if defined(Q3X_CAPTURE_HAS_SCORE_FEED)
    (void)detail::exchange_exact_attention_score_feed_launch_hits_for_test(0U);
#endif
    const auto previous_step = detail::exchange_reference_engine_step_snapshot_hook({step_hook, &capture});
    const auto previous_return = detail::exchange_reference_engine_generate_return_snapshot_hook({return_hook, &capture});
    rt::ReferenceGenerateOptions generation_options;
    generation_options.max_new_tokens = kOutputs;
    generation_options.prefill_chunk_size = 512U;
    generation_options.logits_mode = rt::ReferenceLogitsMode::kPredictedTokenOnly;
    auto result = created.value->generate_prompt_token_ids(ids, generation_options);
    (void)detail::exchange_reference_engine_step_snapshot_hook(previous_step);
    (void)detail::exchange_reference_engine_generate_return_snapshot_hook(previous_return);
    if (!result) throw std::runtime_error("actual generation failed: " +
        result.diagnostic.stage + ": " + result.diagnostic.message);
    const auto& generation = *result.value;
    bool valid = capture.error == nullptr && capture.step_calls == kOutputs &&
        capture.return_calls == 1U && generation.prompt_token_ids == ids &&
        generation.generated_token_ids.size() == kOutputs &&
        !generation.all_prompt_tokens_prefilled_by_tiles && !generation.single_arbitrary_prefill_tiles &&
        generation.decode_graph_replays == 0U && generation.prefill_route_evidence.valid;
    for (std::size_t i = 0U; valid && i < kOutputs; ++i)
      valid = capture.steps[i].argmax == generation.generated_token_ids[i];
    std::vector<std::uint32_t> exposed_predictions;
    for (const auto& step : generation.steps) {
      if (step.logits.has_value()) valid = false;
      if (step.prediction.has_value())
        exposed_predictions.push_back(step.prediction->predicted_token_id);
    }
    valid = valid && exposed_predictions == generation.generated_token_ids &&
        capture.prefill.scalar.hidden == capture.steps.front().hidden &&
        capture.prefill.scalar.logits == capture.steps.front().logits &&
        capture.returned.scalar.hidden == capture.steps.back().hidden &&
        capture.returned.scalar.logits == capture.steps.back().logits;
    std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("output open failed");
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"schema_version\":1,\"status\":" << quoted(valid ? "pass" : "fail")
        << ",\"variant\":" << quoted(variant) << ",\"timing_authority\":false,"
        << "\"final_prompt_policy\":\"ordinary_p_minus_1_then_scalar\",\"logits_mode\":\"predicted_only\","
        << "\"request_sha256\":" << quoted(core::sha256(bytes).hex())
        << ",\"prompt_ids_u32le_sha256\":" << quoted(prompt_hash.finalize().hex())
        << ",\"capture_error\":" << (capture.error == nullptr ? "null" : quoted(capture.error))
        << ",\"cuda_error\":" << capture.cuda_error << ",\"prompt_ids\":";
    write_ids(out, ids); out << ",\"generated_ids\":";
    write_ids(out, generation.generated_token_ids);
    out << ",\"exposed_prediction_ids\":";
    write_ids(out, exposed_predictions);
    out << ",\"generated_text\":" << quoted(generation.generated_text)
        << ",\"generated_text_sha256\":" << quoted(core::sha256(generation.generated_text).hex())
        << ",\"prefill_commit\":"; write_full(out, capture.prefill);
    out << ",\"generation_return\":"; write_full(out, capture.returned);
    out << ",\"steps\":[";
    for (std::size_t i = 0U; i < std::min(capture.step_calls, kOutputs); ++i) {
      if (i != 0U) out << ',';
      write_scalar(out, capture.steps[i]);
    }
    const auto& route = generation.prefill_route_evidence;
    out << "],\"route\":{\"completed_passes\":" << route.completed_layer_passes
        << ",\"expected_passes\":" << route.expected_layer_passes << ",\"operators\":[";
    for (std::size_t i = 0U; i < route.operators.size(); ++i) {
      if (i != 0U) out << ',';
      const auto& counts = route.operators[i];
      out << "{\"role\":" << quoted(rt::to_string(static_cast<rt::PrefillOperatorRole>(i)))
          << ",\"production\":" << counts.production_hits << ",\"exact_fallback\":"
          << counts.exact_fallback_hits << ",\"forbidden\":" << counts.forbidden_hits << '}';
    }
    out << "],\"forbidden_boundaries\":[";
    for (std::size_t i = 0U; i < route.forbidden_boundary_hits.size(); ++i) {
      if (i != 0U) out << ',';
      out << route.forbidden_boundary_hits[i];
    }
    out << "]},\"score_feed_launch_hits\":";
#if defined(Q3X_CAPTURE_HAS_SCORE_FEED)
    out << detail::exchange_exact_attention_score_feed_launch_hits_for_test(0U);
#else
    out << 0;
#endif
    out << "}\n";
    out.flush();
    if (!out) throw std::runtime_error("output write failed");
    return valid ? 0 : 1;
  } catch (const std::exception& error) {
    std::cerr << "ordinary state capture: " << error.what() << '\n';
    std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
    if (out) out << "{\"schema_version\":1,\"status\":\"fail\",\"timing_authority\":false,\"error\":"
                 << quoted(error.what()) << "}\n";
    return 1;
  }
}
