#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/runtime/reference_engine.h"
#include "reference_runner_gdn_chunk64_native_admission.h"

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
constexpr std::uint32_t kKvSamplePositions = 2048U;
constexpr std::size_t kHiddenBytes = rt::kReferenceHiddenSize * 2U;
constexpr std::size_t kLogitsBytes = rt::kReferenceVocabularySize * 2U;

struct LogitSnapshot {
  std::uint32_t sequence = 0U;
  std::uint32_t argmax = 0U;
  float chosen_logit = 0.0F;
  std::array<std::pair<std::uint32_t, float>, 5U> top5{};
};

struct Capture {
  std::vector<std::uint8_t> scratch = std::vector<std::uint8_t>(4U * 1024U * 1024U);
  std::vector<std::uint16_t> logits =
      std::vector<std::uint16_t>(rt::kReferenceVocabularySize);
  std::array<LogitSnapshot, kOutputs> steps{};
  std::size_t step_calls = 0U;
  std::uint32_t prompt_tokens = 0U;
  const char* error = nullptr;
  int cuda_error = 0;
  // Filled by return_hook; consumed by dump_state() called from the hook.
  std::string out_dir;
  std::string base;
  bool dump_done = false;
  // Legacy route prefills P-1 tokens and feeds the final prompt token as the
  // first decode step (seq after step i = P + i). Whole-core prefills all P
  // tokens (seq after step i = P + i + 1).
  std::uint32_t seq_offset = 0U;
};

float bf16_to_float(std::uint16_t bits) noexcept {
  const std::uint32_t word = static_cast<std::uint32_t>(bits) << 16U;
  float result;
  std::memcpy(&result, &word, sizeof(result));
  return result;
}

bool copy_device_bytes(const std::uint8_t* source, std::size_t bytes,
                       std::vector<std::uint8_t>& scratch,
                       std::vector<std::uint8_t>& out, int& cuda_error) noexcept {
  out.resize(bytes);
  for (std::size_t offset = 0U; offset < bytes;) {
    const std::size_t count = std::min(scratch.size(), bytes - offset);
    const auto status = cudaMemcpy(scratch.data(), source + offset, count,
                                   cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) { cuda_error = static_cast<int>(status); return false; }
    std::memcpy(out.data() + offset, scratch.data(), count);
    offset += count;
  }
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

bool capture_logits(const rt::RequestState& state, Capture& capture,
                    LogitSnapshot& out) noexcept {
  out.sequence = state.sequence_length();
  const auto& plan = state.plan();
  const auto* source = region(state, plan.fp32_scratch, kLogitsBytes);
  if (source == nullptr) return false;
  const auto status = cudaMemcpy(capture.logits.data(), source, kLogitsBytes,
                                 cudaMemcpyDeviceToHost);
  if (status != cudaSuccess) { capture.cuda_error = static_cast<int>(status); return false; }
  float maximum = -std::numeric_limits<float>::infinity();
  for (std::size_t index = 0U; index < capture.logits.size(); ++index) {
    const float value = bf16_to_float(capture.logits[index]);
    if (!std::isfinite(value)) return false;
    if (value > maximum) { maximum = value; out.argmax = static_cast<std::uint32_t>(index); }
  }
  out.chosen_logit = maximum;
  for (std::size_t rank = 0U; rank < 5U; ++rank) {
    float best = -std::numeric_limits<float>::infinity();
    std::uint32_t best_id = 0U;
    for (std::size_t index = 0U; index < capture.logits.size(); ++index) {
      const float value = bf16_to_float(capture.logits[index]);
      bool seen = false;
      for (std::size_t r = 0U; r < rank; ++r)
        if (out.top5[r].first == index) seen = true;
      if (!seen && value > best) { best = value; best_id = static_cast<std::uint32_t>(index); }
    }
    out.top5[rank] = {best_id, best};
  }
  return true;
}

void step_hook(const rt::RequestState& state, void* context) noexcept {
  auto& capture = *static_cast<Capture*>(context);
  const std::size_t index = capture.step_calls++;
  if (capture.error != nullptr) return;
  if (index >= kOutputs ||
      state.sequence_length() != capture.prompt_tokens + capture.seq_offset + index) {
    capture.error = "step_sequence_or_count";
    return;
  }
  if (!capture_logits(state, capture, capture.steps[index]))
    capture.error = "step_logits";
}

void dump_state(const rt::RequestState& state, Capture& capture) noexcept {
  if (capture.dump_done) return;
  capture.dump_done = true;
  const auto& plan = state.plan();
  auto fail = [&capture](const char* message) noexcept {
    if (capture.error == nullptr) capture.error = message;
  };
  // GDN state (full).
  {
    const auto* src = region(state, plan.gdn_state, rt::kRequestGdnStateBytes);
    if (src == nullptr) { fail("gdn_state_region"); return; }
    std::vector<std::uint8_t> data;
    if (!copy_device_bytes(src, rt::kRequestGdnStateBytes, capture.scratch, data,
                           capture.cuda_error)) { fail("gdn_state_copy"); return; }
    std::ofstream f(capture.out_dir + "/" + capture.base + "_gdn.bin",
                    std::ios::binary | std::ios::trunc);
    if (!f) { fail("gdn_write"); return; }
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    if (!f) { fail("gdn_write_flush"); return; }
  }
  // Conv state (full).
  {
    const auto* src = region(state, plan.conv_state, rt::kRequestConvStateBytes);
    if (src == nullptr) { fail("conv_state_region"); return; }
    std::vector<std::uint8_t> data;
    if (!copy_device_bytes(src, rt::kRequestConvStateBytes, capture.scratch, data,
                           capture.cuda_error)) { fail("conv_state_copy"); return; }
    std::ofstream f(capture.out_dir + "/" + capture.base + "_conv.bin",
                    std::ios::binary | std::ios::trunc);
    if (!f) { fail("conv_write"); return; }
    f.write(reinterpret_cast<const char*>(data.data()),
            static_cast<std::streamsize>(data.size()));
    if (!f) { fail("conv_write_flush"); return; }
  }
  // K/V cache: fixed absolute prefill window [40000-kKvSamplePositions,
  // 40000), [kv-heads=4][head_dim=256] bf16 per token (2048 bytes). The window
  // lies entirely inside the identical prompt, so it is directly comparable
  // across routes regardless of how many decode tokens each produced.
  {
    const std::uint32_t prompt = capture.prompt_tokens;
    const std::uint32_t start =
        prompt > kKvSamplePositions ? prompt - kKvSamplePositions : 0U;
    const std::uint32_t count = prompt - start;
    const std::size_t row_bytes = 4U * 256U * sizeof(std::uint16_t);
    const std::size_t per_layer_bytes = static_cast<std::size_t>(count) * row_bytes;
    std::vector<std::uint8_t> kv_data(
        rt::kRequestFullLayerCount * 2U * per_layer_bytes);
    for (std::size_t slot = 0U; slot < rt::kRequestFullLayerCount; ++slot) {
      for (int kv = 0; kv < 2; ++kv) {
        const auto& cache =
            kv == 0 ? plan.key_cache[slot] : plan.value_cache[slot];
        const auto* src = region(state, cache, cache.byte_size);
        if (src == nullptr) { fail("kv_region"); return; }
        const std::size_t offset_bytes =
            static_cast<std::size_t>(start) * row_bytes;
        const std::size_t dst_offset =
            (slot * 2U + static_cast<std::size_t>(kv)) * per_layer_bytes;
        const auto status = cudaMemcpy(
            kv_data.data() + dst_offset, src + offset_bytes, per_layer_bytes,
            cudaMemcpyDeviceToHost);
        if (status != cudaSuccess) {
          capture.cuda_error = static_cast<int>(status);
          fail("kv_copy");
          return;
        }
      }
    }
    std::ofstream f(capture.out_dir + "/" + capture.base + "_kv.bin",
                    std::ios::binary | std::ios::trunc);
    if (!f) { fail("kv_write"); return; }
    f.write(reinterpret_cast<const char*>(kv_data.data()),
            static_cast<std::streamsize>(kv_data.size()));
    if (!f) { fail("kv_write_flush"); return; }
  }
}
void return_hook(const rt::RequestState& state, void* context) noexcept {
  auto& capture = *static_cast<Capture*>(context);
  if (capture.error != nullptr) return;
  if (capture.step_calls == 0U || capture.step_calls > kOutputs ||
      state.sequence_length() !=
          capture.prompt_tokens + capture.seq_offset + capture.step_calls - 1U) {
    std::cerr << "return_sequence: actual=" << state.sequence_length()
              << " prompt=" << capture.prompt_tokens
              << " offset=" << capture.seq_offset
              << " steps=" << capture.step_calls << '\n';
    capture.error = "return_sequence";
    return;
  }
  dump_state(state, capture);
}

std::string json_quote(std::string_view input) {
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

void write_ids(std::ostream& out, const std::vector<std::uint32_t>& ids) {
  out << '[';
  for (std::size_t i = 0U; i < ids.size(); ++i) { if (i != 0U) out << ','; out << ids[i]; }
  out << ']';
}

void write_logit_snapshot(std::ostream& out, const LogitSnapshot& value) {
  out << "{\"sequence\":" << value.sequence
      << ",\"argmax\":" << value.argmax
      << ",\"chosen_logit\":" << value.chosen_logit
      << ",\"top5\":[";
  for (std::size_t rank = 0U; rank < 5U; ++rank) {
    if (rank != 0U) out << ',';
    out << "{\"id\":" << value.top5[rank].first
        << ",\"logit\":" << value.top5[rank].second << '}';
  }
  out << "]}";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 5) {
    std::cerr << "usage: MODEL REQUEST_JSON OUTPUT_JSON --route legacy|wholecore\n";
    return 2;
  }
  try {
    std::string route = "legacy";
    for (int i = 4; i < argc; i += 2) {
      if (i + 1 >= argc) throw std::runtime_error("missing option value");
      if (std::string_view(argv[i]) == "--route") route = argv[i + 1];
      else throw std::runtime_error("unknown option");
    }
    const bool whole_core = (route == "wholecore");
    if (route != "legacy" && !whole_core)
      throw std::runtime_error("unknown route: " + route);

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
    if (ids.size() != 40000U)
      throw std::runtime_error("state capture requires P40000");

    core::Sha256 prompt_hash;
    if (!prompt_hash.update(ids.data(), ids.size() * sizeof(ids[0])))
      throw std::runtime_error("prompt hash failed");

    rt::ReferenceEngineOptions options;
    options.projection_backend = rt::ProjectionBackend::kSm87WeightOnly;
    if (whole_core) {
      options.request_options.max_sequence_length = 40'016U;
      options.request_options.max_arena_bytes = 8'641'684'992ULL;
      options.request_options.prefill_chunk_size = rt::kMaximumRequestPrefillChunkSize;
      options.prefill_execution_mode = rt::ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
      options.prefill_full_attention_tactic = rt::LayerMajorPrefillFullAttentionTactic::kNativeFlashInferExactWholePrompt;
      options.prefill_projection_tactic = rt::LayerMajorPrefillProjectionTactic::kNativePromptWideP40WholeCore;
      options.decode_graph_cache_policy = rt::ReferenceDecodeGraphCachePolicy::kDisabled;
    } else {
      options.request_options.max_sequence_length = 44'095U;
      options.request_options.max_arena_bytes = 3'070'908'416ULL;
      options.request_options.prefill_chunk_size = 512U;
      options.prefill_execution_mode = rt::ReferencePrefillExecutionMode::kLegacyC512Tiled;
      options.prefill_full_attention_tactic = rt::LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512;
      options.prefill_projection_tactic = rt::LayerMajorPrefillProjectionTactic::kExactSegmentedC512;
      options.decode_graph_cache_policy = rt::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions;
    }

    auto created = rt::create_reference_engine(argv[1], options);
    if (!created) throw std::runtime_error("engine creation failed: " +
        created.diagnostic.stage + ": " + created.diagnostic.message);

    Capture capture;
    capture.prompt_tokens = static_cast<std::uint32_t>(ids.size());
    capture.out_dir = std::filesystem::path(argv[3]).parent_path().string();
    if (capture.out_dir.empty()) capture.out_dir = ".";
    capture.base = std::filesystem::path(argv[3]).stem().string();
    capture.seq_offset = whole_core ? 1U : 0U;
    std::filesystem::create_directories(capture.out_dir);

    const auto previous_step = detail::exchange_reference_engine_step_snapshot_hook({step_hook, &capture});
    const auto previous_return = detail::exchange_reference_engine_generate_return_snapshot_hook({return_hook, &capture});

    rt::ReferenceGenerateOptions generation_options;
    generation_options.max_new_tokens = kOutputs;
    generation_options.prefill_chunk_size = whole_core ? rt::kMaximumRequestPrefillChunkSize : 512U;
    generation_options.logits_mode = rt::ReferenceLogitsMode::kPredictedTokenOnly;
    generation_options.prefill_execution_mode = whole_core
        ? rt::ReferencePrefillExecutionMode::kWholeRequestLayerMajor
        : rt::ReferencePrefillExecutionMode::kLegacyC512Tiled;
    if (!whole_core) generation_options.use_prepared_decode_graph_cache = true;

    auto result = created.value->generate_prompt_token_ids(ids, generation_options);

    (void)detail::exchange_reference_engine_step_snapshot_hook(previous_step);
    (void)detail::exchange_reference_engine_generate_return_snapshot_hook(previous_return);

    if (!result) throw std::runtime_error("generation failed: " +
        result.diagnostic.stage + ": " + result.diagnostic.message +
        "; capture_error=" + (capture.error == nullptr ? "none" : capture.error) +
        "; cuda_error=" + std::to_string(capture.cuda_error));
    const auto& generation = *result.value;
    if (capture.error != nullptr) throw std::runtime_error(std::string("capture error: ") + capture.error);
    if (capture.step_calls == 0U || capture.step_calls > kOutputs)
      throw std::runtime_error("step count out of range: " +
          std::to_string(capture.step_calls));

    // Write JSON manifest.
    std::ofstream out(argv[3], std::ios::binary | std::ios::trunc);
    if (!out) throw std::runtime_error("output open failed");
    out << std::setprecision(std::numeric_limits<double>::max_digits10)
        << "{\"schema_version\":1,\"route\":" << json_quote(route)
        << ",\"request_sha256\":" << json_quote(core::sha256(bytes).hex())
        << ",\"prompt_ids_u32le_sha256\":" << json_quote(prompt_hash.finalize().hex())
        << ",\"prompt_tokens\":" << ids.size()
        << ",\"generated_ids\":";
    write_ids(out, generation.generated_token_ids);
    out << ",\"generated_text\":" << json_quote(generation.generated_text)
        << ",\"generated_text_sha256\":" << json_quote(core::sha256(generation.generated_text).hex())
        << ",\"steps\":[";
    for (std::size_t i = 0U; i < kOutputs; ++i) {
      if (i != 0U) out << ',';
      write_logit_snapshot(out, capture.steps[i]);
    }
    out << "],\"state_files\":{"
        << "\"gdn\":" << json_quote(capture.base + "_gdn.bin")
        << ",\"conv\":" << json_quote(capture.base + "_conv.bin")
        << ",\"kv\":" << json_quote(capture.base + "_kv.bin")
        << ",\"kv_sample_positions\":" << kKvSamplePositions
        << "}"
        << ",\"memory_profile\":" << json_quote(
             whole_core ? "kLayerMajorP40WholeCore" : "kLegacyC512")
        << ",\"prefill_execution_mode\":" << json_quote(
             whole_core ? "kWholeRequestLayerMajor" : "kLegacyC512Tiled")
        << "}";
    out << "\n";
    out.flush();
    if (!out) throw std::runtime_error("output write failed");
    std::cout << "state capture complete: route=" << route
              << " tokens=" << generation.generated_token_ids.size()
              << " text_sha=" << core::sha256(generation.generated_text).hex() << '\n';
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "state capture: " << error.what() << '\n';
    return 1;
  }
}
