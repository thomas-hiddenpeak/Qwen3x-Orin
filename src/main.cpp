#include "q3x/core/device_info.h"
#include "q3x/kernels/device_probe.h"
#include "q3x/model/model_config.h"
#include "q3x/runtime/reference_benchmark.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/version.h"

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <optional>
#include <ostream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

inline constexpr std::uint32_t kCliDefaultMaxTokens = 16U;
inline constexpr std::uint32_t kCliMaximumMaxTokens = 4'096U;
inline constexpr std::uint32_t kCliDefaultWarmupRounds = 1U;
inline constexpr std::uint32_t kCliDefaultBenchmarkIterations = 3U;
inline constexpr std::uint32_t kCliMaximumBenchmarkRounds = 10'000U;
inline constexpr std::uint32_t kCliDefaultMaxSequenceLength = 512U;
inline constexpr std::uint32_t kCliDefaultPrefillChunkSize =
    q3x::runtime::kDefaultRequestPrefillChunkSize;
inline constexpr std::uint32_t kCliMaximumPrefillChunkSize =
    q3x::runtime::kMaximumRequestPrefillChunkSize;

[[nodiscard]] constexpr q3x::runtime::ReferenceDecodeGraphCachePolicy
DecodeGraphCachePolicyForGeneration(
    const q3x::runtime::ProjectionBackend backend,
    const bool capture_trace,
    const q3x::runtime::ReferenceLogitsMode logits_mode,
    const std::uint32_t max_new_tokens) noexcept {
  return backend == q3x::runtime::ProjectionBackend::kSm87WeightOnly &&
                 !capture_trace &&
                 logits_mode ==
                     q3x::runtime::ReferenceLogitsMode::kPredictedTokenOnly &&
                 max_new_tokens > 1U
             ? q3x::runtime::ReferenceDecodeGraphCachePolicy::
                   kSm87ShortPositions
             : q3x::runtime::ReferenceDecodeGraphCachePolicy::kDisabled;
}

[[nodiscard]] constexpr std::string_view DecodeGraphCachePolicyName(
    const q3x::runtime::ReferenceDecodeGraphCachePolicy policy) noexcept {
  switch (policy) {
    case q3x::runtime::ReferenceDecodeGraphCachePolicy::kDisabled:
      return "disabled";
    case q3x::runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions:
      return "sm87_short_positions";
  }
  return "unknown";
}

void PrintUsage(std::ostream& output) {
  output
      << "Qwen3x-Orin " Q3X_VERSION_STRING "\n\n"
      << "Usage:\n"
      << "  qwen3x-orin version    Show build information\n"
      << "  qwen3x-orin probe      Inspect the CUDA device and run a smoke kernel\n"
      << "  qwen3x-orin models     List catalogued target architectures\n"
      << "  qwen3x-orin generate MODEL_DIR --prompt TEXT "
         "[--max-tokens N] [--trace] [--nvtx-phase-ranges] "
         "[--prefill-chunk-size N] "
         "[--projection-backend reference|sm87]\n"
      << "  qwen3x-orin benchmark MODEL_DIR --prompt TEXT [--prompt TEXT ...] "
         "[--max-tokens N] [--warmup N] [--iterations N] "
         "[--max-sequence-length N] "
         "[--prefill-chunk-size N] "
         "[--nvtx-phase-ranges] "
         "[--projection-backend reference|sm87]\n"
      << "  qwen3x-orin help       Show this help\n";
}

void PrintGenerateUsage(std::ostream& output) {
  output
      << "Usage:\n"
      << "  qwen3x-orin generate MODEL_DIR --prompt TEXT "
         "[--max-tokens N] [--trace] [--nvtx-phase-ranges] "
         "[--prefill-chunk-size N] "
         "[--projection-backend reference|sm87]\n\n"
      << "Options:\n"
      << "  --prompt TEXT    One user message; thinking is disabled\n"
      << "  --max-tokens N  Greedy output limit (1.."
      << kCliMaximumMaxTokens << ", default " << kCliDefaultMaxTokens
      << ")\n"
      << "  --trace         Capture per-step activation boundary digests\n"
      << "  --nvtx-phase-ranges\n"
      << "                  Emit Prefill/Decode NVTX ranges for profiling\n"
      << "  --prefill-chunk-size N\n"
      << "                  Prompt-prefix projection tile size (1.."
      << kCliMaximumPrefillChunkSize << ", default "
      << kCliDefaultPrefillChunkSize
      << "; trace mode has effective size 1)\n"
      << "  --projection-backend reference|sm87\n"
      << "                  Projection policy (default reference)\n";
}

void PrintBenchmarkUsage(std::ostream& output) {
  output
      << "Usage:\n"
      << "  qwen3x-orin benchmark MODEL_DIR --prompt TEXT "
         "[--prompt TEXT ...] [--max-tokens N] [--warmup N] "
         "[--iterations N] [--max-sequence-length N] "
         "[--prefill-chunk-size N] "
         "[--nvtx-phase-ranges] "
         "[--projection-backend reference|sm87]\n\n"
      << "Options:\n"
      << "  --prompt TEXT              Repeatable non-empty user message\n"
      << "  --max-tokens N            Uniform greedy output limit (1.."
      << kCliMaximumMaxTokens << ", default " << kCliDefaultMaxTokens
      << ")\n"
      << "  --warmup N                Warmup rounds (0.."
      << kCliMaximumBenchmarkRounds << ", default "
      << kCliDefaultWarmupRounds << ")\n"
      << "  --iterations N            Measured rounds (1.."
      << kCliMaximumBenchmarkRounds << ", default "
      << kCliDefaultBenchmarkIterations << ")\n"
      << "  --max-sequence-length N   Shared request capacity (1.."
      << q3x::runtime::kAbsoluteRequestMaxSequenceLength << ", default "
      << kCliDefaultMaxSequenceLength
      << "; also limited by the default 2 GiB request arena)\n"
      << "  --prefill-chunk-size N\n"
      << "                              Prompt-prefix projection tile size (1.."
      << kCliMaximumPrefillChunkSize << ", default "
      << kCliDefaultPrefillChunkSize << ")\n"
      << "  --nvtx-phase-ranges       Emit Prefill/Decode NVTX ranges for "
         "profiling\n"
      << "  --projection-backend reference|sm87\n"
      << "                              Projection policy (default reference)\n";
}

void PrintEscaped(std::ostream& output, const std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  output.put('"');
  for (const char character : value) {
    const auto byte = static_cast<unsigned char>(character);
    switch (byte) {
      case '\\':
        output << "\\\\";
        break;
      case '"':
        output << "\\\"";
        break;
      case '\n':
        output << "\\n";
        break;
      case '\r':
        output << "\\r";
        break;
      case '\t':
        output << "\\t";
        break;
      default:
        if (byte < 0x20U) {
          output << "\\u00" << kHex[(byte >> 4U) & 0x0FU]
                 << kHex[byte & 0x0FU];
        } else {
          output.put(character);
        }
        break;
    }
  }
  output.put('"');
}

void PrintStringField(std::ostream& output, const std::string_view key,
                      const std::string_view value) {
  output << key << '=';
  PrintEscaped(output, value);
  output.put('\n');
}

void PrintDecodeGraphCacheLoadStats(
    std::ostream& output,
    const q3x::runtime::ReferenceEngineLoadStats& load) {
  output << "load.decode_graph_cache_requested_policy="
         << DecodeGraphCachePolicyName(
                load.decode_graph_cache_requested_policy)
         << '\n'
         << "load.decode_graph_cache_effective_policy="
         << DecodeGraphCachePolicyName(
                load.decode_graph_cache_effective_policy)
         << '\n'
         << "load.decode_graph_cache_first_position="
         << load.decode_graph_cache_first_position << '\n'
         << "load.decode_graph_cache_last_position="
         << load.decode_graph_cache_last_position << '\n'
         << "load.decode_graph_cache_slot_count="
         << load.decode_graph_cache_slot_count << '\n'
         << "load.decode_graph_cache_capture_enqueue_ms="
         << load.decode_graph_cache_capture_enqueue_milliseconds << '\n'
         << "load.decode_graph_cache_topology_inspection_ms="
         << load.decode_graph_cache_topology_inspection_milliseconds << '\n'
         << "load.decode_graph_cache_instantiate_ms="
         << load.decode_graph_cache_instantiate_milliseconds << '\n'
         << "load.decode_graph_cache_upload_ready_ms="
         << load.decode_graph_cache_upload_ready_milliseconds << '\n'
         << "load.decode_graph_cache_prepare_wall_ms="
         << load.decode_graph_cache_prepare_milliseconds << '\n'
         << "load.decode_graph_cache_free_bytes_before="
         << load.decode_graph_cache_free_bytes_before << '\n'
         << "load.decode_graph_cache_free_bytes_after="
         << load.decode_graph_cache_free_bytes_after << '\n'
         << "load.decode_graph_cache_free_drop_bytes="
         << load.decode_graph_cache_free_drop_bytes << '\n';
  PrintStringField(output, "load.decode_graph_cache_fallback_reason",
                   load.decode_graph_cache_fallback_reason);
}

void PrintIds(std::ostream& output, const std::string_view key,
              const std::vector<std::uint32_t>& ids) {
  output << key << '=';
  for (std::size_t index = 0U; index < ids.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    output << ids[index];
  }
  output.put('\n');
}

void PrintTimings(std::ostream& output, const std::string_view key,
                  const std::vector<double>& timings) {
  output << key << '=' << std::fixed << std::setprecision(3);
  for (std::size_t index = 0U; index < timings.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    output << timings[index];
  }
  output.put('\n');
}

void PrintLatencyStatistics(
    std::ostream& output, const std::string_view prefix,
    const q3x::runtime::ReferenceLatencyStatistics& statistics) {
  output << prefix << ".count=" << statistics.count << '\n'
         << prefix << ".min_ms=" << statistics.minimum_milliseconds << '\n'
         << prefix << ".median_ms=" << statistics.median_milliseconds
         << '\n'
         << prefix << ".p95_ms=" << statistics.p95_milliseconds << '\n'
         << prefix << ".max_ms=" << statistics.maximum_milliseconds << '\n';
}

void PrintBenchmarkSteps(
    std::ostream& output, const std::string_view key,
    const std::vector<q3x::runtime::ReferenceBenchmarkStep>& steps) {
  output << key << '=';
  for (std::size_t index = 0U; index < steps.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const auto& step = steps[index];
    output << step.position << ':' << step.input_token_id << ':';
    if (step.predicted_token_id.has_value()) {
      output << *step.predicted_token_id;
    } else {
      output.put('-');
    }
  }
  output.put('\n');
}

struct GenerateCliOptions {
  std::filesystem::path model_directory;
  std::string prompt;
  std::uint32_t max_tokens = kCliDefaultMaxTokens;
  std::uint32_t prefill_chunk_size = kCliDefaultPrefillChunkSize;
  bool trace = false;
  bool nvtx_phase_ranges = false;
  q3x::runtime::ProjectionBackend projection_backend =
      q3x::runtime::ProjectionBackend::kReference;
};

struct GenerateParseResult {
  std::optional<GenerateCliOptions> value;
  std::string error;
};

[[nodiscard]] bool ParseU32(const std::string_view text,
                            std::uint32_t& value) noexcept {
  if (text.empty()) {
    return false;
  }
  std::uint32_t parsed = 0U;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const std::from_chars_result converted =
      std::from_chars(begin, end, parsed, 10);
  if (converted.ec != std::errc{} || converted.ptr != end) {
    return false;
  }
  value = parsed;
  return true;
}

[[nodiscard]] bool ParsePositiveU32(const std::string_view text,
                                    std::uint32_t& value) noexcept {
  return ParseU32(text, value) && value != 0U;
}

[[nodiscard]] bool ParseProjectionBackend(
    const std::string_view text,
    q3x::runtime::ProjectionBackend& backend) noexcept {
  if (text == "reference") {
    backend = q3x::runtime::ProjectionBackend::kReference;
    return true;
  }
  if (text == "sm87") {
    backend = q3x::runtime::ProjectionBackend::kSm87WeightOnly;
    return true;
  }
  return false;
}

[[nodiscard]] GenerateParseResult ParseGenerateArguments(
    const int argc, char** const argv) {
  GenerateParseResult result;
  if (argc < 3) {
    result.error = "missing MODEL_DIR";
    return result;
  }

  GenerateCliOptions options;
  options.model_directory = argv[2];
  if (options.model_directory.empty()) {
    result.error = "MODEL_DIR must not be empty";
    return result;
  }

  bool prompt_seen = false;
  bool max_tokens_seen = false;
  bool prefill_chunk_size_seen = false;
  bool trace_seen = false;
  bool nvtx_phase_ranges_seen = false;
  bool projection_backend_seen = false;
  for (int index = 3; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--prompt") {
      if (prompt_seen) {
        result.error = "--prompt may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--prompt requires TEXT";
        return result;
      }
      prompt_seen = true;
      options.prompt = argv[++index];
      if (options.prompt.empty()) {
        result.error = "--prompt TEXT must not be empty";
        return result;
      }
      continue;
    }
    if (argument == "--max-tokens") {
      if (max_tokens_seen) {
        result.error = "--max-tokens may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--max-tokens requires N";
        return result;
      }
      max_tokens_seen = true;
      const std::string_view value(argv[++index]);
      if (!ParsePositiveU32(value, options.max_tokens) ||
          options.max_tokens > kCliMaximumMaxTokens) {
        result.error = "--max-tokens must be an integer in [1, " +
                       std::to_string(kCliMaximumMaxTokens) + "]";
        return result;
      }
      continue;
    }
    if (argument == "--trace") {
      if (trace_seen) {
        result.error = "--trace may be specified only once";
        return result;
      }
      trace_seen = true;
      options.trace = true;
      continue;
    }
    if (argument == "--nvtx-phase-ranges") {
      if (nvtx_phase_ranges_seen) {
        result.error = "--nvtx-phase-ranges may be specified only once";
        return result;
      }
      nvtx_phase_ranges_seen = true;
      options.nvtx_phase_ranges = true;
      continue;
    }
    if (argument == "--prefill-chunk-size") {
      if (prefill_chunk_size_seen) {
        result.error = "--prefill-chunk-size may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--prefill-chunk-size requires N";
        return result;
      }
      prefill_chunk_size_seen = true;
      if (!ParsePositiveU32(argv[++index], options.prefill_chunk_size) ||
          options.prefill_chunk_size > kCliMaximumPrefillChunkSize) {
        result.error = "--prefill-chunk-size must be an integer in [1, " +
                       std::to_string(kCliMaximumPrefillChunkSize) + "]";
        return result;
      }
      continue;
    }
    if (argument == "--projection-backend") {
      if (projection_backend_seen) {
        result.error = "--projection-backend may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--projection-backend requires reference or sm87";
        return result;
      }
      projection_backend_seen = true;
      if (!ParseProjectionBackend(argv[++index],
                                  options.projection_backend)) {
        result.error = "--projection-backend must be reference or sm87";
        return result;
      }
      continue;
    }
    result.error = "unknown generate argument: " + std::string(argument);
    return result;
  }
  if (!prompt_seen) {
    result.error = "missing required --prompt TEXT";
    return result;
  }
  result.value.emplace(std::move(options));
  return result;
}

struct BenchmarkCliOptions {
  std::filesystem::path model_directory;
  std::vector<std::string> prompts;
  std::uint32_t max_tokens = kCliDefaultMaxTokens;
  std::uint32_t warmup_rounds = kCliDefaultWarmupRounds;
  std::uint32_t iterations = kCliDefaultBenchmarkIterations;
  std::uint32_t max_sequence_length = kCliDefaultMaxSequenceLength;
  std::uint32_t prefill_chunk_size = kCliDefaultPrefillChunkSize;
  bool nvtx_phase_ranges = false;
  q3x::runtime::ProjectionBackend projection_backend =
      q3x::runtime::ProjectionBackend::kReference;
};

struct BenchmarkParseResult {
  std::optional<BenchmarkCliOptions> value;
  std::string error;
};

[[nodiscard]] BenchmarkParseResult ParseBenchmarkArguments(
    const int argc, char** const argv) {
  BenchmarkParseResult result;
  if (argc < 3) {
    result.error = "missing MODEL_DIR";
    return result;
  }

  BenchmarkCliOptions options;
  options.model_directory = argv[2];
  if (options.model_directory.empty()) {
    result.error = "MODEL_DIR must not be empty";
    return result;
  }

  bool max_tokens_seen = false;
  bool warmup_seen = false;
  bool iterations_seen = false;
  bool max_sequence_length_seen = false;
  bool prefill_chunk_size_seen = false;
  bool nvtx_phase_ranges_seen = false;
  bool projection_backend_seen = false;
  for (int index = 3; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--prompt") {
      if (index + 1 >= argc) {
        result.error = "--prompt requires TEXT";
        return result;
      }
      std::string prompt = argv[++index];
      if (prompt.empty()) {
        result.error = "--prompt TEXT must not be empty";
        return result;
      }
      options.prompts.push_back(std::move(prompt));
      continue;
    }
    if (argument == "--max-tokens") {
      if (max_tokens_seen) {
        result.error = "--max-tokens may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--max-tokens requires N";
        return result;
      }
      max_tokens_seen = true;
      if (!ParsePositiveU32(argv[++index], options.max_tokens) ||
          options.max_tokens > kCliMaximumMaxTokens) {
        result.error = "--max-tokens must be an integer in [1, " +
                       std::to_string(kCliMaximumMaxTokens) + "]";
        return result;
      }
      continue;
    }
    if (argument == "--warmup") {
      if (warmup_seen) {
        result.error = "--warmup may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--warmup requires N";
        return result;
      }
      warmup_seen = true;
      if (!ParseU32(argv[++index], options.warmup_rounds) ||
          options.warmup_rounds > kCliMaximumBenchmarkRounds) {
        result.error = "--warmup must be an integer in [0, " +
                       std::to_string(kCliMaximumBenchmarkRounds) + "]";
        return result;
      }
      continue;
    }
    if (argument == "--iterations") {
      if (iterations_seen) {
        result.error = "--iterations may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--iterations requires N";
        return result;
      }
      iterations_seen = true;
      if (!ParsePositiveU32(argv[++index], options.iterations) ||
          options.iterations > kCliMaximumBenchmarkRounds) {
        result.error = "--iterations must be an integer in [1, " +
                       std::to_string(kCliMaximumBenchmarkRounds) + "]";
        return result;
      }
      continue;
    }
    if (argument == "--max-sequence-length") {
      if (max_sequence_length_seen) {
        result.error = "--max-sequence-length may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--max-sequence-length requires N";
        return result;
      }
      max_sequence_length_seen = true;
      if (!ParsePositiveU32(argv[++index], options.max_sequence_length) ||
          options.max_sequence_length >
              q3x::runtime::kAbsoluteRequestMaxSequenceLength) {
        result.error = "--max-sequence-length must be an integer in [1, " +
                       std::to_string(
                           q3x::runtime::kAbsoluteRequestMaxSequenceLength) +
                       "]";
        return result;
      }
      continue;
    }
    if (argument == "--nvtx-phase-ranges") {
      if (nvtx_phase_ranges_seen) {
        result.error = "--nvtx-phase-ranges may be specified only once";
        return result;
      }
      nvtx_phase_ranges_seen = true;
      options.nvtx_phase_ranges = true;
      continue;
    }
    if (argument == "--projection-backend") {
      if (projection_backend_seen) {
        result.error = "--projection-backend may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--projection-backend requires reference or sm87";
        return result;
      }
      projection_backend_seen = true;
      if (!ParseProjectionBackend(argv[++index],
                                  options.projection_backend)) {
        result.error = "--projection-backend must be reference or sm87";
        return result;
      }
      continue;
    }
    if (argument == "--prefill-chunk-size") {
      if (prefill_chunk_size_seen) {
        result.error = "--prefill-chunk-size may be specified only once";
        return result;
      }
      if (index + 1 >= argc) {
        result.error = "--prefill-chunk-size requires N";
        return result;
      }
      prefill_chunk_size_seen = true;
      if (!ParsePositiveU32(argv[++index], options.prefill_chunk_size) ||
          options.prefill_chunk_size > kCliMaximumPrefillChunkSize) {
        result.error = "--prefill-chunk-size must be an integer in [1, " +
                       std::to_string(kCliMaximumPrefillChunkSize) + "]";
        return result;
      }
      continue;
    }
    result.error = "unknown benchmark argument: " + std::string(argument);
    return result;
  }
  if (options.prompts.empty()) {
    result.error = "missing required --prompt TEXT";
    return result;
  }
  if (options.max_tokens > options.max_sequence_length) {
    result.error = "--max-tokens must not exceed --max-sequence-length";
    return result;
  }
  result.value.emplace(std::move(options));
  return result;
}

void PrintEngineDiagnostic(
    std::ostream& output,
    const q3x::runtime::ReferenceEngineDiagnostic& diagnostic) {
  output << "status=error\n"
         << "error.code=" << q3x::runtime::to_string(diagnostic.code) << '\n';
  PrintStringField(output, "error.stage", diagnostic.stage);
  PrintStringField(output, "error.message", diagnostic.message);
  if (!diagnostic.context.empty()) {
    PrintStringField(output, "error.context", diagnostic.context);
  }
  output << "error.dependency_code=" << diagnostic.dependency_error << '\n'
         << "error.cuda_code=" << diagnostic.cuda_error << '\n';
  if (diagnostic.layer != q3x::runtime::kReferenceNoLayer) {
    output << "error.layer=" << diagnostic.layer << '\n';
  }
  if (!diagnostic.operation.empty()) {
    PrintStringField(output, "error.operation", diagnostic.operation);
  }
}

[[nodiscard]] int EngineFailureExitCode(
    const q3x::runtime::ReferenceEngineError error) noexcept {
  using Error = q3x::runtime::ReferenceEngineError;
  switch (error) {
    case Error::kInvalidArgument:
    case Error::kCapacityExceeded:
    case Error::kArithmeticOverflow:
    case Error::kTokenizerFailure:
    case Error::kDecodeFailure:
      return 2;
    case Error::kResidentLoadFailure:
    case Error::kWeightBindFailure:
    case Error::kRequestStateFailure:
    case Error::kRunnerFactoryFailure:
      return 3;
    case Error::kRunnerStepFailure:
    case Error::kRunnerResetFailure:
    case Error::kMissingLogits:
    case Error::kMissingTiming:
    case Error::kTraceFailure:
    case Error::kMissingPrediction:
      return 4;
    case Error::kAllocationFailure:
      return 5;
    case Error::kNone:
      return 0;
  }
  return 5;
}

void PrintBenchmarkDiagnostic(
    std::ostream& output,
    const q3x::runtime::ReferenceBenchmarkDiagnostic& diagnostic) {
  output << "status=error\n"
         << "error.code=" << q3x::runtime::to_string(diagnostic.code) << '\n';
  PrintStringField(output, "error.message", diagnostic.message);
  if (diagnostic.prompt_index !=
      q3x::runtime::kReferenceBenchmarkNoPrompt) {
    output << "error.prompt_index=" << diagnostic.prompt_index << '\n'
           << "error.round=" << diagnostic.round << '\n'
           << "error.warmup=" << (diagnostic.warmup ? 1 : 0) << '\n';
  }
  if (!diagnostic.mismatch_field.empty()) {
    PrintStringField(output, "error.mismatch_field",
                     diagnostic.mismatch_field);
  }
  output << "error.cuda_code=" << diagnostic.cuda_error << '\n';
  if (diagnostic.generation.code !=
      q3x::runtime::ReferenceEngineError::kNone) {
    output << "error.generation.code="
           << q3x::runtime::to_string(diagnostic.generation.code) << '\n';
    PrintStringField(output, "error.generation.stage",
                     diagnostic.generation.stage);
    PrintStringField(output, "error.generation.message",
                     diagnostic.generation.message);
    if (!diagnostic.generation.context.empty()) {
      PrintStringField(output, "error.generation.context",
                       diagnostic.generation.context);
    }
    output << "error.generation.dependency_code="
           << diagnostic.generation.dependency_error << '\n'
           << "error.generation.cuda_code="
           << diagnostic.generation.cuda_error << '\n';
    if (diagnostic.generation.layer !=
        q3x::runtime::kReferenceNoLayer) {
      output << "error.generation.layer=" << diagnostic.generation.layer
             << '\n';
    }
    if (!diagnostic.generation.operation.empty()) {
      PrintStringField(output, "error.generation.operation",
                       diagnostic.generation.operation);
    }
  }
}

[[nodiscard]] int BenchmarkFailureExitCode(
    const q3x::runtime::ReferenceBenchmarkDiagnostic& diagnostic) noexcept {
  using Error = q3x::runtime::ReferenceBenchmarkError;
  switch (diagnostic.code) {
    case Error::kInvalidArgument:
      return 2;
    case Error::kGenerationFailure: {
      const int code = EngineFailureExitCode(diagnostic.generation.code);
      return code == 0 ? 4 : code;
    }
    case Error::kRepeatabilityFailure:
    case Error::kDeviceMemoryProbeFailure:
    case Error::kInvalidTiming:
      return 4;
    case Error::kAllocationFailure:
      return 5;
    case Error::kNone:
      return 0;
  }
  return 5;
}

void PrintGeneration(
    std::ostream& output,
    const q3x::runtime::ReferenceOneShotGeneration& result,
    const q3x::runtime::ProjectionBackend projection_backend) {
  const auto& load = result.load;
  const auto& generation = result.generation;
  output << std::fixed << std::setprecision(3)
         << "status=ok\n"
         << "projection.backend="
         << q3x::runtime::to_string(projection_backend) << '\n'
         << "prefill.requested_chunk_size="
         << generation.requested_prefill_chunk_size << '\n'
         << "prefill.effective_chunk_size="
         << generation.effective_prefill_chunk_size << '\n'
         << "prefill.all_prompt_tokens_prefilled_by_tiles="
         << (generation.all_prompt_tokens_prefilled_by_tiles ? 1 : 0) << '\n'
         << "prefill.single_arbitrary_prefill_tiles="
         << (generation.single_arbitrary_prefill_tiles ? 1 : 0) << '\n'
         << "prefill.layer_major_prefill="
         << (generation.layer_major_prefill ? 1 : 0) << '\n'
         << "load.total_ms=" << load.total_milliseconds << '\n'
         << "load.tokenizer_ms=" << load.tokenizer_milliseconds << '\n'
         << "load.resident_ms=" << load.resident_load_milliseconds << '\n'
         << "load.weight_bind_ms=" << load.weight_bind_milliseconds << '\n'
         << "load.request_state_ms=" << load.request_state_milliseconds << '\n'
         << "load.fp8_output_sidecar_ms="
         << load.fp8_output_sidecar_milliseconds << '\n'
         << "load.nvfp4_down_scale6_sidecar_ms="
         << load.nvfp4_down_scale6_sidecar_milliseconds << '\n'
         << "load.nvfp4_down_consumer_order_sidecar_ms="
         << load.nvfp4_down_consumer_order_sidecar_milliseconds << '\n'
         << "load.fp8_prefill_qkv_sidecar_ms="
         << load.fp8_prefill_qkv_sidecar_milliseconds << '\n'
         << "load.fp8_prefill_supermatrix_sidecar_ms="
         << load.fp8_prefill_supermatrix_sidecar_milliseconds << '\n'
         << "load.fp8_marlin_prefill_sidecar_ms="
         << load.fp8_marlin_prefill_sidecar_milliseconds << '\n'
         << "load.nvfp4_marlin_prefill_sidecar_ms="
         << load.nvfp4_marlin_prefill_sidecar_milliseconds << '\n'
         << "load.runner_factory_ms=" << load.runner_factory_milliseconds
         << '\n'
         << "load.fp8_output_sidecars_enabled="
         << (load.fp8_output_sidecars_enabled ? 1 : 0) << '\n'
         << "load.fp8_output_sidecar_layers="
         << load.fp8_output_sidecar_layers << '\n'
         << "load.fp8_output_sidecar_bytes="
         << load.fp8_output_sidecar_bytes << '\n'
         << "load.fp8_output_sidecar_fallback_reason="
         << load.fp8_output_sidecar_fallback_reason << '\n'
         << "load.nvfp4_down_scale6_sidecars_enabled="
         << (load.nvfp4_down_scale6_sidecars_enabled ? 1 : 0) << '\n'
         << "load.nvfp4_down_scale6_sidecar_eligible_layers="
         << load.nvfp4_down_scale6_sidecar_eligible_layers << '\n'
         << "load.nvfp4_down_scale6_sidecar_fallback_layers="
         << load.nvfp4_down_scale6_sidecar_fallback_layers << '\n'
         << "load.nvfp4_down_scale6_sidecar_bytes="
         << load.nvfp4_down_scale6_sidecar_bytes << '\n'
         << "load.nvfp4_down_scale6_sidecar_fallback_reason="
         << load.nvfp4_down_scale6_sidecar_fallback_reason << '\n'
         << "load.nvfp4_down_consumer_order_sidecars_requested="
         << (load.nvfp4_down_consumer_order_sidecars_requested ? 1 : 0)
         << '\n'
         << "load.nvfp4_down_consumer_order_sidecars_enabled="
         << (load.nvfp4_down_consumer_order_sidecars_enabled ? 1 : 0)
         << '\n'
         << "load.nvfp4_down_consumer_order_sidecar_layers="
         << load.nvfp4_down_consumer_order_sidecar_layers << '\n'
         << "load.nvfp4_down_consumer_order_sidecar_bytes="
         << load.nvfp4_down_consumer_order_sidecar_bytes << '\n'
         << "load.fp8_prefill_qkv_sidecars_enabled="
         << (load.fp8_prefill_qkv_sidecars_enabled ? 1 : 0) << '\n'
         << "load.fp8_prefill_qkv_sidecar_layers="
         << load.fp8_prefill_qkv_sidecar_layers << '\n'
         << "load.fp8_prefill_qkv_sidecar_bytes="
         << load.fp8_prefill_qkv_sidecar_bytes << '\n'
         << "load.fp8_prefill_qkv_sidecar_fallback_reason="
         << load.fp8_prefill_qkv_sidecar_fallback_reason << '\n'
         << "load.fp8_prefill_supermatrix_sidecars_enabled="
         << (load.fp8_prefill_supermatrix_sidecars_enabled ? 1 : 0) << '\n'
         << "load.fp8_prefill_supermatrix_sidecar_projections="
         << load.fp8_prefill_supermatrix_sidecar_projections << '\n'
         << "load.fp8_prefill_supermatrix_sidecar_bytes="
         << load.fp8_prefill_supermatrix_sidecar_bytes << '\n'
         << "load.fp8_marlin_prefill_sidecars_enabled="
         << (load.fp8_marlin_prefill_sidecars_enabled ? 1 : 0) << '\n'
         << "load.fp8_marlin_prefill_sidecar_projections="
         << load.fp8_marlin_prefill_sidecar_projections << '\n'
         << "load.fp8_marlin_prefill_sidecar_bytes="
         << load.fp8_marlin_prefill_sidecar_bytes << '\n'
         << "load.nvfp4_marlin_prefill_sidecars_enabled="
         << (load.nvfp4_marlin_prefill_sidecars_enabled ? 1 : 0) << '\n'
         << "load.nvfp4_marlin_prefill_sidecar_layers="
         << load.nvfp4_marlin_prefill_sidecar_layers << '\n'
         << "load.nvfp4_marlin_prefill_sidecar_bytes="
         << load.nvfp4_marlin_prefill_sidecar_bytes << '\n'
         << "load.tokenizer_resident_overlap="
         << (load.tokenizer_resident_overlap ? 1 : 0) << '\n'
         << "load.resident.bytes_read=" << load.resident.bytes_read << '\n'
         << "load.resident.bytes_copied=" << load.resident.bytes_copied
         << '\n'
         << "load.resident.bytes_skipped=" << load.resident.bytes_skipped
         << '\n'
         << "load.resident.chunks=" << load.resident.chunks << '\n'
         << "load.resident.memcpy_operations="
         << load.resident.memcpy_operations << '\n'
         << "load.resident.shard_workers="
         << load.resident.shard_workers << '\n'
         << "load.resident.pinned_staging_bytes="
         << load.resident.pinned_staging_bytes << '\n'
         << "load.resident.sha256_backend="
         << q3x::runtime::to_string(load.resident.sha256_backend) << '\n'
         << "load.resident.device_free_before="
         << load.resident.device_free_before << '\n'
         << "load.resident.device_total=" << load.resident.device_total
         << '\n'
         << "load.binding.tensor_views=" << load.binding.tensor_views << '\n'
         << "load.binding.scalar_reads=" << load.binding.scalar_reads << '\n'
         << "load.binding.bf16_projections="
         << load.binding.bf16_projections << '\n'
         << "load.binding.fp8_projections=" << load.binding.fp8_projections
         << '\n'
         << "load.binding.nvfp4_projections="
         << load.binding.nvfp4_projections << '\n'
         << "load.binding.linear_attention_layers="
         << load.binding.linear_attention_layers << '\n'
         << "load.binding.full_attention_layers="
         << load.binding.full_attention_layers << '\n'
         << "load.request_arena_bytes=" << load.request_arena_bytes << '\n'
         << "load.request_max_sequence_length="
         << load.request_max_sequence_length << '\n'
         << "load.request_prefill_chunk_size="
         << load.request_prefill_chunk_size << '\n';
  PrintDecodeGraphCacheLoadStats(output, load);
  PrintStringField(output, "prompt.rendered", generation.rendered_prompt);
  output << "prompt.token_count=" << generation.prompt_token_ids.size()
         << '\n';
  PrintIds(output, "prompt.ids", generation.prompt_token_ids);
  output << "generated.token_count="
         << generation.generated_token_ids.size() << '\n';
  PrintIds(output, "generated.ids", generation.generated_token_ids);
  PrintStringField(output, "generated.text", generation.generated_text);
  output << "generated.stop_reason="
         << q3x::runtime::to_string(generation.stop_reason) << '\n'
         << "generated.step_count=" << generation.steps.size() << '\n'
         << "generated.decode_graph_replays="
         << generation.decode_graph_replays << '\n'
         << "generated.decode_graph_serial_fallbacks="
         << generation.decode_graph_serial_fallbacks << '\n'
         << "timing.finish_prefill_ms="
         << generation.timing.finish_prefill_milliseconds << '\n'
         << "timing.prompt_prefill_ms="
         << generation.timing.prompt_prefill_milliseconds << '\n'
         << "timing.time_to_first_token_ms="
         << generation.timing.time_to_first_token_milliseconds << '\n'
         << "timing.decode_after_first_ms="
         << generation.timing.decode_after_first_milliseconds << '\n'
         << "timing.total_generation_ms="
         << generation.timing.total_generation_milliseconds << '\n';
  PrintTimings(output, "timing.prefix_execution_ms",
               generation.timing.prefix_execution_milliseconds);
  PrintTimings(output, "timing.subsequent_token_ms",
               generation.timing.subsequent_token_milliseconds);
  output << "trace.count=" << generation.traces.size() << '\n';
  for (std::size_t index = 0U; index < generation.traces.size(); ++index) {
    const auto& trace = generation.traces[index];
    const std::string prefix = "trace." + std::to_string(index) + ".";
    output << prefix << "position=" << trace.position << '\n'
           << prefix << "input_token_id=" << trace.input_token_id << '\n'
           << prefix << "element_count=" << trace.element_count << '\n';
    PrintStringField(output, prefix + "full_sha256", trace.full_sha256);
    PrintStringField(output, prefix + "embedding_sha256",
                     trace.embedding_sha256);
    for (std::size_t layer = 0U;
         layer < q3x::runtime::kReferenceDecoderLayerCount; ++layer) {
      const std::string layer_prefix =
          prefix + "layer." + std::to_string(layer) + ".";
      PrintStringField(output, layer_prefix + "hidden_sha256",
                       trace.layer_hidden_sha256[layer]);
      PrintStringField(output, layer_prefix + "residual_sha256",
                       trace.layer_residual_sha256[layer]);
    }
    PrintStringField(output, prefix + "final_norm_sha256",
                     trace.final_norm_sha256);
  }
}

int RunGenerate(const int argc, char** const argv) {
  if (argc == 3 && (std::string_view(argv[2]) == "--help" ||
                    std::string_view(argv[2]) == "-h")) {
    PrintGenerateUsage(std::cout);
    return 0;
  }
  GenerateParseResult parsed = ParseGenerateArguments(argc, argv);
  if (!parsed.value.has_value()) {
    std::cerr << "error: " << parsed.error << "\n\n";
    PrintGenerateUsage(std::cerr);
    return 2;
  }

  q3x::runtime::ReferenceOneShotOptions options;
  options.generation.max_new_tokens = parsed.value->max_tokens;
  options.generation.capture_trace = parsed.value->trace;
  options.generation.emit_nvtx_phase_ranges =
      parsed.value->nvtx_phase_ranges;
  options.generation.prefill_chunk_size = parsed.value->prefill_chunk_size;
  options.generation.logits_mode =
      q3x::runtime::ReferenceLogitsMode::kPredictedTokenOnly;
  options.projection_backend = parsed.value->projection_backend;
  options.decode_graph_cache_policy =
      DecodeGraphCachePolicyForGeneration(
          options.projection_backend, options.generation.capture_trace,
          options.generation.logits_mode,
          options.generation.max_new_tokens);
  std::cerr << "progress=loading_and_generating model_dir=";
  PrintEscaped(std::cerr, parsed.value->model_directory.string());
  std::cerr << " max_tokens=" << parsed.value->max_tokens
            << " trace=" << (parsed.value->trace ? 1 : 0)
            << " nvtx_phase_ranges="
            << (parsed.value->nvtx_phase_ranges ? 1 : 0)
            << " requested_prefill_chunk_size="
            << parsed.value->prefill_chunk_size
            << " effective_prefill_chunk_size="
            << (parsed.value->trace ? kCliDefaultPrefillChunkSize
                                    : parsed.value->prefill_chunk_size)
            << " projection_backend="
            << q3x::runtime::to_string(parsed.value->projection_backend)
            << " decode_graph_cache_policy="
            << DecodeGraphCachePolicyName(options.decode_graph_cache_policy)
            << '\n';
  q3x::runtime::ReferenceOneShotResult generated =
      q3x::runtime::generate_reference(parsed.value->model_directory,
                                       parsed.value->prompt, options);
  if (!generated) {
    PrintEngineDiagnostic(std::cerr, generated.diagnostic);
    return EngineFailureExitCode(generated.diagnostic.code);
  }
  PrintGeneration(std::cout, *generated.value,
                  parsed.value->projection_backend);
  return 0;
}

void PrintBenchmarkReport(
    std::ostream& output,
    const std::filesystem::path& model_directory,
    const q3x::runtime::ReferenceEngineLoadStats& load,
    const q3x::runtime::ReferenceBenchmarkReport& report,
    const q3x::runtime::ProjectionBackend projection_backend,
    const std::size_t nvfp4_marlin_prefill_route_hits,
    const std::size_t fp8_marlin_prefill_route_hits) {
  output << std::fixed << std::setprecision(3)
         << "status=ok\n"
         << "projection.backend="
         << q3x::runtime::to_string(projection_backend) << '\n'
         << "prefill.requested_chunk_size=" << report.prefill_chunk_size
         << '\n'
         << "prefill.effective_chunk_size=" << report.prefill_chunk_size
         << '\n'
         << "prefill.all_prompt_tokens_prefilled_by_tiles="
         << (report.all_prompt_tokens_prefilled_by_tiles ? 1 : 0) << '\n'
         << "prefill.single_arbitrary_prefill_tiles="
         << (report.single_arbitrary_prefill_tiles ? 1 : 0) << '\n'
         << "prefill.layer_major_prefill="
         << (report.layer_major_prefill ? 1 : 0) << '\n'
         << "load.total_ms=" << load.total_milliseconds << '\n'
         << "load.tokenizer_ms=" << load.tokenizer_milliseconds << '\n'
         << "load.resident_ms=" << load.resident_load_milliseconds << '\n'
         << "load.weight_bind_ms=" << load.weight_bind_milliseconds << '\n'
         << "load.request_state_ms=" << load.request_state_milliseconds
         << '\n'
         << "load.fp8_output_sidecar_ms="
         << load.fp8_output_sidecar_milliseconds << '\n'
         << "load.nvfp4_down_scale6_sidecar_ms="
         << load.nvfp4_down_scale6_sidecar_milliseconds << '\n'
         << "load.nvfp4_down_consumer_order_sidecar_ms="
         << load.nvfp4_down_consumer_order_sidecar_milliseconds << '\n'
         << "load.fp8_prefill_qkv_sidecar_ms="
         << load.fp8_prefill_qkv_sidecar_milliseconds << '\n'
         << "load.fp8_prefill_supermatrix_sidecar_ms="
         << load.fp8_prefill_supermatrix_sidecar_milliseconds << '\n'
         << "load.fp8_marlin_prefill_sidecar_ms="
         << load.fp8_marlin_prefill_sidecar_milliseconds << '\n'
         << "load.nvfp4_marlin_prefill_sidecar_ms="
         << load.nvfp4_marlin_prefill_sidecar_milliseconds << '\n'
         << "load.runner_factory_ms=" << load.runner_factory_milliseconds
         << '\n'
         << "load.fp8_output_sidecars_enabled="
         << (load.fp8_output_sidecars_enabled ? 1 : 0) << '\n'
         << "load.fp8_output_sidecar_layers="
         << load.fp8_output_sidecar_layers << '\n'
         << "load.fp8_output_sidecar_bytes="
         << load.fp8_output_sidecar_bytes << '\n'
         << "load.fp8_output_sidecar_fallback_reason="
         << load.fp8_output_sidecar_fallback_reason << '\n'
         << "load.nvfp4_down_scale6_sidecars_enabled="
         << (load.nvfp4_down_scale6_sidecars_enabled ? 1 : 0) << '\n'
         << "load.nvfp4_down_scale6_sidecar_eligible_layers="
         << load.nvfp4_down_scale6_sidecar_eligible_layers << '\n'
         << "load.nvfp4_down_scale6_sidecar_fallback_layers="
         << load.nvfp4_down_scale6_sidecar_fallback_layers << '\n'
         << "load.nvfp4_down_scale6_sidecar_bytes="
         << load.nvfp4_down_scale6_sidecar_bytes << '\n'
         << "load.nvfp4_down_scale6_sidecar_fallback_reason="
         << load.nvfp4_down_scale6_sidecar_fallback_reason << '\n'
         << "load.nvfp4_down_consumer_order_sidecars_requested="
         << (load.nvfp4_down_consumer_order_sidecars_requested ? 1 : 0)
         << '\n'
         << "load.nvfp4_down_consumer_order_sidecars_enabled="
         << (load.nvfp4_down_consumer_order_sidecars_enabled ? 1 : 0)
         << '\n'
         << "load.nvfp4_down_consumer_order_sidecar_layers="
         << load.nvfp4_down_consumer_order_sidecar_layers << '\n'
         << "load.nvfp4_down_consumer_order_sidecar_bytes="
         << load.nvfp4_down_consumer_order_sidecar_bytes << '\n'
         << "load.fp8_prefill_qkv_sidecars_enabled="
         << (load.fp8_prefill_qkv_sidecars_enabled ? 1 : 0) << '\n'
         << "load.fp8_prefill_qkv_sidecar_layers="
         << load.fp8_prefill_qkv_sidecar_layers << '\n'
         << "load.fp8_prefill_qkv_sidecar_bytes="
         << load.fp8_prefill_qkv_sidecar_bytes << '\n'
         << "load.fp8_prefill_qkv_sidecar_fallback_reason="
         << load.fp8_prefill_qkv_sidecar_fallback_reason << '\n'
         << "load.fp8_prefill_supermatrix_sidecars_enabled="
         << (load.fp8_prefill_supermatrix_sidecars_enabled ? 1 : 0) << '\n'
         << "load.fp8_prefill_supermatrix_sidecar_projections="
         << load.fp8_prefill_supermatrix_sidecar_projections << '\n'
         << "load.fp8_prefill_supermatrix_sidecar_bytes="
         << load.fp8_prefill_supermatrix_sidecar_bytes << '\n'
         << "load.fp8_marlin_prefill_sidecars_enabled="
         << (load.fp8_marlin_prefill_sidecars_enabled ? 1 : 0) << '\n'
         << "load.fp8_marlin_prefill_sidecar_projections="
         << load.fp8_marlin_prefill_sidecar_projections << '\n'
         << "load.fp8_marlin_prefill_sidecar_bytes="
         << load.fp8_marlin_prefill_sidecar_bytes << '\n'
         << "load.nvfp4_marlin_prefill_sidecars_enabled="
         << (load.nvfp4_marlin_prefill_sidecars_enabled ? 1 : 0) << '\n'
         << "load.nvfp4_marlin_prefill_sidecar_layers="
         << load.nvfp4_marlin_prefill_sidecar_layers << '\n'
         << "load.nvfp4_marlin_prefill_sidecar_bytes="
         << load.nvfp4_marlin_prefill_sidecar_bytes << '\n'
         << "load.tokenizer_resident_overlap="
         << (load.tokenizer_resident_overlap ? 1 : 0) << '\n'
         << "load.resident.sha256_backend="
         << q3x::runtime::to_string(load.resident.sha256_backend) << '\n'
         << "load.resident.shard_workers="
         << load.resident.shard_workers << '\n'
         << "load.resident.pinned_staging_bytes="
         << load.resident.pinned_staging_bytes << '\n'
         << "load.request_arena_bytes=" << load.request_arena_bytes << '\n'
         << "load.request_max_sequence_length="
         << load.request_max_sequence_length << '\n'
         << "load.request_prefill_chunk_size="
         << load.request_prefill_chunk_size << '\n'
         << "benchmark.prompt_count=" << report.prompts.size() << '\n'
         << "benchmark.warmup_rounds=" << report.warmup_rounds << '\n'
         << "benchmark.measured_rounds=" << report.measured_rounds << '\n'
         << "benchmark.logits_mode="
         << (report.logits_mode ==
                     q3x::runtime::ReferenceLogitsMode::kPredictedTokenOnly
                 ? "predicted_token_only"
                 : "full_statistics")
         << '\n'
         << "benchmark.nvtx_phase_ranges="
         << (report.nvtx_phase_ranges_emitted ? 1 : 0) << '\n'
         << "benchmark.max_new_tokens=" << report.max_new_tokens << '\n'
         << "benchmark.stop_token_id=" << report.stop_token_id << '\n'
         << "benchmark.sample_count=" << report.samples.size() << '\n'
         << "benchmark.decode_graph_replays="
         << report.decode_graph_replays << '\n'
         << "benchmark.decode_graph_serial_fallbacks="
         << report.decode_graph_serial_fallbacks << '\n'
         << "benchmark.nvfp4_marlin_prefill_route_hits="
         << nvfp4_marlin_prefill_route_hits << '\n'
         << "benchmark.fp8_marlin_prefill_route_hits="
         << fp8_marlin_prefill_route_hits << '\n';
  PrintDecodeGraphCacheLoadStats(output, load);
  PrintStringField(output, "model.directory", model_directory.string());
  PrintLatencyStatistics(output, "stats.prompt_prefix",
                         report.prompt_prefix);
  PrintLatencyStatistics(output, "stats.finish_prefill",
                         report.finish_prefill);
  PrintLatencyStatistics(output, "stats.prompt_prefill",
                         report.prompt_prefill);
  PrintLatencyStatistics(output, "stats.decode_after_first",
                         report.decode_after_first);
  PrintLatencyStatistics(output, "stats.time_to_first_token",
                         report.time_to_first_token);
  PrintLatencyStatistics(output, "stats.total_generation",
                         report.total_generation);
  PrintLatencyStatistics(output, "stats.subsequent_token",
                         report.subsequent_token);

  const auto& memory = report.device_memory;
  output << "device_memory.start_free_bytes=" << memory.start_free_bytes
         << '\n'
         << "device_memory.end_free_bytes=" << memory.end_free_bytes << '\n'
         << "device_memory.minimum_free_bytes=" << memory.minimum_free_bytes
         << '\n'
         << "device_memory.total_bytes=" << memory.total_bytes << '\n'
         << "device_memory.persistent_drop_bytes="
         << memory.persistent_drop_bytes << '\n'
         << "device_memory.maximum_observed_drop_bytes="
         << memory.maximum_observed_drop_bytes << '\n'
         << "device_memory.drop_tolerance_bytes="
         << memory.drop_tolerance_bytes << '\n'
         << "device_memory.persistent_drop_detected="
         << (memory.persistent_drop_detected ? 1 : 0) << '\n';

  for (std::size_t index = 0U; index < report.prompts.size(); ++index) {
    const auto& prompt = report.prompts[index];
    const std::string prefix = "prompt." + std::to_string(index);
    PrintStringField(output, prefix + ".text", prompt.prompt);
    PrintIds(output, prefix + ".prompt_ids", prompt.prompt_token_ids);
    PrintIds(output, prefix + ".generated_ids", prompt.generated_token_ids);
    PrintStringField(output, prefix + ".generated_text",
                     prompt.generated_text);
    output << prefix << ".stop_reason="
           << q3x::runtime::to_string(prompt.stop_reason) << '\n'
           << prefix << ".step_count=" << prompt.step_sequence.size()
           << '\n'
           << prefix << ".decode_graph_replays="
           << prompt.decode_graph_replays << '\n'
           << prefix << ".decode_graph_serial_fallbacks="
           << prompt.decode_graph_serial_fallbacks << '\n';
    PrintBenchmarkSteps(output, prefix + ".steps", prompt.step_sequence);
    PrintLatencyStatistics(output, prefix + ".prompt_prefix",
                           prompt.prompt_prefix);
    PrintLatencyStatistics(output, prefix + ".finish_prefill",
                           prompt.finish_prefill);
    PrintLatencyStatistics(output, prefix + ".prompt_prefill",
                           prompt.prompt_prefill);
    PrintLatencyStatistics(output, prefix + ".decode_after_first",
                           prompt.decode_after_first);
    PrintLatencyStatistics(output, prefix + ".time_to_first_token",
                           prompt.time_to_first_token);
    PrintLatencyStatistics(output, prefix + ".total_generation",
                           prompt.total_generation);
    PrintLatencyStatistics(output, prefix + ".subsequent_token",
                           prompt.subsequent_token);
  }

  for (std::size_t index = 0U; index < report.samples.size(); ++index) {
    const auto& sample = report.samples[index];
    const std::string prefix = "sample." + std::to_string(index);
    output << prefix << ".prompt_index=" << sample.prompt_index << '\n'
           << prefix << ".measured_round=" << sample.measured_round << '\n'
           << prefix << ".finish_prefill_ms="
           << sample.timing.finish_prefill_milliseconds << '\n'
           << prefix << ".prompt_prefill_ms="
           << sample.timing.prompt_prefill_milliseconds << '\n'
           << prefix << ".time_to_first_token_ms="
           << sample.timing.time_to_first_token_milliseconds << '\n'
           << prefix << ".decode_after_first_ms="
           << sample.timing.decode_after_first_milliseconds << '\n'
           << prefix << ".total_generation_ms="
           << sample.timing.total_generation_milliseconds << '\n'
           << prefix << ".decode_graph_replays="
           << sample.decode_graph_replays << '\n'
           << prefix << ".decode_graph_serial_fallbacks="
           << sample.decode_graph_serial_fallbacks << '\n';
    PrintTimings(output, prefix + ".prefix_execution_ms",
                 sample.timing.prefix_execution_milliseconds);
    PrintTimings(output, prefix + ".subsequent_token_ms",
                 sample.timing.subsequent_token_milliseconds);
  }
}

int RunBenchmark(const int argc, char** const argv) {
  if (argc == 3 && (std::string_view(argv[2]) == "--help" ||
                    std::string_view(argv[2]) == "-h")) {
    PrintBenchmarkUsage(std::cout);
    return 0;
  }
  BenchmarkParseResult parsed = ParseBenchmarkArguments(argc, argv);
  if (!parsed.value.has_value()) {
    std::cerr << "error: " << parsed.error << "\n\n";
    PrintBenchmarkUsage(std::cerr);
    return 2;
  }
  static_cast<void>(
      q3x::runtime::reference_runner_detail::
          exchange_nvfp4_marlin_prefill_admission_test_hits(0U));
  static_cast<void>(
      q3x::runtime::reference_runner_detail::
          exchange_fp8_marlin_prefill_admission_test_hits(0U));

  q3x::runtime::ReferenceBenchmarkOptions benchmark_options;
  benchmark_options.warmup_rounds = parsed.value->warmup_rounds;
  benchmark_options.measured_rounds = parsed.value->iterations;
  benchmark_options.max_new_tokens = parsed.value->max_tokens;
  benchmark_options.prefill_chunk_size = parsed.value->prefill_chunk_size;
  benchmark_options.emit_nvtx_phase_ranges =
      parsed.value->nvtx_phase_ranges;
  benchmark_options.logits_mode =
      q3x::runtime::ReferenceLogitsMode::kPredictedTokenOnly;

  q3x::runtime::ReferenceEngineOptions engine_options;
  engine_options.request_options.batch_size = 1U;
  engine_options.request_options.max_sequence_length =
      parsed.value->max_sequence_length;
  engine_options.request_options.prefill_chunk_size =
      parsed.value->prefill_chunk_size;
  engine_options.projection_backend = parsed.value->projection_backend;
  engine_options.decode_graph_cache_policy =
      DecodeGraphCachePolicyForGeneration(
          engine_options.projection_backend, false,
          benchmark_options.logits_mode,
          benchmark_options.max_new_tokens);

  const q3x::runtime::RequestPlanResult request_plan =
      q3x::runtime::build_request_memory_plan(engine_options.request_options);
  if (!request_plan) {
    q3x::runtime::ReferenceEngineDiagnostic diagnostic;
    diagnostic.code = q3x::runtime::ReferenceEngineError::kRequestStateFailure;
    diagnostic.stage = "request_memory_plan";
    diagnostic.message = request_plan.diagnostic.message;
    diagnostic.context = request_plan.diagnostic.context;
    if (!request_plan.diagnostic.expected.empty() ||
        !request_plan.diagnostic.actual.empty()) {
      diagnostic.context +=
          (diagnostic.context.empty() ? "" : "; ") +
          std::string("expected=") + request_plan.diagnostic.expected +
          "; actual=" + request_plan.diagnostic.actual;
    }
    diagnostic.dependency_error =
        static_cast<int>(request_plan.diagnostic.code);
    diagnostic.cuda_error = request_plan.diagnostic.cuda_error;
    PrintEngineDiagnostic(std::cerr, diagnostic);
    return EngineFailureExitCode(diagnostic.code);
  }
  std::cerr << "progress=loading_benchmark_engine model_dir=";
  PrintEscaped(std::cerr, parsed.value->model_directory.string());
  std::cerr << " max_sequence_length="
            << parsed.value->max_sequence_length
            << " requested_prefill_chunk_size="
            << parsed.value->prefill_chunk_size
            << " effective_prefill_chunk_size="
            << parsed.value->prefill_chunk_size
            << " projection_backend="
            << q3x::runtime::to_string(parsed.value->projection_backend)
            << " decode_graph_cache_policy="
            << DecodeGraphCachePolicyName(
                   engine_options.decode_graph_cache_policy)
            << '\n';
  q3x::runtime::ReferenceEngineCreateResult created =
      q3x::runtime::create_reference_engine(parsed.value->model_directory,
                                            engine_options);
  if (!created) {
    PrintEngineDiagnostic(std::cerr, created.diagnostic);
    return EngineFailureExitCode(created.diagnostic.code);
  }

  std::cerr << "progress=running_benchmark prompts="
            << parsed.value->prompts.size()
            << " warmup_rounds=" << parsed.value->warmup_rounds
            << " measured_rounds=" << parsed.value->iterations
            << " max_tokens=" << parsed.value->max_tokens
            << " nvtx_phase_ranges="
            << (parsed.value->nvtx_phase_ranges ? 1 : 0)
            << " requested_prefill_chunk_size="
            << parsed.value->prefill_chunk_size
            << " effective_prefill_chunk_size="
            << parsed.value->prefill_chunk_size << '\n';
  q3x::runtime::ReferenceBenchmarkResult benchmark =
      q3x::runtime::benchmark_reference_engine(
          *created.value, parsed.value->prompts, benchmark_options);
  if (!benchmark) {
    PrintBenchmarkDiagnostic(std::cerr, benchmark.diagnostic);
    return BenchmarkFailureExitCode(benchmark.diagnostic);
  }
  if (benchmark.value->device_memory.persistent_drop_detected) {
    std::cerr << "warning=device_memory_persistent_drop bytes="
              << benchmark.value->device_memory.persistent_drop_bytes
              << " tolerance="
              << benchmark.value->device_memory.drop_tolerance_bytes << '\n';
  }
  PrintBenchmarkReport(std::cout, parsed.value->model_directory,
                       created.value->load_stats(), *benchmark.value,
                       parsed.value->projection_backend,
                       q3x::runtime::reference_runner_detail::
                           exchange_nvfp4_marlin_prefill_admission_test_hits(
                               0U),
                       q3x::runtime::reference_runner_detail::
                           exchange_fp8_marlin_prefill_admission_test_hits(
                               0U));
  return 0;
}

int RunProbe() {
  if (q3x::core::DeviceCount() == 0) {
    std::cerr << "No CUDA device found.\n";
    return 2;
  }

  const q3x::core::DeviceInfo info = q3x::core::QueryDeviceInfo();
  std::cout << q3x::core::FormatDeviceInfo(info);

  if (info.support == q3x::core::DeviceSupport::kUnsupported) {
    std::cerr << "Unsupported CUDA compute capability. This project supports "
                 "SM80/SM86 generically and optimizes for SM87.\n";
    return 4;
  }

  std::string smoke_error;
  if (!q3x::kernels::RunDeviceSmokeTest(&smoke_error)) {
    std::cerr << "CUDA smoke test: failed: " << smoke_error << '\n';
    return 3;
  }
  std::cout << "  CUDA smoke test: passed\n";

  return 0;
}

int RunModels() {
  std::cout << "Built-in architecture catalog (execution status: target)\n";
  for (const auto& config : q3x::model::known_model_catalog()) {
    std::cout << "  " << config.name << "\n"
              << "    topology: " << q3x::model::to_string(config.topology)
              << ", layers: " << config.num_hidden_layers
              << ", hidden: " << config.hidden_size;
    if (config.is_moe()) {
      std::cout << ", experts: " << config.num_experts << ", top-k: "
                << config.num_experts_per_token;
    }
    std::cout << "\n    source: " << config.hf_repo << '\n';
  }
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string command = argc > 1 ? argv[1] : "help";
  try {
    if (command == "help" || command == "--help" || command == "-h") {
      PrintUsage(std::cout);
      return 0;
    }
    if (command == "version" || command == "--version") {
      std::cout << "Qwen3x-Orin " Q3X_VERSION_STRING << '\n'
                << "CUDA architectures: " Q3X_CUDA_ARCHITECTURES << '\n';
      return 0;
    }
    if (command == "probe") {
      return RunProbe();
    }
    if (command == "models") {
      return RunModels();
    }
    if (command == "generate") {
      return RunGenerate(argc, argv);
    }
    if (command == "benchmark") {
      return RunBenchmark(argc, argv);
    }
    std::cerr << "Unknown command: " << command << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
