#include "q3x/core/device_info.h"
#include "q3x/kernels/device_probe.h"
#include "q3x/model/model_config.h"
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

void PrintUsage(std::ostream& output) {
  output
      << "Qwen3x-Orin " Q3X_VERSION_STRING "\n\n"
      << "Usage:\n"
      << "  qwen3x-orin version    Show build information\n"
      << "  qwen3x-orin probe      Inspect the CUDA device and run a smoke kernel\n"
      << "  qwen3x-orin models     List catalogued target architectures\n"
      << "  qwen3x-orin generate MODEL_DIR --prompt TEXT "
         "[--max-tokens N] [--trace]\n"
      << "  qwen3x-orin help       Show this help\n";
}

void PrintGenerateUsage(std::ostream& output) {
  output
      << "Usage:\n"
      << "  qwen3x-orin generate MODEL_DIR --prompt TEXT "
         "[--max-tokens N] [--trace]\n\n"
      << "Options:\n"
      << "  --prompt TEXT    One user message; thinking is disabled\n"
      << "  --max-tokens N  Greedy output limit (1.."
      << kCliMaximumMaxTokens << ", default " << kCliDefaultMaxTokens
      << ")\n"
      << "  --trace         Capture per-step activation boundary digests\n";
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

struct GenerateCliOptions {
  std::filesystem::path model_directory;
  std::string prompt;
  std::uint32_t max_tokens = kCliDefaultMaxTokens;
  bool trace = false;
};

struct GenerateParseResult {
  std::optional<GenerateCliOptions> value;
  std::string error;
};

[[nodiscard]] bool ParsePositiveU32(const std::string_view text,
                                    std::uint32_t& value) noexcept {
  if (text.empty()) {
    return false;
  }
  std::uint32_t parsed = 0U;
  const char* const begin = text.data();
  const char* const end = begin + text.size();
  const std::from_chars_result converted =
      std::from_chars(begin, end, parsed, 10);
  if (converted.ec != std::errc{} || converted.ptr != end || parsed == 0U) {
    return false;
  }
  value = parsed;
  return true;
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
  bool trace_seen = false;
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
    const q3x::runtime::ReferenceOneShotGeneration& result) {
  const auto& load = result.load;
  const auto& generation = result.generation;
  output << std::fixed << std::setprecision(3)
         << "status=ok\n"
         << "load.total_ms=" << load.total_milliseconds << '\n'
         << "load.tokenizer_ms=" << load.tokenizer_milliseconds << '\n'
         << "load.resident_ms=" << load.resident_load_milliseconds << '\n'
         << "load.weight_bind_ms=" << load.weight_bind_milliseconds << '\n'
         << "load.request_state_ms=" << load.request_state_milliseconds << '\n'
         << "load.runner_factory_ms=" << load.runner_factory_milliseconds
         << '\n'
         << "load.resident.bytes_read=" << load.resident.bytes_read << '\n'
         << "load.resident.bytes_copied=" << load.resident.bytes_copied
         << '\n'
         << "load.resident.bytes_skipped=" << load.resident.bytes_skipped
         << '\n'
         << "load.resident.chunks=" << load.resident.chunks << '\n'
         << "load.resident.memcpy_operations="
         << load.resident.memcpy_operations << '\n'
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
         << load.request_max_sequence_length << '\n';
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
         << "timing.prompt_prefill_ms="
         << generation.timing.prompt_prefill_milliseconds << '\n'
         << "timing.time_to_first_token_ms="
         << generation.timing.time_to_first_token_milliseconds << '\n'
         << "timing.decode_after_first_ms="
         << generation.timing.decode_after_first_milliseconds << '\n'
         << "timing.total_generation_ms="
         << generation.timing.total_generation_milliseconds << '\n';
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
  std::cerr << "progress=loading_and_generating model_dir=";
  PrintEscaped(std::cerr, parsed.value->model_directory.string());
  std::cerr << " max_tokens=" << parsed.value->max_tokens
            << " trace=" << (parsed.value->trace ? 1 : 0) << '\n';
  q3x::runtime::ReferenceOneShotResult generated =
      q3x::runtime::generate_reference(parsed.value->model_directory,
                                       parsed.value->prompt, options);
  if (!generated) {
    PrintEngineDiagnostic(std::cerr, generated.diagnostic);
    return EngineFailureExitCode(generated.diagnostic.code);
  }
  PrintGeneration(std::cout, *generated.value);
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
    std::cerr << "Unknown command: " << command << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  } catch (const std::exception& error) {
    std::cerr << "error: " << error.what() << '\n';
    return 1;
  }
}
