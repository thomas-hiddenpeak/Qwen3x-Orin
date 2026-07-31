#include "q3x/server/evaluation_server.h"
#include "q3x/version.h"

#include <atomic>
#include <charconv>
#include <csignal>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>

namespace {

std::atomic<bool> g_stop_requested{false};
static_assert(std::atomic<bool>::is_always_lock_free,
              "the signal stop flag must be lock-free on this target");

extern "C" void RequestStop(const int signal) {
  (void)signal;
  g_stop_requested.store(true, std::memory_order_relaxed);
}

void PrintUsage(std::ostream& output) {
  output
      << "Qwen3x-Orin evaluation server " Q3X_VERSION_STRING "\n\n"
      << "Usage:\n"
      << "  qwen3x-eval-server MODEL_DIR [options]\n\n"
      << "Options:\n"
      << "  --host IPv4                 Loopback only (must be 127.0.0.1)\n"
      << "  --port N                    TCP port (default 8000)\n"
      << "  --model ID                  Served OpenAI model id\n"
      << "  --max-sequence-length N     Resident request capacity (default 8192)\n"
      << "  --max-output-tokens N       Per-request output ceiling (default 4096)\n"
      << "  --prefill-chunk-size N      Native Prefill chunk 1..512 (default 512)\n"
      << "  --prefill-a4-payload FILE   Authenticated full-model A4 payload\n"
      << "  --prefill-a4-policy FILE    Matching 400-projection calibration policy\n"
      << "  --prefill-a4-receipt FILE   Optional receipt (default payload.receipt.json)\n"
      << "  --queue-capacity N          Bounded inference queue, max 62 (default 8)\n"
      << "  --ingress-threads N         Fixed HTTP threads, queue+2 min (default 10)\n"
      << "  --projection-backend sm87|reference (default sm87)\n"
      << "  --request-max-arena-bytes N Request arena hard limit\n"
      << "  --min-free-bytes N          Required free CUDA bytes after arena\n"
      << "  --help                      Show this help\n\n"
      << "Comparator environment:\n"
      << "  Q3X_DISABLE_OPTIMIZED_PREFILL=1\n"
      << "                              Disable optimized P513..P40960 dispatch\n"
      << "                              without changing A4 residency or arena\n\n"
      << "The gateway is intentionally batch-one and greedy. Unsupported sampling,\n"
      << "tools, media, and custom stop semantics fail closed. It is an\n"
      << "unauthenticated evaluation surface, not a production serving API.\n";
}

template <typename T>
[[nodiscard]] bool ParseUnsigned(const std::string_view text,
                                 T& output) noexcept {
  static_assert(std::numeric_limits<T>::is_integer &&
                !std::numeric_limits<T>::is_signed);
  if (text.empty()) {
    return false;
  }
  T value = 0;
  const std::from_chars_result parsed =
      std::from_chars(text.data(), text.data() + text.size(), value, 10);
  if (parsed.ec != std::errc{} ||
      parsed.ptr != text.data() + text.size()) {
    return false;
  }
  output = value;
  return true;
}

[[nodiscard]] bool ParseArguments(
    const int argc, char** const argv,
    q3x::server::EvaluationServerOptions& options,
    std::string& error) {
  if (argc < 2) {
    error = "missing MODEL_DIR";
    return false;
  }
  options.model_directory = argv[1];
  if (options.model_directory.empty()) {
    error = "MODEL_DIR must not be empty";
    return false;
  }
  for (int index = 2; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (index + 1 >= argc) {
      error = std::string(argument) + " requires a value";
      return false;
    }
    const std::string_view value(argv[++index]);
    if (argument == "--host") {
      options.bind_address = value;
    } else if (argument == "--model") {
      options.served_model = value;
    } else if (argument == "--port") {
      std::uint32_t port = 0U;
      if (!ParseUnsigned(value, port) || port == 0U || port > 65'535U) {
        error = "--port must be in [1,65535]";
        return false;
      }
      options.port = static_cast<std::uint16_t>(port);
    } else if (argument == "--max-sequence-length") {
      if (!ParseUnsigned(value, options.max_sequence_length) ||
          options.max_sequence_length == 0U) {
        error = "--max-sequence-length must be positive";
        return false;
      }
    } else if (argument == "--max-output-tokens") {
      if (!ParseUnsigned(value, options.maximum_output_tokens) ||
          options.maximum_output_tokens == 0U) {
        error = "--max-output-tokens must be positive";
        return false;
      }
    } else if (argument == "--prefill-chunk-size") {
      if (!ParseUnsigned(value, options.prefill_chunk_size) ||
          options.prefill_chunk_size == 0U ||
          options.prefill_chunk_size >
              q3x::runtime::kMaximumRequestPrefillChunkSize) {
        error = "--prefill-chunk-size must be in [1,512]";
        return false;
      }
    } else if (argument == "--prefill-a4-payload") {
      if (!options.prefill_a4_payload_path.empty() || value.empty()) {
        error = "--prefill-a4-payload requires one non-empty FILE";
        return false;
      }
      options.prefill_a4_payload_path = value;
    } else if (argument == "--prefill-a4-policy") {
      if (!options.prefill_a4_calibration_policy_path.empty() ||
          value.empty()) {
        error = "--prefill-a4-policy requires one non-empty FILE";
        return false;
      }
      options.prefill_a4_calibration_policy_path = value;
    } else if (argument == "--prefill-a4-receipt") {
      if (!options.prefill_a4_receipt_path.empty() || value.empty()) {
        error = "--prefill-a4-receipt requires one non-empty FILE";
        return false;
      }
      options.prefill_a4_receipt_path = value;
    } else if (argument == "--queue-capacity") {
      if (!ParseUnsigned(value, options.inference_queue_capacity) ||
          options.inference_queue_capacity == 0U) {
        error = "--queue-capacity must be positive";
        return false;
      }
    } else if (argument == "--ingress-threads") {
      if (!ParseUnsigned(value, options.ingress_threads) ||
          options.ingress_threads == 0U || options.ingress_threads > 64U) {
        error = "--ingress-threads must be in [1,64]";
        return false;
      }
      options.accepted_connection_capacity =
          std::max<std::size_t>(16U, options.ingress_threads * 4U);
    } else if (argument == "--projection-backend") {
      if (value == "sm87") {
        options.projection_backend =
            q3x::runtime::ProjectionBackend::kSm87WeightOnly;
      } else if (value == "reference") {
        options.projection_backend =
            q3x::runtime::ProjectionBackend::kReference;
      } else {
        error = "--projection-backend must be sm87 or reference";
        return false;
      }
    } else if (argument == "--request-max-arena-bytes") {
      if (!ParseUnsigned(value, options.request_max_arena_bytes) ||
          options.request_max_arena_bytes == 0U) {
        error = "--request-max-arena-bytes must be positive";
        return false;
      }
    } else if (argument == "--min-free-bytes") {
      if (!ParseUnsigned(value,
                         options.request_min_free_bytes_after_create)) {
        error = "--min-free-bytes must be an unsigned integer";
        return false;
      }
    } else {
      error = "unknown option: " + std::string(argument);
      return false;
    }
  }
  const bool a4_payload = !options.prefill_a4_payload_path.empty();
  const bool a4_policy =
      !options.prefill_a4_calibration_policy_path.empty();
  const bool a4_receipt = !options.prefill_a4_receipt_path.empty();
  if (a4_payload != a4_policy || (a4_receipt && !a4_payload)) {
    error = "--prefill-a4-payload and --prefill-a4-policy are required "
            "together; receipt is optional";
    return false;
  }
  return true;
}

}  // namespace

int main(const int argc, char** const argv) {
  if (argc == 2 &&
      (std::string_view(argv[1]) == "--help" ||
       std::string_view(argv[1]) == "-h")) {
    PrintUsage(std::cout);
    return 0;
  }
  q3x::server::EvaluationServerOptions options;
  std::string error;
  if (!ParseArguments(argc, argv, options, error)) {
    std::cerr << "error: " << error << "\n\n";
    PrintUsage(std::cerr);
    return 2;
  }
  std::signal(SIGINT, RequestStop);
  std::signal(SIGTERM, RequestStop);
  try {
    const int status = q3x::server::run_evaluation_server(
        options, g_stop_requested, error);
    if (status != 0) {
      std::cerr << "error: " << error << '\n';
    }
    return status;
  } catch (const std::exception& exception) {
    std::cerr << "fatal evaluation server error: " << exception.what()
              << '\n';
    return 1;
  }
}
