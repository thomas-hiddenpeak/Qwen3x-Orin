#include "q3x/server/evaluation_server.h"
#include "q3x/server/evaluation_server_cli.h"
#include "q3x/version.h"

#include <atomic>
#include <csignal>
#include <exception>
#include <iostream>
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
      << "  --profile-request-index N   Profile 1-based /v1/completions request\n"
      << "                              (0 disables; BUILD_TESTING only)\n"
      << "  --help                      Show this help\n\n"
      << "Comparator environment:\n"
      << "  Q3X_DISABLE_OPTIMIZED_PREFILL=1\n"
      << "                              Disable optimized P513..P40960 dispatch\n"
      << "                              without changing A4 residency or arena\n\n"
      << "The gateway is intentionally batch-one and greedy. Unsupported sampling,\n"
      << "tools, media, and custom stop semantics fail closed. It is an\n"
      << "unauthenticated evaluation surface, not a production serving API.\n";
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
  if (!q3x::server::parse_evaluation_server_arguments(
          argc, argv, options, error)) {
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
