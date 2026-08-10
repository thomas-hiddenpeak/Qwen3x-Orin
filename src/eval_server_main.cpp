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
      << "  --prefill-execution-mode legacy|layer-major (default legacy)\n"
      << "  --prefill-attention-tactic exact-segmented|native-group-q64-panel|\n"
      << "                             native-group-q128-v4-panel|\n"
      << "                             native-flashinfer-exact-panel|\n"
      << "                             native-flashinfer-exact-whole-prompt\n"
      << "  --prefill-projection-tactic exact-segmented|\n"
      << "                              segmented-marlin-operator-panel|\n"
      << "                              native-quantized-large-m-operator-panel|\n"
      << "                              native-nvfp4-true-large-m-operator-panel|\n"
      << "                              native-nvfp4-g2-d2-large-m-operator-panel|\n"
      << "                              native-nvfp4-persistent-p40-layer-wide-mlp|\n"
      << "                              native-prompt-wide-p40-whole-core|\n"
      << "                              native-prompt-wide-p40-projection-reset\n"
      << "  --nvtx-phase-ranges        Emit generation/Prefill/Decode ranges\n"
      << "  --queue-capacity N          Bounded inference queue, max 62 (default 8)\n"
      << "  --ingress-threads N         Fixed HTTP threads, queue+2 min (default 10)\n"
      << "  --projection-backend sm87|reference (default sm87)\n"
      << "  --request-max-arena-bytes N Request arena hard limit\n"
      << "  --min-free-bytes N          Required free CUDA bytes after arena\n"
      << "  --help                      Show this help\n\n"
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
    if (argument == "--nvtx-phase-ranges") {
      options.emit_nvtx_phase_ranges = true;
      continue;
    }
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
    } else if (argument == "--prefill-execution-mode") {
      if (value == "legacy") {
        options.prefill_execution_mode =
            q3x::runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled;
      } else if (value == "layer-major") {
        options.prefill_execution_mode = q3x::runtime::
            ReferencePrefillExecutionMode::kWholeRequestLayerMajor;
      } else {
        error = "--prefill-execution-mode must be legacy or layer-major";
        return false;
      }
    } else if (argument == "--prefill-attention-tactic") {
      if (value == "exact-segmented") {
        options.prefill_full_attention_tactic = q3x::runtime::
            LayerMajorPrefillFullAttentionTactic::kExactSegmentedC512;
      } else if (value == "native-group-q64-panel") {
        options.prefill_full_attention_tactic = q3x::runtime::
            LayerMajorPrefillFullAttentionTactic::kNativeGroupQ64Panel;
      } else if (value == "native-group-q128-v4-panel") {
        options.prefill_full_attention_tactic = q3x::runtime::
            LayerMajorPrefillFullAttentionTactic::kNativeGroupQ128V4Panel;
      } else if (value == "native-flashinfer-exact-panel") {
        options.prefill_full_attention_tactic = q3x::runtime::
            LayerMajorPrefillFullAttentionTactic::
                kNativeFlashInferExactPanel;
      } else if (value == "native-flashinfer-exact-whole-prompt") {
        options.prefill_full_attention_tactic = q3x::runtime::
            LayerMajorPrefillFullAttentionTactic::
                kNativeFlashInferExactWholePrompt;
      } else {
        error = "--prefill-attention-tactic must be exact-segmented or "
                "native-group-q64-panel or native-group-q128-v4-panel or "
                "native-flashinfer-exact-panel or "
                "native-flashinfer-exact-whole-prompt";
        return false;
      }
    } else if (argument == "--prefill-projection-tactic") {
      if (value == "exact-segmented") {
        options.prefill_projection_tactic = q3x::runtime::
            LayerMajorPrefillProjectionTactic::kExactSegmentedC512;
      } else if (value == "segmented-marlin-operator-panel") {
        options.prefill_projection_tactic = q3x::runtime::
            LayerMajorPrefillProjectionTactic::kSegmentedMarlinOperatorPanel;
      } else if (value ==
                 "native-quantized-large-m-operator-panel") {
        options.prefill_projection_tactic = q3x::runtime::
            LayerMajorPrefillProjectionTactic::
                kNativeQuantizedLargeMOperatorPanel;
      } else if (value ==
                 "native-nvfp4-true-large-m-operator-panel") {
        options.prefill_projection_tactic = q3x::runtime::
            LayerMajorPrefillProjectionTactic::
                kNativeNvfp4TrueLargeMOperatorPanel;
      } else if (value ==
                 "native-nvfp4-g2-d2-large-m-operator-panel") {
        options.prefill_projection_tactic = q3x::runtime::
            LayerMajorPrefillProjectionTactic::
                kNativeNvfp4G2D2LargeMOperatorPanel;
      } else if (value ==
                 "native-nvfp4-persistent-p40-layer-wide-mlp") {
        options.prefill_projection_tactic = q3x::runtime::
            LayerMajorPrefillProjectionTactic::
                kNativeNvfp4PersistentP40LayerWideMlp;
      } else if (value == "native-prompt-wide-p40-whole-core") {
        options.prefill_projection_tactic = q3x::runtime::
            LayerMajorPrefillProjectionTactic::
                kNativePromptWideP40WholeCore;
      } else if (value ==
                 "native-prompt-wide-p40-projection-reset") {
        options.prefill_projection_tactic = q3x::runtime::
            LayerMajorPrefillProjectionTactic::
                kNativePromptWideP40ProjectionReset;
      } else {
        error = "--prefill-projection-tactic must be exact-segmented or "
                "segmented-marlin-operator-panel or "
                "native-quantized-large-m-operator-panel or "
                "native-nvfp4-true-large-m-operator-panel or "
                "native-nvfp4-g2-d2-large-m-operator-panel or "
                "native-nvfp4-persistent-p40-layer-wide-mlp or "
                "native-prompt-wide-p40-whole-core or "
                "native-prompt-wide-p40-projection-reset";
        return false;
      }
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
