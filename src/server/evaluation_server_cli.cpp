#include "q3x/server/evaluation_server_cli.h"

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>

namespace q3x::server {
namespace {

template <typename T>
[[nodiscard]] bool parse_unsigned(const std::string_view text,
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

}  // namespace

bool parse_evaluation_server_arguments(
    const int argc, const char* const* const argv,
    EvaluationServerOptions& options, std::string& error) {
  if (argc < 2) {
    error = "missing MODEL_DIR";
    return false;
  }
  options.model_directory = argv[1];
  if (options.model_directory.empty()) {
    error = "MODEL_DIR must not be empty";
    return false;
  }
  bool profile_request_index_seen = false;
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
      if (!parse_unsigned(value, port) || port == 0U || port > 65'535U) {
        error = "--port must be in [1,65535]";
        return false;
      }
      options.port = static_cast<std::uint16_t>(port);
    } else if (argument == "--max-sequence-length") {
      if (!parse_unsigned(value, options.max_sequence_length) ||
          options.max_sequence_length == 0U) {
        error = "--max-sequence-length must be positive";
        return false;
      }
    } else if (argument == "--max-output-tokens") {
      if (!parse_unsigned(value, options.maximum_output_tokens) ||
          options.maximum_output_tokens == 0U) {
        error = "--max-output-tokens must be positive";
        return false;
      }
    } else if (argument == "--prefill-chunk-size") {
      if (!parse_unsigned(value, options.prefill_chunk_size) ||
          options.prefill_chunk_size == 0U ||
          options.prefill_chunk_size >
              runtime::kMaximumRequestPrefillChunkSize) {
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
    } else if (argument == "--prefill-attention-o-k512-payload") {
      if (!options.prefill_attention_o_k512_payload_path.empty() ||
          value.empty()) {
        error = "--prefill-attention-o-k512-payload requires one non-empty "
                "FILE";
        return false;
      }
      options.prefill_attention_o_k512_payload_path = value;
    } else if (argument == "--prefill-attention-o-k512-policy") {
      if (!options.prefill_attention_o_k512_policy_path.empty() ||
          value.empty()) {
        error = "--prefill-attention-o-k512-policy requires one non-empty "
                "FILE";
        return false;
      }
      options.prefill_attention_o_k512_policy_path = value;
    } else if (argument == "--prefill-attention-o-k512-receipt") {
      if (!options.prefill_attention_o_k512_receipt_path.empty() ||
          value.empty()) {
        error = "--prefill-attention-o-k512-receipt requires one non-empty "
                "FILE";
        return false;
      }
      options.prefill_attention_o_k512_receipt_path = value;
    } else if (argument == "--prefill-mlp-k512-payload") {
      if (!options.prefill_mlp_k512_payload_path.empty() || value.empty()) {
        error = "--prefill-mlp-k512-payload requires one non-empty FILE";
        return false;
      }
      options.prefill_mlp_k512_payload_path = value;
    } else if (argument == "--prefill-mlp-k512-policy") {
      if (!options.prefill_mlp_k512_policy_path.empty() || value.empty()) {
        error = "--prefill-mlp-k512-policy requires one non-empty FILE";
        return false;
      }
      options.prefill_mlp_k512_policy_path = value;
    } else if (argument == "--prefill-mlp-k512-receipt") {
      if (!options.prefill_mlp_k512_receipt_path.empty() || value.empty()) {
        error = "--prefill-mlp-k512-receipt requires one non-empty FILE";
        return false;
      }
      options.prefill_mlp_k512_receipt_path = value;
    } else if (argument ==
               "--prefill-mlp-k512-fragment-native-payload") {
      if (!options.prefill_mlp_k512_fragment_native_payload_path.empty() ||
          value.empty()) {
        error = "--prefill-mlp-k512-fragment-native-payload requires one "
                "non-empty FILE";
        return false;
      }
      options.prefill_mlp_k512_fragment_native_payload_path = value;
    } else if (argument ==
               "--prefill-mlp-k512-fragment-native-policy") {
      if (!options.prefill_mlp_k512_fragment_native_policy_path.empty() ||
          value.empty()) {
        error = "--prefill-mlp-k512-fragment-native-policy requires one "
                "non-empty FILE";
        return false;
      }
      options.prefill_mlp_k512_fragment_native_policy_path = value;
    } else if (argument ==
               "--prefill-mlp-k512-fragment-native-receipt") {
      if (!options.prefill_mlp_k512_fragment_native_receipt_path.empty() ||
          value.empty()) {
        error = "--prefill-mlp-k512-fragment-native-receipt requires one "
                "non-empty FILE";
        return false;
      }
      options.prefill_mlp_k512_fragment_native_receipt_path = value;
    } else if (argument == "--queue-capacity") {
      if (!parse_unsigned(value, options.inference_queue_capacity) ||
          options.inference_queue_capacity == 0U) {
        error = "--queue-capacity must be positive";
        return false;
      }
    } else if (argument == "--ingress-threads") {
      if (!parse_unsigned(value, options.ingress_threads) ||
          options.ingress_threads == 0U || options.ingress_threads > 64U) {
        error = "--ingress-threads must be in [1,64]";
        return false;
      }
      options.accepted_connection_capacity =
          std::max<std::size_t>(16U, options.ingress_threads * 4U);
    } else if (argument == "--projection-backend") {
      if (value == "sm87") {
        options.projection_backend =
            runtime::ProjectionBackend::kSm87WeightOnly;
      } else if (value == "reference") {
        options.projection_backend = runtime::ProjectionBackend::kReference;
      } else {
        error = "--projection-backend must be sm87 or reference";
        return false;
      }
    } else if (argument == "--request-max-arena-bytes") {
      if (!parse_unsigned(value, options.request_max_arena_bytes) ||
          options.request_max_arena_bytes == 0U) {
        error = "--request-max-arena-bytes must be positive";
        return false;
      }
    } else if (argument == "--min-free-bytes") {
      if (!parse_unsigned(
              value, options.request_min_free_bytes_after_create)) {
        error = "--min-free-bytes must be an unsigned integer";
        return false;
      }
    } else if (argument == "--profile-request-index") {
      if (profile_request_index_seen ||
          !parse_unsigned(value, options.profile_request_index)) {
        error = "--profile-request-index requires one unsigned integer";
        return false;
      }
      profile_request_index_seen = true;
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
  const bool attention_k512_payload =
      !options.prefill_attention_o_k512_payload_path.empty();
  const bool attention_k512_policy =
      !options.prefill_attention_o_k512_policy_path.empty();
  const bool attention_k512_receipt =
      !options.prefill_attention_o_k512_receipt_path.empty();
  if (attention_k512_payload != attention_k512_policy ||
      attention_k512_payload != attention_k512_receipt) {
    error = "--prefill-attention-o-k512-payload, "
            "--prefill-attention-o-k512-policy, and "
            "--prefill-attention-o-k512-receipt are required together";
    return false;
  }
  if (attention_k512_payload && !a4_payload) {
    error = "the K512 Attention-O overlay requires the explicit K128 A4 "
            "payload and policy";
    return false;
  }
  const bool mlp_k512_payload =
      !options.prefill_mlp_k512_payload_path.empty();
  const bool mlp_k512_policy =
      !options.prefill_mlp_k512_policy_path.empty();
  const bool mlp_k512_receipt =
      !options.prefill_mlp_k512_receipt_path.empty();
  if (mlp_k512_payload != mlp_k512_policy ||
      mlp_k512_payload != mlp_k512_receipt) {
    error = "--prefill-mlp-k512-payload, --prefill-mlp-k512-policy, and "
            "--prefill-mlp-k512-receipt are required together";
    return false;
  }
  if (mlp_k512_payload && !a4_payload) {
    error = "the K512 MLP overlay requires the explicit K128 A4 payload "
            "and policy";
    return false;
  }
  const bool mlp_k512_fragment_native_payload =
      !options.prefill_mlp_k512_fragment_native_payload_path.empty();
  const bool mlp_k512_fragment_native_policy =
      !options.prefill_mlp_k512_fragment_native_policy_path.empty();
  const bool mlp_k512_fragment_native_receipt =
      !options.prefill_mlp_k512_fragment_native_receipt_path.empty();
  if (mlp_k512_fragment_native_payload !=
          mlp_k512_fragment_native_policy ||
      mlp_k512_fragment_native_payload !=
          mlp_k512_fragment_native_receipt) {
    error = "--prefill-mlp-k512-fragment-native-payload, "
            "--prefill-mlp-k512-fragment-native-policy, and "
            "--prefill-mlp-k512-fragment-native-receipt are required "
            "together";
    return false;
  }
  if (mlp_k512_fragment_native_payload && !a4_payload) {
    error = "the fragment-native K512 MLP overlay requires the explicit "
            "K128 A4 payload and policy";
    return false;
  }
  if (mlp_k512_payload && mlp_k512_fragment_native_payload) {
    error = "the K512 MLP v1 and fragment-native v2 publications are "
            "mutually exclusive";
    return false;
  }
  error.clear();
  return true;
}

}  // namespace q3x::server
