#pragma once

#include "q3x/runtime/reference_engine.h"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>

namespace q3x::server {

struct EvaluationServerOptions {
  std::filesystem::path model_directory;
  std::filesystem::path prefill_a4_payload_path;
  std::filesystem::path prefill_a4_calibration_policy_path;
  std::filesystem::path prefill_a4_receipt_path;
  // Authenticated K512 Attention-O overlay. The evaluation surface requires
  // an explicit payload, policy, and receipt as one inseparable publication;
  // it is valid only on top of the complete K128 A4 base above.
  std::filesystem::path prefill_attention_o_k512_payload_path;
  std::filesystem::path prefill_attention_o_k512_policy_path;
  std::filesystem::path prefill_attention_o_k512_receipt_path;
  // Authenticated K512 MLP overlay for every Gate, Up, and Down projection.
  // As with Attention-O, the evaluation surface admits only a complete
  // payload/policy/receipt publication on top of the explicit K128 A4 base.
  std::filesystem::path prefill_mlp_k512_payload_path;
  std::filesystem::path prefill_mlp_k512_policy_path;
  std::filesystem::path prefill_mlp_k512_receipt_path;
  // Authenticated fragment-native K512 MLP v2 publication.  Its policy is
  // the exact source-v1 policy bound by the v2 receipt.  The triplet is
  // all-or-none and mutually exclusive with the v1 triplet above.
  std::filesystem::path prefill_mlp_k512_fragment_native_payload_path;
  std::filesystem::path prefill_mlp_k512_fragment_native_policy_path;
  std::filesystem::path prefill_mlp_k512_fragment_native_receipt_path;
  // Authenticated paired-GateUp/canonical-v1-Down hybrid publication.  This
  // is a distinct all-or-none triplet and never impersonates v2.
  std::filesystem::path
      prefill_mlp_k512_paired_gateup_canonical_down_payload_path;
  std::filesystem::path
      prefill_mlp_k512_paired_gateup_canonical_down_policy_path;
  std::filesystem::path
      prefill_mlp_k512_paired_gateup_canonical_down_receipt_path;
  // Authenticated projection-major-GateUp/canonical-v1-Down publication.
  // This is a fourth, independently authenticated all-or-none triplet and is
  // mutually exclusive with every older K512 MLP publication.
  std::filesystem::path
      prefill_mlp_k512_projection_major_gateup_canonical_down_payload_path;
  std::filesystem::path
      prefill_mlp_k512_projection_major_gateup_canonical_down_policy_path;
  std::filesystem::path
      prefill_mlp_k512_projection_major_gateup_canonical_down_receipt_path;
  std::string bind_address = "127.0.0.1";
  std::uint16_t port = 8000U;
  std::string served_model = "qwen3.6-27b-nvfp4";
  std::uint32_t max_sequence_length = 8'192U;
  std::uint32_t maximum_output_tokens = 4'096U;
  std::uint32_t prefill_chunk_size =
      runtime::kMaximumRequestPrefillChunkSize;
  // One thread may wait on each admitted batch-one request. Keep enough
  // threads for the active request, every bounded queued request, and one
  // control-plane/overload response.
  std::size_t ingress_threads = 10U;
  std::size_t accepted_connection_capacity = 16U;
  std::size_t inference_queue_capacity = 8U;
  std::size_t stream_event_capacity = 16U;
  std::uint32_t read_timeout_milliseconds = 10'000U;
  std::uint32_t write_timeout_milliseconds = 5'000U;
  std::uint64_t request_max_arena_bytes =
      2ULL * 1024ULL * 1024ULL * 1024ULL;
  std::uint64_t request_min_free_bytes_after_create =
      8ULL * 1024ULL * 1024ULL * 1024ULL;
  // Diagnostic-only, 1-based index among /v1/completions jobs dequeued by
  // the single inference worker. Zero leaves profiler APIs and synchronization
  // entirely outside the request path.
  std::uint64_t profile_request_index = 0U;
  runtime::ProjectionBackend projection_backend =
      runtime::ProjectionBackend::kSm87WeightOnly;
};

// True only for BUILD_TESTING builds that explicitly link the CUDA profiler
// API into the evaluation gateway. Non-testing builds reject a nonzero
// profile_request_index before loading the model.
[[nodiscard]] bool evaluation_server_request_profiling_compiled() noexcept;

// Loads one resident model, starts a bounded HTTP ingress and exactly one
// inference worker, and blocks until stop_requested becomes true or a fatal
// server error occurs. The listener is not exposed until model loading has
// succeeded.
[[nodiscard]] int run_evaluation_server(
    const EvaluationServerOptions& options,
    std::atomic<bool>& stop_requested,
    std::string& error_message);

}  // namespace q3x::server
