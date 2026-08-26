#include "q3x/core/sha256.h"
#include "q3x/io/json.h"
#include "q3x/runtime/prefill_execution_plan.h"
#include "q3x/runtime/prefill_workspace_plan.h"
#include "q3x/runtime/reference_engine.h"
#include "q3x/runtime/reference_runner.h"

#include "reference_engine_final_token_policy_internal.h"
#include "reference_engine_prefill_authority.h"
#include "reference_runner_gdn_chunk64_native_admission.h"
#include "reference_runner_selector_exact_persistent_attention_v1_internal.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <new>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace {

namespace core = q3x::core;
namespace runtime = q3x::runtime;
namespace engine_detail = q3x::runtime::reference_engine_detail;
namespace runner_detail = q3x::runtime::reference_runner_detail;

#if !defined(Q3X_ENABLE_SELECTOR_EXACT_PERSISTENT_ATTENTION_V1_P40_TESTING)
#error "The P40000 strict capture must be built only by the candidate profile"
#endif

constexpr std::string_view kSchema = "q3x.p40000-o16.strict-capture.v1";
constexpr std::string_view kMarker = "P40000_STRICT_CAPTURE_JSON ";
constexpr std::string_view kArmLegacy = "legacy-c512-exact";
constexpr std::string_view kArmSelector = "p40016-whole-core-selector";
constexpr std::string_view kLegacyOrdinaryProfileId =
    "q3x.sm87.production.p40.legacy-c512-exact.v3";
constexpr std::string_view kCorpusFileSha256 =
    "8970ac50693f49d1b27d35a0610ecbe5072594330d69b301f4dab731789b6844";
constexpr std::string_view kCorpusTokenSha256 =
    "76cd23e2d60a9473af0ff24767b1b7ca614b36857697b420246731165156f78f";
constexpr std::string_view kCheckpointRepository =
    "nvidia/Qwen3.6-27B-NVFP4";
constexpr std::string_view kCheckpointRevision =
    "0893e1606ff3d5f97a441f405d5fc541a6bdf404";

constexpr std::uint32_t kPromptTokens = 40'000U;
constexpr std::uint32_t kMaxNewTokens = 16U;
constexpr std::uint32_t kRequiredSteps = 40'015U;
constexpr std::uint32_t kLegacyCapacity = 44'095U;
constexpr std::uint32_t kSelectorCapacity = 40'016U;
constexpr std::uint64_t kLegacyArenaBytes = 3'070'908'416ULL;
constexpr std::uint64_t kSelectorArenaBytes = 8'641'684'992ULL;
constexpr std::uint32_t kStopTokenId = 248'056U;
constexpr std::size_t kHiddenSize = runtime::kReferenceHiddenSize;
constexpr std::size_t kBf16Bytes = sizeof(std::uint16_t);
constexpr std::size_t kHiddenRowBytes = kHiddenSize * kBf16Bytes;
constexpr std::size_t kPromptResidualBytes =
    static_cast<std::size_t>(kPromptTokens) * kHiddenRowBytes;
constexpr std::size_t kCopyChunkBytes = 4U * 1024U * 1024U;
constexpr std::size_t kKvBytesPerLayerPosition = 4U * 256U * kBf16Bytes;
constexpr std::size_t kLegacyTileCount = 79U;
constexpr std::size_t kFullAttentionLayerCount =
    runtime::kRequestFullLayerCount;
static_assert(kFullAttentionLayerCount == 16U);
static_assert(kPromptResidualBytes == 409'600'000U);
static_assert(kRequiredSteps == kPromptTokens + kMaxNewTokens - 1U);
static_assert(
    kRequiredSteps ==
    runtime::kSelectorExactPersistentAttentionV1P40RequiredSteps);
static_assert(
    kSelectorCapacity ==
    runtime::kSelectorExactPersistentAttentionV1P40WholeCoreRequestCapacityTokens);
static_assert(
    kSelectorArenaBytes ==
    runtime::kSelectorExactPersistentAttentionV1P40WholeCoreArenaBytes);
static_assert(
    runtime::layer_major_p40_whole_core_gqa_scratch_capacity_tokens(
        kSelectorCapacity) == kSelectorCapacity);

struct CheckpointFileContract {
  std::string_view relative_path;
  std::uint64_t bytes = 0U;
  std::string_view sha256;
};

constexpr std::array<CheckpointFileContract, 7U> kCheckpointFiles{{
    {"config.json", 88'567U,
     "c04a19ba293737ad7be4f6e96d6666cb7e479cbe19ecc0c289fad267135b0338"},
    {"hf_quant_config.json", 54'902U,
     "fd7200cd8bca2a8a5d777061521abf83e2deb97ab6bc2f04e7a0a3d3f8ecd5c1"},
    {"model.safetensors.index.json", 214'866U,
     "7aa103a2582b7d26631988de33dea19e8a308ee9c239e8e14feb374af30905e2"},
    {"tokenizer.json", 12'807'982U,
     "5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42"},
    {"model-00001-of-00003.safetensors", 9'965'652'512ULL,
     "b4a0d9a57ff1859dac1144b53ca285011db072737d8813fc16d8d1e07ecae17d"},
    {"model-00002-of-00003.safetensors", 9'985'757'032ULL,
     "06da4242b0f491118d19d4d4c7564307a7bd6059c6bed284e08c93f6fc5a556d"},
    {"model-00003-of-00003.safetensors", 1'970'287'640ULL,
     "e90f5b2bb16814a0565de284ea179edec201edfb120d13f1debaab66f9e60845"},
}};

struct CheckpointAttestation {
  std::array<core::Sha256Digest, kCheckpointFiles.size()> digests{};
  bool complete = false;
};

struct Corpus {
  std::vector<std::uint32_t> tokens;
  core::Sha256Digest file_sha256;
  core::Sha256Digest token_sha256;
};

enum class CaptureError : std::uint8_t {
  kNone = 0,
  kDuplicateHook,
  kInvalidIdentity,
  kInvalidSequence,
  kInvalidRegion,
  kCuda,
  kHash,
  kLegacyResidual,
};

const char* to_string(const CaptureError error) noexcept {
  switch (error) {
    case CaptureError::kNone:
      return "none";
    case CaptureError::kDuplicateHook:
      return "duplicate-hook";
    case CaptureError::kInvalidIdentity:
      return "invalid-identity";
    case CaptureError::kInvalidSequence:
      return "invalid-sequence";
    case CaptureError::kInvalidRegion:
      return "invalid-region";
    case CaptureError::kCuda:
      return "cuda";
    case CaptureError::kHash:
      return "hash";
    case CaptureError::kLegacyResidual:
      return "legacy-residual";
  }
  return "unknown";
}

struct LegacyPromptResidualCapture {
  core::Sha256 hash;
  std::array<std::uint32_t, kLegacyTileCount> first_positions{};
  std::array<std::uint32_t, kLegacyTileCount> tile_rows{};
  std::vector<std::uint8_t> scratch;
  core::Sha256Digest digest;
  std::size_t tile_count = 0U;
  std::size_t total_bytes = 0U;
  std::uint32_t next_row = 0U;
  int cuda_error = static_cast<int>(cudaSuccess);
  CaptureError error = CaptureError::kNone;
  bool finalized = false;

  LegacyPromptResidualCapture() : scratch(kCopyChunkBytes) {}
};

struct StateSnapshot {
  std::uint32_t expected_sequence_length = 0U;
  std::uint32_t sequence_length = 0U;
  std::size_t hook_calls = 0U;
  std::size_t used_kv_bytes_per_layer = 0U;
  core::Sha256Digest conv_state;
  core::Sha256Digest gdn_state;
  std::array<core::Sha256Digest, kFullAttentionLayerCount> key_cache{};
  std::array<core::Sha256Digest, kFullAttentionLayerCount> value_cache{};
  core::Sha256Digest prompt_residual;
  bool prompt_residual_live_capture = false;
  bool prompt_residual_carried_from_prefill = false;
  std::uint32_t prompt_residual_source_sequence_length = 0U;
  std::string_view prompt_residual_source_owner;
  std::string_view prompt_residual_source_buffer;
  std::uint64_t prompt_residual_source_region_bytes = 0U;
  std::uint64_t prompt_residual_source_region_element_capacity = 0U;
  std::uint64_t kv_storage_bytes_per_layer = 0U;
  std::uint64_t kv_storage_element_capacity_per_layer = 0U;
  core::Sha256Digest final_hidden;
  core::Sha256Digest aggregate;
  std::string_view final_hidden_source_owner;
  std::string_view final_hidden_source_buffer;
  std::size_t final_hidden_source_byte_offset = 0U;
  std::uint64_t final_hidden_source_region_bytes = 0U;
  std::uint64_t final_hidden_source_region_element_capacity = 0U;
  std::uint32_t final_hidden_source_row_capacity = 0U;
  std::uint64_t final_hidden_source_row_stride_elements = 0U;
  std::uint32_t final_hidden_position = 0U;
  int cuda_error = static_cast<int>(cudaSuccess);
  CaptureError error = CaptureError::kNone;
  std::vector<std::uint8_t> scratch;

  explicit StateSnapshot(const std::uint32_t expected)
      : expected_sequence_length(expected), scratch(kCopyChunkBytes) {}
};

struct TokenEvent {
  std::size_t index = 0U;
  std::uint32_t token_id = 0U;
  std::string text_delta;
  std::string generated_text;
  bool is_stop_token = false;
};

struct TokenCollector {
  std::vector<TokenEvent> events;
  bool failed = false;

  TokenCollector() { events.reserve(kMaxNewTokens); }
};

struct StrictCaptureContext {
  explicit StrictCaptureContext(const bool selector)
      : selector_arm(selector), prefill(kPromptTokens), returned(kRequiredSteps) {}

  bool selector_arm = false;
  LegacyPromptResidualCapture legacy_residual;
  StateSnapshot prefill;
  StateSnapshot returned;
  core::Sha256Digest prompt_digest;
  engine_detail::P40000SelectorExactPersistentAttentionV1CompletedReceipt
      completed_receipt{};
  runner_detail::SelectorExactPersistentAttentionV1RouteReceipt
      private_selector_receipt{};
  bool prompt_digest_ready = false;
  bool prefill_hook_completed = false;
  bool return_hook_completed = false;
  CaptureError error = CaptureError::kNone;
};

bool update_sha_from_device(const std::uint8_t* const source,
                            const std::size_t bytes,
                            std::vector<std::uint8_t>& scratch,
                            core::Sha256& hash,
                            int& cuda_error) noexcept {
  if (source == nullptr || bytes == 0U || scratch.empty()) {
    return false;
  }
  std::size_t offset = 0U;
  while (offset < bytes) {
    const std::size_t count = std::min(scratch.size(), bytes - offset);
    const cudaError_t status = cudaMemcpy(
        scratch.data(), source + offset, count, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      cuda_error = static_cast<int>(status);
      return false;
    }
    if (!hash.update(scratch.data(), count)) {
      return false;
    }
    offset += count;
  }
  return true;
}

bool hash_device(const std::uint8_t* const source, const std::size_t bytes,
                 std::vector<std::uint8_t>& scratch,
                 core::Sha256Digest& digest, int& cuda_error) noexcept {
  core::Sha256 hash;
  if (!update_sha_from_device(source, bytes, scratch, hash, cuda_error)) {
    return false;
  }
  digest = hash.finalize();
  return true;
}

bool region_source(const runtime::RequestState& state,
                   const runtime::RequestRegion& region,
                   const std::size_t relative_offset,
                   const std::size_t bytes,
                   const std::uint8_t*& source) noexcept {
  source = nullptr;
  if (state.arena_data() == nullptr || region.byte_size == 0U ||
      region.arena_offset > state.arena_bytes() ||
      region.byte_size > state.arena_bytes() - region.arena_offset ||
      relative_offset > region.byte_size ||
      bytes > region.byte_size - relative_offset ||
      region.arena_offset >
          static_cast<std::uint64_t>(std::numeric_limits<std::size_t>::max()) ||
      relative_offset > std::numeric_limits<std::size_t>::max() -
                            static_cast<std::size_t>(region.arena_offset)) {
    return false;
  }
  const std::size_t absolute =
      static_cast<std::size_t>(region.arena_offset) + relative_offset;
  if (bytes > static_cast<std::size_t>(state.arena_bytes()) - absolute) {
    return false;
  }
  source = static_cast<const std::uint8_t*>(state.arena_data()) + absolute;
  return true;
}

bool hash_region(const runtime::RequestState& state,
                 const runtime::RequestRegion& region,
                 const std::size_t relative_offset, const std::size_t bytes,
                 StateSnapshot& snapshot,
                 core::Sha256Digest& digest) noexcept {
  const std::uint8_t* source = nullptr;
  return region_source(state, region, relative_offset, bytes, source) &&
         hash_device(source, bytes, snapshot.scratch, digest,
                     snapshot.cuda_error);
}

bool collect_legacy_prompt_residual_chunk(
    const std::uint16_t* const residual_bf16,
    const std::uint32_t first_position, const std::uint32_t token_count,
    const std::size_t elements, void* const opaque) noexcept {
  auto* const capture = static_cast<LegacyPromptResidualCapture*>(opaque);
  if (capture == nullptr || capture->finalized ||
      capture->error != CaptureError::kNone ||
      capture->tile_count >= kLegacyTileCount ||
      first_position != capture->next_row || token_count == 0U ||
      token_count !=
          (capture->tile_count + 1U == kLegacyTileCount ? 64U : 512U) ||
      elements != static_cast<std::size_t>(token_count) * kHiddenSize ||
      first_position + token_count > kPromptTokens) {
    if (capture != nullptr) {
      capture->error = CaptureError::kLegacyResidual;
    }
    return false;
  }
  const std::size_t bytes = elements * kBf16Bytes;
  if (!update_sha_from_device(
          reinterpret_cast<const std::uint8_t*>(residual_bf16), bytes,
          capture->scratch, capture->hash, capture->cuda_error)) {
    capture->error = capture->cuda_error == static_cast<int>(cudaSuccess)
                         ? CaptureError::kHash
                         : CaptureError::kCuda;
    return false;
  }
  capture->first_positions[capture->tile_count] = first_position;
  capture->tile_rows[capture->tile_count] = token_count;
  ++capture->tile_count;
  capture->total_bytes += bytes;
  capture->next_row += token_count;
  return true;
}

bool finalize_legacy_prompt_residual(
    LegacyPromptResidualCapture& capture) noexcept {
  if (capture.finalized || capture.error != CaptureError::kNone ||
      capture.tile_count != kLegacyTileCount ||
      capture.next_row != kPromptTokens ||
      capture.total_bytes != kPromptResidualBytes) {
    capture.error = CaptureError::kLegacyResidual;
    return false;
  }
  capture.digest = capture.hash.finalize();
  capture.finalized = true;
  return true;
}

bool capture_prompt_digest(const runtime::RequestState& state,
                           StrictCaptureContext& context,
                           const bool prefill_stage,
                           StateSnapshot& snapshot) noexcept {
  if (!prefill_stage) {
    if (!context.prompt_digest_ready) {
      snapshot.error = CaptureError::kInvalidRegion;
      return false;
    }
    snapshot.prompt_residual = context.prompt_digest;
    snapshot.prompt_residual_live_capture = false;
    snapshot.prompt_residual_carried_from_prefill = true;
    snapshot.prompt_residual_source_sequence_length = kPromptTokens;
    return true;
  }
  if (context.selector_arm) {
    const runtime::LayerMajorRequestMemoryPlan* const layer_major =
        state.layer_major_plan();
    if (layer_major == nullptr ||
        layer_major->prompt_residual_bf16.storage.element_size_bytes !=
            kBf16Bytes ||
        layer_major->prompt_residual_bf16.storage.element_capacity !=
            static_cast<std::uint64_t>(kSelectorCapacity) * kHiddenSize ||
        layer_major->prompt_residual_bf16.storage.byte_size !=
            static_cast<std::uint64_t>(kSelectorCapacity) * kHiddenRowBytes ||
        layer_major->prompt_residual_bf16.columns != kHiddenSize ||
        layer_major->prompt_residual_bf16.row_stride_elements != kHiddenSize ||
        layer_major->prompt_residual_bf16.row_capacity != kSelectorCapacity) {
      snapshot.error = CaptureError::kInvalidRegion;
      return false;
    }
    core::Sha256Digest observed;
    if (!hash_region(state, layer_major->prompt_residual_bf16.storage, 0U,
                     kPromptResidualBytes, snapshot, observed)) {
      snapshot.error = snapshot.cuda_error == static_cast<int>(cudaSuccess)
                           ? CaptureError::kHash
                           : CaptureError::kCuda;
      return false;
    }
    if (context.prompt_digest_ready &&
        !(context.prompt_digest == observed)) {
      snapshot.error = CaptureError::kInvalidRegion;
      return false;
    }
    context.prompt_digest = observed;
    context.prompt_digest_ready = true;
    snapshot.prompt_residual_source_owner =
        "LayerMajorRequestMemoryPlan";
    snapshot.prompt_residual_source_buffer = "prompt_residual_bf16";
    snapshot.prompt_residual_source_region_bytes =
        layer_major->prompt_residual_bf16.storage.byte_size;
    snapshot.prompt_residual_source_region_element_capacity =
        layer_major->prompt_residual_bf16.storage.element_capacity;
  } else {
    if (!context.legacy_residual.finalized ||
        context.legacy_residual.total_bytes != kPromptResidualBytes) {
      snapshot.error = CaptureError::kLegacyResidual;
      return false;
    }
    const runtime::RequestRegion& source = state.plan().hidden_bf16[0U];
    if (source.element_size_bytes != kBf16Bytes ||
        source.element_capacity != 512U * kHiddenSize ||
        source.byte_size != 512U * kHiddenRowBytes) {
      snapshot.error = CaptureError::kInvalidRegion;
      return false;
    }
    context.prompt_digest = context.legacy_residual.digest;
    context.prompt_digest_ready = true;
    snapshot.prompt_residual_source_owner = "RequestMemoryPlan";
    snapshot.prompt_residual_source_buffer = "hidden_bf16[0]";
    snapshot.prompt_residual_source_region_bytes = source.byte_size;
    snapshot.prompt_residual_source_region_element_capacity =
        source.element_capacity;
  }
  snapshot.prompt_residual = context.prompt_digest;
  snapshot.prompt_residual_live_capture = true;
  snapshot.prompt_residual_carried_from_prefill = false;
  snapshot.prompt_residual_source_sequence_length = kPromptTokens;
  return true;
}

bool capture_final_hidden(const runtime::RequestState& state,
                          const StrictCaptureContext& context,
                          const bool prefill_stage,
                          StateSnapshot& snapshot) noexcept {
  const runtime::RequestRegion* region = nullptr;
  std::size_t byte_offset = 0U;
  if (prefill_stage && context.selector_arm) {
    const runtime::LayerMajorRequestMemoryPlan* const layer_major =
        state.layer_major_plan();
    if (layer_major == nullptr ||
        layer_major->final_hidden_bf16.storage.element_size_bytes !=
            kBf16Bytes ||
        layer_major->final_hidden_bf16.storage.element_capacity !=
            kHiddenSize ||
        layer_major->final_hidden_bf16.storage.byte_size != kHiddenRowBytes ||
        layer_major->final_hidden_bf16.row_capacity != 1U ||
        layer_major->final_hidden_bf16.columns != kHiddenSize ||
        layer_major->final_hidden_bf16.row_stride_elements != kHiddenSize) {
      snapshot.error = CaptureError::kInvalidRegion;
      return false;
    }
    region = &layer_major->final_hidden_bf16.storage;
    snapshot.final_hidden_source_owner = "LayerMajorRequestMemoryPlan";
    snapshot.final_hidden_source_buffer = "final_hidden_bf16";
    snapshot.final_hidden_source_row_capacity =
        layer_major->final_hidden_bf16.row_capacity;
    snapshot.final_hidden_source_row_stride_elements =
        layer_major->final_hidden_bf16.row_stride_elements;
  } else {
    const runtime::RequestMemoryPlan& common = state.plan();
    region = &common.hidden_bf16[1U];
    if (region->element_size_bytes != kBf16Bytes ||
        region->element_capacity != 512U * kHiddenSize ||
        region->byte_size != 512U * kHiddenRowBytes) {
      snapshot.error = CaptureError::kInvalidRegion;
      return false;
    }
    byte_offset = prefill_stage ? 63U * kHiddenRowBytes : 0U;
    snapshot.final_hidden_source_owner = "RequestMemoryPlan";
    snapshot.final_hidden_source_buffer = "hidden_bf16[1]";
    snapshot.final_hidden_source_row_capacity = 512U;
    snapshot.final_hidden_source_row_stride_elements = kHiddenSize;
  }
  snapshot.final_hidden_source_region_bytes = region->byte_size;
  snapshot.final_hidden_source_region_element_capacity =
      region->element_capacity;
  snapshot.final_hidden_source_byte_offset = byte_offset;
  snapshot.final_hidden_position = snapshot.expected_sequence_length - 1U;
  if (region == nullptr || region->element_size_bytes != kBf16Bytes ||
      !hash_region(state, *region, byte_offset, kHiddenRowBytes, snapshot,
                   snapshot.final_hidden)) {
    snapshot.error = snapshot.cuda_error == static_cast<int>(cudaSuccess)
                         ? CaptureError::kInvalidRegion
                         : CaptureError::kCuda;
    return false;
  }
  return true;
}

bool capture_state(const runtime::RequestState& state,
                   StrictCaptureContext& context, const bool prefill_stage,
                   StateSnapshot& snapshot) noexcept {
  ++snapshot.hook_calls;
  if (snapshot.hook_calls != 1U) {
    snapshot.error = CaptureError::kDuplicateHook;
    return false;
  }
  if (!state || state.arena_data() == nullptr ||
      state.sequence_length() != snapshot.expected_sequence_length ||
      snapshot.expected_sequence_length > state.max_sequence_length() ||
      state.memory_profile() !=
          (context.selector_arm
               ? runtime::RequestMemoryProfile::kLayerMajorP40WholeCore
               : runtime::RequestMemoryProfile::kLegacyC512)) {
    snapshot.error = CaptureError::kInvalidIdentity;
    return false;
  }
  const cudaError_t synchronized = cudaDeviceSynchronize();
  if (synchronized != cudaSuccess) {
    snapshot.cuda_error = static_cast<int>(synchronized);
    snapshot.error = CaptureError::kCuda;
    return false;
  }
  snapshot.sequence_length = state.sequence_length();

  const runtime::RequestMemoryPlan& common = state.plan();
  if (common.conv_state.element_size_bytes != kBf16Bytes ||
      common.conv_state.element_capacity !=
          runtime::kRequestConvStateBytes / kBf16Bytes ||
      common.conv_state.byte_size != runtime::kRequestConvStateBytes ||
      common.gdn_state.element_size_bytes != kBf16Bytes ||
      common.gdn_state.element_capacity !=
          runtime::kRequestGdnStateBytes / kBf16Bytes ||
      common.gdn_state.byte_size != runtime::kRequestGdnStateBytes ||
      !hash_region(state, common.conv_state, 0U,
                   static_cast<std::size_t>(common.conv_state.byte_size),
                   snapshot, snapshot.conv_state) ||
      !hash_region(state, common.gdn_state, 0U,
                   static_cast<std::size_t>(common.gdn_state.byte_size),
                   snapshot, snapshot.gdn_state)) {
    snapshot.error = snapshot.cuda_error == static_cast<int>(cudaSuccess)
                         ? CaptureError::kInvalidRegion
                         : CaptureError::kCuda;
    return false;
  }

  if (snapshot.expected_sequence_length >
      std::numeric_limits<std::size_t>::max() /
          kKvBytesPerLayerPosition) {
    snapshot.error = CaptureError::kInvalidSequence;
    return false;
  }
  snapshot.used_kv_bytes_per_layer =
      static_cast<std::size_t>(snapshot.expected_sequence_length) *
      kKvBytesPerLayerPosition;
  const std::uint64_t expected_kv_region_bytes =
      static_cast<std::uint64_t>(state.max_sequence_length()) *
      kKvBytesPerLayerPosition;
  const std::uint64_t expected_kv_region_elements =
      static_cast<std::uint64_t>(state.max_sequence_length()) * 4U * 256U;
  snapshot.kv_storage_bytes_per_layer = expected_kv_region_bytes;
  snapshot.kv_storage_element_capacity_per_layer =
      expected_kv_region_elements;
  for (std::size_t slot = 0U; slot < kFullAttentionLayerCount; ++slot) {
    const runtime::RequestRegion& key = common.key_cache[slot];
    const runtime::RequestRegion& value = common.value_cache[slot];
    if (key.element_size_bytes != kBf16Bytes ||
        value.element_size_bytes != kBf16Bytes ||
        key.element_capacity != expected_kv_region_elements ||
        value.element_capacity != expected_kv_region_elements ||
        key.byte_size != expected_kv_region_bytes ||
        value.byte_size != expected_kv_region_bytes ||
        snapshot.used_kv_bytes_per_layer > key.byte_size ||
        !hash_region(state, key, 0U, snapshot.used_kv_bytes_per_layer,
                     snapshot, snapshot.key_cache[slot]) ||
        !hash_region(state, value, 0U, snapshot.used_kv_bytes_per_layer,
                     snapshot, snapshot.value_cache[slot])) {
      snapshot.error = snapshot.cuda_error == static_cast<int>(cudaSuccess)
                           ? CaptureError::kInvalidRegion
                           : CaptureError::kCuda;
      return false;
    }
  }

  if (!capture_prompt_digest(state, context, prefill_stage, snapshot) ||
      !capture_final_hidden(state, context, prefill_stage, snapshot)) {
    return false;
  }
  core::Sha256 aggregate;
  const auto add = [&aggregate](const core::Sha256Digest& digest) noexcept {
    return aggregate.update(digest.bytes.data(), digest.bytes.size());
  };
  bool complete = add(snapshot.conv_state) && add(snapshot.gdn_state);
  for (std::size_t slot = 0U; slot < kFullAttentionLayerCount; ++slot) {
    complete = complete && add(snapshot.key_cache[slot]) &&
               add(snapshot.value_cache[slot]);
  }
  if (prefill_stage) {
    complete = complete && add(snapshot.prompt_residual);
  }
  complete = complete && add(snapshot.final_hidden);
  if (!complete) {
    snapshot.error = CaptureError::kHash;
    return false;
  }
  snapshot.aggregate = aggregate.finalize();
  return true;
}

bool collect_prefill_commit(
    const runtime::RequestState& state,
    const engine_detail::
        P40000SelectorExactPersistentAttentionV1CompletedReceipt& receipt,
    void* const opaque) noexcept {
  auto* const context = static_cast<StrictCaptureContext*>(opaque);
  if (context == nullptr || context->prefill_hook_completed ||
      !receipt.prompt_state_committed ||
      receipt.required_steps != kRequiredSteps ||
      receipt.selector_route != context->selector_arm ||
      receipt.legacy_c512_route == context->selector_arm ||
      receipt.configured_internal_rows !=
          (context->selector_arm ? kSelectorCapacity : kLegacyCapacity) ||
      receipt.guard_rows != (context->selector_arm ? 1U : 0U) ||
      receipt.arena_bytes !=
          (context->selector_arm ? kSelectorArenaBytes : kLegacyArenaBytes)) {
    if (context != nullptr) {
      context->error = CaptureError::kInvalidIdentity;
    }
    return false;
  }
  if (!context->selector_arm &&
      !finalize_legacy_prompt_residual(context->legacy_residual)) {
    context->error = CaptureError::kLegacyResidual;
    return false;
  }
  context->completed_receipt = receipt;
  if (!capture_state(state, *context, true, context->prefill)) {
    context->error = context->prefill.error;
    return false;
  }
  context->prefill_hook_completed = true;
  return true;
}

void collect_generation_return(const runtime::RequestState& state,
                               void* const opaque) noexcept {
  auto* const context = static_cast<StrictCaptureContext*>(opaque);
  if (context == nullptr || context->return_hook_completed) {
    if (context != nullptr) {
      context->error = CaptureError::kDuplicateHook;
    }
    return;
  }
  if (!capture_state(state, *context, false, context->returned)) {
    context->error = context->returned.error;
    return;
  }
  context->return_hook_completed = true;
}

bool collect_token(void* const opaque,
                   const runtime::ReferenceTokenEvent& event) noexcept {
  auto* const collector = static_cast<TokenCollector*>(opaque);
  if (collector == nullptr || collector->failed ||
      event.index != collector->events.size() ||
      collector->events.size() >= kMaxNewTokens) {
    if (collector != nullptr) {
      collector->failed = true;
    }
    return false;
  }
  try {
    collector->events.push_back(
        {event.index, event.token_id, std::string(event.text_delta),
         std::string(event.generated_text), event.is_stop_token});
  } catch (...) {
    collector->failed = true;
    return false;
  }
  return true;
}

bool never_cancel(void*) noexcept { return false; }

class ScopedLegacyResidualHook final {
 public:
  explicit ScopedLegacyResidualHook(LegacyPromptResidualCapture* capture)
      noexcept
      : previous_(runner_detail::exchange_reference_legacy_prefill_residual_chunk_hook(
            capture == nullptr
                ? runner_detail::ReferenceLegacyPrefillResidualChunkHook{}
                : runner_detail::ReferenceLegacyPrefillResidualChunkHook{
                      collect_legacy_prompt_residual_chunk, capture})) {}
  ~ScopedLegacyResidualHook() {
    (void)runner_detail::exchange_reference_legacy_prefill_residual_chunk_hook(
        previous_);
  }
  ScopedLegacyResidualHook(const ScopedLegacyResidualHook&) = delete;
  ScopedLegacyResidualHook& operator=(const ScopedLegacyResidualHook&) = delete;

 private:
  runner_detail::ReferenceLegacyPrefillResidualChunkHook previous_{};
};

class ScopedPrefillHook final {
 public:
  explicit ScopedPrefillHook(StrictCaptureContext& context) noexcept
      : previous_(engine_detail::exchange_reference_engine_prefill_commit_snapshot_hook(
            {collect_prefill_commit, &context})) {}
  ~ScopedPrefillHook() {
    (void)engine_detail::exchange_reference_engine_prefill_commit_snapshot_hook(
        previous_);
  }
  ScopedPrefillHook(const ScopedPrefillHook&) = delete;
  ScopedPrefillHook& operator=(const ScopedPrefillHook&) = delete;

 private:
  engine_detail::ReferenceEnginePrefillCommitSnapshotHook previous_{};
};

class ScopedReturnHook final {
 public:
  explicit ScopedReturnHook(StrictCaptureContext& context) noexcept
      : previous_(runner_detail::exchange_reference_engine_generate_return_snapshot_hook(
            {collect_generation_return, &context})) {}
  ~ScopedReturnHook() {
    (void)runner_detail::exchange_reference_engine_generate_return_snapshot_hook(
        previous_);
  }
  ScopedReturnHook(const ScopedReturnHook&) = delete;
  ScopedReturnHook& operator=(const ScopedReturnHook&) = delete;

 private:
  runner_detail::ReferenceEngineGenerateReturnSnapshotHook previous_{};
};

class ScopedFinalTokenPolicy final {
 public:
  explicit ScopedFinalTokenPolicy(
      const engine_detail::ReferenceEnginePrefillFinalTokenPolicyForTest policy)
      noexcept
      : previous_(
            engine_detail::exchange_reference_engine_prefill_final_token_policy_for_test(
                policy)) {}
  ~ScopedFinalTokenPolicy() {
    (void)engine_detail::exchange_reference_engine_prefill_final_token_policy_for_test(
        previous_);
  }
  ScopedFinalTokenPolicy(const ScopedFinalTokenPolicy&) = delete;
  ScopedFinalTokenPolicy& operator=(const ScopedFinalTokenPolicy&) = delete;

 private:
  engine_detail::ReferenceEnginePrefillFinalTokenPolicyForTest previous_ =
      engine_detail::ReferenceEnginePrefillFinalTokenPolicyForTest::
          kProductionDefault;
};

bool hash_file_exact(const std::filesystem::path& path,
                     const std::uint64_t expected_bytes,
                     core::Sha256Digest& digest,
                     std::string& error) {
  std::error_code status_error;
  const std::filesystem::file_status status =
      std::filesystem::symlink_status(path, status_error);
  if (status_error || std::filesystem::is_symlink(status) ||
      !std::filesystem::is_regular_file(status)) {
    error = path.filename().string() + ":not-regular-or-symlink";
    return false;
  }
  if (std::filesystem::file_size(path, status_error) != expected_bytes ||
      status_error) {
    error = path.filename().string() + ":bytes";
    return false;
  }
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    error = path.filename().string() + ":open";
    return false;
  }
  std::vector<std::uint8_t> buffer(kCopyChunkBytes);
  core::Sha256 hash;
  std::uint64_t consumed = 0U;
  while (consumed < expected_bytes) {
    const std::size_t count = static_cast<std::size_t>(
        std::min<std::uint64_t>(buffer.size(), expected_bytes - consumed));
    input.read(reinterpret_cast<char*>(buffer.data()),
               static_cast<std::streamsize>(count));
    if (input.gcount() != static_cast<std::streamsize>(count) ||
        !hash.update(buffer.data(), count)) {
      error = path.filename().string() + ":read-or-hash";
      return false;
    }
    consumed += count;
  }
  digest = hash.finalize();
  return true;
}

bool attest_checkpoint(const std::filesystem::path& model_directory,
                       CheckpointAttestation& attestation,
                       std::string& error) {
  attestation = {};
  for (std::size_t index = 0U; index < kCheckpointFiles.size(); ++index) {
    const CheckpointFileContract& expected = kCheckpointFiles[index];
    if (!hash_file_exact(model_directory / expected.relative_path,
                         expected.bytes, attestation.digests[index], error) ||
        attestation.digests[index].hex() != expected.sha256) {
      if (error.empty()) {
        error = std::string(expected.relative_path) + ":sha256";
      }
      return false;
    }
  }
  attestation.complete = true;
  return true;
}

bool hash_tokens(const std::vector<std::uint32_t>& tokens,
                 core::Sha256Digest& digest) noexcept {
  core::Sha256 hash;
  for (const std::uint32_t token : tokens) {
    const std::array<std::uint8_t, 4U> bytes{{
        static_cast<std::uint8_t>(token),
        static_cast<std::uint8_t>(token >> 8U),
        static_cast<std::uint8_t>(token >> 16U),
        static_cast<std::uint8_t>(token >> 24U),
    }};
    if (!hash.update(bytes.data(), bytes.size())) {
      return false;
    }
  }
  digest = hash.finalize();
  return true;
}

bool load_corpus(const std::filesystem::path& path, Corpus& corpus,
                 std::string& error) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    error = "corpus-open";
    return false;
  }
  const std::streampos end = input.tellg();
  if (end <= 0 || static_cast<std::uintmax_t>(end) > 1U * 1024U * 1024U) {
    error = "corpus-size";
    return false;
  }
  std::string bytes(static_cast<std::size_t>(end), '\0');
  input.seekg(0, std::ios::beg);
  input.read(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  if (input.gcount() != static_cast<std::streamsize>(bytes.size())) {
    error = "corpus-read";
    return false;
  }
  core::Sha256 file_hash;
  if (!file_hash.update(bytes.data(), bytes.size())) {
    error = "corpus-hash";
    return false;
  }
  corpus.file_sha256 = file_hash.finalize();
  if (corpus.file_sha256.hex() != kCorpusFileSha256) {
    error = "corpus-file-sha256";
    return false;
  }
  q3x::io::json::ParseOptions options;
  options.max_input_bytes = 1U * 1024U * 1024U;
  options.max_values = 50'100U;
  options.max_container_items = 50'100U;
  const q3x::io::json::ParseResult parsed =
      q3x::io::json::parse(bytes, options);
  const q3x::io::json::Value* const prompt =
      parsed ? parsed.value->find("prompt") : nullptr;
  const q3x::io::json::Value::Array* const values =
      prompt == nullptr ? nullptr : prompt->as_array();
  if (!parsed || values == nullptr || values->size() != kPromptTokens) {
    error = "corpus-json-or-shape";
    return false;
  }
  corpus.tokens.clear();
  corpus.tokens.reserve(kPromptTokens);
  for (const q3x::io::json::Value& item : *values) {
    const q3x::io::json::Number* const number = item.as_number();
    std::uint64_t token = 0U;
    if (number == nullptr || !number->to_uint64(token) ||
        token >= runtime::kReferenceVocabularySize) {
      error = "corpus-token";
      return false;
    }
    corpus.tokens.push_back(static_cast<std::uint32_t>(token));
  }
  if (!hash_tokens(corpus.tokens, corpus.token_sha256) ||
      corpus.token_sha256.hex() != kCorpusTokenSha256) {
    error = "corpus-token-sha256";
    return false;
  }
  return true;
}

bool exact_operator_counts(const runtime::PrefillRouteEvidence& evidence,
                           const bool selector_arm) noexcept {
  constexpr std::array<std::uint64_t, runtime::kPrefillOperatorRoleCount>
      kLegacyProduction{{0U, 0U, 7'488U, 3'744U, 5'056U, 1'264U,
                         3'744U}};
  constexpr std::array<std::uint64_t, runtime::kPrefillOperatorRoleCount>
      kLegacyFallback{{5'056U, 5'056U, 96U, 48U, 0U, 0U, 48U}};
  constexpr std::array<std::uint64_t, runtime::kPrefillOperatorRoleCount>
      kSelectorProduction{{64U, 64U, 96U, 48U, 64U, 16U, 48U}};
  const std::uint64_t expected_passes = selector_arm ? 1U : 79U;
  if (!evidence.complete || !evidence.valid || evidence.request_active ||
      evidence.error != runtime::PrefillRouteEvidenceError::kNone ||
      evidence.completed_layer_passes != expected_passes ||
      evidence.expected_layer_passes != expected_passes) {
    return false;
  }
  for (const std::uint64_t count : evidence.forbidden_boundary_hits) {
    if (count != 0U) {
      return false;
    }
  }
  for (std::size_t role = 0U; role < runtime::kPrefillOperatorRoleCount;
       ++role) {
    const auto& count = evidence.operators[role];
    if (count.production_hits !=
            (selector_arm ? kSelectorProduction[role]
                          : kLegacyProduction[role]) ||
        count.exact_fallback_hits !=
            (selector_arm ? 0U : kLegacyFallback[role]) ||
        count.forbidden_hits != 0U) {
      return false;
    }
  }
  return true;
}

bool valid_selector_span(
    const engine_detail::NativePrefillPhysicalSubmissionReceipt& span,
    const std::size_t index) noexcept {
  if (index == 0U || index == 1U) {
    return span.tactic ==
               engine_detail::NativePrefillPhysicalSubmissionTactic::
                   kGroupQ64 &&
           span.first_position == index * 512U && span.token_count == 512U;
  }
  return index == 2U &&
         span.tactic == engine_detail::NativePrefillPhysicalSubmissionTactic::
                            kPersistentGenericQt2Q8 &&
         span.first_position == 1'024U && span.token_count == 38'976U;
}

bool valid_selector_span(
    const runner_detail::
        SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt& span,
    const std::size_t index) noexcept {
  if (index == 0U || index == 1U) {
    return span.tactic ==
               runner_detail::
                   SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
                       kGroupQ64 &&
           span.first_position == index * 512U && span.token_count == 512U;
  }
  return index == 2U &&
         span.tactic ==
             runner_detail::
                 SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
                     kPersistentGenericQt2Q8 &&
         span.first_position == 1'024U && span.token_count == 38'976U;
}

bool empty_native_span(
    const engine_detail::NativePrefillPhysicalSubmissionReceipt& span)
    noexcept {
  return span.tactic ==
             engine_detail::NativePrefillPhysicalSubmissionTactic::kNone &&
         span.first_position == 0U && span.token_count == 0U;
}

bool empty_private_span(
    const runner_detail::
        SelectorExactPersistentAttentionV1PhysicalSubmissionReceipt& span)
    noexcept {
  return span.tactic ==
             runner_detail::
                 SelectorExactPersistentAttentionV1PhysicalSubmissionTactic::
                     kNone &&
         span.first_position == 0U && span.token_count == 0U;
}

bool valid_selector_generation(
    const runtime::ReferenceGeneration& generation,
    const StrictCaptureContext& context) noexcept {
  const auto& receipt = context.completed_receipt;
  const auto& private_receipt = context.private_selector_receipt;
  if (generation.prefill_execution_mode !=
          runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor ||
      generation.prefill_deployment_plan_id !=
          runtime::kSelectorExactPersistentAttentionV1P40DeploymentPlanId ||
      generation.requested_prefill_chunk_size != 512U ||
      generation.effective_prefill_chunk_size != 512U ||
      !generation.all_prompt_tokens_prefilled_by_tiles ||
      generation.single_arbitrary_prefill_tiles ||
      generation.prefill_logical_panel_count != 5U ||
      generation.prefill_route_layer_pass_count != 1U ||
      !generation.prefill_bounded_submission_window ||
      generation.prefill_submission_window_retirements != 768U ||
      generation.prefill_operator_panel_executor_hits != 0U ||
      generation.prefill_native_group_q64_panel_hits != 0U ||
      generation.prefill_native_group_q128_v4_panel_hits != 0U ||
      generation.prefill_native_flashinfer_exact_panel_hits != 0U ||
      generation.prefill_native_flashinfer_exact_whole_prompt_hits != 0U ||
      generation.prefill_generic_qt2_hits != 0U ||
      generation.prefill_mlp_schedule_tactic !=
          runtime::LayerMajorPrefillMlpScheduleTactic::
              kPromptWideP40WholeCore ||
      generation.prefill_layer_wide_p40_mlp_layer_hits != 64U ||
      generation.prefill_persistent_p40_nvfp4_gate_up_hits != 64U ||
      generation.prefill_persistent_p40_nvfp4_down_residual_hits != 64U ||
      generation.prefill_persistent_p40_nvfp4_physical_launches != 128U ||
      generation.prefill_prompt_wide_p40_whole_core_layer_hits != 64U ||
      generation.prefill_prompt_wide_p40_fill_panel_hits != 320U ||
      generation.prefill_prompt_wide_p40_prompt_core_hits != 64U ||
      generation.prefill_prompt_wide_p40_drain_panel_hits != 320U ||
      generation.prefill_prompt_wide_p40_fp8_projection_hits != 1'040U ||
      generation.prefill_prompt_wide_p40_fp8_projection_physical_launches !=
          1'040U ||
      generation.prefill_prompt_wide_p40_bf16_ab_hits != 48U ||
      generation.prefill_prompt_wide_p40_gdn_hits != 48U ||
      generation.prefill_selector_exact_persistent_attention_v1_plan_id !=
          runtime::kSelectorExactPersistentAttentionV1P40DeploymentPlanId ||
      generation
              .prefill_selector_exact_persistent_attention_v1_configured_internal_rows !=
          kSelectorCapacity ||
      generation
              .prefill_selector_exact_persistent_attention_v1_required_steps !=
          kRequiredSteps ||
      generation.prefill_selector_exact_persistent_attention_v1_guard_rows !=
          1U ||
      generation.prefill_selector_exact_persistent_attention_v1_arena_bytes !=
          kSelectorArenaBytes ||
      generation
              .prefill_selector_exact_persistent_attention_v1_full_attention_layer_hits !=
          16U ||
      generation.prefill_selector_exact_persistent_attention_v1_panel_calls !=
          16U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_arithmetic_spans !=
          1'280U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_group_q64_submissions !=
          32U ||
      generation.prefill_selector_exact_persistent_attention_v1_generic_qt2_spans !=
          1'248U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_generic_q8_suffix_submissions !=
          16U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_fallback_submissions !=
          0U ||
      generation.prefill_selector_exact_persistent_attention_v1_persistent_ctas !=
          256U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_physical_submissions !=
          48U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_minimum_physical_tokens !=
          512U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_maximum_physical_tokens !=
          38'976U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_logical_prompt_tokens !=
          kPromptTokens ||
      !generation.prefill_selector_exact_persistent_attention_v1_completed ||
      generation
              .prefill_selector_exact_persistent_attention_v1_completed_layer_count !=
          16U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_physical_submission_count_per_layer !=
          3U ||
      !receipt.prompt_state_committed || !receipt.selector_route ||
      receipt.legacy_c512_route || !receipt.completed_physical_receipt ||
      receipt.configured_internal_rows != kSelectorCapacity ||
      receipt.required_steps != kRequiredSteps || receipt.guard_rows != 1U ||
      receipt.arena_bytes != kSelectorArenaBytes ||
      receipt.logical_panel_count != 1U ||
      receipt.completed_physical_submissions_per_full_attention_layer != 3U ||
      receipt.completed_physical_submissions_total != 48U ||
      receipt.full_attention_layer_hits != 16U ||
      receipt.panel_calls != 16U || receipt.arithmetic_spans != 1'280U ||
      receipt.group_q64_submissions != 32U ||
      receipt.generic_qt2_spans != 1'248U ||
      receipt.generic_q8_suffix_submissions != 16U ||
      receipt.fallback_submissions != 0U ||
      receipt.persistent_ctas != 256U ||
      receipt.physical_submissions != 48U ||
      receipt.minimum_physical_tokens != 512U ||
      receipt.maximum_physical_tokens != 38'976U ||
      receipt.logical_prompt_tokens != kPromptTokens ||
      receipt.completed_physical_submission_count_per_layer != 3U ||
      receipt.completed_layer_count != 16U ||
      private_receipt.panel_calls != 16U ||
      private_receipt.arithmetic_spans != 1'280U ||
      private_receipt.group_q64_submissions != 32U ||
      private_receipt.generic_qt2_spans != 1'248U ||
      private_receipt.generic_q8_suffix_submissions != 16U ||
      private_receipt.fallback_submissions != 0U ||
      private_receipt.persistent_ctas != 256U ||
      private_receipt.minimum_physical_submission_tokens != 512U ||
      private_receipt.maximum_physical_submission_tokens != 38'976U ||
      private_receipt.maximum_logical_panel_tokens != kPromptTokens ||
      !private_receipt.completed_physical_receipt ||
      private_receipt.physical_submission_count_per_panel != 3U ||
      private_receipt.issued_layer_count != 16U ||
      private_receipt.completed_layer_count != 16U ||
      receipt.bound_attention_role.role !=
          runtime::PrefillBindingRole::kExactCausalAttention ||
      receipt.bound_attention_role.tactic !=
          engine_detail::NativePrefillTactic::
              kSelectorExactPersistentAttentionV1WholePrompt ||
      receipt.bound_attention_role.completion !=
          engine_detail::NativePrefillCompletionDomain::kMainStreamBarrier ||
      receipt.bound_attention_role.maximum_logical_panel_m != kPromptTokens ||
      receipt.bound_attention_role.minimum_physical_m != 512U ||
      receipt.bound_attention_role.maximum_physical_m != 38'976U ||
      receipt.bound_attention_role.physical_submission_count_per_logical_panel !=
          3U) {
    return false;
  }
  for (std::size_t span = 0U; span < 3U; ++span) {
    if (!valid_selector_span(receipt.bound_attention_role.physical_submissions[span],
                             span) ||
        !valid_selector_span(receipt.completed_physical_submissions[span],
                             span) ||
        !valid_selector_span(private_receipt.physical_submissions[span], span) ||
        generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_tactics
                    [span] !=
            static_cast<std::uint8_t>(
                receipt.completed_physical_submissions[span].tactic) ||
        generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_first_positions
                    [span] !=
            receipt.completed_physical_submissions[span].first_position ||
        generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_token_counts
                    [span] !=
            receipt.completed_physical_submissions[span].token_count ||
        static_cast<std::uint8_t>(private_receipt.physical_submissions[span].tactic) !=
            generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_tactics
                    [span] ||
        private_receipt.physical_submissions[span].first_position !=
            generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_first_positions
                    [span] ||
        private_receipt.physical_submissions[span].token_count !=
            generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_token_counts
                    [span]) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < 16U; ++index) {
    const auto& completed = receipt.completed_layers[index];
    const auto& private_completed = private_receipt.completed_layers[index];
    const auto& public_completed = generation
        .prefill_selector_exact_persistent_attention_v1_completed_layers[index];
    if (completed.layer != 3U + 4U * index ||
        completed.physical_submission_count != 3U ||
        private_completed.layer != completed.layer ||
        private_completed.physical_submission_count != 3U ||
        public_completed.layer != completed.layer ||
        public_completed.physical_submission_count != 3U) {
      return false;
    }
    for (std::size_t span = 0U; span < 3U; ++span) {
      if (!valid_selector_span(completed.physical_submissions[span], span) ||
          !valid_selector_span(private_completed.physical_submissions[span],
                               span) ||
          static_cast<std::uint8_t>(
              private_completed.physical_submissions[span].tactic) !=
              static_cast<std::uint8_t>(
                  completed.physical_submissions[span].tactic) ||
          private_completed.physical_submissions[span].first_position !=
              completed.physical_submissions[span].first_position ||
          private_completed.physical_submissions[span].token_count !=
              completed.physical_submissions[span].token_count ||
          public_completed.physical_submission_tactics[span] !=
              static_cast<std::uint8_t>(
                  completed.physical_submissions[span].tactic) ||
          public_completed.physical_submission_first_positions[span] !=
              completed.physical_submissions[span].first_position ||
          public_completed.physical_submission_token_counts[span] !=
              completed.physical_submissions[span].token_count) {
        return false;
      }
    }
  }
  constexpr std::size_t kAttentionRole = static_cast<std::size_t>(
      runtime::PrefillOperatorRole::kAttention);
  return exact_operator_counts(generation.prefill_route_evidence, true) &&
         receipt.completed_physical_submissions_total ==
             generation.prefill_route_evidence.operators[kAttentionRole]
                     .production_hits *
                 receipt.bound_attention_role
                     .physical_submission_count_per_logical_panel;
}

bool empty_selector_generation_receipt(
    const runtime::ReferenceGeneration& generation,
    const StrictCaptureContext& context) noexcept {
  const auto& receipt = context.completed_receipt;
  const auto& private_receipt = context.private_selector_receipt;
  if (!generation
           .prefill_selector_exact_persistent_attention_v1_plan_id.empty() ||
      generation
              .prefill_selector_exact_persistent_attention_v1_configured_internal_rows !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_required_steps !=
          0U ||
      generation.prefill_selector_exact_persistent_attention_v1_guard_rows !=
          0U ||
      generation.prefill_selector_exact_persistent_attention_v1_arena_bytes !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_full_attention_layer_hits !=
          0U ||
      generation.prefill_selector_exact_persistent_attention_v1_panel_calls !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_arithmetic_spans !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_group_q64_submissions !=
          0U ||
      generation.prefill_selector_exact_persistent_attention_v1_generic_qt2_spans !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_generic_q8_suffix_submissions !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_fallback_submissions !=
          0U ||
      generation.prefill_selector_exact_persistent_attention_v1_persistent_ctas !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_physical_submissions !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_minimum_physical_tokens !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_maximum_physical_tokens !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_logical_prompt_tokens !=
          0U ||
      generation.prefill_selector_exact_persistent_attention_v1_completed ||
      generation
              .prefill_selector_exact_persistent_attention_v1_completed_layer_count !=
          0U ||
      generation
              .prefill_selector_exact_persistent_attention_v1_physical_submission_count_per_layer !=
          0U ||
      !receipt.prompt_state_committed || !receipt.legacy_c512_route ||
      receipt.selector_route ||
      receipt.configured_internal_rows != kLegacyCapacity ||
      receipt.required_steps != kRequiredSteps || receipt.guard_rows != 0U ||
      receipt.arena_bytes != kLegacyArenaBytes ||
      receipt.logical_panel_count != 79U ||
      receipt.completed_physical_submissions_per_full_attention_layer != 79U ||
      receipt.completed_physical_submissions_total != 1'264U ||
      receipt.bound_attention_role.role !=
          runtime::PrefillBindingRole::kExactCausalAttention ||
      receipt.bound_attention_role.tactic !=
          engine_detail::NativePrefillTactic::
              kExactCausalAttentionOracleSpanC512C16Reference256 ||
      receipt.bound_attention_role.completion !=
          engine_detail::NativePrefillCompletionDomain::kMainStreamBarrier ||
      receipt.bound_attention_role.maximum_logical_panel_m != 512U ||
      receipt.bound_attention_role.minimum_physical_m != 64U ||
      receipt.bound_attention_role.maximum_physical_m != 512U ||
      receipt.bound_attention_role.physical_submission_count_per_logical_panel !=
          1U ||
      receipt.full_attention_layer_hits != 0U || receipt.panel_calls != 0U ||
      receipt.arithmetic_spans != 0U ||
      receipt.group_q64_submissions != 0U ||
      receipt.generic_qt2_spans != 0U ||
      receipt.generic_q8_suffix_submissions != 0U ||
      receipt.fallback_submissions != 0U || receipt.persistent_ctas != 0U ||
      receipt.physical_submissions != 0U ||
      receipt.minimum_physical_tokens != 0U ||
      receipt.maximum_physical_tokens != 0U ||
      receipt.logical_prompt_tokens != 0U ||
      receipt.completed_physical_receipt ||
      receipt.completed_physical_submission_count_per_layer != 0U ||
      receipt.completed_layer_count != 0U ||
      private_receipt.panel_calls != 0U ||
      private_receipt.arithmetic_spans != 0U ||
      private_receipt.group_q64_submissions != 0U ||
      private_receipt.generic_qt2_spans != 0U ||
      private_receipt.generic_q8_suffix_submissions != 0U ||
      private_receipt.fallback_submissions != 0U ||
      private_receipt.persistent_ctas != 0U ||
      private_receipt.minimum_physical_submission_tokens != 0U ||
      private_receipt.maximum_physical_submission_tokens != 0U ||
      private_receipt.maximum_logical_panel_tokens != 0U ||
      private_receipt.completed_physical_receipt ||
      private_receipt.physical_submission_count_per_panel != 0U ||
      private_receipt.issued_layer_count != 0U ||
      private_receipt.completed_layer_count != 0U) {
    return false;
  }
  for (std::size_t span = 0U; span < 3U; ++span) {
    if (generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_tactics
                    [span] != 0U ||
        generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_first_positions
                    [span] != 0U ||
        generation
                .prefill_selector_exact_persistent_attention_v1_physical_submission_token_counts
                    [span] != 0U ||
        !empty_native_span(
            receipt.bound_attention_role.physical_submissions[span]) ||
        !empty_native_span(receipt.completed_physical_submissions[span]) ||
        !empty_private_span(private_receipt.physical_submissions[span])) {
      return false;
    }
  }
  for (std::size_t index = 0U; index < 16U; ++index) {
    const auto& public_layer = generation
        .prefill_selector_exact_persistent_attention_v1_completed_layers[index];
    const auto& completed_layer = receipt.completed_layers[index];
    const auto& private_layer = private_receipt.completed_layers[index];
    if (public_layer.layer != 0U ||
        public_layer.physical_submission_count != 0U ||
        completed_layer.layer != 0U ||
        completed_layer.physical_submission_count != 0U ||
        private_layer.layer != 0U ||
        private_layer.physical_submission_count != 0U) {
      return false;
    }
    for (std::size_t span = 0U; span < 3U; ++span) {
      if (public_layer.physical_submission_tactics[span] != 0U ||
          public_layer.physical_submission_first_positions[span] != 0U ||
          public_layer.physical_submission_token_counts[span] != 0U ||
          !empty_native_span(completed_layer.physical_submissions[span]) ||
          !empty_private_span(private_layer.physical_submissions[span])) {
        return false;
      }
    }
  }
  return true;
}

bool valid_legacy_generation(const runtime::ReferenceGeneration& generation,
                             const StrictCaptureContext& context) noexcept {
  if (generation.prefill_execution_mode !=
          runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled ||
      !generation.prefill_deployment_plan_id.empty() ||
      generation.requested_prefill_chunk_size != 512U ||
      generation.effective_prefill_chunk_size != 512U ||
      !generation.all_prompt_tokens_prefilled_by_tiles ||
      generation.single_arbitrary_prefill_tiles ||
      generation.timing.prefix_execution_milliseconds.size() !=
          kLegacyTileCount ||
      generation.prefill_logical_panel_count != kLegacyTileCount ||
      generation.prefill_route_layer_pass_count != kLegacyTileCount ||
      generation.prefill_bounded_submission_window ||
      generation.prefill_submission_window_retirements != 0U ||
      generation.prefill_mlp_schedule_tactic !=
          runtime::LayerMajorPrefillMlpScheduleTactic::kPerOperatorPanel ||
      generation.prefill_prompt_wide_p40_whole_core_layer_hits != 0U ||
      generation.prefill_prompt_wide_p40_fill_panel_hits != 0U ||
      generation.prefill_prompt_wide_p40_prompt_core_hits != 0U ||
      generation.prefill_prompt_wide_p40_drain_panel_hits != 0U ||
      generation.prefill_prompt_wide_p40_fp8_projection_hits != 0U ||
      generation.prefill_prompt_wide_p40_fp8_projection_physical_launches !=
          0U ||
      generation.prefill_prompt_wide_p40_bf16_ab_hits != 0U ||
      generation.prefill_prompt_wide_p40_gdn_hits != 0U ||
      generation.prefill_native_flashinfer_exact_panel_hits != 0U ||
      generation.prefill_native_flashinfer_exact_whole_prompt_hits != 0U ||
      generation.prefill_generic_qt2_hits != 0U ||
      generation.prefill_layer_wide_p40_mlp_layer_hits != 0U ||
      generation.prefill_persistent_p40_nvfp4_gate_up_hits != 0U ||
      generation.prefill_persistent_p40_nvfp4_down_residual_hits != 0U ||
      generation.prefill_persistent_p40_nvfp4_physical_launches != 0U ||
      !empty_selector_generation_receipt(generation, context) ||
      !exact_operator_counts(generation.prefill_route_evidence, false)) {
    return false;
  }
  if (!context.legacy_residual.finalized ||
      context.legacy_residual.tile_count != kLegacyTileCount ||
      context.legacy_residual.total_bytes != kPromptResidualBytes) {
    return false;
  }
  constexpr std::size_t kAttentionRole = static_cast<std::size_t>(
      runtime::PrefillOperatorRole::kAttention);
  if (context.completed_receipt.completed_physical_submissions_total !=
      generation.prefill_route_evidence.operators[kAttentionRole]
              .production_hits *
          context.completed_receipt.bound_attention_role
              .physical_submission_count_per_logical_panel) {
    return false;
  }
  for (std::size_t index = 0U; index < kLegacyTileCount; ++index) {
    if (context.legacy_residual.first_positions[index] != index * 512U ||
        context.legacy_residual.tile_rows[index] !=
            (index + 1U == kLegacyTileCount ? 64U : 512U)) {
      return false;
    }
  }
  return true;
}

bool valid_state_snapshot(const StateSnapshot& snapshot,
                          const std::uint32_t expected,
                          const bool selector) noexcept {
  const bool prefill = expected == kPromptTokens;
  if (!prefill && expected != kRequiredSteps) {
    return false;
  }
  const std::uint64_t expected_kv_storage_bytes =
      static_cast<std::uint64_t>(selector ? kSelectorCapacity
                                          : kLegacyCapacity) *
      kKvBytesPerLayerPosition;
  const std::uint64_t expected_kv_storage_elements =
      static_cast<std::uint64_t>(selector ? kSelectorCapacity
                                          : kLegacyCapacity) *
      4U * 256U;
  const std::uint64_t expected_final_region_bytes =
      prefill && selector ? kHiddenRowBytes : 512U * kHiddenRowBytes;
  const std::uint64_t expected_final_region_elements =
      prefill && selector ? kHiddenSize : 512U * kHiddenSize;
  const std::string_view expected_final_owner =
      prefill && selector ? "LayerMajorRequestMemoryPlan"
                          : "RequestMemoryPlan";
  const std::string_view expected_final_buffer =
      prefill && selector ? "final_hidden_bf16" : "hidden_bf16[1]";
  const std::size_t expected_final_offset =
      prefill && !selector ? 63U * kHiddenRowBytes : 0U;
  if (snapshot.hook_calls != 1U || snapshot.sequence_length != expected ||
      snapshot.expected_sequence_length != expected ||
      snapshot.used_kv_bytes_per_layer !=
          static_cast<std::size_t>(expected) * kKvBytesPerLayerPosition ||
      snapshot.kv_storage_bytes_per_layer != expected_kv_storage_bytes ||
      snapshot.kv_storage_element_capacity_per_layer !=
          expected_kv_storage_elements ||
      snapshot.prompt_residual_source_sequence_length != kPromptTokens ||
      snapshot.prompt_residual_live_capture != prefill ||
      snapshot.prompt_residual_carried_from_prefill != !prefill ||
      snapshot.final_hidden_position != expected - 1U ||
      snapshot.final_hidden_source_owner != expected_final_owner ||
      snapshot.final_hidden_source_buffer != expected_final_buffer ||
      snapshot.final_hidden_source_byte_offset != expected_final_offset ||
      snapshot.final_hidden_source_region_bytes !=
          expected_final_region_bytes ||
      snapshot.final_hidden_source_region_element_capacity !=
          expected_final_region_elements ||
      snapshot.final_hidden_source_row_capacity !=
          (prefill && selector ? 1U : 512U) ||
      snapshot.final_hidden_source_row_stride_elements != kHiddenSize ||
      snapshot.error != CaptureError::kNone ||
      snapshot.cuda_error != static_cast<int>(cudaSuccess)) {
    return false;
  }
  if (!prefill) {
    return snapshot.prompt_residual_source_owner.empty() &&
           snapshot.prompt_residual_source_buffer.empty() &&
           snapshot.prompt_residual_source_region_bytes == 0U &&
           snapshot.prompt_residual_source_region_element_capacity == 0U;
  }
  return snapshot.prompt_residual_source_owner ==
             (selector ? "LayerMajorRequestMemoryPlan"
                       : "RequestMemoryPlan") &&
         snapshot.prompt_residual_source_buffer ==
             (selector ? "prompt_residual_bf16" : "hidden_bf16[0]") &&
         snapshot.prompt_residual_source_region_bytes ==
             (selector ? static_cast<std::uint64_t>(kSelectorCapacity) *
                             kHiddenRowBytes
                       : 512U * kHiddenRowBytes) &&
         snapshot.prompt_residual_source_region_element_capacity ==
             (selector ? static_cast<std::uint64_t>(kSelectorCapacity) *
                             kHiddenSize
                       : 512U * kHiddenSize);
}

bool validate_generation_behavior(
    const runtime::ReferenceGeneration& generation, const Corpus& corpus,
    const TokenCollector& tokens,
    std::vector<const runtime::ReferenceStepResult*>& logit_steps,
    std::string& error) {
  if (generation.prompt_token_ids != corpus.tokens ||
      generation.generated_token_ids.size() != kMaxNewTokens ||
      generation.stop_reason != runtime::ReferenceStopReason::kMaxNewTokens ||
      generation.steps.size() != kRequiredSteps ||
      generation.timing.subsequent_token_milliseconds.size() !=
          kMaxNewTokens - 1U ||
      tokens.failed || tokens.events.size() != kMaxNewTokens) {
    error = "generation-shape";
    return false;
  }
  logit_steps.clear();
  logit_steps.reserve(kMaxNewTokens);
  for (std::size_t index = 0U; index < generation.steps.size(); ++index) {
    const runtime::ReferenceStepResult& step = generation.steps[index];
    const std::uint32_t expected_input =
        index < kPromptTokens
            ? corpus.tokens[index]
            : generation.generated_token_ids[index - kPromptTokens];
    if (step.position != index || step.input_token_id != expected_input) {
      error = "generation-step-position-input";
      return false;
    }
    if (index + 1U < kPromptTokens) {
      if (step.logits.has_value() || step.prediction.has_value()) {
        error = "generation-prefix-logits";
        return false;
      }
      continue;
    }
    const std::size_t generated_index = index + 1U - kPromptTokens;
    if (!step.logits.has_value() || step.prediction.has_value() ||
        generated_index >= generation.generated_token_ids.size() ||
        step.logits->predicted_token_id !=
            generation.generated_token_ids[generated_index] ||
        !std::isfinite(step.logits->chosen_logit) ||
        !std::isfinite(step.logits->logsumexp) ||
        !std::isfinite(step.logits->max_log_probability)) {
      error = "generation-full-statistics";
      return false;
    }
    logit_steps.push_back(&step);
  }
  if (logit_steps.size() != kMaxNewTokens) {
    error = "generation-full-statistics-count";
    return false;
  }
  std::string cumulative;
  for (std::size_t index = 0U; index < tokens.events.size(); ++index) {
    const TokenEvent& event = tokens.events[index];
    cumulative += event.text_delta;
    if (event.index != index ||
        event.token_id != generation.generated_token_ids[index] ||
        event.is_stop_token || event.generated_text != cumulative) {
      error = "generation-token-event";
      return false;
    }
  }
  if (tokens.events.back().generated_text != generation.generated_text) {
    error = "generation-final-text";
    return false;
  }
  return true;
}

void write_json_string(std::ostream& output, const std::string_view value) {
  constexpr char kHex[] = "0123456789abcdef";
  output.put('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        output << "\\\"";
        break;
      case '\\':
        output << "\\\\";
        break;
      case '\b':
        output << "\\b";
        break;
      case '\f':
        output << "\\f";
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
          output << "\\u00" << kHex[(byte >> 4U) & 0xFU]
                 << kHex[byte & 0xFU];
        } else {
          output.put(static_cast<char>(byte));
        }
        break;
    }
  }
  output.put('"');
}

template <typename Integer>
std::string fixed_hex(const Integer value) {
  std::ostringstream output;
  output << "0x" << std::hex << std::nouppercase << std::setfill('0')
         << std::setw(static_cast<int>(sizeof(Integer) * 2U)) << value;
  return output.str();
}

std::string f32_bits(const float value) {
  std::uint32_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return fixed_hex(bits);
}

std::string f64_bits(const double value) {
  std::uint64_t bits = 0U;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return fixed_hex(bits);
}

void write_component(std::ostream& output, const std::string_view domain,
                     const std::initializer_list<std::uint64_t> shape,
                     const std::size_t bytes,
                     const core::Sha256Digest& digest) {
  output << "{\"domain\":";
  write_json_string(output, domain);
  output << ",\"shape\":[";
  std::size_t index = 0U;
  for (const std::uint64_t dimension : shape) {
    if (index++ != 0U) {
      output.put(',');
    }
    output << dimension;
  }
  output << "],\"bytes\":" << bytes << ",\"sha256\":";
  write_json_string(output, digest.hex());
  output.put('}');
}

void write_checkpoint(std::ostream& output,
                      const CheckpointAttestation& attestation) {
  output << "{\"repository\":";
  write_json_string(output, kCheckpointRepository);
  output << ",\"revision\":";
  write_json_string(output, kCheckpointRevision);
  output << ",\"exact_attestation\":"
         << (attestation.complete ? "true" : "false")
         << ",\"files\":[";
  for (std::size_t index = 0U; index < kCheckpointFiles.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    output << "{\"path\":";
    write_json_string(output, kCheckpointFiles[index].relative_path);
    output << ",\"bytes\":" << kCheckpointFiles[index].bytes
           << ",\"sha256\":";
    write_json_string(output, attestation.digests[index].hex());
    output.put('}');
  }
  output << "]}";
}

void write_prompt_residual(std::ostream& output,
                           const StrictCaptureContext& context,
                           const StateSnapshot& snapshot) {
  output << "{\"domain\":\"q3x.prompt-residual-live-row-major-bf16.v1\""
            ",\"shape\":[40000,5120],\"bytes\":409600000,\"sha256\":";
  write_json_string(output, snapshot.prompt_residual.hex());
  if (snapshot.prompt_residual_carried_from_prefill) {
    output << ",\"capture_available\":false"
              ",\"carried_from_prefill\":true"
              ",\"capture\":\"prefill-commit-attested-reference\""
              ",\"source_sequence_length\":"
           << snapshot.prompt_residual_source_sequence_length << '}';
    return;
  }
  output << ",\"capture_available\":true,\"hash_update_bytes\":409600000"
            ",\"live_rows\":40000,\"capture\":";
  write_json_string(output, context.selector_arm
                                ? "whole-request-live-rows"
                                : "legacy-absolute-row-order-incremental");
  output << ",\"tile_count\":"
         << (context.selector_arm ? 1U : kLegacyTileCount)
         << ",\"hash_update_count\":"
         << (context.selector_arm ? 1U : kLegacyTileCount)
         << ",\"tile_rows\":[";
  if (context.selector_arm) {
    output << kPromptTokens;
  } else {
    for (std::size_t index = 0U; index < kLegacyTileCount; ++index) {
      if (index != 0U) {
        output.put(',');
      }
      output << context.legacy_residual.tile_rows[index];
    }
  }
  output << "],\"tile_first_positions\":[";
  if (context.selector_arm) {
    output << '0';
  } else {
    for (std::size_t index = 0U; index < kLegacyTileCount; ++index) {
      if (index != 0U) {
        output.put(',');
      }
      output << context.legacy_residual.first_positions[index];
    }
  }
  output << "],\"first_row\":0,\"last_row\":39999"
            ",\"absolute_row_order\":true,\"contiguous\":true"
            ",\"source_owner\":";
  write_json_string(output, snapshot.prompt_residual_source_owner);
  output << ",\"source_buffer\":";
  write_json_string(output, snapshot.prompt_residual_source_buffer);
  output << ",\"source_region_bytes\":"
         << snapshot.prompt_residual_source_region_bytes
         << ",\"source_region_element_capacity\":"
         << snapshot.prompt_residual_source_region_element_capacity << '}';
}

void write_state(std::ostream& output, const StrictCaptureContext& context,
                 const StateSnapshot& snapshot) {
  output << "{\"sequence_length\":" << snapshot.sequence_length
         << ",\"conv_state\":";
  write_component(output, "q3x.conv-state-live-bf16.v1", {48U, 10'240U, 3U},
                  runtime::kRequestConvStateBytes, snapshot.conv_state);
  output << ",\"gdn_state\":";
  write_component(output, "q3x.gdn-state-live-bf16.v1",
                  {48U, 48U, 128U, 128U}, runtime::kRequestGdnStateBytes,
                  snapshot.gdn_state);
  output << ",\"used_key_cache\":[";
  for (std::size_t slot = 0U; slot < kFullAttentionLayerCount; ++slot) {
    if (slot != 0U) {
      output.put(',');
    }
    output << "{\"layer\":" << 3U + 4U * slot
           << ",\"domain\":\"q3x.full-attention-key-cache-used-bf16.v1\""
              ",\"shape\":["
           << snapshot.sequence_length
           << ",4,256],\"bytes\":" << snapshot.used_kv_bytes_per_layer
           << ",\"storage_bytes\":"
           << snapshot.kv_storage_bytes_per_layer
           << ",\"storage_element_capacity\":"
           << snapshot.kv_storage_element_capacity_per_layer
           << ",\"sha256\":";
    write_json_string(output, snapshot.key_cache[slot].hex());
    output.put('}');
  }
  output << "],\"used_value_cache\":[";
  for (std::size_t slot = 0U; slot < kFullAttentionLayerCount; ++slot) {
    if (slot != 0U) {
      output.put(',');
    }
    output << "{\"layer\":" << 3U + 4U * slot
           << ",\"domain\":\"q3x.full-attention-value-cache-used-bf16.v1\""
              ",\"shape\":["
           << snapshot.sequence_length
           << ",4,256],\"bytes\":" << snapshot.used_kv_bytes_per_layer
           << ",\"storage_bytes\":"
           << snapshot.kv_storage_bytes_per_layer
           << ",\"storage_element_capacity\":"
           << snapshot.kv_storage_element_capacity_per_layer
           << ",\"sha256\":";
    write_json_string(output, snapshot.value_cache[slot].hex());
    output.put('}');
  }
  output << "],\"prompt_residual\":";
  write_prompt_residual(output, context, snapshot);
  output << ",\"final_hidden\":{\"position\":"
         << snapshot.final_hidden_position << ",\"source_owner\":";
  write_json_string(output, snapshot.final_hidden_source_owner);
  output << ",\"source_buffer\":";
  write_json_string(output, snapshot.final_hidden_source_buffer);
  output << ",\"source_byte_offset\":"
         << snapshot.final_hidden_source_byte_offset
         << ",\"source_region_bytes\":"
         << snapshot.final_hidden_source_region_bytes
         << ",\"source_region_element_capacity\":"
         << snapshot.final_hidden_source_region_element_capacity
         << ",\"source_row_capacity\":"
         << snapshot.final_hidden_source_row_capacity
         << ",\"source_row_stride_elements\":"
         << snapshot.final_hidden_source_row_stride_elements
         << ",\"domain\":\"q3x.final-hidden-live-bf16.v1\""
            ",\"shape\":[5120],\"bytes\":10240,\"sha256\":";
  write_json_string(output, snapshot.final_hidden.hex());
  output << "},\"aggregate_sha256\":";
  write_json_string(output, snapshot.aggregate.hex());
  output.put('}');
}

void write_generation(std::ostream& output,
                      const runtime::ReferenceGeneration& generation,
                      const TokenCollector& tokens,
                      const std::vector<const runtime::ReferenceStepResult*>&
                          logit_steps) {
  output << "{\"sequence_length\":" << kRequiredSteps
         << ",\"generated_token_ids\":[";
  for (std::size_t index = 0U;
       index < generation.generated_token_ids.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    output << generation.generated_token_ids[index];
  }
  output << "],\"generated_text\":";
  write_json_string(output, generation.generated_text);
  output << ",\"stop_reason\":\"max_new_tokens\""
            ",\"full_statistics_steps\":[";
  for (std::size_t index = 0U; index < logit_steps.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const runtime::ReferenceStepResult& step = *logit_steps[index];
    const runtime::ReferenceStepLogits& logits = *step.logits;
    output << "{\"index\":" << index << ",\"position\":" << step.position
           << ",\"input_token_id\":" << step.input_token_id
           << ",\"predicted_token_id\":" << logits.predicted_token_id
           << ",\"chosen_logit_f32_bits\":";
    write_json_string(output, f32_bits(logits.chosen_logit));
    output << ",\"logsumexp_f64_bits\":";
    write_json_string(output, f64_bits(logits.logsumexp));
    output << ",\"max_log_probability_f64_bits\":";
    write_json_string(output, f64_bits(logits.max_log_probability));
    output.put('}');
  }
  output << "],\"token_events\":[";
  for (std::size_t index = 0U; index < tokens.events.size(); ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const TokenEvent& event = tokens.events[index];
    output << "{\"index\":" << event.index << ",\"token_id\":"
           << event.token_id << ",\"text_delta\":";
    write_json_string(output, event.text_delta);
    output << ",\"generated_text\":";
    write_json_string(output, event.generated_text);
    output << ",\"is_stop_token\":"
           << (event.is_stop_token ? "true" : "false") << '}';
  }
  output << "]}";
}

constexpr std::array<std::string_view, runtime::kPrefillOperatorRoleCount>
    kOperatorNames{{"nvfp4_gate_up", "nvfp4_down", "fp8_qkv", "fp8_z",
                    "fp8_o", "attention", "gdn"}};

void write_public_route(std::ostream& output,
                        const runtime::PrefillRouteEvidence& evidence) {
  output << "{\"complete\":" << (evidence.complete ? "true" : "false")
         << ",\"valid\":" << (evidence.valid ? "true" : "false")
         << ",\"request_active\":"
         << (evidence.request_active ? "true" : "false")
         << ",\"error\":";
  write_json_string(output, runtime::to_string(evidence.error));
  output << ",\"completed_layer_passes\":"
         << evidence.completed_layer_passes
         << ",\"expected_layer_passes\":" << evidence.expected_layer_passes
         << ",\"operators\":{";
  for (std::size_t role = 0U; role < runtime::kPrefillOperatorRoleCount;
       ++role) {
    if (role != 0U) {
      output.put(',');
    }
    write_json_string(output, kOperatorNames[role]);
    const auto& count = evidence.operators[role];
    output << ":{\"production_hits\":" << count.production_hits
           << ",\"exact_fallback_hits\":" << count.exact_fallback_hits
           << ",\"forbidden_hits\":" << count.forbidden_hits << '}';
  }
  output << "},\"forbidden_boundary_hits\":[";
  for (std::size_t index = 0U;
       index < runtime::kPrefillForbiddenBoundaryCount; ++index) {
    if (index != 0U) {
      output.put(',');
    }
    output << evidence.forbidden_boundary_hits[index];
  }
  output << "]}";
}

const char* span_tactic(
    const engine_detail::NativePrefillPhysicalSubmissionTactic tactic)
    noexcept {
  switch (tactic) {
    case engine_detail::NativePrefillPhysicalSubmissionTactic::kGroupQ64:
      return "group-q64";
    case engine_detail::NativePrefillPhysicalSubmissionTactic::
        kPersistentGenericQt2Q8:
      return "persistent-generic-qt2-q8";
    case engine_detail::NativePrefillPhysicalSubmissionTactic::kNone:
      return "none";
  }
  return "unknown";
}

template <typename SpanArray>
void write_physical_spans(std::ostream& output, const SpanArray& spans,
                          const std::size_t count) {
  output.put('[');
  for (std::size_t index = 0U; index < count; ++index) {
    if (index != 0U) {
      output.put(',');
    }
    output << "{\"tactic\":";
    write_json_string(output, span_tactic(spans[index].tactic));
    output << ",\"first_position\":" << spans[index].first_position
           << ",\"token_count\":" << spans[index].token_count << '}';
  }
  output.put(']');
}

struct SelectorTotals {
  std::uint64_t full_attention_layer_hits = 0U;
  std::uint64_t panel_calls = 0U;
  std::uint64_t arithmetic_spans = 0U;
  std::uint64_t group_q64_submissions = 0U;
  std::uint64_t generic_qt2_spans = 0U;
  std::uint64_t generic_q8_suffix_submissions = 0U;
  std::uint64_t fallback_submissions = 0U;
  std::uint64_t persistent_ctas = 0U;
  std::uint64_t physical_submissions = 0U;
  std::uint32_t minimum_physical_tokens = 0U;
  std::uint32_t maximum_physical_tokens = 0U;
  std::uint32_t logical_prompt_tokens = 0U;
};

SelectorTotals private_selector_totals(
    const runner_detail::SelectorExactPersistentAttentionV1RouteReceipt&
        receipt) noexcept {
  return {receipt.issued_layer_count,
          receipt.panel_calls,
          receipt.arithmetic_spans,
          receipt.group_q64_submissions,
          receipt.generic_qt2_spans,
          receipt.generic_q8_suffix_submissions,
          receipt.fallback_submissions,
          receipt.persistent_ctas,
          receipt.group_q64_submissions +
              receipt.generic_q8_suffix_submissions,
          receipt.minimum_physical_submission_tokens,
          receipt.maximum_physical_submission_tokens,
          receipt.maximum_logical_panel_tokens};
}

SelectorTotals completed_selector_totals(
    const engine_detail::
        P40000SelectorExactPersistentAttentionV1CompletedReceipt& receipt)
    noexcept {
  return {receipt.full_attention_layer_hits,
          receipt.panel_calls,
          receipt.arithmetic_spans,
          receipt.group_q64_submissions,
          receipt.generic_qt2_spans,
          receipt.generic_q8_suffix_submissions,
          receipt.fallback_submissions,
          receipt.persistent_ctas,
          receipt.physical_submissions,
          receipt.minimum_physical_tokens,
          receipt.maximum_physical_tokens,
          receipt.logical_prompt_tokens};
}

void write_selector_totals(std::ostream& output,
                           const SelectorTotals& totals) {
  output << "\"full_attention_layer_hits\":"
         << totals.full_attention_layer_hits
         << ",\"panel_calls\":" << totals.panel_calls
         << ",\"arithmetic_spans\":" << totals.arithmetic_spans
         << ",\"group_q64_submissions\":" << totals.group_q64_submissions
         << ",\"generic_qt2_spans\":" << totals.generic_qt2_spans
         << ",\"generic_q8_suffix_submissions\":"
         << totals.generic_q8_suffix_submissions
         << ",\"fallback_submissions\":" << totals.fallback_submissions
         << ",\"persistent_ctas\":" << totals.persistent_ctas
         << ",\"physical_submissions\":" << totals.physical_submissions
         << ",\"minimum_physical_tokens\":"
         << totals.minimum_physical_tokens
         << ",\"maximum_physical_tokens\":"
         << totals.maximum_physical_tokens
         << ",\"logical_prompt_tokens\":" << totals.logical_prompt_tokens;
}

const char* attention_role_string(
    const runtime::PrefillBindingRole role) noexcept {
  return role == runtime::PrefillBindingRole::kExactCausalAttention
             ? "exact-causal-attention"
             : "invalid";
}

const char* attention_tactic_string(
    const engine_detail::NativePrefillTactic tactic) noexcept {
  switch (tactic) {
    case engine_detail::NativePrefillTactic::
        kExactCausalAttentionOracleSpanC512C16Reference256:
      return "legacy-c512-exact";
    case engine_detail::NativePrefillTactic::
        kSelectorExactPersistentAttentionV1WholePrompt:
      return "selector-exact-persistent-attention-v1-whole-prompt";
    default:
      return "invalid";
  }
}

const char* attention_completion_string(
    const engine_detail::NativePrefillCompletionDomain completion) noexcept {
  return completion ==
                 engine_detail::NativePrefillCompletionDomain::
                     kMainStreamBarrier
             ? "main-stream-barrier"
             : "invalid";
}

void write_route(std::ostream& output,
                 const runtime::ReferenceGeneration& generation,
                 const StrictCaptureContext& context) {
  const auto& completed = context.completed_receipt;
  const auto& attention = completed.bound_attention_role;
  const bool selector = completed.selector_route;
  output << "{\"public_route_evidence\":";
  write_public_route(output, generation.prefill_route_evidence);
  output << ",\"public_attention_role_receipt\":{\"role\":";
  write_json_string(output, attention_role_string(attention.role));
  output << ",\"runtime_route\":";
  write_json_string(output, selector ? "whole-request-layer-major"
                                     : "legacy-c512-tiled");
  output << ",\"selector\":" << (selector ? "true" : "false");
  if (selector) {
    output << ",\"bound_attention_role\":{\"tactic\":";
    write_json_string(output, attention_tactic_string(attention.tactic));
    output << ",\"completion\":";
    write_json_string(output,
                      attention_completion_string(attention.completion));
    output << ",\"maximum_logical_panel_m\":"
           << attention.maximum_logical_panel_m
           << ",\"minimum_physical_m\":" << attention.minimum_physical_m
           << ",\"maximum_physical_m\":" << attention.maximum_physical_m
           << ",\"physical_submission_count_per_logical_panel\":"
           << attention.physical_submission_count_per_logical_panel
           << ",\"physical_submissions\":";
    write_physical_spans(output, attention.physical_submissions,
                         attention.physical_submission_count_per_logical_panel);
    output.put('}');
  } else {
    output << ",\"validated_legacy_attention_contract\":{\"tactic\":";
    write_json_string(output, attention_tactic_string(attention.tactic));
    output << ",\"completion\":";
    write_json_string(output,
                      attention_completion_string(attention.completion));
    output << ",\"maximum_logical_panel_m\":"
           << attention.maximum_logical_panel_m
           << ",\"minimum_physical_m\":" << attention.minimum_physical_m
           << ",\"maximum_physical_m\":" << attention.maximum_physical_m
           << ",\"physical_submission_count_per_logical_panel\":"
           << attention.physical_submission_count_per_logical_panel
           << ",\"validation_sources\":["
              "\"completed_attention_transaction\","
              "\"public_route_evidence.operators.attention\"]}";
  }
  output << ",\"completed_attention_transaction\":{\"logical_panel_count\":"
         << completed.logical_panel_count
         << ",\"physical_submissions_per_full_attention_layer\":"
         << completed
                .completed_physical_submissions_per_full_attention_layer
         << ",\"physical_submissions_total\":"
         << completed.completed_physical_submissions_total << '}';
  output << "},\"private_selector_receipt\":{";
  write_selector_totals(
      output, private_selector_totals(context.private_selector_receipt));
  output << "},\"completed_selector_receipt\":{";
  write_selector_totals(output, completed_selector_totals(completed));
  output << ",\"completed\":"
         << (completed.completed_physical_receipt ? "true" : "false")
         << ",\"completed_full_attention_layer_count\":"
         << completed.completed_layer_count
         << ",\"physical_submission_count_per_layer\":"
         << completed.completed_physical_submission_count_per_layer
         << ",\"completed_layer_submissions\":[";
  for (std::size_t index = 0U; index < completed.completed_layer_count;
       ++index) {
    if (index != 0U) {
      output.put(',');
    }
    const auto& layer = completed.completed_layers[index];
    output << "{\"layer\":" << layer.layer
           << ",\"physical_submission_count\":"
           << layer.physical_submission_count
           << ",\"physical_submissions\":";
    write_physical_spans(output, layer.physical_submissions,
                         layer.physical_submission_count);
    output.put('}');
  }
  output << "]},\"flashinfer_receipt\":{\"panel_hits\":"
         << generation.prefill_native_flashinfer_exact_panel_hits
         << ",\"whole_prompt_hits\":"
         << generation.prefill_native_flashinfer_exact_whole_prompt_hits
         << "},\"whole_core_receipt\":{\"logical_panel_count\":"
         << (selector ? generation.prefill_logical_panel_count : 0U)
         << ",\"route_layer_pass_count\":"
         << (selector ? generation.prefill_route_layer_pass_count : 0U)
         << ",\"layer_wide_p40_mlp_layer_hits\":"
         << (selector ? generation.prefill_layer_wide_p40_mlp_layer_hits : 0U)
         << ",\"persistent_p40_nvfp4_gate_up_hits\":"
         << (selector ? generation.prefill_persistent_p40_nvfp4_gate_up_hits
                      : 0U)
         << ",\"persistent_p40_nvfp4_down_residual_hits\":"
         << (selector
                 ? generation.prefill_persistent_p40_nvfp4_down_residual_hits
                 : 0U)
         << ",\"persistent_p40_nvfp4_physical_launches\":"
         << (selector
                 ? generation.prefill_persistent_p40_nvfp4_physical_launches
                 : 0U)
         << ",\"whole_core_layer_hits\":"
         << (selector ? generation.prefill_prompt_wide_p40_whole_core_layer_hits
                      : 0U)
         << ",\"fill_panel_hits\":"
         << (selector ? generation.prefill_prompt_wide_p40_fill_panel_hits
                      : 0U)
         << ",\"prompt_core_hits\":"
         << (selector ? generation.prefill_prompt_wide_p40_prompt_core_hits
                      : 0U)
         << ",\"drain_panel_hits\":"
         << (selector ? generation.prefill_prompt_wide_p40_drain_panel_hits
                      : 0U)
         << ",\"fp8_projection_hits\":"
         << (selector ? generation.prefill_prompt_wide_p40_fp8_projection_hits
                      : 0U)
         << ",\"fp8_projection_physical_launches\":"
         << (selector ? generation
                            .prefill_prompt_wide_p40_fp8_projection_physical_launches
                      : 0U)
         << ",\"bf16_ab_hits\":"
         << (selector ? generation.prefill_prompt_wide_p40_bf16_ab_hits : 0U)
         << ",\"gdn_hits\":"
         << (selector ? generation.prefill_prompt_wide_p40_gdn_hits : 0U)
         << ",\"submission_window_retirements\":"
         << (selector ? generation.prefill_submission_window_retirements : 0U)
         << "},\"public_generic_qt2_hits\":"
         << generation.prefill_generic_qt2_hits
         << ",\"legacy_receipt\":{\"complete_64_layer_route\":"
         << (selector ? "false" : "true")
         << ",\"all_prompt_tokens_prefilled_by_tiles\":"
         << (selector ? "false" : "true")
         << ",\"all_prompt_mode\":" << (selector ? "false" : "true")
         << ",\"single_arbitrary_mode\":false,\"global_tile_count\":"
         << (selector ? 0U : kLegacyTileCount)
         << ",\"prefix_execution_count\":"
         << (selector ? 0U : generation.timing.prefix_execution_milliseconds.size())
         << ",\"prefix_timing_count\":"
         << (selector ? 0U : generation.timing.prefix_execution_milliseconds.size())
         << ",\"logical_panel_count\":"
         << (selector ? 0U : generation.prefill_logical_panel_count)
         << ",\"route_layer_pass_count\":"
         << (selector ? 0U : generation.prefill_route_layer_pass_count)
         << ",\"retained_final_token_scalar_passes\":0}}";
}

void write_profile(std::ostream& output, const bool selector) {
  output << "{\"prompt_tokens\":40000,\"max_new_tokens\":16"
            ",\"required_steps\":40015,\"live_sequence_length\":40015"
            ",\"configured_internal_rows\":"
         << (selector ? kSelectorCapacity : kLegacyCapacity);
  if (!selector) {
    output << ",\"ordinary_max_sequence_length\":" << kLegacyCapacity;
  }
  output << ",\"request_arena_bytes\":"
         << (selector ? kSelectorArenaBytes : kLegacyArenaBytes)
         << ",\"deployment_plan_id\":";
  write_json_string(
      output, selector
                  ? runtime::kSelectorExactPersistentAttentionV1P40DeploymentPlanId
                  : std::string_view{});
  output << ",\"ordinary_profile_id\":";
  write_json_string(output, selector ? std::string_view{}
                                     : kLegacyOrdinaryProfileId);
  output << ",\"guard_rows\":" << (selector ? 1U : 0U)
         << ",\"selector_guard_rows\":" << (selector ? 1U : 0U)
         << ",\"ordinary_unused_capacity_rows\":"
         << (selector ? 0U : kLegacyCapacity - kRequiredSteps)
         << ",\"request_memory_profile\":";
  write_json_string(output, selector ? "layer-major-p40-whole-core"
                                     : "legacy-c512");
  output << ",\"prefill_execution_mode\":";
  write_json_string(
      output, selector
                  ? "whole-request-layer-major"
                  : "ReferencePrefillExecutionMode::kLegacyC512Tiled");
  output.put('}');
}

void write_record(std::ostream& output, const bool selector,
                  const CheckpointAttestation& checkpoint,
                  const Corpus& corpus, const StrictCaptureContext& context,
                  const runtime::ReferenceGeneration& generation,
                  const TokenCollector& tokens,
                  const std::vector<const runtime::ReferenceStepResult*>&
                      logit_steps) {
  output << kMarker << "{\"schema\":";
  write_json_string(output, kSchema);
  output << ",\"capture_arm\":";
  write_json_string(output, selector ? kArmSelector : kArmLegacy);
  output << ",\"checkpoint_manifest\":";
  write_checkpoint(output, checkpoint);
  output << ",\"corpus\":{\"sha256\":";
  write_json_string(output, corpus.file_sha256.hex());
  output << ",\"token_ids_u32le_sha256\":";
  write_json_string(output, corpus.token_sha256.hex());
  output << ",\"prompt_tokens\":40000},\"profile\":";
  write_profile(output, selector);
  output << ",\"prefill_commit_state\":";
  write_state(output, context, context.prefill);
  output << ",\"generation_return_state\":";
  write_state(output, context, context.returned);
  output << ",\"generation\":";
  write_generation(output, generation, tokens, logit_steps);
  output << ",\"route\":";
  write_route(output, generation, context);
  output << ",\"claims\":{\"capture_complete\":true"
            ",\"timing_authority\":false,\"release_qualified\":false"
            ",\"production_eligible\":false}}\n";
}

void print_diagnostic(const runtime::ReferenceEngineDiagnostic& diagnostic) {
  std::cerr << "code=" << runtime::to_string(diagnostic.code)
            << " stage=" << diagnostic.stage
            << " message=" << diagnostic.message
            << " context=" << diagnostic.context
            << " cuda_error=" << diagnostic.cuda_error
            << " layer=" << diagnostic.layer
            << " operation=" << diagnostic.operation << '\n';
}

bool validate_selector_host_memory_contract(std::string& error) {
  runtime::LayerMajorP40WholeCoreWorkspaceOptions workspace_options;
  workspace_options.request_sequence_capacity_tokens = kSelectorCapacity;
  workspace_options.request_arena_limit_bytes = kSelectorArenaBytes;
  const runtime::LayerMajorP40WholeCoreWorkspacePlanResult workspace_result =
      runtime::build_unbound_layer_major_p40_whole_core_workspace_plan(
          workspace_options);
  if (!workspace_result ||
      workspace_result.value->persistent_and_kv.required_bytes !=
          2'700'935'168U ||
      workspace_result.value->prompt_residual_bf16.required_bytes !=
          409'763'840U ||
      workspace_result.value->legacy_c512_workspace.required_bytes !=
          90'971'648U ||
      workspace_result.value->rope_cos_sin_fp32.required_bytes !=
          10'244'096U ||
      workspace_result.value->required_bytes != kSelectorArenaBytes ||
      workspace_result.value->capacity !=
          runtime::PrefillMemoryCapacityVerdict::kFitsDeclaredLimit) {
    error = "selector workspace planner disagrees with the P40016 arena ledger";
    return false;
  }

  runtime::LayerMajorP40WholeCoreWorkspaceOptions invalid_capacity =
      workspace_options;
  invalid_capacity.request_sequence_capacity_tokens = kRequiredSteps;
  const runtime::LayerMajorP40WholeCoreWorkspacePlanResult invalid_workspace =
      runtime::build_unbound_layer_major_p40_whole_core_workspace_plan(
          invalid_capacity);
  if (invalid_workspace ||
      invalid_workspace.error !=
          runtime::PrefillWorkspacePlanError::kInvalidArgument) {
    error = "selector workspace planner accepted P40015 as configured capacity";
    return false;
  }

  runtime::LayerMajorRequestMemoryOptions request_options;
  request_options.max_sequence_length = kSelectorCapacity;
  request_options.max_arena_bytes = kSelectorArenaBytes;
  request_options.layout =
      runtime::LayerMajorRequestLayout::kP40WholeCorePromptWide;
  request_options.mlp_layout =
      runtime::LayerMajorRequestMlpLayout::kLayerWideP40PersistentTwoSpan;
  const runtime::LayerMajorRequestPlanResult request_result =
      runtime::build_layer_major_request_memory_plan(request_options);
  if (!request_result) {
    error = "selector RequestState planner rejected the exact P40016 ledger";
    return false;
  }
  const runtime::LayerMajorRequestMemoryPlan& request = *request_result.value;
  const runtime::LayerMajorLegacyC512Regions& legacy = request.legacy_c512;
  if (request.common.max_sequence_length != kSelectorCapacity ||
      request.common.persistent_bytes != 2'700'935'168U ||
      request.prompt_residual_bf16.row_capacity != kSelectorCapacity ||
      request.prompt_residual_bf16.storage.byte_size != 409'763'840U ||
      request.p40_whole_core.family_phase_arena.arena_offset !=
          3'110'699'008U ||
      legacy.hidden_bf16.front().storage.arena_offset != 8'540'459'008U ||
      legacy.fp32_scratch.arena_offset != 8'627'589'120U ||
      legacy.fp32_scratch.element_capacity != 960'384U ||
      legacy.fp32_scratch.byte_size != 3'841'536U ||
      legacy.gqa_probability_scratch.arena_offset !=
          legacy.fp32_scratch.arena_offset ||
      legacy.gqa_probability_scratch.element_capacity != 960'384U ||
      legacy.gqa_probability_scratch.byte_size != 3'841'536U ||
      request.final_hidden_bf16.storage.arena_offset != 8'631'430'656U ||
      request.common.workspace_bytes != 5'930'505'728U ||
      request.common.rope_cos_fp32.arena_offset != 8'631'440'896U ||
      request.common.rope_sin_fp32.arena_offset != 8'636'562'944U ||
      request.common.rope_bytes != 10'244'096U ||
      request.common.arena_bytes != kSelectorArenaBytes) {
    error = "selector RequestState typed views disagree with the P40016 ledger";
    return false;
  }

  const runtime::ReferenceLayerMajorRequestDescriptorOutcome descriptor =
      runtime::build_reference_layer_major_candidate_binding_descriptor(
          request);
  if (!descriptor ||
      descriptor.value->legacy_c512.fp32_scratch.element_capacity !=
          960'384U ||
      descriptor.value->legacy_c512.gqa_probability_scratch.element_capacity !=
          960'384U) {
    error = "selector layer-major descriptor rejected the exact GQA scratch";
    return false;
  }

  runtime::LayerMajorRequestMemoryPlan short_scratch = request;
  --short_scratch.legacy_c512.fp32_scratch.element_capacity;
  short_scratch.legacy_c512.fp32_scratch.byte_size -= sizeof(float);
  short_scratch.legacy_c512.gqa_probability_scratch =
      short_scratch.legacy_c512.fp32_scratch;
  short_scratch.common.fp32_scratch =
      short_scratch.legacy_c512.fp32_scratch;
  short_scratch.common.gqa_probability_scratch =
      short_scratch.legacy_c512.gqa_probability_scratch;
  const runtime::ReferenceLayerMajorRequestDescriptorOutcome rejected =
      runtime::build_reference_layer_major_candidate_binding_descriptor(
          short_scratch);
  if (rejected || rejected.status.operation == nullptr ||
      std::string_view(rejected.status.operation) !=
          "layer_major_legacy_c512_views") {
    error = "selector layer-major descriptor accepted a short GQA scratch";
    return false;
  }

  request_options.max_arena_bytes = kSelectorArenaBytes - 1U;
  const runtime::LayerMajorRequestPlanResult short_arena =
      runtime::build_layer_major_request_memory_plan(request_options);
  if (short_arena || short_arena.diagnostic.code !=
                         runtime::RequestErrorCode::kArenaLimitExceeded) {
    error = "selector RequestState planner accepted a one-byte-short arena";
    return false;
  }
  return true;
}

int run_capture(const std::filesystem::path& model_directory,
                const std::filesystem::path& corpus_path,
                const bool selector) {
  CheckpointAttestation checkpoint;
  Corpus corpus;
  std::string error;
  if (!attest_checkpoint(model_directory, checkpoint, error)) {
    std::cerr << "checkpoint attestation failed: " << error << '\n';
    return 2;
  }
  if (!load_corpus(corpus_path, corpus, error)) {
    std::cerr << "corpus attestation failed: " << error << '\n';
    return 2;
  }

  runtime::ReferenceEngineOptions engine_options;
  engine_options.request_options.prefill_chunk_size = 512U;
  engine_options.request_options.max_sequence_length =
      selector ? kSelectorCapacity : kLegacyCapacity;
  engine_options.request_options.max_arena_bytes =
      selector ? kSelectorArenaBytes : kLegacyArenaBytes;
  engine_options.request_options.min_free_bytes_after_create =
      selector ? 4ULL * 1024ULL * 1024ULL * 1024ULL
               : 8ULL * 1024ULL * 1024ULL * 1024ULL;
  engine_options.projection_backend = runtime::ProjectionBackend::kSm87WeightOnly;
  engine_options.decode_graph_cache_policy =
      runtime::ReferenceDecodeGraphCachePolicy::kSm87ShortPositions;
  engine_options.prefill_execution_mode =
      selector ? runtime::ReferencePrefillExecutionMode::kWholeRequestLayerMajor
               : runtime::ReferencePrefillExecutionMode::kLegacyC512Tiled;
  engine_options.prefill_full_attention_tactic =
      selector
          ? runtime::LayerMajorPrefillFullAttentionTactic::
                kSelectorExactPersistentAttentionV1WholePrompt
          : runtime::LayerMajorPrefillFullAttentionTactic::
                kExactSegmentedC512;
  engine_options.prefill_projection_tactic =
      selector
          ? runtime::LayerMajorPrefillProjectionTactic::
                kNativePromptWideP40WholeCore
          : runtime::LayerMajorPrefillProjectionTactic::kExactSegmentedC512;

  runtime::ReferenceEngineCreateResult created =
      runtime::create_reference_engine(model_directory, engine_options);
  if (!created) {
    std::cerr << "engine creation failed: ";
    print_diagnostic(created.diagnostic);
    return 1;
  }

  StrictCaptureContext context(selector);
  TokenCollector token_collector;
  runtime::ReferenceGenerateResult generated;
  (void)runner_detail::
      exchange_selector_exact_persistent_attention_v1_route_receipt_for_test(
          {});
  {
    const ScopedFinalTokenPolicy final_policy(
        selector
            ? engine_detail::ReferenceEnginePrefillFinalTokenPolicyForTest::
                  kProductionDefault
            : engine_detail::ReferenceEnginePrefillFinalTokenPolicyForTest::
                  kAllPromptTiles);
    const ScopedLegacyResidualHook residual_hook(
        selector ? nullptr : &context.legacy_residual);
    const ScopedPrefillHook prefill_hook(context);
    const ScopedReturnHook return_hook(context);
    runtime::ReferenceGenerateOptions options;
    options.max_new_tokens = kMaxNewTokens;
    options.stop_token_id = kStopTokenId;
    options.prefill_chunk_size = 512U;
    options.logits_mode = runtime::ReferenceLogitsMode::kFullStatistics;
    options.token_observer = collect_token;
    options.token_observer_context = &token_collector;
    options.prefill_execution_mode = engine_options.prefill_execution_mode;
    if (selector) {
      options.prefill_cancellation_probe = never_cancel;
    }
    generated = created.value->generate_prompt_token_ids(corpus.tokens, options);
  }
  context.private_selector_receipt = runner_detail::
      exchange_selector_exact_persistent_attention_v1_route_receipt_for_test(
          {});
  if (!generated) {
    std::cerr << "strict generation failed: ";
    print_diagnostic(generated.diagnostic);
    return 1;
  }
  if (!context.prefill_hook_completed || !context.return_hook_completed ||
      context.error != CaptureError::kNone ||
      !valid_state_snapshot(context.prefill, kPromptTokens, selector) ||
      !valid_state_snapshot(context.returned, kRequiredSteps, selector)) {
    std::cerr << "strict state capture failed context_error="
              << to_string(context.error)
              << " prefill_error=" << to_string(context.prefill.error)
              << " return_error=" << to_string(context.returned.error)
              << " prefill_cuda=" << context.prefill.cuda_error
              << " return_cuda=" << context.returned.cuda_error << '\n';
    return 1;
  }
  const runtime::ReferenceGeneration& generation = *generated.value;
  if (!(selector ? valid_selector_generation(generation, context)
                 : valid_legacy_generation(generation, context))) {
    std::cerr << "strict route receipt rejected for "
              << (selector ? kArmSelector : kArmLegacy) << '\n';
    return 1;
  }
  std::vector<const runtime::ReferenceStepResult*> logit_steps;
  if (!validate_generation_behavior(generation, corpus, token_collector,
                                    logit_steps, error)) {
    std::cerr << "strict generation witness rejected: " << error << '\n';
    return 1;
  }
  write_record(std::cout, selector, checkpoint, corpus, context, generation,
               token_collector, logit_steps);
  return 0;
}

}  // namespace

int main(const int argc, char** const argv) {
  std::string host_memory_error;
  if (!validate_selector_host_memory_contract(host_memory_error)) {
    std::cerr << "selector host memory contract rejected: "
              << host_memory_error << '\n';
    return 1;
  }
  if (argc == 2 && argv[1] != nullptr &&
      std::string_view(argv[1]) == "--host-memory-contract-only") {
    std::cout << "P40016_HOST_MEMORY_CONTRACT_OK\n";
    return 0;
  }
  if (argc != 2 || argv[1] == nullptr || argv[1][0] == '\0') {
    std::cerr << "usage: q3x_reference_p40000_o16_strict_capture_test "
                 "MODEL_DIR|--host-memory-contract-only\n";
    return 2;
  }
  const char* const enabled = std::getenv("Q3X_RUN_P40000_STRICT_CAPTURE");
  const char* const arm = std::getenv("Q3X_P40000_CAPTURE_ARM");
  const char* const corpus =
      std::getenv("Q3X_TEST_PREFILL_ATTENTION_CORPUS");
  if (enabled == nullptr || std::string_view(enabled) != "1" || arm == nullptr ||
      corpus == nullptr || corpus[0] == '\0') {
    std::cerr << "strict capture requires Q3X_RUN_P40000_STRICT_CAPTURE=1, "
                 "Q3X_P40000_CAPTURE_ARM, and the frozen corpus\n";
    return 2;
  }
  const std::string_view arm_value(arm);
  if (arm_value != kArmLegacy && arm_value != kArmSelector) {
    std::cerr << "Q3X_P40000_CAPTURE_ARM must be legacy-c512-exact or "
                 "p40016-whole-core-selector\n";
    return 2;
  }
  try {
    return run_capture(std::filesystem::path(argv[1]),
                       std::filesystem::path(corpus),
                       arm_value == kArmSelector);
  } catch (const std::exception& exception) {
    std::cerr << "strict capture exception: " << exception.what() << '\n';
    return 1;
  } catch (...) {
    std::cerr << "strict capture unknown exception\n";
    return 1;
  }
}
