#include "q3x/runtime/reference_engine.h"

#include "q3x/core/sha256.h"
#if defined(Q3X_ENABLE_FP8_PREFILL_SUPERMATRIX_ADMISSION)
#include "q3x/kernels/sm87_fp8_prefill_supermatrix.h"
#endif
#include "q3x/kernels/sm87_weight_only_gemv.h"
#include "q3x/text/tokenizer.h"

#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <future>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace q3x::runtime {
namespace {

using Clock = std::chrono::steady_clock;

inline constexpr std::uint32_t kProductionDecodeGraphFirstPosition = 19U;
inline constexpr std::uint32_t kProductionDecodeGraphLastPosition = 43U;
inline constexpr std::uint32_t kProductionDecodeGraphCaptureTokenId = 0U;
inline constexpr double kProductionDecodeGraphMaximumPrepareMilliseconds =
    1'000.0;
inline constexpr std::uint64_t kProductionDecodeGraphMaximumFreeDropBytes =
    256ULL * 1024ULL * 1024ULL;

struct DecodeGraphP1DeviceBuffer {
  void* data = nullptr;
  std::size_t bytes = 0U;

  DecodeGraphP1DeviceBuffer() noexcept = default;
  DecodeGraphP1DeviceBuffer(const DecodeGraphP1DeviceBuffer&) = delete;
  DecodeGraphP1DeviceBuffer& operator=(const DecodeGraphP1DeviceBuffer&) =
      delete;
  ~DecodeGraphP1DeviceBuffer() {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
  }

  [[nodiscard]] cudaError_t allocate(const std::size_t requested_bytes) {
    if (data != nullptr || requested_bytes == 0U) {
      return cudaErrorInvalidValue;
    }
    const cudaError_t status = cudaMalloc(&data, requested_bytes);
    if (status == cudaSuccess) {
      bytes = requested_bytes;
    }
    return status;
  }
};

struct DecodeGraphP1StateRestoreGuard {
  RequestState* state = nullptr;
  void* arena = nullptr;
  const void* snapshot = nullptr;
  std::size_t bytes = 0U;
  std::uint32_t position = 0U;
  bool armed = false;

  ~DecodeGraphP1StateRestoreGuard() {
    if (!armed || state == nullptr || arena == nullptr || snapshot == nullptr ||
        bytes == 0U) {
      return;
    }
    if (cudaMemcpy(arena, snapshot, bytes, cudaMemcpyDeviceToDevice) ==
        cudaSuccess) {
      if (cudaDeviceSynchronize() == cudaSuccess) {
        (void)state->set_sequence_length(position);
      }
    }
  }
};

struct Sm87Fp8OutputProjectionSidecars {
  std::uint8_t* data = nullptr;
  std::size_t bytes = 0U;

  Sm87Fp8OutputProjectionSidecars() noexcept = default;
  Sm87Fp8OutputProjectionSidecars(
      const Sm87Fp8OutputProjectionSidecars&) = delete;
  Sm87Fp8OutputProjectionSidecars& operator=(
      const Sm87Fp8OutputProjectionSidecars&) = delete;

  ~Sm87Fp8OutputProjectionSidecars() { release(); }

  void release() noexcept {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
    data = nullptr;
    bytes = 0U;
  }
};

struct Sm87Fp8OutputSidecarPreparation {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t layers = 0U;
  std::uint64_t bytes = 0U;
  int cuda_error = 0;
  std::string message;
  std::string fallback_reason;
};

struct Sm87NvFp4DownScale6Sidecars {
  std::uint8_t* data = nullptr;
  std::size_t bytes = 0U;
  std::vector<NvFp4DownScale6SidecarDescriptor> descriptors;

  Sm87NvFp4DownScale6Sidecars() noexcept = default;
  Sm87NvFp4DownScale6Sidecars(
      const Sm87NvFp4DownScale6Sidecars&) = delete;
  Sm87NvFp4DownScale6Sidecars& operator=(
      const Sm87NvFp4DownScale6Sidecars&) = delete;

  ~Sm87NvFp4DownScale6Sidecars() { release(); }

  void release() noexcept {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
    data = nullptr;
    bytes = 0U;
    descriptors.clear();
  }
};

struct Sm87NvFp4DownScale6Preparation {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t eligible_layers = 0U;
  std::size_t fallback_layers = 0U;
  std::uint64_t bytes = 0U;
  int cuda_error = 0;
  std::string message;
  std::string fallback_reason;
};

struct Sm87Fp8PrefillQkvSidecars {
  std::uint8_t* data = nullptr;
  std::size_t bytes = 0U;
  std::vector<Fp8PrefillQkvRegisterFeedSidecarDescriptor> descriptors;

  Sm87Fp8PrefillQkvSidecars() noexcept = default;
  Sm87Fp8PrefillQkvSidecars(const Sm87Fp8PrefillQkvSidecars&) = delete;
  Sm87Fp8PrefillQkvSidecars& operator=(
      const Sm87Fp8PrefillQkvSidecars&) = delete;

  ~Sm87Fp8PrefillQkvSidecars() { release(); }

  void release() noexcept {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
    data = nullptr;
    bytes = 0U;
    descriptors.clear();
  }
};

struct Sm87Fp8PrefillQkvPreparation {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t layers = 0U;
  std::uint64_t bytes = 0U;
  int cuda_error = 0;
  std::string message;
  std::string fallback_reason;
};

#if defined(Q3X_ENABLE_FP8_PREFILL_SUPERMATRIX_ADMISSION)
struct Sm87Fp8PrefillSupermatrixSidecars {
  std::uint8_t* data = nullptr;
  std::size_t bytes = 0U;

  Sm87Fp8PrefillSupermatrixSidecars() noexcept = default;
  Sm87Fp8PrefillSupermatrixSidecars(
      const Sm87Fp8PrefillSupermatrixSidecars&) = delete;
  Sm87Fp8PrefillSupermatrixSidecars& operator=(
      const Sm87Fp8PrefillSupermatrixSidecars&) = delete;

  ~Sm87Fp8PrefillSupermatrixSidecars() { release(); }

  void release() noexcept {
    if (data != nullptr) {
      (void)cudaFree(data);
    }
    data = nullptr;
    bytes = 0U;
  }
};

struct Sm87Fp8PrefillSupermatrixPreparation {
  bool enabled = false;
  bool hard_failure = false;
  std::size_t projections = 0U;
  std::uint64_t bytes = 0U;
  int cuda_error = 0;
  std::string message;
};

struct Fp8PrefillSupermatrixProjectionPlan {
  const Fp8LinearWeight* projection = nullptr;
};

[[nodiscard]] bool fp8_prefill_supermatrix_admission_requested() noexcept {
  const char* const value =
      std::getenv("Q3X_RUN_FP8_PREFILL_SUPERMATRIX_ADMISSION");
  return value != nullptr && std::strcmp(value, "1") == 0;
}
#endif

constexpr std::size_t kNvFp4DownScale6GroupSize = 16U;
constexpr std::size_t kNvFp4DownScale6ColumnsPerTile = 512U;
constexpr std::size_t kNvFp4DownScale6RowsPerQuad = 4U;
constexpr std::size_t kNvFp4DownScale6BytesPerTile = 96U;
constexpr unsigned int kNvFp4DownScale6MaximumBase = 192U;
constexpr std::size_t kNvFp4DownScale6RequiredAlignment = 32U;
constexpr std::size_t kNvFp4DownCanonicalScaleBytesPerLayer =
    kNvFp4DownScale6Rows *
    (kNvFp4DownScale6Columns / kNvFp4DownScale6GroupSize);
constexpr std::size_t kNvFp4DownDerivedScale6BytesPerLayer =
    (kNvFp4DownScale6Rows / kNvFp4DownScale6RowsPerQuad) *
    (kNvFp4DownScale6Columns / kNvFp4DownScale6ColumnsPerTile) *
    kNvFp4DownScale6BytesPerTile;
static_assert(kNvFp4DownDerivedScale6BytesPerLayer ==
              kNvFp4DownScale6SidecarBytesPerProjection);
static_assert((kNvFp4DownScale6SidecarBytesPerProjection %
               kNvFp4DownScale6RequiredAlignment) == 0U);

struct NvFp4DownScale6LayerPlan {
  std::size_t layer_index = 0U;
  const NvFp4LinearWeight* down = nullptr;
  unsigned int scale_base = 0U;
};

struct Fp8PrefillQkvLayerPlan {
  std::size_t layer_index = 0U;
  const Fp8LinearWeight* qkv = nullptr;
};

[[nodiscard]] const Fp8LinearWeight* attention_output_projection(
    const DecoderLayerWeights& layer) noexcept {
  if (const auto* const linear =
          std::get_if<LinearAttentionWeights>(&layer.attention)) {
    return std::get_if<Fp8LinearWeight>(&linear->out_proj);
  }
  if (const auto* const full =
          std::get_if<FullAttentionWeights>(&layer.attention)) {
    return std::get_if<Fp8LinearWeight>(&full->o_proj);
  }
  return nullptr;
}

[[nodiscard]] const Fp8LinearWeight* linear_attention_qkv_projection(
    const DecoderLayerWeights& layer) noexcept {
  const auto* const attention =
      std::get_if<LinearAttentionWeights>(&layer.attention);
  if (attention == nullptr ||
      attention->in_proj_qkv.valueless_by_exception()) {
    return nullptr;
  }
  return std::get_if<Fp8LinearWeight>(&attention->in_proj_qkv);
}

[[nodiscard]] bool is_exact_fp8_prefill_qkv_payload(
    const Fp8LinearWeight* const qkv) noexcept {
  return qkv != nullptr && qkv->weight != nullptr &&
         qkv->weight_scale_device != nullptr &&
         qkv->input_scale_device != nullptr &&
         std::isfinite(qkv->weight_scale) && qkv->weight_scale >= 0.0F &&
         std::isfinite(qkv->input_scale) && qkv->input_scale >= 0.0F &&
         qkv->output_size == kFp8PrefillQkvRegisterFeedRows &&
         qkv->input_size == kFp8PrefillQkvRegisterFeedColumns &&
         qkv->prefill_qkv_register_feed_sidecar == nullptr &&
         (reinterpret_cast<std::uintptr_t>(qkv->weight) % 16U) == 0U;
}

[[nodiscard]] const NvFp4LinearWeight* exact_nvfp4_down_projection(
    const DecoderLayerWeights& layer) noexcept {
  if (!supports_nvfp4_down_residual_norm_fusion(
          ProjectionBackend::kSm87WeightOnly, layer.mlp.down_proj)) {
    return nullptr;
  }
  return std::get_if<NvFp4LinearWeight>(&layer.mlp.down_proj);
}

[[nodiscard]] bool derive_nvfp4_down_scale6_base(
    const std::vector<std::uint8_t>& canonical_scales,
    unsigned int& scale_base) noexcept {
  if (canonical_scales.size() !=
      kNvFp4DownCanonicalScaleBytesPerLayer) {
    return false;
  }
  std::uint8_t minimum = std::numeric_limits<std::uint8_t>::max();
  std::uint8_t maximum = 0U;
  for (const std::uint8_t scale : canonical_scales) {
    minimum = std::min(minimum, scale);
    maximum = std::max(maximum, scale);
  }
  scale_base = std::min<unsigned int>(minimum,
                                      kNvFp4DownScale6MaximumBase);
  return static_cast<unsigned int>(maximum) - scale_base <= 63U;
}

[[nodiscard]] bool pack_nvfp4_down_scale6_sidecar(
    const std::vector<std::uint8_t>& canonical_scales,
    const unsigned int scale_base,
    std::vector<std::uint8_t>& packed) noexcept {
  if (canonical_scales.size() !=
          kNvFp4DownCanonicalScaleBytesPerLayer ||
      packed.size() != kNvFp4DownScale6SidecarBytesPerProjection ||
      scale_base > kNvFp4DownScale6MaximumBase) {
    return false;
  }
  std::fill(packed.begin(), packed.end(), 0U);
  constexpr std::size_t kScaleColumns =
      kNvFp4DownScale6Columns / kNvFp4DownScale6GroupSize;
  constexpr std::size_t kRowQuads =
      kNvFp4DownScale6Rows / kNvFp4DownScale6RowsPerQuad;
  constexpr std::size_t kTilesPerRowQuad =
      kNvFp4DownScale6Columns / kNvFp4DownScale6ColumnsPerTile;
  for (std::size_t row_quad = 0U; row_quad < kRowQuads; ++row_quad) {
    for (std::size_t tile = 0U; tile < kTilesPerRowQuad; ++tile) {
      const std::size_t tile_byte =
          (row_quad * kTilesPerRowQuad + tile) *
          kNvFp4DownScale6BytesPerTile;
      for (std::size_t lane_pair = 0U; lane_pair < 16U; ++lane_pair) {
        for (std::size_t phase = 0U; phase < 2U; ++phase) {
          for (std::size_t row = 0U; row < 4U; ++row) {
            const std::size_t code =
                8U * lane_pair + 4U * phase + row;
            const std::size_t canonical_column =
                32U * tile + lane_pair + 16U * phase;
            const unsigned int scale = canonical_scales[
                (4U * row_quad + row) * kScaleColumns +
                canonical_column];
            if (scale < scale_base || scale - scale_base > 63U) {
              return false;
            }
            const unsigned int delta = scale - scale_base;
            const std::size_t first_bit = code * 6U;
            const std::size_t byte = tile_byte + first_bit / 8U;
            const unsigned int shift =
                static_cast<unsigned int>(first_bit % 8U);
            const unsigned int payload = delta << shift;
            packed[byte] |= static_cast<std::uint8_t>(payload);
            if (shift > 2U) {
              packed[byte + 1U] |=
                  static_cast<std::uint8_t>(payload >> 8U);
            }
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] bool verify_nvfp4_down_scale6_sidecar(
    const std::vector<std::uint8_t>& canonical_scales,
    const unsigned int scale_base,
    const std::vector<std::uint8_t>& packed) noexcept {
  if (canonical_scales.size() !=
          kNvFp4DownCanonicalScaleBytesPerLayer ||
      packed.size() != kNvFp4DownScale6SidecarBytesPerProjection ||
      scale_base > kNvFp4DownScale6MaximumBase) {
    return false;
  }
  constexpr std::size_t kScaleColumns =
      kNvFp4DownScale6Columns / kNvFp4DownScale6GroupSize;
  constexpr std::size_t kRowQuads =
      kNvFp4DownScale6Rows / kNvFp4DownScale6RowsPerQuad;
  constexpr std::size_t kTilesPerRowQuad =
      kNvFp4DownScale6Columns / kNvFp4DownScale6ColumnsPerTile;
  for (std::size_t row_quad = 0U; row_quad < kRowQuads; ++row_quad) {
    for (std::size_t tile = 0U; tile < kTilesPerRowQuad; ++tile) {
      const std::size_t tile_byte =
          (row_quad * kTilesPerRowQuad + tile) *
          kNvFp4DownScale6BytesPerTile;
      for (std::size_t lane_pair = 0U; lane_pair < 16U; ++lane_pair) {
        for (std::size_t phase = 0U; phase < 2U; ++phase) {
          for (std::size_t row = 0U; row < 4U; ++row) {
            const std::size_t code =
                8U * lane_pair + 4U * phase + row;
            const std::size_t first_bit = code * 6U;
            const std::size_t byte = tile_byte + first_bit / 8U;
            const unsigned int shift =
                static_cast<unsigned int>(first_bit % 8U);
            unsigned int window = packed[byte];
            if (shift > 2U) {
              window |= static_cast<unsigned int>(packed[byte + 1U])
                        << 8U;
            }
            const unsigned int reconstructed =
                scale_base + ((window >> shift) & 63U);
            const std::size_t canonical_column =
                32U * tile + lane_pair + 16U * phase;
            const unsigned int canonical = canonical_scales[
                (4U * row_quad + row) * kScaleColumns +
                canonical_column];
            if (reconstructed != canonical) {
              return false;
            }
          }
        }
      }
    }
  }
  return true;
}

[[nodiscard]] double elapsed_milliseconds(
    const Clock::time_point begin) noexcept {
  return std::chrono::duration<double, std::milli>(Clock::now() - begin)
      .count();
}

[[nodiscard]] constexpr std::uint64_t decode_graph_position_mask(
    const std::uint32_t first_position,
    const std::uint32_t last_position) noexcept {
  if (first_position > last_position || last_position >= 64U) {
    return 0U;
  }
  const std::uint32_t count = last_position - first_position + 1U;
  const std::uint64_t low_bits =
      count == 64U ? std::numeric_limits<std::uint64_t>::max()
                   : (std::uint64_t{1U} << count) - 1U;
  return low_bits << first_position;
}

[[nodiscard]] std::string decode_graph_prepare_fallback_reason(
    const ReferenceRunnerStatus& status) {
  std::string reason = "runner_prepare_failed:";
  reason += reference_runner_error_string(status.error);
  if (status.operation != nullptr) {
    reason += ":";
    reason += status.operation;
  }
  return reason;
}

[[nodiscard]] std::optional<std::uint64_t>
process_private_memory_bytes() {
  std::ifstream input("/proc/self/smaps_rollup");
  if (!input) {
    return std::nullopt;
  }
  std::uint64_t private_clean_kib = 0U;
  std::uint64_t private_dirty_kib = 0U;
  bool found_clean = false;
  bool found_dirty = false;
  std::string line;
  while (std::getline(input, line)) {
    const bool clean = line.rfind("Private_Clean:", 0U) == 0U;
    const bool dirty = line.rfind("Private_Dirty:", 0U) == 0U;
    if (!clean && !dirty) {
      continue;
    }
    std::istringstream fields(line);
    std::string key;
    std::uint64_t kib = 0U;
    std::string unit;
    if (!(fields >> key >> kib >> unit) || unit != "kB") {
      return std::nullopt;
    }
    if (clean) {
      private_clean_kib = kib;
      found_clean = true;
    } else {
      private_dirty_kib = kib;
      found_dirty = true;
    }
  }
  if (!found_clean || !found_dirty ||
      private_clean_kib >
          std::numeric_limits<std::uint64_t>::max() - private_dirty_kib ||
      private_clean_kib + private_dirty_kib >
          std::numeric_limits<std::uint64_t>::max() / 1024U) {
    return std::nullopt;
  }
  return (private_clean_kib + private_dirty_kib) * 1024U;
}

[[nodiscard]] bool same_generation_semantics(
    const ReferenceGeneration& expected,
    const ReferenceGeneration& actual) noexcept {
  if (expected.rendered_prompt != actual.rendered_prompt ||
      expected.prompt_token_ids != actual.prompt_token_ids ||
      expected.generated_token_ids != actual.generated_token_ids ||
      expected.generated_text != actual.generated_text ||
      expected.stop_reason != actual.stop_reason ||
      expected.requested_prefill_chunk_size !=
          actual.requested_prefill_chunk_size ||
      expected.effective_prefill_chunk_size !=
          actual.effective_prefill_chunk_size ||
      expected.steps.size() != actual.steps.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < expected.steps.size(); ++index) {
    const ReferenceStepResult& left = expected.steps[index];
    const ReferenceStepResult& right = actual.steps[index];
    if (left.position != right.position ||
        left.input_token_id != right.input_token_id ||
        left.logits.has_value() != right.logits.has_value() ||
        left.prediction.has_value() != right.prediction.has_value()) {
      return false;
    }
    if (left.logits.has_value() &&
        (left.logits->predicted_token_id !=
             right.logits->predicted_token_id ||
         left.logits->chosen_logit != right.logits->chosen_logit ||
         left.logits->max_log_probability !=
             right.logits->max_log_probability ||
         left.logits->logsumexp != right.logits->logsumexp)) {
      return false;
    }
    if (left.prediction.has_value() &&
        left.prediction->predicted_token_id !=
            right.prediction->predicted_token_id) {
      return false;
    }
  }
  if (expected.traces.size() != actual.traces.size()) {
    return false;
  }
  for (std::size_t index = 0U; index < expected.traces.size(); ++index) {
    const ReferenceTraceDigest& left = expected.traces[index];
    const ReferenceTraceDigest& right = actual.traces[index];
    if (left.position != right.position ||
        left.input_token_id != right.input_token_id ||
        left.element_count != right.element_count ||
        left.full_sha256 != right.full_sha256 ||
        left.embedding_sha256 != right.embedding_sha256 ||
        left.layer_hidden_sha256 != right.layer_hidden_sha256 ||
        left.layer_residual_sha256 != right.layer_residual_sha256 ||
        left.final_norm_sha256 != right.final_norm_sha256) {
      return false;
    }
  }
  return true;
}

[[nodiscard]] bool decode_milliseconds_per_token(
    const ReferenceGeneration& generation,
    const std::size_t expected_steps, double& milliseconds) noexcept {
  if (expected_steps == 0U ||
      generation.timing.subsequent_token_milliseconds.size() !=
          expected_steps ||
      !std::isfinite(generation.timing.decode_after_first_milliseconds) ||
      generation.timing.decode_after_first_milliseconds < 0.0) {
    return false;
  }
  for (const double sample :
       generation.timing.subsequent_token_milliseconds) {
    if (!std::isfinite(sample) || sample < 0.0) {
      return false;
    }
  }
  milliseconds = generation.timing.decode_after_first_milliseconds /
                 static_cast<double>(expected_steps);
  return std::isfinite(milliseconds) && milliseconds >= 0.0;
}

[[nodiscard]] ReferenceEngineDiagnostic engine_diagnostic(
    const ReferenceEngineError code, std::string stage,
    std::string message, std::string context = {}) {
  ReferenceEngineDiagnostic diagnostic;
  diagnostic.code = code;
  diagnostic.stage = std::move(stage);
  diagnostic.message = std::move(message);
  diagnostic.context = std::move(context);
  return diagnostic;
}

[[nodiscard]] std::optional<ReferenceEngineDiagnostic>
sm87_device_diagnostic(const ProjectionBackend backend,
                       int* const active_device = nullptr) {
  if (backend != ProjectionBackend::kSm87WeightOnly) {
    return std::nullopt;
  }

  int device = 0;
  // Match the resident loader's error isolation: a stale caller launch error
  // must not be attributed to this startup device probe.
  (void)cudaGetLastError();
  cudaError_t cuda_status = cudaGetDevice(&device);
  cudaDeviceProp properties{};
  if (cuda_status == cudaSuccess) {
    cuda_status = cudaGetDeviceProperties(&properties, device);
  }
  if (cuda_status != cudaSuccess) {
    ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
        ReferenceEngineError::kRunnerFactoryFailure,
        "projection_backend_device",
        "failed to inspect the active CUDA device before model load");
    diagnostic.cuda_error = static_cast<int>(cuda_status);
    return diagnostic;
  }
  if (properties.major != 8 || properties.minor != 7) {
    return engine_diagnostic(
        ReferenceEngineError::kInvalidArgument,
        "projection_backend_device",
        "sm87 projection backend requires compute capability 8.7",
        "active_device=sm_" + std::to_string(properties.major) +
            std::to_string(properties.minor));
  }
  if (active_device != nullptr) {
    *active_device = device;
  }
  return std::nullopt;
}

[[nodiscard]] ReferenceEngineDiagnostic tokenizer_diagnostic(
    const std::string_view stage, const text::TokenizerError& error) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      ReferenceEngineError::kTokenizerFailure, std::string(stage),
      error.message, "byte_offset=" + std::to_string(error.offset));
  diagnostic.dependency_error = static_cast<int>(error.code);
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic resident_diagnostic(
    const ResidentLoadDiagnostic& error) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      ReferenceEngineError::kResidentLoadFailure, "resident_load",
      error.message, error.context);
  diagnostic.dependency_error = static_cast<int>(error.code);
  diagnostic.cuda_error = error.cuda_error;
  if (!error.shard.empty()) {
    diagnostic.context = error.shard +
                         (diagnostic.context.empty()
                              ? std::string{}
                              : ":" + diagnostic.context);
  }
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic binding_diagnostic(
    const WeightBindDiagnostic& error) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      ReferenceEngineError::kWeightBindFailure, "weight_bind",
      error.message, error.tensor);
  diagnostic.dependency_error = static_cast<int>(error.code);
  diagnostic.cuda_error = error.cuda_error;
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic request_diagnostic(
    const RequestDiagnostic& error) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      ReferenceEngineError::kRequestStateFailure, "request_state",
      error.message, error.context);
  diagnostic.dependency_error = static_cast<int>(error.code);
  diagnostic.cuda_error = error.cuda_error;
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic runner_diagnostic(
    const ReferenceEngineError code, const std::string_view stage,
    const ReferenceRunnerStatus& status) {
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      code, std::string(stage), reference_runner_error_string(status.error));
  diagnostic.dependency_error = static_cast<int>(status.error);
  diagnostic.cuda_error = status.cuda_error;
  diagnostic.layer = status.layer;
  if (status.operation != nullptr) {
    diagnostic.operation = status.operation;
  }
  return diagnostic;
}

[[nodiscard]] ReferenceEngineDiagnostic control_diagnostic(
    const reference_engine_detail::GenerationControlResult& result) {
  using ControlError = reference_engine_detail::GenerationControlError;
  ReferenceEngineError code = ReferenceEngineError::kInvalidArgument;
  switch (result.error) {
    case ControlError::kNone:
      code = ReferenceEngineError::kNone;
      break;
    case ControlError::kInvalidArgument:
      code = ReferenceEngineError::kInvalidArgument;
      break;
    case ControlError::kCapacityExceeded:
      code = ReferenceEngineError::kCapacityExceeded;
      break;
    case ControlError::kArithmeticOverflow:
      code = ReferenceEngineError::kArithmeticOverflow;
      break;
    case ControlError::kRunnerFailure:
      if (result.runner_status.error ==
          ReferenceRunnerError::kTraceUnavailable) {
        code = ReferenceEngineError::kTraceFailure;
      } else if (result.runner_status.error ==
                 ReferenceRunnerError::kAllocationFailure) {
        code = ReferenceEngineError::kAllocationFailure;
      } else {
        code = ReferenceEngineError::kRunnerStepFailure;
      }
      break;
    case ControlError::kUnexpectedStep:
      code = ReferenceEngineError::kRunnerStepFailure;
      break;
    case ControlError::kMissingLogits:
      code = ReferenceEngineError::kMissingLogits;
      break;
    case ControlError::kMissingTiming:
      code = ReferenceEngineError::kMissingTiming;
      break;
    case ControlError::kAllocationFailure:
      code = ReferenceEngineError::kAllocationFailure;
      break;
    case ControlError::kMissingPrediction:
      code = ReferenceEngineError::kMissingPrediction;
      break;
  }
  ReferenceEngineDiagnostic diagnostic = engine_diagnostic(
      code, "generation_control",
      std::string(reference_engine_detail::to_string(result.error)));
  if (result.error == ControlError::kRunnerFailure) {
    diagnostic.dependency_error =
        static_cast<int>(result.runner_status.error);
    diagnostic.cuda_error = result.runner_status.cuda_error;
    diagnostic.layer = result.runner_status.layer;
    if (result.runner_status.operation != nullptr) {
      diagnostic.operation = result.runner_status.operation;
    }
  }
  return diagnostic;
}

[[nodiscard]] text::ChatResult format_single_user_prompt(
    const text::Tokenizer& tokenizer, const std::string_view user_prompt) {
  std::vector<text::ChatMessage> messages;
  messages.reserve(1U);
  messages.push_back({"user", std::string(user_prompt)});
  text::Qwen36ChatOptions options;
  options.add_generation_prompt = true;
  options.enable_thinking = false;
  return tokenizer.format_qwen36_chat(messages, options);
}

[[nodiscard]] bool hash_span(const ConstBf16Span span,
                             const std::size_t expected_elements,
                             std::string& output) {
  if (span.data == nullptr || span.size != expected_elements ||
      span.size > std::numeric_limits<std::size_t>::max() /
                      sizeof(std::uint16_t)) {
    return false;
  }
  core::Sha256 hash;
  if (!hash.update(span.data, span.size * sizeof(std::uint16_t))) {
    return false;
  }
  output = hash.finalize().hex();
  return true;
}

struct EngineStepContext {
  ReferenceRunner* runner = nullptr;
  std::vector<ReferenceTraceDigest>* traces = nullptr;
  bool capture_trace = false;
  std::size_t decode_graph_replays = 0U;
  std::size_t decode_graph_serial_fallbacks = 0U;
};

[[nodiscard]] ReferenceStepOutcome step_with_trace(
    void* const opaque_context, const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) {
  auto& context = *static_cast<EngineStepContext*>(opaque_context);
  ReferenceStepOutcome outcome = context.runner->step(input_token_id, options);
  if (!outcome || !context.capture_trace) {
    return outcome;
  }

  try {
    const std::optional<ReferenceTraceView> view =
        context.runner->last_trace();
    if (!view.has_value() || view->position != outcome.value->position ||
        view->input_token_id != outcome.value->input_token_id ||
        view->element_count != kReferenceTraceElements) {
      return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                   kReferenceNoLayer, "engine_trace_view"}};
    }

    ReferenceTraceDigest digest;
    digest.position = view->position;
    digest.input_token_id = view->input_token_id;
    digest.element_count = view->element_count;
    if (!hash_span(view->raw(), kReferenceTraceElements,
                   digest.full_sha256) ||
        !hash_span(view->embedding(), kReferenceHiddenSize,
                   digest.embedding_sha256)) {
      return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                   kReferenceNoLayer, "engine_trace_hash"}};
    }
    for (std::size_t layer = 0U;
         layer < kReferenceDecoderLayerCount; ++layer) {
      if (!hash_span(view->layer_hidden(layer), kReferenceHiddenSize,
                     digest.layer_hidden_sha256[layer]) ||
          !hash_span(view->layer_residual(layer), kReferenceHiddenSize,
                     digest.layer_residual_sha256[layer])) {
        return {{}, {ReferenceRunnerError::kTraceUnavailable, 0, layer,
                     "engine_trace_boundary_hash"}};
      }
    }
    if (!hash_span(view->final_norm(), kReferenceHiddenSize,
                   digest.final_norm_sha256)) {
      return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                   kReferenceNoLayer, "engine_trace_final_norm_hash"}};
    }
    context.traces->emplace_back(std::move(digest));
    return outcome;
  } catch (const std::bad_alloc&) {
    return {{}, {ReferenceRunnerError::kAllocationFailure, 0,
                 kReferenceNoLayer, "engine_trace_allocation"}};
  } catch (const std::length_error&) {
    return {{}, {ReferenceRunnerError::kAllocationFailure, 0,
                 kReferenceNoLayer, "engine_trace_allocation"}};
  } catch (...) {
    return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                 kReferenceNoLayer, "engine_trace_exception"}};
  }
}

[[nodiscard]] ReferencePrefillTileOutcome prefill_prefix_tile(
    void* const opaque_context,
    const std::uint32_t* const input_token_ids,
    const std::size_t token_count,
    const ReferencePrefillTileOptions& options) {
  auto& context = *static_cast<EngineStepContext*>(opaque_context);
  if (context.capture_trace) {
    return {{}, {ReferenceRunnerError::kTraceUnavailable, 0,
                 kReferenceNoLayer, "engine_prefill_tile_trace"}};
  }
  return context.runner->prefill_prefix_tile(input_token_ids, token_count,
                                             options);
}

[[nodiscard]] ReferenceStepOutcome decode_with_prepared_graph_cache(
    void* const opaque_context, const std::uint32_t input_token_id,
    const ReferenceStepOptions& options) {
  auto& context = *static_cast<EngineStepContext*>(opaque_context);
  if (options.compute_logits && !options.capture_trace &&
      options.logits_mode == ReferenceLogitsMode::kPredictedTokenOnly &&
      context.runner->has_fixed_position_decode_graph_p1(
          context.runner->current_position())) {
    ++context.decode_graph_replays;
    return context.runner->replay_fixed_position_decode_graph_p1(
        input_token_id, options.measure_timing);
  }
  ++context.decode_graph_serial_fallbacks;
  return step_with_trace(opaque_context, input_token_id, options);
}

[[nodiscard]] bool checked_required_steps(
    const std::size_t prompt_tokens, const std::uint32_t max_new_tokens,
    std::uint64_t& result) noexcept {
  if (max_new_tokens == 0U) {
    return false;
  }
  const std::uint64_t prompt = static_cast<std::uint64_t>(prompt_tokens);
  const std::uint64_t additional =
      static_cast<std::uint64_t>(max_new_tokens) - 1U;
  if (additional > std::numeric_limits<std::uint64_t>::max() - prompt) {
    return false;
  }
  result = prompt + additional;
  return true;
}

struct TimedResidentLoad {
  ResidentLoadResult result;
  double milliseconds = 0.0;
};

[[nodiscard]] TimedResidentLoad load_resident_on_device(
    const std::filesystem::path& model_directory,
    const ResidentLoadOptions& options, const int device) {
  TimedResidentLoad timed;
  const Clock::time_point begin = Clock::now();
  const cudaError_t cuda_status = cudaSetDevice(device);
  if (cuda_status != cudaSuccess) {
    timed.result.diagnostic.code = ResidentLoadErrorCode::kCudaFailure;
    timed.result.diagnostic.message =
        "failed to select the caller CUDA device in the startup worker";
    timed.result.diagnostic.context = "cudaSetDevice(startup worker)";
    timed.result.diagnostic.cuda_error = static_cast<int>(cuda_status);
  } else {
    (void)cudaGetLastError();
    timed.result = load_pinned_qwen36_27b(model_directory, options);
  }
  timed.milliseconds = elapsed_milliseconds(begin);
  return timed;
}

[[nodiscard]] Sm87Fp8OutputSidecarPreparation
prepare_sm87_fp8_output_projection_sidecars(
    ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    Sm87Fp8OutputProjectionSidecars& owner) {
  Sm87Fp8OutputSidecarPreparation result;
  if (owner.data != nullptr || owner.bytes != 0U) {
    result.hard_failure = true;
    result.message = "FP8 output sidecar owner was not empty before prepare";
    return result;
  }

  for (const DecoderLayerWeights& layer : model_weights.layers()) {
    const Fp8LinearWeight* const output =
        attention_output_projection(layer);
    if (output == nullptr || output->weight == nullptr ||
        output->output_size != kFp8M1OutputProjectionRows ||
        output->input_size != kFp8M1OutputProjectionColumns ||
        output->m1_aosoa4_preswizzled_weight != nullptr ||
        (reinterpret_cast<std::uintptr_t>(output->weight) %
         alignof(std::uint32_t)) != 0U) {
      result.fallback_reason = "ineligible_model_weights";
      return result;
    }
  }

  constexpr std::size_t kSidecarBytes =
      kQwen36Fp8M1OutputProjectionAosoa4PreswizzledBytes;
  (void)cudaGetLastError();
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaMemGetInfo failed before FP8 output sidecar prepare";
    return result;
  }
  const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
  if (kSidecarBytes > free_u64 ||
      minimum_free_bytes_after_prepare > free_u64 - kSidecarBytes) {
    result.fallback_reason = "insufficient_device_memory_margin";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(&allocation, kSidecarBytes);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    result.fallback_reason = "cuda_memory_allocation";
    return result;
  }
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = "cudaMalloc failed while creating FP8 output sidecars";
    return result;
  }
  owner.data = static_cast<std::uint8_t*>(allocation);
  owner.bytes = kSidecarBytes;

  // Recheck the configured safety margin after the allocation. The first
  // query prevents a predictably unsafe allocation; this second query closes
  // the normal race with other device allocations and accounts for the
  // driver's actual allocation granularity. A margin miss is optional-path
  // fallback, while inability to query CUDA state remains a hard failure.
  std::size_t remaining_free_bytes = 0U;
  std::size_t remaining_total_bytes = 0U;
  status = cudaMemGetInfo(&remaining_free_bytes, &remaining_total_bytes);
  (void)remaining_total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed after FP8 output sidecar allocation";
    owner.release();
    return result;
  }
  if (static_cast<std::uint64_t>(remaining_free_bytes) <
      minimum_free_bytes_after_prepare) {
    result.fallback_reason =
        "insufficient_device_memory_margin_after_allocation";
    owner.release();
    return result;
  }

  cudaStream_t stream = nullptr;
  status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaStreamCreateWithFlags failed for FP8 output sidecar prepare";
    owner.release();
    return result;
  }

  std::size_t layer_index = 0U;
  for (const DecoderLayerWeights& layer : model_weights.layers()) {
    const Fp8LinearWeight* const output =
        attention_output_projection(layer);
    std::uint8_t* const destination =
        owner.data +
        layer_index *
            kFp8M1OutputProjectionAosoa4PreswizzledBytesPerLayer;
    status = static_cast<cudaError_t>(
        kernels::
            launch_sm87_fp8_w8a16_m1_output_projection_aosoa4_pack_cuda(
                output->weight, destination, output->output_size,
                output->input_size, static_cast<void*>(stream)));
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "FP8 output sidecar pack launch failed at layer " +
          std::to_string(layer_index);
      (void)cudaStreamDestroy(stream);
      owner.release();
      return result;
    }
    ++layer_index;
  }

  status = cudaStreamSynchronize(stream);
  const cudaError_t destroy_status = cudaStreamDestroy(stream);
  if (status != cudaSuccess || destroy_status != cudaSuccess) {
    const cudaError_t failure =
        status != cudaSuccess ? status : destroy_status;
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(failure);
    result.message =
        "FP8 output sidecar pack stream failed to synchronize or destroy";
    owner.release();
    return result;
  }

  if (!model_weights.attach_fp8_m1_output_projection_sidecars(
          owner.data, owner.bytes)) {
    result.hard_failure = true;
    result.message =
        "ModelWeights rejected the complete FP8 output sidecar arena";
    owner.release();
    return result;
  }

  result.enabled = true;
  result.layers = kQwen36DenseLayerCount;
  result.bytes = kSidecarBytes;
  return result;
}

[[nodiscard]] Sm87NvFp4DownScale6Preparation
prepare_sm87_nvfp4_down_scale6_sidecars(
    ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    Sm87NvFp4DownScale6Sidecars& owner) {
  Sm87NvFp4DownScale6Preparation result;
  if (owner.data != nullptr || owner.bytes != 0U ||
      !owner.descriptors.empty()) {
    result.hard_failure = true;
    result.message =
        "NVFP4 down scale6 sidecar owner was not empty before prepare";
    return result;
  }

  // Phase one validates the complete 64-layer inventory and derives the
  // compact descriptor count before any device allocation or ModelWeights
  // mutation. A layer that is not an exact NVFP4 down projection, or whose
  // raw E4M3 code span cannot fit one base plus six-bit deltas, remains on the
  // canonical route rather than disabling eligible peers.
  std::vector<std::uint8_t> canonical_scales(
      kNvFp4DownCanonicalScaleBytesPerLayer);
  std::vector<NvFp4DownScale6LayerPlan> plans;
  plans.reserve(kQwen36DenseLayerCount);
  (void)cudaGetLastError();
  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    const NvFp4LinearWeight* const down =
        exact_nvfp4_down_projection(model_weights.layer(layer_index));
    if (down == nullptr) {
      continue;
    }
    const cudaError_t status = cudaMemcpy(
        canonical_scales.data(), down->block_scale,
        kNvFp4DownCanonicalScaleBytesPerLayer, cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "cudaMemcpy failed while scanning NVFP4 down scales at layer " +
          std::to_string(layer_index);
      return result;
    }
    unsigned int scale_base = 0U;
    if (derive_nvfp4_down_scale6_base(canonical_scales, scale_base)) {
      plans.push_back({layer_index, down, scale_base});
    }
  }
  result.eligible_layers = plans.size();
  result.fallback_layers = kQwen36DenseLayerCount - plans.size();
  if (plans.empty()) {
    result.fallback_reason = "no_eligible_nvfp4_down_scale6_layers";
    return result;
  }
  if (plans.size() >
      std::numeric_limits<std::size_t>::max() /
          kNvFp4DownScale6SidecarBytesPerProjection) {
    result.hard_failure = true;
    result.message = "NVFP4 down scale6 sidecar byte count overflowed";
    return result;
  }
  const std::size_t arena_bytes =
      plans.size() * kNvFp4DownScale6SidecarBytesPerProjection;

  // Allocate all host staging before the device admission check. A host
  // allocation exception is handled by the engine's existing allocation
  // diagnostic, while an optional CUDA capacity miss remains a canonical
  // fallback with no partially owned device arena.
  std::vector<std::uint8_t> packed(
      kNvFp4DownScale6SidecarBytesPerProjection);
  std::vector<NvFp4DownScale6SidecarDescriptor> descriptors;
  descriptors.reserve(plans.size());

  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed before NVFP4 down scale6 sidecar prepare";
    return result;
  }
  const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
  const std::uint64_t arena_u64 = static_cast<std::uint64_t>(arena_bytes);
  if (arena_u64 > free_u64 ||
      minimum_free_bytes_after_prepare > free_u64 - arena_u64) {
    result.fallback_reason = "insufficient_device_memory_margin";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(&allocation, arena_bytes);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    result.fallback_reason = "cuda_memory_allocation";
    return result;
  }
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMalloc failed while creating NVFP4 down scale6 sidecars";
    return result;
  }
  owner.data = static_cast<std::uint8_t*>(allocation);
  owner.bytes = arena_bytes;
  if ((reinterpret_cast<std::uintptr_t>(owner.data) %
       kNvFp4DownScale6RequiredAlignment) != 0U) {
    result.hard_failure = true;
    result.message =
        "cudaMalloc returned a misaligned NVFP4 down scale6 arena";
    owner.release();
    return result;
  }

  std::size_t remaining_free_bytes = 0U;
  std::size_t remaining_total_bytes = 0U;
  status =
      cudaMemGetInfo(&remaining_free_bytes, &remaining_total_bytes);
  (void)remaining_total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed after NVFP4 down scale6 sidecar allocation";
    owner.release();
    return result;
  }
  if (static_cast<std::uint64_t>(remaining_free_bytes) <
      minimum_free_bytes_after_prepare) {
    result.fallback_reason =
        "insufficient_device_memory_margin_after_allocation";
    owner.release();
    return result;
  }

  // Phase two rereads every eligible canonical tensor, verifies that its
  // immutable code span still matches phase one, packs and independently
  // reconstructs every six-bit code, then copies the complete compact arena.
  // ModelWeights is attached only after every layer succeeds.
  for (std::size_t descriptor_index = 0U;
       descriptor_index < plans.size(); ++descriptor_index) {
    const NvFp4DownScale6LayerPlan& plan = plans[descriptor_index];
    status = cudaMemcpy(canonical_scales.data(), plan.down->block_scale,
                        kNvFp4DownCanonicalScaleBytesPerLayer,
                        cudaMemcpyDeviceToHost);
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "cudaMemcpy failed while packing NVFP4 down scales at layer " +
          std::to_string(plan.layer_index);
      owner.release();
      return result;
    }
    unsigned int verified_base = 0U;
    if (!derive_nvfp4_down_scale6_base(canonical_scales, verified_base) ||
        verified_base != plan.scale_base ||
        !pack_nvfp4_down_scale6_sidecar(
            canonical_scales, plan.scale_base, packed) ||
        !verify_nvfp4_down_scale6_sidecar(
            canonical_scales, plan.scale_base, packed)) {
      result.hard_failure = true;
      result.message =
          "NVFP4 down scale6 pack validation failed at layer " +
          std::to_string(plan.layer_index);
      owner.release();
      return result;
    }
    std::uint8_t* const destination =
        owner.data +
        descriptor_index * kNvFp4DownScale6SidecarBytesPerProjection;
    if ((reinterpret_cast<std::uintptr_t>(destination) %
         kNvFp4DownScale6RequiredAlignment) != 0U) {
      result.hard_failure = true;
      result.message =
          "NVFP4 down scale6 layer destination was misaligned";
      owner.release();
      return result;
    }
    status = cudaMemcpy(destination, packed.data(), packed.size(),
                        cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "cudaMemcpy failed while uploading NVFP4 down scale6 layer " +
          std::to_string(plan.layer_index);
      owner.release();
      return result;
    }
    descriptors.push_back(
        {plan.layer_index, destination,
         kNvFp4DownScale6SidecarBytesPerProjection, plan.scale_base,
         kNvFp4DownScale6Rows, kNvFp4DownScale6Columns});
  }

  owner.descriptors = std::move(descriptors);
  if (!model_weights.attach_nvfp4_down_scale6_sidecars(
          owner.data, owner.bytes, owner.descriptors.data(),
          owner.descriptors.size())) {
    result.hard_failure = true;
    result.message =
        "ModelWeights rejected the complete NVFP4 down scale6 arena";
    owner.release();
    return result;
  }

  result.enabled = true;
  result.bytes = arena_u64;
  return result;
}

[[nodiscard]] Sm87Fp8PrefillQkvPreparation
prepare_sm87_fp8_prefill_qkv_sidecars(
    ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    Sm87Fp8PrefillQkvSidecars& owner) {
  Sm87Fp8PrefillQkvPreparation result;
  if (owner.data != nullptr || owner.bytes != 0U ||
      !owner.descriptors.empty()) {
    result.hard_failure = true;
    result.message =
        "FP8 Prefill QKV sidecar owner was not empty before prepare";
    return result;
  }

  // Inventory the actual linear-attention layer indices instead of assuming
  // that the 48 layers are contiguous. This phase is allocation-free on the
  // device and leaves every ModelWeights sidecar pointer untouched.
  std::vector<Fp8PrefillQkvLayerPlan> plans;
  plans.reserve(kQwen36LinearAttentionLayerCount);
  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    const DecoderLayerWeights& layer = model_weights.layer(layer_index);
    if (!std::holds_alternative<LinearAttentionWeights>(layer.attention)) {
      continue;
    }
    const Fp8LinearWeight* const qkv =
        linear_attention_qkv_projection(layer);
    if (!is_exact_fp8_prefill_qkv_payload(qkv)) {
      result.fallback_reason = "ineligible_model_weights";
      return result;
    }
    plans.push_back({layer_index, qkv});
  }
  if (plans.size() != kQwen36LinearAttentionLayerCount) {
    result.fallback_reason = "ineligible_model_weights";
    return result;
  }

  constexpr std::size_t kArenaBytes =
      kQwen36Fp8PrefillQkvRegisterFeedSidecarBytes;
  static_assert(kArenaBytes == 2'516'582'400ULL);
  static_assert(
      kArenaBytes ==
      kQwen36LinearAttentionLayerCount *
          kFp8PrefillQkvRegisterFeedSidecarBytesPerLayer);

  // This optional derived layout deliberately uses bounded dual residency:
  // the canonical 48-layer QKV payload stays authoritative while the exact
  // same-byte register-feed layout is admitted only when the request safety
  // margin remains available both before and after cudaMalloc.
  (void)cudaGetLastError();
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed before FP8 Prefill QKV sidecar prepare";
    return result;
  }
  const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
  if (kArenaBytes > free_u64 ||
      minimum_free_bytes_after_prepare > free_u64 - kArenaBytes) {
    result.fallback_reason = "insufficient_device_memory_margin";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(&allocation, kArenaBytes);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    result.fallback_reason = "cuda_memory_allocation";
    return result;
  }
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMalloc failed while creating FP8 Prefill QKV sidecars";
    return result;
  }
  owner.data = static_cast<std::uint8_t*>(allocation);
  owner.bytes = kArenaBytes;
  if ((reinterpret_cast<std::uintptr_t>(owner.data) % 16U) != 0U) {
    result.hard_failure = true;
    result.message =
        "cudaMalloc returned a misaligned FP8 Prefill QKV sidecar arena";
    owner.release();
    return result;
  }

  std::size_t remaining_free_bytes = 0U;
  std::size_t remaining_total_bytes = 0U;
  status =
      cudaMemGetInfo(&remaining_free_bytes, &remaining_total_bytes);
  (void)remaining_total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed after FP8 Prefill QKV sidecar allocation";
    owner.release();
    return result;
  }
  if (static_cast<std::uint64_t>(remaining_free_bytes) <
      minimum_free_bytes_after_prepare) {
    result.fallback_reason =
        "insufficient_device_memory_margin_after_allocation";
    owner.release();
    return result;
  }

  std::vector<Fp8PrefillQkvRegisterFeedSidecarDescriptor> descriptors;
  descriptors.reserve(plans.size());
  cudaStream_t stream = nullptr;
  status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaStreamCreateWithFlags failed for FP8 Prefill QKV sidecar "
        "prepare";
    owner.release();
    return result;
  }

  for (std::size_t descriptor_index = 0U;
       descriptor_index < plans.size(); ++descriptor_index) {
    const Fp8PrefillQkvLayerPlan& plan = plans[descriptor_index];
    std::uint8_t* const destination =
        owner.data +
        descriptor_index *
            kFp8PrefillQkvRegisterFeedSidecarBytesPerLayer;
    status = static_cast<cudaError_t>(
        kernels::
            launch_sm87_fp8_w8a16_whole_chunk_qkv_register_feed_pack_cuda(
                plan.qkv->weight, destination, plan.qkv->output_size,
                plan.qkv->input_size, static_cast<void*>(stream)));
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message =
          "FP8 Prefill QKV sidecar pack launch failed at layer " +
          std::to_string(plan.layer_index);
      (void)cudaStreamDestroy(stream);
      owner.release();
      return result;
    }
    descriptors.push_back(
        {plan.layer_index, destination,
         kFp8PrefillQkvRegisterFeedSidecarBytesPerLayer,
         plan.qkv->output_size, plan.qkv->input_size});
  }

  status = cudaStreamSynchronize(stream);
  const cudaError_t destroy_status = cudaStreamDestroy(stream);
  if (status != cudaSuccess || destroy_status != cudaSuccess) {
    const cudaError_t failure =
        status != cudaSuccess ? status : destroy_status;
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(failure);
    result.message =
        "FP8 Prefill QKV sidecar pack stream failed to synchronize or "
        "destroy";
    owner.release();
    return result;
  }

  if (!model_weights.attach_fp8_prefill_qkv_register_feed_sidecars(
          owner.data, owner.bytes, descriptors.data(),
          descriptors.size())) {
    result.hard_failure = true;
    result.message =
        "ModelWeights rejected the complete FP8 Prefill QKV sidecar arena";
    owner.release();
    return result;
  }

  owner.descriptors = std::move(descriptors);
  result.enabled = true;
  result.layers = plans.size();
  result.bytes = kArenaBytes;
  return result;
}

#if defined(Q3X_ENABLE_FP8_PREFILL_SUPERMATRIX_ADMISSION)
[[nodiscard]] Sm87Fp8PrefillSupermatrixPreparation
prepare_sm87_fp8_prefill_supermatrix_sidecars(
    ModelWeights& model_weights,
    const std::uint64_t minimum_free_bytes_after_prepare,
    Sm87Fp8PrefillSupermatrixSidecars& owner) {
  Sm87Fp8PrefillSupermatrixPreparation result;
  if (owner.data != nullptr || owner.bytes != 0U) {
    result.hard_failure = true;
    result.message =
        "FP8 Prefill supermatrix sidecar owner was not empty before prepare";
    return result;
  }

  std::vector<Fp8PrefillSupermatrixProjectionPlan> plans;
  plans.reserve(kFp8PrefillSupermatrixProjectionCount);
  std::size_t planned_bytes = 0U;
  const auto append = [&](const LinearWeight& binding,
                          const std::size_t rows,
                          const std::size_t columns) {
    const auto* const fp8 = std::get_if<Fp8LinearWeight>(&binding);
    if (fp8 == nullptr || fp8->weight == nullptr ||
        fp8->weight_scale_device == nullptr ||
        fp8->input_scale_device == nullptr ||
        !std::isfinite(fp8->weight_scale) || fp8->weight_scale < 0.0F ||
        !std::isfinite(fp8->input_scale) || fp8->input_scale < 0.0F ||
        fp8->output_size != rows || fp8->input_size != columns ||
        fp8->prefill_supermatrix_sidecar != nullptr ||
        (reinterpret_cast<std::uintptr_t>(fp8->weight) % 16U) != 0U ||
        rows > std::numeric_limits<std::size_t>::max() / columns ||
        rows * columns >
            kQwen36Fp8PrefillSupermatrixSidecarBytes - planned_bytes) {
      return false;
    }
    plans.push_back({fp8});
    planned_bytes += rows * columns;
    return true;
  };

  for (std::size_t layer_index = 0U;
       layer_index < kQwen36DenseLayerCount; ++layer_index) {
    const DecoderLayerWeights& layer = model_weights.layer(layer_index);
    if (const auto* const linear =
            std::get_if<LinearAttentionWeights>(&layer.attention)) {
      if (!append(linear->in_proj_qkv, 10'240U, 5'120U) ||
          !append(linear->in_proj_z, 6'144U, 5'120U) ||
          !append(linear->out_proj, 5'120U, 6'144U)) {
        result.hard_failure = true;
        result.message =
            "ineligible linear-attention FP8 supermatrix projection at layer " +
            std::to_string(layer_index);
        return result;
      }
    } else if (const auto* const full =
                   std::get_if<FullAttentionWeights>(&layer.attention)) {
      if (!append(full->q_proj, 12'288U, 5'120U) ||
          !append(full->k_proj, 1'024U, 5'120U) ||
          !append(full->v_proj, 1'024U, 5'120U) ||
          !append(full->o_proj, 5'120U, 6'144U)) {
        result.hard_failure = true;
        result.message =
            "ineligible full-attention FP8 supermatrix projection at layer " +
            std::to_string(layer_index);
        return result;
      }
    } else {
      result.hard_failure = true;
      result.message = "invalid attention variant while inventorying FP8 "
                       "supermatrix projections";
      return result;
    }
  }
  if (plans.size() != kFp8PrefillSupermatrixProjectionCount ||
      planned_bytes != kQwen36Fp8PrefillSupermatrixSidecarBytes) {
    result.hard_failure = true;
    result.message = "FP8 Prefill supermatrix inventory did not cover the "
                     "fixed 208-projection arena";
    return result;
  }

  constexpr std::size_t kArenaBytes =
      kQwen36Fp8PrefillSupermatrixSidecarBytes;
  (void)cudaGetLastError();
  std::size_t free_bytes = 0U;
  std::size_t total_bytes = 0U;
  cudaError_t status = cudaMemGetInfo(&free_bytes, &total_bytes);
  (void)total_bytes;
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMemGetInfo failed before FP8 Prefill supermatrix prepare";
    return result;
  }
  const std::uint64_t free_u64 = static_cast<std::uint64_t>(free_bytes);
  if (kArenaBytes > free_u64 ||
      minimum_free_bytes_after_prepare > free_u64 - kArenaBytes) {
    result.hard_failure = true;
    result.message = "insufficient device-memory margin for the requested "
                     "FP8 Prefill supermatrix admission";
    return result;
  }

  void* allocation = nullptr;
  status = cudaMalloc(&allocation, kArenaBytes);
  if (status != cudaSuccess || allocation == nullptr) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaMalloc failed for the requested FP8 Prefill supermatrix arena";
    return result;
  }
  owner.data = static_cast<std::uint8_t*>(allocation);
  owner.bytes = kArenaBytes;
  if ((reinterpret_cast<std::uintptr_t>(owner.data) % 16U) != 0U) {
    result.hard_failure = true;
    result.message =
        "cudaMalloc returned a misaligned FP8 Prefill supermatrix arena";
    owner.release();
    return result;
  }

  std::size_t remaining_free_bytes = 0U;
  std::size_t remaining_total_bytes = 0U;
  status = cudaMemGetInfo(&remaining_free_bytes, &remaining_total_bytes);
  (void)remaining_total_bytes;
  if (status != cudaSuccess ||
      static_cast<std::uint64_t>(remaining_free_bytes) <
          minimum_free_bytes_after_prepare) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message = status != cudaSuccess
                         ? "cudaMemGetInfo failed after FP8 Prefill "
                           "supermatrix allocation"
                         : "FP8 Prefill supermatrix allocation violated the "
                           "post-create memory margin";
    owner.release();
    return result;
  }

  cudaStream_t stream = nullptr;
  status = cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
  if (status != cudaSuccess) {
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(status);
    result.message =
        "cudaStreamCreateWithFlags failed for FP8 supermatrix packing";
    owner.release();
    return result;
  }

  std::size_t offset = 0U;
  for (std::size_t index = 0U; index < plans.size(); ++index) {
    const Fp8LinearWeight& projection = *plans[index].projection;
    status = static_cast<cudaError_t>(
        kernels::launch_sm87_fp8_prefill_supermatrix_pack_cuda(
            projection.weight, owner.data + offset, projection.output_size,
            projection.input_size, static_cast<void*>(stream)));
    if (status != cudaSuccess) {
      result.hard_failure = true;
      result.cuda_error = static_cast<int>(status);
      result.message = "FP8 supermatrix pack launch failed at projection " +
                       std::to_string(index);
      (void)cudaStreamDestroy(stream);
      owner.release();
      return result;
    }
    offset += projection.output_size * projection.input_size;
  }
  if (offset != kArenaBytes) {
    result.hard_failure = true;
    result.message = "FP8 supermatrix pack offset did not close the arena";
    (void)cudaStreamDestroy(stream);
    owner.release();
    return result;
  }

  status = cudaStreamSynchronize(stream);
  const cudaError_t destroy_status = cudaStreamDestroy(stream);
  if (status != cudaSuccess || destroy_status != cudaSuccess) {
    const cudaError_t failure =
        status != cudaSuccess ? status : destroy_status;
    result.hard_failure = true;
    result.cuda_error = static_cast<int>(failure);
    result.message = "FP8 supermatrix pack stream failed to synchronize or "
                     "destroy";
    owner.release();
    return result;
  }

  if (!model_weights.attach_fp8_prefill_supermatrix_sidecars(
          owner.data, owner.bytes)) {
    result.hard_failure = true;
    result.message =
        "ModelWeights rejected the complete FP8 supermatrix sidecar arena";
    owner.release();
    return result;
  }

  result.enabled = true;
  result.projections = plans.size();
  result.bytes = kArenaBytes;
  return result;
}
#endif

}  // namespace

struct ReferenceEngine::Impl {
  // Declaration order is part of the safety contract. Destruction is exactly
  // runner -> request_state -> model_weights -> Prefill supermatrix/QKV
  // sidecars -> down scale6 sidecars -> output sidecars -> resident_weights
  // -> tokenizer.
  std::unique_ptr<text::Tokenizer> tokenizer;
  std::optional<ResidentWeights> resident_weights;
  Sm87Fp8OutputProjectionSidecars fp8_output_sidecars;
  Sm87NvFp4DownScale6Sidecars nvfp4_down_scale6_sidecars;
  Sm87Fp8PrefillQkvSidecars fp8_prefill_qkv_sidecars;
#if defined(Q3X_ENABLE_FP8_PREFILL_SUPERMATRIX_ADMISSION)
  Sm87Fp8PrefillSupermatrixSidecars fp8_prefill_supermatrix_sidecars;
#endif
  std::optional<ModelWeights> model_weights;
  std::optional<RequestState> request_state;
  std::optional<ReferenceRunner> runner;
  ReferenceEngineLoadStats load;
  bool trace_enabled = false;
  bool decode_graph_cache_ready = false;

  struct BuildResult {
    std::unique_ptr<Impl> value;
    ReferenceEngineDiagnostic diagnostic;

    [[nodiscard]] explicit operator bool() const noexcept {
      return value != nullptr &&
             diagnostic.code == ReferenceEngineError::kNone;
    }
  };

  [[nodiscard]] static BuildResult build(
      const std::filesystem::path& model_directory,
      const ReferenceEngineOptions& options,
      std::unique_ptr<text::Tokenizer> prepared_tokenizer = {},
      const double prepared_tokenizer_milliseconds = 0.0,
      std::optional<ResidentWeights> prepared_resident = std::nullopt,
      const double prepared_resident_milliseconds = 0.0,
      const double prepared_work_wall_milliseconds = 0.0) {
    BuildResult result;
    if (model_directory.empty()) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kInvalidArgument, "model_directory",
          "model directory must not be empty");
      return result;
    }
    if (!is_valid_projection_backend(options.projection_backend)) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kInvalidArgument, "projection_backend",
          "unknown projection backend");
      return result;
    }
    if (!is_valid_reference_decode_graph_cache_policy(
            options.decode_graph_cache_policy)) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kInvalidArgument,
          "decode_graph_cache_policy",
          "unknown Decode Graph cache policy");
      return result;
    }
    const Clock::time_point build_begin = Clock::now();
    if (std::optional<ReferenceEngineDiagnostic> diagnostic =
            sm87_device_diagnostic(options.projection_backend)) {
      result.diagnostic = std::move(*diagnostic);
      return result;
    }

    try {
      const bool tokenizer_was_prepared = prepared_tokenizer != nullptr;
      const bool resident_was_prepared = prepared_resident.has_value();
      auto impl = std::make_unique<Impl>();
      impl->trace_enabled = options.enable_trace;
      impl->load.decode_graph_cache_requested_policy =
          options.decode_graph_cache_policy;
      if (prepared_tokenizer != nullptr) {
        impl->tokenizer = std::move(prepared_tokenizer);
        impl->load.tokenizer_milliseconds =
            prepared_tokenizer_milliseconds;
      } else {
        const Clock::time_point begin = Clock::now();
        const std::filesystem::path tokenizer_path =
            model_directory / "tokenizer.json";
        text::TokenizerLoadResult tokenizer =
            text::Tokenizer::load_file(tokenizer_path.string());
        impl->load.tokenizer_milliseconds = elapsed_milliseconds(begin);
        if (!tokenizer) {
          result.diagnostic =
              tokenizer_diagnostic("tokenizer_load", tokenizer.error);
          return result;
        }
        impl->tokenizer = std::move(tokenizer.tokenizer);
      }

      if (prepared_resident.has_value()) {
        impl->resident_weights.emplace(std::move(*prepared_resident));
        impl->load.resident_load_milliseconds =
            prepared_resident_milliseconds;
        impl->load.resident = impl->resident_weights->stats();
      } else {
        const Clock::time_point begin = Clock::now();
        ResidentLoadResult resident = load_pinned_qwen36_27b(
            model_directory, options.resident_options);
        impl->load.resident_load_milliseconds = elapsed_milliseconds(begin);
        if (!resident) {
          result.diagnostic = resident_diagnostic(resident.diagnostic);
          return result;
        }
        impl->resident_weights.emplace(std::move(*resident.value));
        impl->load.resident = impl->resident_weights->stats();
      }

      {
        const Clock::time_point begin = Clock::now();
        WeightBindResult weights =
            bind_qwen36_27b_weights(*impl->resident_weights);
        impl->load.weight_bind_milliseconds = elapsed_milliseconds(begin);
        if (!weights) {
          result.diagnostic = binding_diagnostic(weights.diagnostic);
          return result;
        }
        impl->model_weights.emplace(std::move(*weights.value));
        impl->load.binding = impl->model_weights->stats();
      }

      {
        const Clock::time_point begin = Clock::now();
        RequestStateResult request =
            create_request_state(options.request_options);
        impl->load.request_state_milliseconds = elapsed_milliseconds(begin);
        if (!request) {
          result.diagnostic = request_diagnostic(request.diagnostic);
          return result;
        }
        impl->request_state.emplace(std::move(*request.value));
        impl->load.request_arena_bytes =
            impl->request_state->arena_bytes();
        impl->load.request_max_sequence_length =
            impl->request_state->max_sequence_length();
        impl->load.request_prefill_chunk_size =
            impl->request_state->plan().prefill_chunk_size;
      }

      if (options.projection_backend ==
          ProjectionBackend::kSm87WeightOnly) {
        const Clock::time_point begin = Clock::now();
        const Sm87Fp8OutputSidecarPreparation preparation =
            prepare_sm87_fp8_output_projection_sidecars(
                *impl->model_weights,
                options.request_options.min_free_bytes_after_create,
                impl->fp8_output_sidecars);
        impl->load.fp8_output_sidecar_milliseconds =
            elapsed_milliseconds(begin);
        if (preparation.hard_failure) {
          result.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              "fp8_output_sidecar_prepare", preparation.message);
          result.diagnostic.cuda_error = preparation.cuda_error;
          return result;
        }
        impl->load.fp8_output_sidecars_enabled = preparation.enabled;
        impl->load.fp8_output_sidecar_layers = preparation.layers;
        impl->load.fp8_output_sidecar_bytes = preparation.bytes;
        impl->load.fp8_output_sidecar_fallback_reason =
            preparation.fallback_reason;

        const Clock::time_point down_scale6_begin = Clock::now();
        const Sm87NvFp4DownScale6Preparation down_scale6_preparation =
            prepare_sm87_nvfp4_down_scale6_sidecars(
                *impl->model_weights,
                options.request_options.min_free_bytes_after_create,
                impl->nvfp4_down_scale6_sidecars);
        impl->load.nvfp4_down_scale6_sidecar_milliseconds =
            elapsed_milliseconds(down_scale6_begin);
        if (down_scale6_preparation.hard_failure) {
          result.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              "nvfp4_down_scale6_sidecar_prepare",
              down_scale6_preparation.message);
          result.diagnostic.cuda_error =
              down_scale6_preparation.cuda_error;
          return result;
        }
        impl->load.nvfp4_down_scale6_sidecars_enabled =
            down_scale6_preparation.enabled;
        impl->load.nvfp4_down_scale6_sidecar_eligible_layers =
            down_scale6_preparation.eligible_layers;
        impl->load.nvfp4_down_scale6_sidecar_fallback_layers =
            down_scale6_preparation.fallback_layers;
        impl->load.nvfp4_down_scale6_sidecar_bytes =
            down_scale6_preparation.bytes;
        impl->load.nvfp4_down_scale6_sidecar_fallback_reason =
            down_scale6_preparation.fallback_reason;

        // The register-feed compute routes are exact-C512-only. The explicit
        // architecture admission replaces (rather than co-allocates with)
        // the legacy QKV-only sidecars so its real P513 direction cell pays
        // for one unambiguous derived layout.
#if defined(Q3X_ENABLE_FP8_PREFILL_SUPERMATRIX_ADMISSION)
        const bool use_fp8_prefill_supermatrix =
            fp8_prefill_supermatrix_admission_requested();
        if (use_fp8_prefill_supermatrix) {
          if (impl->load.request_prefill_chunk_size !=
              kMaximumRequestPrefillChunkSize) {
            result.diagnostic = engine_diagnostic(
                ReferenceEngineError::kRunnerFactoryFailure,
                "fp8_prefill_supermatrix_sidecar_prepare",
                "the requested FP8 Prefill supermatrix admission requires "
                "a C512 engine plan");
            return result;
          }
          const Sm87Fp8PrefillSupermatrixPreparation preparation =
              prepare_sm87_fp8_prefill_supermatrix_sidecars(
                  *impl->model_weights,
                  options.request_options.min_free_bytes_after_create,
                  impl->fp8_prefill_supermatrix_sidecars);
          if (preparation.hard_failure || !preparation.enabled ||
              preparation.projections !=
                  kFp8PrefillSupermatrixProjectionCount ||
              preparation.bytes !=
                  kQwen36Fp8PrefillSupermatrixSidecarBytes) {
            result.diagnostic = engine_diagnostic(
                ReferenceEngineError::kRunnerFactoryFailure,
                "fp8_prefill_supermatrix_sidecar_prepare",
                preparation.message.empty()
                    ? "the requested FP8 Prefill supermatrix admission did "
                      "not publish its complete arena"
                    : preparation.message);
            result.diagnostic.cuda_error = preparation.cuda_error;
            return result;
          }
        } else
#endif
        if (impl->load.request_prefill_chunk_size ==
            kMaximumRequestPrefillChunkSize) {
          const Clock::time_point prefill_qkv_begin = Clock::now();
          const Sm87Fp8PrefillQkvPreparation prefill_qkv_preparation =
              prepare_sm87_fp8_prefill_qkv_sidecars(
                  *impl->model_weights,
                  options.request_options.min_free_bytes_after_create,
                  impl->fp8_prefill_qkv_sidecars);
          impl->load.fp8_prefill_qkv_sidecar_milliseconds =
              elapsed_milliseconds(prefill_qkv_begin);
          if (prefill_qkv_preparation.hard_failure) {
            result.diagnostic = engine_diagnostic(
                ReferenceEngineError::kRunnerFactoryFailure,
                "fp8_prefill_qkv_sidecar_prepare",
                prefill_qkv_preparation.message);
            result.diagnostic.cuda_error =
                prefill_qkv_preparation.cuda_error;
            return result;
          }
          impl->load.fp8_prefill_qkv_sidecars_enabled =
              prefill_qkv_preparation.enabled;
          impl->load.fp8_prefill_qkv_sidecar_layers =
              prefill_qkv_preparation.layers;
          impl->load.fp8_prefill_qkv_sidecar_bytes =
              prefill_qkv_preparation.bytes;
          impl->load.fp8_prefill_qkv_sidecar_fallback_reason =
              prefill_qkv_preparation.fallback_reason;
        }
      }

      {
        ReferenceRunnerOptions runner_options;
        runner_options.enable_trace = options.enable_trace;
        runner_options.projection_backend = options.projection_backend;
        const Clock::time_point begin = Clock::now();
        ReferenceRunnerFactoryResult runner = create_reference_runner(
            &*impl->model_weights, &*impl->request_state, runner_options);
        impl->load.runner_factory_milliseconds = elapsed_milliseconds(begin);
        if (!runner) {
          result.diagnostic = runner_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              "runner_factory", runner.diagnostic);
          return result;
        }
        impl->runner.emplace(std::move(*runner.value));
      }

      const auto rollback_decode_graph_cache =
          [&impl, &result, &options](const std::string_view stage) -> bool {
        const ReferenceRunnerStatus clear_status =
            impl->runner->clear_fixed_position_decode_graph_cache();
        const ReferenceRunnerStatus reset_status = impl->runner->reset();
        if (!clear_status) {
          result.diagnostic = runner_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure, stage,
              clear_status);
          return false;
        }
        if (!reset_status) {
          result.diagnostic = runner_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure, stage,
              reset_status);
          return false;
        }
        if (impl->runner->fixed_position_decode_graph_cache_mask() != 0U) {
          result.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              std::string(stage),
              "cache rollback retained a published graph slot");
          return false;
        }
        std::size_t rollback_free = 0U;
        std::size_t rollback_total = 0U;
        const cudaError_t memory_status =
            cudaMemGetInfo(&rollback_free, &rollback_total);
        if (memory_status != cudaSuccess || rollback_total == 0U) {
          result.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              std::string(stage),
              "could not verify device memory after cache rollback");
          result.diagnostic.cuda_error = static_cast<int>(memory_status);
          return false;
        }
        if (static_cast<std::uint64_t>(rollback_free) <
            options.request_options.min_free_bytes_after_create) {
          result.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerFactoryFailure,
              std::string(stage),
              "cache rollback did not restore the minimum free-memory "
              "reserve");
          return false;
        }
        return true;
      };

      if (options.decode_graph_cache_policy ==
          ReferenceDecodeGraphCachePolicy::kSm87ShortPositions) {
        if (options.projection_backend !=
            ProjectionBackend::kSm87WeightOnly) {
          impl->load.decode_graph_cache_fallback_reason =
              "requires_sm87_projection_backend";
        } else if (impl->request_state->max_sequence_length() <=
                   kProductionDecodeGraphFirstPosition) {
          impl->load.decode_graph_cache_fallback_reason =
              "window_outside_request_capacity";
        } else {
          const std::uint32_t first_position =
              kProductionDecodeGraphFirstPosition;
          const std::uint32_t last_position = std::min(
              kProductionDecodeGraphLastPosition,
              impl->request_state->max_sequence_length() - 1U);
          const std::size_t expected_graph_count =
              static_cast<std::size_t>(last_position - first_position) + 1U;
          const std::uint64_t expected_mask =
              decode_graph_position_mask(first_position, last_position);

          (void)cudaGetLastError();
          std::size_t free_before = 0U;
          std::size_t total_before = 0U;
          const cudaError_t memory_before_status =
              cudaMemGetInfo(&free_before, &total_before);
          if (memory_before_status != cudaSuccess) {
            impl->load.decode_graph_cache_fallback_reason =
                "memory_probe_before_failed";
            if (!rollback_decode_graph_cache(
                    "decode_graph_cache_memory_before_rollback")) {
              return result;
            }
          } else {
            impl->load.decode_graph_cache_free_bytes_before =
                static_cast<std::uint64_t>(free_before);
            const std::uint64_t free_before_bytes =
                static_cast<std::uint64_t>(free_before);
            const std::uint64_t minimum_free_bytes =
                options.request_options.min_free_bytes_after_create;
            if (free_before_bytes < minimum_free_bytes ||
                free_before_bytes - minimum_free_bytes <
                    kProductionDecodeGraphMaximumFreeDropBytes) {
              impl->load.decode_graph_cache_free_bytes_after =
                  static_cast<std::uint64_t>(free_before);
              impl->load.decode_graph_cache_fallback_reason =
                  "cache_memory_budget_unavailable_before_prepare";
              if (!rollback_decode_graph_cache(
                      "decode_graph_cache_admission_rollback")) {
                return result;
              }
            } else {
              const Clock::time_point prepare_begin = Clock::now();
              ReferenceDecodeGraphCachePrepareOutcome prepared =
                  impl->runner->prepare_fixed_position_decode_graph_cache(
                      first_position, last_position,
                      kProductionDecodeGraphCaptureTokenId);
              impl->load.decode_graph_cache_prepare_milliseconds =
                  elapsed_milliseconds(prepare_begin);

              std::size_t free_after = 0U;
              std::size_t total_after = 0U;
              const cudaError_t memory_after_status =
                  cudaMemGetInfo(&free_after, &total_after);
              if (memory_after_status == cudaSuccess) {
                impl->load.decode_graph_cache_free_bytes_after =
                    static_cast<std::uint64_t>(free_after);
                impl->load.decode_graph_cache_free_drop_bytes =
                    free_before >= free_after
                        ? static_cast<std::uint64_t>(free_before - free_after)
                        : 0U;
              }

              bool preparation_exact = prepared.ok();
              if (preparation_exact) {
                preparation_exact =
                    prepared.value->graph_count == expected_graph_count &&
                    prepared.value->prepared_mask == expected_mask &&
                    impl->runner->fixed_position_decode_graph_cache_mask() ==
                        expected_mask;
              }
              if (preparation_exact) {
                for (std::size_t index = 0U;
                     index < expected_graph_count; ++index) {
                  const ReferenceDecodeGraphP1Stats& stats =
                      prepared.value->graphs[index];
                  const std::uint32_t expected_position =
                      first_position + static_cast<std::uint32_t>(index);
                  const bool timing_valid =
                      std::isfinite(stats.capture_enqueue_milliseconds) &&
                      stats.capture_enqueue_milliseconds >= 0.0 &&
                      std::isfinite(
                          stats.topology_inspection_milliseconds) &&
                      stats.topology_inspection_milliseconds >= 0.0 &&
                      std::isfinite(stats.instantiate_milliseconds) &&
                      stats.instantiate_milliseconds >= 0.0 &&
                      std::isfinite(stats.upload_ready_milliseconds) &&
                      stats.upload_ready_milliseconds >= 0.0 &&
                      std::isfinite(stats.total_prepare_milliseconds) &&
                      stats.total_prepare_milliseconds >= 0.0;
                  if (stats.position != expected_position ||
                      stats.input_token_id !=
                          kProductionDecodeGraphCaptureTokenId ||
                      stats.node_count != 390U ||
                      stats.kernel_node_count != 389U ||
                      stats.memcpy_node_count != 1U ||
                      stats.other_node_count != 0U || !timing_valid) {
                    preparation_exact = false;
                    break;
                  }
                  impl->load
                      .decode_graph_cache_capture_enqueue_milliseconds +=
                      stats.capture_enqueue_milliseconds;
                  impl->load
                      .decode_graph_cache_topology_inspection_milliseconds +=
                      stats.topology_inspection_milliseconds;
                  impl->load.decode_graph_cache_instantiate_milliseconds +=
                      stats.instantiate_milliseconds;
                  impl->load.decode_graph_cache_upload_ready_milliseconds +=
                      stats.upload_ready_milliseconds;
                }
              }

              bool cache_admitted = preparation_exact;
              if (!prepared) {
                impl->load.decode_graph_cache_fallback_reason =
                    decode_graph_prepare_fallback_reason(prepared.status);
              } else if (!preparation_exact) {
                impl->load.decode_graph_cache_fallback_reason =
                    "prepared_cache_contract_mismatch";
              } else if (memory_after_status != cudaSuccess ||
                         total_after != total_before) {
                cache_admitted = false;
                impl->load.decode_graph_cache_fallback_reason =
                    "memory_probe_after_failed";
              } else if (!std::isfinite(
                             impl->load
                                 .decode_graph_cache_prepare_milliseconds) ||
                         impl->load.decode_graph_cache_prepare_milliseconds <
                             0.0 ||
                         impl->load.decode_graph_cache_prepare_milliseconds >
                             kProductionDecodeGraphMaximumPrepareMilliseconds) {
                cache_admitted = false;
                impl->load.decode_graph_cache_fallback_reason =
                    "prepare_time_budget_exceeded";
              } else if (
                  impl->load.decode_graph_cache_free_drop_bytes >
                  kProductionDecodeGraphMaximumFreeDropBytes) {
                cache_admitted = false;
                impl->load.decode_graph_cache_fallback_reason =
                    "device_memory_budget_exceeded";
              } else if (
                  static_cast<std::uint64_t>(free_after) <
                  options.request_options.min_free_bytes_after_create) {
                cache_admitted = false;
                impl->load.decode_graph_cache_fallback_reason =
                    "minimum_free_memory_not_preserved";
              }

              if (cache_admitted) {
                const ReferenceRunnerStatus reset_status =
                    impl->runner->reset();
                if (!reset_status) {
                  if (!rollback_decode_graph_cache(
                          "decode_graph_cache_ready_cleanup")) {
                    return result;
                  }
                  result.diagnostic = runner_diagnostic(
                      ReferenceEngineError::kRunnerFactoryFailure,
                      "decode_graph_cache_ready_reset", reset_status);
                  return result;
                }
                if (impl->runner
                            ->fixed_position_decode_graph_cache_mask() !=
                        expected_mask) {
                  if (!rollback_decode_graph_cache(
                          "decode_graph_cache_ready_contract_cleanup")) {
                    return result;
                  }
                  result.diagnostic = engine_diagnostic(
                      ReferenceEngineError::kRunnerFactoryFailure,
                      "decode_graph_cache_ready_contract",
                      "runner reset did not preserve the prepared cache");
                  return result;
                }
                impl->decode_graph_cache_ready = true;
                impl->load.decode_graph_cache_effective_policy =
                    ReferenceDecodeGraphCachePolicy::kSm87ShortPositions;
                impl->load.decode_graph_cache_first_position =
                    first_position;
                impl->load.decode_graph_cache_last_position = last_position;
                impl->load.decode_graph_cache_slot_count =
                    expected_graph_count;
              } else {
                if (!rollback_decode_graph_cache(
                        "decode_graph_cache_prepare_rollback")) {
                  return result;
                }
              }
            }
          }
        }
      }

      double prepared_milliseconds = 0.0;
      if (prepared_work_wall_milliseconds > 0.0) {
        prepared_milliseconds = prepared_work_wall_milliseconds;
      } else {
        prepared_milliseconds =
            (tokenizer_was_prepared ? prepared_tokenizer_milliseconds : 0.0) +
            (resident_was_prepared ? prepared_resident_milliseconds : 0.0);
      }
      impl->load.total_milliseconds =
          elapsed_milliseconds(build_begin) + prepared_milliseconds;
      result.value = std::move(impl);
      return result;
    } catch (const std::bad_alloc&) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kAllocationFailure, "engine_create",
          "host allocation failed while creating the reference engine");
      return result;
    } catch (const std::length_error& error) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kAllocationFailure, "engine_create",
          error.what());
      return result;
    } catch (const std::exception& error) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kInvalidArgument, "engine_create",
          error.what());
      return result;
    }
  }
};

ReferenceEngine::ReferenceEngine(std::unique_ptr<Impl> impl) noexcept
    : impl_(std::move(impl)) {}

ReferenceEngine::~ReferenceEngine() = default;
ReferenceEngine::ReferenceEngine(ReferenceEngine&&) noexcept = default;
ReferenceEngine& ReferenceEngine::operator=(ReferenceEngine&&) noexcept =
    default;

ReferenceEngine::operator bool() const noexcept {
  return impl_ != nullptr && impl_->tokenizer != nullptr &&
         impl_->resident_weights.has_value() &&
         impl_->model_weights.has_value() &&
         impl_->request_state.has_value() && impl_->runner.has_value();
}

const ReferenceEngineLoadStats& ReferenceEngine::load_stats() const noexcept {
  static const ReferenceEngineLoadStats empty;
  return impl_ != nullptr ? impl_->load : empty;
}

std::uint32_t ReferenceEngine::max_sequence_length() const noexcept {
  return impl_ != nullptr && impl_->request_state.has_value()
             ? impl_->request_state->max_sequence_length()
             : 0U;
}

ReferenceGenerateResult ReferenceEngine::generate(
    const std::string_view user_prompt,
    const ReferenceGenerateOptions& options) {
  ReferenceGenerateResult result;
  if (!*this) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "engine",
        "reference engine is empty");
    return result;
  }
  if (user_prompt.empty() || options.max_new_tokens == 0U ||
      options.prefill_chunk_size == 0U ||
      options.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      options.stop_token_id >= kReferenceVocabularySize ||
      !is_valid_reference_logits_mode(options.logits_mode)) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "generation_options",
        "prompt must be non-empty, max_new_tokens must be positive, "
        "prefill_chunk_size must be in [1,512], and stop_token_id must be in "
        "the pinned vocabulary");
    return result;
  }
  if (options.capture_trace && !impl_->trace_enabled) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "generation_options",
        "capture_trace requires an engine created with enable_trace=true");
    return result;
  }
  if (!options.capture_trace &&
      options.prefill_chunk_size >
          impl_->request_state->plan().prefill_chunk_size) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "generation_options",
        "requested prefill chunk exceeds the engine workspace capacity",
        "requested=" + std::to_string(options.prefill_chunk_size) +
            " capacity=" +
            std::to_string(
                impl_->request_state->plan().prefill_chunk_size));
    return result;
  }

  try {
    text::ChatResult chat =
        format_single_user_prompt(*impl_->tokenizer, user_prompt);
    if (!chat) {
      result.diagnostic = tokenizer_diagnostic("chat_encode", chat.error);
      return result;
    }
    if (chat.token_ids.empty()) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kTokenizerFailure, "chat_encode",
          "the rendered chat prompt encoded to zero tokens");
      return result;
    }

    const ReferenceRunnerStatus reset = impl_->runner->reset();
    if (!reset) {
      result.diagnostic = runner_diagnostic(
          ReferenceEngineError::kRunnerResetFailure, "runner_reset", reset);
      return result;
    }

    std::vector<ReferenceTraceDigest> traces;
    EngineStepContext step_context;
    step_context.runner = &*impl_->runner;
    step_context.traces = &traces;
    step_context.capture_trace = options.capture_trace;

    reference_engine_detail::GenerationControlOptions control_options;
    control_options.max_new_tokens = options.max_new_tokens;
    control_options.stop_token_id = options.stop_token_id;
    control_options.max_sequence_length =
        impl_->request_state->max_sequence_length();
    control_options.prefill_chunk_size = options.prefill_chunk_size;
    control_options.capture_trace = options.capture_trace;
    control_options.logits_mode = options.logits_mode;
    control_options.emit_nvtx_phase_ranges = options.emit_nvtx_phase_ranges;

    reference_engine_detail::PrefillPlan prefill_plan;
    prefill_plan.context = &step_context;
    prefill_plan.prefix_step = step_with_trace;
    prefill_plan.finish_prefill = step_with_trace;
    prefill_plan.prefix_tile = prefill_prefix_tile;

    reference_engine_detail::DecodePlan decode_plan;
    decode_plan.context = &step_context;
    const bool use_production_decode_graph_cache =
        impl_->decode_graph_cache_ready && !options.capture_trace &&
        options.logits_mode == ReferenceLogitsMode::kPredictedTokenOnly;
    decode_plan.decode_step =
        (options.use_prepared_decode_graph_cache ||
         use_production_decode_graph_cache)
            ? decode_with_prepared_graph_cache
            : step_with_trace;

    reference_engine_detail::GenerationControlResult control =
        reference_engine_detail::run_generation_control(
            chat.token_ids, control_options, prefill_plan, decode_plan);
    if (!control) {
      result.diagnostic = control_diagnostic(control);
      if (use_production_decode_graph_cache &&
          step_context.decode_graph_replays != 0U) {
        // Never retry the failed token. Retire the engine-lifetime cache so a
        // later request can recover through reset and the serial scheduler
        // instead of repeatedly launching a graph that has already failed.
        const ReferenceRunnerStatus clear_status =
            impl_->runner->clear_fixed_position_decode_graph_cache();
        impl_->decode_graph_cache_ready = false;
        impl_->load.decode_graph_cache_effective_policy =
            ReferenceDecodeGraphCachePolicy::kDisabled;
        impl_->load.decode_graph_cache_first_position = 0U;
        impl_->load.decode_graph_cache_last_position = 0U;
        impl_->load.decode_graph_cache_slot_count = 0U;
        if (!clear_status) {
          impl_->load.decode_graph_cache_fallback_reason =
              "runtime_graph_failure_demote_cleanup_failed";
          if (!result.diagnostic.context.empty()) {
            result.diagnostic.context += "; ";
          }
          result.diagnostic.context +=
              "decode_graph_cache_runtime_demote_failed:";
          result.diagnostic.context +=
              reference_runner_error_string(clear_status.error);
          if (clear_status.operation != nullptr) {
            result.diagnostic.context += ':';
            result.diagnostic.context += clear_status.operation;
          }
        } else {
          impl_->load.decode_graph_cache_fallback_reason =
              "runtime_graph_failure_demoted_to_serial";
        }
      }
      return result;
    }

    const std::size_t text_token_count =
        reference_engine_detail::generated_text_token_count(
            control.value->generated_token_ids,
            control.value->stop_reason, options.stop_token_id);
    const auto text_end = control.value->generated_token_ids.begin() +
                          static_cast<std::ptrdiff_t>(text_token_count);
    const std::vector<std::uint32_t> text_token_ids(
        control.value->generated_token_ids.begin(), text_end);
    text::DecodeOptions decode_options;
    // The exact generated-id sequence is authoritative. Hide only the stop
    // token explicitly removed above; do not silently discard any other
    // generated special id under a max-token termination.
    decode_options.skip_special_tokens = false;
    text::DecodeResult decoded =
        impl_->tokenizer->decode(text_token_ids, decode_options);
    if (!decoded) {
      result.diagnostic = tokenizer_diagnostic("generated_decode",
                                               decoded.error);
      result.diagnostic.code = ReferenceEngineError::kDecodeFailure;
      return result;
    }

    ReferenceGeneration generation;
    generation.rendered_prompt = std::move(chat.rendered);
    generation.prompt_token_ids = std::move(chat.token_ids);
    generation.generated_token_ids =
        std::move(control.value->generated_token_ids);
    generation.generated_text = std::move(decoded.text);
    generation.stop_reason = control.value->stop_reason;
    generation.requested_prefill_chunk_size = options.prefill_chunk_size;
    generation.effective_prefill_chunk_size =
        options.capture_trace ? kDefaultRequestPrefillChunkSize
                              : options.prefill_chunk_size;
    generation.timing = std::move(control.value->timing);
    generation.steps = std::move(control.value->steps);
    generation.traces = std::move(traces);
    generation.decode_graph_replays = step_context.decode_graph_replays;
    generation.decode_graph_serial_fallbacks =
        step_context.decode_graph_serial_fallbacks;
    result.value.emplace(std::move(generation));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure, "generate",
        "host allocation failed during generation");
    return result;
  } catch (const std::length_error& error) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure, "generate", error.what());
    return result;
  } catch (const std::exception& error) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kTokenizerFailure, "generate", error.what());
    return result;
  }
}

ReferenceDecodeGraphP1ScreenOutcome
ReferenceEngine::screen_fixed_position_decode_graph_p1(
    const std::string_view user_prompt,
    const ReferenceDecodeGraphP1ScreenOptions& options) {
  ReferenceDecodeGraphP1ScreenOutcome outcome;
  if (!*this) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "decode_graph_p1_engine",
        "reference engine is empty");
    return outcome;
  }
  if (user_prompt.empty() || options.expected_position == 0U ||
      options.expected_position >= max_sequence_length() ||
      options.expected_input_token_id >= kReferenceVocabularySize ||
      options.expected_prediction >= kReferenceVocabularySize ||
      options.alternate_input_token_id >= kReferenceVocabularySize ||
      options.expected_alternate_prediction >= kReferenceVocabularySize ||
      options.expected_input_token_id == options.alternate_input_token_id ||
      options.prefill_chunk_size == 0U ||
      options.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      options.prefill_chunk_size >
          impl_->request_state->plan().prefill_chunk_size) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument,
        "decode_graph_p1_options",
        "prompt, fixed position, distinct token ids, and prefill chunk must "
        "fit the owned engine capacities");
    return outcome;
  }

  try {
    ReferenceGenerateOptions generation_options;
    generation_options.max_new_tokens = 1U;
    generation_options.capture_trace = false;
    generation_options.prefill_chunk_size = options.prefill_chunk_size;
    generation_options.logits_mode =
        ReferenceLogitsMode::kPredictedTokenOnly;
    ReferenceGenerateResult seeded = generate(user_prompt, generation_options);
    if (!seeded) {
      outcome.diagnostic = std::move(seeded.diagnostic);
      if (outcome.diagnostic.stage.empty()) {
        outcome.diagnostic.stage = "decode_graph_p1_seed";
      } else {
        outcome.diagnostic.stage =
            "decode_graph_p1_seed/" + outcome.diagnostic.stage;
      }
      return outcome;
    }
    if (seeded.value->generated_token_ids.size() != 1U ||
        seeded.value->generated_token_ids.front() !=
            options.expected_input_token_id ||
        impl_->request_state->current_position() !=
            options.expected_position ||
        impl_->runner->current_position() != options.expected_position) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p1_seed",
          "the oracle prompt did not produce the fixed P1 boundary",
          "generated_count=" +
              std::to_string(seeded.value->generated_token_ids.size()) +
              " generated_token=" +
              (seeded.value->generated_token_ids.empty()
                   ? std::string("none")
                   : std::to_string(
                         seeded.value->generated_token_ids.front())) +
              " position=" +
              std::to_string(impl_->request_state->current_position()));
      return outcome;
    }

    RequestState& state = *impl_->request_state;
    ReferenceRunner& runner = *impl_->runner;
    const std::uint64_t arena_bytes_u64 = state.arena_bytes();
    if (state.arena_data() == nullptr || arena_bytes_u64 == 0U ||
        arena_bytes_u64 >
            static_cast<std::uint64_t>(
                std::numeric_limits<std::size_t>::max())) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kArithmeticOverflow,
          "decode_graph_p1_arena",
          "request arena cannot be represented by the screening harness");
      return outcome;
    }
    const std::size_t arena_bytes =
        static_cast<std::size_t>(arena_bytes_u64);
    void* const arena = state.arena_data();

    auto set_cuda_diagnostic = [&outcome](
                                   const ReferenceEngineError code,
                                   const std::string_view stage,
                                   const std::string_view message,
                                   const cudaError_t status) {
      outcome.diagnostic = engine_diagnostic(
          code, std::string(stage), std::string(message),
          cudaGetErrorString(status));
      outcome.diagnostic.cuda_error = static_cast<int>(status);
    };

    DecodeGraphP1DeviceBuffer baseline;
    DecodeGraphP1DeviceBuffer serial_after;
    cudaError_t cuda_status = baseline.allocate(arena_bytes);
    if (cuda_status != cudaSuccess) {
      set_cuda_diagnostic(ReferenceEngineError::kAllocationFailure,
                          "decode_graph_p1_baseline_allocate",
                          "failed to allocate the device arena snapshot",
                          cuda_status);
      return outcome;
    }
    cuda_status = serial_after.allocate(arena_bytes);
    if (cuda_status != cudaSuccess) {
      set_cuda_diagnostic(
          ReferenceEngineError::kAllocationFailure,
          "decode_graph_p1_serial_allocate",
          "failed to allocate the serial-result device snapshot",
          cuda_status);
      return outcome;
    }
    cuda_status = cudaMemcpy(baseline.data, arena, arena_bytes,
                             cudaMemcpyDeviceToDevice);
    if (cuda_status != cudaSuccess) {
      set_cuda_diagnostic(ReferenceEngineError::kRunnerStepFailure,
                          "decode_graph_p1_baseline_copy",
                          "failed to snapshot the complete request arena",
                          cuda_status);
      return outcome;
    }
    cuda_status = cudaDeviceSynchronize();
    if (cuda_status != cudaSuccess) {
      set_cuda_diagnostic(ReferenceEngineError::kRunnerStepFailure,
                          "decode_graph_p1_baseline_synchronize",
                          "failed to finish the device arena snapshot",
                          cuda_status);
      return outcome;
    }

    DecodeGraphP1StateRestoreGuard restore_guard{
        &state, arena, baseline.data, arena_bytes,
        options.expected_position, true};

    auto restore = [&]() -> bool {
      const cudaError_t copy_status =
          cudaMemcpy(arena, baseline.data, arena_bytes,
                     cudaMemcpyDeviceToDevice);
      if (copy_status != cudaSuccess) {
        set_cuda_diagnostic(ReferenceEngineError::kRunnerStepFailure,
                            "decode_graph_p1_restore",
                            "failed to restore the complete request arena",
                            copy_status);
        return false;
      }
      const cudaError_t synchronize_status = cudaDeviceSynchronize();
      if (synchronize_status != cudaSuccess) {
        set_cuda_diagnostic(ReferenceEngineError::kRunnerStepFailure,
                            "decode_graph_p1_restore_synchronize",
                            "failed to finish restoring the request arena",
                            synchronize_status);
        return false;
      }
      const RequestOperationStatus length_status =
          state.set_sequence_length(options.expected_position);
      if (!length_status) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kRequestStateFailure,
            "decode_graph_p1_restore_position",
            "failed to restore the fixed logical sequence position");
        outcome.diagnostic.dependency_error =
            static_cast<int>(length_status.error);
        outcome.diagnostic.cuda_error = length_status.cuda_error;
        return false;
      }
      return true;
    };

    ReferenceDecodeGraphP1PrepareOutcome prepared =
        runner.prepare_fixed_position_decode_graph_p1(
            options.expected_input_token_id);
    if (!prepared) {
      outcome.diagnostic = runner_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p1_prepare", prepared.status);
      return outcome;
    }

    ReferenceDecodeGraphP1ScreenResult result;
    result.graph = *prepared.value;
    result.compared_arena_bytes = arena_bytes_u64;

    auto run_step = [&](const bool graph, const std::uint32_t input_token_id,
                        const bool measure_timing,
                        std::uint32_t* const prediction,
                        double* const elapsed) -> bool {
      ReferenceStepOutcome step;
      if (graph) {
        step = runner.replay_fixed_position_decode_graph_p1(
            input_token_id, measure_timing);
      } else {
        ReferenceStepOptions step_options;
        step_options.compute_logits = true;
        step_options.capture_trace = false;
        step_options.measure_timing = measure_timing;
        step_options.logits_mode =
            ReferenceLogitsMode::kPredictedTokenOnly;
        step = runner.step(input_token_id, step_options);
      }
      const std::string stage = graph ? "decode_graph_p1_replay"
                                      : "decode_graph_p1_serial";
      if (!step) {
        outcome.diagnostic = runner_diagnostic(
            ReferenceEngineError::kRunnerStepFailure, stage, step.status);
        return false;
      }
      if (step.value->position != options.expected_position ||
          step.value->input_token_id != input_token_id ||
          step.value->logits.has_value() ||
          !step.value->prediction.has_value()) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kMissingPrediction, stage,
            "predicted-only step returned an invalid result shape");
        return false;
      }
      if (measure_timing != step.value->timing.has_value()) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kMissingTiming, stage,
            "step timing presence did not match the request");
        return false;
      }
      if (prediction != nullptr) {
        *prediction = step.value->prediction->predicted_token_id;
      }
      if (elapsed != nullptr) {
        const double milliseconds =
            step.value->timing->elapsed_milliseconds;
        if (!std::isfinite(milliseconds) || milliseconds < 0.0) {
          outcome.diagnostic = engine_diagnostic(
              ReferenceEngineError::kMissingTiming, stage,
              "step returned a non-finite or negative timing");
          return false;
        }
        *elapsed = milliseconds;
      }
      return true;
    };

    auto capture_serial_arena = [&]() -> bool {
      const cudaError_t status =
          cudaMemcpy(serial_after.data, arena, arena_bytes,
                     cudaMemcpyDeviceToDevice);
      if (status != cudaSuccess) {
        set_cuda_diagnostic(
            ReferenceEngineError::kRunnerStepFailure,
            "decode_graph_p1_serial_snapshot",
            "failed to snapshot the serial request arena", status);
        return false;
      }
      const cudaError_t synchronize_status = cudaDeviceSynchronize();
      if (synchronize_status != cudaSuccess) {
        set_cuda_diagnostic(
            ReferenceEngineError::kRunnerStepFailure,
            "decode_graph_p1_serial_snapshot_synchronize",
            "failed to finish the serial request arena snapshot",
            synchronize_status);
        return false;
      }
      return true;
    };

    auto compare_graph_arena = [&](bool* const exact) -> bool {
      constexpr std::size_t kCompareChunkBytes = 8U * 1024U * 1024U;
      const std::size_t host_bytes =
          std::min(arena_bytes, kCompareChunkBytes);
      std::vector<std::uint8_t> serial_chunk(host_bytes);
      std::vector<std::uint8_t> graph_chunk(host_bytes);
      *exact = true;
      const auto* const expected =
          static_cast<const std::uint8_t*>(serial_after.data);
      const auto* const actual = static_cast<const std::uint8_t*>(arena);
      for (std::size_t offset = 0U; offset < arena_bytes;
           offset += host_bytes) {
        const std::size_t bytes =
            std::min(host_bytes, arena_bytes - offset);
        cudaError_t status =
            cudaMemcpy(serial_chunk.data(), expected + offset, bytes,
                       cudaMemcpyDeviceToHost);
        if (status == cudaSuccess) {
          status = cudaMemcpy(graph_chunk.data(), actual + offset, bytes,
                              cudaMemcpyDeviceToHost);
        }
        if (status != cudaSuccess) {
          set_cuda_diagnostic(
              ReferenceEngineError::kRunnerStepFailure,
              "decode_graph_p1_arena_compare",
              "failed to read the complete arena for exact comparison",
              status);
          return false;
        }
        if (std::memcmp(serial_chunk.data(), graph_chunk.data(), bytes) !=
            0) {
          *exact = false;
        }
      }
      return true;
    };

    if (!restore() ||
        !run_step(false, options.expected_input_token_id, false,
                  &result.serial_prediction, nullptr) ||
        !capture_serial_arena() || !restore() ||
        !run_step(true, options.expected_input_token_id, false,
                  &result.graph_prediction, nullptr) ||
        !compare_graph_arena(&result.primary_arena_exact)) {
      return outcome;
    }
    if (result.serial_prediction != options.expected_prediction ||
        result.graph_prediction != options.expected_prediction ||
        !result.primary_arena_exact) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p1_primary_correctness",
          "primary serial and graph results must match the pinned oracle and "
          "complete request arena",
          "serial=" + std::to_string(result.serial_prediction) +
              " graph=" + std::to_string(result.graph_prediction) +
              " expected=" +
              std::to_string(options.expected_prediction) +
              " arena_exact=" +
              std::to_string(result.primary_arena_exact));
      return outcome;
    }

    if (!restore() ||
        !run_step(false, options.alternate_input_token_id, false,
                  &result.alternate_serial_prediction, nullptr) ||
        !capture_serial_arena() || !restore() ||
        !run_step(true, options.alternate_input_token_id, false,
                  &result.alternate_graph_prediction, nullptr) ||
        !compare_graph_arena(&result.alternate_arena_exact)) {
      return outcome;
    }
    if (result.alternate_serial_prediction !=
            options.expected_alternate_prediction ||
        result.alternate_graph_prediction !=
            options.expected_alternate_prediction ||
        !result.alternate_arena_exact) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p1_alternate_correctness",
          "alternate serial and graph results must match the pinned oracle "
          "and complete request arena",
          "serial=" +
              std::to_string(result.alternate_serial_prediction) +
              " graph=" +
              std::to_string(result.alternate_graph_prediction) +
              " expected=" +
              std::to_string(options.expected_alternate_prediction) +
              " arena_exact=" +
              std::to_string(result.alternate_arena_exact));
      return outcome;
    }

    // One unrecorded B-C-C-B quartet warms the same alternating paths used by
    // the measured rounds. Every sample starts from the full baseline arena.
    for (const bool graph : {false, true, true, false}) {
      std::uint32_t prediction = 0U;
      if (!restore() ||
          !run_step(graph, options.expected_input_token_id, false,
                    &prediction, nullptr)) {
        return outcome;
      }
      if (prediction != result.serial_prediction) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kRunnerStepFailure,
            "decode_graph_p1_warmup_prediction",
            "warmup prediction was not deterministic");
        return outcome;
      }
    }

    for (ReferenceDecodeGraphP1RoundTiming& round : result.rounds) {
      std::uint32_t prediction = 0U;
      if (!restore() ||
          !run_step(false, options.expected_input_token_id, true,
                    &prediction, &round.serial_first_milliseconds) ||
          prediction != result.serial_prediction || !restore() ||
          !run_step(true, options.expected_input_token_id, true,
                    &prediction, &round.graph_first_milliseconds) ||
          prediction != result.serial_prediction || !restore() ||
          !run_step(true, options.expected_input_token_id, true,
                    &prediction, &round.graph_second_milliseconds) ||
          prediction != result.serial_prediction || !restore() ||
          !run_step(false, options.expected_input_token_id, true,
                    &prediction, &round.serial_second_milliseconds) ||
          prediction != result.serial_prediction) {
        if (outcome.diagnostic.code == ReferenceEngineError::kNone) {
          outcome.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerStepFailure,
              "decode_graph_p1_timing_prediction",
              "timed prediction was not deterministic");
        }
        return outcome;
      }
    }

    if (!restore()) {
      return outcome;
    }
    restore_guard.armed = false;
    outcome.value.emplace(std::move(result));
    return outcome;
  } catch (const std::bad_alloc&) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure,
        "decode_graph_p1_screen",
        "host allocation failed during the decode graph screen");
    return outcome;
  } catch (const std::length_error& error) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure,
        "decode_graph_p1_screen", error.what());
    return outcome;
  } catch (const std::exception& error) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kRunnerStepFailure,
        "decode_graph_p1_screen", error.what());
    return outcome;
  }
}

ReferenceDecodeGraphP2ScreenOutcome
ReferenceEngine::screen_short_decode_graph_cache_p2(
    const std::string_view user_prompt,
    const ReferenceDecodeGraphP2ScreenOptions& options) {
  ReferenceDecodeGraphP2ScreenOutcome outcome;
  if (!*this) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument,
        "decode_graph_p2_engine", "reference engine is empty");
    return outcome;
  }
  const bool continuous_range_exact =
      options.first_decode_position <= options.last_decode_position &&
      static_cast<std::uint64_t>(options.last_decode_position) -
                  options.first_decode_position +
              1U ==
          kReferenceDecodeGraphP2ContinuousSteps;
  if (user_prompt.empty() || !continuous_range_exact ||
      options.last_decode_position >=
          kReferenceDecodeGraphP2MaximumSlots ||
      options.boundary_graph_position + 1U !=
          kReferenceDecodeGraphP2MaximumSlots ||
      options.capture_input_token_id >= kReferenceVocabularySize ||
      options.boundary_input_token_id >= kReferenceVocabularySize ||
      options.max_new_tokens !=
          kReferenceDecodeGraphP2ContinuousSteps + 1U ||
      options.prefill_chunk_size == 0U ||
      options.prefill_chunk_size > kMaximumRequestPrefillChunkSize ||
      options.prefill_chunk_size >
          impl_->request_state->plan().prefill_chunk_size ||
      max_sequence_length() < options.boundary_graph_position + 2U ||
      !impl_->trace_enabled) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument,
        "decode_graph_p2_options",
        "P2 requires a 25-position range below P64, a P63 boundary, "
        "max_new_tokens=26, trace reservation, and request capacity through "
        "P64");
    return outcome;
  }

  try {
    RequestState& state = *impl_->request_state;
    ReferenceRunner& runner = *impl_->runner;
    const ReferenceRunnerStatus reset_status = runner.reset();
    if (!reset_status) {
      outcome.diagnostic = runner_diagnostic(
          ReferenceEngineError::kRunnerResetFailure,
          "decode_graph_p2_initial_reset", reset_status);
      return outcome;
    }

    const std::uint64_t arena_bytes_u64 = state.arena_bytes();
    if (state.arena_data() == nullptr || arena_bytes_u64 == 0U ||
        arena_bytes_u64 > static_cast<std::uint64_t>(
                              std::numeric_limits<std::size_t>::max())) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kArithmeticOverflow,
          "decode_graph_p2_arena",
          "request arena cannot be represented by the P2 screen");
      return outcome;
    }
    const std::size_t arena_bytes =
        static_cast<std::size_t>(arena_bytes_u64);
    void* const arena = state.arena_data();

    auto set_cuda_diagnostic = [&outcome](
                                   const ReferenceEngineError code,
                                   const std::string_view stage,
                                   const std::string_view message,
                                   const cudaError_t status) {
      outcome.diagnostic = engine_diagnostic(
          code, std::string(stage), std::string(message),
          cudaGetErrorString(status));
      outcome.diagnostic.cuda_error = static_cast<int>(status);
    };

    DecodeGraphP1DeviceBuffer baseline;
    DecodeGraphP1DeviceBuffer serial_after;
    DecodeGraphP1DeviceBuffer boundary_baseline;
    for (auto* const allocation :
         {&baseline, &serial_after, &boundary_baseline}) {
      const cudaError_t status = allocation->allocate(arena_bytes);
      if (status != cudaSuccess) {
        set_cuda_diagnostic(
            ReferenceEngineError::kAllocationFailure,
            "decode_graph_p2_snapshot_allocate",
            "failed to allocate a complete device arena snapshot", status);
        return outcome;
      }
    }

    constexpr std::size_t kCompareChunkBytes = 8U * 1024U * 1024U;
    const std::size_t host_compare_bytes =
        std::min(arena_bytes, kCompareChunkBytes);
    std::vector<std::uint8_t> expected_chunk(host_compare_bytes);
    std::vector<std::uint8_t> actual_chunk(host_compare_bytes);

    auto copy_arena_to = [&](void* const destination,
                             const std::string_view stage) -> bool {
      cudaError_t status = cudaMemcpy(destination, arena, arena_bytes,
                                      cudaMemcpyDeviceToDevice);
      if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
      }
      if (status != cudaSuccess) {
        set_cuda_diagnostic(
            ReferenceEngineError::kRunnerStepFailure, stage,
            "failed to snapshot and synchronize the complete request arena",
            status);
        return false;
      }
      return true;
    };

    if (!copy_arena_to(baseline.data,
                       "decode_graph_p2_baseline_snapshot")) {
      return outcome;
    }

    DecodeGraphP1StateRestoreGuard restore_guard{
        &state, arena, baseline.data, arena_bytes, 0U, true};

    auto set_position = [&](const std::uint32_t position,
                            const std::string_view stage) -> bool {
      const RequestOperationStatus status =
          state.set_sequence_length(position);
      if (!status) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kRequestStateFailure,
            std::string(stage), "failed to set the logical sequence length",
            "position=" + std::to_string(position));
        outcome.diagnostic.dependency_error =
            static_cast<int>(status.error);
        outcome.diagnostic.cuda_error = status.cuda_error;
        return false;
      }
      return true;
    };

    auto restore_from = [&](const void* const snapshot,
                            const std::uint32_t position,
                            const std::string_view stage) -> bool {
      cudaError_t status = cudaMemcpy(arena, snapshot, arena_bytes,
                                      cudaMemcpyDeviceToDevice);
      if (status == cudaSuccess) {
        status = cudaDeviceSynchronize();
      }
      if (status != cudaSuccess) {
        set_cuda_diagnostic(
            ReferenceEngineError::kRunnerStepFailure, stage,
            "failed to restore and synchronize the complete request arena",
            status);
        return false;
      }
      return set_position(position, stage);
    };

    auto compare_arenas = [&](const void* const expected,
                              const void* const actual,
                              bool* const exact,
                              const std::string_view stage) -> bool {
      *exact = true;
      const auto* const expected_bytes =
          static_cast<const std::uint8_t*>(expected);
      const auto* const actual_bytes =
          static_cast<const std::uint8_t*>(actual);
      for (std::size_t offset = 0U; offset < arena_bytes;
           offset += host_compare_bytes) {
        const std::size_t bytes =
            std::min(host_compare_bytes, arena_bytes - offset);
        cudaError_t status = cudaMemcpy(
            expected_chunk.data(), expected_bytes + offset, bytes,
            cudaMemcpyDeviceToHost);
        if (status == cudaSuccess) {
          status = cudaMemcpy(actual_chunk.data(), actual_bytes + offset,
                              bytes, cudaMemcpyDeviceToHost);
        }
        if (status != cudaSuccess) {
          set_cuda_diagnostic(
              ReferenceEngineError::kRunnerStepFailure, stage,
              "failed to read complete arenas for exact comparison", status);
          return false;
        }
        if (std::memcmp(expected_chunk.data(), actual_chunk.data(), bytes) !=
            0) {
          *exact = false;
        }
      }
      return true;
    };

    auto run_generation = [&](const std::string_view prompt,
                              const std::uint32_t max_new_tokens,
                              const bool graph,
                              const ReferenceLogitsMode logits_mode,
                              const bool capture_trace,
                              const std::string_view stage,
                              ReferenceGeneration* const generation) -> bool {
      ReferenceGenerateOptions generate_options;
      generate_options.max_new_tokens = max_new_tokens;
      generate_options.capture_trace = capture_trace;
      generate_options.prefill_chunk_size = options.prefill_chunk_size;
      generate_options.logits_mode = logits_mode;
      generate_options.use_prepared_decode_graph_cache = graph;
      ReferenceGenerateResult generated = generate(prompt, generate_options);
      if (!generated) {
        outcome.diagnostic = std::move(generated.diagnostic);
        outcome.diagnostic.stage =
            std::string(stage) + "/" + outcome.diagnostic.stage;
        return false;
      }
      *generation = std::move(*generated.value);
      return true;
    };

    auto valid_continuous_shape = [&](
                                      const ReferenceGeneration& generation)
        -> bool {
      return generation.prompt_token_ids.size() ==
                 options.first_decode_position &&
             generation.generated_token_ids.size() ==
                 options.max_new_tokens &&
             generation.steps.size() ==
                 generation.prompt_token_ids.size() +
                     generation.generated_token_ids.size() - 1U &&
             generation.timing.subsequent_token_milliseconds.size() ==
                 kReferenceDecodeGraphP2ContinuousSteps;
    };

    ReferenceDecodeGraphP2ScreenResult result;
    result.compared_arena_bytes = arena_bytes_u64;

    // Establish one serial oracle and warm every production kernel/module
    // before taking cache resource readings.
    if (!restore_from(baseline.data, 0U,
                      "decode_graph_p2_serial_restore") ||
        !run_generation(user_prompt, options.max_new_tokens, false,
                        ReferenceLogitsMode::kPredictedTokenOnly, false,
                        "decode_graph_p2_serial_generation",
                        &result.serial_generation) ||
        !valid_continuous_shape(result.serial_generation) ||
        result.serial_generation.decode_graph_replays != 0U ||
        result.serial_generation.decode_graph_serial_fallbacks != 0U ||
        state.current_position() != options.last_decode_position + 1U ||
        !copy_arena_to(serial_after.data,
                       "decode_graph_p2_serial_result_snapshot")) {
      if (outcome.diagnostic.code == ReferenceEngineError::kNone) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kRunnerStepFailure,
            "decode_graph_p2_serial_shape",
            "serial oracle did not produce the exact continuous P2 shape");
      }
      return outcome;
    }

    // Before preparation, the opt-in dispatch must cleanly choose serial for
    // every Decode position and reproduce the complete serial result.
    ReferenceGeneration cache_miss_generation;
    bool cache_miss_arena_exact = false;
    if (!restore_from(baseline.data, 0U,
                      "decode_graph_p2_cache_miss_restore") ||
        !run_generation(user_prompt, options.max_new_tokens, true,
                        ReferenceLogitsMode::kPredictedTokenOnly, false,
                        "decode_graph_p2_cache_miss_generation",
                        &cache_miss_generation) ||
        !compare_arenas(serial_after.data, arena, &cache_miss_arena_exact,
                        "decode_graph_p2_cache_miss_arena")) {
      return outcome;
    }
    result.cache_miss_fallback_exact =
        same_generation_semantics(result.serial_generation,
                                  cache_miss_generation) &&
        cache_miss_generation.decode_graph_replays == 0U &&
        cache_miss_generation.decode_graph_serial_fallbacks ==
            kReferenceDecodeGraphP2ContinuousSteps &&
        cache_miss_arena_exact;
    if (!result.cache_miss_fallback_exact) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_cache_miss_fallback",
          "empty-cache opt-in did not reproduce serial generation exactly");
      return outcome;
    }

    if (!restore_from(baseline.data, 0U,
                      "decode_graph_p2_prepare_restore")) {
      return outcome;
    }
    for (std::uint32_t position = options.first_decode_position;
         position <= options.last_decode_position; ++position) {
      if (runner.has_fixed_position_decode_graph_p1(position)) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kInvalidArgument,
            "decode_graph_p2_preexisting_cache",
            "P2 screen requires an empty continuous cache");
        return outcome;
      }
    }
    if (runner.has_fixed_position_decode_graph_p1(
            options.boundary_graph_position)) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kInvalidArgument,
          "decode_graph_p2_preexisting_boundary_cache",
          "P2 screen requires an empty boundary cache slot");
      return outcome;
    }

    std::size_t free_before = 0U;
    std::size_t total_before = 0U;
    cudaError_t cuda_status =
        cudaMemGetInfo(&free_before, &total_before);
    if (cuda_status != cudaSuccess) {
      set_cuda_diagnostic(ReferenceEngineError::kRunnerStepFailure,
                          "decode_graph_p2_memory_before",
                          "failed to read CUDA free memory before cache prep",
                          cuda_status);
      return outcome;
    }
    const std::optional<std::uint64_t> host_private_before =
        process_private_memory_bytes();
    const Clock::time_point prepare_begin = Clock::now();
    bool topology_exact = true;
    for (std::size_t index = 0U;
         index < kReferenceDecodeGraphP2ContinuousSteps; ++index) {
      const std::uint32_t position =
          options.first_decode_position +
          static_cast<std::uint32_t>(index);
      if (!set_position(position, "decode_graph_p2_prepare_position")) {
        return outcome;
      }
      ReferenceDecodeGraphP1PrepareOutcome prepared =
          runner.prepare_fixed_position_decode_graph_p1(
              options.capture_input_token_id);
      if (!prepared) {
        outcome.diagnostic = runner_diagnostic(
            ReferenceEngineError::kRunnerStepFailure,
            "decode_graph_p2_prepare_slot", prepared.status);
        return outcome;
      }
      if (state.current_position() != position) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kRunnerStepFailure,
            "decode_graph_p2_prepare_slot_position",
            "graph preparation changed the logical sequence length",
            "expected=" + std::to_string(position) + " actual=" +
                std::to_string(state.current_position()));
        return outcome;
      }
      result.continuous_graphs[index] = *prepared.value;
      const ReferenceDecodeGraphP1Stats& stats =
          result.continuous_graphs[index];
      topology_exact =
          topology_exact && stats.position == position &&
          stats.input_token_id == options.capture_input_token_id &&
          stats.node_count == 390U && stats.kernel_node_count == 389U &&
          stats.memcpy_node_count == 1U && stats.other_node_count == 0U &&
          runner.has_fixed_position_decode_graph_p1(position);
    }
    if (!set_position(0U, "decode_graph_p2_prepare_restore_position")) {
      return outcome;
    }
    // The cold-time and memory gates describe the production P19-P43 cache.
    // P63 is prepared separately below only to validate the P63/P64 boundary;
    // it is not part of the selected short-position cache window.
    result.cache_prepare_milliseconds =
        elapsed_milliseconds(prepare_begin);

    std::size_t free_after = 0U;
    std::size_t total_after = 0U;
    cuda_status = cudaMemGetInfo(&free_after, &total_after);
    if (cuda_status != cudaSuccess || total_before != total_after) {
      set_cuda_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_memory_after",
          "failed to read a stable CUDA memory total after cache prep",
          cuda_status == cudaSuccess ? cudaErrorUnknown : cuda_status);
      return outcome;
    }
    result.cache_free_bytes_before = free_before;
    result.cache_free_bytes_after = free_after;
    result.cache_cuda_free_drop_bytes =
        free_before >= free_after ? free_before - free_after : 0U;
    const std::optional<std::uint64_t> host_private_after =
        process_private_memory_bytes();
    result.cache_host_private_observed =
        host_private_before.has_value() && host_private_after.has_value();
    if (result.cache_host_private_observed) {
      result.cache_host_private_bytes_before = *host_private_before;
      result.cache_host_private_bytes_after = *host_private_after;
      result.cache_host_private_increase_bytes =
          *host_private_after >= *host_private_before
              ? *host_private_after - *host_private_before
              : 0U;
    }
    if (!compare_arenas(baseline.data, arena,
                        &result.cache_prepare_arena_exact,
                        "decode_graph_p2_prepare_arena")) {
      return outcome;
    }
    if (!topology_exact || !result.cache_prepare_arena_exact ||
        state.current_position() != 0U ||
        !std::isfinite(result.cache_prepare_milliseconds) ||
        result.cache_prepare_milliseconds < 0.0) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_prepare_contract",
          "cache preparation changed state or violated topology/timing");
      return outcome;
    }

    if (!set_position(options.boundary_graph_position,
                      "decode_graph_p2_prepare_boundary_position")) {
      return outcome;
    }
    ReferenceDecodeGraphP1PrepareOutcome boundary_prepared =
        runner.prepare_fixed_position_decode_graph_p1(
            options.capture_input_token_id);
    if (!boundary_prepared) {
      outcome.diagnostic = runner_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_prepare_boundary", boundary_prepared.status);
      return outcome;
    }
    if (state.current_position() != options.boundary_graph_position) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_prepare_boundary_position_contract",
          "boundary graph preparation changed logical sequence length",
          "expected=" +
              std::to_string(options.boundary_graph_position) +
              " actual=" + std::to_string(state.current_position()));
      return outcome;
    }
    result.boundary_graph = *boundary_prepared.value;
    topology_exact =
        topology_exact &&
        result.boundary_graph.position == options.boundary_graph_position &&
        result.boundary_graph.input_token_id ==
            options.capture_input_token_id &&
        result.boundary_graph.node_count == 390U &&
        result.boundary_graph.kernel_node_count == 389U &&
        result.boundary_graph.memcpy_node_count == 1U &&
        result.boundary_graph.other_node_count == 0U &&
        runner.has_fixed_position_decode_graph_p1(
            options.boundary_graph_position) &&
        !runner.has_fixed_position_decode_graph_p1(
            options.boundary_graph_position + 1U);
    if (!set_position(0U,
                      "decode_graph_p2_prepare_boundary_restore_position") ||
        !compare_arenas(baseline.data, arena,
                        &result.boundary_prepare_arena_exact,
                        "decode_graph_p2_prepare_boundary_arena")) {
      return outcome;
    }
    std::size_t free_after_boundary = 0U;
    std::size_t total_after_boundary = 0U;
    cuda_status = cudaMemGetInfo(&free_after_boundary,
                                 &total_after_boundary);
    if (cuda_status != cudaSuccess ||
        total_before != total_after_boundary) {
      set_cuda_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_boundary_memory_after",
          "failed to read stable CUDA memory after boundary graph prep",
          cuda_status == cudaSuccess ? cudaErrorUnknown : cuda_status);
      return outcome;
    }
    result.boundary_free_bytes_after = free_after_boundary;
    result.boundary_cuda_free_drop_bytes =
        free_after >= free_after_boundary
            ? free_after - free_after_boundary
            : 0U;
    result.cache_plus_boundary_cuda_free_drop_bytes =
        free_before >= free_after_boundary
            ? free_before - free_after_boundary
            : 0U;
    if (!topology_exact || !result.boundary_prepare_arena_exact ||
        state.current_position() != 0U) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_prepare_boundary_contract",
          "boundary graph preparation changed state or violated topology");
      return outcome;
    }

    // This is the first replay of every distinct uploaded slot. It is the
    // production-alignment correctness chain, not a warmed single-exec loop.
    if (!restore_from(baseline.data, 0U,
                      "decode_graph_p2_graph_restore") ||
        !run_generation(user_prompt, options.max_new_tokens, true,
                        ReferenceLogitsMode::kPredictedTokenOnly, false,
                        "decode_graph_p2_graph_generation",
                        &result.graph_generation) ||
        !compare_arenas(serial_after.data, arena,
                        &result.continuous_arena_exact,
                        "decode_graph_p2_continuous_arena")) {
      return outcome;
    }
    result.continuous_generation_exact =
        valid_continuous_shape(result.graph_generation) &&
        same_generation_semantics(result.serial_generation,
                                  result.graph_generation) &&
        result.graph_generation.decode_graph_replays ==
            kReferenceDecodeGraphP2ContinuousSteps &&
        result.graph_generation.decode_graph_serial_fallbacks == 0U &&
        state.current_position() == options.last_decode_position + 1U;
    if (!result.continuous_generation_exact ||
        !result.continuous_arena_exact) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_continuous_correctness",
          "continuous Graph generation did not reproduce serial semantics "
          "and complete arena");
      return outcome;
    }

    ReferenceGeneration full_statistics_serial;
    ReferenceGeneration full_statistics_fallback;
    bool full_statistics_arena_exact = false;
    if (!restore_from(baseline.data, 0U,
                      "decode_graph_p2_full_stats_serial_restore") ||
        !run_generation(user_prompt, options.max_new_tokens, false,
                        ReferenceLogitsMode::kFullStatistics, false,
                        "decode_graph_p2_full_stats_serial",
                        &full_statistics_serial) ||
        !copy_arena_to(serial_after.data,
                       "decode_graph_p2_full_stats_serial_snapshot") ||
        !restore_from(baseline.data, 0U,
                      "decode_graph_p2_full_stats_fallback_restore") ||
        !run_generation(user_prompt, options.max_new_tokens, true,
                        ReferenceLogitsMode::kFullStatistics, false,
                        "decode_graph_p2_full_stats_generation",
                        &full_statistics_fallback) ||
        !compare_arenas(serial_after.data, arena,
                        &full_statistics_arena_exact,
                        "decode_graph_p2_full_stats_arena")) {
      return outcome;
    }
    result.full_statistics_fallback_exact =
        same_generation_semantics(full_statistics_serial,
                                  full_statistics_fallback) &&
        full_statistics_serial.decode_graph_replays == 0U &&
        full_statistics_serial.decode_graph_serial_fallbacks == 0U &&
        full_statistics_fallback.decode_graph_replays == 0U &&
        full_statistics_fallback.decode_graph_serial_fallbacks ==
            kReferenceDecodeGraphP2ContinuousSteps &&
        full_statistics_arena_exact;
    if (!result.full_statistics_fallback_exact) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_full_stats_fallback",
          "full-statistics opt-in did not take an exact serial fallback");
      return outcome;
    }

    ReferenceGeneration trace_serial;
    ReferenceGeneration trace_fallback;
    bool trace_arena_exact = false;
    if (!restore_from(baseline.data, 0U,
                      "decode_graph_p2_trace_serial_restore") ||
        !run_generation(user_prompt, options.max_new_tokens, false,
                        ReferenceLogitsMode::kPredictedTokenOnly, true,
                        "decode_graph_p2_trace_serial", &trace_serial) ||
        !copy_arena_to(serial_after.data,
                       "decode_graph_p2_trace_serial_snapshot") ||
        !restore_from(baseline.data, 0U,
                      "decode_graph_p2_trace_fallback_restore") ||
        !run_generation(user_prompt, options.max_new_tokens, true,
                        ReferenceLogitsMode::kPredictedTokenOnly, true,
                        "decode_graph_p2_trace_generation",
                        &trace_fallback) ||
        !compare_arenas(serial_after.data, arena, &trace_arena_exact,
                        "decode_graph_p2_trace_arena")) {
      return outcome;
    }
    result.trace_fallback_exact =
        same_generation_semantics(trace_serial, trace_fallback) &&
        trace_serial.decode_graph_replays == 0U &&
        trace_serial.decode_graph_serial_fallbacks == 0U &&
        trace_fallback.decode_graph_replays == 0U &&
        trace_fallback.decode_graph_serial_fallbacks ==
            kReferenceDecodeGraphP2ContinuousSteps &&
        trace_fallback.traces.size() == trace_fallback.steps.size() &&
        trace_arena_exact;
    if (!result.trace_fallback_exact) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_trace_fallback",
          "trace opt-in did not take an exact serial fallback");
      return outcome;
    }

    std::string boundary_prompt;
    for (std::size_t word = 0U; word < 51U; ++word) {
      if (!boundary_prompt.empty()) {
        boundary_prompt.push_back(' ');
      }
      boundary_prompt += "hello";
    }
    ReferenceGeneration boundary_seed;
    if (!restore_from(baseline.data, 0U,
                      "decode_graph_p2_boundary_seed_restore") ||
        !run_generation(boundary_prompt, 1U, false,
                        ReferenceLogitsMode::kPredictedTokenOnly, false,
                        "decode_graph_p2_boundary_seed", &boundary_seed) ||
        boundary_seed.prompt_token_ids.size() !=
            options.boundary_graph_position ||
        boundary_seed.generated_token_ids.size() != 1U ||
        boundary_seed.generated_token_ids.front() !=
            options.boundary_input_token_id ||
        state.current_position() != options.boundary_graph_position ||
        !copy_arena_to(boundary_baseline.data,
                       "decode_graph_p2_boundary_baseline")) {
      if (outcome.diagnostic.code == ReferenceEngineError::kNone) {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kRunnerStepFailure,
            "decode_graph_p2_boundary_prompt_shape",
            "the pinned 51-word boundary prompt did not produce P63");
      }
      return outcome;
    }

    ReferenceStepOptions boundary_step_options;
    boundary_step_options.compute_logits = true;
    boundary_step_options.capture_trace = false;
    boundary_step_options.measure_timing = false;
    boundary_step_options.logits_mode =
        ReferenceLogitsMode::kPredictedTokenOnly;
    auto boundary_step = [&](const bool graph,
                             const std::uint32_t input_token_id,
                             std::uint32_t* const prediction) -> bool {
      ReferenceStepOutcome stepped =
          graph ? runner.replay_fixed_position_decode_graph_p1(
                      input_token_id, false)
                : runner.step(input_token_id, boundary_step_options);
      if (!stepped || !stepped.value->prediction.has_value()) {
        if (!stepped) {
          outcome.diagnostic = runner_diagnostic(
              ReferenceEngineError::kRunnerStepFailure,
              graph ? "decode_graph_p2_boundary_graph"
                    : "decode_graph_p2_boundary_serial",
              stepped.status);
        } else {
          outcome.diagnostic = engine_diagnostic(
              ReferenceEngineError::kMissingPrediction,
              "decode_graph_p2_boundary_prediction",
              "boundary step did not return a prediction");
        }
        return false;
      }
      *prediction = stepped.value->prediction->predicted_token_id;
      return true;
    };

    const std::uint32_t boundary_input =
        boundary_seed.generated_token_ids.front();
    if (!boundary_step(false, boundary_input,
                       &result.boundary_serial_first_prediction) ||
        !boundary_step(false, result.boundary_serial_first_prediction,
                       &result.boundary_serial_second_prediction) ||
        !copy_arena_to(serial_after.data,
                       "decode_graph_p2_boundary_serial_snapshot") ||
        !restore_from(boundary_baseline.data,
                      options.boundary_graph_position,
                      "decode_graph_p2_boundary_graph_restore")) {
      return outcome;
    }
    result.boundary_cache_hit =
        runner.has_fixed_position_decode_graph_p1(
            options.boundary_graph_position);
    if (!boundary_step(true, boundary_input,
                       &result.boundary_graph_first_prediction)) {
      return outcome;
    }
    result.boundary_cache_miss_fallback =
        state.current_position() == options.boundary_graph_position + 1U &&
        !runner.has_fixed_position_decode_graph_p1(
            options.boundary_graph_position + 1U);
    EngineStepContext boundary_dispatch_context;
    boundary_dispatch_context.runner = &runner;
    boundary_dispatch_context.capture_trace = false;
    ReferenceStepOutcome boundary_fallback =
        decode_with_prepared_graph_cache(
            &boundary_dispatch_context,
            result.boundary_graph_first_prediction,
            boundary_step_options);
    if (!boundary_fallback ||
        !boundary_fallback.value->prediction.has_value()) {
      if (!boundary_fallback) {
        outcome.diagnostic = runner_diagnostic(
            ReferenceEngineError::kRunnerStepFailure,
            "decode_graph_p2_boundary_dispatch_fallback",
            boundary_fallback.status);
      } else {
        outcome.diagnostic = engine_diagnostic(
            ReferenceEngineError::kMissingPrediction,
            "decode_graph_p2_boundary_dispatch_prediction",
            "P64 dispatcher fallback did not return a prediction");
      }
      return outcome;
    }
    result.boundary_fallback_second_prediction =
        boundary_fallback.value->prediction->predicted_token_id;
    result.boundary_cache_miss_fallback =
        result.boundary_cache_miss_fallback &&
        boundary_dispatch_context.decode_graph_replays == 0U &&
        boundary_dispatch_context.decode_graph_serial_fallbacks == 1U;
    if (!compare_arenas(serial_after.data, arena,
                        &result.boundary_arena_exact,
                        "decode_graph_p2_boundary_arena")) {
      return outcome;
    }
    if (!result.boundary_cache_hit ||
        !result.boundary_cache_miss_fallback ||
        result.boundary_graph_first_prediction !=
            result.boundary_serial_first_prediction ||
        result.boundary_fallback_second_prediction !=
            result.boundary_serial_second_prediction ||
        !result.boundary_arena_exact ||
        state.current_position() != options.boundary_graph_position + 2U) {
      outcome.diagnostic = engine_diagnostic(
          ReferenceEngineError::kRunnerStepFailure,
          "decode_graph_p2_boundary_correctness",
          "P63 Graph plus P64 serial fallback did not reproduce the serial "
          "two-step boundary");
      return outcome;
    }

    auto run_timed_segment = [&](const bool graph,
                                 double* const milliseconds) -> bool {
      if (!restore_from(baseline.data, 0U,
                        graph ? "decode_graph_p2_timed_graph_restore"
                              : "decode_graph_p2_timed_serial_restore")) {
        return false;
      }
      ReferenceGeneration generated;
      if (!run_generation(user_prompt, options.max_new_tokens, graph,
                          ReferenceLogitsMode::kPredictedTokenOnly, false,
                          graph ? "decode_graph_p2_timed_graph"
                                : "decode_graph_p2_timed_serial",
                          &generated) ||
          !same_generation_semantics(result.serial_generation, generated) ||
          (graph &&
           (generated.decode_graph_replays !=
                kReferenceDecodeGraphP2ContinuousSteps ||
            generated.decode_graph_serial_fallbacks != 0U)) ||
          (!graph && (generated.decode_graph_replays != 0U ||
                      generated.decode_graph_serial_fallbacks != 0U)) ||
          !decode_milliseconds_per_token(
              generated, kReferenceDecodeGraphP2ContinuousSteps,
              *milliseconds)) {
        if (outcome.diagnostic.code == ReferenceEngineError::kNone) {
          outcome.diagnostic = engine_diagnostic(
              ReferenceEngineError::kRunnerStepFailure,
              "decode_graph_p2_timed_contract",
              "timed segment did not preserve the serial generation "
              "contract");
        }
        return false;
      }
      return true;
    };

    for (const bool graph : {false, true, true, false}) {
      double ignored = 0.0;
      if (!run_timed_segment(graph, &ignored)) {
        return outcome;
      }
    }
    for (ReferenceDecodeGraphP2RoundTiming& round : result.rounds) {
      if (!run_timed_segment(
              false, &round.serial_first_milliseconds_per_token) ||
          !run_timed_segment(
              true, &round.graph_first_milliseconds_per_token) ||
          !run_timed_segment(
              true, &round.graph_second_milliseconds_per_token) ||
          !run_timed_segment(
              false, &round.serial_second_milliseconds_per_token)) {
        return outcome;
      }
    }

    if (!restore_from(baseline.data, 0U,
                      "decode_graph_p2_final_restore")) {
      return outcome;
    }
    restore_guard.armed = false;
    outcome.value.emplace(std::move(result));
    return outcome;
  } catch (const std::bad_alloc&) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure,
        "decode_graph_p2_screen",
        "host allocation failed during the Decode Graph P2 screen");
    return outcome;
  } catch (const std::length_error& error) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure,
        "decode_graph_p2_screen", error.what());
    return outcome;
  } catch (const std::exception& error) {
    outcome.diagnostic = engine_diagnostic(
        ReferenceEngineError::kRunnerStepFailure,
        "decode_graph_p2_screen", error.what());
    return outcome;
  }
}

ReferenceEngineCreateResult create_reference_engine(
    const std::filesystem::path& model_directory,
    const ReferenceEngineOptions& options) {
  ReferenceEngineCreateResult result;
  ReferenceEngine::Impl::BuildResult built =
      ReferenceEngine::Impl::build(model_directory, options);
  if (!built) {
    result.diagnostic = std::move(built.diagnostic);
    return result;
  }
  result.value.emplace(ReferenceEngine(std::move(built.value)));
  return result;
}

ReferenceOneShotResult generate_reference(
    const std::filesystem::path& model_directory,
    const std::string_view user_prompt,
    const ReferenceOneShotOptions& options) {
  ReferenceOneShotResult result;
  if (model_directory.empty() || user_prompt.empty() ||
      options.generation.max_new_tokens == 0U ||
      options.generation.prefill_chunk_size == 0U ||
      options.generation.prefill_chunk_size >
          kMaximumRequestPrefillChunkSize ||
      options.generation.stop_token_id >= kReferenceVocabularySize ||
      !is_valid_reference_logits_mode(options.generation.logits_mode) ||
      !is_valid_projection_backend(options.projection_backend) ||
      !is_valid_reference_decode_graph_cache_policy(
          options.decode_graph_cache_policy)) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "one_shot_options",
        "model directory and prompt must be non-empty; generation options "
        "must be valid");
    return result;
  }

  try {
    const Clock::time_point one_shot_prepare_begin = Clock::now();
    std::optional<std::future<TimedResidentLoad>> resident_future;
    std::optional<ReferenceEngineDiagnostic> startup_device_diagnostic;
    if (options.overlap_tokenizer_and_resident_load) {
      int active_device = 0;
      startup_device_diagnostic =
          sm87_device_diagnostic(options.projection_backend, &active_device);
      if (!startup_device_diagnostic.has_value() &&
          options.projection_backend != ProjectionBackend::kSm87WeightOnly) {
        (void)cudaGetLastError();
        const cudaError_t cuda_status = cudaGetDevice(&active_device);
        if (cuda_status != cudaSuccess) {
          startup_device_diagnostic = engine_diagnostic(
              ReferenceEngineError::kResidentLoadFailure, "resident_load",
              "failed to inspect the active CUDA device before concurrent "
              "model load",
              "cudaGetDevice(startup)");
          startup_device_diagnostic->cuda_error =
              static_cast<int>(cuda_status);
        }
      }

      if (!startup_device_diagnostic.has_value()) {
        try {
          resident_future.emplace(std::async(
              std::launch::async,
              [directory = std::filesystem::path(model_directory),
               resident_options = options.resident_options,
               active_device]() {
                return load_resident_on_device(directory, resident_options,
                                               active_device);
              }));
        } catch (const std::system_error&) {
          // Resource exhaustion in the host scheduler must not make a model
          // unloadable. Fall back to the existing serial build path.
          resident_future.reset();
        }
      }
    }

    const Clock::time_point tokenizer_begin = Clock::now();
    const std::filesystem::path tokenizer_path =
        model_directory / "tokenizer.json";
    text::TokenizerLoadResult tokenizer =
        text::Tokenizer::load_file(tokenizer_path.string());
    const double tokenizer_milliseconds =
        elapsed_milliseconds(tokenizer_begin);
    if (!tokenizer) {
      result.diagnostic =
          tokenizer_diagnostic("tokenizer_load", tokenizer.error);
      return result;
    }

    text::ChatResult preflight =
        format_single_user_prompt(*tokenizer.tokenizer, user_prompt);
    if (!preflight) {
      result.diagnostic = tokenizer_diagnostic("chat_encode", preflight.error);
      return result;
    }
    if (preflight.token_ids.empty()) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kTokenizerFailure, "chat_encode",
          "the rendered chat prompt encoded to zero tokens");
      return result;
    }

    std::uint64_t required_steps = 0U;
    if (!checked_required_steps(preflight.token_ids.size(),
                                options.generation.max_new_tokens,
                                required_steps)) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kArithmeticOverflow, "request_capacity",
          "prompt plus decode capacity overflows uint64");
      return result;
    }
    if (required_steps == 0U ||
        required_steps > kAbsoluteRequestMaxSequenceLength ||
        required_steps > std::numeric_limits<std::uint32_t>::max()) {
      result.diagnostic = engine_diagnostic(
          ReferenceEngineError::kCapacityExceeded, "request_capacity",
          "prompt plus decode steps exceed the request-state limit",
          "required_steps=" + std::to_string(required_steps));
      return result;
    }
    // Preserve the established tokenizer/chat/capacity diagnostic priority.
    // A device probe is needed before starting the worker, but its failure is
    // reported only after all CPU-only preflight checks succeed.
    if (startup_device_diagnostic.has_value()) {
      result.diagnostic = std::move(*startup_device_diagnostic);
      return result;
    }

    ReferenceEngineOptions engine_options;
    engine_options.resident_options = options.resident_options;
    engine_options.request_options.batch_size = 1U;
    engine_options.request_options.prefill_chunk_size =
        options.generation.prefill_chunk_size;
    engine_options.request_options.max_sequence_length =
        static_cast<std::uint32_t>(required_steps);
    engine_options.request_options.max_arena_bytes =
        options.request_max_arena_bytes;
    engine_options.request_options.min_free_bytes_after_create =
        options.request_min_free_bytes_after_create;
    engine_options.enable_trace = options.generation.capture_trace;
    engine_options.projection_backend = options.projection_backend;
    engine_options.decode_graph_cache_policy =
        options.decode_graph_cache_policy;

    ReferenceEngine::Impl::BuildResult built;
    if (resident_future.has_value()) {
      TimedResidentLoad resident = resident_future->get();
      if (!resident.result) {
        result.diagnostic = resident_diagnostic(resident.result.diagnostic);
        return result;
      }
      std::optional<ResidentWeights> prepared_resident;
      prepared_resident.emplace(std::move(*resident.result.value));
      built = ReferenceEngine::Impl::build(
          model_directory, engine_options, std::move(tokenizer.tokenizer),
          tokenizer_milliseconds, std::move(prepared_resident),
          resident.milliseconds,
          elapsed_milliseconds(one_shot_prepare_begin));
    } else {
      built = ReferenceEngine::Impl::build(
          model_directory, engine_options, std::move(tokenizer.tokenizer),
          tokenizer_milliseconds, std::nullopt, 0.0,
          elapsed_milliseconds(one_shot_prepare_begin));
    }
    if (!built) {
      result.diagnostic = std::move(built.diagnostic);
      return result;
    }
    built.value->load.tokenizer_resident_overlap =
        resident_future.has_value();

    ReferenceEngine engine(std::move(built.value));
    ReferenceGenerateResult generated =
        engine.generate(user_prompt, options.generation);
    if (!generated) {
      result.diagnostic = std::move(generated.diagnostic);
      return result;
    }

    ReferenceOneShotGeneration value;
    value.load = engine.load_stats();
    value.generation = std::move(*generated.value);
    result.value.emplace(std::move(value));
    return result;
  } catch (const std::bad_alloc&) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure, "one_shot",
        "host allocation failed during one-shot generation");
    return result;
  } catch (const std::length_error& error) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kAllocationFailure, "one_shot", error.what());
    return result;
  } catch (const std::exception& error) {
    result.diagnostic = engine_diagnostic(
        ReferenceEngineError::kInvalidArgument, "one_shot", error.what());
    return result;
  }
}

std::string_view to_string(const ReferenceEngineError error) noexcept {
  switch (error) {
    case ReferenceEngineError::kNone:
      return "none";
    case ReferenceEngineError::kInvalidArgument:
      return "invalid_argument";
    case ReferenceEngineError::kCapacityExceeded:
      return "capacity_exceeded";
    case ReferenceEngineError::kArithmeticOverflow:
      return "arithmetic_overflow";
    case ReferenceEngineError::kTokenizerFailure:
      return "tokenizer_failure";
    case ReferenceEngineError::kResidentLoadFailure:
      return "resident_load_failure";
    case ReferenceEngineError::kWeightBindFailure:
      return "weight_bind_failure";
    case ReferenceEngineError::kRequestStateFailure:
      return "request_state_failure";
    case ReferenceEngineError::kRunnerFactoryFailure:
      return "runner_factory_failure";
    case ReferenceEngineError::kRunnerStepFailure:
      return "runner_step_failure";
    case ReferenceEngineError::kRunnerResetFailure:
      return "runner_reset_failure";
    case ReferenceEngineError::kMissingLogits:
      return "missing_logits";
    case ReferenceEngineError::kMissingTiming:
      return "missing_timing";
    case ReferenceEngineError::kDecodeFailure:
      return "decode_failure";
    case ReferenceEngineError::kTraceFailure:
      return "trace_failure";
    case ReferenceEngineError::kAllocationFailure:
      return "allocation_failure";
    case ReferenceEngineError::kMissingPrediction:
      return "missing_prediction";
  }
  return "unknown";
}

std::string_view to_string(const ReferenceStopReason reason) noexcept {
  switch (reason) {
    case ReferenceStopReason::kImEnd:
      return "im_end";
    case ReferenceStopReason::kMaxNewTokens:
      return "max_new_tokens";
  }
  return "unknown";
}

}  // namespace q3x::runtime
