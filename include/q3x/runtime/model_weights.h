#pragma once

#include "q3x/model/model_config.h"
#include "q3x/runtime/resident_weights.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <variant>

namespace q3x::runtime {

inline constexpr std::size_t kQwen36DenseLayerCount = 64U;
inline constexpr std::size_t kQwen36LinearAttentionLayerCount = 48U;
inline constexpr std::size_t kQwen36FullAttentionLayerCount = 16U;

struct Bf16VectorWeight {
  const std::uint16_t* data = nullptr;
  std::size_t element_count = 0U;
};

struct Bf16Tensor3Weight {
  const std::uint16_t* data = nullptr;
  std::array<std::size_t, 3U> shape{};
};

struct Bf16LinearWeight {
  const std::uint16_t* weight = nullptr;
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
};

struct Fp8LinearWeight {
  const std::uint8_t* weight = nullptr;
  const float* weight_scale_device = nullptr;
  const float* input_scale_device = nullptr;
  float weight_scale = 0.0F;
  float input_scale = 0.0F;
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
};

struct NvFp4LinearWeight {
  const std::uint8_t* packed_weight = nullptr;
  const std::uint8_t* block_scale = nullptr;
  const float* weight_scale_2_device = nullptr;
  const float* input_scale_device = nullptr;
  float weight_scale_2 = 0.0F;
  float input_scale = 0.0F;
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
};

// The active alternative is selected strictly from the payload weight dtype:
// BF16, F8_E4M3, or canonical ModelOpt packed U8 NVFP4 respectively.
using LinearWeight =
    std::variant<Bf16LinearWeight, Fp8LinearWeight, NvFp4LinearWeight>;

enum class LinearWeightKind : std::uint8_t {
  kBf16,
  kFp8,
  kNvFp4,
};

// Production projection policy. The correctness reference remains the
// default; the SM87 backend must be selected explicitly by a caller.
enum class ProjectionBackend : std::uint8_t {
  kReference = 0,
  kSm87WeightOnly,
};

[[nodiscard]] bool is_valid_projection_backend(
    ProjectionBackend backend) noexcept;
[[nodiscard]] std::string_view to_string(ProjectionBackend backend) noexcept;

[[nodiscard]] LinearWeightKind linear_weight_kind(
    const LinearWeight& weight) noexcept;
[[nodiscard]] std::size_t linear_output_size(
    const LinearWeight& weight) noexcept;
[[nodiscard]] std::size_t linear_input_size(
    const LinearWeight& weight) noexcept;

struct DenseMlpWeights {
  LinearWeight gate_proj;
  LinearWeight up_proj;
  LinearWeight down_proj;
};

struct LinearAttentionWeights {
  LinearWeight in_proj_qkv;
  LinearWeight in_proj_z;
  LinearWeight in_proj_a;
  LinearWeight in_proj_b;
  Bf16Tensor3Weight conv1d;
  Bf16VectorWeight a_log;
  Bf16VectorWeight dt_bias;
  Bf16VectorWeight norm;
  LinearWeight out_proj;
};

struct FullAttentionWeights {
  LinearWeight q_proj;
  LinearWeight k_proj;
  LinearWeight v_proj;
  LinearWeight o_proj;
  Bf16VectorWeight q_norm;
  Bf16VectorWeight k_norm;
};

using AttentionWeights =
    std::variant<LinearAttentionWeights, FullAttentionWeights>;

struct DecoderLayerWeights {
  Bf16VectorWeight input_layernorm;
  Bf16VectorWeight post_attention_layernorm;
  DenseMlpWeights mlp;
  AttentionWeights attention;
};

struct WeightBindingStats {
  std::size_t tensor_views = 0U;
  std::size_t scalar_reads = 0U;
  std::size_t bf16_projections = 0U;
  std::size_t fp8_projections = 0U;
  std::size_t nvfp4_projections = 0U;
  std::size_t linear_attention_layers = 0U;
  std::size_t full_attention_layers = 0U;
};

class ModelWeightBinder;
struct WeightBindingSource;
struct WeightBindResult;

// ModelWeights is a non-owning, allocation-free-after-bind view. It never
// extends the lifetime of ResidentWeights: the exact ResidentWeights object
// and its CUDA arena must outlive this object and every queued kernel using
// any pointer obtained from it. Moving the owning ResidentWeights is safe
// because a move transfers (rather than relocates) its CUDA allocation;
// destroying or move-assigning that owner invalidates these views.
class ModelWeights {
 public:
  ModelWeights(const ModelWeights&) = delete;
  ModelWeights& operator=(const ModelWeights&) = delete;
  ModelWeights(ModelWeights&&) noexcept = default;
  ModelWeights& operator=(ModelWeights&&) noexcept = default;
  ~ModelWeights() = default;

  [[nodiscard]] const Bf16LinearWeight& embed_tokens() const noexcept {
    return embed_tokens_;
  }
  [[nodiscard]] const Bf16VectorWeight& final_norm() const noexcept {
    return final_norm_;
  }
  [[nodiscard]] const LinearWeight& lm_head() const noexcept {
    return lm_head_;
  }
  [[nodiscard]] const std::array<DecoderLayerWeights,
                                 kQwen36DenseLayerCount>&
  layers() const noexcept {
    return layers_;
  }
  [[nodiscard]] const DecoderLayerWeights& layer(
      const std::size_t index) const noexcept {
    return layers_[index];
  }
  [[nodiscard]] const WeightBindingStats& stats() const noexcept {
    return stats_;
  }

 private:
  friend class ModelWeightBinder;
  friend struct WeightBindResult;
  friend WeightBindResult bind_qwen36_27b_weights(
      const WeightBindingSource&);

  ModelWeights() = default;

  Bf16LinearWeight embed_tokens_;
  Bf16VectorWeight final_norm_;
  LinearWeight lm_head_;
  std::array<DecoderLayerWeights, kQwen36DenseLayerCount> layers_{};
  WeightBindingStats stats_;
};

enum class WeightBindErrorCode : std::uint8_t {
  kNone,
  kInvalidSource,
  kInvalidPinnedArena,
  kMissingTensor,
  kUnsupportedWeightDType,
  kDTypeMismatch,
  kShapeMismatch,
  kByteSizeMismatch,
  kNullDevicePointer,
  kMisalignedTensor,
  kArenaRangeMismatch,
  kInvalidScalar,
  kCudaFailure,
  kInvalidLayerSchedule,
  kArithmeticOverflow,
  kAllocationFailure,
};

struct WeightBindDiagnostic {
  WeightBindErrorCode code = WeightBindErrorCode::kNone;
  std::string tensor;
  std::string message;
  std::string expected;
  std::string actual;
  int cuda_error = 0;
};

using DeviceTensorLookupFn = const DeviceTensorView* (*)(
    const void* context, std::string_view name) noexcept;
using DeviceScalarReadFn = int (*)(const void* context,
                                   const float* device_value,
                                   float* host_value) noexcept;

// Adapter boundary for deterministic tests and already-validated custom
// arenas. Every returned view is still checked for exact dtype, shape, byte
// count, 256-byte alignment, and consistency with [arena_data, arena_bytes).
// A null scalar_read selects a synchronous cudaMemcpyDeviceToHost performed
// once per scalar during binding. A custom callback must return zero on
// success and a CUDA-style nonzero status on failure.
struct WeightBindingSource {
  const void* lookup_context = nullptr;
  DeviceTensorLookupFn lookup = nullptr;
  const void* arena_data = nullptr;
  std::uint64_t arena_bytes = 0U;
  const void* scalar_read_context = nullptr;
  DeviceScalarReadFn scalar_read = nullptr;
};

struct WeightBindResult {
  std::optional<ModelWeights> value;
  WeightBindDiagnostic diagnostic;

  [[nodiscard]] bool ok() const noexcept {
    return value.has_value() && diagnostic.code == WeightBindErrorCode::kNone;
  }
  [[nodiscard]] explicit operator bool() const noexcept { return ok(); }
};

// Binds a checked view source. This does not require the arena byte count to
// equal the official pinned allocation, allowing compact/fake lookup adapters
// in tests; all individual tensors must nevertheless satisfy the exact ABI.
[[nodiscard]] WeightBindResult bind_qwen36_27b_weights(
    const WeightBindingSource& source);

// Production entry point. In addition to all view checks, this requires the
// exact pinned 20,150,786,560-byte ResidentWeights arena contract.
[[nodiscard]] WeightBindResult bind_qwen36_27b_weights(
    const ResidentWeights& resident);

[[nodiscard]] std::string_view to_string(WeightBindErrorCode code) noexcept;

// Asynchronous, allocation-free projection dispatch. Input is BF16 device
// storage and output is FP32 device storage. input_scale is retained and
// validated by binding for checkpoint fidelity but is deliberately not used
// by the SM87 W8A16/W4A16 reference paths.
[[nodiscard]] int launch_projection_reference_cuda(
    const LinearWeight& weight, const std::uint16_t* input,
    float* output, void* cuda_stream = nullptr) noexcept;

// Convenience boundary for the next BF16 activation. fp32_scratch is caller
// owned and must contain at least linear_output_size(weight) elements. No
// allocation or synchronization is performed.
[[nodiscard]] int launch_projection_to_bf16_reference_cuda(
    const LinearWeight& weight, const std::uint16_t* input,
    float* fp32_scratch, std::size_t scratch_elements,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Allocation-free production dispatcher. kReference preserves the two-stage
// FP32-scratch reference path. kSm87WeightOnly writes FP8/NVFP4 projections
// directly to BF16 and falls back to that reference path for BF16 weights.
// Consequently fp32_scratch may be null for an SM87 FP8/NVFP4 launch, but it
// must satisfy the reference contract for kReference and the BF16 fallback.
// Unknown backends or invalid/unknown LinearWeight alternatives fail closed
// with cudaErrorInvalidValue represented as int.
[[nodiscard]] int launch_projection_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, float* fp32_scratch,
    std::size_t scratch_elements, std::uint16_t* output,
    void* cuda_stream = nullptr) noexcept;

// Sequence-tile form of launch_projection_to_bf16_cuda. input is contiguous
// token-major BF16 [token_count, linear_input_size(weight)] and output is
// contiguous token-major BF16 [token_count, linear_output_size(weight)].
// token_count must be in [1, 8]. M=1 delegates to the single-token entry
// point above. SM87 FP8/NVFP4 weights use the fused small-M kernels; the
// reference backend and BF16 weights enqueue the existing single-token path
// in token order while reusing the same output-sized FP32 scratch buffer.
// The complete tile is validated before any work is enqueued.
[[nodiscard]] int launch_projection_tile_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::size_t token_count,
    float* fp32_scratch, std::size_t scratch_elements,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime
