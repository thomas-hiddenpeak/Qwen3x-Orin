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

// True only for the production fused linear-attention A/B projection ABI:
// the explicitly selected SM87 backend and two valid BF16 [48, 5120]
// matrices. This is a shape/type eligibility check; launch-time input,
// output, scratch, range, and alias validation remains the dispatcher's
// responsibility.
[[nodiscard]] bool supports_bf16_projection_pair(
    ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight) noexcept;

// True only for the production full-attention K/V projection ABI: the
// explicitly selected SM87 backend and two valid FP8 [1024, 5120] matrices.
// This is shape/type/payload eligibility only. The fused kernel is M=1-only;
// launch-time token-count, pointer alignment, input, output, range, and alias
// validation remains the dispatcher's responsibility. Eligible but unaligned
// calls preserve the two independent projection fallbacks.
[[nodiscard]] bool supports_fp8_projection_pair(
    ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight) noexcept;

// True only for the production linear-attention QKV/Z projection ABI: the
// explicitly selected SM87 backend followed by a valid FP8 [10240, 5120] QKV
// matrix and a valid FP8 [6144, 5120] Z matrix, in that order. This is
// shape/type/payload eligibility only; launch-time token-count, pointer
// alignment, input, output, range, and alias validation remains the
// dispatcher's responsibility.
[[nodiscard]] bool supports_fp8_qkv_z_projection_pair(
    ProjectionBackend backend, const LinearWeight& qkv_weight,
    const LinearWeight& z_weight) noexcept;

// True only for the production dense-MLP gate/up/SiLU fusion ABI: the
// explicitly selected SM87 backend followed by valid NVFP4 [17408, 5120]
// gate and up matrices, in that order. This is shape/type/payload eligibility
// only; launch-time alignment, input, output, scratch, range, and alias
// validation remains the dispatcher's responsibility.
[[nodiscard]] bool supports_nvfp4_gate_up_silu_fusion(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight) noexcept;

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

// Allocation-free single-token production dispatcher. kReference preserves
// the two-stage FP32-scratch reference path. kSm87WeightOnly writes FP8/NVFP4
// projections directly to BF16 and also directly handles the exact BF16
// [output_size=48, input_size=5120] linear-attention shape. Consequently
// fp32_scratch may be null for those SM87 direct-output launches, but it must
// satisfy the reference contract for kReference and every other BF16 shape.
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
// token_count must be in [1, 16]. M=1 delegates to the single-token entry
// point above. SM87 FP8 and NVFP4 use their fixed-M16 launchers for an exact
// 16-token tile; M=2..15 use fused launches of at most eight tokens each. The
// reference backend and BF16 weights enqueue the existing FP32-scratch
// reference path in token order while reusing the same output-sized buffer;
// only M=1 may select the exact-shape BF16 direct-output route described
// above. The complete tile is validated before any work is enqueued.
[[nodiscard]] int launch_projection_tile_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& weight,
    const std::uint16_t* input, std::size_t token_count,
    float* fp32_scratch, std::size_t scratch_elements,
    std::uint16_t* output, void* cuda_stream = nullptr) noexcept;

// Validates and launches two projections over the same token-major input.
// The two weights may have different output sizes but must have the same
// input size. All host-visible arguments, byte ranges, and cross-projection
// aliases are rejected before either projection is enqueued. The caller's
// FP32 scratch must satisfy both independent projection contracts except on
// the fused SM87 direct-to-BF16 path, where it is unused and may be null.
//
// supports_bf16_projection_pair(...) selects the fused SM87 BF16 A/B kernel
// for M=1..16. supports_fp8_projection_pair(...) selects the fused SM87 FP8
// K/V kernel only for M=1. supports_fp8_qkv_z_projection_pair(...) selects
// the fused SM87 FP8 QKV/Z kernel only for M=1 and the exact ordered
// [10240, 5120] then [6144, 5120] shapes. Every other valid combination,
// including unaligned exact-shape inputs, preserves the existing
// first-then-second tile dispatch and its numerical behavior.
[[nodiscard]] int launch_projection_pair_tile_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight, const std::uint16_t* input,
    std::size_t token_count, float* fp32_scratch,
    std::size_t scratch_elements, std::uint16_t* first_output,
    std::uint16_t* second_output, void* cuda_stream = nullptr) noexcept;

// Validates and launches the single-token dense-MLP gate and up projections,
// then overwrites gate_output with
// BF16(SiLU(BF16(gate)) * BF16(up)). up_output retains the independently
// rounded up projection. The exact aligned SM87 NVFP4 checkpoint shape uses
// the dedicated fused kernel. Every other valid combination preserves the
// existing ordered gate projection, up projection, and reference SiLU
// launches. All arguments and cross-projection aliases are validated before
// any work is enqueued.
[[nodiscard]] int launch_mlp_gate_up_silu_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight, const std::uint16_t* input,
    float* fp32_scratch, std::size_t scratch_elements,
    std::uint16_t* gate_output, std::uint16_t* up_output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime
