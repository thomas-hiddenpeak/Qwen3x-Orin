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
inline constexpr std::size_t kFp8M1OutputProjectionRows = 5'120U;
inline constexpr std::size_t kFp8M1OutputProjectionColumns = 6'144U;
inline constexpr std::size_t
    kFp8M1OutputProjectionAosoa4PreswizzledBytesPerLayer =
        kFp8M1OutputProjectionRows * kFp8M1OutputProjectionColumns;
inline constexpr std::size_t
    kQwen36Fp8M1OutputProjectionAosoa4PreswizzledBytes =
        kQwen36DenseLayerCount *
        kFp8M1OutputProjectionAosoa4PreswizzledBytesPerLayer;
inline constexpr std::size_t kNvFp4DownScale6Rows = 5'120U;
inline constexpr std::size_t kNvFp4DownScale6Columns = 17'408U;
inline constexpr std::size_t
    kNvFp4DownScale6SidecarBytesPerProjection = 4'177'920U;

// One descriptor for an exact Decode down-projection scale6 sidecar. The
// descriptor array must remain valid only for the attach call because its
// fields are copied. The pointed-to arena is non-owning and must outlive
// ModelWeights and every queued kernel that consumes an attached sidecar.
struct NvFp4DownScale6SidecarDescriptor {
  std::size_t layer_index = 0U;
  const std::uint8_t* sidecar = nullptr;
  std::size_t bytes = 0U;
  unsigned int scale_base = 0U;
  std::size_t output_size = 0U;
  std::size_t input_size = 0U;
};

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
  const std::uint8_t* m1_aosoa4_preswizzled_weight = nullptr;
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
  const std::uint8_t* down_scale6_sidecar = nullptr;
  unsigned int down_scale6_base = 0U;
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

// The projection dispatcher accepts two production prefill tiles at once.
// Request scheduling may impose a smaller limit independently; keeping this
// contract local to projection dispatch prevents the low-level composite
// route from silently inheriting the request scheduler's current chunk size.
inline constexpr std::size_t kMaximumProjectionTileTokenCount = 32U;

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

// True only for the exact M=1 step-path linear-attention four-projection
// group: FP8 QKV/Z followed by BF16 A/B on the explicitly selected SM87
// backend. This includes Decode and any C1/finish-prefill delegation to
// ReferenceRunner::step(). Types, payloads, and shapes are checked here;
// pointer alignment and all cross-output ranges remain launch-time concerns.
[[nodiscard]] bool supports_linear_attention_qkv_z_ab_projection_fusion(
    ProjectionBackend backend, const LinearWeight& qkv_weight,
    const LinearWeight& z_weight, const LinearWeight& a_weight,
    const LinearWeight& b_weight) noexcept;

// True only for the production full-attention Q/K/V fusion ABI: the
// explicitly selected SM87 backend followed by valid FP8 [12288, 5120],
// [1024, 5120], and [1024, 5120] matrices in Q, K, V order. This is
// shape/type/payload eligibility only; launch-time pointer alignment, ranges,
// scratch, and aliases remain the dispatcher's responsibility.
[[nodiscard]] bool supports_fp8_q_kv_projection_fusion(
    ProjectionBackend backend, const LinearWeight& q_weight,
    const LinearWeight& key_weight,
    const LinearWeight& value_weight) noexcept;

// True only for the production dense-MLP gate/up/SiLU fusion ABI: the
// explicitly selected SM87 backend followed by valid NVFP4 [17408, 5120]
// gate and up matrices, in that order. This is shape/type/payload eligibility
// only; launch-time alignment, input, output, scratch, range, and alias
// validation remains the dispatcher's responsibility.
[[nodiscard]] bool supports_nvfp4_gate_up_silu_fusion(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight) noexcept;

// True only for the production dense-MLP down projection followed by the
// decoder residual/centered-RMSNorm boundary: the explicitly selected SM87
// backend and one valid NVFP4 [5120, 17408] matrix. This is shape/type/payload
// eligibility only; launch-time alignment, ranges, scratch, residual/norm
// operands, outputs, and aliases remain the dispatcher's responsibility.
[[nodiscard]] bool supports_nvfp4_down_residual_norm_fusion(
    ProjectionBackend backend, const LinearWeight& down_weight) noexcept;

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

  // Attaches one contiguous, 16-byte-aligned, non-owning AoSoA4 sidecar view
  // per decoder layer. Every attention output projection must already be an
  // exact FP8 [5120,6144] binding and bytes must equal the complete 64-layer
  // sidecar contract. All inputs are validated before any view is changed, so
  // failure preserves every previously attached pointer. The sidecar arena
  // must outlive this ModelWeights view and all queued kernels using it.
  [[nodiscard]] bool attach_fp8_m1_output_projection_sidecars(
      const std::uint8_t* arena, std::size_t bytes) noexcept;

  // Atomically attaches a sparse set of exact [5120,17408] NVFP4 down
  // scale6 sidecars. descriptor_count is derived from checkpoint eligibility
  // rather than fixed to a model-wide count. Every descriptor, target down
  // projection, 32-byte alignment, scale base, sidecar range, and arena span
  // is validated before any existing attachment is changed. On success the
  // supplied set replaces all prior down-scale6 attachments; on failure the
  // previous set remains intact. Descriptor fields are copied during the
  // call; only the arena must outlive this view and every queued kernel using
  // an attached pointer.
  [[nodiscard]] bool attach_nvfp4_down_scale6_sidecars(
      const std::uint8_t* arena, std::size_t arena_bytes,
      const NvFp4DownScale6SidecarDescriptor* descriptors,
      std::size_t descriptor_count) noexcept;

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
// token_count must be in [1, 32]. M=1 delegates to the single-token entry
// point above. At M=17 and M=19..31, the two exact aligned NVFP4 MLP shapes
// use one runtime-masked M32 kernel that reads and writes exactly token_count
// rows. M=18 retains its fixed masked-M32 specialization. Other NVFP4 shapes
// or alignments in M=16..31 launch a fixed M16 prefix followed by launches of
// at most eight tokens for the remaining tail. At M=32, the exact aligned FP8
// production shapes and the two exact aligned NVFP4 MLP shapes use one fixed-
// M32 kernel; every other weight-only case uses two ordered M16 launches.
// M=2..15 uses the same at-most-eight-token fused launches. The reference
// backend and BF16 weights enqueue the existing FP32-scratch reference path
// in token order while reusing the same output-sized buffer; only M=1 may
// select the exact-shape BF16 direct-output route described above. The
// complete tile is validated before any work is enqueued.
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
// supports_bf16_projection_pair(...) selects the fused SM87 BF16 A/B route in
// subtiles of at most 16 tokens. An exact M16 subtile uses one fixed-shape
// projection-fused kernel; M1..M15 retain the generic pair kernel. C17..C32 is
// validated as one complete operation before its M16 prefix and generic tail
// (or two M16 subtiles for C32) are enqueued.
// supports_fp8_projection_pair(...) selects the fused SM87 FP8 K/V kernel only
// for M=1.
// supports_fp8_qkv_z_projection_pair(...) selects the fused SM87 FP8 QKV/Z
// kernel only for M=1 and the exact ordered [10240, 5120] then [6144, 5120]
// shapes. At M=17..31, a pair of exact aligned NVFP4 MLP projections bypasses
// pair sub-tiling so each projection can use its single masked-M32 kernel
// (M=18 retains the fixed specialization). Every other valid combination,
// including unaligned exact-shape inputs, preserves the existing tile
// dispatch and its numerical behavior.
[[nodiscard]] int launch_projection_pair_tile_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& first_weight,
    const LinearWeight& second_weight, const std::uint16_t* input,
    std::size_t token_count, float* fp32_scratch,
    std::size_t scratch_elements, std::uint16_t* first_output,
    std::uint16_t* second_output, void* cuda_stream = nullptr) noexcept;

// Launches the exact M=1 linear-attention QKV/Z/A/B group as one SM87 kernel.
// Unsupported backend, types, shapes, or fusion-only QKV/Z-weight/activation
// alignment return cudaErrorNotSupported before enqueue so the runner can
// retain its established two-call fallback. Structurally eligible calls with
// invalid payloads, BF16 alignment, null pointers, ranges, or aliases return
// cudaErrorInvalidValue and must not be retried through that fallback.
[[nodiscard]] int launch_linear_attention_qkv_z_ab_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& qkv_weight,
    const LinearWeight& z_weight, const LinearWeight& a_weight,
    const LinearWeight& b_weight, const std::uint16_t* input,
    std::uint16_t* qkv_output, std::uint16_t* z_output,
    std::uint16_t* a_output, std::uint16_t* b_output,
    void* cuda_stream = nullptr) noexcept;

// Validates and launches the single-token full-attention Q, K, and V
// projections over one shared activation. The exact aligned SM87 FP8
// [12288, 5120] Q plus paired [1024, 5120] K/V route uses one kernel. Every
// other valid combination preserves the existing ordered Q projection then
// K/V pair dispatch. The complete three-projection operation, including
// cross-weight/output/scratch aliases, is rejected before its first enqueue.
[[nodiscard]] int launch_full_attention_q_kv_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& q_weight,
    const LinearWeight& key_weight, const LinearWeight& value_weight,
    const std::uint16_t* input, float* fp32_scratch,
    std::size_t scratch_elements, std::uint16_t* q_output,
    std::uint16_t* key_output, std::uint16_t* value_output,
    void* cuda_stream = nullptr) noexcept;

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

// Fuses the post-attention residual add and centered RMSNorm with the exact
// aligned SM87 NVFP4 M=1 gate/up/SiLU path. residual_right_and_normalized is
// read as the residual's right operand; the fused path consumes its normalized
// value only in CTA-local shared memory, while every fallback writes the
// normalized BF16 vector back to that buffer before launching the existing
// MLP chain. residual_output always receives the rounded residual. All
// unsupported valid combinations preserve the previous two-call ordering.
[[nodiscard]] int
launch_post_attention_residual_norm_mlp_gate_up_silu_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight,
    const std::uint16_t* residual_left,
    std::uint16_t* residual_right_and_normalized,
    const std::uint16_t* norm_weight, float epsilon,
    float* fp32_scratch, std::size_t scratch_elements,
    std::uint16_t* residual_output,
    std::uint16_t* gate_output, std::uint16_t* up_output,
    void* cuda_stream = nullptr) noexcept;

// Decode-runner-only form of the operation above. The exact aligned SM87
// NVFP4 M=1 route keeps rounded gate/up intermediates CTA-local, publishes
// residual_output and final gate_output, and leaves up_workspace untouched.
// Every fallback preserves the existing implementation and may use/write
// up_workspace; callers must treat it as dead until it is overwritten.
// Validation and alias rules remain identical to the generic double-output
// API. This explicit boundary must not be used when rounded up is observable.
[[nodiscard]] int
launch_decode_runner_post_attention_residual_norm_mlp_gate_up_silu_dead_up_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& gate_weight,
    const LinearWeight& up_weight,
    const std::uint16_t* residual_left,
    std::uint16_t* residual_right_and_normalized,
    const std::uint16_t* norm_weight, float epsilon,
    float* fp32_scratch, std::size_t scratch_elements,
    std::uint16_t* residual_output,
    std::uint16_t* gate_output, std::uint16_t* up_workspace,
    void* cuda_stream = nullptr) noexcept;

// Validates and launches a single-token dense-MLP down projection followed by
// the decoder residual add and centered RMSNorm boundary. The exact aligned
// SM87 NVFP4 [5120, 17408] route uses one cooperative kernel. Every other
// valid weight/backend combination preserves the existing ordered down
// projection then residual/norm launches. raw_down_output, residual_output,
// and normalized_output are three distinct BF16 boundaries; keeping raw down
// separate preserves decode trace observability. The complete operation,
// including all weight payloads/device scalars and fallback scratch, is
// rejected before its first enqueue. Read-only operands may overlap each
// other, but no writable range may overlap any other operand.
[[nodiscard]] int launch_mlp_down_residual_norm_to_bf16_cuda(
    ProjectionBackend backend, const LinearWeight& down_weight,
    const std::uint16_t* activation,
    const std::uint16_t* residual_left,
    const std::uint16_t* norm_weight, float epsilon,
    float* fp32_scratch, std::size_t scratch_elements,
    std::uint16_t* raw_down_output,
    std::uint16_t* residual_output,
    std::uint16_t* normalized_output,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::runtime
