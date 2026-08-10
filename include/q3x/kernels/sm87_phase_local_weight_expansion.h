#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Test-only expansion of the exact Qwen3.6-27B checkpoint layouts into a
// phase-local canonical BF16 [N,K] scratch matrix.  This surface is disabled
// by default, has no production dispatcher, and deliberately excludes the
// original tensor's FP32 global scale from the expansion.  The downstream
// GEMM must consume that same FP32 scalar separately.
enum class Sm87PhaseLocalWeightFormat : std::uint8_t {
  kNvFp4ModelOptBlock16 = 0U,
  kFp8E4M3Fn = 1U,
};

enum class Sm87PhaseLocalWeightRole : std::uint8_t {
  kNvFp4Gate = 0U,
  kNvFp4Up = 1U,
  kNvFp4Down = 2U,
  kFp8LinearQkv = 3U,
  kFp8LinearZ = 4U,
  kFp8LinearOutput = 5U,
  kFp8FullQuery = 6U,
  kFp8FullKey = 7U,
  kFp8FullValue = 8U,
  kFp8FullOutput = 9U,
};

[[nodiscard]] constexpr int sm87_phase_local_weight_expected_registers(
    const Sm87PhaseLocalWeightRole role) noexcept {
  switch (role) {
    case Sm87PhaseLocalWeightRole::kNvFp4Gate:
    case Sm87PhaseLocalWeightRole::kNvFp4Up:
    case Sm87PhaseLocalWeightRole::kNvFp4Down:
      return 32;
    case Sm87PhaseLocalWeightRole::kFp8LinearOutput:
    case Sm87PhaseLocalWeightRole::kFp8FullOutput:
      return 22;
    case Sm87PhaseLocalWeightRole::kFp8LinearQkv:
    case Sm87PhaseLocalWeightRole::kFp8LinearZ:
    case Sm87PhaseLocalWeightRole::kFp8FullQuery:
    case Sm87PhaseLocalWeightRole::kFp8FullKey:
    case Sm87PhaseLocalWeightRole::kFp8FullValue:
      return 20;
  }
  return 0;
}

inline constexpr std::size_t kSm87PhaseLocalWeightThreads = 256U;
inline constexpr std::size_t kSm87PhaseLocalNvFp4BlockSize = 16U;
inline constexpr std::size_t kSm87PhaseLocalNvFp4ValuesPerByte = 2U;
inline constexpr std::size_t kSm87PhaseLocalBf16Bytes = 2U;
inline constexpr std::size_t kSm87PhaseLocalFp32GlobalScaleBytes = 4U;
inline constexpr int kSm87PhaseLocalWeightExpectedActiveBlocksPerSm = 6;

struct Sm87PhaseLocalWeightShape {
  Sm87PhaseLocalWeightFormat format =
      Sm87PhaseLocalWeightFormat::kNvFp4ModelOptBlock16;
  std::size_t output_features = 0U;
  std::size_t input_features = 0U;
  bool supported = false;
};

[[nodiscard]] constexpr Sm87PhaseLocalWeightShape
sm87_phase_local_weight_shape(
    const Sm87PhaseLocalWeightRole role) noexcept {
  switch (role) {
    case Sm87PhaseLocalWeightRole::kNvFp4Gate:
    case Sm87PhaseLocalWeightRole::kNvFp4Up:
      return {Sm87PhaseLocalWeightFormat::kNvFp4ModelOptBlock16, 17'408U,
              5'120U, true};
    case Sm87PhaseLocalWeightRole::kNvFp4Down:
      return {Sm87PhaseLocalWeightFormat::kNvFp4ModelOptBlock16, 5'120U,
              17'408U, true};
    case Sm87PhaseLocalWeightRole::kFp8LinearQkv:
      return {Sm87PhaseLocalWeightFormat::kFp8E4M3Fn, 10'240U, 5'120U,
              true};
    case Sm87PhaseLocalWeightRole::kFp8LinearZ:
      return {Sm87PhaseLocalWeightFormat::kFp8E4M3Fn, 6'144U, 5'120U,
              true};
    case Sm87PhaseLocalWeightRole::kFp8LinearOutput:
      return {Sm87PhaseLocalWeightFormat::kFp8E4M3Fn, 5'120U, 6'144U,
              true};
    case Sm87PhaseLocalWeightRole::kFp8FullQuery:
      return {Sm87PhaseLocalWeightFormat::kFp8E4M3Fn, 12'288U, 5'120U,
              true};
    case Sm87PhaseLocalWeightRole::kFp8FullKey:
    case Sm87PhaseLocalWeightRole::kFp8FullValue:
      return {Sm87PhaseLocalWeightFormat::kFp8E4M3Fn, 1'024U, 5'120U,
              true};
    case Sm87PhaseLocalWeightRole::kFp8FullOutput:
      return {Sm87PhaseLocalWeightFormat::kFp8E4M3Fn, 5'120U, 6'144U,
              true};
  }
  return {};
}

// Callers must make every admission property explicit.  Value initialization
// is intentionally invalid, so merely linking the test surface cannot enable
// it.
struct Sm87PhaseLocalWeightExpansionCapability {
  int compute_major = 0;
  int compute_minor = 0;
  bool test_only_opt_in = false;
  bool canonical_checkpoint_layout = false;
  bool phase_local_scratch = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return compute_major == 8 && compute_minor == 7 && test_only_opt_in &&
           canonical_checkpoint_layout && phase_local_scratch;
  }
};

struct Sm87PhaseLocalWeightExpansionPlan {
  Sm87PhaseLocalWeightFormat format =
      Sm87PhaseLocalWeightFormat::kNvFp4ModelOptBlock16;
  Sm87PhaseLocalWeightRole role = Sm87PhaseLocalWeightRole::kNvFp4Gate;
  std::size_t output_features = 0U;
  std::size_t input_features = 0U;
  std::size_t canonical_weight_bytes = 0U;
  std::size_t block_scale_bytes = 0U;
  std::size_t fp32_global_scale_bytes = 0U;
  std::size_t scratch_elements = 0U;
  std::size_t scratch_bytes = 0U;
  std::size_t blocks = 0U;
  std::size_t threads = 0U;
  bool canonical_row_major = false;
  bool phase_local = false;
  bool test_only = false;
  bool global_scale_baked = true;

  [[nodiscard]] constexpr bool valid() const noexcept {
    const auto shape = sm87_phase_local_weight_shape(role);
    if (!shape.supported || format != shape.format ||
        output_features != shape.output_features ||
        input_features != shape.input_features) {
      return false;
    }
    const std::size_t elements =
        shape.output_features * shape.input_features;
    const bool nvfp4 =
        shape.format == Sm87PhaseLocalWeightFormat::kNvFp4ModelOptBlock16;
    return canonical_weight_bytes ==
               (nvfp4 ? elements / kSm87PhaseLocalNvFp4ValuesPerByte
                       : elements) &&
           block_scale_bytes ==
               (nvfp4 ? elements / kSm87PhaseLocalNvFp4BlockSize : 0U) &&
           fp32_global_scale_bytes ==
               kSm87PhaseLocalFp32GlobalScaleBytes &&
           scratch_elements == elements &&
           scratch_bytes == elements * kSm87PhaseLocalBf16Bytes &&
           blocks == shape.output_features &&
           threads == kSm87PhaseLocalWeightThreads &&
           canonical_row_major && phase_local && test_only &&
           !global_scale_baked;
  }
};

struct Sm87PhaseLocalWeightExpansionResources {
  Sm87PhaseLocalWeightRole role = Sm87PhaseLocalWeightRole::kNvFp4Gate;
  int compute_major = 0;
  int compute_minor = 0;
  int binary_version = 0;
  int registers_per_thread = 0;
  std::size_t static_shared_bytes = 0U;
  std::size_t dynamic_shared_bytes = 0U;
  std::size_t local_bytes = 0U;
  int active_blocks_per_sm = 0;
  bool admitted = false;

  [[nodiscard]] constexpr bool valid() const noexcept {
    return sm87_phase_local_weight_shape(role).supported && admitted &&
           compute_major == 8 && compute_minor == 7 &&
           binary_version == 87 &&
           registers_per_thread ==
               sm87_phase_local_weight_expected_registers(role) &&
           static_shared_bytes == 0U && dynamic_shared_bytes == 0U &&
           local_bytes == 0U &&
           active_blocks_per_sm ==
               kSm87PhaseLocalWeightExpectedActiveBlocksPerSm;
  }
};

[[nodiscard]] constexpr Sm87PhaseLocalWeightExpansionPlan
sm87_phase_local_weight_expansion_plan(
    const Sm87PhaseLocalWeightRole role,
    const Sm87PhaseLocalWeightExpansionCapability capability) noexcept {
  const auto shape = sm87_phase_local_weight_shape(role);
  if (!capability.valid() || !shape.supported) {
    return {};
  }
  const std::size_t elements =
      shape.output_features * shape.input_features;
  const bool nvfp4 =
      shape.format == Sm87PhaseLocalWeightFormat::kNvFp4ModelOptBlock16;
  return {shape.format,
          role,
          shape.output_features,
          shape.input_features,
          nvfp4 ? elements / kSm87PhaseLocalNvFp4ValuesPerByte : elements,
          nvfp4 ? elements / kSm87PhaseLocalNvFp4BlockSize : 0U,
          kSm87PhaseLocalFp32GlobalScaleBytes,
          elements,
          elements * kSm87PhaseLocalBf16Bytes,
          shape.output_features,
          kSm87PhaseLocalWeightThreads,
          true,
          true,
          true,
          false};
}

// Host-visible exact decoding witnesses.  NVFP4 returns BF16(E2M1 * E4M3FN
// block scale); FP8 returns BF16(E4M3FN).  Neither function accepts or applies
// the tensor's FP32 global scale.
[[nodiscard]] std::uint16_t
sm87_phase_local_nvfp4_expanded_bf16_reference(
    std::uint8_t packed_weight, bool high_nibble,
    std::uint8_t block_scale) noexcept;

[[nodiscard]] std::uint16_t
sm87_phase_local_fp8_expanded_bf16_reference(
    std::uint8_t weight) noexcept;

// Freezes the exact SM87 cubin/resource inventory for an expansion role. It
// does not allocate, launch, or select a fallback implementation.
int query_sm87_phase_local_weight_expansion_resources_cuda(
    Sm87PhaseLocalWeightRole role,
    Sm87PhaseLocalWeightExpansionResources* resources) noexcept;

// CUDA-facing functions return cudaError_t encoded as int, keeping this
// header CUDA-neutral.  They accept canonical checkpoint data and write a
// canonical row-major BF16 [N,K] scratch.  The original FP32 global scale is
// intentionally absent and must be forwarded unchanged to the next GEMM.
int launch_sm87_phase_local_nvfp4_weight_expansion_test_cuda(
    const Sm87PhaseLocalWeightExpansionPlan& plan,
    const std::uint8_t* packed_weights,
    const std::uint8_t* block_scales,
    std::uint16_t* bf16_scratch,
    std::size_t scratch_bytes,
    void* cuda_stream) noexcept;

int launch_sm87_phase_local_fp8_weight_expansion_test_cuda(
    const Sm87PhaseLocalWeightExpansionPlan& plan,
    const std::uint8_t* weights,
    std::uint16_t* bf16_scratch,
    std::size_t scratch_bytes,
    void* cuda_stream) noexcept;

}  // namespace q3x::kernels
