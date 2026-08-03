#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Projection-span producer/consumer handoff for the authenticated R1 path.
// Both launchers preserve the incumbent BF16 residual and centered-RMS
// boundaries while publishing the next projection's canonical A4 input in
// the same CTA.  The first boundary publishes the lane-count-one R1 Gate
// input.  The second publishes the K256 Attention input for the next layer.
inline constexpr std::size_t kSm87A4W4PrefillHandoffHiddenSize = 5'120U;
inline constexpr std::size_t kSm87A4W4PrefillHandoffThreads = 512U;
inline constexpr std::size_t kSm87A4W4PrefillHandoffMaximumTokens = 4'096U;
inline constexpr std::size_t kSm87A4W4PrefillHandoffMinimumBlocksPerSm = 2U;
inline constexpr std::size_t kSm87A4W4PrefillHandoffMaximumRegisters = 128U;

struct Sm87A4W4PrefillHandoffResources final {
  int attention_to_gate_registers_per_thread{};
  int mlp_to_attention_registers_per_thread{};
  std::size_t attention_to_gate_static_shared_bytes{};
  std::size_t mlp_to_attention_static_shared_bytes{};
  std::size_t attention_to_gate_local_bytes{};
  std::size_t mlp_to_attention_local_bytes{};
  int attention_to_gate_active_blocks_per_sm{};
  int mlp_to_attention_active_blocks_per_sm{};
  int maximum_threads_per_block{};
  int compute_major{};
  int compute_minor{};
  int multiprocessor_count{};
};

[[nodiscard]] int query_sm87_a4w4_prefill_handoff_resources_cuda(
    Sm87A4W4PrefillHandoffResources* resources) noexcept;

// Computes BF16(left + right), publishes that rounded residual, applies the
// exact centered RMSNorm reduction used by the incumbent prompt-span path,
// rounds the normalized values to BF16, then applies authenticated R1
// inverse-alpha and lane-count-one quantization.  Padded rows are zero-code / 
// BF16-one-scale and never read left, right, norm_weight, or inverse_alpha.
[[nodiscard]] int
launch_sm87_a4w4_attention_residual_post_norm_r1_quantize_cuda(
    const std::uint16_t* left_bf16,
    const std::uint16_t* right_bf16,
    const std::uint16_t* centered_norm_weight_bf16,
    const float* authenticated_inverse_alpha_fp32,
    std::size_t inverse_alpha_capacity_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    float epsilon,
    float clip_ratio,
    std::uint16_t* residual_output_bf16,
    std::size_t residual_output_capacity_elements,
    std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Computes and publishes the MLP residual, applies the next layer's exact
// centered RMSNorm, publishes that rounded BF16 row for Linear-A/B, and in
// the same CTA emits the next Attention projection's canonical K256 A4 input.
// Padded rows are zero-code / BF16-one-scale and never read any BF16 input.
[[nodiscard]] int
launch_sm87_a4w4_mlp_residual_next_norm_k256_quantize_cuda(
    const std::uint16_t* left_bf16,
    const std::uint16_t* right_bf16,
    const std::uint16_t* next_centered_norm_weight_bf16,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    float epsilon,
    float clip_ratio,
    std::uint16_t* residual_output_bf16,
    std::size_t residual_output_capacity_elements,
    std::uint16_t* normalized_output_bf16,
    std::size_t normalized_output_capacity_elements,
    std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_k256_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels
