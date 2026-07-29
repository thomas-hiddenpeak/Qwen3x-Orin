#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// One host-side partition descriptor for the test-only exact-C512 FP8
// projection-supermatrix launcher. The launcher copies these values into the
// CUDA kernel argument packet synchronously; the descriptor array itself need
// not remain alive after the call. Device payloads must remain valid until the
// queued kernel retires.
struct Sm87Fp8PrefillSupermatrixPartition {
  const std::uint8_t* register_feed_sidecar = nullptr;
  float weight_scale = 0.0F;
  std::size_t rows = 0U;
  std::uint16_t* output = nullptr;
};

// Builds an exact equal-byte M64xN256 fragment-native permutation from one
// canonical row-major E4M3FN matrix. Rows must be N256 aligned and columns
// K64 aligned. This is engine-load work and never mutates the checkpoint.
[[nodiscard]] int launch_sm87_fp8_prefill_supermatrix_pack_cuda(
    const std::uint8_t* canonical_weights, std::uint8_t* sidecar_weights,
    std::size_t rows, std::size_t columns, void* cuda_stream) noexcept;

// Executes one complete same-input projection group. C512 is exact; columns
// are currently restricted to the checkpoint's K5120 input projections or
// K6144 attention-output projections. One to three N256-aligned partitions
// retain independent scales, row strides, and output addresses.
[[nodiscard]] int launch_sm87_fp8_prefill_supermatrix_gemm_bf16_cuda(
    const Sm87Fp8PrefillSupermatrixPartition* partitions,
    std::size_t partition_count, const std::uint16_t* activations,
    std::size_t token_count, std::size_t columns,
    void* cuda_stream) noexcept;

}  // namespace q3x::kernels
