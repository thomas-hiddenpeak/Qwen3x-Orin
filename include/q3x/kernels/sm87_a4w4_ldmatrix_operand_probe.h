#pragma once

#include <cstddef>
#include <cstdint>

namespace q3x::kernels {

// Compile/test-only admission for proving that an XOR-16B shared-memory
// layout can feed the native SM87 S4 MMA fragments directly through
// ldmatrix. It is deliberately not linked into q3x_kernels or exposed through
// a runner selector.
inline constexpr std::size_t kSm87A4W4LdmatrixProbeLogicalK = 512U;
inline constexpr std::size_t kSm87A4W4LdmatrixProbeK64Groups = 8U;
inline constexpr std::size_t kSm87A4W4LdmatrixProbeOuterBlock = 64U;
inline constexpr std::size_t kSm87A4W4LdmatrixProbePackedK64Bytes = 32U;
inline constexpr std::size_t kSm87A4W4LdmatrixProbePayloadBytes =
    kSm87A4W4LdmatrixProbeK64Groups *
    kSm87A4W4LdmatrixProbeOuterBlock *
    kSm87A4W4LdmatrixProbePackedK64Bytes;
inline constexpr std::size_t kSm87A4W4LdmatrixProbeAccumulatorWords =
    32U * 4U;

struct Sm87A4W4LdmatrixOperandProbeResources final {
  int scalar_registers_per_thread{};
  int ldmatrix_registers_per_thread{};
  int scalar_a_ldmatrix_b_registers_per_thread{};
  int ldmatrix_a_scalar_b_registers_per_thread{};
  std::size_t scalar_static_shared_bytes{};
  std::size_t ldmatrix_static_shared_bytes{};
  std::size_t scalar_a_ldmatrix_b_static_shared_bytes{};
  std::size_t ldmatrix_a_scalar_b_static_shared_bytes{};
  std::size_t scalar_local_bytes{};
  std::size_t ldmatrix_local_bytes{};
  std::size_t scalar_a_ldmatrix_b_local_bytes{};
  std::size_t ldmatrix_a_scalar_b_local_bytes{};
  int scalar_active_blocks_per_sm{};
  int ldmatrix_active_blocks_per_sm{};
  int scalar_a_ldmatrix_b_active_blocks_per_sm{};
  int ldmatrix_a_scalar_b_active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Both inputs use the production consumer K512 code-block layout
// [K64 group][outer=64][packed K64 bytes=32]. Only A rows 0..15 and B rows
// 0..7 are consumed by this one-warp M16N8 probe. Each launch writes the four
// raw S32 accumulator registers owned by every lane, with no epilogue. Four
// outputs independently cover scalar/scalar, LDSM/LDSM, scalar-A/LDSM-B, and
// LDSM-A/scalar-B. The mixed outputs prevent a matching A/B K permutation
// from falsely validating the fully-LDSM dot product.
//
// The probe proves both A ldmatrix.x4 and B ldmatrix.x2 fragment mappings.
// That does not prescribe one production dataflow: Gate keeps its admitted
// paired-B direct .ca v4 feed and can consume only the A-x4 result, whereas a
// Down skeleton can consume both A-x4 and B-x2.
[[nodiscard]] int launch_sm87_a4w4_ldmatrix_operand_probe_cuda(
    const std::uint8_t* packed_a_k512,
    std::size_t packed_a_capacity_bytes,
    const std::uint8_t* packed_b_k512,
    std::size_t packed_b_capacity_bytes,
    std::size_t k64_group,
    std::int32_t* scalar_accumulators,
    std::size_t scalar_accumulator_capacity_words,
    std::int32_t* ldmatrix_accumulators,
    std::size_t ldmatrix_accumulator_capacity_words,
    std::int32_t* scalar_a_ldmatrix_b_accumulators,
    std::size_t scalar_a_ldmatrix_b_accumulator_capacity_words,
    std::int32_t* ldmatrix_a_scalar_b_accumulators,
    std::size_t ldmatrix_a_scalar_b_accumulator_capacity_words,
    void* cuda_stream) noexcept;

[[nodiscard]] int query_sm87_a4w4_ldmatrix_operand_probe_resources_cuda(
    Sm87A4W4LdmatrixOperandProbeResources* resources) noexcept;

}  // namespace q3x::kernels
