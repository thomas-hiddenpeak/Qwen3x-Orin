#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_K512_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_K512_HOST_DEVICE
#endif

namespace q3x::kernels {

// Attention-O complete cell for an independent K512 quantization contract.
// It is deliberately not ABI-compatible with the authenticated K128 path:
// both packed operands and both scale planes must be generated under this
// header's ABI.  Runtime selection remains explicit and fail-closed.
//
// One 256-thread CTA owns M128N64.  Its eight warps each retain M16N64:
// 32 FP32 output accumulators/thread and 32 S32 partials/thread.  A physical
// copy stage contains four K64 code planes (K256) for both operands.  Two
// stages form a 49,152-byte ping-pong ring and one logical K512 scale group.
// Scales never occupy shared memory.
//
// The numerical oracle is, in ascending logical group order:
//
//   P_s32(m,n,g) = sum_{k in K512(g)} A_s4(m,k) * B_s4(n,k)
//   scale_product = round_fp32(A_bf16_scale(m,g) * B_bf16_scale(n,g))
//   C_fp32 = fma_rn(float(P_s32), scale_product, C_fp32)
//
// Exactly one FP32 FMA is issued per output and K512 group.  In particular,
// neither a K128 nor a K256 partial may be dequantized independently.
inline constexpr std::size_t kSm87A4W4AttentionOK512TileM = 128U;
inline constexpr std::size_t kSm87A4W4AttentionOK512TileN = 64U;
inline constexpr std::size_t kSm87A4W4AttentionOK512CopyK = 256U;
inline constexpr std::size_t kSm87A4W4AttentionOK512ScaleK = 512U;
inline constexpr std::size_t kSm87A4W4AttentionOK512PhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4AttentionOK512K64PerCopy = 4U;
inline constexpr std::size_t kSm87A4W4AttentionOK512CopiesPerScale = 2U;
inline constexpr std::size_t kSm87A4W4AttentionOK512Threads = 256U;
inline constexpr std::size_t kSm87A4W4AttentionOK512Warps = 8U;
inline constexpr std::size_t kSm87A4W4AttentionOK512Stages = 2U;
inline constexpr std::size_t kSm87A4W4AttentionOK512PackedRowK64Bytes = 32U;
inline constexpr std::size_t kSm87A4W4AttentionOK512OuterBlock = 64U;
inline constexpr std::size_t kSm87A4W4AttentionOK512StageABytes =
    kSm87A4W4AttentionOK512TileM * kSm87A4W4AttentionOK512CopyK / 2U;
inline constexpr std::size_t kSm87A4W4AttentionOK512StageBBytes =
    kSm87A4W4AttentionOK512TileN * kSm87A4W4AttentionOK512CopyK / 2U;
inline constexpr std::size_t kSm87A4W4AttentionOK512StageBytes =
    kSm87A4W4AttentionOK512StageABytes +
    kSm87A4W4AttentionOK512StageBBytes;
inline constexpr std::size_t kSm87A4W4AttentionOK512SharedBytes =
    kSm87A4W4AttentionOK512Stages * kSm87A4W4AttentionOK512StageBytes;
inline constexpr std::size_t kSm87A4W4AttentionOK512PersistentCtas = 32U;
inline constexpr std::size_t kSm87A4W4AttentionOK512CtasPerSm = 2U;
inline constexpr std::size_t kSm87A4W4AttentionOK512MaximumRegisters = 128U;

[[nodiscard]] constexpr bool sm87_a4w4_attention_o_k512_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_o_k512_outer_block_count(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) /
                        kSm87A4W4AttentionOK512OuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_o_k512_k64_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4AttentionOK512PhysicalK64 == 0U
             ? logical_k / kSm87A4W4AttentionOK512PhysicalK64
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_o_k512_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4AttentionOK512ScaleK == 0U
             ? logical_k / kSm87A4W4AttentionOK512ScaleK
             : 0U;
}

// Packed S4 codes retain the consumer-block physical layout
// [outer/64][K/64][64][32].  This equal-byte code layout is shared only as a
// physical transport convention; the scale plane and numerical contract are
// new K512 data and must be produced by independent requantization.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_o_k512_packed_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_attention_o_k512_outer_block_count(outer_count);
  const std::size_t groups =
      sm87_a4w4_attention_o_k512_k64_group_count(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_attention_o_k512_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  constexpr std::size_t group_bytes =
      kSm87A4W4AttentionOK512OuterBlock *
      kSm87A4W4AttentionOK512PackedRowK64Bytes;
  return sm87_a4w4_attention_o_k512_product_fits(block_groups,
                                                  group_bytes)
             ? block_groups * group_bytes
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_K512_HOST_DEVICE constexpr std::size_t
sm87_a4w4_attention_o_k512_packed_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t byte_in_k64,
    const std::size_t k64_group_count) noexcept {
  return (((outer_coordinate / kSm87A4W4AttentionOK512OuterBlock) *
               k64_group_count +
           k64_group) *
              kSm87A4W4AttentionOK512OuterBlock +
          outer_coordinate % kSm87A4W4AttentionOK512OuterBlock) *
             kSm87A4W4AttentionOK512PackedRowK64Bytes +
         byte_in_k64;
}

// New scale layout: [outer/64][K/512][64] BF16 elements.  These values are
// not derived by pooling or otherwise reinterpreting the K128 scale plane.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_o_k512_scale_capacity_elements(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_attention_o_k512_outer_block_count(outer_count);
  const std::size_t groups =
      sm87_a4w4_attention_o_k512_group_count(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_attention_o_k512_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_attention_o_k512_product_fits(
             block_groups, kSm87A4W4AttentionOK512OuterBlock)
             ? block_groups * kSm87A4W4AttentionOK512OuterBlock
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_K512_HOST_DEVICE constexpr std::size_t
sm87_a4w4_attention_o_k512_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  return ((outer_coordinate / kSm87A4W4AttentionOK512OuterBlock) *
              k512_group_count +
          k512_group) *
             kSm87A4W4AttentionOK512OuterBlock +
         outer_coordinate % kSm87A4W4AttentionOK512OuterBlock;
}

struct Sm87A4W4AttentionOK512Plan final {
  std::size_t token_count{};
  std::size_t output_size{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k256_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_tiles{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4AttentionOK512Plan
sm87_a4w4_attention_o_k512_plan(
    const std::size_t token_count, const std::size_t output_size,
    const std::size_t input_size) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4AttentionOK512TileM != 0U ||
      output_size == 0U ||
      output_size % kSm87A4W4AttentionOK512TileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4AttentionOK512ScaleK != 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4AttentionOK512TileM;
  const std::size_t n_tiles =
      output_size / kSm87A4W4AttentionOK512TileN;
  if (!sm87_a4w4_attention_o_k512_product_fits(m_tiles, n_tiles)) {
    return {};
  }
  const std::size_t work_tiles = m_tiles * n_tiles;
  return {token_count,
          output_size,
          input_size,
          m_tiles,
          n_tiles,
          input_size / kSm87A4W4AttentionOK512ScaleK,
          input_size / kSm87A4W4AttentionOK512CopyK,
          input_size / kSm87A4W4AttentionOK512PhysicalK64,
          work_tiles,
          work_tiles < kSm87A4W4AttentionOK512PersistentCtas
              ? work_tiles
              : kSm87A4W4AttentionOK512PersistentCtas};
}

struct Sm87A4W4AttentionOK512Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int query_sm87_a4w4_attention_o_k512_resources_cuda(
    Sm87A4W4AttentionOK512Resources* resources) noexcept;

// Production launcher for independently generated K512 A and B operands.
// The kernel has no tails: M, N, and K must satisfy the complete-cell plan,
// and row-stride padding is allowed only for output.  Capacities are expressed
// in bytes for packed S4 codes and in elements for BF16 scales/output.  The
// output range must not alias any input range.  Invalid arguments are rejected
// before querying the active device or enqueueing work; non-SM87/non-16-SM
// targets return cudaErrorNotSupported represented as int.
[[nodiscard]] int launch_sm87_a4w4_attention_o_k512_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Correctness-only entry point exposing a smaller persistent-CTA cap.  Its
// data, shape, capacity, target, and alias contract is otherwise identical to
// the production launcher.
[[nodiscard]] int launch_sm87_a4w4_attention_o_k512_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* packed_b,
    std::size_t packed_b_capacity_bytes,
    const std::uint16_t* b_k512_scales_bf16,
    std::size_t b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t output_size,
    std::size_t input_size,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas =
        static_cast<unsigned int>(kSm87A4W4AttentionOK512PersistentCtas),
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4AttentionOK512K64PerCopy *
                      kSm87A4W4AttentionOK512PhysicalK64 ==
                  kSm87A4W4AttentionOK512CopyK &&
              kSm87A4W4AttentionOK512CopiesPerScale *
                      kSm87A4W4AttentionOK512CopyK ==
                  kSm87A4W4AttentionOK512ScaleK);
static_assert(kSm87A4W4AttentionOK512StageABytes == 16'384U);
static_assert(kSm87A4W4AttentionOK512StageBBytes == 8'192U);
static_assert(kSm87A4W4AttentionOK512StageBytes == 24'576U);
static_assert(kSm87A4W4AttentionOK512SharedBytes == 49'152U);
static_assert(kSm87A4W4AttentionOK512SharedBytes *
                  kSm87A4W4AttentionOK512CtasPerSm <=
              96U * 1'024U);
static_assert(sm87_a4w4_attention_o_k512_plan(
                  2'048U, 5'120U, 6'144U)
                  .k512_groups == 12U);
static_assert(sm87_a4w4_attention_o_k512_plan(
                  2'048U, 5'120U, 6'144U)
                  .work_tiles == 1'280U);
static_assert(sm87_a4w4_attention_o_k512_scale_capacity_elements(
                  5'120U, 6'144U) == 61'440U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_K512_HOST_DEVICE
