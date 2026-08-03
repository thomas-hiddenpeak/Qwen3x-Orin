#pragma once

#include "q3x/kernels/sm87_a4w4_attention_k256_m128n256.h"

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_ATTENTION_R1_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_ATTENTION_R1_HOST_DEVICE
#endif

namespace q3x::kernels {

// Default-off Attention projection foundation for the factorized-lane R1
// package.  Codes retain the authenticated K256 consumer order
// [outer/64][K/64][64][32], but each operand owns one BF16 scale per outer
// row.  Consequently every output retains one whole-K S32 reduction and
// applies A/B scales exactly once before its sole BF16-RNE publication.
//
// The fixed launch surface deliberately preserves the incumbent Qwen3.6
// Linear-QKV/Z, Full-Q/K/V, and Attention-O topology/output contract.  It is
// isolated from every runtime selector; a later consumer-native package can
// replace the BF16 publication without changing the projection payload.
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1TileM = 128U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1TileN = 256U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1Threads = 256U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1Warps = 8U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1Stages = 3U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1StageBytes = 49'152U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1ScaleBytes = 768U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1SharedBytes = 148'224U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1MaximumRegisters = 255U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1PersistentCtas = 16U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1CtasPerSm = 1U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1V2MaximumRegisters = 252U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1V2PairN = 16U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1V2OuterBlock = 128U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1V2PairLanes = 32U;
inline constexpr std::size_t
    kSm87A4W4AttentionFactorizedLaneR1V2PairBytes = 16U;

// Equal-byte v2 B publication.  One aligned lane slot owns two adjacent N8
// IMMA fragments: {first.x0, first.x1, second.x0, second.x1}.  The complete
// layout is [N128][K64][N16][lane32][16B], shared with Down, so its capacity
// remains N*K/2.
[[nodiscard]] Q3X_SM87_A4W4_ATTENTION_R1_HOST_DEVICE constexpr std::size_t
sm87_a4w4_attention_factorized_lane_r1_v2_b_pair_offset(
    const std::size_t outer_coordinate,
    const std::size_t k64_group,
    const std::size_t lane,
    const std::size_t k64_group_count) noexcept {
  const std::size_t outer_block =
      outer_coordinate / kSm87A4W4AttentionFactorizedLaneR1V2OuterBlock;
  const std::size_t n16 =
      (outer_coordinate %
       kSm87A4W4AttentionFactorizedLaneR1V2OuterBlock) /
      kSm87A4W4AttentionFactorizedLaneR1V2PairN;
  return (((outer_block * k64_group_count + k64_group) *
               (kSm87A4W4AttentionFactorizedLaneR1V2OuterBlock /
                kSm87A4W4AttentionFactorizedLaneR1V2PairN) +
           n16) *
              kSm87A4W4AttentionFactorizedLaneR1V2PairLanes +
          lane) *
         kSm87A4W4AttentionFactorizedLaneR1V2PairBytes;
}

struct Sm87A4W4AttentionFactorizedLaneR1V2BWordCoordinate final {
  std::size_t outer{};
  std::size_t byte_in_k64{};
  bool valid{};
};

// Logical canonical source coordinate for one u32 in a v2 pair slot.
[[nodiscard]] constexpr
    Sm87A4W4AttentionFactorizedLaneR1V2BWordCoordinate
sm87_a4w4_attention_factorized_lane_r1_v2_b_word_coordinate(
    const std::size_t outer_panel,
    const std::size_t n16,
    const std::size_t lane,
    const std::size_t word) noexcept {
  if (n16 >= kSm87A4W4AttentionFactorizedLaneR1V2OuterBlock /
                   kSm87A4W4AttentionFactorizedLaneR1V2PairN ||
      lane >= kSm87A4W4AttentionFactorizedLaneR1V2PairLanes ||
      word >= 4U) {
    return {};
  }
  return {outer_panel * kSm87A4W4AttentionFactorizedLaneR1V2OuterBlock +
              n16 * kSm87A4W4AttentionFactorizedLaneR1V2PairN +
              (word / 2U) * 8U + lane / 4U,
          (word % 2U) * 16U + 4U * (lane % 4U), true};
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(
    const std::size_t outer_count) noexcept {
  const std::size_t blocks =
      sm87_a4w4_attention_k256_outer_block_count(outer_count);
  return blocks == 0U ||
                 !sm87_a4w4_attention_k256_product_fits(
                     blocks, kSm87A4W4AttentionK256OuterBlock)
             ? 0U
             : blocks * kSm87A4W4AttentionK256OuterBlock;
}

[[nodiscard]] Q3X_SM87_A4W4_ATTENTION_R1_HOST_DEVICE constexpr std::size_t
sm87_a4w4_attention_factorized_lane_r1_scale_offset(
    const std::size_t outer_coordinate) noexcept {
  return (outer_coordinate / kSm87A4W4AttentionK256OuterBlock) *
             kSm87A4W4AttentionK256OuterBlock +
         outer_coordinate % kSm87A4W4AttentionK256OuterBlock;
}

struct Sm87A4W4AttentionFactorizedLaneR1ProjectionView final {
  const std::uint8_t* packed_b{};
  std::size_t packed_b_capacity_bytes{};
  const std::uint16_t* lane_scales_bf16{};
  std::size_t scale_capacity_elements{};
  std::size_t output_size{};
  std::uint16_t* output_bf16{};
  std::size_t output_row_stride_elements{};
  std::size_t output_capacity_elements{};
};

// Explicit even though the foundation currently admits one mode.  A
// consumer-native QKV/Z/O epilogue is a package-level structural variable;
// it must be added as a named ABI mode in this same translation unit rather
// than selected through an environment variable or a hidden kernel branch.
enum class Sm87A4W4AttentionFactorizedLaneR1EpilogueMode : std::uint8_t {
  kTopologyBf16 = 0U,
};

struct Sm87A4W4AttentionFactorizedLaneR1Resources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t configured_dynamic_shared_limit_bytes{};
  std::size_t device_optin_shared_limit_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
  int multiprocessor_count{};
};

// Queries all three fixed topology specializations plus the isolated
// contiguous correctness specialization.  Success is the compile/resource
// admission gate: <=255 registers/thread, no local frame, exactly 148,224 B
// dynamic shared memory, and one active 256-thread CTA/SM on 16-SM SM87.
[[nodiscard]] int
query_sm87_a4w4_attention_factorized_lane_r1_m128n256_resources_cuda(
    Sm87A4W4AttentionFactorizedLaneR1Resources* resources) noexcept;

// v2 consumes only the equal-byte adjacent-N8 paired B publication.  It is a
// separate symbol and cannot silently reinterpret a v1/v4 payload.
[[nodiscard]] int
query_sm87_a4w4_attention_factorized_lane_r1_v2_m128n256_resources_cuda(
    Sm87A4W4AttentionFactorizedLaneR1Resources* resources) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_attention_factorized_lane_r1_m128n256_bf16_cuda(
    Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    std::size_t token_count,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView* projections,
    std::size_t projection_count,
    Sm87A4W4AttentionFactorizedLaneR1EpilogueMode epilogue_mode =
        Sm87A4W4AttentionFactorizedLaneR1EpilogueMode::kTopologyBf16,
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_attention_factorized_lane_r1_v2_m128n256_bf16_cuda(
    Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    std::size_t token_count,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView* projections,
    std::size_t projection_count,
    Sm87A4W4AttentionFactorizedLaneR1EpilogueMode epilogue_mode =
        Sm87A4W4AttentionFactorizedLaneR1EpilogueMode::kTopologyBf16,
    void* cuda_stream = nullptr) noexcept;

// Correctness-only contiguous topology.  One projection contains exactly
// `macro_cells * 256` outputs.  This keeps small-K bit-exact validation out
// of the fixed Qwen3.6 ABI and never allocates or copies device descriptors.
[[nodiscard]] int
launch_sm87_a4w4_attention_factorized_lane_r1_m128n256_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    std::size_t token_count,
    std::size_t input_size,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView* projection,
    std::size_t macro_cells,
    Sm87A4W4AttentionFactorizedLaneR1EpilogueMode epilogue_mode =
        Sm87A4W4AttentionFactorizedLaneR1EpilogueMode::kTopologyBf16,
    unsigned int maximum_launch_ctas =
        static_cast<unsigned int>(
            kSm87A4W4AttentionFactorizedLaneR1PersistentCtas),
    void* cuda_stream = nullptr) noexcept;

[[nodiscard]] int
launch_sm87_a4w4_attention_factorized_lane_r1_v2_m128n256_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_lane_scales_bf16,
    std::size_t a_scale_capacity_elements,
    std::size_t token_count,
    std::size_t input_size,
    const Sm87A4W4AttentionFactorizedLaneR1ProjectionView* projection,
    std::size_t macro_cells,
    Sm87A4W4AttentionFactorizedLaneR1EpilogueMode epilogue_mode =
        Sm87A4W4AttentionFactorizedLaneR1EpilogueMode::kTopologyBf16,
    unsigned int maximum_launch_ctas =
        static_cast<unsigned int>(
            kSm87A4W4AttentionFactorizedLaneR1PersistentCtas),
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4AttentionFactorizedLaneR1TileM ==
              kSm87A4W4AttentionK256TileM);
static_assert(kSm87A4W4AttentionFactorizedLaneR1TileN ==
              kSm87A4W4AttentionK256TileN);
static_assert(kSm87A4W4AttentionFactorizedLaneR1Threads == 256U);
static_assert(kSm87A4W4AttentionFactorizedLaneR1Warps == 8U);
static_assert(kSm87A4W4AttentionFactorizedLaneR1V2OuterBlock %
                      kSm87A4W4AttentionFactorizedLaneR1V2PairN ==
                  0U &&
              kSm87A4W4AttentionFactorizedLaneR1V2PairLanes *
                      kSm87A4W4AttentionFactorizedLaneR1V2PairBytes ==
                  512U);
static_assert(kSm87A4W4AttentionFactorizedLaneR1Stages *
                      kSm87A4W4AttentionFactorizedLaneR1StageBytes +
                  kSm87A4W4AttentionFactorizedLaneR1ScaleBytes ==
              kSm87A4W4AttentionFactorizedLaneR1SharedBytes);
static_assert(
    sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(1U) ==
    64U);
static_assert(
    sm87_a4w4_attention_factorized_lane_r1_scale_capacity_elements(5'120U) ==
    5'120U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_ATTENTION_R1_HOST_DEVICE
