#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_ATTENTION_K256_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_ATTENTION_K256_HOST_DEVICE
#endif

namespace q3x::kernels {

// Qwen3.6 Attention projection macro-cell under an independent K256 A4 ABI.
// One 512-thread CTA owns M128N256.  Warps are laid out as
//
//   warp = n128_half * 8 + m16
//
// and retain M16N128 (64 FP32 outputs/thread).  Each logical K256 group is
// copied as one combined A-code/B-code/A-scale/B-scale cp.async stage.  The
// three-stage ring is exactly 149,760 bytes; a four-stage variant is outside
// this contract and cannot fit the SM87 opt-in shared-memory limit.
//
// Within a group a warp first loads four ordered K64 A fragments, then walks
// its sixteen N8 fragments.  Only the current four-register S32 partial is
// live.  One FP32 scale product and one FP32 FMA per output are performed only
// after all four ordered K64 MMAs, and BF16 RNE occurs once at final store.
inline constexpr std::size_t kSm87A4W4AttentionK256TileM = 128U;
inline constexpr std::size_t kSm87A4W4AttentionK256TileN = 256U;
inline constexpr std::size_t kSm87A4W4AttentionK256PanelN = 64U;
inline constexpr std::size_t kSm87A4W4AttentionK256PanelsPerCell = 4U;
inline constexpr std::size_t kSm87A4W4AttentionK256ScaleK = 256U;
inline constexpr std::size_t kSm87A4W4AttentionK256PhysicalK64 = 64U;
inline constexpr std::size_t kSm87A4W4AttentionK256K64PerGroup = 4U;
inline constexpr std::size_t kSm87A4W4AttentionK256PackedRowK64Bytes = 32U;
inline constexpr std::size_t kSm87A4W4AttentionK256OuterBlock = 64U;
inline constexpr std::size_t kSm87A4W4AttentionK256Threads = 512U;
inline constexpr std::size_t kSm87A4W4AttentionK256Warps = 16U;
inline constexpr std::size_t kSm87A4W4AttentionK256Stages = 3U;
inline constexpr std::size_t kSm87A4W4AttentionK256PersistentCtas = 16U;
inline constexpr std::size_t kSm87A4W4AttentionK256CtasPerSm = 1U;
inline constexpr std::size_t kSm87A4W4AttentionK256MaximumRegisters = 128U;
inline constexpr std::size_t kSm87A4W4AttentionK256StageACodeBytes =
    kSm87A4W4AttentionK256TileM * kSm87A4W4AttentionK256ScaleK / 2U;
inline constexpr std::size_t kSm87A4W4AttentionK256StageBCodeBytes =
    kSm87A4W4AttentionK256TileN * kSm87A4W4AttentionK256ScaleK / 2U;
inline constexpr std::size_t kSm87A4W4AttentionK256StageAScaleBytes =
    kSm87A4W4AttentionK256TileM * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4AttentionK256StageBScaleBytes =
    kSm87A4W4AttentionK256TileN * sizeof(std::uint16_t);
inline constexpr std::size_t kSm87A4W4AttentionK256StageBytes =
    kSm87A4W4AttentionK256StageACodeBytes +
    kSm87A4W4AttentionK256StageBCodeBytes +
    kSm87A4W4AttentionK256StageAScaleBytes +
    kSm87A4W4AttentionK256StageBScaleBytes;
inline constexpr std::size_t kSm87A4W4AttentionK256SharedBytes =
    kSm87A4W4AttentionK256Stages * kSm87A4W4AttentionK256StageBytes;

inline constexpr std::size_t kQwen36AttentionK256LinearQkvPanels = 160U;
inline constexpr std::size_t kQwen36AttentionK256LinearZPanels = 96U;
inline constexpr std::size_t kQwen36AttentionK256FullQPanels = 192U;
inline constexpr std::size_t kQwen36AttentionK256FullKPanels = 16U;
inline constexpr std::size_t kQwen36AttentionK256FullVPanels = 16U;
inline constexpr std::size_t kQwen36AttentionK256OPanels = 80U;
inline constexpr std::size_t kQwen36AttentionK256LinearCells = 64U;
inline constexpr std::size_t kQwen36AttentionK256FullCells = 56U;
inline constexpr std::size_t kQwen36AttentionK256OCells = 20U;
inline constexpr std::size_t kQwen36AttentionK256HiddenSize = 5'120U;
inline constexpr std::size_t kQwen36AttentionK256OInputSize = 6'144U;

enum class Sm87A4W4AttentionK256Topology : std::uint8_t {
  kLinearQkvZ = 0U,
  kFullQkv = 1U,
  kAttentionO = 2U,
};

struct Sm87A4W4AttentionK256PanelDescriptor final {
  std::uint8_t projection{};
  std::size_t panel{};
};

// Host-side projection view.  All pointers address device storage.  The
// launcher reads this small descriptor synchronously and never transfers it.
struct Sm87A4W4AttentionK256ProjectionView final {
  const std::uint8_t* packed_b{};
  std::size_t packed_b_capacity_bytes{};
  const std::uint16_t* k256_scales_bf16{};
  std::size_t scale_capacity_elements{};
  std::size_t output_size{};
  std::uint16_t* output_bf16{};
  std::size_t output_row_stride_elements{};
  std::size_t output_capacity_elements{};
};

[[nodiscard]] constexpr bool sm87_a4w4_attention_k256_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_outer_block_count(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) /
                        kSm87A4W4AttentionK256OuterBlock;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_k64_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4AttentionK256PhysicalK64 == 0U
             ? logical_k / kSm87A4W4AttentionK256PhysicalK64
             : 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4AttentionK256ScaleK == 0U
             ? logical_k / kSm87A4W4AttentionK256ScaleK
             : 0U;
}

// Canonical physical code layout [outer/64][K/64][64][32].
[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_packed_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_attention_k256_outer_block_count(outer_count);
  const std::size_t groups =
      sm87_a4w4_attention_k256_k64_group_count(logical_k);
  constexpr std::size_t group_bytes =
      kSm87A4W4AttentionK256OuterBlock *
      kSm87A4W4AttentionK256PackedRowK64Bytes;
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_attention_k256_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_attention_k256_product_fits(block_groups, group_bytes)
             ? block_groups * group_bytes
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_ATTENTION_K256_HOST_DEVICE constexpr std::size_t
sm87_a4w4_attention_k256_packed_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t byte_in_k64,
    const std::size_t k64_group_count) noexcept {
  return (((outer_coordinate / kSm87A4W4AttentionK256OuterBlock) *
               k64_group_count +
           k64_group) *
              kSm87A4W4AttentionK256OuterBlock +
          outer_coordinate % kSm87A4W4AttentionK256OuterBlock) *
             kSm87A4W4AttentionK256PackedRowK64Bytes +
         byte_in_k64;
}

// Independent K256 scale layout [outer/64][K/256][64] BF16 elements.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_scale_capacity_elements(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t blocks =
      sm87_a4w4_attention_k256_outer_block_count(outer_count);
  const std::size_t groups =
      sm87_a4w4_attention_k256_group_count(logical_k);
  if (blocks == 0U || groups == 0U ||
      !sm87_a4w4_attention_k256_product_fits(blocks, groups)) {
    return 0U;
  }
  const std::size_t block_groups = blocks * groups;
  return sm87_a4w4_attention_k256_product_fits(
             block_groups, kSm87A4W4AttentionK256OuterBlock)
             ? block_groups * kSm87A4W4AttentionK256OuterBlock
             : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_ATTENTION_K256_HOST_DEVICE constexpr std::size_t
sm87_a4w4_attention_k256_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k256_group,
    const std::size_t k256_group_count) noexcept {
  return ((outer_coordinate / kSm87A4W4AttentionK256OuterBlock) *
              k256_group_count +
          k256_group) *
             kSm87A4W4AttentionK256OuterBlock +
         outer_coordinate % kSm87A4W4AttentionK256OuterBlock;
}

// The projection macro-cell consumes complete M128 tiles.  This helper is
// the sole padding authority for its activation producer.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_launch_token_count(
    const std::size_t logical_token_count) noexcept {
  if (logical_token_count == 0U ||
      logical_token_count >
          std::numeric_limits<std::size_t>::max() -
              (kSm87A4W4AttentionK256TileM - 1U)) {
    return 0U;
  }
  return ((logical_token_count + kSm87A4W4AttentionK256TileM - 1U) /
          kSm87A4W4AttentionK256TileM) *
         kSm87A4W4AttentionK256TileM;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_fixed_cell_count(
    const Sm87A4W4AttentionK256Topology topology) noexcept {
  switch (topology) {
    case Sm87A4W4AttentionK256Topology::kLinearQkvZ:
      return kQwen36AttentionK256LinearCells;
    case Sm87A4W4AttentionK256Topology::kFullQkv:
      return kQwen36AttentionK256FullCells;
    case Sm87A4W4AttentionK256Topology::kAttentionO:
      return kQwen36AttentionK256OCells;
  }
  return 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_fixed_input_size(
    const Sm87A4W4AttentionK256Topology topology) noexcept {
  return topology == Sm87A4W4AttentionK256Topology::kAttentionO
             ? kQwen36AttentionK256OInputSize
             : kQwen36AttentionK256HiddenSize;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_fixed_projection_count(
    const Sm87A4W4AttentionK256Topology topology) noexcept {
  switch (topology) {
    case Sm87A4W4AttentionK256Topology::kLinearQkvZ:
      return 2U;
    case Sm87A4W4AttentionK256Topology::kFullQkv:
      return 3U;
    case Sm87A4W4AttentionK256Topology::kAttentionO:
      return 1U;
  }
  return 0U;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_attention_k256_fixed_projection_panels(
    const Sm87A4W4AttentionK256Topology topology,
    const std::size_t projection) noexcept {
  if (topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    return projection == 0U ? kQwen36AttentionK256LinearQkvPanels
                            : projection == 1U
                                  ? kQwen36AttentionK256LinearZPanels
                                  : 0U;
  }
  if (topology == Sm87A4W4AttentionK256Topology::kFullQkv) {
    return projection == 0U ? kQwen36AttentionK256FullQPanels
                            : projection == 1U
                                  ? kQwen36AttentionK256FullKPanels
                                  : projection == 2U
                                        ? kQwen36AttentionK256FullVPanels
                                        : 0U;
  }
  return topology == Sm87A4W4AttentionK256Topology::kAttentionO &&
                 projection == 0U
             ? kQwen36AttentionK256OPanels
             : 0U;
}

[[nodiscard]] constexpr Sm87A4W4AttentionK256PanelDescriptor
sm87_a4w4_attention_k256_fixed_panel(
    const Sm87A4W4AttentionK256Topology topology,
    const std::size_t cell, const std::size_t slot) noexcept {
  if (slot >= kSm87A4W4AttentionK256PanelsPerCell ||
      cell >= sm87_a4w4_attention_k256_fixed_cell_count(topology)) {
    return {3U, 0U};
  }
  if (topology == Sm87A4W4AttentionK256Topology::kLinearQkvZ) {
    if (cell < 48U) {
      return slot < 2U
                 ? Sm87A4W4AttentionK256PanelDescriptor{
                       0U, 2U * cell + slot}
                 : Sm87A4W4AttentionK256PanelDescriptor{
                       1U, 2U * cell + slot - 2U};
    }
    return {0U, 96U + 4U * (cell - 48U) + slot};
  }
  if (topology == Sm87A4W4AttentionK256Topology::kFullQkv) {
    if (cell < 8U) {
      return slot < 2U
                 ? Sm87A4W4AttentionK256PanelDescriptor{
                       0U, 2U * cell + slot}
                 : Sm87A4W4AttentionK256PanelDescriptor{
                       1U, 2U * cell + slot - 2U};
    }
    if (cell < 16U) {
      const std::size_t mixed = cell - 8U;
      return slot < 2U
                 ? Sm87A4W4AttentionK256PanelDescriptor{
                       0U, 16U + 2U * mixed + slot}
                 : Sm87A4W4AttentionK256PanelDescriptor{
                       2U, 2U * mixed + slot - 2U};
    }
    return {0U, 32U + 4U * (cell - 16U) + slot};
  }
  return {0U, 4U * cell + slot};
}

struct Sm87A4W4AttentionK256Plan final {
  std::size_t token_count{};
  std::size_t input_size{};
  std::size_t m_tiles{};
  std::size_t macro_cells{};
  std::size_t k256_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4AttentionK256Plan
sm87_a4w4_attention_k256_test_plan(
    const std::size_t token_count, const std::size_t input_size,
    const std::size_t macro_cells) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4AttentionK256TileM != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4AttentionK256ScaleK != 0U ||
      macro_cells == 0U) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4AttentionK256TileM;
  if (!sm87_a4w4_attention_k256_product_fits(m_tiles, macro_cells)) {
    return {};
  }
  const std::size_t work_cells = m_tiles * macro_cells;
  return {token_count,
          input_size,
          m_tiles,
          macro_cells,
          input_size / kSm87A4W4AttentionK256ScaleK,
          input_size / kSm87A4W4AttentionK256PhysicalK64,
          work_cells,
          work_cells < kSm87A4W4AttentionK256PersistentCtas
              ? work_cells
              : kSm87A4W4AttentionK256PersistentCtas};
}

[[nodiscard]] constexpr Sm87A4W4AttentionK256Plan
sm87_a4w4_attention_k256_fixed_plan(
    const Sm87A4W4AttentionK256Topology topology,
    const std::size_t token_count) noexcept {
  return sm87_a4w4_attention_k256_test_plan(
      token_count, sm87_a4w4_attention_k256_fixed_input_size(topology),
      sm87_a4w4_attention_k256_fixed_cell_count(topology));
}

struct Sm87A4W4AttentionK256Resources final {
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
};

// Queries every fixed topology plus the arbitrary-descriptor correctness
// specialization and reports their worst resource values.  Success is the
// hard gate: <=128 registers/thread, zero local frame/spill, exactly 149,760
// bytes of dynamic shared memory, and one active CTA/SM on the 16-SM SM87.
[[nodiscard]] int
query_sm87_a4w4_attention_k256_m128n256_resources_cuda(
    Sm87A4W4AttentionK256Resources* resources) noexcept;

// K256 activation producer for the macro-cell ABI.  `launch_token_count`
// must equal ceil128(logical_token_count).  Padded rows are published as all
// zero signed-nibble codes with BF16 scale one and never read from input.
[[nodiscard]] int launch_sm87_a4_quantize_bf16_k256_cuda(
    const std::uint16_t* input_bf16,
    std::size_t input_row_stride_elements,
    std::size_t logical_token_count,
    std::size_t launch_token_count,
    std::size_t input_size,
    float clip_ratio,
    std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    std::uint16_t* a_k256_scales_bf16,
    std::size_t a_scale_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Fixed Qwen3.6 topology launcher.  The projection array contains exactly
// 2 views for Linear QKV/Z, 3 for Full Q/K/V, or 1 for Attention O, in that
// order.  All weights and activations are independently quantized under the
// K256 ABI described above.
[[nodiscard]] int
launch_sm87_a4w4_attention_k256_m128n256_bf16_cuda(
    Sm87A4W4AttentionK256Topology topology,
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k256_scales_bf16,
    std::size_t a_scale_capacity_elements,
    std::size_t token_count,
    const Sm87A4W4AttentionK256ProjectionView* projections,
    std::size_t projection_count,
    void* cuda_stream = nullptr) noexcept;

// Correctness-only arbitrary panel mapping.  `panels` is a host array of
// `macro_cells * 4` descriptors in cell-major order.  To keep that interface
// isolated from production, this launcher synchronizes its stream before
// returning and frees its temporary device descriptor storage.
[[nodiscard]] int
launch_sm87_a4w4_attention_k256_m128n256_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k256_scales_bf16,
    std::size_t a_scale_capacity_elements,
    std::size_t token_count,
    std::size_t input_size,
    const Sm87A4W4AttentionK256ProjectionView* projections,
    std::size_t projection_count,
    const Sm87A4W4AttentionK256PanelDescriptor* panels,
    std::size_t macro_cells,
    unsigned int maximum_launch_ctas =
        static_cast<unsigned int>(
            kSm87A4W4AttentionK256PersistentCtas),
    void* cuda_stream = nullptr) noexcept;

static_assert(kSm87A4W4AttentionK256K64PerGroup *
                      kSm87A4W4AttentionK256PhysicalK64 ==
                  kSm87A4W4AttentionK256ScaleK);
static_assert(kSm87A4W4AttentionK256StageACodeBytes == 16'384U);
static_assert(kSm87A4W4AttentionK256StageBCodeBytes == 32'768U);
static_assert(kSm87A4W4AttentionK256StageAScaleBytes == 256U);
static_assert(kSm87A4W4AttentionK256StageBScaleBytes == 512U);
static_assert(kSm87A4W4AttentionK256StageBytes == 49'920U);
static_assert(kSm87A4W4AttentionK256SharedBytes == 149'760U);
static_assert(sm87_a4w4_attention_k256_fixed_plan(
                  Sm87A4W4AttentionK256Topology::kLinearQkvZ, 2'048U)
                      .work_cells == 1'024U);
static_assert(sm87_a4w4_attention_k256_fixed_plan(
                  Sm87A4W4AttentionK256Topology::kFullQkv, 2'048U)
                      .work_cells == 896U);
static_assert(sm87_a4w4_attention_k256_fixed_plan(
                  Sm87A4W4AttentionK256Topology::kAttentionO, 2'048U)
                      .work_cells == 320U);

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_ATTENTION_K256_HOST_DEVICE
