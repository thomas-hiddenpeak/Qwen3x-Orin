#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_HOST_DEVICE \
  __host__ __device__
#else
#define Q3X_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_HOST_DEVICE
#endif

namespace q3x::kernels {

// Test-only SM87 Gate+Up cell with a fragment-native, equal-byte B payload.
// It is deliberately disconnected from CMake and production dispatch.
//
// A CTA owns M64 x N64 and contains eight warps.  Warp w owns the complete
// M64 x N8 strip at n = 8*w for both Gate and Up.  The same warp therefore
// consumes one A fragment, one Gate B fragment, and one Up B fragment before
// applying SwiGLU directly from registers; no B code enters shared memory and
// no Gate/Up exchange buffer exists.
//
// The paired code ABI is an equal-byte permutation of the two canonical
// signed-S4 operands.  For each (N64 block, K512 group, physical K64 phase,
// N8 fragment, lane), one aligned 16-byte slot contains:
//
//   bytes  0.. 3: Gate B register x0 (K  0..31 ownership)
//   bytes  4.. 7: Gate B register x1 (K 32..63 ownership)
//   bytes  8..11: Up   B register x0 (K  0..31 ownership)
//   bytes 12..15: Up   B register x1 (K 32..63 ownership)
//
// Within each u32, nibble ownership is exactly
// sm87_a4w4_b_fragment_coordinate(lane, register*8 + nibble).  The ABI has
// no padding: N*K/2 Gate bytes + N*K/2 Up bytes == N*K paired bytes.
//
// Scales retain K512 semantics and are likewise an equal-byte pairing:
// [N64 block, K512 group, row-in-N64, {Gate,Up}] BF16 elements.
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeTileM = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeTileN = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeThreads = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeWarps = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeWarpN = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativePhysicalK = 64U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeScaleK = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeK64PerScale = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeK128PerScale = 4U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeAStages = 3U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeAStageK = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeAStageBytes =
        kSm87A4W4GateUpK512FragmentNativeTileM *
        kSm87A4W4GateUpK512FragmentNativeAStageK / 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeSharedBytes =
        kSm87A4W4GateUpK512FragmentNativeAStages *
        kSm87A4W4GateUpK512FragmentNativeAStageBytes;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativePairSlotBytes = 16U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeLanes = 32U;
// Projection-major v3 is a second, equal-byte code ABI.  Unlike the v2
// AoS16 slot above, one complete (N8,K64) record stores every Gate lane
// before every Up lane:
//
//   [Gate lane 0..31, 8 bytes each][Up lane 0..31, 8 bytes each]
//
// Record order is N64 block / N8 fragment, K512 group, then K64 phase.
// These constants and helpers are intentionally independent from the v2
// helpers so the deployed AoS16 interpretation cannot change by accident.
inline constexpr std::size_t
    kSm87A4W4GateUpK512ProjectionMajorLaneBytes = 8U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512ProjectionMajorProjectionBytes = 256U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512ProjectionMajorRecordBytes = 512U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512ProjectionMajorGate = 0U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512ProjectionMajorUp = 1U;
static_assert(kSm87A4W4GateUpK512FragmentNativeLanes *
                      kSm87A4W4GateUpK512ProjectionMajorLaneBytes ==
                  kSm87A4W4GateUpK512ProjectionMajorProjectionBytes &&
              2U * kSm87A4W4GateUpK512ProjectionMajorProjectionBytes ==
                  kSm87A4W4GateUpK512ProjectionMajorRecordBytes);
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeMaximumRegisters = 128U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeCtasPerSm = 2U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativePersistentCtas = 32U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeModelIntermediate = 17'408U;
inline constexpr std::size_t
    kSm87A4W4GateUpK512FragmentNativeModelInput = 5'120U;

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_fragment_native_product_fits(
    const std::size_t first, const std::size_t second) noexcept {
  return first == 0U ||
         second <= std::numeric_limits<std::size_t>::max() / first;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_fragment_native_outer_blocks(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) /
                        kSm87A4W4GateUpK512FragmentNativeTileN;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_fragment_native_scale_groups(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4GateUpK512FragmentNativeScaleK == 0U
             ? logical_k /
                   kSm87A4W4GateUpK512FragmentNativeScaleK
             : 0U;
}

// The code capacity is bytes.  It equals two canonical N x K/2 payloads.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_fragment_native_code_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  if (outer_count == 0U ||
      outer_count % kSm87A4W4GateUpK512FragmentNativeTileN != 0U ||
      sm87_a4w4_gateup_k512_fragment_native_scale_groups(logical_k) == 0U ||
      !sm87_a4w4_gateup_k512_fragment_native_product_fits(
          outer_count, logical_k)) {
    return 0U;
  }
  return outer_count * logical_k;
}

// Returns the first byte of a lane's aligned Gate+Up fragment pair.
[[nodiscard]] Q3X_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_HOST_DEVICE
constexpr std::size_t
sm87_a4w4_gateup_k512_fragment_native_code_slot_offset(
    const std::size_t outer_coordinate,
    const std::size_t k512_group,
    const std::size_t k64_phase,
    const std::size_t lane,
    const std::size_t k512_group_count) noexcept {
  const std::size_t outer_block =
      outer_coordinate /
      kSm87A4W4GateUpK512FragmentNativeTileN;
  const std::size_t n8_fragment =
      (outer_coordinate %
       kSm87A4W4GateUpK512FragmentNativeTileN) /
      kSm87A4W4GateUpK512FragmentNativeWarpN;
  return (((((outer_block * k512_group_count + k512_group) *
                  kSm87A4W4GateUpK512FragmentNativeK64PerScale +
              k64_phase) *
                 kSm87A4W4GateUpK512FragmentNativeWarps +
             n8_fragment) *
                kSm87A4W4GateUpK512FragmentNativeLanes +
            lane) *
           kSm87A4W4GateUpK512FragmentNativePairSlotBytes);
}

// Projection-major v3 remains an equal-byte permutation of the two
// canonical N x K/2 payloads.  It deliberately has its own capacity helper
// even though the answer equals v2.
[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_projection_major_code_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  if (outer_count == 0U ||
      outer_count % kSm87A4W4GateUpK512FragmentNativeTileN != 0U ||
      sm87_a4w4_gateup_k512_fragment_native_scale_groups(logical_k) == 0U ||
      !sm87_a4w4_gateup_k512_fragment_native_product_fits(
          outer_count, logical_k)) {
    return 0U;
  }
  return outer_count * logical_k;
}

// Returns the first byte of a projection-major (N8,K64) record.
[[nodiscard]] Q3X_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_HOST_DEVICE
constexpr std::size_t
sm87_a4w4_gateup_k512_projection_major_code_record_offset(
    const std::size_t outer_coordinate,
    const std::size_t k512_group,
    const std::size_t k64_phase,
    const std::size_t k512_group_count) noexcept {
  const std::size_t outer_block =
      outer_coordinate /
      kSm87A4W4GateUpK512FragmentNativeTileN;
  const std::size_t n8_fragment =
      (outer_coordinate %
       kSm87A4W4GateUpK512FragmentNativeTileN) /
      kSm87A4W4GateUpK512FragmentNativeWarpN;
  return ((((outer_block * kSm87A4W4GateUpK512FragmentNativeWarps +
             n8_fragment) *
                k512_group_count +
            k512_group) *
               kSm87A4W4GateUpK512FragmentNativeK64PerScale +
           k64_phase) *
          kSm87A4W4GateUpK512ProjectionMajorRecordBytes);
}

// Returns the first byte of one lane's two-u32 fragment.  projection is
// kSm87A4W4GateUpK512ProjectionMajorGate or ...Up.
[[nodiscard]] Q3X_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_HOST_DEVICE
constexpr std::size_t
sm87_a4w4_gateup_k512_projection_major_code_lane_offset(
    const std::size_t outer_coordinate,
    const std::size_t k512_group,
    const std::size_t k64_phase,
    const std::size_t projection,
    const std::size_t lane,
    const std::size_t k512_group_count) noexcept {
  return sm87_a4w4_gateup_k512_projection_major_code_record_offset(
             outer_coordinate, k512_group, k64_phase,
             k512_group_count) +
         projection *
             kSm87A4W4GateUpK512ProjectionMajorProjectionBytes +
         lane * kSm87A4W4GateUpK512ProjectionMajorLaneBytes;
}

[[nodiscard]] constexpr std::size_t
sm87_a4w4_gateup_k512_fragment_native_scale_capacity_elements(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t groups =
      sm87_a4w4_gateup_k512_fragment_native_scale_groups(logical_k);
  if (outer_count == 0U ||
      outer_count % kSm87A4W4GateUpK512FragmentNativeTileN != 0U ||
      groups == 0U ||
      !sm87_a4w4_gateup_k512_fragment_native_product_fits(
          outer_count, groups) ||
      !sm87_a4w4_gateup_k512_fragment_native_product_fits(
          outer_count * groups, 2U)) {
    return 0U;
  }
  return outer_count * groups * 2U;
}

// Returns the Gate element; Up is the immediately following BF16 element.
[[nodiscard]] Q3X_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_HOST_DEVICE
constexpr std::size_t
sm87_a4w4_gateup_k512_fragment_native_scale_pair_offset(
    const std::size_t outer_coordinate,
    const std::size_t k512_group,
    const std::size_t k512_group_count) noexcept {
  const std::size_t outer_block =
      outer_coordinate /
      kSm87A4W4GateUpK512FragmentNativeTileN;
  const std::size_t row =
      outer_coordinate %
      kSm87A4W4GateUpK512FragmentNativeTileN;
  return (((outer_block * k512_group_count + k512_group) *
               kSm87A4W4GateUpK512FragmentNativeTileN +
           row) *
          2U);
}

struct Sm87A4W4GateUpK512FragmentNativePlan final {
  std::size_t token_count{};
  std::size_t intermediate_size{};
  std::size_t input_size{};
  std::size_t n_start{};
  std::size_t n_count{};
  std::size_t m_tiles{};
  std::size_t n_tiles{};
  std::size_t k512_groups{};
  std::size_t physical_k64_groups{};
  std::size_t work_cells{};
  std::size_t launch_ctas{};
};

[[nodiscard]] constexpr Sm87A4W4GateUpK512FragmentNativePlan
sm87_a4w4_gateup_k512_fragment_native_plan(
    const std::size_t token_count,
    const std::size_t intermediate_size,
    const std::size_t input_size,
    const std::size_t n_start,
    const std::size_t n_count) noexcept {
  if (token_count == 0U ||
      token_count % kSm87A4W4GateUpK512FragmentNativeTileM != 0U ||
      intermediate_size == 0U ||
      intermediate_size % kSm87A4W4GateUpK512FragmentNativeTileN != 0U ||
      input_size == 0U ||
      input_size % kSm87A4W4GateUpK512FragmentNativeScaleK != 0U ||
      n_start % kSm87A4W4GateUpK512FragmentNativeTileN != 0U ||
      n_count == 0U ||
      n_count % kSm87A4W4GateUpK512FragmentNativeTileN != 0U ||
      n_start > intermediate_size ||
      n_count > intermediate_size - n_start) {
    return {};
  }
  const std::size_t m_tiles =
      token_count / kSm87A4W4GateUpK512FragmentNativeTileM;
  const std::size_t n_tiles =
      n_count / kSm87A4W4GateUpK512FragmentNativeTileN;
  if (!sm87_a4w4_gateup_k512_fragment_native_product_fits(
          m_tiles, n_tiles)) {
    return {};
  }
  return {token_count,
          intermediate_size,
          input_size,
          n_start,
          n_count,
          m_tiles,
          n_tiles,
          input_size /
              kSm87A4W4GateUpK512FragmentNativeScaleK,
          input_size /
              kSm87A4W4GateUpK512FragmentNativePhysicalK,
          m_tiles * n_tiles,
          m_tiles < kSm87A4W4GateUpK512FragmentNativePersistentCtas
              ? m_tiles
              : kSm87A4W4GateUpK512FragmentNativePersistentCtas};
}

[[nodiscard]] constexpr bool
sm87_a4w4_gateup_k512_fragment_native_is_model_plan(
    const Sm87A4W4GateUpK512FragmentNativePlan& plan) noexcept {
  return plan.launch_ctas != 0U &&
         plan.intermediate_size ==
             kSm87A4W4GateUpK512FragmentNativeModelIntermediate &&
         plan.input_size ==
             kSm87A4W4GateUpK512FragmentNativeModelInput;
}

struct Sm87A4W4GateUpK512FragmentNativeResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

[[nodiscard]] int
query_sm87_a4w4_gateup_k512_fragment_native_resources_cuda(
    Sm87A4W4GateUpK512FragmentNativeResources* resources) noexcept;

// Model-only launcher.  paired_b_codes and paired_b_scales_bf16 are offline
// products in the ABI above; this kernel never constructs the permutation.
[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* paired_b_codes,
    std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* paired_b_scales_bf16,
    std::size_t paired_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    void* cuda_stream = nullptr) noexcept;

// Shape-flexible correctness launcher.  It preserves the same physical ABI
// and numerical boundary, but does not assert the exact model N/K.
[[nodiscard]] int
launch_sm87_a4w4_gateup_k512_fragment_native_test_bf16_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_capacity_bytes,
    const std::uint16_t* a_k512_scales_bf16,
    std::size_t a_scale_capacity_elements,
    const std::uint8_t* paired_b_codes,
    std::size_t paired_b_code_capacity_bytes,
    const std::uint16_t* paired_b_scales_bf16,
    std::size_t paired_b_scale_capacity_elements,
    std::size_t token_count,
    std::size_t intermediate_size,
    std::size_t input_size,
    std::size_t n_start,
    std::size_t n_count,
    std::uint16_t* output_bf16,
    std::size_t output_row_stride_elements,
    std::size_t output_capacity_elements,
    unsigned int maximum_launch_ctas,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_GATEUP_K512_FRAGMENT_NATIVE_HOST_DEVICE
