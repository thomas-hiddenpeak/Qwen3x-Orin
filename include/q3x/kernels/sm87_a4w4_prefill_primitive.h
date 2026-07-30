#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__CUDACC__)
#define Q3X_SM87_A4W4_HOST_DEVICE __host__ __device__
#else
#define Q3X_SM87_A4W4_HOST_DEVICE
#endif

namespace q3x::kernels {

// Test-admission-only SM87 A4W4 primitive contract.  This header describes
// one warp's m16n8k64 S4 x S4 -> S32 operation.  It is deliberately not a
// GEMM API, a production selector, or a performance result.
//
// Canonical packed matrices use signed two's-complement nibbles.  The even
// inner coordinate occupies bits [3:0] and the odd coordinate bits [7:4]:
//
//   A: [M, K/2] bytes, row-major in logical [M, K]
//   B: [N, K/2] bytes, row-major in logical [N, K]
//
// B is therefore the byte-level representation of the column-major [K, N]
// operand consumed by mma.sync(...row.col...).  No runtime transpose or
// nibble permutation is required between this ABI and the warp fragments.
inline constexpr std::size_t kSm87A4W4WarpThreads = 32U;
inline constexpr std::size_t kSm87A4W4MmaM = 16U;
inline constexpr std::size_t kSm87A4W4MmaN = 8U;
inline constexpr std::size_t kSm87A4W4MmaK = 64U;
inline constexpr std::size_t kSm87A4W4PackedABytes =
    kSm87A4W4MmaM * kSm87A4W4MmaK / 2U;
inline constexpr std::size_t kSm87A4W4PackedBBytes =
    kSm87A4W4MmaN * kSm87A4W4MmaK / 2U;
inline constexpr std::size_t kSm87A4W4ARegistersPerThread = 4U;
inline constexpr std::size_t kSm87A4W4BRegistersPerThread = 2U;
inline constexpr std::size_t kSm87A4W4AccumulatorRegistersPerThread = 4U;
inline constexpr std::size_t kSm87A4W4NibblesPerRegister = 8U;
inline constexpr std::size_t kSm87A4W4ConsumerOuterBlock = 64U;
inline constexpr std::size_t kSm87A4W4ConsumerKBlock = 64U;
inline constexpr std::size_t kSm87A4W4ConsumerPackedKBlockBytes = 32U;
inline constexpr std::int32_t kSm87A4W4MaximumK64Partial =
    static_cast<std::int32_t>(kSm87A4W4MmaK * 8U * 8U);
inline constexpr int kSm87A4W4RequiredComputeMajor = 8;
inline constexpr int kSm87A4W4RequiredComputeMinor = 7;

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr bool
sm87_a4w4_is_signed_nibble(
    const int value) noexcept {
  return value >= -8 && value <= 7;
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::uint8_t
sm87_a4w4_encode_signed_nibble(
    const int value) noexcept {
  return static_cast<std::uint8_t>(value) & 0x0fU;
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::int8_t
sm87_a4w4_decode_signed_nibble(
    const std::uint8_t nibble) noexcept {
  const std::uint8_t code = nibble & 0x0fU;
  return static_cast<std::int8_t>(
      code < 8U ? static_cast<int>(code) : static_cast<int>(code) - 16);
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::uint8_t
sm87_a4w4_pack_signed_pair(
    const int even, const int odd) noexcept {
  return static_cast<std::uint8_t>(
      sm87_a4w4_encode_signed_nibble(even) |
      (sm87_a4w4_encode_signed_nibble(odd) << 4U));
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::int8_t
sm87_a4w4_unpack_signed(
    const std::uint8_t packed, const std::size_t inner_coordinate) noexcept {
  return sm87_a4w4_decode_signed_nibble(static_cast<std::uint8_t>(
      packed >> (4U * static_cast<unsigned int>(inner_coordinate & 1U))));
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_packed_row_bytes(
    const std::size_t logical_k) noexcept {
  return logical_k % 2U == 0U ? logical_k / 2U : 0U;
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_k64_group_count(
    const std::size_t logical_k) noexcept {
  return logical_k % kSm87A4W4MmaK == 0U
             ? logical_k / kSm87A4W4MmaK
             : 0U;
}

// Full-model Prefill consumer layout. N/M=64 is only the physical block, not
// a CTA tile restriction: wider CTA tiles concatenate adjacent outer blocks.
// For logical X[outer,K], packed bytes and scales are respectively
//
//   [ceil(outer/64), K/64, 64, 32]
//   [ceil(outer/64), K/64, 64]
//
// Dynamic activations zero-pad the final outer block. All pinned projection N
// dimensions are exact multiples of 64, so offline weights add no padding.
[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_consumer_outer_block_count(
    const std::size_t outer_count) noexcept {
  return outer_count == 0U
             ? 0U
             : 1U + (outer_count - 1U) / kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_consumer_packed_capacity_bytes(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t groups = sm87_a4w4_k64_group_count(logical_k);
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return groups == 0U || blocks == 0U
             ? 0U
             : blocks * groups * kSm87A4W4ConsumerOuterBlock *
                   kSm87A4W4ConsumerPackedKBlockBytes;
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_consumer_scale_capacity_elements(
    const std::size_t outer_count, const std::size_t logical_k) noexcept {
  const std::size_t groups = sm87_a4w4_k64_group_count(logical_k);
  const std::size_t blocks =
      sm87_a4w4_consumer_outer_block_count(outer_count);
  return groups == 0U || blocks == 0U
             ? 0U
             : blocks * groups * kSm87A4W4ConsumerOuterBlock;
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_consumer_packed_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t byte_in_k64, const std::size_t k64_group_count) noexcept {
  return (((outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
               k64_group_count +
           k64_group) *
              kSm87A4W4ConsumerOuterBlock +
          outer_coordinate % kSm87A4W4ConsumerOuterBlock) *
             kSm87A4W4ConsumerPackedKBlockBytes +
         byte_in_k64;
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_consumer_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t k64_group_count) noexcept {
  return ((outer_coordinate / kSm87A4W4ConsumerOuterBlock) *
              k64_group_count +
          k64_group) *
             kSm87A4W4ConsumerOuterBlock +
         outer_coordinate % kSm87A4W4ConsumerOuterBlock;
}

// Shared-memory row swizzle for one packed K64 block. Each logical row has
// two 16-byte halves. Rows 4..7 (and 12..15) exchange those halves, mapping
// every warp LDS.u32 fragment load across all 32 banks without a 2-way alias.
[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_swizzled_k64_byte_offset(
    const std::size_t row, const std::size_t logical_byte) noexcept {
  const std::size_t half = logical_byte / 16U;
  const std::size_t byte_in_half = logical_byte % 16U;
  const std::size_t physical_half = half ^ ((row >> 2U) & 1U);
  return row * kSm87A4W4ConsumerPackedKBlockBytes +
         physical_half * 16U + byte_in_half;
}

// Real-model scale contract: one BF16 scale for every logical K64 group.
// Activation scales are [M, K/64], weight scales are [N, K/64].  An integer
// partial for group g contributes
//
//   S32(m,n,g) * BF16(a_scale[m,g]) * BF16(b_scale[n,g])
//
// to the FP32 output.  S32 partials from groups with different scale products
// must never be combined before dequantization.  The scales are non-negative,
// finite BF16 values. The nearest_even_v1 producer stores BF16 one for an
// all-zero group and emits zero codes. The primitive
// consumes calibrated scales and does not prescribe or silently redo the
// model's offline/dynamic scale search. The nearest_even_v1 producers first
// round threshold/7 to the stored BF16 scale and then quantize clipped values
// against that decoded stored scale; code generation never divides by an
// unpersisted FP32 scale.
[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr std::size_t
sm87_a4w4_k64_scale_offset(
    const std::size_t outer_coordinate, const std::size_t k64_group,
    const std::size_t k64_group_count) noexcept {
  return outer_coordinate * k64_group_count + k64_group;
}

struct Sm87A4W4FragmentCoordinate final {
  std::uint16_t outer{};
  std::uint16_t k{};

  [[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr bool operator==(
      const Sm87A4W4FragmentCoordinate& other) const noexcept {
    return outer == other.outer && k == other.k;
  }
};

struct Sm87A4W4AccumulatorCoordinate final {
  std::uint16_t m{};
  std::uint16_t n{};

  [[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr bool operator==(
      const Sm87A4W4AccumulatorCoordinate& other) const noexcept {
    return m == other.m && n == other.n;
  }
};

// PTX/CUTLASS fragment ownership for m16n8k64.row.col.s32.s4.s4.s32.
// fragment_nibble is the nibble's linear position in the lane's register
// tuple (register = fragment_nibble / 8, bit = 4*(fragment_nibble % 8)).
[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr
Sm87A4W4FragmentCoordinate
sm87_a4w4_a_fragment_coordinate(const std::size_t lane,
                                const std::size_t fragment_nibble) noexcept {
  if (lane >= kSm87A4W4WarpThreads || fragment_nibble >= 32U) {
    return {0xffffU, 0xffffU};
  }
  const std::size_t value_in_register = fragment_nibble % 8U;
  const std::size_t register_index = fragment_nibble / 8U;
  const std::size_t row_half = register_index & 1U;
  const std::size_t k_half = register_index >> 1U;
  return {static_cast<std::uint16_t>(lane / 4U + 8U * row_half),
          static_cast<std::uint16_t>(8U * (lane % 4U) +
                                     value_in_register + 32U * k_half)};
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr
Sm87A4W4FragmentCoordinate
sm87_a4w4_b_fragment_coordinate(const std::size_t lane,
                                const std::size_t fragment_nibble) noexcept {
  if (lane >= kSm87A4W4WarpThreads || fragment_nibble >= 16U) {
    return {0xffffU, 0xffffU};
  }
  const std::size_t value_in_register = fragment_nibble % 8U;
  const std::size_t register_index = fragment_nibble / 8U;
  return {static_cast<std::uint16_t>(lane / 4U),
          static_cast<std::uint16_t>(8U * (lane % 4U) +
                                     value_in_register +
                                     32U * register_index)};
}

[[nodiscard]] Q3X_SM87_A4W4_HOST_DEVICE constexpr
Sm87A4W4AccumulatorCoordinate
sm87_a4w4_accumulator_coordinate(const std::size_t lane,
                                 const std::size_t register_index) noexcept {
  if (lane >= kSm87A4W4WarpThreads || register_index >= 4U) {
    return {0xffffU, 0xffffU};
  }
  return {static_cast<std::uint16_t>(lane / 4U +
                                     8U * (register_index / 2U)),
          static_cast<std::uint16_t>(2U * (lane % 4U) +
                                     register_index % 2U)};
}

struct alignas(16) Sm87A4W4AFragment final {
  std::uint32_t x0{};
  std::uint32_t x1{};
  std::uint32_t x2{};
  std::uint32_t x3{};
};

struct alignas(8) Sm87A4W4BFragment final {
  std::uint32_t x0{};
  std::uint32_t x1{};
};

struct alignas(16) Sm87A4W4Accumulator final {
  std::int32_t x0{};
  std::int32_t x1{};
  std::int32_t x2{};
  std::int32_t x3{};
};

static_assert(sizeof(Sm87A4W4AFragment) == 16U);
static_assert(sizeof(Sm87A4W4BFragment) == 8U);
static_assert(sizeof(Sm87A4W4Accumulator) == 16U);
static_assert(kSm87A4W4PackedABytes == 512U);
static_assert(kSm87A4W4PackedBBytes == 256U);
static_assert(kSm87A4W4MaximumK64Partial == 4'096);
static_assert(sm87_a4w4_k64_group_count(5'120U) == 80U);
static_assert(sm87_a4w4_k64_group_count(6'144U) == 96U);
static_assert(sm87_a4w4_k64_group_count(17'408U) == 272U);
static_assert(sm87_a4w4_a_fragment_coordinate(0U, 0U) ==
              Sm87A4W4FragmentCoordinate{0U, 0U});
static_assert(sm87_a4w4_a_fragment_coordinate(31U, 31U) ==
              Sm87A4W4FragmentCoordinate{15U, 63U});
static_assert(sm87_a4w4_b_fragment_coordinate(31U, 15U) ==
              Sm87A4W4FragmentCoordinate{7U, 63U});
static_assert(sm87_a4w4_accumulator_coordinate(31U, 3U) ==
              Sm87A4W4AccumulatorCoordinate{15U, 7U});

#if defined(__CUDACC__)

[[nodiscard]] __device__ __forceinline__ std::uint32_t
sm87_a4w4_load_packed_u32(const std::uint8_t* const pointer) noexcept {
  return *reinterpret_cast<const std::uint32_t*>(pointer);
}

// Direct canonical loads are intentionally tiny, reusable building blocks.
// packed_a/packed_b may point to global or shared memory, but each row base,
// row stride, and K64 group must preserve 4-byte alignment.  A future
// persistent M64/M128 kernel may stage canonical bytes with cp.async and call
// these same helpers on shared-memory pointers.
[[nodiscard]] __device__ __forceinline__ Sm87A4W4AFragment
sm87_a4w4_load_a_fragment(const std::uint8_t* const packed_a,
                          const std::size_t packed_row_stride_bytes,
                          const std::size_t k64_group,
                          const unsigned int lane) noexcept {
  const std::size_t row0 = lane / 4U;
  const std::size_t row1 = row0 + 8U;
  const std::size_t byte_in_group = 4U * (lane % 4U);
  const std::size_t group_byte = k64_group * (kSm87A4W4MmaK / 2U);
  return {
      sm87_a4w4_load_packed_u32(
          packed_a + row0 * packed_row_stride_bytes + group_byte +
          byte_in_group),
      sm87_a4w4_load_packed_u32(
          packed_a + row1 * packed_row_stride_bytes + group_byte +
          byte_in_group),
      sm87_a4w4_load_packed_u32(
          packed_a + row0 * packed_row_stride_bytes + group_byte + 16U +
          byte_in_group),
      sm87_a4w4_load_packed_u32(
          packed_a + row1 * packed_row_stride_bytes + group_byte + 16U +
          byte_in_group),
  };
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4BFragment
sm87_a4w4_load_b_fragment(const std::uint8_t* const packed_b,
                          const std::size_t packed_row_stride_bytes,
                          const std::size_t k64_group,
                          const unsigned int lane) noexcept {
  const std::size_t n = lane / 4U;
  const std::size_t byte_in_group = 4U * (lane % 4U);
  const std::size_t group_byte = k64_group * (kSm87A4W4MmaK / 2U);
  return {
      sm87_a4w4_load_packed_u32(
          packed_b + n * packed_row_stride_bytes + group_byte +
          byte_in_group),
      sm87_a4w4_load_packed_u32(
          packed_b + n * packed_row_stride_bytes + group_byte + 16U +
          byte_in_group),
  };
}

// Loads from one shared-memory K64 block staged with
// sm87_a4w4_swizzled_k64_byte_offset(). The logical fragment ownership is
// byte-identical to the canonical helpers above; only the physical shared
// address changes to remove the row-stride bank alias.
[[nodiscard]] __device__ __forceinline__ Sm87A4W4AFragment
sm87_a4w4_load_a_fragment_swizzled_shared(
    const std::uint8_t* const packed_a, const unsigned int lane) noexcept {
  const std::size_t row0 = lane / 4U;
  const std::size_t row1 = row0 + 8U;
  const std::size_t byte_in_half = 4U * (lane % 4U);
  return {
      sm87_a4w4_load_packed_u32(
          packed_a + sm87_a4w4_swizzled_k64_byte_offset(
                         row0, byte_in_half)),
      sm87_a4w4_load_packed_u32(
          packed_a + sm87_a4w4_swizzled_k64_byte_offset(
                         row1, byte_in_half)),
      sm87_a4w4_load_packed_u32(
          packed_a + sm87_a4w4_swizzled_k64_byte_offset(
                         row0, 16U + byte_in_half)),
      sm87_a4w4_load_packed_u32(
          packed_a + sm87_a4w4_swizzled_k64_byte_offset(
                         row1, 16U + byte_in_half)),
  };
}

[[nodiscard]] __device__ __forceinline__ Sm87A4W4BFragment
sm87_a4w4_load_b_fragment_swizzled_shared(
    const std::uint8_t* const packed_b, const unsigned int lane) noexcept {
  const std::size_t row = lane / 4U;
  const std::size_t byte_in_half = 4U * (lane % 4U);
  return {
      sm87_a4w4_load_packed_u32(
          packed_b + sm87_a4w4_swizzled_k64_byte_offset(
                         row, byte_in_half)),
      sm87_a4w4_load_packed_u32(
          packed_b + sm87_a4w4_swizzled_k64_byte_offset(
                         row, 16U + byte_in_half)),
  };
}

__device__ __forceinline__ void sm87_a4w4_mma_m16n8k64(
    Sm87A4W4Accumulator& accumulator,
    const Sm87A4W4AFragment& a,
    const Sm87A4W4BFragment& b) noexcept {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == 870
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.s32.s4.s4.s32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+r"(accumulator.x0), "+r"(accumulator.x1),
        "+r"(accumulator.x2), "+r"(accumulator.x3)
      : "r"(a.x0), "r"(a.x1), "r"(a.x2), "r"(a.x3),
        "r"(b.x0), "r"(b.x1));
#else
  // This admission is intentionally fail-closed.  The host launcher rejects
  // every device except SM87; trap as a second line of defense if a future
  // caller bypasses that launcher.
  asm volatile("trap;");
#endif
}

#endif  // defined(__CUDACC__)

struct Sm87A4W4PrimitiveResources final {
  int registers_per_thread{};
  std::size_t static_shared_bytes{};
  std::size_t dynamic_shared_bytes{};
  std::size_t local_bytes{};
  int maximum_threads_per_block{};
  int active_blocks_per_sm{};
  int compute_major{};
  int compute_minor{};
};

// Query and launch functions live only in the gated test-admission library.
// Both require the current CUDA device to be exactly compute capability 8.7.
[[nodiscard]] int query_sm87_a4w4_primitive_resources_cuda(
    Sm87A4W4PrimitiveResources* resources) noexcept;

// One-warp T1 correctness/smoke launcher for one M16xN8xK64 partial.  It
// overwrites output with the dequantized contribution of k64_group; it does
// not iterate K groups, tile M/N, or claim to implement a complete GEMM.
[[nodiscard]] int launch_sm87_a4w4_k64_primitive_smoke_cuda(
    const std::uint8_t* packed_a,
    std::size_t packed_a_row_stride_bytes,
    const std::uint8_t* packed_b,
    std::size_t packed_b_row_stride_bytes,
    const std::uint16_t* a_k64_scales_bf16,
    const std::uint16_t* b_k64_scales_bf16,
    std::size_t k64_group_count,
    std::size_t k64_group,
    float* output,
    std::size_t output_row_stride_elements = kSm87A4W4MmaN,
    void* cuda_stream = nullptr) noexcept;

}  // namespace q3x::kernels

#undef Q3X_SM87_A4W4_HOST_DEVICE
