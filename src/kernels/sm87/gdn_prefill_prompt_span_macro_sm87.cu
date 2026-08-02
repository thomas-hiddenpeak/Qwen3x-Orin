#include "gdn_prefill_prompt_span_macro_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace q3x::runtime::gdn_prefill_prompt_span_macro_detail {
namespace {

constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarps = kThreads / 32U;
constexpr unsigned int kChunk = 64U;
constexpr unsigned int kDimension = 128U;
constexpr unsigned int kQkHeads = 16U;
constexpr unsigned int kValueHeads = 48U;
constexpr unsigned int kHeadGroup = 3U;
constexpr unsigned int kTile = 16U;
constexpr unsigned int kBlockRows = kChunk / kTile;
constexpr unsigned int kPackedBlocks =
    kBlockRows * (kBlockRows + 1U) / 2U;
constexpr unsigned int kPackedElements =
    kPackedBlocks * kTile * kTile;
constexpr std::size_t kSlotBytes = 16U * 1024U;
constexpr unsigned int kSlotCount = 5U;
constexpr std::size_t kMainSharedBytes = kSlotCount * kSlotBytes;
constexpr std::size_t kScalarBytes = 2U * kChunk * sizeof(float);
constexpr std::size_t kParkedStateBytes =
    8U * 4U * kThreads * sizeof(std::uint32_t);
constexpr std::size_t kSharedBytes =
    kMainSharedBytes + kScalarBytes + kParkedStateBytes;
constexpr std::size_t kPanelElements = kChunk * kChunk;
constexpr std::size_t kPanelBytes =
    kPanelElements * sizeof(std::uint16_t);
constexpr std::size_t kVectorElements = kChunk * kDimension;
constexpr std::size_t kGramElements = kChunk * kChunk;
constexpr std::size_t kPackedFp32Bytes =
    kPackedElements * sizeof(float);
constexpr std::size_t kPackedBf16Bytes =
    kPackedElements * sizeof(std::uint16_t);
constexpr std::size_t kVOffset = kGdnQElements + kGdnKElements;

static_assert(kWarps == 8U);
static_assert(kQkHeads * kHeadGroup == kValueHeads);
static_assert(kDimension == 128U);
static_assert(kPanelBytes == 8U * 1024U);
static_assert(kParkedStateBytes == 32U * 1024U);
static_assert(kSharedBytes == 115'200U);
static_assert(kPackedFp32Bytes + 2U * kPackedBf16Bytes ==
              20U * 1024U);

namespace wmma = nvcuda::wmma;
using Bf16 = __nv_bfloat16;
using Accumulator =
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>;

__device__ __forceinline__ float decode_bf16(
    const std::uint16_t value) {
  return __uint_as_float(static_cast<unsigned int>(value) << 16U);
}

__device__ __forceinline__ std::uint16_t encode_bf16(
    const float value) {
  const unsigned int bits = __float_as_uint(value);
  if ((bits & 0x7fff'ffffU) > 0x7f80'0000U) {
    return static_cast<std::uint16_t>((bits >> 16U) | 0x0040U);
  }
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

[[nodiscard]] __device__ __forceinline__ std::uint32_t pack_bf16_pair(
    const float low, const float high) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  std::uint32_t packed = 0U;
  asm("cvt.rn.bf16x2.f32 %0, %2, %1;"
      : "=r"(packed)
      : "f"(low), "f"(high));
  return packed;
#else
  return static_cast<std::uint32_t>(encode_bf16(low)) |
         (static_cast<std::uint32_t>(encode_bf16(high)) << 16U);
#endif
}

__device__ __forceinline__ void copy_16_async(
    void* const shared_destination, const void* const global_source) {
  const unsigned int shared_address = static_cast<unsigned int>(
      __cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(shared_address), "l"(global_source));
}

__device__ __forceinline__ void commit_async() {
  asm volatile("cp.async.commit_group;\n" ::);
}

__device__ __forceinline__ void wait_all_async() {
  asm volatile("cp.async.wait_group 0;\n" ::);
}

[[nodiscard]] __host__ __device__ constexpr unsigned int
packed_block_index(const unsigned int row_block,
                   const unsigned int column_block) noexcept {
  return row_block * (row_block + 1U) / 2U + column_block;
}

__device__ __forceinline__ void unpack_packed_block_index(
    const unsigned int packed_block, unsigned int& row_block,
    unsigned int& column_block) {
  row_block = packed_block >= 6U
                  ? 3U
                  : (packed_block >= 3U ? 2U
                                        : (packed_block >= 1U ? 1U : 0U));
  column_block =
      packed_block - row_block * (row_block + 1U) / 2U;
}

__device__ __forceinline__ void accumulate_row_major_product(
    Accumulator& accumulator, const Bf16* const a,
    const Bf16* const b) {
  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16,
                 wmma::row_major>
      a_fragment;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16,
                 wmma::row_major>
      b_fragment;
  wmma::load_matrix_sync(a_fragment, a, static_cast<int>(kTile));
  wmma::load_matrix_sync(b_fragment, b, static_cast<int>(kTile));
  wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
}

template <unsigned int Row, unsigned int Inner>
__device__ __forceinline__ void accumulate_diagonal_inverse_column(
    const float* const diagonal_l, const unsigned int lane,
    const float (&inverse_column)[kTile], float& value) {
  if constexpr (Inner < Row) {
    if (lane <= Inner) {
      value -= diagonal_l[Row * kTile + Inner] * inverse_column[Inner];
    }
    accumulate_diagonal_inverse_column<Row, Inner + 1U>(
        diagonal_l, lane, inverse_column, value);
  }
}

template <unsigned int Row>
__device__ __forceinline__ void solve_diagonal_inverse_column(
    const float* const diagonal_l, const unsigned int lane,
    float (&inverse_column)[kTile]) {
  float value = lane == Row ? 1.0F : 0.0F;
  accumulate_diagonal_inverse_column<Row, 0U>(
      diagonal_l, lane, inverse_column, value);
  inverse_column[Row] = lane <= Row ? value : 0.0F;
  if constexpr (Row + 1U < kTile) {
    solve_diagonal_inverse_column<Row + 1U>(
        diagonal_l, lane, inverse_column);
  }
}

__device__ __forceinline__ void form_transform_block(
    const Bf16* const diagonal_inverse, const Bf16* const left0,
    const Bf16* const right0, const Bf16* const left1,
    const Bf16* const right1, const Bf16* const left2,
    const Bf16* const right2, Bf16* const inverse,
    float* const scratch, const float* const beta,
    Bf16* const transform, const unsigned int row_base,
    const unsigned int column_base) {
  const unsigned int lane = threadIdx.x & 31U;
  Accumulator inner;
  wmma::fill_fragment(inner, 0.0F);
  accumulate_row_major_product(inner, left0, right0);
  if (left1 != nullptr) {
    accumulate_row_major_product(inner, left1, right1);
  }
  if (left2 != nullptr) {
    accumulate_row_major_product(inner, left2, right2);
  }
  wmma::store_matrix_sync(scratch, inner, static_cast<int>(kTile),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane; index < kTile * kTile; index += 32U) {
    inverse[index] = __float2bfloat16_rn(scratch[index]);
  }
  __syncwarp();

  Accumulator outer;
  wmma::fill_fragment(outer, 0.0F);
  accumulate_row_major_product(outer, diagonal_inverse, inverse);
  wmma::store_matrix_sync(scratch, outer, static_cast<int>(kTile),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane; index < kTile * kTile; index += 32U) {
    const unsigned int row = index / kTile;
    const unsigned int column = index % kTile;
    const float value = -scratch[index];
    inverse[index] = __float2bfloat16_rn(value);
    transform[row_base + row + (column_base + column) * kChunk] =
        __float2bfloat16_rn(value * beta[column_base + column]);
  }
  __syncwarp();
}

// Recompute fragment mapping is intentionally identical to the admitted
// value-head C64 producer.  Only its publication target changes from global
// canonical W/U to the swizzled shared panels consumed by the recurrence.
struct M16K16 final {
  std::uint32_t x0, x1, x2, x3;
};
struct K16N8 final {
  std::uint32_t x0, x1;
};
struct M16N8Accumulator final {
  float x0, x1, x2, x3;
};

__device__ __forceinline__ void zero(M16N8Accumulator& accumulator) {
  accumulator = {0.0F, 0.0F, 0.0F, 0.0F};
}

__device__ __forceinline__ void mma(
    M16N8Accumulator& accumulator, const M16K16& a,
    const K16N8& b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};"
      : "+f"(accumulator.x0), "+f"(accumulator.x1),
        "+f"(accumulator.x2), "+f"(accumulator.x3)
      : "r"(a.x0), "r"(a.x1), "r"(a.x2), "r"(a.x3),
        "r"(b.x0), "r"(b.x1));
#else
  (void)accumulator;
  (void)a;
  (void)b;
#endif
}

[[nodiscard]] __device__ __forceinline__ unsigned int swizzled_chunk(
    const unsigned int row, const unsigned int logical_chunk) {
  return logical_chunk ^ (row & 7U);
}

__device__ __forceinline__ void load_a_col_major(
    M16K16& packed, const Bf16* const transform,
    const unsigned int m_panel, const unsigned int k16) {
  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16,
                 wmma::col_major>
      fragment;
  wmma::load_matrix_sync(
      fragment,
      transform + m_panel * kTile + k16 * kTile * kChunk,
      static_cast<int>(kChunk));
  const auto* const words =
      reinterpret_cast<const std::uint32_t*>(fragment.x);
  packed = {words[0], words[1], words[2], words[3]};
}

__device__ __forceinline__ void load_b_swizzled(
    K16N8& fragment, const Bf16* const operand,
    const unsigned int n_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int row =
      k16 * kTile + (lane & 7U) + ((lane >> 3U) & 1U) * 8U;
  const unsigned int physical_chunk = swizzled_chunk(row, n_panel);
  const auto* const source =
      operand + row * kChunk + physical_chunk * 8U;
  const unsigned int address = static_cast<unsigned int>(
      __cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(address)
      : "memory");
#else
  (void)fragment;
  (void)operand;
  (void)n_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void store_product_swizzled(
    const M16N8Accumulator (&accumulators)[kBlockRows][kBlockRows],
    std::uint16_t* const output_panel, const unsigned int warp,
    const unsigned int lane) {
  const unsigned int row_in_half = lane >> 2U;
  const unsigned int column_pair = lane & 3U;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kBlockRows; ++m_panel) {
    const unsigned int row0 = m_panel * kTile + row_in_half;
    const unsigned int row1 = row0 + 8U;
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kBlockRows; ++n_panel) {
      const unsigned int logical_chunk = warp * kBlockRows + n_panel;
      const auto& accumulator = accumulators[m_panel][n_panel];
      const unsigned int physical0 = swizzled_chunk(row0, logical_chunk);
      const unsigned int physical1 = swizzled_chunk(row1, logical_chunk);
      *reinterpret_cast<std::uint32_t*>(
          output_panel + row0 * kChunk + physical0 * 8U +
          column_pair * 2U) =
          pack_bf16_pair(accumulator.x0, accumulator.x1);
      *reinterpret_cast<std::uint32_t*>(
          output_panel + row1 * kChunk + physical1 * 8U +
          column_pair * 2U) =
          pack_bf16_pair(accumulator.x2, accumulator.x3);
    }
  }
}

__device__ __forceinline__ void recompute_product(
    const Bf16* const transform, const Bf16* const operand,
    std::uint16_t* const output_panel, const unsigned int warp,
    const unsigned int lane) {
  M16N8Accumulator accumulators[kBlockRows][kBlockRows];
#pragma unroll
  for (unsigned int m = 0U; m < kBlockRows; ++m) {
#pragma unroll
    for (unsigned int n = 0U; n < kBlockRows; ++n) {
      zero(accumulators[m][n]);
    }
  }
#pragma unroll
  for (unsigned int k16 = 0U; k16 < kBlockRows; ++k16) {
    M16K16 a[kBlockRows];
    K16N8 b[kBlockRows];
#pragma unroll
    for (unsigned int m = 0U; m < kBlockRows; ++m) {
      load_a_col_major(a[m], transform, m, k16);
    }
#pragma unroll
    for (unsigned int n = 0U; n < kBlockRows; ++n) {
      load_b_swizzled(b[n], operand, warp * kBlockRows + n, k16,
                       lane);
    }
#pragma unroll
    for (unsigned int m = 0U; m < kBlockRows; ++m) {
#pragma unroll
      for (unsigned int n = 0U; n < kBlockRows; ++n) {
        mma(accumulators[m][n], a[m], b[n]);
      }
    }
  }
  store_product_swizzled(accumulators, output_panel, warp, lane);
}

__device__ __forceinline__ void load_a_swizzled(
    M16K16& fragment, const std::uint16_t* const matrix,
    const unsigned int m_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int quadrant = lane / 8U;
  const unsigned int row =
      m_panel * 16U + lane % 8U + (quadrant & 1U) * 8U;
  const unsigned int logical_chunk = k16 * 2U + (quadrant >> 1U);
  const unsigned int physical_chunk = swizzled_chunk(row, logical_chunk);
  const auto* const source =
      matrix + row * kChunk + physical_chunk * 8U;
  const unsigned int address = static_cast<unsigned int>(
      __cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(fragment.x0), "=r"(fragment.x1),
        "=r"(fragment.x2), "=r"(fragment.x3)
      : "r"(address)
      : "memory");
#else
  (void)fragment;
  (void)matrix;
  (void)m_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void load_b_pair_swizzled(
    K16N8& first, K16N8& second,
    const std::uint16_t* const matrix, const unsigned int n_panel,
    const unsigned int k32, const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int row = n_panel * 8U + (lane & 7U);
  const unsigned int logical_chunk = k32 * 4U + (lane >> 3U);
  const unsigned int physical_chunk = swizzled_chunk(row, logical_chunk);
  const auto* const source =
      matrix + row * kChunk + physical_chunk * 8U;
  const unsigned int address = static_cast<unsigned int>(
      __cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];"
      : "=r"(first.x0), "=r"(first.x1),
        "=r"(second.x0), "=r"(second.x1)
      : "r"(address)
      : "memory");
#else
  (void)first;
  (void)second;
  (void)matrix;
  (void)n_panel;
  (void)k32;
  (void)lane;
#endif
}

__device__ __forceinline__ void stage_canonical_panel_swizzled(
    const std::uint16_t* const source, const unsigned int leading,
    const unsigned int column, std::uint16_t* const destination,
    const unsigned int token_base, const unsigned int token_count,
    const unsigned int thread) {
  constexpr unsigned int kVectors = kPanelBytes / sizeof(uint4);
  for (unsigned int vector = thread; vector < kVectors;
       vector += kThreads) {
    const unsigned int row = vector / 8U;
    const unsigned int logical_chunk = vector % 8U;
    const unsigned int physical_chunk =
        swizzled_chunk(row, logical_chunk);
    auto* const shared_destination =
        destination + row * kChunk + physical_chunk * 8U;
    if (token_base + row < token_count) {
      copy_16_async(shared_destination,
                    source + row * leading + column +
                        logical_chunk * 8U);
    } else {
      *reinterpret_cast<uint4*>(shared_destination) =
          make_uint4(0U, 0U, 0U, 0U);
    }
  }
  commit_async();
}

__device__ __forceinline__ void load_initial_state_bank(
    M16N8Accumulator* const state,
    const std::uint16_t* const state_input,
    const std::size_t head_state_base, const unsigned int value_half,
    const unsigned int bank,
    const unsigned int local_warp, const unsigned int lane) {
  const unsigned int group = lane >> 2U;
  const unsigned int pair = lane & 3U;
  const unsigned int key0 = local_warp * 16U + group;
  const unsigned int key1 = key0 + 8U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const unsigned int value0 = value_half * 64U + panel * 8U +
                                pair * 2U;
    const unsigned int value1 = value0 + 1U;
    const unsigned int bank_key0 = bank * 64U + key0;
    const unsigned int bank_key1 = bank * 64U + key1;
    state[panel] = {
        decode_bf16(state_input[head_state_base +
                                static_cast<std::size_t>(value0) *
                                    kDimension + bank_key0]),
        decode_bf16(state_input[head_state_base +
                                static_cast<std::size_t>(value1) *
                                    kDimension + bank_key0]),
        decode_bf16(state_input[head_state_base +
                                static_cast<std::size_t>(value0) *
                                    kDimension + bank_key1]),
        decode_bf16(state_input[head_state_base +
                                static_cast<std::size_t>(value1) *
                                    kDimension + bank_key1])};
  }
}

__device__ __forceinline__ void store_state_shared(
    const M16N8Accumulator* const state,
    std::uint16_t* const shared_state,
    const unsigned int local_warp, const unsigned int lane) {
  const unsigned int group = lane >> 2U;
  const unsigned int pair = lane & 3U;
  const unsigned int key0 = local_warp * 16U + group;
  const unsigned int key1 = key0 + 8U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const unsigned int value0 = panel * 8U + pair * 2U;
    const unsigned int value1 = value0 + 1U;
    const unsigned int chunk00 = swizzled_chunk(value0, key0 / 8U);
    const unsigned int chunk01 = swizzled_chunk(value0, key1 / 8U);
    const unsigned int chunk10 = swizzled_chunk(value1, key0 / 8U);
    const unsigned int chunk11 = swizzled_chunk(value1, key1 / 8U);
    const std::uint32_t packed0 =
        pack_bf16_pair(state[panel].x0, state[panel].x2);
    const std::uint32_t packed1 =
        pack_bf16_pair(state[panel].x1, state[panel].x3);
    shared_state[value0 * kChunk + chunk00 * 8U + key0 % 8U] =
        static_cast<std::uint16_t>(packed0);
    shared_state[value0 * kChunk + chunk01 * 8U + key1 % 8U] =
        static_cast<std::uint16_t>(packed0 >> 16U);
    shared_state[value1 * kChunk + chunk10 * 8U + key0 % 8U] =
        static_cast<std::uint16_t>(packed1);
    shared_state[value1 * kChunk + chunk11 * 8U + key1 % 8U] =
        static_cast<std::uint16_t>(packed1 >> 16U);
  }
}

__device__ __forceinline__ void load_state_shared(
    M16N8Accumulator* const state,
    const std::uint16_t* const shared_state,
    const unsigned int local_warp, const unsigned int lane) {
  const unsigned int group = lane >> 2U;
  const unsigned int pair = lane & 3U;
  const unsigned int key0 = local_warp * 16U + group;
  const unsigned int key1 = key0 + 8U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const unsigned int value0 = panel * 8U + pair * 2U;
    const unsigned int value1 = value0 + 1U;
    const unsigned int chunk00 = swizzled_chunk(value0, key0 / 8U);
    const unsigned int chunk01 = swizzled_chunk(value0, key1 / 8U);
    const unsigned int chunk10 = swizzled_chunk(value1, key0 / 8U);
    const unsigned int chunk11 = swizzled_chunk(value1, key1 / 8U);
    state[panel] = {
        decode_bf16(shared_state[value0 * kChunk + chunk00 * 8U +
                                 key0 % 8U]),
        decode_bf16(shared_state[value1 * kChunk + chunk10 * 8U +
                                 key0 % 8U]),
        decode_bf16(shared_state[value0 * kChunk + chunk01 * 8U +
                                 key1 % 8U]),
        decode_bf16(shared_state[value1 * kChunk + chunk11 * 8U +
                                 key1 % 8U])};
  }
}

// Park one FP32 state bank only across the W@H accumulator peak.  The
// component-major layout gives each warp contiguous, conflict-free shared
// accesses for every one of the 32 scalar planes.  Volatile is intentional:
// the reload is the new state definition after the intervening CTA barriers,
// rather than an invitation for ptxas to retain the old SSA values.
__device__ __forceinline__ void park_state_bank_one(
    const M16N8Accumulator* const state,
    volatile std::uint32_t* const parked, const unsigned int thread) {
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    parked[(panel * 4U + 0U) * kThreads + thread] =
        __float_as_uint(state[panel].x0);
    parked[(panel * 4U + 1U) * kThreads + thread] =
        __float_as_uint(state[panel].x1);
    parked[(panel * 4U + 2U) * kThreads + thread] =
        __float_as_uint(state[panel].x2);
    parked[(panel * 4U + 3U) * kThreads + thread] =
        __float_as_uint(state[panel].x3);
  }
}

__device__ __forceinline__ void restore_state_bank_one(
    M16N8Accumulator* const state,
    const volatile std::uint32_t* const parked,
    const unsigned int thread) {
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    state[panel] = {
        __uint_as_float(parked[(panel * 4U + 0U) * kThreads + thread]),
        __uint_as_float(parked[(panel * 4U + 1U) * kThreads + thread]),
        __uint_as_float(parked[(panel * 4U + 2U) * kThreads + thread]),
        __uint_as_float(parked[(panel * 4U + 3U) * kThreads + thread])};
  }
}

__device__ __forceinline__ void store_parked_state_shared(
    const volatile std::uint32_t* const parked,
    std::uint16_t* const shared_state, const unsigned int thread,
    const unsigned int local_warp, const unsigned int lane) {
  const unsigned int group = lane >> 2U;
  const unsigned int pair = lane & 3U;
  const unsigned int key0 = local_warp * 16U + group;
  const unsigned int key1 = key0 + 8U;
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const unsigned int value0 = panel * 8U + pair * 2U;
    const unsigned int value1 = value0 + 1U;
    const unsigned int chunk00 = swizzled_chunk(value0, key0 / 8U);
    const unsigned int chunk01 = swizzled_chunk(value0, key1 / 8U);
    const unsigned int chunk10 = swizzled_chunk(value1, key0 / 8U);
    const unsigned int chunk11 = swizzled_chunk(value1, key1 / 8U);
    const float x0 = __uint_as_float(
        parked[(panel * 4U + 0U) * kThreads + thread]);
    const float x1 = __uint_as_float(
        parked[(panel * 4U + 1U) * kThreads + thread]);
    const float x2 = __uint_as_float(
        parked[(panel * 4U + 2U) * kThreads + thread]);
    const float x3 = __uint_as_float(
        parked[(panel * 4U + 3U) * kThreads + thread]);
    const std::uint32_t packed0 = pack_bf16_pair(x0, x2);
    const std::uint32_t packed1 = pack_bf16_pair(x1, x3);
    shared_state[value0 * kChunk + chunk00 * 8U + key0 % 8U] =
        static_cast<std::uint16_t>(packed0);
    shared_state[value0 * kChunk + chunk01 * 8U + key1 % 8U] =
        static_cast<std::uint16_t>(packed0 >> 16U);
    shared_state[value1 * kChunk + chunk10 * 8U + key0 % 8U] =
        static_cast<std::uint16_t>(packed1);
    shared_state[value1 * kChunk + chunk11 * 8U + key1 % 8U] =
        static_cast<std::uint16_t>(packed1 >> 16U);
  }
}

__device__ __forceinline__ std::uint32_t load_shared_pair(
    const std::uint16_t* const matrix, const unsigned int row,
    const unsigned int panel, const unsigned int pair) {
  const unsigned int physical_chunk = swizzled_chunk(row, panel);
  return *reinterpret_cast<const std::uint32_t*>(
      matrix + row * kChunk + physical_chunk * 8U + pair * 2U);
}

__device__ __forceinline__ void store_transposed_value(
    std::uint16_t* const matrix, const unsigned int value,
    const unsigned int token, const std::uint16_t rounded) {
  const unsigned int physical_chunk =
      swizzled_chunk(value, token / 8U);
  matrix[value * kChunk + physical_chunk * 8U + token % 8U] = rounded;
}

__device__ __forceinline__ void copy_state_panel_global(
    const std::uint16_t* const shared_state,
    std::uint16_t* const global_state, const unsigned int value_half,
    const unsigned int bank, const unsigned int local_thread) {
  constexpr unsigned int kVectors = kPanelBytes / sizeof(uint4);
  for (unsigned int vector = local_thread; vector < kVectors;
       vector += 128U) {
    const unsigned int value = vector / 8U;
    const unsigned int logical_chunk = vector % 8U;
    const unsigned int physical_chunk =
        swizzled_chunk(value, logical_chunk);
    const uint4 packed = *reinterpret_cast<const uint4*>(
        shared_state + value * kChunk + physical_chunk * 8U);
    *reinterpret_cast<uint4*>(
        global_state +
        static_cast<std::size_t>(value_half * 64U + value) * kDimension +
        bank * 64U + logical_chunk * 8U) = packed;
  }
}

__device__ __forceinline__ std::uint16_t gated_score(
    const float raw, const unsigned int query, const unsigned int source,
    const float* const gamma) {
  return source <= query
             ? encode_bf16(raw * expf(gamma[query] - gamma[source]))
             : 0U;
}

__device__ __forceinline__ void store_score_accumulator(
    std::uint16_t* const destination,
    const M16N8Accumulator& accumulator, const unsigned int m_panel,
    const unsigned int n_panel, const unsigned int lane,
    const float* const gamma) {
  const unsigned int query0 = m_panel * 16U + lane / 4U;
  const unsigned int query1 = query0 + 8U;
  const unsigned int source0 = n_panel * 8U + 2U * (lane & 3U);
  const unsigned int source1 = source0 + 1U;
  const unsigned int chunk0 = swizzled_chunk(query0, source0 / 8U);
  const unsigned int chunk1 = swizzled_chunk(query1, source0 / 8U);
  *reinterpret_cast<std::uint32_t*>(
      destination + query0 * kChunk + chunk0 * 8U + source0 % 8U) =
      static_cast<std::uint32_t>(
          gated_score(accumulator.x0, query0, source0, gamma)) |
      (static_cast<std::uint32_t>(
           gated_score(accumulator.x1, query0, source1, gamma))
       << 16U);
  *reinterpret_cast<std::uint32_t*>(
      destination + query1 * kChunk + chunk1 * 8U + source0 % 8U) =
      static_cast<std::uint32_t>(
          gated_score(accumulator.x2, query1, source0, gamma)) |
      (static_cast<std::uint32_t>(
           gated_score(accumulator.x3, query1, source1, gamma))
       << 16U);
}

__device__ __forceinline__ void store_raw_output_accumulator(
    std::uint16_t* const raw_output,
    const M16N8Accumulator& accumulator, const unsigned int value_base,
    const unsigned int m_panel, const unsigned int n_panel,
    const unsigned int lane) {
  const unsigned int query0 = m_panel * 16U + lane / 4U;
  const unsigned int query1 = query0 + 8U;
  const unsigned int value =
      value_base + n_panel * 8U + 2U * (lane & 3U);
  *reinterpret_cast<std::uint32_t*>(
      raw_output + query0 * kDimension + value) =
      pack_bf16_pair(accumulator.x0, accumulator.x1);
  *reinterpret_cast<std::uint32_t*>(
      raw_output + query1 * kDimension + value) =
      pack_bf16_pair(accumulator.x2, accumulator.x3);
}

__device__ __forceinline__ void form_decayed_k_transpose(
    const std::uint16_t* const raw_k0,
    const std::uint16_t* const raw_k1, const float* const end_decay,
    std::uint16_t* const decayed_k0,
    std::uint16_t* const decayed_k1, const unsigned int thread) {
  for (unsigned int index = thread; index < kPanelElements;
       index += kThreads) {
    const unsigned int token = index / kChunk;
    const unsigned int key = index % kChunk;
    const unsigned int raw_chunk = swizzled_chunk(token, key / 8U);
    const unsigned int raw_offset =
        token * kChunk + raw_chunk * 8U + key % 8U;
    const unsigned int transposed_chunk =
        swizzled_chunk(key, token / 8U);
    const unsigned int transposed_offset =
        key * kChunk + transposed_chunk * 8U + token % 8U;
    const float scale = end_decay[token];
    decayed_k0[transposed_offset] = encode_bf16(
        decode_bf16(raw_k0[raw_offset]) * scale);
    decayed_k1[transposed_offset] = encode_bf16(
        decode_bf16(raw_k1[raw_offset]) * scale);
  }
}

template <bool Diagnostic>
__global__ __launch_bounds__(kThreads, 1) void c64_macro_kernel(
    const float* const raw_gram, const float* const global_gamma,
    const float* const global_beta,
    const std::uint16_t* const compact_q,
    const std::uint16_t* const compact_k,
    const std::uint16_t* const conv_qkv,
    const unsigned int token_count,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate, const float norm_epsilon,
    std::uint16_t* const output, DiagnosticBoundaries diagnostics) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* const slot0 =
      reinterpret_cast<std::uint16_t*>(shared_raw + 0U * kSlotBytes);
  auto* const slot1 =
      reinterpret_cast<std::uint16_t*>(shared_raw + 1U * kSlotBytes);
  auto* const slot2 =
      reinterpret_cast<std::uint16_t*>(shared_raw + 2U * kSlotBytes);
  auto* const slot3 =
      reinterpret_cast<std::uint16_t*>(shared_raw + 3U * kSlotBytes);
  auto* const slot4 =
      reinterpret_cast<std::uint16_t*>(shared_raw + 4U * kSlotBytes);
  auto* const gamma = reinterpret_cast<float*>(
      shared_raw + kMainSharedBytes);
  float* const beta = gamma + kChunk;
  auto* const parked_state =
      reinterpret_cast<volatile std::uint32_t*>(
          shared_raw + kMainSharedBytes + kScalarBytes);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread >> 5U;
  const unsigned int lane = thread & 31U;
  const unsigned int value_head = blockIdx.x;
  const unsigned int qk_head = value_head / kHeadGroup;
  const unsigned int value_half = warp >> 2U;
  const unsigned int local_warp = warp & 3U;
  const std::size_t state_base =
      static_cast<std::size_t>(value_head) * kDimension * kDimension;

  M16N8Accumulator state_zero[8U];
  load_initial_state_bank(state_zero, state_input, state_base, value_half,
                          0U, local_warp, lane);
  {
    M16N8Accumulator initial_state_one[8U];
    load_initial_state_bank(initial_state_one, state_input, state_base,
                            value_half, 1U, local_warp, lane);
    park_state_bank_one(initial_state_one, parked_state, thread);
  }
  __syncthreads();

  const unsigned int chunk_count =
      1U + (token_count - 1U) / kChunk;
  for (unsigned int chunk_index = 0U; chunk_index < chunk_count;
       ++chunk_index) {
    const unsigned int token_base = chunk_index * kChunk;
    const std::size_t value_matrix =
        static_cast<std::size_t>(chunk_index) * kValueHeads + value_head;
    const std::size_t qk_matrix =
        static_cast<std::size_t>(chunk_index) * kQkHeads + qk_head;
    const float* const matrix_raw =
        raw_gram + qk_matrix * kGramElements;
    const float* const matrix_gamma =
        global_gamma + value_matrix * kChunk;
    const float* const matrix_beta =
        global_beta + value_matrix * kChunk;
    const auto* const matrix_q = compact_q + qk_matrix * kVectorElements;
    const auto* const matrix_k = compact_k + qk_matrix * kVectorElements;

    if (thread < kChunk) {
      gamma[thread] = matrix_gamma[thread];
      beta[thread] = matrix_beta[thread];
    }

    // Phase S: exact lower solve. slot0/slot1 are scratch; slot2 retains the
    // only BF16 transform publication.
    auto* const packed_l = reinterpret_cast<float*>(slot0);
    auto* const packed_l_bf16 = reinterpret_cast<Bf16*>(slot1);
    auto* const packed_inverse = reinterpret_cast<Bf16*>(
        reinterpret_cast<unsigned char*>(slot1) + kPackedBf16Bytes);
    auto* const transform = reinterpret_cast<Bf16*>(slot2);
  for (unsigned int index = thread; index < kGramElements;
       index += kThreads) {
    const unsigned int row = index % kChunk;
    const unsigned int column = index / kChunk;
    if (row < column) {
      transform[index] = __float2bfloat16_rn(0.0F);
    }
  }
  __syncthreads();
  for (unsigned int packed_index = thread;
       packed_index < kPackedElements; packed_index += kThreads) {
    const unsigned int packed_block = packed_index / (kTile * kTile);
    const unsigned int tile_index = packed_index % (kTile * kTile);
    unsigned int row_block = 0U;
    unsigned int column_block = 0U;
    unpack_packed_block_index(packed_block, row_block, column_block);
    const unsigned int row = row_block * kTile + tile_index / kTile;
    const unsigned int column =
        column_block * kTile + tile_index % kTile;
    packed_l[packed_index] =
        row > column
            ? beta[row] * expf(gamma[row] - gamma[column]) *
                  matrix_raw[row * kChunk + column]
            : 0.0F;
  }
  __syncthreads();
  if (warp < 4U) {
    const unsigned int diagonal_offset =
        packed_block_index(warp, warp) * kTile * kTile;
    float inverse_column[kTile]{};
    if (lane < kTile) {
      solve_diagonal_inverse_column<0U>(
          packed_l + diagonal_offset, lane, inverse_column);
#pragma unroll
      for (unsigned int row = 0U; row < kTile; ++row) {
        const float value = inverse_column[row];
        packed_inverse[diagonal_offset + row * kTile + lane] =
            __float2bfloat16_rn(value);
        transform[warp * kTile + row +
                  (warp * kTile + lane) * kChunk] =
            __float2bfloat16_rn(value * beta[warp * kTile + lane]);
      }
    }
  }
  __syncthreads();
  for (unsigned int index = thread; index < kPackedElements;
       index += kThreads) {
    packed_l_bf16[index] = __float2bfloat16_rn(packed_l[index]);
  }
  __syncthreads();
  if (warp < 3U) {
    const unsigned int row_block = warp + 1U;
    const unsigned int column_block = warp;
    const unsigned int output_block =
        packed_block_index(row_block, column_block);
    form_transform_block(
        packed_inverse +
            packed_block_index(row_block, row_block) * kTile * kTile,
        packed_l_bf16 + output_block * kTile * kTile,
        packed_inverse +
            packed_block_index(column_block, column_block) * kTile * kTile,
        nullptr, nullptr, nullptr, nullptr,
        packed_inverse + output_block * kTile * kTile,
        packed_l + warp * kTile * kTile, beta, transform,
        row_block * kTile, column_block * kTile);
  }
  __syncthreads();
  if (warp < 2U) {
    const unsigned int row_block = warp + 2U;
    const unsigned int column_block = warp;
    const unsigned int middle_block = warp + 1U;
    const unsigned int output_block =
        packed_block_index(row_block, column_block);
    form_transform_block(
        packed_inverse +
            packed_block_index(row_block, row_block) * kTile * kTile,
        packed_l_bf16 + output_block * kTile * kTile,
        packed_inverse +
            packed_block_index(column_block, column_block) * kTile * kTile,
        packed_l_bf16 +
            packed_block_index(row_block, middle_block) * kTile * kTile,
        packed_inverse +
            packed_block_index(middle_block, column_block) * kTile * kTile,
        nullptr, nullptr,
        packed_inverse + output_block * kTile * kTile,
        packed_l + warp * kTile * kTile, beta, transform,
        row_block * kTile, column_block * kTile);
  }
  __syncthreads();
  if (warp == 0U) {
    constexpr unsigned int row_block = 3U;
    constexpr unsigned int output_block = packed_block_index(3U, 0U);
    form_transform_block(
        packed_inverse + packed_block_index(3U, 3U) * kTile * kTile,
        packed_l_bf16 + output_block * kTile * kTile,
        packed_inverse + packed_block_index(0U, 0U) * kTile * kTile,
        packed_l_bf16 + packed_block_index(3U, 1U) * kTile * kTile,
        packed_inverse + packed_block_index(1U, 0U) * kTile * kTile,
        packed_l_bf16 + packed_block_index(3U, 2U) * kTile * kTile,
        packed_inverse + packed_block_index(2U, 0U) * kTile * kTile,
        packed_inverse + output_block * kTile * kTile,
        packed_l, beta, transform, row_block * kTile, 0U);
  }
  __syncthreads();
  if constexpr (Diagnostic) {
    for (unsigned int index = thread; index < kGramElements;
         index += kThreads) {
      diagnostics.transform[
          value_matrix * kGramElements + index] =
          reinterpret_cast<std::uint16_t*>(transform)[index];
    }
  }
  __syncthreads();

  // beta is dead after the solve.  Reuse it as one shared exp(gamma)
  // table so W staging and QH gating issue 64 exponentials per CTA/chunk,
  // rather than one per W scalar.
  if (thread < kChunk) {
    beta[thread] = expf(gamma[thread]);
  }
  __syncthreads();

  // Phase R/W: two exact M64N64 panels.  slot0 becomes W, slot1 U, and
  // slot3 is the sole swizzled operand bank.
  auto* const operand = reinterpret_cast<Bf16*>(slot3);
  for (unsigned int panel = 0U; panel < 2U; ++panel) {
    for (unsigned int index = thread; index < kPanelElements;
         index += kThreads) {
      const unsigned int token = index / kChunk;
      const unsigned int column = index % kChunk;
      const unsigned int physical = swizzled_chunk(token, column / 8U);
      const bool valid_token = token_base + token < token_count;
      const float value =
          valid_token
              ? decode_bf16(matrix_k[token * kDimension +
                                     panel * kChunk + column]) *
                    beta[token]
              : 0.0F;
      operand[token * kChunk + physical * 8U + column % 8U] =
          __float2bfloat16_rn(value);
    }
    __syncthreads();
    if (thread < 64U) {
      recompute_product(transform, operand, slot0 + panel * kPanelElements,
                        warp, lane);
    }
    __syncthreads();
  }
  for (unsigned int panel = 0U; panel < 2U; ++panel) {
    for (unsigned int index = thread; index < kPanelElements;
         index += kThreads) {
      const unsigned int token = index / kChunk;
      const unsigned int column = index % kChunk;
      const unsigned int physical = swizzled_chunk(token, column / 8U);
      const std::size_t source =
          static_cast<std::size_t>(token_base + token) * kGdnQkvChannels +
          kVOffset +
          static_cast<std::size_t>(value_head) * kDimension +
          panel * kChunk + column;
      operand[token * kChunk + physical * 8U + column % 8U] =
          token_base + token < token_count
              ? reinterpret_cast<const Bf16*>(conv_qkv)[source]
              : __float2bfloat16_rn(0.0F);
    }
    __syncthreads();
    if (thread < 64U) {
      recompute_product(transform, operand, slot1 + panel * kPanelElements,
                        warp, lane);
    }
    __syncthreads();
  }

  if constexpr (Diagnostic) {
    for (unsigned int index = thread; index < kVectorElements;
         index += kThreads) {
      const unsigned int row = index / kDimension;
      const unsigned int column = index % kDimension;
      const unsigned int panel = column / kChunk;
      const unsigned int local = column % kChunk;
      const unsigned int physical = swizzled_chunk(row, local / 8U);
      const unsigned int offset =
          row * kChunk + physical * 8U + local % 8U;
      const std::size_t destination =
          value_matrix * kVectorElements + index;
      diagnostics.w[destination] = slot0[panel * kPanelElements + offset];
      diagnostics.u[destination] = slot1[panel * kPanelElements + offset];
    }
  }
  __syncthreads();

  // Raw K replaces dead T in slot2.  slot3 is also dead after the W/U
  // producers, so retain both rounded state banks before products becomes
  // live.  FP32 bank1 remains shared-owned across the entire W@H/score/QH
  // interval and is restored only for the final recurrent state update.
  stage_canonical_panel_swizzled(matrix_k, kDimension, 0U, slot2,
                                  token_base, token_count, thread);
  stage_canonical_panel_swizzled(matrix_k, kDimension, 64U,
                                  slot2 + kPanelElements, token_base,
                                  token_count, thread);
  wait_all_async();
  __syncthreads();

  std::uint16_t* const shared_state_zero =
      slot4 + value_half * kPanelElements;
  store_state_shared(state_zero, shared_state_zero, local_warp, lane);
  __syncthreads();
  if constexpr (Diagnostic) {
    copy_state_panel_global(
        shared_state_zero,
        diagnostics.boundary_state +
            value_matrix * kDimension * kDimension,
        value_half, 0U, thread - value_half * 128U);
  }
  __syncthreads();

  std::uint16_t* const shared_state_one =
      slot3 + value_half * kPanelElements;
  store_parked_state_shared(parked_state, shared_state_one, thread,
                            local_warp, lane);
  __syncthreads();
  if constexpr (Diagnostic) {
    copy_state_panel_global(
        shared_state_one,
        diagnostics.boundary_state +
            value_matrix * kDimension * kDimension,
        value_half, 1U, thread - value_half * 128U);
  }
  __syncthreads();

  M16N8Accumulator products[8U];
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    zero(products[panel]);
  }
#pragma unroll
  for (unsigned int bank = 0U; bank < 2U; ++bank) {
    const std::uint16_t* const shared_state =
        (bank == 0U ? slot4 : slot3) + value_half * kPanelElements;
    const std::uint16_t* const selected_w =
        slot0 + bank * kPanelElements;
#pragma unroll 1
    for (unsigned int key_pair = 0U; key_pair < 2U; ++key_pair) {
      M16K16 w0{}, w1{};
      load_a_swizzled(w0, selected_w, local_warp, key_pair * 2U, lane);
      load_a_swizzled(w1, selected_w, local_warp,
                       key_pair * 2U + 1U, lane);
#pragma unroll
      for (unsigned int panel = 0U; panel < 8U; ++panel) {
        K16N8 h0{}, h1{};
        load_b_pair_swizzled(h0, h1, shared_state, panel, key_pair,
                             lane);
        mma(products[panel], w0, h0);
        mma(products[panel], w1, h1);
      }
    }
    // Keep the two bank consumers in distinct scheduler windows.  Without
    // this boundary ptxas hoists fragments from both banks together and
    // recreates the register peak that the state parking removes.
    __syncthreads();
  }

  const unsigned int group = lane >> 2U;
  const unsigned int pair = lane & 3U;
  const unsigned int token0 = local_warp * 16U + group;
  const unsigned int token1 = token0 + 8U;
  std::uint16_t* const shared_vnew =
      slot3 + value_half * kPanelElements;
#pragma unroll
  for (unsigned int panel = 0U; panel < 8U; ++panel) {
    const std::uint32_t u0 =
        load_shared_pair(slot1 + value_half * kPanelElements,
                         token0, panel, pair);
    const std::uint32_t u1 =
        load_shared_pair(slot1 + value_half * kPanelElements,
                         token1, panel, pair);
    const std::uint32_t rounded0 = pack_bf16_pair(
        decode_bf16(static_cast<std::uint16_t>(u0)) - products[panel].x0,
        decode_bf16(static_cast<std::uint16_t>(u0 >> 16U)) -
            products[panel].x1);
    const std::uint32_t rounded1 = pack_bf16_pair(
        decode_bf16(static_cast<std::uint16_t>(u1)) - products[panel].x2,
        decode_bf16(static_cast<std::uint16_t>(u1 >> 16U)) -
            products[panel].x3);
    const unsigned int value0 = panel * 8U + pair * 2U;
    const unsigned int value1 = value0 + 1U;
    store_transposed_value(shared_vnew, value0, token0,
                           static_cast<std::uint16_t>(rounded0));
    store_transposed_value(shared_vnew, value1, token0,
                           static_cast<std::uint16_t>(rounded0 >> 16U));
    store_transposed_value(shared_vnew, value0, token1,
                           static_cast<std::uint16_t>(rounded1));
    store_transposed_value(shared_vnew, value1, token1,
                           static_cast<std::uint16_t>(rounded1 >> 16U));
  }
  __syncthreads();
  if constexpr (Diagnostic) {
    for (unsigned int index = thread; index < kVectorElements;
         index += kThreads) {
      const unsigned int token = index / kDimension;
      const unsigned int value = index % kDimension;
      const unsigned int half = value / 64U;
      const unsigned int local_value = value % 64U;
      const unsigned int physical =
          swizzled_chunk(local_value, token / 8U);
      diagnostics.v_new[
          value_matrix * kVectorElements + index] =
          slot3[half * kPanelElements + local_value * kChunk +
                physical * 8U + token % 8U];
    }
  }
  __syncthreads();

  // Q replaces dead U.  Score is computed once per value head (not once per
  // value half) into slot4[0:8KiB].
  stage_canonical_panel_swizzled(matrix_q, kDimension, 0U, slot1,
                                  token_base, token_count, thread);
  stage_canonical_panel_swizzled(matrix_q, kDimension, 64U,
                                  slot1 + kPanelElements, token_base,
                                  token_count, thread);
  wait_all_async();
  __syncthreads();
  if (warp < 4U) {
    M16N8Accumulator score[8U];
#pragma unroll
    for (unsigned int panel = 0U; panel < 8U; ++panel) {
      zero(score[panel]);
    }
#pragma unroll
    for (unsigned int bank = 0U; bank < 2U; ++bank) {
      const std::uint16_t* const selected_q =
          slot1 + bank * kPanelElements;
      const std::uint16_t* const selected_k =
          slot2 + bank * kPanelElements;
#pragma unroll
      for (unsigned int key_pair = 0U; key_pair < 2U; ++key_pair) {
        M16K16 q0{}, q1{};
        load_a_swizzled(q0, selected_q, warp, key_pair * 2U, lane);
        load_a_swizzled(q1, selected_q, warp,
                         key_pair * 2U + 1U, lane);
#pragma unroll
        for (unsigned int panel = 0U; panel < 8U; ++panel) {
          K16N8 k0{}, k1{};
          load_b_pair_swizzled(k0, k1, selected_k, panel, key_pair,
                               lane);
          mma(score[panel], q0, k0);
          mma(score[panel], q1, k1);
        }
      }
    }
#pragma unroll
    for (unsigned int panel = 0U; panel < 8U; ++panel) {
      store_score_accumulator(slot4, score[panel], warp, panel, lane,
                              gamma);
    }
  }
  __syncthreads();

  // Compute one value half at a time. slot4[8:16KiB] is the rounded H bank;
  // the lower half retains the single causal score tile.
#pragma unroll
  for (unsigned int selected_half = 0U; selected_half < 2U;
       ++selected_half) {
    const bool active_half = value_half == selected_half;
    M16N8Accumulator qh[8U];
#pragma unroll
    for (unsigned int panel = 0U; panel < 8U; ++panel) {
      zero(qh[panel]);
    }
#pragma unroll
    for (unsigned int bank = 0U; bank < 2U; ++bank) {
      std::uint16_t* const shared_state = slot4 + kPanelElements;
      if (active_half) {
        if (bank == 0U) {
          store_state_shared(state_zero, shared_state, local_warp, lane);
        } else {
          store_parked_state_shared(parked_state, shared_state, thread,
                                    local_warp, lane);
        }
      }
      __syncthreads();
      if (active_half) {
        const std::uint16_t* const selected_q =
            slot1 + bank * kPanelElements;
#pragma unroll
        for (unsigned int key_pair = 0U; key_pair < 2U; ++key_pair) {
          M16K16 q0{}, q1{};
          load_a_swizzled(q0, selected_q, local_warp,
                           key_pair * 2U, lane);
          load_a_swizzled(q1, selected_q, local_warp,
                           key_pair * 2U + 1U, lane);
#pragma unroll
          for (unsigned int panel = 0U; panel < 8U; ++panel) {
            K16N8 h0{}, h1{};
            load_b_pair_swizzled(h0, h1, shared_state, panel,
                                 key_pair, lane);
            mma(qh[panel], q0, h0);
            mma(qh[panel], q1, h1);
          }
        }
      }
      __syncthreads();
    }
    if (active_half) {
      const float gate0 = beta[token0];
      const float gate1 = beta[token1];
#pragma unroll
      for (unsigned int panel = 0U; panel < 8U; ++panel) {
        qh[panel].x0 *= gate0;
        qh[panel].x1 *= gate0;
        qh[panel].x2 *= gate1;
        qh[panel].x3 *= gate1;
      }
#pragma unroll
      for (unsigned int source_pair = 0U; source_pair < 2U;
           ++source_pair) {
        M16K16 score0{}, score1{};
        load_a_swizzled(score0, slot4, local_warp,
                         source_pair * 2U, lane);
        load_a_swizzled(score1, slot4, local_warp,
                         source_pair * 2U + 1U, lane);
#pragma unroll
        for (unsigned int panel = 0U; panel < 8U; ++panel) {
          K16N8 v0{}, v1{};
          load_b_pair_swizzled(v0, v1,
                               slot3 + selected_half * kPanelElements,
                               panel, source_pair, lane);
          mma(qh[panel], score0, v0);
          mma(qh[panel], score1, v1);
        }
      }
#pragma unroll
      for (unsigned int panel = 0U; panel < 8U; ++panel) {
        store_raw_output_accumulator(
            slot0, qh[panel], selected_half * 64U, local_warp,
            panel, lane);
      }
    }
    __syncthreads();
  }

  if constexpr (Diagnostic) {
    for (unsigned int index = thread; index < kVectorElements;
         index += kThreads) {
      const unsigned int token = index / kDimension;
      const unsigned int value = index % kDimension;
      diagnostics.raw_output[
          (static_cast<std::size_t>(token_base + token) * kValueHeads +
           value_head) *
              kDimension + value] = slot0[index];
    }
  }
  __syncthreads();

  // Exact rows-8 epilogue, directly into the final consumer buffer.  The
  // load precedes the same-element store, so output==silu_gate is valid.
#pragma unroll 1
  for (unsigned int token = warp; token < kChunk; token += kWarps) {
    const unsigned int global_token = token_base + token;
    if (global_token >= token_count) {
      continue;
    }
    const float value0 = decode_bf16(slot0[token * kDimension + lane]);
    const float value1 =
        decode_bf16(slot0[token * kDimension + lane + 32U]);
    const float value2 =
        decode_bf16(slot0[token * kDimension + lane + 64U]);
    const float value3 =
        decode_bf16(slot0[token * kDimension + lane + 96U]);
    float square_sum = fmaf(value0, value0, 0.0F);
    square_sum = fmaf(value1, value1, square_sum);
    square_sum = fmaf(value2, value2, square_sum);
    square_sum = fmaf(value3, value3, square_sum);
#pragma unroll
    for (unsigned int offset = 16U; offset != 0U; offset >>= 1U) {
      square_sum +=
          __shfl_down_sync(0xffff'ffffU, square_sum, offset);
    }
    const float inverse_rms = __shfl_sync(
        0xffff'ffffU,
        rsqrtf(square_sum / static_cast<float>(kDimension) +
               norm_epsilon),
        0U);
    const float values[4U] = {value0, value1, value2, value3};
#pragma unroll
    for (unsigned int index = 0U; index < 4U; ++index) {
      const unsigned int value = lane + index * 32U;
      const std::size_t destination =
          (static_cast<std::size_t>(global_token) * kValueHeads +
           value_head) *
              kDimension + value;
      const float gate = decode_bf16(silu_gate[destination]);
      const float normalized =
          values[index] * inverse_rms * decode_bf16(norm_weight[value]);
      output[destination] = encode_bf16(
          normalized * gate / (1.0F + expf(-gate)));
    }
  }
  __syncthreads();

  // Only the update phase needs bank1 in FP32 registers.  Its lexical scope
  // begins after score/QH and ends before the next chunk's W@H products.
  {
    M16N8Accumulator state_one[8U];
    restore_state_bank_one(state_one, parked_state, thread);
    __syncthreads();

    // Exact end-decayed-K update lands directly in the resident FP32 states.
    if (thread < kChunk) {
      beta[thread] = expf(gamma[kChunk - 1U] - gamma[thread]);
    }
    __syncthreads();
    // Every lane consumes the same state decay.  Gamma is dead after the
    // end-decay table above, so publish the single exponential through its
    // existing shared scalar bank rather than issuing it once per thread.
    if (thread == 0U) {
      gamma[0] = expf(gamma[kChunk - 1U]);
    }
    __syncthreads();
    form_decayed_k_transpose(
        slot2, slot2 + kPanelElements, beta, slot1,
        slot1 + kPanelElements, thread);
    const float state_decay = gamma[0];
#pragma unroll
    for (unsigned int panel = 0U; panel < 8U; ++panel) {
      state_zero[panel].x0 *= state_decay;
      state_zero[panel].x1 *= state_decay;
      state_zero[panel].x2 *= state_decay;
      state_zero[panel].x3 *= state_decay;
      state_one[panel].x0 *= state_decay;
      state_one[panel].x1 *= state_decay;
      state_one[panel].x2 *= state_decay;
      state_one[panel].x3 *= state_decay;
    }
    __syncthreads();
#pragma unroll
    for (unsigned int bank = 0U; bank < 2U; ++bank) {
      const std::uint16_t* const selected_k =
          slot1 + bank * kPanelElements;
#pragma unroll
      for (unsigned int token_pair = 0U; token_pair < 2U; ++token_pair) {
        M16K16 k0{}, k1{};
        load_a_swizzled(k0, selected_k, local_warp,
                        token_pair * 2U, lane);
        load_a_swizzled(k1, selected_k, local_warp,
                        token_pair * 2U + 1U, lane);
#pragma unroll
        for (unsigned int panel = 0U; panel < 8U; ++panel) {
          K16N8 v0{}, v1{};
          load_b_pair_swizzled(v0, v1,
                               slot3 + value_half * kPanelElements,
                               panel, token_pair, lane);
          if (bank == 0U) {
            mma(state_zero[panel], k0, v0);
            mma(state_zero[panel], k1, v1);
          } else {
            mma(state_one[panel], k0, v0);
            mma(state_one[panel], k1, v1);
          }
        }
      }
    }
    __syncthreads();

    // Match the incumbent long-span contract: every eighth completed C64
    // chunk is rounded through BF16 before the next chunk consumes state.
    if ((chunk_index + 1U) % 8U == 0U &&
        chunk_index + 1U < chunk_count) {
      std::uint16_t* const shared_state =
          slot4 + value_half * kPanelElements;
      store_state_shared(state_zero, shared_state, local_warp, lane);
      __syncthreads();
      load_state_shared(state_zero, shared_state, local_warp, lane);
      __syncthreads();
      store_state_shared(state_one, shared_state, local_warp, lane);
      __syncthreads();
      load_state_shared(state_one, shared_state, local_warp, lane);
      __syncthreads();
    }
    park_state_bank_one(state_one, parked_state, thread);
    __syncthreads();
  }
  }

  // Final state is rounded once at the API boundary.
  std::uint16_t* const shared_state =
      slot4 + value_half * kPanelElements;
  store_state_shared(state_zero, shared_state, local_warp, lane);
  __syncthreads();
  copy_state_panel_global(shared_state, state_output + state_base,
                          value_half, 0U,
                          thread - value_half * 128U);
  __syncthreads();
  store_parked_state_shared(parked_state, shared_state, thread,
                            local_warp, lane);
  __syncthreads();
  copy_state_panel_global(shared_state, state_output + state_base,
                          value_half, 1U,
                          thread - value_half * 128U);
  __syncthreads();
}

[[nodiscard]] bool invalid_arguments(
    const float* const raw_gram, const float* const gamma,
    const float* const beta, const std::uint16_t* const compact_q,
    const std::uint16_t* const compact_k,
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const state_input,
    const std::uint16_t* const state_output,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate, const float norm_epsilon,
    const std::uint16_t* const output) noexcept {
  return raw_gram == nullptr || gamma == nullptr || beta == nullptr ||
         compact_q == nullptr || compact_k == nullptr ||
         conv_qkv == nullptr || state_input == nullptr ||
         state_output == nullptr || norm_weight == nullptr ||
         silu_gate == nullptr || output == nullptr ||
         token_count == 0U ||
         token_count > std::numeric_limits<unsigned int>::max() ||
         !std::isfinite(norm_epsilon) || norm_epsilon <= 0.0F;
}

template <bool Diagnostic>
[[nodiscard]] int launch_impl(
    const float* const raw_gram, const float* const gamma,
    const float* const beta, const std::uint16_t* const compact_q,
    const std::uint16_t* const compact_k,
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate, const float norm_epsilon,
    std::uint16_t* const output, const DiagnosticBoundaries diagnostics,
    void* const cuda_stream) noexcept {
  if (invalid_arguments(raw_gram, gamma, beta, compact_q, compact_k,
                        conv_qkv, token_count, state_input, state_output,
                        norm_weight, silu_gate, norm_epsilon, output)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  if constexpr (Diagnostic) {
    if (diagnostics.transform == nullptr || diagnostics.w == nullptr ||
        diagnostics.u == nullptr || diagnostics.v_new == nullptr ||
        diagnostics.boundary_state == nullptr ||
        diagnostics.raw_output == nullptr) {
      return static_cast<int>(cudaErrorInvalidValue);
    }
  }
  static const int attribute_status = static_cast<int>(
      cudaFuncSetAttribute(c64_macro_kernel<Diagnostic>,
                           cudaFuncAttributeMaxDynamicSharedMemorySize,
                           static_cast<int>(kSharedBytes)));
  if (attribute_status != static_cast<int>(cudaSuccess)) {
    return attribute_status;
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  (void)cudaGetLastError();
  c64_macro_kernel<Diagnostic><<<kValueHeads, kThreads, kSharedBytes,
                                stream>>>(
      raw_gram, gamma, beta, compact_q, compact_k, conv_qkv,
      static_cast<unsigned int>(token_count), state_input, state_output,
      norm_weight, silu_gate, norm_epsilon, output, diagnostics);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

int launch_c64(const float* const raw_gram, const float* const gamma,
               const float* const beta,
               const std::uint16_t* const compact_q,
               const std::uint16_t* const compact_k,
               const std::uint16_t* const conv_qkv,
               const std::size_t token_count,
               const std::uint16_t* const state_input,
               std::uint16_t* const state_output,
               const std::uint16_t* const norm_weight,
               const std::uint16_t* const silu_gate,
               const float norm_epsilon, std::uint16_t* const output,
               void* const cuda_stream) noexcept {
  return launch_impl<false>(
      raw_gram, gamma, beta, compact_q, compact_k, conv_qkv, token_count,
      state_input, state_output, norm_weight, silu_gate, norm_epsilon,
      output, {}, cuda_stream);
}

int launch_c64_diagnostic(
    const float* const raw_gram, const float* const gamma,
    const float* const beta, const std::uint16_t* const compact_q,
    const std::uint16_t* const compact_k,
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count,
    const std::uint16_t* const state_input,
    std::uint16_t* const state_output,
    const std::uint16_t* const norm_weight,
    const std::uint16_t* const silu_gate, const float norm_epsilon,
    std::uint16_t* const output, const DiagnosticBoundaries boundaries,
    void* const cuda_stream) noexcept {
  return launch_impl<true>(
      raw_gram, gamma, beta, compact_q, compact_k, conv_qkv, token_count,
      state_input, state_output, norm_weight, silu_gate, norm_epsilon,
      output, boundaries, cuda_stream);
}

int query_c64_resources(int* const registers_per_thread,
                        std::size_t* const static_shared_bytes,
                        std::size_t* const dynamic_shared_bytes,
                        std::size_t* const local_bytes,
                        int* const maximum_threads_per_block,
                        int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      dynamic_shared_bytes == nullptr || local_bytes == nullptr ||
      maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaError_t status = cudaFuncSetAttribute(
      c64_macro_kernel<false>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  cudaFuncAttributes attributes{};
  status = cudaFuncGetAttributes(&attributes, c64_macro_kernel<false>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, c64_macro_kernel<false>, static_cast<int>(kThreads),
      kSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *dynamic_shared_bytes = kSharedBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active;
  return static_cast<int>(cudaSuccess);
}

const void* c64_kernel_handle_for_test() noexcept {
  return reinterpret_cast<const void*>(c64_macro_kernel<false>);
}

}  // namespace q3x::runtime::gdn_prefill_prompt_span_macro_detail
