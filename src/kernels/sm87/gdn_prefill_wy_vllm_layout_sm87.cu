#include "gdn_prefill_wy_vllm_layout_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_wy_vllm_layout_detail {
namespace {

constexpr unsigned int kChunk = 64U;
constexpr unsigned int kDimension = 128U;
constexpr unsigned int kQkHeads = 16U;
constexpr unsigned int kValueHeads = 48U;
constexpr unsigned int kHeadGroup = 3U;
constexpr unsigned int kTile = 16U;
constexpr unsigned int kBlockRows = kChunk / kTile;
constexpr unsigned int kPackedBlocks =
    kBlockRows * (kBlockRows + 1U) / 2U;
constexpr unsigned int kMaximumChunks = 8U;
constexpr unsigned int kMaximumRawGramChunks = 64U;

constexpr std::size_t kVectorElements = kChunk * kDimension;
constexpr std::size_t kGramElements = kChunk * kChunk;
constexpr std::size_t kTileElements = kTile * kTile;
constexpr std::size_t kPackedElements = kPackedBlocks * kTileElements;

namespace wmma = nvcuda::wmma;
using Bf16 = __nv_bfloat16;
using Accumulator =
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>;

__device__ __forceinline__ void copy_16_async(
    void* const shared_destination,
    const void* const global_source) {
  const unsigned int shared_address = static_cast<unsigned int>(
      __cvta_generic_to_shared(shared_destination));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::
                   "r"(shared_address), "l"(global_source));
}

__device__ __forceinline__ void commit_and_wait_async_copies() {
  asm volatile("cp.async.commit_group;\n"
               "cp.async.wait_group 0;\n" ::);
}

__host__ __device__ constexpr unsigned int packed_block_index(
    const unsigned int row_block,
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

// The executed vLLM specialization streams two K64 panels through an 8 KiB
// shared bank.  This compact variant retains that lifetime but computes KKT
// once per Q/K head.  Only the ten lower 16x16 blocks are issued and
// published; the six upper products are dead in the triangular WY solve.
constexpr unsigned int kGramThreads = 128U;
constexpr unsigned int kGramWarps = kGramThreads / 32U;
constexpr unsigned int kGramPanel = 64U;
constexpr unsigned int kGramVectorsPerRow =
    kGramPanel * sizeof(Bf16) / sizeof(uint4);
constexpr unsigned int kGramVectorCount =
    kChunk * kGramVectorsPerRow;
constexpr std::size_t kGramSharedBytes =
    kChunk * kGramPanel * sizeof(Bf16);

static_assert(kGramWarps == kBlockRows);
static_assert(kGramSharedBytes == 8U * 1024U);

__global__ __launch_bounds__(kGramThreads)
void compact_lower_gram_chunk64_kernel(
    const std::uint16_t* const compact_k,
    const unsigned int chunk_count, float* const raw_gram) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* const shared_k = reinterpret_cast<Bf16*>(shared_raw);
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const std::size_t compact_matrix = blockIdx.x;
  const unsigned int chunk =
      static_cast<unsigned int>(compact_matrix / kQkHeads);
  if (chunk >= chunk_count) {
    return;
  }
  const auto* const matrix_k = reinterpret_cast<const Bf16*>(
      compact_k + compact_matrix * kVectorElements);
  float* const matrix_raw = raw_gram + compact_matrix * kGramElements;

  Accumulator gram[kBlockRows];
#pragma unroll
  for (unsigned int column_block = 0U; column_block < kBlockRows;
       ++column_block) {
    wmma::fill_fragment(gram[column_block], 0.0F);
  }

#pragma unroll
  for (unsigned int panel = 0U; panel < kDimension / kGramPanel;
       ++panel) {
    for (unsigned int vector = thread; vector < kGramVectorCount;
         vector += kGramThreads) {
      const unsigned int row = vector / kGramVectorsPerRow;
      const unsigned int vector_in_row =
          vector % kGramVectorsPerRow;
      const auto* const source = reinterpret_cast<const uint4*>(
          matrix_k + row * kDimension + panel * kGramPanel) +
          vector_in_row;
      copy_16_async(reinterpret_cast<uint4*>(shared_k) + vector,
                    source);
    }
    commit_and_wait_async_copies();
    __syncthreads();

#pragma unroll
    for (unsigned int dimension_block = 0U;
         dimension_block < kGramPanel / kTile; ++dimension_block) {
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16,
                     wmma::row_major>
          row_fragment;
      wmma::load_matrix_sync(
          row_fragment,
          shared_k + warp * kTile * kGramPanel +
              dimension_block * kTile,
          static_cast<int>(kGramPanel));
#pragma unroll
      for (unsigned int column_block = 0U;
           column_block < kBlockRows; ++column_block) {
        if (column_block <= warp) {
          wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16,
                         wmma::col_major>
              column_fragment;
          wmma::load_matrix_sync(
              column_fragment,
              shared_k + column_block * kTile * kGramPanel +
                  dimension_block * kTile,
              static_cast<int>(kGramPanel));
          wmma::mma_sync(gram[column_block], row_fragment,
                         column_fragment, gram[column_block]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (unsigned int column_block = 0U; column_block < kBlockRows;
       ++column_block) {
    if (column_block <= warp) {
      wmma::store_matrix_sync(
          matrix_raw + warp * kTile * kChunk + column_block * kTile,
          gram[column_block], static_cast<int>(kChunk),
          wmma::mem_row_major);
    }
  }
}

template <unsigned int Row, unsigned int Inner>
__device__ __forceinline__ void accumulate_diagonal_inverse_column(
    const float* const diagonal_l, const unsigned int lane,
    const float (&inverse_column)[kTile], float& value) {
  if constexpr (Inner < Row) {
    if (lane <= Inner) {
      value -= diagonal_l[Row * kTile + Inner] *
               inverse_column[Inner];
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
  const unsigned int lane = threadIdx.x % 32U;
  Accumulator inner;
  wmma::fill_fragment(inner, 0.0F);
  accumulate_row_major_product(inner, left0, right0);
  if (left1 != nullptr && right1 != nullptr) {
    accumulate_row_major_product(inner, left1, right1);
  }
  if (left2 != nullptr && right2 != nullptr) {
    accumulate_row_major_product(inner, left2, right2);
  }
  wmma::store_matrix_sync(scratch, inner, static_cast<int>(kTile),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane; index < kTileElements; index += 32U) {
    inverse[index] = __float2bfloat16_rn(scratch[index]);
  }
  __syncwarp();

  Accumulator outer;
  wmma::fill_fragment(outer, 0.0F);
  accumulate_row_major_product(outer, diagonal_inverse, inverse);
  wmma::store_matrix_sync(scratch, outer, static_cast<int>(kTile),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane; index < kTileElements; index += 32U) {
    const unsigned int row = index / kTile;
    const unsigned int column = index % kTile;
    const float value = -scratch[index];
    inverse[index] = __float2bfloat16_rn(value);
    const unsigned int output_row = row_base + row;
    const unsigned int output_column = column_base + column;
    transform[output_row + output_column * kChunk] =
        __float2bfloat16_rn(value * beta[output_column]);
  }
  __syncwarp();
}

// A lower-triangular packed-tile lifetime replaces the incumbent full
// 64x64 L/Lbf16/inverse banks.  FP32 L occupies ten tiles; its dead prefix is
// reused as four independent WMMA scratch tiles after the BF16 publication.
constexpr unsigned int kSolveThreads = 128U;
constexpr unsigned int kSolveWarps = kSolveThreads / 32U;
constexpr std::size_t kPackedFp32Bytes =
    kPackedElements * sizeof(float);
constexpr std::size_t kPackedBf16Bytes =
    kPackedElements * sizeof(Bf16);
constexpr std::size_t kSolveSharedBytes =
    kPackedFp32Bytes + 2U * kPackedBf16Bytes +
    2U * kChunk * sizeof(float);

static_assert(kSolveWarps == kBlockRows);
static_assert(kSolveSharedBytes == 20U * 1024U + 512U);
static_assert(kSolveWarps * kTileElements * sizeof(float) <=
              kPackedFp32Bytes);

__global__ __launch_bounds__(kSolveThreads)
void value_head_solve_chunk64_kernel(
    const float* const compact_raw_gram,
    const float* const cumulative_gate, const float* const beta,
    const unsigned int chunk_count, std::uint16_t* const transform) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* const shared_l = reinterpret_cast<float*>(shared_raw);
  auto* const shared_l_bf16 = reinterpret_cast<Bf16*>(
      shared_raw + kPackedFp32Bytes);
  auto* const shared_inverse = reinterpret_cast<Bf16*>(
      shared_raw + kPackedFp32Bytes + kPackedBf16Bytes);
  auto* const shared_gamma = reinterpret_cast<float*>(
      shared_raw + kPackedFp32Bytes + 2U * kPackedBf16Bytes);
  float* const shared_beta = shared_gamma + kChunk;
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const std::size_t matrix = blockIdx.x;
  const unsigned int chunk =
      static_cast<unsigned int>(matrix / kValueHeads);
  if (chunk >= chunk_count) {
    return;
  }
  const unsigned int value_head =
      static_cast<unsigned int>(matrix % kValueHeads);
  const unsigned int qk_head = value_head / kHeadGroup;
  const std::size_t compact_matrix =
      static_cast<std::size_t>(chunk) * kQkHeads + qk_head;
  const float* const matrix_raw =
      compact_raw_gram + compact_matrix * kGramElements;
  const float* const matrix_gamma =
      cumulative_gate + matrix * kChunk;
  const float* const matrix_beta = beta + matrix * kChunk;
  auto* const matrix_transform = reinterpret_cast<Bf16*>(
      transform + matrix * kGramElements);

  if (thread < kChunk) {
    shared_gamma[thread] = matrix_gamma[thread];
    shared_beta[thread] = matrix_beta[thread];
  }
  for (unsigned int index = thread; index < kGramElements;
       index += kSolveThreads) {
    const unsigned int row = index % kChunk;
    const unsigned int column = index / kChunk;
    if (row < column) {
      matrix_transform[index] = __float2bfloat16_rn(0.0F);
    }
  }
  __syncthreads();

  for (unsigned int packed_index = thread;
       packed_index < kPackedElements; packed_index += kSolveThreads) {
    const unsigned int packed_block = packed_index / kTileElements;
    const unsigned int tile_index = packed_index % kTileElements;
    unsigned int row_block = 0U;
    unsigned int column_block = 0U;
    unpack_packed_block_index(packed_block, row_block, column_block);
    const unsigned int row = row_block * kTile + tile_index / kTile;
    const unsigned int column =
        column_block * kTile + tile_index % kTile;
    float value = 0.0F;
    if (row > column) {
      value = shared_beta[row] *
              expf(shared_gamma[row] - shared_gamma[column]) *
              matrix_raw[row * kChunk + column];
    }
    shared_l[packed_index] = value;
  }
  __syncthreads();

  const unsigned int diagonal_block = packed_block_index(warp, warp);
  const unsigned int diagonal_offset =
      diagonal_block * kTileElements;
  float inverse_column[kTile]{};
  if (lane < kTile) {
    solve_diagonal_inverse_column<0U>(
        shared_l + diagonal_offset, lane, inverse_column);
#pragma unroll
    for (unsigned int row = 0U; row < kTile; ++row) {
      const float value = inverse_column[row];
      shared_inverse[diagonal_offset + row * kTile + lane] =
          __float2bfloat16_rn(value);
      const unsigned int output_row = warp * kTile + row;
      const unsigned int output_column = warp * kTile + lane;
      matrix_transform[output_row + output_column * kChunk] =
          __float2bfloat16_rn(value * shared_beta[output_column]);
    }
  }
  __syncthreads();

  for (unsigned int index = thread; index < kPackedElements;
       index += kSolveThreads) {
    shared_l_bf16[index] = __float2bfloat16_rn(shared_l[index]);
  }
  __syncthreads();

  float* const warp_scratch =
      shared_l + warp * kTileElements;
  if (warp < 3U) {
    const unsigned int row_block = warp + 1U;
    const unsigned int column_block = warp;
    const unsigned int output_block =
        packed_block_index(row_block, column_block);
    form_transform_block(
        shared_inverse +
            packed_block_index(row_block, row_block) * kTileElements,
        shared_l_bf16 + output_block * kTileElements,
        shared_inverse +
            packed_block_index(column_block, column_block) *
                kTileElements,
        nullptr, nullptr, nullptr, nullptr,
        shared_inverse + output_block * kTileElements, warp_scratch,
        shared_beta, matrix_transform, row_block * kTile,
        column_block * kTile);
  }
  __syncthreads();

  if (warp < 2U) {
    const unsigned int row_block = warp + 2U;
    const unsigned int column_block = warp;
    const unsigned int middle_block = warp + 1U;
    const unsigned int output_block =
        packed_block_index(row_block, column_block);
    form_transform_block(
        shared_inverse +
            packed_block_index(row_block, row_block) * kTileElements,
        shared_l_bf16 + output_block * kTileElements,
        shared_inverse +
            packed_block_index(column_block, column_block) *
                kTileElements,
        shared_l_bf16 +
            packed_block_index(row_block, middle_block) *
                kTileElements,
        shared_inverse +
            packed_block_index(middle_block, column_block) *
                kTileElements,
        nullptr, nullptr,
        shared_inverse + output_block * kTileElements, warp_scratch,
        shared_beta, matrix_transform, row_block * kTile,
        column_block * kTile);
  }
  __syncthreads();

  if (warp == 0U) {
    constexpr unsigned int row_block = 3U;
    constexpr unsigned int column_block = 0U;
    constexpr unsigned int output_block =
        packed_block_index(row_block, column_block);
    form_transform_block(
        shared_inverse +
            packed_block_index(row_block, row_block) * kTileElements,
        shared_l_bf16 + output_block * kTileElements,
        shared_inverse +
            packed_block_index(0U, 0U) * kTileElements,
        shared_l_bf16 +
            packed_block_index(row_block, 1U) * kTileElements,
        shared_inverse +
            packed_block_index(1U, 0U) * kTileElements,
        shared_l_bf16 +
            packed_block_index(row_block, 2U) * kTileElements,
        shared_inverse +
            packed_block_index(2U, 0U) * kTileElements,
        shared_inverse + output_block * kTileElements, warp_scratch,
        shared_beta, matrix_transform, row_block * kTile, 0U);
  }
}

__device__ __forceinline__ std::uint32_t encode_bf16_pair(
    const float low, const float high) {
  const auto low_bf16 = __float2bfloat16_rn(low);
  const auto high_bf16 = __float2bfloat16_rn(high);
  return static_cast<std::uint32_t>(__bfloat16_as_ushort(low_bf16)) |
         (static_cast<std::uint32_t>(
              __bfloat16_as_ushort(high_bf16))
          << 16U);
}

__device__ __forceinline__ std::uint32_t scale_bf16_pair(
    const std::uint32_t packed, const float scale) {
  const float low = __uint_as_float((packed & 0xffffU) << 16U);
  const float high = __uint_as_float(packed & 0xffff0000U);
  return encode_bf16_pair(low * scale, high * scale);
}

__device__ __forceinline__ uint4 scale_bf16_vector(
    const uint4 packed, const float scale) {
  return make_uint4(scale_bf16_pair(packed.x, scale),
                    scale_bf16_pair(packed.y, scale),
                    scale_bf16_pair(packed.z, scale),
                    scale_bf16_pair(packed.w, scale));
}

struct RecomputeM16K16 final {
  std::uint32_t x0;
  std::uint32_t x1;
  std::uint32_t x2;
  std::uint32_t x3;
};

struct RecomputeK16N8 final {
  std::uint32_t x0;
  std::uint32_t x1;
};

struct RecomputeM16N8Accumulator final {
  float x0;
  float x1;
  float x2;
  float x3;
};

__device__ __forceinline__ void recompute_zero(
    RecomputeM16N8Accumulator& accumulator) {
  accumulator.x0 = 0.0F;
  accumulator.x1 = 0.0F;
  accumulator.x2 = 0.0F;
  accumulator.x3 = 0.0F;
}

__device__ __forceinline__ void recompute_mma(
    RecomputeM16N8Accumulator& accumulator,
    const RecomputeM16K16& a, const RecomputeK16N8& b) {
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

// The solve publishes logical A[M,K] in column-major order.  Retain that
// public boundary and let the architecture-provided WMMA loader form the
// exact four-register matrix-A fragment directly from the resident tile.
__device__ __forceinline__ void recompute_load_a_col_major(
    RecomputeM16K16& packed,
    const Bf16* const shared_transform,
    const unsigned int m_panel, const unsigned int k16) {
  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16,
                 wmma::col_major>
      fragment;
  wmma::load_matrix_sync(
      fragment,
      shared_transform + m_panel * kTile +
          k16 * kTile * kChunk,
      static_cast<int>(kChunk));
  const auto* const words = reinterpret_cast<const std::uint32_t*>(
      fragment.x);
  packed.x0 = words[0];
  packed.x1 = words[1];
  packed.x2 = words[2];
  packed.x3 = words[3];
}

[[nodiscard]] __device__ __forceinline__ unsigned int
recompute_swizzled_chunk(const unsigned int row,
                         const unsigned int logical_chunk) {
  return logical_chunk ^ (row & 7U);
}

// The operand is a true row-major [K64,N64] tile.  A row-wise XOR swizzle
// removes the eight-way ldmatrix bank alias while x2.trans produces the
// register layout consumed by mma.row.col.
__device__ __forceinline__ void recompute_load_b(
    RecomputeK16N8& fragment,
    const Bf16* const shared_operand,
    const unsigned int n_panel, const unsigned int k16,
    const unsigned int lane) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
  const unsigned int row =
      k16 * kTile + (lane & 7U) + ((lane >> 3U) & 1U) * 8U;
  const unsigned int physical_chunk =
      recompute_swizzled_chunk(row, n_panel);
  const auto* const source =
      shared_operand + row * kChunk + physical_chunk * 8U;
  const unsigned int shared_address = static_cast<unsigned int>(
      __cvta_generic_to_shared(source));
  asm volatile(
      "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 "
      "{%0, %1}, [%2];"
      : "=r"(fragment.x0), "=r"(fragment.x1)
      : "r"(shared_address)
      : "memory");
#else
  (void)fragment;
  (void)shared_operand;
  (void)n_panel;
  (void)k16;
  (void)lane;
#endif
}

__device__ __forceinline__ void recompute_store_product(
    const RecomputeM16N8Accumulator
        (&accumulators)[kBlockRows][kBlockRows],
    Bf16* const output, const unsigned int column_base,
    const unsigned int lane) {
  const unsigned int row_in_half = lane >> 2U;
  const unsigned int column_pair = lane & 3U;
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kBlockRows; ++m_panel) {
    const unsigned int row0 = m_panel * kTile + row_in_half;
    const unsigned int row1 = row0 + 8U;
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kBlockRows; ++n_panel) {
      const unsigned int column =
          column_base + n_panel * 8U + column_pair * 2U;
      const RecomputeM16N8Accumulator& accumulator =
          accumulators[m_panel][n_panel];
      *reinterpret_cast<std::uint32_t*>(
          output + row0 * kDimension + column) =
          encode_bf16_pair(accumulator.x0, accumulator.x1);
      *reinterpret_cast<std::uint32_t*>(
          output + row1 * kDimension + column) =
          encode_bf16_pair(accumulator.x2, accumulator.x3);
    }
  }
}

__device__ __forceinline__ void recompute_m64n64_product(
    const Bf16* const shared_transform,
    const Bf16* const shared_operand, Bf16* const output,
    const unsigned int warp, const unsigned int lane) {
  RecomputeM16N8Accumulator accumulators[kBlockRows][kBlockRows];
#pragma unroll
  for (unsigned int m_panel = 0U; m_panel < kBlockRows; ++m_panel) {
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kBlockRows; ++n_panel) {
      recompute_zero(accumulators[m_panel][n_panel]);
    }
  }

#pragma unroll
  for (unsigned int k16 = 0U; k16 < kBlockRows; ++k16) {
    RecomputeM16K16 a[kBlockRows];
    RecomputeK16N8 b[kBlockRows];
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < kBlockRows; ++m_panel) {
      recompute_load_a_col_major(a[m_panel], shared_transform,
                                 m_panel, k16);
    }
#pragma unroll
    for (unsigned int n_panel = 0U; n_panel < kBlockRows; ++n_panel) {
      recompute_load_b(b[n_panel], shared_operand,
                       warp * kBlockRows + n_panel, k16, lane);
    }
#pragma unroll
    for (unsigned int m_panel = 0U; m_panel < kBlockRows; ++m_panel) {
#pragma unroll
      for (unsigned int n_panel = 0U; n_panel < kBlockRows; ++n_panel) {
        recompute_mma(accumulators[m_panel][n_panel], a[m_panel],
                      b[n_panel]);
      }
    }
  }
  recompute_store_product(accumulators, output, warp * 32U, lane);
}

// One M64N64 owner mirrors the selected vLLM shape: two warps retain the
// complete FP32 result in registers while transform A stays resident.  Four
// 64-column operand phases (K0/K1/V0/V1) reuse one swizzled 8 KiB bank.
constexpr unsigned int kRecomputeThreads = 64U;
constexpr unsigned int kRecomputeWarps = kRecomputeThreads / 32U;
constexpr unsigned int kVectorsPerRow =
    kChunk * sizeof(Bf16) / sizeof(uint4);
constexpr unsigned int kVectorCount = kChunk * kVectorsPerRow;
constexpr std::size_t kTransformBytes =
    kGramElements * sizeof(Bf16);
constexpr std::size_t kOperandBytes =
    kChunk * kChunk * sizeof(Bf16);
constexpr std::size_t kRecomputeSharedBytes =
    kTransformBytes + kOperandBytes + kChunk * sizeof(float);

static_assert(kRecomputeWarps == 2U);
static_assert(kVectorCount == 512U);
static_assert(kRecomputeSharedBytes == 16U * 1024U + 256U);

__global__ __launch_bounds__(kRecomputeThreads)
void value_head_recompute_chunk64_kernel(
    const std::uint16_t* const compact_k,
    const float* const cumulative_gate,
    const std::uint16_t* const conv_qkv,
    const unsigned int token_count,
    const unsigned int chunk_count,
    const std::uint16_t* const transform,
    std::uint16_t* const w, std::uint16_t* const u) {
  extern __shared__ __align__(16) unsigned char shared_raw[];
  auto* const shared_transform = reinterpret_cast<Bf16*>(shared_raw);
  Bf16* const shared_operand =
      shared_transform + kGramElements;
  auto* const shared_gate_scale = reinterpret_cast<float*>(
      shared_raw + kTransformBytes + kOperandBytes);
  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const std::size_t matrix = blockIdx.x;
  const unsigned int chunk =
      static_cast<unsigned int>(matrix / kValueHeads);
  if (chunk >= chunk_count) {
    return;
  }
  const unsigned int value_head =
      static_cast<unsigned int>(matrix % kValueHeads);
  const unsigned int qk_head = value_head / kHeadGroup;
  const std::size_t compact_matrix =
      static_cast<std::size_t>(chunk) * kQkHeads + qk_head;
  const auto* const matrix_k = reinterpret_cast<const Bf16*>(
      compact_k + compact_matrix * kVectorElements);
  const auto* const matrix_transform = reinterpret_cast<const Bf16*>(
      transform + matrix * kGramElements);
  auto* const matrix_w = reinterpret_cast<Bf16*>(
      w + matrix * kVectorElements);
  auto* const matrix_u = reinterpret_cast<Bf16*>(
      u + matrix * kVectorElements);
  const float* const matrix_gamma =
      cumulative_gate + matrix * kChunk;

  constexpr unsigned int transform_vectors =
      kTransformBytes / sizeof(uint4);
  for (unsigned int vector = thread; vector < transform_vectors;
       vector += kRecomputeThreads) {
    copy_16_async(reinterpret_cast<uint4*>(shared_transform) + vector,
                  reinterpret_cast<const uint4*>(matrix_transform) +
                      vector);
  }
  commit_and_wait_async_copies();
  if (thread < kChunk) {
    shared_gate_scale[thread] = expf(matrix_gamma[thread]);
  }
  __syncthreads();

#pragma unroll
  for (unsigned int panel = 0U; panel < kDimension / kChunk; ++panel) {
    for (unsigned int vector = thread; vector < kVectorCount;
         vector += kRecomputeThreads) {
      const unsigned int row = vector / kVectorsPerRow;
      const unsigned int logical_chunk = vector % kVectorsPerRow;
      const unsigned int physical_chunk =
          recompute_swizzled_chunk(row, logical_chunk);
      const unsigned int global_vector =
          row * (kDimension * sizeof(Bf16) / sizeof(uint4)) +
          panel * kVectorsPerRow + logical_chunk;
      const uint4 packed =
          reinterpret_cast<const uint4*>(matrix_k)[global_vector];
      reinterpret_cast<uint4*>(shared_operand)
          [row * kVectorsPerRow + physical_chunk] =
          scale_bf16_vector(packed, shared_gate_scale[row]);
    }
    __syncthreads();
    recompute_m64n64_product(shared_transform, shared_operand,
                             matrix_w + panel * kChunk, warp, lane);
    __syncthreads();
  }

#pragma unroll
  for (unsigned int panel = 0U; panel < kDimension / kChunk; ++panel) {
    for (unsigned int vector = thread; vector < kVectorCount;
         vector += kRecomputeThreads) {
      const unsigned int row = vector / kVectorsPerRow;
      const unsigned int logical_chunk = vector % kVectorsPerRow;
      const unsigned int physical_chunk =
          recompute_swizzled_chunk(row, logical_chunk);
      const std::size_t token =
          static_cast<std::size_t>(chunk) * kChunk + row;
      const std::size_t source =
          token * kGdnQkvChannels + kGdnQElements + kGdnKElements +
          static_cast<std::size_t>(value_head) * kDimension +
          panel * kChunk + logical_chunk *
              (sizeof(uint4) / sizeof(Bf16));
      auto* const destination =
          reinterpret_cast<uint4*>(shared_operand) +
          row * kVectorsPerRow + physical_chunk;
      if (token < token_count) {
        copy_16_async(destination,
                      reinterpret_cast<const uint4*>(conv_qkv + source));
      } else {
        *destination = make_uint4(0U, 0U, 0U, 0U);
      }
    }
    commit_and_wait_async_copies();
    __syncthreads();
    recompute_m64n64_product(shared_transform, shared_operand,
                             matrix_u + panel * kChunk, warp, lane);
    __syncthreads();
  }
}

[[nodiscard]] bool invalid_arguments(
    const std::uint16_t* const compact_k,
    const float* const cumulative_gate, const float* const beta,
    const std::uint16_t* const conv_qkv,
    const std::size_t token_count, const std::size_t chunk_count,
    const float* const raw_gram,
    const std::uint16_t* const transform,
    const std::uint16_t* const w, const std::uint16_t* const u) noexcept {
  return compact_k == nullptr || cumulative_gate == nullptr ||
         beta == nullptr || conv_qkv == nullptr || raw_gram == nullptr ||
         transform == nullptr || w == nullptr || u == nullptr ||
         token_count == 0U || token_count > kMaximumChunks * kChunk ||
         chunk_count == 0U || chunk_count > kMaximumChunks ||
         token_count > chunk_count * kChunk ||
         token_count <= (chunk_count - 1U) * kChunk;
}

[[nodiscard]] bool invalid_raw_gram_arguments(
    const std::uint16_t* const compact_k,
    const std::size_t token_count, const std::size_t chunk_count,
    const float* const raw_gram) noexcept {
  return compact_k == nullptr || raw_gram == nullptr ||
         token_count == 0U ||
         token_count > kMaximumRawGramChunks * kChunk ||
         chunk_count == 0U || chunk_count > kMaximumRawGramChunks ||
         chunk_count != (token_count + kChunk - 1U) / kChunk;
}

}  // namespace

int launch_raw_gram(const std::uint16_t* const compact_k,
                    const std::size_t token_count,
                    const std::size_t chunk_count,
                    float* const raw_gram,
                    void* const cuda_stream) noexcept {
  if (invalid_raw_gram_arguments(compact_k, token_count, chunk_count,
                                 raw_gram)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  compact_lower_gram_chunk64_kernel<<<
      static_cast<unsigned int>(chunk_count * kQkHeads), kGramThreads,
      kGramSharedBytes, stream>>>(
      compact_k, static_cast<unsigned int>(chunk_count), raw_gram);
  return static_cast<int>(cudaGetLastError());
}

int launch_packless(const std::uint16_t* const compact_k,
                    const float* const cumulative_gate,
                    const float* const beta,
                    const std::uint16_t* const conv_qkv,
                    const std::size_t token_count,
                    const std::size_t chunk_count,
                    float* const raw_gram_scratch,
                    std::uint16_t* const transform,
                    std::uint16_t* const w,
                    std::uint16_t* const u,
                    void* const cuda_stream) noexcept {
  if (invalid_arguments(compact_k, cumulative_gate, beta, conv_qkv,
                        token_count, chunk_count, raw_gram_scratch,
                        transform, w, u)) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  compact_lower_gram_chunk64_kernel<<<
      static_cast<unsigned int>(chunk_count * kQkHeads), kGramThreads,
      kGramSharedBytes, stream>>>(
      compact_k, static_cast<unsigned int>(chunk_count),
      raw_gram_scratch);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  value_head_solve_chunk64_kernel<<<
      static_cast<unsigned int>(chunk_count * kValueHeads),
      kSolveThreads, kSolveSharedBytes, stream>>>(
      raw_gram_scratch, cumulative_gate, beta,
      static_cast<unsigned int>(chunk_count), transform);
  status = cudaGetLastError();
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }

  value_head_recompute_chunk64_kernel<<<
      static_cast<unsigned int>(chunk_count * kValueHeads),
      kRecomputeThreads, kRecomputeSharedBytes, stream>>>(
      compact_k, cumulative_gate, conv_qkv,
      static_cast<unsigned int>(token_count),
      static_cast<unsigned int>(chunk_count), transform, w, u);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace q3x::runtime::gdn_prefill_wy_vllm_layout_detail
