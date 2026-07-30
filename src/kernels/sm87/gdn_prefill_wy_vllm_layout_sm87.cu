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

__device__ __forceinline__ uint4 encode_bf16_vector(
    const float* const values) {
  return make_uint4(encode_bf16_pair(values[0], values[1]),
                    encode_bf16_pair(values[2], values[3]),
                    encode_bf16_pair(values[4], values[5]),
                    encode_bf16_pair(values[6], values[7]));
}

__device__ __forceinline__ void recompute_product(
    const Bf16* const transform, const Bf16* const operand,
    Bf16* const output, float* const scratch,
    const unsigned int warp, const unsigned int lane) {
  float* const warp_scratch = scratch + warp * kTileElements;
#pragma unroll
  for (unsigned int token_block = 0U; token_block < kBlockRows;
       ++token_block) {
    Accumulator accumulator;
    wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
    for (unsigned int source_block = 0U; source_block < kBlockRows;
         ++source_block) {
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16,
                     wmma::col_major>
          transform_fragment;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16,
                     wmma::row_major>
          operand_fragment;
      wmma::load_matrix_sync(
          transform_fragment,
          transform + token_block * kTile +
              source_block * kTile * kChunk,
          static_cast<int>(kChunk));
      wmma::load_matrix_sync(
          operand_fragment,
          operand + source_block * kTile * kDimension + warp * kTile,
          static_cast<int>(kDimension));
      wmma::mma_sync(accumulator, transform_fragment,
                     operand_fragment, accumulator);
    }
    wmma::store_matrix_sync(warp_scratch, accumulator,
                            static_cast<int>(kTile),
                            wmma::mem_row_major);
    __syncwarp();
    const unsigned int row = lane / 2U;
    const unsigned int half = lane % 2U;
    const unsigned int row_offset = row * kTile + half * 8U;
    const uint4 packed =
        encode_bf16_vector(warp_scratch + row_offset);
    auto* const destination = reinterpret_cast<uint4*>(
        output + (token_block * kTile + row) * kDimension +
        warp * kTile + half * 8U);
    *destination = packed;
    __syncwarp();
  }
}

// Transform stays resident while the two K/V operand lifetimes reuse one
// 16 KiB bank.  Eight warps keep every 16-column slab active; all global
// staging and output publication use 128-bit vectors.
constexpr unsigned int kRecomputeThreads = 256U;
constexpr unsigned int kRecomputeWarps = kRecomputeThreads / 32U;
constexpr unsigned int kVectorsPerRow =
    kDimension * sizeof(Bf16) / sizeof(uint4);
constexpr unsigned int kVectorCount = kChunk * kVectorsPerRow;
constexpr std::size_t kTransformBytes =
    kGramElements * sizeof(Bf16);
constexpr std::size_t kOperandBytes =
    kVectorElements * sizeof(Bf16);
constexpr std::size_t kRecomputeScratchBytes =
    kRecomputeWarps * kTileElements * sizeof(float);
constexpr std::size_t kRecomputeSharedBytes =
    kTransformBytes + kOperandBytes + kRecomputeScratchBytes +
    kChunk * sizeof(float);

static_assert(kRecomputeWarps == kDimension / kTile);
static_assert(kVectorCount == 1024U);
static_assert(kRecomputeSharedBytes == 32U * 1024U + 256U);

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
  auto* const shared_scratch = reinterpret_cast<float*>(
      shared_raw + kTransformBytes + kOperandBytes);
  float* const shared_gate_scale =
      shared_scratch + kRecomputeWarps * kTileElements;
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

  for (unsigned int vector = thread; vector < kVectorCount;
       vector += kRecomputeThreads) {
    const unsigned int row = vector / kVectorsPerRow;
    const uint4 packed =
        reinterpret_cast<const uint4*>(matrix_k)[vector];
    reinterpret_cast<uint4*>(shared_operand)[vector] =
        scale_bf16_vector(packed, shared_gate_scale[row]);
  }
  __syncthreads();
  recompute_product(shared_transform, shared_operand, matrix_w,
                    shared_scratch, warp, lane);
  __syncthreads();

  for (unsigned int vector = thread; vector < kVectorCount;
       vector += kRecomputeThreads) {
    const unsigned int row = vector / kVectorsPerRow;
    const unsigned int vector_in_row = vector % kVectorsPerRow;
    const std::size_t token =
        static_cast<std::size_t>(chunk) * kChunk + row;
    const std::size_t source =
        token * kGdnQkvChannels + kGdnQElements + kGdnKElements +
        static_cast<std::size_t>(value_head) * kDimension +
        vector_in_row * (sizeof(uint4) / sizeof(Bf16));
    if (token < token_count) {
      copy_16_async(reinterpret_cast<uint4*>(shared_operand) + vector,
                    reinterpret_cast<const uint4*>(conv_qkv + source));
    } else {
      reinterpret_cast<uint4*>(shared_operand)[vector] =
          make_uint4(0U, 0U, 0U, 0U);
    }
  }
  commit_and_wait_async_copies();
  __syncthreads();
  recompute_product(shared_transform, shared_operand, matrix_u,
                    shared_scratch, warp, lane);
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

}  // namespace

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
