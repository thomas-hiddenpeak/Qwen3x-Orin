#include "gdn_prefill_group_wy_sm87.h"

#include "q3x/runtime/gdn_decode.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace q3x::runtime::gdn_prefill_group_wy_detail {
namespace {

constexpr unsigned int kChunk = 64U;
constexpr unsigned int kDimension = 128U;
constexpr unsigned int kQkHeads = 16U;
constexpr unsigned int kValueHeads = 48U;
constexpr unsigned int kHeadGroup = 3U;
constexpr unsigned int kTile = 16U;
constexpr unsigned int kThreads = 256U;
constexpr unsigned int kWarps = kThreads / 32U;
constexpr unsigned int kMaximumChunks = 8U;

constexpr std::size_t kKElements = kChunk * kDimension;
constexpr std::size_t kGramElements = kChunk * kChunk;
constexpr std::size_t kVectorBytes = kKElements * sizeof(std::uint16_t);
constexpr std::size_t kMatrixBf16Bytes =
    kGramElements * sizeof(std::uint16_t);
constexpr std::size_t kMatrixFp32Bytes = kGramElements * sizeof(float);

// Lifetime map:
//   [K compact 16 KiB] [raw Gram FP32 16 KiB]
//   [32 KiB solve workspace -> gated-K/value staging]
//   [transform BF16 8 KiB] [eight warp scratch tiles 8 KiB]
// K and raw Gram remain resident while the three value heads are processed.
constexpr std::size_t kKOffset = 0U;
constexpr std::size_t kGramOffset = kKOffset + kVectorBytes;
constexpr std::size_t kWorkOffset = kGramOffset + kMatrixFp32Bytes;
constexpr std::size_t kWorkBytes =
    kMatrixFp32Bytes + 2U * kMatrixBf16Bytes;
constexpr std::size_t kTransformOffset = kWorkOffset + kWorkBytes;
constexpr std::size_t kScratchOffset =
    kTransformOffset + kMatrixBf16Bytes;
constexpr std::size_t kScratchBytes =
    kWarps * kTile * kTile * sizeof(float);
constexpr std::size_t kPackedSharedBytes =
    kScratchOffset + kScratchBytes;
constexpr std::size_t kGateScaleOffset = kPackedSharedBytes;
constexpr std::size_t kGateScaleBytes = kChunk * sizeof(float);
constexpr std::size_t kPacklessSharedBytes =
    kGateScaleOffset + kGateScaleBytes;

static_assert(kQkHeads * kHeadGroup == kValueHeads);
static_assert(kWarps == 8U);
static_assert(kPackedSharedBytes == 80U * 1024U);
static_assert(kPacklessSharedBytes == 80U * 1024U + 256U);

namespace wmma = nvcuda::wmma;
using Bf16 = __nv_bfloat16;
using Accumulator =
    wmma::fragment<wmma::accumulator, kTile, kTile, kTile, float>;

__device__ __forceinline__ void accumulate_row_major_product(
    Accumulator& accumulator, const Bf16* const a, const int leading_a,
    const Bf16* const b, const int leading_b) {
  wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16,
                 wmma::row_major>
      a_fragment;
  wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16,
                 wmma::row_major>
      b_fragment;
  wmma::load_matrix_sync(a_fragment, a, leading_a);
  wmma::load_matrix_sync(b_fragment, b, leading_b);
  wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
}

template <unsigned int Row, unsigned int Inner>
__device__ __forceinline__ void accumulate_diagonal_inverse_column(
    const float* const diagonal_l, const unsigned int lane,
    const float (&inverse_column)[kTile], float& value) {
  if constexpr (Inner < Row) {
    if (lane <= Inner) {
      value -= diagonal_l[Row * kChunk + Inner] * inverse_column[Inner];
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
    const unsigned int output_base, float* const scratch,
    const float* const beta, Bf16* const transform,
    const unsigned int row_base, const unsigned int column_base) {
  const unsigned int lane = threadIdx.x % 32U;
  Accumulator inner;
  wmma::fill_fragment(inner, 0.0F);
  accumulate_row_major_product(inner, left0, static_cast<int>(kChunk),
                               right0, static_cast<int>(kChunk));
  if (left1 != nullptr && right1 != nullptr) {
    accumulate_row_major_product(inner, left1, static_cast<int>(kChunk),
                                 right1, static_cast<int>(kChunk));
  }
  if (left2 != nullptr && right2 != nullptr) {
    accumulate_row_major_product(inner, left2, static_cast<int>(kChunk),
                                 right2, static_cast<int>(kChunk));
  }
  wmma::store_matrix_sync(scratch, inner, static_cast<int>(kTile),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane; index < kTile * kTile; index += 32U) {
    const unsigned int row = index / kTile;
    const unsigned int column = index % kTile;
    inverse[output_base + row * kChunk + column] =
        __float2bfloat16_rn(scratch[index]);
  }
  __syncwarp();

  Accumulator outer;
  wmma::fill_fragment(outer, 0.0F);
  accumulate_row_major_product(
      outer, diagonal_inverse, static_cast<int>(kChunk),
      inverse + output_base, static_cast<int>(kChunk));
  wmma::store_matrix_sync(scratch, outer, static_cast<int>(kTile),
                          wmma::mem_row_major);
  __syncwarp();
  for (unsigned int index = lane; index < kTile * kTile; index += 32U) {
    const unsigned int row = index / kTile;
    const unsigned int column = index % kTile;
    const float value = -scratch[index];
    inverse[output_base + row * kChunk + column] =
        __float2bfloat16_rn(value);
    const unsigned int output_row = row_base + row;
    const unsigned int output_column = column_base + column;
    transform[output_row + output_column * kChunk] =
        __float2bfloat16_rn(value * beta[output_column]);
  }
  __syncwarp();
}

__device__ __forceinline__ void solve_transform(
    const float* const raw_gram, const float* const cumulative_gate,
    const float* const beta, float* const l, Bf16* const l_bf16,
    Bf16* const inverse_bf16, Bf16* const transform,
    float* const scratch, const unsigned int thread,
    const unsigned int warp, const unsigned int lane) {
  for (unsigned int index = thread; index < kGramElements;
       index += kThreads) {
    const unsigned int row = index / kChunk;
    const unsigned int column = index % kChunk;
    float value = 0.0F;
    if (row > column) {
      value = beta[row] *
              expf(cumulative_gate[row] - cumulative_gate[column]) *
              raw_gram[index];
    }
    l[index] = value;
    if (row < column) {
      transform[row + column * kChunk] = __float2bfloat16_rn(0.0F);
    }
  }
  __syncthreads();

  float inverse_column[kTile]{};
  if (warp < 4U && lane < kTile) {
    const unsigned int diagonal_base =
        warp * kTile * kChunk + warp * kTile;
    solve_diagonal_inverse_column<0U>(l + diagonal_base, lane,
                                      inverse_column);
#pragma unroll
    for (unsigned int row = 0U; row < kTile; ++row) {
      const unsigned int output_row = warp * kTile + row;
      const unsigned int output_column = warp * kTile + lane;
      const float value = inverse_column[row];
      inverse_bf16[diagonal_base + row * kChunk + lane] =
          __float2bfloat16_rn(value);
      transform[output_row + output_column * kChunk] =
          __float2bfloat16_rn(value * beta[output_column]);
    }
  }
  __syncthreads();

  for (unsigned int index = thread; index < kGramElements;
       index += kThreads) {
    l_bf16[index] = __float2bfloat16_rn(l[index]);
  }
  __syncthreads();

  float* const warp_scratch = scratch + warp * kTile * kTile;
  if (warp < 3U) {
    const unsigned int row_block = warp + 1U;
    const unsigned int column_block = warp;
    const unsigned int output_base =
        row_block * kTile * kChunk + column_block * kTile;
    form_transform_block(
        inverse_bf16 + row_block * kTile * kChunk + row_block * kTile,
        l_bf16 + output_base,
        inverse_bf16 + column_block * kTile * kChunk +
            column_block * kTile,
        nullptr, nullptr, nullptr, nullptr, inverse_bf16, output_base,
        warp_scratch, beta, transform, row_block * kTile,
        column_block * kTile);
  }
  __syncthreads();

  if (warp < 2U) {
    const unsigned int row_block = warp + 2U;
    const unsigned int column_block = warp;
    const unsigned int middle_block = warp + 1U;
    const unsigned int output_base =
        row_block * kTile * kChunk + column_block * kTile;
    form_transform_block(
        inverse_bf16 + row_block * kTile * kChunk + row_block * kTile,
        l_bf16 + output_base,
        inverse_bf16 + column_block * kTile * kChunk +
            column_block * kTile,
        l_bf16 + row_block * kTile * kChunk + middle_block * kTile,
        inverse_bf16 + middle_block * kTile * kChunk +
            column_block * kTile,
        nullptr, nullptr, inverse_bf16, output_base, warp_scratch, beta,
        transform, row_block * kTile, column_block * kTile);
  }
  __syncthreads();

  if (warp == 0U) {
    constexpr unsigned int row_block = 3U;
    constexpr unsigned int output_base = row_block * kTile * kChunk;
    form_transform_block(
        inverse_bf16 + row_block * kTile * kChunk + row_block * kTile,
        l_bf16 + output_base, inverse_bf16,
        l_bf16 + output_base + kTile,
        inverse_bf16 + kTile * kChunk,
        l_bf16 + output_base + 2U * kTile,
        inverse_bf16 + 2U * kTile * kChunk, inverse_bf16, output_base,
        warp_scratch, beta, transform, row_block * kTile, 0U);
  }
  __syncthreads();
}

__device__ __forceinline__ void recompute_w_u(
    const Bf16* const transform, const Bf16* const gated_k,
    const Bf16* const value, Bf16* const w, Bf16* const u,
    float* const scratch, const unsigned int warp,
    const unsigned int lane) {
  float* const tile_scratch = scratch + warp * kTile * kTile;
#pragma unroll
  for (unsigned int token_block = 0U; token_block < kChunk / kTile;
       ++token_block) {
    Accumulator w_accumulator;
    Accumulator u_accumulator;
    wmma::fill_fragment(w_accumulator, 0.0F);
    wmma::fill_fragment(u_accumulator, 0.0F);
#pragma unroll
    for (unsigned int source_block = 0U; source_block < kChunk / kTile;
         ++source_block) {
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16,
                     wmma::col_major>
          transform_fragment;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16,
                     wmma::row_major>
          k_fragment;
      wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16,
                     wmma::row_major>
          v_fragment;
      wmma::load_matrix_sync(
          transform_fragment,
          transform + token_block * kTile +
              source_block * kTile * kChunk,
          static_cast<int>(kChunk));
      wmma::load_matrix_sync(
          k_fragment,
          gated_k + source_block * kTile * kDimension + warp * kTile,
          static_cast<int>(kDimension));
      wmma::load_matrix_sync(
          v_fragment,
          value + source_block * kTile * kDimension + warp * kTile,
          static_cast<int>(kDimension));
      wmma::mma_sync(w_accumulator, transform_fragment, k_fragment,
                     w_accumulator);
      wmma::mma_sync(u_accumulator, transform_fragment, v_fragment,
                     u_accumulator);
    }

    wmma::store_matrix_sync(tile_scratch, w_accumulator,
                            static_cast<int>(kTile),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < kTile * kTile; index += 32U) {
      const unsigned int row = index / kTile;
      const unsigned int column = index % kTile;
      w[(token_block * kTile + row) * kDimension + warp * kTile +
        column] = __float2bfloat16_rn(tile_scratch[index]);
    }
    __syncwarp();
    wmma::store_matrix_sync(tile_scratch, u_accumulator,
                            static_cast<int>(kTile),
                            wmma::mem_row_major);
    __syncwarp();
    for (unsigned int index = lane; index < kTile * kTile; index += 32U) {
      const unsigned int row = index / kTile;
      const unsigned int column = index % kTile;
      u[(token_block * kTile + row) * kDimension + warp * kTile +
        column] = __float2bfloat16_rn(tile_scratch[index]);
    }
    __syncwarp();
  }
}

template <bool Packless>
__global__ __launch_bounds__(kThreads)
void gqa_group_wy_chunk64_kernel(
    const std::uint16_t* const k,
    const float* const cumulative_gate,
    const float* const beta,
    const std::uint16_t* const gated_k,
    const std::uint16_t* const value,
    const std::uint16_t* const conv_qkv,
    const unsigned int chunk_count,
    std::uint16_t* const transform,
    std::uint16_t* const w,
    std::uint16_t* const u) {
  extern __shared__ unsigned char shared[];
  auto* const shared_k = reinterpret_cast<Bf16*>(shared + kKOffset);
  auto* const shared_raw_gram =
      reinterpret_cast<float*>(shared + kGramOffset);
  auto* const shared_l = reinterpret_cast<float*>(shared + kWorkOffset);
  auto* const shared_l_bf16 = reinterpret_cast<Bf16*>(
      shared + kWorkOffset + kMatrixFp32Bytes);
  auto* const shared_inverse_bf16 = reinterpret_cast<Bf16*>(
      shared + kWorkOffset + kMatrixFp32Bytes + kMatrixBf16Bytes);
  auto* const shared_gated_k = reinterpret_cast<Bf16*>(shared + kWorkOffset);
  auto* const shared_value =
      reinterpret_cast<Bf16*>(shared + kWorkOffset + kVectorBytes);
  auto* const shared_transform =
      reinterpret_cast<Bf16*>(shared + kTransformOffset);
  auto* const shared_scratch =
      reinterpret_cast<float*>(shared + kScratchOffset);
  auto* const shared_gate_scale =
      reinterpret_cast<float*>(shared + kGateScaleOffset);

  const unsigned int thread = threadIdx.x;
  const unsigned int warp = thread / 32U;
  const unsigned int lane = thread % 32U;
  const unsigned int group = blockIdx.x;
  const unsigned int chunk_index = group / kQkHeads;
  const unsigned int qk_head = group % kQkHeads;
  if (chunk_index >= chunk_count) {
    return;
  }
  const unsigned int first_value_head = qk_head * kHeadGroup;
  const std::size_t first_matrix =
      static_cast<std::size_t>(chunk_index) * kValueHeads +
      first_value_head;
  const std::size_t compact_matrix =
      static_cast<std::size_t>(chunk_index) * kQkHeads + qk_head;
  const auto* const matrix_k = reinterpret_cast<const Bf16*>(
      k + (Packless ? compact_matrix : first_matrix) * kKElements);

  for (unsigned int index = thread; index < kKElements;
       index += kThreads) {
    shared_k[index] = matrix_k[index];
  }
  __syncthreads();

  if (warp < 4U) {
    Accumulator gram[4];
#pragma unroll
    for (unsigned int column_block = 0U; column_block < 4U;
         ++column_block) {
      wmma::fill_fragment(gram[column_block], 0.0F);
    }
#pragma unroll
    for (unsigned int key_block = 0U; key_block < kDimension / kTile;
         ++key_block) {
      wmma::fragment<wmma::matrix_a, kTile, kTile, kTile, Bf16,
                     wmma::row_major>
          row_fragment;
      wmma::load_matrix_sync(
          row_fragment,
          shared_k + warp * kTile * kDimension + key_block * kTile,
          static_cast<int>(kDimension));
#pragma unroll
      for (unsigned int column_block = 0U; column_block < 4U;
           ++column_block) {
        wmma::fragment<wmma::matrix_b, kTile, kTile, kTile, Bf16,
                       wmma::col_major>
            column_fragment;
        wmma::load_matrix_sync(
            column_fragment,
            shared_k + column_block * kTile * kDimension +
                key_block * kTile,
            static_cast<int>(kDimension));
        wmma::mma_sync(gram[column_block], row_fragment, column_fragment,
                       gram[column_block]);
      }
    }
#pragma unroll
    for (unsigned int column_block = 0U; column_block < 4U;
         ++column_block) {
      wmma::store_matrix_sync(
          shared_raw_gram + warp * kTile * kChunk + column_block * kTile,
          gram[column_block], static_cast<int>(kChunk),
          wmma::mem_row_major);
    }
  }
  __syncthreads();

#pragma unroll
  for (unsigned int replica = 0U; replica < kHeadGroup; ++replica) {
    const unsigned int value_head = first_value_head + replica;
    const std::size_t matrix =
        static_cast<std::size_t>(chunk_index) * kValueHeads + value_head;
    const float* const matrix_gate = cumulative_gate + matrix * kChunk;
    const float* const matrix_beta = beta + matrix * kChunk;

    solve_transform(shared_raw_gram, matrix_gate, matrix_beta, shared_l,
                    shared_l_bf16, shared_inverse_bf16,
                    shared_transform, shared_scratch, thread, warp, lane);

    auto* const matrix_transform = reinterpret_cast<Bf16*>(
        transform + matrix * kGramElements);
    auto* const matrix_w =
        reinterpret_cast<Bf16*>(w + matrix * kKElements);
    auto* const matrix_u =
        reinterpret_cast<Bf16*>(u + matrix * kKElements);

    for (unsigned int index = thread; index < kGramElements;
         index += kThreads) {
      matrix_transform[index] = shared_transform[index];
    }
    if constexpr (Packless) {
      if (thread < kChunk) {
        shared_gate_scale[thread] = expf(matrix_gate[thread]);
      }
      __syncthreads();
      for (unsigned int index = thread; index < kKElements;
           index += kThreads) {
        const unsigned int token_in_chunk = index / kDimension;
        const unsigned int dimension = index % kDimension;
        const unsigned int token = chunk_index * kChunk + token_in_chunk;
        shared_gated_k[index] = __float2bfloat16_rn(
            shared_gate_scale[token_in_chunk] *
            __bfloat162float(shared_k[index]));
        const std::size_t source =
            static_cast<std::size_t>(token) * kGdnQkvChannels +
            kGdnQElements + kGdnKElements +
            static_cast<std::size_t>(value_head) * kDimension + dimension;
        shared_value[index] =
            reinterpret_cast<const Bf16*>(conv_qkv)[source];
      }
    } else {
      const auto* const matrix_gated_k = reinterpret_cast<const Bf16*>(
          gated_k + matrix * kKElements);
      const auto* const matrix_value = reinterpret_cast<const Bf16*>(
          value + matrix * kKElements);
      for (unsigned int index = thread; index < kKElements;
           index += kThreads) {
        shared_gated_k[index] = matrix_gated_k[index];
        shared_value[index] = matrix_value[index];
      }
    }
    __syncthreads();

    recompute_w_u(shared_transform, shared_gated_k, shared_value, matrix_w,
                  matrix_u, shared_scratch, warp, lane);
    __syncthreads();
  }
}

[[nodiscard]] bool invalid_common_arguments(
    const std::uint16_t* const k, const float* const cumulative_gate,
    const float* const beta, const std::size_t chunk_count,
    const std::uint16_t* const transform, const std::uint16_t* const w,
    const std::uint16_t* const u) noexcept {
  return k == nullptr || cumulative_gate == nullptr || beta == nullptr ||
         transform == nullptr || w == nullptr || u == nullptr ||
         chunk_count == 0U ||
         chunk_count > kMaximumChunks;
}

template <bool Packless>
[[nodiscard]] int launch_impl(
    const std::uint16_t* const k, const float* const cumulative_gate,
    const float* const beta, const std::uint16_t* const gated_k,
    const std::uint16_t* const value,
    const std::uint16_t* const conv_qkv, const std::size_t chunk_count,
    std::uint16_t* const transform, std::uint16_t* const w,
    std::uint16_t* const u, void* const cuda_stream) noexcept {
  if (invalid_common_arguments(k, cumulative_gate, beta, chunk_count,
                               transform, w, u) ||
      (Packless ? conv_qkv == nullptr
                : (gated_k == nullptr || value == nullptr))) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  const auto stream = reinterpret_cast<cudaStream_t>(cuda_stream);
  constexpr std::size_t shared_bytes =
      Packless ? kPacklessSharedBytes : kPackedSharedBytes;
  gqa_group_wy_chunk64_kernel<Packless><<<
      static_cast<unsigned int>(chunk_count * kQkHeads), kThreads,
      shared_bytes, stream>>>(
      k, cumulative_gate, beta, gated_k, value, conv_qkv,
      static_cast<unsigned int>(chunk_count), transform, w, u);
  return static_cast<int>(cudaGetLastError());
}

}  // namespace

int configure() noexcept {
  cudaError_t status = cudaFuncSetAttribute(
      gqa_group_wy_chunk64_kernel<false>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kPackedSharedBytes));
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = cudaFuncSetAttribute(
      gqa_group_wy_chunk64_kernel<true>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(kPacklessSharedBytes));
  return static_cast<int>(status);
}

int launch(const std::uint16_t* const k,
           const float* const cumulative_gate,
           const float* const beta,
           const std::uint16_t* const gated_k,
           const std::uint16_t* const value,
           const std::size_t chunk_count,
           std::uint16_t* const transform,
           std::uint16_t* const w,
           std::uint16_t* const u,
           void* const cuda_stream) noexcept {
  // The native runner does not own a library context. Configure the opt-in
  // dynamic-shared limit lazily on the first real launch and retain the CUDA
  // status for every later layer invocation.
  static const int configuration_status = configure();
  if (configuration_status != static_cast<int>(cudaSuccess)) {
    return configuration_status;
  }
  return launch_impl<false>(k, cumulative_gate, beta, gated_k, value,
                            nullptr, chunk_count, transform, w, u,
                            cuda_stream);
}

int launch_packless(const std::uint16_t* const compact_k,
                    const float* const cumulative_gate,
                    const float* const beta,
                    const std::uint16_t* const conv_qkv,
                    const std::size_t chunk_count,
                    std::uint16_t* const transform,
                    std::uint16_t* const w,
                    std::uint16_t* const u,
                    void* const cuda_stream) noexcept {
  static const int configuration_status = configure();
  if (configuration_status != static_cast<int>(cudaSuccess)) {
    return configuration_status;
  }
  return launch_impl<true>(compact_k, cumulative_gate, beta, nullptr,
                           nullptr, conv_qkv, chunk_count, transform, w, u,
                           cuda_stream);
}

int query_resources(int* const registers_per_thread,
                    std::size_t* const static_shared_bytes,
                    std::size_t* const local_bytes,
                    int* const maximum_threads_per_block,
                    int* const active_blocks_per_sm) noexcept {
  if (registers_per_thread == nullptr || static_shared_bytes == nullptr ||
      local_bytes == nullptr || maximum_threads_per_block == nullptr ||
      active_blocks_per_sm == nullptr) {
    return static_cast<int>(cudaErrorInvalidValue);
  }
  cudaFuncAttributes attributes{};
  cudaError_t status =
      cudaFuncGetAttributes(&attributes,
                            gqa_group_wy_chunk64_kernel<true>);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  status = static_cast<cudaError_t>(configure());
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  int active = 0;
  status = cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &active, gqa_group_wy_chunk64_kernel<true>,
      static_cast<int>(kThreads), kPacklessSharedBytes);
  if (status != cudaSuccess) {
    return static_cast<int>(status);
  }
  *registers_per_thread = attributes.numRegs;
  *static_shared_bytes = attributes.sharedSizeBytes;
  *local_bytes = attributes.localSizeBytes;
  *maximum_threads_per_block = attributes.maxThreadsPerBlock;
  *active_blocks_per_sm = active;
  return static_cast<int>(cudaSuccess);
}

}  // namespace q3x::runtime::gdn_prefill_group_wy_detail
